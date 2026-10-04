//! `radius` command line front-end.

use std::process::ExitCode;
use std::time::Instant;

use anyhow::{bail, Result};
use clap::Parser;

use radius_rs::io::{self as audio, Audio};
use radius_rs::cli::{
    self, analyse_clipping, clip_warning, effective_container, formant_shift_warning,
    progress_pair, warn, Cli, Mode, Progress,
};
use radius_rs::fft;
use radius_rs::util::corr;
use radius_rs::{TdState, VocoderState};

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("radius: {e:#}");
            ExitCode::FAILURE
        }
    }
}

/// A ratio of exactly 1 means there is nothing to shift or stretch, so neither
/// engine is run.
///
/// This matters for more than speed. Both engines *are* resampling operators even at
/// ratio 1 — the time-domain engine reassembles the signal out of overlapping
/// granules and trims the tail, and the vocoder goes through the whole
/// analysis/synthesis chain. Measured against the input on the acceptance corpus:
///
/// | path | RMS error | correlation | frames lost |
/// |---|---|---|---|
/// | `-m td -s 0` | −12.1 dB | 0.970 | 536 |
/// | `-m vc -s 0` | +4.2 dB | 0.301 | 0 |
///
/// The vocoder figure is not a bug in this port: the reference (`pyradius`) produces
/// the same output for the same input, so it is inherited. What it means is that
/// neither engine can be used as a converter, which is exactly what a bare
/// invocation (`-s 0`) is documented to do — so that case is a straight copy, and it
/// is bit-exact.
fn is_passthrough(cli: &Cli) -> bool {
    cli.semitones == 0.0 && cli.tempo == 100.0
}

/// Give the engine's output the duration `--tempo` asked for, at the pitch it produced.
///
/// The engines apply the pitch with a resampler whose rate is also what sets their
/// output length, and their granule scheduler traverses the input once, so the audio
/// they emit is capped near the input length:
///
/// ```text
///   4 s source, -s 3, -t 200  ->  8.12 s out, 4.07 s of it audio
///   8 s source, -s 3, -t 200  -> 16.06 s out, 8.07 s of it audio
/// ```
///
/// So the duration was never the hard part — the content was. This stage takes the
/// engine's natural-length, correctly pitched signal and re-times it with overlap-add,
/// which changes the duration without touching the pitch (see [`radius_rs::stretch`]).
///
/// This is why the engines are asked for `input.frames()` rather than for the stretched
/// total: giving them the stretched total only makes them pad or cut, and stretching
/// that afterwards would re-time the padding. The engine's job is pitch; this stage's
/// job is duration.
///
/// Skipped when the factor is 1, so the default path costs nothing and stays
/// bit-identical.
fn stretch_to_target(
    cli: &Cli,
    input: &Audio,
    out: Vec<f32>,
    nch: usize,
    sr: u32,
) -> (Vec<f32>, u32, usize) {
    let factor = cli.tempo / 100.0;
    let target = target_frames(cli, input);
    let have = if nch == 0 { 0 } else { out.len() / nch };
    if nch == 0 || have == 0 || (factor - 1.0).abs() < 1e-12 {
        return (out, sr, nch);
    }
    let t0 = Instant::now();
    let (y, stats) = radius_rs::stretch::stretch(&out, nch, target, factor);
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "tempo: {} -> {} frames (x{:.4})  overlap-add, {} windows, mean |offset| {:.1} samples{}  [{:.0} ms]",
        have,
        target,
        factor,
        stats.frames_out,
        stats.mean_offset,
        if stats.clamped > 0 {
            format!(
                ", {} at the search limit, level {:+.2} dB",
                stats.clamped, stats.level_db
            )
        } else {
            format!(", level {:+.2} dB", stats.level_db)
        },
        dt * 1000.0
    );
    // Make it explicit that this part is not the reference engine's work. The engines
    // reproduce libradius; the time stretch is this crate's own overlap-add stage, and a
    // listener comparing against Audition should know which part they are comparing.
    warn(
        "WARNING: time stretching is not part of the Radius engine. The engines reproduce \
         libradius for pitch, but --tempo is applied by this crate's own overlap-add stage \
         (WSOLA), so its result may differ from Audition's. Pitch shifting at --tempo 100 \
         does not use it at all.",
    );
    (y, sr, nch)
}

/// How many output frames the run should produce.
///
/// `--tempo` is a speed control: 200 doubles the duration at the same pitch, 50 halves
/// it. The pitch is what `--semitones` asks for and nothing else.
///
/// Both engines are told this count, but **they cannot deliver it**: the resampler rate
/// that applies the pitch also sets how much they emit, and the granule scheduler walks
/// the input once, so their audio runs out at roughly the input length. They therefore
/// return the right *length* with the wrong *content* — cut short when asked for less,
/// zero-padded when asked for more. [`stretch_to_target`] fixes that afterwards.
///
/// The guard keeps a nonsense `--tempo` from asking for terabytes; 1000x is far beyond
/// any musical use.
fn target_frames(cli: &Cli, input: &Audio) -> usize {
    let stretch = cli.tempo / 100.0;
    let n = (input.frames() as f64 * stretch).round();
    if !n.is_finite() || n < 0.0 {
        return input.frames();
    }
    let cap = input.frames().saturating_mul(1000);
    (n as usize).min(cap)
}

fn run(cli: &Cli) -> Result<()> {
    fft::set_backend(cli.fft.backend());
    let input = audio::read(&cli.input).map_err(anyhow::Error::msg)?;
    println!(
        "in : {}  sr={} ch={} frames={} ({:.3}s)",
        cli.input,
        input.sample_rate,
        input.channels,
        input.frames(),
        input.duration(),
    );
    // The resolved output format, needed for the banner below. Computed here because the
    // writer uses the same value further down.
    let output_format = effective_container(cli).unwrap_or(radius_rs::io::Container::Wav);
    // Every setting that can change the signal, so a run is reproducible from its own
    // output. Only the ones actually in force are shown: `--quality`/`--solo` are td-only
    // and `--formant-shift`/`--no-preserve-voice` are vc-only, and listing an inert knob
    // would imply it did something.
    println!(
        "cfg: mode={:?} semitones={:+.2} tempo={:.1}% fft={:?}{}{}",
        cli.mode,
        cli.semitones,
        cli.tempo,
        cli.fft,
        match cli.mode {
            Mode::Td => format!(" quality={} solo={}", cli.quality, cli.solo),
            Mode::Vc => String::new(),
        },
        {
            let vc = match cli.mode {
                Mode::Td => String::new(),
                Mode::Vc => format!(
                    " formant_shift={:+.2} preserve_voice={}",
                    cli.formant_shift, !cli.no_preserve_voice
                ),
            };
            format!(
                "{vc} gain={:+.2}dB format={:?} bit_depth={:?} ogg_quality={:.2} mp3_bitrate={}",
                cli.gain, output_format, cli.bit_depth, cli.ogg_quality, cli.mp3_bitrate
            )
        }
    );

    // Arguments that cannot do what they say, reported before the render so the
    // pass-through shortcut below cannot skip them.
    if let Some(w) = formant_shift_warning(cli) {
        warn(&w);
    }

    let (out, sr, nch) = if is_passthrough(cli) {
        println!(
            "render: pass-through (semitones 0 at 100% tempo) — no engine run, samples copied"
        );
        (input.samples.clone(), input.sample_rate, input.channels)
    } else {
        match cli.mode {
            Mode::Td => render_td(cli, &input)?,
            Mode::Vc => render_vc(cli, &input)?,
        }
    };

    // `--tempo` cannot be delivered by the engines: their resampler rate *is* the
    // pitch, so it changes pitch and duration together, and the granule scheduler
    // walks the input once, which caps the audio they can emit at roughly the input
    // length. It is applied here instead, as overlap-add on the finished signal, which
    // is what lets the duration change while the pitch the engines produced is left
    // alone. A factor of exactly 1 (the default) skips this entirely, so the default
    // path stays byte-for-byte what it was.
    let (out, sr, nch) = stretch_to_target(cli, &input, out, nch, sr);

    // What a non-float target would have to do with this signal.
    let clip = analyse_clipping(&out);
    // The output format follows the input unless `--format` or the output
    // extension says otherwise, and the name is derived from the input when the
    // caller did not give one. Both are resolved here rather than in the writer so
    // the clipping warning below describes the file actually being produced.
    let container = effective_container(cli).unwrap_or(radius_rs::io::Container::Wav);
    let output_path = cli::unique_path(cli::output_path(cli, container));
    if let Some(w) = clip_warning(&clip, cli.bit_depth, Some(container)) {
        warn(&w);
    } else if clip.over > 0 {
        println!(
            "note: peak {:.4} (> 1.0) — kept exactly because 32-bit float wav stores any value",
            clip.peak
        );
    }

    let audio_out = Audio {
        samples: out.to_vec(),
        sample_rate: sr,
        channels: nch,
    };
    audio::write(&output_path, &audio_out, cli::write_options(cli))
        .map_err(anyhow::Error::msg)?;
    println!(
        "wrote {} ({} frames, {:.3}s, {}, peak {:.4})",
        output_path.display(),
        audio_out.frames(),
        audio_out.duration(),
        cli::describe_container(container, cli),
        clip.peak
    );

    if let Some(truth) = &cli.truth {
        if truth != "-" && !truth.is_empty() {
            let t = audio::read(truth).map_err(anyhow::Error::msg)?;
            let min_ch = nch.min(t.channels);
            let n = audio_out.frames().min(t.frames());
            let mut a = Vec::with_capacity(n * min_ch);
            let mut b = Vec::with_capacity(n * min_ch);
            for i in 0..n {
                for c in 0..min_ch {
                    a.push(audio_out.samples[i * nch + c]);
                    b.push(t.samples[i * t.channels + c]);
                }
            }
            let r = corr(&a, &b);
            let maxd = radius_rs::util::max_abs_diff(&a, &b);
            println!(
                "CORR vs gold: n={} (len {} vs {})  corr={:.8}  max|d|={:.3e}",
                a.len(),
                audio_out.frames(),
                t.frames(),
                r,
                maxd
            );
        }
    }
    Ok(())
}

fn render_td(cli: &Cli, input: &Audio) -> Result<(Vec<f32>, u32, usize)> {
    let nch = input.channels;
    let mut st = TdState::new(input.sample_rate, cli.quality, cli.solo, nch);
    // Pitch only: the tempo is applied *after* the engine, by `stretch_to_target`. Passing
    // it here as well would make the engine stretch too — and its internal stretch is the
    // resampler ratio, which for a 2x tempo at +3 semitones runs at 2.378 and drives the
    // granule overlap into clipping (measured: 10 000 samples over full scale against 9 at
    // tempo 100). The engine's job is pitch; duration is the stretch stage's.
    st.set_ratio(cli.semitones, 100.0);
    st.set_gain(cli.gain);
    let g = st.geometry();
    println!(
        "cfg: hop={} f28={} win_max={} pitch_N={} L1={} maxbin={} taper={} lo={} hi={} ratio={:.12}",
        st.hop,
        st.f28,
        st.win_max,
        g.n,
        g.l1,
        g.maxbin,
        g.taper_len,
        g.lo,
        g.hi,
        st.total_ratio
    );

    // Progress: the engine writes the counter, the watchdog polls the same Arc. The
    // engine renders the *natural* length now — the tempo stretch happens after it —
    // so the bar must be measured against that, not against the stretched total.
    let (counter, total) = progress_pair();
    let expected = expected_granules(&st, input.frames());
    total.store(expected, std::sync::atomic::Ordering::Relaxed);
    st.set_progress_counter(counter.clone(), expected);
    let progress = Progress::new(!cli.no_progress, "td ", counter, total);

    let t0 = Instant::now();
    let out = st.render(&input.samples, input.frames(), input.frames());
    let dt = t0.elapsed().as_secs_f64();
    progress.finish();

    let frames = if nch == 0 { 0 } else { out.len() / nch };
    println!(
        "render: out={} frames ({:.3}s)  granule={} transient={} wrap={}  [{:.0} ms, {:.1}x RT]",
        frames,
        frames as f64 / input.sample_rate as f64,
        st.n_granule,
        st.n_transient,
        st.wrap_cnt,
        dt * 1000.0,
        (input.frames() as f64 / input.sample_rate as f64) / dt.max(1e-9)
    );
    if cli.verbose {
        eprintln!(
            "td: final cursor={} in_pos={}",
            st.cursor_final(),
            st.in_pos()
        );
    }
    Ok((out, input.sample_rate, nch))
}

fn render_vc(cli: &Cli, input: &Audio) -> Result<(Vec<f32>, u32, usize)> {
    if !radius_rs::vocoder::supported_rate(input.sample_rate) {
        bail!(
            "vocoder: unsupported sr={} (44100 and 48000 only)",
            input.sample_rate
        );
    }
    let nch = input.channels;
    let mut st = VocoderState::new(input.sample_rate, nch);
    // Pitch only: the tempo is applied *after* the engine, by `stretch_to_target`. Passing
    // it here as well would make the engine stretch too — and its internal stretch is the
    // resampler ratio, which for a 2x tempo at +3 semitones runs at 2.378 and drives the
    // granule overlap into clipping (measured: 10 000 samples over full scale against 9 at
    // tempo 100). The engine's job is pitch; duration is the stretch stage's.
    st.set_ratio(cli.semitones, 100.0);
    st.set_gain(cli.gain);
    st.set_formant_shift(cli.formant_shift);
    if cli.no_preserve_voice {
        st.set_preserve_voice(false);
    }
    println!(
        "in: {}Hz {}ch {} frames  N={}",
        input.sample_rate,
        nch,
        input.frames(),
        st.cfg.n_fft
    );

    // Progress: output frames made against the frames the engine will make. The engine
    // renders the natural length — the tempo stretch happens after it — so this is the
    // input length, not the stretched total.
    let (counter, total) = progress_pair();
    let expected = input.frames() as u64;
    total.store(expected.max(1), std::sync::atomic::Ordering::Relaxed);
    st.set_progress_counter(counter.clone(), expected as i64);
    let progress = Progress::new(!cli.no_progress, "vc ", counter, total);

    let t0 = Instant::now();
    let out = st.render_to(&input.samples, input.frames());
    let dt = t0.elapsed().as_secs_f64();
    progress.finish();

    let frames = if nch == 0 { 0 } else { out.len() / nch };
    println!(
        "render: out={} frames ({:.3}s)  [{:.0} ms, {:.1}x RT]",
        frames,
        frames as f64 / input.sample_rate as f64,
        dt * 1000.0,
        (input.frames() as f64 / input.sample_rate as f64) / dt.max(1e-9)
    );
    Ok((out, input.sample_rate, nch))
}

/// Rough number of granules a TD render will produce.
///
/// The granule loop advances by about one detected period per iteration, which at
/// the default settings lands near `frames / (hop/2)`. This only drives the
/// progress bar; the exact counts come from `n_granule` afterwards.
fn expected_granules(st: &TdState, frames: usize) -> u64 {
    let hop = st.hop.max(1) as u64;
    (frames as u64) * 2 / hop + 1
}
