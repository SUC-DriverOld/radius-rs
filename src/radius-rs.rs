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

/// How many output frames the run should produce.
///
/// `--tempo` is a speed control: 200 doubles the duration at the same pitch, 50 halves
/// it. The pitch is what `--semitones` asks for and nothing else. The engines deliver
/// this themselves now — the stretch scales their schedule slope — so this is just the
/// shared target count.
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

    // `--tempo` is delivered by the engines themselves: the stretch scales the
    // engine's schedule slope (the rate its output ring advances per input frame),
    // which changes the duration and nothing else — the two rates measured in
    // `dvaaudiodsp.dll` (see `VocoderState::slope` / `TdState::set_ratio`). The old
    // post-engine WSOLA stage was removed with the rest of `src/stretch.rs`.

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
    // The engine delivers both controls, as in the DLL: the total ratio folds the
    // stretch in as its semitone equivalent (TD slot 13 computes
    // 2^((P - T)/12) with T the stretch in semitones), while the render target
    // paces the output timeline to `input * stretch`.
    st.set_ratio(cli.semitones, cli.tempo);
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

    // Progress: the engine writes the counter, the watchdog polls the same Arc.
    // The bar is measured against the stretched total the engine will produce.
    let (counter, total) = progress_pair();
    let target = target_frames(cli, input);
    let expected = expected_granules(&st, input.frames());
    total.store(expected, std::sync::atomic::Ordering::Relaxed);
    st.set_progress_counter(counter.clone(), expected);
    let progress = Progress::new(!cli.no_progress, "td ", counter, total);

    let t0 = Instant::now();
    let out = st.render(&input.samples, input.frames(), target);
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
            "vocoder: unsupported sr={} (8000-192000 Hz only)",
            input.sample_rate
        );
    }
    let nch = input.channels;
    let mut st = VocoderState::new(input.sample_rate, nch);
    // The engine delivers both controls: `semis` is the pitch and `tempo` scales
    // the schedule slope (the output ring's write rate per input frame), which
    // stretches the duration without touching the pitch — the division of labour
    // measured in dvaaudiodsp.dll (renderer+0x20 slope vs +0x1D84 pitch ratio).
    st.set_ratio(cli.semitones, cli.tempo);
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

    // Progress: output frames made against the frames the engine will make —
    // the stretched total, since the engine now paces itself to `target`.
    let target = target_frames(cli, input);
    let (counter, total) = progress_pair();
    total.store((target as u64).max(1), std::sync::atomic::Ordering::Relaxed);
    st.set_progress_counter(counter.clone(), target as i64);
    let progress = Progress::new(!cli.no_progress, "vc ", counter, total);

    let t0 = Instant::now();
    let out = st.render_to(&input.samples, target);
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
