//! Command line front-end for the TD and vocoder engines.
//!
//! Format support is delegated to [`crate::io`], which uses pure-Rust
//! codecs (symphonia for decoding, hound/flacenc/vorbis/lame for encoding) so
//! no external `ffmpeg` is required.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::io::{BitDepth, Container, WriteOptions};
use clap::{Parser, ValueEnum};
use crate::fft::Backend;

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum FftBackend {
    /// Reference-compatible radix-2 FFT (default).
    Radix2,
    /// Faster mixed-radix FFT; vocoder output is not bit-exact.
    Rustfft,
}

impl FftBackend {
    pub fn backend(self) -> Backend {
        match self {
            Self::Radix2 => Backend::Radix2,
            Self::Rustfft => Backend::Rustfft,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Mode {
    /// Time-domain granule engine (fast default).
    Td,
    /// Phase vocoder for dense/polyphonic material (slower).
    Vc,
}

/// Widest formant shift that still resolves, in semitones.
///
/// Audition's 共振变换 runs to +/-36, but the formant operator's gain map clamps to
/// +20/-40 dB, so past roughly an octave the correction saturates and the control
/// stops doing anything: measured on a 200 Hz harmonic series with a formant at
/// 2 kHz, the formant lands within 0.3% of the same place at +9 and +12. Allowing
/// +/-36 would mean offering a range whose top half is a plateau.
pub const FORMANT_SHIFT_LIMIT: f64 = 12.0;

/// Clamp `--formant-shift` into +/-[`FORMANT_SHIFT_LIMIT`].
///
/// Clamping rather than rejecting: a value copied from Audition (which goes to 36)
/// should still render, just at the widest shift this engine can express.
fn parse_formant_shift(s: &str) -> Result<f64, String> {
    let v: f64 = s
        .parse()
        .map_err(|_| format!("'{s}' is not a number of semitones"))?;
    if !v.is_finite() {
        return Err(format!("'{s}' is not a finite number of semitones"));
    }
    Ok(v.clamp(-FORMANT_SHIFT_LIMIT, FORMANT_SHIFT_LIMIT))
}

#[derive(Parser, Debug)]
#[command(
    name = "radius",
    version,
    about = "Pitch-shift and time-stretch audio with the Radius TD or phase-vocoder engine.",
    long_about = "Pitch-shift and time-stretch audio with the Radius TD or phase-vocoder engine. \
        `-h` lists the options one line each. `--help` prints the same list with the reasoning, \
        the measured trade-offs and the Audition equivalents. The full reference is in \
        docs/CLI.md."
)]
pub struct Cli {
    #[arg(
        value_name = "INPUT",
        help = "Input audio file.",
        long_help = "Input audio file: any container or codec ffmpeg can read (wav, flac, \
            ogg/vorbis, mp3, aac/adts, m4a/alac, mkv/webm, aiff, caf, ...). \
            Its sample rate and channel count are used as-is and written unchanged to the \
            output. There is deliberately no rate option, so input and output always agree."
    )]
    pub input: String,

    #[arg(
        value_name = "OUTPUT",
        help = "Output file. Omit it to name the output after the input.",
        long_help = "Output file. Omit it and the name is derived from the input as \
            `<name>_<mode>[_st<N>][_tp<N>]`, written **next to the input**, following the \
            input's format where that format can be written and falling back to WAV \
            otherwise. \
            So `radius in.flac -m vc -s -3 --tempo 200` writes `in_vc_st-3_tp200.flac` beside \
            `in.flac`. Only the shift is in the name: quality, solo, gain and the FFT backend \
            are left out. \
            The extension picks the container unless --format overrides it. An existing file \
            is never overwritten: ` (1)`, ` (2)`, ... is inserted before the extension until \
            the name is free, for a name you typed as well as a derived one."
    )]
    pub output: Option<String>,

    #[arg(
        short = 'm',
        long,
        value_enum,
        default_value_t = Mode::Vc,
        value_name = "td|vc",
        help = "Engine: td (time domain) or vc (phase vocoder, default).",
        long_help = "Engine: td (time domain) or vc (phase vocoder, default). \
            td = the granule engine (rx_td_render): fast, tens of times faster than real \
            time, best for monophonic material. \
            vc = the phase vocoder (rx_vc_render), much slower, better on dense \
            polyphonic music. \
            The default is vc because it handles mixed material better; --mode td \
            is there when speed matters or the source is monophonic. \
            The two take different option sets: --quality and --solo are td only, \
            --formant-shift and --no-preserve-voice are vc only."
    )]
    pub mode: Mode,

    #[arg(
        long,
        value_enum,
        default_value_t = FftBackend::Radix2,
        value_name = "radix2|rustfft",
        help = "FFT backend; radix2 is reference-exact, rustfft is faster but not exact.",
        long_help = "FFT backend. Both kernels are always compiled in. \
            radix2 = this crate's own kernel, bit-exact with the reference (the default). \
            rustfft = the rustfft crate, mixed-radix: about 1.4x on the vocoder and 2.2x on \
            td, but the vocoder's output stops matching the reference (correlation ~0.994) \
            because its peak search and phase unwrapping branch on last-bit differences. \
            td is bit-identical under both on every input tried so far, but that is an \
            empirical result on those inputs, not a guarantee. See docs/FFT.md."
    )]
    pub fft: FftBackend,

    #[arg(
        short = 's',
        long,
        default_value_t = 0.0,
        value_name = "SEMITONES",
        allow_hyphen_values = true,
        help = "Pitch shift in semitones (negative = down).",
        long_help = "Pitch shift in semitones; negative shifts down. Fractional values are \
            fine. \
            At the default 0 nothing is shifted, so `radius in.wav out.flac` is a pure format \
            conversion. This is the only control that changes the pitch."
    )]
    pub semitones: f64,

    #[arg(
        short = 't',
        long,
        default_value_t = 100.0,
        value_name = "PERCENT",
        help = "Time stretch in percent (100 = unchanged).",
        long_help = "Time stretch in percent: 100 keeps the duration, 200 doubles it, 50 \
            halves it. \n\
            This is a speed control, not a second pitch control: it changes the duration and \
            leaves the pitch alone, so `-s 3 --tempo 200` is +3 semitones and twice as long. \
            The engines cannot do this themselves — their resampler rate is the pitch and their \
            granule scheduler walks the input once, so they only emit about the input's length. \
            The stretch is applied after them, by overlap-add on the finished signal, which is \
            why 100 skips it entirely and costs nothing."
    )]
    pub tempo: f64,

    #[arg(
        short = 'q',
        long,
        default_value_t = 37,
        value_name = "1-100",
        help = "TD quality, 1 (coarse) .. 100 (fine).",
        long_help = "TD only: quality, 1 (coarse) .. 100 (fine). Sets the granule length. \
            It defines the granule hop, `hop = round(sr * 0.001 * 1.5 * quality)`. At 48 kHz \
            -q 1 gives hop 72, the default -q 37 gives 2664, -q 100 gives 7200; on a 5 s \
            slice that is 10 609 / 268 / 100 granules. \
            What it changes is granularity and transient handling. Speed is nearly flat across \
            the range (221 ms .. 397 ms for those three), because every granule carries a \
            fixed 8192-point pitch FFT whose cost does not scale with the hop. \
            The reference driver's default is 37, and that is what the acceptance corpus is \
            measured at."
    )]
    pub quality: i32,

    #[arg(
        long,
        default_value_t = 0,
        value_name = "0|1",
        help = "TD only: 0 tracks the pitch each granule, 1 uses a fixed period.",
        long_help = "TD only: pitch-tracking mode. \
            0 (default) is the full engine: it re-estimates the pitch on every granule. \
            1 forces the steady-state path, which uses a fixed period and a pitch-search \
            range about ten times narrower. \
            Measured on a 5 s slice, 1 is roughly 3x faster (84 ms against 235 ms) and \
            produces a different signal (345 granules against 268, peak 1.126 against 1.210). \
            It is a cheaper analysis, not a better one."
    )]
    pub solo: i32,

    #[arg(
        short = 'g',
        long,
        default_value_t = 0.0,
        value_name = "dB",
        allow_hyphen_values = true,
        help = "Output gain in dB, applied after the render (0 = unchanged).",
        long_help = "Output gain in decibels, applied to the finished samples after the \
            engine. \
            It is a plain linear scale, so it cannot interact with the reference arithmetic \
            and `--gain 0` leaves the output bit-exact. Measured: -g -6 scales by exactly \
            0.501187, -g 6 by exactly 1.995262. \
            Use this rather than reaching for a level change inside the engine. The render is \
            not normalised, so a pitch shift can exceed full scale; the CLI warns when it does."
    )]
    pub gain: f64,

    #[arg(
        long,
        default_value_t = 0.0,
        value_name = "N",
        allow_hyphen_values = true,
        value_parser = parse_formant_shift,
        help = "Vocoder only: how the formants follow the pitch shift, in semitones (-12 .. +12).",
        long_help = "Vocoder only: how the formants adapt to the pitch shift, in semitones \
            (-12 .. +12). The shift is absolute and independent of --semitones. \
            0 (the default) shifts the formants and the pitch together, keeping the timbre and \
            the naturalness; it is the reference behaviour. Above 0 moves them up for a \
            brighter result, the classic \"male voice sounds female\"; below 0 does the \
            opposite. \
            The range stops at +/-12 rather than Audition's +/-36 because the operator's gain \
            map clamps to +20/-40 dB and saturates past about an octave; out-of-range values \
            are clamped, not rejected. Needs a pitch shift: nothing happens at --semitones 0, \
            or under --no-preserve-voice."
    )]
    pub formant_shift: f64,

    #[arg(
        long,
        help = "Vocoder only: let the spectral envelope follow the pitch (formants not preserved).",
        long_help = "Vocoder only: drop formant preservation, so the spectral envelope follows \
            the pitch instead of staying put. \
            Measured on a 200 Hz harmonic series with a formant at 2 kHz, at +3 semitones: \
            preserved leaves the formant at 1647 Hz, unpreserved moves it to 2172 Hz, i.e. it \
            follows the pitch. \
            This gives byte-identical output to --formant-shift N at the same value as \
            --semitones. The two only diverge at extreme pitches, where --formant-shift runs \
            out of correction headroom but this flag still works because it removes the \
            operator outright. Does nothing at --semitones 0, where the operator is off anyway."
    )]
    pub no_preserve_voice: bool,

    #[arg(
        short = 'f',
        long,
        value_enum,
        value_name = "wav|flac|ogg|mp3",
        help = "Force the output container instead of taking it from the extension.",
        long_help = "Force the output container. \
            Outranks both the output path's extension and the input's format. Useful when the \
            extension is ambiguous or must stay as-is (out.dat, out.tmp). An unknown extension \
            without this flag is an error rather than a guess."
    )]
    pub format: Option<Container>,

    #[arg(
        short = 'b',
        long,
        value_enum,
        default_value_t = BitDepth::F32,
        value_name = "16|24|32|32f",
        help = "WAV/FLAC sample depth (32f is the engines' native float).",
        long_help = "Sample depth for wav and flac. \
            32f (default) is IEEE float, which is what both engines natively produce, so \
            nothing is quantised on the way out. 16 and 24 are integer PCM. \
            32 is rejected on purpose: ffmpeg has no 32-bit integer PCM encoder, and quietly \
            writing 24 bit instead would be worse than an error. flac stores 32 and 32f as \
            24 bit, the deepest subframe the format has. \
            Integer and lossy targets clamp, so a render that exceeds full scale is reported \
            rather than silently wrapped. See the clipping section of docs/CLI.md."
    )]
    pub bit_depth: BitDepth,

    #[arg(
        long,
        default_value_t = 0.9,
        value_name = "0.0-1.0",
        help = "Ogg/Vorbis quality, 0..1.",
        long_help = "Ogg/Vorbis encoder quality, 0..1. \
            This crate's 0..1 is passed to libvorbis as 0..10. The default 0.9 is the top of \
            the practical range, so an Ogg output is never the reason for audible loss."
    )]
    pub ogg_quality: f32,

    #[arg(
        long,
        default_value_t = 320,
        value_name = "KBPS",
        help = "MP3 constant bitrate in kbit/s.",
        long_help = "MP3 constant bitrate in kbit/s: 64, 96, 128, 160, 192, 256 or 320. \
            The default 320 is the highest LAME offers."
    )]
    pub mp3_bitrate: u16,

    #[arg(
        long,
        help = "Do not draw the progress bar.",
        long_help = "Do not draw the progress bar. \
            The bar goes to stderr and appears only when stderr is a terminal, so pipes and \
            logs are unaffected anyway. RADIUS_PROGRESS=1 forces it on one line per update, \
            for logging. Everything machine-readable stays on stdout."
    )]
    pub no_progress: bool,

    #[arg(
        long,
        value_name = "FILE",
        help = "Compare the output against a reference file (corr, max |d|).",
        long_help = "Render a correlation report against a reference file over the common \
            prefix: sample correlation and maximum absolute difference. \
            This is the C driver's trailing truth.wav check, and it is how the parity numbers \
            in docs/VERIFICATION.md are produced."
    )]
    pub truth: Option<String>,

    #[arg(
        short,
        long,
        help = "Print engine diagnostics to stderr.",
        long_help = "Print engine diagnostics to stderr: the final cursors and write position. \
            For per-stage timing instead, set RADIUS_PROFILE=1. Combine it with \
            RADIUS_THREADS=1, because the stage timers are thread-local and a worker's time \
            would otherwise be missing from the report."
    )]
    pub verbose: bool,
}

/// Whether to colour warnings for the terminal on stderr.
///
/// `NO_COLOR` (set to anything non-empty) vetoes colour first — it is the informal
/// standard at https://no-color.org and the explicit "I never want this" signal, so
/// it also overrides `RADIUS_COLOR`. After that `RADIUS_COLOR` forces colour on or
/// off, which is what makes the yellow testable from a pipe; otherwise colour is used
/// only when stderr is a terminal, so redirected output stays plain.
fn colour_warnings() -> bool {
    if std::env::var_os("NO_COLOR").is_some_and(|v| !v.is_empty()) {
        return false;
    }
    if let Ok(v) = std::env::var("RADIUS_COLOR") {
        if !v.is_empty() {
            return v != "0";
        }
    }
    std::io::IsTerminal::is_terminal(&std::io::stderr())
}

/// Print a warning to stderr, in yellow when colour is enabled.
///
/// All warnings go through here so they look alike: the same colour, a blank line
/// before, and no terminal escape at all once output is redirected or `NO_COLOR` is
/// set.
pub fn warn(message: &str) {
    if colour_warnings() {
        eprintln!("\x1b[33m{message}\x1b[0m");
    } else {
        eprintln!("{message}");
    }
}

/// Warning for a `--formant-shift` that cannot do anything, or `None`.
///
/// There are two ways the formant control ends up inert, and only one of them can be
/// predicted from the arguments alone:
///
/// * **Structural, and caught here.** At `--semitones 0` the operator short-circuits
///   on `ratio == 1.0`, whatever the formant ratio is. Verified directly: every
///   `--formant-shift` from -12 to +12 renders byte-identically at pitch 0. This is
///   also why Audition's 保持语音特性 has nothing to do at pitch 0.
/// * **Saturation, and *not* caught here.** At a large pitch shift the operator's
///   +/-20/-40 dB gain clamps can already be consumed by the pitch correction, leaving
///   no headroom for the extra shift. Measured on a 180 Hz harmonic series with a
///   formant at 2160 Hz, the achieved shift is +12.7/+11.3/+1.4 semitones for
///   `--formant-shift` +12 at pitch +3/+12/+24, and the achieved shift is identical
///   for every formant setting at pitch +36. How early that bites depends on the
///   source spectrum, not just the arguments, so a pitch threshold would be a guess
///   and could easily cry wolf. It is documented instead of warned about.
pub fn formant_shift_warning(cli: &Cli) -> Option<String> {
    if !matches!(cli.mode, Mode::Vc) || cli.formant_shift == 0.0 {
        return None;
    }
    if cli.no_preserve_voice {
        return Some(
            "WARNING: --formant-shift has no effect under --no-preserve-voice, which turns \
             the formant operator off entirely."
                .to_string(),
        );
    }
    if cli.semitones == 0.0 {
        return Some(
            "WARNING: --formant-shift has no effect at --semitones 0. The formant operator \
             only runs alongside a pitch shift, because a shift of 0 preserves the formants \
             exactly and there is nothing left to correct."
                .to_string(),
        );
    }
    None
}

/// A live progress line on stderr.
///
/// It runs on a watchdog thread that polls a shared progress closure, so the
/// renderer itself stays free of I/O. The closure reads an atomic counter that
/// the renderer updates once per granule; the percentage is derived against an
/// estimate, so the line is monotone and approximately right rather than exact.
pub struct Progress {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
    drawn: bool,
}

/// A progress source shared between the renderer's thread and the watchdog.
///
/// A plain `&'static dyn Fn()` is not possible here — the counter lives in the
/// engine state, which is not `Send` — so the engines hand out an
/// `Arc<AtomicU64>` instead (see `TdState::progress_counter`). The watchdog holds
/// that `Arc` and never touches the engine.
pub type ProgressSource = std::sync::Arc<std::sync::atomic::AtomicU64>;

/// How many units of work the current render is expected to produce.
///
/// Stored next to the counter so the watchdog can compute a fraction without
/// reaching into the engine.
pub type ProgressTotal = std::sync::Arc<std::sync::atomic::AtomicU64>;

/// Create a `(counter, total)` pair for one render.
pub fn progress_pair() -> (ProgressSource, ProgressTotal) {
    (
        std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        std::sync::Arc::new(std::sync::atomic::AtomicU64::new(1)),
    )
}

impl Progress {
    /// Start a bar for `label`, or return a disabled instance when
    /// `--no-progress` was passed or stderr is not a terminal.
    ///
    /// `RADIUS_PROGRESS=1` forces the bar on even when stderr is not a terminal;
    /// in that mode each update is a fresh line instead of a carriage-return
    /// repaint, so the output stays readable in a log or a test.
    pub fn new(
        enabled: bool,
        label: &'static str,
        counter: ProgressSource,
        total: ProgressTotal,
    ) -> Self {
        let forced = std::env::var("RADIUS_PROGRESS")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false);
        let tty = std::io::IsTerminal::is_terminal(&std::io::stderr());
        let show = enabled && (tty || forced);
        if !show {
            return Self {
                stop: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true)),
                handle: None,
                drawn: false,
            };
        }
        let plain = forced && !tty;
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop2 = stop.clone();
        let handle = std::thread::spawn(move || {
            const WIDTH: usize = 28;
            let start = Instant::now();
            while !stop2.load(std::sync::atomic::Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(120));
                let done = counter.load(std::sync::atomic::Ordering::Relaxed);
                let total = total.load(std::sync::atomic::Ordering::Relaxed).max(1);
                let frac = (done as f64 / total as f64).clamp(0.0, 1.0);
                let elapsed = start.elapsed().as_secs_f64();
                let eta = if frac > 0.02 {
                    format!("{:.1}s left", elapsed * (1.0 - frac) / frac)
                } else {
                    "eta --".to_string()
                };
                let filled = (frac * WIDTH as f64).round() as usize;
                let bar: String = std::iter::repeat_n('=', filled)
                    .chain(std::iter::repeat_n(' ', WIDTH - filled))
                    .collect();
                if plain {
                    eprintln!(
                        "{label} [{bar}] {:5.1}%  {elapsed:6.1}s elapsed, {eta}",
                        frac * 100.0
                    );
                } else {
                    // `\x1b[K` clears the rest of the line so a shrinking suffix
                    // does not leave debris behind.
                    eprint!(
                        "\r{label} [{bar}] {:5.1}%  {elapsed:6.1}s elapsed, {eta}\x1b[K",
                        frac * 100.0
                    );
                }
            }
            if !plain {
                eprint!("\r\x1b[K");
            }
        });
        Self {
            stop,
            handle: Some(handle),
            drawn: true,
        }
    }

    /// Was a bar actually drawn? (False when suppressed or not a terminal.)
    pub fn is_drawn(&self) -> bool {
        self.drawn
    }

    /// Stop the watchdog and clear the line.
    pub fn finish(mut self) {
        self.stop_now();
    }

    fn stop_now(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for Progress {
    fn drop(&mut self) {
        self.stop_now();
    }
}

/// Result of inspecting the rendered signal against what the output format can
/// represent.
pub struct Clipping {
    /// Largest magnitude actually rendered.
    pub peak: f32,
    /// Samples whose magnitude exceeded 1.0 (they cannot be represented as PCM).
    pub over: usize,
    /// Samples close enough to full scale that the format's quantisation may
    /// round them past 1.0 (`|x| >= 0.9999`).
    pub near: usize,
    pub total: usize,
}

impl Clipping {
    /// Anything worth telling the user about?
    pub fn is_clipping(&self) -> bool {
        self.over > 0
    }
}

/// Inspect an interleaved buffer for values the chosen output depth cannot hold.
pub fn analyse_clipping(x: &[f32]) -> Clipping {
    let mut peak = 0.0f32;
    let mut over = 0usize;
    let mut near = 0usize;
    for &v in x {
        let a = v.abs();
        if a > peak {
            peak = a;
        }
        if a > 1.0 {
            over += 1;
        } else if a >= 0.9999 {
            near += 1;
        }
    }
    Clipping {
        peak,
        over,
        near,
        total: x.len(),
    }
}

/// Does this output configuration quantise to a fixed-point representation?
///
/// `32f` WAV is the engines' native format and stores any value the engine can
/// produce; every other option (integer PCM, FLAC, Ogg, MP3) has to map the
/// signal into a bounded representation and therefore clamps.
pub fn quantises(depth: BitDepth, container: Option<Container>) -> bool {
    match container {
        Some(Container::Wav) => !depth.is_float(),
        // FLAC is lossless for the integer range it stores, but a value outside
        // [-1, 1] still has to be clamped to get there; the lossy codecs clamp
        // as well.
        Some(Container::Flac) | Some(Container::Ogg) | Some(Container::Mp3) => true,
        // Unknown: fall back to the extension decision, handled by the caller.
        None => !depth.is_float(),
    }
}

/// Human-readable name of the "keep everything" option, for the warning.
///
/// Only 32-bit float WAV can hold a value outside `[-1, 1]`; every other target
/// — integer PCM, FLAC (which is lossless *within* its integer range, but still
/// has to clamp to get there), Ogg and MP3 — has to clamp. Saying "use FLAC" here
/// would send the user from one clamping format to another.
pub const CLIP_SAFE_HINT: &str =
    "32-bit float WAV is the only target that stores values beyond [-1, 1] (-f wav -b 32f); \
     lower the level first if you need an integer or lossy format";

/// Build the warning line for a clipped render, or `None` when there is nothing
/// to report.
pub fn clip_warning(c: &Clipping, depth: BitDepth, container: Option<Container>) -> Option<String> {
    if !c.is_clipping() || !quantises(depth, container) {
        return None;
    }
    let pct = 100.0 * c.over as f64 / c.total.max(1) as f64;
    let near = if c.near > 0 {
        format!(" ({} more sample(s) within 0.0001 of full scale)", c.near)
    } else {
        String::new()
    };
    Some(
        format!(
        "WARNING: the render peaks at {:.4} (> 1.0), so {} sample(s) ({:.3}%) clip when written \
         as {}. {CLIP_SAFE_HINT}.",
        c.peak,
        c.over,
        pct,
        describe_format(depth, container),
    ) + &near,
    )
}

/// Short description of the target format, for messages.
pub fn describe_format(depth: BitDepth, container: Option<Container>) -> String {
    match container {
        Some(Container::Wav) => format!("{} wav", depth_name(depth)),
        Some(Container::Flac) => "flac".to_string(),
        Some(Container::Ogg) => "ogg/vorbis".to_string(),
        Some(Container::Mp3) => "mp3".to_string(),
        None => "the selected format".to_string(),
    }
}

/// Name of a sample depth, as printed in the banner and warnings.
pub fn depth_name(depth: BitDepth) -> String {
    match depth {
        BitDepth::I16 => "16-bit PCM".to_string(),
        BitDepth::I24 => "24-bit PCM".to_string(),
        BitDepth::I32 => "32-bit PCM".to_string(),
        BitDepth::F32 => "32-bit float".to_string(),
    }
}

/// `WriteOptions` as the CLI would build them.
pub fn write_options(cli: &Cli) -> WriteOptions {
    WriteOptions {
        depth: cli.bit_depth,
        container: cli.format,
        ogg_quality: cli.ogg_quality,
        mp3_kbps: cli.mp3_bitrate,
    }
}

/// The container this run will actually produce.
///
/// Resolution order, highest priority first:
///
/// 1. `--format`, the explicit override;
/// 2. the output path's extension;
/// 3. the **input's** extension, when no output path was given and that format can
///    be written at all;
/// 4. WAV.
///
/// Step 3 is why a bare `radius in.flac` produces FLAC rather than WAV: the output
/// format follows the input unless told otherwise, and 4 is the fallback for an
/// input this crate can read but not write (compressed-only containers such as
/// AAC/M4A, which ffmpeg can encode but which the size/quality knobs here do not
/// cover).
pub fn effective_container(cli: &Cli) -> Option<Container> {
    if let Some(c) = cli.format {
        return Some(c);
    }
    if let Some(out) = &cli.output {
        if let Ok(c) = Container::from_path(out) {
            return Some(c);
        }
    }
    if let Ok(c) = Container::from_path(&cli.input) {
        return Some(c);
    }
    Some(Container::Wav)
}

/// The output path, deriving one from the input when the caller did not give it.
///
/// The derived suffix records only the engine and the two shifts that change the
/// signal, in that order, e.g. `_vc_st-3_tp200`:
///
/// | piece | shown when | example |
/// |---|---|---|
/// | mode | always | `vc`, `td` |
/// | `st<semitones>` | semitones != 0 | `st3`, `st-3`, `st2.5` |
/// | `tp<tempo>` | tempo != 100 | `tp200`, `tp87.5` |
///
/// So `-m vc -s 0 --tempo 100` yields `name_vc.flac`, and `-m vc -s -3 --tempo 200`
/// yields `name_vc_st-3_tp200.flac`. Anything else about the run (quality, solo, gain,
/// bit depth, FFT backend) is deliberately not in the name. The extension comes from
/// the resolved container, not from the input, so the name always describes the file
/// that will actually be written.
///
/// # Where the file lands
///
/// Next to the **input**, not in the working directory: `radius /music/a.flac -m vc`
/// writes `/music/a_vc.flac`. That is what `flac`, `pngquant` and friends do when
/// they derive an output name, and it is the only choice that survives a batch run —
/// `find . -name '*.flac' -exec radius {} -m vc -s 3 \;` would otherwise pile every
/// result into one directory, colliding on the stems and losing which output came
/// from which input. The ` (1)`/` (2)` collision rule below is borrowed from Finder's
/// "Keep Both", which also works beside the original file.
///
/// A path the user gave explicitly is used exactly as written, relative to the
/// working directory like any other command line path.
pub fn output_path(cli: &Cli, container: Container) -> PathBuf {
    if let Some(p) = &cli.output {
        return PathBuf::from(p);
    }
    let input = Path::new(&cli.input);
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let name = format!("{stem}{}.{}", shift_suffix(cli), container.extension());
    match input.parent() {
        // `parent()` of a bare filename is `Some("")`, which `join` would turn into
        // a CWD-relative path anyway; treating it as "no directory" keeps the
        // common `radius in.wav` case producing a plain relative name.
        Some(dir) if !dir.as_os_str().is_empty() => dir.join(name),
        _ => PathBuf::from(name),
    }
}

/// The `_vc_st-3_tp200` piece of a derived output name. See [`output_path`].
fn shift_suffix(cli: &Cli) -> String {
    let mode = match cli.mode {
        Mode::Td => "td",
        Mode::Vc => "vc",
    };
    let mut s = format!("_{mode}");
    if cli.semitones != 0.0 {
        s.push_str(&format!("_st{}", trim_num(cli.semitones)));
    }
    if cli.tempo != 100.0 {
        s.push_str(&format!("_tp{}", trim_num(cli.tempo)));
    }
    s
}

/// `3.0` -> `3`, `-3.0` -> `-3`, `2.5` -> `2.5`, `87.50` -> `87.5`.
///
/// A whole number loses its fraction entirely so the common case reads `st3` rather
/// than `st3.0`, which also keeps the name stable regardless of how the value was
/// written on the command line (`-s 3` and `-s 3.0` must not produce different
/// files).
fn trim_num(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

/// Return `desired`, or `desired` with ` (1)`, ` (2)`, ... inserted before the
/// extension until it names a file that does not exist.
///
/// The point is that a run never overwrites anything: an existing file of the same
/// name is treated as a previous run's output and stepped over, the way a browser
/// download does. The caller passes an already-unique path through unchanged.
pub fn unique_path(desired: impl AsRef<Path>) -> PathBuf {
    let desired = desired.as_ref();
    if !desired.exists() {
        return desired.to_path_buf();
    }
    let dir = desired.parent();
    let stem = desired
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let ext = desired.extension().and_then(|s| s.to_str());
    for n in 1u32.. {
        let mut name = format!("{stem} ({n})");
        if let Some(ext) = ext {
            name.push('.');
            name.push_str(ext);
        }
        let candidate = match dir {
            Some(d) if !d.as_os_str().is_empty() => d.join(&name),
            _ => PathBuf::from(&name),
        };
        if !candidate.exists() {
            return candidate;
        }
    }
    unreachable!("the counter is unbounded")
}

/// Human-readable description of what the writer will produce.
///
/// Takes the container rather than the `Cli` because the container may have been
/// inferred from the input (see [`effective_container`]) or from a derived output
/// name, so it is not always recoverable from the options alone.
pub fn describe_container(container: Container, cli: &Cli) -> String {
    match container {
        Container::Wav => format!("wav {}", depth_name(cli.bit_depth)),
        Container::Flac => format!(
            "flac {}",
            if cli.bit_depth == BitDepth::I16 {
                "16-bit"
            } else {
                "24-bit"
            }
        ),
        Container::Ogg => format!("ogg/vorbis q={:.2}", cli.ogg_quality),
        Container::Mp3 => format!("mp3 {} kbps", cli.mp3_bitrate),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formant_shift_warning_covers_exactly_the_inert_cases() {
        fn cli(args: &[&str]) -> Cli {
            let mut v = vec!["radius", "in.wav", "out.wav"];
            v.extend_from_slice(args);
            Cli::parse_from(v)
        }

        // Inert: at pitch 0 the operator short-circuits on `ratio == 1.0`, so every
        // formant shift renders byte-identically.
        let w = formant_shift_warning(&cli(&["-m", "vc", "--formant-shift", "6"]))
            .expect("pitch 0 must warn");
        assert!(w.contains("--semitones 0"), "{w}");

        // Inert: --no-preserve-voice switches the operator off outright.
        let w = formant_shift_warning(&cli(&[
            "-m",
            "vc",
            "-s",
            "3",
            "--formant-shift",
            "6",
            "--no-preserve-voice",
        ]))
        .expect("--no-preserve-voice must warn");
        assert!(w.contains("--no-preserve-voice"), "{w}");

        // Working: a pitch shift with preservation left on.
        assert!(
            formant_shift_warning(&cli(&["-m", "vc", "-s", "3", "--formant-shift", "6"])).is_none()
        );

        // Nothing to report at the default, or where the mode ignores the control.
        assert!(formant_shift_warning(&cli(&["-m", "vc", "-s", "3"])).is_none());
        assert!(formant_shift_warning(&cli(&["-m", "td", "--formant-shift", "6"])).is_none());
    }

    #[test]
    fn no_color_vetoes_forced_colour() {
        // NO_COLOR is the explicit "never colour me" signal, so it wins even over an
        // explicit RADIUS_COLOR=1.
        // SAFETY: single-threaded within this test, and env mutation is the point.
        unsafe {
            std::env::set_var("RADIUS_COLOR", "1");
            std::env::set_var("NO_COLOR", "1");
            assert!(!colour_warnings(), "NO_COLOR must veto RADIUS_COLOR");
            std::env::remove_var("NO_COLOR");
            assert!(colour_warnings(), "RADIUS_COLOR=1 must force colour");
            std::env::set_var("RADIUS_COLOR", "0");
            assert!(!colour_warnings(), "RADIUS_COLOR=0 must disable colour");
            std::env::remove_var("RADIUS_COLOR");
        }
    }

    #[test]
    fn clipping_detection_counts_over_and_near() {
        // 1.0 itself is representable, so it counts as "near" (it is exactly at
        // full scale) rather than "over".
        let x = [0.5f32, 1.0, 1.2, -1.3, 0.99995, -0.99995, 0.0];
        let c = analyse_clipping(&x);
        assert_eq!(c.over, 2, "1.2 and -1.3 exceed full scale");
        assert_eq!(
            c.near, 3,
            "1.0 and the two 0.99995 values are at full scale"
        );
        assert!((c.peak - 1.3).abs() < 1e-6);
        assert!(c.is_clipping());
    }

    #[test]
    fn in_range_audio_is_not_flagged() {
        let x = [0.5f32, -0.9, 0.9998, 0.0];
        let c = analyse_clipping(&x);
        assert!(!c.is_clipping());
        assert!(clip_warning(&c, BitDepth::I16, Some(Container::Wav)).is_none());
    }

    #[test]
    fn float_wav_is_never_warned_about() {
        let mut x = vec![0.0f32; 100];
        x[0] = 1.5;
        let c = analyse_clipping(&x);
        assert!(c.is_clipping());
        assert!(!quantises(BitDepth::F32, Some(Container::Wav)));
        assert!(clip_warning(&c, BitDepth::F32, Some(Container::Wav)).is_none());
    }

    #[test]
    fn lossless_and_lossy_both_warn() {
        let mut x = vec![0.0f32; 100];
        x[0] = 1.5;
        let c = analyse_clipping(&x);
        for (depth, container, want) in [
            (BitDepth::I16, Some(Container::Wav), true),
            (BitDepth::I24, Some(Container::Flac), true),
            (BitDepth::F32, Some(Container::Ogg), true),
            (BitDepth::F32, Some(Container::Mp3), true),
            (BitDepth::F32, Some(Container::Wav), false),
        ] {
            let w = clip_warning(&c, depth, container);
            assert_eq!(w.is_some(), want, "depth={depth:?} container={container:?}");
        }
        let w = clip_warning(&c, BitDepth::I16, Some(Container::Wav)).unwrap();
        assert!(w.contains("WARNING"), "{w}");
        assert!(w.contains("32-bit float WAV"), "{w}");
        assert!(w.contains("peaks at 1.5000"), "{w}");
    }

    /// The watchdog must actually read the shared counter while work happens.
    #[test]
    fn progress_watchdog_reads_the_shared_counter() {
        let (counter, total) = progress_pair();
        total.store(100, std::sync::atomic::Ordering::Relaxed);
        let progress = Progress::new(true, "tst", counter.clone(), total);
        for i in 1..=20u64 {
            counter.store(i * 5, std::sync::atomic::Ordering::Relaxed);
            std::thread::sleep(Duration::from_millis(15));
        }
        progress.finish();
        assert_eq!(
            counter.load(std::sync::atomic::Ordering::Relaxed),
            100,
            "the counter the watchdog watched must be the engine's"
        );
    }

    #[test]
    fn progress_can_be_disabled() {
        let (counter, total) = progress_pair();
        let progress = Progress::new(false, "tst", counter, total);
        assert!(!progress.is_drawn());
        progress.finish();
    }
}
