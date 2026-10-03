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

#[derive(Parser, Debug)]
#[command(
    name = "radius",
    version,
    about = "Pitch-shift audio with the Radius TD or phase-vocoder engine.",
    long_about = "Pitch-shift audio with the Radius TD or phase-vocoder engine.\n\nUse --help for the compact option list; see the README for detailed engine and format notes.",
    after_help = "EXAMPLES:\n  \
        radius in.wav -m td -s 3\n  \
        radius in.wav out.mp3 -m vc -s -3 --mp3-bitrate 320\n  \
        radius in.wav out.flac --bit-depth 24\n  \
        radius in.wav out.wav -m td -s 3 --truth reference.wav"
)]
pub struct Cli {
    /// Input audio file.
    #[arg(value_name = "INPUT")]
    pub input: String,

    /// Output audio file. mit it to derive a name from the input.
    #[arg(value_name = "OUTPUT")]
    pub output: Option<String>,

    /// Engine: td (default) or vc.
    #[arg(short, long, value_enum, default_value_t = Mode::Td, value_name = "td|vc")]
    pub mode: Mode,

    /// FFT backend: radix2 preserves reference output; rustfft is faster.
    #[arg(long, value_enum, default_value_t = FftBackend::Radix2, value_name = "radix2|rustfft")]
    pub fft: FftBackend,

    /// Pitch shift in semitones (negative = down).
    #[arg(
        short,
        long,
        default_value_t = 0.0,
        value_name = "SEMITONES",
        allow_hyphen_values = true
    )]
    pub semitones: f64,

    /// Time stretch percentage (100 = unchanged).
    #[arg(short, long, default_value_t = 100.0, value_name = "PERCENT")]
    pub tempo: f64,

    /// TD quality, 1 (fast) .. 100 (fine).
    #[arg(short, long, default_value_t = 37, value_name = "1-100")]
    pub quality: i32,

    /// TD steady-state mode: 0 normal, 1 steady.
    #[arg(long, default_value_t = 0, value_name = "0|1")]
    pub solo: i32,

    /// Vocoder precision, 1 (fast) .. 9 (clean).
    #[arg(short = 'p', long, default_value_t = 2, value_name = "1-9")]
    pub precision: i32,

    /// Force output container.
    #[arg(short = 'f', long, value_enum, value_name = "wav|flac|ogg|mp3")]
    pub format: Option<Container>,

    /// WAV/FLAC sample depth.
    #[arg(short = 'b', long, value_enum, default_value_t = BitDepth::F32,
          value_name = "16|24|32|32f")]
    pub bit_depth: BitDepth,

    /// Ogg/Vorbis quality, 0..1.
    #[arg(long, default_value_t = 0.9, value_name = "0.0-1.0")]
    pub ogg_quality: f32,

    /// MP3 constant bitrate (kbit/s).
    #[arg(long, default_value_t = 320, value_name = "KBPS")]
    pub mp3_bitrate: u16,

    /// Do not draw the progress bar.
    #[arg(long)]
    pub no_progress: bool,

    /// Compare output with a reference file.
    #[arg(long, value_name = "FILE")]
    pub truth: Option<String>,

    /// Print engine diagnostics.
    #[arg(short, long)]
    pub verbose: bool,
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
/// yields `name_vc_st-3_tp200.flac`. Anything else about the run (quality, precision,
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
