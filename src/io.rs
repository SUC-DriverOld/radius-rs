//! Audio input and output, entirely through ffmpeg.
//!
//! There is no hand-written codec here on purpose. Reading and writing audio is a
//! solved problem and ffmpeg solves all of it: every container, every codec, and the
//! sample-format conversion. This module's whole job is to move interleaved `f32`
//! between the engines and that one program.
//!
//! # Locating ffmpeg
//!
//! The binary is run by name (`ffmpeg`, so `PATH` decides) unless `RADIUS_FFMPEG` names
//! one:
//!
//! ```text
//! RADIUS_FFMPEG=/opt/ffmpeg/bin/ffmpeg
//! RADIUS_FFMPEG=C:\Program Files\ffmpeg\bin\ffmpeg.exe
//! ```
//!
//! If it cannot be run, every entry point here fails with an error that says so and
//! says how to fix it. There is no fallback path, because a silent fallback is how you
//! end up with a subtly wrong render.
//!
//! # How audio crosses the boundary
//!
//! As interleaved `f32` little-endian samples (`-f f32le`), which is the engines'
//! native format:
//!
//! * **read**: `ffmpeg -i FILE -f f32le -` and the bytes come back on stdout.
//! * **write**: raw samples on `ffmpeg -f f32le -ar R -ac C -i pipe:0 … FILE`.
//!
//! Sample rate and channel count are never resampled or remixed. They come from the
//! input and are passed to the encoder unchanged; a mismatch between the request and
//! what ffmpeg reports for the result is an error.

use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Interleaved `f32` audio plus its format.
#[derive(Debug, Clone, PartialEq)]
pub struct Audio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: usize,
}

impl Audio {
    pub fn frames(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.samples.len() / self.channels
        }
    }

    pub fn duration(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.frames() as f64 / self.sample_rate as f64
        }
    }

    /// Mono mixdown (the engines' pitch analysis is mono even in stereo mode).
    pub fn mono(&self) -> Vec<f32> {
        if self.channels <= 1 {
            return self.samples.clone();
        }
        let n = self.channels;
        (0..self.frames())
            .map(|i| self.samples[i * n..(i + 1) * n].iter().sum::<f32>() / n as f32)
            .collect()
    }
}

/// Output container, overriding the extension when set.
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Container {
    Wav,
    Flac,
    Ogg,
    Mp3,
}

impl Container {
    /// Container implied by a file extension.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, String> {
        let ext = path
            .as_ref()
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        match ext.as_str() {
            "wav" | "wave" => Ok(Container::Wav),
            "flac" => Ok(Container::Flac),
            "ogg" | "oga" => Ok(Container::Ogg),
            "mp3" => Ok(Container::Mp3),
            other => Err(format!(
                "unsupported output extension {other:?}; use .wav/.flac/.ogg/.mp3 or --format"
            )),
        }
    }

    /// ffmpeg's muxer name.
    fn muxer(self) -> &'static str {
        match self {
            Container::Wav => "wav",
            Container::Flac => "flac",
            Container::Ogg => "ogg",
            Container::Mp3 => "mp3",
        }
    }

    /// The file extension this container is written with, without the dot.
    ///
    /// Used when the CLI has to invent an output name, so that the name it invents
    /// implies the container it is about to write.
    pub fn extension(self) -> &'static str {
        self.muxer()
    }

    /// The codec and its quality options.
    fn encoder_args(self, opts: &WriteOptions) -> Result<Vec<String>, String> {
        let v = |s: &str| s.to_string();
        Ok(match self {
            Container::Wav => match opts.depth {
                BitDepth::I16 => vec![v("-c:a"), v("pcm_s16le")],
                BitDepth::I24 => vec![v("-c:a"), v("pcm_s24le")],
                BitDepth::F32 => vec![v("-c:a"), v("pcm_f32le")],
                BitDepth::I32 => {
                    return Err("32-bit integer WAV is not supported; use -b 24 or -b 32f".into())
                }
            },
            Container::Flac => match opts.depth {
                BitDepth::I16 => vec![v("-c:a"), v("flac"), v("-sample_fmt"), v("s16")],
                // FLAC has no 32-bit integer subframe and cannot store float, so
                // 24/32/32f all become 24-bit.
                _ => vec![v("-c:a"), v("flac"), v("-sample_fmt"), v("s32")],
            },
            Container::Ogg => {
                // libvorbis quality is 0.0..1.0 in this crate's CLI; ffmpeg calls the
                // same scale 0..10.
                let q = (opts.ogg_quality as f64).clamp(0.0, 1.0) * 10.0;
                vec![v("-c:a"), v("libvorbis"), v("-q:a"), format!("{q:.2}")]
            }
            Container::Mp3 => vec![
                v("-c:a"),
                v("libmp3lame"),
                v("-b:a"),
                format!("{}k", opts.mp3_kbps),
            ],
        })
    }
}

/// Sample depth for the output.
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum BitDepth {
    #[value(name = "16")]
    I16,
    #[value(name = "24")]
    I24,
    #[value(name = "32")]
    I32,
    #[value(name = "32f")]
    F32,
}

impl BitDepth {
    pub fn is_float(self) -> bool {
        matches!(self, BitDepth::F32)
    }
}

/// How to encode the output.
#[derive(Debug, Clone)]
pub struct WriteOptions {
    pub depth: BitDepth,
    /// `None` means "take the container from the output path's extension".
    pub container: Option<Container>,
    /// Ogg/Vorbis quality, 0.0 (smallest) .. 1.0 (best).
    pub ogg_quality: f32,
    /// MP3 constant bitrate in kbit/s.
    pub mp3_kbps: u16,
}

impl Default for WriteOptions {
    fn default() -> Self {
        Self {
            depth: BitDepth::F32,
            container: None,
            ogg_quality: 0.9,
            mp3_kbps: 320,
        }
    }
}

// ---------------------------------------------------------------------------
// ffmpeg plumbing
// ---------------------------------------------------------------------------

/// The ffmpeg executable to run: `RADIUS_FFMPEG` if set, otherwise `ffmpeg` on `PATH`.
pub fn ffmpeg_program() -> PathBuf {
    match std::env::var_os("RADIUS_FFMPEG") {
        Some(p) if !p.is_empty() => PathBuf::from(p),
        _ => PathBuf::from("ffmpeg"),
    }
}

/// Is an ffmpeg we can actually run available?
pub fn ffmpeg_available() -> bool {
    Command::new(ffmpeg_program())
        .arg("-version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// The error to report when ffmpeg cannot be started.
fn not_runnable(program: &Path, e: std::io::Error) -> String {
    format!(
        "cannot run {}: {e}\n\
         Install ffmpeg, or point RADIUS_FFMPEG at it \
         (e.g. RADIUS_FFMPEG=/opt/ffmpeg/bin/ffmpeg).",
        program.display()
    )
}

/// Run ffmpeg with `args`, feeding `input` on stdin if given, and return stdout.
///
/// stderr is captured and used for the error message, because ffmpeg reports every real
/// problem there and a bare exit status tells the user nothing.
fn run_ffmpeg(args: &[String], input: Option<&[u8]>) -> Result<Vec<u8>, String> {
    let program = ffmpeg_program();
    let mut cmd = Command::new(&program);
    cmd.args(args)
        .stdin(if input.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd
        .spawn()
        .map_err(|e| not_runnable(&program, e))?;

    if let Some(bytes) = input {
        let mut stdin = child.stdin.take().expect("stdin was piped");
        // Write from a thread: a single-threaded write-then-read deadlocks once the
        // output pipe fills, which it does for any non-trivial file.
        let owned = bytes.to_vec();
        let writer = std::thread::spawn(move || stdin.write_all(&owned));
        let out = child
            .wait_with_output()
            .map_err(|e| format!("ffmpeg failed: {e}"))?;
        let _ = writer.join();
        if !out.status.success() {
            return Err(describe_failure(&program, args, &out.stderr));
        }
        return Ok(out.stdout);
    }

    let out = child
        .wait_with_output()
        .map_err(|e| format!("ffmpeg failed: {e}"))?;
    if !out.status.success() {
        return Err(describe_failure(&program, args, &out.stderr));
    }
    Ok(out.stdout)
}

fn describe_failure(program: &Path, args: &[String], stderr: &[u8]) -> String {
    let text = String::from_utf8_lossy(stderr);
    // The last few non-empty lines carry the actual complaint.
    let tail: Vec<&str> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .rev()
        .take(4)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!(
        "{} {} failed:\n  {}",
        program.display(),
        args.join(" "),
        tail.join("\n  ")
    )
}

/// ffmpeg's own description of the first audio stream in a file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Info {
    pub sample_rate: u32,
    pub channels: usize,
    /// The codec ffmpeg reports, e.g. `pcm_f32le`, `flac`, `vorbis`, `mp3`.
    pub codec: String,
    /// The stream banner, verbatim, for anything not modelled above.
    pub line: String,
}

/// Ask ffmpeg what is in a file without decoding it.
///
/// This runs `ffmpeg -i FILE -f null -` and parses the banner, so it is the same
/// program and the same view of the file that [`read`] and [`write`] use.
pub fn info(path: impl AsRef<Path>) -> Result<Info, String> {
    let path = path.as_ref();
    let args = vec![
        "-hide_banner".to_string(),
        "-nostdin".to_string(),
        "-i".to_string(),
        path.to_string_lossy().into_owned(),
        "-f".to_string(),
        "null".to_string(),
        "-".to_string(),
    ];
    let program = ffmpeg_program();
    let out = Command::new(&program)
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| not_runnable(&program, e))?;
    let text = String::from_utf8_lossy(&out.stderr);
    let line = text
        .lines()
        .find(|l| l.contains("Audio:"))
        .ok_or_else(|| {
            format!(
                "{} is not audio ffmpeg can read:\n  {}",
                path.display(),
                text.lines()
                    .filter(|l| !l.trim().is_empty())
                    .last()
                    .unwrap_or("(no output)")
            )
        })?
        .to_string();
    let (sample_rate, channels, codec) = parse_stream_line(&line).ok_or_else(|| {
        format!(
            "cannot read the stream description for {}:\n  {}",
            path.display(),
            line.trim()
        )
    })?;
    Ok(Info {
        sample_rate,
        channels,
        codec,
        line,
    })
}

/// Sample rate, channel count and codec name out of an ffmpeg `Audio:` line.
///
/// ```text
///   Stream #0:0: Audio: pcm_f32le ([3][0][0][0] / 0x0003), 48000 Hz, stereo, flt
///   Stream #0:0: Audio: flac, 48000 Hz, 5 channels, s32 (24 bit)
///   Stream #0:0: Audio: vorbis, 44100 Hz, mono, fltp, 80 kb/s
/// ```
fn parse_stream_line(line: &str) -> Option<(u32, usize, String)> {
    let audio = line.find("Audio:")?;
    let rest = line[audio + "Audio:".len()..].trim_start();
    let codec: String = rest
        .split(|c: char| c == ',' || c.is_whitespace())
        .next()
        .unwrap_or("")
        .to_string();
    if codec.is_empty() {
        return None;
    }

    let hz = line.find(" Hz")?;
    let before = &line[..hz];
    let rate_start = before
        .rfind(|c: char| !c.is_ascii_digit())
        .map(|i| i + 1)
        .unwrap_or(0);
    let sample_rate: u32 = before[rate_start..].parse().ok()?;

    let after = line[hz + 3..].trim_start_matches([' ', ',']);
    let channels = if let Some(i) = after.find(" channel") {
        let digits: String = after[..i]
            .chars()
            .rev()
            .take_while(|c| c.is_ascii_digit())
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        digits.parse::<usize>().ok()?
    } else if after.starts_with("mono") {
        1
    } else if after.starts_with("stereo") {
        2
    } else {
        layout_channels(after)?
    };
    Some((sample_rate, channels, codec))
}

/// Channel count for ffmpeg's layout names: `2.1`, `5.1(side)`, `7.1`, ...
///
/// The digit before the dot is the number of full-band channels and the digit after it
/// the LFE count, so `5.1` describes six streams.
fn layout_channels(after: &str) -> Option<usize> {
    let (head, tail) = after.split_once('.')?;
    let head: usize = head.trim().parse().ok()?;
    let lfe = tail.chars().next()?.to_digit(10)? as usize;
    Some(head + lfe)
}

/// Read any audio file, by decoding it to raw `f32` with ffmpeg.
pub fn read(path: impl AsRef<Path>) -> Result<Audio, String> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(format!("{}: no such file", path.display()));
    }

    let info = info(path)?;
    let args = vec![
        "-hide_banner".to_string(),
        "-nostdin".to_string(),
        "-loglevel".to_string(),
        "error".to_string(),
        "-i".to_string(),
        path.to_string_lossy().into_owned(),
        "-f".to_string(),
        "f32le".to_string(),
        "-acodec".to_string(),
        "pcm_f32le".to_string(),
        "-".to_string(),
    ];
    let raw = run_ffmpeg(&args, None)?;
    let samples = bytes_to_f32(&raw);
    if samples.is_empty() {
        return Err(format!("{}: decoded to nothing", path.display()));
    }
    Ok(Audio {
        samples,
        sample_rate: info.sample_rate,
        channels: info.channels,
    })
}

/// Write audio to any container ffmpeg can mux.
///
/// The sample rate and channel count are passed through unchanged, and ffmpeg's view of
/// the result is checked against them afterwards.
pub fn write(path: impl AsRef<Path>, audio: &Audio, opts: WriteOptions) -> Result<(), String> {
    let path = path.as_ref();
    let container = match opts.container {
        Some(c) => c,
        None => Container::from_path(path)?,
    };
    if audio.channels == 0 {
        return Err("cannot write audio with zero channels".into());
    }
    if audio.samples.is_empty() {
        return Err("cannot write an empty signal".into());
    }

    let mut args = vec![
        "-hide_banner".to_string(),
        "-nostdin".to_string(),
        "-loglevel".to_string(),
        "error".to_string(),
        "-f".to_string(),
        "f32le".to_string(),
        "-ar".to_string(),
        audio.sample_rate.to_string(),
        "-ac".to_string(),
        audio.channels.to_string(),
        "-i".to_string(),
        "pipe:0".to_string(),
    ];
    args.extend(container.encoder_args(&opts)?);
    args.push("-f".to_string());
    args.push(container.muxer().to_string());
    args.push("-y".to_string());
    args.push(path.to_string_lossy().into_owned());

    let raw = f32_to_bytes(&audio.samples);
    run_ffmpeg(&args, Some(&raw))?;

    // Verify rather than assume: the rate and channel count must be what was asked for,
    // otherwise every caller's "input and output agree" contract is a lie.
    let written = std::fs::metadata(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .len();
    if written == 0 {
        return Err(format!("{}: ffmpeg produced an empty file", path.display()));
    }
    let got = info(path)?;
    if got.sample_rate != audio.sample_rate || got.channels != audio.channels {
        return Err(format!(
            "{}: ffmpeg wrote {} Hz / {} ch but {} Hz / {} ch was requested",
            path.display(),
            got.sample_rate,
            got.channels,
            audio.sample_rate,
            audio.channels
        ));
    }
    Ok(())
}

fn bytes_to_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn f32_to_bytes(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(x.len() * 4);
    for v in x {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Stream interleaved `f32` to a writer as raw little-endian samples.
pub fn write_raw_f32(w: &mut impl Write, x: &[f32]) -> std::io::Result<()> {
    let mut bw = BufWriter::new(w);
    for v in x {
        bw.write_all(&v.to_le_bytes())?;
    }
    bw.flush()
}

/// Read raw little-endian `f32` from a reader.
pub fn read_raw_f32(r: &mut impl Read) -> std::io::Result<Vec<f32>> {
    let mut raw = Vec::new();
    r.read_to_end(&mut raw)?;
    Ok(bytes_to_f32(&raw))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn container_from_extension() {
        assert_eq!(Container::from_path("a.wav"), Ok(Container::Wav));
        assert_eq!(Container::from_path("A.FLAC"), Ok(Container::Flac));
        assert_eq!(Container::from_path("x.ogg"), Ok(Container::Ogg));
        assert_eq!(Container::from_path("y.mp3"), Ok(Container::Mp3));
        assert!(Container::from_path("z.xyz").is_err());
        assert!(Container::from_path("noext").is_err());
    }

    #[test]
    fn stream_line_parsing() {
        let stereo = "  Stream #0:0: Audio: pcm_f32le ([3][0][0][0] / 0x0003), 48000 Hz, stereo, flt, 3072 kb/s";
        assert_eq!(
            parse_stream_line(stereo),
            Some((48_000, 2, "pcm_f32le".to_string()))
        );
        let mono = "  Stream #0:0: Audio: vorbis, 44100 Hz, mono, fltp, 80 kb/s";
        assert_eq!(
            parse_stream_line(mono),
            Some((44_100, 1, "vorbis".to_string()))
        );
        let multi = "  Stream #0:0: Audio: pcm_s16le, 96000 Hz, 5.1(side), s16, 9216 kb/s";
        assert_eq!(
            parse_stream_line(multi),
            Some((96_000, 6, "pcm_s16le".to_string()))
        );
        let five = "  Stream #0:0: Audio: flac, 48000 Hz, 5 channels, s32 (24 bit)";
        assert_eq!(
            parse_stream_line(five),
            Some((48_000, 5, "flac".to_string()))
        );
        assert_eq!(
            parse_stream_line("  Stream #0:0: Video: h264, 1280x720"),
            None
        );
    }

    #[test]
    fn f32_bytes_round_trip() {
        let x = [0.0f32, 1.0, -1.0, 0.5, -0.25, f32::MIN_POSITIVE];
        let bytes = f32_to_bytes(&x);
        assert_eq!(bytes.len(), x.len() * 4);
        assert_eq!(bytes_to_f32(&bytes), x);
        // a partial trailing sample is dropped rather than panicking
        assert_eq!(bytes_to_f32(&bytes[..bytes.len() - 2]), x[..x.len() - 1]);
    }

    #[test]
    fn audio_geometry() {
        let a = Audio {
            samples: vec![0.0; 10],
            sample_rate: 1000,
            channels: 2,
        };
        assert_eq!(a.frames(), 5);
        assert!((a.duration() - 0.005).abs() < 1e-12);
        assert_eq!(a.mono().len(), 5);
    }

    #[test]
    fn ffmpeg_path_is_configurable() {
        let key = "RADIUS_FFMPEG";
        let saved = std::env::var_os(key);
        std::env::set_var(key, "some/other/ffmpeg");
        assert_eq!(ffmpeg_program(), PathBuf::from("some/other/ffmpeg"));
        std::env::set_var(key, "");
        assert_eq!(ffmpeg_program(), PathBuf::from("ffmpeg"));
        match saved {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
    }

    /// A path that cannot work must produce an actionable error, not a panic and not a
    /// silent fallback.
    #[test]
    fn unusable_ffmpeg_error_explains_itself() {
        let key = "RADIUS_FFMPEG";
        let saved = std::env::var_os(key);
        std::env::set_var(key, "definitely-not-ffmpeg-xyz");
        let p = std::env::temp_dir().join("radius_rs_unusable_ffmpeg.bin");
        std::fs::write(&p, b"not audio").unwrap();
        let err = read(&p).unwrap_err();
        assert!(err.contains("definitely-not-ffmpeg-xyz"), "{err}");
        assert!(err.contains("RADIUS_FFMPEG"), "{err}");
        std::env::remove_var(key);
        let err = read(&p).unwrap_err();
        assert!(err.contains("ffmpeg"), "{err}");
        let _ = std::fs::remove_file(&p);
        match saved {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
    }

    /// The probe and the decode must agree, and a missing file must be reported.
    #[test]
    fn missing_file_is_reported() {
        let err = read(std::env::temp_dir().join("no_such_file_anywhere.wav")).unwrap_err();
        assert!(err.contains("no such file"), "{err}");
    }
}
