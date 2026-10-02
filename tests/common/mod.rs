//! Shared helpers for the integration tests.
//!
//! # No third-party dev-dependencies
//!
//! WAV files are read and written through the crate's own
//! [`radius_rs::io::wav`], so `cargo test` needs no extra crate. Codec cases
//! (FLAC/Ogg/MP3) go through ffmpeg like the library does, and **skip** when ffmpeg is
//! not available rather than failing: the engine tests are the point, the codec round
//! trips are a bonus.
//!
//! # Test audio is not shipped
//!
//! The acceptance tests were developed against a 29.03 s 48 kHz stereo music excerpt,
//! and against two Adobe Audition (iZotope Radius) renders of it. **Those three files
//! are not part of this repository** — they are not ours to redistribute — and they
//! are listed in `.gitignore`.
//!
//! So the tests split in two:
//!
//! * **Always-on tests** use [`synthetic_stereo`], which builds a deterministic signal
//!   (a chord, a sweeping vibrato line, a linear chirp and a silent gap). It exercises
//!   every rate/geometry path and both engines, so the CI signal comes from code that
//!   is entirely ours.
//! * **Acceptance tests** need the real corpus, which changes what "correct" means
//!   (granule counts, transient counts, codec behaviour on dense material). Those call
//!   [`acceptance_audio_or_skip`] and **skip with an explanation** when it is absent,
//!   rather than silently testing something weaker.
//!
//! Point `RADIUS_TEST_AUDIO` at a different file to run them against your own material.
//! To use the exact material the numbers in the README were measured on, the file (by
//! default `tests/test.wav`, which is gitignored) must be:
//!
//! | property | value |
//! |---|---|
//! | frames × channels | 1 393 598 × 2 |
//! | sample rate | 48 000 Hz |
//! | format | 32-bit float WAV |
//! | peak / RMS | 0.977238 / 0.418927 |
//! | SHA-256 | `bab5e717630e3b425b1d7e9ed4a70aea80a8506a823bf825504c3d270b665003` |
//!
//! The two Audition references (by default `tests/au+3.wav` and `tests/au-3.wav`, also
//! gitignored) are needed only by `audition_reference.rs`, which skips without them;
//! `RADIUS_AUDITION_REF_DIR` points at wherever they live:
//!
//! | file | SHA-256 | peak | RMS |
//! |---|---|---|---|
//! | `au+3.wav` | `963672e66b8805a5eb4021271c8d9b1d1907c812e09f6782c66aeeda37dc4729` | 1.818313 | 0.394536 |
//! | `au-3.wav` | `1504d05d5572badc664b0b46b26a1cb32440781e9883e7c0296a53a4440b320f` | 1.927238 | 0.397245 |
//!
//! [`require_audio`] enforces the fingerprint, so a same-named file of the wrong
//! material fails loudly instead of producing numbers that look plausible.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// Workspace root (the crate directory).
pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The shipped acceptance audio, if it is present.
///
/// `RADIUS_TEST_AUDIO` overrides the path. The file is *not* part of the crate; see the
/// module docs.
pub fn acceptance_audio() -> PathBuf {
    match std::env::var_os("RADIUS_TEST_AUDIO") {
        Some(p) => PathBuf::from(p),
        None => root().join("tests").join("test.wav"),
    }
}

/// Does an acceptance run make sense? (The corpus may simply not be here.)
pub fn has_acceptance_audio() -> bool {
    acceptance_audio().is_file()
}

/// The acceptance audio, or `None` with a printed explanation.
///
/// Every test that depends on the corpus starts with this, so a checkout without it
/// reports "skipped" rather than failing or — worse — passing vacuously.
pub fn acceptance_audio_or_skip(test: &str) -> Option<PathBuf> {
    let path = acceptance_audio();
    if path.is_file() {
        return Some(path);
    }
    eprintln!(
        "SKIP {test}: acceptance audio not present at {}\n     \
         (the music excerpt and the Audition renders are not redistributed; set \
         RADIUS_TEST_AUDIO to run against your own material, or see tests/common/mod.rs \
         for the fingerprint this file must have)",
        path.display()
    );
    None
}

/// Assert the input is the material the documented numbers were measured on.
///
/// Checked by shape and amplitude rather than by trusting the file name.
pub fn require_audio(input: &Wav, path: &Path) {
    assert_eq!(
        (input.frames(), input.channels, input.rate),
        (1_393_598, 2, 48_000),
        "{}: expected the 29.03 s 48 kHz stereo acceptance corpus",
        path.display()
    );
    let p = peak(&input.samples);
    let r = rms(&input.samples);
    assert!(
        (p - 0.977_238).abs() < 5e-6,
        "{}: peak {p} does not match the documented corpus (0.977238)",
        path.display()
    );
    assert!(
        (r - 0.418_927).abs() < 5e-6,
        "{}: RMS {r} does not match the documented corpus (0.418927)",
        path.display()
    );
}

/// The Audition reference render for `semis`, if it is present.
///
/// `RADIUS_AUDITION_REF_DIR` overrides the directory. Not part of the crate.
pub fn audition_reference(semis: i32) -> Option<PathBuf> {
    let dir = match std::env::var_os("RADIUS_AUDITION_REF_DIR") {
        Some(d) => PathBuf::from(d),
        None => root().join("tests"),
    };
    let path = dir.join(format!("au{semis:+}.wav"));
    path.is_file().then_some(path)
}

/// Can we run the codec tests? (ffmpeg present and runnable.)
pub fn ffmpeg_or_skip(test: &str) -> bool {
    if radius_rs::io::ffmpeg_available() {
        return true;
    }
    eprintln!(
        "SKIP {test}: ffmpeg is not available (looked for {:?}; set RADIUS_FFMPEG to \
         point at it). Codec cases need it; the engine tests do not.",
        radius_rs::io::ffmpeg_program()
    );
    false
}

/// Interleaved `f32` audio plus its format.
#[derive(Debug, Clone, PartialEq)]
pub struct Wav {
    pub samples: Vec<f32>,
    pub rate: u32,
    pub channels: usize,
}

impl Wav {
    pub fn frames(&self) -> usize {
        self.samples.len() / self.channels.max(1)
    }

    pub fn audio(&self) -> radius_rs::io::Audio {
        radius_rs::io::Audio {
            samples: self.samples.clone(),
            sample_rate: self.rate,
            channels: self.channels,
        }
    }
}

impl From<radius_rs::io::Audio> for Wav {
    fn from(a: radius_rs::io::Audio) -> Self {
        Wav {
            samples: a.samples,
            rate: a.sample_rate,
            channels: a.channels,
        }
    }
}

/// Read any audio file. Always through ffmpeg, like the library.
pub fn read_wav(path: &Path) -> Wav {
    read_audio(path)
}

/// Read any audio file through the crate's reader (which is ffmpeg).
pub fn read_audio(path: &Path) -> Wav {
    radius_rs::io::read(path)
        .unwrap_or_else(|e| panic!("{e}"))
        .into()
}

/// Write 32-bit float WAV through ffmpeg.
pub fn write_wav(path: &Path, samples: &[f32], rate: u32, channels: usize) {
    let a = radius_rs::io::Audio {
        samples: samples.to_vec(),
        sample_rate: rate,
        channels,
    };
    radius_rs::io::write(path, &a, radius_rs::io::WriteOptions::default())
        .unwrap_or_else(|e| panic!("{e}"));
}

/// What ffmpeg says is in a file: codec name, sample rate, channel count.
pub fn ffmpeg_info(path: &Path) -> radius_rs::io::Info {
    radius_rs::io::info(path).unwrap_or_else(|e| panic!("{e}"))
}

/// Scratch directory shared by the integration tests.
pub fn tmp_dir() -> PathBuf {
    let dir = std::env::temp_dir().join("radius_rs_tests");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Build the deterministic synthetic programme used by the always-on tests.
///
/// Content, chosen so that every rate/geometry path has real spectral structure rather
/// than something that would pass on noise alone:
///
/// * a three-note chord (`f0`, a perfect fifth, an octave) with a slow amplitude
///   envelope;
/// * a fifth voice with vibrato that **glides** from 420 Hz to 660 Hz, so the engines'
///   per-granule pitch tracker has to follow a change;
/// * a hairpin-shaped linear chirp sweeping 200 Hz → 3.5 kHz and back, which gives the
///   vocoder's peak search and phase unwrapping something dense to chew on;
/// * a 60 ms silent gap near the end, so transient detection has a real edge;
/// * the two channels are mixed slightly differently, so the stereo phase paths are
///   exercised too.
///
/// Everything is a deterministic function of the sample index — no RNG, no clock — so
/// renders are reproducible run to run and machine to machine.
pub fn synthetic_stereo(rate: u32, frames: usize) -> Wav {
    let mut samples = Vec::with_capacity(frames * 2);
    let f0 = 180.0f64;
    for i in 0..frames {
        let t = i as f64 / rate as f64;
        let dur = frames as f64 / rate as f64;

        // slow 0.4 Hz tremolo, so the level varies over the whole file
        let env = 0.55 + 0.35 * (2.0 * std::f64::consts::PI * 0.4 * t).sin();

        // chord: root, fifth, octave
        let mut s = 0.34 * (2.0 * std::f64::consts::PI * f0 * t).sin()
            + 0.22 * (2.0 * std::f64::consts::PI * f0 * 1.5 * t).sin()
            + 0.14 * (2.0 * std::f64::consts::PI * f0 * 2.0 * t).sin();

        // gliding vibrato voice: 420 Hz -> 660 Hz across the file
        let progress = if dur > 0.0 { t / dur } else { 0.0 };
        let fv = 420.0 + 240.0 * progress;
        let vibrato = 1.0 + 0.02 * (2.0 * std::f64::consts::PI * 5.5 * t).sin();
        s += 0.20
            * (2.0 * std::f64::consts::PI * fv * vibrato * t
                + 0.6 * (2.0 * std::f64::consts::PI * fv * t).sin())
            .sin();

        // hairpin chirp: 200 Hz -> 3.5 kHz -> 200 Hz
        let sweep = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * t / dur.max(1e-9)).cos();
        let fc = 200.0 + 3300.0 * sweep;
        s += 0.12 * (2.0 * std::f64::consts::PI * fc * t).sin();

        // a genuinely silent gap, for transient detection. Both channels go to zero, so
        // the gap is a hard edge rather than a quiet patch.
        let gap_start = (frames as f64 * 0.80) as usize;
        let gap_end = gap_start + (rate as usize * 60 / 1000);
        let in_gap = i >= gap_start && i < gap_end;

        let s = if in_gap {
            0.0
        } else {
            (s * env * 0.55) as f32
        };
        samples.push(s);
        // small but non-zero channel difference so the stereo paths are live
        samples.push(if in_gap {
            0.0
        } else {
            s * 0.97 + 0.01 * ((i as f32) * 0.017).sin()
        });
    }
    Wav {
        samples,
        rate,
        channels: 2,
    }
}

/// Write the synthetic programme to a scratch WAV and return its path.
pub fn synthetic_file(tag: &str, rate: u32, frames: usize) -> PathBuf {
    let path = tmp_dir().join(format!("synth_{tag}_{rate}_{frames}.wav"));
    if !path.is_file() {
        let w = synthetic_stereo(rate, frames);
        write_wav(&path, &w.samples, rate, 2);
    }
    path
}

/// Peak absolute value.
pub fn peak(x: &[f32]) -> f32 {
    x.iter().fold(0.0f32, |a, b| a.max(b.abs()))
}

/// RMS of a slice.
pub fn rms(x: &[f32]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let s: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    (s / x.len() as f64).sqrt()
}

/// Pearson correlation over the common prefix.
pub fn corr(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let (a, b) = (&a[..n], &b[..n]);
    let ma = a.iter().map(|v| *v as f64).sum::<f64>() / n as f64;
    let mb = b.iter().map(|v| *v as f64).sum::<f64>() / n as f64;
    let (mut sa, mut sb, mut sab) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (da, db) = (a[i] as f64 - ma, b[i] as f64 - mb);
        sa += da * da;
        sb += db * db;
        sab += da * db;
    }
    if sa > 0.0 && sb > 0.0 {
        sab / (sa.sqrt() * sb.sqrt())
    } else {
        0.0
    }
}

/// Dominant frequency of a mono signal, by parabolic-interpolated FFT peak.
pub fn dominant_freq(x: &[f32], rate: u32) -> f64 {
    // 2^15 window keeps the transform cheap while giving ~1.5 Hz resolution at
    // 48 kHz; the input is Hann-windowed to suppress leakage.
    const N: usize = 1 << 15;
    let n = x.len().min(N);
    assert!(n > 1024, "signal too short for spectral analysis");
    let mut buf = vec![0.0f32; N];
    for i in 0..n {
        let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos();
        buf[i] = x[i] * w as f32;
    }
    let cart = radius_rs::fft::fwd(N, &buf);
    let bins = N / 2;
    let mut mags = vec![0.0f64; bins];
    for k in 1..bins {
        let re = cart[2 * k] as f64;
        let im = cart[2 * k + 1] as f64;
        mags[k] = (re * re + im * im).sqrt();
    }
    let mut best = 1usize;
    for k in 2..bins - 1 {
        if mags[k] > mags[best] {
            best = k;
        }
    }
    let (y0, y1, y2) = (mags[best - 1], mags[best], mags[best + 1]);
    let den = y0 - 2.0 * y1 + y2;
    let delta = if den != 0.0 {
        (0.5 * (y0 - y2) / den).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    (best as f64 + delta) * rate as f64 / N as f64
}

/// Mono mixdown of interleaved audio.
pub fn mono(x: &[f32], channels: usize) -> Vec<f32> {
    if channels == 1 {
        return x.to_vec();
    }
    let frames = x.len() / channels;
    (0..frames)
        .map(|i| {
            let mut s = 0.0f32;
            for c in 0..channels {
                s += x[i * channels + c];
            }
            s / channels as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_signal_is_well_formed() {
        let w = synthetic_stereo(48_000, 48_000);
        assert_eq!(w.samples.len(), 48_000 * 2);
        assert_eq!(w.channels, 2);
        // audible, but with headroom so the always-on render tests do not clip
        let p = peak(&w.samples);
        assert!(p > 0.2 && p <= 1.0, "peak {p} outside the useful range");
        assert!(rms(&w.samples) > 0.05, "signal is too quiet to be useful");
        // contains a real silent gap
        let silent = w
            .samples
            .chunks_exact(2)
            .filter(|f| f[0] == 0.0 && f[1] == 0.0)
            .count();
        assert!(silent > 2000, "expected a ~60 ms silent gap, got {silent}");
        // deterministic
        let again = synthetic_stereo(48_000, 48_000);
        assert_eq!(w.samples, again.samples);
    }

    #[test]
    fn synthetic_signal_has_broadband_content() {
        let w = synthetic_stereo(48_000, 48_000);
        let m = mono(&w.samples, 2);
        let f = dominant_freq(&m, 48_000);
        assert!(
            (60.0..8000.0).contains(&f),
            "dominant frequency {f} Hz is implausible"
        );
    }

    #[test]
    fn round_trip_through_the_wav_writer() {
        let path = tmp_dir().join("synthetic_round_trip.wav");
        let w = synthetic_stereo(44_100, 4410);
        write_wav(&path, &w.samples, 44_100, 2);
        let back = read_wav(&path);
        assert_eq!(back.rate, 44_100);
        assert_eq!(back.channels, 2);
        assert_eq!(back.samples, w.samples, "float WAV must round-trip exactly");
    }
}
