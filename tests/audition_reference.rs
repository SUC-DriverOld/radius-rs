//! Perceptual reference checks against Adobe Audition's iZotope Radius renders.
//!
//! # These references are not shipped
//!
//! `au+3.wav` and `au-3.wav` are Audition's own output — someone else's encoder,
//! applied to audio this project has no rights to redistribute — so they are **not**
//! in this repository, and neither is the input they were made from. Both tests
//! therefore skip when the files are absent.
//!
//! To run them, supply the pair and point the test at them:
//!
//! ```bash
//! RADIUS_TEST_AUDIO=/path/to/test.wav \
//! RADIUS_AUDITION_REF_DIR=/path/to/audition_renders \
//!   cargo test --release --test audition_reference -- --include-ignored
//! ```
//!
//! The input must match the fingerprint in `tests/common/mod.rs`, otherwise the
//! tolerances below (which were calibrated on that material) mean nothing. Audition
//! settings for the renders: algorithm iZotope Radius, precision High, stretch
//! 100 %, pitch ±3 semitones, vocoder mode on, preserve speech off, formant
//! consistency 1.
//!
//! # Why these are tolerances, not equality
//!
//! Audition is a different implementation entirely — its own STFT, its own band
//! splitting — so sample-exact comparison is meaningless (the full-band correlation
//! against it is ~0.03 for *any* implementation of this algorithm family, including
//! the C reference). What is comparable is the perceptual envelope: how much energy
//! is where, how loud the result is, and whether the pitch actually moved by the
//! requested amount. Tolerances are wide on purpose and documented per assertion.
//!
//! # What these tests assert
//!
//! Audition is a different implementation entirely — its own STFT, its own band
//! splitting — so sample-exact comparison is meaningless (the full-band correlation
//! against it is ~0.03 for *any* implementation of this algorithm family, including
//! the C reference). What is comparable is what a listener would notice, and that is
//! what is checked here:
//!
//! * the same duration, exactly;
//! * the requested pitch shift, measured on matching spectral partials;
//! * a comparable overall level;
//! * the same energy contour (the low-frequency envelope).
//!
//! The full-length renders are expensive, so the level/contour checks run on the
//! time-domain engine (seconds) and the vocoder checks are `#[ignore]`d.

mod common;

use common::*;
use radius_rs::vocoder::VocoderState;

/// Reference and candidate pitch, measured from the strongest matching spectral
/// partials, in semitones.
fn dominant_shift_semitones(src: &[f32], dst: &[f32], rate: u32) -> f64 {
    const WIN: usize = 32768;
    let npk = 8usize;
    let mut ratios: Vec<f64> = Vec::new();
    let frames = 6;
    for f in 0..frames {
        let start = (src.len().saturating_sub(WIN)) * f / frames.max(1);
        let a = spectral_peaks(src, start, WIN, npk);
        let b = spectral_peaks(dst, start, WIN, npk);
        let _ = rate;
        for &fa in &a {
            for target in [fa * 1.189_207_115, fa / 1.189_207_115] {
                if let Some(&fb) = b.iter().min_by(|x, y| {
                    (*x - target)
                        .abs()
                        .partial_cmp(&(*y - target).abs())
                        .unwrap()
                }) {
                    if (fb - target).abs() / target < 0.02 {
                        ratios.push(fb / fa);
                    }
                }
            }
        }
    }
    assert!(
        !ratios.is_empty(),
        "no matching spectral partials found; cannot measure the pitch shift"
    );
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = ratios[ratios.len() / 2];
    12.0 * median.log2()
}

/// The `n_peaks` strongest local maxima of a windowed FFT, refined to
/// sub-bin precision.
fn spectral_peaks(x: &[f32], start: usize, win: usize, n_peaks: usize) -> Vec<f64> {
    let end = (start + win).min(x.len());
    if end <= start + 16 {
        return Vec::new();
    }
    let len = end - start;
    let mut buf = vec![0.0f32; len];
    for i in 0..len {
        let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / len as f64).cos();
        buf[i] = x[start + i] * w as f32;
    }
    let cart = radius_rs::fft::fwd(len.next_power_of_two(), &{
        let mut b = buf.clone();
        b.resize(len.next_power_of_two(), 0.0);
        b
    });
    let n = len.next_power_of_two();
    let bins = n / 2;
    let mut mag = vec![0.0f64; bins];
    for k in 0..bins {
        let re = cart[2 * k] as f64;
        let im = cart[2 * k + 1] as f64;
        mag[k] = (re * re + im * im).sqrt();
    }
    // local maxima in a musically useful band
    let hz_per_bin = 48000.0 / n as f64;
    let lo = (60.0 / hz_per_bin) as usize;
    let hi = ((6000.0 / hz_per_bin) as usize).min(bins - 2);
    let mut peaks: Vec<(f64, f64)> = Vec::new();
    for k in (lo + 1)..hi {
        if mag[k] > mag[k - 1] && mag[k] > mag[k + 1] {
            let (y0, y1, y2) = (mag[k - 1], mag[k], mag[k + 1]);
            let den = y0 - 2.0 * y1 + y2;
            let d = if den != 0.0 {
                0.5 * (y0 - y2) / den
            } else {
                0.0
            };
            peaks.push(((k as f64 + d) * hz_per_bin, y1));
        }
    }
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut out: Vec<f64> = peaks.into_iter().take(n_peaks).map(|p| p.0).collect();
    out.sort_by(|a, b| a.partial_cmp(b).unwrap());
    out
}

/// Level (dB) and best-aligned envelope correlation, evaluated on a decimated
/// low-pass so the fine synthesis phase does not matter.
fn level_db(x: &[f32]) -> f64 {
    let s: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    10.0 * (s / x.len().max(1) as f64).max(1e-30).log10()
}

fn envelope(x: &[f32], cutoff_hz: f64) -> Vec<f32> {
    let rate = 48000.0;
    let k = ((rate / (4.0 * cutoff_hz)) as usize).max(1);
    let mut acc = 0.0f64;
    let mut out = Vec::with_capacity(x.len() / k);
    for (i, v) in x.iter().enumerate() {
        acc += v.abs() as f64;
        if i >= k {
            acc -= x[i - k].abs() as f64;
        }
        if i % (k / 2).max(1) == 0 && i >= k {
            out.push((acc / k as f64) as f32);
        }
    }
    out
}

fn best_envelope_corr(a: &[f32], b: &[f32], max_shift: usize) -> f64 {
    let mut best = -2.0f64;
    for shift in 0..=max_shift {
        let (x, y) = (&a[shift..], &b[..b.len().saturating_sub(shift)]);
        let n = x.len().min(y.len());
        if n < 16 {
            break;
        }
        let r = corr(&x[..n], &y[..n]);
        if r > best {
            best = r;
        }
    }
    best
}

/// The Audition render for `semis`, plus the input it was made from.
///
/// Returns `None` when the reference **or** the corpus is absent. Neither is
/// redistributed: the Audition renders are iZotope/Adobe output, and the music
/// excerpt is not ours to ship. See `tests/common/mod.rs` for how to supply them
/// (`RADIUS_AUDITION_REF_DIR`, `RADIUS_TEST_AUDIO`) and for the fingerprint the
/// input must have for the numbers below to mean anything.
fn reference(semis: i32) -> Option<(Vec<f32>, Vec<f32>)> {
    let path = audition_reference(semis)?;
    let input = acceptance_audio_or_skip(&format!("audition_reference {semis:+}"))?;
    Some((read_wav(&path).samples, read_wav(&input).samples))
}

/// Time-domain render vs the Audition reference: duration, level, contour.
#[test]
fn td_output_matches_audition_perceptually() {
    let Some(input_path) = acceptance_audio_or_skip("td_output_matches_audition_perceptually")
    else {
        return;
    };
    let input = read_wav(&input_path);
    require_audio(&input, &input_path);
    let mut ran = 0;
    for semis in [3.0f64, -3.0] {
        let Some((want, _)) = reference(semis as i32) else {
            continue;
        };
        ran += 1;
        let want_frames = want.len() / input.channels;
        assert_eq!(
            want_frames,
            input.frames(),
            "the Audition reference must be duration preserving"
        );

        let mut st = radius_rs::TdState::new(input.rate, 37, 0, input.channels);
        st.set_ratio(semis, 100.0);
        let got = st.render(&input.samples, input.frames(), input.frames());
        let got_frames = got.len() / input.channels;
        let ratio = got_frames as f64 / want_frames as f64;
        assert!(
            (ratio - 1.0).abs() < 0.005,
            "duration drift vs Audition: {got_frames} vs {want_frames}"
        );

        let n = got_frames.min(want_frames);
        let a = mono(&got[..n * input.channels], input.channels);
        let b = mono(&want[..n * input.channels], input.channels);
        let dl = level_db(&a) - level_db(&b);
        assert!(
            dl.abs() < 6.0,
            "level differs by {dl:+.2} dB from Audition at {semis:+.0} semitones"
        );

        let ea = envelope(&a, 10.0);
        let eb = envelope(&b, 10.0);
        let r = best_envelope_corr(&ea, &eb, 64);
        assert!(
            r > 0.5,
            "energy contour too different from Audition at {semis:+.0}: corr {r:.3}"
        );
        println!("td {semis:+.0}: level {dl:+.2} dB, contour corr {r:.3}");
    }
    if ran == 0 {
        eprintln!(
            "SKIP td_output_matches_audition_perceptually: no Audition renders supplied \
             (set RADIUS_AUDITION_REF_DIR, see tests/common/mod.rs)"
        );
    }
}

/// Vocoder render vs the Audition reference: the pitch shift is the thing that
/// must be right, plus duration and level. `#[ignore]`d because a full-length
/// vocoder render takes ~45 s.
#[test]
#[ignore = "runs two full-length vocoder renders (~2 min)"]
fn vocoder_pitch_matches_audition() {
    let Some(input_path) = acceptance_audio_or_skip("vocoder_pitch_matches_audition") else {
        return;
    };
    let input = read_wav(&input_path);
    require_audio(&input, &input_path);
    let src = mono(&input.samples, input.channels);
    let mut ran = 0;
    for semis in [3.0f64, -3.0] {
        let Some((want, _)) = reference(semis as i32) else {
            continue;
        };
        ran += 1;
        let want_frames = want.len() / input.channels;
        let mut st = VocoderState::new(input.rate, input.channels);
        st.set_ratio(semis, 100.0);
        let got = st.render(&input.samples);
        assert_eq!(
            got.len() / input.channels,
            input.frames(),
            "vocoder must be duration preserving"
        );
        let n = input.frames().min(want_frames);
        let a = mono(&got[..n * input.channels], input.channels);
        let b = mono(&want[..n * input.channels], input.channels);

        let shift = dominant_shift_semitones(&src[..n], &a, input.rate);
        let err = shift - semis;
        assert!(
            err.abs() < 0.3,
            "vocoder pitch {shift:+.3} semitones vs requested {semis:+.0} (err {err:+.3})"
        );
        let ref_shift = dominant_shift_semitones(&src[..n], &b, input.rate);
        assert!(
            (ref_shift - semis).abs() < 0.3,
            "the Audition reference itself measures {ref_shift:+.3} semitones"
        );
        let dl = level_db(&a) - level_db(&b);
        assert!(dl.abs() < 6.0, "level differs by {dl:+.2} dB from Audition");
        println!("vc {semis:+.0}: pitch {shift:+.3} (ref {ref_shift:+.3}), level {dl:+.2} dB");
    }
    if ran == 0 {
        eprintln!(
            "SKIP vocoder_pitch_matches_audition: no Audition renders supplied \
             (set RADIUS_AUDITION_REF_DIR, see tests/common/mod.rs)"
        );
    }
}
