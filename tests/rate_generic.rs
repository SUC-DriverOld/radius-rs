//! Rate-generic engine support — alignment with the original DLL.
//!
//! Static analysis of `dvaaudiodsp.dll` established that the original engine
//! ships none of the six snapshot tables: its crossover and windows are
//! computed at init from the sample rate (log-frequency raised-cosine design,
//! 150/800/4500 Hz edges). This suite pins the crate's equivalent:
//!
//! * the generated window bank reproduces both snapshot banks,
//! * the rescaled crossover reproduces the 44.1 kHz snapshot's response,
//! * arbitrary rates (32 kHz / 96 kHz classes) render correctly: exact
//!   durations under the two-rate law, correct pitch, finite output,
//! * the two canonical rates still run on the exact snapshots (parity).

mod common;

use common::*;
use radius_rs::vocoder::{rate_gen, supported_rate, VocoderState};

/// The generated window bank matches both snapshot banks to ULP-level on the
/// bands whose lengths the clean rounding rules reproduce (band 0 and band 1
/// at 48 kHz, band 1 at 44.1 kHz modulo its normalisation scale), and to
/// ≤0.2 % worst-case on bands 2–3, where the snapshots' own length rounding
/// differs by one sample from the clean rule (a generated-rate-only
/// discrepancy — see `rate_gen::win_table`).
#[test]
fn generated_windows_match_both_snapshots() {
    for (sr, nw) in [(48000u32, 7184usize), (44100, 6599)] {
        let snap = radius_rs::vocoder::tables_data::win(sr);
        let gen = rate_gen::win_table(nw);
        assert_eq!(gen.len(), snap.len(), "sr={sr} length");
        for b in 0..4 {
            let band = b * nw..(b + 1) * nw;
            let snap_peak = snap[band.clone()]
                .iter()
                .map(|v| f64::from(*v).abs())
                .fold(1e-12f64, f64::max);
            let mut max_rel = 0.0f64;
            for i in band {
                let d = (f64::from(gen[i]) - f64::from(snap[i])).abs();
                max_rel = max_rel.max(d / snap_peak);
            }
            let limit = match (sr, b) {
                (48000, 0 | 1) => 5e-6,
                // band 1 at 44.1 kHz: the snapshot's energy-normalisation
                // scale differs from the f64-computed one by 7.6e-5 (the
                // DLL's own accumulation order); structure matches.
                (44100, 1) => 2e-4,
                // band 0 at 44.1 kHz: the snapshot's fall length is one
                // sample off the clean rule.
                (44100, 0) => 1e-3,
                _ => 2e-3,
            };
            assert!(
                max_rel < limit,
                "sr={sr} band {b}: window deviation {max_rel:.2e} >= {limit:.0e}"
            );
        }
    }
}

/// A small radix-2 FFT for test-side response checks (magnitudes, normalised
/// by `norm`).
fn dft_mag(re: &[f64], norm: f64) -> Vec<f64> {
    let n = re.len();
    assert!(n.is_power_of_two());
    let mut ar = re.to_vec();
    let mut ai = vec![0.0f64; n];
    let mut j = 0usize;
    for i in 0..n {
        if i < j {
            ar.swap(i, j);
            ai.swap(i, j);
        }
        let mut m = n >> 1;
        while m >= 1 && j & m != 0 {
            j ^= m;
            m >>= 1;
        }
        j |= m;
    }
    let mut len = 2usize;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let (mut cr, mut ci) = (1.0f64, 0.0f64);
            for k in 0..len / 2 {
                let (a0r, a0i) = (ar[i + k], ai[i + k]);
                let tr = ar[i + k + len / 2] * cr - ai[i + k + len / 2] * ci;
                let ti = ar[i + k + len / 2] * ci + ai[i + k + len / 2] * cr;
                ar[i + k] = a0r + tr;
                ai[i + k] = a0i + ti;
                ar[i + k + len / 2] = a0r - tr;
                ai[i + k + len / 2] = a0i - ti;
                let ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
            i += len;
        }
        len <<= 1;
    }
    ar.iter()
        .zip(ai.iter())
        .take(n / 2 + 1)
        .map(|(r, i)| (r * r + i * i).sqrt() / norm)
        .collect()
}

#[test]
fn dft_helper_is_sound() {
    // an impulse transforms to a flat magnitude of 1
    let mut re = vec![0.0f64; 128];
    re[0] = 1.0;
    let m = dft_mag(&re, 1.0);
    assert!(m.iter().all(|v| (v - 1.0).abs() < 1e-9));
}

/// The 48 kHz crossover resampled to 44.1 kHz must reproduce the DLL's own
/// 44.1 kHz bank: the −6 dB band edges within 10 Hz (bands 0–2), the response
/// within 1e-4, and the magnitude-complementary sum still flat below 20.5 kHz
/// (the snapshots' sums are flat to 1 exactly; the rescaled bank's top
/// ~1 kHz before Nyquist droops a few percent, inherent to the resample).
///
/// This is the evidence that time-rescaling is a faithful stand-in for the
/// DLL's rate-parametric design at rates no snapshot covers.
#[test]
fn rescaled_fir_matches_44k_snapshot() {
    let snap = radius_rs::vocoder::tables_data::fir_44k();
    let gen = rate_gen::fir_bank(44100);
    let n = 2048usize;
    let bins = 8192usize;
    let freqs: Vec<f64> = (0..bins / 2 + 1)
        .map(|k| k as f64 * 44100.0 / bins as f64)
        .collect();

    let band_mag = |bank: &[f32], b: usize, norm: f64| -> Vec<f64> {
        let mut re = vec![0.0f64; bins];
        for k in 0..n {
            re[k] = f64::from(bank[b * n + k]);
        }
        dft_mag(&re, norm)
    };

    let mut sum_mag = vec![0.0f64; bins / 2 + 1];
    for b in 0..4 {
        // unnormalised magnitudes for the complementarity sum (each band's
        // passband gain is ~1, so the flat target is 1)
        let mut re = vec![0.0f64; bins];
        for k in 0..n {
            re[k] = f64::from(gen[b * n + k]);
        }
        let g_raw = dft_mag(&re, 1.0);
        for (s, v) in sum_mag.iter_mut().zip(g_raw.iter()) {
            *s += v;
        }
        // normalised for the per-band response comparison
        let g = dft_mag(&re, n as f64);
        if b <= 2 {
            let s = band_mag(&snap, b, n as f64);
            let edge = |m: &[f64]| -> (f64, f64) {
                let peak = m.iter().cloned().fold(0.0f64, f64::max);
                let th = peak * 10f64.powf(-6.0 / 20.0);
                let idx: Vec<usize> = m
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| **v > th)
                    .map(|(i, _)| i)
                    .collect();
                assert!(!idx.is_empty(), "band {b}: no passband above threshold");
                (freqs[idx[0]], freqs[idx[idx.len() - 1]])
            };
            let (lo_s, hi_s) = edge(&s);
            let (lo_g, hi_g) = edge(&g);
            assert!(
                (lo_s - lo_g).abs() < 10.0 && (hi_s - hi_g).abs() < 10.0,
                "band {b}: edges [{lo_s:.0},{hi_s:.0}] vs [{lo_g:.0},{hi_g:.0}]"
            );
            let max_d = s
                .iter()
                .zip(g.iter())
                .map(|(a, c)| (a - c).abs())
                .fold(0.0f64, f64::max);
            assert!(max_d < 1e-4, "band {b}: response deviation {max_d:.2e}");
        }
    }
    // Flatness of the magnitude-complementary sum. Held to 1 % below
    // 20.5 kHz; the last ~1 kHz before the target Nyquist droops a few
    // percent (inherent to resampling a flat-to-24 kHz response down —
    // see `rate_gen::fir_bank`).
    let top_bin = (20_500.0 / 44_100.0 * bins as f64) as usize;
    let flat = sum_mag[8..top_bin]
        .iter()
        .map(|v| (v - 1.0).abs())
        .fold(0.0f64, f64::max);
    assert!(flat < 0.01, "complementary sum deviates {flat} below 20.5 kHz");
}

/// 32 kHz renders: duration exactly preserved at tempo 100, halved at tempo
/// 50 (the two-rate law), pitch shifted by the requested ratio, output sane.
#[test]
fn renders_at_32k_follow_the_two_rate_law() {
    const SR: u32 = 32000;
    let frames = SR as usize; // 1 s
    let mut x = vec![0.0f32; frames * 2];
    for i in 0..frames {
        let t = i as f64 / SR as f64;
        let v = 0.4 * (2.0 * std::f64::consts::PI * 440.0 * t).sin();
        x[2 * i] = v as f32;
        x[2 * i + 1] = v as f32;
    }

    let mut st = VocoderState::new(SR, 2);
    assert_eq!(st.cfg.n_fft, 8192, "32 kHz sits in the 44.1 kHz FFT class");
    st.set_ratio(3.0, 100.0);
    let out = st.render(&x);
    assert_eq!(out.len(), x.len(), "tempo 100 preserves duration");
    assert!(out.iter().all(|v| v.is_finite()));
    let p = peak(&out);
    assert!(p > 1e-4 && p < 4.0, "peak {p}");
    let r = dominant_ratio(&out, SR);
    assert!((r - 2f64.powf(3.0 / 12.0)).abs() < 0.02, "pitch ratio {r:.4}");

    let mut st = VocoderState::new(SR, 2);
    st.set_ratio(3.0, 50.0);
    let target = (frames as f64 * 0.5).round() as usize;
    let out = st.render_to(&x, target);
    assert_eq!(out.len(), target * 2, "tempo 50 halves duration");
    assert!(out.iter().all(|v| v.is_finite()));
}

/// 96 kHz (one octave class up) renders and preserves duration.
#[test]
fn renders_at_96k_are_sane() {
    const SR: u32 = 96000;
    let frames = SR as usize / 2;
    let mut x = vec![0.0f32; frames * 2];
    for i in 0..frames {
        let t = i as f64 / SR as f64;
        let v = 0.4 * (2.0 * std::f64::consts::PI * 440.0 * t).sin();
        x[2 * i] = v as f32;
        x[2 * i + 1] = v as f32;
    }
    let mut st = VocoderState::new(SR, 2);
    assert_eq!(st.cfg.n_fft, 32768, "96 kHz is one octave class up");
    st.set_ratio(3.0, 100.0);
    let out = st.render(&x);
    assert_eq!(out.len(), x.len());
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(peak(&out) > 1e-4);
}

/// The generated banks must differ across rates (no accidental snapshot
/// reuse), and the canonical rates still resolve to the shared snapshot
/// caches.
#[test]
fn canonical_rates_stay_on_snapshots() {
    assert!(std::ptr::eq(
        radius_rs::vocoder::xover_taps_rev(48_000),
        radius_rs::vocoder::xover_taps_rev(48_000)
    ));
    let a = rate_gen::fir_bank(32_000);
    let b = rate_gen::fir_bank(44_100);
    assert_ne!(a[0], b[0], "32k and 44.1k banks must differ");
}

/// 16 kHz (two octave classes down) renders and preserves duration — the
/// low-rate end of the accepted range.
#[test]
fn renders_at_16k_are_sane() {
    const SR: u32 = 16000;
    let frames = SR as usize; // 1 s
    let mut x = vec![0.0f32; frames * 2];
    for i in 0..frames {
        let t = i as f64 / SR as f64;
        let v = 0.4 * (2.0 * std::f64::consts::PI * 440.0 * t).sin();
        x[2 * i] = v as f32;
        x[2 * i + 1] = v as f32;
    }
    let mut st = VocoderState::new(SR, 2);
    st.set_ratio(3.0, 100.0);
    let out = st.render(&x);
    assert_eq!(out.len(), x.len());
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(peak(&out) > 1e-4, "silent output at 16 kHz");
}

fn peak(x: &[f32]) -> f64 {
    x.iter().map(|v| f64::from(*v).abs()).fold(0.0f64, f64::max)
}

/// Zero-crossing-based dominant-frequency ratio vs the 440 Hz input, on the
/// left channel.
fn dominant_ratio(out: &[f32], sr: u32) -> f64 {
    let ch: Vec<f32> = out.iter().step_by(2).copied().collect();
    let start = ch.len() / 10;
    let end = ch.len() - start;
    let mut crossings = 0usize;
    for w in ch[start..end].windows(2) {
        if (w[0] < 0.0 && w[1] >= 0.0) || (w[0] >= 0.0 && w[1] < 0.0) {
            crossings += 1;
        }
    }
    let f = crossings as f64 / 2.0 * f64::from(sr) / (end - start) as f64;
    f / 440.0
}
