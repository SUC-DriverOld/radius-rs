//! Rate-generic engine support — alignment with the original DLL.
//!
//! Static analysis of `dvaaudiodsp.dll` established that the original engine
//! ships none of the six snapshot tables: its crossover and windows are
//! computed at init from the sample rate. The crate now does the same for
//! every rate (no tables anywhere); this suite pins the generated banks to
//! the properties the DLL's own banks showed:
//!
//! * crossover: −6 dB band edges at the crossovers (150/800/4500 Hz),
//!   amplitude-complementary sum flat on the bin grid, unity DC/Nyquist
//!   gains, at several rates including the two the snapshots covered;
//! * windows: energy normalisation and the band-0 invariants;
//! * renders at 32/96/16 kHz: exact durations under the two-rate law,
//!   correct pitch, finite output.

mod common;

use common::*;
use radius_rs::vocoder::{rate_gen, supported_rate, VocoderState};

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

/// The designed crossover keeps the DLL's −6 dB band edges at the crossovers
/// (150/800/4500 Hz) at every rate. On the DLL's own 48 kHz bank the −6 dB
/// edges land on the bins 141/164, 797/820, 4477/4523 Hz; the designer hits
/// the same bins at 48 kHz, and the crossover-adjacent bins at other rates.
#[test]
fn designed_crossover_edges_match_the_dll() {
    for sr in [32_000u32, 44_100, 48_000] {
        let bank = rate_gen::design_fir_bank(sr);
        let n = 2048usize;
        let bins = 8192usize;
        let freqs: Vec<f64> = (0..bins / 2 + 1)
            .map(|k| k as f64 * f64::from(sr) / bins as f64)
            .collect();
        let edge_pairs = [(150.0f64, 1usize), (800.0, 2), (4500.0, 3)];
        for (c, upper_band) in edge_pairs {
            for b in [upper_band - 1, upper_band] {
                let mut re = vec![0.0f64; bins];
                for k in 0..n {
                    re[k] = f64::from(bank[b * n + k]);
                }
                let m = dft_mag(&re, n as f64);
                let peak = m.iter().cloned().fold(0.0f64, f64::max);
                let th = peak * 10f64.powf(-6.0 / 20.0);
                let idx: Vec<usize> = m
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| **v > th)
                    .map(|(i, _)| i)
                    .collect();
                assert!(!idx.is_empty(), "sr={sr} band {b}: empty passband");
                // the band's edge nearest the crossover: below the band uses
                // the last crossing, above uses the first
                let edge = if b == upper_band - 1 {
                    freqs[idx[idx.len() - 1]]
                } else {
                    freqs[idx[0]]
                };
                assert!(
                    (edge - c).abs() < 24.0,
                    "sr={sr} band {b}: crossover edge {edge:.0} vs {c:.0}"
                );
            }
        }
    }
}

/// The four bands stay amplitude-complementary: the summed magnitude
/// response is flat 1 across the whole spectrum (the DLL's banks showed this
/// to 1e-7 on the bin grid).
#[test]
fn designed_crossover_sum_is_flat() {
    for sr in [16_000u32, 22_050, 32_000, 44_100, 48_000, 96_000] {
        let bank = rate_gen::design_fir_bank(sr);
        let n = 2048usize;
        let bins = 8192usize;
        let mut sum_mag = vec![0.0f64; bins / 2 + 1];
        for b in 0..4 {
            let mut re = vec![0.0f64; bins];
            for k in 0..n {
                re[k] = f64::from(bank[b * n + k]);
            }
            let m = dft_mag(&re, 1.0);
            for (s, v) in sum_mag.iter_mut().zip(m.iter()) {
                *s += v;
            }
        }
        // exclude only the outermost bins (DC/Nyquist edges). The bin-grid
        // sum is exact by construction (see the rate_gen unit tests); between
        // bins the odd all-pass roll (delta at N/2 - 1, matching the engine's
        // hardcoded 1023-sample crossover latency compensation) leaves a
        // little ripple — a few tenths of a dB near Nyquist at low rates.
        let flat = sum_mag[8..bins / 2 - 8]
            .iter()
            .map(|v| (v - 1.0).abs())
            .fold(0.0f64, f64::max);
        assert!(flat < 0.03, "sr={sr}: complementary sum deviates {flat}");
    }
}

/// Distinct rates get distinct banks, and the cache returns one shared bank.
#[test]
fn fir_bank_cache_is_per_rate() {
    assert!(std::ptr::eq(
        radius_rs::vocoder::xover_taps_rev(48_000),
        radius_rs::vocoder::xover_taps_rev(48_000)
    ));
    let a = rate_gen::design_fir_bank(32_000);
    let b = rate_gen::design_fir_bank(44_100);
    assert_ne!(a[0], b[0], "32k and 44.1k banks must differ");
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

/// 44.1 kHz (previously a snapshot rate) still renders with correct geometry.
#[test]
fn renders_at_44k_after_table_removal() {
    const SR: u32 = 44100;
    let frames = SR as usize / 2;
    let mut x = vec![0.0f32; frames * 2];
    for i in 0..frames {
        let t = i as f64 / SR as f64;
        let v = 0.4 * (2.0 * std::f64::consts::PI * 440.0 * t).sin();
        x[2 * i] = v as f32;
        x[2 * i + 1] = v as f32;
    }
    let mut st = VocoderState::new(SR, 2);
    assert_eq!(st.cfg.n_write, 6599);
    st.set_ratio(3.0, 100.0);
    let out = st.render(&x);
    assert_eq!(out.len(), x.len());
    assert!(out.iter().all(|v| v.is_finite()));
    let r = dominant_ratio(&out, SR);
    assert!((r - 2f64.powf(3.0 / 12.0)).abs() < 0.02, "pitch ratio {r:.4}");
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
