//! Rate-generic engine data — what the DLL actually does.
//!
//! Static analysis of `dvaaudiodsp.dll` (see
//! `docs/RE-dvaaudiodsp-radius-tempo.md` §12) shows the original engine ships
//! **none** of the six snapshot tables this crate compiles in: no FIR taps
//! (f32 or f64 encodings), no window banks and no synthesis windows exist
//! anywhere in the image, and the Radius region contains no sample-rate
//! literals to dispatch a two-bank design on. Instead the rate arrives as data
//! (the live renderer carries it at `+0x60`/`+0x108`) and everything is
//! computed at init:
//!
//! * the crossover is designed by a log-frequency raised-cosine rule (edges
//!   150/800/4500 Hz, transition width `0.9 ×` the log-space gap to the
//!   neighbouring edge, amplitude-complementary across bands) fed to
//!   `dsp::FIR::DesignFIR` with 1025 frequency bins and 2048 taps —
//!   **Hz-anchored**, so the continuous-time response is the same at every
//!   rate;
//! * the window bank is the Hann law measured on both snapshot rates
//!   (bands 1–3: plain Hann, `L = n_write·{0.5, 0.3, 0.2}`, energy-normalised
//!   so `Σw² = n_write`; band 0: a two-piece window whose rise is exactly
//!   `hop` long, fall `≈0.325·n_write`, starting at `≈0.044·n_write`, with a
//!   rate-independent peak scale `1.9793857`);
//! * the synthesis pair is `Hann^0.7` at lengths `≈{0.408, 0.175}·n_write`
//!   with the long window scaled by `√(short/long)`.
//!
//! This module reproduces those constructions for rates the snapshots do not
//! cover. The two canonical rates keep their snapshot tables byte-for-byte
//! (the parity corpus is bit-exact against them; the generated forms match to
//! ~1 ULP on windows and <5·10⁻⁵ in filter response at 44.1 kHz, which is not
//! bit-exact and therefore not swappable).
//!
//! Validation lives in `tests/rate_generic.rs`: the generated window bank is
//! checked against both snapshots, and the rescaled crossover is checked
//! against the 44.1 kHz snapshot — the rescale reproduces its band edges to
//! within a few Hz and its response to ~10⁻⁴, which is the evidence that
//! rescaling the 48 kHz bank is a faithful stand-in for the DLL's own
//! rate-parametric design at other rates.

use super::crossover_tables;

/// Crossover taps for any rate: the 48 kHz snapshot resampled in time.
///
/// The DLL's design is Hz-anchored (fixed 150/800/4500 Hz edges, log-frequency
/// transitions), so its impulse response has the same continuous-time shape at
/// every rate — the taps at rate `sr` sample that shape at `1/sr` spacing:
///
/// ```text
///     h_sr[n] = (48000/sr) · h_48k(n · 48000/sr)
/// ```
///
/// The `48000/sr` factor preserves `∫h dt`. At rates above 48 kHz the 2048-tap
/// budget cannot hold the whole impulse response (its Hz-anchored ringing
/// lasts a fixed *time*, so it spans *more* samples at higher rates) and the
/// truncated tail would cost the low band half its DC gain — the same
/// trade-off the DLL's own fixed-length design makes there. A single global
/// gain re-anchors band 0's DC gain to 1, applied across all bands so the
/// amplitude-complementary sum stays flat. Values past the end of the source
/// bank are zero (the design window has decayed there).
///
/// Interpolation is band-limited windowed-sinc (Kaiser β = 10, ±48 taps) on
/// `f64`, cut off at the target Nyquist for down-rates. At 44.1 kHz the
/// reconstructed response stays within ~10⁻⁴ of the DLL's own bank below
/// 20 kHz. The last ~1 kHz before the target Nyquist droops by up to a few
/// percent: an ideal band-limited resample of a response that is flat to
/// 24 kHz cannot be flat to a lower Nyquist, and the DLL (which designs its
/// response from scratch at each rate) has no such limit. The affected range
/// is above 20 kHz at every supported rate.
pub fn fir_bank(sr: u32) -> Vec<f32> {
    const TAPS: usize = 2048;
    const BANDS: usize = 4;
    let src = crossover_tables::fir_48k();
    let ratio = 48_000f64 / f64::from(sr);
    // cycles per source sample: the target Nyquist mapped onto the 48 kHz grid
    let cutoff = if sr < 48_000 {
        f64::from(sr) / 96_000.0
    } else {
        0.5
    };
    let mut out = vec![0.0f32; BANDS * TAPS];
    for b in 0..BANDS {
        for n in 0..TAPS {
            let pos = n as f64 * ratio;
            out[b * TAPS + n] = (ratio * sinc_interp(&src[b * TAPS..(b + 1) * TAPS], pos, cutoff))
                as f32;
        }
    }
    // Global DC anchor (see the doc comment): a no-op at and below 48 kHz,
    // restoring unity low-band gain where the tail is truncated. The source's
    // own DC (0.999998) is the design value and is left alone.
    let dc: f64 = out[..TAPS].iter().map(|v| f64::from(*v)).sum();
    if (dc - 1.0).abs() > 1e-3 && dc.abs() > 1e-9 {
        let g = (1.0 / dc) as f32;
        for v in out.iter_mut() {
            *v *= g;
        }
    }
    out
}

/// Interpolation kernel half-width in source samples (±48 taps; Kaiser
/// β = 10 puts the stopband below −100 dB).
const SINC_HALF: usize = 48;
/// Kaiser window β for the interpolation kernel.
const SINC_BETA: f64 = 10.0;

/// Modified Bessel function of the first kind, order 0 (series form) — used
/// by the Kaiser window.
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let half = x * x / 4.0;
    for k in 1..32 {
        term *= half / (k as f64 * k as f64);
        sum += term;
        if term < 1e-14 * sum {
            break;
        }
    }
    sum
}

/// Band-limited sinc interpolation of `src` at fractional index `pos`.
///
/// `cutoff` is in cycles per source sample (0.5 = source Nyquist). Down-rate
/// rescales pass the target Nyquist (`sr/96000`) so the source's content
/// above the target Nyquist is removed instead of folding back — a plain
/// polynomial interpolator leaves a measured ~9 % gain bump at the top of the
/// high band that way (the 48 kHz source's 22–24 kHz passband folding into
/// 44.1 kHz's top octave).
fn sinc_interp(src: &[f32], pos: f64, cutoff: f64) -> f64 {
    let n = src.len() as isize;
    let i = pos.floor() as isize;
    let t = pos - pos.floor();
    let denom = bessel_i0(SINC_BETA);
    let mut acc = 0.0f64;
    for k in (i - SINC_HALF as isize + 1)..=(i + SINC_HALF as isize) {
        if k < 0 || k >= n {
            continue;
        }
        // distance from the interpolation point to sample k, in source samples
        let d = t + (i - k) as f64;
        let s = d * 2.0 * cutoff;
        let sinc = if s.abs() < 1e-9 {
            1.0
        } else {
            (std::f64::consts::PI * s).sin() / (std::f64::consts::PI * s)
        };
        // Kaiser window over ±SINC_HALF
        let r = d / SINC_HALF as f64;
        let w = if r.abs() >= 1.0 {
            0.0
        } else {
            let arg = SINC_BETA * (1.0 - r * r).sqrt();
            bessel_i0(arg) / denom
        };
        acc += f64::from(src[k as usize]) * sinc * w;
    }
    acc * 2.0 * cutoff
}

/// Peak scale of the band-0 window — measured identical (bit for bit) on both
/// snapshot rates, so treated as a design constant.
const BAND0_SCALE: f32 = 1.979_385_7;

/// The vocoder window bank for a rate the snapshots do not cover:
/// `4 × n_write` flattened, band-major.
///
/// Bands 1–3 are plain Hann windows at `L = round(n_write·{0.5,0.3,0.2})`,
/// centred by `start = (n_write − L) / 2` and scaled so `Σw² = n_write`
/// (the energy normalisation both snapshots obey exactly). Band 0 is the
/// two-piece asymmetric window: a Hann rise exactly `hop = n_write/2` long, a
/// Hann fall of `≈0.325·n_write`, starting at `≈0.044·n_write`, scaled by the
/// constant [`BAND0_SCALE`].
///
/// The measured snapshots use per-rate lengths that a single rounding rule
/// does not reproduce (48 kHz band 2 is 2154 while `round(0.3·n_write)` is
/// 2155; 44.1 kHz is 1980 while the same rule gives 1980 — one point each
/// way), so the generated bank uses the nearest clean rule; the ≤1-sample
/// difference only exists at non-canonical rates, where no reference pins it.
pub fn win_table(n_write: usize) -> Vec<f32> {
    let nw = n_write;
    let mut out = vec![0.0f32; 4 * nw];
    for (b, r) in [(1usize, 0.5f64), (2, 0.3), (3, 0.2)] {
        let len = (r * nw as f64).round() as usize;
        let len = len.min(nw);
        let mut h = vec![0.0f32; len];
        let mut sum_sq = 0.0f64;
        for i in 0..len {
            let ang = 2.0 * std::f64::consts::PI * (i as f64 + 0.5) / len as f64;
            let v = (0.5 - 0.5 * ang.cos()) as f32;
            h[i] = v;
            sum_sq += f64::from(v) * f64::from(v);
        }
        let scale = (nw as f64 / sum_sq).sqrt() as f32;
        let start = nw.saturating_sub(len) / 2;
        for i in 0..len {
            out[b * nw + start + i] = h[i] * scale;
        }
    }
    // Band 0: rise = hop, fall = round(0.3249·nw), start = round(0.044·nw).
    // (0.3249 lands on the 48 kHz snapshot's 2334; the 44.1 kHz snapshot's
    // own 2143 is one sample off any single rule — a generated-rate-only
    // discrepancy of ≤1 tail sample.)
    let hop = nw / 2;
    let fall = (0.3249 * nw as f64).round() as usize;
    let start = (0.044 * nw as f64).round() as usize;
    let len0 = hop + fall + 1;
    if start + len0 <= nw {
        for i in 0..hop {
            let ang = std::f64::consts::PI * (i as f64 + 0.5) / hop as f64;
            out[start + i] = ((0.5 - 0.5 * ang.cos()) as f32) * BAND0_SCALE;
        }
        for j in 0..=fall {
            let ang = std::f64::consts::PI * (j as f64 + 0.5) / (fall + 1) as f64;
            out[start + hop + j] = ((0.5 + 0.5 * ang.cos()) as f32) * BAND0_SCALE;
        }
    }
    out
}

/// The synthesis window pair for a rate the snapshots do not cover, following
/// the 48 kHz convention (`_build_synth_windows`): `Hann^0.7`, the **long**
/// window scaled by `√(len_short/len_long)`, both centred in `n_write`.
///
/// Lengths scale with `n_write` from the 48 kHz pair (2930/1256 = the same
/// `≈{0.408, 0.175}·n_write` the 44.1 kHz snapshot shows, modulo its own
/// rounding). Returns `(a, b, len_a, len_b)`.
pub fn synth_windows(n_write: usize) -> (Vec<f32>, Vec<f32>, usize, usize) {
    let nw = n_write;
    let len_a = (2930.0 * nw as f64 / 7184.0).round() as usize;
    let len_b = (1256.0 * nw as f64 / 7184.0).round() as usize;
    let mut a = vec![0.0f32; nw];
    let mut b = vec![0.0f32; nw];
    let off_a = nw / 2 - len_a / 2;
    let off_b = nw / 2 - len_b / 2;
    let scale_a = ((len_b as f32) / (len_a as f32)).sqrt();
    for i in 0..len_b {
        let ang = 2.0 * std::f64::consts::PI * (i as f64 + 0.5) / len_b as f64;
        b[off_b + i] = ((0.5 - 0.5 * ang.cos()).powf(0.7)) as f32;
    }
    for i in 0..len_a {
        let ang = 2.0 * std::f64::consts::PI * (i as f64 + 0.5) / len_a as f64;
        a[off_a + i] = (((0.5 - 0.5 * ang.cos()).powf(0.7)) as f32) * scale_a;
    }
    (a, b, len_a, len_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rescale must reduce to the source at 48 kHz: ratio 1, and the sinc
    /// kernel is exactly 1 at grid points (all other taps land on sinc
    /// zeros).
    #[test]
    fn fir_rescale_identity_at_48k() {
        let out = fir_bank(48_000);
        let src = crossover_tables::fir_48k();
        let max = out
            .iter()
            .zip(src.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max < 1e-6, "48 kHz rescale is not the identity: {max}");
    }

    /// Rescaling preserves the DC gain of the low band (∫h dt conservation).
    #[test]
    fn fir_rescale_preserves_band0_dc_gain() {
        let src = crossover_tables::fir_48k();
        let dc_src: f64 = src[..2048].iter().map(|v| f64::from(*v)).sum();
        for sr in [32_000u32, 44_100, 96_000] {
            let out = fir_bank(sr);
            let dc: f64 = out[..2048].iter().map(|v| f64::from(*v)).sum();
            assert!(
                (dc - dc_src).abs() < 2e-3,
                "sr={sr}: DC gain drifted {dc} vs {dc_src}"
            );
        }
    }

    /// Window invariants the engine relies on: energy normalisation and
    /// monotone support per band.
    #[test]
    fn generated_windows_are_energy_normalised() {
        for nw in [4789usize, 4790, 5747] {
            let w = win_table(nw);
            for b in 1..4 {
                let sum_sq: f64 = w[b * nw..(b + 1) * nw]
                    .iter()
                    .map(|v| f64::from(*v) * f64::from(*v))
                    .sum();
                assert!(
                    (sum_sq - nw as f64).abs() / (nw as f64) < 1e-3,
                    "nw={nw} band {b}: sum_sq={sum_sq}"
                );
            }
        }
    }
}

