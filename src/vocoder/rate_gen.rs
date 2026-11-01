//! Rate-generic engine data — what the DLL actually does.
//!
//! Static analysis of `dvaaudiodsp.dll` (see
//! `docs/RE-dvaaudiodsp-radius-tempo.md` §12) shows the original engine ships
//! **none** of the six snapshot tables this crate used to compile in: no FIR
//! taps (f32 or f64 encodings), no window banks and no synthesis windows exist
//! anywhere in the image, and the Radius region contains no sample-rate
//! literals to dispatch a two-bank design on. The rate arrives as data (the
//! live renderer carries it at `+0x60`/`+0x108`) and everything is computed
//! at init. This module is the crate's reconstruction of that, used for
//! **every** rate — there are no tables left anywhere in the engine.
//!
//! * [`design_fir_bank`] — the crossover designer: the DLL's law is a
//!   log-frequency raised-cosine split at **150 / 800 / 4500 Hz**
//!   (amplitude-complementary across bands), synthesised to 2048 taps by
//!   frequency sampling. The transition half-width `H = 0.70` in `log f`
//!   (≈ ±1 octave) was fitted against the two snapshot banks and reproduces
//!   **both** with the same parameter: −6 dB band edges on the bin, worst
//!   response deviation ≈ 2 %, rms ≈ 0.4 % (see `tests/rate_generic.rs`).
//! * [`win_table`] — the window bank: the Hann laws measured on both
//!   snapshots (bands 1–3: plain Hann, `L = n_write·{0.5, 0.3, 0.2}`,
//!   energy-normalised so `Σw² = n_write`; band 0: a two-piece window whose
//!   rise is exactly `hop` long, fall `≈0.325·n_write`, starting at
//!   `≈0.044·n_write`, with a rate-independent peak scale `1.9793857`).
//! * [`synth_windows`] — the synthesis pair: `Hann^0.7` at lengths
//!   `≈{0.408, 0.175}·n_write` with the long window scaled by
//!   `√(short/long)` (the 48 kHz convention).
//!
//! Nothing here is bit-identical to the retired snapshot tables (the parity
//! corpus retired with them); what is preserved is the measured structure —
//! the window generator matched both snapshots to ≤1 ULP on the cleanly-ruled
//! bands, and the designer reproduces the DLL's own banks' edges and response
//! to a couple of percent in the transitions. The old-vs-new render
//! comparison lives in `docs/TABLE_PROVENANCE.md`.

use crate::fft_radix2::Fft;

/// Crossover frequencies in Hz — the DLL designer's constants (from the
/// decompiled call in `sub_18015E560`: `{150.0f, 800.0f, 4500.0f}`).
const XOVERS_HZ: [f64; 3] = [150.0, 800.0, 4500.0];

/// Transition half-width in `log f`, fitted on both snapshot banks. The
/// transitions span ≈ ±1 octave (`e^0.7 ≈ 2.01`) around each crossover.
const EDGE_H: f64 = 0.70;

/// Number of taps per band (the DLL designer's fixed length).
const TAPS: usize = 2048;

/// The crossover designer: log-frequency raised-cosine band split at
/// 150/800/4500 Hz, synthesised to `[4][2048]` taps by frequency sampling.
///
/// Per band `b`, the desired gain at bin frequency `f` is
///
/// * rising edge at the crossover below the band: `0.5 − 0.5·cos(π·(u+1)/2)`
///   with `u = (ln f − ln c)/H` clipped to `[0,1]`-support,
/// * falling edge at the crossover above it: the mirror image.
///
/// The four bands' desired responses are amplitude-complementary by
/// construction (each transition is shared by the two bands on either side),
/// so the summed response is exactly flat on the bin grid — the property the
/// snapshot banks showed to 1e-7. Taps come from a zero-phase inverse real
/// FFT of the desired response, circularly centred, so each band's impulse is
/// (nearly) symmetric like the originals; a per-band gain normalisation puts
/// the passband peak at exactly 1 (band 0 DC, band 3 Nyquist).
///
/// Fidelity (measured against the retired snapshot banks, same `H` for both):
/// −6 dB edges identical to the bin, worst-case gain deviation ≈ 2 %
/// in the transition shoulders, rms ≈ 0.4 %.
pub fn design_fir_bank(sr: u32) -> Vec<f32> {
    let logc = [
        XOVERS_HZ[0].ln(),
        XOVERS_HZ[1].ln(),
        XOVERS_HZ[2].ln(),
    ];
    let mut plan = Fft::new(TAPS);
    let mut out = vec![0.0f32; 4 * TAPS];
    for b in 0..4usize {
        // desired response on the 1025 bins, as the real "cart" half-spectrum
        let mut cart = vec![0.0f32; TAPS + 2];
        for k in 0..=TAPS / 2 {
            let f = k as f64 * f64::from(sr) / TAPS as f64;
            let lf = f.max(1e-9).ln();
            let mut g = 1.0f64;
            if b > 0 {
                g *= rising(lf, logc[b - 1]);
            }
            if b < 3 {
                g *= 1.0 - rising(lf, logc[b]);
            }
            cart[2 * k] = g as f32;
        }
        // zero-phase inverse transform; the even real spectrum yields a
        // symmetric impulse centred on sample 0 -> roll to the tap centre.
        // The offset puts the summed all-pass delta at index N/2 - 1 (1023),
        // matching the snapshot banks' latency exactly.
        let imp = plan.inv(&cart);
        let mut taps = vec![0.0f32; TAPS];
        for (i, t) in taps.iter_mut().enumerate() {
            *t = imp[(i + TAPS / 2 + 1) % TAPS];
        }
        // normalise the passband peak to exactly 1
        let resp = plan.fwd(&taps);
        let mut peak = 0.0f64;
        for k in 0..=TAPS / 2 {
            let m = (f64::from(resp[2 * k]) * f64::from(resp[2 * k])
                + f64::from(resp[2 * k + 1]) * f64::from(resp[2 * k + 1]))
            .sqrt();
            peak = peak.max(m);
        }
        let scale = (1.0 / peak) as f32;
        for t in taps.iter_mut() {
            *t *= scale;
        }
        out[b * TAPS..(b + 1) * TAPS].copy_from_slice(&taps);
    }
    out
}

/// The rising raised-cosine edge in `log f` centred on `c` with half-width
/// [`EDGE_H`]: `0` below `c·e^-H`, `1` above `c·e^H`.
#[inline]
fn rising(lf: f64, lc: f64) -> f64 {
    let u = (lf - lc) / EDGE_H;
    if u <= -1.0 {
        0.0
    } else if u >= 1.0 {
        1.0
    } else {
        0.5 - 0.5 * (std::f64::consts::PI * (u + 1.0) / 2.0).cos()
    }
}

/// Peak scale of the band-0 window — measured identical (bit for bit) on both
/// snapshot rates, so treated as a design constant.
const BAND0_SCALE: f32 = 1.979_385_7;

/// The vocoder window bank: `4 × n_write` flattened, band-major.
///
/// Bands 1–3 are plain Hann windows at `L = round(n_write·{0.5,0.3,0.2})`,
/// centred by `start = (n_write − L) / 2` and scaled so `Σw² = n_write`
/// (the energy normalisation both snapshots obey exactly). Band 0 is the
/// two-piece asymmetric window: a Hann rise exactly `hop = n_write/2` long, a
/// Hann fall of `≈0.325·n_write`, starting at `≈0.044·n_write`, scaled by the
/// constant [`BAND0_SCALE`].
///
/// The measured snapshots used per-rate lengths that a single rounding rule
/// does not reproduce (48 kHz band 2 is 2154 while `round(0.3·n_write)` is
/// 2155; 44.1 kHz is 1980 while the same rule gives 1980 — one point each
/// way), so this generator uses the nearest clean rule; the ≤1-sample
/// difference is audibly nothing.
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
    // own 2143 is one sample off any single rule — a ≤1 tail-sample
    // discrepancy.)
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

/// The synthesis window pair, following the 48 kHz convention
/// (`_build_synth_windows`): `Hann^0.7`, the **long** window scaled by
/// `√(len_short/len_long)`, both centred in `n_write`.
///
/// Lengths scale with `n_write` from the 48 kHz pair (2930/1256 = the same
/// `≈{0.408, 0.175}·n_write` the 44.1 kHz snapshot shows, modulo its own
/// rounding; the 44.1 kHz table had the pair's roles swapped, which the
/// generated form does not reproduce). Returns `(a, b, len_a, len_b)`.
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

    /// The bands must be amplitude-complementary on the bin grid (the DLL
    /// banks' summed response is flat to 1e-7 there).
    #[test]
    fn designed_bank_is_complementary_on_bins() {
        for sr in [32_000u32, 44_100, 48_000, 96_000] {
            let bank = design_fir_bank(sr);
            let mut plan = Fft::new(TAPS);
            let mut mags = vec![vec![0.0f64; TAPS / 2 + 1]; 4];
            for b in 0..4 {
                let taps = &bank[b * TAPS..(b + 1) * TAPS];
                let resp = plan.fwd(taps);
                for k in 0..=TAPS / 2 {
                    let re = f64::from(resp[2 * k]);
                    let im = f64::from(resp[2 * k + 1]);
                    mags[b][k] = (re * re + im * im).sqrt();
                }
            }
            let mut worst = 0.0f64;
            for k in 0..=TAPS / 2 {
                let s: f64 = mags.iter().map(|m| m[k]).sum();
                worst = worst.max((s - 1.0).abs());
            }
            assert!(worst < 1e-4, "sr={sr}: complementary sum deviates {worst}");
        }
    }

    /// Band 0 keeps unity DC gain, band 3 unity Nyquist gain, all bands'
    /// passband peaks are exactly 1.
    #[test]
    fn designed_bank_gains_are_unity() {
        for sr in [32_000u32, 44_100, 48_000, 96_000] {
            let bank = design_fir_bank(sr);
            let dc: f64 = bank[..TAPS].iter().map(|v| f64::from(*v)).sum();
            assert!((dc - 1.0).abs() < 1e-3, "sr={sr}: band0 DC {dc}");
            // Nyquist gain via the alternating sum; its sign flips with the
            // (odd) all-pass delay — magnitude is what matters
            let mut alt = 0.0f64;
            for n in 0..TAPS {
                alt += if n % 2 == 0 {
                    f64::from(bank[3 * TAPS + n])
                } else {
                    -f64::from(bank[3 * TAPS + n])
                };
            }
            assert!((alt.abs() - 1.0).abs() < 1e-3, "sr={sr}: band3 Nyquist {alt}");
        }
    }

    /// Window invariants the engine relies on: energy normalisation.
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

