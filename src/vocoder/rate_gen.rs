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
//! * [`design_fir_bank`] — the crossover designer, replicated **exactly**
//!   from the decompiled `sub_1800FA4B0` → `dsp::FIR::DesignFIR`
//!   (`0x1800348D0`) → `dsp::AudioProcessor::ApplyKaiserWindow`
//!   (`0x1801634D0`) chain in `dvaaudiodsp.dll`:
//!   1. desired response on the 1025 bins: per band, raised cosines in
//!      `ln f`, one per adjacent crossover. Each ramp spans
//!      `c·e^{±h}` with `h = 0.9·min(distance from ln c to the
//!      log-geometric midpoints on either side)`; the outer boundaries are
//!      `ln(e₀/2)` below the first crossover and the geometric midpoint to
//!      Nyquist above the last. Bin `k` sits at `ln(k·sr/2048)` (`k=0` →
//!      `−10`), `k` on the Nyquist bin keeps the last band's flat 1.
//!   2. `Fft::inv` — the DLL's hermitian c2r inverse FFT (1/N), whose cart
//!      input carries the real spec with zero imaginaries.
//!   3. `taps[j] = imp[(j+1025) mod 2048]` for `j ≤ 2046`, `taps[2047]=0`.
//!   4. Kaiser window of length **2047** (centre 1023), `β = 9.0`, ratio
//!      `I₀(β√(1−r²))/I₀(β)`, plain f32 product — the DLL's mode-2 call
//!      applies no post-normalisation.
//!   Reproduces both retired snapshot banks to ≤ 3·10⁻⁷ relative per tap
//!   (2–3 f32 ulp; the residue is FFT rounding order), with the banks' own
//!   bin complementarity (1e-6) and the all-pass `δ` at index 1023.
//! * [`win_table`] — the window bank: the Hann laws measured on both
//!   snapshots (bands 1–3: plain Hann at `L = 2·⌊r·⌈n_write/2⌉⌋`,
//!   `r = {0.5, 0.3, 0.2}` — the rule that reproduces all six snapshot
//!   lengths exactly — energy-normalised so `Σw² = 2·⌊n_write/2⌋`; band 0:
//!   a two-piece window whose rise is exactly `hop` long, fall
//!   `⌊0.32489·n_write⌋`, starting at `⌊0.044·n_write⌋`, with a
//!   rate-independent peak scale `1.9793857`).
//! * [`synth_windows`] — the synthesis pair: `Hann^0.7` at lengths
//!   `≈{0.408, 0.175}·n_write` with the long window scaled by
//!   `√(short/long)` (the 48 kHz convention).
//!
//! Against the retired snapshot banks: the FIR designer lands within 2–3
//! f32 ulp per tap (the residue is FFT rounding order), and the window
//! generator reproduces **every band's support exactly at both rates** with
//! values ≤ 3·10⁻⁷ relative (the residue is the DLL's `cosf` polynomial vs a
//! correctly-rounded `cos`). Render-level divergence between
//! equivalent-but-not-bit-identical banks is deterministic chaos (the
//! vocoder's phase feedback amplifies any f32 perturbation); the controlled
//! isolation experiment lives in `docs/TABLE_PROVENANCE.md`.

use crate::fft_radix2::Fft;

/// Crossover frequencies in Hz — the DLL designer's constants (from the
/// decompiled call in `sub_18015E560`: `{150.0f, 800.0f, 4500.0f}`).
const XOVERS_HZ: [f32; 3] = [150.0, 800.0, 4500.0];

/// Ramp overlap factor `1 − 0.1` — the `a6` argument of the decompiled
/// `sub_1800FA4B0` call site; each ramp takes 90 % of the distance to the
/// nearest geometric midpoint.
const OVERLAP: f32 = 0.9;

/// Kaiser `β` — the window parameter the DLL passes to `DesignFIR` (`9.0`).
const KAISER_BETA: f64 = 9.0;

/// Number of taps per band (the DLL designer's fixed length).
const TAPS: usize = 2048;

/// The crossover designer: the decompiled DLL law, exactly.
///
/// Per band `b ∈ 0..=3` the desired response on bin `k` (frequency
/// `k·sr/2048`, `k = 0..=1024`, `ln f → −10` at `k = 0`) is built from two
/// raised-cosine ramps in `ln f`:
///
/// * ramp A (the band's lower edge) centred on crossover `e[b−1]`,
/// * ramp B (the upper edge) centred on `e[b]` (band 3's ramp B degenerates
///   to a point at `ln(sr/2)`, keeping its Nyquist bin flat at 1).
///
/// A ramp centred on `c` spans `c·e^{±h}` with
/// `h = 0.9·min(ln c − L, U − ln c)`, where `L`/`U` are the log-geometric
/// midpoints to the neighbouring crossovers; the outer flanks use
/// `ln(e₀/2)` and the geometric midpoint to Nyquist. Bands are pairwise
/// amplitude-complementary by construction (adjacent bands share ramp
/// endpoints), so the summed bin response is flat.
///
/// The impulse comes from the hermitian inverse FFT of the real spec
/// (cart input, zero imaginaries — the DLL's c2r), re-indexed as
/// `taps[j] = imp[(j+1025) mod 2048]`, `taps[2047] = 0`, then multiplied by
/// the length-2047 Kaiser window (`β = 9`, centre 1023, no
/// renormalisation — the DLL's mode 2).
///
/// Fidelity: ≤ 3·10⁻⁷ relative per tap against both retired snapshot banks.
pub fn design_fir_bank(sr: u32) -> Vec<f32> {
    let sr = sr as f32;
    let nyq = sr * 0.5;
    let mut plan = Fft::new(TAPS);
    let mut out = vec![0.0f32; 4 * TAPS];
    for b in 0..4usize {
        // ramp A: centred on the crossover below the band
        let (lo1, hi1) = if b == 0 {
            (-10.0f32, -10.0f32)
        } else {
            let c = XOVERS_HZ[b - 1].ln();
            let l = if b == 1 {
                (XOVERS_HZ[0] * 0.5).ln()
            } else {
                geo_mid(XOVERS_HZ[b - 2], XOVERS_HZ[b - 1])
            };
            let u = if b <= 2 {
                geo_mid(XOVERS_HZ[b - 1], XOVERS_HZ[b])
            } else {
                geo_mid(XOVERS_HZ[2], nyq)
            };
            let h = OVERLAP * (c - l).min(u - c);
            (c - h, c + h)
        };
        // ramp B: centred on the crossover above the band (band 3: Nyquist)
        let (lo2, hi2) = if b == 3 {
            let c = nyq.ln();
            (c, c)
        } else {
            let c = XOVERS_HZ[b].ln();
            let l = if b == 0 {
                (XOVERS_HZ[0] * 0.5).ln()
            } else {
                geo_mid(XOVERS_HZ[b - 1], XOVERS_HZ[b])
            };
            let u = if b <= 1 {
                geo_mid(XOVERS_HZ[b], XOVERS_HZ[b + 1])
            } else {
                geo_mid(XOVERS_HZ[b], nyq)
            };
            let h = OVERLAP * (c - l).min(u - c);
            (c - h, c + h)
        };
        // desired response on the 1025 bins, as the real "cart" half-spectrum
        let mut cart = vec![0.0f32; TAPS + 2];
        for k in 0..=TAPS / 2 {
            let v51 = if k == 0 {
                -10.0f32
            } else {
                ((k as f32 * sr) / TAPS as f32).ln()
            };
            let g = if !(v51 < hi1 || lo2 < v51) {
                1.0f32
            } else if v51 <= lo1 || v51 >= hi2 {
                0.0f32
            } else if hi1 <= v51 {
                let ph =
                    (f64::from(v51 - lo2) * std::f64::consts::PI / f64::from(hi2 - lo2)) as f32;
                (ph.cos() * 0.5) + 0.5
            } else {
                let ph =
                    (f64::from(v51 - lo1) * std::f64::consts::PI / f64::from(hi1 - lo1)) as f32;
                0.5 - (ph.cos() * 0.5)
            };
            cart[2 * k] = g;
        }
        // hermitian c2r inverse FFT of the real spec (the DLL's FFTInv),
        // circular re-index putting the all-pass delta at index 1023.
        let imp = plan.inv(&cart);
        let mut taps = vec![0.0f32; TAPS];
        for (j, t) in taps.iter_mut().enumerate().take(TAPS - 1) {
            *t = imp[(j + TAPS / 2 + 1) % TAPS];
        }
        taps[TAPS - 1] = 0.0;
        // Kaiser window of length 2047 (β = 9), plain f32 product.
        let c = 1023.0f64;
        let i0_beta = bessel_i0(KAISER_BETA);
        for (j, t) in taps.iter_mut().enumerate().take(TAPS - 1) {
            let r = (j as f64 - c) / c;
            let arg = (1.0 - r * r).max(0.0).sqrt();
            let w = (bessel_i0(KAISER_BETA * arg) / i0_beta) as f32;
            *t *= w;
        }
        out[b * TAPS..(b + 1) * TAPS].copy_from_slice(&taps);
    }
    out
}

/// Log-geometric midpoint `(ln a + ln b)/2` of two crossovers, in `f32` —
/// the DLL computes it from two `logf`s, not from `ln√(ab)`.
#[inline]
fn geo_mid(a: f32, b: f32) -> f32 {
    (a.ln() + b.ln()) * 0.5
}

/// Modified Bessel function `I₀(x)` by power series — exact to ~1e-15 for
/// the window's argument range (`x ≤ 9`), matching the DLL's double
/// `BesselI0` well inside f32 rounding.
fn bessel_i0(x: f64) -> f64 {
    let t = x * x * 0.25;
    let mut term = 1.0f64;
    let mut sum = 1.0f64;
    for k in 1..60usize {
        term *= t / (k * k) as f64;
        sum += term;
        if term < 1e-17 * sum {
            break;
        }
    }
    sum
}

/// Peak scale of the band-0 window — measured identical (bit for bit) on both
/// snapshot rates, so treated as a design constant.
const BAND0_SCALE: f32 = 1.979_385_7;

/// The vocoder window bank: `4 × n_write` flattened, band-major.
///
/// Bands 1–3 are plain Hann windows whose **lengths follow the DLL's law**
/// `L_b = 2·⌊r_b·⌈n_write/2⌉⌋` with `r = {0.5, 0.3, 0.2}` (this reproduces
/// both snapshot banks' lengths exactly — {3592, 2154, 1436} at 48 kHz,
/// {3300, 1980, 1320} at 44.1 kHz; a plain `round(r·n_write)` does not).
/// They are centred by `start = (n_write − L)/2` and scaled so
/// `Σw² = 2·⌊n_write/2⌋` (the energy normalisation both snapshots obey).
/// Band 0 is the two-piece asymmetric window: a Hann rise exactly
/// `hop = n_write/2` long, a Hann fall of `≈0.325·n_write`, starting at
/// `≈0.044·n_write`, scaled by the constant [`BAND0_SCALE`].
pub fn win_table(n_write: usize) -> Vec<f32> {
    let nw = n_write;
    let hop_len = (nw + 1) / 2;
    let energy = 2 * (nw / 2);
    let mut out = vec![0.0f32; 4 * nw];
    for (b, r) in [(1usize, 0.5f64), (2, 0.3), (3, 0.2)] {
        let len = 2 * ((r * hop_len as f64).floor() as usize);
        let len = len.min(nw);
        let mut h = vec![0.0f32; len];
        let mut sum_sq = 0.0f64;
        for i in 0..len {
            let ang = 2.0 * std::f64::consts::PI * (i as f64 + 0.5) / len as f64;
            let v = (0.5 - 0.5 * ang.cos()) as f32;
            h[i] = v;
            sum_sq += f64::from(v) * f64::from(v);
        }
        let scale = (energy as f64 / sum_sq).sqrt() as f32;
        let start = nw.saturating_sub(len) / 2;
        for i in 0..len {
            out[b * nw + start + i] = h[i] * scale;
        }
    }
    // Band 0: rise = hop, fall = floor(0.32489·nw), start = floor(0.044·nw)
    // — the constants are pinned to the interval that reproduces BOTH
    // snapshots' fall/start under floor (48k 2334/316, 44k 2143/290).
    let hop = nw / 2;
    let fall = (0.32489 * nw as f64).floor() as usize;
    let start = (0.044 * nw as f64).floor() as usize;
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

    /// Band 0 keeps unity DC gain, band 3 unity Nyquist gain (sign flipped by
    /// the odd all-pass delay). At the snapshot rates the law lands on 1.0 to
    /// 1e-5; far from them the Kaiser window's spectral leakage leaves a few
    /// tenths of a percent on the outermost bins (the DLL's own banks would
    /// show the same — no peak renormalisation happens in mode 2).
    #[test]
    fn designed_bank_gains_are_unity() {
        for sr in [32_000u32, 44_100, 48_000, 96_000] {
            let bank = design_fir_bank(sr);
            let dc: f64 = bank[..TAPS].iter().map(|v| f64::from(*v)).sum();
            let tight = matches!(sr, 44_100 | 48_000);
            let tol = if tight { 1e-3 } else { 6e-3 };
            assert!((dc - 1.0).abs() < tol, "sr={sr}: band0 DC {dc}");
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
            assert!((alt.abs() - 1.0).abs() < tol, "sr={sr}: band3 Nyquist {alt}");
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

