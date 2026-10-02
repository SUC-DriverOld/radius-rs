//! Noise-phase operators and stereo phase synchronization — port of
//! `pyradius/vocoder_ops.py` lines 2363-2558 (`synchronize_stereo_phases`,
//! `adjust_multiphase_diff` body, `ampd_pull_to_peak`) and 3092-3234
//! (`_noise_start_bin`, `randomize_phases`, `substitute_noisy_phases`), which
//! are in turn ports of `libradius/src/ops/synchronize_stereo_phases.c`,
//! `adjust_multiphase_diff.c`, `randomize_phases.c` and
//! `substitute_noisy_phases.c`.
//!
//! ## Float discipline
//!
//! Everything stays `f32`; `a*b + c` is two operations (`-ffp-contract=off`),
//! and every site the reference spells `_fma`/`_fma_arr`/`fmaf` goes through
//! [`fma`] (correctly-rounded `mul_add`).  `_rint` is [`rint`]
//! (`rintf`/`FRINTI`, ties-to-even).  `hi = _fma(ramp, 6, 2)`,
//! `noise_gain = _fma(v14*v14, 6.5, 1)`, the mag factor
//! `_fma(sync_weight, k*0.2, k*0.5) + 1` and the `d`-wrap
//! `_fma(rint(d*inv2pi), -2pi, d)` are all fused in the engine and fused here.
//!
//! `_wrap_pi` is [`wrap_pi_fused`] (the `phase` module's
//! `fma(-rint(x*inv2pi), 2pi, x)`), **not** `vocoder::wrap_pi`: the reference's
//! `_wrap_pi` is `_fma_arr(-rint(x*inv2pi), 2pi, x)`, a single correctly-rounded
//! rounding, while `vocoder::wrap_pi` folds the same expression through `f64`
//! and can in principle differ by one ulp.  `phase.rs` documents the same choice
//! for the same reason, so every wrap in this file uses the fused form.
//!
//! ## API shape
//!
//! `synchronize_stereo_phases` mirrors the Python dict return as a tuple: `dst`
//! comes in by reference, is copied, updated and returned together with
//! `weight` (this is the shape `vocoder::vocoder::ev_sync` calls).  Its Python
//! `nch` argument is dropped because `dst.len()` carries it — the engine always
//! stacks exactly `nch` rows.  `nch > 2` delegates the per-bin pull to
//! [`adjust_multiphase_diff`] (`vocoder::phase`) and the AMPD hook to
//! [`ampd_pull_to_peak`], exactly as the Python does.
//!
//! `randomize_phases`/`substitute_noisy_phases` keep the reference's
//! `[nch][bins]` shape as `&mut [Vec<f32>]` and mutate in place (the C operators
//! are in-place too); they return only what they allocate (`gain_mean`) or
//! compute (`slot`).  numpy-only arguments (`dtype=`, `copy=`) are dropped.
//! Peak/region tables are `i64` (the vocoder state and the reference corpus both
//! carry them that way); the scalar `f372`/`u112`/`slot*` fields stay `i32`, as
//! in the C structs.
//!
//! `synchronize_stereo_phases_nch2_fast` (line 2463) is the same arithmetic with
//! every disjoint peak region flattened into one index set; that is pure
//! scheduling, so this port keeps the straightforward per-peak shape (whose
//! `nch == 2` body is already the elementwise form).  `_fm_h8_opt` (line 2561)
//! belongs to the formant path and is not required by anything here.

use crate::consts::{PI_F, TWO_PI_F};
use crate::simple_rand::SimpleRand;
use crate::vocoder::{adjust_multiphase_diff, ampd_pull_to_peak, wrap_pi_fused};

use super::{fma, rint, EPS1E6_F, INV_2PI_F, K07_F};

/// `NP_INV32767` — `f32(2pi/32767)`, the engine's immediate `0x3949116D`
/// (`mov w22,#0x116d; movk w22,#0x3949,lsl16` @0x16CE34).
///
/// The literal `0.00019175f` is 235 ulp away (`0x39491080`), so the reference
/// forbids it; we construct it from the bit pattern too.
const NP_INV32767: f32 = f32::from_bits(0x3949_116D);

/// `noise_start_bin` @0x16CDB8-0x16CDEC (Python `_noise_start_bin`, line 3100).
///
/// `v = (float)f372 * 150.0f; v /= (float)(uint32)u112;
/// v += (v >= 0 ? 0.5f : -0.5f); start = (int)v` — truncation toward zero
/// (`fcvtzs`), not `rint`.
pub fn noise_start_bin(f372: i32, u112: i32) -> i32 {
    let v = (f372 as f32) * 150.0;
    let v = v / ((u112 as u32) as f32);
    let v = v + if v >= 0.0 { 0.5 } else { -0.5 };
    v as i32
}

/// Clamp a `[start, end)` peak region to `0..=nbins`.
///
/// The C writes `weight_buf[st..en]` and loops `for (i = st; i < en; i++)` with
/// no bounds check, and Python's slice assignment is a no-op for `st >= en`, so
/// this only differs where the C would be undefined.
#[inline]
fn region_bounds(st: i64, en: i64, nbins: usize) -> (usize, usize) {
    let cap = nbins as i64;
    let st = st.max(0).min(cap) as usize;
    let en = en.max(0).min(cap) as usize;
    (st, en.max(st))
}

/// Per-peak sync weight of `synchronize_stereo_phases`: `v9 -> v10 -> v11 -> w`.
///
/// Split out because the `nch == 2` and `nch > 2` branches run their per-bin
/// bodies over differently borrowed buffers.
#[inline]
fn sync_weight_pk(mag: &[Vec<f32>], b: usize, nch: usize, inv: f32) -> f32 {
    // v9 = |m0 - m1| / ((m0 + m1) + 1e-6f), pinned to 0.5 for nch > 2.
    let v9 = if nch > 2 {
        0.5
    } else {
        let m0 = mag[0][b];
        let m1 = mag[1][b];
        (m0 - m1).abs() / ((m0 + m1) + EPS1E6_F)
    };
    let v10 = inv * v9;
    let mut v11 = 0.0f32;
    if v10 > 0.0 {
        v11 = 1.0;
        if v10 < K07_F {
            v11 = v10 / K07_F;
        }
    }
    // sqrtf on the f32 difference: the f64 sqrt of an f32 is correctly rounded
    // and f64 carries the 2p+2 bits f32 (p = 24) needs, so the rounding to f32
    // matches sqrtf bit for bit.
    (1.0f32 - v11).sqrt()
}

/// `rx_synchronize_stereo_phases` @0x16BD64 (Python `synchronize_stereo_phases`,
/// line 2520; the inlined `nch == 2` body of `_adjust_diff_nch2_vec`, line 2441).
///
/// `mag` is the per-channel magnitude, `src` the per-channel mask and `dst` the
/// per-channel phase.  For every peak region `[pk_start, pk_end)` the channels
/// are pulled toward the wrapped circular mean of `dst - src` with weight
/// `w = sqrt(1 - v11)`, and `w` is also stamped into the returned `weight`; bins
/// outside every region keep `weight == 0`.
///
/// Python returns `{"dst", "weight"}` with `dst` a copy of the input; here that
/// is the tuple `(dst_copy, weight)`.  `nch < 2` or `peak_count < 1` returns the
/// untouched copy and an all-zero weight.  The Python `nch` argument is dropped
/// in favour of `dst.len()`; for `nch > 2` the reference reads (and discards)
/// `mag[0][b]`/`mag[1][b]` before pinning `v9 = 0.5` — the C never reads them and
/// neither do we.
///
/// `nch > 2` delegates the per-bin pull to [`adjust_multiphase_diff`] (Python
/// line 2352, ported by `vocoder::phase`) with per-channel rows, exactly the C's
/// `float *const *` arguments.
pub fn synchronize_stereo_phases(
    mag: &[Vec<f32>],
    src: &[Vec<f32>],
    dst: &[Vec<f32>],
    peaks: &[i64],
    pk_start: &[i64],
    pk_end: &[i64],
    peak_count: usize,
    sens: f32,
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let nch = dst.len();
    let nbins = dst.first().map(|r| r.len()).unwrap_or(0);
    let mut out: Vec<Vec<f32>> = dst.to_vec();
    let mut weight = vec![0.0f32; nbins];
    if nch < 2 || peak_count < 1 {
        return (out, weight);
    }
    // inv = 1 / (sens + 1e-6f) — asm 0x16bda8..c0: separate fadd + fdiv.
    let inv = 1.0f32 / (sens + EPS1E6_F);

    if nch == 2 {
        for pk in 0..peak_count {
            let w = sync_weight_pk(mag, peaks[pk] as usize, nch, inv);
            let (st, en) = region_bounds(pk_start[pk], pk_end[pk], nbins);
            weight[st..en].fill(w);
            for bin in st..en {
                let s0 = src[0][bin];
                let s1 = src[1][bin];
                let d0 = out[0][bin];
                let d1 = out[1][bin];
                let a0 = wrap_pi_fused(d0 - s0);
                let a1 = wrap_pi_fused(d1 - s1);
                let mut m = (a0 + a1) * 0.5;
                // `fcsel lt`: |a1-a0| < pi keeps m; NaN or == pi takes +pi.
                if !((a1 - a0).abs() < PI_F) {
                    m = m + PI_F;
                }
                let m = wrap_pi_fused(m);
                let t0 = wrap_pi_fused((s0 + m) - d0);
                out[0][bin] = wrap_pi_fused(fma(w, t0, d0));
                let t1 = wrap_pi_fused((s1 + m) - d1);
                out[1][bin] = wrap_pi_fused(fma(w, t1, d1));
            }
        }
    } else {
        // `nch > 2`: hand the per-channel rows to `phase::adjust_multiphase_diff`
        // (it takes `&[&[f32]]` / `&mut [&mut [f32]]`, bins and nch as i64).
        let src_refs: Vec<&[f32]> = src.iter().map(|r| r.as_slice()).collect();
        let mut dst_refs: Vec<&mut [f32]> = out.iter_mut().map(|r| r.as_mut_slice()).collect();
        for pk in 0..peak_count {
            let w = sync_weight_pk(mag, peaks[pk] as usize, nch, inv);
            let (st, en) = region_bounds(pk_start[pk], pk_end[pk], nbins);
            weight[st..en].fill(w);
            for bin in st..en {
                adjust_multiphase_diff(
                    &src_refs,
                    &mut dst_refs,
                    bin as i64,
                    bin as i64,
                    w,
                    nch as i64,
                );
            }
        }
    }
    (out, weight)
}

/// `rx_randomize_phases` @0x16CE40 (Python `randomize_phases`, line 3108).
///
/// Loop 1 generates the noise phase/gain per bin with the PRNG consumed
/// `bin`-outer / `ch`-inner, loop 2 averages `noise_gain` over the channels and
/// runs the AMPD hook, loop 3 adds the noise phase into `phase` and rescales
/// `mag`.  Python copies and returns all four buffers plus `gain_mean`; the C
/// operator is in place, so the four buffers are `&mut` here and only the fresh
/// `gain_mean` is returned.
///
/// `seed` becomes `rng`: the reference builds `SimpleRand(seed)` per call, but
/// the engine owns one generator seeded once at StartStreaming, so the state has
/// to survive between granules (`&mut SimpleRand::new(seed as u64)` reproduces
/// the Python form exactly).
///
/// `nch == 0` reproduces the reference's `0.0f / 0.0f` NaN `gain_mean` (the C
/// comment is explicit that the NaN must come from a division, not a constant);
/// loops 1 and 3 are gated off for `nch == 0`.
#[allow(clippy::too_many_arguments)]
pub fn randomize_phases(
    noise_phase: &mut [Vec<f32>],
    noise_gain: &mut [Vec<f32>],
    phase: &mut [Vec<f32>],
    mag: &mut [Vec<f32>],
    region_gain: &[Vec<f32>],
    noise_weight: &[f32],
    sync_weight: &[f32],
    a2: f32,
    ramp: f32,
    f372: i32,
    u112: i32,
    rng: &mut SimpleRand,
    nch: usize,
    max_bin: usize,
) -> Vec<f32> {
    let mut gain_mean = vec![0.0f32; max_bin];
    let start = noise_start_bin(f372, u112) as i64;
    // A negative `start` is C out-of-bounds indexing; clamp so the loops are safe.
    let b0 = start.max(0) as usize;

    // ---- loop 1: noise generation (PRNG order: bin outer, channel inner) ----
    if max_bin as i64 > start && nch != 0 {
        let v7 = a2.min(2.0); // fminf(a2, 2.0f) @0x16CE08
        let v8 = ramp;
        let lo = v8 + 1.0;
        let hi = fma(v8, 6.0, 2.0); // fused @0x16CE1C
        let span = hi - lo;
        for bin in b0..max_bin {
            let w0 = noise_weight[bin];
            for ch in 0..nch {
                let g = region_gain[ch][bin];
                let mut t = 0.0f32;
                if !(g <= lo) {
                    t = 1.0;
                    if !(g >= hi) {
                        t = (g - lo) / span;
                    }
                }
                // v14 = (1-t) * (v7*w0): inner product first @0x16CE6C, then outer.
                let v14 = (1.0 - t) * (v7 * w0);
                let n1 = rng.next_u15() as f32;
                let n2 = rng.next_u15() as f32;
                noise_phase[ch][bin] = (v14 * NP_INV32767) * (n1 - n2);
                noise_gain[ch][bin] = fma(v14 * v14, 6.5, 1.0);
            }
        }
    }

    // ---- loop 2: gain mean + per-bin AMPD (nch != 1) ----
    if max_bin as i64 > start {
        for bin in b0..max_bin {
            if nch != 0 {
                let mut ssum = 0.0f32;
                for ch in 0..nch {
                    ssum = ssum + noise_gain[ch][bin];
                }
                gain_mean[bin] = ssum / nch as f32;
                if nch != 1 {
                    // AMPD hook @0x16CF78: (phase[0], noise_phase[0], bin, bin,
                    // sync_weight[bin]).  Both this and the Substitute call site
                    // pass bin == peak_bin, where the pull is numerically a
                    // no-op, but the reference calls it, so we do too.
                    ampd_pull_to_peak(
                        &mut phase[0],
                        &noise_phase[0],
                        bin as i64,
                        bin as i64,
                        sync_weight[bin],
                    );
                }
            } else {
                // 0.0f / 0.0f @0x16CFA0: the reference generates the NaN by
                // division on purpose (never write a NaN constant here).
                let z = 0.0f32;
                gain_mean[bin] = z / z;
            }
        }
    }

    // ---- loop 3: apply (channel outer, bin inner) ----
    if nch != 0 && max_bin as i64 > start {
        for ch in 0..nch {
            for bin in b0..max_bin {
                // Noise phase is read first @0x16D0B0: plain fadd, no wrap.
                phase[ch][bin] = noise_phase[ch][bin] + phase[ch][bin];
                let gain = noise_gain[ch][bin];
                mag[ch][bin] = mag[ch][bin] * fma(sync_weight[bin], gain_mean[bin] - gain, gain);
            }
        }
    }
    gain_mean
}

/// `rx_substitute_noisy_phases` @0x16C540 (Python `substitute_noisy_phases`,
/// line 3175).
///
/// Part A copies one slot of a noise-template row into `noise_phase` (plus the
/// per-bin AMPD hook for `nch >= 2`), Part B blends the noise phase into `phase`
/// with the region ramp and rescales `mag`, and Part C rotates the template slot
/// — unconditionally, also when both gates are shut.
///
/// Python takes `noise_template[0]`/`[1]` out of one list, copies its array
/// arguments and returns `{"phase", "mag", "slot"}`; `tmpl0`/`tmpl1` are separate
/// slices here, `noise_phase`/`phase`/`mag` are in place, and the rotated `slot`
/// is the return value.
///
/// NOTE — template row selection: the Python reference writes
/// `(t1 if (ch & 1) else t0)`, i.e. even channels read `noise_template[0]`.  The
/// C operator (`libradius/src/ops/substitute_noisy_phases.c:46`, asm
/// `mvn w13,w11; and #1` @0x16C528, mirrored by
/// `libradius/tests/test_phase_noise.c:261`) reads `noise_template[1 - (ch & 1)]`,
/// i.e. even channels read `noise_template[1]`.  This port follows the Python
/// reference deliberately (parity is against `pyradius`); the two are
/// indistinguishable in the parity corpus because that harness passes the same
/// row twice (`noise_template=[tmpl, tmpl]`).  Do not "fix" this silently.
#[allow(clippy::too_many_arguments)]
pub fn substitute_noisy_phases(
    noise_phase: &mut [Vec<f32>],
    phase: &mut [Vec<f32>],
    mag: &mut [Vec<f32>],
    region_gain: &[Vec<f32>],
    tmpl0: &[f32],
    tmpl1: &[f32],
    noise_weight: &[f32],
    sync_weight: &[f32],
    a2: f32,
    ramp: f32,
    f372: i32,
    u112: i32,
    slot: i32,
    slot_count: i32,
    tmpl_stride: i32,
    nch: usize,
    max_bin: usize,
) -> i32 {
    let start = noise_start_bin(f372, u112) as i64;
    let b0 = start.max(0) as usize;
    // `off = slot * tmpl_stride` — Python/`int` arithmetic; i64 so the index
    // cannot wrap for large slots.
    let off = (slot as i64) * (tmpl_stride as i64);

    // ---- Part A: template -> noise phase ----
    // Python gates only on `max_bin > start`; for nch == 0 it would index
    // `noise_phase[0]` on an empty array (the C dereferences a null row), so we
    // skip it — Part B is gated off for nch == 0 anyway.
    if max_bin as i64 > start && nch != 0 {
        if nch < 2 {
            for bin in b0..max_bin {
                noise_phase[0][bin] = tmpl0[(bin as i64 + off) as usize];
            }
        } else {
            for bin in b0..max_bin {
                for ch in 0..nch {
                    let t = if (ch & 1) == 1 { tmpl1 } else { tmpl0 };
                    noise_phase[ch][bin] = t[(bin as i64 + off) as usize];
                }
                // One AMPD call per bin @0x16C550, on the row just written.
                ampd_pull_to_peak(
                    &mut phase[0],
                    &noise_phase[0],
                    bin as i64,
                    bin as i64,
                    sync_weight[bin],
                );
            }
        }
    }

    // ---- Part B: gain ramp blend ----
    if nch != 0 && max_bin as i64 > start {
        let v8 = ramp;
        let lo = v8 + 1.0;
        let hi = fma(v8, 6.0, 2.0); // fused @0x16C7E8
        for bin in b0..max_bin {
            let wgt = noise_weight[bin];
            let awgt = a2 * wgt; // fmul @0x16C7CC
            for ch in 0..nch {
                // ratio = region_gain / (a2 * noise_weight): fdiv @0x16C7D0.
                let ratio = region_gain[ch][bin] / awgt;
                let mut t = 0.0f32;
                if !(ratio <= lo) {
                    t = 1.0;
                    if !(ratio >= hi) {
                        t = (ratio - lo) / (hi - lo);
                    }
                }
                let k = 1.0 - t;
                let d = noise_phase[ch][bin] - phase[ch][bin];
                // wrap = fmaf(rintf(d*inv2pi), -2pi, d) — single-rounding fmsub.
                let wrap = fma(rint(d * INV_2PI_F), -TWO_PI_F, d);
                phase[ch][bin] = fma(k, wrap, phase[ch][bin]);
                let magfac = if nch == 1 {
                    // (k*0.5f) + 1.0f: two separately rounded ops @0x16C7A8/AC.
                    (k * 0.5) + 1.0
                } else {
                    // fmaf(sync_w, k*0.2f, k*0.5f) + 1.0f @0x16CB6C-78.
                    fma(sync_weight[bin], k * 0.2, k * 0.5) + 1.0
                };
                mag[ch][bin] = magfac * mag[ch][bin];
            }
        }
    }

    // ---- Part C: slot rotation (unconditional, @0x16CD40) ----
    if slot + 1 < slot_count {
        slot + 1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny deterministic generator for the synthetic fixtures (no deps).
    struct Lcg(u32);

    impl Lcg {
        fn unit(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((self.0 >> 8) as f32) / 16_777_216.0
        }

        /// Symmetric angle source in [-pi, pi).
        fn angle(&mut self) -> f32 {
            (self.unit() * 2.0 - 1.0) * PI_F
        }
    }

    fn rows(nch: usize, n: usize, mut f: impl FnMut(usize, usize) -> f32) -> Vec<Vec<f32>> {
        (0..nch)
            .map(|c| (0..n).map(|i| f(c, i)).collect())
            .collect()
    }

    // ------------------------------------------------------------------
    // noise_start_bin
    // ------------------------------------------------------------------

    #[test]
    fn noise_start_bin_matches_closed_form() {
        // v = f372*150/u112, += 0.5 (or -0.5), then truncate toward zero.
        assert_eq!(noise_start_bin(8, 512), 2); // 2.34375 + .5 = 2.84375
        assert_eq!(noise_start_bin(4, 128), 5); // 4.6875 + .5 = 5.1875
        assert_eq!(noise_start_bin(12, 1023), 2); // 1.7595 + .5 = 2.2595
        assert_eq!(noise_start_bin(0, 512), 0); // 0 + .5 -> 0
        assert_eq!(noise_start_bin(-10, 512), -3); // -2.9297 - .5 -> -3 (toward 0)
        assert_eq!(noise_start_bin(-1, 512), 0); // -0.7930 -> 0
    }

    // ------------------------------------------------------------------
    // synchronize_stereo_phases
    // ------------------------------------------------------------------

    #[test]
    fn sync_gate_leaves_dst_untouched() {
        let peaks = [1i64];
        let st = [0i64];
        let en = [8i64];

        let mag = rows(2, 8, |_, _| 1.0);
        let src = rows(2, 8, |_, i| i as f32 * 0.1);
        let before = rows(2, 8, |c, i| (c + i) as f32 * 0.25);
        let (out, w) = synchronize_stereo_phases(&mag, &src, &before, &peaks, &st, &en, 0, 1.0);
        assert_eq!(w, vec![0.0f32; 8]);
        assert_eq!(out, before);

        // nch == 1 (one row) is also gated off.
        let mag1 = rows(1, 8, |_, _| 1.0);
        let src1 = rows(1, 8, |_, i| i as f32 * 0.1);
        let before1 = rows(1, 8, |_, i| i as f32 * 0.25);
        let (out, w) = synchronize_stereo_phases(&mag1, &src1, &before1, &peaks, &st, &en, 1, 1.0);
        assert_eq!(w, vec![0.0f32; 8]);
        assert_eq!(out, before1);
    }

    #[test]
    fn sync_nch2_exact_case() {
        // sens = inf -> inv = 0 -> v10 = 0 -> v11 = 0 -> w = sqrt(1) = 1 exactly.
        // d0 = 0.5, s0 = 0, d1 = -0.5, s1 = 0: a0 = 0.5, a1 = -0.5, m = 0,
        // t0 = -0.5, t1 = 0.5 -> both channels land on exactly 0.
        let mag = vec![vec![2.0f32; 4], vec![1.0f32; 4]];
        let src = vec![vec![0.0f32; 4], vec![0.0f32; 4]];
        let dst_in = vec![vec![0.5f32; 4], vec![-0.5f32; 4]];
        let peaks = [0i64, 3];
        let st = [0i64, 2];
        let en = [2i64, 4];

        let (dst, w) =
            synchronize_stereo_phases(&mag, &src, &dst_in, &peaks, &st, &en, 2, f32::INFINITY);
        assert_eq!(w, vec![1.0f32, 1.0, 1.0, 1.0]);
        // The input is a copy source, never mutated.
        assert_eq!(dst_in[0], vec![0.5f32; 4]);
        assert_eq!(dst[0], vec![0.0f32; 4]);
        assert_eq!(dst[1], vec![0.0f32; 4]);
    }

    #[test]
    fn sync_nch2_weight_only_inside_regions() {
        let mag = vec![vec![1.0f32; 8], vec![0.0f32; 8]]; // v9 = 1/(1+1e-6)
        let src = rows(2, 8, |_, _| 0.0);
        let dst_in = rows(2, 8, |c, i| (c as f32 + 1.0) * 0.1 * i as f32);
        let peaks = [2i64];
        let st = [3i64];
        let en = [6i64];
        let (dst, w) = synchronize_stereo_phases(&mag, &src, &dst_in, &peaks, &st, &en, 1, 2.0);

        // inv = 1/(2+1e-6), v9 = 1/((1+0)+1e-6), v10 = inv*v9 = 0.4999994 < 0.7
        // -> v11 = v10/0.7 -> w = sqrt(1 - v11) ~ 0.5345.
        let v9 = 1.0f32 / (1.0f32 + EPS1E6_F);
        let v10 = (1.0f32 / (2.0f32 + EPS1E6_F)) * v9;
        let want_w = (1.0f32 - v10 / K07_F).sqrt();
        assert!(want_w > 0.0 && want_w < 1.0, "w = {want_w}");
        assert_eq!(w[..3], [0.0f32; 3]);
        assert_eq!(w[3..6], [want_w; 3]);
        assert_eq!(w[6..], [0.0f32; 2]);

        // Only bins inside the region move.
        for bin in 0..3 {
            assert_eq!(dst[0][bin], dst_in[0][bin]);
            assert_eq!(dst[1][bin], dst_in[1][bin]);
        }
        for bin in 6..8 {
            assert_eq!(dst[0][bin], dst_in[0][bin]);
            assert_eq!(dst[1][bin], dst_in[1][bin]);
        }
        let moved = (0..2).any(|c| (3..6).any(|b| dst[c][b] != dst_in[c][b]));
        assert!(moved, "the region must be modified when w > 0");
    }

    #[test]
    fn sync_nch2_zero_weight_leaves_dst_exactly() {
        // v9 = 1/(1+1e-6), inv = 1/(1+1e-6) -> v10 ~ 1 > 0.7 -> v11 = 1 -> w = 0.
        let mag = vec![vec![1.0f32; 8], vec![0.0f32; 8]];
        let src = rows(2, 8, |_, _| 0.0);
        let dst_in = rows(2, 8, |c, i| (c as f32 + 1.0) * 0.1 * i as f32);
        let peaks = [2i64];
        let st = [3i64];
        let en = [6i64];
        let (dst, w) = synchronize_stereo_phases(&mag, &src, &dst_in, &peaks, &st, &en, 1, 1.0);
        assert_eq!(w, vec![0.0f32; 8]);
        // fma(0, t, d) == d exactly, so nothing moves anywhere.
        assert_eq!(dst, dst_in);
    }

    /// The whole-region `nch == 2` pass must equal an independent per-bin
    /// transcription of the same C body (Python `_adjust_diff_nch2_vec` vs the
    /// scalar per-bin loop it replaces).
    #[test]
    fn sync_nch2_vec_matches_per_bin_scalar() {
        let n = 37usize;
        let mut r = Lcg(0x1234_5678);
        let mag = rows(2, n, |_, _| 0.2 + r.unit());
        let src = rows(2, n, |_, _| r.angle());
        let dst_in = rows(2, n, |_, _| r.angle());
        let peaks = [1i64, 5, 12, 30, 36];
        let st = [0i64, 4, 9, 25, 36];
        let en = [4i64, 9, 25, 33, 37];
        let peak_count = peaks.len();
        let sens = 0.37f32;

        let (dst, w) =
            synchronize_stereo_phases(&mag, &src, &dst_in, &peaks, &st, &en, peak_count, sens);

        // Independent scalar reference: same formulas, one bin at a time.
        let inv = 1.0f32 / (sens + EPS1E6_F);
        let mut want = dst_in.clone();
        let mut want_w = vec![0.0f32; n];
        for pk in 0..peak_count {
            let b = peaks[pk] as usize;
            let m0 = mag[0][b];
            let m1 = mag[1][b];
            let v9 = (m0 - m1).abs() / ((m0 + m1) + EPS1E6_F);
            let v10 = inv * v9;
            let mut v11 = 0.0f32;
            if v10 > 0.0 {
                v11 = 1.0;
                if v10 < K07_F {
                    v11 = v10 / K07_F;
                }
            }
            let ww = (1.0f32 - v11).sqrt();
            for bin in (st[pk] as usize)..(en[pk] as usize) {
                want_w[bin] = ww;
                let s0 = src[0][bin];
                let s1 = src[1][bin];
                let d0 = want[0][bin];
                let d1 = want[1][bin];
                let a0 = wrap_pi_fused(d0 - s0);
                let a1 = wrap_pi_fused(d1 - s1);
                let mut m = (a0 + a1) * 0.5;
                if !((a1 - a0).abs() < PI_F) {
                    m = m + PI_F;
                }
                let m = wrap_pi_fused(m);
                let t0 = wrap_pi_fused((s0 + m) - d0);
                want[0][bin] = wrap_pi_fused(fma(ww, t0, d0));
                let t1 = wrap_pi_fused((s1 + m) - d1);
                want[1][bin] = wrap_pi_fused(fma(ww, t1, d1));
            }
        }
        assert_eq!(w, want_w);
        assert_eq!(dst, want);
    }

    /// With `w == 1` the two channels end up separated by the mask difference
    /// (mod 2pi), which is what the operator is for.
    #[test]
    fn sync_nch2_full_weight_aligns_to_mask_difference() {
        let n = 64usize;
        let mut r = Lcg(0xDEAD_BEEF);
        let mag = vec![vec![4.0f32; n], vec![1.0f32; n]];
        let src = rows(2, n, |c, i| {
            (c as f32 + 1.0) * 0.05 * i as f32 + r.angle() * 0.1
        });
        let dst_in = rows(2, n, |_, _| r.angle());
        let peaks = [0i64, 20, 40];
        let st = [0i64, 16, 32];
        let en = [16i64, 32, n as i64];
        let (dst, w) = synchronize_stereo_phases(&mag, &src, &dst_in, &peaks, &st, &en, 3, 1.0e9);
        for bin in 0..n {
            assert!(w[bin] > 0.999, "bin {bin}: w = {}", w[bin]);
            let got = wrap_pi_fused(dst[0][bin] - dst[1][bin]);
            let want = wrap_pi_fused(src[0][bin] - src[1][bin]);
            assert!(
                (got - want).abs() < 1e-4,
                "bin {bin}: rel diff {got} vs {want}"
            );
        }
    }

    // ------------------------------------------------------------------
    // randomize_phases
    // ------------------------------------------------------------------

    #[test]
    fn randomize_phases_finite_and_bounded() {
        let nch = 2usize;
        let max_bin = 32usize;
        let mut r = Lcg(0x0BAD_F00D);

        let mut noise_phase = rows(nch, max_bin, |_, _| 0.0);
        let mut noise_gain = rows(nch, max_bin, |_, _| 1.0);
        let mut phase = rows(nch, max_bin, |_, _| r.angle());
        let mut mag = rows(nch, max_bin, |_, _| 0.5 + r.unit());
        // region_gain ramps across the run so all three ramp branches fire.
        let region_gain = rows(nch, max_bin, |c, i| (i as f32) * 0.25 + (c as f32) * 0.1);
        let noise_weight = vec![1.0f32; max_bin];
        let sync_weight = vec![0.5f32; max_bin];

        let a2 = 0.5f32;
        let ramp = 0.2f32;
        let mut rng = SimpleRand::new(1);
        let gain_mean = randomize_phases(
            &mut noise_phase,
            &mut noise_gain,
            &mut phase,
            &mut mag,
            &region_gain,
            &noise_weight,
            &sync_weight,
            a2,
            ramp,
            8,
            512, // start = 2
            &mut rng,
            nch,
            max_bin,
        );

        for ch in 0..nch {
            assert_eq!(noise_phase[ch].len(), max_bin);
            assert_eq!(noise_gain[ch].len(), max_bin);
            assert_eq!(phase[ch].len(), max_bin);
            assert_eq!(mag[ch].len(), max_bin);
            for bin in 0..max_bin {
                let np = noise_phase[ch][bin];
                let ng = noise_gain[ch][bin];
                let ph = phase[ch][bin];
                let mg = mag[ch][bin];
                assert!(np.is_finite() && ng.is_finite() && ph.is_finite() && mg.is_finite());
                if bin >= 2 {
                    // |v14| <= min(a2, 2) = 0.5 -> |noise phase| <= 0.5 * 2pi.
                    assert!(np.abs() <= PI_F + 1e-3, "noise_phase {np} at {ch}/{bin}");
                    // noise_gain = fma(v14^2, 6.5, 1) in [1, 1 + 6.5*0.25].
                    assert!((1.0..=2.6251).contains(&ng), "noise_gain {ng}");
                    // mag *= fma(0.5, mean - gain, gain) is a 50/50 blend of
                    // gain_mean and gain, so at most 2.625 * 1.5.
                    assert!(mg > 0.0 && mg <= 3.9376, "mag {mg}");
                } else {
                    // Below `start` nothing is injected.
                    assert_eq!(np, 0.0);
                    assert_eq!(ng, 1.0);
                }
            }
        }
        assert_eq!(gain_mean.len(), max_bin);
        assert!(gain_mean[..2].iter().all(|v| *v == 0.0));
        for bin in 2..max_bin {
            assert!(gain_mean[bin].is_finite());
            assert!((1.0..=2.6251).contains(&gain_mean[bin]));
        }
        // The mean is shared by every channel of a bin: recompute it by hand.
        for bin in 2..max_bin {
            let want = ((noise_gain[0][bin] + noise_gain[1][bin]) / 2.0f32) as f32;
            assert_eq!(gain_mean[bin], want);
        }
    }

    #[test]
    fn randomize_phases_nch0_gain_mean_is_nan() {
        let max_bin = 8usize;
        let mut noise_phase: Vec<Vec<f32>> = Vec::new();
        let mut noise_gain: Vec<Vec<f32>> = Vec::new();
        let mut phase: Vec<Vec<f32>> = Vec::new();
        let mut mag: Vec<Vec<f32>> = Vec::new();
        let region_gain: Vec<Vec<f32>> = Vec::new();
        let mut rng = SimpleRand::new(1);
        let gain_mean = randomize_phases(
            &mut noise_phase,
            &mut noise_gain,
            &mut phase,
            &mut mag,
            &region_gain,
            &[],
            &[],
            0.5,
            0.2,
            8,
            512,
            &mut rng,
            0,
            max_bin,
        );
        assert_eq!(gain_mean.len(), max_bin);
        assert!(gain_mean[..2].iter().all(|v| *v == 0.0));
        assert!(gain_mean[2..].iter().all(|v| v.is_nan()));
    }

    /// Same inputs, two generators started from the same seed -> identical
    /// output, and the PRNG state advances by exactly the draws the reference
    /// makes (2 per channel per bin in `[start, max_bin)`).
    #[test]
    fn randomize_phases_is_deterministic_and_consumes_rng() {
        let max_bin = 16usize;
        let run = |rng: &mut SimpleRand| {
            let mut noise_phase = rows(2, max_bin, |_, _| 0.0);
            let mut noise_gain = rows(2, max_bin, |_, _| 0.0);
            let mut phase = rows(2, max_bin, |_, _| 0.0);
            let mut mag = rows(2, max_bin, |_, _| 1.0);
            let region_gain = rows(2, max_bin, |_, _| 1.0);
            let gain_mean = randomize_phases(
                &mut noise_phase,
                &mut noise_gain,
                &mut phase,
                &mut mag,
                &region_gain,
                &vec![1.0; max_bin],
                &vec![1.0; max_bin],
                0.5,
                0.2,
                8,
                512,
                rng,
                2,
                max_bin,
            );
            (noise_phase, noise_gain, phase, mag, gain_mean)
        };
        let mut a = SimpleRand::new(1);
        let mut b = SimpleRand::new(1);
        let ra = run(&mut a);
        let rb = run(&mut b);
        assert_eq!(ra.0, rb.0);
        assert_eq!(ra.1, rb.1);
        assert_eq!(ra.2, rb.2);
        assert_eq!(ra.3, rb.3);
        assert_eq!(ra.4, rb.4);
        assert_ne!(a.state, 1, "the PRNG state must advance");
        let mut want = SimpleRand::new(1);
        want.skip((2 * (max_bin - 2) * 2) as u64);
        assert_eq!(a.state, want.state);
    }

    // ------------------------------------------------------------------
    // substitute_noisy_phases
    // ------------------------------------------------------------------

    #[test]
    fn substitute_noisy_phases_preserves_lengths() {
        let nch = 3usize;
        let max_bin = 16usize;
        let slot_count = 4i32;
        let tmpl_stride = max_bin as i32;
        let tmpl_len = slot_count as usize * max_bin;

        let mut r = Lcg(0x5EED_1234);
        let tmpl0: Vec<f32> = (0..tmpl_len).map(|_| r.angle()).collect();
        let tmpl1: Vec<f32> = (0..tmpl_len).map(|_| r.angle()).collect();

        let mut noise_phase = rows(nch, max_bin, |_, _| 0.0);
        let mut phase = rows(nch, max_bin, |_, _| r.angle());
        let mut mag = rows(nch, max_bin, |_, _| 0.5 + r.unit());
        let region_gain = rows(nch, max_bin, |c, i| {
            0.5 + 0.05 * (i as f32) + 0.01 * c as f32
        });
        let noise_weight = vec![1.0f32; max_bin];
        let sync_weight = vec![0.5f32; max_bin];

        let before: Vec<Vec<f32>> = phase.clone();
        let mag_before: Vec<Vec<f32>> = mag.clone();

        let slot = substitute_noisy_phases(
            &mut noise_phase,
            &mut phase,
            &mut mag,
            &region_gain,
            &tmpl0,
            &tmpl1,
            &noise_weight,
            &sync_weight,
            0.5,
            0.2,
            8,
            512, // start = 2
            1,
            slot_count,
            tmpl_stride,
            nch,
            max_bin,
        );

        // Part C: slot 1 -> 2.
        assert_eq!(slot, 2);

        // Every buffer the operator touches keeps its length.
        assert_eq!(noise_phase.len(), nch);
        assert_eq!(phase.len(), nch);
        assert_eq!(mag.len(), nch);
        for ch in 0..nch {
            assert_eq!(noise_phase[ch].len(), max_bin);
            assert_eq!(phase[ch].len(), max_bin);
            assert_eq!(mag[ch].len(), max_bin);
            for bin in 0..max_bin {
                assert!(noise_phase[ch][bin].is_finite());
                assert!(phase[ch][bin].is_finite());
                assert!(mag[ch][bin].is_finite() && mag[ch][bin] > 0.0);
            }
        }

        // Part A wrote the template row for bins >= start only; even channels
        // read tmpl[0] and odd channels tmpl[1] (the Python reference; see the
        // doc comment for the C discrepancy).
        for bin in 2..max_bin {
            let idx = bin + tmpl_stride as usize;
            assert_eq!(noise_phase[0][bin], tmpl0[idx]);
            assert_eq!(noise_phase[1][bin], tmpl1[idx]);
            assert_eq!(noise_phase[2][bin], tmpl0[idx]);
            assert_ne!(tmpl0[idx], tmpl1[idx]);
        }
        for ch in 0..nch {
            assert_eq!(noise_phase[ch][..2], [0.0f32, 0.0]);
        }
        // Part B really touched phase/mag inside the region.
        assert_ne!(phase[0][5], before[0][5]);
        assert_ne!(mag[0][5], mag_before[0][5]);

        // Successive calls rotate 2 -> 3 -> 0 -> 1.
        let mut slot_state = slot;
        for want in [3i32, 0, 1] {
            slot_state = substitute_noisy_phases(
                &mut noise_phase,
                &mut phase,
                &mut mag,
                &region_gain,
                &tmpl0,
                &tmpl1,
                &noise_weight,
                &sync_weight,
                0.5,
                0.2,
                8,
                512,
                slot_state,
                slot_count,
                tmpl_stride,
                nch,
                max_bin,
            );
            assert_eq!(slot_state, want);
        }
    }

    #[test]
    fn substitute_noisy_phases_closed_gate_still_rotates_slot() {
        // max_bin = 0 shuts both `max_bin > start` gates while Part C still runs.
        let nch = 2usize;
        let mut noise_phase = rows(nch, 0, |_, _| 0.0);
        let mut phase = rows(nch, 0, |_, _| 0.0);
        let mut mag = rows(nch, 0, |_, _| 0.0);
        let region_gain: Vec<Vec<f32>> = rows(nch, 0, |_, _| 0.0);
        let slot = substitute_noisy_phases(
            &mut noise_phase,
            &mut phase,
            &mut mag,
            &region_gain,
            &[],
            &[],
            &[],
            &[],
            0.5,
            0.2,
            8,
            512,
            2,
            3,
            0,
            nch,
            0,
        );
        assert_eq!(slot, 0); // 2 + 1 == 3 is not < 3 -> back to 0
        assert_eq!(noise_phase.len(), nch);
        assert!(noise_phase.iter().all(|r| r.is_empty()));
    }

    // ------------------------------------------------------------------
    // golden vectors
    // ------------------------------------------------------------------

    /// Flatten `[nch][n]` rows into bit patterns (row-major).
    fn flat(v: &[Vec<f32>]) -> Vec<u32> {
        v.iter().flatten().map(|x| x.to_bits()).collect()
    }

    /// `[nch][n]` rows from a flat bit-pattern table (row-major).
    fn golden_rows(flat_bits: &[u32], nch: usize, n: usize) -> Vec<Vec<f32>> {
        (0..nch)
            .map(|c| {
                (0..n)
                    .map(|i| f32::from_bits(flat_bits[c * n + i]))
                    .collect()
            })
            .collect()
    }

    fn golden_flat(flat_bits: &[u32]) -> Vec<f32> {
        flat_bits.iter().map(|u| f32::from_bits(*u)).collect()
    }

    // Generated by `pyradius.vocoder_ops` (see .tmp/noise_check/gen_golden.py):
    // the reference is on route A (`_route_b()` is False) with the exact numba
    // `_fma_arr`/`_wrap_pi` kernels, i.e. the same single-rounding fmaf the C
    // engine has with `-ffp-contract=off`, so every comparison below is
    // bit-exact.  `RND_NP_OUT` even pins a negative zero.
    // ================= generated golden vectors =================
    const RND_NP_IN: [u32; 16] = [
        0x00000000, 0x3E45E757, 0x3E975D76, 0x3E8496C6, 0x3DCDD115, 0xBDD7856E, 0xBE85DFC1,
        0xBE96E79C, 0xBE41ECD4, 0x3BA54991, 0x3E49D387, 0x3E97C85B, 0x3E834432, 0x3DC40DD5,
        0xBDE12A2F, 0xBE871F0B,
    ];
    const RND_NG_IN: [u32; 16] = [
        0x3FC00000, 0x3FA29450, 0x3F4ABBB3, 0x3F0147ED, 0x3F2C5568, 0x3F922785, 0x3FBD736E,
        0x3FB03FEF, 0x3F6D6041, 0x3F0B6015, 0x3F14994E, 0x3F804883, 0x3FB601B4, 0x3FBA139C,
        0x3F88C04D, 0x3F1EC28C,
    ];
    const RND_PH_IN: [u32; 16] = [
        0x00000000, 0x411A2B58, 0x40A4F5DF, 0xC0DC15CE, 0xC10D5A4D, 0x4009AD42, 0x411FC456,
        0x404C3913, 0xC10473C6, 0xC0F3D5B2, 0x40867415, 0x411DE22E, 0x3F89ECB7, 0xC114A8EB,
        0xC0C18BE8, 0x40C1C5D3,
    ];
    const RND_MAG_IN: [u32; 16] = [
        0x3FB33333, 0x3FAD1944, 0x3F9B605F, 0x3F7F720C, 0x3F398C40, 0x3ED7726B, 0x3F2F688D,
        0x3F76C7BA, 0x3F9831A3, 0x3FAB5EA1, 0x3FB316DC, 0x3FAE9DED, 0x3F9E60F9, 0x3F83EC4D,
        0x3F438B21, 0x3EECB8F2,
    ];
    const RND_RG: [u32; 16] = [
        0x00000000, 0x3F800000, 0x3F99999A, 0x3FC00000, 0x40000000, 0x40200000, 0x404CCCCD,
        0x41100000, 0x41100000, 0x404CCCCD, 0x40200000, 0x40000000, 0x3FC00000, 0x3F99999A,
        0x3F800000, 0x00000000,
    ];
    const RND_NW: [u32; 8] = [
        0x3F800000, 0x3E800000, 0x3F000000, 0x3F400000, 0x3F800000, 0x3F666666, 0x3DCCCCCD,
        0x3F19999A,
    ];
    const RND_SW: [u32; 8] = [
        0x00000000, 0x3E800000, 0x3F000000, 0x3F800000, 0x3F400000, 0x3E99999A, 0x3F666666,
        0x3E4CCCCD,
    ];
    const RND_NP_OUT: [u32; 16] = [
        0x00000000, 0x3E45E757, 0xBF299891, 0x3E21B66D, 0x3DDCB928, 0x3E15B281, 0x80000000,
        0x00000000, 0xBE41ECD4, 0x3BA54991, 0xBE81ED88, 0xBF141D0A, 0xBFAF8F6E, 0x3F1CE4EA,
        0xBB8F7F23, 0x3E25A17D,
    ];
    const RND_NG_OUT: [u32; 16] = [
        0x3FC00000, 0x3FA29450, 0x3F9D4000, 0x3FAF8CAE, 0x3FAA1EB9, 0x3F8B9BFD, 0x3F800000,
        0x3F800000, 0x3F6D6041, 0x3F0B6015, 0x3F839548, 0x3F97B148, 0x3FD48853, 0x3FDEC51E,
        0x3F812B85, 0x3FAA1EB9,
    ];
    const RND_PH_OUT: [u32; 16] = [
        0x00000000, 0x411A2B58, 0x408FC2CD, 0xC0D7081B, 0xC10BA0DB, 0x4013086A, 0x411FC456,
        0x404C3913, 0xC10473C6, 0xC0F3D5B2, 0x407CAA79, 0x4114A05D, 0xBE968ADC, 0xC10ADA9C,
        0xC0C1AFC8, 0x40C6F2DF,
    ];
    const RND_MAG_OUT: [u32; 16] = [
        0x3FB33333, 0x3FAD1944, 0x3FB717E1, 0x3FA34440, 0x3F86D47C, 0x3EFFFBD3, 0x3F302141,
        0x3F7EE69B, 0x3F9831A3, 0x3FAB5EA1, 0x3FC11497, 0x3FDF35FD, 0x3FF34B5B, 0x3FD8BDC1,
        0x3F4486CA, 0x3F196A26,
    ];
    const RND_GM_OUT: [u32; 8] = [
        0x00000000, 0x00000000, 0x3F906AA4, 0x3FA39EFB, 0x3FBF5386, 0x3FB5308E, 0x3F8095C2,
        0x3F950F5C,
    ];
    const RND_A2_BITS: u32 = 0x3EC00000;
    const RND_RAMP_BITS: u32 = 0x3E4CCCCD;
    const RND_SEED: u64 = 1;
    const RND_F372: i32 = 8;
    const RND_U112: i32 = 512;
    const SUB_NP_IN: [u32; 16] = [
        0x00000000, 0x3F966616, 0x3FBAFA8F, 0x3F241D27, 0xBF29ED88, 0xBFBBAF8F, 0xBF945EEC,
        0x3CCE9BF5, 0x3F98625D, 0x3FBA3806, 0x3F1E40E5, 0xBF2FB19C, 0xBFBC56FA, 0xBF924D04,
        0x3D4E947B, 0x3F9A539D,
    ];
    const SUB_PH_IN: [u32; 16] = [
        0x40C00000, 0xBF45E7B7, 0xC0B9A011, 0x401124B9, 0x40A6ECA0, 0xC0672C2E, 0xC0892396,
        0x4096ECED, 0x40447E8D, 0xC0B03E22, 0xBFD35350, 0x40BDDB55, 0x3DFA1238, 0xC0BEDD18,
        0x3FB51ABA, 0x40B3324E,
    ];
    const SUB_MAG_IN: [u32; 16] = [
        0x3E99999A, 0x3F4E376F, 0x3F960FBF, 0x3FA6605C, 0x3F93938E, 0x3F45A3CB, 0x3EAD427D,
        0x3F569A35, 0x3F9861C6, 0x3FA63009, 0x3F90EE24, 0x3F3CE287, 0x3EC0E3F5, 0x3F5EC8F3,
        0x3F9A88C4, 0x3FA5CF77,
    ];
    const SUB_RG: [u32; 16] = [
        0x00000000, 0x3ECCCCCD, 0x3EF5C290, 0x3F19999A, 0x3F4CCCCD, 0x3F800000, 0x3FA3D70B,
        0x40666667, 0x40666667, 0x3FA3D70B, 0x3F800000, 0x3F4CCCCD, 0x3F19999A, 0x3EF5C290,
        0x3ECCCCCD, 0x00000000,
    ];
    const SUB_T0: [u32; 32] = [
        0x00000000, 0x3F4C1686, 0x3FBB2C60, 0x3FF146F2, 0x3FFF6322, 0x3FE32A2E, 0x3FA149B0,
        0x3F095A2B, 0xBE8D46ED, 0xBF8575CE, 0xBFD17A9C, 0xBFFAC65B, 0xBFFA80B0, 0xBFD0B528,
        0xBF84514B, 0xBE87FA9A, 0x3F0BED0F, 0x3FA252E4, 0x3FE3C72E, 0x3FFF79E8, 0x3FF0D3B8,
        0x3FBA423E, 0x3F49A214, 0xBC2B1D71, 0xBF4E898B, 0xBFBC1532, 0xBFF1B87D, 0xBFFF4A93,
        0xBFE28B98, 0xBFA03F5C, 0xBF06C651, 0x3E929243,
    ];
    const SUB_T1: [u32; 32] = [
        0x40000000, 0x3FF94233, 0x3FE563B2, 0x3FC57061, 0x3F9B1708, 0x3F512544, 0x3EC23135,
        0xBDA08980, 0xBF082C6B, 0xBF751BA5, 0xBFAA9111, 0xBFD19897, 0xBFED962C, 0xBFFD106C,
        0xBFFF36A8, 0xBFF3EBE5, 0xBFDBC863, 0xBFB81198, 0xBF8AA909, 0xBF2BE5E6, 0xBE65B039,
        0x3E704FFD, 0x3F2E6A07, 0x3F8BC83C, 0x3FB8FECC, 0x3FDC771B, 0x3FF452ED, 0x3FFF5092,
        0x3FFCDBDB, 0x3FED15E6, 0x3FD0D35C, 0x3FA99144,
    ];
    const SUB_NW: [u32; 8] = [
        0x3F000000, 0x3F000000, 0x3F000000, 0x3F000000, 0x3F000000, 0x3F000000, 0x3F000000,
        0x3F000000,
    ];
    const SUB_SW: [u32; 8] = [
        0x00000000, 0x3E4CCCCD, 0x3ECCCCCD, 0x3F19999A, 0x3F4CCCCD, 0x3F800000, 0x3F000000,
        0x3E800000,
    ];
    const SUB_PH_OUT: [u32; 16] = [
        0x40C00000, 0xBF45E7B7, 0xC0CF52C4, 0x401124B9, 0x40A6ECA0, 0xC0672C2E, 0xC0892396,
        0x4096ECED, 0x40447E8D, 0xC0B03E22, 0xBFD35350, 0x40BDDB55, 0x3DFA1238, 0xC0D65F4A,
        0x403CAAB5, 0x408C14E2,
    ];
    const SUB_MAG_OUT: [u32; 16] = [
        0x3E99999A, 0x3F4E376F, 0x3FB1E9B5, 0x3FA6605C, 0x3F93938E, 0x3F45A3CB, 0x3EAD427D,
        0x3F569A35, 0x3F9861C6, 0x3FA63009, 0x3F90EE24, 0x3F3CE287, 0x3EC0E3F5, 0x3F88582A,
        0x3FCBFC35, 0x400080C9,
    ];
    const SUB_SLOT_IN: i32 = 1;
    const SUB_SLOT_COUNT: i32 = 4;
    const SUB_STRIDE: i32 = 8;
    const SUB_SLOT_OUT: i32 = 2;
    const SYNC_MAG_IN: [u32; 16] = [
        0x3F000000, 0x3F884634, 0x3FB74D0F, 0x3FBCA70A, 0x3F967594, 0x3F242071, 0x3F714905,
        0x3FAF8FCB, 0x3FC00000, 0x3F8F90EA, 0x3F3A29EB, 0x3FB3B8A3, 0x3FB2C8FB, 0x3F35F6B7,
        0x3F913D9E, 0x3FBFFB5E,
    ];
    const SYNC_SRC_IN: [u32; 16] = [
        0x00000000, 0x402B1CA0, 0x401B3B37, 0xBEF24C22, 0xC036B525, 0xC00776B8, 0x3F6F4386,
        0x403DBA72, 0x40400000, 0x4012D986, 0x3F0288E6, 0xBFC1DC64, 0xC034E81D, 0xC033CCB8,
        0xBFBC429A, 0x3F0F3DD3,
    ];
    const SYNC_DST_IN: [u32; 16] = [
        0x00000000, 0x410F2CE7, 0xC13EC9E6, 0x40DE1EF1, 0x402B2DF1, 0xC128163A, 0x41353088,
        0xC092B726, 0x41400000, 0xC0784951, 0xC117DDA5, 0x412043CE, 0x4040F7B2, 0xC13F7501,
        0x409719BA, 0x410E9B9B,
    ];
    const SYNC_DST_OUT: [u32; 16] = [
        0x3FA9819B, 0x402E3E7E, 0xBE9A2560, 0xBEAA3B20, 0x40345150, 0x3FDADCD8, 0xBE9732C0,
        0x3EB8DD00, 0xBFF20078, 0x40162BC9, 0xC010FA62, 0xBFC79550, 0x4037D454, 0x3F738E00,
        0xC0208334, 0xC0143B34,
    ];
    const SYNC_WEIGHT_OUT: [u32; 8] = [
        0x3F7990F2, 0x3F7990F2, 0x3F7990F2, 0x3F6A1628, 0x3F6A1628, 0x3F6A1628, 0x3F685A13,
        0x3F685A13,
    ];
    const SYNC_PEAKS: [i64; 3] = [1, 4, 6];
    const SYNC_ST: [i64; 3] = [0, 3, 6];
    const SYNC_EN: [i64; 3] = [3, 6, 8];
    const SYNC_SENS_BITS: u32 = 0x3F400000;

    /// Bit-exact against `pyradius.vocoder_ops.randomize_phases` (nch = 2,
    /// max_bin = 8, start = 2, seed = 1).
    #[test]
    fn randomize_phases_matches_python_reference() {
        let (nch, mb) = (2usize, 8usize);
        let mut noise_phase = golden_rows(&RND_NP_IN, nch, mb);
        let mut noise_gain = golden_rows(&RND_NG_IN, nch, mb);
        let mut phase = golden_rows(&RND_PH_IN, nch, mb);
        let mut mag = golden_rows(&RND_MAG_IN, nch, mb);
        let region_gain = golden_rows(&RND_RG, nch, mb);
        let gain_mean = randomize_phases(
            &mut noise_phase,
            &mut noise_gain,
            &mut phase,
            &mut mag,
            &region_gain,
            &golden_flat(&RND_NW),
            &golden_flat(&RND_SW),
            f32::from_bits(RND_A2_BITS),
            f32::from_bits(RND_RAMP_BITS),
            RND_F372,
            RND_U112,
            &mut SimpleRand::new(RND_SEED),
            nch,
            mb,
        );
        assert_eq!(flat(&noise_phase), RND_NP_OUT);
        assert_eq!(flat(&noise_gain), RND_NG_OUT);
        assert_eq!(flat(&phase), RND_PH_OUT);
        assert_eq!(flat(&mag), RND_MAG_OUT);
        assert_eq!(flat(&[gain_mean]), RND_GM_OUT);
    }

    /// Bit-exact against `pyradius.vocoder_ops.substitute_noisy_phases`.  The
    /// two template rows differ, so this also pins the Python's
    /// `(t1 if (ch & 1) else t0)` row choice (see the function docs).
    #[test]
    fn substitute_noisy_phases_matches_python_reference() {
        let (nch, mb) = (2usize, 8usize);
        let mut noise_phase = golden_rows(&SUB_NP_IN, nch, mb);
        let mut phase = golden_rows(&SUB_PH_IN, nch, mb);
        let mut mag = golden_rows(&SUB_MAG_IN, nch, mb);
        let region_gain = golden_rows(&SUB_RG, nch, mb);
        let slot = substitute_noisy_phases(
            &mut noise_phase,
            &mut phase,
            &mut mag,
            &region_gain,
            &golden_flat(&SUB_T0),
            &golden_flat(&SUB_T1),
            &golden_flat(&SUB_NW),
            &golden_flat(&SUB_SW),
            f32::from_bits(RND_A2_BITS),
            f32::from_bits(RND_RAMP_BITS),
            RND_F372,
            RND_U112,
            SUB_SLOT_IN,
            SUB_SLOT_COUNT,
            SUB_STRIDE,
            nch,
            mb,
        );
        assert_eq!(slot, SUB_SLOT_OUT);
        assert_eq!(flat(&phase), SUB_PH_OUT);
        assert_eq!(flat(&mag), SUB_MAG_OUT);
    }

    /// Bit-exact against `pyradius.vocoder_ops.synchronize_stereo_phases`
    /// (`nch == 2`, three disjoint regions, sens = 0.75).
    #[test]
    fn synchronize_stereo_phases_matches_python_reference() {
        let nb = 8usize;
        let mag = golden_rows(&SYNC_MAG_IN, 2, nb);
        let src = golden_rows(&SYNC_SRC_IN, 2, nb);
        let dst_in = golden_rows(&SYNC_DST_IN, 2, nb);
        let (dst, weight) = synchronize_stereo_phases(
            &mag,
            &src,
            &dst_in,
            &SYNC_PEAKS,
            &SYNC_ST,
            &SYNC_EN,
            3,
            f32::from_bits(SYNC_SENS_BITS),
        );
        assert_eq!(flat(&dst), SYNC_DST_OUT);
        assert_eq!(flat(&[weight]), SYNC_WEIGHT_OUT);
    }
}
