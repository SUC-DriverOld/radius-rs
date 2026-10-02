//! Phase-domain vocoder operators — port of `pyradius/vocoder_ops.py`:
//! `unwrap_phase` (1757), `apply_pitch_coherence` (2014),
//! `reset_phases_for_transients` (2227), `adjust_multiphase_diff` (2366),
//! `_adjust_diff_nch2_vec` (2441) and `ampd_pull_to_peak` (2429), i.e. the C
//! `libradius/src/ops/{unwrap_phase,apply_pitch_coherence,reset_phases,
//! adjust_multiphase_diff}.c`.
//!
//! Float discipline matches `src/td.rs` / `src/ti.rs`: everything stays `f32`,
//! `a*b + c` stays two operations (the reference is compiled
//! `-ffp-contract=off`), and only the Python `_fma` / `_fma_arr` helpers (the
//! C's explicit `fmaf`) become [`fma`]. `math.sqrt`/`exp`/`pow` are evaluated in
//! `f64` and rounded once, `np.rint` is [`rint`].
//!
//! numpy-array conventions: `&[f32]` / `&mut [f32]` replace the arrays. Buffers
//! the C writes through in place (and that the Python therefore copies) are
//! `&mut [f32]`; where the Python returns a *new* array the function returns a
//! `Vec<f32>`. Every deviation is documented on the function.
//!
//! Integer arguments follow the reference's containers: the index tables
//! (`reg_start`, `reg_end`, `peak_bins`, `seg_bound`, `r_start`, …), counts and
//! bins are `i64` (numpy `int64` / the engine's index tables), and `u_70` stays
//! the engine's `uint32_t`.

use crate::consts::{PI_F, TWO_PI_F};
use crate::vocoder::{fma, rint, INV_2PI_F};

/// `np.float32(-6.2831854820251465)` — `_bits(0xC0C90FDB)`, `f32(-2*pi)`.
const NEG_2PI_F: f32 = f32::from_bits(0xC0C9_0FDB);
/// `_bits(0x3F333333)` — `K07_F`, the `0.7f` peak-clamp constant.
const K07_F: f32 = f32::from_bits(0x3F33_3333);
/// `_bits(0x3B449BA6)` — the `0.003f` scale of `reset_phases.c` B3 (`lim`).
const LIM_0P003_F: f32 = f32::from_bits(0x3B44_9BA6);
/// `np.float32(1e-9)` — `_bits(0x3089705F)`, the Σ guards in `reset_phases.c`.
const EPS1E9_F: f32 = f32::from_bits(0x3089_705F);

/// `_wrap_pi` — the `fmsub` wrap `fma(-rint(x*inv2pi), 2pi, x)`.
///
/// The Python `_wrap_pi` evaluates
/// `_fma_arr(-np.rint((x * INV_2PI_F)), np.float64(TWO_PI_F), x)` and, when
/// numba is present, `_wrap_pi_nb` — both are the correctly-rounded single
/// rounding `fmaf`, which is what the C `ampd_wrap_pi_fast` emits
/// (`fmaf(-rintf(x*k), 2pi, x)`).
///
/// Note that [`crate::vocoder::wrap_pi`] instead folds the same expression
/// through `f64` (`(q as f64) * NEG_TWO_PI_F64 + (x as f64)`, one cast), which
/// is a *different* double-rounded computation and can differ by one ulp; the
/// fused form below is the one the reference (and this module's golden vectors)
/// produce, so the phase operators use it.
#[inline(always)]
pub fn wrap_pi_fused(x: f32) -> f32 {
    fma(-rint(x * INV_2PI_F), TWO_PI_F, x)
}

/// `_RPT_A2ORD` (vocoder_ops.py 2224) — the Σa² 16-element block accumulation
/// order of `reset_phases.c` (asm 0x16d588, "D1"). Σb²/Σa stay in block order.
const RPT_A2ORD: [usize; 16] = [0, 1, 2, 3, 4, 5, 7, 6, 8, 9, 11, 15, 12, 13, 14, 10];

// ===========================================================================
// unwrap_phase.c
// ===========================================================================

/// `vocoder_ops.unwrap_phase` (line 1757) / `rx_unwrap_phase` @0x16B1C8 —
/// region phase unwrap.
///
/// Dropped/changed vs the Python:
/// * `np.array(scratch, copy=True)` / `np.array(phase, copy=True)`: both are the
///   C's in/out buffers (`float *scratch`, `float *phase_2120`), so they are
///   `&mut [f32]` here and nothing is returned; the caller owns the "copy" (the
///   Python's dict return collapses away).
/// * `f1, f2, f3, max_bin, copy_len` lose their Python defaults and become
///   required arguments (the engine's `rx_unwrap_in` fields; `f1`/`f2`/`f3` are
///   used through the Python's `F32(...)`, so they are accepted as `f32` and the
///   integral values the engine stores are reproduced exactly).
/// * the per-region peak scan is written as the C's `MaxIndex` (first strict
///   maximum over `[reg_start, reg_end]`) instead of the Python's
///   `np.maximum.at` / argmax-by-mask construction; both take the lowest index
///   attaining the region maximum, and a region with `reg_end < reg_start`
///   yields `peak = reg_start` with no iterations in either.
#[allow(clippy::too_many_arguments)]
pub fn unwrap_phase(
    mask: &[f32],
    mask_copy: &[f32],
    mag_copy: &[f32],
    reg_start: &[i64],
    reg_end: &[i64],
    reg_prev_peak: &[i64],
    reg_offset: &[f32],
    scratch: &mut [f32],
    phase: &mut [f32],
    f1: f32,
    f2: f32,
    f3: f32,
    max_bin: i64,
    copy_len: i64,
) {
    let nreg = reg_start.len();
    let mb = max_bin - 1;

    let f1f = f1;
    let inv_f1 = 1.0f32 / f1f;
    // ((f1 * 2pi) / f3) * 0.5 — the C's division/multiplication order.
    let v9 = ((f1f * TWO_PI_F) / f3) * 0.5f32;
    let f2f = f2;

    if nreg >= 1 {
        // ---- per-region peak: MaxIndex, first strict maximum ----
        let last = mag_copy.len() as i64 - 1;
        let mut peak = vec![0i64; nreg];
        let mut v16 = vec![0f32; nreg];
        for r in 0..nreg {
            let s = reg_start[r];
            let e = reg_end[r];
            // The C scans `[reg_start, reg_end]` inclusive; an empty region
            // (`reg_end < reg_start`) runs no iterations at all.
            let n = e - s + 1;
            let mut first: i64 = -1;
            if n >= 1 {
                // `np.clip(posall, 0, mag.size - 1)` — the Python clips the scan
                // indices, the C would read out of bounds.
                let at = |i: i64| mag_copy[i.clamp(0, last) as usize];
                let mut best = at(s);
                let mut idx = 0i64;
                for i in 1..n {
                    let v = at(s + i);
                    if v > best {
                        best = v;
                        idx = i;
                    }
                }
                first = s + idx;
            }
            peak[r] = if first >= 0 { first } else { s };
            v16[r] = peak[r] as f32;
        }

        // ---- parabolic peak refinement (bins 0 and max_bin-1 are skipped) ----
        for r in 0..nreg {
            let p = peak[r];
            if p != 0 && p != mb {
                let ap = mag_copy[(p + 1) as usize];
                let am = mag_copy[p as usize];
                let amm = mag_copy[(p - 1) as usize];
                // (a[p+1] - 2*a[p]) - a[p-1]: two independent f32 roundings.
                let v20 = (ap - (2.0f32 * am)) - amm;
                let mut v21 = v16[r];
                if v20 != 0.0 {
                    // (ap - amm) / (v20 + v20) + v16 — divide, then add.
                    v21 = ((ap - amm) / (v20 + v20)) + v16[r];
                }
                let mut t = v16[r] + K07_F;
                t = if t < v21 { t } else { v21 };
                let lo = v16[r] - K07_F;
                t = if t < lo { lo } else { t };
                v16[r] = t;
            }
        }

        // ---- per-region phase update ----
        for r in 0..nreg {
            let s = reg_start[r];
            let e = reg_end[r];
            if s < e {
                // v9 * (v16 + reg_offset[r]) — add, then multiply.
                let v37 = v9 * (v16[r] + reg_offset[r]);
                let dlt = peak[r] - reg_prev_peak[r];
                let n = e - s;
                for j in 0..n {
                    // `np.clip(bins + dlt, 0, mb)`; the C clamps per iteration
                    // (`v35 = (mb < idx) ? mb : idx; v35 &= ~(v35 >> 31)`) and
                    // keeps incrementing the *unclamped* index.
                    let idx = (s + j + dlt).max(0).min(mb) as usize;
                    let b = (s + j) as usize;
                    // (mask[b] - maskCopy[idx]) - v37: two subtractions.
                    let v38 = (mask[b] - mask_copy[idx]) - v37;
                    // fmaf(rintf(v38*inv2pi), -2pi, v38) — single rounding.
                    let wrap = fma(rint(v38 * INV_2PI_F), NEG_2PI_F, v38);
                    // fmaf(f2, inv_f1 * (v37 + wrap), phase[idx]).
                    scratch[b] = fma(f2f, inv_f1 * (v37 + wrap), phase[idx]);
                }
            }
        }
    }

    // Copy(copy_len, scratch, phase_2120) — the C's trailing memcpy.
    let c = copy_len as usize;
    phase[..c].copy_from_slice(&scratch[..c]);
}

// ===========================================================================
// apply_pitch_coherence.c
// ===========================================================================

/// The six 5-entry per-segment tables of `apply_pitch_coherence.c`
/// (`vocoder_ops.py` 1860-1865, image @0x217B28 stride 0x14). Kept as the Python
/// literals; `apc_tables_bit_patterns` pins their exact bit patterns.
pub const APC_T_LO: [f32; 5] = [1.5, 1.35, 1.070_000_05, 1.039_999_96, 1.029_999_97];
pub const APC_T_HI: [f32; 5] = [1.899_999_98, 1.700_000_05, 1.200_000_05, 1.12, 1.100_000_02];
pub const APC_T_GMIX: [f32; 5] = [0.699_999_99, 0.699_999_99, 0.600_000_02, 0.400_000_01, 0.2];
pub const APC_T_LIN: [f32; 5] = [0.600_000_02, 0.400_000_01, 0.2, 0.150_000_01, 0.1];
pub const APC_T_EA: [f32; 5] = [
    1.600_000_02,
    1.700_000_05,
    0.400_000_01,
    0.519_999_98,
    0.550_000_01,
];
pub const APC_T_EB: [f32; 5] = [2.5, 2.0, 0.800_000_01, 0.550_000_01, 0.550_000_01];

/// The `args` dict of [`apply_pitch_coherence`] (`rx_apc_args` in the C).
#[derive(Clone, Copy, Debug)]
pub struct ApcArgs {
    pub precision: i32,
    pub trans_sens: f32,
    pub total_ratio: f64,
    /// `this[0x9B8]` (`int(args["transient_state"])`).
    pub transient_state: i32,
    pub n_fft: i32,
    pub sr: i64,
    pub f580: i64,
    pub f584: i64,
    pub acc_fc0: f32,
    pub coh_center: f32,
    pub seg_count: usize,
    pub peak_count: usize,
    pub vector_fmaf_region: bool,
}

/// `vocoder_ops.apply_pitch_coherence` (line 2014) / `rx_apply_pitch_coherence`
/// @0x16B444 — returns `advance` (the C's `*out_advance`).
///
/// Dropped/changed vs the Python:
/// * `np.array(phase_mod/dir_cur/dir_prev, copy=True)` dropped: all three are
///   the in/out arrays the C mutates (`a->phase_mod`, `a->dir_cur`,
///   `a->dir_prev`); `phase` stays read-only (`a->phase`). The Python's dict
///   return therefore collapses to the `f32` `advance`.
/// * the numba fast path `_apc_full_nb` is route B only (it never runs on the
///   exact route) and is not ported; the scalar segment/peak loops below are the
///   reference. `_apc_region_nb` (numba, route-A reachable) implements the
///   *non*-vector semantics, so `vector_fmaf_region` is only observable in the
///   pure-numpy reference; both forms are implemented here.
/// * `c0 = coh_center` and `a4` do not vary with the segment, so `v27` is
///   hoisted out of the segment loop (same value; the C does the same), and the
///   C also hoists `wLin * PI_F` out of the peak loop.
/// * `F32(v23)`-style clamps use `f32::max`, whose NaN selection matches the C
///   `fmaxf` rather than Python's `max` (unreachable: `v133 > 0` past the gate).
#[allow(clippy::too_many_arguments)]
pub fn apply_pitch_coherence(
    args: &ApcArgs,
    env: &[f32],
    peak_bins: &[i64],
    reg_start: &[i64],
    reg_end: &[i64],
    seg_bound: &[i64],
    phase: &[f32],
    phase_mod: &mut [f32],
    dir_cur: &mut [f32],
    dir_prev: &mut [f32],
    region_gain: &[f32],
    noise_wt: &[f32],
    a3: f32,
    a4: f32,
) -> f32 {
    let trans_sens = args.trans_sens;
    let mut advance = -12345.0f32;
    if args.precision > 9 || trans_sens == 0.0 {
        return advance;
    }

    // total_ratio = max(r, 1/r), then expf((v8 - 1) * -5).
    let mut r = args.total_ratio;
    let inv = 1.0 / r;
    if r < inv {
        r = inv;
    }
    let v8 = r as f32;
    let v13 = (((v8 - 1.0f32) * -5.0f32) as f64).exp() as f32;

    let transient_state = args.transient_state;
    let v15 = if transient_state == 2 { 0.5f32 } else { 1.0f32 };
    let n = args.n_fft as f32;
    let a3f = a3;
    let a4f = a4;

    // phase advance: (acc_fc0 + (f580 - f584)) / a3, then a3 * (v17 - rint(v17)).
    let v17 = (args.acc_fc0 + ((args.f580 - args.f584) as f32)) / a3f;
    advance = a3f * (v17 - rint(v17));

    let nseg = args.seg_count;
    if nseg == 0 {
        return advance;
    }

    let mut v22 = n / a3f;
    let v133 = v15 * ((trans_sens * 1.4f32) * (v13 + 1.0f32));
    let v23 = (1.0f32 - (0.2f32 / v133)).max(0.2f32);
    // (int)(int64_t)rintf((N * 0.9f) / a3), truncated to 32 unsigned bits;
    // a non-finite argument saturates to INT64_MAX -> 0xFFFFFFFF.
    let arg = (n * 0.9f32) / a3f;
    let v24u: u32 = if arg.is_finite() {
        (rint(arg) as i64) as u32
    } else {
        0xFFFF_FFFF
    };
    let peak_cnt = args.peak_count;
    let vec = args.vector_fmaf_region;
    let sr = args.sr;
    let coh_center = args.coh_center;
    let v136 = v22;
    let v130 = ((advance * -2.0f32) * PI_F) / n;
    let v135 = n * 4000.0f32;

    // pow(c0, 1.7) + ((c0 * 1.5) * (a4 - 0.5)) — the pow in f64, the product in
    // f32, the sum in f64, one cast to f32.
    let mut v27 =
        ((coh_center as f64).powf(1.7) + (((coh_center * 1.5f32) * (a4f - 0.5f32)) as f64)) as f32;
    if v27 > 1.0 {
        v27 = 1.0;
    }

    let mut cursor = 0usize;
    for s in 0..nseg {
        let e = env[s];
        let lo = APC_T_LO[s];
        let hi = APC_T_HI[s];
        // The engine's per-condition shape (an inverted hi < lo segment stays
        // self-consistent); do not rewrite as a clamped formula.
        let mut v31 = (e - lo) / (hi - lo);
        if !(e < hi) {
            v31 = 1.0f32;
        }
        let ramp = if e > lo { v31 } else { 0.0f32 };
        let gm = APC_T_GMIX[s];
        // (1-gm) * (ramp + (v27 - ramp) * lin) + gm * sqrt(v27 * ramp)
        let mut v33 = (1.0f32 - gm) * (ramp + ((v27 - ramp) * APC_T_LIN[s]))
            + (gm * (((v27 * ramp) as f64).sqrt() as f32));
        if v33 > 1.0 {
            v33 = 1.0;
        }
        if v33 < 0.0 {
            v33 = 0.0;
        }
        let mut w_lin = ((v33 as f64).powf((APC_T_EA[s] / v133) as f64)) as f32;
        if v23 < w_lin {
            w_lin = v23;
        }
        if w_lin < 0.0 {
            w_lin = 0.0;
        }
        let mut w_exp = ((v33 as f64).powf((APC_T_EB[s] / v133) as f64)) as f32;
        if v23 < w_exp {
            w_exp = v23;
        }
        if w_exp < 0.0 {
            w_exp = 0.0;
        }

        if cursor >= peak_cnt {
            continue;
        }
        let v40 = w_exp * PI_F;
        let w_lin_pi = w_lin * PI_F;
        let bound = seg_bound[s + 1];
        while cursor < peak_cnt && peak_bins[cursor] < bound {
            let p = cursor;
            cursor += 1;
            let bin_ = peak_bins[p];
            // (unsigned)bin < (unsigned)v24: peaks below the harmonic search
            // bound are skipped entirely.
            if ((bin_ as u64) & 0xFFFF_FFFF) < (v24u as u64) {
                continue;
            }
            let bf = bin_ as f32;
            let h = rint(bf / v22) as i64;
            let mut thr = v40;
            if h != 0 {
                let center = v22 * (h as f32);
                let outside = ((center - 2.0f32) >= bf) || ((center + 2.0f32) <= bf);
                if !outside {
                    // Harmonic alignment: refine v22 (state carried across peaks).
                    let ratio = bf / (h as f32);
                    let hf = h as f32;
                    v22 = ratio + ((v136 - ratio) / ((hf as f64).sqrt().sqrt() as f32));
                    let dd = (bf - center).abs();
                    let mut v53 = (dd - 0.7f32) / 1.3f32;
                    if dd >= 2.0 {
                        v53 = 1.0;
                    }
                    let v54 = if dd > 0.7 { v53 } else { 0.0f32 };
                    thr = w_lin_pi + ((v40 - w_lin_pi) * v54);
                }
            }
            // (phase[bin] + bin * v130) - phase_mod[bin]: three f32 operations.
            let dph = (phase[bin_ as usize] + (bf * v130)) - phase_mod[bin_ as usize];
            // NOTE: unlike `_wrap_pi`, this wrap is *not* fused (the C emits a
            // plain fmul + fadd here): dph + rint(dph*inv2pi) * (-2pi).
            let wrapped = dph + (rint(dph * INV_2PI_F) * NEG_2PI_F);
            let dp = dir_prev[bin_ as usize];
            let dc = dir_cur[bin_ as usize];
            let arm1 = (dp > dc) && (wrapped > 0.0) && (dc > wrapped);
            let arm2 = (dp < dc) && (wrapped < 0.0) && (dc < wrapped);
            if arm1 || arm2 {
                let v64 = dp / wrapped;
                let mut v65 = 0.0f32;
                if v64 > 1.0 {
                    v65 = 1.0;
                    if v64 < 4.0 {
                        v65 = (v64 - 1.0) / 3.0;
                    }
                }
                thr = thr * ((v65 * 0.75f32) + 0.25f32);
            }
            if dp == 0.0 && dc == 0.0 {
                let mut m = PI_F - thr;
                if thr < m {
                    m = thr;
                }
                thr = thr + (m * 0.2f32);
            } else {
                let mut m = PI_F - thr;
                if thr < m {
                    m = thr;
                }
                let t68 = thr + (m * 0.15f32);
                if dc == 0.0 {
                    thr = t68;
                }
            }
            let rg = region_gain[bin_ as usize];
            let mut v71 = 0.0f32;
            if rg > 2.0 {
                v71 = 1.0;
                if rg < 6.0 {
                    v71 = (rg - 2.0) * 0.25;
                }
            }
            let nw = noise_wt[bin_ as usize];
            let mut v70 = 0.0f32;
            if nw > 0.4 {
                v70 = 1.0;
                if nw < 0.7 {
                    v70 = (nw - 0.4) / 0.3;
                }
            }
            // `(unsigned)(a->sr * bin)` — 32-bit wraparound, then to f32.
            let srbin = ((sr as u64).wrapping_mul(bin_ as u64) & 0xFFFF_FFFF) as f32;
            if ((1.0f32 - v71) + v70) > 1.0 && v135 < srbin && transient_state != 2 {
                thr = 10.0;
            }
            let rrs = reg_start[p];
            let ree = reg_end[p];
            if rrs < ree {
                if wrapped.abs() > thr {
                    // APPLY: the whole region is pulled to `wrapped`, the old
                    // direction shifts into dir_prev.
                    for b in rrs..ree {
                        let b = b as usize;
                        dir_prev[b] = dir_cur[b];
                        dir_cur[b] = wrapped;
                    }
                } else {
                    // SMOOTH: per-bin phase_mod advance, dir_cur cleared.
                    for b in rrs..ree {
                        let b = b as usize;
                        let pm = phase_mod[b];
                        let bf = b as f32;
                        let dd = if vec {
                            // _fma_arr(b, v130, phase[b]) - pm
                            fma(bf, v130, phase[b]) - pm
                        } else {
                            (phase[b] + (bf * v130)) - pm
                        };
                        let nn = rint(dd * INV_2PI_F);
                        let wp = if vec {
                            fma(nn, NEG_2PI_F, dd)
                        } else {
                            dd + (nn * NEG_2PI_F)
                        };
                        phase_mod[b] = pm + wp;
                        dir_prev[b] = dir_cur[b];
                        dir_cur[b] = 0.0;
                    }
                }
            }
        }
    }
    advance
}

// ===========================================================================
// reset_phases.c
// ===========================================================================

/// `vocoder_ops.reset_phases_for_transients` (line 2227) /
/// `rx_reset_phases_for_transients` @0x16D354 — the three entry branches.
///
/// Dropped/changed vs the Python:
/// * `np.array(phase, copy=True)` / `np.array(mask_table, copy=True)` dropped:
///   both are the C's in/out buffers (`phase_848`, `mask_table_8f8`), so they are
///   `&mut [f32]` here and nothing is returned. The trailing `_n_out` keeps the
///   Python's `[:n_out]` contract visible: the caller only consumes the first
///   `n_out` samples (in the vocoder `n_out == phase.len()`, so the writes below
///   are all inside the returned window).
/// * `e08` is a local initialised to `np.float64(0.0)` in the Python and is
///   never written, so the B3 guard compares `lim` against `0.0f`; kept literal
///   below.
/// * `n578` is unused in the Python body too (`_n578`).
/// * `scale_20` stays `f64`: the Python passes the *double* `self.ratio` and
///   `reset_phases.c` reads `*(double*)(this+0x20)` (`(f64)mask[i] * scale_20`,
///   one rounding). Passing a pre-rounded `f32` here is not bit-exact.
#[allow(clippy::too_many_arguments)]
pub fn reset_phases_for_transients(
    mode_4b8: i32,
    proc_mode_9b8: i32,
    scale_20: f64,
    len_594: i64,
    n9bc: i64,
    n9c0: i64,
    u_70: u32,
    _n578: i64,
    n590: i64,
    n5d0: i64,
    n5d4: i64,
    avg_m1: f32,
    avg_0: f32,
    avg_p1: f32,
    use_p568: bool,
    mag: &[f32],
    b_710: &[f32],
    mask: &[f32],
    r_start: &[i64],
    r_end: &[i64],
    r_bin: &[i64],
    phase: &mut [f32],
    mask_table: &mut [f32],
    _n_out: usize,
) {
    // ---- [B1] this[0x4B8] == 1 ----
    if mode_4b8 == 1 {
        let n = len_594;
        if n >= 1 {
            // (float)((double)mask[i] * scale_20): f64 product, one cast.
            for i in 0..n as usize {
                phase[i] = ((mask[i] as f64) * scale_20) as f32;
            }
        }
        return;
    }

    // ---- [B2] this[0x9B8] == 2 ----
    if proc_mode_9b8 == 2 {
        let v12 = (avg_0 as f64).sqrt() as f32;
        let mut v13 = 0.0f32;
        if v12 > 0.8 {
            v13 = 1.0;
            if v12 < 1.5 {
                v13 = (v12 - 0.8) / 0.7;
            }
        }
        let mut v14 = fma(v13, 0.7f32, 0.5f32);
        if n9c0 < n9bc {
            let q = (n9bc as f32) / (n9c0 as f32);
            v14 = v14 * ((q as f64).sqrt() as f32);
        }
        let r_count = r_start.len();
        if r_count < 1 {
            return;
        }
        for r in 0..r_count {
            let s_ = r_start[r];
            let e_ = r_end[r];
            let mut sa2 = 0.0f32;
            let mut sb2 = 0.0f32;
            let mut sa = 0.0f32;
            if s_ < e_ {
                let n = e_ - s_;
                let mut i: i64 = 0;
                // 16-element blocks: Σa² in the asm's cross-issue order, Σb² and
                // Σa in block order.
                while i + 16 <= n {
                    let mut a2 = [0.0f32; 16];
                    let mut b2 = [0.0f32; 16];
                    let mut al = [0.0f32; 16];
                    for k in 0..16 {
                        let a = mag[(s_ + i + k as i64) as usize];
                        let b = b_710[(s_ + i + k as i64) as usize];
                        a2[k] = a * a;
                        b2[k] = b * b;
                        al[k] = a;
                    }
                    for k in 0..16 {
                        sa2 += a2[RPT_A2ORD[k]];
                    }
                    for k in 0..16 {
                        sb2 += b2[k];
                    }
                    for k in 0..16 {
                        sa += al[k];
                    }
                    i += 16;
                }
                // 4-element blocks.
                while i + 4 <= (n & !3) {
                    for k in 0..4 {
                        let av = mag[(s_ + i + k) as usize];
                        let bv = b_710[(s_ + i + k) as usize];
                        sa2 += av * av;
                        sb2 += bv * bv;
                        sa += av;
                    }
                    i += 4;
                }
                // scalar tail.
                while i < n {
                    let av = mag[(s_ + i) as usize];
                    let bv = b_710[(s_ + i) as usize];
                    sa2 += av * av;
                    sb2 += bv * bv;
                    sa += av;
                    i += 1;
                }
            }
            sa += EPS1E9_F;
            // ((r_bin * u_70) / n5d0) + -1500) / 2000, then fma(v88, -23, 25).
            let mut v88 =
                ((((r_bin[r] as f32) * (u_70 as f32)) / (n5d0 as f32)) + (-1500.0f32)) / 2000.0f32;
            v88 = fma(v88, -23.0f32, 25.0f32);
            if v88 > 25.0 {
                v88 = 25.0;
            }
            if v88 < 2.0 {
                v88 = 2.0;
            }
            // n590 == 0 is preserved (inf), as in the engine.
            let v89 = v88 / (n590 as f32);
            let mut v27 = 0.0f32;
            let thr = sb2 * (v89 + 1.0f32);
            if sa2 > thr {
                let base = fma(v89, 1.5f32, 1.0f32);
                let v91 = fma(base, sb2, EPS1E9_F);
                v27 = 1.0;
                if sa2 < v91 {
                    v27 = (sa2 - thr) / (v91 - thr);
                }
            }
            let v92 = (((sa2 * ((e_ - s_) as f32)) as f64).sqrt() as f32) / sa;
            let mut v93 = 0.0f32;
            if v92 > 1.0 {
                v93 = 1.0;
                if v92 < 1.5 {
                    v93 = (v92 - 1.0) + (v92 - 1.0);
                }
            }
            let score = fma(v14 * (1.0f32 - v93), 0.5f32, v14 * v27);
            if score <= 0.75 {
                if s_ < e_ {
                    if use_p568 {
                        // Conditional copy: only bins whose maskTable != 0.
                        for i in s_..e_ {
                            let i = i as usize;
                            if mask_table[i] != 0.0 {
                                phase[i] = mask[i];
                                mask_table[i] = 1.0;
                            }
                        }
                    } else {
                        for i in s_..e_ {
                            let i = i as usize;
                            phase[i] = mask[i];
                            mask_table[i] = 1.0;
                        }
                    }
                }
            } else if e_ > s_ {
                for i in s_..e_ {
                    let i = i as usize;
                    phase[i] = mask[i];
                    mask_table[i] = 1.0;
                }
            }
        }
        return;
    }

    // ---- [B3] else ----
    let a_m1 = avg_m1;
    let a_0 = avg_0;
    let a_p1 = avg_p1;
    if a_m1 >= a_0 {
        if a_0 <= 2.0 {
            return;
        }
    } else {
        let w8 = (a_0 > a_p1) && (a_0 > 1.2);
        if a_0 <= 2.0 && !w8 {
            return;
        }
    }
    let u70f = u_70 as f32;
    let lim = u70f * LIM_0P003_F;
    // e08 is always np.float64(0.0) here, so float(int(e08)) == 0.0f.
    let e08 = 0.0f64;
    if lim >= (e08 as i64) as f32 {
        return;
    }
    let v130 = ((n5d0 as f32) * 1500.0f32) / u70f;
    let half = if v130 >= 0.0 { 0.5f32 } else { -0.5f32 };
    let t = v130 + half;
    // fcvtzu: negative saturates to 0.
    let start = if t < 0.0 { 0i64 } else { t as i64 };
    if n5d4 <= start {
        return;
    }
    for i in start..n5d4 {
        let i = i as usize;
        phase[i] = mask[i];
    }
}

// ===========================================================================
// adjust_multiphase_diff.c
// ===========================================================================

/// `vocoder_ops.adjust_multiphase_diff` (line 2366) /
/// `rx_adjust_multiphase_diff` @0x16BED4 — pull `dst[ch][dst_bin]` toward
/// `src[ch][src_bin]` plus the circular mean difference, weight `w`.
///
/// Dropped/changed vs the Python:
/// * 2-D arrays `src[ch][bin]` / `dst[ch][bin]` are `&[&[f32]]` /
///   `&mut [&mut [f32]]`.
/// * `np.array(dst, copy=True)` dropped: `dst` is the in/out buffer (the C takes
///   `float *const *dst`) and is mutated in place; the Python's `return dst` is
///   therefore dropped too.
/// * the generic-path scratch buffers (`np.zeros(nch)`, the C's
///   `scratch_f`/`scratch_i` at `this+0xDC8`/`0xDE0`) are allocated per call and
///   zeroed, as the Python does (the engine leaves `scratch_i` dirty across
///   calls — a documented deviation of the C wrapper, not of this port).
pub fn adjust_multiphase_diff(
    src: &[&[f32]],
    dst: &mut [&mut [f32]],
    src_bin: i64,
    dst_bin: i64,
    w: f32,
    nch: i64,
) {
    if nch == 2 {
        let sb = src_bin as usize;
        let db = dst_bin as usize;
        let s0 = src[0][sb];
        let s1 = src[1][sb];
        let d0 = dst[0][db];
        let d1 = dst[1][db];
        let a0 = wrap_pi_fused(d0 - s0);
        let a1 = wrap_pi_fused(d1 - s1);
        let mut m = (a0 + a1) * 0.5f32;
        // `!(|a1 - a0| < pi)` — == pi goes through the +pi flip too.
        if !((a1 - a0).abs() < PI_F) {
            m = m + PI_F;
        }
        m = wrap_pi_fused(m);
        let t0 = wrap_pi_fused((s0 + m) - d0);
        dst[0][db] = wrap_pi_fused(fma(w, t0, d0));
        let t1 = wrap_pi_fused((s1 + m) - d1);
        dst[1][db] = wrap_pi_fused(fma(w, t1, d1));
        return;
    }
    if nch <= 0 {
        return;
    }

    // ---- generic (nch > 2) circular-mean rotation search ----
    let n = nch as usize;
    let sb = src_bin as usize;
    let db = dst_bin as usize;
    let inv = 1.0f32 / (n as f32);
    let step = inv * TWO_PI_F;
    let mut scratch_f = vec![0.0f32; n];
    let mut scratch_i = vec![0i32; n];
    let mut ssum = 0.0f32;
    for ch in 0..n {
        let d = wrap_pi_fused(dst[ch][db]);
        dst[ch][db] = d;
        scratch_f[ch] = wrap_pi_fused(d - src[ch][sb]);
        ssum = ssum + scratch_f[ch];
    }
    let mut mean = inv * ssum;
    let mut best = 1000000.0f32;
    let mut bestk = 0i32;
    for k in 0..n {
        // np.argmax — first maximum (strict >).
        let mut arg = 0usize;
        for ch in 1..n {
            if scratch_f[ch] > scratch_f[arg] {
                arg = ch;
            }
        }
        scratch_i[arg] = k as i32;
        scratch_f[arg] = scratch_f[arg] + TWO_PI_F;
        mean = fma(inv, TWO_PI_F, mean);
        let mut disp = 0.0f32;
        for ch in 0..n {
            let t = scratch_f[ch] - mean;
            let tw = fma(rint(t * INV_2PI_F), -TWO_PI_F, t);
            disp = disp + (tw * tw);
        }
        if best > disp {
            best = disp;
            bestk = k as i32;
        }
    }
    for ch in 0..n {
        if scratch_i[ch] > bestk {
            scratch_f[ch] = scratch_f[ch] - TWO_PI_F;
            mean = mean - step;
        }
    }
    let m = wrap_pi_fused(mean);
    for ch in 0..n {
        let d = dst[ch][db];
        let t = wrap_pi_fused((m + src[ch][sb]) - d);
        dst[ch][db] = wrap_pi_fused(fma(w, t, d));
    }
}

/// `vocoder_ops._adjust_diff_nch2_vec` (line 2441) — the element-wise `nch == 2`
/// body of [`adjust_multiphase_diff`] over a whole slice, with a scalar `w`.
///
/// Identical arithmetic to the `nch == 2` branch above with `src_bin == dst_bin
/// == i`; the two agree bit for bit (`nch2_vec_agrees_with_scalar_loop`). The
/// Python returns two fresh arrays, so this returns `(o0, o1)`.
pub fn adjust_diff_nch2_vec(
    src0: &[f32],
    src1: &[f32],
    dst0: &[f32],
    dst1: &[f32],
    w: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = src0.len().min(src1.len()).min(dst0.len()).min(dst1.len());
    let mut o0 = vec![0.0f32; n];
    let mut o1 = vec![0.0f32; n];
    for i in 0..n {
        let a0 = wrap_pi_fused(dst0[i] - src0[i]);
        let a1 = wrap_pi_fused(dst1[i] - src1[i]);
        let mut m = (a0 + a1) * 0.5f32;
        if !((a1 - a0).abs() < PI_F) {
            m = m + PI_F;
        }
        let m = wrap_pi_fused(m);
        let t0 = wrap_pi_fused((src0[i] + m) - dst0[i]);
        o0[i] = wrap_pi_fused(fma(w, t0, dst0[i]));
        let t1 = wrap_pi_fused((src1[i] + m) - dst1[i]);
        o1[i] = wrap_pi_fused(fma(w, t1, dst1[i]));
    }
    (o0, o1)
}

// ===========================================================================
// ampd hook (adjust_multiphase_diff.c / ProcessVocoder inline @0x168B0C)
// ===========================================================================

/// `vocoder_ops.ampd_pull_to_peak` (line 2429) / `rx_ampd_pull_to_peak`
/// @0x168B0C — loose phase locking, in place on `pm`.
///
/// Dropped/changed vs the Python: `np.array(pm, copy=True)` dropped — `pm` is
/// the C's in/out `float *pm_ch` and is mutated in place, so the Python's
/// `return pm` is dropped. The result is **not** wrapped
/// (`pm[bin] = fma(w, s, pm_bin)`).
pub fn ampd_pull_to_peak(pm: &mut [f32], mask_ch: &[f32], bin: i64, peak_bin: i64, w: f32) {
    let b = bin as usize;
    let pb = peak_bin as usize;
    let pm_bin = pm[b];
    let pm_pk = pm[pb];
    let t0 = wrap_pi_fused(pm_pk - pm_bin);
    let t1 = wrap_pi_fused(mask_ch[b] - mask_ch[pb]);
    let s = wrap_pi_fused(t0 + t1);
    pm[b] = fma(w, s, pm_bin);
}

// ===========================================================================

// ===========================================================================
// tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn f32s(b: &[u32]) -> Vec<f32> {
        b.iter().map(|u| f32::from_bits(*u)).collect()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// Bit-exact comparison with a readable first-divergence message.
    fn assert_bits(got: &[f32], want: &[u32]) {
        assert_eq!(got.len(), want.len(), "length mismatch");
        for i in 0..want.len() {
            assert_eq!(
                got[i].to_bits(),
                want[i],
                "element {i}: got 0x{:08X} want 0x{:08X}",
                got[i].to_bits(),
                want[i]
            );
        }
    }

    // ---------------------------------------------------------------------
    // Generated verbatim data - keep out of rustfmt's way.
    #[rustfmt::skip]
    mod golden {
        // Golden vectors: every input/output below is a value the *reference*
        // Python produces (route A, pyradius.vocoder_ops, numba present),
        // dumped as f32 bit patterns by .ref/gen_phase_golden.py.
        // Regenerate: python .ref/gen_phase_golden.py > .ref/phase_golden.rs
    // ---------------------------------------------------------------------
    pub const UWP_MASK: &[u32] = &[0xBF20AF91,0x3EA01160,0xBE6A11F5,0xBFF5CECF,0x3E863033,0x3FCBA42A,0xC01F3020,0xC00EDBEE,0x3EBB1B80,0x3F277BD0,0x3F24056A,0xBF00C8B4,0x3F674647,0x402B89D1,0xBCF671BC,0xBFF664A3,0xBF58AF0C,0x3F7235C8,0xC01A855B,0xBDAE4D00,0xBFD7AEE1,0x401C07DE,0xBF98E92E,0x3FC40084,0xC03D7669,0x3FA6DE16,0x3F29F263,0x3F3F2714,0x400B4443,0x3ECF39FF,0xC00BF5BF,0x3FD0777C,0xBEA0D636];
    pub const UWP_MASKCOPY: &[u32] = &[0x3FAA6693,0xC023B184,0x40250511,0xC03215C7,0xBF81EDA5,0x4005E066,0x40130462,0xBFE21BFA,0x3F9B4A12,0xBECD0491,0x3FFB74E6,0x403EEFC1,0xC0194E8E,0xC0160BEC,0xBF224DAA,0xBFD1A0B5,0xBF836002,0x403017BC,0xC0220CA7,0xBF1776F0,0xBEF1BA47,0x4021C2ED,0xBF971E89,0xBF2E4984,0xC02B374C,0x3F26C75E,0x3F56B6F8,0x3FC63EAE,0x3F3552E8,0xBF8AFB98,0x3FA96343,0xBF989080,0x401AFF82];
    pub const UWP_MAG: &[u32] = &[0x3F515DF0,0x3D5301E5,0x3EAFFC24,0x3EF6B29F,0x3F35ECCB,0x3DAD4023,0x3EC2F9A5,0x3E6707DC,0x3E81CF31,0x3F568055,0x3F39BB63,0x3E357B81,0x3E1179E3,0x3D9F7EAB,0x3F558915,0x3E190635,0x3D6ABF2B,0x3EB1CD63,0x3F51F4C1,0x3ECA7351,0x3F23BE1E,0x3D58F07A,0x3ED4E96B,0x3F3ED2B1,0x3EDF3454,0x3F640459,0x3F402646,0x3F304BF5,0x3DB0DCD8,0x3EEAE1AF,0x3F71D9EB,0x3F1C4D6F,0x3E63412D];
    pub const UWP_REGOFF: &[u32] = &[0x3E800000,0xBFC00000,0x40700000];
    pub const UWP_SCRATCH_IN: &[u32] = &[0x3EAC3A5A,0x3EC85955,0x3DFD973C,0x3DB5AEA7,0x3E15D27E,0xBE874F20,0x3E2E8081,0xBE9E766E,0xBE56F37F,0xBEFF507D,0x3D8CE47E,0xBC481D5D,0x3E306B77,0x3E0911BD,0xBEED997F,0x3E8C495B,0x3EC87F86,0x3E9D7218,0xBC9A6CC9,0x3E6DB647,0xBDD63545,0x3E721319,0x3EA5E7DE,0x3EACC9B4,0x3E316303,0xBE2B2A74,0xBE95FE5D,0x3DC1D02E,0x3E69B778,0x3EFC42B7,0xBD844915,0x3E2240D1,0x3E8A51B9];
    pub const UWP_PHASE_IN: &[u32] = &[0x3FC9C322,0x3FF083F3,0xBEBF7501,0xBF1BDCCA,0x3F614BA5,0xC01EA6E2,0xBB5ECDB7,0x3FCF55E9,0xC03149A8,0xBEDF8707,0xC00F58D9,0xBF137428,0x40100E0D,0x3F9DC822,0xBFF369EA,0x3FF74E2F,0x3EF07682,0x402FA11B,0xBEE9D027,0xBFD73158,0x401E2384,0x3FF24B49,0xC022854E,0xBFAF4778,0x3FCC718C,0x400DC554,0x3FA79F63,0x400C13B9,0xC03182E5,0x40324567,0xC005BE9B,0xC03DCEC7,0xC01526C1];
    pub const UWP_SCRATCH_OUT: &[u32] = &[0x3FC945C3,0x3FC981F1,0x3FF11903,0xBEBDAC34,0x3E15D27E,0xC01D24C9,0x3CC530F1,0x3FD25C5A,0xC02FD2A9,0xBED1E80B,0xC00DF116,0xBF0EE9ED,0x401140AD,0x3FA09BD7,0xBFF01F04,0x3FFA600C,0x3EC87F86,0xC01BF8C1,0xBFA1934E,0x3FD9A84F,0x40148C1E,0x3FB496BB,0x4012CD7E,0xC02B2010,0x40391AF9,0xBFFEED7B,0xC0374AE6,0xC00E4ACD,0xC00EE620,0xC00E55BE,0xC00EA8AD,0xC00E2E94,0x3E8A51B9];
    pub const UWP_PHASE_OUT: &[u32] = &[0x3FC945C3,0x3FC981F1,0x3FF11903,0xBEBDAC34,0x3E15D27E,0xC01D24C9,0x3CC530F1,0x3FD25C5A,0xC02FD2A9,0xBED1E80B,0xC00DF116,0xBF0EE9ED,0x401140AD,0x3FA09BD7,0xBFF01F04,0x3FFA600C,0x3EC87F86,0xC01BF8C1,0xBFA1934E,0x3FD9A84F,0x40148C1E,0x3FB496BB,0x4012CD7E,0xC02B2010,0x40391AF9,0xBFFEED7B,0xC0374AE6,0xC00E4ACD,0xC00EE620,0xC00E55BE,0xC00EA8AD,0xC00E2E94,0x3E8A51B9];
    pub const UWP2_REGOFF: &[u32] = &[0x00000000,0x40000000,0xBF000000,0x3F800000];
    pub const UWP2_SCRATCH_IN: &[u32] = &[0x3DD6BEAC,0xBE4B9ABE,0xBDA09A2E,0xBC502CEC,0x3E80F232,0xBE975AAA,0xBECDA204,0x3EBF55D3,0x3EF8A690,0xBE6A25EA,0xBEF869D5,0xBE2EF82D,0x3E410B4A,0x3E17873A,0x3E74F2A4,0xBE750E02,0xBEEAB272,0x3E079E33,0x3DD71B6C,0xBE9DC5C3,0xBEF8C1AA,0xBE729AEB,0xBE492B39,0xBCF31C5B,0x3EADD7C2,0x3E0CE8AE,0x3DBCBD79,0xBE1DD219,0x3EA7AA0F,0x3E9117D7,0x3E7EE29F,0x3E43A2DC,0x3E874E31];
    pub const UWP2_PHASE_IN: &[u32] = &[0x3FDE1075,0x3F98BEA6,0xBFF47D02,0xBF12A7DD,0xBE886EE7,0xC03D991B,0x3F4F548A,0xBF7B7450,0xC00187F1,0xBFB7FC8E,0xBF37B7E5,0x403F8D7E,0xBE6F3E21,0xBFAC95B8,0x3FCED8F4,0xBEA059F0,0x3FF098C0,0x3F8DA946,0xBF5C087B,0xC0016337,0xBFC6BB05,0x3FF4E07B,0x3FD50AA3,0x3F341EFB,0xC0312556,0xC02DB652,0x3F2CD0FF,0x3FF58E31,0xBF28E217,0x3C48DCAB,0xBF8E2AEE,0x4019BD95,0x3E163851];
    pub const UWP2_SCRATCH_OUT: &[u32] = &[0x3FDD9316,0xBE4B9ABE,0xBDA09A2E,0xBC502CEC,0x3E80F232,0xC03B4DF3,0x3F565CDF,0xBF72432E,0xBFFE8FC4,0xBFB494CF,0xBF2EF499,0x4041791D,0x3E410B4A,0x3E17873A,0x3E74F2A4,0xBE750E02,0xBEEAB272,0x3E079E33,0x3DD71B6C,0xBE9DC5C3,0xBEF8C1AA,0xBE729AEB,0xBE492B39,0xBCF31C5B,0x3EADD7C2,0x3E0CE8AE,0x3DBCBD79,0xBE1DD219,0x3EA7AA0F,0x3E9117D7,0x3E7EE29F,0x3E43A2DC,0x3E874E31];
    pub const UWP2_PHASE_OUT: &[u32] = &[0x3FDD9316,0xBE4B9ABE,0xBDA09A2E,0xBC502CEC,0x3E80F232,0xC03B4DF3,0x3F565CDF,0xBF72432E,0xBFFE8FC4,0xBFB494CF,0xBF2EF499,0x4041791D,0x3E410B4A,0x3E17873A,0x3E74F2A4,0xBE750E02,0xBEEAB272,0x3E079E33,0x3DD71B6C,0xBE9DC5C3,0xBEF8C1AA,0xBE729AEB,0xBE492B39,0xBCF31C5B,0x3EADD7C2,0x3E0CE8AE,0x3DBCBD79,0xBE1DD219,0x3EA7AA0F,0x3E9117D7,0x3E7EE29F,0x3E43A2DC,0x3E874E31];
    pub const APC_ENV: &[u32] = &[0x3F800000,0x3FA00000,0x3F866666,0x3F933333,0x3F733333];
    pub const APC_PHASE: &[u32] = &[0x400C8B92,0x403C484B,0xBF355345,0x3F1AE182,0xC016C842,0x3E87A149,0xBE97471F,0x3F9D8E0C,0x3F42DC92,0x403631CE,0xBF7432FC,0xBEDA399E,0xC0360626,0xBEB5F98A,0x3F89D342,0x3F9EB813,0x3FE17C2B,0x3FA4F187,0xBFB0A234,0xBFC213A7,0x3D24A18C,0xBF511229,0x3CE4F657,0x3FB81A8C];
    pub const APC_PHASEMOD_IN: &[u32] = &[0x3FC60181,0x3FA39D3D,0xC028130E,0xBF548683,0xC022CCFF,0xBFF7354D,0xBFE5DD77,0x401C8EA8,0x3E6F6020,0x400DAA9D,0x4032D21B,0x400E3801,0x3E8A547C,0x401A7746,0x3F661086,0x3F974A1B,0xBFE82F59,0x3FD8B3EE,0x3FD8DDA1,0x3FE3E690,0x40175483,0xC03C9D64,0xBF695855,0x3F701806];
    pub const APC_DIRCUR_IN: &[u32] = &[0x3F32C220,0xBD2B7DB6,0x3E75F58F,0xBDBCBC72,0xBF097FCC,0xBF42CC77,0x3F2A8A36,0xBEAC7927,0x3E8C9331,0xBECFBF67,0x3F119D9C,0xBF12FD16,0xBE402DB0,0xBEA81FDB,0x3E928909,0xBE1BA256,0x3F406A40,0x3E757AF9,0xBD7FB7CB,0xBED08DEB,0x3D9DBFB0,0x3EEBA711,0xBD4BBB71,0xBF3BB47F];
    pub const APC_DIRPREV_IN: &[u32] = &[0xBF34F575,0x3F0CCD68,0xBED331B6,0xBD9E8BDD,0x3F202FBD,0x3F38E82B,0xBF39E2CB,0xBE789457,0xBF385491,0x3D8C1C3A,0xBE4DC112,0xBEB4CE45,0x3EE0228B,0x3F304826,0xBEA86460,0xBF1B7B1D,0x3CE65C1E,0x3E096244,0xBF074151,0x3F17E27A,0x3F1FC1A0,0x3F32EBBD,0x3EDCD9A1,0x3ECA134E];
    pub const APC_REGG: &[u32] = &[0x402B28A4,0x3FECC212,0x4003F69A,0x409D1217,0x40A02DC9,0x4006B01F,0x40CDD2A9,0x4044CB78,0x40F6F5BD,0x40E9FBB0,0x400EC79B,0x40AC15C4,0x40D7CFAF,0x402002D0,0x40CD7A17,0x40831CB2,0x4095518D,0x40FB4834,0x40AE1DC7,0x40B2B666,0x40B094A7,0x40FE84AB,0x3D9D1E01,0x408CBDD7];
    pub const APC_NOISEWT: &[u32] = &[0x3E7C3034,0x3F014CA9,0x3EB206B4,0x3E80D111,0x3E77F084,0x3EEE20EB,0x3EC11BB8,0x3F0DC14F,0x3F2C2653,0x3F17E50A,0x3F40D836,0x3E9493AD,0x3F65D240,0x3F56BA71,0x3ED81806,0x3E73926C,0x3DE02D9D,0x3F6B1984,0x3ECB14DE,0x3DCC0DD8,0x3F6893C5,0x3BC6A998,0x3EF6E06A,0x3F1B5676];
    pub const APC_PHASEMOD_OUT: &[u32] = &[0x3FC60181,0x3FA39D3D,0xC028130E,0xBF548683,0xC022CCFF,0xBFF7354D,0xBFE5DD77,0x401C8EA8,0x3E6F6020,0x400DAA9D,0x4051208B,0x403D7DE5,0xBE954A24,0x401A7746,0x3FFAEC2B,0x3FA50093,0x3F79E86E,0x3FD8B3EE,0x401CBA38,0x3FBD3290,0x400CCD6B,0xC03C9D64,0xBF695855,0x3F701806];
    pub const APC_DIRCUR_OUT: &[u32] = &[0x3F32C220,0xBD2B7DB6,0xBF88BD44,0xBF88BD44,0xBF88BD44,0xBF42CC77,0xBF4605A0,0xBF4605A0,0xBF4605A0,0xBECFBF67,0x00000000,0x00000000,0x00000000,0xBEA81FDB,0x00000000,0x00000000,0x00000000,0x3E757AF9,0x00000000,0x00000000,0x00000000,0x3FB71500,0x3FB71500,0x3FB71500];
    pub const APC_DIRPREV_OUT: &[u32] = &[0xBF34F575,0x3F0CCD68,0x3E75F58F,0xBDBCBC72,0xBF097FCC,0x3F38E82B,0x3F2A8A36,0xBEAC7927,0x3E8C9331,0x3D8C1C3A,0x3F119D9C,0xBF12FD16,0xBE402DB0,0x3F304826,0x3E928909,0xBE1BA256,0x3F406A40,0x3E096244,0xBD7FB7CB,0xBED08DEB,0x3D9DBFB0,0x3EEBA711,0xBD4BBB71,0xBF3BB47F];
    pub const APC_ADVANCE_BITS: u32 = 0x43080000;
    pub const APC_V_PHASEMOD_OUT: &[u32] = &[0x3FC60181,0x3FA39D3D,0xC028130E,0xBF548683,0xC022CCFF,0xBFF7354D,0xBFE5DD77,0x401C8EA8,0x3E6F6020,0x400DAA9D,0x4051208F,0x403D7DE5,0xBE954A24,0x401A7746,0x3FFAEC2B,0x3FA50093,0x3F79E86E,0x3FD8B3EE,0x401CBA3A,0x3FBD3294,0x400CCD6D,0xC03C9D64,0xBF695855,0x3F701806];
    pub const APC_V_DIRCUR_OUT: &[u32] = &[0x3F32C220,0xBD2B7DB6,0xBF88BD44,0xBF88BD44,0xBF88BD44,0xBF42CC77,0xBF4605A0,0xBF4605A0,0xBF4605A0,0xBECFBF67,0x00000000,0x00000000,0x00000000,0xBEA81FDB,0x00000000,0x00000000,0x00000000,0x3E757AF9,0x00000000,0x00000000,0x00000000,0x3FB71500,0x3FB71500,0x3FB71500];
    pub const APCV_ADVANCE_BITS: u32 = 0x43080000;
    pub const APC2_PHASEMOD_OUT: &[u32] = &[0x3FC60181,0x3FA39D3D,0xC028130E,0xBF548683,0xC022CCFF,0xBFF7354D,0xBFE5DD77,0x401C8EA8,0x3E6F6020,0x400DAA9D,0x4032D21B,0x400E3801,0x3E8A547C,0x401A7746,0xBECAAE74,0x3DE34AF0,0x3F79E886,0x3FD8B3EE,0x3FD8DDA1,0x3FE3E690,0x40175483,0xC03C9D64,0xBF695855,0x3F701806];
    pub const APC2_DIRCUR_OUT: &[u32] = &[0x3F32C220,0xBD2B7DB6,0x401DD335,0x401DD335,0x401DD335,0xBF42CC77,0x3F985104,0x3F985104,0x3F985104,0xBECFBF67,0x3F90CFBE,0x3F90CFBE,0x3F90CFBE,0xBEA81FDB,0x00000000,0x00000000,0x00000000,0x3E757AF9,0xC04347DE,0xC04347DE,0xC04347DE,0x400DCE72,0x400DCE72,0x400DCE72];
    pub const APC2_DIRPREV_OUT: &[u32] = &[0xBF34F575,0x3F0CCD68,0x3E75F58F,0xBDBCBC72,0xBF097FCC,0x3F38E82B,0x3F2A8A36,0xBEAC7927,0x3E8C9331,0x3D8C1C3A,0x3F119D9C,0xBF12FD16,0xBE402DB0,0x3F304826,0x3E928909,0xBE1BA256,0x3F406A40,0x3E096244,0xBD7FB7CB,0xBED08DEB,0x3D9DBFB0,0x3EEBA711,0xBD4BBB71,0xBF3BB47F];
    pub const APC2_ADVANCE_BITS: u32 = 0xC1E00000;
    pub const APCGATE_ADVANCE_BITS: u32 = 0xC640E400;
    pub const APCGATE_PHASEMOD_OUT: &[u32] = &[0x3FC60181,0x3FA39D3D,0xC028130E,0xBF548683,0xC022CCFF,0xBFF7354D,0xBFE5DD77,0x401C8EA8,0x3E6F6020,0x400DAA9D,0x4032D21B,0x400E3801,0x3E8A547C,0x401A7746,0x3F661086,0x3F974A1B,0xBFE82F59,0x3FD8B3EE,0x3FD8DDA1,0x3FE3E690,0x40175483,0xC03C9D64,0xBF695855,0x3F701806];
    pub const RPT_B1_PHASE_OUT: &[u32] = &[0xC05CBC5C,0x3FAE774E,0x3FFE4C00,0x4018526A,0x406956B3,0xBFF6DA76,0xC07EDBB0,0x40810926,0x3EF40365,0xC02C3B90,0x402C2CAF,0x403720A3,0xC0386C8F,0xBEF00693,0xBFAA6B64,0x40389BD5,0x3F405649,0x3FE8A630,0xBFA7CEC8,0xC0120106];
    pub const RPT_B2_PHASE_OUT: &[u32] = &[0xC02FF896,0x3FC43F85,0x3FA98800,0x3FCB188D,0x401B8F22,0xBFA491A4,0xBDD8127D,0xC02A290C,0x3EA2ACEE,0xBFE5A4C0,0x3FE590E9,0x3FF42B84,0x4031FCE5,0x3F94F8A0,0xBFAA6B64,0x40389BD5,0x3F405649,0x3FE8A630,0xBFA7CEC8,0xC0120106];
    pub const RPT_B2_TABLE_OUT: &[u32] = &[0x3F800000,0x00000000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x00000000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x00000000,0x00000000,0x3F800000,0x3F800000,0x00000000,0x00000000];
    pub const RPT_B2C_PHASE_OUT: &[u32] = &[0xC02FF896,0x3FC43F85,0x3FA98800,0x3FCB188D,0x401B8F22,0xC0126079,0xBDD8127D,0xC02A290C,0xBCC1527E,0xBFE5A4C0,0x3F8AC644,0x40076810,0xC0386C8F,0xBEF00693,0xBFAA6B64,0x40389BD5,0x3F405649,0x3FE8A630,0xBFA7CEC8,0xC0120106];
    pub const RPT_B2C_TABLE_OUT: &[u32] = &[0x3F800000,0x00000000,0x3F800000,0x3F800000,0x3F800000,0x00000000,0x3F800000,0x00000000,0x00000000,0x3F800000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x3F800000,0x3F800000,0x00000000,0x00000000];
    pub const RPT_B2C_MAG: &[u32] = &[0x3F096896,0x3F0B36C6,0x3EF3BED0,0x3F0D8775,0x3F142E2A,0x3E572335,0x3E901BA2,0x3EF20F55,0x3F343DA6,0x3DAD5FC7,0x3E39FB38,0x3ECD79DD,0x3F0A8CE0,0x3F0FC31A,0x3F302273,0x3E6B34A3,0x3F084B1C,0x3EF7D8AC,0x3E9E0D81,0x3EF6FA3C];
    pub const RPT_B3_PHASE_OUT: &[u32] = &[0xC02FF896,0x3FC43F85,0x3DC735C8,0x3EC1E864,0x3F887CE5,0xC0126079,0xBDD8127D,0xC02A290C,0xBCC1527E,0xBC35EC45,0x3F8AC644,0x40076810,0xC0386C8F,0xBEF00693,0xBFAA6B64,0x40389BD5,0x3F405649,0x3FE8A630,0xBFA7CEC8,0xC0120106];
    pub const RPT_MAG: &[u32] = &[0x3EF88618,0x3EDB55D1,0x3F223714,0x3F1D3B7D,0x3F583168,0x3E032E2B,0x3F442526,0x3EBEAF3B,0x3E9BA8EB,0x3F406467,0x3EBC2E31,0x3E80EA67,0x3F383286,0x3E24EC2A,0x3F145AA8,0x3EF06204,0x3E19AC1A,0x3F35D9B1,0x3F129E0F,0x3F5BE29A];
    pub const RPT_B710: &[u32] = &[0x3F0C367A,0x3F0E0E18,0x3EF8B842,0x3F106AE0,0x3F173454,0x3E5B8731,0x3E930C86,0x3EF6FFF8,0x3F37EB50,0x3DB0E991,0x3E3DC6E0,0x3ED1AB5F,0x3F0D60BB,0x3F12B22F,0x3F33BAA9,0x3E700177,0x3F0B132C,0x3EFCE78B,0x3EA14740,0x3EFC0491];
    pub const RPT_MASK: &[u32] = &[0xC013283D,0x3F689F13,0x3FA98800,0x3FCB188D,0x401B8F22,0xBFA491A4,0xC029E7CB,0x402C0C33,0x3EA2ACEE,0xBFE5A4C0,0x3FE590E9,0x3FF42B84,0x4031FCE5,0x3F94F8A0,0x3EC21606,0xBFE2FC45,0x3E8A035C,0xC00BA1B0,0x3F7F231B,0xC0054394];
    pub const RPT_PHASE_IN: &[u32] = &[0xC02FF896,0x3FC43F85,0x3DC735C8,0x3EC1E864,0x3F887CE5,0xC0126079,0xBDD8127D,0xC02A290C,0xBCC1527E,0xBC35EC45,0x3F8AC644,0x40076810,0xC0386C8F,0xBEF00693,0xBFAA6B64,0x40389BD5,0x3F405649,0x3FE8A630,0xBFA7CEC8,0xC0120106];
    pub const RPT_TABLE_IN: &[u32] = &[0x3F800000,0x00000000,0x3F800000,0x3F800000,0x3F800000,0x00000000,0x3F800000,0x00000000,0x00000000,0x3F800000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x3F800000,0x3F800000,0x00000000,0x00000000];
    pub const AMPD_SRC0: &[u32] = &[0x3F3326BD,0x402A2B88,0xBF851C17,0x3F0D2A57];
    pub const AMPD_SRC1: &[u32] = &[0x3E550EEF,0x402BE249,0x3F779D6C,0x3F13A878];
    pub const AMPD_DST0: &[u32] = &[0xC021E3AD,0x4039ADEB,0x3EC271BF,0x3F8CAC1F];
    pub const AMPD_DST1: &[u32] = &[0xBFCD68B7,0x401A6535,0x40223EEF,0x3F0BB12A];
    pub const AMPD2_DST0_0: &[u32] = &[0xC0111E2F,0x4039ADEB,0x3EC271BF,0x3F8CAC1F];
    pub const AMPD2_DST1_0: &[u32] = &[0xBFEEF3B3,0x401A6535,0x40223EEF,0x3F0BB12A];
    pub const AMPD2_DST0_1: &[u32] = &[0xC021E3AD,0x40339323,0x3EC271BF,0x3F8CAC1F];
    pub const AMPD2_DST1_1: &[u32] = &[0xBFCD68B7,0x40207FFD,0x40223EEF,0x3F0BB12A];
    pub const AMPD2_DST0_2: &[u32] = &[0xC021E3AD,0x4039ADEB,0x3ED07A2D,0x3F8CAC1F];
    pub const AMPD2_DST1_2: &[u32] = &[0xBFCD68B7,0x401A6535,0x40207DE1,0x3F0BB12A];
    pub const AMPD2_DST0_3: &[u32] = &[0xC021E3AD,0x4039ADEB,0x3EC271BF,0x3F7DF014];
    pub const AMPD2_DST1_3: &[u32] = &[0xBFCD68B7,0x401A6535,0x40223EEF,0x3F271954];
    pub const AMPD_W_BITS: u32 = 0x3EBD70A4;
    pub const AMPD3_SRC: &[u32] = &[0x3F0AB2E8,0xBF34E326,0x3FA2EB98,0x3E8AB2E8,0xBEB4E326,0x3F22EB98,0xBF0AB2E8,0x3F34E326,0xBFA2EB98];
    pub const AMPD3_DST_IN: &[u32] = &[0xC0259595,0xBF76C13A,0xC0054AED,0xBF259595,0xBE76C13A,0xBF054AED,0xC0A59595,0xBFF6C13A,0xC0854AED];
    pub const AMPD3_DST_OUT: &[u32] = &[0xC0259595,0xBFAEC49F,0xC0054AED,0xBF259595,0xBF5D3E56,0xBF054AED,0xC0A59595,0xBF672C5A,0xC0854AED];
    pub const AMPD3_W_BITS: u32 = 0x3F19999A;
    pub const AMPDV_DST0: &[u32] = &[0xC0111E2F,0x40339323,0x3ED07A2D,0x3F7DF014];
    pub const AMPDV_DST1: &[u32] = &[0xBFEEF3B3,0x40207FFD,0x40207DE1,0x3F271954];
    pub const PULL_PM_IN: &[u32] = &[0x4020FCB0,0x4000DFCE,0xBF13DD01,0x3F6C81E8,0x3FF6702E,0x3FD87F84];
    pub const PULL_MASK: &[u32] = &[0xC002621B,0x3FE3CC21,0xBEFFAF07,0x400475C2,0xBF91D63B,0xC0061695];
    pub const PULL_PM_OUT: &[u32] = &[0x4020FCB0,0x4000DFCE,0xBFCBF2A9,0x3F6C81E8,0x3FF6702E,0x3FD87F84];
    pub const PULL_W_BITS: u32 = 0x3ED70A3D;
    }

    use golden::*;

    #[test]
    fn apc_tables_bit_patterns() {
        assert_eq!(
            bits(&APC_T_LO),
            [
                0x3FC0_0000,
                0x3FAC_CCCD,
                0x3F88_F5C3,
                0x3F85_1EB8,
                0x3F83_D70A
            ]
        );
        assert_eq!(
            bits(&APC_T_HI),
            [
                0x3FF3_3333,
                0x3FD9_999A,
                0x3F99_999A,
                0x3F8F_5C29,
                0x3F8C_CCCD
            ]
        );
        assert_eq!(
            bits(&APC_T_GMIX),
            [
                0x3F33_3333,
                0x3F33_3333,
                0x3F19_999A,
                0x3ECC_CCCD,
                0x3E4C_CCCD
            ]
        );
        assert_eq!(
            bits(&APC_T_LIN),
            [
                0x3F19_999A,
                0x3ECC_CCCD,
                0x3E4C_CCCD,
                0x3E19_999A,
                0x3DCC_CCCD
            ]
        );
        assert_eq!(
            bits(&APC_T_EA),
            [
                0x3FCC_CCCD,
                0x3FD9_999A,
                0x3ECC_CCCD,
                0x3F05_1EB8,
                0x3F0C_CCCD
            ]
        );
        assert_eq!(
            bits(&APC_T_EB),
            [
                0x4020_0000,
                0x4000_0000,
                0x3F4C_CCCD,
                0x3F0C_CCCD,
                0x3F0C_CCCD
            ]
        );
        assert_eq!(
            RPT_A2ORD,
            [0, 1, 2, 3, 4, 5, 7, 6, 8, 9, 11, 15, 12, 13, 14, 10]
        );
    }

    // ------------------------------------------------------------ wrap_pi_fused

    #[test]
    fn wrap_pi_fused_wraps_into_pi_range() {
        // fmsub form: fma(-rint(x*inv2pi), 2pi, x). The result is in [-pi, pi]
        // up to a few ulp (both `x` and `2pi` are rounded, so e.g. the wrap of a
        // near-multiple of pi can land one ulp outside).
        let tol = 4.0 * f32::EPSILON * PI_F;
        for i in -400..=400 {
            let x = i as f32 * 0.031_415_927;
            let r = wrap_pi_fused(x);
            assert!(r.is_finite());
            assert!((r.abs() - PI_F) <= tol, "x={x} r={r}");
            if r.abs() < PI_F {
                // idempotent strictly inside the range
                assert_eq!(wrap_pi_fused(r), r, "x={x} r={r}");
            }
        }
        // pi / -pi / 0 / exact multiples of 2pi
        assert_eq!(wrap_pi_fused(0.0), 0.0);
        assert_eq!(wrap_pi_fused(PI_F), PI_F);
        assert_eq!(wrap_pi_fused(-PI_F), -PI_F);
        assert_eq!(wrap_pi_fused(TWO_PI_F), 0.0);
        assert_eq!(wrap_pi_fused(NEG_2PI_F), 0.0);
    }

    // ------------------------------------------------------------- unwrap_phase

    #[test]
    fn unwrap_phase_matches_reference() {
        let mask = f32s(UWP_MASK);
        let mask_copy = f32s(UWP_MASKCOPY);
        let mag = f32s(UWP_MAG);
        let reg_offset = f32s(UWP_REGOFF);
        let mut scratch = f32s(UWP_SCRATCH_IN);
        let mut phase = f32s(UWP_PHASE_IN);
        unwrap_phase(
            &mask,
            &mask_copy,
            &mag,
            &[0, 5, 17],
            &[4, 16, 32],
            &[1, 9, 25],
            &reg_offset,
            &mut scratch,
            &mut phase,
            1024.0,
            2.0,
            2048.0,
            33,
            33,
        );
        assert_bits(&scratch, UWP_SCRATCH_OUT);
        assert_bits(&phase, UWP_PHASE_OUT);
    }

    #[test]
    fn unwrap_phase_empty_region_matches_reference() {
        let mask = f32s(UWP_MASK);
        let mask_copy = f32s(UWP_MASKCOPY);
        let mag = f32s(UWP_MAG);
        let reg_offset = f32s(UWP2_REGOFF);
        let mut scratch = f32s(UWP2_SCRATCH_IN);
        let mut phase = f32s(UWP2_PHASE_IN);
        unwrap_phase(
            &mask,
            &mask_copy,
            &mag,
            &[0, 32, 7, 5],
            &[1, 32, 6, 12],
            &[0, 30, 0, 9],
            &reg_offset,
            &mut scratch,
            &mut phase,
            1024.0,
            2.0,
            2048.0,
            33,
            33,
        );
        assert_bits(&scratch, UWP2_SCRATCH_OUT);
        assert_bits(&phase, UWP2_PHASE_OUT);
    }

    #[test]
    fn unwrap_phase_synthetic_spectrum_is_finite() {
        // small synthetic spectrum: 9 bins, two regions, a clean peak in each
        let nb = 9usize;
        let mut mag = vec![0.0f32; nb];
        for (i, m) in mag.iter_mut().enumerate() {
            *m = 0.1 + 0.2 * ((i as f32) - 3.0).abs().sin();
        }
        mag[3] = 1.0;
        mag[7] = 0.9;
        let mut mask = vec![0.0f32; nb];
        let mut mask_copy = vec![0.0f32; nb];
        for i in 0..nb {
            mask[i] = 0.3 * i as f32 - 1.0;
            mask_copy[i] = -0.7 * i as f32 + 0.5;
        }
        let mut scratch = vec![0.0f32; nb];
        let mut phase = vec![0.25f32; nb];
        unwrap_phase(
            &mask,
            &mask_copy,
            &mag,
            &[0, 5],
            &[4, 8],
            &[2, 7],
            &[0.5, -0.25],
            &mut scratch,
            &mut phase,
            512.0,
            1.0,
            1024.0,
            nb as i64,
            nb as i64,
        );
        assert!(scratch.iter().all(|v| v.is_finite()));
        assert!(phase.iter().all(|v| v.is_finite()));
        assert!(phase.iter().any(|v| *v != 0.25));
    }

    // ---------------------------------------------------- apply_pitch_coherence

    fn apc_args() -> ApcArgs {
        ApcArgs {
            precision: 2,
            trans_sens: 1.0,
            total_ratio: 1.6,
            transient_state: 0,
            n_fft: 1024,
            sr: 44100,
            f580: 255,
            f584: 256,
            acc_fc0: 137.0,
            coh_center: 0.5,
            seg_count: 5,
            peak_count: 6,
            vector_fmaf_region: false,
        }
    }

    const APC_PEAKS: [i64; 6] = [3, 7, 11, 15, 19, 22];
    const APC_RS: [i64; 6] = [2, 6, 10, 14, 18, 21];
    const APC_RE: [i64; 6] = [5, 9, 13, 17, 21, 24];
    const APC_BOUND: [i64; 6] = [0, 5, 9, 13, 17, 24];

    #[test]
    fn apply_pitch_coherence_matches_reference() {
        let env = f32s(APC_ENV);
        let phase = f32s(APC_PHASE);
        let regg = f32s(APC_REGG);
        let nw = f32s(APC_NOISEWT);
        let mut phase_mod = f32s(APC_PHASEMOD_IN);
        let mut dir_cur = f32s(APC_DIRCUR_IN);
        let mut dir_prev = f32s(APC_DIRPREV_IN);
        let advance = apply_pitch_coherence(
            &apc_args(),
            &env,
            &APC_PEAKS,
            &APC_RS,
            &APC_RE,
            &APC_BOUND,
            &phase,
            &mut phase_mod,
            &mut dir_cur,
            &mut dir_prev,
            &regg,
            &nw,
            1024.0,
            0.5,
        );
        assert_eq!(advance.to_bits(), APC_ADVANCE_BITS);
        assert_bits(&phase_mod, APC_PHASEMOD_OUT);
        assert_bits(&dir_cur, APC_DIRCUR_OUT);
        assert_bits(&dir_prev, APC_DIRPREV_OUT);
    }

    #[test]
    fn apply_pitch_coherence_vector_region_matches_reference() {
        let env = f32s(APC_ENV);
        let phase = f32s(APC_PHASE);
        let regg = f32s(APC_REGG);
        let nw = f32s(APC_NOISEWT);
        let mut phase_mod = f32s(APC_PHASEMOD_IN);
        let mut dir_cur = f32s(APC_DIRCUR_IN);
        let mut dir_prev = f32s(APC_DIRPREV_IN);
        let mut args = apc_args();
        args.vector_fmaf_region = true;
        let advance = apply_pitch_coherence(
            &args,
            &env,
            &APC_PEAKS,
            &APC_RS,
            &APC_RE,
            &APC_BOUND,
            &phase,
            &mut phase_mod,
            &mut dir_cur,
            &mut dir_prev,
            &regg,
            &nw,
            1024.0,
            0.5,
        );
        assert_eq!(advance.to_bits(), APCV_ADVANCE_BITS);
        assert_bits(&phase_mod, APC_V_PHASEMOD_OUT);
        assert_bits(&dir_cur, APC_V_DIRCUR_OUT);
    }

    #[test]
    fn apply_pitch_coherence_second_case_and_gate() {
        let env = f32s(APC_ENV);
        let phase = f32s(APC_PHASE);
        let regg = f32s(APC_REGG);
        let nw = f32s(APC_NOISEWT);
        let mut phase_mod = f32s(APC_PHASEMOD_IN);
        let mut dir_cur = f32s(APC_DIRCUR_IN);
        let mut dir_prev = f32s(APC_DIRPREV_IN);
        let mut args = apc_args();
        args.trans_sens = 0.75;
        args.total_ratio = 0.5;
        args.transient_state = 2;
        args.n_fft = 512;
        args.sr = 48000;
        args.f580 = 300;
        args.f584 = 288;
        args.acc_fc0 = -40.0;
        args.coh_center = 1.5;
        let advance = apply_pitch_coherence(
            &args,
            &env,
            &APC_PEAKS,
            &APC_RS,
            &APC_RE,
            &APC_BOUND,
            &phase,
            &mut phase_mod,
            &mut dir_cur,
            &mut dir_prev,
            &regg,
            &nw,
            512.0,
            0.9,
        );
        assert_eq!(advance.to_bits(), APC2_ADVANCE_BITS);
        assert_bits(&phase_mod, APC2_PHASEMOD_OUT);
        assert_bits(&dir_cur, APC2_DIRCUR_OUT);
        assert_bits(&dir_prev, APC2_DIRPREV_OUT);

        // precision > 9 gate: nothing is touched, advance stays -12345.
        let mut args = apc_args();
        args.precision = 12;
        let mut pm2 = f32s(APC_PHASEMOD_IN);
        let mut dc2 = f32s(APC_DIRCUR_IN);
        let mut dp2 = f32s(APC_DIRPREV_IN);
        let advance = apply_pitch_coherence(
            &args, &env, &APC_PEAKS, &APC_RS, &APC_RE, &APC_BOUND, &phase, &mut pm2, &mut dc2,
            &mut dp2, &regg, &nw, 1024.0, 0.5,
        );
        assert_eq!(advance.to_bits(), APCGATE_ADVANCE_BITS);
        assert_bits(&pm2, APCGATE_PHASEMOD_OUT);

        // trans_sens == 0 gate
        let mut args = apc_args();
        args.trans_sens = 0.0;
        let a = apply_pitch_coherence(
            &args, &env, &APC_PEAKS, &APC_RS, &APC_RE, &APC_BOUND, &phase, &mut pm2, &mut dc2,
            &mut dp2, &regg, &nw, 1024.0, 0.5,
        );
        assert_eq!(a, -12345.0);
    }

    #[test]
    fn apply_pitch_coherence_tiny_region_set_terminates_and_stays_finite() {
        // hand-built 8-bin spectrum, two disjoint peak regions, short seg bounds
        let nb = 8usize;
        let mut env = vec![0.0f32; 5];
        env[0] = 1.4;
        env[1] = 1.8;
        env[2] = 1.3;
        env[3] = 0.9;
        let mut phase = vec![0.0f32; nb];
        let mut phase_mod = vec![0.0f32; nb];
        let mut dir_cur = vec![0.0f32; nb];
        let mut dir_prev = vec![0.0f32; nb];
        let mut region_gain = vec![0.0f32; nb];
        let mut noise_wt = vec![0.0f32; nb];
        for i in 0..nb {
            phase[i] = 0.5 * i as f32 - 1.5;
            phase_mod[i] = -0.25 * i as f32;
            dir_cur[i] = 0.1 * i as f32;
            dir_prev[i] = -0.05 * i as f32;
            region_gain[i] = 0.3 * i as f32;
            noise_wt[i] = 0.1 * i as f32;
        }
        let args = ApcArgs {
            precision: 2,
            trans_sens: 0.5,
            total_ratio: 1.25,
            transient_state: 2,
            n_fft: 8,
            sr: 8000,
            f580: 2,
            f584: 4,
            acc_fc0: 1.0,
            coh_center: 0.75,
            seg_count: 4,
            peak_count: 2,
            vector_fmaf_region: false,
        };
        let advance = apply_pitch_coherence(
            &args,
            &env,
            &[2, 6],
            &[1, 5],
            &[3, 7],
            &[0, 3, 7, 7, 8],
            &phase,
            &mut phase_mod,
            &mut dir_cur,
            &mut dir_prev,
            &region_gain,
            &noise_wt,
            8.0,
            0.25,
        );
        assert!(advance.is_finite());
        assert!(phase_mod.iter().all(|v| v.is_finite()));
        assert!(dir_cur.iter().all(|v| v.is_finite()));
        assert!(dir_prev.iter().all(|v| v.is_finite()));
        assert!(phase_mod.iter().any(|v| *v != 0.0));
    }

    // ------------------------------------------ reset_phases_for_transients

    #[test]
    fn reset_phases_b1_matches_reference() {
        let mask = f32s(RPT_MASK);
        let mut phase = f32s(RPT_PHASE_IN);
        let mut table = f32s(RPT_TABLE_IN);
        reset_phases_for_transients(
            1,
            0,
            1.5,
            12,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0.0,
            0.0,
            0.0,
            false,
            &[],
            &[],
            &mask,
            &[],
            &[],
            &[],
            &mut phase,
            &mut table,
            20,
        );
        assert_bits(&phase, RPT_B1_PHASE_OUT);
        assert_bits(&table, RPT_TABLE_IN);
    }

    #[test]
    fn reset_phases_b2_matches_reference() {
        let mag = f32s(RPT_MAG);
        let b710 = f32s(RPT_B710);
        let mask = f32s(RPT_MASK);
        let mut phase = f32s(RPT_PHASE_IN);
        let mut table = f32s(RPT_TABLE_IN);
        reset_phases_for_transients(
            0,
            2,
            1.0,
            20,
            200,
            100,
            44100,
            0,
            4410,
            1024,
            20,
            0.0,
            2.5,
            0.0,
            false,
            &mag,
            &b710,
            &mask,
            &[2, 8],
            &[6, 14],
            &[4, 11],
            &mut phase,
            &mut table,
            20,
        );
        assert_bits(&phase, RPT_B2_PHASE_OUT);
        assert_bits(&table, RPT_B2_TABLE_OUT);
    }

    #[test]
    fn reset_phases_b2_conditional_copy_matches_reference() {
        let mag = f32s(RPT_B2C_MAG);
        let b710 = f32s(RPT_B710);
        let mask = f32s(RPT_MASK);
        let mut phase = f32s(RPT_PHASE_IN);
        let mut table = f32s(RPT_TABLE_IN);
        reset_phases_for_transients(
            0,
            2,
            1.0,
            20,
            0,
            0,
            44100,
            0,
            4410,
            1024,
            20,
            0.0,
            1.0,
            0.0,
            true,
            &mag,
            &b710,
            &mask,
            &[2, 8],
            &[6, 14],
            &[4, 11],
            &mut phase,
            &mut table,
            20,
        );
        assert_bits(&phase, RPT_B2C_PHASE_OUT);
        assert_bits(&table, RPT_B2C_TABLE_OUT);
    }

    #[test]
    fn reset_phases_b3_matches_reference() {
        let mag = f32s(RPT_MAG);
        let b710 = f32s(RPT_B710);
        let mask = f32s(RPT_MASK);
        let mut phase = f32s(RPT_PHASE_IN);
        let mut table = f32s(RPT_TABLE_IN);
        reset_phases_for_transients(
            0,
            0,
            1.0,
            20,
            0,
            0,
            44100,
            0,
            4410,
            1024,
            20,
            0.0,
            3.0,
            0.0,
            false,
            &mag,
            &b710,
            &mask,
            &[],
            &[],
            &[],
            &mut phase,
            &mut table,
            20,
        );
        assert_bits(&phase, RPT_B3_PHASE_OUT);
        assert_bits(&table, RPT_TABLE_IN);
    }

    // ------------------------------------------------ adjust_multiphase_diff

    #[test]
    fn adjust_multiphase_diff_nch2_matches_reference() {
        let s0 = f32s(AMPD_SRC0);
        let s1 = f32s(AMPD_SRC1);
        let d0 = f32s(AMPD_DST0);
        let d1 = f32s(AMPD_DST1);
        let w = f32::from_bits(AMPD_W_BITS);
        let want0: [&[u32]; 4] = [AMPD2_DST0_0, AMPD2_DST0_1, AMPD2_DST0_2, AMPD2_DST0_3];
        let want1: [&[u32]; 4] = [AMPD2_DST1_0, AMPD2_DST1_1, AMPD2_DST1_2, AMPD2_DST1_3];
        for k in 0..4i64 {
            let mut e0 = d0.clone();
            let mut e1 = d1.clone();
            adjust_multiphase_diff(
                &[&s0[..], &s1[..]],
                &mut [&mut e0[..], &mut e1[..]],
                k,
                k,
                w,
                2,
            );
            assert_bits(&e0, want0[k as usize]);
            assert_bits(&e1, want1[k as usize]);
        }
    }

    #[test]
    fn adjust_multiphase_diff_nch3_matches_reference() {
        let s0 = f32s(&AMPD3_SRC[0..3]);
        let s1 = f32s(&AMPD3_SRC[3..6]);
        let s2 = f32s(&AMPD3_SRC[6..9]);
        let mut d0 = f32s(&AMPD3_DST_IN[0..3]);
        let mut d1 = f32s(&AMPD3_DST_IN[3..6]);
        let mut d2 = f32s(&AMPD3_DST_IN[6..9]);
        adjust_multiphase_diff(
            &[&s0[..], &s1[..], &s2[..]],
            &mut [&mut d0[..], &mut d1[..], &mut d2[..]],
            1,
            1,
            f32::from_bits(AMPD3_W_BITS),
            3,
        );
        assert_bits(&d0, &AMPD3_DST_OUT[0..3]);
        assert_bits(&d1, &AMPD3_DST_OUT[3..6]);
        assert_bits(&d2, &AMPD3_DST_OUT[6..9]);
    }

    #[test]
    fn adjust_diff_nch2_vec_matches_reference() {
        let (o0, o1) = adjust_diff_nch2_vec(
            &f32s(AMPD_SRC0),
            &f32s(AMPD_SRC1),
            &f32s(AMPD_DST0),
            &f32s(AMPD_DST1),
            f32::from_bits(AMPD_W_BITS),
        );
        assert_bits(&o0, AMPDV_DST0);
        assert_bits(&o1, AMPDV_DST1);
    }

    #[test]
    fn nch2_vec_agrees_with_scalar_loop() {
        // The vectorised helper is the per-bin scalar body over a whole slice:
        // equal bit for bit for disjoint single-bin regions.
        let s0 = f32s(AMPD_SRC0);
        let s1 = f32s(AMPD_SRC1);
        let d0 = f32s(AMPD_DST0);
        let d1 = f32s(AMPD_DST1);
        let w = f32::from_bits(AMPD_W_BITS);
        let mut e0 = d0.clone();
        let mut e1 = d1.clone();
        for k in 0..4i64 {
            adjust_multiphase_diff(
                &[&s0[..], &s1[..]],
                &mut [&mut e0[..], &mut e1[..]],
                k,
                k,
                w,
                2,
            );
        }
        let (v0, v1) = adjust_diff_nch2_vec(&s0, &s1, &d0, &d1, w);
        assert_eq!(bits(&e0), bits(&v0));
        assert_eq!(bits(&e1), bits(&v1));
    }

    // ------------------------------------------------------- ampd_pull_to_peak

    #[test]
    fn ampd_pull_to_peak_matches_reference() {
        let mut pm = f32s(PULL_PM_IN);
        let mch = f32s(PULL_MASK);
        ampd_pull_to_peak(&mut pm, &mch, 2, 5, f32::from_bits(PULL_W_BITS));
        assert_bits(&pm, PULL_PM_OUT);
        // untouched outside `bin`
        assert_eq!(pm[0].to_bits(), PULL_PM_IN[0]);
        assert_eq!(pm[5].to_bits(), PULL_PM_IN[5]);
    }
}
