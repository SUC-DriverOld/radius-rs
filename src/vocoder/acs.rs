//! ACS (analysis channel spectrum) + `fill_granule.c` operators —
//! `analyze_channel_spectrum.c` / `fill_granule.c`, ported from
//! `pyradius/vocoder_ops.py` and the segment helpers of `pyradius/vocoder_core.py`.
//!
//! | Rust | Python | source line |
//! |---|---|---|
//! | [`time_to_iir_a`] | `time_to_iir_a` | `vocoder_ops.py:1617` |
//! | [`f32f`] | `_f32f` | `vocoder_ops.py:1637` |
//! | [`fill_granule`] | `fill_granule` | `vocoder_ops.py:1467` |
//! | [`acs_spectrum1`] | `acs_spectrum1` | `vocoder_ops.py:1668` |
//! | [`acs_spectrum2`] | `acs_spectrum2` | `vocoder_ops.py:1686` |
//! | [`cart_to_polar`] | `cart_to_polar` | `vocoder_ops.py:1708` |
//! | [`threshold_lt_inplace`] | `threshold_lt_inplace` | `vocoder_ops.py:1746` |
//! | [`seg_minmax`] | `_seg_minmax` | `vocoder_core.py:332` |
//! | [`seg_first_min`] | `_seg_first_min` | `vocoder_core.py:356` |
//!
//! Float discipline (see `lib.rs`): everything stays `f32` with the C op order;
//! `a*b + c` stays two operations (Rust never contracts without `mul_add`);
//! every site the Python sends through `_fma`/`_fma_arr` uses [`fma`], and the
//! `numpy` f32 ufuncs (`sqrt`) are evaluated in `f64` and rounded once, which is
//! how the C compiler lowers `sqrtf`.
//!
//! The reference runs with route B **off** (`PYR_FAST` unset, the default), so
//! the `numpy` reference bodies of the operators are the ones mirrored here; the
//! numba/NEON fast paths (`_fg_bandsum_nb`, `_fgw_nb`, `_ctp_nb`) were checked
//! bit-identical to those bodies on random data (`.acscheck/equiv.py`).
//!
//! The FFT is not ported: [`crate::fft::fwd`] is the bit-exact
//! `rx_fft_fwd` mirror, used here for `_fft_fwd` (`vocoder_ops.py:1660`).
//!
//! Numpy-only arguments dropped: `fill_granule`'s `mode` string is [`FillMode`],
//! and the buffers are flat slices (`ring` is `[n_bands, cap]`, `win` is
//! `[n_bands, 2*hop]`).

use crate::consts::PI_F;
use crate::fft;
use crate::vocoder::{floormod, fma};

/// `np.float32(pi/2)` — `PI_2_F` (`vocoder_ops.py:53`), bits `0x3FC90FDB`.
const PI_2_F: f32 = f32::from_bits(0x3FC9_0FDB);

/// `1e-12f` as `F32(1e-12)` in `acs_spectrum2` (`vocoder_ops.py:1697`).
const EPS1E12_F: f32 = f32::from_bits(0x2B8C_BCCC);

/// `_CTP_COEF` — CartToPolar Horner coefficients, `mov/movk` immediates at
/// `0x2BC58-0x2BCCC` (`vocoder_ops.py:1703`).
const CTP_COEF: [f32; 8] = [
    f32::from_bits(0x3B39_0CCD),
    f32::from_bits(0xBC82_B80D),
    f32::from_bits(0x3D2E_19B6),
    f32::from_bits(0xBD99_5FFA),
    f32::from_bits(0x3DD9_CCF2),
    f32::from_bits(0xBE11_6F9F),
    f32::from_bits(0x3E4C_B9A7),
    f32::from_bits(0xBEAA_AA5D),
];

/// `_f32f` (`vocoder_ops.py:1637`) — round a Python float to `f32`.
///
/// Not used by the operators in this file; kept for the route-B formant path
/// (`vocoder_ops.py:2730+`), which spells its scalar constants with it.
#[inline(always)]
pub fn f32f(x: f64) -> f32 {
    x as f32
}

/// `time_to_iir_a` (`vocoder_ops.py:1617`) — `AudioProcessor::TimeToIirA`
/// @dtk 0x19EAB8.
///
/// `F32(tau)` / `F32(rate)` are the caller's conversion: both arguments are
/// `numpy.float32` in the reference, and only the final `exp` is `math.exp`
/// (libm double) rounded back to `f32`.
#[inline]
pub fn time_to_iir_a(tau: f32, rate: f32) -> f32 {
    if tau == 0.0 {
        return 1.0;
    }
    let tr = tau * rate;
    let inv = (-1.0f32) / tr;
    1.0 - ((inv as f64).exp() as f32)
}

/// `_fft_fwd` (`vocoder_ops.py:1660`) — `transform::FFTFwd`, cart packing of
/// length `n + 2`.
///
/// The C harness hands the operator an `n_fft`-sample `DataTail`; a shorter
/// slice is zero-padded here (the Python reference would index out of bounds),
/// a longer one is truncated to its first `n` samples.
fn fft_fwd_n(n: usize, src: &[f32]) -> Vec<f32> {
    if src.len() == n {
        return fft::fwd(n, src);
    }
    let mut buf = vec![0.0f32; n];
    let k = src.len().min(n);
    buf[..k].copy_from_slice(&src[..k]);
    fft::fwd(n, &buf)
}

/// Result of [`acs_spectrum1`] — the `t1_*` entries of the Python dict.
#[derive(Debug, Clone)]
pub struct AcsSpectrum1 {
    /// `t1_io`: the windowed time buffer (a fresh array in the reference).
    pub t1_io: Vec<f32>,
    /// `t1_cart`: the packed `[n_fft + 2]` spectrum.
    pub t1_cart: Vec<f32>,
    /// `t1_env`: the updated cross-granule IIR envelope.
    pub t1_env: Vec<f32>,
}

/// `acs_spectrum1` (`vocoder_ops.py:1668`) — `rx_acs_spectrum1`: window
/// multiply, FFT, cross-granule IIR envelope update.
///
/// `time`/`win`/`env` may be longer than `n_fft`/`n_bins`; only
/// `n_win = min(time.len(), win.len())` samples are windowed and only the first
/// `n_bins` envelope bins are touched, exactly like the reference.
#[inline]
pub fn acs_spectrum1(
    time: &[f32],
    win: &[f32],
    env: &[f32],
    n_fft: usize,
    n_bins: usize,
    iir_a: f32,
) -> AcsSpectrum1 {
    let mut t_io = time.to_vec();
    let n_win = t_io.len().min(win.len());
    for i in 0..n_win {
        // (t_io * win).astype(f32) — one rounded f32 product.
        t_io[i] = t_io[i] * win[i];
    }
    let cart = fft_fwd_n(n_fft, &t_io);
    let mut env_out = env.to_vec();
    for k in 0..n_bins {
        let re = cart[2 * k];
        let im = cart[2 * k + 1];
        // sqrt((re*re + im*im).astype(f32)) — three separate f32 ops, then the
        // f32 sqrt (libm sqrtf == f64 sqrt rounded once).
        let s = re * re + im * im;
        let m = (s as f64).sqrt() as f32;
        // _fma_arr((m - env), F32(iir_a), env)
        let d = m - env_out[k];
        env_out[k] = fma(d, iir_a, env_out[k]);
    }
    AcsSpectrum1 {
        t1_io: t_io,
        t1_cart: cart,
        t1_env: env_out,
    }
}

/// Result of [`acs_spectrum2`] — the `t2_*` entries of the Python dict.
#[derive(Debug, Clone)]
pub struct AcsSpectrum2 {
    /// `t2_cart`: the packed `[n_fft + 2]` spectrum.
    pub t2_cart: Vec<f32>,
    /// `t2_mag`: `n_maxbin` magnitudes, lower-clamped to `1e-12`.
    pub t2_mag: Vec<f32>,
    /// `t2_phase`: `n_maxbin` phases, `[n_polar, n_maxbin)` left at zero.
    pub t2_phase: Vec<f32>,
}

/// `acs_spectrum2` (`vocoder_ops.py:1686`) — `rx_acs_spectrum2`: FFT,
/// `CartToPolar`, `Threshold_LT` (1e-12), zero tail.
///
/// Note the tail `[n_polar, n_maxbin)` of `t2_mag` is *not* zero after the
/// reference's lower clamp: it holds the fresh-buffer zeros raised to `1e-12`
/// (`vocoder_ops.py:1697`), while `t2_phase` really stays zero there.
#[inline]
pub fn acs_spectrum2(time2: &[f32], n_fft: usize, n_polar: usize, n_maxbin: usize) -> AcsSpectrum2 {
    let cart = fft_fwd_n(n_fft, time2);
    let mut mag = vec![0.0f32; n_maxbin];
    let mut phase = vec![0.0f32; n_maxbin];
    let (m, p) = cart_to_polar(&cart);
    mag[..n_polar].copy_from_slice(&m[..n_polar]);
    phase[..n_polar].copy_from_slice(&p[..n_polar]);
    threshold_lt_inplace(&mut mag, EPS1E12_F);
    AcsSpectrum2 {
        t2_cart: cart,
        t2_mag: mag,
        t2_phase: phase,
    }
}

/// `cart_to_polar` (`vocoder_ops.py:1708`) — dvaaccelerate `CartToPolar`, the
/// NEON polynomial `atan2`.
///
/// Returns `(mag, phase)`, both `cart.len() / 2` long (the C reads `(re, im)`
/// pairs for `k = 0..n-1` from an `[N+2]` cart buffer, i.e. it consumes
/// `2n = N+2` values — the last pair is bin `N/2` plus the pad slot).
///
/// The polynomial is the `_fma_arr` chain (correctly-rounded `fmaf`, as in the
/// `_ctp_nb` reference kernel), the magnitude is `sqrt(fma(im, im, re*re))`.
pub fn cart_to_polar(cart: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = cart.len() / 2;
    let mut mag = vec![0.0f32; n];
    let mut phase = vec![0.0f32; n];
    cart_to_polar_into(cart, &mut mag, &mut phase);
    (mag, phase)
}

/// In-place cartesian-to-polar conversion for the streaming vocoder.  The
/// wrapper above remains allocation-friendly for standalone callers, while the
/// render path can reuse each channel's persistent magnitude/phase buffers.
pub fn cart_to_polar_into(cart: &[f32], mag: &mut [f32], phase: &mut [f32]) {
    let n = cart.len() / 2;
    assert!(mag.len() >= n && phase.len() >= n);
    for k in 0..n {
        let re = cart[2 * k];
        let im = cart[2 * k + 1];
        let are = re.abs();
        let aim = im.abs();
        let mut mx = if are > aim { are } else { aim };
        if mx == 0.0 {
            mx = 1.0;
        }
        let mn = if aim > are { are } else { aim };
        let r = mn / mx;
        let r2 = r * r;
        // p = coef[0]*r2 + coef[1]; p = p*r2 + coef[2..7]; p = p*r2 + 1
        let mut p = fma(CTP_COEF[0], r2, CTP_COEF[1]);
        for c in &CTP_COEF[2..] {
            p = fma(p, r2, *c);
        }
        p = fma(p, r2, 1.0);
        // m = sqrt(_fma_arr(im, im, (re*re).astype(f32)))
        let re2 = re * re;
        mag[k] = (fma(im, im, re2) as f64).sqrt() as f32;
        let mut ang = r * p;
        if aim > are {
            ang = PI_2_F - ang;
        }
        if re < 0.0 {
            ang = PI_F - ang;
        }
        phase[k] = if im.is_sign_negative() { -ang } else { ang };
    }
}

/// `threshold_lt_inplace` (`vocoder_ops.py:1746`) — `Threshold_LT_InPlace`
/// (`vDSP_vthr`): a lower clamp, **not** a zeroing.
#[inline]
pub fn threshold_lt_inplace(v: &mut [f32], thr: f32) {
    for x in v.iter_mut() {
        if *x < thr {
            *x = thr;
        }
    }
}

/// `fill_granule` mode (`vocoder_ops.py:1467`): `"fg"`, `"fgw"` or `"chain"`.
///
/// The Python dispatcher falls through to the `fgw` body for any unrecognised
/// string; the enum makes that impossible (there is nothing to distinguish).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillMode {
    /// `mode == "fg"` — `FillGranule` (plain band sum into `acc`).
    Fg,
    /// `mode == "fgw"` — `FillGranuleWin` (two windowed wings per band).
    Fgw,
    /// `mode == "chain"` — `DataTail1D::Reset` + `fg` + `fgw`.
    Chain,
}

/// `fill_granule` (`vocoder_ops.py:1467`) — port of `rx_fill_granule` /
/// `rx_fill_granule_win` (modes `fg` / `fgw` / `chain`).
///
/// * `ring` — `[n_bands, cap]` flat, `cap = ring_cap0_a + ring_cap0_b`;
/// * `win` — `[n_bands, win_stride]` flat with `win_stride = win.len() / n_bands`
///   (that is exactly the shape of the engine's window table: `n_write` at
///   44.1 kHz, `2*hop` at 48 kHz); only `[..2*hop]` of each row is read;
/// * `acc` — the accumulator, `len() >= max(n_write, n5d0)`;
/// * `hop` / `cap_scalar` / `ring_cap0_*` / `cursor` are the C `int` geometry,
///   `d = cursor - hop`.
///
/// Returns the accumulator truncated to `n_write` (the Python `acc[:N]`), with
/// `acc` itself left holding the in-place result.
///
/// Faithfulness notes:
/// * the `buffered` `fg` band sum uses `(off + i) % cap` for both the
///   contiguous and the wrapping reference branches (identical index values);
/// * the `fgw` wing updates use [`fma`] for the `_fma_arr` / `_fgw_nb` sites,
///   and the plain `acc + (r*w)*g` shape for the `fgw` direct-branch gain
///   (`vocoder_ops.py:1600-1609` — the one place the C is *not* fused);
/// * the `fgw` lower wing is `n5d0 - hop` for the buffered/clamp branches but
///   `n_write - hop` for the direct branch (the head/tail mismatch documented
///   at `vocoder_ops.py:1541-1550`);
/// * a degenerate `cap_scalar == 0` reproduces `np.clip(v, 0, -1)` and the
///   negative-index wraparound that follows; `n5d0 < hop` panics where the
///   Python raises a broadcast `ValueError`.
#[allow(clippy::too_many_arguments)]
pub fn fill_granule<'a>(
    ring: &[f32],
    win: &[f32],
    acc: &'a mut [f32],
    mode: FillMode,
    hop: i64,
    n_write: usize,
    n_bands: usize,
    n5d0: usize,
    cursor: i64,
    buffered: bool,
    cap_scalar: i64,
    ring_cap0_a: i64,
    ring_cap0_b: i64,
    gain: f32,
) -> &'a mut [f32] {
    let n = n_write;
    let nb = n_bands;
    let cap = (ring_cap0_a + ring_cap0_b) as usize;
    let capi = cap as i64;
    let d = cursor - hop;
    // `win[band]` in the reference: a row of the engine's window table, whose
    // length is `n_write` at 44.1 kHz and `2*hop` at 48 kHz.
    let win_stride = if nb > 0 { win.len() / nb } else { 0 };

    if mode == FillMode::Chain {
        // DataTail1D::Reset (full zero), then FillGranule, then FGWin x n_bands.
        for v in acc.iter_mut() {
            *v = 0.0;
        }
        fill_granule(
            ring,
            win,
            &mut *acc,
            FillMode::Fg,
            hop,
            n,
            nb,
            n5d0,
            cursor,
            buffered,
            cap_scalar,
            ring_cap0_a,
            ring_cap0_b,
            gain,
        );
        fill_granule(
            ring,
            win,
            &mut *acc,
            FillMode::Fgw,
            hop,
            n,
            nb,
            n5d0,
            cursor,
            buffered,
            cap_scalar,
            ring_cap0_a,
            ring_cap0_b,
            gain,
        );
        return &mut acc[..n];
    }

    if mode == FillMode::Fg {
        if buffered {
            let off = floormod(d, capi);
            // band 0 seeds acc[:N], bands 1.. accumulate in f32 band by band.
            for i in 0..n {
                acc[i] = ring[(off + i as i64).rem_euclid(capi) as usize];
            }
            for k in 1..nb {
                let base = k * cap;
                for i in 0..n {
                    let j = (off + i as i64).rem_euclid(capi) as usize;
                    acc[i] = ring[base + j] + acc[i];
                }
            }
            return &mut acc[..n];
        }
        if d >= 0 && cursor < cap_scalar - hop {
            if nb < 1 || n < 1 {
                return &mut acc[..n];
            }
            let du = d as usize;
            for i in 0..n {
                acc[i] = ring[du + i];
            }
            for k in 1..nb {
                let base = k * cap;
                for i in 0..n {
                    acc[i] = ring[base + du + i] + acc[i];
                }
            }
            return &mut acc[..n];
        }
        if nb < 1 || n < 1 {
            return &mut acc[..n];
        }
        // idx = clip(arange(N) + d, 0, cap_scalar - 1)
        let capm1 = cap_scalar - 1;
        for i in 0..n {
            let idx = clipped_index(d + i as i64, capm1, cap);
            acc[i] = ring[idx];
            for k in 1..nb {
                acc[i] = ring[k * cap + idx] + acc[i];
            }
        }
        return &mut acc[..n];
    }

    // ---- mode == "fgw": FillGranuleWin over all bands ----
    let g = gain;
    for band in 0..nb {
        let r = &ring[band * cap..band * cap + cap];
        let w = &win[band * win_stride..band * win_stride + win_stride];
        if buffered {
            let off = floormod(d, capi) as usize;
            if n as i64 + off as i64 <= capi {
                if g == 1.0 {
                    // acc[:hop] = fma(r[off+hop : off+2hop], w[hop:2hop], acc)
                    // acc[N5-hop:N5] = fma(r[off:off+hop], w[0:hop], acc)
                    let lo = (n5d0 as i64 - hop) as usize;
                    let hopu = hop as usize;
                    for i in 0..hopu {
                        acc[i] = fma(r[off + hopu + i], w[hopu + i], acc[i]);
                        acc[lo + i] = fma(r[off + i], w[i], acc[lo + i]);
                    }
                } else if hop >= 1 {
                    let lo = (n5d0 as i64 - hop) as usize;
                    let hopu = hop as usize;
                    for i in 0..hopu {
                        // prod = (r*w).astype(f32); acc = fma(prod, g, acc)
                        let p1 = r[off + hopu + i] * w[hopu + i];
                        acc[i] = fma(p1, g, acc[i]);
                        let p2 = r[off + i] * w[i];
                        acc[lo + i] = fma(p2, g, acc[lo + i]);
                    }
                }
            } else if hop >= 1 {
                let lo = (n5d0 as i64 - hop) as usize;
                let hopu = hop as usize;
                for i in 0..hopu {
                    let mut i1 = off + hopu + i;
                    if i1 >= cap {
                        i1 -= cap;
                    }
                    let mut i2 = off + i;
                    if i2 >= cap {
                        i2 -= cap;
                    }
                    // p = (r*w).astype(f32) * g, then a plain f32 add
                    let p1 = (r[i1] * w[hopu + i]) * g;
                    let p2 = (r[i2] * w[i]) * g;
                    acc[i] = acc[i] + p1;
                    acc[lo + i] = acc[lo + i] + p2;
                }
            }
            continue;
        }
        if hop < 1 {
            continue;
        }
        let hopu = hop as usize;
        let capm1 = cap_scalar - 1;
        if cursor < hop || cursor >= cap_scalar - hop {
            let lo = (n5d0 as i64 - hop) as usize;
            for i in 0..hopu {
                let c1 = clipped_index(cursor + i as i64, capm1, cap);
                let c2 = clipped_index(d + i as i64, capm1, cap);
                let p1 = (r[c1] * w[hopu + i]) * g;
                let p2 = (r[c2] * w[i]) * g;
                acc[i] = acc[i] + p1;
                acc[lo + i] = acc[lo + i] + p2;
            }
            continue;
        }
        // direct branch: lower wing targets n_write - hop, not n5d0 - hop.
        let lo_direct = (n as i64 - hop) as usize;
        let cu = cursor as usize;
        let du = d as usize;
        for i in 0..hopu {
            let p1 = r[cu + i] * w[hopu + i];
            let p2 = r[du + i] * w[i];
            if g == 1.0 {
                acc[i] = acc[i] + p1;
                acc[lo_direct + i] = acc[lo_direct + i] + p2;
            } else {
                // the direct scalar tail is a non-fused add-product:
                // acc += (r*w)*g  (only the buffered/clamp branches use the fma)
                acc[i] = acc[i] + (p1 * g);
                acc[lo_direct + i] = acc[lo_direct + i] + (p2 * g);
            }
        }
    }
    &mut acc[..n]
}

/// `np.clip(v, 0, cap_scalar - 1)` followed by the reference's indexing of the
/// row: for the degenerate `cap_scalar == 0` the clip yields `-1` and `numpy`
/// reads the last element of the row.
#[inline(always)]
fn clipped_index(v: i64, capm1: i64, row_len: usize) -> usize {
    let mut i = if v < 0 {
        0
    } else if v > capm1 {
        capm1
    } else {
        v
    };
    if i < 0 {
        i += row_len as i64;
    }
    i as usize
}

/// Index element accepted by the segment helpers: the reference passes `int64`
/// arrays (`vocoder_core.py:846`), in-crate callers usually hold `usize` bin
/// indices.
pub trait SegIdx: Copy {
    /// Widen to the reference's `int64`.
    fn to_i64(self) -> i64;
}

impl SegIdx for i64 {
    #[inline(always)]
    fn to_i64(self) -> i64 {
        self
    }
}

impl SegIdx for i32 {
    #[inline(always)]
    fn to_i64(self) -> i64 {
        self as i64
    }
}

impl SegIdx for usize {
    #[inline(always)]
    fn to_i64(self) -> i64 {
        self as i64
    }
}

/// One segment of [`seg_minmax`] — the per-region form callers walk one region
/// at a time (`vocoder_core.py:855-858` guards the empty regions first, and the
/// reference only ever calls the vectorised form).
///
/// A `[start, end)` window of length 0 yields `x[start]`, which is what
/// `np.minimum.reduceat` returns for a repeated index.
#[inline]
pub fn seg_minmax_span<T: SegIdx>(x: &[f32], start: T, end: T) -> (f32, f32) {
    let start = start.to_i64();
    let end = end.to_i64();
    let s = start as usize;
    if end <= start {
        let v = x[s];
        return (v, v);
    }
    let mut lo = x[s];
    let mut hi = lo;
    for i in (s + 1)..(end as usize) {
        let v = x[i];
        if v <= lo {
            lo = v;
        }
        if v >= hi {
            hi = v;
        }
    }
    (lo, hi)
}

/// `_seg_minmax` (`vocoder_core.py:332`) — per-segment min/max over each
/// `[start, end)` window (C `ev_find_peaks`).
///
/// The reference builds these with `np.minimum.reduceat` on contiguous
/// segments (its last segment runs to `ends[-1]`, hence the truncation); for a
/// segment of length 0 `reduceat` yields `x[start]`, which is reproduced here.
/// Ties follow `numpy`'s "last element wins" reduction rule; unlike
/// `np.minimum`/`np.maximum`, a `NaN` operand does not poison the fold (the
/// engine's envelope is finite).
pub fn seg_minmax<T: SegIdx>(x: &[f32], starts: &[T], ends: &[T]) -> (Vec<f32>, Vec<f32>) {
    let ng = starts.len();
    let mut mn = Vec::with_capacity(ng);
    let mut mx = Vec::with_capacity(ng);
    for r in 0..ng {
        let (lo, hi) = seg_minmax_span(x, starts[r].to_i64(), ends[r].to_i64());
        mn.push(lo);
        mx.push(hi);
    }
    (mn, mx)
}

/// `_seg_first_min` (`vocoder_core.py:356`) — first index of the minimum over
/// each `[start, end)` window (C `MinIndex`).
///
/// `MinIndex` uses a strict `<`, so the FIRST occurrence wins (a naive
/// `argmin` picks the last). Regions whose minimum matches nothing keep the
/// scatter's `0` default, exactly like the reference (`vocoder_core.py:395`).
///
/// Empty segments raise in the Python (`vocoder_core.py:378`) and panic here.
pub fn seg_first_min<T: SegIdx>(x: &[f32], starts: &[T], ends: &[T]) -> Vec<i64> {
    let ng = starts.len();
    let mut out = vec![0i64; ng];
    for r in 0..ng {
        let s = starts[r].to_i64();
        let e = ends[r].to_i64();
        if e <= s {
            panic!("empty segment in seg_first_min");
        }
        let (lo, _) = seg_minmax_span(x, s, e);
        for i in (s as usize)..(e as usize) {
            if x[i] == lo {
                out[r] = i as i64;
                break;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `cart_to_polar` must invert a `(mag, phase)` polar construction.
    #[test]
    fn cart_to_polar_round_trip() {
        let npair = 33usize;
        let mut cart = vec![0.0f32; 2 * npair];
        let mut want_m = vec![0.0f32; npair];
        let mut want_p = vec![0.0f32; npair];
        for k in 0..npair {
            let m = 0.125 + 0.75 * (k as f32 / npair as f32);
            let p = -3.0 + 6.0 * ((k * 7 % npair) as f32 / npair as f32);
            want_m[k] = m;
            want_p[k] = p;
            cart[2 * k] = m * (p as f64).cos() as f32;
            cart[2 * k + 1] = m * (p as f64).sin() as f32;
        }
        let (mag, phase) = cart_to_polar(&cart);
        assert_eq!(mag.len(), npair);
        assert_eq!(phase.len(), npair);
        for k in 0..npair {
            let dm = (mag[k] - want_m[k]).abs() / want_m[k];
            assert!(dm < 1e-4, "mag[{k}]={} want {}", mag[k], want_m[k]);
            let dphi = crate::vocoder::wrap_pi(phase[k] - want_p[k]);
            assert!(
                dphi.abs() < 2e-3,
                "phase[{k}]={} want {}",
                phase[k],
                want_p[k]
            );
            // the polar pair really is the cart pair
            let re = mag[k] * (phase[k] as f64).cos() as f32;
            assert!((re - cart[2 * k]).abs() < 1e-3 * want_m[k] + 1e-6);
        }
        // the zero vector: mx is forced to 1.0, so mag = 0 and ang = 0
        let z = cart_to_polar(&[0.0, 0.0, -0.0, -0.0]);
        assert_eq!(z.0, vec![0.0, 0.0]);
        assert_eq!(z.1, vec![0.0, -0.0]);
    }

    /// `time_to_iir_a` is a one-pole coefficient: it must land in `(0, 1]`.
    #[test]
    fn time_to_iir_a_finite_in_unit() {
        for &tau in &[1e-5f32, 1e-4, 0.001, 0.01, 0.1, 1.0, 10.0] {
            for &rate in &[44100.0f32, 48000.0, 1234.0] {
                let a = time_to_iir_a(tau, rate);
                assert!(a.is_finite(), "tau={tau} rate={rate} a={a}");
                assert!(a > 0.0 && a <= 1.0, "tau={tau} rate={rate} a={a}");
            }
        }
        assert_eq!(time_to_iir_a(0.0, 48000.0), 1.0);
        // shorter time constants smooth less
        assert!(time_to_iir_a(0.01, 48000.0) < time_to_iir_a(0.001, 48000.0));
    }

    /// `fill_granule` mode `fg` on a small buffer: it runs, is finite, and is
    /// exactly the per-band sum of the ring row it addresses.
    #[test]
    fn fill_granule_fg_small_buffer_is_finite() {
        let cap = 32usize;
        let nb = 3usize;
        let n = 16usize;
        let hop = 8i64;
        let mut ring = vec![0.0f32; nb * cap];
        for (i, v) in ring.iter_mut().enumerate() {
            *v = ((i as f32) * 0.37).sin() * 0.5;
        }
        let mut acc = vec![0.25f32; n];
        let out = fill_granule(
            &ring,
            &[],
            &mut acc,
            FillMode::Fg,
            hop,
            n,
            nb,
            0,
            20,
            true,
            cap as i64,
            cap as i64,
            0,
            1.0,
        );
        assert_eq!(out.len(), n);
        assert!(out.iter().all(|v| v.is_finite()));
        let off = (20i64 - hop) as usize % cap;
        for i in 0..n {
            let j = (off + i) % cap;
            let want = ring[j] + ring[cap + j] + ring[2 * cap + j];
            assert_eq!(out[i].to_bits(), want.to_bits(), "bin {i}");
        }
    }

    const T2A_TAU: [u32; 6] = [
        0x3C23D70A, 0x00000000, 0x3BA3D70A, 0x38D1B717, 0x40200000, 0x3E99999A,
    ];
    const T2A_RATE: [u32; 6] = [
        0x473B8000, 0x473B8000, 0x472C4400, 0x473B8000, 0x473B8000, 0x449A4000,
    ];
    const T2A_OUT: [u32; 6] = [
        0x3B086400, 0x3F800000, 0x3B944580, 0x3E4093C4, 0x370C0000, 0x3B30CA00,
    ];
    const CTP_CART: [u32; 34] = [
        0x00000000, 0x80000000, 0x00000000, 0x00000000, 0xBFC00000, 0x3FC00000, 0xBE325FE4,
        0x3F8804F5, 0xBF16DA8A, 0x3EE3841F, 0xBF466DD0, 0x3F666130, 0xBF2560B0, 0x3CC01E58,
        0xBE98909E, 0x3E9353FA, 0xBE270C94, 0xBE3CBB8D, 0xBE9D2740, 0x3EDB7A08, 0xBF8A8B0C,
        0xBF46693E, 0x3E85E707, 0xBF14A91A, 0x3EA1674E, 0xBE598F6B, 0x3E65E3E1, 0xBF47C265,
        0x3EFD61EB, 0x3EB701E5, 0x3F214084, 0x3E767190, 0x3EE42477, 0xBE8DB4BC,
    ];
    const CTP_MAG: [u32; 17] = [
        0x00000000, 0x00000000, 0x4007C3B6, 0x3F89D5B2, 0x3F3CF03C, 0x3F9806E9, 0x3F257C94,
        0x3ED416B3, 0x3E7C0AE0, 0x3F06F838, 0x3FAA663D, 0x3F230A8B, 0x3EC2A380, 0x3F4FDCD8,
        0x3F1C47D0, 0x3F2C9F0E, 0x3F0648E5,
    ];
    const CTP_PHASE: [u32; 17] = [
        0x80000000, 0x00000000, 0x4016CBE4, 0x3FDDDBF6, 0x401FB5E0, 0x4012095E, 0x4046BD54,
        0x4017E9F8, 0xC012E696, 0x400C4D13, 0xC0214A6E, 0xBF92E5E2, 0xBF17D14C, 0xBFA53423,
        0x3F20214F, 0x3EBADC47, 0xBF0E49A4,
    ];
    const THR_IN: [u32; 8] = [
        0x00000000, 0xA9E12E13, 0x2B8CBCCC, 0x2C0CBCCC, 0xBF000000, 0x3F000000, 0x80000000,
        0x0DA24260,
    ];
    const THR_OUT: [u32; 8] = [
        0x2B8CBCCC, 0x2B8CBCCC, 0x2B8CBCCC, 0x2C0CBCCC, 0x2B8CBCCC, 0x3F000000, 0x2B8CBCCC,
        0x2B8CBCCC,
    ];
    const ACS1_TIME: [u32; 32] = [
        0x3D9A4EBB, 0xBE86AC0F, 0xBDB9AF90, 0xBD232005, 0xBE9D292F, 0x3D0784C6, 0x3D5D373F,
        0x3EF1E11C, 0x3DDDDAE4, 0x3E531164, 0xBDF1B267, 0xBE753CA5, 0xBE9A10EC, 0xBE062E51,
        0x3E037803, 0xBE5B41AE, 0xBE3C929E, 0xBDC7CCC1, 0x3E815F91, 0x3EB6CE14, 0xBE83F5AB,
        0xBDCF8B9B, 0xBE41B598, 0xBE78184C, 0xBDDEB459, 0xBD0D6555, 0x3F10ACB1, 0xBE7B5C08,
        0x3E87134E, 0xBCF84C25, 0x3D93BE38, 0x3EA44D76,
    ];
    const ACS1_WIN: [u32; 24] = [
        0xBFB1C62E, 0xBF8DE4D2, 0x3E3C1D4E, 0xBF82AA3B, 0x3EC025C3, 0xBE2B5A07, 0xBD91C195,
        0xBF9D1FD3, 0xBEEB5FD8, 0x3F19673E, 0x3EB2A14E, 0xBE41E8BA, 0x3E81B70F, 0xBEA61E9E,
        0x3C456800, 0xBE1AE731, 0xBFF95DB3, 0xBFAEDC57, 0x3E973C91, 0x3E0CE797, 0x3E4875EE,
        0xBF492FC8, 0xBE09EB2C, 0xBEC7FE13,
    ];
    const ACS1_ENV: [u32; 17] = [
        0x3EE8B4A7, 0x3E15B774, 0x3D72880D, 0x3E61ED1F, 0x3EFFDFA4, 0x3E9CAE39, 0x3D6A2D0A,
        0x3D3BDD50, 0x3E361DF7, 0x3D8BF7E0, 0x3DA63AF1, 0x3ED18E23, 0x3DD49030, 0x3E8DAB48,
        0x3CF14831, 0x3D0FF0BF, 0x3E4A756B,
    ];
    const ACS1_IO: [u32; 32] = [
        0xBDD64FC8, 0x3E954A48, 0xBC88722F, 0x3D268577, 0xBDEBEC24, 0xBBB56AB1, 0xBB7BE724,
        0xBF147520, 0xBD4BFAFF, 0x3DFCF508, 0xBD28A656, 0x3D39C1AA, 0xBD9C2164, 0x3D2E242A,
        0x3ACAC186, 0x3D04AB85, 0x3EB7AFA1, 0x3E087923, 0x3D98DBF4, 0x3D493C25, 0xBD4EA96F,
        0x3DA31B59, 0x3CD0B843, 0x3DC1D11E, 0xBDDEB459, 0xBD0D6555, 0x3F10ACB1, 0xBE7B5C08,
        0x3E87134E, 0xBCF84C25, 0x3D93BE38, 0x3EA44D76,
    ];
    const ACS1_CART: [u32; 34] = [
        0x3F938909, 0x00000000, 0xBD8AB9C4, 0x3F99D2DE, 0x3FA568A2, 0x3F163E8E, 0x3CA0E480,
        0xBE55E062, 0x3EB8A74E, 0xBF4DE303, 0xBF0E033C, 0x3E14FDCA, 0x3F71B325, 0x3F1CB4FC,
        0x3E29DDCE, 0x3EC6B516, 0xBF0F4C41, 0xBF57AC86, 0xBF8DA86F, 0xBEE95333, 0x3EEEE0F8,
        0x3F160414, 0x3EA41F3A, 0x3F82F8B8, 0xBE5A7CB2, 0x3E2D0DA6, 0xC01ABCEC, 0xBF363C24,
        0xBF86B24C, 0xBF5A28AF, 0xBD87EBD2, 0xBF0D1DA5, 0x3EDFB4BC, 0x00000000,
    ];
    const ACS1_ENVOUT: [u32; 17] = [
        0x3EEA3120, 0x3E1A3812, 0x3D84D8FC, 0x3E61E14D, 0x3F0057D2, 0x3E9D4000, 0x3D7C5DB6,
        0x3D42458C, 0x3E39AAA0, 0x3D959408, 0x3DABEAEC, 0x3ED2F774, 0x3DD5FF4A, 0x3E927219,
        0x3D0F338E, 0x3D18CC43, 0x3E4B7A29,
    ];
    const ACS1_IIR_A: f32 = f32::from_bits(0x3B884000);
    const ACS2_TIME: [u32; 32] = [
        0x3EFE6E5A, 0xBE9FF3B6, 0xBC230C2F, 0xBF499EBA, 0xBE850CB7, 0x3E96C845, 0x3F80F073,
        0x3EC72E28, 0xBE483714, 0x3F27AA7D, 0xBD520E25, 0x3E3DD1B9, 0xBDFF8D87, 0x3D6EA98C,
        0x3EE8212E, 0xBD01FB6D, 0xBEF20B61, 0xBE9BE24B, 0x3D0D1A37, 0xBF1DE84F, 0xBE9327A0,
        0x3E84DABE, 0xBDA07172, 0xBE8004CF, 0xBF8E3187, 0xBF1306FD, 0x3E402FC3, 0x3ECF943A,
        0x3F060913, 0x3F22AA11, 0xBA90ED9A, 0xBE19B584,
    ];
    const ACS2_CART: [u32; 34] = [
        0xBD099040, 0x00000000, 0x3FDDFA59, 0xC02CCFF0, 0xBF3C8AAC, 0x40337F14, 0xBF83C8D6,
        0x4063EB33, 0xBFECD160, 0x4041B6D8, 0x3F7F29AF, 0xC024C5CA, 0x404491BC, 0xBFD6262C,
        0x3FF8980C, 0xBF39C541, 0xC03E435F, 0xBFC8AFCA, 0x3F5C6F41, 0x3E4CBFB6, 0x3ED8315C,
        0x3F3C2B33, 0x40343CB2, 0x3E79A6A0, 0xBED5ACB4, 0x3F1606B6, 0xBF50FFF6, 0x3F76550C,
        0x40242D16, 0x3FB6D6CC, 0x3FA06404, 0x3E061C98, 0x3E8356E8, 0x00000000,
    ];
    const ACS2_MAG: [u32; 20] = [
        0x3D099040, 0x404D6254, 0x40399503, 0x406D40A3, 0x40630984, 0x4030B093, 0x405FD718,
        0x4004B0A8, 0x40571A1B, 0x3F624CB7, 0x3F5901F4, 0x4034E945, 0x3F382DF1, 0x3FA1863E,
        0x403BE9C0, 0x3FA143AF, 0x3E8356E8, 0x2B8CBCCC, 0x2B8CBCCC, 0x2B8CBCCC,
    ];
    const ACS2_PHASE: [u32; 20] = [
        0x40490FDB, 0xBF7FF89F, 0x3FE9EEAF, 0x3FED15C8, 0x4007A516, 0xBF99C830, 0xBEFF5FD0,
        0xBEB714BA, 0xC02A005D, 0x3E69A48E, 0x3F865190, 0x3DB0DAE6, 0x400C229F, 0x40118F94,
        0x3F021168, 0x3DD54819, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    ];
    const FG_RING: [u32; 64] = [
        0x3DA62156, 0x3EF63D81, 0x3C57EE1E, 0x3DC56FF0, 0xBF74257D, 0x00000000, 0xBE9FEA47,
        0xBE939F0F, 0x3E99088F, 0xBEE95402, 0x3CB2591E, 0xBDA1A648, 0xBF0039E3, 0xBF7F9CDD,
        0xBF22C6C8, 0xBF2A89F5, 0xBDEA018F, 0x3E933444, 0x3E9E0FF1, 0x3E7F3BFF, 0x3EE10B6B,
        0xBE45EE98, 0x3F086CC4, 0x3F246C74, 0xBDBA5676, 0x3C91E813, 0xBEE01E40, 0xBEA8F255,
        0x3DB91534, 0xBDEC8F11, 0xBE9FA806, 0x3DBA79E5, 0x3F05F874, 0xBE5BAA20, 0xBF2D9B24,
        0x3EE5389D, 0x3E677158, 0xBD7222D0, 0x80000000, 0xBE5298DA, 0x3D7616AC, 0x3EF999AD,
        0x3DF89AE4, 0xBC41F77F, 0x3F54DAB2, 0xBF11D1EC, 0xBE811675, 0xBD3D6E8A, 0x3F1D27E3,
        0x3EC13C21, 0xBF5D770A, 0xBE852ACC, 0xBE63902F, 0x3E1A2291, 0x3EE4BBD2, 0x3E19FEC9,
        0xBE74EE3D, 0xBEFD5A0A, 0xBF0CF352, 0xBEA9E40C, 0x3E696D8B, 0xBE9135E2, 0xBF63F800,
        0xBE5ED021,
    ];
    const FG_WIN: [u32; 32] = [
        0x3FA2511A, 0x3F18ACD1, 0x3EE33AAF, 0xBFA8A139, 0x3F975C85, 0xBF0079EC, 0x3DCD05D7,
        0x3DFF66A5, 0xBDD65C41, 0xBEDF335F, 0xBE914EA0, 0x3E858648, 0xBDC0B273, 0xBF4C8CEA,
        0x3E86997D, 0x3EE220C5, 0x3F5168AF, 0x3E1FEE77, 0xBF7047DA, 0xBF357BDA, 0x3F0D36F7,
        0xBF70A62D, 0xBF842A51, 0x3E179E6B, 0xBF98C7A5, 0xBF14E5BF, 0x3F953A94, 0x3E9995AE,
        0x3F212FC6, 0xBE49C531, 0xBF04D785, 0x3DE7D876,
    ];
    const FG_ACC: [u32; 16] = [
        0xBE88C16B, 0x3EB57E04, 0x3E08B9E9, 0x3E31DAF6, 0xBD0FD6F4, 0xBE8E966E, 0xBDF4DD2A,
        0x3F0F6135, 0xBF2D96D0, 0x3ECF7EF3, 0xBF5E51CD, 0xBE0AD74B, 0xBEE940CA, 0x3E8C0FE1,
        0xBF3DEBE3, 0xBDC1B6CF,
    ];
    const FG_OUT: [u32; 256] = [
        0x3EA9419E, 0xBFC8B764, 0xBF635202, 0xBF3660DE, 0x3EFFCF62, 0x3F2A3832, 0xBF0E6F12,
        0xBC319990, 0x3E5E86A7, 0xBD2F301C, 0x3F7ACAAD, 0x3F4AEC26, 0xBEA90CBC, 0xBEF43B89,
        0xBF7D0272, 0xBF296B30, 0x3E5E86A7, 0xBD2F301C, 0x3F7ACAAD, 0x3F4AEC26, 0xBEA90CBC,
        0xBEF43B89, 0xBF7D0272, 0xBF296B30, 0x3EA2FC12, 0xBECC59A6, 0xBF99E602, 0xBE01932E,
        0x3F1ABC9F, 0x3E886871, 0xBF2A3B6C, 0x3F0B4A4C, 0x3EA9419E, 0xBFC8B764, 0xBF635202,
        0xBF3660DE, 0x3EFFCF62, 0x3F2A3832, 0xBF0E6F12, 0xBC319990, 0x3E5E86A7, 0xBD2F301C,
        0x3F7ACAAD, 0x3F4AEC26, 0xBEA90CBC, 0xBEF43B89, 0xBF7D0272, 0xBF296B30, 0x3F7ACAAD,
        0x3F4AEC26, 0xBEA90CBC, 0xBEF43B89, 0xBF7D0272, 0xBF296B30, 0x3EA2FC12, 0xBECC59A6,
        0xBF99E602, 0xBE01932E, 0xBE01932E, 0xBE01932E, 0xBE01932E, 0xBE01932E, 0xBE01932E,
        0xBE01932E, 0xBD440363, 0x3EB3CF80, 0x3F00CC62, 0x3EC5C8E6, 0xBE35691B, 0xBE47E818,
        0x3D512ED6, 0x3EC0E914, 0xBF2213BC, 0xBE8EF10B, 0xBF69FAF2, 0x3F465AB9, 0xBE810D1C,
        0xBE66EC55, 0x3E3AA5F2, 0xBDD0ECAF, 0xBDE8B613, 0x3EB450A7, 0x3EC8D39F, 0x3EA520AD,
        0xBE09C67F, 0xBE617CB9, 0xB8FB036E, 0x3EDD10AE, 0xBF2587DC, 0xBD973DC5, 0xBF667B67,
        0x3F006F5C, 0xBEA04FD1, 0xBD9B37CF, 0xBDC28103, 0xBDCC5C85, 0xBF0C742A, 0x3F11DDC0,
        0xBF50F0BC, 0x3E07556A, 0x3E92D13D, 0xBF1ED962, 0x3E717644, 0x3F274114, 0xBE9ADF78,
        0x3EA082B0, 0xBF868F0A, 0xBF8B4F17, 0xBF31F26E, 0x3F3AD294, 0xBE5E68C8, 0xBE3D4E02,
        0xBEEDA974, 0x3F015487, 0xBF0800F8, 0x3E141714, 0x3E42C19C, 0xBF049522, 0x3E044B36,
        0x3F201785, 0xBED49084, 0x3EAE9B2B, 0xBF7F13FF, 0xBF4D7212, 0xBF1F8D06, 0x3F17C8FC,
        0xBEBFCBD0, 0xBE219206, 0xBF7CE56B, 0x3C1DB9B0, 0xBF767156, 0x3E24868C, 0xBE5D973E,
        0xBE1D64BE, 0xBE585BF3, 0x3F5C5B53, 0xBE7FE861, 0x3E56A698, 0xBF790374, 0xBCBE2206,
        0xBF16D736, 0x3FA7B1C8, 0xBF0B9295, 0xBE3CF065, 0xBF458A68, 0x3DE79776, 0xBF224164,
        0x3E288645, 0xBE25E6CA, 0xBE43BA61, 0xBE3C2E57, 0x3F45437D, 0xBEC1B89F, 0x3E896064,
        0xBF71015C, 0xBD6927DC, 0xBF0C9391, 0x3F7FC813, 0xBF1AAD60, 0xBE21507F, 0x3F54145C,
        0x3EE1F776, 0xBE158069, 0x3E07556A, 0xBE39060C, 0xBE9DE2BF, 0x3C8D99BC, 0x3F135E5B,
        0x3EB9ED71, 0x3F4FD229, 0xBF2F2F1D, 0x3E44581F, 0xBFA371C2, 0x3F4046CA, 0xBF77DC83,
        0xBE1A9BCF, 0x3EFFE27A, 0x3ED49FD3, 0xBD7E8874, 0x3E141714, 0xBE0C4DF5, 0xBE994BD9,
        0xBCC2B764, 0x3F122C03, 0x3D4FF760, 0x3F30995A, 0xBF3D531E, 0x3DBF93CA, 0xBF83E7B0,
        0x3F1B9A56, 0xBF667ABA, 0xBE09487D, 0x3F0CC14E, 0xBFC92305, 0xBF04B41A, 0xBEFFE650,
        0x3EB715B3, 0x3F3F8963, 0xBEC580FE, 0xBE46CC43, 0x3E86497B, 0xBF3A2B01, 0x3F6F2187,
        0x3FD9FE59, 0xBE01B21E, 0xBF79E0CA, 0xBD836894, 0xBF2B51EC, 0x3CA5B4A8, 0x3DDD1418,
        0x3EA13676, 0x3F437B2D, 0xBDD6C43E, 0xBF3767AF, 0xBF3D53FF, 0xBF18B4E0, 0x3F14CC97,
        0xBEED3D6F, 0xBFAA471C, 0xBF4B210A, 0x3EDF9FFD, 0x3F15F544, 0xBE986AE2, 0x3EF63949,
        0x3F0CC14E, 0xBFC92305, 0xBF04B41A, 0xBEFFE650, 0x3EB715B3, 0x3F3F8963, 0xBEC580FE,
        0xBE46CC44, 0x3E86497C, 0xBF3A2B01, 0x3F6F2188, 0x3FD9FE58, 0xBE01B21C, 0xBF79E0CA,
        0xBD836890, 0xBF2B51EC, 0x3FDF8E50, 0x3F5A7D0E, 0xBF069D5F, 0xBF018EBD, 0xBF8B8C40,
        0xBF2EC5E6, 0x3ED407E6, 0xBEC6C40A, 0xBEF26B7C, 0x3E21D456, 0x3B19E2C0, 0x3DD21C04,
        0xBF3393C7, 0x3E54B668, 0xBE91E744, 0xBE2A0043,
    ];
    const FG_CASE_MODE: [&str; 16] = [
        "fg", "fg", "fg", "fg", "fgw", "fgw", "fgw", "fgw", "fgw", "fgw", "fgw", "fgw", "chain",
        "chain", "chain", "chain",
    ];
    const FG_CASE_CURSOR: [u32; 16] = [
        0x00000014, 0x0000001C, 0x00000014, 0x0000001E, 0x00000014, 0x00000014, 0x0000001C,
        0x0000001C, 0x00000010, 0x00000010, 0x0000001E, 0x0000001E, 0x00000014, 0x0000001C,
        0x00000014, 0x0000001E,
    ];
    const FG_CASE_BUFFERED: [bool; 16] = [
        true, true, false, false, true, true, true, true, false, false, false, false, true, true,
        false, false,
    ];
    const FG_CASE_GAIN: [u32; 16] = [
        0x3F800000, 0x3F800000, 0x3F800000, 0x3F800000, 0x3F800000, 0x3F333333, 0x3F800000,
        0x3F333333, 0x3F800000, 0x3F333333, 0x3F800000, 0x3F333333, 0x3F800000, 0x3F333333,
        0x3F800000, 0x3F333333,
    ];
    const SEG_X: [u32; 12] = [
        0x40A00000, 0x3F800000, 0x40400000, 0xC0000000, 0xC0000000, 0x40E00000, 0x3F000000,
        0xC0400000, 0xC0400000, 0x40000000, 0x41100000, 0x40800000,
    ];
    const SEG_CASES: [(&[i64], &[i64]); 3] =
        [(&[0, 3, 7], &[3, 7, 12]), (&[1, 5], &[3, 10]), (&[2], &[9])];
    const SEG_MN: [u32; 6] = [
        0x3F800000, 0xC0000000, 0xC0400000, 0x3F800000, 0xC0400000, 0xC0400000,
    ];
    const SEG_MX: [u32; 6] = [
        0x40A00000, 0x40E00000, 0x41100000, 0x40400000, 0x40E00000, 0x40E00000,
    ];
    const SEG_FIRST_MIN: [i64; 6] = [1, 3, 7, 1, 7, 7];

    fn fb(v: &[u32]) -> Vec<f32> {
        v.iter().map(|b| f32::from_bits(*b)).collect()
    }

    /// Bit-exactness against the pyradius reference (golden vectors dumped by
    /// `.acscheck/dump_golden.py`; route B off, so the numpy reference paths).
    #[test]
    fn golden_acs_ops() {
        // time_to_iir_a
        let tau = fb(&T2A_TAU);
        let rate = fb(&T2A_RATE);
        let want = fb(&T2A_OUT);
        for i in 0..tau.len() {
            let got = time_to_iir_a(tau[i], rate[i]);
            assert_eq!(got.to_bits(), want[i].to_bits(), "time_to_iir_a[{i}]");
        }

        // cart_to_polar + threshold_lt_inplace
        let cart = fb(&CTP_CART);
        let (mag, ph) = cart_to_polar(&cart);
        assert_eq!(
            mag.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            fb(&CTP_MAG).iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            ph.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            fb(&CTP_PHASE)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );

        let mut tv = fb(&THR_IN);
        threshold_lt_inplace(&mut tv, f32::from_bits(0x2B8CBCCC));
        assert_eq!(
            tv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            fb(&THR_OUT).iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );

        // acs_spectrum1
        let r1 = acs_spectrum1(
            &fb(&ACS1_TIME),
            &fb(&ACS1_WIN),
            &fb(&ACS1_ENV),
            32,
            17,
            ACS1_IIR_A,
        );
        assert_eq!(
            r1.t1_io.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            fb(&ACS1_IO).iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            r1.t1_cart.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            fb(&ACS1_CART)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            r1.t1_env.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            fb(&ACS1_ENVOUT)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );

        // acs_spectrum2
        let r2 = acs_spectrum2(&fb(&ACS2_TIME), 32, 17, 20);
        assert_eq!(
            r2.t2_cart.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            fb(&ACS2_CART)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            r2.t2_mag.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            fb(&ACS2_MAG)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            r2.t2_phase.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            fb(&ACS2_PHASE)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }

    /// Bit-exactness of every `fill_granule` branch (fg / fgw / chain,
    /// buffered and direct, gain 1.0 and 0.7, non-wrapping and wrapping).
    #[test]
    fn golden_fill_granule() {
        let ring = fb(&FG_RING);
        let win = fb(&FG_WIN);
        let acc0 = fb(&FG_ACC);
        let want = fb(&FG_OUT);
        for (ci, mode) in FG_CASE_MODE.iter().enumerate() {
            let m = match *mode {
                "fg" => FillMode::Fg,
                "fgw" => FillMode::Fgw,
                _ => FillMode::Chain,
            };
            let mut acc = acc0.clone();
            let got = fill_granule(
                &ring,
                &win,
                &mut acc,
                m,
                8,
                16,
                2,
                16,
                FG_CASE_CURSOR[ci] as i64,
                FG_CASE_BUFFERED[ci],
                32,
                32,
                0,
                f32::from_bits(FG_CASE_GAIN[ci]),
            );
            assert_eq!(got.len(), 16);
            let w = &want[ci * 16..(ci + 1) * 16];
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                w.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "fill_granule case {ci} mode {mode}"
            );
        }
    }

    #[test]
    fn golden_segments() {
        let x = fb(&SEG_X);
        let mn = fb(&SEG_MN);
        let mx = fb(&SEG_MX);
        let mut o = 0;
        for (ci, (st, en)) in SEG_CASES.iter().enumerate() {
            let (a, b) = seg_minmax(&x, st, en);
            assert_eq!(
                a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                mn[o..o + a.len()]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "seg_minmax min {ci}"
            );
            assert_eq!(
                b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                mx[o..o + b.len()]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "seg_minmax max {ci}"
            );
            let fm = seg_first_min(&x, st, en);
            assert_eq!(
                fm,
                SEG_FIRST_MIN[o..o + fm.len()].to_vec(),
                "seg_first_min {ci}"
            );
            o += a.len();
        }
    }
}
