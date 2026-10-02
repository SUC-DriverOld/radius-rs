//! `overlap_add.c` and `crossover.c` — port of `pyradius/vocoder_ops.py`:
//!
//! * [`overlap_add_channel`] (`vocoder_ops.py` 2992-3089, `rx_overlap_add_channel`)
//! * [`Crossover`] (`vocoder_ops.py` 3255-3329, `rx_crossover_process1`)
//! * [`crossover_process1`] / [`crossover_process`] (3332-3339)
//!
//! Float discipline: `f32` throughout with the C op order (`-ffp-contract=off`,
//! so `a*b + c` stays two operations) except where the reference deliberately
//! goes through `f64`. The `fma` sites are the reference's `_fma`/`_fma_arr`
//! ([`super::fma`] = `f32::mul_add`, one correctly-rounded rounding) and the
//! f64-then-one-cast chains in the window fade and the ring accumulation are
//! kept verbatim: `np.float32(np.float64(a) * np.float64(b))` is *not* a
//! "redundant" cast, it is the reference's rounding point.
//!
//! The C tables these need are embedded by
//! [`crate::vocoder::crossover_tables`].

use super::fma;

// ===========================================================================
// overlap_add.c — rx_overlap_add_channel
// ===========================================================================

/// `rx_overlap_add_channel @0x16EE44` — window fade + ring write.
///
/// Argument order follows the Python (`win1, win2, synth_a, synth_b, out_ring,
/// edge_gain, ch, a3, a4, a5, frame_n, p, A, v8, v9, f1412, cursor, hop,
/// ring_len, f1224, n_write`); `A` is the per-rate ring constant (3300 at
/// 44.1 kHz, 3592 at 48 kHz).
///
/// `win1`/`win2` are modified in place (the Python returns fresh copies of the
/// buffers it was handed, which the caller assigns back). The returned vector is
/// the updated `out_ring` copy; the Python also returns the two gain weights,
/// which the engine call site drops, so they are not returned here (see
/// [`OacGains`] and [`overlap_add_channel_gains`] if they are ever needed).
///
/// `ch` is unused by the operator body — the Python indexes `out_ring` per
/// channel at the call site, exactly as the Rust caller does.
///
/// # Panics
/// If the slices are too short for the requested write (`X + pos0 > ring_len`
/// with `X > ring_len`), matching the Python's `IndexError`.
#[allow(clippy::too_many_arguments)]
pub fn overlap_add_channel(
    win1: &mut [f32],
    win2: &mut [f32],
    synth_a: &[f32],
    synth_b: &[f32],
    out_ring: &[f32],
    edge_gain: &[f32],
    _ch: usize,
    a3: i32,
    a4: i32,
    a5: i32,
    frame_n: i64,
    p: i32,
    a: i64,
    v8: i64,
    v9: i64,
    f1412: i64,
    cursor: i64,
    hop: i64,
    ring_len: i64,
    f1224: i64,
    n_write: i64,
) -> Vec<f32> {
    let mut out = out_ring.to_vec();
    overlap_add_channel_in_place(
        win1, win2, synth_a, synth_b, &mut out, edge_gain, _ch, a3, a4, a5,
        frame_n, p, a, v8, v9, f1412, cursor, hop, ring_len, f1224, n_write,
    );
    out
}

/// In-place variant used by the streaming vocoder.  Keeping the output ring
/// in its existing allocation avoids cloning roughly 120k samples for every
/// channel of every granule.
#[allow(clippy::too_many_arguments)]
pub fn overlap_add_channel_in_place(
    win1: &mut [f32],
    win2: &mut [f32],
    synth_a: &[f32],
    synth_b: &[f32],
    out: &mut [f32],
    edge_gain: &[f32],
    _ch: usize,
    a3: i32,
    a4: i32,
    a5: i32,
    frame_n: i64,
    p: i32,
    a: i64,
    v8: i64,
    v9: i64,
    f1412: i64,
    cursor: i64,
    hop: i64,
    ring_len: i64,
    f1224: i64,
    n_write: i64,
) {
    let half_n = frame_n >> 1;

    // `den_i = int(A) * int(v8) // (int(A) + int(v8))`
    let den_i = a * v8 / (a + v8);
    // `gidx = int(p) if int(p) < 3 else 3`, then `gidx = 0` when `p >= 10`.
    // The Python would index `edge_gain` from the end for a negative `p`; the
    // engine never does that, so clamp instead of relying on wraparound.
    let mut gidx = if p < 3 { p.max(0) } else { 3 };
    if p >= 10 {
        gidx = 0;
    }
    let mut v197 = edge_gain[gidx as usize] * 0.4f32;
    v197 *= (f1412 + a3 as i64) as f32;
    v197 /= den_i as f32;
    let mut v198 = 1.0f32;
    let g09 = v197 * 0.9f32;
    if a4 > 0 {
        v198 = 0.5f32;
        v197 = g09;
    }

    // ---- window fading (in place on win1 / win2) ----
    let half1 = v8 / 2;
    let half2 = v9 / 2;
    let base = half_n - hop;
    if cursor < hop + half1 && hop >= 1 {
        let pos = cursor - hop;
        let n = (2 * hop).min(half1 - pos);
        if n > 0 {
            let n = n as usize;
            let base = base as usize;
            for i in 0..n {
                let pp = (pos + i as i64) as f32;
                let t = pp * super::PI_F;
                let t = t * 0.5f32;
                let t = t / half1 as f32;
                let t = 2.0f32 - (((t as f64).sin() as f32).sqrt());
                win1[base + i] *= t;
            }
        }
    }
    if cursor < hop + half2 && hop > 0 {
        let pos = cursor - hop;
        let n = (2 * hop).min(half2 - pos);
        if n > 0 {
            let n = n as usize;
            let base = base as usize;
            for i in 0..n {
                let pp = (pos + i as i64) as f32;
                let t = pp * super::PI_F;
                let t = t * 0.5f32;
                let t = t / half2 as f32;
                let t = 2.0f32 - (((t as f64).sin() as f32).sqrt());
                win2[base + i] *= t;
            }
        }
    }

    // `gw = [v197, v198]`; the Python returns it with the buffers, but the
    // engine call site drops it (see `overlap_add_channel_gains`), and the
    // `a4 != a5` gate returns before the main write.
    if a4 != a5 {
        return;
    }

    // ---- main write (`_oac_nb`, vocoder_ops.py 1238-1362) ----
    let r = ring_len;
    let big_n = n_write;
    debug_assert_eq!(out.len() as i64, r);
    // C truncating remainder (`math.fmod`), not a floored one.
    let pos0 = (cursor - hop) % r;
    let v35 = (f1224 + hop - cursor).min(big_n);
    let v36 = v35.max(0);
    let wh = base;
    let wrap = cursor < hop || big_n + pos0 > r;
    let pcur = if pos0 < 0 { 0 } else { pos0 };
    let fvi = if cursor - hop < 0 { hop - cursor } else { 0 };

    for i in 0..big_n {
        let w = (wh + i) as usize;
        // `t2 = f32(f64(v198) * f64(f32(f64(win2) * f64(synth_b))))`
        let inner = (win2[w] as f64 * synth_b[i as usize] as f64) as f32;
        let t2 = (v198 as f64 * inner as f64) as f32;
        let t = fma(win1[w], synth_a[i as usize], t2);
        if wrap {
            if i < fvi {
                continue;
            }
            let pidx = (pcur + (i - fvi)) % r;
            let pidx = pidx as usize;
            if i < v35 {
                out[pidx] = fma(t, v197, out[pidx]);
            } else if i >= v36 && i >= v35 {
                // `f32(f64(t) * f64(v197))` — one f32 multiply.
                out[pidx] = t * v197;
            }
        } else {
            let pidx = pos0 + i;
            if pidx >= r {
                continue;
            }
            let pidx = pidx as usize;
            if i < v36 {
                out[pidx] = fma(t, v197, out[pidx]);
            } else {
                out[pidx] = t * v197;
            }
        }
    }
}

/// The two gain weights `rx_overlap_add_channel` computes
/// (`{"gain_weight": [v197, v198]}` in the Python).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OacGains {
    /// `v197` — the write gain (`0.9 * edge_gain * (f1412 + a3) / den_i` when
    /// `a4 > 0`).
    pub v197: f32,
    /// `v198` — `0.5` when `a4 > 0`, else `1.0`.
    pub v198: f32,
}

/// [`overlap_add_channel`] plus the gain weights the Python returns alongside
/// the buffers. Identical work; the plain entry point exists because that is
/// what the engine call site uses.
#[allow(clippy::too_many_arguments)]
pub fn overlap_add_channel_gains(
    win1: &mut [f32],
    win2: &mut [f32],
    synth_a: &[f32],
    synth_b: &[f32],
    out_ring: &[f32],
    edge_gain: &[f32],
    ch: usize,
    a3: i32,
    a4: i32,
    a5: i32,
    frame_n: i64,
    p: i32,
    a: i64,
    v8: i64,
    v9: i64,
    f1412: i64,
    cursor: i64,
    hop: i64,
    ring_len: i64,
    f1224: i64,
    n_write: i64,
) -> (Vec<f32>, OacGains) {
    // The gains only depend on the scalars, so recompute them the same way and
    // let the shared body do the work.
    let den_i = a * v8 / (a + v8);
    let mut gidx = if p < 3 { p.max(0) } else { 3 };
    if p >= 10 {
        gidx = 0;
    }
    let mut v197 = edge_gain[gidx as usize] * 0.4f32;
    v197 *= (f1412 + a3 as i64) as f32;
    v197 /= den_i as f32;
    let mut v198 = 1.0f32;
    if a4 > 0 {
        v198 = 0.5f32;
        v197 *= 0.9f32;
    }
    let out = overlap_add_channel(
        win1, win2, synth_a, synth_b, out_ring, edge_gain, ch, a3, a4, a5, frame_n, p, a, v8, v9,
        f1412, cursor, hop, ring_len, f1224, n_write,
    );
    (out, OacGains { v197, v198 })
}

// ===========================================================================
// crossover.c — rx_crossover_process1
// ===========================================================================

/// `_XOVER_TAPS` — the reference's per-rate cache of the reversed tap bank.
static XOVER_TAPS_48K: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
/// `_XOVER_TAPS[44100]`.
static XOVER_TAPS_44K: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();

/// `_xover_taps_rev(sr)` — `taps_rev[b][k] = taps[b][N-1-k]` as `f32`
/// (`rx_crossover_create` pre-reverses the coefficient table).
///
/// The result is cached per sample rate (`_XOVER_TAPS`), so every
/// [`Crossover`] shares one immutable bank instead of reversing its own copy.
///
/// # Panics
/// On a sample rate the engine does not support.
pub fn xover_taps_rev(sr: u32) -> &'static [f32] {
    let cell = match sr {
        48_000 => &XOVER_TAPS_48K,
        44_100 => &XOVER_TAPS_44K,
        other => panic!("crossover tables: unsupported sample rate {other}"),
    };
    cell.get_or_init(|| {
        let fir = super::crossover_tables::fir(sr);
        let n = fir.len() / XOVER_BANDS;
        let mut rev = vec![0.0f32; fir.len()];
        for b in 0..XOVER_BANDS {
            for k in 0..n {
                rev[b * n + k] = fir[b * n + (n - 1 - k)];
            }
        }
        rev
    })
}

/// Bands in the C table (`[4][2048]`).
const XOVER_BANDS: usize = 4;

/// 4-band FIR bank — port of `rx_crossover_process1`
/// (`pyradius.vocoder_ops.Crossover`).
///
/// Geometry (adjudicated against the C, not inferred from its comments): the
/// engine keeps a double-mirrored history buffer, writes `buf[h] = buf[h+N] =
/// x[t]` and reads `hp = buf + h + 1`, so `hp[k] = x[t-N+1+k]`, while the
/// coefficient table is pre-reversed via `taps_rev[b][k] = taps[b][N-1-k]`.
/// Combining the two gives the ordinary causal convolution
///
/// ```text
///     y[t] = sum_i taps[b][i] * x[t-i]
/// ```
///
/// which is what `.tmp/xo_which.c` verified bit-exact against the parity corpus
/// (`max|d| = 0`, `exact_frac = 1.0`).
///
/// Accumulation follows the engine's NEON tree verbatim: sixteen lane chains
/// `s[j] = vfmaq(s[j], hp[k+j], taps_rev[k+j])` for `k` stepping by 16, then
/// `vaddvq(vaddq(vaddq(s0,s1), vaddq(s2,s3)))` — i.e. lane `l` of the result is
/// `((s[l] + s[4+l]) + (s[8+l] + s[12+l]))`, reduced as `(v0+v1) + (v2+v3)`.
/// This matters far more than its ~1e-7 size suggests: the vocoder's peak
/// detector is a discrete decision, and a 1e-7 crossover difference flipped the
/// measured peak count (236 vs 290) and decorrelated a whole render.
///
/// The reference's *route-B* numba fast path (`_xover_nb`, an `f64` multiply-add
/// chain plus the separate `_xover_fix_nb` repair kernel that
/// `Crossover.process` never calls) is deliberately **not** reproduced: the
/// engine's `vfmaq` chain is the correctly-rounded `fma`, which is also exactly
/// what the Python's non-numba fallback computes through `_fma_arr` (and the two
/// agree bit-for-bit on the golden vectors this module tests).
pub struct Crossover {
    /// `taps_rev[b * n + k] = taps[b][n - 1 - k]`, shared through
    /// [`xover_taps_rev`]'s cache.
    pub taps_rev: &'static [f32],
    /// `N` — taps per band (2048; `RX_XOVER_FIR_TAPS`).
    pub n: usize,
    /// Number of bands used (`rx_vc_state.nbands`, always 4).
    pub n_bands: usize,
    /// `N - 1` previous input samples: the C's mirrored buffer retains them
    /// across calls, so the filter is continuous over feed blocks (not
    /// block-local).
    pub hist: Vec<f32>,
    /// Scratch `[hist, x]` view; the Python rebuilds
    /// `np.concatenate([self.hist, x])` on every call.
    z: Vec<f32>,
}

impl Crossover {
    /// `rx_crossover_create` — build the reversed tap bank for `sr`
    /// (`48000` / `44100`) and zero the history.
    ///
    /// # Panics
    /// On an unsupported sample rate, or `n_bands > 4` (the table has four).
    pub fn new(sr: u32, n_bands: usize) -> Self {
        // `self.N = fir.shape[1]` with a `[4][2048]` table.
        let taps_rev = xover_taps_rev(sr);
        assert_eq!(
            taps_rev.len() % XOVER_BANDS,
            0,
            "crossover FIR is not band-major"
        );
        assert!(
            n_bands <= XOVER_BANDS,
            "crossover: at most {XOVER_BANDS} bands"
        );
        let n = taps_rev.len() / XOVER_BANDS;
        Self {
            taps_rev,
            n,
            n_bands,
            hist: vec![0.0f32; n - 1],
            z: Vec::with_capacity(n - 1),
        }
    }

    /// One channel: returns the band signals, `n_bands` vectors of `seq.len()`
    /// `f32` (`pyradius.vocoder_ops.Crossover.process` returns the same numbers
    /// widened to `f64`, "the C yields band values as double").
    pub fn process(&mut self, seq: &[f32]) -> Vec<Vec<f32>> {
        let n = seq.len();
        let big_n = self.n;
        let mut out = vec![vec![0.0f32; n]; self.n_bands];
        if n == 0 {
            return out;
        }

        // `z = concat(hist, x)`; `hp[t, k] = z[t + k]`.
        self.z.clear();
        self.z.extend_from_slice(&self.hist);
        self.z.extend_from_slice(seq);
        let keep = big_n - 1;
        if self.z.len() >= keep {
            let at = self.z.len() - keep;
            self.hist.copy_from_slice(&self.z[at..]);
        } else {
            // Fewer than `N - 1` samples in total: slide what we have and
            // zero-pad the front. (The Python's `z[-(N-1):]` would *shrink*
            // `hist` here and then break its sliding-window view; the engine
            // always feeds at least `N - 1` samples per block.)
            let m = self.z.len();
            self.hist.copy_within(m.., 0);
            self.hist[keep - m..].copy_from_slice(&self.z);
        }

        let z = &self.z;
        for b in 0..self.n_bands {
            let tr = &self.taps_rev[b * big_n..(b + 1) * big_n];
            for t in 0..n {
                let mut s = [0.0f32; 16];
                let mut k = 0usize;
                while k < big_n {
                    // `s[j] = fma(hp[t, k+j], taps_rev[k+j], s[j])`
                    for j in 0..16 {
                        s[j] = fma(z[t + k + j], tr[k + j], s[j]);
                    }
                    k += 16;
                }
                // NEON tree `vaddvq(vaddq(vaddq(s0,s1), vaddq(s2,s3)))`:
                // lane `l` is `((s[l] + s[4+l]) + (s[8+l] + s[12+l]))` — the
                // Python's `(sj[0] + sj[1]) + (sj[2] + sj[3])` with
                // `sj[i][l] = s[4i + l]` — and the horizontal add reduces the
                // four lanes as `(v0 + v1) + (v2 + v3)`.
                let mut v = [0.0f32; 4];
                for l in 0..4 {
                    v[l] = (s[l] + s[4 + l]) + (s[8 + l] + s[12 + l]);
                }
                out[b][t] = (v[0] + v[1]) + (v[2] + v[3]);
            }
        }
        out
    }
}

/// One-channel fresh-instance run (as in the C op harness).
pub fn crossover_process1(sr: u32, seq: &[f32]) -> Vec<Vec<f32>> {
    Crossover::new(sr, 4).process(seq)
}

/// Whole-sequence convenience form (per-channel instance, 1 channel).
pub fn crossover_process(sr: u32, seq: &[f32]) -> Vec<Vec<f32>> {
    Crossover::new(sr, 4).process(seq)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- golden data (minted from `pyradius.vocoder_ops` by .tmp tooling) ----
    const XO_GOLD_B0: [u32; 8] = [
        0x3166C3DC, 0xB15D997B, 0x31E6D503, 0xB1DD8960, 0xB1ACE09E, 0xAF55F900, 0xB2139F8D,
        0x314214C8,
    ];
    const XO_GOLD_B1: [u32; 8] = [
        0xB168C13F, 0x315F87FD, 0xB1E8CC86, 0x31DF7CE4, 0x31AE6C64, 0x2F58CF20, 0x3214E2B4,
        0xB143E546,
    ];
    const XO_GOLD_B2: [u32; 8] = [
        0x2DE0BA63, 0xADD5CCB5, 0x2E602148, 0xAE5A7CEF, 0xAE2B161A, 0xAC825388, 0xAE91EED0,
        0x2DD2830A,
    ];
    const XO_GOLD_B3: [u32; 8] = [
        0x2C7EB11B, 0xACB0C9F2, 0x2D21B00C, 0xAD50B652, 0xAB236470, 0xAC15A924, 0xAD101A15,
        0x2D11986E,
    ];

    const OAC_NOWRAP_WIN1: [u32; 32] = [
        0x3F800000, 0x3F8147AE, 0x3F828F5C, 0x3F83D70A, 0x3F851EB8, 0x3F866666, 0x3F87AE14,
        0x3F88F5C3, 0x400A3D71, 0x3FC0BB20, 0x3FA333A9, 0x3F939839, 0x3F8F5C29, 0x3F90A3D7,
        0x3F91EB85, 0x3F933333, 0x3F947AE1, 0x3F95C28F, 0x3F970A3D, 0x3F9851EC, 0x3F99999A,
        0x3F9AE148, 0x3F9C28F6, 0x3F9D70A4, 0x3F9EB852, 0x3FA00000, 0x3FA147AE, 0x3FA28F5C,
        0x3FA3D70A, 0x3FA51EB8, 0x3FA66666, 0x3FA7AE14,
    ];

    const OAC_NOWRAP_WIN2: [u32; 32] = [
        0x3F000000, 0x3F051EB8, 0x3F0A3D71, 0x3F0F5C29, 0x3F147AE1, 0x3F19999A, 0x3F1EB852,
        0x3F23D70A, 0x3FA8F5C3, 0x3F7078BE, 0x3F4FB61C, 0x3F3F795F, 0x3F3D70A4, 0x3F428F5C,
        0x3F47AE14, 0x3F4CCCCD, 0x3F51EB85, 0x3F570A3D, 0x3F5C28F6, 0x3F6147AE, 0x3F666666,
        0x3F6B851F, 0x3F70A3D7, 0x3F75C28F, 0x3F7AE148, 0x3F800000, 0x3F828F5C, 0x3F851EB8,
        0x3F87AE14, 0x3F8A3D71, 0x3F8CCCCD, 0x3F8F5C29,
    ];

    const OAC_NOWRAP_OUT: [u32; 32] = [
        0x42278327, 0x41EE02D0, 0x41CD56DD, 0x41BD1869, 0x41BAD80A, 0x41BF91FA, 0x41C44102,
        0x41C8E521, 0x41CD7E56, 0x41D20CA1, 0x41D69004, 0x41DB087C, 0x41DF760A, 0x41E3D8B2,
        0x41E8306E, 0x41EC7D40, 0x3F4CCCCD, 0x3F59999A, 0x3F666666, 0x3F733333, 0x3F800000,
        0x3F866666, 0x3F8CCCCD, 0x3F933333, 0x3F99999A, 0x3FA00000, 0x3FA66666, 0x3FACCCCD,
        0x3FB33333, 0x3FB9999A, 0x3FC00000, 0x3FC66666,
    ];

    const OAC_NOWRAP_GAIN: [u32; 2] = [0x41D51EB8, 0x3F800000];

    const OAC_WRAP_WIN1: [u32; 32] = [
        0x3F800000, 0x3F8147AE, 0x3F828F5C, 0x3F83D70A, 0x3F851EB8, 0x3F866666, 0x3F87AE14,
        0x3F88F5C3, 0x400A3D71, 0x3FC0BB20, 0x3FA333A9, 0x3F939839, 0x3F8F5C29, 0x3F90A3D7,
        0x3F91EB85, 0x3F933333, 0x3F947AE1, 0x3F95C28F, 0x3F970A3D, 0x3F9851EC, 0x3F99999A,
        0x3F9AE148, 0x3F9C28F6, 0x3F9D70A4, 0x3F9EB852, 0x3FA00000, 0x3FA147AE, 0x3FA28F5C,
        0x3FA3D70A, 0x3FA51EB8, 0x3FA66666, 0x3FA7AE14,
    ];

    const OAC_WRAP_WIN2: [u32; 32] = [
        0x3F000000, 0x3F051EB8, 0x3F0A3D71, 0x3F0F5C29, 0x3F147AE1, 0x3F19999A, 0x3F1EB852,
        0x3F23D70A, 0x3FA8F5C3, 0x3F7078BE, 0x3F4FB61C, 0x3F3F795F, 0x3F3D70A4, 0x3F428F5C,
        0x3F47AE14, 0x3F4CCCCD, 0x3F51EB85, 0x3F570A3D, 0x3F5C28F6, 0x3F6147AE, 0x3F666666,
        0x3F6B851F, 0x3F70A3D7, 0x3F75C28F, 0x3F7AE148, 0x3F800000, 0x3F828F5C, 0x3F851EB8,
        0x3F87AE14, 0x3F8A3D71, 0x3F8CCCCD, 0x3F8F5C29,
    ];

    const OAC_WRAP_OUT: [u32; 12] = [
        0x41DAA93D, 0x41DEA57F, 0x41E296D4, 0x41E67D40, 0x41BAD80A, 0x41BF91FA, 0x41C44102,
        0x41C8E521, 0x41CA4B23, 0x41CE7307, 0x41D29004, 0x41D6A216,
    ];

    const OAC_WRAP_GAIN: [u32; 2] = [0x41D51EB8, 0x3F800000];

    /// `crossover_process1(48000, X8)` must reproduce the reference bit for bit
    /// (the Python's default `_xover_nb` path; see the note on the struct about
    /// the `fma` tree — the two agree on these inputs).
    #[test]
    fn crossover_matches_reference_bits() {
        const X8: [f32; 8] = [0.25, -0.5, 0.75, -1.0, 0.125, 0.375, -0.625, 0.875];
        let bands = crossover_process1(48_000, &X8);
        for (b, want) in [XO_GOLD_B0, XO_GOLD_B1, XO_GOLD_B2, XO_GOLD_B3]
            .iter()
            .enumerate()
        {
            let got: Vec<u32> = bands[b].iter().map(|v| v.to_bits()).collect();
            assert_eq!(got, want.to_vec(), "band {b}");
        }
    }

    /// The reference's input buffers for the two `overlap_add_channel` cases
    /// (built in `f64` and rounded once, exactly like the NumPy literals in the
    /// minting script).
    fn oac_reference_case(
        ring_len: usize,
        cursor: i64,
        f1224: i64,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, OacGains) {
        let (frame_n, hop, n_write) = (32i64, 8i64, 16i64);
        let mut win1: Vec<f32> = (0..32).map(|i| (1.0f64 + 0.01 * i as f64) as f32).collect();
        let mut win2: Vec<f32> = (0..32).map(|i| (0.5f64 + 0.02 * i as f64) as f32).collect();
        let synth_a: Vec<f32> = (0..16).map(|i| (0.3f64 + 0.01 * i as f64) as f32).collect();
        let synth_b: Vec<f32> = (0..16).map(|i| (0.7f64 - 0.01 * i as f64) as f32).collect();
        let out_ring: Vec<f32> = (0..ring_len).map(|i| (0.05 * i as f64) as f32).collect();
        let edge_gain = [1.15f32, 1.30, 1.05, 1.10];
        let (out, gains) = overlap_add_channel_gains(
            &mut win1,
            &mut win2,
            &synth_a,
            &synth_b,
            &out_ring,
            &edge_gain,
            0,
            222,
            0,
            0,
            frame_n,
            2,
            3592,
            8,
            8,
            222,
            cursor,
            hop,
            ring_len as i64,
            f1224,
            n_write,
        );
        (win1, win2, out, gains)
    }

    /// Straight-line (non-wrapping) ring write, bit-exact against the reference.
    #[test]
    fn overlap_add_matches_reference_bits_nowrap() {
        let (win1, win2, out, gains) = oac_reference_case(32, 8, 16);
        assert_eq!(bits(&win1), OAC_NOWRAP_WIN1.to_vec(), "win1");
        assert_eq!(bits(&win2), OAC_NOWRAP_WIN2.to_vec(), "win2");
        assert_eq!(bits(&out), OAC_NOWRAP_OUT.to_vec(), "out_ring");
        assert_eq!(gains.v197.to_bits(), OAC_NOWRAP_GAIN[0]);
        assert_eq!(gains.v198.to_bits(), OAC_NOWRAP_GAIN[1]);
    }

    /// The wrap branch (`N + pos0 > ring_len`), where the ring index is folded
    /// modulo `ring_len`, bit-exact against the reference.
    #[test]
    fn overlap_add_matches_reference_bits_wrap() {
        let (win1, win2, out, gains) = oac_reference_case(12, 8, 8);
        assert_eq!(bits(&win1), OAC_WRAP_WIN1.to_vec(), "win1");
        assert_eq!(bits(&win2), OAC_WRAP_WIN2.to_vec(), "win2");
        assert_eq!(bits(&out), OAC_WRAP_OUT.to_vec(), "out_ring");
        assert_eq!(gains.v197.to_bits(), OAC_WRAP_GAIN[0]);
        assert_eq!(gains.v198.to_bits(), OAC_WRAP_GAIN[1]);
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// Compile-time contract with `crate::vocoder::VocoderState`: the argument
    /// types the orchestration layer passes (see `ev_oac`/`feed` in
    /// `src/vocoder/vocoder.rs`) must match this signature.
    #[test]
    fn orchestrator_call_shape_compiles() {
        let mut win1 = vec![0.0f32; 16384];
        let mut win2 = vec![0.0f32; 16384];
        let synth_a = vec![0.0f32; 7184];
        let synth_b = vec![0.0f32; 7184];
        let out_slice = vec![0.0f32; 113536];
        let edge_gain = [1.15f32, 1.30, 1.05, 1.10];
        let (ch, a3, p) = (0usize, 0i32, 2i32);
        let (a_const, f1412, pos_1384) = (3592i64, 0i64, 0i64);
        let (hop, ring_len, f1224, n_write) = (3592i64, 113536i64, 0i64, 7184i64);
        let out: Vec<f32> = overlap_add_channel(
            &mut win1, &mut win2, &synth_a, &synth_b, &out_slice, &edge_gain, ch, a3, 0, 0,
            16384i64, p, a_const, 2930i64, 1256i64, f1412, pos_1384, hop, ring_len, f1224, n_write,
        );
        assert_eq!(out.len(), ring_len as usize);

        let sr: u32 = 48_000;
        let nbands: usize = 4;
        let mut xo = Crossover::new(sr, nbands);
        let mono: Vec<f32> = vec![0.0; 64];
        let bands: Vec<Vec<f32>> = xo.process(&mono);
        assert_eq!(bands.len(), 4);
        assert_eq!(bands[0].len(), 64);
    }

    /// Deterministic white noise — the same integer LCG the Python probe uses,
    /// so the values are bit-identical on both sides:
    /// `f32(f32((s >> 8) & 0xffffff) / 2^23 - 1)`, all exact in `f32`.
    fn noise(n: usize, mut s: u32) -> Vec<f32> {
        let mut x = Vec::with_capacity(n);
        for _ in 0..n {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let v = ((s >> 8) & 0x00FF_FFFF) as f32 / 8_388_608.0 - 1.0;
            x.push(v);
        }
        x
    }

    /// `RX_XOVER_FIR_DELAY` — the linear-phase bank delays by `N/2 - 1`.
    const DELAY: usize = 1023;

    /// The four bands sum back to the (delayed) input. Measured against the
    /// Python reference: `max|d| = 5.8e-07` over 3073 samples of this noise, so
    /// the assertion uses 1e-5 — 17x margin, still ~4 decimal digits tighter
    /// than "the bands are unrelated".
    #[test]
    fn crossover_bands_sum_to_delayed_input() {
        let x = noise(4096, 0x1234_5678);
        let bands = crossover_process(48_000, &x);
        assert_eq!(bands.len(), 4);
        assert!(bands.iter().all(|b| b.len() == x.len()));
        let mut worst = 0.0f32;
        for t in DELAY..x.len() {
            let sum: f32 = bands.iter().map(|b| b[t]).sum();
            worst = worst.max((sum - x[t - DELAY]).abs());
        }
        assert!(worst < 1e-5, "band reconstruction error {worst}");
        // 44.1 kHz uses a different FIR bank but the same geometry.
        let bands44 = crossover_process(44_100, &x);
        let mut worst44 = 0.0f32;
        for t in DELAY..x.len() {
            let sum: f32 = bands44.iter().map(|b| b[t]).sum();
            worst44 = worst44.max((sum - x[t - DELAY]).abs());
        }
        assert!(worst44 < 1e-5, "44.1k band reconstruction error {worst44}");
    }

    /// The history must survive across calls: splitting the input in two must
    /// be bit-identical to filtering it in one go (verified against the Python:
    /// `max|d| == 0`).
    #[test]
    fn crossover_history_is_continuous() {
        let x = noise(2048, 0x0BAD_F00D);
        let whole = crossover_process(48_000, &x);
        let mut c = Crossover::new(48_000, 4);
        let mut split: Vec<Vec<f32>> = vec![Vec::new(); 4];
        let mut off = 0;
        for chunk in [&x[..1000], &x[1000..1500], &x[1500..]] {
            let part = c.process(chunk);
            for b in 0..4 {
                split[b].extend_from_slice(&part[b]);
            }
            off += chunk.len();
        }
        assert_eq!(off, x.len());
        for b in 0..4 {
            assert_eq!(split[b], whole[b], "band {b} differs across call split");
        }
    }

    /// A block shorter than `N - 1` must not corrupt the filter state.
    #[test]
    fn crossover_short_block_keeps_state() {
        let x = noise(4096, 7);
        let whole = crossover_process(48_000, &x);
        let mut c = Crossover::new(48_000, 4);
        let first = c.process(&x[..5]); // 5 < N-1
        assert!(first.iter().all(|b| b.iter().all(|v| v.is_finite())));
        let rest = c.process(&x[5..]);
        for b in 0..4 {
            assert_eq!(rest[b].as_slice(), &whole[b][5..], "band {b}");
        }
    }

    #[test]
    fn crossover_empty_input() {
        let bands = crossover_process1(48_000, &[]);
        assert_eq!(bands.len(), 4);
        assert!(bands.iter().all(|b| b.is_empty()));
    }

    /// `_xover_taps_rev` is cached per rate (`_XOVER_TAPS`) and reverses each
    /// band: `rev[b * n + k] = fir[b * n + n - 1 - k]`.
    #[test]
    fn xover_taps_rev_is_cached_and_reversed() {
        let a = xover_taps_rev(48_000);
        let b = xover_taps_rev(48_000);
        assert!(
            std::ptr::eq(a, b),
            "the tap bank must be cached, not rebuilt"
        );
        let fir = crate::vocoder::crossover_tables::fir_48k();
        assert_eq!(a.len(), fir.len());
        for band in 0..4 {
            for k in 0..2048 {
                assert_eq!(a[band * 2048 + k], fir[band * 2048 + 2047 - k]);
            }
        }
        // both rates are distinct banks
        assert!(!std::ptr::eq(
            xover_taps_rev(48_000),
            xover_taps_rev(44_100)
        ));
    }

    /// The window fade writes `2 - sqrt(sin(t/2 * pi / half))` into the tail of
    /// the windows; with `cursor == hop` the fade is a no-op ramp (`t = 0`).
    #[test]
    fn overlap_add_window_fade_and_write() {
        let frame_n = 64i64;
        let hop = 16i64;
        let ring_len = 128i64;
        let n_write = 32i64;
        let mut win1 = vec![1.0f32; frame_n as usize];
        let mut win2 = vec![1.0f32; frame_n as usize];
        let synth_a = vec![0.5f32; 32];
        let synth_b = vec![0.25f32; 32];
        let out_ring = vec![0.0f32; ring_len as usize];
        let edge_gain = [1.15f32, 1.30, 1.05, 1.10];
        let out = overlap_add_channel(
            &mut win1, &mut win2, &synth_a, &synth_b, &out_ring, &edge_gain, 0, 222, 0, 0, frame_n,
            2, 3592, 8, 8, 222, hop, hop, ring_len, hop, n_write,
        );
        assert_eq!(out.len(), ring_len as usize);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(out.iter().any(|v| *v != 0.0), "nothing was written");
        // the fade only touches win1/win2 while cursor < hop + half, so at
        // cursor == hop the first `2*hop` samples of the fade span are scaled
        for (i, w) in win1.iter().enumerate().take(16) {
            if i < 16 {
                assert!(*w <= 1.0, "fade must not amplify ({i}: {w})");
            }
        }
        // a4 != a5 is the engine's "no write" gate
        let early = overlap_add_channel(
            &mut vec![1.0f32; frame_n as usize],
            &mut vec![1.0f32; frame_n as usize],
            &synth_a,
            &synth_b,
            &out_ring,
            &edge_gain,
            0,
            222,
            1,
            0,
            frame_n,
            2,
            3592,
            8,
            8,
            222,
            hop,
            hop,
            ring_len,
            hop,
            n_write,
        );
        assert!(early.iter().all(|v| *v == 0.0), "a4 != a5 must not write");
    }

    /// The gain weights (`v197`, `v198`) follow the `a4` gate: `v198 = 0.5` and
    /// `v197 *= 0.9` once `a4 > 0`.
    #[test]
    fn overlap_add_gain_gate() {
        let (mut w1, mut w2) = (vec![1.0f32; 64], vec![1.0f32; 64]);
        let (sa, sb) = (vec![1.0f32; 32], vec![1.0f32; 32]);
        let ring = vec![0.0f32; 128];
        let eg = [1.0f32, 1.0, 1.0, 1.0];
        let (_, g0) = overlap_add_channel_gains(
            &mut w1, &mut w2, &sa, &sb, &ring, &eg, 0, 10, 0, 0, 64, 0, 100, 8, 8, 10, 16, 16, 128,
            16, 32,
        );
        assert_eq!(g0.v198, 1.0);
        let (_, g1) = overlap_add_channel_gains(
            &mut w1, &mut w2, &sa, &sb, &ring, &eg, 0, 10, 1, 1, 64, 0, 100, 8, 8, 10, 16, 16, 128,
            16, 32,
        );
        assert_eq!(g1.v198, 0.5);
        assert!((g1.v197 - g0.v197 * 0.9).abs() < 1e-7);
        // den_i = 100 * 8 / 108 = 7 (integer division), so
        // v197 = 1.0 * 0.4 * (f1412 + a3 = 20) / 7
        assert!((g0.v197 - (0.4f32 * 20.0 / 7.0)).abs() < 1e-7);
    }
}
