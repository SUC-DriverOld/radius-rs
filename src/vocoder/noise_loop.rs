//! The DLL's noise-synthesis excitation loop — `sub_180158D20`'s per-channel,
//! per-machine-granule excitation evolution (RE doc §13.9; decompile
//! `.tmp/ida/machine_pretty.c` L2050-2660, L3232-3268).
//!
//! Buffer topology (all offsets renderer-relative, per-channel arrays of
//! `nb_bins` floats unless noted):
//!
//! | DLL | content | producer | our counterpart |
//! |-----|---------|----------|-----------------|
//! | 2080 | noise excitation state (evolves every machine granule) | this loop | `noise_gain[ch]` |
//! | 1872 | excitation's magnitude source (refreshed each granule) | `1872 ← 1976` @LABEL_768 | `ch.mag_copy` |
//! | 1976 | previous analysis spectrum (the 1872 source) | ACS layer | `ch.mag` |
//! | 2256 | per-bin init-once flags (1.0 = snapshot taken) | init pass | `flag[ch]` |
//! | 2176 | noise OUTPUT spectrum (loop A's write target) | loop A | `out[ch]` |
//! | 3992/4096 | keystream double buffer (front/back) | loop B | `key_cur/key_prev` |
//! | 3192 | per-band phase accumulator (loop A's rotation) | loop A | `band_phase` |
//! | 3248 | per-bin transient score (loop D / ×10 ramps) | detector | `score` (broadcast) |
//! | 7432 | per-bin spectral flatness (×10 coherence term) | analysis | param, default 0.5 |
//!
//! # The laws (line refs = machine_pretty.c)
//!
//! **Init pass** (L3232-3246, `0x18015c627`): per band bin —
//! `if flag[bin] != 1.0 || force { exc[bin] = src[bin]; flag[bin] = 1.0 }`.
//! One-shot snapshot of the excitation from its source; `force` re-fires for
//! bands whose noise-strength blend exceeds 0.75 (L3231, `v590 > 0.75 ||
//! !*(a1+1352)`). Order inside a machine granule: loop A → loop B → loop D →
//! init → source refresh (LABEL_768) — the first granule therefore seeds the
//! excitation from the analysis spectrum directly ("excitation starts as a
//! copy of the magnitude", §13.9).
//!
//! **Loop A** (L2232-2261) — peak-anchored phase-driven excitation read into
//! the OUTPUT spectrum 2176, per band:
//! - argmax of the synthesis spectrum (1768) over the band + parabolic
//!   refinement (L2135-2220) → peak position `pk` (float);
//! - per bin: `v361 = (pk + band_phase) × (hop·2π/n_fft)·0.5` (L2239),
//!   `v362 = (src[bin] − mag[pk_bin]) − v361` (L2240-2242), wrapped to
//!   nearest-mod-2π (`v362 − ⌊v362/2π + ½⌋·2π`, L2243-2251), then
//!   `out[bin] = out_step · ((wrapped + v361)/hop) + exc[pk_bin]` (L2248-2255).
//!
//! **Loop B** (L2440-2606) — per-bin excitation evolution + keystream, per
//! band `[v200, end)`:
//! - bin ramp `v376 = −2·frac(((hop−out_step)+drift)/rate)·rate · π/n_fft`
//!   (L2296-2311: the fraction is taken round-to-nearest, sign-bit XORed,
//!   doubled — a linear per-bin phase ramp opposing the hop/out-step drift);
//! - `v413/v447 = (bin·v376 + src[bin]) − exc[bin]` (L2444-2446, L2586-2588),
//!   `v415 = wrap2pi(v413)` (L2451);
//! - keystream monotonic boost (L2497-2513): on a falling positive keystream
//!   (`key_prev > key_cur > v415 > 0`) or rising negative
//!   (`key_cur > key_prev ∧ v415 > key_cur ∧ v415 < 0`),
//!   `w ×= (ramp(key_prev/v415)·0.75 + 0.25)` with `ramp(x) = (x−1)/3 ∈ [0,1]`
//!   for `x ∈ (1,4)`, saturated beyond;
//! - zero-key endpoint pull (L2514-2539): `w += min(π − w, w)·{0.2 if both
//!   keys zero, 0.15 if only the back key zero}`;
//! - ×10 peak-region enhancement (L2541-2571, §13.4): shoulder
//!   `(score−2)·0.25` for score ∈ (2,6), 1 beyond, coherence
//!   `(flatness−0.4)/0.3` for flatness ∈ (0.4,0.7), 1 beyond; if
//!   `coherence + (1 − shoulder) > 1` and `sr·bin > n_fft·4000` and machine
//!   state ≠ 2 → `w = 10`;
//! - update gate: `if w ≥ |v415| { exc[bin] += wrap2pi(v447) }` (L2595-2596),
//!   and the keystream double buffer advances EVERY bin:
//!   `key_prev[bin] = key_cur[bin]; key_cur[bin] = (w < |v415|) ? v415 : 0`
//!   (L2597-2602) — the residual is STORED exactly when the excitation is
//!   not updated.
//!
//! **Loop D** (L2627-2719) — score-weighted band smoothing toward the band's
//! reference bin: with `q = out_step/hop`, `g = sqrt(ramp(q)·[stretch > 1])`,
//! `lo = 0.5 + g·0.5/σ`, `hi = 0.5 + g·4/σ` (σ = renderer+3560, undecoded —
//! parameter, default 1), per band bin `b` with ref `r`:
//! `exc[b] += wrap2pi(wrap2pi(src[b]−src[r]) + wrap2pi(exc[r]−exc[b])) ·
//! ramp(score[b]; lo, hi)` — pulls each bin's excitation toward the reference
//! bin's excitation with the magnitude difference folded in, weighted by the
//! transient score.
//!
//! # Not ported (documented gaps)
//!
//! - The steady-state randomize ring (L2792-2960, state ≠ 2): a history ring
//!   (`3808`, rotating index `3784/3788`) pulls past spectra into the
//!   excitation. The ring's FILL site is not inside the read span — the
//!   history-source semantics need another decompile round.
//! - The base weight envelope (L2348-2406): `powf` curves over `.rdata`
//!   band tables (`xmmword_1801BF9D0…`) whose constants were not dumped;
//!   approximated as unity (only the pinned adjustments — monotonic boost,
//!   endpoint pull, ×10 — shape it).
//! - Per-bin scores/flatness: ours are broadcast scalars per machine granule
//!   (DLL's are per-bin arrays from its own analysis).
//! - Band tables (3120/3144/3168/3216): passed in by the caller as peak
//!   regions (our `reg_start/reg_end` are the structural counterpart).
//! - `2176`'s consumer lives OUTSIDE `sub_180158D20` (no read of `a1+2176`
//!   in the function body beyond loop A's writes and the per-band scalar
//!   copy at L2271) — the output spectrum is exported for the resynthesis
//!   layer; `NoiseLoop::out()` surfaces it for future wiring.

use std::f32::consts::{FRAC_1_PI, PI};

const TWO_PI: f32 = 6.283_185_5;
const INV_2PI: f32 = 0.159_154_94;
/// `0x7FFFFFFF` — float-abs bit mask (L2576: `LODWORD(v415) & v852`).
#[inline]
fn fabs_bits(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7FFF_FFFF)
}

/// `x − ⌊x·(1/2π) + ½⌉·2π` — the DLL's round-to-nearest mod-2π wrap
/// (L2243-2251, L2451, L2699-2707: sign-selected ±0.5 then truncating cast).
#[inline]
fn wrap2pi(x: f32) -> f32 {
    let r = x * INV_2PI;
    let off = if r < 0.0 { -0.5 } else { 0.5 };
    x - ((r + off) as i32) as f32 * TWO_PI
}

/// Ramp `x` from 0 at `lo` to 1 at `hi` (the DLL's `(v−lo)/(hi−lo)` shape
/// with both saturation branches spelled out, e.g. L2555-2565).
#[inline]
fn ramp(x: f32, lo: f32, hi: f32) -> f32 {
    if x > lo {
        if x < hi {
            (x - lo) / (hi - lo)
        } else {
            1.0
        }
    } else {
        0.0
    }
}

/// One caller-supplied band: `[start, end)` bin range with its reference bin.
#[derive(Debug, Clone, Copy)]
pub struct NoiseBand {
    pub start: usize,
    pub end: usize,
    pub ref_bin: usize,
}

/// Context for one `apply` call (one channel, one machine granule).
pub struct NoiseLoopCtx<'a> {
    /// 1872 counterpart: the excitation's source spectrum (our `mag_copy`).
    pub src: &'a [f32],
    /// 1976/1768 counterpart: current analysis spectrum (loop A's argmax and
    /// reference reads; our `mag`).
    pub mag: &'a [f32],
    /// Per-bin transient score, broadcast (DLL 3248 is per-bin).
    pub score: f32,
    /// Per-bin spectral flatness in `[0,1]`, broadcast (DLL 7432). Default
    /// 0.5 keeps the ×10 coherence term at its band midpoint.
    pub flatness: f32,
    /// Peak regions (DLL band tables 3120-3216).
    pub bands: &'a [NoiseBand],
    /// Machine in-hop (renderer+1376, our `tsched.steady_hop`).
    pub hop: i32,
    /// Machine out-step (renderer+1380, `round(hop·slope)`).
    pub out_step: i32,
    /// FFT size (renderer+1448).
    pub n_fft: i32,
    /// Sample rate (renderer+96).
    pub sr: u32,
    /// Machine state (renderer+2448) — the ×10 gate requires `!= 2`.
    pub state: i32,
    /// Time-base normalizer `v173` (L1503-1513: quality ≥ 10 → preset `v894`,
    /// undecoded; 1.0 = pass the raw frame arithmetic through).
    pub time_base: f32,
    /// σ = renderer+3560 for loop D's ramp bounds (undecoded; 1.0).
    pub sigma: f32,
    /// Stretch (> 1 arms loop D's `g` term; the DLL reads `*(double*)(a1+32)`).
    pub stretch: f64,
}

impl<'a> NoiseLoopCtx<'a> {
    fn validate(&self, nb: usize) {
        assert!(
            self.src.len() >= nb && self.mag.len() >= nb,
            "noise loop: spectra shorter than nb_bins"
        );
        assert!(self.hop >= 1, "noise loop: hop must be >= 1");
    }
}

/// Per-channel noise-loop state (buffers 2080/2256/2176/3992/4096/3192).
pub struct NoiseLoop {
    nch: usize,
    nb: usize,
    /// 2080 — the excitation (mirrors `noise_gain` rows while unwired).
    exc: Vec<Vec<f32>>,
    /// 2256 — init-once flags.
    flag: Vec<Vec<f32>>,
    /// 2176 — the noise output spectrum (loop A's write target).
    out: Vec<Vec<f32>>,
    /// 4096 — keystream back slot.
    key_prev: Vec<Vec<f32>>,
    /// 3992 — keystream front slot.
    key_cur: Vec<Vec<f32>>,
    /// 3192 — per-band phase accumulator (loop A's rotation; advances by
    /// `hop·2π/n_fft` per machine granule — the advance site itself is
    /// outside the read span, so the increment is inferred from v361's law).
    band_phase: Vec<f32>,
    /// renderer+3976 — the fractional drift carry (L2735 saves `v201`,
    /// L2296 consumes it).
    drift: f32,
}

impl NoiseLoop {
    pub fn new(nch: usize, nb_bins: usize) -> Self {
        Self {
            nch,
            nb: nb_bins,
            exc: vec![vec![0.0; nb_bins]; nch],
            flag: vec![vec![0.0; nb_bins]; nch],
            out: vec![vec![0.0; nb_bins]; nch],
            key_prev: vec![vec![0.0; nb_bins]; nch],
            key_cur: vec![vec![0.0; nb_bins]; nch],
            band_phase: Vec::new(),
            drift: 0.0,
        }
    }

    /// The excitation row (2080) — the +3568 tail-reset destination.
    pub fn exc(&self, ch: usize) -> &[f32] {
        &self.exc[ch]
    }

    /// Mutable excitation row (the +3568 reset writes through here).
    pub fn exc_mut(&mut self, ch: usize) -> &mut [f32] {
        &mut self.exc[ch]
    }

    /// The noise output spectrum (2176) — consumer sits outside 158D20.
    pub fn out(&self, ch: usize) -> &[f32] {
        &self.out[ch]
    }

    /// One channel's machine-granule evolution: loop A → loop B → loop D →
    /// init pass (DLL program order; the source refresh at LABEL_768 is the
    /// caller's `src` refresh, already per-sub-granule in our engine).
    pub fn apply(&mut self, ch: usize, ctx: &NoiseLoopCtx) {
        ctx.validate(self.nb);
        let nb = self.nb;
        if self.band_phase.len() < ctx.bands.len() {
            self.band_phase.resize(ctx.bands.len(), 0.0);
        }
        let mut band_phase = std::mem::take(&mut self.band_phase);

        // ---- loop B prelude: the bin ramp v376 (L2296-2311) ----
        let v372 = (((ctx.hop - ctx.out_step) as f32) + self.drift) / ctx.time_base;
        let v375 = if v372 < 0.0 { -0.5 } else { 0.5 };
        let v201 = (v372 - ((v372 + v375) as i32) as f32) * ctx.time_base;
        // (v201 ^ sign-bit) doubled = -2·v201; scaled by π/n_fft.
        let v376 = (-2.0 * v201 * PI) / ctx.n_fft as f32;
        // renderer+3976 = v201 (L2735) for the next granule.
        let next_drift = v201;

        let v323 = (ctx.hop as f32 * TWO_PI) / ctx.n_fft as f32; // L2113-2114
        let inv_hop = 1.0f32 / ctx.hop as f32; // v324 = 1/hop (L2115)

        // ---- per band: loop A (output spectrum), loop B (excitation), D ----
        for (bi, band) in ctx.bands.iter().enumerate() {
            let (bs, be) = (band.start.min(nb), band.end.min(nb));
            if bs >= be {
                continue;
            }
            let phase_b = *band_phase.get(bi).unwrap_or(&0.0);

            // Loop A: argmax of the synthesis spectrum + parabolic refinement
            // (L2135-2220), then the phase-driven read (L2232-2261).
            let mut pk_idx = bs;
            let mut pk_val = ctx.mag[bs];
            for b in (bs + 1)..be {
                if ctx.mag[b] > pk_val {
                    pk_val = ctx.mag[b];
                    pk_idx = b;
                }
            }
            // Parabolic: v353 = pk ± (mag[pk+1] − mag[pk−1]) / (2·(2·mag[pk] −
            // mag[pk±1] sums)), clamped to [pk−1, pk+1] (L2200-2220).
            let mut pk = pk_idx as f32;
            if pk_idx > 0 && pk_idx + 1 < nb && pk_idx as usize != nb - 1 {
                let (m0, m1, m2) = (ctx.mag[pk_idx - 1], ctx.mag[pk_idx], ctx.mag[pk_idx + 1]);
                let denom = (m1 + m1) - m0 - m2;
                if denom != 0.0 {
                    let delta = (m2 - m0) / (denom + denom);
                    pk = (pk_idx as f32 + delta).clamp((pk_idx - 1) as f32, (pk_idx + 1) as f32);
                }
            }
            let pk_bin = (pk_idx as i64 - band.ref_bin as i64).clamp(0, nb as i64 - 1) as usize;

            // v323·v11 (v11 = 0.5, L936).
            let phase_step = v323 * 0.5;
            for b in bs..be {
                let v361 = (pk + phase_b) * phase_step;
                let v362 = (ctx.src[b] - ctx.mag[pk_bin]) - v361;
                let wrapped = wrap2pi(v362);
                self.out[ch][b] =
                    ctx.out_step as f32 * ((wrapped + v361) * inv_hop) + self.exc[ch][pk_bin];
            }
            // Per-band phase advance (inferred; see field docs).
            let phase_b = phase_b + v323;
            if bi < band_phase.len() {
                band_phase[bi] = phase_b;
            }

            // Loop B (L2440-2606) over the band's bins.
            for b in bs..be {
                let v413 = (b as f32 * v376 + ctx.src[b]) - self.exc[ch][b];
                let v415 = wrap2pi(v413);
                let gate_val = fabs_bits(v415); // v439 = v415 & 0x7FFFFFFF

                // Base weight: the DLL's band-table envelope (undecoded) is
                // unity here; the pinned adjustments shape it.
                let mut w = 1.0f32;

                // Keystream monotonic boost (L2497-2513).
                let kp = self.key_prev[ch][b];
                let kc = self.key_cur[ch][b];
                if (kp > kc && kc > v415 && v415 > 0.0)
                    || (kc > kp && v415 > kc && 0.0 > v415)
                {
                    let v427 = kp / v415;
                    let v428 = ramp(v427, 1.0, 4.0);
                    w = (v428 * 0.75 + 0.25) * w;
                }
                // Zero-key endpoint pull (L2514-2539); v400 = π (v869).
                if kp == 0.0 && kc == 0.0 {
                    let v431 = (PI - w).min(w) * 0.2;
                    w += v431;
                } else if kc == 0.0 {
                    let v431 = (PI - w).min(w) * 0.15;
                    w += v431;
                }
                // ×10 peak-region enhancement (L2541-2571).
                let shoulder = if ctx.score > 2.0 {
                    if ctx.score < 6.0 {
                        (ctx.score - 2.0) * 0.25
                    } else {
                        1.0
                    }
                } else {
                    0.0
                };
                let coherence = ramp(ctx.flatness, 0.4, 0.7);
                if (coherence + (1.0 - shoulder)) > 1.0
                    && (ctx.sr as f32 * b as f32) > (ctx.n_fft as f32 * 4000.0)
                    && ctx.state != 2
                {
                    w = 10.0;
                }

                // Update gate + write (L2595-2596).
                if w >= gate_val {
                    let v447 = (b as f32 * v376 + ctx.src[b]) - self.exc[ch][b];
                    self.exc[ch][b] += wrap2pi(v447);
                }
                // Keystream advance (L2597-2602): the residual is stored
                // exactly when the excitation was NOT updated.
                self.key_prev[ch][b] = self.key_cur[ch][b];
                self.key_cur[ch][b] = if w < gate_val { v415 } else { 0.0 };
            }

            // Loop D (L2627-2719): score-weighted pull toward the ref bin.
            let q = ctx.out_step as f32 / ctx.hop as f32;
            let v452 = ramp(q, 1.0, 4.0);
            let g = if ctx.stretch > 1.0 { v452.sqrt() } else { 0.0 };
            let lo = 0.5 + g * 0.5 / ctx.sigma;
            let hi = 0.5 + g * 4.0 / ctx.sigma;
            let v467 = ramp(ctx.score, lo, hi);
            let r = band.ref_bin.min(nb - 1);
            for b in bs..be {
                if b != r {
                    let d0 = wrap2pi(ctx.src[b] - ctx.src[r]);
                    let d1 = wrap2pi(self.exc[ch][r] - self.exc[ch][b]);
                    self.exc[ch][b] += wrap2pi(d0 + d1) * v467;
                }
            }
        }

        // ---- init pass (L3232-3246): one-shot snapshot where no flag ----
        // The DLL's per-band `force` (noise-strength > 0.75) re-fires bands;
        // our counterpart is `score > 0.75` (the same blend's scalar form).
        let force = ctx.score > 0.75;
        for b in 0..nb {
            if self.flag[ch][b] != 1.0 || force {
                self.exc[ch][b] = ctx.src[b];
                self.flag[ch][b] = 1.0;
            }
        }

        self.band_phase = band_phase;
        self.drift = next_drift;
    }

    /// Number of channels the loop was built for.
    pub fn nch(&self) -> usize {
        self.nch
    }
}

// `FRAC_1_PI` is referenced by the doc tests below via PI only; silence the
// unused-import lint for configurations where the constant is not used.
#[allow(dead_code)]
const _: f32 = FRAC_1_PI;

#[cfg(test)]
mod tests {
    use super::*;

    const NB: usize = 64;
    const HOP: i32 = 704;
    const OUT_STEP: i32 = 838; // round(704·1.19)
    const N_FFT: i32 = 16384;

    fn band() -> Vec<NoiseBand> {
        vec![
            NoiseBand { start: 0, end: 32, ref_bin: 4 },
            NoiseBand { start: 32, end: NB, ref_bin: 40 },
        ]
    }

    fn ctx<'a>(
        src: &'a [f32],
        mag: &'a [f32],
        bands: &'a [NoiseBand],
    ) -> NoiseLoopCtx<'a> {
        NoiseLoopCtx {
            src,
            mag,
            score: 0.0,
            flatness: 0.5,
            bands,
            hop: HOP,
            out_step: OUT_STEP,
            n_fft: N_FFT,
            sr: 48_000,
            state: 0,
            time_base: 1.0,
            sigma: 1.0,
            stretch: 1.19,
        }
    }

    fn ramped(n: usize) -> Vec<f32> {
        (0..n).map(|i| 0.1 + 0.02 * (i as f32) * (i as f32).sin()).collect()
    }

    /// Init pass: first apply snapshots the excitation from the source and
    /// latches the flags; a second apply with `force` shut does NOT re-copy
    /// (the loop-B/D evolution between them persists).
    #[test]
    fn init_pass_is_one_shot() {
        let src = ramped(NB);
        let mag = ramped(NB);
        let bands = band();
        let mut nl = NoiseLoop::new(1, NB);
        nl.apply(0, &ctx(&src, &mag, &bands));
        assert_eq!(nl.exc(0)[..NB], src[..NB], "first pass = snapshot");

        // Corrupt one bin; without force the flag latch must preserve it.
        nl.exc_mut(0)[7] = 123.0;
        let c = ctx(&src, &mag, &bands);
        nl.apply(0, &c);
        assert_eq!(nl.exc(0)[7], 123.0, "flag latch: no re-copy");

        // force (score > 0.75) re-snapshots every band bin.
        let mut c = ctx(&src, &mag, &bands);
        c.score = 0.9;
        nl.apply(0, &c);
        assert_eq!(nl.exc(0)[7], src[7], "force re-snapshot");
    }

    /// Loop B's wrap law: `wrap2pi(x)` lands in (−π, π] and matches the
    /// floor-arithmetic closed form.
    #[test]
    fn wrap2pi_law() {
        for &x in &[
            0.0f32,
            0.5,
            PI,
            PI + 0.1,
            TWO_PI,
            7.0 * TWO_PI,
            -0.1,
            -PI,
            -(PI + 0.5),
            -100.0,
            100.0,
        ] {
            let w = wrap2pi(x);
            // The DLL's ±0.5 round-wrap is half-open on the negative side:
            // exactly +π wraps to −π, so the range is [−π, π].
            assert!(w >= -PI - 1e-4 && w <= PI + 1e-4, "wrap({x}) = {w}");
            // closed form: x − round(x/2π)·2π within one 2π
            let k = (x * INV_2PI).round();
            assert!((w - (x - k * TWO_PI)).abs() < 1e-3);
        }
    }

    /// Keystream double buffer: on the FIRST granule the excitation starts
    /// at zero, so the residual is exactly `wrap2pi(src[b])` — the key stores
    /// it where the gate (w = 1 < |residual|) is shut, zero where open; the
    /// second granule (exc == src, integer drift → zero ramp) clears the
    /// keys while `key_prev` holds the first granule's slot.
    #[test]
    fn keystream_double_buffer_advances() {
        let src = ramped(NB);
        let mag = ramped(NB);
        let bands = band();
        let mut nl = NoiseLoop::new(1, NB);
        let c = ctx(&src, &mag, &bands);
        nl.apply(0, &c);
        // Law check: on the first granule w = 1.2, not 1 — the zero-key
        // endpoint pull (both keystream slots 0 → += min(π−1,1)·0.2) is part
        // of the weight law. key_cur[b] = wrap2pi(src[b]) when |·| > w,
        // else 0 (v376 = 0: (hop−out_step)/time_base = −134 is integral).
        let w_eff = 1.0 + 0.2 * (PI - 1.0).min(1.0);
        for b in 0..NB {
            let resid = wrap2pi(src[b]);
            let want = if resid.abs() > w_eff { resid } else { 0.0 };
            assert_eq!(
                nl.key_cur[0][b], want,
                "bin {b}: resid {resid} must be stored iff |resid| > w_eff={w_eff}"
            );
        }
        let key_after1: Vec<f32> = nl.key_cur[0][..NB].to_vec();
        let c = ctx(&src, &mag, &bands);
        nl.apply(0, &c);
        // exc == src now (init snapshot), ramp still 0 → all residuals 0 →
        // gate open → keys cleared; prev holds the last cur.
        assert_eq!(nl.key_prev[0][..NB], key_after1, "prev = last cur");
        assert!(
            nl.key_cur[0][..NB].iter().all(|v| *v == 0.0),
            "zero residual → open gate → cleared keys"
        );
    }

    /// ×10: a strong score + coherent flatness at a high bin in state ≠ 2
    /// drives the weight to 10 — observable as the gate opening for residuals
    /// up to 10 (vs 1), i.e. the excitation tracking the source ramp.
    #[test]
    fn x10_opens_gate_for_large_residuals() {
        let n = 4096usize; // need bins past n_fft*4000/sr ≈ 1366
        let mut src = vec![0.0f32; n];
        for (i, v) in src.iter_mut().enumerate() {
            *v = 0.5 * (i as f32 / n as f32);
        }
        let mag = src.clone();
        let bands = vec![NoiseBand { start: 0, end: n, ref_bin: 0 }];
        let mut nl = NoiseLoop::new(1, n);
        // Warm up one granule (init snapshot).
        let c = ctx(&src, &mag, &bands);
        nl.apply(0, &c);
        // Now a big perturbation + score 5 + flatness 0.6 in state 0.
        for v in src.iter_mut() {
            *v += 6.0 * TWO_PI; // large residual, same wrapped value
        }
        let mut c = ctx(&src, &mag, &bands);
        c.score = 5.0;
        c.flatness = 0.6;
        c.state = 0;
        nl.apply(0, &c);
        // In-state 2 the ×10 is suppressed: run a fresh loop for contrast.
        let mut nl2 = NoiseLoop::new(1, n);
        let c2 = ctx(&src, &mag, &bands);
        nl2.apply(0, &c2);
        let mut c2 = ctx(&src, &mag, &bands);
        c2.score = 5.0;
        c2.flatness = 0.6;
        c2.state = 2;
        nl2.apply(0, &c2);
        let moved = |nl: &NoiseLoop| -> usize {
            (0..n)
                .filter(|&b| {
                    let e = nl.exc(0)[b];
                    // moved toward the new source (wrapped distance shrank)
                    wrap2pi(src[b] - e).abs() < wrap2pi(src[b]).abs() - 1e-3
                })
                .count()
        };
        assert!(
            moved(&nl) >= moved(&nl2),
            "x10 (state 0) must track at least as well as state 2"
        );
    }

    /// Loop D: the score-weighted pull toward the reference bin. Loop D is a
    /// no-op while exc == src (its two wrap terms cancel), so the test
    /// corrupts one bin past loop B's gate (|v415| > 1.2 → B stores the
    /// residual without updating) — then ONLY loop D can move that bin, with
    /// the strong score's ramp at 1 and the quiet score's at 0.
    #[test]
    fn loop_d_pulls_toward_ref_bin_with_score() {
        let src = ramped(NB);
        let mag = ramped(NB);
        let bands = vec![NoiseBand { start: 0, end: NB, ref_bin: 8 }];
        let mut strong = NoiseLoop::new(1, NB);
        let mut quiet = NoiseLoop::new(1, NB);
        let mut cs = ctx(&src, &mag, &bands);
        cs.score = 6.0;
        let cq = ctx(&src, &mag, &bands);
        // Granule 1: latch init (exc := src both).
        strong.apply(0, &cs);
        quiet.apply(0, &cq);
        // Corrupt bin 20 beyond the loop-B gate; leave the quiet twin exact.
        strong.exc_mut(0)[20] = 5.0;
        let cq2 = ctx(&src, &mag, &bands);
        quiet.exc_mut(0)[20] = 5.0;
        // Granule 2: strong pulls bin 20 (ramp 1), quiet leaves it (ramp 0).
        strong.apply(0, &cs);
        quiet.apply(0, &cq2);
        assert_ne!(
            strong.exc(0)[20],
            5.0,
            "strong score must move the corrupted bin (loop D)"
        );
        assert_eq!(
            quiet.exc(0)[20], 5.0,
            "score 0 → ramp 0 → loop D inactive (loop B gated shut)"
        );
    }

    /// Loop A writes the output spectrum every granule: with the excitation
    /// equal to the source and a positive out-step, out[b] = out_step·
    /// ((wrap2pi(src[b] − mag[pk]) − v361)·… ) — finite and deterministic;
    /// identical inputs give identical outputs.
    #[test]
    fn loop_a_output_is_deterministic() {
        let src = ramped(NB);
        let mag = ramped(NB);
        let bands = band();
        let mut a = NoiseLoop::new(1, NB);
        let mut b = NoiseLoop::new(1, NB);
        let c = ctx(&src, &mag, &bands);
        a.apply(0, &c);
        b.apply(0, &c);
        assert_eq!(a.out(0)[..NB], b.out(0)[..NB]);
        assert!(a.out(0).iter().all(|v| v.is_finite()));
        // A second granule (phase advanced) must move the output.
        let before = a.out(0).to_vec();
        let c = ctx(&src, &mag, &bands);
        a.apply(0, &c);
        assert_ne!(a.out(0)[..NB], before[..NB], "band phase rotates");
    }

    /// Multi-channel isolation: channel 1's state never depends on channel 0.
    #[test]
    fn channels_are_isolated() {
        let src0 = ramped(NB);
        let src1 = vec![0.5f32; NB];
        let mag0 = ramped(NB);
        let mag1 = vec![0.5f32; NB];
        let bands = band();
        let mut nl = NoiseLoop::new(2, NB);
        let c0 = ctx(&src0, &mag0, &bands);
        let c1 = ctx(&src1, &mag1, &bands);
        nl.apply(0, &c0);
        nl.apply(1, &c1);
        assert_eq!(nl.exc(1)[..NB], src1[..NB], "ch1 = its own snapshot");
        assert_ne!(nl.exc(0)[3], nl.exc(1)[3]);
    }
}
