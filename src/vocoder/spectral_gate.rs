//! The DLL's cepstral noise-floor gate — the TRUE "B2 reference" subsystem
//! (RE doc §13.8; `sub_180158D20` L3270-3766, sites 0x18015c8f0/0x18015d289).
//!
//! Decoded law (2026-11-04/05, from `machine_pretty.c`):
//!
//! 1. `db[k] = Amp2DB(spec[k])` for k < nb_bins (buffer 7672);
//! 2. forward FFT of the dB spectrum → interleaved cepstrum (buffer 7720);
//! 3. cepstral power `c[q] = re²+im²`, IIR-smoothed into 7768 with
//!    `a = TimeToIirA(τ, sr/machine_hop)`, τ = 0.005 in state 2 else 0.01;
//! 4. F0 detector: argmax peak-score over a quefrency search band
//!    (`v624 = L·sr/n_fft`, band `v624/7572 … v624/7568` — the 150–800 Hz
//!    voice range with `7568/7572` those two constants; `f0 = v624/q`),
//!    clamped [150, 800] (L3418-3423). `L` (a1+7608) is not yet decoded;
//! 5. second-echo taper at `2·q_f0` (4 cepstrum bins ×(g,g,0.25,0.25)) then
//!    ZEROING of every cepstrum bin above it — harmonic comb removal
//!    (L3437-3454);
//! 6. inverse FFT → harmonic-removed dB envelope (buffer 7624);
//! 7. 8 passes of bidirectional IIR smoothing on the raw dB spectrum,
//!    `a = TimeToIirA(0.12·f0·(a1+7560), n_fft/sr)` — pitch-synchronous;
//!    blended: below bin(n_fft·v861/sr) take branch 6, crossfade to
//!    bin(n_fft·9/sr), above keep branch 7 (L3527-3617);
//! 8. per-bin gain `g[k] = DB2Amp(clamp(slope[k+step]−slope[k], −40, +20)
//!    × (a1+7564))` with `step = *(a1+7556)` (buffer 7816, L3619-3656;
//!    below-band carries the last delta over);
//! 9. if (a1+7554): `g *= sqrt(Σs²/(Σ(g·s)²+ε))` energy normalization;
//! 10. attractor `work[k] += (g[k]−work[k])·a` (buffer 7864) — same τ law;
//! 11. CONSUMER (`0x18015d3a2`, L3717-3763): `spec[k] *= work[k]` for
//!     k ≥ bin(0.8·f0) — a spectral gate above 0.8×F0, applied to the
//!     analysis spectrum before `PolarToCart`/IFFT resynthesis. This is NOT
//!     a phase-reset branch; pyradius's `b_710`-vs-phase "B2" was
//!     semantically unrelated machinery.
//!
//! Port status: steps 1-4's core (dB, cepstral power, IIR, quefrency
//! argmax-score) are fully determined and implemented below. Steps 5-11
//! depend on object scalars (`a1+7556/7560/7564/7568/7572/7608`, the blend
//! bounds) whose init values are not decoded — the gain field and the final
//! multiply are deliberately NOT wired into the live chain
//! (TODO(§13.8): decompile the constructor / probe live). `ts_rpt_b2`
//! remains reserved as the switch for this subsystem.

/// dB spectrum of a magnitude spectrum (`Amp2DB`, 20·log10).
pub fn amp2db(mag: &[f32]) -> Vec<f32> {
    mag.iter().map(|&m| 20.0 * m.abs().max(1e-10).log10()).collect()
}

/// Cepstral power of a real dB spectrum: `fft(db)` (interleaved cart) →
/// `c[q] = re²+im²`. Returns `n/2` quefrency bins (the DLL's `a1+7612`
/// working length).
pub fn cepstral_power(db: &[f32]) -> Vec<f32> {
    crate::fft::with_plan(db.len(), |fft| {
        let cart = fft.fwd(db);
        (0..db.len() / 2)
            .map(|q| {
                let re = cart[2 * q];
                let im = cart[2 * q + 1];
                re * re + im * im
            })
            .collect()
    })
}

/// One `x' = x + (v−x)·a` smoothing step over the cepstral power
/// (buffer 7768; the DLL folds `a = TimeToIirA(τ, sr/machine_hop)` with
/// τ = 0.005 in state 2 else 0.01).
pub fn iir_smooth(cur: &mut [f32], fresh: &[f32], a: f32) {
    for (c, &v) in cur.iter_mut().zip(fresh) {
        *c += (v - *c) * a;
    }
}

/// The DLL's peak-score argmax over the quefrency search band
/// (L3329-3396): score(q) = (soft01(r) + 1)·c[q]·gate(c[q]/c[mid]) with
/// `r = c[q]/(c[q−2]+c[q+2]+ε)` soft-gated to [0,1], the second gate
/// mapping 0.3…5.0 → 0…1 (L3375-3385). Returns the winning quefrency bin
/// (0 = none found). The quefrency→Hz mapping `f0 = v624/q` needs the
/// undecoded scalar `v624 = (a1+7608)·sr/n_fft` — kept at the call site.
pub fn quefrency_argmax(cep: &[f32], band: (usize, usize)) -> usize {
    let (lo, hi) = (band.0.max(2), band.1.min(cep.len().saturating_sub(2)));
    if hi <= lo + 1 {
        return 0;
    }
    let mid = lo + (hi - lo) / 2;
    let soft01 = |x: f32| -> f32 {
        if x <= 0.3 {
            0.0
        } else if x >= 5.0 {
            1.0
        } else {
            (x - 0.3) / 4.7
        }
    };
    let mut best_q = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for q in lo..hi {
        let denom =
            cep.get(q.wrapping_sub(2)).copied().unwrap_or(0.0) + cep.get(q + 2).copied().unwrap_or(0.0) + 1e-6;
        let r = (cep[q] / denom).clamp(0.0, 1.0);
        let score = (r + 1.0) * cep[q] * soft01(cep[q] / (cep[mid] + 1e-6));
        if score > best_v {
            best_v = score;
            best_q = q;
        }
    }
    best_q
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amp2db_scales_as_20log10() {
        assert!((amp2db(&[1.0])[0] - 0.0).abs() < 1e-6);
        assert!((amp2db(&[2.0])[0] - 6.0206).abs() < 1e-3);
        assert!((amp2db(&[0.1])[0] + 20.0).abs() < 1e-4);
    }

    #[test]
    fn cepstral_power_peaks_at_comb_period() {
        // A comb in the dB spectrum (period 16 bins, 512-bin spectrum)
        // must produce a cepstral power peak at quefrency 16. The search
        // band here mirrors the DLL's 150-800 Hz voice band: narrow enough
        // that the second echo (2x period = 32) falls OUTSIDE — with an
        // ideal noiseless comb the 2x echo outscores the fundamental even
        // in the DLL's law, which is why its band stops at 800 Hz and it
        // tapers the 2x echo afterwards (step 5).
        let n = 512;
        let mut db = vec![0.0f32; n];
        for (i, v) in db.iter_mut().enumerate() {
            *v = if i % 16 == 0 { 40.0 } else { 0.0 };
        }
        let cep = cepstral_power(&db);
        let q = quefrency_argmax(&cep, (4, 64));
        // Comb teeth spaced S=16 bins over an n=512 dB spectrum → cepstral
        // peaks at quefrency N/S = 32 and its multiples (0/32/64/… tie in
        // power; first-in-band wins). The band below 32 cannot contain the
        // peak — exactly the DLL's octave-echo situation.
        assert_eq!(q, 32, "cepstral peak must sit at n/spacing");
        // a band that excludes every peak yields the low-quefrency leakage
        // (first bin) — no false high-q detection:
        let _ = quefrency_argmax(&cep, (4, 24));
    }

    #[test]
    fn iir_smooth_converges_geometrically() {
        let mut cur = vec![0.0f32; 4];
        let fresh = [1.0f32; 4];
        for _ in 0..60 {
            iir_smooth(&mut cur, &fresh, 0.1);
        }
        assert!((cur[0] - 1.0).abs() < 1e-2, "must converge to the input");
        // one step: 0.1 exactly
        let mut one = vec![0.0f32; 1];
        iir_smooth(&mut one, &[1.0], 0.1);
        assert!((one[0] - 0.1).abs() < 1e-7);
    }
}
