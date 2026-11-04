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
//! Port status: steps 1-11 complete (2026-11-05). Object scalars from the
//! live probe: 7552(enable)=1, 7554(energy-norm)=1, 7555(det-fallback
//! byte 7553)=1, 7556=pitch ratio, 7560=1.0, 7564=1.0, 7568=40.0,
//! 7572=800.0, 7608(L)=4096; from the decompile: taper v591=0.75
//! (L3096/3230/3399 → v848, L979), blend bound v861=4.0 Hz (L977),
//! ε=1e-12 (L963), steady τ v613=0.01 (L957→v866, L3300), f0 clamp
//! [150, 832] (L3415-3423: 0x43160000=150, 0x44500000=832). The gate
//! runs on the per-channel analysis magnitudes before resynthesis;
//! `ts_rpt_b2` (default false) is the switch.

use crate::fft::with_plan;
use super::acs::time_to_iir_a;

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

/// `DB2Amp` — inverse of [`amp2db`].
pub fn db2amp(db: f32) -> f32 {
    10.0f32.powf(db / 20.0)
}

fn rint(x: f32) -> i64 {
    (x + if x < 0.0 { -0.5 } else { 0.5 }) as i64
}

/// Step 5: taper the two quefrency bins below float index `t` (×0.75/×0.25)
/// and zero the cepstrum cart from `t` upward.
pub fn taper_and_zero(cart: &mut [f32], t: usize) {
    if t >= 4 && t < cart.len() {
        for (off, w) in [(4usize, 0.75f32), (3, 0.75), (2, 0.25), (1, 0.25)] {
            cart[t - off] *= w;
        }
        for v in cart[t..].iter_mut() {
            *v = 0.0;
        }
    }
}

/// Step 7 core: one forward + one backward one-pole pass over `db`.
pub fn bidir_smooth_pass(db: &mut [f32], a: f32) {
    for i in 1..db.len() {
        db[i] += (db[i - 1] - db[i]) * a;
    }
    for i in (0..db.len() - 1).rev() {
        db[i] += (db[i + 1] - db[i]) * a;
    }
}

/// Steps 8-9 core: the gain field with below-band carry and the [−40, +20]
/// dB clamp (scale 7564 = 1.0).
pub fn gain_field(db: &[f32], pitch_ratio: f32) -> Vec<f32> {
    let nb = db.len();
    let mut last_delta = 0.0f32;
    let mut g = vec![1.0f32; nb];
    for (k, v) in g.iter_mut().enumerate() {
        let t = rint(k as f32 * pitch_ratio) as i64;
        if t >= 0 && (t as usize) < nb {
            last_delta = (db[t as usize] - db[k]).clamp(-40.0, 20.0);
        }
        *v = db2amp(last_delta);
    }
    g
}

/// Step 9: energy normalization `g *= sqrt(Σs²/(Σ(gs)²+ε))`.
pub fn energy_normalize(gains: &mut [f32], mag: &[f32]) {
    let (mut s1, mut s2) = (0.0f32, 0.0f32);
    for k in 0..mag.len() {
        s1 += mag[k] * mag[k];
        s2 += gains[k] * gains[k] * mag[k] * mag[k];
    }
    let scale = (s1 / (s2 + 1e-12)).sqrt();
    for g in gains.iter_mut() {
        *g *= scale;
    }
}

/// The full cepstral gate, one instance per renderer (the DLL's cepstral
/// power buffer 7768 and gain buffer 7816 are renderer-level/shared; only
/// the attractor work spectra 7864 are per channel).
pub struct SpectralGate {
    /// Smoothed cepstral power (buffer 7768), shared across channels.
    cep: Vec<f32>,
    /// Per-channel attractor state (buffer 7864, `+8*ch` pointer array).
    work: Vec<Vec<f32>>,
    /// Scratch: the working dB spectrum (7672) across the 8 smoothing
    /// passes and the harmonic-removed blend.
    db_work: Vec<f32>,
    /// Scratch: zero-padded real dB input for the forward FFT (n_fft).
    pad: Vec<f32>,
    /// Scratch: packed cepstrum cart (n_fft+2 — our FFT's half-spectrum
    /// format; the DLL's full 2N-interleaved buffer with the mirror bins
    /// zeroed is equivalent for the real envelope, since its inverse
    /// discards the imaginary part).
    cart: Vec<f32>,
    /// Scratch: inverse-FFT harmonic-removed envelope (7624).
    env: Vec<f32>,
}

impl SpectralGate {
    pub fn new(n_fft: usize, channels: usize) -> Self {
        Self {
            cep: vec![0.0; n_fft / 2],
            work: vec![vec![0.0; n_fft / 2 + 1]; channels],
            db_work: vec![0.0; n_fft / 2 + 1],
            pad: vec![0.0; n_fft],
            cart: vec![0.0; n_fft + 2],
            env: vec![0.0; n_fft],
        }
    }

    /// Steps 1-11 on one channel's analysis magnitudes. `machine_hop` is the
    /// transient machine's granule (renderer+1376 ≈ 704 @48k), `pitch_ratio`
    /// the current 2^(P/12), `state2` the machine state.
    pub fn apply(
        &mut self,
        ch: usize,
        mag: &mut [f32],
        sr: u32,
        n_fft: usize,
        machine_hop: i64,
        state2: bool,
        pitch_ratio: f32,
    ) {
        // Score and target on the same buffer (the historical single-slice
        // semantics; unit tests use this form).
        let score = mag.to_vec();
        self.apply_split(ch, &score, mag, sr, n_fft, machine_hop, state2, pitch_ratio);
    }

    /// The DLL's actual consumer topology (2026-11-05 live decode): the
    /// eleven-step law SCORES the raw analysis copy (1664, refreshed from
    /// 1768 at LABEL_768) and step 11 MULTIPLIES the gains into the
    /// synthesis spectrum (`spec[k] *= work[k]` at 0x18015d3a2). Scoring
    /// input and application target are therefore distinct buffers.
    pub fn apply_split(
        &mut self,
        ch: usize,
        score: &[f32],
        target: &mut [f32],
        sr: u32,
        n_fft: usize,
        machine_hop: i64,
        state2: bool,
        pitch_ratio: f32,
    ) {
        let nb = score.len().min(target.len());
        let n_half = n_fft / 2;
        // --- steps 1-3: dB, forward FFT (zero-padded to n_fft), cepstral
        //     power, IIR into 7768 (τ 0.005 in state 2 else 0.01, rate
        //     sr/machine-hop) ---
        let db = amp2db(&score[..nb]);
        self.pad[..nb].copy_from_slice(&db);
        for v in self.pad[nb..].iter_mut() {
            *v = 0.0;
        }
        let fresh_cep = with_plan(n_fft, |p| {
            p.fwd_into(&self.pad, &mut self.cart);
            (0..n_half)
                .map(|q| {
                    let re = self.cart[2 * q];
                    let im = self.cart[2 * q + 1];
                    re * re + im * im
                })
                .collect::<Vec<f32>>()
        });
        let tau = if state2 { 0.005 } else { 0.01 };
        let a_cep = time_to_iir_a(tau, sr as f32 / machine_hop.max(1) as f32);
        iir_smooth(&mut self.cep[..n_half], &fresh_cep, a_cep);

        // --- step 4: f0 = v624/q over the search band, clamp [150, 832];
        //     fallback 500 Hz (byte 7553) ---
        let v624 = 4096.0f32 * sr as f32 / n_fft as f32; // L(7608)·sr/n_fft
        let q_bin = quefrency_argmax(&self.cep[..n_half], {
            // band [v624/7572, v624/7568] = [v624/800, v624/40]
            let lo = rint(v624 / 800.0).max(2) as usize;
            let hi = rint(v624 / 40.0) as usize;
            (lo, hi)
        });
        let f0_raw = if q_bin > 0 { v624 / q_bin as f32 } else { 500.0 };
        let f0 = f0_raw.clamp(150.0, 832.0);
        let q_eff = if q_bin > 0 { q_bin as i64 } else { rint(v624 / 500.0) };

        // --- step 5: taper (0.75, 0.25) at quefrency v651 then zero the
        //     cepstrum cart above it (harmonic comb removal) ---
        let v651 = rint((q_eff - 2) as f32 * pitch_ratio).max(2);
        taper_and_zero(&mut self.cart, (2 * v651) as usize);

        // --- step 6: inverse FFT → harmonic-removed dB envelope (7624) ---
        let env_full = with_plan(n_fft, |p| p.inv(&self.cart));
        self.env.copy_from_slice(&env_full[..n_fft]);
        let env = &self.env[..nb.min(n_fft)];

        // --- step 7: 8 bidirectional passes of pitch-synchronous smoothing
        //     on the raw dB (7672), then blend in the envelope ---
        let a7 = time_to_iir_a(0.12 * f0 * 1.0, n_fft as f32 / sr as f32);
        self.db_work[..nb].copy_from_slice(&db);
        for _ in 0..8 {
            bidir_smooth_pass(&mut self.db_work[..nb], a7);
        }
        let blend_lo = rint(n_fft as f32 * 4.0 / sr as f32).min(4096 - 1).max(0) as usize; // v861 = 4 Hz
        let blend_hi = rint(n_fft as f32 * 9.0 / sr as f32).max(4096) as usize; // v674 = max(bin(9 Hz), L)
        for k in 0..blend_lo.min(nb) {
            self.db_work[k] = env[k];
        }
        if blend_hi > blend_lo && blend_lo < nb {
            let span = (blend_hi - blend_lo) as f32;
            for k in blend_lo..blend_hi.min(nb) {
                let w = (blend_hi - k) as f32 / span;
                self.db_work[k] += (env[k] - self.db_work[k]) * w;
            }
        }

        // --- step 8: gains g[k] = DB2Amp(clamp(db[t]−db[k], −40, +20)),
        //     t = rint(k·pitch_ratio), below-band carries the last delta ---
        let mut gains = gain_field(&self.db_work[..nb], pitch_ratio);

        // --- step 9: energy normalization (byte 7554) ---
        energy_normalize(&mut gains, &score[..nb]);

        // --- step 10: attractor work[k] += (g[k]−work[k])·a (buffer 7864) ---
        let a10 = time_to_iir_a(tau, sr as f32 / machine_hop.max(1) as f32);
        let work = &mut self.work[ch];
        if work.len() < nb {
            work.resize(nb, 0.0);
        }
        for k in 0..nb {
            work[k] += (gains[k] - work[k]) * a10;
        }

        // --- step 11: spec[k] *= work[k] for k ≥ bin(0.8·f0) ---
        let start = rint(f0 * 0.8 * n_fft as f32 / sr as f32)
            .min(nb as i64 - 1)
            .max(0) as usize;
        for k in start..nb {
            target[k] *= work[k];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taper_and_zero_straddles_and_clears_above() {
        let mut cart = vec![1.0f32; 16];
        taper_and_zero(&mut cart, 6);
        // untouched: 0..1
        assert!((cart[0] - 1.0).abs() < 1e-7 && (cart[1] - 1.0).abs() < 1e-7);
        // ×0.75 floats t-4,t-3; ×0.25 floats t-2,t-1
        assert!((cart[2] - 0.75).abs() < 1e-7 && (cart[3] - 0.75).abs() < 1e-7);
        assert!((cart[4] - 0.25).abs() < 1e-7 && (cart[5] - 0.25).abs() < 1e-7);
        // zeroed from t up
        assert!(cart[6..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn bidir_smooth_preserves_constant_and_softens_edges() {
        let a = 0.3;
        let mut c = vec![5.0f32; 32];
        bidir_smooth_pass(&mut c, a);
        assert!(c.iter().all(|&v| (v - 5.0).abs() < 1e-6));
        let mut edge = vec![0.0f32; 32];
        edge[20..].fill(10.0);
        for _ in 0..8 {
            bidir_smooth_pass(&mut edge, a);
        }
        // monotone non-decreasing after smoothing (no overshoot ringing)
        for w in edge.windows(2) {
            assert!(w[0] <= w[1] + 1e-6);
        }
        assert!(edge[0] > 0.0 && edge[31] < 10.0 + 1e-6);
    }

    #[test]
    fn gain_field_clamps_carries_and_scales() {
        // pitch 1: delta = 0 everywhere → unity gains
        let db = (0..64).map(|k| -(k as f32) * 0.1).collect::<Vec<_>>();
        let g = gain_field(&db, 1.0);
        assert!(g.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        // pitch 2, db[k] = −0.1k: delta = db[2k]−db[k] = −0.1k
        // → −0.1·62 = −6.2 dB at the last in-band k (31) — no clamp hit yet
        let g2 = gain_field(&db, 2.0);
        assert!((20.0 * g2[10].log10() + 1.0).abs() < 1e-3);
        // below-band (k ≥ 32): t ≥ 64 out of range → carries the last delta
        let last = 20.0 * g2[31].log10();
        assert!(g2[40..].iter().all(|&v| (20.0 * v.log10() - last).abs() < 1e-5));
        // clamp: a −100 dB slope saturates at −40 dB = 0.01
        let steep = (0..64).map(|k| -(k as f32) * 3.0).collect::<Vec<_>>();
        let gs = gain_field(&steep, 2.0);
        assert!(gs.iter().skip(14).all(|&v| (v - db2amp(-40.0)).abs() < 1e-6));
    }

    #[test]
    fn energy_normalize_compenstates_constant_gain() {
        let mag = [0.5f32; 8];
        let mut g = [2.0f32; 8];
        energy_normalize(&mut g, &mag);
        assert!(g.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn gate_converges_and_preserves_energy_per_channel() {
        // End-to-end: after the attractor converges the gated spectrum keeps
        // the DLL's per-channel state separate and the energy-norm identity
        // holds bin-for-bin on the gated band.
        let mut gate = SpectralGate::new(1024, 2);
        let mut a = vec![0.0f32; 513];
        let mut b = vec![0.0f32; 513];
        for granule in 0..60 {
            for k in 0..513 {
                let f = k as f32 / 1024.0;
                a[k] = (1.0 + 0.4 * (2.0 * std::f32::consts::PI * 17.0 * f).sin()) * 0.5
                    + 0.05 * ((granule * 7 + k * 3) % 11) as f32 / 11.0;
                b[k] = (0.6 + 0.2 * (2.0 * std::f32::consts::PI * 31.0 * f).cos()) * 0.4;
            }
            gate.apply(0, &mut a, 48_000, 1024, 16, false, 1.0);
            gate.apply(1, &mut b, 48_000, 1024, 16, false, 1.0);
        }
        assert!(a.iter().chain(b.iter()).all(|v| v.is_finite() && *v >= 0.0));
        // the two channels must NOT be identical (per-channel work state)
        let diff = a.iter().zip(b.iter()).filter(|(x, y)| (**x - **y).abs() > 1e-6).count();
        assert!(diff > 100, "per-channel attractor state diverged ({diff} bins)");
    }

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

    #[test]
    fn split_scores_raw_and_gates_target() {
        // The DLL topology: gains derive from the RAW analysis copy, then
        // multiply the SYNTHESIS spectrum. A flat score must yield ~unity
        // gains (target survives), while scoring a steeply decaying
        // spectrum under pitch > 1 produces sub-unity gains — so gating the
        // same target with different scores must give different targets.
        let mut gate = SpectralGate::new(1024, 2);
        let flat = vec![1.0f32; 513];
        let decay: Vec<f32> = (0..513).map(|k| 10.0 * (-(k as f32) * 0.08).exp()).collect();
        let pr = 2.0f32.powf(3.0 / 12.0);
        // warm each channel's attractor in its own regime (per-channel work)
        for _ in 0..200 {
            let mut t1 = flat.clone();
            gate.apply_split(0, &flat, &mut t1, 48_000, 1024, 16, false, pr);
            let mut t2 = flat.clone();
            let d2 = decay.clone();
            gate.apply_split(1, &d2, &mut t2, 48_000, 1024, 16, false, pr);
        }
        // final pass on identical targets, different scores/channels
        let mut t1 = flat.clone();
        let mut t2 = flat.clone();
        gate.apply_split(0, &flat, &mut t1, 48_000, 1024, 16, false, pr);
        gate.apply_split(1, &decay, &mut t2, 48_000, 1024, 16, false, pr);
        // flat score → target roughly preserved (unity-ish gains)
        assert!(t1.iter().all(|v| v.is_finite() && *v > 0.0));
        assert!(
            (t1[200] - 1.0).abs() < 0.2,
            "flat score should keep unity-ish gains, got {}",
            t1[200]
        );
        // decaying score → stronger attenuation than flat score somewhere
        let min1 = t1[100..513].iter().cloned().fold(f32::INFINITY, f32::min);
        let min2 = t2[100..513].iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(
            min2 < min1 * 0.999,
            "decaying score must attenuate more (min2={min2} vs min1={min1})"
        );
    }
}
