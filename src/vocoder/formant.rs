//! `formant.c` — `Vocoder::ApplyFormantCorrection`
//! (`pyradius/vocoder_ops.py` 2561-2985: `_fm_h8_opt`, `_bits_nb`,
//! [`FormantState`], [`formant_apply`]).
//!
//! The operator is a self-contained frequency-domain formant correction used by
//! the vocoder's assembly step (`_ev_afc`): dB curve → FFT → persistent envelope
//! IIR → peak search → spectral clip + Hermitian mirror → IFFT kernel → 8x
//! forward/backward dB smoothing → kernel mix-in → `db2amp` gain → RMS
//! normalisation → persistent gain envelope → tail multiply.
//!
//! # Which reference path this is
//!
//! `FormantState.apply` dispatches to `_apply_full` unless route B is on
//! (`_route_b()`), and each stage has a fused `_fm_*_nb` kernel and a NumPy
//! fallback. This port follows the **fused-kernel path**, which is what the
//! reference runs whenever numba is importable (the parity gate's configuration)
//! — the docstrings of those kernels state they are bit-equal to the NumPy
//! fallbacks. The route-B-only `_apply_fast`/`_apply_numba` variants
//! (ctypes/NEON and the `sticky-d` fast path) are not ported; they need the
//! optional `pyradius.neon` extension.
//!
//! # Float discipline
//!
//! Everything stays `f32` except where the reference itself goes through `f64`,
//! and those chains are kept verbatim rather than "simplified":
//!
//! * the dB curve is `f64` `log` times the `f64` literal `8.68588924407959`,
//!   stored once to `f32` (a `logf`-based `f32` chain differs on ~18% of bins);
//! * `_fm_peak_nb` receives Python floats, so `v19 / freq_hi`, the `±0.5`
//!   roundings and `pk = v19 / best` are `f64`;
//! * `_fm_gain2_nb`'s exponent is the three-factor `f64` product
//!   `d * strength * 0.115129254758358`, then `exp`, then one `f32` store;
//! * `_iir16_nb` keeps the recurrence in `f64` and rounds to `f32` on every
//!   element (the C stores `float`, SciPy's `lfilter` accumulates in `double`);
//! * the RMS branch sums `f64` with NumPy's *pairwise* reduction (blocks of
//!   128, eight partial sums, recursive halves), not a linear loop.
//!
//! `_round_i` here is **not** [`super::round_i`] (`(int)rint(x)`): the reference's
//! `_round_i` and the C's `rx_round_i` are both
//! `(int)(x + (x < 0 ? -0.5 : 0.5))`, i.e. round-half-away-from-zero followed by
//! truncation, and they differ on exact halves. See [`round_away_i32`].
//!
//! # Known deviation from `libradius`
//!
//! `_fm_clip_mirror_nb` (and its certified C twin `fm_clip_mirror`) scale half
//! spectrum bins `c4/2 - 2 + j` with `0.75, 0.75, 0.25, 0.25` for `j = 0..4`,
//! then zero from `c4/2` on — so bins `c4/2 - 2` *and* `c4/2 - 1` get `0.75`,
//! while `libradius/src/ops/formant.c` scales `spec[c4-2..c4]` (bin `c4/2 - 1`)
//! by `0.25`. The two agree whenever the kernel mix-in window is empty
//! (`h2 == 0`, i.e. `n_fft * 9 / sr < 0.5`), which is why the corpus never
//! caught it. This port follows the Python reference verbatim; flipping to the
//! C's multipliers is the `j < 2` predicate in [`fm_clip_mirror`].

use crate::consts::PI_F;

/// `_bits_nb(0xC9742400)` — the peak-score gate (`-1000000.0f`).
pub const THR_SCORE_BITS: u32 = 0xC974_2400;
/// `_bits_nb(0x2B8CBDDD)` — the RMS epsilon (`1.000029594723506e-12f`).
pub const RMS_EPS_BITS: u32 = 0x2B8C_BDDD;
/// `9.999999999988105e-21` as the exact `f64` bits
/// (`0x3BC79CA10C922342`; the decimal spelling double-rounds).
const AMP2DB_THRESH_BITS: u64 = 0x3BC7_9CA1_0C92_2342;
/// `np.log(x) * 8.68588924407959` — the `f64` literal from the reference.
const AMP2DB_K: f64 = 8.68588924407959;
/// `0.115129254758358` — the `f64` literal from the reference.
const DB2AMP_K: f64 = 0.115129254758358;
/// `_fm_h8_opt()`: the H8 smoothing variant is opt-in and **not** bit-exact.
pub fn fm_h8_opt() -> bool {
    match std::env::var("PYR_FM_H8") {
        Ok(v) => !(v.is_empty() || v == "0" || v == "false" || v == "False"),
        Err(_) => false,
    }
}

/// `(int)(x + (x < 0 ? -0.5 : 0.5))` — the reference's `_round_i` and the C's
/// `rx_round_i` (truncating, round-half-away-from-zero).
///
/// Not [`super::round_i`], which is `(int)rint(x)` (half-to-even) and is used by
/// the TD/vocoder kernels that call `FRINTI`.
#[inline(always)]
fn round_away_i32(x: f64) -> i32 {
    (x + if x < 0.0 { -0.5 } else { 0.5 }) as i32
}

/// `_round_i(f32)` — the value is widened to `f64` first, as in the reference.
#[inline(always)]
fn round_away_f32(x: f32) -> i32 {
    round_away_i32(x as f64)
}

/// `AudioProcessor::TimeToIirA` / `_t2a_fast` / `time_to_iir_a`: the `f32` chain
/// with `exp` evaluated in `f64` and rounded to `f32` **before** the `1.0 - x`
/// subtraction.
fn time_to_iir_a(tau: f32, rate: f32) -> f32 {
    if tau == 0.0 {
        return 1.0;
    }
    let tr = tau * rate;
    let inv = -1.0f32 / tr;
    let e = (inv as f64).exp() as f32;
    1.0f32 - e
}

/// NumPy's `pairwise_sum` for `f64` (`np.add.reduce` on a contiguous 1-D
/// array): linear below 8, eight partial sums up to 128, then split at
/// `n2 = (n / 2)` rounded down to a multiple of 8. The RMS step's `1.7e-12`
/// relative shift was once enough to move a whole render's correlation, so the
/// reduction shape is part of the contract.
fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0f64;
        for v in a {
            res += *v;
        }
        return res;
    }
    if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let stop = n - (n % 8);
        let mut i = 8;
        while i < stop {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        while i < n {
            r[0] += a[i];
            i += 1;
        }
        return ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
}

/// `rx_formant_cfg` geometry — the block `StartStreaming` fills in
/// (`N = +0x5D0`, `NB = N/2 + 1`, `M = N/4`, `MB = N/8 + 1`) plus the live
/// `f580`.
///
/// `f580` is deliberately *not* cached into the state: the engine recomputes it
/// before every `rx_formant_apply` call (`fc->f580 = prev_granule_1408 > 0 ?
/// prev_granule_1408 : step_base`), it genuinely changes between granules, and
/// pinning it to the 222 default skewed the envelope time constant and every
/// AFC gain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FormantConfig {
    /// `NB = N/2 + 1` — bins the magnitude curve has.
    pub nb_bins: usize,
    /// `N` — the vocoder's FFT size.
    pub n_fft: usize,
    /// `M = N/4` — the formant FFT size.
    pub m_fft: usize,
    /// `MB = N/8 + 1` — half spectrum used by the kernel.
    pub m_bins: usize,
    /// Live `+0x580` (`prev_granule_1408`, else `step_base`); `<= 0` means 222.
    pub f580: i32,
    /// Sample rate as `f32` (the engine keeps it in a float register).
    pub sr: f32,
    /// `2` = 5 ms time constant, anything else = 10 ms.
    pub prec_mode: i32,
    /// Channel count (`gain_env` rows).
    pub nb_bands: usize,
}

/// The reference constructor's `cfg` dict plus geometry defaults: the operator
/// knobs the C initialises in `rx_formant_cfg_defaults` and `StartStreaming`.
///
/// Defaults follow `pyradius/vocoder_core.py` (512-519): `active = 1`,
/// `mode_freq = 0`, `mode_rms = 1`, `ratio = width = strength = 1.0`,
/// `freq_lo = 40`, `freq_hi = 800`, `prec_mode = 2`. `ratio` is rewritten by the
/// engine on every granularity change.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FormantCfg {
    pub nb_bins: usize,
    pub n_fft: usize,
    pub m_fft: usize,
    pub m_bins: usize,
    pub f580: i32,
    pub sr: f32,
    pub prec_mode: i32,
    pub nb_bands: usize,
    /// Gate: only `1` runs the operator.
    pub active: i32,
    /// Pitch ratio (`total_ratio`); `1.0` disables the operator.
    pub ratio: f32,
    /// Correction strength; `0.0` disables the operator.
    pub strength: f32,
    /// Peak search width divisor (C default `1.0f`).
    pub width: f32,
    /// Peak search upper frequency (`800.0f`).
    pub freq_hi: f32,
    /// Peak search lower frequency (`40.0f`).
    pub freq_lo: f32,
    /// `1` = peak search drives the cut point, `0` = fixed 500 Hz.
    pub mode_freq: i32,
    /// `1` = RMS-normalise the gain curve.
    pub mode_rms: i32,
}

impl FormantCfg {
    /// `rx_formant_cfg_defaults` + `StartStreaming` for the given geometry.
    pub fn new(geom: FormantConfig) -> Self {
        Self {
            nb_bins: geom.nb_bins,
            n_fft: geom.n_fft,
            m_fft: geom.m_fft,
            m_bins: geom.m_bins,
            f580: geom.f580,
            sr: geom.sr,
            prec_mode: geom.prec_mode,
            nb_bands: geom.nb_bands,
            active: 1,
            ratio: 1.0,
            strength: 1.0,
            width: 1.0,
            freq_hi: 800.0,
            freq_lo: 40.0,
            mode_freq: 0,
            mode_rms: 1,
        }
    }
}

/// One radix-2 DIT stage's twiddles, `c[k] = cos(ang*k)`, `s[k] = sin(ang*k)`
/// with `ang = ±2pi/len` computed in `f32` and the transcendentals evaluated in
/// `f64` and rounded once (that is what the C's `cosf`/`sinf` calls produce).
struct TwStage {
    ln: usize,
    half: usize,
    c: Vec<f32>,
    s: Vec<f32>,
}

/// `rx_formant_state` + `rx_formant_apply` (`pyradius.vocoder_ops.FormantState`).
///
/// `gain_env` is `[nb_bands][nb_bins + 8]` and is the only per-band state; `env`
/// is a single shared vector of `m_bins + 8` (the Python keeps one shared
/// envelope, unlike the C's per-band `st->env + band * M`).
pub struct FormantState {
    /// Geometry + live knobs (the engine rewrites `cfg.f580`/`cfg.ratio`).
    pub cfg: FormantCfg,
    /// Persistent magnitude envelope, `[m_bins + 8]`.
    pub env: Vec<f32>,
    /// Persistent gain envelope, `[nb_bands][nb_bins + 8]`, initialised to 1.
    pub gain_env: Vec<Vec<f32>>,
    /// dB curve scratch, `[max(nb_bins, m_fft) + 8]`; `db[nb_bins..m_fft]` keeps
    /// the previous call's kernel residue, exactly like the C's persistent
    /// scratch and the Python's `self.db`.
    pub db: Vec<f32>,
    /// Gain curve scratch, `[nb_bins + 8]`.
    pub gscr: Vec<f32>,
    /// Time-domain kernel scratch, `[m_fft + 8]`.
    pub ker: Vec<f32>,
    /// Forward FFT real/imag scratch, `[m_fft]`.
    pub ffr: Vec<f32>,
    /// Forward FFT imaginary scratch, `[m_fft]`.
    pub ffi: Vec<f32>,
    /// `1.0 / m_fft` as `f32` (`A-FFT-3`).
    pub inv_scale: f32,
    /// `float(inv_scale)` — the reference multiplies by the widened value.
    inv_scale_f64: f64,
    /// `float(_bits(0xC9742400))`.
    thr_score: f64,
    /// `float(_bits(0x2B8CBDDD))`.
    rms_eps: f64,
    /// `min(round_i(N*4/sr), M-1)`.
    h1: usize,
    /// `min(round_i(N*9/sr), M)`.
    h2: usize,
    /// Bit-reversal permutation of `m_fft`.
    rev: Vec<usize>,
    /// Twiddle stages, `[0]` forward, `[1]` inverse.
    tw: [Vec<TwStage>; 2],
    /// Scratch for the bit-reversal permutation (`ffr[rev]`).
    perm: Vec<f32>,
    /// Scratch for the dB smoothing (`y = db[:NB].astype(np.float64)`).
    iir_buf: Vec<f64>,
    /// Scratch for the RMS branch (`2 * nb_bins` `f64`).
    rms_buf: Vec<f64>,
}

impl FormantState {
    /// `FormantState.__init__` — allocate the persistent state and build the FFT
    /// twiddle tables (the engine does this in `rx_formant_state_init`, so the
    /// per-granule path makes no `sin`/`cos` calls).
    pub fn new(geom: FormantConfig) -> Self {
        let cfg = FormantCfg::new(geom);
        let m = cfg.m_fft;
        let nb = cfg.nb_bins;
        let mb = cfg.m_bins;
        let dbn = nb.max(m);
        let inv_scale = 1.0f32 / m as f32;
        // `_round_i(F32(F32(n_fft) * F32(4.0) / F32(sr)))`, clamped by the
        // constructor (`min(h1, m_fft - 1)` / `min(h2, m_fft)`).
        let h1f = cfg.n_fft as f32 * 4.0f32 / cfg.sr;
        let h1 = (round_away_f32(h1f).max(0) as usize).min(m.saturating_sub(1));
        let h2f = cfg.n_fft as f32 * 9.0f32 / cfg.sr;
        let h2 = (round_away_f32(h2f).max(0) as usize).min(m);
        let mut st = Self {
            cfg,
            env: vec![0.0f32; mb + 8],
            gain_env: vec![vec![1.0f32; nb + 8]; cfg.nb_bands],
            db: vec![0.0f32; dbn + 8],
            gscr: vec![0.0f32; nb + 8],
            ker: vec![0.0f32; m + 8],
            ffr: vec![0.0f32; m],
            ffi: vec![0.0f32; m],
            inv_scale,
            inv_scale_f64: inv_scale as f64,
            thr_score: f64::from(f32::from_bits(THR_SCORE_BITS)),
            rms_eps: f64::from(f32::from_bits(RMS_EPS_BITS)),
            h1,
            h2,
            rev: Vec::new(),
            tw: [Vec::new(), Vec::new()],
            perm: vec![0.0f32; m],
            iir_buf: vec![0.0f64; nb],
            rms_buf: vec![0.0f64; 2 * nb],
        };
        st.build_tw();
        st
    }

    /// `_build_tw` — bit-reversal map and per-stage twiddles for both directions.
    fn build_tw(&mut self) {
        let n = self.cfg.m_fft;
        assert!(n > 0, "formant: m_fft must be positive");
        let bits = (usize::BITS - n.leading_zeros()) as usize - 1;
        let mut rev = vec![0usize; n];
        for (i, r) in rev.iter_mut().enumerate() {
            let mut v = 0usize;
            for b in 0..bits {
                v |= ((i >> b) & 1) << (bits - 1 - b);
            }
            *r = v;
        }
        self.rev = rev;
        let mut tw: [Vec<TwStage>; 2] = [Vec::new(), Vec::new()];
        for (inverse, stages) in tw.iter_mut().enumerate() {
            let mut ln = 2usize;
            while ln <= n {
                let half = ln >> 1;
                // `F32((F32(2.0) if inverse else F32(-2.0)) * F32(pi) / F32(ln))`
                let ang = if inverse == 1 { 2.0f32 } else { -2.0f32 } * PI_F / ln as f32;
                let mut c = Vec::with_capacity(half);
                let mut s = Vec::with_capacity(half);
                for k in 0..half {
                    let arg = ang * k as f32;
                    c.push((arg as f64).cos() as f32);
                    s.push((arg as f64).sin() as f32);
                }
                stages.push(TwStage { ln, half, c, s });
                ln <<= 1;
            }
        }
        self.tw = tw;
    }

    /// `_f580()` — the live `+0x580`, re-read on every call (`<= 0` means 222).
    #[inline]
    pub fn f580(&self) -> i32 {
        if self.cfg.f580 > 0 {
            self.cfg.f580
        } else {
            222
        }
    }

    /// `FormantState.apply` — `rx_formant_apply`, in place.
    ///
    /// # Panics
    /// If `mag` is shorter than `cfg.nb_bins` or `band >= cfg.nb_bands`
    /// (`IndexError` in the reference).
    pub fn apply(&mut self, mag: &mut [f32], band: usize) {
        let c = self.cfg;
        if c.active != 1 {
            return;
        }
        let (ratio, strength) = (c.ratio, c.strength);
        if ratio == 1.0 || strength == 0.0 {
            return;
        }
        let (nb, m, mb, n) = (c.nb_bins, c.m_fft, c.m_bins, c.n_fft);
        if nb < 1 || mb < 1 {
            return;
        }
        assert!(mag.len() >= nb, "formant: mag shorter than nb_bins");
        assert!(band < c.nb_bands, "formant: band out of range");

        // ---- 1 dB curve + 2 FFT prep (fused `_fm_prep_nb`) ----
        self.fm_prep(mag);
        self.fft(0);

        // ---- 3 envelope IIR (`_fm_env2_nb`) ----
        let tconst = if c.prec_mode == 2 { 0.005f32 } else { 0.01f32 };
        let a1 = time_to_iir_a(tconst, c.sr / self.f580() as f32);
        fm_env2(&self.ffr, &self.ffi, &mut self.env, mb, a1 as f64);

        // ---- 4+5 peak search + cut (`_fm_peak_nb`) ----
        // `v19 = F32(F32(sr) * F32(M) / F32(N))` is an f32 chain, widened once;
        // the kernel then treats it as a Python float, so its divisions and
        // roundings are f64.
        let v19 = (c.sr * m as f32 / n as f32) as f64;
        let (best, fc, peak) = fm_peak(
            &self.env,
            mb,
            v19,
            c.freq_hi as f64,
            c.freq_lo as f64,
            c.mode_freq,
            self.thr_score,
        );
        let _ = best;
        let r_ = if ratio < 1.0 { ratio } else { 1.0 };
        let q = (peak - 2) as f32 / c.width;
        let q = q * r_;
        let mut cut = round_away_i32(q as f64);
        if cut < 2 {
            cut = 2;
        }
        let c4 = (2 * cut) as usize;

        // ---- 5-tail + 6 clip, mirror, IFFT, kernel scale ----
        fm_clip_mirror(&mut self.ffr, &mut self.ffi, mb, m, c4);
        self.fft(1);
        fm_ker_scale(&self.ffr, &mut self.ker, m, self.inv_scale_f64);

        // ---- 7 dB curve: 16 single-pole passes over 8 forward+backward sweeps ----
        let w012 = c.width * 0.12f32;
        let tau2 = (fc * w012 as f64) as f32;
        let rate2 = n as f32 / c.sr;
        let a2 = time_to_iir_a(tau2, rate2);
        iir16_scipy(&mut self.db[..nb], &mut self.iir_buf, a2, 1.0f32 - a2);

        // ---- 8 kernel mix-in (`_fm_kermix_nb`) ----
        fm_kermix(&mut self.db, &self.ker, self.h1, self.h2);

        // ---- 9 gain (`_fm_gain2_nb`) ----
        fm_gain2(&self.db, &mut self.gscr, nb, ratio as f64, strength as f64);

        // ---- 10 RMS normalisation ----
        if c.mode_rms != 0 {
            fm_rms(
                &mag[..nb],
                &mut self.gscr,
                nb,
                self.rms_eps,
                &mut self.rms_buf,
            );
        }

        // ---- 11+12 gain envelope + tail multiply (`_fm_ges_tail_nb`) ----
        // `a3` is the same value as `a1` (the reference recomputes it).
        let a3 = a1;
        fm_ges_tail(
            &self.gscr,
            &mut self.gain_env[band],
            mag,
            nb,
            a3 as f64,
            fc,
            n,
            c.sr as f64,
        );
    }

    /// `_fm_prep_nb` — steps 1+2 fused: dB curve into `db` *and* `ffr`, `ffi`
    /// zeroed. `db[nb_bins..m_fft]` keeps the previous call's kernel residue,
    /// which is what the C's persistent scratch holds.
    fn fm_prep(&mut self, mag: &[f32]) {
        let nb = self.cfg.nb_bins;
        let m = self.cfg.m_fft;
        let thr = f64::from_bits(AMP2DB_THRESH_BITS);
        for i in 0..nb {
            let mv = mag[i] as f64;
            if mv >= thr {
                let a = if mv > 1e-300 { mv } else { 1e-300 };
                self.db[i] = (a.ln() * AMP2DB_K) as f32;
            } else {
                self.db[i] = -391.0;
            }
        }
        for i in 0..m {
            self.ffr[i] = self.db[i];
            self.ffi[i] = 0.0;
        }
    }

    /// `FormantState._fft` — complex radix-2 DIT, `inverse = 0/1`.
    fn fft(&mut self, inverse: usize) {
        let m = self.cfg.m_fft;
        let Self {
            ffr,
            ffi,
            perm,
            rev,
            tw,
            ..
        } = self;
        fft_c(ffr, ffi, perm, rev, &tw[inverse], m);
    }

    /// `_amp2db` (scalar): one `f64` `log`, one `f64` multiply, one `f32` store.
    pub fn amp2db(x: f32) -> f32 {
        let thr = f64::from_bits(AMP2DB_THRESH_BITS);
        if x as f64 >= thr {
            ((x as f64).ln() * AMP2DB_K) as f32
        } else {
            -391.0
        }
    }

    /// `_db2amp` (scalar): `expf(x * 0.115129254758358f)`.
    pub fn db2amp(x: f32) -> f32 {
        (x as f64 * DB2AMP_K).exp() as f32
    }

    /// `_db2amp_arr` — the vectorised form (same libm `exp`, same `f32` round).
    pub fn db2amp_arr(x: &[f32]) -> Vec<f32> {
        x.iter()
            .map(|v| ((*v as f64) * DB2AMP_K).exp() as f32)
            .collect()
    }

    /// `_round_away_arr` — C `(int)(v + (v < 0 ? -0.5 : 0.5))` as `f64`
    /// (`np.trunc`, half away from zero), the non-fused fallback's `k2` map.
    pub fn round_away_arr(v: &[f32]) -> Vec<f64> {
        v.iter()
            .map(|x| {
                let x = *x as f64;
                let s = if x.is_sign_negative() { -0.5 } else { 0.5 };
                (x + s).trunc()
            })
            .collect()
    }
}

/// `formant_apply(state, mag, band)` — convenience wrapper, in place (the
/// reference returns the same buffer it was handed after copying it).
pub fn formant_apply(state: &mut FormantState, mag: &mut [f32], band: usize) {
    state.apply(mag, band);
}

/// `_fm_env2_nb` — step 3 reading `ffr`/`ffi` directly: `e = re*re + im*im` stays
/// `f32`, and `a1 * (e - env) + env` is a two-operation `f64` chain.
fn fm_env2(ffr: &[f32], ffi: &[f32], env: &mut [f32], mb: usize, a1: f64) {
    for k in 0..mb {
        let re = ffr[k];
        let im = ffi[k];
        let e = re * re + im * im;
        env[k] = (a1 * ((e - env[k]) as f64) + env[k] as f64) as f32;
    }
}

/// `_fm_peak_nb` — steps 4+5: peak search + peak frequency / clip point.
///
/// `v19`, `freq_hi`, `freq_lo` and `thr_score` are Python floats in the kernel,
/// so the divisions, the `+0.5` roundings and `pk = v19 / best` are `f64`; the
/// weights and the score are `f32` (`np.float32`-expressed constants).
///
/// Returns `(best, fC, peak)` with `fC` already clamped to `[150, 800]`.
fn fm_peak(
    env: &[f32],
    mb: usize,
    v19: f64,
    freq_hi: f64,
    freq_lo: f64,
    mode_freq: i32,
    thr_score: f64,
) -> (i32, f64, i32) {
    let hi = v19 / freq_hi;
    let mut hi2 = round_away_i32(hi);
    if hi2 > mb as i32 - 2 {
        hi2 = mb as i32 - 2;
    }
    if hi2 < 2 {
        hi2 = 2;
    }
    let lo = v19 / freq_lo;
    let mut lo2 = round_away_i32(lo);
    if lo2 > mb as i32 - 2 {
        lo2 = mb as i32 - 2;
    }
    let mut best = 0i32;
    if lo2 > hi2 {
        let n_sc = (lo2 - hi2) as usize;
        let mut bmax = 0.0f32;
        let mut bi = 0i32;
        for j in 0..n_sc {
            let k = hi2 as usize + j;
            let v = env[k];
            let den1 = (env[k - 1] + env[k + 1]) + super::EPS1E6_F;
            let r1 = v / den1;
            let w1 = if r1 <= 0.5f32 {
                0.2f32
            } else if r1 < 2.0f32 {
                (r1 - 0.5f32) / 1.5f32 + 0.2f32
            } else {
                1.2f32
            };
            let r2 = v / (env[k >> 1] + super::EPS1E6_F);
            let w2 = if r2 > 0.3f32 {
                if r2 < 5.0f32 {
                    (r2 - 0.3f32) / 4.7f32
                } else {
                    1.0f32
                }
            } else {
                0.0f32
            };
            let sc = (v * w1) * w2;
            if j == 0 {
                bmax = sc;
                bi = 0;
            } else if sc > bmax {
                bmax = sc;
                bi = j as i32;
            }
        }
        if bmax as f64 > thr_score {
            best = hi2 + bi;
        }
    }
    let (pk, peak) = if mode_freq != 0 {
        let pk = if best != 0 { v19 / best as f64 } else { 0.0f64 };
        (pk, best)
    } else {
        let pk = 500.0f32;
        let q = v19 / pk as f64;
        (pk as f64, round_away_i32(q))
    };
    let mut fc = if pk <= 800.0f32 as f64 {
        pk
    } else {
        800.0f32 as f64
    };
    fc = if fc >= 150.0f32 as f64 {
        fc
    } else {
        150.0f32 as f64
    };
    (best, fc, peak)
}

/// `_fm_clip_mirror_nb` — steps 5-tail + 6-prep on `ffr`/`ffi` directly: the
/// `0.75/0.75/0.25/0.25` clip, the zero tail and the Hermitian mirror.
///
/// See the module docs: the `j < 2` predicate is the reference's behaviour, which
/// scales bins `c4/2 - 2` and `c4/2 - 1` both by `0.75`; `libradius` scales
/// `spec[c4-2..c4]` (bin `c4/2 - 1`) by `0.25`.
fn fm_clip_mirror(ffr: &mut [f32], ffi: &mut [f32], mb: usize, m: usize, c4: usize) {
    let half_c4 = (c4 >> 1) as i64;
    for j in 0..4i64 {
        let k = half_c4 - 2 + j;
        let f = if j < 2 { 0.75f32 } else { 0.25f32 };
        if k >= 0 && (k as usize) < mb {
            let k = k as usize;
            ffr[k] *= f;
            ffi[k] *= f;
        }
    }
    for k in (c4 >> 1)..mb {
        ffr[k] = 0.0;
        ffi[k] = 0.0;
    }
    ffi[0] = 0.0;
    ffi[mb - 1] = 0.0;
    for k in 1..mb - 1 {
        ffr[m - k] = ffr[k];
        ffi[m - k] = -ffi[k];
    }
}

/// `_fm_ker_scale_nb` — step 6-post: `ker = ffr * inv_scale`, with `inv_scale` a
/// Python float, so the product is `f64` and rounded once.
fn fm_ker_scale(ffr: &[f32], ker: &mut [f32], m: usize, inv_scale: f64) {
    for i in 0..m {
        ker[i] = (ffr[i] as f64 * inv_scale) as f32;
    }
}

/// `_fm_kermix_nb` — step 8: `ker[0..h1)` is copied into `db`, then
/// `k in [h1, h2)` blends with `w = (h2 - k) / span` (single-rounding fma).
fn fm_kermix(db: &mut [f32], ker: &[f32], h1: usize, h2: usize) {
    if h1 > 0 {
        db[..h1].copy_from_slice(&ker[..h1]);
    }
    if h2 > h1 {
        let span = (h2 - h1) as f32;
        for k in h1..h2 {
            let w = (h2 - k) as f32 / span;
            db[k] = (w as f64 * ((ker[k] - db[k]) as f64) + db[k] as f64) as f32;
        }
    }
}

/// `_fm_gain2_nb` — step 9: sticky-`d` gain map, `exp` in `f64` over the
/// three-factor product, one `f32` store.
fn fm_gain2(db: &[f32], gscr: &mut [f32], nb: usize, ratio: f64, strength: f64) {
    let mut last_d = 0.0f32;
    for v in 0..nb {
        // `k2f = np.float32(ratio) * np.float32(v)` — an f32 product; the ±0.5
        // step is then a plain f64 `int(...)` truncation, *not* another
        // round-away (`_fm_gain2_nb` spells out `int(k2f - 0.5)` /
        // `int(k2f + 0.5)`).
        let k2f = (ratio as f32) * (v as f32);
        let k2 = if k2f < 0.0 {
            (k2f as f64 - 0.5) as i32
        } else {
            (k2f as f64 + 0.5) as i32
        };
        let d = if k2 >= 0 && (k2 as usize) < nb {
            let d = db[k2 as usize] - db[v];
            last_d = d;
            d
        } else {
            last_d
        };
        let d = if d > 20.0f32 {
            20.0f32
        } else if d < -40.0f32 {
            -40.0f32
        } else {
            d
        };
        gscr[v] = (d as f64 * strength * DB2AMP_K).exp() as f32;
    }
}

/// Step 10 — RMS normalisation.
///
/// The reference (`_apply_full`, lines 2952-2958) converts to `f64` *before*
/// squaring for `den`, but squares `gscr` and `mag` in `f32` for `num`:
///
/// ```text
///   den = sum( (f64)mag ** 2 )
///   num = sum( (f32)(gg*gg) * (f32)(mm*mm) )        # both widened to f64
///   ss  = f32(sqrt(den / (num + eps)))
///   gscr = ss * gscr                                # f32
/// ```
fn fm_rms(mag: &[f32], gscr: &mut [f32], nb: usize, rms_eps: f64, scr: &mut [f64]) {
    {
        let (mm, gg) = scr.split_at_mut(nb);
        for i in 0..nb {
            let m = mag[i] as f64;
            mm[i] = m * m;
            gg[i] = (gscr[i] * gscr[i]) as f64 * (mag[i] * mag[i]) as f64;
        }
    }
    let (mm, gg) = scr.split_at(nb);
    let den = pairwise_sum(&mm[..nb]);
    let num = pairwise_sum(&gg[..nb]);
    let ss = (den / (num + rms_eps)).sqrt() as f32;
    for v in gscr[..nb].iter_mut() {
        *v = ss * *v;
    }
}

/// `_fm_ges_tail_nb` — steps 11+12: gain envelope IIR (persistent, per band) and
/// the tail multiply.
#[allow(clippy::too_many_arguments)]
fn fm_ges_tail(
    gscr: &[f32],
    ges: &mut [f32],
    mag: &mut [f32],
    nb: usize,
    a3: f64,
    fc: f64,
    n: usize,
    sr: f64,
) {
    for v in 0..nb {
        ges[v] = (a3 * ((gscr[v] - ges[v]) as f64) + ges[v] as f64) as f32;
    }
    // `np.float32(np.float32(fC * np.float32(0.8)) * np.float32(N) / np.float32(sr))`
    let t1 = (fc * 0.8f32 as f64) as f32;
    let vt = t1 * (n as f32) / (sr as f32);
    let mut tail = round_away_f32(vt);
    if tail > nb as i32 - 1 {
        tail = nb as i32 - 1;
    }
    // The Python slices `mag[tail:NB]`, so a negative tail would wrap; the
    // engine's `fC >= 150` clamp makes that unreachable, so clamp to 0.
    for v in tail.max(0) as usize..nb {
        mag[v] = ges[v] * mag[v];
    }
}

/// Step 7 of the reference's default path: eight forward+backward first-order
/// sweeps, each one SciPy's `lfilter` in `f64` with `zi = (1 - a2) * y[0]` and
/// the whole array rounded to `f32` after every *pair* of sweeps.
///
/// `lfilter(b, a, x, zi)` is the transposed direct form II: `y[i] = b0*x[i] +
/// z`, `z = b1*x[i] - a1*y[i]` with `b1 = 0` and `a1 = -a1m`, so the state
/// update is the `f64` product `a1m * y[i]`. `buf` is the `f64` scratch the
/// reference allocates (`y = db[:NB].astype(np.float64)`).
///
/// Note this is **not** `pyradius.vocoder_ops._iir16_nb`: that numba kernel
/// keeps `y` in `f32` and rounds on every element instead of once per sweep, and
/// it differs from the SciPy chain by up to ~1e-6 per element (measured), even
/// though its docstring claims to mirror it. `_apply_full` only reaches it under
/// route B, so the default path this port follows is the SciPy chain below.
fn iir16_scipy(y: &mut [f32], buf: &mut [f64], a2: f32, a1m: f32) {
    let n = y.len();
    if n == 0 {
        return;
    }
    let b0 = a2 as f64;
    let om = 1.0f64 - a2 as f64;
    let am = a1m as f64;
    for (slot, v) in buf[..n].iter_mut().zip(y.iter()) {
        *slot = *v as f64;
    }
    for _ in 0..8 {
        let mut z = om * buf[0];
        for slot in buf[..n].iter_mut() {
            let v = b0 * *slot + z;
            z = am * v;
            *slot = v;
        }
        let mut z = om * buf[n - 1];
        for i in (0..n).rev() {
            let v = b0 * buf[i] + z;
            z = am * v;
            buf[i] = v;
        }
        // `y = y.astype(np.float32).astype(np.float64)`
        for slot in buf[..n].iter_mut() {
            *slot = (*slot as f32) as f64;
        }
    }
    for (slot, v) in y.iter_mut().zip(buf[..n].iter()) {
        *slot = *v as f32;
    }
}

/// `FormantState._fft` body: bit-reversal permutation then radix-2 DIT stages.
fn fft_c(
    ffr: &mut [f32],
    ffi: &mut [f32],
    perm: &mut [f32],
    rev: &[usize],
    stages: &[TwStage],
    m: usize,
) {
    for (i, r) in rev.iter().enumerate().take(m) {
        perm[i] = ffr[*r];
    }
    ffr[..m].copy_from_slice(&perm[..m]);
    for (i, r) in rev.iter().enumerate().take(m) {
        perm[i] = ffi[*r];
    }
    ffi[..m].copy_from_slice(&perm[..m]);
    for st in stages {
        let (ln, half) = (st.ln, st.half);
        let mut base = 0usize;
        while base < m {
            for k in 0..half {
                let c = st.c[k];
                let s = st.s[k];
                let xr = ffr[base + k + half];
                let xi = ffi[base + k + half];
                let vr = xr * c - xi * s;
                let vi = xr * s + xi * c;
                let ur = ffr[base + k];
                let ui = ffi[base + k];
                ffr[base + k] = ur + vr;
                ffi[base + k] = ui + vi;
                ffr[base + k + half] = ur - vr;
                ffi[base + k + half] = ui - vi;
            }
            base += ln;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Engine geometry for the unit tests: `N = 256` -> `NB = 129`, `M = 64`,
    /// `MB = 33` (`NB = N/2+1`, `M = N/4`, `MB = N/8+1`).
    fn small_geom(sr: f32, f580: i32) -> FormantConfig {
        FormantConfig {
            nb_bins: 129,
            n_fft: 256,
            m_fft: 64,
            m_bins: 33,
            f580,
            sr,
            prec_mode: 2,
            nb_bands: 4,
        }
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn from_bits(v: &[u32]) -> Vec<f32> {
        v.iter().map(|x| f32::from_bits(*x)).collect()
    }

    // ---- golden data (minted from `pyradius.vocoder_ops` by .tmp tooling) ----
    const SMALL_MAG: [u32; 129] = [
        0x3D53EA9B, 0x3DBEDD54, 0x3E0AF416, 0x3DA384C6, 0x3E01850B, 0x3D9D3267, 0x3E074389,
        0x3E46D870, 0x3E28D0E0, 0x3E7BC7E2, 0x3E7460BB, 0x3EB00DFB, 0x3EEC142F, 0x3EFA21C4,
        0x3F1F701E, 0x3F28B24A, 0x3F4A79C8, 0x3F69747A, 0x3F6A5C5B, 0x3F7EFCEC, 0x3F733333,
        0x3F79DE34, 0x3F79B884, 0x3F5A1851, 0x3F4F9880, 0x3F28B24A, 0x3F1A5166, 0x3F0C6D0B,
        0x3ECD5BDD, 0x3EBA4B6B, 0x3E7460BB, 0x3E674D00, 0x3E664184, 0x3E0967CC, 0x3E1BBE6A,
        0x3D9D3267, 0x3DDA1453, 0x3E0F3307, 0x3D9B06E5, 0x3DE7D316, 0x3D53EA9B, 0x3DBA31EF,
        0x3E05994E, 0x3D8FD587, 0x3DE181E9, 0x3D4D02F6, 0x3DB85E20, 0x3E052163, 0x3D8F5E6B,
        0x3DE1489B, 0x3D4CCD8A, 0x3DB85210, 0x3E051EBF, 0x3D8F5C2E, 0x3DE147B0, 0x3D4CCCCE,
        0x3DB851EC, 0x3E051EB8, 0x3D8F5C29, 0x3DE147AE, 0x3D4CCCCD, 0x3DB851EC, 0x3E051EB8,
        0x3D8F5C29, 0x3DE147AE, 0x3D4CCCCD, 0x3DB851EC, 0x3E051EB8, 0x3D8F5C29, 0x3DE147AE,
        0x3D4CCCCD, 0x3DB851EC, 0x3E051EB8, 0x3D8F5C29, 0x3DE147AE, 0x3D4CCCCD, 0x3DB851EC,
        0x3E051EB8, 0x3D8F5C29, 0x3DE147AE, 0x3D4CCCCD, 0x3DB851EC, 0x3E051EB8, 0x3D8F5C29,
        0x3DE147AE, 0x3D4CCCCD, 0x3DB851EC, 0x3E051EB8, 0x3D8F5C29, 0x3DE147AE, 0x3D4CCCCD,
        0x3DB851EC, 0x3E051EB8, 0x3D8F5C29, 0x3DE147AE, 0x3D4CCCCD, 0x3DB851EC, 0x3E051EB8,
        0x3D8F5C29, 0x3DE147AE, 0x3D4CCCCD, 0x3DB851EC, 0x3E051EB8, 0x3D8F5C29, 0x3DE147AE,
        0x3D4CCCCD, 0x3DB851EC, 0x3E051EB8, 0x3D8F5C29, 0x3DE147AE, 0x3D4CCCCD, 0x3DB851EC,
        0x3E051EB8, 0x3D8F5C29, 0x3DE147AE, 0x3D4CCCCD, 0x3DB851EC, 0x3E051EB8, 0x3D8F5C29,
        0x3DE147AE, 0x3D4CCCCD, 0x3DB851EC, 0x3E051EB8, 0x3D8F5C29, 0x3DE147AE, 0x3D4CCCCD,
        0x3DB851EC, 0x3E051EB8, 0x3D8F5C29,
    ];

    const SMALL_P0: [u32; 129] = [
        0x3D53EA9B, 0x3DBEDD54, 0x3E097BEA, 0x3DB5360A, 0x3E21F1DD, 0x3E0C5BE2, 0x3E61B8E9,
        0x3EB21F57, 0x3EB3F6BE, 0x3F0F36B0, 0x3F0AF0DE, 0x3F472A1D, 0x3F6099FE, 0x3F5CDE26,
        0x3F769A29, 0x3F5D6F09, 0x3F6215CA, 0x3F4DED34, 0x3F3C66DB, 0x3F25E27E, 0x3F0EEB2D,
        0x3F08F3ED, 0x3EFE3EF3, 0x3ECEDC4B, 0x3ECC87FB, 0x3EA5DC20, 0x3E9CBFA4, 0x3E90FDEB,
        0x3E6F840F, 0x3E596102, 0x3E0D2FC5, 0x3E29DDF8, 0x3E1E3CC5, 0x3DC29782, 0x3E062D43,
        0x3DA291A4, 0x3DDB3287, 0x3E03E942, 0x3DC2F5CD, 0x3DFBA9F9, 0x3D675C2C, 0x3DECBB5F,
        0x3DFD61F2, 0x3D89131F, 0x3DF52A28, 0x3D86BF3F, 0x3DD1C9FE, 0x3E01CB19, 0x3DBD690E,
        0x3DFAAD53, 0x3D63E3B7, 0x3DEC8774, 0x3DFD4418, 0x3D88E054, 0x3DF52522, 0x3D86B253,
        0x3DD1C94F, 0x3E01CB0F, 0x3DBD683B, 0x3DFAAD57, 0x3D63E367, 0x3DEC877B, 0x3DFD441B,
        0x3D88E053, 0x3DF52522, 0x3D86B254, 0x3DD1C952, 0x3E01CB0F, 0x3DBD683B, 0x3DFAAD57,
        0x3D63E367, 0x3DEC877B, 0x3DFD441B, 0x3D88E053, 0x3DF52522, 0x3D86B254, 0x3DD1C952,
        0x3E01CB0F, 0x3DBD683B, 0x3DFAAD57, 0x3D63E367, 0x3DEC877B, 0x3DFD4414, 0x3D88DE5B,
        0x3DF4FD32, 0x3D821428, 0x3DEA2449, 0x3E291A34, 0x3DB61C38, 0x3E0F162C, 0x3D821428,
        0x3DEA2449, 0x3E291A34, 0x3DB61C38, 0x3E0F162C, 0x3D821428, 0x3DEA2449, 0x3E291A34,
        0x3DB61C38, 0x3E0F162C, 0x3D821428, 0x3DEA2449, 0x3E291A34, 0x3DB61C38, 0x3E0F162C,
        0x3D821428, 0x3DEA2449, 0x3E291A34, 0x3DB61C38, 0x3E0F162C, 0x3D821428, 0x3DEA2449,
        0x3E291A34, 0x3DB61C38, 0x3E0F162C, 0x3D821428, 0x3DEA2449, 0x3E291A34, 0x3DB61C38,
        0x3E0F162C, 0x3D821428, 0x3DEA2449, 0x3E291A34, 0x3DB61C38, 0x3E0F162C, 0x3D821428,
        0x3DEA2449, 0x3E291A34, 0x3DB61C38,
    ];

    const SMALL_P1: [u32; 129] = [
        0x3D53EA9B, 0x3DBEDD54, 0x3E08E6C0, 0x3DBC3A08, 0x3E2ECD67, 0x3E24D950, 0x3E82CBC5,
        0x3ED1544E, 0x3ED9DAD1, 0x3F2F0AE7, 0x3F29CF36, 0x3F733BE1, 0x3F856DAA, 0x3F816D98,
        0x3F8C9531, 0x3F725881, 0x3F6B726D, 0x3F4302BD, 0x3F2A2D78, 0x3F028D71, 0x3ECE4ED3,
        0x3EB85B5B, 0x3E9D04A3, 0x3E67D92F, 0x3E71FEDA, 0x3E43AEED, 0x3E410A7F, 0x3E363CFB,
        0x3E2BA102, 0x3E1BD54C, 0x3DC8893C, 0x3E1181B5, 0x3E01AE04, 0x3DA2C84E, 0x3DFB3FE9,
        0x3DA4B2F7, 0x3DDBA403, 0x3DFEDEC0, 0x3DD2CB82, 0x3E01C3F4, 0x3D6F11EF, 0x3E0062BC,
        0x3DF7E78C, 0x3D8664F5, 0x3DFCF59B, 0x3D93881F, 0x3DDBDE91, 0x3E007859, 0x3DCFAAD4,
        0x3E025F88, 0x3D6D0B49, 0x3E009DA0, 0x3DF81F0E, 0x3D864E25, 0x3DFD05AC, 0x3D9380D1,
        0x3DDBE274, 0x3E00795A, 0x3DCFAA92, 0x3E025FBA, 0x3D6D0B23, 0x3E009DAC, 0x3DF81F17,
        0x3D864E25, 0x3DFD05AC, 0x3D9380D2, 0x3DDBE278, 0x3E00795A, 0x3DCFAA92, 0x3E025FBA,
        0x3D6D0B23, 0x3E009DAC, 0x3DF81F17, 0x3D864E25, 0x3DFD05AC, 0x3D9380D2, 0x3DDBE278,
        0x3E00795A, 0x3DCFAA92, 0x3E025FBA, 0x3D6D0B23, 0x3E009DAC, 0x3DF81F0E, 0x3D864B66,
        0x3DFCCDE5, 0x3D8D0DE1, 0x3DFDE5C9, 0x3E375ED7, 0x3DC579D4, 0x3E1B28DE, 0x3D8D0DE1,
        0x3DFDE5C9, 0x3E375ED7, 0x3DC579D4, 0x3E1B28DE, 0x3D8D0DE1, 0x3DFDE5C9, 0x3E375ED7,
        0x3DC579D4, 0x3E1B28DE, 0x3D8D0DE1, 0x3DFDE5C9, 0x3E375ED7, 0x3DC579D4, 0x3E1B28DE,
        0x3D8D0DE1, 0x3DFDE5C9, 0x3E375ED7, 0x3DC579D4, 0x3E1B28DE, 0x3D8D0DE1, 0x3DFDE5C9,
        0x3E375ED7, 0x3DC579D4, 0x3E1B28DE, 0x3D8D0DE1, 0x3DFDE5C9, 0x3E375ED7, 0x3DC579D4,
        0x3E1B28DE, 0x3D8D0DE1, 0x3DFDE5C9, 0x3E375ED7, 0x3DC579D4, 0x3E1B28DE, 0x3D8D0DE1,
        0x3DFDE5C9, 0x3E375ED7, 0x3DC579D4,
    ];

    const SMALL2_P0: [u32; 129] = [
        0x3D53EA9B, 0x3DBEDD54, 0x3DFCED14, 0x3D9D7CC1, 0x3DE076D4, 0x3D939324, 0x3DCEDD76,
        0x3E00F49F, 0x3DF019BE, 0x3E3A6346, 0x3E2F868D, 0x3E58B132, 0x3E8DBD29, 0x3E9581FE,
        0x3EC82DB9, 0x3EC50BB3, 0x3EFA4AA8, 0x3F140A4C, 0x3F1F2995, 0x3F28C3B5, 0x3F2E2206,
        0x3F44457D, 0x3F586165, 0x3F48CE43, 0x3F55E4B6, 0x3F48168C, 0x3F49EBB8, 0x3F4ED98C,
        0x3F357B19, 0x3F3F8891, 0x3F11DF33, 0x3F1D41B5, 0x3F2263BE, 0x3ED81DE6, 0x3EFE067E,
        0x3EA28203, 0x3EBC2D6B, 0x3EC100A3, 0x3E4D2914, 0x3E9F14D4, 0x3E0A2023, 0x3E371B48,
        0x3E4EE7DF, 0x3DFC31F3, 0x3E2086E1, 0x3D9485F9, 0x3DBC9453, 0x3DF3E2AB, 0x3D9765BB,
        0x3DF9CB83, 0x3D634E1A, 0x3DB1A993, 0x3DDFED68, 0x3D687151, 0x3DCFCB39, 0x3D586FB8,
        0x3DBC3C08, 0x3DDB808B, 0x3D81AB64, 0x3DCD4767, 0x3D3A8554, 0x3DA7CDE7, 0x3DF25655,
        0x3D90FE8F, 0x3DCE9101, 0x3D556BD7, 0x3D932447, 0x3DC0631C, 0x3D81DC5A, 0x3DE594F5,
        0x3D571558, 0x3DA88BA1, 0x3DD97067, 0x3D63DE38, 0x3DCD94EA, 0x3D5604F3, 0x3DBB507F,
        0x3DDAFC02, 0x3D818336, 0x3DCD079E, 0x3D3A6404, 0x3DA7C06A, 0x3DF24ED1, 0x3D90F9E3,
        0x3DCE8E30, 0x3D556A8F, 0x3D9323E3, 0x3DC062A0, 0x3D81DC35, 0x3DE594D9, 0x3D571552,
        0x3DA88B9B, 0x3DD97063, 0x3D63DE35, 0x3DCD94EA, 0x3D5604F3, 0x3DBB507F, 0x3DDAFC02,
        0x3D818336, 0x3DCD079E, 0x3D3A6404, 0x3DA7C06A, 0x3DF24ED1, 0x3D90F9E3, 0x3DCE8E30,
        0x3D556A8F, 0x3D9323E3, 0x3DC062A0, 0x3D81DC35, 0x3DE594D9, 0x3D571552, 0x3DA88B9B,
        0x3DD97063, 0x3D63DE35, 0x3DCD94EA, 0x3D5604F3, 0x3DBB507F, 0x3DDAFC02, 0x3D818336,
        0x3DCD079E, 0x3D3A6404, 0x3DA7C06D, 0x3DF24EDA, 0x3D90FA1C, 0x3DCE90A6, 0x3D557D8A,
        0x3D936C2D, 0x3DC21A2F, 0x3D861493,
    ];

    const SMALL2_P1: [u32; 129] = [
        0x3D53EA9B, 0x3DBEDD54, 0x3DF6849A, 0x3D9BF0AE, 0x3DD7984C, 0x3D911B43, 0x3DBE88C8,
        0x3DDE0DFD, 0x3DD714EC, 0x3E299CF4, 0x3E1DDD11, 0x3E35F472, 0x3E6B13D2, 0x3E776422,
        0x3EA9BAF2, 0x3EA10B2F, 0x3ED29E3E, 0x3EFC4253, 0x3F0BDF5B, 0x3F12A576, 0x3F1C6A6D,
        0x3F3685D6, 0x3F4FD3F4, 0x3F445EE2, 0x3F578248, 0x3F502405, 0x3F5621C5, 0x3F5FE38F,
        0x3F49B1F1, 0x3F58C586, 0x3F279E7B, 0x3F36C350, 0x3F3D4794, 0x3EFDEE7E, 0x3F159B31,
        0x3EC21D0D, 0x3EDE76A0, 0x3EE02516, 0x3E6DE795, 0x3EB905AA, 0x3E1FF7A6, 0x3E4E321F,
        0x3E61B5E3, 0x3E0BFEFA, 0x3E2CC81C, 0x3DA053EA, 0x3DBDA8E7, 0x3DEE254A, 0x3D9974F7,
        0x3E000A95, 0x3D6913CE, 0x3DAFF456, 0x3DD512BF, 0x3D5A84F2, 0x3DCB4EEA, 0x3D5B6BE0,
        0x3DBD3D18, 0x3DCF834C, 0x3D7C50AE, 0x3DC825F2, 0x3D35D4F2, 0x3DA39153, 0x3DEC34A7,
        0x3D9169E3, 0x3DC9C418, 0x3D57A1FD, 0x3D899ACE, 0x3DAD713D, 0x3D7CCBB9, 0x3DE6AF76,
        0x3D59B8A5, 0x3DA47FB8, 0x3DCCEBAD, 0x3D54C572, 0x3DC88757, 0x3D586262, 0x3DBC1524,
        0x3DCEDCC3, 0x3D7BEBB8, 0x3DC7D5CC, 0x3D35AB17, 0x3DA38061, 0x3DEC2B36, 0x3D916404,
        0x3DC9C08F, 0x3D57A062, 0x3D899A50, 0x3DAD70A0, 0x3D7CCB5E, 0x3DE6AF52, 0x3D59B89F,
        0x3DA47FB1, 0x3DCCEBA7, 0x3D54C56F, 0x3DC88757, 0x3D586262, 0x3DBC1524, 0x3DCEDCC3,
        0x3D7BEBB8, 0x3DC7D5CC, 0x3D35AB17, 0x3DA38061, 0x3DEC2B36, 0x3D916404, 0x3DC9C08F,
        0x3D57A062, 0x3D899A50, 0x3DAD70A0, 0x3D7CCB5E, 0x3DE6AF52, 0x3D59B89F, 0x3DA47FB1,
        0x3DCCEBA7, 0x3D54C56F, 0x3DC88757, 0x3D586262, 0x3DBC1524, 0x3DCEDCC3, 0x3D7BEBB8,
        0x3DC7D5CC, 0x3D35AB17, 0x3DA38064, 0x3DEC2B41, 0x3D91644C, 0x3DC9C3A6, 0x3D57B83B,
        0x3D89F525, 0x3DAF98F1, 0x3D83B330,
    ];

    /// The reference's own two-pass output on this exact configuration
    /// (`pyradius.vocoder_ops.FormantState.apply`, numba kernels, route A):
    /// every one of the 129 bins must match bit for bit, and the second pass
    /// exercises the persistent `env`/`gain_env` state.
    #[test]
    fn golden_two_passes_match_reference() {
        let mut st = FormantState::new(small_geom(48_000.0, 222));
        st.cfg.ratio = 1.5;
        st.cfg.mode_freq = 1;
        // the reference feeds the *same* pristine `mag.copy()` to both passes,
        // so the second pass exercises the persistent `env`/`gain_env` state
        // against the original spectrum (not against pass 0's output).
        let mut mag = from_bits(&SMALL_MAG);
        st.apply(&mut mag, 0);
        assert_eq!(bits(&mag), SMALL_P0.to_vec(), "pass 0 differs");
        let mut mag = from_bits(&SMALL_MAG);
        st.apply(&mut mag, 0);
        assert_eq!(bits(&mag), SMALL_P1.to_vec(), "pass 1 differs");
    }

    /// The same geometry with the engine's vocoder settings
    /// (`mode_freq = 0`, `ratio = 0.75`, `sr = 44.1 kHz`, `f580 = 300`).
    #[test]
    fn golden_mode_freq0_matches_reference() {
        let mut st = FormantState::new(small_geom(44_100.0, 300));
        st.cfg.ratio = 0.75;
        st.cfg.mode_freq = 0;
        let mut mag = from_bits(&SMALL_MAG);
        st.apply(&mut mag, 0);
        assert_eq!(bits(&mag), SMALL2_P0.to_vec(), "pass 0 differs");
        let mut mag = from_bits(&SMALL_MAG);
        st.apply(&mut mag, 0);
        assert_eq!(bits(&mag), SMALL2_P1.to_vec(), "pass 1 differs");
    }

    /// The task's smoke test: construct + apply on a synthetic magnitude
    /// spectrum, everything finite and the buffer length preserved.
    #[test]
    fn new_and_formant_apply_stays_finite() {
        let mut st = FormantState::new(small_geom(48_000.0, 222));
        st.cfg.ratio = 1.25;
        st.cfg.mode_freq = 1;
        let nb = st.cfg.nb_bins;
        let mut mag: Vec<f32> = (0..nb)
            .map(|i| {
                let x = i as f32 / nb as f32;
                0.01 + 0.9 * (-((x - 0.2) * 12.0).powi(2)).exp() + 1e-3 * (i % 7) as f32
            })
            .collect();
        formant_apply(&mut st, &mut mag, 0);
        assert_eq!(mag.len(), nb);
        assert!(mag.iter().all(|v| v.is_finite()), "non-finite output");
        assert!(mag.iter().all(|v| *v >= 0.0), "magnitudes must stay >= 0");
        // a silent spectrum must not produce NaN (`-391 dB` floor)
        let mut silent = vec![0.0f32; nb];
        formant_apply(&mut st, &mut silent, 0);
        assert!(silent.iter().all(|v| v.is_finite()));
    }

    /// The three gates return the input untouched (`active != 1`,
    /// `ratio == 1.0`, `strength == 0.0`).
    #[test]
    fn gates_leave_mag_untouched() {
        let mk = |f: &dyn Fn(&mut FormantState)| {
            let mut st = FormantState::new(small_geom(48_000.0, 222));
            st.cfg.ratio = 1.5;
            f(&mut st);
            let mut mag = from_bits(&SMALL_MAG);
            let before = mag.clone();
            st.apply(&mut mag, 0);
            mag == before
        };
        assert!(mk(&|st| st.cfg.active = 0), "active = 0 must be a no-op");
        assert!(mk(&|st| st.cfg.ratio = 1.0), "ratio = 1 must be a no-op");
        assert!(
            mk(&|st| st.cfg.strength = 0.0),
            "strength = 0 must be a no-op"
        );
    }

    /// `_f580()` falls back to 222 for non-positive values and the persistent
    /// state starts at `env = 0`, `gain_env = 1` (`rx_formant_state_init`).
    #[test]
    fn state_initial_conditions() {
        let st = FormantState::new(small_geom(48_000.0, 0));
        assert_eq!(st.f580(), 222);
        let mut st2 = FormantState::new(small_geom(48_000.0, 3592));
        assert_eq!(st2.f580(), 3592);
        st2.cfg.f580 = -5;
        assert_eq!(st2.f580(), 222);
        assert_eq!(st.env.len(), st.cfg.m_bins + 8);
        assert!(st.env.iter().all(|v| *v == 0.0));
        assert_eq!(st.gain_env.len(), st.cfg.nb_bands);
        assert!(st.gain_env.iter().all(|g| g.len() == st.cfg.nb_bins + 8));
        assert!(st.gain_env.iter().all(|g| g.iter().all(|v| *v == 1.0)));
        assert_eq!(st.inv_scale, 1.0 / 64.0);
        // engine geometry: h1 = round(16384*4/48000) = 1, h2 = round(16384*9/48000) = 3
        let big = FormantState::new(FormantConfig {
            nb_bins: 8193,
            n_fft: 16384,
            m_fft: 4096,
            m_bins: 2049,
            f580: 222,
            sr: 48_000.0,
            prec_mode: 2,
            nb_bands: 4,
        });
        assert_eq!((big.h1, big.h2), (1, 3));
    }

    /// A delta transforms to a flat spectrum and the unnormalised inverse
    /// recovers it, up to the `f32` twiddle noise (`~1e-9` absolute against a
    /// peak of 64, i.e. `~1.5e-11` after the `1/m` normalisation).
    #[test]
    fn fft_round_trip() {
        let mut st = FormantState::new(small_geom(48_000.0, 222));
        let m = st.cfg.m_fft;
        st.ffr[..].copy_from_slice(&[0.0; 64]);
        st.ffi[..].copy_from_slice(&[0.0; 64]);
        st.ffr[3] = 1.0;
        let orig = st.ffr.clone();
        st.fft(0);
        // forward transform of a delta: every bin has magnitude 1
        for k in 0..m {
            let mag = (st.ffr[k] * st.ffr[k] + st.ffi[k] * st.ffi[k]).sqrt();
            assert!((mag - 1.0).abs() < 1e-6, "bin {k} magnitude {mag}");
        }
        st.fft(1);
        let back: Vec<f32> = st.ffr.iter().map(|v| v / m as f32).collect();
        for i in 0..m {
            assert!(
                (back[i] - orig[i]).abs() < 1e-6,
                "bin {i}: {} != {}",
                back[i],
                orig[i]
            );
        }
    }

    /// `pairwise_sum` must match `f64` sequential summation for short inputs and
    /// stay finite/associative-shaped for long ones.
    #[test]
    fn pairwise_sum_shape() {
        let v: Vec<f64> = (0..1000).map(|i| 1.0 + i as f64 * 1e-9).collect();
        let pw = pairwise_sum(&v);
        let lin: f64 = v.iter().sum();
        assert!((pw - lin).abs() / lin < 1e-12);
        assert_eq!(pairwise_sum(&[]), 0.0);
        assert_eq!(pairwise_sum(&[1.5]), 1.5);
    }
}
