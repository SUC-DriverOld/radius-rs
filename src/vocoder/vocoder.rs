//! Phase-vocoder engine — port of `pyradius/vocoder_core.py`
//! (`libradius/src/engine/vocoder_core.c`).
//!
//! Chain per granule: `FillGranule` → `FGWin x4` → ACS (window, FFT, envelope
//! IIR, polar, peak search) → `Unwrap` → `ApplyPitchCoherence` → pull-to-peak →
//! stereo `Sync` → assembly (`RPT` → Copy1 → Copy2 → AFC → P2C → FFTInv → fold +
//! IIR → OAC) → cursor advance. `feed` pushes the input through the crossover
//! into the four band rings; `render` mirrors the C driver's feed rhythm and the
//! `InterpolateNSamples` drain loop.
//!
//! The reference's optional route-B kernels (`PYR_FAST`) are *not* ported: the
//! default NumPy path is the bit-exact reference for the C engine, and
//! `EXACT_FFT = True` is fixed here for the same reason.

use crate::fft::with_plan;
use crate::interp::{interp_nsamples, InterpTable};
use crate::simple_rand::{noise_template_fill, SimpleRand};
use crate::vocoder::crossover_tables as tables_data;
use crate::vocoder::*;
use std::cell::RefCell;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

#[derive(Default, Clone, Copy)]
struct VcProfile {
    fill: Duration,
    acs: Duration,
    unwrap: Duration,
    apc: Duration,
    pull: Duration,
    sync: Duration,
    assembly: Duration,
    formant: Duration,
    phase_cart: Duration,
    inverse_fft: Duration,
    fold_ola: Duration,
    /// Outside the per-granule chain, and therefore invisible to every stage
    /// above: the band split in [`VocoderState::feed`].
    crossover: Duration,
    /// The band samples going into the per-channel rings, also in `feed`.
    ring_scatter: Duration,
    /// `take_output` plus the resampler drain in [`VocoderState::render`].
    drain: Duration,
    /// Copying the drained frames into the caller's interleaved buffer.
    out_copy: Duration,
    granules: u64,
    /// Wall time from the start of `render` to the profile report. Everything
    /// above is compared against this, so "unaccounted" cannot hide again.
    wall: Duration,
}

thread_local! {
    static VC_PROFILE: RefCell<VcProfile> = RefCell::new(VcProfile::default());
}

#[inline]
fn profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("RADIUS_PROFILE").is_some())
}

/// Opt-in approximation path for the expensive phase-to-cartesian conversion.
/// The default remains the reference-compatible f64 `sin_cos`; setting
/// `RADIUS_VC_FAST_MATH=1` uses the platform f32 implementation instead.
fn fast_math_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("RADIUS_VC_FAST_MATH").is_some())
}

#[inline]
fn profile_reset() {
    if profile_enabled() {
        VC_PROFILE.with(|p| *p.borrow_mut() = VcProfile::default());
    }
}

fn profile_report() {
    if !profile_enabled() {
        return;
    }
    VC_PROFILE.with(|p| {
        let p = *p.borrow();
        let ms = |d: Duration| d.as_secs_f64() * 1000.0;
        // The per-granule chain, then the parts of `render` that are outside it.
        // Reporting both, against measured wall time, is the whole point: the
        // crossover used to be 53% of this engine and was invisible here because
        // it runs in `feed`, between granules.
        let chain = p.fill + p.acs + p.unwrap + p.apc + p.pull + p.sync + p.assembly;
        let outside = p.crossover + p.ring_scatter + p.drain + p.out_copy;
        let accounted = chain + outside;
        let unaccounted = p.wall.saturating_sub(accounted);
        let pct = |d: Duration| {
            if p.wall.is_zero() {
                0.0
            } else {
                100.0 * d.as_secs_f64() / p.wall.as_secs_f64()
            }
        };
        eprintln!(
            "vc profile: granules={} wall={:.1} ms | chain={:.1} ({:.0}%) crossover={:.1} ({:.0}%) ring={:.1} drain={:.1} out_copy={:.1} | unaccounted={:.1} ({:.0}%)",
            p.granules,
            ms(p.wall),
            ms(chain),
            pct(chain),
            ms(p.crossover),
            pct(p.crossover),
            ms(p.ring_scatter),
            ms(p.drain),
            ms(p.out_copy),
            ms(unaccounted),
            pct(unaccounted),
        );
        eprintln!(
            "vc profile:   chain detail: fill={:.1} acs={:.1} unwrap={:.1} apc={:.1} pull={:.1} sync={:.1} assembly={:.1} (formant={:.1} phase_cart={:.1} inv_fft={:.1} fold_ola={:.1})",
            ms(p.fill), ms(p.acs), ms(p.unwrap), ms(p.apc), ms(p.pull), ms(p.sync),
            ms(p.assembly), ms(p.formant), ms(p.phase_cart), ms(p.inverse_fft), ms(p.fold_ola),
        );
    });
}

#[inline]
fn profile_add(slot: fn(&mut VcProfile) -> &mut Duration, start: Instant) {
    if profile_enabled() {
        VC_PROFILE.with(|p| *slot(&mut p.borrow_mut()) += start.elapsed());
    }
}

/// `_FOLD_ALPHA` (`vocoder_core.py` line 307).
pub const FOLD_ALPHA: f32 = 0.24935225;
/// `np.float32(0.2)`.
const F02: f32 = 0.2;
/// `_bits(0x2B8CBCCC)` — the lower clamp applied to `mag`.
const MAG_FLOOR_BITS: u32 = 0x2B8C_BCCC;
/// `_bits(0x3D4CCCCD)` / `_bits(0x3DCCCCCD)` — APC IIR time constants.
const APC_TAU_TRANSIENT_BITS: u32 = 0x3D4C_CCCD;
const APC_TAU_STEADY_BITS: u32 = 0x3DCC_CCCD;

/// Supported vocoder sample rates (`rx_vc_render` rejects everything else).
pub fn supported_rate(sr: u32) -> bool {
    sr == 48000 || sr == 44100
}

/// A shared "output frames made" counter for progress reporting.
///
/// `Arc<AtomicU64>` is `Send`/`Sync` even though [`VocoderState`] is not, which is
/// what lets a watchdog thread watch a render running elsewhere.
pub type ProgressCounter = std::sync::Arc<std::sync::atomic::AtomicU64>;

/// Create a zeroed progress counter.
pub fn new_progress_counter() -> ProgressCounter {
    std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0))
}

/// `vc_calc_sched` (`pyradius/tables.py`): `(pos, cdel, in_step, out_step)`.
pub fn vc_calc_sched(g: i64, ratio: f64, nominal_step: f64) -> (i64, i64, i64, i64) {
    if g < 5 {
        return (0, 0, 0, 0);
    }
    let grp = (g - 1) / 4;
    let (in_step, out_step) = if ratio >= 1.0 {
        (
            nominal_step.round() as i64,
            (nominal_step * ratio).round() as i64,
        )
    } else {
        let i = (nominal_step * std::f64::consts::SQRT_2).round() as i64;
        (i, (i as f64 * ratio).round() as i64)
    };
    let pos = grp * out_step;
    let cdel = if g % 4 == 1 { in_step } else { 0 };
    (pos, cdel, in_step, out_step)
}

/// Run configuration (the engine's calibrated case plus the rate-dependent
/// geometry).
#[derive(Debug, Clone)]
pub struct VocoderConfig {
    pub sr: u32,
    pub nch: usize,
    pub precision: i32,
    pub nbands: usize,
    pub n_fft: usize,
    pub nb_bins: usize,
    pub m_fft: usize,
    pub mb_bins: usize,
    pub n_write: usize,
    pub hop: i32,
    pub trans_sens: f32,
    pub quality: f32,
    pub step_base: f64,
    pub ring_out_len: usize,
    /// `stretch * pitchRatio` — the engine's `this[0x20]` (`set_ratio`).
    pub total_ratio: f64,
}

impl VocoderConfig {
    pub fn for_rate(sr: u32, nch: usize, precision: i32) -> Self {
        let (n_fft, nb_bins, m_fft, mb_bins, n_write, hop) = if sr == 44100 {
            (
                8192usize, 4097usize, 2048usize, 1025usize, 6599usize, 3299i32,
            )
        } else {
            (
                16384usize, 8193usize, 4096usize, 2049usize, 7184usize, 3592i32,
            )
        };
        Self {
            sr,
            nch,
            precision,
            nbands: 4,
            n_fft,
            nb_bins,
            m_fft,
            mb_bins,
            n_write,
            hop,
            trans_sens: 1.0,
            quality: 37.0,
            step_base: 222.0,
            ring_out_len: sr as usize + 65536,
            total_ratio: 1.0,
        }
    }
}

/// One channel's persistent analysis/synthesis containers.
struct Channel {
    mag: Vec<f32>,
    mag_copy: Vec<f32>,
    mask: Vec<f32>,
    mask_copy: Vec<f32>,
    phase: Vec<f32>,
}

impl Channel {
    fn new(nb_bins: usize) -> Self {
        Self {
            mag: vec![0.0; nb_bins],
            mag_copy: vec![0.0; nb_bins],
            mask: vec![0.0; nb_bins],
            mask_copy: vec![0.0; nb_bins],
            phase: vec![0.0; nb_bins],
        }
    }
}

/// `VocoderState` — mirrors `rx_vc_state`.
pub struct VocoderState {
    pub cfg: VocoderConfig,
    // ---- geometry ----
    ring_cap: usize,
    win_table: Vec<f32>, // [nbands][n_write] flattened
    synth_a: Vec<f32>,
    synth_b: Vec<f32>,
    v8_len_a: usize,
    v9_len_b: usize,
    // ---- cursors ----
    cursor_1376: i64,
    acc_1440: f64,
    pos_1384: i64,
    writepos_1224: i64,
    hop_1420: i32,
    granule_1400: i32,
    out_step_1404: i32,
    prev_granule_1408: i32,
    step_1412: i32,
    comp_1432: f64,
    transient_state_2488: u32,
    transient_flag_1208: u8,
    /// `+0x1352` — always 1 on the `ProcessBuffered` path (the C asserts it);
    /// kept so `FillGranule`'s visibility condition can mirror the engine.
    #[allow(dead_code)]
    buffered_1352: u8,
    g_index: i64,
    fed_total: i64,
    out_total: i64,
    /// Output frames produced, for progress reporting.
    ///
    /// Shared rather than owned so a watchdog thread can poll it while the
    /// renderer runs on this thread. Written with `Relaxed` ordering only: a
    /// progress counter, not a synchronisation primitive.
    progress: ProgressCounter,
    /// Input frames this render expects (set by [`VocoderState::render`]).
    progress_total: i64,
    out_reported: i64,
    out_read_pos: i64,
    copy_len_594: usize,
    sched_cdel: i64,
    noise_slot_958: i32,
    // ---- containers ----
    ring: Vec<f32>, // [nch][nbands][ring_cap] — one band ring per channel
    ch: Vec<Channel>,
    cart: Vec<f32>,
    frame: Vec<f32>,
    acc: Vec<f32>,
    acs_env: Vec<f32>,     // [nch][nb_bins]
    region_gain: Vec<f32>, // [nch][nb_bins]
    noise_phase: Vec<f32>,
    noise_gain: Vec<f32>,
    noise_template: Vec<f32>,
    noise_weight_3472: Vec<f32>,
    /// `+0x3504` and `+0x3808`: allocated by `rx_vc_init` and written by the
    /// route-B substitute/randomize path, but never read by the engine's
    /// default chain (verified against `pyradius`, which keeps them as dead
    /// state). Kept so the container topology matches the C struct.
    #[allow(dead_code)]
    sync_weight_3504: Vec<f32>,
    #[allow(dead_code)]
    gain_mean_3808: Vec<f32>,
    dir_cur: Vec<f32>,
    dir_prev: Vec<f32>,
    seg_bound_1830: Vec<i64>,
    peak_bins: Vec<i64>,
    reg_start: Vec<i64>,
    reg_end: Vec<i64>,
    reg_offset: Vec<f32>,
    scratch_8a8: Vec<f32>,
    b_710: Vec<f32>,
    mask_table_8f8: Vec<f32>,
    sync_weight_buf: Vec<f32>,
    peak_count: usize,
    sync_sens_3496: f32,
    pitch_freq_169: f32,
    pitch_metric_168: f32,
    win1: Vec<f32>,
    win2: Vec<f32>,
    out_ring: Vec<f32>, // [nch][ring_out_len]
    edge_gain: [f32; 4],
    rng: SimpleRand,
    /// One 4-band crossover **per channel**.
    ///
    /// The reference feeds only channel 0 and lets every channel's analysis read that
    /// same band data, which makes the vocoder collapse stereo to a phantom centre (a
    /// phase-inverted pair comes back in phase, and a hard-panned right channel comes
    /// back at the left channel's level). See [`VocoderState::feed`].
    xovers: Vec<Crossover>,
    formant: FormantState,
    // ---- ratio ----
    pub ratio: f64,
    pub stretch: f64,
    pub pitch_ratio: f64,
    pub total_ratio: f64,
    trace: bool,
}

impl VocoderState {
    pub fn new(sr: u32, nch: usize, precision: i32) -> Self {
        Self::with_config(VocoderConfig::for_rate(sr, nch, precision))
    }

    pub fn with_config(cfg: VocoderConfig) -> Self {
        let cfg = cfg;
        let (nb, n_ch) = (cfg.nb_bins, cfg.nch);
        let ring_cap = 4 * cfg.n_write;
        let (win_table, synth_a, synth_b, v8, v9) = if cfg.sr == 44100 {
            let win = tables_data::win_44k().to_vec();
            let (a, b) = tables_data::synth_44k();
            (win, a.to_vec(), b.to_vec(), 2688usize, 1152usize)
        } else {
            let win = tables_data::win_48k().to_vec();
            let (a, b) = Self::build_synth_windows(cfg.n_write);
            (win, a, b, 2930usize, 1256usize)
        };
        let noise_template = vec![0.0f32; 2 * 128 * nb];
        let noise_weight = (0..nb)
            .map(|k| ((k as f32) / nb as f32).powf(F02))
            .collect();
        let formant = FormantState::new(FormantConfig {
            nb_bins: nb,
            n_fft: cfg.n_fft,
            m_fft: cfg.m_fft,
            m_bins: cfg.mb_bins,
            f580: cfg.step_base as i32,
            sr: cfg.sr as f32,
            prec_mode: 2,
            nb_bands: cfg.nbands,
        });
        Self {
            ring_cap,
            win_table,
            synth_a,
            synth_b,
            v8_len_a: v8,
            v9_len_b: v9,
            cursor_1376: 0,
            acc_1440: 0.0,
            pos_1384: 0,
            writepos_1224: 0,
            hop_1420: cfg.hop,
            granule_1400: cfg.hop,
            out_step_1404: cfg.hop,
            prev_granule_1408: cfg.hop,
            step_1412: 0,
            comp_1432: 0.0,
            transient_state_2488: 0,
            transient_flag_1208: 0,
            buffered_1352: 1,
            g_index: 0,
            fed_total: 0,
            out_total: 0,
            progress: new_progress_counter(),
            progress_total: 1,
            out_reported: 0,
            out_read_pos: 0,
            copy_len_594: nb,
            sched_cdel: -1,
            noise_slot_958: 0,
            ring: vec![0.0; n_ch * cfg.nbands * ring_cap],
            ch: (0..n_ch).map(|_| Channel::new(nb)).collect(),
            cart: vec![0.0; cfg.n_fft + 2],
            frame: vec![0.0; cfg.n_fft],
            acc: vec![0.0; cfg.n_fft],
            acs_env: vec![0.0; n_ch * nb],
            region_gain: vec![0.0; n_ch * nb],
            noise_phase: vec![0.0; n_ch * nb],
            noise_gain: vec![0.0; n_ch * nb],
            noise_template,
            noise_weight_3472: noise_weight,
            sync_weight_3504: vec![0.0; nb],
            gain_mean_3808: vec![0.0; nb],
            dir_cur: vec![0.0; n_ch * nb],
            dir_prev: vec![0.0; n_ch * nb],
            seg_bound_1830: vec![nb as i64; 6],
            peak_bins: vec![0; nb],
            reg_start: vec![0; nb],
            reg_end: vec![0; nb],
            reg_offset: vec![0.0; nb],
            scratch_8a8: vec![0.0; nb],
            b_710: vec![0.0; nb],
            mask_table_8f8: vec![0.0; nb],
            sync_weight_buf: vec![0.0; nb],
            peak_count: 0,
            // Stereo phase-synchronisation sensitivity. This was a `0.0` placeholder,
            // which is not a neutral value: the sync weight is
            // `sqrt(1 - clamp(inv * v9 / 0.7))` with `inv = 1/(sens + 1e-6)`, so
            // `sens = 0` drives `v11` to 1 and the weight to exactly 0, making
            // `SynchronizeStereoPhases` a no-op. That is why vocoder output used to
            // lose the inter-channel phase relationship entirely (L/R coherence fell
            // from 0.2-0.7 on the input to 0.02-0.11), while Audition keeps it.
            //
            // 0.75 is the value this crate's own golden test for the operator passes
            // (`noise.rs`, SYNC_SENS_BITS = 0x3F400000), i.e. the value the ported
            // operator was validated against.
            sync_sens_3496: f32::from_bits(0x3F40_0000),
            pitch_freq_169: 0.0,
            pitch_metric_168: 0.0,
            win1: vec![0.0; cfg.n_fft],
            win2: vec![0.0; cfg.n_fft],
            out_ring: vec![0.0; n_ch * cfg.ring_out_len],
            edge_gain: [1.15, 1.30, 1.05, 1.10],
            rng: SimpleRand::new(1),
            xovers: Vec::new(),
            formant,
            ratio: 1.0,
            stretch: 1.0,
            pitch_ratio: 1.0,
            total_ratio: 1.0,
            trace: false,
            cfg,
        }
    }

    /// `_build_synth_windows` (48 kHz): `HanningWindow^0.7` at two lengths.
    fn build_synth_windows(n_write: usize) -> (Vec<f32>, Vec<f32>) {
        let len_a = 2930usize;
        let len_b = 1256usize;
        let mut a = vec![0.0f32; n_write];
        let mut b = vec![0.0f32; n_write];
        let pi_f = std::f64::consts::PI as f32;
        let off_b = (n_write / 2) - (len_b / 2);
        let off_a = (n_write / 2) - (len_a / 2);
        let scale_a = ((len_b as f32) / (len_a as f32)).sqrt();
        for i in 0..len_b {
            let ang = (2.0f32 * pi_f * (i as f32 + 0.5) / len_b as f32) as f64;
            let val = 0.5f32 * (1.0f32 - (ang.cos() as f32));
            b[off_b + i] = (val as f64).powf(0.7) as f32;
        }
        for i in 0..len_a {
            let ang = (2.0f32 * pi_f * (i as f32 + 0.5) / len_a as f32) as f64;
            let val = 0.5f32 * (1.0f32 - (ang.cos() as f32));
            a[off_a + i] = ((val as f64).powf(0.7) as f32) * scale_a;
        }
        (a, b)
    }

    /// `set_ratio` — `pitch_chain(semis)` + the driver's `exp2` chain.
    pub fn set_ratio(&mut self, semis: f64, tempo: f64) {
        self.stretch = tempo / 100.0;
        let pr = 2.0f64.powf(semis / 12.0);
        let s = 12.0f32 * (pr as f32).log2();
        self.ratio = 2.0f64.powf(s as f64 / 12.0) * self.stretch;
        self.pitch_ratio = self.ratio;
        self.total_ratio = self.ratio;
        self.cfg.total_ratio = self.ratio;
        self.comp_1432 = self.ratio * self.hop_1420 as f64;
        self.formant.cfg.ratio = self.ratio as f32;
    }

    #[inline]
    fn env_of(&self, ch: usize) -> &[f32] {
        let n = self.cfg.nb_bins;
        &self.acs_env[ch * n..(ch + 1) * n]
    }

    #[inline]
    fn region_gain_of(&self, ch: usize) -> &[f32] {
        let n = self.cfg.nb_bins;
        &self.region_gain[ch * n..(ch + 1) * n]
    }

    #[inline]
    fn fft_fwd_into(n: usize, src: &[f32], dst: &mut [f32]) {
        with_plan(n, |p| p.fwd_into(src, dst));
    }

    #[inline]
    fn fft_inv_into(n: usize, cart: &[f32], dst: &mut [f32]) {
        with_plan(n, |p| p.inv_into(cart, dst));
    }

    // -----------------------------------------------------------------
    // lifecycle
    // -----------------------------------------------------------------

    /// `rx_vc_start_streaming`.
    pub fn start_streaming(&mut self) {
        self.rng = SimpleRand::new(1);
        self.transient_flag_1208 = 1;
        self.noise_slot_958 = 0;
        noise_template_fill(&mut self.noise_template, &mut self.rng);
        for v in self.acs_env.iter_mut() {
            *v = 0.0;
        }
        for v in self.noise_phase.iter_mut() {
            *v = 0.0;
        }
        for v in self.noise_gain.iter_mut() {
            *v = 0.0;
        }
    }

    // -----------------------------------------------------------------
    // events
    // -----------------------------------------------------------------

    /// `_ev_fill_granule`.
    fn ev_fill_granule(&mut self, ch: usize) {
        let t0 = profile_enabled().then(Instant::now);
        for v in self.acc.iter_mut() {
            *v = 0.0;
        }
        let buffered = self.cursor_1376 >= self.hop_1420 as i64;
        // This channel's own band ring, not channel 0's.
        let cap = self.ring_cap;
        let bands = self.cfg.nbands;
        let base = (ch % self.cfg.nch.max(1)) * bands * cap;
        let ring = &self.ring[base..base + bands * cap];
        let win = &self.win_table;
        let acc = &mut self.acc;
        fill_granule(
            ring,
            win,
            acc,
            FillMode::Fg,
            self.hop_1420 as i64,
            self.cfg.n_write,
            self.cfg.nbands,
            0,
            self.cursor_1376,
            buffered,
            cap as i64,
            (cap / 2) as i64,
            (cap / 2) as i64,
            1.0,
        );
        if let Some(t0) = t0 { profile_add(|p| &mut p.fill, t0); }
    }

    /// `_ev_fgwin_x4`.
    fn ev_fgwin_x4(&mut self, ch: usize) {
        let cap = self.ring_cap;
        let bands = self.cfg.nbands;
        let base = (ch % self.cfg.nch.max(1)) * bands * cap;
        let ring = &self.ring[base..base + bands * cap];
        let win = &self.win_table;
        let acc = &mut self.acc;
        fill_granule(
            ring,
            win,
            acc,
            FillMode::Fgw,
            self.hop_1420 as i64,
            self.cfg.n_write,
            self.cfg.nbands,
            self.cfg.n_fft,
            self.cursor_1376,
            true,
            cap as i64,
            (cap / 2) as i64,
            (cap / 2) as i64,
            1.0,
        );
    }

    /// `_ev_acs_front` — window, spectrum #1, envelope IIR.
    fn ev_acs_front(&mut self, ch: usize) {
        let n_write = self.cfg.n_write;
        let nb = self.cfg.nb_bins;
        let win0 = &self.win_table[..n_write];
        for i in 0..n_write {
            self.acc[i] *= win0[i];
        }
        // Reuse the persistent cartesian spectrum buffer.  A 16384-point
        // transform returns ~64 KiB; allocating two such vectors per granule
        // was a measurable part of vc render time.
        with_plan(self.cfg.n_fft, |p| p.fwd_in_place(&mut self.acc, &mut self.cart));
        let tau = if self.transient_state_2488 == 2 {
            f32::from_bits(APC_TAU_TRANSIENT_BITS)
        } else {
            f32::from_bits(APC_TAU_STEADY_BITS)
        };
        let hopf = if self.step_1412 > 0 {
            self.step_1412 as f32
        } else {
            288.0
        };
        let iir_a = time_to_iir_a(tau, self.cfg.sr as f32 / hopf);
        {
            let cart = &self.cart;
            let env_start = ch * nb;
            let env = &mut self.acs_env[env_start..env_start + nb];
            for k in 0..nb {
                let re = cart[2 * k];
                let im = cart[2 * k + 1];
                let m = (re * re + im * im).sqrt();
                env[k] = fma(m - env[k], iir_a, env[k]);
            }
        }
        for v in self.acc.iter_mut() {
            *v = 0.0;
        }
    }

    /// `_ev_acs` — spectrum #1 + envelope, FGWin, spectrum #2, peaks.
    fn ev_acs(&mut self, ch: usize) {
        let t0 = profile_enabled().then(Instant::now);
        self.ev_acs_front(ch);
        self.ev_fgwin_x4(ch);
        Self::fft_fwd_into(self.cfg.n_fft, &self.acc, &mut self.cart);
        let nb = self.cfg.nb_bins;
        let floor = f32::from_bits(MAG_FLOOR_BITS);
        {
            let c = &mut self.ch[ch];
            cart_to_polar_into(&self.cart, &mut c.mag, &mut c.mask);
            for k in 0..nb {
                c.mag[k] = c.mag[k].max(floor);
            }
        }
        self.ev_find_peaks(ch);
        if let Some(t0) = t0 { profile_add(|p| &mut p.acs, t0); }
    }

    /// `_ev_find_peaks` — peaks, parabolic offsets, regions, region gains.
    fn ev_find_peaks(&mut self, ch: usize) {
        let nb = self.cfg.nb_bins;
        let mag = &self.ch[ch].mag;
        let mut hit_count = 0usize;
        for i in 1..nb.saturating_sub(1) {
            if mag[i] > mag[i - 1] && mag[i] > mag[i + 1] {
                self.peak_bins[hit_count] = i as i64;
                hit_count += 1;
            }
        }
        let cnt = hit_count;
        if cnt == 0 {
            self.peak_bins[0] = (nb / 2) as i64;
            self.reg_offset[0] = (nb / 2) as f32;
            self.reg_start[0] = 0;
            // FIX #1 (tools/libradius_vc_oob_fix.md): the C sets nbins, which
            // makes UnwrapPhase read mag[nbins]; the port clamps to nbins-1.
            self.reg_end[0] = nb as i64 - 1;
            self.peak_count = 1;
            return;
        }
        // v20 = (ap - 2*am) - amm ; off = (ap - amm)/(v20+v20), clamped +-0.7
        for i in 0..cnt {
            let h = self.peak_bins[i] as usize;
            let ap = mag[h + 1];
            let am = mag[h];
            let amm = mag[h - 1];
            let v20 = (ap - (2.0f32 * am)) - amm;
            let mut off = 0.0f32;
            if v20 != 0.0 {
                off = (ap - amm) / (v20 + v20);
            }
            off = off.clamp(-0.7, 0.7);
            self.reg_offset[i] = h as f32 + off;
        }
        self.peak_count = cnt;

        let mids = seg_first_min(mag, &self.peak_bins[..cnt - 1], &self.peak_bins[1..cnt]);
        // reg_start[0] = argmin(mag[..hits[0]+1])
        let mut best = 0usize;
        let first_peak = self.peak_bins[0] as usize;
        for i in 1..=first_peak {
            if mag[i] < mag[best] {
                best = i;
            }
        }
        self.reg_start[0] = best as i64;
        for k in 0..cnt - 1 {
            self.reg_end[k] = mids[k];
            self.reg_start[k + 1] = mids[k];
        }
        let pk_last = self.peak_bins[cnt - 1] as usize;
        let mut best = pk_last;
        for i in pk_last..nb {
            if mag[i] < mag[best] {
                best = i;
            }
        }
        self.reg_end[cnt - 1] = best as i64;
        self.peak_bins[cnt..nb].fill(0);

        // region gains from the envelope's per-region min/max ratio
        let n = self.cfg.nb_bins;
        let env = &self.acs_env[ch * n..(ch + 1) * n];
        let rg = &mut self.region_gain[ch * n..(ch + 1) * n];
        for k in 0..cnt {
            let s = self.reg_start[k] as usize;
            let e = self.reg_end[k] as usize;
            if e > s {
                let (mn, mx) = seg_minmax_span(env, s, e);
                let g = if mn > 0.0 { mx / mn } else { 1.5f32 };
                for b in rg[s..e].iter_mut() {
                    *b = g;
                }
            }
        }
    }

    /// `_ev_unwrap`.
    fn ev_unwrap(&mut self, ch: usize) {
        let t0 = profile_enabled().then(Instant::now);
        let cnt = self.peak_count;
        let nb = self.cfg.nb_bins;
        let mut scratch = std::mem::take(&mut self.scratch_8a8);
        let mut phase = std::mem::take(&mut self.ch[ch].phase);
        unwrap_phase(
            &self.ch[ch].mask,
            &self.ch[ch].mask_copy,
            &self.ch[ch].mag_copy,
            &self.reg_start[..cnt],
            &self.reg_end[..cnt],
            &self.peak_bins[..cnt],
            &self.reg_offset[..cnt],
            &mut scratch,
            &mut phase,
            self.prev_granule_1408 as f32,
            self.step_1412 as f32,
            self.cfg.n_fft as f32,
            nb as i64,
            self.copy_len_594 as i64,
        );
        self.scratch_8a8 = scratch;
        self.ch[ch].phase = phase;
        if let Some(t0) = t0 { profile_add(|p| &mut p.unwrap, t0); }
    }

    /// `_ev_apc` — `ApplyPitchCoherence`, gated on precision and sensitivity.
    fn ev_apc(&mut self, ch: usize) {
        let t0 = profile_enabled().then(Instant::now);
        if self.precision_gate_off() {
            return;
        }
        let cnt = self.peak_count;
        let args = ApcArgs {
            precision: self.cfg.precision,
            trans_sens: self.cfg.trans_sens,
            total_ratio: self.ratio,
            transient_state: self.transient_state_2488 as i32,
            n_fft: self.cfg.n_fft as i32,
            sr: self.cfg.sr as i64,
            f580: self.hop_1420 as i64,
            f584: self.hop_1420 as i64,
            acc_fc0: 0.0,
            coh_center: 0.0,
            seg_count: 5,
            peak_count: cnt,
            vector_fmaf_region: false,
        };
        let mut phase_mod = self.ch[ch].phase.clone();
        let mut dir_cur = self.dir_cur[ch * self.cfg.nb_bins..(ch + 1) * self.cfg.nb_bins].to_vec();
        let mut dir_prev =
            self.dir_prev[ch * self.cfg.nb_bins..(ch + 1) * self.cfg.nb_bins].to_vec();
        let env = self.env_of(ch).to_vec();
        let region_gain = self.region_gain_of(ch).to_vec();
        let mut seg_bound = std::mem::take(&mut self.seg_bound_1830);
        apply_pitch_coherence(
            &args,
            &env,
            &self.peak_bins,
            &self.reg_start,
            &self.reg_end,
            &mut seg_bound,
            &self.ch[ch].mask,
            &mut phase_mod,
            &mut dir_cur,
            &mut dir_prev,
            &region_gain,
            &self.noise_weight_3472,
            self.pitch_freq_169,
            self.pitch_metric_168,
        );
        self.seg_bound_1830 = seg_bound;
        self.ch[ch].phase.copy_from_slice(&phase_mod);
        let n = self.cfg.nb_bins;
        self.dir_cur[ch * n..(ch + 1) * n].copy_from_slice(&dir_cur);
        self.dir_prev[ch * n..(ch + 1) * n].copy_from_slice(&dir_prev);
        if let Some(t0) = t0 { profile_add(|p| &mut p.apc, t0); }
    }

    fn precision_gate_off(&self) -> bool {
        self.cfg.precision > 9 || self.cfg.trans_sens == 0.0
    }

    /// `_ev_pull_to_peak` — loose phase locking.
    fn ev_pull_to_peak(&mut self, ch: usize) {
        let t0 = profile_enabled().then(Instant::now);
        let cnt = self.peak_count;
        if cnt < 1 {
            return;
        }
        let f1 = self.prev_granule_1408 as f32;
        let f2 = self.step_1412 as f32;
        if f1 <= 0.0 {
            return;
        }
        let ratio = f2 / f1;
        let mut v74 = 0.0f32;
        if ratio > 1.0 {
            let r_clamp = if ratio >= 4.0 {
                1.0f32
            } else {
                (ratio - 1.0) / 3.0
            };
            v74 = r_clamp.sqrt();
        }
        let lo = 0.5f32 + (0.5f32 * v74);
        let hi = 0.5f32 + (4.0f32 * v74);
        let inv_range = if hi > lo { 1.0f32 / (hi - lo) } else { 0.0 };
        let rs: Vec<i64> = self.reg_start[..cnt].to_vec();
        let re: Vec<i64> = self.reg_end[..cnt].to_vec();
        let pk: Vec<i64> = self.peak_bins[..cnt].to_vec();
        let rg = self.region_gain_of(ch).to_vec();
        let mut phase = std::mem::take(&mut self.ch[ch].phase);
        let mask = &self.ch[ch].mask;
        for r in 0..cnt {
            let (s, e) = (rs[r], re[r]);
            if e <= s {
                continue;
            }
            for b in s..e {
                let b = b as usize;
                if b >= phase.len() {
                    break;
                }
                let g = rg[b];
                let w = if g >= hi {
                    1.0f32
                } else if g > lo {
                    (g - lo) * inv_range
                } else {
                    0.0f32
                };
                let p = pk[r] as usize;
                if w > 0.0 && b != p {
                    let pm_bin = phase[b];
                    let t0 = wrap_pi_fused(phase[p] - pm_bin);
                    let t1 = wrap_pi_fused(mask[b] - mask[p]);
                    phase[b] = fma(w, wrap_pi_fused(t0 + t1), pm_bin);
                }
            }
        }
        self.ch[ch].phase = phase;
        if let Some(t0) = t0 { profile_add(|p| &mut p.pull, t0); }
    }

    /// `_ev_sync` — `SynchronizeStereoPhases`.
    fn ev_sync(&mut self) {
        let t0 = profile_enabled().then(Instant::now);
        if self.cfg.nch < 2 || self.peak_count < 1 {
            return;
        }
        let cnt = self.peak_count;
        let nb = self.cfg.nb_bins;
        let mags: Vec<Vec<f32>> = (0..self.cfg.nch).map(|c| self.ch[c].mag.clone()).collect();
        let masks: Vec<Vec<f32>> = (0..self.cfg.nch).map(|c| self.ch[c].mask.clone()).collect();
        let phases: Vec<Vec<f32>> = (0..self.cfg.nch)
            .map(|c| self.ch[c].phase.clone())
            .collect();
        let (dst, weight) = synchronize_stereo_phases(
            &mags,
            &masks,
            &phases,
            &self.peak_bins[..cnt],
            &self.reg_start[..cnt],
            &self.reg_end[..cnt],
            cnt,
            self.sync_sens_3496,
        );
        for c in 0..self.cfg.nch {
            self.ch[c].phase.copy_from_slice(&dst[c]);
        }
        self.sync_weight_buf[..nb].copy_from_slice(&weight[..nb]);
        if let Some(t0) = t0 { profile_add(|p| &mut p.sync, t0); }
    }

    /// `_ev_rpt` — `ResetPhasesForTransients`.
    fn ev_rpt(&mut self, b: usize) {
        let nb = self.cfg.nb_bins;
        let mut phase = std::mem::take(&mut self.ch[b].phase);
        let mut mask_table = std::mem::take(&mut self.mask_table_8f8);
        reset_phases_for_transients(
            self.transient_flag_1208 as i32,
            self.transient_state_2488 as i32,
            self.ratio,
            nb as i64,
            0,
            0,
            self.cfg.sr,
            self.granule_1400 as i64,
            self.hop_1420 as i64,
            self.cfg.n_fft as i64,
            nb as i64,
            0.0,
            0.0,
            0.0,
            false,
            &self.ch[b].mag,
            &self.b_710,
            &self.ch[b].mask,
            &self.reg_start,
            &self.reg_end,
            &self.peak_bins,
            &mut phase,
            &mut mask_table,
            nb,
        );
        self.ch[b].phase = phase;
        self.mask_table_8f8 = mask_table;
    }

    /// `_ev_afc` — `ApplyFormantCorrection`.
    fn ev_afc(&mut self, b: usize) {
        let t0 = profile_enabled().then(Instant::now);
        self.formant.cfg.f580 = if self.prev_granule_1408 > 0 {
            self.prev_granule_1408
        } else {
            self.cfg.step_base as i32
        };
        let mag = std::mem::take(&mut self.ch[b].mag);
        let mut mag = mag;
        self.formant.apply(&mut mag, b);
        self.ch[b].mag = mag;
        if let Some(t0) = t0 { profile_add(|p| &mut p.formant, t0); }
    }

    /// `_ev_fold_iir_sub` — fftshift + forward/backward IIR + subtract.
    fn ev_fold_iir_sub(&mut self) {
        let t0 = profile_enabled().then(Instant::now);
        let n = self.cfg.n_fft;
        let hop = self.hop_1420 as usize;
        let half = n >> 1;
        let c = 1.0f32 - FOLD_ALPHA;
        {
            let src = &self.acc;
            let (w1, w2) = (&mut self.win1, &mut self.win2);
            w1[..half].copy_from_slice(&src[half..]);
            w1[half..].copy_from_slice(&src[..half]);
            w2[..half].copy_from_slice(&src[half..]);
            w2[half..].copy_from_slice(&src[..half]);
        }
        // forward recurrence (scipy lfilter with zi = [c*w1[0]])
        let (w1, w2) = (&mut self.win1, &mut self.win2);
        let mut prev = FOLD_ALPHA * w1[0] + c * w1[0];
        w1[0] = prev;
        for i in 1..n {
            prev = FOLD_ALPHA * w1[i] + c * prev;
            w1[i] = prev;
        }
        // backward recurrence from N-2 down to half-hop
        let stop = half - hop;
        let mut prev = {
            let xn = w1[n - 1];
            FOLD_ALPHA * xn + c * xn
        };
        w1[n - 1] = prev;
        for i in (stop..n - 1).rev() {
            prev = FOLD_ALPHA * w1[i] + c * prev;
            w1[i] = prev;
        }
        for (dst, src) in w2[stop..stop + 2 * hop]
            .iter_mut()
            .zip(w1[stop..stop + 2 * hop].iter())
        {
            *dst -= *src;
        }
        if let Some(t0) = t0 { profile_add(|p| &mut p.fold_ola, t0); }
    }

    /// `_ev_oac` — `OverlapAddChannel`.
    fn ev_oac(&mut self, ch: usize) {
        let sched_now = vc_calc_sched(self.g_index, self.ratio, self.cfg.step_base);
        let sched_next = vc_calc_sched(self.g_index + 1, self.ratio, self.cfg.step_base);
        let mut a3 = (sched_next.0 - sched_now.0) as i32;
        if a3 <= 0 {
            a3 = self.step_1412;
        }
        let f1412 = self.step_1412.max(0) as i64;
        let p = self.cfg.precision;
        let a_const = if self.cfg.sr == 44100 {
            3300i64
        } else {
            3592i64
        };
        let ring_len = self.cfg.ring_out_len as i64;
        let f1224 = self.pos_1384 - a3 as i64 + self.hop_1420 as i64;
        let nb = self.cfg.ring_out_len;
        let base = ch * nb;
        let out_slice = &mut self.out_ring[base..base + nb];
        overlap_add_channel_in_place(
            &mut self.win1,
            &mut self.win2,
            &self.synth_a,
            &self.synth_b,
            out_slice,
            &self.edge_gain,
            ch,
            a3,
            0,
            0,
            self.cfg.n_fft as i64,
            p,
            a_const,
            self.v8_len_a as i64,
            self.v9_len_b as i64,
            f1412,
            self.pos_1384,
            self.hop_1420 as i64,
            ring_len,
            f1224,
            self.cfg.n_write as i64,
        );
    }

    /// `_ev_assembly` — RPT → Copy1 → Copy2 → AFC → P2C → FFTInv → fold → OAC.
    fn ev_assembly(&mut self) {
        let t0 = profile_enabled().then(Instant::now);
        for r in 0..self.cfg.nch {
            self.ev_rpt(r);
            let nb = self.cfg.nb_bins;
            {
                let c = &mut self.ch[r];
                c.mask_copy[..nb].copy_from_slice(&c.mask[..nb]);
                c.mag_copy[..nb].copy_from_slice(&c.mag[..nb]);
            }
            let t_phase = profile_enabled().then(Instant::now);
            self.ev_afc(r);
            let n = nb;
            {
                let (mag, phase) = (&self.ch[r].mag, &self.ch[r].phase);
                let cart = &mut self.cart;
                for k in 0..n {
                    let ph = phase[k];
                    if fast_math_enabled() {
                        let (s, c) = ph.sin_cos();
                        cart[2 * k] = mag[k] * c;
                        cart[2 * k + 1] = mag[k] * s;
                    } else {
                        let (s, c) = (ph as f64).sin_cos();
                        cart[2 * k] = mag[k] * (c as f32);
                        cart[2 * k + 1] = mag[k] * (s as f32);
                    }
                }
            }
            if let Some(t_phase) = t_phase { profile_add(|p| &mut p.phase_cart, t_phase); }
            let t_inv = profile_enabled().then(Instant::now);
            Self::fft_inv_into(self.cfg.n_fft, &self.cart, &mut self.frame);
            if let Some(t_inv) = t_inv { profile_add(|p| &mut p.inverse_fft, t_inv); }
            // `acc` is the inverse transform the fold/OLA stages consume; the
            // engine keeps it in a separate frame buffer (`+0x5E0`) as well.
            self.acc.copy_from_slice(&self.frame);
            self.ev_fold_iir_sub();
            self.ev_oac(r);
        }
        if let Some(t0) = t0 { profile_add(|p| &mut p.assembly, t0); }
    }

    /// `_ev_cursor_advance` — the LABEL_144 cursor advance.
    fn ev_cursor_advance(&mut self) {
        let pos_old = self.pos_1384;
        if self.sched_cdel >= 0 {
            self.cursor_1376 += self.sched_cdel;
            self.transient_flag_1208 = 0;
            self.out_total += self.sched_cdel;
            self.progress.store(
                self.out_total.max(0) as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            self.sched_cdel = -1;
            return;
        }
        self.writepos_1224 = self.pos_1384 + self.hop_1420 as i64;
        self.cursor_1376 += self.granule_1400 as i64;
        let v158 = if self.out_step_1404 > 1 {
            self.out_step_1404
        } else {
            1
        };
        let v157 = if self.granule_1400 > 1 {
            self.granule_1400
        } else {
            1
        };
        let v9 = if v158 == self.out_step_1404 {
            self.comp_1432
        } else {
            v158 as f64
        };
        self.acc_1440 += v9;
        self.pos_1384 = (self.acc_1440 + if self.acc_1440 < 0.0 { -0.5 } else { 0.5 }) as i64;
        self.step_1412 = (self.pos_1384 - pos_old) as i32;
        self.prev_granule_1408 = v157;
        self.transient_flag_1208 = 0;
        self.out_total += v157 as i64;
        self.progress.store(
            self.out_total.max(0) as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    /// `process_granule` — schedule, warm-up g0..g7, 4-phase chain.
    pub fn process_granule(&mut self) {
        if profile_enabled() {
            VC_PROFILE.with(|p| p.borrow_mut().granules += 1);
        }
        let se_curr = vc_calc_sched(self.g_index, self.ratio, self.cfg.step_base);
        let se_next = vc_calc_sched(self.g_index + 1, self.ratio, self.cfg.step_base);
        self.pos_1384 = se_curr.0;
        self.writepos_1224 = self.pos_1384 + self.hop_1420 as i64;
        self.sched_cdel = se_curr.1;
        if se_curr.1 > 0 {
            self.granule_1400 = se_curr.1 as i32;
        }
        if se_curr.2 > 0 {
            self.prev_granule_1408 = se_curr.2 as i32;
        }
        if se_curr.3 > 0 {
            self.step_1412 = se_curr.3 as i32;
        }
        self.out_step_1404 = (se_next.0 - se_curr.0) as i32;
        let ph = self.g_index & 3;
        let nch = self.cfg.nch;
        let ch1 = 1 % nch;

        if self.g_index < 8 {
            // warm-up: g0 empty | g1 FillG | g2 FillG+ACS(ch0) | g3 FillG+ACS(ch1)
            // g4 FillG+Sync+assembly(no Unwrap) | g5 FillG | g6 FillG+ACS(ch0)
            // g7 FillG+Unwrap+APC+ACS(ch1); steady state starts at g8
            let g = self.g_index;
            if g >= 1 {
                self.ev_fill_granule(0);
            }
            if g == 2 || g == 3 || g == 6 {
                self.ev_acs(if g == 3 { ch1 } else { 0 });
            }
            if g == 4 {
                self.ev_sync();
                self.ev_assembly();
            }
            if g == 7 {
                self.ev_unwrap(0);
                self.ev_apc(0);
                self.ev_pull_to_peak(0);
                self.ev_acs(ch1);
            }
            self.g_index += 1;
            self.sched_cdel = -1;
            if self.trace {
                eprintln!("G {} warmup", self.g_index - 1);
            }
            return;
        }

        match ph {
            1 => self.ev_fill_granule(0),
            2 => {
                self.ev_fill_granule(0);
                self.ev_acs(0);
            }
            3 => {
                self.ev_fill_granule(0);
                self.ev_unwrap(0);
                self.ev_apc(0);
                self.ev_pull_to_peak(0);
                self.ev_acs(ch1);
            }
            _ => {
                self.ev_fill_granule(0);
                self.ev_fgwin_x4(ch1);
                self.ev_unwrap(ch1);
                self.ev_apc(ch1);
                self.ev_pull_to_peak(ch1);
                self.ev_sync();
                self.ev_assembly();
            }
        }
        self.g_index += 1;
        if self.trace {
            eprintln!(
                "G {} ph={} pos={} cursor={} granule={} step={}",
                self.g_index - 1,
                ph,
                self.pos_1384,
                self.cursor_1376,
                self.granule_1400,
                self.step_1412
            );
        }
        if ph == 0 {
            self.sched_cdel = vc_calc_sched(self.g_index, self.ratio, self.cfg.step_base).1;
            self.ev_cursor_advance();
        } else {
            self.sched_cdel = -1;
        }
    }

    /// `feed` — crossover into the band rings, then pump granules.
    ///
    /// # Stereo
    ///
    /// Each input channel is split into bands **separately** and written into its own
    /// analysis ring, so channel `c`'s granule is built from channel `c`'s audio.
    ///
    /// The reference engine does not do this: `rx_vc_feed` calls
    /// `rx_crossover_process1_f(..., 0, src[i * nch], ...)` — always channel 0 — and
    /// scatters those bands into the ring that every channel later reads. The
    /// synthesis side *is* per channel (`ev_assembly` loops `for r in 0..nch` and
    /// `ev_oac` writes `out_ring[r]`), so the result is not literally one mono channel:
    /// it is the left channel's spectrum synthesised twice, with identical magnitudes
    /// and synchronised phases, which a listener hears as mono with a fake wide edge.
    /// The port reproduced that faithfully until this was fixed; on stereo material it
    /// now deliberately differs from the reference, and the parity tests above are
    /// measured on material where the two agree.
    pub fn feed(&mut self, x: &[f32], nin: usize) -> i64 {
        let nch = self.cfg.nch;
        let nbands = self.cfg.nbands;
        if self.xovers.is_empty() {
            self.xovers = (0..nch)
                .map(|_| Crossover::new(self.cfg.sr, nbands))
                .collect();
        }
        // ring scatter: j = (base + i - 1023) mod cap, per channel
        let cap = self.ring_cap;
        let base = self.fed_total as i64;
        for (c, xo) in self.xovers.iter_mut().enumerate() {
            let input: Vec<f32> = (0..nin).map(|i| x[i * nch + c]).collect();
            let t_xo = profile_enabled().then(Instant::now);
            let bands = xo.process(&input);
            if let Some(t0) = t_xo {
                profile_add(|p| &mut p.crossover, t0);
            }
            let t_ring = profile_enabled().then(Instant::now);
            for b in 0..nbands {
                for i in 0..nin {
                    let j = crate::consts::c_mod(base + i as i64 - 1023, cap as i64) as usize;
                    self.ring[(c * nbands + b) * cap + j] = bands[b][i];
                }
            }
            if let Some(t0) = t_ring {
                profile_add(|p| &mut p.ring_scatter, t0);
            }
        }
        self.fed_total += nin as i64;
        while self.cursor_1376 + (self.hop_1420 as i64) < self.fed_total {
            self.process_granule();
        }
        let made = self.out_total - self.out_reported;
        self.out_reported = self.out_total;
        made
    }

    /// `take_output` — drain the settled region of the output ring.
    pub fn take_output(&mut self, dst: &mut [f32], max_frames: usize) -> usize {
        let safe_end = self.pos_1384 - self.hop_1420 as i64 - self.out_step_1404 as i64;
        let mut avail = safe_end - self.out_read_pos;
        if avail > max_frames as i64 {
            avail = max_frames as i64;
        }
        if avail <= 0 {
            return 0;
        }
        let avail = avail as usize;
        let ring = self.cfg.ring_out_len;
        let nch = self.cfg.nch;
        for i in 0..avail {
            let g = (self.out_read_pos + i as i64).rem_euclid(ring as i64) as usize;
            for c in 0..nch {
                dst[i * nch + c] = self.out_ring[c * ring + g];
            }
        }
        self.out_read_pos += avail as i64;
        avail
    }

    /// `vc_render` — the C driver's feed rhythm and resampler drain loop.
    pub fn render(&mut self, x: &[f32]) -> Vec<f32> {
        profile_reset();
        let t_render = profile_enabled().then(Instant::now);
        self.start_streaming();
        let nch = self.cfg.nch;
        let nframes = x.len() / nch;
        let ring_len = self.cfg.ring_out_len as i64;
        let ratio = self.ratio;
        // Progress is reported as output frames made against input frames: the
        // engine is duration preserving, so the two converge.
        self.progress_total = nframes.max(1) as i64;
        self.progress.store(0, std::sync::atomic::Ordering::Relaxed);
        let mut out = vec![0.0f32; nframes * nch];
        let tbl = InterpTable::new(8192, 6, 16.0);
        let mut drain_phase = 0.0f64;
        let mut out_n = 0usize;
        let mut pos = 0usize;
        const CH: usize = 1024;
        let mut chunk = vec![0.0f32; CH * nch];
        while out_n < nframes {
            let n = if pos == 0 { 448 } else { 1024 };
            for v in chunk[..n * nch].iter_mut() {
                *v = 0.0;
            }
            if pos < nframes {
                let avail = (nframes - pos).min(n);
                chunk[..avail * nch].copy_from_slice(&x[pos * nch..(pos + avail) * nch]);
            }
            pos += n;
            self.feed(&chunk, n);
            loop {
                let writepos = if self.pos_1384 > self.hop_1420 as i64 {
                    self.pos_1384 - self.hop_1420 as i64
                } else {
                    0
                };
                let head = (writepos - 100) as f64;
                if head <= drain_phase {
                    break;
                }
                let target_out = (out_n as f64 + (head - drain_phase) / ratio + 0.5) as i64;
                let mut chunk_n = target_out - out_n as i64;
                if chunk_n > 4096 {
                    chunk_n = 4096;
                }
                if out_n as i64 + chunk_n > nframes as i64 {
                    chunk_n = nframes as i64 - out_n as i64;
                }
                if chunk_n <= 0 {
                    break;
                }
                let chunk_n = chunk_n as usize;
                let t_drain = profile_enabled().then(Instant::now);
                let mut ph = drain_phase % ring_len as f64;
                if ph < 0.0 {
                    ph += ring_len as f64;
                }
                let planes: Vec<Vec<f32>> = (0..nch)
                    .map(|c| {
                        let base = c * ring_len as usize;
                        self.out_ring[base..base + ring_len as usize].to_vec()
                    })
                    .collect();
                let plane =
                    interp_nsamples(&tbl, &planes, ring_len, ph, chunk_n, ratio, ratio as f32);
                if let Some(t0) = t_drain {
                    profile_add(|p| &mut p.drain, t0);
                }
                let t_copy = profile_enabled().then(Instant::now);
                for i in 0..chunk_n {
                    for c in 0..nch {
                        out[(out_n + i) * nch + c] = plane[c][i];
                    }
                }
                if let Some(t0) = t_copy {
                    profile_add(|p| &mut p.out_copy, t0);
                }
                out_n += chunk_n;
                drain_phase += chunk_n as f64 * ratio;
                if out_n >= nframes {
                    break;
                }
            }
        }
        if let Some(t0) = t_render {
            profile_add(|p| &mut p.wall, t0);
        }
        profile_report();
        out
    }

    /// Enable per-granule tracing on stderr.
    pub fn set_trace(&mut self, on: bool) {
        self.trace = on;
    }

    /// Progress of the render in flight: `(output frames made, input frames)`.
    ///
    /// The returned counter is shared with the state, so it can be polled from
    /// another thread while [`VocoderState::render`] runs.
    pub fn progress_counter(&self) -> ProgressCounter {
        self.progress.clone()
    }

    /// Install an externally owned progress counter and the expected total.
    ///
    /// Used by the CLI so a watchdog thread can poll the render while it runs.
    pub fn set_progress_counter(&mut self, counter: ProgressCounter, total: i64) {
        self.progress = counter;
        self.progress_total = total.max(1);
    }

    /// Progress as a `(done, total)` pair, for callers on the same thread.
    pub fn progress(&self) -> (i64, i64) {
        (
            self.progress.load(std::sync::atomic::Ordering::Relaxed) as i64,
            self.progress_total.max(1),
        )
    }

    /// Final analysis cursor (diagnostics).
    pub fn cursor(&self) -> i64 {
        self.cursor_1376
    }
    /// Final output write position (diagnostics).
    pub fn write_pos(&self) -> i64 {
        self.pos_1384
    }
    /// Number of granules made (diagnostics).
    pub fn granules_made(&self) -> i64 {
        self.out_total
    }
}
