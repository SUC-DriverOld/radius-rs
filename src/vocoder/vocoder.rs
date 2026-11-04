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
use crate::vocoder::*;
use crate::vocoder::spectral_gate::SpectralGate;
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

/// Number of threads the vocoder may use for per-channel work.
///
/// Defaults to `min(available_cores, 2)`. The cap is 2 because the engine's
/// parallel regions are per **channel** and stereo is the supported case — a third
/// thread would have no work to take. Capping also keeps the acceptance timings
/// reproducible. `RADIUS_THREADS` overrides it: `0` forces the serial path, which
/// is what the profiling path wants (the stage timers are thread-local, so a
/// worker's time would not reach the report) and what the reference measurements
/// compare against.
///
/// The split is exact by construction: each channel computes the operations it
/// would compute alone, in the same order, and writes its own ring plane.
pub(crate) fn thread_count() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        if let Ok(v) = std::env::var("RADIUS_THREADS") {
            match v.trim().to_ascii_lowercase().as_str() {
                "off" | "false" | "no" => return 1,
                other => {
                    if let Ok(n) = other.parse::<usize>() {
                        return n.clamp(1, 2);
                    }
                }
            }
        }
        std::thread::available_parallelism()
            .map(|n| n.get().min(2))
            .unwrap_or(1)
    })
}

/// Whether per-channel work should be split across threads at all.
pub(crate) fn parallel_enabled() -> bool {
    thread_count() > 1
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

/// Supported vocoder sample rates.
///
/// Every rate in `8000..=192000` runs on [`crate::vocoder::rate_gen`]'s
/// banks, designed at init from the rate exactly like the original engine
/// (which contains no per-rate tables at all). The design laws were measured
/// on the reference engine's 44.1/48 kHz data before it was retired; the
/// `rate_generic` tests pin the structural guarantees.
pub fn supported_rate(sr: u32) -> bool {
    (8000..=192_000).contains(&sr)
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
///
/// The two rates of the tempo mechanism meet here (see [`VocoderState::slope`]):
/// the **branch** — whether the analysis step widens by √2 — is selected by the
/// *pitch* ratio, exactly as in the pitch-only reference (that is the table
/// lookup pyradius mirrors, and the pitch corpus pins it bit-exactly), while
/// `out_step` always paces the output ring by the **slope** `pitch × stretch`.
///
/// So at pitch +3 / tempo 50 the schedule is (222, 132) — the plain law — and
/// not the √2 pair pyradius would print for a ratio below 1, because the pitch
/// is still above 1. Measured against the Audition reference render this is the
/// better law by a wide margin (envelope correlation 0.843 vs 0.789, spectrum
/// 0.963 vs 0.956); pacing the ring by the stretch alone instead of the slope
/// collapses it to 0.32.
pub fn vc_calc_sched(g: i64, pitch_ratio: f64, slope: f64, nominal_step: f64) -> (i64, i64, i64, i64) {
    if g < 5 {
        return (0, 0, 0, 0);
    }
    let grp = (g - 1) / 4;
    let (in_step, out_step) = if pitch_ratio >= 1.0 {
        let i = nominal_step.round() as i64;
        (i, (i as f64 * slope).round() as i64)
    } else {
        let i = (nominal_step * std::f64::consts::SQRT_2).round() as i64;
        (i, (i as f64 * slope).round() as i64)
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
    /// Enable the transient-preservation subsystem (input-side onset scoring,
    /// the granule hop scheduler, and the `state == 2` chain gates) recovered
    /// from the AU DLL — see `docs/RE-dvaaudiodsp-radius-tempo.md` §13.
    ///
    /// **Defaults to `true`**: AU's engine runs this machinery at any tempo ≥
    /// 100, so parity with AU requires it. Renders therefore diverge from
    /// pyradius (which never had it) by design; pyradius/libradius are now a
    /// code reference only. The bit-exact pyradius regression tests opt out
    /// explicitly via [`VocoderState::set_transient_sched`]`(false)`.
    pub transient_sched: bool,
    /// Experimental: let RPT take the DLL's B2 noise-floor branch while the
    /// transient machine is in state 2. Our `b_710` reference is a structural
    /// approximation and B2 measurably degrades all three AU references —
    /// default off (B3 trio everywhere). See RE doc §13.4 audit round.
    pub ts_rpt_b2: bool,
    /// Experimental: run the DLL's noise-synthesis excitation loop
    /// (`sub_180158D20` L2050-2660/L3232-3268, RE §13.9) once per machine
    /// granule per channel — loop A (peak-anchored output spectrum), loop B
    /// (excitation evolution + keystream double buffer + the ×10 peak-region
    /// enhancement), loop D (score-weighted band smoothing) and the init-pass
    /// snapshot. The evolved `noise_gain` rows and the §13.9 output spectrum
    /// are then live state; the +3568 tail reset reads the loop's own source
    /// mirror instead of the sub-granule `mag_copy`. Default off — the loop's
    /// output consumer (buffer 2176's reader) sits outside `sub_180158D20`
    /// and several object scalars (base weight envelope, randomize-ring
    /// history fill, `v173`/`v894`) remain undecoded, so the audio effect of
    /// enabling it must be A/B-measured first.
    pub ts_noise_loop: bool,
}

impl VocoderConfig {
    /// The reference driver's `precision` value, fixed.
    ///
    /// This used to be a public knob, but it is not a precision and not a quality
    /// setting: it indexes the overlap-add write-gain table
    /// (`edge_gain[p.clamp(0, 3)]`), so it changed the output *level* and nothing
    /// else, and `3..=9` were byte-identical. It is pinned to the reference default
    /// so the parity corpus keeps matching; use the CLI's `--gain` to change level.
    pub const REFERENCE_PRECISION: i32 = 2;

    pub fn for_rate(sr: u32, nch: usize) -> Self {
        Self::for_rate_with_precision(sr, nch, Self::REFERENCE_PRECISION)
    }

    /// As [`VocoderConfig::for_rate`], with an explicit reference `precision`.
    ///
    /// Exposed for the C ABI, which keeps `rx_vc_init`'s third argument so that
    /// existing callers do not break. Prefer [`VocoderConfig::for_rate`].
    pub fn for_rate_with_precision(sr: u32, nch: usize, precision: i32) -> Self {
        let (n_fft, nb_bins, m_fft, mb_bins, n_write, hop) = match sr {
            // The two reference rates: the engine's exact measured anchors
            // (n_write 7184/6599 and their FFT classes).
            44100 => (8192usize, 4097usize, 2048usize, 1025usize, 6599usize, 3299i32),
            48000 => (16384usize, 8193usize, 4096usize, 2049usize, 7184usize, 3592i32),
            // Every other rate: the DLL's rate-parametric laws, fitted on the
            // two anchors. Time-domain spans scale with the rate (a ~149.7 ms
            // write span, hop = half of it); the FFT geometry is
            // octave-quantised (`n_fft = 16384 << round(log2(sr/48000))`,
            // i.e. 8192 for the 32 kHz class, 32768 for 96 kHz), with the
            // 44.1 kHz class boundary taken from its anchor rather than the
            // rounding (44.1 kHz sits an octave down despite being >0.5 oct).
            _ => {
                let shift = ((sr as f64 / 48000.0).log2().round() as i32).clamp(-2, 2);
                let n_fft = if shift >= 0 {
                    16384usize << shift
                } else {
                    16384usize >> (-shift)
                };
                // Exact DLL law (sub_180157BC0 inputs, live-verified at both
                // anchors): n_write = 3*sr/20 - 16 (48k -> 7184, 44.1k ->
                // 6599; the earlier 7184*sr/48 fit was off by one at 44.1k).
                let n_write = (3.0 * f64::from(sr) / 20.0 - 16.0).round() as usize;
                let hop = (n_write / 2) as i32;
                (
                    n_fft,
                    n_fft / 2 + 1,
                    n_fft / 4,
                    n_fft / 8 + 1,
                    n_write,
                    hop,
                )
            }
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
            transient_sched: true,
            ts_rpt_b2: false,
            ts_noise_loop: false,
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

/// Immutable inputs of the assembly tail (P2C → IFFT → fold → OAC), shared
/// with the persistent worker via `Arc`. Everything here is written only by
/// the `VocoderState` constructor; the per-granule scalars (`g_index`,
/// `ratio`, `slope`, `ts_out_step`, `step_1412`, `pos_1384`, `hop_1420`)
/// travel with each job instead.
struct AssemblyCtx {
    synth_a: Vec<f32>,
    synth_b: Vec<f32>,
    edge_gain: [f32; 4],
    step_base: f64,
    precision: i32,
    nb_bins: usize,
    n_fft: usize,
    n_write: usize,
    ring_out_len: usize,
    v8_len_a: usize,
    v9_len_b: usize,
}

/// One assembly-tail job for the worker: channel `r`'s magnitude and phase
/// buffers (moved, so no copy) plus a copy of its output-ring plane (the OAC
/// accumulates in place, so the plane content matters on the way in too).
struct AssemblyJob {
    mag: Vec<f32>,
    phase: Vec<f32>,
    out_plane: Vec<f32>,
    g_index: i64,
    ratio: f64,
    slope: f64,
    ts_out_step: i32,
    step_1412: i32,
    pos_1384: i64,
    hop_1420: i32,
    ctx: std::sync::Arc<AssemblyCtx>,
}

/// The persistent assembly-tail worker. Created lazily on the first parallel
/// `ev_assembly`; dies when the sender drops (i.e. with the `VocoderState`).
struct AssemblyWorker {
    tx: Option<std::sync::mpsc::Sender<AssemblyJob>>,
    rx: std::sync::mpsc::Receiver<(Vec<f32>, Vec<f32>, Vec<f32>)>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl AssemblyWorker {
    fn spawn() -> Self {
        let (tx, job_rx) = std::sync::mpsc::channel::<AssemblyJob>();
        let (reply_tx, rx) = std::sync::mpsc::channel::<(Vec<f32>, Vec<f32>, Vec<f32>)>();
        let handle = std::thread::Builder::new()
            .name("vc-assembly".into())
            .spawn(move || {
                // Warm per-worker scratch; sized on the first job and kept.
                let mut cart = Vec::new();
                let mut frame = Vec::new();
                let mut acc = Vec::new();
                let mut win1 = Vec::new();
                let mut win2 = Vec::new();
                for job in job_rx {
                    let ctx = &job.ctx;
                    if cart.len() != ctx.n_fft + 2 {
                        cart = vec![0.0; ctx.n_fft + 2];
                        frame = vec![0.0; ctx.n_fft];
                        acc = vec![0.0; ctx.n_fft];
                        win1 = vec![0.0; ctx.n_fft];
                        win2 = vec![0.0; ctx.n_fft];
                    }
                    let mut out_plane = job.out_plane;
                    VocoderState::assembly_tail_compute(
                        ctx,
                        job.g_index,
                        job.ratio,
                        job.slope,
                        job.ts_out_step,
                        job.step_1412,
                        job.pos_1384,
                        job.hop_1420,
                        &job.mag,
                        &job.phase,
                        &mut out_plane,
                        &mut cart,
                        &mut frame,
                        &mut acc,
                        &mut win1,
                        &mut win2,
                    );
                    if reply_tx.send((job.mag, job.phase, out_plane)).is_err() {
                        return;
                    }
                }
            })
            .expect("spawn vc-assembly worker");
        Self { tx: Some(tx), rx, handle: Some(handle) }
    }
}

impl Drop for AssemblyWorker {
    fn drop(&mut self) {
        // Dropping the sender closes the job channel; the worker's `for job
        // in job_rx` loop ends and the thread exits, then we join it.
        self.tx.take();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
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
    /// `renderer+3568`: input frames since the last accepted B3 transient
    /// (advanced by the machine granule hop each scheduler step, cleared when
    /// the throttled tail reset fires — DLL `0x18015e31a` / `0x18015c783`).
    ts_frames_since_transient: i64,
    /// `+0x1352` — always 1 on the `ProcessBuffered` path (the C asserts it);
    /// kept so `FillGranule`'s visibility condition can mirror the engine.
    #[allow(dead_code)]
    buffered_1352: u8,
    g_index: i64,
    fed_total: i64,
    out_total: i64,
    // ---- §4 streaming parameter commit chain ----
    // The DLL stages mid-stream parameter changes in pending slots
    // (flag[296] stretch / flag[297] pitch) and applies them at the top of
    // the processing loop when the commit gate passes: input base == 0 or
    // ≥ 32 frames advanced since the last commit (doc §4). `set_ratio`
    // before the first feed applies immediately (identical to the old
    // behaviour); after it, the change is staged here.
    pending_semis: Option<f64>,
    pending_tempo: Option<f64>,
    /// `fed_total` at the last mid-stream commit — the DLL's `in_base`.
    commit_in_base: i64,
    /// Schedule-continuity anchor — the additive counterpart of the DLL's
    /// `out_base`/`in_base` time-map remap. Applied to `vc_calc_sched`
    /// positions in `process_granule` so a commit never jumps the granule
    /// lattice: after a commit the next granule lands at
    /// `pos_last + out_step_new`. Zero until the first mid-stream commit,
    /// so single-parameter renders stay bit-identical.
    sched_pos_offset: i64,
    /// Output frames produced, for progress reporting.
    ///
    /// Shared rather than owned so a watchdog thread can poll it while the
    /// renderer runs on this thread. Written with `Relaxed` ordering only: a
    /// progress counter, not a synchronisation primitive.
    progress: ProgressCounter,
    /// Final output frames this render expects (set by [`VocoderState::render_to`]).
    progress_total: i64,
    out_reported: i64,
    out_read_pos: i64,
    copy_len_594: usize,
    sched_cdel: i64,
    noise_slot_958: i32,
    // ---- containers ----
    ring: Vec<f32>, // [nch][nbands][ring_cap] — one band ring per channel
    ch: Vec<Channel>,
    /// Analysis-side scratch (`ev_fill_granule`/`ev_acs`): windowed frame in
    /// `acc`, cartesian spectrum in `cart`. The synthesis side moved to the
    /// per-channel `syn_*` buffers (see `ev_assembly`).
    cart: Vec<f32>,
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
    /// The §13.8 cepstral spectral gate (per-renderer instance; per-channel
    /// attractor state inside).
    ts_gate: SpectralGate,
    /// Synthesis-copy scratch for the §13.8 gate (the DLL gates buffer
    /// 1664, a copy of the analysis 1768 — never the analysis itself).
    ts_gate_scratch: Vec<f32>,
    /// The §13.9 noise-synthesis excitation loop (per-channel excitation/
    /// keystream state; runs once per machine granule while `ts_noise_loop`
    /// is on). `None` until first used.
    ts_noise: Option<noise_loop::NoiseLoop>,
    /// Generation counters: `ts_plan_gen` bumps per new machine plan,
    /// `ts_gate_gen` records the last gated generation — together they
    /// give the DLL's once-per-machine-granule gate rate.
    ts_plan_gen: u64,
    ts_gate_gen: u64,
    mask_table_8f8: Vec<f32>,
    sync_weight_buf: Vec<f32>,
    /// Reused scratch for `ev_apc`, so the per-granule work does not reallocate
    /// five spectrum-sized buffers for every channel. Contents are meaningless
    /// between calls; each is cleared and refilled on entry.
    apc_dir_cur: Vec<f32>,
    apc_dir_prev: Vec<f32>,
    apc_env: Vec<f32>,
    apc_region_gain: Vec<f32>,
    /// Reused scratch for `ev_pull_to_peak`.
    pull_rs: Vec<i64>,
    pull_re: Vec<i64>,
    pull_pk: Vec<i64>,
    pull_rg: Vec<f32>,
    peak_count: usize,
    sync_sens_3496: f32,
    pitch_freq_169: f32,
    pitch_metric_168: f32,
    /// Transient subsystem (`transient.rs`): score queue, input-side detector
    /// and the hop scheduler. Only touched when
    /// `cfg.transient_sched` is set; the machine reports the DLL's `state`
    /// (`renderer+2448`) whose `== 2` value feeds the existing
    /// `transient_state_2488` gates.
    tqueue: crate::vocoder::transient::TransientQueue,
    tdetector: crate::vocoder::transient::TransientDetector,
    tsched: crate::vocoder::transient::TransientSched,
    /// The active machine window's plan and its in/out base positions (the
    /// targets the sub-granules chase).
    ts_plan: Option<crate::vocoder::transient::SchedOut>,
    ts_in_base: i64,
    /// Dual-queue emission accumulators (RE §13.6): `ts_cum_out` is the
    /// absolute output position the override path has emitted so far;
    /// `ts_win_in0` the input coordinate where the current plan window
    /// opened. Together they give the two-sided re-anchoring the DLL gets
    /// from its (in_acc, out_pos, rate) triple.
    ts_cum_out: f64,
    ts_win_in0: i64,
    ts_win_out0: f64,
    ts_out_base: i64,
    /// Per-granule cache of the scheduler's output step override (DLL `v853`).
    ts_out_step: i32,
    /// Diagnostic: when true (and `transient_sched` true), the scheduler also
    /// re-times granule output onto detected attacks, as the DLL does. The
    /// state-2 chain gates run either way.
    ///
    /// Default **true** since the dual-queue re-anchor port (RE §13.6): each
    /// plan window's cumulative output chases an ABSOLUTE target
    /// (`ts_win_out0 + n_j`, `n_j = round(in_hop × rate_j)`), so input-lattice
    /// overshoot self-corrects at every window boundary — the DLL's
    /// two-sided (in_acc, out_pos, rate) mechanism. Measured on the AU
    /// corpus: contour 0.6935/0.6398/0.6474 vs gates-only
    /// 0.6407/0.6424/0.6363 — strong wins at tempo 100/50, parity at 200.
    /// (The earlier one-sided ratio-modulation variant that collapsed to
    /// 0.52 is superseded; see RE doc §13.4/§13.6.)
    pub ts_hop_override: bool,
    /// Transient-subsystem diagnostics (tests/logging).
    ts_events: u64,
    ts_state2_granules: u64,
    /// Per-channel synthesis scratch for `ev_assembly`'s tail (polar→cart,
    /// IFFT frame, fold accumulator, fold windows). Channel `r` always uses
    /// index `r`, serial or threaded, so the tail is exact by isolation; the
    /// single-buffer `cart`/`acc` above stay on the analysis side
    /// (`ev_fill_granule`/`ev_acs`), which the granule pump runs one channel
    /// per granule.
    syn_cart: Vec<Vec<f32>>,
    syn_frame: Vec<Vec<f32>>,
    syn_acc: Vec<Vec<f32>>,
    syn_win1: Vec<Vec<f32>>,
    syn_win2: Vec<Vec<f32>>,
    /// Lazily-built immutable context for the assembly tail (the `synth_*`
    /// window tables, `edge_gain` and the geometry constants are written only
    /// by the constructor — verified by grep — so one `Arc` serves the whole
    /// stream; the per-granule scalars ride each job instead).
    assembly_ctx: Option<std::sync::Arc<AssemblyCtx>>,
    /// Persistent worker thread for channel 1's assembly tail. A scoped
    /// thread per granule measured ~1.2 s of pure spawn/join overhead across
    /// a 30 s render (plus the FFT plan rebuilds fixed separately), so the
    /// tail hands off to one long-lived worker via an ownership-transfer
    /// protocol (moved `Vec`s, no large memcpys except the output plane).
    assembly_worker: Option<AssemblyWorker>,
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
    /// The formant operator. Its `cfg` carries the live ratio, which is the pitch
    /// ratio offset by [`VocoderState::set_formant_shift`].
    pub formant: FormantState,
    // ---- ratio ----
    pub ratio: f64,
    pub stretch: f64,
    /// The schedule slope `stretch * pitch_ratio` — the write rate of the output
    /// ring per input frame.
    ///
    /// Reverse-engineered from `dvaaudiodsp.dll`'s vocoder (`RadiusImplVoc`):
    /// the renderer keeps three ratios. The pitch ratio `2^(P/12)` drives the
    /// synthesis interpolation and the final resampler drain, while the *ramp
    /// slope* `stretch * 2^(P/12)` (renderer+0x20, recomputed by `0x157BC0`)
    /// drives how far the output ring advances per input frame. Measured live on
    /// a 200 % / +3 semitone render: the drain phase advances exactly
    /// 1.189207 (= 2^(3/12)) per output frame, while the ring write cursor
    /// advances 2.378414 (= 2 * 2^(3/12)) per input frame, so the finished audio
    /// runs `2.378414 / 1.189207 = 2.0x` as long at the requested pitch.
    ///
    /// Here that means: the granule schedule (`vc_calc_sched`) is generated for
    /// `slope`, so `pos` (the ring write cursor) advances `slope` per input
    /// frame, while the drain in [`VocoderState::render_to`] still reads at
    /// `ratio`. A `stretch` of 1.0 leaves `slope == ratio` bit-exactly, so the
    /// pitch-only path is unchanged.
    pub slope: f64,
    /// Output gain in dB, applied to the finished samples. 0.0 is a no-op.
    pub gain_db: f64,
    /// Formant (spectral envelope) shift in semitones; `0.0` preserves the formants
    /// as the pitch moves, which is the reference behaviour.
    pub formant_semitones: f64,
    pub pitch_ratio: f64,
    pub total_ratio: f64,
    trace: bool,
}

impl VocoderState {
    /// A vocoder for `sr` (any rate in 8000..=192000, see [`supported_rate`])
    /// with `nch` channels.
    pub fn new(sr: u32, nch: usize) -> Self {
        Self::with_config(VocoderConfig::for_rate(sr, nch))
    }

    /// As [`VocoderState::new`], with an explicit reference `precision`.
    ///
    /// Only the C ABI needs this; see [`VocoderConfig::REFERENCE_PRECISION`].
    pub fn with_precision(sr: u32, nch: usize, precision: i32) -> Self {
        Self::with_config(VocoderConfig::for_rate_with_precision(sr, nch, precision))
    }

    /// Multiply the output by `db` decibels, as a plain linear scale.
    ///
    /// Applied to the finished samples, after the engines, so it cannot interact
    /// with the reference arithmetic: `set_gain(0.0)` is a no-op and leaves the
    /// output bit-exact. Returns `self` for chaining.
    pub fn set_gain(&mut self, db: f64) -> &mut Self {
        self.gain_db = db;
        self
    }

    /// The configured output gain in dB.
    pub fn gain_db(&self) -> f64 {
        self.gain_db
    }

    /// How the formants adapt to the pitch shift, in semitones.
    ///
    /// Audition calls this 共振变换 and describes it as determining "how the formant peaks
    /// adapt to the pitch shift": `0` moves the formants and the pitch together, keeping
    /// the timbre and the naturalness; a value above `0` gives a brighter result by
    /// moving the formants up (a male voice sounding female is the usual example), and
    /// below `0` the opposite.
    ///
    /// It works by *offsetting the formant operator's ratio* away from the pitch ratio.
    /// The operator applies a gain curve of `db[round(R * v)] - db[v]`, whose passband
    /// sits a factor `R` above the analysed envelope's; since the vocoder has already
    /// transposed that envelope by the pitch ratio, the envelope the listener finally
    /// hears sits at `f0 * pitch_ratio / R`. Setting
    ///
    /// ```text
    ///   R = pitch_ratio * 2^(-semitones / 12)
    /// ```
    ///
    /// therefore puts it at `f0 * 2^(semitones / 12)`, i.e. exactly `semitones` away
    /// from where it started and independent of the pitch — which is what makes this an
    /// absolute control rather than a second pitch knob. Measured on a 100 Hz harmonic
    /// series with a formant near 4 kHz, `+6` lands 6.0 .. 7.2 semitones up across pitch
    /// `+1/+6/+12` (a spread of 1.2) and `0` stays within 0.8.
    ///
    /// While the two ratios are equal — the `0` default — the envelope rides along with
    /// the pitch exactly and the formants are preserved, which is the reference
    /// behaviour and bit-exact.
    ///
    /// Has no effect where the formant operator is off: `--mode td`, a vocoder run with
    /// no pitch shift, or [`VocoderState::set_preserve_voice`] set to `false`.
    pub fn set_formant_shift(&mut self, semitones: f64) -> &mut Self {
        self.formant_semitones = semitones;
        self.sync_formant_ratio();
        self
    }

    /// The formant shift in semitones.
    pub fn formant_shift(&self) -> f64 {
        self.formant_semitones
    }

    /// Push the formant operator's ratio, which is the pitch ratio offset by the
    /// requested formant shift. Called from `set_ratio` and
    /// [`VocoderState::set_formant_shift`] so the two can never disagree, and so that
    /// either call order gives the same result.
    fn sync_formant_ratio(&mut self) {
        let f = 2.0f64.powf(-self.formant_semitones / 12.0);
        self.formant.cfg.ratio = (self.ratio * f) as f32;
    }

    /// Audition's 音调一致 (pitch coherence), `0.0 .. 4.0`, default `1.0`.
    ///
    /// This is the reference's `trans_sens`: it scales the threshold that decides
    /// whether `ApplyPitchCoherence` rewrites a region's phase (`v133 =
    /// 1.4 * trans_sens * (v13 + 1)`). `0.0` disables the operator entirely — the
    /// reference documents that as the "no pitch coherence" setting.
    ///
    /// **Not exposed on the CLI.** Every signal tried so far renders byte-identically
    /// at every value in range: the stage runs, but no region crosses its threshold.
    /// It is kept here because the reference has the parameter and the wiring is real,
    /// so a future input or a corrected threshold could still use it — but it is not
    /// offered as a knob that appears to do nothing.
    pub fn set_pitch_coherence(&mut self, v: f32) -> &mut Self {
        self.cfg.trans_sens = v;
        self
    }

    /// The configured pitch coherence (the reference's `trans_sens`).
    pub fn pitch_coherence(&self) -> f32 {
        self.cfg.trans_sens
    }

    /// Audition's 保持语音特性 (preserve voice characteristics), default `true`.
    ///
    /// This is the formant operator's `active` flag. With it on — the default, and the
    /// reference's hardcoded value — the spectral envelope stays where it is while the
    /// pitch moves, so a shifted voice still sounds like the same voice. Turning it
    /// off makes the envelope follow the pitch instead, which is what an unpreserved
    /// shift sounds like.
    ///
    /// Measured on a 200 Hz harmonic series with a formant at 2 kHz, shifted +3
    /// semitones: preserved leaves the formant at 1647 Hz (roughly held), unpreserved
    /// moves it to 2172 Hz (following the pitch).
    ///
    /// With this off, [`VocoderState::set_formant_shift`] has nothing to offset and
    /// does nothing. Also has no effect where the operator is already inert: the
    /// time-domain engine, or a vocoder run with no pitch shift.
    pub fn set_preserve_voice(&mut self, on: bool) -> &mut Self {
        self.formant.cfg.active = i32::from(on);
        self
    }

    /// Whether formant preservation is on.
    pub fn preserve_voice(&self) -> bool {
        self.formant.cfg.active == 1
    }

    /// Apply [`VocoderState::gain_db`] in place. Called by the render entry points.
    fn apply_gain(&self, buf: &mut [f32]) {
        if self.gain_db == 0.0 {
            return;
        }
        let m = 10.0f64.powf(self.gain_db / 20.0) as f32;
        for v in buf.iter_mut() {
            *v *= m;
        }
    }

    pub fn with_config(cfg: VocoderConfig) -> Self {
        let cfg = cfg;
        let (nb, n_ch) = (cfg.nb_bins, cfg.nch);
        let ring_cap = 4 * cfg.n_write;
        // All rates run on the generated banks — the DLL computes everything
        // at init from the rate, and so does this engine (rate_gen).
        let win = crate::vocoder::rate_gen::win_table(cfg.n_write);
        let (synth_a, synth_b, v8, v9) = crate::vocoder::rate_gen::synth_windows(cfg.n_write);
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
            win_table: win,
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
            ts_frames_since_transient: 0,
            buffered_1352: 1,
            g_index: 0,
            fed_total: 0,
            out_total: 0,
            pending_semis: None,
            pending_tempo: None,
            commit_in_base: 0,
            sched_pos_offset: 0,
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
            ts_gate: SpectralGate::new(cfg.n_fft, cfg.nch),
            ts_gate_scratch: Vec::new(),
            ts_noise: None,
            ts_plan_gen: 0,
            ts_gate_gen: 0,
            mask_table_8f8: vec![0.0; nb],
            sync_weight_buf: vec![0.0; nb],
            apc_dir_cur: Vec::with_capacity(nb),
            apc_dir_prev: Vec::with_capacity(nb),
            apc_env: Vec::with_capacity(nb),
            apc_region_gain: Vec::with_capacity(nb),
            pull_rs: Vec::with_capacity(nb),
            pull_re: Vec::with_capacity(nb),
            pull_pk: Vec::with_capacity(nb),
            pull_rg: Vec::with_capacity(nb),
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
            tqueue: crate::vocoder::transient::TransientQueue::new(128, 0),
            tdetector: crate::vocoder::transient::TransientDetector::new(cfg.sr),
            tsched: crate::vocoder::transient::TransientSched::new(),
            ts_plan: None,
            ts_in_base: 0,
            ts_cum_out: 0.0,
            ts_win_in0: 0,
            ts_win_out0: 0.0,
            ts_out_base: 0,
            ts_out_step: 0,
            ts_hop_override: true,
            ts_events: 0,
            ts_state2_granules: 0,
            syn_cart: (0..n_ch).map(|_| vec![0.0; cfg.n_fft + 2]).collect(),
            syn_frame: (0..n_ch).map(|_| vec![0.0; cfg.n_fft]).collect(),
            syn_acc: (0..n_ch).map(|_| vec![0.0; cfg.n_fft]).collect(),
            syn_win1: (0..n_ch).map(|_| vec![0.0; cfg.n_fft]).collect(),
            syn_win2: (0..n_ch).map(|_| vec![0.0; cfg.n_fft]).collect(),
            assembly_ctx: None,
            assembly_worker: None,
            out_ring: vec![0.0; n_ch * cfg.ring_out_len],
            edge_gain: [1.15, 1.30, 1.05, 1.10],
            rng: SimpleRand::new(1),
            xovers: Vec::new(),
            formant,
            ratio: 1.0,
            stretch: 1.0,
            slope: 1.0,
            gain_db: 0.0,
            formant_semitones: 0.0,
            pitch_ratio: 1.0,
            total_ratio: 1.0,
            trace: false,
            cfg,
        }
    }

    /// `set_ratio` — `pitch_chain(semis)` + the driver's `exp2` chain.
    /// Set the pitch shift (semitones) and the time stretch (percent).
    ///
    /// `ratio` stays the pitch control: the synthesis interpolation and the final
    /// resampler drain both run at it. The stretch never touches it — feeding the
    /// tempo into it made `--tempo 200` an octave shift (measured on a 440 Hz tone
    /// at +3 semitones: 523 Hz / 1047 Hz / 262 Hz).
    ///
    /// Instead the stretch scales the **schedule slope** [`Self::slope`] — the rate
    /// the output ring is written at per input frame. This is the division of labour
    /// measured in `dvaaudiodsp.dll`'s vocoder: the ramp slope stored at
    /// renderer+0x20 is `stretch * 2^(P/12)` and only drives the granule positions,
    /// while the renderer's float ratio (+0x1D84) stays at the pitch and drives the
    /// synthesis. A 200 % tempo therefore doubles the duration and leaves the pitch
    /// where `semis` put it.
    pub fn set_ratio(&mut self, semis: f64, tempo: f64) {
        if self.fed_total > 0 {
            // §4: once streaming has started, parameter changes are staged in
            // the pending slots (the DLL's flag[296]/flag[297]) and committed
            // at the next `feed` boundary — see `commit_pending`.
            self.pending_semis = Some(semis);
            self.pending_tempo = Some(tempo);
            return;
        }
        self.apply_ratio(semis, tempo);
    }

    /// Apply the parameters immediately (the pre-stream path of
    /// [`Self::set_ratio`], also used by the §4 commit).
    fn apply_ratio(&mut self, semis: f64, tempo: f64) {
        self.stretch = tempo / 100.0;
        let pr = 2.0f64.powf(semis / 12.0);
        let s = 12.0f32 * (pr as f32).log2();
        self.ratio = 2.0f64.powf(s as f64 / 12.0);
        self.pitch_ratio = self.ratio;
        self.total_ratio = self.ratio;
        self.cfg.total_ratio = self.ratio;
        // The schedule slope (see the field doc): the DLL's renderer+0x20, which
        // scales the output-ring write rate and nothing else. At tempo 100 this
        // is exactly `ratio`, so the pitch-only path is unchanged.
        self.slope = self.ratio * self.stretch;
        self.comp_1432 = self.slope * self.hop_1420 as f64;
        // The transient scheduler's geometry (`renderer+2452` etc.) from
        // sub_180157BC0 — the machine's own ~15 ms granule, not the schedule
        // hop.
        self.tsched.set_geometry(self.cfg.n_write as i64, self.slope, 37);
        // Keep the formant operator in step: its ratio is the pitch ratio offset by
        // the formant shift, and it must follow the pitch as well.
        self.sync_formant_ratio();
    }

    /// The §4 streaming commit, run at the top of every [`Self::feed`] (the
    /// DLL does it at the head of its processing loop):
    ///
    /// ```text
    /// if (in_base == 0 || fed_total - in_base >= 32):
    ///     stretch slot: out_base += (cur_in - in_base) * stretch_old
    ///     pitch  slot: out_base += (cur_in - in_base) * stretch  // Δin == 0 here
    /// ```
    ///
    /// In table coordinates the remap is one additive anchor on the last
    /// processed group: after a commit,
    /// `pos(g) = pos_last + (grp(g) - grp_prev) * out_step_new`, i.e.
    /// `sched_pos_offset = pos_last - grp_prev * out_step_new`
    /// (`pos(g) = grp(g)*out_step + offset`; `grp_prev` is the group of the
    /// last processed granule). Both slots remapped in one anchor because the
    /// pitch slot's Δin is zero once the stretch slot has reset `in_base`.
    /// `acc_1440` re-anchors to the same lattice so the fallback cursor path
    /// stays consistent.
    fn commit_pending(&mut self) {
        if self.pending_semis.is_none() && self.pending_tempo.is_none() {
            return;
        }
        // §4 gate: commit when the input base is still 0 or ≥ 32 frames have
        // arrived since the last commit.
        if !(self.commit_in_base == 0 || self.fed_total - self.commit_in_base >= 32) {
            return;
        }
        let semis = self.pending_semis.unwrap_or_else(|| {
            12.0 * self.ratio.log2()
        });
        let tempo = self.pending_tempo.unwrap_or(self.stretch * 100.0);
        // Size the new table step with the pending params (mirror of
        // apply_ratio's derivation) so the anchor uses the post-commit pace.
        let pr = 2.0f64.powf(semis / 12.0);
        let s = 12.0f32 * (pr as f32).log2();
        let new_ratio = 2.0f64.powf(s as f64 / 12.0);
        let new_slope = new_ratio * (tempo / 100.0);
        let g_next = self.g_index.max(5);
        let os_new = vc_calc_sched(g_next, new_ratio, new_slope, self.cfg.step_base).3;
        if os_new > 0 {
            // `pos_1384` is constant within each group of 4 sub-granules and
            // advances one `out_step` per group, so the anchor is the last
            // processed group: pos(g) = pos_last + (grp(g) - grp_prev)*os_new.
            let grp_prev = (g_next - 2).max(0) / 4;
            self.sched_pos_offset = self.pos_1384 - grp_prev * os_new;
        }
        // Apply (stretch slot first, then pitch — one call for both, matching
        // the remap algebra above), then D46: a slope change resets a transient
        // machine in state 1/2 to 0 (§13.2). The machine's inpos re-mapping is
        // absorbed by the cursor-driven plan machinery: the next plan anchors
        // at the current `cursor_1376`.
        self.apply_ratio(semis, tempo);
        self.acc_1440 = self.pos_1384 as f64;
        if self.tsched.state == 1 || self.tsched.state == 2 {
            self.tsched.state = 0;
        }
        self.commit_in_base = self.fed_total;
        self.pending_semis = None;
        self.pending_tempo = None;
    }

    /// Enable or disable the transient-preservation subsystem
    /// ([`VocoderConfig::transient_sched`]). Must be called before feeding;
    /// returns `self` for chaining.
    pub fn set_transient_sched(&mut self, on: bool) -> &mut Self {
        self.cfg.transient_sched = on;
        self
    }

    /// Whether the transient-preservation subsystem is enabled.
    pub fn transient_sched(&self) -> bool {
        self.cfg.transient_sched
    }

    /// Transient-subsystem diagnostics: (detected event chunks, granules in
    /// the `state == 2` processing mode). Zero when disabled.
    pub fn transient_stats(&self) -> (u64, u64) {
        (self.ts_events, self.ts_state2_granules)
    }

    /// Enable per-granule scheduler tracing (diagnostics/tests); see
    /// [`Self::transient_trace`].
    pub fn set_transient_trace(&mut self, on: bool) {
        self.tsched.trace_enabled = on;
    }

    /// The scheduler trace: `(inpos, state, query_result)` per granule while
    /// tracing was enabled.
    pub fn transient_trace(&self) -> &[[i64; 3]] {
        &self.tsched.trace
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
        // Transient subsystem: fresh detector/queue/scheduler per stream (the
        // DLL re-runs its state machine per streaming session).
        if self.cfg.transient_sched {
            let trace = self.tsched.trace_enabled;
            self.tsched = crate::vocoder::transient::TransientSched::new();
            self.tsched.trace_enabled = trace;
            self.tsched
                .set_geometry(self.cfg.n_write as i64, self.slope, 37);
            // new_with_sr enables the 0xab3 progressive refinement
            // (sub_1801525E0) in the queue — see transient.rs refine().
            self.tqueue = crate::vocoder::transient::TransientQueue::new_with_sr(128, 0, self.cfg.sr);
            self.tdetector = crate::vocoder::transient::TransientDetector::new(self.cfg.sr);
        }
        self.ts_plan = None;
        self.ts_frames_since_transient = 0;
        self.ts_in_base = 0;
        self.ts_cum_out = 0.0;
        self.ts_win_in0 = 0;
        self.ts_win_out0 = 0.0;
        self.ts_out_base = 0;
        self.ts_out_step = 0;
        self.ts_events = 0;
        self.ts_state2_granules = 0;
        // §4 commit chain: a fresh stream starts with no pending parameters
        // and an un-anchored schedule.
        self.pending_semis = None;
        self.pending_tempo = None;
        self.commit_in_base = 0;
        self.sched_pos_offset = 0;
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
        // 2026-11-04 audit D50: the DLL folds τ with TimeToIirA(τ, sr/hop)
        // where hop = renderer+1376 = the MACHINE hop (≈704 @48k), not our
        // sub-granule step nor the schedule hop.
        let hopf = if self.tsched.steady_hop > 0 {
            self.tsched.steady_hop as f32
        } else if self.step_1412 > 0 {
            // machine inactive (transient_sched off): keep the historical
            // rate basis so the pyradius golden path is unchanged
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
        // The fast per-bin envelope the DLL keeps at `renderer+0x710` (our
        // `b_710`, RPT's B2 noise-floor reference): spectrum #2 smoothed by a
        // one-pole IIR with τ = 0.01 s, switching to 0.005 s in the `state ==
        // 2` transient mode (`sub_180158D20` @ 0x18015c8f0/0x18015d289,
        // `TimeToIirA(τ, sr/hop)`). pyradius left this buffer at zero, which
        // is why its B2 branch never did anything even when entered.
        if self.cfg.transient_sched {
            let tau = if self.transient_state_2488 == 2 { 0.005 } else { 0.01 };
            // Audit D50: same rate-basis fix — sr/machine-hop, which turns
            // a≈0.999 (no smoothing, B2 fed noise) into a≈0.77 like the DLL.
            let hopf = if self.tsched.steady_hop > 0 {
                self.tsched.steady_hop as f32
            } else {
                self.hop_1420 as f32
            };
            let a = time_to_iir_a(tau, self.cfg.sr as f32 / hopf);
            let mag = &self.ch[ch].mag;
            for k in 0..nb {
                self.b_710[k] = fma(mag[k] - self.b_710[k], a, self.b_710[k]);
            }
        }
        // (The §13.8 gate is no longer applied here: the DLL scores and
        // smooths on the UNgated analysis spectrum (a1+1768) and gates only
        // the synthesis copy (a1+1664) at the polar->cart boundary — see
        // `ev_assembly`. Gating the persistent `mag` here contaminated every
        // analysis consumer, which is what the first A/B measured.)
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
        let mut phase_mod = std::mem::take(&mut self.ch[ch].phase);
        let n = self.cfg.nb_bins;
        let mut dir_cur = std::mem::take(&mut self.apc_dir_cur);
        let mut dir_prev = std::mem::take(&mut self.apc_dir_prev);
        dir_cur.clear();
        dir_cur.extend_from_slice(&self.dir_cur[ch * n..(ch + 1) * n]);
        dir_prev.clear();
        dir_prev.extend_from_slice(&self.dir_prev[ch * n..(ch + 1) * n]);
        let mut env = std::mem::take(&mut self.apc_env);
        env.clear();
        env.extend_from_slice(self.env_of(ch));
        let mut region_gain = std::mem::take(&mut self.apc_region_gain);
        region_gain.clear();
        region_gain.extend_from_slice(self.region_gain_of(ch));
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
        self.apc_env = env;
        self.apc_region_gain = region_gain;
        self.ch[ch].phase = phase_mod;
        self.dir_cur[ch * n..(ch + 1) * n].copy_from_slice(&dir_cur);
        self.dir_prev[ch * n..(ch + 1) * n].copy_from_slice(&dir_prev);
        self.apc_dir_cur = dir_cur;
        self.apc_dir_prev = dir_prev;
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
        // Take the peak/region io out of the state instead of reallocating it on
        // every granule. `ampd_pull_to_peak` writes through `&mut`, so it works
        // straight on these buffers; the big three (32/26/16 KB) used to be a
        // `to_vec()` per channel per granule, which is tens of thousands of
        // allocations over a render.
        let mut rs = std::mem::take(&mut self.pull_rs);
        let mut re = std::mem::take(&mut self.pull_re);
        let mut pk = std::mem::take(&mut self.pull_pk);
        rs.clear();
        rs.extend_from_slice(&self.reg_start[..cnt]);
        re.clear();
        re.extend_from_slice(&self.reg_end[..cnt]);
        pk.clear();
        pk.extend_from_slice(&self.peak_bins[..cnt]);
        let mut rg = std::mem::take(&mut self.pull_rg);
        rg.clear();
        rg.extend_from_slice(self.region_gain_of(ch));
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
        self.pull_rs = rs;
        self.pull_re = re;
        self.pull_pk = pk;
        self.pull_rg = rg;
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
        // NOTE: these three clones are deliberate, not an oversight. An in-place
        // rewrite of `synchronize_stereo_phases` was written to avoid them and
        // twice produced output that no longer matched the parity corpus — the
        // operator's `dst` parameter is not just a shape hint, it initialises the
        // output rows, so `mag`/`mask`/`phase` each play a distinct role and
        // conflating two of them changes results. The allocation is worth keeping
        // over a rewrite that is this easy to get subtly wrong.
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
        // The B3 average trio: the DLL takes the transient queue's score RMS
        // at `inpos−hop / inpos / inpos+hop` (`sub_180153A70`); pyradius had
        // no queue and passed zeros, killing the branch's local-peak test.
        let (avg_m1, avg_0, avg_p1) = if self.cfg.transient_sched {
            // Audit D59: the DLL's trio windows are ±machine-hop (≈704),
            // not the 3592-frame schedule hop — a 28-chunk window washed
            // out the local-peak test.
            let h: i64 = if self.tsched.steady_hop > 0 {
                self.tsched.steady_hop
            } else {
                self.hop_1420 as i64
            };
            let p = self.cursor_1376;
            (
                self.tqueue.rms_around(p - h, h),
                self.tqueue.rms_around(p, h),
                self.tqueue.rms_around(p + h, h),
            )
        } else {
            (0.0, 0.0, 0.0)
        };
        let mut phase = std::mem::take(&mut self.ch[b].phase);
        let mut mask_table = std::mem::take(&mut self.mask_table_8f8);
        // Measured decision (2026-11-04 A/B): the DLL switches RPT to the B2
        // noise-floor branch in state 2, but our b_710 is a structural
        // approximation of its reference and B2 measurably degrades all three
        // AU references (0.48/0.41/0.30 vs 0.64 with B3). Default to the B3
        // trio everywhere; opt into B2 for experiments.
        let rpt_state = if self.cfg.ts_rpt_b2 {
            self.transient_state_2488 as i32
        } else {
            0
        };
        let reset = reset_phases_for_transients(
            self.transient_flag_1208 as i32,
            rpt_state,
            self.ratio,
            nb as i64,
            0,
            0,
            self.cfg.sr,
            self.granule_1400 as i64,
            self.hop_1420 as i64,
            self.cfg.n_fft as i64,
            nb as i64,
            avg_m1,
            avg_0,
            avg_p1,
            false,
            self.ts_frames_since_transient,
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
        if let Some(start) = reset {
            // The B3 tail reset fired (accepted transient, ≥ sr*0.003 frames
            // since the last one): the DLL also snaps the noise working
            // buffer's tail from its per-bin analysis reference
            // (`renderer+2080[ch][start..] := renderer+1872[ch][start..]`,
            // `0x18015c755`) and clears the frame counter (`0x18015c783`).
            // With the §13.9 loop running, the excitation state lives in the
            // loop's own 2080 mirror (evolved per machine granule and copied
            // into `noise_gain`); the reset then writes through the loop so
            // the loop's keystream/flags stay coherent with its excitation.
            // The source side: 1872 = the mag_copy refresh of 1976 (ch.mag),
            // pinned this round — `mag_copy` IS the true reference.
            let base = b * nb;
            if let Some(nl) = self.ts_noise.as_mut() {
                let (dst, src) = (nl.exc_mut(b), &self.ch[b].mag_copy);
                dst[start as usize..nb].copy_from_slice(&src[start as usize..nb]);
                self.noise_gain[base..base + nb].copy_from_slice(&nl.exc(b)[..nb]);
            } else {
                let (dst, src) = (
                    &mut self.noise_gain[base..base + nb],
                    &self.ch[b].mag_copy,
                );
                dst[start as usize..nb].copy_from_slice(&src[start as usize..nb]);
            }
            self.ts_frames_since_transient = 0;
        }
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
    ///
    /// Buffers are explicit so the assembly tail can run per channel on its
    /// own scratch (see `ev_assembly`); the math is a pure function of `acc`.
    fn fold_iir_sub(acc: &[f32], win1: &mut [f32], win2: &mut [f32], n: usize, hop: usize) {
        let t0 = profile_enabled().then(Instant::now);
        let half = n >> 1;
        let c = 1.0f32 - FOLD_ALPHA;
        {
            win1[..half].copy_from_slice(&acc[half..]);
            win1[half..].copy_from_slice(&acc[..half]);
            win2[..half].copy_from_slice(&acc[half..]);
            win2[half..].copy_from_slice(&acc[..half]);
        }
        // forward recurrence (scipy lfilter with zi = [c*w1[0]])
        let mut prev = FOLD_ALPHA * win1[0] + c * win1[0];
        win1[0] = prev;
        for i in 1..n {
            prev = FOLD_ALPHA * win1[i] + c * prev;
            win1[i] = prev;
        }
        // backward recurrence from N-2 down to half-hop
        let stop = half - hop;
        let mut prev = {
            let xn = win1[n - 1];
            FOLD_ALPHA * xn + c * xn
        };
        win1[n - 1] = prev;
        for i in (stop..n - 1).rev() {
            prev = FOLD_ALPHA * win1[i] + c * prev;
            win1[i] = prev;
        }
        for (dst, src) in win2[stop..stop + 2 * hop]
            .iter_mut()
            .zip(win1[stop..stop + 2 * hop].iter())
        {
            *dst -= *src;
        }
        if let Some(t0) = t0 { profile_add(|p| &mut p.fold_ola, t0); }
    }

    /// `_ev_oac` — `OverlapAddChannel`, on explicit buffers so the assembly
    /// tail can run per channel. Everything it reads is either per-channel or
    /// an immutable scheduling input with the same value for both channels of
    /// one granule (the serial loop observed `g_index` unchanged).
    #[allow(clippy::too_many_arguments)]
    fn oac_channel(
        win1: &mut [f32],
        win2: &mut [f32],
        out_slice: &mut [f32],
        synth_a: &[f32],
        synth_b: &[f32],
        edge_gain: &[f32; 4],
        g_index: i64,
        ratio: f64,
        slope: f64,
        step_base: f64,
        ts_out_step: i32,
        step_1412: i32,
        pos_1384: i64,
        hop_1420: i32,
        precision: i32,
        n_fft: usize,
        n_write: usize,
        ring_out_len: usize,
        v8_len_a: usize,
        v9_len_b: usize,
    ) {
        let sched_now = vc_calc_sched(g_index, ratio, slope, step_base);
        let sched_next = vc_calc_sched(g_index + 1, ratio, slope, step_base);
        let mut a3 = (sched_next.0 - sched_now.0) as i32;
        if a3 <= 0 {
            a3 = step_1412;
        }
        // The transient scheduler's per-granule output step override (DLL
        // `v853`): replaces the table delta while the machine is engaged.
        if ts_out_step > 0 {
            a3 = ts_out_step;
        }
        let f1412 = step_1412.max(0) as i64;
        // `a_const` = the band-1 window length (round(n_write/2); the anchors
        // are 3300 at 44.1 kHz and 3592 at 48 kHz).
        let a_const = (n_write as i64 + 1) / 2;
        let ring_len = ring_out_len as i64;
        let f1224 = pos_1384 - a3 as i64 + hop_1420 as i64;
        overlap_add_channel_in_place(
            win1,
            win2,
            synth_a,
            synth_b,
            out_slice,
            edge_gain,
            0,
            a3,
            0,
            0,
            n_fft as i64,
            precision,
            a_const,
            v8_len_a as i64,
            v9_len_b as i64,
            f1412,
            pos_1384,
            hop_1420 as i64,
            ring_len,
            f1224,
            n_write as i64,
        );
    }

    /// `_ev_assembly` — RPT → Copy1 → Copy2 → AFC → P2C → FFTInv → fold → OAC.
    ///
    /// # Parallel split (exact by construction)
    ///
    /// The per-channel work splits into a serial **prefix** — RPT (its
    /// `ts_frames_since_transient` reset and the shared `mask_table_8f8`
    /// order channels strictly), the analysis copies, AFC (the formant
    /// scratch carries a documented cross-call residue), and the §13.8 gate
    /// (a single `SpectralGate`; the gen-counter law also orders channels) —
    /// and a heavy **tail**: P2C → inverse FFT → acc → fold-IIR → OAC. The
    /// tail's shared state is scratch only (cart/frame/acc/win1/win2 are
    /// fully rewritten before use) plus each channel's own `Channel` and
    /// `out_ring` plane, so channel `r` on its own `syn_*` scratch computes
    /// exactly the operations it would alone, in the same order — the same
    /// argument as the `feed` crossover split. `RADIUS_THREADS=1` (or mono)
    /// runs the same tail function serially.
    ///
    /// Profiling note: the stage timers are thread-local, so in the parallel
    /// path the reported phase_cart/inverse_fft/fold_ola are channel 0's;
    /// the `assembly` timer wraps the whole (joined) region on the main
    /// thread either way.
    fn ev_assembly(&mut self) {
        let t0 = profile_enabled().then(Instant::now);
        let nb = self.cfg.nb_bins;
        // ---- serial prefix, in channel order exactly as the old loop ----
        for r in 0..self.cfg.nch {
            self.ev_rpt(r);
            {
                let c = &mut self.ch[r];
                c.mask_copy[..nb].copy_from_slice(&c.mask[..nb]);
                c.mag_copy[..nb].copy_from_slice(&c.mag[..nb]);
            }
            self.ev_afc(r);
            // The §13.8 cepstral gate, at the DLL's consumer point
            // (0x18015d3a2: spec[k] *= work[k] just before PolarToCart/IFFT).
            // Faithfulness constraints from the L3256-3268 decode:
            //  - gates the SYNTHESIS COPY only (1664 ← 1768): the persistent
            //    analysis mag stays ungated for scoring/b_710/AFC;
            //  - once per MACHINE granule per channel (158D20's own rate),
            //    not per sub-granule — the attractor's τ law assumes it;
            //  - only when a pitch shift is active ([7556] = 2^(P/12) ≠ 1.0).
            let gate_now = self.cfg.transient_sched
                && self.cfg.ts_rpt_b2
                && self.ratio != 1.0
                && self.ts_plan_gen != self.ts_gate_gen;
            if gate_now {
                let hop = if self.tsched.steady_hop > 0 {
                    self.tsched.steady_hop
                } else {
                    self.hop_1420 as i64
                };
                // Score/target split (2026-11-05 live decode): the DLL
                // scores the RAW analysis copy (1664 ← 1768, LABEL_768 —
                // 1768 is the same buffer the 0xab3 peak search reads,
                // i.e. pre-coherence/pre-AFC) and multiplies the gains
                // into the synthesis spectrum (spec[k] *= work[k]). Our
                // pre-AFC snapshot is mag_copy, taken above before ev_afc.
                let mut score = std::mem::take(&mut self.ts_gate_scratch);
                score.resize(nb, 0.0);
                score.copy_from_slice(&self.ch[r].mag_copy[..nb]);
                let st2 = self.transient_state_2488 == 2;
                let pratio = self.ratio as f32;
                let mut target = std::mem::take(&mut self.ch[r].mag);
                self.ts_gate.apply_split(
                    r,
                    &score,
                    &mut target,
                    self.cfg.sr,
                    self.cfg.n_fft,
                    hop,
                    st2,
                    pratio,
                );
                self.ch[r].mag = target;
                self.ts_gate_scratch = score;
                // The old loop updated the gen after channel r's P2C but
                // always before channel r+1's gate_now evaluation; the P2C
                // never reads it, so updating right after the gate pass
                // preserves the observable order.
                self.ts_gate_gen = self.ts_plan_gen;
            }
        }
        // ---- heavy tail: P2C → FFTInv → acc → fold → OAC, per channel ----
        // Shared implementation (also used by the worker): the same
        // operations in the same order as the old serial loop.
        let ctx = self
            .assembly_ctx
            .get_or_insert_with(|| {
                std::sync::Arc::new(AssemblyCtx {
                    synth_a: self.synth_a.clone(),
                    synth_b: self.synth_b.clone(),
                    edge_gain: self.edge_gain,
                    step_base: self.cfg.step_base,
                    precision: self.cfg.precision,
                    nb_bins: self.cfg.nb_bins,
                    n_fft: self.cfg.n_fft,
                    n_write: self.cfg.n_write,
                    ring_out_len: self.cfg.ring_out_len,
                    v8_len_a: self.v8_len_a,
                    v9_len_b: self.v9_len_b,
                })
            })
            .clone();
        // Immutable scheduling inputs shared by both channels; the serial
        // loop observed these unchanged across the r-loop.
        let g_index = self.g_index;
        let ratio = self.ratio;
        let slope = self.slope;
        let ts_out_step = self.ts_out_step;
        let step_1412 = self.step_1412;
        let pos_1384 = self.pos_1384;
        let hop_1420 = self.hop_1420;
        let ring_out_len = self.cfg.ring_out_len;
        let nch = self.cfg.nch;

        if parallel_enabled() && nch > 1 {
            // Hand channel 1's tail to the persistent worker: mag/phase are
            // MOVED (no copy; restored on reply), the output plane rides a
            // copy because the flat `out_ring` cannot split ownership.
            let worker = self
                .assembly_worker
                .get_or_insert_with(AssemblyWorker::spawn);
            let base1 = ring_out_len;
            let mag1 = std::mem::take(&mut self.ch[1].mag);
            let phase1 = std::mem::take(&mut self.ch[1].phase);
            let plane1 = self.out_ring[base1..2 * base1].to_vec();
            worker
                .tx
                .as_ref()
                .expect("assembly worker sender")
                .send(AssemblyJob {
                    mag: mag1,
                    phase: phase1,
                    out_plane: plane1,
                    g_index,
                    ratio,
                    slope,
                    ts_out_step,
                    step_1412,
                    pos_1384,
                    hop_1420,
                    ctx: ctx.clone(),
                })
                .expect("assembly worker alive");
            // Channel 0 runs inline on this thread (its FFT plans are warm
            // from the analysis chain).
            let out0 = &mut self.out_ring[..ring_out_len];
            Self::assembly_tail_compute(
                &ctx,
                g_index,
                ratio,
                slope,
                ts_out_step,
                step_1412,
                pos_1384,
                hop_1420,
                &self.ch[0].mag,
                &self.ch[0].phase,
                out0,
                &mut self.syn_cart[0],
                &mut self.syn_frame[0],
                &mut self.syn_acc[0],
                &mut self.syn_win1[0],
                &mut self.syn_win2[0],
            );
            // Reap the worker's reply and restore the moved buffers.
            let (mag, phase, plane) = worker.rx.recv().expect("assembly worker reply");
            self.ch[1].mag = mag;
            self.ch[1].phase = phase;
            self.out_ring[base1..2 * base1].copy_from_slice(&plane);
        } else {
            for r in 0..nch {
                let (s, e) = (r * ring_out_len, (r + 1) * ring_out_len);
                let out = &mut self.out_ring[s..e];
                Self::assembly_tail_compute(
                    &ctx,
                    g_index,
                    ratio,
                    slope,
                    ts_out_step,
                    step_1412,
                    pos_1384,
                    hop_1420,
                    &self.ch[r].mag,
                    &self.ch[r].phase,
                    out,
                    &mut self.syn_cart[r],
                    &mut self.syn_frame[r],
                    &mut self.syn_acc[r],
                    &mut self.syn_win1[r],
                    &mut self.syn_win2[r],
                );
            }
        }
        if let Some(t0) = t0 { profile_add(|p| &mut p.assembly, t0); }
    }

    /// The assembly tail on explicit buffers: P2C → inverse FFT → acc copy →
    /// fold-IIR → OAC. Pure with respect to its arguments, which is what
    /// makes the worker split exact — channel `r` computes the operations it
    /// would alone, in the same order.
    #[allow(clippy::too_many_arguments)]
    fn assembly_tail_compute(
        ctx: &AssemblyCtx,
        g_index: i64,
        ratio: f64,
        slope: f64,
        ts_out_step: i32,
        step_1412: i32,
        pos_1384: i64,
        hop_1420: i32,
        mag: &[f32],
        phase: &[f32],
        out_plane: &mut [f32],
        cart: &mut [f32],
        frame: &mut [f32],
        acc: &mut [f32],
        win1: &mut [f32],
        win2: &mut [f32],
    ) {
        let nb = ctx.nb_bins;
        let n = nb;
        let t_phase = profile_enabled().then(Instant::now);
        {
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
        Self::fft_inv_into(ctx.n_fft, cart, frame);
        if let Some(t_inv) = t_inv { profile_add(|p| &mut p.inverse_fft, t_inv); }
        // `acc` is the inverse transform the fold/OLA stages consume; the
        // engine keeps it in a separate frame buffer (`+0x5E0`) as well.
        acc.copy_from_slice(frame);
        Self::fold_iir_sub(acc, win1, win2, ctx.n_fft, hop_1420 as usize);
        Self::oac_channel(
            win1,
            win2,
            out_plane,
            &ctx.synth_a,
            &ctx.synth_b,
            &ctx.edge_gain,
            g_index,
            ratio,
            slope,
            ctx.step_base,
            ts_out_step,
            step_1412,
            pos_1384,
            hop_1420,
            ctx.precision,
            ctx.n_fft,
            ctx.n_write,
            ctx.ring_out_len,
            ctx.v8_len_a,
            ctx.v9_len_b,
        );
    }

    /// `_ev_cursor_advance` — the LABEL_144 cursor advance.
    fn ev_cursor_advance(&mut self) {
        let pos_old = self.pos_1384;
        if self.sched_cdel >= 0 {
            self.cursor_1376 += self.sched_cdel;
            self.transient_flag_1208 = 0;
            self.out_total += self.sched_cdel;
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
    }

    /// `process_granule` — schedule, warm-up g0..g7, 4-phase chain.
    pub fn process_granule(&mut self) {
        if profile_enabled() {
            VC_PROFILE.with(|p| p.borrow_mut().granules += 1);
        }
        let se_curr = vc_calc_sched(self.g_index, self.ratio, self.slope, self.cfg.step_base);
        let se_next = vc_calc_sched(self.g_index + 1, self.ratio, self.slope, self.cfg.step_base);
        // §4 commit anchor: keeps the granule lattice continuous across
        // mid-stream parameter commits (zero until one happens — see
        // `commit_pending`).
        self.pos_1384 = se_curr.0 + self.sched_pos_offset;
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
        // Transient hop scheduler (`sub_180158D20`'s state machine). The DLL
        // machine runs on its OWN ~15 ms granule (`renderer+1368` ≈ 705 frames
        // at 48 kHz), i.e. roughly every 3 of our 222-frame sub-granules —
        // drive it from the cursor, not from a fixed sub-granule count: a new
        // plan starts whenever the previous one's input span is consumed, and
        // each sub-granule chases the plan's in/out targets exactly.
        if self.cfg.transient_sched && self.g_index >= 8 {
            let sched = &mut self.tsched;
            let mut plan = self.ts_plan;
            let (cursor, pos) = (self.cursor_1376, self.pos_1384);
            if let Some(p) = plan {
                if cursor >= self.ts_in_base + p.in_hop {
                    plan = None; // plan fully consumed — machine re-steps below
                }
            }
            if plan.is_none() {
                let out = sched.step(
                    &self.tqueue,
                    cursor,
                    pos,
                    self.slope,
                    self.cfg.sr as i64,
                    sched.steady_hop,
                    sched.steady_step,
                );
                if out.state2 {
                    self.ts_state2_granules += 1;
                }
                // `renderer+3568 += hop` at the machine-granule exit
                // (`0x18015e31a`): the counter tracks input frames since the
                // last accepted transient.
                self.ts_frames_since_transient += out.in_hop;
                // §13.2 second clear site: state-2 half-hold resets the same
                // counter (one-shot pulse from the machine).
                if out.mid_hold_clear {
                    self.ts_frames_since_transient = 0;
                }
                self.ts_in_base = cursor;
                self.ts_win_in0 = cursor;
                self.ts_win_out0 = self.ts_cum_out;
                self.ts_out_base = pos;
                plan = Some(out);
                self.ts_plan_gen += 1;
                // The §13.9 noise-synthesis loop runs once per MACHINE
                // granule per channel (the DLL's own rate, after the state
                // machine inside the same channel iteration). Bands = the
                // current peak regions; the score is the covering chunk's
                // refined score, broadcast per bin (the DLL's 3248 buffer is
                // per-bin; ours is per-time-chunk).
                if self.cfg.transient_sched && self.cfg.ts_noise_loop {
                    let nb = self.cfg.nb_bins;
                    let stride = self.tqueue.stride().max(1);
                    let chunk = cursor / stride;
                    let score = self.tqueue.score_at(chunk);
                    let cnt = self.peak_count.min(self.reg_start.len());
                    let bands: Vec<noise_loop::NoiseBand> = (0..cnt)
                        .map(|k| {
                            let s = (self.reg_start[k].max(0) as usize).min(nb);
                            let e = (self.reg_end[k].max(0) as usize).min(nb);
                            noise_loop::NoiseBand { start: s, end: e.max(s), ref_bin: s }
                        })
                        .collect();
                    if self.ts_noise.is_none() {
                        self.ts_noise = Some(noise_loop::NoiseLoop::new(
                            self.cfg.nch,
                            nb,
                        ));
                    }
                    let state = out.state;
                    let (hop_m, step_m) = (out.in_hop, out.out_step);
                    let n_fft = self.cfg.n_fft as i32;
                    let sr = self.cfg.sr;
                    let stretch = self.stretch;
                    let mut snap_mag: Vec<Vec<f32>> = Vec::new();
                    let mut snap_copy: Vec<Vec<f32>> = Vec::new();
                    for r in 0..self.cfg.nch {
                        snap_mag.push(self.ch[r].mag[..nb].to_vec());
                        snap_copy.push(self.ch[r].mag_copy[..nb].to_vec());
                    }
                    let nl = self.ts_noise.as_mut().unwrap();
                    for r in 0..self.cfg.nch {
                        let c = noise_loop::NoiseLoopCtx {
                            src: &snap_copy[r],
                            mag: &snap_mag[r],
                            score,
                            flatness: 0.5,
                            bands: &bands,
                            hop: hop_m as i32,
                            out_step: step_m as i32,
                            n_fft,
                            sr,
                            state,
                            time_base: 1.0,
                            sigma: 1.0,
                            stretch,
                        };
                        nl.apply(r, &c);
                        // Mirror the evolved excitation into the engine's
                        // noise_gain rows (the DLL's 2080 is the same object
                        // the +3568 reset and the noise operators touch).
                        let base = r * nb;
                        self.noise_gain[base..base + nb]
                            .copy_from_slice(&nl.exc(r)[..nb]);
                    }
                }
            }
            if let Some(p) = plan {
                if self.ts_hop_override {
                    // Dual-queue emission (RE §13.6): per plan WINDOW the
                    // output target is n_j = rint(span_in × rate_j) with
                    // rate_j = out_step/in_hop, anchored ABSOLUTELY — every
                    // sub-granule chases the window's cumulative-out endpoint
                    // (ts_win_out0 + n_j × frac), so input-lattice overshoot
                    // self-corrects into the next window instead of drifting
                    // (the failure mode of the earlier one-sided variant).
                    let rate_j = if p.in_hop > 0 {
                        p.out_step as f64 / p.in_hop as f64
                    } else {
                        self.slope
                    };
                    let span = (cursor + self.granule_1400 as i64 - self.ts_win_in0)
                        .max(0) as f64;
                    let frac = (span / p.in_hop.max(1) as f64).clamp(0.0, 1.0);
                    let desired_cum = self.ts_win_out0
                        + (p.in_hop as f64 * rate_j).round() * frac;
                    // DLL emission never retreats (`sub_18014DF50` targets are
                    // monotone node positions) — clamp the per-sub-granule
                    // advance at zero so a mid-stream commit's table jump
                    // can't make the chase pull output backwards (the commit
                    // re-anchor case; offline single-parameter renders always
                    // have positive ~table-step advances).
                    let advance =
                        (desired_cum - self.ts_cum_out).round().clamp(0.0, f64::MAX) as i64;
                    self.ts_cum_out += advance as f64;
                    let delta = advance - self.out_step_1404 as i64;
                    if delta != 0 {
                        self.pos_1384 += delta;
                        self.writepos_1224 += delta;
                    }
                    self.ts_out_step = advance as i32;
                }
                // Consumers compare `== 2` (phase.rs:279/446, the τ picks
                // below), so this must carry the machine's RAW state
                // (0..3), not the state2 bool — the bool silently killed
                // all six gates (found in the 2026-11-04 audit).
                self.transient_state_2488 = p.state as u32;
            }
            self.ts_plan = plan;
            let _ = (cursor, pos);
        } else {
            self.ts_out_step = 0;
            self.transient_state_2488 = 0;
        }
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
            self.sched_cdel = vc_calc_sched(self.g_index, self.ratio, self.slope, self.cfg.step_base).1;
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
        // §4 commit gate: parameter changes staged by a mid-stream
        // `set_ratio` are applied here, at the head of the processing loop.
        self.commit_pending();
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

        // The band split is per channel and shares nothing between channels, so it
        // is split across threads. That is exact by construction — each channel
        // computes the same operations in the same order it would alone — and it is
        // worth doing because this filter is the single largest cost in the
        // vocoder (27% of a render after the hardware-FMA fix).
        let base2 = base;
        let cap2 = cap;
        let work = |c: usize, xo: &mut Crossover, ring: &mut [f32]| {
            let input: Vec<f32> = (0..nin).map(|i| x[i * nch + c]).collect();
            let t_xo = profile_enabled().then(Instant::now);
            let bands = xo.process(&input);
            if let Some(t0) = t_xo {
                profile_add(|p| &mut p.crossover, t0);
            }
            let t_ring = profile_enabled().then(Instant::now);
            for b in 0..nbands {
                for i in 0..nin {
                    let j = crate::consts::c_mod(base2 + i as i64 - 1023, cap2 as i64) as usize;
                    ring[b * cap2 + j] = bands[b][i];
                }
            }
            if let Some(t0) = t_ring {
                profile_add(|p| &mut p.ring_scatter, t0);
            }
        };

        let per_ch = nbands * cap;
        let mut xovers = std::mem::take(&mut self.xovers);
        let mut rings: Vec<&mut [f32]> = self.ring.chunks_mut(per_ch).collect();
        if parallel_enabled() && nch > 1 && rings.len() >= nch {
            let (xo0, xo_rest) = xovers.split_first_mut().expect("at least one channel");
            let (ring0, ring_rest) = rings.split_first_mut().expect("at least one channel");
            std::thread::scope(|s| {
                let mut handles = Vec::with_capacity(nch - 1);
                for (c, (xo, ring)) in xo_rest.iter_mut().zip(ring_rest.iter_mut()).enumerate() {
                    let w = &work;
                    handles.push(s.spawn(move || w(c + 1, xo, ring)));
                }
                work(0, xo0, ring0);
                for h in handles {
                    h.join().expect("crossover worker panicked");
                }
            });
        } else {
            for (c, (xo, ring)) in xovers.iter_mut().zip(rings.iter_mut()).enumerate() {
                work(c, xo, ring);
            }
        }
        drop(rings);
        self.xovers = xovers;

        self.fed_total += nin as i64;
        // Input-side transient scoring: mono mixdown in, finalised chunk scores
        // into the queue (the DLL runs the detector on the block level in
        // `sub_18014D7A0` before any granule work).
        if self.cfg.transient_sched {
            let nch_f = nch as f32;
            let mono: Vec<f32> = (0..nin)
                .map(|i| {
                    let mut s = 0.0f32;
                    for c in 0..nch {
                        s += x[i * nch + c];
                    }
                    s / nch_f
                })
                .collect();
            for (t, s) in self.tdetector.push(&mono) {
                if s > 1.0 {
                    self.ts_events += 1;
                }
                self.tqueue.push_at(t, s);
            }
        }
        // The detector finalises a chunk's score only 7 chunks (5 comb
        // look-ahead + 2 NMS window) after its audio was fed, and the
        // scheduler's forward queries reach further than the old 7-chunk
        // margin assumed: the state-2 re-query window is round(sr*0.1)
        // (4800 @48k) and the compress-side range is
        // round(hop*0.5*max_w/slope + step/slope) (measured 6326/8380 on the
        // AU corpus) — hold the pump back far enough that both always see
        // finalised scores (2026-11-04 audit D60).
        let margin = if self.cfg.transient_sched {
            let need = (((self.cfg.sr as f64) * 0.1) as i64).max(8380) + 7 * 128;
            ((need + 127) / 128) * 128
        } else {
            0
        };
        while self.cursor_1376 + (self.hop_1420 as i64) + margin < self.fed_total {
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
    /// `vc_render` — the C driver's feed rhythm and resampler drain loop.
    ///
    /// Produces exactly as many frames as the input has, which is what the reference
    /// driver does and what the parity corpus is measured against.
    pub fn render(&mut self, x: &[f32]) -> Vec<f32> {
        let n = x.len() / self.cfg.nch.max(1);
        self.render_to(x, n)
    }

    /// As [`VocoderState::render`], but producing `target_frames` output frames.
    ///
    /// With a stretch configured ([`VocoderState::set_ratio`]'s `tempo`), the
    /// caller asks for `round(input_frames * stretch)`: the schedule runs at the
    /// slope `stretch * pitch_ratio` (see [`Self::slope`]), so the output ring is
    /// written `stretch` times as fast as the pitch-only run while the drain still
    /// reads it at the pitch ratio — exactly the two rates measured in
    /// `dvaaudiodsp.dll` — and the render lands on the stretched duration at the
    /// unshifted tempo of the pitch. Feeding past the end of the input (as
    /// zeros) only covers the drain's settling lag, a hop or so, not the whole
    /// stretched tail.
    ///
    /// A `target_frames` equal to the input length (and a stretch of 1) reproduces
    /// [`VocoderState::render`] bit for bit.
    pub fn render_to(&mut self, x: &[f32], target_frames: usize) -> Vec<f32> {
        profile_reset();
        let t_render = profile_enabled().then(Instant::now);
        self.start_streaming();
        let nch = self.cfg.nch;
        // Real input frames available; everything past this reads as silence.
        let in_frames = x.len() / nch;
        let target = target_frames;
        // §8 law: the DLL's emitter bypasses the whole chain with a plain
        // ring copy when the ratio is exactly 1 (`a9 == 1.0`), i.e. tempo 100
        // + pitch 0 — our engine would otherwise still vocoder-color a
        // no-op render (2026-11-04 audit D17).
        if self.slope == 1.0 && self.ratio == 1.0 && target == in_frames && in_frames > 0 {
            self.progress_total = target as i64;
            self.progress
                .store(target as u64, std::sync::atomic::Ordering::Relaxed);
            return x[..target * nch].to_vec();
        }
        let ring_len = self.cfg.ring_out_len as i64;
        // The drain resampler's `rate` **is** the pitch: reading a ring that already
        // holds the shifted signal at rate `r` scales its frequency by `r` (verified
        // directly — a 440 Hz ring read at 1.1892 gives 524 Hz). So the rate must be
        // the pitch ratio alone. Putting the stretch here too is what made `--tempo`
        // an octave shift.
        //
        // The stretch acts one layer up, in the schedule slope (`self.slope`): the
        // ring's write cursor advances `stretch * r` per input frame while this drain
        // consumes `r` per output frame, so the output runs `stretch` times the input
        // length at the pitch the synthesis already applied.
        let ratio = self.ratio;
        // Progress is reported against the frames we are actually going to make.
        self.progress_total = target.max(1) as i64;
        self.progress.store(0, std::sync::atomic::Ordering::Relaxed);
        let mut out = vec![0.0f32; target * nch];
        let tbl = InterpTable::new(8192, 6, 16.0);
        let mut drain_phase = 0.0f64;
        let mut out_n = 0usize;
        let mut pos = 0usize;
        const CH: usize = 1024;
        let mut chunk = vec![0.0f32; CH * nch];
        // Feed the whole input, then keep feeding silence while the drain still has
        // signal to take — the reference's own loop, which runs until it has
        // `target` output frames.
        //
        // There used to be a `feed_limit = target * ratio` here. That was wrong at low
        // ratios: the resampler reads the output ring slowly, so a downward shift needs
        // *more* ring frames than it emits output frames, and the limit starved it. At
        // -28 semitones it cut the feed to 9 525 frames, `pos_1384` never passed the hop
        // (`writepos` stayed 0, so `head` was -100 and the drain took nothing), and the
        // whole render came out silent. The reference renders -36 semitones fine, and
        // now so does this. Termination is the `out_n >= target` check plus the drain's
        // own `head <= drain_phase`, exactly as in the reference.
        while out_n < target {
            let n = if pos == 0 { 448 } else { 1024 };
            for v in chunk[..n * nch].iter_mut() {
                *v = 0.0;
            }
            if pos < in_frames {
                let avail = (in_frames - pos).min(n);
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
                if out_n as i64 + chunk_n > target as i64 {
                    chunk_n = target as i64 - out_n as i64;
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
                self.progress.store(
                    out_n as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                drain_phase += chunk_n as f64 * ratio;
                if out_n >= target {
                    break;
                }
            }
        }
        if let Some(t0) = t_render {
            profile_add(|p| &mut p.wall, t0);
        }
        profile_report();
        self.apply_gain(&mut out);
        out
    }

    /// Enable per-granule tracing on stderr.
    pub fn set_trace(&mut self, on: bool) {
        self.trace = on;
    }

    /// Progress of the render in flight, measured in final output frames.
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

#[cfg(test)]
mod commit_tests {
    use super::*;

    /// Deterministic mono test signal (two-tone + AM).
    fn signal(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f64 / 48_000.0;
                (0.5 * (2.0 * std::f64::consts::PI * 220.0 * t).sin()
                    + 0.3 * (2.0 * std::f64::consts::PI * 554.0 * t).sin()
                    + 0.1 * (2.0 * std::f64::consts::PI * 13.0 * t).cos())
                    as f32
            })
            .collect()
    }

    /// (a) A `set_ratio` before the first feed applies immediately: one call
    /// and a replaced pre-stream call give bit-identical renders, and no
    /// commit/anchor ever fires (offset stays 0) — the pre-change behaviour.
    #[test]
    fn pre_stream_set_ratio_is_immediate_and_bit_identical() {
        let x = signal(16_384);
        let mut a = VocoderState::new(48_000, 1);
        a.set_ratio(3.0, 150.0);
        let ra = a.render(&x);
        let mut b = VocoderState::new(48_000, 1);
        b.set_ratio(0.0, 100.0);
        b.set_ratio(3.0, 150.0);
        let rb = b.render(&x);
        assert_eq!(ra.len(), rb.len());
        assert!(ra.iter().zip(&rb).all(|(u, v)| u.to_bits() == v.to_bits()));
        assert_eq!(b.sched_pos_offset, 0);
        assert!(b.pending_semis.is_none() && b.pending_tempo.is_none());
    }

    /// (b) A mid-stream `set_ratio` commits at a feed boundary with a
    /// continuous granule lattice: within the current group the position is
    /// unchanged, the next group steps by the NEW `out_step`, and nothing
    /// jumps backwards.
    #[test]
    fn mid_stream_commit_keeps_schedule_continuous() {
        // The transient subsystem is disabled here to observe the pure
        // schedule table + commit anchor (the dual-queue chase redistributes
        // output within plan windows on top of this lattice).
        let mut st = VocoderState::new(48_000, 1);
        st.set_transient_sched(false);
        st.set_ratio(0.0, 100.0);
        st.start_streaming();
        let sil = vec![0.0f32; 30_000];
        st.feed(&sil, 30_000);
        assert!(st.g_index >= 8, "warm-up done");

        // Step granules one at a time to watch the position sequence.
        let mut seq = Vec::new();
        for _ in 0..6 {
            st.process_granule();
            seq.push(st.pos_1384);
        }
        // Mid-group commit (g_index % 4 != 1): the position must hold.
        st.set_ratio(3.0, 200.0);
        st.commit_pending();
        let os_new =
            vc_calc_sched(st.g_index.max(5), st.ratio, st.slope, st.cfg.step_base).3;
        assert!(os_new > 0);
        let mut post = Vec::new();
        for _ in 0..6 {
            st.process_granule();
            post.push(st.pos_1384);
        }
        // Commit applied...
        assert!((st.slope - (st.ratio * 2.0)).abs() < 1e-12);
        assert_eq!(st.pending_semis, None);
        // ...first post-commit granule stays anchored (same group, no jump)…
        assert_eq!(post[0], *seq.last().unwrap());
        // …the sequence never decreases…
        assert!(post.windows(2).all(|w| w[1] >= w[0]), "post={post:?}");
        // …and group-boundary steps are exactly the new table step.
        let steps: Vec<i64> = post.windows(2).map(|w| w[1] - w[0]).collect();
        assert!(steps.iter().all(|&d| d == 0 || d == os_new), "steps={steps:?}");
    }

    /// (c) The 32-frame gate: the first commit (input base 0) passes, a
    /// second one waits until ≥ 32 frames have arrived since the last commit.
    #[test]
    fn commit_gates_on_32_frames() {
        let mut st = VocoderState::new(48_000, 1);
        st.set_ratio(0.0, 100.0);
        st.start_streaming();
        let sil = vec![0.0f32; 30_000];
        st.feed(&sil, 30_000);
        let slope0 = st.slope;

        // First mid-stream change: in_base == 0 → commits immediately.
        st.set_ratio(3.0, 200.0);
        let small = vec![0.0f32; 8];
        st.feed(&small, 8);
        assert!((st.slope - slope0).abs() > 1e-9, "first commit fired");

        // Second change: fewer than 32 frames since the commit → deferred.
        st.set_ratio(-3.0, 50.0);
        let slope1 = st.slope;
        st.feed(&small, 8); // 8 more (< 32)
        assert!((st.slope - slope1).abs() < 1e-12, "gate holds under 32");
        assert!(st.pending_semis.is_some());
        st.feed(&small, 4); // 12 total (< 32)
        assert!((st.slope - slope1).abs() < 1e-12, "gate still holds at 12");
        let more = vec![0.0f32; 24];
        st.feed(&more, 24); // 36 ≥ 32 at the NEXT feed top → commits
        assert!((st.slope - slope1).abs() < 1e-12, "still old at 36 (gate sees pre-feed 12)");
        st.feed(&more, 24); // top sees 36 ≥ 32 → commits
        assert!((st.slope - slope1).abs() > 1e-12, "commit at ≥32");
        assert_eq!(st.pending_semis, None);
    }
}
