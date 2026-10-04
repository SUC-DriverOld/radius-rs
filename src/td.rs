//! Time-domain (TD) pitch/time engine — `libradius/src/engine/td_core.c` and
//! `src/ops/td_pitch.c` / `td_octave.c`, mirroring `pyradius.td_core`.
//!
//! The chain is: input ring -> per-granule `ProcessCore` (int64 cursor + phase
//! accumulator) -> transient detection (`TransientsInfo`) -> `DoOla`
//! overlap-add -> drained through the `InterpolateNSamples` resampler.
//!
//! Float discipline (see `pyradius/README.md`): everything is `f32` unless the C
//! source computes in `double`; `a*b + c` stays two operations to mirror
//! `-ffp-contract=off`; transcendental calls are evaluated in `f64` and rounded
//! once, which is what the C compiler emits for `expf`/`logf`/`powf` here.

use crate::consts::*;
use crate::fft::with_plan;
use crate::interp::{interp_nsamples, InterpTable};
use crate::tables;
use crate::tables::PitchGeom;
use crate::ti::TiState;

pub const RX_TD_NTAB: usize = 32;
pub const RX_TD_NBLK: usize = 5;

/// A shared "work done" counter for progress reporting.
///
/// `Arc<AtomicU64>` is `Send`/`Sync` even though [`TdState`] is not, which is what
/// lets a watchdog thread watch a render that is running elsewhere.
pub type ProgressCounter = std::sync::Arc<std::sync::atomic::AtomicU64>;

/// Create a zeroed progress counter.
pub fn new_progress_counter() -> ProgressCounter {
    std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0))
}

const NO_TRANSIENT: u32 = 0xFFFF_FFFF;

/// `rx_td_build_env_tables` output: 5 gain blocks x 32 spans, each span a
/// doubled Hann^gain record.
struct EnvTables {
    spans: Vec<Vec<usize>>,
    tabs: Vec<Vec<Vec<f32>>>,
    gains: Vec<f32>,
}

/// Pitch front-end state (`Pitch::SetOptions` + `AnalyzePitchPrecise`).
struct PitchState {
    geom: PitchGeom,
    win_taper: Vec<f32>,
    win_an: Vec<f32>,
    win_acf: Vec<f32>,
    win_lin: Vec<f32>,
    octave_en: bool,
}

pub struct TdState {
    pub sr: u32,
    pub nch: usize,
    pub solo: i32,
    pub quality: i32,

    pub hop: u32,
    pub f28: u32,
    pub win_max: u32,
    pub state_330: u32,

    env: EnvTables,
    pitch: PitchState,

    ring_tot: usize,
    out_len: usize,

    // scratch
    ring: Vec<f32>,
    out_ring: Vec<f32>,
    mix: Vec<f32>,

    pitch_buf: Vec<f32>,

    ti: Option<TiState>,

    pub stretch: f64,
    /// Output gain in dB, applied to the finished samples. 0.0 is a no-op.
    pub gain_db: f64,
    pub total_ratio: f64,
    pub pitch_ratio: f64,

    // runtime cursors
    cursor: i64,
    cursor_prev: i64,
    acc: f64,
    skip_next_318: bool,
    last_transient: u32,
    dec_flag: bool,
    advanced: bool,
    forced_t: u32,
    write_ptr_acc: f64,
    pub wrap_cnt: u64,
    pub n_granule: u64,
    pub n_transient: u64,
    /// Granules completed, for progress reporting.
    ///
    /// Shared rather than owned so a watchdog thread can poll it while the
    /// renderer (which is not `Send`) runs on this thread. The renderer only ever
    /// writes it with `Relaxed` ordering: it is a progress counter, never a means
    /// of synchronising anything.
    progress: ProgressCounter,
    /// Granules this render is expected to produce (set by [`TdState::render`]).
    progress_total: u64,
    dedup: Vec<u32>,
    dedup_cap: usize,
    last_pick_j: usize,
    /// DataTail visible bound (`+0x2F4`): ring positions `< fed_visible` hold
    /// real input, everything else reads as zero.
    fed_visible: i64,
    /// Per-granule trace to stderr (mirrors `PYR_TD_DBG`).
    trace: bool,
    /// Final ring read position (diagnostics only; not part of the C state).
    #[doc(hidden)]
    pub in_pos_final: i64,
}

impl TdState {
    pub fn new(sr: u32, quality: i32, solo: i32, nch: usize) -> Self {
        let s8 = quality as f32;
        let s1 = 1.5f32 * s8;
        let s2 = sr as f32 * 0.001f32;
        let hop = (s2 * s1 + 0.5) as u32;
        let f28 = (((sr as f32 * 0.1f32) + 0.5) as u32) & !7u32;
        let win_max = 4u32;
        let state_330 = (solo as u32) ^ 1;
        let ring_tot = (f28 as usize) * 16;

        let env = Self::build_env_tables(f28 as usize, win_max as usize);
        let geom = tables::pitch_geom(sr, hop, solo != 0, win_max as usize);
        let pitch = PitchState {
            win_taper: tables::pitch_win_taper(geom.taper_len),
            win_an: tables::pitch_win_an(geom.l1),
            win_acf: tables::pitch_win_acf(geom.n, geom.l1),
            win_lin: tables::pitch_win_lin(geom.n),
            geom,
            octave_en: true,
        };

        let mut out_len = ring_tot * 8 + 65536;
        if out_len < (hop as usize) * 4 + 8192 {
            out_len = (hop as usize) * 4 + 8192;
        }
        let dedup_cap = (ring_tot / 32 + 64).max(64);

        Self {
            sr,
            nch,
            solo,
            quality,
            hop,
            f28,
            win_max,
            state_330,
            env,
            pitch,
            ring_tot,
            out_len,
            ring: vec![0.0; ring_tot * nch],
            out_ring: vec![0.0; (out_len + 16) * nch],
            mix: Vec::new(),
            pitch_buf: vec![0.0; geom.l1 + geom.n],
            ti: None,
            stretch: 1.0,
            gain_db: 0.0,
            total_ratio: 1.0,
            pitch_ratio: 1.0,
            cursor: 0,
            cursor_prev: 0,
            acc: 0.0,
            skip_next_318: false,
            last_transient: NO_TRANSIENT,
            dec_flag: false,
            advanced: false,
            forced_t: NO_TRANSIENT,
            write_ptr_acc: 0.0,
            wrap_cnt: 0,
            n_granule: 0,
            n_transient: 0,
            progress: new_progress_counter(),
            progress_total: 0,
            dedup: Vec::new(),
            dedup_cap,
            last_pick_j: 0,
            fed_visible: 0,
            trace: false,
            in_pos_final: 0,
        }
    }

    /// Final input cursor after a render (diagnostics).
    pub fn cursor_final(&self) -> i64 {
        self.cursor
    }

    /// Progress of the render in flight: `(granules done, expected granules)`.
    ///
    /// The returned counter is shared with the state, so it can be polled from
    /// another thread while [`TdState::render`] runs. `total` is an estimate;
    /// treat the ratio as a progress indicator, not as a completion guarantee.
    pub fn progress_counter(&self) -> ProgressCounter {
        self.progress.clone()
    }

    /// Install an externally owned progress counter and the expected total.
    ///
    /// Used by the CLI so a watchdog thread can poll the render while it runs.
    pub fn set_progress_counter(&mut self, counter: ProgressCounter, total: u64) {
        self.progress = counter;
        self.progress_total = total.max(1);
    }

    /// Progress as a `(done, total)` pair, for callers on the same thread.
    pub fn progress(&self) -> (u64, u64) {
        (
            self.progress.load(std::sync::atomic::Ordering::Relaxed),
            self.progress_total.max(1),
        )
    }

    /// Final ring read position after a render (diagnostics).
    pub fn in_pos(&self) -> i64 {
        self.in_pos_final
    }

    /// `rx_td_build_env_tables`.
    fn build_env_tables(f28: usize, win_max: usize) -> EnvTables {
        let spans = tables::td_env_spans(f28 as u32, RX_TD_NBLK, RX_TD_NTAB);
        let mut tabs = Vec::with_capacity(RX_TD_NBLK);
        let mut gains = Vec::with_capacity(RX_TD_NBLK);
        for a8 in 0..RX_TD_NBLK {
            let g = if a8 >= win_max {
                1.0f32
            } else {
                0.5625f32 + 0.125f32 * a8 as f32
            };
            gains.push(g);
            let mut row = Vec::with_capacity(RX_TD_NTAB);
            for j in 0..RX_TD_NTAB {
                row.push(tables::hann_pow(spans[a8][j], g));
            }
            tabs.push(row);
        }
        EnvTables { spans, tabs, gains }
    }

    /// `rx_td_set_ratio`.
    pub fn set_ratio(&mut self, semis: f64, tempo: f64) {
        self.stretch = tempo / 100.0;
        self.pitch_ratio = 2.0f64.powf(semis / 12.0);
        self.total_ratio = self.pitch_ratio * self.stretch;
    }

    /// Multiply the output by `db` decibels, as a plain linear scale.
    ///
    /// Applied to the finished samples, after the engine, so it cannot interact with
    /// the reference arithmetic: `set_gain(0.0)` is a no-op and leaves the output
    /// bit-exact.
    pub fn set_gain(&mut self, db: f64) -> &mut Self {
        self.gain_db = db;
        self
    }

    pub fn geometry(&self) -> &PitchGeom {
        &self.pitch.geom
    }

    /// Fill the input ring from the head of `x` and set the visible bound; used
    /// by the diagnostics binary and the parity tests.
    pub fn fill_ring(&mut self, x: &[f32], fed_visible: i64) {
        let n = self
            .ring_tot
            .min(if self.nch == 0 { 0 } else { x.len() / self.nch });
        for i in 0..n {
            for c in 0..self.nch {
                self.ring_set(c, i, x[i * self.nch + c]);
            }
        }
        self.fed_visible = fed_visible;
    }

    /// Enable the per-granule trace on stderr (mirrors `PYR_TD_DBG`).
    pub fn set_trace(&mut self, on: bool) {
        self.trace = on;
    }

    #[inline]
    fn ring_at(&self, ch: usize, pos: usize) -> f32 {
        unsafe { *self.ring.get_unchecked(ch * self.ring_tot + pos) }
    }

    #[inline]
    fn ring_set(&mut self, ch: usize, pos: usize, v: f32) {
        self.ring[ch * self.ring_tot + pos] = v;
    }

    // ------------------------------------------------------------------
    // rx_td_analyze_pitch
    // ------------------------------------------------------------------

    /// Gather `l1` samples starting at `in_pos - l1/2`, summing channels while
    /// the ring position is still visible, and return the raw energy.
    fn gather_pitch(&mut self, in_pos: i64, fed_visible: i64) -> f64 {
        let l1 = self.pitch.geom.l1;
        let modl = self.ring_tot as i64;
        let nch = self.nch;
        let ring_tot = self.ring_tot;
        let mut energy = 0.0f64;
        {
            let ring = &self.ring;
            let x = &mut self.pitch_buf;
            for v in x[..l1].iter_mut() {
                *v = 0.0;
            }
            for i in 0..l1 {
                let idx = wrap_ring(in_pos - (l1 as i64 >> 1) + i as i64, 0, modl) as usize;
                if (idx as i64) < fed_visible {
                    let mut s = ring[idx];
                    if nch > 1 {
                        s += ring[ring_tot + idx];
                    }
                    x[i] = s;
                }
            }
            for i in 0..l1 {
                let v = x[i] as f64;
                energy += v * v;
            }
        }
        energy
    }

    /// `rx_td_pitch_analyze` — returns `(lag, win, rms)`.
    pub fn analyze_pitch(&mut self, in_pos: i64, fed_visible: i64) -> (f64, i32, f32) {
        let n = self.pitch.geom.n;
        let l1 = self.pitch.geom.l1;
        let energy_raw = self.gather_pitch(in_pos, fed_visible);

        let rms = (energy_raw / l1 as f64).sqrt() as f32;

        // analysis window
        {
            let wa = &self.pitch.win_an;
            let x = &mut self.pitch_buf;
            for i in 0..l1 {
                x[i] *= wa[i];
            }
        }
        let mut sa = with_plan(n, |p| p.fwd_split(&self.pitch_buf, l1));
        let m = n >> 1;
        sa[1] = 0.0;
        for k in 0..m {
            let re = sa[2 * k];
            let im = sa[2 * k + 1];
            sa[2 * k] = re * re + im * im;
            sa[2 * k + 1] = 0.0;
        }
        sa[n + 1] = 0.0;
        let mut x = with_plan(n, |p| p.inv(&sa));
        {
            let w = &self.pitch.win_acf;
            for i in 0..n {
                x[i] *= w[i];
            }
        }
        let sb = with_plan(n, |p| p.fwd(&x));

        // whitening
        let v387 = (0.5 + 20.0 / self.sr as f64 * n as f64) as i64;
        let v386 = (0.5 + 150.0 / self.sr as f64 * n as f64) as i64 + 1;
        let e02 = energy_raw * 0.2;
        let noise = (if e02 > 1e-6 { e02 } else { 1e-6 }) * (self.sr as f64 / 44100.0);
        let noisef = noise as f32;
        let maxbin = self.pitch.geom.maxbin;
        let taper_len = self.pitch.geom.taper_len;
        let mut kw = maxbin + taper_len;
        if kw > m {
            kw = m;
        }
        let mut out = vec![0.0f32; m];
        if kw as i64 > v387 {
            let ks_lo = v387.max(0) as usize;
            for k in ks_lo..kw {
                let mag = sb[2 * k].abs();
                // NOTE: f64 pow then one rounding == the C's powf here
                let den = noisef + (mag as f64).powf(0.95) as f32;
                let mut o = sa[2 * k] / den;
                if k >= maxbin {
                    let ti = (k - maxbin).min(taper_len - 1);
                    o = (o as f64 * self.pitch.win_taper[ti] as f64) as f32;
                }
                if (k as i64) < v386 {
                    let t = (k as f64 - v387 as f64) / (v386 - v387) as f64;
                    let fac = 0.5 - 0.5 * (PI_F64 * t).cos();
                    o = (o as f64 * fac) as f32;
                }
                out[k] = o;
            }
        }
        let mut x2 = vec![0.0f32; n + 2];
        for k in 0..m {
            x2[2 * k] = out[k];
        }
        let mut acf = with_plan(n, |p| p.inv(&x2));
        let nb = (n >> 1) | 1;
        {
            let wl = &self.pitch.win_lin;
            for i in 0..nb {
                acf[i] *= wl[i];
            }
        }
        // leading decreasing suppression
        {
            let mut i = 0usize;
            while i + 1 < nb && acf[i] > acf[i + 1] {
                acf[i] = -1000.0;
                i += 1;
            }
        }

        let mut v160 = self.pitch.geom.lo as i64;
        let mut v138 = self.pitch.geom.hi as i64;
        if v160 < 1 {
            v160 = 1;
        }
        if v138 > nb as i64 - 2 {
            v138 = nb as i64 - 2;
        }
        if v138 <= v160 {
            return (0.0, 0, rms);
        }
        // argmax over acf[v160..=v138]
        let mut p = v160 as usize;
        let mut best = acf[v160 as usize];
        for i in (v160 as usize + 1)..=(v138 as usize) {
            if acf[i] > best {
                best = acf[i];
                p = i;
            }
        }
        let run_oct = !self.pitch.octave_en;
        if run_oct {
            if let Some((p2, _sus, _ratio)) =
                octave_correct_peak(&acf, nb, v160 as usize, v138 as usize, true)
            {
                p = p2;
            }
        }
        self.pitch.octave_en = false;

        let mut lag = p as f64;
        let y0 = acf[p - 1] as f64;
        let y1 = acf[p] as f64;
        let y2 = acf[p + 1] as f64;
        let den = 2.0 * y1 - y2 - y0;
        if den != 0.0 {
            let mut d = (y2 - y0) / (2.0 * den);
            if d < -1.0 {
                d = -1.0;
            }
            if d > 1.0 {
                d = 1.0;
            }
            lag = p as f64 + d;
        }

        // clarity -> win
        let v14 = acf[p] as f64;
        let mut comp = 0.0f64;
        let lb = (0.6 * p as f64) as i64;
        let rb = (1.8 * p as f64 + 0.5) as i64;
        if p as i64 - 4 >= 1 {
            let mut g = p as i64 - 4;
            while g > lb && (acf[g as usize] as f64) < (acf[g as usize - 1] as f64) {
                g -= 1;
            }
            if g >= 1 && acf[g as usize] as f64 > comp {
                comp = acf[g as usize] as f64;
            }
        }
        if (p as i64) + 4 < nb as i64 - 1 {
            let mut g = p as i64 + 4;
            while g < rb
                && g < nb as i64 - 2
                && (acf[g as usize] as f64) < (acf[g as usize + 1] as f64)
            {
                g += 1;
            }
            if acf[g as usize] as f64 > comp {
                comp = acf[g as usize] as f64;
            }
        }
        let l22 = (2.2 * p as f64) as i64;
        let r28 = (2.8 * p as f64 + 0.5) as i64;
        let hi2 = r28.min(nb as i64 - 2);
        if hi2 >= l22 {
            let base = l22.max(1);
            if hi2 >= base {
                let mut mx = f32::NEG_INFINITY;
                for i in base..=hi2 {
                    let v = acf[i as usize];
                    if v > mx {
                        mx = v;
                    }
                }
                if mx as f64 > comp {
                    comp = mx as f64;
                }
            }
        }
        if comp < 0.0 {
            comp = 0.0;
        }
        let clarity = if v14 != 0.0 { 1.0 - comp / v14 } else { 0.25 };
        let clarity = if clarity < 0.0 { 0.0 } else { clarity };
        let mut w = ((self.win_max as f64 + 0.4) * clarity.powf(1.1) + 0.5) as i64;
        if w > self.win_max as i64 - 1 {
            w = self.win_max as i64 - 1;
        }
        if w < 0 {
            w = 0;
        }
        (lag, w as i32, rms)
    }

    // ------------------------------------------------------------------
    // rx_td_pick_env
    // ------------------------------------------------------------------

    /// Returns `(env_record, half, w10)`.
    pub fn pick_env(
        &mut self,
        a8_in: i32,
        a9: i32,
        a10: i32,
        boolean: bool,
    ) -> (Vec<f32>, usize, i64) {
        let a8 = if (a8_in as usize) < RX_TD_NBLK {
            a8_in as usize
        } else {
            RX_TD_NBLK - 1
        };
        let v12 = if boolean { (2 * a10) / 3 } else { a9 };
        let mut j = RX_TD_NTAB - 1;
        for k in 0..RX_TD_NTAB {
            if (v12 as i64) < self.env.spans[a8][k] as i64 {
                j = if k == 0 { 0 } else { k - 1 };
                break;
            }
        }
        let span = self.env.spans[a8][j];
        let env = self.env.tabs[a8][j].clone();
        self.last_pick_j = j;
        let half = span >> 1;
        let w10 = if boolean {
            0
        } else {
            ((a9 >> 1) as i64) - half as i64
        };
        (env, half, w10)
    }

    pub fn last_pick_j(&self) -> usize {
        self.last_pick_j
    }

    /// `hann_pow` gain of block `a8` (used by tests/diagnostics).
    pub fn env_gain(&self, a8: usize) -> f32 {
        self.env.gains[a8]
    }

    // ------------------------------------------------------------------
    // DoOla
    // ------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn do_ola(
        &mut self,
        fed_visible: i64,
        a4: i64,
        wr: i64,
        a8: i32,
        a9: i32,
        a10: i32,
        boolean: bool,
        gain: f32,
    ) {
        if self.nch == 0 || a9 == 0 {
            return;
        }
        let rcap = self.ring_tot as i64;
        let rstart = 0i64;
        let outlen = self.out_len as i64;

        if (wr | a4) == 0 {
            for i in 0..a9 as i64 {
                let ri = wrap_ring(i, rstart, rcap) as usize;
                for c in 0..self.nch {
                    let sv = if (ri as i64) < fed_visible {
                        self.ring_at(c, ri)
                    } else {
                        0.0
                    };
                    let oi = (c as i64 * (outlen + 16) + i) as usize;
                    self.out_ring[oi] = gain * sv;
                }
            }
        }
        let (env, half, w10) = self.pick_env(a8, a9, a10, boolean);
        if half == 0 {
            return;
        }
        let rbase = a4 + w10;
        let wbase = wr + w10;
        let envhalf = 2 * half;
        if env.len() < envhalf {
            return;
        }
        let stride = outlen + 16;
        // segment 1: crossfade accumulate
        for i in 0..half as i64 {
            let ri = wrap_ring(rbase + i, rstart, rcap) as usize;
            let wi = wrap_out(wbase + i, outlen) as usize;
            let s1 = env[half + i as usize];
            let s0 = env[i as usize];
            for c in 0..self.nch {
                let sv = if (ri as i64) < fed_visible {
                    self.ring_at(c, ri)
                } else {
                    0.0
                };
                let oi = c as i64 * stride + wi as i64;
                let oi = oi as usize;
                self.out_ring[oi] = self.out_ring[oi] * s1 + gain * sv * s0;
            }
        }
        // segment 2: plain copy tail
        let w11 = if boolean {
            a10
        } else if a10 > a9 {
            a10
        } else {
            a9
        };
        if w11 > half as i32 {
            for i in (half as i64)..(w11 as i64) {
                let ri = wrap_ring(rbase + i, rstart, rcap) as usize;
                let wi = wrap_out(wbase + i, outlen) as usize;
                for c in 0..self.nch {
                    let sv = if (ri as i64) < fed_visible {
                        self.ring_at(c, ri)
                    } else {
                        0.0
                    };
                    let oi = (c as i64 * stride + wi as i64) as usize;
                    self.out_ring[oi] = gain * sv;
                }
            }
        }
    }

    /// `rx_td_render` — render `nframes` interleaved frames into an interleaved
    /// output buffer, returning the number of output frames.
    /// Render the first `nframes` input frames, producing `target_out` output frames.
    ///
    /// `nframes` is how much **input** is real; everything past it reads as silence.
    /// `target_out` is how many **output** frames to make, which is what `--tempo`
    /// moves: the stretch scales the output timeline while the pitch ratio moves the
    /// read cursor, so `target_out = round(nframes * tempo / 100)` is longer than the
    /// input when the tempo is above 100 and shorter when it is below.
    ///
    /// The reference driver produces exactly `nframes`, so
    /// `render(x, nframes, nframes)` is the reference-faithful path and the one the
    /// parity corpus is measured on.
    pub fn render(&mut self, x: &[f32], nframes: usize, target_out: usize) -> Vec<f32> {
        let hop = self.hop as i64;
        let win_max = self.win_max;
        let nch = self.nch;
        let ring_tot = self.ring_tot as i64;
        let out_len = self.out_len as i64;

        for v in self.ring.iter_mut() {
            *v = 0.0;
        }
        for v in self.out_ring.iter_mut() {
            *v = 0.0;
        }
        // Dense source buffer so arbitrary (even "negative") offsets behave.
        self.mix.clear();
        if nch >= 2 {
            self.mix.reserve(nframes);
            for i in 0..nframes {
                self.mix.push(x[i * nch] + x[i * nch + 1]);
            }
        } else {
            self.mix.reserve(nframes);
            for i in 0..nframes {
                self.mix.push(x[i * nch]);
            }
        }

        let tbl = InterpTable::new(1024, 6, 12.0);
        if self.ti.is_none() {
            self.ti = Some(TiState::new(nch, self.sr, 1.0));
        }

        let mut out_chunks: Vec<Vec<f32>> = Vec::new();
        let mut out_n = 0usize;
        let wp = 0.9499998092651367f64;
        let mut cursor: i64 = 0;
        let mut in_pos: i64 = 0;
        let mut fed_pos: i64 = 0;
        let mut acc: f64 = 0.0;

        self.write_ptr_acc = wp;
        self.cursor = 0;
        self.cursor_prev = 0;
        self.acc = 0.0;
        self.advanced = false;
        self.skip_next_318 = false;
        self.last_transient = NO_TRANSIENT;
        self.n_granule = 0;
        self.n_transient = 0;
        self.dedup.clear();

        let gate = hop + ((hop >> 1) << (if self.state_330 != 0 { 3 } else { 0 })) + 100;
        // The granule pump runs until the *output* is finished, so its bounds are
        // driven by `target_out`, not by the input length. When the tempo stretches
        // the timeline the pump has to keep going past the end of the input (reading
        // the zero tail), and when it compresses the pump stops early.
        let nstop = as_u32(target_out as i64) as i64;
        let total_ext = as_u32(target_out as i64 + 4 * gate + 65536) as i64;
        let mut cursor_append: i64 = 0;
        let loop_bound = as_u32(target_out as i64 + gate + 8 * hop) as i64;
        let mut feed_n = 0usize;
        let mut fed_visible: i64;
        let mix_len = self.mix.len();

        // Progress estimate for callers: the granule loop runs until `fed_pos`
        // reaches `nstop`, and it advances by roughly one period per granule, so
        // the count lands near `nframes / (hop/2)` at the default settings. It is
        // an estimate on purpose — the bar only has to look monotone, while the
        // authoritative counters (`n_granule` etc.) are exact.
        self.progress_total = if hop > 0 {
            (nframes as u64) * 2 / hop as u64 + 1
        } else {
            1
        };
        self.progress.store(0, std::sync::atomic::Ordering::Relaxed);

        while cursor_append < total_ext && cursor_append < loop_bound {
            let mut feed: i64 = if feed_n < 14 { 1024 } else { 0x2000 };
            if cursor_append + feed > total_ext {
                feed = total_ext - cursor_append;
            }
            if feed <= 0 {
                break;
            }
            // write the feed window into the ring (zero past EOF)
            for k in 0..feed {
                let g = cursor_append + k;
                let gi = (g % ring_tot) as usize;
                let ok = g < nframes as i64;
                let src = if nframes > 0 {
                    (g.min(nframes as i64 - 1)) as usize
                } else {
                    0
                };
                for c in 0..nch {
                    let v = if ok { x[src * nch + c] } else { 0.0 };
                    self.ring_set(c, gi, v);
                }
            }
            cursor_append += feed;
            let fed_total = cursor_append;
            fed_visible = fed_total;

            // TransientsInfo: 1024-sample chunks. The engine's `set_source` keeps
            // a fixed buffer origin, so the chunk index (`c394`) is the only
            // thing that advances across chunks — `base` stays at the origin of
            // `self.mix`, and the absolute read position is `base + g9`.
            {
                let mut off = 0i64;
                while off < feed {
                    let mut cl = feed - off;
                    if cl > 1024 {
                        cl = 1024;
                    }
                    let ti = self.ti.as_mut().unwrap();
                    ti.process_from(0, cl as usize, &self.mix, mix_len);
                    ti.analyze();
                    off += cl;
                }
            }
            feed_n += 1;

            let mut gate_target = if fed_total > gate {
                fed_total - gate
            } else {
                0
            };
            if gate_target > nstop {
                gate_target = nstop;
            }

            let mut guard = 0u64;
            while fed_pos < gate_target && guard < 400000 {
                guard += 1;
                #[allow(unused_assignments)]
                let mut win: i32;
                let period: i64;
                if self.state_330 == 1 {
                    let (lag, w, _rms) = self.analyze_pitch(in_pos + (3 * hop) / 8, fed_visible);
                    let mut p = (lag + 0.5) as i64;
                    if p < 1 {
                        p = 1;
                    }
                    period = p;
                    win = w;
                } else {
                    period = hop >> 2;
                    win = period as i32;
                }
                if win as i64 > win_max as i64 - 1 {
                    win = win_max as i32 - 1;
                }
                let mut istep = period;
                if istep < 1 || istep > 65536 {
                    istep = period;
                }
                self.dec_flag = false;
                self.forced_t = NO_TRANSIENT;
                let skip = self.skip_next_318;
                self.skip_next_318 = false;
                let s11 = (istep as f32) / (self.sr as f32);
                let d13 = s11 as f64;

                // pre-scan
                let mut t_prescan = NO_TRANSIENT;
                {
                    let wf0 = (2 * period) as f32;
                    let lo0 = as_u32((wf0 * 0.1f32) as i64);
                    let tl0 = as_u32((wf0 * 0.9f32) as i64);
                    let ti = self.ti.as_ref().unwrap();
                    let t = ti.get_transient_pos(in_pos + lo0 as i64, tl0 as i64);
                    if t >= 0 {
                        t_prescan = t as u32;
                    }
                }

                if !skip {
                    let h2 = hop >> 1;
                    {
                        let ti = self.ti.as_ref().unwrap();
                        let t = t_prescan;
                        if t != NO_TRANSIENT && t != self.last_transient {
                            self.forced_t = t;
                            let stp = ti.stride();
                            let fr = (t as i64) / (if stp != 0 { stp as i64 } else { 1 });
                            let fr = fr as u32;
                            let mut dup = false;
                            if !self.dedup.is_empty() {
                                if self.dedup.contains(&fr) {
                                    dup = true;
                                } else if self.dedup.len() >= self.dedup_cap {
                                    // evict the smallest entry
                                    let (oi, _) = self
                                        .dedup
                                        .iter()
                                        .enumerate()
                                        .min_by_key(|(_, v)| **v)
                                        .unwrap();
                                    self.dedup[oi] = fr;
                                } else {
                                    self.dedup.push(fr);
                                }
                            } else {
                                self.dedup.push(fr);
                            }
                            if !dup {
                                let d1 = acc * self.sr as f64
                                    + (self.total_ratio - 1.0) * (t as f64 - in_pos as f64);
                                let v37 = if d1 >= 0.0 {
                                    (d1 + 0.5) as i64
                                } else {
                                    (d1 - 0.5) as i64
                                };
                                let av = (v37 as f64).abs();
                                if (self.sr as f32 * 0.005f32) as f64 <= av {
                                    self.dec_flag = true;
                                    self.n_transient += 1;
                                } else {
                                    self.last_transient = t;
                                }
                            }
                        }
                    }
                    if self.dec_flag {
                        let mut t = self.forced_t;
                        if t == NO_TRANSIENT {
                            let wf2 = (2 * period) as f32;
                            let ti = self.ti.as_ref().unwrap();
                            let r = ti.get_transient_pos(
                                in_pos + as_u32((wf2 * 0.1f32) as i64) as i64,
                                as_u32((wf2 * 0.9f32) as i64) as i64,
                            );
                            if r >= 0 {
                                t = r as u32;
                            }
                        }
                        if t != NO_TRANSIENT {
                            let d1 = acc * self.sr as f64
                                + (self.total_ratio - 1.0) * (t as f64 - in_pos as f64);
                            let v22 = if d1 >= 0.0 {
                                (d1 + 0.5) as i64
                            } else {
                                (d1 - 0.5) as i64
                            };
                            let rmin = if self.total_ratio < 1.0 {
                                self.total_ratio
                            } else {
                                1.0
                            };
                            let extra = (h2 as f32 * rmin as f32) as i64;
                            let span = 2 * period + extra;
                            let a9v = span;
                            let delta = span - h2;
                            let mut wr_off = 0i64;
                            let mut prev_abs = cursor;
                            let mut v22e = v22;
                            if v22 < 1 {
                                let cand = cursor + v22;
                                let x12 = self.cursor_prev;
                                let mut w11b = if cand < x12 { x12 - cursor } else { v22 };
                                if !(w11b + delta > 0) {
                                    w11b = -delta;
                                }
                                wr_off = w11b;
                                v22e = w11b;
                                prev_abs = cursor + w11b;
                            }
                            let mut wpv = cursor + wr_off;
                            wpv = wrap_out(wpv, out_len);
                            let a4t = in_pos - if v22 >= 0 { v22 } else { -v22 };
                            self.do_ola(
                                fed_visible,
                                if a4t > 0 { a4t } else { 0 },
                                wpv,
                                0,
                                a9v as i32,
                                hop as i32,
                                true,
                                1.0,
                            );
                            acc = acc + (self.total_ratio - 1.0) * (delta as f64 / self.sr as f64)
                                - v22e as f64 / self.sr as f64;
                            self.cursor_prev = prev_abs;
                            cursor = cursor + delta + v22e;
                            self.last_transient = t;
                            in_pos = (in_pos + delta) % ring_tot;
                            fed_pos += delta;
                            self.advanced = true;
                        }
                    }
                    if !self.advanced {
                        if self.dec_flag {
                            let a9t = 2 * period
                                + (h2 as f32
                                    * (if self.total_ratio < 1.0 {
                                        self.total_ratio
                                    } else {
                                        1.0
                                    }) as f32) as i64;
                            let wpv = wrap_out(cursor, out_len);
                            self.do_ola(
                                fed_visible,
                                in_pos,
                                wpv,
                                0,
                                a9t as i32,
                                hop as i32,
                                true,
                                1.0,
                            );
                        } else {
                            let wpv = wrap_out(cursor, out_len);
                            self.do_ola(
                                fed_visible,
                                in_pos,
                                wpv,
                                win_max as i32,
                                (2 * period) as i32,
                                hop as i32,
                                false,
                                1.0,
                            );
                        }
                        acc = acc + (self.total_ratio - 1.0) * d13;
                        self.cursor_prev = cursor;
                        cursor += istep;
                    }
                    if !self.advanced && self.total_ratio >= 1.0 {
                        let mut w9 = 0i64;
                        if 2.0f32 * (acc as f32) > s11 {
                            loop {
                                if !(2.0 * self.total_ratio > w9 as f64) {
                                    break;
                                }
                                let wpv = wrap_out(cursor, out_len);
                                self.do_ola(
                                    fed_visible,
                                    in_pos,
                                    wpv,
                                    win_max as i32,
                                    (2 * period) as i32,
                                    hop as i32,
                                    false,
                                    1.0,
                                );
                                acc -= d13;
                                w9 += 1;
                                self.cursor_prev = cursor;
                                cursor += istep;
                                if w9 > 64 {
                                    break;
                                }
                                if !(2.0f32 * (acc as f32) > s11) {
                                    break;
                                }
                            }
                        }
                    }
                } else {
                    acc = acc + self.total_ratio * d13;
                }
                if self.total_ratio < 1.0 {
                    self.skip_next_318 = 2.0f32 * (acc as f32) < -s11;
                }
                self.n_granule += 1;
                self.progress
                    .store(self.n_granule, std::sync::atomic::Ordering::Relaxed);
                if self.trace {
                    eprintln!(
                        "G {} in={} cursor={} curprev={} fed={} acc={:.9} per={} win={} skip={} dec={} adv={} lt={} T={}",
                        self.n_granule - 1,
                        in_pos,
                        cursor,
                        self.cursor_prev,
                        fed_pos,
                        acc,
                        period,
                        win,
                        skip as i32,
                        self.dec_flag as i32,
                        self.advanced as i32,
                        self.last_transient,
                        self.forced_t
                    );
                }
                if !self.advanced {
                    in_pos = (in_pos + istep) % ring_tot;
                    fed_pos += istep;
                }
                self.advanced = false;
                self.cursor = cursor;
                self.acc = acc;
            }

            // ---- drain ----
            {
                let cp_s = as_u32(self.cursor_prev) as i64;
                let head = if cp_s > 100 { cp_s - 100 } else { 0 };
                let wp_acc = self.write_ptr_acc;
                let dd0 = head as f64 - wp_acc;
                let w9 = if dd0 > 0.0 { dd0 as i64 } else { 0 };
                if w9 != 0 {
                    let count = (w9 as f64 / self.pitch_ratio) as i64;
                    if count > 0 {
                        let mut ph = wp_acc % out_len as f64;
                        if ph < 0.0 {
                            ph += out_len as f64;
                        }
                        // per-channel source views over the logical output ring
                        let src: Vec<Vec<f32>> = (0..nch)
                            .map(|c| {
                                let base = c * (out_len as usize + 16);
                                self.out_ring[base..base + out_len as usize].to_vec()
                            })
                            .collect();
                        let dst = interp_nsamples(
                            &tbl,
                            &src,
                            out_len,
                            ph,
                            count as usize,
                            self.pitch_ratio,
                            self.pitch_ratio as f32,
                        );
                        let mut chunk = vec![0.0f32; count as usize * nch];
                        for i in 0..count as usize {
                            for c in 0..nch {
                                chunk[i * nch + c] = dst[c][i];
                            }
                        }
                        out_chunks.push(chunk);
                        out_n += count as usize;
                        self.write_ptr_acc = wp_acc + self.total_ratio * count as f64;
                    }
                }
            }

            if cursor_append >= as_u32(target_out as i64 + gate + 8 * hop) as i64 {
                break;
            }
            if fed_pos >= nstop && cursor_append >= nframes as i64 {
                break;
            }
        }

        self.cursor = cursor;
        self.in_pos_final = in_pos;
        self.acc = acc;
        let mut out = vec![0.0f32; out_n * nch];
        let mut off = 0usize;
        for c in out_chunks {
            out[off..off + c.len()].copy_from_slice(&c);
            off += c.len();
        }
        if self.gain_db != 0.0 {
            let m = 10.0f64.powf(self.gain_db / 20.0) as f32;
            for v in out.iter_mut() {
                *v *= m;
            }
        }
        out
    }
}

/// `td_octave_correct_peak` (`src/ops/td_octave.c`).
///
/// Returns `(peak, subharmonic_suspect, subharmonic_ratio)`.
pub fn octave_correct_peak(
    acf: &[f32],
    acf_len: usize,
    search_start: usize,
    search_end: usize,
    octave_enable: bool,
) -> Option<(usize, u32, f32)> {
    let mut ss = search_start;
    let mut se = search_end;
    if ss < 1 {
        ss = 1;
    }
    if se > acf_len - 2 {
        se = acf_len - 2;
    }
    if ss >= se {
        return None;
    }
    let mut p = ss;
    let mut max_val = acf[ss];
    for i in (ss + 1)..se {
        if acf[i] > max_val {
            max_val = acf[i];
            p = i;
        }
    }
    let mut sub_sus = 0u32;
    let mut sub_ratio = 0.0f32;
    if octave_enable {
        let mut cand_p2 = (p >> 1).wrapping_sub(1);
        if cand_p2 >= ss {
            let mut idx_p2 = cand_p2;
            let mut val_p2 = acf[idx_p2];
            if cand_p2 + 1 < acf_len && acf[cand_p2 + 1] > val_p2 {
                idx_p2 = p >> 1;
                val_p2 = acf[idx_p2];
            }
            if cand_p2 + 2 < acf_len && acf[cand_p2 + 2] > val_p2 {
                idx_p2 = cand_p2 + 2;
                val_p2 = acf[idx_p2];
            }
            let p15 = (3 * p) >> 1;
            let mut idx_p15 = p15 as i64 - 2;
            let mut val_p15 = if idx_p15 >= 0 && (idx_p15 as usize) < acf_len {
                acf[idx_p15 as usize]
            } else {
                -1000.0
            };
            for k in (p15 as i64 - 1)..(p15 as i64 + 3) {
                if k >= 0 && (k as usize) < acf_len && acf[k as usize] > val_p15 {
                    idx_p15 = k;
                    val_p15 = acf[k as usize];
                }
            }
            let _ = idx_p15;
            let r_p2 = val_p2.max(0.0).sqrt().sqrt();
            let r_p15 = val_p15.max(0.0).sqrt().sqrt();
            let sum_r = r_p2 + r_p15;
            let sum_r2 = sum_r * sum_r;
            let crit = 0.1f32 * (sum_r2 * sum_r2);
            if crit > acf[p] {
                p = idx_p2;
            }
            if acf[p] > 1e-12f32 {
                let ratio = crit / acf[p];
                if ratio > 0.1f32 && ratio < 1.0f32 {
                    sub_sus = 1;
                    sub_ratio = ratio;
                }
            }
            cand_p2 = (p >> 1).wrapping_sub(1);
        }
        if cand_p2 >= ss {
            let p_div3 = p / 3;
            let mut idx_p3 = p_div3 as i64 - 1;
            let mut val_p3 = if idx_p3 >= 0 && (idx_p3 as usize) < acf_len {
                acf[idx_p3 as usize]
            } else {
                -1000.0
            };
            if p_div3 < acf_len && acf[p_div3] > val_p3 {
                idx_p3 = p_div3 as i64;
                val_p3 = acf[p_div3];
            }
            if p_div3 + 1 < acf_len && acf[p_div3 + 1] > val_p3 {
                idx_p3 = p_div3 as i64 + 1;
                val_p3 = acf[p_div3 + 1];
            }
            let p_2div3 = (2 * p) / 3;
            let mut idx_2p3 = p_2div3 as i64 - 1;
            let mut val_2p3 = if idx_2p3 >= 0 && (idx_2p3 as usize) < acf_len {
                acf[idx_2p3 as usize]
            } else {
                -1000.0
            };
            if p_2div3 < acf_len && acf[p_2div3] > val_2p3 {
                idx_2p3 = p_2div3 as i64;
                val_2p3 = acf[p_2div3];
            }
            if p_2div3 + 1 < acf_len && acf[p_2div3 + 1] > val_2p3 {
                idx_2p3 = p_2div3 as i64 + 1;
                val_2p3 = acf[p_2div3 + 1];
            }
            let _ = idx_2p3;
            let r_p3 = val_p3.max(0.0).sqrt().sqrt();
            let r_2p3 = val_2p3.max(0.0).sqrt().sqrt();
            let sum_r3 = r_p3 + r_2p3;
            let sum_r3_2 = sum_r3 * sum_r3;
            let crit3 = 0.1f32 * (sum_r3_2 * sum_r3_2);
            if crit3 > acf[p] {
                p = idx_p3 as usize;
            }
        }
    }
    Some((p, sub_sus, sub_ratio))
}
