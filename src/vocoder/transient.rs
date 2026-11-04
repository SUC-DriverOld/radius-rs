//! Transient preservation subsystem — the part of `dvaaudiodsp.dll`'s vocoder
//! that `pyradius` never ported (and neither did this crate, until now).
//!
//! Reverse-engineered 2026-11-03 from the AU 2024 build; see
//! `docs/RE-dvaaudiodsp-radius-tempo.md` §13 for the full derivation. Three
//! pieces:
//!
//! * [`TransientQueue`] — the per-chunk transient score array the DLL keeps at
//!   `renderer+112` (= `state+696`, the "transient 增益事件输出" of the RE
//!   notes), with the three query primitives `sub_180153BD0` / `sub_180153CD0`
//!   / `sub_180153A70` ported exactly.
//! * [`TransientDetector`] — the input-side scorer `sub_180153EA0`. The DLL's
//!   exact windows/band tables are init-constant tables not yet captured, so
//!   this is a structural port: windowed FFT per 1024-frame chunk, 24 log
//!   bands, `x^(1/8)` compression, the ±4-chunk asymmetric flux comb
//!   (`sum(bands[t+1..=t+4]) − sum(bands[t−4..=t−1])`, negative terms × −0.1),
//!   normalised by the band count so the scheduler's `> 1.0` event threshold
//!   keeps the DLL's scale, and a non-maximum-suppression pass mirroring
//!   `sub_180153190`.
//! * [`TransientSched`] — the granule hop scheduler state machine inside
//!   `sub_180158D20` (fields `renderer+2448..2504`). It re-times the input hop
//!   and output step so granules *land* on detected attacks, reports the
//!   `state == 2` mode that gates the six chain branches (envelope IIR τ
//!   0.1→0.05, APC v15 1.0→0.5, RPT B2/B3, the fast noise-floor τ 0.01→0.005
//!   that feeds `b_710`), and hands RPT's B3 the score RMS trio the DLL takes
//!   from the queue.
//!
//! Everything here is opt-in: [`crate::vocoder::VocoderConfig::transient_sched`]
//! defaults to `false`, which keeps the engine byte-identical to the pyradius
//! parity corpus. With it on, output legitimately diverges from pyradius —
//! that is the point — and approaches AU, whose renders at tempo ≥ 100 run
//! this machinery (pyradius's bit-exact pure-pitch agreement with AU was only
//! ever possible because neither had it).

use crate::fft::with_plan;

/// `(int)(float)(x ± 0.5)` — the DLL's round-half-away-from-zero, computed in
/// f32 like the original `(int)(float)` casts.
#[inline]
fn rint_away(x: f64) -> i64 {
    let f = x as f32;
    let c = if f < 0.0 { -0.5f32 } else { 0.5f32 };
    (f + c) as i64
}

/// The event-distance ramp: 0 at ≤ 2 % of a second, 1 at ≥ 15 %, linear
/// between (`v858/v868/v16` in `sub_180158D20`).
#[inline]
fn dist_ramp(d_sec: f64) -> f64 {
    if d_sec <= 0.02 {
        0.0
    } else if d_sec >= 0.15 {
        1.0
    } else {
        (d_sec - 0.02) / 0.13
    }
}

// ---------------------------------------------------------------------------
// TransientQueue
// ---------------------------------------------------------------------------

/// The transient score array. Chunk `i` (absolute, un-compacted index) covers
/// input frames `[base + i*stride, base + (i+1)*stride)` and its score lives at
/// `scores[i - head]` once scored.
#[derive(Debug, Clone)]
pub struct TransientQueue {
    stride: i64,
    base: i64,
    head: i64,
    scores: Vec<f32>,
    cap: usize,
    // ---- `sub_1801525E0` ("0xab3") progressive refinement state ----
    /// `*((float*)a1+44)` — sample rate; 0 disables the refinement pipeline.
    sr: f32,
    /// `a1[70]` — pre-refinement score snapshot (the NMS confidence weight:
    /// every suppression pass keys on `refined * raw`, never raw alone).
    raw: Vec<f32>,
    /// `a1[3]`/`a1[67]` — post-sharpening snapshot (the 5-point argmax
    /// reference of the swap pass).
    snap: Vec<f32>,
    /// `*((_DWORD*)a1+226)` — absolute chunk index sharpened so far.
    refined_to: i64,
    /// `*((float*)a1+227)` — the sharpening IIR state (persists across calls).
    sharp_prev: f32,
    /// Chunks sharpened since the last cascade pass; the DLL runs the
    /// cascade once per process block (~8 chunks), not per chunk.
    since_cascade: i64,
}

impl TransientQueue {
    pub fn new(stride: i64, base: i64) -> Self {
        Self { stride: stride.max(1), base, head: 0, scores: Vec::new(), cap: 4096,
               sr: 0.0, raw: Vec::new(), snap: Vec::new(),
               refined_to: 0, sharp_prev: 0.0, since_cascade: 0 }
    }

    /// Construct with the sample rate that enables the `0xab3` refinement
    /// pipeline (`TransientQueue::new` leaves it at 0 = disabled, which the
    /// unit tests rely on for the raw query semantics).
    pub fn new_with_sr(stride: i64, base: i64, sr: u32) -> Self {
        let mut q = Self::new(stride, base);
        q.sr = sr as f32;
        q
    }

    pub fn stride(&self) -> i64 {
        self.stride
    }

    pub fn len(&self) -> usize {
        self.scores.len()
    }

    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }

    /// Score of absolute chunk `i`, or 0 outside the window.
    pub fn score_at(&self, i: i64) -> f32 {
        let j = i - self.head;
        if j < 0 || j >= self.scores.len() as i64 {
            0.0
        } else {
            self.scores[j as usize]
        }
    }

    /// Append the score for the next chunk; returns its absolute index.
    pub fn push(&mut self, score: f32) -> i64 {
        let i = self.head + self.scores.len() as i64;
        self.scores.push(score);
        self.raw.push(score);
        self.snap.push(score);
        self.compact();
        self.refine();
        i
    }

    /// Push the score of absolute chunk `i` (from the detector, whose first
    /// scoreable chunk is 4); earlier unscored chunks fill with 0.
    pub fn push_at(&mut self, i: i64, score: f32) {
        while self.head + (self.scores.len() as i64) < i {
            self.scores.push(0.0);
            self.raw.push(0.0);
            self.snap.push(0.0);
            self.compact();
        }
        let last = self.head + (self.scores.len() as i64) - 1;
        if i <= last {
            // Chunks are finalised in order; a late overwrite should not
            // happen, but a defensive write keeps the newest value.
            self.scores[(i - self.head) as usize] = score;
            self.raw[(i - self.head) as usize] = score;
            self.snap[(i - self.head) as usize] = score;
            return;
        }
        self.scores.push(score);
        self.raw.push(score);
        self.snap.push(score);
        self.compact();
        self.refine();
    }

    fn compact(&mut self) {
        if self.scores.len() > self.cap {
            let drop = self.scores.len() - self.cap;
            self.scores.drain(0..drop);
            self.raw.drain(0..drop);
            self.snap.drain(0..drop);
            self.head += drop as i64;
        }
    }

    /// `sub_1801525E0` — the DLL's progressive score refinement ("0xab3"),
    /// decoded 2026-11-04 from `.tmp/ida/refine_9chunk.c`. Runs in-place on
    /// the score store whenever new chunks land (the DLL runs it once per
    /// process block). Stages:
    ///
    /// **S1 sharpen** (`0x18015263b..0x1801527e0`): a persistent one-pole
    /// over the score timeline — `x'[i] = x[i] − 0.8·(a·x[i] + (1−a)·x'[i−1])`
    /// with `a = TimeToIirA(0.1, sr/stride)` and the state carried in
    /// `*((float*)a1+227)` across calls. DC gain 0.2: it de-means the score
    /// track and sharpens onset edges before anything else sees them.
    ///
    /// **S2 snapshot** (`0x1801527e2..`): the sharpened span is copied to a
    /// side buffer (`a1[3]`, our `snap`) that later stages read while the
    /// live buffer keeps being mutated.
    ///
    /// **S3 floor** (`0x1801528ca..`): `x ≤ 1/g → 0` over the new span, where
    /// `g = *((float*)a1+45)` is an (unknown) calibration — we take g = 1
    /// (sub-threshold scores never satisfy any downstream `> 1/thr` query).
    /// // TODO(0xab3): confirm `a1[45]` dynamically.
    ///
    /// **S4 swap** (`0x180152aa9..0x180152bd0`): for `i` in
    /// `[max(3, old−(rA+3)), new−3)`: if `live[i]` is above the floor, find
    /// the argmax of `snap[i−3..=i+1]` (ties → first) and, when it is not `i`
    /// itself, SWAP `live[argmax] ↔ live[i]` — the local peak of the frozen
    /// snapshot migrates onto the current index in the live buffer.
    ///
    /// **S5 cascade** — four suppression passes, each over the age band
    /// `[max(0, old−guard), new−guard)` (older chunks survive coarser passes),
    /// each killing `live[i]` iff some `k ∈ [i−w, i+w]` (clamped to
    /// `[0, new−1]`) has `live[k]·raw[k] > live[i]·factor·raw[i]`:
    ///
    /// | pass | window w | guard | factor |
    /// |------|----------|-------|--------|
    /// | A `0x1801529c5` | rA = ⌊0.01·sr/stride+0.5⌋ | rA | 1 |
    /// | B `0x180152bfc` | rA | 2·rA+6 | 1 |
    /// | C `0x180152cf8` | rC = ⌊0.025·sr/stride+0.5⌋ | rC+2·rA+6 | 2.7 |
    /// | D `0x180152df6` | rD = ⌊0.05·sr/stride+0.5⌋ | rD+rC+2·rA+6 | 6 |
    ///
    /// `raw` is the pre-refinement score — the product key makes suppression
    /// respect the ORIGINAL confidence, not the sharpened peak. @48k/128:
    /// rA=4, rC=9, rD=19 chunks.
    ///
    /// NOT ported (recorded): the two `sub_180153570` gain-curve calls
    /// (τ≈0.03/gain 3.0 and τ≈0.035/gain 1.0) and the final export into the
    /// `a1[0]` host-event buffer (fully-refined, floor-applied scores for the
    /// host event stream at lag `rD+rC+2·rA+6+4·⌊0.1·sr/stride+0.5⌋` chunks)
    /// — // TODO(0xab3). The machine's queries read the live buffer at rA
    /// lag, which this port preserves.
    fn refine(&mut self) {
        if self.sr <= 0.0 {
            return; // refinement disabled (unit-test construction)
        }
        let new = self.head + self.scores.len() as i64; // a1+224
        let old = self.refined_to;                     // a1+226
        if old >= new {
            return;
        }
        let hop = self.stride as f32; // a1+13: the recurrence advances one
        let sr = self.sr;             // chunk per step, so rate = sr/stride
        // S1 + S2: sharpen the new span, snapshotting post-sharpen values.
        let a = crate::vocoder::acs::time_to_iir_a(0.1, sr / hop);
        let mut prev = self.sharp_prev;
        for i in old..new {
            let j = (i - self.head) as usize;
            let x = self.scores[j];
            let y = x - (((x - prev) * a + prev) * 0.8);
            self.scores[j] = y;
            self.snap[j] = y;
            prev = y;
        }
        self.sharp_prev = prev;
        // S3: floor the new span.
        let floor = 1.0f32; // 1/a1[45] with a1[45] = 1 (see TODO above)
        for i in old..new {
            let j = (i - self.head) as usize;
            if floor >= self.scores[j] {
                self.scores[j] = 0.0;
            }
        }
        self.refined_to = new;
        self.since_cascade += new - old;

        // The DLL drives S4+S5 from its process block (~8 chunks/call at
        // 48k); running them per chunk would re-visit each index ~8× more
        // than the original and over-migrate the S4 swaps.
        if self.since_cascade < 8 {
            return;
        }
        self.since_cascade = 0;

        let rh = |t: f32| -> i64 { ((t * sr / hop) + 0.5) as i64 }; // (int)(x+0.5)
        let ra = rh(0.01);
        let rc = rh(0.025);
        let rd = rh(0.05);

        // S4: 5-point argmax swap on the fresh band.
        let head = self.head;
        let s_start = (old - (ra + 3)).max(3).max(head);
        let s_end = new - 3;
        let mut i = s_start;
        while i < s_end {
            let ji = (i - self.head) as usize;
            if self.scores[ji] > floor {
                let lo = (i - 3).max(head);
                let mut best = i;
                let mut best_v = f32::NEG_INFINITY;
                for k in lo..=i + 1 {
                    let jk = (k - self.head) as usize;
                    let v = self.snap[jk];
                    if v > best_v {
                        best_v = v;
                        best = k;
                    }
                }
                if best != i {
                    let jb = (best - self.head) as usize;
                    self.scores.swap(ji, jb);
                }
            }
            i += 1;
        }

        // S5: the four-pass suppression cascade.
        let sc = &mut self.scores;
        let raw = &self.raw;
        let mut pass = |window: i64, guard: i64, factor: f32| {
            let start = (old - guard).max(0).max(head);
            let end = new - guard;
            if end <= start {
                return;
            }
            for i in start..end {
                let ji = (i - head) as usize;
                let s = sc[ji];
                if floor > s {
                    continue;
                }
                let lo = (i - window).max(0);
                let hi = (i + window).min(new - 1);
                if lo > hi {
                    continue;
                }
                let key = s * factor * raw[ji];
                for k in lo..=hi {
                    let jk = (k - head) as usize;
                    let prod = if k < head || jk >= sc.len() {
                        0.0
                    } else {
                        sc[jk] * raw[jk]
                    };
                    if prod > key {
                        sc[ji] = 0.0;
                        break;
                    }
                }
            }
        };
        pass(ra, ra, 1.0);
        pass(ra, 2 * ra + 6, 1.0);
        pass(rc, rc + 2 * ra + 6, 2.7);
        pass(rd, rd + rc + 2 * ra + 6, 6.0);
    }

    /// Chunk index whose window contains `pos` (the DLL's `v14`).
    fn chunk_of(&self, pos: i64) -> i64 {
        (pos + self.stride / 2 - self.base).div_euclid(self.stride)
    }

    /// `sub_180153BD0` — first event (score > `1/thr`) at/after (`mode 1`) or
    /// before (`mode 2`) `pos`, scanning chunk offsets from `start`; `mode 0`
    /// accepts either direction. Returns the event position, or −1.
    pub fn query(&self, pos: i64, start: i64, thr: f32, mode: i32, cap: i64) -> i64 {
        let n = self.scores.len() as i64;
        let limit = n.min(cap);
        let v14 = self.chunk_of(pos);
        let thr_inv = 1.0f32 / thr;
        let mut k = start;
        if k >= limit {
            return -1;
        }
        loop {
            if mode != 2 {
                let idx = v14 + k - self.head;
                if idx >= 0 && idx < n && self.scores[idx as usize] > thr_inv {
                    return self.base + self.stride * (k + v14);
                }
            }
            if mode != 1 {
                let idx = v14 - self.head - k;
                if idx >= 0 && idx < n && self.scores[idx as usize] > thr_inv {
                    return self.base + self.stride * (v14 - k);
                }
            }
            k += 1;
            if k >= limit {
                return -1;
            }
        }
    }

    /// `sub_180153CD0` — position of the highest-scored chunk in
    /// `[pos, pos + range]`, or −1 if the range holds no chunk.
    pub fn query_max(&self, pos: i64, range: i64) -> i64 {
        let n = self.scores.len() as i64;
        let first = self.chunk_of(pos).max(self.head);
        let last = self.chunk_of(pos + range).min(self.head + n - 1);
        if first > last {
            return -1;
        }
        let mut best = first;
        let mut best_v = f32::NEG_INFINITY;
        for i in first..=last {
            let v = self.score_at(i);
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        // 2026-11-04 audit D34: a zero score is NOT a valid maximum — the
        // DLL returns "no event" (−1) when the window holds only
        // unsettled/zero-scored chunks; accepting zeros fed the compress
        // side a stream of phantom events on steady passages.
        if best_v <= 0.0 {
            return -1;
        }
        self.base + self.stride * best
    }

    /// `sub_180153A70` — RMS of the scores of the chunks within `range` frames
    /// of `pos` (used for RPT's B3 average trio).
    pub fn rms_around(&self, pos: i64, range: i64) -> f32 {
        let n = self.scores.len() as i64;
        let half = (range / self.stride) + ((range % self.stride) * 2 >= self.stride) as i64;
        let v6 = self.chunk_of(pos);
        let first = (v6 - half).max(self.head);
        let last = (v6 + half).min(self.head + n - 1);
        if first > last {
            return 0.0;
        }
        let mut acc = 0.0f32;
        for i in first..=last {
            let s = self.score_at(i);
            acc += s * s;
        }
        (acc / (last - first + 1) as f32).sqrt()
    }
}

// ---------------------------------------------------------------------------
// TransientDetector
// ---------------------------------------------------------------------------

/// Input-side transient scorer. Structural port of `sub_180153EA0`; the DLL's
/// own window/band tables are init constants still to be captured (§13.4), so
/// the band layout and FFT size are engineering choices fixed here. The
/// normalisation (flux / band count over `x^(1/8)`-compressed band energies)
/// follows the original, which is what makes the scheduler's `> 1.0`
/// threshold meaningful.
pub struct TransientDetector {
    stride: usize,
    n_fft: usize,
    win: Vec<f32>,
    bands: Vec<(usize, usize)>,
    tail: Vec<f32>,
    fed: i64,
    slots: std::collections::VecDeque<Vec<f32>>,
    n_bands: usize,
    /// Scores not yet through non-maximum suppression (index offset from
    /// `pending_base`).
    pending: Vec<f32>,
    pending_base: i64,
    next_chunk: i64,
}

impl TransientDetector {
    pub fn new(sr: u32) -> Self {
        // Dynamically confirmed against the running AU engine (x64dbg,
        // 2026-11-03): the score queue's stride field (renderer+112+52) is
        // **128 frames** — a 2.7 ms chunk at 48 kHz, so the ±4-chunk comb
        // spans ±10.7 ms around an attack. The FFT window spans four chunks.
        let stride = 128usize;
        let n_fft = 512usize;
        let win: Vec<f32> = (0..n_fft)
            .map(|i| {
                (std::f64::consts::PI * (i as f64) / (n_fft as f64)).sin() as f32
            })
            .collect();
        // 24 log-spaced bands from 40 Hz to Nyquist (edges in FFT bins).
        let nyq = sr as f64 / 2.0;
        let f_lo = 40.0f64;
        let n_b = 24usize;
        let bin_hz = sr as f64 / n_fft as f64;
        let mut bands = Vec::with_capacity(n_b);
        for k in 0..n_b {
            let f0 = f_lo * (nyq / f_lo).powf(k as f64 / n_b as f64);
            let f1 = f_lo * (nyq / f_lo).powf((k + 1) as f64 / n_b as f64);
            let b0 = ((f0 / bin_hz).round() as usize).max(1);
            let b1 = ((f1 / bin_hz).round() as usize).max(b0 + 1).min(n_fft / 2);            bands.push((b0, b1));
        }
        Self {
            stride,
            n_fft,
            win,
            bands,
            tail: vec![0.0; n_fft],
            fed: 0,
            slots: std::collections::VecDeque::with_capacity(9),
            n_bands: n_b,
            pending: Vec::new(),
            pending_base: 0,
            next_chunk: 0,
        }
    }

    /// Feed `n` mono frames; returns (chunk index, score) pairs finalised by
    /// this call, in order. Band arrays are computed the moment a chunk's
    /// audio completes (the window is the `n_fft` frames ending at the chunk
    /// end); the ±4-flux score follows four chunks later from the 9-slot ring,
    /// and non-maximum suppression delays each value by a further 2 chunks —
    /// all in ascending chunk order, so the queue stays contiguous.
    pub fn push(&mut self, x: &[f32]) -> Vec<(i64, f32)> {
        let stride = self.stride as i64;
        for &v in x {
            self.tail.push(v);
        }
        let keep = self.n_fft + self.stride;
        if self.tail.len() > keep {
            let drop = self.tail.len() - keep;
            self.tail.drain(0..drop);
        }
        self.fed += x.len() as i64;

        let mut out = Vec::new();
        // 1) band arrays for every chunk whose audio just completed.
        while self.fed >= (self.next_chunk + 1) * stride {
            self.next_chunk += 1;
            let end = self.tail.len();
            let start = end - self.n_fft;
            let mut frame: Vec<f32> = (0..self.n_fft)
                .map(|i| self.tail[start + i] * self.win[i])
                .collect();
            let mut spec = vec![0.0f32; self.n_fft + 2];
            with_plan(self.n_fft, |p| p.fwd_into(&frame, &mut spec));
            frame.clear();
            let mut bands = vec![0.0f32; self.n_bands];
            for (k, &(b0, b1)) in self.bands.iter().enumerate() {
                let mut e = 0.0f32;
                for b in b0..b1 {
                    let re = spec[2 * b];
                    let im = spec[2 * b + 1];
                    e += re * re + im * im;
                }
                // x^(1/8): three square roots, as in the DLL.
                bands[k] = e.sqrt().sqrt().sqrt();
            }
            self.slots.push_back(bands);
            while self.slots.len() > 9 {
                self.slots.pop_front();
            }
            // 2) once nine slots are queued (chunks t-4..t+4), score the
            //    middle one.
            if self.slots.len() == 9 {
                let mut acc = 0.0f32;
                for k in 0..self.n_bands {
                    let mut comb = 0.0f32;
                    for j in [5usize, 6, 7, 8] {
                        comb += self.slots[j][k];
                    }
                    for j in [0usize, 1, 2, 3] {
                        comb -= self.slots[j][k];
                    }
                    if comb <= 0.0 {
                        acc -= comb * 0.1;
                    } else {
                        acc += comb;
                    }
                }
                let score = acc / self.n_bands as f32;
                self.pending.push(score);
                // NMS with a ±2 window: a neighbour carrying ≥ 1/0.7 of this
                // score kills it (`sub_180153190`, factor 0.7). The newest
                // score is index len-1; emit the one two places back, whose
                // window is now complete on both sides.
                if self.pending.len() >= 5 {
                    let i = self.pending.len() - 5;
                    let done = self.pending_base + i as i64;
                    let s = self.pending[i];
                    let mut alive = s;
                    // 2026-11-04 audit D30: the DLL applies suppression to
                    // every score, not only those above 1.0.
                    for d in [-2i64, -1, 1, 2] {
                        let j = i as i64 + d;
                        if j >= 0 && (j as usize) < self.pending.len() {
                            let nb = self.pending[j as usize];
                            if nb * 0.7 > s {
                                alive = 0.0;
                                break;
                            }
                        }
                    }
                    out.push((done, alive));
                }
                // Keep the pending window bounded: entries through len-5 are
                // final; the last four may still gain neighbours.
                if self.pending.len() > 8 {
                    let keep = self.pending.len() - 4;
                    self.pending.drain(0..keep);
                    self.pending_base += keep as i64;
                }
            }
        }
        // The pending stream's index 0 is chunk 4 (the first chunk with a full
        // ±4 comb); the emitted stream indices are relative to that.
        for (t, _s) in out.iter_mut() {
            *t += 4;
        }
        out
    }

    /// Flush remaining pending scores (end of stream).
    pub fn finalize(&mut self) -> Vec<(i64, f32)> {
        let mut out = Vec::new();
        while self.pending.len() >= 3 {
            let i = self.pending.len() - 3;
            let done = self.pending_base + i as i64;
            let s = self.pending[i];
            let mut alive = s;
            if s > 1.0 {
                for d in [-2i64, -1, 1, 2] {
                    let j = i as i64 + d;
                    if j >= 0 && (j as usize) < self.pending.len() {
                        if self.pending[j as usize] * 0.7 > s {
                            alive = 0.0;
                            break;
                        }
                    }
                }
            }
            out.push((done, alive));
            self.pending.drain(0..i + 1);
            self.pending_base += i as i64 + 1;
        }
        out
    }

    /// Frames that must be fed before chunk `t`'s score exists: 4 comb chunks
    /// plus the 2-chunk NMS window, all in `stride`-frame units.
    pub fn frames_needed(&self, t: i64) -> i64 {
        ((t + 7) * self.stride as i64 - self.fed).max(0)
    }
}

// ---------------------------------------------------------------------------
// TransientSched
// ---------------------------------------------------------------------------

/// One granule's scheduler decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SchedOut {
    /// Input hop for this granule (DLL `v864` / our `granule_1400`).
    pub in_hop: i64,
    /// Output step for this granule (DLL `v853`, our out-step override).
    pub out_step: i64,
    /// Whether the DLL's `state == 2` processing mode is active (gates the six
    /// chain branches this granule).
    pub state2: bool,
    /// Current machine state (0..3), for diagnostics.
    pub state: i32,
    /// One-shot: this granule is where the state-2 hold counter crossed
    /// HALF the hold target — the DLL clears `renderer+3568` here (§13.2,
    /// the second clear site alongside the B3-accept one).
    pub mid_hold_clear: bool,
}

/// The hop scheduler state machine of `sub_180158D20` (fields
/// `renderer+2448..2504`). Field names carry the renderer offsets.
pub struct TransientSched {
    pub state: i32,        // 2448
    pub max_w: i32,        // 2452
    /// The machine's steady granule hop (`renderer+1368`, ~705 frames at
    /// 48 kHz / slope 1.19) and out-step (`+1372`), from `sub_180157BC0`.
    pub steady_hop: i64,
    pub steady_step: i64,
    pub hold_target: i32,  // 2456
    pub hold_cnt: i32,     // 2460
    /// One-shot latch for the state-2 half-hold `+3568` clear (§13.2):
    /// `mid_cleared` arms at state-2 entry, fires once when the counter
    /// crosses half the target, `mid_cleared_this_step` carries the pulse
    /// to the next `SchedOut`.
    mid_cleared: bool,
    mid_cleared_this_step: bool,
    pub appr_target: i32,  // 2464
    pub appr_cnt: i32,     // 2468
    pub appr_in_hop: i32,  // 2472
    pub appr_out_step: i32,// 2476
    pub appr_hop_inc: i32, // 2480
    pub settle_hop_inc: i32, // 2484
    pub trans_hold: i32,   // 2488
    pub trans_in_hop: i32, // 2492
    pub trans_cnt: i32,    // 2496
    pub trans_out_step: i32, // 2500
    pub trans_hop_rem: i32,  // 2504
    pub last_event: i64,   // 1360
    /// Granules spent outside state 0 (diagnostics).
    pub engaged_granules: u64,
    /// (inpos, state, query result) per granule, capped; diagnostics only.
    pub trace: Vec<[i64; 3]>,
    /// Enable `trace` recording (diagnostics/tests).
    pub trace_enabled: bool,
}

/// C-style truncated division with the sign of the dividend (the DLL's
/// `v90 / v84`, `v90 % v84` pair).
fn c_div_rem(a: i64, b: i64) -> (i64, i64) {
    let q = a / b;
    let r = a % b;
    (q, r)
}

impl TransientSched {
    pub fn new() -> Self {
        Self {
            state: 0,
            max_w: 2,
            steady_hop: 705,
            steady_step: 838,
            hold_target: 1,
            hold_cnt: 0,
            mid_cleared: false,
            mid_cleared_this_step: false,
            appr_target: 1,
            appr_cnt: 0,
            appr_in_hop: 1,
            appr_out_step: 1,
            appr_hop_inc: 0,
            settle_hop_inc: 0,
            trans_hold: 0,
            trans_in_hop: 1,
            trans_cnt: 0,
            trans_out_step: 1,
            trans_hop_rem: 0,
            last_event: -1,
            engaged_granules: 0,
            trace: Vec::new(),
            trace_enabled: false,
        }
    }

    /// `sub_180157BC0`'s geometry laws, full divisor table (2026-11-04 audit
    /// D12): `base = round(n_write*0.7/D)` with `D = {3,3,6}[min(q-10,2)]`
    /// when `q >= 10`, else `D = 4 << q`; then
    /// `hop = round(base/slope)` (~705 at 48 kHz, slope 1.19, q=37 — a ~15 ms
    /// granule, ~3 of our sub-granules, NOT the 3592-frame schedule hop),
    /// `step = round(hop*slope)` and
    /// `max_w = round(round(n_write*0.7/hop) * f * max(sqrt(slope), 1))` with
    /// `f = 1.4` (q>=10) / `0.7`.
    pub fn set_geometry(&mut self, n_write: i64, slope: f64, quality: i32) {
        let (div, f) = if quality >= 10 {
            let d = match (quality - 10).min(2) {
                0 | 1 => 3.0f64,
                _ => 6.0,
            };
            (d, 1.4f64)
        } else {
            ((4 << quality) as f64, 0.7f64)
        };
        let hop = rint_away(rint_away(n_write as f64 * 0.7 / div) as f64 / slope).max(1);
        self.steady_hop = hop;
        self.steady_step = rint_away(hop as f64 * slope);
        let span = rint_away(n_write as f64 * 0.7 / hop as f64);
        let g = slope.sqrt().max(1.0);
        self.max_w = rint_away(span as f64 * f * g).max(1) as i32;
    }

    /// Advance the machine one granule. `inpos` is the input read position and
    /// `curpos` the output write position (both in frames); `hop`/`step` are
    /// this granule's steady schedule values. The time map is the fixed-slope
    /// `out = in * slope` our engine runs (the DLL's ramp anchors collapse to
    /// that when nothing changes mid-stream).
    pub fn step(
        &mut self,
        q: &TransientQueue,
        inpos: i64,
        curpos: i64,
        slope: f64,
        sr: i64,
        hop: i64,
        step: i64,
    ) -> SchedOut {
        let mut v30 = hop;
        let mut v31 = step;
        match self.state {
            1 => {
                // ---- case 1: approach ----
                self.appr_cnt += 1;
                if self.appr_cnt < self.appr_target {
                    // stay in 1
                } else {
                    self.state = 2;
                    self.hold_cnt = 0;
                    self.mid_cleared = false;
                    v30 = self.appr_in_hop as i64 + self.appr_hop_inc as i64;
                }
                v31 = self.appr_out_step as i64;
                if v30 < 1 {
                    v30 = 1;
                }
            }
            2 => {
                // ---- case 2: settle / transient processing ----
                self.hold_cnt += 1;
                // §13.2 (L527): when 2460 reaches half of 2456, renderer+3568
                // clears — a second clear site alongside the B3-accept one,
                // active during the state-2 plateau. One-shot per entry.
                if !self.mid_cleared && self.hold_cnt * 2 >= self.hold_target {
                    self.mid_cleared = true;
                    self.mid_cleared_this_step = true;
                }
                let reached = self.hold_cnt >= self.hold_target;
                if !reached {
                    // stay in 2
                } else if slope > 1.0 && self.trans_hold > 0 {
                    self.state = 3;
                    self.trans_cnt = 0;
                } else {
                    self.state = 0;
                }
                // 2026-11-04 audit D48: the DLL's steady out-step is
                // renderer+1372 = round(hop*slope), NOT the input hop.
                v31 = step;
                if reached {
                    v30 = hop + self.settle_hop_inc as i64;
                    // Event re-query at the boundary (the 0x1801592e8 block):
                    // a fresh attack inside round(sr*0.1) restarts the cycle.
                    let win = rint_away(sr as f64 * 0.1);
                    let ev = q.query(inpos, 1, 1.0, 1, 2000);
                    if slope > 1.0
                        && ev != -1
                        && ev != self.last_event
                        && ev - inpos < win
                    {
                        self.last_event = ev;
                        let prev_ev = q.query(ev, 1, 1.0, 2, 2000);
                        let d_prev = if prev_ev == -1 { sr } else { (ev - prev_ev).abs() };
                        let vmin = ((ev - inpos) as f64 * 0.9).min(d_prev as f64 * 0.8);
                        let smin = slope.min(1.0);
                        let pre_n =
                            rint_away(vmin / hop as f64 * smin).clamp(1, self.max_w as i64);
                        let next_ev = q.query(ev, 1, 1.0, 1, 2000);
                        let d_next = if next_ev == -1 { sr } else { (ev - next_ev).abs() };
                        let r = dist_ramp(d_next as f64 / sr as f64);
                        let hold2 = rint_away((r * 0.9 + 0.1) * self.max_w as f64)
                            .clamp(1, self.max_w as i64);
                        self.hold_target = rint_away((pre_n + hold2) as f64 * 0.5) as i32;
                        self.state = 1;
                        self.appr_cnt = 0;
                        let back = (pre_n as f64 * hop as f64) * 0.5;
                        let mid_out = ev as f64 * slope - back;
                        let mid_in = ev as f64 - back;
                        let v69 = mid_out - curpos as f64;
                        self.appr_target = rint_away(v69 / step as f64).max(1) as i32;
                        self.appr_out_step = rint_away(v69 / self.appr_target as f64) as i32;
                        self.appr_in_hop =
                            rint_away((mid_in - inpos as f64) / self.appr_target as f64) as i32;
                        self.appr_hop_inc = 0;
                        self.trans_out_step = self.appr_out_step;
                        let v80 = hop as f64 * self.hold_target as f64;
                        let v81 = v80 + mid_out;
                        let v82 = v80 + mid_in;
                        let v83 = (v81 - ev as f64 * slope) / slope + ev as f64;
                        let denom = self.appr_in_hop as f64 - self.appr_out_step as f64 / slope;
                        self.trans_hold = if denom != 0.0 {
                            let v84 = ((v83 - v82) / denom) as i32;
                            if v84 < 1 { 1 } else { v84 }
                        } else {
                            1
                        };
                        let v87 = rint_away(
                            self.appr_out_step as f64 * self.trans_hold as f64 / slope + v83,
                        );
                        let v88 = self.appr_in_hop as f64 * self.trans_hold as f64 + v82;
                        let v90 = v87 - rint_away(v88);
                        let (q_, r_) = c_div_rem(v90, self.trans_hold as i64);
                        self.trans_in_hop = self.appr_in_hop + q_ as i32;
                        self.trans_hop_rem = r_ as i32;
                    }
                }
            }
            3 => {
                // ---- case 3: transient replay ----
                self.trans_cnt += 1;
                let done = self.trans_cnt >= self.trans_hold;
                if !done {
                    // stay in 3
                } else {
                    self.state = 0;
                }
                v30 = self.trans_in_hop as i64;
                v31 = self.trans_out_step as i64;
                if done {
                    v30 += self.trans_hop_rem as i64;
                }
                if v30 < 1 {
                    v30 = 1;
                }
            }
            _ => {
                // ---- state 0: steady, run detection ----
                if slope > 1.0 {
                    let win = rint_away(sr as f64 * 0.05);
                    let ev = q.query(inpos, 1, 1.0, 1, 2000);
                    let min_pos = (sr as f64 * 0.01) as i64;
                    if ev != -1
                        && ev > min_pos
                        && ev != self.last_event
                        && ev - inpos < win
                    {
                        self.last_event = ev;
                        let next_ev = q.query(ev, 1, 1.0, 1, 2000);
                        let d_next = if next_ev == -1 { sr } else { (ev - next_ev).abs() };
                        let r_next = dist_ramp(d_next as f64 / sr as f64);
                        let hold_next = rint_away((r_next * 0.9 + 0.1) * self.max_w as f64)
                            .clamp(1, self.max_w as i64);
                        let prev_ev = q.query(ev, 1, 1.0, 2, 2000);
                        let d_prev = if prev_ev == -1 { sr } else { (ev - prev_ev).abs() };
                        let r_prev = dist_ramp(d_prev as f64 / sr as f64);
                        let hold_prev = rint_away((r_prev * 0.9 + 0.1) * self.max_w as f64)
                            .clamp(1, self.max_w as i64);
                        self.hold_target =
                            rint_away((hold_prev + hold_next) as f64 * 0.5) as i32;
                        self.state = 1;
                        self.appr_cnt = 0;
                        let back = (hop as f64 * hold_prev as f64) * 0.5;
                        let mid_out = ev as f64 * slope - back;
                        let mid_in = ev as f64 - back;
                        let v122 = mid_out - curpos as f64;
                        self.appr_target = rint_away(v122 / step as f64).max(1) as i32;
                        self.appr_out_step = rint_away(v122 / self.appr_target as f64) as i32;
                        self.appr_in_hop =
                            rint_away((mid_in - inpos as f64) / self.appr_target as f64) as i32;
                        self.appr_hop_inc = 0;
                        self.trans_out_step = self.appr_out_step;
                        self.trans_hold = self.appr_target;
                        let v132 = hop as f64 * self.hold_target as f64;
                        let v133 = v132
                            + 2.0 * self.appr_target as f64 * self.appr_in_hop as f64;
                        let v134 = (v132
                            + 2.0 * self.appr_target as f64 * self.appr_out_step as f64)
                            / slope;
                        let v136 = rint_away(v134) - v133 as i64;
                        let (q_, r_) = c_div_rem(v136, self.appr_target as i64);
                        self.trans_in_hop = self.appr_in_hop + q_ as i32;
                        self.trans_hop_rem = r_ as i32;
                    }
                } else {
                    // Compression side (slope ≤ 1): two-stage, exactly as in
                    // the DLL. Stage 1 finds a candidate peak and derives the
                    // hold target from the distance to the nearest event
                    // (mode-0 query); **2456 is written even if the second
                    // stage rejects** — the stale value feeds the next
                    // granule's stage-2 range.
                    let range = rint_away(
                        (hop as f64 * 0.5 * self.max_w as f64 + step as f64) / slope,
                    );
                    let cand = q.query_max(inpos, range);
                    if cand != -1 {
                        let near = q.query(cand, 1, 1.0, 0, 2000);
                        let d = if near == -1 { sr } else { (cand - near).abs() };
                        let r = dist_ramp(d as f64 / sr as f64);
                        self.hold_target = rint_away((r * 0.9 + 0.1) * self.max_w as f64)
                            .clamp(1, self.max_w as i64) as i32;
                        let range2 = rint_away(
                            hop as f64 * 0.5 * self.hold_target as f64 + step as f64 / slope,
                        );
                        let best = q.query_max(inpos, range2);
                        let min_pos = (sr as f64 * 0.009_999_999_8) as i64;
                        if best != -1 && best != self.last_event && best > min_pos {
                            self.last_event = best;
                            self.state = 2;
                            self.hold_cnt = 0;
                            let v156 = ((best - inpos) as f64 * (slope + slope)
                                / hop as f64) as i64;
                            let v157 = v156.max(1) as i32;
                            self.hold_target = v157;
                            let back = (hop as f64 * v157 as f64) * 0.5;
                            let v161 = best as f64 - back;
                            let v160 = best as f64 * slope - back;
                            v30 = rint_away(v161 - inpos as f64);
                            v31 = rint_away(v160 - curpos as f64);
                            self.settle_hop_inc = rint_away(
                                (hop as f64 * v157 as f64 + v160 - best as f64 * slope) / slope
                                    + best as f64
                                    - (hop as f64 * v157 as f64 + v161),
                            ) as i32;
                        }
                    }
                }
            }
        }
        if self.state != 0 {
            self.engaged_granules += 1;
        }
        if self.trace_enabled && self.trace.len() < 8192 {
            let ev = if self.state == 0 {
                if slope > 1.0 {
                    q.query(inpos, 1, 1.0, 1, 2000)
                } else {
                    -2
                }
            } else {
                self.last_event
            };
            self.trace.push([inpos, self.state as i64, ev]);
        }
        let mid_hold_clear = self.mid_cleared_this_step;
        self.mid_cleared_this_step = false;
        SchedOut {
            in_hop: v30,
            out_step: v31,
            state2: self.state == 2,
            state: self.state,
            mid_hold_clear,
        }
    }
}

impl Default for TransientSched {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn q_with(stride: i64, base: i64, scores: &[f32]) -> TransientQueue {
        let mut q = TransientQueue::new(stride, base);
        for &s in scores {
            q.push(s);
        }
        q
    }

    #[test]
    fn refine_0xab3_floors_subthreshold_and_nms_twin_peaks() {
        // sr = 48000, stride = 128 → rA = 4, rC = 9, rD = 19 chunk windows.
        let mut q = TransientQueue::new_with_sr(128, 0, 48_000);
        let mut scores = vec![0.5f32; 30]; // sub-floor bed
        scores[10] = 5.0; // twin peaks 3 chunks apart — inside rA of each other
        scores[13] = 5.0;
        for (i, &s) in scores.iter().enumerate() {
            q.push_at(i as i64, s);
        }
        // S3 floor: the far bed (outside the S4 migration band 6..18, where
        // the swap pass may legitimately move a peak onto a bed chunk) must
        // be zero.
        for i in (0..6).chain(18..30) {
            assert_eq!(
                q.score_at(i),
                0.0,
                "sub-floor chunk {i} must floor to zero"
            );
        }
        // Law-level assertions on this adversarial input: the dominant twin
        // keeps most of its score somewhere in the migration band (S4 moves
        // peaks toward the current index), and the two original spike
        // positions cannot BOTH stay dominant — pass A's product key (with
        // the sharpening having attenuated whichever spike followed the
        // other) resolves the pair. Migrated copies pairing with bed-level
        // raw weights (product ≈ 0.6) may legitimately survive: that is the
        // product-key semantics, not a bug.
        let band: Vec<f32> = (6..18).map(|i| q.score_at(i)).collect();
        let mx = band.iter().copied().fold(0.0f32, f32::max);
        assert!(mx > 4.0, "dominant twin survives strongly, band={band:?}");
        let dominant = band.iter().copied().filter(|&v| v >= 4.0).count();
        assert_eq!(
            dominant, 1,
            "two equal spikes cannot both stay dominant anywhere, band={band:?}"
        );
    }

    #[test]
    fn refine_0xab3_disabled_without_sr() {
        // TransientQueue::new (sr = 0) keeps the raw scores: the classic
        // query semantics the rest of the unit tests rely on.
        let mut q = TransientQueue::new(128, 0);
        for i in 0..30 {
            q.push_at(i, 0.5);
        }
        assert_eq!(q.score_at(7), 0.5);
    }

    #[test]
    fn query_forward_finds_first_above_threshold() {
        // chunks 0..8 at stride 1024, base 0; event at chunk 4.
        let scores = [0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0];
        let q = q_with(1024, 0, &scores);
        // pos 0 → chunk 0; first event forward from offset 1 → chunk 1? no,
        // event at chunk 4 → position 4096.
        assert_eq!(q.query(0, 1, 1.0, 1, 2000), 4 * 1024);
        // querying past the event finds nothing.
        assert_eq!(q.query(5 * 1024, 1, 1.0, 1, 2000), -1);
    }

    #[test]
    fn query_backward_mode_2() {
        let scores = [0.0, 0.0, 3.0, 0.0, 0.0];
        let q = q_with(512, 1000, &scores);
        // pos 3*512 → chunk ~4 (with base 1000 the arithmetic shifts; just
        // check we land on the chunk holding the 3.0, i.e. position base+2*512)
        let ev = q.query(4 * 512 + 1000, 1, 1.0, 2, 2000);
        assert_eq!(ev, 1000 + 2 * 512);
    }

    #[test]
    fn query_max_picks_strongest_in_range() {
        let scores = [0.5, 2.0, 0.6, 3.0, 0.1];
        let q = q_with(1024, 0, &scores);
        assert_eq!(q.query_max(0, 5 * 1024), 3 * 1024);
        assert_eq!(q.query_max(0, 2 * 1024), 1 * 1024);
    }

    #[test]
    fn rms_around_centres_on_pos() {
        let scores = [1.0, 2.0, 3.0, 2.0, 1.0];
        let q = q_with(1024, 0, &scores);
        // range 1024 → half = 1 chunk (the DLL's trunc(range/stride)): chunks
        // 1..=3 around pos 2*1024.
        let r = q.rms_around(2 * 1024, 1024);
        let expect = ((4.0f32 + 9.0 + 4.0) / 3.0).sqrt();
        assert!((r - expect).abs() < 1e-5, "{r} vs {expect}");
    }

    #[test]
    fn compaction_shifts_head() {
        let mut q = TransientQueue::new(100, 0);
        q.cap = 4;
        for i in 0..6 {
            q.push(i as f32);
        }
        // head moved by 2; score_at(5) == 5.0, score_at(0) == 0 (dropped).
        assert_eq!(q.head, 2);
        assert_eq!(q.score_at(5), 5.0);
        assert_eq!(q.score_at(0), 0.0);
    }

    #[test]
    fn detector_scores_a_click_above_threshold() {
        let mut det = TransientDetector::new(48_000);
        // silence, then a sharp broadband burst, then silence. The comb fires
        // one chunk *before* the burst's chunk (B(t+1..=t+4) − B(t−4..=t−1)),
        // which is the pre-roll the scheduler wants.
        let mut x = vec![0.0f32; 36 * 512];
        for i in 0..512 {
            x[4096 + i] = ((i % 2) as f32 * 2.0 - 1.0) * 0.95;
        }
        let mut best = 0.0f32;
        let mut best_chunk = -1i64;
        for chunk in x.chunks(512) {
            for (t, s) in det.push(chunk) {
                if s > best {
                    best = s;
                    best_chunk = t;
                }
            }
        }
        assert!(best > 1.0, "burst should score above the event threshold, got {best}");
        // Burst starts at frame 4096 = chunk 32; the comb fires a chunk or
        // two ahead of it (the DLL's pre-roll, now 128-frame chunks).
        assert!(
            (29..=34).contains(&best_chunk),
            "score should fire at the burst chunk, got {best_chunk}"
        );
    }

    #[test]
    fn detector_ignores_flat_signal() {
        let mut det = TransientDetector::new(48_000);
        let mut best = 0.0f32;
        // constant tone: no positive flux, compressed values identical.
        let x: Vec<f32> = (0..16384).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        for chunk in x.chunks(512) {
            for (_t, s) in det.push(chunk) {
                best = best.max(s);
            }
        }
        assert!(best <= 1.0, "steady tone must not fire, got {best}");
    }

    #[test]
    fn sched_enters_approach_then_replay_on_event() {
        // Event at chunk 2 (position 2048): the steady-state query window is
        // round(sr*0.05) = 2400 frames, so the first granule (inpos 0) sees it.
        let mut q = q_with(1024, 0, &[0.0, 0.0, 2.5]);
        let mut s = TransientSched::new();
        s.set_geometry(7184, 1.5, 37);
        let mut states = Vec::new();
        let mut inpos = 0i64;
        let mut curpos = 0i64;
        // max_w is 15 at this geometry, so the full cycle needs ~20 granules.
        for _ in 0..24 {
            let out = s.step(&q, inpos, curpos, 1.5, 48000, 3592, 5388);
            states.push(out.state);
            inpos += out.in_hop;
            curpos += out.out_step;
        }
        assert!(states.contains(&1), "must pass through approach: {states:?}");
        assert!(states.contains(&2), "must reach settle: {states:?}");
        assert!(states.contains(&3), "must reach transient replay: {states:?}");
        assert_eq!(*states.last().unwrap(), 0, "returns to steady: {states:?}");
        assert!(s.engaged_granules > 0);
    }

    #[test]
    fn sched_mid_hold_clear_fires_once_at_half() {
        // The 13.2 second +3568 clear site: exactly one granule in the
        // state-2 plateau pulses mid_hold_clear, when hold_cnt crosses half
        // the hold target — and never on B3-accept granules (that clear is
        // a separate site in vocoder.rs).
        let mut q = q_with(1024, 0, &[0.0, 0.0, 2.5]);
        let mut s = TransientSched::new();
        s.set_geometry(7184, 1.5, 37);
        let mut inpos = 0i64;
        let mut curpos = 0i64;
        let mut pulses = Vec::new();
        for _ in 0..24 {
            let out = s.step(&q, inpos, curpos, 1.5, 48000, 3592, 5388);
            if out.mid_hold_clear {
                pulses.push((out.state, s.hold_cnt, s.hold_target));
            }
            inpos += out.in_hop;
            curpos += out.out_step;
        }
        assert_eq!(pulses.len(), 1, "exactly one pulse: {pulses:?}");
        assert_eq!(pulses[0].0, 2, "pulse lands inside state 2");
        let (_, cnt, target) = pulses[0];
        assert!(
            cnt * 2 >= target && (cnt - 1) * 2 < target,
            "at the half-crossing: cnt={cnt} target={target}"
        );
    }

    #[test]
    fn sched_stays_steady_without_events() {
        let q = q_with(1024, 0, &[0.0; 20]);
        let mut s = TransientSched::new();
        s.set_geometry(7184, 1.5, 37);
        for g in 0..8 {
            let out = s.step(&q, g * 3592, g * 5388, 1.5, 48000, 3592, 5388);
            assert_eq!(out.state, 0);
            assert_eq!(out.in_hop, 3592);
            assert_eq!(out.out_step, 5388);
            assert!(!out.state2);
        }
    }

    #[test]
    fn max_w_law_matches_dll_formula() {
        let mut s = TransientSched::new();
        // 48 kHz: n_write 7184, slope 1.19 (q≥10, divisor 6):
        // hop = round(round(7184*0.7/6)/1.19) = round(838/1.19) = 704,
        // step = round(704*1.19) = round(837.76) = 838, span = round(7184*0.7/704) = 7,
        // max_w = round(7*1.4*sqrt(1.19)) = round(10.69) = 11.
        s.set_geometry(7184, 1.19, 37);
        assert_eq!(s.steady_hop, 704);
        assert_eq!(s.steady_step, 838);
        assert_eq!(s.max_w, 11);
        // slope < 1 clamps the sqrt factor at 1: hop 1117, span 5, max_w 7.
        s.set_geometry(7184, 0.75, 37);
        assert_eq!(s.steady_hop, 1117);
        assert_eq!(s.max_w, 7);
    }
}
