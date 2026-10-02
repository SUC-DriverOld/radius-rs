//! `TransientsInfo` (`libradius/src/ops/transients_info.c`, driven by
//! `pyradius.td_core.TIState`).
//!
//! Kaiser-windowed STFT at two resolutions, an IIR-shaped magnitude envelope,
//! three progressive max-filter passes, a 7-point argmax swap on the pool
//! vector and the final `v00` threshold pass. `rx_td_transient_pos` then reads
//! the argmax of a frame window.

use crate::fft::with_plan;
use crate::tables::ti_kaiser;

/// 24 log-spaced band edges used by the coarse vector.
pub const TI_BOUNDS: [usize; 25] = [
    0, 1, 2, 3, 5, 6, 7, 9, 11, 13, 15, 17, 20, 23, 27, 31, 37, 43, 51, 62, 74, 89, 110, 139, 180,
];

/// `(f32)(2.7996967 * 2.7996967)`.
pub const TI_WIN_SCALE2: f32 = 2.7996967f32 * 2.7996967f32;

pub const NBAND9: usize = 9;

pub struct TiState {
    pub nch: usize,
    pub sr: u32,
    pub sens: f32,
    pub n: usize,
    pub nm: usize,
    pub nh: usize,
    pub n114: usize,
    win1: Vec<f32>,
    win2: Vec<f32>,
    fft_in: Vec<f32>,
    sub_in: Vec<f32>,
    r9: Vec<f32>,
    pw: Vec<f32>,
    band65: Vec<f32>,
    band_raw: Vec<f32>,
    ring_r9: Vec<Vec<f32>>,
    ring_pool: Vec<Vec<f32>>,
    ring_r24: Vec<Vec<f32>>,
    cap: usize,
    w240: Vec<f32>,
    w228: Vec<f32>,
    mag: Vec<f32>,
    v00: Vec<f32>,
    pub w240_n: usize,
    pub w228_n: usize,
    pub mag_n: usize,
    pub v00_n: usize,
    lp_state: f32,
    last_idx: usize,
    last398: i64,
    pub c394: usize,
    pub purge: i64,
    pub w22: i64,
    fft_dump: bool,
    fft_dump_buf: String,
}

impl TiState {
    pub fn new(nch: usize, sr: u32, sens: f32) -> Self {
        let mut v = 16usize;
        let need = ((sr as f32) * 0.008 + 0.5) as i64;
        if need >= 17 {
            while (v as i64) < need {
                v *= 2;
            }
        }
        let n = v;
        let nm = v >> 2;
        let nh = v / 2 + 1;
        let n114 = ((v >> 2) >> 2) + 1;
        let win1: Vec<f32> = (0..n).map(|i| ti_kaiser(i, n, 12.0)).collect();
        let win2: Vec<f32> = (0..nm).map(|i| ti_kaiser(i, nm, 5.0)).collect();
        let s = if sr == 0 { 44100.0f64 } else { sr as f64 };
        let nmf = if nm == 0 { 128.0f64 } else { nm as f64 };
        let wa = (0.01 * s / nmf + 0.5) as i64;
        let w21 = (0.1 * s / nmf + 0.5) as i64;
        let a2 = (0.01 * s / nmf + 0.5) as i64;
        let b2 = (0.025 * s / nmf + 0.5) as i64;
        let c2 = (0.05 * s / nmf + 0.5) as i64;
        let w22 = wa + a2 + 6 + b2 + c2 + 4 * w21;
        let cap = 4096usize;
        Self {
            nch,
            sr,
            sens,
            n,
            nm,
            nh,
            n114,
            win1,
            win2,
            fft_in: vec![0.0; n],
            sub_in: vec![0.0; nm],
            r9: vec![0.0; nh],
            pw: vec![0.0; nh],
            band65: vec![0.0; 65],
            band_raw: vec![0.0; 24],
            ring_r9: vec![vec![0.0; nh]; NBAND9],
            ring_pool: vec![vec![0.0; 65]; NBAND9],
            ring_r24: vec![vec![0.0; 24]; NBAND9],
            cap,
            w240: vec![0.0; cap],
            w228: vec![0.0; cap],
            mag: vec![0.0; cap],
            v00: vec![0.0; cap],
            w240_n: 0,
            w228_n: 0,
            mag_n: 0,
            v00_n: 0,
            lp_state: 0.0,
            last_idx: 0,
            last398: 0,
            c394: 0,
            purge: 0,
            w22,
            fft_dump: false,
            fft_dump_buf: String::new(),
        }
    }

    /// Enable dumping every big-window FFT input (parity diagnostics).
    pub fn set_fft_dump(&mut self, on: bool) {
        self.fft_dump = on;
        self.fft_dump_buf.clear();
    }

    /// Take the accumulated big-window FFT input dump.
    pub fn take_fft_dump(&mut self) -> String {
        std::mem::take(&mut self.fft_dump_buf)
    }

    pub fn stride(&self) -> usize {
        self.nm
    }

    /// Shaped magnitude vector (`mag[..mag_n]`), for parity diagnostics.
    pub fn mag_slice(&self) -> &[f32] {
        &self.mag[..self.mag_n]
    }

    /// Thresholded transient vector (`v00[..v00_n]`), for parity diagnostics.
    pub fn v00_slice(&self) -> &[f32] {
        &self.v00[..self.v00_n]
    }

    /// Diagnostics used by the parity binaries.
    pub fn win1_prefix(&self, n: usize) -> &[f32] {
        &self.win1[..n]
    }
    pub fn win2_prefix(&self, n: usize) -> &[f32] {
        &self.win2[..n]
    }
    pub fn win1_range(&self, a: usize, b: usize) -> &[f32] {
        &self.win1[a..b]
    }
    pub fn mag_range(&self, a: usize, b: usize) -> &[f32] {
        &self.mag[a..b]
    }
    pub fn w240_range(&self, a: usize, b: usize) -> &[f32] {
        &self.w240[a..b]
    }
    pub fn w228_range(&self, a: usize, b: usize) -> &[f32] {
        &self.w228[a..b]
    }
    pub fn ring_r24_slot(&self, s: usize) -> &[f32] {
        &self.ring_r24[s]
    }
    pub fn ring_pool_slot(&self, s: usize) -> &[f32] {
        &self.ring_pool[s]
    }

    /// `iir_alpha`: `1 - exp(-1/(0.1*sr/Nm))` computed in `f64`.
    pub fn iir_alpha(&self) -> f32 {
        let a = 0.1f64;
        let b = self.sr as f64 / self.nm as f64;
        if a == 0.0 || b == 0.0 {
            return 1.0;
        }
        (1.0 - (-1.0 / (a * b)).exp()) as f32
    }

    fn ensure(&mut self, want: usize) {
        if want <= self.cap {
            return;
        }
        let mut nc = self.cap;
        while nc < want {
            nc *= 2;
        }
        for buf in [&mut self.w240, &mut self.w228, &mut self.mag, &mut self.v00] {
            buf.resize(nc, 0.0);
        }
        self.cap = nc;
    }

    /// `ProcessStreaming`: consume `count` samples of the summed input from the
    /// head of `mix`.
    pub fn process(&mut self, count: usize, mix: &[f32]) {
        let n = mix.len();
        self.process_from(0, count, mix, n);
    }

    /// Like [`TiState::process`] but positioned inside a sliding source window:
    /// `base` is the absolute frame index of `mix[0]`, `mix_len` the number of
    /// valid frames. Indices outside `[0, mix_len)` read as zero, which is what
    /// the engine's ring does beyond the fed region.
    pub fn process_from(&mut self, base: usize, count: usize, mix: &[f32], mix_len: usize) {
        let stride = self.nm;
        let nframes = count / stride;
        for _ in 0..nframes {
            let c = self.c394 + 1;
            let g9 = stride as i64 * c as i64 - 384;
            let g6 = stride as i64 * c as i64 - 192;
            if g9 >= 0 {
                let abs = base as i64 + g9;
                let v = Self::gather_window(mix, mix_len, abs, self.n);
                self.big(&v, c, abs);
            }
            if g6 >= 0 {
                let abs = base as i64 + g6;
                let v = Self::gather_window(mix, mix_len, abs, self.nm);
                self.sub(&v, c);
            }
            self.synth(c);
            self.c394 = c;
        }
    }

    /// Read `n` frames starting at `start`; out-of-range frames are zero.
    fn gather_window(mix: &[f32], mix_len: usize, start: i64, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        if start >= 0 {
            let s = start as usize;
            if s < mix_len {
                let end = (s + n).min(mix_len);
                let k = end - s;
                out[..k].copy_from_slice(&mix[s..end]);
            }
        }
        out
    }

    fn big(&mut self, v: &[f32], c: usize, start: i64) {
        for i in 0..self.n {
            // C: (float)((double)win * (double)v)
            self.fft_in[i] = (self.win1[i] as f64 * v[i] as f64) as f32;
        }
        if self.fft_dump {
            let mut s = format!("BIG c={c} g9={start} ");
            for x in &self.fft_in {
                s.push_str(&format!("{:08x} ", x.to_bits()));
            }
            s.push('\n');
            self.fft_dump_buf.push_str(&s);
        }
        let sa = with_plan(self.n, |p| p.fwd(&self.fft_in));
        self.big_post(&sa, c);
    }

    fn big_post(&mut self, sa: &[f32], c: usize) {
        let nh = self.nh;
        // |FFT|^2 then ^(1/4) -> sqrt(sqrt(sqrt(power)))
        for k in 0..nh {
            let re = sa[2 * k];
            let im = sa[2 * k + 1];
            self.pw[k] = re * re + im * im;
            self.pw[k] *= TI_WIN_SCALE2;
        }
        for k in 0..nh {
            self.r9[k] = self.pw[k].sqrt().sqrt().sqrt();
        }
        // 24 log bands: sequential f32 accumulation from 0.0f, then ^(1/8)
        for b in 0..24 {
            let lo = TI_BOUNDS[b];
            let hi = TI_BOUNDS[b + 1];
            let mut acc = 0.0f32;
            for k in lo..hi {
                acc += self.pw[k];
            }
            self.band_raw[b] = acc.sqrt().sqrt().sqrt();
        }
        // w240: full power ^(1/4) (sequential f32 accumulation as in C)
        let mut accf = 0.0f32;
        for k in 0..nh {
            accf += self.pw[k];
        }
        let w = if accf > 0.0 { accf.sqrt().sqrt() } else { 0.0 };
        let wi = c as i64 - self.purge;
        if wi >= 0 {
            let wi = wi as usize;
            self.ensure(wi + 1);
            self.w240[wi] = w;
            if wi + 1 > self.w240_n {
                self.w240_n = wi + 1;
            }
        }
        let slot = c % NBAND9;
        self.ring_r9[slot].copy_from_slice(&self.r9);
        self.ring_r24[slot].copy_from_slice(&self.band_raw);
    }

    fn sub(&mut self, v: &[f32], c: usize) {
        for i in 0..self.nm {
            self.sub_in[i] = (self.win2[i] as f64 * v[i] as f64) as f32;
        }
        let sa = with_plan(self.nm, |p| p.fwd(&self.sub_in));
        for k in 0..65 {
            let re = sa[2 * k];
            let im = sa[2 * k + 1];
            let p = re * re + im * im;
            self.band65[k] = if p > 0.0 {
                1.16609546f32 * p.sqrt().sqrt().sqrt()
            } else {
                0.0
            };
        }
        let slot = c % NBAND9;
        self.ring_pool[slot].copy_from_slice(&self.band65);
    }

    fn synth(&mut self, c: usize) {
        let wb = c as i64 - 4;
        let mut ix = [0usize; 8];
        for (k, q) in [-4i64, -3, -2, -1, 1, 2, 3, 4].iter().enumerate() {
            let v = wb + q;
            ix[k] = if v >= 0 { v as usize % NBAND9 } else { 0 };
        }
        let (i_m4, i_m3, i_m2, i_m1, i_p1, i_p2, i_p3, i_p4) =
            (ix[0], ix[1], ix[2], ix[3], ix[4], ix[5], ix[6], ix[7]);

        // 24 polyphase difference bands
        let mut v = vec![0.0f32; 24];
        for b in 0..24 {
            let p = self.ring_r24[i_p1][b]
                + self.ring_r24[i_p2][b]
                + self.ring_r24[i_p3][b]
                + self.ring_r24[i_p4][b];
            let m = self.ring_r24[i_m1][b]
                + self.ring_r24[i_m2][b]
                + self.ring_r24[i_m3][b]
                + self.ring_r24[i_m4][b];
            v[b] = p - m;
            if v[b] <= 0.0 {
                v[b] = -0.1 * v[b];
            }
        }
        let mut acc = 0.0f32;
        for x in &v {
            acc += *x;
        }
        let shape = acc / 24.0;

        // 65-subband two-slot difference, rectify, mean over n114
        let mut pacc = 0.0f32;
        for b in 0..65 {
            let mut m = self.ring_pool[i_p1][b] - self.ring_pool[i_m1][b];
            if m <= 0.0 {
                m = -0.1 * m;
            }
            pacc += m;
        }
        let pool = pacc / self.n114 as f32;
        let wi = (c as i64 - 4) - self.purge;
        if wi >= 0 {
            let wi = wi as usize;
            self.ensure(wi + 1);
            self.w228[wi] = pool;
            if wi + 1 > self.w228_n {
                self.w228_n = wi + 1;
            }
        }

        // big-window ring: 257-bin polyphase
        let mut vacc = 0.0f32;
        for b in 0..self.nh {
            let mut x = self.ring_r9[i_p1][b]
                + self.ring_r9[i_p2][b]
                + self.ring_r9[i_p3][b]
                + self.ring_r9[i_p4][b];
            let m = self.ring_r9[i_m1][b]
                + self.ring_r9[i_m2][b]
                + self.ring_r9[i_m3][b]
                + self.ring_r9[i_m4][b];
            x -= m;
            if x <= 0.0 {
                x = -0.1 * x;
            }
            vacc += x;
        }
        let vsum = vacc / self.nh as f32;
        let s0 = 2.0 * (vsum + shape + 0.5 * pool);
        let mi = (c as i64 - 4) - self.purge;
        if mi >= 0 {
            let mi = mi as usize;
            self.ensure(mi + 1);
            self.mag[mi] = s0;
            if mi + 1 > self.mag_n {
                self.mag_n = mi + 1;
            }
        }
    }

    /// `AnalyzeStreaming`: envelope shaping plus the max-filter/threshold chain.
    pub fn analyze(&mut self) {
        let n = self.mag_n;
        if n == 0 {
            return;
        }
        let alpha = self.iir_alpha();
        let mut k_lo = self.last_idx;
        let k_hi = n - 1;
        if k_lo > k_hi {
            k_lo = k_hi;
        }
        let mut lp = self.lp_state;
        for k in k_lo..=k_hi {
            let x = self.mag[k];
            lp = lp + alpha * (x - lp);
            let y = x - 0.8 * lp;
            self.mag[k] = y;
            lp = y;
        }
        self.lp_state = lp;
        self.last_idx = k_hi + 1;

        let thr = if self.sens > 0.0 {
            1.0 / self.sens
        } else {
            1.0
        };
        let last398 = self.last398;
        let purge = self.purge;
        let nrec = (n as i64 - 1) + purge;
        // threshold cleanup over [last398, c390)
        {
            let a0 = if last398 > purge { last398 } else { purge };
            let mut a1 = n as i64 - 1 + purge;
            if a1 > purge + n as i64 {
                a1 = purge + n as i64;
            }
            if a1 > a0 {
                let lo = (a0 - purge) as usize;
                let hi = (a1 - purge) as usize;
                for v in self.mag[lo..hi].iter_mut() {
                    if *v < thr {
                        *v = 0.0;
                    }
                }
            }
        }

        let sr = self.sr as f32;
        let wa = ((sr * 0.01) / self.nm as f32 + 0.5) as i64;

        // ---- T1: weighted max-filter kill ----
        let k10 = nrec - wa;
        if k10 > 0 && last398 < nrec {
            let mut k_start = last398 - wa;
            if k_start < 0 {
                k_start = 0;
            }
            let w15 = nrec - 1;
            for k in k_start..=k10 {
                let kr = k - purge;
                if !(0 <= kr && (kr as usize) < n) {
                    continue;
                }
                let vk = self.mag[kr as usize];
                if !(vk >= thr) {
                    continue;
                }
                let mut w3 = k - wa;
                if w3 < 0 {
                    w3 = 0;
                }
                let mut w2 = k + wa;
                if w15 < w2 {
                    w2 = w15;
                }
                if w3 > w2 {
                    continue;
                }
                let keyk = vk
                    * if (kr as usize) < self.w240_n {
                        self.w240[kr as usize]
                    } else {
                        0.0
                    };
                let mut killed = false;
                for j in w3..w2 {
                    let jr = j - purge;
                    if !(0 <= jr && (jr as usize) < self.w240_n) {
                        continue;
                    }
                    if self.mag[jr as usize] * self.w240[jr as usize] > keyk {
                        killed = true;
                        break;
                    }
                }
                if killed {
                    self.mag[kr as usize] = 0.0;
                }
            }
        }

        // ---- A2/B2/C2 progressive max-filters ----
        let gstep = [0.01f32, 0.025, 0.05];
        let gain = [1.0f32, 2.7, 6.0];
        let mut w9 = wa;
        for p in 0..3 {
            let w10 = ((sr * gstep[p]) / self.nm as f32 + 0.5) as i64;
            w9 += w10;
            if p == 0 {
                w9 += 6;
            }
            let k_hi2 = nrec - w9;
            if k_hi2 < 1 {
                continue;
            }
            let mut k_start = last398 - w9;
            if k_start < 0 {
                k_start = 0;
            }
            let w15 = nrec - 1;
            for k in k_start..k_hi2 {
                let kr = k - purge;
                if !(0 <= kr && (kr as usize) < n) {
                    continue;
                }
                let vk = self.mag[kr as usize];
                if !(vk >= thr) {
                    continue;
                }
                let mut w4 = k - w10;
                if w4 < 0 {
                    w4 = 0;
                }
                let mut w3 = k + w10;
                if w3 > w15 {
                    w3 = w15;
                }
                if w4 > w3 {
                    continue;
                }
                let keyk = vk
                    * gain[p]
                    * if (kr as usize) < self.w240_n {
                        self.w240[kr as usize]
                    } else {
                        0.0
                    };
                let mut killed = false;
                for j in w4..w3 {
                    let jr = j - purge;
                    if !(0 <= jr && (jr as usize) < self.w240_n) {
                        continue;
                    }
                    if self.mag[jr as usize] * self.w240[jr as usize] > keyk {
                        killed = true;
                        break;
                    }
                }
                if killed {
                    self.mag[kr as usize] = 0.0;
                }
            }
        }

        // ---- T2: 7-point argmax on w228, swap mag ----
        let mut lo = last398 - wa - 3;
        if lo < 3 {
            lo = 3;
        }
        let hi = n as i64 - 3;
        if hi > lo {
            for k in lo..hi {
                let vk = self.mag[(k - purge) as usize];
                if vk > thr {
                    let mut best = 0.0f32;
                    let mut bj = k;
                    for q in -3i64..=3 {
                        let j = k + q;
                        let jr = j - purge;
                        if j < 0 || !(0 <= jr && (jr as usize) < self.w228_n) {
                            continue;
                        }
                        let key = self.w228[jr as usize];
                        if key > best {
                            best = key;
                            bj = j;
                        }
                    }
                    if bj != k {
                        let a = (k - purge) as usize;
                        let b = (bj - purge) as usize;
                        let t1 = self.mag[b];
                        self.mag[b] = vk;
                        self.mag[a] = t1;
                    }
                }
            }
        }

        // ---- final pass: v00 ----
        let w22 = self.w22;
        let mut i0 = last398 - w22;
        if i0 < purge {
            i0 = purge;
        }
        let mut i1 = n as i64 - 1 + purge - w22;
        let i1c = self.cap as i64 + purge;
        if i1 > i1c {
            i1 = i1c;
        }
        if i1 > i0 {
            let lo = (i0 - purge) as usize;
            let mut hi = (i1 - purge) as usize;
            hi = hi.min(n);
            if hi > lo {
                for k in lo..hi {
                    let s = self.mag[k];
                    self.v00[k] = if s > thr { s } else { 0.0 };
                }
            }
        }
        let mut ln = i1 - purge;
        if ln < 0 {
            ln = 0;
        }
        if ln as usize > self.v00_n {
            self.v00_n = ln as usize;
        }

        self.last_idx = n;
        self.last398 = n as i64 - 1 + purge;
    }

    /// `GetTransientPos`: argmax of `v00`/`mag` over the frame window.
    pub fn get_transient_pos(&self, frm: i64, length: i64) -> i64 {
        if self.mag_n == 0 {
            return -1;
        }
        let stride = self.nm as i64;
        let origin = -(self.nm as i64);
        let purge = self.purge;
        let mut k0 = (frm - origin + (stride >> 1)) / stride;
        let mut k1 = (frm + length - origin + (stride >> 1)) / stride;
        if k0 < purge {
            k0 = purge;
        }
        let k_hi_max = purge + self.mag_n as i64 - 1;
        if k1 > k_hi_max {
            k1 = k_hi_max;
        }
        if k0 > k1 {
            return -1;
        }
        let lo = k0 - purge;
        let hi = k1 - purge;
        let mut best_k: i64 = k0;
        let mut best = f32::NEG_INFINITY;
        for idx in lo..=hi {
            let v = if self.v00_n > 0 && (idx as usize) < self.v00_n {
                self.v00[idx as usize]
            } else {
                self.mag[idx as usize]
            };
            if v > best {
                best = v;
                best_k = idx + purge;
            }
        }
        if !(best > 0.0) {
            return -1;
        }
        let t = origin + stride * best_k;
        if t >= frm + length {
            return -1;
        }
        if t < frm {
            return -1;
        }
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_48k() {
        let ti = TiState::new(2, 48000, 1.0);
        // need = (int)(48000*0.008+0.5) = 384 -> v doubles 16..512
        assert_eq!(ti.n, 512);
        assert_eq!(ti.nm, 128);
        assert_eq!(ti.nh, 257);
        assert_eq!(ti.n114, 33);
        assert_eq!(ti.stride(), 128);
    }

    #[test]
    fn impulse_produces_transient() {
        let mut ti = TiState::new(2, 48000, 1.0);
        let mut mix = vec![0.0f32; 8192];
        mix[3000] = 1.0;
        for _ in 0..8 {
            ti.process(1024, &mix);
            ti.analyze();
        }
        let t = ti.get_transient_pos(2500, 1000);
        assert!(t >= 2500 && t < 3500, "transient at {t}");
    }

    #[test]
    fn silence_has_no_transient() {
        let mut ti = TiState::new(2, 48000, 1.0);
        let mix = vec![0.0f32; 8192];
        for _ in 0..8 {
            ti.process(1024, &mix);
            ti.analyze();
        }
        assert_eq!(ti.get_transient_pos(0, 5000), -1);
    }
}
