//! `pyradius.sampler` — `Resampler::InterpolateNSamples` (`interp_nsamples.c`).
//!
//! Table build: master kernel `sinc(j*pi/X)` (`f64` sin/div, cast `f32`), Kaiser
//! `beta=12`, strict-sequential `f32` sum normalize. Per level:
//! `pitch = 1 + 2*level/n_levels` (`f32`), `ns = (int)(X/pitch + 0.5)`, and the
//! sub-phase coefficients `ratio * master[v24 + j*ns]`.
//!
//! Runtime, per output sample: `frac` from the running phase, the sub-phase index
//! from the single-rounded `fmaf(frac, nsubs, 0.5)`, the coefficient window and a
//! strictly ascending `f32` dot product.

use crate::tables::i0;

#[derive(Debug, Clone)]
pub struct InterpTable {
    pub x: usize,
    pub n_levels: usize,
    pub len: usize,
    pub half: usize,
    /// `nsubs[level]`
    pub nsubs: Vec<i64>,
    /// `offsets[level][sub]`
    pub offsets: Vec<Vec<i64>>,
    /// `pools[level][sub][coef]` zero-padded to the level's max tap count
    pub pools: Vec<Vec<Vec<f32>>>,
    /// `taps[level][sub]`
    pub taps: Vec<Vec<i64>>,
}

fn kaiser_in_place(k: &mut [f32], beta: f64) {
    let n = k.len();
    if n < 2 {
        if n == 1 {
            k[0] = 1.0;
        }
        return;
    }
    let denom = i0(beta);
    let n1 = (n - 1) as f64;
    for (i, v) in k.iter_mut().enumerate() {
        let x = 2.0 * i as f64 / n1 - 1.0;
        let t = (1.0 - x * x).max(0.0).sqrt();
        let val = (i0(beta * t) / denom) as f32;
        *v *= val;
    }
}

impl InterpTable {
    pub fn new(x: usize, n_levels: usize, quality: f32) -> Self {
        let xf = x as f32;
        let lq = xf * quality;
        let ln = ((lq + lq) as i64) as usize | 1;
        let half = ln >> 1;
        assert!(ln >= 1);

        // master kernel: x = (j*pi)/X computed in double from integer j
        let pi = std::f64::consts::PI;
        let mut master = vec![0.0f32; ln];
        for (idx, slot) in master.iter_mut().enumerate() {
            let j = idx as f64 - half as f64;
            let xv = (j * pi) / x as f64;
            let sinc = if xv == 0.0 { 1.0 } else { xv.sin() / xv };
            *slot = sinc as f32;
        }
        kaiser_in_place(&mut master, 12.0);
        // strict sequential f32 sum
        let mut ssum = 0.0f32;
        for v in &master {
            ssum += *v;
        }
        let scale = xf / ssum;
        for v in master.iter_mut() {
            *v = scale * *v;
        }

        let mut nsubs = vec![0i64; n_levels];
        let mut offsets = Vec::with_capacity(n_levels);
        let mut pools = Vec::with_capacity(n_levels);
        let mut taps_all = Vec::with_capacity(n_levels);
        for level in 0..n_levels {
            let pitch = ((level + level) as f32) / (n_levels as f32) + 1.0;
            let mut ns = x as i64;
            let mut ratio = 1.0f32;
            if pitch > 1.0 {
                ns = ((x as f32 / pitch) + 0.5) as i64;
                ratio = ns as f32 / x as f32;
            }
            assert!(ns >= 1 && half as i64 >= ns);
            nsubs[level] = ns;
            let ns_u = ns as usize;
            let mut offs = vec![0i64; ns_u + 1];
            let mut v24s = vec![0i64; ns_u + 1];
            for s in 0..=ns_u {
                let mut v24 = half as i64 - s as i64;
                let mut o = 0i64;
                if v24 >= ns {
                    loop {
                        o += 1;
                        v24 -= ns;
                        if v24 < ns {
                            break;
                        }
                    }
                }
                offs[s] = o;
                v24s[s] = v24;
            }
            let mut taps = vec![0i64; ns_u + 1];
            for s in 0..=ns_u {
                taps[s] = (ln as i64 - v24s[s]) / ns + 1;
            }
            let maxt = *taps.iter().max().unwrap() as usize;
            let mut pool = vec![vec![0.0f32; maxt]; ns_u + 1];
            for s in 0..=ns_u {
                let mut idx = v24s[s];
                let mut col = 0usize;
                while idx < ln as i64 {
                    pool[s][col] = ratio * master[idx as usize];
                    idx += ns;
                    col += 1;
                }
            }
            offsets.push(offs);
            pools.push(pool);
            taps_all.push(taps);
        }
        Self {
            x,
            n_levels,
            len: ln,
            half,
            nsubs,
            offsets,
            pools,
            taps: taps_all,
        }
    }
}

/// `interp_kidx` — which resampler level a quality value selects.
pub fn interp_kidx(quality: f32, n_levels: usize) -> usize {
    let t = (quality - 1.0) * 0.5;
    let mut k = (t * n_levels as f32 + 0.5) as i64;
    if k > n_levels as i64 - 1 {
        k = n_levels as i64 - 1;
    }
    if k < 0 {
        k = 0;
    }
    k as usize
}

/// Round `frac * nsubs + 0.5` once in `f32` (the C `fmaf` single rounding) and
/// truncate.
#[inline]
fn sub_index(frac: f64, nsubs: f64) -> i64 {
    let frac_f = frac as f32;
    let subs_f = nsubs as f32;
    let v = (frac_f as f64) * (subs_f as f64) + 0.5;
    (v as f32) as i64
}

/// `rx_interp_nsamples`. `src` is a single channel of `ring_size` samples;
/// returns `count` samples.
#[allow(clippy::too_many_arguments)]
pub fn interp_nsamples_ch(
    tbl: &InterpTable,
    src: &[f32],
    ring_size: i64,
    phase: f64,
    count: usize,
    rate: f64,
    quality: f32,
) -> Vec<f32> {
    let mut dst = vec![0.0f32; count];
    if count == 0 {
        return dst;
    }
    let kidx = interp_kidx(quality, tbl.n_levels);
    let dnsubs = tbl.nsubs[kidx] as f64;
    let offs = &tbl.offsets[kidx];
    let pool = &tbl.pools[kidx];
    let taps_arr = &tbl.taps[kidx];
    let rs = ring_size;

    let mut ph = phase;
    for out in dst.iter_mut() {
        let iph = ph as i64; // trunc, phase >= 0
        let frac = ph - iph as f64;
        let sub = sub_index(frac, dnsubs);
        let taps = taps_arr[sub as usize];
        if taps <= 0 {
            ph += rate;
            continue;
        }
        let base = iph - offs[sub as usize];
        let row = &pool[sub as usize];
        let mut acc = 0.0f32;
        for j in 0..taps as usize {
            let mut idx = base + j as i64;
            if idx < 0 {
                idx += rs;
            } else if idx >= rs {
                idx -= rs;
            }
            acc += src[idx as usize] * row[j];
        }
        *out = acc;
        ph += rate;
    }
    dst
}

/// Multi-channel wrapper keeping the C's per-channel independence.
#[allow(clippy::too_many_arguments)]
pub fn interp_nsamples(
    tbl: &InterpTable,
    src: &[Vec<f32>],
    ring_size: i64,
    phase: f64,
    count: usize,
    rate: f64,
    quality: f32,
) -> Vec<Vec<f32>> {
    src.iter()
        .map(|ch| interp_nsamples_ch(tbl, ch, ring_size, phase, count, rate, quality))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_geometry_matches_engine() {
        let t = InterpTable::new(1024, 6, 12.0);
        // lq = 1024*12 = 12288 -> len = 24576|1 = 24577, half = 12288
        assert_eq!(t.len, 24577);
        assert_eq!(t.half, 12288);
        assert_eq!(t.nsubs[0], 1024);
        assert_eq!(interp_kidx(12.0, 6), 5);
        // level 5: pitch = 1+10/6 = 2.6667 -> ns = int(1024/2.6667+0.5) = 384
        assert_eq!(t.nsubs[5], ((1024.0f32 / (1.0 + 10.0 / 6.0)) + 0.5) as i64);
    }

    #[test]
    fn unit_gain_dc_passthrough() {
        // a constant signal must come back as (approximately) the same constant
        let t = InterpTable::new(1024, 6, 12.0);
        let src = vec![0.5f32; 4096];
        let out = interp_nsamples_ch(&t, &src, 4096, 0.0, 64, 1.0, 12.0);
        for v in out {
            assert!((v - 0.5).abs() < 2e-3, "got {v}");
        }
    }
}
