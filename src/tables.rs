//! Window/taper table builders shared by the TD and vocoder engines.
//!
//! Everything here is *computed*, matching the corresponding C constructor
//! (`rx_td_build_env_tables`, `rx_td_pitch_set_options`, …). The genuinely
//! tabulated data the engine reads (crossover FIR banks, vocoder window and
//! synthesis-window banks) lives in [`crate::tables_data`]; the vocoder schedule is
//! not tabulated at all, it is computed by
//! [`crate::vocoder::vc_calc_sched`].

use crate::consts::{PI_F64, TWO_PI_F};

/// `hann_pow` (td_core.c): `[Hann^gain(len), Hann^gain(len)]`, one record.
pub fn hann_pow(len: usize, gain: f32) -> Vec<f32> {
    let mut v = Vec::with_capacity(len * 2);
    let mut half = Vec::with_capacity(len);
    for i in 0..len {
        let u = (i as f64 + 0.5) / len as f64;
        let h = 0.5 - 0.5 * (TWO_PI_F as f64 * u).cos();
        let p = h.powf(gain as f64) as f32;
        half.push(p);
    }
    v.extend_from_slice(&half);
    v.extend_from_slice(&half);
    v
}

/// `rx_td_build_env_tables` spans: 32 log-spaced spans per gain block.
pub fn td_env_spans(f28: u32, nblk: usize, ntab: usize) -> Vec<Vec<usize>> {
    let s8 = ((f28 / 10) as f32).ln();
    let s9 = (f28 as f32).ln();
    let c1 = (ntab - 1) as f32;
    let mut out = vec![vec![0usize; ntab]; nblk];
    for row in out.iter_mut() {
        for (j, slot) in row.iter_mut().enumerate() {
            let v = (s8 * (c1 - j as f32) + s9 * (j as f32)) / c1;
            let l = (v.exp() + 0.5) as i64;
            *slot = if l < 2 { 2 } else { l as usize };
        }
    }
    out
}

/// `rx_td_build_env_tables` gain blocks: `0.5625 + 0.125*k`, block 4 = 1.0.
pub fn td_env_gain(a8: usize, win_max: usize) -> f32 {
    if a8 >= win_max {
        1.0
    } else {
        0.5625 + 0.125 * a8 as f32
    }
}

/// `rx_td_pitch_set_options` derived geometry.
#[derive(Debug, Clone, Copy)]
pub struct PitchGeom {
    pub n: usize,
    pub l1: usize,
    pub maxbin: usize,
    pub taper_len: usize,
    pub lo: usize,
    pub hi: usize,
}

pub fn pitch_geom(sr: u32, hop: u32, solo: bool, win_max: usize) -> PitchGeom {
    let _ = win_max;
    let a4 = (hop >> 1) as usize;
    let mut n = 128usize;
    let l1 = (2.5 * a4 as f64 + 0.5) as usize;
    while n < 5 * a4 {
        n <<= 1;
    }
    let m = (4000.0 / sr as f64 * n as f64 + 0.5) as usize;
    let cap = n / 2 - 1;
    let mut maxbin = if m < cap { m } else { cap };
    if maxbin < 1 {
        maxbin = 1;
    }
    let mut v58 = (10000.0 / sr as f64 * n as f64 + 0.5) as usize + 1;
    if v58 > n - 1 {
        v58 = n - 1;
    }
    let mut vlen = v58.saturating_sub(maxbin);
    if vlen < 1 {
        vlen = 1;
    }
    let lo = if solo {
        (hop / 40) as usize
    } else {
        (hop >> 2) as usize
    };
    let hi = (hop >> 1) as usize;
    PitchGeom {
        n,
        l1,
        maxbin,
        taper_len: vlen,
        lo,
        hi,
    }
}

/// Frequency-domain taper: `0.5 + 0.5*cos(pi*i/vlen)`.
pub fn pitch_win_taper(vlen: usize) -> Vec<f32> {
    (0..vlen)
        .map(|i| (0.5 + 0.5 * (PI_F64 * i as f64 / vlen as f64).cos()) as f32)
        .collect()
}

/// Periodic Hann analysis window of length `l1`.
pub fn pitch_win_an(l1: usize) -> Vec<f32> {
    (0..l1)
        .map(|i| (0.5 - 0.5 * (TWO_PI_F as f64 * i as f64 / l1 as f64).cos()) as f32)
        .collect()
}

/// Raised-cosine ramps at both ends of the ACF window.
pub fn pitch_win_acf(n: usize, l1: usize) -> Vec<f32> {
    let r = l1 >> 3;
    let mut w = vec![0.0f32; n];
    for i in 0..r {
        w[i] = (0.5 + 0.5 * (PI_F64 * i as f64 / r as f64).cos()) as f32;
    }
    for k in 1..r {
        w[n - k] = w[k];
    }
    w
}

/// Linear decay `1 - i/(n/2+1)`, clamped at 0.
pub fn pitch_win_lin(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let v = 1.0 - i as f64 / (n as f64 / 2.0 + 1.0);
            if v > 0.0 {
                v as f32
            } else {
                0.0
            }
        })
        .collect()
}

/// `_ti_kaiser` — Kaiser window with a 60-term series `I0`.
pub fn ti_kaiser(i: usize, n: usize, beta: f64) -> f32 {
    if n < 2 {
        return 1.0;
    }
    let a = (2.0 * i as f64 / (n as f64 - 1.0)) - 1.0;
    let r = 1.0 - a * a;
    (ti_i0(beta * (if r > 0.0 { r } else { 0.0 }).sqrt()) / ti_i0(beta)) as f32
}

/// `_ti_i0` — plain 60-term series (deliberately not the usual convergence test).
pub fn ti_i0(z: f64) -> f64 {
    let mut s = 1.0f64;
    let mut t = 1.0f64;
    for k in 1..60 {
        let d = z / 2.0 / k as f64;
        t *= d * d;
        s += t;
    }
    s
}

/// `_i0` used by the resampler kernel (convergence-tested variant).
pub fn i0(x: f64) -> f64 {
    let mut s = 1.0f64;
    let mut term = 1.0f64;
    let x2 = x * x * 0.25;
    for k in 1..64 {
        term *= x2 / ((k * k) as f64);
        s += term;
        if term < 1e-18 * s {
            break;
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_spans_are_monotone_and_clamped() {
        let spans = td_env_spans(4800, 5, 32);
        for row in &spans {
            // log-interpolated between f28/10 and f28, so non-decreasing
            assert_eq!(row[0], 480);
            assert_eq!(row[31], 4800);
            for w in row.windows(2) {
                assert!(w[1] >= w[0], "spans must be non-decreasing");
            }
            assert!(row.iter().all(|v| *v >= 2));
        }
        // every block shares the same span vector (ref_env measurement)
        for row in &spans[1..] {
            assert_eq!(row, &spans[0]);
        }
    }

    #[test]
    fn hann_pow_is_symmetric_record() {
        let r = hann_pow(16, 0.5625);
        assert_eq!(r.len(), 32);
        assert_eq!(&r[..16], &r[16..]);
        assert!(r.iter().all(|v| (0.0..=1.0).contains(v)));
    }

    #[test]
    fn pitch_geometry_48k_q37() {
        let g = pitch_geom(48000, 2664, false, 4);
        assert_eq!(g.n, 8192);
        assert_eq!(g.l1, 3330);
        assert_eq!(g.maxbin, (4000.0f64 / 48000.0 * 8192.0 + 0.5) as usize);
        assert_eq!(g.lo, 2664 >> 2);
        assert_eq!(g.hi, 2664 >> 1);
    }
}
