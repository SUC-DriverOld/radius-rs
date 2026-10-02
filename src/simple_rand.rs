//! `pyradius.simple_rand` — MSVC LCG plus the vocoder noise template.
//!
//! `state = state*214013 + 2531011` (u64 wraparound), output `(state>>16) & 0x7FFF`.

pub const MUL: u64 = 214013;
pub const ADD: u64 = 2531011;

use crate::consts::{PI_F, TWO_PI_F};

#[derive(Debug, Clone)]
pub struct SimpleRand {
    pub state: u64,
}

impl Default for SimpleRand {
    fn default() -> Self {
        Self::new(1)
    }
}

impl SimpleRand {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    pub fn next_u15(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(MUL).wrapping_add(ADD);
        ((self.state >> 16) & 0x7FFF) as u32
    }

    /// Advance the LCG by `k` steps in `O(log k)`, bitwise-equal to `k` calls.
    pub fn skip(&mut self, k: u64) {
        let mut mk = MUL;
        let mut ask = ADD;
        let mut s = self.state;
        let mut e = k;
        while e > 0 {
            if e & 1 == 1 {
                s = s.wrapping_mul(mk).wrapping_add(ask);
            }
            ask = ask.wrapping_mul(mk.wrapping_add(1));
            mk = mk.wrapping_mul(mk);
            e >>= 1;
        }
        self.state = s;
    }
}

/// `noise_template_fill`: `dst[i] = u * 2pi - pi` with `u = next()/32767` in f32.
pub fn noise_template_fill(dst: &mut [f32], rng: &mut SimpleRand) {
    let mut s = rng.state;
    for v in dst.iter_mut() {
        s = s.wrapping_mul(MUL).wrapping_add(ADD);
        let out15 = ((s >> 16) & 0x7FFF) as f32;
        let u = (out15 as f64 / 32767.0) as f32;
        *v = u * TWO_PI_F - PI_F;
    }
    rng.state = s;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcg_matches_reference_prefix() {
        let mut r = SimpleRand::new(1);
        let got: Vec<u32> = (0..8).map(|_| r.next_u15()).collect();
        // recomputed by hand from the recurrence, independent of the Rust code
        let mut s: u64 = 1;
        let mut want = Vec::new();
        for _ in 0..8 {
            s = s.wrapping_mul(214013).wrapping_add(2531011);
            want.push(((s >> 16) & 0x7FFF) as u32);
        }
        assert_eq!(got, want);
    }

    #[test]
    fn skip_matches_stepwise() {
        for k in [0u64, 1, 2, 7, 64, 1000] {
            let mut a = SimpleRand::new(12345);
            for _ in 0..k {
                a.next_u15();
            }
            let mut b = SimpleRand::new(12345);
            b.skip(k);
            assert_eq!(a.state, b.state, "k={k}");
        }
    }

    #[test]
    fn template_range() {
        let mut r = SimpleRand::new(1);
        let mut buf = vec![0.0f32; 1024];
        noise_template_fill(&mut buf, &mut r);
        for v in buf {
            assert!((-PI_F - 1e-3..=PI_F + 1e-3).contains(&v));
        }
    }
}
