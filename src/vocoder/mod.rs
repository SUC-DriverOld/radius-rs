//! Vocoder DSP operators — port of `pyradius/vocoder_ops.py`
//! (`libradius/src/ops/*`).
//!
//! Everything here is `f32` with the C op order; `fma` sites use [`fma`], the
//! correctly-rounded single-rounding shim the Python port calls `_fma`.

use crate::consts::*;

/// `F32(x)` — a `numpy.float32` scalar in the reference.
pub type F32 = f32;

/// Correctly-rounded single-rounding `a*b + c` (the C `fmaf` shim).
#[inline(always)]
pub fn fma(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}

/// `np.rint` (round half to even).
#[inline(always)]
pub fn rint(x: f32) -> f32 {
    // Rust's `round_ties_even` is stable since 1.77.
    x.round_ties_even()
}

/// `_round_i` — `(int)rint(x)`.
#[inline(always)]
pub fn round_i(x: f32) -> i32 {
    rint(x) as i32
}

/// `_wrap_pi` — wrap an angle into `[-pi, pi)` with the reference's op order.
#[inline(always)]
pub fn wrap_pi(x: f32) -> f32 {
    let q = rint(x * INV_2PI_F);
    let t = (q as f64) * NEG_TWO_PI_F64 + (x as f64);
    t as f32
}

/// `_floormod` for non-negative `cap`.
#[inline(always)]
pub fn floormod(d: i64, cap: i64) -> i64 {
    let r = d % cap;
    if r < 0 {
        r + cap
    } else {
        r
    }
}

/// `np.float32(pi)` bits.
pub const PI_F_BITS_LOCAL: u32 = PI_F_BITS;
/// `np.float32(2pi)` bits.
pub const TWO_PI_F_BITS_LOCAL: u32 = TWO_PI_F_BITS;
/// `1/(2pi)` as `f32`.
pub const INV_2PI_F: f32 = 0.159_154_94;
/// `-2pi` as `f64` (the engine's folded constant).
pub const NEG_2PI_F64_LOCAL: f64 = NEG_TWO_PI_F64;
/// `0.7f`
pub const K07_F: f32 = 0.7;
/// `1e-6f`
pub const EPS1E6_F: f32 = 1e-6;

pub mod acs;
pub mod crossover_tables;
pub mod formant;
pub mod noise;
pub mod phase;
pub mod synth;
pub mod vocoder;

pub use acs::*;
pub use crossover_tables as tables_data;
pub use formant::*;
pub use noise::*;
pub use phase::*;
pub use synth::*;

pub use vocoder::{supported_rate, vc_calc_sched, VocoderConfig, VocoderState};
