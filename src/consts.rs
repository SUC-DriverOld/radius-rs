//! Bit-exact constants shared by every stage of the port.
//!
//! The C engine uses `RX_PI` as a `float` literal that the compiler
//! constant-folds; the double forms below are the *exact* bit patterns the
//! engine's folded constants have (verified against the built binary by
//! `pyradius.fft`), so the twiddle tables reproduce the C values bit for bit
//! instead of relying on a decimal literal surviving two conversions.

/// `f32` bits of `RX_PI` (0x40490FDB).
pub const PI_F_BITS: u32 = 0x4049_0FDB;
/// `f32` bits of `2*RX_PI` (0x40C90FDB).
pub const TWO_PI_F_BITS: u32 = 0x40C9_0FDB;

pub const PI_F: f32 = f32::from_bits(PI_F_BITS);
pub const TWO_PI_F: f32 = f32::from_bits(TWO_PI_F_BITS);

/// `(double)(-2.0f * RX_PI)` as the engine's constant folder emits it:
/// `-2.0 * (double)(float)RX_PI` -> 0xC01921FB60000000.
pub const NEG_TWO_PI_F64: f64 = f64::from_bits(0xC019_21FB_6000_0000);
/// Plain f64 pi, used where the C source itself computes in double.
pub const PI_F64: f64 = std::f64::consts::PI;

/// `FLT_MAX` as used by the vocoder region logic.
pub const FLT_MAX: f32 = f32::MAX;

/// Largest finite ring/loop bounds used by the offline renderer.
pub const MASK64: u64 = u64::MAX;

#[inline(always)]
pub fn f32_from_bits(u: u32) -> f32 {
    f32::from_bits(u)
}

/// `(unsigned)v` — 32-bit wraparound, matching the C casts on the hot paths.
#[inline(always)]
pub fn as_u32(v: i64) -> u32 {
    v as u32
}

#[inline(always)]
pub fn as_u32_us(v: u64) -> u32 {
    v as u32
}

/// C `(int)` truncation toward zero for values in range.
#[inline(always)]
pub fn c_trunc_f32(v: f32) -> i32 {
    v as i32
}

#[inline(always)]
pub fn c_trunc_f64(v: f64) -> i32 {
    v as i32
}

/// C `lround` (round half away from zero) on a double.
#[inline(always)]
pub fn lround(v: f64) -> i64 {
    v.round() as i64
}

/// C `(int)(x + 0.5)` for `x >= 0`, the engine's dominant rounding idiom.
#[inline(always)]
pub fn round_pos(v: f32) -> i32 {
    (v + 0.5) as i32
}

/// `rx_td_wrap_ring`: positions inside `[0, rtot)` pass through, anything else
/// folds into `[rstart, rtot)`.
#[inline(always)]
pub fn wrap_ring(pos: i64, rstart: i64, rtot: i64) -> i64 {
    if pos >= 0 && pos < rtot {
        return pos;
    }
    let span = if rtot > rstart { rtot - rstart } else { 1 };
    rstart + (pos - rtot).rem_euclid(span)
}

#[inline(always)]
pub fn wrap_out(pos: i64, out_len: i64) -> i64 {
    pos.rem_euclid(out_len)
}

/// C `%` on non-negative operands (kept explicit for readability of the ports).
#[inline(always)]
pub fn c_mod(a: i64, b: i64) -> i64 {
    a.rem_euclid(b)
}
