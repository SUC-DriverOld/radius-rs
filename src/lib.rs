//! # radius-rs
//!
//! A Rust rewrite of the Radius time-domain (TD) and phase-vocoder pitch-shift
//! algorithms, ported from the `pyradius` Python/NumPy reference (which is
//! itself a line-by-line port of the `libradius` C engine).
//!
//! Three ways in:
//!
//! * **crate** — [`TdState::render`] / [`vocoder::VocoderState::render`] plus the
//!   lower level operator modules.
//! * **CLI** — the `radius-rs` binary (feature `cli`) reads and writes
//!   wav/flac/ogg/mp3/aac through pure-Rust codecs, no `ffmpeg`.
//! * **C ABI** — [`ffi`], exported from `cdylib`/`staticlib` builds against
//!   `include/radius_rs.h`.
//!
//! ## Float discipline
//!
//! The reference engine is compiled with `-ffp-contract=off` and keeps almost
//! everything in `f32`. The port mirrors that: no `f64` creep in the hot paths,
//! `a*b + c` stays two operations (never `mul_add`), and transcendentals are
//! evaluated in `f64` and rounded once, matching how the C compiler lowers
//! `expf`/`logf`/`powf`/`cosf`/`sinf` here.

pub mod consts;
pub mod fft;
/// The reference radix-2 FFT kernel, always compiled.
///
/// This is the kernel with which the crate reproduces the reference engine: same
/// twiddle table, same butterfly order, same per-operation `f32` rounding.
/// [`fft`] selects between it and the `rustfft` backend at run time, so both are
/// reachable from a single build and the tests can compare them.
pub mod fft_radix2;
pub mod interp;
pub mod simple_rand;
pub mod tables;
pub mod ti;

pub mod td;
pub mod vocoder;

pub mod ffi;

/// Audio input/output, entirely through ffmpeg. See the module docs for how the
/// binary is located (`RADIUS_FFMPEG`, otherwise `PATH`).
pub mod io;
/// Command line front-end (`radius`).
pub mod cli;

/// Version string reported by the CLI and the C ABI.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Re-exported conveniences for crate users.
pub use interp::InterpTable;
pub use td::TdState;
pub use ti::TiState;
pub use vocoder::{VocoderConfig, VocoderState};

/// Decode-free helpers shared by the library, the CLI and the tests.
pub mod util {
    /// Peak absolute sample of an interleaved buffer.
    pub fn peak(x: &[f32]) -> f32 {
        x.iter().fold(0.0f32, |a, b| a.max(b.abs()))
    }

    /// Pearson correlation of two interleaved buffers over their common prefix.
    pub fn corr(a: &[f32], b: &[f32]) -> f64 {
        let n = a.len().min(b.len());
        if n == 0 {
            return 0.0;
        }
        let a = &a[..n];
        let b = &b[..n];
        let ma = a.iter().map(|v| *v as f64).sum::<f64>() / n as f64;
        let mb = b.iter().map(|v| *v as f64).sum::<f64>() / n as f64;
        let mut sa = 0.0;
        let mut sb = 0.0;
        let mut sab = 0.0;
        for i in 0..n {
            let da = a[i] as f64 - ma;
            let db = b[i] as f64 - mb;
            sa += da * da;
            sb += db * db;
            sab += da * db;
        }
        if sa > 0.0 && sb > 0.0 {
            sab / (sa.sqrt() * sb.sqrt())
        } else {
            0.0
        }
    }

    /// Maximum absolute difference over the common prefix.
    pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let mut m = 0.0f32;
        for i in 0..n {
            let d = (a[i] - b[i]).abs();
            if d > m {
                m = d;
            }
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn td_geometry_matches_reference() {
        let st = TdState::new(48000, 37, 0, 2);
        assert_eq!(st.hop, 2664);
        assert_eq!(st.f28, 4800);
        assert_eq!(st.win_max, 4);
        assert_eq!(st.geometry().n, 8192);
        assert_eq!(st.geometry().l1, 3330);
    }

    #[test]
    fn td_render_is_finite() {
        let sr = 48000u32;
        let n = 48000usize;
        let mut x = vec![0.0f32; n * 2];
        for i in 0..n {
            let t = i as f32 / sr as f32;
            let v = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            x[2 * i] = v;
            x[2 * i + 1] = v;
        }
        let mut st = TdState::new(sr, 37, 0, 2);
        st.set_ratio(3.0, 100.0);
        let y = st.render(&x, n);
        assert!(!y.is_empty());
        assert!(y.iter().all(|v| v.is_finite()));
        // duration preserving
        let ratio = y.len() as f64 / x.len() as f64;
        assert!((ratio - 1.0).abs() < 0.02, "length ratio {ratio}");
    }
}
