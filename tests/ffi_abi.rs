//! FFI (C ABI) integration test: the exported symbols must be callable from C
//! and produce the same audio as the Rust API.

mod common;

use common::*;
use std::ffi::{c_char, c_void, CStr};

// Declared exactly as `include/radius_rs.h` declares them.
extern "C" {
    fn radius_version() -> *const c_char;
    fn rx_td_init(sr: u32, quality: i32, solo: i32, nch: i32) -> *mut c_void;
    fn rx_td_free(st: *mut c_void);
    fn rx_td_set_ratio(st: *mut c_void, semis: f64, tempo: f64);
    fn rx_td_render(
        st: *mut c_void,
        input: *const f32,
        nframes: i64,
        out: *mut f32,
        out_cap: i64,
    ) -> i64;
    fn rx_td_geometry(sr: u32, quality: i32, solo: i32, out: *mut RxTdGeometry) -> i32;
    fn rx_vc_init(sr: u32, nch: i32) -> *mut c_void;
    fn rx_vc_free(st: *mut c_void);
    #[allow(dead_code)]
    fn rx_vc_set_ratio(st: *mut c_void, semis: f64, tempo: f64);
    fn rx_vc_render(
        st: *mut c_void,
        input: *const f32,
        nframes: i64,
        out: *mut f32,
        out_cap: i64,
    ) -> i64;
}

#[test]
fn vocoder_ffi_render_is_reachable() {
    // A short render through the C ABI: the symbol must exist, accept a
    // supported rate, and produce finite, non-silent, duration-preserving audio.
    const SR: u32 = 48000;
    let frames = SR as usize / 4;
    let mut x = vec![0.0f32; frames * 2];
    for i in 0..frames {
        let t = i as f32 / SR as f32;
        let v = 0.5 * (2.0 * std::f32::consts::PI * 500.0 * t).sin();
        x[2 * i] = v;
        x[2 * i + 1] = v;
    }
    let mut out = vec![0.0f32; x.len()];
    let n = unsafe {
        let h = rx_vc_init(SR, 2);
        assert!(!h.is_null());
        rx_vc_set_ratio(h, 3.0, 100.0);
        let n = rx_vc_render(
            h,
            x.as_ptr(),
            frames as i64,
            out.as_mut_ptr(),
            frames as i64,
        );
        rx_vc_free(h);
        n
    };
    assert!(n > 0, "rx_vc_render returned {n}");
    let got = &out[..n as usize * 2];
    assert!(got.iter().all(|v| v.is_finite()));
    assert!(got.iter().fold(0.0f32, |a, b| a.max(b.abs())) > 1e-4);
}

#[repr(C)]
#[derive(Default, Debug)]
struct RxTdGeometry {
    hop: u32,
    f28: u32,
    win_max: u32,
    pitch_n: u32,
    pitch_l1: u32,
    pitch_maxbin: u32,
    pitch_taper_len: u32,
    pitch_lo: u32,
    pitch_hi: u32,
}

#[test]
fn version_is_non_empty() {
    let v = unsafe { CStr::from_ptr(radius_version()) };
    assert!(!v.to_str().unwrap().is_empty());
}

#[test]
fn geometry_matches_rust_api() {
    let mut g = RxTdGeometry::default();
    let rc = unsafe { rx_td_geometry(48000, 37, 0, &mut g) };
    assert_eq!(rc, 0);
    let st = radius_rs::TdState::new(48000, 37, 0, 2);
    let want = st.geometry();
    assert_eq!(g.hop, st.hop);
    assert_eq!(g.f28, st.f28);
    assert_eq!(g.pitch_n as usize, want.n);
    assert_eq!(g.pitch_l1 as usize, want.l1);
    assert_eq!(g.pitch_maxbin as usize, want.maxbin);
}

#[test]
fn td_render_through_ffi_matches_rust_api() {
    // Synthetic input: the FFI contract is about pointers, lengths and the ABI, so
    // it does not need the acceptance corpus (which is not redistributed).
    let input = read_wav(&synthetic_file("ffi", 48_000, 48_000 * 4));
    // keep the FFI test quick: 4 seconds is plenty to exercise every stage
    let frames = (input.rate as usize * 4).min(input.frames());
    let src = &input.samples[..frames * input.channels];

    // Rust API
    let mut st = radius_rs::TdState::new(input.rate, 37, 0, input.channels);
    st.set_ratio(3.0, 100.0);
    let want = st.render(src, frames, frames);

    // C ABI
    let mut out = vec![0.0f32; src.len() + 262144 * input.channels];
    let n = unsafe {
        let h = rx_td_init(input.rate, 37, 0, input.channels as i32);
        assert!(!h.is_null());
        rx_td_set_ratio(h, 3.0, 100.0);
        let n = rx_td_render(
            h,
            src.as_ptr(),
            frames as i64,
            out.as_mut_ptr(),
            (out.len() / input.channels) as i64,
        );
        rx_td_free(h);
        n
    };
    assert!(n > 0, "rx_td_render returned {n}");
    let got = &out[..n as usize * input.channels];
    let c = corr(got, &want);
    let maxd = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(c > 0.99999, "FFI vs Rust api corr {c}");
    assert!(maxd == 0.0, "FFI vs Rust api max|d| {maxd:e}");
}

#[test]
fn ffi_handles_null_and_bad_arguments() {
    unsafe {
        rx_td_free(std::ptr::null_mut());
        rx_vc_free(std::ptr::null_mut());
        assert_eq!(rx_td_init(48000, 37, 0, 0), std::ptr::null_mut());
        // 22050 is not a supported vocoder rate
        assert_eq!(rx_vc_init(22050, 2), std::ptr::null_mut());
        let mut g = RxTdGeometry::default();
        assert_eq!(rx_td_geometry(48000, 37, 0, std::ptr::null_mut()), -1);
        assert_eq!(rx_td_geometry(48000, 37, 0, &mut g), 0);
        // -1 on bad arguments
        assert_eq!(
            rx_td_render(
                std::ptr::null_mut(),
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
                0
            ),
            -1
        );
    }
}
