//! C ABI surface (`include/radius_rs.h`).
//!
//! The `cdylib`/`staticlib` artefacts export these symbols; the `rlib` build
//! keeps them available to Rust callers too. Every entry point is
//! `#[no_mangle]`/`extern "C"` and uses opaque handles so the layout of the
//! Rust state structs never leaks into the header.

use std::os::raw::{c_char, c_double, c_float, c_int, c_void};

use crate::td::TdState;
use crate::vocoder::VocoderState;

/// `radius_version` — NUL-terminated static version string.
#[no_mangle]
pub extern "C" fn radius_version() -> *const c_char {
    static VERSION: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();
    VERSION.as_ptr() as *const c_char
}

// ---------------------------------------------------------------------------
// Time-domain engine
// ---------------------------------------------------------------------------

/// `rx_td_init` — allocate a TD state. `sr` in Hz, `quality` and `solo` as in
/// the reference renderer, `nch` channels.
///
/// Returns NULL when `nch` is 0.
#[no_mangle]
pub extern "C" fn rx_td_init(sr: u32, quality: c_int, solo: c_int, nch: c_int) -> *mut c_void {
    if nch <= 0 {
        return std::ptr::null_mut();
    }
    let st = Box::new(TdState::new(sr, quality, solo, nch as usize));
    Box::into_raw(st) as *mut c_void
}

/// `rx_td_free`.
///
/// # Safety
/// `st` must come from [`rx_td_init`] and not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn rx_td_free(st: *mut c_void) {
    if !st.is_null() {
        drop(Box::from_raw(st as *mut TdState));
    }
}

/// `rx_td_set_ratio` — `semis` semitones at `tempo` percent.
///
/// # Safety
/// `st` must be a live handle from [`rx_td_init`].
#[no_mangle]
pub unsafe extern "C" fn rx_td_set_ratio(st: *mut c_void, semis: c_double, tempo: c_double) {
    if st.is_null() {
        return;
    }
    let st = &mut *(st as *mut TdState);
    st.set_ratio(semis, tempo);
}

/// `rx_td_render` — render `nframes` interleaved frames from `in` into `out`
/// (`out_cap` frames). Returns the number of frames written, or -1 on a bad
/// argument.
///
/// # Safety
/// `st` must be a live handle; `in` must hold `nframes*nch` floats and `out`
/// must hold `out_cap*nch` floats for the handle's channel count.
#[no_mangle]
pub unsafe extern "C" fn rx_td_render(
    st: *mut c_void,
    input: *const c_float,
    nframes: i64,
    out: *mut c_float,
    out_cap: i64,
) -> i64 {
    if st.is_null() || input.is_null() || nframes < 0 {
        return -1;
    }
    let st = &mut *(st as *mut TdState);
    let n = nframes as usize;
    let src = std::slice::from_raw_parts(input, n * st.nch);
    // The C ABI keeps the reference contract: produce as many frames as were fed.
    let y = st.render(src, n, n);
    let frames = (y.len() / st.nch).min(out_cap.max(0) as usize);
    if !out.is_null() && frames > 0 {
        std::ptr::copy_nonoverlapping(y.as_ptr(), out, frames * st.nch);
    }
    frames as i64
}

/// `rx_td_geometry` — derived geometry of a configuration, for diagnostics.
#[repr(C)]
#[repr(C)]
pub struct RxTdGeometry {
    pub hop: u32,
    pub f28: u32,
    pub win_max: u32,
    pub pitch_n: u32,
    pub pitch_l1: u32,
    pub pitch_maxbin: u32,
    pub pitch_taper_len: u32,
    pub pitch_lo: u32,
    pub pitch_hi: u32,
}

/// # Safety
/// `out` must be a valid writable pointer.
#[no_mangle]
pub unsafe extern "C" fn rx_td_geometry(
    sr: u32,
    quality: c_int,
    solo: c_int,
    out: *mut RxTdGeometry,
) -> c_int {
    if out.is_null() {
        return -1;
    }
    let st = TdState::new(sr, quality, solo, 2);
    let g = st.geometry();
    *out = RxTdGeometry {
        hop: st.hop,
        f28: st.f28,
        win_max: st.win_max,
        pitch_n: g.n as u32,
        pitch_l1: g.l1 as u32,
        pitch_maxbin: g.maxbin as u32,
        pitch_taper_len: g.taper_len as u32,
        pitch_lo: g.lo as u32,
        pitch_hi: g.hi as u32,
    };
    0
}

// ---------------------------------------------------------------------------
// Vocoder engine
// ---------------------------------------------------------------------------

/// `rx_vc_init` — allocate a vocoder state, or NULL when the sample rate is
/// unsupported (anything other than 44100/48000).
///
/// The reference's `precision` argument is deliberately **not** part of this API. It
/// was never a precision: it indexed the overlap-add write-gain table, so it only
/// changed the output level, and 3..=9 were byte-identical. The state is created at
/// the reference's value, which is what the parity corpus is measured at; use the
/// CLI's `--gain` (or scale the samples) for level.
#[no_mangle]
pub extern "C" fn rx_vc_init(sr: u32, nch: c_int) -> *mut c_void {
    if nch <= 0 || !crate::vocoder::supported_rate(sr) {
        return std::ptr::null_mut();
    }
    let st = Box::new(VocoderState::new(sr, nch as usize));
    Box::into_raw(st) as *mut c_void
}

/// # Safety
/// `st` must come from [`rx_vc_init`] and not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn rx_vc_free(st: *mut c_void) {
    if !st.is_null() {
        drop(Box::from_raw(st as *mut VocoderState));
    }
}

/// # Safety
/// `st` must be a live vocoder handle.
#[no_mangle]
pub unsafe extern "C" fn rx_vc_set_ratio(st: *mut c_void, semis: c_double, tempo: c_double) {
    if st.is_null() {
        return;
    }
    (*(st as *mut VocoderState)).set_ratio(semis, tempo);
}

/// `rx_vc_render` — render `nframes` interleaved frames. Returns frames written.
///
/// # Safety
/// Same contract as [`rx_td_render`].
#[no_mangle]
pub unsafe extern "C" fn rx_vc_render(
    st: *mut c_void,
    input: *const c_float,
    nframes: i64,
    out: *mut c_float,
    out_cap: i64,
) -> i64 {
    if st.is_null() || input.is_null() || nframes < 0 {
        return -1;
    }
    let st = &mut *(st as *mut VocoderState);
    let n = nframes as usize;
    let nch = st.cfg.nch;
    let src = std::slice::from_raw_parts(input, n * nch);
    let y = st.render(src);
    let frames = (y.len() / nch).min(out_cap.max(0) as usize);
    if !out.is_null() && frames > 0 {
        std::ptr::copy_nonoverlapping(y.as_ptr(), out, frames * nch);
    }
    frames as i64
}
