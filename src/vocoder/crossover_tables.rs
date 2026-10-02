//! Engine data tables, compiled in as Rust source.
//!
//! The vocoder reads six constant tables (`rx_crossover_create`, `rx_vc_init`).
//! They are *data*, not formulas: the C compiler folds them into `__const` and the
//! reference Python reads them through `pyradius/tables.py`.
//!
//! ## Where they live
//!
//! Each table is a generated `.rs` file under `src/vocoder/table_data/`, compiled
//! into this module with `include!`. Two consequences matter:
//!
//! * **The binaries are self-contained.** Nothing is read from disk at build time
//!   or at run time — no build script, no asset directory, no data file that can
//!   go missing next to a copied `radius.exe` or `radius_rs.dll`. A `cargo build`
//!   on a bare checkout produces a working executable, and copying just the `.exe`
//!   (or just the `.dll`) anywhere keeps it working.
//! * **The values are source**, so they are reviewable, diffable and type checked
//!   by the same compiler as everything else.
//!
//! The literals are exact *bit patterns*: `f32` values are written
//! `f32::from_bits(0x…)` and `i32` values as integers, so no decimal formatting
//! step can round a value a second time. `tools/gen_table_source.py` is the
//! converter that produced these files from the parse of the original C headers;
//! its `--check` mode regenerates the text and compares it byte for byte, and
//! `docs/TABLE_PROVENANCE.md` records where the numbers came from.
//!
//! ## Cost
//!
//! The six tables are ~1.7 MB of source and, as `const` arrays, are materialised
//! in the binary image. Every one of them is read by the engine:
//!
//! | table | read by | size |
//! |---|---|---|
//! | `XOVER_FIR_48K` / `XOVER_FIR_44K` | [`fir`] → `Crossover` | 32 KB each |
//! | `VC_WIN_48K` / `VC_WIN_44K` | [`win`] → `VocoderState::new` | 112 / 103 KB |
//! | `VC_SYNTH_WIN_A_44K` / `_B_44K` | [`synth_44k`] | 26 KB each |
//!
//! Nothing here is dead weight: a table that no code path reads does not belong in
//! `src/`. The nomodule reference data that used to sit here — the research
//! corpus' per-granule schedule (`vc_schedule.h`, 28993 granules × 4 `i32`
//! columns) — has been moved to `docs/reference/`, because the engine does not
//! read it (see below) and it was 464 KB, 90 % of the table payload.
//!
//! ### Why there is no schedule table
//!
//! `vocoder_core.c` states it outright ("本文件不引用 vc_schedule.h … 的逐粒真值表"):
//! the engine computes its schedule with `rx_vc_calc_sched`, ported as
//! [`crate::vocoder::vc_calc_sched`], which also covers the negative-semitone case
//! (`ratio < 1` → `round(nominal·√2)`). The header's table is the reverse-engineering
//! corpus' *record* of one particular render, and it is not a parity oracle: its
//! step sizes differ from the engine's own (48k 264/373, 44.1k 204/288, against the
//! table's 222/444 early and 155/311 late), so comparing the two would test
//! nothing. `docs/TABLE_PROVENANCE.md` keeps the details and
//! `docs/reference/vc_schedule.bin` keeps the bytes.

/// `f32` tables, in the reference engine's declaration order.
mod data {
    #![allow(clippy::all)]
    include!("table_data/xover_fir_48k.rs");
    include!("table_data/xover_fir_44k.rs");
    include!("table_data/vc_win_48k.rs");
    include!("table_data/vc_win_44k.rs");
    include!("table_data/vc_synth_win_a_44k.rs");
    include!("table_data/vc_synth_win_b_44k.rs");
}

use data::*;

// Element counts. Each is an array length in the generated files too, so a
// truncated or mismatched table is a compile error rather than a silent
// shortening — and the `const` assertion block below catches a wrong *count* as
// well.
pub const FIR_48K_LEN: usize = 8192;
pub const FIR_44K_LEN: usize = 8192;
pub const WIN_48K_LEN: usize = 28736;
pub const WIN_44K_LEN: usize = 26396;
pub const SYNTH_A_44K_LEN: usize = 6599;
pub const SYNTH_B_44K_LEN: usize = 6599;

/// Reinterpret a `const` array of `f32` as a slice.
///
/// No copy and no allocation: the array is a static, so the slice points straight
/// at the binary's own image. `f32` is not `u8`, so the array is already
/// correctly aligned for its own type.
#[inline(always)]
const fn as_slice<const N: usize>(a: &'static [f32; N]) -> &'static [f32] {
    a
}

/// Crossover FIR taps for `sr` (`48000` or `44100`), `[4][2048]` row-major.
///
/// `pyradius.tables.get_tables(sr)["fir"]`: 8192 `f32`, band `b` at
/// `b * 2048 .. (b + 1) * 2048`, in the header's original (unreversed) order —
/// [`crate::vocoder::Crossover::new`] reverses each band, as the C does in
/// `rx_crossover_create`.
///
/// # Panics
/// On a sample rate the engine does not support (`ValueError` in the Python).
pub fn fir(sr: u32) -> &'static [f32] {
    match sr {
        48_000 => as_slice(&XOVER_FIR_48K),
        44_100 => as_slice(&XOVER_FIR_44K),
        other => panic!("crossover tables: unsupported sample rate {other}"),
    }
}

/// Default vocoder window table for `sr`, flattened `4 x n_write` (`f32`).
///
/// `pyradius.tables.get_tables(sr)["win"]`: 28736 values at 48 kHz
/// (`4 * 7184`), 26396 at 44.1 kHz (`4 * 6599`); band `b` starts at
/// `b * n_write`.
///
/// # Panics
/// On a sample rate the engine does not support.
pub fn win(sr: u32) -> &'static [f32] {
    match sr {
        48_000 => as_slice(&VC_WIN_48K),
        44_100 => as_slice(&VC_WIN_44K),
        other => panic!("vocoder window tables: unsupported sample rate {other}"),
    }
}

/// 44.1 kHz synthesis windows `(a, b)` — 6599 `f32` each
/// (`rx_vc_synth_win_a_44k` / `rx_vc_synth_win_b_44k`).
pub fn synth_44k() -> (&'static [f32], &'static [f32]) {
    (as_slice(&VC_SYNTH_WIN_A_44K), as_slice(&VC_SYNTH_WIN_B_44K))
}

// ---- per-rate aliases ----
//
// `src/vocoder/mod.rs` re-exports this module as `tables_data`
// (`pub use crossover_tables as tables_data;`), which is how the orchestration
// layer names the tables by rate; these are the same slices.

/// Crossover FIR taps, 48 kHz.
pub fn fir_48k() -> &'static [f32] {
    as_slice(&XOVER_FIR_48K)
}
/// Crossover FIR taps, 44.1 kHz.
pub fn fir_44k() -> &'static [f32] {
    as_slice(&XOVER_FIR_44K)
}
/// Default vocoder window table, 48 kHz.
pub fn win_48k() -> &'static [f32] {
    as_slice(&VC_WIN_48K)
}
/// Default vocoder window table, 44.1 kHz.
pub fn win_44k() -> &'static [f32] {
    as_slice(&VC_WIN_44K)
}

// The declared counts and the compiled-in array lengths must agree. This is a
// `const` block, so a mismatch fails the build instead of producing a short table
// at run time.
const _: () = {
    assert!(XOVER_FIR_48K.len() == FIR_48K_LEN);
    assert!(XOVER_FIR_44K.len() == FIR_44K_LEN);
    assert!(VC_WIN_48K.len() == WIN_48K_LEN);
    assert!(VC_WIN_44K.len() == WIN_44K_LEN);
    assert!(VC_SYNTH_WIN_A_44K.len() == SYNTH_A_44K_LEN);
    assert!(VC_SYNTH_WIN_B_44K.len() == SYNTH_B_44K_LEN);
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Counts from `pyradius/tests/test_smoke.py` and the C `#define`s.
    #[test]
    fn table_element_counts() {
        assert_eq!(fir(48_000).len(), 4 * 2048);
        assert_eq!(fir(44_100).len(), 4 * 2048);
        assert_eq!(win(48_000).len(), 28736);
        assert_eq!(win(44_100).len(), 26396);
        let (a, b) = synth_44k();
        assert_eq!(a.len(), 6599);
        assert_eq!(b.len(), 6599);
    }

    /// The declared consts must agree with the slices they size.
    #[test]
    fn generated_lengths_agree() {
        assert_eq!(FIR_48K_LEN, fir_48k().len());
        assert_eq!(FIR_44K_LEN, fir_44k().len());
        assert_eq!(WIN_48K_LEN, win_48k().len());
        assert_eq!(WIN_44K_LEN, win_44k().len());
        assert_eq!(SYNTH_A_44K_LEN + SYNTH_B_44K_LEN, 2 * 6599);
    }

    #[test]
    fn float_tables_are_finite() {
        for sr in [48_000u32, 44_100] {
            assert!(fir(sr).iter().all(|v| v.is_finite()), "fir {sr}");
            assert!(win(sr).iter().all(|v| v.is_finite()), "win {sr}");
        }
        let (a, b) = synth_44k();
        assert!(a.iter().all(|v| v.is_finite()));
        assert!(b.iter().all(|v| v.is_finite()));
    }

    /// Spot values pin the conversion end to end: first band, first taps and the
    /// 44.1 kHz FIR's larger first tap, read straight out of the headers as
    /// decimals and rounded once to `f32`.
    ///
    /// The expectations are written as decimal literals on purpose: they are the
    /// numbers in the C header, so this test fails if the generator ever starts
    /// emitting a different bit pattern for the same header value.
    #[test]
    fn parser_spot_values() {
        assert_eq!(fir_48k()[0], 1.343_229_53e-08f32);
        assert_eq!(fir_48k()[1], 1.396_580_01e-08f32);
        assert_eq!(fir_44k()[0], 1.167_316_64e-08f32);
        // the windows open with a zero run (the C zero-fills the edges)
        assert_eq!(win_48k()[0], 0.0);
        assert_eq!(win_44k()[0], 0.0);
        assert_eq!(synth_44k().0[0], 0.0);
    }

    /// The two rates must be genuinely different banks, not one table reused.
    #[test]
    fn rates_are_distinct_banks() {
        assert!(!std::ptr::eq(fir(48_000), fir(44_100)));
        assert!(!std::ptr::eq(win(48_000), win(44_100)));
        assert_ne!(fir_48k()[0], fir_44k()[0]);
        assert_ne!(win_48k().len(), win_44k().len());
    }

    /// The embedded tables must be the *same memory*: taking the slice twice has
    /// to return the same pointer, and the slices must not have been copied.
    #[test]
    fn tables_are_static_not_copied() {
        assert!(std::ptr::eq(fir(48_000), fir_48k()));
        assert!(std::ptr::eq(win(44_100), win_44k()));
    }

    #[test]
    #[should_panic(expected = "unsupported sample rate")]
    fn unsupported_rate_panics() {
        let _ = fir(22_050);
    }
}
