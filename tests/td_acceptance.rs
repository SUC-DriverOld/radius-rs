//! Time-domain engine integration tests.
//!
//! Most of these run on the **synthetic** programme from `common::synthetic_stereo`,
//! which is entirely our own signal: it carries a chord, a gliding vibrato voice, a
//! hairpin chirp and a silent gap, so every rate and geometry path is exercised.
//! The tests that need the real corpus (granule counts, codec behaviour on dense
//! material) call `acceptance_audio_or_skip` and report a skip when it is absent —
//! the music excerpt is not redistributed, see `tests/common/mod.rs`.

mod common;

use common::*;
use radius_rs::TdState;

/// Render a corpus WAV at `semis` with the TD engine.
fn render_wav(path: &std::path::Path, semis: f64) -> (Wav, Vec<f32>, TdState) {
    let input = read_wav(path);
    let mut st = TdState::new(input.rate, 37, 0, input.channels);
    st.set_ratio(semis, 100.0);
    let out = st.render(&input.samples, input.frames(), input.frames());
    (input, out, st)
}

/// Render the acceptance corpus, or `None` when it is not present.
fn render_acceptance(semis: f64, test: &str) -> Option<(Wav, Vec<f32>, TdState)> {
    let path = acceptance_audio_or_skip(test)?;
    Some(render_wav(&path, semis))
}

/// Duration and level are preserved, output is finite, and the engine did work.
fn assert_duration_and_level(input: &Wav, out: &[f32], st: &TdState, label: &str) {
    // `out` is interleaved, so frame count needs the channel division; getting this
    // wrong makes the ratio off by exactly the channel count.
    assert_eq!(
        out.len() % input.channels,
        0,
        "{label}: ragged interleaved output"
    );
    let frames = out.len() / input.channels;
    assert!(st.n_granule > 0, "{label}: no granules rendered");
    assert!(
        out.iter().all(|v| v.is_finite()),
        "{label}: non-finite sample in TD output"
    );
    // Duration preserving to within a granule's worth of the tail. The engine runs a
    // granule past the end and trims to the last complete granule, so on a *short*
    // signal the relative drift is larger than on the corpus: at 44.1 kHz the hop is
    // 2520 frames, which is 1.9 % of a 3 s file, and the measured drift is 0.8 %.
    let ratio = frames as f64 / input.frames() as f64;
    assert!(
        (ratio - 1.0).abs() < 0.02,
        "{label}: length ratio {ratio} out of tolerance ({frames} vs {})",
        input.frames()
    );
    // amplitude preserved, no silence, no blow-up
    let p_in = peak(&input.samples);
    let p_out = peak(out);
    assert!(
        p_out > 0.05,
        "{label}: output is (nearly) silent: peak {p_out}"
    );
    assert!(
        p_out < 4.0 * p_in.max(1e-6),
        "{label}: output peak {p_out} exploded vs input {p_in}"
    );
    let r_in = rms(&input.samples);
    let r_out = rms(out);
    assert!(
        (r_out / r_in - 1.0).abs() < 0.5,
        "{label}: rms changed too much: {r_in} -> {r_out}"
    );
}

/// Always-on: the synthetic programme must render sanely in both directions.
#[test]
fn td_synthetic_duration_and_level() {
    let path = synthetic_file("td_basic", 48_000, 48_000 * 5);
    for semis in [3.0f64, -3.0, 6.0, -6.0, 0.0] {
        let (input, out, st) = render_wav(&path, semis);
        assert_duration_and_level(&input, &out, &st, &format!("semis {semis}"));
        assert!(
            st.n_granule > 100,
            "semis {semis}: only {} granules",
            st.n_granule
        );
    }
}

/// Always-on: 44.1 kHz takes the other geometry path.
#[test]
fn td_synthetic_works_at_44100() {
    let path = synthetic_file("td_44k", 44_100, 44_100 * 3);
    let (input, out, st) = render_wav(&path, 3.0);
    assert_eq!(input.rate, 44_100);
    assert_duration_and_level(&input, &out, &st, "44.1 kHz");
}

/// Always-on: the engine must not collapse the stereo image.
#[test]
fn td_synthetic_preserves_channels() {
    let path = synthetic_file("td_stereo", 48_000, 48_000 * 2);
    let (input, out, _) = render_wav(&path, 3.0);
    assert_eq!(out.len() % input.channels, 0, "ragged interleaved output");
    let frames = out.len() / 2;
    let mut diff = 0.0f64;
    for i in 0..frames {
        diff += (out[2 * i] as f64 - out[2 * i + 1] as f64).abs();
    }
    assert!(
        diff / frames as f64 > 1e-6,
        "left and right channels are identical"
    );
}

/// The acceptance corpus must render with the granule structure the README
/// documents — this is the part that genuinely needs the real material, because
/// granule *counts* depend on the pitch tracker's behaviour on dense music.
#[test]
fn td_acceptance_granule_structure() {
    let Some((input, out, st)) = render_acceptance(3.0, "td_acceptance_granule_structure") else {
        return;
    };
    require_audio(&input, &acceptance_audio());
    assert_duration_and_level(&input, &out, &st, "+3");
    assert_eq!(st.n_granule, 1702, "+3 granule count");
    assert_eq!(st.n_transient, 27, "+3 transient count");

    let Some((_, out_neg, st_neg)) = render_acceptance(-3.0, "td_acceptance_granule_structure")
    else {
        return;
    };
    assert_eq!(st_neg.n_granule, 1719, "-3 granule count");
    assert_eq!(st_neg.n_transient, 19, "-3 transient count");
    assert_eq!(out.len() / 2, 1_393_118, "+3 output frames");
    assert_eq!(out_neg.len() / 2, 1_393_182, "-3 output frames");
}

/// The acceptance corpus at four shifts: finite, duration preserving, working.
#[test]
fn td_acceptance_both_directions() {
    let Some(path) = acceptance_audio_or_skip("td_acceptance_both_directions") else {
        return;
    };
    let input = read_wav(&path);
    require_audio(&input, &path);
    for semis in [3.0, -3.0, 6.0, -6.0] {
        let (_, out, st) = render_wav(&path, semis);
        assert_duration_and_level(&input, &out, &st, &format!("semis {semis}"));
    }
}

/// A synthetic stereo tone must come out at `2^(semis/12)` times the pitch.
#[test]
fn td_pitch_ratio_is_measured_correctly() {
    const SR: u32 = 48000;
    const FREQ: f64 = 440.0;
    let frames = SR as usize * 6;
    let mut x = vec![0.0f32; frames * 2];
    for i in 0..frames {
        let t = i as f64 / SR as f64;
        // a two-partial tone so the pitch tracker has something realistic
        let v = 0.4 * (2.0 * std::f64::consts::PI * FREQ * t).sin()
            + 0.2 * (2.0 * std::f64::consts::PI * 2.0 * FREQ * t).sin();
        x[2 * i] = v as f32;
        x[2 * i + 1] = v as f32;
    }
    let base = dominant_freq(&mono(&x, 2), SR);
    assert!(
        (base - FREQ).abs() < 2.0,
        "synthetic tone measured at {base} Hz"
    );

    for semis in [3.0f64, -3.0] {
        let mut st = TdState::new(SR, 37, 0, 2);
        st.set_ratio(semis, 100.0);
        let out = st.render(&x, frames, frames);
        assert!(out.iter().all(|v| v.is_finite()), "semis {semis}");
        // analyse a steady middle section, free of the startup transient
        let m = mono(&out, 2);
        let start = SR as usize;
        let seg = &m[start..(start + 1 << 15).min(m.len())];
        let got = dominant_freq(seg, SR);
        let want = FREQ * 2f64.powf(semis / 12.0);
        let err_semis = 12.0 * (got / want).log2();
        assert!(
            err_semis.abs() < 0.25,
            "semis {semis}: measured {got:.2} Hz, want {want:.2} Hz ({err_semis:+.3} semitones)"
        );
    }
}

/// Compare against a `pyradius` reference render when one is available.
///
/// The references are produced by `tools/gen_refs.py`; the test is skipped when
/// either the reference or the corpus is absent, so a clean checkout still passes.
#[test]
fn td_matches_pyradius_reference_when_present() {
    let ref_path = root().join(".ref").join("td_+3.wav");
    if !ref_path.exists() {
        eprintln!(
            "SKIP td_matches_pyradius_reference_when_present: no {}",
            ref_path.display()
        );
        return;
    }
    let Some(path) = acceptance_audio_or_skip("td_matches_pyradius_reference_when_present") else {
        return;
    };
    let (_, out, _) = render_wav(&path, 3.0);
    let want = read_wav(&ref_path);
    let n = (out.len() / 2).min(want.frames());
    let a = &out[..n * 2];
    let b = &want.samples[..n * 2];
    let c = corr(a, b);
    let maxd = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(c > 0.9999, "corr vs pyradius reference {c}");
    assert!(maxd < 5e-4, "max|d| vs pyradius reference {maxd:e}");
}
