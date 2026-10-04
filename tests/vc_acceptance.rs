//! Phase-vocoder integration tests.
//!
//! The full vocoder render is expensive (the reference Python takes ~2 minutes
//! for 29 s of audio), so the heavy acceptance case is `#[ignore]`d by default
//! and the always-on cases use short synthetic signals.

mod common;

use common::*;
use radius_rs::vocoder::{supported_rate, vc_calc_sched, VocoderState};

#[test]
fn scheduler_closed_form_matches_reference() {
    // g < 5 is the engine's warm-up window
    for g in 0..5i64 {
        assert_eq!(vc_calc_sched(g, 1.189207115, 222.0), (0, 0, 0, 0));
    }
    // ratio >= 1 -> in_step = round(nominal), out_step = round(nominal*ratio)
    let (pos, cdel, ins, outs) = vc_calc_sched(9, 2.0, 222.0);
    assert_eq!((ins, outs), (222, 444));
    assert_eq!(cdel, 222, "g % 4 == 1 carries the input step");
    assert_eq!(pos, ((9 - 1) / 4) * 444);
    // ratio < 1 -> in_step = round(nominal*sqrt(2))
    let (_pos, _cdel, ins2, outs2) = vc_calc_sched(9, 0.5, 222.0);
    assert_eq!(ins2, (222.0f64 * std::f64::consts::SQRT_2).round() as i64);
    assert_eq!(outs2, (ins2 as f64 * 0.5).round() as i64);
}

/// The scheduler the engine actually runs, checked over a whole corpus-length run.
///
/// This replaces the old test that read the corpus' `vc_schedule.h` table out of
/// `src/`. That table is archived in `docs/reference/` rather than compiled in,
/// because no code path reads it — and it could not have been a parity oracle
/// anyway: its steps are *not constant* (it holds six distinct `(f1, f2)` pairs,
/// drifting from 222/222 to 155/311), whereas the engine's scheduler is a closed
/// form with two fixed steps per ratio. Comparing them would have compared two
/// different schedulers.
///
/// So these are the invariants that are actually true of the shipped scheduler.
#[test]
fn scheduler_is_structurally_sound_over_a_full_run() {
    // The corpus geometry: ratio sqrt(2) (the +6-semitone corpus render) with the
    // 222 nominal step the schedule header documents.
    const NOMINAL: f64 = 222.0;
    const RATIO: f64 = 1.4142135623730951;
    const COUNT: i64 = 28993;

    let (_, _, in_step, out_step) = vc_calc_sched(COUNT - 1, RATIO, NOMINAL);
    assert_eq!(
        (in_step, out_step),
        (222, 314),
        "ratio > 1 rounds both steps"
    );
    assert_eq!(
        out_step,
        (NOMINAL * RATIO).round() as i64,
        "out_step = round(nominal*ratio)"
    );

    let mut prev_pos = 0i64;
    let mut sum_cdel = 0i64;
    for g in 0..COUNT {
        let (pos, cdel, ins, outs) = vc_calc_sched(g, RATIO, NOMINAL);
        // warm-up: g < 5 emits nothing at all, including no step
        if g < 5 {
            assert_eq!((pos, cdel, ins, outs), (0, 0, 0, 0), "warm-up at g={g}");
        } else {
            assert_eq!((ins, outs), (in_step, out_step), "steps are constant");
            assert_eq!(cdel, if g % 4 == 1 { in_step } else { 0 }, "cdel at g={g}");
            assert_eq!(pos, ((g - 1) / 4) * out_step, "pos at g={g}");
        }
        assert!(pos >= prev_pos, "pos must be non-decreasing (g={g})");
        prev_pos = pos;
        sum_cdel += cdel;
    }

    // Consumption follows from the step size and the 4-phase schedule: cdel fires on
    // `g % 4 == 1`, but the warm-up suppresses the first such granule (g = 1).
    let expected = (5..COUNT).filter(|g| g % 4 == 1).count() as i64 * in_step;
    assert_eq!(
        sum_cdel, expected,
        "Σ cdel is fixed by the step and the phase"
    );
    assert_eq!(prev_pos, ((COUNT - 1) - 1) / 4 * out_step, "final pos");

    // And the engine's real 48 kHz geometry consumes a 1 393 598-frame corpus
    // without running short — that is the property the gate has to guarantee.
    let real = VocoderState::new(48_000, 2);
    let base = real.cfg.step_base;
    let ratio = 2f64.powf(3.0 / 12.0);
    let mut consumed = 0i64;
    for g in 0..COUNT {
        consumed += vc_calc_sched(g, ratio, base).1;
    }
    assert!(
        consumed >= 1_393_598,
        "at step_base={base}, ratio={ratio:.6}: Σ cdel = {consumed} would leave the \
         1 393 598-frame acceptance input unfed"
    );
}

/// The negative-semitone branch (`ratio < 1`) is the reason the corpus' separate
/// `m3`/`m6` tables are not needed: the same closed form covers it.
#[test]
fn scheduler_covers_negative_semitones() {
    const NOMINAL: f64 = 222.0;
    for semis in [-1.0f64, -3.0, -5.0, -6.0, -12.0] {
        let ratio = 2f64.powf(semis / 12.0);
        assert!(ratio < 1.0);
        let (_, _, ins, outs) = vc_calc_sched(5, ratio, NOMINAL);
        assert_eq!(ins, (NOMINAL * std::f64::consts::SQRT_2).round() as i64);
        assert_eq!(outs, (ins as f64 * ratio).round() as i64);
        assert!(outs < ins, "down-shifting must consume output slower");
        // and the schedule stays monotone for the whole run
        let mut prev = 0i64;
        for g in 0..5000i64 {
            let (pos, _, _, _) = vc_calc_sched(g, ratio, NOMINAL);
            assert!(pos >= prev, "semis={semis} pos went backwards at g={g}");
            prev = pos;
        }
    }
}

#[test]
fn supported_rates_match_the_engine() {
    assert!(supported_rate(44100));
    assert!(supported_rate(48000));
    for sr in [8000, 16000, 22050, 32000, 88200, 96000] {
        assert!(!supported_rate(sr), "{sr} must be rejected");
    }
}

#[test]
fn vocoder_config_geometry() {
    let s48 = VocoderState::new(48000, 2);
    assert_eq!(s48.cfg.n_fft, 16384);
    assert_eq!(s48.cfg.nb_bins, 8193);
    assert_eq!(s48.cfg.m_fft, 4096);
    assert_eq!(s48.cfg.mb_bins, 2049);
    assert_eq!(s48.cfg.n_write, 7184);
    assert_eq!(s48.cfg.hop, 3592);
    let s44 = VocoderState::new(44100, 2);
    assert_eq!(s44.cfg.n_fft, 8192);
    assert_eq!(s44.cfg.nb_bins, 4097);
    assert_eq!(s44.cfg.n_write, 6599);
    assert_eq!(s44.cfg.hop, 3299);
}

#[test]
fn set_ratio_matches_the_reference_chain() {
    let mut st = VocoderState::new(48000, 2);
    st.set_ratio(3.0, 100.0);
    // pr = 2^(3/12); s = 12*log2(pr) in f32; ratio = 2^(s/12)
    let pr = 2f64.powf(0.25);
    let s = 12.0f32 * (pr as f32).log2();
    let want = 2f64.powf(s as f64 / 12.0);
    assert!((st.ratio - want).abs() < 1e-12, "{} vs {want}", st.ratio);
}

/// A short synthetic tone through the whole vocoder chain must come out
/// finite, the same length, and pitched by the requested ratio.
#[test]
fn vocoder_short_render_is_sane() {
    const SR: u32 = 48000;
    const FREQ: f64 = 500.0;
    let frames = SR as usize / 2; // 0.5 s
    let mut x = vec![0.0f32; frames * 2];
    for i in 0..frames {
        let t = i as f64 / SR as f64;
        let v = 0.5 * (2.0 * std::f64::consts::PI * FREQ * t).sin();
        x[2 * i] = v as f32;
        x[2 * i + 1] = v as f32;
    }
    let mut st = VocoderState::new(SR, 2);
    st.set_ratio(3.0, 100.0);
    let out = st.render(&x);
    assert!(!out.is_empty(), "vocoder produced no output");
    assert_eq!(out.len(), x.len(), "vocoder must be duration preserving");
    assert!(
        out.iter().all(|v| v.is_finite()),
        "non-finite sample in vocoder output"
    );
    let p = peak(&out);
    assert!(p > 1e-4, "vocoder output is silent (peak {p})");
    assert!(p < 4.0, "vocoder output exploded (peak {p})");
}

/// The vocoder must cover the whole +/-36 semitone range Audition offers.
///
/// Regression test. The vocoder used to go **silent** below about -24 semitones: the
/// feed loop stopped after `target * pitch_ratio` input frames, which at a low ratio
/// is far less than the resampler needs, so `pos_1384` never passed the hop,
/// `writepos` stayed 0, and the drain took nothing. The reference renders -36 fine.
#[test]
fn vocoder_covers_the_full_pitch_range() {
    const SR: u32 = 48000;
    const FREQ: f64 = 440.0;
    let frames = SR as usize / 2; // 0.5 s
    let mut x = vec![0.0f32; frames * 2];
    for i in 0..frames {
        let t = i as f64 / SR as f64;
        let v = 0.5 * (2.0 * std::f64::consts::PI * FREQ * t).sin();
        x[2 * i] = v as f32;
        x[2 * i + 1] = v as f32;
    }
    for semis in [-36.0f64, -30.0, -24.0, -18.0, 18.0, 24.0, 30.0, 36.0] {
        let mut st = VocoderState::new(SR, 2);
        st.set_ratio(semis, 100.0);
        let out = st.render(&x);
        assert_eq!(
            out.len(),
            x.len(),
            "{semis} semitones: the vocoder must be duration preserving"
        );
        let p = peak(&out);
        assert!(
            p > 1e-3,
            "{semis} semitones: the vocoder went silent (peak {p}). \
             Check the feed loop, which must feed the whole input at low ratios"
        );
        assert!(p < 4.0, "{semis} semitones: output exploded (peak {p})");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "{semis} semitones: non-finite sample"
        );
    }
}

/// The pitch really moves across the range, measured as a spectral peak.
///
/// A pure tone makes this exact: the strongest bin must land within a couple of
/// percent of `440 * 2^(semis/12)`. The earlier octave bug and the silent tail were
/// both only visible this way.
#[test]
fn vocoder_pitch_range_lands_on_the_requested_ratio() {
    const SR: u32 = 48000;
    const FREQ: f64 = 440.0;
    let frames = SR as usize; // 1 s gives enough resolution for the low end
    let mut x = vec![0.0f32; frames * 2];
    for i in 0..frames {
        let t = i as f64 / SR as f64;
        let v = 0.5 * (2.0 * std::f64::consts::PI * FREQ * t).sin();
        x[2 * i] = v as f32;
        x[2 * i + 1] = v as f32;
    }
    for semis in [-36.0f64, -24.0, -12.0, 0.0, 12.0, 24.0, 36.0] {
        let mut st = VocoderState::new(SR, 2);
        st.set_ratio(semis, 100.0);
        let out = st.render(&x);
        let mono: Vec<f64> = out.chunks_exact(2).map(|f| f[0] as f64).collect();
        let got = strongest_bin_hz(&mono, SR as f64);
        let want = FREQ * 2f64.powf(semis / 12.0);
        let err = (got / want - 1.0).abs();
        assert!(
            err < 0.03,
            "{semis} semitones: wanted {want:.1} Hz, strongest bin was {got:.1} Hz \
             ({:.2}% off)",
            err * 100.0
        );
    }
}

/// `--no-preserve-voice` is exactly `--formant-shift <semitones>`.
///
/// Both mean "no envelope correction": dropping preservation switches the formant
/// operator off (`active = 0`), while a formant shift equal to the pitch leaves it on
/// with a ratio of exactly 1.0, which `FormantState::apply` short-circuits. The two
/// paths must therefore agree bit for bit — this pins that down, and with it both the
/// operator's `ratio == 1.0` fast path and the `--formant-shift` calibration basis.
#[test]
fn no_preserve_voice_equals_formant_shift_at_the_pitch() {
    const SR: u32 = 48_000;
    // A harmonic series with a formant, so the envelope is actually doing something.
    let frames = SR as usize / 2;
    let mut x = vec![0.0f32; frames * 2];
    for i in 0..frames {
        let t = i as f64 / SR as f64;
        let mut v = 0.0f64;
        for k in 1..=40 {
            let f = 180.0 * k as f64;
            if f > SR as f64 * 0.45 {
                break;
            }
            let env = 1.0 / (1.0 + ((f - 2200.0) / 1100.0).powi(2)) + 0.01;
            v += env * (2.0 * std::f64::consts::PI * f * t).sin();
        }
        let s = 0.4 * v;
        x[2 * i] = s as f32;
        x[2 * i + 1] = s as f32;
    }

    for semis in [3.0f64, -3.0, 7.0, -7.0, 12.0, -12.0] {
        let mut off = VocoderState::new(SR, 2);
        off.set_ratio(semis, 100.0);
        off.set_preserve_voice(false);

        let mut on = VocoderState::new(SR, 2);
        on.set_ratio(semis, 100.0);
        on.set_formant_shift(semis);

        assert!(
            on.preserve_voice(),
            "{semis}: a formant shift must not switch preservation off — the two settings \
             agree by different routes, not by both disabling the operator"
        );
        assert!(
            (on.formant.cfg.ratio - 1.0).abs() < 1e-6,
            "{semis}: a formant shift equal to the pitch must drive the operator's ratio to \
             exactly 1.0, got {}",
            on.formant.cfg.ratio
        );

        let a = off.render(&x);
        let b = on.render(&x);
        assert_eq!(a.len(), b.len(), "{semis}: length mismatch");
        assert!(
            a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()),
            "{semis} semitones: --no-preserve-voice and --formant-shift {semis} must be \
             bit-identical, but the renders differ"
        );
    }
}

/// Frequency of the strongest DFT bin of `x`, refined to sub-bin accuracy.
fn strongest_bin_hz(x: &[f64], rate: f64) -> f64 {
    // Use the middle half, windowed, so the granule edges do not dominate.
    let q = x.len() / 4;
    let seg = &x[q..3 * q];
    let n = seg.len();
    let w: Vec<f64> = (0..n)
        .map(|i| {
            let h = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos();
            seg[i] * h
        })
        .collect();
    let mut best = (0usize, f64::MIN);
    for k in 2..n / 2 {
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (i, v) in w.iter().enumerate() {
            let a = -2.0 * std::f64::consts::PI * k as f64 * i as f64 / n as f64;
            re += v * a.cos();
            im += v * a.sin();
        }
        let m = re * re + im * im;
        if m > best.1 {
            best = (k, m);
        }
    }
    // Parabolic refinement around the peak bin.
    let k = best.0;
    let mag = |k: usize| -> f64 {
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (i, v) in w.iter().enumerate() {
            let a = -2.0 * std::f64::consts::PI * k as f64 * i as f64 / n as f64;
            re += v * a.cos();
            im += v * a.sin();
        }
        (re * re + im * im).sqrt()
    };
    let (a, b, c) = (mag(k - 1), mag(k), mag(k + 1));
    let d = if (a - 2.0 * b + c) != 0.0 {
        0.5 * (a - c) / (a - 2.0 * b + c)
    } else {
        0.0
    };
    (k as f64 + d) * rate / n as f64
}

/// The vocoder over the synthetic programme: both directions, both sample rates,
/// finite, duration preserving, audible, and actually pitched by the ratio.
///
/// This is the always-on substitute for the corpus render, so it has to check more
/// than "it ran": the synthetic signal carries a chirp and a gliding voice, which is
/// what stresses the peak search and phase unwrapping.
#[test]
fn vocoder_synthetic_render_is_sane() {
    for (sr, semis) in [(48_000u32, 3.0f64), (48_000, -3.0), (44_100, 3.0)] {
        let frames = (sr as usize / 2).max(1); // 0.5 s keeps the test fast
        let w = synthetic_stereo(sr, frames);
        let mut st = VocoderState::new(sr, w.channels);
        st.set_ratio(semis, 100.0);
        let out = st.render(&w.samples);
        let label = format!("{sr} Hz {semis:+}");
        assert!(!out.is_empty(), "{label}: no output");
        assert_eq!(
            out.len(),
            w.samples.len(),
            "{label}: not duration preserving"
        );
        assert!(
            out.iter().all(|v| v.is_finite()),
            "{label}: non-finite sample"
        );
        let p = peak(&out);
        assert!(p > 1e-3, "{label}: output is silent (peak {p})");
        assert!(p < 8.0, "{label}: output exploded (peak {p})");
        assert!(st.granules_made() > 0, "{label}: no granules processed");
    }
}

/// The vocoder must process the two channels separately.
///
/// This is a regression test for a real defect: `feed()` split only channel 0 into
/// bands and let both channels' analysis read that same data, so the output was the
/// left channel synthesised twice — a phantom centre. A phase-inverted pair came back
/// **in** phase (correlation −1 → +1) and a hard-panned right channel came back at the
/// left channel's level. Neither is audible as anything but mono.
#[test]
fn vocoder_preserves_stereo_separation() {
    const SR: u32 = 48_000;
    let frames = SR as usize / 2; // 0.5 s
    let mut anti = Vec::with_capacity(frames * 2);
    let mut lopsided = Vec::with_capacity(frames * 2);
    for i in 0..frames {
        let t = i as f64 / SR as f64;
        let a = 0.5 * (2.0 * std::f64::consts::PI * 440.0 * t).sin();
        let b = 0.2 * (2.0 * std::f64::consts::PI * 1200.0 * t).sin();
        anti.push(a as f32);
        anti.push(-a as f32);
        lopsided.push(a as f32);
        lopsided.push(b as f32);
    }

    // 1. an anti-phase pair must stay anti-phase
    let mut st = VocoderState::new(SR, 2);
    st.set_ratio(3.0, 100.0);
    let out = st.render(&anti);
    let (l, r) = split(&out);
    let c = corr(&l, &r);
    assert!(
        c < -0.9,
        "anti-phase input came back with correlation {c:+.4}: the channels are being \
         merged instead of processed separately"
    );

    // 2. a quieter, different right channel must stay quieter and different
    let mut st = VocoderState::new(SR, 2);
    st.set_ratio(3.0, 100.0);
    let out = st.render(&lopsided);
    let (l, r) = split(&out);
    let dl = 20.0 * (rms(&r) / rms(&l)).log10();
    assert!(
        (-12.0..-4.0).contains(&dl),
        "right channel level is {dl:+.2} dB relative to left; the input was -7.96 dB, so \
         the right channel is not being carried through"
    );
    assert!(
        corr(&l, &r).abs() < 0.5,
        "unrelated left/right content came back correlated: {}",
        corr(&l, &r)
    );

    // 3. identical channels must still come out identical (no gratuitous divergence)
    let mut mono_pair = Vec::with_capacity(frames * 2);
    for i in 0..frames {
        let t = i as f64 / SR as f64;
        let a = 0.5 * (2.0 * std::f64::consts::PI * 440.0 * t).sin();
        mono_pair.push(a as f32);
        mono_pair.push(a as f32);
    }
    let mut st = VocoderState::new(SR, 2);
    st.set_ratio(3.0, 100.0);
    let out = st.render(&mono_pair);
    let (l, r) = split(&out);
    assert!(
        corr(&l, &r) > 0.9999,
        "a dual-mono input must stay dual-mono, got correlation {}",
        corr(&l, &r)
    );
}

/// Split interleaved stereo into `(left, right)`.
fn split(x: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = x.len() / 2;
    (
        (0..n).map(|i| x[2 * i]).collect(),
        (0..n).map(|i| x[2 * i + 1]).collect(),
    )
}

/// Full-length parity against the `pyradius` reference render, when present.
#[test]
#[ignore = "runs the 29 s acceptance render (slow)"]
fn vocoder_matches_pyradius_reference() {
    let ref_path = root().join(".ref").join("vc_+3.wav");
    if !ref_path.exists() {
        eprintln!(
            "SKIP vocoder_matches_pyradius_reference: no {}",
            ref_path.display()
        );
        return;
    }
    let Some(path) = acceptance_audio_or_skip("vocoder_matches_pyradius_reference") else {
        return;
    };
    let input = read_wav(&path);
    require_audio(&input, &path);
    let mut st = VocoderState::new(input.rate, input.channels);
    st.set_ratio(3.0, 100.0);
    let out = st.render(&input.samples);
    let want = read_wav(&ref_path);
    let nch = input.channels;
    let n = (out.len() / nch).min(want.frames());
    let a = &out[..n * nch];
    let b = &want.samples[..n * nch];
    let c = corr(a, b);
    let maxd = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(c > 0.95, "corr vs pyradius vocoder reference {c}");
    assert!(maxd < 0.2, "max|d| vs pyradius vocoder reference {maxd:e}");
}

/// Full-length acceptance render (structure only, no reference needed).
#[test]
#[ignore = "runs the 29 s acceptance render (slow)"]
fn vocoder_acceptance_render() {
    let Some(path) = acceptance_audio_or_skip("vocoder_acceptance_render") else {
        return;
    };
    let input = read_wav(&path);
    require_audio(&input, &path);
    let mut st = VocoderState::new(input.rate, input.channels);
    st.set_ratio(3.0, 100.0);
    let out = st.render(&input.samples);
    assert_eq!(out.len(), input.samples.len());
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(peak(&out) > 0.05);
}
