//! Transient-scheduler integration tests: engagement on percussive material
//! and the on/off behavioural contract.
//!
//! The subsystem defaults to ON (AU parity); pyradius-parity tests opt out
//! explicitly.

use radius_rs::vocoder::VocoderState;

/// A synthetic drum train: 3 kHz sine "sticks" with exponential decay every
/// 0.25 s over a quiet 220 Hz tone, 2 s at 48 kHz stereo.
fn drum_train() -> Vec<f32> {
    let sr = 48_000.0f32;
    let n = (2.0 * sr) as usize;
    let mut x = vec![0.0f32; 2 * n];
    for i in 0..n {
        let t = i as f32 / sr;
        // 8 hits at 4 Hz.
        let phase = t * 4.0;
        let hit_t = t - phase.floor() / 4.0;
        let stick = if hit_t < 0.05 {
            let env = (-hit_t * 60.0).exp();
            (2.0 * std::f32::consts::PI * 3000.0 * hit_t).sin() * env * 0.9
        } else {
            0.0
        };
        let body = (2.0 * std::f32::consts::PI * 220.0 * t).sin() * 0.02;
        let s = stick + body;
        x[2 * i] = s;
        x[2 * i + 1] = s;
    }
    x
}

#[test]
fn transient_sched_engages_on_drum_train() {
    let x = drum_train();
    let mut st = VocoderState::new(48_000, 2);
    st.set_ratio(4.0, 100.0);
    let out = st.render(&x);
    let (events, state2) = st.transient_stats();
    assert!(events >= 5, "expected the drum hits to score, got {events}");
    assert!(state2 > 0, "scheduler should reach state 2, got {state2}");
    // Output must be sane: full length, bounded.
    assert_eq!(out.len(), x.len());
    let peak = out.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(peak > 0.01 && peak < 4.0, "peak {peak}");
}

#[test]
fn transient_sched_default_is_on() {
    let mut st = VocoderState::new(48_000, 2);
    assert!(st.transient_sched(), "the subsystem defaults to on (AU parity)");
    st.set_transient_sched(false);
    assert!(!st.transient_sched());
}

#[test]
fn transient_sched_changes_pitch_up_output() {
    // The subsystem (gates at default; hop re-timing is the experimental
    // `ts_hop_override` path) must change the render.
    let x = drum_train();
    let mut off = VocoderState::new(48_000, 2);
    off.set_ratio(4.0, 100.0);
    off.set_transient_sched(false);
    let out_off = off.render(&x);
    assert_eq!(off.transient_stats(), (0, 0), "off must not touch the counters");
    let mut on = VocoderState::new(48_000, 2);
    on.set_ratio(4.0, 100.0);
    let out_on = on.render(&x);
    assert!(on.transient_stats().0 > 0, "on must see events");
    assert!(on.transient_stats().1 > 0, "gates must reach state 2");
    let same = out_off
        .iter()
        .zip(out_on.iter())
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    let frac = same as f32 / out_off.len() as f32;
    assert!(
        frac < 0.999,
        "scheduler must alter the render, {frac:.4} of samples identical"
    );
}

#[test]
fn transient_hop_override_alters_timing() {
    // The dual-queue re-timing path (default since the §13.6 port) must itself
    // change the render relative to gates-only — the baseline now opts out
    // explicitly.
    let x = drum_train();
    let mut gates = VocoderState::new(48_000, 2);
    gates.set_ratio(4.0, 100.0);
    gates.ts_hop_override = false;
    let out_gates = gates.render(&x);
    let mut hop = VocoderState::new(48_000, 2);
    hop.set_ratio(4.0, 100.0);
    hop.ts_hop_override = true;
    let out_hop = hop.render(&x);
    assert_ne!(out_gates, out_hop);
}

#[test]
fn transient_sched_engages_on_tempo_stretch() {
    // Down-tempo (slope < 1) takes the compression-side branch into state 2.
    let x = drum_train();
    let mut st = VocoderState::new(48_000, 2);
    st.set_ratio(0.0, 50.0);
    st.set_transient_sched(false); // off-baseline for the same render length
    let _ = st.render_to(&x, x.len() / 2);
    assert_eq!(st.transient_stats(), (0, 0));

    let mut st = VocoderState::new(48_000, 2);
    st.set_ratio(0.0, 50.0);
    let out = st.render_to(&x, x.len() / 2);
    let (events, state2) = st.transient_stats();
    assert!(events >= 3, "hits should score at tempo 50, got {events}");
    assert!(state2 > 0, "compress-side should reach state 2, got {state2}");
    assert!(!out.is_empty());
}

#[test]
fn vocoder_does_not_accumulate_beat_drift() {
    const RATE: usize = 44_100;
    const HITS: usize = 24;
    let beat_frame = |i| RATE / 4 + i * (RATE / 2);
    let mut rng = 241u32;
    let hit: Vec<_> = (0..RATE * 60 / 1000)
        .map(|i| {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = f64::from(rng) / f64::from(u32::MAX) - 0.5;
            (noise * (-(i as f64) / (RATE as f64 * 0.008)).exp()) as f32
        })
        .collect();
    let mut input = vec![0.0; RATE * HITS / 2];
    for i in 0..HITS {
        let start = beat_frame(i);
        input[start..start + hit.len()].copy_from_slice(&hit);
    }

    // Use several attacks at each end so a constant offset or the jitter of
    // one hit is not mistaken for drift. The old schedule rounds +4 up and
    // +5 down, producing about 10 ms of drift in opposite directions here.
    for (pitch, tempo) in [(4.0, 100.0), (5.0, 100.0), (4.0, 90.0)] {
        let stretch = tempo / 100.0;
        let target = (input.len() as f64 * stretch).round() as usize;
        let mut st = VocoderState::new(RATE as u32, 1);
        st.set_ratio(pitch, tempo);
        let out = st.render_to(&input, target);
        assert_eq!(out.len(), target);

        let mean_offset = |beats: std::ops::Range<usize>| {
            let n = beats.len() as f64;
            beats
                .map(|i| {
                    let expected = (beat_frame(i) as f64 * stretch).round() as usize;
                    let radius = RATE / 8;
                    let (sum, weighted) = out[expected - radius..expected + radius]
                        .iter()
                        .enumerate()
                        .fold((0.0, 0.0), |(sum, weighted), (k, &x)| {
                            let energy = f64::from(x).powi(2);
                            (sum + energy, weighted + energy * k as f64)
                        });
                    assert!(sum > 0.01, "missing attack {i} at {pitch:+} st, {tempo}%");
                    weighted / sum - radius as f64
                })
                .sum::<f64>()
                / n
        };
        let drift_ms = (mean_offset(HITS - 4..HITS) - mean_offset(0..4)) * 1000.0 / RATE as f64;
        assert!(
            drift_ms.abs() < 2.0,
            "{pitch:+} st, {tempo}%: {drift_ms:+.3} ms drift"
        );
    }
}

#[test]
fn vocoder_reuses_the_output_ring_without_impulses() {
    for (rate, pitch) in [(44_100, 4.0), (48_000, 5.0)] {
        let n = rate as usize;
        let input: Vec<f32> = (0..6 * n)
            .map(|i| {
                let phase = std::f64::consts::TAU * 440.0 * i as f64 / rate as f64;
                ((11_000.0 * phase.sin()).round() / 32_768.0) as f32
            })
            .collect();
        let mut st = VocoderState::new(rate, 1);
        st.set_ratio(pitch, 100.0);
        let out = st.render(&input);
        assert_eq!(out.len(), input.len());

        // Skip startup and the tail, but include several ring passes. A
        // stale sample creates a sharp jump even if none of the PCM clips.
        let steady = &out[n..5 * n];
        let peak = radius_rs::util::peak(steady);
        assert!(peak > 0.002, "missing steady tone at {rate} Hz");
        let (i, jump) = steady
            .windows(3)
            .enumerate()
            .map(|(i, w)| (i + n + 1, (w[0] - 2.0 * w[1] + w[2]).abs()))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        assert!(
            jump < peak * 0.1,
            "{rate} Hz, {pitch:+} st: impulse {jump:.5} against peak {peak:.5} at frame {i}"
        );
    }
}
