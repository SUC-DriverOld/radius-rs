use radius_rs::fft::{self, Backend};
use radius_rs::vocoder::VocoderState;

#[test]
fn short_low_rate_pitch_shift_keeps_audible_signal() {
    let mut failures = Vec::new();
    for backend in [Backend::Radix2, Backend::Rustfft] {
        fft::set_backend(backend);
        for sr in [8_000, 12_345, 16_000, 44_100, 48_000] {
            let input: Vec<f32> = (0..sr / 2)
                .map(|i| {
                    (0.1 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / sr as f64).sin()) as f32
                })
                .collect();
            for transients in [false, true] {
                let mut state = VocoderState::new(sr, 1);
                state.set_ratio(3.0, 100.0);
                state.set_transient_sched(transients);
                let output = state.render(&input);
                let peak = radius_rs::util::peak(&output);
                eprintln!("backend={backend:?} sr={sr} transients={transients} peak={peak}");
                assert_eq!(output.len(), input.len());
                assert!(output.iter().all(|sample| sample.is_finite()));
                if peak < 0.001 {
                    failures.push((backend, sr, transients, peak));
                }
            }
        }
    }
    assert!(failures.is_empty(), "near-silent output: {failures:?}");
}

#[test]
fn large_streaming_feed_matches_bounded_feeds() {
    let sr = 8_000;
    let frames = 20_000;
    let input: Vec<f32> = (0..frames)
        .map(|i| {
            let amplitude = 0.02 + 0.08 * i as f64 / frames as f64;
            (amplitude * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / sr as f64).sin()) as f32
        })
        .collect();
    for backend in [Backend::Radix2, Backend::Rustfft] {
        fft::set_backend(backend);
        let mut whole = VocoderState::new(sr, 1);
        whole.set_ratio(3.0, 100.0);
        let made = whole.feed(&input, frames);
        let mut bounded = VocoderState::new(sr, 1);
        bounded.set_ratio(3.0, 100.0);
        let mut expected_made = 0;
        for block in input.chunks(1024) {
            expected_made += bounded.feed(block, block.len());
        }
        assert_eq!(made, expected_made);
        let mut actual = vec![0.0; frames * 2];
        let mut expected = vec![0.0; frames * 2];
        let actual_len = whole.take_output(&mut actual, frames * 2);
        let expected_len = bounded.take_output(&mut expected, frames * 2);
        assert_eq!(actual_len, expected_len);
        assert!(actual_len > 0);
        assert!(radius_rs::util::peak(&expected[..expected_len]) > 0.001);
        assert!(
            actual[..actual_len] == expected[..expected_len],
            "large feed changed unread audio for {backend:?}"
        );
    }
}
