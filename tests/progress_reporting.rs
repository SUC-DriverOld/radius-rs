//! Progress is telemetry: its units must match the rendered output timeline.

use std::sync::atomic::Ordering;

use radius_rs::VocoderState;

#[test]
fn vocoder_progress_counts_final_frames_at_each_tempo() {
    let frames = 48_000;
    let input: Vec<f32> = (0..frames)
        .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin() * 0.1)
        .collect();
    for tempo in [50.0, 100.0, 200.0] {
        let target = (frames as f64 * tempo / 100.0) as usize;
        let mut engine = VocoderState::new(48_000, 1);
        engine.set_ratio(3.0, tempo);
        let counter = engine.progress_counter();
        let rendered = engine.render_to(&input, target);
        assert_eq!(rendered.len(), target);
        assert_eq!(
            counter.load(Ordering::Relaxed), target as u64,
            "progress must count final frames, not the internal schedule at tempo {tempo}"
        );
        assert_eq!(engine.progress(), (target as i64, target as i64));
    }
}
