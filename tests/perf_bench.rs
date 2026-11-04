//! Performance bench for the vocoder chain (goal: restore ~2x real-time,
//! bit-identical output). Opt-in:
//!
//!   cargo test --release --test perf_bench -- --ignored --nocapture
//!
//! Renders the AU 44.1k raw input (~28.7 s stereo) twice (warm + measured)
//! at +3 semitones / 100% and prints frames/s and the real-time multiple.
//! Set PERF_HASH=<sha256> to assert bit-identity against a recorded hash:
//! the run prints `perf-sha256: <hash>` on every pass.
mod common;

use common::{read_wav, root};
use radius_rs::vocoder::VocoderState;
use sha2::{Digest, Sha256};
use std::time::Instant;

fn render_once(input: &common::Wav, semis: f64, tempo: f64) -> (Vec<f32>, f64) {
    let target = (input.frames() as f64 * tempo / 100.0).round() as usize;
    let mut st = VocoderState::new(input.rate, input.channels);
    st.set_ratio(semis, tempo);
    let t0 = Instant::now();
    let out = st.render_to(&input.samples, target);
    let dt = t0.elapsed().as_secs_f64();
    (out, dt)
}

#[test]
#[ignore = "perf bench: ~1 min, prints throughput + sha256 of the render"]
fn bench_441_vc() {
    let path = root().join(".local-testdata/au-vc_441_raw.wav");
    if !path.exists() {
        eprintln!("SKIP perf_bench: no {path:?}");
        return;
    }
    let input = read_wav(&path);
    let dur = input.frames() as f64 / f64::from(input.rate);
    println!(
        "input: {} ch, {:.1}s @ {} Hz ({} frames)",
        input.channels,
        dur,
        input.rate,
        input.frames()
    );

    for (semis, tempo, tag) in [(3.0, 100.0, "+3/100"), (3.0, 200.0, "+3/200")] {
        let (warm, _) = render_once(&input, semis, tempo);
        let (out, dt) = render_once(&input, semis, tempo);
        let mut h = Sha256::new();
        for s in &out {
            h.update(s.to_le_bytes());
        }
        let hash = format!("{:x}", h.finalize());
        let rt = dur / dt;
        println!(
            "{tag}: {:.2}s wall, {:.2}x real-time ({:.0} frames/s), out {} frames",
            dt,
            rt,
            input.frames() as f64 / dt,
            out.len() / input.channels
        );
        println!("perf-sha256[{tag}]: {hash}");
        if warm.len() != out.len() || warm
            .iter()
            .zip(&out)
            .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            panic!("{tag}: warm and measured passes differ");
        }
        if let Ok(want) = std::env::var(format!("PERF_HASH_{tag}")) {
            assert_eq!(hash, want, "{tag}: output hash changed");
        }
    }
}

/// Decode `path` to interleaved f32 through the crate's own io layer
/// (symphonia, the CLI decode path).
fn decode_any(path: &std::path::PathBuf) -> Option<common::Wav> {
    let t0 = std::time::Instant::now();
    let audio = radius_rs::io::read(path).ok()?;
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "decode {}: {:.2}s, {} ch @ {} Hz, {} frames ({:.1}s audio)",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
        dt,
        audio.channels,
        audio.sample_rate,
        audio.frames(),
        audio.frames() as f64 / f64::from(audio.sample_rate),
    );
    Some(common::Wav {
        rate: audio.sample_rate,
        channels: audio.channels,
        samples: audio.samples,
    })
}

#[test]
#[ignore = "full-song perf bench: decode + render, prints real-time factors"]
fn bench_full_song() {
    let path = root().join(".local-testdata/full-song.mp3");
    if !path.exists() {
        eprintln!("SKIP bench_full_song: no {path:?}");
        return;
    }
    let input = match decode_any(&path) {
        Some(w) => w,
        None => {
            eprintln!("SKIP bench_full_song: decode failed");
            return;
        }
    };
    let dur = input.frames() as f64 / f64::from(input.rate);
    for (semis, tempo, tag) in [(3.0, 100.0, "+3/100"), (3.0, 200.0, "+3/200")] {
        let (_, warm_dt) = render_once(&input, semis, tempo);
        let (out, dt) = render_once(&input, semis, tempo);
        println!(
            "full-song {tag}: {:.2}s wall (warm pass {:.2}s), {:.2}x real-time, out {} frames",
            dt,
            warm_dt,
            dur / dt,
            out.len() / input.channels
        );
    }
}
