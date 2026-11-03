//! Compare encoded file sizes against ffmpeg on identical PCM and settings.
//! ffmpeg is optional; the FLAC expansion regression also runs in unit tests.

mod common;

use radius_rs::io::{self, Audio};
use std::process::Command;

// Output filename, encoder arguments, reference arguments, ratio and overhead.
type SizeSetting = (
    &'static str,
    &'static [&'static str],
    &'static [&'static str],
    f64,
    u64,
);

#[test]
fn four_format_sizes_stay_in_ffmpegs_range() {
    if !io::ffmpeg_available() {
        eprintln!("SKIP audio size reference: ffmpeg is unavailable");
        return;
    }
    let rate = 44_100;
    let frames = rate as usize * 4;
    let music = common::synthetic_stereo(rate, frames).audio();
    let mut state = 0x5eed_1234u32;
    let noise = Audio {
        samples: (0..frames * 2)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let pcm = ((state >> 8) as i32 - (1 << 23)) >> 1;
                pcm as f32 / 8_388_608.0
            })
            .collect(),
        sample_rate: rate,
        channels: 2,
    };
    let dir = common::tmp_dir().join(format!("audio_sizes_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // WAV/MP3 have almost fixed sizes. FLAC depends on the prediction and
    // Rice coding; Vorbis uses different VBR quality mappings in each encoder.
    let settings: &[SizeSetting] = &[
        ("16.wav", &["-b", "16"], &["-c:a", "pcm_s16le"], 1.0, 256),
        ("24.wav", &["-b", "24"], &["-c:a", "pcm_s24le"], 1.0, 256),
        ("32.wav", &["-b", "32"], &["-c:a", "pcm_s32le"], 1.0, 256),
        ("32f.wav", &["-b", "32f"], &["-c:a", "pcm_f32le"], 1.0, 256),
        (
            "16.flac",
            &["-b", "16"],
            &[
                "-c:a",
                "flac",
                "-sample_fmt",
                "s16",
                "-compression_level",
                "8",
            ],
            1.5,
            8192,
        ),
        (
            "24.flac",
            &["-b", "24"],
            &[
                "-c:a",
                "flac",
                "-sample_fmt",
                "s32",
                "-compression_level",
                "8",
            ],
            1.5,
            8192,
        ),
        (
            "128.ogg",
            &["--ogg-bitrate", "128"],
            &["-c:a", "libvorbis", "-b:a", "128k"],
            2.0,
            8192,
        ),
        (
            "500.ogg",
            &["--ogg-bitrate", "500"],
            &["-c:a", "libvorbis", "-b:a", "500k"],
            2.0,
            8192,
        ),
        (
            "128.mp3",
            &["--mp3-bitrate", "128"],
            &["-c:a", "libmp3lame", "-b:a", "128k"],
            1.0,
            4096,
        ),
        (
            "320.mp3",
            &["--mp3-bitrate", "320"],
            &["-c:a", "libmp3lame", "-b:a", "320k"],
            1.0,
            4096,
        ),
    ];
    for (name, audio) in [("music", music), ("noise", noise)] {
        let input = dir.join(format!("{name}_input.wav"));
        let mut writer = hound::WavWriter::create(
            &input,
            hound::WavSpec {
                channels: 2,
                sample_rate: rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )
        .unwrap();
        for &sample in &audio.samples {
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
        for &(setting, rust_args, ffmpeg_args, ratio_limit, overhead) in settings {
            let rust = dir.join(format!("{name}_rust_{setting}"));
            let reference = dir.join(format!("{name}_ffmpeg_{setting}"));
            let run = Command::new(env!("CARGO_BIN_EXE_radius"))
                .arg(&input)
                .arg(&rust)
                .arg("--no-progress")
                .args(rust_args)
                .env("RADIUS_FFMPEG_IO", "0")
                .output()
                .unwrap();
            assert!(
                run.status.success(),
                "{name} {setting}: {}",
                String::from_utf8_lossy(&run.stderr)
            );
            let run = Command::new(io::ffmpeg_program())
                .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i"])
                .arg(&input)
                .args(["-map", "0:a:0"])
                .args(ffmpeg_args)
                .arg(&reference)
                .output()
                .unwrap();
            assert!(
                run.status.success(),
                "ffmpeg {name} {setting}: {}",
                String::from_utf8_lossy(&run.stderr)
            );
            let rust_size = std::fs::metadata(&rust).unwrap().len();
            let reference_size = std::fs::metadata(&reference).unwrap().len();
            println!(
                "{name} {setting}: Rust {rust_size}, ffmpeg {reference_size}, ratio {:.3}",
                rust_size as f64 / reference_size as f64
            );
            assert!(
                rust_size as f64 <= reference_size as f64 * ratio_limit + overhead as f64,
                "{name} {setting}: Rust {rust_size} vs ffmpeg {reference_size} (limit {ratio_limit}x + {overhead})"
            );
            // Headers alone must never satisfy the size check.
            let decoded = io::read(&rust).unwrap();
            assert_eq!(decoded.channels, audio.channels);
            assert_eq!(decoded.sample_rate, audio.sample_rate);
            if !setting.ends_with("mp3") {
                assert_eq!(decoded.frames(), audio.frames());
            } else {
                assert!(decoded.frames().abs_diff(audio.frames()) < 4608);
            }
            std::fs::remove_file(rust).unwrap();
            std::fs::remove_file(reference).unwrap();
        }
        std::fs::remove_file(input).unwrap();
    }
    std::fs::remove_dir(dir).unwrap();
}
