//! CLI end-to-end tests.
//!
//! These drive the release binary the way a user does: build an input, render it, and
//! read the result back. Every container the CLI writes is decoded again through the
//! CLI's own reader, so the check uses ffmpeg exactly the way the library does.

mod common;

use common::*;
use std::process::Command;

fn bin() -> std::path::PathBuf {
    // `CARGO_BIN_EXE_<name>` is set for integration tests of a binary target.
    std::path::PathBuf::from(env!("CARGO_BIN_EXE_radius"))
}

fn run(args: &[&str]) -> (bool, String, String) {
    let out = Command::new(bin())
        .args(args)
        .output()
        .expect("failed to spawn radius");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// Like [`run`], with `RADIUS_PROGRESS` forced so the progress bar is emitted
/// even though the test pipes stderr (where it is off by default).
fn run_with_progress(args: &[&str]) -> (bool, String, String) {
    let out = Command::new(bin())
        .args(args)
        .env("RADIUS_PROGRESS", "1")
        .output()
        .expect("failed to spawn radius");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// Are the codec cases runnable? With ffmpeg doing the encoding there is nothing left
/// to feature-gate, only a run time capability to check; without it those cases skip.
fn codec_available(_what: &str) -> bool {
    ffmpeg_or_skip("the container round trips")
}

/// Scratch path for this test.
///
/// Unique per *call*, because cargo runs tests in parallel and writing now goes through
/// an ffmpeg subprocess, so two tests sharing a fixed name can genuinely overlap. Each
/// test therefore keeps its own paths in local variables, as below.
fn tmp(name: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join("radius_rs_cli_tests");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(format!(
        "{:03}_{name}",
        N.fetch_add(1, Ordering::Relaxed)
    ))
}

/// A slice of the synthetic programme with a chosen peak.
fn slice_with_peak(name: &str, target_peak: f32) -> std::path::PathBuf {
    let path = tmp(name);
    let src = synthetic_stereo(48_000, 48_000 * 3);
    let mut samples = src.samples.clone();
    let gain = target_peak / peak(&samples).max(1e-9);
    for v in samples.iter_mut() {
        *v *= gain;
    }
    write_wav(&path, &samples, 48_000, 2);
    path
}

/// The input every always-on CLI test drives: 8 s of the shared synthetic programme
/// from `common::synthetic_stereo`. No external audio is needed.
fn input_wav() -> std::path::PathBuf {
    synthetic_file("cli_input", 48_000, 48_000 * 8)
}

/// A ~3 s synthetic slice, peak above full scale, one file per call.
fn hot_slice() -> std::path::PathBuf {
    slice_with_peak("hot_slice.wav", 1.3)
}

/// The same slice at -6 dBFS, which must never trigger a clipping message.
fn quiet_slice() -> std::path::PathBuf {
    slice_with_peak("quiet_slice.wav", 0.5)
}

/// The 1 s slice the container/depth tests work on, cut from `input_wav()`.
fn one_second_slice() -> std::path::PathBuf {
    let src = read_audio(&input_wav());
    let n = src.rate as usize;
    let path = tmp("slice.wav");
    write_wav(&path, &src.samples[..n * src.channels], src.rate, src.channels);
    path
}

/// The progress bar must appear on stderr when forced, and must be suppressible.
#[test]
fn cli_progress_bar_is_reported_and_suppressible() {
    let hot = hot_slice();
    let out = tmp("progress.wav");
    let _ = std::fs::remove_file(&out);

    let (ok, stdout, stderr) = run_with_progress(&[
        hot.to_str().unwrap(),
        out.to_str().unwrap(),
        "-m",
        "td",
        "-s",
        "3",
    ]);
    assert!(ok, "render failed: {stderr}");
    assert!(
        stderr.contains("td  [") && stderr.contains('%') && stderr.contains("elapsed"),
        "progress bar missing from stderr:\n{stderr}"
    );
    // the bar is a status line, not part of the machine-readable result
    assert!(
        !stdout.contains("elapsed,"),
        "progress leaked onto stdout:\n{stdout}"
    );

    let out2 = tmp("progress_off.wav");
    let (ok, _, stderr) = run_with_progress(&[
        hot.to_str().unwrap(),
        out2.to_str().unwrap(),
        "-m",
        "td",
        "-s",
        "0",
        "--no-progress",
    ]);
    assert!(ok, "--no-progress run failed: {stderr}");
    assert!(
        !stderr.contains("elapsed,"),
        "--no-progress still drew a bar:\n{stderr}"
    );

    // Without a terminal and without the forcing variable, the bar stays off so
    // logs and pipelines are unaffected.
    let out3 = tmp("progress_auto.wav");
    let (ok, _, stderr) = run(&[
        hot.to_str().unwrap(),
        out3.to_str().unwrap(),
        "-m",
        "td",
        "-s",
        "0",
    ]);
    assert!(ok, "auto run failed: {stderr}");
    assert!(
        !stderr.contains("elapsed,"),
        "the bar appeared on a non-terminal stderr by default:\n{stderr}"
    );
}

/// A render that exceeds full scale must warn for every target that cannot store
/// it, and stay quiet for 32-bit float WAV, which can.
#[test]
fn cli_warns_about_clipping_only_for_formats_that_clamp() {
    let hot = hot_slice();
    // `-s 3` on this material overshoots full scale by ~26 %; `-s 0` happens not
    // to, so the shift is part of the fixture, not incidental.
    let args_base = [hot.to_str().unwrap(), "", "-m", "td", "-s", "3"];

    // 32-bit float wav keeps the overshoot exactly: a note, not a warning.
    let f32_out = tmp("clip_f32.wav");
    let mut a: Vec<&str> = args_base.to_vec();
    a[1] = f32_out.to_str().unwrap();
    a.extend(["-b", "32f"]);
    let (ok, stdout, stderr) = run(&a);
    assert!(ok, "32f render failed: {stderr}");
    assert!(
        !stderr.contains("WARNING"),
        "32-bit float wav must not warn:\n{stderr}"
    );
    assert!(
        stdout.contains("note: peak") && stdout.contains("kept exactly"),
        "expected a note about the overshoot:\n{stdout}"
    );
    // and the overshoot really is in the file
    let kept = read_wav(&f32_out);
    assert!(
        peak(&kept.samples) > 1.0,
        "32-bit float output should keep the overshoot, peak={}",
        peak(&kept.samples)
    );

    // Every clamping target must warn, and name itself.
    for (name, extra, expect_format) in [
        ("clip_i16.wav", vec!["-b", "16"], "16-bit PCM wav"),
        ("clip_i24.wav", vec!["-b", "24"], "24-bit PCM wav"),
        ("clip.flac", vec![], "flac"),
        ("clip.ogg", vec![], "ogg/vorbis"),
        ("clip.mp3", vec![], "mp3"),
    ] {
        if !codec_available(name) {
            continue;
        }
        let out = tmp(name);
        let _ = std::fs::remove_file(&out);
        let mut a: Vec<&str> = args_base.to_vec();
        a[1] = out.to_str().unwrap();
        a.extend(extra.iter().copied());
        let (ok, stdout, stderr) = run(&a);
        assert!(ok, "{name} render failed: {stderr}\n{stdout}");
        assert!(
            stderr.contains("WARNING: the render peaks at"),
            "{name}: no clipping warning:\n{stderr}"
        );
        assert!(
            stderr.contains(expect_format),
            "{name}: warning does not name the format:\n{stderr}"
        );
        assert!(
            stderr.contains("32-bit float WAV"),
            "{name}: warning does not point at the lossless alternative:\n{stderr}"
        );
    }

    // Material below full scale must not trigger any of this.
    let quiet = quiet_slice();
    let quiet_out = tmp("clip_quiet.wav");
    let (ok, _stdout, stderr) = run(&[
        quiet.to_str().unwrap(),
        quiet_out.to_str().unwrap(),
        "-m",
        "td",
        "-s",
        "0",
        "-b",
        "16",
    ]);
    assert!(ok, "quiet render failed: {stderr}");
    assert!(
        !stderr.contains("WARNING") && !stderr.contains("note: peak"),
        "a -6 dBFS render must not mention clipping:\n{stderr}"
    );
}

/// Decode any format the CLI reads by converting it to WAV with the CLI itself
/// (`-s 0` is a no-op render, so the WAV holds the decoded input), then reading
/// that WAV through [`radius_rs::io`]. This keeps the check on the shipped reader
/// rather than duplicating a decoder in the test.
fn decode_via_cli(path: &std::path::Path, tag: &str) -> Wav {
    let wav = tmp(&format!("decoded_{tag}.wav"));
    let _ = std::fs::remove_file(&wav);
    let (ok, stdout, err) = run(&[
        path.to_str().unwrap(),
        wav.to_str().unwrap(),
        "-m",
        "td",
        "-s",
        "0",
    ]);
    assert!(ok, "decoding {} failed: {err}\n{stdout}", path.display());
    read_wav(&wav)
}

#[test]
fn cli_version_and_help() {
    let (ok, out, _) = run(&["--version"]);
    assert!(ok);
    assert!(out.contains(env!("CARGO_PKG_VERSION")));
    let (ok, out, _) = run(&["--help"]);
    assert!(ok);
    assert!(out.contains("--mode"));
    assert!(out.contains("--semitones"));
}

#[test]
fn cli_missing_args_fails() {
    let (ok, _, err) = run(&[]);
    assert!(!ok);
    assert!(err.contains("Usage") || err.contains("usage"));
}

/// Running with nothing but input and output must produce the documented
/// defaults: time-domain engine, **no pitch change** (`-s 0` — a bare invocation is
/// a format conversion, a shift has to be asked for), no time stretch, the reference
/// drivers' quality values, and **32-bit float WAV** out.
#[test]
fn cli_bare_invocation_uses_the_documented_defaults() {
    let out = tmp("bare_defaults.wav");
    let _ = std::fs::remove_file(&out);
    let (ok, stdout, err) = run(&[input_wav().to_str().unwrap(), out.to_str().unwrap()]);
    assert!(ok, "bare invocation failed: {err}");
    assert!(
        stdout.contains("mode=Td semitones=+0.00 tempo=100.0% quality=37 solo=0"),
        "unexpected defaults banner: {stdout}"
    );
    assert!(
        stdout.contains("pass-through"),
        "a bare invocation should be a pass-through: {stdout}"
    );
    assert!(stdout.contains("wav 32-bit float"), "banner: {stdout}");

    // And the pass-through must be exact: that is the whole point of it. Comparing
    // through ffmpeg is the honest check now — the samples make a full round trip
    // through ffmpeg's own f32 WAV encoder and decoder, so this asserts something
    // stronger than "the bytes I wrote are the bytes I read".
    let src = read_audio(&input_wav());
    let got = read_audio(&out);
    assert_eq!(got.frames(), src.frames(), "pass-through changed the length");
    assert_eq!(
        got.samples, src.samples,
        "a 32-bit float WAV pass-through must survive ffmpeg unchanged"
    );

    let info = ffmpeg_info(&out);
    assert_eq!(
        info.codec, "pcm_f32le",
        "the default output container must be 32-bit float WAV, got {}",
        info.line
    );
    assert_eq!(info.sample_rate, 48000, "sample rate must pass through");
    assert_eq!(info.channels, 2, "channel count must pass through");
}

/// A pass-through to FLAC keeps every sample (within 24-bit quantisation) and adds
/// only trailing zeros.
///
/// This pins the documented limitation of the FLAC writer (fixed blocksize, so the
/// signal is zero-padded to a block boundary) so it cannot silently get worse: the
/// whole source must be there, and everything after it must be zero.
#[test]
fn cli_passthrough_to_flac_keeps_every_sample() {
    let flac = tmp("passthrough.flac");
    let _ = std::fs::remove_file(&flac);
    let (ok, stdout, err) = run(&[input_wav().to_str().unwrap(), flac.to_str().unwrap()]);
    assert!(ok, "flac pass-through failed: {err}");
    assert!(stdout.contains("pass-through"), "{stdout}");

    // Read it back through the CLI's own decoder.
    let back = tmp("passthrough_back.wav");
    let _ = std::fs::remove_file(&back);
    let (ok, _, err) = run(&[flac.to_str().unwrap(), back.to_str().unwrap(), "-b", "32f"]);
    assert!(ok, "flac re-decode failed: {err}");
    let got = read_wav(&back);
    let src = read_wav(&input_wav());
    assert!(
        got.frames() >= src.frames(),
        "flac round trip lost frames: {} vs {}",
        got.frames(),
        src.frames()
    );
    let n = src.samples.len();
    // Compare without `assert_eq!` on the slices: a failing assert would try to
    // format a megabyte of samples into the panic message.
    assert!(
        got.samples.len() >= n,
        "flac round trip returned {} samples, expected at least {n}",
        got.samples.len()
    );
    // FLAC here is 24-bit integer PCM, so the samples are the source quantised to
    // 24 bit: at most half a step (2^-24) away, and never different in sign.
    let step = 1.0f32 / (1u32 << 23) as f32;
    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    for i in 0..n {
        let d = (got.samples[i] - src.samples[i]).abs();
        if d > worst {
            worst = d;
            worst_at = i;
        }
    }
    assert!(
        worst <= step,
        "flac round trip changed sample {worst_at} by {worst:e} (24-bit step is {step:e})"
    );
    assert!(
        got.samples[n..].iter().all(|v| *v == 0.0),
        "everything past the source must be the zero padding"
    );
}

/// The lossy-codec defaults must be the top of each range.
#[test]
fn cli_lossy_defaults_are_the_best_available() {
    for (ext, expected) in [("ogg", "q=0.90"), ("mp3", "320 kbps")] {
        if !codec_available(ext) {
            continue;
        }
        let out = tmp(&format!("default_quality.{ext}"));
        let _ = std::fs::remove_file(&out);
        let (ok, stdout, err) = run(&[
            input_wav().to_str().unwrap(),
            out.to_str().unwrap(),
            "-s",
            "0",
        ]);
        assert!(ok, "default .{ext} write failed: {err}");
        assert!(
            stdout.contains(expected),
            "default .{ext} should be {expected}, banner: {stdout}"
        );
    }
}

#[test]
fn cli_td_wav_round_trip() {
    let out = tmp("td_out.wav");
    let out_s = out.to_str().unwrap();
    let (ok, stdout, err) = run(&[input_wav().to_str().unwrap(), out_s, "-m", "td", "-s", "3"]);
    assert!(ok, "cli failed: {err}");
    assert!(stdout.contains("render:"));
    let y = read_wav(&out);
    assert_eq!(y.rate, 48000);
    assert_eq!(y.channels, 2);
    let x = read_wav(&input_wav());
    let ratio = y.frames() as f64 / x.frames() as f64;
    assert!((ratio - 1.0).abs() < 0.005, "length ratio {ratio}");
    assert!(peak(&y.samples) > 0.05);
}

#[test]
fn cli_td_truth_report() {
    let out = tmp("td_truth_out.wav");
    let (ok, stdout, err) = run(&[
        input_wav().to_str().unwrap(),
        out.to_str().unwrap(),
        "-m",
        "td",
        "-s",
        "3",
        "--truth",
        input_wav().to_str().unwrap(),
    ]);
    assert!(ok, "cli failed: {err}");
    assert!(stdout.contains("CORR vs gold"), "stdout: {stdout}");
}

#[test]
fn cli_rejects_unsupported_vocoder_rate() {
    // build a 22.05 kHz wav on the fly from the synthetic input
    let src = read_wav(&input_wav());
    let take = src.rate as usize; // 1 second
    let mut small = Vec::with_capacity(take * src.channels);
    for i in 0..take {
        for c in 0..src.channels {
            small.push(src.samples[i * src.channels + c]);
        }
    }
    let path = tmp("rate22050.wav");
    write_wav(&path, &small, 22050, src.channels);

    let out = tmp("bad_rate_vc.wav");
    let (ok, _, err) = run(&[
        path.to_str().unwrap(),
        out.to_str().unwrap(),
        "-m",
        "vc",
        "-s",
        "3",
    ]);
    assert!(!ok, "vocoder accepted 22050 Hz");
    assert!(err.contains("unsupported sr"), "stderr: {err}");
}

/// Every writable container must survive a write/read round trip.
#[test]
fn cli_writes_and_reads_back_each_container() {
    // a short slice keeps FLAC/Ogg/MP3 encoding fast
    let sliced = one_second_slice();
    let src = read_wav(&sliced);
    let n = src.frames();

    // A no-op render (`-s 0`) isolates the *codec* from the pitch shifter:
    // the engine is duration- and amplitude-preserving at ratio 1, so a high
    // correlation against the source measures the container round trip.
    for ext in ["wav", "flac", "ogg", "mp3"] {
        if !codec_available(ext) {
            continue;
        }
        let out = tmp(&format!("container_out.{ext}"));
        let _ = std::fs::remove_file(&out);
        let (ok, stdout, err) = run(&[
            sliced.to_str().unwrap(),
            out.to_str().unwrap(),
            "-m",
            "td",
            "-s",
            "0",
        ]);
        assert!(ok, ".{ext} write failed: {err}\n{stdout}");
        assert!(out.exists(), ".{ext} was not created");
        // decode it back through the CLI's own reader (symphonia)
        let back = decode_via_cli(&out, ext);
        assert_eq!(back.rate, 48000, ".{ext} sample rate");
        assert_eq!(back.channels, 2, ".{ext} channels");
        assert!(back.frames() > 0, ".{ext} empty");
        assert!(peak(&back.samples) > 0.01, ".{ext} silent");
        // Codecs may pad/trim the head and tail (LAME's encoder delay alone is
        // 576 samples), so correlate at the best whole-frame alignment within a
        // generous window instead of assuming index 0.
        let src = &src.samples[..n * 2];
        let mut best = -2.0f64;
        let mut best_shift = 0i64;
        for shift in -6144i64..=6144 {
            let (a, b) = if shift >= 0 {
                (&back.samples[..], &src[shift as usize * 2..])
            } else {
                (&back.samples[(-shift) as usize * 2..], &src[..])
            };
            let len = a.len().min(b.len());
            if len < 4096 {
                continue;
            }
            let r = corr(&a[..len], &b[..len]);
            if r > best {
                best = r;
                best_shift = shift;
            }
        }
        assert!(
            best > 0.9,
            ".{ext} round-trip correlation {best} (shift {best_shift})"
        );
        println!(
            ".{ext}: samples={} best corr={best:.6} at shift {best_shift}",
            back.samples.len()
        );
    }
}

/// Every sample-depth / quality knob must be accepted and must actually change
/// the output (bit depth changes the file size and precision; bitrate/quality
/// change the encoded size).
#[test]
fn cli_quality_options_are_honoured() {
    let sliced = one_second_slice();
    let src = read_wav(&sliced);
    let n = src.frames();

    // ---- WAV sample depths: 16 < 24 < 32f in bytes -------------------------
    let mut wav_sizes = Vec::new();
    for depth in ["16", "24", "32f"] {
        let out = tmp(&format!("depth_{depth}.wav"));
        let _ = std::fs::remove_file(&out);
        let (ok, _, err) = run(&[
            sliced.to_str().unwrap(),
            out.to_str().unwrap(),
            "-m",
            "td",
            "-s",
            "0",
            "-b",
            depth,
        ]);
        assert!(ok, "wav -b {depth} failed: {err}");
        // ffmpeg's own name for what it wrote: s16 / s24 / float32.
        let info = ffmpeg_info(&out);
        let want = match depth {
            "16" => "pcm_s16le",
            "24" => "pcm_s24le",
            _ => "pcm_f32le",
        };
        assert_eq!(info.codec, want, "-b {depth} wrote {}", info.line);
        wav_sizes.push((depth, std::fs::metadata(&out).unwrap().len()));
    }
    let s16 = wav_sizes.iter().find(|(d, _)| *d == "16").unwrap().1;
    let s24 = wav_sizes.iter().find(|(d, _)| *d == "24").unwrap().1;
    let s32 = wav_sizes.iter().find(|(d, _)| *d == "32f").unwrap().1;
    assert!(
        s16 < s24 && s24 < s32,
        "wav sizes not monotone: {s16} {s24} {s32}"
    );

    // ---- FLAC depths: lossless, and 16 bit must be the smaller file --------
    //
    // No size formula here any more: ffmpeg's FLAC encoder actually predicts and
    // Rice-codes the subframes, so the size depends on the signal. What must hold is
    // that the stream is lossless at the requested depth and that the shallower depth
    // is not the larger file.
    let mut flac_sizes = Vec::new();
    for depth in ["16", "24"] {
        let out = tmp(&format!("depth_{depth}.flac"));
        let _ = std::fs::remove_file(&out);
        let (ok, _, err) = run(&[
            sliced.to_str().unwrap(),
            out.to_str().unwrap(),
            "-m",
            "td",
            "-s",
            "0",
            "-b",
            depth,
        ]);
        assert!(ok, "flac -b {depth} failed: {err}");
        let size = std::fs::metadata(&out).unwrap().len();
        let back = decode_via_cli(&out, &format!("flac{depth}"));
        assert_eq!(back.channels, 2);
        assert!(back.frames() >= n, "flac -b {depth} lost frames");
        assert!(peak(&back.samples) > 0.01);
        // Lossless: the decode must match the input within that depth's quantisation.
        let step = 1.0f32 / (1u64 << (depth.parse::<u32>().unwrap() - 1)) as f32;
        let worst = src.samples[..n * src.channels]
            .iter()
            .zip(back.samples.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst <= step,
            "flac -b {depth} is not lossless: worst error {worst:e} > step {step:e}"
        );
        println!("flac -b {depth}: {size} bytes, worst error {worst:e}");
        flac_sizes.push((depth, size));
    }
    let f16 = flac_sizes.iter().find(|(d, _)| *d == "16").unwrap().1;
    let f24 = flac_sizes.iter().find(|(d, _)| *d == "24").unwrap().1;
    assert!(f16 < f24, "flac 16-bit ({f16}) should be smaller than 24-bit ({f24})");

    // ---- MP3 bitrate: 320 kbps must be bigger than 128 --------------------
    if codec_available("mp3") {
        let mut sizes = Vec::new();
        for kbps in ["128", "320"] {
            let out = tmp(&format!("br_{kbps}.mp3"));
            let _ = std::fs::remove_file(&out);
            let (ok, _, err) = run(&[
                sliced.to_str().unwrap(),
                out.to_str().unwrap(),
                "-m",
                "td",
                "-s",
                "0",
                "--mp3-bitrate",
                kbps,
            ]);
            assert!(ok, "mp3 {kbps} failed: {err}");
            sizes.push((kbps, std::fs::metadata(&out).unwrap().len()));
        }
        assert!(
            sizes[0].1 < sizes[1].1,
            "mp3 320 kbps ({}) should exceed 128 kbps ({})",
            sizes[1].1,
            sizes[0].1
        );
    }

    // ---- Ogg quality -------------------------------------------------------
    if codec_available("ogg") {
        let mut sizes = Vec::new();
        for q in ["0.1", "0.9"] {
            let out = tmp(&format!("q_{}.ogg", q.replace('.', "_")));
            let _ = std::fs::remove_file(&out);
            let (ok, _, err) = run(&[
                sliced.to_str().unwrap(),
                out.to_str().unwrap(),
                "-m",
                "td",
                "-s",
                "0",
                "--ogg-quality",
                q,
            ]);
            assert!(ok, "ogg q={q} failed: {err}");
            sizes.push((q, std::fs::metadata(&out).unwrap().len()));
        }
        assert!(
            sizes[0].1 < sizes[1].1,
            "ogg q=0.9 ({}) should exceed q=0.1 ({})",
            sizes[1].1,
            sizes[0].1
        );
    }

    // ---- --format overrides the extension ---------------------------------
    let out = tmp("forced.dat");
    let _ = std::fs::remove_file(&out);
    let (ok, _, err) = run(&[
        sliced.to_str().unwrap(),
        out.to_str().unwrap(),
        "-m",
        "td",
        "-s",
        "0",
        "--format",
        "flac",
    ]);
    assert!(ok, "--format flac failed: {err}");
    let magic = std::fs::read(&out).unwrap();
    assert_eq!(&magic[..4], b"fLaC", "--format did not select FLAC");
    // an unknown extension without --format must be rejected
    let bad = tmp("unknown.xyz");
    let (ok, _, err) = run(&[
        sliced.to_str().unwrap(),
        bad.to_str().unwrap(),
        "-m",
        "td",
        "-s",
        "0",
    ]);
    assert!(!ok, "unknown extension was accepted");
    assert!(
        err.contains("unsupported output extension"),
        "stderr: {err}"
    );
}

/// Run the binary with `dir` as its working directory, so tests of the derived
/// output name (which lands in the CWD) do not litter the source tree.
fn run_in(dir: &std::path::Path, args: &[&str]) -> (bool, String, String) {
    let out = Command::new(bin())
        .current_dir(dir)
        .args(args)
        .output()
        .expect("failed to spawn radius");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// Work dir for the derived-name tests, unique per call.
fn cwd(tag: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let d = std::env::temp_dir().join(format!(
        "radius_rs_cwd_{tag}_{}",
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The output argument is optional. With no output path the CLI derives one from the
/// input name plus a suffix describing the run, places it beside the input, and keeps
/// running when a file of that name already exists instead of clobbering it.
#[test]
fn cli_derives_the_output_name_when_none_is_given() {
    let dir = cwd("derived");
    // Copy the synthetic input in so the derived name is predictable.
    let src = input_wav();
    let local = dir.join("song.wav");
    std::fs::copy(&src, &local).unwrap();

    // mode only: semitones 0 and tempo 100 contribute nothing to the name
    let (ok, stdout, err) = run_in(&dir, &["song.wav", "-m", "td"]);
    assert!(ok, "derived-output run failed: {err}");
    assert!(stdout.contains("song_td.wav"), "stdout: {stdout}");
    assert!(dir.join("song_td.wav").is_file());

    // semitones and tempo appear, mode first, in that order
    let (ok, stdout, err) = run_in(&dir, &["song.wav", "-m", "vc", "-s", "-3", "--tempo", "200"]);
    assert!(ok, "run failed: {err}");
    assert!(stdout.contains("song_vc_st-3_tp200.wav"), "stdout: {stdout}");
    assert!(dir.join("song_vc_st-3_tp200.wav").is_file());

    // a whole-number shift reads `st3`, not `st3.0`
    let (ok, _, err) = run_in(&dir, &["song.wav", "-m", "vc", "-s", "3"]);
    assert!(ok, "run failed: {err}");
    assert!(dir.join("song_vc_st3.wav").is_file());

    // never overwrite: the same command again must step over the existing file
    let (ok, stdout, err) = run_in(&dir, &["song.wav", "-m", "td"]);
    assert!(ok, "second run failed: {err}");
    assert!(stdout.contains("song_td (1).wav"), "stdout: {stdout}");
    let (ok, stdout, _) = run_in(&dir, &["song.wav", "-m", "td"]);
    assert!(ok);
    assert!(stdout.contains("song_td (2).wav"), "stdout: {stdout}");
    // ...and all three still exist
    for n in ["song_td.wav", "song_td (1).wav", "song_td (2).wav"] {
        assert!(dir.join(n).is_file(), "{n} went missing");
    }

    // A derived name goes beside the *input*, not into the working directory: run
    // from `elsewhere` against `sub/song.wav` and the result must join the input.
    let sub = dir.join("sub");
    let elsewhere = dir.join("elsewhere");
    std::fs::create_dir_all(&sub).unwrap();
    std::fs::create_dir_all(&elsewhere).unwrap();
    std::fs::copy(&src, sub.join("song.wav")).unwrap();
    let (ok, stdout, err) = run_in(&elsewhere, &["../sub/song.wav", "-m", "td", "-s", "4"]);
    assert!(ok, "run from another directory failed: {err}");
    assert!(stdout.contains("song_td_st4.wav"), "stdout: {stdout}");
    assert!(
        sub.join("song_td_st4.wav").is_file(),
        "the output should sit beside the input, not in the working directory"
    );
    assert!(
        !elsewhere.join("song_td_st4.wav").exists(),
        "the output must not be created in the working directory"
    );

    // An explicit path is still taken relative to the working directory.
    let (ok, _, err) = run_in(&elsewhere, &["../sub/song.wav", "picked.wav", "-m", "td"]);
    assert!(ok, "explicit-output run failed: {err}");
    assert!(elsewhere.join("picked.wav").is_file());

    let _ = std::fs::remove_dir_all(&dir);
}

/// With no `--format` and no output extension, the output format follows the *input*
/// when the input's format is one this crate can write, and falls back to WAV when it
/// is not. `--format` still wins over both.
#[test]
fn cli_output_format_follows_the_input() {
    if !codec_available("the input-format-following cases") {
        return;
    }
    let dir = cwd("follows");
    let src = input_wav();
    // A lossless FLAC input: the output should be FLAC, not WAV.
    let flac = dir.join("song.flac");
    let a = read_audio(&src);
    radius_rs::io::write(
        &flac,
        &radius_rs::io::Audio {
            samples: a.samples.clone(),
            sample_rate: a.rate,
            channels: a.channels,
        },
        radius_rs::io::WriteOptions {
            container: Some(radius_rs::io::Container::Flac),
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| panic!("{e}"));

    let (ok, stdout, err) = run_in(&dir, &["song.flac", "-m", "td"]);
    assert!(ok, "flac-input run failed: {err}");
    assert!(stdout.contains("song_td.flac"), "stdout: {stdout}");
    let out = dir.join("song_td.flac");
    assert!(out.is_file(), "no .flac output was written");
    assert_eq!(
        ffmpeg_info(&out).codec,
        "flac",
        "the output should have followed the input's format"
    );

    // A format this crate cannot write falls back to WAV. `m4a`/AAC is exactly such
    // an input: ffmpeg decodes it, but the CLI has no encoder for it, so the fixture
    // has to be built with ffmpeg directly rather than through the CLI.
    let m4a = dir.join("song.m4a");
    let st = Command::new(radius_rs::io::ffmpeg_program())
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(&flac)
        .args(["-c:a", "aac"])
        .arg(&m4a)
        .status()
        .expect("failed to spawn ffmpeg");
    assert!(st.success(), "could not build the m4a fixture");
    let (ok, stdout, err) = run_in(&dir, &["song.m4a", "-m", "td"]);
    assert!(ok, "unsupported-format-input run failed: {err}");
    assert!(
        stdout.contains("song_td.wav"),
        "an unwritable input format must fall back to wav: {stdout}"
    );
    assert_eq!(ffmpeg_info(&dir.join("song_td.wav")).codec, "pcm_f32le");

    // --format outranks the input's format.
    let (ok, stdout, err) = run_in(&dir, &["song.flac", "-m", "td", "-s", "2", "--format", "mp3"]);
    assert!(ok, "override run failed: {err}");
    assert!(stdout.contains("song_td_st2.mp3"), "stdout: {stdout}");
    assert_eq!(ffmpeg_info(&dir.join("song_td_st2.mp3")).codec, "mp3");
    let _ = std::fs::remove_dir_all(&dir);
}

