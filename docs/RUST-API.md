# Using it as a Rust crate

The library is plain Rust: hand an engine an interleaved `f32` buffer and it returns one. Nothing here needs a file, a cargo feature or a system dependency.

## In memory

```rust
use radius_rs::TdState;

// Interleaved f32, `nch` wide. Any sample rate works for the TD engine.
let rate = 48_000u32;
let nch = 2usize;
let frames = rate as usize; // 1 s
let mut input = vec![0.0f32; frames * nch];
for i in 0..frames {
    let t = i as f32 / rate as f32;
    let v = 0.4 * (std::f32::consts::TAU * 440.0 * t).sin();
    input[i * nch] = v;
    input[i * nch + 1] = v;
}

// quality 37 / solo 0 are the reference defaults; nch is the channel count.
let mut engine = TdState::new(rate, 37, 0, nch);
engine.set_ratio(3.0, 100.0); // +3 semitones, no time stretch

let output = engine.render(&input, frames); // interleaved f32, duration preserving

println!(
    "{} -> {} frames, {} granules, peak {:.4}",
    frames,
    output.len() / nch,
    engine.n_granule,
    output.iter().fold(0.0f32, |a, b| a.max(b.abs())),
);
```

Swap in the vocoder for polyphonic material. It takes the whole buffer and no frame count, because it always returns exactly what it was given:

```rust
use radius_rs::VocoderState;

let mut engine = VocoderState::new(48_000, 2, 2); // rate, channels, precision
engine.set_ratio(-3.0, 100.0);
let output = engine.render(&input);
```

## Reading and writing files

The same crate does the format work, through ffmpeg, so a wav to flac conversion is three calls. This is exactly what the `radius` binary does:

```rust
use radius_rs::io::{self as audio, Audio, WriteOptions};
use radius_rs::TdState;

let input: Audio = audio::read("in.wav")?;   // anything ffmpeg reads
let mut engine = TdState::new(input.sample_rate, 37, 0, input.channels);
engine.set_ratio(3.0, 100.0);
let out = engine.render(&input.samples, input.frames());

audio::write(
    "out.flac",
    &Audio { samples: out, sample_rate: input.sample_rate, channels: input.channels },
    WriteOptions::default(),                 // 32f for wav/flac, ogg q=0.9, mp3 320k
)?;
```

`WriteOptions` carries the same knobs as the CLI: `depth`, `container`, `ogg_quality` and `mp3_kbps`. [`radius_rs::io::info`] asks ffmpeg what is actually in a file, [`radius_rs::io::ffmpeg_available`] tells you whether any of this will work at run time, and `radius_rs::cli::clip_warning` is the clipping check the binary prints, if you want the same warning in your own tool.

## Feature flags

There are no optional Cargo features. Both FFT backends are always compiled and selected at run time with [`radius_rs::fft::set_backend`] or the `--fft` flag, and codec support comes from the external ffmpeg binary rather than from a bundled library.

## Two rules worth knowing

Both are inherited from the reference engines rather than chosen here:

* **The vocoder accepts 44100 and 48000 Hz only.** Check with `radius_rs::vocoder::supported_rate(rate)`. The TD engine has no such restriction.
* **The output length can differ from the input by up to a granule** at the tail, and a render can exceed full scale. The engines are duration- and roughly amplitude-preserving, not normalising. See the clipping section of [CLI.md](CLI.md).

## Errors

Every fallible entry point returns a `String` error rather than panicking, and the I/O ones name the program and the environment variable to set when ffmpeg is missing. Use `.map_err(anyhow::Error::msg)?` to lift them into `anyhow`, as the CLI does.

## See also

* [CLI.md](CLI.md) for the same knobs on the command line.
* [C-ABI.md](C-ABI.md) for calling the library from C or Python.
* [DESIGN.md](DESIGN.md) for the deliberate behaviour differences from the reference engine.
