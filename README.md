# radius-rs

Rust rewrite of the **Radius** time-domain and phase-vocoder pitch-shift algorithms, ported from a [pyradius](https://github.com/baicai-1145/pyradius) Python/NumPy reference implementation, which is itself a line-by-line port of the [libradius](https://github.com/baicai-1145/libradius) C engine.

This repository contains three entry points, all using the same underlying engines:

* **crate**: `radius_rs` — the engines plus the operator modules, usable on an in-memory `f32` buffer
* **CLI**: `radius` binary — pitch shift and time stretch for any container or codec ffmpeg knows
* **C ABI**: `cdylib` / `staticlib` with header, usable from C, C++, Python, etc.

Radius supports two pitch-shifting engines, selectable at run time:

* **Phase vocoder** (`-m vc`, the default) — the reference's vocoder mode, much slower, better for dense polyphonic music.
* **Time domain** (`-m td`) — granule-based, fast, best for monophonic material.

## Requirements

* **Rust** 1.75 or newer, edition 2021.
* **ffmpeg** at run time, for any audio I/O. The library itself has no audio codec in it: reading and writing go through ffmpeg, so every format it supports works in both directions.
* A CPU with **AVX and FMA** on x86-64, i.e. Intel Haswell (2013) or AMD Excavator (2015) and newer. [`.cargo/config.toml`](.cargo/config.toml) turns the feature on because the engine mirrors the reference's `a*b + c` as one correctly-rounded operation, and without it every fused multiply-add becomes a software call — worth about 2x on the vocoder. It changes no output bits. See [docs/DESIGN.md](docs/DESIGN.md); build with `RUSTFLAGS=-C target-feature=-fma` to drop the requirement.

ffmpeg is looked up as `ffmpeg` on `PATH`, or wherever `RADIUS_FFMPEG` points:

```bash
export RADIUS_FFMPEG=/opt/ffmpeg/bin/ffmpeg                 # Linux / macOS
set RADIUS_FFMPEG=C:\Program Files\ffmpeg\bin\ffmpeg.exe    # Windows
```

## Quick start

```bash
# library, cdylib, staticlib and the radius binary
cargo build --release

# +3 semitones with the time-domain engine
radius in.wav out.wav -m td -s 3

# the output name is optional: this writes in_vc_st-3_tp200.flac next to the
# input, following the input's format, and refuses to overwrite anything
radius in.flac -m vc -s -3 --tempo 200

# -3 semitones with the phase vocoder, written as 24-bit FLAC
radius in.flac out.flac -m vc -s -3 -b 24

# format conversion only (the default is no pitch change)
radius in.wav out.mp3 --mp3-bitrate 320

# faster FFT backend (not bit-exact; see docs/FFT.md)
radius in.wav out.wav -m vc -s 3 --fft rustfft

# engine, CLI and C ABI tests; needs no external audio
cargo test --release
```

## Profiling and benchmarks

`RADIUS_PROFILE=1` prints a per-stage breakdown of the vocoder, which is the first thing to reach for before optimising anything. Add `RADIUS_THREADS=1` alongside it — the stage timers are thread-local, so worker time would otherwise be missing from the report:

```bash
RADIUS_PROFILE=1 RADIUS_THREADS=1 radius in.wav out.wav -m vc -s 3
```

Note that the stages cover the per-granule chain only, and the crossover is reported separately for exactly that reason: it runs in `feed`, between granules, and that placement once hid the largest cost in the engine. The report includes an `unaccounted` figure so the next such blind spot shows up instead of being absorbed silently. [docs/VERIFICATION.md](docs/VERIFICATION.md) has the details and current numbers.

## Learn more

Everything detailed lives in [`docs/`](docs/):

| document | contents |
|---|---|
| [docs/CLI.md](docs/CLI.md) | every option, its default and why, with the equivalent control in Adobe Audition |
| [docs/RUST-API.md](docs/RUST-API.md) | using the crate from Rust, with examples |
| [docs/C-ABI.md](docs/C-ABI.md) | the C header, linking the shared or static library, worked C and Python examples |
| [docs/VERIFICATION.md](docs/VERIFICATION.md) | accuracy against the reference, the Audition comparison, performance, what the test suite covers |
| [docs/DESIGN.md](docs/DESIGN.md) | self-containment, float discipline, and the deliberate behaviour differences |
| [docs/FFT.md](docs/FFT.md) | the two FFT backends and the measured tradeoff between them |
| [docs/TABLE_PROVENANCE.md](docs/TABLE_PROVENANCE.md) | where every compiled-in constant table comes from |
