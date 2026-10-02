# radius-rs

Rust rewrite of the **Radius** time-domain and phase-vocoder pitch-shift algorithms, ported from a [pyradius](https://github.com/baicai-1145/pyradius) Python/NumPy reference implementation, which is itself a line-by-line port of the [libradius](https://github.com/baicai-1145/libradius) C engine.

This repository contains three entry points, all using the same underlying engines:

* **crate**: `radius_rs` — the engines plus the operator modules, usable on an in-memory `f32` buffer
* **CLI**: `radius` binary — pitch shift and time stretch for any container or codec ffmpeg knows
* **C ABI**: `cdylib` / `staticlib` with header, usable from C, C++, Python, etc.

Radius supports two pitch-shifting engines, selectable at run time:

* **Time domain** (`-m td`, the default) — granule-based, fast, best for monophonic material.
* **Phase vocoder** (`-m vc`) — the reference's vocoder mode, much slower, better for dense polyphonic music.

## Requirements

* **Rust** 1.75 or newer, edition 2021.
* **ffmpeg** at run time, for any audio I/O. The library itself has no audio codec in it: reading and writing go through ffmpeg, so every format it supports works in both directions.

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

# -3 semitones with the phase vocoder, written as 24-bit FLAC
radius in.flac out.flac -m vc -s -3 -b 24

# format conversion only (the default is no pitch change)
radius in.wav out.mp3 --mp3-bitrate 320

# engine, CLI and C ABI tests; needs no external audio
cargo test --release
```

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
