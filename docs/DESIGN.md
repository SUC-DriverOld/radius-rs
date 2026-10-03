# Design notes

Why the crate is put together the way it is, in particular the properties that are deliberate rather than incidental.

## Self-contained binaries

`radius.exe` and `radius_rs.dll` carry their own constant tables, compiled in as Rust source under `src/vocoder/table_data/`. There is no data directory, no asset file and no build script, so you can copy either one anywhere and it still runs.

What they do need at run time is **ffmpeg**, for any audio I/O. Without it the CLI exits with an error naming the program and the variable to set; the engines themselves are unaffected, which is what the C ABI and engine tests exercise.

```
$ copy radius.exe C:\somewhere\else\        # no data files needed
set RADIUS_FFMPEG=C:\Program Files\ffmpeg\bin\ffmpeg.exe
$ C:\somewhere\else\radius.exe in.wav out.wav -m vc -s 3
```

## What is deliberately not in here

* **No build script.** `build.rs` does not exist. The constant tables were parsed out of the reference C headers once and converted to source by `tools/gen_table_source.py`, which can still verify the conversion byte for byte. [TABLE_PROVENANCE.md](TABLE_PROVENANCE.md) records the layout, provenance and element counts.
* **No audio codec.** Every format comes from ffmpeg, so there is no bundled libvorbis or LAME and no bespoke encoder to keep correct.
* **No dependence on the reference projects.** The port was derived from a `libradius` C engine and a `pyradius` Python port, but neither is part of this crate or needed to build it: they are named in `package.exclude`, and their directories need not be present. `cargo build` and `cargo test` are green on a checkout that contains only this crate.
* **Almost no dependencies.** The engines, the C ABI and the tables build with nothing but `std`. The crate adds `clap` for the CLI's argument parsing and `anyhow` for its error reporting, and both FFT backends are always compiled.
* **Nothing dead in `src/`.** Every constant table under `src/vocoder/table_data/` is read by a code path. The research corpus' per-granule schedule, which the engine does not read because it computes the schedule from a closed form, is archived under [reference/](reference/) rather than compiled in.

## Float discipline

The reference engine is compiled with `-ffp-contract=off` and keeps almost everything in `f32`. The port mirrors that exactly:

* no `f64` creep in the hot paths;
* `a*b + c` stays **two** operations, matching `-ffp-contract=off`. `mul_add` is used only where the C source calls `fmaf`, through `radius_rs::vocoder::fma`;
* transcendentals are evaluated in `f64` and rounded once, which is how the C compiler lowers `expf`, `logf`, `powf`, `cosf` and `sinf` here;
* twiddle factors and engine constants are written as **bit patterns**, so a decimal literal can never drift across two conversions.

That discipline is what buys the `max|d| = 3.6e-07` agreement reported in [VERIFICATION.md](VERIFICATION.md).

### Hardware FMA: required for speed, and free for accuracy

Keeping `a*b + c` a single correctly-rounded operation has a consequence that is easy to miss: on the default `x86-64` target, whose feature set has no `fma`, `f32::mul_add` **cannot** be a hardware instruction. Every one of the crate's ~73 `mul_add` call sites then calls a software correctly-rounded `fmaf` instead. The vocoder's crossover is 4 bands by 2048 taps of it per input sample, and that alone measured **53% of a whole render**.

[`.cargo/config.toml`](../.cargo/config.toml) therefore builds with `-C target-feature=+fma`. This changes **no output bits**: hardware FMA is one correctly-rounded operation, exactly what the software path computes, so the reference agreement above is untouched and the bit-exact tests pass under either setting. What it does cost is a higher CPU floor (AVX + FMA, i.e. Intel Haswell 2013 / AMD Excavator 2015 and newer). Measured on 3 s of 48 kHz stereo:

| | software `fma` | hardware `fma` |
|---|---|---|
| crossover | 3.40 s | 1.11 s |
| full vocoder render | 6.38 s | 3.83 s |

`Crossover::process` additionally dispatches on `is_x86_feature_detected!("fma")` so the hardware path is still taken if the flags are overridden, and the crate still behaves correctly on a CPU without FMA. Build with `RUSTFLAGS=-C target-feature=-fma` for a baseline-CPU binary; use `RUSTFLAGS=-C target-cpu=native` for about 12% more on a known machine.

## Deliberate behaviour differences

Two places where this crate knowingly does not reproduce the reference, both because the reference is wrong:

* **Stereo in the vocoder.** The reference handles stereo incorrectly in two independent ways: the band split reads only channel 0, and `sync_sens_3496` sits at a `0.0` placeholder that reduces the stereo phase synchroniser's weight to exactly zero, so it never runs. Both are fixed here. See the stereo section of [VERIFICATION.md](VERIFICATION.md). The Python reference carries the same two fixes, so the two remain bit-exact with each other, and both now differ from the C engine on stereo material.
* **32-bit integer WAV output.** Rejected with an error rather than silently downgraded, because ffmpeg has no 32-bit integer PCM encoder. See [CLI.md](CLI.md).

`--fft rustfft` is a third case, but it is opt-in rather than a change of default: see [FFT.md](FFT.md).

## Environment variables

| variable | effect |
|---|---|
| `RADIUS_FFMPEG` | path to the ffmpeg binary; otherwise `ffmpeg` on `PATH` |
| `RADIUS_FFT` | default FFT backend, `radix2` or `rustfft`, equivalent to `--fft` |
| `RADIUS_PROGRESS` | force the progress bar on even when stderr is not a terminal |
| `RADIUS_TEST_AUDIO` | acceptance corpus for the tests |
| `RADIUS_AUDITION_REF_DIR` | directory holding the Audition reference renders |
| `RADIUS_VC_FAST_MATH` | opt-in approximation in the vocoder's phase-to-cartesian step, faster but not reference-compatible |
| `RADIUS_PROFILE` | enable internal stage profiling |

The last two are development aids. They change numerical results or add output, so they are not for normal use.
