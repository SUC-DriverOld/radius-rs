# Verification

## Accuracy against the reference

Measured on the acceptance corpus (29.03 s, 48 kHz stereo, 1 393 598 frames, TD `quality=37 solo=0`, vocoder `precision=2`) against the `pyradius` reference renders produced by `tools/gen_refs.py`.

The corpus is not redistributed, so reproducing these numbers means supplying it yourself: see [Test audio](#test-audio) below. The counts are also asserted by `td_acceptance.rs` when the file is present, which is why a substituted file with the right name but different content is rejected.

| case | engine | max&nbsp;\|d\| | mean&nbsp;\|d\| | corr | structural |
|---|---|---|---|---|---|
| `+3` semitones | TD | 3.58e-07 | 1.30e-08 | 1.0000000000 | 1702 granules / 27 transients, exact match |
| `-3` semitones | TD | 3.58e-07 | 1.30e-08 | 1.0000000000 | 1719 granules / 19 transients, exact match |
| `+3` semitones | vocoder | 3.58e-07 | — | 1.0000000000 on both channels | output length exact (1 393 598 frames) |
| `-3` semitones | vocoder | 3.58e-07 | — | see the stereo note | output length exact |

The reference's own acceptance floor is `max|d| < 5e-4` **and** `corr > 0.9999`. Every row sits orders of magnitude inside it.

The residual is pure `f32` associativity noise: sample counts, granule counts, transient counts, output lengths and amplitudes all match exactly, and roughly two thirds of all samples are bit-identical.

The references are produced by the development tool `tools/gen_refs.py`, which runs the `pyradius` reference implementation and writes the renders into `.ref/`. That tool needs a `pyradius` checkout, which is not part of this crate; the parity tests pick up `.ref/` automatically and skip when it is absent.

## Stereo, and why the vocoder used to collapse it

The reference's `rx_vc_feed` splits **only channel 0** into bands, by calling `rx_crossover_process1_f(..., 0, src[i * nch], ...)`, and scatters them into the single band ring that every channel's analysis later reads. The synthesis side *is* per channel, since `ev_assembly` loops `for r in 0..nch`, so the output is not literally one mono channel: it is the left channel's spectrum synthesised twice, with identical magnitudes and synchronised phases. That measures exactly like it sounds:

| input | before (C engine semantics) | after |
|---|---|---|
| stereo pair with R = −L | corr(L,R) **+1.0000**, an inverted pair collapses | corr(L,R) **−1.0000** |
| L 440 Hz, R 1200 Hz at −8 dB | R/L **+0.00 dB**, the right channel is discarded | R/L **−8.46 dB** |

Both this crate (`VocoderState::feed`) and the Python reference (`VocoderState.feed` in `pyradius.vocoder_core`) now run one `Crossover` **per channel**, and each channel's granule is built from its own band ring. The two remain bit-exact with each other: on the corpus at +3 semitones, `max|d| = 3.6e-07` and `corr = 1.0000000000` on both channels.

The consequence to know about is that **on stereo input the vocoder no longer matches the C engine**, which still has the defect. The C engine was not changed. On dual-mono input the three agree, which is the third assertion of `vocoder_preserves_stereo_separation`.

## Reference renders from Adobe Audition (perceptual)

The two Audition renders used for this comparison are **not in this repository**, since they are another vendor's encoder output applied to audio this project has no rights to, and neither is the input. `tests/audition_reference.rs` documents how to supply them with `RADIUS_AUDITION_REF_DIR` and skips when they are absent.

They were made with 算法 = IZotope Radius, 精度 = 高, 伸缩 = 100 %, 变调 = ±3 半音, 声码器模式 on, 保持语音特性 on, 音调一致 = 1.

They are a **perceptual reference, not a sample-exact oracle**. Audition's engine and the `libradius` lineage this crate follows are two different STFT implementations, so they do not correlate sample by sample, and that is true of the `pyradius` reference as well:

| comparison | full-band sample corr | low-frequency envelope corr |
|---|---|---|
| Audition vs this crate | 0.03 | 0.91 |
| Audition vs `pyradius` | 0.03 | — |

The agreement that does hold, and that `tests/audition_reference.rs` asserts:

| quantity | Audition | this crate |
|---|---|---|
| pitch shift, +3 case | +2.99 semitones | +3.01 semitones |
| pitch shift, −3 case | −2.99 semitones | −3.01 semitones |
| output length | exactly the input length | exactly the input length |
| overall level | — | +0.9 dB (+3) / +1.1 dB (−3) |
| per-band delay spread vs Audition | — | ≈ 8.6 ms across 40 Hz … 8 kHz |

The remaining differences are the proprietary engine's window and hop geometry, its transient handling and its spectral tilt. Closing them would need the engine binary to reverse-engineer, not more work on this port.

## What the test suite covers

```bash
cargo test --release                       # engine, CLI and C ABI tests
cargo test --release -- --ignored          # the slow full-length renders
cargo test --release -- --include-ignored  # everything, including those
```

All of it runs with no external audio files and no extra cargo features. The corpus-dependent assertions, meaning the exact granule and transient counts, the output lengths and the parity against `pyradius`, require the acceptance audio and report `SKIP` without it, and the container round trips skip when ffmpeg is missing.

* **engine unit tests** — bit-exact golden vectors minted from the Python reference for every operator: the FFT, `fill_granule` across 16 branch and parameter combinations, `cart_to_polar`, `unwrap_phase`, `ApplyPitchCoherence`, `ResetPhasesForTransients`, `adjust_multiphase_diff`, `SynchronizeStereoPhases`, `Randomize` and `SubstituteNoisyPhases`, `OverlapAddChannel`, `Crossover`, `FormantState`, the transients-info geometry, the sampler, and the constant tables.
* **TD acceptance** — duration, amplitude and stereo-image preservation, both pitch directions, a *measured* semitone check on a synthetic tone, and the corpus granule and transient counts.
* **vocoder acceptance** — geometry for both sample rates, the scheduler's structural invariants, a short synthetic render, stereo separation, and an `#[ignore]`d full-length parity run.
* **CLI acceptance** — help, version and argument errors, a real render, the `--truth` correlation report, rejection of an unsupported vocoder rate, a write and read round trip for every container, the documented defaults including the identity-render note, the progress bar in all three modes, the clipping warning per format, and an exhaustive check of the quality knobs: wav `-b 16/24/32f` are confirmed by ffmpeg to be `pcm_s16le`, `pcm_s24le` and `pcm_f32le` and grow in that order, FLAC 16 and 24 are lossless to within their depth's quantisation step, MP3 320 kbps is larger than 128 kbps, Ogg q=0.9 is larger than q=0.1, `--format` overrides the extension, and an unknown extension is rejected.
* **C ABI acceptance** — the exported symbols called through `extern "C"`, `rx_td_render` bit-identical to the Rust API, the null and bad-argument contract, and the vocoder render reachable from C.
* **Audition reference** — the duration, level and energy-contour checks above.

## Test audio

`cargo test` needs no external audio files. The always-on tests drive a deterministic synthetic programme, a chord plus a gliding vibrato voice plus a hairpin chirp plus a silent gap, built by `tests/common/mod.rs`, which exercises both engines, both sample rates, the stereo paths, the codecs and the FFI.

The **acceptance** tests were calibrated on a 29.03 s 48 kHz stereo music excerpt that is **not redistributed**, because it is not this project's to ship, and neither are the two Adobe Audition renders. Those tests detect the absence and print a `SKIP` reason instead of failing or passing vacuously:

```
SKIP td_acceptance_granule_structure: acceptance audio not present at tests/test.wav
     (the music excerpt and the Audition renders are not redistributed; set
     RADIUS_TEST_AUDIO to run against your own material, ...)
```

To run them, supply the material yourself:

```bash
RADIUS_TEST_AUDIO=/path/to/test.wav \
RADIUS_AUDITION_REF_DIR=/path/to/audition_renders \
  cargo test --release -- --include-ignored
```

`tests/common/mod.rs` documents the fingerprint the corpus must have: 1 393 598 frames by 2 channels, 48 kHz, 32-bit float, peak 0.977238, RMS 0.418927, and its SHA-256. A file of the right name but the wrong content fails loudly rather than producing plausible-looking numbers, because several assertions are exact counts of *that* material.

## Performance

Release build, one machine, single-threaded, 29.03 s of 48 kHz stereo. Timings are from the CLI's own `render:` line, which reports the time it measured.

| engine | `--fft` | time | throughput |
|---|---|---|---|
| TD, `+3` | `radix2` | 1.83 s | ~15.9x realtime |
| TD, `+3` | `rustfft` | 0.75 s | ~38.7x realtime |
| vocoder, `+3` | `radix2` | 68.5 s | ~0.42x realtime |
| vocoder, `+3` | `rustfft` | 57.1 s | ~0.51x realtime |

The vocoder is inherently expensive, since it does 16384-point FFTs per granule; the TD engine runs far faster than real time. [FFT.md](FFT.md) explains what the faster backend costs in accuracy.
