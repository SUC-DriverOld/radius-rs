# FFT backend

The engines need a real FFT at 8192 points (the TD pitch front-end), 4096/2048
(TD transients info) and 16384 (the vocoder). Two implementations are available:

| backend | source | selectable |
|---|---|---|
| `radix2` **(default)** | [`src/fft_radix2.rs`](../src/fft_radix2.rs) — radix-2 decimation-in-time, the engine's own twiddle table, separate mul/add per butterfly | always |
| `rustfft` | the [`rustfft`](https://crates.io/crates/rustfft) crate — mixed-radix | `--fft rustfft` |

```bash
cargo build --release                      # radix2 only
cargo build --release                    # both, selected at run time

radius in.wav out.wav -m td -s 3 --fft rustfft
```

`src/fft.rs` is a thin wrapper: it fixes the "cart" packing
(`[0]=DC`, `[1]=0`, `[2k]=re[k]`, `[2k+1]=im[k]`, `[N]=Nyquist`, `[N+1]=0`) and
the `1/N` inverse scaling, and picks the kernel. Selection is per thread
(`fft::set_backend`) or process-wide via `RADIUS_FFT`, and the plan cache is
keyed by `(size, backend)` so a mid-process switch cannot hand out the wrong
plan.

## Shipped default: radix-2

The reason is not sentimentality about the code. This crate's acceptance
criterion is agreement with the reference engine, and the reference's own floor
is `max|d| < 5e-4` **and** `corr > 0.9999`. Measured on the acceptance corpus
(+3 semitones, 29.03 s, 48 kHz stereo — not redistributed, see the README), radix-2
reproduces it to `max|d| = 3.6e-07`, `corr = 1.0000000000`, with identical
granule/transient
counts (1702/27).

`rustfft` cannot hold that, because its `f32` output differs from the radix-2
kernel in the last bits of every bin — and in this engine the last bits are load
bearing. The vocoder's peak search, `UnwrapPhase`, and `ApplyPitchCoherence` all
branch on magnitudes and phase differences, so a 1-ULP change can flip a
discrete decision. Measured with `RADIUS_FFT=rustfft`:

| path | result |
|---|---|
| TD (`-m td`) | **bit-identical** to radix-2 (max\|d\| = 0) on the acceptance audio |
| vocoder (`-m vc`) | correlates ~0.994 with radix-2; ~0.02 % of samples identical |

So `rustfft` is a legitimate speed option for the **TD** path, and not a
substitute for the vocoder.

## Speed

Measured on the same 29.03 s file with one binary, one process at a time, selecting
the backend with `--fft` (the `RADIUS_FFT` environment variable remains available
for library callers):

| path | radix-2 | rustfft | speedup |
|---|---|---|---|
| TD (`-m td`) | 1.26 s | 0.54 s | **2.3x** |
| vocoder (`-m vc`) | 45.1 s | 35.9 s | **1.26x** |

Bare transform cost on the same machine (400 iterations of a real `N`-point
forward plus the cart packing each backend needs):

| size | radix-2 | rustfft | speedup |
|---|---|---|---|
| 2048 | 23 µs | 3 µs | 7.5x |
| 4096 | 52 µs | 7 µs | 7.8x |
| 8192 | 149 µs | 54 µs | 2.8x |
| 16384 | 346 µs | 74 µs | 4.7x |

The TD engine's whole-render gain (2.3x) tracks the 8192-point figure because
its pitch front-end is dominated by those transforms. The vocoder gains much
less overall (1.26x) because at 16384 points its per-granule cost is spread
across dozens of operator loops, not just the transform.

## Why not just use `rustfft` everywhere?

Because the artifact's value here is *reproducing a specific engine*, not
producing pleasant-sounding pitch shift. Both properties matter, but they are
different properties:

* If you want the reference's exact output (regression corpus, bit-exact
  acceptance, debugging a difference against the C engine), use the default.
* If you want the TD path faster and can accept that a future input might land
  a pitch decision differently, `--fft rustfft` is a one-word switch. It is
  bit-identical on the shipped corpus, but that is an empirical result on that
  corpus, not a guarantee.

An earlier revision of this crate went further and made `rustfft` a *compile
time* default for the whole engine. That was reverted: it silently cost the
vocoder its acceptance margin, and a build flag is exactly the kind of thing
that gets forgotten between a benchmark and a release.

## Correctness tests

`src/fft_radix2.rs` and `src/fft.rs` carry the checks that make the above
statements falsifiable rather than anecdotal:

* `fwd_matches_a_direct_dft` — the radix-2 forward transform is compared against
  a direct float64 DFT of the same input at 16/64/256/1024 points. A round-trip
  test alone cannot catch a consistently wrong kernel (the inverse undoes the
  same mistake), which is exactly how a dropped `1/N` in the wrapper survived a
  refactor until this test existed.
* `wrapper_matches_direct_radix2` — the cached wrapper must be bit-identical to
  a freshly built plan at 128/512/2048/8192 points, forward and inverse.
* `round_trip_sizes_every_backend` — round trip under every compiled backend.
* `rustfft_agrees_with_radix2_within_tolerance` — the two kernels must at least
  be numerically close, relative to the spectrum peak.
* `pitch_front_end_decisions_are_backend_independent` — the engine's own pitch
  front-end (zero-padded real input, three transforms, whitening, ACF peak
  search) must return the same *discrete* answer under both kernels.
