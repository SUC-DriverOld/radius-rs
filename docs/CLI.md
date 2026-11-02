# Command line reference

```
radius <INPUT> [OUTPUT] [OPTIONS]
```

`radius -h` prints the option list with one line each; `radius --help` prints the same list with the reasoning, the measured trade-offs and the Audition equivalents. This document is the full version of the latter.

Every option is listed below with its default and the reason for it, together with the equivalent control in Adobe Audition's 效果 → 伸缩与变调 (Stretch and Pitch) dialog, which drives the same Radius algorithm family.

## The verified pitch range is the full ±36 semitones

Both engines were measured on a 440 Hz tone from `+36` down to `−36`, and both hit the requested shift across the whole range:

| requested | TD achieved | vocoder achieved |
|---|---|---|
| +36 | +36.01 | +36.00 |
| +12 | +12.01 | +12.00 |
| 0 | 0.00 | 0.00 |
| −12 | −12.01 | −12.00 |
| −24 | −24.03 | −24.00 |
| −36 | −36.11 | −36.00 |

So Audition's −36 … +36 变调 range is real, with two caveats worth knowing:

* The **vocoder is exact and TD drifts slightly** at the extremes (±0.11 semitones at −36). TD's error grows with the shift; the vocoder's does not.
* The **vocoder used to go silent below about −24 semitones**. That was a port bug, not an engine limit: the feed loop stopped after `target × pitch_ratio` input frames, which at a low ratio is far less than the resampler consumes, so the ring's write position never passed the hop and the drain took nothing. The reference renders −36 fine. The fix was to feed the whole input, and there are now two regression tests over the range (`vocoder_covers_the_full_pitch_range`, `vocoder_pitch_range_lands_on_the_requested_ratio`).

## Input and output

| argument | meaning |
|---|---|
| `INPUT` | Input audio file. Any container the pure-Rust decoders take: wav, flac, ogg/vorbis, mp3, aac/adts, m4a, aiff, au, wavpack, musepack. Its **sample rate and channel count are used as-is** and written unchanged to the output. There is deliberately no rate option, so input and output always agree. Channel counts are carried through the layouts the codecs model: 1, 2, 4, 6, 8 or 12; anything else fails with an error that says so. |
| `OUTPUT` | **Optional.** Output audio file; the extension selects the container unless `--format` overrides it. An unknown extension without `--format` is an error rather than a guess. Omit it entirely and a name is derived from the input, in the input's own format where that format can be written. |

## Omitting the output name

`radius in.flac -m vc -s -3 --tempo 200` writes `in_vc_st-3_tp200.flac` **next to the input**, so `radius /music/a.flac -m vc` writes `/music/a_vc.flac`. The suffix records only what changed about the signal:

| piece | shown when | examples |
|---|---|---|
| mode | always | `vc`, `td` |
| `st<semitones>` | semitones ≠ 0 | `st3`, `st-3`, `st2.5` |
| `tp<tempo>` | tempo ≠ 100 | `tp200`, `tp87.5` |

A whole number loses its fraction, so `-s 3` and `-s 3.0` produce the same name. Everything else about the run — quality, solo, gain, bit depth, FFT backend — is deliberately left out, so the name describes the *shift* rather than every knob.

Beside the input rather than in the working directory is deliberate, and matches what `flac`, `pngquant` and similar tools do when they derive a name. It is the only placement that survives a batch run: `find . -name '*.flac' -exec radius {} -m vc -s 3 \;` would otherwise collect every result into one directory, colliding on the stems and losing track of which output came from which input. It is also what Finder's "Keep Both" does, and the ` (1)` rule below is borrowed from there.

Two further rules make this safe to use from a shell loop:

* **The output format follows the input.** With no `--format` and no output extension, the container comes from the input file, so `in.flac` produces FLAC and `in.wav` produces WAV. An input this crate can read but not write — AAC/M4A, for instance, which the decoders take but for which there is no encoder here — falls back to **WAV**. `--format` outranks both of those, and an explicit output extension outranks the input as well.
* **Nothing is ever overwritten.** If the chosen name already exists, ` (1)`, ` (2)`, … is inserted before the extension until it does not, the way a browser names a second download. This applies to a name you typed as well as to a derived one, so a mistyped output path cannot silently destroy an earlier render.

Quality is never traded for convenience: the bit depth and the lossy-quality defaults are untouched by either rule, so a derived `_vc_st3.flac` is the same 24-bit lossless file you would have got by naming it yourself.

An explicit `OUTPUT` is used exactly as written, relative to the working directory like any other command line path — only the derived case is placed relative to the input.

## Defaults

`radius in.wav out.wav` with no options is a format conversion: it gives the best-quality, least-surprising result and changes nothing about the audio.

| option | default | why this default |
|---|---|---|
| `--mode` | `vc` | The phase vocoder, because it is the mode Audition's 声码器模式 maps to and it holds up better on mixed material. It is much slower — roughly 30x — so `-m td` is the one to reach for when speed matters or the source is monophonic. The reference tooling defaults to the time-domain path (`rx_td_render`, `pyr-td`), so this default is deliberately *not* the reference's. |
| `--semitones` | `0` | A bare invocation is a format conversion, not a silent transposition: nothing is shifted unless you ask. Pass `-s 3` (or any value) to pitch shift. |
| `--tempo` | `100` | No time stretch. With `--semitones` at its default the whole run is an identity render, and any change is one you asked for. |
| `--quality` (td) | `37` | The reference drivers' own default (`[quality=37]`), and the value the acceptance measurements were taken at. |
| `--solo` (td) | `0` | The reference default. `1` (forced steady state) is no better on the acceptance audio, so there is no reason to deviate. |
| `-g, --gain <dB>` | `0` | Output gain, applied to the finished samples after the engine. It is a plain linear scale, so it cannot interact with the reference arithmetic and `--gain 0` leaves the output bit-exact. `-g -6` measures exactly 0.501187x, `-g 6` exactly 1.995262x. | — |
| `--formant-shift <N>` | `0` | Vocoder only. How the formant peaks adapt to the pitch shift, in semitones, range **−12 … +12**. This is Audition's **共振变换**, which Audition documents as "determines how the formant peaks adapt to the pitch shift": `0` shifts the formants and the pitch together and so keeps the timbre and the naturalness (the reference behaviour, bit-exact); above `0` gives a brighter, higher result by moving the formants up, the classic "male voice sounds female"; below `0` does the opposite. The value is an **absolute envelope shift in semitones and does not depend on `--semitones`** — measured on a 100 Hz harmonic series with a formant near 4 kHz, `+6` lands 6.0 … 7.2 semitones up across pitch `+1/+6/+12` (spread 1.2), `−6` lands 4.8 … 6.0 down, and `0` stays within 0.8 of where it started. The range is narrower than Audition's −36 … +36 because the operator's gain map is clamped to +20/−40 dB (`fm_gain2`), so past about an octave the correction saturates and the control stops resolving (measured formant positions at `+9` and `+12` land within 0.3% of the same place); out-of-range values are **clamped, not rejected**. The operator only runs when there is a pitch shift, so this does nothing at `--semitones 0`, and nothing under `--no-preserve-voice`. | 共振变换 (range narrowed) |
| `--no-preserve-voice` | off | Vocoder only. Drops formant preservation so the spectral envelope follows the pitch instead of staying put — Audition's **保持语音特性** unchecked. On a 200 Hz harmonic series with a formant at 2 kHz at +3 semitones: preserved leaves the formant at 1647 Hz (roughly held), unpreserved moves it to 2172 Hz (following the pitch). **Byte-identical to `--formant-shift <semitones>`** across the formant range, so it is the convenient spelling of "formants follow the pitch"; see the relationship note below. | 保持语音特性 (unchecked) |

### `--no-preserve-voice` and `--formant-shift` are the same result

The two settings reach the same output by **different routes**, and both mean "no envelope correction":

| setting | operator state | why nothing happens |
|---|---|---|
| `--no-preserve-voice` | `active = 0` | the whole operator is skipped |
| `--formant-shift <semitones>` | `active = 1`, `ratio = pitch_ratio · 2^(-pitch/12) = 1.0` | `FormantState::apply` returns early on `ratio == 1.0` |

So the **default** — preservation on, `--formant-shift` at its default `0` — already behaves exactly like `--formant-shift <semitones>`: in both cases nothing corrects the envelope, so the formants ride along with the pitch. It is not that the default writes a shift for you; the default and the explicit shift simply land on the same no-op.

That identity is pinned by `no_preserve_voice_equals_formant_shift_at_the_pitch` at ±3, ±7 and ±12 semitones, which also asserts the operator's ratio really is exactly 1.0 in the `--formant-shift` case.

**They diverge at extreme pitches.** At +36 semitones the operator's ±20/−40 dB gain clamps are already fully consumed by the pitch correction, so `--formant-shift` stops resolving. Measured on a 180 Hz harmonic series with a formant at 2160 Hz, the achieved shift is +12.7 / +11.3 / +1.4 semitones for `--formant-shift +12` at pitch +3 / +12 / +24, and at pitch +36 the measured formant is identical for every formant setting, so the control is inert there. How early that bites depends on the source spectrum and not only on the arguments, so it is **not** warned about — the CLI would be guessing.

## Warnings are yellow

Everything the CLI reports as a problem goes through one helper, so it looks alike: a blank line, then the message in yellow on stderr.

* Yellow only when stderr is a terminal, so redirected output and logs stay plain.
* `NO_COLOR` (set to anything non-empty) disables it and **vetoes `RADIUS_COLOR`**, per https://no-color.org.
* `RADIUS_COLOR=1` / `=0` forces colour on or off regardless of the terminal, which is what makes it testable from a pipe.

The warnings are:

| warning | when |
|---|---|
| clipping | the render exceeds full scale and the target quantises (see the clipping section below) |
| `--formant-shift` has no effect at `--semitones 0` | the formant operator only runs alongside a pitch shift. Every value from −12 to +12 renders byte-identically at pitch 0, so the argument genuinely cannot do anything. |
| `--formant-shift` has no effect under `--no-preserve-voice` | that flag turns the operator off outright |
| `--fft` | `radix2` | Reference-compatible FFT. See [FFT.md](FFT.md) for what `rustfft` costs. |
| `--bit-depth` | `32f` | 32-bit IEEE float, the engines' **native** output, so nothing is quantised on the way out. Integers re-quantise to 16 or 24 bit on request. |
| `--ogg-quality` | `0.9` | Top of the practical Vorbis range, so an Ogg output is never the reason for audible loss. |
| `--mp3-bitrate` | `320` | The highest MP3 constant bitrate LAME offers. |

All of these are overridable per run, and `--help` prints the same defaults.

## Engine and shift

| option | default | meaning | Audition equivalent |
|---|---|---|---|
| `-m, --mode <td\|vc>` | `td` | `td` is the time-domain granule engine (`rx_td_render`); `vc` is the phase vocoder (`rx_vc_render`). | **声码器模式** off = `td`, on = `vc` |
| `-s, --semitones <F>` | `0` | Pitch shift in semitones; fractional and negative values are fine. At the default nothing is shifted, so `radius in.wav out.flac` is a lossless format conversion. | **变调** slider (−36 … +36 半音) |
| `--tempo <F>` | `100` | Time stretch in percent: 100 keeps the duration, 200 doubles it, 50 halves it. The pitch is unaffected — the engines deliver this themselves (see the note below). | **伸缩** slider, same percentage readout |
| `--fft <radix2\|rustfft>` | `radix2` | Select the FFT implementation at run time. Both are always compiled in. | — |

### `--tempo` is delivered by the engines, the way `dvaaudiodsp.dll` does it

This used to be a post-engine overlap-add stage in this crate; it is now the real
mechanism, reverse-engineered from Audition's `dsp::stretch::radius::RadiusEngine`
(notes in [RE-dvaaudiodsp-radius-tempo.md](RE-dvaaudiodsp-radius-tempo.md)). Each
engine keeps **two rates**:

* a **drain rate** — the pitch ratio `r = 2^(semitones/12)` — at which its output
  ring is resampled into the final stream;
* a **schedule slope** — `r × stretch` — the rate at which the engine's internal
  output ring advances per input frame.

The stretch scales only the slope. The render then comes out at
`(slope / drain) = stretch` times the input duration with the pitch still at
`semitones`, and the output length is exactly `round(input × stretch)` — the same
arithmetic Audition's reference renders follow (2,787,196 frames for a 2× of this
corpus' 1,393,598, 696,799 for the 50 % case).

In the time-domain engine the slope drives the granule pump — the accumulator
spawns *extra granules* that replay windows, so the ring holds a time-stretched
copy of the input (measured: ring write = `slope × input` frames, exact, at 50 %,
100 % and 200 %) — and the drain resamples it back by the pitch ratio. In the
vocoder the slope is the out-step of the grain scheduler (`vc_calc_sched`).

At `--tempo 100` the slope equals the drain rate and the whole path is
bit-identical to the pitch-only render, so the parity corpus still holds.

### The output can exceed full scale, and Audition's does too

A pitch shift raises the level, so the render can peak above 1.0 — and so does
Audition's. On the corpus at `+3` semitones:

| | peak | samples over full scale | RMS |
|---|---|---|---|
| Audition, time domain, tempo 100 | 1.3247 | 0.2202 % | −7.49 dB |
| this crate, time domain, tempo 100 | 1.2099 | 0.0032 % | −7.69 dB |
| Audition, time domain, tempo 200 | 1.3256 | 0.8070 % | −7.27 dB |
| this crate, time domain, tempo 200 | 1.2110 | 0.0020 % | −8.02 dB |
| Audition, time domain, tempo 50 | 1.2916 | 0.0146 % | −8.00 dB |
| this crate, time domain, tempo 50 | 1.3155 | 0.0352 % | −8.01 dB |
| Audition, vocoder, tempo 100 | 1.8183 | 0.8633 % | −8.08 dB |
| this crate, vocoder, tempo 100 | 1.8940 | 1.1014 % | −7.88 dB |
| Audition, vocoder, tempo 200 | 1.8329 | 0.7510 % | −8.21 dB |
| this crate, vocoder, tempo 200 | 1.8519 | 0.8782 % | −8.08 dB |
| Audition, vocoder, tempo 50 | 1.6253 | 0.4165 % | −8.77 dB |
| this crate, vocoder, tempo 50 | 1.6059 | 0.6385 % | −8.33 dB |

Float WAV stores those values as they are, and the float codecs (ogg, mp3) encode them rather than clip. An integer PCM or FLAC target clamps them, and the CLI warns in yellow when it will (see the warnings section). Use `--gain` to bring the level down first if the target is one of those.

## Engine quality

| option | default | meaning | Audition equivalent |
|---|---|---|---|
| `-q, --quality <1-100>` | `37` | TD only. Sets the granule hop, `hop = round(sr·0.001 · 1.5·quality)`, that is how many analysis granules per second. Larger = less granular = faster. At 48 kHz: `-q 1` → hop 72, `-q 37` → hop 2664, `-q 100` → hop 7200, which on a 5 s slice means 10 609 / 268 / 100 granules. | 精度 (affects the TD path) |
| `--solo <0\|1>` | `0` | TD only, and it is a real mode switch rather than a knob. `0` is the default and the full engine: it re-estimates the pitch every granule. `1` forces the engine's steady-state path, which uses a fixed period and a narrow pitch search — measured **about 3x faster** (84 ms against 235 ms on a 5 s slice) and audibly different (345 granules against 268, peak 1.126 against 1.210). It is not a quality improvement, just a cheaper analysis. | — |
| `-g, --gain <dB>` | `0` | Output gain, applied to the finished samples after the engine. It is a plain linear scale, so it cannot interact with the reference arithmetic and `--gain 0` leaves the output bit-exact. `-g -6` measures exactly 0.501187x, `-g 6` exactly 1.995262x. | — |

### `--precision` and `--pitch-coherence` are not exposed

Two reference parameters are deliberately absent from the CLI because they cannot do what their name suggests:

* **`precision`** (the old `-p, --precision <1-9>`) selected an overlap-add write gain, so it only changed the output level, and `3..=9` were byte-identical. Level is now the explicit `--gain`, and the reference's value is pinned internally so the parity corpus keeps matching. The C ABI's `rx_vc_init` dropped its third argument for the same reason.
* **`pitch-coherence`** (Audition's 音调一致, the reference's `trans_sens`) is still a library setter, `VocoderState::set_pitch_coherence`, but is not on the command line. Every signal tried — a pure tone, a six-note chord with vibrato and noise, a 200 Hz harmonic series — renders byte-identically at every value in range: the stage runs, but no region crosses its threshold. It is kept in the library because the reference has the parameter and the wiring is real, so a corrected threshold could still use it.

## Output format and quality

| option | default | meaning | notes |
|---|---|---|---|
| `-f, --format <wav\|flac\|ogg\|mp3>` | from the extension | Force the output container, for when the extension is ambiguous or must stay as-is. | An unknown extension **without** `--format` is an error |
| `-b, --bit-depth <16\|24\|32\|32f>` | `32f` | Sample depth for wav and flac. `32f` is IEEE float, which is what both engines natively produce; `16`, `24` and `32` are integer PCM. FLAC stores `32` and `32f` as 24 bit, the deepest subframe the format has. | wav: `16`, `24`, `32`, `32f`; flac: `16`, `24` |
| `--ogg-quality <0.0-1.0>` | `1.0` | Ogg/Vorbis encoder quality, on the pure-Rust encoder's normalized scale. | higher is bigger and better |
| `--ogg-bitrate <KBPS>` | unset | Target average bitrate for Ogg/Vorbis: the quality whose *measured* bitrate is closest to the request. Vorbis is VBR, so the result is approximate (roughly ±10% on programme-length material); the encoder's embedded profile reaches about 30–180 kbit/s stereo, and requests outside that range clamp to its ends. On very short files the fixed lead/pad overhead (~230 ms of encoded audio) pushes the achieved rate up. | conflicts with `--ogg-quality` |
| `--mp3-bitrate <KBPS>` | `320` | MP3 constant bitrate, snapped to the nearest value the MPEG tables carry (64, 96, 128, 160, 192, 256, 320, …). | higher is bigger and better |

The bitrate→quality table behind `--ogg-bitrate` is measured, not theoretical: the curve in `src/io.rs` (`VORBIS_KBPS_AT_Q`) was produced by encoding a dense seven-tone mix with vibrato and noise at 44.1 kHz stereo, four seconds of it, at nineteen quality points. Keep it in mind when reading the numbers: a pure tone codes for a fraction of any target, because VBR spends bits on entropy, not on the knob.

The CLI prints what it actually wrote, for example `wrote out.flac (1393118 frames, 29.023s, flac 24-bit, peak 1.2133)`, so the effective container, depth and achieved peak are always visible.

## Diagnostics

| option | meaning |
|---|---|
| `--truth <FILE>` | Render a correlation report against a reference render over the common prefix (`corr`, `max\|d\|`), like the C driver's trailing `truth.wav`. |
| `-v, --verbose` | Print the final cursors and write position. |
| `--no-progress` | Do not draw the progress bar. |
| `-h`, `-V` | `--help` lists everything, `--version` prints the crate version. |

When a run changes nothing (`-s 0` at `--tempo 100`, which is the bare-invocation default) the CLI says so, so an accidental no-op is visible rather than silent:

```text
$ radius in.wav out.flac
cfg: mode=Td semitones=+0.00 tempo=100.0% quality=37 solo=0
render: out=1393598 frames (29.033s)  granule=1 transient=0 wrap=0  [ 40 ms, 725.8x RT]
note: -s 0 at 100% tempo is an identity render (format conversion only)
wrote out.flac (1393598 frames, 29.033s, flac 24-bit, peak 0.9772)
```

The renderer also reports its own timing, for example `[1830 ms, 15.9x realtime]`. The `x RT` figure is the input's duration divided by the render time, so it is how many seconds of audio the engine produces per second of wall clock.

## Progress

A render can take a while, because the vocoder runs slower than real time, so the CLI draws a progress bar **on stderr**:

```
vc  [=======                     ]  24.3%     9.2s elapsed, 28.6s left
```

It appears only when stderr is a terminal, so logs, pipes and scripts are unaffected. `--no-progress` suppresses it regardless, and `RADIUS_PROGRESS=1` forces it on for logging, one line per update instead of a carriage-return repaint. The percentage is driven by the engine's own per-granule counter, so the bar is monotone, and the remaining-time estimate starts as `eta --` and settles within about a second. Everything machine-readable (`in :`, `cfg:`, `render:`, `wrote`) stays on **stdout**, so `radius ... > log` still gives a clean log.

## Clipping

Whether a render that exceeds full scale matters depends on the output format, and the CLI says so instead of letting it pass silently. The engines always hand the codecs their native `f32` samples; the difference is the codec on the other side. **32-bit float WAV** stores a sample outside `[-1, 1]` exactly, and the float codecs **Ogg/Vorbis and MP3** take the same `f32` stream unclamped — they encode the overshoot rather than clip it, approximately, being lossy (the MP3 encoder clamps only beyond ±8, far past anything a render produces). Only integer PCM WAV and FLAC (which has no float subframe, so every depth lands in an integer one) have to clamp, and only they warn:

```
$ radius in.wav out.flac -m vc --semitones=-3
WARNING: the render peaks at 1.8278 (> 1.0), so 38422 sample(s) (1.601%) clip when
written as flac. 32-bit float WAV stores values beyond [-1, 1] exactly
(-f wav -b 32f); the float codecs (ogg, mp3) keep them approximately; lower the
level first if you need an integer or FLAC target. (24 more sample(s) within
0.0001 of full scale)

$ radius in.wav out.wav -m vc --semitones=-3     # -b 32f is the default
note: peak 1.8278 (> 1.0) — kept exactly (32-bit float)

$ radius in.wav out.mp3 -m vc --semitones=-3 --mp3-bitrate 320
note: peak 1.8278 (> 1.0) — not clipped (float codec; lossy, so approximate)
```

Overshoot is normal for these algorithms rather than a bug: overlapping granules and transient re-phasing sum past unity, and the amounts seen here (up to about 1.8 on the acceptance material at −3 semitones) are what the reference engine produces too. The warning exists so that an integer or lossy export is a decision rather than a surprise. It is silent below full scale, and near-full-scale samples are counted separately because `f32` quantisation can round a sample sitting just under 1.0 up to exactly 1.0.

## Codecs

All audio I/O is in-process and pure Rust — the day-one crate doc's promise, finally true:

| direction | format | engine |
|---|---|---|
| read | wav, flac, ogg/vorbis, mp3, aac/adts, m4a, aiff, au, wavpack, musepack | `oxiaudio` (symphonia underneath) |
| write | wav (16/24/32-bit integer, 32-bit float) | `oxiaudio` (hound) |
| write | flac (16/24-bit integer) | `oxiaudio` (flacenc) |
| write | ogg/vorbis (mono/stereo) | `rusty_vorbis`, framed by the `ogg` crate |
| write | mp3 (mono/stereo, CBR, gapless LAME tag) | `rusty_mp3` |

Two stories worth knowing:

* **Why not oxiaudio's Vorbis encoder?** It was tried first and is unusable: on real material it writes a stream that both symphonia and ffmpeg decode to garbage (max |diff| ~1e7, SNR below −130 dB), truncates to a block boundary, and its quality knob changes nothing — the same file at q0.2 and q0.9, byte for byte. `rusty_vorbis` (same family as `rusty_mp3`) behaves like a real encoder and took the job.
* **Ogg length exactness.** The Vorbis encoder hops in 1024-sample blocks and its raw packets make every decoder skip the first hop and never render the last one's overlap. The mux layer compensates: one silent *lead* hop as the overlap partner of the first real hop (its content never survives decoding — copy, reversed copy and zeros all decode identically, so it is zeros), a tail pad so the last real hop gets its partner, a final page granule of `hop + input_frames`, and the first audio packet alone on its page the way libvorbis lays out a stream. Read back through this crate the length equals the input length exactly, sample-aligned at lag zero — pinned by the io and acceptance tests. One honest caveat: a decoder that ignores granule trimming altogether (ffmpeg's vorbis path does) plays about two extra hops of silence around the programme instead; rusty_vorbis cannot produce libvorbis's priming-free first block, so that is the floor for a pure-Rust Vorbis today.

MP3 writing carries a gapless LAME tag (encoder delay and end padding recorded, `rusty_mp3`'s own doing), so tag-aware players decode exactly the rendered length. MP3 itself only carries nine sample rates (8/11.025/12/16/22.05/24/32/44.1/48 kHz); a render at any other rate refuses the mp3 target with the list rather than silently resampling — which is also what ffmpeg did.

## Sample rates

The TD engine accepts any rate its geometry supports. The vocoder accepts **8000–192000 Hz**: its crossover, window and synthesis banks are designed at init from the rate, exactly like the original DLL (static analysis found no per-rate tables in the DLL — see `docs/TABLE_PROVENANCE.md` and the RE notes §12; the crossover FIR law is decompiled from the DLL and the window laws were measured on the reference engine's own 44.1/48 kHz data, both validated against Audition reference renders). Nothing is ever resampled: output rate equals input rate.

## Examples

```bash
# time-domain, +3 semitones, verified against a reference render
radius in.wav out_td.wav -m td -s 3 --truth reference.wav

# phase vocoder (Audition's 声码器模式), -3 semitones, 24-bit FLAC
radius in.wav out_vc.flac -m vc --semitones=-3 -b 24

# 320 kbps MP3 from a FLAC source, forced container
radius in.flac out.dat --format mp3 --mp3-bitrate 320 -m vc -s -3

# format conversion only
radius in.wav out.flac -s 0 -b 24

# faster FFT on the time-domain path, bit-identical on the corpus
radius in.wav out.wav -m td -s 3 --fft rustfft
```
