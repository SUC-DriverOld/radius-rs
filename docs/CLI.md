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
| `INPUT` | Input audio file. Any container or codec ffmpeg can read: wav, flac, ogg/vorbis, mp3, aac/adts, m4a/alac, mkv/webm, aiff, caf, … Its **sample rate and channel count are used as-is** and written unchanged to the output. There is deliberately no rate option, so input and output always agree. |
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

* **The output format follows the input.** With no `--format` and no output extension, the container comes from the input file, so `in.flac` produces FLAC and `in.wav` produces WAV. An input this crate can read but not write — AAC/M4A, for instance, which ffmpeg decodes but for which there is no encoder here — falls back to **WAV**. `--format` outranks both of those, and an explicit output extension outranks the input as well.
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
| time stretching is this crate's own | `--tempo` is anything but 100. The engines reproduce libradius for *pitch*; the stretch is a separate overlap-add stage written here, so its result may differ from Audition's. At `--tempo 100` the stage never runs and there is nothing to warn about. |
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
| `--tempo <F>` | `100` | Time stretch in percent: 100 keeps the duration, 200 doubles it, 50 halves it. The pitch is unaffected — see the note below, because this one is **not** delivered by the engines, and the engines are deliberately told `tempo = 100` so they do not also stretch internally. | **伸缩** slider, same percentage readout |
| `--fft <radix2\|rustfft>` | `radix2` | Select the FFT implementation at run time. Both are always compiled in. | — |

### `--tempo` is a stage after the engine, not a knob on it

Both engines apply the pitch with a resampler whose rate *is* the pitch, and their granule scheduler walks the input exactly once. That one mechanism sets both the pitch and how much audio they can emit, so they cannot stretch: given more output frames than the input's length they render the input and then pad with silence, and given fewer they render a prefix of it and stop. Measured on a 4 s tone at `-s 3`, the output was 8.12 s at `-t 200` but only 4.07 s of it was audio, and 1.96 s at `-t 50` containing just the first 1.96 s at the original speed.

So `--tempo` is applied **after** the engine, by [`src/stretch.rs`](RUST-API.md): the engines are asked for the natural-length, correctly pitched signal, and that signal is then re-timed by overlap-add with a similarity search (WSOLA), which changes the duration without touching the pitch. The run reports the stage separately:

```text
render: out=1393118 frames (29.023s)  granule=1702 transient=27 wrap=0  [1350 ms, 21.5x RT]
tempo: 1393118 -> 2787196 frames (x2.0000)  overlap-add, 1361 windows, mean |offset| 252.2 samples, level +2.59 dB  [1134 ms]
```

### The engines are told the pitch and nothing else

`set_ratio` is called with `tempo = 100` on both engines. This is not a detail — passing the real tempo as well makes the engine stretch *internally* too, and then the output is stretched twice:

* The time-domain engine's `total_ratio = pitch_ratio × stretch`, so `-s 3 -t 200` drives it at **2.378**. That doubles the granule overlap and clips: measured on the corpus, the engine's own output peaked at **1.288 with 10 000 samples (0.36%) over full scale**, against 1.111 and 9 samples at tempo 100. The vocoder was less affected because its pitch and stretch ratios are separate, which is why only `--mode td` sounded broken.
* Once the engine does pitch only, its output peaks at 1.111 like any other tempo, and the result matches Audition: **RMS −8.29 dB against their −7.95, peak 1.284 against 1.279**.

So the split is: **the engine owns pitch, the stretch stage owns duration.** Neither should do the other's job.

Two more consequences worth knowing:

* **`--tempo 100` skips the stage entirely**, so the default path is byte-for-byte what it always was and the parity corpus still holds.
* The two engines share one stretch implementation, so they cannot disagree about it.

### The output can exceed full scale, and Audition's does too

A pitch shift raises the level, so the render can peak above 1.0 — and so does Audition's. On the corpus at `-s 3 -t 200`:

| | peak | samples over full scale | RMS |
|---|---|---|---|
| Audition, time domain | 1.2790 | 0.3301% | −7.95 dB |
| this crate, time domain | 1.2844 | 0.3615% | −8.29 dB |
| Audition, vocoder | 1.6449 | 0.3187% | −8.88 dB |
| this crate, vocoder | 1.6221 | 0.4112% | −8.45 dB |

Float WAV stores those values as they are. An integer or lossy target clamps them, and the CLI warns in yellow when it will (see the warnings section). Use `--gain` to bring the level down first if the target is not float.

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
| `-b, --bit-depth <16\|24\|32\|32f>` | `32f` | Sample depth for wav and flac. `32f` is IEEE float, which is what both engines natively produce; `16` and `24` are integer PCM. `32` is rejected on purpose: ffmpeg has no 32-bit integer PCM encoder, and quietly writing 24 bit instead would be worse. FLAC stores `32` and `32f` as 24 bit, the deepest subframe the format has. | wav: `16`, `24`, `32f`; flac: `16`, `24` |
| `--ogg-quality <0.0-1.0>` | `0.9` | Ogg/Vorbis encoder quality. This crate's 0..1 is passed to libvorbis as 0..10. | higher is bigger and better |
| `--mp3-bitrate <KBPS>` | `320` | MP3 constant bitrate: 64, 96, 128, 160, 192, 256 or 320. | higher is bigger and better |

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

Whether a render that exceeds full scale matters depends on the output format, and the CLI says so instead of letting it pass silently. Of the four targets, only **32-bit float WAV** can store a sample outside `[-1, 1]`; integer PCM, FLAC (lossless *within* its integer range), Ogg and MP3 all have to clamp:

```
$ radius in.wav out.flac -m vc --semitones=-3
WARNING: the render peaks at 1.8278 (> 1.0), so 38422 sample(s) (1.601%) clip when
written as flac. 32-bit float WAV is the only target that stores values beyond
[-1, 1] (-f wav -b 32f); lower the level first if you need an integer or lossy
format. (24 more sample(s) within 0.0001 of full scale)

$ radius in.wav out.wav -m vc --semitones=-3     # -b 32f is the default
note: peak 1.8278 (> 1.0) — kept exactly because 32-bit float wav stores any value
```

Overshoot is normal for these algorithms rather than a bug: overlapping granules and transient re-phasing sum past unity, and the amounts seen here (up to about 1.8 on the acceptance material at −3 semitones) are what the reference engine produces too. The warning exists so that an integer or lossy export is a decision rather than a surprise. It is silent below full scale, and near-full-scale samples are counted separately because `f32` quantisation can round a sample sitting just under 1.0 up to exactly 1.0.

## Sample rates

The TD engine accepts any rate its geometry supports. The vocoder accepts **44100 and 48000** only, like the C `rx_vc_render`; anything else is rejected with a clear message rather than resampled.

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
