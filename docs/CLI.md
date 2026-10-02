# Command line reference

```
radius <INPUT> <OUTPUT> [OPTIONS]
```

Every option is listed below with its default and the reason for it, together with the equivalent control in Adobe Audition's 效果 → 伸缩与变调 (Stretch and Pitch) dialog, which drives the same Radius algorithm family.

## Input and output

| argument | meaning |
|---|---|
| `INPUT` | Input audio file. Any container or codec ffmpeg can read: wav, flac, ogg/vorbis, mp3, aac/adts, m4a/alac, mkv/webm, aiff, caf, … Its **sample rate and channel count are used as-is** and written unchanged to the output. There is deliberately no rate option, so input and output always agree. |
| `OUTPUT` | Output audio file. The extension selects the container unless `--format` overrides it. An unknown extension without `--format` is an error rather than a guess. |

## Defaults

`radius in.wav out.wav` with no options is a format conversion: it gives the best-quality, least-surprising result and changes nothing about the audio.

| option | default | why this default |
|---|---|---|
| `--mode` | `td` | The reference tooling's default path (`rx_td_render`, `pyr-td`). Roughly 30x faster than the vocoder, and on the acceptance audio it tracks the Audition/iZotope reference at least as closely (level +0.37 dB, energy-contour correlation 0.864, against +0.86 dB / 0.828 for the vocoder). Use `-m vc` for dense polyphonic material. |
| `--semitones` | `0` | A bare invocation is a format conversion, not a silent transposition: nothing is shifted unless you ask. Pass `-s 3` (or any value) to pitch shift. |
| `--tempo` | `100` | No time stretch. With `--semitones` at its default the whole run is an identity render, and any change is one you asked for. |
| `--quality` (td) | `37` | The reference drivers' own default (`[quality=37]`), and the value the acceptance measurements were taken at. |
| `--solo` (td) | `0` | The reference default. `1` (forced steady state) is no better on the acceptance audio, so there is no reason to deviate. |
| `--precision` (vc) | `2` | The reference driver's default (`[precision=2]`) and the value corresponding to Audition's 精度 = 高. |
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
| `--tempo <F>` | `100` | Time stretch in percent: 100 keeps the duration, 200 doubles it, 50 halves it. | **伸缩** slider, same percentage readout |
| `--fft <radix2\|rustfft>` | `radix2` | Select the FFT implementation at run time. Both are always compiled in. | — |

## Engine quality

| option | default | meaning | Audition equivalent |
|---|---|---|---|
| `-q, --quality <1-100>` | `37` | TD only. Sets the granule hop, `hop = round(sr·0.001 · 1.5·quality)`, that is how many analysis granules per second. Higher is finer and slower. At 48 kHz, 37 gives hop 2664. | 精度 (affects the TD path) |
| `--solo <0\|1>` | `0` | TD only. `0` re-estimates the pitch every granule; `1` forces the engine's steady-state path with a fixed period. | — |
| `-p, --precision <1-9>` | `2` | Vocoder only. Controls the analysis and synthesis refinement: 1 is the fastest and roughest, 9 the slowest and cleanest. | 精度 = 高 maps to the default `2` |

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
render: out=1393598 frames (29.033s)  granule=1 transient=0 wrap=0  [ 40 ms, 34839.9x RT]
note: -s 0 at 100% tempo is an identity render (format conversion only)
wrote out.flac (1393598 frames, 29.033s, flac 24-bit, peak 0.9772)
```

The renderer also reports its own timing, for example `[1830 ms, 15.9x realtime]`.

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
