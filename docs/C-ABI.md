# C ABI

The library exports a small opaque-handle C ABI, declared in [`include/radius_rs.h`](../include/radius_rs.h).

```c
#include "radius_rs.h"

void *st = rx_td_init(48000, 37, 0, 2);
rx_td_set_ratio(st, 3.0, 100.0);
float *out = malloc((frames * 2 + 65536) * sizeof *out);
int64_t n = rx_td_render(st, in, frames, out, frames * 2 + 65536);
rx_td_free(st);
```

See the header for the full contract: opaque handles, interleaved `f32`, offline whole-signal renders, and `-1` on bad arguments.

## Linking

The library is self-contained, so linking is the only build step:

```bash
cargo build --release      # radius_rs.dll/.so/.dylib + radius_rs.lib/.a

# sharing the shared object: nothing extra to link
cc -Iinclude my.c -Ltarget/release -lradius_rs -o my            # Linux / macOS
clang -Iinclude my.c target/release/radius_rs.dll.lib -o my     # Windows

# pulling in the static library instead: the Rust std references system libraries
cc -Iinclude my.c -Ltarget/release -lradius_rs \
   -lws2_32 -luserenv -lbcrypt -lntdll -ladvapi32 -lole32 -loleaut32 \
   -luser32 -lkernel32 -lsynchronization -o my        # Windows
cc -Iinclude my.c -Ltarget/release -lradius_rs -lpthread -ldl -lm -o my   # Linux / macOS
```

On Windows `cargo` names the **static** archive `radius_rs.lib` and the import library `radius_rs.dll.lib`, so a bare `-lradius_rs` silently picks the static one and then fails on missing `__imp_WSAStartup` and friends unless you add the system libraries. Naming `radius_rs.dll.lib` explicitly is what shares the DLL.

## Worked examples

Both examples are deliberately minimal, with no file I/O and no WAV parsing, so the ABI is the only thing on screen. Each builds a synthetic stereo signal in memory (a sine per channel, a slow tremolo, a touch of noise), pushes it through both engines, prints the frame count and peak, and shows that an unsupported sample rate is reported rather than undefined.

* [`include/examples/example.c`](../include/examples/example.c) — C, standard library only.
* [`include/examples/example.py`](../include/examples/example.py) — the same from Python with `ctypes` and numpy. It finds the library next to `target/release/`, or takes `--lib PATH` or `RADIUS_RS_LIB`.

```bash
cc -Iinclude include/examples/example.c -Ltarget/release -lradius_rs -lm -o example
./example

python include/examples/example.py
```

```text
radius-rs 0.1.0
in : 24000 frames, 2 ch, peak 0.3589
td : hop 2664, pitch FFT 8192, window 3330
td : +3 semitones -> 23439 frames, peak 0.3596
vc : -3 semitones -> 24000 frames, peak 0.3724
vc : 22050 Hz correctly rejected
```

The two examples print very close but not identical numbers: they share the signal generator, but the noise differs, since the C file uses a linear congruential generator and Python uses numpy's PCG64. The granule count depends on the pitch tracker, which keys off the tonal part, so the two agree on structure while differing slightly in the last digits.

The C example is what `ffi_abi.rs` uses to check the ABI from outside Rust, and it links the static library. `tools/c_abi_check.c` is a third, non-distributed smoke test used during development.
