/* include/examples/example.c — call the radius-rs C ABI from C.
 *
 * Deliberately minimal: no file I/O, no WAV parsing, just a synthetic stereo signal
 * pushed through both engines. The point is the ABI itself — handle lifetime, the
 * interleaved f32 layout, and the `(frames written, -1 on error)` return.
 *
 * ---------------------------------------------------------------------------
 * Build (from the repository root, after `cargo build --release`)
 * ---------------------------------------------------------------------------
 * Sharing the library. Watch out on Windows: `cargo` writes the **static** archive
 * as `radius_rs.lib` and the import library as `radius_rs.dll.lib`, so a bare
 * `-lradius_rs` picks the static one — name the import library explicitly to share
 * the DLL, otherwise you are doing the static link below and need the system
 * libraries with it.
 *
 *   clang -Iinclude include/examples/example.c target/release/radius_rs.dll.lib \
 *         -o example                                        # Windows
 *   cc -Iinclude include/examples/example.c -Ltarget/release -lradius_rs \
 *      -o example                                            # Linux / macOS
 *
 * (ELF/Mach-O have no such ambiguity: `libradius_rs.so` is the shared object,
 * `libradius_rs.a` the archive, and `-lradius_rs` prefers the shared object.)
 *
 * Linking it in statically, so there is no shared object to ship:
 *
 *   clang -Iinclude include/examples/example.c -Ltarget/release -lradius_rs \
 *      -lws2_32 -luserenv -lbcrypt -lntdll -ladvapi32 -lole32 -loleaut32 \
 *      -luser32 -lkernel32 -lsynchronization -o example      # Windows
 *   cc -Iinclude include/examples/example.c -Ltarget/release -lradius_rs \
 *      -lpthread -ldl -lm -o example                          # Linux / macOS
 *
 * The extra Windows libraries are what the Rust standard library itself references.
 * There is no Cargo feature to worry about: audio I/O goes through the external
 * ffmpeg binary, so nothing third-party is compiled into the archive.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "radius_rs.h"

/* `M_PI` is not standard C: it is a POSIX/glibc extension that MSVC's <math.h>
 * hides unless `_USE_MATH_DEFINES` is set, so define it here. */
#define PI 3.14159265358979323846

#define SR 48000
#define NCH 2
#define FRAMES (SR / 2) /* 0.5 s is plenty for a demo */
#define SEMITONES 3.0

/* Same generator as include/examples/example.py: a sine on the left, a detuned
 * fifth on the right, a slow tremolo, a touch of noise. */
static void make_signal(float *buf, int frames, int nch) {
    unsigned int rng = 0x12345678u;
    for (int i = 0; i < frames; ++i) {
        double t = (double)i / (double)SR;
        double env = 0.6 + 0.4 * sin(2.0 * PI * 0.7 * t);
        rng = rng * 1664525u + 1013904223u;
        double noise = ((double)(rng >> 8) / (double)(1u << 24) - 0.5) * 0.02;
        for (int c = 0; c < nch; ++c) {
            double f = (c == 0) ? 440.0 : 659.255; /* A4 / E5 */
            buf[i * nch + c] = (float)(0.35 * env * sin(2.0 * PI * f * t) + noise);
        }
    }
}

static float peak_of(const float *x, size_t n) {
    float p = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float a = fabsf(x[i]);
        if (a > p) p = a;
    }
    return p;
}

int main(void) {
    printf("radius-rs %s\n", radius_version());

    const int frames = FRAMES;
    float *in = malloc((size_t)frames * NCH * sizeof *in);
    /* The engines are duration preserving, so the input length plus a granule of
     * slack is always enough. */
    float *out = calloc((size_t)frames * NCH + 65536, sizeof *out);
    if (!in || !out) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    make_signal(in, frames, NCH);
    printf("in : %d frames, %d ch, peak %.4f\n", frames, NCH, peak_of(in, (size_t)frames * NCH));

    /* ---- time-domain engine ------------------------------------------------
     * Quality 37 and solo 0 are the reference renderer's defaults. */
    rx_td_geometry_t geom;
    if (rx_td_geometry(SR, 37, 0, &geom) == 0) {
        printf("td : hop %u, pitch FFT %u, window %u\n",
               geom.hop, geom.pitch_n, geom.pitch_l1);
    }
    void *td = rx_td_init(SR, 37, 0, NCH);
    if (!td) {
        fprintf(stderr, "rx_td_init failed\n");
        return 1;
    }
    rx_td_set_ratio(td, SEMITONES, 100.0);
    int64_t n_td = rx_td_render(td, in, frames, out, frames + 65536);
    rx_td_free(td);
    if (n_td < 0) {
        fprintf(stderr, "rx_td_render failed\n");
        return 1;
    }
    printf("td : %+g semitones -> %lld frames, peak %.4f\n",
           SEMITONES, (long long)n_td, peak_of(out, (size_t)n_td * NCH));

    /* ---- phase vocoder -----------------------------------------------------
     * Only 44100 and 48000 Hz are supported. Create a fresh handle per render: the
     * renderers are stateful. */
    void *vc = rx_vc_init(SR, NCH, 2);
    if (!vc) {
        fprintf(stderr, "rx_vc_init failed (only 44100/48000 Hz are supported)\n");
        return 1;
    }
    rx_vc_set_ratio(vc, -SEMITONES, 100.0);
    int64_t n_vc = rx_vc_render(vc, in, frames, out, frames + 65536);
    rx_vc_free(vc);
    if (n_vc < 0) {
        fprintf(stderr, "rx_vc_render failed\n");
        return 1;
    }
    printf("vc : %+g semitones -> %lld frames, peak %.4f\n",
           -SEMITONES, (long long)n_vc, peak_of(out, (size_t)n_vc * NCH));

    /* ---- bad arguments are reported, not undefined ------------------------- */
    if (rx_vc_init(22050, NCH, 2) != NULL) {
        fprintf(stderr, "expected 22050 Hz to be rejected by the vocoder\n");
        return 1;
    }
    printf("vc : 22050 Hz correctly rejected\n");

    free(in);
    free(out);
    return 0;
}
