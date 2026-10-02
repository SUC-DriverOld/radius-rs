/* include/radius_rs.h — C ABI of the Rust rewrite of the Radius TD and phase
 * vocoder engines.
 *
 * Link against the `cdylib` (`radius_rs.dll` / `libradius_rs.so` /
 * `libradius_rs.dylib`) or the `staticlib` (`radius_rs.lib` / `libradius_rs.a`)
 * produced by `cargo build --release`.
 *
 * Contract
 * --------
 *  - Every state object is an opaque handle created by an `*_init` call and
 *    released by the matching `*_free`; a handle must not be used after free.
 *  - Audio is interleaved `float` in the range [-1, 1] and is always `f32`,
 *    matching the engine's native format.
 *  - `rx_td_render` / `rx_vc_render` are offline whole-signal entry points:
 *    they take the complete input and write at most `out_cap` frames, returning
 *    the number of frames written (or -1 on a bad argument).
 *  - The renderers are stateful; create a fresh handle for every render.
 *
 * Example
 * -------
 *   void *st = rx_td_init(48000, 37, 0, 2);
 *   rx_td_set_ratio(st, 3.0, 100.0);
 *   float *out = malloc((frames * 2 + 65536) * sizeof *out);
 *   long n = rx_td_render(st, in, frames, out, frames * 2 + 65536);
 *   rx_td_free(st);
 */
#ifndef RADIUS_RS_H
#define RADIUS_RS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- library-wide -------------------------------------------------------- */

/* NUL-terminated crate version string ("0.1.0"). */
const char *radius_version(void);

/* ---- time-domain (TD) engine -------------------------------------------- */

/* Allocate a TD renderer. `sr` in Hz, `quality` and `solo` match the reference
 * renderer (typical: quality=37, solo=0) and `nch` is the channel count.
 * Returns NULL for nch <= 0. */
void *rx_td_init(uint32_t sr, int quality, int solo, int nch);

/* Release a handle from rx_td_init. NULL is accepted. */
void rx_td_free(void *st);

/* Set the pitch in semitones and the time stretch in percent (100 = unchanged).
 * Call before rx_td_render. */
void rx_td_set_ratio(void *st, double semis, double tempo);

/* Render `nframes` interleaved frames from `in` into `out` (capacity
 * `out_cap` frames). Returns frames written, or -1 on a bad argument. */
int64_t rx_td_render(void *st, const float *in, int64_t nframes,
                     float *out, int64_t out_cap);

/* Derived geometry of a configuration (diagnostics and regression tests). */
typedef struct rx_td_geometry {
    uint32_t hop;             /* +0x334: round(sr*0.001f * (1.5f*quality)) */
    uint32_t f28;             /* +0xF28: (uint)(sr*0.1f+0.5f) & ~7 */
    uint32_t win_max;         /* 4 */
    uint32_t pitch_n;         /* pitch FFT size (next pow2 >= 5*(hop/2)) */
    uint32_t pitch_l1;        /* analysis window length */
    uint32_t pitch_maxbin;    /* whitening upper bin */
    uint32_t pitch_taper_len; /* frequency taper length */
    uint32_t pitch_lo;        /* ACF search start */
    uint32_t pitch_hi;        /* ACF search end */
} rx_td_geometry_t;

/* Fill `out` with the geometry for (sr, quality, solo). Returns 0 on success,
 * -1 when `out` is NULL. */
int rx_td_geometry(uint32_t sr, int quality, int solo, rx_td_geometry_t *out);

/* ---- phase vocoder ------------------------------------------------------ */

/* Allocate a vocoder renderer. Only 44100 and 48000 Hz are supported; `nch`
 * must be > 0 and `precision` matches the reference (typical: 2). Returns NULL
 * for an unsupported configuration. */
void *rx_vc_init(uint32_t sr, int nch, int precision);

/* Release a handle from rx_vc_init. NULL is accepted. */
void rx_vc_free(void *st);

/* Set the pitch in semitones and the time stretch in percent. */
void rx_vc_set_ratio(void *st, double semis, double tempo);

/* Render `nframes` interleaved frames from `in` into `out` (capacity `out_cap`
 * frames). Returns frames written, or -1 on a bad argument. */
int64_t rx_vc_render(void *st, const float *in, int64_t nframes,
                     float *out, int64_t out_cap);

#ifdef __cplusplus
}
#endif
#endif /* RADIUS_RS_H */
