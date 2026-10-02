#!/usr/bin/env python3
"""example.py — call the radius-rs C ABI from Python with `ctypes`.

Deliberately minimal: no file I/O, no WAV parsing, just a synthetic stereo signal
pushed through both engines. The point is the ABI itself — handle lifetime, the
interleaved f32 layout, and the `(frames written, -1 on error)` return.

Requires numpy (for the signal) and the built shared library:

    cargo build --release

    python include/examples/example.py                  # finds target/release/
    python include/examples/example.py --lib /path/to/radius_rs.dll
    RADIUS_RS_LIB=/path/to/radius_rs.dll python include/examples/example.py
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
from pathlib import Path

import numpy as np

SR = 48_000
NCH = 2
FRAMES = SR // 2  # 0.5 s is plenty for a demo
SEMITONES = 3.0


class RxTdGeometry(ctypes.Structure):
    """`rx_td_geometry_t` from include/radius_rs.h (diagnostics only)."""

    _fields_ = [
        ("hop", ctypes.c_uint32),
        ("f28", ctypes.c_uint32),
        ("win_max", ctypes.c_uint32),
        ("pitch_n", ctypes.c_uint32),
        ("pitch_l1", ctypes.c_uint32),
        ("pitch_maxbin", ctypes.c_uint32),
        ("pitch_taper_len", ctypes.c_uint32),
        ("pitch_lo", ctypes.c_uint32),
        ("pitch_hi", ctypes.c_uint32),
    ]


def load_library(path: str | None = None) -> ctypes.CDLL:
    """Load the shared library and declare every prototype we use."""
    if path is None:
        path = os.environ.get("RADIUS_RS_LIB")
    if path is None:
        root = Path(__file__).resolve().parents[2]
        names = {"win32": "radius_rs.dll", "darwin": "libradius_rs.dylib"}.get(
            sys.platform, "libradius_rs.so"
        )
        candidate = root / "target" / "release" / names
        if not candidate.exists():
            raise SystemExit(
                f"cannot find {candidate}; run `cargo build --release` or pass --lib PATH"
            )
        path = str(candidate)

    lib = ctypes.CDLL(path)
    f32p = ctypes.POINTER(ctypes.c_float)

    lib.radius_version.restype = ctypes.c_char_p
    lib.radius_version.argtypes = []

    lib.rx_td_init.restype = ctypes.c_void_p
    lib.rx_td_init.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.rx_td_free.restype = None
    lib.rx_td_free.argtypes = [ctypes.c_void_p]
    lib.rx_td_set_ratio.restype = None
    lib.rx_td_set_ratio.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
    lib.rx_td_render.restype = ctypes.c_int64
    lib.rx_td_render.argtypes = [
        ctypes.c_void_p, f32p, ctypes.c_int64, f32p, ctypes.c_int64
    ]
    lib.rx_td_geometry.restype = ctypes.c_int
    lib.rx_td_geometry.argtypes = [
        ctypes.c_uint32, ctypes.c_int, ctypes.c_int, ctypes.POINTER(RxTdGeometry)
    ]

    lib.rx_vc_init.restype = ctypes.c_void_p
    lib.rx_vc_init.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_int]
    lib.rx_vc_free.restype = None
    lib.rx_vc_free.argtypes = [ctypes.c_void_p]
    lib.rx_vc_set_ratio.restype = None
    lib.rx_vc_set_ratio.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
    lib.rx_vc_render.restype = ctypes.c_int64
    lib.rx_vc_render.argtypes = [
        ctypes.c_void_p, f32p, ctypes.c_int64, f32p, ctypes.c_int64
    ]
    return lib


def make_signal(frames: int = FRAMES, nch: int = NCH, rate: int = SR) -> np.ndarray:
    """Same generator as include/examples/example.c: a sine per channel, a slow
    tremolo, a touch of noise. Interleaved float32, shape (frames * nch,).

    The noise differs between the two languages (numpy's PCG64 vs the C file's LCG),
    so the two examples print very close numbers rather than identical ones: the
    pitch tracker keys off the tonal part, and the granule count follows it.
    """
    t = np.arange(frames, dtype=np.float64) / rate
    env = 0.6 + 0.4 * np.sin(2.0 * np.pi * 0.7 * t)
    rng = np.random.default_rng(0x12345678)
    noise = (rng.random(frames) - 0.5) * 0.02
    freqs = [440.0, 659.255]  # A4 / E5
    x = np.empty((frames, nch), dtype=np.float32)
    for c in range(nch):
        x[:, c] = 0.35 * env * np.sin(2.0 * np.pi * freqs[c % len(freqs)] * t) + noise
    return x.reshape(-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lib", default=None, help="path to the shared library")
    args = ap.parse_args()

    lib = load_library(args.lib)
    print(f"radius-rs {lib.radius_version().decode()}")

    x = make_signal()
    frames = x.size // NCH
    print(f"in : {frames} frames, {NCH} ch, peak {np.abs(x).max():.4f}")

    # A contiguous float32 array is exactly what the ABI expects. The engines are
    # duration preserving, so input length plus a granule of slack is enough.
    src = np.ascontiguousarray(x, dtype=np.float32)
    out = np.zeros(frames * NCH + 65536, dtype=np.float32)

    # ---- time-domain engine ------------------------------------------------
    geom = RxTdGeometry()
    if lib.rx_td_geometry(SR, 37, 0, ctypes.byref(geom)) == 0:
        print(f"td : hop {geom.hop}, pitch FFT {geom.pitch_n}, window {geom.pitch_l1}")
    st = lib.rx_td_init(SR, 37, 0, NCH)
    if not st:
        raise SystemExit("rx_td_init returned NULL")
    try:
        lib.rx_td_set_ratio(st, SEMITONES, 100.0)
        n_td = lib.rx_td_render(st, src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                               frames, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                               frames + 65536)
    finally:
        lib.rx_td_free(st)  # a fresh handle per render: the renderers are stateful
    if n_td < 0:
        raise SystemExit("rx_td_render failed")
    print(f"td : {SEMITONES:+g} semitones -> {n_td} frames, peak {np.abs(out[: n_td * NCH]).max():.4f}")

    # ---- phase vocoder -----------------------------------------------------
    out[:] = 0.0
    st = lib.rx_vc_init(SR, NCH, 2)
    if not st:
        raise SystemExit("rx_vc_init returned NULL")
    try:
        lib.rx_vc_set_ratio(st, -SEMITONES, 100.0)
        n_vc = lib.rx_vc_render(st, src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                               frames, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                               frames + 65536)
    finally:
        lib.rx_vc_free(st)
    if n_vc < 0:
        raise SystemExit("rx_vc_render failed")
    print(f"vc : {-SEMITONES:+g} semitones -> {n_vc} frames, peak {np.abs(out[: n_vc * NCH]).max():.4f}")

    # ---- bad arguments are reported, not undefined -------------------------
    if lib.rx_vc_init(22050, NCH, 2):
        raise SystemExit("expected 22050 Hz to be rejected by the vocoder")
    print("vc : 22050 Hz correctly rejected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
