# Engine data provenance

The crate used to compile in six constant tables (crossover FIR banks, window
banks, synthesis windows at 44.1/48 kHz) parsed from the reference C engine's
headers. **They are gone.** Static analysis of `dvaaudiodsp.dll` established
the original engine ships none of them — everything is computed at init from
the sample rate — and the crate now does the same, in
`src/vocoder/rate_gen.rs`. This file records where every number in the
generated data comes from.

## What the DLL does (decompiled)

Byte-signature searches over the whole `dvaaudiodsp.dll` image find no FIR
taps (neither `f32` nor `f64`-encoded), no window banks, no synthesis
windows, and no 44100/48000/32000 literals anywhere in the Radius region to
dispatch a two-bank design on (full reasoning in
[`RE-dvaaudiodsp-radius-tempo.md`](RE-dvaaudiodsp-radius-tempo.md) §12). The
rate arrives as data (the live renderer carries it at `+0x60`/`+0x108`), and
the engine computes everything at init.

The crossover's design chain is now **fully decompiled**, not fitted:

`sub_18015E560` (init) → `sub_1800FA4B0` (spec builder, called with
`{150.0f, 800.0f, 4500.0f}`, `sr`, `0.1`, `9.0`) →
`dsp::FIR::DesignFIR@0x1800348D0` (hermitian c2r inverse FFT, 1025 bins →
2048 taps) → `dsp::AudioProcessor::ApplyKaiserWindow@0x1801634D0`
(length-2047 Kaiser, β = 9.0, mode 2 = no post-normalisation).

The exact law, per band `b ∈ 0..3` with edges `e = {150, 800, 4500}` and
`nyq = sr/2`:

* ramp A (lower edge) centred at `ln(e[b−1])`, ramp B (upper edge) at
  `ln(e[b])`; band 3's ramp B is degenerate at `ln(nyq)` so the Nyquist bin
  stays flat 1;
* each ramp's half-width is `h = 0.9 · min(c − L, U − c)` where `L`/`U` are
  the log-geometric midpoints `(ln a + ln b)·0.5` to the neighbouring
  crossovers (outer flanks: `ln(e[0]·0.5)` and `geo-mid(e[2], nyq)`);
* bin `k` at frequency `f = k·sr/2048` (`v51 = ln f` as `f32`, `k = 0` →
  `−10.0`): 1 in the flat region, `0.5 − 0.5·cos(π·(v51−lo)/(hi−lo))` on a
  rising ramp, `0.5 + 0.5·cos(...)` on a falling ramp (the phase in `f64`,
  cast to `f32`), 0 outside; the boundary comparisons are `f32`;
* the impulse response is windowed by the length-2047 Kaiser (`w =
  I₀(9·√(1−r²))/I₀(9)` in `f64`, one cast to `f32`) with **no
  post-normalisation** — the DC gain 1.000 at 44.1/48 kHz is the law's own.

This reproduces both retired banks to **max 2.9·10⁻⁷ relative per tap**
(2–3 `f32` ulp of the peak; the residue is the DLL's unknown c2r FFT
operation order, not the design law). The 44.1 kHz "width anomaly" resolves
naturally: the same law with `sr = 44100` (the last flank's geometric
midpoint `geo-mid(4500, 22050)` differs from 48 kHz's) reproduces that bank
to 1.5·10⁻⁷.

## The crate's reconstruction (`rate_gen`)

| data | construction | verified against |
|---|---|---|
| crossover FIR | the decompiled law above, verbatim (`design_fir_bank`) | both retired banks: max 2.9·10⁻⁷ relative per tap, exact bin complementarity (1.2·10⁻⁶, the banks' own residue is 1.1·10⁻⁶), DC 1.000 / Nyquist −1.000 at 44.1/48 kHz |
| window bank | bands 1–3: plain Hann, **`L = 2·⌊r·⌈n_write/2⌉⌋`** with `r = {0.5, 0.3, 0.2}` (all six snapshot lengths exactly: {3592, 2154, 1436} at 48 kHz, {3300, 1980, 1320} at 44.1 kHz — no single `round(r·nw)` rule does), start `(nw−L)/2`, energy **`Σw² = 2·⌊nw/2⌋`**; band 0: two-piece, rise = `hop`, fall `⌊0.32489·nw⌋`, start `⌊0.044·nw⌋` (constants pinned to the interval that hits both snapshots under floor), peak scale constant `1.9793857` | both retired banks: every band's support exact at both rates, values ≤ 3·10⁻⁷ relative (the residue is the DLL's `cosf` polynomial vs a correctly-rounded `cos`, not the law) |
| synthesis pair | `Hann^0.7`, lengths `round({2930,1256}·n_write/7184)`, long window scaled `√(short/long)` (48 kHz convention) | the 48 kHz bank's builder formula and the 44.1 kHz bank's ratios |

## Validation after the removal

* Structural (`tests/rate_generic.rs`, `rate_gen` unit tests): band edges at
  the crossovers at 32/44.1/48 kHz, complementary sum flat on the bin grid,
  unity DC/Nyquist gains (0.9965 DC at 96 kHz is the law's own Kaiser
  leakage — the DLL's 96 kHz bank behaves identically), exact all-pass
  reconstruction of delayed input, render correctness at 16/32/96 kHz.
* Against the retired tables (48 kHz, the 29 s acceptance input, +3 semis):
  sample-level correlation rose from **0.22** (the old fitted designer) to
  **0.78** with the exact laws, envelope corr 0.88, first-5 s corr 0.98.
* Against Audition (authoritative, `tests/audition_reference.rs`): all
  metrics within noise of the tables build — vc+3 pitch +3.008 (was +3.012),
  level +0.20 dB (was +0.20), vc+3 50 % contour corr 0.822 (was 0.818).

### Why sample-level correlation will not reach 1.0

A controlled isolation experiment (engine patched to load the retired
banks from files) proved the remaining divergence is **deterministic
chaos, not design error**:

* **both** retired banks loaded → render **bit-identical** to the tables-era
  engine (corr 1.000000, max |Δ| = 0) — every other line of the engine is an
  exact port;
* retired windows + law-designed FIR (2–3 ulp tap differences **only**) →
  corr decays 0.98 → 0.76 over 25 s: the vocoder's per-granule phase
  feedback amplifies any `f32` perturbation exponentially;
* law-designed windows + retired FIR → the same end state.

Long renders therefore cannot be sample-identical unless every floating-point
operation order of the original (including its c2r FFT internals and its
`cosf` polynomial) is cloned instruction-for-instruction; the design laws
themselves are now exact.

## History

The tables lived in `src/vocoder/table_data/` (six `.rs` files of
`f32::from_bits` literals, generated by `tools/gen_table_source.py` from the
C headers) from the crate's first port until their removal. The parity corpus
against `pyradius` (bit-exact at 48 kHz) was a property of those tables and
retired with them; the acceptance tests now pin perceptual parity. The
reference headers remain recorded for provenance:

    libradius/src/ops/crossover_fir_table.h
    libradius/src/ops/crossover_fir_table_44k.h
    libradius/src/engine/vc_win_table.h
    libradius/src/engine/vc_win_table_44k.h
    libradius/src/engine/vc_synth_win_44k.h
