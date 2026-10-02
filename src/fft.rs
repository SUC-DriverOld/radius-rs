//! FFT selection: one bit-exact kernel (default) and one fast kernel (opt-in).
//!
//! The engines call [`with_plan`] / [`fwd`] / [`inv`] and read a plain `&mut Fft`;
//! which implementation `Fft` wraps is decided at *run time* per thread, so a
//! single build carries both.
//!
//! | backend | kernel | character |
//! |---|---|---|
//! | [`Backend::Radix2`] (default) | [`crate::fft_radix2`] | radix-2 DIT with the reference's twiddle table and per-operation `f32` rounding — **bit-exact** with the engine this crate reproduces |
//! | [`Backend::Rustfft`] (`--fft rustfft`) | the `rustfft` crate | mixed-radix, faster but not bit-exact |
//!
//! Why the default is not simply `rustfft`, measured on `tests/test.wav`
//! (+3 semitones, 29.03 s, 48 kHz stereo):
//!
//! | path | rustfft vs radix-2 | time |
//! |---|---|---|
//! | TD (`-m td`) | **bit-identical** (max\|d\| = 0) | 0.55 s vs 1.23 s (2.2x faster) |
//! | vocoder (`-m vc`) | corr 0.9944, max\|d\| 0.50, SNR 19.5 dB, 0.02 % of samples equal | 33.8 s vs 45.4 s (1.3x faster) |
//!
//! The TD path tolerates the swap because its pitch front-end only thresholds a
//! whitened autocorrelation peak, and on this corpus every decision lands the
//! same way. The vocoder cannot: its peak search, phase unwrapping and pitch
//! coherence all branch on magnitudes, so the last-bit differences flip discrete
//! decisions and the output decorrelates to 0.994 — far outside the reference's
//! own acceptance floor of `corr > 0.9999`. That is why the fast kernel is
//! opt-in and never the default.
//!
//! Selecting a backend:
//!
//! * per thread, from code: [`set_backend`] / [`backend`];
//! * process-wide, from the environment: `RADIUS_FFT=rustfft` (or `=radix2`,
//!   the default) — read once per thread on first use.

use std::cell::Cell;

use crate::fft_radix2;

use rustfft::num_complex::Complex32;
use rustfft::{Fft as RustFft, FftPlanner};

/// Which FFT implementation to use.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Backend {
    /// The crate's own radix-2 kernel — bit-exact with the reference engine.
    Radix2,
    /// The `rustfft` crate — faster, but not bit-exact.
    Rustfft,
}

impl Backend {
    /// Parse a backend name (`radix2`, `rustfft`, case-insensitive).
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "radix2" | "radix-2" | "exact" | "default" => Some(Backend::Radix2),
            "rustfft" | "fast" => Some(Backend::Rustfft),
            _ => None,
        }
    }

    /// Backend name as used by `RADIUS_FFT`.
    pub fn name(self) -> &'static str {
        match self {
            Backend::Radix2 => "radix2",
            Backend::Rustfft => "rustfft",
        }
    }

    /// Is this backend compiled in?
    pub fn is_available(self) -> bool {
        match self {
            Backend::Radix2 => true,
            Backend::Rustfft => true,
        }
    }
}

thread_local! {
    /// `None` means "read `RADIUS_FFT` on first use".
    static SELECTED: Cell<Option<Backend>> = const { Cell::new(None) };
}

/// Select the FFT backend for the current thread.
///
/// Select the backend for the current thread.
pub fn set_backend(b: Backend) {
    let b = if b.is_available() { b } else { Backend::Radix2 };
    SELECTED.with(|s| s.set(Some(b)));
}

/// The backend this thread is using.
pub fn backend() -> Backend {
    SELECTED.with(|s| match s.get() {
        Some(b) => b,
        None => {
            let b = std::env::var("RADIUS_FFT")
                .ok()
                .and_then(|v| Backend::parse(&v))
                .filter(|b| b.is_available())
                .unwrap_or(Backend::Radix2);
            s.set(Some(b));
            b
        }
    })
}

/// An FFT plan for one size, wrapping whichever backend is selected.
pub struct Fft {
    n: usize,
    inner: Inner,
    // Reused real/imaginary work buffers.  The vocoder executes thousands of
    // transforms per render; allocating these on every call is needlessly
    // expensive and puts constant pressure on the allocator/cache.
    work_re: Vec<f32>,
    work_im: Vec<f32>,
}

enum Inner {
    Radix2(fft_radix2::Fft),
    Rustfft {
        forward: std::sync::Arc<dyn RustFft<f32>>,
        inverse: std::sync::Arc<dyn RustFft<f32>>,
        scratch: Vec<Complex32>,
    },
}

impl Fft {
    /// Build a plan for size `n` using the thread's selected backend.
    pub fn new(n: usize) -> Self {
        assert!(n.is_power_of_two(), "FFT size must be a power of two: {n}");
        let inner = match backend() {
            Backend::Radix2 => Inner::Radix2(fft_radix2::Fft::new(n)),
            Backend::Rustfft => {
                let mut planner = FftPlanner::<f32>::new();
                let forward = planner.plan_fft_forward(n);
                let inverse = planner.plan_fft_inverse(n);
                let scratch = vec![
                    Complex32::new(0.0, 0.0);
                    forward
                        .get_inplace_scratch_len()
                        .max(inverse.get_inplace_scratch_len())
                ];
                Inner::Rustfft {
                    forward,
                    inverse,
                    scratch,
                }
            }
        };
        Self {
            n,
            inner,
            work_re: vec![0.0; n],
            work_im: vec![0.0; n],
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// `rx_fft_fwd` over a logical buffer shorter than the transform: everything
    /// past `used` is zero (the C pitch front-end leaves its scratch zeroed).
    pub fn fwd_split(&mut self, src: &[f32], used: usize) -> Vec<f32> {
        let n = self.n;
        let mut re = vec![0.0f32; n];
        re[..used].copy_from_slice(&src[..used]);
        let mut im = vec![0.0f32; n];
        self.cfft(&mut re, &mut im, false);
        pack_cart(n, &re, &im)
    }

    /// `rx_fft_fwd`: real `N` -> cart `[N+2]`.
    pub fn fwd(&mut self, src: &[f32]) -> Vec<f32> {
        let mut dst = vec![0.0f32; self.n + 2];
        self.fwd_into(src, &mut dst);
        dst
    }

    /// Forward real transform writing into caller-owned cartesian storage.
    /// The transform work arrays are retained by the plan and reused.
    pub fn fwd_into(&mut self, src: &[f32], dst: &mut [f32]) {
        let n = self.n;
        assert_eq!(src.len(), n);
        assert_eq!(dst.len(), n + 2);
        let mut re = std::mem::take(&mut self.work_re);
        let mut im = std::mem::take(&mut self.work_im);
        re.copy_from_slice(src);
        im.fill(0.0);
        self.cfft(&mut re, &mut im, false);
        pack_cart_into(n, &re, &im, dst);
        self.work_re = re;
        self.work_im = im;
    }

    /// Forward transform using the caller's real buffer as the FFT real
    /// workspace. The caller must not need `src` after this call. This avoids
    /// copying a 16k-sample frame into a second persistent buffer in the
    /// vocoder's ACS path.
    pub fn fwd_in_place(&mut self, src: &mut [f32], dst: &mut [f32]) {
        let n = self.n;
        assert_eq!(src.len(), n);
        assert_eq!(dst.len(), n + 2);
        let mut im = std::mem::take(&mut self.work_im);
        im.fill(0.0);
        self.cfft(src, &mut im, false);
        pack_cart_into(n, src, &im, dst);
        self.work_im = im;
    }

    /// `rx_fft_inv`: cart `[N+2]` -> real `[N]`, scaled by `1/N`.
    pub fn inv(&mut self, cart: &[f32]) -> Vec<f32> {
        let mut dst = vec![0.0f32; self.n];
        self.inv_into(cart, &mut dst);
        dst
    }

    /// Inverse cartesian transform writing into caller-owned real storage.
    pub fn inv_into(&mut self, cart: &[f32], dst: &mut [f32]) {
        let n = self.n;
        assert_eq!(cart.len(), n + 2);
        assert_eq!(dst.len(), n);
        let m = n >> 1;
        let mut re = std::mem::take(&mut self.work_re);
        let mut im = std::mem::take(&mut self.work_im);
        re.fill(0.0);
        im.fill(0.0);
        for k in 1..m {
            let ck = cart[2 * k];
            let dk = cart[2 * k + 1];
            let h = n - k;
            re[k] = ck;
            im[k] = dk;
            re[h] = ck;
            im[h] = -dk;
        }
        re[0] = cart[0];
        re[m] = cart[n];
        self.cfft(&mut re, &mut im, true);
        dst.copy_from_slice(&re);
        self.work_re = re;
        self.work_im = im;
    }

    /// In-place complex transform. Unscaled going forward; the inverse is scaled
    /// by `1/N`, matching the C wrapper (which divides after the transform), so
    /// `fwd` followed by `inv` is the identity.
    pub fn cfft(&mut self, re: &mut [f32], im: &mut [f32], inverse: bool) {
        match &mut self.inner {
            Inner::Radix2(plan) => {
                plan.cfft(re, im, inverse);
                if inverse {
                    let s = 1.0f32 / (self.n as f32);
                    for v in re.iter_mut() {
                        *v *= s;
                    }
                    for v in im.iter_mut() {
                        *v *= s;
                    }
                }
            }
            Inner::Rustfft {
                forward,
                inverse: inv,
                scratch,
            } => {
                let n = self.n;
                let mut buf: Vec<Complex32> =
                    (0..n).map(|i| Complex32::new(re[i], im[i])).collect();
                if inverse {
                    inv.process_with_scratch(&mut buf, scratch);
                    let s = 1.0f32 / (n as f32);
                    for v in buf.iter_mut() {
                        v.re *= s;
                        v.im *= s;
                    }
                } else {
                    forward.process_with_scratch(&mut buf, scratch);
                }
                for i in 0..n {
                    re[i] = buf[i].re;
                    im[i] = buf[i].im;
                }
            }
        }
    }
}

fn pack_cart_into(n: usize, re: &[f32], im: &[f32], dst: &mut [f32]) {
    let m = n >> 1;
    dst[0] = re[0];
    dst[1] = 0.0;
    for k in 1..m {
        dst[2 * k] = re[k];
        dst[2 * k + 1] = im[k];
    }
    dst[n] = re[m];
    dst[n + 1] = 0.0;
}

fn pack_cart(n: usize, re: &[f32], im: &[f32]) -> Vec<f32> {
    let mut dst = vec![0.0f32; n + 2];
    pack_cart_into(n, re, im, &mut dst);
    dst
}

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

thread_local! {
    /// Plans are cached per (size, backend); the key includes the backend so a
    /// switch mid-process cannot hand out a plan built for the other kernel.
    static PLANS: RefCell<HashMap<(usize, Backend), Rc<RefCell<Fft>>>> =
        RefCell::new(HashMap::new());
}

/// Run `f` with a cached plan for size `n` and the thread's current backend.
pub fn with_plan<R>(n: usize, f: impl FnOnce(&mut Fft) -> R) -> R {
    let b = backend();
    PLANS.with(|p| {
        let rc = {
            let mut map = p.borrow_mut();
            map.entry((n, b))
                .or_insert_with(|| Rc::new(RefCell::new(Fft::new(n))))
                .clone()
        };
        let mut plan = rc.borrow_mut();
        f(&mut plan)
    })
}

/// Real forward transform using the cached plan.
pub fn fwd(n: usize, src: &[f32]) -> Vec<f32> {
    with_plan(n, |p| p.fwd(src))
}

/// Real inverse transform using the cached plan.
pub fn inv(n: usize, cart: &[f32]) -> Vec<f32> {
    with_plan(n, |p| p.inv(cart))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(n: usize) {
        let mut x = vec![0.0f32; n];
        for (i, v) in x.iter_mut().enumerate() {
            let t = i as f64 / n as f64;
            *v = ((t * 6.0).sin() * 0.7 + (t * 23.0).cos() * 0.3) as f32;
        }
        let cart = fwd(n, &x);
        let y = inv(n, &cart);
        let max_diff = x
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "n={n} roundtrip max_diff={max_diff}");
    }

    #[test]
    fn round_trip_sizes_every_backend() {
        for n in [128usize, 256, 1024, 2048, 4096, 8192] {
            round_trip(n);
        }
        {
            set_backend(Backend::Rustfft);
            for n in [128usize, 256, 1024, 2048, 4096, 8192, 16384] {
                round_trip(n);
            }
            set_backend(Backend::Radix2);
        }
    }

    /// The cached wrapper must agree with a freshly built radix-2 plan.
    #[test]
    fn wrapper_matches_direct_radix2() {
        for n in [128usize, 512, 2048, 8192] {
            let mut x = vec![0.0f32; n];
            let mut seed = 12345u32;
            for v in x.iter_mut() {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *v = ((seed >> 9) as f32 / (1u32 << 23) as f32) - 0.5;
            }
            let mut direct = fft_radix2::Fft::new(n);
            let a = direct.fwd(&x);
            let a_inv = direct.inv(&a);
            let mut cached_plan = Fft::new(n);
            let b = cached_plan.fwd(&x);
            let b_inv = cached_plan.inv(&b);
            let max_f = a
                .iter()
                .zip(b.iter())
                .map(|(p, q)| (p - q).abs())
                .fold(0.0f32, f32::max);
            let max_i = a_inv
                .iter()
                .zip(b_inv.iter())
                .map(|(p, q)| (p - q).abs())
                .fold(0.0f32, f32::max);
            assert_eq!(max_f, 0.0, "wrapper fwd differs at n={n} by {max_f:e}");
            assert_eq!(max_i, 0.0, "wrapper inv differs at n={n} by {max_i:e}");
        }
    }

    #[test]
    fn cart_layout_dc_and_nyquist() {
        let n = 256usize;
        let mut x = vec![0.0f32; n];
        x[0] = 1.0;
        let cart = fwd(n, &x);
        assert!((cart[0] - 1.0).abs() < 1e-6);
        assert!((cart[n] - 1.0).abs() < 1e-6);
        assert_eq!(cart[1], 0.0);
        assert_eq!(cart[n + 1], 0.0);
    }

    #[test]
    fn backend_names_parse() {
        assert_eq!(Backend::parse("radix2"), Some(Backend::Radix2));
        assert_eq!(Backend::parse(" RustFFT "), Some(Backend::Rustfft));
        assert_eq!(Backend::parse("nope"), None);
        // selecting an unavailable backend must fall back cleanly
        set_backend(Backend::Rustfft);
        assert!(backend().is_available());
        set_backend(Backend::Radix2);
        assert_eq!(backend(), Backend::Radix2);
    }

    /// The two backends must at least be numerically close, even though they are
    /// not bit-equal. Relative to the spectrum peak the difference is a few ULPs.
    #[test]
    fn rustfft_agrees_with_radix2_within_tolerance() {
        let n = 1024usize;
        let mut x = vec![0.0f32; n];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i as f64 * 0.37).sin()) as f32;
        }
        set_backend(Backend::Radix2);
        let exact = fwd(n, &x);
        set_backend(Backend::Rustfft);
        let fast = fwd(n, &x);
        set_backend(Backend::Radix2);
        let peak = exact.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        let max_abs = exact
            .iter()
            .zip(fast.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs < 1e-3 * peak.max(1.0),
            "backends differ by {max_abs} (peak {peak})"
        );
    }

    /// End-to-end check that the backend actually reaches the transform: the
    /// engine's own pitch front-end (zero-padded real input, then three FFTs and
    /// a whitening pass) must give the same *discrete* answer on both kernels.
    #[test]
    fn pitch_front_end_decisions_are_backend_independent() {
        // Build a signal with a clear period so the ACF peak is unambiguous.
        let n = 8192usize;
        let l1 = 3330usize;
        let mut src = vec![0.0f32; l1];
        for (i, v) in src.iter_mut().enumerate() {
            let t = i as f64;
            *v = ((t * 2.0 * std::f64::consts::PI / 219.0).sin() * 0.5
                + (t * 2.0 * std::f64::consts::PI / 731.0).cos() * 0.3) as f32;
        }
        let run = |b: Backend| -> (f32, f32) {
            set_backend(b);
            let mut plan = Fft::new(n);
            let sa = plan.fwd_split(&src, l1);
            // power spectrum, then inverse -> autocorrelation, as the engine does
            let m = n >> 1;
            let mut cart = sa.clone();
            for k in 0..m {
                let re = sa[2 * k];
                let im = sa[2 * k + 1];
                cart[2 * k] = re * re + im * im;
                cart[2 * k + 1] = 0.0;
            }
            let acf = plan.inv(&cart);
            // where is the first strong peak, and what is its value?
            let mut best = (0usize, f32::MIN);
            for (i, v) in acf.iter().enumerate().take(2000).skip(1) {
                if *v > best.1 {
                    best = (i, *v);
                }
            }
            (best.0 as f32, best.1)
        };
        let a = run(Backend::Radix2);
        let b = run(Backend::Rustfft);
        set_backend(Backend::Radix2);
        assert_eq!(
            a.0, b.0,
            "the ACF peak index differs between backends ({} vs {})",
            a.0, b.0
        );
        let rel = ((a.1 - b.1).abs() / a.1.abs().max(1e-12)) as f64;
        assert!(rel < 1e-4, "ACF peak value differs by {rel:e}");
    }
}
