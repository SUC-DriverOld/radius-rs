//! `pyradius.fft` / `libradius` FFT kernels.
//!
//! Two transforms, both scalar-op-exact against the C engine:
//!
//! * [`Fft::cfft`] — in-place radix-2 decimation-in-time complex FFT, unscaled.
//!   The butterfly keeps every intermediate in `f32` and performs each mul/add
//!   separately, which is what `-ffp-contract=off` compiles the C to.
//! * [`Fft::fwd`] — real `N`-point transform packed into a `N+2` "cart" array
//!   (`[0]=DC`, `[1]=0`, `[2k]=re[k]`, `[2k+1]=im[k]`, `[N]=Nyquist`, `[N+1]=0`).
//! * [`Fft::inv`] — hermitian expansion, complex inverse, `1/N` scale.
//!
//! Twiddles are `(f32)cos/sin(k * (-2*pi) / n)` computed in `f64` from an
//! integer `k`, exactly like the C table builder.

use crate::consts::NEG_TWO_PI_F64;

#[derive(Debug)]
pub struct Fft {
    n: usize,
    /// Bit-reversal permutation.
    rev: Vec<u32>,
    /// Cosine twiddles `W[k].re` for `k < n/2`.
    wr: Vec<f32>,
    /// Sine twiddles `W[k].im` for `k < n/2`.
    wi: Vec<f32>,
    /// Per-stage twiddle strides: `(ln, half, step)`.
    stages: Vec<(usize, usize, usize)>,
    /// Scratch for the out-of-place bit-reversal gather.
    bre: Vec<f32>,
    bim: Vec<f32>,
}

impl Fft {
    pub fn new(n: usize) -> Self {
        assert!(n.is_power_of_two(), "FFT size must be a power of two: {n}");
        let half_n = n >> 1;
        let mut wr = vec![0.0f32; half_n];
        let mut wi = vec![0.0f32; half_n];
        for k in 0..half_n {
            let ang = (k as f64) * NEG_TWO_PI_F64 / (n as f64);
            wr[k] = ang.cos() as f32;
            wi[k] = ang.sin() as f32;
        }
        let bits = n.trailing_zeros();
        let mut rev = vec![0u32; n];
        for (i, slot) in rev.iter_mut().enumerate() {
            *slot = (i as u32).reverse_bits() >> (32 - bits);
        }
        let mut stages = Vec::new();
        let mut ln = 2usize;
        while ln <= n {
            stages.push((ln, ln >> 1, n / ln));
            ln <<= 1;
        }
        Self {
            n,
            rev,
            wr,
            wi,
            stages,
            bre: vec![0.0; n],
            bim: vec![0.0; n],
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

    /// In-place complex FFT, `re`/`im` of length `n`. Unscaled in both
    /// directions (the caller applies `1/N` for the inverse).
    pub fn process_with_scratch(&mut self, re: &mut [f32], im: &mut [f32], inverse: bool) {
        self.cfft(re, im, inverse);
    }

    /// Scratch length needed by [`Fft::process_with_scratch`]; the radix-2
    /// kernel keeps its own buffers, so it needs none.
    #[inline]
    pub fn get_inplace_scratch_len(&self) -> usize {
        0
    }

    /// In-place complex FFT, `re`/`im` of length `n`.
    pub fn cfft(&mut self, re: &mut [f32], im: &mut [f32], inverse: bool) {
        let n = self.n;
        debug_assert_eq!(re.len(), n);
        debug_assert_eq!(im.len(), n);
        for i in 0..n {
            let j = self.rev[i] as usize;
            self.bre[i] = re[j];
            self.bim[i] = im[j];
        }
        re.copy_from_slice(&self.bre);
        im.copy_from_slice(&self.bim);

        let sign = if inverse { -1.0f32 } else { 1.0f32 };
        for si in 0..self.stages.len() {
            let (ln, half, step) = self.stages[si];
            let mut base = 0usize;
            while base < n {
                for j in 0..half {
                    let t = j * step;
                    let c = self.wr[t];
                    let s = sign * self.wi[t];
                    let xr = re[base + j + half];
                    let xi = im[base + j + half];
                    // tr = xr*c - xi*s ; ti = xr*s + xi*c (separate ops)
                    let tr = xr * c - xi * s;
                    let ti = xr * s + xi * c;
                    let vl = re[base + j];
                    let ul = im[base + j];
                    re[base + j] = vl + tr;
                    re[base + j + half] = vl - tr;
                    im[base + j] = ul + ti;
                    im[base + j + half] = ul - ti;
                }
                base += ln;
            }
        }
    }

    /// `rx_fft_fwd` over a logical buffer shorter than the transform: everything
    /// past `used` is zero (the C pitch front-end leaves its scratch zeroed).
    pub fn fwd_split(&mut self, src: &[f32], used: usize) -> Vec<f32> {
        let n = self.n;
        let mut re = vec![0.0f32; n];
        re[..used].copy_from_slice(&src[..used]);
        let mut im = vec![0.0f32; n];
        self.cfft(&mut re, &mut im, false);
        let m = n >> 1;
        let mut dst = vec![0.0f32; n + 2];
        dst[0] = re[0];
        dst[1] = 0.0;
        for k in 1..m {
            dst[2 * k] = re[k];
            dst[2 * k + 1] = im[k];
        }
        dst[n] = re[m];
        dst[n + 1] = 0.0;
        dst
    }

    /// `rx_fft_fwd`: real `N` -> cart `[N+2]`.
    pub fn fwd(&mut self, src: &[f32]) -> Vec<f32> {
        let n = self.n;
        assert_eq!(src.len(), n);
        let mut re = src.to_vec();
        let mut im = vec![0.0f32; n];
        self.cfft(&mut re, &mut im, false);
        let m = n >> 1;
        let mut dst = vec![0.0f32; n + 2];
        dst[0] = re[0];
        dst[1] = 0.0;
        for k in 1..m {
            dst[2 * k] = re[k];
            dst[2 * k + 1] = im[k];
        }
        dst[n] = re[m];
        dst[n + 1] = 0.0;
        dst
    }

    /// `rx_fft_inv`: cart `[N+2]` -> real `[N]`, scaled by `1/N`.
    pub fn inv(&mut self, cart: &[f32]) -> Vec<f32> {
        let n = self.n;
        assert_eq!(cart.len(), n + 2);
        let m = n >> 1;
        let mut re = vec![0.0f32; n];
        let mut im = vec![0.0f32; n];
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
        let s = 1.0f32 / (n as f32);
        for v in re.iter_mut() {
            *v *= s;
        }
        re
    }
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
        let mut plan = Fft::new(n);
        let cart = plan.fwd(&x);
        let y = plan.inv(&cart);
        let max_diff = x
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "n={n} roundtrip max_diff={max_diff}");
    }

    #[test]
    fn fft_round_trip_sizes() {
        for n in [128usize, 256, 1024, 2048, 4096, 8192] {
            round_trip(n);
        }
    }

    /// The real transform must match a direct float64 DFT of the same input.
    ///
    /// This is the test that catches a broken kernel: the round-trip test above
    /// can pass with a consistently wrong transform, because the inverse would
    /// undo the same mistake.
    #[test]
    fn fwd_matches_a_direct_dft() {
        for n in [16usize, 64, 256, 1024] {
            let mut x = vec![0.0f32; n];
            for (i, v) in x.iter_mut().enumerate() {
                *v = ((i as f64 * 0.7).sin() * 0.5 + (i as f64 * 0.13).cos() * 0.25) as f32;
            }
            let mut plan = Fft::new(n);
            let cart = plan.fwd(&x);
            let peak = x.iter().fold(0.0f64, |a, b| a.max(b.abs() as f64));
            let mut worst = 0.0f64;
            for k in 0..=n / 2 {
                // direct DFT bin k (float64, so it is the accurate answer)
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for (i, &v) in x.iter().enumerate() {
                    let ang = -2.0 * std::f64::consts::PI * (k as f64) * (i as f64) / n as f64;
                    re += v as f64 * ang.cos();
                    im += v as f64 * ang.sin();
                }
                let got_re = cart[2 * k] as f64;
                let got_im = cart[2 * k + 1] as f64;
                worst = worst
                    .max((got_re - re).abs() / (peak * n as f64))
                    .max((got_im - im).abs() / (peak * n as f64));
            }
            assert!(
                worst < 1e-5,
                "n={n}: forward transform deviates from a direct DFT by {worst:e} \
                 (relative to the spectrum peak)"
            );
        }
    }

    #[test]
    fn cart_layout_dc_and_nyquist() {
        let n = 256usize;
        let mut x = vec![0.0f32; n];
        x[0] = 1.0;
        let mut plan = Fft::new(n);
        let cart = plan.fwd(&x);
        assert!((cart[0] - 1.0).abs() < 1e-6);
        assert!((cart[n] - 1.0).abs() < 1e-6);
        assert_eq!(cart[1], 0.0);
        assert_eq!(cart[n + 1], 0.0);
    }
}
