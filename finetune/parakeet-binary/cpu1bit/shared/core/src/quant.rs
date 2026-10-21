//! LocalFlow's per-frame residual INT8 activation quantization, reimplemented (no LocalFlow file is
//! copied). Source of the semantics: the measured x86 snapshot
//! `/mnt/hd/wilderness-labs-stt/parakeet-ternary/cpu-bitplanes/20261008T121546/snapshot/crates/lf-cpu/src/quant.rs`
//! (sha256 b18d7b97c27d8405ef72c0cd52f283cfe1f5f16a93be1567f767a1413526c01c, revision 3bbbb0c).
//!
//! For a frame `x` with `m = max|x|`, `s = m / 127`, `inv = 127 / m`, and component weights
//! `wt = [1, 254, 254²]`, component `c` of input `i` is
//! `q = clamp(round_ties_even(r * (inv * wt)), -127, 127)` followed by `r = fma(-q, s / wt, r)`,
//! starting from `r = x[i]` (all in f32). A frame is all-zero with scale 0 when `m == 0` or
//! `(127 / m) * wt[n - 1]` is not finite; any NaN or infinity rejects the whole call.
//!
//! Storage differs from LocalFlow only in representation: LocalFlow stores `u = q + 128` as u8 for
//! VPDPBUSD; here the parts are stored as s8 `q` for SDOT (the plan's "no +128 offset is needed").
//! `u = (q as u8) ^ 0x80` exactly, see [`QuantizedActs::u_bytes`]; golden vectors compare `u`.
//! Layout `[rows][components][cols]`, as LocalFlow.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::aligned::AlignedVec;
use crate::pool::ThreadPool;
use crate::{SendPtr, split_range};

pub const RATIO: f32 = 254.0;
pub const MAX_COMPONENTS: usize = 3;
pub const WEIGHTS: [f32; MAX_COMPONENTS] = [1.0, RATIO, RATIO * RATIO];

/// An activation was NaN or infinite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NonFiniteActivation;

impl std::fmt::Display for NonFiniteActivation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("non-finite activation")
    }
}

impl std::error::Error for NonFiniteActivation {}

/// Quantized frames: `q[t][c][i]` (s8, the contract domain is all of [-128, 127]; the quantizer
/// itself produces [-127, 127]) and the per-frame scale `s`.
pub struct QuantizedActs {
    components: usize,
    rows: usize,
    cols: usize,
    q: AlignedVec<i8>,
    scale: Vec<f32>,
}

impl QuantizedActs {
    pub fn with_capacity(max_rows: usize, cols: usize, components: usize) -> Self {
        assert!((1..=MAX_COMPONENTS).contains(&components));
        assert!(cols > 0);
        QuantizedActs {
            components,
            rows: 0,
            cols,
            q: AlignedVec::zeroed(max_rows.checked_mul(components * cols).expect("size")),
            scale: vec![0.0; max_rows],
        }
    }

    /// Test and contract constructor: arbitrary parts (any s8, including -128) and scales.
    pub fn from_parts(rows: usize, cols: usize, components: usize, q: &[i8], scale: &[f32]) -> Self {
        assert_eq!(q.len(), rows * components * cols);
        assert_eq!(scale.len(), rows);
        let mut out = QuantizedActs::with_capacity(rows, cols, components);
        out.q.as_mut_slice()[..q.len()].copy_from_slice(q);
        out.scale[..rows].copy_from_slice(scale);
        out.rows = rows;
        out
    }

    pub fn components(&self) -> usize {
        self.components
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// `[rows][components][cols]` s8 parts. The base pointer is 128-byte aligned; row `t`,
    /// component `c` starts at `(t * components + c) * cols`.
    pub fn q(&self) -> &[i8] {
        &self.q.as_slice()[..self.rows * self.components * self.cols]
    }

    /// Part `c` of frame `t`, `cols` values.
    pub fn part(&self, t: usize, c: usize) -> &[i8] {
        assert!(t < self.rows && c < self.components);
        &self.q()[(t * self.components + c) * self.cols..][..self.cols]
    }

    /// Per-frame scale `s` (0 for zero and too-small frames).
    /// Heap bytes held (the storage audit's harness accounting).
    pub fn heap_bytes(&self) -> usize {
        self.q.heap_bytes() + self.scale.capacity() * std::mem::size_of::<f32>()
    }
    pub fn scale(&self) -> &[f32] {
        &self.scale[..self.rows]
    }

    /// LocalFlow's stored bytes `u = q + 128`, `[rows][components][cols]`.
    pub fn u_bytes(&self) -> Vec<u8> {
        self.q().iter().map(|&v| (v as u8) ^ 0x80).collect()
    }

    /// Quantizes `rows` frames of `x` (`[rows][cols]`) on the pool, growing the buffers if needed.
    /// Fails without a usable result (rows() == 0) if any activation is NaN or infinite.
    pub fn quantize(&mut self, pool: &ThreadPool, x: &[f32], rows: usize) -> Result<(), NonFiniteActivation> {
        let (k, n) = (self.cols, self.components);
        assert!(x.len() >= rows.checked_mul(k).expect("size"));
        if self.scale.len() < rows {
            self.scale.resize(rows, 0.0);
            let mut q = AlignedVec::zeroed(rows * n * k);
            q.as_mut_slice()[..self.q.len()].copy_from_slice(self.q.as_slice());
            self.q = q;
        }
        self.rows = 0;
        let q = SendPtr(self.q.as_mut_ptr());
        let s = SendPtr(self.scale.as_mut_ptr());
        let threads = pool.threads();
        let bad = AtomicBool::new(false);
        pool.run(&|idx| {
            for t in split_range(rows, threads, idx) {
                let row = &x[t * k..(t + 1) * k];
                // Each thread writes disjoint frames; both buffers hold at least `rows`.
                let out = unsafe { std::slice::from_raw_parts_mut(q.get().add(t * n * k), n * k) };
                let scale = unsafe { &mut *s.get().add(t) };
                if !quantize_row(row, n, out, scale) {
                    bad.store(true, Ordering::Relaxed);
                }
            }
        });
        if bad.load(Ordering::Relaxed) {
            return Err(NonFiniteActivation);
        }
        self.rows = rows;
        Ok(())
    }
}

/// LocalFlow's `too_small`: frames whose finest step would overflow f32 are treated as zero.
#[inline]
fn too_small(m: f32, n: usize) -> bool {
    m == 0.0 || !((127.0 / m) * WEIGHTS[n - 1]).is_finite()
}

/// The production row quantizer: NEON on aarch64, the scalar reference elsewhere. Bit-identical
/// to [`quantize_row_scalar`] for every input (tested, and checked against x86 golden vectors).
#[inline]
pub fn quantize_row(x: &[f32], n: usize, out: &mut [i8], scale: &mut f32) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::quantize_row(x, n, out, scale) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        quantize_row_scalar(x, n, out, scale)
    }
}

/// Scalar reference, operation for operation LocalFlow's `quantize_row_scalar`.
/// Returns false if `x` holds a NaN or infinity (the output is then unspecified).
pub fn quantize_row_scalar(x: &[f32], n: usize, out: &mut [i8], scale: &mut f32) -> bool {
    let k = x.len();
    assert!((1..=MAX_COMPONENTS).contains(&n) && out.len() >= n * k);
    if x.iter().any(|v| !v.is_finite()) {
        return false;
    }
    let m = x.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    if too_small(m, n) {
        out[..n * k].fill(0);
        *scale = 0.0;
        return true;
    }
    let (inv, s) = (127.0 / m, m / 127.0);
    for (i, &v) in x.iter().enumerate() {
        let mut r = v;
        for (c, &wt) in WEIGHTS.iter().enumerate().take(n) {
            let q = (r * (inv * wt)).round_ties_even().clamp(-127.0, 127.0);
            r = (-q).mul_add(s / wt, r);
            out[c * k + i] = q as i8;
        }
    }
    *scale = s;
    true
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{WEIGHTS, too_small};
    use std::arch::aarch64::*;

    /// Four frames' worth of lanes at a time; the tail (k % 4) runs the scalar element code. The
    /// maximum of non-negative, non-NaN values is order independent, and every element uses the
    /// same f32 operations as the scalar code (`frintn`, float clamp, fused `r - q*step`).
    #[target_feature(enable = "neon")]
    pub unsafe fn quantize_row(x: &[f32], n: usize, out: &mut [i8], scale: &mut f32) -> bool {
        let k = x.len();
        assert!((1..=3).contains(&n) && out.len() >= n * k);
        let p = x.as_ptr();
        let k4 = k & !3;
        let inf = vdupq_n_f32(f32::INFINITY);
        let mut mx = vdupq_n_f32(0.0);
        let mut ok = vdupq_n_u32(u32::MAX);
        let mut i = 0;
        while i < k4 {
            let a = vabsq_f32(unsafe { vld1q_f32(p.add(i)) });
            ok = vandq_u32(ok, vcltq_f32(a, inf)); // false for +inf and NaN
            mx = vmaxq_f32(mx, a);
            i += 4;
        }
        if vminvq_u32(ok) == 0 {
            return false;
        }
        let mut m = vmaxvq_f32(mx);
        for &v in &x[k4..] {
            if !v.is_finite() {
                return false;
            }
            m = m.max(v.abs());
        }
        if too_small(m, n) {
            out[..n * k].fill(0);
            *scale = 0.0;
            return true;
        }
        let (inv, s) = (127.0f32 / m, m / 127.0f32);
        let lo = vdupq_n_f32(-127.0);
        let hi = vdupq_n_f32(127.0);
        let o = out.as_mut_ptr();
        let mut i = 0;
        while i < k4 {
            let mut r = unsafe { vld1q_f32(p.add(i)) };
            for (c, &wt) in WEIGHTS.iter().enumerate().take(n) {
                let mult = vdupq_n_f32(inv * wt);
                let step = vdupq_n_f32(s / wt);
                let q = vminq_f32(vmaxq_f32(vrndnq_f32(vmulq_f32(r, mult)), lo), hi);
                r = vfmsq_f32(r, q, step); // r - q * step, fused
                let qi = vcvtq_s32_f32(q); // exact: q is integral in [-127, 127]
                let h = vmovn_s32(qi);
                let b = vmovn_s16(vcombine_s16(h, h));
                // Store the low 4 bytes.
                let lane = vget_lane_u32::<0>(vreinterpret_u32_s8(b));
                unsafe { (o.add(c * k + i) as *mut u32).write_unaligned(lane) };
            }
            i += 4;
        }
        for i in k4..k {
            let mut r = x[i];
            for (c, &wt) in WEIGHTS.iter().enumerate().take(n) {
                let q = (r * (inv * wt)).round_ties_even().clamp(-127.0, 127.0);
                r = (-q).mul_add(s / wt, r);
                out[c * k + i] = q as i8;
            }
        }
        *scale = s;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    fn frame(k: usize, seed: u64) -> Vec<f32> {
        let mut rng = SplitMix64::new(seed);
        (0..k).map(|i| rng.next_gaussian() * if i % 37 == 0 { 20.0 } else { 1.0 }).collect()
    }

    #[test]
    fn production_row_matches_scalar_bit_for_bit() {
        for n in 1..=3 {
            for k in [1usize, 3, 4, 5, 16, 17, 64, 1023, 1024, 4096] {
                for seed in 0..4u64 {
                    let mut x = frame(k, seed * 31 + k as u64);
                    if seed == 3 {
                        // Ties: values whose first-component product is exactly .5.
                        let m = x.iter().fold(0.0f32, |a, v| a.max(v.abs()));
                        for (j, v) in x.iter_mut().enumerate().skip(1).step_by(3) {
                            *v = (j as f32 % 7.0 + 0.5) * (m / 127.0);
                        }
                    }
                    let (mut a, mut b) = (vec![0i8; n * k], vec![0i8; n * k]);
                    let (mut sa, mut sb) = (0.0f32, 0.0f32);
                    assert!(quantize_row_scalar(&x, n, &mut a, &mut sa));
                    assert!(quantize_row(&x, n, &mut b, &mut sb));
                    assert_eq!(sa.to_bits(), sb.to_bits());
                    assert_eq!(a, b, "n={n} k={k} seed={seed}");
                }
            }
        }
    }

    #[test]
    fn zero_tiny_and_non_finite_frames() {
        let pool = ThreadPool::new(1, crate::pool::Qos::Inherit);
        let mut qa = QuantizedActs::with_capacity(0, 33, 3);
        qa.quantize(&pool, &[0.0; 33], 1).unwrap();
        assert_eq!(qa.scale()[0], 0.0);
        assert!(qa.q().iter().all(|&b| b == 0));
        qa.quantize(&pool, &[1e-35; 33], 1).unwrap();
        assert_eq!(qa.scale()[0], 0.0);
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for pos in [0, 17, 32] {
                let mut x = [1.0f32; 33];
                x[pos] = bad;
                assert_eq!(qa.quantize(&pool, &x, 1), Err(NonFiniteActivation));
                assert_eq!(qa.rows(), 0);
                let (mut out, mut s) = (vec![0i8; 99], 0.0);
                assert!(!quantize_row(&x, 3, &mut out, &mut s));
            }
        }
    }

    #[test]
    fn u_bytes_are_q_plus_128() {
        let q: Vec<i8> = (-128..=127).collect();
        let qa = QuantizedActs::from_parts(1, 256, 1, &q, &[1.0]);
        let u = qa.u_bytes();
        for (a, b) in q.iter().zip(&u) {
            assert_eq!(*a as i32 + 128, *b as i32);
        }
    }
}
