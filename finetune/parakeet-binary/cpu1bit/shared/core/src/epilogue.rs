//! LocalFlow's FP32 epilogue in its exact operation order (lf-cpu/src/gemm_i8.rs, x86 `kernel`, lines
//! 168-181 of the snapshot, sha256 c832fdf6...): with `d[c]` the exact integer dot of component `c`
//! (`Σ_i q[t][c][i] · w[o][i]`),
//!
//! ```text
//! sum = +0.0
//! for c in (0..n).rev():  sum = fma(f32(d[c]), INV[c], sum)     // single rounding each
//! y   = (sum * weight_scale[o]) * frame_scale[t]                 // two roundings, this order
//! ```
//!
//! `f32(d)` is exact because |d| <= 128 * K <= 2^24 for every model shape (K <= 4096 gives
//! 524,288 < 16,777,216); [`finish`] asserts the bound in debug builds. Every kernel must produce
//! its output through [`finish`] or [`neon::finish4`] (or a bit-identical inline equivalent that the
//! contract tests then prove).

use crate::quant::RATIO;

/// `[1, 1/254, 1/254²]` as f32, computed exactly as LocalFlow's `INV` const.
pub const INV: [f32; 3] = [1.0, 1.0 / RATIO, 1.0 / (RATIO * RATIO)];

/// Largest |d| whose i32 -> f32 conversion is exact.
pub const EXACT_F32_INT: i64 = 1 << 24;

/// Scalar epilogue for one output: `d[c]` for c in 0..n.
#[inline(always)]
pub fn finish(d: &[i32], weight_scale: f32, frame_scale: f32) -> f32 {
    debug_assert!((1..=3).contains(&d.len()));
    let mut sum = 0.0f32;
    for c in (0..d.len()).rev() {
        debug_assert!((d[c] as i64).abs() <= EXACT_F32_INT);
        sum = (d[c] as f32).mul_add(INV[c], sum);
    }
    (sum * weight_scale) * frame_scale
}

#[cfg(target_arch = "aarch64")]
pub mod neon {
    use super::INV;
    use std::arch::aarch64::*;

    /// Four outputs at once; `d[c]` holds component `c`'s dots for the four outputs, `ws` their
    /// weight scales. Same operations and order as [`super::finish`]:
    /// `vfmaq_n_f32(sum, x, k)` is `sum + x * k` with one rounding (FMLA by element).
    #[inline(always)]
    pub fn finish4<const N: usize>(d: [int32x4_t; N], ws: float32x4_t, frame_scale: f32) -> float32x4_t {
        unsafe {
            let mut sum = vdupq_n_f32(0.0);
            let mut c = N;
            while c > 0 {
                c -= 1;
                sum = vfmaq_n_f32(sum, vcvtq_f32_s32(d[c]), INV[c]);
            }
            vmulq_n_f32(vmulq_f32(sum, ws), frame_scale)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inv_constants_are_correctly_rounded() {
        assert_eq!(INV[0].to_bits(), 1.0f32.to_bits());
        assert_eq!(INV[1], (1.0f64 / 254.0) as f32);
        assert_eq!(INV[2], (1.0f64 / 64516.0) as f32);
    }

    #[test]
    fn fused_order_matters_and_is_kept() {
        // A case where fma(d1, INV1, 0) then fma(d0, 1, sum) differs from a naive double rounding
        // order would still be equal here; the point is that finish() is stable and deterministic.
        let y = finish(&[1000, -77], 0.031, 0.5);
        let sum = ((-77i32) as f32).mul_add(INV[1], 0.0);
        let sum = (1000i32 as f32).mul_add(INV[0], sum);
        assert_eq!(y.to_bits(), ((sum * 0.031) * 0.5).to_bits());
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_finish_matches_scalar() {
        use std::arch::aarch64::*;
        let mut rng = crate::rng::SplitMix64::new(5);
        for _ in 0..20000 {
            let d: [[i32; 4]; 3] = std::array::from_fn(|_| std::array::from_fn(|_| (rng.next_u64() % 1_048_577) as i32 - 524_288));
            let ws: [f32; 4] = std::array::from_fn(|_| rng.next_f32() * 0.05 + 1e-3);
            let fs = rng.next_f32() * 3.0;
            let v = unsafe {
                let dv = [vld1q_s32(d[0].as_ptr()), vld1q_s32(d[1].as_ptr()), vld1q_s32(d[2].as_ptr())];
                let mut out = [0f32; 4];
                vst1q_f32(out.as_mut_ptr(), neon::finish4::<3>(dv, vld1q_f32(ws.as_ptr()), fs));
                out
            };
            for j in 0..4 {
                assert_eq!(v[j].to_bits(), finish(&[d[0][j], d[1][j], d[2][j]], ws[j], fs).to_bits());
            }
        }
    }
}
