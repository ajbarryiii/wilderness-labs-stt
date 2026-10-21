//! V's NEON microkernel: 1-bit weights expanded in registers right before `SDOT`.
//!
//! Weight layout (`layout.rs`): chunk `c` of row group `g` is 16 bytes; byte `4 nl + kl`, bit `j`
//! holds the code of row `4g + nl` at input `32c + 4j + kl` (1 = +1). `CMTST` of that vector with
//! the constant `1 << j` in every byte (`CMLT #0` for j = 7) yields the `SDOT` operand of input quad
//! `8c + j` for the 4 rows: bytes 0xFF where the code is +1, 0x00 where it is -1, i.e. `-b` in s8.
//!
//! With `z = Σ_k (-b_k) q_k` from `SDOT` and `U = Σ_k q_k`, the exact dot is
//! `d = Σ_k w_k q_k = Σ_k (2 b_k - 1) q_k = -2 z - U` (i32; |z|, |U| <= 128 K, no overflow).
//!
//! Microtile: one group (4 rows) x R frame-parts, R accumulators v0..v(R-1). Each frame-part's
//! 16-byte activation load covers 4 input quads; the by-element `SDOT` (lane = quad) pairs it
//! with the 4 expanded weight vectors, so each expanded vector feeds R `SDOT`s. Per 32 inputs:
//! 1 code load, 8 expansion ops, 2R activation loads, 8R `SDOT`s. The loop is inline asm
//! (`build.rs`): compiled intrinsics spilled activation registers and re-materialised the masks.

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use core::arch::aarch64::*;
use shared::epilogue::neon::finish4;

/// Generated inline-asm kernels: `dot_asm::<R>(w, a, lda, chunks)` returns the R accumulators
/// `z` of one group over `chunks` 32-input chunks (zero for `chunks == 0`), frame-part `r` read at
/// `a + r * lda` (32 bytes per chunk).
pub mod asm {
    use core::arch::aarch64::*;
    include!(concat!(env!("OUT_DIR"), "/dot_asm.rs"));
}

/// `Σ_k q_k` over a row, exactly.
#[inline]
pub fn part_sum(row: &[i8]) -> i32 {
    let k = row.len();
    let p = row.as_ptr();
    unsafe {
        let ones = vdupq_n_s8(1);
        let (mut s0, mut s1) = (vdupq_n_s32(0), vdupq_n_s32(0));
        let mut i = 0;
        while i + 32 <= k {
            s0 = vdotq_s32(s0, vld1q_s8(p.add(i)), ones);
            s1 = vdotq_s32(s1, vld1q_s8(p.add(i + 16)), ones);
            i += 32;
        }
        let mut s = vaddvq_s32(vaddq_s32(s0, s1));
        for &v in &row[i..] {
            s += v as i32;
        }
        s
    }
}

/// One call's operands (see `lib.rs` `Job`).
pub struct PanelArgs {
    pub bits: *const u8,
    pub chunks: usize,
    pub scale: *const f32,
    pub a: *const i8,
    pub lda: usize,
    pub u: *const i32,
    pub fscale: *const f32,
    pub out: *mut u8,
    pub n: usize,
}

/// Frames `t0 .. t0 + F` (R = F * C frame-parts) against row groups `g0 .. g1`. One routine for
/// both outputs: `DOTS` changes only the final store.
///
/// # Safety
/// `args` must describe live buffers: `bits` with `g1` groups of `chunks` chunks, `scale` with
/// `4 g1` values, frame-parts `t0 C .. (t0 + F) C` of `a` readable for `32 chunks` bytes at stride
/// `lda`, `u`/`fscale` covering them, and `out` holding `[T][N]` f32 or `[T][C][N]` i32 with
/// `t0 + F <= T`; no other thread writes the same (frames, groups) region.
#[inline(never)]
pub unsafe fn panel<const C: usize, const F: usize, const R: usize, const DOTS: bool>(
    args: &PanelArgs,
    t0: usize,
    g0: usize,
    g1: usize,
) {
    debug_assert_eq!(R, F * C);
    unsafe {
        let a = args.a.add(t0 * C * args.lda);
        let u = args.u.add(t0 * C);
        let n = args.n;
        // U per accumulator, hoisted out of the group loop.
        let uv: [int32x4_t; R] = core::array::from_fn(|r| vdupq_n_s32(*u.add(r)));
        let fs: [f32; F] = core::array::from_fn(|m| *args.fscale.add(t0 + m));
        for g in g0..g1 {
            let z = asm::dot_asm::<R>(args.bits.add(g * args.chunks * 16), a, args.lda, args.chunks);
            let ws = vld1q_f32(args.scale.add(4 * g));
            let nv = (n - 4 * g).min(4);
            for m in 0..F {
                let d: [int32x4_t; C] =
                    core::array::from_fn(|c| vnegq_s32(vaddq_s32(vshlq_n_s32::<1>(z[m * C + c]), uv[m * C + c])));
                if DOTS {
                    let base = (args.out as *mut i32).add((t0 + m) * C * n + 4 * g);
                    for c in 0..C {
                        store_i32(base.add(c * n), d[c], nv);
                    }
                } else {
                    let y = finish4::<C>(d, ws, fs[m]);
                    store_f32((args.out as *mut f32).add((t0 + m) * n + 4 * g), y, nv);
                }
            }
        }
    }
}

#[inline(always)]
unsafe fn store_i32(p: *mut i32, v: int32x4_t, nv: usize) {
    unsafe {
        if nv == 4 {
            vst1q_s32(p, v);
        } else {
            let mut tmp = [0i32; 4];
            vst1q_s32(tmp.as_mut_ptr(), v);
            core::ptr::copy_nonoverlapping(tmp.as_ptr(), p, nv);
        }
    }
}

#[inline(always)]
unsafe fn store_f32(p: *mut f32, v: float32x4_t, nv: usize) {
    unsafe {
        if nv == 4 {
            vst1q_f32(p, v);
        } else {
            let mut tmp = [0f32; 4];
            vst1q_f32(tmp.as_mut_ptr(), v);
            core::ptr::copy_nonoverlapping(tmp.as_ptr(), p, nv);
        }
    }
}
