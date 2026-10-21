//! NEON micro-kernels: one tile of `4 G` outputs x `F` frame-parts over all K steps, returning the
//! exact integer dots `d = C - (127 K + P)` as i32x4 per (output group, frame-part).
//!
//! Per (output, frame-part) and step, the work is EOR + CNT + one accumulate: 3 SIMD uops per
//! 128 bit-products (16 int8 MACs). On Firestorm every one of these is 1 uop at 4 per cycle on the
//! four SIMD units (Dougall Johnson's M1 tables), so the floor is 0.75 cycle per 16 MACs and core,
//! three times SDOT's. Loads go to the separate load units (LDR q: 1 load uop; LD1R: 1 load uop
//! plus 1 SIMD uop).
#![allow(unsafe_op_in_unsafe_fn, unused_unsafe)]

use core::arch::aarch64::*;

pub type D4 = int32x4_t;

/// Plane weights for UDOT by element: `PW[4 l .. 4 l + 4] = 2^l` (planes 0..3), then 2^(4 + l).
/// LLVM folds the lane broadcast of these constants and emits the vector UDOT against 8 splat
/// registers (same uop and throughput; it is why Udot tiles stop at 16 accumulators).
static PW: [u8; 32] = [
    1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, //
    16, 16, 16, 16, 32, 32, 32, 32, 64, 64, 64, 64, 128, 128, 128, 128,
];
/// Plane weights 2^r of the u16 lanes of `Uadalp`.
static W16: [u16; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// `Uadalp` flushes its u16 lanes every this many steps: each step adds at most two CNT bytes
/// (<= 16) per lane, and 4095 * 16 = 65,520 < 65,536.
pub const FLUSH_STEPS: usize = 4095;

/// `[C(o0), C(o1), C(o2), C(o3)]` from four accumulators whose lanes sum to each C.
#[inline(always)]
unsafe fn hsum4(a: uint32x4_t, b: uint32x4_t, c: uint32x4_t, d: uint32x4_t) -> uint32x4_t {
    vpaddq_u32(vpaddq_u32(a, b), vpaddq_u32(c, d))
}

#[inline(always)]
unsafe fn finish_dots<const G: usize, const F: usize>(acc: &[[[uint32x4_t; F]; 4]; G], corr: *const u32) -> [[D4; F]; G] {
    let mut d = [[vdupq_n_s32(0); F]; G];
    for g in 0..G {
        let cv = vld1q_u32(corr.add(4 * g));
        for f in 0..F {
            let c = hsum4(acc[g][0][f], acc[g][1][f], acc[g][2][f], acc[g][3][f]);
            // Wrapping u32 subtraction; the true d lies in [-128 K, 127 K] and so fits i32.
            d[g][f] = vreinterpretq_s32_u32(vsubq_u32(c, cv));
        }
    }
    d
}


/// `UDOT acc.4s, c.16b, pw.4b[L]` pinned in its by-element encoding. Through the intrinsic, LLVM
/// hoists the lane broadcast out of the loop and keeps 8 splat registers; this single-instruction
/// asm keeps the 2 `PW` registers instead (`UdotLane`, F = 6).
/// Same uop on Firestorm (UDOT by element: latency 3, 4 per cycle, u11-14).
#[inline(always)]
unsafe fn udot_lane_asm<const L: i32>(acc: uint32x4_t, c: uint8x16_t, pw: uint8x16_t) -> uint32x4_t {
    let mut a = acc;
    macro_rules! lane {
        ($s:literal) => {
            core::arch::asm!($s, a = inout(vreg) a, c = in(vreg) c, p = in(vreg) pw,
                             options(pure, nomem, nostack, preserves_flags))
        };
    }
    match L {
        0 => lane!("udot {a:v}.4s, {c:v}.16b, {p:v}.4b[0]"),
        1 => lane!("udot {a:v}.4s, {c:v}.16b, {p:v}.4b[1]"),
        2 => lane!("udot {a:v}.4s, {c:v}.16b, {p:v}.4b[2]"),
        _ => lane!("udot {a:v}.4s, {c:v}.16b, {p:v}.4b[3]"),
    }
    a
}

/// Plane `r` (UDOT lane `L` of `pw`; `pa` = plane r of frame-part 0) of every frame-part against
/// the tile's weight vectors. Plane-outer order: consecutive triples cycle through all 4 G F
/// accumulators, so each accumulator's UDOT chain (latency 3) has 4 G F - 1 triples of slack, and
/// only one plane vector is live at a time.
#[inline(always)]
unsafe fn udot_plane<const G: usize, const F: usize, const L: i32>(
    acc: &mut [[[uint32x4_t; F]; 4]; G],
    wv: &[[uint8x16_t; 4]; G],
    pa: *const u8,
    pw: uint8x16_t,
) {
    for f in 0..F {
        let av = vld1q_u8(pa.add(128 * f));
        for g in 0..G {
            for j in 0..4 {
                let c = vcntq_u8(veorq_u8(av, wv[g][j]));
                acc[g][j][f] = vdotq_laneq_u32::<L>(acc[g][j][f], c, pw);
            }
        }
    }
}

/// `Udot` tile: `w` = this tile's codes (`[steps][4G][16]`), `a` = the frame tile's planes
/// (`[steps][F][8][16]`), `corr` = the tile's `4G` corrections.
///
/// Each u32 lane gains at most `Σ_r 2^r · 4 · 8 = 8160` per step; with `steps <= 2^17`
/// (K <= 2^24, asserted in `prepare`) the lanes stay below 2^30, and C <= 255 K < 2^32.
#[inline(always)]
pub unsafe fn tile_udot<const G: usize, const F: usize>(w: *const u8, a: *const u8, steps: usize, corr: *const u32) -> [[D4; F]; G] {
    let pw0 = vld1q_u8(PW.as_ptr());
    let pw1 = vld1q_u8(PW.as_ptr().add(16));
    let mut acc = [[[vdupq_n_u32(0); F]; 4]; G];
    let (mut wp, mut ap) = (w, a);
    for _ in 0..steps {
        let mut wv = [[vdupq_n_u8(0); 4]; G];
        for g in 0..G {
            for j in 0..4 {
                wv[g][j] = vld1q_u8(wp.add(16 * (4 * g + j)));
            }
        }
        udot_plane::<G, F, 0>(&mut acc, &wv, ap, pw0);
        udot_plane::<G, F, 1>(&mut acc, &wv, ap.add(16), pw0);
        udot_plane::<G, F, 2>(&mut acc, &wv, ap.add(32), pw0);
        udot_plane::<G, F, 3>(&mut acc, &wv, ap.add(48), pw0);
        udot_plane::<G, F, 0>(&mut acc, &wv, ap.add(64), pw1);
        udot_plane::<G, F, 1>(&mut acc, &wv, ap.add(80), pw1);
        udot_plane::<G, F, 2>(&mut acc, &wv, ap.add(96), pw1);
        udot_plane::<G, F, 3>(&mut acc, &wv, ap.add(112), pw1);
        wp = wp.add(64 * G);
        ap = ap.add(128 * F);
    }
    finish_dots::<G, F>(&acc, corr)
}

/// `Uadalp` tile: `w` = this tile's codes (`[steps][4G]` u16), `a` = the frame tile's planes
/// (`[steps][F][16]`), `corr` = the tile's `4G` corrections. u16 lane `r` of each accumulator
/// counts plane `r`; the 2^r weighting happens once per flush (UMULL / UMLAL2 by `W16`).
#[inline(always)]
pub unsafe fn tile_uadalp<const G: usize, const F: usize>(w: *const u16, a: *const u8, steps: usize, corr: *const u32) -> [[D4; F]; G] {
    let wt = vld1q_u16(W16.as_ptr());
    let wt_lo = vget_low_u16(wt);
    let mut sum = [[[vdupq_n_u32(0); F]; 4]; G];
    let (mut wp, mut ap) = (w, a);
    let mut left = steps;
    while left > 0 {
        let blk = left.min(FLUSH_STEPS);
        let mut acc = [[[vdupq_n_u16(0); F]; 4]; G];
        for _ in 0..blk {
            let mut av = [vdupq_n_u8(0); F];
            for (f, v) in av.iter_mut().enumerate() {
                *v = vld1q_u8(ap.add(16 * f));
            }
            for g in 0..G {
                for j in 0..4 {
                    let wv = vreinterpretq_u8_u16(vld1q_dup_u16(wp.add(4 * g + j)));
                    for f in 0..F {
                        acc[g][j][f] = vpadalq_u8(acc[g][j][f], vcntq_u8(veorq_u8(wv, av[f])));
                    }
                }
            }
            wp = wp.add(4 * G);
            ap = ap.add(16 * F);
        }
        for g in 0..G {
            for j in 0..4 {
                for f in 0..F {
                    let x = acc[g][j][f];
                    sum[g][j][f] = vmlal_high_u16(vmlal_u16(sum[g][j][f], vget_low_u16(x), wt_lo), x, wt);
                }
            }
        }
        left -= blk;
    }
    finish_dots::<G, F>(&sum, corr)
}

/// FP32 epilogue for 4 outputs of one frame: `shared::epilogue::neon::finish4`, then a full or
/// partial (ragged N) store.
#[inline(always)]
pub unsafe fn store_f32<const C: usize>(y: *mut f32, d: [D4; C], ws: *const f32, fs: f32, valid: usize) {
    let v = shared::epilogue::neon::finish4::<C>(d, vld1q_f32(ws), fs);
    if valid == 4 {
        vst1q_f32(y, v);
    } else {
        let mut tmp = [0f32; 4];
        vst1q_f32(tmp.as_mut_ptr(), v);
        std::ptr::copy_nonoverlapping(tmp.as_ptr(), y, valid);
    }
}

/// Raw dots of 4 outputs of one frame-part.
#[inline(always)]
pub unsafe fn store_dots(dst: *mut i32, d: D4, valid: usize) {
    if valid == 4 {
        vst1q_s32(dst, d);
    } else {
        let mut tmp = [0i32; 4];
        vst1q_s32(tmp.as_mut_ptr(), d);
        std::ptr::copy_nonoverlapping(tmp.as_ptr(), dst, valid);
    }
}

pub use crate::layout::neon::pack_planes;

/// One plane `r` of the `UdotLane` tile, in two output halves (outputs 0-1, then 2-3). The empty
/// `asm!` is a compiler memory barrier: LLVM must reload the half's 2 weight vectors and the F
/// plane vectors from L1 after it instead of keeping all 4 weight vectors and 6 plane vectors
/// live, so 24 accumulators + 2 weights + 1 plane + 1 temporary + 2 constants = 30 registers fit.
/// Cost: 2 + F loads per 2F triples (0.67 L1 loads per triple at F = 6; Firestorm has 3 load
/// units).
#[inline(always)]
unsafe fn udot_lane_plane<const F: usize, const L: i32>(acc: &mut [[uint32x4_t; F]; 4], wp: *const u8, pa: *const u8, pw: uint8x16_t) {
    for h in 0..2 {
        core::arch::asm!("", options(nostack, preserves_flags));
        let w0 = vld1q_u8(wp.add(32 * h));
        let w1 = vld1q_u8(wp.add(32 * h + 16));
        for f in 0..F {
            let av = vld1q_u8(pa.add(128 * f));
            let c0 = vcntq_u8(veorq_u8(av, w0));
            acc[2 * h][f] = udot_lane_asm::<L>(acc[2 * h][f], c0, pw);
            let c1 = vcntq_u8(veorq_u8(av, w1));
            acc[2 * h + 1][f] = udot_lane_asm::<L>(acc[2 * h + 1][f], c1, pw);
        }
    }
}

/// `UdotLane` tile (4 outputs, F frame-parts): the `Udot` layouts and identity with UDOT pinned to
/// its by-element encoding and the weights re-read per output half (`udot_lane_plane`), which
/// leaves room for F = 6 (24 accumulators).
#[inline(always)]
pub unsafe fn tile_udot_lane<const G: usize, const F: usize>(w: *const u8, a: *const u8, steps: usize, corr: *const u32) -> [[D4; F]; G] {
    assert!(G == 1, "UdotLane tiles have 4 outputs");
    let pw0 = vld1q_u8(PW.as_ptr());
    let pw1 = vld1q_u8(PW.as_ptr().add(16));
    let mut acc = [[vdupq_n_u32(0); F]; 4];
    let (mut wp, mut ap) = (w, a);
    for _ in 0..steps {
        udot_lane_plane::<F, 0>(&mut acc, wp, ap, pw0);
        udot_lane_plane::<F, 1>(&mut acc, wp, ap.add(16), pw0);
        udot_lane_plane::<F, 2>(&mut acc, wp, ap.add(32), pw0);
        udot_lane_plane::<F, 3>(&mut acc, wp, ap.add(48), pw0);
        udot_lane_plane::<F, 0>(&mut acc, wp, ap.add(64), pw1);
        udot_lane_plane::<F, 1>(&mut acc, wp, ap.add(80), pw1);
        udot_lane_plane::<F, 2>(&mut acc, wp, ap.add(96), pw1);
        udot_lane_plane::<F, 3>(&mut acc, wp, ap.add(112), pw1);
        wp = wp.add(64);
        ap = ap.add(128 * F);
    }
    let d = finish_dots::<1, F>(&[acc], corr);
    core::array::from_fn(|_| d[0])
}
