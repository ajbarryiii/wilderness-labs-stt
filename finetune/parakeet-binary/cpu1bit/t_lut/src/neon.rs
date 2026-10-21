//! NEON (aarch64) table construction, index expansion and the TBL block kernel.
//! Data layouts and integer widths are documented in lib.rs; every routine here has a scalar twin
//! in portable.rs with the same layouts.

// Index loops over const-sized register arrays are deliberate: they fully unroll into registers.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use core::arch::aarch64::*;

use shared::epilogue::neon::finish4;

use crate::{BlockArgs, BlockFn, RUN_GROUPS};

/// Bits 0, 1, 2 of the entry index n = 0..7, as i16 lanes.
const B0: [i16; 8] = [0, 1, 0, 1, 0, 1, 0, 1];
const B1: [i16; 8] = [0, 0, 1, 1, 0, 0, 1, 1];
const B2: [i16; 8] = [0, 0, 0, 0, 1, 1, 1, 1];

/// One group's table from lanes L0..L3 of `x` (its 4 activations as i16):
/// `e0[n] = Σ_{k<3, bit k of n} q_k` (entries 0..7), `e1 = e0 + q_3` (entries 8..15).
macro_rules! group_table {
    ($x:expr, $l0:literal, $l1:literal, $l2:literal, $l3:literal, $b:expr, $out:expr) => {{
        let (b0, b1, b2, ones) = $b;
        let e0 = vmulq_laneq_s16::<$l0>(b0, $x);
        let e0 = vmlaq_laneq_s16::<$l1>(e0, b1, $x);
        let e0 = vmlaq_laneq_s16::<$l2>(e0, b2, $x);
        let e1 = vmlaq_laneq_s16::<$l3>(e0, ones, $x);
        vst1q_s16($out, e0);
        vst1q_s16($out.add(8), e1);
    }};
}

/// Tables of one frame-part over `len` inputs at `q` (a K-chunk): `ceil(len / 4)` tables of
/// 16 i16 at `out` (32 bytes each); inputs past `len` count as 0. Returns `Σ q` over the inputs.
///
/// # Safety
/// `q` readable for `len` bytes, `out` writable for `ceil(len / 4) * 16` i16, 16-byte aligned.
#[target_feature(enable = "neon")]
pub unsafe fn build_tables(q: *const i8, len: usize, out: *mut i16) -> i32 {
    unsafe {
        let b = (vld1q_s16(B0.as_ptr()), vld1q_s16(B1.as_ptr()), vld1q_s16(B2.as_ptr()), vdupq_n_s16(1));
        // Pairwise sums of 16 s8 per step: each lane gains at most 2 * 128 per step, so 64 steps
        // stay within i16 (64 * 256 = 16384) before the reduction to i32.
        let mut s16 = vdupq_n_s16(0);
        let mut sum = 0i32;
        let full = len / 16;
        let mut o = out;
        let mut step = 0;
        for ch in 0..full {
            let x = vld1q_s8(q.add(ch * 16));
            s16 = vpadalq_s8(s16, x);
            step += 1;
            if step == 64 {
                sum += vaddlvq_s16(s16);
                s16 = vdupq_n_s16(0);
                step = 0;
            }
            let lo = vmovl_s8(vget_low_s8(x));
            let hi = vmovl_high_s8(x);
            group_table!(lo, 0, 1, 2, 3, b, o);
            group_table!(lo, 4, 5, 6, 7, b, o.add(16));
            group_table!(hi, 0, 1, 2, 3, b, o.add(32));
            group_table!(hi, 4, 5, 6, 7, b, o.add(48));
            o = o.add(64);
        }
        sum += vaddlvq_s16(s16);
        let rem = len - full * 16;
        if rem > 0 {
            let mut buf = [0i8; 16];
            core::ptr::copy_nonoverlapping(q.add(full * 16), buf.as_mut_ptr(), rem);
            let x = vld1q_s8(buf.as_ptr());
            sum += vaddlvq_s8(x) as i32;
            let lo = vmovl_s8(vget_low_s8(x));
            let hi = vmovl_high_s8(x);
            let groups = rem.div_ceil(4);
            group_table!(lo, 0, 1, 2, 3, b, o);
            if groups > 1 {
                group_table!(lo, 4, 5, 6, 7, b, o.add(16));
            }
            if groups > 2 {
                group_table!(hi, 0, 1, 2, 3, b, o.add(32));
            }
            if groups > 3 {
                group_table!(hi, 4, 5, 6, 7, b, o.add(48));
            }
        }
        sum
    }
}

/// TBL2 index vectors for out-blocks `ob0..ob1` (8 * `v` outputs each) and code bytes `p0..p1`
/// (groups `2 (p - p0)` and `2 (p - p0) + 1` of the chunk). Panel layout:
/// `panel[((ob - ob0) * gstride + j) * v + vec]` = 8 u16 lanes `0x0100 + 0x0202 * n` for
/// outputs `ob * 8v + 8 vec ..`, with `gstride = 2 (p1 - p0)`.
///
/// # Safety
/// `codes` is a tiled layout with `kb` bytes per row covering blocks `ob1 * v / 2`; `panel`
/// writable for `(ob1 - ob0) * gstride * v * 16` bytes, 16-byte aligned; `v` is 2 or 4.
#[target_feature(enable = "neon")]
pub unsafe fn expand_panel(v: usize, codes: *const u8, kb: usize, ob0: usize, ob1: usize, p0: usize, p1: usize, panel: *mut u8) {
    unsafe {
        let mask = vdupq_n_u8(0x0f);
        let c1 = vdupq_n_u16(0x0100);
        let c2 = vdupq_n_u16(0x0202);
        let gstride = 2 * (p1 - p0);
        let halves = v / 2;
        for ob in ob0..ob1 {
            let base = panel.add((ob - ob0) * gstride * v * 16);
            for h in 0..halves {
                let src = codes.add((ob * halves + h) * kb * 16);
                for p in p0..p1 {
                    let x = vld1q_u8(src.add(p * 16));
                    let a = vandq_u8(x, mask);
                    let b = vshrq_n_u8::<4>(x);
                    let j = 2 * (p - p0);
                    let d0 = base.add((j * v + 2 * h) * 16) as *mut u16;
                    let d1 = base.add(((j + 1) * v + 2 * h) * 16) as *mut u16;
                    vst1q_u16(d0, vmlaq_u16(c1, vmovl_u8(vget_low_u8(a)), c2));
                    vst1q_u16(d0.add(8), vmlaq_u16(c1, vmovl_high_u8(a), c2));
                    vst1q_u16(d1, vmlaq_u16(c1, vmovl_u8(vget_low_u8(b)), c2));
                    vst1q_u16(d1.add(8), vmlaq_u16(c1, vmovl_high_u8(b), c2));
                }
            }
        }
    }
}

/// The block kernel: `R = F * C` frame-parts (rows `r = f * C + c`) x `8 V` outputs over one
/// K-chunk. Per group: `V` index loads, per row one 32-byte table load and `V` (TBL2 + ADD.8H).
/// INT16 runs of at most [`RUN_GROUPS`] groups are widened (SADDW/SADDW2) into the i32 scratch;
/// then either the chunk sums go to the persistent buffer, or (last chunk) the epilogue runs:
/// `d = 2 F - Q`, stored as dots or finished with `finish4::<C>`.
///
/// # Safety
/// All pointers of `a` valid as documented on [`BlockArgs`]; `R == F * C`, `V` in {2, 4}.
#[inline(never)]
#[target_feature(enable = "neon")]
pub unsafe fn block<const C: usize, const F: usize, const R: usize, const V: usize, const DOTS: bool>(a: &BlockArgs) {
    debug_assert!(R == F * C && (V == 2 || V == 4));
    unsafe {
        let w = 8 * V; // outputs per out-block = i32 per scratch row
        let acc = a.acc;
        if a.first {
            for r in 0..R {
                for u in 0..2 * V {
                    vst1q_s32(acc.add(r * w + 4 * u), vdupq_n_s32(0));
                }
            }
        } else {
            for r in 0..R {
                for u in 0..2 * V {
                    vst1q_s32(acc.add(r * w + 4 * u), vld1q_s32(a.persist.add(r * a.persist_stride + 4 * u)));
                }
            }
        }
        let tp: [*const u8; R] = core::array::from_fn(|r| a.tables.add(r * a.table_stride));
        let mut done = 0;
        while done < a.groups {
            let run = RUN_GROUPS.min(a.groups - done);
            let mut s = [[vdupq_n_s16(0); V]; R];
            let mut ip = a.panel.add(done * V * 16);
            let mut off = done * 32;
            for _ in 0..run {
                let idx: [uint8x16_t; V] = core::array::from_fn(|v| vld1q_u8(ip.add(v * 16)));
                for r in 0..R {
                    let t = vld1q_u8_x2(tp[r].add(off));
                    for v in 0..V {
                        s[r][v] = vaddq_s16(s[r][v], vreinterpretq_s16_u8(vqtbl2q_u8(t, idx[v])));
                    }
                }
                ip = ip.add(V * 16);
                off += 32;
            }
            for r in 0..R {
                for v in 0..V {
                    let p = acc.add(r * w + 8 * v);
                    vst1q_s32(p, vaddw_s16(vld1q_s32(p), vget_low_s16(s[r][v])));
                    vst1q_s32(p.add(4), vaddw_high_s16(vld1q_s32(p.add(4)), s[r][v]));
                }
            }
            // Memory barrier for the compiler only: the scratch is re-read at the next flush
            // instead of being kept live (and spilled) across the run's loop.
            core::arch::asm!("", options(nostack, preserves_flags));
            done += run;
        }
        if !a.last {
            for r in 0..R {
                for u in 0..2 * V {
                    vst1q_s32(a.persist.add(r * a.persist_stride + 4 * u), vld1q_s32(acc.add(r * w + 4 * u)));
                }
            }
            return;
        }
        for f in 0..F {
            let fs = *a.fscale.add(f);
            let q: [int32x4_t; C] = core::array::from_fn(|c| vdupq_n_s32(*a.qsum.add(f * C + c)));
            for u in 0..2 * V {
                if 4 * u >= a.valid {
                    break;
                }
                let d: [int32x4_t; C] = core::array::from_fn(|c| {
                    let fsum = vld1q_s32(acc.add((f * C + c) * w + 4 * u));
                    vsubq_s32(vshlq_n_s32::<1>(fsum), q[c])
                });
                let full = 4 * u + 4 <= a.valid;
                let lanes = (a.valid - 4 * u).min(4);
                if DOTS {
                    let out = a.out as *mut i32;
                    for c in 0..C {
                        let dst = out.add((f * C + c) * a.n + 4 * u);
                        if full {
                            vst1q_s32(dst, d[c]);
                        } else {
                            let mut tmp = [0i32; 4];
                            vst1q_s32(tmp.as_mut_ptr(), d[c]);
                            core::ptr::copy_nonoverlapping(tmp.as_ptr(), dst, lanes);
                        }
                    }
                } else {
                    let y = finish4::<C>(d, vld1q_f32(a.wscale.add(4 * u)), fs);
                    let dst = (a.out as *mut f32).add(f * a.n + 4 * u);
                    if full {
                        vst1q_f32(dst, y);
                    } else {
                        let mut tmp = [0f32; 4];
                        vst1q_f32(tmp.as_mut_ptr(), y);
                        core::ptr::copy_nonoverlapping(tmp.as_ptr(), dst, lanes);
                    }
                }
            }
        }
    }
}

macro_rules! pick {
    ($c:literal, $v:literal, $dots:literal, $f:expr, [$(($ff:literal, $rr:literal)),*]) => {
        match $f {
            $($ff => block::<$c, $ff, $rr, $v, $dots> as BlockFn,)*
            f => panic!("t_lut: no block kernel for C={} F={f} V={}", $c, $v),
        }
    };
}

/// The block kernel for `c` components, `f` frames, `v` index vectors and the output kind.
pub fn block_fn(c: usize, f: usize, v: usize, dots: bool) -> BlockFn {
    match (c, v, dots) {
        (1, 4, false) => pick!(1, 4, false, f, [(1, 1), (2, 2), (3, 3), (4, 4)]),
        (1, 4, true) => pick!(1, 4, true, f, [(1, 1), (2, 2), (3, 3), (4, 4)]),
        (2, 4, false) => pick!(2, 4, false, f, [(1, 2), (2, 4)]),
        (2, 4, true) => pick!(2, 4, true, f, [(1, 2), (2, 4)]),
        (3, 4, false) => pick!(3, 4, false, f, [(1, 3)]),
        (3, 4, true) => pick!(3, 4, true, f, [(1, 3)]),
        (1, 2, false) => pick!(1, 2, false, f, [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]),
        (1, 2, true) => pick!(1, 2, true, f, [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]),
        (2, 2, false) => pick!(2, 2, false, f, [(1, 2), (2, 4), (3, 6), (4, 8)]),
        (2, 2, true) => pick!(2, 2, true, f, [(1, 2), (2, 4), (3, 6), (4, 8)]),
        (3, 2, false) => pick!(3, 2, false, f, [(1, 3), (2, 6)]),
        (3, 2, true) => pick!(3, 2, true, f, [(1, 3), (2, 6)]),
        _ => panic!("t_lut: unsupported C={c} V={v}"),
    }
}
