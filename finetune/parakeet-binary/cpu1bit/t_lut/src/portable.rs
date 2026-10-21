//! Scalar twin of neon.rs for non-aarch64 builds (NixOS check builds and portable tests): the
//! same table, panel and scratch layouts and the same integer widths (INT16 runs use wrapping i16
//! lanes exactly like ADD.8H, so a wrong run length fails here too). Never timed.

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use shared::epilogue::finish;

use crate::{BlockArgs, BlockFn, RUN_GROUPS};

/// # Safety
/// As `neon::build_tables`.
pub unsafe fn build_tables(q: *const i8, len: usize, out: *mut i16) -> i32 {
    unsafe {
        let mut sum = 0i32;
        for g in 0..len.div_ceil(4) {
            let x: [i16; 4] = core::array::from_fn(|b| if 4 * g + b < len { *q.add(4 * g + b) as i16 } else { 0 });
            for n in 0..16 {
                let e: i16 = (0..4).filter(|b| (n >> b) & 1 == 1).map(|b| x[b]).fold(0i16, i16::wrapping_add);
                *out.add(g * 16 + n) = e;
            }
            sum += x.iter().map(|&v| v as i32).sum::<i32>();
        }
        sum
    }
}

/// # Safety
/// As `neon::expand_panel`.
pub unsafe fn expand_panel(v: usize, codes: *const u8, kb: usize, ob0: usize, ob1: usize, p0: usize, p1: usize, panel: *mut u8) {
    unsafe {
        let gstride = 2 * (p1 - p0);
        let halves = v / 2;
        for ob in ob0..ob1 {
            let base = panel.add((ob - ob0) * gstride * v * 16) as *mut u16;
            for h in 0..halves {
                let src = codes.add((ob * halves + h) * kb * 16);
                for p in p0..p1 {
                    let j = 2 * (p - p0);
                    for l in 0..16 {
                        let byte = *src.add(p * 16 + l);
                        for (jj, nib) in [(j, byte & 0x0f), (j + 1, byte >> 4)] {
                            *base.add((jj * v + 2 * h) * 8 + l) = 0x0100 + 0x0202 * nib as u16;
                        }
                    }
                }
            }
        }
    }
}

/// Byte-level TBL2 on a 32-byte table: out-of-range indices give 0, as on NEON.
fn tbl2(t: &[u8; 32], i: u8) -> u8 {
    if (i as usize) < 32 { t[i as usize] } else { 0 }
}

/// # Safety
/// As `neon::block`.
pub unsafe fn block<const C: usize, const F: usize, const R: usize, const V: usize, const DOTS: bool>(a: &BlockArgs) {
    unsafe {
        let w = 8 * V;
        let acc = a.acc;
        for r in 0..R {
            for o in 0..w {
                *acc.add(r * w + o) = if a.first { 0 } else { *a.persist.add(r * a.persist_stride + o) };
            }
        }
        let mut done = 0;
        while done < a.groups {
            let run = RUN_GROUPS.min(a.groups - done);
            let mut s = [[0i16; 32]; R];
            for j in done..done + run {
                let ip = a.panel.add(j * V * 16);
                for r in 0..R {
                    let t: &[u8; 32] = &*(a.tables.add(r * a.table_stride + j * 32) as *const [u8; 32]);
                    for o in 0..w {
                        let lo = tbl2(t, *ip.add(2 * o));
                        let hi = tbl2(t, *ip.add(2 * o + 1));
                        s[r][o] = s[r][o].wrapping_add(i16::from_le_bytes([lo, hi]));
                    }
                }
            }
            for r in 0..R {
                for o in 0..w {
                    *acc.add(r * w + o) += s[r][o] as i32;
                }
            }
            done += run;
        }
        if !a.last {
            for r in 0..R {
                for o in 0..w {
                    *a.persist.add(r * a.persist_stride + o) = *acc.add(r * w + o);
                }
            }
            return;
        }
        for f in 0..F {
            for o in 0..a.valid {
                let d: [i32; C] = core::array::from_fn(|c| 2 * *acc.add((f * C + c) * w + o) - *a.qsum.add(f * C + c));
                if DOTS {
                    for (c, &dc) in d.iter().enumerate() {
                        *(a.out as *mut i32).add((f * C + c) * a.n + o) = dc;
                    }
                } else {
                    *(a.out as *mut f32).add(f * a.n + o) = finish(&d, *a.wscale.add(o), *a.fscale.add(f));
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
