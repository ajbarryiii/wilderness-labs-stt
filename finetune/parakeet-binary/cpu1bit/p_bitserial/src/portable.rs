//! Portable scalar path (non-aarch64 hosts only: NixOS unit tests of the workspace). Same layouts,
//! same identity and the same driver as the NEON path, with u64 popcounts. Never timed.

use crate::layout::{pack_planes_scalar, plane_tile_bytes};
use crate::Accum;

pub type D4 = [i32; 4];

pub unsafe fn tile_udot<const G: usize, const F: usize>(w: *const u8, a: *const u8, steps: usize, corr: *const u32) -> [[D4; F]; G] {
    let mut c = [[[0u64; 4]; F]; G];
    for s in 0..steps {
        for g in 0..G {
            for j in 0..4 {
                let wr = unsafe { std::slice::from_raw_parts(w.add(s * 64 * G + 16 * (4 * g + j)), 16) };
                for f in 0..F {
                    for r in 0..8 {
                        let pl = unsafe { std::slice::from_raw_parts(a.add(s * 128 * F + 128 * f + 16 * r), 16) };
                        let pc: u32 = wr.iter().zip(pl).map(|(x, y)| (x ^ y).count_ones()).sum();
                        c[g][f][j] += u64::from(pc) << r;
                    }
                }
            }
        }
    }
    finish(c, corr)
}

pub unsafe fn tile_uadalp<const G: usize, const F: usize>(w: *const u16, a: *const u8, steps: usize, corr: *const u32) -> [[D4; F]; G] {
    let mut c = [[[0u64; 4]; F]; G];
    for s in 0..steps {
        for g in 0..G {
            for j in 0..4 {
                let wv = unsafe { *w.add(s * 4 * G + 4 * g + j) };
                for f in 0..F {
                    let pl = unsafe { std::slice::from_raw_parts(a.add(s * 16 * F + 16 * f), 16) };
                    for r in 0..8 {
                        let lane = u16::from_le_bytes([pl[2 * r], pl[2 * r + 1]]);
                        c[g][f][j] += u64::from((lane ^ wv).count_ones()) << r;
                    }
                }
            }
        }
    }
    finish(c, corr)
}

fn finish<const G: usize, const F: usize>(c: [[[u64; 4]; F]; G], corr: *const u32) -> [[D4; F]; G] {
    let mut d = [[[0i32; 4]; F]; G];
    for g in 0..G {
        for f in 0..F {
            for j in 0..4 {
                let cv = unsafe { *corr.add(4 * g + j) };
                d[g][f][j] = (c[g][f][j] as u32).wrapping_sub(cv) as i32;
            }
        }
    }
    d
}

pub unsafe fn store_f32<const C: usize>(y: *mut f32, d: [D4; C], ws: *const f32, fs: f32, valid: usize) {
    for lane in 0..valid {
        let parts: [i32; C] = std::array::from_fn(|c| d[c][lane]);
        unsafe { *y.add(lane) = shared::epilogue::finish(&parts, *ws.add(lane), fs) };
    }
}

pub unsafe fn store_dots(dst: *mut i32, d: D4, valid: usize) {
    for (lane, &v) in d.iter().enumerate().take(valid) {
        unsafe { *dst.add(lane) = v };
    }
}

pub unsafe fn pack_planes(q: *const i8, m_count: usize, k: usize, accum: Accum, f: usize, tiles: std::ops::Range<usize>, out: *mut u8) {
    let tb = plane_tile_bytes(accum, k, f);
    let q = unsafe { std::slice::from_raw_parts(q, m_count * k) };
    let dst = unsafe { std::slice::from_raw_parts_mut(out.add(tiles.start * tb), tiles.len() * tb) };
    pack_planes_scalar(q, m_count, k, accum, f, tiles, dst);
}

/// `UdotLane` computes exactly what `Udot` computes (same layouts and identity).
pub unsafe fn tile_udot_lane<const G: usize, const F: usize>(w: *const u8, a: *const u8, steps: usize, corr: *const u32) -> [[D4; F]; G] {
    unsafe { tile_udot::<G, F>(w, a, steps, corr) }
}
