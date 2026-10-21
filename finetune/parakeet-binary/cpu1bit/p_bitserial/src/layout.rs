//! P's weight and activation-plane layouts. Both keep exactly one bit per weight and eight bits
//! (planes) per activation; only the interleave differs between the two accumulation variants.
//!
//! Notation: `K̄ = steps * S` is K rounded up to the step `S` (128 positions for `Udot` and
//! `UdotLane`, which share every layout, 16 for
//! `Uadalp`); a tile is `O` consecutive outputs (weights) or `F` consecutive frame-parts
//! (activations; frame-part `m = t * C + c`, LocalFlow's `[T][C][K]` order).
//!
//! Weights store `n = ¬b ∧ valid` (b = the export bit, 1 = +1), zero for positions `>= K` and
//! rows `>= N`:
//! - `Udot`:   `[N/O][steps][O][16]` bytes; byte `i` bit `j` = n at position `128 s + 8 i + j`
//!   (the export's own LSB-first order, 16 export bytes per step).
//! - `Uadalp`: `[N/O][steps][O]` u16 (little-endian); bit `j` = n at position `16 s + j`.
//!
//! Activation planes store the bits `a_r` of `u = q + 128 = q ^ 0x80`, zero for positions `>= K`
//! and frame-parts `>= M` (padding of the last tile):
//! - `Udot`:   `[M/F][steps][F][r = 0..8][16]` bytes; plane `r` byte `i` bit `j` = a_r at position
//!   `128 s + 8 i + j`, the same bit order as the weights, so one EOR pairs them.
//! - `Uadalp`: `[M/F][steps][F][16]` bytes viewed as u16 x 8: lane `r` bit `j` = a_r at position
//!   `16 s + j`; LD1R.8H of a weight u16 lines it up with every lane.
//!
//! With these zeros, `(a_r ⊕ n)` is 0 at every padded position, i.e. it equals the plan's
//! `(a_r XOR n) AND valid` with `valid` folded into the stored operands (no AND at run time).

use shared::BinaryMatrix;
use shared::aligned::AlignedVec;

use crate::Accum;

/// Positions per K step.
pub const fn step_positions(a: Accum) -> usize {
    match a {
        Accum::Udot | Accum::UdotLane => 128,
        Accum::Uadalp => 16,
    }
}

/// Bits of an 8-position group `byte` (positions `8 byte .. 8 byte + 8`) that lie below `k`.
#[inline]
pub fn valid_mask(byte: usize, k: usize) -> u8 {
    let lo = 8 * byte;
    if lo + 8 <= k {
        0xff
    } else if lo >= k {
        0
    } else {
        (1u8 << (k - lo)) - 1
    }
}

/// One prepared matrix.
pub struct PackedWeights {
    pub accum: Accum,
    pub n: usize,
    pub k: usize,
    /// Outputs per tile (4 or 8).
    pub o: usize,
    pub tiles: usize,
    pub steps: usize,
    /// See the module docs; `tiles * steps * o * S / 8` bytes.
    pub codes: AlignedVec<u8>,
    /// `127 K + P` per output (P = number of +1 weights), so `d = C - corr`; 0 past N.
    pub corr: AlignedVec<u32>,
    /// Weight scales, 0 past N.
    pub scale: AlignedVec<f32>,
}

impl PackedWeights {
    /// Code bytes of one output tile.
    pub fn tile_code_bytes(&self) -> usize {
        self.steps * self.o * step_positions(self.accum) / 8
    }
}

pub fn pack_weights(m: &BinaryMatrix, accum: Accum, o: usize) -> PackedWeights {
    let (n, k) = (m.n, m.k);
    let rb = m.row_bytes();
    let s = step_positions(accum);
    let wb = s / 8;
    let steps = k.div_ceil(s);
    let tiles = n.div_ceil(o);
    let mut codes = AlignedVec::<u8>::zeroed(tiles * steps * o * wb);
    let mut corr = AlignedVec::<u32>::zeroed(tiles * o);
    let mut scale = AlignedVec::<f32>::zeroed(tiles * o);
    for row in 0..n {
        let (tile, j) = (row / o, row % o);
        let src = &m.codes[row * rb..(row + 1) * rb];
        for st in 0..steps {
            for i in 0..wb {
                let byte = st * wb + i;
                let nb = if byte < rb { !src[byte] & valid_mask(byte, k) } else { 0 };
                codes[((tile * steps + st) * o + j) * wb + i] = nb;
            }
        }
        let p: u32 = src.iter().map(|b| b.count_ones()).sum();
        corr[row] = 127 * k as u32 + p;
        scale[row] = m.scale[row];
    }
    PackedWeights { accum, n, k, o, tiles, steps, codes, corr, scale }
}

/// `[N][K]` signs recovered from the stored `n` bits (bit 1 = -1).
pub fn unpack_weights(w: &PackedWeights) -> Vec<i8> {
    let s = step_positions(w.accum);
    let wb = s / 8;
    let mut out = Vec::with_capacity(w.n * w.k);
    for row in 0..w.n {
        let (tile, j) = (row / w.o, row % w.o);
        for pos in 0..w.k {
            let (st, within) = (pos / s, pos % s);
            let byte = w.codes[((tile * w.steps + st) * w.o + j) * wb + within / 8];
            out.push(if (byte >> (within % 8)) & 1 == 1 { -1 } else { 1 });
        }
    }
    out
}

/// Bytes of one activation tile (`F` frame-parts, all steps) and of all tiles for `m` frame-parts.
pub fn plane_tile_bytes(accum: Accum, k: usize, f: usize) -> usize {
    let s = step_positions(accum);
    k.div_ceil(s) * s * f
}

/// Scalar reference packer (and the portable path): frame-part tiles `tiles` of `q` (`[M][K]` s8)
/// into `out`, which holds exactly those tiles, tile `tiles.start` first.
pub fn pack_planes_scalar(q: &[i8], m_count: usize, k: usize, accum: Accum, f: usize, tiles: std::ops::Range<usize>, out: &mut [u8]) {
    let s = step_positions(accum);
    let steps = k.div_ceil(s);
    let first = tiles.start;
    for tile in tiles {
        for st in 0..steps {
            for fi in 0..f {
                let base = (((tile - first) * steps + st) * f + fi) * s;
                let block = &mut out[base..base + s];
                block.fill(0);
                let m = tile * f + fi;
                if m >= m_count {
                    continue;
                }
                for pos in 0..s {
                    let p = st * s + pos;
                    if p >= k {
                        break;
                    }
                    let u = (q[m * k + p] as u8) ^ 0x80;
                    for r in 0..8 {
                        if (u >> r) & 1 == 1 {
                            match accum {
                                Accum::Udot | Accum::UdotLane => block[r * 16 + pos / 8] |= 1 << (pos % 8),
                                Accum::Uadalp => block[2 * r + pos / 8] |= 1 << (pos % 8),
                            }
                        }
                    }
                }
            }
        }
    }
}

/// `u` bytes recovered from a packed plane buffer (tests).
pub fn unpack_planes(buf: &[u8], m_count: usize, k: usize, accum: Accum, f: usize) -> Vec<u8> {
    let s = step_positions(accum);
    let steps = k.div_ceil(s);
    let mut u = vec![0u8; m_count * k];
    for m in 0..m_count {
        let (tile, fi) = (m / f, m % f);
        for p in 0..k {
            let (st, pos) = (p / s, p % s);
            let block = &buf[((tile * steps + st) * f + fi) * s..][..s];
            let mut v = 0u8;
            for r in 0..8 {
                let byte = match accum {
                    Accum::Udot | Accum::UdotLane => block[r * 16 + pos / 8],
                    Accum::Uadalp => block[2 * r + pos / 8],
                };
                v |= ((byte >> (pos % 8)) & 1) << r;
            }
            u[m * k + p] = v;
        }
    }
    u
}

#[cfg(target_arch = "aarch64")]
pub mod neon {
    //! NEON plane packing. 128 positions at a time: LD4 + UZP put byte `j` of every 8-byte group
    //! into register `j`, a three-stage BSL bit transpose (blocks 1, 2, 4) turns register `j` bit
    //! `r` into register `r` bit `j`, and plane 7 is inverted (`u = q ^ 0x80`). `Uadalp` then
    //! transposes the 8 x 8 u16 elements (TRN on 16, 32, 64 bits) into its plane-in-lane order.
    #![allow(unsafe_op_in_unsafe_fn, unused_unsafe)]
    use super::step_positions;
    use crate::Accum;
    use core::arch::aarch64::*;

    #[inline(always)]
    unsafe fn swap<const B: i32>(x: uint8x16_t, y: uint8x16_t, m: uint8x16_t) -> (uint8x16_t, uint8x16_t) {
        (vbslq_u8(m, vshlq_n_u8::<B>(y), x), vbslq_u8(m, y, vshrq_n_u8::<B>(x)))
    }

    /// Planes of 128 s8 values at `src`: `p[r]` byte `i` bit `j` = bit `r` of `u[8 i + j]`.
    #[inline(always)]
    pub unsafe fn planes128(src: *const i8) -> [uint8x16_t; 8] {
        let a = vld4q_u8(src as *const u8);
        let b = vld4q_u8((src as *const u8).add(64));
        // w[j] byte i = x[8 i + j].
        let mut w = [
            vuzp1q_u8(a.0, b.0),
            vuzp1q_u8(a.1, b.1),
            vuzp1q_u8(a.2, b.2),
            vuzp1q_u8(a.3, b.3),
            vuzp2q_u8(a.0, b.0),
            vuzp2q_u8(a.1, b.1),
            vuzp2q_u8(a.2, b.2),
            vuzp2q_u8(a.3, b.3),
        ];
        let m1 = vdupq_n_u8(0xaa);
        for j in [0, 2, 4, 6] {
            (w[j], w[j + 1]) = swap::<1>(w[j], w[j + 1], m1);
        }
        let m2 = vdupq_n_u8(0xcc);
        for j in [0, 1, 4, 5] {
            (w[j], w[j + 2]) = swap::<2>(w[j], w[j + 2], m2);
        }
        let m4 = vdupq_n_u8(0xf0);
        for j in [0, 1, 2, 3] {
            (w[j], w[j + 4]) = swap::<4>(w[j], w[j + 4], m4);
        }
        w[7] = vmvnq_u8(w[7]);
        w
    }

    /// 8 x 8 transpose of u16 elements: `out[t].u16[r] = p[r].u16[t]`.
    #[inline(always)]
    pub unsafe fn transpose_u16(p: [uint8x16_t; 8]) -> [uint8x16_t; 8] {
        let r: [uint16x8_t; 8] = core::array::from_fn(|i| vreinterpretq_u16_u8(p[i]));
        let a = [
            vtrn1q_u16(r[0], r[1]),
            vtrn2q_u16(r[0], r[1]),
            vtrn1q_u16(r[2], r[3]),
            vtrn2q_u16(r[2], r[3]),
            vtrn1q_u16(r[4], r[5]),
            vtrn2q_u16(r[4], r[5]),
            vtrn1q_u16(r[6], r[7]),
            vtrn2q_u16(r[6], r[7]),
        ];
        let a: [uint32x4_t; 8] = core::array::from_fn(|i| vreinterpretq_u32_u16(a[i]));
        let b = [
            vtrn1q_u32(a[0], a[2]),
            vtrn1q_u32(a[1], a[3]),
            vtrn2q_u32(a[0], a[2]),
            vtrn2q_u32(a[1], a[3]),
            vtrn1q_u32(a[4], a[6]),
            vtrn1q_u32(a[5], a[7]),
            vtrn2q_u32(a[4], a[6]),
            vtrn2q_u32(a[5], a[7]),
        ];
        let b: [uint64x2_t; 8] = core::array::from_fn(|i| vreinterpretq_u64_u32(b[i]));
        let c = [
            vtrn1q_u64(b[0], b[4]),
            vtrn1q_u64(b[1], b[5]),
            vtrn1q_u64(b[2], b[6]),
            vtrn1q_u64(b[3], b[7]),
            vtrn2q_u64(b[0], b[4]),
            vtrn2q_u64(b[1], b[5]),
            vtrn2q_u64(b[2], b[6]),
            vtrn2q_u64(b[3], b[7]),
        ];
        core::array::from_fn(|i| vreinterpretq_u8_u64(c[i]))
    }

    /// NEON equivalent of `pack_planes_scalar` (bit-identical, tested), on raw pointers so pool
    /// threads can fill disjoint tiles of one buffer.
    pub unsafe fn pack_planes(q: *const i8, m_count: usize, k: usize, accum: Accum, f: usize,
                              tiles: std::ops::Range<usize>, out: *mut u8) {
        let s = step_positions(accum);
        let steps = k.div_ceil(s);
        let chunks = k.div_ceil(128);
        let full = k / 128;
        let mut tmp = [0i8; 128];
        for tile in tiles {
            for fi in 0..f {
                let m = tile * f + fi;
                if m >= m_count {
                    for st in 0..steps {
                        std::ptr::write_bytes(out.add(((tile * steps + st) * f + fi) * s), 0, s);
                    }
                    continue;
                }
                let row = q.add(m * k);
                for ch in 0..chunks {
                    let src = if ch < full {
                        row.add(128 * ch)
                    } else {
                        let rest = k - 128 * ch;
                        tmp.fill(0x80u8 as i8); // q = -128 -> u = 0: zero planes past K
                        std::ptr::copy_nonoverlapping(row.add(128 * ch), tmp.as_mut_ptr(), rest);
                        tmp.as_ptr()
                    };
                    let p = planes128(src);
                    match accum {
                        Accum::Udot | Accum::UdotLane => {
                            let dst = out.add(((tile * steps + ch) * f + fi) * 128);
                            for (r, v) in p.iter().enumerate() {
                                vst1q_u8(dst.add(16 * r), *v);
                            }
                        }
                        Accum::Uadalp => {
                            let v = transpose_u16(p);
                            for (t, vt) in v.iter().enumerate() {
                                let st = 8 * ch + t;
                                if st < steps {
                                    vst1q_u8(out.add(((tile * steps + st) * f + fi) * 16), *vt);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared::rng::SplitMix64;

    fn random_q(len: usize, seed: u64) -> Vec<i8> {
        let mut rng = SplitMix64::new(seed);
        (0..len)
            .map(|i| match i % 11 {
                0 => -128,
                1 => 127,
                2 => 0,
                3 => -1,
                _ => rng.next_u64() as i8,
            })
            .collect()
    }

    #[test]
    fn scalar_planes_round_trip_with_zero_padding() {
        for accum in [Accum::Udot, Accum::Uadalp] {
            for (m, k, f) in [(1, 1, 1), (3, 7, 2), (5, 129, 4), (6, 1031, 3), (7, 4096, 6), (2, 16, 4)] {
                let q = random_q(m * k, (m * k + f) as u64);
                let tiles = m.div_ceil(f);
                let mut buf = vec![0xa5u8; tiles * plane_tile_bytes(accum, k, f)];
                pack_planes_scalar(&q, m, k, accum, f, 0..tiles, &mut buf);
                let u: Vec<u8> = q.iter().map(|&v| (v as u8) ^ 0x80).collect();
                assert_eq!(unpack_planes(&buf, m, k, accum, f), u, "{accum:?} m={m} k={k} f={f}");
                // Every bit of every padded position and padded frame-part is zero.
                let s = step_positions(accum);
                let steps = k.div_ceil(s);
                for tile in 0..tiles {
                    for st in 0..steps {
                        for fi in 0..f {
                            let block = &buf[((tile * steps + st) * f + fi) * s..][..s];
                            for pos in 0..s {
                                let (p, mm) = (st * s + pos, tile * f + fi);
                                if p < k && mm < m {
                                    continue;
                                }
                                for r in 0..8 {
                                    let byte = match accum {
                                        Accum::Udot | Accum::UdotLane => block[r * 16 + pos / 8],
                                        Accum::Uadalp => block[2 * r + pos / 8],
                                    };
                                    assert_eq!((byte >> (pos % 8)) & 1, 0, "{accum:?} padding bit set");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_planes_equal_scalar_planes() {
        for accum in [Accum::Udot, Accum::Uadalp] {
            for (m, k, f) in [(1, 1, 1), (3, 7, 2), (5, 129, 4), (6, 1031, 3), (7, 4096, 6), (9, 1024, 4), (4, 255, 1), (5, 384, 2)] {
                let q = random_q(m * k, (7 * m + k) as u64);
                let tiles = m.div_ceil(f);
                let len = tiles * plane_tile_bytes(accum, k, f);
                let mut a = vec![0x5au8; len];
                let mut b = vec![0xa5u8; len];
                pack_planes_scalar(&q, m, k, accum, f, 0..tiles, &mut a);
                unsafe { neon::pack_planes(q.as_ptr(), m, k, accum, f, 0..tiles, b.as_mut_ptr()) };
                assert!(a == b, "{accum:?} m={m} k={k} f={f}");
            }
        }
    }

    #[test]
    fn weights_unpack_to_signs_with_masked_padding() {
        for accum in [Accum::Udot, Accum::Uadalp] {
            for o in [4, 8] {
                for (n, k) in [(1, 1), (5, 9), (9, 127), (13, 1031), (8, 4096)] {
                    let m = BinaryMatrix::random(n, k, (n * k) as u64);
                    let w = pack_weights(&m, accum, o);
                    assert_eq!(unpack_weights(&w), m.signs(), "{accum:?} O={o} {n}x{k}");
                    let s = step_positions(accum);
                    let wb = s / 8;
                    // Positions >= K and rows >= N hold zero n bits.
                    for row in 0..w.tiles * o {
                        for pos in 0..w.steps * s {
                            if row < n && pos < k {
                                continue;
                            }
                            let byte = w.codes[(((row / o) * w.steps + pos / s) * o + row % o) * wb + (pos % s) / 8];
                            assert_eq!((byte >> (pos % 8)) & 1, 0);
                        }
                    }
                    for row in 0..n {
                        assert_eq!(w.corr[row], 127 * k as u32 + m.popcount(row));
                    }
                }
            }
        }
    }
}
