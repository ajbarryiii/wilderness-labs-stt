//! Prepare-time weight layout: the export's 1-bit codes, re-tiled, never expanded.
//!
//! Outputs are padded to a multiple of [`N_ALIGN`] (zero codes; never stored) and grouped in
//! blocks of 16. For block `b` and code byte `p` (inputs `8p .. 8p + 8`, i.e. table groups `2p`
//! and `2p + 1`), the 16 bytes at `codes[(b * kb + p) * 16 ..][..16]` are byte `p` of rows
//! `16b .. 16b + 16`: one 16-byte load gives the index nibbles of 16 outputs for two groups
//! (low nibble = group `2p`, high nibble = group `2p + 1`, bit `i % 4` of a nibble = input `i`).
//! Storage is exactly the export's `N * ceil(K / 8)` bytes when `N % 32 == 0` (every model shape).

use shared::BinaryMatrix;
use shared::aligned::AlignedVec;

/// Output padding: the widest out-block (V = 4 index vectors of 8 outputs).
pub const N_ALIGN: usize = 32;

pub struct Codes {
    pub n: usize,
    pub k: usize,
    pub n_pad: usize,
    /// Code bytes per row, `ceil(K / 8)`.
    pub kb: usize,
    /// `[n_pad / 16][kb][16]`.
    pub codes: AlignedVec<u8>,
    /// Weight scales, zero-padded to `n_pad`.
    pub scale: Vec<f32>,
}

pub fn tile(m: &BinaryMatrix) -> Codes {
    m.validate().expect("t_lut: invalid binary matrix");
    let (n, k) = (m.n, m.k);
    let kb = m.row_bytes();
    let n_pad = n.div_ceil(N_ALIGN) * N_ALIGN;
    let mut codes = AlignedVec::<u8>::zeroed(n_pad * kb);
    {
        let dst = codes.as_mut_slice();
        for o in 0..n {
            let (b, l) = (o / 16, o % 16);
            for p in 0..kb {
                dst[(b * kb + p) * 16 + l] = m.codes[o * kb + p];
            }
        }
    }
    let mut scale = m.scale.clone();
    scale.resize(n_pad, 0.0);
    Codes { n, k, n_pad, kb, codes, scale }
}

/// `[N][K]` signs from the tiled codes (the inverse of [`tile`]).
pub fn unpack(c: &Codes) -> Vec<i8> {
    let src = c.codes.as_slice();
    let mut out = Vec::with_capacity(c.n * c.k);
    for o in 0..c.n {
        let (b, l) = (o / 16, o % 16);
        for i in 0..c.k {
            let byte = src[(b * c.kb + i / 8) * 16 + l];
            out.push(if (byte >> (i % 8)) & 1 == 1 { 1 } else { -1 });
        }
    }
    out
}
