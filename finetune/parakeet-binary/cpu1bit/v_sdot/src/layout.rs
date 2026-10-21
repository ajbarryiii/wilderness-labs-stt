//! The v-layout of 1-bit codes (see `kernel.rs` for how the kernel reads it) and its inverse.
//!
//! Input: MB1's `parakeet-binary-v1` codes, `[n][ceil(k/8)]` bytes, code `j` of a row at bit
//! `j % 8` of byte `j / 8` (LSB first), 1 = +1, 0 = -1, tail bits zero.
//!
//! Output rows are grouped by 4 (`groups = ceil(n / 4)`) and inputs chunked by 32
//! (`chunks = ceil(k / 32)`). Group `g`, chunk `c` is 16 bytes at `(g * chunks + c) * 16`; its byte
//! `4 nl + kl`, bit `j` holds the code of row `4g + nl`, input `32c + 4j + kl`. Rows `>= n` and
//! inputs `>= k` are 0 bits: they never reach an output (rows) or meet a zero activation (inputs;
//! the kernel reads zero-padded activations whenever `k % 32 != 0`). For the model's shapes
//! (n % 4 == 0, k % 32 == 0) this is a pure permutation of the export's bits: exactly
//! `n * k / 8` bytes.

pub const GROUP_ROWS: usize = 4;
pub const CHUNK_K: usize = 32;
pub const CHUNK_BYTES: usize = 16;

/// Code bit (0/1) of row `r`, input `i` in MB1's row-major LSB-first layout.
#[inline]
pub fn code_bit(codes: &[u8], row_bytes: usize, r: usize, i: usize) -> u8 {
    (codes[r * row_bytes + i / 8] >> (i % 8)) & 1
}

/// The v-layout of an `n x k` matrix, padded to `groups` groups and `chunks` chunks.
pub fn build_main(codes: &[u8], row_bytes: usize, n: usize, k: usize, groups: usize, chunks: usize) -> Vec<u8> {
    assert!(groups * GROUP_ROWS >= n && chunks * CHUNK_K >= k && codes.len() >= n * row_bytes);
    let mut out = vec![0u8; groups * chunks * CHUNK_BYTES];
    for g in 0..groups {
        for c in 0..chunks {
            let dst = &mut out[(g * chunks + c) * CHUNK_BYTES..][..CHUNK_BYTES];
            for nl in 0..GROUP_ROWS {
                let r = g * GROUP_ROWS + nl;
                if r >= n {
                    continue;
                }
                for kl in 0..4 {
                    let mut byte = 0u8;
                    for j in 0..8 {
                        let i = c * CHUNK_K + 4 * j + kl;
                        if i < k {
                            byte |= code_bit(codes, row_bytes, r, i) << j;
                        }
                    }
                    dst[4 * nl + kl] = byte;
                }
            }
        }
    }
    out
}

/// Inverse of [`build_main`]: the stored bit of (row, input) for any row < 4 groups and input
/// < 32 chunks.
#[inline]
pub fn main_bit(main: &[u8], chunks: usize, r: usize, i: usize) -> u8 {
    let (g, nl) = (r / GROUP_ROWS, r % GROUP_ROWS);
    let (c, rem) = (i / CHUNK_K, i % CHUNK_K);
    let (j, kl) = (rem / 4, rem % 4);
    (main[(g * chunks + c) * CHUNK_BYTES + 4 * nl + kl] >> j) & 1
}
