//! Exact bit-serial INT8 x ternary GEMM, injected only into a benchmark snapshot.
//! Quantization and the FP32 epilogue are identical to LocalFlow's VNNI path.
//! Eight u64 lanes hold eight outputs over 64 input positions. Weights retain
//! two bits per coefficient; activations retain eight bits per component.

use crate::{PackedTernary, QuantizedActs, SendPtr, ThreadPool, split_range};
use std::collections::HashMap;

const INV: [f32; 3] = [1.0, 1.0 / 254.0, 1.0 / (254.0 * 254.0)];

pub struct BitWeights {
    rows: usize,
    cols: usize,
    // [output / 8][K / 64][positive-or-active, negative][output % 8]
    words: Vec<u64>,
    correction: Vec<i64>,
}

impl BitWeights {
    pub fn new(w: &PackedTernary, masked: bool, swar: bool) -> Self {
        let (rows, cols) = (w.rows(), w.cols());
        assert!(rows.is_multiple_of(64) && cols.is_multiple_of(64));
        let mut words = vec![0u64; rows * cols / if swar { 8 } else { 32 }];
        let mut correction = vec![0i64; rows];
        for o in 0..rows {
            let block = w.block(o / 64);
            let mut np = 0;
            let mut nn = 0;
            for k in 0..cols {
                let code = (block[(k / 4) * 64 + o % 64] >> (2 * (k % 4))) & 3;
                assert!(code != 3, "invalid ternary code");
                if swar {
                    let b = match code {
                        0 => 1u64,
                        1 => 2,
                        2 => 0,
                        _ => unreachable!(),
                    };
                    words[((o / 8) * (cols / 8) + k / 8) * 8 + o % 8] |= b << (8 * (7 - k % 8));
                    continue;
                }
                let base = ((o / 8) * (cols / 64) + k / 64) * 16 + o % 8;
                if code == 1 {
                    words[base] |= 1 << (k % 64);
                    np += 1;
                } else if code == 2 {
                    words[base + 8] |= 1 << (k % 64);
                    if masked {
                        words[base] |= 1 << (k % 64);
                    }
                    nn += 1;
                }
            }
            // q is already stored as u = q + 128. For masked counts:
            // C = sum(u * positive + (255-u) * negative), so
            // sum(q*w) = C - 128*npositive - 127*nnegative.
            correction[o] = if swar {
                128 * i64::from(w.row_sum()[o])
            } else if masked {
                128 * np + 127 * nn
            } else {
                128 * (np - nn)
            };
        }
        Self {
            rows,
            cols,
            words,
            correction,
        }
    }
}

#[derive(Default)]
struct BitActs {
    words: Vec<u64>,
    rows: usize,
    components: usize,
    cols: usize,
    sums: Vec<i64>,
}

impl BitActs {
    fn pack(&mut self, pool: &ThreadPool, q: &QuantizedActs) {
        let (rows, components, cols) = (q.rows(), q.components(), q.cols());
        assert!(cols.is_multiple_of(64));
        self.words.resize(rows * components * cols / 8, 0);
        self.rows = rows;
        self.components = components;
        self.cols = cols;
        let out = SendPtr(self.words.as_mut_ptr());
        pool.run(&|thread| {
            for r in split_range(rows * components, pool.threads(), thread) {
                unsafe {
                    pack_row(
                        q.q().as_ptr().add(r * cols),
                        out.get().add(r * cols / 8),
                        cols,
                    );
                }
            }
        });
    }

    fn pack_sums(&mut self, pool: &ThreadPool, q: &QuantizedActs) {
        self.rows = q.rows();
        self.components = q.components();
        self.cols = q.cols();
        self.sums.resize(q.rows() * q.components(), 0);
        let output = SendPtr(self.sums.as_mut_ptr());
        pool.run(&|thread| {
            for r in split_range(q.rows() * q.components(), pool.threads(), thread) {
                unsafe {
                    *output.get().add(r) = sum_row(q.q().as_ptr().add(r * q.cols()), q.cols());
                }
            }
        });
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn sum_row(q: *const u8, cols: usize) -> i64 {
    use std::arch::x86_64::*;
    let mut sums = _mm512_setzero_si512();
    for k in (0..cols).step_by(64) {
        let values = unsafe { _mm512_loadu_si512(q.add(k).cast()) };
        sums = _mm512_add_epi64(sums, _mm512_sad_epu8(values, _mm512_setzero_si512()));
    }
    _mm512_reduce_add_epi64(sums)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn pack_row(q: *const u8, out: *mut u64, cols: usize) {
    use std::arch::x86_64::*;
    for k in (0..cols).step_by(64) {
        let v = unsafe { _mm512_loadu_si512(q.add(k).cast()) };
        for b in 0..8 {
            let bit = _mm512_set1_epi8((1u8 << b) as i8);
            let plane = _mm512_test_epi8_mask(v, bit);
            unsafe {
                *out.add((k / 64) * 8 + b) = plane;
            }
        }
    }
}

pub struct Engine {
    masked: bool,
    swar: bool,
    frames: usize,
    weights: HashMap<usize, BitWeights>,
    slots: [BitActs; 3],
}

impl Engine {
    pub fn from_env() -> Option<Self> {
        let name = std::env::var("LF_BITPLANE_BACKEND").unwrap_or_else(|_| "vnni".into());
        if name == "vnni" {
            return None;
        }
        assert!(matches!(name.as_str(), "pn" | "masked" | "swar"));
        assert!(std::arch::is_x86_feature_detected!("avx512vpopcntdq"));
        assert!(std::arch::is_x86_feature_detected!("avx512dq"));
        let frames = std::env::var("LF_BITPLANE_FRAMES")
            .unwrap_or_else(|_| "2".into())
            .parse()
            .unwrap();
        assert!(matches!(frames, 1 | 2) || (name == "swar" && matches!(frames, 4 | 8)));
        Some(Self {
            masked: name == "masked",
            swar: name == "swar",
            frames,
            weights: HashMap::new(),
            slots: std::array::from_fn(|_| BitActs::default()),
        })
    }

    pub fn gemm(
        &mut self,
        pool: &ThreadPool,
        w: &PackedTernary,
        q: &QuantizedActs,
        y: &mut [f32],
        slot: usize,
        reuse: bool,
    ) {
        assert!(crate::cpu_supported());
        assert!(slot < 3 && q.cols() == w.cols());
        assert!(y.len() >= q.rows() * w.rows());
        let weights = self
            .weights
            .entry(w as *const _ as usize)
            .or_insert_with(|| BitWeights::new(w, self.masked, self.swar));
        let acts = &mut self.slots[slot];
        if !reuse {
            if self.swar {
                acts.pack_sums(pool, q);
            } else {
                acts.pack(pool, q);
            }
        }
        assert_eq!(
            (acts.rows, acts.components, acts.cols),
            (q.rows(), q.components(), q.cols())
        );
        let output = SendPtr(y.as_mut_ptr());
        pool.run(&|thread| {
            for block in split_range(weights.rows / 64, pool.threads(), thread) {
                for group in block * 8..(block + 1) * 8 {
                    let mut frame = 0;
                    while frame < q.rows() {
                        let count = self.frames.min(q.rows() - frame);
                        unsafe {
                            if self.swar {
                                match count {
                                    1 => swar_kernel::<1>(
                                        weights,
                                        acts,
                                        q,
                                        w,
                                        group,
                                        frame,
                                        output.get(),
                                    ),
                                    2 => swar_kernel::<2>(
                                        weights,
                                        acts,
                                        q,
                                        w,
                                        group,
                                        frame,
                                        output.get(),
                                    ),
                                    3 => swar_kernel::<3>(
                                        weights,
                                        acts,
                                        q,
                                        w,
                                        group,
                                        frame,
                                        output.get(),
                                    ),
                                    4 => swar_kernel::<4>(
                                        weights,
                                        acts,
                                        q,
                                        w,
                                        group,
                                        frame,
                                        output.get(),
                                    ),
                                    5 => swar_kernel::<5>(
                                        weights,
                                        acts,
                                        q,
                                        w,
                                        group,
                                        frame,
                                        output.get(),
                                    ),
                                    6 => swar_kernel::<6>(
                                        weights,
                                        acts,
                                        q,
                                        w,
                                        group,
                                        frame,
                                        output.get(),
                                    ),
                                    7 => swar_kernel::<7>(
                                        weights,
                                        acts,
                                        q,
                                        w,
                                        group,
                                        frame,
                                        output.get(),
                                    ),
                                    8 => swar_kernel::<8>(
                                        weights,
                                        acts,
                                        q,
                                        w,
                                        group,
                                        frame,
                                        output.get(),
                                    ),
                                    _ => unreachable!(),
                                }
                            } else {
                                match (self.masked, count) {
                                    (false, 1) => kernel::<false, 1>(
                                        weights,
                                        acts,
                                        q,
                                        w,
                                        group,
                                        frame,
                                        output.get(),
                                    ),
                                    (false, 2) => kernel::<false, 2>(
                                        weights,
                                        acts,
                                        q,
                                        w,
                                        group,
                                        frame,
                                        output.get(),
                                    ),
                                    (true, 1) => kernel::<true, 1>(
                                        weights,
                                        acts,
                                        q,
                                        w,
                                        group,
                                        frame,
                                        output.get(),
                                    ),
                                    (true, 2) => kernel::<true, 2>(
                                        weights,
                                        acts,
                                        q,
                                        w,
                                        group,
                                        frame,
                                        output.get(),
                                    ),
                                    _ => unreachable!(),
                                }
                            }
                        }
                        frame += count;
                    }
                }
            }
        });
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx2,fma")]
// Const-sized indexed loops deliberately unroll into accumulator registers.
#[allow(clippy::needless_range_loop)]
unsafe fn swar_kernel<const FRAMES: usize>(
    weights: &BitWeights,
    acts: &BitActs,
    q: &QuantizedActs,
    w: &PackedTernary,
    group: usize,
    frame: usize,
    output: *mut f32,
) {
    use std::arch::x86_64::*;
    let chunks = weights.cols / 8;
    let mut sum = [_mm256_setzero_ps(); FRAMES];
    let correction =
        unsafe { _mm512_loadu_si512(weights.correction.as_ptr().add(group * 8).cast()) };
    let scale = unsafe { _mm256_loadu_ps(w.scale().as_ptr().add(group * 8)) };
    let mask = _mm512_set1_epi64(0x0f0f0f0f0f0f0f0f);
    for c in (0..q.components()).rev() {
        let mut acc = [_mm512_setzero_si512(); FRAMES];
        for k in 0..chunks {
            let wp = unsafe { weights.words.as_ptr().add((group * chunks + k) * 8) };
            let b = unsafe { _mm512_loadu_si512(wp.cast()) };
            for r in 0..FRAMES {
                let qp = unsafe {
                    q.q()
                        .as_ptr()
                        .add(((frame + r) * q.components() + c) * q.cols() + k * 8)
                };
                let a = _mm512_set1_epi64(unsafe { qp.cast::<i64>().read_unaligned() });
                let lo = _mm512_and_si512(a, mask);
                let hi = _mm512_and_si512(_mm512_srli_epi64::<4>(a), mask);
                // Byte fields never carry: every convolution coefficient
                // is bounded by 8*15*2 = 240. Extract every product before
                // accumulating; combining products first would overflow.
                let cl = _mm512_srli_epi64::<56>(_mm512_mullo_epi64(lo, b));
                let ch = _mm512_srli_epi64::<56>(_mm512_mullo_epi64(hi, b));
                let dot = _mm512_add_epi64(cl, _mm512_slli_epi64::<4>(ch));
                acc[r] = _mm512_add_epi64(acc[r], dot);
            }
        }
        for r in 0..FRAMES {
            let offset = _mm512_add_epi64(
                correction,
                _mm512_set1_epi64(acts.sums[(frame + r) * q.components() + c]),
            );
            let d = _mm512_cvtepi64_ps(_mm512_sub_epi64(acc[r], offset));
            sum[r] = _mm256_fmadd_ps(d, _mm256_set1_ps(INV[c]), sum[r]);
        }
    }
    for r in 0..FRAMES {
        let out = _mm256_mul_ps(
            _mm256_mul_ps(sum[r], scale),
            _mm256_set1_ps(q.scale()[frame + r]),
        );
        unsafe {
            _mm256_storeu_ps(output.add((frame + r) * weights.rows + group * 8), out);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vpopcntdq,avx2,fma")]
// Keep the same explicit accumulator indexing as LocalFlow's VNNI kernel.
#[allow(clippy::needless_range_loop)]
unsafe fn kernel<const MASKED: bool, const FRAMES: usize>(
    weights: &BitWeights,
    acts: &BitActs,
    q: &QuantizedActs,
    w: &PackedTernary,
    group: usize,
    frame: usize,
    output: *mut f32,
) {
    use std::arch::x86_64::*;
    let chunks = weights.cols / 64;
    let mut sum = [_mm256_setzero_ps(); FRAMES];
    let correction =
        unsafe { _mm512_loadu_si512(weights.correction.as_ptr().add(group * 8).cast()) };
    let scale = unsafe { _mm256_loadu_ps(w.scale().as_ptr().add(group * 8)) };
    // A component at a time keeps the eight-plane accumulators in registers.
    // Count whole K first; bit significance is combined only once at the end.
    for component in (0..q.components()).rev() {
        let mut counts = [[_mm512_setzero_si512(); 8]; FRAMES];
        for chunk in 0..chunks {
            let wp = unsafe { weights.words.as_ptr().add((group * chunks + chunk) * 16) };
            let p = unsafe { _mm512_loadu_si512(wp.cast()) };
            let n = unsafe { _mm512_loadu_si512(wp.add(8).cast()) };
            for r in 0..FRAMES {
                let ap = unsafe {
                    acts.words
                        .as_ptr()
                        .add((((frame + r) * q.components() + component) * chunks + chunk) * 8)
                };
                for b in 0..8 {
                    let a = _mm512_set1_epi64(unsafe { *ap.add(b) } as i64);
                    let c = if MASKED {
                        // LUT 0x28 is (a XOR n) AND p; p is the active mask.
                        _mm512_popcnt_epi64(_mm512_ternarylogic_epi64::<0x28>(a, n, p))
                    } else {
                        _mm512_sub_epi64(
                            _mm512_popcnt_epi64(_mm512_and_si512(a, p)),
                            _mm512_popcnt_epi64(_mm512_and_si512(a, n)),
                        )
                    };
                    counts[r][b] = _mm512_add_epi64(counts[r][b], c);
                }
            }
        }
        for r in 0..FRAMES {
            let mut dot = counts[r][0];
            for b in 1..8 {
                dot = _mm512_add_epi64(
                    dot,
                    _mm512_sllv_epi64(counts[r][b], _mm512_set1_epi64(b as i64)),
                );
            }
            let d = _mm512_cvtepi64_ps(_mm512_sub_epi64(dot, correction));
            sum[r] = _mm256_fmadd_ps(d, _mm256_set1_ps(INV[component]), sum[r]);
        }
    }
    for r in 0..FRAMES {
        let fs = _mm256_set1_ps(q.scale()[frame + r]);
        let y = _mm256_mul_ps(_mm256_mul_ps(sum[r], scale), fs);
        unsafe {
            _mm256_storeu_ps(output.add((frame + r) * weights.rows + group * 8), y);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Scratch, gemm_i8};
    use lf_model::TernaryMatrix;

    #[test]
    fn exact_vnni_match_components_tails_extremes_and_reuse() {
        if !crate::cpu_supported() || !std::arch::is_x86_feature_detected!("avx512vpopcntdq") {
            return;
        }
        let cpu = std::fs::read_to_string("/proc/self/status").unwrap();
        let list = cpu
            .lines()
            .find(|l| l.starts_with("Cpus_allowed_list:"))
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap();
        let first: usize = list.split([',', '-']).next().unwrap().parse().unwrap();
        let pool = ThreadPool::new(&[first]);
        for (rows, cols, frames) in [(64, 64, 1), (128, 128, 3), (192, 1024, 5), (64, 4096, 2)] {
            let m = TernaryMatrix::random(rows, cols, (rows + cols) as u64);
            let w = PackedTernary::new(&m).unwrap();
            let x: Vec<f32> = (0..frames * cols)
                .map(|i| match i % 7 {
                    0 => -127.0,
                    1 => 127.0,
                    2 => 0.0,
                    _ => ((i * 129 % 65521) as f32 - 32000.0) / 137.0,
                })
                .collect();
            for components in 1..=3 {
                let mut q = QuantizedActs::with_capacity(frames, cols, components);
                q.quantize(&pool, &x, frames).unwrap();
                let scratch = Scratch::new(1, gemm_i8::scratch_bytes(cols));
                let mut reference = vec![0.0; rows * frames];
                gemm_i8::gemm(&pool, &scratch, &w, &q, &mut reference);
                for (masked, swar) in [(false, false), (true, false), (false, true)] {
                    for tile in if swar { vec![1, 2, 4, 8] } else { vec![1, 2] } {
                        let mut engine = Engine {
                            masked,
                            swar,
                            frames: tile,
                            weights: HashMap::new(),
                            slots: std::array::from_fn(|_| BitActs::default()),
                        };
                        let mut y = vec![0.0; rows * frames];
                        for reuse in [false, true] {
                            engine.gemm(&pool, &w, &q, &mut y, 0, reuse);
                            assert!(
                                y.iter()
                                    .zip(&reference)
                                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                                "masked={masked}, components={components}, tile={tile}, shape={rows},{cols},{frames}"
                            );
                        }
                    }
                }
            }
        }
    }
}
