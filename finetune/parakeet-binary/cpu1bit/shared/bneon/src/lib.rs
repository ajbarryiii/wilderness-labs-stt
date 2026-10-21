//! B-NEON: the baseline (plans/cpu-benchmark-mac.md "Arms", M7), a NEON port of LocalFlow's x86
//! method (lf-cpu/src/gemm_i8.rs + pack.rs of the measured snapshot), structure for structure:
//! - prepare: the 2-bit codes are repacked into output blocks of NB rows, block-major, quad-major
//!   inside a block (LocalFlow `PackedTernary`: `tiled[(b * K/4 + q) * NB + j] = codes[b*NB + j][q]`);
//!   codes stay 2 bits per weight in memory;
//! - per call, each pool thread takes whole blocks (`split_range` over blocks); per block it expands
//!   the codes once into an s8 tile (NB x K bytes, L2-resident) and streams every frame past it;
//! - the inner product is SDOT (s8 x s8): activations are the s8 parts, so no +128 offset and no
//!   row-sum correction are needed (LocalFlow needs them for VPDPBUSD's u8 operand);
//! - LocalFlow's FP32 epilogue in its exact order (shared::epilogue::neon::finish4).
//!
//! NEON-specific (allowed by M7): the tile is laid out micro-tile-major, `[NB/NR][K/4][NR][4]`, and a
//! register-blocked micro-kernel computes FR frames x C components x NR outputs with SDOT by element
//! (`vdotq_laneq_s32`: one 16-byte activation register serves four K-quads), accumulators
//! FR*C*NR/4 + activations FR*C + weights NR/4 <= 29 of the 32 vector registers.
//!
//! Variants (same micro-kernel, same tile):
//! - `BNeon` ("b-neon"): from parakeet-ternary-v1 2-bit codes (R-M1, and R-MB1 via the derived export);
//! - `BNeon1BitTile` ("b-neon-1bit-tile"): from parakeet-binary-v1 1-bit codes expanded into the same
//!   tile (isolates storage from kernel design; descriptive);
//! - `BNeonPreExpanded` ("b-neon-pre-expanded"): the whole matrix stored as the s8 tile, no
//!   per-call expansion — the M7 ceiling control for this arithmetic, not an arm.
//!
//! Tuning (KernelConfig.tuning): {"nb": 32|64|128, "nr": 8|16}; frames per micro-kernel call are
//! the register-budget maximum for (C, NR) (`max_frames`).

use shared::aligned::AlignedVec;
use shared::kernel::{Call, Kernel, KernelConfig, Out, WeightBytes};
use shared::pool::{PerThread, ThreadPool};
use shared::quant::QuantizedActs;
use shared::weights::{WeightFormat, WeightsRef};
use shared::{SendPtr, split_range};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tuning {
    /// Output rows per expanded block (LocalFlow: 64).
    pub nb: usize,
    /// Output rows per micro-tile (4 per vector register).
    pub nr: usize,
}

impl Default for Tuning {
    fn default() -> Self {
        Tuning { nb: 64, nr: 16 }
    }
}

impl Tuning {
    pub fn from_json(v: &serde_json::Value) -> Tuning {
        let d = Tuning::default();
        let t = Tuning {
            nb: v.get("nb").and_then(|x| x.as_u64()).map_or(d.nb, |x| x as usize),
            nr: v.get("nr").and_then(|x| x.as_u64()).map_or(d.nr, |x| x as usize),
        };
        assert!(matches!(t.nb, 16 | 32 | 64 | 128 | 256), "nb {}", t.nb);
        assert!(matches!(t.nr, 8 | 16), "nr {}", t.nr);
        assert!(t.nb % t.nr == 0 && t.nb % 16 == 0);
        t
    }
}

/// Frames per micro-kernel call: the most that fit the register budget
/// FR*C*(NR/4) + FR*C + NR/4 <= 29.
pub fn max_frames(c: usize, nr: usize) -> usize {
    match (c, nr) {
        (1, 16) => 5,
        (2, 16) => 2,
        (3, 16) => 1,
        (1, 8) => 9,
        (2, 8) => 4,
        (3, 8) => 3,
        _ => panic!("no micro-kernel for C={c} NR={nr}"),
    }
}

/// Weight storage of a prepared matrix.
pub enum Codes {
    /// 2-bit codes, `[block][quad][NB]` bytes (LocalFlow's block layout).
    Two(Vec<u8>),
    /// 1-bit codes, `[block][oct][NB]` bytes, LSB = lowest k.
    One(Vec<u8>),
    /// The expanded s8 tiles of every block (pre-expanded control).
    Expanded(AlignedVec<i8>),
}

pub struct Prepared {
    pub n: usize,
    pub k: usize,
    /// K rounded up to 16 (the SDOT step); padded weights multiply zero activations.
    pub kpad: usize,
    /// N rounded up to NB.
    pub npad: usize,
    pub tuning: Tuning,
    pub codes: Codes,
    /// Weight scales padded with zeros to npad.
    pub scale: Vec<f32>,
}

impl Prepared {
    fn blocks(&self) -> usize {
        self.npad / self.tuning.nb
    }
    fn tile_bytes(&self) -> usize {
        self.tuning.nb * self.kpad
    }
    /// Offset of (micro-tile mt, quad q, output-in-micro-tile i, m) inside a block's tile.
    fn tile_index(&self, mt: usize, q: usize, i: usize, m: usize) -> usize {
        let nr = self.tuning.nr;
        mt * (self.kpad * nr) + q * nr * 4 + i * 4 + m
    }
}

/// MODE 0: 2-bit codes; 1: 1-bit codes, same tile; 2: pre-expanded control.
pub struct BNeonImpl<const MODE: u8> {
    tuning: Tuning,
    tiles: PerThread<AlignedVec<i8>>,
    /// Zero-padded activation copy for K % 16 != 0 (contract shapes only; model shapes never pad).
    apad: AlignedVec<i8>,
    tile_bytes: usize,
}

pub type BNeon = BNeonImpl<0>;
pub type BNeon1BitTile = BNeonImpl<1>;
pub type BNeonPreExpanded = BNeonImpl<2>;

fn round_up(x: usize, m: usize) -> usize {
    x.div_ceil(m) * m
}

impl<const MODE: u8> Kernel for BNeonImpl<MODE> {
    type Prepared = Prepared;
    const NAME: &'static str = match MODE {
        0 => "b-neon",
        1 => "b-neon-1bit-tile",
        _ => "b-neon-pre-expanded",
    };
    const FORMAT: WeightFormat = if MODE == 1 { WeightFormat::Binary } else { WeightFormat::Ternary };

    fn new(cfg: &KernelConfig) -> Self {
        let tuning = Tuning::from_json(&cfg.tuning);
        let kpad = round_up(cfg.max_k.max(1), 16);
        let tile_bytes = if MODE == 2 { 0 } else { tuning.nb * kpad };
        BNeonImpl {
            tuning,
            tiles: PerThread::new(cfg.threads, |_| AlignedVec::zeroed(tile_bytes)),
            apad: AlignedVec::zeroed(0),
            tile_bytes,
        }
    }

    fn prepare(&self, w: WeightsRef<'_>) -> Prepared {
        let (n, k) = (w.n(), w.k());
        let t = self.tuning;
        let kpad = round_up(k, 16);
        let npad = round_up(n, t.nb);
        let blocks = npad / t.nb;
        let mut scale = w.scale().to_vec();
        scale.resize(npad, 0.0);
        let two = |m: &shared::TernaryMatrix| {
            let (quads, rb) = (kpad / 4, m.row_bytes());
            let mut c = vec![0u8; blocks * quads * t.nb];
            for b in 0..blocks {
                for q in 0..quads.min(rb) {
                    for j in 0..t.nb {
                        let o = b * t.nb + j;
                        if o < n {
                            c[(b * quads + q) * t.nb + j] = m.codes[o * rb + q];
                        }
                    }
                }
            }
            c
        };
        let codes = match (MODE, w) {
            (0, WeightsRef::Ternary(m)) => Codes::Two(two(m)),
            (1, WeightsRef::Binary(m)) => {
                let (octs, rb) = (kpad / 8, m.row_bytes());
                let mut c = vec![0u8; blocks * octs * t.nb];
                for b in 0..blocks {
                    for q8 in 0..octs.min(rb) {
                        for j in 0..t.nb {
                            let o = b * t.nb + j;
                            if o < n {
                                c[(b * octs + q8) * t.nb + j] = m.codes[o * rb + q8];
                            }
                        }
                    }
                }
                Codes::One(c)
            }
            (2, WeightsRef::Ternary(m)) => {
                let packed = two(m);
                let mut e: AlignedVec<i8> = AlignedVec::zeroed(blocks * t.nb * kpad);
                let quads = kpad / 4;
                for b in 0..blocks {
                    unsafe {
                        expand_two(packed.as_ptr().add(b * quads * t.nb), quads, t.nb, t.nr, e.as_mut_ptr().add(b * t.nb * kpad))
                    };
                }
                Codes::Expanded(e)
            }
            _ => panic!("{}: wrong weight format", Self::NAME),
        };
        Prepared { n, k, kpad, npad, tuning: t, codes, scale }
    }

    fn gemm(&mut self, pool: &ThreadPool, w: &Prepared, qa: &QuantizedActs, _call: Call, out: Out<'_>) {
        assert_eq!(qa.cols(), w.k, "K mismatch");
        assert_eq!(w.tuning, self.tuning, "prepared with another tuning");
        let (t_n, c_n, n) = (qa.rows(), qa.components(), w.n);
        let (y, d, dots) = match out {
            Out::F32(y) => {
                assert!(y.len() >= t_n * n);
                (SendPtr(y.as_mut_ptr()), SendPtr(std::ptr::null_mut()), false)
            }
            Out::Dots(d) => {
                assert!(d.len() >= t_n * c_n * n);
                (SendPtr(std::ptr::null_mut()), SendPtr(d.as_mut_ptr()), true)
            }
        };
        if t_n == 0 {
            return;
        }
        assert!(self.tiles.len() >= pool.threads(), "kernel built for fewer threads than the pool");
        let a: SendPtr<i8> = if w.k == w.kpad {
            SendPtr(qa.q().as_ptr() as *mut i8)
        } else {
            let need = t_n * c_n * w.kpad;
            if self.apad.len() < need {
                self.apad = AlignedVec::zeroed(need);
            }
            let dst = self.apad.as_mut_slice();
            for r in 0..t_n * c_n {
                dst[r * w.kpad..r * w.kpad + w.k].copy_from_slice(&qa.q()[r * w.k..(r + 1) * w.k]);
                dst[r * w.kpad + w.k..(r + 1) * w.kpad].fill(0);
            }
            SendPtr(self.apad.as_mut_ptr())
        };
        let t = w.tuning;
        let threads = pool.threads();
        let fr_max = max_frames(c_n, t.nr);
        let tiles = &self.tiles;
        let tile_bytes = self.tile_bytes;
        let frame_scale = SendPtr(qa.scale().as_ptr() as *mut f32);
        pool.run(&|idx| {
            for b in split_range(w.blocks(), threads, idx) {
                let wt: *const i8 = match &w.codes {
                    Codes::Expanded(e) => unsafe { e.as_ptr().add(b * w.tile_bytes()) },
                    Codes::Two(c) => {
                        let tile = unsafe { tiles.get(idx) };
                        debug_assert!(tile.len() >= w.tile_bytes() && tile_bytes >= w.tile_bytes());
                        let quads = w.kpad / 4;
                        unsafe { expand_two(c.as_ptr().add(b * quads * t.nb), quads, t.nb, t.nr, tile.as_mut_ptr()) };
                        tile.as_ptr()
                    }
                    Codes::One(c) => {
                        let tile = unsafe { tiles.get(idx) };
                        debug_assert!(tile.len() >= w.tile_bytes());
                        let octs = w.kpad / 8;
                        unsafe { expand_one(c.as_ptr().add(b * octs * t.nb), octs, t.nb, t.nr, tile.as_mut_ptr()) };
                        tile.as_ptr()
                    }
                };
                for mt in 0..t.nb / t.nr {
                    let o0 = b * t.nb + mt * t.nr;
                    if o0 >= n {
                        break;
                    }
                    let wmt = unsafe { wt.add(mt * w.kpad * t.nr) };
                    let mut f0 = 0;
                    while f0 < t_n {
                        let fr = fr_max.min(t_n - f0);
                        let st = Store {
                            y: y.get(),
                            d: d.get(),
                            n,
                            c: c_n,
                            o0,
                            valid: t.nr.min(n - o0),
                            t0: f0,
                            ws: unsafe { w.scale.as_ptr().add(o0) },
                            fs: unsafe { frame_scale.get().add(f0) },
                        };
                        let f = micro_for(c_n, fr, t.nr / 4, dots);
                        unsafe { f(wmt, a.get().add(f0 * c_n * w.kpad), w.kpad, w.kpad / 16, &st) };
                        f0 += fr;
                    }
                }
            }
        });
    }

    fn unpack(&self, w: &Prepared) -> Vec<i8> {
        let (n, k, nb) = (w.n, w.k, w.tuning.nb);
        let mut out = vec![0i8; n * k];
        for o in 0..n {
            let (b, j) = (o / nb, o % nb);
            for i in 0..k {
                out[o * k + i] = match &w.codes {
                    Codes::Two(c) => {
                        let byte = c[(b * (w.kpad / 4) + i / 4) * nb + j];
                        shared::weights::ternary_field_value(byte >> (2 * (i % 4)))
                    }
                    Codes::One(c) => {
                        if (c[(b * (w.kpad / 8) + i / 8) * nb + j] >> (i % 8)) & 1 == 1 { 1 } else { -1 }
                    }
                    Codes::Expanded(e) => {
                        let (mt, ii) = (j / w.tuning.nr, j % w.tuning.nr);
                        e[b * w.tile_bytes() + w.tile_index(mt, i / 4, ii, i % 4)]
                    }
                };
            }
        }
        out
    }

    fn shape(&self, w: &Prepared) -> (usize, usize) {
        (w.n, w.k)
    }

    fn weight_bytes(&self, w: &Prepared) -> WeightBytes {
        let codes = match &w.codes {
            Codes::Two(c) | Codes::One(c) => c.len(),
            Codes::Expanded(e) => e.len(),
        };
        WeightBytes { codes, per_row: 4 * w.scale.len() }
    }

    fn workspace_bytes(&self) -> usize {
        self.tiles.len() * self.tile_bytes + self.apad.len()
    }

    fn describe(&self) -> serde_json::Value {
        serde_json::json!({"variant": Self::NAME, "nb": self.tuning.nb, "nr": self.tuning.nr,
            "frames_per_call": {"1": max_frames(1, self.tuning.nr), "2": max_frames(2, self.tuning.nr), "3": max_frames(3, self.tuning.nr)},
            "tile_layout": "[NB/NR][K/4][NR][4] s8 per block", "inner": "vdotq_laneq_s32 (SDOT by element)"})
    }
}

/// Output pointers and scales for one micro-kernel call.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub struct Store {
    y: *mut f32,
    d: *mut i32,
    /// Row stride of y (N) and of d per component.
    n: usize,
    c: usize,
    o0: usize,
    /// Valid outputs in this micro-tile (< NR only for the last tile of a ragged N).
    valid: usize,
    t0: usize,
    ws: *const f32,
    fs: *const f32,
}

type Micro = unsafe fn(*const i8, *const i8, usize, usize, &Store);

macro_rules! micro_table {
    ($c:expr, $rows:expr, $nrv:expr, $dots:expr; $(($C:literal, $R:literal, $V:literal)),* $(,)?) => {
        match ($c, $rows, $nrv, $dots) {
            $(
                ($C, $R, $V, false) => micro::<$C, $R, $V, false> as Micro,
                ($C, $R, $V, true) => micro::<$C, $R, $V, true> as Micro,
            )*
            _ => panic!("no micro-kernel for C={} rows={} NR/4={}", $c, $rows, $nrv),
        }
    };
}

fn micro_for(c: usize, frames: usize, nrv: usize, dots: bool) -> Micro {
    micro_table!(c, c * frames, nrv, dots;
        (1, 1, 4), (1, 2, 4), (1, 3, 4), (1, 4, 4), (1, 5, 4),
        (1, 1, 2), (1, 2, 2), (1, 3, 2), (1, 4, 2), (1, 5, 2), (1, 6, 2), (1, 7, 2), (1, 8, 2), (1, 9, 2),
        (2, 2, 4), (2, 4, 4),
        (2, 2, 2), (2, 4, 2), (2, 6, 2), (2, 8, 2),
        (3, 3, 4),
        (3, 3, 2), (3, 6, 2), (3, 9, 2),
    )
}

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// ROWS = frames * C rows of activations (row stride `lda`, `k16` steps of 16) against one
/// micro-tile of NRV*4 outputs. acc[r][v] lane i = output o0 + 4v + i of row r.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn micro<const C: usize, const ROWS: usize, const NRV: usize, const DOTS: bool>(
    w: *const i8,
    a: *const i8,
    lda: usize,
    k16: usize,
    st: &Store,
) {
    debug_assert!(ROWS % C == 0 && st.c == C);
    let mut acc = [[vdupq_n_s32(0); NRV]; ROWS];
    let mut wq = w;
    for kk in 0..k16 {
        let av: [int8x16_t; ROWS] = std::array::from_fn(|r| unsafe { vld1q_s8(a.add(r * lda + kk * 16)) });
        macro_rules! quad {
            ($j:literal) => {{
                let wv: [int8x16_t; NRV] = std::array::from_fn(|v| unsafe { vld1q_s8(wq.add(($j * NRV + v) * 16)) });
                for r in 0..ROWS {
                    for v in 0..NRV {
                        acc[r][v] = vdotq_laneq_s32::<$j>(acc[r][v], wv[v], av[r]);
                    }
                }
            }};
        }
        quad!(0);
        quad!(1);
        quad!(2);
        quad!(3);
        wq = unsafe { wq.add(4 * NRV * 16) };
    }
    let frames = ROWS / C;
    for f in 0..frames {
        let t = st.t0 + f;
        for v in 0..NRV {
            let o = st.o0 + 4 * v;
            let lanes = st.valid.saturating_sub(4 * v).min(4);
            if lanes == 0 {
                break;
            }
            if DOTS {
                for c in 0..C {
                    let dst = unsafe { st.d.add((t * C + c) * st.n + o) };
                    if lanes == 4 {
                        unsafe { vst1q_s32(dst, acc[f * C + c][v]) };
                    } else {
                        let mut tmp = [0i32; 4];
                        unsafe { vst1q_s32(tmp.as_mut_ptr(), acc[f * C + c][v]) };
                        for (l, &x) in tmp.iter().enumerate().take(lanes) {
                            unsafe { *dst.add(l) = x };
                        }
                    }
                }
            } else {
                let parts: [int32x4_t; C] = std::array::from_fn(|c| acc[f * C + c][v]);
                let ws = unsafe { vld1q_f32(st.ws.add(4 * v)) };
                let yv = shared::epilogue::neon::finish4::<C>(parts, ws, unsafe { *st.fs.add(f) });
                let dst = unsafe { st.y.add(t * st.n + o) };
                if lanes == 4 {
                    unsafe { vst1q_f32(dst, yv) };
                } else {
                    let mut tmp = [0f32; 4];
                    unsafe { vst1q_f32(tmp.as_mut_ptr(), yv) };
                    for (l, &x) in tmp.iter().enumerate().take(lanes) {
                        unsafe { *dst.add(l) = x };
                    }
                }
            }
        }
    }
}

/// Byte i*4 + m of vector u takes code byte 4u + i (16 code bytes = 16 outputs).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn replicate_idx() -> [uint8x16_t; 4] {
    std::array::from_fn(|u| {
        let b: [u8; 16] = std::array::from_fn(|p| (4 * u + p / 4) as u8);
        unsafe { vld1q_u8(b.as_ptr()) }
    })
}

/// Expands one block of 2-bit codes (`[quad][nb]` bytes) into the micro-tile-major s8 tile:
/// field m of code byte (output o, quad q) -> tile[o / nr][q][o % nr][m], 00 -> 0, 01 -> +1, 10 -> -1.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn expand_two(codes: *const u8, quads: usize, nb: usize, nr: usize, tile: *mut i8) {
    let idx = replicate_idx();
    let shifts: [i8; 16] = std::array::from_fn(|p| -2 * (p % 4) as i8);
    let shifts = unsafe { vld1q_s8(shifts.as_ptr()) };
    let lut: [u8; 16] = [0, 1, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let lut = unsafe { vld1q_u8(lut.as_ptr()) };
    let three = vdupq_n_u8(3);
    let mt_stride = quads * 4 * nr;
    for q in 0..quads {
        let src = unsafe { codes.add(q * nb) };
        for c16 in 0..nb / 16 {
            let cv = unsafe { vld1q_u8(src.add(c16 * 16)) };
            for (u, iv) in idx.iter().enumerate() {
                let o = c16 * 16 + 4 * u;
                let x = vqtbl1q_u8(cv, *iv);
                let x = vandq_u8(vshlq_u8(x, shifts), three);
                let x = vqtbl1q_u8(lut, x);
                let dst = unsafe { tile.add((o / nr) * mt_stride + q * nr * 4 + (o % nr) * 4) };
                unsafe { vst1q_u8(dst as *mut u8, x) };
            }
        }
    }
}

/// Expands one block of 1-bit codes (`[oct][nb]` bytes) into the same tile: bit (4h + m) of code
/// byte (output o, oct q8) -> tile[o / nr][2*q8 + h][o % nr][m], 1 -> +1, 0 -> -1.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn expand_one(codes: *const u8, octs: usize, nb: usize, nr: usize, tile: *mut i8) {
    let idx = replicate_idx();
    let pats: [uint8x16_t; 2] = std::array::from_fn(|h| {
        let b: [u8; 16] = std::array::from_fn(|p| 1u8 << (4 * h + p % 4));
        unsafe { vld1q_u8(b.as_ptr()) }
    });
    let (two, one) = (vdupq_n_u8(2), vdupq_n_u8(1));
    let quads = 2 * octs;
    let mt_stride = quads * 4 * nr;
    for q8 in 0..octs {
        let src = unsafe { codes.add(q8 * nb) };
        for c16 in 0..nb / 16 {
            let cv = unsafe { vld1q_u8(src.add(c16 * 16)) };
            for (u, iv) in idx.iter().enumerate() {
                let o = c16 * 16 + 4 * u;
                let x = vqtbl1q_u8(cv, *iv);
                for (h, pat) in pats.iter().enumerate() {
                    let val = vsubq_u8(vandq_u8(vtstq_u8(x, *pat), two), one);
                    let q = 2 * q8 + h;
                    let dst = unsafe { tile.add((o / nr) * mt_stride + q * nr * 4 + (o % nr) * 4) };
                    unsafe { vst1q_u8(dst as *mut u8, val) };
                }
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
unsafe fn micro<const C: usize, const ROWS: usize, const NRV: usize, const DOTS: bool>(
    _: *const i8, _: *const i8, _: usize, _: usize, _: &Store) {
    unimplemented!("B-NEON needs aarch64")
}
#[cfg(not(target_arch = "aarch64"))]
unsafe fn expand_two(_: *const u8, _: usize, _: usize, _: usize, _: *mut i8) {
    unimplemented!("B-NEON needs aarch64")
}
#[cfg(not(target_arch = "aarch64"))]
unsafe fn expand_one(_: *const u8, _: usize, _: usize, _: usize, _: *mut i8) {
    unimplemented!("B-NEON needs aarch64")
}
