//! Arm T: T-MAC-style lookup tables for MB1's 1-bit weights on the M1 Pro (plans/cpu-benchmark-mac.md
//! "Arms"; plans/cpu-benchmark.md row T; Wei et al. 2024, "T-MAC").
//!
//! # Identity (exact)
//! With weight bits `b` (1 = +1), `w = 2b - 1`, so for one frame-part `q` (s8, full range):
//! `d = Σ_i w_i q_i = 2 F - Q`, `F = Σ_i b_i q_i`, `Q = Σ_i q_i`.
//! Inputs form groups of g = 4. Per frame-part and group `j`, a table holds the 16 subset sums
//! `f_j(n) = Σ_{k<4, bit k of n} q_{4j+k}` (n = 0..15) as INT16; a weight row's 4 bits of group
//! `j` are the index `n_j`, and `F = Σ_j f_j(n_j)`. There is no table requantization.
//!
//! # Widths (proven intervals)
//! - entries: `f ∈ [4 · (-128), 4 · 127] = [-512, 508]` (the contract domain is all of s8);
//! - INT16 runs: at most [`RUN_GROUPS`] = 64 entries are added in an INT16 lane, giving
//!   `[64 · (-512), 64 · 508] = [-32768, 32512] ⊂ [-32768, 32767]`, so ADD.8H never wraps; each run
//!   is then sign-extended (SADDW/SADDW2) into INT32;
//! - INT32: `|F| ≤ 128 K`, `|Q| ≤ 128 K`, `|2F - Q| = |d| ≤ 128 K ≤ 2^24` for every model shape, so
//!   `f32(d)` in the epilogue is exact (shared::epilogue).
//!
//! # Lookup (the inner loop)
//! A table is a natural `[i16; 16]` (32 bytes = two q registers). The index of output `o` is the
//! u16 lane `0x0100 + 0x0202 · n` (bytes `2n`, `2n + 1`), so `TBL Vd.16B, {Vt.16B, Vt+1.16B}`
//! returns 8 little-endian INT16 entries for 8 outputs, accumulated with one ADD.8H. Per group and
//! frame-part: one 32-byte table load shared by `8 V` outputs; per group: `V` index loads shared
//! by the block's frame-parts. On Firestorm both TBL (2-register table) and ADD are 1 uop with 4
//! per cycle (dougallj.github.io/applecpu/firestorm-simd.html): 2 uops per 32 MACs, the same
//! density as SDOT (1 uop per 16 MACs), with no weight expansion beyond the index panel.
//!
//! # Work layout
//! - Threads split the frames (`split_range`); each thread owns its frames' tables, so no table is
//!   shared or synchronized. Per thread and K-chunk (`k_chunk` inputs): build the tables of all its
//!   frame-parts (`8 · k_chunk` bytes each, reused for K and V after Q when the call allows it);
//!   then per output panel (`n_panel` outputs) expand the panel's index vectors once (2 bytes per
//!   output and group, L1-resident) and run every frame-block x out-block of the panel.
//! - Block kernels are monomorphised over (C, F frames, V index vectors, output kind): `F · C`
//!   frame-parts x `8 V` outputs, with `F · C · V ≤ 16` INT16 accumulators.
//! - K-chunks after the first add into a per-thread INT32 buffer; the last chunk runs the epilogue.
//!
//! # Weights and workspace (plans/cpu-benchmark-mac.md "Storage and workspace")
//! - Weights: `layout`, the 1-bit export codes re-tiled into 16-output blocks, exactly
//!   `N · ceil(K / 8)` bytes for the model shapes (72,351,744 B for the 240 replayed matrices), plus
//!   the FP32 weight scale per output row. Nothing derived from the codes is kept with them.
//! - Workspace (declared exactly by `workspace_bytes`, tests/storage.rs): per worker the frame-part
//!   tables, their Q sums, ONE index panel buffer (`n_panel · k_chunk / 4 · 2` bytes, 32 KiB at the
//!   default), the INT32 scratch and the K-chunk sums. The index panel is rebuilt from the codes for
//!   every (call, K-chunk, panel) inside `gemm`, i.e. inside the timed replay, and is never read by a
//!   later call; its size does not depend on the number of matrices.

use shared::aligned::AlignedVec;
use shared::{
    Call, Kernel, KernelConfig, Out, PerThread, QuantizedActs, SendPtr, Slot, ThreadPool, WeightBytes, WeightFormat,
    WeightsRef, split_range,
};

pub mod layout;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "aarch64")]
use neon as imp;
#[cfg(not(target_arch = "aarch64"))]
mod portable;
#[cfg(not(target_arch = "aarch64"))]
use portable as imp;

/// Most table entries added in one INT16 lane before widening (see "Widths").
pub const RUN_GROUPS: usize = 64;

/// The frozen tuning grid file (t_lut/tuning_grid.json), embedded at compile time.
pub const TUNING_GRID_JSON: &str = include_str!("../tuning_grid.json");
/// SHA-256 of [`TUNING_GRID_JSON`] (checked by tests/lut.rs; recorded in cpu1bit/NOTES.md).
pub const TUNING_GRID_SHA256: &str = "ba854aa640dec54272a88a985bbfa92029b581199a11beafe63b65b40823399a";
/// Grid size, equal to B-NEON's (plans/cpu-benchmark-mac.md: the same tuning budget for every arm).
pub const TUNING_GRID_POINTS: usize = 6;

/// Arguments of one block-kernel call. Rows `r = f * C + c` of the block.
pub(crate) struct BlockArgs {
    /// Table of row 0, group 0 of the chunk; row `r` at `+ r * table_stride`, group `j` at `+ 32 j`.
    pub(crate) tables: *const u8,
    pub(crate) table_stride: usize,
    /// Index vectors of the out-block, group 0 of the chunk; group `j` at `+ j * V * 16`.
    pub(crate) panel: *const u8,
    /// Groups in this chunk.
    pub(crate) groups: usize,
    /// Scratch, `R * 8 V` i32.
    pub(crate) acc: *mut i32,
    /// Chunk partial sums of row 0 at the out-block's first output; row `r` at `+ r * persist_stride`.
    /// Read unless `first`, written unless `last`; unused (may be null) for single-chunk calls.
    pub(crate) persist: *mut i32,
    pub(crate) persist_stride: usize,
    pub(crate) first: bool,
    pub(crate) last: bool,
    /// `Q` per block row.
    pub(crate) qsum: *const i32,
    /// Weight scales from the out-block's first output (8 V readable).
    pub(crate) wscale: *const f32,
    /// Frame scales of the block's frames.
    pub(crate) fscale: *const f32,
    /// F32: `y` at (first frame, first output); Dots: `d` at (first frame, component 0, first output).
    pub(crate) out: *mut u8,
    /// Output row stride in elements (N).
    pub(crate) n: usize,
    /// Valid outputs of this out-block, 1..=8 V.
    pub(crate) valid: usize,
}

pub(crate) type BlockFn = unsafe fn(&BlockArgs);

/// Blocking parameters (the frozen tuning grid is [`TLut::tuning_grid`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tuning {
    /// Index vectors (8 outputs each) per out-block: 2 or 4.
    pub v: usize,
    /// Outputs per index panel: a multiple of `8 v`.
    pub n_panel: usize,
    /// Inputs per K-chunk: a multiple of 256.
    pub k_chunk: usize,
}

impl Default for Tuning {
    fn default() -> Self {
        Tuning { v: 4, n_panel: 64, k_chunk: 1024 }
    }
}

impl Tuning {
    pub fn parse(v: &serde_json::Value) -> Tuning {
        let mut t = Tuning::default();
        if v.is_null() {
            return t;
        }
        let obj = v.as_object().expect("t_lut tuning: expected a JSON object");
        for (key, val) in obj {
            let x = val.as_u64().unwrap_or_else(|| panic!("t_lut tuning: {key} must be an integer")) as usize;
            match key.as_str() {
                "v" => t.v = x,
                "n_panel" => t.n_panel = x,
                "k_chunk" => t.k_chunk = x,
                other => panic!("t_lut tuning: unknown key {other}"),
            }
        }
        assert!(t.v == 2 || t.v == 4, "t_lut tuning: v must be 2 or 4");
        assert!(t.n_panel >= 8 * t.v && t.n_panel % (8 * t.v) == 0, "t_lut tuning: n_panel must be a multiple of 8 v");
        assert!(t.k_chunk >= 256 && t.k_chunk % 256 == 0, "t_lut tuning: k_chunk must be a multiple of 256");
        t
    }

    pub fn to_json(self) -> serde_json::Value {
        serde_json::json!({"v": self.v, "n_panel": self.n_panel, "k_chunk": self.k_chunk})
    }

    /// Frames per block: the most with `F · C · v ≤ 16` INT16 accumulators.
    pub fn frames(self, c: usize) -> usize {
        (16 / (c * self.v)).max(1)
    }
}

/// Prepared weights: the tiled 1-bit codes and padded scales.
pub struct Prepared {
    w: layout::Codes,
}

/// Per-thread workspace (tables, panel, scratch, chunk sums).
#[derive(Default)]
struct ThreadWs {
    tables: Option<AlignedVec<i16>>,
    qsum: Option<AlignedVec<i32>>,
    panel: Option<AlignedVec<u8>>,
    acc: Option<AlignedVec<i32>>,
    persist: Option<AlignedVec<i32>>,
}

fn grow<T: shared::aligned::Zeroable>(v: &mut Option<AlignedVec<T>>, len: usize) {
    if v.as_ref().is_none_or(|b| b.len() < len) {
        *v = Some(AlignedVec::zeroed(len));
    }
}

/// Heap bytes actually allocated for one buffer (AlignedVec rounds up to 128-byte lines).
fn heap<T: shared::aligned::Zeroable>(v: &Option<AlignedVec<T>>) -> usize {
    v.as_ref().map_or(0, AlignedVec::heap_bytes)
}

impl ThreadWs {
    fn heap_bytes(&self) -> usize {
        heap(&self.tables) + heap(&self.qsum) + heap(&self.panel) + heap(&self.acc) + heap(&self.persist)
    }
}

/// Identity of the activations the current tables were built from (reuse check).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tag {
    slot: Slot,
    q: usize,
    rows: usize,
    k: usize,
    c: usize,
    threads: usize,
}

pub struct TLut {
    tuning: Tuning,
    ws: PerThread<ThreadWs>,
    tag: Option<Tag>,
    /// Live heap of the workspace: every thread's buffers plus the per-thread vector itself. Buffers
    /// only grow, so this is also the peak. Kept exact (tests/storage.rs) because the storage audit
    /// counts every undeclared live byte as weight bytes.
    workspace: usize,
}

impl TLut {
    /// The frozen tuning grid, the points of [`TUNING_GRID_JSON`] in order (the default first).
    /// Panics unless the embedded file has exactly [`TUNING_GRID_POINTS`] valid points.
    pub fn tuning_grid() -> Vec<serde_json::Value> {
        let doc: serde_json::Value = serde_json::from_str(TUNING_GRID_JSON).expect("t_lut/tuning_grid.json parses");
        let points = doc["points"].as_array().expect("tuning_grid.json: points").clone();
        assert_eq!(points.len(), TUNING_GRID_POINTS, "t_lut tuning grid must have {TUNING_GRID_POINTS} points");
        assert_eq!(Tuning::parse(&points[0]), Tuning::default(), "the grid's first point is the default");
        for p in &points {
            Tuning::parse(p);
        }
        points
    }

    pub fn tuning(&self) -> Tuning {
        self.tuning
    }

    /// Grows every thread's buffers to the call's needs (no-op once the largest call has run).
    fn ensure(&mut self, threads: usize, rows: usize, n_pad: usize, multi_chunk: bool) {
        if self.ws.len() < threads {
            self.ws = PerThread::new(threads, |_| ThreadWs::default());
            self.tag = None;
        }
        let tu = self.tuning;
        let gmax = tu.k_chunk / 4;
        for ws in self.ws.iter_mut() {
            grow(&mut ws.tables, rows * gmax * 16);
            grow(&mut ws.qsum, rows.max(1));
            grow(&mut ws.panel, tu.n_panel * gmax * 2);
            grow(&mut ws.acc, 8 * 32);
            if multi_chunk {
                grow(&mut ws.persist, rows * n_pad);
            }
        }
        let buffers: usize = self.ws.iter_mut().map(|w| w.heap_bytes()).sum();
        self.workspace = buffers + self.ws.len() * std::mem::size_of::<ThreadWs>();
    }
}

/// Dev-only phase timing support for src/bin/tdev.rs (never on the arm's path): each runs one
/// phase of a call on the calling thread over the whole call, writing into a caller buffer.
#[doc(hidden)]
pub mod dev {
    use super::*;

    /// Tables of every frame-part of `qa` over all K-chunks (one thread's work at `threads = 1`).
    pub fn tables_only(qa: &QuantizedActs, tuning: Tuning, buf: &mut AlignedVec<i16>) -> i64 {
        let (t, c, k) = (qa.rows(), qa.components(), qa.cols());
        let row_i16 = (tuning.k_chunk / 4) * 16;
        assert!(buf.len() >= t * c * row_i16);
        let mut s = 0i64;
        for i0 in (0..k).step_by(tuning.k_chunk) {
            let i1 = (i0 + tuning.k_chunk).min(k);
            for r in 0..t * c {
                s += unsafe { imp::build_tables(qa.q().as_ptr().add(r * k + i0), i1 - i0, buf.as_mut_ptr().add(r * row_i16)) } as i64;
            }
        }
        s
    }

    /// Every index panel of `w` over all K-chunks (one thread's expansion work per call).
    pub fn panels_only(w: &Prepared, tuning: Tuning, buf: &mut AlignedVec<u8>) {
        let w = &w.w;
        let v = tuning.v;
        assert!(buf.len() >= tuning.n_panel * (tuning.k_chunk / 4) * 2);
        for i0 in (0..w.k).step_by(tuning.k_chunk) {
            let i1 = (i0 + tuning.k_chunk).min(w.k);
            for np0 in (0..w.n).step_by(tuning.n_panel) {
                let np1 = (np0 + tuning.n_panel).min(w.n);
                unsafe { imp::expand_panel(v, w.codes.as_ptr(), w.kb, np0 / (8 * v), np1.div_ceil(8 * v), i0 / 8, i1.div_ceil(8), buf.as_mut_ptr()) };
            }
        }
    }
}

impl Kernel for TLut {
    type Prepared = Prepared;
    const NAME: &'static str = "t-lut";
    const FORMAT: WeightFormat = WeightFormat::Binary;

    fn new(cfg: &KernelConfig) -> Self {
        let tuning = Tuning::parse(&cfg.tuning);
        let mut k = TLut { tuning, ws: PerThread::new(cfg.threads, |_| ThreadWs::default()), tag: None, workspace: 0 };
        let rows = cfg.max_rows.div_ceil(cfg.threads.max(1)) * cfg.components;
        // Sized for the largest call; the per-thread chunk buffer is allocated on the first call
        // that needs it (K > k_chunk), at that call's N (the model's K = 4096 calls have N = 1024).
        k.ensure(cfg.threads, rows, 0, false);
        k
    }

    fn prepare(&self, w: WeightsRef<'_>) -> Prepared {
        match w {
            WeightsRef::Binary(m) => Prepared { w: layout::tile(m) },
            WeightsRef::Ternary(_) => panic!("t_lut consumes the binary export"),
        }
    }

    fn gemm(&mut self, pool: &ThreadPool, w: &Prepared, qa: &QuantizedActs, call: Call, out: Out<'_>) {
        let w = &w.w;
        let (t, c, k, n) = (qa.rows(), qa.components(), qa.cols(), w.n);
        assert_eq!(k, w.k, "t_lut: activation width {k} != weight K {}", w.k);
        assert!((1..=3).contains(&c));
        let (outp, dots) = match out {
            Out::F32(y) => {
                assert!(y.len() >= t * n, "t_lut: output too short");
                (y.as_mut_ptr() as *mut u8, false)
            }
            Out::Dots(d) => {
                assert!(d.len() >= t * c * n, "t_lut: dots too short");
                (d.as_mut_ptr() as *mut u8, true)
            }
        };
        if t == 0 {
            self.tag = None;
            return;
        }
        let threads = pool.threads();
        let tu = self.tuning;
        let chunks = k.div_ceil(tu.k_chunk);
        let tag = Tag { slot: call.slot, q: qa.q().as_ptr() as usize, rows: t, k, c, threads };
        // Tables cover the whole K only for single-chunk calls; otherwise they are rebuilt.
        let reuse = call.reuse && chunks == 1 && self.tag == Some(tag);
        let frames_max = t.div_ceil(threads);
        self.ensure(threads, frames_max * c, w.n_pad, chunks > 1);
        let row_i16 = (tu.k_chunk / 4) * 16;
        let fb = tu.frames(c);
        let v = tu.v;
        let outp = SendPtr(outp);
        let ws_all = &self.ws;
        pool.run(&|idx| {
            let frames = split_range(t, threads, idx);
            if frames.is_empty() {
                return;
            }
            // Only this pool index touches its workspace during the job.
            let ws = unsafe { ws_all.get(idx) };
            let tables = ws.tables.as_mut().unwrap().as_mut_ptr();
            let qsum = ws.qsum.as_mut().unwrap().as_mut_ptr();
            let panel = ws.panel.as_mut().unwrap().as_mut_ptr();
            let acc = ws.acc.as_mut().unwrap().as_mut_ptr();
            let persist = ws.persist.as_mut().map_or(core::ptr::null_mut(), |p| p.as_mut_ptr());
            let (t0, nf) = (frames.start, frames.len());
            let rows = nf * c;
            for kc in 0..chunks {
                let i0 = kc * tu.k_chunk;
                let i1 = (i0 + tu.k_chunk).min(k);
                let groups = (i1 - i0).div_ceil(4);
                let (p0, p1) = (i0 / 8, i1.div_ceil(8));
                let gstride = 2 * (p1 - p0);
                if !reuse {
                    for lr in 0..rows {
                        let (f, cc) = (lr / c, lr % c);
                        unsafe {
                            let q = qa.q().as_ptr().add(((t0 + f) * c + cc) * k + i0);
                            let s = imp::build_tables(q, i1 - i0, tables.add(lr * row_i16));
                            *qsum.add(lr) = if kc == 0 { s } else { *qsum.add(lr) + s };
                        }
                    }
                }
                let mut np0 = 0;
                while np0 < n {
                    let np1 = (np0 + tu.n_panel).min(n);
                    let (ob0, ob1) = (np0 / (8 * v), np1.div_ceil(8 * v));
                    unsafe { imp::expand_panel(v, w.codes.as_ptr(), w.kb, ob0, ob1, p0, p1, panel) };
                    let mut f0 = 0;
                    while f0 < nf {
                        let f = fb.min(nf - f0);
                        let kern = imp::block_fn(c, f, v, dots);
                        for ob in ob0..ob1 {
                            let o0 = ob * 8 * v;
                            let out = unsafe {
                                if dots {
                                    (outp.get() as *mut i32).add((t0 + f0) * c * n + o0) as *mut u8
                                } else {
                                    (outp.get() as *mut f32).add((t0 + f0) * n + o0) as *mut u8
                                }
                            };
                            let args = BlockArgs {
                                tables: unsafe { tables.add(f0 * c * row_i16) } as *const u8,
                                table_stride: row_i16 * 2,
                                panel: unsafe { panel.add((ob - ob0) * gstride * v * 16) },
                                groups,
                                acc,
                                persist: if chunks > 1 { unsafe { persist.add(f0 * c * w.n_pad + o0) } } else { persist },
                                persist_stride: w.n_pad,
                                first: kc == 0,
                                last: kc + 1 == chunks,
                                qsum: unsafe { qsum.add(f0 * c) },
                                wscale: unsafe { w.scale.as_ptr().add(o0) },
                                fscale: unsafe { qa.scale().as_ptr().add(t0 + f0) },
                                out,
                                n,
                                valid: (n - o0).min(8 * v),
                            };
                            unsafe { kern(&args) };
                        }
                        f0 += f;
                    }
                    np0 = np1;
                }
            }
        });
        self.tag = if chunks == 1 { Some(tag) } else { None };
    }

    fn unpack(&self, w: &Prepared) -> Vec<i8> {
        layout::unpack(&w.w)
    }

    fn shape(&self, w: &Prepared) -> (usize, usize) {
        (w.w.n, w.w.k)
    }

    fn weight_bytes(&self, w: &Prepared) -> WeightBytes {
        WeightBytes { codes: w.w.codes.heap_bytes(), per_row: w.w.scale.capacity() * std::mem::size_of::<f32>() }
    }

    fn workspace_bytes(&self) -> usize {
        self.workspace
    }

    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "arm": Self::NAME,
            "variant": "tbl2-i16-g4",
            "group": 4,
            "table": "16 INT16 subset sums f(n) = sum of q over set bits of n, per frame-part and group; d = 2F - Q",
            "accumulation": format!("INT16 runs of <= {RUN_GROUPS} groups ([-32768, 32512]), widened to INT32"),
            "lookup": "TBL 2-register table, u16 index 0x0100 + 0x0202 n, ADD.8H",
            "tuning": self.tuning.to_json(),
            "tuning_grid_sha256": TUNING_GRID_SHA256,
            "simd": if cfg!(target_arch = "aarch64") { "neon" } else { "portable-scalar" },
        })
    }
}
