//! Agent P: the single-plane binary bit-serial kernel on NEON, the measured control for the
//! expectation that bit-serial still loses to SDOT (plans/cpu-benchmark-mac.md "Arms", row P;
//! plans/cpu-benchmark.md, row P).
//!
//! # The identity (exact, over the full contract domain q in [-128, 127])
//!
//! For one frame-part (`q[i]`, `i < K`) and one output (weight bits `b[i]`, 1 = +1):
//! `u = q + 128 = q ^ 0x80` in [0, 255] with bits `a_r` (`u = Σ_r 2^r a_r`), `n = ¬b`,
//! `N₊ = P = Σ b`, `N₋ = K − P`. Per position, `Σ_r 2^r (a_r ⊕ n) = u` when `b = 1` and
//! `255 − u` when `b = 0`, so
//!
//! ```text
//! C = Σ_r 2^r popcount((a_r XOR n) AND valid) = Σ_{b=1} u + Σ_{b=0} (255 − u)
//! Σ_i w_i q_i = Σ_{b=1} (u − 128) − Σ_{b=0} (u − 128) = C − 128 N₊ − 127 N₋ = C − (127 K + P).
//! ```
//!
//! `valid` is folded into the stored operands: `n` is stored as `¬b ∧ valid` and the planes are
//! zero past K and past the last frame-part, so `a_r ⊕ n = 0` there (layout.rs). The correction
//! `127 K + P` is stored per output; `d` is computed in wrapping u32 arithmetic and is exact in
//! i32 because `−128 K <= d <= 127 K` with `K <= 2^24` (asserted).
//!
//! # Kernels (neon.rs)
//!
//! - `Udot` (default): K along the 16 bytes of a vector, 128 positions per step. Per (output,
//!   frame-part, plane r): EOR, CNT, then `UDOT acc.4s, cnt.16b, 2^r` (written as UDOT by element;
//!   LLVM emits the vector form against 8 splat constants), which widens and weights the CNT
//!   bytes in one uop. One u32x4 accumulator per (output, frame-part); C = the sum of its
//!   lanes. Weights and planes are plain `LDR q` loads (paired into `LDP q` by LLVM).
//! - `Uadalp` (the classic form): the 8 planes in the u16 lanes of a vector, 16 positions per
//!   step. Per (output, frame-part): the weight u16 broadcast by `LD1R.8H` (shared by the F
//!   frame-parts), EOR, CNT, `UADALP` into u16 lane r = plane r's count; the 2^r weighting happens
//!   once at the end. 3 + 1/F SIMD uops per 16 MACs, since LD1R also occupies a SIMD unit.
//!
//! Both are register-blocked over `O = 4 G` outputs x `F` frame-parts (F a multiple of C, so a
//! tile holds whole frames and the epilogue runs straight from registers). Threads split the
//! output tiles; each thread walks the frame tiles in order, so its weight slice (at most
//! N K / 8 / threads = 64 KiB for MB1 at 8 threads) and the current activation tile stay in its
//! 128 KiB L1D.
//!
//! Activation planes are built per `(slot, quantization)` and reused exactly when the schedule
//! says `reuse` (LocalFlow's slot semantics), inside the timed scope.

use std::marker::PhantomData;

use shared::aligned::AlignedVec;
use shared::kernel::{Call, Kernel, KernelConfig, Out, WeightBytes};
use shared::pool::ThreadPool;
use shared::quant::QuantizedActs;
use shared::weights::{WeightFormat, WeightsRef};
use shared::{SendPtr, split_range};

pub mod layout;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "aarch64")]
use neon as arch;
#[cfg(not(target_arch = "aarch64"))]
mod portable;
#[cfg(not(target_arch = "aarch64"))]
use portable as arch;

#[cfg(all(target_arch = "aarch64", not(target_feature = "dotprod")))]
compile_error!("p_bitserial needs the dotprod feature (build with -C target-cpu=apple-m1)");

use layout::{PackedWeights, pack_weights, plane_tile_bytes, step_positions, unpack_weights};

/// How CNT results are accumulated (see the crate docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Accum {
    /// UDOT against the plane weight 2^r (LLVM's vector form with 8 splat constants).
    Udot,
    /// The same, with UDOT pinned to its by-element encoding (2 constant registers), which frees
    /// registers for F = 6 tiles. Same layouts as `Udot`.
    UdotLane,
    /// The classic form: planes in u16 lanes, LD1R weights, UADALP.
    Uadalp,
}

impl Accum {
    pub fn name(self) -> &'static str {
        match self {
            Accum::Udot => "udot",
            Accum::UdotLane => "udot_lane",
            Accum::Uadalp => "uadalp",
        }
    }
}

/// How the pool's threads share one GEMM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    /// Threads take contiguous output tiles and walk every frame tile (weights slice L1-resident);
    /// planes are packed first by all threads, then a barrier.
    Outputs,
    /// Threads take contiguous frame tiles, pack each one inside the GEMM job and run every output
    /// tile on it while it is L1-hot (no packing barrier, no cross-core plane traffic; the weights
    /// stream from L2).
    Frames,
}

impl Split {
    pub fn name(self) -> &'static str {
        match self {
            Split::Outputs => "outputs",
            Split::Frames => "frames",
        }
    }
}

/// A tuning point: accumulation variant, outputs per tile `o`, frames per tile for C = 1, 2, 3
/// components (`F = frames[C - 1] * C` frame-parts per tile) and the thread split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tuning {
    pub accum: Accum,
    pub o: usize,
    pub frames: [usize; 3],
    pub split: Split,
}

/// The compiled kernel instances: frame-parts per tile `F` for each (variant, `o`, split). Every
/// one is spill-free in the emitted assembly (dev/asm_loops.py; 32 NEON registers): `Udot` holds
/// 4 G F u32x4 accumulators, 4 G weight vectors and LLVM's 8 splat constants (so 4 G F <= 16);
/// `UdotLane` the same with 2 constants (F = 6: 24 accumulators); `Uadalp` 4 G F u16x8
/// accumulators and F plane vectors.
pub const TILE_F: [(Accum, usize, Split, &[usize]); 4] = [
    (Accum::Udot, 4, Split::Outputs, &[1, 2, 3, 4]),
    (Accum::Udot, 4, Split::Frames, &[1, 2, 3, 4]),
    (Accum::UdotLane, 4, Split::Outputs, &[6]),
    (Accum::Uadalp, 4, Split::Outputs, &[1, 2, 3, 4, 6]),
];

/// The frozen tuning grid (plans/cpu-benchmark-mac.md M7: the same grid size as B-NEON's, 6
/// points, the default first), embedded at compile time; SHA-256 pinned here and in NOTES.md.
pub const TUNING_GRID_JSON: &str = include_str!("../tuning_grid.json");
pub const TUNING_GRID_SHA256: &str = "6734bf4334cc554e6cbc50c090d6a61d44e7ae7998beb3a2d70daed32f59cda1";
pub const TUNING_GRID_POINTS: usize = 6;

/// The points of [`TUNING_GRID_JSON`] in order. Panics unless there are exactly
/// [`TUNING_GRID_POINTS`] valid points.
pub fn tuning_grid() -> Vec<Tuning> {
    let doc: serde_json::Value = serde_json::from_str(TUNING_GRID_JSON).expect("p_bitserial/tuning_grid.json parses");
    let points: Vec<Tuning> =
        doc["points"].as_array().expect("tuning_grid.json: points").iter().map(Tuning::from_json).collect();
    assert_eq!(points.len(), TUNING_GRID_POINTS, "p_bitserial tuning grid must have {TUNING_GRID_POINTS} points");
    points
}

impl Default for Tuning {
    fn default() -> Self {
        Tuning { accum: Accum::Udot, o: 4, frames: [4, 2, 1], split: Split::Outputs }
    }
}

impl Tuning {
    pub fn tile_f(&self, components: usize) -> usize {
        self.frames[components - 1] * components
    }

    pub fn validate(&self) -> Result<(), String> {
        let allowed = TILE_F
            .iter()
            .find(|(a, o, s, _)| *a == self.accum && *o == self.o && *s == self.split)
            .ok_or_else(|| format!("{} o = {} split = {} is not compiled", self.accum.name(), self.o, self.split.name()))?
            .3;
        for c in 1..=3 {
            let f = self.tile_f(c);
            if self.frames[c - 1] == 0 || !allowed.contains(&f) {
                return Err(format!("{} o = {}, C = {c}: F = {f} not in {allowed:?}", self.accum.name(), self.o));
            }
        }
        Ok(())
    }

    /// `{"accum": "udot" | "udot_lane" | "uadalp", "o": 4, "frames": [f1, f2, f3],
    /// "split": "outputs" | "frames"}` (all four keys required); Null = the default point.
    pub fn from_json(v: &serde_json::Value) -> Tuning {
        if v.is_null() {
            return Tuning::default();
        }
        let obj = v.as_object().expect("p_bitserial tuning: a JSON object");
        for key in obj.keys() {
            assert!(["accum", "o", "frames", "split"].contains(&key.as_str()), "p_bitserial tuning: unknown key {key}");
        }
        let accum = match v["accum"].as_str() {
            Some("udot") => Accum::Udot,
            Some("udot_lane") => Accum::UdotLane,
            Some("uadalp") => Accum::Uadalp,
            other => panic!("p_bitserial tuning: accum {other:?}"),
        };
        let split = match v["split"].as_str() {
            Some("outputs") => Split::Outputs,
            Some("frames") => Split::Frames,
            other => panic!("p_bitserial tuning: split {other:?}"),
        };
        let o = v["o"].as_u64().expect("p_bitserial tuning: o") as usize;
        let fr = v["frames"].as_array().expect("p_bitserial tuning: frames");
        assert_eq!(fr.len(), 3, "p_bitserial tuning: frames has 3 entries (C = 1, 2, 3)");
        let frames = std::array::from_fn(|i| fr[i].as_u64().expect("p_bitserial tuning: frames") as usize);
        let t = Tuning { accum, o, frames, split };
        t.validate().unwrap_or_else(|e| panic!("p_bitserial tuning: {e}"));
        t
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({"accum": self.accum.name(), "o": self.o, "frames": self.frames, "split": self.split.name()})
    }
}

/// Fixes the tuning at the type level (tests instantiate every grid point through H's harness).
pub trait Preset: 'static {
    const NAME: &'static str;
    /// `None`: take the tuning from `KernelConfig::tuning` (Null = default).
    fn tuning() -> Option<Tuning>;
}

/// The arm as configured by the frozen grid.
pub struct FromConfig;
impl Preset for FromConfig {
    const NAME: &'static str = "p-bitserial";
    fn tuning() -> Option<Tuning> {
        None
    }
}

pub type BitSerial = BitSerialWith<FromConfig>;

/// Activation planes of one slot and the quantization they were built from.
#[derive(Default)]
struct SlotPlanes {
    buf: Option<AlignedVec<u8>>,
    key: Option<(usize, usize, usize)>,
}

pub struct BitSerialWith<P: Preset> {
    tuning: Tuning,
    slots: [SlotPlanes; 3],
    _preset: PhantomData<fn() -> P>,
}

pub struct Prepared {
    w: PackedWeights,
}

/// Packing work deferred into the GEMM job (`Split::Frames`): each thread packs its frame tiles.
#[derive(Clone, Copy)]
struct PackJob {
    q: usize,
    m_count: usize,
    k: usize,
    accum: Accum,
}

impl<P: Preset> BitSerialWith<P> {
    pub fn with_tuning(_cfg: &KernelConfig, tuning: Tuning) -> Self {
        tuning.validate().unwrap_or_else(|e| panic!("p_bitserial tuning: {e}"));
        BitSerialWith { tuning, slots: Default::default(), _preset: PhantomData }
    }

    pub fn tuning(&self) -> Tuning {
        self.tuning
    }

    /// The slot's planes: checked with `reuse`; otherwise (re)built here by all threads
    /// (`Split::Outputs`), or sized here and returned as a [`PackJob`] for the GEMM job
    /// (`Split::Frames`). Returns the planes' base pointer.
    fn planes(&mut self, pool: &ThreadPool, qa: &QuantizedActs, call: Call) -> (*const u8, Option<PackJob>) {
        let (t, c, k) = (qa.rows(), qa.components(), qa.cols());
        let accum = self.tuning.accum;
        let f = self.tuning.tile_f(c);
        let mf = self.tuning.frames[c - 1];
        let split = self.tuning.split;
        let slot = &mut self.slots[call.slot.index()];
        let key = (t, c, k);
        if call.reuse {
            assert_eq!(slot.key, Some(key), "p_bitserial: reuse of a slot built for a different quantization");
            return (slot.buf.as_ref().unwrap().as_ptr(), None);
        }
        let tiles = t.div_ceil(mf);
        let need = tiles * plane_tile_bytes(accum, k, f);
        if slot.buf.as_ref().is_none_or(|b| b.len() < need) {
            slot.buf = Some(AlignedVec::zeroed(need));
        }
        let out = SendPtr(slot.buf.as_mut().unwrap().as_mut_ptr());
        let job = PackJob { q: qa.q().as_ptr() as usize, m_count: t * c, k, accum };
        slot.key = Some(key);
        if split == Split::Frames {
            return (out.get(), Some(job));
        }
        let threads = pool.threads();
        pool.run(&|idx| {
            let range = split_range(tiles, threads, idx);
            if !range.is_empty() {
                unsafe { arch::pack_planes(job.q as *const i8, job.m_count, k, accum, f, range, out.get()) };
            }
        });
        (out.get(), None)
    }
}

/// Final stores of one tile, monomorphised over the output kind (one inner routine for both).
trait Sink: Copy + Send + Sync {
    /// Frame `t`, outputs `o .. o + valid` (`valid <= 4`), components' dots `d`.
    unsafe fn put<const C: usize>(self, t: usize, o: usize, valid: usize, d: [arch::D4; C], ws: *const f32, fs: f32);
}

#[derive(Clone, Copy)]
struct F32Sink {
    y: SendPtr<f32>,
    n: usize,
}

impl Sink for F32Sink {
    #[inline(always)]
    unsafe fn put<const C: usize>(self, t: usize, o: usize, valid: usize, d: [arch::D4; C], ws: *const f32, fs: f32) {
        unsafe { arch::store_f32::<C>(self.y.get().add(t * self.n + o), d, ws, fs, valid) };
    }
}

#[derive(Clone, Copy)]
struct DotSink {
    d: SendPtr<i32>,
    n: usize,
}

impl Sink for DotSink {
    #[inline(always)]
    unsafe fn put<const C: usize>(self, t: usize, o: usize, valid: usize, d: [arch::D4; C], _ws: *const f32, _fs: f32) {
        for (c, &dc) in d.iter().enumerate() {
            unsafe { arch::store_dots(self.d.get().add((t * C + c) * self.n + o), dc, valid) };
        }
    }
}

const KIND_UADALP: u8 = 0;
const KIND_UDOT: u8 = 1;
const KIND_UDOT_LANE: u8 = 2;

/// The whole GEMM for one (variant, split, G = O / 4, F, C) instance.
#[allow(clippy::too_many_arguments)]
unsafe fn run<S: Sink, const KIND: u8, const FRAMES: bool, const G: usize, const F: usize, const C: usize>(
    pool: &ThreadPool,
    w: &PackedWeights,
    planes: *const u8,
    pack: Option<PackJob>,
    t_n: usize,
    frame_scale: &[f32],
    sink: S,
) {
    let mf = F / C;
    let (n, steps) = (w.n, w.steps);
    let tile_codes = w.tile_code_bytes();
    let tile_planes = plane_tile_bytes(w.accum, w.k, F);
    let frame_tiles = t_n.div_ceil(mf);
    let threads = pool.threads();
    let (codes, corr, scale) = (w.codes.as_ptr() as usize, w.corr.as_ptr() as usize, w.scale.as_ptr() as usize);
    let planes = planes as usize;
    // One (frame tile, output tile): the micro-kernel, then the epilogue straight from registers.
    let tile = |ft: usize, ot: usize| {
        let a = (planes + ft * tile_planes) as *const u8;
        let frames = mf.min(t_n - ft * mf);
        let o0 = ot * 4 * G;
        let wp = (codes + ot * tile_codes) as *const u8;
        let cp = (corr as *const u32).wrapping_add(o0);
        let d: [[arch::D4; F]; G] = unsafe {
            match KIND {
                KIND_UDOT => arch::tile_udot::<G, F>(wp, a, steps, cp),
                KIND_UDOT_LANE => arch::tile_udot_lane::<G, F>(wp, a, steps, cp),
                _ => arch::tile_uadalp::<G, F>(wp as *const u16, a, steps, cp),
            }
        };
        for (g, dg) in d.iter().enumerate() {
            let og = o0 + 4 * g;
            if og >= n {
                break;
            }
            let valid = (n - og).min(4);
            let ws = (scale as *const f32).wrapping_add(og);
            // Constant trip count (F / C), so after unrolling every index into `dg` is a
            // constant and the dots stay in registers; the last frame tile may be partial.
            for j in 0..F / C {
                if j >= frames {
                    break;
                }
                let t = ft * mf + j;
                let parts: [arch::D4; C] = std::array::from_fn(|c| dg[j * C + c]);
                unsafe { sink.put::<C>(t, og, valid, parts, ws, frame_scale[t]) };
            }
        }
    };
    pool.run(&|idx| {
        if FRAMES {
            for ft in split_range(frame_tiles, threads, idx) {
                if let Some(p) = pack {
                    unsafe { arch::pack_planes(p.q as *const i8, p.m_count, p.k, p.accum, F, ft..ft + 1, planes as *mut u8) };
                }
                for ot in 0..w.tiles {
                    tile(ft, ot);
                }
            }
        } else {
            let range = split_range(w.tiles, threads, idx);
            for ft in 0..frame_tiles {
                for ot in range.clone() {
                    tile(ft, ot);
                }
            }
        }
    });
}

/// Dispatch to the monomorphised instance; exactly the instances of `TILE_F`.
#[allow(clippy::too_many_arguments)]
fn dispatch<S: Sink>(tuning: &Tuning, c: usize, pool: &ThreadPool, w: &PackedWeights, planes: *const u8,
                     pack: Option<PackJob>, t_n: usize, fs: &[f32], sink: S) {
    let f = tuning.tile_f(c);
    macro_rules! arms {
        ($($kind:ident, $split:ident, $frames:literal: [$($g:literal, $f:literal, $c:literal);*]);* $(;)?) => {
            match (tuning.accum, tuning.split, tuning.o / 4, f, c) {
                $($((Accum::$kind, Split::$split, $g, $f, $c) => unsafe {
                    run::<S, { kind(Accum::$kind) }, $frames, $g, $f, $c>(pool, w, planes, pack, t_n, fs, sink)
                },)*)*
                other => panic!("p_bitserial: no kernel instance for (accum, split, G, F, C) = {other:?}"),
            }
        };
    }
    arms!(
        Udot, Outputs, false: [1, 1, 1; 1, 2, 1; 1, 2, 2; 1, 3, 1; 1, 3, 3; 1, 4, 1; 1, 4, 2];
        Udot, Frames, true: [1, 1, 1; 1, 2, 1; 1, 2, 2; 1, 3, 1; 1, 3, 3; 1, 4, 1; 1, 4, 2];
        UdotLane, Outputs, false: [1, 6, 1; 1, 6, 2; 1, 6, 3];
        Uadalp, Outputs, false: [1, 1, 1; 1, 2, 1; 1, 2, 2; 1, 3, 1; 1, 3, 3; 1, 4, 1; 1, 4, 2; 1, 6, 1; 1, 6, 2; 1, 6, 3];
    );
}

const fn kind(a: Accum) -> u8 {
    match a {
        Accum::Udot => KIND_UDOT,
        Accum::UdotLane => KIND_UDOT_LANE,
        Accum::Uadalp => KIND_UADALP,
    }
}

/// The two output kinds as non-generic functions, so every kernel instance is compiled once, in
/// this crate (one codegen for tests, benches and the replay; inspectable with --emit asm).
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn gemm_f32(tuning: &Tuning, c: usize, pool: &ThreadPool, w: &PackedWeights, planes: *const u8, pack: Option<PackJob>,
            t: usize, fs: &[f32], y: &mut [f32]) {
    dispatch(tuning, c, pool, w, planes, pack, t, fs, F32Sink { y: SendPtr(y.as_mut_ptr()), n: w.n });
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn gemm_dots(tuning: &Tuning, c: usize, pool: &ThreadPool, w: &PackedWeights, planes: *const u8, pack: Option<PackJob>,
             t: usize, fs: &[f32], d: &mut [i32]) {
    dispatch(tuning, c, pool, w, planes, pack, t, fs, DotSink { d: SendPtr(d.as_mut_ptr()), n: w.n });
}

impl<P: Preset> Kernel for BitSerialWith<P> {
    type Prepared = Prepared;
    const NAME: &'static str = P::NAME;
    const FORMAT: WeightFormat = WeightFormat::Binary;

    fn new(cfg: &KernelConfig) -> Self {
        let tuning = P::tuning().unwrap_or_else(|| Tuning::from_json(&cfg.tuning));
        Self::with_tuning(cfg, tuning)
    }

    fn prepare(&self, w: WeightsRef<'_>) -> Prepared {
        let m = match w {
            WeightsRef::Binary(m) => m,
            WeightsRef::Ternary(_) => panic!("p_bitserial consumes parakeet-binary-v1 codes"),
        };
        m.validate().unwrap_or_else(|e| panic!("p_bitserial: {e}"));
        assert!(m.k <= 1 << 24, "p_bitserial: K = {} > 2^24 (d must stay exact in i32 and f32)", m.k);
        Prepared { w: pack_weights(m, self.tuning.accum, self.tuning.o) }
    }

    fn gemm(&mut self, pool: &ThreadPool, w: &Prepared, qa: &QuantizedActs, call: Call, out: Out<'_>) {
        let w = &w.w;
        let (t, c, k, n) = (qa.rows(), qa.components(), qa.cols(), w.n);
        assert_eq!(k, w.k, "p_bitserial: activation width {k} != K {}", w.k);
        assert!((1..=3).contains(&c));
        if t == 0 {
            // Nothing to compute; the slot still records this (empty) quantization.
            if !call.reuse {
                self.slots[call.slot.index()].key = Some((t, c, k));
            }
            return;
        }
        let (planes, pack) = self.planes(pool, qa, call);
        let tuning = self.tuning;
        let fs = qa.scale();
        match out {
            Out::F32(y) => {
                assert!(y.len() >= t * n);
                gemm_f32(&tuning, c, pool, w, planes, pack, t, fs, y);
            }
            Out::Dots(d) => {
                assert!(d.len() >= t * c * n);
                gemm_dots(&tuning, c, pool, w, planes, pack, t, fs, d);
            }
        }
    }

    fn unpack(&self, w: &Prepared) -> Vec<i8> {
        unpack_weights(&w.w)
    }

    fn shape(&self, w: &Prepared) -> (usize, usize) {
        (w.w.n, w.w.k)
    }

    fn weight_bytes(&self, w: &Prepared) -> WeightBytes {
        WeightBytes { codes: w.w.codes.bytes(), per_row: w.w.corr.bytes() + w.w.scale.bytes() }
    }

    fn workspace_bytes(&self) -> usize {
        self.slots.iter().filter_map(|s| s.buf.as_ref()).map(|b| b.bytes()).sum()
    }

    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "arm": "P (bit-serial control)",
            "tuning": self.tuning.to_json(),
            "tuning_grid_sha256": TUNING_GRID_SHA256,
            "tuning_grid_index": tuning_grid().iter().position(|p| *p == self.tuning),
            "split": match self.tuning.split {
                Split::Outputs => "threads split output tiles; planes packed by all threads before the GEMM",
                Split::Frames => "threads split frame tiles; each packs its tiles inside the GEMM job",
            },
            "step_positions": step_positions(self.tuning.accum),
            "identity": "d = C - (127 K + P), C = sum_r 2^r popcount((a_r XOR n) AND valid), u = q ^ 0x80",
            "inner_loop": match self.tuning.accum {
                Accum::Udot => "LDP q weights/planes; per (output, frame-part, plane): EOR, CNT, UDOT against splat(2^r) bytes",
                Accum::UdotLane => "LDP q weights/planes; per (output, frame-part, plane): EOR, CNT, UDOT by element (2^r lane, asm-pinned)",
                Accum::Uadalp => "LD1R.8H weights, LDR q planes; EOR, CNT, UADALP per (output, frame-part); 2^r at flush",
            },
        })
    }
}
