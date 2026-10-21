//! Agent V (plans/cpu-benchmark-mac.md "Arms", V): MB1 projections from 1-bit weights that stay
//! 1-bit in memory and are expanded in NEON registers right before `SDOT`.
//!
//! - Weights: the v-layout (`layout.rs`), a bit permutation of the export, `n * k / 8` bytes for
//!   every model shape. One `CMTST` (or `CMLT #0` for bit 7) per 16 weights gives the `SDOT`
//!   operand as bytes {0x00, 0xFF} = `-b` (`kernel.rs`).
//! - Arithmetic: LocalFlow's. Activations are H's shared residual INT8 parts (s8); the dot of each
//!   (row, frame-part) is exact in i32: `d = -2 z - U`, `z` from `SDOT`, `U = Σ_k q_k`;
//!   the FP32 epilogue is `shared::epilogue::neon::finish4` (LocalFlow's exact order).
//! - Per-call activation state: `U` per frame-part (and a zero-padded activation copy, only when
//!   `k % 32 != 0`), kept per quantization slot and reused when the schedule says `reuse`.
//! - Threads: a `tm x tn` grid over (frames, row groups). By default every thread takes a
//!   contiguous range of frames (`split_range`, the same split as the shared quantizer) and all
//!   row groups; few frames switch to splitting the groups.

pub mod layout;

#[cfg(target_arch = "aarch64")]
pub mod kernel;

use shared::aligned::AlignedVec;
use shared::split_range;
use shared::{Call, Kernel, KernelConfig, Out, QuantizedActs, SendPtr, ThreadPool, WeightBytes, WeightFormat, WeightsRef};

use layout::{CHUNK_K, GROUP_ROWS};

/// Most frames per panel for C components (panel = F frames x C parts <= 16 accumulators).
pub const DEFAULT_FMAX: [usize; 4] = [0, 16, 8, 5];

/// Calls with more inputs than this use the `fmax{c}_wide` caps (the model's K = 4096 calls,
/// whose activation panel is 4x larger in L1).
pub const WIDE_K: usize = 2048;

/// How the pool's threads divide one call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    /// Frames when every thread gets at least one full panel, else as few frame ranges as needed.
    Auto,
    /// Frames only (tn = 1).
    Frames,
    /// Row groups only (tm = 1).
    Groups,
}

pub struct VSdot {
    split: Split,
    /// `fmax[wide][c]`: most frames per panel, for K <= WIDE_K (0) and K > WIDE_K (1).
    fmax: [[usize; 4]; 2],
    slots: [SlotState; 3],
}

#[derive(Default)]
struct SlotState {
    valid: bool,
    rows: usize,
    comps: usize,
    cols: usize,
    /// Identity of the activations the state was built from (defensive reuse check).
    q_addr: usize,
    /// `U = Σ_k q[fp][k]` per frame-part `fp = t * C + c`.
    u: Vec<i32>,
    /// Zero-padded copy `[fp][kp]`, `kp = ceil(k / 32) * 32`, only for `k % 32 != 0`.
    apad: Vec<i8>,
}

impl SlotState {
    fn bytes(&self) -> usize {
        4 * self.u.capacity() + self.apad.capacity()
    }
}

/// One prepared projection.
pub struct Prepared {
    n: usize,
    k: usize,
    groups: usize,
    chunks: usize,
    /// v-layout, `groups * chunks * 16` bytes, 128-byte aligned.
    bits: AlignedVec<u8>,
    /// Row scales, padded with 0 to `groups * 4`.
    scale: AlignedVec<f32>,
}

impl VSdot {
    fn fmax_for(&self, c: usize, k: usize) -> usize {
        self.fmax[usize::from(k > WIDE_K)][c]
    }

    fn tm_tn(&self, threads: usize, t: usize, f: usize) -> (usize, usize) {
        match self.split {
            Split::Frames => (threads, 1),
            Split::Groups => (1, threads),
            Split::Auto => {
                // The largest tm dividing `threads` with at least one full panel per frame range.
                let mut tm = threads;
                while tm > 1 && (t < tm * f || threads % tm != 0) {
                    tm -= 1;
                }
                (tm, threads / tm)
            }
        }
    }
}

impl Kernel for VSdot {
    type Prepared = Prepared;
    const NAME: &'static str = "v-sdot";
    const FORMAT: WeightFormat = WeightFormat::Binary;

    fn new(cfg: &KernelConfig) -> Self {
        let mut split = Split::Auto;
        let mut narrow = DEFAULT_FMAX;
        let mut wide: [Option<usize>; 4] = [None; 4];
        if let Some(obj) = cfg.tuning.as_object() {
            for (key, v) in obj {
                match key.as_str() {
                    "split" => {
                        split = match v.as_str() {
                            Some("auto") => Split::Auto,
                            Some("frames") => Split::Frames,
                            Some("groups") => Split::Groups,
                            _ => panic!("v-sdot: tuning split {v}"),
                        }
                    }
                    "fmax1" | "fmax2" | "fmax3" | "fmax1_wide" | "fmax2_wide" | "fmax3_wide" => {
                        let c = (key.as_bytes()[4] - b'0') as usize;
                        let f = v.as_u64().expect("fmax integer") as usize;
                        assert!(f >= 1 && f * c <= 16, "v-sdot: {key} = {f} exceeds 16 accumulators");
                        if key.ends_with("_wide") {
                            wide[c] = Some(f);
                        } else {
                            narrow[c] = f;
                        }
                    }
                    _ => panic!("v-sdot: unknown tuning key {key}"),
                }
            }
        }
        let mut slots: [SlotState; 3] = Default::default();
        let comps = cfg.components.clamp(1, 3);
        for s in &mut slots {
            s.u.reserve_exact(cfg.max_rows * comps);
        }
        // The wide caps default to the narrow ones.
        let wide = core::array::from_fn(|c| wide[c].unwrap_or(narrow[c]));
        VSdot { split, fmax: [narrow, wide], slots }
    }

    fn prepare(&self, w: WeightsRef<'_>) -> Prepared {
        let m = match w {
            WeightsRef::Binary(m) => m,
            WeightsRef::Ternary(_) => panic!("v-sdot consumes parakeet-binary-v1"),
        };
        m.validate().expect("v-sdot: invalid binary matrix");
        let (n, k) = (m.n, m.k);
        let (groups, chunks) = (n.div_ceil(GROUP_ROWS), k.div_ceil(CHUNK_K));
        let main = layout::build_main(&m.codes, m.row_bytes(), n, k, groups, chunks);
        let mut bits = AlignedVec::<u8>::zeroed(main.len());
        bits.as_mut_slice().copy_from_slice(&main);
        let mut scale = AlignedVec::<f32>::zeroed(groups * GROUP_ROWS);
        scale.as_mut_slice()[..n].copy_from_slice(&m.scale);
        Prepared { n, k, groups, chunks, bits, scale }
    }

    fn gemm(&mut self, pool: &ThreadPool, w: &Prepared, qa: &QuantizedActs, call: Call, out: Out<'_>) {
        let (t, c, k, n) = (qa.rows(), qa.components(), qa.cols(), w.n);
        assert_eq!(k, w.k, "v-sdot: activation width {k} != matrix inputs {}", w.k);
        assert!((1..=3).contains(&c));
        let (optr, dots) = match out {
            Out::F32(y) => {
                assert!(y.len() >= t * n, "v-sdot: output too short");
                (y.as_mut_ptr() as *mut u8, false)
            }
            Out::Dots(d) => {
                assert!(d.len() >= t * c * n, "v-sdot: dot output too short");
                (d.as_mut_ptr() as *mut u8, true)
            }
        };
        if t == 0 {
            return;
        }
        let threads = pool.threads();
        let fps = t * c;
        let kp = w.chunks * CHUNK_K;
        let padded = k % CHUNK_K != 0;
        let q = qa.q();
        let fmax = self.fmax_for(c, k);
        let (tm, tn) = self.tm_tn(threads, t, fmax);
        let st = &mut self.slots[call.slot.index()];
        let fresh =
            call.reuse && st.valid && st.rows == t && st.comps == c && st.cols == k && st.q_addr == q.as_ptr() as usize;
        let rebuild = !fresh;
        if rebuild {
            st.valid = false;
            if st.u.len() < fps {
                st.u.resize(fps, 0);
            }
            if padded && st.apad.len() < fps * kp {
                st.apad.resize(fps * kp, 0);
            }
        }
        // Activation state of frame-parts `fps`: U and, for ragged K, the zero-padded copy. Each
        // caller passes disjoint frame-parts.
        let up = SendPtr(st.u.as_mut_ptr());
        let ap = SendPtr(st.apad.as_mut_ptr());
        let build = |range: std::ops::Range<usize>| {
            for fp in range {
                let row = &q[fp * k..(fp + 1) * k];
                unsafe { *up.get().add(fp) = part_sum(row) };
                if padded {
                    let dst = unsafe { std::slice::from_raw_parts_mut(ap.get().add(fp * kp), kp) };
                    dst[..k].copy_from_slice(row);
                    dst[k..].fill(0);
                }
            }
        };
        // With frames-only splitting each thread builds exactly the frame-parts it multiplies,
        // inside the GEMM job; otherwise threads sharing frames need a separate pass first.
        let fused = rebuild && tn == 1;
        if rebuild && !fused {
            pool.run(&|idx| build(split_range(fps, threads, idx)));
        }
        let (a, lda) = if padded { (st.apad.as_ptr(), kp) } else { (q.as_ptr(), k) };
        let job = Job {
            bits: w.bits.as_ptr(),
            chunks: w.chunks,
            scale: w.scale.as_ptr(),
            a,
            lda,
            u: st.u.as_ptr(),
            fscale: qa.scale().as_ptr(),
            out: optr,
            n,
        };
        let groups = w.groups;
        pool.run(&|idx| {
            let (im, jn) = (idx / tn, idx % tn);
            let frames = split_range(t, tm, im);
            if fused {
                build(frames.start * c..frames.end * c);
            }
            let gr = split_range(groups, tn, jn);
            if frames.is_empty() || gr.is_empty() {
                return;
            }
            // Balanced panels of at most fmax frames.
            let len = frames.len();
            let panels = len.div_ceil(fmax);
            for p in 0..panels {
                let f = split_range(len, panels, p);
                unsafe { job.panel(c, dots, frames.start + f.start, f.len(), gr.start, gr.end) };
            }
        });
        if rebuild {
            st.rows = t;
            st.comps = c;
            st.cols = k;
            st.q_addr = q.as_ptr() as usize;
            st.valid = true;
        }
    }

    fn unpack(&self, w: &Prepared) -> Vec<i8> {
        let mut out = Vec::with_capacity(w.n * w.k);
        for o in 0..w.n {
            for i in 0..w.k {
                out.push(if layout::main_bit(w.bits.as_slice(), w.chunks, o, i) == 1 { 1 } else { -1 });
            }
        }
        out
    }

    fn shape(&self, w: &Prepared) -> (usize, usize) {
        (w.n, w.k)
    }

    fn weight_bytes(&self, w: &Prepared) -> WeightBytes {
        WeightBytes { codes: w.bits.heap_bytes(), per_row: w.scale.heap_bytes() }
    }

    fn workspace_bytes(&self) -> usize {
        self.slots.iter().map(SlotState::bytes).sum()
    }

    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "layout": "v-layout: 4-row groups x 32-input chunks, 16 B each, bit j of byte 4nl+kl = (row nl, input 4j+kl)",
            "expand": "cmtst vs 1<<j (cmlt #0 for bit 7) -> {0,-1} bytes; d = -2*sdot - U",
            "microtile": "4 rows x R = F*C frame-parts (R <= 16), inline asm, by-element sdot",
            "split": format!("{:?}", self.split),
            "fmax": {"1": self.fmax[0][1], "2": self.fmax[0][2], "3": self.fmax[0][3]},
            "fmax_wide": {"wide_k_above": WIDE_K, "1": self.fmax[1][1], "2": self.fmax[1][2], "3": self.fmax[1][3]},
        })
    }
}

/// `Σ_k q_k` exactly (|sum| <= 128 k).
fn part_sum(row: &[i8]) -> i32 {
    #[cfg(target_arch = "aarch64")]
    {
        kernel::part_sum(row)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        row.iter().map(|&v| v as i32).sum()
    }
}

/// Everything a pool thread needs for one call.
#[derive(Clone, Copy)]
struct Job {
    bits: *const u8,
    chunks: usize,
    scale: *const f32,
    /// Frame-part 0 of the call; frame-part `fp` at `a + fp * lda`.
    a: *const i8,
    lda: usize,
    u: *const i32,
    fscale: *const f32,
    /// `y` (F32, `[T][N]`) or `d` (Dots, `[T][C][N]`).
    out: *mut u8,
    n: usize,
}

// The pointers are read-only except `out`, whose (frame, group) regions are disjoint per thread.
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

#[cfg(target_arch = "aarch64")]
macro_rules! panel_dispatch {
    ($args:expr, $c:literal, $dots:ident, $f:ident, $t0:ident, $g0:ident, $g1:ident; $($fv:literal => $r:literal),*) => {
        match ($f, $dots) {
            $(
                ($fv, false) => kernel::panel::<$c, $fv, $r, false>($args, $t0, $g0, $g1),
                ($fv, true) => kernel::panel::<$c, $fv, $r, true>($args, $t0, $g0, $g1),
            )*
            _ => unreachable!("panel of {} frames for C = {}", $f, $c),
        }
    };
}

impl Job {
    /// Frames `t0 .. t0 + f` (1 <= f <= fmax) against row groups `g0 .. g1`.
    #[cfg(target_arch = "aarch64")]
    unsafe fn panel(&self, c: usize, dots: bool, t0: usize, f: usize, g0: usize, g1: usize) {
        let args = kernel::PanelArgs {
            bits: self.bits,
            chunks: self.chunks,
            scale: self.scale,
            a: self.a,
            lda: self.lda,
            u: self.u,
            fscale: self.fscale,
            out: self.out,
            n: self.n,
        };
        unsafe {
            match c {
                1 => panel_dispatch!(&args, 1, dots, f, t0, g0, g1;
                    1 => 1, 2 => 2, 3 => 3, 4 => 4, 5 => 5, 6 => 6, 7 => 7, 8 => 8,
                    9 => 9, 10 => 10, 11 => 11, 12 => 12, 13 => 13, 14 => 14, 15 => 15, 16 => 16),
                2 => panel_dispatch!(&args, 2, dots, f, t0, g0, g1;
                    1 => 2, 2 => 4, 3 => 6, 4 => 8, 5 => 10, 6 => 12, 7 => 14, 8 => 16),
                3 => panel_dispatch!(&args, 3, dots, f, t0, g0, g1;
                    1 => 3, 2 => 6, 3 => 9, 4 => 12, 5 => 15),
                _ => unreachable!(),
            }
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    unsafe fn panel(&self, _c: usize, _dots: bool, _t0: usize, _f: usize, _g0: usize, _g1: usize) {
        panic!("v-sdot runs on aarch64 only");
    }
}

/// V's frozen tuning grid (`grid.json`, 6 points = B-NEON's grid size; point 0 = the default =
/// Null tuning). Pinned by SHA-256; the unit tests fail if the file changes.
pub const GRID_JSON: &str = include_str!("../grid.json");
pub const GRID_SHA256: &str = "4980267adfe0d821598abd6f812fa67805fdf5ac758482b57a2e0fcaa8f9aa65";

/// The grid's points (JSON objects for `KernelConfig::tuning`).
pub fn grid() -> Vec<serde_json::Value> {
    let doc: serde_json::Value = serde_json::from_str(GRID_JSON).expect("grid.json parses");
    doc["points"].as_array().expect("grid.json points").clone()
}

/// A fixed tuning point, so H's unchanged contract harness can test every grid point
/// (`tests/grid.rs`).
pub trait Preset: Send + Sync + 'static {
    const NAME: &'static str;
    fn tuning() -> serde_json::Value;
}

/// [`VSdot`] at the preset's tuning point, whatever `KernelConfig::tuning` says.
pub struct VSdotWith<P: Preset> {
    inner: VSdot,
    _p: std::marker::PhantomData<P>,
}

impl<P: Preset> Kernel for VSdotWith<P> {
    type Prepared = Prepared;
    const NAME: &'static str = P::NAME;
    const FORMAT: WeightFormat = WeightFormat::Binary;

    fn new(cfg: &KernelConfig) -> Self {
        let mut cfg = cfg.clone();
        cfg.tuning = P::tuning();
        VSdotWith { inner: VSdot::new(&cfg), _p: std::marker::PhantomData }
    }
    fn prepare(&self, w: WeightsRef<'_>) -> Prepared {
        self.inner.prepare(w)
    }
    fn gemm(&mut self, pool: &ThreadPool, w: &Prepared, qa: &QuantizedActs, call: Call, out: Out<'_>) {
        self.inner.gemm(pool, w, qa, call, out)
    }
    fn unpack(&self, w: &Prepared) -> Vec<i8> {
        self.inner.unpack(w)
    }
    fn shape(&self, w: &Prepared) -> (usize, usize) {
        self.inner.shape(w)
    }
    fn weight_bytes(&self, w: &Prepared) -> WeightBytes {
        self.inner.weight_bytes(w)
    }
    fn workspace_bytes(&self) -> usize {
        self.inner.workspace_bytes()
    }
    fn describe(&self) -> serde_json::Value {
        self.inner.describe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(tuning: serde_json::Value) -> KernelConfig {
        KernelConfig { threads: 1, components: 2, max_rows: 8, max_k: 64, max_n: 64, tuning }
    }

    #[test]
    fn grid_is_frozen_six_distinct_points_with_the_default_first() {
        assert_eq!(shared::data::sha256_hex(GRID_JSON.as_bytes()), GRID_SHA256, "grid.json changed after the freeze");
        let points = grid();
        assert_eq!(points.len(), 6, "grid size must equal B-NEON's (6)");
        let described: Vec<_> = points.iter().map(|p| VSdot::new(&cfg(p.clone())).describe()).collect();
        for i in 0..described.len() {
            for j in 0..i {
                assert_ne!(described[i], described[j], "grid points {j} and {i} configure the same kernel");
            }
        }
        assert_eq!(described[0], VSdot::new(&cfg(serde_json::Value::Null)).describe(), "point 0 must be the default");
    }
}
