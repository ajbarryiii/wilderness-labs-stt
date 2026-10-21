//! THE INTERFACE every arm implements (B-NEON, V, T, P; plans/cpu-benchmark-mac.md "Arms").
//!
//! One projection call, as LocalFlow's `Projector::run` (lf-asr/src/encoder.rs:112-144):
//! 1. activation quantization into residual INT8 parts — done by the replay with the shared
//!    [`crate::quant::QuantizedActs::quantize`] (identical for every arm, inside the timed scope),
//!    skipped when the schedule says `reuse` (Q -> K, V share one quantization);
//! 2. the GEMM — [`Kernel::gemm`], which may do any per-call preparation it needs (tables, planes,
//!    expanded tiles: all inside the timed scope);
//! 3. the FP32 epilogue — inside `gemm`, through [`crate::epilogue`], exact order.
//!
//! Contract for `gemm(w, qa, call, out)` with `T = qa.rows()`, `C = qa.components()`,
//! `K = qa.cols()`, `N` = the matrix's outputs, `q = qa.q()` (s8, the full range [-128, 127]):
//! - `d[t][c][o] = Σ_{i<K} q[t][c][i] · w[o][i]` exactly, with `w` the {-1, 0, +1} code values;
//! - `Out::F32(y)`: `y[t * N + o] = epilogue::finish(&d[t][..][o], scale[o], qa.scale()[t])`,
//!   bit-identical; `y.len() >= T * N`; elements past `T * N` untouched;
//! - `Out::Dots(d)`: the raw `d[(t * C + c) * N + o]`; the SAME code path as `F32` up to the final
//!   store (monomorphise one inner routine over the output kind; do not write a second kernel);
//! - any T >= 0 (T = 0 is a no-op), any K >= 1 and N >= 1 (ragged tails handled), C in 1..=3;
//! - `call.reuse == true` promises `qa` is the same object, unchanged, as in the previous call
//!   with this `call.slot`, so activation-derived state for that slot may be reused;
//!   `reuse == false` means it must be rebuilt (the contract tests change inputs, lengths and
//!   components between calls to catch stale state);
//! - threads: use `pool.run` with `split_range`; every pool index must return; results must not
//!   depend on the thread count (tested with 1, 4 and 8);
//! - `gemm` must not allocate on the timed path after the first call at the largest shape
//!   (allocate in `new` from `KernelConfig`, or grow once).

use crate::pool::ThreadPool;
use crate::quant::QuantizedActs;
use crate::weights::{WeightFormat, WeightsRef};

/// LocalFlow's quantization slots (encoder.rs `Slot`): inputs of different widths or lifetimes
/// never overwrite each other. `Pos` exists for completeness; `linear_pos` is cached and not
/// replayed (schedule.json).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Slot {
    /// `D_MODEL`-wide inputs (LN outputs, attention output, depthwise output).
    Model = 0,
    /// `D_FF`-wide inputs (silu(linear1) of the feed-forward modules).
    Ff = 1,
    /// The relative position embedding (cached; not replayed).
    Pos = 2,
}

impl Slot {
    pub const ALL: [Slot; 3] = [Slot::Model, Slot::Ff, Slot::Pos];
    pub fn index(self) -> usize {
        self as usize
    }
    pub fn parse(s: &str) -> Option<Slot> {
        match s {
            "model" => Some(Slot::Model),
            "ff" => Some(Slot::Ff),
            "pos" => Some(Slot::Pos),
            _ => None,
        }
    }
}

/// Per-call information from the schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Call {
    pub slot: Slot,
    pub reuse: bool,
}

/// Where the results go.
pub enum Out<'a> {
    /// `[T][N]` FP32 after the epilogue.
    F32(&'a mut [f32]),
    /// `[T][C][N]` exact integer dots, no epilogue.
    Dots(&'a mut [i32]),
}

/// Process-wide sizing, known before any timing.
#[derive(Clone, Debug)]
pub struct KernelConfig {
    /// Pool threads (1, 4 or 8 in the contract tests; 8 or 4 in timed blocks).
    pub threads: usize,
    /// Activation components the process will use (2 = i8x2, 3 = i8x3; tests use 1..=3).
    pub components: usize,
    /// Upper bounds over every call of the process.
    pub max_rows: usize,
    pub max_k: usize,
    pub max_n: usize,
    /// Kernel-specific tuning point from the frozen grid (JSON object), or Null for the default.
    pub tuning: serde_json::Value,
}

/// Weight storage accounting for one prepared matrix (plans/cpu-benchmark.md "Storage").
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WeightBytes {
    /// Bytes holding weight codes in any form (packed bits, tiles, padding, auxiliary code tables).
    pub codes: usize,
    /// Scales, row sums / popcounts and other per-row values (reported, excluded from the limit).
    pub per_row: usize,
}

pub trait Kernel: Send + Sized {
    /// Prepared weights of one projection, built once per process before any timing.
    type Prepared: Send + Sync;

    /// Arm name used in records, e.g. "b-neon", "v-sdot".
    const NAME: &'static str;
    /// The export layout `prepare` consumes (B-NEON: Ternary; V, T, P: Binary).
    const FORMAT: WeightFormat;

    fn new(cfg: &KernelConfig) -> Self;

    /// Packs one matrix. Receives `WeightsRef` of `Self::FORMAT`; may keep no reference to it
    /// (the export buffer is released after loading). Panics on unsupported input.
    fn prepare(&self, w: WeightsRef<'_>) -> Self::Prepared;

    /// One projection: see the module docs for the exact contract.
    fn gemm(&mut self, pool: &ThreadPool, w: &Self::Prepared, qa: &QuantizedActs, call: Call, out: Out<'_>);

    /// The prepared codes unpacked to `[N][K]` values in {-1, 0, +1} ("native equals derived").
    fn unpack(&self, w: &Self::Prepared) -> Vec<i8>;

    fn shape(&self, w: &Self::Prepared) -> (usize, usize);

    fn weight_bytes(&self, w: &Self::Prepared) -> WeightBytes;

    /// Peak per-process workspace bytes (tiles, tables, per-slot activation state).
    fn workspace_bytes(&self) -> usize;

    /// Variant and tuning description recorded with every block.
    fn describe(&self) -> serde_json::Value {
        serde_json::Value::Null
    }
}
