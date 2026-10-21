//! The projection replay (plans/cpu-benchmark-mac.md "Scope", M1, M2): for one utterance, the 240
//! replayed calls of schedule.json in order, each as LocalFlow's `Projector::run` — quantize the
//! captured input into the call's slot unless `reuse`, then the arm's GEMM with the FP32 epilogue.
//! The only GEMM path in a replay process is `Kernel::gemm`; the replayer counts every call
//! (dispatch audit) and can record a trace (M1 trace contracts) or per-call digests (exactness).

use crate::aligned::AlignedVec;
use crate::data::{Matrix, sha256_hex};
use crate::kernel::{Call, Kernel, KernelConfig, Out, Slot};
use crate::pool::ThreadPool;
use crate::quant::{NonFiniteActivation, QuantizedActs};
use crate::schedule::{Schedule, ScheduledCall};

/// Captured inputs of one utterance.
pub trait Inputs {
    fn t(&self) -> usize;
    fn input(&self, name: &str) -> &[f32];
}

impl Inputs for crate::data::Clip {
    fn t(&self) -> usize {
        self.t
    }
    fn input(&self, name: &str) -> &[f32] {
        crate::data::Clip::input(self, name)
    }
}

/// Synthetic inputs (tests): every capture of the schedule, Gaussian `[T][K]`.
pub struct SyntheticInputs {
    pub t: usize,
    pub data: std::collections::HashMap<String, Vec<f32>>,
}

impl SyntheticInputs {
    pub fn new(sched: &Schedule, t: usize, seed: u64) -> Self {
        let mut rng = crate::rng::SplitMix64::new(seed);
        let data = sched
            .captures
            .iter()
            .map(|c| (c.name.clone(), (0..t * c.k).map(|i| rng.next_gaussian() * if i % 41 == 0 { 9.0 } else { 1.0 }).collect()))
            .collect();
        SyntheticInputs { t, data }
    }
}

impl Inputs for SyntheticInputs {
    fn t(&self) -> usize {
        self.t
    }
    fn input(&self, name: &str) -> &[f32] {
        &self.data[name]
    }
}

/// One recorded call (trace contract).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceEvent {
    pub index: usize,
    pub layer: usize,
    pub module: String,
    pub n: usize,
    pub k: usize,
    pub rows: usize,
    pub components: usize,
    pub slot: Slot,
    pub reuse: bool,
    /// The capture quantized for this call (None when reused).
    pub quantized: Option<String>,
    /// The capture whose quantization the GEMM consumed.
    pub input: String,
    /// Output buffer length the GEMM was given (rows * n).
    pub out_len: usize,
}

/// Per-call digests of the FP32 output bits and of the exact integer dots.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallDigest {
    pub index: usize,
    pub module: String,
    pub y_sha256: String,
    pub d_sha256: String,
}

pub enum Mode<'a> {
    Timed,
    Trace(&'a mut Vec<TraceEvent>),
    Digest(&'a mut Vec<CallDigest>),
}

pub struct Replayer<K: Kernel> {
    pub kernel: K,
    pub prepared: Vec<K::Prepared>,
    pub calls: Vec<ScheduledCall>,
    slots: [QuantizedActs; 2],
    slot_input: [Option<String>; 2],
    y: AlignedVec<f32>,
    d: Vec<i32>,
    /// Dispatch audit: Kernel::gemm invocations and quantizations since construction.
    pub gemm_calls: u64,
    pub quantizations: u64,
    pub components: usize,
}

fn slot_index(s: Slot) -> usize {
    match s {
        Slot::Model => 0,
        Slot::Ff => 1,
        Slot::Pos => panic!("linear_pos is cached and never replayed"),
    }
}

pub fn f32_digest(v: &[f32]) -> String {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    sha256_hex(&b)
}

pub fn i32_digest(v: &[i32]) -> String {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    sha256_hex(&b)
}

impl<K: Kernel> Replayer<K> {
    /// The replayer's own live heap (quantization slots, output and dot buffers, the call list): the
    /// harness part of the storage audit (plans/cpu-benchmark-mac.md "Storage and workspace"). The
    /// vector of prepared matrices is deliberately not counted: it is part of the arm's weights.
    pub fn harness_bytes(&self) -> usize {
        let strings: usize = self
            .calls
            .iter()
            .map(|c| c.module.capacity() + c.short.capacity() + c.quant.as_ref().map_or(0, |q| q.capacity()))
            .sum();
        self.slots.iter().map(|s| s.heap_bytes()).sum::<usize>()
            + self.slot_input.iter().map(|s| s.as_ref().map_or(0, |x| x.capacity())).sum::<usize>()
            + self.y.heap_bytes()
            + self.d.capacity() * std::mem::size_of::<i32>()
            + self.calls.capacity() * std::mem::size_of::<ScheduledCall>()
            + strings
    }

    /// Prepares every replayed matrix (schedule order) and frees the source matrices.
    pub fn new(cfg: &KernelConfig, matrices: Vec<Matrix>, sched: &Schedule) -> Self {
        let calls: Vec<ScheduledCall> = sched.replayed().cloned().collect();
        assert_eq!(matrices.len(), calls.len());
        let kernel = K::new(cfg);
        let mut prepared = Vec::with_capacity(calls.len());
        for (m, c) in matrices.into_iter().zip(&calls) {
            let p = kernel.prepare(m.as_ref());
            assert_eq!(kernel.shape(&p), (c.n, c.k), "{}", c.module);
            prepared.push(p);
            drop(m);
        }
        let model_k = calls.iter().find(|c| c.slot == Slot::Model).unwrap().k;
        let ff_k = calls.iter().find(|c| c.slot == Slot::Ff).unwrap().k;
        let max_n = calls.iter().map(|c| c.n).max().unwrap();
        Replayer {
            kernel,
            prepared,
            calls,
            slots: [
                QuantizedActs::with_capacity(cfg.max_rows, model_k, cfg.components),
                QuantizedActs::with_capacity(cfg.max_rows, ff_k, cfg.components),
            ],
            slot_input: [None, None],
            y: AlignedVec::zeroed(cfg.max_rows * max_n),
            d: Vec::new(),
            gemm_calls: 0,
            quantizations: 0,
            components: cfg.components,
        }
    }

    /// Replays one utterance. Inputs must hold every capture of the schedule at `t()` rows.
    pub fn replay(&mut self, pool: &ThreadPool, inputs: &dyn Inputs, mut mode: Mode<'_>) -> Result<(), NonFiniteActivation> {
        let t = inputs.t();
        let max_n = self.calls.iter().map(|c| c.n).max().unwrap();
        if self.y.len() < t * max_n {
            self.y = AlignedVec::zeroed(t * max_n);
        }
        for i in 0..self.calls.len() {
            let c = &self.calls[i];
            let s = slot_index(c.slot);
            let quant = c.quant.as_ref().expect("replayed calls have a capture");
            let quantized = if c.reuse {
                assert_eq!(self.slot_input[s].as_deref(), Some(quant.as_str()), "{}: reuse of a different quantization", c.module);
                None
            } else {
                let x = inputs.input(quant);
                assert_eq!(x.len(), t * c.k, "{}: capture {quant} has {} floats, want {}x{}", c.module, x.len(), t, c.k);
                self.slots[s].quantize(pool, x, t)?;
                self.slot_input[s] = Some(quant.clone());
                self.quantizations += 1;
                Some(quant.clone())
            };
            let qa = &self.slots[s];
            debug_assert_eq!(qa.cols(), c.k);
            let call = Call { slot: c.slot, reuse: c.reuse };
            let y = &mut self.y.as_mut_slice()[..t * c.n];
            self.kernel.gemm(pool, &self.prepared[i], qa, call, Out::F32(y));
            self.gemm_calls += 1;
            match &mut mode {
                Mode::Timed => {}
                Mode::Trace(events) => events.push(TraceEvent {
                    index: c.index,
                    layer: c.layer,
                    module: c.module.clone(),
                    n: c.n,
                    k: c.k,
                    rows: qa.rows(),
                    components: qa.components(),
                    slot: c.slot,
                    reuse: c.reuse,
                    quantized,
                    input: quant.clone(),
                    out_len: t * c.n,
                }),
                Mode::Digest(out) => {
                    let y_sha256 = f32_digest(&self.y.as_slice()[..t * c.n]);
                    let need = t * self.components * c.n;
                    if self.d.len() < need {
                        self.d.resize(need, 0);
                    }
                    self.kernel.gemm(pool, &self.prepared[i], qa, Call { slot: c.slot, reuse: true }, Out::Dots(&mut self.d[..need]));
                    self.gemm_calls += 1;
                    out.push(CallDigest { index: c.index, module: c.module.clone(), y_sha256, d_sha256: i32_digest(&self.d[..need]) });
                }
            }
        }
        Ok(())
    }

    /// Output of the last call (tests).
    pub fn last_output(&self, len: usize) -> &[f32] {
        &self.y.as_slice()[..len]
    }
}

/// The oracle's per-call digests for one utterance: the same quantizer, then `oracle::dots`
/// (parallel over output rows, `threads` scoped threads) and `oracle::outputs`.
pub fn oracle_digests(matrices: &[Matrix], sched: &Schedule, inputs: &dyn Inputs, components: usize, threads: usize) -> Vec<CallDigest> {
    let calls: Vec<&ScheduledCall> = sched.replayed().collect();
    assert_eq!(calls.len(), matrices.len());
    let pool = ThreadPool::new(1, crate::pool::Qos::Inherit);
    let t = inputs.t();
    let mut out = Vec::new();
    let mut slots: [Option<QuantizedActs>; 2] = [None, None];
    for (c, m) in calls.iter().zip(matrices) {
        let s = slot_index(c.slot);
        if !c.reuse {
            let mut qa = QuantizedActs::with_capacity(t, c.k, components);
            qa.quantize(&pool, inputs.input(c.quant.as_ref().unwrap()), t).expect("finite captures");
            slots[s] = Some(qa);
        }
        let qa = slots[s].as_ref().unwrap();
        let w = m.as_ref();
        let (n, k) = (w.n(), w.k());
        let values = w.values();
        // Parallel over output-row chunks; each chunk is the scalar oracle on a row slice.
        let chunk = n.div_ceil(threads);
        let parts: Vec<Vec<i32>> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..threads)
                .map(|ti| {
                    let (lo, hi) = ((ti * chunk).min(n), ((ti + 1) * chunk).min(n));
                    let vals = &values[lo * k..hi * k];
                    sc.spawn(move || crate::oracle::dots(vals, hi - lo, k, qa))
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut d = vec![0i32; t * components * n];
        for (ti, part) in parts.iter().enumerate() {
            let lo = (ti * chunk).min(n);
            let w_n = ((ti + 1) * chunk).min(n) - lo;
            for r in 0..t * components {
                d[r * n + lo..r * n + lo + w_n].copy_from_slice(&part[r * w_n..(r + 1) * w_n]);
            }
        }
        let y = crate::oracle::outputs(&d, n, w.scale(), qa);
        out.push(CallDigest { index: c.index, module: c.module.clone(), y_sha256: f32_digest(&y), d_sha256: i32_digest(&d) });
    }
    out
}

pub fn digests_to_json(d: &[CallDigest]) -> serde_json::Value {
    serde_json::Value::Array(
        d.iter().map(|c| serde_json::json!({"index": c.index, "module": c.module, "y": c.y_sha256, "d": c.d_sha256})).collect(),
    )
}

pub fn digests_from_json(v: &serde_json::Value) -> Vec<CallDigest> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|c| CallDigest {
            index: c["index"].as_u64().unwrap() as usize,
            module: c["module"].as_str().unwrap().to_string(),
            y_sha256: c["y"].as_str().unwrap().to_string(),
            d_sha256: c["d"].as_str().unwrap().to_string(),
        })
        .collect()
}
