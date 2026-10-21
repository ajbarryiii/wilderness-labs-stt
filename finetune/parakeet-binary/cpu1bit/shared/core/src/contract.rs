//! Contract tests every kernel crate runs unchanged (plans/cpu-benchmark.md "Contract tests",
//! plans/cpu-benchmark-mac.md "Exactness", M1, M8). A kernel crate registers by adding
//!
//! ```ignore
//! // <crate>/tests/contract.rs
//! shared::contract_tests!(my_crate::MyKernel);
//! ```
//!
//! Every check compares BOTH the exact integer dots (`Out::Dots`) and the FP32 outputs (`Out::F32`,
//! compared bit for bit) with the independent scalar oracle. These tests are the acceptance
//! criteria: kernels are fixed to satisfy them, never the reverse.

use crate::kernel::{Call, Kernel, KernelConfig, Out, Slot};
use crate::oracle;
use crate::pool::{Qos, ThreadPool};
use crate::quant::QuantizedActs;
use crate::rng::SplitMix64;
use crate::weights::{BinaryMatrix, TernaryMatrix, WeightFormat, WeightsRef};

/// The thread counts of the contract (plans/cpu-benchmark-mac.md "Exactness": 1, 4 and 8).
pub const THREADS: [usize; 3] = [1, 4, 8];

/// Test pools run with the timed QoS on macOS and inherit elsewhere.
pub fn test_pool(threads: usize) -> ThreadPool {
    ThreadPool::new(threads, if cfg!(target_os = "macos") { Qos::UserInteractive } else { Qos::Inherit })
}

/// A weight matrix in both layouts when it is binary, ternary only otherwise.
pub struct TestMatrix {
    pub ternary: TernaryMatrix,
    pub binary: Option<BinaryMatrix>,
}

impl TestMatrix {
    pub fn binary(b: BinaryMatrix) -> Self {
        TestMatrix { ternary: b.to_ternary(), binary: Some(b) }
    }
    pub fn as_ref(&self, format: WeightFormat) -> Option<WeightsRef<'_>> {
        match format {
            WeightFormat::Ternary => Some(WeightsRef::Ternary(&self.ternary)),
            WeightFormat::Binary => self.binary.as_ref().map(WeightsRef::Binary),
        }
    }
    pub fn n(&self) -> usize {
        self.ternary.n
    }
    pub fn k(&self) -> usize {
        self.ternary.k
    }
}

/// Sign patterns of the contract (all +1, all -1, alternating, a single -1, random).
pub fn sign_patterns(n: usize, k: usize, seed: u64) -> Vec<(&'static str, BinaryMatrix)> {
    let mut rng = SplitMix64::new(seed);
    let scale: Vec<f32> = (0..n).map(|_| 0.01 + 0.04 * rng.next_f32()).collect();
    let pat = |f: &dyn Fn(usize, usize) -> i8| {
        let s: Vec<i8> = (0..n * k).map(|j| f(j / k, j % k)).collect();
        BinaryMatrix::from_signs(n, k, &s, scale.clone())
    };
    vec![
        ("all+1", pat(&|_, _| 1)),
        ("all-1", pat(&|_, _| -1)),
        ("alternating", pat(&|o, i| if (o + i) % 2 == 0 { 1 } else { -1 })),
        ("single-1", pat(&|o, i| if i == (o * 7919) % k { -1 } else { 1 })),
        ("random", BinaryMatrix::random(n, k, seed ^ 0x5eed)),
    ]
}

/// Activation patterns: (name, parts [T][C][K], frame scales).
pub fn activation_patterns(t: usize, c: usize, k: usize, seed: u64) -> Vec<(&'static str, QuantizedActs)> {
    let mut rng = SplitMix64::new(seed);
    let scales: Vec<f32> = (0..t).map(|i| if i % 5 == 3 { 0.0 } else { 1e-3 + rng.next_f32() }).collect();
    let make = |f: &mut dyn FnMut(usize) -> i8| {
        let q: Vec<i8> = (0..t * c * k).map(&mut *f).collect();
        QuantizedActs::from_parts(t, k, c, &q, &scales)
    };
    let mut r2 = SplitMix64::new(seed ^ 77);
    vec![
        ("random-s8", make(&mut |_| r2.next_u64() as i8)),
        ("all-128", make(&mut |_| -128)),
        ("all+127", make(&mut |_| 127)),
        ("all0", make(&mut |_| 0)),
        ("alternating-128/127", make(&mut |j| if j % 2 == 0 { -128 } else { 127 })),
    ]
}

/// Real quantizer output from Gaussian frames with outliers (the production path).
pub fn quantized_gaussian(pool: &ThreadPool, t: usize, c: usize, k: usize, seed: u64) -> QuantizedActs {
    let mut rng = SplitMix64::new(seed);
    let x: Vec<f32> = (0..t * k).map(|i| rng.next_gaussian() * if i % 37 == 0 { 20.0 } else { 1.0 }).collect();
    let mut qa = QuantizedActs::with_capacity(t, k, c);
    qa.quantize(pool, &x, t).unwrap();
    qa
}

fn config(threads: usize, components: usize, max_rows: usize, max_k: usize, max_n: usize) -> KernelConfig {
    KernelConfig { threads, components, max_rows, max_k, max_n, tuning: serde_json::Value::Null }
}

/// Runs one call in both output modes and compares with the oracle bit for bit. The output
/// buffers carry a sentinel tail that must stay untouched.
pub fn check_call<K: Kernel>(
    kernel: &mut K,
    pool: &ThreadPool,
    prepared: &K::Prepared,
    m: &TestMatrix,
    qa: &QuantizedActs,
    call: Call,
    what: &str,
) {
    let (n, k) = (m.n(), m.k());
    let values = m.ternary.values();
    let want_d = oracle::dots(&values, n, k, qa);
    let want_y = oracle::outputs(&want_d, n, &m.ternary.scale, qa);
    let (t, c) = (qa.rows(), qa.components());
    const SENTINEL: usize = 17;
    let mut d = vec![i32::MIN + 1; t * c * n + SENTINEL];
    kernel.gemm(pool, prepared, qa, call, Out::Dots(&mut d));
    if let Some(i) = (0..t * c * n).find(|&i| d[i] != want_d[i]) {
        let (tt, cc, o) = (i / (c * n), (i / n) % c, i % n);
        panic!("{}: {what}: dot mismatch at t={tt} c={cc} o={o}: got {} want {}", K::NAME, d[i], want_d[i]);
    }
    assert!(d[t * c * n..].iter().all(|&v| v == i32::MIN + 1), "{}: {what}: wrote past T*C*N dots", K::NAME);
    // F32 runs as a reuse of the same quantization when the call allows it, so a kernel's cached
    // activation state is exercised too.
    let mut y = vec![f32::from_bits(0x7fc0_dead); t * n + SENTINEL];
    kernel.gemm(pool, prepared, qa, Call { slot: call.slot, reuse: true }, Out::F32(&mut y));
    if let Some(i) = (0..t * n).find(|&i| y[i].to_bits() != want_y[i].to_bits()) {
        panic!("{}: {what}: f32 mismatch at t={} o={}: got {:e} ({:#x}) want {:e} ({:#x})", K::NAME, i / n, i % n, y[i],
               y[i].to_bits(), want_y[i], want_y[i].to_bits());
    }
    assert!(y[t * n..].iter().all(|v| v.to_bits() == 0x7fc0_dead), "{}: {what}: wrote past T*N outputs", K::NAME);
}

fn matrices_for<K: Kernel>(n: usize, k: usize, seed: u64) -> Vec<(String, TestMatrix)> {
    let mut out: Vec<(String, TestMatrix)> =
        sign_patterns(n, k, seed).into_iter().map(|(name, b)| (name.to_string(), TestMatrix::binary(b))).collect();
    if K::FORMAT == WeightFormat::Ternary {
        out.push(("ternary-random".into(), TestMatrix { ternary: TernaryMatrix::random(n, k, seed ^ 3), binary: None }));
    }
    out
}

/// Ragged and extremal shapes, every sign and activation pattern, C = 1..=3, threads 1, 4, 8.
pub fn ragged_and_extremal<K: Kernel>() {
    let shapes: &[(usize, usize, usize)] = &[
        // (N, K, T)
        (1, 1, 1),
        (1, 7, 3),
        (5, 63, 2),
        (63, 65, 5),
        (65, 100, 1),
        (64, 64, 0),
        (128, 1000, 7),
        (130, 1031, 9),
        (192, 2056, 4),
        (17, 4096, 3),
    ];
    for &threads in &THREADS {
        let pool = test_pool(threads);
        for c in 1..=3 {
            let mut kernel = K::new(&config(threads, c, 16, 4096, 256));
            for (si, &(n, k, t)) in shapes.iter().enumerate() {
                for (wname, m) in matrices_for::<K>(n, k, 100 + si as u64) {
                    let w = m.as_ref(K::FORMAT).unwrap();
                    let p = kernel.prepare(w);
                    assert_eq!(kernel.shape(&p), (n, k));
                    for (aname, qa) in activation_patterns(t, c, k, 7 + si as u64) {
                        let what = format!("threads={threads} C={c} N={n} K={k} T={t} w={wname} a={aname}");
                        check_call(&mut kernel, &pool, &p, &m, &qa, Call { slot: Slot::Model, reuse: false }, &what);
                    }
                    let qa = quantized_gaussian(&pool, t, c, k, 9 + si as u64);
                    let what = format!("threads={threads} C={c} N={n} K={k} T={t} w={wname} a=quantized");
                    check_call(&mut kernel, &pool, &p, &m, &qa, Call { slot: Slot::Ff, reuse: false }, &what);
                }
            }
        }
    }
}

/// The four model shapes at several frame counts, i8x2 and i8x3, threads 1, 4, 8.
pub fn model_shapes<K: Kernel>() {
    let shapes = crate::schedule::Schedule::load().shapes();
    for &threads in &THREADS {
        let pool = test_pool(threads);
        for c in [2, 3] {
            let mut kernel = K::new(&config(threads, c, 67, 4096, 4096));
            for (si, &(n, k)) in shapes.iter().enumerate() {
                let m = TestMatrix::binary(BinaryMatrix::random(n, k, 40 + si as u64));
                let p = kernel.prepare(m.as_ref(K::FORMAT).unwrap());
                for t in [1, 6, 67] {
                    let qa = quantized_gaussian(&pool, t, c, k, (si * 100 + t) as u64);
                    let what = format!("threads={threads} C={c} N={n} K={k} T={t}");
                    check_call(&mut kernel, &pool, &p, &m, &qa, Call { slot: Slot::Model, reuse: false }, &what);
                }
            }
        }
    }
}

/// Slot / reuse semantics with changed inputs and lengths (stale tables or tiles are caught):
/// Q quantizes, K and V reuse; then the same slot is requantized with different values and a
/// different T; the Ff slot is interleaved.
pub fn reuse_with_changed_inputs<K: Kernel>() {
    let (k_model, k_ff, n) = (96, 160, 72);
    for &threads in &THREADS {
        let pool = test_pool(threads);
        for c in [2, 3] {
            let mut kernel = K::new(&config(threads, c, 13, k_ff, n));
            let mats: Vec<TestMatrix> =
                (0..3).map(|s| TestMatrix::binary(BinaryMatrix::random(n, k_model, 500 + s))).collect();
            let ff = TestMatrix::binary(BinaryMatrix::random(n, k_ff, 599));
            let preps: Vec<_> = mats.iter().map(|m| kernel.prepare(m.as_ref(K::FORMAT).unwrap())).collect();
            let pff = kernel.prepare(ff.as_ref(K::FORMAT).unwrap());
            let mut qm = QuantizedActs::with_capacity(13, k_model, c);
            let mut qf = QuantizedActs::with_capacity(13, k_ff, c);
            let mut rng = SplitMix64::new(threads as u64 * 10 + c as u64);
            for (round, t) in [5usize, 5, 13, 1, 8].into_iter().enumerate() {
                let x: Vec<f32> = (0..t * k_model).map(|_| rng.next_gaussian() * (1 + round) as f32).collect();
                qm.quantize(&pool, &x, t).unwrap();
                for (j, (m, p)) in mats.iter().zip(&preps).enumerate() {
                    let what = format!("reuse threads={threads} C={c} round={round} T={t} qkv={j}");
                    check_call(&mut kernel, &pool, p, m, &qm, Call { slot: Slot::Model, reuse: j > 0 }, &what);
                }
                let xf: Vec<f32> = (0..(t + 1) * k_ff).map(|_| rng.next_gaussian()).collect();
                qf.quantize(&pool, &xf, t + 1).unwrap();
                let what = format!("reuse threads={threads} C={c} round={round} ff");
                check_call(&mut kernel, &pool, &pff, &ff, &qf, Call { slot: Slot::Ff, reuse: false }, &what);
                // The Model slot again, reused after the Ff call: still the same qm.
                let what = format!("reuse threads={threads} C={c} round={round} model-after-ff");
                check_call(&mut kernel, &pool, &preps[0], &mats[0], &qm, Call { slot: Slot::Model, reuse: true }, &what);
            }
        }
    }
}

/// The prepared codes unpack to exactly the derived 2-bit codes (and to the binary signs).
pub fn native_equals_derived<K: Kernel>() {
    let kernel = K::new(&config(1, 2, 4, 4096, 4096));
    let shapes = crate::schedule::Schedule::load().shapes();
    for (si, &(n, k)) in shapes.iter().chain([(65, 1031), (3, 9)].iter()).enumerate() {
        let b = BinaryMatrix::random(n, k, 900 + si as u64);
        let m = TestMatrix::binary(b);
        let p = kernel.prepare(m.as_ref(K::FORMAT).unwrap());
        let unpacked = kernel.unpack(&p);
        assert_eq!(unpacked.len(), n * k);
        assert!(unpacked == m.ternary.values(), "{}: unpack differs from the derived codes for {n}x{k}", K::NAME);
        assert!(unpacked == m.binary.as_ref().unwrap().signs());
        let wb = kernel.weight_bytes(&p);
        assert!(wb.codes > 0);
    }
}

/// M1 trace contracts and the dispatch audit on the full schedule: real shapes (random weights),
/// synthetic captures at two lengths, C = 2 and 3, threads 1, 4, 8. Asserts the call count (240),
/// shapes, quantization reuse (192 quantizations, K and V reuse Q's), capture layouts (each call
/// consumes the capture the schedule names, [T][K]), output ordering ([T][N] per call) and that
/// every call's dots and FP32 outputs equal the oracle's.
pub fn trace_and_dispatch<K: Kernel>() {
    use crate::data::Matrix;
    use crate::replay::{Mode, Replayer, SyntheticInputs, oracle_digests};
    let sched = crate::schedule::Schedule::load();
    let calls: Vec<_> = sched.replayed().cloned().collect();
    let matrices: Vec<Matrix> = calls
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let b = BinaryMatrix::random(c.n, c.k, 31_000 + i as u64);
            match K::FORMAT {
                WeightFormat::Binary => Matrix::Binary(b),
                WeightFormat::Ternary => Matrix::Ternary(b.to_ternary()),
            }
        })
        .collect();
    let clone = |m: &Matrix| match m {
        Matrix::Binary(b) => Matrix::Binary(b.clone()),
        Matrix::Ternary(t) => Matrix::Ternary(t.clone()),
    };
    let utterances = [SyntheticInputs::new(&sched, 5, 1), SyntheticInputs::new(&sched, 2, 2)];
    for c_n in [2usize, 3] {
        let refs: Vec<_> = utterances.iter().map(|u| oracle_digests(&matrices, &sched, u, c_n, 8)).collect();
        for &threads in &THREADS {
            let pool = test_pool(threads);
            let cfg = config(threads, c_n, 5, 4096, 4096);
            let mut rp = Replayer::<K>::new(&cfg, matrices.iter().map(clone).collect(), &sched);
            for (u, inputs) in utterances.iter().enumerate() {
                let (g0, q0) = (rp.gemm_calls, rp.quantizations);
                let mut ev = Vec::new();
                rp.replay(&pool, inputs, Mode::Trace(&mut ev)).unwrap();
                assert_eq!(ev.len(), 240, "{}: replayed calls", K::NAME);
                assert_eq!((rp.gemm_calls - g0, rp.quantizations - q0), (240, 192), "{}: dispatch audit", K::NAME);
                let quantized: Vec<String> = ev.iter().filter_map(|e| e.quantized.clone()).collect();
                let captures: Vec<String> = sched.captures.iter().map(|c| c.name.clone()).collect();
                assert_eq!(quantized, captures, "{}: quantization order", K::NAME);
                for (e, c) in ev.iter().zip(&calls) {
                    assert_eq!((e.index, e.layer, &e.module, e.n, e.k, e.slot, e.reuse), (c.index, c.layer, &c.module, c.n, c.k, c.slot, c.reuse));
                    assert_eq!((e.rows, e.components, e.out_len), (inputs.t, c_n, inputs.t * c.n));
                    assert_eq!(&e.input, c.quant.as_ref().unwrap());
                    assert_eq!(e.quantized.is_none(), c.reuse);
                    assert_eq!(inputs.data[&e.input].len(), inputs.t * c.k, "capture layout [T][K]");
                }
                let mut dg = Vec::new();
                rp.replay(&pool, inputs, Mode::Digest(&mut dg)).unwrap();
                for (got, want) in dg.iter().zip(&refs[u]) {
                    assert_eq!(got.d_sha256, want.d_sha256, "{}: dots of call {} {} (C={c_n}, threads={threads}, T={})", K::NAME, got.index, got.module, inputs.t);
                    assert_eq!(got.y_sha256, want.y_sha256, "{}: f32 of call {} {} (C={c_n}, threads={threads}, T={})", K::NAME, got.index, got.module, inputs.t);
                }
                assert_eq!(dg.len(), 240);
            }
        }
    }
}

/// Generates the `#[test]` functions of the contract for a kernel type.
#[macro_export]
macro_rules! contract_tests {
    ($k:ty) => {
        #[test]
        fn contract_ragged_and_extremal() {
            $crate::contract::ragged_and_extremal::<$k>();
        }
        #[test]
        fn contract_model_shapes() {
            $crate::contract::model_shapes::<$k>();
        }
        #[test]
        fn contract_reuse_with_changed_inputs() {
            $crate::contract::reuse_with_changed_inputs::<$k>();
        }
        #[test]
        fn contract_native_equals_derived() {
            $crate::contract::native_equals_derived::<$k>();
        }
        #[test]
        fn contract_trace_and_dispatch() {
            $crate::contract::trace_and_dispatch::<$k>();
        }
        #[test]
        fn contract_golden_vectors() {
            $crate::golden::check_kernel::<$k>(&$crate::golden::Golden::load());
        }
    };
}
