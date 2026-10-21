//! x86 LocalFlow golden vectors (plans/cpu-benchmark-mac.md M8), generated on NixOS by
//! cpu1bit/shared/golden from the measured snapshot and pushed to the Mac with push.py. The NEON
//! quantizer, the oracle and every arm must reproduce the component bytes, scale bits and final
//! output bits exactly. The files are pinned by SHA-256; a missing or different file FAILS the
//! tests (never skipped).

use std::path::PathBuf;

use crate::kernel::{Call, Kernel, KernelConfig, Out, Slot};
use crate::oracle;
use crate::pool::ThreadPool;
use crate::quant::{QuantizedActs, quantize_row, quantize_row_scalar};
use crate::weights::{TernaryMatrix, WeightFormat, WeightsRef};

/// Frozen generation (golden.json `bin_sha256`); set when the vectors were generated.
pub const GOLDEN_STAMP: &str = "20261021T070704Z";
pub const GOLDEN_BIN_SHA256: &str = "f1e80a8839eef18306aa64e5f2c42224f288e76d08436548232e1ac7f3e0efbd";

pub struct Golden {
    pub doc: serde_json::Value,
    pub bin: Vec<u8>,
    pub dir: PathBuf,
}

/// `CPU1BIT_GOLDEN` (a directory with golden.json + golden.bin), else the pinned stamp under the
/// Mac data root (`$CPU1BIT_ROOT`, default `$HOME/wilderness-labs-stt-artifacts/cpu1bit`), else under the
/// NixOS artifacts.
pub fn default_dir() -> PathBuf {
    if let Ok(d) = std::env::var("CPU1BIT_GOLDEN") {
        return PathBuf::from(d);
    }
    let root = std::env::var("CPU1BIT_ROOT").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("wilderness-labs-stt-artifacts/cpu1bit")
    });
    let mac = root.join(format!("data/golden/{GOLDEN_STAMP}"));
    if mac.exists() {
        return mac;
    }
    PathBuf::from(format!("/mnt/hd/wilderness-labs-stt/parakeet-binary/cpu1bit/golden/{GOLDEN_STAMP}"))
}

impl Golden {
    pub fn load() -> Golden {
        let dir = default_dir();
        let bin = std::fs::read(dir.join("golden.bin"))
            .unwrap_or_else(|e| panic!("golden vectors missing at {}: {e} (push them with shared/push.py)", dir.display()));
        let doc: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("golden.json")).expect("golden.json")).expect("parse");
        let digest = {
            use sha2::Digest;
            let h = sha2::Sha256::digest(&bin);
            h.iter().map(|b| format!("{b:02x}")).collect::<String>()
        };
        assert_eq!(doc["bin_sha256"].as_str(), Some(digest.as_str()), "golden.bin does not match golden.json");
        assert_eq!(digest, GOLDEN_BIN_SHA256, "golden vectors are not the pinned generation");
        Golden { doc, bin, dir }
    }

    fn slice(&self, a: &serde_json::Value) -> &[u8] {
        let off = a["offset"].as_u64().unwrap() as usize;
        let len = a["len"].as_u64().unwrap() as usize;
        let size = match a["dtype"].as_str().unwrap() {
            "f32" => 4,
            "u8" => 1,
            d => panic!("dtype {d}"),
        };
        &self.bin[off..off + len * size]
    }

    pub fn f32s(&self, a: &serde_json::Value) -> Vec<f32> {
        assert_eq!(a["dtype"], "f32");
        self.slice(a).chunks_exact(4).map(|c| f32::from_bits(u32::from_le_bytes(c.try_into().unwrap()))).collect()
    }

    pub fn u8s(&self, a: &serde_json::Value) -> Vec<u8> {
        assert_eq!(a["dtype"], "u8");
        self.slice(a).to_vec()
    }

    pub fn quant_cases(&self) -> &[serde_json::Value] {
        self.doc["quant_cases"].as_array().unwrap()
    }

    pub fn gemm_cases(&self) -> &[serde_json::Value] {
        self.doc["gemm_cases"].as_array().unwrap()
    }

    /// A GEMM case's inputs: weights, LocalFlow's quantized activations (u -> q), expected y.
    pub fn gemm_case(&self, c: &serde_json::Value) -> (TernaryMatrix, QuantizedActs, Vec<f32>, Vec<f32>) {
        let (n, k) = (c["n_out"].as_u64().unwrap() as usize, c["k"].as_u64().unwrap() as usize);
        let (t, comps) = (c["rows"].as_u64().unwrap() as usize, c["components"].as_u64().unwrap() as usize);
        let m = TernaryMatrix { n, k, codes: self.u8s(&c["codes"]), scale: self.f32s(&c["weight_scale"]) };
        m.validate().unwrap();
        let q: Vec<i8> = self.u8s(&c["u"]).iter().map(|&u| (u ^ 0x80) as i8).collect();
        let qa = QuantizedActs::from_parts(t, k, comps, &q, &self.f32s(&c["frame_scale"]));
        (m, qa, self.f32s(&c["y"]), self.f32s(&c["x"]))
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The shared quantizer (production NEON row, scalar row, and the pooled `quantize` at 1, 4 and
/// 8 threads) reproduces LocalFlow's component bytes, scale bits and non-finite rejections.
pub fn check_quantizer(g: &Golden) {
    let pools: Vec<ThreadPool> = crate::contract::THREADS.iter().map(|&t| crate::contract::test_pool(t)).collect();
    for c in g.quant_cases() {
        let name = c["name"].as_str().unwrap();
        let (k, n, rows) = (c["k"].as_u64().unwrap() as usize, c["n"].as_u64().unwrap() as usize, c["rows"].as_u64().unwrap() as usize);
        let x = g.f32s(&c["x"]);
        if c["expect"] == "nonfinite" {
            for r in 0..rows {
                let row = &x[r * k..(r + 1) * k];
                if row.iter().all(|v| v.is_finite()) {
                    continue;
                }
                let (mut out, mut s) = (vec![0i8; n * k], 0.0f32);
                assert!(!quantize_row(row, n, &mut out, &mut s), "{name} n={n}: production accepted non-finite");
                assert!(!quantize_row_scalar(row, n, &mut out, &mut s), "{name} n={n}: scalar accepted non-finite");
            }
            for pool in &pools {
                let mut qa = QuantizedActs::with_capacity(rows, k, n);
                assert!(qa.quantize(pool, &x, rows).is_err(), "{name}: pooled quantize accepted non-finite");
                assert_eq!(qa.rows(), 0);
            }
            continue;
        }
        let want_u = g.u8s(&c["u"]);
        let want_s = bits(&g.f32s(&c["scale"]));
        for r in 0..rows {
            let row = &x[r * k..(r + 1) * k];
            for (tag, f) in [("production", quantize_row as fn(&[f32], usize, &mut [i8], &mut f32) -> bool),
                             ("scalar", quantize_row_scalar)] {
                let (mut out, mut s) = (vec![0i8; n * k], 0.0f32);
                assert!(f(row, n, &mut out, &mut s));
                let u: Vec<u8> = out.iter().map(|&q| (q as u8) ^ 0x80).collect();
                assert_eq!(s.to_bits(), want_s[r], "{name} n={n} row={r} {tag}: scale bits");
                if let Some(i) = (0..n * k).find(|&i| u[i] != want_u[r * n * k + i]) {
                    panic!("{name} n={n} row={r} {tag}: component byte {i} (c={}, i={}) got {} want {}", i / k, i % k, u[i],
                           want_u[r * n * k + i]);
                }
            }
        }
        for pool in &pools {
            let mut qa = QuantizedActs::with_capacity(rows, k, n);
            qa.quantize(pool, &x, rows).unwrap();
            assert_eq!(qa.u_bytes(), want_u, "{name} n={n} threads={}", pool.threads());
            assert_eq!(bits(qa.scale()), want_s);
        }
    }
}

/// The oracle reproduces LocalFlow's FP32 outputs bit for bit, from LocalFlow's quantized inputs
/// and from this crate's own quantization of the same frames.
pub fn check_oracle(g: &Golden) {
    let pool = crate::contract::test_pool(1);
    for c in g.gemm_cases() {
        let name = c["name"].as_str().unwrap();
        let (m, qa, want, x) = g.gemm_case(c);
        let y = oracle::gemm(&m.values(), m.n, m.k, &m.scale, &qa);
        assert_eq!(bits(&y), bits(&want), "{name}: oracle vs LocalFlow");
        let mut ours = QuantizedActs::with_capacity(qa.rows(), m.k, qa.components());
        ours.quantize(&pool, &x, qa.rows()).unwrap();
        assert_eq!(ours.q(), qa.q(), "{name}: quantization");
        assert_eq!(bits(ours.scale()), bits(qa.scale()));
    }
}

/// Every golden GEMM case the kernel's format can hold (binary kernels: the no-zero cases) at
/// 1, 4 and 8 threads, bit for bit against LocalFlow's x86 output.
pub fn check_kernel<K: Kernel>(g: &Golden) {
    let mut checked = 0;
    for &threads in &crate::contract::THREADS {
        let pool = crate::contract::test_pool(threads);
        for c in g.gemm_cases() {
            let name = c["name"].as_str().unwrap();
            let (m, qa, want, _) = g.gemm_case(c);
            let binary = m.to_binary();
            let w = match (K::FORMAT, &binary) {
                (WeightFormat::Ternary, _) => WeightsRef::Ternary(&m),
                (WeightFormat::Binary, Some(b)) => WeightsRef::Binary(b),
                (WeightFormat::Binary, None) => continue,
            };
            let cfg = KernelConfig { threads, components: qa.components(), max_rows: qa.rows(), max_k: m.k, max_n: m.n,
                                     tuning: serde_json::Value::Null };
            let mut kernel = K::new(&cfg);
            let p = kernel.prepare(w);
            let mut y = vec![0f32; qa.rows() * m.n];
            kernel.gemm(&pool, &p, &qa, Call { slot: Slot::Model, reuse: false }, Out::F32(&mut y));
            assert_eq!(bits(&y), bits(&want), "{}: golden {name} threads={threads}", K::NAME);
            checked += 1;
        }
    }
    assert!(checked >= 9, "{}: only {checked} golden cases applied", K::NAME);
}
