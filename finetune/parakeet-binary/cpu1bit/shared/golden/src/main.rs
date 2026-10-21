//! Golden vectors from the measured x86 LocalFlow build (plans/cpu-benchmark-mac.md M8).
//!
//! Every expected value is produced by LocalFlow's own code from the read-only snapshot
//! (`lf_cpu::QuantizedActs::quantize`: AVX-512 when K % 16 == 0, its scalar path otherwise;
//! `lf_cpu::gemm_i8::gemm`: VBMI expansion + VPDPBUSD + the FP32 epilogue). This program only
//! chooses inputs; the local helpers below are used solely to SEARCH for interesting inputs
//! (rounding ties, FMA-sensitive residual updates), never to produce expectations.
//!
//! Output (OUT_DIR): golden.bin (little-endian arrays) and golden.json (cases with byte offsets,
//! provenance). Usage: cpu1bit-golden-x86 OUT_DIR

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;

use lf_cpu::{PackedTernary, QuantizedActs, Scratch, ThreadPool, gemm_i8};
use lf_model::TernaryMatrix;
use lf_model::rng::SplitMix64;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const WEIGHTS: [f32; 3] = [1.0, 254.0, 254.0 * 254.0];

struct Blob {
    data: Vec<u8>,
}

impl Blob {
    fn align(&mut self) {
        while self.data.len() % 16 != 0 {
            self.data.push(0);
        }
    }
    fn f32s(&mut self, v: &[f32]) -> Value {
        self.align();
        let off = self.data.len();
        for x in v {
            self.data.extend_from_slice(&x.to_bits().to_le_bytes());
        }
        json!({"offset": off, "len": v.len(), "dtype": "f32"})
    }
    fn u8s(&mut self, v: &[u8]) -> Value {
        self.align();
        let off = self.data.len();
        self.data.extend_from_slice(v);
        json!({"offset": off, "len": v.len(), "dtype": "u8"})
    }
}

/// Search helper (NOT an expectation): the scalar residual chain, reporting per element whether
/// a component's product is an exact tie and whether the fused update differs from unfused.
fn probe(x: &[f32], n: usize) -> (Vec<[bool; 3]>, Vec<[bool; 3]>) {
    let m = x.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    let (inv, s) = (127.0 / m, m / 127.0);
    let mut ties = vec![[false; 3]; x.len()];
    let mut fma = vec![[false; 3]; x.len()];
    for (i, &v) in x.iter().enumerate() {
        let mut r = v;
        for c in 0..n {
            let p = r * (inv * WEIGHTS[c]);
            ties[i][c] = (p - p.trunc()).abs() == 0.5 && p.abs() < 127.0;
            let q = p.round_ties_even().clamp(-127.0, 127.0);
            let fused = (-q).mul_add(s / WEIGHTS[c], r);
            let plain = r - q * (s / WEIGHTS[c]);
            fma[i][c] = fused.to_bits() != plain.to_bits();
            r = fused;
        }
    }
    (ties, fma)
}

fn gaussian(rng: &mut SplitMix64, k: usize, outliers: bool) -> Vec<f32> {
    (0..k).map(|i| rng.next_gaussian() * if outliers && i % 37 == 0 { 20.0 } else { 1.0 }).collect()
}

/// Frames whose elements hit exact ties at component `c` (searched by ulp nudges), max fixed.
fn tie_frame(rng: &mut SplitMix64, k: usize, c: usize, n: usize) -> Vec<f32> {
    let m = 3.0f32;
    let s = m / 127.0;
    let mut out = vec![0.0f32; k];
    out[0] = m;
    let mut filled = 1;
    let mut tries = 0;
    while filled < k && tries < 2_000_000 {
        tries += 1;
        let q0 = (rng.below(253) as f64) - 126.0;
        let frac = match c {
            0 => 0.5,
            1 => ((rng.below(253) as f64) - 126.0 + 0.5) / 254.0,
            _ => ((rng.below(253) as f64) - 126.0) / 254.0 + ((rng.below(253) as f64) - 126.0 + 0.5) / 64516.0,
        };
        let base = (s as f64 * (q0 + frac)) as f32;
        for nudge in -4i32..=4 {
            let cand = f32::from_bits((base.to_bits() as i32 + nudge) as u32);
            if !cand.is_finite() || cand.abs() >= m {
                continue;
            }
            let mut probe_frame = vec![m, cand];
            probe_frame.truncate(2);
            let (ties, _) = probe(&probe_frame, n);
            if ties[1][c] {
                out[filled] = cand;
                filled += 1;
                break;
            }
        }
    }
    out
}

/// Gaussian frames whose elements were chosen so the fused residual update differs from the
/// unfused one at some component.
fn fma_frame(rng: &mut SplitMix64, k: usize, n: usize) -> Vec<f32> {
    let m = 4.0f32;
    let mut out = vec![m];
    while out.len() < k {
        let cand = (rng.next_gaussian() * 1.2).clamp(-3.99, 3.99);
        let (_, fma) = probe(&[m, cand], n);
        if fma[1][..n].iter().any(|&b| b) {
            out.push(cand);
        }
    }
    out
}

/// Smallest positive f32 m such that (127/m) * WEIGHTS[n-1] is finite (LocalFlow's too_small edge).
fn threshold(n: usize) -> f32 {
    let mut m = 1e-30f32;
    while ((127.0 / m) * WEIGHTS[n - 1]).is_finite() {
        m = f32::from_bits(m.to_bits() - 1 - (m.to_bits() >> 12).min(1 << 16));
    }
    // m is now too small; walk up to the first finite one.
    while !((127.0 / m) * WEIGHTS[n - 1]).is_finite() {
        m = f32::from_bits(m.to_bits() + 1);
    }
    m
}

fn quant_frames(rng: &mut SplitMix64) -> Vec<(String, usize, Vec<f32>)> {
    // (name, k, rows * k values)
    let mut cases: Vec<(String, usize, Vec<f32>)> = Vec::new();
    for &k in &[1usize, 3, 15, 16, 17, 64, 1024, 4096] {
        let mut x = Vec::new();
        for _ in 0..3 {
            x.extend(gaussian(rng, k, true));
        }
        cases.push((format!("gaussian-k{k}"), k, x));
    }
    for &k in &[509usize, 512] {
        // Exact first-component ties: max 15.875 gives s = 0.125, inv = 8 exactly; x = j/16.
        let mut x: Vec<f32> = (0..k).map(|j| ((j as i32 % 509) - 254) as f32 * 0.0625).collect();
        x[0] = 15.875;
        cases.push((format!("ties-c0-grid-k{k}"), k, x));
        for c in 0..3 {
            cases.push((format!("ties-c{c}-search-k{k}"), k, tie_frame(rng, k, c, 3)));
        }
        cases.push((format!("fma-sensitive-k{k}"), k, fma_frame(rng, k, 3)));
    }
    for &k in &[16usize, 17] {
        cases.push((format!("zero-k{k}"), k, vec![0.0; k]));
        cases.push((format!("negzero-k{k}"), k, vec![-0.0; k]));
        cases.push((format!("tiny1e-35-k{k}"), k, (0..k).map(|i| if i % 2 == 0 { 1e-35 } else { -3e-36 }).collect()));
        cases.push((format!("subnormal-k{k}"), k, (0..k).map(|i| f32::from_bits(1 + i as u32 * 977)).collect()));
        let mut mixed = gaussian(rng, k, false);
        for (i, v) in mixed.iter_mut().enumerate().skip(1) {
            if i % 2 == 0 {
                *v = f32::from_bits(1 + i as u32 * 31) * if i % 4 == 0 { -1.0 } else { 1.0 };
            }
        }
        cases.push((format!("normal-with-subnormals-k{k}"), k, mixed));
        for n in 1..=3 {
            let t = threshold(n);
            for (tag, m) in [("at", t), ("below", f32::from_bits(t.to_bits() - 1)), ("above", f32::from_bits(t.to_bits() + 1))] {
                let x: Vec<f32> = (0..k).map(|i| if i == k / 2 { -m } else { m * ((i % 7) as f32 / 7.0) }).collect();
                cases.push((format!("threshold-n{n}-{tag}-k{k}"), k, x));
            }
        }
        cases.push((format!("huge-k{k}"), k, (0..k).map(|i| if i == 3 { f32::MAX } else { (i as f32 - 8.0) * 1e37 }).collect()));
        cases.push((format!("single-nonzero-k{k}"), k, (0..k).map(|i| if i == k - 1 { -2.5 } else { 0.0 }).collect()));
        for (tag, bad) in [("nan", f32::NAN), ("posinf", f32::INFINITY), ("neginf", f32::NEG_INFINITY)] {
            for pos in [0, k / 2, k - 1] {
                let mut x = gaussian(rng, k, false);
                x[pos] = bad;
                cases.push((format!("nonfinite-{tag}-at{pos}-k{k}"), k, x));
            }
        }
    }
    cases
}

fn first_cpu() -> usize {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set);
        (0..libc::CPU_SETSIZE as usize).find(|&c| libc::CPU_ISSET(c, &set)).unwrap_or(0)
    }
}

fn allowed_cpus(count: usize) -> Vec<usize> {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set);
        (0..libc::CPU_SETSIZE as usize).filter(|&c| libc::CPU_ISSET(c, &set)).take(count).collect()
    }
}

fn sha256(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn main() {
    let out_dir = PathBuf::from(std::env::args().nth(1).expect("usage: cpu1bit-golden-x86 OUT_DIR"));
    assert!(lf_cpu::cpu_supported(), "needs AVX-512 F/BW/VNNI/VBMI (the measured x86 path)");
    std::fs::create_dir_all(&out_dir).unwrap();
    let pool = ThreadPool::new(&[first_cpu()]);
    let pool4 = ThreadPool::new(&allowed_cpus(4));
    let mut rng = SplitMix64::new(20261021);
    let mut blob = Blob { data: Vec::new() };

    // --- quantizer cases -----------------------------------------------------------------------
    let mut quant_cases = Vec::new();
    let mut tie_count = [0usize; 3];
    let mut fma_count = 0usize;
    for (name, k, x) in quant_frames(&mut rng) {
        let rows = x.len() / k;
        for n in 1..=3 {
            let mut qa = QuantizedActs::with_capacity(rows, k, n);
            let res = qa.quantize(&pool, &x, rows);
            let mut case = json!({"name": name, "k": k, "n": n, "rows": rows, "x": blob.f32s(&x),
                                  "avx512_path": k % 16 == 0});
            match res {
                Ok(()) => {
                    let (ties, fma) = (0..rows).fold((vec![], vec![]), |(mut t, mut f), r| {
                        let (a, b) = probe(&x[r * k..(r + 1) * k], n);
                        t.extend(a);
                        f.extend(b);
                        (t, f)
                    });
                    let nt: Vec<usize> = (0..3).map(|c| ties.iter().filter(|e| e[c]).count()).collect();
                    let nf = fma.iter().filter(|e| e.iter().any(|&b| b)).count();
                    for c in 0..3 {
                        tie_count[c] += nt[c];
                    }
                    fma_count += nf;
                    let scale_bits: Vec<f32> = qa.scale().to_vec();
                    case["expect"] = json!("ok");
                    case["u"] = blob.u8s(qa.q());
                    case["scale"] = blob.f32s(&scale_bits);
                    case["search_ties_per_component"] = json!(nt);
                    case["search_fma_sensitive_elements"] = json!(nf);
                }
                Err(_) => {
                    assert!(name.starts_with("nonfinite"), "{name}: unexpected non-finite rejection");
                    case["expect"] = json!("nonfinite");
                    case["rows_after"] = json!(qa.rows());
                }
            }
            quant_cases.push(case);
        }
    }

    // --- GEMM + epilogue cases (LocalFlow requires N % 64 == 0, K % 4 == 0) ---------------------
    let mut gemm_cases = Vec::new();
    let shapes: &[(&str, usize, usize, usize, bool)] = &[
        ("ternary", 64, 64, 7, false),
        ("ternary", 128, 1024, 13, false),
        ("ternary", 64, 4096, 5, false),
        ("binary", 192, 1024, 9, true),
        ("binary", 64, 4096, 6, true),
        ("binary", 1024, 1024, 3, true),
    ];
    for (si, &(kind, n_out, k, t, binary)) in shapes.iter().enumerate() {
        let mut m = TernaryMatrix::random(n_out, k, 700 + si as u64);
        if binary {
            // No zero codes: 00 -> 01 (+1) keeps the random sign mix of the other fields.
            for b in m.codes.iter_mut() {
                for j in 0..4 {
                    if (*b >> (2 * j)) & 3 == 0 {
                        *b |= if rng.next_u64() & 1 == 1 { 1 } else { 2 } << (2 * j);
                    }
                }
            }
        }
        // Scales: random, plus extreme rows (MB1's floor 1e-8, large, denormal-producing).
        for (o, s) in m.scale.iter_mut().enumerate() {
            match o % 16 {
                3 => *s = 1e-8,
                7 => *s = 3.7e4,
                11 => *s = 1.17549435e-38,
                _ => {}
            }
        }
        m.validate().unwrap();
        let w = PackedTernary::new(&m).unwrap();
        // Frames: gaussian, fma-sensitive, ties, zero, tiny, all-equal (extreme dots).
        let mut x = Vec::new();
        for f in 0..t {
            let frame = match f % 6 {
                0 => gaussian(&mut rng, k, true),
                1 => fma_frame(&mut rng, k, 3),
                2 => tie_frame(&mut rng, k, 1, 3),
                3 => vec![0.0; k],
                4 => (0..k).map(|i| if i % 3 == 0 { 1e-35 } else { 2e-36 }).collect(),
                _ => vec![1.0; k],
            };
            x.extend(frame);
        }
        for n in 1..=3 {
            let mut qa = QuantizedActs::with_capacity(t, k, n);
            qa.quantize(&pool, &x, t).unwrap();
            let mut y = vec![0.0f32; t * n_out];
            let scratch = Scratch::new(1, gemm_i8::scratch_bytes(k));
            gemm_i8::gemm(&pool, &scratch, &w, &qa, &mut y);
            let mut y4 = vec![0.0f32; t * n_out];
            let scratch4 = Scratch::new(pool4.threads(), gemm_i8::scratch_bytes(k));
            gemm_i8::gemm(&pool4, &scratch4, &w, &qa, &mut y4);
            assert!(y.iter().zip(&y4).all(|(a, b)| a.to_bits() == b.to_bits()), "LocalFlow 1 vs 4 threads differ");
            gemm_cases.push(json!({
                "name": format!("{kind}-{n_out}x{k}-t{t}-n{n}"), "binary": binary, "n_out": n_out, "k": k,
                "rows": t, "components": n, "x": blob.f32s(&x), "codes": blob.u8s(&m.codes),
                "weight_scale": blob.f32s(&m.scale), "u": blob.u8s(qa.q()), "frame_scale": blob.f32s(qa.scale()),
                "y": blob.f32s(&y),
            }));
        }
    }

    let bin_sha = sha256(&blob.data);
    std::fs::write(out_dir.join("golden.bin"), &blob.data).unwrap();
    let snapshot = "/mnt/hd/wilderness-labs-stt/parakeet-ternary/cpu-bitplanes/20261008T121546/snapshot";
    let mut src = BTreeMap::new();
    for f in ["crates/lf-cpu/src/quant.rs", "crates/lf-cpu/src/gemm_i8.rs", "crates/lf-cpu/src/pack.rs",
              "crates/lf-model/src/ternary.rs", "crates/lf-cpu/src/lib.rs"] {
        src.insert(f, sha256(&std::fs::read(format!("{snapshot}/{f}")).unwrap()));
    }
    let exe = std::fs::read(std::env::current_exe().unwrap()).unwrap();
    let doc = json!({
        "about": "LocalFlow x86 golden vectors for the cpu1bit quantizer, oracle and every arm (plans/cpu-benchmark-mac.md M8)",
        "generator": "cpu1bit/shared/golden (expectations from lf_cpu::QuantizedActs::quantize and lf_cpu::gemm_i8::gemm)",
        "snapshot": snapshot, "snapshot_source_sha256": src, "generator_binary_sha256": sha256(&exe),
        "rustc": env!("GOLDEN_RUSTC"), "cpu_supported_avx512": lf_cpu::cpu_supported(),
        "bin_sha256": bin_sha, "bin_bytes": blob.data.len(),
        "coverage": {"tie_elements_per_component": tie_count, "fma_sensitive_elements": fma_count,
                     "quant_cases": quant_cases.len(), "gemm_cases": gemm_cases.len()},
        "layout": "u: LocalFlow bytes q+128, [rows][components][k]; codes: parakeet-ternary-v1 [n_out][k/4]; f32 arrays as raw bits",
        "quant_cases": quant_cases, "gemm_cases": gemm_cases,
    });
    let mut f = std::fs::File::create(out_dir.join("golden.json")).unwrap();
    f.write_all(serde_json::to_string_pretty(&doc).unwrap().as_bytes()).unwrap();
    println!("{}", json!({"out": out_dir, "bin_sha256": bin_sha, "coverage": doc["coverage"]}));
}
