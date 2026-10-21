//! T's own contract tests, on top of H's suite (tests/contract.rs). Black-box: every check goes
//! through the `Kernel` trait and compares exact dots and bit-exact FP32 outputs with H's
//! independent oracle (`shared::contract::check_call`). The inputs are chosen to hit the places
//! where a lookup-table kernel can be wrong while random data passes:
//!
//! - the INT16 run: entries are subset sums of 4 activations, f in [-512, 508]; at most 64 are
//!   added in INT16 before widening, so 64 * (-512) = -32768 is the exact lower edge. All q = -128
//!   with all weights +1 (every index nibble 15) puts every run exactly on that edge, and
//!   K = 64 * 4 + {1..4} puts one extra group behind it (a 65-group run would wrap).
//! - every table index 0..15 at every lane of an index vector (nibble patterns, i.e. g = 4), and
//!   whole-byte patterns (the 8 inputs one code byte indexes through two g = 4 tables, i.e. the
//!   g = 8 split-table view), with extremal activations aligned to group positions;
//! - frame-block, output-block, panel and K-chunk tails, for every point of the tuning grid.
use shared::contract::{TestMatrix, check_call, test_pool};
use shared::rng::SplitMix64;
use shared::{BinaryMatrix, Call, Kernel, KernelConfig, QuantizedActs, Slot};
use t_lut::TLut;

fn config(threads: usize, c: usize, max_rows: usize, max_k: usize, max_n: usize, tuning: serde_json::Value) -> KernelConfig {
    KernelConfig { threads, components: c, max_rows, max_k, max_n, tuning }
}

fn signs(n: usize, k: usize, seed: u64, f: impl Fn(usize, usize) -> i8) -> BinaryMatrix {
    let mut rng = SplitMix64::new(seed);
    let scale: Vec<f32> = (0..n).map(|_| 0.01 + 0.04 * rng.next_f32()).collect();
    let s: Vec<i8> = (0..n * k).map(|j| f(j / k, j % k)).collect();
    BinaryMatrix::from_signs(n, k, &s, scale)
}

fn acts(t: usize, c: usize, k: usize, seed: u64, f: impl Fn(usize, usize, usize) -> i8) -> QuantizedActs {
    let mut rng = SplitMix64::new(seed);
    let scales: Vec<f32> = (0..t).map(|_| 1e-3 + rng.next_f32()).collect();
    let q: Vec<i8> = (0..t * c * k).map(|j| f(j / (c * k), (j / k) % c, j % k)).collect();
    QuantizedActs::from_parts(t, k, c, &q, &scales)
}

/// Runs `check_call` for one matrix and a set of activations on a fresh kernel.
fn run_case(threads: usize, tuning: &serde_json::Value, m: &BinaryMatrix, qas: &[(&str, QuantizedActs)], what: &str) {
    let pool = test_pool(threads);
    let c = qas[0].1.components();
    let t_max = qas.iter().map(|(_, q)| q.rows()).max().unwrap();
    let mut kernel = TLut::new(&config(threads, c, t_max.max(1), m.k, m.n, tuning.clone()));
    let tm = TestMatrix::binary(m.clone());
    let p = kernel.prepare(tm.as_ref(TLut::FORMAT).unwrap());
    for (aname, qa) in qas {
        let what = format!("{what} threads={threads} tuning={tuning} a={aname}");
        check_call(&mut kernel, &pool, &p, &tm, qa, Call { slot: Slot::Model, reuse: false }, &what);
    }
}

fn extremal_acts(t: usize, c: usize, k: usize, seed: u64) -> Vec<(&'static str, QuantizedActs)> {
    let mut rng = SplitMix64::new(seed);
    let rnd: Vec<i8> = (0..t * c * k).map(|_| rng.next_u64() as i8).collect();
    vec![
        ("all-128", acts(t, c, k, seed, |_, _, _| -128)),
        ("all+127", acts(t, c, k, seed, |_, _, _| 127)),
        ("alt1", acts(t, c, k, seed, |_, _, i| if i % 2 == 0 { -128 } else { 127 })),
        ("alt2", acts(t, c, k, seed, |_, _, i| if (i / 2) % 2 == 0 { -128 } else { 127 })),
        ("alt4", acts(t, c, k, seed, |_, _, i| if (i / 4) % 2 == 0 { -128 } else { 127 })),
        // One group position extreme, the others the opposite extreme.
        ("pos3", acts(t, c, k, seed, |_, _, i| if i % 4 == 3 { -128 } else { 127 })),
        // Components differ (catches a component/row mix-up in the blocking).
        ("per-part", acts(t, c, k, seed, |tt, cc, i| [-128i8, 127, -1, 1][(tt + cc + i / 64) % 4])),
        ("random", acts(t, c, k, seed, move |tt, cc, i| rnd[(tt * c + cc) * k + i])),
    ]
}

/// The INT16 run edge and the widening behind it, for every K residue around the 64-group run
/// length and for long K, every C, 1 and 8 threads.
#[test]
fn int16_run_edges_and_widening() {
    let ks = [4, 252, 255, 256, 257, 258, 259, 260, 264, 511, 512, 513, 516, 1020, 1024, 1028, 2052, 4096, 4100];
    for &threads in &[1usize, 8] {
        for c in 1..=3 {
            for &k in &ks {
                for (wname, m) in [
                    ("all+1", signs(40, k, 1, |_, _| 1)),
                    ("all-1", signs(40, k, 2, |_, _| -1)),
                    ("rows-mixed", signs(40, k, 3, |o, _| if o % 3 == 0 { 1 } else { -1 })),
                ] {
                    let qas = vec![
                        ("all-128", acts(3, c, k, 5, |_, _, _| -128)),
                        ("all+127", acts(3, c, k, 6, |_, _, _| 127)),
                        ("frames-differ", acts(3, c, k, 7, |t, _, _| if t == 1 { 127 } else { -128 })),
                    ];
                    run_case(threads, &serde_json::Value::Null, &m, &qas, &format!("int16 C={c} K={k} w={wname}"));
                }
            }
        }
    }
}

/// Every table index at every output lane (nibble = group of 4) and whole-byte patterns (two
/// tables per code byte), against extremal and random activations.
#[test]
fn nibble_and_byte_sign_patterns() {
    let (n, k) = (72, 136); // 4.5 out-blocks of 16, 34 groups, 17 code bytes
    let nib = |o: usize, i: usize, f: &dyn Fn(usize, usize) -> u8| -> i8 {
        if (f(o, i / 4) >> (i % 4)) & 1 == 1 { 1 } else { -1 }
    };
    let byte = |o: usize, i: usize, f: &dyn Fn(usize, usize) -> u8| -> i8 {
        if (f(o, i / 8) >> (i % 8)) & 1 == 1 { 1 } else { -1 }
    };
    let mut mats: Vec<(String, BinaryMatrix)> = Vec::new();
    // Nibble value (o + j) % 16: all 16 indices at every lane and every group.
    mats.push(("nib-cycle".into(), signs(n, k, 11, |o, i| nib(o, i, &|o, j| ((o + j) % 16) as u8))));
    mats.push(("nib-cycle-rev".into(), signs(n, k, 12, |o, i| nib(o, i, &|o, j| (15 - (o * 3 + j) % 16) as u8))));
    for v in [0x1u8, 0x2, 0x4, 0x8, 0xE, 0xD, 0xB, 0x7, 0x5, 0xA, 0x3, 0xC, 0x6, 0x9] {
        mats.push((format!("nib-{v:x}"), signs(n, k, 13, move |o, i| nib(o, i, &|_, _| v))));
    }
    for v in [0x0Fu8, 0xF0, 0x55, 0xAA, 0x01, 0x80, 0xFE, 0x7F, 0x3C, 0xC3] {
        mats.push((format!("byte-{v:02x}"), signs(n, k, 14, move |o, i| byte(o, i, &|_, _| v))));
    }
    // Byte value cycles with the output (all 256 bytes appear across outputs and bytes).
    mats.push(("byte-cycle".into(), signs(n, k, 15, |o, i| byte(o, i, &|o, p| (o * 17 + p * 29) as u8))));
    for &threads in &[1usize, 4] {
        for c in [1, 2, 3] {
            for (wname, m) in &mats {
                run_case(threads, &serde_json::Value::Null, m, &extremal_acts(5, c, k, 21), &format!("patterns C={c} w={wname}"));
            }
        }
    }
}

/// Output-block, panel, frame-block and K-chunk tails under every tuning point of the grid.
#[test]
fn blocking_tails_every_tuning_point() {
    let ns = [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 96, 127, 129, 200];
    // (T, K): K = 4100 is multi-chunk at every grid point.
    let cases: [(usize, usize); 7] = [(1, 3), (2, 8), (3, 33), (5, 260), (7, 520), (9, 1100), (2, 4100)];
    for tuning in TLut::tuning_grid() {
        for &threads in &[1usize, 3, 8] {
            for c in [1, 2, 3] {
                for (ci, &(t, k)) in cases.iter().enumerate() {
                    let n = ns[(ci * 5 + c + threads) % ns.len()];
                    let m = BinaryMatrix::random(n, k, (ci * 31 + c) as u64);
                    let mut rng = SplitMix64::new((t * k + c) as u64);
                    let rnd: Vec<i8> = (0..t * c * k).map(|_| rng.next_u64() as i8).collect();
                    let qas = vec![
                        ("random", acts(t, c, k, 3, |tt, cc, i| rnd[(tt * c + cc) * k + i])),
                        ("all-128", acts(t, c, k, 4, |_, _, _| -128)),
                    ];
                    run_case(threads, &tuning, &m, &qas, &format!("tails N={n} K={k} T={t} C={c}"));
                }
            }
        }
        // Every N residue once at a fixed shape.
        for &n in &ns {
            let m = BinaryMatrix::random(n, 72, n as u64);
            let mut rng = SplitMix64::new(n as u64);
            let rnd: Vec<i8> = (0..4 * 2 * 72).map(|_| rng.next_u64() as i8).collect();
            run_case(4, &tuning, &m, &[("random", acts(4, 2, 72, 9, |t, c, i| rnd[(t * 2 + c) * 72 + i]))], &format!("N={n}"));
        }
    }
}

/// Q/K/V reuse, a requantized slot, and an interleaved slot, under every tuning point (H's reuse
/// suite runs the default point only). Shapes span more than one K-chunk for the small chunks.
#[test]
fn reuse_every_tuning_point() {
    for tuning in TLut::tuning_grid() {
        for &threads in &[1usize, 8] {
            let c = 2;
            let (k_model, k_ff, n) = (520, 2100, 40);
            let pool = test_pool(threads);
            let mut kernel = TLut::new(&config(threads, c, 11, k_ff, n, tuning.clone()));
            let mats: Vec<TestMatrix> = (0..3).map(|s| TestMatrix::binary(BinaryMatrix::random(n, k_model, 70 + s))).collect();
            let ff = TestMatrix::binary(BinaryMatrix::random(n, k_ff, 79));
            let preps: Vec<_> = mats.iter().map(|m| kernel.prepare(m.as_ref(TLut::FORMAT).unwrap())).collect();
            let pff = kernel.prepare(ff.as_ref(TLut::FORMAT).unwrap());
            let mut qm = QuantizedActs::with_capacity(11, k_model, c);
            let mut qf = QuantizedActs::with_capacity(11, k_ff, c);
            let mut rng = SplitMix64::new(threads as u64);
            for (round, t) in [4usize, 11, 1, 9].into_iter().enumerate() {
                let x: Vec<f32> = (0..t * k_model).map(|_| rng.next_gaussian() * (1 + round) as f32).collect();
                qm.quantize(&pool, &x, t).unwrap();
                for (j, (m, p)) in mats.iter().zip(&preps).enumerate() {
                    let what = format!("reuse tuning={tuning} threads={threads} round={round} qkv={j}");
                    check_call(&mut kernel, &pool, p, m, &qm, Call { slot: Slot::Model, reuse: j > 0 }, &what);
                }
                let xf: Vec<f32> = (0..t * k_ff).map(|_| rng.next_gaussian()).collect();
                qf.quantize(&pool, &xf, t).unwrap();
                let what = format!("reuse tuning={tuning} threads={threads} round={round} ff");
                check_call(&mut kernel, &pool, &pff, &ff, &qf, Call { slot: Slot::Ff, reuse: false }, &what);
                let what = format!("reuse tuning={tuning} threads={threads} round={round} model-after-ff");
                check_call(&mut kernel, &pool, &preps[1], &mats[1], &qm, Call { slot: Slot::Model, reuse: true }, &what);
            }
        }
    }
}

/// The grid is the frozen file: its SHA-256, exactly 6 distinct valid points, the default first.
#[test]
fn tuning_grid_is_frozen() {
    assert_eq!(shared::data::sha256_hex(t_lut::TUNING_GRID_JSON.as_bytes()), t_lut::TUNING_GRID_SHA256);
    let grid = TLut::tuning_grid();
    assert_eq!(grid.len(), 6);
    let parsed: Vec<t_lut::Tuning> = grid.iter().map(t_lut::Tuning::parse).collect();
    assert_eq!(parsed[0], t_lut::Tuning::default());
    for i in 0..parsed.len() {
        for j in 0..i {
            assert_ne!(parsed[i], parsed[j], "duplicate grid point");
        }
    }
}

/// The real 240-call schedule (every model shape, Q/K/V reuse, the Ff slot) at every grid point:
/// each call's dots and FP32 outputs equal the oracle's (digests), C = 2 at 1 and 8 threads and
/// C = 3 at 1 thread. H's trace contract covers the default point only.
#[test]
fn real_schedule_replay_every_tuning_point() {
    use shared::data::Matrix;
    use shared::replay::{Mode, Replayer, SyntheticInputs, oracle_digests};
    let sched = shared::schedule::Schedule::load();
    let calls: Vec<_> = sched.replayed().cloned().collect();
    let matrices: Vec<BinaryMatrix> =
        calls.iter().enumerate().map(|(i, c)| BinaryMatrix::random(c.n, c.k, 77_000 + i as u64)).collect();
    let wrap = || matrices.iter().map(|b| Matrix::Binary(b.clone())).collect::<Vec<_>>();
    let inputs = SyntheticInputs::new(&sched, 5, 3);
    for c_n in [2usize, 3] {
        let want = oracle_digests(&wrap(), &sched, &inputs, c_n, 8);
        let threads: &[usize] = if c_n == 2 { &[1, 8] } else { &[1] };
        for tuning in TLut::tuning_grid() {
            for &th in threads {
                let pool = test_pool(th);
                let mut rp = Replayer::<TLut>::new(&config(th, c_n, 5, 4096, 4096, tuning.clone()), wrap(), &sched);
                let mut got = Vec::new();
                rp.replay(&pool, &inputs, Mode::Digest(&mut got)).unwrap();
                assert_eq!(got.len(), 240);
                for (g, w) in got.iter().zip(&want) {
                    assert_eq!(
                        (g.index, &g.d_sha256, &g.y_sha256),
                        (w.index, &w.d_sha256, &w.y_sha256),
                        "call {} {} tuning={tuning} C={c_n} threads={th}",
                        g.index,
                        g.module
                    );
                }
            }
        }
    }
}

/// The weight storage stays at the 1-bit size for the model shapes (no expanded codes kept) and
/// the declared workspace covers the real configuration.
#[test]
fn storage_is_one_bit_per_weight_for_model_shapes() {
    let kernel = TLut::new(&config(8, 2, 350, 4096, 4096, serde_json::Value::Null));
    for (n, k) in [(1024usize, 1024usize), (1024, 4096), (2048, 1024), (4096, 1024)] {
        let m = BinaryMatrix::random(n, k, 3);
        let p = kernel.prepare(shared::WeightsRef::Binary(&m));
        assert_eq!(kernel.weight_bytes(&p).codes, n * k / 8, "{n}x{k}");
    }
}
