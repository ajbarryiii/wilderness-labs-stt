//! The storage and workspace acceptance audit (plans/cpu-benchmark-mac.md "Storage and workspace";
//! H's `shared::bench::storage_audit`, unchanged) on V's arm itself, measured exactly as H's contract
//! (shared/core/tests/storage_audit.rs) and the `block` process measure it: the counting allocator's
//! live heap from before loading the 240 replayed matrices, through `prepare` and one warm-up replay.
//!
//! Checked for every frozen grid point (`v_sdot::grid()`) at 8 workers, i8x2 and i8x3, at a frame count
//! that takes the production frames-split path at 8 threads (T >= 8 panels):
//! - H's audit passes (`capped && ok`: weight bytes <= 1.10 x 72,351,744, workspace <= 32 MiB,
//!   per-row bytes <= 16 per output row);
//! - V's storage claims: the codes are exactly the packed 1-bit size (72,351,744 B), the per-row bytes
//!   are the FP32 scales only (4 per output row), and nothing else is held beyond 1 MiB;
//! - a further replay leaves the live heap unchanged (no allocation retained by `gemm` after warm-up).
//!
//! One test function: the counting allocator is process-wide, so no other test may allocate meanwhile.
#![cfg(target_arch = "aarch64")]

use shared::bench::{self, CountingAlloc, StorageAudit};
use shared::contract::test_pool;
use shared::data::Matrix;
use shared::replay::{Mode, Replayer, SyntheticInputs};
use shared::schedule::Schedule;
use shared::{BinaryMatrix, Kernel, KernelConfig, ThreadPool};
use v_sdot::{Preset, VSdotWith};

#[global_allocator]
static GA: CountingAlloc = CountingAlloc;

const MIB: usize = 1 << 20;
/// Frames: 8 threads x 8 frames (i8x2 R = 16) <= 67, so both C take the fused frames-split path.
const T: usize = 67;

macro_rules! point {
    ($name:ident, $idx:expr) => {
        struct $name;
        impl Preset for $name {
            const NAME: &'static str = concat!("v-sdot/", stringify!($name));
            fn tuning() -> serde_json::Value {
                v_sdot::grid()[$idx].clone()
            }
        }
    };
}
point!(P0, 0);
point!(P1, 1);
point!(P2, 2);
point!(P3, 3);
point!(P4, 4);
point!(P5, 5);

/// Loads the matrices (fresh copies, allocated after `live0` like the export), prepares them, warms up
/// with one replay and audits; then checks that one more replay leaves the live heap unchanged.
fn audit<K: Kernel>(base: &[BinaryMatrix], sched: &Schedule, pool: &ThreadPool, inputs: &SyntheticInputs, c: usize) -> StorageAudit {
    let live0 = bench::heap_live();
    let matrices: Vec<Matrix> = base.iter().map(|b| Matrix::Binary(b.clone())).collect();
    let cfg = KernelConfig { threads: pool.threads(), components: c, max_rows: inputs.t, max_k: 4096, max_n: 4096, tuning: serde_json::Value::Null };
    let mut rp = Replayer::<K>::new(&cfg, matrices, sched);
    rp.replay(pool, inputs, Mode::Timed).unwrap();
    let live = bench::heap_live() - live0;
    let a = bench::storage_audit(&rp, live);
    rp.replay(pool, inputs, Mode::Timed).unwrap();
    assert_eq!(bench::heap_live() - live0, live, "{}: C={c}: the live heap grew during a post-warm-up replay", K::NAME);
    drop(rp);
    a
}

#[test]
fn v_sdot_passes_the_storage_and_workspace_audit_at_every_grid_point() {
    let sched = Schedule::load();
    let base: Vec<BinaryMatrix> =
        sched.replayed().enumerate().map(|(i, c)| BinaryMatrix::random(c.n, c.k, 81_000 + i as u64)).collect();
    let rows: usize = sched.replayed().map(|c| c.n).sum();
    let pool = test_pool(8);
    let inputs = SyntheticInputs::new(&sched, T, 9);
    type Run = fn(&[BinaryMatrix], &Schedule, &ThreadPool, &SyntheticInputs, usize) -> StorageAudit;
    let points: [(&str, Run); 6] = [
        ("p0 (default)", audit::<VSdotWith<P0>>),
        ("p1", audit::<VSdotWith<P1>>),
        ("p2", audit::<VSdotWith<P2>>),
        ("p3", audit::<VSdotWith<P3>>),
        ("p4", audit::<VSdotWith<P4>>),
        ("p5", audit::<VSdotWith<P5>>),
    ];
    for (name, run) in points {
        for c in [2usize, 3] {
            let a = run(&base, &sched, &pool, &inputs, c);
            println!("v-sdot storage audit {name} C={c} threads=8 T={T}: {}", a.to_json());
            assert!(a.capped && a.ok, "{name} C={c}: {a:?}");
            assert_eq!(a.declared_code_bytes, bench::PACKED_1BIT_REPLAYED_BYTES, "{name} C={c}: codes are not exactly 1 bit per weight");
            assert_eq!(a.per_row_bytes, 4 * rows, "{name} C={c}: per-row bytes are more than the FP32 scales");
            assert!(a.weight_bytes >= a.declared_code_bytes, "{name} C={c}: {a:?}");
            assert!(a.weight_bytes - a.declared_code_bytes <= MIB, "{name} C={c}: unexplained live bytes: {a:?}");
            assert!(a.workspace_bytes <= bench::WORKSPACE_CAP_BYTES, "{name} C={c}: {a:?}");
        }
    }
}
