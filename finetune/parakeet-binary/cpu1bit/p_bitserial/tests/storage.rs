//! Storage and workspace acceptance (plans/cpu-benchmark-mac.md "Storage and workspace") for P at every
//! frozen grid point, through H's audit (`shared::bench::storage_audit`) exactly as the `block` / `audit`
//! process runs it: the counting allocator as the global allocator, the live heap measured from before
//! the weights are loaded, the 240 replayed matrices (random 1-bit codes of the real schedule) prepared
//! by `Replayer::new`, one replay, then the audit. 8 workers, i8x2 and i8x3.
//!
//! P's claims, all observable through the audit:
//! - it passes (weight bytes <= 1.10 x 72,351,744; workspace <= 32 MiB; per-row <= 16 B per row);
//! - it holds exactly one bit per weight: declared codes = 72,351,744 bytes, and everything live that is
//!   not harness, declared workspace or per-row values is at most 64 KiB more (the `prepared` vector's
//!   own headers), i.e. nothing weight-derived is retained beyond the codes;
//! - per-row values are 8 bytes per output row (u32 correction + f32 scale);
//! - the declared workspace is the activation planes of the two replayed slots, one byte per padded
//!   frame-part and input: `ceil(T / mf) * mf * C * (1024 + 4096)` (mf = frames per tile at C). The
//!   real-capture sizes (T up to 353) come from `replay-p audit` on the MB1 export.
//!
//! One test function: the counting allocator is process-wide, so nothing else may allocate meanwhile.

use p_bitserial::{BitSerial, tuning_grid};
use shared::bench::{self, CountingAlloc, StorageAudit};
use shared::contract::test_pool;
use shared::data::Matrix;
use shared::replay::{Mode, Replayer, SyntheticInputs};
use shared::schedule::Schedule;
use shared::{BinaryMatrix, KernelConfig, ThreadPool};

#[global_allocator]
static GA: CountingAlloc = CountingAlloc;

const T: usize = 67;

fn audit(sched: &Schedule, pool: &ThreadPool, inputs: &SyntheticInputs, src: &[BinaryMatrix], c: usize,
         tuning: serde_json::Value) -> StorageAudit {
    let live0 = bench::heap_live();
    let matrices: Vec<Matrix> = src.iter().map(|m| Matrix::Binary(m.clone())).collect();
    let cfg = KernelConfig { threads: pool.threads(), components: c, max_rows: inputs.t, max_k: 4096, max_n: 4096, tuning };
    let mut rp = Replayer::<BitSerial>::new(&cfg, matrices, sched);
    rp.replay(pool, inputs, Mode::Timed).unwrap();
    let a = bench::storage_audit(&rp, bench::heap_live() - live0);
    drop(rp);
    a
}

#[test]
fn storage_and_workspace_at_every_grid_point() {
    let sched = Schedule::load();
    let src: Vec<BinaryMatrix> =
        sched.replayed().enumerate().map(|(i, c)| BinaryMatrix::random(c.n, c.k, 31_000 + i as u64)).collect();
    let rows: usize = sched.replayed().map(|c| c.n).sum();
    let pool = test_pool(8);
    let inputs = SyntheticInputs::new(&sched, T, 7);
    for c in [2usize, 3] {
        for (i, point) in tuning_grid().into_iter().enumerate() {
            let a = audit(&sched, &pool, &inputs, &src, c, point.to_json());
            let what = format!("grid point {i} {point:?} C={c}");
            assert!(a.capped && a.ok, "{what}: {a:?}");
            assert_eq!(a.declared_code_bytes, bench::PACKED_1BIT_REPLAYED_BYTES, "{what}: one bit per weight");
            assert!(a.weight_bytes >= a.declared_code_bytes, "{what}: {a:?}");
            assert!(a.weight_bytes - a.declared_code_bytes <= 64 << 10, "{what}: weight-derived bytes beyond the codes: {a:?}");
            assert_eq!(a.per_row_bytes, 8 * rows, "{what}: {a:?}");
            let mf = point.frames[c - 1];
            let planes = T.div_ceil(mf) * mf * c * (1024 + 4096);
            assert_eq!(a.workspace_bytes, planes, "{what}: declared workspace = the two slots' planes: {a:?}");
            assert!(a.workspace_bytes <= bench::WORKSPACE_CAP_BYTES, "{what}: {a:?}");
            println!("{what}: weight {} (codes {}, +{}), per-row {}, workspace {}, harness {}, heap {}", a.weight_bytes,
                     a.declared_code_bytes, a.weight_bytes - a.declared_code_bytes, a.per_row_bytes, a.workspace_bytes,
                     a.harness_bytes, a.heap_live_bytes);
        }
    }
}
