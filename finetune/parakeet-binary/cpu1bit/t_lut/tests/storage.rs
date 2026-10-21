//! T's storage and workspace contract (plans/cpu-benchmark-mac.md "Storage and workspace"), measured
//! with H's counting allocator and H's `bench::storage_audit`, at the timed clips' largest T (353) and
//! 8 workers, for every point of the frozen grid and C = 2, 3:
//!
//! 1. Declared workspace is exact: right after `new`, and after calls at every model shape (including
//!    a call longer than `max_rows`, which grows the buffers once), the kernel's live heap equals
//!    `workspace_bytes()`. Nothing is hidden (it would count as weight bytes) and nothing is
//!    over-declared (it would hide weight bytes).
//! 2. The audit of the real 240-call schedule passes: declared codes are exactly the packed 1-bit
//!    size, 72,351,744 bytes (no expansion or index panel is kept with the weights); everything else
//!    the arm holds across calls is the 240 `Prepared` headers (at most 64 KiB); workspace at most
//!    32 MiB; per-row bytes 4 per output row (the weight scale).
//!
//! One test function: the counting allocator is process-wide, so nothing else may allocate meanwhile.
//! Other threads may allocate transiently (libtest's progress line), hence a 256-byte tolerance on
//! the exact comparisons; every T buffer is at least one 128-byte line per worker (1 KiB at 8).
use shared::bench::{self, CountingAlloc};
use shared::contract::test_pool;
use shared::data::Matrix;
use shared::replay::{Mode, Replayer, SyntheticInputs};
use shared::rng::SplitMix64;
use shared::schedule::Schedule;
use shared::{BinaryMatrix, Call, Kernel, KernelConfig, Out, QuantizedActs, Slot, WeightsRef};
use t_lut::TLut;

#[global_allocator]
static GA: CountingAlloc = CountingAlloc;

const T_MAX: usize = 353;
const THREADS: usize = 8;
const SLACK: usize = 256;

fn cfg(c: usize, max_rows: usize, tuning: &serde_json::Value) -> KernelConfig {
    KernelConfig { threads: THREADS, components: c, max_rows, max_k: 4096, max_n: 4096, tuning: tuning.clone() }
}

fn near(got: usize, want: usize) -> bool {
    got.abs_diff(want) <= SLACK
}

fn acts(t: usize, k: usize, c: usize, seed: u64) -> QuantizedActs {
    let mut rng = SplitMix64::new(seed);
    let q: Vec<i8> = (0..t * c * k).map(|_| rng.next_u64() as i8).collect();
    QuantizedActs::from_parts(t, k, c, &q, &vec![0.5; t])
}

#[test]
fn storage_and_workspace() {
    let sched = Schedule::load();
    let pool = test_pool(THREADS);
    let shapes = sched.shapes();
    let mats: Vec<BinaryMatrix> = shapes.iter().map(|&(n, k)| BinaryMatrix::random(n, k, (n * 7 + k) as u64)).collect();
    let schedule_mats: Vec<BinaryMatrix> =
        sched.replayed().enumerate().map(|(i, c)| BinaryMatrix::random(c.n, c.k, 91_000 + i as u64)).collect();
    let inputs = SyntheticInputs::new(&sched, T_MAX, 4);
    let rows: usize = sched.replayed().map(|c| c.n).sum();
    for tuning in TLut::tuning_grid() {
        for c in [2usize, 3] {
            let what = format!("tuning={tuning} C={c}");
            // 1. Exact workspace declaration (kernel only; weights, activations and outputs allocated first).
            {
                let preps: Vec<_> = {
                    let k0 = TLut::new(&cfg(c, 1, &tuning));
                    mats.iter().map(|m| k0.prepare(WeightsRef::Binary(m))).collect()
                };
                let qas: Vec<QuantizedActs> = shapes.iter().map(|&(_, k)| acts(T_MAX, k, c, k as u64)).collect();
                let long: Vec<QuantizedActs> = shapes.iter().map(|&(_, k)| acts(T_MAX + 40, k, c, k as u64 + 1)).collect();
                let mut y = vec![0f32; (T_MAX + 40) * 4096];
                let live0 = bench::heap_live();
                let mut kern = TLut::new(&cfg(c, T_MAX, &tuning));
                assert!(near(bench::heap_live() - live0, kern.workspace_bytes()), "{what}: after new: live {} declared {}",
                        bench::heap_live() - live0, kern.workspace_bytes());
                for (p, qa) in preps.iter().zip(&qas) {
                    kern.gemm(&pool, p, qa, Call { slot: Slot::Model, reuse: false }, Out::F32(&mut y));
                }
                let live = bench::heap_live() - live0;
                assert!(near(live, kern.workspace_bytes()), "{what}: after the model shapes: live {live} declared {}", kern.workspace_bytes());
                assert!(kern.workspace_bytes() <= bench::WORKSPACE_CAP_BYTES, "{what}: workspace {}", kern.workspace_bytes());
                for (p, qa) in preps.iter().zip(&long) {
                    kern.gemm(&pool, p, qa, Call { slot: Slot::Model, reuse: false }, Out::F32(&mut y));
                }
                let live = bench::heap_live() - live0;
                assert!(near(live, kern.workspace_bytes()), "{what}: after growth: live {live} declared {}", kern.workspace_bytes());
                drop(kern);
                assert!(near(bench::heap_live(), live0), "{what}: the kernel's workspace is released with it");
            }
            // 2. The audit of the real schedule, measured as the block process does (heap from before loading).
            let live0 = bench::heap_live();
            let loaded: Vec<Matrix> = schedule_mats.iter().map(|b| Matrix::Binary(b.clone())).collect();
            let mut rp = Replayer::<TLut>::new(&cfg(c, T_MAX, &tuning), loaded, &sched);
            rp.replay(&pool, &inputs, Mode::Timed).unwrap();
            let a = bench::storage_audit(&rp, bench::heap_live() - live0);
            println!("{what}: {}", a.to_json());
            assert!(a.capped && a.ok, "{what}: {a:?}");
            assert_eq!(a.declared_code_bytes, bench::PACKED_1BIT_REPLAYED_BYTES, "{what}: codes are exactly 1 bit per weight");
            assert!(a.weight_bytes >= a.declared_code_bytes, "{what}: {a:?}");
            let headers = rp.prepared.capacity() * std::mem::size_of::<t_lut::Prepared>();
            assert!(a.weight_bytes - a.declared_code_bytes <= (64 << 10).max(headers + SLACK),
                    "{what}: weight-derived bytes beyond the codes (retained expansions or panels?): {a:?}");
            assert!(a.workspace_bytes <= bench::WORKSPACE_CAP_BYTES, "{what}: {a:?}");
            assert_eq!(a.per_row_bytes, 4 * rows, "{what}: per-row bytes are the weight scales");
            drop(rp);
        }
    }
}
