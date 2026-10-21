//! Storage and workspace acceptance contract (plans/cpu-benchmark-mac.md "Storage and workspace"; plan
//! review p8 major 3, restoring plans/cpu-benchmark.md's rule). Written before the implementation.
//!
//! Contract (shared::bench, shared::replay):
//! - `bench::PACKED_1BIT_REPLAYED_BYTES = 72_351_744` (the 240 replayed matrices at 1 bit per weight),
//!   `bench::WEIGHT_CAP_BYTES = 79_586_918` (floor of 1.10 x that), `bench::WORKSPACE_CAP_BYTES = 32 MiB`,
//!   `bench::PER_ROW_MAX_BYTES = 16` (per output row; excluded from the cap, reported);
//! - `bench::heap_live() -> usize`: live bytes of the counting allocator;
//! - `Replayer::harness_bytes(&self) -> usize`: the replayer's own live buffers;
//! - `bench::storage_audit(&rp, heap_live_since_load) -> StorageAudit` with `weight_bytes` = heap live
//!   since before loading minus the harness, the declared workspace and the declared per-row bytes;
//!   `capped` = the kernel loads the 1-bit export (`K::FORMAT == Binary`); `ok` and `reasons`: weight
//!   bytes above the cap (capped kernels), workspace above its cap (every kernel), per-row bytes above
//!   16 per output row (every kernel). The `block` process emits it after warmup and exits non-zero
//!   unless `ok`.
//!
//! One test function: the counting allocator is process-wide, so no other test may allocate meanwhile.

use shared::bench::{self, CountingAlloc, StorageAudit};
use shared::contract::test_pool;
use shared::data::Matrix;
use shared::reference::ScalarBinary;
use shared::replay::{Mode, Replayer, SyntheticInputs};
use shared::schedule::Schedule;
use shared::{BinaryMatrix, Call, Kernel, KernelConfig, Out, QuantizedActs, ThreadPool, WeightBytes, WeightFormat, WeightsRef};

#[global_allocator]
static GA: CountingAlloc = CountingAlloc;

const MIB: usize = 1 << 20;

/// What a test double keeps beyond the packed reference.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Extra {
    /// Nothing: a compliant 1-bit kernel.
    None,
    /// A 4x expansion of the codes kept between calls (e.g. retained index panels), declared as codes.
    RetainedPanels,
    /// An undeclared second copy of the codes (e.g. an export buffer that was never released).
    HiddenCopy,
    /// 40 MiB of declared workspace.
    WideWorkspace,
    /// 40 MiB allocated in `new` but not declared.
    HiddenWorkspace,
}

struct Double<const E: u8> {
    inner: ScalarBinary,
    _scratch: Vec<u8>,
}

struct Prepared {
    inner: <ScalarBinary as Kernel>::Prepared,
    kept: Vec<u8>,
    codes: usize,
}

const fn extra(e: u8) -> Extra {
    match e {
        0 => Extra::None,
        1 => Extra::RetainedPanels,
        2 => Extra::HiddenCopy,
        3 => Extra::WideWorkspace,
        _ => Extra::HiddenWorkspace,
    }
}

impl<const E: u8> Kernel for Double<E> {
    type Prepared = Prepared;
    const NAME: &'static str = "storage-double";
    const FORMAT: WeightFormat = WeightFormat::Binary;

    fn new(cfg: &KernelConfig) -> Self {
        let scratch = match extra(E) {
            Extra::WideWorkspace | Extra::HiddenWorkspace => vec![1u8; 40 * MIB],
            _ => Vec::new(),
        };
        Double { inner: ScalarBinary::new(cfg), _scratch: scratch }
    }

    fn prepare(&self, w: WeightsRef<'_>) -> Prepared {
        let codes = match w {
            WeightsRef::Binary(m) => m.codes.len(),
            WeightsRef::Ternary(_) => panic!("binary double"),
        };
        let kept = match (extra(E), w) {
            (Extra::RetainedPanels, WeightsRef::Binary(m)) => m.codes.iter().flat_map(|&b| [b, b ^ 1, b ^ 2, b ^ 3]).collect(),
            (Extra::HiddenCopy, WeightsRef::Binary(m)) => m.codes.clone(),
            _ => Vec::new(),
        };
        Prepared { inner: self.inner.prepare(w), kept, codes }
    }

    fn gemm(&mut self, pool: &ThreadPool, w: &Prepared, qa: &QuantizedActs, call: Call, out: Out<'_>) {
        self.inner.gemm(pool, &w.inner, qa, call, out);
    }

    fn unpack(&self, w: &Prepared) -> Vec<i8> {
        self.inner.unpack(&w.inner)
    }

    fn shape(&self, w: &Prepared) -> (usize, usize) {
        self.inner.shape(&w.inner)
    }

    fn weight_bytes(&self, w: &Prepared) -> WeightBytes {
        let per_row = self.inner.weight_bytes(&w.inner).per_row;
        let codes = match extra(E) {
            Extra::RetainedPanels => w.codes + w.kept.len(),
            _ => w.codes,
        };
        WeightBytes { codes, per_row }
    }

    fn workspace_bytes(&self) -> usize {
        if extra(E) == Extra::WideWorkspace { 40 * MIB } else { 0 }
    }
}

/// Loads random 1-bit weights of the real schedule (the "export"), prepares them, warms up with one replay,
/// and audits, measuring the heap from before loading as the `block` process does.
fn audit<K: Kernel>(sched: &Schedule, pool: &ThreadPool, inputs: &SyntheticInputs) -> StorageAudit {
    let live0 = bench::heap_live();
    let matrices: Vec<Matrix> = sched
        .replayed()
        .enumerate()
        .map(|(i, c)| Matrix::Binary(BinaryMatrix::random(c.n, c.k, 77_000 + i as u64)))
        .collect();
    let cfg = KernelConfig { threads: pool.threads(), components: 2, max_rows: inputs.t, max_k: 4096, max_n: 4096, tuning: serde_json::Value::Null };
    let mut rp = Replayer::<K>::new(&cfg, matrices, sched);
    rp.replay(pool, inputs, Mode::Timed).unwrap();
    let live = bench::heap_live() - live0;
    let a = bench::storage_audit(&rp, live);
    drop(rp);
    a
}

#[test]
fn storage_and_workspace_contract() {
    assert_eq!(bench::PACKED_1BIT_REPLAYED_BYTES, 72_351_744);
    assert_eq!(bench::WEIGHT_CAP_BYTES, 79_586_918);
    assert_eq!(bench::WORKSPACE_CAP_BYTES, 32 * MIB);
    assert_eq!(bench::PER_ROW_MAX_BYTES, 16);
    let sched = Schedule::load();
    let packed: usize = sched.replayed().map(|c| c.n * c.k.div_ceil(8)).sum();
    assert_eq!(packed, bench::PACKED_1BIT_REPLAYED_BYTES, "the cap's denominator is the replayed matrices' packed size");
    let pool = test_pool(1);
    let inputs = SyntheticInputs::new(&sched, 1, 5);

    // A compliant packed 1-bit kernel passes, and the audit is tight: everything it holds is its codes.
    let a = audit::<ScalarBinary>(&sched, &pool, &inputs);
    assert!(a.capped && a.ok, "{a:?}");
    assert!(a.weight_bytes >= a.declared_code_bytes, "{a:?}");
    assert!(a.weight_bytes - a.declared_code_bytes <= MIB, "unexplained live bytes for a compliant kernel: {a:?}");
    assert!(a.harness_bytes > 0, "{a:?}");
    let d = audit::<Double<0>>(&sched, &pool, &inputs);
    assert!(d.ok, "{d:?}");

    // Retained expansions, undeclared copies and undeclared workspace all count as weight bytes.
    for (case, a) in [
        ("retained 4x panels", audit::<Double<1>>(&sched, &pool, &inputs)),
        ("undeclared copy of the codes", audit::<Double<2>>(&sched, &pool, &inputs)),
        ("undeclared 40 MiB scratch", audit::<Double<4>>(&sched, &pool, &inputs)),
    ] {
        assert!(!a.ok, "{case}: {a:?}");
        assert!(a.weight_bytes > bench::WEIGHT_CAP_BYTES, "{case}: {a:?}");
        assert!(a.reasons.iter().any(|r| r.contains("weight bytes")), "{case}: {:?}", a.reasons);
    }

    // Declared workspace above 32 MiB fails for every kernel.
    let a = audit::<Double<3>>(&sched, &pool, &inputs);
    assert!(!a.ok, "{a:?}");
    assert!(a.reasons.iter().any(|r| r.contains("workspace")), "{:?}", a.reasons);
    assert!(a.weight_bytes <= bench::WEIGHT_CAP_BYTES, "declared workspace is not weight bytes: {a:?}");
}
