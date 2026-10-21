//! Every point of B-NEON's frozen tuning grid (nb x nr) is exact on the model shapes and on a
//! ragged shape, for C = 2 and 3, against the oracle (so tuning can never select an inexact point).
use shared::contract::{TestMatrix, check_call, quantized_gaussian, test_pool};
use shared::kernel::{Call, Kernel, KernelConfig, Slot};
use shared::weights::BinaryMatrix;

pub const GRID: [(usize, usize); 6] = [(32, 8), (32, 16), (64, 8), (64, 16), (128, 8), (128, 16)];

fn grid<K: Kernel>() {
    for &threads in &[1usize, 8] {
        let pool = test_pool(threads);
        for (nb, nr) in GRID {
            for c in [2usize, 3] {
                let cfg = KernelConfig { threads, components: c, max_rows: 23, max_k: 4096, max_n: 4096,
                                         tuning: serde_json::json!({"nb": nb, "nr": nr}) };
                let mut kernel = K::new(&cfg);
                for (si, &(n, k)) in [(1024usize, 1024usize), (4096, 1024), (1024, 4096), (2048, 1024), (200, 1000)].iter().enumerate() {
                    let m = TestMatrix::binary(BinaryMatrix::random(n, k, 77 + si as u64));
                    let p = kernel.prepare(m.as_ref(K::FORMAT).unwrap());
                    let qa = quantized_gaussian(&pool, 23, c, k, 5 + si as u64);
                    let what = format!("grid nb={nb} nr={nr} C={c} threads={threads} {n}x{k}");
                    check_call(&mut kernel, &pool, &p, &m, &qa, Call { slot: Slot::Model, reuse: false }, &what);
                }
            }
        }
    }
}

#[test]
fn grid_b_neon() {
    grid::<bneon::BNeon>();
}
#[test]
fn grid_b_neon_1bit_tile() {
    grid::<bneon::BNeon1BitTile>();
}
#[test]
fn grid_b_neon_pre_expanded() {
    grid::<bneon::BNeonPreExpanded>();
}
