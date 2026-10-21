//! P-specific extremes beyond H's shapes, through the public `Kernel` API against H's oracle
//! (`shared::contract::check_call`, both output modes, bit for bit):
//! - `Uadalp` flushes its u16 plane counters into u32 every 4,095 steps of 16 positions
//!   (65,520 positions); K = 65,520, 65,521 and 131,056 cover exactly one block, one block plus a
//!   partial step, and two full blocks plus one step;
//! - the maximal count C = 255 K (all u = 255 under +1 weights, all u = 0 under -1 weights) and
//!   the extremal patterns of the contract, at those K, for C = 1..=3 and 1 / 4 / 8 threads.
//! |d| <= 128 K = 16,775,168 < 2^24 at the largest K, so the FP32 epilogue stays exact.

use p_bitserial::{Accum, BitSerialWith, Preset, Split, Tuning};
use shared::contract::{THREADS, TestMatrix, activation_patterns, check_call, sign_patterns, test_pool};
use shared::{Call, Kernel, KernelConfig, Slot};

macro_rules! fixed {
    ($ty:ident, $name:literal, $accum:ident, [$f1:expr, $f2:expr, $f3:expr], $split:ident) => {
        struct $ty;
        impl Preset for $ty {
            const NAME: &'static str = $name;
            fn tuning() -> Option<Tuning> {
                Some(Tuning { accum: Accum::$accum, o: 4, frames: [$f1, $f2, $f3], split: Split::$split })
            }
        }
    };
}

fixed!(UadalpO4, "p-bitserial/uadalp-o4", Uadalp, [4, 2, 1], Outputs);
fixed!(UdotO4, "p-bitserial/udot-o4", Udot, [4, 2, 1], Outputs);
fixed!(UdotLaneF6, "p-bitserial/udot-lane-f6", UdotLane, [6, 3, 2], Outputs);
fixed!(UdotFrames, "p-bitserial/udot-frames", Udot, [2, 1, 1], Frames);

fn long_k<K: Kernel>() {
    for &threads in &THREADS {
        let pool = test_pool(threads);
        for c in 1..=3 {
            for (ki, &k) in [65_520usize, 65_521, 131_056].iter().enumerate() {
                let (n, t) = (9, 3);
                let cfg = KernelConfig { threads, components: c, max_rows: t, max_k: k, max_n: n, tuning: serde_json::Value::Null };
                let mut kernel = K::new(&cfg);
                for (wname, b) in sign_patterns(n, k, 31 + ki as u64) {
                    let m = TestMatrix::binary(b);
                    let p = kernel.prepare(m.as_ref(K::FORMAT).unwrap());
                    for (aname, qa) in activation_patterns(t, c, k, 41 + ki as u64) {
                        let what = format!("threads={threads} C={c} K={k} w={wname} a={aname}");
                        check_call(&mut kernel, &pool, &p, &m, &qa, Call { slot: Slot::Ff, reuse: false }, &what);
                    }
                }
            }
        }
    }
}

#[test]
fn uadalp_o4_flush_boundaries_and_maximal_counts() {
    long_k::<BitSerialWith<UadalpO4>>();
}

#[test]
fn udot_long_k_and_maximal_counts() {
    long_k::<BitSerialWith<UdotO4>>();
}

#[test]
fn udot_lane_f6_long_k_and_maximal_counts() {
    long_k::<BitSerialWith<UdotLaneF6>>();
}

#[test]
fn udot_frame_split_long_k_and_maximal_counts() {
    long_k::<BitSerialWith<UdotFrames>>();
}

/// Storage (plans/cpu-benchmark.md "Storage"): one bit per weight for every model shape and every
/// grid point, no padding, so MB1's 264 matrices hold exactly the 75,497,472 code bytes.
#[test]
fn model_shapes_store_exactly_one_bit_per_weight() {
    let cfg = KernelConfig { threads: 1, components: 2, max_rows: 4, max_k: 4096, max_n: 4096, tuning: serde_json::Value::Null };
    for tuning in p_bitserial::tuning_grid() {
        let (accum, o) = (tuning.accum, tuning.o);
        let kernel = p_bitserial::BitSerial::with_tuning(&cfg, tuning);
        for (n, k) in shared::schedule::Schedule::load().shapes() {
            let m = shared::BinaryMatrix::random(n, k, (n * 31 + k) as u64);
            let p = kernel.prepare(shared::WeightsRef::Binary(&m));
            let wb = kernel.weight_bytes(&p);
            assert_eq!(wb.codes, n * k / 8, "{accum:?} O={o} {n}x{k}");
            assert_eq!(wb.per_row, 2 * 4 * n, "{accum:?} O={o} {n}x{k}: u32 correction + f32 scale per row");
        }
    }
}
