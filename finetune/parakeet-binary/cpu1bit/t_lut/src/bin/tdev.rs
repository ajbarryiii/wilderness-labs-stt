//! T dev-tuning bench (dev-tuning, background QoS, not comparable). Dev inputs only: random binary
//! weights at the model shapes and the shared quantizer on Gaussian frames with outliers.
//!
//!   tdev [threads=1] [frames=64] [components=2] [trials=15]
//!
//! For each model shape: every tuning point of T (fresh quantization, i.e. tables built, and
//! reuse, i.e. tables skipped) and an in-process pre-expanded SDOT control (s8 weights, no
//! expansion cost, a 4 x 16 SDOT-by-element microkernel; frames split across threads like T).
//! Trials interleave the variants; the minimum time per variant is reported, with a dependent-add
//! frequency probe around each round (background QoS runs on time-sliced E-cores).
#![allow(clippy::needless_range_loop)]
#[cfg(target_arch = "aarch64")]
use std::arch::asm;
use std::hint::black_box;
use std::time::Instant;

use shared::contract::quantized_gaussian;
use shared::{BinaryMatrix, Call, Kernel, KernelConfig, Out, Qos, QuantizedActs, SendPtr, Slot, ThreadPool, WeightsRef, split_range};
use shared::aligned::AlignedVec;
use t_lut::{TLut, Tuning};

fn probe_ghz() -> f64 {
    let iters: u64 = 3_000_000;
    let t = Instant::now();
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let mut x: u64 = 1;
        asm!(
            "2:",
            "add {x}, {x}, {one}", "add {x}, {x}, {one}", "add {x}, {x}, {one}", "add {x}, {x}, {one}",
            "add {x}, {x}, {one}", "add {x}, {x}, {one}", "add {x}, {x}, {one}", "add {x}, {x}, {one}",
            "add {x}, {x}, {one}", "add {x}, {x}, {one}", "add {x}, {x}, {one}", "add {x}, {x}, {one}",
            "add {x}, {x}, {one}", "add {x}, {x}, {one}", "add {x}, {x}, {one}", "add {x}, {x}, {one}",
            "subs {n}, {n}, #1",
            "b.ne 2b",
            x = inout(reg) x, one = in(reg) 1u64, n = inout(reg) iters => _,
            options(nostack)
        );
        black_box(x);
    }
    (iters * 16) as f64 / t.elapsed().as_secs_f64() / 1e9
}

/// Pre-expanded SDOT control: s8 weights `[N/16][K/4][4 vectors of 16 B]` (vector v = outputs
/// 4v..4v+3 x 4 inputs), dots only (no epilogue).
struct SdotControl {
    n: usize,
    k: usize,
    tile: Vec<i8>,
}

impl SdotControl {
    fn new(m: &BinaryMatrix) -> Self {
        assert!(m.n.is_multiple_of(16) && m.k.is_multiple_of(16));
        let (n, k) = (m.n, m.k);
        let mut tile = vec![0i8; n * k];
        for ob in 0..n / 16 {
            for kq in 0..k / 4 {
                for v in 0..4 {
                    for l in 0..4 {
                        for b in 0..4 {
                            tile[((ob * (k / 4) + kq) * 4 + v) * 16 + l * 4 + b] = m.sign(ob * 16 + 4 * v + l, 4 * kq + b);
                        }
                    }
                }
            }
        }
        SdotControl { n, k, tile }
    }

    fn gemm(&self, pool: &ThreadPool, qa: &QuantizedActs, d: &mut [i32]) {
        let (t, c, k, n) = (qa.rows(), qa.components(), self.k, self.n);
        let rows = t * c;
        let dp = SendPtr(d.as_mut_ptr());
        let threads = pool.threads();
        pool.run(&|idx| {
            let fr = split_range(t, threads, idx);
            let (r0, r1) = (fr.start * c, fr.end * c);
            for ob in 0..n / 16 {
                let w = unsafe { self.tile.as_ptr().add(ob * 16 * k) };
                let mut r = r0;
                while r + 4 <= r1 {
                    unsafe { sdot4x16(w, qa.q().as_ptr().add(r * k), k, dp.get().add(r * n + ob * 16), n) };
                    r += 4;
                }
                while r < r1 {
                    unsafe { sdot1x16(w, qa.q().as_ptr().add(r * k), k, dp.get().add(r * n + ob * 16)) };
                    r += 1;
                }
            }
        });
        black_box(rows);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(never)]
#[target_feature(enable = "neon,dotprod")]
unsafe fn sdot4x16(w: *const i8, a: *const i8, k: usize, d: *mut i32, ldd: usize) {
    use std::arch::aarch64::*;
    unsafe {
        let mut acc = [[vdupq_n_s32(0); 4]; 4];
        let mut q = 0;
        while q < k / 4 {
            let av: [int8x16_t; 4] = std::array::from_fn(|r| vld1q_s8(a.add(r * k + q * 4)));
            macro_rules! step {
                ($l:literal) => {{
                    let wp = w.add((q + $l) * 64);
                    let wv: [int8x16_t; 4] = std::array::from_fn(|v| vld1q_s8(wp.add(v * 16)));
                    for r in 0..4 {
                        for v in 0..4 {
                            acc[r][v] = vdotq_laneq_s32::<$l>(acc[r][v], wv[v], av[r]);
                        }
                    }
                }};
            }
            step!(0);
            step!(1);
            step!(2);
            step!(3);
            q += 4;
        }
        for r in 0..4 {
            for v in 0..4 {
                vst1q_s32(d.add(r * ldd + 4 * v), acc[r][v]);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(never)]
#[target_feature(enable = "neon,dotprod")]
unsafe fn sdot1x16(w: *const i8, a: *const i8, k: usize, d: *mut i32) {
    use std::arch::aarch64::*;
    unsafe {
        let mut acc = [vdupq_n_s32(0); 4];
        for q in 0..k / 4 {
            let av = vreinterpretq_s8_s32(vld1q_dup_s32(a.add(q * 4) as *const i32));
            for (v, a) in acc.iter_mut().enumerate() {
                *a = vdotq_s32(*a, vld1q_s8(w.add(q * 64 + v * 16)), av);
            }
        }
        for (v, a) in acc.iter().enumerate() {
            vst1q_s32(d.add(4 * v), *a);
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
unsafe fn sdot4x16(w: *const i8, a: *const i8, k: usize, d: *mut i32, ldd: usize) {
    for r in 0..4 {
        unsafe { sdot1x16(w, a.add(r * k), k, d.add(r * ldd)) };
    }
}

#[cfg(not(target_arch = "aarch64"))]
unsafe fn sdot1x16(w: *const i8, a: *const i8, k: usize, d: *mut i32) {
    unsafe {
        for o in 0..16 {
            let (v, l) = (o / 4, o % 4);
            let mut s = 0i32;
            for q in 0..k / 4 {
                for b in 0..4 {
                    s += *w.add(q * 64 + v * 16 + l * 4 + b) as i32 * *a.add(q * 4 + b) as i32;
                }
            }
            *d.add(o) = s;
        }
    }
}

fn arg(i: usize, default: usize) -> usize {
    std::env::args().nth(i).map(|s| s.parse().expect("integer argument")).unwrap_or(default)
}

fn main() {
    let (threads, frames, comps, trials) = (arg(1, 1), arg(2, 64), arg(3, 2), arg(4, 15));
    println!("dev-tuning, background QoS, not comparable");
    println!("tdev threads={threads} frames={frames} components={comps} trials={trials} rustc={}", shared::RUSTC);
    let pool = ThreadPool::new(threads, Qos::Inherit);
    let shapes = [(1024usize, 1024usize), (1024, 4096), (2048, 1024), (4096, 1024)];
    let grid = TLut::tuning_grid();
    for (si, &(n, k)) in shapes.iter().enumerate() {
        let m = BinaryMatrix::random(n, k, 11 + si as u64);
        let qa = quantized_gaussian(&pool, frames, comps, k, 100 + si as u64);
        let macs = (frames * comps * n * k) as f64;
        let mut kernels: Vec<(String, TLut, t_lut::Prepared)> = grid
            .iter()
            .map(|tu| {
                let cfg = KernelConfig { threads, components: comps, max_rows: frames, max_k: 4096, max_n: 4096, tuning: tu.clone() };
                let kern = TLut::new(&cfg);
                let p = kern.prepare(WeightsRef::Binary(&m));
                (tu.to_string(), kern, p)
            })
            .collect();
        let control = SdotControl::new(&m);
        let mut y = vec![0f32; frames * n];
        let mut d = vec![0i32; frames * comps * n];
        // Correctness of the control against T's dots (same integer d).
        {
            let (_, kern, p) = &mut kernels[0];
            kern.gemm(&pool, p, &qa, Call { slot: Slot::Model, reuse: false }, Out::Dots(&mut d));
            let mut dc = vec![0i32; frames * comps * n];
            control.gemm(&pool, &qa, &mut dc);
            assert_eq!(d, dc, "control dots differ from T at {n}x{k}");
        }
        let nv = kernels.len();
        let tu0 = Tuning::parse(&grid[0]);
        let mut tbuf = AlignedVec::<i16>::zeroed(frames * comps * (tu0.k_chunk / 4) * 16);
        let mut pbuf = AlignedVec::<u8>::zeroed(tu0.n_panel * (tu0.k_chunk / 4) * 2);
        // Per round: [fresh, reuse] per variant, then control, then the two phases of the default.
        let mut rounds: Vec<Vec<f64>> = Vec::new();
        let mut probes = Vec::new();
        for _ in 0..trials {
            probes.push(probe_ghz());
            let mut r = Vec::with_capacity(2 * nv + 3);
            for (_, kern, p) in kernels.iter_mut() {
                let t0 = Instant::now();
                kern.gemm(&pool, p, &qa, Call { slot: Slot::Model, reuse: false }, Out::F32(&mut y));
                r.push(t0.elapsed().as_secs_f64());
                let t1 = Instant::now();
                kern.gemm(&pool, p, &qa, Call { slot: Slot::Model, reuse: true }, Out::F32(&mut y));
                r.push(t1.elapsed().as_secs_f64());
            }
            let t0 = Instant::now();
            control.gemm(&pool, &qa, &mut d);
            r.push(t0.elapsed().as_secs_f64());
            let t0 = Instant::now();
            black_box(t_lut::dev::tables_only(&qa, tu0, &mut tbuf));
            r.push(t0.elapsed().as_secs_f64());
            let t0 = Instant::now();
            t_lut::dev::panels_only(&kernels[0].2, tu0, &mut pbuf);
            black_box(&pbuf);
            r.push(t0.elapsed().as_secs_f64());
            black_box(&y);
            rounds.push(r);
        }
        let min = |i: usize| rounds.iter().map(|r| r[i]).fold(f64::INFINITY, f64::min);
        let med = |f: &dyn Fn(&Vec<f64>) -> f64| {
            let mut v: Vec<f64> = rounds.iter().map(f).collect();
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        probes.sort_by(f64::total_cmp);
        let ctl = min(2 * nv);
        println!(
            "shape N={n} K={k}: probe GHz min/med/max {:.2}/{:.2}/{:.2}; SDOT control {:.3} ms {:.1} GMAC/s",
            probes[0],
            probes[probes.len() / 2],
            probes[probes.len() - 1],
            ctl * 1e3,
            macs / ctl / 1e9
        );
        for (vi, (name, kern, _)) in kernels.iter().enumerate() {
            let (f, r) = (min(2 * vi), min(2 * vi + 1));
            let paired = med(&|x: &Vec<f64>| x[2 * vi] / x[2 * nv]);
            println!(
                "  T {name:<40} fresh {:.3} ms {:>5.1} GMAC/s, vs control: min {:+.1}% paired-median {:+.1}% | reuse {:.3} ms | ws {:.2} MB",
                f * 1e3,
                macs / f / 1e9,
                (f / ctl - 1.0) * 100.0,
                (paired - 1.0) * 100.0,
                r * 1e3,
                kern.workspace_bytes() as f64 / 1e6
            );
        }
        let f0 = min(0);
        println!(
            "  phases of the default point (single-thread equivalents): tables {:.1}%, index panels {:.1}% of its fresh call; reuse saves {:.1}%",
            min(2 * nv + 1) / f0 * 100.0,
            min(2 * nv + 2) / f0 * 100.0,
            (1.0 - min(1) / f0) * 100.0
        );
    }
}
