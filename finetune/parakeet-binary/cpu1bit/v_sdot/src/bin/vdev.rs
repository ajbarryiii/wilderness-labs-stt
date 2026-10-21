//! Dev-tuning microbenchmarks for V's kernel. NOT valid timings: they run under the Mac job
//! supervisor's background priority while the user works ("dev-tuning, background QoS, not
//! comparable"). To damp interference every number is the minimum over interleaved trials, and
//! kernel rates are also given relative to an SDOT-peak loop measured in the same trial rounds.
//!
//!   vdev [kernel|gemm|all] [--trials N]

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("aarch64 only");
}

#[cfg(target_arch = "aarch64")]
fn main() {
    imp::main();
}

#[cfg(target_arch = "aarch64")]
mod imp {
    use core::arch::aarch64::*;
    use shared::rng::SplitMix64;
    use shared::{BinaryMatrix, Call, Kernel, KernelConfig, Out, Qos, QuantizedActs, Slot, ThreadPool, WeightsRef};
    use std::hint::black_box;
    use std::time::Instant;
    use v_sdot::{VSdot, layout};

    fn qos() -> String {
        let (q, r) = shared::pool::read_qos();
        format!("qos={q:#x} rel={r}")
    }

    /// Min over trials of `f()`'s wall time in ns.
    fn min_ns(trials: usize, mut f: impl FnMut()) -> f64 {
        (0..trials)
            .map(|_| {
                let t = Instant::now();
                f();
                t.elapsed().as_nanos() as f64
            })
            .fold(f64::INFINITY, f64::min)
    }

    /// 16 accumulators of by-element SDOTs, `iters * 64` SDOTs.
    #[inline(never)]
    fn sdot_peak(iters: u64) {
        unsafe {
            let w: [int8x16_t; 4] = core::array::from_fn(|i| vdupq_n_s8(black_box(i as i8 + 1)));
            let a = vdupq_n_s8(black_box(3));
            let mut acc = [vdupq_n_s32(0); 16];
            for _ in 0..iters {
                for r in 0..16 {
                    acc[r] = vdotq_laneq_s32::<0>(acc[r], w[0], a);
                    acc[r] = vdotq_laneq_s32::<1>(acc[r], w[1], a);
                    acc[r] = vdotq_laneq_s32::<2>(acc[r], w[2], a);
                    acc[r] = vdotq_laneq_s32::<3>(acc[r], w[3], a);
                }
                core::arch::asm!("", options(nomem, nostack));
            }
            black_box(acc);
        }
    }

    /// `n` dependent 1-cycle integer adds (the clock reference).
    #[inline(never)]
    fn add_chain(n: u64) {
        let mut x: u64 = black_box(1);
        let mut n = n;
        unsafe {
            core::arch::asm!(
                "2:",
                "add {x}, {x}, #1", "add {x}, {x}, #1", "add {x}, {x}, #1", "add {x}, {x}, #1",
                "add {x}, {x}, #1", "add {x}, {x}, #1", "add {x}, {x}, #1", "add {x}, {x}, #1",
                "subs {n}, {n}, #8",
                "b.ne 2b",
                x = inout(reg) x, n = inout(reg) n,
                options(nomem, nostack),
            );
        }
        black_box((x, n));
    }

    /// `n` by-element SDOTs, each depending on the previous one through the accumulator.
    #[inline(never)]
    fn sdot_chain(n: u64) {
        unsafe {
            let mut acc = vdupq_n_s32(0);
            let w = vdupq_n_s8(black_box(1));
            core::arch::asm!(
                "2:",
                "sdot {acc:v}.4s, {w:v}.16b, {w:v}.4b[0]", "sdot {acc:v}.4s, {w:v}.16b, {w:v}.4b[1]",
                "sdot {acc:v}.4s, {w:v}.16b, {w:v}.4b[2]", "sdot {acc:v}.4s, {w:v}.16b, {w:v}.4b[3]",
                "sdot {acc:v}.4s, {w:v}.16b, {w:v}.4b[0]", "sdot {acc:v}.4s, {w:v}.16b, {w:v}.4b[1]",
                "sdot {acc:v}.4s, {w:v}.16b, {w:v}.4b[2]", "sdot {acc:v}.4s, {w:v}.16b, {w:v}.4b[3]",
                "subs {n}, {n}, #8",
                "b.ne 2b",
                acc = inout(vreg) acc, w = in(vreg) w, n = inout(reg) n => _,
                options(nomem, nostack),
            );
            black_box(acc);
        }
    }

    /// Dependent-SDOT latency in core cycles (ratio to the 1-cycle add chain, interleaved trials).
    fn sdot_latency(trials: usize) -> f64 {
        let n = 8_000_000u64;
        let (mut ta, mut ts) = (f64::INFINITY, f64::INFINITY);
        for _ in 0..trials {
            ta = ta.min(min_ns(1, || add_chain(n)));
            ts = ts.min(min_ns(1, || sdot_chain(n)));
        }
        ts / ta
    }

    /// The compiler-scheduled intrinsics version of the same loop (for the asm comparison).
    #[inline(never)]
    unsafe fn dot_intrinsics<const R: usize>(w: *const u8, a: *const i8, lda: usize, chunks: usize) -> [int32x4_t; R] {
        unsafe {
            let m: [uint8x16_t; 8] = core::array::from_fn(|j| vdupq_n_u8(1 << j));
            let mut acc = [vdupq_n_s32(0); R];
            let (mut wp, mut ap) = (w, a);
            for _ in 0..chunks {
                let src = vld1q_u8(wp);
                for h in 0..2 {
                    let w0 = vreinterpretq_s8_u8(vtstq_u8(src, m[4 * h]));
                    let w1 = vreinterpretq_s8_u8(vtstq_u8(src, m[4 * h + 1]));
                    let w2 = vreinterpretq_s8_u8(vtstq_u8(src, m[4 * h + 2]));
                    let w3 = vreinterpretq_s8_u8(vtstq_u8(src, m[4 * h + 3]));
                    for r in 0..R {
                        let av = vld1q_s8(ap.add(r * lda + 16 * h));
                        acc[r] = vdotq_laneq_s32::<0>(acc[r], w0, av);
                        acc[r] = vdotq_laneq_s32::<1>(acc[r], w1, av);
                        acc[r] = vdotq_laneq_s32::<2>(acc[r], w2, av);
                        acc[r] = vdotq_laneq_s32::<3>(acc[r], w3, av);
                    }
                }
                wp = wp.add(16);
                ap = ap.add(32);
            }
            acc
        }
    }

    type DotFn<const R: usize> = unsafe fn(*const u8, *const i8, usize, usize) -> [int32x4_t; R];

    /// How to turn a variant's accumulators into the exact dot: 0 = {0,-1}: d = -2z - U;
    /// 1 = ±1 as -w: d = -z; 2 = {0,1}: d = 2z - U.
    #[derive(Clone, Copy)]
    enum Ident {
        Z,
        Pm,
        B01,
    }

    struct Variant<const R: usize> {
        name: &'static str,
        f: DotFn<R>,
        ident: Ident,
        /// Weights pre-expanded to {0,-1} bytes (128 B per chunk) instead of the 1-bit v-layout.
        tile: bool,
    }

    fn median(v: &mut [f64]) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if v.is_empty() { f64::NAN } else { v[v.len() / 2] }
    }

    /// Variants on one L1-resident panel of R frame-parts against n/4 groups: exactness first (each
    /// with its identity), then per trial, back to back: the add-chain clock, the SDOT peak loop
    /// with the same SDOT count, and every variant. Each trial is classified by the peak loop's
    /// SDOT/cycle (P-core ~4, E-core ~2; the job's QoS lets macOS move us), and the median ratio
    /// variant / peak loop is reported per class.
    fn variants_case<const R: usize>(n: usize, k: usize, trials: usize, variants: &[Variant<R>]) {
        let mut rng = SplitMix64::new(42 + R as u64);
        let rb = k / 8;
        let codes: Vec<u8> = (0..n * rb).map(|_| rng.next_u64() as u8).collect();
        let q: Vec<i8> = (0..R * k).map(|_| rng.next_u64() as i8).collect();
        let (groups, chunks) = (n / 4, k / 32);
        let main = layout::build_main(&codes, rb, n, k, groups, chunks);
        let tile: Vec<u8> = (0..groups * chunks * 128)
            .map(|i| {
                let (gc, j, b) = (i / 128, (i / 16) % 8, i % 16);
                if (main[gc * 16 + b] >> j) & 1 == 1 { 0xFF } else { 0 }
            })
            .collect();
        let wb = |v: &Variant<R>| if v.tile { (tile.as_ptr(), chunks * 128) } else { (main.as_ptr(), chunks * 16) };
        let lanes = |v: int32x4_t| {
            let mut o = [0i32; 4];
            unsafe { vst1q_s32(o.as_mut_ptr(), v) };
            o
        };
        for v in variants {
            let (w, stride) = wb(v);
            for g in [0, groups - 1] {
                let z = unsafe { (v.f)(w.add(g * stride), q.as_ptr(), k, chunks) };
                for r in 0..R {
                    let row = &q[r * k..(r + 1) * k];
                    let u: i32 = row.iter().map(|&x| x as i32).sum();
                    for nl in 0..4 {
                        let want: i32 = row
                            .iter()
                            .enumerate()
                            .map(|(i, &x)| (2 * layout::code_bit(&codes, rb, 4 * g + nl, i) as i32 - 1) * x as i32)
                            .sum();
                        let zl = lanes(z[r])[nl];
                        let got = match v.ident {
                            Ident::Z => -2 * zl - u,
                            Ident::Pm => -zl,
                            Ident::B01 => 2 * zl - u,
                        };
                        assert_eq!(got, want, "{} R={R} g={g} r={r}", v.name);
                    }
                }
            }
        }
        let sdots = (groups * R * k / 4) as f64;
        let adds = 1_000_000u64;
        let mut ratios: [Vec<Vec<f64>>; 2] = [vec![Vec::new(); variants.len()], vec![Vec::new(); variants.len()]];
        let mut cyc: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
        let mut other = 0;
        for _ in 0..trials {
            let tc = min_ns(1, || add_chain(adds));
            let tp = min_ns(1, || sdot_peak((sdots / 64.0) as u64));
            let tv: Vec<f64> = variants
                .iter()
                .map(|v| {
                    let (w, stride) = wb(v);
                    min_ns(1, || {
                        for g in 0..groups {
                            black_box(unsafe { (v.f)(w.add(g * stride), q.as_ptr(), k, chunks) });
                        }
                    })
                })
                .collect();
            let per_cycle = sdots / tp / (adds as f64 / tc);
            let class = if per_cycle > 3.3 && per_cycle < 4.6 { 0 } else if per_cycle > 1.6 && per_cycle < 2.4 { 1 } else { other += 1; continue };
            cyc[class].push(per_cycle);
            for (x, t) in tv.iter().enumerate() {
                ratios[class][x].push(tp / t);
            }
        }
        for (class, label) in [(0, "P-core-like (peak ~4 SDOT/cycle)"), (1, "E-core-like (peak ~2 SDOT/cycle)")] {
            if cyc[class].is_empty() {
                continue;
            }
            let mut line = format!(
                "R={R:2} n={n} k={k} {label}: {} trials, peak {:.2} SDOT/cycle; % of peak loop:",
                cyc[class].len(),
                median(&mut cyc[class])
            );
            for (x, v) in variants.iter().enumerate() {
                line += &format!(" {} {:.1}% |", v.name, 100.0 * median(&mut ratios[class][x]));
            }
            println!("{line}");
        }
        if other > 0 {
            println!("R={R:2} n={n} k={k}: {other} trials rejected (peak loop outside both bands: preempted or migrated within the trial)");
        }
    }

    struct GemmCase {
        name: String,
        p: v_sdot::Prepared,
        qa: QuantizedActs,
        y: Vec<f32>,
        mac: f64,
        best: f64,
    }

    /// The four model shapes at T frames, C in {2, 3}, measured round-robin (min per case).
    fn gemm_cases(threads: usize, t: usize, trials: usize, split: &str) {
        let pool = ThreadPool::new(threads, Qos::Inherit);
        let mut cases = Vec::new();
        let mut kerns = Vec::new();
        for c in [2usize, 3] {
            let cfg = KernelConfig {
                threads,
                components: c,
                max_rows: t,
                max_k: 4096,
                max_n: 4096,
                tuning: serde_json::json!({ "split": split }),
            };
            kerns.push(VSdot::new(&cfg));
            for &(n, k) in &[(1024usize, 1024usize), (4096, 1024), (1024, 4096), (2048, 1024)] {
                let m = BinaryMatrix::random(n, k, (n * 31 + k) as u64);
                let p = kerns[0].prepare(WeightsRef::Binary(&m));
                let mut rng = SplitMix64::new(7);
                let x: Vec<f32> = (0..t * k).map(|_| rng.next_gaussian()).collect();
                let mut qa = QuantizedActs::with_capacity(t, k, c);
                qa.quantize(&pool, &x, t).unwrap();
                cases.push((c, GemmCase { name: format!("C{c} {n}x{k}"), p, qa, y: vec![0f32; t * n], mac: (t * c * n * k) as f64, best: f64::INFINITY }));
            }
        }
        for _ in 0..trials {
            for (c, case) in cases.iter_mut() {
                let kern = &mut kerns[*c - 2];
                let ns = min_ns(1, || {
                    kern.gemm(&pool, &case.p, &case.qa, Call { slot: Slot::Model, reuse: false }, Out::F32(&mut case.y));
                });
                case.best = case.best.min(ns);
            }
        }
        let mut line = String::new();
        for c in [2usize, 3] {
            let (mut mac, mut ns) = (0f64, 0f64);
            for (cc, case) in &cases {
                if *cc == c {
                    line += &format!(" {} {:.0} us {:.0} GMAC/s |", case.name, case.best / 1e3, case.mac / case.best);
                    mac += case.mac;
                    ns += case.best;
                }
            }
            line += &format!(" C{c} all {:.0} GMAC/s ||", mac / ns);
        }
        println!("gemm threads={threads} T={t} split={split}:{line}");
    }

    pub fn main() {
        let args: Vec<String> = std::env::args().collect();
        let what = args.get(1).map(String::as_str).unwrap_or("all");
        let trials = args.iter().position(|a| a == "--trials").map(|i| args[i + 1].parse().unwrap()).unwrap_or(9);
        println!("vdev: dev-tuning, background QoS, not comparable. {} trials={trials}", qos());
        if what == "lat" || what == "kernel" || what == "all" {
            println!("dependent by-element SDOT latency: {:.2} cycles (vs 1-cycle add chain)", sdot_latency(trials));
        }
        if what == "kernel" || what == "all" {
            use v_sdot::kernel::asm::*;
            fn z<const R: usize>(name: &'static str, f: DotFn<R>) -> Variant<R> {
                Variant { name, f, ident: Ident::Z, tile: false }
            }
            for &(n, k) in &[(1024usize, 1024usize), (256, 4096)] {
                variants_case::<4>(n, k, trials, &[z("{0,-1}", dot_r4)]);
                variants_case::<8>(n, k, trials, &[z("{0,-1}", dot_r8)]);
                variants_case::<12>(n, k, trials, &[z("{0,-1}", dot_r12)]);
                variants_case::<15>(n, k, trials, &[z("{0,-1}", dot_r15)]);
                variants_case::<16>(
                    n,
                    k,
                    trials,
                    &[
                        z("{0,-1} row-major (production)", dot_r16),
                        z("{0,-1} lane-major", dot_lm_r16),
                        z("intrinsics", dot_intrinsics::<16>),
                        Variant { name: "tile", f: dot_tile_r16, ident: Ident::Z, tile: true },
                    ],
                );
            }
        }
        if what == "forms" || what == "all" {
            use v_sdot::kernel::asm::*;
            variants_case::<15>(
                1024,
                1024,
                trials,
                &[
                    Variant { name: "{0,-1}", f: dot_r15, ident: Ident::Z, tile: false },
                    Variant { name: "±1", f: dot_pm_r15, ident: Ident::Pm, tile: false },
                    Variant { name: "{0,1}", f: dot_b01_r15, ident: Ident::B01, tile: false },
                    Variant { name: "tile", f: dot_tile_r15, ident: Ident::Z, tile: true },
                ],
            );
        }
        if what == "gemm" || what == "all" {
            for threads in [1, 4, 8] {
                gemm_cases(threads, 200, trials.min(7), "auto");
            }
            gemm_cases(8, 200, trials.min(7), "groups");
        }
    }
}
