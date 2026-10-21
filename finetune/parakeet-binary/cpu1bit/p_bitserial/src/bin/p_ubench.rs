//! P's dev-tuning microbenchmarks. Every number this prints is "dev-tuning, background QoS, not
//! comparable": it runs under the Mac job supervisor's background band (PRIO_DARWIN_BG, which
//! confines threads to the E-cores) on synthetic dev data, never in the timed screening.
//!
//!   p-ubench isa                      # per-instruction throughput in ops/cycle (+ clock probe)
//!   p-ubench kernel [T] [reps] [threads] [points...]
//!                                     # per model shape and tuning point: pack and GEMM times
//!
//! A tuning point is `udot:4:4,2,1` (accum:o:frames for C = 1,2,3). Times are the thread's own CPU
//! time (single-thread runs), clocks are bracketed around every measurement.

use std::time::Duration;

use p_bitserial::{Accum, BitSerial, Split, Tuning};
use shared::kernel::{Call, Kernel, KernelConfig, Out, Slot};
use shared::pool::{Qos, ThreadPool};
use shared::rng::SplitMix64;
use shared::{BinaryMatrix, QuantizedActs, WeightsRef};

const LABEL: &str = "dev-tuning, background QoS, not comparable";

/// The calling thread's own CPU time (CLOCK_THREAD_CPUTIME_ID). Under the background band the
/// thread shares two E-cores with other work, so wall time includes time it was not running;
/// CPU time does not (DVFS is caught by the clock brackets).
#[derive(Clone, Copy)]
struct CpuClock(Duration);

impl CpuClock {
    fn now() -> CpuClock {
        let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        let rc = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
        assert_eq!(rc, 0, "clock_gettime(CLOCK_THREAD_CPUTIME_ID)");
        CpuClock(Duration::new(ts.tv_sec as u64, ts.tv_nsec as u32))
    }
    fn elapsed(self) -> Duration {
        CpuClock::now().0.saturating_sub(self.0)
    }
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // `--qos-ui`: run at QOS_CLASS_USER_INTERACTIVE (the overnight one-off stage; read back and
    // printed). Without it everything inherits the job supervisor's background band.
    let ui = args.iter().any(|a| a == "--qos-ui");
    args.retain(|a| a != "--qos-ui");
    let qos = if ui { Qos::UserInteractive } else { Qos::Inherit };
    match args.first().map(String::as_str) {
        Some("isa") => isa(qos),
        Some("kernel") => kernel(&args[1..], qos),
        _ => {
            eprintln!("usage: p-ubench [--qos-ui] isa | kernel [T] [reps] [threads] [accum:o:f1,f2,f3[:split] ...]");
            std::process::exit(2);
        }
    }
}

fn label(qos: Qos) -> &'static str {
    match qos {
        Qos::UserInteractive => "USER_INTERACTIVE one-off stage, non-comparison",
        Qos::Inherit => LABEL,
    }
}

fn qos_line() -> String {
    let (got, rel) = shared::pool::read_qos();
    format!("main-thread qos class {got:#x} rel {rel}")
}

// ---------------------------------------------------------------------------------------------
// Instruction throughput.

#[cfg(target_arch = "aarch64")]
mod probes {
    use std::arch::asm;
    use std::time::Duration;

    /// Runs `$body` (a list of instruction strings) `iters` times; `x9` points at a 4 KiB buffer.
    macro_rules! probe {
        ($iters:expr, $buf:expr, $($ins:literal),+ $(,)?) => {{
            let n: u64 = $iters;
            let p: *const u8 = $buf;
            let t0 = super::CpuClock::now();
            unsafe {
                asm!(
                    "2:",
                    $($ins,)+
                    "subs {n}, {n}, #1",
                    "b.ne 2b",
                    n = inout(reg) n => _,
                    in("x9") p,
                    out("x10") _, out("x11") _,
                    out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                    out("v8") _, out("v9") _, out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _, out("v15") _,
                    out("v16") _, out("v17") _, out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                    out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _, out("v30") _, out("v31") _,
                    options(nostack, readonly),
                );
            }
            t0.elapsed()
        }};
    }

    pub struct Probe {
        pub name: &'static str,
        /// Instructions of the measured kind per iteration.
        pub ops: u64,
        pub run: fn(u64, *const u8) -> Duration,
    }

    pub fn clock(iters: u64, buf: *const u8) -> Duration {
        // A chain of dependent 1-cycle scalar adds: 16 per iteration.
        probe!(iters, buf,
            "add x10, x10, #1", "add x10, x10, #1", "add x10, x10, #1", "add x10, x10, #1",
            "add x10, x10, #1", "add x10, x10, #1", "add x10, x10, #1", "add x10, x10, #1",
            "add x10, x10, #1", "add x10, x10, #1", "add x10, x10, #1", "add x10, x10, #1",
            "add x10, x10, #1", "add x10, x10, #1", "add x10, x10, #1", "add x10, x10, #1")
    }

    macro_rules! tp16 {
        ($fname:ident, $a:literal, $b:literal, $c:literal, $d:literal, $e:literal, $f:literal, $g:literal, $h:literal,
         $i:literal, $j:literal, $k:literal, $l:literal, $m:literal, $n:literal, $o:literal, $p:literal) => {
            pub fn $fname(iters: u64, buf: *const u8) -> Duration {
                probe!(iters, buf, $a, $b, $c, $d, $e, $f, $g, $h, $i, $j, $k, $l, $m, $n, $o, $p)
            }
        };
    }

    tp16!(eor, "eor v0.16b, v0.16b, v16.16b", "eor v1.16b, v1.16b, v16.16b", "eor v2.16b, v2.16b, v16.16b", "eor v3.16b, v3.16b, v16.16b",
        "eor v4.16b, v4.16b, v16.16b", "eor v5.16b, v5.16b, v16.16b", "eor v6.16b, v6.16b, v16.16b", "eor v7.16b, v7.16b, v16.16b",
        "eor v8.16b, v8.16b, v16.16b", "eor v9.16b, v9.16b, v16.16b", "eor v10.16b, v10.16b, v16.16b", "eor v11.16b, v11.16b, v16.16b",
        "eor v12.16b, v12.16b, v16.16b", "eor v13.16b, v13.16b, v16.16b", "eor v14.16b, v14.16b, v16.16b", "eor v15.16b, v15.16b, v16.16b");
    tp16!(cnt, "cnt v0.16b, v16.16b", "cnt v1.16b, v16.16b", "cnt v2.16b, v16.16b", "cnt v3.16b, v16.16b",
        "cnt v4.16b, v16.16b", "cnt v5.16b, v16.16b", "cnt v6.16b, v16.16b", "cnt v7.16b, v16.16b",
        "cnt v8.16b, v16.16b", "cnt v9.16b, v16.16b", "cnt v10.16b, v16.16b", "cnt v11.16b, v16.16b",
        "cnt v12.16b, v16.16b", "cnt v13.16b, v16.16b", "cnt v14.16b, v16.16b", "cnt v15.16b, v16.16b");
    tp16!(add, "add v0.16b, v0.16b, v16.16b", "add v1.16b, v1.16b, v16.16b", "add v2.16b, v2.16b, v16.16b", "add v3.16b, v3.16b, v16.16b",
        "add v4.16b, v4.16b, v16.16b", "add v5.16b, v5.16b, v16.16b", "add v6.16b, v6.16b, v16.16b", "add v7.16b, v7.16b, v16.16b",
        "add v8.16b, v8.16b, v16.16b", "add v9.16b, v9.16b, v16.16b", "add v10.16b, v10.16b, v16.16b", "add v11.16b, v11.16b, v16.16b",
        "add v12.16b, v12.16b, v16.16b", "add v13.16b, v13.16b, v16.16b", "add v14.16b, v14.16b, v16.16b", "add v15.16b, v15.16b, v16.16b");
    tp16!(uadalp, "uadalp v0.8h, v16.16b", "uadalp v1.8h, v16.16b", "uadalp v2.8h, v16.16b", "uadalp v3.8h, v16.16b",
        "uadalp v4.8h, v16.16b", "uadalp v5.8h, v16.16b", "uadalp v6.8h, v16.16b", "uadalp v7.8h, v16.16b",
        "uadalp v8.8h, v16.16b", "uadalp v9.8h, v16.16b", "uadalp v10.8h, v16.16b", "uadalp v11.8h, v16.16b",
        "uadalp v12.8h, v16.16b", "uadalp v13.8h, v16.16b", "uadalp v14.8h, v16.16b", "uadalp v15.8h, v16.16b");
    tp16!(udot_lane, "udot v0.4s, v16.16b, v17.4b[1]", "udot v1.4s, v16.16b, v17.4b[1]", "udot v2.4s, v16.16b, v17.4b[1]", "udot v3.4s, v16.16b, v17.4b[1]",
        "udot v4.4s, v16.16b, v17.4b[1]", "udot v5.4s, v16.16b, v17.4b[1]", "udot v6.4s, v16.16b, v17.4b[1]", "udot v7.4s, v16.16b, v17.4b[1]",
        "udot v8.4s, v16.16b, v17.4b[1]", "udot v9.4s, v16.16b, v17.4b[1]", "udot v10.4s, v16.16b, v17.4b[1]", "udot v11.4s, v16.16b, v17.4b[1]",
        "udot v12.4s, v16.16b, v17.4b[1]", "udot v13.4s, v16.16b, v17.4b[1]", "udot v14.4s, v16.16b, v17.4b[1]", "udot v15.4s, v16.16b, v17.4b[1]");
    tp16!(sdot, "sdot v0.4s, v16.16b, v17.16b", "sdot v1.4s, v16.16b, v17.16b", "sdot v2.4s, v16.16b, v17.16b", "sdot v3.4s, v16.16b, v17.16b",
        "sdot v4.4s, v16.16b, v17.16b", "sdot v5.4s, v16.16b, v17.16b", "sdot v6.4s, v16.16b, v17.16b", "sdot v7.4s, v16.16b, v17.16b",
        "sdot v8.4s, v16.16b, v17.16b", "sdot v9.4s, v16.16b, v17.16b", "sdot v10.4s, v16.16b, v17.16b", "sdot v11.4s, v16.16b, v17.16b",
        "sdot v12.4s, v16.16b, v17.16b", "sdot v13.4s, v16.16b, v17.16b", "sdot v14.4s, v16.16b, v17.16b", "sdot v15.4s, v16.16b, v17.16b");
    tp16!(ldr_q, "ldr q0, [x9]", "ldr q1, [x9, #16]", "ldr q2, [x9, #32]", "ldr q3, [x9, #48]",
        "ldr q4, [x9, #64]", "ldr q5, [x9, #80]", "ldr q6, [x9, #96]", "ldr q7, [x9, #112]",
        "ldr q8, [x9, #128]", "ldr q9, [x9, #144]", "ldr q10, [x9, #160]", "ldr q11, [x9, #176]",
        "ldr q12, [x9, #192]", "ldr q13, [x9, #208]", "ldr q14, [x9, #224]", "ldr q15, [x9, #240]");
    tp16!(ld1r, "ld1r {{v0.8h}}, [x9]", "ld1r {{v1.8h}}, [x9]", "ld1r {{v2.8h}}, [x9]", "ld1r {{v3.8h}}, [x9]",
        "ld1r {{v4.8h}}, [x9]", "ld1r {{v5.8h}}, [x9]", "ld1r {{v6.8h}}, [x9]", "ld1r {{v7.8h}}, [x9]",
        "ld1r {{v8.8h}}, [x9]", "ld1r {{v9.8h}}, [x9]", "ld1r {{v10.8h}}, [x9]", "ld1r {{v11.8h}}, [x9]",
        "ld1r {{v12.8h}}, [x9]", "ld1r {{v13.8h}}, [x9]", "ld1r {{v14.8h}}, [x9]", "ld1r {{v15.8h}}, [x9]");
    // 12 EOR + 4 LD1R: does LD1R take a SIMD slot (Firestorm: 2 uops, one on a SIMD unit)?
    tp16!(eor12_ld1r4, "eor v0.16b, v0.16b, v16.16b", "eor v1.16b, v1.16b, v16.16b", "eor v2.16b, v2.16b, v16.16b", "ld1r {{v20.8h}}, [x9]",
        "eor v3.16b, v3.16b, v16.16b", "eor v4.16b, v4.16b, v16.16b", "eor v5.16b, v5.16b, v16.16b", "ld1r {{v21.8h}}, [x9]",
        "eor v6.16b, v6.16b, v16.16b", "eor v7.16b, v7.16b, v16.16b", "eor v8.16b, v8.16b, v16.16b", "ld1r {{v22.8h}}, [x9]",
        "eor v9.16b, v9.16b, v16.16b", "eor v10.16b, v10.16b, v16.16b", "eor v11.16b, v11.16b, v16.16b", "ld1r {{v23.8h}}, [x9]");
    tp16!(eor12_ldr4, "eor v0.16b, v0.16b, v16.16b", "eor v1.16b, v1.16b, v16.16b", "eor v2.16b, v2.16b, v16.16b", "ldr q20, [x9]",
        "eor v3.16b, v3.16b, v16.16b", "eor v4.16b, v4.16b, v16.16b", "eor v5.16b, v5.16b, v16.16b", "ldr q21, [x9, #16]",
        "eor v6.16b, v6.16b, v16.16b", "eor v7.16b, v7.16b, v16.16b", "eor v8.16b, v8.16b, v16.16b", "ldr q22, [x9, #32]",
        "eor v9.16b, v9.16b, v16.16b", "eor v10.16b, v10.16b, v16.16b", "eor v11.16b, v11.16b, v16.16b", "ldr q23, [x9, #48]");

    /// The `Udot` inner triple, 5 independent accumulators (v0..v4), 15 + 1 instructions.
    pub fn triple_udot(iters: u64, buf: *const u8) -> Duration {
        probe!(iters, buf,
            "eor v8.16b, v16.16b, v20.16b", "cnt v8.16b, v8.16b", "udot v0.4s, v8.16b, v18.4b[0]",
            "eor v9.16b, v16.16b, v21.16b", "cnt v9.16b, v9.16b", "udot v1.4s, v9.16b, v18.4b[1]",
            "eor v10.16b, v16.16b, v22.16b", "cnt v10.16b, v10.16b", "udot v2.4s, v10.16b, v18.4b[2]",
            "eor v11.16b, v16.16b, v23.16b", "cnt v11.16b, v11.16b", "udot v3.4s, v11.16b, v18.4b[3]",
            "eor v12.16b, v17.16b, v20.16b", "cnt v12.16b, v12.16b", "udot v4.4s, v12.16b, v19.4b[0]",
            "nop")
    }
    /// The default kernel's triple as LLVM emits it: vector UDOT against a splat constant.
    pub fn triple_udot_vec(iters: u64, buf: *const u8) -> Duration {
        probe!(iters, buf,
            "eor v8.16b, v16.16b, v20.16b", "cnt v8.16b, v8.16b", "udot v0.4s, v8.16b, v18.16b",
            "eor v9.16b, v16.16b, v21.16b", "cnt v9.16b, v9.16b", "udot v1.4s, v9.16b, v18.16b",
            "eor v10.16b, v16.16b, v22.16b", "cnt v10.16b, v10.16b", "udot v2.4s, v10.16b, v18.16b",
            "eor v11.16b, v16.16b, v23.16b", "cnt v11.16b, v11.16b", "udot v3.4s, v11.16b, v18.16b",
            "eor v12.16b, v17.16b, v20.16b", "cnt v12.16b, v12.16b", "udot v4.4s, v12.16b, v19.16b",
            "nop")
    }
    /// The `Uadalp` inner triple, 5 independent accumulators.
    pub fn triple_uadalp(iters: u64, buf: *const u8) -> Duration {
        probe!(iters, buf,
            "eor v8.16b, v16.16b, v20.16b", "cnt v8.16b, v8.16b", "uadalp v0.8h, v8.16b",
            "eor v9.16b, v16.16b, v21.16b", "cnt v9.16b, v9.16b", "uadalp v1.8h, v9.16b",
            "eor v10.16b, v16.16b, v22.16b", "cnt v10.16b, v10.16b", "uadalp v2.8h, v10.16b",
            "eor v11.16b, v16.16b, v23.16b", "cnt v11.16b, v11.16b", "uadalp v3.8h, v11.16b",
            "eor v12.16b, v17.16b, v20.16b", "cnt v12.16b, v12.16b", "uadalp v4.8h, v12.16b",
            "nop")
    }

    pub fn all() -> Vec<Probe> {
        vec![
            Probe { name: "eor.16b", ops: 16, run: eor },
            Probe { name: "cnt.16b", ops: 16, run: cnt },
            Probe { name: "add.16b", ops: 16, run: add },
            Probe { name: "uadalp.8h", ops: 16, run: uadalp },
            Probe { name: "udot.4s by element", ops: 16, run: udot_lane },
            Probe { name: "sdot.4s", ops: 16, run: sdot },
            Probe { name: "ldr q (L1)", ops: 16, run: ldr_q },
            Probe { name: "ld1r .8h (L1)", ops: 16, run: ld1r },
            Probe { name: "eor (12) + ld1r (4): eor per cycle", ops: 12, run: eor12_ld1r4 },
            Probe { name: "eor (12) + ldr q (4): eor per cycle", ops: 12, run: eor12_ldr4 },
            Probe { name: "triple eor+cnt+udot (by element): triples/cycle", ops: 5, run: triple_udot },
            Probe { name: "triple eor+cnt+udot (vector): triples/cycle", ops: 5, run: triple_udot_vec },
            Probe { name: "triple eor+cnt+uadalp: triples per cycle", ops: 5, run: triple_uadalp },
        ]
    }
}

/// Core clock from a dependent chain of 1-cycle scalar adds (~25 ms), or 0 off aarch64.
fn clock_hz() -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        let buf = [0u8; 64];
        let iters = 1_500_000u64;
        let d = (0..3).map(|_| probes::clock(iters, buf.as_ptr())).min().unwrap();
        (iters * 16) as f64 / d.as_secs_f64()
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        0.0
    }
}

/// `(mean GHz, drift flag)` from clocks measured right before and after a measurement.
fn bracket(h0: f64, h1: f64) -> (f64, &'static str) {
    let mean = 0.5 * (h0 + h1);
    (mean / 1e9, if (h0 - h1).abs() > 0.03 * mean { " CLOCK-DRIFT" } else { "" })
}

#[cfg(target_arch = "aarch64")]
fn isa(qos: Qos) {
    let _pool = ThreadPool::new(1, qos); // sets (and reads back) the main thread's QoS
    println!("# p-ubench isa ({}); {}", label(qos), qos_line());
    println!("# per-cycle rates use the mean of clocks measured just before and after each probe");
    let buf = vec![0u8; 4096];
    let iters = 4_000_000u64;
    for p in probes::all() {
        let h0 = clock_hz();
        let d = (0..3).map(|_| (p.run)(iters, buf.as_ptr())).min().unwrap();
        let h1 = clock_hz();
        let (ghz, flag) = bracket(h0, h1);
        let per_cycle = (iters * p.ops) as f64 / (d.as_secs_f64() * ghz * 1e9);
        println!("{:<44} {:6.3} per cycle  (clock {:.3} -> {:.3} GHz){flag}", p.name, per_cycle, h0 / 1e9, h1 / 1e9);
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn isa(_qos: Qos) {
    eprintln!("isa probes exist on aarch64 only");
}

// ---------------------------------------------------------------------------------------------
// Kernel timings.

/// `accum:o:f1,f2,f3[:split]`, e.g. `udot:4:4,2,1` or `udot:4:2,1,1:frames`.
fn parse_point(s: &str) -> Tuning {
    let parts: Vec<&str> = s.split(':').collect();
    assert!(parts.len() == 3 || parts.len() == 4, "point {s}: accum:o:f1,f2,f3[:split]");
    let accum = match parts[0] {
        "udot" => Accum::Udot,
        "udot_lane" => Accum::UdotLane,
        "uadalp" => Accum::Uadalp,
        a => panic!("accum {a}"),
    };
    let split = match parts.get(3).copied().unwrap_or("outputs") {
        "outputs" => Split::Outputs,
        "frames" => Split::Frames,
        x => panic!("split {x}"),
    };
    let frames: Vec<usize> = parts[2].split(',').map(|x| x.parse().unwrap()).collect();
    let t = Tuning { accum, o: parts[1].parse().unwrap(), frames: [frames[0], frames[1], frames[2]], split };
    t.validate().unwrap();
    t
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn kernel(args: &[String], qos: Qos) {
    let t: usize = args.first().map(|s| s.parse().unwrap()).unwrap_or(200);
    let reps: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(7);
    let threads: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1);
    let points: Vec<Tuning> = if args.len() > 3 {
        args[3..].iter().map(|s| parse_point(s)).collect()
    } else {
        p_bitserial::tuning_grid() // the frozen grid, default first
    };
    let pool = ThreadPool::new(threads, qos);
    println!("# p-ubench kernel ({}); T={t} reps={reps} threads={threads}; {}", label(qos), qos_line());
    println!("# shape NxK, C, point, pack_us, gemm_us (reuse), gmac_per_s, ns_per_16mac_per_thread");
    let shapes = shared::schedule::Schedule::load().shapes();
    for c in [2usize, 3] {
        for &(n, k) in &shapes {
            let m = BinaryMatrix::random(n, k, (n + 7 * k) as u64);
            let mut rng = SplitMix64::new(42);
            let x: Vec<f32> = (0..t * k).map(|i| rng.next_gaussian() * if i % 37 == 0 { 20.0 } else { 1.0 }).collect();
            let mut qa = QuantizedActs::with_capacity(t, k, c);
            qa.quantize(&pool, &x, t).unwrap();
            let mut y = vec![0f32; t * n];
            if threads == 1 {
                let h0 = clock_hz();
                let d = sdot_ceiling(n, k, t * c, reps);
                let (ghz, flag) = bracket(h0, clock_hz());
                let ns16 = d.as_secs_f64() * 1e9 / ((n * k * t * c) as f64 / 16.0);
                println!("{n}x{k} C={c} sdot-ceiling(pre-expanded s8, 4x4, dots only) gemm_us={:.0} ns_per_16mac={ns16:.4} \
                          ghz={ghz:.3} cycles_per_sdot={:.3}{flag}", d.as_secs_f64() * 1e6, ns16 * ghz);
            }
            for &pt in &points {
                let cfg = KernelConfig { threads, components: c, max_rows: t, max_k: k, max_n: n, tuning: pt.to_json() };
                let mut kern = BitSerial::new(&cfg);
                let p = kern.prepare(WeightsRef::Binary(&m));
                let fresh = Call { slot: Slot::Model, reuse: false };
                let reuse = Call { slot: Slot::Model, reuse: true };
                kern.gemm(&pool, &p, &qa, fresh, Out::F32(&mut y)); // warm, allocate
                let mut full = Vec::new();
                let mut gemm = Vec::new();
                let h0 = clock_hz();
                for _ in 0..reps {
                    let t0 = CpuClock::now();
                    kern.gemm(&pool, &p, &qa, fresh, Out::F32(&mut y));
                    full.push(t0.elapsed());
                    let t0 = CpuClock::now();
                    kern.gemm(&pool, &p, &qa, reuse, Out::F32(&mut y));
                    gemm.push(t0.elapsed());
                }
                let h1 = clock_hz();
                let (ghz, flag) = bracket(h0, h1);
                let (full, gemm) = (median(full), median(gemm));
                let pack = full.saturating_sub(gemm);
                let macs = (n * k * t * c) as f64;
                let gmacs = macs / gemm.as_secs_f64() / 1e9;
                let ns16 = gemm.as_secs_f64() * 1e9 * threads as f64 / (macs / 16.0);
                // One (EOR, CNT, accumulate) triple = 16 MACs: cycles per triple per thread.
                println!("{n}x{k} C={c} {}:{}:{:?}:{} pack_us={:.0} gemm_us={:.0} gmac_per_s={gmacs:.2} ns_per_16mac={ns16:.4} \
                          ghz={ghz:.3} cycles_per_triple={:.3}{flag}",
                         pt.accum.name(), pt.o, pt.frames, pt.split.name(), pack.as_secs_f64() * 1e6, gemm.as_secs_f64() * 1e6, ns16 * ghz);
                std::hint::black_box(&y);
            }
        }
    }
}

/// Same-core reference: a register-blocked SDOT GEMM on pre-expanded s8 weights (no expansion,
/// no epilogue; 4 outputs x 4 frame-parts, K along the lanes, 16 positions per step: one SDOT
/// per 16 MACs). Raw integer dots only. Returns the median time of one GEMM over `m` frame-parts.
#[cfg(target_arch = "aarch64")]
fn sdot_ceiling(n: usize, k: usize, m: usize, reps: usize) -> Duration {
    use std::arch::aarch64::*;
    assert!(n % 4 == 0 && k % 16 == 0);
    let mt = m.div_ceil(4);
    let mut rng = SplitMix64::new(9);
    let w: Vec<i8> = (0..n * k).map(|_| if rng.next_u64() & 1 == 1 { 1 } else { -1 }).collect();
    let a: Vec<i8> = (0..mt * 4 * k).map(|_| rng.next_u64() as i8).collect();
    let mut out = vec![0i32; mt * 4 * n];
    let steps = k / 16;
    let mut times = Vec::new();
    for _ in 0..reps + 1 {
        let t0 = CpuClock::now();
        for ft in 0..mt {
            for ot in 0..n / 4 {
                unsafe {
                    let wp = w.as_ptr().add(ot * 4 * k);
                    let ap = a.as_ptr().add(ft * 4 * k);
                    let mut acc = [[vdupq_n_s32(0); 4]; 4];
                    for s in 0..steps {
                        let wv: [int8x16_t; 4] = std::array::from_fn(|o| vld1q_s8(wp.add((s * 4 + o) * 16)));
                        for f in 0..4 {
                            let av = vld1q_s8(ap.add((s * 4 + f) * 16));
                            for o in 0..4 {
                                acc[o][f] = vdotq_s32(acc[o][f], wv[o], av);
                            }
                        }
                    }
                    for f in 0..4 {
                        let s01 = vpaddq_s32(acc[0][f], acc[1][f]);
                        let s23 = vpaddq_s32(acc[2][f], acc[3][f]);
                        vst1q_s32(out.as_mut_ptr().add((ft * 4 + f) * n + ot * 4), vpaddq_s32(s01, s23));
                    }
                }
            }
        }
        times.push(t0.elapsed());
        std::hint::black_box(&out);
    }
    times.remove(0);
    median(times)
}

#[cfg(not(target_arch = "aarch64"))]
fn sdot_ceiling(_n: usize, _k: usize, _m: usize, _reps: usize) -> Duration {
    Duration::ZERO
}
