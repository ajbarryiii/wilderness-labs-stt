//! The per-arm replay binary (one fresh process per block, holding one arm; plans/cpu-benchmark-mac.md
//! "Process and threads"). Each kernel crate adds `src/bin/replay.rs`:
//!
//! ```ignore
//! shared::bench_main!(my_crate::MyKernel);
//! ```
//!
//! Subcommands (`--key value` arguments):
//! - `describe`: kernel description, build provenance.
//! - `exactness --export DIR --captures GROUP_DIR --reference REF_DIR [--clips 0,3] [--components 2,3]
//!   [--threads 1,4,8] [--tuning JSON] [--out FILE]`: replays each clip in digest mode and compares
//!   every call's dots and FP32 output digests with the oracle's (shared `oracle-digests` binary),
//!   plus a trace check of the first replay against the schedule; exit 1 on any mismatch.
//! - `block --export DIR --captures GROUP_DIR --clips ORDER --threads N --components C --replays N0,N1,..
//!   --guard-s G --settle-s S --settle-post-s S2 [--tuning JSON] [--block-id ID] [--allow-background 1]`:
//!   the timed protocol (plans/cpu-benchmark-mac.md "Block protocol"). JSON lines on stdout, timestamps
//!   from CLOCK_UPTIME_RAW in ns (the runner and powermetrics' reader use the same clock): start, qos
//!   (abort with exit 3 on a mismatch), loaded, warmup_done (longest clip twice, then one pass over all),
//!   storage (the audit after warmup; exit 4 on a breach), then per clip: load, one untimed replay,
//!   sleep S + G (workers park), the window of exactly N replays with t0 read just before the first and
//!   t1 after the last returns, sleep G + S2, and only then window_start / window_end; qos_end, end.
//! - `audit` (same weight and clip arguments, no timing): load, warmup and the storage audit only; exit 4
//!   on a breach. An arm must pass it before its freeze.
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};
use std::time::Duration;

use serde_json::{Value, json};

use crate::data::{Clip, group_clips, load_weights};
use crate::kernel::{Kernel, KernelConfig};
use crate::pool::{Qos, ThreadPool};
use crate::replay::{Mode, Replayer, digests_from_json};
use crate::schedule::Schedule;

/// Live / peak heap bytes (storage audit, plans/cpu-benchmark.md "Storage").
pub struct CountingAlloc;
pub static LIVE: AtomicIsize = AtomicIsize::new(0);
pub static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            let now = LIVE.fetch_add(l.size() as isize, Ordering::Relaxed) + l.size() as isize;
            PEAK.fetch_max(now.max(0) as usize, Ordering::Relaxed);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            let now = LIVE.fetch_add(l.size() as isize, Ordering::Relaxed) + l.size() as isize;
            PEAK.fetch_max(now.max(0) as usize, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        LIVE.fetch_sub(l.size() as isize, Ordering::Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, new) };
        if !q.is_null() {
            let now = LIVE.fetch_add(new as isize - l.size() as isize, Ordering::Relaxed) + new as isize - l.size() as isize;
            PEAK.fetch_max(now.max(0) as usize, Ordering::Relaxed);
        }
        q
    }
}

#[macro_export]
macro_rules! bench_main {
    ($k:ty) => {
        #[global_allocator]
        static CPU1BIT_ALLOC: $crate::bench::CountingAlloc = $crate::bench::CountingAlloc;
        fn main() {
            $crate::bench::main::<$k>();
        }
    };
}

/// Darwin `<time.h>`: `_CLOCK_UPTIME_RAW = 8` (mach_absolute_time in ns, stops in sleep; the block
/// holds caffeinate). Declared here because libc 0.2.177 lacks `clock_gettime_nsec_np`. The Mac
/// orchestrator reads the same clock (`time.clock_gettime_ns(time.CLOCK_UPTIME_RAW)`, also 8) for
/// powermetrics arrival times, which is what the M3/M10 alignment validation measures.
#[cfg(target_os = "macos")]
pub const CLOCK_UPTIME_RAW: u32 = 8;

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn clock_gettime_nsec_np(clock_id: u32) -> u64;
}

/// CLOCK_UPTIME_RAW (macOS) or CLOCK_MONOTONIC_RAW (Linux), in ns.
pub fn now_ns() -> u64 {
    #[cfg(target_os = "macos")]
    unsafe {
        clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC_RAW, &mut ts) };
        ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
    }
}

pub fn max_rss_bytes() -> u64 {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    if cfg!(target_os = "macos") { ru.ru_maxrss as u64 } else { ru.ru_maxrss as u64 * 1024 }
}

pub fn build_info() -> Value {
    json!({"rustc": crate::RUSTC, "rustflags": crate::RUSTFLAGS, "target": crate::TARGET,
           "schedule_content_sha256": crate::schedule::SCHEDULE_CONTENT_SHA256,
           "exe_sha256": std::env::current_exe().ok().and_then(|p| std::fs::read(p).ok()).map(|b| crate::data::sha256_hex(&b))})
}

pub struct Args {
    pub cmd: String,
    kv: HashMap<String, String>,
}

impl Args {
    pub fn parse() -> Args {
        let mut it = std::env::args().skip(1).peekable();
        let cmd = match it.peek() {
            Some(a) if !a.starts_with("--") => it.next().unwrap(),
            _ => "describe".into(),
        };
        Self::collect(cmd, it)
    }
    /// Flags only (binaries without subcommands).
    pub fn parse_from_flags() -> Args {
        Self::collect(String::new(), std::env::args().skip(1).peekable())
    }
    fn collect(cmd: String, mut it: impl Iterator<Item = String>) -> Args {
        let mut kv = HashMap::new();
        while let Some(k) = it.next() {
            let key = k.strip_prefix("--").unwrap_or_else(|| panic!("unexpected argument {k}")).to_string();
            let v = it.next().unwrap_or_else(|| panic!("--{key} needs a value"));
            kv.insert(key, v);
        }
        Args { cmd, kv }
    }
    pub fn get(&self, k: &str) -> Option<&str> {
        self.kv.get(k).map(String::as_str)
    }
    pub fn req(&self, k: &str) -> &str {
        self.get(k).unwrap_or_else(|| panic!("--{k} is required"))
    }
    pub fn list(&self, k: &str, default: &str) -> Vec<usize> {
        self.get(k).unwrap_or(default).split(',').filter(|s| !s.is_empty()).map(|s| s.parse().unwrap()).collect()
    }
    pub fn tuning(&self) -> Value {
        self.get("tuning").map(|s| serde_json::from_str(s).expect("--tuning JSON")).unwrap_or(Value::Null)
    }
    pub fn to_json(&self) -> Value {
        json!({"cmd": self.cmd, "args": self.kv})
    }
}

fn emit(v: Value) {
    let mut out = std::io::stdout().lock();
    writeln!(out, "{v}").unwrap();
    out.flush().unwrap();
}

pub fn main<K: Kernel>() {
    let args = Args::parse();
    match args.cmd.as_str() {
        "describe" => {
            let cfg = KernelConfig { threads: 1, components: 2, max_rows: 1, max_k: 4096, max_n: 4096, tuning: args.tuning() };
            emit(json!({"kernel": K::NAME, "format": format!("{:?}", K::FORMAT), "describe": K::new(&cfg).describe(), "build": build_info()}));
        }
        "exactness" => exactness::<K>(&args),
        "block" => block::<K>(&args),
        "audit" => audit::<K>(&args),
        other => panic!("unknown subcommand {other}"),
    }
}

fn clip_dirs(args: &Args) -> Vec<PathBuf> {
    let all = group_clips(Path::new(args.req("captures")));
    match args.get("clips") {
        None => all,
        Some(_) => args.list("clips", "").into_iter().map(|i| all[i].clone()).collect(),
    }
}

fn max_t(dirs: &[PathBuf]) -> usize {
    dirs.iter()
        .map(|d| {
            let idx: Value = serde_json::from_slice(&std::fs::read(d.join("index.json")).unwrap()).unwrap();
            idx["T"].as_u64().unwrap() as usize
        })
        .max()
        .unwrap()
}

fn exactness<K: Kernel>(args: &Args) {
    let sched = Schedule::load();
    let dirs = clip_dirs(args);
    let mt = max_t(&dirs);
    let reference = PathBuf::from(args.req("reference"));
    let mut results = Vec::new();
    let mut failures = 0usize;
    for &threads in &args.list("threads", "1,4,8") {
        let pool = ThreadPool::new(threads, if cfg!(target_os = "macos") { Qos::UserInteractive } else { Qos::Inherit });
        for &c_n in &args.list("components", "2,3") {
            let (mats, info) = load_weights(Path::new(args.req("export")), K::FORMAT, &sched).expect("weights");
            let cfg = KernelConfig { threads, components: c_n, max_rows: mt, max_k: 4096, max_n: 4096, tuning: args.tuning() };
            let mut rp = Replayer::<K>::new(&cfg, mats, &sched);
            for (ci, d) in dirs.iter().enumerate() {
                let clip = Clip::load(d, true);
                let name = d.file_name().unwrap().to_string_lossy().to_string();
                let refdoc: Value = serde_json::from_slice(
                    &std::fs::read(reference.join(format!("{name}.C{c_n}.json"))).expect("reference digests"),
                )
                .unwrap();
                assert_eq!(refdoc["clip_sha256"].as_str(), Some(clip.sha256.as_str()), "reference is for another capture");
                let want = digests_from_json(&refdoc["digests"]);
                if ci == 0 {
                    let mut ev = Vec::new();
                    rp.replay(&pool, &clip, Mode::Trace(&mut ev)).unwrap();
                    let calls: Vec<_> = sched.replayed().collect();
                    assert_eq!(ev.len(), calls.len());
                    for (e, c) in ev.iter().zip(&calls) {
                        assert!(e.index == c.index && e.n == c.n && e.k == c.k && e.reuse == c.reuse && e.rows == clip.t);
                    }
                }
                let mut got = Vec::new();
                rp.replay(&pool, &clip, Mode::Digest(&mut got)).unwrap();
                let bad: Vec<usize> = got
                    .iter()
                    .zip(&want)
                    .filter(|(g, w)| g.index != w.index || g.y_sha256 != w.y_sha256 || g.d_sha256 != w.d_sha256)
                    .map(|(g, _)| g.index)
                    .collect();
                failures += bad.len() + (got.len() != want.len()) as usize;
                let rec = json!({"threads": threads, "components": c_n, "clip": name, "id": clip.id(), "T": clip.t,
                                 "calls": got.len(), "mismatched_calls": bad, "export": info.to_json(),
                                 "reference_export": refdoc["export"]});
                emit(rec.clone());
                results.push(rec);
            }
        }
    }
    let doc = json!({"kernel": K::NAME, "describe": Value::Null, "build": build_info(), "args": args.to_json(),
                     "results": results, "failures": failures, "ok": failures == 0});
    if let Some(out) = args.get("out") {
        std::fs::write(out, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    }
    emit(json!({"event": "exactness_done", "ok": failures == 0, "failures": failures}));
    if failures != 0 {
        std::process::exit(1);
    }
}

/// The packed 1-bit size of the 240 replayed matrices (plans/cpu-benchmark-mac.md "Storage and
/// workspace"; the inherited 75,497,472 bytes counted all 264 projections, linear_pos is not replayed).
pub const PACKED_1BIT_REPLAYED_BYTES: usize = 72_351_744;
/// floor(1.10 x PACKED_1BIT_REPLAYED_BYTES): the weight cap of arms that load the 1-bit export.
pub const WEIGHT_CAP_BYTES: usize = 79_586_918;
/// Declared per-call workspace, every arm, at 8 workers.
pub const WORKSPACE_CAP_BYTES: usize = 32 << 20;
/// Per-row values (scales, row sums) excluded from the cap, per output row.
pub const PER_ROW_MAX_BYTES: usize = 16;

/// Live bytes of the counting allocator (0 unless it is the global allocator).
pub fn heap_live() -> usize {
    LIVE.load(Ordering::Relaxed).max(0) as usize
}

/// The storage audit of one prepared, warmed-up replayer.
#[derive(Clone, Debug)]
pub struct StorageAudit {
    /// Live heap allocated since before loading the weights.
    pub heap_live_bytes: usize,
    /// The replayer's own buffers (Replayer::harness_bytes).
    pub harness_bytes: usize,
    pub workspace_bytes: usize,
    pub per_row_bytes: usize,
    /// What the kernel reports holding as codes.
    pub declared_code_bytes: usize,
    /// heap_live - harness - workspace - per_row: everything else counts as weight bytes.
    pub weight_bytes: usize,
    /// The arm loads the 1-bit export, so the weight cap applies.
    pub capped: bool,
    pub ok: bool,
    pub reasons: Vec<String>,
}

impl StorageAudit {
    pub fn to_json(&self) -> Value {
        json!({"heap_live_bytes": self.heap_live_bytes, "harness_bytes": self.harness_bytes,
               "workspace_bytes": self.workspace_bytes, "workspace_cap_bytes": WORKSPACE_CAP_BYTES,
               "per_row_bytes": self.per_row_bytes, "declared_code_bytes": self.declared_code_bytes,
               "weight_bytes": self.weight_bytes, "cap_bytes": if self.capped { Some(WEIGHT_CAP_BYTES) } else { None },
               "capped": self.capped, "ok": self.ok, "reasons": self.reasons})
    }
}

/// Audits a replayer given the live heap allocated since before its weights were loaded.
pub fn storage_audit<K: Kernel>(rp: &Replayer<K>, heap_live_since_load: usize) -> StorageAudit {
    let (codes, per_row) = rp
        .prepared
        .iter()
        .map(|p| rp.kernel.weight_bytes(p))
        .fold((0usize, 0usize), |a, w| (a.0 + w.codes, a.1 + w.per_row));
    let rows: usize = rp.calls.iter().map(|c| c.n).sum();
    let workspace = rp.kernel.workspace_bytes();
    let harness = rp.harness_bytes();
    let weight = heap_live_since_load.saturating_sub(harness.saturating_add(workspace).saturating_add(per_row));
    let capped = K::FORMAT == crate::weights::WeightFormat::Binary;
    let mut reasons = Vec::new();
    if capped && weight > WEIGHT_CAP_BYTES {
        reasons.push(format!("weight bytes {weight} above the cap {WEIGHT_CAP_BYTES}"));
    }
    if workspace > WORKSPACE_CAP_BYTES {
        reasons.push(format!("workspace {workspace} above the cap {WORKSPACE_CAP_BYTES}"));
    }
    if per_row > PER_ROW_MAX_BYTES * rows {
        reasons.push(format!("per-row bytes {per_row} above {PER_ROW_MAX_BYTES} per output row ({rows} rows)"));
    }
    StorageAudit {
        heap_live_bytes: heap_live_since_load,
        harness_bytes: harness,
        workspace_bytes: workspace,
        per_row_bytes: per_row,
        declared_code_bytes: codes,
        weight_bytes: weight,
        capped,
        ok: reasons.is_empty(),
        reasons,
    }
}

fn secs_ns(args: &Args, key: &str) -> u64 {
    let v: f64 = args.req(key).parse().unwrap_or_else(|_| panic!("--{key} must be seconds"));
    assert!(v.is_finite() && v >= 0.0, "--{key} must be a non-negative number of seconds");
    (v * 1e9).round() as u64
}

/// Loads the weights, prepares them and warms up (the longest clip twice, then one pass over all clips,
/// one clip resident at a time); emits `loaded`, `warmup_done` and `storage`. Exits 4 on a breach.
fn load_warm_audit<K: Kernel>(args: &Args, sched: &Schedule, dirs: &[PathBuf], pool: &ThreadPool, threads: usize, c_n: usize) -> Replayer<K> {
    let live0 = heap_live();
    let (mats, info) = load_weights(Path::new(args.req("export")), K::FORMAT, sched).expect("weights");
    let cfg = KernelConfig { threads, components: c_n, max_rows: max_t(dirs), max_k: 4096, max_n: 4096, tuning: args.tuning() };
    let mut rp = Replayer::<K>::new(&cfg, mats, sched);
    let wb = rp.prepared.iter().map(|p| rp.kernel.weight_bytes(p)).fold((0usize, 0usize), |a, w| (a.0 + w.codes, a.1 + w.per_row));
    emit(json!({"event": "loaded", "t_ns": now_ns(), "export": info.to_json(), "weight_code_bytes": wb.0, "weight_per_row_bytes": wb.1,
                "workspace_bytes": rp.kernel.workspace_bytes(), "heap_live_bytes": heap_live().saturating_sub(live0),
                "heap_peak_bytes": PEAK.load(Ordering::Relaxed), "max_rss_bytes": max_rss_bytes(), "describe": rp.kernel.describe()}));
    let longest = dirs.iter().enumerate().max_by_key(|(i, d)| (max_t(std::slice::from_ref(*d)), usize::MAX - i)).unwrap().1.clone();
    {
        let clip = Clip::load(&longest, true);
        for _ in 0..2 {
            rp.replay(pool, &clip, Mode::Timed).unwrap();
        }
    }
    for d in dirs {
        let clip = Clip::load(d, true);
        rp.replay(pool, &clip, Mode::Timed).unwrap();
    }
    emit(json!({"event": "warmup_done", "t_ns": now_ns(), "workspace_bytes": rp.kernel.workspace_bytes()}));
    let audit = storage_audit(&rp, heap_live().saturating_sub(live0));
    let mut ev = audit.to_json();
    ev["event"] = json!("storage");
    ev["t_ns"] = json!(now_ns());
    emit(ev);
    if !audit.ok {
        std::process::exit(4);
    }
    drop(info);
    rp
}

fn audit<K: Kernel>(args: &Args) {
    let sched = Schedule::load();
    let dirs = clip_dirs(args);
    let threads: usize = args.req("threads").parse().unwrap();
    let c_n: usize = args.req("components").parse().unwrap();
    emit(json!({"event": "start", "t_ns": now_ns(), "pid": std::process::id(), "kernel": K::NAME,
                "build": build_info(), "args": args.to_json()}));
    let pool = ThreadPool::new(threads, if cfg!(target_os = "macos") { Qos::UserInteractive } else { Qos::Inherit });
    let _rp = load_warm_audit::<K>(args, &sched, &dirs, &pool, threads, c_n);
    emit(json!({"event": "end", "t_ns": now_ns()}));
}

fn block<K: Kernel>(args: &Args) {
    let t_start = now_ns();
    let sched = Schedule::load();
    let dirs = clip_dirs(args);
    let threads: usize = args.req("threads").parse().unwrap();
    let c_n: usize = args.req("components").parse().unwrap();
    let replays = args.list("replays", "");
    assert_eq!(replays.len(), dirs.len(), "--replays needs one fixed N per clip, in clip order");
    assert!(replays.iter().all(|&n| n >= 1), "--replays: every N must be >= 1");
    let (g, s_pre, s_post) = (secs_ns(args, "guard-s"), secs_ns(args, "settle-s"), secs_ns(args, "settle-post-s"));
    emit(json!({"event": "start", "t_ns": t_start, "pid": std::process::id(), "kernel": K::NAME,
                "block_id": args.get("block-id"), "build": build_info(), "args": args.to_json()}));
    let pool = ThreadPool::new(threads, Qos::UserInteractive);
    let qos: Vec<Value> = pool.qos_report().iter().map(|r| r.to_json()).collect();
    // Timed blocks need the effective class: set, read back, and no darwin-background clamp.
    // `--allow-background 1` (runner plumbing checks only, never a measurement) accepts the clamp.
    let plumbing = args.get("allow-background") == Some("1");
    let qos_ok = pool.qos_report().iter().all(|r| if plumbing { r.ok() } else { r.effective_ok() });
    emit(json!({"event": "qos", "t_ns": now_ns(), "records": qos, "ok": qos_ok, "plumbing": plumbing}));
    if !qos_ok {
        std::process::exit(3);
    }
    let mut rp = load_warm_audit::<K>(args, &sched, &dirs, &pool, threads, c_n);
    for (wi, d) in dirs.iter().enumerate() {
        let clip = Clip::load(d, false);
        rp.replay(&pool, &clip, Mode::Timed).unwrap();
        std::thread::sleep(Duration::from_nanos(s_pre + g));
        let (g0, q0) = (rp.gemm_calls, rp.quantizations);
        let mut per = Vec::with_capacity(replays[wi]);
        let t0 = now_ns();
        let mut last = t0;
        for _ in 0..replays[wi] {
            rp.replay(&pool, &clip, Mode::Timed).unwrap();
            let now = now_ns();
            per.push(now - last);
            last = now;
        }
        let t1 = last;
        let (calls, quants) = (rp.gemm_calls - g0, rp.quantizations - q0);
        // Nothing is written (and nothing is parsed by the runner) inside [t0 - G, t1 + G].
        std::thread::sleep(Duration::from_nanos(g + s_post));
        emit(json!({"event": "window_start", "window": wi, "clip": d.file_name().unwrap().to_string_lossy(), "id": clip.id(),
                    "T": clip.t, "t_ns": t0}));
        emit(json!({"event": "window_end", "window": wi, "t_ns": t1, "replays": per.len(), "replay_ns": per,
                    "gemm_calls": calls, "quantizations": quants}));
    }
    let end: Vec<_> = pool.readback_qos();
    let end_ok = end.iter().all(|r| r.got == crate::pool::QOS_CLASS_USER_INTERACTIVE && (plumbing || r.effective_ok()));
    emit(json!({"event": "qos_end", "t_ns": now_ns(), "records": end.iter().map(|r| r.to_json()).collect::<Vec<_>>(), "ok": end_ok}));
    emit(json!({"event": "end", "t_ns": now_ns(), "gemm_calls": rp.gemm_calls, "quantizations": rp.quantizations,
                "max_rss_bytes": max_rss_bytes(), "heap_peak_bytes": PEAK.load(Ordering::Relaxed)}));
    if !end_ok {
        std::process::exit(3);
    }
}
