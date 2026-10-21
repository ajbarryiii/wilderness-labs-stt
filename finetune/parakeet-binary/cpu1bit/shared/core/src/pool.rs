//! Persistent thread pool, LocalFlow's design (lf-cpu/src/pool.rs of the snapshot: the caller is
//! index 0, workers spin `SPIN_ITERATIONS` then park, a job panic aborts), with macOS QoS in place
//! of Linux pinning (plans/cpu-benchmark-mac.md "Threads", M4).
//!
//! Every thread, the caller included, sets its QoS class when the pool is built and reads it back
//! with `pthread_get_qos_class_np`; [`ThreadPool::qos_report`] has the start-up readback and
//! [`ThreadPool::readback_qos`] re-reads it on every thread inside a job (used at block end).
//! macOS gives no thread-to-core evidence, so nothing here claims P-core placement.

use std::cell::UnsafeCell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread::{self, JoinHandle, Thread};

type Job = *const (dyn Fn(usize) + Sync);

/// Same constant as LocalFlow's pool.
pub const SPIN_ITERATIONS: u32 = 200_000;

/// Requested thread QoS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qos {
    /// QOS_CLASS_USER_INTERACTIVE (0x21): the timed condition "8 / 4 USER_INTERACTIVE workers".
    UserInteractive,
    /// Leave the inherited class unchanged (unit tests on non-Mac hosts).
    Inherit,
}

/// One thread's QoS readback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QosRecord {
    pub index: usize,
    /// Raw class requested (0x21) or 0 for Inherit.
    pub requested: u32,
    /// `pthread_set_qos_class_self_np` return code (0 = success); -1 when not attempted.
    pub set_rc: i32,
    /// Raw class read back by `pthread_get_qos_class_np`; u32::MAX when unsupported.
    pub got: u32,
    pub relative_priority: i32,
    /// `getpriority(PRIO_DARWIN_PROCESS, 0)`: 1 when the whole process is darwin-background (the
    /// job supervisor's low priority does this: E-cores only, whatever the thread QoS says);
    /// -1 when unsupported.
    pub darwin_process_bg: i32,
    /// `getpriority(PRIO_DARWIN_THREAD, 0)` of this thread.
    pub darwin_thread_bg: i32,
}

impl QosRecord {
    /// The requested class was set and reads back (pthread_get_qos_class_np).
    pub fn ok(&self) -> bool {
        self.requested == 0 || (self.set_rc == 0 && self.got == self.requested && self.relative_priority == 0)
    }
    /// `ok()` and neither the process nor the thread is darwin-background: required for timed
    /// blocks (a background process is clamped to the E-cores regardless of thread QoS).
    pub fn effective_ok(&self) -> bool {
        self.ok() && self.darwin_process_bg == 0 && self.darwin_thread_bg == 0
    }
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({"index": self.index, "requested": format!("{:#x}", self.requested), "set_rc": self.set_rc,
                           "got": format!("{:#x}", self.got), "relative_priority": self.relative_priority,
                           "darwin_process_bg": self.darwin_process_bg, "darwin_thread_bg": self.darwin_thread_bg})
    }
}

pub const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;

#[cfg(target_os = "macos")]
mod sys {
    use libc::{c_int, pthread_t};
    unsafe extern "C" {
        pub fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: c_int) -> c_int;
        pub fn pthread_get_qos_class_np(thread: pthread_t, qos_class: *mut u32, relative_priority: *mut c_int) -> c_int;
    }
}

fn set_qos(index: usize, q: Qos) -> QosRecord {
    let requested = match q {
        Qos::UserInteractive => QOS_CLASS_USER_INTERACTIVE,
        Qos::Inherit => 0,
    };
    let set_rc = if requested == 0 {
        -1
    } else {
        #[cfg(target_os = "macos")]
        {
            unsafe { sys::pthread_set_qos_class_self_np(requested, 0) }
        }
        #[cfg(not(target_os = "macos"))]
        {
            panic!("QoS classes exist only on macOS")
        }
    };
    let (got, relative_priority) = read_qos();
    let (darwin_process_bg, darwin_thread_bg) = darwin_bg();
    QosRecord { index, requested, set_rc, got, relative_priority, darwin_process_bg, darwin_thread_bg }
}

/// (process, thread) darwin background state via getpriority(PRIO_DARWIN_PROCESS / _THREAD, 0).
pub fn darwin_bg() -> (i32, i32) {
    #[cfg(target_os = "macos")]
    {
        // <sys/resource.h>: PRIO_DARWIN_THREAD 3, PRIO_DARWIN_PROCESS 4.
        unsafe { (libc::getpriority(4, 0), libc::getpriority(3, 0)) }
    }
    #[cfg(not(target_os = "macos"))]
    {
        (-1, -1)
    }
}

/// The calling thread's QoS class and relative priority.
pub fn read_qos() -> (u32, i32) {
    #[cfg(target_os = "macos")]
    {
        let (mut got, mut rel) = (0u32, 0i32);
        let rc = unsafe { sys::pthread_get_qos_class_np(libc::pthread_self(), &mut got, &mut rel) };
        if rc != 0 {
            return (u32::MAX, rc);
        }
        (got, rel)
    }
    #[cfg(not(target_os = "macos"))]
    {
        (u32::MAX, 0)
    }
}

struct Shared {
    generation: AtomicU64,
    pending: AtomicUsize,
    shutdown: AtomicBool,
    job: UnsafeCell<Option<Job>>,
}

unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

pub struct ThreadPool {
    shared: Arc<Shared>,
    workers: Vec<(Thread, JoinHandle<()>)>,
    busy: AtomicBool,
    qos: Qos,
    report: Vec<QosRecord>,
}

impl ThreadPool {
    /// `threads` threads in all; the caller becomes index 0 and gets the same QoS as the workers.
    pub fn new(threads: usize, qos: Qos) -> Self {
        assert!(threads >= 1);
        let caller = set_qos(0, qos);
        let shared = Arc::new(Shared {
            generation: AtomicU64::new(0),
            pending: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            job: UnsafeCell::new(None),
        });
        let records = Arc::new(Mutex::new(vec![caller]));
        let started = Arc::new(AtomicUsize::new(0));
        let workers: Vec<_> = (1..threads)
            .map(|i| {
                let shared = Arc::clone(&shared);
                let records = Arc::clone(&records);
                let started = Arc::clone(&started);
                let handle = thread::Builder::new()
                    .name(format!("cpu1bit-{i}"))
                    .spawn(move || {
                        let rec = set_qos(i, qos);
                        records.lock().unwrap().push(rec);
                        started.fetch_add(1, Ordering::Release);
                        worker_loop(&shared, i);
                    })
                    .expect("spawn worker");
                (handle.thread().clone(), handle)
            })
            .collect();
        while started.load(Ordering::Acquire) != threads - 1 {
            std::hint::spin_loop();
        }
        let mut report = records.lock().unwrap().clone();
        report.sort_by_key(|r| r.index);
        ThreadPool { shared, workers, busy: AtomicBool::new(false), qos, report }
    }

    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    pub fn qos(&self) -> Qos {
        self.qos
    }

    /// QoS set and read back by every thread when the pool was built.
    pub fn qos_report(&self) -> &[QosRecord] {
        &self.report
    }

    /// Re-reads every thread's QoS inside a job (each index reports its own thread).
    pub fn readback_qos(&self) -> Vec<QosRecord> {
        let requested = self.report[0].requested;
        let out: Vec<Mutex<Option<QosRecord>>> = (0..self.threads()).map(|_| Mutex::new(None)).collect();
        self.run(&|idx| {
            let (got, rel) = read_qos();
            let (pbg, tbg) = darwin_bg();
            *out[idx].lock().unwrap() = Some(QosRecord {
                index: idx,
                requested,
                set_rc: 0,
                got,
                relative_priority: rel,
                darwin_process_bg: pbg,
                darwin_thread_bg: tbg,
            });
        });
        out.into_iter().map(|m| m.into_inner().unwrap().expect("every index ran")).collect()
    }

    /// Runs `f(idx)` once on every thread, the caller as index 0; returns when all finish.
    pub fn run(&self, f: &(dyn Fn(usize) + Sync)) {
        if self.busy.swap(true, Ordering::Acquire) {
            panic!("ThreadPool::run called while the pool is already running a job");
        }
        if self.workers.is_empty() {
            run_job(f, 0);
            self.busy.store(false, Ordering::Release);
            return;
        }
        // Erase the lifetime: `run` does not return (and cannot unwind) until every worker is done.
        let job: Job = unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), Job>(f) };
        unsafe { *self.shared.job.get() = Some(job) };
        self.shared.pending.store(self.workers.len(), Ordering::Relaxed);
        self.shared.generation.fetch_add(1, Ordering::Release);
        for (t, _) in &self.workers {
            t.unpark();
        }
        run_job(f, 0);
        while self.shared.pending.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
        unsafe { *self.shared.job.get() = None };
        self.busy.store(false, Ordering::Release);
    }
}

fn run_job(f: &(dyn Fn(usize) + Sync), idx: usize) {
    if let Err(payload) = catch_unwind(AssertUnwindSafe(|| f(idx))) {
        std::mem::forget(payload);
        std::process::abort();
    }
}

fn worker_loop(shared: &Shared, idx: usize) {
    let mut seen = 0u64;
    loop {
        let mut spins = 0;
        let generation = loop {
            let g = shared.generation.load(Ordering::Acquire);
            if g != seen {
                break g;
            }
            if spins < SPIN_ITERATIONS {
                spins += 1;
                std::hint::spin_loop();
            } else {
                thread::park();
            }
        };
        seen = generation;
        if shared.shutdown.load(Ordering::Acquire) {
            return;
        }
        let job = unsafe { (*shared.job.get()).expect("job published") };
        run_job(unsafe { &*job }, idx);
        shared.pending.fetch_sub(1, Ordering::Release);
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.generation.fetch_add(1, Ordering::Release);
        for (t, _) in &self.workers {
            t.unpark();
        }
        for (_, handle) in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

/// Per-thread mutable state indexed by pool index (kernel tiles, tables).
pub struct PerThread<T> {
    cells: Vec<UnsafeCell<T>>,
}

unsafe impl<T: Send> Sync for PerThread<T> {}
unsafe impl<T: Send> Send for PerThread<T> {}

impl<T> PerThread<T> {
    pub fn new(threads: usize, mut make: impl FnMut(usize) -> T) -> Self {
        PerThread { cells: (0..threads).map(|i| UnsafeCell::new(make(i))).collect() }
    }
    pub fn len(&self) -> usize {
        self.cells.len()
    }
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }
    /// # Safety
    /// Only the thread running pool index `idx` may hold this reference, once per job.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn get(&self, idx: usize) -> &mut T {
        unsafe { &mut *self.cells[idx].get() }
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.cells.iter_mut().map(|c| c.get_mut())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_index_runs_once_per_job() {
        let pool = ThreadPool::new(4, Qos::Inherit);
        let hits: Vec<AtomicUsize> = (0..4).map(|_| AtomicUsize::new(0)).collect();
        for _ in 0..100 {
            pool.run(&|i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
        }
        assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 100));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn user_interactive_qos_is_set_and_read_back_on_every_thread() {
        for threads in [1, 4, 8] {
            let pool = ThreadPool::new(threads, Qos::UserInteractive);
            assert_eq!(pool.qos_report().len(), threads);
            for r in pool.qos_report() {
                assert!(r.ok(), "{r:?}");
            }
            for r in pool.readback_qos() {
                assert_eq!(r.got, QOS_CLASS_USER_INTERACTIVE, "{r:?}");
            }
        }
    }
}
