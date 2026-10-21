//! cpu1bit shared crate (agent H): the interface, quantizer, epilogue, oracle, weights, thread pool,
//! schedule, captures, contract harness and replay runtime for the native 1-bit CPU kernels on the
//! M1 Pro (plans/cpu-benchmark-mac.md, amendments M1-M12).

pub mod aligned;
pub mod bench;
pub mod contract;
pub mod data;
pub mod epilogue;
pub mod golden;
pub mod kernel;
pub mod model;
pub mod oracle;
pub mod pool;
pub mod quant;
pub mod reference;
pub mod replay;
pub mod rng;
pub mod schedule;
pub mod weights;

pub use kernel::{Call, Kernel, KernelConfig, Out, Slot, WeightBytes};
pub use pool::{PerThread, Qos, ThreadPool};
pub use quant::QuantizedActs;
pub use weights::{BinaryMatrix, TernaryMatrix, WeightFormat, WeightsRef};

/// Build provenance recorded with every result.
pub const RUSTC: &str = env!("CPU1BIT_RUSTC");
pub const RUSTFLAGS: &str = env!("CPU1BIT_RUSTFLAGS");
pub const TARGET: &str = env!("CPU1BIT_TARGET");

/// Splits `0..len` into `parts` nearly equal contiguous ranges (LocalFlow's `split_range`).
pub fn split_range(len: usize, parts: usize, idx: usize) -> std::ops::Range<usize> {
    let base = len / parts;
    let extra = len % parts;
    let start = idx * base + idx.min(extra);
    start..start + base + usize::from(idx < extra)
}

/// Raw pointer wrapper so disjoint output regions can be written from pool threads.
pub struct SendPtr<T>(pub *mut T);
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SendPtr<T> {}
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    /// Method access makes closures capture the whole wrapper, not the raw field.
    pub fn get(self) -> *mut T {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::split_range;
    #[test]
    fn split_covers_everything_once() {
        for len in [0, 1, 5, 16, 17] {
            for parts in [1, 3, 8] {
                let mut next = 0;
                for i in 0..parts {
                    let r = split_range(len, parts, i);
                    assert_eq!(r.start, next);
                    next = r.end;
                }
                assert_eq!(next, len);
            }
        }
    }
}
