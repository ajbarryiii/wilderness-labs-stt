//! A minimal scalar kernel on 1-bit codes: the template for kernel crates and the self-test of the
//! contract harness (shared/core/tests/contract.rs). Not an arm, never timed.

use crate::epilogue::finish;
use crate::kernel::{Call, Kernel, KernelConfig, Out, WeightBytes};
use crate::pool::ThreadPool;
use crate::quant::QuantizedActs;
use crate::weights::{BinaryMatrix, WeightFormat, WeightsRef};
use crate::{SendPtr, split_range};

pub struct ScalarBinary;

pub struct Prepared {
    m: BinaryMatrix,
}

impl Kernel for ScalarBinary {
    type Prepared = Prepared;
    const NAME: &'static str = "scalar-binary-reference";
    const FORMAT: WeightFormat = WeightFormat::Binary;

    fn new(_cfg: &KernelConfig) -> Self {
        ScalarBinary
    }

    fn prepare(&self, w: WeightsRef<'_>) -> Prepared {
        match w {
            WeightsRef::Binary(m) => Prepared { m: m.clone() },
            WeightsRef::Ternary(_) => panic!("binary kernel"),
        }
    }

    fn gemm(&mut self, pool: &ThreadPool, w: &Prepared, qa: &QuantizedActs, _call: Call, out: Out<'_>) {
        let (n, k, t_n, c_n) = (w.m.n, w.m.k, qa.rows(), qa.components());
        assert_eq!(qa.cols(), k);
        let threads = pool.threads();
        let (yp, dp, dots) = match out {
            Out::F32(y) => {
                assert!(y.len() >= t_n * n);
                (SendPtr(y.as_mut_ptr()), SendPtr(std::ptr::null_mut::<i32>()), false)
            }
            Out::Dots(d) => {
                assert!(d.len() >= t_n * c_n * n);
                (SendPtr(std::ptr::null_mut::<f32>()), SendPtr(d.as_mut_ptr()), true)
            }
        };
        pool.run(&|idx| {
            for o in split_range(n, threads, idx) {
                for t in 0..t_n {
                    let mut d = [0i32; 3];
                    for (c, dc) in d.iter_mut().enumerate().take(c_n) {
                        let a = qa.part(t, c);
                        // d = 2 * Σ_{b=1} q - Σ q.
                        let (mut pos, mut all) = (0i32, 0i32);
                        for (i, &av) in a.iter().enumerate() {
                            all += av as i32;
                            if w.m.bit(o, i) {
                                pos += av as i32;
                            }
                        }
                        *dc = 2 * pos - all;
                    }
                    unsafe {
                        if dots {
                            for (c, &dc) in d.iter().enumerate().take(c_n) {
                                *dp.get().add((t * c_n + c) * n + o) = dc;
                            }
                        } else {
                            *yp.get().add(t * n + o) = finish(&d[..c_n], w.m.scale[o], qa.scale()[t]);
                        }
                    }
                }
            }
        });
    }

    fn unpack(&self, w: &Prepared) -> Vec<i8> {
        w.m.signs()
    }
    fn shape(&self, w: &Prepared) -> (usize, usize) {
        (w.m.n, w.m.k)
    }
    fn weight_bytes(&self, w: &Prepared) -> WeightBytes {
        WeightBytes { codes: w.m.codes.len(), per_row: 4 * w.m.n }
    }
    fn workspace_bytes(&self) -> usize {
        0
    }
}
