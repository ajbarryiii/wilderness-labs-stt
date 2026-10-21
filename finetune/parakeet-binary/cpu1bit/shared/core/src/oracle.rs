//! The independent scalar oracle (plans/cpu-benchmark-mac.md "Exactness"): plain integer dots in
//! i64 over the {-1, 0, +1} code values and LocalFlow's epilogue in its exact order. No SIMD, no
//! blocking, no shared code with any kernel except `epilogue::finish`, which is itself checked
//! against the x86 golden vectors.

use crate::epilogue::finish;
use crate::quant::QuantizedActs;

/// `d[(t * C + c) * N + o] = Σ_i q[t][c][i] · w[o][i]` for `w` = `[N][K]` values.
pub fn dots(w: &[i8], n: usize, k: usize, qa: &QuantizedActs) -> Vec<i32> {
    assert_eq!(qa.cols(), k);
    assert_eq!(w.len(), n * k);
    let (t_n, c_n) = (qa.rows(), qa.components());
    let mut d = vec![0i32; t_n * c_n * n];
    for t in 0..t_n {
        for c in 0..c_n {
            let a = qa.part(t, c);
            for o in 0..n {
                let row = &w[o * k..(o + 1) * k];
                let s: i64 = row.iter().zip(a).map(|(&wv, &av)| wv as i64 * av as i64).sum();
                d[(t * c_n + c) * n + o] = i32::try_from(s).expect("dot exceeds i32");
            }
        }
    }
    d
}

/// FP32 outputs `[T][N]` from oracle dots.
pub fn outputs(d: &[i32], n: usize, scale: &[f32], qa: &QuantizedActs) -> Vec<f32> {
    let (t_n, c_n) = (qa.rows(), qa.components());
    assert_eq!(d.len(), t_n * c_n * n);
    let mut y = vec![0f32; t_n * n];
    let mut parts = [0i32; 3];
    for t in 0..t_n {
        for o in 0..n {
            for c in 0..c_n {
                parts[c] = d[(t * c_n + c) * n + o];
            }
            y[t * n + o] = finish(&parts[..c_n], scale[o], qa.scale()[t]);
        }
    }
    y
}

pub fn gemm(w: &[i8], n: usize, k: usize, scale: &[f32], qa: &QuantizedActs) -> Vec<f32> {
    outputs(&dots(w, n, k, qa), n, scale, qa)
}
