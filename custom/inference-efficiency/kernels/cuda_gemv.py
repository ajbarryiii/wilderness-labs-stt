"""Optional NVRTC-compiled warp-per-row GEMV for packed W1/W2 decode.

Compiled lazily at first use through torch's NVRTC binding; the NVRTC shared
library comes from the ``nvidia-cuda-nvrtc-cu12`` wheel installed under the
experiment deps directory (see requirements.in). If compilation is
unavailable, ``try_load()`` warns once and returns None so ``PackedWeight``
can fall back to Triton. ``load()`` raises instead, for explicit CUDA tests.
Warm up before CUDA graph capture; launches use torch's current stream.

Kernel structure: one warp per output row and one M index per grid row.
Lane gi preloads packed words gi, gi+32, ... and broadcasts them through
warp shuffles; activation reads are directly coalesced. Large output widths
can use a row-per-lane variant with shared-memory activation/weight tiles.
A general warp-per-row path handles ragged or larger K. All variants decode
signed coefficients before FP32 accumulation and round scales to FP16,
matching the dense reference weights. No CUDA headers are required: FP16
values move through raw 16-bit registers with inline cvt instructions.
"""

import ctypes
import functools
import os
import tempfile
import warnings
from pathlib import Path

import torch

_WARPS = 4

_SOURCE = r"""
typedef unsigned int u32;
typedef unsigned short fp16_t;

__device__ __forceinline__ float h2f(fp16_t h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

__device__ __forceinline__ fp16_t f2h(float f) {
    fp16_t h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}

// Lane gi owns packed words gi, gi+32, ... (all loads issued up front: one
// memory-latency round per row, fully coalesced). Step s covers k=32*s+gi;
// the needed word is broadcast from its owner lane with __shfl_sync while
// the bit position stays a fixed per-lane register, and x reads are
// coalesced direct __ldg (no shared memory staging: a per-block fill was
// measured to dominate runtime when L2 is contended). Per-value cost:
// LDG + CVT + SHFL + BFE + select/I2F + FMA.

__device__ __forceinline__ void store_result(
        float acc, const float* S, const fp16_t* Bias,
        fp16_t* Y, int row, int m, int n, int has_bias) {
    // Match PackedWeight.dense() and Triton: scales are stored as FP32,
    // but the logical weights use the FP16-rounded scale.
    const float scale = h2f(f2h(__ldg(S + row)));
    float value = acc * scale;
    if (has_bias) value += h2f(__ldg(Bias + row));
    Y[(long long)m * n + row] = f2h(value);
}

__device__ __forceinline__ void epilogue(
        float acc, const float* S, const fp16_t* Bias,
        fp16_t* Y, int row, int gi, int m, int n, int has_bias) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (gi == 0) store_result(acc, S, Bias, Y, row, m, n, has_bias);
}

template <int BITS, int WPL>
__device__ __forceinline__ void gemv_warp_impl(
        const fp16_t* X, const u32* W, const float* S, const fp16_t* Bias,
        fp16_t* Y, int N, int K, int KW, int has_bias) {
    const int gi = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= N) return;
    const int m = blockIdx.y;
    const u32* wrow = W + (long long)row * KW;
    const fp16_t* xm = X + (long long)m * K + gi;
    u32 w[WPL];
    #pragma unroll
    for (int j = 0; j < WPL; ++j)
        w[j] = (gi + (j << 5) < KW) ? __ldg(wrow + gi + (j << 5)) : 0u;
    const int bpos = BITS == 1 ? gi : ((gi & 15) << 1);
    const int wsel = BITS == 1 ? 0 : (gi >> 4);
    float acc = 0.0f;
    #pragma unroll
    for (int j = 0; j < WPL; ++j) {
        #pragma unroll
        for (int t = 0; t < 32 / BITS; ++t) {
            const int s = (j << (BITS == 1 ? 5 : 4)) + t;
            const int src = BITS == 1 ? t : (((s << 1) | wsel) & 31);
            const u32 wv = __shfl_sync(0xffffffffu, w[j], src);
            const float xv = h2f(__ldg(xm + (s << 5)));  // K % 32 == 0
            if (BITS == 1) {
                const float sgn = (wv & (1u << bpos)) ? 1.0f : -1.0f;
                acc = fmaf(xv, sgn, acc);
            } else {
                const float c = (float)((wv >> bpos) & 3u);
                acc = fmaf(xv, c - 1.0f, acc);
            }
        }
    }
    epilogue(acc, S, Bias, Y, row, gi, m, N, has_bias);
}

// Row-per-lane kernel for large N: each lane owns one whole row (32 rows
// per warp), so there is no shuffle and no reduction. Weights are staged
// as a padded tile in shared memory (the block's 128 rows are contiguous
// in DRAM; an odd row stride keeps lane reads bank-conflict free) and x
// is staged as float (broadcast reads, converted once per block).
// Decode signed values directly: sum(x * code) - sum(x) can catastrophically
// cancel, including nonzero outputs for an all-zero ternary row.
template <int BITS>
__device__ __forceinline__ void gemv_rpl_impl(
        const fp16_t* X, const u32* W, const float* S, const fp16_t* Bias,
        fp16_t* Y, int N, int K, int KW, int has_bias) {
    constexpr int VPW = 32 / BITS;
    extern __shared__ float smem[];  // float xs[K], then u32 wt[128][KW|1]
    float* xs = smem;
    u32* wt = (u32*)(smem + K);
    const int m = blockIdx.y;
    const fp16_t* xm = X + (long long)m * K;
    for (int i = threadIdx.x; i < K; i += 128)
        xs[i] = h2f(__ldg(xm + i));
    const int r0 = blockIdx.x << 7;
    const int gi = threadIdx.x & 31;
    const int stride = KW | 1;
    for (int r = threadIdx.x >> 5; r < 128; r += 4)
        for (int c = gi; c < KW; c += 32)
            wt[r * stride + c] = (r0 + r < N)
                ? __ldg(W + (long long)(r0 + r) * KW + c) : 0u;
    __syncthreads();
    const int rl = threadIdx.x;
    const int row = r0 + rl;
    const u32* wrow = wt + rl * stride;
    float acc = 0.0f;
    for (int t = 0; t < KW; ++t) {
        const u32 wv = wrow[t];
        #pragma unroll
        for (int jj = 0; jj < VPW; ++jj) {
            const float c = (float)((wv >> (jj * BITS)) & ((1u << BITS) - 1u));
            const float code = BITS == 1 ? 2.0f * c - 1.0f : c - 1.0f;
            acc = fmaf(xs[t * VPW + jj], code, acc);
        }
    }
    if (row < N) store_result(acc, S, Bias, Y, row, m, N, has_bias);
}

// General fallback: scalar word loop, no shared memory, ragged-K safe.
template <int BITS>
__device__ __forceinline__ void gemv_gen_impl(
        const fp16_t* X, const u32* W, const float* S, const fp16_t* Bias,
        fp16_t* Y, int N, int K, int KW, int has_bias) {
    constexpr int VPW = 32 / BITS;
    const int gi = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= N) return;
    const int m = blockIdx.y;
    const u32* wrow = W + (long long)row * KW;
    const fp16_t* xm = X + (long long)m * K;
    const int bpos = BITS == 1 ? gi : ((gi & 15) << 1);
    const int wsel = BITS == 1 ? 0 : (gi >> 4);
    const int nsteps = (KW * VPW + 31) >> 5;
    float acc = 0.0f;
    for (int s = 0; s < nsteps; ++s) {
        const int wi = BITS == 1 ? s : ((s << 1) | wsel);
        const int kk = (s << 5) + gi;
        u32 wv = 0u;
        if (wi < KW) wv = __ldg(wrow + wi);
        float xv = 0.0f;
        if (kk < K) xv = h2f(__ldg(xm + kk));
        if (BITS == 1) {
            const float s = (wv & (1u << bpos)) ? 1.0f : -1.0f;
            acc = fmaf(xv, s, acc);
        } else {
            const float c = (float)((wv >> bpos) & 3u);
            acc = fmaf(xv, c - 1.0f, acc);
        }
    }
    epilogue(acc, S, Bias, Y, row, gi, m, N, has_bias);
}

#define GEMV_ARGS const fp16_t* X, const u32* W, const float* S,           \
                  const fp16_t* Bias, fp16_t* Y, int N, int K, int KW,     \
                  int has_bias
#define GEMV_WARP_WRAPPER(NAME, BITS, WPL)                                 \
    extern "C" __global__ void __launch_bounds__(128) NAME(GEMV_ARGS) {    \
        gemv_warp_impl<BITS, WPL>(X, W, S, Bias, Y, N, K, KW, has_bias);   \
    }
#define GEMV_GEN_WRAPPER(NAME, BITS)                                       \
    extern "C" __global__ void __launch_bounds__(128) NAME(GEMV_ARGS) {    \
        gemv_gen_impl<BITS>(X, W, S, Bias, Y, N, K, KW, has_bias);         \
    }

GEMV_WARP_WRAPPER(gemv_b1_w1, 1, 1)
GEMV_WARP_WRAPPER(gemv_b1_w2, 1, 2)
GEMV_WARP_WRAPPER(gemv_b1_w4, 1, 4)
GEMV_WARP_WRAPPER(gemv_b2_w1, 2, 1)
GEMV_WARP_WRAPPER(gemv_b2_w2, 2, 2)
GEMV_WARP_WRAPPER(gemv_b2_w4, 2, 4)
GEMV_WARP_WRAPPER(gemv_b2_w8, 2, 8)
GEMV_GEN_WRAPPER(gemv_b1_gen, 1)
GEMV_GEN_WRAPPER(gemv_b2_gen, 2)

#define GEMV_RPL_WRAPPER(NAME, BITS)                                       \
    extern "C" __global__ void __launch_bounds__(128) NAME(GEMV_ARGS) {    \
        gemv_rpl_impl<BITS>(X, W, S, Bias, Y, N, K, KW, has_bias);         \
    }

GEMV_RPL_WRAPPER(gemv_b1_rpl, 1)
GEMV_RPL_WRAPPER(gemv_b2_rpl, 2)
"""


def _preload_nvrtc() -> None:
    """Make libnvrtc visible to torch before its first NVRTC use."""
    root = os.environ.get("EFFICIENCY_ARTIFACT_ROOT")
    if root:
        lib = (Path(root) / "deps" / "nvidia" / "cuda_nvrtc" / "lib"
               / "libnvrtc.so.12")
        if lib.is_file():
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
    # torch's _nvrtc_compile insists on CUDA_HOME/include existing; the
    # kernel below is headerless, so an empty include dir is sufficient.
    if not os.environ.get("CUDA_HOME"):
        home = Path(root) / "deps" / "cuda-home" if root else Path(
            tempfile.gettempdir()) / "wl-cuda-home"
        (home / "include").mkdir(parents=True, exist_ok=True)
        os.environ["CUDA_HOME"] = str(home)
    # cpp_extension caches CUDA_HOME at import time. Other model/runtime
    # imports may have initialized it before our headerless CUDA shim.
    from torch.utils import cpp_extension
    if cpp_extension.CUDA_HOME is None:
        cpp_extension.CUDA_HOME = os.environ["CUDA_HOME"]


_WPL_MAX = {1: 4, 2: 8}  # words-per-lane template instantiations
_RPL_MAX_KW = 64  # row-per-lane weight tile: 128 rows x 65 words max
_RPL_MIN_N = 8192  # below this the warp-per-row kernels win


@functools.lru_cache(maxsize=1)
def load():
    """Compile {(bits, variant): kernel}, preserving errors for CUDA validation."""
    _preload_nvrtc()
    from torch.cuda import _compile_kernel
    kernels = {}
    for bits, wpls in ((1, (1, 2, 4)), (2, (1, 2, 4, 8))):
        for wpl in wpls:
            kernels[(bits, wpl)] = _compile_kernel(
                _SOURCE, f"gemv_b{bits}_w{wpl}")
        kernels[(bits, 0)] = _compile_kernel(_SOURCE, f"gemv_b{bits}_gen")
        kernels[(bits, "rpl")] = _compile_kernel(
            _SOURCE, f"gemv_b{bits}_rpl")
    return kernels


@functools.lru_cache(maxsize=1)
def try_load():
    """Use CUDA when available; report the reason for falling back once."""
    try:
        return load()
    except Exception as exc:
        if os.environ.get("EFFICIENCY_REQUIRE_CUDA_GEMV") == "1":
            raise
        warnings.warn(f"CUDA GEMV unavailable; using Triton: {exc}",
                      RuntimeWarning, stacklevel=2)
        return None


def gemv(kernels, x: torch.Tensor, words: torch.Tensor, scales: torch.Tensor,
         bias: torch.Tensor | None, out: torch.Tensor,
         n: int, k: int, kw: int, bits: int, m: int) -> None:
    """out[m] = (x[m] . unpack(W)) * scale + bias for flat contiguous x/out."""
    args = [x, words, scales, bias if bias is not None else out, out,
            n, k, kw, 1 if bias is not None else 0]
    if (n >= _RPL_MIN_N and k % 32 == 0 and kw <= _RPL_MAX_KW
            and k <= 2048):
        kernels[(bits, "rpl")](grid=((n + 127) // 128, m, 1),
                               block=(128, 1, 1), args=args,
                               shared_mem=4 * (k + 128 * (kw | 1)))
        return
    wpl = (kw + 31) // 32
    kern = None
    if k % 32 == 0 and wpl <= _WPL_MAX[bits] and kw == wpl * 32:
        kern = kernels.get((bits, wpl))
    if kern is None:
        kern = kernels[(bits, 0)]
    kern(grid=((n + _WARPS - 1) // _WARPS, m, 1), block=(_WARPS * 32, 1, 1),
         args=args)
