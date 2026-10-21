"""Experimental SM120 CUDA/PTX residual-INT8 matrix multiplies.

NVRTC compiles headerless CUDA, with explicit m16n8k32 integer Tensor Core
instructions. No model weights or activation precision change. Compile before
graph capture; tensors and launch arguments must remain alive until completion.
The public runtime continues to select its validated Triton implementation.
"""
import ctypes
import functools
import os
from pathlib import Path

import torch


_SOURCE = r'''
typedef unsigned int u32;
typedef signed char i8;

__device__ __forceinline__ void mma(int *c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ u32 load_word(const i8 *p, bool valid) {
    return valid ? *reinterpret_cast<const u32 *>(p) : 0u;
}

__device__ __forceinline__ void copy16(i8 *dst, const i8 *src, int valid_bytes) {
    unsigned long long wide_address;
    asm("cvta.to.shared.u64 %0, %1;" : "=l"(wide_address) : "l"(dst));
    u32 address = (u32)wide_address;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(address), "l"(src), "r"(valid_bytes));
}

template<int BM, int BN, int BK, bool ASYNC, bool WAIT=true>
__device__ __forceinline__ void stage_tile(const i8 *X, const i8 *W, i8 *sm,
                                          int M, int N, int K, int mbase, int nbase, int start) {
    // Padding by 16 bytes distributes the fragment's four-thread groups
    // across shared-memory banks and preserves 16-byte row alignment.
    const int stride = BK + 16;
    const int words = BK / 16;
    for (int i = threadIdx.x; i < (3 * BM + BN) * words; i += blockDim.x) {
        const int row = i / words, col = (i % words) * 16;
        const bool is_a = row < 3 * BM;
        const int component = row / BM;
        const int global_row = is_a ? mbase + row % BM : nbase + row - 3 * BM;
        const bool valid = global_row < (is_a ? M : N) && start + col < K;
        const i8 *src = valid ? (is_a ? X + (long long)(component * M + global_row) * K + start + col
                                      : W + (long long)global_row * K + start + col) : X;
        i8 *dst = sm + row * stride + col;
        if constexpr (ASYNC) {
            copy16(dst, src, valid ? 16 : 0);
        } else {
            u32 *d = reinterpret_cast<u32 *>(dst);
            const u32 *s = reinterpret_cast<const u32 *>(src);
            #pragma unroll
            for (int j = 0; j < 4; ++j) d[j] = valid ? s[j] : 0u;
        }
    }
    if constexpr (ASYNC) {
        asm volatile("cp.async.commit_group;");
        if constexpr (WAIT) asm volatile("cp.async.wait_group 0;");
    }
    if constexpr (WAIT) __syncthreads();
}

template<int WM, int WN, int RM, int RN, int BK, int SPLIT, int MODE>
__device__ __forceinline__ void residual_gemm(const i8 *X, const float *XS, const i8 *W,
        const float *WS, const float *Bias, float *Y, float *P, int M, int N, int K,
        int has_bias, int silu, i8 *sm) {
    constexpr int BM = WM * RM * 16, BN = WN * RN * 8;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int group = lane >> 2, sub = lane & 3;
    const int mb = blockIdx.y * BM, nb = blockIdx.x * BN;
    const int wm = warp / WN, wn = warp % WN;
    const int stride = BK + 16;
    const int tile_bytes = (3 * BM + BN) * stride;
    int accum[3][RM][RN][4] = {};
    int buffer = 0;
    if constexpr (MODE == 3) stage_tile<BM,BN,BK,true>(X,W,sm,M,N,K,mb,nb,blockIdx.z * BK);
    for (int start = blockIdx.z * BK; start < K; start += BK * SPLIT) {
        i8 *tile_sm = sm;
        if constexpr (MODE == 1 || MODE == 2) stage_tile<BM,BN,BK,MODE == 2>(X,W,sm,M,N,K,mb,nb,start);
        if constexpr (MODE == 3) {
            tile_sm = sm + buffer * tile_bytes;
            if (start + BK * SPLIT < K)
                stage_tile<BM,BN,BK,true,false>(X,W,sm + (1-buffer) * tile_bytes,M,N,K,mb,nb,start + BK * SPLIT);
        }
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 32) {
            u32 af[3][RM][4], bf[RN][2];
            #pragma unroll
            for (int j = 0; j < RN; ++j) {
                const int n = nb + (wn * RN + j) * 8 + group;
                const i8 *p = MODE == 0 ? W + (long long)n * K + start + kk + sub * 4
                                       : tile_sm + (3 * BM + (wn * RN + j) * 8 + group) * stride + kk + sub * 4;
                bf[j][0] = load_word(p, MODE != 0 || (n < N && start + kk + sub * 4 < K));
                bf[j][1] = load_word(p + 16, MODE != 0 || (n < N && start + kk + sub * 4 + 16 < K));
            }
            #pragma unroll
            for (int component = 0; component < 3; ++component) {
                #pragma unroll
                for (int i = 0; i < RM; ++i) {
                    const int local_m = (wm * RM + i) * 16 + group;
                    const int m = mb + local_m;
                    const i8 *p = MODE == 0 ? X + (long long)(component * M + m) * K + start + kk + sub * 4
                                           : tile_sm + (component * BM + local_m) * stride + kk + sub * 4;
                    const int row_stride = MODE == 0 ? K : stride;
                    af[component][i][0] = load_word(p, MODE != 0 || (m < M && start + kk + sub * 4 < K));
                    af[component][i][1] = load_word(p + 8 * row_stride, MODE != 0 || (m + 8 < M && start + kk + sub * 4 < K));
                    af[component][i][2] = load_word(p + 16, MODE != 0 || (m < M && start + kk + sub * 4 + 16 < K));
                    af[component][i][3] = load_word(p + 8 * row_stride + 16, MODE != 0 || (m + 8 < M && start + kk + sub * 4 + 16 < K));
                }
            }
            #pragma unroll
            for (int component = 0; component < 3; ++component)
                #pragma unroll
                for (int i = 0; i < RM; ++i)
                    #pragma unroll
                    for (int j = 0; j < RN; ++j)
                        mma(accum[component][i][j],af[component][i][0],af[component][i][1],
                            af[component][i][2],af[component][i][3],bf[j][0],bf[j][1]);
        }
        if constexpr (MODE == 3) {
            asm volatile("cp.async.wait_group 0;");
            buffer = 1 - buffer;
        }
        if constexpr (MODE != 0) __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < RM; ++i) {
        #pragma unroll
        for (int j = 0; j < RN; ++j) {
            #pragma unroll
            for (int element = 0; element < 4; ++element) {
                const int m = mb + (wm * RM + i) * 16 + group + (element >= 2 ? 8 : 0);
                const int n = nb + (wn * RN + j) * 8 + sub * 2 + (element & 1);
                if (m < M && n < N) {
                    float result = float(accum[0][i][j][element]) * XS[m];
                    result += float(accum[1][i][j][element]) * XS[M + m];
                    result += float(accum[2][i][j][element]) * XS[2 * M + m];
                    if constexpr (SPLIT == 1) {
                        result *= WS[n];
                        if (has_bias) result += Bias[n];
                        if (silu) result /= 1.0f + expf(-result);
                        Y[(long long)m * N + n] = result;
                    } else {
                        P[((long long)blockIdx.z * M + m) * N + n] = result;
                    }
                }
            }
        }
    }
}

#define ARGS const i8 *X,const float *XS,const i8 *W,const float *WS,const float *Bias, \
             float *Y,float *P,int M,int N,int K,int has_bias,int silu
#define WRAP(NAME,WM,WN,RM,RN,BK,SPLIT,MODE) \
extern "C" __global__ void NAME(ARGS) { \
    extern __shared__ i8 sm[]; \
    residual_gemm<WM,WN,RM,RN,BK,SPLIT,MODE>(X,XS,W,WS,Bias,Y,P,M,N,K,has_bias,silu,sm); \
}
// More candidates are generated by the Python loader without source duplication.
'''


CONFIGS = {
    # (M warps, N warps, M atoms/warp, N atoms/warp, K tile, split K, staging)
    'direct': (2, 2, 1, 2, 128, 1, 0),
    'direct_split4': (2, 2, 1, 2, 128, 4, 0),
    'direct_wide': (2, 2, 1, 4, 128, 1, 0),
    'direct_wide_split4': (2, 2, 1, 4, 128, 4, 0),
    'shared': (2, 2, 1, 4, 128, 1, 1),
    'shared_split4': (2, 2, 1, 4, 128, 4, 1),
    'async': (2, 2, 1, 4, 128, 1, 2),
    'async_split4': (2, 2, 1, 4, 128, 4, 2),
    'async_split8': (2, 2, 1, 4, 128, 8, 2),
    'async_large': (2, 2, 2, 4, 128, 1, 2),
    'async_deep': (2, 2, 1, 4, 256, 1, 2),
    'direct_small': (1, 4, 1, 1, 128, 1, 0),
    'direct_small_split4': (1, 4, 1, 1, 128, 4, 0),
    'direct_m16': (1, 4, 1, 2, 128, 1, 0),
    'direct_m16_split4': (1, 4, 1, 2, 128, 4, 0),
    'pipeline': (2, 2, 1, 4, 128, 1, 3),
    'pipeline_split4': (2, 2, 1, 4, 128, 4, 3),
    'pipeline_m16': (1, 4, 1, 2, 128, 1, 3),
    'pipeline_m16_split4': (1, 4, 1, 2, 128, 4, 3),
    'cublas': (1, 4, 1, 2, 128, 1, 0),
}

# Selected from the 15-shape sweep; this is the best CUDA candidate per shape,
# even when it loses to Triton. Keep selection separate from the public runtime.
AUTO = {
    (1024,1024): ('pipeline_m16_split4','async_split4','async_deep'),
    (3072,1024): ('async_split4','async_deep','async_large'),
    (4096,1024): ('async_split4','async_deep','async_deep'),
    (2048,1024): ('pipeline_m16_split4','pipeline_m16','async_large'),
    (1024,4096): ('async_split8','async_split8','async_split4'),
}


def selected(name, m, n, k):
    if name != 'auto':
        return name
    return AUTO.get((n,k), ('async_split4',)*3)[0 if m <= 64 else 1 if m <= 192 else 2]


@functools.lru_cache(None)
def kernel(name):
    import paths
    paths.require_mount()
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('compile CUDA matmul before graph capture')
    if torch.cuda.get_device_capability() != (12, 0):
        raise ValueError('experimental kernels target SM120')
    lib = Path('/mnt/hd/wilderness-labs-stt/inference-efficiency/deps/nvidia/cuda_nvrtc/lib/libnvrtc.so.12')
    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
    if not os.environ.get('CUDA_HOME'):
        cuda_home = paths.ARTIFACTS / 'caches' / 'cuda-headerless'
        (cuda_home / 'include').mkdir(parents=True, exist_ok=True)
        os.environ['CUDA_HOME'] = str(cuda_home)
    from torch.utils import cpp_extension
    if cpp_extension.CUDA_HOME is None:
        cpp_extension.CUDA_HOME = os.environ['CUDA_HOME']
    symbol = 'residual_' + name
    source = _SOURCE + '\nWRAP(' + symbol + ',' + ','.join(map(str, CONFIGS[name])) + ')\n'
    return torch.cuda._compile_kernel(source, symbol, compute_capability='120',
                                     nvcc_options=['--std=c++17', '--fmad=false'])


def dot(q, qs, weight, scale, bias, y, partial, *, name='async', activation=''):
    """Launch on the current torch stream, with column-major INT8 weights."""
    components, m, k = q.shape
    n = scale.numel()
    if components != 3 or q.dtype != torch.int8 or not q.is_contiguous() or k % 128:
        raise ValueError('expected three contiguous INT8 components, K divisible by 128')
    if weight.dtype != torch.int8 or weight.shape != (k, n) or weight.stride() != (1, k):
        raise ValueError('expected exact column-major INT8 weights')
    if activation not in ('', 'silu'):
        raise ValueError('unsupported activation')
    name = selected(name, m, n, k)
    if name == 'cublas':
        from .cuda_cublas import dot as cublas_dot
        return cublas_dot(q, qs, weight, scale, bias, y, activation=activation)
    wm, wn, rm, rn, bk, split, mode = CONFIGS[name]
    bm, bn = wm * rm * 16, wn * rn * 8
    shared = (3 * bm + bn) * (bk + 16) if mode else 0
    if mode == 3:
        shared *= 2
    kernel(name)(grid=((n + bn - 1) // bn, (m + bm - 1) // bm, split),
                 block=(wm * wn * 32, 1, 1), shared_mem=shared,
                 args=[q, qs, weight, scale, bias if bias is not None else y,
                       y, partial, m, n, k, int(bias is not None), int(activation == 'silu')])


class CudaProjection:
    """Experiment-only ExpandedProjection adapter; public defaults are unchanged."""
    def __init__(self, name='async', families=None):
        from .encoder4 import ExpandedProjection
        self.fallback = ExpandedProjection('triton_col_tuned')
        self.name = name
        self.families = families
        if name == 'auto':
            for candidate in set(v for choices in AUTO.values() for v in choices):
                kernel(candidate)
        elif name != 'cublas':
            kernel(name)

    @property
    def bytes(self):
        return self.fallback.bytes

    def __call__(self, x, mod, conv=False, components=3, activation='', norm=None,
                 config=None, input_activation=''):
        from .encoder4 import _expand
        from .fused_kernels import _quantize
        from .kernels import _finish
        import triton
        xx = x.transpose(1, 2) if conv else x
        if xx.ndim == 2:
            xx = xx.unsqueeze(0)
        b, t, k = xx.shape
        m, n = b * t, mod.out_features
        family = 'down' if k == 4096 else {4096: 'up', 3072: 'qkv', 2048: 'gate', 1024: 'mix'}.get(n, 'other')
        if components != 3 or k % 128 or (self.families is not None and family not in self.families):
            return self.fallback(x, mod, conv, components, activation, norm, config, input_activation)
        key = (mod.packed_t.data_ptr(), k, n, True)
        if key not in self.fallback.weights:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('warm expanded weights before capture')
            weight = torch.empty((n, k), device=x.device, dtype=torch.int8)
            _expand[(triton.cdiv(k * n, 512),)](mod.packed_t, weight, k, n, True, 512)
            self.fallback.weights[key] = weight.T
        weight = self.fallback.weights[key]
        q = torch.empty((3, m, k), device=x.device, dtype=torch.int8)
        qs = torch.empty((3, m), device=x.device, dtype=torch.float32)
        y = torch.empty((b, t, n), device=x.device, dtype=torch.float32)
        split = CONFIGS[selected(self.name, m, n, k)][5]
        partial = torch.empty((split, m, n), device=x.device) if split > 1 else y
        _quantize[(m,)](xx, q, qs, norm.weight if norm is not None else None,
                       norm.bias if norm is not None else None, m, k, t, *xx.stride(),
                       triton.next_power_of_2(k), 3, norm is not None,
                       norm.eps if norm is not None else 0.,
                       INPUT_SILU=input_activation == 'silu', enable_fp_fusion=False)
        dot(q, qs, weight, mod.scale, mod.bias, y, partial, name=self.name, activation=activation)
        if split > 1:
            _finish[(triton.cdiv(m * n, 256),)](partial, mod.scale, mod.bias, y,
                      m, n, t, t * n, n, 1, mod.bias is not None, split, 256, ACTIVATION=activation)
        return y.transpose(1, 2) if conv else y.reshape(*x.shape[:-1], n)
