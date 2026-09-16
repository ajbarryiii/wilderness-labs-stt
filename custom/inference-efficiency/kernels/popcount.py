"""Experimental W1A1/W2A1 CUDA kernels; never installed in model dispatch.

Binary sign is 1 for positive; zero FP16 activations quantize to +1.
Ternary weights use separate positive-sign and nonzero bit planes.
Tail bits are masked. Outputs use FP16-rounded row scales and FP32 epilogues.
"""
import functools

import torch

from .cuda_gemv import _preload_nvrtc


SOURCE = r'''
typedef unsigned int u32;
typedef unsigned short half;
__device__ float h2f(half h) {
    float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f;
}
__device__ half f2h(float f) {
    half h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h;
}
extern "C" __global__ void pack_sign(const half* X, u32* A, int M, int K, int KW) {
    int lane = threadIdx.x & 31;
    int word = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int m = word / KW, k = (word % KW) * 32 + lane;
    bool positive = m < M && k < K && h2f(X[(long long)m*K+k]) >= 0.0f;
    u32 v = __ballot_sync(0xffffffffu, positive);
    if (lane == 0 && m < M) A[word] = v;
}
__device__ u32 valid_mask(int q, int K) {
    int tail = K - q * 32;
    return tail >= 32 ? 0xffffffffu : ((1u << tail) - 1u);
}
template<int TERNARY>
__device__ int contribution(u32 a, u32 s, u32 nz, u32 mask) {
    nz = TERNARY ? nz & mask : mask;
    // Equivalent to 2*popcount(XNOR(a,s)&nz)-popcount(nz).
    return __popc(nz) - 2 * __popc((a ^ s) & nz);
}
template<int TERNARY, int KFIX=0>
__device__ __forceinline__ void warp_dot(const u32* A, const u32* S, const u32* NZ,
                        const float* Scale, const half* Bias, half* Y,
                        int M, int N, int KARG, int KWARG, int has_bias) {
    const int K = KFIX ? KFIX : KARG;
    const int KW = KFIX ? (KFIX+31)/32 : KWARG;
    int lane = threadIdx.x & 31;
    int n = blockIdx.x * 4 + (threadIdx.x >> 5), m = blockIdx.y;
    if (n >= N) return;
    int sum = 0;
    for (int q = lane; q < KW; q += 32) {
        u32 s = S[(long long)n*KW+q];
        u32 nz = TERNARY ? NZ[(long long)n*KW+q] : 0;
        sum += contribution<TERNARY>(A[(long long)m*KW+q], s, nz, valid_mask(q,K));
    }
    for (int off=16; off; off>>=1) sum += __shfl_down_sync(0xffffffffu,sum,off);
    if (lane==0) {
        float value = (float)sum * h2f(f2h(Scale[n]));
        if (has_bias) value += h2f(Bias[n]);
        Y[(long long)m*N+n] = f2h(value);
    }
}
template<int TERNARY, int KFIX=0>
__device__ __forceinline__ void tiled_dot(const u32* A, const u32* S, const u32* NZ,
                         const float* Scale, const half* Bias, half* Y,
                         int M, int N, int KARG, int KWARG, int has_bias) {
    const int K = KFIX ? KFIX : KARG;
    const int KW = KFIX ? (KFIX+31)/32 : KWARG;
    // Four input rows x 32 outputs/block. Each lane owns an output;
    // weight words are contiguous across lanes in the transposed planes.
    int n = blockIdx.x*32 + (threadIdx.x & 31);
    int m = blockIdx.y*4 + (threadIdx.x >> 5);
    if (n >= N || m >= M) return;
    int sum = 0;
    for (int q=0; q<KW; ++q) {
        u32 s = S[(long long)q*N+n];
        u32 nz = TERNARY ? NZ[(long long)q*N+n] : 0;
        sum += contribution<TERNARY>(A[(long long)m*KW+q],s,nz,valid_mask(q,K));
    }
    float value = (float)sum * h2f(f2h(Scale[n]));
    if (has_bias) value += h2f(Bias[n]);
    Y[(long long)m*N+n] = f2h(value);
}
#define WRAP(NAME, FN, T, FIX) \
extern "C" __global__ void NAME(const u32* A,const u32* S,const u32* NZ, \
 const float* Scale,const half* Bias,half* Y,int M,int N,int K,int KW,int has_bias) { \
 FN<T,FIX>(A,S,NZ,Scale,Bias,Y,M,N,K,KW,has_bias); }
WRAP(binary_warp,warp_dot,0,0)
WRAP(ternary_warp,warp_dot,1,0)
WRAP(binary_tile,tiled_dot,0,0)
WRAP(ternary_tile,tiled_dot,1,0)
WRAP(binary_warp_k1024,warp_dot,0,1024)
WRAP(ternary_warp_k1024,warp_dot,1,1024)
WRAP(binary_tile_k1024,tiled_dot,0,1024)
WRAP(ternary_tile_k1024,tiled_dot,1,1024)
WRAP(binary_warp_k4096,warp_dot,0,4096)
WRAP(ternary_warp_k4096,warp_dot,1,4096)
WRAP(binary_tile_k4096,tiled_dot,0,4096)
WRAP(ternary_tile_k4096,tiled_dot,1,4096)
'''


@functools.lru_cache(maxsize=1)
def load():
    _preload_nvrtc()
    from torch.cuda import _compile_kernel
    names = ['pack_sign'] + [f'{alphabet}_{layout}{suffix}'
                            for alphabet in ('binary', 'ternary')
                            for layout in ('warp', 'tile')
                            for suffix in ('', '_k1024', '_k4096')]
    return {name: _compile_kernel(SOURCE, name) for name in names}


def pack(x, out):
    m, k = x.shape
    kw = (k + 31) // 32
    load()['pack_sign'](grid=((m * kw + 7) // 8, 1, 1), block=(256, 1, 1),
                        args=[x, out, m, k, kw])


def pack_plane(values):
    """CPU preparation also supports direct independent representation tests."""
    n, k = values.shape
    result = torch.zeros(n, (k + 31) // 32, dtype=torch.int64, device=values.device)
    for bit in range(32):
        part = values[:, bit::32].to(torch.int64)
        result[:, :part.shape[1]] |= part << bit
    return result.to(torch.int32)


class BitWeight:
    def __init__(self, codes, scales, bits):
        self.n, self.k = codes.shape
        self.bits = bits
        self.sign = pack_plane(codes > 0)
        self.nonzero = pack_plane(codes != 0) if bits == 2 else self.sign
        self.sign_t = self.sign.T.contiguous()
        self.nonzero_t = self.nonzero.T.contiguous() if bits == 2 else self.sign_t
        self.scales = scales

    def linear(self, a, out, variant='warp', bias=None):
        m = a.shape[0]
        tile = variant == 'tile'
        sign, nz = (self.sign_t, self.nonzero_t) if tile else (self.sign, self.nonzero)
        name = ('binary' if self.bits == 1 else 'ternary') + '_' + variant
        if self.k in (1024, 4096):
            name += f'_k{self.k}'
        load()[name](grid=((self.n + (31 if tile else 3)) // (32 if tile else 4),
                           (m + 3) // 4 if tile else m, 1), block=(128, 1, 1),
                     args=[a, sign, nz, self.scales, bias if bias is not None else out,
                           out, m, self.n, self.k, a.shape[1], int(bias is not None)])
        return out
