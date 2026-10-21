"""Agent C: FP32 SIMT GEMM straight from 1-bit weights (SM120, NVRTC, headerless CUDA).

Why SIMT: mma.sync .b1 is emulated on SM120 (NOTES.md, C 07:49), so activation bit-planes on tensor
cores lose to INT8. On CUDA cores, y_j = alpha_j * sum_i w_ji x_i with w = +-1 costs one FFMA per
multiply-add once the sign is materialized: per k-tile the CTA expands its weight bits into shared
memory as +-1.0f (each sign built once per CTA, reused by all BM rows), then a register-tiled FP32
FFMA loop runs. No activation quantization, FP32 accumulation; weights stay in MB1's own layout.

Weights: MB1 codes uint8 [N, Kb], Kb = ceil(K/8), LSB first, bit 1 = +1. Resident exactly as
exported (no padding, no copy beyond the one upload), so weight_bytes == N*Kb + 4N (+4N bias).

Split-K: grid.z = SPLIT CTAs per output tile form one thread-block cluster (__cluster_dims__);
each accumulates a K-slice, writes its partial tile to its own shared memory, and after a cluster
barrier every CTA reduces a 1/SPLIT slice of the tile by reading all SPLIT partials through
distributed shared memory in fixed rank order. One kernel, no workspace, no atomics, deterministic.

Accumulation: INNER (16) k-steps of FFMA into a fresh partial, partials summed with Fast2Sum
compensation, split-K partials summed in fixed rank order (real-input numerics, NOTES.md C).

Kernel parameters (compile time): BM BN BK TM TN SPLIT XMODE WFAST HAS_BIAS ACT IN_ACT.
  XMODE 0: generic strides (scalar loads); 1: k-contiguous rows, float4 loads (needs K % 4 == 0,
           16-byte aligned rows); 2: m-contiguous (conv [B, K, T] layout, xt == 1), scalar loads
           with consecutive threads on consecutive frames.
  WFAST 1: K % BK == 0, Kb % 4 == 0: weight words loaded as aligned u32; 0: byte loads, bounds.
  ACT 1: SiLU on the output; IN_ACT 1: SiLU on the input (applied while staging x).
Runtime: X, T, xb, xt, xk (element strides; row m = b*T + t), W, Kb, scale, bias, Y (contiguous
[M, N]), M, N, K.
"""
from __future__ import annotations

import dataclasses

import torch

from . import nvrtc

SOURCE = r'''
typedef unsigned int u32;
typedef unsigned char u8;

#define INNER 16
#define TX (BN / TN)
#define TY (BM / TM)
#define THREADS (TX * TY)
#define XS (BM + 4)
#define WS (BN + 4)
#define XTILE (BK * XS)
#define WTILE (BK * WS)

static __device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }

#if SPLIT > 1
static __device__ __forceinline__ u32 shared_address(const void *p) {
    unsigned long long a;
    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(a) : "l"(p));
    return (u32)a;
}
static __device__ __forceinline__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
static __device__ __forceinline__ float4 load_remote4(u32 local, u32 rank) {
    u32 remote;
    float4 v;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local), "r"(rank));
    asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(remote) : "memory");
    return v;
}
#endif

#define X_TASKS (XMODE == 1 ? BM * BK / 4 : BM * BK)
#define X_PER ((X_TASKS + THREADS - 1) / THREADS)
#define W_WORDS (BN * BK / 32)
#define W_PER ((W_WORDS + THREADS - 1) / THREADS)

extern "C" __global__ void __launch_bounds__(THREADS)
#if SPLIT > 1
__cluster_dims__(1, 1, SPLIT)
#endif
bitserial_ffma(const float *__restrict__ X, int T, int xb, int xt, int xk,
               const u8 *__restrict__ W, int Kb, const float *__restrict__ Scale,
               const float *__restrict__ Bias, float *__restrict__ Y, int M, int N, int K) {
    extern __shared__ __align__(16) float smem[];
    float *Xs = smem;                 // [2][BK][XS]
    float *Wsm = smem + 2 * XTILE;    // [2][BK][WS]
    const int tid = threadIdx.x;
    const int tx = tid % TX, ty = tid / TX;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int tiles = (K + BK - 1) / BK;
#if SPLIT > 1
    const int per = (tiles + SPLIT - 1) / SPLIT;
    const int t_begin = blockIdx.z * per;
    const int t_end = min(tiles, t_begin + per);
#else
    const int t_begin = 0, t_end = tiles;
#endif

    // Per-thread staging assignments (fixed across k-tiles). Out-of-range rows/k stage zeros.
    float xr[X_PER][XMODE == 1 ? 4 : 1];
    u32 wr[W_PER];
    long long xrow[X_PER];
    int xlm[X_PER], xlk[X_PER];
    bool xin[X_PER];
    #pragma unroll
    for (int i = 0; i < X_PER; ++i) {
        const int task = tid + i * THREADS;
        int lm, lk;
        if (XMODE == 1) { lm = task / (BK / 4); lk = (task % (BK / 4)) * 4; }
        else if (XMODE == 2) { lm = task % BM; lk = task / BM; }
        else { lm = task / BK; lk = task % BK; }
        const int m = m0 + lm;
        xlm[i] = lm;
        xlk[i] = lk;
        xin[i] = task < X_TASKS && m < M;
        xrow[i] = xin[i] ? (long long)(m / T) * xb + (long long)(m % T) * xt : 0;
    }

    auto load_tile = [&](int tile) {
        const int kbase = tile * BK;
        #pragma unroll
        for (int i = 0; i < X_PER; ++i) {
            const int k = kbase + xlk[i];
            if (XMODE == 1) {
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (xin[i] && k < K) v = *reinterpret_cast<const float4 *>(X + xrow[i] + k);
                xr[i][0] = v.x; xr[i][1] = v.y; xr[i][2] = v.z; xr[i][3] = v.w;
            } else {
                xr[i][0] = (xin[i] && k < K) ? X[xrow[i] + (long long)k * xk] : 0.f;
            }
        }
        #pragma unroll
        for (int i = 0; i < W_PER; ++i) {
            const int task = tid + i * THREADS;
            const int ln = task % BN, word = task / BN;
            const int n = n0 + ln;
            u32 w = 0;
            if (task < W_WORDS && n < N) {
                const int byte = (kbase >> 3) + word * 4;
                if (WFAST) {
                    w = *reinterpret_cast<const u32 *>(W + (long long)n * Kb + byte);
                } else {
                    #pragma unroll
                    for (int b = 0; b < 4; ++b)
                        if (byte + b < Kb) w |= (u32)W[(long long)n * Kb + byte + b] << (8 * b);
                }
            }
            wr[i] = w;
        }
    };

    auto store_tile = [&](int buf) {
        float *xs = Xs + buf * XTILE;
        float *ws = Wsm + buf * WTILE;
        #pragma unroll
        for (int i = 0; i < X_PER; ++i) {
            if (tid + i * THREADS >= X_TASKS) continue;
            #pragma unroll
            for (int e = 0; e < (XMODE == 1 ? 4 : 1); ++e) {
                float v = xr[i][e];
                if (IN_ACT) v = silu(v);
                xs[(xlk[i] + e) * XS + xlm[i]] = v;
            }
        }
        #pragma unroll
        for (int i = 0; i < W_PER; ++i) {
            const int task = tid + i * THREADS;
            if (task < W_WORDS) {
                const int ln = task % BN, word = task / BN;
                const u32 inv = ~wr[i];                 // bit 0 (weight -1) -> sign bit set
                float *col = ws + word * 32 * WS + ln;
                #pragma unroll
                for (int j = 0; j < 32; ++j)
                    col[j * WS] = __int_as_float(0x3F800000u | ((inv << (31 - j)) & 0x80000000u));
            }
        }
    };

    // acc + comp: Fast2Sum-compensated sum of the INNER-k partials (C numerics probe on dev
    // fixtures: plain blocked FP32 summation exceeded M1's A3 limits by up to 2.1x on real inputs).
    float acc[TM][TN], comp[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) { acc[i][j] = 0.f; comp[i][j] = 0.f; }

    if (t_begin < t_end) {
        load_tile(t_begin);
        store_tile(0);
    }
    __syncthreads();
    int buf = 0;
    for (int tile = t_begin; tile < t_end; ++tile) {
        const bool more = tile + 1 < t_end;
        if (more) load_tile(tile + 1);
        const float *xs = Xs + buf * XTILE;
        const float *ws = Wsm + buf * WTILE;
        // Blocked summation: INNER k-steps accumulate into a fresh partial that is then added to
        // acc, so rounding error grows with INNER + K/INNER terms, not with the K-slice length.
        float part[TM][TN];
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            if (kk % INNER == 0) {
                #pragma unroll
                for (int i = 0; i < TM; ++i)
                    #pragma unroll
                    for (int j = 0; j < TN; ++j) part[i][j] = 0.f;
            }
            float a[TM], b[TN];
            #pragma unroll
            for (int q = 0; q < TM / 4; ++q) {
                const float4 v = *reinterpret_cast<const float4 *>(xs + kk * XS + ty * 4 + q * TY * 4);
                a[4 * q] = v.x; a[4 * q + 1] = v.y; a[4 * q + 2] = v.z; a[4 * q + 3] = v.w;
            }
            #pragma unroll
            for (int q = 0; q < TN / 4; ++q) {
                const float4 v = *reinterpret_cast<const float4 *>(ws + kk * WS + tx * 4 + q * TX * 4);
                b[4 * q] = v.x; b[4 * q + 1] = v.y; b[4 * q + 2] = v.z; b[4 * q + 3] = v.w;
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) part[i][j] = fmaf(a[i], b[j], part[i][j]);
            if (kk % INNER == INNER - 1) {
                #pragma unroll
                for (int i = 0; i < TM; ++i)
                    #pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        const float t = acc[i][j] + part[i][j];
                        comp[i][j] += part[i][j] - (t - acc[i][j]);
                        acc[i][j] = t;
                    }
            }
        }
        if (more) store_tile(buf ^ 1);
        __syncthreads();
        buf ^= 1;
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += comp[i][j];

#if SPLIT > 1
    // Partial tile [BM][BN] in this CTA's shared memory, then a fixed-order cluster reduction.
    float *red = smem;
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int lm = ty * 4 + (i % 4) + (i / 4) * TY * 4;
        #pragma unroll
        for (int q = 0; q < TN / 4; ++q) {
            const int ln = tx * 4 + q * TX * 4;
            *reinterpret_cast<float4 *>(red + lm * BN + ln) =
                make_float4(acc[i][4 * q], acc[i][4 * q + 1], acc[i][4 * q + 2], acc[i][4 * q + 3]);
        }
    }
    cluster_sync();
    u32 rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    const u32 base = shared_address(red);
    const int quads = BM * BN / 4;
    const int chunk = (quads + SPLIT - 1) / SPLIT;
    const int q_end = min(quads, (int)(rank + 1) * chunk);
    for (int q = rank * chunk + tid; q < q_end; q += THREADS) {
        float4 s = load_remote4(base + q * 16, 0);
        #pragma unroll
        for (int r = 1; r < SPLIT; ++r) {
            const float4 p = load_remote4(base + q * 16, r);
            s.x += p.x; s.y += p.y; s.z += p.z; s.w += p.w;
        }
        const int lm = (q * 4) / BN, ln = (q * 4) % BN;
        const int m = m0 + lm;
        if (m < M) {
            float v[4] = {s.x, s.y, s.z, s.w};
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int n = n0 + ln + e;
                if (n < N) {
                    float y = v[e] * Scale[n];
                    if (HAS_BIAS) y += Bias[n];
                    if (ACT) y = silu(y);
                    Y[(long long)m * N + n] = y;
                }
            }
        }
    }
    cluster_sync();
#else
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int m = m0 + ty * 4 + (i % 4) + (i / 4) * TY * 4;
        if (m >= M) continue;
        #pragma unroll
        for (int q = 0; q < TN / 4; ++q) {
            const int nq = n0 + tx * 4 + q * TX * 4;
            float v[4];
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int n = min(nq + e, N - 1);
                float y = acc[i][4 * q + e] * Scale[n];
                if (HAS_BIAS) y += Bias[n];
                if (ACT) y = silu(y);
                v[e] = y;
            }
            if ((N & 3) == 0) {
                if (nq < N) *reinterpret_cast<float4 *>(Y + (long long)m * N + nq) = make_float4(v[0], v[1], v[2], v[3]);
            } else {
                #pragma unroll
                for (int e = 0; e < 4; ++e) if (nq + e < N) Y[(long long)m * N + nq + e] = v[e];
            }
        }
    }
#endif
}
'''

# Activation prep for project(): one CTA per row reads x (any strides), optionally applies
# LayerNorm (two-pass FP32 mean / variance like F.layer_norm and M1's fused quantizer, then affine)
# and SiLU, and writes a contiguous FP32 row for the GEMM. Like M1's INT8x3 quantize kernel, this is
# the one extra launch per projection; it removes per-N-tile recomputation inside the GEMM.
PREP_SOURCE = r'''
static __device__ __forceinline__ float block_sum(float v, float *red) {
    #pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float s = 0.f;
    #pragma unroll
    for (int w = 0; w < PREP_THREADS / 32; ++w) s += red[w];
    return s;
}

extern "C" __global__ void __launch_bounds__(PREP_THREADS)
row_prep(const float *__restrict__ X, int T, int xb, int xt, int xk, int K,
         const float *__restrict__ Gamma, const float *__restrict__ Beta, double eps,
         float *__restrict__ Z) {
    __shared__ float red[PREP_THREADS / 32];
    const int m = blockIdx.x, tid = threadIdx.x;
    const float *row = X + (long long)(m / T) * xb + (long long)(m % T) * xt;
    float *out = Z + (long long)m * K;
    float v[PREP_PER];
    #pragma unroll
    for (int i = 0; i < PREP_PER; ++i) {
        const int k = tid + i * PREP_THREADS;
        v[i] = k < K ? row[(long long)k * xk] : 0.f;
    }
#if NORM
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < PREP_PER; ++i) s += v[i];
    const float mean = block_sum(s, red) / (float)K;
    float q = 0.f;
    #pragma unroll
    for (int i = 0; i < PREP_PER; ++i) {
        const int k = tid + i * PREP_THREADS;
        const float d = k < K ? v[i] - mean : 0.f;
        q += d * d;
    }
    const float rstd = 1.0f / sqrtf(block_sum(q, red) / (float)K + (float)eps);
#endif
    #pragma unroll
    for (int i = 0; i < PREP_PER; ++i) {
        const int k = tid + i * PREP_THREADS;
        if (k < K) {
            float z = v[i];
#if NORM
            z = (z - mean) * rstd * Gamma[k] + Beta[k];
#endif
#if SILU
            z = z / (1.0f + expf(-z));
#endif
            out[k] = z;
        }
    }
}
'''


def prep_kernel(k: int, norm: bool, silu: bool):
    threads = 128 if k <= 1024 else 256
    per = max(1, -(-k // threads))
    head = f'#define PREP_THREADS {threads}\n#define PREP_PER {per}\n#define NORM {int(norm)}\n#define SILU {int(silu)}\n'
    return nvrtc.load(head + PREP_SOURCE, 'row_prep', options=OPTIONS)[0], threads


def row_prep(x3: torch.Tensor, norm=None, silu: bool = False) -> torch.Tensor:
    """Contiguous FP32 [M, K]: rows of the [B, T, K] view x3, LayerNorm(norm) and/or SiLU applied."""
    b, t, k = x3.shape
    z = torch.empty((b * t, k), device=x3.device, dtype=torch.float32)
    if b * t:
        fn, threads = prep_kernel(k, norm is not None, silu)
        xb, xt, xk = x3.stride()
        gamma = norm.weight if norm is not None else z
        beta = norm.bias if norm is not None else z
        fn(grid=(b * t, 1, 1), block=(threads, 1, 1),
           args=[x3, t, xb, xt, xk, k, gamma, beta, float(norm.eps) if norm is not None else 0.0, z])
    return z


@dataclasses.dataclass(frozen=True)
class Config:
    bm: int
    bn: int
    bk: int
    tm: int
    tn: int
    split: int

    @property
    def threads(self) -> int:
        return (self.bm // self.tm) * (self.bn // self.tn)

    def shared_bytes(self) -> int:
        tiles = 2 * self.bk * ((self.bm + 4) + (self.bn + 4)) * 4
        reduce = self.bm * self.bn * 4 if self.split > 1 else 0
        return max(tiles, reduce)

    def valid(self) -> bool:
        return (self.tm in (4, 8) and self.tn in (4, 8) and self.bm % self.tm == 0 and self.bn % self.tn == 0
                and self.bk % 32 == 0 and 32 <= self.threads <= 1024 and self.threads % 32 == 0
                and 1 <= self.split <= 8 and self.shared_bytes() <= 99 * 1024
                and (self.bm // self.tm) % 1 == 0)


def source(config: Config, *, xmode: int, wfast: bool, has_bias: bool, act: int, in_act: int) -> str:
    defines = dict(BM=config.bm, BN=config.bn, BK=config.bk, TM=config.tm, TN=config.tn,
                   SPLIT=config.split, XMODE=xmode, WFAST=int(wfast), HAS_BIAS=int(has_bias),
                   ACT=act, IN_ACT=in_act)
    return ''.join(f'#define {k} {v}\n' for k, v in defines.items()) + SOURCE


OPTIONS = ('--std=c++17', '--fmad=true')


def precompile(config: Config, *, xmode: int, wfast: bool, has_bias: bool, act: int = 0, in_act: int = 0) -> int:
    """Fill the on-disk CUBIN cache without touching the GPU; returns the CUBIN size."""
    return len(nvrtc.disk_cubin(source(config, xmode=xmode, wfast=wfast, has_bias=has_bias, act=act, in_act=in_act),
                                'bitserial_ffma', '120a', OPTIONS))


def kernel(config: Config, *, xmode: int, wfast: bool, has_bias: bool, act: int = 0, in_act: int = 0):
    """Compiled kernel (cached per process and on disk). Load before CUDA-graph capture."""
    if not config.valid():
        raise ValueError(f'invalid config {config}')
    fn, _cubin = nvrtc.load(source(config, xmode=xmode, wfast=wfast, has_bias=has_bias, act=act, in_act=in_act),
                            'bitserial_ffma', options=OPTIONS)
    shared = config.shared_bytes()
    if shared >= 48 * 1024 and getattr(fn, '_max_shared_mem_bytes', 0) < shared:
        fn.set_shared_memory_config(shared)
    return fn


def launch(fn, config: Config, x3: torch.Tensor, T: int, w: torch.Tensor, kb: int, scale: torch.Tensor,
           bias: torch.Tensor | None, y: torch.Tensor, m: int, n: int, k: int) -> None:
    """x3: [B, T, K] view (any strides); y: contiguous [M, N]. Enqueued on the current stream."""
    xb, xt, xk = x3.stride()
    grid = ((n + config.bn - 1) // config.bn, (m + config.bm - 1) // config.bm, config.split)
    fn(grid=grid, block=(config.threads, 1, 1), shared_mem=config.shared_bytes(),
       args=[x3, T, xb, xt, xk, w, kb, scale, bias if bias is not None else scale, y, m, n, k])


