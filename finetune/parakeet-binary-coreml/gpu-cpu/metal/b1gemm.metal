// 1-bit-weight x FP16-activation GEMM for MB1 on the Apple GPU (microbenchmark; compiled at run time).
//
// Y[m, n] = s[n] * sum_k X[m, k] * c[n, k],  c = +1 for bit 1, -1 for bit 0.
//   X  half  [M, K] row-major
//   W  uint  [N, K / 32]: code k of row n is bit k % 32 of word k / 32 (= the parakeet-binary-v1 bytes, LSB first,
//            read as little-endian uint32)
//   S  half  [N]  (FP16(s), the row scale)
//   Y  half or float [M, N]
// FP32 accumulation. Requirements: K % 32 == 0, N % BN == 0 (true for every MB1 projection); any M.
//
// Variants:
//   b1_mma   simdgroup_matrix 8x8 FP32 MMA. A (X) tiles go through threadgroup memory; the B (weight) fragments are
//            decoded straight from the packed words into each thread's two fragment elements (no threadgroup
//            traffic for weights): element (row k, col n) = as_float(0xBF800000 ^ (bit << 31)) = -1.0 / +1.0.
//   b1_simt  plain FMA micro-tiles (TM rows x TN cols per thread), sign decoded once per (n, k) and reused over TM.
//   b1_lut   group-of-4 lookup tables (T-MAC style): per frame and 4 inputs the 16 signed sums are built in
//            threadgroup memory; every output then needs one table read + one add per 4 inputs.
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint M;
    uint N;
    uint K;
};

#define BK 32
#define APAD 8

// Fragment coordinates of simdgroup_matrix<T, 8, 8> element pairs (as MLX's steel BaseMMAFrag::get_coord):
// lane holds (fm, fn) and (fm, fn + 1).
static inline void frag_coord(ushort lane, thread ushort &fm, thread ushort &fn) {
    const ushort qid = lane / 4;
    fm = (qid & 4) + ((lane / 2) % 4);
    fn = (qid & 2) * 2 + (lane % 2) * 2;
}

template <int FM, int FN, int SGM, int SGN, typename OutT>
[[kernel]] void b1_mma(device const half *X [[buffer(0)]],
                       device const uint *W [[buffer(1)]],
                       device const half *S [[buffer(2)]],
                       device OutT *Y [[buffer(3)]],
                       constant Params &p [[buffer(4)]],
                       uint3 tg [[threadgroup_position_in_grid]],
                       ushort sg [[simdgroup_index_in_threadgroup]],
                       ushort lane [[thread_index_in_simdgroup]],
                       ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int BM = 8 * FM * SGM;
    constexpr int BN = 8 * FN * SGN;
    constexpr int NT = 32 * SGM * SGN;
    threadgroup half As[BM][BK + APAD];

    const int M = p.M, N = p.N, K = p.K, KW = K / 32;
    const int sgm = sg / SGN, sgn = sg % SGN;
    const int m0 = tg.y * BM, n0 = tg.x * BN;
    const int msg = m0 + sgm * FM * 8;            // first row of this simdgroup
    const int mvalid = (M - msg + 7) / 8;         // valid 8-row fragments (uniform per simdgroup)
    ushort fm, fn;
    frag_coord(lane, fm, fn);

    simdgroup_matrix<float, 8, 8> acc[FM][FN];
    for (int i = 0; i < FM; ++i)
        for (int j = 0; j < FN; ++j)
            acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    device const uint *wrow0[FN];
    device const uint *wrow1[FN];
    for (int j = 0; j < FN; ++j) {
        const int n = n0 + sgn * FN * 8 + j * 8 + fn;
        wrow0[j] = W + (size_t)n * KW;
        wrow1[j] = W + (size_t)(n + 1) * KW;
    }

    for (int kw = 0; kw < KW; ++kw) {
        for (int c = tid; c < BM * (BK / 4); c += NT) {
            const int r = c / (BK / 4), cc = (c % (BK / 4)) * 4;
            const int gm = m0 + r;
            half4 v = gm < M ? *(device const half4 *)(X + (size_t)gm * K + kw * BK + cc) : half4(0.0h);
            *(threadgroup half4 *)(&As[r][cc]) = v;
        }
        uint w0[FN], w1[FN];
        for (int j = 0; j < FN; ++j) {
            w0[j] = wrow0[j][kw] >> fm;
            w1[j] = wrow1[j][kw] >> fm;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (mvalid > 0) {
#pragma unroll
            for (int kk = 0; kk < BK; kk += 8) {
                simdgroup_matrix<float, 8, 8> a[FM], b[FN];
#pragma unroll
                for (int i = 0; i < FM; ++i) {
                    const half2 h = *(threadgroup const half2 *)(&As[sgm * FM * 8 + i * 8 + fm][kk + fn]);
                    a[i].thread_elements()[0] = float(h.x);
                    a[i].thread_elements()[1] = float(h.y);
                }
#pragma unroll
                for (int j = 0; j < FN; ++j) {
                    b[j].thread_elements()[0] = as_type<float>(0xBF800000u ^ (((w0[j] >> kk) & 1u) << 31));
                    b[j].thread_elements()[1] = as_type<float>(0xBF800000u ^ (((w1[j] >> kk) & 1u) << 31));
                }
#pragma unroll
                for (int i = 0; i < FM; ++i) {
                    if (i < mvalid) {
#pragma unroll
                        for (int j = 0; j < FN; ++j)
                            simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int j = 0; j < FN; ++j) {
        const int col = n0 + sgn * FN * 8 + j * 8 + fn;
        const float s0 = float(S[col]), s1 = float(S[col + 1]);
        for (int i = 0; i < FM; ++i) {
            const int row = msg + i * 8 + fm;
            if (row < M) {
                const auto v = acc[i][j].thread_elements();
                Y[(size_t)row * N + col] = OutT(v[0] * s0);
                Y[(size_t)row * N + col + 1] = OutT(v[1] * s1);
            }
        }
    }
}

// SIMT: thread (lane) owns TN columns n = nbase + j * 32 + lane and TM rows; the simdgroup shares the rows, so
// activation reads from threadgroup memory are broadcasts.
template <int TM, int TN, int SGM, int SGN, typename OutT>
[[kernel]] void b1_simt(device const half *X [[buffer(0)]],
                        device const uint *W [[buffer(1)]],
                        device const half *S [[buffer(2)]],
                        device OutT *Y [[buffer(3)]],
                        constant Params &p [[buffer(4)]],
                        uint3 tg [[threadgroup_position_in_grid]],
                        ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]],
                        ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int BM = TM * SGM;
    constexpr int BN = 32 * TN * SGN;
    constexpr int NT = 32 * SGM * SGN;
    threadgroup float As[BM][BK];

    const int M = p.M, N = p.N, K = p.K, KW = K / 32;
    const int sgm = sg / SGN, sgn = sg % SGN;
    const int m0 = tg.y * BM, n0 = tg.x * BN;
    const int rbase = sgm * TM;
    const int nbase = n0 + sgn * 32 * TN;
    float acc[TM][TN];
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;
    device const uint *wrow[TN];
    for (int j = 0; j < TN; ++j)
        wrow[j] = W + (size_t)(nbase + j * 32 + lane) * KW;

    for (int kw = 0; kw < KW; ++kw) {
        for (int c = tid; c < BM * (BK / 4); c += NT) {
            const int r = c / (BK / 4), cc = (c % (BK / 4)) * 4;
            const int gm = m0 + r;
            half4 v = gm < M ? *(device const half4 *)(X + (size_t)gm * K + kw * BK + cc) : half4(0.0h);
            *(threadgroup float4 *)(&As[r][cc]) = float4(v);
        }
        uint w[TN];
        for (int j = 0; j < TN; ++j)
            w[j] = wrow[j][kw];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (m0 + rbase < M) {
#pragma unroll
            for (int k = 0; k < BK; ++k) {
                float sgnv[TN];
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    sgnv[j] = as_type<float>(0xBF800000u ^ (((w[j] >> k) & 1u) << 31));
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    const float x = As[rbase + i][k];
#pragma unroll
                    for (int j = 0; j < TN; ++j)
                        acc[i][j] = fma(x, sgnv[j], acc[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int j = 0; j < TN; ++j) {
        const int col = nbase + j * 32 + lane;
        const float s = float(S[col]);
        for (int i = 0; i < TM; ++i) {
            const int row = m0 + rbase + i;
            if (row < M)
                Y[(size_t)row * N + col] = OutT(acc[i][j] * s);
        }
    }
}

// LUT (g = 4): per K chunk of 32 inputs (8 groups) and per frame, T[m][g][e] = sum_j (bit j of e ? +x : -x)[4g + j],
// FP32 entries; 16 entries x 4 B = 64 B per (m, g), so lanes reading different entries hit different banks and
// equal entries broadcast. Thread owns TN columns (n = nbase + j * 32 + lane) and all BM frames.
template <int BMF, int TN, int NSG, typename OutT>
[[kernel]] void b1_lut(device const half *X [[buffer(0)]],
                       device const uint *W [[buffer(1)]],
                       device const half *S [[buffer(2)]],
                       device OutT *Y [[buffer(3)]],
                       constant Params &p [[buffer(4)]],
                       uint3 tg [[threadgroup_position_in_grid]],
                       ushort sg [[simdgroup_index_in_threadgroup]],
                       ushort lane [[thread_index_in_simdgroup]],
                       ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int BN = 32 * TN * NSG;
    constexpr int NT = 32 * NSG;
    constexpr int G = BK / 4;  // groups per chunk
    threadgroup float T[BMF][G][16];

    const int M = p.M, N = p.N, K = p.K, KW = K / 32;
    const int m0 = tg.y * BMF, n0 = tg.x * BN;
    const int nbase = n0 + sg * 32 * TN;
    float acc[BMF][TN];
    for (int i = 0; i < BMF; ++i)
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;
    device const uint *wrow[TN];
    for (int j = 0; j < TN; ++j)
        wrow[j] = W + (size_t)(nbase + j * 32 + lane) * KW;

    for (int kw = 0; kw < KW; ++kw) {
        // build: one (m, g) table per thread-iteration (16 entries from 4 inputs)
        for (int c = tid; c < BMF * G; c += NT) {
            const int m = c / G, g = c % G;
            const int gm = m0 + m;
            half4 h = gm < M ? *(device const half4 *)(X + (size_t)gm * K + kw * BK + g * 4) : half4(0.0h);
            const float4 x = float4(h);
            const float a0 = -x.x - x.y, a1 = x.x - x.y, a2 = -x.x + x.y, a3 = x.x + x.y;  // bits 0..1
            const float b0 = -x.z - x.w, b1 = x.z - x.w, b2 = -x.z + x.w, b3 = x.z + x.w;  // bits 2..3
            const float lo[4] = {a0, a1, a2, a3};
            const float hi[4] = {b0, b1, b2, b3};
            for (int e = 0; e < 16; ++e)
                T[m][g][e] = lo[e & 3] + hi[e >> 2];
        }
        uint w[TN];
        for (int j = 0; j < TN; ++j)
            w[j] = wrow[j][kw];
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (int g = 0; g < G; ++g) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const uint e = (w[j] >> (4 * g)) & 15u;
#pragma unroll
                for (int i = 0; i < BMF; ++i)
                    acc[i][j] += T[i][g][e];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int j = 0; j < TN; ++j) {
        const int col = nbase + j * 32 + lane;
        const float s = float(S[col]);
        for (int i = 0; i < BMF; ++i) {
            const int row = m0 + i;
            if (row < M)
                Y[(size_t)row * N + col] = OutT(acc[i][j] * s);
        }
    }
}

#define MMA(a, b, c, d, OUTT, sfx)                                                                                   \
    template [[host_name("b1_mma_" #a "_" #b "_" #c "_" #d "_" #sfx)]] [[kernel]] void b1_mma<a, b, c, d, OUTT>(     \
        device const half *, device const uint *, device const half *, device OUTT *, constant Params &, uint3,      \
        ushort, ushort, ushort);
#define SIMT(a, b, c, d, OUTT, sfx)                                                                                  \
    template [[host_name("b1_simt_" #a "_" #b "_" #c "_" #d "_" #sfx)]] [[kernel]] void b1_simt<a, b, c, d, OUTT>(   \
        device const half *, device const uint *, device const half *, device OUTT *, constant Params &, uint3,      \
        ushort, ushort, ushort);
#define LUT(a, b, c, OUTT, sfx)                                                                                      \
    template [[host_name("b1_lut_" #a "_" #b "_" #c "_" #sfx)]] [[kernel]] void b1_lut<a, b, c, OUTT>(               \
        device const half *, device const uint *, device const half *, device OUTT *, constant Params &, uint3,      \
        ushort, ushort, ushort);

#define BOTH_MMA(a, b, c, d) MMA(a, b, c, d, half, h) MMA(a, b, c, d, float, f)
#define BOTH_SIMT(a, b, c, d) SIMT(a, b, c, d, half, h) SIMT(a, b, c, d, float, f)
#define BOTH_LUT(a, b, c) LUT(a, b, c, half, h) LUT(a, b, c, float, f)

// FM, FN, SGM, SGN  (BM = 8 FM SGM, BN = 8 FN SGN)
BOTH_MMA(1, 4, 2, 2)
BOTH_MMA(2, 4, 1, 2)
BOTH_MMA(2, 4, 2, 2)
BOTH_MMA(2, 4, 1, 4)
BOTH_MMA(4, 4, 1, 2)
BOTH_MMA(4, 4, 2, 2)
BOTH_MMA(4, 2, 2, 2)
BOTH_MMA(2, 8, 2, 1)
BOTH_MMA(4, 8, 1, 2)
BOTH_MMA(8, 4, 1, 2)
// TM, TN, SGM, SGN  (BM = TM SGM, BN = 32 TN SGN)
BOTH_SIMT(8, 4, 1, 2)
BOTH_SIMT(8, 4, 2, 2)
BOTH_SIMT(8, 8, 1, 1)
BOTH_SIMT(16, 4, 1, 2)
BOTH_SIMT(4, 8, 2, 2)
// BMF, TN, NSG  (BM = BMF, BN = 32 TN NSG)
BOTH_LUT(8, 1, 4)
BOTH_LUT(8, 2, 4)
BOTH_LUT(16, 1, 4)
BOTH_LUT(4, 2, 4)
