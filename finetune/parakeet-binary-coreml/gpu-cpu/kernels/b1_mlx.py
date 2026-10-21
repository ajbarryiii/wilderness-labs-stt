"""The 1-bit GEMMs of metal/b1gemm.metal as MLX custom kernels (mx.fast.metal_kernel), so the full MLX encoder can
run on 1-bit weights.

y [M, N] (FP16) = FP16(s)[n] * sum_k x[m, k] * c[n, k], c = +1 for bit 1, -1 for bit 0; FP32 accumulation.
words uint32 [N, K / 32] (= the export bytes as little-endian words), s16 FP16 [N]. K % 32 == 0, N % BN == 0.

Kernels (same code as b1gemm.metal, bit-exact there and here):
  lut  group-of-4 lookup tables in threadgroup memory (the fastest family in results/metal_bench.json)
  mma  simdgroup_matrix 8x8 FP32 MMA with the weight fragments decoded from the packed bits
Tile per (N, K, M): TILES (the fastest b1_lut tile per shape and bucket from metal_bench.json), else DEFAULT.
"""
from __future__ import annotations

import mlx.core as mx

HEADER = """
static inline void frag_coord(ushort lane, thread ushort &fm, thread ushort &fn) {
    const ushort qid = lane / 4;
    fm = (qid & 4) + ((lane / 2) % 4);
    fn = (qid & 2) * 2 + (lane % 2) * 2;
}
"""

LUT_SOURCE = """
    constexpr int BK = 32;
    constexpr int BN = 32 * TN * NSG;
    constexpr int NT = 32 * NSG;
    constexpr int G = BK / 4;
    threadgroup float T[BMF][G][16];

    const int M = x_shape[0], K = x_shape[1], N = w_shape[0], KW = K / 32;
    const ushort sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const ushort tid = thread_index_in_threadgroup;
    const int m0 = threadgroup_position_in_grid.y * BMF, n0 = threadgroup_position_in_grid.x * BN;
    const int nbase = n0 + sg * 32 * TN;
    float acc[BMF][TN];
    for (int i = 0; i < BMF; ++i)
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;
    const device uint32_t *wrow[TN];
    for (int j = 0; j < TN; ++j)
        wrow[j] = w + (size_t)(nbase + j * 32 + lane) * KW;

    for (int kw = 0; kw < KW; ++kw) {
        for (int c = tid; c < BMF * G; c += NT) {
            const int m = c / G, g = c % G;
            const int gm = m0 + m;
            half4 h = gm < M ? *(const device half4 *)(x + (size_t)gm * K + kw * BK + g * 4) : half4(0.0h);
            const float4 xv = float4(h);
            const float lo[4] = {-xv.x - xv.y, xv.x - xv.y, -xv.x + xv.y, xv.x + xv.y};
            const float hi[4] = {-xv.z - xv.w, xv.z - xv.w, -xv.z + xv.w, xv.z + xv.w};
            for (int e = 0; e < 16; ++e)
                T[m][g][e] = lo[e & 3] + hi[e >> 2];
        }
        uint wv[TN];
        for (int j = 0; j < TN; ++j)
            wv[j] = wrow[j][kw];
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (int g = 0; g < G; ++g) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const uint e = (wv[j] >> (4 * g)) & 15u;
#pragma unroll
                for (int i = 0; i < BMF; ++i)
                    acc[i][j] += T[i][g][e];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int j = 0; j < TN; ++j) {
        const int col = nbase + j * 32 + lane;
        const float sc = float(s[col]);
        for (int i = 0; i < BMF; ++i) {
            const int row = m0 + i;
            if (row < M)
                y[(size_t)row * N + col] = half(acc[i][j] * sc);
        }
    }
"""

MMA_SOURCE = """
    constexpr int BK = 32;
    constexpr int BM = 8 * FM * SGM;
    constexpr int BN = 8 * FN * SGN;
    constexpr int NT = 32 * SGM * SGN;
    threadgroup half As[BM][BK + 8];

    const int M = x_shape[0], K = x_shape[1], N = w_shape[0], KW = K / 32;
    const ushort sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const ushort tid = thread_index_in_threadgroup;
    const int sgm = sg / SGN, sgn = sg % SGN;
    const int m0 = threadgroup_position_in_grid.y * BM, n0 = threadgroup_position_in_grid.x * BN;
    const int msg = m0 + sgm * FM * 8;
    const int mvalid = (M - msg + 7) / 8;
    ushort fm, fn;
    frag_coord(lane, fm, fn);

    simdgroup_matrix<float, 8, 8> acc[FM][FN];
    for (int i = 0; i < FM; ++i)
        for (int j = 0; j < FN; ++j)
            acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
    const device uint32_t *wrow0[FN];
    const device uint32_t *wrow1[FN];
    for (int j = 0; j < FN; ++j) {
        const int n = n0 + sgn * FN * 8 + j * 8 + fn;
        wrow0[j] = w + (size_t)n * KW;
        wrow1[j] = w + (size_t)(n + 1) * KW;
    }
    for (int kw = 0; kw < KW; ++kw) {
        for (int c = tid; c < BM * (BK / 4); c += NT) {
            const int r = c / (BK / 4), cc = (c % (BK / 4)) * 4;
            const int gm = m0 + r;
            half4 v = gm < M ? *(const device half4 *)(x + (size_t)gm * K + kw * BK + cc) : half4(0.0h);
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
        const float s0 = float(s[col]), s1 = float(s[col + 1]);
        for (int i = 0; i < FM; ++i) {
            const int row = msg + i * 8 + fm;
            if (row < M) {
                const auto v = acc[i][j].thread_elements();
                y[(size_t)row * N + col] = half(v[0] * s0);
                y[(size_t)row * N + col + 1] = half(v[1] * s1);
            }
        }
    }
"""

LUT = mx.fast.metal_kernel(name="b1_lut", input_names=["x", "w", "s"], output_names=["y"], source=LUT_SOURCE,
                           header=HEADER, ensure_row_contiguous=True)
MMA = mx.fast.metal_kernel(name="b1_mma", input_names=["x", "w", "s"], output_names=["y"], source=MMA_SOURCE,
                           header=HEADER, ensure_row_contiguous=True)

DEFAULT = ("lut", (8, 1, 4))
TILES: dict = {}  # (N, K, M) -> ("lut", (BMF, TN, NSG)) or ("mma", (FM, FN, SGM, SGN))


def set_tiles(table: dict) -> None:
    TILES.clear()
    TILES.update(table)


def load_tiles(bench_json: str, family: str = "lut") -> dict:
    """Fastest tile of one kernel family per (N, K, M) from b1bench's bench output."""
    import json

    best: dict = {}
    for r in json.load(open(bench_json))["rows"]:
        fam = r["variant"].split("_")[0]
        if fam != family:
            continue
        key = (r["N"], r["K"], r["M"])
        if key not in best or r["us"] < best[key][0]:
            best[key] = (r["us"], (fam, tuple(int(v) for v in r["variant"].split("_")[1:])))
    set_tiles({k: v[1] for k, v in best.items()})
    return dict(TILES)


def b1_linear(x, words, s16, n_in: int):
    m = x.shape[0]
    n = words.shape[0]
    if x.dtype != mx.float16:
        raise TypeError("b1_linear expects FP16 activations")
    fam, t = TILES.get((n, n_in, m), DEFAULT)
    if fam == "lut":
        bmf, tn, nsg = t
        bm, bn, nt = bmf, 32 * tn * nsg, 32 * nsg
        kern, tmpl = LUT, [("BMF", bmf), ("TN", tn), ("NSG", nsg)]
    else:
        fm, fn, sgm, sgn = t
        bm, bn, nt = 8 * fm * sgm, 8 * fn * sgn, 32 * sgm * sgn
        kern, tmpl = MMA, [("FM", fm), ("FN", fn), ("SGM", sgm), ("SGN", sgn)]
    return kern(inputs=[x, words, s16], template=tmpl, grid=((n // bn) * nt, (m + bm - 1) // bm, 1),
                threadgroup=(nt, 1, 1), output_shapes=[(m, n)], output_dtypes=[mx.float16])[0]
