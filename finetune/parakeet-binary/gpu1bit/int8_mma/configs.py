"""Tile choices for b1_dot on the RTX 5090 (170 SMs).

TUNED_M holds the measured winners of tune.py per (N, K, M) (make_configs.py; raw sweeps under
/mnt/hd/.../gpu1bit/B/tune/: *-sweep4-* over ~600-1400 exact configs per entry, then *-pf-m* with the
prefetch flag on the 12 best). All are INT32-exact (bitwise the SPLIT == 1 result). A call with M rows
uses the tuned entry for (N, K) whose M is nearest in log scale; any other (N, K) uses `heuristic`.
Selection is a pure function of (M, N, K): no inference-time autotuning. TUNED_DEFERRED serves only
B-hybrid's deferred N = 1024 calls (hybrid.py).
"""
from __future__ import annotations

import math

from gpu1bit.int8_mma.kernels import DotConfig

SMS = 170

TUNED_M: dict[tuple[int, int, int], DotConfig] = {  # exact INT32 configs; sweep4 + PF retune (B/tune/*-pf-m*)
    (1024, 1024, 38): DotConfig(*[1, 4, 1, 1, 3, 1, 0, 2, 1]),  # w1x4x2_r1x1_s3_c1_pf hot 2.88 cold 2.92 us; R-expanded dot 3.01 / 3.59
    (1024, 1024, 126): DotConfig(*[1, 4, 1, 2, 4, 1, 0, 2, 1]),  # w1x4x2_r1x2_s4_c1_pf hot 3.53 cold 3.55 us; R-expanded dot 4.56 / 5.09
    (1024, 1024, 256): DotConfig(*[1, 8, 1, 2, 4, 1, 0, 1, 1]),  # w1x8_r1x2_s4_c1_pf hot 4.93 cold 5.01 us; R-expanded dot 7.05 / 7.72
    (1024, 1024, 368): DotConfig(*[1, 4, 1, 3, 3, 1, 0, 1, 1]),  # w1x4_r1x3_s3_c1_pf hot 6.13 cold 6.34 us; R-expanded dot 8.08 / 8.82
    (1024, 4096, 38): DotConfig(*[1, 4, 1, 3, 3, 8, 3, 1, 0]),  # w1x4_r1x3_s3_c8_p hot 5.14 cold 5.23 us; R-expanded dot 5.65 / 6.89
    (1024, 4096, 126): DotConfig(*[1, 4, 1, 4, 4, 4, 3, 1, 0]),  # w1x4_r1x4_s4_c4_p hot 9.09 cold 9.20 us; R-expanded dot 10.20 / 11.40
    (1024, 4096, 256): DotConfig(*[1, 4, 1, 4, 3, 2, 3, 1, 0]),  # w1x4_r1x4_s3_c2_p hot 15.21 cold 15.45 us; R-expanded dot 17.47 / 19.14
    (1024, 4096, 368): DotConfig(*[1, 8, 1, 3, 2, 1, 0, 2, 1]),  # w1x8x2_r1x3_s2_c1_pf hot 20.84 cold 21.17 us; R-expanded dot 24.04 / 27.95
    (2048, 1024, 38): DotConfig(*[1, 4, 1, 2, 4, 1, 0, 2, 1]),  # w1x4x2_r1x2_s4_c1_pf hot 3.52 cold 3.51 us; R-expanded dot 4.12 / 4.80
    (2048, 1024, 126): DotConfig(*[1, 8, 1, 2, 4, 1, 0, 1, 1]),  # w1x8_r1x2_s4_c1_pf hot 4.90 cold 4.94 us; R-expanded dot 6.13 / 7.47
    (2048, 1024, 256): DotConfig(*[1, 4, 1, 4, 2, 1, 0, 1, 1]),  # w1x4_r1x4_s2_c1_pf hot 7.58 cold 7.74 us; R-expanded dot 9.20 / 10.19
    (2048, 1024, 368): DotConfig(*[1, 8, 1, 3, 4, 1, 0, 1, 1]),  # w1x8_r1x3_s4_c1_pf hot 10.42 cold 10.68 us; R-expanded dot 13.52 / 14.64
    (3072, 1024, 38): DotConfig(*[1, 4, 1, 2, 4, 1, 0, 2, 1]),  # w1x4x2_r1x2_s4_c1_pf hot 3.53 cold 3.56 us; R-expanded dot 4.84 / 6.60
    (3072, 1024, 126): DotConfig(*[1, 4, 1, 3, 4, 1, 0, 1, 1]),  # w1x4_r1x3_s4_c1_pf hot 6.33 cold 6.45 us; R-expanded dot 7.98 / 9.46
    (3072, 1024, 256): DotConfig(*[1, 8, 1, 3, 2, 1, 0, 1, 1]),  # w1x8_r1x3_s2_c1_pf hot 10.52 cold 10.73 us; R-expanded dot 14.44 / 15.02
    (3072, 1024, 368): DotConfig(*[1, 8, 1, 4, 4, 1, 0, 1, 1]),  # w1x8_r1x4_s4_c1_pf hot 13.31 cold 13.48 us; R-expanded dot 15.40 / 17.56
    (4096, 1024, 38): DotConfig(*[1, 4, 1, 3, 3, 1, 0, 2, 1]),  # w1x4x2_r1x3_s3_c1_pf hot 4.31 cold 4.36 us; R-expanded dot 5.20 / 7.09
    (4096, 1024, 126): DotConfig(*[1, 4, 1, 4, 4, 1, 0, 1, 1]),  # w1x4_r1x4_s4_c1_pf hot 7.52 cold 7.63 us; R-expanded dot 8.72 / 9.85
    (4096, 1024, 256): DotConfig(*[1, 8, 1, 4, 4, 1, 0, 1, 1]),  # w1x8_r1x4_s4_c1_pf hot 13.09 cold 13.30 us; R-expanded dot 15.80 / 16.29
    (4096, 1024, 368): DotConfig(*[1, 8, 2, 3, 3, 1, 0, 1, 0]),  # w1x8_r2x3_s3_c1 hot 19.24 cold 19.53 us; R-expanded dot 21.51 / 24.10
}
# B-hybrid's deferred N = 1024 calls (xchg 4 FP32 slices for R-hybrid's row epilogue), measured as
# dot + _row_epilogue against the exact TUNED_M config + _row_epilogue(SPLIT = 1); None where the
# exact path is faster (tune.py --deferred).
TUNED_DEFERRED: dict[tuple[int, int, int], DotConfig | None] = {  # tune.py --deferred with PF (B/tune/*-deferredpf-*)
    (1024, 1024, 38): DotConfig(*[1, 4, 1, 3, 3, 4, 4, 1, 1]),  # w1x4_r1x3_s3_c4_d_pf 4.03 us vs exact + epilogue 4.91 (w1x4x2_r1x1_s3_c1_pf)
    (1024, 1024, 126): DotConfig(*[1, 4, 1, 4, 3, 4, 4, 1, 1]),  # w1x4_r1x4_s3_c4_d_pf 5.45 us vs exact + epilogue 5.78 (w1x4x2_r1x2_s4_c1_pf)
    (1024, 1024, 256): DotConfig(*[1, 4, 1, 4, 2, 2, 4, 1, 1]),  # w1x4_r1x4_s2_c2_d_pf 7.58 us vs exact + epilogue 7.90 (w1x8_r1x2_s4_c1_pf)
    (1024, 1024, 368): None,  # exact 8.80 us beats deferred 9.50
    (1024, 4096, 38): DotConfig(*[1, 4, 1, 3, 3, 8, 4, 1, 1]),  # w1x4_r1x3_s3_c8_d_pf 6.16 us vs exact + epilogue 7.17 (w1x4_r1x3_s3_c8_p)
    (1024, 4096, 126): DotConfig(*[1, 4, 1, 4, 4, 4, 4, 1, 1]),  # w1x4_r1x4_s4_c4_d_pf 10.15 us vs exact + epilogue 11.49 (w1x4_r1x4_s4_c4_p)
    (1024, 4096, 256): DotConfig(*[1, 4, 1, 4, 4, 2, 4, 1, 1]),  # w1x4_r1x4_s4_c2_d_pf 16.74 us vs exact + epilogue 17.88 (w1x4_r1x4_s3_c2_p)
    (1024, 4096, 368): DotConfig(*[1, 8, 2, 4, 4, 3, 4, 1, 1]),  # w1x8_r2x4_s4_c3_d_pf 22.49 us vs exact + epilogue 23.48 (w1x8x2_r1x3_s2_c1_pf)
}


def heuristic(m: int, n: int, k: int) -> DotConfig:
    rm = min(4, -(-m // 16))
    wn, rn = 4, 2
    base = DotConfig(1, wn, rm, rn, 2, 1)
    ctas = base.grid(m, n)[0] * base.grid(m, n)[1]
    kb = -(-k // 128)
    cluster = max(1, min(8, kb, SMS // max(ctas, 1)))
    stages = 3 if -(-kb // cluster) > 2 else 2
    for s in (stages, 2):
        cfg = DotConfig(1, wn, rm, rn, s, cluster, 1 if cluster > 1 else 0)
        if cfg.shared_bytes <= 101376:
            return cfg
    return DotConfig(1, wn, rm, rn, 2, 1)


def choose(m: int, n: int, k: int) -> DotConfig:
    tuned = [mm for (nn, kk, mm) in TUNED_M if (nn, kk) == (n, k)]
    if not tuned or m < 1:
        return heuristic(m, n, k)
    nearest = min(tuned, key=lambda mm: (abs(math.log(m / mm)), mm))
    return TUNED_M[n, k, nearest]


def choose_deferred(m: int, n: int, k: int) -> DotConfig | None:
    """The deferred-split config for (M, N, K), or None to keep the exact in-kernel finish."""
    tuned = [mm for (nn, kk, mm) in TUNED_DEFERRED if (nn, kk) == (n, k)]
    if not tuned or m < 1:
        return None
    nearest = min(tuned, key=lambda mm: (abs(math.log(m / mm)), mm))
    return TUNED_DEFERRED[n, k, nearest]
