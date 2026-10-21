"""Agent A (plans/gpu-kernel.md): Triton kernels that run MB1's 1-bit weights on float tensor cores.

Resident weights (built once by layer.prepare): uint8 `bits_t` [ceil(K/8), N], MB1's codes transposed:
byte (kb, n) holds codes k = 8kb .. 8kb+7 of output row n, LSB first, bit 1 = +1, bit 0 = -1. Exactly
MB1's packed code bytes, so the weight storage is 1.000x; a [BK/8, BN] tile load is coalesced along N.
Unused tail bits are 0 and only ever multiply masked (zero) activations.

Decode (in registers): bit -> exact +-1 in BF16/FP16 by integer ops and a bitcast
((bit ^ 1) << 15 | 1.0's bits), either from the byte tile reshaped [BK/8, 8, BN] -> [BK, BN] or
elementwise from a [BK, BN] tile of repeated byte addresses (EW, no reshape). Triton then moves the
decoded tile to the dot-operand layout through shared memory.

Arithmetic (FP32 in, FP32 out, FP32 accumulation; tensor-core MMA m16n8k16 with FP32 accumulate):
- "bf16x3" (the A3-compliant candidate): each FP32 activation is split into three non-overlapping
  BF16 parts hi + mid + lo (M1's compact scheme). Per K tile, hi's products go into a fresh
  tensor-core accumulator and mid + lo's into a second one (chains of BK/16 MMAs); the hi tile sums
  are added to the FP32 running sum with a compensated (Kahan/TwoSum) add, mid + lo with a plain add.
  Root cause this fixes (NOTES, A 2026-10-20 08:39): plain FP32 running sums of large per-tile partials
  (positional-embedding rows, linear_pos) lose ~0.5 ulp of the large sum per add; M1's R-bf16x3 only
  passes the A3 gate because its split-K 8 keeps per-part sums short. "bf16x3l" keeps M1's two
  long-lived tensor-core accumulators (comparison only).
- "fp16x2": two FP16 parts with a per-(row, K-tile) power-of-two scale (scaled row max in
  [2^14, 2^15): no overflow, residual normal down to 2^-25 of the row max); 2/3 of bf16x3's MMAs but
  22 of FP32's 24 significand bits, so a quality trade-off (fails A3's synthetic single_nonzero by 6%).
- "...p" modes: a prep kernel (_split_rows: optional LayerNorm and input SiLU, then the split) writes
  the parts once as row-major planes, and _pgemm runs a pure plane x decoded-bits GEMM; avoids
  re-splitting A in every N-tile CTA (pays one extra launch at small M).
Split-K: SPLIT CTAs per output tile (K tiles part, part + SPLIT, ...), FP32 partials reduced in fixed
order with a compensated sum: "workspace" (separate _finish kernel, used by every tuned config) or
"serial" (the last-arriving CTA reduces after an acq_rel counter; one kernel, measured no faster).
Epilogue: y = acc * alpha[n] (+ bias[n]) (optional SiLU) stored through output strides (linear
[.., N] and conv [B, N, T] without copies). LayerNorm fusion for the in-kernel modes: _rowstats
(mean, rstd per row) then normalization of each A tile inside the GEMM.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_pm1(w, BK: tl.constexpr, BN: tl.constexpr, FP16: tl.constexpr):
    """uint8 [BK//8, BN] packed codes -> [BK, BN] exact +-1 in BF16 or FP16 (bitcast, no I2F)."""
    lane = tl.arange(0, 8)
    bits = (w.to(tl.int32)[:, None, :] >> lane[None, :, None]) & 1
    bits = tl.reshape(bits, (BK, BN))
    one: tl.constexpr = 0x3C00 if FP16 else 0x3F80   # 1.0 in FP16 / BF16; bit 15 is the sign
    h = (((bits ^ 1) << 15) | one).to(tl.int16)
    return h.to(tl.float16 if FP16 else tl.bfloat16, bitcast=True)


@triton.jit
def _decode_pm1_ew(w, k, FP16: tl.constexpr):
    """Elementwise decode: w uint8 [BK, BN] (byte k//8 of column n at every row k) -> +-1 at bit k % 8.

    No reshape, so Triton can move the conversion to the dot-operand layout before these ops (the
    byte tile, not the decoded FP16 tile, goes through shared memory)."""
    bits = (w.to(tl.int32) >> (k % 8)[:, None]) & 1
    one: tl.constexpr = 0x3C00 if FP16 else 0x3F80
    h = (((bits ^ 1) << 15) | one).to(tl.int16)
    return h.to(tl.float16 if FP16 else tl.bfloat16, bitcast=True)


@triton.jit
def _epilogue(acc, S, Bias, Y, m, n, M, N: tl.constexpr, T, Y_B, Y_T, Y_N,
              HAS_BIAS: tl.constexpr, ACT: tl.constexpr):
    acc = acc * tl.load(S + n, n < N, other=0.)[None, :]
    if HAS_BIAS:
        acc += tl.load(Bias + n, n < N, other=0.)[None, :]
    if ACT == "silu":
        acc = acc / (1.0 + tl.exp(-acc))
    yrow = (m // T) * Y_B + (m % T) * Y_T
    tl.store(Y + yrow[:, None] + n[None, :] * Y_N, acc, (m[:, None] < M) & (n[None, :] < N))


@triton.jit
def _rowstats(X, MEAN, RSTD, M, T, X_B, X_T, X_K, K: tl.constexpr, FLAT: tl.constexpr, EPS: tl.constexpr,
              BR: tl.constexpr, BLOCK: tl.constexpr, PDL_TRIGGER: tl.constexpr = False):
    """LayerNorm statistics per row (M1 _quantize's formula: two-pass mean / centered variance)."""
    if PDL_TRIGGER:
        tl.extra.cuda.gdc_launch_dependents()
    m = tl.program_id(0) * BR + tl.arange(0, BR)
    k = tl.arange(0, BLOCK)
    if FLAT:
        xrow = m * X_T
    else:
        xrow = (m // T) * X_B + (m % T) * X_T
    mask = (m[:, None] < M) & (k[None, :] < K)
    x = tl.load(X + xrow[:, None] + k[None, :] * X_K, mask, other=0.)
    mean = tl.sum(x, 1) / K
    d = tl.where(mask, x - mean[:, None], 0.)
    var = tl.sum(d * d, 1) / K
    tl.store(MEAN + m, mean, m < M)
    tl.store(RSTD + m, tl.rsqrt(var + EPS), m < M)


@triton.jit
def _bgemm(X, W, S, Bias, Y, P, C,
           M, T, X_B, X_T, X_K, Y_B, Y_T, Y_N,
           N: tl.constexpr, K: tl.constexpr, HAS_BIAS: tl.constexpr, MODE: tl.constexpr,
           ACT: tl.constexpr, IN_ACT: tl.constexpr, FLAT: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr,
           REDUCE: tl.constexpr, EW: tl.constexpr = False, KAHAN: tl.constexpr = True,
           NORM: tl.constexpr = False, MEAN=None, RSTD=None, NW=None, NB=None,
           PDL_WAIT: tl.constexpr = False, PDL_TRIGGER: tl.constexpr = False):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    part = tl.program_id(2)
    if PDL_WAIT:   # launched with programmatic dependent launch after _rowstats: wait for its writes
        tl.extra.cuda.gdc_wait()
    m = pid_m * BM + tl.arange(0, BM)
    n = pid_n * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    kb = tl.arange(0, BK // 8)
    if FLAT:   # x viewable as 2-D [M, K] with row stride X_T (B == 1 or B-stride == T * X_T)
        xrow = m * X_T
    else:
        xrow = (m // T) * X_B + (m % T) * X_T
    KB: tl.constexpr = (K + 7) // 8
    TILES: tl.constexpr = (K + BK - 1) // BK
    acc = tl.zeros((BM, BN), tl.float32)
    acc2 = tl.zeros((BM, BN), tl.float32)
    comp = tl.zeros((BM, BN), tl.float32)      # KAHAN: compensation of the hi running sum
    if NORM:
        mean = tl.load(MEAN + m, m < M, other=0.)
        rstd = tl.load(RSTD + m, m < M, other=0.)
    for tile in range(part, TILES, SPLIT):
        k = tile * BK + kk
        amask = (m[:, None] < M) & (k[None, :] < K)
        a = tl.load(X + xrow[:, None] + k[None, :] * X_K, amask, other=0.)
        if NORM:     # LayerNorm fused on the A tile (M1's formula: centered * rstd * w + b)
            a = (a - mean[:, None]) * rstd[:, None] * tl.load(NW + k, k < K, other=0.)[None, :] \
                + tl.load(NB + k, k < K, other=0.)[None, :]
            a = tl.where(amask, a, 0.)
        if IN_ACT == "silu":
            a = a * tl.sigmoid(a)              # as M1's _quantize INPUT_SILU
        if EW:
            w = tl.load(W + (k // 8)[:, None] * N + n[None, :], (k[:, None] < K) & (n[None, :] < N), other=0)
        else:
            wb = tile * (BK // 8) + kb
            w = tl.load(W + wb[:, None] * N + n[None, :], (wb[:, None] < KB) & (n[None, :] < N), other=0)
        if MODE == "fp16x2":
            b = _decode_pm1_ew(w, k, True) if EW else _decode_pm1(w, BK, BN, True)
            amax = tl.max(tl.abs(a), axis=1)
            e = (amax.to(tl.int32, bitcast=True) >> 23) & 255
            sf = tl.minimum(268 - e, 253)                      # 2^(14 - exponent), clamped normal
            sc = (sf << 23).to(tl.float32, bitcast=True)
            inv = ((254 - sf) << 23).to(tl.float32, bitcast=True)
            xs = a * sc[:, None]
            hi = xs.to(tl.float16)
            lo = (xs - hi.to(tl.float32)).to(tl.float16)
            # fresh tensor-core accumulators per K tile (chains of BK/16 MMAs), hi and lo apart,
            # rescaled exactly (power of two) and added with round-to-nearest FP32 FMAs
            th = tl.dot(hi, b) * inv[:, None]
            if KAHAN:
                yk = th - comp
                sk = acc + yk
                comp = (sk - acc) - yk
                acc = sk
            else:
                acc += th
            acc2 += tl.dot(lo, b) * inv[:, None]
        else:
            b = _decode_pm1_ew(w, k, False) if EW else _decode_pm1(w, BK, BN, False)
            hi = a.to(tl.bfloat16)
            r = a - hi.to(tl.float32)
            mid = r.to(tl.bfloat16)
            lo = (r - mid.to(tl.float32)).to(tl.bfloat16)
            if MODE == "bf16x3t":
                t = tl.dot(lo, b)
                t = tl.dot(mid, b, t)
                t = tl.dot(hi, b, t)
                acc += t
            elif MODE == "bf16x3l":   # M1's structure: two long-lived tensor-core accumulators
                acc2 = tl.dot(lo, b, acc2)
                acc2 = tl.dot(mid, b, acc2)
                acc = tl.dot(hi, b, acc)
            else:                     # bf16x3: short chains, hi apart from mid + lo, RN adds
                t2 = tl.dot(lo, b)
                acc2 += tl.dot(mid, b, t2)
                th = tl.dot(hi, b)
                if KAHAN:
                    yk = th - comp
                    sk = acc + yk
                    comp = (sk - acc) - yk
                    acc = sk
                else:
                    acc += th
    if PDL_TRIGGER:   # let the split-K finish kernel launch while this grid drains
        tl.extra.cuda.gdc_launch_dependents()
    if KAHAN:
        acc2 -= comp
    if MODE != "bf16x3t":
        acc += acc2
    if SPLIT == 1:
        _epilogue(acc, S, Bias, Y, m, n, M, N, T, Y_B, Y_T, Y_N, HAS_BIAS, ACT)
    else:
        pmask = (m[:, None] < M) & (n[None, :] < N)
        tl.store(P + part * M * N + m[:, None] * N + n[None, :], acc, pmask)
        if REDUCE == "serial":
            tl.debug_barrier()
            tid = pid_m * tl.num_programs(1) + pid_n
            arrived = tl.atomic_add(C + tid, 1, sem="acq_rel", scope="gpu")
            if arrived == SPLIT - 1:
                total = tl.zeros((BM, BN), tl.float32)
                c = tl.zeros((BM, BN), tl.float32)
                for p in tl.static_range(SPLIT):   # fixed order, compensated
                    v = tl.load(P + p * M * N + m[:, None] * N + n[None, :], pmask, other=0.,
                                cache_modifier=".cg") - c
                    sk = total + v
                    c = (sk - total) - v
                    total = sk
                tl.atomic_xchg(C + tid, 0, sem="relaxed", scope="gpu")
                _epilogue(total, S, Bias, Y, m, n, M, N, T, Y_B, Y_T, Y_N, HAS_BIAS, ACT)


@triton.jit
def _split_rows(X, A, RS, M, T, X_B, X_T, X_K, K: tl.constexpr, FLAT: tl.constexpr, IN_ACT: tl.constexpr,
                FP16: tl.constexpr, BR: tl.constexpr, BKP: tl.constexpr,
                NORM: tl.constexpr = False, NW=None, NB=None, EPS: tl.constexpr = 0.0,
                PDL_TRIGGER: tl.constexpr = False):
    """Activation prep: x [M, K] (any strides) -> planes A [PARTS, M, K] row-major.

    Optional LayerNorm (two-pass mean / centered variance per row, M1 _quantize's formula, then
    centered * rstd * w + b) and input SiLU first. FP16 (2 parts): per-row power-of-two scale 2^s with
    the scaled row max in [2^14, 2^15); RS[m] = 2^-s; hi = fp16(x 2^s), lo = fp16(x 2^s - hi).
    BF16 (3 parts): hi, mid, lo as M1, no scale.
    """
    if PDL_TRIGGER:
        tl.extra.cuda.gdc_launch_dependents()
    m = tl.program_id(0) * BR + tl.arange(0, BR)
    kk = tl.arange(0, BKP)
    if FLAT:
        xrow = m * X_T
    else:
        xrow = (m // T) * X_B + (m % T) * X_T
    STEPS: tl.constexpr = (K + BKP - 1) // BKP
    if NORM:
        tot = tl.zeros((BR,), tl.float32)
        for s in range(STEPS):
            k = s * BKP + kk
            tot += tl.sum(tl.load(X + xrow[:, None] + k[None, :] * X_K, (m[:, None] < M) & (k[None, :] < K),
                                  other=0.), 1)
        mean = tot / K
        sq = tl.zeros((BR,), tl.float32)
        for s in range(STEPS):
            k = s * BKP + kk
            mask = (m[:, None] < M) & (k[None, :] < K)
            d = tl.where(mask, tl.load(X + xrow[:, None] + k[None, :] * X_K, mask, other=0.) - mean[:, None], 0.)
            sq += tl.sum(d * d, 1)
        rstd = tl.rsqrt(sq / K + EPS)
    if FP16:
        amax = tl.zeros((BR,), tl.float32)
        for s in range(STEPS):
            k = s * BKP + kk
            mask = (m[:, None] < M) & (k[None, :] < K)
            x = tl.load(X + xrow[:, None] + k[None, :] * X_K, mask, other=0.)
            if NORM:
                x = tl.where(mask, (x - mean[:, None]) * rstd[:, None] * tl.load(NW + k, k < K, other=0.)[None, :]
                             + tl.load(NB + k, k < K, other=0.)[None, :], 0.)
            if IN_ACT == "silu":
                x = x * tl.sigmoid(x)
            amax = tl.maximum(amax, tl.max(tl.abs(x), axis=1))
        e = (amax.to(tl.int32, bitcast=True) >> 23) & 255
        sf = tl.minimum(268 - e, 253)
        sc = (sf << 23).to(tl.float32, bitcast=True)
        tl.store(RS + m, ((254 - sf) << 23).to(tl.float32, bitcast=True), m < M)
    # without a row reduction (BF16 parts, no norm) the grid also splits K (program axis 1)
    for s in range(tl.program_id(1), STEPS, tl.num_programs(1)):
        k = s * BKP + kk
        mask = (m[:, None] < M) & (k[None, :] < K)
        x = tl.load(X + xrow[:, None] + k[None, :] * X_K, mask, other=0.)
        if NORM:
            x = tl.where(mask, (x - mean[:, None]) * rstd[:, None] * tl.load(NW + k, k < K, other=0.)[None, :]
                         + tl.load(NB + k, k < K, other=0.)[None, :], 0.)
        if IN_ACT == "silu":
            x = x * tl.sigmoid(x)
        out = A + m[:, None] * K + k[None, :]
        if FP16:
            xs = x * sc[:, None]
            hi = xs.to(tl.float16)
            tl.store(out, hi, mask)
            tl.store(out + M * K, (xs - hi.to(tl.float32)).to(tl.float16), mask)
        else:
            hi = x.to(tl.bfloat16)
            r = x - hi.to(tl.float32)
            mid = r.to(tl.bfloat16)
            tl.store(out, hi, mask)
            tl.store(out + M * K, mid, mask)
            tl.store(out + 2 * M * K, (r - mid.to(tl.float32)).to(tl.bfloat16), mask)


@triton.jit
def _pgemm(A, RS, W, S, Bias, Y, P, C, M, T, Y_B, Y_T, Y_N,
           N: tl.constexpr, K: tl.constexpr, HAS_BIAS: tl.constexpr, FP16: tl.constexpr, ACT: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr, REDUCE: tl.constexpr,
           EW: tl.constexpr = False, KAHAN: tl.constexpr = True,
           PDL_WAIT: tl.constexpr = False, PDL_TRIGGER: tl.constexpr = False):
    """GEMM on prepared planes: FP16x2 (hi, lo; separate accumulators; row scale RS) or BF16x3."""
    if PDL_WAIT:      # launched with PDL after the prep kernel: wait for its planes
        tl.extra.cuda.gdc_wait()
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    part = tl.program_id(2)
    m = pid_m * BM + tl.arange(0, BM)
    n = pid_n * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    kb = tl.arange(0, BK // 8)
    KB: tl.constexpr = (K + 7) // 8
    TILES: tl.constexpr = (K + BK - 1) // BK
    acc = tl.zeros((BM, BN), tl.float32)
    acc2 = tl.zeros((BM, BN), tl.float32)
    comp = tl.zeros((BM, BN), tl.float32)
    for tile in range(part, TILES, SPLIT):
        k = tile * BK + kk
        amask = (m[:, None] < M) & (k[None, :] < K)
        aptr = A + m[:, None] * K + k[None, :]
        if EW:
            w = tl.load(W + (k // 8)[:, None] * N + n[None, :], (k[:, None] < K) & (n[None, :] < N), other=0)
            b = _decode_pm1_ew(w, k, FP16)
        else:
            wb = tile * (BK // 8) + kb
            w = tl.load(W + wb[:, None] * N + n[None, :], (wb[:, None] < KB) & (n[None, :] < N), other=0)
            b = _decode_pm1(w, BK, BN, FP16)
        if FP16:
            lo = tl.load(aptr + M * K, amask, other=0.)
            acc2 += tl.dot(lo, b)
        else:
            lo = tl.load(aptr + 2 * M * K, amask, other=0.)
            mid = tl.load(aptr + M * K, amask, other=0.)
            acc2 += tl.dot(mid, b, tl.dot(lo, b))
        hi = tl.load(aptr, amask, other=0.)
        th = tl.dot(hi, b)
        if KAHAN:
            yk = th - comp
            sk = acc + yk
            comp = (sk - acc) - yk
            acc = sk
        else:
            acc += th
    if PDL_TRIGGER:
        tl.extra.cuda.gdc_launch_dependents()
    acc += acc2 - comp
    if FP16:
        acc *= tl.load(RS + m, m < M, other=0.)[:, None]
    if SPLIT == 1:
        _epilogue(acc, S, Bias, Y, m, n, M, N, T, Y_B, Y_T, Y_N, HAS_BIAS, ACT)
    else:
        pmask = (m[:, None] < M) & (n[None, :] < N)
        tl.store(P + part * M * N + m[:, None] * N + n[None, :], acc, pmask)
        if REDUCE == "serial":
            tl.debug_barrier()
            tid = pid_m * tl.num_programs(1) + pid_n
            arrived = tl.atomic_add(C + tid, 1, sem="acq_rel", scope="gpu")
            if arrived == SPLIT - 1:
                total = tl.zeros((BM, BN), tl.float32)
                c = tl.zeros((BM, BN), tl.float32)
                for p in tl.static_range(SPLIT):   # fixed order, compensated
                    v = tl.load(P + p * M * N + m[:, None] * N + n[None, :], pmask, other=0.,
                                cache_modifier=".cg") - c
                    sk = total + v
                    c = (sk - total) - v
                    total = sk
                tl.atomic_xchg(C + tid, 0, sem="relaxed", scope="gpu")
                _epilogue(total, S, Bias, Y, m, n, M, N, T, Y_B, Y_T, Y_N, HAS_BIAS, ACT)


@triton.jit
def _finish(P, S, Bias, Y, M, N: tl.constexpr, T, Y_B, Y_T, Y_N, HAS_BIAS: tl.constexpr,
            ACT: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr, PDL_WAIT: tl.constexpr = False):
    if PDL_WAIT:
        tl.extra.cuda.gdc_wait()
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    c = tl.zeros((BLOCK,), tl.float32)
    for p in tl.static_range(SPLIT):   # fixed order, compensated (Kahan)
        v = tl.load(P + p * M * N + i, i < M * N, other=0.) - c
        sk = acc + v
        c = (sk - acc) - v
        acc = sk
    m, n = i // N, i % N
    acc *= tl.load(S + n, i < M * N, other=0.)
    if HAS_BIAS:
        acc += tl.load(Bias + n, i < M * N, other=0.)
    if ACT == "silu":
        acc = acc / (1.0 + tl.exp(-acc))
    tl.store(Y + (m // T) * Y_B + (m % T) * Y_T + n * Y_N, acc, i < M * N)


MODES = ("bf16x3", "bf16x3l", "bf16x3t", "fp16x2", "fp16x2p", "bf16x3p")   # "...p": prep kernel + plane GEMM
REDUCES = ("serial", "workspace")


def default_config(m: int, n: int, k: int, mode: str) -> tuple:
    """(BM, BN, BK, SPLIT, warps, stages, REDUCE[, GROUPS, EW, KAHAN]) for shapes without a tuned entry
    (configs.py). GROUPS must be 1; EW selects the elementwise decode; KAHAN defaults to on."""
    if m <= 64:
        return (32, 64, 64, 8 if k >= 2048 else 4, 4, 2, "serial")
    if m <= 192:
        return (64, 64, 64, 4 if k >= 2048 else 2, 4, 2, "serial")
    return (64, 64, 64, 2 if k >= 2048 else 1, 4, 2, "serial")


def _launch(kernel, grid, warmup, *args, **kwargs):
    """Launch, or (warmup=True) only compile for these arguments' specialization, without running."""
    if warmup:
        return kernel.warmup(*args, grid=grid, **kwargs)
    return kernel[grid](*args, **kwargs)


def prep_config(m: int, k: int, k_contiguous: bool, rowwise: bool) -> tuple[int, int, int, int]:
    """(rows per program, K chunk, warps, K programs) for _split_rows; coalesce along the contiguous
    axis; split K across programs when no row reduction (row scale or LayerNorm) is needed."""
    if k_contiguous:
        if rowwise:
            return (2, min(1024, triton.next_power_of_2(k)), 4, 1)
        bkp = min(512, triton.next_power_of_2(k))
        return (1, bkp, 4, triton.cdiv(k, bkp))
    if rowwise:
        return (8, 128, 4, 1)
    return (32, 64, 4, triton.cdiv(k, 64))


def fit_counters(config: tuple, m: int, n: int, counters: int) -> tuple:
    """Serial split-K needs one counter per output tile; else use the two-kernel reduction."""
    bm, bn, bk, split, warps, stages, reduce, *rest = config
    if split > 1 and reduce == "serial" and triton.cdiv(m, bm) * triton.cdiv(n, bn) > counters:
        return (bm, bn, bk, split, warps, stages, "workspace", *rest)
    return config


# Programmatic dependent launch inside one op (never for the op's first kernel, so an op cannot overlap
# the previous op or a previous benchmark call): _finish after a split-K GEMM, the GEMM after _rowstats
# and the plane GEMM after _split_rows launch with launch_pdl and griddepcontrol.wait before reading
# their predecessor's output. Measured (probe_pdl.py, alternating windows, 2026-10-20): 0.7-3.7% lower
# latency on all 12 op x M cases, bitwise-identical results (test_layer.PDL, incl. graph capture).
PDL = True


def bgemm(x3: torch.Tensor, bits_t: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None,
          counters: torch.Tensor, n: int, k: int, y: torch.Tensor, ystrides: tuple[int, int, int], *,
          mode: str = "bf16x3", config: tuple | None = None, activation: str = "",
          input_activation: str = "", warmup: bool = False, norm=None) -> torch.Tensor:
    """y (preallocated) <- act(alpha * (x3 @ W^T) + bias) for x3 a [B, T, K] view (any strides).

    No validation beyond what the caller (layer.py) has done; this is the hot path.
    """
    b, t, _ = x3.shape
    m = b * t
    if m == 0:
        return y
    config = fit_counters(config or default_config(m, n, k, mode), m, n, counters.numel())
    bm, bn, bk, split, warps, stages, reduce, *rest = config
    groups = rest[0] if rest else 1
    ew = bool(rest[1]) if len(rest) > 1 else False
    kahan = bool(rest[2]) if len(rest) > 2 else True    # compensated running sum: the A3 fix
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn), split)
    xb, xt, xk = x3.stride()
    flat = b == 1 or xb == t * xt
    partial = torch.empty((split, m, n), device=x3.device, dtype=torch.float32) if split > 1 else y
    yb, yt, yn = ystrides
    if mode.endswith("p"):
        fp16 = mode == "fp16x2p"
        prep_norm = {} if norm is None else dict(NORM=True, NW=norm[0], NB=norm[1], EPS=float(norm[2]))
        planes = torch.empty((2 if fp16 else 3, m, k), device=x3.device,
                             dtype=torch.float16 if fp16 else torch.bfloat16)
        rs = torch.empty((m,), device=x3.device, dtype=torch.float32) if fp16 else scale
        br, bkp, pw, nk = prep_config(m, k, xk == 1, fp16 or norm is not None)
        with torch.cuda.device(x3.device):
            pdl_finish = PDL and split > 1 and reduce == "workspace"
            _launch(_split_rows, (triton.cdiv(m, br), nk), warmup, x3, planes, rs, m, t, xb, xt, xk, k, flat, input_activation,
                    fp16, br, bkp, **prep_norm, PDL_TRIGGER=PDL, num_warps=pw)
            gemm_args = (planes, rs, bits_t, scale, bias if bias is not None else scale, y, partial, counters,
                         m, t, yb, yt, yn, n, k, bias is not None, fp16, activation, bm, bn, bk, split, reduce)
            if groups != 1:
                raise ValueError("GROUPS > 1 (batched-dot intra-CTA split-K) was a measured dead end and is removed")
            _launch(_pgemm, grid, warmup, *gemm_args, EW=ew, KAHAN=kahan, PDL_TRIGGER=pdl_finish,
                    **(dict(PDL_WAIT=True, launch_pdl=True) if PDL else {}), num_warps=warps, num_stages=stages)
            if split > 1 and reduce == "workspace":
                _launch(_finish, (triton.cdiv(m * n, 256),), warmup, partial, scale, bias if bias is not None else scale, y,
                        m, n, t, yb, yt, yn, bias is not None, activation, split, 256,
                        **(dict(PDL_WAIT=True, launch_pdl=True) if pdl_finish else {}))
        return y
    norm_args = {}
    if norm is not None:     # (weight, bias, eps): row statistics first, normalization in the GEMM
        nw, nb, eps = norm
        mean = torch.empty((m,), device=x3.device, dtype=torch.float32)
        rstd = torch.empty((m,), device=x3.device, dtype=torch.float32)
        block = triton.next_power_of_2(k)
        br = max(1, min(8, 4096 // block))
        with torch.cuda.device(x3.device):
            _launch(_rowstats, (triton.cdiv(m, br),), warmup, x3, mean, rstd, m, t, xb, xt, xk, k, flat,
                    float(eps), br, block, PDL_TRIGGER=PDL, num_warps=4)
        norm_args = dict(NORM=True, MEAN=mean, RSTD=rstd, NW=nw, NB=nb)
        if PDL:
            norm_args.update(PDL_WAIT=True, launch_pdl=True)
    pdl_finish = PDL and split > 1 and reduce == "workspace"
    with torch.cuda.device(x3.device):
        _launch(_bgemm, grid, warmup, x3, bits_t, scale, bias if bias is not None else scale, y, partial, counters,
                     m, t, xb, xt, xk, yb, yt, yn,
                     n, k, bias is not None, mode, activation, input_activation, flat,
                     bm, bn, bk, split, reduce, EW=ew, KAHAN=kahan, PDL_TRIGGER=pdl_finish, **norm_args,
                     num_warps=warps, num_stages=stages)
        if split > 1 and reduce == "workspace":
            _launch(_finish, (triton.cdiv(m * n, 256),), warmup, partial, scale, bias if bias is not None else scale, y,
                    m, n, t, yb, yt, yn, bias is not None, activation, split, 256,
                    **(dict(PDL_WAIT=True, launch_pdl=True) if pdl_finish else {}))
    return y
