"""Agent A: register-direct Gluon kernels (no shared memory in the main loop).

Same weights (uint8 bits_t [ceil(K/8), N]) and arithmetic as kernels.py, but both MMA operands are
loaded from global memory straight into their mma.sync dot-operand register layouts:
- A: FP32 x (any strides) loaded in DotOperandLayout(0) (k_width 4: 4 consecutive k per thread, one
  16-byte load), split in registers (fp16x2 with a per-(row, K-tile) power-of-two scale, or bf16x3).
- B: the byte holding code (k, n) loaded per element in DotOperandLayout(1) (the 4 consecutive k a
  thread owns share one byte; repeated addresses are served by L1), decoded by shift/and/or + bitcast.
- The next K tile's A and B are loaded into registers before the current tile's MMAs (software
  double buffering), so there is no shared memory, no barrier and no async-copy wait in the loop.
Split-K across CTAs uses kernels._finish (FP32 workspace, fixed-order deterministic reduction).
"""
from __future__ import annotations

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2

from . import kernels


@gluon.jit
def _decode(w, k, FP16: gl.constexpr):
    bits = (w.to(gl.int32) >> (k % 8)[:, None]) & 1
    one: gl.constexpr = 0x3C00 if FP16 else 0x3F80
    h = (((bits ^ 1) << 15) | one).to(gl.int16)
    return h.to(gl.float16 if FP16 else gl.bfloat16, bitcast=True)


@gluon.jit
def _rd_gemm(X, W, S, Bias, Y, P, M, T, X_B, X_T, X_K, Y_B, Y_T, Y_N,
             N: gl.constexpr, K: gl.constexpr, HAS_BIAS: gl.constexpr, FP16: gl.constexpr,
             ACT: gl.constexpr, FLAT: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
             SPLIT: gl.constexpr, WARPS_M: gl.constexpr, WARPS_N: gl.constexpr):
    MMA: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[WARPS_M, WARPS_N],
                                                  instr_shape=[16, 8])
    AL: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MMA, k_width=4)
    BL: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MMA, k_width=4)
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    part = gl.program_id(2)
    ma = pid_m * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, AL))
    ka = gl.arange(0, BK, layout=gl.SliceLayout(0, AL))
    kb = gl.arange(0, BK, layout=gl.SliceLayout(1, BL))
    nb = pid_n * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, BL))
    if FLAT:
        xrow = ma * X_T
    else:
        xrow = (ma // T) * X_B + (ma % T) * X_T
    TILES: gl.constexpr = (K + BK - 1) // BK
    acc = gl.zeros((BM, BN), gl.float32, layout=MMA)
    acc2 = gl.zeros((BM, BN), gl.float32, layout=MMA)
    comp = gl.zeros((BM, BN), gl.float32, layout=MMA)
    # prologue: first tile of this part
    k = part * BK + ka
    a = gl.load(X + xrow[:, None] + k[None, :] * X_K, (ma[:, None] < M) & (k[None, :] < K), other=0.)
    kw = part * BK + kb
    w = gl.load(W + (kw // 8)[:, None] * N + nb[None, :], (kw[:, None] < K) & (nb[None, :] < N), other=0)
    for tile in range(part, TILES, SPLIT):
        # prefetch the next tile into registers (masked off past the end)
        k = (tile + SPLIT) * BK + ka
        a_next = gl.load(X + xrow[:, None] + k[None, :] * X_K, (ma[:, None] < M) & (k[None, :] < K), other=0.)
        kn = (tile + SPLIT) * BK + kb
        w_next = gl.load(W + (kn // 8)[:, None] * N + nb[None, :], (kn[:, None] < K) & (nb[None, :] < N),
                         other=0)
        b = _decode(w, tile * BK + kb, FP16)
        if FP16:
            amax = gl.max(gl.abs(a), axis=1)
            e = (amax.to(gl.int32, bitcast=True) >> 23) & 255
            sf = gl.minimum(268 - e, 253)
            sc = (sf << 23).to(gl.float32, bitcast=True)
            inv = ((254 - sf) << 23).to(gl.float32, bitcast=True)
            xs = a * sc[:, None]
            hi = xs.to(gl.float16)
            lo = (xs - hi.to(gl.float32)).to(gl.float16)
            invm = gl.convert_layout(inv, gl.SliceLayout(1, MMA))[:, None]
            th = mma_v2(hi, b, gl.zeros((BM, BN), gl.float32, layout=MMA)) * invm
            acc2 += mma_v2(lo, b, gl.zeros((BM, BN), gl.float32, layout=MMA)) * invm
        else:
            hi = a.to(gl.bfloat16)
            r = a - hi.to(gl.float32)
            mid = r.to(gl.bfloat16)
            lo = (r - mid.to(gl.float32)).to(gl.bfloat16)
            t2 = mma_v2(lo, b, gl.zeros((BM, BN), gl.float32, layout=MMA))
            acc2 += mma_v2(mid, b, t2)
            th = mma_v2(hi, b, gl.zeros((BM, BN), gl.float32, layout=MMA))
        yk = th - comp                       # compensated running sum (A3 fix, see kernels.py)
        sk = acc + yk
        comp = (sk - acc) - yk
        acc = sk
        a = a_next
        w = w_next
    acc += acc2 - comp
    m = pid_m * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, MMA))
    n = pid_n * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, MMA))
    omask = (m[:, None] < M) & (n[None, :] < N)
    if SPLIT == 1:
        acc = acc * gl.load(S + n, n < N, other=0.)[None, :]
        if HAS_BIAS:
            acc += gl.load(Bias + n, n < N, other=0.)[None, :]
        if ACT == "silu":
            acc = acc / (1.0 + gl.exp(-acc))
        yrow = (m // T) * Y_B + (m % T) * Y_T
        gl.store(Y + yrow[:, None] + n[None, :] * Y_N, acc, omask)
    else:
        gl.store(P + part * M * N + m[:, None] * N + n[None, :], acc, omask)


def rd_gemm(x3, bits_t, scale, bias, n, k, y, ystrides, *, mode, config, activation="", warmup=False):
    """config: (BM, BN, BK, SPLIT, WARPS_M, WARPS_N)."""
    b, t, _ = x3.shape
    m = b * t
    bm, bn, bk, split, wm, wn = config
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn), split)
    xb, xt, xk = x3.stride()
    flat = b == 1 or xb == t * xt
    partial = torch.empty((split, m, n), device=x3.device, dtype=torch.float32) if split > 1 else y
    yb, yt, yn = ystrides
    with torch.cuda.device(x3.device):
        kernels._launch(_rd_gemm, grid, warmup, x3, bits_t, scale, bias if bias is not None else scale, y, partial,
                        m, t, xb, xt, xk, yb, yt, yn, n, k, bias is not None, mode == "fp16x2", activation, flat,
                        bm, bn, bk, split, wm, wn, num_warps=wm * wn)
        if split > 1:
            kernels._launch(kernels._finish, (triton.cdiv(m * n, 256),), warmup, partial, scale,
                            bias if bias is not None else scale, y, m, n, t, yb, yt, yn, bias is not None,
                            activation, split, 256)
    return y
