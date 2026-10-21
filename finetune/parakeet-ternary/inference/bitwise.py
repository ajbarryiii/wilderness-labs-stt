"""Experimental packed SIMT and FP32 lookup-table projections, without BF16/MMA.

These are explicitly selected research paths, not production dispatch. The LUT
keeps floating-point activations: for each group of four inputs, make the 16
subset sums, then index the positive/negative weight bit planes and subtract.
No activation quantization or whole-weight expansion is needed.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .kernels import _finish, _gemv


@triton.jit
def _lookup(X, W, S, Bias, Y, Partial, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
            BG: tl.constexpr, SPLIT: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    part = tl.program_id(2)
    group = tl.arange(0, BG)
    words = tl.arange(0, BG // 4)
    byte = tl.arange(0, 4)
    pattern = tl.arange(0, 16)
    acc = tl.zeros((BM, BN), tl.float32)
    for tile in range(tl.cdiv(K, 4 * BG * SPLIT)):
        start = (tile * SPLIT + part) * BG
        g = start + group
        a0 = tl.load(X + m[:, None] * K + g[None, :] * 4,
                     (m[:, None] < M) & (g[None, :] * 4 < K), other=0)
        a1 = tl.load(X + m[:, None] * K + g[None, :] * 4 + 1,
                     (m[:, None] < M) & (g[None, :] * 4 + 1 < K), other=0)
        a2 = tl.load(X + m[:, None] * K + g[None, :] * 4 + 2,
                     (m[:, None] < M) & (g[None, :] * 4 + 2 < K), other=0)
        a3 = tl.load(X + m[:, None] * K + g[None, :] * 4 + 3,
                     (m[:, None] < M) & (g[None, :] * 4 + 3 < K), other=0)
        lut = (tl.where((pattern[None, None, :] & 1) != 0, a0[:, :, None], 0.)
               + tl.where((pattern[None, None, :] & 2) != 0, a1[:, :, None], 0.)
               + tl.where((pattern[None, None, :] & 4) != 0, a2[:, :, None], 0.)
               + tl.where((pattern[None, None, :] & 8) != 0, a3[:, :, None], 0.))
        w = tl.load(W + (start // 4 + words[:, None]) * N + n[None, :],
                    (start // 4 + words[:, None] < tl.cdiv(K, 16)) & (n[None, :] < N), other=0)
        c = ((w[:, None, :].to(tl.uint32) >> (byte[None, :, None] * 8)) & 255).reshape(BG, BN)
        pos = (c & 1) | ((c >> 1) & 2) | ((c >> 2) & 4) | ((c >> 3) & 8)
        neg = ((c >> 1) & 1) | ((c >> 2) & 2) | ((c >> 3) & 4) | ((c >> 4) & 8)
        plus = tl.gather(lut, tl.broadcast_to(pos[None, :, :], (BM, BG, BN)).to(tl.int32), axis=2)
        minus = tl.gather(lut, tl.broadcast_to(neg[None, :, :], (BM, BG, BN)).to(tl.int32), axis=2)
        acc += tl.sum(plus - minus, axis=1)
    if SPLIT == 1:
        acc *= tl.load(S + n, n < N, other=0)[None, :]
        if HAS_BIAS:
            acc += tl.load(Bias + n, n < N, other=0)[None, :]
        tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))
    else:
        tl.store(Partial + part * M * N + m[:, None] * N + n[None, :],
                 acc, (m[:, None] < M) & (n[None, :] < N))


def linear(x, layer, *, variant="lookup", config=None):
    """FP32 contiguous 2-D inputs; layer is a validated PackedLinear.

    LUT config = (BM, BN, groups-per-tile, split-K, warps).
    SIMT config = (outputs-per-tile, split-K, warps).
    """
    if (x.ndim != 2 or not x.is_cuda or x.device != layer.packed_t.device
            or x.dtype != torch.float32 or not x.is_contiguous() or x.shape[1] != layer.in_features):
        raise ValueError("expected contiguous FP32 CUDA [M,K] input on the layer device")
    if torch.is_grad_enabled():
        raise RuntimeError("experimental kernels require inference_mode/no_grad")
    m, k = x.shape
    n = layer.out_features
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if not m:
        return y
    if variant == "lookup":
        bm, bn, bg, split, warps = config or (4, 32, 32, 4, 4)
    elif variant == "simt":
        bn, split, warps = config or (8, 4 if k > 2048 else 1, 4)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    partial = torch.empty((split, m, n), device=x.device, dtype=torch.float32) if split > 1 else y
    bias = layer.bias if layer.bias is not None else layer.scale
    with torch.cuda.device(x.device):
        if variant == "lookup":
            _lookup[(triton.cdiv(m, bm), triton.cdiv(n, bn), split)](
                x, layer.packed_t, layer.scale, bias, y, partial, m, n, k,
                layer.bias is not None, bm, bn, bg, split, num_warps=warps)
        else:
            _gemv[(triton.cdiv(n, bn), m, split)](
                x, layer.packed_t, layer.scale, bias, y, partial, m, n, k, m,
                m*k, k, 1, m*n, n, 1, layer.bias is not None, bn,
                max(16, triton.next_power_of_2(triton.cdiv(k, split))), split, num_warps=warps)
        if split > 1:
            _finish[(triton.cdiv(m*n, 256),)](
                partial, layer.scale, bias, y, m, n, m, m*n, n, 1,
                layer.bias is not None, split, 256)
    return y
