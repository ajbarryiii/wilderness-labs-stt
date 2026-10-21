"""int8_mma:b01, the post-selection {0, 1}-operand variant of the frozen Int8MmaLayer (layer.py, unchanged).

Same weights (fragment layout), same activation quantization (b1_quantize / M1's _quantize), same
tuned configs (configs.py) and the same FP32 epilogues; only b1_dot's weight expansion changes to
{0, 1} bytes with an in-kernel INT32 row-sum correction (kernels_b01.py), so outputs are bitwise those
of int8_mma for every config. Registered as "int8_mma:b01" through b01_run.py's registry patch (the
frozen layer.py exposes only IMPLEMENTATION = Int8MmaLayer and must stay byte-identical).
"""
from __future__ import annotations

import dataclasses

import torch
from torch import Tensor

from gpu1bit.int8_mma import configs, cuda, kernels, kernels_b01
from gpu1bit.int8_mma.layer import Int8MmaLayer


def _round_up(v: int, m: int) -> int:
    return -(-v // m) * m


class Int8MmaB01Layer(Int8MmaLayer):
    """Int8MmaLayer whose fast-path dot is the b01 kernel; prepare / storage / project inherited."""

    def dot(self, q: Tensor, qs: Tensor, ldq: int, m: int, y: Tensor, *, silu: bool = False,
            config: kernels.DotConfig | None = None, order: int = 0) -> None:
        if not self.fast:  # MB1-layout fallback for shapes whose fragment layout exceeds the budget
            return super().dot(q, qs, ldq, m, y, silu=silu, config=config, order=order)
        dot_b01(self, q, qs, ldq, m, y, silu=silu, config=config, order=order)


def dot_b01(layer: Int8MmaLayer, q: Tensor, qs: Tensor, ldq: int, m: int, y: Tensor, *, silu: bool = False,
            config: kernels.DotConfig | None = None, order: int = 0) -> None:
    """layer.py's dot() with the b01 kernel (identical argument list and launch geometry)."""
    n, k = layer.out_features, layer.in_features
    if ldq % 16 or q.stride(1) != ldq or q.stride(2) != 1 or q.data_ptr() % 16:
        raise ValueError("fast path needs 16-byte aligned rows of q")
    config = config or configs.choose(m, n, k)
    if config.xchg == 3 and 127 * k >= 2**20:  # packed lanes would overflow: exact INT32 exchange
        config = dataclasses.replace(config, xchg=1)
    cfg = kernels_b01.B01Config.of(config)
    source, name = kernels_b01.dot_source(cfg, silu, order)
    ws_ints = cfg.workspace_ints(m, n)
    workspace = torch.empty(ws_ints, device=q.device, dtype=torch.int32) if ws_ints else None
    cuda.load(source, name, cfg.shared_bytes).launch(
        cfg.grid(m, n), (cfg.threads, 1, 1),
        [cuda.ptr(q), cuda.i64(q.stride(0)), cuda.i32(ldq), cuda.i32(_round_up(k, 16)), cuda.ptr(qs),
         cuda.i32(qs.stride(0)), cuda.ptr(layer.words), cuda.i32(layer.words.shape[0]),
         cuda.i32(layer.words.shape[1]), cuda.ptr(layer.scale), cuda.ptr(layer.bias), cuda.ptr(y),
         cuda.i32(y.stride(0)), cuda.i32(m), cuda.i32(n), cuda.ptr(workspace)])


IMPLEMENTATIONS = {"b01": Int8MmaB01Layer}
