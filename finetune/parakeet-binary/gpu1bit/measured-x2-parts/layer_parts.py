"""int8_mma with 1, 2 or 3 activation parts (plans/activation-precision.md), ±1 or {0, 1} weight operands.

`Int8MmaPartsLayer` is the frozen Int8MmaLayer (layer.py, unchanged) with:
  * project(x, *, conv, components, activation, norm, input_activation) honouring components in
    {1, 2, 3} (H's PartCount wrapper, R1, overrides it on every provider call): M1's own _quantize at
    COMPONENTS = components (fused LayerNorm / input SiLU as in R-expanded), then the P-part b1_dot;
  * dot(q, qs, ldq, m, y, ..., parts=None): the P-part kernel, P = q.shape[0] unless given;
  * linear / conv (the BinaryLayer interface) at the variant's fixed part count PARTS: the first PARTS
    parts of b1_quantize (bitwise M1's 3-part quantizer, whose first P parts are its P-part output).
With P = 3 and ±1 operands every call takes the frozen code path, so the x3 arm is frozen int8_mma
bit for bit; P = 3 with B01 is int8_mma:b01. Tile tables are the frozen x3 ones (R6). Shapes whose
fragment layout exceeds the storage budget use the frozen MB1-layout fallback kernel (3 parts): its
missing parts are zero-padded, which gives the same values (a zero part adds an exact 0).
Registered (parts_run.py): int8_mma:x2, int8_mma:x1, int8_mma:x2:b01, int8_mma:x1:b01.
"""
from __future__ import annotations

import dataclasses

import torch
from torch import Tensor

from gpu1bit.int8_mma import configs, cuda, kernels, kernels_parts
from gpu1bit.int8_mma.layer import Int8MmaLayer, quantize
from gpu1bit.int8_mma.layer_b01 import dot_b01


def _round_up(v: int, m: int) -> int:
    return -(-v // m) * m


def m1_quantize_parts(xx: Tensor, parts: int, norm=None, input_activation: str = "") -> tuple[Tensor, Tensor]:
    """M1's inference.fused_kernels._quantize at COMPONENTS = parts, called as ExpandedProjection does."""
    import triton
    from inference.fused_kernels import _quantize
    b, t, k = xx.shape
    m = b * t
    q = torch.empty(parts, m, k, device=xx.device, dtype=torch.int8)
    qs = torch.empty(parts, m, device=xx.device, dtype=torch.float32)
    _quantize[(m,)](xx, q, qs, norm.weight if norm is not None else None, norm.bias if norm is not None else None,
                    m, k, t, *xx.stride(), triton.next_power_of_2(k), parts, norm is not None,
                    norm.eps if norm is not None else 0., INPUT_SILU=input_activation == "silu", enable_fp_fusion=False)
    return q, qs


class Int8MmaPartsLayer(Int8MmaLayer):
    PARTS = 3
    B01 = False

    # --- matmul -----------------------------------------------------------------------------------
    def dot(self, q: Tensor, qs: Tensor, ldq: int, m: int, y: Tensor, *, silu: bool = False,
            config: kernels.DotConfig | None = None, order: int = 0, parts: int | None = None) -> None:
        parts = q.shape[0] if parts is None else parts
        if parts not in (1, 2, 3) or q.shape[0] != parts or qs.shape[0] != parts:
            raise ValueError(f"q / qs must hold {parts} parts")
        if not self.fast:  # frozen 3-part fallback with zero parts appended (exact zeros)
            if parts < 3:
                q = torch.cat([q, q.new_zeros(3 - parts, *q.shape[1:])])
                qs = torch.cat([qs, qs.new_zeros(3 - parts, qs.shape[1])])
            return Int8MmaLayer.dot(self, q, qs, ldq, m, y, silu=silu, config=config, order=order)
        if parts == 3:
            if self.B01:
                return dot_b01(self, q, qs, ldq, m, y, silu=silu, config=config, order=order)
            return Int8MmaLayer.dot(self, q, qs, ldq, m, y, silu=silu, config=config, order=order)
        dot_parts(self, q, qs, ldq, m, y, silu=silu, config=config, order=order, parts=parts, b01=self.B01)

    # --- provider op ------------------------------------------------------------------------------
    def project(self, x: Tensor, *, conv: bool = False, components: int | None = None, activation: str = "",
                norm=None, input_activation: str = "") -> Tensor:
        parts = self.PARTS if components is None else components
        if parts not in (1, 2, 3):
            raise ValueError(f"components must be 1, 2 or 3, got {components!r}")
        if parts == 3 and not self.B01:
            return Int8MmaLayer.project(self, x, conv=conv, components=3, activation=activation, norm=norm,
                                        input_activation=input_activation)
        if activation not in ("", "silu") or input_activation not in ("", "silu"):
            raise ValueError(f"unsupported activation {activation!r} / {input_activation!r}")
        xx = x.transpose(1, 2) if conv else x
        self._check(xx, "linear")
        if xx.dim() == 2:
            xx = xx.unsqueeze(0)
        if xx.dim() != 3:
            raise ValueError("project expects [B, T, K] (or [T, K]) input, as M1's provider calls")
        b, t, k = xx.shape
        m, n = b * t, self.out_features
        y = torch.empty(b, t, n, device=self.device, dtype=torch.float32)
        if m:
            q, qs = m1_quantize_parts(xx, parts, norm, input_activation)
            q, ldq = _aligned(q, k)
            self.dot(q, qs, ldq, m, y.view(m, n), silu=activation == "silu", parts=parts)
        if conv:
            return y.transpose(1, 2)
        return y if x.dim() == 3 else y.view(*x.shape[:-1], n)

    # --- BinaryLayer calls at the fixed part count ----------------------------------------------------
    def _run(self, x: Tensor, rows: tuple, m: int, y: Tensor) -> None:
        """linear / conv: b1_quantize (bitwise M1's quantizer, layer.py) gives the three residual parts;
        the first PARTS of them are exactly M1's quantizer at COMPONENTS = PARTS (its loop is sequential)."""
        if self.PARTS == 3:
            return Int8MmaLayer._run(self, x, rows, m, y)
        q, qs, ldq = quantize(x, rows, m, self.in_features)
        self.dot(q[:self.PARTS], qs[:self.PARTS], ldq, m, y, parts=self.PARTS)


def _aligned(q: Tensor, k: int) -> tuple[Tensor, int]:
    """q [P, M, K] -> rows padded with zeros to a multiple of 16 bytes (cp.async alignment) if needed."""
    if k % 16 == 0 and q.data_ptr() % 16 == 0:
        return q, k
    ldq = _round_up(k, 16)
    padded = q.new_zeros(q.shape[0], q.shape[1], ldq)
    padded[:, :, :k] = q
    return padded, ldq


def dot_parts(layer: Int8MmaLayer, q: Tensor, qs: Tensor, ldq: int, m: int, y: Tensor, *, silu: bool = False,
              config: kernels.DotConfig | None = None, order: int = 0, parts: int = 3, b01: bool = False) -> None:
    """layer.py's dot() with the P-part kernel (identical argument list and launch geometry)."""
    n, k = layer.out_features, layer.in_features
    if ldq % 16 or q.stride(1) != ldq or q.stride(2) != 1 or q.data_ptr() % 16:
        raise ValueError("fast path needs 16-byte aligned rows of q")
    config = config or configs.choose(m, n, k)
    if config.xchg == 3 and 127 * k >= 2**20:
        config = dataclasses.replace(config, xchg=1)
    cfg = kernels_parts.PartsConfig.of(config, parts, b01)
    source, name = kernels_parts.dot_source(cfg, silu, order, parts, b01)
    ws_ints = cfg.workspace_ints(m, n)
    workspace = torch.empty(ws_ints, device=q.device, dtype=torch.int32) if ws_ints else None
    cuda.load(source, name, cfg.shared_bytes).launch(
        cfg.grid(m, n), (cfg.threads, 1, 1),
        [cuda.ptr(q), cuda.i64(q.stride(0)), cuda.i32(ldq), cuda.i32(_round_up(k, 16)), cuda.ptr(qs),
         cuda.i32(qs.stride(0)), cuda.ptr(layer.words), cuda.i32(layer.words.shape[0]),
         cuda.i32(layer.words.shape[1]), cuda.ptr(layer.scale), cuda.ptr(layer.bias), cuda.ptr(y),
         cuda.i32(y.stride(0)), cuda.i32(m), cuda.i32(n), cuda.ptr(workspace)])


def variant(parts: int, b01: bool):
    return type(f"Int8MmaX{parts}{'B01' if b01 else ''}Layer", (Int8MmaPartsLayer,), {"PARTS": parts, "B01": b01})


IMPLEMENTATIONS = {"x2": variant(2, False), "x1": variant(1, False), "x2:b01": variant(2, True), "x1:b01": variant(1, True)}
