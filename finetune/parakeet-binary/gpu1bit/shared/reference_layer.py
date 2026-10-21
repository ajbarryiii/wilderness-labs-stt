"""H's known-good BinaryLayer: resident MB1 codes and FP32 scales only; each call unpacks the bits
into a transient FP64 matrix and multiplies in FP64 (cuBLAS DGEMM), then rounds to FP32 once.

Slow by design. It exists to validate the contract tests themselves (it must pass all of them) and
to drive provider and pipeline tests with numerics far inside the M1-relative thresholds. It is
registered as "h_reference" and is never a screening candidate.
"""
from __future__ import annotations

import torch
from torch import Tensor

from gpu1bit.shared import api

_SHIFTS = {}


def _shifts(device) -> Tensor:
    if device not in _SHIFTS:  # a 1-byte-per-entry process-wide table (within UNACCOUNTED_SLACK_BYTES)
        _SHIFTS[device] = torch.arange(8, dtype=torch.uint8, device=device)
    return _SHIFTS[device]


class ReferenceLayer:
    def __init__(self, codes: Tensor, scale: Tensor, bias: Tensor | None, k: int):
        self.codes, self.scale, self.bias, self.in_features = codes, scale, bias, k
        self.out_features = codes.shape[0]

    @classmethod
    def prepare(cls, codes, scale, bias, in_features, device):
        api.validate_weights(codes, scale, bias, in_features)
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("ReferenceLayer runs on CUDA")
        _shifts(device)
        return cls(codes.to(device, copy=True).contiguous(), scale.to(device, copy=True).contiguous(),
                   None if bias is None else bias.to(device, copy=True).contiguous(), in_features)

    def weight_bytes(self) -> int:
        return api.resident_bytes(self.codes, self.scale, self.bias)

    def _check(self, x: Tensor, rank_ok: bool, k: int):
        if not isinstance(x, Tensor) or x.dtype != torch.float32 or x.device != self.codes.device or not rank_ok \
                or k != self.in_features:
            raise ValueError(f"expected FP32 input on {self.codes.device} with K={self.in_features}")
        if torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError("inference only; use torch.inference_mode()")

    def _dense64(self) -> Tensor:
        bits = ((self.codes[..., None] >> _shifts(self.codes.device)) & 1).flatten(1)[:, :self.in_features]
        return (bits.to(torch.float64) * 2 - 1) * self.scale.to(torch.float64)[:, None]

    def linear(self, x: Tensor) -> Tensor:
        self._check(x, x.dim() >= 1, x.shape[-1] if x.dim() else -1)
        with torch.no_grad():
            if x.numel() == 0:
                return torch.empty((*x.shape[:-1], self.out_features), device=x.device, dtype=torch.float32)
            y = torch.matmul(x.to(torch.float64), self._dense64().t())
            if self.bias is not None:
                y = y + self.bias.to(torch.float64)
            return y.to(torch.float32).contiguous()

    def conv(self, x: Tensor) -> Tensor:
        self._check(x, x.dim() == 3, x.shape[1] if x.dim() == 3 else -1)
        return self.linear(x.transpose(1, 2)).transpose(1, 2)


IMPLEMENTATION = ReferenceLayer
