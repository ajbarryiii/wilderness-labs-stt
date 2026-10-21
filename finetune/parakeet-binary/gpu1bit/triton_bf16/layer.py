"""Agent A's BinaryLayer (gpu1bit/shared/api.py): MB1 1-bit weights, Triton float tensor-core kernels.

`IMPLEMENTATION` is what the contract tests discover. See kernels.py for layout and arithmetic.
"""
from __future__ import annotations

import torch
from torch import Tensor

from gpu1bit.shared import api
from . import configs as tuned
from . import kernels

# Split-K tile counters for the single-kernel ("serial") reduction. The tuned tables never select it
# (it measured no faster than the two-kernel reduction), so none are allocated by default and resident
# storage is exactly MB1's code bytes (transposed) plus FP32 scales (and bias). prepare(counters=...)
# re-enables it for experiments; it must fit the 1.10x budget slack.
MAX_COUNTERS = 4096


class TritonBF16Layer:
    """1-bit weights resident as uint8 [ceil(K/8), N] (MB1's code bytes, transposed), FP32 scale/bias."""

    mode = "bf16x3"

    def __init__(self, bits_t: Tensor, scale: Tensor, bias: Tensor | None, counters: Tensor,
                 n: int, k: int, mode: str, configs: dict | None):
        self.bits_t, self.scale, self.bias, self.counters = bits_t, scale, bias, counters
        self.out_features, self.in_features = n, k
        self.mode = mode
        self.configs = dict(configs or {})   # optional per-M overrides: {m: config tuple}
        self.device = bits_t.device

    @classmethod
    def prepare(cls, codes: Tensor, scale: Tensor, bias: Tensor | None, in_features: int, device,
                *, mode: str | None = None, configs: dict | None = None,
                counters: bool = False) -> "TritonBF16Layer":
        mode = mode or cls.mode
        if mode not in kernels.MODES:
            raise ValueError(f"unknown mode {mode!r}")
        n, k = api.validate_weights(codes, scale, bias, in_features)
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("TritonBF16Layer needs a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        bits_t = codes.to(device).t().contiguous()                       # new storage, [ceil(K/8), N]
        scale_d = scale.to(device=device, dtype=torch.float32).clone().contiguous()
        bias_d = None if bias is None else bias.to(device=device, dtype=torch.float32).clone().contiguous()
        used = api.resident_bytes(bits_t, scale_d, bias_d)
        spare = api.weight_budget_bytes(n, k, bias is not None) - used
        size = max(0, min(MAX_COUNTERS, spare // 4)) if counters else 0
        counters = torch.zeros(size, dtype=torch.int32, device=device)
        return cls(bits_t, scale_d, bias_d, counters, n, k, mode, configs)

    def weight_bytes(self) -> int:
        return api.resident_bytes(self.bits_t, self.scale, self.bias, self.counters)

    # -- hot path -------------------------------------------------------------------------------
    def _check(self, x: Tensor) -> None:
        if not isinstance(x, Tensor) or x.dtype != torch.float32 or x.device != self.device:
            raise ValueError("expected an FP32 CUDA tensor on the layer's device")
        if torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError("inference only: use torch.no_grad() or torch.inference_mode()")

    def _run(self, x3: Tensor, y: Tensor, ystrides: tuple, activation: str = "",
             input_activation: str = "", norm=None, kind: str = "linear") -> None:
        m = x3.shape[0] * x3.shape[1]
        n, k = self.out_features, self.in_features
        mode = self.mode
        if m in self.configs:                     # explicit per-M override (tests, tuning)
            config = self.configs[m]
        else:
            hit = tuned.lookup(mode, kind, n, k, m)
            if hit is None:
                config = kernels.default_config(m, n, k, mode)
            else:                                 # tuned: faster of in-kernel split and prep + planes
                config, path = hit
                if path == "planes" and not mode.endswith("p"):
                    mode = mode + "p"
        config = kernels.fit_counters(config, m, self.out_features, self.counters.numel())
        kernels.bgemm(x3, self.bits_t, self.scale, self.bias, self.counters, self.out_features,
                      self.in_features, y, ystrides, mode=mode, config=config,
                      activation=activation, input_activation=input_activation, norm=norm)

    def project(self, x: Tensor, *, conv: bool = False, components: int = 3, activation: str = "",
                norm=None, input_activation: str = "") -> Tensor:
        """M1's projection-provider operation (gpu1bit/shared/provider.py) as one fused op.

        xx = x.transpose(1, 2) if conv else x; z = LayerNorm(norm)(xx) if norm; z = silu(z) if
        input_activation; y = alpha * (W z) + bias; silu(y) if activation; conv returns y.transpose(1, 2).
        `components` (M1's INT8 residual part count) is ignored: this arithmetic class is float
        (BF16x3 or FP16x2 split of FP32 activations). Both SiLUs are fused. LayerNorm: on the in-kernel
        ("fused") path a row-statistics kernel (mean, rstd) then normalization of each A tile inside the
        GEMM; on the "planes" path inside the prep kernel. The path per op and M band is configs.py's.
        """
        self._check(x)
        if activation not in ("", "silu") or input_activation not in ("", "silu"):
            raise ValueError("activation and input_activation must be '' or 'silu'")
        k, n = self.in_features, self.out_features
        if conv:
            if x.dim() != 3 or x.shape[1] != k:
                raise ValueError(f"expected [B, {k}, T] input, got {tuple(x.shape)}")
            x3 = x.transpose(1, 2)
        else:
            if x.dim() < 1 or x.shape[-1] != k:
                raise ValueError(f"expected [..., {k}] input, got {tuple(x.shape)}")
            x3 = x if x.dim() == 3 else (x.reshape(1, -1, k) if x.dim() < 3 else x.reshape(-1, x.shape[-2], k))
        lead = x3.shape[:-1] if conv else x.shape[:-1]
        if norm is not None:
            if tuple(norm.normalized_shape) != (k,) or norm.weight is None or norm.bias is None:
                raise ValueError("norm must be an affine LayerNorm over the last dimension K")
            norm = (norm.weight, norm.bias, norm.eps)
        y = torch.empty((*lead, n), device=x.device, dtype=torch.float32)
        if y.numel():
            t = x3.shape[1]
            self._run(x3, y, (t * n, n, 1), activation=activation, input_activation=input_activation, norm=norm,
                      kind="conv" if conv else "linear")
        return y.transpose(1, 2) if conv else y

    def linear(self, x: Tensor) -> Tensor:
        self._check(x)
        k, n = self.in_features, self.out_features
        if x.dim() < 1 or x.shape[-1] != k:
            raise ValueError(f"expected [..., {k}] input, got {tuple(x.shape)}")
        lead = x.shape[:-1]
        if x.dim() == 1:
            x3 = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x3 = x.unsqueeze(0)
        elif x.dim() == 3:
            x3 = x
        else:
            x3 = x.reshape(-1, x.shape[-2], k)   # a copy only if the leading dims cannot be merged
        y = torch.empty((*lead, n), device=x.device, dtype=torch.float32)
        if y.numel():
            t = x3.shape[1]
            self._run(x3, y, (t * n, n, 1))
        return y

    def conv(self, x: Tensor) -> Tensor:
        self._check(x)
        k, n = self.in_features, self.out_features
        if x.dim() != 3 or x.shape[1] != k:
            raise ValueError(f"expected [B, {k}, T] input, got {tuple(x.shape)}")
        b, _, t = x.shape
        y = torch.empty((b, t, n), device=x.device, dtype=torch.float32)
        if y.numel():
            self._run(x.transpose(1, 2), y, (t * n, n, 1), kind="conv")
        return y.transpose(1, 2)


def _variant(mode: str) -> type:
    return type(f"Triton_{mode}", (TritonBF16Layer,), {"mode": mode})


# Registered as triton_bf16:<mode> (gpu1bit/shared/registry.py). "bf16x3" is the default candidate.
IMPLEMENTATIONS = {mode: (TritonBF16Layer if mode == "bf16x3" else _variant(mode))
                   for mode in ("bf16x3", "fp16x2", "fp16x2p", "bf16x3l")}
IMPLEMENTATION = TritonBF16Layer
