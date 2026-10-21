"""Agent C's BinaryLayer: FP32 SIMT kernel straight from MB1's 1-bit codes (see kernels.py).

IMPLEMENTATION is discovered by H's contract tests (gpu1bit/tests). Weights stay in MB1's exact
export layout on the GPU (N * ceil(K/8) bytes) plus FP32 scale (and bias); nothing else persists.
Arithmetic: exact +-x products, FP32 accumulation (FFMA), FP32 scale/bias epilogue; no activation
quantization. Split-K partial sums are reduced in fixed order inside one kernel (deterministic).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from gpu1bit.shared import api

from . import kernels
from .kernels import Config

# Per-shape choices (N, K) -> [(max M, Config)], screened on this RTX 5090 (tune.py); the first
# band whose max M >= M is used. Shapes or M outside the table use `default_config`.
TUNED: dict[tuple[int, int], list[tuple[int, Config]]] = {
    # C sweep 20261020T084728 (tune.py; M = 38 / 126 / 368 dev clips; warm graph replay)
    (3072, 1024): [(64, Config(16, 64, 64, 4, 4, 2)), (192, Config(32, 128, 32, 8, 4, 3)), (1073741824, Config(64, 256, 32, 8, 8, 2))],
    (1024, 1024): [(64, Config(16, 64, 64, 4, 4, 6)), (192, Config(32, 128, 32, 8, 4, 4)), (1073741824, Config(32, 128, 32, 8, 4, 3))],
    (4096, 1024): [(64, Config(40, 128, 32, 4, 4, 4)), (192, Config(32, 128, 32, 8, 4, 2)), (1073741824, Config(96, 128, 32, 8, 4, 1))],
    (1024, 4096): [(64, Config(16, 64, 64, 4, 4, 6)), (192, Config(32, 128, 32, 8, 4, 8)), (1073741824, Config(32, 128, 32, 8, 4, 3))],
    (2048, 1024): [(64, Config(40, 64, 64, 4, 4, 4)), (192, Config(32, 128, 64, 8, 4, 2)), (1073741824, Config(64, 256, 32, 8, 8, 3))],
}


def default_config(m: int, n: int, k: int) -> Config:
    if m <= 16:
        bm, tm = 16, 4
    elif m <= 32:
        bm, tm = 32, 4
    else:
        bm, tm = 64, 8
    bn, tn, bk = 128, 8, 32
    tiles = math.ceil(m / bm) * math.ceil(n / bn)
    ktiles = math.ceil(k / bk)
    split = max(1, min(8, ktiles, round(2 * 170 / tiles)))
    while split > 1 and math.ceil(ktiles / split) * (split - 1) >= ktiles:  # avoid empty splits
        split -= 1
    return Config(bm, bn, bk, tm, tn, split)


def choose(m: int, n: int, k: int) -> Config:
    for limit, config in TUNED.get((n, k), []):
        if m <= limit:
            return config
    return default_config(m, n, k)


class BitSerialLayer:
    """One MB1 layer: codes uint8 [N, ceil(K/8)] resident as exported, FP32 scale, optional bias."""

    def __init__(self, codes: Tensor, scale: Tensor, bias: Tensor | None, n: int, k: int):
        self.codes, self.scale, self.bias = codes, scale, bias
        self.out_features, self.in_features = n, k
        self.kb = api.packed_row_bytes(k)
        self.device = codes.device
        self.config_override: Config | None = None  # tuning hook; None = choose()

    @classmethod
    def prepare(cls, codes: Tensor, scale: Tensor, bias: Tensor | None, in_features: int,
                device) -> "BitSerialLayer":
        n, k = api.validate_weights(codes, scale, bias, in_features)
        device = torch.device(device)
        if device.type != 'cuda':
            raise ValueError('bitserial kernels need a CUDA device')
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        if torch.cuda.get_device_capability(device) != (12, 0):
            raise ValueError('bitserial kernels target SM120 (RTX 5090)')
        # Own copies (never alias the caller's tensors), contiguous, on the device.
        own = lambda t: t.detach().to(device=device, copy=True).contiguous()  # noqa: E731
        return cls(own(codes), own(scale), None if bias is None else own(bias), n, k)

    def weight_bytes(self) -> int:
        return api.resident_bytes(self.codes, self.scale, self.bias)

    # -- calls ---------------------------------------------------------------------------------

    def _check(self, x: Tensor) -> None:
        if not isinstance(x, Tensor) or x.dtype != torch.float32:
            raise ValueError('expected an FP32 tensor')
        if x.device != self.device:
            raise ValueError(f'expected input on {self.device}, got {x.device}')
        if torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError('bitserial kernels are inference-only; use torch.no_grad()/inference_mode()')

    def _run(self, x3: Tensor, *, act: int = 0, in_act: int = 0) -> Tensor:
        """x3: [B, T, K] view, any strides -> new contiguous [B*T, N]."""
        b, t, k = x3.shape
        m, n = b * t, self.out_features
        y = torch.empty((m, n), device=self.device, dtype=torch.float32)
        if m == 0:
            return y
        if any(s >= 2**31 for s in x3.stride()) or m >= 2**31:
            raise ValueError('input too large for 32-bit strides')
        config = self.config_override or choose(m, n, k)
        xb, xt, xk = x3.stride()
        if (xk == 1 and k % 4 == 0 and (b == 1 or xb % 4 == 0) and (t == 1 or xt % 4 == 0)
                and x3.data_ptr() % 16 == 0):
            xmode = 1
        elif xt == 1 and t > 1:
            xmode = 2
        else:
            xmode = 0
        wfast = k % config.bk == 0 and self.kb % 4 == 0
        fn = kernels.kernel(config, xmode=xmode, wfast=wfast, has_bias=self.bias is not None,
                            act=act, in_act=in_act)
        kernels.launch(fn, config, x3, t, self.codes, self.kb, self.scale, self.bias, y, m, n, k)
        return y

    def linear(self, x: Tensor, *, activation: str = '', input_activation: str = '') -> Tensor:
        self._check(x)
        if x.dim() < 1 or x.shape[-1] != self.in_features:
            raise ValueError(f'expected [..., {self.in_features}] input, got {tuple(x.shape)}')
        y = self._run(_as3(x), act=_act(activation), in_act=_act(input_activation))
        return y.view(*x.shape[:-1], self.out_features)

    def conv(self, x: Tensor, *, activation: str = '', input_activation: str = '') -> Tensor:
        self._check(x)
        if x.dim() != 3 or x.shape[1] != self.in_features:
            raise ValueError(f'expected [B, {self.in_features}, T] input, got {tuple(x.shape)}')
        b, _, t = x.shape
        y = self._run(x.transpose(1, 2), act=_act(activation), in_act=_act(input_activation))
        return y.view(b, t, self.out_features).transpose(1, 2)


    def project(self, x: Tensor, *, conv: bool = False, components: int = 3, activation: str = '',
                norm=None, input_activation: str = '') -> Tensor:
        """Fused provider op (gpu1bit/shared/provider.py, semantics of M1's ExpandedProjection):
        LayerNorm(norm) and/or SiLU on the input, the 1-bit projection, SiLU on the output.
        `components` is ignored: this arithmetic is FP32 (exact +-x products, FP32 accumulation),
        not INT8 residual parts. With a norm or input SiLU, one row-prep kernel writes the
        transformed rows contiguously (M1's INT8x3 path likewise runs its quantize kernel first),
        then the GEMM runs with the output SiLU fused in its epilogue."""
        self._check(x)
        act, in_silu = _act(activation), _act(input_activation) == 1
        if conv:
            if x.dim() != 3 or x.shape[1] != self.in_features:
                raise ValueError(f'expected [B, {self.in_features}, T] input, got {tuple(x.shape)}')
            x3 = x.transpose(1, 2)
        else:
            if x.dim() < 1 or x.shape[-1] != self.in_features:
                raise ValueError(f'expected [..., {self.in_features}] input, got {tuple(x.shape)}')
            x3 = _as3(x)
        if norm is not None:
            _norm_params(norm, self.in_features, self.device)
        if norm is not None or in_silu:
            x3 = kernels.row_prep(x3, norm, in_silu).unsqueeze(0)
        y = self._run(x3, act=act)
        if conv:
            b, _, t = x.shape
            return y.view(b, t, self.out_features).transpose(1, 2)
        return y.view(*x.shape[:-1], self.out_features)


def _as3(x: Tensor) -> Tensor:
    """[..., K] -> a [B, T, K] view (no copy for 1-3 dims; >3 dims merge leading dims, copying if needed)."""
    if x.dim() == 1:
        return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 2:
        return x.unsqueeze(0)
    if x.dim() == 3:
        return x
    return x.reshape(-1, x.shape[-2], x.shape[-1])


def _norm_params(norm, k: int, device) -> tuple[Tensor, Tensor]:
    weight, bias = getattr(norm, 'weight', None), getattr(norm, 'bias', None)
    if weight is None or bias is None or tuple(norm.normalized_shape) != (k,):
        raise ValueError('expected an affine LayerNorm over the last dimension')
    for p in (weight, bias):
        if p.dtype != torch.float32 or p.device != device or not p.is_contiguous():
            raise ValueError('LayerNorm parameters must be contiguous FP32 on the layer device')
    return weight, bias


def _act(name: str) -> int:
    if name == '':
        return 0
    if name == 'silu':
        return 1
    raise ValueError(f'unknown activation {name!r}')


IMPLEMENTATION = BitSerialLayer
