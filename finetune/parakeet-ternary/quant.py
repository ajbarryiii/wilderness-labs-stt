"""Ternary absmean weight quantization for Parakeet-TDT QAT. See DESIGN.md "Quantized set" and "Quantizer".

The quantizer math is not reimplemented: ternary_quantize, dequantize, the
straight-through estimator and the lerp(W, W_hat_ste, f) weight-fraction ramp are
imported unchanged from finetune/whisper-ternary/quant.py (loaded by file path as
module "whisper_ternary_quant", because this directory's quant.py shadows it on
PYTHONPATH). Per output row s = max(mean|W|, 1e-8), C = clamp(round(W / s), -1, 1),
forward weight exactly C * s at fraction 1, identity gradient to the FP32 latent
weight at every fraction, no gradient to the scale, activations not quantized.

Quantized set (QUANTIZED_PATTERN, fullmatch on named_modules() names): in each of
the 24 FastConformer layers feed_forward{1,2}.linear{1,2}, self_attn.linear_{q,k,v,
out,pos} and conv.pointwise_conv{1,2}; 24 * 11 = 264 modules. The pointwise
convolutions are Conv1d with kernel 1; their [out, in, 1] latent weight is quantized
as the [out, in] matrix. Everything else (pre_encode, depthwise convolutions, norms,
biases, prediction and joint networks) stays in floating point.

The ternary modules hold the original Parameter objects, so state_dict keys, shapes
(pointwise conv weights stay [out, in, 1]) and optimizer bindings are unchanged by
quantize_parakeet. The weight fraction is a plain attribute, not a buffer: it is not
saved in state_dict, so a trainer must call set_weight_fraction after loading. A
fraction below 1 is a training-time ramp state only; export refuses it.
"""
from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import paths

WHISPER_DIR = paths.REPO / "finetune" / "whisper-ternary"


def load_whisper_module(name: str) -> ModuleType:
    """Import finetune/whisper-ternary/<name>.py as module "whisper_ternary_<name>" (cached in sys.modules)."""
    key = f"whisper_ternary_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, WHISPER_DIR / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def whisper_source_sha256(name: str) -> str:
    return hashlib.sha256((WHISPER_DIR / f"{name}.py").read_bytes()).hexdigest()


_wq = load_whisper_module("quant")
ternary_quantize = _wq.ternary_quantize          # (weight [out,in]) -> int8 codes [out,in], fp32 scale [out]
dequantize = _wq.dequantize                      # codes * scale per row, FP32
_effective_weight = _wq._effective_weight        # lerp(W, W_hat_ste, f); exactly W_hat_ste at f >= 1
_check_fraction = _wq._check_fraction
_WeightFraction = _wq._WeightFraction

LAYERS = 24
PER_LAYER = ("feed_forward1.linear1", "feed_forward1.linear2", "feed_forward2.linear1", "feed_forward2.linear2",
             "self_attn.linear_q", "self_attn.linear_k", "self_attn.linear_v", "self_attn.linear_out",
             "self_attn.linear_pos", "conv.pointwise_conv1", "conv.pointwise_conv2")
EXPECTED_MODULES = LAYERS * len(PER_LAYER)  # 264
TOTAL_PARAMETERS = 617_825_926
QUANTIZED_PATTERN = re.compile(
    r"encoder\.layers\.\d+\.(feed_forward[12]\.linear[12]|self_attn\.linear_(q|k|v|out|pos)"
    r"|conv\.pointwise_conv[12])")


class TernaryLinear(_WeightFraction, nn.Module):
    """nn.Linear replacement whose forward weight is lerp(W, W_hat_ste, fraction) (exactly W_hat at 1).

    Holds the caller's Parameter objects, not copies.
    """

    def __init__(self, weight: nn.Parameter, bias: nn.Parameter | None) -> None:
        super().__init__()
        _check_parameter(weight, 2)
        self.out_features, self.in_features = weight.shape
        self.weight = weight
        self.register_parameter("bias", bias)
        self.fraction = 1.0

    def matrix(self) -> Tensor:
        """The latent weight as [out, in] (a view that keeps the autograd link)."""
        return self.weight

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, _effective_weight(self.weight, self.fraction), self.bias)

    def quantized_weight(self) -> tuple[Tensor, Tensor]:
        """int8 codes [out, in] and FP32 scales [out] of the fully quantized weight, whatever the fraction."""
        return ternary_quantize(self.weight)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, "
                f"fraction={self.fraction}")

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> TernaryLinear:
        if type(linear) is not nn.Linear:
            raise ValueError(f"expected nn.Linear, got {type(linear).__name__}")
        return cls(linear.weight, linear.bias)


class TernaryPointwiseConv1d(_WeightFraction, nn.Module):
    """Conv1d(kernel_size=1) replacement: the [out, in, 1] latent weight is quantized as the [out, in] matrix.

    The forward is F.conv1d with the effective weight reshaped back to [out, in, 1], the same
    call the original module makes, so fraction 0 reproduces it bit for bit.
    """

    def __init__(self, weight: nn.Parameter, bias: nn.Parameter | None) -> None:
        super().__init__()
        _check_parameter(weight, 3)
        if weight.shape[2] != 1:
            raise ValueError(f"expected a kernel-1 Conv1d weight [out, in, 1], got {tuple(weight.shape)}")
        self.out_channels, self.in_channels = weight.shape[:2]
        self.weight = weight
        self.register_parameter("bias", bias)
        self.fraction = 1.0

    def matrix(self) -> Tensor:
        return self.weight.squeeze(2)

    def forward(self, x: Tensor) -> Tensor:
        return F.conv1d(x, _effective_weight(self.matrix(), self.fraction).unsqueeze(2), self.bias)

    def quantized_weight(self) -> tuple[Tensor, Tensor]:
        return ternary_quantize(self.matrix())

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, kernel_size=(1,), bias={self.bias is not None}, "
                f"fraction={self.fraction}")

    @classmethod
    def from_conv(cls, conv: nn.Conv1d) -> TernaryPointwiseConv1d:
        if type(conv) is not nn.Conv1d:
            raise ValueError(f"expected nn.Conv1d, got {type(conv).__name__}")
        plain = (conv.kernel_size == (1,) and conv.stride == (1,) and conv.padding == (0,)
                 and conv.dilation == (1,) and conv.groups == 1 and conv.padding_mode == "zeros")
        if not plain:
            raise ValueError(f"expected a plain pointwise Conv1d, got {conv}")
        return cls(conv.weight, conv.bias)


TERNARY_TYPES = (TernaryLinear, TernaryPointwiseConv1d)


def _check_parameter(weight: Tensor, dim: int) -> None:
    if not isinstance(weight, nn.Parameter) or weight.dtype != torch.float32 or weight.dim() != dim:
        raise ValueError(f"ternary modules need a {dim}-D FP32 nn.Parameter latent weight")


def quantized_module_names(model: nn.Module) -> list[str]:
    return sorted(n for n, m in model.named_modules() if isinstance(m, TERNARY_TYPES))


def _ternary_modules(model: nn.Module) -> list[nn.Module]:
    modules = [m for m in model.modules() if isinstance(m, TERNARY_TYPES)]
    if not modules:
        raise ValueError("model has no ternary modules")
    return modules


def quantize_parakeet(model: nn.Module) -> list[str]:
    """Swap the DESIGN.md quantized set for ternary modules in place; return the sorted names.

    Raises ValueError if the model already has ternary modules or the pattern does not match
    exactly 264 modules. All swapped modules start at fraction 1.
    """
    if quantized_module_names(model):
        raise ValueError("model already contains ternary modules")
    targets = [n for n, _ in model.named_modules() if QUANTIZED_PATTERN.fullmatch(n)]
    if len(targets) != EXPECTED_MODULES:
        raise ValueError(f"matched {len(targets)} modules, expected {EXPECTED_MODULES}")
    for name in targets:
        module = model.get_submodule(name)
        ternary = (TernaryPointwiseConv1d.from_conv(module) if isinstance(module, nn.Conv1d)
                   else TernaryLinear.from_linear(module))
        model.set_submodule(name, ternary)
    names = quantized_module_names(model)
    assert names == sorted(targets) and len(names) == EXPECTED_MODULES
    return names


def set_weight_fraction(model: nn.Module, fraction: float) -> int:
    """Set the ramp fraction on every ternary module (validated first); return how many were set."""
    fraction = _check_fraction(fraction)
    modules = _ternary_modules(model)
    for module in modules:
        module.fraction = fraction
    return len(modules)


def weight_fraction(model: nn.Module) -> float:
    """The common fraction of every ternary module; ValueError if there are none or they disagree."""
    fractions = {module.fraction for module in _ternary_modules(model)}
    if len(fractions) != 1:
        raise ValueError(f"ternary modules disagree on the weight fraction: {sorted(fractions)}")
    return fractions.pop()


def _fractions(counts: Tensor) -> dict[str, float]:
    total = int(counts.sum())
    return {key: int(c) / total for key, c in zip(("minus_one", "zero", "plus_one"), counts.tolist())}


@torch.no_grad()
def code_histogram(model: nn.Module) -> dict:
    """Fractions of -1 / 0 / +1 over all quantized weights and per module (fully quantized codes)."""
    names = quantized_module_names(model)
    if not names:
        raise ValueError("model has no ternary modules")
    total = torch.zeros(3, dtype=torch.long)
    per_module = {}
    for name in names:
        codes, _ = model.get_submodule(name).quantized_weight()
        counts = torch.bincount(codes.flatten().long() + 1, minlength=3).cpu()
        total += counts
        per_module[name] = _fractions(counts)
    return {**_fractions(total), "per_module": per_module}


def parameter_accounting(model: nn.Module) -> dict[str, int]:
    """Unique parameter counts split into ternary (quantized weights) and float (everything else)."""
    total = sum(p.numel() for p in model.parameters())
    ternary = sum(model.get_submodule(n).weight.numel() for n in quantized_module_names(model))
    return {"ternary_parameters": ternary, "float_parameters": total - ternary, "total_parameters": total}
