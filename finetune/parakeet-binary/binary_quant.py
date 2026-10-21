"""Binary (sign) weight quantization for the Parakeet-TDT binary pilot. See plans/pilot.md "What changes".

Per output row alpha = max(mean|W|, 1e-8) (the least-squares scale for sign codes, the same
statistic as the ternary scale), codes c = +1 where W >= 0 and -1 where W < 0 (an exact zero,
of either sign, maps to +1; torch.sign is not used because it returns 0), forward weight
exactly c * alpha at weight fraction 1, identity straight-through gradient to the FP32 latent
weight, no gradient to alpha, activations not quantized.

Ramps (plans/pilot.md "Ramps"), one per model, set by quantize_parakeet:
- "fp32" (arm A): W_eff = lerp(W, B_ste(W), f), the ternary project's ramp with the binary
  quantizer; f = 0 is the float module bit for bit.
- "ternary" (arms B5, B2): W_eff = lerp(T_ste(W), B_ste(W), f), ternary to binary. T_ste is
  parakeet-ternary quant._effective_weight(W, 1.0), i.e. exactly the ternary module's forward
  weight (Whisper quantizer, imported unchanged), so f = 0 is M1's forward bit for bit.
Both are exactly B_ste(W) at f = 1, and the gradient to W is the identity at every f (up to
the rounding of (1 - f) + f, as in the ternary ramp).

Quantized set: parakeet-ternary's QUANTIZED_PATTERN (imported, not copied), 264 modules; the
binary modules hold the original Parameter objects, so state_dict keys, shapes (pointwise
conv weights stay [out, in, 1]) and optimizer bindings are unchanged. The weight fraction and
the ramp are plain attributes, not in state_dict: a trainer quantizes with the arm's ramp and
calls set_weight_fraction after loading. A fraction below 1 is a training-time state only;
the export refuses it.
"""
from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import quant as ternary  # finetune/parakeet-ternary/quant.py, unchanged (M1's provenance)

QUANTIZED_PATTERN = ternary.QUANTIZED_PATTERN
EXPECTED_MODULES = ternary.EXPECTED_MODULES  # 264
TOTAL_PARAMETERS = ternary.TOTAL_PARAMETERS
dequantize = ternary.dequantize  # codes * scale per row, FP32 (exact for codes in {-1, +1})
_check_fraction = ternary._check_fraction
_WeightFraction = ternary._WeightFraction
RAMPS = ("fp32", "ternary")
FAMILIES = (("feed_forward", ".feed_forward"), ("self_attn", ".self_attn."), ("pointwise_conv", ".conv.pointwise_conv"))


def binary_quantize(weight: Tensor) -> tuple[Tensor, Tensor]:
    """Per-row sign codes (int8 [out, in], +1 for W >= 0, else -1) and absmean scales (FP32 [out]); not differentiable."""
    if weight.dim() != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(weight.shape)}")
    w = weight.detach().float()
    scale = w.abs().mean(dim=1).clamp_min(1e-8)
    codes = torch.where(w >= 0, 1, -1).to(torch.int8)
    return codes, scale


def _binary_ste(weight: Tensor) -> Tensor:
    """Value bit-identical to codes * scale; identity gradient to the latent weight."""
    return dequantize(*binary_quantize(weight)) + (weight - weight.detach())


def _ternary_ste(weight: Tensor) -> Tensor:
    """The ternary module's forward weight at fraction 1 (Whisper ternary STE)."""
    return ternary._effective_weight(weight, 1.0)


def _check_ramp(ramp: str) -> str:
    if ramp not in RAMPS:
        raise ValueError(f"ramp must be one of {RAMPS}, got {ramp!r}")
    return ramp


def _effective_weight(weight: Tensor, fraction: float, ramp: str) -> Tensor:
    """lerp(start, B_ste, f) with start W ("fp32") or T_ste(W) ("ternary"); exactly B_ste at f >= 1."""
    b_hat = _binary_ste(weight)
    if fraction >= 1.0:
        return b_hat
    start = weight if ramp == "fp32" else _ternary_ste(weight)
    return torch.lerp(start, b_hat, fraction)


class BinaryLinear(_WeightFraction, nn.Module):
    """nn.Linear replacement whose forward weight is lerp(start, B_ste, fraction) (exactly B_hat at 1).

    Holds the caller's Parameter objects, not copies.
    """

    def __init__(self, weight: nn.Parameter, bias: nn.Parameter | None, ramp: str) -> None:
        super().__init__()
        _check_parameter(weight, 2)
        self.out_features, self.in_features = weight.shape
        self.weight = weight
        self.register_parameter("bias", bias)
        self.ramp = _check_ramp(ramp)
        self.fraction = 1.0

    def matrix(self) -> Tensor:
        """The latent weight as [out, in] (a view that keeps the autograd link)."""
        return self.weight

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, _effective_weight(self.weight, self.fraction, self.ramp), self.bias)

    def quantized_weight(self) -> tuple[Tensor, Tensor]:
        """int8 codes [out, in] in {-1, +1} and FP32 scales [out] of the fully binary weight, whatever the fraction."""
        return binary_quantize(self.weight)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, "
                f"ramp={self.ramp}, fraction={self.fraction}")

    @classmethod
    def from_linear(cls, linear: nn.Linear, ramp: str) -> BinaryLinear:
        plain = ternary.TernaryLinear.from_linear(linear)  # the ternary project's type check
        return cls(plain.weight, plain.bias, ramp)


class BinaryPointwiseConv1d(_WeightFraction, nn.Module):
    """Conv1d(kernel_size=1) replacement: the [out, in, 1] latent weight is quantized as the [out, in] matrix.

    The forward is F.conv1d with the effective weight reshaped back to [out, in, 1], the same
    call the original module makes, so the "fp32" ramp at fraction 0 reproduces it bit for bit.
    """

    def __init__(self, weight: nn.Parameter, bias: nn.Parameter | None, ramp: str) -> None:
        super().__init__()
        _check_parameter(weight, 3)
        if weight.shape[2] != 1:
            raise ValueError(f"expected a kernel-1 Conv1d weight [out, in, 1], got {tuple(weight.shape)}")
        self.out_channels, self.in_channels = weight.shape[:2]
        self.weight = weight
        self.register_parameter("bias", bias)
        self.ramp = _check_ramp(ramp)
        self.fraction = 1.0

    def matrix(self) -> Tensor:
        return self.weight.squeeze(2)

    def forward(self, x: Tensor) -> Tensor:
        return F.conv1d(x, _effective_weight(self.matrix(), self.fraction, self.ramp).unsqueeze(2), self.bias)

    def quantized_weight(self) -> tuple[Tensor, Tensor]:
        return binary_quantize(self.matrix())

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, kernel_size=(1,), bias={self.bias is not None}, "
                f"ramp={self.ramp}, fraction={self.fraction}")

    @classmethod
    def from_conv(cls, conv: nn.Conv1d, ramp: str) -> BinaryPointwiseConv1d:
        plain = ternary.TernaryPointwiseConv1d.from_conv(conv)  # the ternary project's plain-pointwise check
        return cls(plain.weight, plain.bias, ramp)


BINARY_TYPES = (BinaryLinear, BinaryPointwiseConv1d)


def _check_parameter(weight: Tensor, dim: int) -> None:
    if not isinstance(weight, nn.Parameter) or weight.dtype != torch.float32 or weight.dim() != dim:
        raise ValueError(f"binary modules need a {dim}-D FP32 nn.Parameter latent weight")


def quantized_module_names(model: nn.Module) -> list[str]:
    return sorted(n for n, m in model.named_modules() if isinstance(m, BINARY_TYPES))


def _binary_modules(model: nn.Module) -> list[nn.Module]:
    modules = [m for m in model.modules() if isinstance(m, BINARY_TYPES)]
    if not modules:
        raise ValueError("model has no binary modules")
    return modules


def quantize_parakeet(model: nn.Module, ramp: str) -> list[str]:
    """Swap the quantized set for binary modules with this ramp in place; return the sorted names.

    Raises ValueError if the model already has binary or ternary modules or the pattern does
    not match exactly EXPECTED_MODULES (264) modules. All swapped modules start at fraction 1.
    """
    ramp = _check_ramp(ramp)
    if quantized_module_names(model) or ternary.quantized_module_names(model):
        raise ValueError("model already contains binary or ternary modules")
    targets = [n for n, _ in model.named_modules() if QUANTIZED_PATTERN.fullmatch(n)]
    if len(targets) != EXPECTED_MODULES:
        raise ValueError(f"matched {len(targets)} modules, expected {EXPECTED_MODULES}")
    for name in targets:
        module = model.get_submodule(name)
        binary = (BinaryPointwiseConv1d.from_conv(module, ramp) if isinstance(module, nn.Conv1d)
                  else BinaryLinear.from_linear(module, ramp))
        model.set_submodule(name, binary)
    names = quantized_module_names(model)
    assert names == sorted(targets) and len(names) == EXPECTED_MODULES
    return names


def set_weight_fraction(model: nn.Module, fraction: float) -> int:
    """Set the ramp fraction on every binary module (validated first); return how many were set."""
    fraction = _check_fraction(fraction)
    modules = _binary_modules(model)
    for module in modules:
        module.fraction = fraction
    return len(modules)


def weight_fraction(model: nn.Module) -> float:
    """The common fraction of every binary module; ValueError if there are none or they disagree."""
    fractions = {module.fraction for module in _binary_modules(model)}
    if len(fractions) != 1:
        raise ValueError(f"binary modules disagree on the weight fraction: {sorted(fractions)}")
    return fractions.pop()


def ramp_kind(model: nn.Module) -> str:
    """The common ramp of every binary module; ValueError if there are none or they disagree."""
    ramps = {module.ramp for module in _binary_modules(model)}
    if len(ramps) != 1:
        raise ValueError(f"binary modules disagree on the ramp: {sorted(ramps)}")
    return ramps.pop()


def family(name: str) -> str:
    """feed_forward, self_attn or pointwise_conv for a quantized module name."""
    for label, marker in FAMILIES:
        if marker in name:
            return label
    raise ValueError(f"{name} is not in a quantized family")


def _fractions(counts: Tensor) -> dict[str, float]:
    total = int(counts.sum())
    return {key: int(c) / total for key, c in zip(("minus_one", "zero", "plus_one"), counts.tolist())}


@torch.no_grad()
def code_balance(model: nn.Module, names: list[str]) -> dict:
    """Fractions of -1 / 0 / +1 over the fully quantized codes of these modules (binary or ternary,
    via their quantized_weight()), overall, per family and per module."""
    if not names:
        raise ValueError("no quantized modules")
    total = torch.zeros(3, dtype=torch.long)
    families: dict[str, Tensor] = defaultdict(lambda: torch.zeros(3, dtype=torch.long))
    per_module = {}
    for name in names:
        codes, _ = model.get_submodule(name).quantized_weight()
        counts = torch.bincount(codes.flatten().long() + 1, minlength=3).cpu()
        total += counts
        families[family(name)] += counts
        per_module[name] = _fractions(counts)
    return {**_fractions(total), "per_family": {k: _fractions(v) for k, v in sorted(families.items())},
            "per_module": per_module}


def code_histogram(model: nn.Module) -> dict:
    """Fractions of -1 / 0 / +1 (0 is always 0 here) over all binary weights, per family and per module."""
    names = quantized_module_names(model)
    if not names:
        raise ValueError("model has no binary modules")
    return code_balance(model, names)


def parameter_accounting(model: nn.Module) -> dict[str, int]:
    """Unique parameter counts split into binary (quantized weights) and float (everything else)."""
    total = sum(p.numel() for p in model.parameters())
    binary = sum(model.get_submodule(n).weight.numel() for n in quantized_module_names(model))
    return {"binary_parameters": binary, "float_parameters": total - binary, "total_parameters": total}

