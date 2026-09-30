"""Ternary absmean weight quantization for Whisper QAT. See DESIGN.md "Quantizer" and "Revision 2".

Codes are {-1, 0, +1} with one FP32 absmean scale per output row (per vocabulary
row for the embedding), recomputed from the FP32 latent weight on every forward.
The straight-through estimator W_hat + (W - W.detach()) uses exactly W_hat in
the forward pass (the added term is exactly zero, so at weight fraction 1 the
training graph matches the export bit for bit) and passes the gradient to the
latent weight unchanged; no gradient reaches the scale. Activations are not
quantized. The forward pass still runs dense floating-point matrices: nothing
here claims a speed or energy benefit, only the accuracy of a model whose
weights are ternary.

Progressive quantization (Revision 2): every ternary module has a weight
fraction f in [0, 1], default 1. Its forward weight is lerp(W, W_hat_ste, f):
the latent weight itself at f = 0 (bit for bit a plain nn.Linear / nn.Embedding),
exactly W_hat at f = 1. The gradient to the latent weight is the identity for
every f. f < 1 is a training-time ramp state only and never a deployable model:
the export refuses it. quantized_weight(), code_histogram() and
parameter_accounting() always describe the fully quantized weights, whatever f is.
"""
from __future__ import annotations

import math
import numbers
import re
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn

if TYPE_CHECKING:
    from transformers import WhisperForConditionalGeneration

# DESIGN.md "Quantized in A2"; use with fullmatch on named_modules() names.
QUANTIZED_LINEAR_PATTERN = re.compile(
    r"model\.(encoder\.layers\.\d+\.self_attn|decoder\.layers\.\d+\.(self_attn|encoder_attn))"
    r"\.(q|k|v|out)_proj|model\.(encoder|decoder)\.layers\.\d+\.fc[12]")
EMBEDDING = "model.decoder.embed_tokens"
OUTPUT = "proj_out"


def ternary_quantize(weight: Tensor) -> tuple[Tensor, Tensor]:
    """Per-row absmean codes (int8 [out, in]) and scales (FP32 [out]); not differentiable."""
    if weight.dim() != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(weight.shape)}")
    w = weight.detach().float()
    scale = w.abs().mean(dim=1).clamp_min(1e-8)
    codes = torch.round(w / scale[:, None]).clamp_(-1, 1).to(torch.int8)
    return codes, scale


def dequantize(codes: Tensor, scale: Tensor) -> Tensor:
    """codes * scale per row, FP32 [out, in]."""
    return codes.float() * scale.float()[:, None]


def _ste(weight: Tensor) -> Tensor:
    """Value bit-identical to codes*scale; identity gradient to the latent weight."""
    return dequantize(*ternary_quantize(weight)) + (weight - weight.detach())


def _check_fraction(fraction: float) -> float:
    """The validated weight fraction as a Python float; rejects bools, non-reals, NaN and values outside [0, 1]."""
    if isinstance(fraction, bool) or not isinstance(fraction, numbers.Real):
        raise TypeError(f"weight fraction must be a real number, got {type(fraction).__name__}")
    fraction = float(fraction)
    if math.isnan(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError(f"weight fraction must be in [0, 1], got {fraction}")
    return fraction


def _effective_weight(weight: Tensor, fraction: float) -> Tensor:
    """lerp(W, W_hat_ste, f); exactly W_hat_ste at f >= 1. Identity gradient to W for every f."""
    w_hat = _ste(weight)
    return w_hat if fraction >= 1.0 else torch.lerp(weight, w_hat, fraction)


def _check_latent(weight: Tensor) -> None:
    if not isinstance(weight, nn.Parameter) or weight.dtype != torch.float32 or weight.dim() != 2:
        raise ValueError("ternary modules need a 2-D FP32 nn.Parameter latent weight")


class _WeightFraction:
    """The progressive-quantization weight fraction f in [0, 1] (default 1), validated on assignment.

    A plain attribute, not a buffer: it is not in state_dict, so checkpoints do
    not carry it; the trainer sets it (set_weight_fraction) before every step.
    f < 1 is a training-time ramp state only, never a deployable model.
    """

    _fraction: float = 1.0

    @property
    def fraction(self) -> float:
        return self._fraction

    @fraction.setter
    def fraction(self, value: float) -> None:
        self._fraction = _check_fraction(value)


class TernaryLinear(_WeightFraction, nn.Module):
    """nn.Linear replacement whose forward uses the ternary dequantized weight.

    Holds the caller's Parameter objects (not copies), so optimizer state,
    state_dict keys and weight ties are unchanged by the swap. With
    fraction < 1 (the Revision 2 training ramp only; never deployable) the
    forward weight is lerp(W, W_hat, fraction) instead of W_hat.
    """

    def __init__(self, weight: nn.Parameter, bias: nn.Parameter | None) -> None:
        super().__init__()
        _check_latent(weight)
        self.out_features, self.in_features = weight.shape
        self.weight = weight
        self.register_parameter("bias", bias)
        self.fraction = 1.0

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, _effective_weight(self.weight, self.fraction), self.bias)

    def quantized_weight(self) -> tuple[Tensor, Tensor]:
        """Codes and scales of the fully quantized weight (what the export stores), independent of fraction."""
        return ternary_quantize(self.weight)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, "
                f"fraction={self.fraction}")

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> TernaryLinear:
        if not isinstance(linear, nn.Linear):
            raise ValueError(f"expected nn.Linear, got {type(linear).__name__}")
        return cls(linear.weight, linear.bias)

    @classmethod
    def from_parameter(cls, weight: nn.Parameter, bias: nn.Parameter | None) -> TernaryLinear:
        return cls(weight, bias)


class TernaryEmbedding(_WeightFraction, nn.Module):
    """nn.Embedding replacement returning dequantized rows (per-vocabulary-row scale).

    With fraction < 1 (the Revision 2 training ramp only; never deployable) the
    rows come from lerp(W, W_hat, fraction) instead of W_hat.
    """

    def __init__(self, weight: nn.Parameter, padding_idx: int | None) -> None:
        super().__init__()
        _check_latent(weight)
        self.num_embeddings, self.embedding_dim = weight.shape
        self.padding_idx = padding_idx
        self.weight = weight
        self.fraction = 1.0

    def forward(self, ids: Tensor) -> Tensor:
        return F.embedding(ids, _effective_weight(self.weight, self.fraction), padding_idx=self.padding_idx)

    def quantized_weight(self) -> tuple[Tensor, Tensor]:
        """Codes and scales of the fully quantized weight (what the export stores), independent of fraction."""
        return ternary_quantize(self.weight)

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}, padding_idx={self.padding_idx}, fraction={self.fraction}"

    @classmethod
    def from_embedding(cls, emb: nn.Embedding) -> TernaryEmbedding:
        if not isinstance(emb, nn.Embedding) or emb.max_norm is not None or emb.scale_grad_by_freq or emb.sparse:
            raise ValueError("expected a plain nn.Embedding (no max_norm, scale_grad_by_freq or sparse)")
        return cls(emb.weight, emb.padding_idx)


def quantized_module_names(model: nn.Module) -> list[str]:
    return sorted(n for n, m in model.named_modules() if isinstance(m, (TernaryLinear, TernaryEmbedding)))


def _ternary_modules(model: nn.Module) -> list[TernaryLinear | TernaryEmbedding]:
    modules = [m for m in model.modules() if isinstance(m, (TernaryLinear, TernaryEmbedding))]
    if not modules:
        raise ValueError("model has no ternary modules")
    return modules


def set_weight_fraction(model: nn.Module, fraction: float) -> int:
    """Set the progressive-quantization fraction on every ternary module; return how many were set.

    Validates 0 <= fraction <= 1 before touching any module (TypeError for a
    non-real or bool, ValueError for NaN or out of range) and raises ValueError
    if the model has no ternary modules. Both halves of the tied embedding /
    output projection are set. fraction < 1 is a training-time ramp state only;
    set 1.0 before exporting or reporting a deployable result.
    """
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


def quantize_model(model: WhisperForConditionalGeneration, include_embedding: bool) -> list[str]:
    """Swap the DESIGN.md projection set (and optionally the tied embedding) in place."""
    if quantized_module_names(model):
        raise ValueError("model already contains ternary modules")
    targets = [n for n, _ in model.named_modules() if QUANTIZED_LINEAR_PATTERN.fullmatch(n)]
    expected = 6 * model.config.encoder_layers + 10 * model.config.decoder_layers
    if len(targets) != expected:
        raise ValueError(f"matched {len(targets)} projection modules, expected {expected}")
    for name in targets:
        model.set_submodule(name, TernaryLinear.from_linear(model.get_submodule(name)))
    if include_embedding:
        emb = model.get_submodule(EMBEDDING)
        if model.proj_out.weight is not emb.weight:
            raise ValueError("proj_out is not tied to the decoder embedding")
        ternary = TernaryEmbedding.from_embedding(emb)
        model.set_submodule(EMBEDDING, ternary)
        model.set_submodule(OUTPUT, TernaryLinear.from_parameter(ternary.weight, None))
        if model.proj_out.weight is not model.get_submodule(EMBEDDING).weight:
            raise RuntimeError("tie between proj_out and the decoder embedding was lost")
    return quantized_module_names(model)


def _unique_quantized(model: nn.Module) -> list[tuple[str, TernaryLinear | TernaryEmbedding]]:
    """Quantized modules in name order, skipping any whose latent weight was already seen (the tie)."""
    seen: set[int] = set()
    result = []
    for name in quantized_module_names(model):
        module = model.get_submodule(name)
        if id(module.weight) not in seen:
            seen.add(id(module.weight))
            result.append((name, module))
    return result


def _fractions(counts: Tensor) -> dict[str, float]:
    total = int(counts.sum())
    return {key: int(c) / total for key, c in zip(("minus_one", "zero", "plus_one"), counts.tolist())}


def code_histogram(model: nn.Module) -> dict:
    """Fractions of -1 / 0 / +1 over all quantized weights (tie counted once) and per layer.

    Always describes the fully quantized codes, independent of the weight fraction.
    """
    modules = _unique_quantized(model)
    if not modules:
        raise ValueError("model has no ternary modules")
    total = torch.zeros(3, dtype=torch.long)
    per_layer = {}
    for name, module in modules:
        codes, _ = module.quantized_weight()
        counts = torch.bincount(codes.flatten().long() + 1, minlength=3).cpu()
        total += counts
        per_layer[name] = _fractions(counts)
    return {**_fractions(total), "per_layer": per_layer}


def parameter_accounting(model: nn.Module) -> dict[str, int]:
    """Unique parameter counts (the tie counted once) split into ternary and residual."""
    total = sum(p.numel() for p in model.parameters())
    ternary = sum(module.weight.numel() for _, module in _unique_quantized(model))
    return {"ternary_parameters": ternary, "residual_parameters": total - ternary, "total_parameters": total}
