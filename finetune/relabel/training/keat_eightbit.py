"""The 8-bit training forms of Kea-T (M1), the ternary counterpart of training/eightbit.py's Kea-B forms (keat-prime
plan "Step 0: 8-bit check of the floating-point parts (M1)" and "Code and review").

Stage 4's type-agnostic pieces are reused unchanged (training/eightbit.py): the INT8 format (int8_quantize, w8),
quantize_frozen (the frozen prediction and joint networks replaced by fp32(W8) once, at load), W8Linear and
wrap_pre_encode_out (pre_encode.out computing with W8 of its latent weight, straight-through), half_scale and
check_parts, so there is one INT8 quantizer. What is ternary lives here:

- Scopes (KEAT_SCOPES, stage 4's KEAB_SCOPES): none; S1 = the prediction network; S2 = S1 + the joint +
  pre_encode.out, with the encoder's ternary row scales stored as FP16.
- The S2 ternary form (HalfScaleTernaryLinear, HalfScaleTernaryPointwiseConv1d): at weight fraction 1 only (a lower
  fraction is refused), each module codes its latent weight as M1's quantizer does, from the FP32 scale:
  s = max(mean|W|, 1e-8) per output row, codes = clamp(round(W / s), -1, 1) (quant.ternary_quantize, so the
  threshold uses the FP32 s); the forward weight is codes * fp32(fp16(s)), and the gradient to the latent weight is
  the identity (straight-through). quantized_weight() answers (codes, fp32(fp16(s))), which the bundle stores as
  fp16(s) and the scoring view widens back exactly.
- apply_keat_scope builds a form on a ternary model (264 modules in Parakeet, every module of the quantized pattern
  in a test model) at fraction 1; keat_form reads it back and checks the state against it.
- ternarize swaps every module of quant.QUANTIZED_PATTERN for its ternary module at fraction 1, keeping the
  Parameters (quant.quantize_parakeet, which insists on 264, is the real model's swap).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import quant  # finetune/parakeet-ternary/quant.py, unchanged
from training import eightbit as eb

KEAT_SCOPES = {"none": (), "S1": ("prediction_network",), "S2": ("prediction_network", "joint", "pre_encode_out")}
SCOPES = tuple(KEAT_SCOPES)


# --- the S2 ternary form ---------------------------------------------------------------------------------

def half_scale_weight(matrix: Tensor) -> tuple[Tensor, Tensor]:
    """(codes from the FP32 scale, fp32(fp16(s))) of a [out, in] latent matrix; not differentiable."""
    codes, scale = quant.ternary_quantize(matrix)
    return codes, eb.half_scale(scale)


def _half_scale_ste(matrix: Tensor) -> Tensor:
    """Value codes * fp32(fp16(s)) bit for bit; identity gradient to the latent matrix."""
    return quant.dequantize(*half_scale_weight(matrix)) + (matrix - matrix.detach())


def _require_full(module) -> None:
    if module.fraction != 1.0:
        raise ValueError("FP16 ternary scales are defined at weight fraction 1 only")


class HalfScaleTernaryLinear(quant.TernaryLinear):
    """TernaryLinear at fraction 1 whose forward weight is codes * fp32(fp16(s)) (module docstring)."""

    def forward_weight(self) -> Tensor:
        _require_full(self)
        return _half_scale_ste(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.forward_weight(), self.bias)

    def quantized_weight(self) -> tuple[Tensor, Tensor]:
        return half_scale_weight(self.weight)


class HalfScaleTernaryPointwiseConv1d(quant.TernaryPointwiseConv1d):
    """TernaryPointwiseConv1d at fraction 1 whose forward weight is codes * fp32(fp16(s)) as an [out, in] matrix."""

    def forward_weight(self) -> Tensor:
        _require_full(self)
        return _half_scale_ste(self.matrix())

    def forward(self, x: Tensor) -> Tensor:
        return F.conv1d(x, self.forward_weight().unsqueeze(2), self.bias)

    def quantized_weight(self) -> tuple[Tensor, Tensor]:
        return half_scale_weight(self.matrix())


HALF_SCALE_TYPES = (HalfScaleTernaryLinear, HalfScaleTernaryPointwiseConv1d)


def half_scale_ternary(model: nn.Module) -> int:
    """Swap every ternary module for its FP16-scale class (same Parameters, fraction 1); returns the count."""
    names = require_keat(model)
    for name in names:
        module = model.get_submodule(name)
        if isinstance(module, HALF_SCALE_TYPES):
            raise ValueError(f"{name} already has FP16 scales")
        cls = (HalfScaleTernaryPointwiseConv1d if isinstance(module, quant.TernaryPointwiseConv1d)
               else HalfScaleTernaryLinear)
        model.set_submodule(name, cls(module.weight, module.bias))
    return len(names)


def ternarize(model: nn.Module) -> list[str]:
    """Swap every module matching quant.QUANTIZED_PATTERN for its ternary module at fraction 1, keeping the
    Parameters; returns the sorted names."""
    if quant.quantized_module_names(model):
        raise ValueError("model already has ternary modules")
    targets = [n for n, _ in model.named_modules() if quant.QUANTIZED_PATTERN.fullmatch(n)]
    if not targets:
        raise ValueError("no module matches the quantized pattern")
    for name in targets:
        module = model.get_submodule(name)
        ternary = (quant.TernaryPointwiseConv1d.from_conv(module) if isinstance(module, nn.Conv1d)
                   else quant.TernaryLinear.from_linear(module))
        model.set_submodule(name, ternary)
    return quant.quantized_module_names(model)


# --- training forms ----------------------------------------------------------------------------------------

def require_keat(model: nn.Module) -> list[str]:
    """The ternary module names; ValueError unless there are some and all are at weight fraction 1."""
    names = quant.quantized_module_names(model)
    if not names:
        raise ValueError("a Kea-T scope needs a model with ternary modules")
    low = [n for n in names if model.get_submodule(n).fraction != 1.0]
    if low:
        raise ValueError(f"{len(low)} ternary modules are below weight fraction 1 (first {low[0]})")
    return names


def _check_scope(scope: str) -> tuple:
    if scope not in KEAT_SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
    return KEAT_SCOPES[scope]


def apply_keat_scope(model: nn.Module, scope: str) -> dict:
    """The INT8 training form of a Kea-T model for `scope` (module docstring); returns keat_form(model)."""
    parts = _check_scope(scope)
    names = require_keat(model)
    eb.check_parts(model)
    if eb.frozen_keys(model) or isinstance(model.encoder.pre_encode.out, eb.W8Linear) or any(
            isinstance(model.get_submodule(n), HALF_SCALE_TYPES) for n in names):
        raise ValueError("the model is already in a training form")
    eb.quantize_frozen(model, [p for p in parts if p in eb.FROZEN_PARTS])
    if "pre_encode_out" in parts:
        eb.wrap_pre_encode_out(model)
        half_scale_ternary(model)
    return keat_form(model)


@torch.no_grad()
def keat_form(model: nn.Module) -> dict:
    """The INT8 training form of a Kea-T model, read from its modules and checked against its state: ValueError if it
    is no KEAT_SCOPES form (a binary or palettized model, mixed scales, a frozen weight that is not fp32(W8) of its
    load-time codes)."""
    names = require_keat(model)
    eb.check_parts(model)
    if eb.frozen_palette(model):
        raise ValueError("the model has palettized parts; only INT8 forms are bundled")
    record = eb.frozen_int8(model)
    half = {isinstance(model.get_submodule(n), HALF_SCALE_TYPES) for n in names}
    if len(half) != 1:
        raise ValueError("some ternary modules have FP16 scales and some do not")
    w8_pre = isinstance(model.encoder.pre_encode.out, eb.W8Linear)
    frozen = eb.frozen_keys(model)
    for scope, parts in KEAT_SCOPES.items():
        want = sorted(eb.keys_of([p for p in parts if p in eb.FROZEN_PARTS]))
        if frozen == want and w8_pre == ("pre_encode_out" in parts) and half == {scope == "S2"}:
            break
    else:
        raise ValueError(f"not a Kea-T training form: frozen {frozen}, W8 pre_encode.out {w8_pre}, FP16 scales {half}")
    for key, (q, s) in record.items():
        p = model.get_parameter(key)
        if not torch.equal(p.detach(), eb.w8(q.to(p.device), s.to(p.device)).to(torch.float32)):
            raise ValueError(f"{key}: the weight is not fp32(W8) of its load-time codes")
    return {"scope": scope, "frozen": frozen, "pre_encode_out": "w8_ste" if w8_pre else "fp",
            "ternary_scales": "fp16" if scope == "S2" else "fp32"}
