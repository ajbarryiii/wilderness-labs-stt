"""The 8-bit floating-point parts of Kea-B (keab-prime plan "Step 0: 8-bit check of the floating-point parts" and
"Use"): the INT8 format, the scopes, the training forms that stage 4 trains and exports, and the informational
palettization and W8A8 emulation that step 0 only measures.

INT8, the one trained and exported format (Core ML linear quantization, symmetric per output channel):
  - a channel is a row (axis 0) of a 2-D weight: the output channel of a linear or LSTM weight, the token of the
    embedding;
  - s_c = max(max|W_c| / 127, 1e-30) and q = round-half-even(W / s_c) clamped to [-127, 127], in FP32 from the
    FP32 weight, with IEEE divisions (the same codes and scales on the CPU and on CUDA; an all-zero channel stores
    zeros with s_c = 1e-30);
  - the weight the model computes with is W8 = fp16(q * s_c): the FP32 product rounded once to FP16. The scoring
    view stores exactly these FP16 values (training/keab_export.py), so training, scoring and the bundle agree.
  As in Core ML, only the weight matrices are quantized; the parts' biases stay FP16 like every other float tensor.

Parts (PARTS, the real NeMo module names; check_parts refuses a model with any other weight matrix there):
  prediction_network  the embedding and the 2-layer LSTM's input and hidden weights
  joint               joint.enc, joint.pred and joint.joint_net.2
  pre_encode_out      encoder.pre_encode.out (Kea-B's; B0' scopes never include it)
Scopes: KEAB_SCOPES (S1 = prediction network; S2 = S1 + joint + pre_encode.out, with the encoder's binary row scales
stored as FP16) and DECODER_ONLY_SCOPES for B0' (S2 = S1 + joint, no binary scales).

Training forms ("Use"):
  - frozen parts (prediction network, joint): the weights are replaced by fp32(W8) once, at load
    (quantize_frozen), in place (the Parameter objects stay, so the LSTM's flat weights follow); the codes and
    scales are kept on the model (frozen_int8) and a second quantization is refused;
  - pre_encode.out, if in scope, computes with W8 of its latent weight with a straight-through gradient (W8Linear);
  - under S2 each binary module's row scale alpha is computed as in training and rounded through FP16,
    fp32(fp16(alpha)) (HalfScaleBinaryLinear / HalfScaleBinaryPointwiseConv1d), so the stored FP16 scale widens
    exactly to what training used.
  apply_keab_scope builds the form of a Kea-B model (binary modules at fraction 1); keab_form reads it back and
  checks the state against it. apply_decoder_scope quantizes B0''s decoder parts (INT8 or palettized).

Informational only (never trained or exported as a bundle):
  - palettization: one k-means table of 256 entries per tensor (Core ML palettization: k-means++ seeding and Lloyd
    iterations, which coremltools runs through scikit-learn with a fixed random state), here k-means++ seeding from
    a fixed torch generator, then 1-D Lloyd iterations on the sorted values (FP64 sums) until no centroid changes (at
    most 1000); a tensor with at most 256 distinct values keeps them. The entries are stored in FP16 and each weight
    becomes its entry (palettize, palettize_frozen, apply_keab_palette);
  - x1_fake_quant: the x1 per-frame activation quantizer emulated in PyTorch for the pre_encode.out W8A8
    measurement (per frame across the last axis, s = max(max|x| / 127, 1e-30), q = round-half-even(x / s), x̂ = q s,
    computed in FP32 and returned in the input's dtype).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_BINARY_DIR = str(Path(__file__).resolve().parents[2] / "parakeet-binary")
if _BINARY_DIR not in sys.path:
    sys.path.append(_BINARY_DIR)   # after parakeet-ternary, as score_domain.binary_on_path
import binary_quant  # noqa: E402  finetune/parakeet-binary/binary_quant.py, unchanged

FLOOR = 1e-30
QMAX = 127
PALETTE_ENTRIES = 256
PALETTE_MAX_ITERATIONS = 1000
PALETTE_SEED = 0
PARTS = {
    "prediction_network": ("decoder.prediction.embed.weight",
                           "decoder.prediction.dec_rnn.lstm.weight_ih_l0", "decoder.prediction.dec_rnn.lstm.weight_hh_l0",
                           "decoder.prediction.dec_rnn.lstm.weight_ih_l1", "decoder.prediction.dec_rnn.lstm.weight_hh_l1"),
    "joint": ("joint.enc.weight", "joint.pred.weight", "joint.joint_net.2.weight"),
    "pre_encode_out": ("encoder.pre_encode.out.weight",),
}
PART_PREFIX = {"prediction_network": "decoder.", "joint": "joint.", "pre_encode_out": "encoder.pre_encode.out."}
FROZEN_PARTS = ("prediction_network", "joint")
SCOPES = ("none", "S1", "S2")
KEAB_SCOPES = {"none": (), "S1": ("prediction_network",), "S2": ("prediction_network", "joint", "pre_encode_out")}
DECODER_ONLY_SCOPES = {"none": (), "S1": ("prediction_network",), "S2": ("prediction_network", "joint")}
FORMATS = ("int8", "palette")
_FROZEN = "_keab_frozen"      # model attribute: {key: {"format", "q", "s" | "lut", "index"}}


# --- the INT8 format ------------------------------------------------------------------------------

def int8_quantize(weight: Tensor) -> tuple[Tensor, Tensor]:
    """(q int8 [rows, cols], s FP32 [rows]) of a 2-D weight, per row (module docstring); not differentiable."""
    if weight.dim() != 2:
        raise ValueError(f"INT8 per-channel quantization needs a 2-D weight, got shape {tuple(weight.shape)}")
    w = weight.detach().to(torch.float32)
    if not torch.isfinite(w).all():
        raise ValueError("cannot quantize a weight with non-finite values")
    s = _div_qmax(w.abs().amax(dim=1)).clamp_min(FLOOR)
    q = torch.round(w / s[:, None]).clamp_(-QMAX, QMAX).to(torch.int8)
    return q, s


def _div_qmax(m: Tensor) -> Tensor:
    """m / 127 by IEEE division on every device: CUDA turns division by a Python scalar into multiplication by
    its rounded reciprocal (a different FP32 result for some m), so the divisor is a tensor of m's shape."""
    return m / torch.full_like(m, float(QMAX))


def w8(q: Tensor, s: Tensor) -> Tensor:
    """W8 = fp16(q * s_c): the FP32 product rounded to FP16."""
    if q.dtype != torch.int8 or s.dtype != torch.float32 or q.dim() != 2 or s.shape != (q.shape[0],):
        raise ValueError("expected int8 codes [rows, cols] and FP32 scales [rows]")
    return (q.to(torch.float32) * s[:, None]).to(torch.float16)


def w8_of(weight: Tensor) -> Tensor:
    return w8(*int8_quantize(weight))


def widen(half: Tensor) -> Tensor:
    """FP16 -> FP32 (exact for every finite FP16 value)."""
    if half.dtype != torch.float16:
        raise ValueError(f"expected FP16, got {half.dtype}")
    return half.to(torch.float32)


def int8_ste(weight: Tensor) -> Tensor:
    """Value bit-identical to fp32(W8); identity gradient to the latent weight."""
    return w8_of(weight).to(weight.dtype) + (weight - weight.detach())


# --- palettization (informational) ----------------------------------------------------------------

def palette_table(weight: Tensor, entries: int = PALETTE_ENTRIES,
                  max_iterations: int = PALETTE_MAX_ITERATIONS) -> tuple[Tensor, Tensor, dict]:
    """(FP16 table [k], index uint8 shaped like the weight, info) of one tensor (module docstring)."""
    x = weight.detach().to(torch.float64).flatten().cpu()
    if not torch.isfinite(x).all():
        raise ValueError("cannot palettize a weight with non-finite values")
    xs, _ = torch.sort(x)
    n = len(xs)
    distinct = torch.unique(xs)
    csum = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(xs, 0)])
    converged, iterations = len(distinct) <= entries, 0
    if converged:                                             # few values: the table is the values themselves
        c = distinct
    else:                                                     # k-means++ seeding, a fixed generator
        g = torch.Generator().manual_seed(PALETTE_SEED)
        seeds = [float(xs[torch.randint(n, (1,), generator=g)])]
        d2 = (xs - seeds[0]) ** 2
        for _ in range(entries - 1):
            seeds.append(float(xs[torch.multinomial((d2 / d2.sum()).to(torch.float32), 1, generator=g)]))
            d2 = torch.minimum(d2, (xs - seeds[-1]) ** 2)
        c = torch.unique(torch.tensor(seeds, dtype=torch.float64))   # sorted
    for iterations in range(1, 0 if converged else max_iterations + 1):
        bounds = (c[1:] + c[:-1]) / 2
        ends = torch.cat([torch.searchsorted(xs, bounds, right=True), torch.tensor([n])])
        starts = torch.cat([torch.tensor([0]), ends[:-1]])
        counts = ends - starts
        new = torch.where(counts > 0, (csum[ends] - csum[starts]) / counts.clamp_min(1), c)
        if torch.equal(new, c):
            converged = True
            break
        c = new
    bounds = (c[1:] + c[:-1]) / 2
    index = torch.searchsorted(bounds, x, right=False)       # x on a boundary goes to the lower entry
    lut = c.to(torch.float16)
    info = {"entries": len(c), "iterations": iterations, "converged": converged, "centroids": c,
            "max_abs_error": float((lut[index].to(torch.float64) - x).abs().max()) if n else 0.0}
    return lut, index.to(torch.uint8).reshape(weight.shape), info


def palettize(weight: Tensor, entries: int = PALETTE_ENTRIES) -> tuple[Tensor, dict]:
    """(the palettized weight in FP16, shaped like the weight, info)."""
    lut, index, info = palette_table(weight, entries)
    return lut[index.long()].to(weight.device), {k: v for k, v in info.items() if k != "centroids"}


# --- parts and frozen quantization ----------------------------------------------------------------

def keys_of(parts) -> list[str]:
    for part in parts:
        if part not in PARTS:
            raise ValueError(f"unknown part {part!r}; expected one of {tuple(PARTS)}")
    return [key for part in parts for key in PARTS[part]]


def check_parts(model: nn.Module) -> dict[str, list[str]]:
    """{part: its weight matrices}; ValueError unless every part's float matrices are exactly PARTS[part]."""
    found = {part: [] for part in PARTS}
    for key, p in model.named_parameters():
        for part, prefix in PART_PREFIX.items():
            if key.startswith(prefix) and p.is_floating_point() and p.dim() >= 2:
                found[part].append(key)
    for part, keys in PARTS.items():
        if sorted(found[part]) != sorted(keys):
            raise ValueError(f"{part}: weight matrices {found[part]} are not {list(keys)} (not the Parakeet layout)")
    return {part: list(keys) for part, keys in PARTS.items()}


def _frozen(model: nn.Module) -> dict:
    record = getattr(model, _FROZEN, None)
    if record is None:
        record = {}
        object.__setattr__(model, _FROZEN, record)   # a plain attribute: never in state_dict
    return record


def frozen_int8(model: nn.Module) -> dict[str, tuple[Tensor, Tensor]]:
    """{key: (q, s)} of the parts quantized to INT8 at load."""
    return {k: (v["q"], v["s"]) for k, v in getattr(model, _FROZEN, {}).items() if v["format"] == "int8"}


def frozen_palette(model: nn.Module) -> dict[str, dict]:
    return {k: v for k, v in getattr(model, _FROZEN, {}).items() if v["format"] == "palette"}


def frozen_keys(model: nn.Module) -> list[str]:
    return sorted(getattr(model, _FROZEN, {}))


@torch.no_grad()
def _freeze(model: nn.Module, parts, fmt: str) -> dict:
    check_parts(model)
    keys = keys_of(parts)
    record = _frozen(model)
    again = sorted(set(keys) & set(record))
    if again:
        raise ValueError(f"already quantized once: {again}")
    out = {}
    for key in keys:
        p = model.get_parameter(key)
        if p.dtype != torch.float32:
            raise ValueError(f"{key}: expected an FP32 weight, got {p.dtype}")
        if fmt == "int8":
            q, s = int8_quantize(p)
            p.copy_(w8(q, s).to(torch.float32))
            entry = {"format": "int8", "q": q, "s": s}
        else:
            lut, index, info = palette_table(p)
            p.copy_(lut[index.long()].to(device=p.device, dtype=torch.float32))
            entry = {"format": "palette", "lut": lut, "index": index,
                     "info": {k: v for k, v in info.items() if k != "centroids"}}
        out[key] = entry
    record.update(out)
    return out


def quantize_frozen(model: nn.Module, parts) -> dict[str, tuple[Tensor, Tensor]]:
    """Replace the parts' weights by fp32(W8) once, in place; returns {key: (q, s)}."""
    return {k: (v["q"], v["s"]) for k, v in _freeze(model, parts, "int8").items()}


def palettize_frozen(model: nn.Module, parts) -> dict[str, dict]:
    """Replace the parts' weights by their FP16 palette entries once, in place (informational)."""
    return _freeze(model, parts, "palette")


# --- pre_encode.out with W8 and a straight-through gradient --------------------------------------

class W8Linear(nn.Module):
    """nn.Linear replacement that computes with W8 of its latent FP32 weight (identity gradient to it). Holds the
    caller's Parameter objects, so state_dict keys and optimizer bindings are unchanged."""

    def __init__(self, weight: nn.Parameter, bias: nn.Parameter | None) -> None:
        super().__init__()
        if not isinstance(weight, nn.Parameter) or weight.dim() != 2 or weight.dtype != torch.float32:
            raise ValueError("W8Linear needs a 2-D FP32 nn.Parameter weight")
        self.out_features, self.in_features = weight.shape
        self.weight = weight
        self.register_parameter("bias", bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> W8Linear:
        if type(linear) is not nn.Linear:
            raise ValueError(f"expected nn.Linear, got {type(linear).__name__}")
        return cls(linear.weight, linear.bias)

    def forward_weight(self) -> Tensor:
        return int8_ste(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.forward_weight(), self.bias)

    def int8(self) -> tuple[Tensor, Tensor]:
        return int8_quantize(self.weight)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, w8"


def wrap_pre_encode_out(model: nn.Module) -> W8Linear:
    current = model.encoder.pre_encode.out
    if isinstance(current, W8Linear):
        raise ValueError("pre_encode.out is already W8")
    wrapped = W8Linear.from_linear(current)
    model.encoder.pre_encode.out = wrapped
    return wrapped


# --- binary modules with FP16-rounded row scales (S2) ---------------------------------------------

def half_scale(scale: Tensor) -> Tensor:
    """fp32(fp16(alpha)); ValueError if any alpha is not a positive finite FP16 value."""
    half = scale.to(torch.float16)
    if not (torch.isfinite(half).all() and (half > 0).all()):
        raise ValueError(f"a binary row scale is not a positive finite FP16 value (min alpha {float(scale.min()):.3g})")
    return half.to(torch.float32)


def _half_scale_weight(matrix: Tensor) -> tuple[Tensor, Tensor]:
    codes, scale = binary_quant.binary_quantize(matrix)
    return codes, half_scale(scale)


def _half_scale_ste(matrix: Tensor) -> Tensor:
    return binary_quant.dequantize(*_half_scale_weight(matrix)) + (matrix - matrix.detach())


def _require_full(module) -> None:
    if module.fraction != 1.0:
        raise ValueError("FP16 binary scales are defined at weight fraction 1 only")


class HalfScaleBinaryLinear(binary_quant.BinaryLinear):
    """BinaryLinear at fraction 1 whose row scale is fp32(fp16(alpha))."""

    def forward_weight(self) -> Tensor:
        _require_full(self)
        return _half_scale_ste(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.forward_weight(), self.bias)

    def quantized_weight(self) -> tuple[Tensor, Tensor]:
        return _half_scale_weight(self.weight)


class HalfScaleBinaryPointwiseConv1d(binary_quant.BinaryPointwiseConv1d):
    """BinaryPointwiseConv1d at fraction 1 whose row scale is fp32(fp16(alpha))."""

    def forward_weight(self) -> Tensor:
        _require_full(self)
        return _half_scale_ste(self.matrix())

    def forward(self, x: Tensor) -> Tensor:
        return F.conv1d(x, self.forward_weight().unsqueeze(2), self.bias)

    def quantized_weight(self) -> tuple[Tensor, Tensor]:
        return _half_scale_weight(self.matrix())


HALF_SCALE_TYPES = (HalfScaleBinaryLinear, HalfScaleBinaryPointwiseConv1d)


def half_scale_binary(model: nn.Module) -> int:
    """Swap every binary module for its FP16-scale class (same Parameters, ramp and fraction); returns the count."""
    names = binary_quant.quantized_module_names(model)
    if not names:
        raise ValueError("model has no binary modules")
    for name in names:
        module = model.get_submodule(name)
        if isinstance(module, HALF_SCALE_TYPES):
            raise ValueError(f"{name} already has FP16 scales")
        cls = (HalfScaleBinaryPointwiseConv1d if isinstance(module, binary_quant.BinaryPointwiseConv1d)
               else HalfScaleBinaryLinear)
        swapped = cls(module.weight, module.bias, module.ramp)
        swapped.fraction = module.fraction
        model.set_submodule(name, swapped)
    return len(names)


def binarize(model: nn.Module, ramp: str = "fp32") -> list[str]:
    """Swap every module matching binary_quant.QUANTIZED_PATTERN (264 in Parakeet, fewer in a tiny test model) for
    its binary module at fraction 1, keeping the Parameters; returns the sorted names."""
    if binary_quant.quantized_module_names(model):
        raise ValueError("model already has binary modules")
    targets = [n for n, _ in model.named_modules() if binary_quant.QUANTIZED_PATTERN.fullmatch(n)]
    if not targets:
        raise ValueError("no module matches the quantized pattern")
    for name in targets:
        module = model.get_submodule(name)
        binary = (binary_quant.BinaryPointwiseConv1d.from_conv(module, ramp) if isinstance(module, nn.Conv1d)
                  else binary_quant.BinaryLinear.from_linear(module, ramp))
        model.set_submodule(name, binary)
    return binary_quant.quantized_module_names(model)


# --- training forms --------------------------------------------------------------------------------

def _require_keab(model: nn.Module) -> list[str]:
    names = binary_quant.quantized_module_names(model)
    if not names:
        raise ValueError("a Kea-B scope needs a model with binary modules")
    low = [n for n in names if model.get_submodule(n).fraction != 1.0]
    if low:
        raise ValueError(f"{len(low)} binary modules are below weight fraction 1 (first {low[0]})")
    return names


def _check_scope(scope: str, scopes: dict) -> tuple:
    if scope not in scopes:
        raise ValueError(f"scope must be one of {tuple(scopes)}, got {scope!r}")
    return scopes[scope]


def apply_keab_scope(model: nn.Module, scope: str) -> dict:
    """The INT8 training form of a Kea-B model for `scope` (module docstring); returns keab_form(model)."""
    parts = _check_scope(scope, KEAB_SCOPES)
    _require_keab(model)
    check_parts(model)
    if frozen_keys(model) or isinstance(model.encoder.pre_encode.out, W8Linear) or any(
            isinstance(model.get_submodule(n), HALF_SCALE_TYPES) for n in binary_quant.quantized_module_names(model)):
        raise ValueError("the model is already in a training form")
    quantize_frozen(model, [p for p in parts if p in FROZEN_PARTS])
    if "pre_encode_out" in parts:
        wrap_pre_encode_out(model)
        half_scale_binary(model)
    return keab_form(model)


def apply_keab_palette(model: nn.Module, scope: str) -> dict:
    """The informational palettized form of a Kea-B model: every part in scope palettized once (pre_encode.out
    included, as a frozen replacement), FP16 binary scales under S2."""
    parts = _check_scope(scope, KEAB_SCOPES)
    _require_keab(model)
    if frozen_keys(model):
        raise ValueError("the model is already in a training form")
    palettize_frozen(model, parts)
    if scope == "S2":
        half_scale_binary(model)
    names = binary_quant.quantized_module_names(model)
    return {"scope": scope, "format": "palette", "frozen": frozen_keys(model),
            "binary_scales": "fp16" if isinstance(model.get_submodule(names[0]), HALF_SCALE_TYPES) else "fp32"}


def apply_decoder_scope(model: nn.Module, scope: str, fmt: str) -> dict:
    """B0''s decoder parts for `scope` (DECODER_ONLY_SCOPES) in `fmt` (int8 or palette), replaced once at load."""
    parts = _check_scope(scope, DECODER_ONLY_SCOPES)
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}, got {fmt!r}")
    if parts:
        (quantize_frozen if fmt == "int8" else palettize_frozen)(model, parts)
    return {"scope": scope, "format": fmt, "frozen": frozen_keys(model)}


@torch.no_grad()
def keab_form(model: nn.Module) -> dict:
    """The INT8 training form of a Kea-B model, read from its modules and checked against its state: ValueError if
    it is no KEAB_SCOPES form (a palettized model, mixed scales, a frozen weight that is not fp32(W8) of its codes)."""
    names = _require_keab(model)
    check_parts(model)
    record = getattr(model, _FROZEN, {})
    if any(v["format"] != "int8" for v in record.values()):
        raise ValueError("the model has palettized parts; only INT8 forms are bundled")
    half = {isinstance(model.get_submodule(n), HALF_SCALE_TYPES) for n in names}
    if len(half) != 1:
        raise ValueError("some binary modules have FP16 scales and some do not")
    w8_pre = isinstance(model.encoder.pre_encode.out, W8Linear)
    frozen = sorted(record)
    for scope, parts in KEAB_SCOPES.items():
        want = sorted(keys_of([p for p in parts if p in FROZEN_PARTS]))
        if frozen == want and w8_pre == ("pre_encode_out" in parts) and half == {scope == "S2"}:
            break
    else:
        raise ValueError(f"not a Kea-B training form: frozen {frozen}, W8 pre_encode.out {w8_pre}, FP16 scales {half}")
    for key, entry in record.items():
        p = model.get_parameter(key)
        if not torch.equal(p.detach(), w8(entry["q"].to(p.device), entry["s"].to(p.device)).to(torch.float32)):
            raise ValueError(f"{key}: the weight is not fp32(W8) of its load-time codes")
    return {"scope": scope, "frozen": frozen, "pre_encode_out": "w8_ste" if w8_pre else "fp",
            "binary_scales": "fp16" if scope == "S2" else "fp32"}


# --- the x1 per-frame activation quantizer, emulated (informational W8A8) ------------------------

def x1_fake_quant(x: Tensor) -> Tensor:
    """x̂ = q s per frame across the last axis (module docstring); FP32 arithmetic, output in x's dtype."""
    x32 = x.to(torch.float32)
    s = _div_qmax(x32.abs().amax(dim=-1, keepdim=True)).clamp_min(FLOOR)
    return (torch.round(x32 / s) * s).to(x.dtype)
