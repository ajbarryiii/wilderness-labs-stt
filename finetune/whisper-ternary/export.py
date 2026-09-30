"""Packed ternary export of a QAT Whisper model and its FP32 reconstruction. See DESIGN.md "Export".

The file holds what a deployment reads: 2-bit packed codes, FP32 row scales and
FP32 biases for every quantized layer, and FP16 for every other tensor. The
tied embedding / output projection is stored once. load_export() rebuilds a
plain Hugging Face model with dequantized FP32 weights; it does not run packed
kernels and makes no speed or energy claim.

Only a fully quantized model (weight fraction 1 on every ternary module, see
quant.py and DESIGN.md "Revision 2") is deployable: export_model and
reconstruction_check refuse a model that is still inside the progressive
quantization ramp (fraction < 1), which is a training-time state only.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor
from transformers import GenerationConfig, WhisperConfig, WhisperForConditionalGeneration

import paths
from quant import (EMBEDDING, OUTPUT, TernaryEmbedding, code_histogram, dequantize,
                   parameter_accounting, quantized_module_names)

FORMAT = "whisper-ternary-v1"
FILE = "export.safetensors"
MANIFEST = "manifest.json"
_SHIFTS = torch.tensor([0, 2, 4, 6], dtype=torch.uint8)


def pack_codes(codes: Tensor) -> Tensor:
    """int8 [out, in] in {-1,0,1} -> uint8 [out, ceil(in/4)]; 0->00, +1->01, -1->10, code j at bits 2*(j%4)."""
    if codes.dtype != torch.int8 or codes.dim() != 2 or ((codes < -1) | (codes > 1)).any():
        raise ValueError("expected 2-D int8 codes in {-1, 0, 1}")
    rows, n = codes.shape
    fields = torch.zeros(rows, 4 * math.ceil(n / 4), dtype=torch.uint8, device=codes.device)
    fields[:, :n] = (codes == 1).to(torch.uint8) | ((codes == -1).to(torch.uint8) << 1)
    fields = fields.view(rows, -1, 4) << _SHIFTS.to(codes.device)
    return fields[..., 0] | fields[..., 1] | fields[..., 2] | fields[..., 3]


def unpack_codes(packed: Tensor, in_features: int) -> Tensor:
    """Inverse of pack_codes; rejects the unused 11 pattern and nonzero tail padding."""
    if packed.dtype != torch.uint8 or packed.dim() != 2 or packed.shape[1] != math.ceil(in_features / 4):
        raise ValueError(f"expected uint8 [out, {math.ceil(in_features / 4)}], got {packed.dtype} {tuple(packed.shape)}")
    fields = ((packed[..., None] >> _SHIFTS.to(packed.device)) & 3).flatten(1)
    if (fields == 3).any() or fields[:, in_features:].any():
        raise ValueError("invalid packed codes")
    fields = fields[:, :in_features]
    return (fields == 1).to(torch.int8) - (fields == 2).to(torch.int8)


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _artifact_dir(out_dir: Path) -> Path:
    paths.require_mount()
    out_dir = Path(out_dir).resolve()
    if not out_dir.is_relative_to(paths.ARTIFACTS.resolve()):
        raise ValueError(f"exports must live under {paths.ARTIFACTS}, got {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _require_full_quantization(model: WhisperForConditionalGeneration, action: str) -> None:
    """ValueError if any ternary module is still inside the progressive quantization ramp (fraction < 1)."""
    partial = {name: model.get_submodule(name).fraction for name in quantized_module_names(model)}
    partial = {name: fraction for name, fraction in partial.items() if fraction < 1.0}
    if partial:
        raise ValueError(
            f"cannot {action}: {len(partial)} ternary module(s) have weight fraction < 1 (lowest "
            f"{min(partial.values())}, first {min(partial)}). A fraction below 1 is a training-time "
            "quantization ramp, not a deployable model; finish the ramp first (train on to fraction 1.0, "
            "set by quant.set_weight_fraction(model, 1.0)).")


def export_model(model: WhisperForConditionalGeneration, out_dir: Path, extra: dict) -> dict:
    """Write out_dir/export.safetensors and out_dir/manifest.json; return the manifest.

    Raises ValueError, before writing anything, if any ternary module has
    weight fraction < 1: the export stores the fully quantized codes, so a
    model still inside the ramp would be exported as something it never ran as.

    Codes and scales are computed on the model's current device; run
    reconstruction_check with the QAT model on that same device (FP32 row means
    can differ by one ulp between CPU and GPU).
    """
    names = quantized_module_names(model)
    if not names:
        raise ValueError("model has no ternary modules")
    _require_full_quantization(model, "export")
    if model.proj_out.weight is not model.get_submodule(EMBEDDING).weight:
        raise ValueError("proj_out must be tied to the decoder embedding")
    if (OUTPUT in names) != (EMBEDDING in names):
        raise ValueError("tied embedding and output projection must be quantized together")
    out_dir = _artifact_dir(out_dir)
    tensors: dict[str, Tensor] = {}
    sizes = dict.fromkeys(("packed_code_bytes", "scale_bytes", "bias_bytes", "fp16_residual_bytes"), 0)
    layers: dict[str, dict] = {}
    covered = {f"{OUTPUT}.weight"}  # the tie: stored once, under the embedding
    for name in names:
        module = model.get_submodule(name)
        bias = getattr(module, "bias", None)
        layers[name] = {"kind": "embedding" if isinstance(module, TernaryEmbedding) else "linear",
                        "shape": list(module.weight.shape), "bias": bias is not None}
        covered.update((f"{name}.weight", f"{name}.bias"))
        if name == OUTPUT:
            continue
        codes, scale = module.quantized_weight()
        tensors[f"{name}.codes"] = pack_codes(codes).cpu()
        tensors[f"{name}.scale"] = scale.cpu()
        sizes["packed_code_bytes"] += tensors[f"{name}.codes"].nbytes
        sizes["scale_bytes"] += tensors[f"{name}.scale"].nbytes
        if bias is not None:
            tensors[f"{name}.bias"] = bias.detach().float().cpu()
            sizes["bias_bytes"] += tensors[f"{name}.bias"].nbytes
    for key, value in model.state_dict().items():
        if key in covered:
            continue
        half = value.detach().to(device="cpu", dtype=torch.float16)
        if not value.is_floating_point() or not torch.isfinite(half).all():
            raise ValueError(f"{key} is not a finite floating-point tensor in FP16")
        tensors[key] = half
        sizes["fp16_residual_bytes"] += half.nbytes
    path = out_dir / FILE
    save_file(tensors, path, metadata={"format": FORMAT})
    accounting = parameter_accounting(model)
    file_bytes = path.stat().st_size
    sizes.update(header_bytes=file_bytes - sum(sizes.values()), file_bytes=file_bytes,
                 original_fp32_bytes=4 * accounting["total_parameters"],
                 original_fp16_bytes=2 * accounting["total_parameters"])
    manifest = {
        "format": FORMAT,
        "file": FILE,
        "sha256": _sha256(path),
        "quantization_device": model.proj_out.weight.device.type,
        "config": model.config.to_dict(),
        "generation_config": model.generation_config.to_dict(),
        "quantized_layers": layers,
        "tied": {OUTPUT: EMBEDDING},
        "code_histogram": code_histogram(model),
        "parameter_accounting": accounting,
        "bytes": sizes,
        "extra": extra,
    }
    text = json.dumps(manifest, indent=2) + "\n"
    (out_dir / MANIFEST).write_text(text)
    return json.loads(text)  # exactly what is on disk (JSON turns int dict keys into strings)


def load_export(out_dir: Path, device: str | torch.device = "cpu") -> WhisperForConditionalGeneration:
    """Rebuild a plain FP32 Hugging Face model (dequantized weights, restored tie) from an export."""
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / MANIFEST).read_text())
    if manifest["format"] != FORMAT:
        raise ValueError(f"unknown export format {manifest['format']!r}")
    path = out_dir / manifest["file"]
    if _sha256(path) != manifest["sha256"]:
        raise ValueError(f"{path} does not match the manifest SHA-256")
    tensors = load_file(path)
    state: dict[str, Tensor] = {}
    for name, layer in manifest["quantized_layers"].items():
        if name in manifest["tied"]:
            continue
        codes = unpack_codes(tensors.pop(f"{name}.codes"), layer["shape"][1])
        state[f"{name}.weight"] = dequantize(codes, tensors.pop(f"{name}.scale"))
        if layer["bias"]:
            state[f"{name}.bias"] = tensors.pop(f"{name}.bias")
    state.update((key, value.float()) for key, value in tensors.items())
    model = WhisperForConditionalGeneration(WhisperConfig(**manifest["config"]))
    for name, target in manifest["tied"].items():
        model.get_submodule(name).weight = model.get_submodule(target).weight
        state[f"{name}.weight"] = state[f"{target}.weight"]
    model.load_state_dict(state, strict=True)
    model.generation_config = GenerationConfig(**manifest["generation_config"])
    return model.to(device=device, dtype=torch.float32).eval()


def reconstruction_check(qat_model: WhisperForConditionalGeneration, rebuilt_model: WhisperForConditionalGeneration,
                         input_features: Tensor, decoder_input_ids: Tensor) -> dict:
    """Exact code/scale agreement plus FP32 logit agreement on one fixed batch.

    Re-running absmean quantization on a dequantized row does not return its
    scale whenever the row has zero codes (the absmean shrinks by nnz/in), so
    codes are recovered from the rebuilt weight as sign(W) and scales as the
    row max |W|. A row whose codes are all zero is the zero vector for any
    scale, so its scale is not observable there and is not compared. The
    training forward uses exactly W_hat, so logits may differ only through
    FP16 storage of the residual tensors.

    The QAT model must be at weight fraction 1 on every ternary module
    (ValueError otherwise); below 1 its forward is not the exported W_hat.
    """
    if qat_model.training or rebuilt_model.training:
        raise ValueError("both models must be in eval mode")
    _require_full_quantization(qat_model, "run the reconstruction check")
    device = qat_model.proj_out.weight.device
    if rebuilt_model.proj_out.weight.device != device:
        raise ValueError("both models must be on the same device")
    names = quantized_module_names(qat_model)
    codes_exact = scales_exact = True
    with torch.no_grad(), torch.autocast(device.type, enabled=False):
        for name in names:
            codes, scale = qat_model.get_submodule(name).quantized_weight()
            weight = rebuilt_model.get_submodule(name).weight
            observable = codes.ne(0).any(dim=1)
            codes_exact &= torch.equal(torch.sign(weight).to(torch.int8), codes)
            scales_exact &= torch.equal(weight.abs().amax(dim=1)[observable], scale[observable])
        inputs = {"input_features": input_features.to(device, torch.float32),
                  "decoder_input_ids": decoder_input_ids.to(device), "use_cache": False}
        qat_logits = qat_model(**inputs).logits.float()
        rebuilt_logits = rebuilt_model(**inputs).logits.float()
    return {
        "codes_exact": bool(codes_exact),
        "scales_exact": bool(scales_exact),
        "logits_max_abs_diff": float((qat_logits - rebuilt_logits).abs().max()),
        "argmax_agreement": float((qat_logits.argmax(-1) == rebuilt_logits.argmax(-1)).float().mean()),
        "checked_layers": len(names),
    }
