"""Packed 1-bit export of a binary QAT Parakeet-TDT model and its FP32 NeMo reconstruction. See plans/pilot.md "Export".

Format "parakeet-binary-v1", one directory, laid out like parakeet-ternary/export.py's
"parakeet-ternary-v1" except for the codes:
- export.safetensors: for each of the 264 quantized modules "<name>.codes" (uint8, one bit per
  weight packed eight per byte by pack_codes, [out, ceil(in/8)]: code j of a row is bit j % 8
  of byte j // 8, least significant bit first, bit 1 = +1, bit 0 = -1, unused tail bits 0;
  pointwise convolutions as their [out, in] matrix) and "<name>.scale" (FP32 [out]), plus
  "<name>.bias" in FP32 if the module has one (none in this model). Every other state_dict
  tensor as in the ternary export: FP16, except integer buffers (their dtype) and the
  feature-extraction constants export.FP32_BUFFERS (FP32).
- manifest.json: format, SHA-256 of every file, the NeMo model config, per-layer shapes,
  code histogram (overall, per family, per module), parameter accounting, byte breakdown,
  the quantizer's source hashes and the caller's `extra`.
- tokenizer/: extracted from the pinned .nemo by the ternary export's _extract_tokenizer.

load_export() rebuilds a plain FP32 eval-mode NeMo EncDecRNNTBPEModel with dequantized weights
(codes * scale) from the stored config and tokenizer files alone. reconstruction_check() is the
ternary export's check, run against binary_quant: it recovers codes as sign(W), exact here
because every dequantized weight is +-scale with scale >= 1e-8, and scales as the row max |W|.
It runs dense FP32 matrices; nothing here claims a speed or energy benefit.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

import binary_paths
import binary_quant
import export as ternary_export  # finetune/parakeet-ternary/export.py, unchanged
import paths

FORMAT = "parakeet-binary-v1"
FILE = ternary_export.FILE
MANIFEST = ternary_export.MANIFEST
TOKENIZER_DIR = ternary_export.TOKENIZER_DIR
TOKENIZER_FILES = ternary_export.TOKENIZER_FILES
FP32_BUFFERS = ternary_export.FP32_BUFFERS
BIT_LAYOUT = ("uint8 [out, ceil(in/8)]; code j of a row is bit j % 8 of byte j // 8 (least significant bit "
              "first); bit 1 = +1, bit 0 = -1; unused tail bits 0")
_SHIFTS = torch.arange(8, dtype=torch.uint8)
sha256_file = ternary_export.sha256_file


def pack_codes(codes: Tensor) -> Tensor:
    """int8 [out, in] in {-1, +1} -> uint8 [out, ceil(in/8)]; +1 -> 1, -1 -> 0, code j at bit j % 8 of byte j // 8."""
    if codes.dtype != torch.int8 or codes.dim() != 2 or ((codes != 1) & (codes != -1)).any():
        raise ValueError("expected 2-D int8 codes in {-1, +1}")
    rows, n = codes.shape
    bits = torch.zeros(rows, 8 * math.ceil(n / 8), dtype=torch.uint8, device=codes.device)
    bits[:, :n] = (codes == 1).to(torch.uint8)
    bits = bits.view(rows, -1, 8) << _SHIFTS.to(codes.device)
    packed = bits[..., 0]
    for k in range(1, 8):
        packed = packed | bits[..., k]
    return packed


def unpack_codes(packed: Tensor, in_features: int) -> Tensor:
    """Inverse of pack_codes; rejects a wrong dtype or shape and nonzero tail padding."""
    if packed.dtype != torch.uint8 or packed.dim() != 2 or packed.shape[1] != math.ceil(in_features / 8):
        raise ValueError(f"expected uint8 [out, {math.ceil(in_features / 8)}], got {packed.dtype} {tuple(packed.shape)}")
    bits = ((packed[..., None] >> _SHIFTS.to(packed.device)) & 1).flatten(1)
    if bits[:, in_features:].any():
        raise ValueError("invalid packed codes: nonzero tail padding")
    return (2 * bits[:, :in_features].to(torch.int8) - 1).to(torch.int8)


def _artifact_dir(out_dir: Path) -> Path:
    binary_paths.require_mount()
    out_dir = Path(out_dir).resolve()
    if not out_dir.is_relative_to(binary_paths.ARTIFACTS.resolve()):
        raise ValueError(f"binary exports must live under {binary_paths.ARTIFACTS}, got {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _require_full_quantization(model: nn.Module, action: str) -> list[str]:
    """The binary module names; ValueError if there are none or any has weight fraction < 1."""
    names = binary_quant.quantized_module_names(model)
    if not names:
        raise ValueError(f"cannot {action}: model has no binary modules")
    partial = {n: model.get_submodule(n).fraction for n in names if model.get_submodule(n).fraction < 1.0}
    if partial:
        raise ValueError(
            f"cannot {action}: {len(partial)} binary module(s) have weight fraction < 1 (lowest "
            f"{min(partial.values())}, first {min(partial)}). A fraction below 1 is a training-time "
            "quantization ramp, not a deployable model; finish the ramp (binary_quant.set_weight_fraction(model, 1.0)).")
    return names


@torch.no_grad()
def export_model(model: nn.Module, out_dir: Path, extra: dict) -> dict:
    """Write export.safetensors, tokenizer/ and manifest.json under out_dir; return the manifest.

    Raises ValueError before writing anything if the model has no binary modules, any has
    weight fraction < 1, or out_dir is outside the binary artifacts. Codes and scales are
    computed on the model's current device; run reconstruction_check with the QAT model on
    that same device (FP32 row means can differ by one ulp between CPU and GPU).
    """
    from omegaconf import OmegaConf

    names = _require_full_quantization(model, "export")
    out_dir = _artifact_dir(out_dir)
    tensors: dict[str, Tensor] = {}
    sizes = dict.fromkeys(("packed_code_bytes", "scale_bytes", "bias_bytes", "fp16_float_bytes",
                           "fp32_feature_constant_bytes", "integer_buffer_bytes"), 0)
    layers: dict[str, dict] = {}
    covered: set[str] = set()
    for name in names:
        module = model.get_submodule(name)
        codes, scale = module.quantized_weight()
        bias = module.bias
        layers[name] = {"kind": "pointwise_conv1d" if isinstance(module, binary_quant.BinaryPointwiseConv1d)
                        else "linear", "shape": list(codes.shape), "bias": bias is not None}
        covered.update((f"{name}.weight", f"{name}.bias"))
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
        value = value.detach().cpu()
        if not value.is_floating_point():
            tensors[key] = value.clone()
            sizes["integer_buffer_bytes"] += value.nbytes
        elif key in FP32_BUFFERS:
            tensors[key] = value.float().clone()
            sizes["fp32_feature_constant_bytes"] += value.nbytes
        else:
            half = value.to(torch.float16)
            if not torch.isfinite(half).all():
                raise ValueError(f"{key} is not finite in FP16")
            tensors[key] = half
            sizes["fp16_float_bytes"] += half.nbytes
    files = ternary_export._extract_tokenizer(model, out_dir / TOKENIZER_DIR)
    path = out_dir / FILE
    tmp = out_dir / f"{FILE}.tmp{os.getpid()}"
    save_file(tensors, tmp, metadata={"format": FORMAT})
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)
    files = {FILE: sha256_file(path), **files}
    accounting = binary_quant.parameter_accounting(model)
    file_bytes = path.stat().st_size
    sizes.update(header_bytes=file_bytes - sum(sizes.values()), file_bytes=file_bytes,
                 tokenizer_bytes=sum((out_dir / f).stat().st_size for f in files if f != FILE),
                 original_fp32_bytes=4 * accounting["total_parameters"],
                 original_nemo_bytes=paths.MODEL_FILE.stat().st_size)
    import nemo
    manifest = {
        "format": FORMAT,
        "file": FILE,
        "sha256": files[FILE],
        "files": files,
        "base_model": {"id": paths.MODEL_ID, "revision": paths.MODEL_REVISION, "license": "CC-BY-4.0"},
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "quantization_device": next(model.parameters()).device.type,
        "quantizer": "binary: per-row scale max(mean|W|, 1e-8), code +1 for W >= 0 else -1",
        "bit_layout": BIT_LAYOUT,
        "quantizer_source_sha256": binary_paths.quantizer_sha256("binary"),
        "versions": {"nemo": nemo.__version__, "torch": torch.__version__},
        "config": OmegaConf.to_container(model.cfg, resolve=True),
        "quantized_layers": layers,
        "fp32_buffers": list(FP32_BUFFERS),
        "code_histogram": binary_quant.code_histogram(model),
        "parameter_accounting": accounting,
        "bytes": sizes,
        "extra": extra,
    }
    text = json.dumps(manifest, indent=2) + "\n"
    tmp = out_dir / f"{MANIFEST}.tmp{os.getpid()}"
    tmp.write_text(text)
    os.replace(tmp, out_dir / MANIFEST)
    return json.loads(text)


def load_export(out_dir: Path, device: str | torch.device = "cpu"):
    """Rebuild an FP32 eval-mode NeMo EncDecRNNTBPEModel with dequantized weights from an export directory."""
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    from omegaconf import OmegaConf, open_dict

    out_dir = Path(out_dir).resolve()
    manifest = json.loads((out_dir / MANIFEST).read_text())
    if manifest["format"] != FORMAT:
        raise ValueError(f"unknown export format {manifest['format']!r}")
    for rel, digest in manifest["files"].items():
        if sha256_file(out_dir / rel) != digest:
            raise ValueError(f"{out_dir / rel} does not match the manifest SHA-256")
    tensors = load_file(out_dir / manifest["file"])
    state: dict[str, Tensor] = {}
    for name, layer in manifest["quantized_layers"].items():
        out_features, in_features = layer["shape"]
        weight = binary_quant.dequantize(unpack_codes(tensors.pop(f"{name}.codes"), in_features),
                                         tensors.pop(f"{name}.scale"))
        if weight.shape != (out_features, in_features):
            raise ValueError(f"{name}: shape {tuple(weight.shape)} != {layer['shape']}")
        state[f"{name}.weight"] = weight.unsqueeze(2) if layer["kind"] == "pointwise_conv1d" else weight
        if layer["bias"]:
            state[f"{name}.bias"] = tensors.pop(f"{name}.bias")
    state.update((k, v.float() if v.is_floating_point() else v) for k, v in tensors.items())
    cfg = OmegaConf.create(manifest["config"])
    if cfg.get("target") != "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel":
        raise ValueError(f"unexpected model target {cfg.get('target')!r}")
    with open_dict(cfg):
        cfg.tokenizer.dir = str(out_dir / TOKENIZER_DIR)
        for key, filename in TOKENIZER_FILES.items():
            cfg.tokenizer[key] = str(out_dir / TOKENIZER_DIR / filename)
        for ds in ("train_ds", "validation_ds", "test_ds"):  # stored for reference; no dataloaders at load
            if ds in cfg:
                cfg[ds] = None
    model = EncDecRNNTBPEModel(cfg=cfg)
    model.load_state_dict(state, strict=True)
    return model.to(device=device, dtype=torch.float32).eval()


@contextlib.contextmanager
def _ternary_check_reads_binary_modules():
    """export.reconstruction_check finds the quantized modules through its module global `quant`."""
    saved = ternary_export.quant
    ternary_export.quant = binary_quant
    try:
        yield
    finally:
        ternary_export.quant = saved


def reconstruction_check(qat_model: nn.Module, rebuilt: nn.Module, batch: tuple) -> dict:
    """parakeet-ternary export.reconstruction_check against the binary modules: exact codes and
    scales, FP32 encoder, joint (logit) and greedy agreement on one fixed audio batch. Both
    models in eval mode on one device; the QAT model at fraction 1."""
    with _ternary_check_reads_binary_modules():
        result = ternary_export.reconstruction_check(qat_model, rebuilt, batch)
    result["format"] = FORMAT
    return result
