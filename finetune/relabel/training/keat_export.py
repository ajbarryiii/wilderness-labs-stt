"""The relabel-owned export adapter of stage 5 (keat-prime plan "Export (the ternary adapter)"): the 8-bit bundle of a
Kea-T training model and the parakeet-ternary-v1 scoring view derived from it. The counterpart of
training/keab_export.py, whose type-agnostic helpers it reuses (the NeMo model built from a stored config).

M1's export.py stores ternary scales in FP32, has no 8-bit representation, and its _artifact_dir refuses any directory
outside $PARAKEET_TERNARY_ROOT. This module writes, per model:

1. The bundle, the artifact of record (format "keat-8bit-v1"; write_bundle), one directory:
   - bundle.safetensors (metadata {"format": "keat-8bit-v1"}):
       <ternary module>.codes      the ternary codes, packed as export.pack_codes (2 bits; uint8 [out, ceil(in/4)])
       <ternary module>.scale      the row scales: FP32, or FP16 under S2 (the training form's fp16(s))
       <ternary module>.bias       FP32, if the module has one (none in Parakeet)
       <weight key>.int8           INT8 codes [rows, cols] of each weight in 8 bits (keat_eightbit.KEAT_SCOPES)
       <weight key>.int8_scale     their FP32 per-channel (row) scales
       every other state_dict tensor exactly as export.export_model stores it: FP16 for float weights, FP32 for the
       feature constants (export.FP32_BUFFERS: the preprocessor's window and filterbank), integer buffers as they
       are;
   - manifest.json: format, scope and training form, SHA-256 of every file, the NeMo config, the ternary layers, the
     8-bit tensors, the code histogram, every tensor's part, dtype, shape and bytes, and the bytes per part (PARTS)
     with the header, file and tokenizer bytes, plus the caller's `extra`;
   - tokenizer/: copied from the pinned .nemo by export._extract_tokenizer.
   The scope comes from the model itself: keat_eightbit.keat_form reads its training form and refuses a state that
   does not match it. The ternary codes and scales are the modules' quantized_weight() (under S2 the codes from the
   FP32 scale and fp16(s)); the frozen parts' INT8 codes are the ones recorded at load (never re-quantized from W8),
   and pre_encode.out's are int8_quantize of its latent weight (the codes its forward uses).

2. The scoring view (derive_view): the parakeet-ternary-v1 export that M1's x1 stack reads
   (inference.runtime.load_packed), derived deterministically from the bundle alone and written by
   export.export_model itself, through a seam that points paths.ARTIFACTS (its artifact-root guard) at
   $RELABEL_ROOT/keat-prime/ (artifact_root; binary_train used the same seam for its arm C):
   - an FP32 NeMo model is built from the bundle's config and tokenizer, its config then restored to the stored one;
   - its state is the bundle's tensors widened exactly: FP16 -> FP32, W8 = fp16(q * s) for the 8-bit weights,
     codes * scale for the ternary weights; the ternary modules are swapped in (keat_eightbit.ternarize, 264 in
     Parakeet) and each answers quantized_weight() with the bundle's stored codes and scales (FP16 scales widened to
     FP32, which load_packed requires). Recomputing them from the dequantized weights would give another scale:
     mean|c * s| = s * (1 - zero fraction);
   - export_model then writes the stored codes, FP32 scales (fp32(fp16(s)) under S2), FP16 W8 (exact), and every
     other tensor as the bundle holds it. For a fake-quantized pre_encode.out the forward weight W8 is exported,
     never the latent.
   check_view compares the written view with the bundle tensor by tensor. With no scope in 8 bits the view's
   export.safetensors equals export.export_model of the training model byte for byte (for M1: its own export, sha
   a287e977...); the contract tests check this.
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

import export as ternary_export  # finetune/parakeet-ternary/export.py, unchanged
import paths  # finetune/parakeet-ternary/paths.py: export_model's artifact-root guard reads paths.ARTIFACTS
import quant
from training import eightbit as eb
from training import keab_export as _keab   # stage 4's adapter: its type-agnostic helpers only
from training import keat_eightbit as teb

FORMAT = "keat-8bit-v1"
FILE = "bundle.safetensors"
MANIFEST = "manifest.json"
TOKENIZER_DIR = ternary_export.TOKENIZER_DIR
INT8 = ".int8"
INT8_SCALE = ".int8_scale"
PARTS = ("ternary_codes", "ternary_scales", "ternary_biases", "prediction_network", "joint", "pre_encode_out",
         "other_fp16", "fp32_feature_constants", "integer_buffers")
ROOT_NAME = "keat-prime"


def keat_root() -> Path:
    """$RELABEL_ROOT/keat-prime, the stage's artifact root (and the scoring views' export root)."""
    import relabel_paths
    relabel_paths.require_mount()
    return relabel_paths.ARTIFACTS / ROOT_NAME


@contextlib.contextmanager
def artifact_root(root: Path):
    """export.export_model's artifact-root guard (paths.ARTIFACTS) pointed at `root` for the duration."""
    saved = paths.ARTIFACTS
    paths.ARTIFACTS = Path(root)
    try:
        yield
    finally:
        paths.ARTIFACTS = saved


def _float_part(key: str) -> str | None:
    for part, prefix in eb.PART_PREFIX.items():
        if key.startswith(prefix):
            return part
    return None


# --- the bundle --------------------------------------------------------------------------------------

@torch.no_grad()
def write_bundle(model: nn.Module, out_dir: Path, *, extra: dict | None = None, root: Path | None = None) -> dict:
    """Write the keat-8bit-v1 bundle of a Kea-T training model (module docstring) under out_dir; return the manifest.
    Nothing is written if the model is not in a training form or out_dir is outside the root."""
    from omegaconf import OmegaConf

    form = teb.keat_form(model)
    out_dir = _keab._inside(out_dir, root or keat_root())
    scope = form["scope"]
    in8 = eb.keys_of(teb.KEAT_SCOPES[scope])
    frozen = eb.frozen_int8(model)
    names = quant.quantized_module_names(model)
    covered = {f"{n}.weight" for n in names} | {f"{n}.bias" for n in names}
    tensors: dict[str, Tensor] = {}
    parts: dict[str, str] = {}
    layers = {}
    for name in names:
        module = model.get_submodule(name)
        codes, scale = module.quantized_weight()
        layers[name] = {"kind": "pointwise_conv1d" if isinstance(module, quant.TernaryPointwiseConv1d) else "linear",
                        "shape": list(codes.shape), "bias": module.bias is not None}
        tensors[f"{name}.codes"] = ternary_export.pack_codes(codes).cpu()
        parts[f"{name}.codes"] = "ternary_codes"
        if scope == "S2":
            half = scale.to(torch.float16)
            if not torch.equal(half.to(torch.float32), scale):
                raise ValueError(f"{name}: the S2 training scale is not exactly an FP16 value")
            tensors[f"{name}.scale"] = half.cpu()
        else:
            tensors[f"{name}.scale"] = scale.cpu()
        parts[f"{name}.scale"] = "ternary_scales"
        if module.bias is not None:
            tensors[f"{name}.bias"] = module.bias.detach().float().cpu()
            parts[f"{name}.bias"] = "ternary_biases"
    int8_tensors = {}
    for key in in8:
        q, s = frozen[key] if key in frozen else model.get_submodule(key.removesuffix(".weight")).int8()
        tensors[key + INT8], tensors[key + INT8_SCALE] = q.cpu(), s.cpu()
        parts[key + INT8] = parts[key + INT8_SCALE] = _float_part(key)
        int8_tensors[key] = {"part": _float_part(key), "shape": list(q.shape), "channel_axis": 0,
                             "codes": key + INT8, "scales": key + INT8_SCALE,
                             "source": "load-time codes (frozen)" if key in frozen else "latent weight (W8 STE)"}
        covered.add(key)
    for key, value in model.state_dict().items():
        if key in covered:
            continue
        value = value.detach().cpu()
        if not value.is_floating_point():
            tensors[key], parts[key] = value.clone(), "integer_buffers"
        elif key in ternary_export.FP32_BUFFERS:
            tensors[key], parts[key] = value.float().clone(), "fp32_feature_constants"
        else:
            half = value.to(torch.float16)
            if not torch.isfinite(half).all():
                raise ValueError(f"{key} is not finite in FP16")
            tensors[key], parts[key] = half, _float_part(key) or "other_fp16"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = ternary_export._extract_tokenizer(model, out_dir / TOKENIZER_DIR)
    path = out_dir / FILE
    tmp = out_dir / f"{FILE}.tmp{os.getpid()}"
    save_file(tensors, tmp, metadata={"format": FORMAT})
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)
    files = {FILE: ternary_export.sha256_file(path), **files}
    listing = {k: {"part": parts[k], "dtype": str(t.dtype).removeprefix("torch."), "shape": list(t.shape),
                   "bytes": t.numel() * t.element_size()} for k, t in sorted(tensors.items())}
    per_part = dict.fromkeys(PARTS, 0)
    for entry in listing.values():
        per_part[entry["part"]] += entry["bytes"]
    file_bytes = path.stat().st_size
    tensor_bytes = sum(per_part.values())
    manifest = {
        "format": FORMAT, "file": FILE, "sha256": files[FILE], "files": files,
        "scope": scope, "training_form": form,
        "int8": {"format": "symmetric per output channel (row): s = max(max|W_c| / 127, 1e-30), "
                           "q = round-half-even(W / s) clamped to [-127, 127]; forward weight W8 = fp16(q * s)",
                 "tensors": int8_tensors},
        "ternary_scale_dtype": "float16" if scope == "S2" else "float32",
        "ternary_codes": "quant.ternary_quantize from the FP32 scale; packed as export.pack_codes (2 bits)",
        "ternary_layers": layers, "fp32_buffers": list(ternary_export.FP32_BUFFERS),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "quantization_device": next(model.parameters()).device.type,
        "config": OmegaConf.to_container(model.cfg, resolve=True),
        "code_histogram": quant.code_histogram(model),
        "parameter_accounting": quant.parameter_accounting(model),
        "tensors": listing,
        "bytes": {"parts": per_part, "tensor_bytes": tensor_bytes, "header_bytes": file_bytes - tensor_bytes,
                  "file_bytes": file_bytes,
                  "tokenizer_bytes": sum((out_dir / f).stat().st_size for f in files if f != FILE)},
        "extra": extra or {},
    }
    return json.loads(_keab._atomic_json(out_dir / MANIFEST, manifest))


def read_bundle(bundle_dir: Path) -> tuple[dict, dict[str, Tensor]]:
    """(manifest, tensors) of an intact bundle; ValueError for another format, a file that does not match its
    SHA-256, or a tensor list or byte count that disagrees with the manifest."""
    bundle_dir = Path(bundle_dir).resolve()
    manifest = json.loads((bundle_dir / MANIFEST).read_text())
    if manifest.get("format") != FORMAT or manifest.get("file") != FILE:
        raise ValueError(f"{bundle_dir}: not a {FORMAT} bundle")
    for rel, digest in manifest["files"].items():
        if ternary_export.sha256_file(bundle_dir / rel) != digest:
            raise ValueError(f"{bundle_dir / rel} does not match the manifest SHA-256")
    with safe_open(bundle_dir / FILE, framework="pt") as f:
        if (f.metadata() or {}).get("format") != FORMAT:
            raise ValueError(f"{bundle_dir}: safetensors metadata format is not {FORMAT!r}")
    tensors = load_file(bundle_dir / FILE)
    if set(tensors) != set(manifest["tensors"]):
        raise ValueError(f"{bundle_dir}: tensors differ from the manifest")
    for key, entry in manifest["tensors"].items():
        t = tensors[key]
        if entry["bytes"] != t.numel() * t.element_size() or entry["shape"] != list(t.shape):
            raise ValueError(f"{bundle_dir}: {key} differs from the manifest")
    if manifest["bytes"]["file_bytes"] != (bundle_dir / FILE).stat().st_size:
        raise ValueError(f"{bundle_dir}: file size differs from the manifest")
    return manifest, tensors


def _widened(scale: Tensor) -> Tensor:
    return eb.widen(scale) if scale.dtype == torch.float16 else scale


def bundle_state(manifest: dict, tensors: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict]:
    """(the FP32 state_dict a NeMo model loads, {ternary module: (codes int8, FP32 scale)}) of a bundle."""
    tensors = dict(tensors)
    state: dict[str, Tensor] = {}
    stored = {}
    for name, layer in manifest["ternary_layers"].items():
        rows, cols = layer["shape"]
        codes = ternary_export.unpack_codes(tensors.pop(f"{name}.codes"), cols)
        scale = _widened(tensors.pop(f"{name}.scale"))
        if scale.dtype != torch.float32 or scale.shape != (rows,):
            raise ValueError(f"{name}: bad scale {scale.dtype} {tuple(scale.shape)}")
        weight = quant.dequantize(codes, scale)
        state[f"{name}.weight"] = weight.unsqueeze(2) if layer["kind"] == "pointwise_conv1d" else weight
        if layer["bias"]:
            state[f"{name}.bias"] = tensors.pop(f"{name}.bias")
        stored[name] = (codes, scale)
    for key in manifest["int8"]["tensors"]:
        state[key] = eb.w8(tensors.pop(key + INT8), tensors.pop(key + INT8_SCALE)).to(torch.float32)
    for key, value in tensors.items():
        state[key] = eb.widen(value) if value.dtype == torch.float16 else value
    return state, stored


def _stored_weight(codes: Tensor, scale: Tensor):
    def quantized_weight():
        return codes, scale
    return quantized_weight


@torch.no_grad()
def derive_view(bundle_dir: Path, view_dir: Path, *, root: Path | None = None, extra: dict | None = None) -> dict:
    """The scoring view of a bundle (module docstring), written by export.export_model under view_dir; returns its
    manifest. The view is checked against the bundle tensor by tensor before returning."""
    root = Path(root or keat_root())
    view_dir = _keab._inside(view_dir, root)
    bundle_dir = Path(bundle_dir).resolve()
    manifest, tensors = read_bundle(bundle_dir)
    state, stored = bundle_state(manifest, tensors)
    model = _keab._build_model(manifest["config"], bundle_dir / TOKENIZER_DIR)
    model.load_state_dict(state, strict=True)
    names = teb.ternarize(model)
    if names != sorted(manifest["ternary_layers"]):
        raise ValueError("the bundle's ternary layers are not the model's quantized modules")
    full = manifest["parameter_accounting"]["total_parameters"] == quant.TOTAL_PARAMETERS
    if full and len(names) != quant.EXPECTED_MODULES:
        raise ValueError(f"a full Parakeet bundle must have {quant.EXPECTED_MODULES} ternary layers")
    for name in names:
        model.get_submodule(name).quantized_weight = _stored_weight(*stored[name])
    keat = {"bundle": str(bundle_dir), "bundle_format": FORMAT, "bundle_sha256": manifest["sha256"],
            "scope": manifest["scope"], "training_form": manifest["training_form"]}
    with artifact_root(root):
        view = ternary_export.export_model(model, view_dir, {"keat": keat, **(extra or {})})
    check_view(view_dir, manifest, tensors)
    return view


def check_view(view_dir: Path, manifest: dict, tensors: dict[str, Tensor]) -> None:
    """The view's tensors are the bundle's, widened exactly (ValueError otherwise): the same ternary codes, the
    scales widened to FP32, W8 for the INT8 weights, every other tensor in the bundle's dtype and value."""
    view = load_file(Path(view_dir) / ternary_export.FILE)
    int8 = manifest["int8"]["tensors"]
    expect = set()
    for key, t in tensors.items():
        if key.endswith(INT8_SCALE):
            continue
        if key.endswith(INT8):
            name = key.removesuffix(INT8)
            if name not in int8:
                raise ValueError(f"{name}: INT8 codes not listed in the manifest")
            want = eb.w8(t, tensors[name + INT8_SCALE])
        elif key.endswith(".scale") and key.removesuffix(".scale") in manifest["ternary_layers"]:
            name, want = key, _widened(t)
        else:
            name, want = key, t
        expect.add(name)
        got = view.get(name)
        if got is None or got.dtype != want.dtype or not torch.equal(got, want):
            raise ValueError(f"scoring view: {name} differs from the bundle")
    if expect != set(view):
        raise ValueError(f"scoring view: tensors differ from the bundle's ({sorted(set(view) ^ expect)[:5]})")


def megabytes(n: int) -> float:
    return n / 1e6


def tensor_megabytes(manifest: dict) -> float:
    """Tensor bytes (file minus header) in MB (10^6 bytes), the unit of the plan's 180.688 / 173.524 / 166.734 MB."""
    return megabytes(manifest["bytes"]["tensor_bytes"])


def view_tensor_bytes(view_manifest: dict) -> int:
    """export.export_model's tensor bytes: the file minus its header."""
    b = view_manifest["bytes"]
    return b["file_bytes"] - b["header_bytes"]
