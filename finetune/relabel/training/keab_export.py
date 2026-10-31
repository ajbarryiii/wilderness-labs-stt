"""The relabel-owned export adapter of stage 4 (keab-prime plan "Export"): the 8-bit bundle of a Kea-B training model
and the parakeet-binary-v1 scoring view derived from it.

binary_export.py stores binary scales in FP32 and every other float weight in FP16; it has no 8-bit representation
and refuses directories outside the binary artifact root. This module writes, per model:

1. The bundle, the artifact of record (format "keab-8bit-v1"; write_bundle), one directory:
   - bundle.safetensors (metadata {"format": "keab-8bit-v1"}):
       <binary module>.codes        the binary codes, packed as binary_export.pack_codes (uint8 [out, ceil(in/8)])
       <binary module>.scale        the row scales: FP32, or FP16 under S2 (the training form's fp16(alpha))
       <binary module>.bias         FP32, if the module has one (none in Parakeet)
       <weight key>.int8            INT8 codes [rows, cols] of each weight in 8 bits (eightbit.KEAB_SCOPES)
       <weight key>.int8_scale      their FP32 per-channel (row) scales
       every other state_dict tensor exactly as binary_export.export_model stores it: FP16 for float weights,
       FP32 for its feature constants (binary_export.FP32_BUFFERS: the preprocessor's window and filterbank),
       integer buffers as they are;
   - manifest.json: format, scope and training form, SHA-256 of every file, the NeMo config, the binary layers,
     the 8-bit tensors, every tensor's part, dtype, shape and bytes, and the bytes per part (PARTS) with the header,
     file and tokenizer bytes, plus the caller's `extra`;
   - tokenizer/: copied from the pinned .nemo by parakeet-ternary export._extract_tokenizer (as binary_export).
   The scope comes from the model itself: eightbit.keab_form reads its training form and refuses a state that does
   not match it. The frozen parts' codes are the ones recorded at load (never re-quantized from W8), and
   pre_encode.out's are int8_quantize of its latent weight (the codes its forward uses).

2. The scoring view (derive_view): the parakeet-binary-v1 export that the gpu1bit runtime reads, derived
   deterministically from the bundle alone and written by binary_export.export_model itself, through a seam that
   points binary_paths.ARTIFACTS (its artifact-root guard) at $RELABEL_ROOT/keab-prime/ (artifact_root):
   - an FP32 NeMo model is built from the bundle's config and tokenizer, its config then restored to the stored one;
   - its state is the bundle's tensors widened exactly: FP16 -> FP32, W8 = fp16(q * s) for the 8-bit weights,
     codes * scale for the binary weights; the binary modules are swapped in (eightbit.binarize, 264 in Parakeet)
     and each answers quantized_weight() with the bundle's codes and scales (FP16 scales widened to FP32);
   - export_model then writes FP16 W8 (exact), FP32 scales fp32(fp16(alpha)) under S2, and every other tensor as
     the bundle holds it. For a fake-quantized pre_encode.out the forward weight W8 is exported, never the latent.
   With no scope in 8 bits the view's export.safetensors equals binary_export.export_model of the training model
   byte for byte (for MB1: its existing export, sha 9873d2f1...); the contract tests check this.

write_view exports a model directly (the informational palettized views of step 0, which have no bundle).
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

from training import eightbit as eb

import binary_export  # noqa: E402  finetune/parakeet-binary (on sys.path through training.eightbit)
import binary_paths  # noqa: E402
import binary_quant  # noqa: E402
import export as ternary_export  # noqa: E402  finetune/parakeet-ternary/export.py

FORMAT = "keab-8bit-v1"
FILE = "bundle.safetensors"
MANIFEST = "manifest.json"
TOKENIZER_DIR = binary_export.TOKENIZER_DIR
INT8 = ".int8"
INT8_SCALE = ".int8_scale"
PARTS = ("binary_codes", "binary_scales", "binary_biases", "prediction_network", "joint", "pre_encode_out",
         "other_fp16", "fp32_feature_constants", "integer_buffers")
ROOT_NAME = "keab-prime"


def keab_root() -> Path:
    """$RELABEL_ROOT/keab-prime, the stage's artifact root (and the scoring views' export root)."""
    import relabel_paths
    relabel_paths.require_mount()
    return relabel_paths.ARTIFACTS / ROOT_NAME


def _inside(path: Path, root: Path) -> Path:
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{path} is outside the artifact root {root}")
    return path


@contextlib.contextmanager
def artifact_root(root: Path):
    """binary_export's artifact-root guard (binary_paths.ARTIFACTS) pointed at `root` for the duration."""
    saved = binary_paths.ARTIFACTS
    binary_paths.ARTIFACTS = Path(root)
    try:
        yield
    finally:
        binary_paths.ARTIFACTS = saved


def _float_part(key: str) -> str | None:
    for part, prefix in eb.PART_PREFIX.items():
        if key.startswith(prefix):
            return part
    return None


def _atomic_json(path: Path, data: dict) -> str:
    text = json.dumps(data, indent=2) + "\n"
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)
    return text


# --- the bundle --------------------------------------------------------------------------------------

@torch.no_grad()
def write_bundle(model: nn.Module, out_dir: Path, *, extra: dict | None = None, root: Path | None = None) -> dict:
    """Write the keab-8bit-v1 bundle of a Kea-B training model (module docstring) under out_dir; return the manifest.
    Nothing is written if the model is not in a training form or out_dir is outside the root."""
    from omegaconf import OmegaConf

    form = eb.keab_form(model)
    out_dir = _inside(out_dir, root or keab_root())
    scope = form["scope"]
    in8 = eb.keys_of(eb.KEAB_SCOPES[scope])
    frozen = eb.frozen_int8(model)
    names = binary_quant.quantized_module_names(model)
    covered = {f"{n}.weight" for n in names} | {f"{n}.bias" for n in names}
    tensors: dict[str, Tensor] = {}
    parts: dict[str, str] = {}
    layers = {}
    for name in names:
        module = model.get_submodule(name)
        codes, scale = module.quantized_weight()
        layers[name] = {"kind": "pointwise_conv1d" if isinstance(module, binary_quant.BinaryPointwiseConv1d)
                        else "linear", "shape": list(codes.shape), "bias": module.bias is not None}
        tensors[f"{name}.codes"] = binary_export.pack_codes(codes).cpu()
        parts[f"{name}.codes"] = "binary_codes"
        if scope == "S2":
            half = scale.to(torch.float16)
            if not torch.equal(half.to(torch.float32), scale):
                raise ValueError(f"{name}: the S2 training scale is not exactly an FP16 value")
            tensors[f"{name}.scale"] = half.cpu()
        else:
            tensors[f"{name}.scale"] = scale.cpu()
        parts[f"{name}.scale"] = "binary_scales"
        if module.bias is not None:
            tensors[f"{name}.bias"] = module.bias.detach().float().cpu()
            parts[f"{name}.bias"] = "binary_biases"
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
        elif key in binary_export.FP32_BUFFERS:
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
    files = {FILE: binary_export.sha256_file(path), **files}
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
        "binary_scale_dtype": "float16" if scope == "S2" else "float32",
        "binary_layers": layers, "bit_layout": binary_export.BIT_LAYOUT, "fp32_buffers": list(binary_export.FP32_BUFFERS),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "quantization_device": next(model.parameters()).device.type,
        "config": OmegaConf.to_container(model.cfg, resolve=True),
        "parameter_accounting": binary_quant.parameter_accounting(model),
        "tensors": listing,
        "bytes": {"parts": per_part, "tensor_bytes": tensor_bytes, "header_bytes": file_bytes - tensor_bytes,
                  "file_bytes": file_bytes,
                  "tokenizer_bytes": sum((out_dir / f).stat().st_size for f in files if f != FILE)},
        "extra": extra or {},
    }
    return json.loads(_atomic_json(out_dir / MANIFEST, manifest))


def read_bundle(bundle_dir: Path) -> tuple[dict, dict[str, Tensor]]:
    """(manifest, tensors) of an intact bundle; ValueError for another format, a file that does not match its
    SHA-256, or a tensor list or byte count that disagrees with the manifest."""
    bundle_dir = Path(bundle_dir).resolve()
    manifest = json.loads((bundle_dir / MANIFEST).read_text())
    if manifest.get("format") != FORMAT or manifest.get("file") != FILE:
        raise ValueError(f"{bundle_dir}: not a {FORMAT} bundle")
    for rel, digest in manifest["files"].items():
        if binary_export.sha256_file(bundle_dir / rel) != digest:
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


def bundle_state(manifest: dict, tensors: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict]:
    """(the FP32 state_dict a NeMo model loads, {binary module: (codes int8, FP32 scale)}) of a bundle."""
    tensors = dict(tensors)
    state: dict[str, Tensor] = {}
    stored = {}
    for name, layer in manifest["binary_layers"].items():
        rows, cols = layer["shape"]
        codes = binary_export.unpack_codes(tensors.pop(f"{name}.codes"), cols)
        scale = tensors.pop(f"{name}.scale")
        scale = eb.widen(scale) if scale.dtype == torch.float16 else scale
        if scale.dtype != torch.float32 or scale.shape != (rows,):
            raise ValueError(f"{name}: bad scale {scale.dtype} {tuple(scale.shape)}")
        weight = binary_quant.dequantize(codes, scale)
        state[f"{name}.weight"] = weight.unsqueeze(2) if layer["kind"] == "pointwise_conv1d" else weight
        if layer["bias"]:
            state[f"{name}.bias"] = tensors.pop(f"{name}.bias")
        stored[name] = (codes, scale)
    for key in manifest["int8"]["tensors"]:
        state[key] = eb.w8(tensors.pop(key + INT8), tensors.pop(key + INT8_SCALE)).to(torch.float32)
    for key, value in tensors.items():
        state[key] = eb.widen(value) if value.dtype == torch.float16 else value
    return state, stored


def _build_model(config: dict, tokenizer_dir: Path):
    """An FP32 eval-mode EncDecRNNTBPEModel from a stored config (tokenizer read from tokenizer_dir), with the
    model's config then set back to the stored one exactly."""
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    from omegaconf import OmegaConf, open_dict

    cfg = OmegaConf.create(config)
    if cfg.get("target") is not None and cfg.get("target") != "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel":
        raise ValueError(f"unexpected model target {cfg.get('target')!r}")
    with open_dict(cfg):
        cfg.tokenizer.dir = str(tokenizer_dir)
        for key, filename in binary_export.TOKENIZER_FILES.items():
            cfg.tokenizer[key] = str(tokenizer_dir / filename)
        for ds in ("train_ds", "validation_ds", "test_ds"):
            if ds in cfg:
                cfg[ds] = None
    model = EncDecRNNTBPEModel(cfg=cfg).to(torch.float32)
    model.cfg = OmegaConf.create(config)
    return model.eval()


def _stored_weight(codes: Tensor, scale: Tensor):
    def quantized_weight():
        return codes, scale
    return quantized_weight


@torch.no_grad()
def derive_view(bundle_dir: Path, view_dir: Path, *, root: Path | None = None, extra: dict | None = None) -> dict:
    """The scoring view of a bundle (module docstring), written by binary_export.export_model under view_dir; returns
    its manifest. The view is checked against the bundle tensor by tensor before returning."""
    root = Path(root or keab_root())
    view_dir = _inside(view_dir, root)
    bundle_dir = Path(bundle_dir).resolve()
    manifest, tensors = read_bundle(bundle_dir)
    state, stored = bundle_state(manifest, tensors)
    model = _build_model(manifest["config"], bundle_dir / TOKENIZER_DIR)
    model.load_state_dict(state, strict=True)
    names = eb.binarize(model)
    if names != sorted(manifest["binary_layers"]):
        raise ValueError("the bundle's binary layers are not the model's quantized modules")
    full = manifest["parameter_accounting"]["total_parameters"] == binary_quant.TOTAL_PARAMETERS
    if full and len(names) != binary_quant.EXPECTED_MODULES:
        raise ValueError(f"a full Parakeet bundle must have {binary_quant.EXPECTED_MODULES} binary layers")
    for name in names:
        model.get_submodule(name).quantized_weight = _stored_weight(*stored[name])
    keab = {"bundle": str(bundle_dir), "bundle_format": FORMAT, "bundle_sha256": manifest["sha256"],
            "scope": manifest["scope"], "training_form": manifest["training_form"]}
    with artifact_root(root):
        view = binary_export.export_model(model, view_dir, {"keab": keab, **(extra or {})})
    check_view(view_dir, manifest, tensors)
    return view


def check_view(view_dir: Path, manifest: dict, tensors: dict[str, Tensor]) -> None:
    """The view's tensors are the bundle's, widened exactly (ValueError otherwise)."""
    view = load_file(Path(view_dir) / binary_export.FILE)
    int8 = manifest["int8"]["tensors"]
    expect = set()
    for key, t in tensors.items():
        if key.endswith(INT8_SCALE):
            continue
        if key.endswith(INT8):
            name = key.removesuffix(INT8)
            want = eb.w8(t, tensors[name + INT8_SCALE])
            if name not in int8:
                raise ValueError(f"{name}: INT8 codes not listed in the manifest")
        elif key.endswith(".scale") and key.removesuffix(".scale") in manifest["binary_layers"]:
            name, want = key, (eb.widen(t) if t.dtype == torch.float16 else t)
        else:
            name, want = key, t
        expect.add(name)
        got = view.get(name)
        if got is None or got.dtype != want.dtype or not torch.equal(got, want):
            raise ValueError(f"scoring view: {name} differs from the bundle")
    if expect != set(view):
        raise ValueError(f"scoring view: tensors differ from the bundle's ({sorted(set(view) ^ expect)[:5]})")


@torch.no_grad()
def write_view(model: nn.Module, view_dir: Path, *, root: Path | None = None, extra: dict | None = None) -> dict:
    """binary_export.export_model of `model` under view_dir inside the artifact-root seam (no bundle)."""
    root = Path(root or keab_root())
    view_dir = _inside(view_dir, root)
    with artifact_root(root):
        return binary_export.export_model(model, view_dir, extra or {})


def megabytes(n: int) -> float:
    return n / 1e6


def tensor_megabytes(manifest: dict) -> float:
    """Tensor bytes (file minus header) in MB (10^6 bytes), the unit of the plan's 105.19 / ~98.0 / ~91.2 MB."""
    return megabytes(manifest["bytes"]["tensor_bytes"])


def view_tensor_bytes(view_manifest: dict) -> int:
    b = view_manifest["bytes"]
    return b["file_bytes"] - b["header_bytes"]

