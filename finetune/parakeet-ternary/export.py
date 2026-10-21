"""Packed ternary export of a QAT Parakeet-TDT model and its FP32 NeMo reconstruction. See DESIGN.md "Export".

Format "parakeet-ternary-v1", one directory:
- export.safetensors: for each of the 264 quantized modules "<name>.codes" (uint8, 2-bit
  codes packed four per byte by pack_codes, [out, ceil(in/4)]; pointwise convolutions as
  their [out, in] matrix) and "<name>.scale" (FP32 [out]), plus "<name>.bias" in FP32 if
  the module has one (none in this model). Every other state_dict tensor in FP16, except:
  integer buffers (BatchNorm num_batches_tracked) keep their dtype, and the
  feature-extraction constants preprocessor.featurizer.{window,fb} (not learned, 33k
  values) stay FP32 so the rebuilt model computes the same log-mel features.
- manifest.json: format, SHA-256 of every file, the NeMo model config
  (OmegaConf.to_container, resolved), per-layer shapes, code histogram, parameter
  accounting, byte breakdown and the caller's `extra`.
- tokenizer/: tokenizer.model, vocab.txt and tokenizer.vocab extracted from the pinned
  .nemo (the SentencePiece model is checked against the exported model's tokenizer).

load_export() rebuilds a plain NeMo EncDecRNNTBPEModel from the stored config and
tokenizer files alone (it does not read paths.MODEL_FILE), with dequantized FP32
weights and no ternary modules. It runs dense FP32 matrices; nothing here claims a
speed or energy benefit.

pack_codes / unpack_codes are copied verbatim from finetune/whisper-ternary/export.py
(same bit layout); tests/test_export.py checks them against that file.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tarfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

import paths
import quant

FORMAT = "parakeet-ternary-v1"
FILE = "export.safetensors"
MANIFEST = "manifest.json"
TOKENIZER_DIR = "tokenizer"
# config.tokenizer key -> file name in the export's tokenizer directory
TOKENIZER_FILES = {"model_path": "tokenizer.model", "vocab_path": "vocab.txt",
                   "spe_tokenizer_vocab": "tokenizer.vocab"}
FP32_BUFFERS = ("preprocessor.featurizer.window", "preprocessor.featurizer.fb")
_SHIFTS = torch.tensor([0, 2, 4, 6], dtype=torch.uint8)


# --- copied verbatim from finetune/whisper-ternary/export.py -------------------------------------
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
# --------------------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _artifact_dir(out_dir: Path) -> Path:
    paths.require_mount()
    out_dir = Path(out_dir).resolve()
    if not out_dir.is_relative_to(paths.ARTIFACTS.resolve()):
        raise ValueError(f"exports must live under {paths.ARTIFACTS}, got {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _require_full_quantization(model: nn.Module, action: str) -> list[str]:
    """The ternary module names; ValueError if there are none or any has weight fraction < 1."""
    names = quant.quantized_module_names(model)
    if not names:
        raise ValueError(f"cannot {action}: model has no ternary modules")
    partial = {n: model.get_submodule(n).fraction for n in names if model.get_submodule(n).fraction < 1.0}
    if partial:
        raise ValueError(
            f"cannot {action}: {len(partial)} ternary module(s) have weight fraction < 1 (lowest "
            f"{min(partial.values())}, first {min(partial)}). A fraction below 1 is a training-time "
            "quantization ramp, not a deployable model; finish the ramp (quant.set_weight_fraction(model, 1.0)).")
    return names


def _extract_tokenizer(model: nn.Module, target: Path) -> dict[str, str]:
    """Copy the tokenizer artifacts from the pinned .nemo into target; return {file: sha256}."""
    tmp = target.with_name(target.name + f".tmp{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with tarfile.open(paths.MODEL_FILE, "r:*") as tar:
        members = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}
        for key, filename in TOKENIZER_FILES.items():
            ref = str(model.cfg.tokenizer[key])
            member = members[ref.removeprefix("nemo:")]
            (tmp / filename).write_bytes(tar.extractfile(member).read())
    proto = model.tokenizer.tokenizer.serialized_model_proto()
    if (tmp / TOKENIZER_FILES["model_path"]).read_bytes() != proto:
        raise ValueError(f"the tokenizer in {paths.MODEL_FILE} is not the exported model's tokenizer")
    shutil.rmtree(target, ignore_errors=True)
    os.replace(tmp, target)
    return {f"{TOKENIZER_DIR}/{f}": sha256_file(target / f) for f in TOKENIZER_FILES.values()}


@torch.no_grad()
def export_model(model: nn.Module, out_dir: Path, extra: dict) -> dict:
    """Write export.safetensors, tokenizer/ and manifest.json under out_dir; return the manifest.

    Raises ValueError before writing anything if the model has no ternary modules or any
    has weight fraction < 1. Codes and scales are computed on the model's current device;
    run reconstruction_check with the QAT model on that same device (FP32 row means can
    differ by one ulp between CPU and GPU).
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
        layers[name] = {"kind": "pointwise_conv1d" if isinstance(module, quant.TernaryPointwiseConv1d) else "linear",
                        "shape": list(codes.shape), "bias": bias is not None}
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
    files = _extract_tokenizer(model, out_dir / TOKENIZER_DIR)
    path = out_dir / FILE
    tmp = out_dir / f"{FILE}.tmp{os.getpid()}"
    save_file(tensors, tmp, metadata={"format": FORMAT})
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)
    files = {FILE: sha256_file(path), **files}
    accounting = quant.parameter_accounting(model)
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
        "quantizer_source_sha256": {"whisper-ternary/quant.py": quant.whisper_source_sha256("quant")},
        "versions": {"nemo": nemo.__version__, "torch": torch.__version__},
        "config": OmegaConf.to_container(model.cfg, resolve=True),
        "quantized_layers": layers,
        "fp32_buffers": list(FP32_BUFFERS),
        "code_histogram": quant.code_histogram(model),
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
        weight = quant.dequantize(unpack_codes(tensors.pop(f"{name}.codes"), in_features),
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


def _pad_tokens(seqs: list, device: torch.device) -> tuple[Tensor, Tensor]:
    seqs = [torch.as_tensor(s, dtype=torch.long).flatten().cpu() for s in seqs]
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    targets = torch.zeros(len(seqs), max(1, int(lengths.max())), dtype=torch.long)
    for i, s in enumerate(seqs):
        targets[i, :len(s)] = s
    return targets.to(device), lengths.to(device)


@torch.no_grad()
def reconstruction_check(qat_model: nn.Module, rebuilt: nn.Module, batch: tuple) -> dict:
    """Exact code/scale agreement plus FP32 encoder, joint and greedy agreement on one fixed audio batch.

    batch = (audio [B, T] float32 at 16 kHz, lengths [B] int64[, texts]). Codes are recovered from the
    rebuilt weight as sign(W) and scales as the row max |W| (re-running absmean on a
    dequantized row with zero codes would shrink the scale); rows whose codes are all zero
    have no observable scale and are skipped. Differences in the forward pass come only
    from FP16 storage of the non-ternary tensors. The joint output is compared on
    teacher-forced targets, valid frames and tokens only: the tokenized texts if the batch
    has a third element (e.g. reference transcripts), else the QAT model's greedy tokens.
    Both models must be in eval mode on the same device; the QAT model at fraction 1.
    """
    import evaluate as ev

    if qat_model.training or rebuilt.training:
        raise ValueError("both models must be in eval mode")
    names = _require_full_quantization(qat_model, "run the reconstruction check")
    device = next(qat_model.parameters()).device
    if next(rebuilt.parameters()).device != device:
        raise ValueError("both models must be on the same device")
    if quant.quantized_module_names(rebuilt):
        raise ValueError("the rebuilt model must not contain ternary modules")
    codes_exact = scales_exact = True
    with ev.strict_fp32(device):
        for name in names:
            codes, scale = qat_model.get_submodule(name).quantized_weight()
            weight = rebuilt.get_submodule(name).weight
            weight = weight.squeeze(2) if weight.dim() == 3 else weight
            observable = codes.ne(0).any(dim=1)
            codes_exact &= torch.equal(torch.sign(weight).to(torch.int8), codes)
            scales_exact &= torch.equal(weight.abs().amax(dim=1)[observable], scale[observable])
        audio, lengths = batch[0].to(device, torch.float32), batch[1].to(device)
        out = {}
        for key, model in (("qat", qat_model), ("rebuilt", rebuilt)):
            with ev.inference_settings(model):
                enc, enc_len = model.forward(input_signal=audio, input_signal_length=lengths)
                hyps = model.decoding.rnnt_decoder_predictions_tensor(
                    encoder_output=enc, encoded_lengths=enc_len, return_hypotheses=True)
            out[key] = (enc, enc_len, hyps)
        enc_q, len_q, hyps_q = out["qat"]
        enc_r, len_r, hyps_r = out["rebuilt"]
        if not torch.equal(len_q, len_r):
            raise RuntimeError("encoder lengths differ")
        frames = torch.arange(enc_q.shape[2], device=device)[None, :] < len_q[:, None]  # [B, T]
        enc_diff = (enc_q - enc_r).abs().transpose(1, 2)[frames].max()
        if len(batch) > 2:
            target_source = "reference texts"
            targets, target_len = _pad_tokens([qat_model.tokenizer.text_to_ids(t) for t in batch[2]], device)
        else:
            target_source = "qat greedy tokens"
            targets, target_len = _pad_tokens([h.y_sequence for h in hyps_q], device)
        joints = []
        for model, enc in ((qat_model, enc_q), (rebuilt, enc_r)):
            dec, _, _ = model.decoder(targets=targets, target_length=target_len)
            joints.append(model.joint.joint(enc.transpose(1, 2), dec.transpose(1, 2)))  # [B, T, U+1, V+1+D]
        tokens = torch.arange(targets.shape[1] + 1, device=device)[None, :] <= target_len[:, None]  # [B, U+1]
        valid = frames[:, :, None] & tokens[:, None, :]
        joint_diff = (joints[0] - joints[1]).abs()[valid].max()
    texts_q = [h.text for h in hyps_q]
    texts_r = [h.text for h in hyps_r]
    return {
        "codes_exact": bool(codes_exact),
        "scales_exact": bool(scales_exact),
        "encoder_max_abs_diff": float(enc_diff),
        "encoder_max_abs": float(enc_q.abs().transpose(1, 2)[frames].max()),
        "joint_output_max_abs_diff": float(joint_diff),
        "joint_targets": target_source,
        "joint_target_tokens": int(target_len.sum()),
        "greedy_hyps_equal": texts_q == texts_r,
        "greedy_hyps_qat": texts_q,
        "greedy_hyps_rebuilt": texts_r,
        "checked_layers": len(names),
        "batch_size": int(audio.shape[0]),
    }
