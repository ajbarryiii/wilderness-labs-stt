# load_kea_b.py: standalone loader for Kea-B 0.1 (1-bit encoder, 8-bit decoder Parakeet-TDT-0.6B-v2).
#
# MIT License
#
# Copyright (c) 2026 ajbarryiii
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# This license covers this file only. The model weights it loads are a modified
# version of nvidia/parakeet-tdt-0.6b-v2 and are licensed CC-BY-4.0 (see LICENSE
# and NOTICE in the model repository).
"""Load and run Kea-B 0.1 with only NeMo (nemo_toolkit[asr] and its torch), safetensors, numpy and
soundfile. Standalone: it does not need the training code.

    python load_kea_b.py clip.wav [clip2.flac ...]            # 16 kHz audio; one transcript per file
    python load_kea_b.py --compat clip.wav                     # the same model from compat/

    import sys; sys.path.insert(0, "kea-b-0.1")               # directory holding this repository's files
    from load_kea_b import load, transcribe
    model = load("kea-b-0.1", device="cuda")                   # or "cpu"; "kea-b-0.1/compat" also works
    print(transcribe(model, ["clip.wav"]))                     # list in -> list of str out

Two formats, told apart by manifest.json's "format"; both rebuild the same FP32 weights, bit for bit:

"keab-8bit-v1" (the repository root; the artifact of record): bundle.safetensors holds
- for each of the 264 binary encoder matrices, "<name>.codes": one bit per weight packed eight per
  byte (code j of a row is bit j % 8 of byte j // 8, least significant bit first; bit 1 = +1,
  bit 0 = -1; unused tail bits 0; uint8 [out, ceil(in/8)]; pointwise convolutions as their
  [out, in] matrix) and "<name>.scale", one scale per output row (FP16 in this release);
  the weight is codes * scale, computed in FP32;
- for each 8-bit matrix (the prediction network, the joint, the encoder's pre_encode.out),
  "<key>.int8" (int8 [rows, cols]) and "<key>.int8_scale" (FP32 [rows]); the weight is
  fp16(q * s): the FP32 product rounded once to FP16, then widened to FP32;
- every other state_dict tensor in FP16, except the feature-extraction window and filterbank
  (FP32) and integer BatchNorm counters (int64).
"parakeet-binary-v1" (compat/): export.safetensors holds the same binary codes with FP32 row
scales and every other tensor in FP16 (the 8-bit matrices as their FP16 values fp16(q * s)).

Each directory has manifest.json (format, SHA-256 of the weights and the three tokenizer files, the
NeMo model config, the layer list) and tokenizer/ (the base model's SentencePiece files). load()
checks every listed file against its SHA-256 and refuses unknown formats, missing files and
mismatches. It rebuilds a plain NeMo EncDecRNNTBPEModel in FP32; it does not use 1-bit or 8-bit
kernels and needs the memory and compute of the FP32 original, so no speed or energy benefit is
claimed for it.

Tested environment: NeMo 3.0.0, torch 2.11.0+cu128. The reported WERs were measured decoding on
an NVIDIA RTX 5090 (CUDA, strict FP32). Loading on the CPU was tested, decoding on the CPU was not.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

BUNDLE_FORMAT = "keab-8bit-v1"
VIEW_FORMAT = "parakeet-binary-v1"
WEIGHTS_FILES = {BUNDLE_FORMAT: "bundle.safetensors", VIEW_FORMAT: "export.safetensors"}
MANIFEST = "manifest.json"
TOKENIZER_DIR = "tokenizer"
# config.tokenizer key -> file name in the tokenizer directory
TOKENIZER_FILES = {"model_path": "tokenizer.model", "vocab_path": "vocab.txt",
                   "spe_tokenizer_vocab": "tokenizer.vocab"}
MODEL_TARGET = "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel"
LAYER_KINDS = ("linear", "pointwise_conv1d")
INT8, INT8_SCALE = ".int8", ".int8_scale"
SAMPLE_RATE = 16000
# As in the evaluation: inputs shorter than 30 ms are zero-padded to 30 ms (NeMo's per-feature
# normalization fails on a one-frame input). Every reported test utterance is longer.
MIN_DECODE_SAMPLES = 480
MAX_BATCH_SECONDS = 1200.0
# NeMo's greedy TDT decoder CUDA-graph mode used by the evaluation (native PyTorch CUDA graphs).
DECODER_GRAPHS_MODE = "no_while_loops"
_SHIFTS = torch.arange(8, dtype=torch.uint8)


def unpack_codes(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """uint8 [out, ceil(in/8)] -> int8 [out, in] in {-1, +1}; rejects a wrong dtype or shape and nonzero tail bits."""
    if packed.dtype != torch.uint8 or packed.dim() != 2 or packed.shape[1] != math.ceil(in_features / 8):
        raise ValueError(f"expected uint8 [out, {math.ceil(in_features / 8)}], got {packed.dtype} {tuple(packed.shape)}")
    bits = ((packed[..., None] >> _SHIFTS.to(packed.device)) & 1).flatten(1)
    if bits[:, in_features:].any():
        raise ValueError("invalid packed codes: nonzero tail padding")
    return (2 * bits[:, :in_features].to(torch.int8) - 1).to(torch.int8)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_format(directory: str | Path = ".") -> str:
    """The manifest's format ("keab-8bit-v1" or "parakeet-binary-v1"); ValueError for any other."""
    path = Path(directory) / MANIFEST
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing")
    fmt = json.loads(path.read_text()).get("format")
    if fmt not in WEIGHTS_FILES:
        raise ValueError(f"unknown format {fmt!r} in {path}; expected one of {sorted(WEIGHTS_FILES)}")
    return fmt


def _checked_manifest(directory: Path) -> dict:
    """manifest.json after the format, file-list and SHA-256 checks (module docstring)."""
    fmt = detect_format(directory)
    manifest = json.loads((directory / MANIFEST).read_text())
    weights = WEIGHTS_FILES[fmt]
    if manifest.get("file") != weights:
        raise ValueError(f"manifest names weights file {manifest.get('file')!r}, not {weights!r}")
    files = manifest.get("files") or {}
    expected = {weights, *(f"{TOKENIZER_DIR}/{f}" for f in TOKENIZER_FILES.values())}
    if set(files) != expected or files[weights] != manifest.get("sha256"):
        raise ValueError(f"manifest is inconsistent: files {sorted(files)}")
    for rel, digest in files.items():
        path = directory / rel
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing")
        if _sha256(path) != digest:
            raise ValueError(f"{path} does not match the manifest SHA-256")
    return manifest


def _binary_weight(tensors: dict, name: str, layer: dict, scale_dtypes: tuple) -> torch.Tensor:
    if layer.get("kind") not in LAYER_KINDS:
        raise ValueError(f"{name}: unknown layer kind {layer.get('kind')!r}")
    out_features, in_features = layer["shape"]
    scale = tensors.pop(f"{name}.scale")
    if scale.dtype not in scale_dtypes or tuple(scale.shape) != (out_features,):
        raise ValueError(f"{name}.scale: expected {scale_dtypes} [{out_features}], got {scale.dtype} {tuple(scale.shape)}")
    # FP16 scales widen exactly; codes are +-1, so codes * scale is exact: every weight is +-scale of its row.
    weight = unpack_codes(tensors.pop(f"{name}.codes"), in_features).float() * scale.float()[:, None]
    if weight.shape != (out_features, in_features):
        raise ValueError(f"{name}: shape {tuple(weight.shape)} != {layer['shape']}")
    return weight.unsqueeze(2) if layer["kind"] == "pointwise_conv1d" else weight


def _bundle_state(directory: Path, manifest: dict) -> dict[str, torch.Tensor]:
    path = directory / WEIGHTS_FILES[BUNDLE_FORMAT]
    with safe_open(path, framework="pt") as f:
        if (f.metadata() or {}).get("format") != BUNDLE_FORMAT:
            raise ValueError(f"{path}: safetensors metadata format is not {BUNDLE_FORMAT!r}")
    tensors = load_file(path)
    listed = manifest.get("tensors")
    if listed is not None and set(listed) != set(tensors):
        raise ValueError(f"{path}: tensors differ from the manifest")
    state: dict[str, torch.Tensor] = {}
    for name, layer in manifest["binary_layers"].items():
        state[f"{name}.weight"] = _binary_weight(tensors, name, layer, (torch.float16, torch.float32))
        if layer.get("bias"):
            state[f"{name}.bias"] = tensors.pop(f"{name}.bias").float()
    for key in manifest["int8"]["tensors"]:
        q, s = tensors.pop(key + INT8), tensors.pop(key + INT8_SCALE)
        if q.dtype != torch.int8 or s.dtype != torch.float32 or q.dim() != 2 or tuple(s.shape) != (q.shape[0],):
            raise ValueError(f"{key}: expected int8 codes [rows, cols] and FP32 scales [rows]")
        state[key] = (q.to(torch.float32) * s[:, None]).to(torch.float16).to(torch.float32)
    leftover = [k for k in tensors if k.endswith((INT8, INT8_SCALE, ".codes"))]
    if leftover:
        raise ValueError(f"{path}: quantized tensors not listed in the manifest: {leftover[:5]}")
    state.update((k, v.float() if v.is_floating_point() else v) for k, v in tensors.items())
    return state


def _view_state(directory: Path, manifest: dict) -> dict[str, torch.Tensor]:
    tensors = load_file(directory / WEIGHTS_FILES[VIEW_FORMAT])
    state: dict[str, torch.Tensor] = {}
    for name, layer in manifest["quantized_layers"].items():
        state[f"{name}.weight"] = _binary_weight(tensors, name, layer, (torch.float32,))
        if layer.get("bias"):
            state[f"{name}.bias"] = tensors.pop(f"{name}.bias")
    state.update((k, v.float() if v.is_floating_point() else v) for k, v in tensors.items())
    return state


def load_state(directory: str | Path = ".") -> tuple[dict, dict[str, torch.Tensor]]:
    """(manifest, FP32 state_dict) of a checked directory in either format, without building a model."""
    directory = Path(directory).resolve()
    manifest = _checked_manifest(directory)
    if manifest["format"] == BUNDLE_FORMAT:
        return manifest, _bundle_state(directory, manifest)
    return manifest, _view_state(directory, manifest)


def load(directory: str | Path = ".", device: str | torch.device = "cpu"):
    """Rebuild the FP32 eval-mode NeMo EncDecRNNTBPEModel from a directory holding manifest.json, the
    weights file of its format and tokenizer/: the repository root (bundle) or compat/ (view)."""
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    from omegaconf import OmegaConf, open_dict

    directory = Path(directory).resolve()
    manifest, state = load_state(directory)
    cfg = OmegaConf.create(manifest["config"])
    if cfg.get("target") != MODEL_TARGET:
        raise ValueError(f"unexpected model target {cfg.get('target')!r}")
    with open_dict(cfg):
        cfg.tokenizer.dir = str(directory / TOKENIZER_DIR)
        for key, filename in TOKENIZER_FILES.items():
            cfg.tokenizer[key] = str(directory / TOKENIZER_DIR / filename)
        for ds in ("train_ds", "validation_ds", "test_ds"):  # stored for reference; no dataloaders at load
            if ds in cfg:
                cfg[ds] = None
    model = EncDecRNNTBPEModel(cfg=cfg)
    model.load_state_dict(state, strict=True)  # shape mismatches and missing/unexpected keys raise
    return model.to(device=device, dtype=torch.float32).eval()


def _pin_decoder_graphs(model) -> None:
    """On CUDA, put NeMo's greedy TDT decoder in the evaluation's CUDA-graph mode (no_while_loops),
    when this NeMo version exposes it and the decoder config allows CUDA graphs."""
    computer = getattr(getattr(getattr(model, "decoding", None), "decoding", None), "decoding_computer", None)
    if computer is None or not hasattr(computer, "force_cuda_graphs_mode") or not computer.allow_cuda_graphs:
        return
    mode = computer.cuda_graphs_mode
    if mode is None or str(mode.value) != DECODER_GRAPHS_MODE:
        computer.force_cuda_graphs_mode(DECODER_GRAPHS_MODE)


@contextlib.contextmanager
def _strict_fp32(device: torch.device):
    """Autocast off and TF32 off (matmul and cuDNN) for the duration; previous settings restored."""
    matmul = torch.backends.cuda.matmul.allow_tf32
    precision = torch.get_float32_matmul_precision()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        with torch.autocast(device.type, enabled=False), torch.backends.cudnn.flags(
                enabled=True, benchmark=torch.backends.cudnn.benchmark,
                deterministic=torch.backends.cudnn.deterministic, allow_tf32=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.set_float32_matmul_precision(precision)


@contextlib.contextmanager
def _inference_settings(model):
    """What NeMo's transcribe() sets (eval mode, dither 0, pad_to 0), restored on exit."""
    featurizer = getattr(getattr(model, "preprocessor", None), "featurizer", None)
    saved = {k: getattr(featurizer, k) for k in ("dither", "pad_to") if hasattr(featurizer, k)}
    training = model.training
    try:
        for key in saved:
            setattr(featurizer, key, 0.0 if key == "dither" else 0)
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for key, value in saved.items():
            setattr(featurizer, key, value)
        model.train(training)


def read_audio(path: str | Path) -> np.ndarray:
    """float32 mono samples of a 16 kHz audio file; multichannel files are averaged to mono."""
    import soundfile

    audio, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path}: {rate} Hz; convert with: ffmpeg -i {path} -ar 16000 -ac 1 out.wav")
    return audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]


def _as_clip(item) -> np.ndarray:
    if isinstance(item, (str, Path)):
        clip = read_audio(item)
    else:
        clip = item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else np.asarray(item)
        if clip.ndim != 1:
            raise ValueError(f"in-memory audio must be 1-D mono 16 kHz samples, got shape {clip.shape}")
        clip = clip.astype(np.float32, copy=False)
    if len(clip) < MIN_DECODE_SAMPLES:
        clip = np.pad(clip, (0, MIN_DECODE_SAMPLES - len(clip)))
    return clip


def transcribe(model, paths_or_audio, batch_size: int = 16, max_batch_seconds: float = MAX_BATCH_SECONDS):
    """Greedy TDT transcripts (punctuated, cased), decoded as the reported evaluation decoded.

    paths_or_audio: one item or a list of items, each a path to a 16 kHz audio file or a 1-D
    float array/tensor of 16 kHz mono samples in [-1, 1]. Returns a str for a single item and a
    list of str (in input order) for a list.

    Decoding: preprocessor -> encoder -> model.decoding.rnnt_decoder_predictions_tensor with the
    model's own decoding config (greedy_batch TDT, durations 0-4, max 10 symbols per frame) in
    strict FP32 (autocast off, TF32 off for matmuls and cuDNN), dither 0; on CUDA in NeMo's
    no_while_loops CUDA-graph mode, as evaluated. Inputs are batched longest first, zero-padded,
    at most batch_size items and max_batch_seconds of audio per batch.

    Long audio: each input is decoded in a single pass with full attention; nothing is chunked.
    Memory grows roughly quadratically with input length. The model was trained on clips of up to
    about 30 s; split long recordings (ideally at pauses) into segments of about 30 s or less.
    """
    single = not isinstance(paths_or_audio, (list, tuple))
    items = [paths_or_audio] if single else list(paths_or_audio)
    if not items:
        return []
    params = list(model.parameters())
    if any(p.dtype != torch.float32 for p in params):
        raise ValueError("decoding is specified in FP32; load() returns an FP32 model")
    device = params[0].device
    if device.type == "cuda":
        _pin_decoder_graphs(model)
    clips = [_as_clip(item) for item in items]
    order = sorted(range(len(clips)), key=lambda i: (-len(clips[i]), i))
    batches, current, seconds = [], [], 0.0
    for i in order:
        duration = len(clips[i]) / SAMPLE_RATE
        if current and (len(current) >= batch_size or seconds + duration > max_batch_seconds):
            batches.append(current)
            current, seconds = [], 0.0
        current.append(i)
        seconds += duration
    batches.append(current)
    texts: list[str | None] = [None] * len(clips)
    with _strict_fp32(device), _inference_settings(model):
        for idx in batches:
            lengths = torch.tensor([len(clips[i]) for i in idx], dtype=torch.long)
            audio = torch.zeros(len(idx), int(lengths.max()), dtype=torch.float32)
            for row, i in enumerate(idx):
                audio[row, :len(clips[i])] = torch.from_numpy(clips[i])
            enc, enc_len = model.forward(input_signal=audio.to(device), input_signal_length=lengths.to(device))
            hyps = model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=enc, encoded_lengths=enc_len, return_hypotheses=False)
            if isinstance(hyps, tuple):
                hyps = hyps[0]
            for i, hyp in zip(idx, hyps):
                texts[i] = hyp.text if hasattr(hyp, "text") else str(hyp)
    return texts[0] if single else texts


def main(argv: list[str]) -> None:
    compat = bool(argv) and argv[0] == "--compat"
    files = argv[1:] if compat else argv
    if not files or files[0] in ("-h", "--help"):
        sys.exit("usage: python load_kea_b.py [--compat] clip.wav [clip2.wav ...]   (16 kHz audio)")
    from nemo.utils import logging as nemo_logging
    nemo_logging.setLevel(nemo_logging.ERROR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    here = Path(__file__).resolve().parent
    model = load(here / "compat" if compat else here, device=device)
    for name, text in zip(files, transcribe(model, files)):
        print(text if len(files) == 1 else f"{name}\t{text}")


if __name__ == "__main__":
    main(sys.argv[1:])
