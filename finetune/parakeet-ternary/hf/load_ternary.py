# load_ternary.py: standalone loader for the ternary Parakeet-TDT-0.6B-v2 export.
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
"""Load and run the ternary Parakeet-TDT-0.6B-v2 export with only NeMo (nemo_toolkit[asr] and its
torch), safetensors and soundfile. Standalone copy of the repository's export.load_export, so the
model can be used without the training code.

    python load_ternary.py clip.wav [clip2.flac ...]    # 16 kHz audio; prints one transcript per file

    import sys; sys.path.insert(0, "parakeet-ternary")  # directory holding this repository's files
    from load_ternary import load, transcribe
    model = load("parakeet-ternary", device="cuda")      # or "cpu"
    print(transcribe(model, ["clip.wav"]))               # list in -> list of str out

Format "parakeet-ternary-v1" (one directory):
- export.safetensors: each of the 264 quantized encoder weights as "<name>.codes", 2-bit codes
  packed four per byte (00 = 0, 01 = +1, 10 = -1; code j of a row at bits 2*(j % 4) of byte
  j // 4; uint8 [out, ceil(in/4)]; pointwise convolutions as their [out, in] matrix) and
  "<name>.scale", one FP32 scale per output row; the weight is codes * scale. Every other
  state_dict tensor is FP16, except the feature-extraction constants
  preprocessor.featurizer.{window,fb} (FP32) and integer BatchNorm counters (int64).
- manifest.json: format, SHA-256 of export.safetensors and the three tokenizer files, the NeMo
  model config, per-layer shapes and kinds, code histogram, parameter and byte accounting,
  training provenance.
- tokenizer/: the base model's SentencePiece files (tokenizer.model, tokenizer.vocab, vocab.txt).

load() rebuilds a plain NeMo EncDecRNNTBPEModel in FP32 with the dequantized weights (accuracy,
not packed-kernel speed): it does not use packed ternary kernels and needs the same memory and
compute as the FP32 original. No speed or energy benefit is claimed.

Tested environment: NeMo 3.0.0, torch 2.11.0+cu128. The reported WERs were measured decoding on
an NVIDIA RTX 5090 (CUDA, strict FP32); loading on the CPU was tested, decoding on the CPU was
not. Other environments rebuild the same weights, but WER was only measured as described.
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
from safetensors.torch import load_file

FORMAT = "parakeet-ternary-v1"
MANIFEST = "manifest.json"
TOKENIZER_DIR = "tokenizer"
# config.tokenizer key -> file name in the tokenizer directory
TOKENIZER_FILES = {"model_path": "tokenizer.model", "vocab_path": "vocab.txt",
                   "spe_tokenizer_vocab": "tokenizer.vocab"}
MODEL_TARGET = "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel"
SAMPLE_RATE = 16000
# As in the evaluation: inputs shorter than 30 ms are zero-padded to 30 ms (NeMo's per-feature
# normalization fails on a one-frame input). Every reported test utterance is longer.
MIN_DECODE_SAMPLES = 480
MAX_BATCH_SECONDS = 1200.0
_SHIFTS = torch.tensor([0, 2, 4, 6], dtype=torch.uint8)


def unpack_codes(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """uint8 [out, ceil(in/4)] -> int8 [out, in] in {-1, 0, +1}; rejects the unused 11 pattern and nonzero tail padding."""
    if packed.dtype != torch.uint8 or packed.dim() != 2 or packed.shape[1] != math.ceil(in_features / 4):
        raise ValueError(f"expected uint8 [out, {math.ceil(in_features / 4)}], got {packed.dtype} {tuple(packed.shape)}")
    fields = ((packed[..., None] >> _SHIFTS.to(packed.device)) & 3).flatten(1)
    if (fields == 3).any() or fields[:, in_features:].any():
        raise ValueError("invalid packed codes")
    fields = fields[:, :in_features]
    return (fields == 1).to(torch.int8) - (fields == 2).to(torch.int8)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(directory: str | Path = ".", device: str | torch.device = "cpu"):
    """Rebuild the FP32 eval-mode NeMo EncDecRNNTBPEModel from a directory holding manifest.json,
    export.safetensors and tokenizer/.

    The SHA-256 of export.safetensors, tokenizer/tokenizer.model, tokenizer/vocab.txt and
    tokenizer/tokenizer.vocab is checked against manifest.json (which must list exactly these
    four files); manifest.json itself is not hash-checked.
    """
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    from omegaconf import OmegaConf, open_dict

    directory = Path(directory).resolve()
    manifest = json.loads((directory / MANIFEST).read_text())
    if manifest["format"] != FORMAT:
        raise ValueError(f"unknown export format {manifest['format']!r}")
    expected_files = {manifest["file"], *(f"{TOKENIZER_DIR}/{f}" for f in TOKENIZER_FILES.values())}
    if set(manifest["files"]) != expected_files or manifest["files"][manifest["file"]] != manifest["sha256"]:
        raise ValueError(f"manifest is inconsistent: files {sorted(manifest['files'])}")
    for rel, digest in manifest["files"].items():
        if _sha256(directory / rel) != digest:
            raise ValueError(f"{directory / rel} does not match the manifest SHA-256")
    tensors = load_file(directory / manifest["file"])
    state: dict[str, torch.Tensor] = {}
    for name, layer in manifest["quantized_layers"].items():
        if layer["kind"] not in ("linear", "pointwise_conv1d"):
            raise ValueError(f"{name}: unknown layer kind {layer['kind']!r}")
        out_features, in_features = layer["shape"]
        scale = tensors.pop(f"{name}.scale")
        if scale.dtype != torch.float32 or tuple(scale.shape) != (out_features,):
            raise ValueError(f"{name}.scale: expected float32 [{out_features}], got {scale.dtype} {tuple(scale.shape)}")
        weight = unpack_codes(tensors.pop(f"{name}.codes"), in_features).float() * scale.float()[:, None]
        if weight.shape != (out_features, in_features):
            raise ValueError(f"{name}: shape {tuple(weight.shape)} != {layer['shape']}")
        state[f"{name}.weight"] = weight.unsqueeze(2) if layer["kind"] == "pointwise_conv1d" else weight
        if layer["bias"]:
            state[f"{name}.bias"] = tensors.pop(f"{name}.bias")
    state.update((k, v.float() if v.is_floating_point() else v) for k, v in tensors.items())
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
    """Greedy TDT transcripts (punctuated, cased), decoded exactly as the reported evaluation.

    paths_or_audio: one item or a list of items, each a path to a 16 kHz audio file or a 1-D
    float array/tensor of 16 kHz mono samples in [-1, 1]. Returns a str for a single item and a
    list of str (in input order) for a list.

    Decoding: preprocessor -> encoder -> model.decoding.rnnt_decoder_predictions_tensor with the
    model's own decoding config (greedy_batch TDT, durations 0-4, max 10 symbols per frame) in
    strict FP32 (autocast off, TF32 off for matmuls and cuDNN), dither 0. Inputs are batched
    longest first, zero-padded, at most batch_size items and max_batch_seconds of audio per batch.

    Long audio: each input is decoded in a single pass with full attention; nothing is chunked.
    Memory grows roughly quadratically with input length. The model was trained on 1-30 s
    utterances and evaluated on test utterances of up to about 106 s; longer inputs were not
    evaluated, so split long recordings (ideally at pauses) into segments of about 30 s or less.
    """
    single = not isinstance(paths_or_audio, (list, tuple))
    items = [paths_or_audio] if single else list(paths_or_audio)
    if not items:
        return []
    params = list(model.parameters())
    if any(p.dtype != torch.float32 for p in params):
        raise ValueError("decoding is specified in FP32; load() returns an FP32 model")
    device = params[0].device
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
    if not argv or argv[0] in ("-h", "--help"):
        sys.exit("usage: python load_ternary.py clip.wav [clip2.wav ...]   (16 kHz audio)")
    from nemo.utils import logging as nemo_logging
    nemo_logging.setLevel(nemo_logging.ERROR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load(Path(__file__).resolve().parent, device=device)
    for name, text in zip(argv, transcribe(model, argv)):
        print(text if len(argv) == 1 else f"{name}\t{text}")


if __name__ == "__main__":
    main(sys.argv[1:])
