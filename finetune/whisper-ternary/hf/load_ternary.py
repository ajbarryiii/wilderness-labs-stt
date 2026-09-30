"""Load and run the ternary Whisper tiny.en export with only torch, transformers,
safetensors and soundfile. Standalone copy of the repository's export.load_export,
so the model can be used without the training code or its pinned runtime.

    python load_ternary.py clip.wav            # 16 kHz audio; prints the transcript

    from load_ternary import load
    model, processor = load(".")               # directory holding this repo's files

Format "whisper-ternary-v1": each quantized weight is stored as 2-bit codes, four per
byte (00 = 0, 01 = +1, 10 = -1; code j of a row at bits 2*(j % 4) of byte j // 4),
plus one FP32 scale per output row; the weight is codes * scale. Biases of quantized
layers are FP32; every other tensor is FP16. The model is rebuilt in FP32 with
dequantized weights, which reproduces the reported accuracy; it does not use packed
ternary kernels.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import (GenerationConfig, WhisperConfig, WhisperForConditionalGeneration,
                          WhisperProcessor)

FORMAT = "whisper-ternary-v1"
SAMPLE_RATE = 16000
_SHIFTS = torch.tensor([0, 2, 4, 6], dtype=torch.uint8)


def unpack_codes(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """uint8 [out, ceil(in/4)] -> int8 [out, in] in {-1, 0, +1}."""
    if (packed.dtype != torch.uint8 or packed.dim() != 2
            or packed.shape[1] != math.ceil(in_features / 4)):
        raise ValueError(f"unexpected packed tensor {packed.dtype} {tuple(packed.shape)}")
    fields = ((packed[..., None] >> _SHIFTS) & 3).flatten(1)
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


def load_model(directory: str | Path, device: str = "cpu") -> WhisperForConditionalGeneration:
    """Rebuild the FP32 model from export.safetensors + manifest.json (SHA-256 checked)."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["format"] != FORMAT:
        raise ValueError(f"unknown format {manifest['format']!r}")
    path = directory / manifest["file"]
    if _sha256(path) != manifest["sha256"]:
        raise ValueError(f"{path} does not match the manifest SHA-256")
    tensors = load_file(path)
    state: dict[str, torch.Tensor] = {}
    for name, layer in manifest["quantized_layers"].items():
        if name in manifest["tied"]:
            continue
        codes = unpack_codes(tensors.pop(f"{name}.codes"), layer["shape"][1])
        state[f"{name}.weight"] = codes.float() * tensors.pop(f"{name}.scale").float()[:, None]
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


def load(directory: str | Path, device: str = "cpu"):
    """(model, processor); the tokenizer and feature extractor ship in the same directory."""
    return load_model(directory, device), WhisperProcessor.from_pretrained(directory)


@torch.no_grad()
def transcribe(model, processor, audio, max_new_tokens: int = 225) -> str:
    """Greedy decoding of mono 16 kHz float audio in consecutive 30 s windows.

    Strict FP32 as evaluated: autocast off and TF32 off for CUDA matmuls and
    convolutions, whatever the caller's global settings.
    """
    texts = []
    device = next(model.parameters()).device
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.autocast(device.type, enabled=False), torch.backends.cudnn.flags(
                enabled=True, benchmark=False, deterministic=False, allow_tf32=False):
            for start in range(0, len(audio), 30 * SAMPLE_RATE):
                features = processor(audio[start:start + 30 * SAMPLE_RATE],
                                     sampling_rate=SAMPLE_RATE,
                                     return_tensors="pt").input_features.to(device)
                ids = model.generate(features, num_beams=1, do_sample=False,
                                     max_new_tokens=max_new_tokens)
                texts.append(processor.batch_decode(ids, skip_special_tokens=True)[0].strip())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    return " ".join(t for t in texts if t)


def main(argv: list[str]) -> None:
    import soundfile
    import transformers
    transformers.logging.set_verbosity_error()
    if not argv:
        sys.exit("usage: python load_ternary.py clip.wav [clip2.wav ...]")
    model, processor = load(Path(__file__).resolve().parent)
    for name in argv:
        audio, rate = soundfile.read(name, dtype="float32", always_2d=True)
        if rate != SAMPLE_RATE:
            sys.exit(f"{name}: {rate} Hz; convert with: ffmpeg -i {name} -ar 16000 -ac 1 out.wav")
        print(transcribe(model, processor, audio.mean(axis=1)))


if __name__ == "__main__":
    main(sys.argv[1:])
