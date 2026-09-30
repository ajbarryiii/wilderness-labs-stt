"""Transcribe a 16 kHz audio file with the ternary Whisper tiny.en export.

Loads an export directory (export.safetensors + manifest.json, SHA-256 checked),
rebuilds the FP32 model from its ternary codes and scales, and decodes exactly
as the reported evaluation does: greedy, the shipped generation config, at most
225 new tokens per 30 s window, FP32 with TF32 off. Audio longer than 30 s is cut
into consecutive 30 s windows whose transcripts are joined; words split at a
window boundary can be misrecognized. --compare-fp32 also runs the pretrained
FP32 tiny.en on the same audio. --duration-cap applies the secondary
ceil(4.5 x seconds) + 5 word cap from DESIGN.md "Secondary analyses"; it is off
by default so output matches the primary reported metric.

The model runs through dequantized FP32 weights: this demonstrates accuracy and
file size, not packed-kernel speed.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile
import torch
import transformers
from transformers import WhisperForConditionalGeneration, WhisperProcessor

import checkpoint
import export
import paths

DEFAULT_EXPORT = paths.RUNS / "v2-ternary-embed-lr1e-3" / "export"
WINDOW_S = 30
MAX_NEW_TOKENS = 225
CAP_WORDS_PER_S, CAP_EXTRA_WORDS = 4.5, 5
PROCESSOR_FILES = ("preprocessor_config.json", "tokenizer_config.json")


def read_audio(path: Path) -> np.ndarray:
    """Mono float32 samples at 16 kHz; channels are averaged, other rates are refused."""
    audio, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    if rate != paths.SAMPLE_RATE:
        raise ValueError(f"{path}: {rate} Hz; convert first, e.g. "
                         f"ffmpeg -i {path} -ar 16000 -ac 1 out.wav")
    if audio.shape[0] == 0:
        raise ValueError(f"{path}: no samples")
    return audio.mean(axis=1)


def windows(audio: np.ndarray, seconds: int = WINDOW_S) -> list[np.ndarray]:
    """Consecutive non-overlapping windows; the last one may be shorter."""
    step = seconds * paths.SAMPLE_RATE
    return [audio[i:i + step] for i in range(0, len(audio), step)]


def cap_words(text: str, seconds: float) -> str:
    """Keep the first ceil(4.5 x seconds) + 5 printed words.

    Same formula as the DESIGN.md secondary duration cap, but counted on the
    printed text; analysis.py counts Whisper-normalized words, which can differ
    slightly (e.g. "don't" normalizes to two words).
    """
    limit = math.ceil(CAP_WORDS_PER_S * seconds) + CAP_EXTRA_WORDS
    words = text.split()
    return text.strip() if len(words) <= limit else " ".join(words[:limit])


def processor_dir(export_dir: Path) -> Path:
    """Tokenizer/feature-extractor files shipped with the export, else the pinned checkpoint.

    An export counts as shipping a processor when it has the feature-extractor
    config, the tokenizer config, and either a fast tokenizer (tokenizer.json) or
    a slow one (vocab.json + merges.txt). The fallback verifies every file in the
    pinned checkpoint's lock.json, since the fast tokenizer reads several of them.
    """
    has = lambda name: (export_dir / name).is_file()  # noqa: E731
    if all(has(n) for n in PROCESSOR_FILES) and (
            has("tokenizer.json") or (has("vocab.json") and has("merges.txt"))):
        return export_dir
    checkpoint.verify_lock()
    return paths.MODEL_DIR


@torch.no_grad()
def transcribe(model: WhisperForConditionalGeneration, processor: WhisperProcessor,
               audio: np.ndarray, device: torch.device, duration_cap: bool = False) -> str:
    texts = []
    with torch.autocast(device.type, enabled=False), torch.backends.cudnn.flags(
            enabled=True, benchmark=False, deterministic=False, allow_tf32=False):
        for chunk in windows(audio):
            features = processor(chunk, sampling_rate=paths.SAMPLE_RATE,
                                 return_tensors="pt").input_features.to(device)
            ids = model.generate(features, num_beams=1, do_sample=False,
                                 max_new_tokens=MAX_NEW_TOKENS)
            text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            if duration_cap:
                text = cap_words(text, len(chunk) / paths.SAMPLE_RATE)
            texts.append(text)
    return " ".join(t for t in texts if t)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, nargs="+", help="16 kHz audio files (wav, flac, ...)")
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT,
                        help="export directory (default: the v2 ternary projections + embedding run)")
    parser.add_argument("--compare-fp32", action="store_true",
                        help="also transcribe with the pretrained FP32 tiny.en")
    parser.add_argument("--duration-cap", action="store_true",
                        help="apply the secondary words-per-second cap (off matches the reported WER)")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # The shipped English-only generation config triggers known, harmless notices about
    # suppress_tokens and forced_decoder_ids on every call (see README "Experiment").
    transformers.logging.set_verbosity_error()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    processor = WhisperProcessor.from_pretrained(processor_dir(args.export), local_files_only=True)
    models = {"ternary": export.load_export(args.export, device=device)}
    size = (args.export / "export.safetensors").stat().st_size
    print(f"# ternary export: {args.export} ({size / 1e6:.1f} MB)", file=sys.stderr)
    if args.compare_fp32:
        models["fp32"] = checkpoint.load_model(device)
        print(f"# fp32 reference: {paths.MODEL_DIR} "
              f"({(paths.MODEL_DIR / 'model.safetensors').stat().st_size / 1e6:.1f} MB)",
              file=sys.stderr)
    for path in args.audio:
        audio = read_audio(path)
        seconds = len(audio) / paths.SAMPLE_RATE
        for name, model in models.items():
            start = time.perf_counter()
            text = transcribe(model, processor, audio, device, args.duration_cap)
            elapsed = time.perf_counter() - start
            label = f"[{name}] " if len(models) > 1 else ""
            print(f"{label}{text}")
            print(f"# {path.name}: {seconds:.1f} s audio, {name} {elapsed:.2f} s on {device}",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
