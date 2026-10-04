"""In-loop teacher labeling for streamed training batches. See DESIGN.md "Supervision" and "Distillation variants".

Training audio is streamed, so teacher transcripts are produced inside the training loop:

    texts, keep, enc, enc_len = teacher_label_batch(teacher, audio, lengths, human_texts)

- teacher: the frozen pretrained model (evaluate.load_pretrained(device); FP32 weights,
  eval mode; call teacher.requires_grad_(False) once). It sees the clean, unaugmented input.
- audio [B, T] float32 16 kHz on the teacher's device, zero-padded; lengths [B] int64 samples;
  human_texts: one str or None per row (None or blank = no human transcript).
- Path: teacher.preprocessor (log-mel, FP32; NeMo disables autocast inside it) ->
  teacher.encoder -> teacher.decoding.rnnt_decoder_predictions_tensor (greedy TDT with the
  model's own decoding config, batched), all under torch.no_grad with dither 0 and pad_to 0.
- Precision: precision="bf16" (default) runs the encoder under BF16 autocast, as DESIGN.md
  specifies for the teacher in encoder-output matching; the decoding (prediction and joint
  networks) always runs in FP32 with autocast off, on the encoder output cast to FP32.
  precision="fp32" runs everything in FP32 with TF32 off, exactly evaluate.py's path.
- Returns texts (raw greedy hypotheses, punctuated and cased; "" where the teacher emitted
  nothing), keep (bool [B] on the device), encoder_out (FP32 [B, D=1024, T'] in NeMo's
  channels-first layout, T' = ceil(T_frames / 8)) and encoder_lengths ([B] int64), so the
  encoder-matching loss can reuse the teacher forward.
- keep applies the DESIGN.md filters: duration within [min_s, max_s] seconds (1 to 30 by
  default; pass None to skip), a non-empty hypothesis, and, where a human text is present,
  Whisper-normalized WER(teacher vs human) <= 0.5 (evaluate.utterance_wer). Pass a dict as
  `counts` to accumulate per-reason drop counts.

python teacher.py measures throughput on LibriSpeech train-clean-100 (under paths.gpu_lock)
and writes paths.EVAL/"teacher-throughput.json".
"""
from __future__ import annotations

import argparse
import contextlib
import json
import random
import time
from pathlib import Path

import torch
from torch import Tensor, nn

import evaluate
import paths

MIN_S, MAX_S, MAX_WER = 1.0, 30.0, 0.5
REASONS = ("duration_out_of_range", "empty_teacher_text", "teacher_human_wer_gt_0.5")


def drop_reason(duration_s: float, teacher_text: str, human_text: str | None,
                min_s: float | None = MIN_S, max_s: float | None = MAX_S) -> str | None:
    """The DESIGN.md filter that drops this utterance, or None to keep it."""
    if min_s is not None and duration_s < min_s or max_s is not None and duration_s > max_s:
        return "duration_out_of_range"
    if not teacher_text.strip():
        return "empty_teacher_text"
    if isinstance(human_text, str) and human_text.strip() and evaluate.utterance_wer(human_text, teacher_text) > MAX_WER:
        return "teacher_human_wer_gt_0.5"
    return None


@torch.no_grad()
def teacher_label_batch(teacher: nn.Module, audio_signal: Tensor, lengths: Tensor, human_texts: list[str | None],
                        precision: str = "bf16", min_s: float | None = MIN_S, max_s: float | None = MAX_S,
                        counts: dict | None = None) -> tuple[list[str], Tensor, Tensor, Tensor]:
    """(texts, keep [B] bool, encoder_out [B, D, T'] FP32, encoder_lengths [B]); see the module docstring."""
    if precision not in ("bf16", "fp32"):
        raise ValueError(f"precision must be 'bf16' or 'fp32', got {precision!r}")
    if audio_signal.dim() != 2 or len(human_texts) != audio_signal.shape[0] or lengths.shape != (audio_signal.shape[0],):
        raise ValueError("expected audio [B, T], lengths [B] and B human texts")
    device = audio_signal.device
    audio_signal = audio_signal.float()
    with evaluate.strict_fp32(device), evaluate.inference_settings(teacher):
        features, feature_len = teacher.preprocessor(input_signal=audio_signal, length=lengths)
        autocast = (torch.autocast(device.type, dtype=torch.bfloat16) if precision == "bf16"
                    else contextlib.nullcontext())
        with autocast:
            encoded, encoded_len = teacher.encoder(audio_signal=features, length=feature_len)
        encoded = encoded.float()
        hyps = teacher.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=False)
    if isinstance(hyps, tuple):
        hyps = hyps[0]
    texts = [h.text if hasattr(h, "text") else str(h) for h in hyps]
    durations = (lengths.float() / paths.SAMPLE_RATE).tolist()
    reasons = [drop_reason(d, t, h, min_s, max_s) for d, t, h in zip(durations, texts, human_texts)]
    if counts is not None:
        for reason in reasons:
            key = reason or "kept"
            counts[key] = counts.get(key, 0) + 1
    keep = torch.tensor([r is None for r in reasons], dtype=torch.bool, device=device)
    return texts, keep, encoded, encoded_len


# --- throughput measurement ----------------------------------------------------------------------

LIBRISPEECH_TC100 = Path("/mnt/hd/wilderness-labs-stt/stt-distillation/datasets/libri/LibriSpeech/train-clean-100")


def sample_librispeech(n: int, seed: int = paths.SEED) -> list[dict]:
    """A seeded sample of n train-clean-100 utterances {"audio_filepath","duration","text","id"}."""
    import soundfile as sf

    trans = {}
    for path in sorted(LIBRISPEECH_TC100.glob("*/*/*.trans.txt")):
        for line in path.read_text().splitlines():
            uid, text = line.split(" ", 1)
            trans[uid] = (path.parent / f"{uid}.flac", text)
    ids = sorted(trans)
    random.Random(seed).shuffle(ids)
    out = []
    for uid in ids[:n]:
        flac, text = trans[uid]
        out.append({"audio_filepath": str(flac), "duration": sf.info(flac).frames / paths.SAMPLE_RATE,
                    "text": text, "id": uid})
    return out


def stream_batches(records: list[dict], batch_seconds: float) -> list[list[int]]:
    """Batches in sampled (shuffled, not duration-sorted) order up to batch_seconds of audio, like a streamed loader."""
    batches, current, seconds = [], [], 0.0
    for i, r in enumerate(records):
        if current and seconds + r["duration"] > batch_seconds:
            batches.append(current)
            current, seconds = [], 0.0
        current.append(i)
        seconds += r["duration"]
    if current:
        batches.append(current)
    return batches


def measure(teacher: nn.Module, records: list[dict], audio: list[Tensor], batch_seconds: float, precision: str,
            device: torch.device) -> tuple[dict, dict[str, str]]:
    batches = stream_batches(records, batch_seconds)
    padded = []
    for idx in batches:
        lengths = torch.tensor([len(audio[i]) for i in idx])
        x = torch.zeros(len(idx), int(lengths.max()))
        for row, i in enumerate(idx):
            x[row, :len(audio[i])] = audio[i]
        padded.append((idx, x.pin_memory(), lengths))
    # warm-up on the first batch, untimed
    idx, x, lengths = padded[0]
    teacher_label_batch(teacher, x.to(device), lengths.to(device), [records[i]["text"] for i in idx], precision)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    counts: dict = {}
    texts: dict[str, str] = {}
    start = time.perf_counter()
    for idx, x, lengths in padded:
        out, keep, enc, enc_len = teacher_label_batch(teacher, x.to(device, non_blocking=True), lengths.to(device),
                                                      [records[i]["text"] for i in idx], precision, counts=counts)
        texts.update((records[i]["id"], t) for i, t in zip(idx, out))
    torch.cuda.synchronize(device)
    wall = time.perf_counter() - start
    hours = sum(r["duration"] for r in records) / 3600
    return {"precision": precision, "batch_audio_seconds": batch_seconds, "batches": len(batches),
            "mean_batch_size": len(records) / len(batches), "utterances": len(records), "audio_hours": hours,
            "wall_s": wall, "audio_hours_per_gpu_hour": hours / (wall / 3600),
            "peak_gpu_memory_gb": torch.cuda.max_memory_allocated(device) / 2**30, "filter_counts": counts}, texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure teacher_label_batch throughput on train-clean-100.")
    parser.add_argument("--utterances", type=int, default=2000)
    parser.add_argument("--batch-seconds", type=float, nargs="+", default=[300.0, 600.0, 1200.0])
    parser.add_argument("--precisions", nargs="+", default=["bf16", "fp32"])
    parser.add_argument("--out", type=Path, default=paths.EVAL / "teacher-throughput.json")
    args = parser.parse_args()
    paths.require_mount()
    device = torch.device("cuda")
    records = sample_librispeech(args.utterances)
    audio = [torch.from_numpy(evaluate.load_audio(r["audio_filepath"])) for r in records]
    results = []
    labels: dict[str, dict[str, str]] = {}
    with paths.gpu_lock("teacher-throughput"):
        teacher = evaluate.load_pretrained(device).requires_grad_(False)
        for precision in args.precisions:
            for seconds in args.batch_seconds:
                result, texts = measure(teacher, records, audio, seconds, precision, device)
                labels[f"{precision}-{seconds:g}"] = texts
                results.append(result)
                print(f"{precision} {seconds:g} s/batch: {result['audio_hours_per_gpu_hour']:.0f} audio h per GPU h, "
                      f"peak {result['peak_gpu_memory_gb']:.1f} GB, mean batch {result['mean_batch_size']:.1f}, "
                      f"filters {result['filter_counts']}", flush=True)
    agreement = {}
    if "fp32" in args.precisions and "bf16" in args.precisions:
        for seconds in args.batch_seconds:
            fp32, bf16 = labels[f"fp32-{seconds:g}"], labels[f"bf16-{seconds:g}"]
            records_cmp = [{"ref_norm": evaluate.normalize(fp32[k]), "hyp_norm": evaluate.normalize(bf16[k])}
                           for k in fp32]
            agreement[f"{seconds:g}"] = {
                "identical_raw_text": sum(fp32[k] == bf16[k] for k in fp32) / len(fp32),
                "normalized_wer_bf16_vs_fp32": evaluate.corpus_wer(records_cmp)["wer"]}
        print(f"bf16 vs fp32 labels: {agreement}", flush=True)
    human = [{"ref_norm": evaluate.normalize(r["text"]),
              "hyp_norm": evaluate.normalize(labels[f"fp32-{args.batch_seconds[0]:g}"][r["id"]])}
             for r in records] if "fp32" in args.precisions else []
    out = {"source": "LibriSpeech train-clean-100, seeded sample", "seed": paths.SEED, "utterances": len(records),
           "audio_hours": sum(r["duration"] for r in records) / 3600,
           "timing": "teacher_label_batch only (pinned CPU batch -> GPU, preprocessor, encoder, greedy TDT, "
                     "filters); audio decoding excluded; one untimed warm-up batch",
           "results": results, "bf16_vs_fp32_label_agreement": agreement,
           "fp32_teacher_vs_human_wer": evaluate.corpus_wer(human)["wer"] if human else None,
           "gpu": torch.cuda.get_device_name(device)}
    evaluate.write_json(args.out, out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
