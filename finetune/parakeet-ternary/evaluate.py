"""Greedy TDT decoding and Whisper-normalized corpus WER, identical for every arm. See DESIGN.md "Evaluation".

Decoding: the batched path NeMo's model.transcribe() runs (preprocessor -> encoder ->
model.decoding.rnnt_decoder_predictions_tensor with the model's own decoding config,
greedy_batch TDT), with transcribe()'s inference settings (eval mode, dither 0, pad_to 0),
in strict FP32: autocast off, TF32 off for matmuls and cuDNN. It is called directly rather
than through transcribe() because transcribe() unfreezes the encoder, prediction and joint
networks when it finishes, which would silently change a trainer's frozen modules; the
caller's training mode and preprocessor settings are restored afterwards. Audio is read
from the manifest's 16 kHz FLAC files; utterances are batched in descending duration
(at most batch_size utterances and max_batch_seconds of audio per batch).

Normalizer: transformers' Whisper EnglishTextNormalizer
(transformers.models.whisper.english_normalizer) with the English spelling mapping from
Whisper's normalizer.json (NORMALIZER_JSON, SHA-256 pinned). This is the normalizer of the
Open ASR Leaderboard (huggingface/open_asr_leaderboard normalizer/normalizer.py, a copy of
Whisper's, as of commit 431dd91, the version current when NVIDIA published this model's
numbers in 2025; the leaderboard added name, acronym and compound-word rules in 2026-04).
As in the leaderboard (normalizer/data_utils.py is_target_text_in_range), utterances whose
normalized reference is empty or "ignore time segment in scoring" are excluded from the
WER; they are still decoded and kept in the records with "scored": false.

WER: corpus-level (S + D + I) / reference words with the Levenshtein edit counts of
finetune/whisper-ternary/wer.py (imported, not copied).

CLI: python evaluate.py --source {pretrained,export,checkpoint} [--path P] --sets test|dev|NAME...
     [--limit N] [--batch-size B] --out-dir DIR
Writes DIR/<set>.json per set and DIR/summary.json.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn

import paths
import quant

NORMALIZER_JSON = paths.REPO / "finetune" / "stt" / "models" / "whisper-tiny.en" / "normalizer.json"
NORMALIZER_SHA256 = "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd"
NORMALIZER = ("transformers.models.whisper.english_normalizer.EnglishTextNormalizer with Whisper "
              "normalizer.json spelling mapping (= Open ASR Leaderboard normalizer at commit 431dd91)")
EXCLUDED_REFERENCES = {"", "ignore time segment in scoring"}
REPRODUCTION_GATE = 0.2  # absolute WER points per set, DESIGN.md "Reproduction gate"
MAX_BATCH_SECONDS = 1200.0
# Utterances shorter than this are zero-padded to exactly this length before feature extraction
# (identically for every source and arm): NeMo's per_feature normalization fails on a
# one-frame input (ami_dev has a 0.020 s utterance). Every test-set utterance is >= 0.040 s.
MIN_DECODE_SECONDS = 0.03
MIN_DECODE_SAMPLES = round(MIN_DECODE_SECONDS * paths.SAMPLE_RATE)  # 480

_wer = quant.load_whisper_module("wer")
edit_counts = _wer.edit_counts
corpus_wer = _wer.corpus_wer


class Normalizer:
    def __init__(self, spelling_json: Path = NORMALIZER_JSON) -> None:
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

        data = Path(spelling_json).read_bytes()
        if hashlib.sha256(data).hexdigest() != NORMALIZER_SHA256:
            raise ValueError(f"{spelling_json} is not the pinned Whisper normalizer.json")
        self._normalize = EnglishTextNormalizer(json.loads(data))

    def __call__(self, text: str) -> str:
        return self._normalize(text)


_NORMALIZER: Normalizer | None = None


def normalize(text: str) -> str:
    global _NORMALIZER
    if _NORMALIZER is None:
        _NORMALIZER = Normalizer()
    return _NORMALIZER(text)


def score(ref: str, hyp: str) -> dict:
    """Normalized texts and edit counts of one utterance; 'scored' False for leaderboard-excluded references."""
    ref_norm, hyp_norm = normalize(ref), normalize(hyp)
    s, d, i = edit_counts(ref_norm.split(), hyp_norm.split())
    return {"ref_norm": ref_norm, "hyp_norm": hyp_norm, "S": s, "D": d, "I": i,
            "scored": ref_norm.strip() not in EXCLUDED_REFERENCES}


def utterance_wer(ref: str, hyp: str) -> float:
    """Normalized WER of one pair; 0 if both normalize to nothing, inf if only the reference does."""
    r = score(ref, hyp)
    words = len(r["ref_norm"].split())
    if not words:
        return 0.0 if not r["hyp_norm"].split() else math.inf
    return (r["S"] + r["D"] + r["I"]) / words


# --- decoding ------------------------------------------------------------------------------------

@contextlib.contextmanager
def strict_fp32(device: torch.device | str):
    """Autocast off and TF32 off (matmul and cuDNN) for the duration; previous settings restored."""
    device = torch.device(device)
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
def inference_settings(model: nn.Module):
    """What transcribe() sets: eval mode, dither 0, pad_to 0; restored on exit (no freeze/unfreeze)."""
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


def load_audio(path: str | Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != paths.SAMPLE_RATE or audio.shape[1] != 1:
        raise ValueError(f"{path}: expected 16 kHz mono, got {rate} Hz x {audio.shape[1]}")
    return audio[:, 0]


def make_batches(records: list[dict], batch_size: int, max_batch_seconds: float = MAX_BATCH_SECONDS) -> list[list[int]]:
    """Indices into records, longest first, at most batch_size utterances and max_batch_seconds of audio each."""
    order = sorted(range(len(records)), key=lambda i: (-records[i]["duration"], records[i]["id"]))
    batches, current, seconds = [], [], 0.0
    for i in order:
        d = records[i]["duration"]
        if current and (len(current) >= batch_size or seconds + d > max_batch_seconds):
            batches.append(current)
            current, seconds = [], 0.0
        current.append(i)
        seconds += d
    if current:
        batches.append(current)
    return batches


def pad_short(clip: np.ndarray) -> tuple[np.ndarray, bool]:
    """(clip zero-padded at the end to MIN_DECODE_SAMPLES if shorter, whether it was padded)."""
    if len(clip) >= MIN_DECODE_SAMPLES:
        return clip, False
    return np.pad(clip, (0, MIN_DECODE_SAMPLES - len(clip))), True


class _BatchSet(torch.utils.data.Dataset):
    def __init__(self, records: list[dict], batches: list[list[int]]) -> None:
        self.records, self.batches = records, batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, k: int):
        idx = self.batches[k]
        clips, padded = zip(*(pad_short(load_audio(self.records[i]["audio_filepath"])) for i in idx))
        lengths = torch.tensor([len(c) for c in clips], dtype=torch.long)
        audio = torch.zeros(len(clips), int(lengths.max()), dtype=torch.float32)
        for row, clip in enumerate(clips):
            audio[row, :len(clip)] = torch.from_numpy(clip)
        return idx, audio, lengths, [i for i, p in zip(idx, padded) if p]


def audio_batch(records: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """(audio [B, T] zero-padded float32, lengths [B]) for a small fixed batch (reconstruction checks)."""
    _, audio, lengths, _ = _BatchSet(records, [list(range(len(records)))])[0]
    return audio, lengths


def transcribe_records(model: nn.Module, records: list[dict], batch_size: int, device: torch.device | str,
                       max_batch_seconds: float = MAX_BATCH_SECONDS, workers: int = 2,
                       stats: dict | None = None) -> list[str]:
    """Raw greedy transcripts (punctuated, cased) in record order. Model must be FP32 on device.

    Utterances shorter than MIN_DECODE_SECONDS are zero-padded to it (pad_short); their ids are
    reported in stats["padded_short_ids"] and their count in stats["padded_short"].
    """
    device = torch.device(device)
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("decoding is specified in FP32")
    batches = make_batches(records, batch_size, max_batch_seconds)
    loader = torch.utils.data.DataLoader(_BatchSet(records, batches), batch_size=None, shuffle=False,
                                         num_workers=workers, pin_memory=device.type == "cuda",
                                         persistent_workers=False)
    texts: list[str | None] = [None] * len(records)
    padded_short: list[int] = []
    start = time.perf_counter()
    with strict_fp32(device), inference_settings(model):
        for idx, audio, lengths, padded in loader:
            padded_short += padded
            enc, enc_len = model.forward(input_signal=audio.to(device, non_blocking=True),
                                         input_signal_length=lengths.to(device, non_blocking=True))
            hyps = model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=enc, encoded_lengths=enc_len, return_hypotheses=False)
            if isinstance(hyps, tuple):
                hyps = hyps[0]
            for i, hyp in zip(idx, hyps):
                texts[i] = hyp.text if hasattr(hyp, "text") else str(hyp)
            del enc, enc_len
    if stats is not None:
        stats.update(decode_wall_s=time.perf_counter() - start, batches=len(batches),
                     padded_short=len(padded_short),
                     padded_short_ids=sorted(records[i]["id"] for i in padded_short))
    assert all(t is not None for t in texts)
    return texts


def decode(model: nn.Module, manifest_records: list[dict], batch_size: int, device: torch.device | str,
           max_batch_seconds: float = MAX_BATCH_SECONDS, workers: int = 2, stats: dict | None = None) -> list[dict]:
    """Per-utterance records {"id","ref","hyp","ref_norm","hyp_norm","S","D","I","duration","scored"} in manifest order."""
    hyps = transcribe_records(model, manifest_records, batch_size, device, max_batch_seconds, workers, stats)
    out = []
    for record, hyp in zip(manifest_records, hyps):
        ref = record["text"]
        out.append({"id": record["id"], "ref": ref, "hyp": hyp, **score(ref, hyp), "duration": record["duration"]})
    return out


def wer_summary(records: list[dict]) -> dict:
    """Corpus WER over scored records plus exclusion count (whisper-ternary wer.corpus_wer)."""
    scored = [r for r in records if r.get("scored", True)]
    result = corpus_wer(scored)
    result["excluded_empty_reference"] = len(records) - len(scored)
    return result


def decoding_info(model: nn.Module, batch_size: int, max_batch_seconds: float) -> dict:
    from omegaconf import OmegaConf

    return {"method": "preprocessor -> encoder -> decoding.rnnt_decoder_predictions_tensor (transcribe() path)",
            "config": OmegaConf.to_container(model.cfg.decoding, resolve=True),
            "precision": "float32, TF32 off (matmul and cuDNN), autocast disabled",
            "batch_size": batch_size, "max_batch_seconds": max_batch_seconds,
            "batch_order": "descending duration"}


# --- models and manifests ------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_pretrained(device: torch.device | str = "cpu"):
    """The pinned FP32 Parakeet-TDT-0.6B-v2 (paths.MODEL_FILE) in eval mode."""
    import logging as pylogging

    from nemo.collections.asr.models import ASRModel
    from nemo.utils import logging

    paths.require_mount()
    level = logging.get_verbosity()
    logging.set_verbosity(pylogging.ERROR)
    try:
        model = ASRModel.restore_from(str(paths.MODEL_FILE), map_location="cpu")
    finally:
        logging.set_verbosity(level)
    return model.to(device=device, dtype=torch.float32).eval()


def base_lock() -> dict:
    return json.loads((paths.MODEL_DIR / "lock.json").read_text())


def load_checkpoint(path: Path, device: torch.device | str, kind: str):
    """A train.py checkpoint: torch.save dict with the model state_dict under "model" (or a bare state_dict).

    kind "ternary": pretrained architecture + quantize_parakeet, strict load, fraction set to 1
    (refuses a checkpoint whose recorded "weight_fraction" is below 1). kind "float": plain FP32 load.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    fraction = ckpt.get("weight_fraction", 1.0) if isinstance(ckpt, dict) else 1.0
    model = load_pretrained("cpu")
    if kind == "ternary":
        if fraction < 1.0:
            raise ValueError(f"{path}: weight_fraction {fraction} < 1 is not a deployable model")
        quant.quantize_parakeet(model)
        quant.set_weight_fraction(model, 1.0)
    model.load_state_dict(state, strict=True)
    return model.to(device=device, dtype=torch.float32).eval()


def resolve_manifest(name: str) -> tuple[str, Path]:
    """(set name, manifest path): test sets -> MANIFESTS/test_<name>.jsonl, else MANIFESTS/<name>.jsonl or a path."""
    if name in paths.TEST_SETS:
        return name, paths.MANIFESTS / f"test_{name}.jsonl"
    if name.endswith(".jsonl"):
        path = Path(name)
        return path.stem, path
    return name, paths.MANIFESTS / f"{name}.jsonl"


def read_manifest(path: Path) -> list[dict]:
    with open(path) as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return sorted(records, key=lambda r: r["id"])


def describe_source(kind: str, path: Path | None, description: str, checkpoint_kind: str | None = None) -> dict:
    lock = base_lock()
    if kind == "pretrained":
        weights = paths.MODEL_FILE
        digest = sha256_file(weights)
        if digest != lock["files"][weights.name]:
            raise ValueError(f"{weights} does not match {paths.MODEL_DIR / 'lock.json'}")
    elif kind == "export":
        weights = Path(path) / "export.safetensors"
        digest = sha256_file(weights)
    else:
        weights = Path(path)
        digest = sha256_file(weights)
    source = {"kind": kind, "description": description, "path": str(path or paths.MODEL_FILE),
              "weights_file": str(weights), "sha256": digest,
              "base_model": {"id": paths.MODEL_ID, "revision": paths.MODEL_REVISION,
                             "nemo_sha256": lock["files"][paths.MODEL_FILE.name]}}
    if checkpoint_kind:
        source["checkpoint_kind"] = checkpoint_kind
    return source


def evaluate_set(model: nn.Module, name: str, manifest: Path, source: dict, batch_size: int,
                 limit: int | None, device: torch.device, max_batch_seconds: float = MAX_BATCH_SECONDS) -> dict:
    records = read_manifest(manifest)[:limit]
    stats: dict = {}
    start = time.perf_counter()
    out = decode(model, records, batch_size, device, max_batch_seconds, stats=stats)
    wall = time.perf_counter() - start
    hours = sum(r["duration"] for r in out) / 3600
    result = {"source": source, "set": name, "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
              "utterances": len(out), "limit": limit, "audio_hours": hours,
              "decoding": decoding_info(model, batch_size, max_batch_seconds), "normalizer": NORMALIZER,
              "wer": wer_summary(out), "wall_s": wall,
              "padded_short": stats["padded_short"], "padded_short_ids": stats["padded_short_ids"],
              "min_decode_seconds": MIN_DECODE_SECONDS, "rtfx": hours * 3600 / wall,
              "peak_gpu_memory_gb": (torch.cuda.max_memory_allocated(device) / 2**30
                                     if device.type == "cuda" else None),
              "records": out}
    if name in paths.TEST_SETS:
        meta = manifest.with_suffix(".meta.json")
        if meta.exists():
            m = json.loads(meta.read_text())
            result["dataset"] = {"repo": m["repo"], "revision": m["revision"], "config": m["config"],
                                 "split": m["split"]}
    return result


def summarize(results: dict[str, dict], source: dict) -> dict:
    per_set = {}
    for name, r in results.items():
        entry = {"wer": 100 * r["wer"]["wer"], "utterances": r["utterances"],
                 "scored_utterances": r["wer"]["utterances"],
                 "excluded_empty_reference": r["wer"]["excluded_empty_reference"],
                 "padded_short": r.get("padded_short", 0),
                 "audio_hours": r["audio_hours"], "wall_s": r["wall_s"], "limit": r["limit"]}
        if source["kind"] == "pretrained" and name in paths.PUBLISHED_WER:
            published = paths.PUBLISHED_WER[name]
            entry["published_wer"] = published
            entry["diff_vs_published"] = entry["wer"] - published
            # Only a full set can pass or fail the gate; a --limit subset gets None.
            entry["reproduction_gate_pass"] = (abs(entry["wer"] - published) <= REPRODUCTION_GATE
                                               if r["limit"] is None else None)
        per_set[name] = entry
    summary = {"source": source, "sets": per_set, "wall_s": sum(r["wall_s"] for r in results.values())}
    if all(n in per_set and per_set[n]["limit"] is None for n in paths.MEAN_SETS):
        summary["mean_wer"] = sum(per_set[n]["wer"] for n in paths.MEAN_SETS) / len(paths.MEAN_SETS)
        summary["mean_sets"] = paths.MEAN_SETS
        summary["mean_note"] = "unweighted mean over DESIGN.md MEAN_SETS; TED-LIUM is not in the bundle"
        if source["kind"] == "pretrained":
            summary["published_mean_same_sets"] = (sum(paths.PUBLISHED_WER[n] for n in paths.MEAN_SETS)
                                                   / len(paths.MEAN_SETS))
    if all(n in per_set and per_set[n]["limit"] is None for n in paths.DEV_SETS):
        summary["dev_mean_wer"] = sum(per_set[n]["wer"] for n in paths.DEV_SETS) / len(paths.DEV_SETS)
        summary["dev_mean_sets"] = list(paths.DEV_SETS)
        summary["dev_mean_note"] = "unweighted mean over paths.DEV_SETS (DESIGN.md selection metric)"
    summary["padded_short"] = sum(e["padded_short"] for e in per_set.values())
    if source["kind"] == "pretrained":
        gates = [e["reproduction_gate_pass"] for n, e in per_set.items() if n in paths.MEAN_SETS]
        summary["reproduction_gate"] = {"threshold_points": REPRODUCTION_GATE,
                                        "all_pass": bool(gates) and all(g is True for g in gates),
                                        "complete": all(n in per_set and per_set[n]["limit"] is None
                                                        for n in paths.MEAN_SETS)}
    return summary


def write_json(path: Path, data) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("pretrained", "export", "checkpoint"), required=True)
    parser.add_argument("--path", type=Path, help="export directory or checkpoint file")
    parser.add_argument("--checkpoint-kind", choices=("ternary", "float"), default="ternary")
    parser.add_argument("--sets", nargs="+", required=True, help="'test', 'dev', or set names / manifest paths")
    parser.add_argument("--limit", type=int, help="first N id-sorted utterances per set")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batch-seconds", type=float, default=MAX_BATCH_SECONDS)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ptq", action="store_true",
                        help="with --source pretrained: B1 ternary PTQ (no training); quantize, export to "
                             "RUNS/b1-ptq/export, run the reconstruction check, and score the rebuilt export")
    args = parser.parse_args(argv)
    if (args.source == "pretrained") != (args.path is None):
        parser.error("--path is required for export and checkpoint, and not accepted for pretrained")
    if args.ptq and args.source != "pretrained":
        parser.error("--ptq applies only to --source pretrained")
    return args


REFERENCE_SET, REFERENCE_POSITIONS = "librispeech_clean", (0, 1000, 2000, 2618)


def reference_batch() -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """The fixed reconstruction-check batch: four id-sorted LibriSpeech test-clean utterances with references."""
    records = read_manifest(resolve_manifest(REFERENCE_SET)[1])
    picked = [records[i] for i in REFERENCE_POSITIONS]
    return (*audio_batch(picked), [r["text"] for r in picked])


def ptq_export(device: torch.device):
    """B1: quantize the pretrained model (no training), export, check, return (rebuilt model, export dir, check)."""
    import export

    model = load_pretrained(device)
    quant.quantize_parakeet(model)
    out_dir = paths.run_dir("b1-ptq") / "export"
    export.export_model(model, out_dir, extra={
        "run_name": "b1-ptq", "arm": "B1",
        "notes": "ternary post-training quantization of the pretrained checkpoint; no training"})
    rebuilt = export.load_export(out_dir, device)
    check = export.reconstruction_check(model.eval(), rebuilt, reference_batch())
    check["reference_batch"] = {"set": REFERENCE_SET, "positions": list(REFERENCE_POSITIONS)}
    write_json(out_dir.parent / "reconstruction.json", check)
    del model
    return rebuilt, out_dir, check


def expand_sets(names: list[str]) -> list[str]:
    out = []
    for name in names:
        out += list(paths.TEST_SETS) if name == "test" else list(paths.DEV_SETS) if name == "dev" else [name]
    return list(dict.fromkeys(out))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths.require_mount()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA is required")
    sets = [resolve_manifest(n) for n in expand_sets(args.sets)]
    missing = [str(p) for _, p in sets if not p.exists()]
    if missing:
        sys.exit(f"missing manifests: {missing}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with paths.gpu_lock(f"evaluate:{args.source}"):
        if args.ptq:
            model, export_dir, check = ptq_export(device)
            print(f"reconstruction: { {k: v for k, v in check.items() if 'hyps' not in k} }", flush=True)
            source = describe_source("export", export_dir, "B1 ternary PTQ of the pretrained model, rebuilt from export")
        elif args.source == "pretrained":
            model = load_pretrained(device)
            source = describe_source("pretrained", None, "pretrained Parakeet-TDT-0.6B-v2, FP32, no training")
        elif args.source == "export":
            import export
            model = export.load_export(args.path, device)
            source = describe_source("export", args.path, "ternary export, rebuilt with dequantized FP32 weights")
            source["export_manifest"] = {k: v for k, v in json.loads((args.path / "manifest.json").read_text()).items()
                                         if k in ("format", "sha256", "extra", "parameter_accounting", "bytes")}
        else:
            model = load_checkpoint(args.path, device, args.checkpoint_kind)
            source = describe_source("checkpoint", args.path, f"{args.checkpoint_kind} training checkpoint",
                                     args.checkpoint_kind)
        results = {}
        for name, manifest in sets:
            result = evaluate_set(model, name, manifest, source, args.batch_size, args.limit, device,
                                  args.max_batch_seconds)
            write_json(args.out_dir / f"{name}.json", result)
            results[name] = result
            w = result["wer"]
            print(f"{name}: n={result['utterances']} scored={w['utterances']} WER {100 * w['wer']:.2f}% "
                  f"(S {w['substitutions']} D {w['deletions']} I {w['insertions']} / {w['ref_words']}) "
                  f"{result['wall_s']:.0f} s RTFx {result['rtfx']:.0f}", flush=True)
    # Merge per-set results already in out_dir that scored the same weights (incremental runs).
    for path in sorted(args.out_dir.glob("*.json")):
        if path.name == "summary.json" or path.stem in results:
            continue
        previous = json.loads(path.read_text())
        if previous.get("source", {}).get("sha256") == source["sha256"]:
            results[path.stem] = previous
    summary = summarize(results, source)
    write_json(args.out_dir / "summary.json", summary)
    for name, e in summary["sets"].items():
        verdict = {True: "PASS", False: "FAIL", None: "(subset, no verdict)"}[e.get("reproduction_gate_pass")]
        extra = (f" published {e['published_wer']:.2f} diff {e['diff_vs_published']:+.2f} {verdict}"
                 if "published_wer" in e else "")
        print(f"{name:20s} {e['wer']:6.2f}{extra}")
    if "mean_wer" in summary:
        print(f"mean over {len(paths.MEAN_SETS)} sets: {summary['mean_wer']:.2f}")
    if "dev_mean_wer" in summary:
        print(f"dev mean over {len(paths.DEV_SETS)} sets: {summary['dev_mean_wer']:.2f}")
    print(f"padded short utterances (< {MIN_DECODE_SECONDS} s): {summary['padded_short']}")


if __name__ == "__main__":
    main()
