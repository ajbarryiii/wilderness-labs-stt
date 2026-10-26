"""Ternary QAT (or FP32 control) of Parakeet-TDT-0.6B-v2 on online teacher labels. See DESIGN.md.

One invocation is one (arm, learning rate) run in RUNS/<run-name>. Arms:

- P1: TDT loss on teacher transcripts only; prediction and joint networks frozen.
- P2: plus encoder-output matching against the frozen FP32 teacher; frozen pred/joint.
- P3: as P2 with trainable prediction and joint networks.
- M1: the main run; --recipe {P1,P2,P3} names the selected pilot recipe.
- A1: FP32 control (no quantization); --recipe as for M1.

Data: training audio is streamed (stream.make_training_stream(phase, seed)) and
labeled online: every batch, teacher.teacher_label_batch runs the frozen
pretrained model on the clean audio, returns greedy transcripts (the TDT targets),
a keep mask (DESIGN.md filters; dropped rows leave the batch) and the teacher's
encoder output, which the encoder-matching loss reuses. Every arm loads the teacher.

Ternary arms quantize the DESIGN.md module list with quant.py and ramp the ternary
weight fraction f = min(1, k / ramp_steps) over the first 25% of updates. Only
evaluations at f == 1 (or any evaluation of A1) can select the best checkpoint. At
the end the best checkpoint is exported (ternary) or saved as .nemo (A1), rebuilt
from that artifact, and scored on the full development sets; summary.json is
written last and atomically.

Resume (README.md "Training"): a checkpoint (latent FP32 weights, optimizer,
scheduler, every RNG, ramp step, metric accumulators, best-checkpoint record, the
stream position) is written every --ckpt-minutes, on SIGTERM/SIGINT (exit 76) and
when the stream fails after its own retries (exit 75), to a temporary file renamed
over ckpt/latest.pt after the old one is kept as ckpt/previous.pt. A restart
without --overwrite resumes from the newest complete checkpoint.
"""
from __future__ import annotations

import os

# Must precede CUDA initialisation: deterministic cuBLAS workspace, and an allocator
# that copes with the variable batch shapes of duration bucketing.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# NeMo's TDT loss is a numba CUDA kernel. This runtime has no CUDA toolkit for numba:
# point it at the Nix CUDA 12.9 libNVVM through a CUDA_HOME shim (numba expects
# nvvm/lib64), and target compute_90 PTX, which the driver JIT-compiles for the 5090
# (sm_120); NVVM 12.9 segfaults on numba's IR for compute_120. Must precede numba import.
NVVM_GLOB = "/nix/store/*-cuda12.9-cuda_nvcc-12.9.*/nvvm"
CUDA_HOME_SHIM = "/mnt/hd/wilderness-labs-stt/parakeet-ternary/caches/cuda-home"


def configure_numba_cuda() -> dict:
    import glob
    if "CUDA_HOME" not in os.environ and os.path.ismount("/mnt/hd"):
        nvvm = sorted(glob.glob(NVVM_GLOB))
        if nvvm:
            shim = os.path.join(CUDA_HOME_SHIM, "nvvm")
            os.makedirs(shim, exist_ok=True)
            for name, target in (("lib64", "lib"), ("libdevice", "libdevice")):
                link, want = os.path.join(shim, name), os.path.join(nvvm[-1], target)
                if os.path.realpath(link) != os.path.realpath(want):
                    tmp = f"{link}.tmp{os.getpid()}"
                    os.symlink(want, tmp)
                    os.replace(tmp, link)
            os.environ["CUDA_HOME"] = CUDA_HOME_SHIM
    os.environ.setdefault("NUMBA_FORCE_CUDA_CC", "9.0")
    return {"CUDA_HOME": os.environ.get("CUDA_HOME"),
            "NUMBA_FORCE_CUDA_CC": os.environ["NUMBA_FORCE_CUDA_CC"]}


NUMBA_CUDA = configure_numba_cuda()

import argparse  # noqa: E402
import contextlib  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import queue  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.utils.checkpoint  # noqa: E402

import paths  # noqa: E402

ARMS = ("P1", "P2", "P3", "M1", "A1")
RECIPES = {  # DESIGN.md "Distillation variants compared in the pilot"
    "P1": {"encoder_matching": False, "train_pred_joint": False},
    "P2": {"encoder_matching": True, "train_pred_joint": False},
    "P3": {"encoder_matching": True, "train_pred_joint": True},
}
RAMP_FRACTION = 0.25
WARMUP_FRACTION = 0.02
ENCODER_MATCHING_WEIGHT = 1.0
TEACHER_PRECISION = "bf16"  # DESIGN.md: the teacher encoder runs under BF16 autocast
MIN_SECONDS, MAX_SECONDS = 1.0, 30.0  # DESIGN.md filter; applied before bucketing as well
LOG_EVERY = 20
POOL_BATCHES = 8  # duration bucketing sorts pools of about this many batches
PREFETCH_BATCHES = 16
EXIT_STREAM_FAILURE = 75  # checkpointed; sweep.py waits and resumes
EXIT_INTERRUPTED = 76  # checkpointed on SIGTERM/SIGINT; rerun the same command to resume
EXIT_UNREADABLE_CHECKPOINT = 77  # checkpoint files exist but none loads; never restart silently
EXIT_RESUME_MISMATCH = 78  # the invocation's arguments differ from the checkpoint's
SELECT = ("final", "best")
# Every source module whose change can alter what a run trains or how it is scored,
# relative to the repository's finetune/ directory; sweep.py mirrors and checks the list.
# Excluded on purpose: powerlog.py (a separate recorder process; it cannot affect results),
# sweep.py (orchestration only), testsets.py (test-set preparation, not read by training).
TRAINING_SOURCES = ("parakeet-ternary/train.py", "parakeet-ternary/quant.py",
                    "parakeet-ternary/stream.py", "parakeet-ternary/teacher.py",
                    "parakeet-ternary/evaluate.py", "parakeet-ternary/export.py",
                    "parakeet-ternary/data.py", "parakeet-ternary/paths.py",
                    "whisper-ternary/quant.py", "whisper-ternary/wer.py")
# Arguments that may differ between the original invocation and a resume.
RESUME_ALLOWED_CHANGES = ("ckpt_minutes", "eval_batch_size", "no_powerlog", "overwrite")
# DESIGN.md "Distillation variants": the recipe flags each arm must have (M1, A1: per recipe).
ARM_FLAGS = {"P1": {"encoder_matching": False, "train_pred_joint": False, "quantized": True},
             "P2": {"encoder_matching": True, "train_pred_joint": False, "quantized": True},
             "P3": {"encoder_matching": True, "train_pred_joint": True, "quantized": True}}
CKPT_FORMAT = 2
UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
STREAMS = ("hub", "local-librispeech")
LIBRISPEECH_TC100 = Path("/mnt/hd/wilderness-labs-stt/stt-distillation/datasets/libri/"
                         "LibriSpeech/train-clean-100")
# Everything a run writes. --overwrite clears all of it; a fresh start clears all but ckpt/.
OUTPUTS = ("summary.json", "config.json", "metrics.jsonl", "resumes.jsonl", "dev-subset",
           "export", "best.nemo", "eval-dev.json", "eval-dev", "probe.json")
# summary.json contract checked by sweep.py.
SUMMARY_TYPES = {"run_name": str, "arm": str, "recipe": str, "lr": float, "max_steps": int,
                 "steps": int, "select": str, "selected_step": int, "eval_every": int,
                 "grad_checkpointing": bool, "source_hashes": dict,
                 "dev_subset_mean_wer": float,
                 "dev_mean_wer": float, "dev_wer": dict, "seed": int, "batch_seconds": float,
                 "train": str, "stream": dict, "quantized": bool, "encoder_matching": bool,
                 "train_pred_joint": bool, "ramp_steps": int, "warmup_steps": int,
                 "audio_seconds_seen": float, "smoke": bool, "started_utc": str,
                 "finished_utc": str}
# Arguments that must match for a checkpoint to be resumed (the rest are operational).
RESUME_KEYS = ("arm", "recipe", "lr", "max_steps", "train", "stream", "batch_seconds", "seed",
               "eval_every", "dev_limit", "smoke", "select")


def utc() -> str:
    return time.strftime(UTC_FORMAT, time.gmtime())


def log(msg: str) -> None:
    print(f"[train {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- arguments

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--recipe", choices=list(RECIPES),
                        help="distillation recipe; P1-P3 imply their own, M1 and A1 require it")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-steps", type=int, help="optimizer updates (authoritative)")
    parser.add_argument("--train", choices=("pilot", "main"),
                        help="stream phase (default pilot for P1-P3, main for M1; required for A1)")
    parser.add_argument("--stream", choices=STREAMS, default="hub",
                        help="hub: stream.make_training_stream (the experiment); "
                             "local-librispeech: train-clean-100 from disk (probes and plumbing)")
    parser.add_argument("--batch-seconds", type=float, default=600.0,
                        help="padded audio seconds per batch (batch size x longest utterance)")
    parser.add_argument("--grad-checkpointing", action="store_true",
                        help="recompute each encoder layer in the backward pass")
    parser.add_argument("--eval-every", type=int, help="default: max_steps // 10")
    parser.add_argument("--select", choices=SELECT, default="final",
                        help="model scored and exported: final = the last checkpoint (step == "
                             "max_steps, f == 1; DESIGN.md pilot selection); best = the "
                             "f == 1 checkpoint with the lowest dev-subset mean WER")
    parser.add_argument("--dev-limit", type=int,
                        help="first N utterances of each dev set (subset and full); smoke only")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--ckpt-minutes", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=paths.SEED)
    parser.add_argument("--probe-steps", type=int,
                        help="throughput/memory probe: 2 worst-case (30 s utterance) updates, "
                             "then N timed updates; writes probe.json (no eval/checkpoint/export)")
    parser.add_argument("--no-powerlog", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="plumbing run: 20 steps, eval every 10 on 16 utterances per set")
    args = parser.parse_args(argv)
    return resolve_args(args, parser)


def resolve_args(args: argparse.Namespace, parser: argparse.ArgumentParser | None = None
                 ) -> argparse.Namespace:
    def fail(msg: str):
        if parser is not None:
            parser.error(msg)
        raise ValueError(msg)

    if args.arm in RECIPES:
        if args.recipe not in (None, args.arm):
            fail(f"--recipe {args.recipe} contradicts arm {args.arm}")
        args.recipe = args.arm
    elif args.recipe is None:
        fail(f"arm {args.arm} requires --recipe")
    if args.train is None:
        if args.arm == "A1":
            fail("arm A1 requires --train")
        args.train = "main" if args.arm == "M1" else "pilot"
    if args.smoke:
        args.max_steps = args.max_steps or 20
        args.eval_every = args.eval_every or 10
        args.dev_limit = args.dev_limit or 16
    if args.probe_steps:
        args.max_steps = args.max_steps or args.probe_steps + 2
    if args.max_steps is None or args.max_steps < 1:
        fail("--max-steps is required and must be positive")
    args.eval_every = args.eval_every or max(1, args.max_steps // 10)
    if not (math.isfinite(args.lr) and args.lr > 0):
        fail("--lr must be positive")
    if not (math.isfinite(args.batch_seconds) and args.batch_seconds > 0):
        fail("--batch-seconds must be positive")
    return args


def recipe_of(args) -> dict:
    flags = {**RECIPES[args.recipe], "quantized": args.arm != "A1"}
    expected = ARM_FLAGS.get(args.arm, {**RECIPES[args.recipe], "quantized": args.arm == "M1"})
    if flags != expected:
        raise ValueError(f"arm {args.arm} recipe {args.recipe}: flags {flags} != {expected}")
    return flags


def verify_recipe(student, recipe: dict, quant=None) -> dict:
    """Check the configured model against the recipe flags' values; returns what was found.

    Prediction and joint networks: trainable iff train_pred_joint, encoder always
    trainable; ternary modules present iff quantized; dropout 0 in prediction and joint.
    """
    found = {
        "encoder_trainable": all(p.requires_grad for p in student.encoder.parameters()),
        "pred_joint_trainable": [any(p.requires_grad for p in m.parameters())
                                 for m in (student.decoder, student.joint)],
        "pred_joint_frozen": [not any(p.requires_grad for p in m.parameters())
                              for m in (student.decoder, student.joint)],
        "ternary_modules": len(quant.quantized_module_names(student)) if quant else 0,
        "pred_joint_dropout": sorted({m.p for part in (student.decoder, student.joint)
                                      for m in part.modules() if isinstance(m, torch.nn.Dropout)}
                                     | {m.dropout for part in (student.decoder, student.joint)
                                        for m in part.modules()
                                        if isinstance(m, torch.nn.RNNBase)})}
    problems = []
    if not found["encoder_trainable"]:
        problems.append("encoder not fully trainable")
    if recipe["train_pred_joint"] and not all(found["pred_joint_trainable"]):
        problems.append("prediction/joint networks should be trainable")
    if not recipe["train_pred_joint"] and not all(found["pred_joint_frozen"]):
        problems.append("prediction/joint networks should be frozen")
    if recipe["quantized"] != (found["ternary_modules"] > 0):
        problems.append(f"quantized={recipe['quantized']} but {found['ternary_modules']} "
                        "ternary modules")
    if any(p != 0.0 for p in found["pred_joint_dropout"]):
        problems.append(f"prediction/joint dropout {found['pred_joint_dropout']} != 0")
    if problems:
        raise RuntimeError(f"model does not match recipe {recipe}: {'; '.join(problems)}")
    return found


# ----------------------------------------------------------------------------- schedules

def lr_factor(step: int, warmup: int, total: int) -> float:
    """Linear warmup reaching the peak on update `warmup`, then linear decay to 0 at `total`."""
    if step < warmup:
        return (step + 1) / warmup
    return max(0.0, (total - step) / max(1, total - warmup))


def warmup_steps(max_steps: int) -> int:
    return max(1, round(WARMUP_FRACTION * max_steps))


def ramp_steps(max_steps: int) -> int:
    return max(1, round(RAMP_FRACTION * max_steps))


def ramp_weight_fraction(update: int, steps: int) -> float:
    """f for 1-based update k: min(1, k / ramp_steps)."""
    if update < 1:
        raise ValueError(f"updates are 1-based, got {update}")
    return 1.0 if steps <= 0 else min(1.0, update / steps)


def selectable(weight_fraction: float | None) -> bool:
    """Only fully ternary (f == 1) or FP32 (None) evaluations may select the best checkpoint."""
    return weight_fraction is None or weight_fraction == 1.0


# ----------------------------------------------------------------------------- io helpers

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, obj: object) -> None:
    """Atomic: temp file, fsync, rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as handle:
        handle.write(json.dumps(obj, indent=1) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    """Records of a JSONL file written in one piece (manifests); any bad line raises."""
    records = []
    with open(path) as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
                if limit is not None and len(records) >= limit:
                    break
    return records


def read_log_jsonl(path: Path) -> tuple[list[dict], bool]:
    """(records, torn) of an append-only JSONL log (metrics.jsonl, resumes.jsonl).

    A power cut during an append can leave the last line incomplete: it is ignored with a
    warning (torn = True). A damaged line anywhere else raises, since that is not a torn
    append. A missing file is empty.
    """
    if not path.exists():
        return [], False
    lines = path.read_bytes().split(b"\n")
    records, torn = [], False
    for i, raw in enumerate(lines):
        if not raw.strip(b" \x00\r"):
            continue
        try:
            record = json.loads(raw)
            if not isinstance(record, dict):
                raise ValueError("not an object")
        except ValueError as error:
            last = all(not x.strip(b" \x00\r") for x in lines[i + 1:])
            if not last:
                raise ValueError(f"{path}: line {i + 1} is damaged ({error})") from error
            log(f"warning: ignoring a torn last line in {path} ({len(raw)} bytes)")
            torn = True
            continue
        records.append(record)
    return records, torn


def rewrite_jsonl(path: Path, records: list[dict]) -> int:
    """Atomically replace a JSONL log with these records; returns the new size."""
    tmp = path.with_name(path.name + ".tmp")
    data = "".join(json.dumps(r, default=str) + "\n" for r in records).encode()
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
    fsync_dir(path.parent)
    return len(data)


def append_jsonl(path: Path, record: dict) -> int:
    """Append one complete line with a single write() and fsync; returns the new size.

    If the file ends in a torn line (no final newline), it is first rewritten without it,
    so the new record never fuses with the fragment.
    """
    if path.exists() and path.stat().st_size:
        with open(path, "rb") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                rewrite_jsonl(path, read_log_jsonl(path)[0])
    data = (json.dumps(record, default=str) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        written = os.write(fd, data)
        if written != len(data):
            raise OSError(f"short write to {path}: {written} of {len(data)} bytes")
        os.fsync(fd)
        return os.fstat(fd).st_size
    finally:
        os.close(fd)


# ----------------------------------------------------------------------------- data

class StreamFailure(RuntimeError):
    """The training stream raised (after its own retries); the run checkpoints and exits 75."""


class LocalLibriSpeechStream:
    """A resumable stand-in for the training stream: LibriSpeech train-clean-100 from disk.

    Yields the stream contract {"audio" float32 16 kHz, "duration", "text", "id", "source"}
    in a seeded per-epoch permutation. concat_seconds > 0 concatenates consecutive
    utterances (same chapter order) up to that length, for worst-case memory probes.
    """

    def __init__(self, seed: int, root: Path = LIBRISPEECH_TC100,
                 concat_seconds: float | None = None, limit: int | None = None) -> None:
        self.seed, self.concat = seed, concat_seconds
        items = []
        for trans in sorted(Path(root).glob("*/*/*.trans.txt")):
            for line in trans.read_text().splitlines():
                if line.strip():
                    uid, text = line.split(" ", 1)
                    items.append((uid, str(trans.parent / f"{uid}.flac"), text.strip()))
        self.items = items[:limit]
        self.epoch, self.position = 0, 0

    def state_dict(self) -> dict:
        return {"seed": self.seed, "epoch": self.epoch, "position": self.position,
                "concat_seconds": self.concat, "items": len(self.items)}

    def load_state_dict(self, state: dict) -> None:
        if (state["seed"], state["concat_seconds"], state["items"]) != (
                self.seed, self.concat, len(self.items)):
            raise ValueError(f"stream state {state} does not belong to this stream")
        self.epoch, self.position = state["epoch"], state["position"]

    def _order(self) -> np.ndarray:
        if self.concat:
            return np.arange(len(self.items))
        return np.random.default_rng([self.seed, 3, self.epoch]).permutation(len(self.items))

    def __iter__(self):
        import soundfile as sf
        while True:
            order = self._order()
            while self.position < len(order):
                pieces, texts, ids = [], [], []
                while self.position < len(order):
                    uid, path, text = self.items[order[self.position]]
                    wav, _ = sf.read(path, dtype="float32")
                    if self.concat and pieces and (sum(map(len, pieces)) + len(wav)) / \
                            paths.SAMPLE_RATE > self.concat:
                        break
                    self.position += 1
                    pieces.append(wav)
                    texts.append(text)
                    ids.append(uid)
                    if not self.concat:
                        break
                audio = np.concatenate(pieces)
                yield {"audio": audio, "duration": len(audio) / paths.SAMPLE_RATE,
                       "text": " ".join(texts), "id": "+".join(ids), "source": "librispeech_local"}
            self.epoch, self.position = self.epoch + 1, 0


def open_stream(args, concat_seconds: float | None = None):
    if args.stream == "local-librispeech":
        return LocalLibriSpeechStream(args.seed, concat_seconds=concat_seconds)
    stream = local_module("stream")
    return stream.make_training_stream(args.train, args.seed)


def stream_identity(args, stream=None) -> dict:
    """What determines the data order; sweep.py requires it to match across compared arms."""
    out = {"kind": args.stream, "phase": args.train, "seed": args.seed}
    state = stream.state_dict() if stream is not None else {}
    if "config_hash" in state:
        out["config_hash"] = state["config_hash"]
    if args.stream == "hub":
        path = paths.HERE / "stream.py"
        out["stream_py_sha256"] = sha256(path) if path.exists() else None
    teacher_py = paths.HERE / "teacher.py"
    out["teacher_py_sha256"] = sha256(teacher_py) if teacher_py.exists() else None
    return out


def cut_pool(durations: list[float], batch_seconds: float) -> list[list[int]]:
    """Sort a pool by duration (stable) and cut batches with count x longest <= batch_seconds;
    a single longer utterance forms its own batch; the last batch may be smaller."""
    order = sorted(range(len(durations)), key=lambda i: durations[i])
    batches, current = [], []
    for i in order:
        if current and (len(current) + 1) * durations[i] > batch_seconds:
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    return batches


def collate(items: list[dict]) -> dict:
    lengths = torch.tensor([len(it["audio"]) for it in items], dtype=torch.long)
    audio = torch.zeros(len(items), int(lengths.max()))
    for row, it in enumerate(items):
        audio[row, :len(it["audio"])] = torch.from_numpy(np.ascontiguousarray(it["audio"],
                                                                              dtype=np.float32))
    return {"audio": audio.pin_memory() if torch.cuda.is_available() else audio,
            "audio_len": lengths, "texts": [it.get("text") for it in items],
            "ids": [it["id"] for it in items], "sources": [it.get("source") for it in items],
            "seconds": float(lengths.sum()) / paths.SAMPLE_RATE}


class Batcher:
    """Duration-bucketed batches from a resumable stream, produced by a background thread.

    Pools of about POOL_BATCHES x batch_seconds of audio are read, sorted by duration,
    cut into batches and shuffled with a generator seeded by (seed, pool index). Each
    batch carries the stream state from just before its pool was read, so the data
    position after a batch is (pool-start state, pool index, batches consumed): resume
    restores the stream to the pool start, re-reads the pool, rebuilds the same batches
    and skips the consumed ones. A deterministic stream therefore continues exactly; no
    audio is stored in checkpoints. Utterances outside [1, 30] s are dropped before
    bucketing (counted). Stream exceptions are re-raised in the consumer as StreamFailure.
    """

    def __init__(self, stream, batch_seconds: float, seed: int, position: dict | None = None,
                 pool_batches: int | None = None, prefetch: int = PREFETCH_BATCHES) -> None:
        self.stream, self.batch_seconds, self.seed = stream, batch_seconds, seed
        self.pool_seconds = (pool_batches or POOL_BATCHES) * batch_seconds
        self.start = position
        self.queue: queue.Queue = queue.Queue(maxsize=prefetch)
        self.stop = threading.Event()
        self.prefiltered = {"short": 0, "long": 0}
        self.thread = threading.Thread(target=self._run, name="batcher", daemon=True)
        self.thread.start()

    def _put(self, item) -> bool:
        while not self.stop.is_set():
            try:
                self.queue.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        try:
            pool, skip = 0, 0
            if self.start is not None:
                self.stream.load_state_dict(copy.deepcopy(self.start["pool_state"]))
                pool, skip = self.start["pool"], self.start["consumed"]
            iterator = iter(self.stream)
            while not self.stop.is_set():
                pool_state = copy.deepcopy(self.stream.state_dict())
                items, seconds = [], 0.0
                while seconds < self.pool_seconds:
                    item = next(iterator)
                    d = float(item["duration"])
                    if d < MIN_SECONDS or d > MAX_SECONDS:
                        self.prefiltered["short" if d < MIN_SECONDS else "long"] += 1
                        continue
                    items.append(item)
                    seconds += d
                next_state = copy.deepcopy(self.stream.state_dict())
                cut = cut_pool([float(it["duration"]) for it in items], self.batch_seconds)
                order = np.random.default_rng([self.seed, 2, pool]).permutation(len(cut))
                batches = [cut[j] for j in order]
                for j, idx in enumerate(batches):
                    if j < skip:
                        continue
                    last = j + 1 == len(batches)
                    after = ({"pool_state": next_state, "pool": pool + 1, "consumed": 0} if last
                             else {"pool_state": pool_state, "pool": pool, "consumed": j + 1})
                    batch = collate([items[i] for i in idx])
                    batch["position_after"] = after
                    batch["pool"], batch["index_in_pool"] = pool, j
                    if not self._put(batch):
                        return
                pool, skip = pool + 1, 0
                del items
        except BaseException as error:  # noqa: BLE001 - forwarded to the consumer
            self._put(error)

    def next(self, timeout: float | None = None) -> dict:
        item = self.queue.get(timeout=timeout)
        if isinstance(item, BaseException):
            if is_stream_error(item):
                raise StreamFailure(f"{type(item).__name__}: {item}") from item
            raise RuntimeError(f"batcher thread failed: {type(item).__name__}: {item}") from item
        return item

    def close(self, close_stream: bool = True) -> None:
        self.stop.set()
        while True:  # unblock a producer waiting on a full queue
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        if close_stream and callable(getattr(self.stream, "close", None)):
            try:
                self.stream.close()  # stop the stream's download workers
            except Exception as error:  # noqa: BLE001 - shutdown is best effort
                log(f"warning: closing the stream failed ({type(error).__name__}: {error})")
        self.thread.join(timeout=10)


def is_stream_error(error: BaseException) -> bool:
    """stream.StreamError (raised after the stream's own retries) or a network/OS error.
    Anything else from the producer is a bug and is not retried by sweep.py."""
    return (type(error).__name__ == "StreamError" or isinstance(error, (OSError, TimeoutError))
            or isinstance(error, StopIteration))


# ----------------------------------------------------------------------------- model

def local_module(name: str):
    """Import an experiment module, refusing the same-named Whisper module that PYTHONPATH
    would otherwise supply when the Parakeet one is missing."""
    import importlib
    module = importlib.import_module(name)
    if Path(module.__file__).resolve().parent != paths.HERE:
        raise ImportError(f"{name} resolved to {module.__file__}, not {paths.HERE}")
    return module


def verify_base() -> dict:
    lock = json.loads((paths.MODEL_DIR / "lock.json").read_text())
    if lock.get("revision") != paths.MODEL_REVISION:
        raise ValueError(f"lock revision {lock.get('revision')} != {paths.MODEL_REVISION}")
    name = paths.MODEL_FILE.name
    if sha256(paths.MODEL_FILE) != lock["files"][name]:
        raise ValueError(f"checksum mismatch for {paths.MODEL_FILE}")
    return {"base_model": lock["model"], "base_revision": paths.MODEL_REVISION,
            "base_sha256": lock["files"][name],
            "base_lock_sha256": sha256(paths.MODEL_DIR / "lock.json")}


def load_pretrained(device: str = "cpu"):
    """The pinned FP32 checkpoint (caller verifies the hash once with verify_base)."""
    import logging as pylogging
    from nemo.collections.asr.models import ASRModel
    from nemo.utils import logging
    level = logging.get_verbosity()
    logging.set_verbosity(pylogging.ERROR)
    try:
        model = ASRModel.restore_from(str(paths.MODEL_FILE), map_location="cpu")
    finally:
        logging.set_verbosity(level)
    return model.to(device=device, dtype=torch.float32)


def load_teacher(model) -> torch.nn.Module:
    """The frozen FP32 teacher (never quantized): eval mode, no gradients."""
    model.eval()
    model.requires_grad_(False)
    return model


def seed_everything(seed: int, model=None) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    for r in private_rngs(model):
        r.seed(seed)  # legacy (non-vectorized) SpecAugment path


def deterministic_mode() -> dict:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {"cudnn_deterministic": True, "cudnn_benchmark": False,
            "use_deterministic_algorithms": "warn_only",
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def private_rngs(model) -> list[random.Random]:
    augment = getattr(model, "spec_augmentation", None) if model is not None else None
    return [m._rng for m in ([] if augment is None else augment.modules())
            if isinstance(getattr(m, "_rng", None), random.Random)]


def rng_state(model) -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "private": [r.getstate() for r in private_rngs(model)]}


def set_rng_state(model, state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    for r, s in zip(private_rngs(model), state["private"], strict=True):
        r.setstate(s)


def reseed_cudnn_rnn_dropout() -> None:
    """Make cuDNN's LSTM dropout state a function of the checkpointed CUDA generator.

    PyTorch caches the cuDNN RNN dropout state per device and re-seeds it (from the CUDA
    generator) only after the generator's state is set; the cached state is not part of
    get_rng_state. Setting the generator to its own state at the start of every step
    re-seeds it there, so a resumed run draws the same LSTM dropout masks (P3 trains the
    prediction network in train mode).
    """
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(torch.cuda.get_rng_state())


@contextlib.contextmanager
def isolated_rng(model):
    """Restore every RNG afterwards, so evaluation never shifts the training trajectory."""
    state = rng_state(model)
    try:
        yield
    finally:
        set_rng_state(model, state)


class _KeepRunningStats:
    """Recompute context for checkpointed layers: BatchNorm recomputes exactly as in the
    first forward (batch statistics, same saved tensors), and its running statistics are
    restored afterwards so they are updated once per step, as without checkpointing."""

    def __init__(self, layer: torch.nn.Module) -> None:
        self.norms = [m for m in layer.modules()
                      if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        self.saved: list = []

    def __enter__(self):
        self.saved = [[None if b is None else b.clone() for b in
                       (m.running_mean, m.running_var, m.num_batches_tracked)] for m in self.norms]

    def __exit__(self, *exc):
        for m, buffers in zip(self.norms, self.saved):
            for b, s in zip((m.running_mean, m.running_var, m.num_batches_tracked), buffers):
                if b is not None:
                    b.copy_(s)


def enable_grad_checkpointing(encoder: torch.nn.Module) -> int:
    """Wrap every conformer layer's forward in non-reentrant activation checkpointing.

    NeMo calls the layers with keyword arguments only, and torch.utils.checkpoint finds
    the devices whose RNG state it preserves from the positional tensor arguments; so the
    arguments are bound by the layer's signature and passed positionally. Otherwise the
    CUDA RNG is not restored for the recompute and dropout masks (hence gradients) differ.
    """
    import inspect
    for layer in encoder.layers:
        inner = layer.forward
        names = list(inspect.signature(inner).parameters)

        def forward(*a, _inner=inner, _layer=layer, _names=names, **kw):
            if not (_layer.training and torch.is_grad_enabled()):
                return _inner(*a, **kw)
            bound = {**dict(zip(_names, a)), **kw}
            unknown = set(bound) - set(_names)
            if unknown:
                raise TypeError(f"unexpected conformer layer arguments {sorted(unknown)}")
            values = [bound.get(n) for n in _names]

            def run(*vals):
                return _inner(**dict(zip(_names, vals)))

            return torch.utils.checkpoint.checkpoint(
                run, *values, use_reentrant=False, preserve_rng_state=True,
                context_fn=lambda: (contextlib.nullcontext(), _KeepRunningStats(_layer)))

        layer.forward = forward
    return len(encoder.layers)


def disable_pred_joint_dropout(student) -> int:
    """Dropout p = 0 in the prediction and joint networks (nn.Dropout and LSTM inter-layer
    dropout); returns how many modules were changed.

    DESIGN.md lets the arms differ only in the extra loss and in whether the prediction
    and joint networks are trainable. The pinned config has 20% dropout there; P1/P2 run
    them frozen in eval mode, so P3 (trainable, in train mode because cuDNN's LSTM
    backward requires it) would otherwise also differ by dropout. Dropout is not stored
    in state_dict, so exports and evaluation are unaffected (they run in eval mode).
    """
    changed = 0
    for part in (student.decoder, student.joint):
        for module in part.modules():
            if isinstance(module, torch.nn.Dropout) and module.p != 0.0:
                module.p = 0.0
                changed += 1
            elif isinstance(module, torch.nn.RNNBase) and module.dropout != 0.0:
                module.dropout = 0.0
                changed += 1
    return changed


def configure_trainable(student, recipe: dict) -> dict:
    """requires_grad, train/eval mode and dropout per recipe; returns a parameter count record.

    The encoder always trains (with its configured dropout). The prediction and joint
    networks train only when the recipe says so; in every arm their dropout is 0, and
    frozen they run in eval mode, so the student is trained through exactly the function
    the teacher's decoding uses.
    """
    dropout_disabled = disable_pred_joint_dropout(student)
    student.requires_grad_(False)
    student.encoder.requires_grad_(True)
    if recipe["train_pred_joint"]:
        student.decoder.requires_grad_(True)
        student.joint.requires_grad_(True)
    set_train_modes(student, recipe)
    count = lambda m, t: sum(p.numel() for p in m.parameters() if p.requires_grad == t)  # noqa: E731
    return {"trainable": count(student, True), "frozen": count(student, False),
            "pred_joint_dropout_modules_zeroed": dropout_disabled,
            "encoder_trainable": count(student.encoder, True),
            "decoder_trainable": count(student.decoder, True),
            "joint_trainable": count(student.joint, True),
            "trainable_modules": [n for n in ("encoder", "decoder", "joint")
                                  if any(p.requires_grad for p in getattr(student, n).parameters())]}


def set_train_modes(student, recipe: dict) -> None:
    student.train()
    if not recipe["train_pred_joint"]:
        student.decoder.eval()
        student.joint.eval()


def param_groups(model, weight_decay: float = 0.01) -> list[dict]:
    """Weight decay on matrices (dim >= 2) only; trainable parameters in module order."""
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if p.requires_grad:
            (decay if p.dim() >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def make_optimizer(model, lr: float, max_steps: int):
    optimizer = torch.optim.AdamW(param_groups(model), lr=lr, betas=(0.9, 0.98), eps=1e-8)
    warm = warmup_steps(max_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, warm, max_steps))
    return optimizer, scheduler


# ----------------------------------------------------------------------------- losses

_TDT_CLAMP_LINES = (("g = min(g, clamp)", "g = g if g < clamp else clamp"),
                    ("g = max(g, -clamp)", "g = g if g > -clamp else -clamp"))


_PATCHED_GRAD_KERNELS = ("compute_tdt_grad_kernel", "compute_grad_kernel")


def prepare_tdt_loss() -> str:
    """Rebuild NeMo's transducer gradient kernels without the builtins min/max.

    numba 0.67's built-in CUDA target types builtin min/max through a *args overload it
    cannot compile ("Signature mismatch: 2 argument types given, but function takes 1").
    The TDT loss launches compute_tdt_grad_kernel, and with probability omega (0.1 in
    this model's config) the standard RNN-T compute_grad_kernel; both use min/max only
    for optional gradient clamping, inside `if clamp > 0.0` (this config disables
    clamping). Those two lines are replaced by the equivalent conditional expressions,
    the source text is asserted, nothing else changes. tests/test_train.py checks the
    TDT loss and gradient against NeMo's pure-PyTorch reference and the omega path
    against finite differences. Idempotent; returns the SHA-256 of the rebuilt sources.
    """
    import inspect
    import re
    from nemo.collections.asr.parts.numba.rnnt_loss.utils.cuda_utils import gpu_rnnt_kernel as k
    from numba import cuda
    if getattr(k, "_parakeet_ternary_patched", False):
        return k._parakeet_ternary_patched
    digest = hashlib.sha256()
    for name in _PATCHED_GRAD_KERNELS:
        source = inspect.getsource(getattr(k, name).py_func)
        for old, new in _TDT_CLAMP_LINES:
            if source.count(old) != 1:
                raise RuntimeError(f"NeMo {name} changed; expected one {old!r}")
            source = source.replace(old, new)
        source, n = re.subn(r"^@cuda\.jit\(\)\n", "", source)
        if n != 1:
            raise RuntimeError(f"NeMo {name} decorator changed")
        namespace: dict = {}
        exec(compile(source, k.__file__, "exec"), k.__dict__, namespace)
        setattr(k, name, cuda.jit()(namespace[name]))
        digest.update(source.encode())
    k._parakeet_ternary_patched = digest.hexdigest()
    return k._parakeet_ternary_patched


def encoder_matching_loss(student_enc: torch.Tensor, teacher_enc: torch.Tensor,
                          lengths: torch.Tensor) -> torch.Tensor:
    """mean((E_s - E_t)^2) / var(E_t) over valid frames of [B, D, T] encoder outputs, in FP32.

    Both the mean and the (population) variance run over every element of every valid
    frame in the batch; the variance is the teacher's and carries no gradient.
    """
    frames = student_enc.shape[-1]
    mask = torch.arange(frames, device=lengths.device)[None, :] < lengths[:, None]  # [B, T]
    s = student_enc.float().transpose(1, 2)[mask]  # [N, D]
    t = teacher_enc.float().transpose(1, 2)[mask].detach()
    return (s - t).pow(2).mean() / t.var(unbiased=False).clamp_min(1e-8)


def tokenize(model, texts: list[str], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    ids = [model.tokenizer.text_to_ids(t) for t in texts]
    lengths = torch.tensor([len(x) for x in ids], dtype=torch.long)
    tokens = torch.zeros(len(ids), max(1, int(lengths.max())), dtype=torch.long)
    for row, x in enumerate(ids):
        tokens[row, :len(x)] = torch.tensor(x, dtype=torch.long)
    return tokens.to(device), lengths.to(device)


def restore_loss_reduction(student) -> None:
    """Reset the TDT loss reduction to the model's configured value.

    NeMo's fused joint sets loss.reduction = None around each sub-batch loss and restores
    it afterwards; an exception inside (a CUDA OOM caught by the throughput probe) leaves
    it None, after which the joint returns per-utterance losses. Resetting before every
    loss makes a caught exception harmless.
    """
    configured = student.cfg.get("rnnt_reduction", "mean_batch")
    if student.loss.reduction != configured:
        student.loss.reduction = configured
    joint_loss = getattr(student.joint, "_loss", None)
    if joint_loss is not None and joint_loss.reduction != configured:
        joint_loss.reduction = configured


def compute_losses(student, audio: torch.Tensor, audio_len: torch.Tensor, tokens: torch.Tensor,
                   token_len: torch.Tensor, recipe: dict, teacher_enc: torch.Tensor | None = None,
                   teacher_len: torch.Tensor | None = None
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """(loss, tdt, enc_mse) for one labeled batch; enc_mse is None without encoder matching.

    The student computes its own features (preprocessor in training mode, i.e. the
    model's 1e-5 dither) and applies the model's SpecAugment; its encoding of the
    augmented input is matched against the teacher's encoding of the clean audio
    (teacher_label_batch's output, no second teacher forward). Both losses use the
    one student forward.
    """
    restore_loss_reduction(student)
    with torch.no_grad():
        feats, feat_len = student.preprocessor(input_signal=audio, length=audio_len)
        if student.spec_augmentation is not None and student.training:
            feats = student.spec_augmentation(input_spec=feats, length=feat_len)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        enc, enc_len = student.encoder(audio_signal=feats, length=feat_len)
        grad_ctx = contextlib.nullcontext() if recipe["train_pred_joint"] else torch.no_grad()
        with grad_ctx:
            dec, _, _ = student.decoder(targets=tokens, target_length=token_len)
        tdt, _, _, _ = student.joint(encoder_outputs=enc, decoder_outputs=dec,
                                     encoder_lengths=enc_len, transcripts=tokens,
                                     transcript_lengths=token_len, compute_wer=False)
    if tdt.dim() != 0:
        raise RuntimeError(f"TDT loss is not a scalar (shape {tuple(tdt.shape)}); "
                           f"loss reduction is {student.loss.reduction!r}")
    if not recipe["encoder_matching"]:
        return tdt, tdt, None
    if not torch.equal(teacher_len, enc_len) or teacher_enc.shape[-1] != enc.shape[-1]:
        raise RuntimeError("teacher and student encoder lengths differ")
    enc_mse = encoder_matching_loss(enc, teacher_enc, enc_len)
    return tdt + ENCODER_MATCHING_WEIGHT * enc_mse, tdt, enc_mse


# ----------------------------------------------------------------------------- evaluation

def corpus_wer(evaluate, records: list[dict]) -> float:
    """Corpus WER (a fraction) over the scored records, via evaluate.wer_summary."""
    return float(evaluate.wer_summary(records)["wer"])


def dev_records(name: str, subset: bool, limit: int | None) -> list[dict]:
    path = paths.MANIFESTS / (f"{name}_400.jsonl" if subset else f"{name}.jsonl")
    return read_jsonl(path, limit)


class DevEvaluator:
    """Decodes the fixed development subsets (or full sets) with evaluate.decode."""

    def __init__(self, limit: int | None, batch_size: int, device: str) -> None:
        self.evaluate = local_module("evaluate")
        self.limit, self.batch_size, self.device = limit, batch_size, device
        self.subsets = {name: dev_records(name, True, limit) for name in paths.DEV_SETS}

    def score(self, model, sets: dict[str, list[dict]]) -> dict[str, dict]:
        out = {}
        for name, records in sets.items():
            decoded = self.evaluate.decode(model, records, self.batch_size, self.device)
            out[name] = {"wer": corpus_wer(self.evaluate, decoded), "utterances": len(decoded),
                         "records": decoded}
        return out

    def subset(self, model) -> dict[str, dict]:
        return self.score(model, self.subsets)

    def full(self, model) -> dict[str, dict]:
        return self.score(model, {n: dev_records(n, False, self.limit) for n in paths.DEV_SETS})


def mean_wer(results: dict[str, dict]) -> float:
    return sum(r["wer"] for r in results.values()) / len(results)


# ----------------------------------------------------------------------------- checkpoints

def ckpt_paths(run: Path) -> tuple[Path, Path, Path]:
    d = run / "ckpt"
    return d / "latest.pt", d / "previous.pt", d / "latest.pt.tmp"


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def drop_page_cache(path: Path, sync: bool = True) -> None:
    """fsync a file, then ask the kernel to drop its cached pages (POSIX_FADV_DONTNEED).

    Multi-GB checkpoint and export writes otherwise stay in the page cache, which is
    charged to the job's memory-capped systemd unit. Best effort: never raises.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        if sync:
            os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass
    finally:
        os.close(fd)


def drop_tree_cache(root: Path) -> None:
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            drop_page_cache(path)


def save_checkpoint(run: Path, payload: dict) -> Path:
    """Write tmp, fsync, drop its page cache, keep the old latest as previous, rename tmp
    to latest."""
    latest, previous, tmp = ckpt_paths(run)
    latest.parent.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "format": CKPT_FORMAT, "complete": True}
    with open(tmp, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    if latest.exists():
        os.replace(latest, previous)
    os.replace(tmp, latest)
    fsync_dir(latest.parent)
    return latest


class CheckpointUnreadable(RuntimeError):
    """Checkpoint files exist but none is a complete, readable checkpoint."""


def load_checkpoint(run: Path) -> dict | None:
    """Newest complete checkpoint (latest, else previous); None only when no checkpoint
    file exists. Raises CheckpointUnreadable when files exist but none loads: a run with
    progress on disk is never silently restarted from step 0.

    An unreadable latest.pt next to a good previous.pt is renamed latest.pt.unreadable-*
    so that the next save does not rotate the torn file over the good one.
    """
    latest, previous, tmp = ckpt_paths(run)
    if tmp.exists():
        tmp.unlink()  # a torn write; the files it would have replaced are intact
    failures = []
    for path in (latest, previous):
        if not path.exists():
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            drop_page_cache(path, sync=False)
            if not (payload.get("complete") and payload.get("format") == CKPT_FORMAT):
                raise ValueError(f"incomplete or format {payload.get('format')!r} != {CKPT_FORMAT}")
        except Exception as error:  # truncated or corrupt: fall back to the older one
            log(f"checkpoint {path} unreadable ({type(error).__name__}: {error}); trying older")
            failures.append(f"{path.name}: {type(error).__name__}: {error}")
            continue
        if path == previous and latest.exists():
            quarantine = latest.with_name(f"latest.pt.unreadable-{time.strftime('%Y%m%dT%H%M%S')}")
            os.replace(latest, quarantine)
            log(f"moved unreadable {latest.name} to {quarantine.name}; resuming from previous.pt")
        payload["_path"] = str(path)
        return payload
    if failures:
        raise CheckpointUnreadable(f"{run}: no readable checkpoint ({'; '.join(failures)})")
    return None


def check_checkpoints_readable(run: Path) -> None:
    """Fail fast (exit 77) before loading models if checkpoint files exist but none loads."""
    latest, previous, _ = ckpt_paths(run)
    if latest.exists() or previous.exists():
        payload = load_for_resume(run)
        del payload


def comparable_args(args) -> dict:
    values = vars(args) if isinstance(args, argparse.Namespace) else dict(args)
    return {k: v for k, v in values.items() if k not in RESUME_ALLOWED_CHANGES}


def resume_mismatches(recorded: dict, args) -> list[str]:
    """Differences between the checkpoint's recorded invocation and this one, other than
    RESUME_ALLOWED_CHANGES."""
    old, new = comparable_args(recorded), comparable_args(args)
    return [f"--{k.replace('_', '-')} {old.get(k)!r} -> {new.get(k)!r}"
            for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)]


def load_resume_payload(run: Path, args) -> dict | None:
    """The checkpoint to resume (None: fresh start), loaded once, before any model.
    Exits 77 if checkpoint files are unreadable and 78 if this invocation's arguments
    differ from the checkpoint's (except RESUME_ALLOWED_CHANGES)."""
    payload = load_for_resume(run)
    if payload is None:
        return None
    recorded = payload.get("args")
    problems = (resume_mismatches(recorded, args) if recorded is not None else
                [f"{k}: {payload['resume_key'].get(k)!r} -> {v!r}"
                 for k, v in resume_key(args).items() if payload["resume_key"].get(k) != v])
    if problems:
        log(f"refusing to resume {run}: the arguments differ from the checkpoint's: "
            f"{'; '.join(problems)}. Rerun with the original arguments.")
        sys.exit(EXIT_RESUME_MISMATCH)
    return payload


def load_for_resume(run: Path) -> dict | None:
    """load_checkpoint, exiting 77 (never starting over) if checkpoint files are unreadable."""
    try:
        return load_checkpoint(run)
    except CheckpointUnreadable as error:
        log(f"refusing to start: {error}. Inspect {run / 'ckpt'}; rerun with --overwrite only "
            "to discard this run.")
        sys.exit(EXIT_UNREADABLE_CHECKPOINT)


def resume_key(args) -> dict:
    return {k: getattr(args, k) for k in RESUME_KEYS}


# ----------------------------------------------------------------------------- training

def _new_acc() -> dict:
    return {"n": 0, "loss": 0.0, "tdt": 0.0, "enc": 0.0, "norm": 0.0, "seconds": 0.0,
            "rows": 0, "dropped_rows": 0, "data_wait_s": 0.0, "label_s": 0.0}


@dataclass
class TrainState:
    step: int = 0
    batches_read: int = 0  # including batches whose rows were all dropped
    skipped_batches: int = 0
    audio_seconds_seen: float = 0.0  # kept (trained-on) audio
    audio_seconds_read: float = 0.0
    seconds_by_source: dict = field(default_factory=dict)
    drop_counts: dict = field(default_factory=dict)
    position: dict | None = None  # stream position after the last consumed batch
    best_wer: float = math.inf
    best_step: int = -1
    best_file: str | None = None  # --select best only: weights of the best f == 1 evaluation
    best_subset: dict = field(default_factory=dict)
    final_subset: dict = field(default_factory=dict)  # dev-subset WERs at step == max_steps
    metrics_offset: int = 0
    evals: int = 0
    session: int = 0  # 0 for the first process, +1 per resume; tags every metrics line
    acc: dict = field(default_factory=_new_acc)


class Context:
    """Everything the loop needs; built by main() or by the tests."""

    def __init__(self, args, run: Path, student, teacher, stream, evaluator, quant=None,
                 device: str = "cuda", label_fn=None) -> None:
        self.args, self.run, self.student, self.teacher = args, run, student, teacher
        self.stream, self.evaluator, self.quant, self.device = stream, evaluator, quant, device
        self.label_fn = label_fn or local_module("teacher").teacher_label_batch
        self.tdt_kernel_sha256 = prepare_tdt_loss()
        self.recipe = recipe_of(args)
        self.ramp = ramp_steps(args.max_steps) if self.recipe["quantized"] else None
        self.optimizer, self.scheduler = make_optimizer(student, args.lr, args.max_steps)
        self.state = TrainState()
        self.stop_reason: str | None = None  # set by the SIGTERM/SIGINT handler
        self.last_ckpt = time.monotonic()
        self.latest_best_ref: str | None = None  # best file referenced by ckpt/latest.pt

    def fraction(self) -> float | None:
        return None if self.ramp is None else ramp_weight_fraction(max(1, self.state.step), self.ramp)

    def checkpoint_payload(self) -> dict:
        return {"step": self.state.step, "student": self.student.state_dict(),
                "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                "rng": rng_state(self.student), "state": asdict(self.state),
                "weight_fraction": self.fraction(), "ramp_steps": self.ramp,
                "resume_key": resume_key(self.args), "args": vars(self.args)}

    def save(self) -> None:
        t = time.perf_counter()
        path = save_checkpoint(self.run, self.checkpoint_payload())
        self.last_ckpt = time.monotonic()
        # Durable now: latest references state.best_file and previous references what latest
        # referenced before. Best files neither references can go.
        keep = {self.state.best_file, self.latest_best_ref}
        self.latest_best_ref = self.state.best_file
        collect_best_files(self.run, keep)
        log(f"checkpoint step {self.state.step} -> {path} ({time.perf_counter() - t:.1f}s)")

    def restore(self, payload: dict) -> None:
        if payload["resume_key"] != resume_key(self.args):
            raise RuntimeError(f"checkpoint {payload['_path']} was written by a different "
                               f"configuration: {payload['resume_key']} != {resume_key(self.args)}")
        self.student.load_state_dict(payload["student"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.state = TrainState(**payload["state"])
        if self.ramp is not None and self.state.step > 0:
            self.quant.set_weight_fraction(self.student, self.fraction())
        set_rng_state(self.student, payload["rng"])
        self.state.session += 1
        self.latest_best_ref = self.state.best_file
        self.state.metrics_offset = prune_evidence(self.run, self.state.step,
                                                   self.state.best_file)


def record_resume(run: Path, entry: dict) -> None:
    """Append a resume record; a torn last line from a power cut is dropped first."""
    path = run / "resumes.jsonl"
    records, torn = read_log_jsonl(path)
    if torn:
        rewrite_jsonl(path, records)
    append_jsonl(path, entry)


def collect_best_files(run: Path, keep: set) -> None:
    """Delete best-*.pt files (and temp files) not in `keep`."""
    for path in run.glob("best-*.pt*"):
        if path.name not in keep:
            path.unlink()


def best_file_step(name: str) -> int | None:
    try:
        return int(name.split("-")[1].split(".")[0])
    except (IndexError, ValueError):
        return None


def prune_evidence(run: Path, step: int, best_file: str | None) -> int:
    """Make the run directory agree with a checkpoint at `step`; returns the new
    metrics.jsonl size.

    metrics.jsonl is rewritten atomically (temp file, fsync, rename) keeping only the
    lines with step <= the checkpoint step, in order; lines after it (written by a session
    that died before its next checkpoint) and a torn last line are dropped, since the
    resumed run writes those steps again. dev-subset evidence files for later steps are
    deleted, and so are best-*.pt files written after the checkpoint (unreferenced by any
    durable checkpoint) and temp files; older best files may still be referenced by
    ckpt/previous.pt and are left for the next durable checkpoint's collection.
    """
    metrics = run / "metrics.jsonl"
    size = 0
    if metrics.exists():
        records, _ = read_log_jsonl(metrics)
        size = rewrite_jsonl(metrics, [r for r in records if r.get("step", math.inf) <= step])
    resumes = run / "resumes.jsonl"
    if resumes.exists():
        records, torn = read_log_jsonl(resumes)
        if torn:
            rewrite_jsonl(resumes, records)
    for path in (run / "dev-subset").glob("*/step-*.json*"):
        try:
            if int(path.name.split("-")[1].split(".")[0]) > step:
                path.unlink()
        except ValueError:
            path.unlink()  # a stray temp file
    for path in run.glob("best-*.pt*"):
        file_step = best_file_step(path.name)
        if path.name != best_file and (path.name.endswith(".tmp") or file_step is None
                                       or file_step > step):
            path.unlink()
    return size


def write_metrics(ctx: Context, record: dict) -> None:
    record = {**record, "session": ctx.state.session}
    ctx.state.metrics_offset = append_jsonl(ctx.run / "metrics.jsonl", record)


def run_eval(ctx: Context) -> None:
    """Dev-subset evaluation at the current step and weight fraction; may update best-*.pt."""
    st, f = ctx.state, ctx.fraction()
    t = time.perf_counter()
    with isolated_rng(ctx.student), torch.no_grad():
        ctx.student.eval()
        results = ctx.evaluator.subset(ctx.student)
        set_train_modes(ctx.student, ctx.recipe)
    eligible = selectable(f)
    score = mean_wer(results)
    for name, r in results.items():
        write_json(ctx.run / "dev-subset" / name / f"step-{st.step:06d}.json",
                   {"step": st.step, "set": name, "weight_fraction": f, "selectable": eligible,
                    "wer": r["wer"], "utterances": r["utterances"], "records": r["records"]})
    improved = eligible and score < st.best_wer
    if improved:  # tracked for information under --select final
        st.best_wer, st.best_step = score, st.step
        st.best_subset = {n: r["wer"] for n, r in results.items()}
    if improved and ctx.args.select == "best":
        # A new name every time; the file the durable checkpoints reference is never
        # touched here (Context.save collects unreferenced files once a checkpoint is durable).
        name = f"best-{st.step:06d}.pt"
        tmp = ctx.run / (name + ".tmp")
        # "model" + "weight_fraction": the layout evaluate.py --source checkpoint reads.
        torch.save({"model": ctx.student.state_dict(), "step": st.step,
                    "dev_subset_mean_wer": score, "weight_fraction": f,
                    "arm": ctx.args.arm, "recipe": ctx.args.recipe}, tmp)
        drop_page_cache(tmp)
        os.replace(tmp, ctx.run / name)
        fsync_dir(ctx.run)
        st.best_file = name
    if st.step == ctx.args.max_steps:
        st.final_subset = {n: r["wer"] for n, r in results.items()}
    st.evals += 1
    write_metrics(ctx, {"step": st.step, "eval": "dev-subset",
                        "dev_subset_wer": {n: r["wer"] for n, r in results.items()},
                        "dev_subset_mean_wer": score, "weight_fraction": f,
                        "selectable": eligible, "improved": improved,
                        "best_dev_subset_mean_wer": st.best_wer if math.isfinite(st.best_wer) else None,
                        "best_step": st.best_step, "eval_s": time.perf_counter() - t})
    note = "" if eligible else f" (not selectable at f={f:.4f})"
    log(f"eval step {st.step}: dev-subset mean WER {100 * score:.2f}%{note}")


def label_batch(ctx: Context, batch: dict) -> dict | None:
    """Teacher-label a collated batch on the GPU; None if every row was dropped."""
    st = ctx.state
    audio = batch["audio"].to(ctx.device, non_blocking=True)
    audio_len = batch["audio_len"].to(ctx.device, non_blocking=True)
    texts, keep, enc, enc_len = ctx.label_fn(ctx.teacher, audio, audio_len, batch["texts"],
                                             TEACHER_PRECISION, counts=st.drop_counts)
    keep_list = keep.tolist()
    st.acc["dropped_rows"] += keep_list.count(False)
    if not any(keep_list):
        return None
    rows = torch.nonzero(keep).flatten()
    texts = [t for t, k in zip(texts, keep_list) if k]
    seconds = 0.0
    for k, n, source in zip(keep_list, batch["audio_len"].tolist(), batch["sources"]):
        if k:
            seconds += n / paths.SAMPLE_RATE
            st.seconds_by_source[source] = st.seconds_by_source.get(source, 0.0) + n / paths.SAMPLE_RATE
    tokens, token_len = tokenize(ctx.student, texts, ctx.device)
    # Rows are dropped but the padded length is kept, so student and teacher frames align.
    return {"audio": audio[rows], "audio_len": audio_len[rows], "tokens": tokens,
            "token_len": token_len, "teacher_enc": enc[rows], "teacher_len": enc_len[rows],
            "seconds": seconds, "rows": len(texts)}


def train_step(ctx: Context, labeled: dict, trainable: list) -> dict:
    loss, tdt, enc = compute_losses(ctx.student, labeled["audio"], labeled["audio_len"],
                                    labeled["tokens"], labeled["token_len"], ctx.recipe,
                                    labeled["teacher_enc"], labeled["teacher_len"])
    ctx.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    ctx.optimizer.step()
    ctx.scheduler.step()
    # One host sync per step: exact Python-float accumulators, and a non-finite loss is
    # caught at the step where it occurs.
    values = {"loss": loss.item(), "tdt": tdt.item(), "enc": 0.0 if enc is None else enc.item(),
              "norm": norm.item()}
    if not all(math.isfinite(v) for v in values.values()):
        raise RuntimeError(f"non-finite loss or gradient norm at step {ctx.state.step + 1}: {values}")
    return values


def train_loop(ctx: Context, stop_after: int | None = None) -> dict:
    """Optimize from ctx.state.step to max_steps; checkpoints on time, signal, stream
    failure, and at the end. `stop_after` (tests) checkpoints and returns once that many
    total steps are done, exactly as a SIGTERM at that step boundary would."""
    args, st, student = ctx.args, ctx.state, ctx.student
    trainable = [p for group in ctx.optimizer.param_groups for p in group["params"]]
    batcher = Batcher(ctx.stream, args.batch_seconds, args.seed, st.position)
    mark, mark_step = time.perf_counter(), st.step
    start_step, t0 = st.step, time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    set_train_modes(student, ctx.recipe)
    try:
        while st.step < args.max_steps:
            t_wait = time.perf_counter()
            try:
                batch = batcher.next()
            except StreamFailure as error:
                log(f"stream failure at step {st.step}: {error}; checkpointing")
                ctx.save()
                raise
            st.acc["data_wait_s"] += time.perf_counter() - t_wait
            st.batches_read += 1
            st.audio_seconds_read += batch["seconds"]
            t_label = time.perf_counter()
            labeled = label_batch(ctx, batch)
            st.acc["label_s"] += time.perf_counter() - t_label
            st.position = batch["position_after"]
            if labeled is None:
                st.skipped_batches += 1
                continue
            reseed_cudnn_rnn_dropout()
            if ctx.ramp is not None:
                ctx.quant.set_weight_fraction(student, ramp_weight_fraction(st.step + 1, ctx.ramp))
            lr = ctx.optimizer.param_groups[0]["lr"]
            values = train_step(ctx, labeled, trainable)
            st.step += 1
            st.audio_seconds_seen += labeled["seconds"]
            for key, value in values.items():
                st.acc[key] += value
            st.acc["n"] += 1
            st.acc["seconds"] += labeled["seconds"]
            st.acc["rows"] += labeled["rows"]
            del labeled, batch
            if st.step % LOG_EVERY == 0 or st.step == args.max_steps:
                flush_metrics(ctx, lr, (time.perf_counter() - mark) / (st.step - mark_step))
                mark, mark_step = time.perf_counter(), st.step
            if st.step % args.eval_every == 0 or st.step == args.max_steps:
                run_eval(ctx)
                mark, mark_step = time.perf_counter(), st.step
            due = time.monotonic() - ctx.last_ckpt >= args.ckpt_minutes * 60
            if due or ctx.stop_reason or st.step == args.max_steps or st.step == stop_after:
                ctx.save()
                mark, mark_step = time.perf_counter(), st.step
            if ctx.stop_reason or st.step == stop_after:
                break
    finally:
        batcher.close()
    return {"train_time_s": time.perf_counter() - t0, "steps_this_session": st.step - start_step}


def flush_metrics(ctx: Context, lr: float, step_time: float) -> None:
    st = ctx.state
    n = st.acc["n"]
    if n == 0:
        return
    values = {k: st.acc[k] / n for k in ("loss", "tdt", "enc", "norm")}
    record = {"step": st.step, "loss": values["loss"], "tdt": values["tdt"],
              "enc_mse": values["enc"] if ctx.recipe["encoder_matching"] else None,
              "lr": lr, "grad_norm": values["norm"], "weight_fraction": ctx.fraction(),
              "audio_seconds_seen": st.audio_seconds_seen, "audio_seconds_read": st.audio_seconds_read,
              "audio_seconds_per_step": st.acc["seconds"] / n, "rows_per_step": st.acc["rows"] / n,
              "dropped_rows": st.acc["dropped_rows"], "skipped_batches": st.skipped_batches,
              "step_time_s": step_time, "data_wait_s_per_step": st.acc["data_wait_s"] / n,
              "label_s_per_step": st.acc["label_s"] / n,
              "audio_seconds_per_second": st.acc["seconds"] / n / step_time if step_time else None,
              "drop_counts": dict(st.drop_counts), "seconds_by_source": dict(st.seconds_by_source)}
    if torch.cuda.is_available():
        record["gpu_max_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        record["gpu_max_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        torch.cuda.reset_peak_memory_stats()
    record["utc"] = utc()
    write_metrics(ctx, record)
    st.acc = _new_acc()
    if st.step % (5 * LOG_EVERY) == 0 or st.step == ctx.args.max_steps:
        enc = "" if record["enc_mse"] is None else f" enc {record['enc_mse']:.4f}"
        f = "" if record["weight_fraction"] is None else f" f {record['weight_fraction']:.3f}"
        log(f"step {st.step}/{ctx.args.max_steps} loss {record['loss']:.4f} tdt "
            f"{record['tdt']:.4f}{enc}{f} lr {lr:.2e} gnorm {record['grad_norm']:.2f} "
            f"{step_time:.2f}s/step (data wait {record['data_wait_s_per_step']:.2f}s, label "
            f"{record['label_s_per_step']:.2f}s) {record['audio_seconds_per_second'] or 0:.0f} "
            f"audio-s/s")


# ----------------------------------------------------------------------------- probe

def run_probe(ctx: Context, steps: int) -> dict:
    """End-to-end throughput (stream wait + teacher labeling + student update) and memory.

    Two untimed worst-case updates first: utterances concatenated to about 30 s (the
    transducer joint grows with duration x tokens), then `steps` timed updates on the
    configured stream. Timings are per phase, with a GPU sync at each boundary.
    """
    args, student = ctx.args, ctx.student
    trainable = [p for group in ctx.optimizer.param_groups for p in group["params"]]
    set_train_modes(student, ctx.recipe)
    if ctx.ramp is not None:
        ctx.quant.set_weight_fraction(student, 0.5)  # mid-ramp: lerp(W, W_hat) path
    torch.cuda.reset_peak_memory_stats()
    phases = {"data_wait_s": [], "label_s": [], "student_s": []}
    seconds, rows, status, peaks = [], [], {}, {}
    worst = LocalLibriSpeechStream(args.seed, concat_seconds=MAX_SECONDS - 0.5)
    for name, stream, count in (("worst", worst, 2), ("timed", ctx.stream, steps)):
        batcher = Batcher(stream, args.batch_seconds, args.seed)
        status[name] = "ok"
        torch.cuda.reset_peak_memory_stats()
        try:
            for j in range(count):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                batch = batcher.next()
                t1 = time.perf_counter()
                labeled = label_batch(ctx, batch)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                if labeled is None:
                    continue
                train_step(ctx, labeled, trainable)
                torch.cuda.synchronize()
                t3 = time.perf_counter()
                if name == "timed" and j >= 1:  # first timed step warms up allocator shapes
                    phases["data_wait_s"].append(t1 - t0)
                    phases["label_s"].append(t2 - t1)
                    phases["student_s"].append(t3 - t2)
                    seconds.append(labeled["seconds"])
                    rows.append(labeled["rows"])
                del labeled, batch
        except torch.OutOfMemoryError:
            status[name] = f"oom at step {j}"
            labeled = batch = None
            ctx.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        finally:
            batcher.close()
            peaks[name] = {"allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                           "reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
    ctx.optimizer.zero_grad(set_to_none=True)
    result = {"arm": args.arm, "recipe": args.recipe, "batch_seconds": args.batch_seconds,
              "grad_checkpointing": args.grad_checkpointing, "stream": args.stream,
              "status": status, "peak_memory": peaks, "timed_steps": len(seconds),
              "drop_counts": dict(ctx.state.drop_counts)}
    if seconds:
        total = sum(sum(v) for v in phases.values())
        result.update({k + "_mean": sum(v) / len(v) for k, v in phases.items()})
        result.update(step_time_s=total / len(seconds),
                      audio_seconds_per_step=sum(seconds) / len(seconds),
                      rows_per_step=sum(rows) / len(rows),
                      audio_hours_per_gpu_hour=sum(seconds) / total,
                      student_only_audio_hours_per_gpu_hour=sum(seconds) / sum(phases["student_s"]),
                      label_only_audio_hours_per_gpu_hour=sum(seconds) / sum(phases["label_s"]))
    return result


# ----------------------------------------------------------------------------- finish

RECONSTRUCTION_UTTERANCES = 4


def reconstruction(export, evaluator, model, scorer) -> dict:
    """Export reconstruction check (exact codes and scales, encoder/joint/greedy agreement) on
    the first utterances of the first dev subset; both models in eval mode on one device."""
    records = next(iter(evaluator.subsets.values()))[:RECONSTRUCTION_UTTERANCES]
    batch = evaluator.evaluate.audio_batch(records)
    model.eval()
    scorer.eval()
    result = export.reconstruction_check(model, scorer, batch)
    result["utterance_ids"] = [r["id"] for r in records]
    return result


def selected_model(ctx: Context) -> dict:
    """Load the selected weights into ctx.student; returns the selection record.

    final: the model as of step == max_steps (in memory after training, or restored from
    the final checkpoint after a crash during finishing), which must be at f == 1.
    best: the best-*.pt of the lowest f == 1 dev-subset mean WER.
    """
    st, args, run = ctx.state, ctx.args, ctx.run
    info = {"best_interim_subset_step": st.best_step,
            "best_interim_subset_mean_wer": st.best_wer if math.isfinite(st.best_wer) else None,
            "best_interim_subset_wer": st.best_subset}
    if args.select == "final":
        if st.step != args.max_steps or not selectable(ctx.fraction()):
            raise RuntimeError(f"final model not ready: step {st.step}/{args.max_steps}, "
                               f"f = {ctx.fraction()}")
        if not st.final_subset:
            raise RuntimeError("no dev-subset evaluation at the final step")
        return {"select": "final", "selected_step": st.step,
                "dev_subset_mean_wer": mean_wer({n: {"wer": w} for n, w in st.final_subset.items()}),
                "selected_dev_subset_wer": st.final_subset, **info}
    if not st.best_file or not (run / st.best_file).exists():
        raise RuntimeError(f"best checkpoint {st.best_file!r} missing: no selectable (f == 1) "
                           "evaluation, or the file was lost")
    best = torch.load(run / st.best_file, map_location="cpu", weights_only=False)
    drop_page_cache(run / st.best_file, sync=False)
    if not selectable(best["weight_fraction"]):
        raise RuntimeError(f"best checkpoint was scored at weight fraction {best['weight_fraction']}")
    ctx.student.load_state_dict(best["model"])
    return {"select": "best", "selected_step": int(best["step"]),
            "dev_subset_mean_wer": best["dev_subset_mean_wer"],
            "selected_dev_subset_wer": st.best_subset, **info}


def finish(ctx: Context, config_sha: str) -> dict:
    """Load the selected model, write the deployable artifact, rebuild it, score full dev."""
    st, args, run = ctx.state, ctx.args, ctx.run
    result = selected_model(ctx)
    model = ctx.student
    model.eval()
    if ctx.recipe["quantized"]:
        export = local_module("export")
        ctx.quant.set_weight_fraction(model, 1.0)  # the export refuses f < 1
        extra = {"run_name": args.run_name, "arm": args.arm, "recipe": args.recipe, "lr": args.lr,
                 "select": result["select"], "selected_step": result["selected_step"],
                 "dev_subset_mean_wer": result["dev_subset_mean_wer"],
                 "config_sha256": config_sha, "source_hashes": source_hashes()}
        manifest = export.export_model(model, run / "export", extra)
        drop_tree_cache(run / "export")
        scorer = export.load_export(run / "export", ctx.device)
        check = reconstruction(export, ctx.evaluator, model, scorer)
        write_json(run / "export" / "reconstruction.json", check)
        result.update(export_dir=str(run / "export"), export_bytes=manifest.get("bytes"),
                      code_histogram={k: v for k, v in ctx.quant.code_histogram(model).items()
                                      if k != "per_module"},
                      reconstruction=check, scored_on="model rebuilt from export")
    else:
        model.save_to(str(run / "best.nemo"))
        drop_page_cache(run / "best.nemo")
        scorer = load_nemo(run / "best.nemo", ctx.device)
        result.update(nemo=str(run / "best.nemo"), nemo_sha256=sha256(run / "best.nemo"),
                      scored_on="model restored from best.nemo")
    scorer.eval()
    with torch.no_grad():
        full = ctx.evaluator.full(scorer)
    for name, r in full.items():
        write_json(run / "eval-dev" / f"{name}.json",
                   {"set": name, "wer": r["wer"], "utterances": r["utterances"],
                    "records": r["records"], "source": result["scored_on"]})
    evaluation = {"run_name": args.run_name, "dev_wer": {n: r["wer"] for n, r in full.items()},
                  "dev_utterances": {n: r["utterances"] for n, r in full.items()},
                  "dev_mean_wer": mean_wer(full), "limit": args.dev_limit,
                  "select": result["select"], "selected_step": int(result["selected_step"]),
                  "source": result["scored_on"]}
    write_json(run / "eval-dev.json", evaluation)
    result.update(dev_wer=evaluation["dev_wer"], dev_mean_wer=evaluation["dev_mean_wer"],
                  dev_utterances=evaluation["dev_utterances"])
    return result


def load_nemo(path: Path, device: str):
    from nemo.collections.asr.models import ASRModel
    return ASRModel.restore_from(str(path), map_location=device).float()


# ----------------------------------------------------------------------------- provenance

def git_head() -> str:
    out = subprocess.run(["git", "-C", str(paths.REPO), "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def source_hashes() -> dict[str, str]:
    """SHA-256 of every *.py next to this file (provenance)."""
    return {p.name: sha256(p) for p in sorted(paths.HERE.glob("*.py"))}


def training_source_hashes() -> dict[str, str]:
    """SHA-256 of TRAINING_SOURCES (what sweep.py validates); a missing file is an error."""
    root = paths.HERE.parent
    return {name: sha256(root / name) for name in TRAINING_SOURCES}


def versions() -> dict:
    import nemo
    import numba
    return {"python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "nemo": nemo.__version__, "numba": numba.__version__,
            "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}


def clear_outputs(run: Path, include_checkpoints: bool) -> None:
    names = list(OUTPUTS) + (["ckpt"] if include_checkpoints else [])
    for name in names:
        path = run / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    for path in run.glob("best-*.pt*"):
        path.unlink()


# ----------------------------------------------------------------------------- power log

POWERLOG_LAUNCH = ("import sys, paths, powerlog; powerlog.POWER_DIR = paths.POWER; "
                   "paths._RUN_NAME = paths._NAME; powerlog.main(sys.argv[1:])")


def start_powerlog(name: str) -> subprocess.Popen | None:
    """powerlog.py's recorder writing to paths.POWER (its own POWER_DIR is the Whisper runs
    directory, and running it as a script would import the Whisper paths.py)."""
    try:
        paths.POWER.mkdir(parents=True, exist_ok=True)
        cmd = [str(paths.HERE / "python"), "-c", POWERLOG_LAUNCH, "record", "--name", name,
               "--parent-pid", str(os.getpid())]
        with open(paths.POWER / f"{name}.log", "ab") as out:
            return subprocess.Popen(cmd, cwd=paths.HERE, stdin=subprocess.DEVNULL, stdout=out,
                                    stderr=subprocess.STDOUT, start_new_session=True)
    except Exception as error:  # the power log must never block the experiment
        log(f"warning: power recorder not started ({type(error).__name__}: {error})")
        return None


def stop_powerlog(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        import powerlog
        powerlog.stop_recorder(proc)
    except Exception as error:
        log(f"warning: stopping the power recorder failed ({type(error).__name__}: {error})")


# ----------------------------------------------------------------------------- main

def build_context(args, run: Path, device: str = "cuda") -> tuple[Context, dict]:
    """Load models, quantizer, stream and evaluator (inside the GPU lock)."""
    recipe = recipe_of(args)
    base = verify_base()
    student = load_pretrained("cpu")
    seed_everything(args.seed, student)
    quant, quantized = None, []
    if recipe["quantized"]:
        quant = local_module("quant")
        quantized = list(quant.quantize_parakeet(student))
    params = configure_trainable(student, recipe)
    params["verified"] = verify_recipe(student, recipe, quant)
    student.to(device)
    if args.grad_checkpointing:
        enable_grad_checkpointing(student.encoder)
    teacher = load_teacher(load_pretrained(device))
    evaluator = None if args.probe_steps else DevEvaluator(args.dev_limit, args.eval_batch_size,
                                                           device)
    ctx = Context(args, run, student, teacher, open_stream(args), evaluator, quant, device)
    info = {"base": base, "quantized_modules": quantized,
            "quantized_modules_sha256": (hashlib.sha256(json.dumps(sorted(quantized)).encode())
                                         .hexdigest() if quantized else None),
            "trainable": params,
            "parameter_accounting": quant.parameter_accounting(student) if quant else None}
    return ctx, info


def heavy_unit() -> str | None:
    """The parakeet-* systemd unit this process runs in (./heavy), or None."""
    try:
        text = Path("/proc/self/cgroup").read_text()
    except OSError:
        return None
    for part in text.replace("\n", "/").split("/"):
        if part.startswith("parakeet-") and part.endswith(".service"):
            return part
    return None


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    started_utc = utc()
    if heavy_unit() is None:
        log("warning: not inside a parakeet-* systemd unit; launch GPU jobs with ./heavy "
            "(README.md 'Training')")
    paths.require_mount()
    if not torch.cuda.is_available():
        sys.exit("CUDA is required (the TDT loss is CUDA-only)")
    run = paths.run_dir(args.run_name)
    if (run / "summary.json").exists() and not args.overwrite and not args.probe_steps:
        sys.exit(f"run {args.run_name} already has summary.json; pass --overwrite to replace it")
    if args.overwrite:
        clear_outputs(run, include_checkpoints=True)
    payload = None if args.probe_steps else load_resume_payload(run, args)
    with paths.gpu_lock(f"train {args.run_name}"):
        recorder = None if args.no_powerlog else start_powerlog(f"train-{args.run_name}")
        try:
            determinism = deterministic_mode()
            ctx, info = build_context(args, run)
            if args.probe_steps:
                result = {**run_probe(ctx, args.probe_steps), "run_name": args.run_name,
                          "determinism": determinism, "utc": utc(), "versions": versions()}
                write_json(run / "probe.json", result)
                log(json.dumps({k: v for k, v in result.items() if k != "versions"}))
                return
            if payload is None:
                if (run / "config.json").exists():
                    log(f"{args.run_name}: started before but no checkpoint exists; starting at "
                        "step 0 (no progress was saved)")
                clear_outputs(run, include_checkpoints=False)
                config = {"args": vars(args), "recipe": recipe_of(args),
                          "started_utc": started_utc, "git_head": git_head(),
                          "source_hashes": source_hashes(),
                          "training_source_hashes": training_source_hashes(),
                          "versions": versions(), "determinism": determinism, "model_file": str(paths.MODEL_FILE),
                          "numba_cuda": {**NUMBA_CUDA,
                                         "tdt_grad_kernel_patched_sha256": ctx.tdt_kernel_sha256},
                          "stream": stream_identity(args, ctx.stream),
                          "stream_state_at_start": ctx.stream.state_dict(),
                          "bucketing": {"pool_batches": POOL_BATCHES,
                                        "padded_batch_seconds": args.batch_seconds,
                                        "prefilter_seconds": [MIN_SECONDS, MAX_SECONDS]},
                          "teacher": {"labels": "teacher.teacher_label_batch (greedy, online)",
                                      "precision": TEACHER_PRECISION,
                                      "encoder_matching_input": "clean audio, no dither, no "
                                                                "SpecAugment"},
                          "ramp_steps": ctx.ramp, "warmup_steps": warmup_steps(args.max_steps),
                          "encoder_matching_weight": ENCODER_MATCHING_WEIGHT, **info}
                write_json(run / "config.json", config)
                log(f"{args.run_name}: arm {args.arm} recipe {args.recipe} lr {args.lr:g}, "
                    f"{len(info['quantized_modules'])} ternary modules, trainable "
                    f"{info['trainable']['trainable']:,}, stream {stream_identity(args, ctx.stream)}")
            else:
                ctx.restore(payload)
                log(f"resumed from {payload['_path']} at step {ctx.state.step}")
                record_resume(run, {"utc": utc(), "step": ctx.state.step,
                                    "checkpoint": payload["_path"],
                                    "position": ctx.state.position, "git_head": git_head(),
                                    "source_hashes": source_hashes(),
                                    "training_source_hashes": training_source_hashes(),
                                    "args": vars(args)})
                del payload

            def on_signal(num, _frame):
                if ctx.stop_reason is None:
                    ctx.stop_reason = signal.Signals(num).name
                    log(f"received {ctx.stop_reason}; checkpointing at the next step boundary")

            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, on_signal)
            stats = {"train_time_s": 0.0}
            if ctx.state.step < args.max_steps:
                try:
                    stats = train_loop(ctx)
                except StreamFailure:
                    sys.exit(EXIT_STREAM_FAILURE)
            if ctx.state.step < args.max_steps:
                log(f"stopped at step {ctx.state.step} ({ctx.stop_reason}); rerun to resume")
                sys.exit(EXIT_INTERRUPTED)
            ctx.teacher = None
            del ctx.optimizer
            torch.cuda.empty_cache()
            config_sha = sha256(run / "config.json")
            result = finish(ctx, config_sha)
            resumes = len(read_log_jsonl(run / "resumes.jsonl")[0])
            config = json.loads((run / "config.json").read_text())
            st = ctx.state
            summary = {"run_name": args.run_name, "arm": args.arm, "recipe": args.recipe,
                       "lr": args.lr, "max_steps": args.max_steps, "steps": st.step,
                       "seed": args.seed, "batch_seconds": args.batch_seconds, "train": args.train,
                       "stream": config["stream"], **recipe_of(args),
                       "grad_checkpointing": args.grad_checkpointing,
                       "eval_every": args.eval_every,
                       "source_hashes": training_source_hashes(),  # the finishing session's
                       "quantized_modules_sha256": config.get("quantized_modules_sha256"),
                       "ramp_steps": ctx.ramp or 0, "warmup_steps": warmup_steps(args.max_steps),
                       "audio_seconds_seen": st.audio_seconds_seen,
                       "audio_hours_seen": st.audio_seconds_seen / 3600,
                       "audio_seconds_read": st.audio_seconds_read,
                       "seconds_by_source": st.seconds_by_source, "drop_counts": st.drop_counts,
                       "batches_read": st.batches_read, "skipped_batches": st.skipped_batches,
                       "evals": st.evals, "resumes": resumes, "smoke": bool(args.smoke),
                       "dev_limit": args.dev_limit, "config_sha256": config_sha,
                       "last_session_train_time_s": stats["train_time_s"],
                       "max_memory_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                       **result, "started_utc": config["started_utc"], "finished_utc": utc()}
            wrong = [k for k, t in SUMMARY_TYPES.items() if type(summary.get(k)) is not t]
            if wrong:
                raise TypeError(f"summary.json fields missing or mistyped: {wrong}")
            write_json(run / "summary.json", summary)  # last write
            previous = ckpt_paths(run)[1]
            if previous.exists():
                previous.unlink()
            log(f"{args.run_name}: selected ({result['select']}) step {result['selected_step']}, "
                f"dev-subset mean WER "
                f"{100 * result['dev_subset_mean_wer']:.2f}%, full dev mean WER "
                f"{100 * result['dev_mean_wer']:.2f}%")
        finally:
            stop_powerlog(recorder)


if __name__ == "__main__":
    main()
