"""Stage 2b: decoder-only fine-tuning of Kea-B (MB1) and Kea-T (M1) on the new labels (decoder-only plan).

    ./python train_decoder.py split                                    freeze the dev split (CPU; before any run)
    ./python train_decoder.py equivalence --student B|T                encoder-equivalence check, 20 dev clips (GPU)
    ./python train_decoder.py probe --student B|T                      100-step probe: peak VRAM, step time (GPU)
    ./python train_decoder.py train --arm B-T --lr 1e-5 --labels-run RUN [--budget-minutes 110]
                                                                       one chunk of one run (GPU; resumable)
    ./python train_decoder.py score-original --student B|T             the unchanged student's scores (GPU)
    ./python train_decoder.py chunk --labels-run RUN [--budget-minutes 110]
                                                                       one chunk job: the next unfinished items in
                                                                       ORDER, each a subprocess, within the budget
    ./python train_decoder.py status | report [--out PATH]

Arms (ARMS): B-T and B-0 train Kea-B's decoder, T-T and T-0 Kea-T's, on T* labels (`text`) or on B0's labels
(`b0_raw`) under T*'s disposition mask (training/labels.py). Each arm runs at the three peak learning rates LRS:
12 runs, $RELABEL_ROOT/decoder-only/<arm>/lr<lr>/.

Training is finetune/parakeet-ternary/train.py's loop (train.main: checkpoints, resume, signals, metrics,
summary.json), unchanged, inside seam(): training/seam.py's offline-label replacements (the batch plan's pools
0-155, re-cut from the stage 2 journal (consumed batches 0-19,967; decoder-only plan "Data", 2026-10-24),
read from the materialized pool audio; the label lookup in place of the online teacher; the WSD learning rate per
planned batch: warmup 2% of the planned batches, stable through pool 132, cooldown over pools 133-155) and this
module's:

1. train.parse_args: the run's arguments (train.py's fields plus the student, label set, label run, stop batch,
   schedule and the hashes of the labels, plan, replay, export, checkpoint and dev split); max_steps is the number
   of planned batches with a labeled row (fully rejected batches are consumed without a step), so train_loop ends
   exactly at the last batch with a step. RESUME_KEYS gains the new fields (a checkpoint of another
   configuration is refused, exit 78); RESUME_ALLOWED_CHANGES gains budget_minutes.
2. train.build_context: the student's model (build_student): the student's reconstructed export (score_domain's
   loader, pinned SHA-256) with dequantized constant encoder weights and no quantization-aware machinery; its
   prediction and joint networks replaced by the final checkpoint's FP32 tensors (which must round to the
   export's FP16 tensors exactly; the MB1 checkpoint's SHA-256 is recorded at its first use, M1's must be
   binary_paths.M1_SHA256). Only prediction and joint parameters require gradients, so only they are in AdamW
   (FP32 parameters and state; their arithmetic runs under train.compute_losses' BF16 autocast); their dropout is
   0; SpecAugment and the training-mode preprocessor stay on. Before training, in every session (first start and
   every resume), the encoder-equivalence check compares the configured model's encoder output with a freshly
   loaded export's on 20 dev clips (strict FP32); a failure stops the run. A budget timer requests a checkpoint and
   stop after --budget-minutes (exit 0 for a budget stop).
3. train.recipe_of / train.set_train_modes: recipe "decoder-only" = no encoder matching, trainable prediction and
   joint networks, frozen encoder: set_train_modes puts the model in training mode and the encoder back in eval
   mode (normalization statistics and dropout fixed) after every call (train_loop, resume).
4. train.run_eval: no dev-subset evaluation during training (selection uses the frozen split after training).
5. DecoderContext: checkpoints hold the prediction and joint state only (plus the encoder's SHA-256, checked on
   restore); the encoder is rebuilt from the export at every start.
6. train.finish: the encoder must be bit-identical to the start (SHA-256); the trained prediction and joint
   tensors are written (decoder.safetensors, FP32) and scored as the export would store them (FP16) on top of the
   unchanged student export: full stage 1 dev (eval-dev/), dev-medical and dev-technical through score_domain.py's
   settings and term recall (domain/), and the decoder-step proxy on full dev (proxy/).
7. paths.RUNS: $RELABEL_ROOT/decoder-only/<arm>; train.TRAINING_SOURCES: plus this stage's sources.

The original students are scored by score-original into decoder-only/original-<B|T>/ with the same code, and
their dev scores must reproduce the recorded stage 1 student records within CROSSCHECK_POINTS per set.
decoder_report.py selects each arm's learning rate on the selection part and writes the results.
"""
from __future__ import annotations

import train  # noqa: I001  finetune/parakeet-ternary/train.py first: it configures CUDA/numba

import argparse
import contextlib
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

import evaluate  # finetune/parakeet-ternary
import paths  # finetune/parakeet-ternary
import relabel_paths
import score_domain
from labeling import layout as llayout
from training import bootstrap, decoder_steps, devsplit, schedule
from training import labels as tlabels
from training import replay as treplay
from training import seam as tseam

score_domain.binary_on_path()
import binary_paths  # noqa: E402  finetune/parakeet-binary
import binary_test_paths  # noqa: E402

ARMS = {"B-T": ("B", "T*"), "B-0": ("B", "B0"), "T-T": ("T", "T*"), "T-0": ("T", "B0")}
LRS = (1e-5, 3e-5, 1e-4)
LR_NAMES = {1e-5: "lr1e-5", 3e-5: "lr3e-5", 1e-4: "lr1e-4"}
PLAN_POOLS = schedule.DECODER_ONLY_POOLS         # the batch plan's pools 0-155 (consumed batches 0-19,967)
BATCH_SECONDS = 600.0
SEED = paths.SEED
STUDENTS = {"B": {"name": "Kea-B", "model": "MB1", "run": binary_test_paths.RUN,
                  "export_sha256": binary_test_paths.WEIGHTS_SHA256, "checkpoint_sha256": None},
            "T": {"name": "Kea-T", "model": "M1", "run": binary_paths.M1_RUN,
                  "export_sha256": binary_test_paths.M1_EXPORT_SHA256, "checkpoint_sha256": binary_paths.M1_SHA256}}
CHECKPOINT_STEP = 250_000
TRAINED_PREFIXES = ("decoder.", "joint.")
RECIPE = {"encoder_matching": False, "train_pred_joint": True, "quantized": False, "encoder_frozen": True}
EVAL_BATCH = score_domain.BATCH_SIZE              # 64, the MB1 dev check's and the domain scores' batch
EVAL_MAX_SECONDS = score_domain.MAX_BATCH_SECONDS
DEV_SETS = list(paths.DEV_SETS)
EQUIVALENCE_PER_SET = 4                           # 4 x 5 dev sets = 20 clips
EQUIVALENCE_RTOL = EQUIVALENCE_ATOL = 1e-5
CROSSCHECK_POINTS = 0.10
PROBE_STEPS = 100
PROBE_WORST = 3
WORST_SECONDS = train.MAX_SECONDS - 0.5
CHUNK_MINUTES = 110.0
MIN_CHUNK_MINUTES = 15.0
EXIT_BUDGET = 0
ORDER = [("original", "B"), ("original", "T")] + [(arm, lr) for arm in ARMS for lr in LRS]
SOURCES = ("relabel/train_decoder.py", "relabel/decoder_report.py", "relabel/training/__init__.py",
           "relabel/training/labels.py", "relabel/training/replay.py", "relabel/training/schedule.py",
           "relabel/training/batchplan.py", "relabel/training/poolaudio.py",
           "relabel/training/seam.py", "relabel/training/devsplit.py", "relabel/training/bootstrap.py",
           "relabel/training/decoder_steps.py", "relabel/labeling/shards.py", "relabel/labeling/c2.py",
           "relabel/labeling/config.py", "relabel/score_domain.py", "relabel/relabel_paths.py",
           "parakeet-binary/binary_export.py", "parakeet-binary/binary_quant.py",
           "parakeet-binary/binary_train.py")
_ACTIVE: dict = {"data": None, "ctx": None}
_TRAIN = {name: getattr(train, name) for name in ("set_train_modes", "recipe_of")}
_PATHS_GPU_LOCK = paths.gpu_lock

if Path(binary_paths.ARTIFACTS) != relabel_paths.PARAKEET_BINARY:
    raise RuntimeError("PARAKEET_BINARY_ROOT differs from finetune/parakeet-binary/binary_paths.py ARTIFACTS")


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [decoder-only] {msg}", flush=True)


# ----------------------------------------------------------------------------- layout

def root() -> Path:
    return relabel_paths.ARTIFACTS / "decoder-only"


def split_path() -> Path:
    return root() / "dev-split.json"


def student_path(s: str) -> Path:
    return root() / "students" / f"{s}.json"


def run_dir(arm: str, lr: float) -> Path:
    return root() / arm / LR_NAMES[lr]


def original_dir(s: str) -> Path:
    return root() / f"original-{s}"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def parse_lr(value: str) -> float:
    lr = float(value)
    if lr not in LR_NAMES:
        raise argparse.ArgumentTypeError(f"--lr must be one of {sorted(LR_NAMES)}")
    return lr


# ----------------------------------------------------------------------------- students

def cached_sha256(path: Path) -> str:
    """SHA-256 of a large file, cached under $RELABEL_ROOT/caches/sha256 by the file's identity."""
    path = Path(path).resolve()
    st = path.stat()
    identity = {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns, "inode": st.st_ino}
    entry = relabel_paths.ARTIFACTS / "caches" / "sha256" / f"{hashlib.sha256(str(path).encode()).hexdigest()[:32]}.json"
    try:
        cached = json.loads(entry.read_text())
        if {k: cached.get(k) for k in identity} == identity:
            return cached["sha256"]
    except (OSError, ValueError, KeyError):
        pass
    digest = evaluate.sha256_file(path)
    if path.stat().st_mtime_ns == st.st_mtime_ns:
        write_json(entry, {**identity, "sha256": digest})
    return digest


def checkpoint_path(s: str) -> Path:
    return Path(STUDENTS[s]["run"]) / "ckpt" / "latest.pt"


def export_dir(s: str) -> Path:
    return Path(STUDENTS[s]["run"]) / "export"


def ensure_student(s: str) -> dict:
    """The student's identity: export SHA-256 (pinned) and final checkpoint SHA-256 (M1 pinned; MB1 recorded in
    students/B.json at its first use, then required)."""
    spec = STUDENTS[s]
    ckpt = checkpoint_path(s)
    digest = cached_sha256(ckpt)
    if spec["checkpoint_sha256"] is not None and digest != spec["checkpoint_sha256"]:
        raise RuntimeError(f"{ckpt}: SHA-256 {digest} is not the pinned {spec['checkpoint_sha256']}")
    record = {"student": s, "name": spec["name"], "model": spec["model"], "export_dir": str(export_dir(s)),
              "export_sha256": spec["export_sha256"], "checkpoint": str(ckpt), "checkpoint_sha256": digest,
              "checkpoint_step": CHECKPOINT_STEP}
    path = student_path(s)
    if path.exists():
        old = json.loads(path.read_text())
        if {k: old.get(k) for k in record} != record:
            raise RuntimeError(f"{path} records another student identity: {old} != {record}")
        return old
    relabel_paths.require_mount()
    write_json(path, record | {"recorded_utc": train.utc(),
                               "note": "MB1's final checkpoint SHA-256 was never recorded before; recorded here"
                               if s == "B" else "M1's SHA-256 is pinned in binary_paths.M1_SHA256"})
    return record


def read_checkpoint_trained(s: str) -> dict[str, torch.Tensor]:
    """The final checkpoint's prediction and joint tensors (FP32), after its completeness, step and fraction
    checks. The file is memory-mapped: nothing else of it is read."""
    path = checkpoint_path(s)
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if not (isinstance(payload, dict) and payload.get("complete") and payload.get("format") == train.CKPT_FORMAT):
        raise RuntimeError(f"{path} is not a complete train.py checkpoint")
    if payload.get("step") != CHECKPOINT_STEP:
        raise RuntimeError(f"{path}: step {payload.get('step')} != {CHECKPOINT_STEP}")
    fraction = payload.get("weight_fraction")
    if not isinstance(fraction, (int, float)) or float(fraction) != 1.0:
        raise RuntimeError(f"{path}: weight fraction {fraction!r} != 1")
    out = {k: v.detach().clone().float() for k, v in payload["student"].items() if k.startswith(TRAINED_PREFIXES)}
    del payload
    train.drop_page_cache(path, sync=False)
    return out


def trained_state(model) -> dict[str, torch.Tensor]:
    return {k: v for k, v in model.state_dict().items() if k.startswith(TRAINED_PREFIXES)}


def load_trained(model, state: dict[str, torch.Tensor]) -> int:
    """Load prediction and joint tensors; the key set must be exactly the model's (ValueError otherwise)."""
    own = trained_state(model)
    if set(own) != set(state):
        raise ValueError(f"prediction/joint keys differ: missing {sorted(set(own) - set(state))[:5]}, "
                         f"extra {sorted(set(state) - set(own))[:5]}")
    bad = [k for k in own if tuple(own[k].shape) != tuple(state[k].shape)]
    if bad:
        raise ValueError(f"shape mismatches: {bad[:5]}")
    with torch.no_grad():
        for k, v in own.items():
            v.copy_(state[k].to(v.dtype))
    return len(state)


def encoder_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.encoder.state_dict().items()):
        t = tensor.detach().cpu().contiguous()
        digest.update(f"{name}|{t.dtype}|{tuple(t.shape)}\n".encode())
        digest.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return digest.hexdigest()


def load_export_model(s: str, device="cpu"):
    """The unchanged student export (score_domain's loader and decoder-mode pin) with its pinned SHA-256."""
    source = score_domain.describe(STUDENTS[s]["model"])
    if source["sha256"] != STUDENTS[s]["export_sha256"]:
        raise RuntimeError(f"{s}: export SHA-256 {source['sha256']} is not pinned {STUDENTS[s]['export_sha256']}")
    net, mode = score_domain.load_model(STUDENTS[s]["model"], torch.device(device))
    return net, source, mode


def set_train_modes(student, recipe: dict) -> None:
    """train.set_train_modes, then the frozen encoder back in eval mode (decoder-only recipe)."""
    _TRAIN["set_train_modes"](student, recipe)
    if recipe.get("encoder_frozen"):
        student.encoder.eval()


def recipe_of(args) -> dict:
    if getattr(args, "recipe", None) == "decoder-only":
        return dict(RECIPE)
    return _TRAIN["recipe_of"](args)


def configure(model) -> dict:
    """Freeze everything but the prediction and joint networks; their dropout 0; training modes (frozen encoder)."""
    zeroed = train.disable_pred_joint_dropout(model)
    model.requires_grad_(False)
    model.decoder.requires_grad_(True)
    model.joint.requires_grad_(True)
    set_train_modes(model, RECIPE)
    return verify(model) | {"pred_joint_dropout_modules_zeroed": zeroed}


def verify(model) -> dict:
    count = lambda m, t: sum(p.numel() for p in m.parameters() if p.requires_grad == t)  # noqa: E731
    found = {"encoder_trainable": count(model.encoder, True), "decoder_trainable": count(model.decoder, True),
             "joint_trainable": count(model.joint, True), "frozen": count(model, False),
             "trainable": count(model, True),
             "encoder_modules_in_training_mode": sum(m.training for m in model.encoder.modules()),
             "pred_joint_dropout": sorted({m.p for part in (model.decoder, model.joint) for m in part.modules()
                                           if isinstance(m, torch.nn.Dropout)}
                                          | {m.dropout for part in (model.decoder, model.joint) for m in part.modules()
                                             if isinstance(m, torch.nn.RNNBase)}),
             "fp32_trainable": all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad),
             "quantized_modules": sum("quant" in type(m).__module__ for m in model.modules())}
    problems = []
    if found["encoder_trainable"]:
        problems.append("encoder parameters require gradients")
    if not found["decoder_trainable"] or not found["joint_trainable"]:
        problems.append("prediction/joint networks not trainable")
    if any(m.requires_grad for n, m in model.named_parameters() if not n.startswith(TRAINED_PREFIXES)):
        problems.append("a parameter outside prediction/joint requires gradients")
    if found["encoder_modules_in_training_mode"]:
        problems.append("encoder modules in training mode")
    if any(p != 0.0 for p in found["pred_joint_dropout"]):
        problems.append("prediction/joint dropout is not 0")
    if not found["fp32_trainable"]:
        problems.append("trainable parameters are not FP32")
    if found["quantized_modules"]:
        problems.append("quantization modules present")
    if problems:
        raise RuntimeError(f"decoder-only model check failed: {'; '.join(problems)}")
    return found


def build_student(s: str) -> tuple[torch.nn.Module, dict]:
    """The training model on the CPU (see the module docstring, item 2)."""
    ident = ensure_student(s)
    net, source, mode = load_export_model(s, "cpu")
    ckpt = read_checkpoint_trained(s)
    own = trained_state(net)
    mismatch = [k for k in own if k not in ckpt or not torch.equal(ckpt[k].half().float(), own[k].float())]
    if mismatch or set(ckpt) != set(own):
        raise RuntimeError(f"{s}: the checkpoint's prediction/joint tensors do not round to the export's "
                           f"({mismatch[:5]}): the export is not this checkpoint's")
    load_trained(net, ckpt)
    found = configure(net)
    return net, {"student": ident, "export_source": source, "decoder_graphs_mode": mode, "trainable": found,
                 "init": "prediction/joint FP32 tensors of the final checkpoint; encoder from the export"}


def scoring_model(s: str, decoder_file: Path | None, device):
    """The unchanged export, or the export with a trained decoder file's tensors stored as the export stores
    them (FP16), on device, eval mode, decoder mode pinned."""
    from safetensors.torch import load_file
    net, source, mode = load_export_model(s, "cpu")
    if decoder_file is not None:
        state = {k: v.half().float() for k, v in load_file(str(decoder_file)).items()}
        load_trained(net, state)
        source = dict(source) | {"decoder_file": str(decoder_file), "decoder_sha256": evaluate.sha256_file(decoder_file),
                                 "description": source["description"] + "; prediction/joint networks from "
                                 "decoder.safetensors (FP16 storage, as the export format stores them)"}
    net = net.to(device).eval()
    import binary_train
    mode = binary_train.pin_decoder_graphs_mode(net, "decoder-only scoring model")
    return net, source, mode


# ----------------------------------------------------------------------------- equivalence

def equivalence_records() -> list[dict]:
    out = []
    for name in DEV_SETS:
        out += evaluate.read_manifest(paths.MANIFESTS / f"{name}.jsonl")[:EQUIVALENCE_PER_SET]
    return out


@torch.no_grad()
def encoder_equivalence(model, reference, records: list[dict], device) -> dict:
    """The configured model's encoder (in the mode training uses) against the export's, on the same features."""
    audio, lengths = evaluate.audio_batch(records)
    audio, lengths = audio.to(device), lengths.to(device)
    training_modes = sum(m.training for m in model.encoder.modules())
    with evaluate.strict_fp32(device), evaluate.inference_settings(reference):
        feats, feat_len = reference.preprocessor(input_signal=audio, length=lengths)
        e_ref, l_ref = reference.encoder(audio_signal=feats, length=feat_len)
        e_mod, l_mod = model.encoder(audio_signal=feats, length=feat_len)
    frames = torch.arange(e_ref.shape[-1], device=e_ref.device)[None, :] < l_ref[:, None]
    a, b = e_ref.transpose(1, 2)[frames], e_mod.transpose(1, 2)[frames]
    diff = (a - b).abs()
    ok = bool(torch.equal(l_ref, l_mod)) and bool(torch.allclose(b, a, rtol=EQUIVALENCE_RTOL, atol=EQUIVALENCE_ATOL))
    return {"pass": ok and training_modes == 0, "clips": len(records), "ids": [r["id"] for r in records],
            "max_abs_diff": float(diff.max()), "max_abs": float(a.abs().max()), "lengths_equal": bool(torch.equal(l_ref, l_mod)),
            "rtol": EQUIVALENCE_RTOL, "atol": EQUIVALENCE_ATOL, "encoder_modules_in_training_mode": training_modes,
            "precision": "strict FP32 (evaluate.strict_fp32), features from the export model's eval preprocessor"}


def check_equivalence(model, s: str, device) -> dict:
    reference, source, _ = load_export_model(s, device)
    try:
        result = encoder_equivalence(model, reference, equivalence_records(), device)
    finally:
        del reference
        torch.cuda.empty_cache()
    result.update(student=s, export_sha256=source["sha256"], utc=train.utc())
    return result


# ----------------------------------------------------------------------------- the trainer seam

class DecoderContext(train.Context):
    """train.Context whose checkpoints hold the prediction and joint state only, with the encoder's SHA-256."""

    def __init__(self, args, run: Path, student, data, device: str, encoder_sha: str):
        super().__init__(args, run, student, None, data, None, None, device, label_fn=_no_teacher)
        self.encoder_sha256 = encoder_sha
        self.timer: threading.Timer | None = None

    def checkpoint_payload(self) -> dict:
        payload = super().checkpoint_payload()
        payload["student"] = {k: v for k, v in trained_state(self.student).items()}
        payload["encoder_sha256"] = self.encoder_sha256
        return payload

    def restore(self, payload: dict) -> None:
        """train.Context.restore for a prediction/joint checkpoint."""
        if payload["resume_key"] != train.resume_key(self.args):
            raise RuntimeError(f"checkpoint {payload['_path']} was written by another configuration")
        if payload.get("encoder_sha256") != self.encoder_sha256:
            raise RuntimeError("the rebuilt encoder differs from the checkpoint's (SHA-256)")
        load_trained(self.student, payload["student"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.state = train.TrainState(**payload["state"])
        train.set_rng_state(self.student, payload["rng"])
        self.state.session += 1
        self.latest_best_ref = self.state.best_file
        self.state.metrics_offset = train.prune_evidence(self.run, self.state.step, self.state.best_file)


def _no_teacher(*_, **__):
    raise RuntimeError("decoder-only training has no online teacher")


def run_eval(ctx) -> None:
    """No dev-subset evaluation during training (decoder-only plan selects on the frozen split)."""
    train.log(f"step {ctx.state.step}: no dev-subset evaluation (decoder-only)")


def start_budget(ctx, minutes: float | None, deadline: float | None = None) -> None:
    """Request a checkpoint and stop after `minutes` from now, or at the epoch `deadline` (a chunk's end, which
    includes any wait for the GPU lock), whichever comes first."""
    limits = [m * 60 for m in (minutes,) if m] + [deadline - time.time() for d in (deadline,) if d]
    if not limits:
        return
    seconds = max(0.0, min(limits))

    def expire():
        if ctx.stop_reason is None:
            ctx.stop_reason = "budget"
            log(f"chunk budget reached ({seconds / 60:.0f} min after the start of training); checkpointing at the "
                "next step boundary")

    ctx.timer = threading.Timer(seconds, expire)
    ctx.timer.daemon = True
    ctx.timer.start()


def offline_data(arm: str, labels_run: str, lr: float) -> tseam.OfflineData:
    """The batch plan's pools 0-155 (materialized audio) with the arm's label set and the WSD schedule."""
    from labeling import config as lconfig
    from labeling.__main__ import c1_dir, pinned
    from training import batchplan, poolaudio
    pin = pinned()
    if float(pin["batch"]["batch_seconds"]) != BATCH_SECONDS:
        raise RuntimeError(f"the pinned stream batches {pin['batch']['batch_seconds']} s, not {BATCH_SECONDS}")
    pool, cid = c1_dir(), lconfig.config_id(pin)
    plan = batchplan.Plan.load(batchplan.production_dir(cid, PLAN_POOLS))
    if plan.meta["pinned_hash"] != pin["pinned_hash"] or plan.budget != BATCH_SECONDS * batchplan.SAMPLE_RATE:
        raise RuntimeError("the batch plan is of another pinned configuration or budget")
    label_set = ARMS[arm][1]
    labels = tlabels.LabelStore.open(tlabels.production_run_dir(labels_run), pool, label_set=label_set,
                                     through_batch=plan.consumed_batches)
    replay = treplay.PlannedReplay(plan, pool, audio="materialized", store=poolaudio.store_dir(cid, plan.pool_batches))
    return tseam.OfflineData(replay, labels, schedule.decoder_only(lr, plan), plan.n_batches)


def train_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="train_decoder.py train")
    p.add_argument("--arm", choices=sorted(ARMS), required=True)
    p.add_argument("--lr", type=parse_lr, required=True)
    p.add_argument("--labels-run", required=True, help="the C2 run name (labels/<T*>/<run>)")
    p.add_argument("--budget-minutes", type=float, default=CHUNK_MINUTES)
    p.add_argument("--deadline", type=float, help="epoch seconds: stop (checkpointed) by then (chunk jobs)")
    p.add_argument("--ckpt-minutes", type=float, default=30.0)
    p.add_argument("--overwrite", action="store_true", help="discard this run's outputs and checkpoints")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """train.parse_args for a decoder-only run (see the module docstring, item 1)."""
    a = train_parser().parse_args(argv)
    student, label_set = ARMS[a.arm]
    split = devsplit.load(split_path(), devsplit.dev_manifests())
    ident = ensure_student(student)
    data = offline_data(a.arm, a.labels_run, a.lr)
    _ACTIVE["data"] = data
    steps = data.plan["steps"]
    log(f"{a.arm} {LR_NAMES[a.lr]}: batch plan {data.replay.identity['plan_id']}, {data.replay.consumed_batches} "
        f"consumed batches -> {data.stop} planned batches, {steps} with a labeled row "
        f"({len(data.plan['empty_batches'])} fully rejected)")
    return argparse.Namespace(
        arm=a.arm, recipe="decoder-only", lr=a.lr, run_name=LR_NAMES[a.lr], max_steps=steps,
        train="relabel-stage2-journal", stream="c1-batch-plan", batch_seconds=BATCH_SECONDS,
        grad_checkpointing=False, eval_every=steps, select="final", dev_limit=None, eval_batch_size=EVAL_BATCH,
        ckpt_minutes=a.ckpt_minutes, seed=SEED, probe_steps=None, no_powerlog=True, overwrite=a.overwrite,
        smoke=False, student=student, label_set=label_set, labels_run=a.labels_run,
        labels_sha256=data.labels.identity["sha256"], replay_sha256=data.replay.identity["sha256"],
        plan_sha256=data.replay.identity["plan_sha256"], stop_batch=data.stop, schedule=data.schedule.to_dict(), export_sha256=ident["export_sha256"],
        init_sha256=ident["checkpoint_sha256"], split_sha256=split.sha256, budget_minutes=a.budget_minutes,
        deadline=a.deadline)


EXTRA_OUTPUTS = ("decoder.safetensors", "domain", "proxy", "scores.json")
RESUME_EXTRA = ("student", "label_set", "labels_run", "labels_sha256", "plan_sha256", "replay_sha256", "stop_batch",
                "schedule",
                "export_sha256", "init_sha256")


def build_context(args, run: Path, device: str = "cuda"):
    """train.build_context for a decoder-only run (see the module docstring, item 2)."""
    data = _ACTIVE["data"]
    if data is None or data.labels.label_set != args.label_set:
        raise RuntimeError("parse_args did not prepare this run's data")
    student, info = build_student(args.student)
    train.seed_everything(args.seed, student)
    enc_sha = encoder_sha256(student)
    student.to(device)
    eq = check_equivalence(student, args.student, device)
    train.append_jsonl(run / "equivalence.jsonl", eq)
    if not eq["pass"]:
        raise RuntimeError(f"encoder-equivalence check failed: {eq}")
    ctx = DecoderContext(args, run, student, data, device, enc_sha)
    _ACTIVE["ctx"] = ctx
    start_budget(ctx, args.budget_minutes, args.deadline)
    info = {"quantized_modules": [], "trainable": info["trainable"], "parameter_accounting": None,
            "model_file": info["export_source"]["weights_file"],
            "teacher": {"labels": f"offline label store ({args.label_set}: {data.labels.identity['column']}) under "
                                  f"{tlabels.MASK}", "online_teacher": None},
            "bucketing": {"source": "batch plan re-cut from C1's journal (b0prime plan \"Batch plan\")",
                          "plan": data.replay.identity, "stop_batch": data.stop},
            "decoder_only": {"arm": args.arm, "student": info["student"], "export_source": info["export_source"],
                             "labels": data.labels.identity, "replay": data.replay.identity, "plan": {
                                 k: v for k, v in data.plan.items() if k != "empty_batches"},
                             "empty_batches": data.plan["empty_batches"], "schedule": data.schedule.to_dict(),
                             "encoder_sha256": enc_sha, "equivalence": eq, "recipe": RECIPE,
                             "precision": "FP32 parameters and AdamW state; BF16 autocast (train.compute_losses)"}}
    return ctx, info


def finish(ctx, config_sha: str) -> dict:
    """train.finish for a decoder-only run (see the module docstring, item 6)."""
    from safetensors.torch import save_file
    st, args, run = ctx.state, ctx.args, ctx.run
    if ctx.timer is not None:
        ctx.timer.cancel()
    if st.step != args.max_steps:
        raise RuntimeError(f"final model not ready: step {st.step}/{args.max_steps}")
    end_sha = encoder_sha256(ctx.student)
    if end_sha != ctx.encoder_sha256:
        raise RuntimeError("the frozen encoder changed during training")
    path = run / "decoder.safetensors"
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    save_file({k: v.detach().cpu().contiguous() for k, v in trained_state(ctx.student).items()}, str(tmp),
              metadata={"content": "decoder-only prediction and joint networks, FP32", "arm": args.arm,
                        "lr": str(args.lr), "step": str(st.step)})
    os.replace(tmp, path)
    ctx.student.cpu()
    torch.cuda.empty_cache()
    net, source, mode = scoring_model(args.student, path, ctx.device)
    scores = score_model(net, run, source, ctx.device, f"{args.arm} {LR_NAMES[args.lr]}")
    del net
    torch.cuda.empty_cache()
    full = scores["dev_wer"]
    split = devsplit.load(split_path())
    selection = part_wer(run, split, "selection")
    return {"select": "final", "selected_step": st.step,
            "dev_subset_mean_wer": selection["h4"] / 100,
            "dev_subset_meaning": "H4 on the selection part of the frozen dev split (fraction), not a 400-subset",
            "dev_wer": full, "dev_mean_wer": sum(full.values()) / len(full),
            "dev_utterances": scores["dev_utterances"], "scored_on": source["description"],
            "warmup_steps": ctx.stream.schedule.warmup, "ramp_steps": 0,
            "decoder_only": {"arm": args.arm, "student": args.student, "label_set": args.label_set,
                             "lr": args.lr, "decoder_file": str(path), "decoder_sha256": source["decoder_sha256"],
                             "encoder_sha256_start": ctx.encoder_sha256, "encoder_sha256_end": end_sha,
                             "encoder_bit_identical": True, "consumed_batches": st.batches_read,
                             "fully_rejected_batches": st.skipped_batches, "selection_part": selection,
                             "decoder_graphs_mode": mode, "split_sha256": split.sha256,
                             "proxy": scores["proxy"], "domain": scores["domain_sets"]}}


def bounded_lock(deadline: float | None):
    """paths.gpu_lock (the parakeet GPU lock), waiting at most until the epoch `deadline` (score_teacher.gpu_lock:
    score_teacher.LockBusy when it is not free by then); without a deadline, paths.gpu_lock's blocking wait."""
    @contextlib.contextmanager
    def gpu_lock(label: str):
        if deadline is None:
            with _PATHS_GPU_LOCK(label):
                yield
            return
        import score_teacher
        with score_teacher.gpu_lock(label, max(0.0, deadline - time.time())):
            yield
    return gpu_lock


@contextlib.contextmanager
def seam(arm: str, deadline: float | None = None):
    """Install the decoder-only replacements in train.py and paths.py; restore them on exit."""
    items = tseam.replacements() + [
        (train, "parse_args", parse_args), (train, "build_context", build_context), (train, "finish", finish),
        (train, "recipe_of", recipe_of), (train, "set_train_modes", set_train_modes), (train, "run_eval", run_eval),
        (train, "RESUME_KEYS", tuple(train.RESUME_KEYS) + RESUME_EXTRA),
        (train, "RESUME_ALLOWED_CHANGES", tuple(train.RESUME_ALLOWED_CHANGES) + ("budget_minutes", "deadline")),
        (train, "TRAINING_SOURCES", tuple(train.TRAINING_SOURCES) + SOURCES),
        (train, "OUTPUTS", tuple(train.OUTPUTS) + EXTRA_OUTPUTS), (paths, "RUNS", root() / arm),
        (paths, "gpu_lock", bounded_lock(deadline))]
    with tseam.installed(items):
        yield


def run_train(argv: list[str]) -> int:
    a = train_parser().parse_args(argv)
    relabel_paths.require_mount()
    out = run_dir(a.arm, a.lr)
    if (out / "summary.json").exists() and not a.overwrite:
        log(f"{out} is complete")
        return 0
    _ACTIVE.update(data=None, ctx=None)
    import score_teacher
    with seam(a.arm, a.deadline):
        try:
            train.main(argv)
        except score_teacher.LockBusy as busy:
            log(f"{a.arm} {LR_NAMES[a.lr]}: the GPU lock was not free before the chunk deadline ({busy}); the next "
                "chunk resumes")
            return EXIT_BUDGET
        except SystemExit as stop:
            ctx = _ACTIVE["ctx"]
            if stop.code == train.EXIT_INTERRUPTED and ctx is not None and ctx.stop_reason == "budget":
                log(f"{a.arm} {LR_NAMES[a.lr]}: stopped at step {ctx.state.step} (consumed batch "
                    f"{ctx.state.batches_read}) for the chunk budget; the next chunk resumes")
                return EXIT_BUDGET
            raise
    return 0


# ----------------------------------------------------------------------------- scoring

def _config(source: dict, kind: str, extra: dict | None = None) -> dict:
    return {"weights_sha256": source["sha256"], "decoder_sha256": source.get("decoder_sha256"), "kind": kind,
            "batch_size": EVAL_BATCH, "max_batch_seconds": EVAL_MAX_SECONDS, **(extra or {})}


def _done(path: Path, config: dict) -> dict | None:
    if path.exists():
        data = json.loads(path.read_text())
        if data.get("config") == config:
            return data
    return None


def score_model(net, out: Path, source: dict, device, label: str, crosscheck: str | None = None) -> dict:
    """Full stage 1 dev (eval-dev/<set>.json), the domain sets (domain/), the decoder-step proxy on full dev
    (proxy/); each part skipped when already written with the same configuration."""
    import binary_train
    import devsets
    device = torch.device(device)
    dev = {}
    for name in DEV_SETS:
        path = out / "eval-dev" / f"{name}.json"
        config = _config(source, "dev")
        result = _done(path, config)
        if result is None:
            manifest = paths.MANIFESTS / f"{name}.jsonl"
            result = evaluate.evaluate_set(net, name, manifest, source, EVAL_BATCH, None, device, EVAL_MAX_SECONDS)
            result.update(config=config, decoder_graphs_mode=binary_train.require_decoder_graphs_mode(net, label))
            write_json(path, result)
            log(f"{label} {name}: WER {100 * result['wer']['wer']:.3f}")
        dev[name] = result
    check = None
    if crosscheck is not None:
        check = dev_crosscheck(dev, crosscheck)
        write_json(out / "crosscheck.json", check)
        if not check["pass"]:
            raise RuntimeError(f"{label}: dev scores do not reproduce the recorded {crosscheck} records: {check}")
    parts = [p for s in devsets.DEFAULT_SETS for p in devsets.SETS[s]]
    inputs = score_domain.part_manifests(parts, None)
    matcher = None
    domain = {}
    for part in parts:
        manifest, meta = inputs[part]
        path = out / "domain" / f"{part}.json"
        config = _config(source, "domain", {"manifest_sha256": meta["manifest_sha256"]})
        result = _done(path, config)
        if result is None:
            result = evaluate.evaluate_set(net, part, manifest, source, score_domain.BATCH_SIZE, None, device,
                                           score_domain.MAX_BATCH_SECONDS)
            if devsets.set_of(part) in devsets.TERM_KIND:
                import terms
                matcher = matcher or terms.matcher("medical")
                result["term_recall"] = score_domain.term_recall(result["records"], matcher)
            result.update(config=config, part=part, set=devsets.set_of(part),
                          decoder_graphs_mode=binary_train.require_decoder_graphs_mode(net, label))
            write_json(path, result)
        domain[part] = result
    domain_summary = score_domain.summarize(label, domain, list(devsets.DEFAULT_SETS))
    write_json(out / "domain" / "summary.json", domain_summary)
    proxy = {}
    for name in DEV_SETS:
        path = out / "proxy" / f"{name}.json"
        config = _config(source, "proxy")
        result = _done(path, config)
        if result is None:
            records = evaluate.read_manifest(paths.MANIFESTS / f"{name}.jsonl")
            result = decoder_steps.proxy_set(net, records, EVAL_BATCH, EVAL_MAX_SECONDS, device) | {"config": config}
            write_json(path, result)
        proxy[name] = result
    proxy_summary = {"sets": {n: {k: v for k, v in r.items() if k != "nemo_token_mismatched_ids"}
                              for n, r in proxy.items()}, "pooled": decoder_steps.pooled(proxy)}
    write_json(out / "proxy" / "summary.json", proxy_summary)
    summary = {"label": label, "source": source, "complete": True, "utc": train.utc(),
               "dev_wer": {n: r["wer"]["wer"] for n, r in dev.items()},
               "dev_utterances": {n: r["utterances"] for n, r in dev.items()},
               "domain_sets": domain_summary["sets"], "proxy": proxy_summary["pooled"], "crosscheck": check}
    write_json(out / "scores.json", summary)
    return summary


def dev_crosscheck(dev: dict[str, dict], model: str) -> dict:
    """Each set's WER against the recorded stage 1 student record (relabel_paths.STUDENT_RECORDS)."""
    out = {"model": model, "tolerance_points": CROSSCHECK_POINTS, "sets": {}}
    for name, result in dev.items():
        recorded = json.loads((relabel_paths.STUDENT_RECORDS[model] / f"{name}.json").read_text())
        rw = recorded["wer"]["wer"] if isinstance(recorded["wer"], dict) else recorded["wer"]
        old = {r["id"]: r["hyp"] for r in recorded["records"]}
        out["sets"][name] = {"wer": 100 * result["wer"]["wer"], "recorded_wer": 100 * rw,
                             "difference_points": 100 * (result["wer"]["wer"] - rw),
                             "identical_hypotheses": sum(old.get(r["id"]) == r["hyp"] for r in result["records"]),
                             "utterances": len(result["records"])}
    out["pass"] = all(abs(v["difference_points"]) <= CROSSCHECK_POINTS for v in out["sets"].values())
    return out


def part_wer(out: Path, split: devsplit.Split, part: str) -> dict:
    """Corpus WER (points) of a scored model's dev records on one part of the split, per H4 set, and H4."""
    sets = {}
    for name in devsplit.H4:
        records = json.loads((out / "eval-dev" / f"{name}.json").read_text())["records"]
        ids = split.ids(name, part)
        rows = [r for r in records if r["id"] in ids and r.get("scored", True)]
        if {r["id"] for r in records if r["id"] in ids} != ids:
            raise RuntimeError(f"{out}: {name} records do not cover the {part} part")
        words = sum(len(r["ref_norm"].split()) for r in rows)
        sets[name] = 100 * sum(r["S"] + r["D"] + r["I"] for r in rows) / words
    return {"part": part, "sets": sets, "h4": sum(sets.values()) / len(sets)}


def score_original(s: str, lock_wait: float, deadline: float | None = None) -> int:
    """Score the unchanged student; with a deadline (chunk jobs), a lock not free by then is a budget stop (0)."""
    import score_teacher
    relabel_paths.require_mount()
    devsplit.load(split_path(), devsplit.dev_manifests())
    ensure_student(s)
    out = original_dir(s)
    wait = lock_wait if deadline is None else max(0.0, min(lock_wait, deadline - time.time()))
    try:
        with score_teacher.gpu_lock(f"decoder-only score-original {s}", wait):
            train.deterministic_mode()
            net, source, _ = scoring_model(s, None, "cuda")
            summary = score_model(net, out, source, "cuda", f"original {s}", crosscheck=STUDENTS[s]["model"])
    except score_teacher.LockBusy as busy:
        if deadline is None:
            raise
        log(f"original {s}: the GPU lock was not free before the chunk deadline ({busy}); the next chunk scores it")
        return EXIT_BUDGET
    log(f"original {s}: dev {summary['dev_wer']}")
    return 0


# ----------------------------------------------------------------------------- equivalence and probe jobs

def equivalence_job(s: str, lock_wait: float) -> int:
    import score_teacher
    relabel_paths.require_mount()
    with score_teacher.gpu_lock(f"decoder-only equivalence {s}", lock_wait):
        train.deterministic_mode()
        net, info = build_student(s)
        net.to("cuda")
        result = check_equivalence(net, s, "cuda") | {"trainable": info["trainable"], "student": info["student"]}
    write_json(root() / "equivalence" / f"{s}.json", result)
    log(f"equivalence {s}: pass={result['pass']} max |diff| {result['max_abs_diff']:.3g} "
        f"(max |enc| {result['max_abs']:.3g})")
    return 0 if result["pass"] else 1


def probe_batches(seed: int) -> tuple[list[list[dict]], list[list[dict]]]:
    """Worst-case batches (dev-clean utterances concatenated per chapter to about WORST_SECONDS, references
    joined) and timed batches (all dev sets, cut as train.cut_pool at BATCH_SECONDS, seeded order); dev audio and
    dev references only, never training labels."""
    worst, cur, texts, chapter = [], [], [], None
    for r in evaluate.read_manifest(paths.MANIFESTS / "librispeech_dev_clean.jsonl"):
        if r["duration"] > WORST_SECONDS:
            continue
        ch = r["id"].rsplit("-", 1)[0]
        if cur and (ch != chapter or sum(len(x) for x in cur) / paths.SAMPLE_RATE + r["duration"] > WORST_SECONDS):
            if sum(len(x) for x in cur) / paths.SAMPLE_RATE >= WORST_SECONDS - 6:
                a = np.concatenate(cur)
                worst.append({"audio": a, "duration": len(a) / paths.SAMPLE_RATE, "id": f"concat:{len(worst)}",
                              "source": "probe-worst", "text": " ".join(texts)})
            cur, texts = [], []
        chapter = ch
        cur.append(evaluate.load_audio(r["audio_filepath"]))
        texts.append(r["text"])
        if len(worst) >= 64:
            break
    per = int(BATCH_SECONDS // WORST_SECONDS)
    worst_batches = [worst[i:i + per] for i in range(0, PROBE_WORST * per, per)]
    pool = []
    for name in DEV_SETS:
        pool += [r for r in evaluate.read_manifest(paths.MANIFESTS / f"{name}.jsonl")
                 if train.MIN_SECONDS <= r["duration"] <= train.MAX_SECONDS]
    rng = np.random.default_rng([seed, 7])
    pool = [pool[i] for i in rng.permutation(len(pool))]
    cut = train.cut_pool([r["duration"] for r in pool], BATCH_SECONDS)
    order = rng.permutation(len(cut))
    timed = [[pool[i] for i in cut[j]] for j in order[:PROBE_STEPS + 1]]
    return worst_batches, timed


def probe_job(s: str, steps: int, lock_wait: float) -> int:
    import types
    import score_teacher
    relabel_paths.require_mount()
    worst, timed = probe_batches(SEED)
    timed = timed[:steps + 1]
    with score_teacher.gpu_lock(f"decoder-only probe {s}", lock_wait):
        train.deterministic_mode()
        tdt_sha = train.prepare_tdt_loss()
        net, info = build_student(s)
        train.seed_everything(SEED, net)
        net.to("cuda")
        optimizer, scheduler = tseam.make_optimizer(net, 1e-4, steps)
        ctx = types.SimpleNamespace(student=net, recipe=dict(RECIPE), optimizer=optimizer, scheduler=scheduler,
                                    state=types.SimpleNamespace(step=0))
        trainable = [p for g in optimizer.param_groups for p in g["params"]]
        set_train_modes(net, RECIPE)
        verify(net)
        torch.cuda.reset_peak_memory_stats()
        times, seconds, rows, status = [], [], [], {"worst": "ok", "timed": "ok"}
        for kind, batches in (("worst", worst), ("timed", timed)):
            for j, items in enumerate(batches):
                if kind == "timed" and not isinstance(items[0].get("audio"), np.ndarray):
                    items = [{"audio": evaluate.load_audio(r["audio_filepath"]), "duration": r["duration"],
                              "id": r["id"], "source": r.get("source"), "text": r["text"]} for r in items]
                batch = train.collate(items)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                tokens, token_len = train.tokenize(net, [it["text"] for it in items], "cuda")
                labeled = {"audio": batch["audio"].cuda(), "audio_len": batch["audio_len"].cuda(), "tokens": tokens,
                           "token_len": token_len, "teacher_enc": None, "teacher_len": None}
                try:
                    train.reseed_cudnn_rnn_dropout()
                    train.train_step(ctx, labeled, trainable)
                except torch.OutOfMemoryError:
                    status[kind] = f"oom at batch {j}"
                    break
                torch.cuda.synchronize()
                if kind == "timed" and j >= 1:
                    times.append(time.perf_counter() - t0)
                    seconds.append(batch["seconds"])
                    rows.append(len(items))
                ctx.state.step += 1
        peak_alloc = torch.cuda.max_memory_allocated() / 2**30
        peak_reserved = torch.cuda.max_memory_reserved() / 2**30
        process_now = gpu_process_gb()
        reserved_now = torch.cuda.memory_reserved() / 2**30
        enc_ok = encoder_sha256(net)
    peak_process = peak_reserved + max(0.0, (process_now or reserved_now) - reserved_now)
    step = float(np.mean(times)) if times else None
    result = {"student": s, "steps_timed": len(times), "status": status, "step_time_s": step,
              "step_time_p90_s": float(np.percentile(times, 90)) if times else None,
              "audio_seconds_per_step": float(np.mean(seconds)) if seconds else None,
              "rows_per_step": float(np.mean(rows)) if rows else None,
              "peak_allocated_gib": peak_alloc, "peak_reserved_gib": peak_reserved,
              "process_gpu_gib_at_end": process_now, "peak_process_gib_estimate": peak_process,
              "declared_vram_gb": math.ceil(peak_process + llayout.VRAM_MARGIN_GB),
              "host_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
              "worst_case": {"batches": len(worst), "rows": [len(b) for b in worst],
                             "seconds": [round(sum(x["duration"] for x in b), 1) for b in worst]},
              "tdt_kernel_sha256": tdt_sha, "encoder_sha256": enc_ok, "trainable": info["trainable"],
              "data": "stage 1 dev audio and references (no training labels); worst case: dev-clean concatenated "
                      f"per chapter to about {WORST_SECONDS} s", "utc": train.utc()}
    if step:
        per_run = planned_batches() * step / 3600
        result["estimate"] = {"gpu_hours_per_run": per_run, "runs": len(LRS) * 2, "gpu_hours_this_student": per_run * 6,
                              "note": "probe step time on in-memory dev batches x the planned batches of pools "
                                      "0-155 (the batch plan's, or 0.827 x 19,968 before it is built); replay "
                                      "reading, checkpoints and finishing (dev, domain, proxy scoring) not included"}
    write_json(root() / "probe" / f"{s}.json", result)
    log(json.dumps({k: v for k, v in result.items() if k not in ("trainable", "worst_case")}))
    return 0


def planned_batches() -> int:
    """The decoder-only plan's planned batches, or the measured ratio's estimate before the plan is built."""
    from labeling import config as lconfig
    from labeling.__main__ import pinned
    from training import batchplan
    d = batchplan.production_dir(lconfig.config_id(pinned()), PLAN_POOLS)
    if (d / "plan.json").exists():
        return batchplan.Plan.load(d).n_batches
    return round(0.827 * PLAN_POOLS * batchplan.POOL_BATCHES)


def gpu_process_gb() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        pid, used = [x.strip() for x in line.split(",")[:2]]
        if pid == str(os.getpid()):
            return float(used) / 1024
    return None


# ----------------------------------------------------------------------------- split, chunks, status

def freeze_split() -> int:
    relabel_paths.require_mount()
    split = devsplit.freeze(split_path(), devsplit.build(devsplit.dev_manifests()))
    counts = devsplit.Split(split).counts()
    log(f"dev split {split['sha256'][:12]} at {split_path()}")
    print(json.dumps(counts, indent=1))
    return 0


def item_done(item: tuple) -> bool:
    kind, value = item
    if kind == "original":
        p = original_dir(value) / "scores.json"
        return p.exists() and json.loads(p.read_text()).get("complete") is True
    return (run_dir(kind, value) / "summary.json").exists()


def item_command(item: tuple, labels_run: str, minutes: float, deadline: float) -> list[str]:
    here = Path(__file__).resolve().parent
    kind, value = item
    if kind == "original":
        return [str(here / "python"), str(here / "train_decoder.py"), "score-original", "--student", value,
                "--deadline", f"{deadline:.0f}"]
    return [str(here / "python"), str(here / "train_decoder.py"), "train", "--arm", kind, "--lr", repr(value),
            "--labels-run", labels_run, "--budget-minutes", f"{minutes:.1f}", "--deadline", f"{deadline:.0f}"]


def chunk(labels_run: str, budget_minutes: float, runner=subprocess.run) -> int:
    """One chunk job: the next unfinished items of ORDER, each in its own process (which takes the parakeet GPU
    lock for its own duration), until the budget leaves less than MIN_CHUNK_MINUTES or a run stops on it."""
    relabel_paths.require_mount()
    devsplit.load(split_path(), devsplit.dev_manifests())
    t0, deadline = time.monotonic(), time.time() + budget_minutes * 60
    for item in ORDER:
        if item_done(item):
            continue
        left = budget_minutes - (time.monotonic() - t0) / 60
        if left < MIN_CHUNK_MINUTES:
            break
        log(f"chunk: {item} with {left:.0f} min left")
        rc = runner(item_command(item, labels_run, left, deadline), cwd=str(Path(__file__).resolve().parent)).returncode
        if rc != 0:
            log(f"chunk: {item} exited {rc}")
            return rc
        if not item_done(item):
            break
    log(f"chunk: {sum(item_done(i) for i in ORDER)}/{len(ORDER)} items done")
    return 0


def status() -> dict:
    out = {}
    for item in ORDER:
        kind, value = item
        key = f"original-{value}" if kind == "original" else f"{kind}/{LR_NAMES[value]}"
        if item_done(item):
            out[key] = "done"
        elif kind != "original" and (run_dir(kind, value) / "ckpt" / "latest.pt").exists():
            m = run_dir(kind, value) / "metrics.jsonl"
            last = train.read_log_jsonl(m)[0][-1:] if m.exists() else []
            out[key] = f"in progress (step {last[0]['step'] if last else '?'})"
        else:
            out[key] = "pending"
    return out


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "train":
        return run_train(argv[1:])
    parser = argparse.ArgumentParser(prog="train_decoder.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("split")
    for name in ("equivalence", "probe", "score-original"):
        p = sub.add_parser(name)
        p.add_argument("--student", choices=sorted(STUDENTS), required=True)
        p.add_argument("--lock-wait", type=float, default=4 * 3600.0)
        if name == "probe":
            p.add_argument("--steps", type=int, default=PROBE_STEPS)
        if name == "score-original":
            p.add_argument("--deadline", type=float, help="epoch seconds (chunk jobs): give up waiting for the lock")
    p = sub.add_parser("chunk")
    p.add_argument("--labels-run", required=True)
    p.add_argument("--budget-minutes", type=float, default=CHUNK_MINUTES)
    sub.add_parser("status")
    p = sub.add_parser("report")
    p.add_argument("--out", type=Path, help="RESULTS-decoder-only.md (default: $RELABEL_ROOT/decoder-only/)")
    args = parser.parse_args(argv)
    if args.cmd == "split":
        return freeze_split()
    if args.cmd == "equivalence":
        return equivalence_job(args.student, args.lock_wait)
    if args.cmd == "probe":
        return probe_job(args.student, args.steps, args.lock_wait)
    if args.cmd == "score-original":
        return score_original(args.student, args.lock_wait, args.deadline)
    if args.cmd == "chunk":
        return chunk(args.labels_run, args.budget_minutes)
    if args.cmd == "status":
        print(json.dumps(status(), indent=1))
        return 0
    import decoder_report
    return decoder_report.main(args.out)


if __name__ == "__main__":
    sys.exit(main())
