"""Single-device streaming CTC training with durable, fail-closed recovery.

Training keeps latent parameters and Adam states in FP32 and uses BF16 matmuls
on CUDA. Packed XNOR/popcount is an inference export concern. Checkpoints are
trusted local PyTorch files: do not load arbitrary downloaded pickle files.
"""

from __future__ import annotations

import contextlib
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import signal
import sqlite3
import time
import traceback
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .augmentation import spec_augment
from .features import CausalLogMel
from .health import HealthMonitor, TrainingCollapse
from .model import BinaryCTCModel
from .notifications import Notifier
from .storage import (append_json, atomic_torch_save, check_free_space, digest,
                      ensure_artifact_path, fsync_directory, gpu_lock, heartbeat,
                      object_digest, write_json)
from .tokenizer import CharacterTokenizer, SpeechTokenizer, normalize_text


def _safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)


def load_tokenizer(config):
    """Return tokenizer and a fingerprint of its actual encoding contract."""
    casefold = bool(config.get("casefold", True))
    if config.get("kind", "sentencepiece") == "character":
        kwargs = {"casefold": casefold}
        if "alphabet" in config:
            kwargs["alphabet"] = config["alphabet"]
        tokenizer = CharacterTokenizer(**kwargs)
        fingerprint = object_digest({"kind": "character", "alphabet": tokenizer.alphabet,
                                     "casefold": casefold, "blank": 0, "version": 1})
    elif config.get("kind", "sentencepiece") == "sentencepiece":
        path = ensure_artifact_path(config["path"])
        tokenizer = SpeechTokenizer(path, casefold=casefold)
        fingerprint = object_digest({"kind": "sentencepiece", "sha256": digest(path),
                                     "casefold": casefold, "blank": 0, "shift": 1, "version": 1})
    else:
        raise ValueError("tokenizer.kind must be sentencepiece or character")
    return tokenizer, fingerprint


def quantization_at(step, config, quantizer="binary"):
    def fraction(prefix):
        start = int(config.get(f"{prefix}_start_step", 0))
        ramp = int(config.get(f"{prefix}_ramp_steps", 0))
        if start < 0 or ramp < 0:
            raise ValueError("Quantization starts and ramps must be nonnegative")
        return (float(step >= start) if ramp == 0 else
                min(1.0, max(0.0, (step - start) / ramp)))
    weight, activation = fraction("weight"), fraction("activation")
    def phase(value):
        return "fp" if value == 0 else quantizer if value == 1 else "ramp"
    return {"weight": weight, "activation": activation,
            "phase": f"w:{phase(weight)}/a:{phase(activation)}"}


def schedule_at(step, config):
    """Warmup + cosine, with an explicitly separate final cooldown phase."""
    maximum = int(config["max_steps"])
    warmup = int(config.get("warmup_steps", 2000))
    peak = float(config.get("lr", 2e-4))
    minimum = float(config.get("min_lr", peak * .1))
    cooldown = float(config.get("cooldown_start_fraction", .7))
    if not 0 <= cooldown < 1 or maximum <= 0 or warmup < 0 or not 0 <= minimum <= peak:
        raise ValueError("Invalid optimizer schedule")
    boundary = max(warmup, maximum * cooldown)
    if warmup and step < warmup:
        lr = peak * (step + 1) / warmup
    elif step < boundary:
        progress = (step - warmup) / max(1, boundary - warmup)
        # Reach a reduced rate before final cooldown, continuously.
        floor = peak * float(config.get("cooldown_lr_multiplier", .1))
        lr = floor + (peak - floor) * .5 * (1 + math.cos(math.pi * progress))
    else:
        floor = peak * float(config.get("cooldown_lr_multiplier", .1))
        progress = min(1., (step - boundary) / max(1, maximum - boundary - 1))
        lr = minimum + (floor - minimum) * .5 * (1 + math.cos(math.pi * progress))
    wd = float(config.get("weight_decay", .01)) if step < maximum * cooldown else 0.
    return lr, wd


def optimizer_for(model, config):
    decayed, exempt = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (exempt if parameter.ndim < 2 or any(
            part in name.lower() for part in ("norm", "scale", "threshold", "bias")
        ) else decayed).append(parameter)
    return torch.optim.AdamW([
        {"params": decayed, "decay_enabled": True},
        {"params": exempt, "weight_decay": 0., "decay_enabled": False},
    ], lr=float(config.get("lr", 2e-4)), betas=tuple(config.get("betas", (.9, .95))),
        weight_decay=float(config.get("weight_decay", .01)))


def _finite_tensors(named, description):
    """One device synchronization per tensor device in the healthy case."""
    groups = {}
    for name, value in named:
        if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
            groups.setdefault(value.device, []).append((name, value))
    for entries in groups.values():
        checks = [torch.isfinite(value.detach()).all() for _, value in entries]
        if checks and not bool(torch.stack(checks).all()):
            bad = [name for (name, _), check in zip(entries, checks) if not bool(check)]
            raise TrainingCollapse(f"Nonfinite {description}", {"tensors": bad[:20]})


def check_finite_state(model, optimizer=None):
    # Checking logits alone misses NaNs hidden by a sign comparison.
    _finite_tensors(list(model.named_parameters()) + list(model.named_buffers()), "latent model state")
    if optimizer is not None:
        _finite_tensors(((f"{index}.{key}", value)
                         for index, state in enumerate(optimizer.state.values())
                         for key, value in state.items()), "optimizer state")


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class UniqueLedger:
    """Disk-backed exact-PCM uniqueness; augmented/repeated exposure is separate."""
    def __init__(self, path, resume_step):
        self.connection = sqlite3.connect(ensure_artifact_path(path))
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS audio (content_id TEXT PRIMARY KEY, samples INTEGER NOT NULL, first_step INTEGER NOT NULL)")
        self.connection.execute("DELETE FROM audio WHERE first_step > ?", (resume_step,))
        self.connection.commit()
        self.count, self.samples = self.connection.execute("SELECT COUNT(*), COALESCE(SUM(samples),0) FROM audio").fetchone()

    def add(self, examples, step):
        for example in examples:
            samples = int(example["audio"].numel())
            cursor = self.connection.execute("INSERT OR IGNORE INTO audio VALUES (?, ?, ?)",
                                             (example["content_id"], samples, step))
            if cursor.rowcount:
                self.count += 1
                self.samples += samples
        self.connection.commit()

    def summary(self):
        return {"unique_recordings": self.count, "unique_audio_seconds": self.samples / 16000}

    def close(self):
        self.connection.close()


class Batcher:
    def __init__(self, dataset, tokenizer, config, validation_ids, state=None):
        self.dataset, self.tokenizer, self.config = dataset, tokenizer, config
        self.validation_ids = set(validation_ids)
        self.pending = None
        self.attempted, self.rejected, self.consecutive_rejected = 0, 0, 0
        if state:
            self.pending = state.get("pending")
            self.attempted, self.rejected = state.get("attempted", 0), state.get("rejected", 0)
            self.consecutive_rejected = state.get("consecutive_rejected", 0)
        self.iterator = iter(dataset)

    def state_dict(self):
        return {"pending": self.pending, "attempted": self.attempted,
                "rejected": self.rejected, "consecutive_rejected": self.consecutive_rejected}

    def _reject(self, reason):
        self.rejected += 1
        self.consecutive_rejected += 1
        # A lifetime count would eventually kill a healthy million-hour run
        # for a tiny rejection rate. An absolute cap is only an explicit opt-in.
        limit = self.config.get("max_alignment_rejections")
        consecutive = int(self.config.get("max_consecutive_rejections", 100))
        minimum = int(self.config.get("rejection_fraction_min_samples", 100))
        maximum_fraction = float(self.config.get("max_rejection_fraction", .5))
        if ((limit is not None and self.rejected >= int(limit)) or self.consecutive_rejected >= consecutive or
                (self.attempted >= minimum and self.rejected / self.attempted > maximum_fraction)):
            raise TrainingCollapse("Excessive unusable or CTC-unalignable training examples", {
                "last_reason": reason, "attempted": self.attempted, "rejected": self.rejected})

    def next(self):
        examples, seconds = [], 0.
        budget = float(self.config.get("batch_audio_seconds", 30))
        maximum = int(self.config.get("max_batch_size", 8))
        while len(examples) < maximum:
            if self.pending is not None:
                example, self.pending = self.pending, None
            else:
                example = next(self.iterator)
                self.attempted += 1
                audio = example["audio"]
                if example["content_id"] in self.validation_ids:
                    self._reject("validation audio overlaps training")
                    continue
                if (int(example["sample_rate"]) != 16000 or audio.ndim != 1 or audio.numel() == 0
                        or not torch.isfinite(audio).all()):
                    self._reject("invalid waveform")
                    continue
                try:
                    target = self.tokenizer.encode(example["text"])
                except ValueError:
                    self._reject("transcript cannot be encoded")
                    continue
                frames = (audio.numel() + 159) // 160
                required = len(target) + sum(a == b for a, b in zip(target, target[1:]))
                if not target or required > (frames + 7) // 8:
                    self._reject("CTC target length plus adjacent repeats exceeds encoder frames")
                    continue
                example = {**example, "target": target, "seconds": audio.numel() / 16000}
                self.consecutive_rejected = 0
            if examples and seconds + example["seconds"] > budget:
                self.pending = example
                break
            # A single long utterance may exceed the batching target, but is
            # bounded independently by the dataset's max_seconds constraint.
            examples.append(example)
            seconds += example["seconds"]
        return examples


def _collate(examples, extractor, device):
    features = [extractor(example["audio"].float()) for example in examples]
    lengths = torch.tensor([feature.shape[-1] for feature in features], dtype=torch.long, device=device)
    padded = features[0].new_zeros(len(features), 80, max(feature.shape[-1] for feature in features))
    for index, feature in enumerate(features):
        padded[index, :, :feature.shape[-1]] = feature
    target_lengths = torch.tensor([len(e["target"]) for e in examples], dtype=torch.long)
    targets = torch.tensor([value for e in examples for value in e["target"]], dtype=torch.long, device=device)
    return padded.to(device), lengths, targets, target_lengths


def _loss(logits, lengths, targets, target_lengths, examples):
    _finite_tensors([("logits", logits)], "logits")
    required = [len(e["target"]) + sum(a == b for a, b in zip(e["target"], e["target"][1:])) for e in examples]
    if any(need > int(actual) for need, actual in zip(required, lengths.detach().cpu())):
        raise TrainingCollapse("Model emitted fewer frames than the verified CTC alignment requires")
    # FP32 log probabilities/CTC and explicit normalization per utterance.
    raw = F.ctc_loss(logits.float().log_softmax(-1).transpose(0, 1), targets,
                     lengths.detach().cpu(), target_lengths, blank=0, reduction="none", zero_infinity=False)
    _finite_tensors([("raw_ctc_loss", raw)], "CTC loss")
    return raw / target_lengths.to(raw.device).clamp_min(1)


def _edit_distance(reference, hypothesis):
    row = list(range(len(hypothesis) + 1))
    for i, item in enumerate(reference, 1):
        updated = [i]
        for j, other in enumerate(hypothesis, 1):
            updated.append(min(updated[-1] + 1, row[j] + 1, row[j - 1] + (item != other)))
        row = updated
    return row[-1]


@torch.no_grad()
def evaluate(model, examples, tokenizer, extractor, device, run_dir, step, max_batch_size=8):
    model.eval()
    check_finite_state(model)
    def empty_counts():
        return dict(loss_sum=0., words=0, word_errors=0, chars=0, char_errors=0,
                    empty=0, blanks=0, frames=0, examples=0)
    aggregate, sources = empty_counts(), {}
    for offset in range(0, len(examples), max_batch_size):
        heartbeat(run_dir, "validation", step=step, example=offset)
        batch = examples[offset:offset + max_batch_size]
        features, lengths, targets, target_lengths = _collate(batch, extractor, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, output_lengths = model(features, lengths)
        losses = _loss(logits, output_lengths, targets, target_lengths, batch).cpu()
        for example, ids, length, loss in zip(batch, logits.argmax(-1).cpu(), output_lengths.cpu(), losses):
            ids = ids[:int(length)]
            prediction = tokenizer.decode_ctc(ids.tolist())
            reference = normalize_text(example["text"], casefold=tokenizer.casefold)
            prediction = normalize_text(prediction, casefold=tokenizer.casefold)
            counts = {"word_errors": _edit_distance(reference.split(), prediction.split()),
                      "words": len(reference.split()), "char_errors": _edit_distance(reference, prediction),
                      "chars": len(reference), "empty": int(not prediction.strip()),
                      "blanks": int((ids == 0).sum()), "frames": ids.numel(),
                      "loss_sum": float(loss), "examples": 1}
            source = sources.setdefault(str(example["source"]), empty_counts())
            for totals in (aggregate, source):
                for name, value in counts.items():
                    totals[name] += value
    def metrics(counts):
        if any(not counts[name] for name in ("examples", "words", "chars", "frames")):
            raise ValueError("Validation must contain nonempty, alignable speech and transcripts")
        return {"loss": counts["loss_sum"] / counts["examples"],
                "wer": counts["word_errors"] / counts["words"],
                "cer": counts["char_errors"] / counts["chars"],
                "blank_fraction": counts["blanks"] / counts["frames"],
                "empty_fraction": counts["empty"] / counts["examples"],
                "examples": counts["examples"], "reference_words": counts["words"],
                "reference_characters": counts["chars"], "encoder_frames": counts["frames"]}
    model.train()
    return {**metrics(aggregate), "sources": {name: metrics(counts) for name, counts in sources.items()}}


def _promote(run_dir):
    """Atomic hard-link promotion retains the old inode when latest is replaced."""
    source = ensure_artifact_path(run_dir / "latest.pt")
    target = ensure_artifact_path(run_dir / "last_known_good.pt")
    temporary = ensure_artifact_path(run_dir / ".last_known_good.pt.tmp")
    temporary.unlink(missing_ok=True)
    os.link(source, temporary)
    os.replace(temporary, target)
    fsync_directory(run_dir)


def train(run_dir: Path, resume: bool = False, stop_after: int | None = None,
          *, dataset=None, model=None) -> dict:
    """Train prepared config/validation; stop_after is an absolute step limit.

    Optional dataset/model injection supports bounded CPU integration tests.
    Resume is explicit, requires matching artifacts, and is refused for runs
    whose status records collapse/failure. STOP/SIGINT/SIGTERM finish the current
    optimizer step and checkpoint; an external supervisor handles hangs/SIGKILL.
    """
    run_dir = ensure_artifact_path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    cfg = config["training"]
    device = torch.device(cfg.get("device", "cuda"))
    notifier = Notifier(run_dir, config.get("notifications", {}))
    state = {"state": "starting", "step": 0, "audio_seconds": 0., "wall_seconds": 0.,
             "optimized_examples": 0, "optimized_sources": {}, "reader_stats": {}}
    ledger = None
    handlers = {}
    stop_requested = {"reason": None}
    started = time.monotonic()
    owns_run = False
    try:
        heartbeat(run_dir, "startup", step=0)
        if hasattr(notifier, "check_delivery_config"):
            notifier.check_delivery_config()
        check_free_space(float(cfg.get("minimum_free_disk_gib", 20)))
        if float(cfg.get("distillation_weight", 0)) != 0 or cfg.get("teacher_checkpoint"):
            raise ValueError("Aligned teacher-logit distillation is not implemented; use teacher-filtered transcripts or a zero distillation_weight")
        maximum = int(cfg["max_steps"])
        accumulation = int(cfg.get("grad_accumulation", 1))
        if maximum <= 0 or accumulation <= 0 or int(cfg.get("max_batch_size", 8)) <= 0 or float(cfg.get("batch_audio_seconds", 30)) <= 0:
            raise ValueError("Step, accumulation and batching budgets must be positive")
        feature_cfg = config.get("features", {})
        if (config["model"].get("n_mels", 80) != 80 or
                any(feature_cfg.get(name, value) != value for name, value in
                    {"sample_rate": 16000, "n_mels": 80, "hop_length": 160}.items())):
            raise ValueError("Trainer requires 80-bin features with a 160-sample hop at 16000 Hz")
        for key in ("eval_every", "checkpoint_every", "log_every"):
            if int(cfg.get(key, {"eval_every": 500, "checkpoint_every": 250, "log_every": 10}[key])) <= 0:
                raise ValueError(f"training.{key} must be positive")
        tokenizer, tokenizer_fingerprint = load_tokenizer(config["tokenizer"])
        if tokenizer.vocab_size != config["model"]["vocab_size"]:
            raise ValueError("Tokenizer vocabulary does not match the model's CTC head")
        contract = object_digest(config)
        previous_status = json.loads((run_dir / "status.json").read_text()) if (run_dir / "status.json").exists() else {}
        if resume and previous_status.get("state") in {"collapse", "failed", "external_failure"}:
            raise RuntimeError("This run stopped after collapse/failure; recovery requires an explicitly reviewed new run")
        if not resume and ((run_dir / "latest.pt").exists() or (run_dir / "unique.sqlite3").exists()):
            raise FileExistsError("Run already contains training state; use explicit resume")
        checkpoint = None
        if resume:
            heartbeat(run_dir, "checkpoint_load")
            checkpoint = torch.load(ensure_artifact_path(run_dir / "latest.pt"), map_location="cpu", weights_only=False)
            if checkpoint["config_fingerprint"] != contract or checkpoint["tokenizer_fingerprint"] != tokenizer_fingerprint:
                raise ValueError("Configuration/tokenizer changed; refusing inconsistent resume")
            state.update(checkpoint["progress"])
        seed = int(config.get("seed", 0))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        with gpu_lock(str(device)):
            model = BinaryCTCModel(config["model"]) if model is None else model
            model = model.to(device=device, dtype=torch.float32)
            optimizer = optimizer_for(model, cfg)
            health = HealthMonitor(config.get("health", {}))
            source_health = {}
            extractor = CausalLogMel(**config.get("features", {}))
            heartbeat(run_dir, "validation_load", step=state["step"])
            validation_path = ensure_artifact_path(run_dir / "validation.pt")
            validation_fingerprint = digest(validation_path)
            if checkpoint and checkpoint.get("validation_fingerprint") != validation_fingerprint:
                raise ValueError("Validation artifact changed; refusing inconsistent resume")
            validation = torch.load(validation_path, map_location="cpu", weights_only=False)
            # Exclude every prepared holdout recording even when this run uses
            # only a subset to keep periodic validation inexpensive.
            validation_ids = {example["content_id"] for example in validation}
            validation = validation[:int(cfg.get("validation_examples", len(validation)))]
            if not validation:
                raise ValueError("Prepared validation set is empty")
            for example in validation:
                example["target"] = tokenizer.encode(example["text"])
                audio = example["audio"]
                required = len(example["target"]) + sum(a == b for a, b in zip(example["target"], example["target"][1:]))
                if (int(example["sample_rate"]) != 16000 or audio.ndim != 1 or not audio.numel()
                        or not torch.isfinite(audio).all() or not example["target"]
                        or required > ((audio.numel() + 159) // 160 + 7) // 8):
                    raise ValueError("Prepared validation contains an invalid/CTC-unalignable example")
            if dataset is None:
                from .data import StreamingSpeechDataset
                data_cfg = config["data"]
                supported = {key: data_cfg[key] for key in (
                    "shuffle_buffer", "min_seconds", "max_seconds", "max_consecutive_bad",
                    "max_rejection_fraction", "rejection_fraction_min_samples",
                ) if key in data_cfg}
                dataset = StreamingSpeechDataset(data_cfg["train_sources"], seed=seed, repeat=True,
                                                 casefold=tokenizer.casefold, **supported)
            if checkpoint:
                model.load_state_dict(checkpoint["model"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                health.load_state_dict(checkpoint["health"])
                for name, health_state in checkpoint.get("source_health", {}).items():
                    source_health[name] = HealthMonitor(config.get("health", {}))
                    source_health[name].load_state_dict(health_state)
                dataset.load_state_dict(checkpoint["stream"])
            batcher = Batcher(dataset, tokenizer, cfg, validation_ids, checkpoint.get("batcher") if checkpoint else None)
            ledger = UniqueLedger(run_dir / "unique.sqlite3", state["step"])
            if checkpoint and ledger.summary() != checkpoint["unique"]:
                raise ValueError("Unique-audio ledger disagrees with recovery checkpoint")
            if checkpoint:
                _restore_rng(checkpoint["rng"])
            check_finite_state(model, optimizer)
            base_wall = float(state.get("wall_seconds", 0))
            def update_progress():
                state["wall_seconds"] = base_wall + time.monotonic() - started
                state.update(ledger.summary())
                # Reader acceptance includes prefetch and examples subsequently
                # rejected by CTC/holdout guards; it is NOT optimized exposure.
                state["reader_stats"] = deepcopy(getattr(dataset, "stats", {}))
            def save(validated=False):
                heartbeat(run_dir, "checkpoint", step=state["step"])
                check_free_space(float(cfg.get("minimum_free_disk_gib", 20)))
                check_finite_state(model, optimizer)
                update_progress()
                atomic_torch_save(run_dir / "latest.pt", {
                    "version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "health": health.state_dict(), "rng": _rng_state(), "stream": dataset.state_dict(),
                    "source_health": {name: monitor.state_dict() for name, monitor in source_health.items()},
                    "batcher": batcher.state_dict(), "progress": dict(state), "unique": ledger.summary(),
                    "config_fingerprint": contract, "tokenizer_fingerprint": tokenizer_fingerprint,
                    "validation_fingerprint": validation_fingerprint,
                    "model_fingerprint": object_digest(config["model"]), "quantization": quantization,
                })
                if (validated and health.validation_bad_count == health.loss_bad_count == health.zero_grad_count == 0
                        and all(monitor.validation_bad_count == 0 for monitor in source_health.values())):
                    _promote(run_dir)
                write_json(run_dir / "status.json", state)
            def validate():
                metrics = evaluate(model, validation, tokenizer, extractor, device, run_dir,
                                   state["step"], int(cfg.get("max_batch_size", 8)))
                # Keep the actual failing observation in the append-only log.
                append_json(run_dir / "metrics.jsonl", {"kind": "validation", "step": state["step"], **metrics})
                health.observe_validation(state["step"], metrics)
                for name, source_metrics in metrics["sources"].items():
                    monitor = source_health.setdefault(name, HealthMonitor(config.get("health", {})))
                    if monitor.phase != health.phase:
                        monitor.validation_bad_count = 0
                        monitor.phase = health.phase
                    monitor.grace_until_step = health.grace_until_step
                    try:
                        monitor.observe_validation(state["step"], source_metrics)
                    except TrainingCollapse as exc:
                        raise TrainingCollapse(f"Validation collapse in source {name}: {exc.reason}",
                                               {**exc.metrics, "source": name}) from exc
                state["last_validation_step"] = state["step"]
                state["validation"] = metrics
                return metrics
            def handle_signal(signum, _frame):
                stop_requested["reason"] = signal.Signals(signum).name
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    handlers[sig] = signal.getsignal(sig)
                    signal.signal(sig, handle_signal)
                except ValueError:  # A test may call the worker in a thread.
                    handlers.pop(sig, None)
            quantization = checkpoint["quantization"] if checkpoint else quantization_at(0, config.get("quantization", {}), config["model"].get("quantizer", "binary"))
            model.set_quantization(quantization["weight"], quantization["activation"])
            state["state"] = "running"
            owns_run = True
            write_json(run_dir / "status.json", state)
            if not checkpoint:
                validate()
                save(validated=True)
            model.train()
            while state["step"] < maximum:
                update_progress()
                reason = stop_requested["reason"]
                if (run_dir / "STOP").exists():
                    reason = "STOP file"
                if stop_after is not None and state["step"] >= stop_after:
                    reason = "requested step limit"
                if cfg.get("max_audio_hours") and state["audio_seconds"] >= float(cfg["max_audio_hours"]) * 3600:
                    reason = "audio exposure budget"
                if cfg.get("max_wall_hours") and state["wall_seconds"] >= float(cfg["max_wall_hours"]) * 3600:
                    reason = "wall-clock budget"
                if reason:
                    state.update(state="stopped", reason=reason)
                    save()
                    notifier.notify("stopped", reason, step=state["step"])
                    return state
                step = state["step"] + 1
                quantization = quantization_at(state["step"], config.get("quantization", {}), config["model"].get("quantizer", "binary"))
                model.set_quantization(quantization["weight"], quantization["activation"])
                lr, weight_decay = schedule_at(state["step"], cfg)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                    group["weight_decay"] = weight_decay if group["decay_enabled"] else 0.
                optimizer.zero_grad(set_to_none=True)
                batches = []
                for micro in range(accumulation):
                    heartbeat(run_dir, "stream_read", step=step, microbatch=micro)
                    batches.append(batcher.next())
                examples_this_step = sum(len(batch) for batch in batches)
                loss_sum = 0.
                for micro, batch in enumerate(batches):
                    heartbeat(run_dir, "train", step=step, microbatch=micro)
                    check_finite_state(model)
                    features, lengths, targets, target_lengths = _collate(batch, extractor, device)
                    features = spec_augment(features, lengths, config.get("augmentation"), state["step"])
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        logits, output_lengths = model(features, lengths)
                    losses = _loss(logits, output_lengths, targets, target_lengths, batch)
                    loss_sum += float(losses.detach().sum())
                    (losses.sum() / examples_this_step).backward()
                    del logits, losses, features
                _finite_tensors(((name, p.grad) for name, p in model.named_parameters() if p.grad is not None), "gradients")
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1)), error_if_nonfinite=True))
                loss = loss_sum / examples_this_step
                health.observe_train(step, loss, grad_norm, quantization=quantization)
                heartbeat(run_dir, "optimizer", step=step)
                optimizer.step()
                check_finite_state(model, optimizer)
                examples = [example for batch in batches for example in batch]
                ledger.add(examples, step)
                state["step"] = step
                state["audio_seconds"] += sum(e["seconds"] for e in examples)
                state["optimized_examples"] += len(examples)
                for example in examples:
                    totals = state["optimized_sources"].setdefault(str(example["source"]),
                                                                  {"examples": 0, "audio_seconds": 0.})
                    totals["examples"] += 1
                    totals["audio_seconds"] += example["seconds"]
                update_progress()
                if step % int(cfg.get("log_every", 10)) == 0:
                    heartbeat(run_dir, "metrics", step=step)
                    append_json(run_dir / "metrics.jsonl", {
                        "kind": "train", "step": step, "loss": loss, "grad_norm": grad_norm,
                        "lr": lr, "weight_decay": weight_decay, "quantization": quantization,
                        "examples": len(examples), "rejected": batcher.rejected, **state,
                    })
                    write_json(run_dir / "status.json", state)
                if step % int(cfg.get("eval_every", 500)) == 0:
                    validate()
                    save(validated=True)
                elif step % int(cfg.get("checkpoint_every", 250)) == 0:
                    save()
            if state.get("last_validation_step") != state["step"]:
                validate()
            state.update(state="completed", reason="maximum optimizer steps")
            save(validated=True)
            notifier.notify("completed", "Training reached its configured optimizer-step budget", step=state["step"])
            return state
    except BaseException as exc:
        # A failed state is diagnostic only: never checkpoint suspect weights or
        # overwrite either finite recovery file from an exception handler.
        state.update(state="collapse" if isinstance(exc, TrainingCollapse) else "failed",
                     reason=str(exc), error_type=type(exc).__name__)
        if dataset is not None:
            state["reader_stats"] = deepcopy(getattr(dataset, "stats", {}))
        detail = _safe({**state, "metrics": getattr(exc, "metrics", {}), "traceback": traceback.format_exc()})
        with contextlib.suppress(Exception):
            write_json(run_dir / "failure.json", detail)
            if owns_run or not (run_dir / "status.json").exists():
                write_json(run_dir / "status.json", _safe(state))
        # Raw HTTP exceptions/tracebacks can contain signed dataset URLs. Keep
        # them local; email only controlled collapse diagnostics or error type.
        public_reason = str(exc) if isinstance(exc, TrainingCollapse) else f"{type(exc).__name__}: worker failed; inspect failure.json"
        notifier.notify(state["state"], public_reason, step=state["step"],
                        error_type=type(exc).__name__, metrics=_safe(getattr(exc, "metrics", {})))
        raise
    finally:
        for sig, previous in handlers.items():
            signal.signal(sig, previous)
        if dataset is not None and callable(getattr(dataset, "close", None)):
            dataset.close()
        if ledger is not None:
            ledger.close()
