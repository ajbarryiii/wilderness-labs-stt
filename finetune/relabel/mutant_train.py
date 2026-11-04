"""Mutant (encoder probe 2): one Parakeet TDT decoder recipe trained on each arm's frozen-encoder features, its
held-out metric, and the dev and domain scores (mutant plan "Shared decoder", "Training protocol", "Evaluation").

    ./python mutant_train.py pilot --arm X --lr {1e-4,3e-4,1e-3} [--budget-minutes M]     one pilot (GPU; resumable)
    ./python mutant_train.py main --arm X [--budget-minutes M]                            the main run (GPU; resumable)
    ./python mutant_train.py extend --arm X --confirm-extension [--budget-minutes M]      the one extension epoch
    ./python mutant_train.py score --arm X [--stage main|extension] [--sets ...]          H4 + YODAS and T5 records
    ./python mutant_train.py probe --arm X [--steps N]                                    step time, memory, held-out
    ./python mutant_train.py status
    every command: [--smoke-shards N] (smoke layout, relabel_paths.SMOKE only) [--lock-wait S]
    smoke only: main/pilot --lr and --steps, score --limit N

Model (identical recipe for D-B, D-T and D-Q): B0 (Parakeet-TDT-0.6B-v2, evaluate.load_pretrained, FP32) supplies the
prediction network (its own weights, bitwise), the joint's architecture (encoder input 1024-d), the tokenizer, the TDT
loss configuration and the greedy TDT decoding. The joint starts fresh: NeMo builds B0's joint configuration under
torch.manual_seed(JOINT_SEED) (PyTorch's default Linear initialization, the same state in every arm). Prediction and
joint dropout are 0 (train.disable_pred_joint_dropout, as stage 2b); only the prediction and joint networks have
gradients; B0's own encoder is never run by the trainer (each arm's features come from mutant_features).

Training (one batch plan, mutant_data; global step t serves plan.position(t)): AdamW as train.make_optimizer
(train.param_groups: weight decay 0.01 on matrices; betas 0.9/0.98, eps 1e-8; FP32 parameters and state), gradient
norm clipped at 1.0, the fused joint+TDT loss (finetune/tdt-fused joint_loss: one loss call and one omega draw per
batch) under BF16 autocast on the stored bfloat16 features, and the non-finite guard of training/seam.py (a non-finite
loss or gradient norm skips the optimizer step; a run's fourth stops it). Learning rate per step (Schedule):
- pilot: lr x min(1, (t + 1) / 300) for t < 3,000 (the first 3,000 positions of epoch 0); not counted in S;
- main: S = 2 epochs of the plan; lr x (t + 1) / W for t < W = round(2% of S), then
  lr x (0.1 + 0.9 x (1 + cos(pi (t - W) / (S - W))) / 2), which is 10% at t = S;
- extension (at most once, all arms): epoch 2 (t in [S, 3 epochs)) at a constant 10% of lr, from the step-S state.
The Python, numpy, torch and CUDA RNGs are seeded with TRAIN_SEED at a run's start (pilot, main), identically in every
arm; a checkpoint (ckpt/latest.pt: prediction and joint state, AdamW state, step, every RNG state, the non-finite
count, the held-out evaluations) is written every --ckpt-minutes, at a budget or signal stop and at the end, and a
restart resumes from it (refused for another configuration: ResumeError, exit 78). The learning rate is a function of
the step, so the step is the scheduler's whole state.

Held-out metric (the only metric for learning-rate selection and the plateau check): greedy TDT decoding of the
held-out slice's stored features (B0's decoding, strict FP32, batches of mutant_features.make_batches), each clip
scored by evaluate.score against its stored reference; the metric is the corpus WER over the scored clips (100 x
errors / reference words), per source for information. Every RNG is restored afterwards and decoding never changes
the training trajectory. Evaluated at step 3,000 (pilot), at round(25, 50, 75, 100% of S) (main) and at the
extension's end; the step-75% and step-S weights are kept (ckpt/step-<n>.pt). Pilot checkpoints are discarded at the
end. select_lr: the lowest held-out WER at step 3,000, ties to the smaller rate.

Scores: the final weights of a stage, decoded greedily on the arm's features (mutant_features.encode_clips, the code
path of the training features) for H4 + YODAS (parakeet-ternary manifests) and T5 (the domain manifests), records in
evaluate.py's per-set layout: $RELABEL_ROOT/encoder-probe/decoder/<arm>/<set>.json (main) and
.../decoder/<arm>/extension/<set>.json (after the extension).

Layout: runs in $RELABEL_ROOT/encoder-probe/mutant/runs/<arm>/{pilot-lr<lr>,main,extension}/; --smoke-shards N uses
the smoke layout under relabel_paths.SMOKE only. GPU discipline: the parakeet GPU lock (relabel_paths.GPU_LOCK) is
taken before anything touches CUDA; the job's VRAM lease is applied on the current device index.
"""
from __future__ import annotations

import train  # noqa: I001  finetune/parakeet-ternary/train.py first: it configures CUDA/numba

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import random
import resource
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

import evaluate  # finetune/parakeet-ternary
import mutant_data as md
import mutant_features as mf
import relabel_paths
from training import seam as tseam
from training import t5clusters

PLAN = md.PLAN
PILOT_LRS = (1e-4, 3e-4, 1e-3)
LR_NAMES = {1e-4: "lr1e-4", 3e-4: "lr3e-4", 1e-3: "lr1e-3"}
PILOT_STEPS, PILOT_WARMUP = 3000, 300
EPOCHS, EXTENSION_EPOCHS = 2, 1
WARMUP_FRACTION, FINAL_FRACTION = 0.02, 0.10
EVAL_FRACTIONS, KEEP_FRACTION = (0.25, 0.5, 0.75, 1.0), 0.75
JOINT_SEED, TRAIN_SEED = 20261105, 20261106
BETAS, EPS, WEIGHT_DECAY, CLIP = (0.9, 0.98), 1e-8, 0.01, 1.0
MAX_NONFINITE = tseam.MAX_NONFINITE_STEPS
CKPT_FORMAT = "relabel-mutant-ckpt-v1"
TRAINED_PREFIXES = ("decoder.", "joint.")
LOG_EVERY = 50
EXIT_RESUME_MISMATCH = 78
SCORE_CHUNK = 512                  # clips encoded and decoded together when scoring a set
STAGES = ("pilot", "main", "extension")


class ResumeError(RuntimeError):
    pass


class NonfiniteSteps(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [mutant] {msg}", flush=True)


# ----------------------------------------------------------------------------- schedule

@dataclasses.dataclass(frozen=True)
class Schedule:
    stage: str
    lr: float
    n_batches: int
    pilot_steps: int = PILOT_STEPS
    pilot_warmup: int = PILOT_WARMUP

    @property
    def total(self) -> int:
        """S: the main run's budget, EPOCHS epochs of the plan."""
        return EPOCHS * self.n_batches

    @property
    def warmup(self) -> int:
        return max(1, round(WARMUP_FRACTION * self.total))

    @property
    def start_step(self) -> int:
        return self.total if self.stage == "extension" else 0

    @property
    def end_step(self) -> int:
        if self.stage == "pilot":
            return self.pilot_steps
        if self.stage == "main":
            return self.total
        return self.total + EXTENSION_EPOCHS * self.n_batches

    def factor(self, step: int) -> float:
        if self.stage == "pilot":
            return min(1.0, (step + 1) / self.pilot_warmup)
        if self.stage == "extension":
            return FINAL_FRACTION
        w, s = self.warmup, self.total
        if step < w:
            return (step + 1) / w
        x = min(1.0, (step - w) / max(1, s - w))
        return FINAL_FRACTION + (1 - FINAL_FRACTION) * 0.5 * (1 + math.cos(math.pi * x))

    def lr_at(self, step: int) -> float:
        return self.lr * self.factor(step)

    @property
    def eval_steps(self) -> list[int]:
        if self.stage == "main":
            return [round(f * self.total) for f in EVAL_FRACTIONS]
        return [self.end_step]

    @property
    def keep_steps(self) -> list[int]:
        if self.stage == "main":
            return [round(KEEP_FRACTION * self.total), self.total]
        return [self.end_step] if self.stage == "extension" else []

    def to_dict(self) -> dict:
        return {**dataclasses.asdict(self), "S": self.total, "warmup": self.warmup, "start_step": self.start_step,
                "end_step": self.end_step, "eval_steps": self.eval_steps, "keep_steps": self.keep_steps,
                "rule": {"pilot": "lr * min(1, (t+1)/pilot_warmup), t < pilot_steps (epoch 0 positions)",
                         "main": "lr * (t+1)/W for t < W = round(0.02 S); then lr * (0.1 + 0.9 * (1 + cos(pi (t-W)/(S-W)))/2)",
                         "extension": "lr * 0.1 for t in [S, S + n_batches)"}[self.stage]}


def schedule(stage: str, lr: float, n_batches: int, pilot_steps: int | None = None,
             pilot_warmup: int | None = None) -> Schedule:
    if stage not in STAGES:
        raise ValueError(stage)
    return Schedule(stage, float(lr), int(n_batches), pilot_steps or PILOT_STEPS, pilot_warmup or PILOT_WARMUP)


def select_lr(results: dict[float, dict]) -> float:
    """The pilot rate with the lowest held-out WER at its end (errors / words, exact); ties to the smaller rate."""
    if set(results) != set(PILOT_LRS):
        raise ValueError(f"pilot results for {sorted(results)}, need {sorted(PILOT_LRS)}")
    from fractions import Fraction
    key = {lr: Fraction(int(r["errors"]), int(r["words"])) for lr, r in results.items()}
    return min(sorted(key), key=lambda lr: key[lr])


# ----------------------------------------------------------------------------- model

def state_sha256(state: dict) -> str:
    digest = hashlib.sha256()
    for name, t in sorted(state.items()):
        t = t.detach().cpu().contiguous()
        digest.update(f"{name}|{t.dtype}|{tuple(t.shape)}\n".encode())
        digest.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return digest.hexdigest()


def fresh_joint_state(model, seed: int) -> dict[str, torch.Tensor]:
    """The state of a new joint built by NeMo from the model's joint configuration under torch.manual_seed(seed)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        joint = model.from_config_dict(model.cfg.joint)
    return {k: v.detach().clone().float() for k, v in joint.state_dict().items()}


def init_decoder(model, joint_seed: int = JOINT_SEED) -> dict:
    """B0's prediction network as loaded, a fresh joint, dropout 0, gradients only for prediction and joint."""
    zeroed = train.disable_pred_joint_dropout(model)
    pred = state_sha256(model.decoder.state_dict())
    model.joint.load_state_dict(fresh_joint_state(model, joint_seed), strict=True)
    model.requires_grad_(False)
    model.decoder.requires_grad_(True)
    model.joint.requires_grad_(True)
    model.eval()
    return {"pred_sha256": pred, "joint_init_sha256": state_sha256(model.joint.state_dict()), "joint_seed": joint_seed,
            "pred_joint_dropout_modules_zeroed": zeroed,
            "joint_encoder_input": int(model.joint.enc.in_features),
            "init": "prediction network: B0's weights; joint: NeMo's build of B0's joint config under the seed"}


def trainable_parameters(model) -> list[torch.nn.Parameter]:
    out = [p for n, p in model.named_parameters() if n.startswith(TRAINED_PREFIXES)]
    others = [n for n, p in model.named_parameters() if p.requires_grad and not n.startswith(TRAINED_PREFIXES)]
    if others or not all(p.requires_grad for p in out):
        raise RuntimeError(f"only the prediction and joint networks may train ({others[:3]})")
    return out


def trained_state(model) -> dict[str, torch.Tensor]:
    return {k: v for k, v in model.state_dict().items() if k.startswith(TRAINED_PREFIXES)}


def load_trained(model, state: dict) -> None:
    own = trained_state(model)
    if set(own) != set(state):
        raise ValueError("prediction/joint keys differ")
    with torch.no_grad():
        for k, v in own.items():
            v.copy_(state[k].to(v.dtype))


def make_optimizer(model, lr: float):
    return torch.optim.AdamW(train.param_groups(model, WEIGHT_DECAY), lr=lr, betas=BETAS, eps=EPS)


def load_tdt_fused():
    """finetune/tdt-fused's module, on the path after every root as train_b0prime.py puts it; Triton's libcuda found
    on LD_LIBRARY_PATH when /sbin/ldconfig is missing (train_b0prime._triton_libcuda's rule)."""
    if not os.environ.get("TRITON_LIBCUDA_PATH"):
        for d in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
            if d and os.path.exists(os.path.join(d, "libcuda.so.1")):
                os.environ["TRITON_LIBCUDA_PATH"] = d
                break
    where = str(Path(__file__).resolve().parent.parent / "tdt-fused")
    if where not in sys.path:
        sys.path.append(where)
    import tdt_fused
    return tdt_fused


def fused_loss(model, batch: dict) -> torch.Tensor:
    """The TDT loss of one batch through the fused joint+TDT seam (one loss call), under BF16 autocast."""
    tdt_fused = load_tdt_fused()
    train.restore_loss_reduction(model)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        dec, _, _ = model.decoder(targets=batch["tokens"], target_length=batch["token_len"])
        loss = tdt_fused.joint_loss(model.joint, batch["feats"], dec, batch["feat_len"], batch["tokens"],
                                    batch["token_len"])
    if loss.dim() != 0:
        raise RuntimeError(f"TDT loss is not a scalar: {tuple(loss.shape)}")
    return loss


# ----------------------------------------------------------------------------- RNG

def seed_training(seed: int = TRAIN_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextlib.contextmanager
def isolated_rng():
    state = rng_state()
    try:
        yield
    finally:
        set_rng_state(state)


# ----------------------------------------------------------------------------- decoding and the held-out metric

def pad_features(feats: list[torch.Tensor], dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """[B, D, T] zero-padded and [B] lengths from per-clip [T_i, D] features."""
    lengths = torch.tensor([int(f.shape[0]) for f in feats], dtype=torch.long)
    dim = int(feats[0].shape[1])
    out = torch.zeros(len(feats), dim, max(1, int(lengths.max())), dtype=dtype)
    for row, f in enumerate(feats):
        out[row, :, :f.shape[0]] = f.T.to(dtype)
    return out, lengths


def reset_decoder_cache(model) -> None:
    """Drop the greedy decoder's CUDA-graph state: NeMo's label-looping computer caches the prediction network's
    state and projected output after the start symbol when it builds its graphs, and keeps them while the batch shape
    fits, so after training steps they would be the old weights'. The pinned graph mode is kept."""
    computer = getattr(getattr(getattr(model, "decoding", None), "decoding", None), "decoding_computer", None)
    if computer is not None and hasattr(computer, "reset_cuda_graphs_state"):
        computer.reset_cuda_graphs_state()


def decode_features(model, feats: list[torch.Tensor], durations: list[float], device) -> list[str]:
    """Greedy TDT transcripts of per-clip features with the model's decoding (B0's greedy_batch config), strict FP32,
    eval mode, in mutant_features.make_batches batches; RNG and modes restored. The decoder's cached graph state is
    reset first (reset_decoder_cache), so every pass decodes with the current weights."""
    device = torch.device(device)
    reset_decoder_cache(model)
    if device.type == "cuda":
        import score_domain
        score_domain.binary_on_path()
        import binary_train
        binary_train.pin_decoder_graphs_mode(model, "mutant decoding")
    modes = (model.decoder.training, model.joint.training)
    texts: list[str | None] = [None] * len(feats)
    try:
        model.decoder.eval()
        model.joint.eval()
        with isolated_rng(), evaluate.strict_fp32(device), torch.no_grad():
            for batch in mf.make_batches(durations):
                enc, lengths = pad_features([feats[i] for i in batch])
                hyps = model.decoding.rnnt_decoder_predictions_tensor(
                    encoder_output=enc.to(device), encoded_lengths=lengths.to(device), return_hypotheses=False)
                if isinstance(hyps, tuple):
                    hyps = hyps[0]
                for i, h in zip(batch, hyps):
                    texts[i] = h.text if hasattr(h, "text") else str(h)
    finally:
        model.decoder.train(modes[0])
        model.joint.train(modes[1])
    return texts


def heldout_metric(model, items: list[dict], device) -> dict:
    """The held-out metric: corpus WER (points) of greedy TDT on the held-out features, evaluate.score per clip."""
    hyps = decode_features(model, [it["feats"] for it in items], [it["duration"] for it in items], device)
    errors = words = scored = 0
    sources = {s: {"errors": 0, "words": 0} for s in md.SOURCES}
    for it, h in zip(items, hyps):
        r = evaluate.score(it["text"], h)
        if not r["scored"]:
            continue
        e, w = r["S"] + r["D"] + r["I"], len(r["ref_norm"].split())
        errors, words, scored = errors + e, words + w, scored + 1
        sources[it["source"]]["errors"] += e
        sources[it["source"]]["words"] += w
    for s in sources.values():
        s["wer"] = 100.0 * s["errors"] / s["words"] if s["words"] else None
    return {"errors": errors, "words": words, "wer": 100.0 * errors / words if words else None,
            "utterances": len(items), "scored": scored, "sources": sources, "hyps": hyps}


# ----------------------------------------------------------------------------- the trainer

def config_sha256(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    torch.save(payload, tmp)
    with open(tmp, "rb") as h:
        os.fsync(h.fileno())
    if path.exists():
        os.replace(path, path.with_name("previous.pt" if path.name == "latest.pt" else path.name + ".previous"))
    os.replace(tmp, path)


class Trainer:
    """One run (pilot, main or extension) of one arm: train from schedule.start_step (or a checkpoint) to its end."""

    def __init__(self, model, batches, sched: Schedule, run_dir: Path, config: dict, device, loss_fn=None,
                 heldout: list[dict] | None = None, ckpt_minutes: float = 20.0, budget_minutes: float | None = None,
                 stop_after: int | None = None, init: dict | None = None):
        self.model, self.batches, self.schedule = model, batches, sched
        self.run_dir, self.device = Path(run_dir), torch.device(device)
        self.config = {**config, "schedule": sched.to_dict()}
        self.config_sha = config_sha256(self.config)
        self.loss_fn = loss_fn or fused_loss
        self.heldout = heldout
        self.ckpt_minutes, self.budget_minutes, self.stop_after = ckpt_minutes, budget_minutes, stop_after
        model.decoder.to(self.device)
        model.joint.to(self.device)
        self.params = trainable_parameters(model)
        self.optimizer = make_optimizer(model, sched.lr)
        self.step, self.nonfinite, self.evals = sched.start_step, 0, []
        self.acc = {"n": 0, "loss": 0.0, "norm": 0.0, "seconds": 0.0, "time": 0.0}
        self.stop_reason: str | None = None
        latest = self.run_dir / "ckpt" / "latest.pt"
        cfg_path = self.run_dir / "config.json"
        if cfg_path.exists() and json.loads(cfg_path.read_text()).get("config_sha256") != self.config_sha:
            raise ResumeError(f"{self.run_dir} was made by another configuration")
        if latest.exists():
            self._restore(torch.load(latest, map_location="cpu", weights_only=False), resume=True)
            log(f"resumed {self.run_dir} at step {self.step}")
        elif init is not None:
            self._restore(init, resume=False)
            if self.step != sched.start_step:
                raise ResumeError(f"the initial state is at step {self.step}, the run starts at {sched.start_step}")
            self.nonfinite, self.evals = 0, []
        else:
            seed_training()
        if not cfg_path.exists():
            md.write_json(cfg_path, {"config_sha256": self.config_sha, "config": self.config,
                                     "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        self._train_modes()

    # --- state
    def _train_modes(self) -> None:
        self.model.eval()
        self.model.decoder.train()
        self.model.joint.train()

    def payload(self, complete: bool = False) -> dict:
        return {"format": CKPT_FORMAT, "config_sha256": self.config_sha, "step": self.step,
                "model": {k: v.detach().cpu().clone() for k, v in trained_state(self.model).items()},
                "optimizer": self.optimizer.state_dict(), "rng": rng_state(), "nonfinite": self.nonfinite,
                "evals": list(self.evals), "complete": complete, "stage": self.schedule.stage}

    def _restore(self, payload: dict, resume: bool) -> None:
        if payload.get("format") != CKPT_FORMAT:
            raise ResumeError("not a mutant checkpoint")
        if resume and payload.get("config_sha256") != self.config_sha:
            raise ResumeError(f"{self.run_dir}: the checkpoint was written by another configuration")
        load_trained(self.model, payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        set_rng_state(payload["rng"])
        self.step, self.nonfinite, self.evals = int(payload["step"]), int(payload["nonfinite"]), list(payload["evals"])
        if resume:
            self._prune_metrics()

    def checkpoint(self, complete: bool = False) -> None:
        _save(self.run_dir / "ckpt" / "latest.pt", self.payload(complete))
        self._last_ckpt = time.monotonic()

    def _prune_metrics(self) -> None:
        path = self.run_dir / "metrics.jsonl"
        if path.exists():
            rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
            path.write_text("".join(json.dumps(r) + "\n" for r in rows if r["step"] <= self.step))

    # --- steps
    def run_steps(self, n: int) -> None:
        for _ in range(n):
            self._step()

    def _step(self) -> None:
        t0 = time.perf_counter()
        b = self.batches.batch(self.step)
        lr = self.schedule.lr_at(self.step)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        batch = {"feats": b["feats"].to(self.device, non_blocking=True), "feat_len": b["feat_len"].to(self.device),
                 "tokens": b["tokens"].to(self.device), "token_len": b["token_len"].to(self.device)}
        train.reseed_cudnn_rnn_dropout()
        loss = self.loss_fn(self.model, batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.params, CLIP)
        values = (float(loss.item()), float(norm.item()))
        if all(math.isfinite(v) for v in values):
            self.optimizer.step()
            self.acc["n"] += 1
            self.acc["loss"] += values[0]
            self.acc["norm"] += values[1]
            self.acc["seconds"] += float(b.get("seconds", 0.0))
        else:
            self.optimizer.zero_grad(set_to_none=True)
            if self.nonfinite >= MAX_NONFINITE:
                raise NonfiniteSteps(f"non-finite loss or gradient norm at step {self.step + 1}: {values}; "
                                     f"{self.nonfinite} steps already skipped (at most {MAX_NONFINITE})")
            self.nonfinite += 1
            log(f"non-finite loss or gradient norm at step {self.step + 1}: {values}; optimizer step skipped "
                f"({self.nonfinite}/{MAX_NONFINITE})")
        self.step += 1
        self.acc["time"] += time.perf_counter() - t0
        if self.step % LOG_EVERY == 0 or self.step == self.schedule.end_step:
            self._flush(lr)
        if self.step in self.schedule.eval_steps:
            self.evaluate()
        if self.step in self.schedule.keep_steps:
            (self.run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
            torch.save({"step": self.step, "model": {k: v.detach().cpu().clone() for k, v in
                                                     trained_state(self.model).items()},
                        "config_sha256": self.config_sha}, self.run_dir / "ckpt" / f"step-{self.step}.pt")

    def _flush(self, lr: float) -> None:
        a = self.acc
        if a["n"] or a["time"]:
            row = {"step": self.step, "lr": lr, "loss": a["loss"] / max(a["n"], 1), "norm": a["norm"] / max(a["n"], 1),
                   "steps": a["n"], "audio_seconds": a["seconds"], "step_time_s": a["time"] / max(a["n"], 1),
                   "nonfinite": self.nonfinite}
            train.append_jsonl(self.run_dir / "metrics.jsonl", row)
        self.acc = {"n": 0, "loss": 0.0, "norm": 0.0, "seconds": 0.0, "time": 0.0}

    def evaluate(self) -> dict:
        if self.heldout is None:
            raise RuntimeError("no held-out slice")
        t0 = time.perf_counter()
        r = heldout_metric(self.model, self.heldout, self.device)
        md.write_json(self.run_dir / "heldout" / f"step-{self.step}.json",
                      {"step": self.step, "hyps": {it["id"]: h for it, h in zip(self.heldout, r["hyps"])}})
        row = {"step": self.step, **{k: r[k] for k in ("wer", "errors", "words", "utterances", "scored", "sources")}}
        self.evals.append(row)
        md.write_json(self.run_dir / "heldout.json", {"evals": self.evals})
        log(f"{self.run_dir.name} step {self.step}: held-out WER {row['wer']:.3f} ({time.perf_counter() - t0:.0f} s)")
        return row

    # --- the run
    def _stop(self, signum, frame) -> None:
        self.stop_reason = "signal"

    def run(self) -> str:
        if (self.run_dir / "summary.json").exists() and self.step >= self.schedule.end_step:
            return "complete"
        handlers = {}
        if threading.current_thread() is threading.main_thread():
            for s in (signal.SIGTERM, signal.SIGINT):
                handlers[s] = signal.signal(s, self._stop)
        start = self._last_ckpt = time.monotonic()
        try:
            while self.step < self.schedule.end_step:
                self._step()
                if self.step >= self.schedule.end_step:
                    break
                reason = self.stop_reason
                if self.stop_after is not None and self.step >= self.stop_after:
                    reason = "stopped"
                elif self.budget_minutes is not None and time.monotonic() - start >= self.budget_minutes * 60:
                    reason = "budget"
                if reason:
                    self.checkpoint()
                    log(f"{self.run_dir.name}: {reason} stop at step {self.step}; the next run resumes")
                    return reason
                if time.monotonic() - self._last_ckpt >= self.ckpt_minutes * 60:
                    self.checkpoint()
        finally:
            for s, h in handlers.items():
                signal.signal(s, h)
        self.checkpoint(complete=True)
        final = {k: v.detach().cpu().clone() for k, v in trained_state(self.model).items()}
        torch.save({"step": self.step, "model": final, "config_sha256": self.config_sha}, self.run_dir / "final.pt")
        md.write_json(self.run_dir / "summary.json", {
            "status": "complete", "stage": self.schedule.stage, "lr": self.schedule.lr, "steps": self.step,
            "evals": self.evals, "nonfinite_skipped_steps": self.nonfinite, "config_sha256": self.config_sha,
            "final_weights_sha256": state_sha256(final), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                         time.gmtime())})
        return "complete"


# ----------------------------------------------------------------------------- data for training

class FeatureBatches:
    """Global step -> the plan's batch of an arm's stored training features (pool files read whole, the next pool
    prefetched in a thread)."""

    def __init__(self, lay: md.Layout, arm: str, data: md.Data, identity: dict):
        self.lay, self.arm, self.data, self.plan = lay, arm, data, data.plan
        self.identity = identity
        self.cache: dict[int, mf.FeatureFile] = {}
        self.pending: dict[int, object] = {}
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.wait_s = 0.0

    def _paths(self, p: int):
        return mf.store_paths(self.lay.features / self.arm, "pools", p)

    def _load(self, p: int) -> mf.FeatureFile:
        f = mf.FeatureFile(self._paths(p))
        if f.meta["identity"] != self.identity:
            raise mf.FeatureError(f"pool {p}: features of another identity")
        return f

    def _pool_sequence(self, epoch: int) -> list[int]:
        key = ("seq", epoch)
        if key not in self.__dict__:
            seq = []
            for b in self.plan.epoch_order(epoch).tolist():
                p = self.plan.pool_of_batch(b)
                if not seq or seq[-1] != p:
                    seq.append(p)
            self.__dict__[key] = seq
        return self.__dict__[key]

    def _get(self, p: int, epoch: int) -> mf.FeatureFile:
        if p not in self.cache:
            t0 = time.perf_counter()
            fut = self.pending.pop(p, None)
            self.cache = {p: fut.result() if fut is not None else self._load(p)}
            self.wait_s += time.perf_counter() - t0
            seq = self._pool_sequence(epoch)
            k = seq.index(p)
            nxt = seq[k + 1] if k + 1 < len(seq) else self._pool_sequence(epoch + 1)[0]
            if nxt != p and nxt not in self.pending:
                self.pending = {nxt: self.pool.submit(self._load, nxt)}
        return self.cache[p]

    def batch(self, step: int) -> dict:
        epoch, _, b = self.plan.position(step)
        p = self.plan.pool_of_batch(b)
        f = self._get(p, epoch)
        ids = [self.plan.ids[int(s)] for s in self.plan.batch(b)]
        feats, lengths = pad_features([f.get(i) for i in ids], dtype=mf.STORE_DTYPE)
        toks = [self.data.clips[i]["tokens"] for i in ids]
        token_len = torch.tensor([len(t) for t in toks], dtype=torch.long)
        tokens = torch.zeros(len(ids), max(1, int(token_len.max())), dtype=torch.long)
        for r, t in enumerate(toks):
            tokens[r, :len(t)] = torch.tensor(t, dtype=torch.long)
        return {"feats": feats, "feat_len": lengths, "tokens": tokens, "token_len": token_len, "ids": ids,
                "seconds": sum(self.data.clips[i]["duration"] for i in ids)}


def heldout_items(lay: md.Layout, arm: str, data: md.Data, identity: dict) -> list[dict]:
    items = []
    for p in range(data.plan.n_pools):
        f = mf.FeatureFile(mf.store_paths(lay.features / arm, "heldout", p))
        if f.meta["identity"] != identity:
            raise mf.FeatureError(f"held-out pool {p}: features of another identity")
        for i in f.ids:
            c = data.clips[i]
            items.append({"id": i, "feats": f.get(i).clone(), "text": c["text"], "source": c["source"],
                          "duration": c["duration"]})
    if sorted(it["id"] for it in items) != data.heldout_ids():
        raise mf.FeatureError("the stored held-out features do not cover the held-out slice")
    return items


def feature_identity(lay: md.Layout, arm: str, data: md.Data) -> dict:
    """The arm's store identity, after checking that every pool (training and held-out) is committed with it."""
    ids = {}
    for p in range(data.plan.n_pools):
        for kind, split in (("pools", "train"), ("heldout", "heldout")):
            paths = mf.store_paths(lay.features / arm, kind, p)
            if not paths["meta"].exists():
                raise mf.FeatureError(f"{arm}: {kind} pool {p} is not extracted")
            meta = json.loads(paths["meta"].read_text())
            want = [i for i in data.pool_ids(p) if data.clips[i]["split"] == split]
            if meta["ids"] != want or meta["identity"].get("plan_sha256") != data.plan.sha256:
                raise mf.FeatureError(f"{arm}: {kind} pool {p} is of another plan")
            ids[json.dumps(meta["identity"], sort_keys=True)] = meta["identity"]
    if len(ids) != 1:
        raise mf.FeatureError(f"{arm}: pools of {len(ids)} identities")
    return next(iter(ids.values()))


# ----------------------------------------------------------------------------- run layout and configuration

def run_dir(lay: md.Layout, arm: str, stage: str, lr: float | None = None) -> Path:
    if stage == "pilot":
        return lay.runs / arm / f"pilot-{LR_NAMES[float(lr)]}"
    return lay.runs / arm / stage


def records_dir(lay: md.Layout, arm: str, stage: str) -> Path:
    return lay.records / arm if stage == "main" else lay.records / arm / stage


def run_config(arm: str, stage: str, lr: float, data, features: dict, model_info: dict, smoke: bool,
               loss: dict | None = None) -> dict:
    return {"plan": PLAN, "arm": arm, "stage": stage, "lr": float(lr), "plan_sha256": data.plan.sha256,
            "clips_sha256": data.summary["clips_sha256"], "features": features, "model": model_info,
            "seeds": {"joint": JOINT_SEED, "train": TRAIN_SEED, "plan": md.PLAN_SEED},
            "optimizer": {"kind": "AdamW", "betas": list(BETAS), "eps": EPS, "weight_decay": WEIGHT_DECAY,
                          "weight_decay_on": "matrices (train.param_groups)", "clip_grad_norm": CLIP},
            "precision": "FP32 parameters and AdamW state; BF16 autocast; bfloat16 stored features",
            "loss": loss, "max_nonfinite_steps": MAX_NONFINITE, "smoke": smoke}


def pilot_results(lay: md.Layout, arm: str) -> dict[float, dict]:
    out = {}
    for lr in PILOT_LRS:
        path = run_dir(lay, arm, "pilot", lr) / "summary.json"
        if not path.exists():
            raise SystemExit(f"{arm}: pilot {LR_NAMES[lr]} is not complete")
        s = json.loads(path.read_text())
        out[lr] = s["evals"][-1]
    return out


# ----------------------------------------------------------------------------- scoring

SCORE_SETS = tuple(relabel_paths.SETS) + tuple(t5clusters.SETS)     # dev and domain sets only: never a test set


def check_sets(names: list[str]) -> list[str]:
    """The requested sets, if every one is an H4, YODAS or T5 set (by name, never a path); else ValueError."""
    bad = [n for n in names if n not in SCORE_SETS]
    if bad:
        raise ValueError(f"only the dev and T5 sets are scored ({', '.join(SCORE_SETS)}); refused: {bad}")
    return list(names)


def set_manifests(names: list[str]) -> dict[str, Path]:
    import score_domain
    names = check_sets(names)
    out = {}
    t5 = [n for n in names if n in t5clusters.SETS]
    domain = score_domain.part_manifests(t5, None) if t5 else {}
    for n in names:
        out[n] = domain[n][0] if n in t5clusters.SETS else evaluate.resolve_manifest(n)[1]
    return out


def score_set(model, enc: mf.ArmEncoder, name: str, manifest: Path, limit: int | None, config: dict, device) -> dict:
    import score_teacher
    records = score_teacher.select_records(manifest, limit, 0)
    rows, padded, t0 = [], [], time.perf_counter()
    for lo in range(0, len(records), SCORE_CHUNK):
        chunk = records[lo:lo + SCORE_CHUNK]
        clips = [evaluate.load_audio(r["audio_filepath"]) for r in chunk]
        padded += [r["id"] for r, c in zip(chunk, clips) if len(c) < evaluate.MIN_DECODE_SAMPLES]
        durations = [r["duration"] for r in chunk]
        feats = mf.encode_clips(enc, clips, durations)
        hyps = decode_features(model, feats, durations, device)
        for r, h in zip(chunk, hyps):
            rows.append({"id": r["id"], "ref": r["text"], "hyp": h, **evaluate.score(r["text"], h),
                         "duration": r["duration"]})
    hours = sum(r["duration"] for r in rows) / 3600
    return {"source": config["source"], "set": name, "manifest": str(manifest),
            "manifest_sha256": evaluate.sha256_file(manifest), "utterances": len(rows), "limit": limit,
            "audio_hours": hours, "decoding": {"method": "arm features -> decoding.rnnt_decoder_predictions_tensor",
                                               "config": _decoding_config(model), "precision": "float32, TF32 off, "
                                               "autocast disabled; features stored in bfloat16",
                                               "batching": f"mutant_features.make_batches within chunks of "
                                                           f"{SCORE_CHUNK} id-sorted clips"},
            "normalizer": evaluate.NORMALIZER, "wer": evaluate.wer_summary(rows) if rows else None,
            "wall_s": time.perf_counter() - t0, "padded_short": len(padded), "padded_short_ids": sorted(padded),
            "min_decode_seconds": evaluate.MIN_DECODE_SECONDS, "config": config, "records": rows}


def _decoding_config(model) -> dict | None:
    try:
        from omegaconf import OmegaConf
        return OmegaConf.to_container(model.cfg.decoding, resolve=True)
    except Exception:
        return None


def scoring_model(weights: Path, device):
    model = evaluate.load_pretrained("cpu")
    init_decoder(model)
    payload = torch.load(weights, map_location="cpu", weights_only=False)
    load_trained(model, payload["model"])
    model.decoder.to(device)
    model.joint.to(device)
    model.eval()
    return model, payload


def score(lay: md.Layout, arm: str, stage: str, names: list[str], limit: int | None, device) -> dict:
    names = check_sets(names)
    weights = run_dir(lay, arm, stage) / "final.pt"
    if not (run_dir(lay, arm, stage) / "summary.json").exists():
        raise SystemExit(f"{arm} {stage} is not complete")
    model, payload = scoring_model(weights, device)
    enc = mf.load_arm(arm, device)
    source = {"arm": arm, "stage": stage, "weights": str(weights), "weights_sha256": evaluate.sha256_file(weights),
              "trained_sha256": state_sha256(payload["model"]), "step": payload["step"],
              "run_config_sha256": payload["config_sha256"], "encoder": enc.identity()}
    out_dir = records_dir(lay, arm, stage)
    summary = {}
    for name, manifest in set_manifests(names).items():
        config = {"source": source, "limit": limit, "feature_version": mf.FEATURE_VERSION}
        path = out_dir / f"{name}.json"
        if path.exists() and json.loads(path.read_text()).get("config") == config:
            summary[name] = json.loads(path.read_text())["wer"]
            continue
        r = score_set(model, enc, name, manifest, limit, config, device)
        md.write_json(path, r)
        summary[name] = r["wer"]
        log(f"{arm} {stage} {name}: n={r['utterances']} WER {100 * r['wer']['wer']:.2f} ({r['wall_s']:.0f} s)")
    return summary


# ----------------------------------------------------------------------------- probe

def probe(lay: md.Layout, arm: str, steps: int, device) -> dict:
    """Step time (the real loss, lr 3e-4, from the shared initial state, nothing saved), peak memory, data wait, and
    the held-out metric's time on the layout's held-out slice."""
    data = md.Data.load(lay)
    identity = feature_identity(lay, arm, data)
    model = evaluate.load_pretrained("cpu")
    info = init_decoder(model)
    sched = schedule("pilot", 3e-4, data.plan.n_batches, pilot_steps=steps + 1, pilot_warmup=max(1, steps // 10))
    batches = FeatureBatches(lay, arm, data, identity)
    tmp = lay.runs / arm / f"probe-tmp-{os.getpid()}"
    t = Trainer(model, batches, sched, tmp, {"probe": True}, device, heldout=None)
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        t.run_steps(1)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    items = heldout_items(lay, arm, data, identity)
    h0 = time.perf_counter()
    metric = heldout_metric(model, items, device)
    held_s = time.perf_counter() - h0
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    warm = times[5:] or times
    out = {"arm": arm, "steps_timed": len(warm), "step_time_s": float(np.mean(warm)),
           "step_time_p90_s": float(np.percentile(warm, 90)), "data_wait_s": batches.wait_s,
           "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
           "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
           "host_max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
           "n_batches": data.plan.n_batches, "heldout_clips": len(items), "heldout_metric_s": held_s,
           "heldout_wer_untrained_decoder_probe": metric["wer"], "model": info}
    md.write_json(lay.runs / arm / "probe.json", out)
    return out


# ----------------------------------------------------------------------------- CLI

def _status(lay: md.Layout) -> dict:
    out = {}
    for arm in mf.ARMS:
        for stage, lr in [("pilot", x) for x in PILOT_LRS] + [("main", None), ("extension", None)]:
            d = run_dir(lay, arm, stage, lr)
            key = f"{arm}/{d.name}"
            if (d / "summary.json").exists():
                s = json.loads((d / "summary.json").read_text())
                out[key] = {"done": True, "evals": [(e["step"], round(e["wer"], 3)) for e in s["evals"]]}
            elif (d / "ckpt" / "latest.pt").exists():
                out[key] = {"done": False, "in_progress": True}
    return out


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=("pilot", "main", "extend", "score", "probe", "status"))
    p.add_argument("--arm", choices=mf.ARMS)
    p.add_argument("--lr", type=float)
    p.add_argument("--smoke-shards", type=int)
    p.add_argument("--steps", type=int, help="smoke pilot steps; probe steps")
    p.add_argument("--budget-minutes", type=float, default=110.0)
    p.add_argument("--ckpt-minutes", type=float, default=20.0)
    p.add_argument("--stage", choices=("main", "extension"), default="main")
    p.add_argument("--sets", nargs="+", choices=SCORE_SETS)
    p.add_argument("--limit", type=int)
    p.add_argument("--confirm-extension", action="store_true")
    p.add_argument("--lock-wait", type=float, default=3600.0)
    a = p.parse_args(argv)
    if a.cmd != "status" and a.arm is None:
        p.error("--arm is required")
    smoke = a.smoke_shards is not None
    if a.limit is not None and not smoke:
        p.error("--limit is a smoke option (needs --smoke-shards)")
    if a.cmd == "main" and a.lr is not None and not smoke:
        p.error("the main run's rate comes from the pilots (--lr only for smoke runs)")
    if a.cmd == "pilot" and (a.lr not in PILOT_LRS):
        p.error(f"--lr must be one of {PILOT_LRS}")
    if a.steps is not None and not smoke and a.cmd != "probe":
        p.error("--steps is a smoke or probe option")
    return a


def main(argv: list[str] | None = None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    lay = md.layout(a.smoke_shards)
    relabel_paths.require_mount()
    if a.cmd == "status":
        print(json.dumps(_status(lay), indent=1))
        return 0
    device = torch.device("cuda")

    def run() -> int:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is required")
        mf.apply_lease(device)
        train.deterministic_mode()
        if a.cmd == "score":
            names = a.sets or (list(relabel_paths.SETS) + list(t5clusters.SETS))
            score(lay, a.arm, a.stage, names, a.limit, device)
            return 0
        if a.cmd == "probe":
            print(json.dumps(probe(lay, a.arm, a.steps or 200, device), indent=1, default=str))
            return 0
        return train_command(a, lay, device)

    import score_teacher
    try:
        with score_teacher.gpu_lock(f"relabel mutant_train {a.cmd} {a.arm}", a.lock_wait):
            return run()
    except score_teacher.LockBusy as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return score_teacher.LOCK_BUSY_EXIT
    except ResumeError as exc:
        print(f"refusing to resume: {exc}", file=sys.stderr)
        return EXIT_RESUME_MISMATCH


def train_command(a, lay: md.Layout, device) -> int:
    tdt_fused = load_tdt_fused()
    data = md.Data.load(lay)
    identity = feature_identity(lay, a.arm, data)
    smoke = lay.smoke
    if a.cmd == "pilot":
        stage, lr = "pilot", a.lr
    elif a.cmd == "main":
        stage = "main"
        lr = a.lr if a.lr is not None else select_lr(pilot_results(lay, a.arm))
    else:
        stage = "extension"
        if not a.confirm_extension:
            raise SystemExit("the extension is applied once to every arm, when the readout calls for it: "
                             "--confirm-extension")
        main_summary = run_dir(lay, a.arm, "main") / "summary.json"
        if not main_summary.exists():
            raise SystemExit(f"{a.arm}: the main run is not complete")
        lr = json.loads(main_summary.read_text())["lr"]
    if (run_dir(lay, a.arm, stage, lr) / "summary.json").exists():
        log(f"{a.arm} {run_dir(lay, a.arm, stage, lr).name}: already complete")
        return 0
    steps = a.steps if (smoke and stage == "pilot") else None
    sched = schedule(stage, lr, data.plan.n_batches, pilot_steps=steps,
                     pilot_warmup=max(1, steps // 10) if steps else None)
    model = evaluate.load_pretrained("cpu")
    info = init_decoder(model)
    config = run_config(a.arm, stage, lr, data, identity, info, smoke, tdt_fused.settings())
    if stage == "main" and a.lr is None:
        config["selected_from_pilots"] = {LR_NAMES[k]: v for k, v in pilot_results(lay, a.arm).items()}
    init = None
    if stage == "extension":
        init = torch.load(run_dir(lay, a.arm, "main") / "ckpt" / "latest.pt", map_location="cpu", weights_only=False)
        if not init.get("complete"):
            raise SystemExit("the main run's last checkpoint is not its final state")
    batches = FeatureBatches(lay, a.arm, data, identity)
    items = heldout_items(lay, a.arm, data, identity)
    out = run_dir(lay, a.arm, stage, lr)
    t = Trainer(model, batches, sched, out, config, device, heldout=items, ckpt_minutes=a.ckpt_minutes,
                budget_minutes=a.budget_minutes, init=init)
    status = t.run()
    log(f"{a.arm} {out.name}: {status} at step {t.step}/{sched.end_step}; data wait {batches.wait_s:.0f} s")
    if status == "complete" and stage == "pilot":
        import shutil
        shutil.rmtree(out / "ckpt", ignore_errors=True)
        (out / "final.pt").unlink(missing_ok=True)
        log(f"{a.arm} {out.name}: pilot checkpoints discarded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
