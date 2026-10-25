"""train.py replacements for offline-label training (finetune/parakeet-ternary/train.py, never modified; the
seam pattern of finetune/parakeet-binary/binary_main.py). Shared by stage 2b and stage 3.

replacements(data) gives the (module, attribute, value) list that train.train_loop needs to train on the label
store instead of the online teacher, installed by `installed(...)` and restored on exit:

1. train.Batcher: a factory with Batcher's signature returning a replay.ReplayBatcher over `data` (the context's
   "stream") from the checkpointed position {"batch": k} (None: batch 0) to data.stop;
2. train.label_batch: label_batch below. It replaces the online teacher: the batch's rows carry the label store's
   texts (None = rejected); rejected rows leave the batch (padded length kept, as train.label_batch does), a batch
   with no labeled row returns None (consumed, no optimizer step). It also sets the learning rate of every
   parameter group to data.schedule.lr(k) for the batch's consumed-batch index k, which must equal
   state.batches_read - 1 (train_loop counts every batch it reads); the optimizer step that follows uses it;
3. train.make_optimizer: AdamW exactly as train.make_optimizer (train.param_groups: trainable parameters only,
   weight decay 0.01 on matrices, betas 0.9/0.98, eps 1e-8) with a ConstantScheduler: the schedule lives in
   label_batch, on the consumed-batch index, not on the optimizer-step count.

OfflineData.state_dict() is what train.py records as the stream identity (config.json "stream"): a hash over
the replay's shards, the label store's shards and label set, the schedule and the stop batch.
"""
from __future__ import annotations

import contextlib
import hashlib
import json

import torch

from training import replay as treplay
from training.schedule import WSD


class OfflineData:
    """The trainer's "stream": replay + labels + WSD schedule + stop batch (+ optional extra clip source)."""

    def __init__(self, replay: treplay.JournalReplay, labels, schedule: WSD, stop: int, extra=None):
        if stop > replay.through_batch:
            raise ValueError(f"stop {stop} beyond the replay's {replay.through_batch} batches")
        if schedule.stop_batch is not None and stop > schedule.stop_batch:
            raise ValueError(f"stop {stop} beyond the schedule's last batch {schedule.last_batch}")
        self.replay, self.labels, self.schedule, self.stop, self.extra = replay, labels, schedule, int(stop), extra
        self.plan = treplay.plan(replay, labels, self.stop)

    def state_dict(self) -> dict:
        body = {"replay_sha256": self.replay.identity["sha256"], "labels_sha256": self.labels.identity["sha256"],
                "label_set": self.labels.label_set, "schedule": self.schedule.to_dict(), "stop_batch": self.stop,
                "extra": None if self.extra is None else getattr(self.extra, "identity", repr(type(self.extra)))}
        return {"kind": "c1-journal-replay", **body,
                "config_hash": hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()}

    def load_state_dict(self, state: dict) -> None:
        if state.get("config_hash") != self.state_dict()["config_hash"]:
            raise ValueError("offline data state of another configuration")

    def batcher(self, position: dict | None) -> treplay.ReplayBatcher:
        start = 0 if position is None else int(position["batch"])
        return treplay.ReplayBatcher(self.replay, self.labels, start=start, stop=self.stop, extra=self.extra)

    def close(self) -> None:
        pass


def batcher_factory(stream, batch_seconds: float, seed: int, position: dict | None = None, *_, **__):
    if not isinstance(stream, OfflineData):
        raise TypeError("the offline seam needs OfflineData as the context's stream")
    return stream.batcher(position)


def label_batch(ctx, batch: dict) -> dict | None:
    """train.label_batch for offline labels (see the module docstring)."""
    import paths  # finetune/parakeet-ternary
    import train
    st = ctx.state
    k = int(batch["index"])
    if k != st.batches_read - 1:
        raise RuntimeError(f"batch index {k} != consumed batches {st.batches_read} - 1")
    lr = ctx.stream.schedule.lr(k)
    for group in ctx.optimizer.param_groups:
        group["lr"] = lr
    keep_list = [t is not None for t in batch["labels"]]
    dropped = keep_list.count(False)
    st.acc["dropped_rows"] += dropped
    if dropped:
        st.drop_counts["rejected"] = st.drop_counts.get("rejected", 0) + dropped
    if not any(keep_list):
        return None
    rows = torch.tensor([i for i, k in enumerate(keep_list) if k], dtype=torch.long)
    texts = [t for t in batch["labels"] if t is not None]
    seconds = 0.0
    for kept, n, source in zip(keep_list, batch["audio_len"].tolist(), batch["sources"]):
        if kept:
            seconds += n / paths.SAMPLE_RATE
            st.seconds_by_source[source] = st.seconds_by_source.get(source, 0.0) + n / paths.SAMPLE_RATE
    audio = batch["audio"].to(ctx.device, non_blocking=True)
    audio_len = batch["audio_len"].to(ctx.device, non_blocking=True)
    rows = rows.to(audio.device)
    tokens, token_len = train.tokenize(ctx.student, texts, ctx.device)
    return {"audio": audio[rows], "audio_len": audio_len[rows], "tokens": tokens, "token_len": token_len,
            "teacher_enc": None, "teacher_len": None, "seconds": seconds, "rows": len(texts)}


class ConstantScheduler:
    """A scheduler that never changes the learning rate (label_batch sets it per consumed batch)."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.steps = 0

    def step(self) -> None:
        self.steps += 1

    def state_dict(self) -> dict:
        return {"kind": "lr set per consumed batch by training.seam.label_batch", "steps": self.steps}

    def load_state_dict(self, state: dict) -> None:
        self.steps = int(state["steps"])

    def get_last_lr(self) -> list[float]:
        return [g["lr"] for g in self.optimizer.param_groups]


def make_optimizer(model, lr: float, max_steps: int):
    import train
    optimizer = torch.optim.AdamW(train.param_groups(model), lr=lr, betas=(0.9, 0.98), eps=1e-8)
    return optimizer, ConstantScheduler(optimizer)


def replacements() -> list[tuple]:
    import train
    return [(train, "Batcher", batcher_factory), (train, "label_batch", label_batch),
            (train, "make_optimizer", make_optimizer)]


@contextlib.contextmanager
def installed(items: list[tuple]):
    """Set every (object, attribute, value); restore the originals on exit."""
    saved = [(obj, name, getattr(obj, name)) for obj, name, _ in items]
    try:
        for obj, name, value in items:
            setattr(obj, name, value)
        yield
    finally:
        for obj, name, value in reversed(saved):
            setattr(obj, name, value)
