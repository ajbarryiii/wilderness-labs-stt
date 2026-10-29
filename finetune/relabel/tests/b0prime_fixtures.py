"""Fixture world for the test_b0prime* contract tests (no tests here).

On training_fixtures.LabeledWorld (real C1 shards and C2 label records in a temporary $RELABEL_ROOT, a hallucinated
repetition in batch 5): a batch plan of 4 pools of 4 consumed batches (the last pool short), its materialized pool
audio, the run labels (train_b0prime.open_labels: T* through repetition rejection, cleaning version 0), and small
stage 3 runs on that plan with the production runs' shapes: a trunk of pools 0-2 with branch points at pools 1 and 3
(the last one its end), a branch from pool 1 cooling down over pool 1, a branch from pool 3 cooling down over pool
3, and pilot-like WSD and linear runs over pools 0-3. Harness builds a run's context as train.main does (parse_args's
namespace, then build_context inside the seam) with the tiny model, and drives train.train_loop.
"""
from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path
from unittest import mock

import train  # finetune/parakeet-ternary
import train_b0prime as tb
from decoder_fixtures import tiny_model
from labeling import filters
from labeling_fixtures import relabel_root
from training import batchplan as bp
from training import poolaudio, stage3
from training import repetition as trep
from training import replay as treplay
from training import seam as tseam
from training_fixtures import LabeledWorld

P, N_POOLS, BUDGET, SEED = 4, 4, 160_000, 7
PEAK = 2e-3
RUNS = {
    "trunk": stage3.Run("trunk", "trunk", 3, 3, 11, branch_points=(1, 3)),
    "ba": stage3.Run("ba", "branch", 2, 3, 11, "wsd", (1, 2), branch_pool=1),
    "bb": stage3.Run("bb", "branch", 4, 3, 11, "wsd", (3, 4), branch_pool=3),
    "pw": stage3.Run("pw", "pilot", 4, 4, 11, "wsd", (2, 4), PEAK),
    "pl": stage3.Run("pl", "sanity", 4, 4, 11, "linear"),
}
_TOKENIZER: dict = {}


def repetition_spec(cache: Path) -> trep.Repetition:
    if "file" not in _TOKENIZER:
        _TOKENIZER["dir"] = tempfile.TemporaryDirectory()
        _TOKENIZER["file"] = filters.b0_tokenizer_file(Path(_TOKENIZER["dir"].name))
    return trep.Repetition(_TOKENIZER["file"], cache)


class World:
    """The fixture world (module docstring); enter() sets $RELABEL_ROOT, close() restores it."""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = relabel_root(self.root)
        self.env.__enter__()
        self.w = LabeledWorld(self.root, hallucinate=True)
        self.plan = bp.build(self.w.pool, self.root / "plans" / "full", N_POOLS, pool_batches=P, budget=BUDGET,
                             seed=SEED)
        self.store = self.root / "pool-audio-store"
        poolaudio.materialize(self.plan, self.w.pool, self.store)
        self.labels = self.open_labels()
        self.replay = treplay.PlannedReplay(self.plan, self.w.pool, audio="materialized", store=self.store)

    def open_labels(self, cleaning_version: int = 0, transform=None, extra=None):
        with mock.patch.object(trep, "production", lambda run: repetition_spec(self.root / "rep.json")):
            return tb.open_labels("fixture", self.w.pool, self.plan.consumed_batches, cleaning_version,
                                  transform=transform, extra=extra, run_dir_=self.w.labels_dir)

    def close(self):
        self.replay.close()
        self.env.__exit__(None, None, None)
        self.tmp.cleanup()


class FakeEvaluator:
    """train.DevEvaluator's subset() without decoding."""

    def __init__(self, *_, **__):
        self.subsets = {}

    def subset(self, model):
        return {"fixture": {"wer": 0.5, "utterances": 1, "records": []}}


def model_factory(seed: int = 7):
    def build():
        model = tiny_model(seed)
        return model, {"base": {"base_sha256": "b0"}, "trainable": tb.configure(model)}
    return build


class Harness:
    def __init__(self, world: World, device: str = "cpu"):
        self.world, self.device = world, device

    def data(self, run: stage3.Run, peak: float = PEAK, labels=None) -> tb.Data:
        return tb.data_for(run, self.world.plan, self.world.replay, labels or self.world.labels, peak)

    def args(self, run: stage3.Run, data, peak: float = PEAK, cleaning_version: int = 0, **kw) -> argparse.Namespace:
        a = argparse.Namespace(labels_run="fixture", ckpt_minutes=30.0, overwrite=False, budget_minutes=None,
                               deadline=None, cleaning_version=cleaning_version)
        args = tb.namespace(a, run, data, peak, None, False, "b0")
        args.__dict__.update(kw)
        return args

    def context(self, name: str, peak: float = PEAK, labels=None, **kw):
        """parse_args's namespace and build_context (inside the seam), the tiny model, a fake dev evaluator."""
        run = RUNS[name]
        data = self.data(run, peak, labels)
        args = self.args(run, data, peak, **kw)
        path = tb.run_dir(name)
        path.mkdir(parents=True, exist_ok=True)
        with tb.seam(), mock.patch.object(tb, "build_model", model_factory()), \
                mock.patch.object(train, "DevEvaluator", FakeEvaluator):
            ctx, info = tb.build_context(args, path, self.device)
        return ctx

    def loop(self, ctx, compute_losses=None, stop_after: int | None = None) -> list[dict]:
        """train.train_loop through the seam; per optimizer step its planned batch, learning rate and kept ids."""
        seen, label = [], tseam.label_batch

        def remember(c, batch):
            out = label(c, batch)
            if out is not None:
                seen.append({"index": int(batch["index"]), "lr": c.optimizer.param_groups[0]["lr"],
                             "lrs": {g["lr"] for g in c.optimizer.param_groups},
                             "kept": [i for i, t in zip(batch["ids"], batch["labels"]) if t is not None]})
            return out
        patches = [mock.patch.object(train, "label_batch", remember)]
        if compute_losses is not None:
            patches.append(mock.patch.object(train, "compute_losses", compute_losses))
        with tb.seam():
            for p in patches:
                p.start()
            try:
                train.train_loop(ctx, stop_after=stop_after)
            finally:
                for p in reversed(patches):
                    p.stop()
        return seen

    def restore(self, ctx):
        with tb.seam():
            payload = train.load_resume_payload(ctx.run, ctx.args)
            ctx.restore(payload)
        return ctx


def tree_hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(root).rglob("*")) if p.is_file()}


class FakeLosses:
    """train.compute_losses stand-in (CPU): scale x the sum of squares of the trainable parameters."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def __call__(self, student, audio, audio_len, tokens, token_len, recipe, teacher_enc=None, teacher_len=None):
        loss = self.scale * sum(p.float().pow(2).sum() for p in student.parameters() if p.requires_grad)
        return loss, loss, None
