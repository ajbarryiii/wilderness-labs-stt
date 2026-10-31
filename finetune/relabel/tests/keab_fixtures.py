"""Fixture world for the test_keab_trainer* contract tests (no tests here).

On b0prime_fixtures.World (real C1 shards and C2 label records in a temporary $RELABEL_ROOT, a version 1 plan of 4 pools
of 4 consumed batches, materialized pool audio): three extra stores, a synthetic union of two clip stores (served
through stage 2c's TS augmentation, whose speed perturbation changes a clip's length), a source drawn once, and an AMI
pair store whose IHM ids are fixture ids (four kept, one rejected; SDM and IHM recordings of equal length, different
samples); batch plan version 3 over them; each arm's labels (train_keab.open_labels: T* or B0 through repetition
rejection, cleaning version 0 unless a test asks for 3); tiny NeMo models for the student's base, its initialization
and the two teachers (B0' for the main arm, B0 for the control). Harness4 builds a run's context as train.main does
(train_keab.namespace, then build_context inside the stage 4 seam) and drives train.train_loop.
"""
from __future__ import annotations

import argparse
import functools
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import b0prime_fixtures as fx
import train  # finetune/parakeet-ternary
import train_keab as tk
from decoder_fixtures import tiny_model
from synthetic import augment, mixing
from test_ami_sdm import make_clips, pcm_of
from test_synthetic_mixing import make_aug_pool, make_pool
from training import ami_store
from training import batchplan as bp
from training import repetition as trep
from training import replay as treplay
from training import seam as tseam
from training import stage4
from training.clip_union import ClipUnion

P, N_POOLS, BUDGET, SEED, PEAK = fx.P, fx.N_POOLS, fx.BUDGET, fx.SEED, fx.PEAK
RUNS = {
    "pm": stage4.Run("pm", "main", "pilot", 4, 4, 11, "wsd", (2, 4), PEAK),
    "pc": stage4.Run("pc", "control", "pilot", 4, 4, 11, "wsd", (2, 4), PEAK),
    "tm": stage4.Run("tm", "main", "trunk", 3, 3, 11, branch_points=(1, 3)),
    "bm": stage4.Run("bm", "main", "branch", 2, 3, 11, "wsd", (1, 2), branch_pool=1, trunk="tm"),
}
TEACHER_SEEDS = {"main": 21, "control": 22}
TEACHER_RECORDS = {"main": {"name": "B0p", "path": "fixture-b0p", "sha256": "b0p" * 8},
                   "control": {"name": "B0", "path": "fixture-b0", "sha256": "b0" * 12}}
STUDENT_SEED, INIT_SEED = 7, 8


def clip_store(root: Path, n: int, identity: str, prefix: str, seconds: tuple[float, float] = (0.2, 0.9)):
    clips = make_pool(root, n=n).clips
    for c in clips:
        c["id"] = c["id"].replace("pilot", prefix)
    return mixing.ClipPool(clips, identity=identity)


def teacher(arm: str):
    return tiny_model(TEACHER_SEEDS[arm]).eval()


def init_state() -> dict:
    return tiny_model(INIT_SEED).state_dict()


def student_factory():
    """train_keab.build_student on the tiny models (every module of the quantized pattern binary)."""
    original = tk.build_student

    def build(teacher_decoder, scope, resuming, **kw):
        return original(teacher_decoder, scope, resuming, base_model=tiny_model(STUDENT_SEED),
                        init_state=init_state(), expected_modules=None)
    return build


def teacher_loader(arm: str, device: str = "cpu"):
    return teacher(arm).to(device), dict(TEACHER_RECORDS[arm])


class World4(fx.World):
    def __init__(self):
        super().__init__()
        root = self.root
        full = clip_store(root / "src-full", 6, "full-store", "full")
        pairs = clip_store(root / "src-pairs", 4, "pairs-store", "pairs-r1")
        self.union = ClipUnion([("full", full), ("pairs", pairs)])
        self.once = clip_store(root / "src-once", 10, "once-store", "cv")
        lab = self.labels.real.store
        kept = sorted(i for i in lab.ids() if lab.text(i) is not None)
        rejected = sorted(i for i in lab.ids() if lab.text(i) is None)
        self.pair_ihm = kept[:4] + rejected[:1]
        self.pair_specs = [(f"AMI_ES2002a_sdm_FEE005_{k:07d}_{k + 100:07d}", hid, pcm_of(k, 9000 + 1601 * k),
                            pcm_of(70 + k, 9000 + 1601 * k)) for k, hid in enumerate(self.pair_ihm)]
        self.ami = ami_store.PairStore(make_clips(root / "pair-audio", self.pair_specs), identity="pair-store")
        cfg = [("synthetic", "syn:", "fixture union", "about:2", True, 11, "as stored"),
               ("cv27-en", "cv27-en:", "fixture once", "once", False, 13, "layer 1"),
               ("ami-sdm", "ami-sdm:", "fixture pairs", "times:3", False, 17, "pair")]
        self.stores = list(zip(cfg, (self.union, self.once, self.ami)))
        self.sources = tk.plan_sources(self.stores)
        self.plan3 = bp.build(self.w.pool, root / "plans" / "v3", N_POOLS, sources=self.sources, pool_batches=P,
                              budget=BUDGET, seed=SEED, version=bp.V3)
        self.augmenter = augment.Augmenter(make_aug_pool(root / "aug"), seed=3)
        self.extras = [treplay.ExtraSource(self.sources[0], self.augmenter), treplay.ExtraSource(self.sources[1]),
                       treplay.ExtraSource(self.sources[2])]
        self.replay3 = treplay.PlannedReplay(self.plan3, self.w.pool, audio="materialized", store=self.store,
                                             sources=self.extras)
        self.arm_labels = {arm: self.open_arm(arm) for arm in stage4.ARMS}

    def open_arm(self, arm: str, cleaning_version: int = 0, reject=None):
        with mock.patch.object(trep, "production", lambda run: fx.repetition_spec(self.root / "rep.json")):
            return tk.open_labels("fixture", self.w.pool, self.plan3, stage4.LABEL_SETS[arm], cleaning_version,
                                  extra=lambda real: tk.label_providers(self.stores, self.sources, real, "fixture",
                                                                        cleaning_version),
                                  run_dir_=self.w.labels_dir, reject=reject, cleaning_versions=(0, 3))

    def close(self):
        self.replay3.close()
        super().close()


class Harness4:
    def __init__(self, world: World4, device: str = "cpu"):
        self.world, self.device = world, device

    def data(self, run: stage4.Run, labels=None) -> "tk.tb.Data":
        return tk.tb.data_for(run, self.world.plan3, self.world.replay3, labels or self.world.arm_labels[run.arm],
                              run.lr or PEAK)

    def args(self, run: stage4.Run, data, scope: str = "S1", selection=None, **kw) -> argparse.Namespace:
        a = argparse.Namespace(labels_run="fixture", ckpt_minutes=30.0, overwrite=False, budget_minutes=None,
                               deadline=None)
        with mock.patch.object(tk, "teacher_record", lambda arm: dict(TEACHER_RECORDS[arm])), \
                mock.patch.object(tk, "RUNS", RUNS):
            args = tk.namespace(a, run, data, run.lr or PEAK, selection, False, "b0", scope)
        args.__dict__.update(kw)
        return args

    def patches(self):
        return [mock.patch.object(tk, "build_student", student_factory()),
                mock.patch.object(tk.tb, "label_manifest",                     # the fixture run has no manifest.json
                                  lambda run: {"path": f"labels/fixture/{run}/manifest.json", "sha256": "fixture"}),
                mock.patch.object(tk, "load_teacher_model", teacher_loader),
                mock.patch.object(tk, "RUNS", RUNS),
                mock.patch.object(train, "DevEvaluator", fx.FakeEvaluator)]

    def context(self, name: str, scope: str = "S1", labels=None, **kw):
        run = RUNS[name]
        data = self.data(run, labels)
        args = self.args(run, data, scope, **kw)
        path = tk.run_dir(name)
        path.mkdir(parents=True, exist_ok=True)
        with tk.seam(), contextlib_all(self.patches()):
            ctx, info = tk.build_context(args, path, self.device)
        ctx.info = info
        return ctx

    def loop(self, ctx, compute_losses=None, stop_after: int | None = None) -> list[dict]:
        """train.train_loop through the stage 4 seam; per optimizer step its planned batch, learning rate, kept ids
        and the teacher's input rows (the label_batch output)."""
        seen, label = [], tk.label_batch

        def remember(c, batch):
            out = label(c, batch)
            if out is not None:
                seen.append({"index": int(batch["index"]), "lr": c.optimizer.param_groups[0]["lr"],
                             "kept": [i for i, t in zip(batch["ids"], batch["labels"]) if t is not None],
                             "teacher_len": out["teacher_len"].tolist(), "audio_len": out["audio_len"].tolist()})
            return out
        patches = [mock.patch.object(train, "label_batch", remember)] + self.patches()
        if compute_losses is not None:
            patches.append(mock.patch.object(train, "compute_losses", compute_losses))
        with tk.seam(), contextlib_all(patches):
            train.train_loop(ctx, stop_after=stop_after)
        return seen


def contextlib_all(patches):
    import contextlib
    stack = contextlib.ExitStack()
    for p in patches:
        stack.enter_context(p)
    return stack


class EncoderLosses:
    """train.compute_losses stand-in for the CPU: the student's encoder on its own features (no SpecAugment, so the
    test is deterministic) matched to the teacher's encoding by train.encoder_matching_loss, with the pinned length
    check (no TDT loss: it is CUDA-only)."""

    def __init__(self):
        self.calls = 0

    def __call__(self, student, audio, audio_len, tokens, token_len, recipe, teacher_enc=None, teacher_len=None):
        self.calls += 1
        feats, feat_len = student.preprocessor(input_signal=audio, length=audio_len)
        enc, enc_len = student.encoder(audio_signal=feats, length=feat_len)
        if not torch.equal(teacher_len, enc_len) or teacher_enc.shape[-1] != enc.shape[-1]:
            raise RuntimeError("teacher and student encoder lengths differ")
        mse = train.encoder_matching_loss(enc, teacher_enc, enc_len)
        return mse, mse.detach() * 0, mse


def pair_rows(batch: dict) -> list[int]:
    return [r for r, s in enumerate(batch["sources"]) if s == "ami-sdm"]


def f32(pcm: np.ndarray) -> np.ndarray:
    return pcm.astype(np.float32) / 32768.0


__all__ = ["World4", "Harness4", "RUNS", "EncoderLosses", "functools", "tseam"]
