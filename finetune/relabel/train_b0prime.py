"""Stage 3 (b0prime plan v4): B0', the FP32 Parakeet TDT 0.6B v2 (B0) fine-tuned on the new labels.

    ./python train_b0prime.py plan                                    build the multi-source plan of pools 0-781 (CPU)
    ./python train_b0prime.py train --run NAME --labels-run RUN [--cleaning-version 2] [--budget-minutes 110]
                                    [--deadline EPOCH]                one chunk of one run (GPU; resumable)
    ./python train_b0prime.py score-b0 [--deadline EPOCH]             re-score B0 on the gate's sets (GPU)
    ./python train_b0prime.py score-run --run NAME                    (re)score a finished run's model.nemo (GPU)
    ./python train_b0prime.py probe --labels-run RUN [--steps 200]    the memory probe (GPU)
    ./python train_b0prime.py chunk --labels-run RUN --through pilot|main [--cleaning-version 2]
                                    [--budget-minutes 110]            one chunk job: the next unfinished items of
                                                                      ORDER (each a subprocess) within the budget
    ./python train_b0prime.py status | report

v4 (2026-10-29): labels through cleaning version 2 (layer 0 punctuation transfer, compact acronyms); the
multi-source version 2 plan (stage 2c's synthetic clips at 2.81% with the TS augmentation, and the cv27-en clip store
drawn exactly once), extra rows labeled by their stores' providers; the read-time reject list (the ted-test-text
containment rule's main-pool artifact, $RELABEL_ROOT/<REJECT_LIST>) in the label identity; eval-stem, dev-technical
term recall and the cv27-en dev sample scored; style-neutral and formatted metrics in the report.

Runs, spans, schedules and rules: training/stage3.py (RUNS: pilot-lr3e-5, pilot-lr1e-4, pilot-lr3e-4, sanity-wsd,
sanity-linear, trunk, b25, b50, b100). Outputs under $RELABEL_ROOT/b0prime/: runs/<name>/ (train.py's run layout),
scores/B0/, selection.json, wsd-rule.json, probe/, results.json; RESULTS-b0prime.md next to the plan
($RELABEL_ROOT/plans/).

Training is finetune/parakeet-ternary/train.py's loop (train.main: checkpoints every 30 minutes as latest/previous,
resume, signals, metrics, summary.json), unchanged, inside seam(): training/seam.py's offline-label replacements (the
learning rate set per planned batch, AdamW as train.make_optimizer with a constant scheduler, rejected rows dropped
after the cut, a fully rejected planned batch consumed without a step, the non-finite step guard) and this module's:

1. train.parse_args: the run's arguments: arm A1, recipe "b0prime-a1", the run spec, the peak (a pilot run's own; the
   frozen selection for the sanity pair and the main runs, which also need the frozen WSD rule to allow the main run),
   seed, max_steps = planned batches before the run's stop with a labeled row (so train_loop ends at the last one),
   the labels (labels_run, cleaning_version, labels_sha256 and its parts, label_transform, label_reject,
   extra_sources), the plan and replay hashes, the schedule, B0's SHA-256, gradient checkpointing (the memory probe's
   decision), and a branch's branch_of (its trunk branch point's record). RESUME_KEYS gains them all; a checkpoint of
   another configuration is refused (exit 78). RESUME_ALLOWED_CHANGES gains budget_minutes and deadline. A selected run
   (sanity pair, main) must have the labels, plan and B0 (IDENTITY_KEYS) the pilot runs shared; a finished run of
   another labels run or cleaning version is refused rather than skipped (run_train, chunk: exit 78).
2. train.build_context: B0 (train.load_pretrained, lock-file SHA-256 checked by train.verify_base), configured as arm
   A1 with every parameter trainable: train.configure_trainable(RECIPE) (encoder, prediction and joint networks
   trainable, prediction and joint dropout 0, SpecAugment and the training-mode preprocessor on), checked by
   train.verify_recipe and verify() (all 617.8M parameters trainable, FP32, no quantization module); FP32 parameters
   and AdamW state, BF16 autocast (train.compute_losses); no teacher, no B0 forward, no encoder matching, no
   waveform augmentation. A branch with no checkpoint of its own starts from its trunk branch point (start_branch:
   weights, AdamW, scheduler, RNG and the plan position; the file read-only). A budget timer requests a checkpoint and
   stop after --budget-minutes or at --deadline (exit 0).
3. The data (Data, a seam.OfflineData): the multi-source version 2 batch plan of pools 0-781 (training/batchplan.py
   compute_sources over EXTRA_SOURCES; materialized pool audio and the sources' clip stores) served by a
   label_sources.Batcher, labels from label_sources.Labels (the label store at the run's cleaning version with
   repetition rejection and the read-time reject list; each extra source's rows from its store's ClipLabels; a
   read-time transform is an open seam, unused), the run's stop B(stop pool). The trunk's batcher saves a branch point
   (save_branch_point) just before it serves the first planned
   batch of pools 166 and 332, so the payload is the state after B(p) planned batches; the one at pool 664 is the
   trunk's final checkpoint (finish). Branch points are written once and kept; the rotating checkpoints are train.py's.
4. train.run_eval: the trunk's dev-subset evaluations (train.run_eval, informational only); none for the other runs.
5. train.finish: a scored run (every run but the trunk) writes model.nemo and model.json (its SHA-256, the label
   store's identity and its run manifest's SHA-256, the training-source hashes) and scores the model restored from
   model.nemo with the scorer below; the trunk writes its final branch point and is not scored.
6. paths.RUNS: $RELABEL_ROOT/b0prime/runs; paths.gpu_lock: train_decoder.bounded_lock(deadline); TRAINING_SOURCES and
   OUTPUTS gain this stage's.

Scorer (stage 1's, as stage 2b's train_decoder.score_model without its decoder-step proxy): evaluate.evaluate_set at
batch 64 and 1200 s per batch with the decoder pinned to the mode M1 and MB1 decoded in, on the five stage 1 dev sets
(eval-dev/<set>.json, per-utterance records kept), the dev-medical and dev-technical parts with critical-term recall
(TERM_KIND: the medical and technical lists; domain/), eval-stem's parts with technical term recall (eval-stem/) and,
once its manifest exists, the cv27-en dev sample (cv-dev/). score-b0 scores B0 the same way into scores/B0/ and checks every
dev set against its stage 1 record within RESCORE_POINTS (crosscheck.json; a failure fails the job).
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
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import evaluate  # finetune/parakeet-ternary
import paths  # finetune/parakeet-ternary
import relabel_paths
import score_domain
import train_decoder as td
from labeling import layout as llayout
from training import bootstrap, devsplit, label_sources, stage3
from training import labels as tlabels
from training import repetition as trep
from training import replay as treplay
from training import seam as tseam

RECIPE = {"encoder_matching": False, "train_pred_joint": True, "quantized": False}
RECIPE_NAME = "b0prime-a1"
ARM = "A1"
LABEL_SET = "T*"
BATCH_SECONDS = 600.0
CLEANING_VERSION = 2         # b0prime plan v4: stage 3 reads cleaning version 2 (cleaning plan v6)
CHUNK_MINUTES = 110.0
MIN_CHUNK_MINUTES = 15.0
EXIT_BUDGET = 0
EXIT_NOT_READY = 75
EXIT_STOPPED = 80            # the preregistered WSD rule stopped stage 3 (the architect reports to the user)
EXIT_RESCORE = 81            # B0's re-score does not reproduce stage 1
PROBE_STEPS = 200
FREE_GB = 6.0                # b0prime plan: gradient checkpointing if the run leaves less than this for others
LEASE_MARGIN_GB = 2.0        # the chunk jobs' lease: the probe's peak plus this
DEV_SETS = list(paths.DEV_SETS)
H4 = list(devsplit.H4)
ORDER = [("score", "B0")] + [("train", n) for n in stage3.PILOT + stage3.SANITY + stage3.MAIN]
THROUGH = {"pilot": len(["score"]) + len(stage3.PILOT) + len(stage3.SANITY), "main": len(ORDER)}
SOURCES = ("relabel/train_b0prime.py", "relabel/b0prime_report.py", "relabel/train_decoder.py",
           "relabel/training/stage3.py", "relabel/training/label_sources.py", "relabel/training/labels.py",
           "relabel/training/replay.py", "relabel/training/schedule.py", "relabel/training/batchplan.py",
           "relabel/training/poolaudio.py", "relabel/training/seam.py", "relabel/training/devsplit.py",
           "relabel/training/bootstrap.py", "relabel/training/cleaning.py", "relabel/training/repetition.py",
           "relabel/terms/cleaning/forms.tsv", "relabel/terms/cleaning/variants.tsv",
           "relabel/terms/cleaning/dialect.tsv", "relabel/terms/cleaning/choices.tsv",
           "relabel/labeling/shards.py", "relabel/labeling/c2.py", "relabel/labeling/config.py",
           "relabel/score_domain.py", "relabel/relabel_paths.py", "relabel/synthetic/mixing.py",
           "relabel/synthetic/augment.py", "relabel/synthetic/layout.py", "relabel/style_metrics.py")
RESUME_EXTRA = ("spec", "label_set", "labels_run", "labels_sha256", "cleaning_version", "cleaning", "repetition",
                "label_transform", "label_reject", "extra_sources", "plan_sha256", "replay_sha256", "stop_batch",
                "schedule", "base_sha256", "lr_selection", "branch_of")
EXTRA_OUTPUTS = ("branch-points", "model.nemo", "model.json", "domain", "scores.json")
_ACTIVE: dict = {"data": None, "ctx": None, "run": None}


class NotReady(RuntimeError):
    """An input a run needs does not exist yet (the selection, the WSD rule, a branch point, the probe)."""


class Stopped(RuntimeError):
    """The preregistered WSD rule stopped stage 3."""


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [b0prime] {msg}", flush=True)


# ----------------------------------------------------------------------------- layout

def root() -> Path:
    return llayout.relabel_root() / "b0prime"


def run_dir(name: str) -> Path:
    return root() / "runs" / name


def b0_dir() -> Path:
    return root() / "scores" / "B0"


def selection_path() -> Path:
    return root() / "selection.json"


def wsd_rule_path() -> Path:
    return root() / "wsd-rule.json"


def probe_path() -> Path:
    return root() / "probe" / "probe.json"


def results_md() -> Path:
    return llayout.relabel_root() / "plans" / "RESULTS-b0prime.md"


write_json = td.write_json


def _sha256_file(path: Path) -> str:
    return evaluate.sha256_file(Path(path))


# ----------------------------------------------------------------------------- labels and data

def open_labels(labels_run: str, pool_dir: Path, through_batch: int, cleaning_version: int = CLEANING_VERSION,
                repetition: bool = True, transform=None, extra: dict | None = None,
                run_dir_: Path | None = None, reject: label_sources.RejectList | None = None) -> label_sources.Labels:
    """The run's labels (label_sources): T*'s label store through cleaning `cleaning_version` and repetition
    rejection, the read-time reject list, an optional read-time transform, the extra sources' providers."""
    store = tlabels.LabelStore.open(run_dir_ or tlabels.production_run_dir(labels_run), pool_dir,
                                    label_set=LABEL_SET, through_batch=through_batch,
                                    cleaning=label_sources.cleaning_spec(labels_run, cleaning_version),
                                    repetition=trep.production(labels_run) if repetition else None)
    return label_sources.Labels(label_sources.StoreLabels(store, transform, reject), extra)


# ----------------------------------------------------------------------------- extra sources (b0prime plan v4)

# (name, row-id prefix, store under $RELABEL_ROOT ("synthetic run" for stage 2c's), share, TS augmentation, seed).
# The second source is the registry alias's clip store, in the stage 2c store format with a label per clip, written
# by its preparation; its share is ONCE: its kept seconds over the whole journal's real seconds, each clip once.
EXTRA_SOURCES = (("synthetic", "syn:", "synthetic run full", 0.0281, True, 20261025),     # stage 2c's seed
                 ("cv27-en", "cv27-en:", "cv27-en/store", "once", False, 20261029))
REJECT_LIST = "ted-test-text/main-pool-reject.json"     # the ted-test-text containment rule's read-time reject list


def source_stores() -> list:
    """(config, clip store) of EXTRA_SOURCES; NotReady while a store is not written yet."""
    from synthetic import layout as slayout
    from synthetic import mixing
    out = []
    for name, prefix, where, share, augment, seed in EXTRA_SOURCES:
        try:
            if where.startswith("synthetic run "):
                pool = mixing.ClipPool.load(where.split()[-1])
            else:
                pool = mixing.ClipPool.load(llayout.relabel_root() / where, synthetic_hygiene=False)
        except FileNotFoundError as error:
            raise NotReady(f"extra source {name}: no clip store yet ({error})") from error
        if name == "synthetic" and seed != slayout.SEED:
            raise RuntimeError("the synthetic source's seed is stage 2c's")
        out.append(((name, prefix, where, share, augment, seed), pool))
    return out


def plan_sources(stores) -> list:
    from training import batchplan
    return [batchplan.Source(name, prefix, pool, seed, batchplan.ONCE if share == "once" else share, augment)
            for (name, prefix, _, share, augment, seed), pool in stores]


def replay_sources(stores, sources) -> list:
    """treplay.ExtraSource per source: an augmented source's rows go through stage 2c's TS augmentation."""
    from synthetic import mixing
    out = []
    for ((name, _, _, _, augment, seed), _), src in zip(stores, sources):
        out.append(treplay.ExtraSource(src, mixing.transform_for("TS", seed=seed) if augment else None))
    return out


def reject_list() -> label_sources.RejectList:
    path = llayout.relabel_root() / REJECT_LIST
    if not path.exists():
        raise NotReady(f"no read-time reject list at $RELABEL_ROOT/{REJECT_LIST} yet")
    return label_sources.RejectList.load(path)


class HookedBatcher:
    """A label_sources.Batcher that calls data.on_batch(index) on the main thread just before it returns one of the
    hooked planned batches (the trunk's branch points), when the trainer's state is that after `index` batches."""

    def __init__(self, inner, data):
        self.inner, self.data = inner, data

    def next(self, timeout: float | None = None) -> dict:
        batch = self.inner.next(timeout)
        if int(batch["index"]) in self.data.hooks and self.data.on_batch is not None:
            self.data.on_batch(int(batch["index"]))
        return batch

    def close(self, close_stream: bool = True) -> None:
        self.inner.close(close_stream)


class Data(tseam.OfflineData):
    """seam.OfflineData over a planned replay, served by label_sources.Batcher, with hooked planned batches
    {planned batch: pool} (HookedBatcher)."""

    def __init__(self, replay, labels, schedule, stop: int, hooks: dict[int, int] | None = None):
        super().__init__(replay, labels, schedule, stop)
        self.hooks = dict(hooks or {})
        self.on_batch = None

    def state_dict(self) -> dict:
        body = {"replay_sha256": self.replay.identity["sha256"], "labels_sha256": self.labels.identity["sha256"],
                "label_set": self.labels.label_set, "schedule": stage3.schedule_dict(self.schedule),
                "stop_batch": self.stop, "hooks": sorted(self.hooks)}
        return {"kind": "c1-batch-plan", **body,
                "config_hash": hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()}

    def batcher(self, position: dict | None):
        start = 0 if position is None else int(position["batch"])
        return HookedBatcher(label_sources.Batcher(self.replay, self.labels, start=start, stop=self.stop), self)


def data_for(run: stage3.Run, plan, replay, labels, peak: float) -> Data:
    """The run's Data on a plan and replay (its stop B(stop pool), its schedule, the trunk's branch-point hooks)."""
    stop = stage3.stop_batch(run, plan)
    points = stage3.branch_points(run, plan)
    if any(b > stop for b in points.values()):
        raise ValueError(f"{run.name}: a branch point lies beyond the run's stop")
    hooks = {b: p for p, b in points.items() if b < stop}
    return Data(replay, labels, stage3.schedule_of(run, plan, peak), stop, hooks)


def pinned_base():
    """(pin, C1 directory, config id) of the pinned stream."""
    from labeling import config as lconfig
    from labeling.__main__ import c1_dir, pinned
    pin = pinned()
    if float(pin["batch"]["batch_seconds"]) != BATCH_SECONDS:
        raise RuntimeError(f"the pinned stream batches {pin['batch']['batch_seconds']} s, not {BATCH_SECONDS}")
    return pin, c1_dir(), lconfig.config_id(pin)


def check_plan(plan, pin, version: int) -> None:
    from training import batchplan
    if plan.meta["pinned_hash"] != pin["pinned_hash"] or plan.budget != BATCH_SECONDS * batchplan.SAMPLE_RATE:
        raise RuntimeError("the batch plan is of another pinned configuration or budget")
    if plan.version != version or plan.n_pools != stage3.PLAN_POOLS:
        raise RuntimeError(f"stage 3 trains on a version {version} plan of {stage3.PLAN_POOLS} pools")


def v1_plan():
    """(pin, C1 directory, config id, the version 1 plan of pools 0-781: the real occurrences and pools)."""
    from training import batchplan
    pin, pool, cid = pinned_base()
    plan = batchplan.Plan.load(batchplan.production_dir(cid, stage3.PLAN_POOLS))
    check_plan(plan, pin, batchplan.VERSION)
    return pin, pool, cid, plan


def pinned_plan(stores=None):
    """(pin, C1 directory, config id, the production multi-source version 2 plan of pools 0-781 (b0prime plan v4),
    its batchplan.Source list); NotReady before a store or the plan exists."""
    from training import batchplan
    pin, pool, cid = pinned_base()
    stores = source_stores() if stores is None else stores
    sources = plan_sources(stores)
    d = batchplan.production_dir(cid, stage3.PLAN_POOLS, sources=sources)
    if not (d / "plan.json").exists():
        raise NotReady(f"no stage 3 plan at {d} yet (train_b0prime.py plan)")
    plan = batchplan.Plan.load(d)
    check_plan(plan, pin, batchplan.SYNTHETIC_VERSION)
    if plan.source_configs() != [s.config() for s in sources]:
        raise RuntimeError("the stage 3 plan was drawn from other sources")
    return pin, pool, cid, plan, sources


def build_plan() -> int:
    """Build the production multi-source version 2 plan of pools 0-781 (CPU; b0prime plan v4)."""
    from training import batchplan
    llayout.require_mount()
    pin, pool, cid = pinned_base()
    v1 = v1_plan()[3]
    sources = plan_sources(source_stores())
    plan = batchplan.build(pool, batchplan.production_dir(cid, stage3.PLAN_POOLS, sources=sources),
                           stage3.PLAN_POOLS, sources=sources, config_id=cid)
    if plan.meta["base_plan_sha256"] != v1.sha256:
        raise RuntimeError("the stage 3 plan is not drawn on the version 1 plan of the same journal")
    log(json.dumps({**plan.identity(), "n_extra": plan.meta["n_extra"], "stats": plan.meta["stats"],
                    "sources": [{k: m[k] for k in ("config", "share_value", "draws", "distinct", "seconds")}
                                for m in plan.meta["sources"]]}, default=str))
    return 0


def offline_data(run: stage3.Run, labels_run: str, peak: float, cleaning_version: int = CLEANING_VERSION) -> Data:
    """The production Data of a run: the multi-source plan of pools 0-781 on materialized audio and the extra
    sources' stores, the labels through the plan's consumed batches (every run reads the same label identity) with
    the extra sources' providers and the read-time reject list."""
    from training import poolaudio
    stores = source_stores()
    _, pool, cid, plan, sources = pinned_plan(stores)
    extra = {src.prefix: label_sources.ClipLabels(src.clips, src.prefix) for src in sources}
    labels = open_labels(labels_run, pool, plan.consumed_batches, cleaning_version, extra=extra, reject=reject_list())
    replay = treplay.PlannedReplay(plan, pool, audio="materialized", store=poolaudio.store_dir(cid, plan.pool_batches),
                                   sources=replay_sources(stores, sources))
    return data_for(run, plan, replay, labels, peak)


# ----------------------------------------------------------------------------- the model

def verify(model) -> dict:
    """Arm A1 with every parameter trainable (module docstring, item 2); RuntimeError otherwise."""
    count = lambda m, t: sum(p.numel() for p in m.parameters() if p.requires_grad == t)  # noqa: E731
    found = {"trainable": count(model, True), "frozen": count(model, False),
             "encoder_trainable": count(model.encoder, True), "decoder_trainable": count(model.decoder, True),
             "joint_trainable": count(model.joint, True),
             "fp32": all(p.dtype == torch.float32 for p in model.parameters()),
             "quantized_modules": sum("quant" in type(m).__module__ for m in model.modules()),
             "spec_augmentation": model.spec_augmentation is not None,
             "modules_in_eval_mode": sum(not m.training for m in model.modules())}
    found["recipe_check"] = train.verify_recipe(model, RECIPE)
    problems = []
    if found["frozen"]:
        problems.append(f"{found['frozen']} parameters are frozen")
    if not found["fp32"]:
        problems.append("parameters are not FP32")
    if found["quantized_modules"]:
        problems.append("quantization modules present")
    if not found["spec_augmentation"]:
        problems.append("no SpecAugment")
    if found["modules_in_eval_mode"]:
        problems.append("modules in eval mode")
    if problems:
        raise RuntimeError(f"A1 model check failed: {'; '.join(problems)}")
    return found


def configure(model) -> dict:
    """train.configure_trainable for the A1 recipe (prediction and joint dropout 0), then verify."""
    params = train.configure_trainable(model, RECIPE)
    return params | {"verified": verify(model)}


def build_model() -> tuple[torch.nn.Module, dict]:
    """B0 on the CPU, configured as A1 (module docstring, item 2)."""
    base = train.verify_base()
    model = train.load_pretrained("cpu")
    return model, {"base": base, "trainable": configure(model)}


def recipe_of(args) -> dict:
    if getattr(args, "recipe", None) == RECIPE_NAME:
        return dict(RECIPE)
    return _TRAIN["recipe_of"](args)


_TRAIN = {name: getattr(train, name) for name in ("recipe_of", "run_eval")}


# ----------------------------------------------------------------------------- branch points

def branch_point_files(trunk_dir: Path, pool: int) -> tuple[Path, Path]:
    d = Path(trunk_dir) / "branch-points"
    return d / f"pool-{pool:05d}.pt", d / f"pool-{pool:05d}.json"


TRUNK_KEYS = ("labels_sha256", "plan_sha256", "replay_sha256", "lr", "seed", "cleaning_version", "cleaning",
              "repetition", "label_transform", "label_reject", "extra_sources", "base_sha256", "grad_checkpointing",
              "lr_selection", "batch_seconds")


def read_branch_point(trunk_dir: Path, pool: int) -> dict:
    """The committed marker of a trunk branch point (NotReady before it exists)."""
    pt, marker = branch_point_files(trunk_dir, pool)
    if not marker.exists() or not pt.exists():
        raise NotReady(f"no branch point at pool {pool} in {trunk_dir} yet")
    m = json.loads(marker.read_text())
    if m.get("pool") != pool or m.get("bytes") != pt.stat().st_size:
        raise RuntimeError(f"{marker}: does not describe {pt}")
    return m


def _commit_marker(ctx, pool: int, planned_batch: int, pt: Path, marker: Path, how: str) -> dict:
    st, args = ctx.state, ctx.args
    m = {"pool": pool, "planned_batch": planned_batch, "position": st.position, "step": st.step,
         "batches_read": st.batches_read, "how": how, "bytes": pt.stat().st_size, "sha256": _sha256_file(pt),
         "trunk": {"run_name": args.run_name, **{k: getattr(args, k) for k in TRUNK_KEYS},
                   "schedule": args.schedule, "stop_batch": args.stop_batch}, "utc": train.utc()}
    train.drop_page_cache(pt, sync=False)
    write_json(marker, m)
    return m


def save_branch_point(ctx, pool: int, planned_batch: int) -> dict:
    """Write the trunk's state (train.Context.checkpoint_payload: weights, AdamW, scheduler, RNG, state with the plan
    position) as the branch point of `pool`, once: an existing committed branch point is kept."""
    pt, marker = branch_point_files(ctx.run, pool)
    if marker.exists() and pt.exists():
        log(f"branch point at pool {pool} exists ({marker}); kept")
        return json.loads(marker.read_text())
    if ctx.state.batches_read != planned_batch or (ctx.state.position or {}).get("batch", 0) != planned_batch:
        raise RuntimeError(f"branch point {pool}: the state is at planned batch {ctx.state.batches_read}, not "
                           f"{planned_batch}")
    pt.parent.mkdir(parents=True, exist_ok=True)
    for stale in pt.parent.glob(f"{pt.name}.tmp*"):
        stale.unlink()                                # a torn write of an earlier session
    t = time.perf_counter()
    tmp = pt.with_name(f"{pt.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as handle:
        torch.save({**ctx.checkpoint_payload(), "format": train.CKPT_FORMAT, "complete": True,
                    "branch_point": {"pool": pool, "planned_batch": planned_batch}}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, pt)
    train.fsync_dir(pt.parent)
    m = _commit_marker(ctx, pool, planned_batch, pt, marker, "saved before the pool's first planned batch")
    log(f"branch point at pool {pool} (planned batch {planned_batch}, step {ctx.state.step}) -> {pt} "
        f"({time.perf_counter() - t:.0f} s)")
    return m


def final_branch_point(ctx, pool: int, planned_batch: int) -> dict:
    """The trunk's last branch point: its final checkpoint (ckpt/latest.pt, written by train_loop at the last step),
    hard-linked, once. Fully rejected planned batches between its position and B(pool) are allowed (a branch consumes
    them without a step)."""
    pt, marker = branch_point_files(ctx.run, pool)
    if marker.exists() and pt.exists():
        return json.loads(marker.read_text())
    latest = train.ckpt_paths(ctx.run)[0]
    payload = torch.load(latest, map_location="cpu", mmap=True, weights_only=False)
    step, position = payload.get("step"), payload["state"]["position"]
    del payload
    if step != ctx.args.max_steps or step != ctx.state.step or position != ctx.state.position:
        raise RuntimeError(f"{latest} is not the trunk's final state (step {step}, position {position})")
    gap = range(int(position["batch"]), planned_batch)
    empty = set(ctx.stream.plan["empty_batches"])
    if int(position["batch"]) > planned_batch or any(b not in empty for b in gap):
        raise RuntimeError(f"the trunk ends at planned batch {position['batch']}, not at B({pool}) = {planned_batch}")
    pt.parent.mkdir(parents=True, exist_ok=True)
    if pt.exists():
        pt.unlink()                                   # an uncommitted link of an earlier session
    os.link(latest, pt)
    return _commit_marker(ctx, pool, planned_batch, pt, marker, "the trunk's final checkpoint (hard link)")


def start_branch(ctx, branch_of: dict, trunk_dir: Path) -> None:
    """A branch's first session: its trunk branch point into ctx (weights, AdamW, scheduler, state with the plan
    position, RNG), the file read-only; the branch's own metrics and evaluation state start fresh."""
    pt, _ = branch_point_files(trunk_dir, branch_of["pool"])
    digest = _sha256_file(pt)
    if digest != branch_of["sha256"]:
        raise RuntimeError(f"{pt}: SHA-256 {digest} is not its marker's {branch_of['sha256']}")
    payload = torch.load(pt, map_location="cpu", mmap=True, weights_only=False)
    if not (payload.get("complete") and payload.get("format") == train.CKPT_FORMAT):
        raise RuntimeError(f"{pt} is not a complete checkpoint")
    state = dict(payload["state"])
    if state["position"] != branch_of["position"] or payload["step"] != branch_of["step"]:
        raise RuntimeError(f"{pt}: its state differs from its marker")
    ctx.student.load_state_dict(payload["student"])
    ctx.optimizer.load_state_dict(payload["optimizer"])
    ctx.scheduler.load_state_dict(payload["scheduler"])
    state.update(session=0, metrics_offset=0, evals=0, best_wer=math.inf, best_step=-1, best_file=None,
                 best_subset={}, final_subset={})
    ctx.state = train.TrainState(**state)
    train.set_rng_state(ctx.student, payload["rng"])
    del payload
    train.drop_page_cache(pt, sync=False)
    log(f"branch {ctx.args.run_name}: started from the trunk's branch point at pool {branch_of['pool']} (step "
        f"{ctx.state.step}, planned batch {ctx.state.position['batch']})")


# ----------------------------------------------------------------------------- the trainer seam

class Context(train.Context):
    """train.Context without a teacher (label_fn refuses), with the run spec and a budget timer slot."""

    def __init__(self, args, run: Path, student, data, evaluator, device: str, spec: stage3.Run):
        super().__init__(args, run, student, None, data, evaluator, None, device, label_fn=td._no_teacher)
        self.spec = spec
        self.timer: threading.Timer | None = None
        self.state.drop_counts[tseam.NONFINITE_KEY] = 0


def frozen_record(path: Path, record: dict, what: str) -> dict:
    """Write `record` at its first use; later the same record must result (RuntimeError otherwise)."""
    if path.exists():
        old = json.loads(path.read_text())
        if {k: old.get(k) for k in record} != json.loads(json.dumps(record, default=str)):
            raise RuntimeError(f"{path} records another {what} than the runs give now")
        return old
    llayout.require_mount()
    write_json(path, record | {"recorded_utc": train.utc()})
    return json.loads(path.read_text())


def run_done(name: str) -> bool:
    return (run_dir(name) / "summary.json").exists()


# what every stage 3 run must share: the labels (run, cleaning, reject list, transform, extra sources), the plan, B0
IDENTITY_KEYS = ("labels_run", "cleaning_version", "labels_sha256", "plan_sha256", "label_reject", "label_transform",
                 "extra_sources", "base_sha256")


def recorded_args(name: str) -> dict | None:
    """The arguments a finished run recorded (config.json "args"), or None."""
    path = run_dir(name) / "config.json"
    return json.loads(path.read_text()).get("args") if run_done(name) and path.exists() else None


def identity_of(args: dict) -> dict:
    return json.loads(json.dumps({k: args.get(k) for k in IDENTITY_KEYS}, default=str))


def shared_identity(names) -> dict:
    """The label and plan identity the finished runs `names` share (RuntimeError if they differ)."""
    idents = {n: identity_of(recorded_args(n) or {}) for n in names}
    distinct = {json.dumps(v, sort_keys=True) for v in idents.values()}
    if len(distinct) != 1:
        diff = sorted(k for k in IDENTITY_KEYS if len({json.dumps(v.get(k), sort_keys=True) for v in idents.values()}) > 1)
        raise RuntimeError(f"runs {list(names)} trained on different {diff}; they cannot be compared")
    return next(iter(idents.values()))


def run_mismatch(name: str, labels_run: str, cleaning_version: int) -> str | None:
    """Why a finished run does not belong with this invocation's runs (another label run or cleaning version);
    None when it matches or is not finished."""
    rec = recorded_args(name)
    if rec is None:
        return None
    got = (rec.get("labels_run"), rec.get("cleaning_version"))
    if got != (labels_run, cleaning_version):
        return f"{run_dir(name)} finished with labels run and cleaning version {got}, not {(labels_run, cleaning_version)}"
    return None


def dev_wers(out: Path) -> dict[str, float]:
    """Full-dev corpus WER (points) per stage 1 dev set of a scored model directory (eval-dev/<set>.json)."""
    return {n: 100 * json.loads((Path(out) / "eval-dev" / f"{n}.json").read_text())["wer"]["wer"] for n in DEV_SETS}


def h4(wers: dict[str, float]) -> float:
    return sum(wers[n] for n in H4) / len(H4)


def frozen_selection() -> dict:
    """stage3.select_lr over the three pilot runs, frozen at first use in selection.json (NotReady before)."""
    pending = [n for n in stage3.PILOT if not run_done(n)]
    if pending:
        raise NotReady(f"the learning-rate selection needs the pilot runs ({', '.join(pending)} pending)")
    sel = stage3.select_lr({stage3.RUNS[n].lr: h4(dev_wers(run_dir(n))) for n in stage3.PILOT})
    sel["summaries_sha256"] = {n: _sha256_file(run_dir(n) / "summary.json") for n in stage3.PILOT}
    sel["identity"] = shared_identity(stage3.PILOT)
    return frozen_record(selection_path(), sel, "selection")


def model_records(out: Path, sets=H4) -> dict[str, list]:
    return {n: json.loads((Path(out) / "eval-dev" / f"{n}.json").read_text())["records"] for n in sets}


def full_tables(models: dict[str, dict[str, list]]) -> dict:
    """bootstrap.SetTable per H4 set on the full stage 1 dev population (clusters as training/devsplit.py)."""
    return {n: bootstrap.SetTable.from_records({m: recs[n] for m, recs in models.items()},
                                               cluster_of=lambda i, n=n: devsplit.cluster_of(n, i)) for n in H4}


def frozen_wsd_rule() -> dict:
    """stage3.wsd_rule on the sanity pair (d = H4(WSD) - H4(linear), 90% paired cluster bootstrap on full dev),
    frozen at first use in wsd-rule.json (NotReady before)."""
    pending = [n for n in stage3.SANITY if not run_done(n)]
    if pending:
        raise NotReady(f"the WSD rule needs the sanity pair ({', '.join(pending)} pending)")
    models = {"wsd": model_records(run_dir("sanity-wsd")), "linear": model_records(run_dir("sanity-linear"))}
    res = bootstrap.paired(full_tables(models), {"d": ("wsd", "linear")}, mean_sets=H4)
    rule = stage3.wsd_rule(res["mean"]["d"]["point"], res["mean"]["d"]["ci"])
    rule |= {"per_set": {n: res["sets"][n]["d"] for n in H4}, "wer": res["wer"], "clusters": res["clusters"],
             "replicates": res["replicates"], "seed": res["seed"], "selection": frozen_selection()["selected"],
             "summaries_sha256": {n: _sha256_file(run_dir(n) / "summary.json") for n in stage3.SANITY},
             "identity": shared_identity(stage3.PILOT + stage3.SANITY)}
    return frozen_record(wsd_rule_path(), rule, "WSD rule")


def peak_of(run: stage3.Run) -> tuple[float, dict | None]:
    """(peak, lr_selection record): a pilot learning-rate run's own; else the frozen selection, and for the main runs
    the frozen WSD rule must allow them (Stopped otherwise)."""
    if run.lr is not None:
        return run.lr, None
    sel = frozen_selection()
    record = {k: sel[k] for k in ("selected", "selected_name", "h4", "rule", "summaries_sha256", "identity")}
    if run.kind in ("trunk", "branch"):
        rule = frozen_wsd_rule()
        if rule["stop"]:
            raise Stopped(f"the WSD sanity rule stops stage 3: {rule['reasons']} (d {rule['d']:+.3f}, interval "
                          f"{rule['ci']}); the architect reports to the user")
        record["wsd_rule"] = {k: rule[k] for k in ("d", "ci", "use_wsd")}
    return float(sel["selected"]), record


def grad_checkpointing() -> bool:
    """The memory probe's decision (b0prime plan "Resources"); NotReady before the probe."""
    if not probe_path().exists():
        raise NotReady(f"no memory probe result at {probe_path()} (train_b0prime.py probe)")
    decision = json.loads(probe_path().read_text()).get("decision") or {}
    if decision.get("fits") is not True:
        raise RuntimeError(f"the memory probe found no configuration that fits: {decision}")
    return bool(decision["grad_checkpointing"])


def branch_of(run: stage3.Run, plan, args_like: dict) -> dict | None:
    """A branch's record of its trunk branch point (its marker), after checking that the trunk trained with this
    run's labels, plan, peak and settings."""
    if run.kind != "branch":
        return None
    m = read_branch_point(run_dir("trunk"), run.branch_pool)
    want = stage3.RUNS["trunk"]
    if m["planned_batch"] != plan.pool_start(run.branch_pool) or m["trunk"]["run_name"] != want.name:
        raise RuntimeError(f"{run.name}: the branch point is not the trunk's at pool {run.branch_pool}")
    diff = [k for k in TRUNK_KEYS if json.dumps(m["trunk"].get(k), sort_keys=True, default=str)
            != json.dumps(args_like.get(k), sort_keys=True, default=str)]
    trunk_s, own = m["trunk"]["schedule"], args_like["schedule"]
    if (trunk_s.get("kind"), trunk_s.get("peak"), trunk_s.get("warmup"), trunk_s.get("cooldown_start")) != \
            ("wsd", own.get("peak"), own.get("warmup"), None):
        diff.append("schedule (the trunk's warmup and peak)")
    if diff:
        raise RuntimeError(f"{run.name}: the trunk trained with other {diff} than this branch")
    return {"trunk": want.name, "pool": run.branch_pool, "planned_batch": m["planned_batch"], "sha256": m["sha256"],
            "position": m["position"], "step": m["step"]}


def train_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="train_b0prime.py train")
    p.add_argument("--run", choices=list(stage3.RUNS), required=True)
    p.add_argument("--labels-run", required=True, help="the C2 run name (labels/<T*>/<run>)")
    p.add_argument("--budget-minutes", type=float, default=CHUNK_MINUTES)
    p.add_argument("--deadline", type=float, help="epoch seconds: stop (checkpointed) by then (chunk jobs)")
    p.add_argument("--ckpt-minutes", type=float, default=30.0)
    p.add_argument("--overwrite", action="store_true", help="discard this run's outputs and checkpoints")
    p.add_argument("--cleaning-version", type=int, choices=label_sources.CLEANING_VERSIONS, default=CLEANING_VERSION,
                   help=f"the label cleaning version (default {CLEANING_VERSION}; recorded, part of labels_sha256)")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """train.parse_args for a stage 3 run (module docstring, item 1)."""
    a = train_parser().parse_args(argv)
    run = stage3.RUNS[a.run]
    peak, selection = peak_of(run)
    data = offline_data(run, a.labels_run, peak, a.cleaning_version)
    return namespace(a, run, data, peak, selection, grad_checkpointing(), _sha256_base())


def _sha256_base() -> str:
    return train.verify_base()["base_sha256"]


def namespace(a, run: stage3.Run, data: Data, peak: float, selection: dict | None, ckpt: bool,
              base_sha256: str) -> argparse.Namespace:
    """The run's train.py arguments (module docstring, item 1)."""
    _ACTIVE.update(data=data, run=run)
    steps = data.plan["steps"]
    ident = data.labels.identity
    args = argparse.Namespace(
        arm=ARM, recipe=RECIPE_NAME, lr=peak, run_name=run.name, max_steps=steps, train="relabel-stage3",
        stream="c1-batch-plan", batch_seconds=BATCH_SECONDS, grad_checkpointing=ckpt,
        eval_every=max(1, steps // 10) if run.kind == "trunk" else steps, select="final", dev_limit=None,
        eval_batch_size=32, ckpt_minutes=a.ckpt_minutes, seed=run.seed, probe_steps=None, no_powerlog=True,
        overwrite=a.overwrite, smoke=False, spec=run.to_dict(), label_set=LABEL_SET, labels_run=a.labels_run,
        labels_sha256=ident["sha256"], cleaning_version=a.cleaning_version, cleaning=ident.get("cleaning"),
        repetition=ident.get("repetition"), label_transform=ident.get("transform"), label_reject=ident.get("reject"),
        extra_sources=ident.get("extra") or None, plan_sha256=data.replay.identity["plan_sha256"],
        replay_sha256=data.replay.identity["sha256"], stop_batch=data.stop,
        schedule=stage3.schedule_dict(data.schedule), base_sha256=base_sha256, lr_selection=selection,
        branch_of=None, budget_minutes=a.budget_minutes, deadline=a.deadline)
    if selection is not None and identity_of(vars(args)) != selection["identity"]:
        raise RuntimeError(f"{run.name}: its labels, plan or B0 differ from the pilot runs' that chose its learning "
                           f"rate ({identity_of(vars(args))} != {selection['identity']})")
    args.branch_of = branch_of(run, data.replay.plan, vars(args))
    log(f"{run.name}: plan {data.replay.identity['plan_id']}, stop at planned batch {data.stop} (pool "
        f"{run.stop_pool}), {steps} steps ({len(data.plan['empty_batches'])} fully rejected batches), peak {peak:g}, "
        f"schedule {args.schedule}")
    return args


def build_context(args, run: Path, device: str = "cuda"):
    """train.build_context for a stage 3 run (module docstring, item 2)."""
    data, spec = _ACTIVE["data"], _ACTIVE["run"]
    if data is None or spec is None or spec.name != args.run_name:
        raise RuntimeError("parse_args did not prepare this run's data")
    student, info = build_model()
    train.seed_everything(args.seed, student)
    student.to(device)
    if args.grad_checkpointing:
        train.enable_grad_checkpointing(student.encoder)
    evaluator = train.DevEvaluator(args.dev_limit, args.eval_batch_size, device) if spec.kind == "trunk" else None
    ctx = Context(args, run, student, data, evaluator, device, spec)
    _ACTIVE["ctx"] = ctx
    latest, previous, _ = train.ckpt_paths(run)
    if spec.kind == "branch" and not (latest.exists() or previous.exists()):
        start_branch(ctx, args.branch_of, run_dir("trunk"))
    if data.hooks:
        data.on_batch = lambda index: save_branch_point(ctx, data.hooks[index], index)
    td.start_budget(ctx, args.budget_minutes, args.deadline)
    info = {"base": info["base"], "quantized_modules": [], "trainable": info["trainable"], "parameter_accounting": None,
            "teacher": {"labels": f"offline label store ({LABEL_SET}: {data.labels.identity['column']}) under "
                                  f"{data.labels.identity['mask']}", "online_teacher": None,
                        "cleaning": data.labels.identity.get("cleaning"),
                        "repetition": data.labels.identity.get("repetition"),
                        "transform": data.labels.identity.get("transform"),
                        "extra": data.labels.identity.get("extra")},
            "bucketing": {"source": "batch plan re-cut from C1's journal (b0prime plan \"Batch plan\")",
                          "plan": data.replay.identity, "stop_batch": data.stop},
            "b0prime": {"run": spec.to_dict(), "labels": data.labels.identity, "replay": data.replay.identity,
                        "reject_counts": data.labels.real.reject_counts(),
                        "plan": {k: v for k, v in data.plan.items() if k != "empty_batches"},
                        "empty_batches": data.plan["empty_batches"], "schedule": stage3.schedule_dict(data.schedule),
                        "branch_points": {p: b for b, p in data.hooks.items()}, "branch_of": args.branch_of,
                        "recipe": RECIPE, "precision": "FP32 parameters and AdamW state; BF16 autocast "
                                                      "(train.compute_losses)"}}
    return ctx, info


def run_eval(ctx) -> None:
    """The trunk's dev-subset evaluation (train.run_eval, informational); none for the other runs."""
    if ctx.evaluator is not None:
        _TRAIN["run_eval"](ctx)
        return
    train.log(f"step {ctx.state.step}: no dev-subset evaluation ({ctx.args.run_name} is scored after training)")


def label_manifest(labels_run: str) -> dict:
    path = tlabels.production_run_dir(labels_run) / "manifest.json"
    return {"path": str(path.relative_to(llayout.relabel_root())), "sha256": _sha256_file(path)}


def finish(ctx, config_sha: str) -> dict:
    """train.finish for a stage 3 run (module docstring, item 5)."""
    st, args, run, spec = ctx.state, ctx.args, ctx.run, ctx.spec
    if ctx.timer is not None:
        ctx.timer.cancel()
    if st.step != args.max_steps:
        raise RuntimeError(f"final model not ready: step {st.step}/{args.max_steps}")
    common = {"select": "final", "selected_step": st.step, "ramp_steps": 0,
              "warmup_steps": int(ctx.stream.schedule.warmup)}
    if not spec.scored:
        final = [p for p in spec.branch_points if ctx.stream.replay.plan.pool_start(p) == ctx.stream.stop]
        points = {p: final_branch_point(ctx, p, ctx.stream.stop) for p in final}
        subset = st.final_subset or {}
        return common | {"dev_subset_mean_wer": train.mean_wer({n: {"wer": w} for n, w in subset.items()})
                         if subset else math.nan,
                         "dev_subset_meaning": "the trainer's final dev-subset evaluation (informational: a trunk is "
                                               "never scored)",
                         "dev_wer": {}, "dev_mean_wer": math.nan, "scored_on": "not scored (trunk)",
                         "b0prime": {"run": spec.to_dict(), "scored": False, "final_branch_points": points,
                                     "consumed_batches": st.batches_read, "fully_rejected_batches": st.skipped_batches,
                                     "nonfinite_skipped_steps": st.drop_counts.get(tseam.NONFINITE_KEY, 0)}}
    model = ctx.student
    model.eval()
    nemo = run / "model.nemo"
    tmp = nemo.with_name(f"model.tmp{os.getpid()}.nemo")
    model.save_to(str(tmp))
    os.replace(tmp, nemo)
    train.drop_page_cache(nemo)
    model.cpu()
    torch.cuda.empty_cache()
    source = evaluate.describe_source("nemo", nemo, f"stage 3 {spec.name}: B0 fine-tuned on T* labels (A1, FP32)")
    manifest = {"model": "model.nemo", "sha256": source["sha256"], "bytes": nemo.stat().st_size, "run": spec.name,
                "b0prime": spec.name == stage3.B0PRIME, "base_sha256": args.base_sha256,
                "labels": {"labels_run": args.labels_run, "labels_sha256": args.labels_sha256,
                           "label_manifest": label_manifest(args.labels_run), "cleaning_version": args.cleaning_version,
                           "cleaning": args.cleaning, "repetition": args.repetition, "transform": args.label_transform,
                           "extra_sources": args.extra_sources},
                "plan_sha256": args.plan_sha256, "schedule": args.schedule, "lr": args.lr, "seed": args.seed,
                "step": st.step, "config_sha256": config_sha, "git_head": train.git_head(),
                "training_source_hashes": train.training_source_hashes(), "utc": train.utc()}
    write_json(run / "model.json", manifest)
    scorer = train.load_nemo(nemo, ctx.device).eval()
    scores = score_model(scorer, run, source, ctx.device, spec.name)
    del scorer
    torch.cuda.empty_cache()
    full = scores["dev_wer"]
    return common | {"dev_subset_mean_wer": h4({n: 100 * w for n, w in full.items()}) / 100,
                     "dev_subset_meaning": "H4 on full stage 1 dev (fraction), not a 400-subset",
                     "dev_wer": full, "dev_mean_wer": sum(full.values()) / len(full),
                     "dev_utterances": scores["dev_utterances"], "scored_on": "model restored from model.nemo",
                     "nemo": str(nemo), "nemo_sha256": source["sha256"],
                     "b0prime": {"run": spec.to_dict(), "scored": True, "h4": h4({n: 100 * w for n, w in full.items()}),
                                 "consumed_batches": st.batches_read, "fully_rejected_batches": st.skipped_batches,
                                 "nonfinite_skipped_steps": st.drop_counts.get(tseam.NONFINITE_KEY, 0),
                                 "domain": scores["domain_sets"], "model_manifest": "model.json"}}


@contextlib.contextmanager
def seam(deadline: float | None = None):
    """Install the stage 3 replacements in train.py and paths.py; restore them on exit."""
    items = tseam.replacements() + [
        (train, "parse_args", parse_args), (train, "build_context", build_context), (train, "finish", finish),
        (train, "recipe_of", recipe_of), (train, "run_eval", run_eval), (train, "train_step", tseam.guarded_train_step),
        (train, "RESUME_KEYS", tuple(train.RESUME_KEYS) + RESUME_EXTRA),
        (train, "RESUME_ALLOWED_CHANGES", tuple(train.RESUME_ALLOWED_CHANGES) + ("budget_minutes", "deadline")),
        (train, "TRAINING_SOURCES", tuple(train.TRAINING_SOURCES) + SOURCES),
        (train, "OUTPUTS", tuple(train.OUTPUTS) + EXTRA_OUTPUTS), (paths, "RUNS", root() / "runs"),
        (paths, "gpu_lock", td.bounded_lock(deadline))]
    with tseam.installed(items):
        yield


def run_train(argv: list[str]) -> int:
    a = train_parser().parse_args(argv)
    llayout.require_mount()
    if run_done(a.run) and not a.overwrite:
        mismatch = run_mismatch(a.run, a.labels_run, a.cleaning_version)
        if mismatch:
            log(f"refusing: {mismatch}")
            return train.EXIT_RESUME_MISMATCH
        log(f"{run_dir(a.run)} is complete")
        return 0
    _ACTIVE.update(data=None, ctx=None, run=None)
    import score_teacher
    with seam(a.deadline):
        try:
            train.main(argv)
        except NotReady as wait:
            log(f"{a.run}: not started: {wait}")
            return EXIT_NOT_READY
        except Stopped as stop:
            log(f"{a.run}: {stop}")
            return EXIT_STOPPED
        except score_teacher.LockBusy as busy:
            log(f"{a.run}: the GPU lock was not free before the chunk deadline ({busy}); the next chunk resumes")
            return EXIT_BUDGET
        except SystemExit as stop:
            ctx = _ACTIVE["ctx"]
            if stop.code == train.EXIT_INTERRUPTED and ctx is not None and ctx.stop_reason == "budget":
                log(f"{a.run}: stopped at step {ctx.state.step} (planned batch {ctx.state.batches_read}) for the "
                    "chunk budget; the next chunk resumes")
                return EXIT_BUDGET
            raise
    return 0


# ----------------------------------------------------------------------------- scoring

def score_model(net, out: Path, source: dict, device, label: str) -> dict:
    """The stage 1 dev sets (eval-dev/<set>.json) and the domain parts (domain/), each skipped when already written
    with the same configuration (module docstring, "Scorer")."""
    import binary_train
    import devsets
    import terms
    device = torch.device(device)
    mode = binary_train.pin_decoder_graphs_mode(net, label)
    dev = {}
    for name in DEV_SETS:
        path = out / "eval-dev" / f"{name}.json"
        config = td._config(source, "dev")
        result = td._done(path, config)
        if result is None:
            manifest = paths.MANIFESTS / f"{name}.jsonl"
            result = evaluate.evaluate_set(net, name, manifest, source, td.EVAL_BATCH, None, device,
                                           td.EVAL_MAX_SECONDS)
            result.update(config=config, decoder_graphs_mode=binary_train.require_decoder_graphs_mode(net, label))
            write_json(path, result)
            log(f"{label} {name}: WER {100 * result['wer']['wer']:.3f}")
        dev[name] = result
    parts = [p for s in devsets.DEFAULT_SETS for p in devsets.SETS[s]]
    inputs = score_domain.part_manifests(parts, None)
    matchers, domain = {}, {}
    for part in parts:
        manifest, meta = inputs[part]
        path = out / "domain" / f"{part}.json"
        config = td._config(source, "domain", {"manifest_sha256": meta["manifest_sha256"]})
        result = td._done(path, config)
        if result is None:
            result = evaluate.evaluate_set(net, part, manifest, source, score_domain.BATCH_SIZE, None, device,
                                           score_domain.MAX_BATCH_SECONDS)
            kind = TERM_KIND.get(devsets.set_of(part))
            if kind is not None:
                matcher = matchers.get(kind) or matchers.setdefault(kind, terms.matcher(kind))
                result["term_recall"] = score_domain.term_recall(result["records"], matcher)
                result["term_kind"] = kind
            result.update(config=config, part=part, set=devsets.set_of(part),
                          decoder_graphs_mode=binary_train.require_decoder_graphs_mode(net, label))
            write_json(path, result)
        domain[part] = result
    domain_summary = score_domain.summarize(label, domain, list(devsets.DEFAULT_SETS))
    for name in devsets.DEFAULT_SETS:          # recall of the sets stage 3 adds to score_domain's (dev-technical)
        entry = domain_summary["sets"].get(name)
        if entry is not None and name not in devsets.TERM_KIND and name in TERM_KIND:
            got = [domain[p]["term_recall"] for p in devsets.SETS[name]]
            occ, rep = sum(g["occurrences"] for g in got), sum(g["reproduced"] for g in got)
            entry["mean_term_recall"] = (sum(g["recall"] for g in got) / len(got)
                                         if all(g["recall"] is not None for g in got) else None)
            entry["pooled_term_recall"] = {"occurrences": occ, "reproduced": rep, "recall": rep / occ if occ else None}
            entry["term_kind"] = TERM_KIND[name]
    write_json(out / "domain" / "summary.json", domain_summary)
    # eval-stem (b0prime plan v4: targeted and held-out term recall on eval-stem and dev-technical)
    stem_parts = list(devsets.SETS[STEM_SET])
    stem_inputs = score_domain.part_manifests(stem_parts, None)
    stem, stem_matcher = {}, None
    for part in stem_parts:
        manifest, meta = stem_inputs[part]
        path = out / STEM_SET / f"{part}.json"
        config = td._config(source, STEM_SET, {"manifest_sha256": meta["manifest_sha256"]})
        result = td._done(path, config)
        if result is None:
            result = evaluate.evaluate_set(net, part, manifest, source, score_domain.BATCH_SIZE, None, device,
                                           score_domain.MAX_BATCH_SECONDS)
            stem_matcher = stem_matcher or matchers.get(devsets.TERM_KIND[STEM_SET]) or \
                terms.matcher(devsets.TERM_KIND[STEM_SET])
            result["term_recall"] = score_domain.term_recall(result["records"], stem_matcher)
            result.update(config=config, part=part, set=STEM_SET,
                          decoder_graphs_mode=binary_train.require_decoder_graphs_mode(net, label))
            write_json(path, result)
        stem[part] = result
    stem_summary = score_domain.summarize(label, stem, [STEM_SET])
    write_json(out / STEM_SET / "summary.json", stem_summary)
    # the cv27-en dev sample (informational, b0prime plan v4), once its manifest exists
    cv = None
    manifest = llayout.relabel_root() / CV_DEV_SAMPLE
    if manifest.exists():
        path = out / "cv-dev" / f"{CV_DEV_NAME}.json"
        config = td._config(source, "cv-dev", {"manifest_sha256": evaluate.sha256_file(manifest)})
        cv = td._done(path, config)
        if cv is None:
            cv = evaluate.evaluate_set(net, CV_DEV_NAME, manifest, source, score_domain.BATCH_SIZE, None, device,
                                       score_domain.MAX_BATCH_SECONDS)
            cv.update(config=config, decoder_graphs_mode=binary_train.require_decoder_graphs_mode(net, label))
            write_json(path, cv)
    summary = {"label": label, "source": source, "complete": True, "utc": train.utc(), "decoder_graphs_mode": mode,
               "dev_wer": {n: r["wer"]["wer"] for n, r in dev.items()},
               "dev_utterances": {n: r["utterances"] for n, r in dev.items()},
               "h4": h4({n: 100 * r["wer"]["wer"] for n, r in dev.items()}), "domain_sets": domain_summary["sets"],
               "eval_stem": stem_summary["sets"].get(STEM_SET), "cv_dev_wer": None if cv is None else cv["wer"]["wer"]}
    write_json(out / "scores.json", summary)
    return summary


STEM_SET = "eval-stem"
# critical-term recall per domain set: score_domain's (dev-medical: medical) plus dev-technical with the technical
# list (b0prime plan "Gate" item 3: WER and critical-term recall of both domain sets)
TERM_KIND = {"dev-medical": "medical", "dev-technical": "technical"}
CV_DEV_SAMPLE = "cv27-en/dev-sample.jsonl"     # the cv27-en dev sample manifest (its preparation writes it)
CV_DEV_NAME = "cv27-en-dev"


def score_run(name: str, lock_wait: float) -> int:
    """(Re)score a finished run's model.nemo: every part not yet written with this model (e.g. a later cv27-en dev
    sample)."""
    import score_teacher
    llayout.require_mount()
    nemo = run_dir(name) / "model.nemo"
    if not run_done(name) or not nemo.exists():
        raise NotReady(f"{name} has no finished model.nemo")
    source = evaluate.describe_source("nemo", nemo, f"stage 3 {name}: B0 fine-tuned on T* labels (A1, FP32)")
    with score_teacher.gpu_lock(f"b0prime score {name}", lock_wait):
        train.deterministic_mode()
        net = train.load_nemo(nemo, "cuda").eval()
        score_model(net, run_dir(name), source, "cuda", name)
    return 0


def stage1_wers() -> dict[str, float]:
    """B0's stage 1 dev WER (points) per set (relabel_paths.STUDENT_RECORDS["B0"])."""
    out = {}
    for name in DEV_SETS:
        rec = json.loads((relabel_paths.STUDENT_RECORDS["B0"] / f"{name}.json").read_text())
        out[name] = 100 * (rec["wer"]["wer"] if isinstance(rec["wer"], dict) else rec["wer"])
    return out


def b0_done() -> bool:
    p, c = b0_dir() / "scores.json", b0_dir() / "crosscheck.json"
    return (p.exists() and json.loads(p.read_text()).get("complete") is True and c.exists()
            and json.loads(c.read_text()).get("pass") is True)


def score_b0(lock_wait: float, deadline: float | None = None) -> int:
    """B0 re-scored (scores/B0/) and checked against stage 1 (crosscheck.json); a lock not free by the chunk deadline
    is a budget stop (0); a failed check exits EXIT_RESCORE."""
    import score_teacher
    llayout.require_mount()
    wait = lock_wait if deadline is None else max(0.0, min(lock_wait, deadline - time.time()))
    try:
        with score_teacher.gpu_lock("b0prime score-b0", wait):
            train.deterministic_mode()
            source = score_domain.describe("B0")
            net, _ = score_domain.load_model("B0", torch.device("cuda"))
            summary = score_model(net, b0_dir(), source, "cuda", "B0")
    except score_teacher.LockBusy as busy:
        if deadline is None:
            raise
        log(f"score-b0: the GPU lock was not free before the chunk deadline ({busy}); the next chunk scores it")
        return EXIT_BUDGET
    check = stage3.rescore_check({n: 100 * w for n, w in summary["dev_wer"].items()}, stage1_wers())
    write_json(b0_dir() / "crosscheck.json", check | {"utc": train.utc()})
    log(f"B0 re-score: {json.dumps(check['sets'])}; pass={check['pass']}")
    return 0 if check["pass"] else EXIT_RESCORE


# ----------------------------------------------------------------------------- memory probe

def probe_plan_dir():
    """The materialized plan of pools 0-155 (stage 2b's; a prefix of the stage 3 plan) for the probe's batches."""
    from training import batchplan
    _, pool, cid, plan = v1_plan()
    short = batchplan.Plan.load(batchplan.production_dir(cid, schedule_pools()))
    if not batchplan.is_prefix(short, plan):
        raise RuntimeError("the materialized 156-pool plan is not a prefix of the stage 3 plan")
    return pool, cid, plan, short


def schedule_pools() -> int:
    from training import schedule
    return schedule.DECODER_ONLY_POOLS


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def current_rss_gib() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return math.nan


def probe_mode(ckpt: bool, batches, labels, steps: int) -> dict:
    """One probe configuration: a fresh A1 model (gradient checkpointing on or off), the worst-case batches, then
    `steps` timed planned batches (the first untimed); peak VRAM, step time, rows and seconds."""
    import types
    model, info = build_model()
    train.seed_everything(stage3.SEED, model)
    model.to("cuda")
    if ckpt:
        train.enable_grad_checkpointing(model.encoder)
    optimizer, scheduler = tseam.make_optimizer(model, 1e-4, steps)
    ctx = types.SimpleNamespace(student=model, recipe=dict(RECIPE), optimizer=optimizer, scheduler=scheduler,
                                state=types.SimpleNamespace(step=0, batches_read=0, acc={"dropped_rows": 0},
                                                            drop_counts={}, seconds_by_source={}),
                                device="cuda", stream=types.SimpleNamespace(
                                    schedule=types.SimpleNamespace(lr=lambda k: 1e-4)))
    trainable = [p for g in optimizer.param_groups for p in g["params"]]
    train.set_train_modes(model, RECIPE)
    torch.cuda.reset_peak_memory_stats()
    times, seconds, rows, status, worst_peak = [], [], [], {"worst": "ok", "timed": "ok"}, None
    try:
        for kind, source in (("worst", batches["worst"]), ("timed", batches["timed"])):
            for j, batch in enumerate(source):
                ctx.state.batches_read = int(batch["index"]) + 1
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                try:
                    labeled = tseam.label_batch(ctx, batch)
                    if labeled is None:
                        continue
                    train.reseed_cudnn_rnn_dropout()
                    train.train_step(ctx, labeled, trainable)
                except torch.OutOfMemoryError:
                    status[kind] = f"oom at batch {j}"
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    break
                torch.cuda.synchronize()
                if kind == "timed" and j >= 1:
                    times.append(time.perf_counter() - t0)
                    seconds.append(labeled["seconds"])
                    rows.append(labeled["rows"])
                ctx.state.step += 1
            if kind == "worst":
                worst_peak = torch.cuda.max_memory_reserved() / 2**30
            if status[kind] != "ok":
                break
    finally:
        batches["timed"].close()
    peak_alloc = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    process_now = td.gpu_process_gb()
    reserved_now = torch.cuda.memory_reserved() / 2**30
    peak_process = peak_reserved + max(0.0, (process_now or reserved_now) - reserved_now)
    out = {"grad_checkpointing": ckpt, "status": status, "steps_timed": len(times),
           "step_time_s": float(np.mean(times)) if times else None,
           "step_time_p90_s": float(np.percentile(times, 90)) if times else None,
           "audio_seconds_per_step": float(np.mean(seconds)) if seconds else None,
           "rows_per_step": float(np.mean(rows)) if rows else None, "peak_allocated_gib": peak_alloc,
           "peak_reserved_gib": peak_reserved, "worst_reserved_gib": worst_peak, "process_gpu_gib_at_end": process_now,
           "peak_process_gib_estimate": peak_process, "trainable": info["trainable"]["trainable"],
           "host_peak_rss_gib": rss_gib()}
    return out | {"_model": model, "_optimizer": optimizer, "_ctx": ctx}


def plan_worst(replay, labels) -> int:
    """The planned batch with the largest kept rows x longest clip x longest label (characters): a proxy of the
    transducer joint's size."""
    plan, best, worst = replay.plan, -1, 0
    for t in range(plan.n_batches):
        rows = plan.batch(t).tolist()
        kept = [x for x in map(labels.text, replay.batch_ids(t)) if x is not None]
        if kept:
            score = len(kept) * max(int(plan.samples[e]) for e in rows) * max(len(x) for x in kept)
            if score > best:
                best, worst = score, t
    return worst


def probe_batches(replay, labels, steps: int, seed: int) -> dict:
    """Worst-case batches (train_decoder.probe_batches: stage 1 dev-clean concatenated to about 29.5 s per row, 20
    rows, references joined; and plan_worst's planned batch) and a batcher over steps + 1 consecutive planned batches
    from a seeded pool of pools 0-155 (real T* labels, rejected rows dropped by the seam)."""
    worst, _ = td.probe_batches(seed)
    collated = []
    for k, items in enumerate(worst):
        b = train.collate(items)
        b.update(index=k, labels=[it["text"] for it in items])
        collated.append(b)
    w = plan_worst(replay, labels)
    one = label_sources.Batcher(replay, labels, start=w, stop=w + 1)
    try:
        collated.append(one.next(timeout=900))
    finally:
        one.close()
    plan = replay.plan
    pools = [p for p in range(plan.n_pools) if plan.pool_start(p) + steps + 1 <= plan.n_batches]
    p = int(np.random.default_rng([int(seed), 3]).choice(pools))
    start = plan.pool_start(p)
    return {"worst": collated, "timed": TimedBatches(replay, labels, start, steps + 1), "timed_start": start,
            "timed_pool": p}


class TimedBatches:
    """n consecutive planned batches from `start` (label_sources.Batcher, prefetched), iterable once."""

    def __init__(self, replay, labels, start: int, n: int):
        self.batcher, self.n = label_sources.Batcher(replay, labels, start=start, stop=start + n), n

    def __iter__(self):
        for _ in range(self.n):
            yield self.batcher.next(timeout=900)

    def close(self) -> None:
        self.batcher.close()


def checkpoint_round_trip(ctx, out: Path) -> dict:
    """Save a train.py checkpoint of the probe model and load it back as a resume does (host memory and time)."""
    payload = {"step": 1, "student": ctx.student.state_dict(), "optimizer": ctx.optimizer.state_dict(),
               "rng": train.rng_state(ctx.student), "format": train.CKPT_FORMAT, "complete": True}
    out.mkdir(parents=True, exist_ok=True)
    path = out / "probe-ckpt.pt"
    t0 = time.perf_counter()
    with open(path, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    saved = time.perf_counter() - t0
    size = path.stat().st_size
    del payload
    train.drop_page_cache(path)
    t0 = time.perf_counter()
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    load_s = time.perf_counter() - t0
    rss_loaded = current_rss_gib()
    del loaded
    path.unlink()
    return {"bytes": size, "save_s": saved, "load_s": load_s, "rss_with_loaded_checkpoint_gib": rss_loaded,
            "host_peak_rss_gib": rss_gib()}


def data_measure(labels_run: str, pool, plan) -> dict:
    """Host memory and time of the full plan's label store (pools 0-781) and journal rows, opened and released before
    the GPU part (the probe itself trains on pools 0-155)."""
    from training import batchplan
    before, t0 = current_rss_gib(), time.perf_counter()
    labels = open_labels(labels_run, pool, plan.consumed_batches)
    labels_s, with_labels = time.perf_counter() - t0, current_rss_gib()
    t0 = time.perf_counter()
    j = batchplan.read_journal(pool, plan.consumed_batches, with_rows=True)
    journal_s, with_journal = time.perf_counter() - t0, current_rss_gib()
    out = {"pools": plan.n_pools, "labels_open_s": labels_s, "labels_rss_gib": with_labels - before,
           "journal_read_s": journal_s, "journal_rss_gib": with_journal - with_labels, "kept": labels.identity["kept"],
           "records": labels.identity["records"], "occurrences": len(j["ids"])}
    del labels, j
    return out


V2_FACTOR = 1.137            # b0prime plan v4 "Effect": the extra sources add about 13.7% planned batches


def gpu_hours(plan, step_s: float, factor: float = 1.0) -> dict:
    """The pilot's and the main run's planned batches x the probe's step time (b0prime plan "Estimate"); `plan` is the
    stage 3 plan, or the version 1 plan with `factor` the extra sources' expected growth until the plan is built."""
    B = plan.pool_start
    pilot = 5 * B(stage3.PILOT_POOLS) * factor
    trunk = B(stage3.TRUNK_POOLS) * factor
    cool = sum(B(e) - B(s) for s, e in stage3.BRANCHES.values()) * factor
    return {"step_time_s": step_s, "plan": plan.meta["plan_id"], "factor": factor,
            "pilot_planned_batches": round(pilot), "trunk_planned_batches": round(trunk),
            "cooldown_planned_batches": round(cool), "pilot_gpu_hours": pilot * step_s / 3600,
            "main_gpu_hours": (trunk + cool) * step_s / 3600, "total_gpu_hours": (pilot + trunk + cool) * step_s / 3600,
            "note": "probe step time x planned batches (fully rejected batches take no step); excludes replay waits, "
                    "checkpoints, the trunk's dev-subset evaluations and the finishing scores"
                    + ("" if factor == 1.0 else f"; the version 1 plan's batches x {factor} until the stage 3 plan "
                                                "is built")}


def estimate_plan():
    """(the stage 3 plan, 1.0) once built, else (the version 1 plan, V2_FACTOR)."""
    try:
        return pinned_plan()[3], 1.0
    except NotReady:
        return v1_plan()[3], V2_FACTOR


def _swap(plan_factor: tuple, step_s: float) -> tuple:
    return plan_factor[0], step_s, plan_factor[1]


def probe_job(labels_run: str, steps: int, lock_wait: float) -> int:
    """The memory probe (b0prime plan "Resources"): A1 at 600 s batches with and without gradient checkpointing,
    peak VRAM, host memory, step time, the decision and the GPU-hour estimate; probe/probe.json."""
    import score_teacher
    from training import poolaudio
    llayout.require_mount()
    pool, cid, full_plan, plan = probe_plan_dir()
    measured = data_measure(labels_run, pool, full_plan)
    log(f"full-plan data: {json.dumps(measured)}")
    t0 = time.perf_counter()
    labels = open_labels(labels_run, pool, plan.consumed_batches)
    replay = treplay.PlannedReplay(plan, pool, audio="materialized", store=poolaudio.store_dir(cid, plan.pool_batches))
    open_s = time.perf_counter() - t0
    modes, round_trip = {}, None
    with score_teacher.gpu_lock("b0prime memory probe", lock_wait):
        train.deterministic_mode()
        tdt_sha = train.prepare_tdt_loss()
        for ckpt in (False, True):
            batches = probe_batches(replay, labels, steps, stage3.SEED)
            res = probe_mode(ckpt, batches, labels, steps)
            res["timed_from_planned_batch"] = batches["timed_start"]
            res["timed_pool"] = batches["timed_pool"]
            ctx = res.pop("_ctx")
            if round_trip is None and res["status"]["timed"] == "ok":
                round_trip = checkpoint_round_trip(ctx, root() / "probe")
            del ctx, res["_model"], res["_optimizer"]
            torch.cuda.empty_cache()
            modes["ckpt" if ckpt else "no_ckpt"] = res
            log(json.dumps(res))
    replay.close()
    total = 32.0 if not torch.cuda.is_available() else torch.cuda.get_device_properties(0).total_memory / 2**30
    fits = {k: m["status"] == {"worst": "ok", "timed": "ok"} for k, m in modes.items()}
    free = {k: total - m["peak_process_gib_estimate"] for k, m in modes.items()}
    if fits["no_ckpt"] and free["no_ckpt"] >= FREE_GB:
        choice = "no_ckpt"
    elif fits["ckpt"]:
        choice = "ckpt"
    else:
        choice = None
    decision = {"fits": choice is not None, "grad_checkpointing": choice == "ckpt" if choice else None,
                "chosen": choice, "gpu_total_gib": total, "free_gib": free,
                "rule": f"gradient checkpointing if the run without it leaves less than {FREE_GB} GB for others or "
                        "does not fit", "lease_vram_gb": math.ceil(modes[choice]["peak_process_gib_estimate"]
                                                                   + LEASE_MARGIN_GB) if choice else None}
    result = {"steps": steps, "modes": modes, "decision": decision, "checkpoint_round_trip": round_trip,
              "full_plan_data": measured, "probe_data": {"plan": plan.identity(), "open_s": open_s,
                                                          "labels_sha256": labels.identity["sha256"]},
              "host_peak_rss_gib": rss_gib(), "tdt_kernel_sha256": tdt_sha, "utc": train.utc(),
              "estimate": gpu_hours(*_swap(estimate_plan(), modes[choice]["step_time_s"]))
              if choice and modes[choice]["step_time_s"] else None}
    write_json(probe_path(), result)
    log(json.dumps({k: v for k, v in result.items() if k != "modes"}))
    return 0 if choice else 1


# ----------------------------------------------------------------------------- chunks and status

def item_done(item: tuple) -> bool:
    kind, name = item
    return b0_done() if kind == "score" else run_done(name)


def item_command(item: tuple, labels_run: str, minutes: float, deadline: float,
                 cleaning_version: int = CLEANING_VERSION) -> list[str]:
    here = Path(__file__).resolve().parent
    kind, name = item
    cmd = [str(here / "python"), str(here / "train_b0prime.py")]
    if kind == "score":
        return cmd + ["score-b0", "--deadline", f"{deadline:.0f}"]
    return cmd + ["train", "--run", name, "--labels-run", labels_run, "--budget-minutes", f"{minutes:.1f}",
                  "--deadline", f"{deadline:.0f}", "--cleaning-version", str(cleaning_version)]


def chunk(labels_run: str, budget_minutes: float, through: str, runner=subprocess.run,
          cleaning_version: int = CLEANING_VERSION) -> int:
    """One chunk job: the next unfinished items of ORDER up to `through` (pilot: the B0 re-score, the three pilot runs
    and the sanity pair; main: also the trunk and the branches), each in its own process (which takes the parakeet GPU
    lock for its own duration), until the budget leaves less than MIN_CHUNK_MINUTES, an item stops on the budget, or
    an item fails (its exit code: EXIT_STOPPED when the WSD rule stops stage 3, EXIT_RESCORE, ...)."""
    llayout.require_mount()
    items = ORDER[:THROUGH[through]]
    mismatched = [m for m in (run_mismatch(n, labels_run, cleaning_version) for k, n in items if k == "train") if m]
    if mismatched:
        log("chunk: refusing to mix runs of other labels: move these runs aside (or pass their labels run and "
            "cleaning version): " + "; ".join(mismatched))
        return train.EXIT_RESUME_MISMATCH
    t0, deadline = time.monotonic(), time.time() + budget_minutes * 60
    for item in items:
        if item_done(item):
            continue
        left = budget_minutes - (time.monotonic() - t0) / 60
        if left < MIN_CHUNK_MINUTES:
            break
        log(f"chunk: {item} with {left:.0f} min left")
        rc = runner(item_command(item, labels_run, left, deadline, cleaning_version),
                    cwd=str(Path(__file__).resolve().parent)).returncode
        if rc != 0:
            log(f"chunk: {item} exited {rc}")
            return rc
        if not item_done(item):
            break
    log(f"chunk: {sum(item_done(i) for i in items)}/{len(items)} items done (through {through})")
    return 0


def status() -> dict:
    out = {"B0 re-score": "done" if b0_done() else "pending"}
    for _, name in ORDER[1:]:
        if run_done(name):
            out[name] = "done"
        elif (run_dir(name) / "ckpt" / "latest.pt").exists():
            m = run_dir(name) / "metrics.jsonl"
            last = train.read_log_jsonl(m)[0][-1:] if m.exists() else []
            out[name] = f"in progress (step {last[0]['step'] if last else '?'})"
        else:
            out[name] = "pending"
    for name, path in (("selection", selection_path()), ("wsd rule", wsd_rule_path())):
        out[name] = json.loads(path.read_text()) if path.exists() else "not frozen"
    return out


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "train":
        return run_train(argv[1:])
    parser = argparse.ArgumentParser(prog="train_b0prime.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    p = sub.add_parser("score-run")
    p.add_argument("--run", choices=[n for n, r in stage3.RUNS.items() if r.scored], required=True)
    p.add_argument("--lock-wait", type=float, default=4 * 3600.0)
    p = sub.add_parser("score-b0")
    p.add_argument("--lock-wait", type=float, default=4 * 3600.0)
    p.add_argument("--deadline", type=float, help="epoch seconds (chunk jobs): give up waiting for the lock")
    p = sub.add_parser("probe")
    p.add_argument("--labels-run", required=True)
    p.add_argument("--steps", type=int, default=PROBE_STEPS)
    p.add_argument("--lock-wait", type=float, default=4 * 3600.0)
    p = sub.add_parser("chunk")
    p.add_argument("--labels-run", required=True)
    p.add_argument("--through", choices=sorted(THROUGH), required=True)
    p.add_argument("--budget-minutes", type=float, default=CHUNK_MINUTES)
    p.add_argument("--cleaning-version", type=int, choices=label_sources.CLEANING_VERSIONS, default=CLEANING_VERSION,
                   help="passed to every train item")
    sub.add_parser("status")
    sub.add_parser("report")
    args = parser.parse_args(argv)
    if args.cmd == "plan":
        try:
            return build_plan()
        except NotReady as wait:
            log(f"plan: not ready: {wait}")
            return EXIT_NOT_READY
    if args.cmd == "score-run":
        return score_run(args.run, args.lock_wait)
    if args.cmd == "score-b0":
        return score_b0(args.lock_wait, args.deadline)
    if args.cmd == "probe":
        return probe_job(args.labels_run, args.steps, args.lock_wait)
    if args.cmd == "chunk":
        return chunk(args.labels_run, args.budget_minutes, args.through, cleaning_version=args.cleaning_version)
    if args.cmd == "status":
        print(json.dumps(status(), indent=1, default=str))
        return 0
    import b0prime_report
    return b0prime_report.main()


if __name__ == "__main__":
    sys.exit(main())
