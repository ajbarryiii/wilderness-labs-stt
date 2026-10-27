"""Stage 2c: the synthetic arms of the decoder-only stage (synthetic plan "Arms", "Mixing", "Augmentation",
"Evaluation", "Resources"; decoder-only plan).

    ./python train_synthetic.py train --arm B-TS --labels-run RUN [--budget-minutes 110]
                                                                       one chunk of one run (GPU; resumable)
    ./python train_synthetic.py score-stem --arm B-T|T-T               eval-stem of the selected 2b T* run (GPU)
    ./python train_synthetic.py probe --student B|T --labels-run RUN   the mixed-batch probe (GPU)
    ./python train_synthetic.py read-probe [--regions 0:3,76:79,153:156] [--kinds TS,TSn]
                                                                       synthetic reads, replay probe's rule (CPU)
    ./python train_synthetic.py chunk --labels-run RUN [--budget-minutes 110]
                                                                       one chunk job: the next unfinished items in
                                                                       ORDER, each a subprocess, within the budget
    ./python train_synthetic.py status | report [--out PATH]

Arms (ARMS): B-TS and B-TSn train Kea-B's decoder, T-TS and T-TSn Kea-T's, on T* labels plus the kept clips of the
synthetic generation run "full" (synthetic.mixing.ClipPool.load: refused while the run is provisional or its hygiene
ran under another term split), cut into pools 0-155 by the batch plan's version 2 (training/batchplan.py
compute_synthetic, synthetic plan "Mixing": each pool's draw cut together with its real occurrences). TS passes the
synthetic rows through the augmentation (synthetic/augment.py, mixing.transform_for); TSn serves them as stored. Both
use the same plan, so they get the same clips in the same batches. Everything else is stage 2b's, reused from
train_decoder.py: the frozen-encoder recipe, the student, the label lookup with rejected rows dropped after the cut,
the WSD schedule on the planned-batch index (warmup 2% of the version 2 plan's planned batches, stable through pool
132, cooldown over pools 133-155), checkpoints, resume, chunk budgets and the finishing scores (full dev,
dev-medical and dev-technical, the decoder-step proxy). Runs: $RELABEL_ROOT/decoder-only/<arm>/lr<lr>/.

Learning rate (no new search): each arm runs at the learning rate decoder_report.select chose for its student's T*
arm (BASE: B-T, T-T) on the selection part of the frozen dev split. frozen_selection writes it at its first use to
decoder-only/selection/<T* arm>.json (the three runs' selection-part H4, the rule, the split's and the runs' summary
SHA-256) and refuses a later different result; before all three T* runs have finished it raises NotReady, and no
2c run starts (exit EXIT_NOT_READY). The record is a resume key (lr_selection) and goes into the run's summary.

The T* arm's data: parse_args reads the selected T* run's recorded arguments (its config.json) and refuses another
label run, another labels_sha256 (the same labels through the same label cleaning and the same repetition rejection,
each following that run: on iff it ran with it) and a version 2 plan whose base plan is not the T* run's version 1
plan. So a 2c arm trains on every real occurrence the T* arm trains on, with the same label and disposition, in the
same pools and pool order, plus each pool's synthetic draw; within a pool its batches differ, since they are cut with
the clips (synthetic plan "Arms"; synthetic_data's refusals hold the same checks for any caller).

Seam: train_decoder.seam(arm) (stage 2b's replacements, unchanged, the non-finite step guard included) plus this
module's parse_args, build_context (train_decoder.build_context, plus a "synthetic" block in config.json), finish
(train_decoder.finish, then eval-stem), RESUME_KEYS + RESUME_EXTRA and TRAINING_SOURCES + EXTRA_SOURCES, all restored
on exit. train_decoder.py and the shared training modules are not modified, so the 2b arms train, record and resume
exactly as before.

eval-stem (synthetic plan "Evaluation"; stage 2b does not score it): every 2c run after training, and each selected
T* run (ORDER's "stem" items), with score_domain.py's settings (evaluate.evaluate_set at its batch, the pinned
decoder mode, term recall with the set's committed list), into decoder-only/eval-stem/<arm>/lr<lr>/; a part already
written with the same configuration is skipped. synthetic_report.py computes the targeted and held-out recall.

ORDER: the two T* runs' eval-stem, then B-TS, B-TSn, T-TS, T-TSn. A chunk runs nothing (EXIT_NOT_READY) until
readiness() is empty: every stage 2b item (train_decoder.ORDER) done, both selections frozen, the "full" clip pool
loadable, the version 1 and version 2 plans of pools 0-155 built, and the augmentation pool loadable.

Probes (synthetic plan "Resources"):
- probe (GPU): the complete mixed batch (mixed_batch: planned batch b as the replay batcher assembles it, synthetic
  rows through the TS augmentation, train.collate's padding; then the seam's label_batch, which drops rejected rows
  and tokenizes, and a train step) on the version 2 plan's batch with the most trained rows and the one with the most
  synthetic rows, each as augmented and at the bound (BoundTransform: every synthetic row at its planned key length,
  the longest any augmentation may make it), plus PROBE_SAMPLE seeded batches for the step time; peak VRAM, the
  declared VRAM (labeling/layout.py's margin) and the GPU-hour estimate. Output decoder-only/synthetic-probe/<s>.json.
- read-probe (CPU): training/replay_probe.py's rule on the version 2 plan: per region (pool ranges; default the
  start, middle and end) and kind (TS: reads and augmentation; TSn: reads), a fresh reader on materialized audio,
  the page cache of the region's pool files and synthetic clip files dropped, batches taken as fast as they come, the
  rate against replay_probe.required_rate(); the synthetic clips' read and transform times are reported apart.
  Output training/batch-plan/probes/<version 2 plan id>.json.
"""
from __future__ import annotations

import train_decoder as td  # noqa: I001  imports finetune/parakeet-ternary/train.py first (CUDA/numba setup)
import train  # finetune/parakeet-ternary

import argparse
import collections
import contextlib
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

import evaluate  # finetune/parakeet-ternary
import relabel_paths
from labeling import layout as llayout
from synthetic import augment, mixing
from synthetic import layout as slayout
from training import batchplan as bp
from training import devsplit, schedule
from training import labels as tlabels
from training import replay as treplay
from training import seam as tseam

ARMS = {"B-TS": ("B", "TS"), "B-TSn": ("B", "TSn"), "T-TS": ("T", "TS"), "T-TSn": ("T", "TSn")}
BASE = {arm: f"{s}-T" for arm, (s, _) in ARMS.items()}       # the student's T* arm of stage 2b
LABEL_SET = "T*"
SYNTHETIC_RUN = "full"
STEM_SET = "eval-stem"
ORDER = [("stem", t_arm) for t_arm in dict.fromkeys(BASE.values())] + [("train", arm) for arm in ARMS]
EXIT_NOT_READY = 75
RESUME_EXTRA = ("synthetic_arm", "base_arm", "lr_selection", "synthetic_run", "clip_pool", "augment",
                "base_plan_sha256")
EXTRA_SOURCES = ("relabel/train_synthetic.py", "relabel/synthetic_report.py", "relabel/synthetic/__init__.py",
                 "relabel/synthetic/mixing.py", "relabel/synthetic/augment.py", "relabel/synthetic/layout.py",
                 "relabel/synthetic/textgen.py", "relabel/synthetic/vocab.py")
PROBE_LR = 1e-4
PROBE_SAMPLE = 20
PROBE_REPEATS = 3


class NotReady(RuntimeError):
    """An input a 2c run needs does not exist yet (the 2b selection, the clip pool, a plan)."""


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [synthetic-arms] {msg}", flush=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- learning rate

def selection_path(t_arm: str) -> Path:
    return td.root() / "selection" / f"{t_arm}.json"


def frozen_selection(t_arm: str) -> dict:
    """decoder_report.select for a 2b T* arm, frozen at first use (module docstring); NotReady while a run is pending."""
    import decoder_report
    split = devsplit.load(td.split_path(), devsplit.dev_manifests())
    sel = decoder_report.select(t_arm, split)
    if sel["selected"] is None:
        raise NotReady(f"{t_arm}: the learning-rate selection needs its three runs ({', '.join(sel['pending'])} "
                       "pending)")
    record = {"arm": t_arm, "part": "selection", "selected": sel["selected"], "selected_name": sel["selected_name"],
              "selection_h4": sel["selection_h4"], "rule": sel["rule"], "split_sha256": split.sha256,
              "summaries_sha256": {td.LR_NAMES[lr]: _sha256(td.run_dir(t_arm, lr) / "summary.json") for lr in td.LRS}}
    path = selection_path(t_arm)
    if path.exists():
        old = json.loads(path.read_text())
        if {k: old.get(k) for k in record} != record:
            raise RuntimeError(f"{path} records another selection than the runs give now: {old} != {record}")
        return old
    relabel_paths.require_mount()
    td.write_json(path, record | {"recorded_utc": train.utc(),
                                  "note": "the learning rate of the stage 2c arms of this student (synthetic plan "
                                          "\"Arms\": no new learning-rate search)"})
    return record


def selected_lr(t_arm: str) -> float | None:
    """The frozen selection's learning rate, or None before it is frozen (no recomputation)."""
    path = selection_path(t_arm)
    return float(json.loads(path.read_text())["selected"]) if path.exists() else None


def base_args(t_arm: str, lr: float) -> dict:
    """The recorded arguments of a 2b run (config.json "args")."""
    return json.loads((td.run_dir(t_arm, lr) / "config.json").read_text())["args"]


# ----------------------------------------------------------------------------- data

def synthetic_data(kind: str, plan, base_plan, pool_dir: Path, store: Path, labels, clips, sched,
                   augment_pool=None, seed: int = slayout.SEED) -> tseam.OfflineData:
    """The trainer's data of a synthetic arm of kind "TS" or "TSn": the version 2 plan `plan` (drawn from `clips`, on
    the version 1 plan `base_plan`) on materialized audio, its synthetic rows through mixing.transform_for(kind), the
    T* label store, the schedule, every planned batch. Refuses any other plan, base plan, clip pool or label set."""
    transform = mixing.transform_for(kind, seed=seed, augment_pool=augment_pool)
    if plan.version != bp.SYNTHETIC_VERSION or base_plan.version != bp.VERSION:
        raise RuntimeError("a synthetic arm trains on a version 2 plan drawn on the version 1 plan")
    if plan.meta["base_plan_sha256"] != base_plan.sha256:
        raise RuntimeError("the version 2 plan was not drawn on this version 1 plan")
    if plan.meta["synthetic"]["clip_pool"] != str(clips.identity):
        raise RuntimeError("the version 2 plan was drawn from another clip pool")
    if labels.label_set != LABEL_SET:
        raise RuntimeError(f"the synthetic arms train on {LABEL_SET} labels, not {labels.label_set}")
    replay = treplay.PlannedReplay(plan, pool_dir, audio="materialized", store=store, synthetic=clips,
                                   synthetic_transform=transform)
    return tseam.OfflineData(replay, labels, sched, plan.n_batches)


def _pinned() -> tuple[dict, Path, str]:
    from labeling import config as lconfig
    from labeling.__main__ import c1_dir, pinned
    pin = pinned()
    if float(pin["batch"]["batch_seconds"]) != td.BATCH_SECONDS:
        raise RuntimeError(f"the pinned stream batches {pin['batch']['batch_seconds']} s, not {td.BATCH_SECONDS}")
    return pin, c1_dir(), lconfig.config_id(pin)


def plan_dirs(clips) -> tuple[Path, Path]:
    """The production version 1 plan of pools 0-155 (the 2b arms') and its version 2 plan drawn from `clips`."""
    _, _, cid = _pinned()
    return bp.production_dir(cid, td.PLAN_POOLS), bp.production_dir(cid, td.PLAN_POOLS, clips=clips)


def offline_data(arm: str, labels_run: str, lr: float, cleaning: bool = True,
                 repetition: bool = True) -> tseam.OfflineData:
    """synthetic_data on the production plans, clip pool, T* store (read through label cleaning unless cleaning is
    False, with repetition rejection unless repetition is False) and schedule (schedule.decoder_only on the version
    2 plan)."""
    from training import poolaudio
    pin, pool, cid = _pinned()
    clips = mixing.ClipPool.load(SYNTHETIC_RUN)
    d1, d2 = plan_dirs(clips)
    base_plan, plan = bp.Plan.load(d1), bp.Plan.load(d2)
    for p in (base_plan, plan):
        if p.meta["pinned_hash"] != pin["pinned_hash"] or p.budget != td.BATCH_SECONDS * bp.SAMPLE_RATE:
            raise RuntimeError(f"{p.dir}: a batch plan of another pinned configuration or budget")
    labels = tlabels.LabelStore.open(tlabels.production_run_dir(labels_run), pool, label_set=LABEL_SET,
                                     through_batch=plan.consumed_batches,
                                     cleaning=td.label_cleaning(labels_run, cleaning),
                                     repetition=td.label_repetition(labels_run, repetition))
    return synthetic_data(ARMS[arm][1], plan, base_plan, pool, poolaudio.store_dir(cid, plan.pool_batches), labels,
                          clips, schedule.decoder_only(lr, plan))


# ----------------------------------------------------------------------------- the trainer seam

def train_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="train_synthetic.py train")
    p.add_argument("--arm", choices=sorted(ARMS), required=True)
    p.add_argument("--labels-run", required=True, help="the C2 run name; must be the selected T* run's")
    p.add_argument("--budget-minutes", type=float, default=td.CHUNK_MINUTES)
    p.add_argument("--deadline", type=float, help="epoch seconds: stop (checkpointed) by then (chunk jobs)")
    p.add_argument("--ckpt-minutes", type=float, default=30.0)
    p.add_argument("--overwrite", action="store_true", help="discard this run's outputs and checkpoints")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """train.parse_args for a synthetic arm: train_decoder.parse_args's fields at the frozen selection's learning
    rate, on the selected T* run's labels and plan (module docstring), plus RESUME_EXTRA."""
    a = train_parser().parse_args(argv)
    student, kind = ARMS[a.arm]
    t_arm = BASE[a.arm]
    split = devsplit.load(td.split_path(), devsplit.dev_manifests())
    ident = td.ensure_student(student)
    sel = frozen_selection(t_arm)
    lr = float(sel["selected"])
    base = base_args(t_arm, lr)
    if base["labels_run"] != a.labels_run:
        raise RuntimeError(f"{a.arm}: --labels-run {a.labels_run} is not the {t_arm} run's ({base['labels_run']})")
    data = offline_data(a.arm, a.labels_run, lr, cleaning=base.get("cleaning") is not None,
                        repetition=base.get("repetition") is not None)
    if data.labels.identity["sha256"] != base["labels_sha256"]:
        raise RuntimeError(f"{a.arm}: the T* labels differ from the {t_arm} run's (labels_sha256)")
    base_plan_sha = data.replay.plan.meta["base_plan_sha256"]
    if base_plan_sha != base["plan_sha256"]:
        raise RuntimeError(f"{a.arm}: the version 2 plan is not drawn on the {t_arm} run's plan")
    td._ACTIVE["data"] = data
    steps = data.plan["steps"]
    syn = data.replay.identity["synthetic"]
    log(f"{a.arm} at {t_arm}'s {td.LR_NAMES[lr]}: batch plan {data.replay.identity['plan_id']}, "
        f"{data.replay.consumed_batches} consumed batches -> {data.stop} planned batches, {steps} with a trained row, "
        f"{data.plan['synthetic_rows']} synthetic rows")
    lr_selection = {"base_arm": t_arm, "base_run": f"{t_arm}/{td.LR_NAMES[lr]}", "selected": lr,
                    "selected_name": td.LR_NAMES[lr], "selection_h4": sel["selection_h4"], "rule": sel["rule"],
                    "split_sha256": sel["split_sha256"], "summaries_sha256": sel["summaries_sha256"],
                    "why": "the learning rate selected for this student's T* arm on the selection part of the frozen "
                           "dev split (synthetic plan \"Arms\": no new learning-rate search)"}
    return argparse.Namespace(
        arm=a.arm, recipe="decoder-only", lr=lr, run_name=td.LR_NAMES[lr], max_steps=steps,
        train="relabel-stage2-journal", stream="c1-batch-plan-v2", batch_seconds=td.BATCH_SECONDS,
        grad_checkpointing=False, eval_every=steps, select="final", dev_limit=None, eval_batch_size=td.EVAL_BATCH,
        ckpt_minutes=a.ckpt_minutes, seed=td.SEED, probe_steps=None, no_powerlog=True, overwrite=a.overwrite,
        smoke=False, student=student, label_set=LABEL_SET, labels_run=a.labels_run,
        labels_sha256=data.labels.identity["sha256"], cleaning=data.labels.identity.get("cleaning"),
        repetition=data.labels.identity.get("repetition"),
        replay_sha256=data.replay.identity["sha256"], plan_sha256=data.replay.identity["plan_sha256"],
        stop_batch=data.stop, schedule=data.schedule.to_dict(), export_sha256=ident["export_sha256"],
        init_sha256=ident["checkpoint_sha256"], split_sha256=split.sha256, budget_minutes=a.budget_minutes,
        deadline=a.deadline, synthetic_arm=kind, base_arm=t_arm, lr_selection=lr_selection,
        synthetic_run=SYNTHETIC_RUN, clip_pool=syn["clip_pool"], augment=syn["transform"],
        base_plan_sha256=base_plan_sha)


def synthetic_block(args, data) -> dict:
    plan = data.replay.plan
    return {"arm": args.arm, "kind": args.synthetic_arm, "base_arm": args.base_arm, "lr_selection": args.lr_selection,
            "run": args.synthetic_run, "clip_pool": args.clip_pool, "augment": args.augment,
            "plan": {"plan_id": plan.meta["plan_id"], "version": plan.version,
                     "base_plan_sha256": plan.meta["base_plan_sha256"], "n_synthetic": plan.meta["n_synthetic"],
                     "synthetic": plan.meta["synthetic"], "stats": plan.meta["stats"]},
            "synthetic_rows": data.plan["synthetic_rows"], "real_occurrences": data.plan["occurrences"],
            "kept_occurrences": data.plan["kept_occurrences"],
            "mixing": "synthetic plan \"Mixing\": each pool's draw cut together with its real occurrences"}


def build_context(args, run: Path, device: str = "cuda"):
    """train_decoder.build_context, plus the synthetic block in config.json."""
    ctx, info = td.build_context(args, run, device)
    info["synthetic"] = synthetic_block(args, ctx.stream)
    return ctx, info


def finish(ctx, config_sha: str) -> dict:
    """train_decoder.finish (decoder file, full dev, domain sets, proxy), then eval-stem of the trained model."""
    result = td.finish(ctx, config_sha)
    args = ctx.args
    net, source, _ = td.scoring_model(args.student, ctx.run / "decoder.safetensors", ctx.device)
    stem = score_stem(net, stem_dir(args.arm, args.lr), source, ctx.device, f"{args.arm} {td.LR_NAMES[args.lr]}")
    del net
    torch.cuda.empty_cache()
    result["synthetic"] = synthetic_block(args, ctx.stream) | {
        "eval_stem": stem["sets"].get(STEM_SET), "eval_stem_dir": str(stem_dir(args.arm, args.lr).relative_to(td.root()))}
    return result


@contextlib.contextmanager
def seam(arm: str, deadline: float | None = None):
    """train_decoder.seam plus this module's replacements (module docstring); everything restored on exit."""
    with td.seam(arm, deadline):
        items = [(train, "parse_args", parse_args), (train, "build_context", build_context), (train, "finish", finish),
                 (train, "RESUME_KEYS", tuple(train.RESUME_KEYS) + RESUME_EXTRA),
                 (train, "TRAINING_SOURCES", tuple(train.TRAINING_SOURCES) + EXTRA_SOURCES)]
        with tseam.installed(items):
            yield


def run_train(argv: list[str]) -> int:
    a = train_parser().parse_args(argv)
    relabel_paths.require_mount()
    try:
        lr = float(frozen_selection(BASE[a.arm])["selected"])
    except NotReady as wait:
        log(f"{a.arm}: not started: {wait}")
        return EXIT_NOT_READY
    out = td.run_dir(a.arm, lr)
    if (out / "summary.json").exists() and not a.overwrite:
        log(f"{out} is complete")
        return 0
    td._ACTIVE.update(data=None, ctx=None)
    import score_teacher
    with seam(a.arm, a.deadline):
        try:
            train.main(argv)
        except score_teacher.LockBusy as busy:
            log(f"{a.arm} {td.LR_NAMES[lr]}: the GPU lock was not free before the chunk deadline ({busy}); the next "
                "chunk resumes")
            return td.EXIT_BUDGET
        except SystemExit as stop:
            ctx = td._ACTIVE["ctx"]
            if stop.code == train.EXIT_INTERRUPTED and ctx is not None and ctx.stop_reason == "budget":
                log(f"{a.arm} {td.LR_NAMES[lr]}: stopped at step {ctx.state.step} (planned batch "
                    f"{ctx.state.batches_read}) for the chunk budget; the next chunk resumes")
                return td.EXIT_BUDGET
            raise
    return 0


# ----------------------------------------------------------------------------- eval-stem

def stem_dir(arm: str, lr: float) -> Path:
    return td.root() / "eval-stem" / arm / td.LR_NAMES[lr]


def stem_done(arm: str, lr: float) -> bool:
    p = stem_dir(arm, lr) / "summary.json"
    return p.exists() and json.loads(p.read_text()).get("complete") is True


def score_stem(net, out: Path, source: dict, device, label: str) -> dict:
    """eval-stem's parts with score_domain.py's settings into out/<part>.json and out/summary.json (module docstring)."""
    import binary_train
    import devsets
    import score_domain
    import terms
    parts = list(devsets.SETS[STEM_SET])
    inputs = score_domain.part_manifests(parts, None)
    matcher, results = None, {}
    for part in parts:
        manifest, meta = inputs[part]
        path = out / f"{part}.json"
        config = td._config(source, STEM_SET, {"manifest_sha256": meta["manifest_sha256"]})
        result = td._done(path, config)
        if result is None:
            result = evaluate.evaluate_set(net, part, manifest, source, score_domain.BATCH_SIZE, None,
                                           torch.device(device), score_domain.MAX_BATCH_SECONDS)
            matcher = matcher or terms.matcher(devsets.TERM_KIND[STEM_SET])
            result["term_recall"] = score_domain.term_recall(result["records"], matcher)
            result.update(config=config, part=part, set=STEM_SET,
                          decoder_graphs_mode=binary_train.require_decoder_graphs_mode(net, label))
            td.write_json(path, result)
            log(f"{label} {part}: WER {100 * result['wer']['wer']:.3f}")
        results[part] = result
    summary = score_domain.summarize(label, results, [STEM_SET]) | {"complete": True, "utc": train.utc()}
    td.write_json(out / "summary.json", summary)
    return summary


def score_base_stem(t_arm: str, lock_wait: float, deadline: float | None = None) -> int:
    """eval-stem of the selected 2b T* run; with a deadline (chunk jobs), a lock not free by then is a budget stop."""
    import score_teacher
    relabel_paths.require_mount()
    lr = float(frozen_selection(t_arm)["selected"])
    run = td.run_dir(t_arm, lr)
    wait = lock_wait if deadline is None else max(0.0, min(lock_wait, deadline - time.time()))
    try:
        with score_teacher.gpu_lock(f"synthetic eval-stem {t_arm}", wait):
            train.deterministic_mode()
            net, source, _ = td.scoring_model(td.ARMS[t_arm][0], run / "decoder.safetensors", "cuda")
            summary = score_stem(net, stem_dir(t_arm, lr), source, "cuda", f"{t_arm} {td.LR_NAMES[lr]}")
    except score_teacher.LockBusy as busy:
        if deadline is None:
            raise
        log(f"eval-stem {t_arm}: the GPU lock was not free before the chunk deadline ({busy}); the next chunk scores it")
        return td.EXIT_BUDGET
    log(f"eval-stem {t_arm} {td.LR_NAMES[lr]}: mean WER {summary['sets'][STEM_SET]['mean_wer']:.3f}")
    return 0


# ----------------------------------------------------------------------------- probes

def probe_batches(replay, labels, sample: int, seed: int) -> dict:
    """The version 2 plan's planned batch with the most trained rows (kept real rows plus synthetic rows; ties to the
    earliest), the one with the most synthetic rows (ties: more trained rows, then the earliest), and `sample` distinct
    seeded batches with a trained row (sorted)."""
    rows, syn = [], []
    for t in range(replay.through_batch):
        mask = replay.synthetic_mask(t)
        s = sum(mask)
        rows.append(s + sum(labels.text(i) is not None for i, m in zip(replay.batch_ids(t), mask) if not m))
        syn.append(s)
    most_rows = int(np.argmax(rows))
    most_syn = min(range(len(rows)), key=lambda t: (-syn[t], -rows[t], t))
    candidates = [t for t, n in enumerate(rows) if n > 0]
    rng = np.random.default_rng([int(seed), 9])
    picked = sorted(int(x) for x in rng.choice(candidates, size=min(sample, len(candidates)), replace=False))
    return {"most_rows": most_rows, "most_synthetic": int(most_syn), "sample": picked,
            "trained_rows": {"most_rows": rows[most_rows], "most_synthetic": rows[most_syn]},
            "synthetic_rows": {"most_rows": syn[most_rows], "most_synthetic": syn[most_syn]}}


def mixed_batch(replay, labels, b: int, transform=None) -> dict:
    """Planned batch b as training assembles it (replay.ReplayBatcher.assemble: rows in cut order, synthetic rows
    through the replay's transform, or through `transform` when given, train.collate's padding), before the seam's
    label_batch drops rejected rows."""
    saved = replay.synthetic_transform
    if transform is not None:
        replay.synthetic_transform = transform
    batcher = treplay.ReplayBatcher(replay, labels, start=b, stop=b)
    try:
        return batcher.assemble(b)
    finally:
        batcher.close()
        replay.synthetic_transform = saved


class BoundTransform:
    """Every synthetic row at its planned key (batchplan.synthetic_key: the longest the TS augmentation can make it),
    its audio repeated to that length, its label unchanged: the padded worst case of any augmentation draw."""

    identity = {"kind": "bound", "rule": bp.SYN_KEY_RULE}

    def __call__(self, item: dict, index: int) -> dict:
        n = bp.synthetic_key(len(item["audio"]))
        audio = np.resize(np.asarray(item["audio"], dtype=np.float32), n)
        return {**item, "audio": audio, "duration": n / bp.SAMPLE_RATE, "augment": [["bound", {"samples": n}]]}


def probe_job(s: str, labels_run: str, lock_wait: float, sample: int = PROBE_SAMPLE,
              repeats: int = PROBE_REPEATS) -> int:
    """The mixed-batch probe of student s (module docstring): batches are assembled before the GPU lock is taken."""
    import score_teacher
    relabel_paths.require_mount()
    data = offline_data(f"{s}-TS", labels_run, PROBE_LR)
    replay, labels = data.replay, data.labels
    picks = probe_batches(replay, labels, sample, td.SEED)
    jobs = [("most_rows", picks["most_rows"], None), ("most_rows_bound", picks["most_rows"], BoundTransform()),
            ("most_synthetic", picks["most_synthetic"], None),
            ("most_synthetic_bound", picks["most_synthetic"], BoundTransform())]
    jobs += [("sample", t, None) for t in picks["sample"]]
    prepared = [(name, t, mixed_batch(replay, labels, t, transform)) for name, t, transform in jobs]
    replay.close()
    with score_teacher.gpu_lock(f"synthetic mixed-batch probe {s}", lock_wait):
        train.deterministic_mode()
        tdt_sha = train.prepare_tdt_loss()
        net, info = td.build_student(s)
        train.seed_everything(td.SEED, net)
        net.to("cuda")
        optimizer, scheduler = tseam.make_optimizer(net, PROBE_LR, 1)
        state = types.SimpleNamespace(step=0, batches_read=0, acc=collections.defaultdict(float), drop_counts={},
                                      seconds_by_source={})
        ctx = types.SimpleNamespace(student=net, recipe=dict(td.RECIPE), optimizer=optimizer, scheduler=scheduler,
                                    state=state, device="cuda",
                                    stream=types.SimpleNamespace(schedule=types.SimpleNamespace(lr=lambda k: PROBE_LR)))
        trainable = [p for g in optimizer.param_groups for p in g["params"]]
        td.set_train_modes(net, td.RECIPE)
        td.verify(net)
        torch.cuda.reset_peak_memory_stats()
        entries, warm = [], False
        for name, t, batch in prepared:
            width = int(batch["audio"].shape[1])
            entry = {"name": name, "batch": t, "pool": replay.pool_of(t), "rows_served": len(batch["ids"]),
                     "synthetic_rows": sum(x == "synthetic" for x in batch["sources"]), "width_samples": width,
                     "padded_seconds": len(batch["ids"]) * width / bp.SAMPLE_RATE, "audio_seconds": batch["seconds"],
                     "times_s": [], "status": "ok"}
            for r in range(1 + (repeats if name != "sample" else 0)):
                state.batches_read = t + 1
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                try:
                    labeled = tseam.label_batch(ctx, batch)
                    if labeled is None:
                        entry["status"] = "no trained row"
                        break
                    train.reseed_cudnn_rnn_dropout()
                    train.train_step(ctx, labeled, trainable)
                except torch.OutOfMemoryError:
                    entry["status"] = f"oom at repetition {r}"
                    break
                torch.cuda.synchronize()
                if warm and (r >= 1 or name == "sample"):
                    entry["times_s"].append(time.perf_counter() - t0)
                warm = True
                entry["rows_trained"] = labeled["rows"]
                state.step += 1
            entry["step_time_s"] = float(np.mean(entry["times_s"])) if entry["times_s"] else None
            entries.append(entry)
            log(json.dumps({k: v for k, v in entry.items() if k != "times_s"}))
        peak_alloc = torch.cuda.max_memory_allocated() / 2**30
        peak_reserved = torch.cuda.max_memory_reserved() / 2**30
        process_now = td.gpu_process_gb()
        reserved_now = torch.cuda.memory_reserved() / 2**30
        enc_sha = td.encoder_sha256(net)
    peak_process = peak_reserved + max(0.0, (process_now or reserved_now) - reserved_now)
    sampled = [x for e in entries if e["name"] == "sample" for x in e["times_s"]]
    step = float(np.mean(sampled)) if sampled else None
    result = {"student": s, "arm": f"{s}-TS", "plan": replay.identity["plan_id"], "picks": picks, "batches": entries,
              "status": {e["name"]: e["status"] for e in entries if e["name"] != "sample"} |
              {"sample": sorted({e["status"] for e in entries if e["name"] == "sample"})},
              "step_time_s": step, "step_time_p90_s": float(np.percentile(sampled, 90)) if sampled else None,
              "peak_allocated_gib": peak_alloc, "peak_reserved_gib": peak_reserved,
              "process_gpu_gib_at_end": process_now, "peak_process_gib_estimate": peak_process,
              "declared_vram_gb": math.ceil(peak_process + llayout.VRAM_MARGIN_GB),
              "host_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
              "tdt_kernel_sha256": tdt_sha, "encoder_sha256": enc_sha, "trainable": info["trainable"],
              "labels": {"run": labels_run, "sha256": labels.identity["sha256"]},
              "data": "version 2 plan batches as training assembles them (TS augmentation as drawn at the batch, and "
                      "the bound: every synthetic row at its planned key length), rejected rows dropped by the seam",
              "utc": train.utc()}
    if step:
        per_run = replay.through_batch * step / 3600
        result["estimate"] = {"gpu_hours_per_run": per_run, "runs": 2, "gpu_hours_this_student": 2 * per_run,
                              "note": "mean step time of the sampled mixed batches x the version 2 plan's planned "
                                      "batches; replay reading, checkpoints and finishing scores not included"}
    td.write_json(td.root() / "synthetic-probe" / f"{s}.json", result)
    log(json.dumps({k: v for k, v in result.items() if k not in ("batches", "trainable", "picks")}))
    return 0


class TimedClips:
    """A ClipPool whose reads are counted and timed (the read probe's synthetic clips)."""

    def __init__(self, pool):
        self.pool, self.clips, self.identity, self.by_id = pool, pool.clips, pool.identity, pool.by_id
        self.n, self.seconds, self.bytes = 0, 0.0, 0

    def read(self, clip_id: str) -> np.ndarray:
        t0 = time.perf_counter()
        audio = self.pool.read(clip_id)
        self.seconds += time.perf_counter() - t0
        self.n += 1
        self.bytes += os.path.getsize(self.by_id[clip_id]["audio_filepath"])
        return audio


class TimedTransform:
    """A synthetic transform whose calls are timed; its identity is the wrapped one's."""

    def __init__(self, transform):
        self.transform, self.identity = transform, getattr(transform, "identity", None)
        self.n, self.seconds = 0, 0.0

    def __call__(self, item: dict, index: int) -> dict:
        t0 = time.perf_counter()
        out = self.transform(item, index)
        self.seconds += time.perf_counter() - t0
        self.n += 1
        return out


def read_probe_run(plan, pool_dir: Path, store: Path, clips, regions: list[tuple[int, int]], kinds: list[str],
                   max_seconds: float, augment_pool=None, seed: int = slayout.SEED) -> dict:
    """The synthetic read probe (module docstring) on the version 2 plan `plan` over materialized audio."""
    from training import poolaudio, replay_probe
    need, need_info = replay_probe.required_rate()
    out = {"plan": plan.identity(), "required_batches_per_s": need, "required": need_info, "regions": [],
           "rule": "training/replay_probe.py (b0prime plan \"Replay throughput probe\") on the version 2 plan",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    by_index = sorted(clips.clips, key=lambda c: c["id"])
    for kind in kinds:
        for lo, hi in regions:
            timed = TimedClips(clips)
            transform = mixing.transform_for(kind, seed=seed, augment_pool=augment_pool)
            timed_t = None if transform is None else TimedTransform(transform)
            t0 = time.monotonic()
            r = treplay.PlannedReplay(plan, pool_dir, audio="materialized", store=store, synthetic=timed,
                                      synthetic_transform=timed_t)
            opened = time.monotonic() - t0
            start, stop = plan.pool_start(lo), plan.pool_start(hi)
            clip_files = sorted({by_index[e - bp.SYN_FLAG]["audio_filepath"] for t in range(start, stop)
                                 for e in plan.batch(t).tolist() if e >= bp.SYN_FLAG})
            files = [store / "pools" / f"{poolaudio.pool_name(p)}.flac" for p in range(lo, min(hi + 1, plan.n_pools))]
            dropped = replay_probe.drop_cache(files + clip_files)
            res = replay_probe.time_region(r, start, stop, max_seconds)
            res.update(kind=kind, reader="materialized", pools=[lo, hi], open_seconds=opened,
                       cache_dropped_files=dropped, rss_gib=replay_probe.rss_gib(), hwm_gib=replay_probe.hwm_gib(),
                       synthetic={"clips": timed.n, "distinct_clip_files": len(clip_files), "bytes": timed.bytes,
                                  "read_seconds": timed.seconds,
                                  "read_ms_per_clip": 1000 * timed.seconds / timed.n if timed.n else None,
                                  "transform_seconds": timed_t.seconds if timed_t else 0.0,
                                  "transform_ms_per_clip": 1000 * timed_t.seconds / timed_t.n if timed_t and timed_t.n
                                  else None},
                       meets_required=res["batches_per_s"] is not None and res["batches_per_s"] >= need)
            out["regions"].append(res)
            print(json.dumps(res), flush=True)
            r.close()
            del r
    out["host_peak_gib"] = replay_probe.hwm_gib()
    out["meets_required"] = bool(out["regions"]) and all(x["meets_required"] for x in out["regions"])
    return out


def read_probe_job(regions_text: str | None, kinds: list[str], max_seconds: float) -> int:
    from training import poolaudio, replay_probe
    relabel_paths.require_mount()
    _, pool, cid = _pinned()
    clips = mixing.ClipPool.load(SYNTHETIC_RUN)
    plan = bp.Plan.load(plan_dirs(clips)[1])
    n = plan.n_pools
    regions = replay_probe.parse_regions(regions_text) if regions_text else [(0, 3), (n // 2 - 1, n // 2 + 2),
                                                                              (n - 3, n)]
    res = read_probe_run(plan, pool, poolaudio.store_dir(cid, plan.pool_batches), clips, regions, kinds, max_seconds)
    path = slayout.relabel_root() / "training" / "batch-plan" / "probes" / f"{plan.meta['plan_id']}.json"
    td.write_json(path, res)
    print(json.dumps({k: v for k, v in res.items() if k != "regions"}, indent=1))
    return 0


# ----------------------------------------------------------------------------- chunks and status

def readiness() -> list[str]:
    """What is missing before the 2c runs may start (module docstring, ORDER); empty when ready."""
    problems = []
    pending = [i for i in td.ORDER if not td.item_done(i)]
    if pending:
        problems.append(f"stage 2b items not done: {pending}")
    for t_arm in dict.fromkeys(BASE.values()):
        try:
            frozen_selection(t_arm)
        except NotReady as wait:
            problems.append(str(wait))
    try:
        clips = mixing.ClipPool.load(SYNTHETIC_RUN)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        problems.append(f"clip pool {SYNTHETIC_RUN!r} is not finalized: {error}")
    else:
        for name, d in zip(("version 1", "version 2"), plan_dirs(clips)):
            if not (d / "plan.json").exists():
                problems.append(f"no {name} batch plan at {d} (./python -m training.batchplan build --pools "
                                f"{td.PLAN_POOLS}{' --synthetic-run ' + SYNTHETIC_RUN if name == 'version 2' else ''})")
    try:
        augment.Pool.load()
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        problems.append(f"the augmentation pool does not load: {error}")
    return problems


def item_done(item: tuple) -> bool:
    kind, name = item
    lr = selected_lr(name if kind == "stem" else BASE[name])
    if lr is None:
        return False
    if kind == "stem":
        return stem_done(name, lr)
    return (td.run_dir(name, lr) / "summary.json").exists()


def item_command(item: tuple, labels_run: str, minutes: float, deadline: float) -> list[str]:
    here = Path(__file__).resolve().parent
    kind, name = item
    cmd = [str(here / "python"), str(here / "train_synthetic.py")]
    if kind == "stem":
        return cmd + ["score-stem", "--arm", name, "--deadline", f"{deadline:.0f}"]
    return cmd + ["train", "--arm", name, "--labels-run", labels_run, "--budget-minutes", f"{minutes:.1f}",
                  "--deadline", f"{deadline:.0f}"]


def chunk(labels_run: str, budget_minutes: float, runner=subprocess.run) -> int:
    """One chunk job: nothing before readiness() is empty; then the next unfinished items of ORDER, each in its own
    process (which takes the parakeet GPU lock for its own duration), until the budget leaves less than
    train_decoder.MIN_CHUNK_MINUTES or a run stops on it."""
    relabel_paths.require_mount()
    devsplit.load(td.split_path(), devsplit.dev_manifests())
    problems = readiness()
    if problems:
        for p in problems:
            log(f"chunk: not ready: {p}")
        return EXIT_NOT_READY
    t0, deadline = time.monotonic(), time.time() + budget_minutes * 60
    for item in ORDER:
        if item_done(item):
            continue
        left = budget_minutes - (time.monotonic() - t0) / 60
        if left < td.MIN_CHUNK_MINUTES:
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
        kind, name = item
        base = name if kind == "stem" else BASE[name]
        lr = selected_lr(base)
        key = f"eval-stem {name}" if kind == "stem" else name
        if lr is None:
            out[key] = f"waiting for the {base} learning-rate selection"
            continue
        key = f"{key}/{td.LR_NAMES[lr]}"
        if item_done(item):
            out[key] = "done"
        elif kind == "train" and (td.run_dir(name, lr) / "ckpt" / "latest.pt").exists():
            m = td.run_dir(name, lr) / "metrics.jsonl"
            last = train.read_log_jsonl(m)[0][-1:] if m.exists() else []
            out[key] = f"in progress (step {last[0]['step'] if last else '?'})"
        else:
            out[key] = "pending"
    return out


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "train":
        return run_train(argv[1:])
    parser = argparse.ArgumentParser(prog="train_synthetic.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("score-stem")
    p.add_argument("--arm", choices=sorted(set(BASE.values())), required=True)
    p.add_argument("--lock-wait", type=float, default=4 * 3600.0)
    p.add_argument("--deadline", type=float, help="epoch seconds (chunk jobs): give up waiting for the lock")
    p = sub.add_parser("probe")
    p.add_argument("--student", choices=sorted(td.STUDENTS), required=True)
    p.add_argument("--labels-run", required=True)
    p.add_argument("--sample", type=int, default=PROBE_SAMPLE)
    p.add_argument("--repeats", type=int, default=PROBE_REPEATS)
    p.add_argument("--lock-wait", type=float, default=4 * 3600.0)
    p = sub.add_parser("read-probe")
    p.add_argument("--regions", help="pool ranges lo:hi, comma-separated (default: start, middle, end)")
    p.add_argument("--kinds", default="TS,TSn")
    p.add_argument("--max-seconds", type=float, default=120.0)
    p = sub.add_parser("chunk")
    p.add_argument("--labels-run", required=True)
    p.add_argument("--budget-minutes", type=float, default=td.CHUNK_MINUTES)
    sub.add_parser("status")
    p = sub.add_parser("report")
    p.add_argument("--out", type=Path, help="RESULTS-decoder-only.md (default: $RELABEL_ROOT/decoder-only/)")
    args = parser.parse_args(argv)
    try:
        if args.cmd == "score-stem":
            return score_base_stem(args.arm, args.lock_wait, args.deadline)
        if args.cmd == "probe":
            return probe_job(args.student, args.labels_run, args.lock_wait, args.sample, args.repeats)
        if args.cmd == "read-probe":
            return read_probe_job(args.regions, args.kinds.split(","), args.max_seconds)
    except NotReady as wait:
        log(f"{args.cmd}: not ready: {wait}")
        return EXIT_NOT_READY
    if args.cmd == "chunk":
        return chunk(args.labels_run, args.budget_minutes)
    if args.cmd == "status":
        print(json.dumps(status(), indent=1))
        return 0
    import synthetic_report
    return synthetic_report.main(args.out)


if __name__ == "__main__":
    sys.exit(main())
