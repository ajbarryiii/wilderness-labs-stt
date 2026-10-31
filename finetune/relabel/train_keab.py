"""Stage 4 (keab-prime plan v1.2): Kea-B' = Kea-B (MB1) continued with B0' as teacher, and its matched B0 control.

    ./python train_keab.py plan [--provisional]                     batch plan version 3 of pools 0-781 (CPU job)
    ./python train_keab.py read-probe [--provisional]               the read probe of that plan (CPU job)
    ./python train_keab.py labels --labels-run RUN [--provisional]  both arms' label indexes on that plan (CPU job)
    ./python train_keab.py train --run NAME --labels-run RUN [--budget-minutes 110] [--deadline EPOCH]
                                                                    one chunk of one run (GPU; resumable)
    ./python train_keab.py probe --labels-run RUN --scope S [--provisional] [--steps 200]
                                                                    the memory probe (GPU)
    ./python train_keab.py smoke --arm A --labels-run RUN --scope S --steps N [--provisional]
                                                                    a short training run under $RELABEL_ROOT/keab-prime/
                                                                    smoke/ (GPU; never a production path)
    ./python train_keab.py chunk --labels-run RUN --through pilot|main|score [--budget-minutes 110]
                                                                    one chunk job: the next unfinished items of ORDER
    ./python-gpu1bit train_keab.py score --model M --mode x1|x3 [--limit N]
    ./python train_keab.py score --model M --mode torch [--limit N]
                                                                    deployed-mode (x1, x3) or PyTorch fake-quant scoring
    ./python train_keab.py export-init                              the main arm's step 0 student exported (GPU)
    ./python train_keab.py status | report

Arms (training/stage4.py ARMS; the teacher package): "main" trains on T*'s labels with B0' ($RELABEL_ROOT/b0prime/runs/
b100/model.nemo, sha 497ac8ca...) as encoder target and B0''s prediction and joint networks; "control" on B0's labels
(b0_raw under T*'s disposition mask) with B0 (the pinned .nemo) as encoder target and B0's decoder. Everything else is
shared and recorded (MATCHED_KEYS): the plan, the extra sources and their labels, cleaning version 3, the student, the
8-bit scope, the activation quantizer, the loss seam, the schedule and the seed.

Training is finetune/parakeet-ternary/train.py's loop (unchanged) inside seam(): stage 3's offline-label replacements
(training/seam.py: the learning rate per planned batch, AdamW with a fresh state, rejected rows dropped after the cut,
the non-finite step guard), the fused joint+TDT loss seam (finetune/tdt-fused), and this module's:

1. train.parse_args: the run's arguments (namespace): arm P2 with recipe P2 (encoder matching at weight 1.0, the
   prediction and joint networks frozen, a quantized student: train.recipe_of's own flags), the run spec and its arm,
   the peak (a pilot's own, else the frozen selection), the seed, max_steps (the planned batches with a labeled row
   before the run's stop), the labels (labels_run, label_set, cleaning version 3, labels_sha256 with its parts), the
   plan and replay hashes, the schedule, the 8-bit scope (step 0's chosen scope, read from
   $RELABEL_ROOT/keab-prime/step0/report.json; a smoke or the probe names it), the teacher (name and SHA-256), the
   student's initialization (MB1's latent checkpoint, sha e1b76065...), the activation quantizer's and 8-bit form's
   identities, gradient checkpointing (the memory probe's decision, on this plan and these labels), a branch's
   branch_of. RESUME_KEYS gain them all, so a checkpoint of another configuration is refused (exit 78).
2. train.build_context: the student (build_student): the pinned B0 .nemo with MB1's 264 binary modules (its ramp kind,
   at weight fraction 1 from step 0: Context.ramp = 0) and MB1's latent FP32 weights (a fresh start; a resume restores
   the checkpoint's), the teacher's prediction and joint weights copied in bit for bit, then the 8-bit training form of
   the scope (training/eightbit.apply_keab_scope: the frozen decoder parts replaced by fp32(W8) once; under S2
   pre_encode.out computing with W8 and a straight-through gradient, binary row scales fp32(fp16(alpha))), then
   train.configure_trainable (the whole encoder trainable, prediction and joint frozen in eval mode, their dropout 0;
   SpecAugment on), then the activation quantizer on the input of every binary module (training/actquant.py); verified
   by verify(). The teacher (B0' or B0, FP32) frozen in eval mode, never in the optimizer.
3. The data (Data, stage 3's): batch plan version 3 (training/batchplan.py, EXTRA_SOURCES) on materialized pool audio
   and the three extra stores, labels from label_sources.Labels (the arm's label store at cleaning version 3 with
   repetition rejection and the read-time reject list, persisted per label identity under $RELABEL_ROOT/keab-prime/
   label-index/; the synthetic union's stored hygiene forms; Common Voice through cleaning layer 1 of version 3; the
   AMI pairs' IHM labels through the arm's own store, training/ami_store.PairLabels).
4. train.label_batch (label_batch): stage 3's, then the teacher's encoder output for the kept rows (teacher_encode, as
   P2's teacher: features with dither 0 and pad_to 0, the encoder under BF16 autocast, FP32 out). The teacher's input
   (teacher_inputs) is the batch's own audio, padded as the student's, i.e. the clean audio of a real row and the same
   TS-augmented waveform of a synthetic row (before the student's dither and SpecAugment), except an AMI pair's row,
   whose teacher input is its IHM recording (equal encoder frames are checked here and by train.compute_losses).
5. train.finish (finish): a pilot writes its latent model (model.pt) and is scored in the training-equivalent PyTorch
   mode (the student itself in eval mode with its fake-quantized activations and 8-bit parts) on stage 1's dev sets;
   a branch also writes its 8-bit bundle and scoring view (training/keab_export.py) for the deployed-mode score items;
   a trunk writes its final branch point.
6. paths.RUNS: $RELABEL_ROOT/keab-prime/runs; paths.gpu_lock: bounded by the chunk's deadline; TRAINING_SOURCES gain
   SOURCES (the cleaning version 3 tables included).

Scoring (score, keab-prime plan "Scoring and gate"): deployed mode (x1: the scoring view on the real i8x1 kernels,
training/deployed.py; x3: the same view on the deployed int8_mma stack) under ./python-gpu1bit with CUBLAS_WORKSPACE_
CONFIG unset; PyTorch fake-quant mode (torch) under ./python. Sets (SET_GROUPS): dev (H4 and YODAS), domain
(dev-medical, dev-technical with critical-term recall), stem (eval-stem with technical term recall), cv (the cv27-en dev
sample), ami-sdm (AMI SDM dev). Per utterance records keep token IDs (token-sequence mismatch). Outputs:
$RELABEL_ROOT/keab-prime/scores/<model>/<mode>/<set>.json.
"""
from __future__ import annotations

import os as _os
_CUBLAS_AT_START = _os.environ.get("CUBLAS_WORKSPACE_CONFIG")  # train.py sets it on import; deployed scoring unsets it
import train  # noqa: I001, E402  finetune/parakeet-ternary/train.py first: it configures CUDA/numba

import argparse
import contextlib
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

import train_b0prime as tb  # noqa: E402  stage 3's trainer (its seam machinery; puts finetune/tdt-fused on sys.path)
import tdt_fused  # noqa: E402
import evaluate  # noqa: E402  finetune/parakeet-ternary
import paths  # noqa: E402  finetune/parakeet-ternary
import train_decoder as td  # noqa: E402
from labeling import layout as llayout  # noqa: E402
from training import actquant, label_index, label_sources, stage3, stage4  # noqa: E402
from training import eightbit as eb  # noqa: E402  puts finetune/parakeet-binary on sys.path
from training import labels as tlabels  # noqa: E402
from training import repetition as trep  # noqa: E402
from training import replay as treplay  # noqa: E402
from training import seam as tseam  # noqa: E402

import binary_quant  # noqa: E402  finetune/parakeet-binary/binary_quant.py, unchanged

RECIPE = {"encoder_matching": True, "train_pred_joint": False, "quantized": True}   # train.recipe_of("P2", "P2")
ARM = RECIPE_NAME = "P2"
CLEANING_VERSION = 3                 # keab-prime plan "Inputs": T*'s and B0's labels through cleaning version 3
CLEANING_VERSIONS = (3,)
BATCH_SECONDS = tb.BATCH_SECONDS
PLAN_VERSION = 3
CHUNK_MINUTES, MIN_CHUNK_MINUTES = tb.CHUNK_MINUTES, tb.MIN_CHUNK_MINUTES
EXIT_BUDGET, EXIT_NOT_READY = tb.EXIT_BUDGET, tb.EXIT_NOT_READY
EXIT_CHECK = 81
PROBE_STEPS = 200
FREE_GB, LEASE_MARGIN_GB = tb.FREE_GB, tb.LEASE_MARGIN_GB
BINARY_MODULES = binary_quant.EXPECTED_MODULES          # 264
MB1_SHA256 = "e1b760657eff102a1a1b19d70a82ce63c2e18e822b3faec05436cdc63f320d51"
MB1_STEP = 250_000
MB1_RAMP = "ternary"                 # MB1's recorded ramp kind (m1 + binary); at weight fraction 1 every ramp is B_ste
B0P_SHA256 = "497ac8ca2f08807797f0de89dd93e24c689a03013eaa5af207b34201251175af"
DEV_SETS, H4 = tb.DEV_SETS, tb.H4
SOURCES = tuple(tb.SOURCES) + (
    "relabel/train_keab.py", "relabel/keab_report.py", "relabel/keab_step0.py", "relabel/training/stage4.py",
    "relabel/training/actquant.py", "relabel/training/clip_union.py", "relabel/training/ami_store.py",
    "relabel/training/eightbit.py", "relabel/training/keab_export.py", "relabel/training/deployed.py",
    "relabel/training/replay_probe.py", "relabel/synthetic/pairs.py", "relabel/synthetic/vocab.py",
    "relabel/terms/cleaning/variants_v3.tsv", "relabel/terms/cleaning/choices_v3.tsv",
    "relabel/terms/cleaning/spelling_cues.tsv", "relabel/terms/cleaning/units.tsv",
    "parakeet-binary/binary_quant.py", "parakeet-binary/binary_export.py")
STAGE4_KEYS = ("keab_arm", "scope", "teacher", "init", "actquant", "eightbit", "teacher_input", "smoke_of")
RESUME_EXTRA = tuple(tb.RESUME_EXTRA) + STAGE4_KEYS
# what the arms share (keab-prime plan "Arms matched": everything but the teacher package)
MATCHED_KEYS = ("labels_run", "cleaning_version", "cleaning", "repetition", "label_reject", "label_transform",
                "extra_sources", "plan_sha256", "replay_sha256", "base_sha256", "loss_seam", "scope", "init",
                "actquant", "eightbit", "teacher_input", "grad_checkpointing", "batch_seconds", "recipe", "arm")
ARM_KEYS = ("keab_arm", "label_set", "labels_sha256", "teacher")      # the teacher package
EXTRA_OUTPUTS = ("branch-points", "model.pt", "model.json", "export", "scores.json", "eval-dev")
RUNS = stage4.RUNS                    # the run table (tests substitute a small one)
TEACHER_INPUT = {"real": "the row's own (clean) audio", "synthetic": "the same TS-augmented waveform as the student",
                 "ami-sdm": "the pair's IHM recording", "before": "the student's dither and SpecAugment",
                 "padding": "the student's padded length", "encoder": "BF16 autocast, FP32 output, dither 0, pad_to 0"}
_ACTIVE: dict = {"data": None, "ctx": None, "run": None, "scope": None, "smoke": None}


class NotReady(tb.NotReady):
    """An input a run needs does not exist yet (step 0's scope, a store, the plan, the selection, the probe)."""


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [keab] {msg}", flush=True)


# ----------------------------------------------------------------------------- layout

def root() -> Path:
    return llayout.relabel_root() / "keab-prime"


def run_dir(name: str) -> Path:
    if _ACTIVE.get("smoke"):
        return smoke_root() / _ACTIVE["smoke"] / "runs" / name
    return root() / "runs" / name


def smoke_root() -> Path:
    out = root() / "smoke"
    assert not out.is_relative_to(root() / "runs")
    return out


def selection_path() -> Path:
    return root() / "selection.json"


def probe_path() -> Path:
    return root() / "probe" / "probe.json"


def read_probe_path() -> Path:
    return root() / "read-probe" / "read-probe.json"


def label_index_root() -> Path:
    return root() / "label-index"


def scores_dir(model: str, mode: str) -> Path:
    return root() / "scores" / model / mode


def step0_report() -> Path:
    return root() / "step0" / "report.json"


def results_md() -> Path:
    return llayout.relabel_root() / "plans" / "RESULTS-keab-prime.md"


write_json = td.write_json


def _sha256_file(path: Path) -> str:
    return evaluate.sha256_file(Path(path))


def vram_lease() -> None:
    """Keep PyTorch's allocator within the job's shared VRAM lease (AGENT_JOBS_VRAM_FRACTION; none outside a job)."""
    fraction = os.environ.get("AGENT_JOBS_VRAM_FRACTION")
    if fraction and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(fraction))


def cached_sha256(path: Path) -> str:
    """SHA-256 of a large file, cached under $RELABEL_ROOT/keab-prime/caches/sha256 (keab_step0's cache)."""
    import keab_step0
    return keab_step0.cached_sha256(path)


# ----------------------------------------------------------------------------- the 8-bit scope (step 0)

def step0_scope() -> str:
    """Step 0's chosen scope (none, S1 or S2) from its production report; NotReady before it exists."""
    path = step0_report()
    if not path.exists():
        raise NotReady(f"no step 0 report at {path} yet (keab_step0.py report): the 8-bit scope is not chosen")
    r = json.loads(path.read_text())
    scope = r.get("chosen_scope")
    if r.get("decision") is not True or r.get("limit") is not None or scope not in eb.SCOPES:
        raise NotReady(f"{path} holds no decision (chosen_scope {scope!r}, decision {r.get('decision')!r})")
    return scope


def scope_record(scope: str) -> dict:
    """What the 8-bit form of a scope is (recorded with the run)."""
    if scope not in eb.SCOPES:
        raise ValueError(f"scope must be one of {eb.SCOPES}, got {scope!r}")
    parts = eb.KEAB_SCOPES[scope]
    return {"scope": scope, "format": "int8", "parts": list(parts), "frozen_parts": [p for p in parts
                                                                                      if p in eb.FROZEN_PARTS],
            "pre_encode_out": "w8_ste" if "pre_encode_out" in parts else "fp", "binary_scales":
            "fp32(fp16(alpha))" if scope == "S2" else "fp32", "eightbit_sha256": _sha256_file(Path(eb.__file__))}


# ----------------------------------------------------------------------------- extra sources (batch plan version 3)

SYNTHETIC_RUNS = ("full", "pairs", "pairs-c")      # synthetic plan v5.2: the source is their union, under split v3
SYNTHETIC_DRAWS = 2                                 # each clip drawn about twice
AMI_DRAWS = 7                                       # ami_sdm.py draws: 506,910 / 69,136 = 7.33 -> 7
SYNTHETIC_SEED, CV_SEED, AMI_SEED = 20261025, 20261029, 20261030    # 2c's, stage 3's, new
# (name, row-id prefix, store, share, TS augmentation, seed, labels)
EXTRA_SOURCES = (
    ("synthetic", "syn:", "synthetic runs " + " ".join(SYNTHETIC_RUNS), f"about:{SYNTHETIC_DRAWS}", True,
     SYNTHETIC_SEED, "as stored"),
    ("cv27-en", "cv27-en:", "corpora/cv27-en/store", "once", False, CV_SEED, "layer 1"),
    ("ami-sdm", "ami-sdm:", "corpora/ami-sdm/store", f"times:{AMI_DRAWS}", False, AMI_SEED, "pair"))
AMI_DRAWS_FILE = "corpora/ami-sdm/draws.json"


def source_stores(provisional: bool = False) -> list:
    """(config, clip store) of EXTRA_SOURCES; NotReady while a store is not written yet. provisional: the synthetic
    union of the runs that exist (the probe and smokes before run pairs-c is final; never a production run)."""
    from synthetic import layout as slayout
    from synthetic import mixing
    from training import ami_store
    from training.clip_union import ClipUnion
    out = []
    for cfg in EXTRA_SOURCES:
        name, prefix, where, share, augment, seed, labels = cfg
        try:
            if where.startswith("synthetic runs "):
                parts = []
                for run in where.split()[2:]:
                    try:
                        parts.append((run, mixing.ClipPool.load(run, split_path=slayout.split_version_path(3))))
                    except FileNotFoundError:
                        if not (provisional and run == SYNTHETIC_RUNS[-1]):
                            raise
                        log(f"provisional: synthetic run {run} has no clip store yet; the union leaves it out")
                pool = ClipUnion(parts)
            elif labels == "pair":
                pool = ami_store.PairStore.load(llayout.relabel_root() / where)
                check_ami_draws(pool)
            else:
                pool = mixing.ClipPool.load(llayout.relabel_root() / where, synthetic_hygiene=False)
        except FileNotFoundError as error:
            raise NotReady(f"extra source {name}: no clip store yet ({error})") from error
        out.append((cfg, pool))
    return out


def check_ami_draws(store) -> dict:
    """The AMI share's draws per clip (ami_sdm.py draws) agree with AMI_DRAWS on this store."""
    path = llayout.relabel_root() / AMI_DRAWS_FILE
    if not path.exists():
        raise NotReady(f"no {path} yet (ami_sdm.py draws)")
    d = json.loads(path.read_text())
    if d.get("draws") != AMI_DRAWS or d.get("store_clips_sha256") != str(store.identity):
        raise RuntimeError(f"{path}: draws {d.get('draws')} of store {d.get('store_clips_sha256')}, this code has "
                           f"{AMI_DRAWS} on {store.identity}")
    return d


def is_provisional(stores) -> bool:
    syn = next(pool for (cfg, pool) in stores if cfg[0] == "synthetic")
    return [p[0] for p in syn.parts] != list(SYNTHETIC_RUNS)


def plan_sources(stores) -> list:
    from training import batchplan
    return [batchplan.Source(name, prefix, pool, seed, share, augment)
            for (name, prefix, _, share, augment, seed, _), pool in stores]


def replay_sources(stores, sources) -> list:
    """treplay.ExtraSource per source: the synthetic rows through stage 2c's TS augmentation, the others as stored."""
    from synthetic import mixing
    return [treplay.ExtraSource(src, mixing.transform_for("TS", seed=cfg[5]) if cfg[4] else None)
            for (cfg, _), src in zip(stores, sources)]


def label_providers(stores, sources, real, labels_run: str, cleaning_version: int = CLEANING_VERSION) -> dict:
    """{prefix: provider} of the extra sources (EXTRA_SOURCES' label rule): "as stored" (label_sources.ClipLabels),
    "layer 1" (ClipLabels through cleaning layer 1 of the run's version, the label run's acronym lexicon), "pair"
    (ami_store.PairLabels over `real`, the arm's own StoreLabels)."""
    from training import ami_store
    from training import cleaning as tcleaning
    out = {}
    for ((name, _, _, _, _, _, rule), _), src in zip(stores, sources):
        if rule == "as stored":
            out[src.prefix] = label_sources.ClipLabels(src.clips, src.prefix)
        elif rule == "layer 1":
            lexicon = tcleaning.production_dir(labels_run, cleaning_version) if cleaning_version >= 3 else None
            out[src.prefix] = label_sources.ClipLabels(
                src.clips, src.prefix, transform=label_sources.layer1(cleaning_version, name, lexicon))
        elif rule == "pair":
            out[src.prefix] = ami_store.PairLabels(src.clips, real, prefix=src.prefix)
        else:
            raise ValueError(f"extra source {name}: unknown label rule {rule!r}")
    return out


# ----------------------------------------------------------------------------- the plan

def pinned_plan(stores=None, provisional: bool = False):
    """(pin, C1 directory, config id, batch plan version 3 of pools 0-781, its batchplan.Source list); NotReady before
    a store or the plan exists."""
    from training import batchplan
    pin, pool, cid = tb.pinned_base()
    stores = source_stores(provisional) if stores is None else stores
    sources = plan_sources(stores)
    d = batchplan.production_dir(cid, stage4.PLAN_POOLS, sources=sources, version=PLAN_VERSION)
    if not (d / "plan.json").exists():
        raise NotReady(f"no batch plan version 3 at {d} yet (train_keab.py plan{' --provisional' if provisional else ''})")
    plan = batchplan.Plan.load(d)
    tb.check_plan(plan, pin, PLAN_VERSION)
    if plan.source_configs() != [s.config() for s in sources]:
        raise RuntimeError("the batch plan was drawn from other sources")
    return pin, pool, cid, plan, sources


def plan_counts(plan, stores) -> dict:
    """The build's report: planned batches, per-source draws, distinct clips, hours and shares (keab-prime plan "Batch
    plan version 3": "The counts are reported at build")."""
    real_h = plan.meta["stats"]["real_seconds"] / 3600 - sum(m["seconds"] for m in plan.meta["sources"]) / 3600
    out = {"plan_id": plan.meta["plan_id"], "plan_sha256": plan.sha256, "n_batches": plan.n_batches,
           "n_pools": plan.n_pools, "n_entries": plan.meta["n_entries"], "n_extra": plan.meta["n_extra"],
           "real_hours": real_h, "padding": plan.meta["stats"]["padding"], "by_key_padding":
           plan.meta["stats"]["by_key"]["padding"], "provisional": is_provisional(stores), "sources": {}}
    for m, (cfg, pool) in zip(plan.meta["sources"], stores):
        out["sources"][cfg[0]] = {"share_rule": m["share_rule"], "share_value": m["share_value"],
                                  "share_of_real": m["share_of_real"], "clips": m["n_clips"],
                                  "store_hours": sum(c["duration"] for c in pool.clips) / 3600,
                                  "draws": m["draws"], "distinct": m["distinct"], "hours": m["seconds"] / 3600,
                                  "draws_per_clip": m.get("draws_per_clip")}
        if cfg[0] == "synthetic":
            out["sources"][cfg[0]]["parts"] = [{"run": n, "clips": c, "hours": h} for n, c, h, _ in pool.parts]
    return out


def build_plan(provisional: bool = False) -> int:
    """Build batch plan version 3 of pools 0-781 (CPU; keab-prime plan "Batch plan version 3") and report its counts."""
    from training import batchplan
    llayout.require_mount()
    pin, pool, cid = tb.pinned_base()
    v1 = tb.v1_plan()[3]
    stores = source_stores(provisional)
    if is_provisional(stores) and not provisional:
        raise RuntimeError("a production plan needs every synthetic run")
    sources = plan_sources(stores)
    plan = batchplan.build(pool, batchplan.production_dir(cid, stage4.PLAN_POOLS, sources=sources,
                                                          version=PLAN_VERSION),
                           stage4.PLAN_POOLS, sources=sources, config_id=cid, version=PLAN_VERSION)
    if plan.meta["base_plan_sha256"] != v1.sha256:
        raise RuntimeError("the version 3 plan is not drawn on the version 1 plan of the same journal")
    counts = plan_counts(plan, stores)
    write_json(plan.dir.parent / "reports" / f"{plan.meta['plan_id']}.json", counts | {"utc": train.utc()})
    log(json.dumps(counts, default=str))
    return 0


def reading_gate(plan) -> None:
    """A run reads materialized audio only after stage 3's replay throughput probe of the version 1 plan and this
    stage's read probe of the version 3 plan (all three stores) met the rate rule in every region."""
    v1 = tb.v1_plan()[3]
    tb._probe_passed(tb.replay_probe_path(v1), v1, "the replay throughput probe of the version 1 plan")
    tb._probe_passed(read_probe_path(), plan, "the read probe of batch plan version 3")


def read_probe_job(max_seconds: float, provisional: bool = False) -> int:
    """The read probe of batch plan version 3 (stage 3's rule, training/replay_probe.py): the materialized reader with
    the three extra stores (both recordings of every AMI pair) on the plan's start, middle and last pools."""
    from training import poolaudio, replay_probe
    llayout.require_mount()
    stores = source_stores(provisional)
    _, pool, cid, plan, sources = pinned_plan(stores, provisional)
    n = plan.n_pools
    regions = [(0, 3), (n // 2 - 1, n // 2 + 2), (n - 3, n)]
    res = replay_probe.run(plan, pool, poolaudio.store_dir(cid, plan.pool_batches), regions, ["materialized"],
                           max_seconds, sources=replay_sources(stores, sources))
    res["pass"] = bool(res["materialized_meets_required"])
    res["provisional"] = is_provisional(stores)
    write_json(read_probe_path() if not res["provisional"] else root() / "read-probe" / "provisional.json", res)
    log(json.dumps({k: v for k, v in res.items() if k != "regions"}, default=str))
    return 0 if res["pass"] else 1


# ----------------------------------------------------------------------------- labels

def label_key(labels_run: str, run_dir_: Path, pool_dir: Path, plan, label_set: str, cleaning, repetition,
              reject) -> tuple[dict, str]:
    """train_b0prime.label_key with the label set as a parameter (the control's B0 labels have their own index)."""
    from labeling import c2, shards
    run = c2.load_run(run_dir_)
    if c2._abs(run["pool_dir"]).resolve() != Path(pool_dir).resolve():
        raise RuntimeError(f"{run_dir_} labels {run['pool_dir']}, not {pool_dir}")
    contract = hashlib.sha256(json.dumps(c2.contract(run), sort_keys=True).encode()).hexdigest()
    listed = []
    for k in tlabels.needed_shards(pool_dir, plan.consumed_batches):
        meta = Path(run_dir_) / "shards" / f"{shards.shard_name(k)}.json"
        if not meta.exists():
            raise RuntimeError(f"no label shard meta {meta.name} (C2 has not labeled C1 shard {k} yet)")
        listed.append([k, json.loads(meta.read_text())["sha256"]])
    if cleaning is not None:
        cleaning.bind_run(contract)
    if repetition is not None:
        repetition.bind_run(contract)
    return label_index.key_of({
        "labels_run": labels_run, "run_dir": c2._rel(Path(run_dir_)), "pool_dir": c2._rel(Path(pool_dir)),
        "label_set": label_set, "through_batch": int(plan.consumed_batches), "contract_sha256": contract,
        "shards": listed, "cleaning": None if cleaning is None else cleaning.identity(),
        "repetition": None if repetition is None else repetition.rule(),
        "reject": None if reject is None else dict(reject.identity), "real_plan_sha256": tb.real_plan_sha256(plan),
        "code": label_index.code_sha256()})


def open_labels(labels_run: str, pool_dir: Path, plan, label_set: str, cleaning_version: int = CLEANING_VERSION,
                repetition: bool = True, extra=None, run_dir_: Path | None = None,
                reject: label_sources.RejectList | None = None,
                cleaning_versions: tuple[int, ...] = CLEANING_VERSIONS) -> label_sources.Labels:
    """An arm's labels: its label set of the label store (through the plan's consumed batches, cleaning version 3,
    repetition rejection, opened through its persisted index), the read-time reject list, and the extra providers;
    `extra` is a dict or a function of the arm's StoreLabels (the AMI pairs read their IHM labels through it)."""
    run_dir_ = run_dir_ or tlabels.production_run_dir(labels_run)
    cleaning = label_sources.cleaning_spec(labels_run, cleaning_version, cleaning_versions)
    rep = trep.production(labels_run) if repetition else None
    key, sha = label_key(labels_run, run_dir_, pool_dir, plan, label_set, cleaning, rep, reject)
    directory = label_index.index_dir(label_index_root(), sha)
    if not (directory / "index.json").exists():
        t0 = time.perf_counter()
        store = tlabels.LabelStore.open(run_dir_, pool_dir, label_set=label_set, through_batch=plan.consumed_batches,
                                        cleaning=cleaning, repetition=rep)
        label_index.write(directory, store, key, sha)
        log(f"label index {directory.name} ({label_set}): written in {time.perf_counter() - t0:.0f} s "
            f"({len(store)} records)")
        del store
    real = label_sources.StoreLabels(label_index.load(directory, key, sha), None, reject)
    return label_sources.Labels(real, extra(real) if callable(extra) else extra)


def arm_labels(arm: str, labels_run: str, pool, plan, stores, sources, cleaning_version: int = CLEANING_VERSION):
    return open_labels(labels_run, pool, plan, stage4.LABEL_SETS[arm], cleaning_version,
                       extra=lambda real: label_providers(stores, sources, real, labels_run, cleaning_version),
                       reject=tb.reject_list())


def labels_job(labels_run: str, provisional: bool = False) -> int:
    """Both arms' label indexes on batch plan version 3 (CPU; the first open of a label identity writes its index, so
    the probe and the runs only memory-map it), with their identities and counts."""
    llayout.require_mount()
    stores = source_stores(provisional)
    _, pool, cid, plan, sources = pinned_plan(stores, provisional)
    out = {}
    for arm in stage4.ARMS:
        t0 = time.perf_counter()
        lab = arm_labels(arm, labels_run, pool, plan, stores, sources)
        out[arm] = {"label_set": lab.label_set, "sha256": lab.identity["sha256"], "kept": lab.identity["kept"],
                    "records": lab.identity["records"], "cleaning": lab.identity.get("cleaning"),
                    "reject_counts": lab.real.reject_counts(), "open_s": time.perf_counter() - t0,
                    "index": str(lab.real.store.dir.relative_to(root()))}
        log(f"{arm}: {json.dumps(out[arm], default=str)}")
    write_json(root() / "label-index" / "arms.json", out | {"plan_sha256": plan.sha256, "utc": train.utc()})
    return 0


def offline_data(run: stage4.Run, labels_run: str, peak: float, provisional: bool = False) -> tb.Data:
    """The production Data of a run (module docstring, item 3)."""
    from training import poolaudio
    stores = source_stores(provisional)
    _, pool, cid, plan, sources = pinned_plan(stores, provisional)
    if not provisional:
        reading_gate(plan)
    labels = arm_labels(run.arm, labels_run, pool, plan, stores, sources)
    replay = treplay.PlannedReplay(plan, pool, audio="materialized", store=poolaudio.store_dir(cid, plan.pool_batches),
                                   sources=replay_sources(stores, sources))
    return tb.data_for(run, plan, replay, labels, peak)


# ----------------------------------------------------------------------------- the student and the teacher

def mb1_checkpoint() -> Path:
    import relabel_paths
    return relabel_paths.PARAKEET_BINARY / "runs" / "main-binB-m1-lr2e-4" / "ckpt" / "latest.pt"


def b0p_nemo() -> Path:
    return llayout.relabel_root() / "b0prime" / "runs" / "b100" / "model.nemo"


def load_mb1_state(path: Path | None = None) -> tuple[dict, dict]:
    """(student state_dict, record) of MB1's latent training checkpoint: SHA-256 MB1_SHA256 (refused before
    unpickling otherwise), complete, train.py's format, step MB1_STEP, weight fraction 1, quantizer binary with ramp
    MB1_RAMP (binary_train.read_init_state's checks). Memory-mapped: its optimizer state is never read."""
    path = Path(path or mb1_checkpoint())
    digest = cached_sha256(path)
    if digest != MB1_SHA256:
        raise RuntimeError(f"{path}: SHA-256 {digest} is not MB1's {MB1_SHA256}")
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    args = payload.get("args") or {}
    facts = {"path": str(path), "sha256": digest, "step": payload.get("step"),
             "weight_fraction": payload.get("weight_fraction"), "quantizer": args.get("quantizer"),
             "ramp_kind": args.get("ramp_kind"), "run_name": args.get("run_name")}
    if not (payload.get("complete") and payload.get("format") == train.CKPT_FORMAT) or facts["step"] != MB1_STEP or \
            facts["weight_fraction"] != 1.0 or facts["quantizer"] != "binary" or facts["ramp_kind"] != MB1_RAMP:
        raise RuntimeError(f"{path} is not MB1's final binary checkpoint: {facts}")
    return payload["student"], facts


def init_record() -> dict:
    return {"model": "MB1", "path": str(mb1_checkpoint()), "sha256": MB1_SHA256, "step": MB1_STEP,
            "quantizer": "binary", "ramp_kind": MB1_RAMP, "weight_fraction": 1.0}


def load_teacher_model(arm: str, device: str = "cpu"):
    """(the arm's teacher in FP32, its record): B0' (b0prime b100's model.nemo, SHA-256 checked) or B0 (the pinned
    .nemo, train.verify_base)."""
    teacher = stage4.TEACHERS[arm]
    if teacher == "B0p":
        path = b0p_nemo()
        digest = cached_sha256(path)
        if digest != B0P_SHA256:
            raise RuntimeError(f"{path}: SHA-256 {digest} is not B0''s {B0P_SHA256}")
        model = train.load_nemo(path, device)
        return model, {"name": "B0p", "path": str(path), "sha256": digest}
    base = train.verify_base()
    return train.load_pretrained(device), {"name": "B0", "path": str(paths.MODEL_FILE), "sha256": base["base_sha256"]}


def teacher_record(arm: str) -> dict:
    if stage4.TEACHERS[arm] == "B0p":
        return {"name": "B0p", "path": str(b0p_nemo()), "sha256": B0P_SHA256}
    return {"name": "B0", "path": str(paths.MODEL_FILE), "sha256": train.verify_base()["base_sha256"]}


DECODER_PREFIXES = ("decoder.", "joint.")


def decoder_state(model) -> dict[str, torch.Tensor]:
    """The prediction and joint networks' tensors of a model's state_dict (the teacher package's decoder)."""
    return {k: v for k, v in model.state_dict().items() if k.startswith(DECODER_PREFIXES)}


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        h.update(state[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def install_decoder(student, state: dict[str, torch.Tensor]) -> str:
    """Copy the teacher's prediction and joint tensors into the student, bit for bit; returns their SHA-256."""
    own = decoder_state(student)
    if set(own) != set(state) or any(own[k].shape != state[k].shape or own[k].dtype != state[k].dtype for k in own):
        raise RuntimeError("the teacher's prediction and joint networks do not match the student's")
    for k, v in state.items():
        own[k].copy_(v.to(own[k].device))
    got = state_sha256(decoder_state(student))
    if got != state_sha256(state):
        raise RuntimeError("the student's decoder is not the teacher's after the copy")
    return got


def build_student(teacher_decoder: dict, scope: str, resuming: bool, base_model=None, init_state=None,
                  expected_modules: int | None = BINARY_MODULES) -> tuple[torch.nn.Module, dict]:
    """The stage 4 student on the CPU (module docstring, item 2). base_model / init_state: the pinned B0 and MB1's
    latent state (tests pass a tiny model and its state); resuming: the checkpoint's state replaces the student's."""
    student = train.load_pretrained("cpu") if base_model is None else base_model
    if expected_modules == BINARY_MODULES:
        binary_quant.quantize_parakeet(student, MB1_RAMP)
    else:                                       # a test model: every module of the quantized pattern
        eb.binarize(student, MB1_RAMP)
    init = {"loaded": not resuming, **(init_record() if init_state is None else {"model": "fixture"})}
    if not resuming:
        state, facts = (load_mb1_state() if init_state is None else (init_state, {}))
        student.load_state_dict(state, strict=True)
        del state
        init.update(facts)
    binary_quant.set_weight_fraction(student, 1.0)
    decoder_sha = install_decoder(student, teacher_decoder)
    form = eb.apply_keab_scope(student, scope)
    params = train.configure_trainable(student, RECIPE)
    params["verified"] = train.verify_recipe(student, RECIPE, binary_quant)
    names = actquant.install(student, expected_modules)
    found = verify(student, scope, expected_modules)
    return student, {"init": init, "teacher_decoder_sha256": decoder_sha, "form": form, "trainable": params,
                     "verified": found, "binary_modules": len(names)}


def verify(model, scope: str, expected_modules: int | None = BINARY_MODULES) -> dict:
    """The stage 4 student: the whole encoder trainable and nothing else, FP32 parameters, the binary modules at
    weight fraction 1, the scope's 8-bit form, one activation quantizer per binary module and none elsewhere,
    SpecAugment present; RuntimeError otherwise."""
    count = lambda m, t: sum(p.numel() for p in m.parameters() if p.requires_grad == t)  # noqa: E731
    problems = []
    names = binary_quant.quantized_module_names(model)
    found = {"trainable": count(model, True), "frozen": count(model, False),
             "encoder_trainable": count(model.encoder, True), "encoder_frozen": count(model.encoder, False),
             "decoder_trainable": count(model.decoder, True), "joint_trainable": count(model.joint, True),
             "fp32": all(p.dtype == torch.float32 for p in model.parameters()),
             "binary_modules": len(names), "spec_augmentation": model.spec_augmentation is not None}
    try:
        found["form"] = eb.keab_form(model)
        found["actquant"] = actquant.verify(model, expected_modules)
    except ValueError as error:
        problems.append(str(error))
    if found.get("form", {}).get("scope") not in (None, scope):
        problems.append(f"8-bit form {found['form']['scope']}, not {scope}")
    if found["encoder_frozen"] or found["decoder_trainable"] or found["joint_trainable"]:
        problems.append("not exactly the encoder trainable")
    if found["trainable"] != found["encoder_trainable"]:
        problems.append("parameters outside the encoder are trainable")
    if not found["fp32"]:
        problems.append("parameters are not FP32")
    if expected_modules is not None and len(names) != expected_modules:
        problems.append(f"{len(names)} binary modules, not {expected_modules}")
    if any(model.get_submodule(n).fraction != 1.0 for n in names):
        problems.append("a binary module below weight fraction 1")
    if not found["spec_augmentation"]:
        problems.append("no SpecAugment")
    if problems:
        raise RuntimeError(f"stage 4 student check failed: {'; '.join(problems)}")
    return found


@torch.no_grad()
def teacher_encode(teacher, audio: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The teacher's encoder output, FP32 [B, D, T'], and lengths: parakeet-ternary teacher.teacher_label_batch's path
    at precision "bf16" without its decoding (features in strict FP32 with dither 0 and pad_to 0, the encoder under
    BF16 autocast, the output cast to FP32)."""
    device = audio.device
    with evaluate.strict_fp32(device), evaluate.inference_settings(teacher):
        features, feature_len = teacher.preprocessor(input_signal=audio.float(), length=lengths)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            encoded, encoded_len = teacher.encoder(audio_signal=features, length=feature_len)
    return encoded.float(), encoded_len


def teacher_inputs(batch: dict, rows: list[int], audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """(teacher audio [len(rows), L], lengths, counts) for the kept rows `rows` of a collated batch, whose student
    audio (on the device, the kept rows, the batch's padded length L) is `audio`: a row's own audio, except an AMI
    pair's, which is its IHM recording (batch["teacher_audio"]), zero-padded to L, with equal encoder frames."""
    from training import ami_store
    over = batch.get("teacher_audio") or {}
    lengths = batch["audio_len"][rows].to(audio.device)
    counts = {"rows": len(rows), "teacher_audio_rows": 0}
    if not any(r in over for r in rows):
        return audio, lengths, counts
    out = audio.clone()
    student_len = batch["audio_len"].tolist()
    for j, r in enumerate(rows):
        if r not in over:
            continue
        a = torch.from_numpy(np.ascontiguousarray(over[r], dtype=np.float32))
        n = len(a)
        if n > out.shape[1]:
            raise RuntimeError(f"row {batch['ids'][r]}: its teacher recording ({n} samples) is longer than the batch")
        if ami_store.encoder_frames(n) != ami_store.encoder_frames(student_len[r]):
            raise RuntimeError(f"row {batch['ids'][r]}: teacher and student encoder frame counts differ")
        out[j].zero_()
        out[j, :n] = a.to(out.device)
        lengths[j] = n
        counts["teacher_audio_rows"] += 1
    return out, lengths, counts


TEACHER_ROWS_KEY = "teacher_audio_rows"


def label_batch(ctx, batch: dict) -> dict | None:
    """train.label_batch for stage 4 (module docstring, item 4): stage 3's label_batch, then the teacher's encoding."""
    out = tseam.label_batch(ctx, batch)
    if out is None:
        return None
    rows = [i for i, t in enumerate(batch["labels"]) if t is not None]
    t_audio, t_len, counts = teacher_inputs(batch, rows, out["audio"])
    enc, enc_len = teacher_encode(ctx.teacher, t_audio, t_len)
    st = ctx.state
    st.drop_counts[TEACHER_ROWS_KEY] = st.drop_counts.get(TEACHER_ROWS_KEY, 0) + counts["teacher_audio_rows"]
    out.update(teacher_enc=enc, teacher_len=enc_len)
    return out


class Context(train.Context):
    """train.Context with the frozen teacher, MB1's binary modules at weight fraction 1 from the first step (ramp 0,
    as parakeet-binary's BinaryContext for ramp "none"), the run spec and a budget timer slot."""

    def __init__(self, args, run: Path, student, teacher, data, evaluator, device: str, spec: stage4.Run):
        super().__init__(args, run, student, teacher, data, evaluator, binary_quant, device, label_fn=td._no_teacher)
        self.ramp = 0
        self.spec = spec
        self.timer: threading.Timer | None = None
        self.state.drop_counts[tseam.NONFINITE_KEY] = 0
        self.state.drop_counts[TEACHER_ROWS_KEY] = 0


# ----------------------------------------------------------------------------- runs: selection, branches, identity

def run_done(name: str) -> bool:
    return (run_dir(name) / "summary.json").exists()


def recorded_args(name: str) -> dict | None:
    path = run_dir(name) / "config.json"
    return json.loads(path.read_text()).get("args") if run_done(name) and path.exists() else None


def _canon(obj):
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


def matched_identity(args: dict) -> dict:
    return _canon({k: args.get(k) for k in MATCHED_KEYS})


def arm_identity(args: dict) -> dict:
    return _canon({k: args.get(k) for k in ARM_KEYS})


def shared_identity(names) -> dict:
    """The matched identity (MATCHED_KEYS) of the finished runs `names`, and each arm's teacher package (ARM_KEYS)
    shared by its runs (RuntimeError if either differs)."""
    recs = {n: recorded_args(n) or {} for n in names}
    matched = {json.dumps(matched_identity(a), sort_keys=True) for a in recs.values()}
    if len(matched) != 1:
        diff = sorted(k for k in MATCHED_KEYS if len({json.dumps(_canon(a.get(k)), sort_keys=True)
                                                       for a in recs.values()}) > 1)
        raise RuntimeError(f"runs {list(names)} differ in {diff}; they cannot be compared")
    arms = {}
    for n, a in recs.items():
        ident = arm_identity(a)
        arm = RUNS[n].arm
        if arms.setdefault(arm, ident) != ident:
            raise RuntimeError(f"the {arm} arm's runs differ in their teacher package")
    return {"matched": json.loads(matched.pop()), "arms": arms}


def run_scores(name: str) -> dict:
    path = run_dir(name) / "scores.json"
    if not path.exists():
        raise NotReady(f"{name} has no scores yet")
    return json.loads(path.read_text())


def frozen_selection() -> dict:
    """stage4.select_lr over the six pilots' PyTorch-mode H4, frozen at first use in selection.json."""
    pending = [n for n in stage4.PILOT if not run_done(n)]
    if pending:
        raise NotReady(f"the learning-rate selection needs the six pilot runs ({', '.join(pending)} pending)")
    sel = stage4.select_lr({n: run_scores(n)["h4"] for n in stage4.PILOT})
    sel["summaries_sha256"] = {n: _sha256_file(run_dir(n) / "summary.json") for n in stage4.PILOT}
    sel["identity"] = shared_identity(stage4.PILOT)
    return tb.frozen_record(selection_path(), sel, "selection")


def peak_of(run: stage4.Run) -> tuple[float, dict | None]:
    if run.lr is not None:
        return run.lr, None
    sel = frozen_selection()
    return float(sel["selected"]), {k: sel[k] for k in ("selected", "selected_name", "mean_h4", "h4", "rule",
                                                        "summaries_sha256", "identity")}


def grad_checkpointing(plan_sha256: str, labels_sha256: str) -> bool:
    """The memory probe's decision (keab-prime plan "Resources"), which must have been made on this plan and with
    this arm's labels among the ranked ones; NotReady before the probe exists."""
    if not probe_path().exists():
        raise NotReady(f"no memory probe result at {probe_path()} (train_keab.py probe)")
    body = json.loads(probe_path().read_text())
    decision = body.get("decision") or {}
    if decision.get("fits") is not True:
        raise RuntimeError(f"the memory probe found no configuration that fits: {decision}")
    if body.get("plan_sha256") != plan_sha256:
        raise RuntimeError(f"{probe_path()} probed another plan ({body.get('plan_sha256')}, provisional "
                           f"{body.get('provisional')}); run the probe again on this one")
    if labels_sha256 not in (body.get("labels_sha256") or {}).values():
        raise RuntimeError(f"{probe_path()} ranked other labels than this arm's {labels_sha256}; run the probe again")
    return bool(decision["grad_checkpointing"])


TRUNK_CHECK = STAGE4_KEYS + ("label_set", "keab_arm", "loss_seam")


def branch_of(run: stage4.Run, plan, args_like: dict) -> dict | None:
    """A branch's record of its trunk's branch point (tb.read_branch_point), after checking that the trunk trained
    with this run's labels, plan, peak and settings: stage 3's TRUNK_KEYS from the marker and this stage's from the
    trunk's config.json."""
    if run.kind != "branch":
        return None
    trunk = RUNS[run.trunk]
    m = tb.read_branch_point(run_dir(trunk.name), run.branch_pool)
    if m["planned_batch"] != plan.pool_start(run.branch_pool) or m["trunk"]["run_name"] != trunk.name:
        raise RuntimeError(f"{run.name}: the branch point is not {trunk.name}'s at pool {run.branch_pool}")
    diff = [k for k in tb.TRUNK_KEYS if json.dumps(m["trunk"].get(k), sort_keys=True, default=str)
            != json.dumps(args_like.get(k), sort_keys=True, default=str)]
    cfg = run_dir(trunk.name) / "config.json"
    rec = json.loads(cfg.read_text())["args"] if cfg.exists() else {}
    diff += [k for k in TRUNK_CHECK if k != "smoke_of" and _canon(rec.get(k)) != _canon(args_like.get(k))]
    trunk_s, own = m["trunk"]["schedule"], args_like["schedule"]
    if (trunk_s.get("kind"), trunk_s.get("peak"), trunk_s.get("warmup"), trunk_s.get("cooldown_start")) != \
            ("wsd", own.get("peak"), own.get("warmup"), None):
        diff.append("schedule (the trunk's warmup and peak)")
    if diff:
        raise RuntimeError(f"{run.name}: {trunk.name} trained with other {diff} than this branch")
    return {"trunk": trunk.name, "pool": run.branch_pool, "planned_batch": m["planned_batch"], "sha256": m["sha256"],
            "position": m["position"], "step": m["step"]}


def train_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="train_keab.py train")
    p.add_argument("--run", choices=list(RUNS), required=True)
    p.add_argument("--labels-run", required=True, help="the C2 run name (labels/<T*>/<run>)")
    p.add_argument("--budget-minutes", type=float, default=CHUNK_MINUTES)
    p.add_argument("--deadline", type=float, help="epoch seconds: stop (checkpointed) by then (chunk jobs)")
    p.add_argument("--ckpt-minutes", type=float, default=30.0)
    p.add_argument("--overwrite", action="store_true", help="discard this run's outputs and checkpoints")
    p.add_argument("--smoke", help="a smoke run: its name under $RELABEL_ROOT/keab-prime/smoke/ (with --steps)")
    p.add_argument("--steps", type=int, help="smoke: optimizer steps (the first planned batches of the run)")
    p.add_argument("--scope", choices=eb.SCOPES, help="smoke: the 8-bit scope (production runs read step 0's)")
    p.add_argument("--provisional", action="store_true", help="smoke: the provisional plan (before run pairs-c)")
    p.add_argument("--grad-checkpointing", choices=("on", "off"), help="smoke: override the probe's decision")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """train.parse_args for a stage 4 run (module docstring, item 1)."""
    a = train_parser().parse_args(argv)
    run = RUNS[a.run]
    smoke = a.smoke is not None
    if smoke != (a.steps is not None) or (not smoke and (a.scope or a.provisional or a.grad_checkpointing)):
        raise SystemExit("--smoke goes with --steps; --scope, --provisional and --grad-checkpointing are smoke options")
    scope = a.scope if smoke and a.scope else step0_scope()
    peak, selection = peak_of(run) if not smoke else (run.lr or stage4.PILOT_LRS[1], None)
    data = offline_data(run, a.labels_run, peak, a.provisional)
    if smoke:
        data = smoke_data(data, a.steps)
        ckpt = {"on": True, "off": False}.get(a.grad_checkpointing)
        if ckpt is None:
            ckpt = bool(((json.loads(probe_path().read_text()).get("decision") or {}).get("grad_checkpointing", True))
                        if probe_path().exists() else True)
    else:
        ckpt = grad_checkpointing(data.replay.identity["plan_sha256"], data.labels.identity["sha256"])
    return namespace(a, run, data, peak, selection, ckpt, train.verify_base()["base_sha256"], scope,
                     {"steps": a.steps, "provisional": a.provisional, "name": a.smoke} if smoke else None)


def smoke_data(data: tb.Data, steps: int) -> tb.Data:
    """The run's Data stopped after `steps` optimizer steps: its first planned batches (a smoke)."""
    stop = min(steps, data.stop)
    while True:
        acc = treplay.plan(data.replay, data.labels, stop)
        if acc["steps"] >= steps or stop >= data.stop:
            break
        stop = min(data.stop, stop + steps - acc["steps"])
    return tb.Data(data.replay, data.labels, data.schedule, stop, {})


def namespace(a, run: stage4.Run, data: tb.Data, peak: float, selection: dict | None, ckpt: bool, base_sha256: str,
              scope: str, smoke_of: dict | None = None) -> argparse.Namespace:
    """The run's train.py arguments (module docstring, item 1)."""
    _ACTIVE.update(data=data, run=run, scope=scope)
    steps = data.plan["steps"]
    ident = data.labels.identity
    args = argparse.Namespace(
        arm=ARM, recipe=RECIPE_NAME, lr=peak, run_name=run.name, max_steps=steps, train="relabel-stage4",
        stream="c1-batch-plan", batch_seconds=BATCH_SECONDS, grad_checkpointing=ckpt,
        eval_every=max(1, steps // 10) if run.kind == "trunk" and smoke_of is None else steps, select="final",
        dev_limit=None, eval_batch_size=32, ckpt_minutes=a.ckpt_minutes, seed=run.seed, probe_steps=None,
        no_powerlog=True, overwrite=a.overwrite, smoke=smoke_of is not None, spec=run.to_dict(),
        label_set=run.label_set, labels_run=a.labels_run, labels_sha256=ident["sha256"],
        cleaning_version=CLEANING_VERSION, cleaning=ident.get("cleaning"), repetition=ident.get("repetition"),
        label_transform=ident.get("transform"), label_reject=ident.get("reject"),
        extra_sources=ident.get("extra") or None, plan_sha256=data.replay.identity["plan_sha256"],
        replay_sha256=data.replay.identity["sha256"], stop_batch=data.stop,
        schedule=stage4.schedule_dict(data.schedule), base_sha256=base_sha256, lr_selection=selection,
        branch_of=None, budget_minutes=a.budget_minutes, deadline=a.deadline, loss_seam=tdt_fused.settings(),
        keab_arm=run.arm, scope=scope, teacher=teacher_record(run.arm), init=init_record(),
        actquant=actquant.identity(), eightbit=scope_record(scope), teacher_input=dict(TEACHER_INPUT),
        smoke_of=smoke_of)
    if selection is not None:
        if matched_identity(vars(args)) != selection["identity"]["matched"]:
            raise RuntimeError(f"{run.name}: its plan, sources or settings differ from the pilots' that chose its "
                               "learning rate")
        if arm_identity(vars(args)) != selection["identity"]["arms"].get(run.arm):
            raise RuntimeError(f"{run.name}: its teacher package differs from its arm's pilots'")
    args.branch_of = branch_of(run, data.replay.plan, vars(args))
    log(f"{run.name} ({run.arm}): plan {data.replay.identity['plan_id']}, stop at planned batch {data.stop}, {steps} "
        f"steps ({len(data.plan['empty_batches'])} fully rejected batches), peak {peak:g}, scope {scope}, schedule "
        f"{args.schedule}")
    return args


def build_context(args, run: Path, device: str = "cuda"):
    """train.build_context for a stage 4 run (module docstring, item 2)."""
    data, spec = _ACTIVE["data"], _ACTIVE["run"]
    if data is None or spec is None or spec.name != args.run_name:
        raise RuntimeError("parse_args did not prepare this run's data")
    latest, previous, _ = train.ckpt_paths(run)
    resuming = latest.exists() or previous.exists()
    teacher, teacher_rec = load_teacher_model(spec.arm, "cpu")
    if teacher_rec != args.teacher:
        raise RuntimeError(f"the teacher is {teacher_rec}, the run records {args.teacher}")
    fresh_branch = spec.kind == "branch" and not resuming
    student, info = build_student(decoder_state(teacher), args.scope, resuming or fresh_branch)
    train.seed_everything(args.seed, student)
    student.to(device)
    if args.grad_checkpointing:
        train.enable_grad_checkpointing(student.encoder)
    teacher = train.load_teacher(teacher.to(device))
    evaluator = train.DevEvaluator(args.dev_limit, args.eval_batch_size, device) \
        if spec.kind == "trunk" and not args.smoke else None
    ctx = Context(args, run, student, teacher, data, evaluator, device, spec)
    _ACTIVE["ctx"] = ctx
    if fresh_branch:
        tb.start_branch(ctx, args.branch_of, run_dir(spec.trunk))
    if data.hooks:
        data.on_batch = lambda index: tb.save_branch_point(ctx, data.hooks[index], index)
    td.start_budget(ctx, args.budget_minutes, args.deadline)
    info = {"base": {"base_sha256": args.base_sha256}, "quantized_modules": binary_quant.quantized_module_names(student),
            "trainable": info["trainable"], "parameter_accounting": binary_quant.parameter_accounting(student),
            "teacher": {"labels": f"offline label store ({args.label_set}: {data.labels.identity['column']}) under "
                                  f"{data.labels.identity['mask']}", "encoder_target": args.teacher,
                        "decoder": {"from": args.teacher["name"], "sha256_before_8bit": info["teacher_decoder_sha256"]},
                        "input": TEACHER_INPUT, "cleaning": data.labels.identity.get("cleaning"),
                        "extra": data.labels.identity.get("extra")},
            "bucketing": {"source": "batch plan version 3 (keab-prime plan)", "plan": data.replay.identity,
                          "stop_batch": data.stop},
            "keab": {"run": spec.to_dict(), "labels": data.labels.identity, "replay": data.replay.identity,
                     "reject_counts": data.labels.real.reject_counts(),
                     "plan": {k: v for k, v in data.plan.items() if k != "empty_batches"},
                     "empty_batches": data.plan["empty_batches"], "schedule": stage4.schedule_dict(data.schedule),
                     "branch_points": {p: b for b, p in data.hooks.items()}, "branch_of": args.branch_of,
                     "init": info["init"], "form": info["form"], "verified": info["verified"], "recipe": RECIPE,
                     "precision": "FP32 latent parameters and AdamW state; BF16 autocast (train.compute_losses)",
                     "loss_seam": args.loss_seam}}
    return ctx, info


def run_eval(ctx) -> None:
    """The trunks' dev-subset evaluations (train.run_eval, informational); none for the other runs."""
    if ctx.evaluator is not None:
        tb._TRAIN["run_eval"](ctx)
        return
    train.log(f"step {ctx.state.step}: no dev-subset evaluation ({ctx.args.run_name} is scored after training)")


def save_model(ctx) -> dict:
    """model.pt (the latent state_dict, the training form's) and model.json; a branch or a smoke also writes its
    8-bit bundle and scoring view (export/bundle, export/view; training/keab_export.py) from the CUDA model, whose
    binary scales are the training forward's."""
    from training import keab_export as ke
    args, run, spec = ctx.args, ctx.run, ctx.spec
    model = ctx.student
    model.eval()
    path = run / "model.pt"
    tmp = path.with_name(f"model.tmp{os.getpid()}.pt")
    torch.save({"student": model.state_dict(), "scope": args.scope, "arm": args.keab_arm, "run": spec.name,
                "step": ctx.state.step, "format": "keab-prime latent v1"}, tmp)
    os.replace(tmp, path)
    train.drop_page_cache(path)
    manifest = {"model": "model.pt", "sha256": _sha256_file(path), "bytes": path.stat().st_size, "run": spec.name,
                "arm": args.keab_arm, "keab_prime": spec.name == stage4.KEAB_PRIME, "scope": args.scope,
                "teacher": args.teacher, "init": args.init, "labels": {
                    "labels_run": args.labels_run, "label_set": args.label_set, "labels_sha256": args.labels_sha256,
                    "label_manifest": tb.label_manifest(args.labels_run), "cleaning_version": args.cleaning_version,
                    "cleaning": args.cleaning, "repetition": args.repetition, "extra_sources": args.extra_sources},
                "plan_sha256": args.plan_sha256, "schedule": args.schedule, "lr": args.lr, "seed": args.seed,
                "step": ctx.state.step, "git_head": train.git_head(),
                "training_source_hashes": train.training_source_hashes(), "utc": train.utc()}
    if spec.kind == "branch" or args.smoke:
        extra = {"stage": "keab-prime", "run": spec.name, "arm": args.keab_arm, "model_sha256": manifest["sha256"],
                 "labels_sha256": args.labels_sha256, "teacher": args.teacher}
        bundle = ke.write_bundle(model, run / "export" / "bundle", extra=extra, root=root())
        view = ke.derive_view(run / "export" / "bundle", run / "export" / "view", root=root(), extra=extra)
        manifest["export"] = {"bundle_sha256": bundle["sha256"], "bundle_tensor_mb": ke.tensor_megabytes(bundle),
                              "bundle_parts": bundle["bytes"].get("parts"), "view_sha256": view["sha256"]}
    write_json(run / "model.json", manifest)
    return manifest


def finish(ctx, config_sha: str) -> dict:
    """train.finish for a stage 4 run (module docstring, item 5)."""
    st, args, run, spec = ctx.state, ctx.args, ctx.run, ctx.spec
    if ctx.timer is not None:
        ctx.timer.cancel()
    if st.step != args.max_steps:
        raise RuntimeError(f"final model not ready: step {st.step}/{args.max_steps}")
    common = {"select": "final", "selected_step": st.step, "ramp_steps": 0, "warmup_steps": int(ctx.stream.schedule.warmup)}
    counts = {"consumed_batches": st.batches_read, "fully_rejected_batches": st.skipped_batches,
              "nonfinite_skipped_steps": st.drop_counts.get(tseam.NONFINITE_KEY, 0),
              "teacher_audio_rows": st.drop_counts.get(TEACHER_ROWS_KEY, 0)}
    if not spec.scored and not args.smoke:
        final = [p for p in spec.branch_points if ctx.stream.replay.plan.pool_start(p) == ctx.stream.stop]
        points = {p: tb.final_branch_point(ctx, p, ctx.stream.stop) for p in final}
        return common | {"dev_subset_mean_wer": math.nan, "dev_subset_meaning": "a trunk is never scored",
                         "dev_wer": {}, "dev_mean_wer": math.nan, "scored_on": "not scored (trunk)",
                         "keab": {"run": spec.to_dict(), "scored": False, "final_branch_points": points, **counts}}
    manifest = save_model(ctx)
    if args.smoke:
        return common | {"dev_subset_mean_wer": math.nan, "dev_subset_meaning": "a smoke is not scored",
                         "dev_wer": {}, "dev_mean_wer": math.nan, "scored_on": "not scored (smoke)",
                         "keab": {"run": spec.to_dict(), "scored": False, "model": manifest, **counts}}
    ctx.student.eval()
    scores = score_torch(ctx.student, scores_dir(spec.name, "torch"), model_source(manifest), ctx.device, spec.name,
                         ("dev",))
    write_json(run / "scores.json", scores)
    full = scores["dev_wer"]
    return common | {"dev_subset_mean_wer": scores["h4"] / 100, "dev_subset_meaning": "H4 on full stage 1 dev "
                     "(fraction), PyTorch fake-quant mode", "dev_wer": full, "dev_mean_wer": sum(full.values()) / len(full),
                     "dev_utterances": scores["dev_utterances"], "scored_on": "the trained model in PyTorch fake-quant "
                     "mode (training-equivalent)", "keab": {"run": spec.to_dict(), "scored": True, "h4": scores["h4"],
                                                          "model": manifest, **counts}}


def model_source(manifest: dict) -> dict:
    return {"kind": "keab-prime latent", "path": manifest["model"], "sha256": manifest["sha256"],
            "description": f"stage 4 {manifest['run']} ({manifest['arm']}), scope {manifest['scope']}"}


@contextlib.contextmanager
def seam(deadline: float | None = None):
    """Install the stage 4 replacements in train.py and paths.py, then the fused loss seam; restore them on exit."""
    items = tseam.replacements() + [
        (train, "parse_args", parse_args), (train, "build_context", build_context), (train, "finish", finish),
        (train, "run_eval", run_eval), (train, "train_step", tseam.guarded_train_step),
        (train, "label_batch", label_batch),
        (train, "RESUME_KEYS", tuple(train.RESUME_KEYS) + RESUME_EXTRA),
        (train, "RESUME_ALLOWED_CHANGES", tuple(train.RESUME_ALLOWED_CHANGES) + ("budget_minutes", "deadline")),
        (train, "TRAINING_SOURCES", tuple(train.TRAINING_SOURCES) + SOURCES),
        (train, "OUTPUTS", tuple(train.OUTPUTS) + EXTRA_OUTPUTS), (paths, "RUNS", runs_root()),
        (paths, "gpu_lock", td.bounded_lock(deadline))]
    with tseam.installed(items), tdt_fused.installed():
        yield


def runs_root() -> Path:
    return run_dir("x").parent


def run_mismatch(name: str, labels_run: str) -> str | None:
    rec = recorded_args(name)
    if rec is None:
        return None
    if (rec.get("labels_run"), rec.get("cleaning_version")) != (labels_run, CLEANING_VERSION):
        return (f"{run_dir(name)} finished with labels run {rec.get('labels_run')} and cleaning version "
                f"{rec.get('cleaning_version')}, not {labels_run} and {CLEANING_VERSION}")
    return None


def run_train(argv: list[str]) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--smoke")
    known, _ = pre.parse_known_args(argv)
    _ACTIVE.update(data=None, ctx=None, run=None, smoke=known.smoke)
    a = train_parser().parse_args(argv)
    llayout.require_mount()
    if run_done(a.run) and not a.overwrite:
        mismatch = run_mismatch(a.run, a.labels_run)
        if mismatch:
            log(f"refusing: {mismatch}")
            return train.EXIT_RESUME_MISMATCH
        log(f"{run_dir(a.run)} is complete")
        return 0
    import score_teacher
    vram_lease()
    try:
        with seam(a.deadline):
            try:
                train.main(argv)
            except tb.NotReady as wait:
                log(f"{a.run}: not started: {wait}")
                return EXIT_NOT_READY
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
    finally:
        _ACTIVE["smoke"] = None
    return 0


# ----------------------------------------------------------------------------- scoring

SET_GROUPS = ("dev", "domain", "stem", "cv", "ami-sdm")
DOMAIN_SETS = ("dev-medical", "dev-technical")
STEM_SET = tb.STEM_SET
TERM_KIND = dict(tb.TERM_KIND)
CV_DEV_SAMPLE, CV_DEV_NAME = tb.CV_DEV_SAMPLE, tb.CV_DEV_NAME
AMI_SDM_DEV, AMI_SDM_DEV_NAME = "corpora/ami-sdm/dev/ami-sdm-dev.jsonl", "ami-sdm-dev"
EVAL_BATCH, EVAL_MAX_SECONDS = td.EVAL_BATCH, td.EVAL_MAX_SECONDS
MODE_GROUPS = {"x1": SET_GROUPS, "x3": ("dev",), "torch": ("dev",), "fp32": ("ami-sdm",)}
MB1_X1_GROUPS = ("stem", "cv", "ami-sdm")          # MB1's dev and domain x1 scores are step 0's (scores/MB1/none)
REFERENCE_MODELS = ("MB1", "B0", "B0p")
INIT_MODEL = "init-main"          # the main arm's student at step 0: MB1's encoder with B0''s decoder (informational)
SCORED_RUNS = ("b100-main", "b100-control", "b50-main")


def expand_sets(groups) -> list[tuple[str, str]]:
    """[(set or part name, its group: "dev", a domain set, "eval-stem", "cv", "ami-sdm")]."""
    import devsets
    out = []
    for g in groups:
        if g == "dev":
            out += [(n, "dev") for n in DEV_SETS]
        elif g == "domain":
            out += [(p, s) for s in DOMAIN_SETS for p in devsets.SETS[s]]
        elif g == "stem":
            out += [(p, STEM_SET) for p in devsets.SETS[STEM_SET]]
        elif g == "cv":
            out += [(CV_DEV_NAME, "cv")]
        elif g == "ami-sdm":
            out += [(AMI_SDM_DEV_NAME, "ami-sdm")]
        else:
            raise ValueError(f"set group must be one of {SET_GROUPS}, got {g!r}")
    return out


def manifest_of(name: str, group: str) -> Path:
    if group == "dev":
        return paths.MANIFESTS / f"{name}.jsonl"
    if group == "cv":
        return llayout.relabel_root() / CV_DEV_SAMPLE
    if group == "ami-sdm":
        return llayout.relabel_root() / AMI_SDM_DEV
    import score_domain
    return score_domain.part_manifests([name], None)[name][0]


def term_kind(group: str) -> str | None:
    import devsets
    if group == STEM_SET:
        return devsets.TERM_KIND[STEM_SET]
    return TERM_KIND.get(group)


def add_term_recall(result: dict, group: str) -> None:
    kind = term_kind(group)
    if kind is not None:
        import score_domain
        import terms
        result["term_recall"] = score_domain.term_recall(result["records"], terms.matcher(kind))
        result["term_kind"] = kind


def set_config(model: dict, mode: str, name: str, group: str, limit: int | None) -> dict:
    return {"model": model, "mode": mode, "set": name, "group": group, "limit": limit,
            "manifest_sha256": _sha256_file(manifest_of(name, group)),
            "scorer": {"kind": "deployed", "batch": 1} if mode in ("x1", "x3") else
            {"kind": "evaluate_set", "batch": EVAL_BATCH, "max_batch_seconds": EVAL_MAX_SECONDS}}


def _pending(out: Path, configs: dict) -> list[str]:
    return [n for n, c in configs.items() if not ((out / f"{n}.json").exists()
                                                  and json.loads((out / f"{n}.json").read_text()).get("config") == c)]


def _write_set(out: Path, name: str, group: str, result: dict, config: dict) -> None:
    add_term_recall(result, group)
    result.update(config=config, group=group)
    write_json(out / f"{name}.json", result)
    tr = result.get("term_recall")
    log(f"{config['model'].get('name')} {config['mode']} {name}: n={result['utterances']} WER "
        f"{100 * result['wer']['wer']:.3f}" + (f" term recall {tr['recall']:.4f}" if tr and tr.get("recall") is not None
                                               else ""))


def score_torch(net, out: Path, source: dict, device, label: str, groups, limit: int | None = None) -> dict:
    """PyTorch-mode scores (evaluate.evaluate_set at batch 64 and 1200 s, the decoder pinned to the stages' CUDA-graph
    mode, token IDs captured) of `net` on the groups' sets into out/<set>.json; a set already written with the same
    configuration is skipped. Returns the summary (also out/summary.json; a run's scores.json is the dev summary)."""
    import binary_train
    import keab_step0 as k0
    sets = [(n, g) for n, g in expand_sets(groups) if manifest_of(n, g).exists()]
    model = {"name": label, **source}
    configs = {n: set_config(model, "torch", n, g, limit) for n, g in sets}
    todo = _pending(out, configs)
    if todo:
        mode = binary_train.pin_decoder_graphs_mode(net, label)
        captured, restore = k0.capture_tokens(net)
        try:
            for name, group in sets:
                if name not in todo:
                    continue
                manifest = manifest_of(name, group)
                captured.clear()
                result = evaluate.evaluate_set(net, name, manifest, source, EVAL_BATCH, limit, torch.device(device),
                                               EVAL_MAX_SECONDS)
                k0.attach_tokens(result["records"], evaluate.read_manifest(manifest)[:limit], captured)
                result["decoder_graphs_mode"] = binary_train.require_decoder_graphs_mode(net, label)
                _write_set(out, name, group, result, configs[name])
        finally:
            restore()
        log(f"{label}: torch-mode scores in decoder mode {mode}")
    return summarize(out, sets, label, "torch")


def summarize(out: Path, sets, label: str, mode: str) -> dict:
    """{set: WER} etc. of the written set files; H4 when the four H4 sets are present."""
    res = {n: json.loads((out / f"{n}.json").read_text()) for n, _ in sets}
    wer = {n: r["wer"]["wer"] for n, r in res.items()}
    summary = {"label": label, "mode": mode, "complete": True, "sets": [n for n, _ in sets], "dev_wer":
               {n: wer[n] for n in DEV_SETS if n in wer}, "dev_utterances": {n: res[n]["utterances"] for n in DEV_SETS
                                                                           if n in res},
               "wer": wer, "term_recall": {n: r["term_recall"] for n, r in res.items() if "term_recall" in r},
               "utc": train.utc()}
    if all(n in wer for n in H4):
        summary["h4"] = sum(100 * wer[n] for n in H4) / len(H4)
    write_json(out / "summary.json", summary)
    return summary


def view_of(model: str) -> Path:
    """The scoring view a model is scored from in deployed mode: a branch's export/view, MB1's own export (the step 0
    "none" view is byte-identical to it)."""
    if model == "MB1":
        import keab_step0 as k0
        return k0.mb1_export()
    if model == INIT_MODEL:
        return init_dir() / "export" / "view"
    if model not in SCORED_RUNS:
        raise ValueError(f"{model} has no scoring view (deployed scoring: {SCORED_RUNS} and MB1)")
    return run_dir(model) / "export" / "view"


def init_dir() -> Path:
    return root() / INIT_MODEL


def init_export_job(lock_wait: float, deadline: float | None = None) -> int:
    """The main arm's student at step 0 (keab-prime plan "Teacher": "At step 0 the main arm's student is MB1's encoder
    with B0''s decoder. It is scored once, for information"): built as every main-arm run builds it (step 0's scope),
    exported on CUDA (the training forward's binary scales) to its 8-bit bundle and scoring view under
    $RELABEL_ROOT/keab-prime/init-main/export, for the x1 score item."""
    import keab_step0 as k0
    import score_teacher
    from training import keab_export as ke
    llayout.require_mount()
    scope = step0_scope()
    out = init_dir()
    if (out / "export" / "view" / "manifest.json").exists() and (out / "model.json").exists():
        log(f"{out}: exported")
        return 0
    wait = lock_wait if deadline is None else max(0.0, min(lock_wait, deadline - time.time()))
    try:
        with k0.waiting_lock("keab init-main export", wait):
            vram_lease()
            teacher, rec = load_teacher_model("main", "cpu")
            student, info = build_student(decoder_state(teacher), scope, resuming=False)
            del teacher
            student.to("cuda").eval()
            extra = {"stage": "keab-prime", "model": INIT_MODEL, "scope": scope, "teacher": rec, "init": info["init"]}
            bundle = ke.write_bundle(student, out / "export" / "bundle", extra=extra, root=root())
    except score_teacher.LockBusy as busy:
        if deadline is None:
            raise
        log(f"init-main export: the GPU lock was not free before the chunk deadline ({busy})")
        return EXIT_BUDGET
    view = ke.derive_view(out / "export" / "bundle", out / "export" / "view", root=root(), extra=extra)
    write_json(out / "model.json", {"model": INIT_MODEL, "scope": scope, "teacher": rec, "init": info["init"],
                                    "teacher_decoder_sha256": info["teacher_decoder_sha256"],
                                    "bundle_sha256": bundle["sha256"], "view_sha256": view["sha256"],
                                    "utc": train.utc()})
    return 0


def load_x3(view: Path, device: str = "cuda"):
    """The view on the deployed int8_mma stack (three activation parts), its cubin misses compiled under
    $RELABEL_ROOT/keab-prime (training/deployed.relabel_cubin_cache)."""
    from gpu1bit import runtime
    from training import deployed
    deployed.require_wrapper()
    stack = contextlib.ExitStack()
    stack.enter_context(deployed.relabel_cubin_cache(root()))
    try:
        with torch.inference_mode():
            model = runtime.load_binary_packed(Path(view), runtime="provider", kernel="int8_mma", device=device)
    except BaseException:
        stack.close()
        raise
    model._keab_stack = stack
    rt = model._gpu1bit_runtime
    model._keab_x1 = {"export": str(Path(view).resolve()), "export_sha256": rt.export_sha256, "kernel": rt.kernel,
                      "parts": rt.parts, "runtime": "provider"}
    return model


def score_deployed(model_name: str, mode: str, out: Path, groups, limit: int | None = None, view: Path | None = None
                   ) -> dict:
    """Deployed-mode scores (x1: training/deployed.load_x1, the recorded x1 arm's builder; x3: load_x3) of a model's
    scoring view, per utterance at batch 1 (fulldev.process_one), on the groups' sets into out/<set>.json."""
    import keab_step0 as k0
    from training import deployed
    deployed.require_wrapper()
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        raise SystemExit("deployed scoring runs as the recorded x1 arm ran: unset CUBLAS_WORKSPACE_CONFIG")
    view = Path(view or view_of(model_name))
    vman = json.loads((view / "manifest.json").read_text())
    sets = [(n, g) for n, g in expand_sets(groups) if manifest_of(n, g).exists()]
    model = {"name": model_name, "view": str(view), "view_sha256": vman["sha256"]}
    configs = {n: set_config(model, mode, n, g, limit) for n, g in sets}
    todo = _pending(out, configs)
    if todo:
        k0.vram_lease()
        net = deployed.load_x1(view) if mode == "x1" else load_x3(view)
        try:
            identity = hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest()
            for name, group in sets:
                if name in todo:
                    if group == "dev":
                        records = k0.deployed_records(name, "dev", limit)
                    else:
                        records = list(enumerate(evaluate.read_manifest(manifest_of(name, group))[:limit]))
                    result = deployed.score_records(net, records, name, identity)
                    _write_set(out, name, group, result, configs[name])
            write_json(out / "runtime.json", {"runtime": {k: v for k, v in net._keab_x1.items() if k != "sources"},
                                              "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 2**30})
        finally:
            deployed.close(net) if mode == "x1" else _close_x3(net)
    return summarize(out, sets, model_name, mode)


def deployed_environment() -> None:
    """The recorded x1 arm's environment for a deployed-mode process (training/deployed.py, keab_step0): when the caller
    started this process with CUBLAS_WORKSPACE_CONFIG unset, the value train.py set on import is removed again, before
    any CUDA work (cuBLAS reads it when its first handle is made)."""
    if _CUBLAS_AT_START is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


def _close_x3(net) -> None:
    from gpu1bit import runtime
    try:
        runtime.close_binary_packed(net)
    finally:
        net._keab_stack.close()


def mb1_torch_model(scope: str = "none"):
    """MB1 in PyTorch fake-quant mode (kernel fidelity's reference): its latent weights and B0's decoder (as MB1 was
    trained), the scope's 8-bit form, the activation quantizer."""
    teacher, _ = load_teacher_model("control", "cpu")
    student, _ = build_student(decoder_state(teacher), scope, resuming=False)
    del teacher
    return student


def latent_model(name: str, smoke: str | None = None):
    """A finished run's (or smoke's) latent model.pt in its training form (the activation quantizer on)."""
    d = (smoke_root() / smoke / "runs" / name) if smoke else root() / "runs" / name
    payload = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
    teacher, _ = load_teacher_model(payload["arm"], "cpu")
    student, _ = build_student(decoder_state(teacher), payload["scope"], resuming=True)
    del teacher
    student.load_state_dict(payload["student"], strict=True)
    eb.keab_form(student)
    return student, {"name": name, "path": str(d / "model.pt"), "sha256": _sha256_file(d / "model.pt")}


def score_groups(model: str, mode: str) -> tuple[str, ...]:
    """The set groups a score item covers: MODE_GROUPS[mode]; MB1 at x1 only what step 0 did not score; the step 0
    student (informational) on the dev and domain sets."""
    if (model, mode) == ("MB1", "x1"):
        return MB1_X1_GROUPS
    if model == INIT_MODEL:
        return ("dev", "domain")
    return MODE_GROUPS[mode]


def score_job(model: str, mode: str, limit: int | None, lock_wait: float, deadline: float | None) -> int:
    """One score item (ORDER): a model in one mode on its groups (MODE_GROUPS; MB1's x1 adds what step 0 did not
    score)."""
    import keab_step0 as k0
    import score_teacher
    llayout.require_mount()
    groups = score_groups(model, mode)
    out = scores_dir(model, mode) if limit is None else smoke_root() / "scores" / f"limit{limit}" / model / mode
    wait = lock_wait if deadline is None else max(0.0, min(lock_wait, deadline - time.time()))
    try:
        with k0.waiting_lock(f"keab score {model} {mode}", wait):
            vram_lease()
            if mode in ("torch", "fp32"):
                train.deterministic_mode()
            if mode in ("x1", "x3"):
                score_deployed(model, mode, out, groups, limit)
            elif mode == "torch":
                net, source = (mb1_torch_model(), {"name": "MB1", "sha256": MB1_SHA256}) if model == "MB1" else \
                    latent_model(model)
                score_torch(net.to("cuda").eval(), out, source, "cuda", model, groups, limit)
            elif mode == "fp32":
                if model not in ("B0", "B0p"):
                    raise ValueError("fp32 scoring is for the comparison models B0 and B0p")
                net = (train.load_pretrained("cuda") if model == "B0" else train.load_nemo(b0p_nemo(), "cuda")).eval()
                source = {"name": model, "sha256": B0P_SHA256 if model == "B0p" else train.verify_base()["base_sha256"]}
                score_torch(net, out, source, "cuda", model, groups, limit)
            else:
                raise ValueError(f"unknown mode {mode!r}")
    except score_teacher.LockBusy as busy:
        if deadline is None:
            raise
        log(f"score {model} {mode}: the GPU lock was not free before the chunk deadline ({busy})")
        return EXIT_BUDGET
    return 0


FIDELITY_LIMIT = 100           # utterances per dev set of the export check's slice


STEP0_VIEWS = {"none": "none", "S1": "S1-int8", "S2": "S2-int8"}     # step 0's MB1 scoring view of each scope


def fidelity_job(smoke: str | None, run: str | None, limit: int, lock_wait: float, scope: str | None = None) -> int:
    """The export contract check (keab-prime plan "Code and review", "Exported checkpoints"): a model's PyTorch
    fake-quant transcripts against its scoring view's on the i8x1 kernels, on the first `limit` utterances of every
    dev set. MB1 first, at a scope (its training model in that 8-bit form with the activation quantizer, against step
    0's MB1 scoring view of the scope): its token-sequence mismatch sets the bound (fidelity_bound); then a trained
    smoke model's, against MB1's at the smoke's own scope. Writes $RELABEL_ROOT/keab-prime/smoke/fidelity/<model>.json;
    exit EXIT_CHECK when the smoke model exceeds the bound."""
    import keab_step0 as k0
    llayout.require_mount()
    out = smoke_root() / "fidelity"
    if smoke is None:
        scope = scope or "none"
        name = f"MB1-{scope}"
    else:
        scope = torch.load(smoke_root() / smoke / "runs" / run / "model.pt", map_location="cpu", mmap=True,
                           weights_only=False)["scope"]
        name = f"{smoke}-{run}"
    with k0.waiting_lock(f"keab fidelity {name}", lock_wait):
        vram_lease()
        if smoke is None:
            net, source = mb1_torch_model(scope), {"name": "MB1", "sha256": MB1_SHA256, "scope": scope}
            view = k0.view_dir(STEP0_VIEWS[scope])
        else:
            net, source = latent_model(run, smoke)
            view = smoke_root() / smoke / "runs" / run / "export" / "view"
        torch_res = score_torch(net.to("cuda").eval(), out / name / "torch", source, "cuda", name, ("dev",), limit)
        del net
        torch.cuda.empty_cache()
        x1_res = score_deployed(name, "x1", out / name / "x1", ("dev",), limit, view=view)
    rows = {m: [r for n in DEV_SETS for r in json.loads((out / name / m / f"{n}.json").read_text())["records"]]
            for m in ("torch", "x1")}
    mis = k0.mismatch(rows["x1"], rows["torch"])
    result = {"model": name, "scope": scope, "limit": limit, "mismatch": mis,
              "h4": {"torch": torch_res.get("h4"), "x1": x1_res.get("h4")}, "utc": train.utc()}
    if smoke is not None:
        ref_path = out / f"MB1-{scope}.json"
        if not ref_path.exists():
            raise NotReady(f"no MB1 reference at scope {scope} ({ref_path}): run fidelity --scope {scope} first")
        ref = json.loads(ref_path.read_text())
        if ref["limit"] != limit:
            raise RuntimeError(f"{ref_path} was measured on {ref['limit']} utterances per set, not {limit}")
        bound = fidelity_bound(ref["mismatch"]["rate"])
        result.update(bound=bound, mb1_rate=ref["mismatch"]["rate"], within=mis["rate"] <= bound)
    write_json(out / f"{name}.json", result)
    log(json.dumps(result))
    return 0 if result.get("within", True) else EXIT_CHECK


def fidelity_bound(mb1_rate: float) -> float:
    """The stated bound on a trained model's x1-versus-PyTorch token-sequence mismatch: twice MB1's on the same slice,
    and at least MB1's plus one point (a 500-utterance slice's noise)."""
    return max(2 * mb1_rate, mb1_rate + 0.01)


# ----------------------------------------------------------------------------- memory probe

EXTREME_METRICS = tuple(tb.EXTREME_METRICS) + ("ami_teacher_audio",)
EXTREMES_PER_METRIC = tb.EXTREMES_PER_METRIC


def plan_extremes(plan, replay, labels, tokens, teacher_samples: dict[str, int], k: int = EXTREMES_PER_METRIC) -> dict:
    """stage 3's extremes (train_b0prime.plan_extremes: rows x longest clip at the plan's keys, the seam's padded
    lattice, rows x longest label, over the rows that keep a label) and, over the batches with a kept AMI pair, rows x
    the longest teacher-side clip (the IHM recording's samples for a pair, the row's key otherwise)."""
    per, texts = [], {}
    for t in range(plan.n_batches):
        b = int(plan.shuffle[t])
        keys = plan.samples[plan.order[int(plan.offsets[b]):int(plan.offsets[b + 1])]].tolist()
        ids = replay.batch_ids(t)
        kept = []
        for key, i in zip(keys, ids):
            if i not in texts:
                texts[i] = labels.text(i)
            if texts[i] is not None:
                kept.append((int(key), i))
        per.append(kept)
    order = [i for i, x in texts.items() if x is not None]
    lengths = dict(zip(order, tokens([texts[i] for i in order])))
    del texts
    values = {}
    for t, kept in enumerate(per):
        if kept:
            longest = max(s for s, _ in kept)
            label = max(lengths[i] for _, i in kept)
            ami = [i for _, i in kept if i in teacher_samples]
            values[t] = {"rows": len(kept), "longest_samples": longest, "longest_tokens": label, "ami_rows": len(ami),
                         "rows_x_longest_clip": len(kept) * longest,
                         "padded_lattice": len(kept) * tb.encoder_frames(longest) * (label + 1),
                         "rows_x_longest_label": len(kept) * label,
                         "ami_teacher_audio": len(kept) * max(teacher_samples.get(i, s) for s, i in kept) if ami else 0}
    top = {m: [t for t in sorted(values, key=lambda t: (-values[t][m], t)) if values[t][m] > 0][:k]
           for m in EXTREME_METRICS}
    batches = sorted({t for ts in top.values() for t in ts})
    return {"k": k, "top": top, "batches": batches, "values": {t: values[t] for t in batches},
            "note": tb.EXTREMES_NOTE + "; plus rows x the longest teacher-side clip over batches with an AMI pair",
            "ami_batches": sum(v["ami_rows"] > 0 for v in values.values())}


def probe_mode(ckpt: bool, batches: dict, steps: int, scope: str, arm: str = "main") -> dict:
    """One probe configuration: the arm's student and teacher (gradient checkpointing on or off) training through the
    stage 4 label_batch (the teacher's forward included) and the fused loss seam, every worst-case batch, then
    `steps` timed planned batches (the first untimed)."""
    import types
    teacher, _ = load_teacher_model(arm, "cpu")
    model, info = build_student(decoder_state(teacher), scope, resuming=False)
    train.seed_everything(stage4.SEED, model)
    model.to("cuda")
    if ckpt:
        train.enable_grad_checkpointing(model.encoder)
    teacher = train.load_teacher(teacher.to("cuda"))
    optimizer, scheduler = tseam.make_optimizer(model, 1e-4, steps)
    ctx = types.SimpleNamespace(student=model, teacher=teacher, recipe=dict(RECIPE), optimizer=optimizer,
                                scheduler=scheduler, quant=binary_quant, device="cuda",
                                state=types.SimpleNamespace(step=0, batches_read=0, acc={"dropped_rows": 0},
                                                            drop_counts={}, seconds_by_source={}),
                                stream=types.SimpleNamespace(schedule=types.SimpleNamespace(lr=lambda k: 1e-4)))
    trainable = [p for g in optimizer.param_groups for p in g["params"]]
    train.set_train_modes(model, RECIPE)
    times, seconds, rows, failed, peaks = [], [], [], [], {}

    def step(kind: str, j: int, batch):
        ctx.state.batches_read = int(batch["index"]) + 1
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        labeled = label_batch(ctx, batch)
        if labeled is None:
            return
        train.reseed_cudnn_rnn_dropout()
        binary_quant.set_weight_fraction(model, 1.0)
        train.train_step(ctx, labeled, trainable)
        torch.cuda.synchronize()
        if kind == "timed" and j >= 1:
            times.append(time.perf_counter() - t0)
            seconds.append(labeled["seconds"])
            rows.append(labeled["rows"])
        ctx.state.step += 1

    def after_oom(kind: str, j: int) -> None:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        failed.append(batches["worst_names"][j] if kind == "worst" else f"timed batch {j}")

    def phase_peak(kind: str, start: bool) -> None:
        if start:
            torch.cuda.reset_peak_memory_stats()
        else:
            peaks[kind] = (torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30)

    try:
        with tdt_fused.installed():
            status = tb.probe_phases([("worst", batches["worst"]), ("timed", batches["timed"])], step, after_oom,
                                     phase_peak)
    finally:
        batches["timed"].close()
    peak_alloc = max((a for a, _ in peaks.values()), default=None)
    peak_reserved = max((r for _, r in peaks.values()), default=None)
    process_now = td.gpu_process_gb()
    reserved_now = torch.cuda.memory_reserved() / 2**30
    peak_process = peak_reserved + max(0.0, (process_now or reserved_now) - reserved_now)
    out = {"grad_checkpointing": ckpt, "arm": arm, "scope": scope, "status": status, "failed": failed, "steps_timed": len(times),
           "step_time_s": float(np.mean(times)) if times else None,
           "step_time_p90_s": float(np.percentile(times, 90)) if times else None,
           "audio_seconds_per_step": float(np.mean(seconds)) if seconds else None,
           "rows_per_step": float(np.mean(rows)) if rows else None, "peak_allocated_gib": peak_alloc,
           "peak_reserved_gib": peak_reserved, "worst_reserved_gib": peaks.get("worst", (None, None))[1],
           "timed_reserved_gib": peaks.get("timed", (None, None))[1], "process_gpu_gib_at_end": process_now,
           "peak_process_gib_estimate": peak_process, "trainable": info["trainable"]["trainable"],
           "teacher_audio_rows": ctx.state.drop_counts.get(TEACHER_ROWS_KEY, 0), "host_peak_rss_gib": tb.rss_gib()}
    return out | {"_ctx": types.SimpleNamespace(student=model, optimizer=optimizer)}


def gpu_hours(plan, step_s: float) -> dict:
    """The plan's planned batches of the pilots, trunks and branches x the probe's step time (keab-prime plan
    "Estimate": six 78-pool pilots, two 664-pool trunks, two 118-pool cooldowns and one 59-pool cooldown)."""
    B = plan.pool_start
    pilot = len(stage4.PILOT) * B(stage4.PILOT_POOLS)
    trunk = len(stage4.TRUNKS) * B(stage4.TRUNK_POOLS)
    cool = sum(B(stage4.BRANCHES[n.split("-")[0]][1]) - B(stage4.BRANCHES[n.split("-")[0]][0])
               for n in stage4.BRANCH_RUNS)
    pools = (len(stage4.PILOT) * stage4.PILOT_POOLS + len(stage4.TRUNKS) * stage4.TRUNK_POOLS
             + sum(stage4.BRANCHES[n.split("-")[0]][1] - stage4.BRANCHES[n.split("-")[0]][0]
                   for n in stage4.BRANCH_RUNS))
    return {"step_time_s": step_s, "plan": getattr(plan, "meta", {}).get("plan_id"), "pools": pools, "pilot_planned_batches": pilot,
            "trunk_planned_batches": trunk, "cooldown_planned_batches": cool,
            "planned_batches": pilot + trunk + cool, "pilot_gpu_hours": pilot * step_s / 3600,
            "main_gpu_hours": (trunk + cool) * step_s / 3600, "total_gpu_hours": (pilot + trunk + cool) * step_s / 3600,
            "note": "probe step time x planned batches (fully rejected batches take no step); excludes replay waits, "
                    "checkpoints, the trunks' dev-subset evaluations, the finishing exports and every score item"}


def probe_job(labels_run: str, steps: int, scope: str, provisional: bool, lock_wait: float) -> int:
    """The memory probe (keab-prime plan "Resources"): the main arm's student and teacher at 600 s batches, with and
    without gradient checkpointing, on every worst-case batch (stage 2b's synthetic stress batches and the version 3
    plan's extremes by EXTREME_METRICS, ranked on each arm's labels and served with them), then `steps` timed planned
    batches; peak VRAM, host memory, step time, the decision (a mode only if both phases pass; the 6 GB rule) and the
    GPU-hour estimate; probe/probe.json."""
    import score_teacher
    from training import ami_store, poolaudio
    llayout.require_mount()
    stores = source_stores(provisional)
    _, pool, cid, plan, sources = pinned_plan(stores, provisional)
    t0 = time.perf_counter()
    replay = treplay.PlannedReplay(plan, pool, audio="materialized", store=poolaudio.store_dir(cid, plan.pool_batches),
                                   sources=replay_sources(stores, sources))
    opened = time.perf_counter() - t0
    labels, measured = {}, {"replay_open_s": opened}
    for arm in stage4.ARMS:
        t0, before = time.perf_counter(), tb.current_rss_gib()
        labels[arm] = arm_labels(arm, labels_run, pool, plan, stores, sources)
        measured[f"labels_open_s_{arm}"] = time.perf_counter() - t0
        measured[f"labels_rss_gib_{arm}"] = tb.current_rss_gib() - before
    ami = next(pool_ for (cfg, pool_) in stores if cfg[6] == "pair")
    teacher_samples = {f"{ami_store.PREFIX}{c['id']}": int(c["ihm_samples"]) for c in ami.clips}
    tokens = tb.token_counter(labels_run)
    extremes, worst, names = {}, [], []
    synthetic, _ = td.probe_batches(stage4.SEED)
    for k, items in enumerate(synthetic):
        b = train.collate(items)
        b.update(index=k, labels=[it["text"] for it in items])
        worst.append(b)
        names.append(f"synthetic {k}")
    for arm in stage4.ARMS:
        t0 = time.perf_counter()
        extremes[arm] = plan_extremes(plan, replay, labels[arm], tokens, teacher_samples)
        extremes[arm]["ranking_s"] = time.perf_counter() - t0
        log(f"plan extremes ({arm} labels): {json.dumps(extremes[arm], default=str)}")
        for t in extremes[arm]["batches"]:
            one = label_sources.Batcher(replay, labels[arm], start=t, stop=t + 1)
            try:
                worst.append(one.next(timeout=1800))
            finally:
                one.close()
            names.append(f"planned batch {t} ({arm} labels; "
                         f"{', '.join(m for m in EXTREME_METRICS if t in extremes[arm]['top'][m])})")
    modes, round_trip = {}, None
    with score_teacher.gpu_lock("keab memory probe", lock_wait):
        vram_lease()
        train.deterministic_mode()
        tdt_sha = train.prepare_tdt_loss()
        for ckpt in (False, True):
            batches = {"worst": worst, "worst_names": names} | tb.timed_batches(replay, labels["main"], steps,
                                                                                 stage4.SEED)
            res = probe_mode(ckpt, batches, steps, scope)
            res["timed_from_planned_batch"], res["timed_pool"] = batches["timed_start"], batches["timed_pool"]
            ctx = res.pop("_ctx")
            if round_trip is None and res["status"]["timed"] == "ok":
                round_trip = tb.checkpoint_round_trip(ctx, root() / "probe")
            del ctx
            import gc
            gc.collect()                                   # the mode's student, teacher and AdamW state
            torch.cuda.empty_cache()
            modes["ckpt" if ckpt else "no_ckpt"] = res
            log(json.dumps(res))
    replay.close()
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    fits = {k: tb.probe_fits(m["status"]) for k, m in modes.items()}
    free = {k: total - m["peak_process_gib_estimate"] for k, m in modes.items()}
    choice = "no_ckpt" if fits["no_ckpt"] and free["no_ckpt"] >= FREE_GB else ("ckpt" if fits["ckpt"] else None)
    decision = {"fits": choice is not None, "grad_checkpointing": choice == "ckpt" if choice else None,
                "chosen": choice, "gpu_total_gib": total, "free_gib": free,
                "rule": f"a mode only if both phases pass; gradient checkpointing if the run without it leaves less "
                        f"than {FREE_GB} GB for others or does not fit",
                "lease_vram_gb": math.ceil(modes[choice]["peak_process_gib_estimate"] + LEASE_MARGIN_GB)
                if choice else None, "coverage": extremes["main"]["note"]}
    host = tb.rss_gib()
    result = {"steps": steps, "scope": scope, "plan_sha256": plan.sha256, "plan_id": plan.meta["plan_id"],
              "provisional": is_provisional(stores), "labels_sha256": {a: labels[a].identity["sha256"]
                                                                        for a in stage4.ARMS},
              "modes": modes, "decision": decision, "worst_batches": names, "extremes": extremes,
              "checkpoint_round_trip": round_trip, "data": measured, "loss_seam": tdt_fused.settings(),
              "actquant": actquant.identity(), "host_peak_rss_gib": host,
              "mem_cap_gb": math.ceil(host * 1.25), "tdt_kernel_sha256": tdt_sha, "utc": train.utc(),
              "estimate": gpu_hours(plan, modes[choice]["step_time_s"]) if choice and modes[choice]["step_time_s"]
              else None}
    write_json(probe_path() if not result["provisional"] else root() / "probe" / "provisional.json", result)
    if result["provisional"]:
        log("provisional plan: written to probe/provisional.json; every production run needs the probe on the final "
            "plan (probe.json)")
    log(json.dumps({k: v for k, v in result.items() if k not in ("modes", "extremes")}, default=str))
    return 0 if choice else 1


# ----------------------------------------------------------------------------- chunks and status

ORDER = ([("train", n) for n in stage4.PILOT] + [("train", n) for n in stage4.MAIN]
         + [("score", m, "x1") for m in SCORED_RUNS] + [("score", m, "x3") for m in SCORED_RUNS]
         + [("score", "MB1", "x1"), ("score", "MB1", "x3"), ("score", "MB1", "torch"),
            ("score", "B0", "fp32"), ("score", "B0p", "fp32"), ("export", INIT_MODEL), ("score", INIT_MODEL, "x1")])
THROUGH = {"pilot": len(stage4.PILOT), "main": len(stage4.PILOT) + len(stage4.MAIN), "score": len(ORDER)}
GPU1BIT_MODES = ("x1", "x3")


def item_done(item: tuple) -> bool:
    if item[0] == "train":
        return run_done(item[1])
    if item[0] == "export":
        return (init_dir() / "model.json").exists()
    _, model, mode = item
    path = scores_dir(model, mode) / "summary.json"
    if not path.exists():
        return False
    want = {n for n, g in expand_sets(score_groups(model, mode)) if manifest_of(n, g).exists()}
    return set(json.loads(path.read_text()).get("sets") or []) >= want


def item_command(item: tuple, labels_run: str, minutes: float, deadline: float) -> list[str]:
    here = Path(__file__).resolve().parent
    if item[0] == "train":
        return [str(here / "python"), str(here / "train_keab.py"), "train", "--run", item[1], "--labels-run",
                labels_run, "--budget-minutes", f"{minutes:.1f}", "--deadline", f"{deadline:.0f}"]
    if item[0] == "export":
        return [str(here / "python"), str(here / "train_keab.py"), "export-init", "--deadline", f"{deadline:.0f}"]
    _, model, mode = item
    py = ["env", "-u", "CUBLAS_WORKSPACE_CONFIG", str(here / "python-gpu1bit")] if mode in GPU1BIT_MODES \
        else [str(here / "python")]
    return py + [str(here / "train_keab.py"), "score", "--model", model, "--mode", mode, "--deadline", f"{deadline:.0f}"]


def chunk(labels_run: str, budget_minutes: float, through: str, runner=subprocess.run) -> int:
    """One chunk job: the next unfinished items of ORDER up to `through` (pilot: the six pilots; main: also the
    trunks and branches; score: also the deployed-mode and comparison scores), each in its own process (which takes
    the parakeet GPU lock for its own duration), until the budget leaves less than MIN_CHUNK_MINUTES, an item stops on
    the budget, or an item fails (its exit code)."""
    llayout.require_mount()
    items = ORDER[:THROUGH[through]]
    mismatched = [m for m in (run_mismatch(i[1], labels_run) for i in items if i[0] == "train") if m]
    if mismatched:
        log("chunk: refusing to mix runs of other labels: " + "; ".join(mismatched))
        return train.EXIT_RESUME_MISMATCH
    t0, deadline = time.monotonic(), time.time() + budget_minutes * 60
    for item in items:
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
    log(f"chunk: {sum(item_done(i) for i in items)}/{len(items)} items done (through {through})")
    return 0


def status() -> dict:
    out = {}
    for item in ORDER:
        key = item[1] if item[0] == "train" else " ".join(item)
        if item_done(item):
            out[key] = "done"
        elif item[0] == "train" and (run_dir(item[1]) / "ckpt" / "latest.pt").exists():
            m = run_dir(item[1]) / "metrics.jsonl"
            last = train.read_log_jsonl(m)[0][-1:] if m.exists() else []
            out[key] = f"in progress (step {last[0]['step'] if last else '?'})"
        else:
            out[key] = "pending"
    out["selection"] = json.loads(selection_path().read_text()) if selection_path().exists() else "not frozen"
    for name, path in (("memory probe", probe_path()), ("read probe", read_probe_path())):
        body = json.loads(path.read_text()) if path.exists() else None
        out[name] = "pending" if body is None else (body.get("decision") or {}).get("chosen") if "decision" in body \
            else ("passed" if body.get("pass") else "FAILED")
    try:
        out["8-bit scope (step 0)"] = step0_scope()
    except NotReady as wait:
        out["8-bit scope (step 0)"] = f"pending ({wait})"
    return out


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "train":
        return run_train(argv[1:])
    if argv and argv[0] == "smoke":
        s = argparse.ArgumentParser(prog="train_keab.py smoke")
        s.add_argument("--arm", choices=stage4.ARMS, required=True)
        s.add_argument("--labels-run", required=True)
        s.add_argument("--scope", choices=eb.SCOPES, required=True)
        s.add_argument("--steps", type=int, required=True)
        s.add_argument("--provisional", action="store_true")
        s.add_argument("--name", default="smoke")
        s.add_argument("--grad-checkpointing", choices=("on", "off"))
        a = s.parse_args(argv[1:])
        extra = ["--provisional"] if a.provisional else []
        extra += ["--grad-checkpointing", a.grad_checkpointing] if a.grad_checkpointing else []
        return run_train(["--run", stage4.pilot_name(a.arm, stage4.PILOT_LRS[1]), "--labels-run", a.labels_run,
                          "--smoke", a.name, "--steps", str(a.steps), "--scope", a.scope, "--budget-minutes", "0",
                          *extra])
    parser = argparse.ArgumentParser(prog="train_keab.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "read-probe", "labels"):
        p = sub.add_parser(name)
        p.add_argument("--provisional", action="store_true")
        if name == "read-probe":
            p.add_argument("--max-seconds", type=float, default=120.0)
        if name == "labels":
            p.add_argument("--labels-run", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--labels-run", required=True)
    p.add_argument("--scope", choices=eb.SCOPES, required=True)
    p.add_argument("--steps", type=int, default=PROBE_STEPS)
    p.add_argument("--provisional", action="store_true")
    p.add_argument("--lock-wait", type=float, default=4 * 3600.0)
    p = sub.add_parser("score")
    p.add_argument("--model", choices=SCORED_RUNS + REFERENCE_MODELS + stage4.PILOT + (INIT_MODEL,), required=True)
    p.add_argument("--mode", choices=tuple(MODE_GROUPS), required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--lock-wait", type=float, default=4 * 3600.0)
    p.add_argument("--deadline", type=float)
    p = sub.add_parser("export-init")
    p.add_argument("--lock-wait", type=float, default=4 * 3600.0)
    p.add_argument("--deadline", type=float)
    p = sub.add_parser("fidelity")
    p.add_argument("--smoke", help="a smoke's name (default: MB1, which sets the bound)")
    p.add_argument("--scope", choices=eb.SCOPES, help="MB1's scope (default none; a smoke uses its own)")
    p.add_argument("--run", help="the smoke's run (pilot-<arm>-lr6e-5)")
    p.add_argument("--limit", type=int, default=FIDELITY_LIMIT)
    p.add_argument("--lock-wait", type=float, default=4 * 3600.0)
    p = sub.add_parser("chunk")
    p.add_argument("--labels-run", required=True)
    p.add_argument("--through", choices=sorted(THROUGH), required=True)
    p.add_argument("--budget-minutes", type=float, default=CHUNK_MINUTES)
    sub.add_parser("status")
    sub.add_parser("report")
    args = parser.parse_args(argv)
    if args.cmd == "fidelity" or (args.cmd == "score" and args.mode in GPU1BIT_MODES):
        deployed_environment()
    try:
        if args.cmd == "plan":
            return build_plan(args.provisional)
        if args.cmd == "read-probe":
            return read_probe_job(args.max_seconds, args.provisional)
        if args.cmd == "labels":
            return labels_job(args.labels_run, args.provisional)
        if args.cmd == "probe":
            return probe_job(args.labels_run, args.steps, args.scope, args.provisional, args.lock_wait)
        if args.cmd == "score":
            return score_job(args.model, args.mode, args.limit, args.lock_wait, args.deadline)
        if args.cmd == "export-init":
            return init_export_job(args.lock_wait, args.deadline)
        if args.cmd == "fidelity":
            return fidelity_job(args.smoke, args.run, args.limit, args.lock_wait, args.scope)
    except tb.NotReady as wait:
        log(f"{args.cmd}: not ready: {wait}")
        return EXIT_NOT_READY
    if args.cmd == "chunk":
        return chunk(args.labels_run, args.budget_minutes, args.through)
    if args.cmd == "status":
        print(json.dumps(status(), indent=1, default=str))
        return 0
    import keab_report
    return keab_report.main()


if __name__ == "__main__":
    sys.exit(main())
