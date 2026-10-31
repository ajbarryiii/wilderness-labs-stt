"""The batch plan: training batches re-cut once, offline, from C1's journal (b0prime plan "Batch plan" and
"Units", 2026-10-24). Teacher-independent: it reads only the journal, so every arm of stages 2b, 2c, 3 and 4 uses
the same batches.

Contract (the cut is shared with an independent implementation and must match it byte for byte):
- Sequence: every journal occurrence of consumed batches [0, C), in (batch, pos) order across shards; its index
  in that order is its sequence number. A repeated id is a separate occurrence.
- Key: samples = round(duration * 16000); the build refuses a row with |duration * 16000 - samples| > 1e-6.
- Pools: P consecutive consumed batches from batch 0 (pool p = consumed batches [P p, P p + P)); a plan covers
  whole pools, and only the journal's last pool may be short (when C1 is complete).
- Cut, per pool: stable sort by (samples, sequence number); walk in that order and start a new batch before
  clip i when the current batch is non-empty and (len(current) + 1) * samples[i] > budget (train.cut_pool's rule
  in integers; evaluated as len + 1 > budget // samples[i], which is equivalent for positive samples). Batches
  are emitted pool by pool, in cut order; rows within a batch are in cut order.
- Shuffle (Python only, not part of the cross-implementation contract): the batches of pool p are served in
  the order numpy.random.default_rng([seed, 2, p]).permutation(n_p), train.Batcher's convention: training
  position j of the pool gets cut batch perm[j] of the pool.

Files of a plan directory (every array little-endian uint32, no header):
    plan.json        kind, version, plan_id, pinned_hash, pool_dir (relative to $RELABEL_ROOT), sample_rate,
                     pool_batches (P), budget_samples, seed, the rules as text, consumed_batches (C), n_pools,
                     n_occurrences, n_batches, journal_shards [{shard, batches, journal_sha256}],
                     files {name: {sha256, bytes, count}}, stats
    samples.u32      n_occurrences: the key of each sequence number (the cut's input)
    pools.u32        n_pools + 1: pool start offsets into the sequence (the cut's input)
    order.u32        n_occurrences: sequence numbers in cut order (the cut's output)
    offsets.u32      n_batches + 1: batch start offsets into order.u32, 0 .. n_occurrences (the cut's output)
    batch_pools.u32  n_pools + 1: pool start offsets into the batch list (cut batches of pool p are
                     [batch_pools[p], batch_pools[p + 1]); also the training positions of pool p)
    shuffle.u32      n_batches: the cut batch served at each training position (a permutation within each pool)

Training position t serves order[offsets[b] : offsets[b + 1]] with b = shuffle[t]. plan.json is written
deterministically, so its SHA-256 is the plan's identity (Plan.sha256). Pools are independent, so the files of a
plan over the first N pools are prefixes of a longer plan's (is_prefix).

Version 2 (2026-10-24, synthetic plan "Mixing"): the synthetic arms' plan. Synthetic clips join each pool
before the cut. The cut runs over ENTRIES instead of journal occurrences:
- Entry sequence: pool by pool, the pool's real occurrences in sequence order, then the pool's synthetic draw in
  draw order. samples.u32, pools.u32, order.u32 and offsets.u32 index this entry sequence (order.u32 holds entry
  indices), so the cut contract above applies unchanged to (samples.u32, pools.u32, budget): ties are broken by
  entry index, which puts a real row before a synthetic one of the same key.
- entries.u32 (n_entries): what an entry is. A value v < 2^31 is the journal sequence number v (a real row);
  v >= 2^31 is synthetic clip v - 2^31, an index into the clip pool's clips sorted by id.
- Keys: a real entry's key is its samples (as in version 1); a synthetic entry's key is
  ceil(samples * 10 / 9), the longest the TS arm's speed perturbation (factor >= 0.9) can make the clip, so every
  served batch keeps count x longest <= budget with or without augmentation (both arms share the plan).
- Draw: synthetic.mixing.plan over the pools: pool p's target is SHARE x the seconds of ALL its real occurrences
  (teacher-independent) plus the carry from earlier pools; clips come from the seeded clip order (each pass a
  permutation of the whole clip pool) and are taken while the next one closes more of the target than it
  overshoots. So each pool's synthetic seconds are within one clip of its target and the running total within
  half a clip.
- plan.json adds: base_plan_sha256 (the version 1 plan of the same journal, P, budget and seed: the no-synthetic
  arms' plan), n_entries, n_synthetic, synthetic {clip_pool (its identity), n_clips, clip_order, seed, share,
  share_basis, key_rule, draw_rule}, and stats.synthetic; stats are of the unaugmented audio as served (synthetic
  rows at their stored length), stats.by_key at the keys.

Version 2 with several sources (2026-10-29, b0prime plan v4 "Extra sources"): compute_sources, the same rule
generalized to a list of extra sources (Source: a clip store, a name and row-id prefix, a seed, a share and an
augmentation flag). Each source has its own seeded clip order (synthetic.mixing.clip_order) and its own
error-diffused draw over the pools (synthetic.mixing.plan) against SHARE x all real audio of the pool; a source whose
share is ONCE is drawn from one pass only, never repeated, at share = its total seconds / the real seconds of the
whole journal (so a plan over the first N pools is a prefix of the whole journal's), and on the journal's last pool
it also takes every clip not drawn yet, so each clip is drawn exactly once. A pool's entries are its real
occurrences, then source 0's draw, source 1's, ...; an extra entry v >= 2^31 is (v - 2^31) = s << SOURCE_BITS | c:
clip c (sorted by id) of source s (stage 2c's single source is source 0, so its entries are unchanged). Keys: an
augmented source's ceil(samples * 10 / 9), else its samples. plan.json has "sources" (each source's "config" and
its counts) instead of "synthetic", "n_extra" instead of "n_synthetic", and stats.sources. The plan id ends in
"-x" + 12 hex digits of the sources' configs.

Version 3 (keab-prime plan "Batch plan version 3", stage 4): the multi-source rule of version 2, unchanged, with two
more share kinds and a plan of its own version (plan.json "version": 3, plan id "-v3x" + the sources' 12 hex digits):
- "times:K": every clip is drawn exactly K times over the complete journal (the AMI distant-microphone pairs, K = the
  average AMI IHM train id's occurrences in pools 0-781): share = K x the source's seconds / the whole journal's real
  seconds, K seeded passes (synthetic.mixing.plan passes=K), never more, and on the journal's last pool every clip of
  them not drawn yet. A prefix plan draws from the same passes (never repeating a clip beyond K);
- "about:K": share = K x the source's seconds / the whole journal's real seconds, drawn by the repeating rule (a clip
  repeats only after the whole source is drawn), so each clip is drawn about K times over the journal (the synthetic
  union full + pairs + pairs-c, K = 2);
- ONCE and a fixed fraction as in version 2.
Keys: an augmented source's ceil(samples * 10 / 9), else its samples (the AMI pairs: the student's, SDM, samples).
plan.json's per-source entry adds "kind", "k" and "draws_per_clip" ({min, max} over the source's clips) in version 3;
a version 2 plan's bytes are unchanged.

Cut engines (`engine`, CLI --cut): "numpy" (cut(), the default) or "bend", finetune/bend-planner's binary (a Bend 2
implementation of the cut contract with its laws proved; build it per its README). The bend engine writes the cut's
inputs to a temporary directory under $RELABEL_ROOT/tmp (on the data mount), runs the planner, also runs cut() and
refuses the plan unless order.u32 and offsets.u32 are byte-identical to numpy's; the temporary directory is removed
either way. Since the cut is checked before anything is written, a refused cut writes no plan, and a plan
directory (plan.json included) is the same whichever engine produced it. Which engine ran is not part of the plan;
the CLI prints it.

    ./python -m training.batchplan build --pools N        the production plan over pools 0 .. N-1
                     [--cut bend [--planner PATH] [--threads N]]
    ./python -m training.batchplan stats --plan DIR       counts and padding
    ./python -m training.batchplan materialize ...        training/poolaudio.py
    ./python -m training.batchplan probe ...              training/replay_probe.py
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from labeling import layout, shards

KIND = "relabel-batch-plan"
VERSION = 1
SAMPLE_RATE = 16_000
POOL_BATCHES = 128
BUDGET_SAMPLES = 600 * SAMPLE_RATE        # 9,600,000: the main runs' 600 s padded batch
SEED = 20261030
EXACT_TOLERANCE = 1e-6
U32_MAX = 2**32 - 1
FILES = ("samples.u32", "pools.u32", "order.u32", "offsets.u32", "batch_pools.u32", "shuffle.u32")
CUT_RULE = ("per pool: stable sort by (samples, sequence number); a new batch starts before clip i when the "
            "batch is non-empty and (len + 1) * samples[i] > budget_samples")
SHUFFLE_RULE = "training position j of pool p serves cut batch numpy.random.default_rng([seed, 2, p]).permutation(n_p)[j]"
KEY_RULE = "samples = round(duration * 16000), refused if |duration * 16000 - samples| > 1e-6"
SYNTHETIC_VERSION = 2
SYN_FLAG = 1 << 31
STRETCH = (10, 9)            # synthetic/augment.py: speed factor >= 0.9 lengthens a clip to at most ceil(n * 10 / 9)
SYN_KEY_RULE = "ceil(samples * 10 / 9): the longest speed-perturbed length (augment.SPEED[0] = 0.9)"
SYN_DRAW_RULE = ("synthetic.mixing.plan(real seconds per pool, clip seconds, seed, share): error diffusion over "
                 "pools in order, clips in the seeded clip order")
SYN_SHARE_BASIS = "all real occurrences of the pool (teacher-independent)"
FILES_V2 = FILES + ("entries.u32",)
SOURCE_BITS = 27                 # an extra entry: (v - SYN_FLAG) = source << SOURCE_BITS | clip
ONCE = "once"
V3 = 3                           # keab-prime plan: version 3, the multi-source rule with the share kinds below
TIMES, ABOUT = "times", "about"  # "times:K" (each clip exactly K times), "about:K" (share = K x its seconds / real)
MULTI_VERSIONS = (SYNTHETIC_VERSION, V3)
ENTRY_RULE = (f"an extra entry v >= 2^31 is clip (v - 2^31) & (2^{SOURCE_BITS} - 1), sorted by id, of source "
              f"(v - 2^31) >> {SOURCE_BITS} in the plan's source order")
ENGINES = ("numpy", "bend")
BEND_PLANNER = Path(__file__).resolve().parents[2] / "bend-planner" / "build" / "planner"
BEND_BUILD = "nix develop <repo> -c bend planner.bend -o build/planner (in finetune/bend-planner)"


class PlanError(RuntimeError):
    """A plan or journal that does not satisfy the contract."""


# ----------------------------------------------------------------------------- journal

def journal_end(pool_dir: Path) -> int | None:
    """The journal's total consumed batches when C1 is complete (its manifest), else None."""
    p = Path(pool_dir) / "manifest.json"
    return int(json.loads(p.read_text())["batches"]) if p.exists() else None


def read_journal(pool_dir: Path, through_batch: int, with_rows: bool = True) -> dict:
    """The sequence of consumed batches [0, through_batch): samples (uint32), batch (int64), and with_rows the
    ids, sources and durations; journal shards checked against their metas' SHA-256; one pinned configuration."""
    import pyarrow.parquet as pq
    from training.labels import needed_shards
    pool_dir = Path(pool_dir)
    samples, batch, ids, sources, durations, listed, pinned = [], [], [], [], [], [], set()
    for k in needed_shards(pool_dir, through_batch):
        meta = shards.load_meta(pool_dir, k)
        path = shards.paths(pool_dir, k)["journal"]
        if meta.get("kind") != "replay":
            raise PlanError(f"C1 shard {k} is not a replay shard")
        if shards.sha256_file(path) != meta["files"]["journal"]["sha256"]:
            raise PlanError(f"C1 shard {k}: journal SHA-256 differs from its meta")
        pinned.add(meta["pinned_hash"])
        t = pq.read_table(path, columns=["batch", "pos", "id", "source", "duration"])
        b = t.column("batch").to_numpy().astype(np.int64)
        pos = t.column("pos").to_numpy().astype(np.int64)
        d = t.column("duration").to_numpy().astype(np.float64)
        lo, hi = meta["batches"]
        if len(b) and (b.min() < lo or b.max() >= hi):
            raise PlanError(f"journal of shard {k} has batches outside {meta['batches']}")
        keep = b < through_batch
        b, pos, d = b[keep], pos[keep], d[keep]
        # (batch, pos) order, positions 0 .. n-1 within each batch, every batch of the range present
        if len(b) and (np.any(np.diff(b) < 0) or b[0] != lo):
            raise PlanError(f"journal of shard {k} is not in consumed-batch order")
        new = np.r_[True, b[1:] != b[:-1]] if len(b) else np.zeros(0, bool)
        starts = np.flatnonzero(new)
        expect_pos = np.arange(len(b)) - np.repeat(starts, np.diff(np.r_[starts, len(b)]))
        if not np.array_equal(pos, expect_pos):
            raise PlanError(f"journal of shard {k}: positions out of order")
        present = np.unique(b)
        want = np.arange(lo, min(hi, through_batch))
        if not np.array_equal(present, want):
            raise PlanError(f"journal of shard {k} has no rows for some batches of {meta['batches']}")
        x = d * SAMPLE_RATE
        r = np.rint(x)
        bad = np.flatnonzero(np.abs(x - r) > EXACT_TOLERANCE)
        if len(bad):
            raise PlanError(f"journal of shard {k}: duration {d[bad[0]]!r} is not a whole number of samples")
        if len(r) and (r.min() < 1 or r.max() > U32_MAX):
            raise PlanError(f"journal of shard {k}: a clip length is outside [1, 2^32)")
        samples.append(r.astype(np.uint32))
        batch.append(b)
        if with_rows:
            n = int(keep.sum())
            ids += t.column("id").to_pylist()[:n]
            sources += t.column("source").to_pylist()[:n]
            durations += d.tolist()
        listed.append({"shard": k, "batches": meta["batches"], "journal_sha256": meta["files"]["journal"]["sha256"]})
    if len(pinned) != 1:
        raise PlanError(f"C1 shards from {len(pinned)} pinned configurations")
    out = {"samples": np.concatenate(samples), "batch": np.concatenate(batch), "shards": listed,
           "pinned_hash": pinned.pop()}
    if with_rows:
        out.update(ids=ids, sources=sources, durations=durations)
    return out


def pool_starts(batch: np.ndarray, n_pools: int, pool_batches: int) -> np.ndarray:
    """n_pools + 1 sequence offsets: pool p starts at the first occurrence of consumed batch P p."""
    bounds = np.arange(n_pools + 1, dtype=np.int64) * pool_batches
    return np.searchsorted(batch, bounds, side="left").astype(np.int64)


# ----------------------------------------------------------------------------- cut and shuffle

def cut(samples: np.ndarray, pools: np.ndarray, budget: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(order, offsets, batch_pools) per the module contract."""
    samples = np.asarray(samples, dtype=np.int64)
    pools = np.asarray(pools, dtype=np.int64)
    if len(pools) < 1 or pools[0] != 0 or pools[-1] != len(samples) or np.any(np.diff(pools) < 0):
        raise PlanError("pool offsets must start at 0, never decrease and end at n")
    if not 0 < budget <= U32_MAX:
        raise PlanError(f"budget {budget} is not a positive u32")
    order = np.empty(len(samples), dtype=np.int64)
    offsets, batch_pools = [0], [0]
    for lo, hi in zip(pools[:-1].tolist(), pools[1:].tolist()):
        seq = np.arange(lo, hi, dtype=np.int64)
        seq = seq[np.lexsort((seq, samples[lo:hi]))]      # primary: samples; ties: sequence number
        order[lo:hi] = seq
        count = 0
        for j, s in enumerate(samples[seq].tolist()):
            if count and s > 0 and count + 1 > budget // s:
                offsets.append(lo + j)
                count = 0
            count += 1
        if count:
            offsets.append(hi)
        batch_pools.append(len(offsets) - 1)
    return order, np.asarray(offsets, dtype=np.int64), np.asarray(batch_pools, dtype=np.int64)


def bend_cut(samples: np.ndarray, pools: np.ndarray, budget: int, planner: Path | None = None,
             threads: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """cut() by the Bend planner (default BEND_PLANNER), refused unless numpy's cut is byte-identical."""
    layout.require_mount()
    want = cut(samples, pools, budget)                    # validates the input; the cross-check
    planner = (BEND_PLANNER if planner is None else Path(planner)).absolute()   # never looked up on PATH
    if not (planner.is_file() and os.access(planner, os.X_OK)):
        raise PlanError(f"no Bend planner at {planner}; build it with {BEND_BUILD}")
    if threads is not None and int(threads) < 1:
        raise PlanError(f"threads {threads} is not positive")
    area = layout.relabel_root() / "tmp"
    area.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="bend-cut-", dir=area))
    try:
        s, p, o, f = (tmp / n for n in ("samples.u32", "pools.u32", "order.u32", "offsets.u32"))
        s.write_bytes(_u32(samples))
        p.write_bytes(_u32(pools))
        argv = [str(planner), str(s), str(p), str(int(budget)), str(o), str(f)]
        if threads is not None:
            argv += ["--threads", str(int(threads))]     # bend 2.0.37's runtime option (bend-planner README)
        try:
            proc = subprocess.run(argv, capture_output=True, text=True)
        except OSError as e:
            raise PlanError(f"the Bend planner {planner} did not run: {e}") from e
        if proc.returncode:
            raise PlanError(f"the Bend planner exited {proc.returncode}: {proc.stderr.strip()}")
        got = []
        for path, w in ((o, want[0]), (f, want[1])):
            raw = path.read_bytes() if path.exists() else b""
            if len(raw) != 4 * len(w):
                raise PlanError(f"the Bend planner wrote {len(raw)} bytes of {path.name}, not {4 * len(w)}")
            got.append(raw)
        if got[0] != _u32(want[0]) or got[1] != _u32(want[1]):
            raise PlanError("the Bend planner's cut differs from numpy's; refusing the plan")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    order, offsets = (np.frombuffer(b, dtype="<u4").astype(np.int64) for b in got)
    batch_pools = np.searchsorted(offsets, np.asarray(pools, dtype=np.int64)).astype(np.int64)
    if not np.array_equal(batch_pools, want[2]):
        raise PlanError("the Bend planner's batches do not split at numpy's pool boundaries")
    return order, offsets, batch_pools


def engine_cut(samples: np.ndarray, pools: np.ndarray, budget: int, engine: str = "numpy",
               planner: Path | None = None, threads: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """cut() by `engine` (module docstring, "Cut engines"); planner and threads are the bend engine's."""
    if engine == "numpy":
        if planner is not None or threads is not None:
            raise PlanError("planner and threads belong to the bend engine")
        return cut(samples, pools, budget)
    if engine == "bend":
        return bend_cut(samples, pools, budget, planner=planner, threads=threads)
    raise PlanError(f"unknown cut engine {engine!r} (one of {', '.join(ENGINES)})")


def shuffle(batch_pools: np.ndarray, seed: int) -> np.ndarray:
    out = np.empty(int(batch_pools[-1]), dtype=np.int64)
    for p, (lo, hi) in enumerate(zip(batch_pools[:-1].tolist(), batch_pools[1:].tolist())):
        out[lo:hi] = lo + np.random.default_rng([int(seed), 2, p]).permutation(hi - lo)
    return out


def stats(samples: np.ndarray, order: np.ndarray, offsets: np.ndarray, consumed: int) -> dict:
    s = samples[order].astype(np.int64)
    counts = np.diff(offsets)
    longest = np.maximum.reduceat(s, offsets[:-1]) if len(counts) else np.zeros(0, np.int64)
    real, padded = int(s.sum()), int((counts * longest).sum())
    n = len(counts)
    return {"padding": 1.0 - real / padded if padded else 0.0, "real_seconds": real / SAMPLE_RATE,
            "padded_seconds": padded / SAMPLE_RATE, "real_seconds_per_batch": real / SAMPLE_RATE / max(n, 1),
            "rows_min": int(counts.min()) if n else 0, "rows_max": int(counts.max()) if n else 0,
            "planned_per_consumed": n / max(consumed, 1)}


# ----------------------------------------------------------------------------- files

def _u32(values: np.ndarray) -> bytes:
    a = np.asarray(values, dtype=np.int64)
    if len(a) and (a.min() < 0 or a.max() > U32_MAX):
        raise PlanError("value outside u32")
    return a.astype("<u4").tobytes()


def read_u32(path: Path) -> np.ndarray:
    raw = Path(path).read_bytes()
    if len(raw) % 4:
        raise PlanError(f"{path}: length {len(raw)} is not a multiple of 4")
    return np.frombuffer(raw, dtype="<u4").astype(np.int64)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dumps(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=1) + "\n").encode()


def plan_id(config_id: str, pool_batches: int, n_pools: int) -> str:
    return f"{config_id}-p{int(pool_batches)}-n{int(n_pools)}"


def production_dir(config_id: str, n_pools: int, pool_batches: int = POOL_BATCHES, clips=None,
                   sources: list | None = None, version: int = SYNTHETIC_VERSION) -> Path:
    """$RELABEL_ROOT/training/batch-plan/<plan id> (with `clips`: the version 2 plan drawn from that clip pool; with
    `sources`: the multi-source plan of `version`, 2 or 3)."""
    pid = plan_id(config_id, pool_batches, n_pools)
    if clips is not None:
        pid = f"{pid}-syn{str(clips.identity)[:12]}"
    if sources is not None:
        pid = f"{pid}{sources_suffix(sources, version)}"
    return layout.relabel_root() / "training" / "batch-plan" / pid


def _rel(path: Path) -> str:
    path = Path(path).resolve()
    root = layout.relabel_root().resolve()
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def compute(pool_dir: Path, n_pools: int, *, pool_batches: int = POOL_BATCHES, budget: int = BUDGET_SAMPLES,
            seed: int = SEED, config_id: str | None = None, engine: str = "numpy", planner: Path | None = None,
            threads: int | None = None) -> tuple[dict, dict[str, bytes]]:
    """(plan.json body, file bytes) of the plan over pools 0 .. n_pools - 1 (no plan written; the cut by `engine`)."""
    if n_pools < 1 or pool_batches < 1:
        raise PlanError("n_pools and pool_batches must be positive")
    end = journal_end(pool_dir)
    consumed = n_pools * pool_batches
    if end is not None and consumed > end:
        if (end + pool_batches - 1) // pool_batches != n_pools:
            raise PlanError(f"{n_pools} pools of {pool_batches} exceed the journal's {end} batches")
        consumed = end                                   # the journal's last pool is short
    j = read_journal(pool_dir, consumed, with_rows=False)
    samples, batch = j["samples"], j["batch"]
    if int(samples.max()) > budget:
        raise PlanError(f"a clip of {int(samples.max())} samples exceeds the budget {budget}")
    pools = pool_starts(batch, n_pools, pool_batches)
    order, offsets, bpools = engine_cut(samples, pools, budget, engine, planner, threads)
    shuf = shuffle(bpools, seed)
    data = {"samples.u32": _u32(samples), "pools.u32": _u32(pools), "order.u32": _u32(order),
            "offsets.u32": _u32(offsets), "batch_pools.u32": _u32(bpools), "shuffle.u32": _u32(shuf)}
    counts = {"samples.u32": len(samples), "pools.u32": len(pools), "order.u32": len(order),
              "offsets.u32": len(offsets), "batch_pools.u32": len(bpools), "shuffle.u32": len(shuf)}
    cid = config_id or Path(pool_dir).name
    body = {"kind": KIND, "version": VERSION, "plan_id": plan_id(cid, pool_batches, n_pools),
            "pinned_hash": j["pinned_hash"], "pool_dir": _rel(pool_dir), "sample_rate": SAMPLE_RATE,
            "pool_batches": int(pool_batches), "budget_samples": int(budget), "seed": int(seed),
            "key_rule": KEY_RULE, "cut_rule": CUT_RULE, "shuffle_rule": SHUFFLE_RULE,
            "consumed_batches": int(consumed), "n_pools": int(n_pools), "n_occurrences": int(len(samples)),
            "n_batches": int(len(offsets) - 1), "journal_shards": j["shards"],
            "files": {n: {"sha256": _sha(b), "bytes": len(b), "count": counts[n]} for n, b in data.items()},
            "stats": stats(samples, order, offsets, consumed)}
    return body, data


def synthetic_key(samples: int) -> int:
    return -(-int(samples) * STRETCH[0] // STRETCH[1])


def check_stretch() -> None:
    """The synthetic key's bound follows the augmentation's slowest speed factor."""
    from fractions import Fraction

    from synthetic import augment
    if Fraction(1.0 / augment.SPEED[0]).limit_denominator(200) != Fraction(*STRETCH):
        raise PlanError(f"augment.SPEED {augment.SPEED} does not match the synthetic key's stretch {STRETCH}")


def clip_samples(clips) -> list[int]:
    """Whole-sample lengths of the clip pool's clips sorted by id (refused if not whole samples)."""
    out = []
    for c in sorted(clips.clips, key=lambda c: c["id"]):
        x = float(c["duration"]) * SAMPLE_RATE
        if abs(x - round(x)) > EXACT_TOLERANCE or round(x) < 1:
            raise PlanError(f"synthetic clip {c['id']}: duration {c['duration']!r} is not a whole number of samples")
        out.append(int(round(x)))
    return out


def compute_synthetic(pool_dir: Path, n_pools: int, clips, *, pool_batches: int = POOL_BATCHES,
                      budget: int = BUDGET_SAMPLES, seed: int = SEED, syn_seed: int | None = None,
                      share: float | None = None, config_id: str | None = None, engine: str = "numpy",
                      planner: Path | None = None, threads: int | None = None) -> tuple[dict, dict[str, bytes]]:
    """(plan.json body, file bytes) of the version 2 plan: the base plan's pools plus a synthetic draw per pool,
    cut together (module docstring)."""
    from synthetic import layout as slayout
    from synthetic import mixing
    check_stretch()
    syn_seed = slayout.SEED if syn_seed is None else int(syn_seed)
    share = slayout.SYNTHETIC_SHARE if share is None else float(share)
    base_body, base = compute(pool_dir, n_pools, pool_batches=pool_batches, budget=budget, seed=seed,
                              config_id=config_id, engine=engine, planner=planner, threads=threads)
    real = np.frombuffer(base["samples.u32"], dtype="<u4").astype(np.int64)
    rpools = np.frombuffer(base["pools.u32"], dtype="<u4").astype(np.int64)
    csamples = clip_samples(clips)
    ckeys = [synthetic_key(s) for s in csamples]
    if max(ckeys) > budget:
        raise PlanError(f"a synthetic clip's key {max(ckeys)} exceeds the budget {budget}")
    real_seconds = [int(real[a:b].sum()) / SAMPLE_RATE for a, b in zip(rpools[:-1], rpools[1:])]
    draws = mixing.plan(real_seconds, [s / SAMPLE_RATE for s in csamples], syn_seed, share)
    entries, keys, pools = [], [], [0]
    for p, (a, b) in enumerate(zip(rpools[:-1].tolist(), rpools[1:].tolist())):
        entries += list(range(a, b)) + [SYN_FLAG | c for c in draws[p]]
        keys += real[a:b].tolist() + [ckeys[c] for c in draws[p]]
        pools.append(len(entries))
    keys_a, pools_a = np.asarray(keys, dtype=np.int64), np.asarray(pools, dtype=np.int64)
    order, offsets, bpools = engine_cut(keys_a, pools_a, budget, engine, planner, threads)
    shuf = shuffle(bpools, seed)
    data = {"samples.u32": _u32(keys_a), "pools.u32": _u32(pools_a), "order.u32": _u32(order),
            "offsets.u32": _u32(offsets), "batch_pools.u32": _u32(bpools), "shuffle.u32": _u32(shuf),
            "entries.u32": np.asarray(entries, dtype=np.int64).astype("<u4").tobytes()}
    counts = {"samples.u32": len(keys), "pools.u32": len(pools), "order.u32": len(order), "offsets.u32": len(offsets),
              "batch_pools.u32": len(bpools), "shuffle.u32": len(shuf), "entries.u32": len(entries)}
    syn_s = [sum(csamples[c] for c in d) / SAMPLE_RATE for d in draws]
    ratio = [s / r for s, r in zip(syn_s, real_seconds) if r > 0]
    lengths = np.asarray([int(real[e]) if e < SYN_FLAG else csamples[e - SYN_FLAG] for e in entries], dtype=np.int64)
    st = stats(lengths, order, offsets, base_body["consumed_batches"])   # the unaugmented audio as served
    by_key = stats(keys_a, order, offsets, base_body["consumed_batches"])
    st["by_key"] = {"padding": by_key["padding"], "padded_seconds": by_key["padded_seconds"],
                    "note": "synthetic rows at their longest augmented length (the cut's keys)"}
    st["synthetic"] = {"seconds": sum(syn_s), "share_of_real": sum(syn_s) / sum(real_seconds),
                       "pool_share_min": min(ratio), "pool_share_max": max(ratio),
                       "max_pool_error_seconds": max(abs(s - share * r) for s, r in zip(syn_s, real_seconds))}
    n_syn = sum(len(d) for d in draws)
    body = {**base_body, "version": SYNTHETIC_VERSION,
            "plan_id": f"{base_body['plan_id']}-syn{str(clips.identity)[:12]}",
            "base_plan_sha256": _sha(_dumps(base_body)), "key_rule": f"real: {KEY_RULE}; synthetic: {SYN_KEY_RULE}",
            "n_entries": len(entries), "n_synthetic": n_syn, "n_batches": int(len(offsets) - 1),
            "synthetic": {"clip_pool": str(clips.identity), "n_clips": len(csamples), "clip_order": "sorted by id",
                          "seed": syn_seed, "share": share, "share_basis": SYN_SHARE_BASIS, "key_rule": SYN_KEY_RULE,
                          "draw_rule": SYN_DRAW_RULE},
            "files": {n: {"sha256": _sha(b), "bytes": len(b), "count": counts[n]} for n, b in data.items()},
            "stats": st}
    return body, data


@dataclasses.dataclass(frozen=True)
class Source:
    """An extra clip source of a multi-source version 2 plan (module docstring)."""
    name: str               # the rows' source tag
    prefix: str             # the rows' id prefix: "<prefix><clip id>"
    clips: object           # a clip store: .clips [{"id", "duration", ...}], .identity (synthetic.mixing.ClipPool)
    seed: int
    share: float | str      # a fraction of the real audio, or ONCE
    augment: bool           # rows augmented on the fly: keys at the longest augmented length

    def config(self) -> dict:
        return {"name": self.name, "prefix": self.prefix, "clip_pool": str(self.clips.identity), "seed": int(self.seed),
                "share": self.share if isinstance(self.share, str) else float(self.share), "augment": bool(self.augment)}

    def key(self, samples: int) -> int:
        return synthetic_key(samples) if self.augment else int(samples)


def extra_entry(source: int, clip: int) -> int:
    if not (0 <= clip < 1 << SOURCE_BITS and 0 <= source < 1 << (31 - SOURCE_BITS)):
        raise PlanError(f"extra entry ({source}, {clip}) does not fit the entry encoding")
    return SYN_FLAG | (source << SOURCE_BITS) | clip


def extra_parts(v: int) -> tuple[int, int]:
    """(source, clip) of an extra entry v >= SYN_FLAG."""
    x = int(v) - SYN_FLAG
    return x >> SOURCE_BITS, x & ((1 << SOURCE_BITS) - 1)


def sources_id(sources: list[Source]) -> str:
    return _sha(_dumps([s.config() for s in sources]))[:12]


def sources_suffix(sources: list[Source], version: int = SYNTHETIC_VERSION) -> str:
    """The plan id's suffix of a multi-source plan: "-x" (version 2) or "-v3x" (version 3) + sources_id."""
    if version not in MULTI_VERSIONS:
        raise PlanError(f"a multi-source plan is version {SYNTHETIC_VERSION} or {V3}, not {version}")
    return f"-{'v3' if version == V3 else ''}x{sources_id(sources)}"


def share_kind(share, version: int = SYNTHETIC_VERSION) -> tuple[str, float | int]:
    """(kind, value) of a source's share: ("fixed", fraction), ("once", 1), and in version 3 ("times", K) or ("about",
    K) for "times:K" / "about:K" (K a positive integer); PlanError for anything else."""
    if share == ONCE:
        return "once", 1
    if isinstance(share, str):
        m = re.fullmatch(rf"({TIMES}|{ABOUT}):([1-9][0-9]*)", share)
        if m is None or version != V3:
            raise PlanError(f"share {share!r} is not a share of a version {version} plan")
        return m.group(1), int(m.group(2))
    if isinstance(share, bool) or not isinstance(share, (int, float)) or not 0 < float(share) < 1:
        raise PlanError(f"share {share!r} is not a fraction in (0, 1)")
    return "fixed", float(share)


def journal_real_samples(pool_dir: Path) -> int:
    """The samples of every occurrence of the complete journal (a ONCE source's share is its seconds over these)."""
    end = journal_end(pool_dir)
    if end is None:
        raise PlanError("a source drawn once needs the complete journal (C1's manifest) to fix its share")
    return int(read_journal(pool_dir, end, with_rows=False)["samples"].astype(np.int64).sum())


def compute_sources(pool_dir: Path, n_pools: int, sources: list[Source], *, pool_batches: int = POOL_BATCHES,
                    budget: int = BUDGET_SAMPLES, seed: int = SEED, config_id: str | None = None,
                    engine: str = "numpy", planner: Path | None = None,
                    threads: int | None = None, version: int = SYNTHETIC_VERSION) -> tuple[dict, dict[str, bytes]]:
    """(plan.json body, file bytes) of the multi-source plan of `version`: 2, or 3 with its share kinds (module
    docstring)."""
    from synthetic import mixing
    if version not in MULTI_VERSIONS:
        raise PlanError(f"a multi-source plan is version {SYNTHETIC_VERSION} or {V3}, not {version}")
    if not sources:
        raise PlanError("a multi-source plan needs at least one source")
    kinds = [share_kind(s.share, version) for s in sources]
    if len({s.name for s in sources}) != len(sources) or len({s.prefix for s in sources}) != len(sources):
        raise PlanError("source names and prefixes must be distinct")
    if len(sources) > 1 << (31 - SOURCE_BITS):
        raise PlanError("too many sources for the entry encoding")
    if any(s.augment for s in sources):
        check_stretch()
    base_body, base = compute(pool_dir, n_pools, pool_batches=pool_batches, budget=budget, seed=seed,
                              config_id=config_id, engine=engine, planner=planner, threads=threads)
    real = np.frombuffer(base["samples.u32"], dtype="<u4").astype(np.int64)
    rpools = np.frombuffer(base["pools.u32"], dtype="<u4").astype(np.int64)
    real_seconds = [int(real[a:b].sum()) / SAMPLE_RATE for a, b in zip(rpools[:-1], rpools[1:])]
    end = journal_end(pool_dir)
    complete = end is not None and base_body["consumed_batches"] == end
    total_real = None
    draws, meta_sources = [], []
    for s_index, src in enumerate(sources):
        csamples = clip_samples(src.clips)
        ckeys = [src.key(x) for x in csamples]
        if max(ckeys) > budget:
            raise PlanError(f"source {src.name}: a clip's key {max(ckeys)} exceeds the budget {budget}")
        kind, k = kinds[s_index]
        if kind != "fixed":
            if total_real is None:
                total_real = int(real.sum()) if complete else journal_real_samples(pool_dir)
            share = k * sum(csamples) / total_real
            rule = {"once": "once: the source's seconds / the whole journal's real seconds",
                    TIMES: f"times: {k} x the source's seconds / the whole journal's real seconds, {k} passes",
                    ABOUT: f"about: {k} x the source's seconds / the whole journal's real seconds, the repeating "
                           "rule"}[kind]
        else:
            share, rule = float(src.share), "fixed"
        fixed_passes = {"once": 1, TIMES: k}.get(kind)
        d = mixing.plan(real_seconds, [x / SAMPLE_RATE for x in csamples], int(src.seed), share,
                        once=kind == "once", passes=fixed_passes if kind == TIMES else None,
                        take_rest=fixed_passes is not None and complete)
        draws.append(d)
        drawn = [c for pool in d for c in pool]
        syn_s = [sum(csamples[c] for c in pool) / SAMPLE_RATE for pool in d]
        ratio = [x / r for x, r in zip(syn_s, real_seconds) if r > 0]
        entry = {"config": src.config(), "share_value": share, "share_rule": rule,
                 "key_rule": SYN_KEY_RULE if src.augment else "samples", "n_clips": len(csamples),
                 "clip_order": "sorted by id", "draw_rule": SYN_DRAW_RULE, "draws": len(drawn),
                 "distinct": len(set(drawn)), "seconds": sum(syn_s),
                 "share_of_real": sum(syn_s) / sum(real_seconds),
                 "pool_share_min": min(ratio) if ratio else None,
                 "pool_share_max": max(ratio) if ratio else None,
                 "max_pool_error_seconds": max(abs(x - share * r) for x, r in zip(syn_s, real_seconds))}
        if version == V3:
            counts = np.bincount(np.asarray(drawn, dtype=np.int64), minlength=len(csamples))
            entry.update(kind=kind, k=k, draws_per_clip={"min": int(counts.min()), "max": int(counts.max())})
        meta_sources.append(entry)
        draws[-1] = (d, csamples, ckeys)
    entries, keys, lengths, pools = [], [], [], [0]
    for p, (a, b) in enumerate(zip(rpools[:-1].tolist(), rpools[1:].tolist())):
        entries += list(range(a, b))
        keys += real[a:b].tolist()
        lengths += real[a:b].tolist()
        for s_index, (d, csamples, ckeys) in enumerate(draws):
            entries += [extra_entry(s_index, c) for c in d[p]]
            keys += [ckeys[c] for c in d[p]]
            lengths += [csamples[c] for c in d[p]]
        pools.append(len(entries))
    keys_a, pools_a = np.asarray(keys, dtype=np.int64), np.asarray(pools, dtype=np.int64)
    order, offsets, bpools = engine_cut(keys_a, pools_a, budget, engine, planner, threads)
    shuf = shuffle(bpools, seed)
    data = {"samples.u32": _u32(keys_a), "pools.u32": _u32(pools_a), "order.u32": _u32(order),
            "offsets.u32": _u32(offsets), "batch_pools.u32": _u32(bpools), "shuffle.u32": _u32(shuf),
            "entries.u32": np.asarray(entries, dtype=np.int64).astype("<u4").tobytes()}
    counts = {"samples.u32": len(keys), "pools.u32": len(pools), "order.u32": len(order), "offsets.u32": len(offsets),
              "batch_pools.u32": len(bpools), "shuffle.u32": len(shuf), "entries.u32": len(entries)}
    st = stats(np.asarray(lengths, dtype=np.int64), order, offsets, base_body["consumed_batches"])
    by_key = stats(keys_a, order, offsets, base_body["consumed_batches"])
    st["by_key"] = {"padding": by_key["padding"], "padded_seconds": by_key["padded_seconds"],
                    "note": "augmented rows at their longest augmented length (the cut's keys)"}
    st["sources"] = {m["config"]["name"]: {k: m[k] for k in ("seconds", "share_of_real", "pool_share_min",
                                                               "pool_share_max", "max_pool_error_seconds")}
                     for m in meta_sources}
    body = {**base_body, "version": version,
            "plan_id": f"{base_body['plan_id']}{sources_suffix(sources, version)}",
            "base_plan_sha256": _sha(_dumps(base_body)),
            "key_rule": f"real: {KEY_RULE}; an augmented source: {SYN_KEY_RULE}; another source: samples",
            "entry_rule": ENTRY_RULE, "n_entries": len(entries), "n_extra": len(entries) - len(real),
            "n_batches": int(len(offsets) - 1), "journal_complete": complete, "sources": meta_sources,
            "files": {n: {"sha256": _sha(b), "bytes": len(b), "count": counts[n]} for n, b in data.items()},
            "stats": st}
    return body, data


def build(pool_dir: Path, out_dir: Path, n_pools: int, clips=None, sources: list[Source] | None = None,
          **kw) -> "Plan":
    """Write the plan into out_dir (atomically: a temporary sibling, then a rename); with `clips` (a
    synthetic.mixing.ClipPool) the version 2 plan of stage 2c, with `sources` the multi-source plan (version 2, or 3
    with kw version=3); `kw` are compute's, the cut engine included. Nothing is written unless the plan is computed (and, with engine="bend",
    cross-checked). An existing plan is kept if it is the same plan and refused otherwise."""
    out_dir = Path(out_dir)
    if clips is not None and sources is not None:
        raise PlanError("a plan has either stage 2c's synthetic clip pool or a list of sources")
    if sources is not None:
        body, data = compute_sources(pool_dir, n_pools, sources, **kw)
    else:
        body, data = compute(pool_dir, n_pools, **kw) if clips is None else compute_synthetic(pool_dir, n_pools, clips,
                                                                                              **kw)
    meta = _dumps(body)
    if out_dir.exists():
        existing = (out_dir / "plan.json").read_bytes() if (out_dir / "plan.json").exists() else None
        if existing != meta:
            raise PlanError(f"{out_dir} holds another plan; refusing to replace it")
        return Plan.load(out_dir)
    tmp = out_dir.with_name(f".{out_dir.name}.tmp{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for name, b in data.items():
        (tmp / name).write_bytes(b)
    (tmp / "plan.json").write_bytes(meta)
    for p in tmp.iterdir():
        with open(p, "rb") as handle:
            os.fsync(handle.fileno())
    os.replace(tmp, out_dir)
    return Plan.load(out_dir)


class Plan:
    """A verified plan directory (every file against plan.json; the structure against the contract)."""

    def __init__(self, directory: Path, meta: dict, arrays: dict[str, np.ndarray], sha256: str):
        self.dir, self.meta, self.sha256 = Path(directory), meta, sha256
        self.samples, self.pools = arrays["samples.u32"], arrays["pools.u32"]
        self.order, self.offsets = arrays["order.u32"], arrays["offsets.u32"]
        self.batch_pools, self.shuffle = arrays["batch_pools.u32"], arrays["shuffle.u32"]
        self.n_batches, self.n_pools = int(meta["n_batches"]), int(meta["n_pools"])
        self.consumed_batches, self.pool_batches = int(meta["consumed_batches"]), int(meta["pool_batches"])
        self.budget = int(meta["budget_samples"])
        self.version = int(meta["version"])
        self.entries = arrays.get("entries.u32")          # version 2; version 1: entry = sequence number
        self.n_entries = int(meta.get("n_entries", meta["n_occurrences"]))
        if self.entries is None:
            self.real_pools = self.pools
        else:                                             # per pool, its real sequence numbers [real_pools[p], ...)
            real = self.entries < SYN_FLAG
            counts = [int(real[a:b].sum()) for a, b in zip(self.pools[:-1], self.pools[1:])]
            self.real_pools = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    @classmethod
    def load(cls, directory: Path) -> "Plan":
        d = Path(directory)
        raw = (d / "plan.json").read_bytes()
        meta = json.loads(raw)
        if meta.get("kind") != KIND or meta.get("version") not in (VERSION, SYNTHETIC_VERSION, V3):
            raise PlanError(f"{d}: not a version {VERSION}, {SYNTHETIC_VERSION} or {V3} batch plan")
        if meta["version"] == V3 and "sources" not in meta:
            raise PlanError(f"{d}: a version {V3} plan without sources")
        names = FILES if meta["version"] == VERSION else FILES_V2
        if set(meta["files"]) != set(names):
            raise PlanError(f"{d}: plan.json lists files {sorted(meta['files'])}")
        arrays = {}
        for name in names:
            b = (d / name).read_bytes()
            f = meta["files"][name]
            if len(b) != f["bytes"] or _sha(b) != f["sha256"] or len(b) != 4 * f["count"]:
                raise PlanError(f"{d / name}: differs from plan.json")
            arrays[name] = np.frombuffer(b, dtype="<u4").astype(np.int64)
        plan = cls(d, meta, arrays, _sha(raw))
        plan.check_structure()
        return plan

    def check_structure(self) -> None:
        n, nb, npl = self.n_entries, self.n_batches, self.n_pools
        o, off, bp, sh, pools = self.order, self.offsets, self.batch_pools, self.shuffle, self.pools
        if (len(self.samples), len(o), len(off), len(bp), len(sh), len(pools)) != (n, n, nb + 1, npl + 1, nb, npl + 1):
            raise PlanError("plan arrays have the wrong lengths")
        if off[0] != 0 or off[-1] != n or np.any(np.diff(off) < 1):
            raise PlanError("batch offsets are not increasing from 0 to n")
        if pools[0] != 0 or pools[-1] != n or np.any(np.diff(pools) < 0):
            raise PlanError("pool offsets are not 0 .. n")
        if bp[0] != 0 or bp[-1] != nb or np.any(np.diff(bp) < 0):
            raise PlanError("batch-pool offsets are not 0 .. n_batches")
        if not np.array_equal(np.sort(o), np.arange(n)):
            raise PlanError("order is not a permutation of the sequence")
        for p in range(npl):
            lo, hi = int(bp[p]), int(bp[p + 1])
            if not np.array_equal(np.sort(sh[lo:hi]), np.arange(lo, hi)):
                raise PlanError(f"pool {p}: shuffle is not a permutation of its batches")
            a, b = int(off[lo]), int(off[hi])
            if (a, b) != (int(pools[p]), int(pools[p + 1])):
                raise PlanError(f"pool {p}: its batches do not hold exactly its occurrences")
        if self.entries is not None:
            if len(self.entries) != n:
                raise PlanError("entries.u32 has the wrong length")
            real = self.entries[self.entries < SYN_FLAG]
            if not np.array_equal(real, np.arange(int(self.meta["n_occurrences"]))) or \
                    len(self.entries) - len(real) != int(self.meta.get("n_synthetic", self.meta.get("n_extra", -1))):
                raise PlanError("entries do not hold every real occurrence once, in order, plus the draw")

    @property
    def multi_source(self) -> bool:
        return "sources" in self.meta

    def source_configs(self) -> list[dict]:
        return [m["config"] for m in self.meta.get("sources", [])]

    def verify_journal(self, pool_dir: Path, clips=None, sources: list[Source] | None = None) -> dict:
        """Re-read the journal: the same shards (SHA-256) and keys, and the cut and shuffle recomputed from them
        equal the plan's (version 2: the whole plan recomputed from the journal and the clip pool `clips`, or the
        `sources` of a multi-source plan, equals this one). Returns read_journal's rows (ids, sources, durations)."""
        if self.version in MULTI_VERSIONS and self.multi_source:
            if sources is None or [s.config() for s in sources] != self.source_configs():
                raise PlanError("a multi-source plan needs the sources it was drawn from")
            body, data = compute_sources(pool_dir, self.n_pools, sources, pool_batches=self.pool_batches,
                                         budget=self.budget, seed=int(self.meta["seed"]), config_id=self.config_id(),
                                         version=self.version)
            if _dumps(body) != _dumps(self.meta) or any(_sha(b) != self.meta["files"][n]["sha256"]
                                                        for n, b in data.items()):
                raise PlanError("the plan differs from the one the journal and the sources give")
            return read_journal(pool_dir, self.consumed_batches, with_rows=True)
        if sources is not None:
            raise PlanError("only a multi-source plan takes sources")
        if self.version == SYNTHETIC_VERSION:
            if clips is None or str(clips.identity) != self.meta["synthetic"]["clip_pool"]:
                raise PlanError("a version 2 plan needs the clip pool it was drawn from")
            s = self.meta["synthetic"]
            body, data = compute_synthetic(pool_dir, self.n_pools, clips, pool_batches=self.pool_batches,
                                           budget=self.budget, seed=int(self.meta["seed"]), syn_seed=s["seed"],
                                           share=s["share"], config_id=self.config_id())
            if _dumps(body) != _dumps(self.meta) or any(_sha(b) != self.meta["files"][n]["sha256"]
                                                        for n, b in data.items()):
                raise PlanError("the plan differs from the one the journal and clip pool give")
            return read_journal(pool_dir, self.consumed_batches, with_rows=True)
        j = read_journal(pool_dir, self.consumed_batches, with_rows=True)
        if j["shards"] != self.meta["journal_shards"] or j["pinned_hash"] != self.meta["pinned_hash"]:
            raise PlanError("the journal's shards differ from the plan's")
        if not np.array_equal(j["samples"], self.samples):
            raise PlanError("the journal's clip lengths differ from the plan's")
        if not np.array_equal(pool_starts(j["batch"], self.n_pools, self.pool_batches), self.pools):
            raise PlanError("the journal's pools differ from the plan's")
        order, offsets, bpools = cut(self.samples, self.pools, self.budget)
        if not (np.array_equal(order, self.order) and np.array_equal(offsets, self.offsets)
                and np.array_equal(bpools, self.batch_pools)
                and np.array_equal(shuffle(bpools, int(self.meta["seed"])), self.shuffle)):
            raise PlanError("the plan's cut or shuffle differs from the contract's")
        return j

    def config_id(self) -> str:
        base = re.sub(r"-(?:v3)?x[0-9a-f]{12}$", "", self.meta["plan_id"]).split("-syn")[0]
        return base[:-len(f"-p{self.pool_batches}-n{self.n_pools}")]

    def batch(self, t: int) -> np.ndarray:
        """The rows of training position t in cut order: journal sequence numbers (version 2: entries, a value
        >= SYN_FLAG being synthetic clip value - SYN_FLAG)."""
        b = int(self.shuffle[t])
        rows = self.order[int(self.offsets[b]):int(self.offsets[b + 1])]
        return rows if self.entries is None else self.entries[rows]

    def pool_of(self, t: int) -> int:
        return int(np.searchsorted(self.batch_pools, t, side="right") - 1)

    def pool_start(self, p: int) -> int:
        """The training position of pool p's first planned batch (p = n_pools: n_batches)."""
        return int(self.batch_pools[p])

    def identity(self) -> dict:
        return {"plan_id": self.meta["plan_id"], "plan_sha256": self.sha256, "version": self.version,
                "n_pools": self.n_pools,
                "consumed_batches": self.consumed_batches, "n_batches": self.n_batches,
                "pool_batches": self.pool_batches, "budget_samples": self.budget, "seed": self.meta["seed"]}


def is_prefix(short: Plan, long: Plan) -> bool:
    """True when `short` is the plan of `long`'s first pools (same journal, P, budget and seed)."""
    keys = ("pinned_hash", "pool_batches", "budget_samples", "seed", "sample_rate")
    if any(short.meta[k] != long.meta[k] for k in keys) or short.n_pools > long.n_pools:
        return False
    if short.version != long.version or short.meta.get("synthetic") != long.meta.get("synthetic"):
        return False
    if short.source_configs() != long.source_configs() or \
            [m.get("share_value") for m in short.meta.get("sources", [])] != \
            [m.get("share_value") for m in long.meta.get("sources", [])]:
        return False
    n, nb = short.n_entries, short.n_batches
    if short.entries is not None and not np.array_equal(short.entries, long.entries[:n]):
        return False
    return (np.array_equal(short.samples, long.samples[:n]) and np.array_equal(short.pools, long.pools[:short.n_pools + 1])
            and np.array_equal(short.order, long.order[:n]) and np.array_equal(short.offsets, long.offsets[:nb + 1])
            and np.array_equal(short.batch_pools, long.batch_pools[:short.n_pools + 1])
            and np.array_equal(short.shuffle, long.shuffle[:nb]))


# ----------------------------------------------------------------------------- CLI

def _c1_and_config() -> tuple[Path, str, dict]:
    from labeling import config
    from labeling.__main__ import c1_dir, pinned
    pin = pinned()
    if float(pin["batch"]["batch_seconds"]) * SAMPLE_RATE != BUDGET_SAMPLES:
        raise SystemExit(f"the pinned stream batches {pin['batch']['batch_seconds']} s, not {BUDGET_SAMPLES / SAMPLE_RATE}")
    return c1_dir(), config.config_id(pin), pin


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m training.batchplan")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="the production plan over pools 0 .. N-1")
    b.add_argument("--pools", type=int, required=True)
    b.add_argument("--out", type=Path, help="a smoke plan directory under $RELABEL_ROOT/smoke/")
    b.add_argument("--synthetic-run", help="the version 2 plan of stage 2c's synthetic arms, drawn from this "
                                           "generation run's clip pool (synthetic/runs/<run>)")
    b.add_argument("--cut", choices=ENGINES, default="numpy", help="the cut engine (default numpy; module docstring)")
    b.add_argument("--planner", type=Path, help="--cut bend: the Bend planner binary "
                                                "(default finetune/bend-planner/build/planner)")
    b.add_argument("--threads", type=int, help="--cut bend: the planner's runtime threads")
    s = sub.add_parser("stats")
    s.add_argument("--plan", type=Path)
    s.add_argument("--pools", type=int)
    s.add_argument("--synthetic-run", help="the clip pool of a version 2 plan")
    sub.add_parser("materialize", add_help=False)
    sub.add_parser("probe", add_help=False)
    args, rest = ap.parse_known_args(argv)
    if args.cmd == "materialize":
        from training import poolaudio
        return poolaudio.main(rest)
    if args.cmd == "probe":
        from training import replay_probe
        return replay_probe.main(rest)
    if rest:
        ap.error(f"unrecognized arguments: {rest}")
    if args.cmd == "build" and args.cut != "bend" and (args.planner is not None or args.threads is not None):
        ap.error("--planner and --threads need --cut bend")
    layout.require_mount()
    pool, cid, _ = _c1_and_config()
    if args.cmd == "build":
        if args.out is not None and layout.is_production(args.out):
            raise SystemExit("--out is for smoke plans under $RELABEL_ROOT/smoke/")
        clips = None
        if args.synthetic_run:
            from synthetic import mixing
            clips = mixing.ClipPool.load(args.synthetic_run)
        try:
            plan = build(pool, args.out or production_dir(cid, args.pools, clips=clips), args.pools, clips=clips,
                         config_id=cid, engine=args.cut, planner=args.planner, threads=args.threads)
        except PlanError as e:                            # nothing was written
            raise SystemExit(f"batchplan build: {e}")
        print(json.dumps({"dir": _rel(plan.dir), "cut": args.cut, **plan.identity(), "stats": plan.meta["stats"]},
                         indent=1))
        return 0
    clips = None
    if args.synthetic_run:
        from synthetic import mixing
        clips = mixing.ClipPool.load(args.synthetic_run)
    plan = Plan.load(args.plan or production_dir(cid, args.pools, clips=clips))
    plan.verify_journal(pool, clips=clips)
    print(json.dumps({**plan.identity(), "stats": plan.meta["stats"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
