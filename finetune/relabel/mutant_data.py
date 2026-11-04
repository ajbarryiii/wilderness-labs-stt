"""Mutant (encoder probe 2) training data: the human-transcribed clips, the held-out slice, the batch plan and a
compact audio store (mutant plan "Training data" and "Training protocol").

    ./python mutant_data.py select [--smoke-shards N] [--force]     the clip table and the batch plan (CPU)
    ./python mutant_data.py audio [--smoke-shards N] [--budget-minutes M]   the pool audio store (CPU and disk)
    ./python mutant_data.py stats [--smoke-shards N]

Clips. Every unique clip of LibriSpeech, VoxPopuli English train and AMI IHM train in the C1 replay store
($RELABEL_ROOT/pool-audio/<config id>/shards/<k>.audio.parquet: one row per id, at its first occurrence) whose
reference_kind is "human", after the global denylist ($RELABEL_ROOT/pool/denylist.json; it lists no recording of these
three sources, which select checks and records). The text is the stored reference, verbatim, tokenized with B0's
tokenizer (model.tokenizer.text_to_ids, as train.tokenize). A clip whose tokens include the tokenizer's unknown id is
not covered and is dropped from every arm (counted and listed). Hygiene: no training id is an H4 dev id, and no
LibriSpeech speaker or AMI meeting of the H4 dev sets is a training one.

Recordings (the held-out unit): LibriSpeech the chapter (<speaker>-<chapter>), VoxPopuli the event audio the segment
was cut from (the id before "-en_": <date>-<time>-PLENARY, with an agenda-item suffix when it has one), AMI the meeting
(all headsets of a meeting are one recording).

Held-out slice: per source, the recordings ranked by SHA-256(HELDOUT_SALT \\0 source \\0 recording) (then by name) are
held out in rank order while the source's held-out clips are fewer than round(1% x its covered clips). So 1% of the
clips (per source and overall, to within one recording), drawn by recording with a fixed seed (the salt). Held-out
clips are never trained on; they give the held-out metric.

Batch plan (one for every arm and run; its plan.json SHA-256 is the plan identity):
- sequence: the training clips in C1 order (shard, row): the stream's random order;
- pools: pool p holds the clips of C1 shards [SHARDS_PER_POOL p, SHARDS_PER_POOL (p + 1));
- cut: training/batchplan.py's cut per pool (sort by samples, ties by sequence number; a new batch when
  (count + 1) x samples > BUDGET_SAMPLES, 600 s of padded 16 kHz audio, the stage 2b and 3 batch);
- epoch e serves every batch once: the pools in the order numpy default_rng([PLAN_SEED, 3, e]).permutation(P), each
  pool's batches in the order default_rng([PLAN_SEED, 2, p, e]).permutation(n_p). Global step t (0-based) serves
  epoch t // n_batches at position t % n_batches. Pilots take the first 3,000 positions of epoch 0, the main run
  epochs 0 and 1, the extension epoch 2.

Audio store ($RELABEL_ROOT/encoder-probe/mutant/audio/pools/<ppppp>.flac + .json): per pool, the FLAC bytes of
every selected clip of the pool's shards (training and held-out), C1's bytes unchanged, concatenated in C1 order;
the .json marker (written last) lists id, offset, length, samples and C1's pcm_sha256 per clip, and the file's
SHA-256. Every decoded clip is checked against C1's pcm_sha256. Pools are written in order, each committed once, so
the pass resumes at the first uncommitted pool.

Layout: production $RELABEL_ROOT/encoder-probe/mutant/{data,audio}; --smoke-shards N (the first N C1 shards) writes
only under relabel_paths.SMOKE/mutant/.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

import relabel_paths
from labeling import shards as lshards
from training import batchplan

PLAN = "mutant plan"
SOURCES = ("librispeech", "voxpopuli", "ami")
HUMAN = "human"
HELDOUT_FRACTION = 0.01
HELDOUT_SALT = "relabel-mutant-heldout-v1"
SHARDS_PER_POOL = 20
BUDGET_SAMPLES = batchplan.BUDGET_SAMPLES          # 600 s at 16 kHz
SAMPLE_RATE = batchplan.SAMPLE_RATE
PLAN_SEED = 20261104
PLAN_KIND, PLAN_VERSION = "relabel-mutant-plan", 1
META_COLUMNS = ("id", "source", "duration", "samples", "reference", "reference_kind", "pcm_sha256")
_REC = {"librispeech": re.compile(r"^librispeech:((\d+)-\d+)-\d+$"),
        "voxpopuli": re.compile(r"^voxpopuli:(\d{8}-\d{4}-PLENARY(?:-[A-Za-z0-9]+)?)-en_"),
        "ami": re.compile(r"^ami:AMI_([A-Za-z0-9]+)_")}
RULES = {
    "clips": "C1 audio-shard rows (first occurrences) with source in librispeech/voxpopuli/ami and reference_kind "
             "human, minus denylisted recordings",
    "coverage": "B0 tokenizer text_to_ids of the stored reference has no unknown id",
    "recording": "librispeech <speaker>-<chapter>; voxpopuli the event id before '-en_' (<date>-<time>-PLENARY[-<suffix>]); "
                 "ami the meeting",
    "heldout": "per source, recordings by sha256(salt \\0 source \\0 recording) then name, held out while the "
               "source's held-out clips < round(fraction x covered clips)",
    "pools": "pool p = training clips of C1 shards [G p, G (p + 1)), in C1 order (shard, row)",
    "cut": "training/batchplan.cut per pool on samples with the budget (ties by sequence number)",
    "epochs": "epoch e: pools by default_rng([seed, 3, e]).permutation(P); pool p's batches by "
              "default_rng([seed, 2, p, e]).permutation(n_p)",
}


class DataError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- layout

@dataclasses.dataclass(frozen=True)
class Layout:
    """Where mutant writes. Production: $RELABEL_ROOT/encoder-probe/...; smoke: relabel_paths.SMOKE/mutant/..."""
    root: Path            # data and audio (mutant/), runs
    features: Path        # features/<arm>/
    records: Path         # decoder/<arm>/<set>.json
    smoke_shards: int | None

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def audio(self) -> Path:
        return self.root / "audio"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def smoke(self) -> bool:
        return self.smoke_shards is not None


def production_root() -> Path:
    return relabel_paths.ARTIFACTS / "encoder-probe"


def layout(smoke_shards: int | None = None) -> Layout:
    """The production layout, or with smoke_shards the smoke layout (only under relabel_paths.SMOKE)."""
    if smoke_shards is None:
        base = production_root()
        return Layout(base / "mutant", base / "features", base / "decoder", None)
    if smoke_shards <= 0:
        raise ValueError("--smoke-shards must be positive")
    base = relabel_paths.SMOKE / "mutant" / f"shards{int(smoke_shards)}"
    out = Layout(base / "mutant", base / "features", base / "decoder", int(smoke_shards))
    for p in (out.root, out.features, out.records):
        assert p.is_relative_to(relabel_paths.SMOKE) and not p.is_relative_to(production_root())
    return out


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def _dumps(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=1) + "\n").encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ----------------------------------------------------------------------------- clips

def recording_of(source: str, item_id: str) -> str:
    m = _REC[source].match(item_id) if source in _REC else None
    if m is None:
        raise DataError(f"{source}: cannot read the recording of {item_id!r}")
    return m.group(1)


def denied_recordings(deny: dict) -> dict[str, set[str]]:
    """{source: recordings} of the global denylist (denylist.json "recordings")."""
    return {s: set(v) for s, v in (deny.get("recordings") or {}).items()}


def select_rows(rows, denied: dict[str, set[str]]) -> tuple[list[dict], dict]:
    """The selected clips (source in SOURCES, human reference, recording not denylisted) and counts per source.
    A denylist entry for one of SOURCES must be a recording in this module's key (recording_of)."""
    counts = {s: {"rows": 0, "human": 0, "denylisted": 0, "selected": 0} for s in SOURCES}
    out, seen = [], set()
    for r in rows:
        s = r["source"]
        if s not in SOURCES:
            continue
        c = counts[s]
        c["rows"] += 1
        if r["reference_kind"] != HUMAN or r["reference"] is None:
            continue
        c["human"] += 1
        if r["id"] in seen:
            raise DataError(f"duplicate id {r['id']} in C1's audio shards (one row per id expected)")
        seen.add(r["id"])
        rec = recording_of(s, r["id"])
        if rec in denied.get(s, ()):
            c["denylisted"] += 1
            continue
        c["selected"] += 1
        out.append({**r, "recording": rec})
    return out, counts


def coverage(tokenizer, text: str) -> tuple[list[int], bool]:
    """(token ids of B0's tokenizer, covered: no unknown id)."""
    ids = [int(i) for i in tokenizer.text_to_ids(text)]
    return ids, unk_id(tokenizer) not in ids


def unk_id(tokenizer) -> int:
    for owner in (tokenizer, getattr(tokenizer, "tokenizer", None)):
        value = getattr(owner, "unk_id", None)
        if value is not None:
            value = value() if callable(value) else value
            if value is not None and int(value) >= 0:
                return int(value)
    raise DataError("the tokenizer has no unknown id")


def recording_rank(source: str, recording: str, salt: str = HELDOUT_SALT) -> str:
    return hashlib.sha256(f"{salt}\0{source}\0{recording}".encode()).hexdigest()


def heldout_recordings(clips: list[dict], fraction: float = HELDOUT_FRACTION, salt: str = HELDOUT_SALT
                       ) -> dict[str, set[str]]:
    """{source: held-out recordings} by the module's rule."""
    out = {}
    for s in SOURCES:
        per: dict[str, int] = {}
        for c in clips:
            if c["source"] == s:
                per[c["recording"]] = per.get(c["recording"], 0) + 1
        target = round(fraction * sum(per.values()))
        held, n = set(), 0
        for rec in sorted(per, key=lambda r: (recording_rank(s, r, salt), r)):
            if n >= target:
                break
            held.add(rec)
            n += per[rec]
        out[s] = held
    return out


# ----------------------------------------------------------------------------- plan

@dataclasses.dataclass
class Plan:
    meta: dict
    ids: list[str]            # training clips in sequence order
    samples: np.ndarray       # [n] samples (the cut key)
    pools: np.ndarray         # [P + 1] pool offsets into the sequence
    order: np.ndarray         # [n] sequence numbers in cut order
    offsets: np.ndarray       # [n_batches + 1] batch offsets into order
    batch_pools: np.ndarray   # [P + 1] pool offsets into the batch list
    sha256: str

    @property
    def n_batches(self) -> int:
        return len(self.offsets) - 1

    @property
    def n_pools(self) -> int:
        return len(self.pools) - 1

    def batch(self, b: int) -> np.ndarray:
        """Sequence numbers of cut batch b."""
        return self.order[self.offsets[b]:self.offsets[b + 1]]

    def pool_of_batch(self, b: int) -> int:
        return int(np.searchsorted(self.batch_pools, b, side="right") - 1)

    def epoch_order(self, epoch: int) -> np.ndarray:
        return epoch_order(self.batch_pools, int(self.meta["seed"]), epoch)

    def position(self, step: int) -> tuple[int, int, int]:
        """(epoch, position in the epoch, cut batch) of global step `step` (0-based)."""
        e, j = divmod(int(step), self.n_batches)
        return e, j, int(self._order(e)[j])

    def _order(self, epoch: int) -> np.ndarray:
        cache = self.__dict__.setdefault("_orders", {})
        if epoch not in cache:
            cache[epoch] = self.epoch_order(epoch)
        return cache[epoch]

    def identity(self) -> dict:
        return {"sha256": self.sha256, "n_batches": self.n_batches, "n_pools": self.n_pools,
                "n_clips": len(self.ids), "seed": self.meta["seed"], "budget_samples": self.meta["budget_samples"]}


def epoch_order(batch_pools: np.ndarray, seed: int, epoch: int) -> np.ndarray:
    """The cut batches in the order epoch `epoch` serves them (module docstring)."""
    n_pools = len(batch_pools) - 1
    out = []
    for p in np.random.default_rng([int(seed), 3, int(epoch)]).permutation(n_pools).tolist():
        lo, hi = int(batch_pools[p]), int(batch_pools[p + 1])
        out.append(lo + np.random.default_rng([int(seed), 2, p, int(epoch)]).permutation(hi - lo))
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def build_plan(train: list[dict], n_shards: int, shards_per_pool: int = SHARDS_PER_POOL,
               budget: int = BUDGET_SAMPLES, seed: int = PLAN_SEED, extra: dict | None = None) -> Plan:
    """The plan over training clips (dicts with id, shard, row, samples), pools of shards_per_pool C1 shards."""
    seq = sorted(train, key=lambda c: (int(c["shard"]), int(c["row"])))
    if len({c["id"] for c in seq}) != len(seq):
        raise DataError("duplicate training ids")
    n_pools = -(-int(n_shards) // shards_per_pool)
    pool_of = np.array([int(c["shard"]) // shards_per_pool for c in seq], dtype=np.int64)
    if len(pool_of) and (pool_of.min() < 0 or pool_of.max() >= n_pools):
        raise DataError("a training clip's shard is outside the plan's shards")
    pools = np.searchsorted(pool_of, np.arange(n_pools + 1), side="left").astype(np.int64)
    samples = np.array([int(c["samples"]) for c in seq], dtype=np.int64)
    order, offsets, batch_pools = batchplan.cut(samples, pools, budget)
    arrays = {"samples": samples, "pools": pools, "order": order, "offsets": offsets, "batch_pools": batch_pools}
    ids = [c["id"] for c in seq]
    files = {name: _sha(np.ascontiguousarray(a, dtype="<i8").tobytes()) for name, a in arrays.items()}
    files["ids"] = _sha("\n".join(ids).encode())
    meta = {"kind": PLAN_KIND, "version": PLAN_VERSION, "seed": int(seed), "budget_samples": int(budget),
            "sample_rate": SAMPLE_RATE, "shards_per_pool": int(shards_per_pool), "n_shards": int(n_shards),
            "n_pools": n_pools, "n_clips": len(ids), "n_batches": len(offsets) - 1, "rules": RULES, "files": files,
            **(extra or {})}
    return Plan(meta, ids, samples, pools, order, offsets, batch_pools, _sha(_dumps(meta)))


def save_plan(plan: Plan, directory: Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "plan-arrays.npz", samples=plan.samples, pools=plan.pools, order=plan.order,
             offsets=plan.offsets, batch_pools=plan.batch_pools)
    (directory / "plan-ids.txt").write_text("\n".join(plan.ids) + "\n")
    (directory / "plan.json").write_bytes(_dumps(plan.meta))


def load_plan(directory: Path) -> Plan:
    """The saved plan, after checking every array and the ids against plan.json's SHA-256s."""
    directory = Path(directory)
    raw = (directory / "plan.json").read_bytes()
    meta = json.loads(raw)
    with np.load(directory / "plan-arrays.npz") as z:
        arrays = {k: z[k].astype(np.int64) for k in ("samples", "pools", "order", "offsets", "batch_pools")}
    ids = (directory / "plan-ids.txt").read_text().splitlines()
    got = {name: _sha(np.ascontiguousarray(a, dtype="<i8").tobytes()) for name, a in arrays.items()}
    got["ids"] = _sha("\n".join(ids).encode())
    if got != meta["files"]:
        raise DataError(f"{directory}: plan files differ from plan.json")
    return Plan(meta, ids, arrays["samples"], arrays["pools"], arrays["order"], arrays["offsets"],
                arrays["batch_pools"], _sha(raw))


# ----------------------------------------------------------------------------- clip table

TABLE_COLUMNS = ("id", "source", "recording", "shard", "row", "duration", "samples", "text", "pcm_sha256", "tokens",
                 "split")


def write_table(path: Path, clips: list[dict]) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq
    cols = {k: [c[k] for c in clips] for k in TABLE_COLUMNS}
    table = pa.table({**cols, "tokens": pa.array(cols["tokens"], type=pa.list_(pa.int32()))})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    pq.write_table(table, tmp)
    os.replace(tmp, path)
    return lshards.sha256_file(path)


def read_table(path: Path) -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


@dataclasses.dataclass
class Data:
    """The selected data of one layout: clip table, plan and summary."""
    layout: Layout
    clips: dict[str, dict]
    plan: Plan
    summary: dict

    @classmethod
    def load(cls, lay: Layout) -> "Data":
        summary = json.loads((lay.data / "data.json").read_text())
        path = lay.data / "clips.parquet"
        if lshards.sha256_file(path) != summary["clips_sha256"]:
            raise DataError(f"{path} differs from data.json")
        plan = load_plan(lay.data)
        if plan.sha256 != summary["plan_sha256"]:
            raise DataError("plan.json differs from data.json")
        clips = {c["id"]: c for c in read_table(path)}
        return cls(lay, clips, plan, summary)

    def heldout_ids(self) -> list[str]:
        return sorted(i for i, c in self.clips.items() if c["split"] == "heldout")

    def pool_ids(self, p: int) -> list[str]:
        """Every clip of pool p's shards (training and held-out), C1 order: the audio and feature stores' order."""
        if "_by_pool" not in self.__dict__:
            g = self.plan.meta["shards_per_pool"]
            by_pool: dict[int, list[dict]] = {}
            for c in self.clips.values():
                by_pool.setdefault(int(c["shard"]) // g, []).append(c)
            self.__dict__["_by_pool"] = {q: [c["id"] for c in sorted(v, key=lambda c: (c["shard"], c["row"]))]
                                         for q, v in by_pool.items()}
        return list(self.__dict__["_by_pool"].get(int(p), []))


# ----------------------------------------------------------------------------- select

def c1_store() -> tuple[Path, dict]:
    """C1's directory (the pinned configuration's) and its status (complete, shard count)."""
    from labeling.__main__ import c1_dir
    d = c1_dir()
    status = json.loads((d / "status.json").read_text())
    return d, status


def read_shard_meta(c1: Path, k: int) -> list[dict]:
    """The metadata rows of SOURCES in C1 audio shard k, with their shard and row index (no audio)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    t = pq.read_table(lshards.paths(c1, k)["audio"], columns=list(META_COLUMNS))
    t = t.append_column("row", pa.array(np.arange(t.num_rows, dtype=np.int64)))
    rows = t.filter(pc.is_in(t["source"], value_set=pa.array(list(SOURCES)))).to_pylist()
    for r in rows:
        r["shard"] = k
    return rows


def dev_hygiene(clips: list[dict]) -> dict:
    """No training id is an H4 dev id; no H4 dev LibriSpeech speaker or AMI meeting is a training one."""
    import evaluate
    from training import devsplit
    dev_ids, speakers, meetings = set(), set(), set()
    for name, path in devsplit.dev_manifests().items():
        for r in evaluate.read_manifest(path):
            dev_ids.add(r["id"])
            if name.startswith("librispeech"):
                speakers.add(devsplit.cluster_of(name, r["id"]))
            elif name == "ami_dev":
                meetings.add(devsplit.cluster_of(name, r["id"]))
    shared_ids = sorted(c["id"] for c in clips if c["id"] in dev_ids)
    shared_speakers = sorted({c["recording"].split("-")[0] for c in clips if c["source"] == "librispeech"} & speakers)
    shared_meetings = sorted({c["recording"] for c in clips if c["source"] == "ami"} & meetings)
    return {"passed": not (shared_ids or shared_speakers or shared_meetings), "shared_ids": shared_ids[:20],
            "shared_librispeech_speakers": shared_speakers, "shared_ami_meetings": shared_meetings,
            "dev_ids": len(dev_ids), "dev_librispeech_speakers": len(speakers), "dev_ami_meetings": len(meetings)}


def b0_tokenizer():
    import evaluate
    model = evaluate.load_pretrained("cpu")
    return model.tokenizer, evaluate.describe_source("pretrained", None, "B0 tokenizer")["sha256"]


def summarize_hours(clips: list[dict]) -> dict:
    out = {}
    for s in SOURCES:
        rows = [c for c in clips if c["source"] == s]
        out[s] = {"clips": len(rows), "hours": sum(c["duration"] for c in rows) / 3600,
                  "recordings": len({c["recording"] for c in rows})}
    out["total"] = {"clips": len(clips), "hours": sum(c["duration"] for c in clips) / 3600}
    return out


def select(lay: Layout, force: bool = False, tokenizer=None, tokenizer_sha256: str | None = None,
           c1: Path | None = None, n_shards: int | None = None, hygiene: bool = True) -> dict:
    """Build and write the clip table, the plan and data.json (refuses to replace different data unless force)."""
    relabel_paths.require_mount()
    if c1 is None:
        c1, status = c1_store()
        if not status.get("complete"):
            raise DataError(f"C1 at {c1} is not complete")
        n_shards = int(status["shards_total"])
    if lay.smoke:
        n_shards = min(int(n_shards), lay.smoke_shards)
    deny_path = relabel_paths.ARTIFACTS / "pool" / "denylist.json"
    deny = json.loads(deny_path.read_text())
    denied = denied_recordings(deny)
    for s in SOURCES:
        if denied.get(s):
            raise DataError(f"the denylist lists {len(denied[s])} {s} recordings; recording_of's key must be "
                            "checked against the denylist's key for that source first")
    t0 = time.time()

    def rows():
        for k in range(int(n_shards)):
            if k % 200 == 0:
                print(f"select: shard {k}/{n_shards} ({time.time() - t0:.0f} s)", flush=True)
            yield from read_shard_meta(c1, k)

    selected, counts = select_rows(rows(), denied)
    if tokenizer is None:
        tokenizer, tokenizer_sha256 = b0_tokenizer()
    uncovered, empty, kept = [], 0, []
    for c in selected:
        ids, ok = coverage(tokenizer, c["reference"])
        if not ok:
            uncovered.append({"id": c["id"], "text": c["reference"]})
            counts[c["source"]].setdefault("uncovered", 0)
            counts[c["source"]]["uncovered"] += 1
            continue
        empty += not ids
        kept.append({"id": c["id"], "source": c["source"], "recording": c["recording"], "shard": c["shard"],
                     "row": c["row"], "duration": float(c["duration"]), "samples": int(c["samples"]),
                     "text": c["reference"], "pcm_sha256": c["pcm_sha256"], "tokens": ids})
    held = heldout_recordings(kept)
    for c in kept:
        c["split"] = "heldout" if c["recording"] in held[c["source"]] else "train"
    train = [c for c in kept if c["split"] == "train"]
    heldout = [c for c in kept if c["split"] == "heldout"]
    hyg = dev_hygiene(kept) if hygiene else {"passed": None, "skipped": True}
    if hyg["passed"] is False:
        raise DataError(f"dev hygiene failed: {hyg}")
    plan = build_plan(train, n_shards, shards_per_pool=SHARDS_PER_POOL, extra={"clip_ids_sha256": _sha("\n".join(sorted(c["id"] for c in kept))
                                                                       .encode())})
    data_dir = lay.data
    if (data_dir / "data.json").exists() and not force:
        old = json.loads((data_dir / "data.json").read_text())
        if old.get("plan_sha256") != plan.sha256:
            raise DataError(f"{data_dir} holds another plan ({old.get('plan_sha256')}); --force replaces it")
    clips_sha = write_table(data_dir / "clips.parquet", sorted(kept, key=lambda c: (c["shard"], c["row"])))
    save_plan(plan, data_dir)
    summary = {
        "plan": PLAN, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "smoke_shards": lay.smoke_shards,
        "c1": {"dir": str(c1), "shards_read": int(n_shards)}, "rules": RULES, "sources": list(SOURCES),
        "counts": counts, "uncovered": {"count": len(uncovered), "items": uncovered},
        "empty_text_kept": empty, "tokenizer_sha256_of_model": tokenizer_sha256,
        "denylist": {"path": str(deny_path), "sha256": lshards.sha256_file(deny_path),
                     "recordings_for_these_sources": {s: len(denied.get(s, ())) for s in SOURCES}},
        "heldout": {"fraction": HELDOUT_FRACTION, "salt": HELDOUT_SALT,
                    "recordings": {s: sorted(v) for s, v in held.items()}, "hours": summarize_hours(heldout)},
        "train": {"hours": summarize_hours(train)}, "all": {"hours": summarize_hours(kept)}, "dev_hygiene": hyg,
        "clips_sha256": clips_sha, "plan_sha256": plan.sha256, "plan_identity": plan.identity(),
        "select_s": time.time() - t0}
    write_json(data_dir / "data.json", summary)
    return summary


# ----------------------------------------------------------------------------- audio store

def audio_paths(lay: Layout, p: int) -> dict[str, Path]:
    base = lay.audio / "pools"
    return {"flac": base / f"{p:05d}.flac", "meta": base / f"{p:05d}.json"}


def audio_committed(lay: Layout, data: Data, p: int) -> bool:
    meta = audio_paths(lay, p)["meta"]
    if not meta.exists():
        return False
    m = json.loads(meta.read_text())
    return m.get("plan_sha256") == data.plan.sha256 and [e["id"] for e in m["entries"]] == data.pool_ids(p)


def write_pool_audio(lay: Layout, data: Data, p: int, c1: Path) -> dict:
    """Pool p's audio file from C1's shards (C1's FLAC bytes unchanged), then its marker."""
    import pyarrow.parquet as pq
    ids = data.pool_ids(p)
    want = {i: data.clips[i] for i in ids}
    by_shard: dict[int, dict[int, str]] = {}
    for i in ids:
        by_shard.setdefault(int(want[i]["shard"]), {})[int(want[i]["row"])] = i
    found: dict[str, bytes] = {}
    for k in sorted(by_shard):
        f = pq.ParquetFile(lshards.paths(c1, k)["audio"])
        starts = np.cumsum([0] + [f.metadata.row_group(g).num_rows for g in range(f.metadata.num_row_groups)])
        rows = by_shard[k]
        groups = sorted({int(np.searchsorted(starts, r, side="right") - 1) for r in rows})
        for g in groups:
            t = f.read_row_group(g, columns=["id", "flac"]).to_pydict()
            for j, (item, flac) in enumerate(zip(t["id"], t["flac"])):
                row = int(starts[g]) + j
                if row in rows:
                    if rows[row] != item:
                        raise DataError(f"C1 shard {k} row {row} is {item}, the table says {rows[row]}")
                    found[item] = flac
    missing = [i for i in ids if i not in found]
    if missing:
        raise DataError(f"pool {p}: {len(missing)} clips not found in C1 ({missing[:3]})")
    paths = audio_paths(lay, p)
    paths["flac"].parent.mkdir(parents=True, exist_ok=True)
    tmp = paths["flac"].with_name(paths["flac"].name + f".tmp{os.getpid()}")
    entries, offset = [], 0
    digest = hashlib.sha256()
    with open(tmp, "wb") as out:
        for i in ids:
            b = found[i]
            out.write(b)
            digest.update(b)
            c = want[i]
            entries.append({"id": i, "offset": offset, "length": len(b), "samples": int(c["samples"]),
                            "pcm_sha256": c["pcm_sha256"]})
            offset += len(b)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, paths["flac"])
    meta = {"pool": p, "plan_sha256": data.plan.sha256, "clips": len(entries), "bytes": offset,
            "flac_sha256": digest.hexdigest(), "entries": entries}
    write_json(paths["meta"], meta)
    return meta


class PoolAudio:
    """Decoded clips of one pool's audio file (each checked against C1's pcm_sha256)."""

    def __init__(self, lay: Layout, data: Data, p: int):
        paths = audio_paths(lay, p)
        if not audio_committed(lay, data, p):
            raise DataError(f"pool {p}: audio not committed for this plan")
        self.meta = json.loads(paths["meta"].read_text())
        self.path = paths["flac"]
        self.entries = {e["id"]: e for e in self.meta["entries"]}

    def raw(self, item_id: str) -> bytes:
        e = self.entries[item_id]
        with open(self.path, "rb") as f:
            return os.pread(f.fileno(), e["length"], e["offset"])

    def clip(self, item_id: str) -> np.ndarray:
        return decode_checked(item_id, self.raw(item_id), self.entries[item_id])


def decode_checked(item_id: str, data: bytes, entry: dict) -> np.ndarray:
    pcm = lshards.decode_flac(data)
    if pcm.size != entry["samples"] or lshards.pcm_sha256(pcm) != entry["pcm_sha256"]:
        raise DataError(f"{item_id}: decoded samples differ from C1's stored SHA-256")
    return pcm.astype(np.float32) / 32768.0


def build_audio(lay: Layout, budget_minutes: float | None = None) -> int:
    relabel_paths.require_mount()
    data = Data.load(lay)
    c1 = Path(data.summary["c1"]["dir"])
    t0 = time.time()
    for p in range(data.plan.n_pools):
        if audio_committed(lay, data, p):
            continue
        if budget_minutes is not None and time.time() - t0 > budget_minutes * 60:
            print(f"audio: budget reached before pool {p}; the next run resumes", flush=True)
            return 0
        m = write_pool_audio(lay, data, p, c1)
        print(f"audio: pool {p}/{data.plan.n_pools} committed, {m['clips']} clips, {m['bytes'] / 2**30:.2f} GiB "
              f"({time.time() - t0:.0f} s)", flush=True)
    print("audio: every pool committed", flush=True)
    return 0


# ----------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("select", "audio", "stats"))
    parser.add_argument("--smoke-shards", type=int, help="smoke: the first N C1 shards (writes under SMOKE only)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--budget-minutes", type=float)
    args = parser.parse_args(argv)
    lay = layout(args.smoke_shards)
    if args.cmd == "select":
        s = select(lay, force=args.force)
        print(json.dumps({k: s[k] for k in ("counts", "train", "heldout", "plan_identity", "empty_text_kept")},
                         indent=1, default=str))
        print(f"uncovered: {s['uncovered']['count']}")
        return 0
    if args.cmd == "audio":
        return build_audio(lay, args.budget_minutes)
    data = Data.load(lay)
    print(json.dumps({k: data.summary[k] for k in ("counts", "train", "heldout", "plan_identity")}, indent=1,
                     default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
