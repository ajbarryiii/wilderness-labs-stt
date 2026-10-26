"""Materialized pool audio: the batch plan's clips stored pool by pool, so training reads each pool sequentially
(b0prime plan "Reading the plan", 2026-10-24).

A pool's content depends only on C1's journal and the pool size P (not on the cut, the budget or the seed): the
FLAC of every distinct id that occurs in the pool's consumed batches, a repeated id once per pool that uses it.
Every plan over the same journal and P shares one store:

    <store>/store.json                  kind, version, pinned_hash, pool_batches, pool_dir
    <store>/pools/<ppppp>.flac          the pool's clips' FLAC bytes (C1's, unchanged), concatenated
    <store>/pools/<ppppp>.index.parquet id, offset, length, samples, pcm_sha256 (C1's), in file order
    <store>/pools/<ppppp>.json          commit marker, written last: pool, consumed_batches [lo, hi), clips,
                                        flac_bytes, index_sha256, ids_sha256 (of the sorted ids, one per line)
    <store>/work/                       an unfinished pass: <ppppp>.flac.part, <ppppp>.index.part, state.json

The pass (materialize) reads C1's audio shards once, in order, each file whole (its SHA-256 checked against its
meta), and pushes every stored clip to every target pool that uses it; a clip's audio is stored in the shard of its
id's first occurrence, so a pool is complete once the pass has read the shard holding the pool's last consumed
batch, and is then committed. It is resumable: every `checkpoint_shards` shards (and at a budget or requested
stop) every part file is flushed and fsynced and work/state.json records their lengths; a restart truncates the
parts to those lengths and continues with the next shard, so a resumed pass writes the same bytes as an
uninterrupted one. A state written for other target pools is discarded.

The reader (PoolAudio) checks store.json against the plan and every plan pool's marker (present, the plan's
consumed range); a loaded pool's file size and index hash against its marker, and its ids against the plan's
(PlannedReplay); every decoded clip's samples are checked against C1's stored SHA-256 (replay.AudioStore.decode).
It caches the current pool and one prefetched pool.

    ./python -m training.batchplan materialize --pools N [--budget-minutes 110]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

from labeling import layout, shards

KIND = "relabel-pool-audio"
VERSION = 1
CHECKPOINT_SHARDS = 20
FLUSH_BYTES = 4 << 20            # a part file is appended in pieces of at least this size
BUFFER_LIMIT = 1 << 30           # total buffered bytes before the largest buffers are flushed
CACHE_POOLS = 2
COLUMNS = ("id", "samples", "pcm_sha256", "flac")


def pool_name(p: int) -> str:
    return f"{int(p):05d}"


def store_dir(config_id: str, pool_batches: int) -> Path:
    """$RELABEL_ROOT/training/pool-audio/<config id>-p<P>."""
    return layout.relabel_root() / "training" / "pool-audio" / f"{config_id}-p{int(pool_batches)}"


def _log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [poolaudio] {msg}", flush=True)


def _atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _json(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=1) + "\n").encode()


def _store_meta(plan) -> dict:
    return {"kind": KIND, "version": VERSION, "pinned_hash": plan.meta["pinned_hash"],
            "pool_batches": plan.pool_batches, "pool_dir": plan.meta["pool_dir"]}


def _check_store(store: Path, plan, create: bool) -> None:
    path = Path(store) / "store.json"
    want = _store_meta(plan)
    if not path.exists():
        if not create:
            raise RuntimeError(f"{store} is not a pool-audio store")
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic(path, _json(want))
        return
    got = json.loads(path.read_text())
    if {k: got.get(k) for k in want} != want:
        raise RuntimeError(f"{store} holds pools of another journal or pool size ({got})")


def pool_range(plan, p: int) -> list[int]:
    lo = p * plan.pool_batches
    return [lo, min(lo + plan.pool_batches, plan.consumed_batches)]


def _ids_sha(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def committed(store: Path, plan, p: int) -> dict | None:
    """The marker of pool p if it is committed for the plan's consumed range, else None."""
    path = Path(store) / "pools" / f"{pool_name(p)}.json"
    if not path.exists():
        return None
    meta = json.loads(path.read_text())
    if meta.get("pool") != p or meta.get("consumed_batches") != pool_range(plan, p):
        return None
    return meta


# ----------------------------------------------------------------------------- the pass

class _Parts:
    """Part files of the target pools: buffered appends, lengths, flush and fsync."""

    def __init__(self, work: Path, sizes: dict[int, dict]):
        self.work = work
        self.size = {p: dict(v) for p, v in sizes.items()}     # p -> {"flac", "index", "clips"} (logical)
        self.buf: dict[int, list[bytes]] = collections.defaultdict(list)
        self.ibuf: dict[int, list[bytes]] = collections.defaultdict(list)
        self.pending: dict[int, int] = collections.defaultdict(int)
        self.total = 0
        self.dirty: set[int] = set()

    def path(self, p: int, kind: str) -> Path:
        return self.work / f"{pool_name(p)}.{kind}.part"

    def add(self, p: int, item_id: str, samples: int, digest: str, flac: bytes) -> None:
        s = self.size.setdefault(p, {"flac": 0, "index": 0, "clips": 0})
        line = (json.dumps([item_id, s["flac"], len(flac), int(samples), digest]) + "\n").encode()
        self.buf[p].append(flac)
        self.ibuf[p].append(line)
        s["flac"] += len(flac)
        s["index"] += len(line)
        s["clips"] += 1
        self.pending[p] += len(flac) + len(line)
        self.total += len(flac) + len(line)
        if self.pending[p] >= FLUSH_BYTES:
            self.flush(p)
        if self.total > BUFFER_LIMIT:
            for q in sorted(self.pending, key=self.pending.get, reverse=True)[:max(1, len(self.pending) // 4)]:
                self.flush(q)

    def flush(self, p: int) -> None:
        if not self.buf.get(p):
            return
        with open(self.path(p, "flac"), "ab") as handle:
            handle.write(b"".join(self.buf.pop(p)))
        with open(self.path(p, "index"), "ab") as handle:
            handle.write(b"".join(self.ibuf.pop(p)))
        self.total -= self.pending.pop(p)
        self.dirty.add(p)

    def sync(self) -> None:
        for p in list(self.buf):
            self.flush(p)
        for p in self.dirty:
            for kind in ("flac", "index"):
                with open(self.path(p, kind), "rb+") as handle:
                    os.fsync(handle.fileno())
        self.dirty.clear()


def _restore(store: Path, work: Path, state: dict, active: set[int]) -> dict[int, dict]:
    """Truncate the active pools' part files to the checkpoint's lengths; remove every other part. A pool whose
    commit was interrupted after its FLAC file moved (no marker yet) gets the file back as its part."""
    sizes = {int(p): v for p, v in state["parts"].items() if int(p) in active}
    for p in sizes:
        part, moved = work / f"{pool_name(p)}.flac.part", Path(store) / "pools" / f"{pool_name(p)}.flac"
        if not part.exists() and moved.exists():
            os.replace(moved, part)
    for f in work.glob("*.part"):
        p, kind = int(f.name.split(".")[0]), f.name.split(".")[1]
        if p not in sizes:
            f.unlink()
            continue
        length = sizes[p][kind]
        if f.stat().st_size < length:
            raise RuntimeError(f"{f} is shorter than its checkpoint ({f.stat().st_size} < {length})")
        with open(f, "rb+") as handle:
            handle.truncate(length)
    for p, v in sizes.items():
        for kind in ("flac", "index"):
            f = work / f"{pool_name(p)}.{kind}.part"
            if not f.exists() or f.stat().st_size != v[kind]:
                raise RuntimeError(f"{f} is missing or differs from its checkpoint")
    return sizes


def _finalize(store: Path, parts: _Parts, plan, p: int, expected: set[str]) -> None:
    """Commit pool p: index parquet, then the FLAC file, then the marker (written last)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    parts.flush(p)
    fpath, ipath = parts.path(p, "flac"), parts.path(p, "index")
    if not fpath.exists() or not ipath.exists():
        raise RuntimeError(f"pool {p}: no materialized clips")
    rows = [json.loads(line) for line in ipath.read_text().splitlines()]
    ids = [r[0] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise RuntimeError(f"pool {p}: materialized ids differ from the plan's ({len(ids)} vs {len(expected)})")
    flac_bytes = sum(r[2] for r in rows)
    if fpath.stat().st_size != flac_bytes:
        raise RuntimeError(f"pool {p}: FLAC bytes differ from the index")
    table = pa.table({"id": pa.array(ids, pa.string()), "offset": pa.array([r[1] for r in rows], pa.int64()),
                      "length": pa.array([r[2] for r in rows], pa.int64()),
                      "samples": pa.array([r[3] for r in rows], pa.int64()),
                      "pcm_sha256": pa.array([r[4] for r in rows], pa.string())})
    sink = io.BytesIO()
    pq.write_table(table, sink, compression="zstd")
    index = sink.getvalue()
    pools = Path(store) / "pools"
    pools.mkdir(parents=True, exist_ok=True)
    _atomic(pools / f"{pool_name(p)}.index.parquet", index)
    with open(fpath, "rb+") as handle:
        os.fsync(handle.fileno())
    os.replace(fpath, pools / f"{pool_name(p)}.flac")
    marker = {"pool": p, "consumed_batches": pool_range(plan, p), "clips": len(ids), "flac_bytes": int(flac_bytes),
              "index_sha256": hashlib.sha256(index).hexdigest(), "ids_sha256": _ids_sha(ids)}
    _atomic(pools / f"{pool_name(p)}.json", _json(marker))
    ipath.unlink()
    parts.size.pop(p, None)
    parts.dirty.discard(p)


def materialize(plan, pool_dir: Path, store: Path, *, budget_s: float | None = None,
                checkpoint_shards: int = CHECKPOINT_SHARDS, stop_after_shards: int | None = None) -> dict:
    """Run (or continue) the pass for every plan pool not yet committed in the store (module docstring)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    t0 = time.monotonic()
    store, pool_dir = Path(store), Path(pool_dir)
    j = plan.verify_journal(pool_dir)
    _check_store(store, plan, create=True)
    pending = [p for p in range(plan.n_pools) if committed(store, plan, p) is None]
    status = {"pools": plan.n_pools, "pending": len(pending)}
    work = store / "work"
    state_path = work / "state.json"
    if not pending:
        return {**status, "complete": True}
    work.mkdir(parents=True, exist_ok=True)
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    base = {"consumed_batches": plan.consumed_batches, "pool_batches": plan.pool_batches,
            "pinned_hash": plan.meta["pinned_hash"]}
    if state is not None and ({k: state["key"].get(k) for k in base} != base
                              or not set(pending) <= set(state["key"]["targets"])):
        _log("discarding an unfinished pass for other target pools")
        state = None
    if state is None:
        for f in work.iterdir():
            f.unlink()
    key = state["key"] if state is not None else {**base, "targets": pending}
    active = set(pending)
    sizes = _restore(store, work, state, active) if state is not None else {}
    pool_of = np.searchsorted(plan.pools, np.arange(len(j["ids"])), side="right") - 1
    dest: dict[str, list[int]] = {}
    expected: dict[int, set[str]] = {p: set() for p in pending}
    for k in range(int(plan.pools[pending[0]]), len(j["ids"])):
        p = int(pool_of[k])
        if p in active and j["ids"][k] not in expected[p]:
            expected[p].add(j["ids"][k])
            dest.setdefault(j["ids"][k], []).append(p)
    ranges = {s["shard"]: s["batches"] for s in plan.meta["journal_shards"]}
    ks = sorted(ranges)
    done_after = collections.defaultdict(list)          # shard -> pools complete once it is read
    for p in pending:
        last = pool_range(plan, p)[1] - 1
        done_after[next(k for k in ks if ranges[k][0] <= last < ranges[k][1])].append(p)
    parts = _Parts(work, sizes)
    start = ks.index(state["through_shard"]) + 1 if state is not None else 0
    read = 0
    for k in ks[start:]:
        meta = shards.load_meta(pool_dir, k)
        if meta.get("audio_deleted"):
            raise RuntimeError(f"C1 shard {k} has no audio")
        raw = shards.paths(pool_dir, k)["audio"].read_bytes()
        if len(raw) != meta["files"]["audio"]["bytes"] or hashlib.sha256(raw).hexdigest() != meta["files"]["audio"]["sha256"]:
            raise RuntimeError(f"C1 shard {k}: audio differs from its meta's SHA-256")
        pf = pq.ParquetFile(pa.BufferReader(raw))
        for g in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(g, columns=list(COLUMNS)).to_pydict()
            for i, n, digest, flac in zip(t["id"], t["samples"], t["pcm_sha256"], t["flac"]):
                for p in dest.get(i, ()):
                    parts.add(p, i, n, digest, flac)
        del raw, pf
        for p in done_after.get(k, ()):
            _finalize(store, parts, plan, p, expected.pop(p))
            active.discard(p)
            _log(f"pool {p} committed (after C1 shard {k})")
        read += 1
        if not active:
            break
        over = budget_s is not None and time.monotonic() - t0 > budget_s
        stop = stop_after_shards is not None and read >= stop_after_shards
        if read % checkpoint_shards == 0 or over or stop:
            parts.sync()
            _atomic(state_path, _json({"key": key, "through_shard": k,
                                       "parts": {str(p): v for p, v in sorted(parts.size.items())}}))
            if over or stop:
                _log(f"stopped after C1 shard {k} ({'budget' if over else 'requested'}); rerun to continue")
                return {**status, "complete": False, "through_shard": k}
    if active:
        raise RuntimeError(f"pools {sorted(active)[:5]} not committed at the end of the pass")
    for f in work.iterdir():
        f.unlink()
    work.rmdir()
    return {**status, "complete": True, "seconds": time.monotonic() - t0}


# ----------------------------------------------------------------------------- the reader

class PoolAudio:
    """Clips of a plan's pools from a materialized store: load(p) -> {id: (flac, samples, pcm_sha256)}."""

    def __init__(self, store: Path, plan, cache_pools: int = CACHE_POOLS):
        self.store, self.plan, self.cache_pools = Path(store), plan, cache_pools
        _check_store(self.store, plan, create=False)
        self.markers = {}
        for p in range(plan.n_pools):
            m = committed(self.store, plan, p)
            if m is None:
                raise RuntimeError(f"pool {p} is not materialized in {self.store} for this plan")
            self.markers[p] = m
        self._cache: collections.OrderedDict = collections.OrderedDict()
        self._loading: dict[int, threading.Event] = {}
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    def index(self, p: int) -> dict:
        import pyarrow.parquet as pq
        raw = (self.store / "pools" / f"{pool_name(p)}.index.parquet").read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.markers[p]["index_sha256"]:
            raise RuntimeError(f"pool {p}: index differs from its marker")
        import pyarrow as pa
        return pq.read_table(pa.BufferReader(raw)).to_pydict()

    def _read(self, p: int) -> dict:
        index = self.index(p)
        data = (self.store / "pools" / f"{pool_name(p)}.flac").read_bytes()
        if len(data) != self.markers[p]["flac_bytes"]:
            raise RuntimeError(f"pool {p}: FLAC file size differs from its marker")
        view = memoryview(data)
        out = {i: (view[o:o + n], s, d) for i, o, n, s, d in
               zip(index["id"], index["offset"], index["length"], index["samples"], index["pcm_sha256"])}
        if len(out) != self.markers[p]["clips"] or _ids_sha(out) != self.markers[p]["ids_sha256"]:
            raise RuntimeError(f"pool {p}: index ids differ from its marker")
        return out

    def load(self, p: int) -> dict:
        while True:
            with self._lock:
                hit = self._cache.get(p)
                if hit is not None:
                    self._cache.move_to_end(p)
                    return hit
                event = self._loading.get(p)
                if event is None:
                    event = self._loading[p] = threading.Event()
                    mine = True
                else:
                    mine = False
            if not mine:
                event.wait()
                continue
            try:
                with self._lock:       # make room before the new pool is allocated: at most cache_pools resident
                    while len(self._cache) + len(self._loading) > self.cache_pools and self._cache:
                        self._cache.popitem(last=False)
                clips = self._read(p)
                with self._lock:
                    self._cache[p] = clips
                    while len(self._cache) > self.cache_pools:
                        self._cache.popitem(last=False)
                return clips
            finally:
                with self._lock:
                    self._loading.pop(p).set()

    def prefetch(self, p: int) -> None:
        if 0 <= p < self.plan.n_pools:
            with self._lock:
                if p in self._cache or p in self._loading:
                    return
            thread = threading.Thread(target=self._prefetch, args=(p,), name=f"pool-prefetch-{p}", daemon=True)
            self._threads = [x for x in self._threads if x.is_alive()] + [thread]
            thread.start()

    def drain(self) -> None:
        """Wait for every prefetch, then drop the cache (the probe starts each region cold)."""
        for thread in list(self._threads):
            thread.join()
        with self._lock:
            self._cache.clear()

    def _prefetch(self, p: int) -> None:
        try:
            self.load(p)
        except Exception:      # noqa: BLE001 - the foreground load raises it again
            pass

    def identity(self) -> dict:
        return {"store": self.store.name, "pool_batches": self.plan.pool_batches,
                "markers_sha256": hashlib.sha256(_json([self.markers[p] for p in range(self.plan.n_pools)])).hexdigest()}


# ----------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    from training import batchplan
    ap = argparse.ArgumentParser(prog="python -m training.batchplan materialize")
    ap.add_argument("--pools", type=int, required=True, help="the production plan over pools 0 .. N-1")
    ap.add_argument("--budget-minutes", type=float, default=110.0)
    ap.add_argument("--plan-dir", type=Path, help="a smoke plan (with --store under $RELABEL_ROOT/smoke/)")
    ap.add_argument("--store", type=Path)
    a = ap.parse_args(argv)
    layout.require_mount()
    pool, cid, _ = batchplan._c1_and_config()
    plan = batchplan.Plan.load(a.plan_dir or batchplan.production_dir(cid, a.pools))
    store = a.store or store_dir(cid, plan.pool_batches)
    if a.plan_dir is not None and (a.store is None or layout.is_production(store)):
        raise SystemExit("a smoke plan materializes into a store under $RELABEL_ROOT/smoke/")
    _log(f"plan {plan.meta['plan_id']} ({plan.sha256[:12]}), store {store}")
    status = materialize(plan, pool, store, budget_s=a.budget_minutes * 60.0)
    import resource
    status["host_peak_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    _log(json.dumps(status))
    return 0
