"""The labeling pool's training stream (labeling plan Part C "Pool and stream").

The existing stream (finetune/parakeet-ternary/stream.py, imported, never modified) with three additions:

- **Include filter by recording id, reading only the listed row groups** (the domain subsets): a subset
  source's read units are (parquet file, row group) pairs from Part B's locators; a slot reads exactly one
  such row group, and only rows whose recording is in the subset's include list are kept (other rows of a
  listed row group are dropped as `not_in_subset`).
- **Exclusions by recording on every source:** the global denylist (`denylisted`), and for the general
  `yodas` / `peoples_speech` sources the recordings of their subsets (`in_subset`), so a recording is
  drawn through exactly one route. Recording keys: yodas `original_audio_id`; peoples_speech the row id
  without its `_<segment>.flac` suffix (Part B's pool.common.ps_recording); local corpora the manifest's
  `recording`.
- **Local sources** (Part B's segmented corpora): manifest rows whose FLAC files are read from
  $RELABEL_ROOT/corpora/<alias>/, decoded by the same path as the Hub audio (data.convert).

Everything else is stream.py's: files (here: units) split across W workers (unit i to worker i mod W);
per worker, seeded source draws with hour-weighted probabilities, seeded unit order per pass, `slots` units
open at once, the 1-30 s filter; the parent yields from workers in strict round robin (stream.TrainingStream,
subclassed only to start these workers). State, resume and the configuration hash follow stream.py; the hash
covers every field of PoolStreamConfig (units, include and exclude lists, local rows, shares, mean seconds,
seed, filters) and the worker count.

Items: stream.py's {"audio", "duration", "text", "id", "source"} ("text" is the human transcript or None,
exactly as in stream.py, so training's filters behave as today) plus "reference" (the source's own text:
human transcript, or YODAS-Granary's pseudo-label, or None) and "reference_kind" ("human", "pseudo" or None).
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import multiprocessing as mp
import os
import queue as queue_mod
import time
import traceback
from pathlib import Path

import numpy as np

import data       # finetune/parakeet-ternary
import paths      # finetune/parakeet-ternary
import stream     # finetune/parakeet-ternary

STATE_VERSION = 1
KINDS = ("hf", "subset", "local")
RECORDING_KEYS = (None, "group", "ps_id", "local")


@dataclasses.dataclass(frozen=True)
class PoolSourceSpec:
    name: str                              # source alias (an item's "source", the id prefix of Hub sources)
    kind: str                              # hf | subset | local
    units: tuple                           # hf: file URLs; subset: (file URL, row group); local: row indices
    share: float
    id_col: str = "id"
    text_col: str | None = None            # human transcript column (the stream's "text")
    pseudo_col: str | None = None          # pseudo-label column (reference only, kind "pseudo")
    group_col: str | None = None
    audio_col: str = "audio"
    slots: int = 1
    recording_key: str | None = None       # how a row maps to its recording (RECORDING_KEYS)
    include: tuple[str, ...] = ()          # subset: the recordings kept
    exclude: tuple[tuple[str, str], ...] = ()   # (recording, reason): `denylisted` or `in_subset`, sorted
    local_dir: str | None = None           # local: corpus directory relative to $RELABEL_ROOT
    local_rows: tuple = ()                 # local: (id, relative audio path, text, recording) per unit

    def columns(self) -> list[str]:
        return list(dict.fromkeys(c for c in (self.id_col, self.text_col, self.pseudo_col, self.audio_col,
                                              self.group_col) if c))


@dataclasses.dataclass(frozen=True)
class PoolStreamConfig:
    sources: tuple[PoolSourceSpec, ...]
    seed: int
    weighting: str = "hours"
    mean_seconds: tuple[tuple[str, float], ...] = ()
    min_seconds: float = data.MIN_SECONDS
    max_seconds: float = data.MAX_SECONDS
    batch_rows: int = 16
    block_size: int = 8 * 2**20
    retry_attempts: int = 12
    retry_base_seconds: float = 5.0
    retry_max_seconds: float = 300.0

    def probabilities(self) -> np.ndarray:
        shares = np.array([s.share for s in self.sources], dtype=np.float64)
        if self.weighting == "hours":
            means = dict(self.mean_seconds)
            shares = shares / np.array([means[s.name] for s in self.sources])
        elif self.weighting != "utterances":
            raise ValueError(f"unknown weighting {self.weighting!r}")
        return shares / shares.sum()

    def digest(self, partitions: int) -> str:
        blob = json.dumps({"config": dataclasses.asdict(self), "partitions": partitions,
                           "version": STATE_VERSION, "kind": "pool-stream"}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    # stream.TrainingStream reads cfg.seed and cfg.sources only.


def config_from_dict(obj: dict) -> PoolStreamConfig:
    """Rebuild a PoolStreamConfig from its JSON form (dataclasses.asdict output after a JSON round trip)."""
    def tup(x):
        return tuple(tup(v) for v in x) if isinstance(x, list) else x

    sources = tuple(PoolSourceSpec(**{k: tup(v) for k, v in s.items()}) for s in obj["sources"])
    rest = {k: tup(v) for k, v in obj.items() if k != "sources"}
    cfg = PoolStreamConfig(sources=sources, **rest)
    for s in cfg.sources:
        if s.kind not in KINDS or s.recording_key not in RECORDING_KEYS:
            raise ValueError(f"source {s.name}: bad kind {s.kind!r} or recording key {s.recording_key!r}")
    return cfg


def recording_of(spec: PoolSourceSpec, row: dict) -> str | None:
    if spec.recording_key is None:
        return None
    if spec.recording_key == "group":
        value = row.get(spec.group_col)
        return None if value is None else str(value)
    if spec.recording_key == "ps_id":
        from pool import common
        return common.ps_recording(row[spec.id_col])[0]
    return row.get("recording")


# ---------------------------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------------------------

class _UnitSlot:
    """One read unit, read sequentially in record batches from an absolute row position: a whole parquet
    file (hf), a single row group of one (subset), or one local manifest row (local)."""

    def __init__(self, cfg: PoolStreamConfig, spec: PoolSourceSpec, unit, epoch: int, pos: int, row: int = 0):
        self.cfg, self.spec, self.unit = cfg, spec, unit
        self.epoch, self.pos, self.row = epoch, pos, row
        self._handle = self._batches = None
        self._buffer: list[dict] = []
        self.retries = 0

    def state(self) -> dict:
        return {"epoch": self.epoch, "pos": self.pos, "row": self.row}

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception:
                pass
        self._handle = self._batches = None
        self._buffer = []

    def _local_rows(self) -> list[dict]:
        ident, rel, text, recording = self.spec.local_rows[self.unit]
        base = Path(os.environ["RELABEL_ROOT"]) / self.spec.local_dir
        return [{"id": ident, "text": text, "recording": recording,
                 "audio": {"bytes": (base / rel).read_bytes()}}]

    def _open_at_row(self) -> None:
        import pyarrow.parquet as pq
        self.close()
        if self.spec.kind == "local":
            rows = self._local_rows()[self.row:]
            self._batches = iter([rows] if rows else [])
            self._buffer = []
            return
        url, group = (self.unit, None) if self.spec.kind == "hf" else (self.unit[0], int(self.unit[1]))
        self._handle = stream._open_file(url, self.cfg.block_size)
        pf = pq.ParquetFile(self._handle, pre_buffer=False, buffer_size=self.cfg.block_size)
        meta = pf.metadata
        if group is None:
            start, g = 0, 0
            while g < meta.num_row_groups and start + meta.row_group(g).num_rows <= self.row:
                start += meta.row_group(g).num_rows
                g += 1
            groups = range(g, meta.num_row_groups)
        else:
            if not 0 <= group < meta.num_row_groups:
                raise stream.StreamError(self.spec.name, f"{url} has no row group {group}")
            start, groups = 0, [group] if self.row < meta.row_group(group).num_rows else []
        if not len(groups):
            self._batches = iter(())
            return
        batches = pf.iter_batches(batch_size=self.cfg.batch_rows, row_groups=groups,
                                  columns=self.spec.columns(), use_threads=False)
        self._batches = (b.to_pylist() for b in batches)
        skip = self.row - start
        while skip > 0:
            rows = next(self._batches)
            if len(rows) > skip:
                self._buffer = rows[skip:]
                break
            skip -= len(rows)

    def next_row(self) -> dict | None:
        attempt = 0
        while not self._buffer:
            try:
                if self._batches is None:
                    self._open_at_row()
                    continue
                rows = next(self._batches, None)
                if rows is None:
                    self.close()
                    return None
                self._buffer = list(rows)
            except stream.StreamError:
                raise
            except Exception as exc:
                attempt += 1
                self.retries += 1
                self.close()
                if attempt >= self.cfg.retry_attempts:
                    raise stream.StreamError(self.spec.name, f"{self.unit} row {self.row}: giving up after "
                                                             f"{attempt} attempts: {exc!r}") from exc
                wait = min(self.cfg.retry_max_seconds, self.cfg.retry_base_seconds * 2 ** (attempt - 1))
                stream._log(f"{self.spec.name} {self.unit} row {self.row}: {exc!r}; retry {attempt} in {wait:.0f}s")
                time.sleep(wait)
        self.row += 1
        return self._buffer.pop(0)


class _UnitReader:
    """stream._SourceReader over read units: seeded unit order per pass, `slots` units open at once."""

    def __init__(self, cfg: PoolStreamConfig, index: int, worker: int, partitions: int, state: dict | None):
        self.cfg, self.spec, self.index, self.worker = cfg, cfg.sources[index], index, worker
        self.units = sorted(self.spec.units)[worker::partitions]
        if not self.units:
            raise ValueError(f"source {self.spec.name}: no read units for worker {worker} of {partitions}")
        self.rng = np.random.default_rng([cfg.seed, worker, index, 1])
        self.cursor = {"epoch": 0, "pos": 0}
        self.slots: list[_UnitSlot | None] = [None] * self.spec.slots
        self.counters = stream._empty_counters()
        if state:
            self.rng.bit_generator.state = state["rng"]
            self.cursor = dict(state["cursor"])
            self.counters = copy.deepcopy(state["counters"])
            self.slots = [None if s is None else self._slot(s["epoch"], s["pos"], s["row"]) for s in state["slots"]]

    def _order(self, epoch: int) -> np.ndarray:
        return np.random.default_rng([self.cfg.seed, self.worker, self.index, 2, epoch]).permutation(len(self.units))

    def _slot(self, epoch: int, pos: int, row: int = 0) -> _UnitSlot:
        return _UnitSlot(self.cfg, self.spec, self.units[self._order(epoch)[pos]], epoch, pos, row)

    def _next_unit(self) -> _UnitSlot:
        if self.cursor["pos"] >= len(self.units):
            self.cursor = {"epoch": self.cursor["epoch"] + 1, "pos": 0}
            self.counters["passes_started"] += 1
        slot = self._slot(self.cursor["epoch"], self.cursor["pos"])
        self.cursor["pos"] += 1
        self.counters["files_opened"] += 1
        return slot

    def next_row(self) -> dict:
        k = int(self.rng.integers(len(self.slots)))
        for _ in range(len(self.units) + 1):
            if self.slots[k] is None:
                self.slots[k] = self._next_unit()
            slot = self.slots[k]
            before = slot.retries
            row = slot.next_row()
            self.counters["retries"] += slot.retries - before
            if row is not None:
                return row
            self.slots[k] = None
        raise stream.StreamError(self.spec.name, "every read unit is empty")

    def state(self) -> dict:
        return {"rng": self.rng.bit_generator.state, "cursor": dict(self.cursor),
                "slots": [None if s is None else s.state() for s in self.slots],
                "counters": copy.deepcopy(self.counters)}

    def close(self) -> None:
        for slot in self.slots:
            if slot is not None:
                slot.close()


class _PoolWorker:
    """The item stream of one partition (stream._WorkerStream with include/exclude filters and references)."""

    def __init__(self, cfg: PoolStreamConfig, worker: int, partitions: int, state: dict | None):
        self.cfg, self.worker = cfg, worker
        self.probs = cfg.probabilities()
        self.mix = np.random.default_rng([cfg.seed, worker, 0])
        self.include = [frozenset(s.include) for s in cfg.sources]
        self.exclude = [dict(s.exclude) for s in cfg.sources]
        sstates = (state or {}).get("sources", {})
        self.readers = [_UnitReader(cfg, i, worker, partitions, sstates.get(s.name)) for i, s in enumerate(cfg.sources)]
        if state:
            self.mix.bit_generator.state = state["mix"]

    def state(self) -> dict:
        return {"mix": self.mix.bit_generator.state, "sources": {r.spec.name: r.state() for r in self.readers}}

    def _route_reason(self, i: int, row: dict) -> str | None:
        spec = self.cfg.sources[i]
        if spec.recording_key is None:
            return None
        rec = recording_of(spec, row)
        if spec.kind == "subset" and rec not in self.include[i]:
            return "not_in_subset"
        return self.exclude[i].get(rec) if rec is not None else None

    def next_item(self) -> dict:
        i = int(self.mix.choice(len(self.readers), p=self.probs))
        reader = self.readers[i]
        spec, c = reader.spec, reader.counters
        while True:
            row = reader.next_row()
            c["rows"] += 1
            reason = self._route_reason(i, row)
            pcm = None
            if reason is None:
                pcm, reason = stream._decode((row.get(spec.audio_col) or {}).get("bytes"))
            if reason is None:
                reason = data.duration_drop_reason(pcm.size / paths.SAMPLE_RATE, self.cfg.min_seconds,
                                                   self.cfg.max_seconds)
            if reason:
                c["dropped"][reason] = c["dropped"].get(reason, 0) + 1
                continue
            duration = pcm.size / paths.SAMPLE_RATE
            c["kept"] += 1
            c["kept_seconds"] += duration
            text = row.get(spec.text_col) if spec.text_col else None
            text = text.strip() if isinstance(text, str) else ""
            pseudo = row.get(spec.pseudo_col) if spec.pseudo_col else None
            pseudo = pseudo.strip() if isinstance(pseudo, str) else ""
            if text:
                reference, kind = text, "human"
            elif pseudo:
                reference, kind = pseudo, "pseudo"
            else:
                reference, kind = None, None
            ident = row[spec.id_col] if spec.kind == "local" else f"{spec.name}:{row[spec.id_col]}"
            return {"audio": pcm.astype(np.float32) / 32768.0, "duration": duration, "text": text or None,
                    "id": ident, "source": spec.name, "reference": reference, "reference_kind": kind}

    def close(self) -> None:
        for r in self.readers:
            r.close()


def _pool_worker_main(cfg: PoolStreamConfig, worker: int, partitions: int, state: dict | None,
                      out: mp.Queue, stop: mp.Event) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        import pyarrow as pa
        pa.set_cpu_count(1)
        pa.set_io_thread_count(2)
        w = _PoolWorker(cfg, worker, partitions, state)
        while not stop.is_set():
            msg = ("item", w.next_item(), w.state())
            while not stop.is_set():
                try:
                    out.put(msg, timeout=1.0)
                    break
                except queue_mod.Full:
                    continue
    except stream.StreamError as exc:
        out.put(("error", exc.source, str(exc)))
    except BaseException:
        out.put(("error", None, traceback.format_exc()))


class PoolTrainingStream(stream.TrainingStream):
    """stream.TrainingStream (state, resume, round robin, worker restarts) running _PoolWorker partitions."""

    def __init__(self, cfg: PoolStreamConfig, num_workers: int, phase: str = "main", prefetch: int = 64,
                 max_restarts: int = 5):
        super().__init__(cfg, num_workers=num_workers, phase=phase, prefetch=prefetch, max_restarts=max_restarts)

    def _spawn(self, w: int) -> None:
        ctx = mp.get_context("spawn")
        q = ctx.Queue(maxsize=self.prefetch)
        p = ctx.Process(target=_pool_worker_main, args=(self.cfg, w, self.partitions, self._wstates[w], q, self._stop),
                        daemon=True, name=f"pool-stream-worker-{w}")
        p.start()
        if w < len(self._procs):
            self._procs[w], self._queues[w] = p, q
        else:
            self._procs.append(p)
            self._queues.append(q)

    def _start(self) -> None:
        self._started = True
        if self.num_workers == 0:
            self._inproc = _PoolWorker(self.cfg, 0, 1, self._wstates[0])
            return
        self._stop = mp.get_context("spawn").Event()
        for w in range(self.partitions):
            self._spawn(w)

    def pids(self) -> list[int]:
        return [p.pid for p in self._procs if p.pid is not None]
