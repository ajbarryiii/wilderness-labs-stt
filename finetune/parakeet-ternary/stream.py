"""Streamed training data for the ternary Parakeet experiment (DESIGN.md "Training data").

Training audio is read from the Hugging Face Hub while training runs and is not
stored on this machine. Development sets are local (data.py).

    stream = make_training_stream("pilot", seed=paths.SEED, num_workers=6)
    for item in stream:            # {"audio", "duration", "text", "id", "source"}
        ...
        ckpt["stream"] = stream.state_dict()
    # after a restart:
    stream = make_training_stream("pilot", seed=paths.SEED, num_workers=6)
    stream.load_state_dict(ckpt["stream"])   # continues with the next unconsumed item

Items: "audio" float32 numpy, 16 kHz mono, in [-1, 1); "duration" seconds;
"text" the corpus' human transcript or None (YODAS-Granary labels are model
pseudo-labels, so YODAS gives None); "id" "<source>:<corpus id>"; "source".

How it works. Every source is a list of parquet files at a pinned revision,
read through huggingface_hub's HfFileSystem with pyarrow (the layer that
`datasets` streaming itself uses), selecting only the needed columns. Files
are split across worker processes (file i goes to worker i mod N). Inside a
worker, each source keeps a few files open at once ("slots", a seeded buffer
over files) and reads each file sequentially in small record batches; the next
item comes from a source drawn with the mixture probabilities, then from a
uniformly drawn slot of that source, both from seeded generators. The file
order of each source is a seeded permutation per pass (epoch); a source that
runs out starts its next pass, so small corpora (AMI) repeat while YODAS does
not. The parent yields items from the workers in strict round robin.

Mixture. DESIGN.md gives target shares of exposure; by default ("hours") the
per-item source probabilities are share / mean kept duration, normalized, so the
expected share of audio hours matches the table. weighting="utterances" uses the
shares directly as per-item probabilities.

Filters (inline): 1.0 <= duration <= 30.0 s; audio decodable, nonempty, finite
and not all zeros; YODAS rows from the 16 reserved development shards are never
read (the shards are not in the file list) and rows whose source recording
appears in a reserved shard are dropped. Every drop is counted per source and
reason. Audio is decoded with soundfile and resampled (soxr) only when the
source is not 16 kHz mono.

Resume semantics. state_dict() records, per worker, the source mixture and slot
generator states, each source's file cursor and, for every open slot, its file
and absolute row position, as of the last item the caller consumed (items a
worker had prefetched but the caller had not consumed are produced again). A
stream rebuilt with the same configuration and load_state_dict() yields exactly
the same next items, in the same order, as the uninterrupted stream (verified in
tests/test_stream.py). Resuming re-reads at most the current row group of each
open slot. The state is a small JSON-serializable dict. It is tied to the
configuration hash (file lists, mixture and mean durations, seed, filters, open
files per source, read block size, worker count); loading it into a different
configuration raises ValueError rather than resuming at a wrong position.

Memory. Parquet files are opened with pre_buffer=False and an 8 MB read buffer,
and the Hub file object has no cache: pyarrow's default pre-buffering keeps every
column chunk read so far alive while a file is open (the 2026-09-30 OOM). A
worker holds 6 open files (2 YODAS, 1 per other source) and stays near 2 GB.

Robustness. Every read is retried with exponential backoff (5 s doubling to
5 min, 12 attempts, about 30 minutes in total) and logged; a file that keeps failing
raises StreamError naming the source and file. A worker process that dies is
restarted from the state of its last consumed item (at most 5 times).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import multiprocessing as mp
import os
import queue as queue_mod
import sys
import time
import traceback
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

import data
import paths

SR = paths.SAMPLE_RATE
SHARES = {"yodas": 0.65, "librispeech": 0.10, "peoples_speech": 0.10, "voxpopuli": 0.10, "ami": 0.05}
# Mean kept utterance duration (s) per source, used only to turn hour shares into per-item
# probabilities. These are the initial estimates the stream was first run with; benchmarks on the
# Hub stream (65k+ kept items, 2026-10-01) measured yodas 7.76, librispeech 12.39, peoples_speech
# 14.39, voxpopuli 10.21, ami 3.90, and realized hour shares within ~1 point of SHARES. They are
# part of the configuration hash, so changing them invalidates existing stream checkpoints.
MEAN_SECONDS = {"yodas": 7.05, "librispeech": 12.30, "peoples_speech": 13.33, "voxpopuli": 10.23, "ami": 3.79}
DEFAULT_WORKERS = 8  # measured 2026-10-01: 4 -> 493, 6 -> 766, 8 -> 1001 audio-h/h at ~2.1 GB each; GPU ~580
STATE_VERSION = 1
LIBRISPEECH = {"repo": "openslr/librispeech_asr", "revision": "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
               "config": "all", "dirs": ("all/train.clean.100", "all/train.clean.360", "all/train.other.500"),
               "license": "CC-BY-4.0"}


class StreamError(RuntimeError):
    """A source failed persistently (or a worker crashed repeatedly)."""

    def __init__(self, source: str | None, message: str):
        super().__init__(f"source {source or '?'}: {message}")
        self.source = source


@dataclasses.dataclass(frozen=True)
class SourceSpec:
    name: str
    files: tuple[str, ...]       # "hf://datasets/<repo>@<revision>/<path>" URLs or local paths
    id_col: str
    text_col: str | None         # None: the corpus has no human transcript
    share: float
    slots: int = 2               # files open at once per worker
    group_col: str | None = None
    audio_col: str = "audio"


@dataclasses.dataclass(frozen=True)
class StreamConfig:
    sources: tuple[SourceSpec, ...]
    seed: int = paths.SEED
    weighting: str = "hours"     # or "utterances"
    mean_seconds: tuple[tuple[str, float], ...] = tuple(MEAN_SECONDS.items())
    min_seconds: float = data.MIN_SECONDS
    max_seconds: float = data.MAX_SECONDS
    exclude_groups: tuple[str, ...] = ()
    batch_rows: int = 16
    block_size: int = 8 * 2**20      # HTTP range request size; no remote cache (see _open_file)
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
                           "version": STATE_VERSION}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()


def _log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [stream pid {os.getpid()}] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------------------------

def _open_file(url: str, block_size: int):
    """A binary file object for a local path or an hf:// URL (HTTP range reads)."""
    if url.startswith("hf://"):
        from huggingface_hub import HfFileSystem
        return HfFileSystem().open(url[len("hf://"):], "rb", block_size=block_size, cache_type="none")
    return open(url, "rb")


class _Slot:
    """One open parquet file read sequentially in record batches from an absolute row position."""

    def __init__(self, cfg: StreamConfig, spec: SourceSpec, url: str, epoch: int, pos: int, row: int = 0):
        self.cfg, self.spec, self.url = cfg, spec, url
        self.epoch, self.pos, self.row = epoch, pos, row
        self.columns = [c for c in (spec.id_col, spec.text_col, spec.audio_col, spec.group_col) if c]
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

    def _open_at_row(self) -> None:
        import pyarrow.parquet as pq
        self.close()
        self._handle = _open_file(self.url, self.cfg.block_size)
        # pre_buffer=False: pyarrow's default pre-buffering keeps every column chunk read so far
        # alive while the file is open, so a worker's memory grew with bytes read (about 1.4 GB per
        # 586 MB shard; 7-8 GB per worker with 14 open files caused the 2026-09-30 OOM). Without it
        # memory per open file plateaus at roughly 0.3-0.4 GB.
        pf = pq.ParquetFile(self._handle, pre_buffer=False, buffer_size=self.cfg.block_size)
        meta = pf.metadata
        start, group = 0, 0
        while group < meta.num_row_groups and start + meta.row_group(group).num_rows <= self.row:
            start += meta.row_group(group).num_rows
            group += 1
        if group >= meta.num_row_groups:
            self._batches = iter(())
            return
        self._batches = pf.iter_batches(batch_size=self.cfg.batch_rows, row_groups=range(group, meta.num_row_groups),
                                        columns=self.columns, use_threads=False)
        skip = self.row - start
        while skip > 0:
            rows = next(self._batches).to_pylist()
            if len(rows) > skip:
                self._buffer = rows[skip:]
                break
            skip -= len(rows)

    def next_row(self) -> dict | None:
        """The next row of the file, or None at its end; transient errors are retried."""
        attempt = 0
        while not self._buffer:
            try:
                if self._batches is None:
                    self._open_at_row()
                    continue
                batch = next(self._batches, None)
                if batch is None:
                    self.close()
                    return None
                self._buffer = batch.to_pylist()
            except Exception as exc:
                attempt += 1
                self.retries += 1
                self.close()
                if attempt >= self.cfg.retry_attempts:
                    raise StreamError(self.spec.name, f"{self.url} row {self.row}: giving up after "
                                                      f"{attempt} attempts: {exc!r}") from exc
                wait = min(self.cfg.retry_max_seconds, self.cfg.retry_base_seconds * 2 ** (attempt - 1))
                _log(f"{self.spec.name} {self.url} row {self.row}: {exc!r}; retry {attempt} in {wait:.0f}s")
                time.sleep(wait)
        self.row += 1
        return self._buffer.pop(0)


def _empty_counters() -> dict:
    return {"rows": 0, "kept": 0, "kept_seconds": 0.0, "dropped": {}, "retries": 0, "files_opened": 0,
            "passes_started": 1}


class _SourceReader:
    """One source's files for one worker: seeded file order per pass, `slots` files open at once."""

    def __init__(self, cfg: StreamConfig, index: int, worker: int, partitions: int, state: dict | None):
        self.cfg, self.spec, self.index, self.worker = cfg, cfg.sources[index], index, worker
        self.files = sorted(self.spec.files)[worker::partitions]
        if not self.files:
            raise ValueError(f"source {self.spec.name}: no files for worker {worker} of {partitions}")
        self.rng = np.random.default_rng([cfg.seed, worker, index, 1])
        self.cursor = {"epoch": 0, "pos": 0}
        self.slots: list[_Slot | None] = [None] * self.spec.slots
        self.counters = _empty_counters()
        if state:
            self.rng.bit_generator.state = state["rng"]
            self.cursor = dict(state["cursor"])
            self.counters = copy.deepcopy(state["counters"])
            self.slots = [None if s is None else self._slot(s["epoch"], s["pos"], s["row"]) for s in state["slots"]]

    def _order(self, epoch: int) -> np.ndarray:
        return np.random.default_rng([self.cfg.seed, self.worker, self.index, 2, epoch]).permutation(len(self.files))

    def _slot(self, epoch: int, pos: int, row: int = 0) -> _Slot:
        return _Slot(self.cfg, self.spec, self.files[self._order(epoch)[pos]], epoch, pos, row)

    def _next_file(self) -> _Slot:
        if self.cursor["pos"] >= len(self.files):
            self.cursor = {"epoch": self.cursor["epoch"] + 1, "pos": 0}
            self.counters["passes_started"] += 1
        slot = self._slot(self.cursor["epoch"], self.cursor["pos"])
        self.cursor["pos"] += 1
        self.counters["files_opened"] += 1
        return slot

    def next_row(self) -> dict:
        k = int(self.rng.integers(len(self.slots)))
        for _ in range(len(self.files) + 1):
            if self.slots[k] is None:
                self.slots[k] = self._next_file()
            slot = self.slots[k]
            before = slot.retries
            row = slot.next_row()
            self.counters["retries"] += slot.retries - before
            if row is not None:
                return row
            self.slots[k] = None  # file finished: open the next one in this slot
        raise StreamError(self.spec.name, "every file is empty")

    def state(self) -> dict:
        return {"rng": self.rng.bit_generator.state, "cursor": dict(self.cursor),
                "slots": [None if s is None else s.state() for s in self.slots],
                "counters": copy.deepcopy(self.counters)}

    def close(self) -> None:
        for slot in self.slots:
            if slot is not None:
                slot.close()


def _decode(data_bytes: bytes | None) -> tuple[np.ndarray | None, str | None]:
    if not data_bytes:
        return None, "empty_audio"
    pcm, reason, _ = data.convert(data_bytes)
    return pcm, reason


class _WorkerStream:
    """The item stream of one partition of the files (runs in a worker process or in-process)."""

    def __init__(self, cfg: StreamConfig, worker: int, partitions: int, state: dict | None):
        self.cfg, self.worker = cfg, worker
        self.probs = cfg.probabilities()
        self.mix = np.random.default_rng([cfg.seed, worker, 0])
        self.exclude = frozenset(cfg.exclude_groups)
        sstates = (state or {}).get("sources", {})
        self.readers = [_SourceReader(cfg, i, worker, partitions, sstates.get(s.name))
                        for i, s in enumerate(cfg.sources)]
        if state:
            self.mix.bit_generator.state = state["mix"]

    def state(self) -> dict:
        return {"mix": self.mix.bit_generator.state, "sources": {r.spec.name: r.state() for r in self.readers}}

    def next_item(self) -> dict:
        reader = self.readers[int(self.mix.choice(len(self.readers), p=self.probs))]
        spec, c = reader.spec, reader.counters
        while True:  # draw from the chosen source until a row passes the filters
            row = reader.next_row()
            c["rows"] += 1
            reason = None
            if self.exclude and spec.group_col and row.get(spec.group_col) in self.exclude:
                reason = "dev_recording_overlap"
            pcm = None
            if reason is None:
                pcm, reason = _decode((row.get(spec.audio_col) or {}).get("bytes"))
            if reason is None:
                reason = data.duration_drop_reason(pcm.size / SR, self.cfg.min_seconds, self.cfg.max_seconds)
            if reason:
                c["dropped"][reason] = c["dropped"].get(reason, 0) + 1
                continue
            duration = pcm.size / SR
            c["kept"] += 1
            c["kept_seconds"] += duration
            text = row.get(spec.text_col) if spec.text_col else None
            text = text.strip() if isinstance(text, str) else ""
            return {"audio": pcm.astype(np.float32) / 32768.0, "duration": duration, "text": text or None,
                    "id": f"{spec.name}:{row[spec.id_col]}", "source": spec.name}

    def close(self) -> None:
        for r in self.readers:
            r.close()


def _worker_main(cfg: StreamConfig, worker: int, partitions: int, state: dict | None,
                 out: mp.Queue, stop: mp.Event) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        import pyarrow as pa
        pa.set_cpu_count(1)
        pa.set_io_thread_count(2)
        stream = _WorkerStream(cfg, worker, partitions, state)
        while not stop.is_set():
            msg = ("item", stream.next_item(), stream.state())
            while not stop.is_set():
                try:
                    out.put(msg, timeout=1.0)
                    break
                except queue_mod.Full:
                    continue
    except StreamError as exc:
        out.put(("error", exc.source, str(exc)))
    except BaseException:
        out.put(("error", None, traceback.format_exc()))


# ---------------------------------------------------------------------------------------------
# Public stream
# ---------------------------------------------------------------------------------------------

class TrainingStream:
    """Infinite, resumable, mixed item stream. num_workers=0 runs in-process (one partition)."""

    def __init__(self, cfg: StreamConfig, num_workers: int = DEFAULT_WORKERS, phase: str = "pilot",
                 prefetch: int = 64, max_restarts: int = 5):
        if num_workers < 0 or num_workers > 32:
            raise ValueError("num_workers must be in 0..32")
        self.cfg, self.num_workers, self.phase = cfg, num_workers, phase
        self.partitions = max(1, num_workers)
        self.prefetch, self.max_restarts = prefetch, max_restarts
        self.config_hash = cfg.digest(self.partitions)
        self._wstates: list[dict | None] = [None] * self.partitions
        self._next_worker = 0
        self._items = 0
        self._started = False
        self._procs: list = []
        self._queues: list = []
        self._restarts = 0
        self._inproc: _WorkerStream | None = None

    # -- state ---------------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"version": STATE_VERSION, "config_hash": self.config_hash, "phase": self.phase,
                "seed": self.cfg.seed, "partitions": self.partitions, "next_worker": self._next_worker,
                "items": self._items, "workers": copy.deepcopy(self._wstates)}

    def load_state_dict(self, state: dict) -> None:
        if self._started:
            raise RuntimeError("load_state_dict must be called before iterating")
        if state.get("version") != STATE_VERSION or state.get("config_hash") != self.config_hash:
            raise ValueError("stream state was saved with a different configuration "
                             "(file lists, mixture, seed, filters or worker count)")
        self._wstates = copy.deepcopy(state["workers"])
        self._next_worker, self._items = state["next_worker"], state["items"]

    def counters(self) -> dict:
        """Per-source and global counters as of the last consumed item."""
        per: dict[str, dict] = {s.name: _empty_counters() | {"passes_started": 0} for s in self.cfg.sources}
        for ws in self._wstates:
            for name, st in (ws or {}).get("sources", {}).items():
                c, src = per[name], st["counters"]
                for key in ("rows", "kept", "kept_seconds", "retries", "files_opened", "passes_started"):
                    c[key] += src[key]
                for reason, n in src["dropped"].items():
                    c["dropped"][reason] = c["dropped"].get(reason, 0) + n
        total_seconds = sum(c["kept_seconds"] for c in per.values())
        for c in per.values():
            c["kept_hours"] = c["kept_seconds"] / 3600
            c["share_of_hours"] = c["kept_seconds"] / total_seconds if total_seconds else 0.0
        return {"items": self._items, "kept_hours": total_seconds / 3600,
                "rows": sum(c["rows"] for c in per.values()),
                "dropped": sum(sum(c["dropped"].values()) for c in per.values()),
                "retries": sum(c["retries"] for c in per.values()),
                "worker_restarts": self._restarts, "sources": per}

    # -- workers -------------------------------------------------------------------------
    def _spawn(self, w: int) -> None:
        ctx = mp.get_context("spawn")
        q = ctx.Queue(maxsize=self.prefetch)
        p = ctx.Process(target=_worker_main, args=(self.cfg, w, self.partitions, self._wstates[w], q, self._stop),
                        daemon=True, name=f"stream-worker-{w}")
        p.start()
        if w < len(self._procs):
            self._procs[w], self._queues[w] = p, q
        else:
            self._procs.append(p)
            self._queues.append(q)

    def _start(self) -> None:
        self._started = True
        if self.num_workers == 0:
            self._inproc = _WorkerStream(self.cfg, 0, 1, self._wstates[0])
            return
        self._stop = mp.get_context("spawn").Event()
        for w in range(self.partitions):
            self._spawn(w)

    def _get(self, w: int) -> tuple:
        while True:
            try:
                return self._queues[w].get(timeout=5.0)
            except queue_mod.Empty:
                if not self._procs[w].is_alive():
                    self._restarts += 1
                    if self._restarts > self.max_restarts:
                        self.close()
                        raise StreamError(None, f"worker {w} died {self._restarts} times")
                    _log(f"worker {w} died (exit {self._procs[w].exitcode}); restarting from its last "
                         f"consumed item")
                    self._spawn(w)

    def __iter__(self) -> Iterator[dict]:
        if not self._started:
            self._start()
        while True:
            w = self._next_worker
            if self._inproc is not None:
                item = self._inproc.next_item()
                wstate = self._inproc.state()
            else:
                msg = self._get(w)
                if msg[0] == "error":
                    self.close()
                    raise StreamError(msg[1], msg[2])
                _, item, wstate = msg
            self._wstates[w] = wstate
            self._next_worker = (w + 1) % self.partitions
            self._items += 1
            yield item

    def close(self) -> None:
        if self._inproc is not None:
            self._inproc.close()
        if self._procs:
            self._stop.set()
            for q in self._queues:  # drain so blocked puts return
                try:
                    while True:
                        q.get_nowait()
                except Exception:
                    pass
            for p in self._procs:
                p.join(timeout=10)
                if p.is_alive():
                    p.terminate()
            self._procs, self._queues = [], []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------------------------
# The experiment's sources
# ---------------------------------------------------------------------------------------------

def _hf_url(repo: str, revision: str, path: str) -> str:
    return f"hf://datasets/{repo}@{revision}/{path}"


def _librispeech_files() -> list[dict]:
    cache = paths.DATA / "hf" / "librispeech.train.files.json"
    if cache.exists():
        return json.loads(cache.read_text())["files"]
    from huggingface_hub import HfApi
    api = HfApi()
    files = []
    for d in LIBRISPEECH["dirs"]:
        for f in api.list_repo_tree(LIBRISPEECH["repo"], path_in_repo=d, repo_type="dataset",
                                    revision=LIBRISPEECH["revision"]):
            if f.path.endswith(".parquet"):
                files.append({"path": f.path, "size": f.size, "sha256": f.lfs.sha256 if f.lfs else None})
    files.sort(key=lambda f: f["path"])
    data.atomic_write_json(cache, {"repo": LIBRISPEECH["repo"], "revision": LIBRISPEECH["revision"],
                                   "listed": data._now(), "files": files})
    return files


def yodas_training_files(files: Sequence[str], reserved: Sequence[int]) -> list[str]:
    """sorted(files) without the reserved indices (indices into sorted(files))."""
    ordered = sorted(files)
    excluded = set(reserved)
    return [f for i, f in enumerate(ordered) if i not in excluded]


def source_files() -> dict[str, list[str]]:
    """Training parquet URLs per source at the pinned revisions (listings cached on the data disk)."""
    out = {}
    spec = data.HF_SOURCES["yodas"]
    files, reserved = data.yodas_reserved()
    keep = yodas_training_files([f["path"] for f in files], reserved)
    reserved_paths = {sorted(f["path"] for f in files)[i] for i in reserved}
    assert not reserved_paths & set(keep)
    out["yodas"] = [_hf_url(spec["repo"], spec["revision"], p) for p in keep]
    out["librispeech"] = [_hf_url(LIBRISPEECH["repo"], LIBRISPEECH["revision"], f["path"])
                          for f in _librispeech_files()]
    for name in ("peoples_speech", "voxpopuli", "ami"):
        spec = data.HF_SOURCES[name]
        out[name] = [_hf_url(spec["repo"], spec["revision"], f["path"]) for f in data.list_hf_files(name, "train")]
    return out


def file_list_sha256(files: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(files)).encode()).hexdigest()


def experiment_config(seed: int = paths.SEED, weighting: str = "hours") -> StreamConfig:
    files = source_files()
    hf = data.HF_SOURCES
    sources = (
        SourceSpec("yodas", tuple(files["yodas"]), hf["yodas"]["id_col"], None, SHARES["yodas"], slots=2,
                   group_col="original_audio_id"),
        SourceSpec("librispeech", tuple(files["librispeech"]), "id", "text", SHARES["librispeech"], slots=1),
        SourceSpec("peoples_speech", tuple(files["peoples_speech"]), "id", "text", SHARES["peoples_speech"],
                   slots=1),
        SourceSpec("voxpopuli", tuple(files["voxpopuli"]), "audio_id", "normalized_text", SHARES["voxpopuli"],
                   slots=1),
        SourceSpec("ami", tuple(files["ami"]), "audio_id", "text", SHARES["ami"], slots=1),
    )
    exclude = tuple(sorted(data.reserved_recordings()))
    cfg = StreamConfig(sources=sources, seed=seed, weighting=weighting, exclude_groups=exclude)
    record_sources(cfg)
    return cfg


def record_sources(cfg: StreamConfig) -> Path:
    """Write DATA/stream/sources.json (file lists, their hashes, revisions, mixture) once per config."""
    out = paths.DATA / "stream" / "sources.json"
    revisions = {"yodas": data.HF_SOURCES["yodas"], "librispeech": LIBRISPEECH,
                 **{n: data.HF_SOURCES[n] for n in ("peoples_speech", "voxpopuli", "ami")}}
    payload = {
        "seed": cfg.seed, "weighting": cfg.weighting, "shares": SHARES,
        "per_item_probabilities": dict(zip([s.name for s in cfg.sources], cfg.probabilities().round(6).tolist())),
        "mean_seconds": dict(cfg.mean_seconds), "filters": {"duration_seconds": [cfg.min_seconds, cfg.max_seconds]},
        "excluded_recordings": len(cfg.exclude_groups), "reserved_shards": str(data.reserved_path()),
        "sources": {s.name: {"repo": revisions[s.name]["repo"], "revision": revisions[s.name]["revision"],
                             "license": revisions[s.name]["license"], "files": len(s.files),
                             "files_sha256": file_list_sha256(s.files), "text_col": s.text_col,
                             "slots_per_worker": s.slots}
                    for s in cfg.sources}}
    data.atomic_write_json(out, payload)
    data.atomic_write_json(paths.DATA / "stream" / "yodas_train_files.json",
                           {"files": sorted(cfg.sources[0].files), "sha256": file_list_sha256(cfg.sources[0].files)})
    return out


def make_training_stream(phase: str, seed: int = paths.SEED, num_workers: int = DEFAULT_WORKERS,
                         weighting: str = "hours") -> TrainingStream:
    """The experiment's training stream. "pilot" and "main" are the same stream (same mixture
    and order); the pilot simply runs fewer steps."""
    if phase not in ("pilot", "main"):
        raise ValueError(f"phase must be 'pilot' or 'main', not {phase!r}")
    paths.require_mount()
    return TrainingStream(experiment_config(seed, weighting), num_workers=num_workers, phase=phase)


# ---------------------------------------------------------------------------------------------
# Throughput benchmark
# ---------------------------------------------------------------------------------------------

def _net_rx_bytes() -> int:
    total = 0
    for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
        name, rest = line.split(":", 1)
        if name.strip() != "lo":
            total += int(rest.split()[0])
    return total


def _cpu_seconds(pids: Sequence[int]) -> float:
    tick = os.sysconf("SC_CLK_TCK")
    total = 0.0
    for pid in pids:
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            total += (int(fields[11]) + int(fields[12])) / tick
        except (FileNotFoundError, ProcessLookupError, IndexError):
            pass
    return total


def _rss_mb(pids: Sequence[int]) -> float:
    total = 0
    for pid in pids:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])
        except FileNotFoundError:
            pass
    return total / 1024


def bench(num_workers: int, seconds: float, warmup: float) -> dict:
    stream = make_training_stream("pilot", num_workers=num_workers)
    pids = lambda: [os.getpid()] + [p.pid for p in stream._procs]  # noqa: E731
    t0 = time.time()
    it = iter(stream)
    mark = None
    for item in it:
        now = time.time()
        if mark is None and now - t0 >= warmup:
            mark = {"t": now, "hours": stream.counters()["kept_hours"], "items": stream._items,
                    "rx": _net_rx_bytes(), "cpu": _cpu_seconds(pids())}
        if mark is not None and now - mark["t"] >= seconds:
            break
    end = {"t": time.time(), "c": stream.counters(), "rx": _net_rx_bytes(), "cpu": _cpu_seconds(pids()),
           "rss": _rss_mb(pids())}
    stream.close()
    wall = end["t"] - mark["t"]
    hours = end["c"]["kept_hours"] - mark["hours"]
    result = {"workers": num_workers, "measured_seconds": round(wall, 1), "warmup_seconds": warmup,
              "audio_hours": round(hours, 3), "audio_hours_per_wall_hour": round(hours * 3600 / wall, 1),
              "items_per_second": round((end["c"]["items"] - mark["items"]) / wall, 1),
              "network_MB_per_s": round((end["rx"] - mark["rx"]) / wall / 1e6, 1),
              "cpu_cores_used": round((end["cpu"] - mark["cpu"]) / wall, 2), "rss_MB": round(end["rss"]),
              "time_to_first_hours": round(mark["t"] - t0, 1),
              "per_source": {n: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()}
                             for n, c in end["c"]["sources"].items()},
              "retries": end["c"]["retries"], "finished": data._now()}
    out = paths.DATA / "stream" / "bench.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as handle:
        handle.write(json.dumps(result) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bench", help="measure delivered audio hours per wall hour (no GPU)")
    b.add_argument("--workers", type=int, nargs="+", default=[4, 8])
    b.add_argument("--seconds", type=float, default=240)
    b.add_argument("--warmup", type=float, default=60)
    sub.add_parser("sources", help="list the training files and write DATA/stream/sources.json")
    args = parser.parse_args(argv)
    paths.require_mount()
    if args.command == "sources":
        cfg = experiment_config()
        for s, p in zip(cfg.sources, cfg.probabilities()):
            print(f"{s.name}: {len(s.files)} files, share {s.share}, per-item p {p:.4f}, "
                  f"sha256 {file_list_sha256(s.files)[:16]}")
        return
    for n in args.workers:
        print(json.dumps(bench(n, args.seconds, args.warmup)), flush=True)


if __name__ == "__main__":
    main()
