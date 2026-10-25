"""The journal replay from C1's local shards (b0prime plan "Data", decoder-only plan "Data").

C1 (labeling/replay.py) recorded every consumed batch of the pinned stream: the journal (batch, pos, id, source,
duration) and, for each id at its first occurrence, its clean PCM16 samples (FLAC, with their SHA-256). A
JournalReplay serves consumed batch b as the stream served it: the journal's ids in `pos` order, each with the
stream's float32 audio (int16 / 32768, exact) read from the shard holding its first occurrence. Every clip's
samples are checked against the stored SHA-256; journals are checked against their metas' SHA-256 and audio
files against their sizes when the replay opens. The labeled sequence is that journal, so training on a replay
is batch for batch the labeled sequence (tests/test_training_data.py checks it against a fresh stream replay,
labeling.batches = train.Batcher's sequence, and train.collate).

Consumed-batch accounting: batch indices count from 0 and include batches whose rows are all rejected (they
are consumed and take no optimizer step). A ReplayBatcher serves [start, stop) and then raises ReplayStopped;
its positions are {"batch": next index}, so a run resumes at the batch after its checkpoint's. plan() counts
the batches with at least one labeled row (the optimizer steps of a run that stops at `stop`).

Extra clips (open for stage 2c, unused by stage 2b): an optional `extra` object with
`items(index, real_items) -> list[dict]` ({"audio" float32, "duration", "id", "source", "label"}) appends clips
with their own labels after the real rows of batch `index`; an optional `extra.transform(item, index)` is
applied to each extra clip only. Without `extra` the batches are exactly the replayed ones.

PlannedReplay (2026-10-24, b0prime plan "Batch plan"): the same interface over a batch plan
(training/batchplan.py): position t serves the plan's planned batch t (its rows in cut order) instead of consumed
batch t, so ReplayBatcher, plan() and the trainer's seam index planned batches, and a run's stop and resume points
are planned-batch indices. Audio comes from materialized pool audio (training/poolaudio.py, read pool by pool, the
next pool prefetched) or, for checks, from C1's shards (AudioStore); either way every clip's samples are checked
against C1's stored SHA-256. The plan is verified against the journal when the replay opens.

A version 2 plan (stage 2c's synthetic arms) needs its clip pool (`synthetic`, a synthetic.mixing.ClipPool) and
serves its synthetic rows in place, in cut order among the real ones: {"id": "syn:<clip id>", "source":
"synthetic", "duration", "audio", "label"}; the audio is the clip pool's file (SHA-256 checked) of exactly the
clip's samples, passed through `synthetic_transform(item, b)` when given (the TS arms' augmentation; TSn passes
none), and refused if longer than the plan's key for it. ReplayBatcher takes a row's text from its "label" when it
has one (synthetic rows), else from the label store.
"""
from __future__ import annotations

import collections
import concurrent.futures as cf
import hashlib
import json
import queue
import threading

import numpy as np

from labeling import shards

PREFETCH = 4
DECODE_THREADS = 6
ROW_GROUP_CACHE = 48


class ReplayStopped(RuntimeError):
    """The batcher reached its stop batch."""


class AudioStore:
    """Clips by id from C1 audio shards: (shard, row group, row) index, an LRU cache of raw row groups."""

    def __init__(self, pool_dir, ks: list[int], cache_groups: int = ROW_GROUP_CACHE):
        import pyarrow.parquet as pq
        self.pool_dir, self.cache_groups = pool_dir, cache_groups
        self.index: dict[str, tuple[int, int, int]] = {}
        self.files = {}
        for k in ks:
            path = shards.paths(pool_dir, k)["audio"]
            pf = pq.ParquetFile(path)
            self.files[k] = path
            ids = pf.read(columns=["id"]).column("id").to_pylist()
            pos = 0
            for g in range(pf.metadata.num_row_groups):
                n = pf.metadata.row_group(g).num_rows
                for j in range(n):
                    i = ids[pos + j]
                    if i in self.index:
                        raise RuntimeError(f"id {i} stored twice in C1 audio")
                    self.index[i] = (k, g, j)
                pos += n
        self._cache: collections.OrderedDict = collections.OrderedDict()
        self._lock = threading.Lock()

    def _group(self, k: int, g: int) -> dict:
        import pyarrow.parquet as pq
        with self._lock:
            hit = self._cache.get((k, g))
            if hit is not None:
                self._cache.move_to_end((k, g))
                return hit
        t = pq.ParquetFile(self.files[k]).read_row_group(g, columns=["id", "samples", "pcm_sha256", "flac"]).to_pydict()
        with self._lock:
            self._cache[(k, g)] = t
            while len(self._cache) > self.cache_groups:
                self._cache.popitem(last=False)
        return t

    def raw(self, item_id: str) -> tuple[bytes, int, str]:
        k, g, j = self.index[item_id]
        t = self._group(k, g)
        if t["id"][j] != item_id:
            raise RuntimeError(f"C1 audio index mismatch for {item_id}")
        return t["flac"][j], t["samples"][j], t["pcm_sha256"][j]

    @staticmethod
    def decode(item_id: str, raw: tuple[bytes, int, str]) -> np.ndarray:
        data, samples, digest = raw
        pcm = shards.decode_flac(data)
        if pcm.size != samples or shards.pcm_sha256(pcm) != digest:
            raise RuntimeError(f"{item_id}: decoded samples differ from C1's stored SHA-256")
        return pcm.astype(np.float32) / 32768.0


class JournalReplay:
    """Consumed batches [0, through_batch) of a C1 directory, from its journals and audio shards."""

    def __init__(self, pool_dir, through_batch: int, decode_threads: int = DECODE_THREADS):
        from training.labels import needed_shards
        self.pool_dir, self.through_batch = pool_dir, int(through_batch)
        self.ks = needed_shards(pool_dir, self.through_batch)
        self.batches: list[list[tuple[str, str, float]]] = [[] for _ in range(self.through_batch)]
        listed, pinned = [], set()
        for k in self.ks:
            meta = shards.verify_shard(pool_dir, k, audio_hash=False)
            if meta.get("kind") != "replay" or meta.get("audio_deleted"):
                raise RuntimeError(f"C1 shard {k} is not a replay shard with audio")
            pinned.add(meta["pinned_hash"])
            rows = shards.read_journal(shards.paths(pool_dir, k)["journal"])
            for r in rows:
                b = r["batch"]
                if not meta["batches"][0] <= b < meta["batches"][1]:
                    raise RuntimeError(f"journal of shard {k} has batch {b} outside {meta['batches']}")
                if b < self.through_batch:
                    if r["pos"] != len(self.batches[b]):
                        raise RuntimeError(f"journal batch {b}: position {r['pos']} out of order")
                    self.batches[b].append((r["id"], r["source"], float(r["duration"])))
            listed.append({"shard": k, "batches": meta["batches"], "journal_sha256": meta["files"]["journal"]["sha256"],
                           "audio_sha256": meta["files"]["audio"]["sha256"], "audio_bytes": meta["files"]["audio"]["bytes"]})
        if len(pinned) != 1:
            raise RuntimeError(f"C1 shards from {len(pinned)} pinned configurations")
        empty = [b for b, rows in enumerate(self.batches) if not rows]
        if empty:
            raise RuntimeError(f"journal has no rows for batches {empty[:5]}")
        self.audio = AudioStore(pool_dir, self.ks)
        missing = {i for rows in self.batches for i, _, _ in rows} - set(self.audio.index)
        if missing:
            raise RuntimeError(f"{len(missing)} replayed ids have no audio in C1 shards {self.ks[0]}-{self.ks[-1]}")
        self.decode_threads = decode_threads
        self.identity = {"pinned_hash": pinned.pop(), "through_batch": self.through_batch, "shards": listed}
        self.identity["sha256"] = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()

    def batch_ids(self, b: int) -> list[str]:
        return [i for i, _, _ in self.batches[b]]

    def read(self, b: int, pool: cf.Executor | None = None) -> list[dict]:
        """The items of consumed batch b: {"id", "source", "duration", "audio" float32}, in journal order."""
        if not 0 <= b < self.through_batch:
            raise IndexError(f"batch {b} outside the replay's [0, {self.through_batch})")
        rows = self.batches[b]
        raws = [self.audio.raw(i) for i, _, _ in rows]
        if pool is None:
            audio = [AudioStore.decode(i, r) for (i, _, _), r in zip(rows, raws)]
        else:
            audio = list(pool.map(AudioStore.decode, [i for i, _, _ in rows], raws))
        return [{"id": i, "source": s, "duration": d, "audio": a} for (i, s, d), a in zip(rows, audio)]


class PlannedReplay:
    """Planned batches [0, n_batches) of a batch plan, with JournalReplay's interface (module docstring)."""

    AUDIO = ("materialized", "c1")

    def __init__(self, plan, pool_dir, audio: str = "materialized", store=None, decode_threads: int = DECODE_THREADS,
                 synthetic=None, synthetic_transform=None):
        from training import batchplan, poolaudio
        from training.labels import needed_shards
        if audio not in self.AUDIO:
            raise ValueError(f"audio must be one of {self.AUDIO}")
        if (plan.version == batchplan.SYNTHETIC_VERSION) != (synthetic is not None):
            raise ValueError("a version 2 plan needs its clip pool, and only a version 2 plan takes one")
        self.plan, self.pool_dir, self.kind = plan, pool_dir, audio
        j = plan.verify_journal(pool_dir, clips=synthetic)
        self.through_batch, self.consumed_batches = plan.n_batches, plan.consumed_batches
        self._ids = j["ids"]
        ids, src, dur = j["ids"], j["sources"], j["durations"]
        self.synthetic, self.synthetic_transform = synthetic, synthetic_transform
        self._clips = sorted(synthetic.clips, key=lambda c: c["id"]) if synthetic is not None else []
        csamples = batchplan.clip_samples(synthetic) if synthetic is not None else []
        flag = batchplan.SYN_FLAG
        self._syn = {}                                     # (batch, row) -> (clip, samples, key) of synthetic rows
        self.batches: list[list[tuple[str, str, float]]] = []
        for t in range(plan.n_batches):
            rows = []
            for r, e in enumerate(plan.batch(t).tolist()):
                if e >= flag:
                    c = self._clips[e - flag]
                    self._syn[(t, r)] = (c, csamples[e - flag], batchplan.synthetic_key(csamples[e - flag]))
                    rows.append((f"syn:{c['id']}", "synthetic", csamples[e - flag] / batchplan.SAMPLE_RATE))
                else:
                    rows.append((ids[e], src[e], dur[e]))
            self.batches.append(rows)
        self.pool_starts = plan.batch_pools.tolist()
        self._pool_of = np.repeat(np.arange(plan.n_pools), np.diff(plan.batch_pools))
        self._checked: set[int] = set()
        if audio == "materialized":
            if store is None:
                raise ValueError("materialized audio needs its store")
            self.audio = poolaudio.PoolAudio(store, plan)
            audio_identity = {"kind": "materialized", **self.audio.identity()}
        else:
            ks = needed_shards(pool_dir, self.consumed_batches)
            for k in ks:
                meta = shards.verify_shard(pool_dir, k, audio_hash=False)
                if meta.get("kind") != "replay" or meta.get("audio_deleted"):
                    raise RuntimeError(f"C1 shard {k} is not a replay shard with audio")
            self.audio = AudioStore(pool_dir, ks)
            missing = set(ids) - set(self.audio.index)
            if missing:
                raise RuntimeError(f"{len(missing)} planned ids have no audio in C1 shards {ks[0]}-{ks[-1]}")
            audio_identity = {"kind": "c1"}
        self.decode_threads = decode_threads
        self.identity = {"kind": "batch-plan", **plan.identity(), "pinned_hash": plan.meta["pinned_hash"],
                         "through_batch": self.through_batch, "audio": audio_identity}
        if synthetic is not None:
            self.identity["synthetic"] = {"clip_pool": str(synthetic.identity), "transform": None if synthetic_transform
                                          is None else getattr(synthetic_transform, "identity", repr(type(synthetic_transform)))}
        self.identity["sha256"] = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()

    def batch_ids(self, b: int) -> list[str]:
        return [i for i, _, _ in self.batches[b]]

    def synthetic_mask(self, b: int) -> list[bool]:
        return [(b, r) in self._syn for r in range(len(self.batches[b]))]

    def pool_of(self, b: int) -> int:
        return int(self._pool_of[b])

    def _pool_clips(self, p: int) -> dict:
        clips = self.audio.load(p)
        if p not in self._checked:
            lo, hi = int(self.plan.real_pools[p]), int(self.plan.real_pools[p + 1])
            if set(clips) != set(self._ids[lo:hi]):
                raise RuntimeError(f"materialized pool {p} holds other ids than the plan's pool")
            self._checked.add(p)
        self.audio.prefetch(p + 1)
        return clips

    def close(self) -> None:
        """Finish outstanding prefetches and release cached audio."""
        if self.kind == "materialized":
            self.audio.drain()

    def read(self, b: int, pool: cf.Executor | None = None) -> list[dict]:
        """The items of planned batch b: {"id", "source", "duration", "audio" float32}, in cut order."""
        if not 0 <= b < self.through_batch:
            raise IndexError(f"batch {b} outside the plan's [0, {self.through_batch})")
        all_rows = self.batches[b]
        real = [r for r in range(len(all_rows)) if (b, r) not in self._syn]
        rows = [all_rows[r] for r in real]
        if self.kind == "materialized":
            clips = self._pool_clips(self.pool_of(b))
            raws = [clips[i] for i, _, _ in rows]
        else:
            raws = [self.audio.raw(i) for i, _, _ in rows]
        if pool is None:
            audio = [AudioStore.decode(i, r) for (i, _, _), r in zip(rows, raws)]
        else:
            audio = list(pool.map(AudioStore.decode, [i for i, _, _ in rows], raws))
        out: list[dict | None] = [None] * len(all_rows)
        for r, (i, s, d), a in zip(real, rows, audio):
            out[r] = {"id": i, "source": s, "duration": d, "audio": a}
        for r in range(len(all_rows)):
            if (b, r) in self._syn:
                out[r] = self._synthetic_item(b, r)
        return out

    def _synthetic_item(self, b: int, r: int) -> dict:
        c, samples, key = self._syn[(b, r)]
        a = self.synthetic.read(c["id"])
        if len(a) != samples:
            raise RuntimeError(f"synthetic clip {c['id']}: {len(a)} samples, the plan has {samples}")
        item = {"id": f"syn:{c['id']}", "source": "synthetic", "duration": samples / 16000, "audio": a,
                "label": c["label"]}
        if self.synthetic_transform is not None:
            item = self.synthetic_transform(item, b)
            if len(item["audio"]) > key or item["label"] != c["label"]:
                raise RuntimeError(f"batch {b}: the transform of {c['id']} exceeds its planned length or changed its label")
        return item


def plan(replay: JournalReplay, labels, stop: int) -> dict:
    """Consumed-batch accounting of a run that stops at consumed batch `stop`: optimizer steps (batches with at
    least one labeled row), fully rejected batches (consumed, no step), occurrences."""
    if stop > replay.through_batch:
        raise ValueError(f"stop {stop} beyond the replay's {replay.through_batch} batches")
    empty, occ, kept, syn, last = [], 0, 0, 0, None
    for b in range(stop):
        ids = replay.batch_ids(b)
        mask = replay.synthetic_mask(b) if hasattr(replay, "synthetic_mask") else [False] * len(ids)
        k = sum(labels.text(i) is not None for i, m in zip(ids, mask) if not m)
        s = sum(mask)
        occ += len(ids) - s
        kept += k
        syn += s
        if k + s:
            last = b
        else:
            empty.append(b)
    return {"stop_batch": int(stop), "steps": stop - len(empty), "empty_batches": empty, "occurrences": occ,
            "kept_occurrences": kept, "synthetic_rows": syn, "last_step_batch": last}


class ReplayBatcher:
    """train.Batcher's interface over a replay: next() -> train.collate(items) plus "index" (the consumed-batch
    index), "position_after" {"batch": index + 1}, "labels" (per row: the label store's text, None = rejected;
    extra clips carry their own) and "extra_rows"; produced by a background thread with a bounded queue.
    Errors reach the consumer as in train.Batcher (stream/OS errors as train.StreamFailure)."""

    def __init__(self, replay: JournalReplay, labels, start: int, stop: int, extra=None, prefetch: int = PREFETCH):
        if not 0 <= start <= stop <= replay.through_batch:
            raise ValueError(f"invalid range [{start}, {stop}) for a replay of {replay.through_batch} batches")
        self.replay, self.labels, self.start, self.stop, self.extra = replay, labels, start, stop, extra
        self.queue: queue.Queue = queue.Queue(maxsize=prefetch)
        self.halt = threading.Event()
        self.pool = cf.ThreadPoolExecutor(max_workers=max(1, replay.decode_threads), thread_name_prefix="flac")
        self.thread = threading.Thread(target=self._run, name="replay-batcher", daemon=True)
        self.thread.start()

    def _put(self, item) -> bool:
        while not self.halt.is_set():
            try:
                self.queue.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def assemble(self, b: int) -> dict:
        import train  # finetune/parakeet-ternary
        items = self.replay.read(b, self.pool)
        texts = [it["label"] if "label" in it else self.labels.text(it["id"]) for it in items]
        extra = []
        if self.extra is not None:
            extra = list(self.extra.items(b, items))
            transform = getattr(self.extra, "transform", None)
            if transform is not None:
                extra = [transform(it, b) for it in extra]
            texts += [it["label"] for it in extra]
        batch = train.collate(items + extra)
        batch.update(index=b, position_after={"batch": b + 1}, labels=texts, extra_rows=len(extra))
        return batch

    def _run(self) -> None:
        try:
            for b in range(self.start, self.stop):
                if self.halt.is_set() or not self._put(self.assemble(b)):
                    return
            self._put(ReplayStopped(f"replay reached its stop batch {self.stop}"))
        except BaseException as error:  # noqa: BLE001 - forwarded to the consumer
            self._put(error)

    def next(self, timeout: float | None = None) -> dict:
        import train
        item = self.queue.get(timeout=timeout)
        if isinstance(item, ReplayStopped):
            self._put(item)   # every later call raises too
            raise item
        if isinstance(item, BaseException):
            if train.is_stream_error(item):
                raise train.StreamFailure(f"{type(item).__name__}: {item}") from item
            raise RuntimeError(f"replay batcher failed: {type(item).__name__}: {item}") from item
        return item

    def close(self, close_stream: bool = True) -> None:
        self.halt.set()
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.thread.join(timeout=10)
        self.pool.shutdown(wait=True)
