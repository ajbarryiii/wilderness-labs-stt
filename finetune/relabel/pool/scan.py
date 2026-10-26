"""One local pass over the text cache (and the local corpora): per-recording statistics and held-out 8-gram hits.

Per recording: rows, seconds, seconds of rows inside the stream's 1-30 s filter, normalized words, medical and
technical term occurrences, the (file, row groups) holding its rows, and which held-out 8-grams occur in its
transcript (its rows' normalized words concatenated in recording order).

A first, cheap pass reads only the `recording` column of every cache shard to find the recordings whose rows
lie in more than one file. In the main pass, a recording held by one file is processed by the worker that
reads that file; the rows of a multi-file recording are sent back and merged in the parent once every file
holding it has been read, sorted by (order, file, row), so its 8-grams and term counts are those of the whole
recording, including across file boundaries (and correct even if files interleave its rows).

Corpora recordings (manifest rows with `recording`, ordered by `start`) take part in the 8-gram checks only.
Refuses a cache whose listing (files.json) has files without a done marker.
"""
from __future__ import annotations

import collections
import concurrent.futures as cf
import dataclasses
import json
import multiprocessing as mp
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from pool import common, fetch

_STATE: dict = {}
STAT_KEYS = ("rows", "seconds", "seconds_kept", "words", "medical", "technical")


def _init(hashes: np.ndarray, matchers: Mapping) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    _STATE.update(hashes=hashes, matchers=dict(matchers), hasher=common.WordHasher())


def _hits(gram_lists: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(local recording index, held-out hash index) pairs, unique."""
    hashes = _STATE["hashes"]
    lens = np.array([len(g) for g in gram_lists], dtype=np.int64)
    if not len(hashes) or not lens.sum():
        return np.empty(0, np.int64), np.empty(0, np.int64)
    flat = np.concatenate(gram_lists)
    owner = np.repeat(np.arange(len(gram_lists), dtype=np.int64), lens)
    idx = np.searchsorted(hashes, flat)
    idx[idx >= len(hashes)] = 0
    ok = hashes[idx] == flat
    key = np.unique((owner[ok] << 32) | idx[ok])
    return key >> 32, key & 0xFFFFFFFF


def _row_stats(durations: list[float], words: int) -> dict:
    return {"rows": len(durations), "seconds": float(sum(durations)),
            "seconds_kept": float(sum(d for d in durations if common.MIN_SECONDS <= d <= common.MAX_SECONDS)),
            "words": words, "medical": 0, "technical": 0}


def _text_stats(words: list[str], with_terms: bool) -> tuple[dict, np.ndarray]:
    """Term counts (if with_terms) and 8-gram hashes of one recording's complete word sequence."""
    terms = {}
    if with_terms:
        terms = {"medical": _STATE["matchers"]["medical"].count(words),
                 "technical": _STATE["matchers"]["technical"].count(words)}
    return terms, common.ngram_hashes(_STATE["hasher"](words))


def recordings_in(task: tuple) -> list[str]:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    return pc.unique(pq.read_table(task[1], columns=["recording"])["recording"]).to_pylist()


def scan_cache_file(task: tuple) -> dict:
    """Statistics and hits of every recording in one cache shard; the rows of `multi` recordings (also in
    other files) are returned for merging instead of being hashed here."""
    import pyarrow.parquet as pq
    source, cache_path, repo_path, multi = task
    multi = set(multi)
    t = pq.read_table(cache_path, columns=["row", "row_group", "recording", "order", "duration", "norm"]).to_pydict()
    groups: dict[str, list[int]] = {}
    for i, rec in enumerate(t["recording"]):
        groups.setdefault(rec, []).append(i)
    recs, stats, grams, pieces = [], [], [], {}
    for rec, idx in groups.items():
        idx.sort(key=lambda i: (t["order"][i], t["row"][i]))
        rows_words = [t["norm"][i].split() for i in idx]
        s = _row_stats([t["duration"][i] for i in idx], sum(len(w) for w in rows_words))
        s["row_groups"] = sorted({t["row_group"][i] for i in idx})
        if rec in multi:
            pieces[rec] = [(t["order"][i], t["row"][i], w) for i, w in zip(idx, rows_words)]
            g = np.empty(0, dtype=np.uint64)
        else:
            terms, g = _text_stats([w for ws in rows_words for w in ws], True)
            s.update(terms)
        recs.append(rec)
        stats.append(s)
        grams.append(g)
    hit_rec, hit_h = _hits(grams)
    return {"source": source, "file": repo_path, "recordings": recs, "stats": stats,
            "hit_rec": hit_rec, "hit_h": hit_h, "pieces": pieces}


def scan_corpus(alias: str, manifest: Path) -> dict:
    rows = common.read_jsonl(manifest)
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(str(r["recording"]), []).append(r)
    recs, stats, grams = [], [], []
    for rec, rs in groups.items():
        rs.sort(key=lambda r: (float(r.get("start", 0.0)), str(r["id"])))
        words = [w for r in rs for w in common.normalize(r.get("text")).split()]
        s = _row_stats([float(r.get("duration") or 0.0) for r in rs], len(words))
        s["row_groups"] = []
        _, g = _text_stats(words, False)
        recs.append(rec)
        stats.append(s)
        grams.append(g)
    hit_rec, hit_h = _hits(grams)
    return {"source": alias, "file": None, "recordings": recs, "stats": stats, "hit_rec": hit_rec, "hit_h": hit_h,
            "pieces": {}}


@dataclasses.dataclass
class ScanResult:
    keys: list[tuple[str, str]]              # gid -> (source, recording)
    stats: list[dict]                        # gid -> rows, seconds, seconds_kept, words, medical, technical
    locators: list[dict[str, set[int]]]      # gid -> {file: row groups}
    pairs: np.ndarray                        # unique (gid << 32 | held-out hash index), sorted
    sources: dict[str, dict]                 # per source / corpus: files, recordings, listing hash


def cache_tasks(source: str, cache_dir: Path) -> tuple[list[tuple[str, str, str]], dict]:
    listing_path = cache_dir / "files.json"
    if not listing_path.exists():
        raise RuntimeError(f"{cache_dir}: no files.json (run `pool fetch --source {source}`)")
    listing = json.loads(listing_path.read_text())
    tasks, missing = [], []
    for entry in listing["listing"]:
        out, done = fetch.cache_paths(cache_dir, entry)
        if not done.exists() or not out.exists():
            missing.append(entry["path"])
        tasks.append((source, str(out), entry["path"]))
    if missing:
        raise RuntimeError(f"{source}: {len(missing)} of {len(tasks)} files are not in the text cache "
                           f"(first: {missing[0]}); finish `pool fetch --source {source}` first")
    return tasks, {"files": len(tasks), "files_sha256": listing.get("files_sha256"),
                   "listing_sha256": common.sha256_file(listing_path)}


def _run(fn, tasks: list, workers: int, hashes: np.ndarray, matchers: Mapping, chunksize: int = 4):
    """fn over tasks, in order (in-process, or `workers` spawned processes)."""
    if workers <= 0:
        _init(hashes, matchers)
        yield from map(fn, tasks)
        return
    pool = cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init,
                                  initargs=(hashes, dict(matchers)))
    try:
        yield from pool.map(fn, tasks, chunksize=chunksize)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def scan(caches: Mapping[str, Path], corpora: Mapping[str, Path], hashes: np.ndarray, matchers: Mapping,
         workers: int = 0) -> ScanResult:
    tasks, sources = [], {}
    for source, cache_dir in caches.items():
        t, info = cache_tasks(source, Path(cache_dir))
        tasks += t
        sources[source] = info

    # Pass 1: which recordings have rows in more than one file.
    file_recs = list(_run(recordings_in, tasks, workers, hashes, matchers, chunksize=16))
    nfiles = collections.Counter((task[0], rec) for task, recs in zip(tasks, file_recs) for rec in recs)
    multi = {key: n for key, n in nfiles.items() if n > 1}
    tasks = [task + (sorted(rec for rec in recs if (task[0], rec) in multi),) for task, recs in zip(tasks, file_recs)]
    del file_recs, nfiles
    common.log(f"scan: {len(tasks)} files, {len(multi)} recordings span more than one file")

    keys: list[tuple[str, str]] = []
    gid_of: dict[tuple[str, str], int] = {}
    stats: list[dict] = []
    locators: list[dict[str, set[int]]] = []
    pair_chunks: list[np.ndarray] = []
    pending: dict[int, dict] = {}   # gid -> {"rows": [...], "seen": files}

    def add_pairs(gids: np.ndarray, hidx: np.ndarray) -> None:
        if len(gids):
            pair_chunks.append((gids.astype(np.uint64) << np.uint64(32)) | hidx.astype(np.uint64))

    def merge(gid: int, rows: list) -> None:
        rows.sort(key=lambda r: (r[0], r[1], r[2]))
        terms, g = _text_stats([w for r in rows for w in r[3]], True)
        stats[gid].update(terms)
        _, h = _hits([g])
        add_pairs(np.full(len(h), gid, dtype=np.int64), h)

    def absorb(res: dict, file_index: int | None) -> None:
        local = np.empty(len(res["recordings"]), dtype=np.int64)
        for j, (rec, s) in enumerate(zip(res["recordings"], res["stats"])):
            key = (res["source"], rec)
            gid = gid_of.get(key)
            if gid is None:
                gid = gid_of[key] = len(keys)
                keys.append(key)
                stats.append({k: 0 for k in STAT_KEYS})
                locators.append({})
            for k in STAT_KEYS:
                stats[gid][k] += s[k]
            if res["file"] is not None:
                locators[gid].setdefault(res["file"], set()).update(s["row_groups"])
            local[j] = gid
            if rec in res["pieces"]:
                p = pending.setdefault(gid, {"rows": [], "seen": 0})
                p["rows"] += [(order, file_index, row, words) for order, row, words in res["pieces"][rec]]
                p["seen"] += 1
                if p["seen"] == multi[key]:
                    merge(gid, pending.pop(gid)["rows"])
        add_pairs(local[res["hit_rec"]], res["hit_h"])

    _init(hashes, matchers)   # the parent merges multi-file recordings itself (also when workers > 0)
    for n, res in enumerate(_run(scan_cache_file, tasks, workers, hashes, matchers), 1):
        absorb(res, n)
        if n % 1000 == 0:
            common.log(f"scan: {n}/{len(tasks)} files, {len(keys)} recordings, {len(pending)} merges pending")
    if pending:
        raise RuntimeError(f"scan: {len(pending)} multi-file recordings were not complete after the pass")
    for alias, manifest in corpora.items():
        absorb(scan_corpus(alias, Path(manifest)), None)
        sources[alias] = {"manifest": str(manifest), "manifest_sha256": common.sha256_file(Path(manifest))}
    for name in sources:
        sources[name]["recordings"] = sum(1 for s, _ in keys if s == name)
    sources["_scan"] = {"multi_file_recordings": len(multi)}
    pairs = np.unique(np.concatenate(pair_chunks)) if pair_chunks else np.empty(0, dtype=np.uint64)
    return ScanResult(keys, stats, locators, pairs, sources)
