"""C1, the pool replay (labeling plan Part C "C1, pool replay"; CPU only, no GPU lease).

Replays the pinned stream at W through consumed batch TOTAL_BATCHES (batches.iter_batches = train.Batcher's
sequence) and writes, per shard of SHARD_BATCHES consumed batches (shards.py): the consumption journal
(every id occurrence, in order, with batch index, id, source alias and duration) and, for each id seen for
the first time, its clean, unaugmented PCM16 samples as FLAC with the stream's reference and its kind.

- Lossless: the stream's float32 audio is int16 / 32768 exactly; it is converted back (shards.to_int16, which
  refuses anything else) and stored as PCM16 FLAC with the SHA-256 of the samples.
- Bounded buffering: the stream's own queues, one consumed batch, and at most AUDIO_ROW_GROUP encoded clips
  before they are written; FLAC encoding runs in a small thread pool per batch.
- Distinct ids: a 64-bit hash of each id seen so far maps to a 64-bit hash of its samples. A later occurrence
  whose samples differ is an id conflict (two different clips under one id): recorded per shard and reported by
  the gate (labels are per id, so a conflict is never silently merged).
- Resumable at shard granularity: a shard is committed by its meta file (written last). A restart removes
  uncommitted files, rebuilds the seen-id map from the committed shards, and, by default, re-derives the last
  committed shard from the position before it and checks that it reproduces that shard's journal and new ids
  exactly (ResumeMismatch otherwise) before writing the next shard.
- Chunked: --budget-minutes stops at the first shard boundary after the budget; the next chunk resumes.

The output directory is $RELABEL_ROOT/pool-audio/<config id>/ for the pinned configuration; probes and smoke
runs write under $RELABEL_ROOT/smoke/labeling/.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import time
from pathlib import Path

from labeling import config as lconfig
from labeling import layout, shards
from labeling.batches import iter_batches


class ResumeMismatch(RuntimeError):
    """A resumed replay did not reproduce the committed shard it re-derived."""


def h64(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")


def _hex64(hexdigest: str) -> int:
    return int(hexdigest[:16], 16)


def _log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [replay] {msg}", flush=True)


def check_config(out_dir: Path, pinned: dict) -> None:
    path = Path(out_dir) / "config.json"
    if path.exists():
        recorded = json.loads(path.read_text())
        if recorded.get("pinned_hash") != pinned["pinned_hash"]:
            raise RuntimeError(f"{out_dir} was written by another configuration")
    else:
        lconfig.write_json(path, pinned)


def committed_prefix(out_dir: Path) -> list[int]:
    ks = shards.committed(out_dir)
    if ks != list(range(len(ks))):
        raise RuntimeError(f"committed shards are not a contiguous prefix: {ks[:5]}...")
    return ks


def load_seen(out_dir: Path, ks: list[int]) -> dict[int, int]:
    seen: dict[int, int] = {}
    for k in ks:
        for r in shards.iter_audio(shards.paths(out_dir, k)["audio"], columns=("id", "pcm_sha256"), batch_rows=8192):
            seen[h64(r["id"])] = _hex64(r["pcm_sha256"])
    return seen


class _Shard:
    def __init__(self, out_dir: Path, k: int, write: bool):
        self.k, self.write = k, write
        self.p = shards.paths(out_dir, k)
        self.journal: list[dict] = []
        self.new: list[tuple[str, str]] = []       # (id, pcm_sha256) of first occurrences
        self.conflicts: list[str] = []
        self.occ_seconds = 0.0
        self.writer = shards.AudioWriter(self.p["audio"]) if write else None


def replay(pinned: dict, out_dir: Path, *, total_batches: int = layout.TOTAL_BATCHES,
           shard_batches: int = layout.SHARD_BATCHES, budget_s: float | None = None, num_workers: int | None = None,
           encode_threads: int = 4, verify_resume: bool = True, stop_after_shards: int | None = None,
           stream_factory=None, keep_audio: bool = True) -> dict:
    """Run (or continue) C1 into out_dir; returns the status. keep_audio False (probes) writes the audio
    files and deletes them once the shard is committed (the meta keeps their SHA-256)."""
    out_dir = Path(out_dir)
    check_config(out_dir, pinned)
    removed = shards.remove_uncommitted(out_dir)
    if removed:
        _log(f"removed uncommitted files: {removed}")
    ks = committed_prefix(out_dir)
    n_total = -(-total_batches // shard_batches)
    status_path = out_dir / "status.json"
    if len(ks) >= n_total:
        return finish(out_dir, pinned, total_batches, shard_batches)
    factory = stream_factory or (lambda: lconfig.make_stream(pinned, num_workers))
    stream = factory()
    batch_seconds, seed = pinned["batch"]["batch_seconds"], pinned["batch"]["seed"]
    start_shard = ks[-1] if (ks and verify_resume) else len(ks)
    seen = load_seen(out_dir, ks[:start_shard])
    position = shards.load_meta(out_dir, start_shard - 1)["position_after"] if start_shard > 0 else None
    prefiltered: dict = {}
    t0 = time.monotonic()
    written = 0
    pool = cf.ThreadPoolExecutor(max_workers=max(1, encode_threads), thread_name_prefix="flac")
    it = iter_batches(stream, batch_seconds, seed, position=position, prefiltered=prefiltered)
    cur: _Shard | None = None
    try:
        for b in range(start_shard * shard_batches, total_batches):
            if cur is None:
                k = b // shard_batches
                cur = _Shard(out_dir, k, write=k >= len(ks))
            items, after = next(it)
            fresh = []
            for pos, item in enumerate(items):
                idh = h64(item["id"])
                pcm = shards.to_int16(item["audio"])
                digest = shards.pcm_sha256(pcm)
                first = idh not in seen
                if first:
                    seen[idh] = _hex64(digest)
                    cur.new.append((item["id"], digest))
                    fresh.append((item, pcm, digest))
                elif seen[idh] != _hex64(digest):
                    cur.conflicts.append(item["id"])
                cur.journal.append({"batch": b, "pos": pos, "id": item["id"], "source": item["source"],
                                    "duration": float(item["duration"]), "first": first})
                cur.occ_seconds += float(item["duration"])
            if cur.write and fresh:
                flacs = list(pool.map(lambda f: shards.encode_flac(f[1]), fresh))
                for (item, pcm, digest), flac in zip(fresh, flacs):
                    cur.writer.add({"id": item["id"], "source": item["source"], "duration": float(item["duration"]),
                                    "samples": int(pcm.size), "reference": item.get("reference"),
                                    "reference_kind": item.get("reference_kind"), "pcm_sha256": digest,
                                    "flac": flac})
            last = (b + 1) % shard_batches == 0 or b + 1 == total_batches
            if not last:
                continue
            if cur.write:
                _commit(out_dir, pinned, cur, b, shard_batches, after, stream, prefiltered, keep_audio)
                written += 1
                _log(f"shard {cur.k} committed: batches {cur.k * shard_batches}-{b}, {len(cur.journal)} occurrences, "
                     f"{len(cur.new)} new ids, {time.monotonic() - t0:.0f} s")
            else:
                _verify(out_dir, cur)
                _log(f"resume check: shard {cur.k} reproduced exactly ({len(cur.journal)} occurrences)")
            cur = None
            if b + 1 < total_batches and ((budget_s is not None and time.monotonic() - t0 >= budget_s)
                                          or (stop_after_shards is not None and written >= stop_after_shards)):
                break
    except BaseException:
        if cur is not None and cur.writer is not None:
            cur.writer.abort()
        raise
    finally:
        pool.shutdown(wait=True)
        stream.close()
    done = len(committed_prefix(out_dir))
    if done >= n_total:
        return finish(out_dir, pinned, total_batches, shard_batches)
    status = {"complete": False, "shards_committed": done, "shards_total": n_total,
              "written_this_run": written, "seconds_this_run": round(time.monotonic() - t0, 1),
              "pinned_hash": pinned["pinned_hash"]}
    lconfig.write_json(status_path, status)
    return status


def _commit(out_dir: Path, pinned: dict, cur: _Shard, b: int, shard_batches: int, after: dict, stream,
            prefiltered: dict, keep_audio: bool) -> None:
    sha_a, size_a = cur.writer.commit()
    sha_j, size_j = shards.write_journal(cur.p["journal"], cur.journal)
    meta = {"shard": cur.k, "kind": "replay", "pinned_hash": pinned["pinned_hash"],
            "batches": [cur.k * shard_batches, b + 1], "occurrences": len(cur.journal), "new_ids": len(cur.new),
            "occurrence_seconds": cur.occ_seconds,
            "audio_seconds_new": float(sum(r["duration"] for r in cur.journal if r["first"])),
            "id_conflicts": cur.conflicts,
            "files": {"audio": {"name": cur.p["audio"].name, "sha256": sha_a, "bytes": size_a},
                      "journal": {"name": cur.p["journal"].name, "sha256": sha_j, "bytes": size_j}},
            "position_after": after, "stream_counters": stream.counters(), "prefiltered": dict(prefiltered),
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if not keep_audio:
        cur.p["audio"].unlink()
        meta["audio_deleted"] = True
    shards.write_meta(cur.p["meta"], meta)


def _verify(out_dir: Path, cur: _Shard) -> None:
    stored = shards.read_journal(cur.p["journal"])
    if stored != cur.journal:
        raise ResumeMismatch(f"shard {cur.k}: the resumed replay does not reproduce its journal")
    meta = shards.verify_shard(out_dir, cur.k)
    if not meta.get("audio_deleted"):
        audio = [(r["id"], r["pcm_sha256"]) for r in shards.iter_audio(cur.p["audio"], columns=("id", "pcm_sha256"))]
        if audio != cur.new:
            raise ResumeMismatch(f"shard {cur.k}: the resumed replay does not reproduce its new clips")
    if meta["id_conflicts"] != cur.conflicts:
        raise ResumeMismatch(f"shard {cur.k}: id conflicts differ on resume")


def finish(out_dir: Path, pinned: dict, total_batches: int, shard_batches: int) -> dict:
    """manifest.json (every shard's meta; journals' SHA-256 and audio sizes checked) and status.json "complete"."""
    out_dir = Path(out_dir)
    ks = committed_prefix(out_dir)
    metas = [shards.verify_shard(out_dir, k, audio_hash=False) for k in ks]
    expect = 0
    for m in metas:
        if m["batches"][0] != expect:
            raise RuntimeError(f"shard {m['shard']}: batch range {m['batches']} does not follow {expect}")
        expect = m["batches"][1]
    if expect != total_batches:
        raise RuntimeError(f"shards cover {expect} batches, not {total_batches}")
    manifest = {"pinned_hash": pinned["pinned_hash"], "config_id": lconfig.config_id(pinned),
                "batches": total_batches, "shard_batches": shard_batches, "shards": len(metas),
                "occurrences": sum(m["occurrences"] for m in metas), "distinct_ids": sum(m["new_ids"] for m in metas),
                "occurrence_hours": sum(m["occurrence_seconds"] for m in metas) / 3600,
                "distinct_hours": sum(m["audio_seconds_new"] for m in metas) / 3600,
                "audio_bytes": sum(m["files"]["audio"]["bytes"] for m in metas),
                "id_conflicts": [i for m in metas for i in m["id_conflicts"]],
                "stream_counters": metas[-1]["stream_counters"],
                "shard_files": [{"shard": m["shard"], "batches": m["batches"], **{k: v for k, v in m["files"].items()}}
                                for m in metas]}
    lconfig.write_json(out_dir / "manifest.json", manifest)
    status = {"complete": True, "shards_committed": len(metas), "shards_total": len(metas),
              "pinned_hash": pinned["pinned_hash"], "manifest_sha256": shards.sha256_file(out_dir / "manifest.json")}
    lconfig.write_json(out_dir / "status.json", status)
    return status


# --- independent replay (gate item 3) ---------------------------------------------------------------

def journal_of(pinned: dict, n_batches: int, num_workers: int | None = None, stream_factory=None) -> list[dict]:
    """The journal of the first n_batches consumed batches from a fresh stream (nothing written)."""
    factory = stream_factory or (lambda: lconfig.make_stream(pinned, num_workers))
    stream = factory()
    out, seen = [], set()
    try:
        it = iter_batches(stream, pinned["batch"]["batch_seconds"], pinned["batch"]["seed"])
        for b in range(n_batches):
            items, _ = next(it)
            for pos, item in enumerate(items):
                first = item["id"] not in seen
                seen.add(item["id"])
                out.append({"batch": b, "pos": pos, "id": item["id"], "source": item["source"],
                            "duration": float(item["duration"]), "first": first,
                            "pcm_sha256": shards.pcm_sha256(shards.to_int16(item["audio"])) if first else None})
    finally:
        stream.close()
    return out


def stored_journal(out_dir: Path, n_batches: int) -> list[dict]:
    rows = []
    for k in committed_prefix(out_dir):
        meta = shards.load_meta(out_dir, k)
        if meta["batches"][0] >= n_batches:
            break
        rows += [r for r in shards.read_journal(shards.paths(out_dir, k)["journal"]) if r["batch"] < n_batches]
    return rows


# --- replay probe and the choice of W -------------------------------------------------------------

def probe(candidate: dict, workers: int, minutes: float, out_dir: Path) -> dict:
    """C1 for `minutes` at W=workers from batch 0 into out_dir (audio written, then deleted per shard)."""
    import shutil
    from labeling.monitor import Monitor
    pinned = lconfig.pin(candidate, workers)
    if Path(out_dir).exists():
        shutil.rmtree(out_dir)
    with Monitor(interval=1.0) as mon:
        t0 = time.monotonic()
        status = replay(pinned, out_dir, budget_s=minutes * 60, num_workers=workers, keep_audio=False,
                        shard_batches=layout.SHARD_BATCHES)
        wall = time.monotonic() - t0
    metas = [shards.load_meta(out_dir, k) for k in shards.committed(out_dir)]
    occ_h = sum(m["occurrence_seconds"] for m in metas) / 3600
    new_h = sum(m["audio_seconds_new"] for m in metas) / 3600
    batches = sum(m["batches"][1] - m["batches"][0] for m in metas)
    res = mon.result()
    result = {"workers": workers, "minutes": minutes, "wall_s": wall, "shards": len(metas), "batches": batches,
              "occurrence_hours": occ_h, "distinct_hours": new_h,
              "audio_hours_per_wall_hour": occ_h / (wall / 3600), "distinct_hours_per_wall_hour": new_h / (wall / 3600),
              "batches_per_wall_hour": batches / (wall / 3600),
              "flac_bytes_per_distinct_hour": (sum(m["files"]["audio"]["bytes"] for m in metas) / new_h) if new_h else None,
              "host_peak_gb": res["peak_tree_rss_gb"], "monitor": res, "status": status,
              "pinned_hash": pinned["pinned_hash"], "candidate_sha256": lconfig.sha256_json(candidate)}
    lconfig.write_json(Path(out_dir) / "probe.json", result)
    return result


def choose_workers(results: dict[int, dict], limit_gb: float = layout.HOST_PEAK_LIMIT_GB,
                   faster_by: float = layout.FASTER_BY) -> dict:
    """labeling plan "Worker sizing" step 2: 8 is the baseline if its host peak is within the limit; 12, then 16,
    replaces the current candidate only if its peak is within the limit and it is at least 10% faster."""
    base = min(layout.WORKER_CANDIDATES)
    r = results.get(base)
    if r is None:
        raise ValueError(f"no result for W={base}")
    if r["host_peak_gb"] > limit_gb:
        return {"workers": None, "stop": True, "reason": f"W={base} host peak {r['host_peak_gb']:.1f}G > {limit_gb}G"}
    chosen, notes = base, []
    for w in sorted(layout.WORKER_CANDIDATES)[1:]:
        r = results.get(w)
        if r is None:
            notes.append(f"W={w}: not measured")
            continue
        cur = results[chosen]["audio_hours_per_wall_hour"]
        if r["host_peak_gb"] <= limit_gb and r["audio_hours_per_wall_hour"] >= (1 + faster_by) * cur:
            notes.append(f"W={w} replaces W={chosen}")
            chosen = w
        else:
            notes.append(f"W={w} kept out (peak {r['host_peak_gb']:.1f}G, rate {r['audio_hours_per_wall_hour']:.0f} "
                         f"vs {cur:.0f})")
    return {"workers": chosen, "stop": False, "notes": notes}


def cost(probe_result: dict, total_batches: int = layout.TOTAL_BATCHES) -> dict:
    """C1 wall time extrapolated from the replay probe at W (batches per wall hour)."""
    rate = probe_result["batches_per_wall_hour"]
    return {"workers": probe_result["workers"], "batches_per_wall_hour": rate,
            "c1_wall_hours": total_batches / rate if rate else None,
            "occurrence_hours_estimate": probe_result["occurrence_hours"] / probe_result["batches"] * total_batches
            if probe_result["batches"] else None,
            "note": "extrapolated from a 15-minute probe at the start of the stream; later passes over small "
                    "sources (repeats) write less audio"}
