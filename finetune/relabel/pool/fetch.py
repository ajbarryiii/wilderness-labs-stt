"""The text cache: one local parquet per training file of a streamed source, text columns only.

    <out>/<source>/files.json            the file list read (paths, sizes, sha256; the stream's files_sha256)
    <out>/<source>/<key>.parquet         one row per parquet row of the source file (CACHE_COLUMNS)
    <out>/<source>/<key>.done.json       written last (atomic): counts, bytes fetched, timing

Resumable at file granularity: a file with a done marker is never read again; a file whose read fails is
logged and left without a marker (rerun to retry it); the run exits non-zero if any file failed.
"""
from __future__ import annotations

import concurrent.futures as cf
import io
import json
import math
import multiprocessing as mp
import os
import random
import time
from collections.abc import Sequence
from pathlib import Path

from pool import common, remote

CACHE_COLUMNS = ("row", "row_group", "recording", "order", "duration", "text", "norm")


def _open_source(spec: common.TextSource, entry: dict, threads: int):
    if entry.get("local_path"):  # tests and local copies
        return remote.LocalRangeSource(entry["local_path"])
    return remote.HubRangeSource(spec.repo, spec.revision, entry["path"], entry.get("size"), entry.get("sha256"),
                                 threads=threads)


def cache_paths(out_dir: Path, entry: dict) -> tuple[Path, Path]:
    key = common.file_key(entry["path"])
    return out_dir / f"{key}.parquet", out_dir / f"{key}.done.json"


def fetch_file(source: str, entry: dict, out_dir: str | Path, threads: int = 8,
               specs: dict | None = None) -> dict:
    """Read one file's projected text columns and write its cache shard; idempotent via the done marker."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    out_dir = Path(out_dir)
    out, done = cache_paths(out_dir, entry)
    if done.exists():
        return json.loads(done.read_text())
    spec = (specs or common.text_sources())[source]
    t0 = time.time()
    src = _open_source(spec, entry, threads)
    try:
        tables, stats = remote.read_projected(src, spec.columns)
    finally:
        src.close()
    cols = {c: [] for c in CACHE_COLUMNS}
    row_index, empty_text, missing_duration, seconds = 0, 0, 0, 0.0
    recordings = set()
    for g, table in tables:
        d = table.to_pydict()
        for i in range(table.num_rows):
            row = {c: d[c][i] for c in spec.columns}
            rec, order = spec.recording(row)
            text = row[spec.text_col]
            text = text.strip() if isinstance(text, str) else ""
            dur = row[spec.duration_col]
            if dur is None or not math.isfinite(float(dur)):
                missing_duration += 1
                dur = 0.0
            dur = float(dur) * spec.duration_scale
            norm = common.normalize(text) if text else ""
            empty_text += not norm
            seconds += dur
            recordings.add(rec)
            for c, v in (("row", row_index), ("row_group", g), ("recording", rec), ("order", order),
                         ("duration", dur), ("text", text), ("norm", norm)):
                cols[c].append(v)
            row_index += 1
    if row_index != stats["rows"]:
        raise RuntimeError(f"{entry['path']}: read {row_index} rows, metadata says {stats['rows']}")
    schema = pa.schema([("row", pa.int32()), ("row_group", pa.int32()), ("recording", pa.string()),
                        ("order", pa.float64()), ("duration", pa.float32()), ("text", pa.string()),
                        ("norm", pa.string())])
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pydict(cols, schema=schema), buf, compression="zstd")
    common.atomic_write_bytes(out, buf.getvalue())
    summary = {"source": source, "path": entry["path"], "size": entry.get("size"), "sha256": entry.get("sha256"),
               "rows": row_index, "row_groups": stats["row_groups"], "recordings": len(recordings),
               "empty_text_rows": empty_text, "missing_duration_rows": missing_duration,
               "hours": round(seconds / 3600, 6), "ranges": stats["ranges"],
               "bytes_fetched": stats["bytes_fetched"],
               "retries": getattr(src, "retries", 0), "seconds": round(time.time() - t0, 2),
               "cache_sha256": common.sha256_file(out), "finished": common.now()}
    common.atomic_write_json(done, summary)
    return summary


def _worker_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


def _job(args: tuple) -> tuple[str, dict | None, str | None]:
    source, entry, out_dir, threads = args
    try:
        return entry["path"], fetch_file(source, entry, out_dir, threads), None
    except Exception as exc:  # logged by the parent; the file stays without a done marker
        return entry["path"], None, repr(exc)


def write_listing(source: str, files: Sequence[dict], out_dir: Path) -> dict:
    listing = {"source": source, "files": len(files),
               "files_sha256": common.stream_files_sha256(source, files) if not any(
                   f.get("local_path") for f in files) else None,
               "listing": [{k: f[k] for k in ("path", "size", "sha256") if k in f} for f in files]}
    path = out_dir / "files.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["listing"] != listing["listing"]:
            raise RuntimeError(f"{path}: a different file list was fetched into this directory before")
        return old
    common.atomic_write_json(path, listing | {"created": common.now()})
    return listing


def run_fetch(source: str, files: Sequence[dict], out_dir: Path, workers: int, threads: int) -> dict:
    """Fetch every file without a done marker with `workers` processes x `threads` range requests each."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_listing(source, files, out_dir)
    todo = [f for f in files if not cache_paths(out_dir, f)[1].exists()]
    common.log(f"fetch {source}: {len(files)} files, {len(files) - len(todo)} already done, {len(todo)} to read "
               f"({workers} processes x {threads} threads)")
    t0, n, failed = time.time(), 0, []
    jobs = [(source, f, str(out_dir), threads) for f in todo]
    if workers <= 0:
        results = map(_job, jobs)
        pool = None
    else:
        pool = cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_worker_init)
        results = pool.map(_job, jobs, chunksize=1)
    try:
        for path, summary, error in results:
            n += 1
            if error:
                failed.append({"path": path, "error": error})
                common.log(f"fetch {source}: FAILED {path}: {error}")
            if n % 50 == 0 or n == len(jobs):
                rate = n / max(1e-9, time.time() - t0)
                common.log(f"fetch {source}: {n}/{len(jobs)} files, {rate * 3600:.0f} files/h, "
                           f"ETA {(len(jobs) - n) / max(rate, 1e-9) / 3600:.2f} h, {len(failed)} failed")
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    status = cache_status(source, files, out_dir)
    status.update(failed=failed, wall_seconds=round(time.time() - t0, 1))
    return status


def cache_status(source: str, files: Sequence[dict], out_dir: Path) -> dict:
    done = [cache_paths(out_dir, f)[1] for f in files]
    summaries = [json.loads(p.read_text()) for p in done if p.exists()]
    return {"source": source, "files": len(files), "done": len(summaries),
            "rows": sum(s["rows"] for s in summaries), "hours": round(sum(s["hours"] for s in summaries), 2),
            "bytes_fetched": sum(s["bytes_fetched"] for s in summaries),
            "file_bytes": sum(s["size"] or 0 for s in summaries)}


def run_probe(source: str, files: Sequence[dict], n: int, workers: int, threads: int, out_dir: Path) -> dict:
    """Time `n` seeded random files (into the probe directory, never the production cache) and project
    the full pass's wall time at the same concurrency."""
    sample = sorted(random.Random(common.SAMPLE_SEED).sample(list(files), min(n, len(files))),
                    key=lambda f: f["path"])
    sample_dir = out_dir / source / time.strftime("%Y%m%dT%H%M%S")
    t0 = time.time()
    status = run_fetch(source, sample, sample_dir, workers, threads)
    wall = time.time() - t0
    per = [json.loads(cache_paths(sample_dir, f)[1].read_text()) for f in sample
           if cache_paths(sample_dir, f)[1].exists()]
    result = {"source": source, "sample_files": len(sample), "done": len(per), "workers": workers,
              "threads": threads, "wall_seconds": round(wall, 1),
              "mean_file_seconds": round(sum(p["seconds"] for p in per) / max(1, len(per)), 2),
              "bytes_fetched": sum(p["bytes_fetched"] for p in per),
              "file_bytes": sum(p["size"] or 0 for p in per),
              "projected_hours": round(wall / max(1, len(per)) * len(files) / 3600, 2),
              "total_files": len(files), "failed": status["failed"], "dir": str(sample_dir),
              "finished": common.now()}
    common.atomic_write_json(sample_dir / "probe.json", result)
    return result
