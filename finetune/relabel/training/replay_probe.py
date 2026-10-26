"""The replay throughput probe (b0prime plan "Replay throughput probe", 2026-10-24). CPU only.

Times the batch plan's readers, as training uses them (replay.ReplayBatcher: decode threads, bounded prefetch,
train.collate), on regions of the plan given as pool ranges, and compares each region's rate with the rate
training needs: REQUIRED_MARGIN / the fastest decoder-only probe step time (decoder-only/probe/<B|T>.json; the
fastest student consumes batches fastest).
Readers: "materialized" (training/poolaudio.py) and "c1" (C1's shards in plan order, replay.AudioStore). Every
region gets a fresh reader (no application cache carried over; the previous reader's prefetches are finished
first), and the page cache of every file the reader may touch is dropped (posix_fadvise DONTNEED), so a region
starts cold. The consumer takes batches as fast as they come, so the rate is the reader's capacity; a region
stops early after --max-seconds. Labels are not needed (the plan is teacher-independent): every row is served
with a placeholder text.

    ./python -m training.batchplan probe --pools 156 [--regions 0:3,76:79,153:156] [--readers materialized,c1]

Output: $RELABEL_ROOT/training/batch-plan/probes/<plan id>.json (smoke runs: --out under $RELABEL_ROOT/smoke/).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

from labeling import layout, shards

REQUIRED_MARGIN = 1.25
FALLBACK_STEP_S = 0.43
MAX_SECONDS = 120.0


class AnyText:
    """A label store stand-in: every id has a text."""

    def text(self, item_id: str) -> str:
        return "x"


def required_rate() -> tuple[float, dict]:
    steps = {}
    for s in ("B", "T"):
        p = layout.relabel_root() / "decoder-only" / "probe" / f"{s}.json"
        if p.exists():
            steps[s] = float(json.loads(p.read_text())["step_time_s"])
    step = min(steps.values()) if steps else FALLBACK_STEP_S
    return REQUIRED_MARGIN / step, {"step_time_s": steps or {"fallback": FALLBACK_STEP_S}, "margin": REQUIRED_MARGIN}


def drop_cache(paths) -> int:
    n = 0
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
        except FileNotFoundError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            n += 1
        finally:
            os.close(fd)
    return n


def hwm_gib() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 2**20
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def rss_gib() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 2**20
    return float("nan")


def time_region(replay, start: int, stop: int, max_seconds: float) -> dict:
    from training import replay as treplay
    t0 = time.monotonic()
    b = treplay.ReplayBatcher(replay, AnyText(), start=start, stop=stop)
    n, audio_s, rows, waits = 0, 0.0, 0, []
    try:
        while True:
            w0 = time.monotonic()
            try:
                batch = b.next(timeout=600)
            except treplay.ReplayStopped:
                break
            waits.append(time.monotonic() - w0)
            n += 1
            audio_s += float(batch["seconds"])
            rows += len(batch["ids"])
            if time.monotonic() - t0 > max_seconds:
                break
    finally:
        b.close()
    dt = time.monotonic() - t0
    waits.sort()
    return {"planned_batches": n, "of": stop - start, "seconds": dt, "batches_per_s": n / dt if dt else None,
            "audio_seconds_per_s": audio_s / dt if dt else None, "rows": rows,
            "wait_p50_s": waits[len(waits) // 2] if waits else None,
            "wait_p90_s": waits[int(len(waits) * 0.9)] if waits else None, "wait_max_s": waits[-1] if waits else None,
            "stopped_early": n < stop - start}


def run(plan, pool_dir: Path, store: Path, regions: list[tuple[int, int]], readers: list[str],
        max_seconds: float = MAX_SECONDS) -> dict:
    from training import poolaudio
    from training import replay as treplay
    from training.labels import needed_shards
    need, need_info = required_rate()
    out = {"plan": plan.identity(), "required_batches_per_s": need, "required": need_info, "regions": [],
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    c1_files = [shards.paths(pool_dir, k)["audio"] for k in needed_shards(pool_dir, plan.consumed_batches)]
    for reader in readers:
        for lo, hi in regions:
            t0 = time.monotonic()
            r = treplay.PlannedReplay(plan, pool_dir, audio=reader, store=store)
            opened = time.monotonic() - t0
            if reader == "materialized":
                files = [store / "pools" / f"{poolaudio.pool_name(p)}.flac" for p in range(lo, min(hi + 1, plan.n_pools))]
            else:
                files = c1_files
            dropped = drop_cache(files)
            res = time_region(r, plan.pool_start(lo), plan.pool_start(hi), max_seconds)
            res.update(reader=reader, pools=[lo, hi], open_seconds=opened, cache_dropped_files=dropped,
                       rss_gib=rss_gib(), hwm_gib=hwm_gib(),
                       meets_required=res["batches_per_s"] is not None and res["batches_per_s"] >= need)
            out["regions"].append(res)
            print(json.dumps(res), flush=True)
            r.close()
            del r
    out["host_peak_gib"] = hwm_gib()
    mat = [x for x in out["regions"] if x["reader"] == "materialized"]
    out["materialized_meets_required"] = bool(mat) and all(x["meets_required"] for x in mat)
    return out


def parse_regions(text: str) -> list[tuple[int, int]]:
    out = []
    for part in text.split(","):
        lo, hi = part.split(":")
        out.append((int(lo), int(hi)))
    return out


def main(argv: list[str] | None = None) -> int:
    from training import batchplan, poolaudio
    ap = argparse.ArgumentParser(prog="python -m training.batchplan probe")
    ap.add_argument("--pools", type=int, required=True, help="the production plan over pools 0 .. N-1")
    ap.add_argument("--regions", help="pool ranges lo:hi, comma-separated (default: start, middle, end)")
    ap.add_argument("--readers", default="materialized,c1")
    ap.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    ap.add_argument("--plan-dir", type=Path, help="a smoke plan (with --store and --out)")
    ap.add_argument("--store", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    layout.require_mount()
    pool, cid, _ = batchplan._c1_and_config()
    plan = batchplan.Plan.load(a.plan_dir or batchplan.production_dir(cid, a.pools))
    store = a.store or poolaudio.store_dir(cid, plan.pool_batches)
    n = plan.n_pools
    regions = parse_regions(a.regions) if a.regions else [(0, 3), (n // 2 - 1, n // 2 + 2), (n - 3, n)]
    out_path = a.out or layout.relabel_root() / "training" / "batch-plan" / "probes" / f"{plan.meta['plan_id']}.json"
    if a.plan_dir is not None and layout.is_production(out_path):
        raise SystemExit("a smoke probe writes under $RELABEL_ROOT/smoke/")
    res = run(plan, pool, store, regions, a.readers.split(","), a.max_seconds)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "regions"}, indent=1))
    return 0
