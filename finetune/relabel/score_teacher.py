"""Step 1 of diagnostic plan: score one teacher on the development sets.

    ./python score_teacher.py --teacher NAME [--sets dev|SET...] [--limit N [--shortest K]]
                              [--batch-size B] [--max-batch-seconds S] [--lock-wait SECONDS] [--force]

Reads the parakeet-ternary dev manifests (evaluate.resolve_manifest / read_manifest: id-sorted), decodes
every utterance with the teacher (teachers.py), and scores it with evaluate.score (the same normalizer,
edit counts and "scored" rule as the B0, M1 and MB1 records). Clips shorter than
evaluate.MIN_DECODE_SECONDS are zero-padded to it exactly as the B0 scorer does (evaluate.pad_short);
any further padding inside a teacher's own pipeline is recorded as internal_min_seconds.

Writes, per set, <out>/<set>.json in evaluate.py's per-set layout (records with id, ref, hyp, ref_norm,
hyp_norm, S, D, I, scored, duration) plus throughput fields, and <out>/summary.json. <out> is
relabel_paths.DIAGNOSTIC/<teacher> for a full run and relabel_paths.SMOKE/<teacher>-limit<N> with
--limit (never the production path). Sets already written by an identical configuration are skipped
(resumable after an interruption) unless --force.

Throughput (diagnostic plan Step 1): all audio of a set is read into memory before the timed pass,
so inference_wall_s excludes audio decoding; it covers feature extraction, encoding and generation for
every batch (batches length-sorted, longest first, at most --batch-size clips and --max-batch-seconds of
audio; a batch that runs out of GPU memory is split in halves and the split is counted). Model load time
is recorded separately. audio_hours_per_gpu_hour = audio hours / inference hours.

GPU discipline: the parakeet GPU lock (relabel_paths.GPU_LOCK, fcntl flock as paths.gpu_lock) is taken
before anything touches CUDA and held until the process exits. If it is not free within --lock-wait
seconds the run refuses to start (exit 3). The job lease (`job start --needs gpu`) is the caller's.
--device cpu (smoke runs only) hides CUDA, takes no lock and writes to the smoke path with a -cpu suffix.
The host has no swap: the teacher is not loaded if less than --min-available-gb of RAM is available (exit 4).
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import evaluate  # finetune/parakeet-ternary/evaluate.py: normalizer, score, manifests, audio, padding
import relabel_paths

LOCK_BUSY_EXIT = 3
LOW_MEMORY_EXIT = 4
PROJECTION_HOURS = 33000.0
PROJECTION_NOTE = ("estimate: dev audio is not the training mix (length and domain differ); "
                   "treat as +-30% for a go/no-go")


class LockBusy(RuntimeError):
    pass


@contextlib.contextmanager
def gpu_lock(label: str, wait_s: float, poll_s: float = 5.0):
    """Hold relabel_paths.GPU_LOCK (exclusive flock, the parakeet convention) or raise LockBusy after wait_s."""
    relabel_paths.require_mount()
    lock_path = Path(relabel_paths.GPU_LOCK)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as handle:
        deadline = time.monotonic() + wait_s
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    handle.seek(0)
                    holder = handle.read().strip()
                    raise LockBusy(f"{lock_path} is held ({holder or 'holder unknown'}); waited {wait_s:.0f} s")
                time.sleep(min(poll_s, max(0.0, deadline - time.monotonic())) or 0.01)
        handle.seek(0); handle.truncate(); handle.write(f"{os.getpid()} {label}\n"); handle.flush()
        try:
            yield
        finally:
            handle.seek(0); handle.truncate(); handle.flush()
            fcntl.flock(handle, fcntl.LOCK_UN)


def available_memory_gb() -> float:
    with open("/proc/meminfo") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def expand_sets(names: list[str]) -> list[str]:
    out = []
    for name in names:
        out += relabel_paths.SETS if name == "dev" else [name]
    unknown = [n for n in out if n not in relabel_paths.SETS]
    if unknown:
        raise SystemExit(f"not a development set: {unknown} (test sets are out of scope)")
    return list(dict.fromkeys(out))


def select_records(manifest: Path, limit: int | None, shortest: int) -> list[dict]:
    """First `limit` id-sorted utterances (evaluate.py --limit semantics), plus the `shortest` shortest ones."""
    records = evaluate.read_manifest(manifest)
    if limit is None:
        return records
    picked = records[:limit]
    ids = {r["id"] for r in picked}
    for r in sorted(records, key=lambda r: (r["duration"], r["id"]))[:shortest]:
        if r["id"] not in ids:
            picked.append(r); ids.add(r["id"])
    return sorted(picked, key=lambda r: r["id"])


def run_config(teacher, args) -> dict:
    return {"teacher": teacher.name, "revision": teacher.revision, "limit": args.limit, "shortest": args.shortest,
            "batch_size": args.batch_size, "max_batch_seconds": args.max_batch_seconds,
            "max_new_tokens": teacher.decoding()["max_new_tokens"], "device": args.device}


def decode_set(teacher, records: list[dict], batch_size: int, max_batch_seconds: float, device) -> tuple[list[str], dict]:
    import torch

    start = time.perf_counter()
    clips, padded = [], []
    for r in records:
        clip, was_padded = evaluate.pad_short(evaluate.load_audio(r["audio_filepath"]))
        clips.append(clip)
        if was_padded:
            padded.append(r["id"])
    audio_load_s = time.perf_counter() - start

    batches = evaluate.make_batches(records, batch_size, max_batch_seconds)
    texts: list[str | None] = [None] * len(records)
    capped: list[str] = []
    stats = {"oom_splits": 0, "batch_calls": 0}

    def run(idx: list[int]) -> None:
        out = hit = None
        try:
            out, hit = teacher.transcribe([clips[i] for i in idx])
            stats["batch_calls"] += 1
        except torch.cuda.OutOfMemoryError:
            if len(idx) == 1:
                raise
        if out is None:
            # Retry outside the handler, so the failed batch's traceback (and its tensors) is released first.
            torch.cuda.empty_cache()
            stats["oom_splits"] += 1
            half = len(idx) // 2
            run(idx[:half]); run(idx[half:])
            return
        if len(out) != len(idx):
            raise RuntimeError(f"{teacher.name} returned {len(out)} texts for {len(idx)} clips")
        for i, text, h in zip(idx, out, hit):
            texts[i] = text
            if h:
                capped.append(records[i]["id"])

    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for k, idx in enumerate(batches):
        run(idx)
        if k % 50 == 0:
            print(f"  batch {k + 1}/{len(batches)} ({time.perf_counter() - start:.0f} s)", flush=True)
    if cuda:
        torch.cuda.synchronize(device)
    wall = time.perf_counter() - start
    assert all(t is not None for t in texts)
    hours = sum(r["duration"] for r in records) / 3600
    throughput = {
        "batch_size": batch_size, "max_batch_seconds": max_batch_seconds, "batch_order": "descending duration",
        "batches": len(batches), "batch_calls": stats["batch_calls"], "oom_splits": stats["oom_splits"],
        "audio_load_s": audio_load_s, "inference_wall_s": wall, "audio_hours": hours,
        "audio_hours_per_gpu_hour": hours / (wall / 3600) if wall > 0 else None,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated(device) / 2**30 if cuda else None,
        "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved(device) / 2**30 if cuda else None,
    }
    extra = {"padded_short": len(padded), "padded_short_ids": sorted(padded),
             "hit_token_cap": len(capped), "hit_token_cap_ids": sorted(capped)}
    return texts, {"throughput": throughput, **extra}


def score_set(teacher, name: str, manifest: Path, args, device, source: dict) -> dict:
    records = select_records(manifest, args.limit, args.shortest)
    texts, stats = decode_set(teacher, records, args.batch_size, args.max_batch_seconds, device)
    out = [{"id": r["id"], "ref": r["text"], "hyp": hyp, **evaluate.score(r["text"], hyp), "duration": r["duration"]}
           for r, hyp in zip(records, texts)]
    t = stats["throughput"]
    return {"source": source, "set": name, "manifest": str(manifest), "manifest_sha256": evaluate.sha256_file(manifest),
            "utterances": len(out), "limit": args.limit, "shortest": args.shortest, "audio_hours": t["audio_hours"],
            "config": run_config(teacher, args), "decoding": teacher.decoding(), "normalizer": evaluate.NORMALIZER,
            "wer": evaluate.wer_summary(out) if any(r["scored"] for r in out) else None,
            "wall_s": t["inference_wall_s"], "model_load_s_this_run": teacher.load_s,
            "padded_short": stats["padded_short"], "padded_short_ids": stats["padded_short_ids"],
            "min_decode_seconds": evaluate.MIN_DECODE_SECONDS, "internal_min_seconds": teacher.internal_min_seconds,
            "hit_token_cap": stats["hit_token_cap"], "hit_token_cap_ids": stats["hit_token_cap_ids"],
            "throughput": t, "records": out}


def summarize(out_dir: Path, teacher_info: dict, config: dict, load_s: float | None) -> dict:
    per_set, results = {}, {}
    for name in relabel_paths.SETS:
        path = out_dir / f"{name}.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        if r.get("config") != config:
            continue
        results[name] = r
        w = r["wer"]
        per_set[name] = {"wer": 100 * w["wer"] if w else None, "utterances": r["utterances"],
                         "scored_utterances": w["utterances"] if w else 0, "audio_hours": r["audio_hours"],
                         "padded_short": r["padded_short"], "hit_token_cap": r["hit_token_cap"],
                         "throughput": r["throughput"]}
    complete = config["limit"] is None and set(results) == set(relabel_paths.SETS)
    hours = sum(r["throughput"]["audio_hours"] for r in results.values())
    wall = sum(r["throughput"]["inference_wall_s"] for r in results.values())
    rate = hours / (wall / 3600) if wall > 0 else None
    throughput = {
        "audio_hours": hours, "inference_wall_s": wall, "audio_hours_per_gpu_hour": rate,
        "model_load_s": load_s,
        "peak_gpu_memory_gb": max((r["throughput"]["peak_gpu_memory_gb"] or 0 for r in results.values()), default=None),
        "batch_size": config["batch_size"], "max_batch_seconds": config["max_batch_seconds"],
        "oom_splits": sum(r["throughput"]["oom_splits"] for r in results.values()),
        "projection": {"audio_hours": PROJECTION_HOURS,
                       "gpu_hours_estimate": PROJECTION_HOURS / rate if rate else None, "note": PROJECTION_NOTE},
    }
    lease = {k: v for k, v in os.environ.items() if k.startswith("AGENT_JOBS_")}
    status = "scored" if complete else ("smoke" if config["limit"] is not None else "partial")
    return {"status": status, "complete": complete, "teacher": teacher_info, "config": config, "sets": per_set,
            "throughput": throughput, "job_lease_env": lease,
            "gpu_lock": str(relabel_paths.GPU_LOCK) if config["device"] == "cuda" else None}


def write_not_scorable(out_dir: Path, teacher, config: dict, reason: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluate.write_json(out_dir / "summary.json", {
        "status": "not_scorable", "complete": False, "reason": reason, "config": config,
        "teacher": {"name": teacher.name, "hf_id": teacher.hf_id, "revision": teacher.revision}})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    import teachers

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", required=True, choices=list(teachers.REGISTRY))
    parser.add_argument("--sets", nargs="+", default=["dev"])
    parser.add_argument("--limit", type=int, help="smoke: first N id-sorted utterances per set (smoke path)")
    parser.add_argument("--shortest", type=int, default=0, help="smoke: also the K shortest utterances per set")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-batch-seconds", type=float)
    parser.add_argument("--device", default="cuda", help="cuda, or cpu for a smoke run without the GPU (no lock)")
    parser.add_argument("--lock-wait", type=float, default=1800.0, help="seconds to wait for the GPU lock")
    parser.add_argument("--force", action="store_true", help="re-score sets already written")
    parser.add_argument("--min-available-gb", type=float, default=16.0,
                        help="refuse to load the teacher if the host has less RAM available (exit 4)")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.shortest and args.limit is None:
        parser.error("--shortest is a smoke option and needs --limit")
    if args.device not in ("cuda", "cpu"):
        parser.error("--device is cuda or cpu")
    if args.device == "cpu" and args.limit is None:
        parser.error("--device cpu is only for --limit smoke runs")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    relabel_paths.require_mount()
    import teachers

    teacher = teachers.get(args.teacher)
    args.batch_size = args.batch_size or teacher.batch_size
    args.max_batch_seconds = args.max_batch_seconds or teacher.max_batch_seconds
    out_dir = relabel_paths.output_dir(teacher.name, args.limit)
    sets = [evaluate.resolve_manifest(n) for n in expand_sets(args.sets)]
    missing = [str(p) for _, p in sets if not p.exists()]
    if missing:
        raise SystemExit(f"missing manifests: {missing}")
    config = run_config(teacher, args)
    if args.device == "cpu":
        # CPU smoke (adapter and schema check when the GPU is busy): CUDA hidden, so no GPU lock is needed.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return _run_locked(teacher, args, out_dir.with_name(out_dir.name + "-cpu"), sets, config)
    try:
        with gpu_lock(f"relabel score_teacher {teacher.name}", args.lock_wait):
            return _run_locked(teacher, args, out_dir, sets, config)
    except LockBusy as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return LOCK_BUSY_EXIT


def _run_locked(teacher, args, out_dir: Path, sets, config: dict) -> int:
    import torch
    import teachers

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    fraction = os.environ.get("AGENT_JOBS_VRAM_FRACTION")
    if fraction and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(fraction), device)
    todo = []
    for name, manifest in sets:
        path = out_dir / f"{name}.json"
        if path.exists() and not args.force and json.loads(path.read_text()).get("config") == config:
            print(f"{name}: already written with this configuration, skipping", flush=True)
            continue
        todo.append((name, manifest))
    out_dir.mkdir(parents=True, exist_ok=True)
    load_s = None
    if todo and available_memory_gb() < args.min_available_gb:
        # No swap on this host: loading a teacher with too little free RAM risks a host-wide OOM.
        print(f"refusing to load: {available_memory_gb():.1f} GB host memory available, "
              f"--min-available-gb {args.min_available_gb}", file=sys.stderr)
        return LOW_MEMORY_EXIT
    if todo:
        # The summary is rewritten only after all sets of this run; until then it must not claim "scored".
        evaluate.write_json(out_dir / "summary.json", {"status": "in_progress", "complete": False, "config": config})
        try:
            teacher.load(str(device))
        except teachers.NotScorable as exc:
            write_not_scorable(out_dir, teacher, config, str(exc))
            print(f"{teacher.name}: NOT SCORABLE: {exc}", flush=True)
            return 0
        load_s = teacher.load_s
        print(f"{teacher.name}: loaded in {load_s:.1f} s", flush=True)
        source = teacher.info()
        source["model_load_s"] = load_s
        for name, manifest in todo:
            result = score_set(teacher, name, manifest, args, device, source)
            evaluate.write_json(out_dir / f"{name}.json", result)
            w, t = result["wer"], result["throughput"]
            wer = f"WER {100 * w['wer']:.2f}%" if w else "no scored utterances"
            peak = "N/A" if t["peak_gpu_memory_gb"] is None else f"{t['peak_gpu_memory_gb']:.1f} GB"
            print(f"{name}: n={result['utterances']} {wer} infer {t['inference_wall_s']:.1f} s "
                  f"({t['audio_hours_per_gpu_hour']:.0f} audio h/GPU h) peak {peak} "
                  f"capped {result['hit_token_cap']} padded {result['padded_short']} oom_splits {t['oom_splits']}",
                  flush=True)
        info = source
    else:
        info = json.loads((out_dir / f"{sets[0][0]}.json").read_text())["source"]
    summary = summarize(out_dir, info, config, load_s if load_s is not None else info.get("model_load_s"))
    evaluate.write_json(out_dir / "summary.json", summary)
    print(f"summary: status {summary['status']}, {summary['throughput']['audio_hours_per_gpu_hour']} audio h/GPU h; "
          f"{out_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
