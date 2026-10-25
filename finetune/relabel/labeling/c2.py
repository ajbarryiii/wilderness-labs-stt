"""C2, the labeling pass (labeling plan Part C "C2, labeling"): T* and B0 labels for every distinct id of C1.

Run directory $RELABEL_ROOT/labels/<teacher>/<run>/ (or a check's or smoke's directory):

    run.json                    the run's pinned settings: pool directory, group size, engine and its settings,
                                B0 batch seconds, declared VRAM; a later step refuses a different run.json
    work/teacher/<g>.parquet    T* raw outputs of shard group g (id, raw, hit_cap, and the confidence signals
                                token_ids, logprob, runner_up) and <g>.json (engine info, clips, seconds, wall time,
                                SHA-256), written atomically per group
    work/b0/<g>.parquet         B0 raw outputs (id, b0) and <g>.json
    shards/<k>.labels.parquet   one record per id first seen in C1 shard k (filters.py), and <k>.json
    groups/<g>.json             group done marker (all its label shards written)
    manifest.json               written by `finish` once every group is done and C1 is complete

Groups are contiguous runs of `group_shards` committed C1 shards, in order; a group's clips are sorted by
descending duration (ties by id) and labeled in that order (no global sort). Every step is resumable at group
granularity and skips groups already done; label shards are per C1 shard, so the pass resumes at shard
boundaries.

Confidence signals (labeling plan "Confidence signals", 2026-10-24): with "confidence": true in the engine spec (the
production settings), every id, kept or rejected, stores T*'s generated token ids (int32) and per token the chosen
token's and the runner-up's log-probability (float16), as t_token_ids, t_logprob, t_runner_up in its label record,
next to B0's raw and normalized label (b0_raw, b0_norm). They are recorded only: filters, dispositions and label text
do not read them. With confidence off the three columns are null. A run whose spec has the signals refuses engine
outputs without them, and one without refuses outputs with them.

Steps (each its own process, since vLLM lives in its own environment):
    teacher   T* over ready groups without teacher output (vLLM env for engine vllm; shared runtime otherwise)
    b0        B0 over groups with teacher output and without B0 output (shared runtime)
    merge     filters and label shards for groups with both (CPU)
    chunk     one chunk job: holds the parakeet GPU lock for its own duration only, runs teacher (until
              --budget-minutes x TEACHER_SHARE), then b0 and merge for the groups the teacher finished
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from labeling import layout, shards

TEACHER_SHARE = 0.75      # of a chunk's budget for the teacher step; b0 and merge follow
DECODE_THREADS = 6
LABEL_COLUMNS = ("id", "source", "duration", "reference_kind", "raw", "text", "norm", "b0_raw", "b0_norm",
                 "hit_token_cap", "disposition", "reasons", "human_wer", "b0_wer", "words_per_second", "teacher",
                 "teacher_revision", "engine", "t_token_ids", "t_logprob", "t_runner_up")
SIGNAL_KEYS = ("token_ids", "logprob", "runner_up")         # engine output -> work columns
SIGNAL_COLUMNS = ("t_token_ids", "t_logprob", "t_runner_up")  # label columns


def _schemas():
    import pyarrow as pa
    sig = [pa.list_(pa.int32()), pa.list_(pa.float16()), pa.list_(pa.float16())]
    teacher = pa.schema([("id", pa.string()), ("raw", pa.string()), ("hit_cap", pa.bool_())] +
                        list(zip(SIGNAL_KEYS, sig)))
    s, f, b = pa.string(), pa.float64(), pa.bool_()
    types = {"id": s, "source": s, "duration": f, "reference_kind": s, "raw": s, "text": s, "norm": s, "b0_raw": s,
             "b0_norm": s, "hit_token_cap": b, "disposition": s, "reasons": pa.list_(s), "human_wer": f, "b0_wer": f,
             "words_per_second": f, "teacher": s, "teacher_revision": s, "engine": s} | dict(zip(SIGNAL_COLUMNS, sig))
    return teacher, pa.schema([(c, types[c]) for c in LABEL_COLUMNS])


def teacher_schema():
    return _schemas()[0]


def label_schema():
    return _schemas()[1]


def signal_row(sig: dict | None) -> dict:
    """An engine's signals (or None) as the work file's columns."""
    return {k: (None if sig is None else sig[k]) for k in SIGNAL_KEYS}


def check_signals(sig, want: bool, ident: str) -> None:
    """Structural check of one engine output's signals against the run's setting (values are the engine's)."""
    if not want:
        if sig is not None:
            raise RuntimeError(f"{ident}: the engine returned confidence signals but the run's engine spec has them off")
        return
    if sig is None:
        raise RuntimeError(f"{ident}: no confidence signals (the run's engine spec has them on)")
    n = len(sig["token_ids"])
    if n == 0 or len(sig["logprob"]) != n or len(sig["runner_up"]) != n:
        raise RuntimeError(f"{ident}: misaligned or empty confidence signals")


def _log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [c2 pid {os.getpid()}] {msg}", flush=True)


def write_json(path: Path, obj) -> None:
    from pool import common
    common.atomic_write_json(path, obj)


# --- run settings ----------------------------------------------------------------------------------

def init_run(run_dir: Path, *, pool_dir: Path, engine: dict, group_shards: int, b0_batch_seconds: float,
             declared_vram_gb: float | None, extra: dict | None = None) -> dict:
    """Write run.json, or check that an existing one has the same settings."""
    import teachers
    from labeling import filters
    spec = teachers.REGISTRY[layout.TEACHER]
    if engine.get("kind") not in ("vllm", "transformers", "fake"):
        raise ValueError(f"unknown engine {engine!r}")
    if engine["kind"] == "vllm" and declared_vram_gb is not None:
        total = engine.get("total_vram_gb")
        if total and engine["gpu_memory_utilization"] * total > declared_vram_gb + 1e-6:
            raise ValueError("vLLM's memory fraction exceeds the declared VRAM")
    run = {"teacher": layout.TEACHER, "teacher_revision": spec.revision, "pool_dir": _rel(pool_dir),
           "group_shards": int(group_shards), "engine": engine, "b0": {"batch_seconds": float(b0_batch_seconds)},
           "declared_vram_gb": declared_vram_gb, "filters_version": filters.VERSION,
           **(extra or {})}
    path = Path(run_dir) / "run.json"
    if path.exists():
        old = json.loads(path.read_text())
        old.pop("created", None)
        if old != run:
            raise RuntimeError(f"{path} exists with different settings")
        return json.loads(path.read_text())
    write_json(path, run | {"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return run


def _rel(path: Path) -> str:
    path = Path(path).resolve()
    rr = layout.relabel_root().resolve()
    return str(path.relative_to(rr)) if path.is_relative_to(rr) else str(path)


def _abs(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else layout.relabel_root() / p


def load_run(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "run.json").read_text())


def contract(run: dict) -> dict:
    """Everything in a run's settings that can change a label or a disposition: the engine and its settings,
    the group size (batch composition), B0's batch seconds and the declared VRAM (the memory cap, hence
    out-of-memory splits). Checks qualify a production run only under an identical contract."""
    return {"engine": {k: v for k, v in run["engine"].items() if k != "total_vram_gb"},
            "group_shards": run["group_shards"], "b0_batch_seconds": run["b0"]["batch_seconds"],
            "declared_vram_gb": run["declared_vram_gb"], "teacher_revision": run["teacher_revision"],
            "filters_version": run["filters_version"]}


# --- groups ----------------------------------------------------------------------------------------

def pool_complete(pool_dir: Path) -> bool:
    st = Path(pool_dir) / "status.json"
    if st.exists() and json.loads(st.read_text()).get("complete"):
        return True
    # A materialized pool (checks) is complete when every shard is committed.
    return (Path(pool_dir) / "materialized.json").exists()


def groups(pool_dir: Path, group_shards: int) -> list[list[int]]:
    """Ready groups: contiguous runs of group_shards committed shards; the last, shorter group only once the
    pool is complete."""
    ks = shards.committed(pool_dir)
    if ks != list(range(len(ks))):
        raise RuntimeError("committed C1 shards are not a contiguous prefix")
    out = [ks[i:i + group_shards] for i in range(0, len(ks), group_shards)]
    if out and len(out[-1]) < group_shards and not pool_complete(pool_dir):
        out.pop()
    return out


def gname(g: int) -> str:
    return f"{g:05d}"


def load_group(pool_dir: Path, ks: list[int], with_audio: bool = True) -> list[dict]:
    cols = ("id", "source", "duration", "samples", "reference", "reference_kind") + (("flac",) if with_audio else ())
    rows = []
    for k in ks:
        rows += shards.read_audio(shards.paths(pool_dir, k)["audio"], columns=cols)
    rows.sort(key=lambda r: (-r["duration"], r["id"]))
    return rows


def _pcm(row: dict) -> np.ndarray:
    return shards.decode_flac(row["flac"]).astype(np.float32) / 32768.0


def label_rows(engine, rows: list[dict], threads: int = DECODE_THREADS) -> list:
    """Run engine over rows (already sorted) unit by unit as engine.plan() cuts them, decoding the next unit's
    audio while the current one runs (at most two units of decoded audio in memory)."""
    units = engine.plan([r["duration"] for r in rows])
    flat = sorted(i for u in units for i in u)
    if flat != list(range(len(rows))):
        raise RuntimeError("engine plan does not cover every clip exactly once")
    out: list = [None] * len(rows)
    with cf.ThreadPoolExecutor(max_workers=threads) as ex:
        pending = [ex.submit(_pcm, rows[i]) for i in units[0]] if units else []
        for u, idx in enumerate(units):
            clips = [f.result() for f in pending]
            pending = [ex.submit(_pcm, rows[i]) for i in units[u + 1]] if u + 1 < len(units) else []
            res = engine.run(clips)
            if len(res) != len(idx):
                raise RuntimeError(f"engine returned {len(res)} outputs for {len(idx)} clips")
            for i, r in zip(idx, res):
                out[i] = r
    return out


def _write_parquet(path: Path, rows: list[dict], schema=None) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp, compression="zstd")
    os.replace(tmp, path)
    return shards.sha256_file(path)


def _read_parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def _done(run_dir: Path, step: str, g: int) -> bool:
    return (Path(run_dir) / "work" / step / f"{gname(g)}.json").exists()


# --- engines ---------------------------------------------------------------------------------------

def wants_signals(spec: dict) -> bool:
    return bool(spec.get("confidence", False))


def make_engine(spec: dict, run: dict, timing: bool = False):
    from labeling import engines
    kind = spec["kind"]
    if kind == "vllm":
        return engines.VllmTeacher(spec["max_num_seqs"], spec["gpu_memory_utilization"],
                                   max_model_len=spec.get("max_model_len", 2048),
                                   submit_clips=spec.get("submit_clips", 2048), confidence=wants_signals(spec),
                                   batch_invariant=bool(spec.get("batch_invariant", False)),
                                   encoder_patch=spec.get("encoder_patch"))
    if kind == "transformers":
        return engines.TransformersTeacher(spec["batch_size"], spec["max_batch_seconds"],
                                           declared_vram_gb=run.get("declared_vram_gb"), timing=timing,
                                           confidence=wants_signals(spec))
    if kind == "fake":
        return FakeTeacher(spec)
    raise ValueError(kind)


class FakeTeacher:
    """Deterministic CPU stand-in for tests and smoke plumbing: a fixed text per clip length, and fixed signals when the
    spec has confidence on."""
    kind = "fake"

    def __init__(self, spec: dict):
        self.spec = spec
        self.confidence = wants_signals(spec)

    def load(self) -> None:
        pass

    def plan(self, durations):
        n, b = len(durations), int(self.spec.get("batch_size", 8))
        return [list(range(i, min(n, i + b))) for i in range(0, n, b)]

    def run(self, clips, max_tokens=None, confidence=None):
        conf = self.confidence if confidence is None else confidence
        out = []
        for c in clips:
            text = f"clip of {len(c)} samples"
            n = len(text.split()) + 1
            sig = {"token_ids": list(range(n)), "logprob": [-0.5] * n, "runner_up": [-2.0] * n}
            out.append((text, False, sig) if conf else (text, False))
        return out

    def info(self) -> dict:
        return {"engine": "fake", "confidence": wants_signals(self.spec)}


# --- steps -----------------------------------------------------------------------------------------

def teacher_step(run_dir: Path, deadline: float | None = None, stop_after_groups: int | None = None,
                 engine=None) -> dict:
    run_dir = Path(run_dir)
    run = load_run(run_dir)
    pool_dir = _abs(run["pool_dir"])
    todo = [(g, ks) for g, ks in enumerate(groups(pool_dir, run["group_shards"])) if not _done(run_dir, "teacher", g)]
    done = []
    if not todo:
        return {"groups": done}
    want = wants_signals(run["engine"])
    engine = engine or make_engine(run["engine"], run)
    t_load = time.perf_counter()
    engine.load()
    load_s = time.perf_counter() - t_load
    info = engine.info()
    for g, ks in todo:
        if deadline is not None and time.time() >= deadline:
            break
        if stop_after_groups is not None and len(done) >= stop_after_groups:
            break
        t0 = time.perf_counter()
        rows = load_group(pool_dir, ks)
        t1 = time.perf_counter()
        out = label_rows(engine, rows)
        t2 = time.perf_counter()
        recs = []
        for r, o in zip(rows, out):
            t, h, sig = (tuple(o) + (None,))[:3]
            check_signals(sig, want, r["id"])
            recs.append({"id": r["id"], "raw": t, "hit_cap": bool(h)} | signal_row(sig))
        path = run_dir / "work" / "teacher" / f"{gname(g)}.parquet"
        sha = _write_parquet(path, recs, teacher_schema())
        seconds = sum(r["duration"] for r in rows)
        write_json(path.with_suffix(".json"), {
            "group": g, "shards": ks, "clips": len(rows), "audio_seconds": seconds, "read_s": t1 - t0,
            "decode_and_engine_s": t2 - t1, "group_s": time.perf_counter() - t0, "load_s": load_s, "sha256": sha,
            "engine": info, "oom_splits": getattr(engine, "oom_splits", None), "signals": want,
            "signal_tokens": sum(len(x["token_ids"]) for x in recs if x["token_ids"] is not None)})
        done.append(g)
        _log(f"teacher group {g} (shards {ks[0]}-{ks[-1]}): {len(rows)} clips, {seconds / 3600:.2f} h, "
             f"engine {t2 - t1:.1f} s ({seconds / 3600 / max(t2 - t1, 1e-9) * 3600:.0f} audio h/h)")
    return {"groups": done, "load_s": load_s, "engine": info}


def b0_step(run_dir: Path, labeler=None, deadline: float | None = None) -> dict:
    run_dir = Path(run_dir)
    run = load_run(run_dir)
    pool_dir = _abs(run["pool_dir"])
    todo = [(g, ks) for g, ks in enumerate(groups(pool_dir, run["group_shards"]))
            if _done(run_dir, "teacher", g) and not _done(run_dir, "b0", g)]
    done = []
    if not todo:
        return {"groups": done}
    if labeler is None:
        from labeling.engines import B0Labeler
        labeler = B0Labeler(run["b0"]["batch_seconds"], declared_vram_gb=run.get("declared_vram_gb"))
    labeler.load()
    for g, ks in todo:
        if deadline is not None and time.time() >= deadline:
            break
        t0 = time.perf_counter()
        rows = load_group(pool_dir, ks)
        t1 = time.perf_counter()
        texts = label_rows(labeler, rows)
        t2 = time.perf_counter()
        path = run_dir / "work" / "b0" / f"{gname(g)}.parquet"
        sha = _write_parquet(path, [{"id": r["id"], "b0": t} for r, t in zip(rows, texts)])
        write_json(path.with_suffix(".json"), {"group": g, "shards": ks, "clips": len(rows),
                                               "audio_seconds": sum(r["duration"] for r in rows),
                                               "read_s": t1 - t0, "decode_and_engine_s": t2 - t1,
                                               "group_s": time.perf_counter() - t0, "sha256": sha,
                                               "engine": labeler.info()})
        done.append(g)
        _log(f"b0 group {g}: {len(rows)} clips, {t2 - t1:.1f} s")
    return {"groups": done}


def _signal_bytes(path: Path) -> int:
    """Compressed bytes of the signal columns in a label file (parquet column-chunk sizes)."""
    import pyarrow.parquet as pq
    md = pq.ParquetFile(path).metadata
    total = 0
    for g in range(md.num_row_groups):
        rg = md.row_group(g)
        for c in range(rg.num_columns):
            col = rg.column(c)
            if col.path_in_schema.split(".")[0] in SIGNAL_COLUMNS:
                total += col.total_compressed_size
    return total


def merge_step(run_dir: Path, coverage=None) -> dict:
    import evaluate  # finetune/parakeet-ternary
    from labeling import filters
    run_dir = Path(run_dir)
    run = load_run(run_dir)
    want = wants_signals(run["engine"])
    pool_dir = _abs(run["pool_dir"])
    done = []
    todo = [(g, ks) for g, ks in enumerate(groups(pool_dir, run["group_shards"]))
            if _done(run_dir, "teacher", g) and _done(run_dir, "b0", g) and not (run_dir / "groups" / f"{gname(g)}.json").exists()]
    if not todo:
        return {"groups": done}
    if coverage is None:
        coverage = filters.Coverage(filters.b0_tokenizer_file(layout.relabel_root() / "caches" / "labeling"))
    for g, ks in todo:
        t0 = time.perf_counter()
        tpath = run_dir / "work" / "teacher" / f"{gname(g)}.parquet"
        bpath = run_dir / "work" / "b0" / f"{gname(g)}.parquet"
        for path in (tpath, bpath):
            if shards.sha256_file(path) != json.loads(path.with_suffix(".json").read_text())["sha256"]:
                raise RuntimeError(f"{path}: SHA-256 differs from its meta")
        t = {r["id"]: r for r in _read_parquet(tpath)}
        b = {r["id"]: r["b0"] for r in _read_parquet(bpath)}
        engine = json.loads(tpath.with_suffix(".json").read_text())["engine"].get("engine")
        counts = {}
        for k in ks:
            rows = shards.read_audio(shards.paths(pool_dir, k)["audio"],
                                     columns=("id", "source", "duration", "reference", "reference_kind"))
            records = []
            for r in rows:
                if r["id"] not in t or r["id"] not in b:
                    raise RuntimeError(f"group {g}: no label output for {r['id']}")
                tr = t[r["id"]]
                sig = None if tr.get("token_ids") is None else {k: tr[k] for k in SIGNAL_KEYS}
                check_signals(sig, want, r["id"])
                res = filters.check(r["duration"], tr["raw"], r["reference"], r["reference_kind"], b[r["id"]],
                                    coverage)
                v = res["values"]
                records.append({"id": r["id"], "source": r["source"], "duration": r["duration"],
                                "reference_kind": r["reference_kind"], "raw": res["raw"], "text": res["text"],
                                "norm": res["norm"], "b0_raw": b[r["id"]],
                                "b0_norm": evaluate.normalize(b[r["id"]] or "").strip(),
                                "hit_token_cap": tr["hit_cap"],
                                "disposition": res["disposition"], "reasons": res["reasons"],
                                "human_wer": v.get("human_wer"), "b0_wer": v.get("b0_wer"),
                                "words_per_second": v.get("words_per_second"), "teacher": run["teacher"],
                                "teacher_revision": run["teacher_revision"], "engine": engine}
                               | dict(zip(SIGNAL_COLUMNS, (tr.get(k) for k in SIGNAL_KEYS))))
            path = run_dir / "shards" / f"{shards.shard_name(k)}.labels.parquet"
            sha = _write_parquet(path, records, label_schema())
            per: dict[str, dict] = {}
            for rec in records:
                c = per.setdefault(rec["source"], {"records": 0, "kept": 0, "reasons": {}, "hallucination": 0})
                c["records"] += 1
                c["kept"] += rec["disposition"] == "kept"
                for reason in rec["reasons"]:
                    c["reasons"][reason] = c["reasons"].get(reason, 0) + 1
                c["hallucination"] += any(x in filters.HALLUCINATION for x in rec["reasons"])
            write_json(path.with_name(f"{shards.shard_name(k)}.json"),
                       {"shard": k, "group": g, "records": len(records), "sha256": sha, "by_source": per,
                        "audio_seconds": sum(rec["duration"] for rec in records),
                        "signal_tokens": sum(len(rec["t_token_ids"] or ()) for rec in records),
                        "signal_bytes": _signal_bytes(path)})
            counts[k] = len(records)
        write_json(run_dir / "groups" / f"{gname(g)}.json", {"group": g, "shards": ks, "records": counts,
                                                              "merge_s": time.perf_counter() - t0})
        done.append(g)
    return {"groups": done}


def finish(run_dir: Path) -> dict:
    """manifest.json once C1 is complete and every group is done; else a status without it."""
    run_dir = Path(run_dir)
    run = load_run(run_dir)
    pool_dir = _abs(run["pool_dir"])
    gs = groups(pool_dir, run["group_shards"])
    missing = [g for g in range(len(gs)) if not (run_dir / "groups" / f"{gname(g)}.json").exists()]
    if missing or not pool_complete(pool_dir):
        return {"complete": False, "groups": len(gs), "groups_missing": len(missing), "pool_complete": pool_complete(pool_dir)}
    files, totals = [], {}
    sig = {"tokens": 0, "bytes": 0, "audio_seconds": 0.0}
    for ks in gs:
        for k in ks:
            meta = json.loads((run_dir / "shards" / f"{shards.shard_name(k)}.json").read_text())
            sig["tokens"] += meta.get("signal_tokens", 0)
            sig["bytes"] += meta.get("signal_bytes", 0)
            sig["audio_seconds"] += meta.get("audio_seconds", 0.0)
            path = run_dir / "shards" / f"{shards.shard_name(k)}.labels.parquet"
            if shards.sha256_file(path) != meta["sha256"]:
                raise RuntimeError(f"{path}: SHA-256 differs from its meta")
            files.append({"shard": k, "file": path.name, "sha256": meta["sha256"], "records": meta["records"]})
            for src, c in meta["by_source"].items():
                t = totals.setdefault(src, {"records": 0, "kept": 0, "reasons": {}, "hallucination": 0})
                t["records"] += c["records"]
                t["kept"] += c["kept"]
                t["hallucination"] += c["hallucination"]
                for r, n in c["reasons"].items():
                    t["reasons"][r] = t["reasons"].get(r, 0) + n
    hours = sig["audio_seconds"] / 3600
    sig |= {"on": wants_signals(run["engine"]), "bytes_per_audio_hour": sig["bytes"] / hours if hours else None,
            "tokens_per_audio_hour": sig["tokens"] / hours if hours else None,
            "note": "compressed parquet bytes of t_token_ids, t_logprob, t_runner_up in the label shards (the work "
                    "files hold a second copy)"}
    manifest = {"complete": True, "run": run, "shards": files, "records": sum(f["records"] for f in files),
                "by_source_distinct_ids": totals, "signals": sig}
    write_json(run_dir / "manifest.json", manifest)
    return {"complete": True, "records": manifest["records"]}


# --- chunk driver ----------------------------------------------------------------------------------

def _here() -> Path:
    return Path(__file__).resolve().parents[1]


def step_command(run: dict, step: str, run_dir: Path, deadline: float | None = None) -> list[str]:
    here = _here()
    if step == "teacher" and run["engine"]["kind"] == "vllm":
        cmd = [str(here / "labeling" / "vllm_env.sh"), "python", "-m", "labeling.c2", step]
    else:
        cmd = [str(here / "python"), "-m", "labeling.c2", step]
    cmd += ["--run-dir", str(run_dir)]
    if deadline is not None:
        cmd += ["--deadline", f"{deadline:.0f}"]
    return cmd


def chunk(run_dir: Path, budget_minutes: float, lock_wait: float, runner=subprocess.run) -> dict:
    """One chunk job: holds the parakeet GPU lock for its own duration only (score_teacher.gpu_lock)."""
    import score_teacher
    run_dir = Path(run_dir)
    run = load_run(run_dir)
    t0 = time.time()
    deadline = t0 + budget_minutes * 60 * TEACHER_SHARE
    with score_teacher.gpu_lock(f"relabel labeling c2 {run_dir.name}", lock_wait):
        for step, dl in (("teacher", deadline), ("b0", None)):
            res = runner(step_command(run, step, run_dir, dl), cwd=str(_here()))
            if res.returncode != 0:
                raise RuntimeError(f"{step} step exited {res.returncode}")
        merged = merge_step(run_dir)
    status = finish(run_dir)
    status |= {"merged_this_chunk": merged["groups"], "chunk_wall_s": time.time() - t0}
    write_json(run_dir / "status.json", status)
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="labeling.c2", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("teacher", "b0", "merge", "finish"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", required=True, type=Path)
        p.add_argument("--deadline", type=float, help="epoch seconds after which no new group starts")
        p.add_argument("--stop-after-groups", type=int)
    c = sub.add_parser("chunk")
    c.add_argument("--run-dir", required=True, type=Path)
    c.add_argument("--budget-minutes", type=float, default=layout.CHUNK_MINUTES)
    c.add_argument("--lock-wait", type=float, default=1800.0)
    args = parser.parse_args(argv)
    layout.require_mount()
    if args.cmd == "teacher":
        print(json.dumps(teacher_step(args.run_dir, args.deadline, args.stop_after_groups)["groups"]))
    elif args.cmd == "b0":
        print(json.dumps(b0_step(args.run_dir, deadline=args.deadline)["groups"]))
    elif args.cmd == "merge":
        print(json.dumps(merge_step(args.run_dir)["groups"]))
    elif args.cmd == "finish":
        print(json.dumps(finish(args.run_dir)))
    else:
        print(json.dumps(chunk(args.run_dir, args.budget_minutes, args.lock_wait)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
