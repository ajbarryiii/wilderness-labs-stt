"""The equivalence check, the consistency gate, the GPU probe and the gate to stage 3 (labeling plan Part C).

Equivalence check (vLLM only, preregistered):
  set: EQUIVALENCE_PER_SOURCE distinct ids per source drawn from C1 (seeded) plus every dev-medical and
  dev-technical utterance, materialized as shards (shards.materialize) in a check directory.
  runs: `b32`   stage 1 decoder at batch 32 / 600 s with stage 1's batching (one group = the whole set)
        `bmax`  stage 1 decoder at the largest batch with C2's bucketing (production group size)
        `vllm`  vLLM at the production settings, straight through
        `vllm-resumed`  the same, stopped after one group and resumed (a shard boundary)
        B0 labels in `b32` (for dispositions).
  passes iff the two vLLM runs are bitwise identical (label text, cap flag and, with the confidence signals on, token
  ids and stored log-probabilities; 2026-10-24, deterministic vLLM), and per C1 source: rate(norm vLLM != norm b32) <= 2 x
  rate(norm bmax != norm b32) + 1 point, and rate of changed dispositions (kept/rejected, filters with the same
  B0 label) <= 1%; and on dev-medical and dev-technical (unweighted mean of parts' corpus WER, as Part A)
  |WER(vLLM) - WER(b32)| <= 0.05 points.

Consistency gate (gate item 1): the labeling decoder at production settings, no filters, on stage 1's dev ids
(every stage 1 utterance, padded as stage 1 did), scored with evaluate.score / wer_summary, reproduces T*'s
stage 1 WER on every set within 0.05 points. Amended 2026-10-25 (gate_version 2): per set, clips whose stage 1 or
run transcript the hallucination guard rejects are left out of both WERs (stage 1 looped to the token cap on a clip
the engines transcribe correctly), and the run may lose at most max(2, ceil(0.5%)) more clips to the guard than
stage 1; the unfiltered WERs (stage 1's from diagnostic/<T*>/summary.json) are still reported.

GPU probe: per engine and setting, C2's teacher step for PROBE minutes on C1's first groups, then B0 on the
same groups; complete shard-to-label throughput (reads, FLAC decoding, T*, B0) in audio hours per GPU-hour,
peak VRAM (nvidia-smi, process tree), and T*'s time split (transformers: encoder, prefill and decode from
synchronized forward hooks; vLLM: encoder + prefill from a max_tokens=1 pass on the same clips, decode the rest), and
the confidence signals' cost: the same clips decoded once with and once without them, with the share of labels (text
or cap flag) that differ (labeling plan "GPU probe", 2026-10-24). vLLM's prefix and multimodal caches are reset before
each timed pass, so no pass is served from an earlier one.

Gate to stage 3: consistency (above); drops per source (all filters <= 5%, hallucination guard <= 2%, both by
distinct ids and by occurrences in the journal; a larger drop stops and is reported); completeness (every id
occurrence of the first TOTAL_BATCHES batches has exactly one record, no unexplained missing ids, no id
conflicts, and an independent replay of the first 1,000 batches gives the same journal); spot check files (100
seeded random records per source, with B0's labels and a confidence summary when the labels have the signals).
"""
from __future__ import annotations

import contextlib
import json
import math
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from labeling import c2, layout, shards

MAX_DROP, MAX_HALLUCINATION = 0.05, 0.02
WER_POINTS = 0.05
GATE_VERSION = 2          # consistency results: 2 = hallucination-guard exclusion (2026-10-25); 1 had no "gate_version"
GUARD_ALLOWANCE_MIN, GUARD_ALLOWANCE_SHARE = 2, 0.005
GUARD_CRITERION = ("per set: clips whose stage 1 or run transcript the hallucination guard (filter 3) rejects are "
                   "excluded; on the kept clips |corpus WER(run) - corpus WER(stage 1)| <= 0.05 points, and the guard "
                   "rejects the run's transcripts on at most stage 1's count + max(2, ceil(0.5% of the set's clips))")
DISPOSITION_MAX = 0.01
FLOOR_FACTOR, FLOOR_POINTS = 2.0, 0.01
VLLM_SETTINGS = ({"max_num_seqs": 64, "gpu_memory_utilization": 0.40},
                 {"max_num_seqs": 128, "gpu_memory_utilization": 0.50},
                 {"max_num_seqs": 256, "gpu_memory_utilization": 0.70})
TRANSFORMERS_SETTINGS = ({"batch_size": 32, "max_batch_seconds": 600.0},
                         {"batch_size": 64, "max_batch_seconds": 1200.0},
                         {"batch_size": 128, "max_batch_seconds": 2400.0})
STAGE1_BATCH = {"batch_size": 32, "max_batch_seconds": 600.0}
GROUP_SHARDS = 4
B0_BATCH_SECONDS = 1200.0
SPLIT_CLIPS = 512
IO_START_MAX = 10.0       # the probe starts only while the data disk's IO pressure ("some", 60 s average, %) is <= this
IO_WAIT_S, IO_POLL_S = 1800.0, 15.0   # how long it waits for that, and how often it looks
CHECK_SHARD_CLIPS = 3000


def write_json(path: Path, obj) -> None:
    c2.write_json(path, obj)


def _normalize(text: str) -> str:
    import evaluate
    return evaluate.normalize(text or "").strip()


def _teacher_outputs(run_dir: Path) -> dict[str, tuple[str, bool]]:
    out = {}
    for p in sorted((Path(run_dir) / "work" / "teacher").glob("*.parquet")):
        for r in c2._read_parquet(p):
            out[r["id"]] = (r["raw"], r["hit_cap"])
    return out


def teacher_signals(run_dir: Path) -> dict[str, dict | None]:
    """id -> the stored signals ({token_ids, logprob, runner_up}) or None; refuses a run whose engine spec has the
    signals on but lacks them for any clip."""
    want = c2.wants_signals(c2.load_run(run_dir)["engine"])
    out = {}
    for p in sorted((Path(run_dir) / "work" / "teacher").glob("*.parquet")):
        for r in c2._read_parquet(p):
            sig = None if r.get("token_ids") is None else {k: r[k] for k in c2.SIGNAL_KEYS}
            if want and sig is None:
                raise RuntimeError(f"{run_dir}: no confidence signals for {r['id']}")
            out[r["id"]] = sig
    return out


def compare_signals(a: dict[str, dict | None], b: dict[str, dict | None]) -> dict:
    """Two runs' signals on the same ids: token ids identical, stored log-probabilities bitwise identical (and the
    largest difference, for the report)."""
    import numpy as np
    ids_same, same, worst = True, True, 0.0
    for i in sorted(set(a) | set(b)):
        x, y = a.get(i), b.get(i)
        if x is None or y is None:
            ids_same &= x is None and y is None
            continue
        if x["token_ids"] != y["token_ids"]:
            ids_same = False
            continue
        for k in ("logprob", "runner_up"):
            u, v = np.asarray(x[k], np.float64), np.asarray(y[k], np.float64)
            if u.size:
                worst = max(worst, float(np.max(np.abs(u - v))))
            same &= x[k] == y[k]
    return {"token_ids_identical": ids_same, "logprobs_identical": same, "max_abs_diff": worst}


def _b0_outputs(run_dir: Path) -> dict[str, str]:
    out = {}
    for p in sorted((Path(run_dir) / "work" / "b0").glob("*.parquet")):
        for r in c2._read_parquet(p):
            out[r["id"]] = r["b0"]
    return out


def _pool_rows(pool_dir: Path) -> dict[str, dict]:
    rows = {}
    for k in shards.committed(pool_dir):
        for r in shards.read_audio(shards.paths(pool_dir, k)["audio"],
                                   columns=("id", "source", "duration", "reference", "reference_kind")):
            rows[r["id"]] = r
    return rows


# --- check sets ----------------------------------------------------------------------------------------

def sample_c1(pool_dir: Path, per_source: int, seed: int) -> dict[str, list[str]]:
    """Seeded sample of distinct ids per source from C1's committed shards."""
    by_source: dict[str, list[str]] = defaultdict(list)
    for k in shards.committed(pool_dir):
        for r in shards.iter_audio(shards.paths(pool_dir, k)["audio"], columns=("id", "source"), batch_rows=8192):
            by_source[r["source"]].append(r["id"])
    return {s: sorted(random.Random(f"{seed}:{s}").sample(sorted(ids), min(per_source, len(ids))))
            for s, ids in sorted(by_source.items())}


def copy_c1_rows(pool_dir: Path, ids: set[str]) -> list[dict]:
    """Full audio rows (with FLAC) of the given ids, reading only the row groups that hold them."""
    import pyarrow.parquet as pq
    out = []
    for k in shards.committed(pool_dir):
        path = shards.paths(pool_dir, k)["audio"]
        pf = pq.ParquetFile(path)
        for g in range(pf.metadata.num_row_groups):
            ids_g = pf.read_row_group(g, columns=["id"]).column("id").to_pylist()
            if ids & set(ids_g):
                out += [r for r in pf.read_row_group(g).to_pylist() if r["id"] in ids]
    return out


def write_rows_as_shards(rows: list[dict], directory: Path, shard_clips: int) -> int:
    """Audio rows (C1 schema) into materialized shards of shard_clips rows, in the given order."""
    directory = Path(directory)
    n = 0
    for k, start in enumerate(range(0, len(rows), shard_clips)):
        p = shards.paths(directory, k)
        if p["meta"].exists():
            n += 1
            continue
        w = shards.AudioWriter(p["audio"])
        for r in rows[start:start + shard_clips]:
            w.add({c: r[c] for c in shards.AUDIO_COLUMNS})
        sha, size = w.commit()
        shards.write_meta(p["meta"], {"shard": k, "kind": "materialized", "new_ids": w.count,
                                      "audio_seconds_new": w.seconds,
                                      "files": {"audio": {"name": p["audio"].name, "sha256": sha, "bytes": size}}})
        n += 1
    return n


def dev_part_records(set_names=("dev-medical", "dev-technical")) -> dict[str, list[dict]]:
    import devsets
    base = layout.relabel_root() / "devsets" / "manifests"
    out = {}
    for name in set_names:
        for part in devsets.SETS[name]:
            path = base / f"{part}.jsonl"
            if not path.exists():
                raise RuntimeError(f"domain dev part {part} has no manifest ({path})")
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            out[part] = sorted(rows, key=lambda r: r["id"])
    return out


def prepare_equivalence(pool_dir: Path, out_dir: Path, per_source: int = layout.EQUIVALENCE_PER_SOURCE,
                        seed: int = layout.EQUIVALENCE_SEED, shard_clips: int = CHECK_SHARD_CLIPS,
                        allow_partial: bool = False) -> dict:
    pool_dir, out_dir = Path(pool_dir), Path(out_dir)
    if not c2.pool_complete(pool_dir) and not allow_partial:
        raise RuntimeError("C1 is not complete (the equivalence set is drawn from its journal)")
    set_path = out_dir / "set.json"
    if set_path.exists():
        return json.loads(set_path.read_text())
    picked = sample_c1(pool_dir, per_source, seed)
    wanted = {i for ids in picked.values() for i in ids}
    rows = copy_c1_rows(pool_dir, wanted)
    rows.sort(key=lambda r: (r["source"], r["id"]))
    if len(rows) != len(wanted):
        raise RuntimeError(f"found {len(rows)} of {len(wanted)} sampled clips")
    parts = dev_part_records()
    tmp = out_dir / "materialize-dev"
    dev_rows = []
    for part, recs in parts.items():
        ks = shards.materialize(recs, tmp / part, shard_size=len(recs), source_of=lambda r, p=part: p)
        for k in ks:
            dev_rows += shards.read_audio(shards.paths(tmp / part, k)["audio"])
    n = write_rows_as_shards(rows + dev_rows, out_dir / "pool", shard_clips)
    write_json(out_dir / "pool" / "materialized.json", {"shards": n, "clips": len(rows) + len(dev_rows)})
    import shutil
    shutil.rmtree(tmp)
    info = {"c1_pool": c2._rel(pool_dir), "per_source": per_source, "seed": seed, "c1_ids": picked,
            "dev_parts": {p: len(r) for p, r in parts.items()}, "shards": n, "shard_clips": shard_clips}
    write_json(set_path, info)
    return info


def prepare_consistency(out_dir: Path, shard_clips: int = CHECK_SHARD_CLIPS, limit: int | None = None) -> dict:
    """The stage 1 dev sets (every utterance, or the first `limit` id-sorted ones per set for smokes)."""
    import evaluate
    import relabel_paths
    out_dir = Path(out_dir)
    info_path = out_dir / "set.json"
    if info_path.exists():
        return json.loads(info_path.read_text())
    rows, sets = [], {}
    for name in relabel_paths.SETS:
        _, manifest = evaluate.resolve_manifest(name)
        recs = evaluate.read_manifest(manifest)[:limit]
        sets[name] = {"manifest_sha256": evaluate.sha256_file(manifest), "utterances": len(recs)}
        tmp = out_dir / "materialize" / name
        for k in shards.materialize(recs, tmp, shard_size=len(recs), source_of=lambda r, n=name: n):
            rows += shards.read_audio(shards.paths(tmp, k)["audio"])
    n = write_rows_as_shards(rows, out_dir / "pool", shard_clips)
    write_json(out_dir / "pool" / "materialized.json", {"shards": n, "clips": len(rows)})
    import shutil
    shutil.rmtree(out_dir / "materialize")
    info = {"sets": sets, "limit": limit, "shards": n}
    write_json(info_path, info)
    return info


# --- evaluation of the checks ------------------------------------------------------------------------------

def corpus_wer(pairs: list[tuple[str, str]]) -> float | None:
    import evaluate
    recs = [evaluate.score(ref, hyp) for ref, hyp in pairs]
    scored = [r for r in recs if r["scored"]]
    return 100 * evaluate.wer_summary(scored)["wer"] if scored else None


def evaluate_equivalence(check_dir: Path, coverage=None) -> dict:
    from labeling import filters
    check_dir = Path(check_dir)
    info = json.loads((check_dir / "set.json").read_text())
    pool = _pool_rows(check_dir / "pool")
    lab = {name: _teacher_outputs(check_dir / name) for name in ("b32", "bmax", "vllm", "vllm-resumed")}
    b0 = _b0_outputs(check_dir / "b32")
    for name, d in lab.items():
        if set(d) != set(pool):
            raise RuntimeError(f"run {name} is incomplete ({len(d)} of {len(pool)} clips)")
    if set(b0) != set(pool):
        raise RuntimeError("B0 labels are incomplete")
    if coverage is None:
        coverage = filters.Coverage(filters.b0_tokenizer_file(layout.relabel_root() / "caches" / "labeling"))
    norm = {name: {i: _normalize(t) for i, (t, _) in d.items()} for name, d in lab.items()}

    def disp(i: str, label: str) -> str:
        r = pool[i]
        return filters.check(r["duration"], label, r["reference"], r["reference_kind"], b0[i], coverage)["disposition"]

    sig = compare_signals(teacher_signals(check_dir / "vllm"), teacher_signals(check_dir / "vllm-resumed"))
    identical = lab["vllm"] == lab["vllm-resumed"] and sig["token_ids_identical"]
    sources = {}
    for src, ids in info["c1_ids"].items():
        n = len(ids)
        floor = sum(norm["bmax"][i] != norm["b32"][i] for i in ids) / n
        diff = sum(norm["vllm"][i] != norm["b32"][i] for i in ids) / n
        flips = sum(disp(i, lab["vllm"][i][0]) != disp(i, lab["b32"][i][0]) for i in ids) / n
        limit = FLOOR_FACTOR * floor + FLOOR_POINTS
        sources[src] = {"n": n, "floor": floor, "vllm_vs_b32": diff, "limit": limit, "label_ok": diff <= limit,
                        "disposition_changes": flips, "disposition_ok": flips <= DISPOSITION_MAX}
    import devsets
    dev = {}
    for set_name in ("dev-medical", "dev-technical"):
        parts = {}
        for part in devsets.SETS[set_name]:
            ids = [i for i, r in pool.items() if r["source"] == part]
            parts[part] = {run: corpus_wer([(pool[i]["reference"] or "", lab[run][i][0]) for i in ids])
                           for run in ("b32", "bmax", "vllm")}
        mean = {run: sum(p[run] for p in parts.values()) / len(parts) for run in ("b32", "bmax", "vllm")}
        dev[set_name] = {"parts": parts, "mean_wer": mean, "vllm_minus_b32": mean["vllm"] - mean["b32"],
                         "ok": abs(mean["vllm"] - mean["b32"]) <= WER_POINTS}
    contracts = {name: c2.contract(c2.load_run(check_dir / name)) for name in ("vllm", "vllm-resumed")}
    same = contracts["vllm"] == contracts["vllm-resumed"]
    passed = same and identical and sig["logprobs_identical"] and \
        all(s["label_ok"] and s["disposition_ok"] for s in sources.values()) and all(d["ok"] for d in dev.values())
    engines = {name: {"engine_spec": c2.load_run(check_dir / name)["engine"],
                      "engine_info": json.loads(next((check_dir / name / "work" / "teacher").glob("*.json")).read_text())["engine"]}
               for name in lab}
    result = {"passed": passed, "vllm_runs_identical": identical, "vllm_logprobs_identical": sig["logprobs_identical"],
              "vllm_logprob_max_abs_diff": sig["max_abs_diff"], "contract": contracts["vllm"],
              "sources": sources, "dev": dev,
              "criteria": {"label": "rate <= 2 x floor + 1 point", "disposition": "<= 1%", "wer": "within 0.05 points",
                           "resumed": "bitwise identical text, cap flags, token ids and stored log-probabilities"},
              "engines": engines, "set": {k: v for k, v in info.items() if k != "c1_ids"}}
    write_json(check_dir / "result.json", result)
    return result


def guard_allowance(n_clips: int) -> int:
    """How many more clips of a set the hallucination guard may reject in the run's transcripts than in stage 1's."""
    return max(GUARD_ALLOWANCE_MIN, math.ceil(round(n_clips * GUARD_ALLOWANCE_SHARE, 9)))


def guard_rejects(duration: float, label: str, reference: str) -> bool:
    """Whether the hallucination guard (filter 3) rejects a transcript, applied as C2 applies it to a label with a human
    reference (so the B0 rule does not apply; tokenizer coverage is filter 4, not the guard)."""
    from labeling import filters
    reasons = filters.check(duration, label, reference, "human", None, lambda _: True)["reasons"]
    return any(r in filters.HALLUCINATION for r in reasons)


def evaluate_consistency(check_dir: Path, run_name: str = "run", stage1_dir: Path | None = None) -> dict:
    """Gate item 1 (gate_version 2, labeling plan amendment 2026-10-25): see GUARD_CRITERION. The unfiltered WERs
    (`*_unfiltered`; stage 1's from its summary.json on the full population) are reported next to the compared ones."""
    import relabel_paths
    check_dir = Path(check_dir)
    info = json.loads((check_dir / "set.json").read_text())
    stage1_dir = Path(stage1_dir or relabel_paths.DIAGNOSTIC / layout.TEACHER)
    summary = json.loads((stage1_dir / "summary.json").read_text())
    pool = _pool_rows(check_dir / "pool")
    lab = _teacher_outputs(check_dir / run_name)
    if set(lab) != set(pool):
        raise RuntimeError(f"incomplete decode ({len(lab)} of {len(pool)} clips)")
    sets = {}
    for name, meta in info["sets"].items():
        stage1 = json.loads((stage1_dir / f"{name}.json").read_text())
        ids = sorted(i for i, r in pool.items() if r["source"] == name)
        limit = info.get("limit")
        expect = sorted(r["id"] for r in stage1["records"])[:limit] if limit else sorted(r["id"] for r in stage1["records"])
        if ids != expect:
            raise RuntimeError(f"{name}: the decoded population is not stage 1's")
        if stage1["manifest_sha256"] != meta["manifest_sha256"]:
            raise RuntimeError(f"{name}: manifest differs from stage 1's")
        refs = {r["id"]: r["ref"] for r in stage1["records"]}
        hyps = {r["id"]: r["hyp"] for r in stage1["records"]}
        wer_all = corpus_wer([(refs[i], lab[i][0]) for i in ids])
        ref_all = summary["sets"][name]["wer"] if not limit else corpus_wer([(refs[i], hyps[i]) for i in ids])
        bad_run = {i for i in ids if guard_rejects(pool[i]["duration"], lab[i][0], refs[i])}
        bad_s1 = {i for i in ids if guard_rejects(pool[i]["duration"], hyps[i], refs[i])}
        kept = [i for i in ids if i not in bad_run | bad_s1]
        wer = corpus_wer([(refs[i], lab[i][0]) for i in kept])
        ref_wer = corpus_wer([(refs[i], hyps[i]) for i in kept])
        # no scored clip left (a small --limit smoke): nothing compared, so the set fails
        diff = None if wer is None or ref_wer is None else wer - ref_wer
        allowance = guard_allowance(len(ids))
        sets[name] = {"wer": wer, "stage1_wer": ref_wer, "diff": diff, "wer_unfiltered": wer_all,
                      "stage1_wer_unfiltered": ref_all,
                      "diff_unfiltered": None if wer_all is None or ref_all is None else wer_all - ref_all,
                      "clips": len(ids), "excluded": len(ids) - len(kept), "guard_rejected_run": len(bad_run),
                      "guard_rejected_stage1": len(bad_s1), "guard_allowance": allowance,
                      "ok": diff is not None and abs(diff) <= WER_POINTS and len(bad_run) <= len(bad_s1) + allowance}
    engine = json.loads(next((check_dir / run_name / "work" / "teacher").glob("*.json")).read_text())["engine"]
    run = c2.load_run(check_dir / run_name)
    full = info.get("limit") is None and sorted(sets) == sorted(relabel_paths.SETS)
    result = {"passed": all(s["ok"] for s in sets.values()), "gate_version": GATE_VERSION,
              "criterion": GUARD_CRITERION, "full_population": full, "sets": sets,
              "points": WER_POINTS, "engine": engine, "run": run, "contract": c2.contract(run),
              "limit": info.get("limit")}
    write_json(check_dir / f"result-{run_name}.json", result)
    return result


# --- driving the checks (one job each; GPU lock held for the job only) ------------------------------------

def _run_steps(steps: list[tuple[str, Path, dict]], lock_wait: float, label: str, runner=subprocess.run) -> None:
    import score_teacher
    with score_teacher.gpu_lock(label, lock_wait):
        for step, run_dir, kw in steps:
            run = c2.load_run(run_dir)
            cmd = c2.step_command(run, step, run_dir)
            if kw.get("stop_after_groups") is not None:
                cmd += ["--stop-after-groups", str(kw["stop_after_groups"])]
            res = runner(cmd, cwd=str(c2._here()))
            if res.returncode != 0:
                raise RuntimeError(f"{step} {run_dir.name} exited {res.returncode}")


def vllm_engine(setting: dict, total_vram_gb: float | None = None) -> dict:
    """A production vLLM engine spec (confidence signals on, batch-invariant mode; labeling plan 2026-10-24) with the
    per-clip Granite encoder (encoder_patch: the patch identity, so part of c2.contract; 2026-10-25)."""
    from labeling import vllm_granite_patch
    return {"kind": "vllm", "max_num_seqs": int(setting["max_num_seqs"]),
            "gpu_memory_utilization": float(setting["gpu_memory_utilization"]), "max_model_len": 2048,
            "submit_clips": 2048, "confidence": True, "batch_invariant": True,
            "encoder_patch": vllm_granite_patch.identity(), "total_vram_gb": total_vram_gb}


def tf_engine(setting: dict) -> dict:
    """A production transformers engine spec (confidence signals on)."""
    return {"kind": "transformers", "batch_size": int(setting["batch_size"]),
            "max_batch_seconds": float(setting["max_batch_seconds"]), "confidence": True}


def run_equivalence(check_dir: Path, vllm_setting: dict, tf_max: dict, declared_vram_gb: float, total_vram_gb: float,
                    group_shards: int = GROUP_SHARDS, lock_wait: float = 1800.0, runner=subprocess.run) -> dict:
    check_dir = Path(check_dir)
    pool = check_dir / "pool"
    n_groups = len(c2.groups(pool, group_shards))
    if n_groups < 2:
        raise RuntimeError("the equivalence set needs at least two groups to test an interruption")
    n_all = len(shards.committed(pool))
    runs = {"b32": (tf_engine(STAGE1_BATCH), n_all), "bmax": (tf_engine(tf_max), group_shards),
            "vllm": (vllm_engine(vllm_setting, total_vram_gb), group_shards),
            "vllm-resumed": (vllm_engine(vllm_setting, total_vram_gb), group_shards)}
    for name, (engine, g) in runs.items():
        c2.init_run(check_dir / name, pool_dir=pool, engine=engine, group_shards=g, b0_batch_seconds=B0_BATCH_SECONDS,
                    declared_vram_gb=declared_vram_gb)
    steps = [("teacher", check_dir / "b32", {}), ("b0", check_dir / "b32", {}), ("teacher", check_dir / "bmax", {}),
             ("teacher", check_dir / "vllm", {}),
             ("teacher", check_dir / "vllm-resumed", {"stop_after_groups": 1}),
             ("teacher", check_dir / "vllm-resumed", {})]
    _run_steps(steps, lock_wait, f"relabel labeling equivalence {check_dir.name}", runner)
    return evaluate_equivalence(check_dir)


def run_consistency(check_dir: Path, engine: dict, declared_vram_gb: float, group_shards: int = GROUP_SHARDS,
                    lock_wait: float = 1800.0, runner=subprocess.run, run_name: str = "run") -> dict:
    check_dir = Path(check_dir)
    c2.init_run(check_dir / run_name, pool_dir=check_dir / "pool", engine=engine, group_shards=group_shards,
                b0_batch_seconds=B0_BATCH_SECONDS, declared_vram_gb=declared_vram_gb)
    _run_steps([("teacher", check_dir / run_name, {})], lock_wait, f"relabel labeling consistency {check_dir.name}",
               runner)
    return evaluate_consistency(check_dir, run_name)


def io_pressure() -> dict | None:
    """The host's IO pressure (/proc/pressure/io): cumulative stall microseconds ("some", "full") and the 10 s average."""
    try:
        lines = Path("/proc/pressure/io").read_text().splitlines()
    except OSError:
        return None
    out = {}
    for line in lines:
        kind, *fields = line.split()
        f = dict(x.split("=") for x in fields)
        out[kind] = {"total_us": int(f["total"]), "avg10": float(f["avg10"]), "avg60": float(f["avg60"])}
    return out


def wait_for_low_io(max_pct: float | None = None, wait_s: float | None = None) -> tuple[dict | None, float]:
    """Wait until the IO pressure's 60 s average ("some") is at most max_pct; (that reading, seconds waited). Raises
    after wait_s (C1 shares the disk; a probe under contention would misstate throughput). Without pressure
    information (no /proc/pressure/io) it does not wait."""
    max_pct = IO_START_MAX if max_pct is None else max_pct
    wait_s = IO_WAIT_S if wait_s is None else wait_s
    t0 = time.monotonic()
    while True:
        p = io_pressure()
        if p is None or p["some"]["avg60"] <= max_pct:
            return p, time.monotonic() - t0
        if time.monotonic() - t0 >= wait_s:
            raise RuntimeError(f"IO pressure stayed above {max_pct}% (60 s average {p['some']['avg60']}%) for "
                               f"{wait_s:.0f} s; not probing under disk contention")
        time.sleep(IO_POLL_S)


@contextlib.contextmanager
def locked_when_quiet(label: str, lock_wait: float):
    """Hold the parakeet GPU lock with the IO pressure low: wait for low pressure, take the lock, and check again under
    it; if pressure rose meanwhile, release the lock and wait again (IO_WAIT_S overall). Yields (the reading taken
    under the lock, seconds waited)."""
    import score_teacher
    t0 = time.monotonic()
    while True:
        wait_for_low_io(wait_s=max(0.0, IO_WAIT_S - (time.monotonic() - t0)))
        with score_teacher.gpu_lock(label, lock_wait):
            p = io_pressure()
            if p is None or p["some"]["avg60"] <= IO_START_MAX:
                yield p, time.monotonic() - t0
                return
        if time.monotonic() - t0 >= IO_WAIT_S:
            raise RuntimeError(f"IO pressure rose above {IO_START_MAX}% each time the GPU lock was taken; not probing")
        time.sleep(IO_POLL_S)


def io_share(before: dict | None, after: dict | None, wall_s: float) -> dict | None:
    """Mean share of wall time with some / all tasks stalled on IO between two io_pressure() readings."""
    if not before or not after or wall_s <= 0:
        return None
    return {k: (after[k]["total_us"] - before[k]["total_us"]) / (wall_s * 1e6) for k in before if k in after}


def run_gpu_probe(pool_dir: Path, out_dir: Path, engine: dict, minutes: float, declared_vram_gb: float,
                  group_shards: int = GROUP_SHARDS, lock_wait: float = 1800.0, runner=subprocess.run,
                  merge=None) -> dict:
    """GPU probe of one engine setting (labeling plan "GPU probe"); see the module docstring. It runs what a chunk
    runs (teacher for `minutes`, then B0 and merge for the same groups, each step its own process with its model
    load) under the GPU lock, and reports: the probe's own rate (all wall time, start-up included); the steady
    rate (per-group reads, decoding, T*, B0, writes and merge); and the production estimate, which amortizes the
    measured per-chunk start-up over a CHUNK_MINUTES chunk.

    Every measurement starts from an empty directory, so the group times and the wall times come from the same
    invocation: a completed probe (probe.json present) is returned as it is, and the remains of an interrupted
    one are deleted first."""
    import shutil
    import score_teacher
    from labeling.monitor import Monitor
    out_dir = Path(out_dir)
    if (out_dir / "probe.json").exists():
        return json.loads((out_dir / "probe.json").read_text())
    if out_dir.exists():
        shutil.rmtree(out_dir)
    c2.init_run(out_dir, pool_dir=pool_dir, engine=engine, group_shards=group_shards, b0_batch_seconds=B0_BATCH_SECONDS,
                declared_vram_gb=declared_vram_gb)
    run = c2.load_run(out_dir)
    result = {"engine": engine, "minutes": minutes, "group_shards": group_shards, "contract": c2.contract(run)}
    with locked_when_quiet(f"relabel labeling gpu probe {out_dir.name}", lock_wait) as (start_io, waited):
        t_all = time.perf_counter()
        deadline = time.time() + minutes * 60
        result["io_pressure"] = {"start": start_io, "waited_s": waited, "start_max_pct": IO_START_MAX}
        for step, dl in (("teacher", deadline), ("b0", None)):
            io0 = io_pressure()
            with Monitor(interval=0.5, gpu=True) as mon:
                res = runner(c2.step_command(run, step, out_dir, dl), cwd=str(c2._here()))
            if res.returncode != 0:
                raise RuntimeError(f"probe {step} exited {res.returncode}")
            result[f"{step}_monitor"] = mon.result()
            result["io_pressure"][step] = io_share(io0, io_pressure(), result[f"{step}_monitor"]["wall_s"])
        t_merge = time.perf_counter()
        (merge or c2.merge_step)(out_dir)
        result["merge_wall_s"] = time.perf_counter() - t_merge
        result["probe_wall_s"] = time.perf_counter() - t_all
        cmd = c2.step_command(run, "teacher", out_dir)
        cmd[cmd.index("labeling.c2"):cmd.index("labeling.c2") + 2] = ["labeling.checks", "split"]
        io0, t_split = io_pressure(), time.perf_counter()
        res = runner(cmd, cwd=str(c2._here()))
        if res.returncode != 0:
            raise RuntimeError(f"probe split exited {res.returncode}")
        result["io_pressure"]["split"] = io_share(io0, io_pressure(), time.perf_counter() - t_split)
        result["io_pressure"]["end"] = io_pressure()
    metas = {s: [json.loads(p.read_text()) for p in sorted((out_dir / "work" / s).glob("*.json"))]
             for s in ("teacher", "b0")}
    groups = [m["group"] for m in metas["teacher"]]
    b0 = {m["group"]: m for m in metas["b0"] if m["group"] in groups}
    teach = {m["group"]: m for m in metas["teacher"]}
    merged = {g: json.loads((out_dir / "groups" / f"{c2.gname(g)}.json").read_text()) for g in groups}
    group_s = {g: teach[g]["group_s"] + b0[g]["group_s"] + merged[g]["merge_s"] for g in groups}
    group_h = {g: teach[g]["audio_seconds"] / 3600 for g in groups}
    hours = sum(group_h.values())
    t_s = sum(m["group_s"] for m in teach.values())
    b_s = sum(m["group_s"] for m in b0.values())
    m_s = sum(m["merge_s"] for m in merged.values())
    # Steady state (labeling plan 2026-10-24): each step's process compiles and warms up in its first group, so the
    # steady rate comes from the later groups; the first group's excess over it is start-up, with the model loads.
    steady_groups = groups[1:] if len(groups) > 1 else groups
    steady_h = sum(group_h[g] for g in steady_groups)
    steady_s = sum(group_s[g] for g in steady_groups)
    s_per_h = steady_s / steady_h if steady_h else None
    excess = (group_s[groups[0]] - group_h[groups[0]] * s_per_h) if groups and s_per_h and len(groups) > 1 else 0.0
    walls = result["teacher_monitor"]["wall_s"] + result["b0_monitor"]["wall_s"] + result["merge_wall_s"]
    startup_s = walls - hours * s_per_h if s_per_h else walls
    chunk_s = layout.CHUNK_MINUTES * 60
    per_chunk_h = max(0.0, chunk_s - startup_s) / s_per_h if s_per_h else 0.0
    peak = max(result["teacher_monitor"]["peak_gpu_gb"] or 0, result["b0_monitor"]["peak_gpu_gb"] or 0)
    result |= {"groups": groups, "steady_groups": steady_groups, "audio_hours": hours, "teacher_s": t_s, "b0_s": b_s,
               "merge_s": m_s, "group_s": {str(g): group_s[g] for g in groups},
               "first_group_excess_s": excess, "startup_s_per_chunk": startup_s,
               "probe_audio_hours_per_gpu_hour": hours / (result["probe_wall_s"] / 3600),
               "steady_audio_hours_per_gpu_hour": 3600 / s_per_h if s_per_h else None,
               "audio_hours_per_gpu_hour": per_chunk_h / (chunk_s / 3600),
               "rate_note": f"production estimate: steady rate (groups after the first) with the measured start-up "
                            f"(model loads, the first group's compilation and warm-up excess, per-step overheads) "
                            f"amortized over {layout.CHUNK_MINUTES:.0f}-minute chunks"
                            + ("" if len(groups) > 1 else "; only one group: steady rate includes its warm-up"),
               "peak_vram_gb": peak, "declared_vram_gb_for_c2": peak + layout.VRAM_MARGIN_GB,
               "split": json.loads((out_dir / "split.json").read_text()),
               "teacher_model_load_s": metas["teacher"][0]["load_s"] if metas["teacher"] else None}
    write_json(out_dir / "probe.json", result)
    return result


def split_step(run_dir: Path, n_clips: int = SPLIT_CLIPS, engine=None) -> dict:
    """T*'s time split on the first n_clips clips of group 0 (run in the engine's environment); `engine`: an already
    loaded engine of the run's spec (smokes), else one is made and loaded."""
    run_dir = Path(run_dir)
    run = c2.load_run(run_dir)
    pool_dir = c2._abs(run["pool_dir"])
    ks = c2.groups(pool_dir, run["group_shards"])[0]
    rows = c2.load_group(pool_dir, ks)[:n_clips]
    seconds = sum(r["duration"] for r in rows)
    kind = run["engine"]["kind"]
    want = c2.wants_signals(run["engine"])
    if engine is None:
        engine = c2.make_engine(run["engine"], run, timing=kind == "transformers")
        engine.load()
    c2.label_rows(engine, rows)                        # warm-up: every clip with the production setting (compilation)
    warm = rows[:min(32, len(rows))]
    engine.confidence = not want                       # and the other setting's path
    c2.label_rows(engine, warm)
    engine.confidence = want
    reset = getattr(engine, "reset_caches", lambda: None)
    if kind == "transformers":
        engine.times = {k: 0.0 for k in engine.times}
        t0 = time.perf_counter()
        with_out = c2.label_rows(engine, rows)
        wall = time.perf_counter() - t0
        out = {"clips": len(rows), "audio_seconds": seconds, "wall_s": wall, **engine.times,
               "method": "CUDA-synchronized forward hooks (encoder + projector, first LLM forward, later ones)"}
        engine.confidence = not want
        t0 = time.perf_counter()
        other_out = c2.label_rows(engine, rows)
        other = time.perf_counter() - t0
        engine.confidence = want
        again_out = c2.label_rows(engine, rows)
    else:
        clips = [c2._pcm(r) for r in rows]
        reset()
        t0 = time.perf_counter()
        with_out = engine.run(clips)
        full = time.perf_counter() - t0
        reset()
        t0 = time.perf_counter()
        other_out = engine.run(clips, confidence=not want)
        other = time.perf_counter() - t0
        reset()
        again_out = engine.run(clips)
        reset()
        t0 = time.perf_counter()
        engine.run(clips, max_tokens=1, confidence=False)
        first = time.perf_counter() - t0
        out = {"clips": len(rows), "audio_seconds": seconds, "wall_s": full, "encoder_and_prefill_s": first,
               "decode_s": full - first, "encoder_s": None,
               "method": "max_tokens=1 pass on the same clips (encoder + prefill); decode = full - that. The encoder "
                         "alone is not separable in vLLM; see the transformers probe's encoder share. Prefix and "
                         "multimodal caches reset before each pass"}
        wall = full
    on_s, off_s = (wall, other) if want else (other, wall)
    on_out, off_out = (with_out, other_out) if want else (other_out, with_out)
    diff = lambda xs, ys: sum(tuple(a)[:2] != tuple(b)[:2] for a, b in zip(xs, ys))
    changed, repeat = diff(on_out, off_out), diff(with_out, again_out)
    n = len(rows) or 1
    out["signals"] = {"production_has_signals": want, "clips": len(rows), "with_s": on_s, "without_s": off_s,
                      "cost_share": on_s / off_s - 1 if off_s else None, "labels_changed": changed,
                      "labels_changed_share": changed / n, "repeat_changed": repeat, "repeat_changed_share": repeat / n,
                      "note": "the same clips with and without the confidence signals (labels_changed), and twice at the "
                              "production setting (repeat_changed: the engine's run-to-run noise); labels = raw text "
                              "and cap flag"}
    write_json(run_dir / "split.json", out)
    return out


def qualifies(run_contract: dict, consistency: dict, equivalence: dict | None) -> tuple[bool, list[str]]:
    """Whether checks qualify a run under run_contract (c2.contract): a passing consistency result on the full
    stage 1 population under the identical contract, and for vLLM a passing equivalence check under it too."""
    why = []
    if not (consistency and consistency.get("passed")):
        why.append("consistency did not pass")
    elif not consistency.get("full_population") or consistency.get("limit") is not None:
        why.append("consistency result is a limited (smoke) run")
    elif consistency.get("contract") != run_contract:
        why.append("consistency ran under another contract")
    if run_contract["engine"].get("kind") == "vllm":
        if not (equivalence and equivalence.get("passed")):
            why.append("equivalence check did not pass")
        elif equivalence.get("contract") != run_contract:
            why.append("equivalence check ran under another contract")
    return not why, why


def select_production(probes: list[dict], consistency: list[dict], equivalence: dict | None) -> dict:
    """labeling plan "GPU probe": the fastest probed setting whose engine passes the consistency gate and, for vLLM,
    the equivalence check, both under the contract production will use: the probed engine setting, group size
    and B0 batch, with a declared VRAM of at least the probe's peak + 2 GB (that check's declared VRAM is then
    production's)."""
    table, ok = [], []
    for p in sorted(probes, key=lambda p: -(p.get("audio_hours_per_gpu_hour") or 0)):
        pc = p["contract"]
        cands = [c for c in consistency if c.get("contract") and
                 {k: v for k, v in c["contract"].items() if k != "declared_vram_gb"} ==
                 {k: v for k, v in pc.items() if k != "declared_vram_gb"} and
                 c["contract"]["declared_vram_gb"] is not None and
                 c["contract"]["declared_vram_gb"] >= p["declared_vram_gb_for_c2"] - 1e-6]
        good = [c for c in cands if qualifies(c["contract"], c, equivalence)[0]]
        row = {"engine": pc["engine"], "group_shards": pc["group_shards"],
               "audio_hours_per_gpu_hour": p.get("audio_hours_per_gpu_hour"), "peak_vram_gb": p.get("peak_vram_gb"),
               "eligible": bool(good)}
        table.append(row)
        if good and p.get("audio_hours_per_gpu_hour"):
            ok.append((p, good[0]))
    if not ok:
        return {"selected": None, "table": table}
    p, c = ok[0]
    return {"selected": {"contract": c["contract"], "audio_hours_per_gpu_hour": p["audio_hours_per_gpu_hour"]},
            "table": table}


def c2_cost(probe: dict, distinct_hours: float) -> dict:
    rate = probe["audio_hours_per_gpu_hour"]
    return {"distinct_hours": distinct_hours, "audio_hours_per_gpu_hour": rate,
            "c2_gpu_hours": distinct_hours / rate if rate else None,
            "note": "GPU-hours of C2 (T* and B0, reads and decoding included) extrapolated from the probe"}


# --- gate to stage 3 -----------------------------------------------------------------------------------

def drop_table(labels: dict[int, tuple], journal_counts: dict[int, int]) -> dict:
    """Per source: rejected and hallucination-guard shares by distinct ids and by occurrences.
    labels: id hash -> (source, kept, reasons tuple)."""
    from labeling import filters
    t: dict[str, dict] = {}
    for idh, (src, kept, reasons) in labels.items():
        c = t.setdefault(src, {"ids": 0, "ids_rejected": 0, "ids_hallucination": 0, "occurrences": 0,
                               "occ_rejected": 0, "occ_hallucination": 0, "reasons_ids": {}, "reasons_occ": {}})
        n = journal_counts.get(idh, 0)
        hall = any(r in filters.HALLUCINATION for r in reasons)
        c["ids"] += 1
        c["occurrences"] += n
        c["ids_rejected"] += not kept
        c["occ_rejected"] += n * (not kept)
        c["ids_hallucination"] += hall
        c["occ_hallucination"] += n * hall
        for r in reasons:
            c["reasons_ids"][r] = c["reasons_ids"].get(r, 0) + 1
            c["reasons_occ"][r] = c["reasons_occ"].get(r, 0) + n
    for c in t.values():
        c["drop_rate_ids"] = c["ids_rejected"] / c["ids"] if c["ids"] else 0.0
        c["drop_rate_occurrences"] = c["occ_rejected"] / c["occurrences"] if c["occurrences"] else 0.0
        c["hallucination_rate_ids"] = c["ids_hallucination"] / c["ids"] if c["ids"] else 0.0
        c["hallucination_rate_occurrences"] = c["occ_hallucination"] / c["occurrences"] if c["occurrences"] else 0.0
        c["ok"] = max(c["drop_rate_ids"], c["drop_rate_occurrences"]) <= MAX_DROP and \
            max(c["hallucination_rate_ids"], c["hallucination_rate_occurrences"]) <= MAX_HALLUCINATION
    return t


def completeness(pool_dir: Path, run_dir: Path, total_batches: int, audio_hash: bool = True) -> tuple[dict, dict, dict]:
    """(result, labels by id hash, occurrences by id hash). audio_hash: also re-hash every C1 audio file against
    its meta (C1 itself checks only sizes when it finishes)."""
    from labeling.replay import committed_prefix, h64
    pool_dir, run_dir = Path(pool_dir), Path(run_dir)
    problems = []
    for k in committed_prefix(pool_dir):
        try:
            shards.verify_shard(pool_dir, k, audio_hash=audio_hash)
        except (RuntimeError, OSError) as exc:
            problems.append(str(exc))
    status = json.loads((pool_dir / "status.json").read_text()) if (pool_dir / "status.json").exists() else {}
    if not status.get("complete"):
        problems.append("C1 is not complete")
    manifest = json.loads((pool_dir / "manifest.json").read_text()) if (pool_dir / "manifest.json").exists() else {}
    if manifest and manifest.get("batches") != total_batches:
        problems.append(f"C1 covers {manifest.get('batches')} batches, not {total_batches}")
    labels: dict[int, tuple] = {}
    duplicates = 0
    import pyarrow.parquet as pq
    for p in sorted((run_dir / "shards").glob("*.labels.parquet")):
        for r in pq.read_table(p, columns=["id", "source", "disposition", "reasons"]).to_pylist():
            idh = h64(r["id"])
            if idh in labels:
                duplicates += 1
            labels[idh] = (r["source"], r["disposition"] == "kept", tuple(r["reasons"]))
    occ: dict[int, int] = defaultdict(int)
    missing, wrong_source, batches_seen, prev = 0, 0, set(), (-1, -1)
    order_ok = True
    firsts = set()
    for k in committed_prefix(pool_dir):
        for r in shards.read_journal(shards.paths(pool_dir, k)["journal"]):
            key = (r["batch"], r["pos"])
            order_ok &= key > prev
            prev = key
            batches_seen.add(r["batch"])
            idh = h64(r["id"])
            occ[idh] += 1
            if r["first"]:
                firsts.add(idh)
            lab = labels.get(idh)
            if lab is None:
                missing += 1
            elif lab[0] != r["source"]:
                wrong_source += 1
    extra = len(set(labels) - set(occ))
    conflicts = manifest.get("id_conflicts", [])
    gaps = total_batches - len(batches_seen & set(range(total_batches)))
    for name, n in (("occurrences without a record", missing), ("records without an occurrence", extra),
                    ("duplicate records", duplicates), ("records with another source", wrong_source),
                    ("id conflicts", len(conflicts)), ("batches without an occurrence", gaps)):
        if n:
            problems.append(f"{n} {name}")
    if not order_ok:
        problems.append("journal is not in consumption order")
    if firsts != set(occ):
        problems.append("first-occurrence flags do not match the distinct ids")
    result = {"ok": not problems, "problems": problems, "occurrences": sum(occ.values()), "distinct_ids": len(occ),
              "records": len(labels), "id_conflicts": conflicts[:100]}
    return result, labels, occ


def independent_replay(pinned: dict, pool_dir: Path, n_batches: int = layout.INDEPENDENT_REPLAY_BATCHES,
                       stream_factory=None) -> dict:
    from labeling import replay
    fresh = replay.journal_of(pinned, n_batches, stream_factory=stream_factory)
    stored = replay.stored_journal(pool_dir, n_batches)
    keys = ("batch", "pos", "id", "source", "duration", "first")
    same_seq = [{k: r[k] for k in keys} for r in fresh] == [{k: r[k] for k in keys} for r in stored]
    audio_ok = True
    if same_seq:
        want = {r["id"]: r["pcm_sha256"] for r in fresh if r["first"]}
        got = {}
        for k in shards.committed(pool_dir):
            meta = shards.load_meta(pool_dir, k)
            if meta["batches"][0] >= n_batches:
                break
            for r in shards.iter_audio(shards.paths(pool_dir, k)["audio"], columns=("id", "pcm_sha256")):
                if r["id"] in want:
                    got[r["id"]] = r["pcm_sha256"]
        audio_ok = got == want
    return {"ok": same_seq and audio_ok, "batches": n_batches, "occurrences": len(fresh), "same_sequence": same_seq,
            "same_audio": audio_ok}


def confidence_summary(rec: dict) -> dict | None:
    """Token count, mean and minimum chosen log-probability, mean margin to the runner-up (for reading only)."""
    import numpy as np
    if rec.get("t_token_ids") is None:
        return None
    lp = np.asarray(rec["t_logprob"], np.float64)
    ru = np.asarray(rec["t_runner_up"], np.float64)
    return {"tokens": len(rec["t_token_ids"]), "mean_logprob": float(lp.mean()), "min_logprob": float(lp.min()),
            "mean_margin": float((lp - ru).mean())}


def spot_check(run_dir: Path, out_dir: Path, per_source: int = layout.SPOT_CHECK_PER_SOURCE,
               seed: int = layout.SPOT_CHECK_SEED, pool_dir: Path | None = None) -> dict:
    """per_source seeded random records per source (all dispositions), with the reference when the source has
    one and B0's label. Two passes over the label shards, so only the sampled records are held in memory."""
    import pyarrow.parquet as pq
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    files = sorted((run_dir / "shards").glob("*.labels.parquet"))
    by_source: dict[str, list[str]] = defaultdict(list)
    for p in files:
        for r in pq.read_table(p, columns=["id", "source"]).to_pylist():
            by_source[r["source"]].append(r["id"])
    pick = {src: set(random.Random(f"{seed}:{src}").sample(sorted(ids), min(per_source, len(ids))))
            for src, ids in by_source.items()}
    wanted = set().union(*pick.values()) if pick else set()
    recs = {}
    for p in files:
        for r in c2._read_parquet(p):
            if r["id"] in wanted:
                recs[r["id"]] = r
    refs = {}
    if pool_dir is not None:
        for k in shards.committed(pool_dir):
            for r in shards.iter_audio(shards.paths(pool_dir, k)["audio"], columns=("id", "reference"), batch_rows=8192):
                if r["id"] in wanted:
                    refs[r["id"]] = r["reference"]
    counts = {}
    from pool import common
    for src in sorted(pick):
        chosen = [recs[i] for i in sorted(pick[src])]
        lines = [f"# {src}: {len(chosen)} of {len(by_source[src])} records (seed {seed}); T* label, then B0's"]
        for r in chosen:
            lines.append(f"\n## {r['id']} | {r['duration']:.2f} s | {r['disposition']}"
                         + (f" ({', '.join(r['reasons'])})" if r["reasons"] else ""))
            if refs.get(r["id"]) is not None:
                lines.append(f"ref ({r['reference_kind']}): {refs[r['id']]}")
            lines.append(f"T*: {r['raw']}")
            lines.append(f"B0: {r['b0_raw']}")
            conf = confidence_summary(r)
            if conf is not None:
                lines.append(f"conf: {conf['tokens']} tokens, mean logprob {conf['mean_logprob']:.3f}, "
                             f"min {conf['min_logprob']:.3f}, mean margin {conf['mean_margin']:.3f}")
        common.atomic_write_bytes(out_dir / "spotcheck" / f"{src}.txt", ("\n".join(lines) + "\n").encode())
        counts[src] = len(chosen)
    return counts


def gate(pinned: dict, pool_dir: Path, run_dir: Path, consistency_result: Path, out_dir: Path,
         total_batches: int = layout.TOTAL_BATCHES, stream_factory=None, equivalence_result: Path | None = None,
         audio_hash: bool = True) -> dict:
    out_dir = Path(out_dir)
    cons = json.loads(Path(consistency_result).read_text())
    eq = json.loads(Path(equivalence_result).read_text()) if equivalence_result else None
    run = c2.load_run(run_dir)
    engine_same, why = qualifies(c2.contract(run), cons, eq)
    comp, labels, occ = completeness(pool_dir, run_dir, total_batches, audio_hash=audio_hash)
    drops = drop_table(labels, occ)
    rep = independent_replay(pinned, pool_dir, min(layout.INDEPENDENT_REPLAY_BATCHES, total_batches), stream_factory)
    spots = spot_check(run_dir, out_dir, pool_dir=pool_dir)
    items = {"consistency": engine_same, "drops": all(c["ok"] for c in drops.values()),
             "completeness": comp["ok"] and rep["ok"], "spot_check_written": bool(spots)}
    result = {"passed": all(items.values()), "items": items, "consistency": {"passed": cons["passed"],
              "qualifies_this_run": engine_same, "why_not": why, "sets": cons["sets"],
              "equivalence": None if eq is None else {"passed": eq["passed"]}}, "drops": drops,
              "completeness": comp, "independent_replay": rep, "spot_check": spots,
              "thresholds": {"drop": MAX_DROP, "hallucination": MAX_HALLUCINATION,
                             "drop_rates": "both by distinct ids and by journal occurrences"}}
    write_json(out_dir / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="labeling.checks")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--run-dir", type=Path, required=True)
    s.add_argument("--deadline", type=float)
    args = parser.parse_args(argv)
    layout.require_mount()
    if args.cmd == "split":
        print(json.dumps(split_step(args.run_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
