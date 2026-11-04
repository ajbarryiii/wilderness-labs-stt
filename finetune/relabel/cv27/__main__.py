"""CLI of the read-speech extra source (b0prime plan "Extra sources"); run from finetune/relabel as
`./python -m cv27 COMMAND`. Long commands run as `job` chunks (cv27/launch.sh).

    boilerplate [--workers 6]       reproduce the global denylist's boilerplate 8-grams (pool scan, CPU, ~6 min)
    select                          the selection table and the dev sample (CPU)
    decode [--workers 8] [--budget-minutes 110]     decode pending chunks to the FLAC store (CPU, resumable)
    dev [--workers 4]               the dev sample's FLAC and its scorer manifest (CPU)
    confirm --batch-seconds S --declared-vram-gb V [--budget-minutes 110] [--lock-wait 1800]
                                    B0 over pending chunks, holding the parakeet GPU lock for the job (GPU)
    finalize                        the clip source, B0 transcripts and report.json
    probe-decode [--clips 2000] [--workers 8]       decode a seeded sample into smoke/cv27-en/probe (CPU)
    probe-b0 --batch-seconds S --declared-vram-gb V time B0 on the probe sample; the GPU-hour estimate (GPU)
    status                          progress of the store and the B0 check
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import resource
import sys
import time
from pathlib import Path

from cv27 import confirm, dev, heldout, layout, select, store


def log(msg: str) -> None:
    from pool import common
    common.log(msg)


def _denylist() -> tuple[dict, str]:
    from pool import common
    path = common.pool_dir() / "denylist.json"
    return json.loads(path.read_text()), common.sha256_file(path)


def cmd_boilerplate(args) -> int:
    from pool import common
    layout.require_mount()
    paths = layout.Paths(layout.out_dir())
    deny, deny_sha = _denylist()
    if paths.boilerplate.exists():
        heldout.load_boilerplate(paths.boilerplate, deny_sha)
        log(f"boilerplate: {paths.boilerplate} is current")
        return 0
    index = heldout.load_denylist_index(deny)
    caches = {s: common.pool_dir() / "text" / s for s in ("yodas", "peoples_speech")}
    corpora = {k: Path(v["manifest"]) for k, v in deny["training"].items() if isinstance(v, dict) and "manifest" in v}
    hashes, info = heldout.reproduce_boilerplate(deny, index, caches, corpora, args.workers)
    heldout.save_boilerplate(paths.boilerplate, hashes, info, deny_sha)
    log(f"boilerplate: {len(hashes)} 8-grams, reproduced {info['reproduced']}")
    return 0


def _file_record(path) -> dict:
    from pool import common
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": common.sha256_file(path)}


def cmd_select(args) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import testsets  # finetune/parakeet-ternary
    from labeling import filters
    from pool import common
    layout.require_mount()
    t0 = time.time()
    paths = layout.Paths(layout.out_dir())
    raw = layout.raw_dir()
    deny, deny_sha = _denylist()
    boiler = heldout.load_boilerplate(paths.boilerplate, deny_sha)
    normalize = heldout.Normalizer()
    esb_files = testsets.parquet_files(layout.esb_test_set())

    def tsv_rows(name):
        return heldout.read_tsv(raw / name, ("client_id", "path", "sentence"))

    held = heldout.build_heldout(heldout.esb_rows(esb_files), tsv_rows, normalize)
    log(f"select: held out {held.counts}")
    talk_path = layout.talk_text()
    talk = heldout.talk_grams(heldout.read_talk_texts(talk_path), normalize)
    index = heldout.load_denylist_index(deny)
    containment = heldout.Containment(index, boiler)
    tok = filters.b0_tokenizer_file(layout.tokenizer_cache())
    coverage = filters.Coverage(tok)
    durations = heldout.read_durations(raw / layout.DURATIONS_TSV)
    validated = heldout.read_tsv(raw / layout.TRAIN_TSV, ("client_id", "path", "sentence", "accents"))
    clips, counts = select.select(validated, held, durations, talk, containment, coverage, normalize)
    log(f"select: stages {counts['stages']}")
    table_sha = select.write_table(paths.selection, clips)

    eligible = dev.eligible(heldout.read_tsv(raw / "dev.tsv", ("client_id", "path", "sentence", "accents")), held,
                            normalize)
    sample = dev.sample(eligible)
    ms = {c: durations.get(c) for c in (r["path"] for r in sample)}
    t = pa.table({"id": [heldout.clip_id(r["path"]) for r in sample], "path": [r["path"] for r in sample],
                  "client_id": [r["client_id"] for r in sample], "sentence": [r["sentence"] for r in sample],
                  "accents": [r.get("accents") or "" for r in sample],
                  "ms": pa.array([ms[r["path"]] for r in sample], pa.int64())})
    paths.dev_sample.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, paths.dev_sample)
    counts["dev"] = {"eligible": len(eligible), "sample": len(sample),
                     "speakers": len({r["client_id"] for r in sample}),
                     "hours": sum(v or 0 for v in ms.values()) / 3.6e6}
    meta = {"alias": layout.ALIAS, "created": common.now(), "seconds": round(time.time() - t0, 1),
            "table_sha256": table_sha, "dev_sample_sha256": common.sha256_file(paths.dev_sample),
            "params": {"speaker_cap": layout.SPEAKER_CAP, "sentence_cap": layout.SENTENCE_CAP,
                       "cap_seed": layout.CAP_SEED, "ms_range": [layout.MIN_MS, layout.MAX_MS],
                       "filters_version": layout.FILTERS_VERSION, "ngram": layout.NGRAM,
                       "dev": {"size": layout.DEV_SIZE, "per_speaker": layout.DEV_PER_SPEAKER, "seed": layout.DEV_SEED},
                       "reasons": list(select.REASONS), "normalizer": "finetune/parakeet-ternary/evaluate.py normalize"},
            "inputs": {"tsvs": [_file_record(raw / n) for n in (*layout.TSVS, layout.DURATIONS_TSV)],
                       "esb_test": [_file_record(p) for p in esb_files], "talk_text": _file_record(talk_path),
                       "denylist_sha256": deny_sha, "boilerplate": json.loads(paths.boilerplate.read_text()),
                       "tokenizer_sha256": common.sha256_file(tok)},
            "counts": counts}
    common.atomic_write_json(paths.selection_meta, meta)
    print(json.dumps({"stages": counts["stages"], "dev": counts["dev"]}, indent=1))
    return 0


def _selected():
    """(production paths, selected rows, selection meta)."""
    paths = layout.Paths(layout.out_dir())
    rows, meta = select.read_selected(paths)
    return paths, rows, meta


def _deadline(minutes: float | None) -> float | None:
    return None if minutes is None else time.monotonic() + 60.0 * minutes


def cmd_decode(args) -> int:
    layout.require_mount()
    paths, rows, meta = _selected()
    sha = meta["table_sha256"]
    body = store.plan(paths, sha, [r["id"] for r in rows])
    out = store.decode_chunks(paths, rows, sha, layout.raw_dir() / "clips", range(len(body["chunks"])), args.workers,
                              _deadline(args.budget_minutes), log)
    per, _ = store.chunk_records(paths, rows, sha)
    out["chunks_done"] = sum(p is not None for p in per)
    out["chunks"] = len(per)
    out["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(json.dumps(out))
    return 0


def cmd_dev(args) -> int:
    import concurrent.futures as cf
    import multiprocessing as mp

    import pyarrow.parquet as pq
    from pool import common
    layout.require_mount()
    paths = layout.Paths(layout.out_dir())
    meta = json.loads(paths.selection_meta.read_text())
    if common.sha256_file(paths.dev_sample) != meta["dev_sample_sha256"]:
        raise RuntimeError(f"{paths.dev_sample} does not match the selection meta")
    sample = pq.read_table(paths.dev_sample).to_pylist()
    raw_clips = layout.raw_dir() / "clips"
    tasks = [(r["id"], str(raw_clips / r["path"]), str(paths.dev_audio / f"{heldout.clip_stem(r['path'])}.flac"))
             for r in sample]
    with cf.ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as ex:
        recs = {r["id"]: r for r in ex.map(store.decode_clip, tasks, chunksize=8)}
    bad = {k: v["status"] for k, v in recs.items() if v["status"] != "ok"}
    kept = [r for r in sample if recs[r["id"]]["status"] == "ok"]
    rows = dev.manifest_rows(kept, recs)
    common.atomic_write_lines(paths.dev_manifest, (json.dumps(r, ensure_ascii=False) for r in rows))
    dmeta = {"name": layout.DEV_SET, "path": str(paths.dev_manifest), "utterances": len(rows),
             "hours": sum(r["duration"] for r in rows) / 3600, "source": layout.ALIAS, "license": "CC0-1.0",
             "text_field": "text", "text_kind": "the read prompt sentence as written (human)", "kind": "dev",
             "informational": True, "selection": dev.__doc__.split("\n\n")[1].replace("\n", " "),
             "not_decoded": bad, "speakers": len({r["client_id"] for r in kept}),
             "manifest_sha256": common.sha256_file(paths.dev_manifest),
             "dev_sample_sha256": meta["dev_sample_sha256"],
             "audio": "16 kHz mono PCM_16 FLAC (cv27/store.py decode_bytes)"}
    common.atomic_write_json(paths.dev_manifest.with_suffix(".meta.json"), dmeta)
    print(json.dumps({k: dmeta[k] for k in ("utterances", "hours", "speakers", "not_decoded")}))
    return 0


def _gpu_lock(label: str, wait: float):
    import score_teacher
    return score_teacher.gpu_lock(label, wait)


def _labeler(args):
    from labeling.engines import B0Labeler
    labeler = B0Labeler(args.batch_seconds, declared_vram_gb=args.declared_vram_gb)
    labeler.load()
    return labeler


def cmd_confirm(args) -> int:
    layout.require_mount()
    paths, rows, meta = _selected()
    sha = meta["table_sha256"]
    with _gpu_lock("relabel cv27 confirm", args.lock_wait):
        labeler = _labeler(args)
        per, body = store.chunk_records(paths, rows, sha)
        ready = [k for k, p in enumerate(per) if p is not None]
        out = confirm.confirm_chunks(paths, rows, sha, labeler, args.batch_seconds, ready,
                                     _deadline(args.budget_minutes), args.threads, log)
    import torch
    out["peak_vram_gb"] = torch.cuda.max_memory_reserved() / 2**30
    out["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    out["pending_decode"] = sum(p is None for p in per)
    print(json.dumps(out))
    return 0


def cmd_finalize(args) -> int:
    layout.require_mount()
    paths, rows, meta = _selected()
    report = confirm.finalize(paths, rows, meta)
    print(json.dumps({"store": report["store"], "final": report["final"]}, indent=1))
    return 0


def _probe_rows(n: int) -> tuple[list[dict], dict]:
    import numpy as np
    rows, meta = select.read_selected(layout.Paths(layout.out_dir()))
    pick = np.sort(np.random.default_rng(layout.PROBE_SEED).choice(len(rows), size=min(n, len(rows)), replace=False))
    return [rows[i] for i in pick.tolist()], meta


def cmd_probe_decode(args) -> int:
    import shutil
    layout.require_mount()
    base = layout.smoke_dir("probe")
    if layout.is_production(base):
        raise RuntimeError(f"{base} is a production path")
    if base.exists():     # a fresh probe: reused chunks would make the timing meaningless (review r1)
        shutil.rmtree(base)
    paths = layout.Paths(base)
    rows, meta = _probe_rows(args.clips)
    sha = meta["table_sha256"]
    body = store.plan(paths, sha, [r["id"] for r in rows])
    t0 = time.monotonic()
    out = store.decode_chunks(paths, rows, sha, layout.raw_dir() / "clips", range(len(body["chunks"])), args.workers,
                              None, log)
    wall = time.monotonic() - t0
    if out["already_done"]:
        raise RuntimeError("probe-decode reused chunks; its timing is not a measurement")
    per, _ = store.chunk_records(paths, rows, sha)
    recs = [r for p in per for r in p]
    audio_s = sum(r.get("samples", 0) for r in recs if r["status"] == "ok") / layout.SAMPLE_RATE
    res = {**out, "clips": len(recs), "status": dict(collections.Counter(r["status"] for r in recs)),
           "wall_s": wall, "clips_per_s": len(recs) / wall, "audio_s_per_s": audio_s / wall, "workers": args.workers,
           "peak_rss_parent_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
           "flac_bytes": sum(os.path.getsize(r["audio_filepath"]) for r in recs if r["status"] == "ok"),
           "selected_clips": meta["counts"]["stages"]["selected"]["clips"],
           "selected_hours": meta["counts"]["stages"]["selected"]["hours"]}
    res["estimate"] = {"decode_hours_at_these_workers": res["selected_clips"] / res["clips_per_s"] / 3600,
                       "store_tb": res["flac_bytes"] / max(audio_s, 1e-9) * res["selected_hours"] * 3600 / 1e12}
    from pool import common
    common.atomic_write_json(paths.base / "probe-decode.json", res)
    print(json.dumps(res, indent=1))
    return 0


def cmd_probe_b0(args) -> int:
    import torch
    from pool import common
    layout.require_mount()
    paths = layout.Paths(layout.smoke_dir("probe"))
    rows, meta = _probe_rows(layout.PROBE_CLIPS)
    per, _ = store.chunk_records(paths, rows, meta["table_sha256"])
    if any(p is None for p in per):
        raise RuntimeError("run probe-decode first")
    labels = {r["id"]: r["label"] for r in rows}
    with _gpu_lock("relabel cv27 probe-b0", args.lock_wait):
        t0 = time.monotonic()
        labeler = _labeler(args)
        t_load = time.monotonic() - t0
        read_s = b0_s = audio_s = 0.0
        results = []
        for recs in per:
            t1 = time.monotonic()
            audio = confirm.load_audio(recs, args.threads)
            t2 = time.monotonic()
            results += confirm.run_chunk(labeler, recs, labels, audio)
            torch.cuda.synchronize()
            t3 = time.monotonic()
            read_s += t2 - t1
            b0_s += t3 - t2
            audio_s += sum(r["samples"] for r in recs if r["status"] == "ok") / layout.SAMPLE_RATE
    hours = meta["counts"]["stages"]["selected"]["hours"]
    res = {"batch_seconds": args.batch_seconds, "declared_vram_gb": args.declared_vram_gb, "clips": len(results),
           "audio_hours": audio_s / 3600, "model_load_s": t_load, "read_s": read_s, "b0_s": b0_s,
           "rtfx_b0": audio_s / b0_s, "rtfx_with_read": audio_s / (b0_s + read_s),
           "peak_vram_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
           "peak_vram_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
           "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
           "dropped": sum(not r["kept"] for r in results), "selected_hours": hours,
           "gpu_hours_estimate_b0_only": hours * 3600 / (audio_s / b0_s) / 3600,
           "gpu_hours_estimate_serial_reads": hours * 3600 / (audio_s / (b0_s + read_s)) / 3600,
           "note": "the full pass reads the next chunk while B0 runs (confirm.confirm_chunks); reads here are serial"}
    common.atomic_write_json(paths.base / f"probe-b0-{int(args.batch_seconds)}.json", res)
    print(json.dumps(res, indent=1))
    return 0


def cmd_status(args) -> int:
    paths, rows, meta = _selected()
    if not paths.decode_plan.exists():
        print(json.dumps({"selected": len(rows), "decode_plan": None}))
        return 0
    per, body = store.chunk_records(paths, rows, meta["table_sha256"])
    conf = [k for k, p in enumerate(per) if p is not None and confirm.read_confirm(paths, k, p) is not None]
    print(json.dumps({"selected": len(rows), "chunks": len(per), "decoded": sum(p is not None for p in per),
                      "confirmed": len(conf), "finalized": paths.clips_meta.exists()}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cv27", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("boilerplate")
    b.add_argument("--workers", type=int, default=6)
    sub.add_parser("select")
    d = sub.add_parser("decode")
    d.add_argument("--workers", type=int, default=8)
    d.add_argument("--budget-minutes", type=float, default=None)
    v = sub.add_parser("dev")
    v.add_argument("--workers", type=int, default=4)
    for name in ("confirm", "probe-b0"):
        c = sub.add_parser(name)
        c.add_argument("--batch-seconds", type=float, required=True)
        c.add_argument("--declared-vram-gb", type=float, required=True)
        c.add_argument("--lock-wait", type=float, default=1800.0)
        c.add_argument("--threads", type=int, default=8)
        if name == "confirm":
            c.add_argument("--budget-minutes", type=float, default=None)
    sub.add_parser("finalize")
    p = sub.add_parser("probe-decode")
    p.add_argument("--clips", type=int, default=layout.PROBE_CLIPS)
    p.add_argument("--workers", type=int, default=8)
    sub.add_parser("status")
    args = parser.parse_args(argv)
    return {"boilerplate": cmd_boilerplate, "select": cmd_select, "decode": cmd_decode, "dev": cmd_dev,
            "confirm": cmd_confirm, "finalize": cmd_finalize, "probe-decode": cmd_probe_decode,
            "probe-b0": cmd_probe_b0, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
