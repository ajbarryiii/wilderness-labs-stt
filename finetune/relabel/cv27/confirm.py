"""B0's confirmation of the read sentence (b0prime plan "Extra sources" "B0 confirmation").

B0 is the pinned Parakeet TDT 0.6B v2 .nemo (evaluate.describe_source verifies its SHA-256 against the lock), run as
the B0 labels of the labeling pass were (labeling/engines.py B0Labeler: parakeet-ternary teacher.teacher_label_batch,
bf16 encoder, FP32 greedy TDT, filters off), on batches cut by train.cut_pool with count x longest <= batch_seconds.
Each decoded clip (store status "ok") gets B0's greedy transcript and

    wer = evaluate.utterance_wer(label, B0's transcript)      the evaluation normalizer; the reference is the label

The reference is the sentence as it is trained on: its filters version 3 text (the stored label, before cleaning).
The evaluation normalizer deletes bracketed text, so scoring the sentence as written would leave a parenthetical's
words, which the label keeps, unchecked (the reason filters version 3 scores mapped text; review r1). A clip is kept
iff wer <= B0_MAX_WER (0.5); an infinite WER (the label normalizes to nothing and B0 wrote words) drops it. B0's
transcripts are kept for every checked clip.

Chunks follow decode.json: confirm/chunks/<k>.jsonl ({"id", "b0", "wer", "wer_infinite"} per "ok" clip, in id order)
with a <k>.json sidecar (settings, timings), each written atomically once the chunk is done, so a job resumes at a
chunk boundary. confirm/run.json fixes the settings (selection SHA-256, batch_seconds, engine, model SHA-256, rule) at
the first chunk; a later job with other settings is refused, so every clip is judged by one configuration, and every
step refuses a store made from another selection table (store.chunk_records).
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from cv27 import layout, store

RULE = ("keep iff evaluate.utterance_wer(label, B0 greedy) <= 0.5, label = the sentence's filters version 3 text "
        "(an infinite WER drops)")


def utterance_wer(reference: str, hyp: str) -> float:
    import evaluate  # finetune/parakeet-ternary
    return evaluate.utterance_wer(reference or "", hyp or "")


def keep(wer: float) -> bool:
    return math.isfinite(wer) and wer <= layout.B0_MAX_WER


def judged(label: str, hyp: str) -> dict:
    """B0's transcript judged against the clip's training label."""
    w = utterance_wer(label, hyp)
    return {"b0": hyp, "wer": w if math.isfinite(w) else None, "wer_infinite": not math.isfinite(w),
            "kept": keep(w)}


def accent_report(rows: Iterable[tuple[str, bool]]) -> dict:
    """Drop rate per accent field value (the empty field as "") and per accent tag (the field split on "|")."""
    field, tag = defaultdict(Counter), defaultdict(Counter)
    for accents, dropped in rows:
        accents = accents or ""
        field[accents]["checked"] += 1
        field[accents]["dropped"] += int(dropped)
        for t in ({a.strip() for a in accents.split("|") if a.strip()} or {""}):
            tag[t]["checked"] += 1
            tag[t]["dropped"] += int(dropped)

    def table(d):
        return {k: {"checked": v["checked"], "dropped": v["dropped"], "rate": v["dropped"] / v["checked"]}
                for k, v in sorted(d.items(), key=lambda kv: (-kv[1]["checked"], kv[0]))}
    return {"by_field": table(field), "by_tag": table(tag)}


# --- audio ----------------------------------------------------------------------------------------------------------

def load_audio(records: Sequence[dict], threads: int = 8) -> dict[str, np.ndarray]:
    """PCM16 of the chunk's "ok" clips, read in file order, each checked against its FLAC and PCM SHA-256."""
    import concurrent.futures as cf
    import hashlib

    def one(r):
        raw = Path(r["audio_filepath"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != r["sha256"]:
            raise RuntimeError(f"{r['audio_filepath']}: SHA-256 differs from the decode record")
        pcm = store.decode_flac(raw)
        if len(pcm) != r["samples"] or store.pcm_sha256(pcm) != r["pcm_sha256"]:
            raise RuntimeError(f"{r['audio_filepath']}: samples differ from the decode record")
        return r["id"], pcm

    ok = [r for r in records if r["status"] == "ok"]
    with cf.ThreadPoolExecutor(threads) as ex:
        return dict(ex.map(one, ok))


def run_chunk(labeler, records: Sequence[dict], labels: dict[str, str], audio: dict[str, np.ndarray]) -> list[dict]:
    """B0 over the chunk's "ok" clips, each judged against its training label; in record (id) order."""
    ok = [r for r in records if r["status"] == "ok"]
    durations = [r["samples"] / layout.SAMPLE_RATE for r in ok]
    hyps: dict[str, str] = {}
    for batch in labeler.plan(durations):
        clips = [audio[ok[i]["id"]].astype(np.float32) / 32768.0 for i in batch]
        for i, text in zip(batch, labeler.run(clips)):
            hyps[ok[i]["id"]] = text
    return [{"id": r["id"], **judged(labels[r["id"]], hyps[r["id"]])} for r in ok]


def settings(labeler, batch_seconds: float, selection_sha256: str) -> dict:
    import evaluate
    src = evaluate.describe_source("pretrained", None, "B0")
    return {"selection_sha256": selection_sha256, "batch_seconds": float(batch_seconds), "engine": labeler.info(),
            "model_sha256": src["sha256"], "rule": RULE, "max_wer": layout.B0_MAX_WER,
            "normalizer": evaluate.NORMALIZER}


def fix_settings(paths: layout.Paths, current: dict) -> None:
    from pool import common
    if paths.confirm_run.exists():
        old = json.loads(paths.confirm_run.read_text())
        if old != current:
            raise RuntimeError(f"{paths.confirm_run} fixes other settings: {old} != {current}")
        return
    common.atomic_write_json(paths.confirm_run, current)


def read_confirm(paths: layout.Paths, k: int, records: Sequence[dict]) -> list[dict] | None:
    path = paths.confirm_chunk(k)
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if [r["id"] for r in rows] != [r["id"] for r in records if r["status"] == "ok"]:
        raise RuntimeError(f"{path}: its ids are not the chunk's decoded clips")
    return rows


def confirm_chunks(paths: layout.Paths, rows: Sequence[dict], selection_sha256: str, labeler, batch_seconds: float,
                   chunks: Sequence[int], deadline: float | None = None, threads: int = 8, log=print) -> dict:
    """B0 over the given chunks (decoded; not yet confirmed); stops starting chunks after `deadline`. The next
    chunk's audio is read in a background thread while B0 runs on the current one."""
    import concurrent.futures as cf

    from pool import common
    per_chunk, body = store.chunk_records(paths, rows, selection_sha256)
    labels = {r["id"]: r["label"] for r in rows}
    fix_settings(paths, settings(labeler, batch_seconds, selection_sha256))
    todo = []
    for k in chunks:
        if per_chunk[k] is None:
            raise RuntimeError(f"chunk {k} is not decoded yet")
        if read_confirm(paths, k, per_chunk[k]) is None:
            todo.append(k)
    done, skipped = 0, len(chunks) - len(todo)
    totals = Counter()
    reader = cf.ThreadPoolExecutor(1)
    pending = None
    try:
        for i, k in enumerate(todo):
            if deadline is not None and time.monotonic() > deadline:
                log(f"confirm: budget reached before chunk {k}")
                break
            recs = per_chunk[k]
            t0 = time.monotonic()
            audio = pending.result() if pending is not None else load_audio(recs, threads)
            pending = reader.submit(load_audio, per_chunk[todo[i + 1]], threads) if i + 1 < len(todo) else None
            t1 = time.monotonic()
            out = run_chunk(labeler, recs, labels, audio)
            t2 = time.monotonic()
            del audio
            audio_s = sum(r["samples"] for r in recs if r["status"] == "ok") / layout.SAMPLE_RATE
            common.atomic_write_lines(paths.confirm_chunk(k), (json.dumps({x: o[x] for x in ("id", "b0", "wer", "wer_infinite")},
                                                                          ensure_ascii=False) for o in out))
            common.atomic_write_json(paths.confirm_chunk(k).with_suffix(".json"),
                                     {"chunk": k, "clips": len(out), "audio_seconds": audio_s, "read_wait_seconds": t1 - t0,
                                      "b0_seconds": t2 - t1, "dropped": sum(not o["kept"] for o in out)})
            totals.update(clips=len(out), audio_seconds=audio_s, read_wait_seconds=t1 - t0, b0_seconds=t2 - t1)
            done += 1
            log(f"confirm: chunk {k}: {len(out)} clips, {audio_s / 3600:.2f} h, read wait {t1 - t0:.1f} s, "
                f"B0 {t2 - t1:.1f} s, dropped {sum(not o['kept'] for o in out)}")
    finally:
        reader.shutdown(wait=True, cancel_futures=True)
    return {"confirmed_chunks": done, "already_done": skipped, **totals}


def finalize(paths: layout.Paths, rows: Sequence[dict], selection_meta: dict) -> dict:
    """The clip source, B0's transcripts and the counts, once every chunk is decoded and confirmed."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pool import common
    per_chunk, body = store.chunk_records(paths, rows, selection_meta["table_sha256"])
    run = json.loads(paths.confirm_run.read_text())
    if run["selection_sha256"] != selection_meta["table_sha256"]:
        raise RuntimeError(f"{paths.confirm_run} judged another selection table")
    by_id = {r["id"]: r for r in rows}
    status, hours = Counter(), Counter()
    kept, b0rows, accent_rows = [], [], []
    for k, recs in enumerate(per_chunk):
        if recs is None:
            raise RuntimeError(f"chunk {k} is not decoded")
        conf = read_confirm(paths, k, recs)
        if conf is None:
            raise RuntimeError(f"chunk {k} is not confirmed")
        conf_by_id = {c["id"]: c for c in conf}
        for r in recs:
            sel = by_id[r["id"]]
            if r["status"] != "ok":
                status[r["status"]] += 1
                hours[r["status"]] += (sel["ms"] or 0) / 3.6e6
                continue
            c = conf_by_id[r["id"]]
            w = math.inf if c["wer_infinite"] else c["wer"]
            ok = keep(w)
            accent_rows.append((sel["accents"], not ok))
            b0rows.append({"id": r["id"], "b0": c["b0"], "wer": c["wer"], "wer_infinite": c["wer_infinite"], "kept": ok})
            if not ok:
                status["b0_wer_gt_0.5"] += 1
                hours["b0_wer_gt_0.5"] += r["samples"] / layout.SAMPLE_RATE / 3600
                continue
            kept.append({**r, "label": sel["label"], "sentence": sel["sentence"], "accents": sel["accents"],
                         "b0": c["b0"], "b0_wer": c["wer"], "speaker": sel["client_id"]})
    meta = store.write_source(paths, kept, {
        "selection_sha256": body["selection_sha256"], "decode_plan_sha256": common.sha256_file(paths.decode_plan),
        "label": {"field": "label", "filters_version": layout.FILTERS_VERSION, "cleaning":
                  "not applied: the reader applies cleaning layer 1 alone at its cleaning version "
                  "(cv27.store.label_layer1); layers 0 and 2 never apply to this source"},
        "b0_confirmation": {k: run[k] for k in ("rule", "max_wer", "batch_seconds", "model_sha256")},
        "speakers": len({c["speaker"] for c in kept})})
    t = pa.table({k: [r[k] for r in b0rows] for k in ("id", "b0", "wer", "wer_infinite", "kept")})
    paths.b0_transcripts.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, paths.b0_transcripts, compression="zstd")
    checked = len(b0rows)
    report = {"selection": selection_meta["counts"], "store": {
        "selected": len(rows), "undecodable": status["undecodable"], "decoded_duration": status["decoded_duration"],
        "checked_by_b0": checked, "b0_wer_gt_0.5": status["b0_wer_gt_0.5"],
        "b0_drop_rate": status["b0_wer_gt_0.5"] / checked if checked else None, "hours_dropped": dict(hours)},
        "accents": accent_report(accent_rows),
        "final": {"clips": meta["clips"], "hours": meta["hours"], "speakers": meta["speakers"],
                  "clips_sha256": meta["clips_sha256"]},
        "b0_transcripts_sha256": common.sha256_file(paths.b0_transcripts), "confirm_settings": run}
    common.atomic_write_json(paths.report, report)
    return report
