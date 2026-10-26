"""Round-trip filter, filter report and listening sample (synthetic plan "Speech").

transcribe_chunk: T* (stage 2's labeling engine, labeling/engines.py: the transformers decoder in the shared
runtime, or the vLLM engine in its own environment) transcribes every rendered clip. Clips are processed in blocks
of BLOCK clip ids (sorted), each written atomically to runs/<run>/roundtrip/<block>.jsonl; within a block the
engine's plan cuts the clips sorted by descending duration (ties by id), as C2 does. A clip is KEPT iff
    its duration is within [MIN_SECONDS, MAX_SECONDS], T* did not hit its token cap, and
    evaluate.utterance_wer(text, T*'s transcript) <= ROUNDTRIP_MAX_WER (0.25, normalized).
The GPU lock is the caller's (the CLI holds the parakeet GPU lock for the whole job).

The keep rule and term agreement always compare T*'s transcript with the LABEL, never with the TTS input (tts_text,
synthetic/pronounce.py).

finalize (CPU): joins hygiene, TTS and round-trip records (a run rendered before the pronunciation layer, without
pronunciation.json, had the label as its TTS input) into
- clips.jsonl: the kept clips whose sentence the CURRENT hygiene keeps with the same label, sorted by id: {id, audio_filepath, duration, sha256, label, tts_text, domain, engine, voice,
  terms}; the clip source of the TS/TSn arms (mixing.ClipPool), and clips.meta.json (counts, hours, input hashes,
  the split and hygiene provenance, provisional flag, and split_stale: hygiene did not record running under the
  current split, so mixing.ClipPool refuses the run for training);
- report.json: per domain and per engine, generated vs kept clips and hours (retained hours per generated hour),
  the reasons, requested vs kept term coverage (distinct targeted terms requested by prompts, present in hygiene-
  kept sentences, present in rendered clips, present in kept clips) and term-level agreement: for every term
  occurrence of the split in a rendered clip's text, whether T* reproduced it (score_domain's aligned-occurrence
  definition), over all rendered clips, kept and rejected; also split into acronym and other terms
  (pronounce.is_acronym_term, under the lexicon that rendered the run: finalize refuses another one) per domain
  ("term_agreement_by_kind") and pooled over domains (total "term_agreement", "term_agreement_by_kind"), with the
  run's pronunciation summary ("pronunciation": clips whose tts_text differs from the label);
- listening/<domain>/{kept,rejected}/: LISTEN_KEPT kept clips and LISTEN_REJECTED rejected clips whose rejection
  is the teacher's disagreement (WER > 0.25), seeded, copied with index.tsv (id, engine, voice, WER, text, tts_text,
  T*).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from synthetic import layout
from synthetic.textgen import _rng, read_jsonl, write_jsonl

BLOCK = 1024


def tts_records(run_dir: Path) -> list[dict]:
    rows = []
    for alias in layout.TTS_ENGINES:
        for p in sorted((run_dir / "tts" / alias).glob("*.jsonl")):
            rows += read_jsonl(p)
    rows.sort(key=lambda r: r["id"])
    return rows


def blocks(ids: list[str]) -> list[list[str]]:
    return [ids[i:i + BLOCK] for i in range(0, len(ids), BLOCK)]


def decide(text: str, hyp: str | None, hit_cap: bool, duration: float) -> tuple[bool, str | None, float | None]:
    """(kept, reason, WER) of one clip."""
    import evaluate
    if not layout.MIN_SECONDS <= duration <= layout.MAX_SECONDS:
        return False, "duration", None
    if hyp is None:
        return False, "no_transcript", None
    wer = evaluate.utterance_wer(text, hyp)
    wer = None if not np.isfinite(wer) else float(wer)
    if hit_cap:
        return False, "hit_cap", wer
    if wer is None or wer > layout.ROUNDTRIP_MAX_WER:
        return False, "wer_gt_0.25", wer
    return True, None, wer


def load_audio(path: str) -> np.ndarray:
    import soundfile as sf
    a, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != layout.SAMPLE_RATE or a.shape[1] != 1:
        raise ValueError(f"{path}: expected 16 kHz mono")
    return a[:, 0]


def transcribe_chunk(run_dir: Path, engine, budget_minutes: float, expected_total: int | None = None, log=print) -> dict:
    import time
    recs = tts_records(run_dir)
    if expected_total is not None and len(recs) != expected_total:
        raise RuntimeError(f"{len(recs)} rendered clips, expected {expected_total} (finish TTS first)")
    ids = [r["id"] for r in recs]
    by_id = {r["id"]: r for r in recs}
    t0 = time.monotonic()
    todo = [(k, b) for k, b in enumerate(blocks(ids)) if not (run_dir / "roundtrip" / f"{k:05d}.jsonl").exists()]
    if todo:
        engine.load()
    done = 0
    for k, block in todo:
        if (time.monotonic() - t0) / 60 > budget_minutes:
            break
        tb = time.monotonic()
        clips = [by_id[i] for i in block]
        inrange = [c for c in clips if layout.MIN_SECONDS <= c["duration"] <= layout.MAX_SECONDS]
        order = sorted(inrange, key=lambda c: (-c["duration"], c["id"]))
        hyps = {}
        for unit in engine.plan([c["duration"] for c in order]):
            outs = engine.run([load_audio(order[i]["audio_filepath"]) for i in unit])
            for i, (text, hit) in zip(unit, outs):
                hyps[order[i]["id"]] = (text, bool(hit))
        rows = []
        for c in clips:
            hyp, hit = hyps.get(c["id"], (None, False))
            kept, reason, wer = decide(c["label"], hyp, hit, c["duration"])
            rows.append({"id": c["id"], "hyp": hyp, "hit_cap": hit, "wer": wer, "kept": kept, "reason": reason})
        write_jsonl(run_dir / "roundtrip" / f"{k:05d}.jsonl", rows)
        done += 1
        secs = sum(c["duration"] for c in clips)
        try:
            import torch
            peak = f", peak reserved {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB" if torch.cuda.is_available() else ""
        except Exception:  # noqa: BLE001
            peak = ""
        log(f"roundtrip block {k}{peak}: {len(rows)} clips ({secs / 3600:.3f} h), kept {sum(r['kept'] for r in rows)}, "
            f"{time.monotonic() - tb:.0f} s ({secs / max(time.monotonic() - tb, 1e-9):.0f} x real time)")
    remaining = sum(1 for k, _ in enumerate(blocks(ids)) if not (run_dir / "roundtrip" / f"{k:05d}.jsonl").exists())
    return {"blocks_done": done, "blocks_remaining": remaining, "engine": engine.info() if todo else None}


def term_agreement(text: str, hyp: str | None, split) -> list[tuple[str, str, bool]]:
    """(phrase, domain, reproduced) for every split-term occurrence in a clip's text."""
    import evaluate
    import score_domain
    ref = evaluate.normalize(text)
    h = evaluate.normalize(hyp or "")
    out = []
    for s, e, ok in score_domain.term_occurrences(ref, h, split.matcher()):
        phrase = tuple(ref.split()[s:e])
        domain, _ = split.classify(phrase)
        out.append((" ".join(phrase), domain, bool(ok)))
    return out


def _sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _rate(c: Counter) -> dict:
    return {"occurrences": c["occurrences"], "reproduced": c["reproduced"],
            "rate": c["reproduced"] / c["occurrences"] if c["occurrences"] else None}


def finalize(run_dir: Path, split, listen: bool = True, lexicon=None) -> dict:
    from synthetic import pronounce
    lexicon = pronounce.Lexicon.load() if lexicon is None else lexicon
    pron_path = run_dir / "pronunciation.json"
    pron = json.loads(pron_path.read_text()) if pron_path.exists() else None
    if pron is not None and pron["lexicon"] != lexicon.record():
        raise RuntimeError(f"the lexicon differs from the one that rendered {run_dir.name} ({pron_path})")
    acronym = {e.phrase for e in split.entries if pronounce.is_acronym_term(e.term, e.phrase, lexicon)}
    prompts = read_jsonl(run_dir / "prompts.jsonl")
    sentences = read_jsonl(run_dir / "sentences.jsonl")
    hyg = json.loads((run_dir / "hygiene.json").read_text())
    plan = read_jsonl(run_dir / "tts_plan.jsonl")
    tts = {r["id"]: r for r in tts_records(run_dir)}
    rt = {}
    for p in sorted((run_dir / "roundtrip").glob("*.jsonl")):
        for r in read_jsonl(p):
            rt[r["id"]] = r
    missing = [r["id"] for r in plan if r["id"] not in tts or r["id"] not in rt]
    if missing:
        raise RuntimeError(f"{len(missing)} planned clips lack TTS or round-trip records (first {missing[0]})")
    # A run rendered before the pronunciation layer (no pronunciation.json; the 2026-10-23 pilot) gave the engine the
    # label itself, so its TTS input is the label. A run with the layer must record tts_text for every clip.
    no_text = [i for i, t in tts.items() if "tts_text" not in t]
    if no_text and pron is not None:
        raise RuntimeError(f"{len(no_text)} TTS records lack tts_text (first {no_text[0]})")
    tts = {i: t if "tts_text" in t else t | {"tts_text": t["label"]} for i, t in tts.items()}
    targeted_phrase = {e.phrase: e for e in split.entries}
    norm_of = {}
    import evaluate

    def requested_set(rows):
        out = defaultdict(set)
        for r in rows:
            for t in r["requested"]:
                ph = tuple(evaluate.normalize(t).split())
                out[r["domain"]].add(" ".join(ph))
        return out

    def present_set(rows, text_key):
        out = defaultdict(set)
        for r in rows:
            for o in split.occurrences(r[text_key]):
                if o["side"] == "targeted":
                    out[o["domain"]].add(o["phrase"])
        return out

    sent_by_id = {s["id"]: s for s in sentences}
    # Only clips whose sentence is kept by the CURRENT hygiene, with the same label, are published (a hygiene rerun,
    # e.g. once eval-ext / eval-stem exist, can only remove clips; sentences it newly keeps have no audio).
    current = [r for r in plan if r["id"] in sent_by_id and sent_by_id[r["id"]]["label"] == r["label"]]
    current_ids = {r["id"] for r in current}
    excluded = [r["id"] for r in plan if r["id"] not in current_ids]
    not_rendered = sorted(set(sent_by_id) - {r["id"] for r in plan})
    plan = current
    clips, per = [], defaultdict(lambda: Counter())
    seconds = defaultdict(float)
    agree = defaultdict(lambda: Counter())
    changed = 0
    for r in plan:
        t, x = tts[r["id"]], rt[r["id"]]
        d, e = r["domain"], r["engine"]
        seconds[(d, e, "rendered")] += t["duration"]
        per[(d, e)]["rendered"] += 1
        per[(d, e)][x["reason"] or "kept"] += 1
        changed += t["tts_text"] != r["label"]
        for phrase, dom, ok in term_agreement(r["label"], x["hyp"], split):   # by the TERM's domain
            ph = tuple(phrase.split())
            side = targeted_phrase[ph].side
            kind = "acronym" if ph in acronym else "other"
            for k in ("all", "kept" if x["kept"] else "rejected"):
                for key in ((dom, k, side), (dom, k, side, kind)):
                    agree[key]["occurrences"] += 1
                    agree[key]["reproduced"] += int(ok)
        if x["kept"]:
            seconds[(d, e, "kept")] += t["duration"]
            clips.append({"id": r["id"], "audio_filepath": t["audio_filepath"], "duration": t["duration"],
                          "sha256": t["sha256"], "label": r["label"], "tts_text": t["tts_text"], "domain": d,
                          "engine": e, "voice": r["voice"],
                          "terms": [o["phrase"] for o in split.occurrences(r["label"])], "wer": x["wer"]})
    clips.sort(key=lambda c: c["id"])
    req = requested_set(prompts)
    present_sent = present_set(sentences, "label")
    rendered_rows = [{"label": r["label"]} | {"domain": r["domain"]} for r in plan]
    present_rendered = present_set(rendered_rows, "label")
    present_kept = present_set(clips, "label")
    n_targeted = {d: len(split.targeted(d)) for d in layout.DOMAINS}
    report = {"run": run_dir.name, "domains": {}, "engines": {}, "hygiene": {k: hyg[k] for k in ("generated", "kept", "reasons")},
              "excluded_by_current_hygiene": len(excluded), "kept_by_hygiene_not_rendered": len(not_rendered),
              "pronunciation": {"legacy_no_layer": pron is None,
                                "clips_changed": changed, "clips": len(plan), "lexicon": lexicon.record(),
                                "acronym_terms": len(acronym),
                                "substitutions_by_rule": (pron or {}).get("substitutions_by_rule"),
                                "default_rule_tokens": (pron or {}).get("default_rule_tokens")},
              "notes": {"domains": "clip counts, hours and term coverage by the prompt's domain; term agreement by "
                                   "the term's domain"}}
    for d in layout.DOMAINS:
        rend = sum(seconds[(d, e, "rendered")] for e in layout.TTS_ENGINES)
        kept = sum(seconds[(d, e, "kept")] for e in layout.TTS_ENGINES)
        counts = Counter()
        for e in layout.TTS_ENGINES:
            counts.update(per[(d, e)])
        report["domains"][d] = {
            "prompts": sum(1 for p in prompts if p["domain"] == d),
            "hygiene_kept": sum(1 for s in sentences if s["domain"] == d),
            "rendered": counts["rendered"], "kept": counts["kept"],
            "reasons": {k: v for k, v in counts.items() if k not in ("rendered", "kept")},
            "rendered_hours": rend / 3600, "kept_hours": kept / 3600,
            "retained_hours_per_generated_hour": kept / rend if rend else None,
            "term_coverage": {"targeted_terms": n_targeted[d], "requested": len(req[d]),
                              "in_hygiene_kept_sentences": len(present_sent[d]),
                              "in_rendered_clips": len(present_rendered[d]), "in_kept_clips": len(present_kept[d]),
                              "requested_but_not_kept": len(req[d] - present_kept[d])},
            "term_agreement": {f"{k}/{side}": {"occurrences": agree[(d, k, side)]["occurrences"],
                                               "reproduced": agree[(d, k, side)]["reproduced"],
                                               "rate": (agree[(d, k, side)]["reproduced"] / agree[(d, k, side)]["occurrences"])
                                               if agree[(d, k, side)]["occurrences"] else None}
                               for k in ("all", "kept", "rejected") for side in ("targeted", "heldout")},
            "term_agreement_by_kind": {f"{k}/{side}": {kind: _rate(agree[(d, k, side, kind)])
                                                       for kind in ("acronym", "other")}
                                       for k in ("all", "kept", "rejected") for side in ("targeted", "heldout")},
        }
    for e in layout.TTS_ENGINES:
        rend = sum(seconds[(d, e, "rendered")] for d in layout.DOMAINS)
        kept = sum(seconds[(d, e, "kept")] for d in layout.DOMAINS)
        n = sum(per[(d, e)]["rendered"] for d in layout.DOMAINS)
        k = sum(per[(d, e)]["kept"] for d in layout.DOMAINS)
        report["engines"][e] = {"rendered": n, "kept": k, "rendered_hours": rend / 3600, "kept_hours": kept / 3600,
                                "retained_hours_per_generated_hour": kept / rend if rend else None,
                                "voices_used": len({r["voice"] for r in plan if r["engine"] == e})}
    rend = sum(v for (d, e, s), v in seconds.items() if s == "rendered")
    kept = sum(v for (d, e, s), v in seconds.items() if s == "kept")
    report["total"] = {"prompts": len(prompts), "hygiene_kept": len(sentences), "rendered": len(plan),
                       "kept": len(clips), "rendered_hours": rend / 3600, "kept_hours": kept / 3600,
                       "retained_hours_per_generated_hour": kept / rend if rend else None,
                       "kept_clips_per_prompt": len(clips) / len(prompts) if prompts else None,
                       "kept_hours_per_prompt": kept / 3600 / len(prompts) if prompts else None}

    def pooled(*key):
        c = Counter()
        for d in layout.DOMAINS:
            c.update(agree[(d, *key)])
        return _rate(c)
    report["total"]["term_agreement"] = {f"{k}/{side}": pooled(k, side)
                                         for k in ("all", "kept", "rejected") for side in ("targeted", "heldout")}
    report["total"]["term_agreement_by_kind"] = {f"{k}/{side}": {kind: pooled(k, side, kind)
                                                                 for kind in ("acronym", "other")}
                                                 for k in ("all", "kept", "rejected") for side in ("targeted", "heldout")}
    write_jsonl(run_dir / "clips.jsonl", clips)
    meta = {"clips": len(clips), "hours": kept / 3600, "clips_sha256": _sha256(run_dir / "clips.jsonl"),
            "split_sha256": _sha256(layout.split_path()), "hygiene_heldout": hyg["heldout"],
            "hygiene_sha256": _sha256(run_dir / "hygiene.json"), "sentences_sha256": _sha256(run_dir / "sentences.jsonl"),
            "excluded_by_current_hygiene": excluded[:1000],
            "hygiene_split_sha256": hyg.get("split_sha256"),
            "split_stale": hyg.get("split_sha256") != _sha256(layout.split_path()),
            "provisional": bool(hyg["heldout"]["provisional"]), "roundtrip_max_wer": layout.ROUNDTRIP_MAX_WER}
    (run_dir / "clips.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    (run_dir / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    if listen:
        listening(run_dir, plan, tts, rt)
    return report


def listening(run_dir: Path, plan: list[dict], tts: dict, rt: dict) -> None:
    out = run_dir / "listening"
    if out.exists():
        shutil.rmtree(out)
    for d in layout.DOMAINS:
        rows = [r for r in plan if r["domain"] == d]
        kept = sorted((r for r in rows if rt[r["id"]]["kept"]), key=lambda r: r["id"])
        rej = sorted((r for r in rows if rt[r["id"]]["reason"] == "wer_gt_0.25"), key=lambda r: r["id"])
        for name, pool, n in (("kept", kept, layout.LISTEN_KEPT), ("rejected", rej, layout.LISTEN_REJECTED)):
            pick = list(pool)
            _rng(layout.SEED, "listen", d, name).shuffle(pick)
            pick = sorted(pick[:n], key=lambda r: r["id"])
            target = out / d / name
            target.mkdir(parents=True, exist_ok=True)
            with open(target / "index.tsv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f, delimiter="\t")
                w.writerow(["file", "id", "engine", "voice", "wer", "text", "tts_text", "teacher"])
                for i, r in enumerate(pick):
                    fn = f"{i:02d}-{r['id']}.flac"
                    shutil.copyfile(tts[r["id"]]["audio_filepath"], target / fn)
                    x = rt[r["id"]]
                    w.writerow([fn, r["id"], r["engine"], r["voice"],
                                "" if x["wer"] is None else f"{x['wer']:.3f}", r["label"], tts[r["id"]]["tts_text"],
                                x["hyp"] or ""])
