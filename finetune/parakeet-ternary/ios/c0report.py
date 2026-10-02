"""Summarize parakeet-bench C0 records (JSON lines from the Mac) against clips.json and the B0 traces.

NixOS (the parent experiment's scorer: Whisper-normalized WER, evaluate.score / wer_summary):
  CUDA_VISIBLE_DEVICES= ../python ios/c0report.py RESULTS.jsonl [--plan SUMMARY.json] [--out SUMMARY.json]

Reports, over timed (non-warm-up) records: WER of the natural clips vs the LibriSpeech transcripts and vs B0's
transcripts (traces.json), exact token-sequence agreement with B0, per-bucket medians over clips of each clip's
median stage times (Mac timings are informational: the Mac is shared), first-call (warm-up 0) totals, and
physical model-call counts against the B0 trace's logical steps. Replay records add C0's argmax agreement
with the trace's token and duration decisions.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

STAGES = ("preprocess", "encoder", "decode", "decoder_model", "joint_model", "total")


def per_clip_median(records: list[dict], key) -> dict[str, float]:
    by_clip: dict[str, list[float]] = {}
    for r in records:
        by_clip.setdefault(r["clip"], []).append(key(r))
    return {c: statistics.median(v) for c, v in by_clip.items()}


def main() -> None:
    import traces as tracemod

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("results")
    parser.add_argument("--plan", help="parakeet-bench plan summary JSON to embed")
    parser.add_argument("--out")
    args = parser.parse_args()
    lines = [json.loads(l) for l in Path(args.results).read_text().splitlines() if l.strip()]
    load = next((l for l in lines if l.get("record") == "load"), {})
    end = next((l for l in lines if l.get("record") == "end"), {})
    calls = [l for l in lines if "result" in l]
    timed = [c for c in calls if not c["warmup"]]
    first = [c for c in calls if c["warmup"] and c["rep"] == 0]
    clips = {c["id"]: c for c in json.loads((HERE / "clips.json").read_text())["clips"]}
    trace = {t["id"]: t for t in json.loads((HERE / "traces.json").read_text())["clips"]}
    mode = timed[0]["mode"] if timed else None

    out: dict = {"results_file": Path(args.results).name, "mode": mode, "compute_units": load.get("compute_units"),
                 "preprocessor_units": load.get("preprocessor_units"), "os": load.get("os"),
                 "load_ms": load.get("load_ms"), "timed_records": len(timed), "clips": len({c["clip"] for c in timed}),
                 "warmups": load.get("warmups"), "timed_per_clip": load.get("timed"),
                 "phys_footprint_mb": {"after_load": load.get("phys_footprint_mb_after_load"),
                                       "end": end.get("phys_footprint_mb"), "peak": end.get("phys_footprint_peak_mb")}}

    natural = [c for c in timed if c["kind"] == "natural"]
    one = {c["clip"]: c for c in natural}  # last timed record per clip (decoding is deterministic per clip)
    if mode == "free" and one:
        refs = [(clips[i]["transcript"], r["text"]) for i, r in one.items()]
        vs_b0 = [(trace[i]["text"], r["text"]) for i, r in one.items()]
        out["wer_vs_reference"] = tracemod.wer_block(refs)
        out["wer_vs_b0"] = tracemod.wer_block(vs_b0)
        out["b0_wer_vs_reference_same_clips"] = tracemod.wer_block([(clips[i]["transcript"], trace[i]["text"]) for i in one])
        out["per_bucket_wer_vs_reference"] = {
            b: tracemod.wer_block([(clips[i]["transcript"], r["text"]) for i, r in one.items() if clips[i]["bucket"] == b])
            for b in sorted({clips[i]["bucket"] for i in one})}
        same = [i for i, r in one.items() if r["result"]["tokens"] == trace[i]["tokens"]]
        out["token_sequence_equal_to_b0"] = {"clips": len(same), "of": len(one)}
        out["texts_differing_from_b0"] = [{"clip": i, "b0": trace[i]["text"], "c0": r["text"]}
                                          for i, r in one.items() if r["result"]["tokens"] != trace[i]["tokens"]]
        stable = all(len({json.dumps(c["result"]["tokens"]) for c in calls if c["clip"] == i}) == 1 for i in one)
        out["tokens_identical_across_reps"] = stable

    buckets: dict = {}
    for b in sorted({c["bucket"] for c in timed}):
        recs = [c for c in timed if c["bucket"] == b]
        entry = {"clips": len({c["clip"] for c in recs})}
        for s in STAGES:
            meds = per_clip_median(recs, lambda r: r["result"]["times_ms"][s])
            entry[f"{s}_ms_median"] = round(statistics.median(meds.values()), 3)
        firsts = [c["result"]["times_ms"]["total"] for c in first if c["bucket"] == b]
        if firsts:
            entry["first_call_total_ms_median"] = round(statistics.median(firsts), 3)
        dec = [c["result"]["decoder_calls"] for c in recs]
        joi = [c["result"]["joint_calls"] for c in recs]
        entry["decoder_calls_mean"] = round(statistics.mean(dec), 2)
        entry["joint_calls_mean"] = round(statistics.mean(joi), 2)
        entry["decoder_ms_per_call"] = round(sum(c["result"]["times_ms"]["decoder_model"] for c in recs) / max(sum(dec), 1), 4)
        entry["joint_ms_per_call"] = round(sum(c["result"]["times_ms"]["joint_model"] for c in recs) / max(sum(joi), 1), 4)
        buckets[str(b)] = entry
    out["per_bucket"] = buckets

    per_clip = {}
    for i, r in {c["clip"]: c for c in timed}.items():
        t = trace[i]
        per_clip[i] = {"joint_calls": r["result"]["joint_calls"], "decoder_calls": r["result"]["decoder_calls"],
                       "b0_steps": t["steps"], "b0_decoder_runs": 1 + sum(t["pred_updated"]),
                       "encoder_length": r["result"]["encoder_length"], "b0_frames": t["num_frames"],
                       "effective_frames": r["result"]["effective_frames"]}
    out["calls"] = {
        "preprocessor_per_call": 1, "encoder_per_call": 1,
        "joint_calls_total": sum(v["joint_calls"] for v in per_clip.values()),
        "b0_steps_total": sum(v["b0_steps"] for v in per_clip.values()),
        "decoder_calls_total": sum(v["decoder_calls"] for v in per_clip.values()),
        "b0_prediction_runs_total": sum(v["b0_decoder_runs"] for v in per_clip.values()),
        "encoder_length_minus_b0_frames": sorted({v["encoder_length"] - v["b0_frames"] for v in per_clip.values()}),
        "effective_minus_b0_frames": sorted({v["effective_frames"] - v["b0_frames"] for v in per_clip.values()}),
    }
    if mode == "replay":
        steps = sum(c["result"]["replay_steps"] for c in one.values()) if one else 0
        allc = {c["clip"]: c for c in timed}
        tot = sum(c["result"]["replay_steps"] for c in allc.values())
        out["replay"] = {
            "steps": tot,
            "token_agree": sum(c["result"]["replay_token_agree"] for c in allc.values()) / max(tot, 1),
            "duration_agree": sum(c["result"]["replay_duration_agree"] for c in allc.values()) / max(tot, 1),
            "natural_steps": steps}
    if args.plan:
        out["compute_plan"] = json.loads(Path(args.plan).read_text())
    text = json.dumps(out, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
