"""Summarize parakeet-bench records of any arm (JSON lines from the Mac); optionally pair them with a C0 baseline.

NixOS (the parent experiment's scorer: Whisper-normalized WER, evaluate.score / wer_summary):
  CUDA_VISIBLE_DEVICES= ../python ios/armreport.py RESULTS.jsonl [--baseline C0.jsonl] [--plan SUMMARY.json]
      [--ref NATIVE_REF_DIR] [--out SUMMARY.json] [--min-timed 10] [--smoke] [--bootstrap 2000]

(Renamed from c0report.py in WP4; the C0-only summaries in results/smoke/ were made by that version.)

Repetition completeness is enforced. Every clip in the load record's clip_ids, or every clip seen if the run
predates that field, must have exactly the timed calls with rep = warmups .. warmups + timed - 1, and its
first warm-up record (rep 0) when warm-ups ran. Otherwise the report fails.

The summary is labelled "baseline-eligible" only if the run is complete and has at least --min-timed timed
calls per clip (DESIGN.md "Repetition and statistics": 10 on the Mac, 5 on the phone). Anything else needs
--smoke and is labelled "smoke".

Reports, over the timed records only:
- WER of the natural clips vs the LibriSpeech transcripts and vs B0's transcripts (traces.json), and exact
  token-sequence agreement with B0. With --ref, also agreement with that model's FP32 reference greedy
  tokens (native.py reference).
- Per bucket, the two estimands for every stage present (preprocess, encoder, preprojection, decode, total and
  each decode-loop component): typical latency, the median over clips of each clip's median; and tail latency,
  the Harrell-Davis p95 over the bucket's pooled calls.
- First-call (warm-up rep 0) totals.
- Physical model-call counts per component against the B0 trace's logical steps.

With --baseline (a C0 run of the same clips; DESIGN.md "Repetition and statistics": comparisons with C0 are
paired), per bucket and over all clips:
- the typical-latency ratio = median over clips of (arm clip median / C0 clip median);
- the p95 ratio = HD p95 of the arm's pooled calls / HD p95 of C0's;
- each with a percentile bootstrap 95% interval that resamples clips with replacement and keeps each clip's arm
  and C0 calls together (cluster = clip; one session, so no session level), for total, encoder and decode.

Untimed "diagnostic" records (parakeet-bench --diag-dir) are summarized separately. Mac timings are
informational: the Mac is shared.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

STAGES = ("preprocess", "encoder", "preprojection", "decode", "total", "decoder_model", "joint_model",
          "fused_model", "native_predict", "native_joint", "native_fused")
PAIRED_STAGES = ("total", "encoder", "decode")


def harrell_davis(values, p: float = 0.95) -> float:
    """Harrell-Davis quantile estimate: sum_i w_i x_(i), w_i = I_{i/n}(a, b) - I_{(i-1)/n}(a, b),
    a = p (n + 1), b = (1 - p)(n + 1), I the regularized incomplete beta function."""
    from scipy.stats import beta

    x = np.sort(np.asarray(values, dtype=np.float64))
    n = x.size
    if n == 0:
        raise ValueError("no values")
    if n == 1:
        return float(x[0])
    a, b = p * (n + 1), (1 - p) * (n + 1)
    cdf = beta.cdf(np.arange(n + 1) / n, a, b)
    return float(np.dot(np.diff(cdf), x))


def check_complete(load: dict, calls: list[dict]) -> dict:
    """Repetition completeness per clip; raises ValueError listing every problem."""
    warmups, timed = load.get("warmups"), load.get("timed")
    if warmups is None or timed is None:
        raise ValueError("load record lacks warmups/timed")
    clip_ids = load.get("clip_ids") or sorted({c["clip"] for c in calls})
    problems = []
    for cid in clip_ids:
        mine = [c for c in calls if c["clip"] == cid]
        reps = sorted(c["rep"] for c in mine if not c["warmup"])
        if reps != list(range(warmups, warmups + timed)):
            problems.append(f"{cid}: timed reps {reps}, expected {warmups}..{warmups + timed - 1}")
        if warmups and not any(c["warmup"] and c["rep"] == 0 for c in mine):
            problems.append(f"{cid}: no first warm-up record")
    extra = sorted({c["clip"] for c in calls} - set(clip_ids))
    if extra:
        problems.append(f"records for clips not in the run: {extra}")
    if problems:
        raise ValueError("incomplete run: " + "; ".join(problems[:20]))
    return {"clips": len(clip_ids), "warmups": warmups, "timed_per_clip": timed,
            "clip_ids_recorded": bool(load.get("clip_ids"))}


def per_clip_median(records: list[dict], key) -> dict[str, float]:
    by_clip: dict[str, list[float]] = {}
    for r in records:
        by_clip.setdefault(r["clip"], []).append(key(r))
    return {c: statistics.median(v) for c, v in by_clip.items()}


def load_run(path: str) -> tuple[dict, dict, list[dict], list[dict], dict]:
    lines = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    load = next((l for l in lines if l.get("record") == "load"), {})
    end = next((l for l in lines if l.get("record") == "end"), {})
    diagnostics = [l for l in lines if l.get("record") == "diagnostic"]
    calls = [l for l in lines if "result" in l]
    return load, end, diagnostics, calls, check_complete(load, calls)


def stage_values(records: list[dict], stage: str) -> dict[str, list[float]]:
    by_clip: dict[str, list[float]] = {}
    for r in records:
        by_clip.setdefault(r["clip"], []).append(r["result"]["times_ms"][stage])
    return by_clip


def paired(arm: list[dict], base: list[dict], stage: str, n_boot: int, seed: int = 0) -> dict:
    """Typical-latency ratio (median over clips of per-clip median ratios) and HD-p95 ratio, arm / baseline, with
    percentile bootstrap 95% intervals resampling clips (each clip's arm and baseline calls kept together)."""
    a, b = stage_values(arm, stage), stage_values(base, stage)
    ids = sorted(set(a) & set(b))
    if not ids:
        return {"clips": 0}

    def stats(sample: list[str]) -> tuple[float, float]:
        ratios = [statistics.median(a[i]) / statistics.median(b[i]) for i in sample]
        pa = [v for i in sample for v in a[i]]
        pb = [v for i in sample for v in b[i]]
        return statistics.median(ratios), harrell_davis(pa) / harrell_davis(pb)

    typ, p95 = stats(ids)
    rng = np.random.default_rng(seed)
    boots = np.array([stats([ids[k] for k in rng.integers(0, len(ids), len(ids))]) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return {"clips": len(ids), "typical_ratio": round(typ, 4), "typical_ratio_ci95": [round(lo[0], 4), round(hi[0], 4)],
            "p95_ratio": round(p95, 4), "p95_ratio_ci95": [round(lo[1], 4), round(hi[1], 4)],
            "arm_typical_ms": round(statistics.median(statistics.median(a[i]) for i in ids), 3),
            "baseline_typical_ms": round(statistics.median(statistics.median(b[i]) for i in ids), 3)}


def main() -> None:
    import traces as tracemod

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("results")
    parser.add_argument("--plan", help="parakeet-bench plan summary JSON to embed")
    parser.add_argument("--out", help="summary JSON: under ios/results/ (text, no audio or weights) or the artifact area")
    parser.add_argument("--min-timed", type=int, default=10, help="timed calls per clip for a baseline (Mac 10, phone 5)")
    parser.add_argument("--smoke", action="store_true", help="accept fewer timed calls and label the summary smoke")
    parser.add_argument("--baseline", help="a C0 run of the same clips (JSON lines) for paired comparisons")
    parser.add_argument("--ref", help="native.py reference output directory (ref/<id>.npz greedy tokens) of the arm's model")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    load, end, diagnostics, calls, completeness = load_run(args.results)
    timed = [c for c in calls if not c["warmup"]]
    first = [c for c in calls if c["warmup"] and c["rep"] == 0]
    eligible = completeness["timed_per_clip"] >= args.min_timed and completeness["warmups"] >= 3
    if not eligible and not args.smoke:
        raise SystemExit(f"{completeness['timed_per_clip']} timed calls / {completeness['warmups']} warm-ups per clip "
                         f"(< {args.min_timed} / 3): pass --smoke")
    clips = {c["id"]: c for c in json.loads((HERE / "clips.json").read_text())["clips"]}
    trace = {t["id"]: t for t in json.loads((HERE / "traces.json").read_text())["clips"]}
    mode = load.get("mode") or (timed[0]["mode"] if timed else None)

    out: dict = {"label": "baseline-eligible" if eligible else "smoke",
                 "label_note": None if eligible else (
                     f"{completeness['timed_per_clip']} timed call(s) per clip, fewer than the {args.min_timed} "
                     "DESIGN.md prescribes: a functional check, not a comparison baseline"),
                 "results_file": Path(args.results).name, "arm": load.get("arm"), "arm_spec": load.get("arm_spec"),
                 "mode": mode, "compute_units": load.get("compute_units"),
                 "preprocessor_units": load.get("preprocessor_units"), "os": load.get("os"),
                 "load_ms": load.get("load_ms"),
                 "load_cache_evidence": load.get("load_cache_evidence", "none: MLModel load wall time only; "
                                                 "no Instruments Core ML trace"),
                 "completeness": completeness, "timed_records": len(timed),
                 "phys_footprint_mb": {"after_load": load.get("phys_footprint_mb_after_load"),
                                       "end": end.get("phys_footprint_mb"), "peak": end.get("phys_footprint_peak_mb")}}

    natural = [c for c in timed if c["kind"] == "natural"]
    one = {c["clip"]: c for c in natural}  # one timed record per clip for scoring
    if mode == "free" and one:
        refs = [(clips[i]["transcript"], r["text"]) for i, r in one.items()]
        out["wer_vs_reference"] = tracemod.wer_block(refs)
        out["wer_vs_b0"] = tracemod.wer_block([(trace[i]["text"], r["text"]) for i, r in one.items()])
        out["b0_wer_vs_reference_same_clips"] = tracemod.wer_block([(clips[i]["transcript"], trace[i]["text"]) for i in one])
        out["per_bucket_wer_vs_reference"] = {
            b: tracemod.wer_block([(clips[i]["transcript"], r["text"]) for i, r in one.items() if clips[i]["bucket"] == b])
            for b in sorted({clips[i]["bucket"] for i in one})}
        same = [i for i, r in one.items() if r["result"]["tokens"] == trace[i]["tokens"]]
        out["token_sequence_equal_to_b0"] = {"clips": len(same), "of": len(one)}
        out["texts_differing_from_b0"] = [{"clip": i, "b0": trace[i]["text"], "arm": r["text"]}
                                          for i, r in one.items() if r["result"]["tokens"] != trace[i]["tokens"]]
        if args.ref:
            refs_np = {i: np.load(Path(args.ref) / "ref" / f"{i}.npz")["greedy_tokens"].tolist() for i in one}
            out["token_sequence_equal_to_model_reference"] = {
                "reference": args.ref, "clips": sum(r["result"]["tokens"] == refs_np[i] for i, r in one.items()), "of": len(one)}
        out["tokens_identical_across_reps"] = all(
            len({json.dumps(c["result"]["tokens"]) for c in calls if c["clip"] == i}) == 1 for i in one)

    buckets: dict = {}
    for b in sorted({c["bucket"] for c in timed}):
        recs = [c for c in timed if c["bucket"] == b]
        entry = {"clips": len({c["clip"] for c in recs}), "pooled_calls": len(recs)}
        for s in [x for x in STAGES if all(x in r["result"]["times_ms"] for r in recs)]:
            meds = per_clip_median(recs, lambda r: r["result"]["times_ms"][s])
            entry[f"{s}_ms_typical"] = round(statistics.median(meds.values()), 3)
            entry[f"{s}_ms_p95_hd"] = round(harrell_davis([r["result"]["times_ms"][s] for r in recs]), 3)
        firsts = [c["result"]["times_ms"]["total"] for c in first if c["bucket"] == b]
        if firsts:
            entry["first_call_total_ms_median"] = round(statistics.median(firsts), 3)
        components = sorted({k for c in recs for k in c["result"].get("physical_calls", {})})
        for comp in components:
            n = [c["result"]["physical_calls"].get(comp, 0) for c in recs]
            entry[f"{comp}_calls_mean"] = round(statistics.mean(n), 2)
            entry[f"{comp}_ms_per_call"] = round(sum(c["result"]["times_ms"].get(comp, 0) for c in recs) / max(sum(n), 1), 4)
        buckets[str(b)] = entry
    out["per_bucket"] = buckets
    out["estimands"] = ("typical = median over clips of each clip's median; p95_hd = Harrell-Davis p95 over the "
                        "bucket's pooled timed calls")

    per_clip = {}
    for i, r in {c["clip"]: c for c in timed}.items():
        t = trace[i]
        res = r["result"]
        per_clip[i] = {"joint_calls": res.get("logical_joint_steps") or res["joint_calls"],
                       "decoder_calls": res.get("logical_predictions") or res["decoder_calls"],
                       "b0_steps": t["steps"], "b0_decoder_runs": 1 + sum(t["pred_updated"]),
                       "encoder_length": r["result"]["encoder_length"], "b0_frames": t["num_frames"],
                       "effective_frames": r["result"]["effective_frames"]}
    out["calls"] = {
        "preprocessor_per_call": sorted({c["result"]["preprocessor_calls"] for c in timed}),
        "encoder_per_call": sorted({c["result"]["encoder_calls"] for c in timed}),
        "physical_totals": {comp: sum(c["result"].get("physical_calls", {}).get(comp, 0) for c in {x["clip"]: x for x in timed}.values())
                            for comp in sorted({k for c in timed for k in c["result"].get("physical_calls", {})})},
        "joint_calls_total": sum(v["joint_calls"] for v in per_clip.values()),
        "b0_steps_total": sum(v["b0_steps"] for v in per_clip.values()),
        "decoder_calls_total": sum(v["decoder_calls"] for v in per_clip.values()),
        "b0_prediction_runs_total": sum(v["b0_decoder_runs"] for v in per_clip.values()),
        "encoder_length_minus_b0_frames": sorted({v["encoder_length"] - v["b0_frames"] for v in per_clip.values()}),
        "effective_minus_b0_frames": sorted({v["effective_frames"] - v["b0_frames"] for v in per_clip.values()}),
    }
    if mode == "replay":
        bad = [i for i, v in per_clip.items() if v["joint_calls"] != v["b0_steps"] or v["decoder_calls"] != v["b0_decoder_runs"]]
        out["calls"]["replay_calls_equal_trace"] = not bad
    if diagnostics:
        rep = [d["replay"] for d in diagnostics if "replay" in d]
        out["diagnostics"] = {
            "records": len(diagnostics), "untimed": True,
            "logits": diagnostics[0].get("logits") or "unavailable (argmax-only joint outputs)",
            "arrays": [d["arrays"]["file"] for d in diagnostics]}
        if rep:
            steps = sum(r["steps"] for r in rep)
            out["diagnostics"]["replay"] = {
                "steps": steps, "token_agree": sum(r["token_agree"] for r in rep) / max(steps, 1),
                "duration_agree": sum(r["duration_agree"] for r in rep) / max(steps, 1),
                "all_steps_executed": all(r["steps"] == r["trace_steps"] for r in rep)}
    if args.baseline:
        bload, _, _, bcalls, bcomp = load_run(args.baseline)
        # pairing provenance (review WP4/5 finding 8): same session (pairing block), manifest, mode, clips, protocol
        problems = []
        if bload.get("arm") != "C0":
            problems.append(f"baseline arm is {bload.get('arm')!r}, not C0")
        for key in ("pairing", "clips_json_sha256", "clip_ids", "warmups", "timed", "mode"):
            if load.get(key) is None or load.get(key) != bload.get(key):
                problems.append(f"{key} missing or different between the arm and the baseline run")
        pa, pb = load.get("pairing") or {}, bload.get("pairing") or {}
        if not pa:
            problems.append("the arm run was not paired with C0 in one process (no pairing block)")
        elif "session" in pa or "session" in pb:
            evidence = "pairing session id"  # equality is checked with the whole block above
        elif Path(args.results).resolve().parent == Path(args.baseline).resolve().parent:
            evidence = "same run directory (records written before session ids, WP5)"
        else:
            problems.append("no pairing session id and the files are not from one run directory")
        if sorted({c['clip'] for c in timed}) != sorted({c['clip'] for c in bcalls if not c['warmup']}):
            problems.append("clip coverage differs")
        if problems:
            raise SystemExit("unpaired runs: " + "; ".join(problems))
        btimed = [c for c in bcalls if not c["warmup"]]
        out["paired_vs_baseline"] = {
            "baseline_file": Path(args.baseline).name, "baseline_arm": bload.get("arm"), "baseline_completeness": bcomp,
            "pairing_evidence": evidence,
            "method": "per-clip ratios arm/baseline; typical = median over clips; p95 = HD p95 ratio of pooled calls; "
                      f"percentile bootstrap ({args.bootstrap} resamples of clips, pairs kept together), seed 0",
            "per_bucket": {str(b): {s: paired([c for c in timed if c["bucket"] == b], [c for c in btimed if c["bucket"] == b],
                                              s, args.bootstrap) for s in PAIRED_STAGES}
                           for b in sorted({c["bucket"] for c in timed})},
            "all_clips": {s: paired(timed, btimed, s, args.bootstrap) for s in PAIRED_STAGES}}
    if args.plan:
        out["compute_plan"] = json.loads(Path(args.plan).read_text())
    text = json.dumps(out, indent=1)
    if args.out:
        dest = Path(args.out).resolve()
        if not dest.is_relative_to((HERE / "results").resolve()):
            import artifacts
            dest = artifacts.check(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
