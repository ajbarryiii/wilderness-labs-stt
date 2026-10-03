"""Correctness gates 4 and 5 of the Core ML arms on the Mac (DESIGN.md "Correctness gates"); gate 3 is refcache.py.

  cd ios && python -m mil.gates encoder --model mp2 --arm C4 --variant fixed --units cpuAndNeuralEngine
  cd ios && python -m mil.gates decoder --model mp2 --units cpuAndNeuralEngine
  cd ios && python -m mil.gates g0 --units cpuAndNeuralEngine
  cd ios && python -m mil.gates summary

Inputs: the cached FP32 reference (refcache.py; "fp16s" = FP16-rounded row scales, the weights the arms
encode) and the compiled models under <artifacts>/arms/<model>/. Error definitions (DESIGN.md): rel =
||a - r||_2 / max(||r||_2, tau sqrt(n)), abs = max|a - r| / max(RMS(r), tau), tau = 1e-3 for FP16 arms.

encoder: every clip runs in every bucket it fits (fixed: 15 s only; multi: functions b2..b15; enum: the
four enumerated shapes), mel = the reference features zero-padded to F_b, mel_length = M (valid mel frames).
- gate 4 (encoder): valid frames [1024, E] vs the fp16s reference: rel <= 2e-2 and abs <= 0.25, every clip
  and bucket, finite, encoder_length == E.
- gate 5: valid frames of each bucket vs the same arm's 15 s output, gate-4 ceilings; gated on the
  boundary, silence and impulse clips (natural clips reported).
- gate 4 (TDT heads): forced replay of traces.json through our Decoder and JointLogits models, fed with the
  arm's encoder output from the clip's own bucket (fixed: 15 s), vs the reference's replay (fp16s): token
  logits (incl. blank), duration logits, LSTM h and c each rel <= 2e-2 per clip; per head, decision
  agreement >= 99.5% of decisive steps (reference top-1 margin > 4 x the arm's max logit error at that
  step) with >= 50% of steps decisive, pooled over clips; all finite.
decoder: the same heads gate with the reference's own encoder output (isolates Decoder/JointLogits).
g0: G0 vs C0's Encoder on the same mel (15 s window), gate-4 encoder ceilings (G0 is C0's tensors in our
graph; its FP32 reference would be C0's dequantized weights, which C0 does not ship for linear_pos).
Results: ios/results/gates/<model>-<arm>-<variant>-<units>.json (per clip and bucket; no weights or audio).

These are the revision 2-5 gates (DESIGN.md), kept for the record and as diagnostics. The revision 7 gates
(4a/4b, eligibility) are mil/gates7.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "mil"

import numpy as np

from . import IOS, refcache  # noqa: E402

GATES = IOS / "results" / "gates"
TAU = 1e-3
REL_MAX, ABS_MAX = 2e-2, 0.25
AGREE_MIN, DECISIVE_MIN = 0.995, 0.5
BUCKETS = {2: 201, 4: 401, 8: 801, 15: 1501}
GATE5_KINDS = ("boundary", "silence", "impulse")
N_DUR = 5


def purge_cache() -> None:
    """Empty this job's own Core ML (e5rt) cache directory after a model is released (device-specialized copies,
    GBs each). Only the directory named by WP3_COREML_CACHE (set by mil/job.sh: ~/Library/Caches/wp3py, used by
    no other process) is touched; nothing happens without it. Shared caches are never purged (review finding 6)."""
    import os
    import shutil

    cache = os.environ.get("WP3_COREML_CACHE")
    if cache and Path(cache).name == "wp3py" and Path(cache).is_dir():
        shutil.rmtree(cache, ignore_errors=True)


def units_of(name: str):
    import coremltools as ct

    return {"cpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE, "cpuOnly": ct.ComputeUnit.CPU_ONLY,
            "cpuAndGPU": ct.ComputeUnit.CPU_AND_GPU}[name]


def arms_root() -> Path:
    import artifacts

    return artifacts.root() / "arms"


def clips_manifest() -> list[dict]:
    import clips as clipmod

    return clipmod.load_manifest()["clips"]


def padded(features: np.ndarray, frames: int) -> np.ndarray:
    out = np.zeros((1, 128, frames), dtype=np.float32)
    out[0, :, :features.shape[1]] = features
    return out


def _peak_mb() -> float:
    import resource

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / 2 ** 20 if sys.platform == "darwin" else r / 1024, 1)


# --- replay through Core ML ---------------------------------------------------------------------------------

def _config():
    import randomweights as rw
    import reference

    return reference.Config.from_model_config(rw.load_stats()["model_config"])


def coreml_replay(dec, joint, enc_valid: np.ndarray, trace_rec: dict):
    """Forced replay of a traces.json record with Core ML Decoder + JointLogits: (logits [S, 1030], h, c [S, 2, 640])."""
    import torch

    import reference
    import traces as tracemod

    trace = tracemod.record_to_trace(trace_rec)
    cfg = _config()
    e = enc_valid.shape[1]
    if trace.num_frames != e:
        raise ValueError(f"trace has {trace.num_frames} frames, encoder {e}")

    def predict(token, state):
        h, c = state
        out = dec.predict({"targets": np.array([[token]], dtype=np.int32), "target_length": np.array([1], dtype=np.int32),
                           "h_in": h.numpy().astype(np.float32), "c_in": c.numpy().astype(np.float32)})
        g = torch.from_numpy(np.asarray(out["decoder"], dtype=np.float32).reshape(1, 1, -1))
        return g, (torch.from_numpy(np.asarray(out["h_out"], dtype=np.float32)),
                   torch.from_numpy(np.asarray(out["c_out"], dtype=np.float32)))

    def joint_step(f, g):
        out = joint.predict({"encoder_step": f.numpy().reshape(1, -1, 1).astype(np.float32),
                             "decoder_step": g.numpy().reshape(1, -1, 1).astype(np.float32)})
        return torch.from_numpy(np.concatenate([np.asarray(out["token_logits"], dtype=np.float32).reshape(1, -1),
                                                np.asarray(out["duration_logits"], dtype=np.float32).reshape(1, -1)], 1))

    def decide(i, t, tok, dur):
        if i >= len(trace) or trace.frame[i] != t:
            raise ValueError(f"replay diverged from the trace at step {i}")
        return trace.token[i], trace.duration[i]

    state = (torch.zeros(cfg.pred_rnn_layers, 1, cfg.pred_hidden), torch.zeros(cfg.pred_rnn_layers, 1, cfg.pred_hidden))
    enc_proj = torch.from_numpy(np.ascontiguousarray(enc_valid.T[None]))
    replayed, steps = reference.run_steps(cfg, enc_proj, e, state, predict, joint_step, decide, record=True)
    if len(replayed) != len(trace):
        raise ValueError("replay length differs from the trace")
    return steps.logits.numpy(), steps.h.numpy(), steps.c.numpy()


def head_metrics(a_logits, a_h, a_c, r_logits, r_h, r_c) -> dict:
    """Per-clip errors per head and the per-step decision bookkeeping (pooled later)."""
    out = {"finite": bool(np.isfinite(a_logits).all() and np.isfinite(a_h).all() and np.isfinite(a_c).all()),
           "steps": int(r_logits.shape[0])}
    heads = {"token": (a_logits[:, :-N_DUR], r_logits[:, :-N_DUR]), "duration": (a_logits[:, -N_DUR:], r_logits[:, -N_DUR:])}
    for name, (a, r) in heads.items():
        rel, ab = refcache.errors(a, r, TAU)
        srt = np.sort(r, axis=1)
        margin = srt[:, -1] - srt[:, -2]
        err = np.abs(a.astype(np.float64) - r).max(axis=1)
        decisive = margin > 4 * err
        agree = a.argmax(axis=1) == r.argmax(axis=1)
        out[name] = {"rel": rel, "abs": ab, "decisive": int(decisive.sum()), "agree_decisive": int((agree & decisive).sum()),
                     "agree_all": int(agree.sum()), "margin_quantiles": np.quantile(margin, [0.05, 0.5, 0.95]).tolist() if len(margin) else []}
    for name, a, r in (("h", a_h, r_h), ("c", a_c, r_c)):
        rel, ab = refcache.errors(a, r, TAU)
        out[name] = {"rel": rel, "abs": ab}
    return out


def pool_heads(per_clip: dict) -> dict:
    total = sum(v["steps"] for v in per_clip.values())
    res = {"clips": len(per_clip), "steps": total, "all_finite": all(v["finite"] for v in per_clip.values())}
    passed = res["all_finite"] and total > 0
    for q in ("token", "duration", "h", "c"):
        rels = [v[q]["rel"] for v in per_clip.values()]
        res[q] = {"rel_max": float(max(rels)), "rel_median": float(np.median(rels)),
                  "clips_over_rel_ceiling": [k for k, v in per_clip.items() if v[q]["rel"] > REL_MAX]}
        passed &= not res[q]["clips_over_rel_ceiling"]
    for head in ("token", "duration"):
        dec = sum(v[head]["decisive"] for v in per_clip.values())
        agr = sum(v[head]["agree_decisive"] for v in per_clip.values())
        res[head]["decisive_fraction"] = dec / total if total else 0.0
        res[head]["agreement_on_decisive"] = agr / dec if dec else 0.0
        res[head]["agreement_all_steps"] = sum(v[head]["agree_all"] for v in per_clip.values()) / total if total else 0.0
        res[head]["degenerate"] = res[head]["decisive_fraction"] < DECISIVE_MIN
        passed &= (not res[head]["degenerate"]) and res[head]["agreement_on_decisive"] >= AGREE_MIN
    res["pass"] = bool(passed)
    return res


def load_decoders(model: str, units: str):
    import coremltools as ct

    base = arms_root() / model / "decoder"
    cu = units_of(units)
    return (ct.models.CompiledMLModel(str(base / "Decoder.mlmodelc"), compute_units=cu),
            ct.models.CompiledMLModel(str(base / "JointLogits.mlmodelc"), compute_units=cu))


def traces_by_id() -> dict:
    return {r["id"]: r for r in json.loads((IOS / "traces.json").read_text())["clips"]}


# --- encoder gate ---------------------------------------------------------------------------------------

def encoder_gate(args) -> dict:
    import coremltools as ct

    t0 = time.time()
    clips = clips_manifest()
    if args.ids:
        clips = [c for c in clips if c["id"] in args.ids.split(",")]
    model_dir = arms_root() / args.model / args.arm
    path = model_dir / f"{args.variant}.mlmodelc"
    cu = units_of(args.units)
    buckets = [15] if args.variant == "fixed" else [15, 8, 4, 2]
    full15: dict[str, np.ndarray] = {}
    own: dict[str, np.ndarray] = {}
    rows: dict[str, dict] = {}
    load_s = {}
    for b in buckets:
        fn = {"fixed": None, "multi": f"b{b}", "enum": None}[args.variant]
        if args.variant != "enum" or b == 15:
            tl = time.time()
            m = ct.models.CompiledMLModel(str(path), compute_units=cu, function_name=fn)
            load_s[fn or "main"] = round(time.time() - tl, 2)
        for clip in clips:
            if clip["bucket"] > b:
                continue
            ref = refcache.load_clip(args.model, clip["id"])
            e = clip["encoder_frames"]
            pred = m.predict({"mel": padded(ref["features"], BUCKETS[b]), "mel_length": np.array([ref["mel_length"]], dtype=np.int32)})
            enc = np.asarray(pred["encoder"], dtype=np.float32)[0]
            length = int(np.asarray(pred["encoder_length"]).reshape(-1)[0])
            valid = enc[:, :e]
            rel, ab = refcache.errors(valid, ref["fp16s_enc"], TAU)
            row = {"bucket": b, "kind": clip["kind"], "encoder_length_ok": length == e, "finite_valid": bool(np.isfinite(valid).all()),
                   "finite_all": bool(np.isfinite(enc).all()), "rel": rel, "abs": ab}
            row["pass"] = row["encoder_length_ok"] and row["finite_valid"] and rel <= REL_MAX and ab <= ABS_MAX
            if b == 15:
                full15[clip["id"]] = valid
            else:
                r5, a5 = refcache.errors(valid, full15[clip["id"]], TAU)
                row["vs_15s"] = {"rel": r5, "abs": a5, "pass": r5 <= REL_MAX and a5 <= ABS_MAX and row["finite_valid"]}
            if b == clip["bucket"] or (args.variant == "fixed"):
                own[clip["id"]] = valid
            rows[f"{clip['id']}@{b}"] = row
        if args.variant != "enum" or b == buckets[-1]:
            del m
            purge_cache()
        print(f"bucket {b} done at {time.time() - t0:.0f}s", flush=True)
    # TDT heads through our decoder/joint, fed with the arm's own-bucket encoder output
    dec, joint = load_decoders(args.model, args.units)
    traces = traces_by_id()
    heads = {}
    for clip in clips:
        ref = refcache.load_clip(args.model, clip["id"])
        a = coreml_replay(dec, joint, own[clip["id"]], traces[clip["id"]])
        heads[clip["id"]] = head_metrics(*a, ref["fp16s_logits"], ref["fp16s_h"], ref["fp16s_c"])
    del dec, joint
    purge_cache()
    g4 = [r for r in rows.values()]
    g5 = [r for r in rows.values() if "vs_15s" in r]
    g5_gated = [r for r in g5 if r["kind"] in GATE5_KINDS]
    summary = {
        "gate4_encoder": {"pass": all(r["pass"] for r in g4), "cases": len(g4),
                          "rel_max": max(r["rel"] for r in g4), "abs_max": max(r["abs"] for r in g4),
                          "rel_median": float(np.median([r["rel"] for r in g4])),
                          "failing_cases": sum(1 for r in rows.values() if not r["pass"]),
                          "failures": [k for k, r in rows.items() if not r["pass"]][:40],
                          "per_bucket": {b: {"cases": sum(1 for r in g4 if r["bucket"] == b),
                                             "rel_max": max((r["rel"] for r in g4 if r["bucket"] == b), default=None),
                                             "abs_max": max((r["abs"] for r in g4 if r["bucket"] == b), default=None),
                                             "pass": all(r["pass"] for r in g4 if r["bucket"] == b)} for b in buckets}},
        "gate5_buckets_vs_15s": ({"pass": all(r["vs_15s"]["pass"] for r in g5_gated), "gated_cases": len(g5_gated),
                                  "gated_rel_max": max((r["vs_15s"]["rel"] for r in g5_gated), default=None),
                                  "gated_abs_max": max((r["vs_15s"]["abs"] for r in g5_gated), default=None),
                                  "all_cases": len(g5), "all_rel_max": max((r["vs_15s"]["rel"] for r in g5), default=None),
                                  "all_abs_max": max((r["vs_15s"]["abs"] for r in g5), default=None),
                                  "all_pass": all(r["vs_15s"]["pass"] for r in g5)}
                                 if g5 else {"pass": None, "note": "fixed 15 s window: no smaller buckets"}),
        "gate4_heads": pool_heads(heads),
    }
    doc = {"model": args.model, "arm": args.arm, "variant": args.variant, "units": args.units,
           "thresholds": {"rel": REL_MAX, "abs": ABS_MAX, "tau": TAU, "agreement": AGREE_MIN, "decisive_min": DECISIVE_MIN},
           "summary": summary, "load_s": load_s, "seconds": round(time.time() - t0, 1), "peak_rss_mb": _peak_mb(),
           "encoder_cases": rows, "heads_per_clip": heads,
           "heads_input": "arm encoder output from the clip's own bucket" + (" (15 s window)" if args.variant == "fixed" else "")}
    write(doc, f"{args.model}-{args.arm}-{args.variant}-{args.units}.json")
    print(json.dumps({"summary": {k: {kk: vv for kk, vv in v.items() if kk not in ("failures", "per_bucket")}
                                  for k, v in summary.items()}, "seconds": doc["seconds"], "peak_rss_mb": doc["peak_rss_mb"]},
                     default=str), flush=True)
    return doc


def write(doc: dict, name: str) -> None:
    GATES.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=0, default=float) + "\n"
    (GATES / name).write_text(text)


def decoder_gate(args) -> dict:
    t0 = time.time()
    clips = clips_manifest()
    dec, joint = load_decoders(args.model, args.units)
    traces = traces_by_id()
    heads = {}
    for clip in clips:
        ref = refcache.load_clip(args.model, clip["id"])
        a = coreml_replay(dec, joint, ref["fp16s_enc"], traces[clip["id"]])
        heads[clip["id"]] = head_metrics(*a, ref["fp16s_logits"], ref["fp16s_h"], ref["fp16s_c"])
    doc = {"model": args.model, "arm": "decoder", "units": args.units, "heads_input": "the reference's own (fp16s) encoder output",
           "summary": {"gate4_heads": pool_heads(heads)}, "heads_per_clip": heads, "seconds": round(time.time() - t0, 1),
           "peak_rss_mb": _peak_mb()}
    write(doc, f"{args.model}-decoder-{args.units}.json")
    print(json.dumps(doc["summary"], default=str), flush=True)
    return doc


def g0_gate(args) -> dict:
    import coremltools as ct

    from .weights import default_path

    t0 = time.time()
    clips = clips_manifest()
    cu = units_of(args.units)
    c0 = ct.models.CompiledMLModel(str(default_path("c0")), compute_units=cu)
    c0_out = {}
    for clip in clips:
        ref = refcache.load_clip("mp2", clip["id"])
        pred = c0.predict({"mel": padded(ref["features"], 1501), "mel_length": np.array([ref["mel_length"]], dtype=np.int32)})
        c0_out[clip["id"]] = (np.asarray(pred["encoder"], dtype=np.float32)[0], int(np.asarray(pred["encoder_length"]).reshape(-1)[0]))
    del c0
    purge_cache()
    g0 = ct.models.CompiledMLModel(str(arms_root() / "c0" / "G0" / "fixed.mlmodelc"), compute_units=cu)
    rows = {}
    for clip in clips:
        ref = refcache.load_clip("mp2", clip["id"])
        pred = g0.predict({"mel": padded(ref["features"], 1501), "mel_length": np.array([ref["mel_length"]], dtype=np.int32)})
        enc = np.asarray(pred["encoder"], dtype=np.float32)[0]
        length = int(np.asarray(pred["encoder_length"]).reshape(-1)[0])
        e = clip["encoder_frames"]
        c0_enc, c0_len = c0_out[clip["id"]]
        rel, ab = refcache.errors(enc[:, :e], c0_enc[:, :e], TAU)
        rows[clip["id"]] = {"kind": clip["kind"], "rel": rel, "abs": ab, "g0_length": length, "c0_length": c0_len,
                            "expected_length": e, "finite": bool(np.isfinite(enc[:, :e]).all()),
                            "pass": rel <= REL_MAX and ab <= ABS_MAX and length == e and bool(np.isfinite(enc[:, :e]).all())}
    del g0
    vals = list(rows.values())
    doc = {"arm": "G0", "variant": "fixed", "units": args.units,
           "comparison": "G0 (C0's tensors, our plain graph, iOS17) vs C0's Encoder.mlmodelc on the same mel and mel_length "
                         "(reference features padded to 1501, mel_length = N // 160), valid frames, gate-4 encoder ceilings",
           "summary": {"pass": all(r["pass"] for r in vals), "clips": len(vals), "rel_max": max(r["rel"] for r in vals),
                       "abs_max": max(r["abs"] for r in vals), "rel_median": float(np.median([r["rel"] for r in vals])),
                       "c0_length_differs": sum(1 for r in vals if r["c0_length"] != r["expected_length"])},
           "per_clip": rows, "seconds": round(time.time() - t0, 1), "peak_rss_mb": _peak_mb()}
    write(doc, f"c0-G0-fixed-{args.units}.json")
    print(json.dumps(doc["summary"]), flush=True)
    return doc


def summary(args) -> None:
    rows = []
    for p in sorted(GATES.glob("*.json")):
        if p.name.endswith("-gate3.json") or p.name in ("summary.json",):
            continue
        d = json.loads(p.read_text())
        s = d["summary"]
        row = {"file": p.name, "model": d.get("model", "c0"), "arm": d["arm"], "variant": d.get("variant"), "units": d["units"]}
        if "gate4_encoder" in s:
            row["g4_enc"] = s["gate4_encoder"]["pass"]
            row["g4_enc_rel_max"] = s["gate4_encoder"]["rel_max"]
            row["g4_enc_abs_max"] = s["gate4_encoder"]["abs_max"]
            row["g5"] = s["gate5_buckets_vs_15s"].get("pass")
            row["g5_rel_max"] = s["gate5_buckets_vs_15s"].get("gated_rel_max")
        if "gate4_heads" in s:
            h = s["gate4_heads"]
            row["g4_heads"] = h["pass"]
            for q in ("token", "duration", "h", "c"):
                row[f"{q}_rel_max"] = h[q]["rel_max"]
            for head in ("token", "duration"):
                row[f"{head}_agree"] = h[head]["agreement_on_decisive"]
                row[f"{head}_decisive"] = h[head]["decisive_fraction"]
        if d["arm"] == "G0":
            row["g4_enc_vs_c0"] = s["pass"]
            row["g4_enc_rel_max"] = s["rel_max"]
            row["g4_enc_abs_max"] = s["abs_max"]
        rows.append(row)
    (GATES / "summary.json").write_text(json.dumps(rows, indent=1) + "\n")
    cols = ["model", "arm", "variant", "units", "g4_enc", "g4_enc_rel_max", "g4_enc_abs_max", "g5", "g5_rel_max",
            "g4_heads", "token_rel_max", "duration_rel_max", "h_rel_max", "c_rel_max", "token_agree", "token_decisive",
            "duration_agree", "duration_decisive"]

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3g}"
        return "" if v is None else str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    (GATES / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("encoder")
    p.add_argument("--model", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--variant", required=True, choices=("fixed", "multi", "enum"))
    p.add_argument("--units", default="cpuAndNeuralEngine")
    p.add_argument("--ids")
    p = sub.add_parser("decoder")
    p.add_argument("--model", required=True)
    p.add_argument("--units", default="cpuAndNeuralEngine")
    p = sub.add_parser("g0")
    p.add_argument("--units", default="cpuAndNeuralEngine")
    sub.add_parser("summary")
    args = parser.parse_args()
    {"encoder": encoder_gate, "decoder": decoder_gate, "g0": g0_gate, "summary": summary}[args.cmd](args)


if __name__ == "__main__":
    main()
