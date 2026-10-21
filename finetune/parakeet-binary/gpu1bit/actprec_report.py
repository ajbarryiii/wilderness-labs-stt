"""Activation-precision measurements and reading around the full-dev arms (plans/activation-precision.md
"Measurements" 1 and 3, "Reading", R5 and R7). Analysis only: the arms' code is gpu1bit/actprec/parts.py
(hashed into the xN identities). This file stays outside those identities so it can change without
invalidating recorded arms. Every GPU process takes one gpu lock before CUDA and yields first to the
sel-20261020b timing rerun (parts.timing_has_priority).

  encoder-check --run R --model M1|MB1
      Encoder output of x2 and x1 against x3 on the 512-utterance dev sample: finite, length equal
      to x3, relative Frobenius on x3's valid frames. Each arm is built by parts.build (x3 = the
      deployed stack), one at a time, through its pipeline graph cache. The frozen identities of
      the run's three arms must match the current code.
  a3
      Descriptive per-layer A3 ratios (plans/gpu-kernel.md A3 limits: 1.5x the worse of M1's
      R-bf16x3 / R-expanded INT8x3) on the 264 layers x 15 fixture utterances, for:
      B's int8_mma:x2 / :x1 (pinned B/wip/parts, the variant's own linear / conv), and M1's
      ExpandedProjection('triton_col_tuned') with components 2 / 1 on the same derived weights.
  report --run R
      The reading for each xN arm against its x3 arm: R5 structural validity (finite, decoding,
      encoder lengths equal x3), mean WER within 0.02 points, no set worse by more than 0.10
      points, token mismatches at most 0.5% pooled and 1.0% per set (R7), per-set mismatch
      rates, empty hypotheses; encoder-check and A3 summaries.
      Inputs are identity-bound (review c1). Summaries must name the right arm and run, cover the full
      subset completely and carry the identity check_identity verifies. Comparison artifacts must carry
      both identities and complete counts. Encoder checks must carry the model's three identities and
      512 utterances. Anything else is refused.
      An x3 reference must pass parts.require_reference (complete, verified, R5-valid, cross-check
      passed); otherwise the model's xN arms are NOT CLASSIFIED (R2).
  timing --roster <timing roster>
      P-paired (encoder and full scope) and P-clips/P-memory of the six arms, read only from artifacts
      the orchestrator accepted and that are still bound to the roster's identities, recognition run and
      references. Reported: x2 and x1 against x3 per model and MB1-xN against M1-xN (median paired
      change and count lower, for energy, latency and average power), absolute medians, and per clip.
Outputs: .../gpu1bit/H/actprec/
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from gpu1bit.shared import env, fulldev  # noqa: E402
from gpu1bit.actprec import parts  # noqa: E402

ARMS = {"M1": {3: "M1-x3", 2: "M1-x2", 1: "M1-x1"},
        "MB1": {3: "MB1-int8_mma-x3", 2: "MB1-int8_mma-x2", 1: "MB1-int8_mma-x1"}}
MEAN_POINTS, SET_WORSE_POINTS, POOLED_MISMATCH, SET_MISMATCH = 0.02, 0.10, 0.005, 0.010
A12_ENCODER_REL = 1e-3
EPS = 1e-9   # points / rates computed from float percentages: a value exactly at a limit is within it


def out_dir() -> Path:
    return env.artifacts("actprec")


def wait_turn() -> float:
    waited = 0.0
    while (reason := parts.timing_has_priority()) is not None:
        if waited == 0.0:
            print(json.dumps({"waiting": reason}), flush=True)
        time.sleep(30)
        waited += 30
    return waited


def _quantiles(values: list[float]) -> dict:
    v = sorted(values)
    if not v:
        return {}
    pick = lambda q: v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]  # noqa: E731
    return {"min": v[0], "p50": pick(0.5), "p95": pick(0.95), "p99": pick(0.99), "max": v[-1], "n": len(v)}


# --- encoder check ---------------------------------------------------------------------------------

def encoder_check(args) -> None:
    import torch
    env.require_shared_wrapper()
    parts.install()
    arms = ARMS[args.model]
    parts.require_reference(args.run, arms[2])   # R2 gate (review c2): no lock, no build against a bad x3
    identities = {n: fulldev.check_identity(args.run, arm, "full")["hash"] for n, arm in arms.items()}
    records = [(name, i, rec) for name in fulldev.DEV_SETS for i, rec in fulldev.subset_records(name, "sample")]
    yielded = wait_turn()
    import evaluate as ev
    from nemo.utils import logging
    logging.setLevel(logging.ERROR)
    with env.gpu_lock(f"H:actprec encoder-check {args.model}") as lock:
        ref, rows = {}, {}
        for n in (3, 2, 1):
            with torch.inference_mode():
                model, info = parts.build(args.model, n)
            try:
                for name, i, rec in records:
                    env.check_hold(lock)
                    clip, _ = ev.pad_short(ev.load_audio(rec["audio_filepath"]))
                    audio = torch.from_numpy(clip).unsqueeze(0).cuda()
                    length = torch.tensor([len(clip)], dtype=torch.long).cuda()
                    with torch.inference_mode(), ev.strict_fp32("cuda"), ev.inference_settings(model):
                        enc, enc_len = model.forward(input_signal=audio, input_signal_length=length)
                        t_n = int(enc_len[0])
                        finite = bool(torch.isfinite(enc).all())
                        if n == 3:
                            ref[rec["id"]] = (enc[..., :t_n].double().cpu(), t_n)
                            rows[rec["id"]] = {"set": name, "i": i, "id": rec["id"], "x3_len": t_n, "x3_finite": finite}
                        else:
                            r, t = ref[rec["id"]]
                            y = enc[..., :t].double().cpu()
                            rel = ((y - r).norm() / r.norm()).item() if t_n == t else None
                            rows[rec["id"]][f"x{n}"] = {"finite": finite, "length_equal": t_n == t, "rel_frobenius_valid": rel}
            finally:
                parts.close(model)
                del model
                gc.collect()
                torch.cuda.empty_cache()
        held = time.monotonic() - lock["acquired"]
    rows = list(rows.values())
    summary = {}
    for n in (2, 1):
        rels = [r[f"x{n}"]["rel_frobenius_valid"] for r in rows if r[f"x{n}"]["rel_frobenius_valid"] is not None]
        summary[f"x{n}"] = {"utterances": len(rows), "not_finite": [r["id"] for r in rows if not r[f"x{n}"]["finite"]],
                            "length_differences": [r["id"] for r in rows if not r[f"x{n}"]["length_equal"]],
                            "rel_frobenius_valid": _quantiles(rels),
                            "above_a12_bound_1e-3": sum(v > A12_ENCODER_REL for v in rels),
                            "per_set_max": {s: max(r[f"x{n}"]["rel_frobenius_valid"] or 0 for r in rows if r["set"] == s)
                                            for s in fulldev.DEV_SETS}}
    doc = {"run": args.run, "model": args.model, "arms": arms, "identities": identities, "subset": "sample",
           "x3_not_finite": [r["id"] for r in rows if not r["x3_finite"]], "summary": summary,
           "lock_held_s": held, "yielded_s": yielded, "wrapper": env.wrapper(), "rows": rows}
    path = out_dir() / f"encoder-check-{args.run}-{args.model}.json"
    path.write_text(json.dumps(doc) + "\n")
    print(json.dumps({"path": str(path), "summary": summary}, indent=1))


# --- A3 descriptive ratios -------------------------------------------------------------------------

def a3(args) -> None:
    import torch
    from gpu1bit.shared import api, fixtures, m1, registry
    env.require_shared_wrapper()
    parts.install()
    loaded = parts.load_parts()
    selection = list(range(15))
    yielded = wait_turn()
    thresholds = fixtures.m1_thresholds()
    idx = fixtures.index()
    weights = fixtures.mb1_layers()
    ops = ("B int8_mma:x2", "B int8_mma:x1", "M1 ExpandedProjection x2", "M1 ExpandedProjection x1")
    rows = []
    with env.gpu_lock("H:actprec a3") as lock, torch.inference_mode():
        torch.backends.cuda.matmul.allow_tf32 = False
        impls = {n: registry.get(f"int8_mma:x{n}") for n in (2, 1)}
        b_layers = {n: {name: impls[n].prepare(w["codes"], w["scale"], w["bias"], w["K"], "cuda") for name, w in weights.items()}
                    for n in (2, 1)}
        m1_ops = {name: m1.LayerOps(w["codes"], w["scale"], w["bias"], w["K"]) for name, w in weights.items()}
        for i in selection:
            entry = idx["utterances"][i]
            inputs = fixtures.utterance_inputs(i)
            limits = thresholds[entry["file"]]
            for name, w in weights.items():
                env.check_hold(lock)
                x, kind = inputs[name], w["kind"]
                ref = api.reference(kind, x, w["codes"], w["scale"], w["bias"], w["K"])
                conv = kind == "conv"
                o = m1_ops[name]
                outs = {"B int8_mma:x2": api.call(b_layers[2][name], kind, x),
                        "B int8_mma:x1": api.call(b_layers[1][name], kind, x),
                        "M1 ExpandedProjection x2": o.projection(x, o.conv_module if conv else o.module, conv, 2),
                        "M1 ExpandedProjection x1": o.projection(x, o.conv_module if conv else o.module, conv, 1)}
                for op, y in outs.items():
                    e = api.errors(kind, x, y, ref, w["scale"])
                    lim = limits[name]
                    rows.append({"utterance": entry["file"], "layer": name, "M": entry["layers"][name]["M"], "op": op, **e,
                                 "limits": lim, "within": api.within(e, lim),
                                 "ratio": {m: (e[m] / lim[m]) if lim[m] > 0 else (0.0 if e[m] == 0 else float("inf"))
                                           for m in ("relative_frobenius", "elementwise_guard")}})
            del inputs
            gc.collect()
        held = time.monotonic() - lock["acquired"]
    summary = {}
    for op in ops:
        r = [x for x in rows if x["op"] == op]
        summary[op] = {"evaluations": len(r), "within_a3": sum(x["within"] for x in r), "not_finite": sum(not x["finite"] for x in r),
                       "ratio_relative_frobenius": _quantiles([x["ratio"]["relative_frobenius"] for x in r]),
                       "ratio_elementwise_guard": _quantiles([x["ratio"]["elementwise_guard"] for x in r]),
                       "relative_frobenius": _quantiles([x["relative_frobenius"] for x in r])}
    agreement = {}
    for n in (2, 1):
        b = {(x["utterance"], x["layer"]): x["relative_frobenius"] for x in rows if x["op"] == f"B int8_mma:x{n}"}
        m = {(x["utterance"], x["layer"]): x["relative_frobenius"] for x in rows if x["op"] == f"M1 ExpandedProjection x{n}"}
        agreement[f"x{n}"] = _quantiles([b[k] / m[k] for k in b if m[k] > 0])
    doc = {"selection": selection, "ops": list(ops), "wip_loaded": loaded, "pinned": parts.PINNED,
           "limits": "plans/gpu-kernel.md A3: 1.5 x max(R-bf16x3, R-expanded INT8x3) per layer and utterance (fixtures.m1_thresholds)",
           "summary": summary, "b_over_m1_relative_frobenius": agreement, "lock_held_s": held, "yielded_s": yielded,
           "wrapper": env.wrapper(), "rows": rows}
    path = out_dir() / "a3-parts.json"
    path.write_text(json.dumps(doc) + "\n")
    print(json.dumps({"path": str(path), "summary": summary, "b_over_m1": agreement}, indent=1))


# --- report ----------------------------------------------------------------------------------------

def _full_counts() -> dict:
    return {n: len(fulldev.subset_records(n, "full")) for n in fulldev.DEV_SETS}


def verified_summary(run: str, arm: str, counts: dict) -> tuple[dict, str]:
    """(summary, verified identity hash) of a complete full-subset arm, or ValueError (R4)."""
    s = json.loads((fulldev.arm_dir(run, arm, "full") / "summary.json").read_text())
    h = fulldev.check_identity(run, arm, "full")["hash"]
    problems = [k for k, ok in (("run", s.get("run") == run), ("arm", s.get("arm") == arm), ("subset", s.get("subset") == "full"),
                                ("complete", s.get("complete") is True), ("utterances", s.get("utterances") == counts),
                                ("identity", s.get("identity") == h)) if not ok]
    if problems:
        raise ValueError(f"{run}/{arm}: summary refused ({problems})")
    return s, h


def verified_compare(run: str, arm: str, ref: str, ids: dict, counts: dict) -> dict:
    """The comparison artifact of arm vs ref, bound to both verified identities and complete, or ValueError."""
    c = json.loads((fulldev.ROOT / run / "compare" / f"{arm}--vs--{ref}@full.json").read_text())
    sets = c.get("sets") or {}
    problems = [k for k, ok in (("run", c.get("run") == run), ("arm", c.get("arm") == arm), ("ref", c.get("ref") == ref),
                                ("subset", c.get("subset") == "full"),
                                ("identities", c.get("identities") == {"arm": ids[arm], "ref": ids[ref]}),
                                ("sets", {n: (sets.get(n) or {}).get("utterances") for n in fulldev.DEV_SETS} == counts),
                                ("pooled", (c.get("pooled") or {}).get("utterances") == sum(counts.values()))) if not ok]
    if problems:
        raise ValueError(f"{run}: comparison {arm} vs {ref} refused ({problems})")
    return c


def verified_encoder_check(run: str, model: str, ids: dict) -> dict:
    """The 512-sample encoder check document of a model, bound to its three verified identities and
    carrying its x3 finiteness record, or ValueError."""
    doc = json.loads((out_dir() / f"encoder-check-{run}-{model}.json").read_text())
    want = {str(n): ids[a] for n, a in ARMS[model].items()}
    summary = doc.get("summary") or {}
    problems = [k for k, ok in (("run", doc.get("run") == run), ("model", doc.get("model") == model),
                                ("subset", doc.get("subset") == "sample"), ("identities", doc.get("identities") == want),
                                ("x3_not_finite", isinstance(doc.get("x3_not_finite"), list)),
                                ("utterances", all((summary.get(f"x{n}") or {}).get("utterances") == 512 for n in (2, 1)))) if not ok]
    if problems:
        raise ValueError(f"{run}: encoder check of {model} refused ({problems})")
    return doc


def reading(arm_sum: dict, ref_sum: dict, cmp: dict, enc: dict | None) -> dict:
    sets = fulldev.DEV_SETS
    delta = {s: arm_sum["dev_wer_percent"][s] - ref_sum["dev_wer_percent"][s] for s in sets}
    mean_delta = arm_sum["dev_mean_wer_percent"] - ref_sum["dev_mean_wer_percent"]
    rates = {s: cmp["sets"][s]["mismatch_rate"] for s in sets}
    length_diffs = {s: len(cmp["sets"][s]["encoder_length_differences"]) for s in sets}
    structural = {"finite_and_decoded": arm_sum["structurally_valid"],
                  "encoder_lengths_equal_x3": not any(length_diffs.values()),
                  "sample_encoder_check": None if enc is None else not (enc["not_finite"] or enc["length_differences"])}
    valid = structural["finite_and_decoded"] and structural["encoder_lengths_equal_x3"] and structural["sample_encoder_check"] is not False
    criteria = {"mean_within_0.02": abs(mean_delta) <= MEAN_POINTS + EPS,
                "no_set_worse_than_0.10": all(v <= SET_WORSE_POINTS + EPS for v in delta.values()),
                "pooled_mismatch_le_0.5%": cmp["pooled"]["mismatch_rate"] <= POOLED_MISMATCH + EPS,
                "each_set_mismatch_le_1.0%": all(v <= SET_MISMATCH + EPS for v in rates.values())}
    verdict = "INVALID" if not valid else ("recognition-equivalent" if all(criteria.values()) else "trade-off")
    return {"verdict": verdict, "structural": structural, "criteria": criteria,
            "mean_wer_percent": arm_sum["dev_mean_wer_percent"], "mean_delta_points": mean_delta,
            "set_wer_percent": arm_sum["dev_wer_percent"], "set_delta_points": delta,
            "token_mismatch_rate": {"pooled": cmp["pooled"]["mismatch_rate"], **rates},
            "token_mismatches": {"pooled": cmp["pooled"]["token_mismatches"],
                                 **{s: cmp["sets"][s]["token_mismatches"] for s in sets}},
            "text_mismatches": {s: cmp["sets"][s]["text_mismatches"] for s in sets},
            "empty_hypotheses": arm_sum["empty_hypotheses"], "empty_hypotheses_x3": ref_sum["empty_hypotheses"],
            "encoder_length_differences": length_diffs}


def report(args) -> dict:
    """The reading of every xN arm, from identity-bound inputs only (review c1 majors 1 and 3)."""
    parts.install()
    run = args.run
    counts = _full_counts()
    out = {"run": run, "rule": {"mean_points": MEAN_POINTS, "set_worse_points": SET_WORSE_POINTS,
                                "pooled_mismatch": POOLED_MISMATCH, "set_mismatch": SET_MISMATCH,
                                "source": "plans/activation-precision.md Reading + R2 + R4 + R5 + R7"},
           "full_dev_utterances": counts, "models": {}}
    for model, arms in ARMS.items():
        ref = arms[3]
        try:
            ref_sum = parts.require_reference(run, arms[2])
        except SystemExit as exc:  # R2: no classification against a failed / invalid / unverified reference
            reason = str(exc)
            out["models"][model] = {"x3": {"arm": ref, "valid_reference": False, "reason": reason},
                                    **{f"x{n}": {"arm": arms[n], "verdict": "NOT CLASSIFIED", "reason": reason} for n in (2, 1)}}
            continue
        ids = {ref: ref_sum["identity"]}
        sums = {}
        for n in (2, 1):
            sums[n], ids[arms[n]] = verified_summary(run, arms[n], counts)
        enc_doc = verified_encoder_check(run, model, ids)
        if enc_doc["x3_not_finite"]:  # review c2: a non-finite x3 encoder output invalidates the reference (R5)
            reason = (f"R5/R2: x3 reference {ref} encoder output not finite on {len(enc_doc['x3_not_finite'])} of the "
                      f"512-sample utterances (e.g. {enc_doc['x3_not_finite'][:3]}); classification withheld")
            out["models"][model] = {"x3": {"arm": ref, "valid_reference": False, "reason": reason},
                                    **{f"x{n}": {"arm": arms[n], "verdict": "NOT CLASSIFIED", "reason": reason} for n in (2, 1)}}
            continue
        enc = enc_doc["summary"]
        entry = {"x3": {"arm": ref, "valid_reference": True, "identity": ids[ref], "mean_wer_percent": ref_sum["dev_mean_wer_percent"],
                        "set_wer_percent": ref_sum["dev_wer_percent"], "cross_check": ref_sum.get("cross_check"),
                        "structurally_valid": ref_sum["structurally_valid"], "empty_hypotheses": ref_sum["empty_hypotheses"]}}
        for n in (2, 1):
            arm = arms[n]
            cmp = verified_compare(run, arm, ref, ids, counts)
            entry[f"x{n}"] = {"arm": arm, "identity": ids[arm], **reading(sums[n], ref_sum, cmp, enc[f"x{n}"]),
                              "sample_encoder_rel_frobenius": enc[f"x{n}"]["rel_frobenius_valid"]}
        out["models"][model] = entry
    a3_path = out_dir() / "a3-parts.json"
    if a3_path.exists():
        doc = json.loads(a3_path.read_text())
        if doc.get("pinned") != parts.PINNED:
            raise ValueError("a3-parts.json was not computed from the pinned parts files")
        out["a3_descriptive"] = {"summary": doc["summary"], "b_over_m1_relative_frobenius": doc["b_over_m1_relative_frobenius"],
                                 "note": "descriptive only; B's pinned parts files and M1's own ExpandedProjection"}
    path = out_dir() / f"report-{run}.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    for model, entry in out["models"].items():
        x3 = entry["x3"]
        if not x3["valid_reference"]:
            print(f"{model}: x3 reference NOT VALID; xN NOT CLASSIFIED: {x3['reason'][:300]}")
            continue
        print(f"{model} x3 {x3['arm']}: mean {x3['mean_wer_percent']:.4f}%  cross-check {(x3['cross_check'] or {}).get('pass')}")
        for n in (2, 1):
            e = entry[f"x{n}"]
            print(f"  x{n}: {e['verdict']}: mean {e['mean_wer_percent']:.4f}% ({e['mean_delta_points']:+.4f}), "
                  f"worst set {max(e['set_delta_points'].values()):+.3f}, mismatches pooled "
                  f"{100 * e['token_mismatch_rate']['pooled']:.2f}%, max set "
                  f"{100 * max(v for k, v in e['token_mismatch_rate'].items() if k != 'pooled'):.2f}%")
    print(path)
    return out


# --- timing summary (Measurements 2) ---------------------------------------------------------------

TIMING_PAIRS = [("M1-x2", "M1-x3"), ("M1-x1", "M1-x3"),
                ("MB1-int8_mma-x2", "MB1-int8_mma-x3"), ("MB1-int8_mma-x1", "MB1-int8_mma-x3"),
                ("MB1-int8_mma-x3", "M1-x3"), ("MB1-int8_mma-x2", "M1-x2"), ("MB1-int8_mma-x1", "M1-x1")]
_KEYS = {"energy": "joules_per_iteration", "latency": "median_ms", "average_power": "average_watts"}


def paired_changes(rows: list[dict], a: str, b: str) -> dict:
    """Median of per-utterance (a - b) / b in percent, with the count of utterances where a is lower."""
    from gpu1bit.shared import selection
    out = {name: selection.paired(rows, a, b, key) for name, key in _KEYS.items()}
    return {**out, "n": len(rows)}


def bound_timing_docs(roster: dict, root: Path, T) -> dict:
    """{step name: artifact} of every timing step, each accepted, byte-identical to what was accepted and
    bound to the roster's identities, run and references (T = gpu1bit.rerun_timing), or ValueError."""
    bind = T.make_binding(roster, T.identity_guard(roster))
    docs, unbound = {}, []
    for step in T.timing_steps(roster["run"], roster):
        if not T.reusable(step, T._sidecar(root, step), bind, roster):
            unbound.append(step["name"])
            continue
        docs[step["name"]] = json.loads(step["output"].read_text())
    if unbound:
        raise ValueError(f"timing artifacts not bound to the roster / identities / references: {unbound}")
    return docs


def timing(args) -> dict:
    import gpu1bit.rerun_timing as T
    roster_path = Path(args.roster).resolve()
    roster = json.loads(roster_path.read_text())
    docs = bound_timing_docs(roster, roster_path.parent, T)
    out = {"roster": roster, "identity_run": roster["identity_run"], "scopes": {}, "clips": {},
           "identities": T.identity_guard(roster)["identities"]}
    for scope in ("encoder", "full"):
        rows = sorted((r for n, d in docs.items() if n.startswith(f"paired-{scope}-") for r in d["rows"]),
                      key=lambda r: r["index"])
        if [r["index"] for r in rows] != list(range(24)):
            raise ValueError(f"{scope}: dev_24 rows {[r['index'] for r in rows]}")
        import statistics
        absolute = {v: {k: statistics.median(r["summary"][v][key] for r in rows) for k, key in _KEYS.items()}
                    for v in roster["variants"]}
        entry = {"absolute_median": absolute, "pairs": {f"{a} vs {b}": paired_changes(rows, a, b) for a, b in TIMING_PAIRS}}
        if scope == "full":
            entry["text_identical_to_x3"] = {a: sum(r["checks"][a]["text"] == r["checks"][b]["text"] for r in rows)
                                             for a, b in TIMING_PAIRS[:4]}
        else:
            entry["encoder_checks_vs_first"] = {v: {"all_finite": all(r["checks"][v]["finite"] for r in rows),
                                                    "lengths_equal": all(r["checks"][v]["lengths_equal"] for r in rows)}
                                                for v in roster["variants"]}
        out["scopes"][scope] = entry
    for name, d in docs.items():
        if name.startswith("clips"):
            order = name.split("-", 1)[0]
            out["clips"].setdefault(d["variant"], {})[order] = {
                "clips": [{"id": w["record"]["id"], "duration": w["record"]["duration"], **w["summary"],
                           "process_peak_incl_first_bytes": w["process_sampled_peak_including_first_bytes"],
                           "hypothesis": w["hypothesis"]} for w in d["workloads"]],
                "performance_phase_process_peak_bytes": d["performance_phase_process_peak_bytes"],
                "load_seconds": d["load_seconds"]}
    path = out_dir() / f"timing-{roster['run']}.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    for scope in ("full", "encoder"):
        print(f"[{scope}]")
        for pair, c in out["scopes"][scope]["pairs"].items():
            print(f"  {pair}: energy {c['energy']['median_pct']:+.2f}% (lower {c['energy']['lower']}/24), latency "
                  f"{c['latency']['median_pct']:+.2f}% ({c['latency']['lower']}/24), power {c['average_power']['median_pct']:+.2f}%")
    print(path)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    s = sub.add_parser("encoder-check")
    s.add_argument("--run", required=True)
    s.add_argument("--model", choices=sorted(ARMS), required=True)
    sub.add_parser("a3")
    s = sub.add_parser("timing")
    s.add_argument("--roster", required=True, help="the timing roster file (H/actprec/timing-<run>/roster.json)")
    s = sub.add_parser("report")
    s.add_argument("--run", required=True)
    args = p.parse_args()
    {"encoder-check": encoder_check, "a3": a3, "report": report, "timing": timing}[args.action](args)


if __name__ == "__main__":
    main()
