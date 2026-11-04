"""Results of stage 4 (keab-prime plan "Scoring and gate (preregistered)", "Outputs"): $RELABEL_ROOT/keab-prime/
results.json and RESULTS-keab-prime.md next to the plan ($RELABEL_ROOT/plans/).

    ./python train_keab.py report      (CPU; reads $RELABEL_ROOT/keab-prime/, step 0's and stage 3's scores)

WER in points on stage 1's full dev population; intervals are 90% paired cluster bootstraps (training/bootstrap.py,
10,000 replicates, clusters as training/devsplit.py, H4 recomputed per replicate). Sections, each "pending" until its
inputs exist:

- the batch plan's counts, the memory probe and the GPU-hour estimate;
- the pilot: each arm's H4 per learning rate (PyTorch fake-quant mode) and the frozen selection;
- the MB1 check: step 0's re-score of MB1 by the x1 builder against gpu1bit/RESULTS.md's x1 numbers (0.02 per set);
- the gate (deployed mode, x1): delta = H4(control b100) - H4(Kea-B') with its interval (>= 0.10, lower bound above
  0), per H4 set WER(Kea-B') - WER(control) (no set more than 0.25 worse), the domain sets' WER and term recall (a
  regression above 0.5 against the control is reported to AJ before stage 5); pass;
- without a threshold: Kea-B' and the control against MB1 (deployed mode); kernel fidelity (x1 against x3 and against
  the PyTorch fake-quantized model: WER and token-sequence mismatch, for Kea-B', the control and MB1); cleaning v7's
  scorers (style-neutral WER with the symmetric acr canonicalizer and the unit group; term recall, whose occurrences
  count a unit and its abbreviation after a number as one term, rescored from stored transcripts for every model);
  synthetic v5's metrics (synthetic/pair_metrics.py: stable held-out recall, pair recall with and without the
  dev-informed seed, partner substitution, near forms, split v3's targeted recall); formatted metrics; YODAS dev, CV
  dev and AMI SDM dev; the b50-to-b100 curve; med-dev-2's letter forms (a development diagnostic only); the bundle size
  of Kea-B''s export and step 0's table.

Comparison models are rescored from their stored transcripts: B0 ($RELABEL_ROOT/b0prime/scores/B0), B0'
($RELABEL_ROOT/b0prime/runs/b100), MB1 (step 0's deployed x1 scores of its unquantized export, plus this stage's).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import train_b0prime as tb
import train_keab as tk
from labeling import layout as llayout
from training import bootstrap, stage4

EPS = stage4.EPS
MODELS = ("Kea-B'", "control", "b50", "MB1", "B0", "B0p")
LETTER_FORM = re.compile(r"\b(?:[A-Za-z]-){2,}[A-Za-z]\b")


def _json(path: Path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def score_dirs() -> dict[str, dict[str, Path]]:
    """{model: {mode: directory of <set>.json}}: this stage's scores, step 0's MB1 x1, stage 3's B0 and B0'."""
    b0p = llayout.relabel_root() / "b0prime"
    step0 = tk.root() / "step0" / "scores" / "MB1" / "none"
    out = {"Kea-B'": {m: tk.scores_dir("b100-main", m) for m in ("x1", "x3", "torch")},
           "control": {m: tk.scores_dir("b100-control", m) for m in ("x1", "x3", "torch")},
           "b50": {m: tk.scores_dir("b50-main", m) for m in ("x1", "x3", "torch")},
           "MB1": {"x1": [step0, tk.scores_dir("MB1", "x1")], "x3": tk.scores_dir("MB1", "x3"),
                   "torch": tk.scores_dir("MB1", "torch")},
           "B0": {"fp32": [b0p / "scores" / "B0" / sub for sub in ("eval-dev", "domain", "eval-stem", "cv-dev")]
                  + [tk.scores_dir("B0", "fp32")]},
           "B0p": {"fp32": [b0p / "runs" / "b100" / sub for sub in ("eval-dev", "domain", "eval-stem", "cv-dev")]
                   + [tk.scores_dir("B0p", "fp32")]}}
    return out


def records(model: str, name: str, mode: str | None = None) -> list | None:
    """A model's per-utterance records of a set in a mode (default: its deployed or FP32 scoring); None if absent."""
    dirs = score_dirs()[model]
    mode = mode or ("x1" if "x1" in dirs else "fp32")
    where = dirs.get(mode)
    for d in (where if isinstance(where, list) else [where]):
        if d is not None and (Path(d) / f"{name}.json").exists():
            return _json(Path(d) / f"{name}.json")["records"]
    return None


def _corpus(recs: list[dict]) -> float:
    rows = [r for r in recs if r.get("scored", True)]
    return 100 * sum(r["S"] + r["D"] + r["I"] for r in rows) / max(1, sum(len(r["ref_norm"].split()) for r in rows))


def h4_of(model: str, mode: str | None = None) -> float | None:
    recs = {n: records(model, n, mode) for n in tb.H4}
    if any(r is None for r in recs.values()):
        return None
    return sum(_corpus(r) for r in recs.values()) / len(tb.H4)


def paired(a: str, b: str, mode_a: str | None = None, mode_b: str | None = None) -> dict | None:
    """bootstrap.paired of WER(a) - WER(b) on the H4 sets (training/devsplit.py clusters), or None if not scored."""
    recs = {"a": {n: records(a, n, mode_a) for n in tb.H4}, "b": {n: records(b, n, mode_b) for n in tb.H4}}
    if any(r is None for m in recs.values() for r in m.values()):
        return None
    res = bootstrap.paired(tb.full_tables(recs), {"d": ("a", "b")}, mean_sets=tb.H4)
    return {"h4": res["mean"]["d"], "sets": {n: res["sets"][n]["d"] for n in tb.H4},
            "wer": {n: res["wer"][n] for n in list(tb.H4) + ["mean"]}, "models": {"a": a, "b": b}}


def mismatch(a: list[dict], b: list[dict]) -> dict:
    by = {r["id"]: r.get("tokens") for r in b}
    ids = [r["id"] for r in a if r["id"] in by and r.get("tokens") is not None and by[r["id"]] is not None]
    n = sum(r["tokens"] != by[r["id"]] for r in a if r["id"] in ids)
    return {"utterances": len(ids), "mismatches": n, "rate": n / len(ids) if ids else None}


# ----------------------------------------------------------------------------- sections

def plan_section() -> dict:
    reports = sorted((llayout.relabel_root() / "training" / "batch-plan" / "reports").glob("*-v3x*.json"))
    return {"status": "complete", "reports": [_json(p) for p in reports]} if reports else {"status": "pending"}


def probe_section() -> dict:
    body = _json(tk.probe_path()) or _json(tk.root() / "probe" / "provisional.json")
    if body is None:
        return {"status": "pending"}
    return {"status": "complete", "provisional": body.get("provisional"), "decision": body["decision"],
            "modes": {k: {x: m.get(x) for x in ("status", "step_time_s", "peak_process_gib_estimate",
                                                "peak_reserved_gib", "host_peak_rss_gib", "audio_seconds_per_step")}
                      for k, m in body["modes"].items()}, "estimate": body.get("estimate"),
            "host_peak_rss_gib": body.get("host_peak_rss_gib")}


def pilot_section() -> dict:
    h4 = {}
    for n in stage4.PILOT:
        s = _json(tk.run_dir(n) / "scores.json")
        h4[n] = None if s is None else s.get("h4")
    return {"h4": h4, "selection": _json(tk.selection_path())}


def mb1_check_section() -> dict:
    import keab_step0 as k0
    body = _json(tk.root() / "step0" / "mb1-x1-check" / "summary.json")
    if body is None:
        return {"status": "pending (step 0's mb1-check)"}
    got = body.get("wer_percent") or (body.get("check") or {}).get("wer_percent")
    if got is None:
        return {"status": "complete", "step0": body}
    return {"status": "complete", "check": stage4.mb1_check({n: got[n] for n in k0.DEV_SETS}, k0.RECORDED_X1)}


def domain_means(model: str) -> dict:
    import devsets
    import score_domain
    import terms
    out = {}
    for s in tk.DOMAIN_SETS:
        parts = {p: records(model, p) for p in devsets.SETS[s]}
        if any(r is None for r in parts.values()):
            continue
        matcher = terms.matcher(tk.TERM_KIND[s])
        rec = {p: score_domain.term_recall(r, matcher) for p, r in parts.items()}       # v7: rescored
        occ, rep = sum(x["occurrences"] for x in rec.values()), sum(x["reproduced"] for x in rec.values())
        out[s] = {"mean_wer": sum(_corpus(r) for r in parts.values()) / len(parts),
                  "term_recall_mean": sum(x["recall"] for x in rec.values()) / len(rec)
                  if all(x["recall"] is not None for x in rec.values()) else None,
                  "term_recall_pooled": rep / occ if occ else None}
    return out


def gate_section() -> dict:
    if records("Kea-B'", tb.H4[0]) is None or records("control", tb.H4[0]) is None:
        return {"status": "pending (deployed-mode scores of b100-main and b100-control)"}
    res = paired("control", "Kea-B'")
    if res is None:
        return {"status": "pending"}
    per_set = {n: -res["sets"][n]["point"] for n in tb.H4}                 # WER(Kea-B') - WER(control)
    dm = {m: domain_means(m) for m in ("Kea-B'", "control")}
    domain = {s: dm["Kea-B'"][s]["mean_wer"] - dm["control"][s]["mean_wer"] for s in dm["Kea-B'"] if s in dm["control"]}
    decision = stage4.gate(res["h4"], per_set, domain)
    return {"status": "complete", "delta": res, "per_set_regression": per_set, "domain": dm,
            "domain_difference_points": domain, "decision": decision}


def against_mb1_section() -> dict:
    return {m: paired("MB1", m) for m in ("Kea-B'", "control")}


def fidelity_section() -> dict:
    out = {}
    for m in ("Kea-B'", "control", "MB1"):
        row = {}
        for other in ("x3", "torch"):
            pairs = [(records(m, n, "x1"), records(m, n, other)) for n in tb.DEV_SETS]
            if any(a is None or b is None for a, b in pairs):
                row[other] = None
                continue
            a = [r for x, _ in pairs for r in x]
            b = [r for _, y in pairs for r in y]
            row[other] = {"mismatch": mismatch(a, b), "h4": {"x1": h4_of(m, "x1"), other: h4_of(m, other)}}
        out[m] = row
    out["note"] = "MB1's recorded x1-versus-x3 token-sequence mismatch is 5.35% (gpu1bit/RESULTS.md)"
    return out


def style_section() -> dict:
    """Cleaning v7's style-neutral WER (style_metrics: the symmetric acr canonicalizer and the unit group) per set and
    model, from stored transcripts; Kea-B' - control per variant with paired cluster-bootstrap intervals."""
    import b0prime_report as br
    import style_metrics as sm
    out, scored_all = {"sets": {}, "contrast": {}}, {}
    for name in list(tb.DEV_SETS) + list(br.DOMAIN_PARTS) + [tk.CV_DEV_NAME, tk.AMI_SDM_DEV_NAME]:
        recs = {m: records(m, name) for m in MODELS}
        recs = {m: r for m, r in recs.items() if r is not None}
        if not recs:
            continue
        scored = {m: sm.score_records(r) for m, r in recs.items()}
        scored_all[name] = scored
        out["sets"][name] = {m: {"wer": s["wer"], "contribution": s["contribution"]} for m, s in scored.items()}
    for v in sm.VARIANTS:
        tables = {}
        for name, scored in scored_all.items():
            fn = br.cluster_fn(name)
            if fn is not None and {"Kea-B'", "control"} <= set(scored):
                tables[name] = br.variant_table({m: scored[m] for m in ("Kea-B'", "control")}, v, fn)
        if tables:
            full = set(tb.H4) <= set(tables)
            res = br.variant_bootstrap(tables, ("Kea-B'", "control"), list(tb.H4) if full else [])
            out["contrast"][v] = {"sets": res["sets"], "H4": res.get("mean"), "h4_wer": res["wer"].get("mean")}
    return out


def formatted_section() -> dict:
    import b0prime_report as br
    import style_metrics as sm
    out = {}
    for name in br.FORMATTED_SETS + br.PUNCT_ONLY_SETS:
        recs = {m: records(m, name) for m in MODELS}
        out[name] = {m: sm.formatted_summary(sm.score_records(r)["formatted"], punct_only=name in br.PUNCT_ONLY_SETS)
                     for m, r in recs.items() if r is not None}
    return out


def synthetic_section() -> dict:
    """Synthetic v5's metrics (synthetic/pair_metrics.py) of every model scored on dev-technical and eval-stem."""
    from synthetic import pair_metrics
    parts = pair_metrics.TECH_PARTS + pair_metrics.STEM_PARTS
    have = [m for m in MODELS if all(records(m, p) is not None for p in parts)]
    if not {"Kea-B'", "control"} <= set(have):
        return {"status": "pending"}
    from synthetic import layout as slayout
    from synthetic import vocab
    data = {p: {m: records(m, p) for m in have} for p in parts}             # pair_metrics' {part: {model: records}}
    res = pair_metrics.analyze(data, vocab.Split.load(slayout.split_version_path(2)),
                               vocab.Split.load(slayout.split_version_path(3)),
                               {"Kea-B'-control": ("Kea-B'", "control")})
    return {"status": "complete", "models": have, "metrics": res}


def other_sets_section() -> dict:
    out = {}
    for name in ("yodas_dev", tk.CV_DEV_NAME, tk.AMI_SDM_DEV_NAME):
        out[name] = {m: (None if records(m, name) is None else _corpus(records(m, name))) for m in MODELS}
    return out


def curve_section() -> dict:
    return {"b50": h4_of("b50"), "b100": h4_of("Kea-B'")}


def letter_forms_section() -> dict:
    """med-dev-2's letter-by-letter forms (hyphenated letter runs) per model's hypotheses and in the references: a
    development diagnostic only, not evidence that spelling was restored (cleaning plan v7)."""
    out = {}
    for m in MODELS:
        r = records(m, "med-dev-2")
        if r is not None:
            out[m] = {"hypotheses_with_letter_forms": sum(bool(LETTER_FORM.search(x.get("hyp") or "")) for x in r),
                      "letter_forms": sum(len(LETTER_FORM.findall(x.get("hyp") or "")) for x in r),
                      "utterances": len(r)}
            out.setdefault("references", {"letter_forms": sum(len(LETTER_FORM.findall(x.get("ref") or ""))
                                                              for x in r)})
    return out


def init_section() -> dict:
    """The main arm's student at step 0 (MB1's encoder, B0''s decoder; deployed x1), informational, against MB1."""
    d = tk.scores_dir(tk.INIT_MODEL, "x1")
    recs = {n: _json(d / f"{n}.json") for n in tb.H4}
    if any(r is None for r in recs.values()):
        return {"status": "pending"}
    h4 = sum(_corpus(r["records"]) for r in recs.values()) / len(tb.H4)
    return {"status": "complete", "h4": h4, "mb1_h4": h4_of("MB1"), "model": _json(tk.init_dir() / "model.json")}


def eightbit_section() -> dict:
    model = _json(tk.run_dir(stage4.KEAB_PRIME) / "model.json")
    step0 = _json(tk.step0_report())
    return {"keab_prime_export": None if model is None else model.get("export"),
            "step0": None if step0 is None else {"chosen_scope": step0.get("chosen_scope"),
                                                 "passes": step0.get("passes"), "exports": step0.get("exports")}}


def runs_table() -> list[dict]:
    rows = []
    for name in stage4.RUNS:
        s = _json(tk.run_dir(name) / "summary.json")
        if s is not None:
            k = s.get("keab", {})
            rows.append({"run": name, "arm": stage4.RUNS[name].arm, "lr": s["lr"], "seed": s["seed"],
                         "steps": s["steps"], "consumed": k.get("consumed_batches"),
                         "fully_rejected": k.get("fully_rejected_batches"),
                         "nonfinite_skipped": k.get("nonfinite_skipped_steps"),
                         "teacher_audio_rows": k.get("teacher_audio_rows"), "h4_torch": k.get("h4")})
    return rows


def analyze_all() -> dict:
    out = {"plan": plan_section(), "probe": probe_section(), "pilot": pilot_section(), "mb1_check": mb1_check_section(),
           "gate": gate_section(), "runs": runs_table()}
    for key, fn in (("against_mb1", against_mb1_section), ("fidelity", fidelity_section), ("style", style_section),
                    ("formatted", formatted_section), ("synthetic", synthetic_section),
                    ("other_sets", other_sets_section), ("curve", curve_section), ("letter_forms", letter_forms_section),
                    ("init_main", init_section),
                    ("eightbit", eightbit_section)):
        try:
            out[key] = fn()
        except (FileNotFoundError, KeyError, ValueError) as error:
            out[key] = {"status": f"pending ({type(error).__name__}: {error})"}
    return out


def _f(x, nd=2):
    return "–" if x is None else f"{x:.{nd}f}"


def _ci(e):
    return "–" if not e else f"{e['point']:+.2f} [{e['ci'][0]:+.2f}, {e['ci'][1]:+.2f}]"


def markdown(r: dict) -> str:
    L = ["# Stage 4 results: Kea-B′, Kea-B continued from MB1 with B0′ as teacher", "",
         "Generated by `train_keab.py report` from `$RELABEL_ROOT/keab-prime/` (keab-prime plan). WER in points on stage "
         "1's full dev population; intervals are 90% paired cluster-bootstrap intervals (10,000 replicates). The "
         "learning rate is selected on these dev sets, which earlier stages also used: the gate is a development "
         "decision, not a confirmatory result.", ""]
    p = r["probe"]
    L += ["## Memory probe", ""]
    if p["status"] == "complete":
        for k, m in p["modes"].items():
            L.append(f"- {k}: status {m['status']}, peak {_f(m['peak_process_gib_estimate'], 1)} GiB, step "
                     f"{_f(m['step_time_s'], 3)} s, host {_f(m['host_peak_rss_gib'], 1)} GiB")
        d = p["decision"]
        L.append(f"- Decision: {d['chosen']}; lease {d['lease_vram_gb']} GB{' (provisional plan)' if p['provisional'] else ''}.")
        if p.get("estimate"):
            e = p["estimate"]
            L.append(f"- Estimate: {e['planned_batches']} planned batches over {e['pools']} pools: pilot "
                     f"{e['pilot_gpu_hours']:.1f}, main {e['main_gpu_hours']:.1f}, total {e['total_gpu_hours']:.1f} GPU-h.")
    else:
        L.append("pending")
    L += ["", "## Pilot (pools 0–77, PyTorch fake-quant mode)", "", "| Run | H4 |", "|---|---:|"]
    L += [f"| {n} | {_f(v, 3)} |" for n, v in r["pilot"]["h4"].items()]
    sel = r["pilot"]["selection"]
    if sel:
        L += ["", f"Selected peak learning rate: **{sel['selected_name']}** (mean H4 over the arms: "
              + ", ".join(f"{k} {v:.3f}" for k, v in sel["mean_h4"].items()) + ")."]
    g = r["gate"]
    L += ["", "## Gate (deployed mode, i8x1 kernels at x1)", ""]
    if g["status"] == "complete":
        L += ["| Set | WER control | WER Kea-B′ | Kea-B′ − control |", "|---|---:|---:|---:|"]
        w = g["delta"]["wer"]
        for n in tb.H4:
            L.append(f"| {n} | {_f(w[n]['a'])} | {_f(w[n]['b'])} | {g['per_set_regression'][n]:+.2f} "
                     f"[{-g['delta']['sets'][n]['ci'][1]:+.2f}, {-g['delta']['sets'][n]['ci'][0]:+.2f}] |")
        L.append(f"| H4 | {_f(w['mean']['a'])} | {_f(w['mean']['b'])} | Δ = control − Kea-B′ "
                 f"{_ci(g['delta']['h4'])} |")
        d = g["decision"]
        L += ["", f"- Improvement (Δ ≥ 0.10, interval excluding 0): **{'yes' if d['improvement'] else 'no'}**.",
              f"- No large regression (worst {d['worst_set_regression']:+.2f}): **{'yes' if d['no_large_regression'] else 'no'}**.",
              f"- **Gate: {'PASS' if d['pass'] else 'FAIL'}.**"]
        if d["domain_regressions_over_flag"]:
            L.append(f"- **Reported to AJ before stage 5:** domain regression above 0.5 on {d['domain_regressions_over_flag']}.")
    else:
        L.append(g["status"])
    L += ["", "## Kernel fidelity", ""]
    for m, row in r["fidelity"].items():
        if m == "note":
            continue
        for other, v in (row or {}).items():
            if v:
                L.append(f"- {m}, x1 vs {other}: token mismatch {_f(100 * v['mismatch']['rate'] if v['mismatch']['rate'] is not None else None)}%"
                         f", H4 x1 {_f(v['h4']['x1'], 3)} vs {_f(v['h4'][other], 3)}")
    L += ["", f"({r['fidelity'].get('note')})", "", "## Other sets", "", "| Set | " + " | ".join(MODELS) + " |",
          "|---|" + "---:|" * len(MODELS)]
    for name, row in (r["other_sets"] if isinstance(r["other_sets"], dict) and "status" not in r["other_sets"] else {}).items():
        L.append(f"| {name} | " + " | ".join(_f(row[m]) for m in MODELS) + " |")
    L += ["", "## Accuracy curve (main arm)", "", f"b50 {_f(r['curve'].get('b50'), 3)}, b100 {_f(r['curve'].get('b100'), 3)}",
          "", "Further sections (cleaning v7 scorers, synthetic v5 metrics, formatted metrics, med-dev-2 letter forms, "
          "8-bit) are in `results.json`."]
    return "\n".join(L) + "\n"


def main() -> int:
    llayout.require_mount()
    r = analyze_all()
    tk.write_json(tk.root() / "results.json", r)
    text = markdown(r)
    tk.results_md().write_text(text)
    print(text)
    return 0
