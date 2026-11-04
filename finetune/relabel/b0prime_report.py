"""Results of stage 3 (b0prime plan "Gate to stage 4", "Outputs and reporting"): $RELABEL_ROOT/b0prime/results.json and
RESULTS-b0prime.md next to the plan ($RELABEL_ROOT/plans/).

    ./python train_b0prime.py report      (CPU; reads $RELABEL_ROOT/b0prime/)

Everything is on stage 1's full dev population, scored by train_b0prime.score_model (stage 1's scorer); WER in
points; intervals are 90% paired cluster bootstraps (training/bootstrap.py, 10,000 replicates, clusters as
training/devsplit.py, H4 recomputed per replicate). Sections, each reported as pending until its inputs exist:

- the memory probe and the GPU-hour estimate (probe/probe.json);
- the B0 re-score sanity check (scores/B0/crosscheck.json: every dev set within 0.05 points of stage 1);
- the pilot: the three learning rates' H4 and the frozen selection (selection.json); the WSD sanity pair and the
  frozen rule (wsd-rule.json), reported as a sanity check at one learning rate and a short horizon;
- the curve: H4 of b25, b50 and b100, with b25 or b50 beating b100 by more than 0.10 flagged for the user;
- the gate on B0' = b100 (fixed in advance) against the re-scored B0: H4 delta = WER(B0) - WER(B0') with its interval
  (>= 0.20, lower bound above 0), per H4 set delta with its interval (no set more than 0.25 points worse), pass;
- domain: dev-medical and dev-technical WER and critical-term recall of B0, B0' and T* (T*'s from
  devsets/scores/<T*>/summary.json when it exists; there is no such scorer run yet), a regression above 0.5 flagged;
- without a threshold: the share of the B0-to-T* gap B0' closes per set and on H4 (T* = stage 1's records), style
  rates (punctuation and capitalization per set) of B0, B0' and T*, YODAS dev, and every run's training counts.
"""
from __future__ import annotations

import json
from pathlib import Path

import analyze
import relabel_paths
import train_b0prime as tb
from labeling import layout as llayout
from training import bootstrap, devsplit, stage3

T_STAR = "granite-speech-5.0-2b"
EPS = stage3.EPS


def _json(path: Path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def t_star_records(sets=tb.DEV_SETS) -> dict[str, list]:
    return {n: json.loads((relabel_paths.DIAGNOSTIC / T_STAR / f"{n}.json").read_text())["records"] for n in sets}


def t_star_domain() -> dict | None:
    s = _json(llayout.relabel_root() / "devsets" / "scores" / T_STAR / "summary.json")
    return s.get("sets") if s and s.get("status") == "complete" else None


def scored(name: str) -> bool:
    return tb.run_done(name) and (tb.run_dir(name) / "scores.json").exists()


def gate_section() -> dict:
    """The gate on b100 against the re-scored B0 (stage3.gate), with the gap and style rates."""
    if not tb.b0_done():
        return {"status": "B0 re-score pending or failed",
                "crosscheck": _json(tb.b0_dir() / "crosscheck.json")}
    if not scored(stage3.B0PRIME):
        return {"status": f"{stage3.B0PRIME} pending"}
    recs = {"B0": tb.model_records(tb.b0_dir(), tb.DEV_SETS),
            "B0p": tb.model_records(tb.run_dir(stage3.B0PRIME), tb.DEV_SETS)}
    res = bootstrap.paired(tb.full_tables(recs), {"delta": ("B0", "B0p")}, mean_sets=tb.H4)
    per_set = {n: -res["sets"][n]["delta"]["point"] for n in tb.H4}            # WER(B0') - WER(B0)
    b0_domain = _json(tb.b0_dir() / "domain" / "summary.json") or {}
    bp_domain = _json(tb.run_dir(stage3.B0PRIME) / "domain" / "summary.json") or {}
    domain_diff = {s: bp_domain["sets"][s]["mean_wer"] - b0_domain["sets"][s]["mean_wer"]
                   for s in bp_domain.get("sets", {}) if s in b0_domain.get("sets", {})}
    crosscheck = _json(tb.b0_dir() / "crosscheck.json")
    decision = stage3.gate(res["mean"]["delta"], per_set, crosscheck, domain_diff)
    ts = t_star_records()
    gap = {}
    for n in tb.H4 + ["H4"]:
        if n == "H4":
            w = {"B0": res["wer"]["mean"]["B0"], "B0p": res["wer"]["mean"]["B0p"],
                 "T*": sum(100 * corpus(ts[x]) for x in tb.H4) / len(tb.H4)}
        else:
            w = {"B0": res["wer"][n]["B0"], "B0p": res["wer"][n]["B0p"], "T*": 100 * corpus(ts[n])}
        g = w["B0"] - w["T*"]
        gap[n] = {"wer": w, "gap_points": g, "closed_share": (w["B0"] - w["B0p"]) / g if abs(g) > EPS else None}
    style = {n: analyze.style({"B0": recs["B0"][n], "B0p": recs["B0p"][n], "T*": ts[n]}) for n in tb.DEV_SETS}
    dirs = {"B0": tb.b0_dir(), "B0p": tb.run_dir(stage3.B0PRIME)}
    neutral = style_section(dirs)
    return {"status": "complete", "bootstrap": res, "decision": decision, "per_set_regression": per_set,
            "style_neutral": neutral, "style_neutral_gate": style_gate(neutral, crosscheck, decision),
            "formatted": formatted_section(dirs), "terms": terms_section(dirs),
            "domain": {"B0": b0_domain.get("sets"), "B0p": bp_domain.get("sets"), "T*": t_star_domain(),
                       "T*_note": None if t_star_domain() else "no T* domain scores exist (score_teacher.py scores "
                                                               "the stage 1 dev sets only)",
                       "difference_points": domain_diff},
            "gap": gap, "style": style,
            "yodas_dev": {m: 100 * corpus(r["yodas_dev"]) for m, r in recs.items()}}


# ----------------------------------------------------------------------------- v4 reporting additions

DOMAIN_PARTS = ("med-dev-1", "med-dev-2", "med-dev-3", "tech-dev-1", "tech-dev-2")
FORMATTED_SETS = ("tech-dev-1", "tech-dev-2", "med-dev-1", "yodas_dev")
PUNCT_ONLY_SETS = ("voxpopuli_dev",)
F1_FLAG = 0.03


def set_records(model_dir: Path | None, name: str) -> list | None:
    """A scored model's records of a dev set (eval-dev/), a domain part (domain/), an eval-stem part (eval-stem/) or
    the cv27-en dev sample (cv-dev/); None if not scored."""
    if model_dir is None:
        return None
    sub = ("eval-dev" if name in tb.DEV_SETS else "eval-stem" if name.startswith("stem-") else
           "cv-dev" if name == tb.CV_DEV_NAME else "domain")
    data = _json(Path(model_dir) / sub / f"{name}.json")
    return None if data is None else data["records"]


def t_star_set_records(name: str) -> list | None:
    """T*'s stage 1 records of a dev set (no T* scores exist for the domain parts or the cv27-en dev sample)."""
    return t_star_records([name])[name] if name in tb.DEV_SETS else None


def cluster_fn(name: str):
    import synthetic_report
    if name in tb.H4:
        return lambda i: devsplit.cluster_of(name, i)
    if name in DOMAIN_PARTS or name.startswith("stem-"):
        return synthetic_report.recording_of
    return None                                            # YODAS, the cv27-en dev sample: informational, no interval


def variant_table(scored: dict[str, dict], variant: str, cluster_of) -> dict:
    """Per-cluster errors AND reference words of each model on one set (style_metrics.score_records results,
    {model: result}, the same utterances in every model). A canonicalizer may read the other side (acronym runs), so
    each model has its own canonical reference length."""
    import numpy as np
    models = sorted(scored)
    ids = scored[models[0]]["ids"]
    for m in models[1:]:
        if scored[m]["ids"] != ids:
            raise ValueError(f"{m} does not cover the same scored utterances as {models[0]}")
    names = sorted({cluster_of(i) for i in ids})
    index = {c: j for j, c in enumerate(names)}
    idx = np.array([index[cluster_of(i)] for i in ids], dtype=np.int64)
    words = {m: np.bincount(idx, weights=[w for _, w in scored[m]["per"][variant]], minlength=len(names))
             for m in models}
    errs = {m: np.bincount(idx, weights=[e for e, _ in scored[m]["per"][variant]], minlength=len(names))
            for m in models}
    return {"clusters": names, "words": words, "errors": errs, "utterances": len(ids)}


def variant_bootstrap(tables: dict[str, dict], contrast: tuple[str, str], mean_sets: list[str],
                      replicates: int = bootstrap.REPLICATES, seed: int = bootstrap.SEED, ci: float = bootstrap.CI) -> dict:
    """bootstrap.paired's rule (clusters drawn within each set from SeedSequence([seed, set index]) in sorted set
    order, the same draw for both models, the mean over mean_sets recomputed per replicate) with each model's own
    reference words: WER(m) = 100 x errors(m) / words(m) per replicate. Contrast (a, b) = WER(a) - WER(b)."""
    import numpy as np
    a, b = contrast
    out = {"replicates": replicates, "seed": seed, "ci_level": ci, "sets": {}, "wer": {}}
    reps = {}
    for s_index, name in enumerate(sorted(tables)):
        t = tables[name]
        n = len(t["clusters"])
        rng = np.random.default_rng(np.random.SeedSequence([seed, s_index]))
        counts = np.empty((replicates, n), dtype=np.float64)
        for k in range(replicates):
            counts[k] = np.bincount(rng.integers(0, n, n), minlength=n)
        w = {m: 100.0 * float(t["errors"][m].sum()) / float(t["words"][m].sum()) for m in (a, b)}
        r = {m: 100.0 * (counts @ t["errors"][m]) / (counts @ t["words"][m]) for m in (a, b)}
        reps[name] = r[a] - r[b]
        out["wer"][name] = w
        lo, hi = np.percentile(reps[name], [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
        out["sets"][name] = {"point": w[a] - w[b], "ci": [float(lo), float(hi)], "clusters": n}
    if mean_sets and set(mean_sets) <= set(tables):
        diff = np.mean([reps[s] for s in mean_sets], axis=0)
        lo, hi = np.percentile(diff, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
        out["mean"] = {"point": float(np.mean([out["sets"][s]["point"] for s in mean_sets])),
                       "ci": [float(lo), float(hi)]}
        out["wer"]["mean"] = {m: float(np.mean([out["wer"][s][m] for s in mean_sets])) for m in (a, b)}
    return out


def style_section(dirs: dict[str, Path | None]) -> dict:
    """Style-neutral WER (style_metrics) per dev set and domain part for B0, B0' and T*, the groups' contributions,
    and paired cluster-bootstrap intervals of B0' - B0 per variant (H4 sets and domain parts), with H4."""
    import style_metrics as sm
    out, scored_all = {"sets": {}, "contrast": {}}, {}
    for name in list(tb.DEV_SETS) + list(DOMAIN_PARTS) + [tb.CV_DEV_NAME]:
        recs = {m: set_records(d, name) for m, d in dirs.items()}
        if name in tb.DEV_SETS:
            recs["T*"] = t_star_set_records(name)
        recs = {m: r for m, r in recs.items() if r is not None}
        if not recs:
            continue
        scored = {m: sm.score_records(r) for m, r in recs.items()}
        scored_all[name] = scored
        out["sets"][name] = {m: {"wer": s["wer"], "contribution": s["contribution"]} for m, s in scored.items()}
    for v in sm.VARIANTS:
        tables = {}
        for name, scored in scored_all.items():
            fn = cluster_fn(name)
            if fn is not None and {"B0", "B0p"} <= set(scored):
                tables[name] = variant_table({m: scored[m] for m in ("B0", "B0p")}, v, fn)
        if not tables:
            continue
        full = set(tb.H4) <= set(tables)
        res = variant_bootstrap(tables, ("B0p", "B0"), list(tb.H4) if full else [])
        out["contrast"][v] = {"sets": res["sets"], "H4": res.get("mean"),
                              "h4_wer": res["wer"].get("mean")}
    return out


def formatted_section(dirs: dict[str, Path | None]) -> dict:
    """Formatted metrics (style_metrics) on the sets whose references carry case and punctuation (VoxPopuli for
    punctuation only), for B0, B0' and T* (where scored); B0' any-mark F1 more than F1_FLAG below B0's is flagged."""
    import style_metrics as sm
    out, flags = {}, []
    for name in FORMATTED_SETS + PUNCT_ONLY_SETS:
        recs = {m: set_records(d, name) for m, d in dirs.items()}
        if name in tb.DEV_SETS:
            recs["T*"] = t_star_set_records(name)
        out[name] = {m: sm.formatted_summary(sm.score_records(r)["formatted"], punct_only=name in PUNCT_ONLY_SETS)
                     for m, r in recs.items() if r is not None}
        f = {m: (s["any_mark"]["f1"]) for m, s in out[name].items()}
        if f.get("B0p") is not None and f.get("B0") is not None and f["B0"] - f["B0p"] > F1_FLAG + EPS:
            flags.append(name)
    return {"sets": out, "f1_flags": flags, "rule": f"B0' any-mark F1 more than {F1_FLAG} below B0's is reported to "
                                                    "AJ before stage 4"}


def terms_section(dirs: dict[str, Path | None]) -> dict:
    """Critical-term recall, targeted and held-out (stage 2c's term split and groups), on eval-stem and
    dev-technical, B0' against B0 with paired cluster-bootstrap intervals over recordings (synthetic_report)."""
    import synthetic_report as srep
    from synthetic import vocab
    parts = {p: {m: set_records(dirs[m], p) for m in ("B0", "B0p")} for ps in srep.TERM_SETS.values() for p in ps}
    if any(r is None for v in parts.values() for r in v.values()):
        return {"status": "pending (B0 or B0' not scored on eval-stem and dev-technical)"}
    split = vocab.Split.load()
    cov = srep.coverage_info(split)
    if cov["underpowered"] is None:
        return {"status": "coverage.json is missing or of another split"}
    tables = {p: srep.RecallTable.from_records(r, split) for p, r in parts.items()}
    res = srep.recall_bootstrap(tables, {"B0p-B0": ("B0p", "B0")}, srep.groups(cov["underpowered"]))
    return {"status": "complete", "recall": res, "underpowered": cov["underpowered"]}


def style_gate(style: dict, crosscheck: dict, standard: dict) -> dict:
    """The gate's items on the style-neutral ("all") WER, and the note: standard fails but style-neutral passes."""
    c = style.get("contrast", {}).get("all")
    if not c or c.get("H4") is None:
        return {"status": "pending"}
    delta = {"point": -c["H4"]["point"], "ci": [-c["H4"]["ci"][1], -c["H4"]["ci"][0]]}     # WER(B0) - WER(B0')
    per_set = {n: c["sets"][n]["point"] for n in tb.H4}                                       # WER(B0') - WER(B0)
    g = stage3.gate(delta, per_set, crosscheck)
    return {"status": "complete", "decision": g, "delta": delta,
            "report_both_to_aj": (not standard["pass"]) and g["pass"],
            "rule": "the gate is preregistered on standard WER; if it fails there but passes on style-neutral WER, "
                    "the architect reports both to AJ before stage 4"}


def corpus(records: list[dict]) -> float:
    rows = [r for r in records if r.get("scored", True)]
    return sum(r["S"] + r["D"] + r["I"] for r in rows) / sum(len(r["ref_norm"].split()) for r in rows)


def runs_table() -> list[dict]:
    rows = []
    for name, run in stage3.RUNS.items():
        s = _json(tb.run_dir(name) / "summary.json")
        if s is None:
            rows.append({"run": name, "status": "pending"})
            continue
        b = s.get("b0prime", {})
        rows.append({"run": name, "status": "done", "lr": s["lr"], "seed": s["seed"], "steps": s["steps"],
                     "consumed_batches": b.get("consumed_batches"), "fully_rejected": b.get("fully_rejected_batches"),
                     "nonfinite_skipped": b.get("nonfinite_skipped_steps"), "audio_hours": s.get("audio_hours_seen"),
                     "resumes": s.get("resumes"), "h4": b.get("h4"),
                     "dev_wer": {n: 100 * w for n, w in (s.get("dev_wer") or {}).items()}})
    return rows


def analyze_all() -> dict:
    curve = {n: tb.h4(tb.dev_wers(tb.run_dir(n))) for n in stage3.BRANCHES if scored(n)}
    return {"plan": "b0prime plan", "probe": _json(tb.probe_path()),
            "b0_rescore": _json(tb.b0_dir() / "crosscheck.json"),
            "pilot": {n: tb.h4(tb.dev_wers(tb.run_dir(n))) if scored(n) else None for n in stage3.PILOT},
            "selection": _json(tb.selection_path()), "wsd_rule": _json(tb.wsd_rule_path()),
            "sanity": {n: tb.h4(tb.dev_wers(tb.run_dir(n))) if scored(n) else None for n in stage3.SANITY},
            "curve": curve, "branch_flags": stage3.branch_flags(curve), "gate": gate_section(), "runs": runs_table()}


def _f(x, nd=2):
    return "–" if x is None else f"{x:.{nd}f}"


def _ci(e):
    return f"{e['point']:+.2f} [{e['ci'][0]:+.2f}, {e['ci'][1]:+.2f}]"


def markdown(r: dict) -> str:
    L = ["# Stage 3 results: B0′, FP32 Parakeet v2 fine-tuned on the new labels", "",
         "Generated by `train_b0prime.py report` from `$RELABEL_ROOT/b0prime/` (b0prime plan). WER in points on stage "
         "1's full dev population; intervals are 90% paired cluster-bootstrap intervals (10,000 replicates). The dev "
         "sets were used to select the learning rate and in stages 1 and 2: this is a development go/no-go, not a "
         "confirmatory result.", ""]
    p = r["probe"]
    L += ["## Memory probe", ""]
    if p:
        if p.get("loss_seam"):
            L.append(f"- Loss: the fused joint+TDT seam ({p['loss_seam']['source']}); worst cases: "
                     f"{len(p.get('worst_batches') or [])} batches ({p['decision'].get('coverage')}; "
                     f"{p['decision'].get('recheck')}).")
        for k, m in p["modes"].items():
            L.append(f"- {k}: status {m['status']}, peak {_f(m['peak_process_gib_estimate'], 1)} GiB (timed batches "
                     f"{_f(m.get('timed_reserved_gib'), 1)} GiB reserved), step {_f(m['step_time_s'], 3)} s, "
                     f"{_f(m['audio_seconds_per_step'], 0)} s of audio per step"
                     + (f"; out of memory: {', '.join(m['failed'])}" if m.get("failed") else ""))
        d, e = p["decision"], p.get("estimate") or {}
        if d["fits"]:
            L.append(f"- Decision: {d['chosen']} (gradient checkpointing {d['grad_checkpointing']}); lease "
                     f"{d['lease_vram_gb']} GB; host peak {_f(p['host_peak_rss_gib'], 1)} GiB.")
        else:
            e = p.get("estimate_unfit") or {}
            L.append(f"- Decision: **A1 does not pass** at 600-second batches in either mode (the architect reports); "
                     f"host peak {_f(p['host_peak_rss_gib'], 1)} GiB."
                     + (f" The estimate below uses {e['mode']}'s step time on planned batches." if e else ""))
        L.append(f"- Estimate: pilot {_f(e.get('pilot_gpu_hours'), 1)} GPU-h, main run {_f(e.get('main_gpu_hours'), 1)} "
                 "GPU-h (training steps only).")
    else:
        L.append("Pending.")
    c = r["b0_rescore"]
    L += ["", "## B0 re-score (sanity check)", ""]
    if c:
        L += ["| Set | Re-scored | Stage 1 | Difference |", "|---|---:|---:|---:|"]
        for n, v in c["sets"].items():
            L.append(f"| {n} | {v['wer']:.3f} | {v['stage1_wer']:.3f} | {v['difference_points']:+.3f} |")
        L.append(f"\nWithin {c['tolerance_points']} points on every set: **{'yes' if c['pass'] else 'no'}**.")
    else:
        L.append("Pending.")
    L += ["", "## Pilot (pools 0–77)", "", "| Run | H4 |", "|---|---:|"]
    for n, v in {**r["pilot"], **r["sanity"]}.items():
        L.append(f"| {n} | {_f(v, 3)} |")
    s = r["selection"]
    L.append("")
    L.append(f"Selected peak learning rate: **{s['selected_name']}** ({s['rule']})." if s else "Selection: pending.")
    w = r["wsd_rule"]
    if w:
        L += ["", f"WSD sanity check (seed {stage3.SANITY_SEED}, at the selected rate): d = H4(WSD) − H4(linear) = "
              f"{w['d']:+.3f}, 90% interval [{w['ci'][0]:+.3f}, {w['ci'][1]:+.3f}]. "
              + ("**Stage 3 stops** (" + "; ".join(w["reasons"]) + ")." if w["stop"] else "The main run uses WSD.")
              + " This is a sanity check at one learning rate and a short horizon, not a demonstration that WSD "
                "equals linear decay at independently tuned settings."]
    else:
        L += ["", "WSD sanity check: pending."]
    L += ["", "## Accuracy curve (cooldown branches of one trunk)", "", "| Branch | H4 |", "|---|---:|"]
    for n in stage3.BRANCHES:
        L.append(f"| {n} | {_f(r['curve'].get(n), 3)} |")
    if r["branch_flags"]:
        L.append(f"\n**Reported to the user before stage 4:** {', '.join(r['branch_flags'])} beat b100 by more than "
                 f"{stage3.BRANCH_FLAG} points of H4.")
    L += ["", "## Runs", "", "| Run | lr | Seed | Steps | Consumed | Fully rejected | Non-finite skipped | H4 |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in r["runs"]:
        L.append(f"| {row['run']} | {row.get('lr', '–')} | {row.get('seed', '–')} | {row.get('steps', '–')} | "
                 f"{row.get('consumed_batches', '–')} | {row.get('fully_rejected', '–')} | "
                 f"{row.get('nonfinite_skipped', '–')} | {_f(row.get('h4'), 3)} |")
    g = r["gate"]
    L += ["", "## Gate to stage 4 (B0′ = b100 against the re-scored B0)", ""]
    if g.get("status") != "complete":
        L.append(f"Status: {g.get('status')}.")
        return "\n".join(L) + "\n"
    b, d = g["bootstrap"], g["decision"]
    L += ["| Set | WER B0 | WER B0′ | Δ = B0 − B0′ |", "|---|---:|---:|---:|"]
    for n in tb.H4 + ["H4"]:
        wr = b["wer"]["mean"] if n == "H4" else b["wer"][n]
        e = b["mean"]["delta"] if n == "H4" else b["sets"][n]["delta"]
        L.append(f"| {n} | {wr['B0']:.2f} | {wr['B0p']:.2f} | {_ci(e)} |")
    L += ["", f"- B0 re-score within tolerance: **{'yes' if d['rescore_pass'] else 'no'}**.",
          f"- Improvement ({d['rules']['improvement']}): **{'yes' if d['improvement'] else 'no'}**.",
          f"- No large regression ({d['rules']['regression']}; worst {d['worst_set_regression']:+.2f}): "
          f"**{'yes' if d['no_large_regression'] else 'no'}**.",
          f"- **Gate: {'PASS' if d['pass'] else 'FAIL'}.**", "",
          "Domain sets (score_domain.py settings): mean WER over parts and mean critical-term recall.", "",
          "| Model | dev-medical WER | dev-medical term recall | dev-technical WER | dev-technical term recall |",
          "|---|---:|---:|---:|---:|"]
    for m in ("B0", "B0p", "T*"):
        dm = g["domain"].get(m) or {}
        med, tech = dm.get("dev-medical", {}), dm.get("dev-technical", {})
        rec, trec = med.get("mean_term_recall"), tech.get("mean_term_recall")
        L.append(f"| {m} | {_f(med.get('mean_wer'))} | {_f(100 * rec if rec is not None else None, 1)} | "
                 f"{_f(tech.get('mean_wer'))} | {_f(100 * trec if trec is not None else None, 1)} |")
    if g["domain"]["T*_note"]:
        L.append(f"\nT*: {g['domain']['T*_note']}.")
    if d["domain_regressions_over_flag"]:
        L.append(f"\n**Reported to the user before stage 4:** domain regression above {stage3.DOMAIN_FLAG} points on "
                 f"{', '.join(d['domain_regressions_over_flag'])}.")
    L += ["", "Share of the B0-to-T* gap B0′ closes: "
          + ", ".join(f"{n} {_f(None if v['closed_share'] is None else 100 * v['closed_share'], 0)}%"
                      for n, v in g["gap"].items()) + ".", "",
          "Style (share of scored hypotheses with punctuation / with an uppercase letter):", "",
          "| Set | B0 | B0′ | T* |", "|---|---|---|---|"]
    for n, st in g["style"].items():
        L.append(f"| {n} | " + " | ".join(f"{100 * st[m]['punctuation_share']:.0f} / {100 * st[m]['uppercase_share']:.0f}"
                                          for m in ("B0", "B0p", "T*")) + " |")
    L += ["", "YODAS dev (informational): " + ", ".join(f"{m} {v:.2f}" for m, v in g["yodas_dev"].items()) + "."]
    L += markdown_v4(g)
    return "\n".join(L) + "\n"


def markdown_v4(g: dict) -> list[str]:
    """The v4 reporting additions (style-neutral WER, the gate note, formatted metrics, term recall, the cv27-en dev sample)."""
    import style_metrics as sm
    L = ["", "## Style-neutral WER (v4)", "",
         "Canonicalizers applied identically to reference and hypothesis (style_metrics): acr (acronym runs), cmp "
         "(compound list), dis (fillers, false starts, repetitions), unit (dose units after a number), all. B0′ − B0 "
         "per variant, 90% paired "
         "cluster-bootstrap interval:", "", "| Variant | H4 B0 | H4 B0′ | B0′ − B0 (H4) |", "|---|---:|---:|---:|"]
    for v in sm.VARIANTS:
        c = g["style_neutral"]["contrast"].get(v) or {}
        w = c.get("h4_wer") or {}
        L.append(f"| {v} | {_f(w.get('B0'))} | {_f(w.get('B0p'))} | {_ci(c['H4']) if c.get('H4') else '–'} |")
    L += ["", "Contribution of each group (WER(std) − WER(group), points), per set:", "",
          "| Set | Model | " + " | ".join(v for v in sm.VARIANTS if v != "std") + " |",
          "|---|---|" + "---:|" * (len(sm.VARIANTS) - 1)]
    for name, models in g["style_neutral"]["sets"].items():
        for m, s in models.items():
            L.append(f"| {name} | {m} | " + " | ".join(_f(s["contribution"][v]) for v in sm.VARIANTS if v != "std")
                     + " |")
    sg = g["style_neutral_gate"]
    if sg.get("status") == "complete":
        L += ["", f"Gate items on style-neutral (all) WER: improvement **{'yes' if sg['decision']['improvement'] else 'no'}**"
              f" (Δ {_ci(sg['delta'])}), no large regression **{'yes' if sg['decision']['no_large_regression'] else 'no'}**."]
        if sg["report_both_to_aj"]:
            L.append("**The gate fails on standard WER but passes on style-neutral WER: the architect reports both to AJ "
                     "before stage 4.**")
    f = g["formatted"]
    L += ["", "## Formatted metrics (v4)", "", "| Set | Model | Any-mark P / R / F1 | . F1 | , F1 | ? F1 | Case errors | "
          "Formatted WER |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for name, models in f["sets"].items():
        for m, s in models.items():
            a = s["any_mark"]
            L.append(f"| {name} | {m} | {_f(a['precision'], 3)} / {_f(a['recall'], 3)} / {_f(a['f1'], 3)} | "
                     + " | ".join(_f(s["per_mark"][k]["f1"], 3) for k in sm.PER_MARK)
                     + f" | {_f(s.get('case_error_rate'), 4)} | {_f(s.get('formatted_wer'))} |")
    if f["f1_flags"]:
        L.append(f"\n**Reported to AJ before stage 4:** B0′'s any-mark F1 is more than {F1_FLAG} below B0's on "
                 f"{', '.join(f['f1_flags'])}.")
    t = g["terms"]
    L += ["", "## Critical-term recall, targeted and held-out (v4; eval-stem and dev-technical)", ""]
    if t.get("status") != "complete":
        L.append(f"Status: {t.get('status')}.")
    else:
        L += ["| Group | Occurrences | B0 | B0′ | B0′ − B0 |", "|---|---:|---:|---:|---:|"]
        for name in ("targeted", "heldout"):
            e = t["recall"]["groups"][name]
            c = e["contrasts"].get("B0p-B0")
            L.append(f"| {name} | {e['occurrences']} | {_f(e['recall']['B0'])} | {_f(e['recall']['B0p'])} | "
                     f"{_ci(c) if c else '–'} |")
    cv = g["style_neutral"]["sets"].get(tb.CV_DEV_NAME)
    L += ["", "## cv27-en dev sample (v4, informational)", ""]
    L.append("Not scored yet." if not cv else ", ".join(
        f"{m}: standard {s['wer']['std']:.2f}, style-neutral {s['wer']['all']:.2f}" for m, s in cv.items()) + ".")
    return L


def main() -> int:
    llayout.require_mount()
    r = analyze_all()
    tb.write_json(tb.root() / "results.json", r)
    path = tb.results_md()
    path.write_text(markdown(r))
    print(f"wrote {tb.root() / 'results.json'} and {path}")
    return 0
