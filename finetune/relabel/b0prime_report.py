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
from training import bootstrap, stage3

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
    return {"status": "complete", "bootstrap": res, "decision": decision, "per_set_regression": per_set,
            "domain": {"B0": b0_domain.get("sets"), "B0p": bp_domain.get("sets"), "T*": t_star_domain(),
                       "T*_note": None if t_star_domain() else "no T* domain scores exist (score_teacher.py scores "
                                                               "the stage 1 dev sets only)",
                       "difference_points": domain_diff},
            "gap": gap, "style": style,
            "yodas_dev": {m: 100 * corpus(r["yodas_dev"]) for m, r in recs.items()}}


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
        for k, m in p["modes"].items():
            L.append(f"- {k}: status {m['status']}, peak {_f(m['peak_process_gib_estimate'], 1)} GiB, step "
                     f"{_f(m['step_time_s'], 3)} s, {_f(m['audio_seconds_per_step'], 0)} s of audio per step")
        d, e = p["decision"], p.get("estimate") or {}
        L += [f"- Decision: {d['chosen']} (gradient checkpointing {d['grad_checkpointing']}); lease "
              f"{d['lease_vram_gb']} GB; host peak {_f(p['host_peak_rss_gib'], 1)} GiB.",
              f"- Estimate: pilot {_f(e.get('pilot_gpu_hours'), 1)} GPU-h, main run {_f(e.get('main_gpu_hours'), 1)} "
              "GPU-h (training steps only)."]
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
          "| Model | dev-medical WER | dev-medical term recall | dev-technical WER |", "|---|---:|---:|---:|"]
    for m in ("B0", "B0p", "T*"):
        dm = g["domain"].get(m) or {}
        med, tech = dm.get("dev-medical", {}), dm.get("dev-technical", {})
        rec = med.get("mean_term_recall")
        L.append(f"| {m} | {_f(med.get('mean_wer'))} | {_f(100 * rec if rec is not None else None, 1)} | "
                 f"{_f(tech.get('mean_wer'))} |")
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
    return "\n".join(L) + "\n"


def main() -> int:
    llayout.require_mount()
    r = analyze_all()
    tb.write_json(tb.root() / "results.json", r)
    path = tb.results_md()
    path.write_text(markdown(r))
    print(f"wrote {tb.root() / 'results.json'} and {path}")
    return 0
