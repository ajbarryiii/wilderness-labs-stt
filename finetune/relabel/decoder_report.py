"""Results of stage 2b (decoder-only plan "Results (preregistered)"): results.json and RESULTS-decoder-only.md.

    ./python train_decoder.py report [--out PATH]      (CPU; reads $RELABEL_ROOT/decoder-only/)

Selection (selection part of the frozen split): for each arm, among its three learning rates (all must be
finished), the run with the lowest H4; runs within NEAR_TIE points of the lowest go to the lower learning rate.

Per student, on the evaluation part only, with O the original student (original-<s>/), C the selected control run
(B0 labels) and A the selected T* run:
    L = WER(C) - WER(A)     label effect
    R = WER(O) - WER(C)     refit effect
    L + R = WER(O) - WER(A) total
per H4 set and for H4, each with its 90% paired cluster-bootstrap interval (training/bootstrap.py, 10,000
replicates; clusters as training/devsplit.py). Claims, mechanically:
    1. the new labels help through the decoder: L(H4) > 0 and its interval excludes 0;
    2. a decoder-only accuracy candidate (pending latency and energy): total(H4) >= 0.10 points, its interval
       excludes 0, and on no H4 set is A more than 0.25 points worse than O.
Also reported, without a threshold: dev-medical and dev-technical WER and critical-term recall (score_domain.py
settings) for O, C and A; the decoder-step proxy on full dev; the share of the stage 1 B0-to-T* gap (evaluation
part) that A closes, (WER(O) - WER(A)) / (WER(B0) - WER(T*)); style rates (analyze.style) on the evaluation part;
YODAS dev (full set, informational); every run's selection-part H4 and training counts.
"""
from __future__ import annotations

import json
from pathlib import Path

import analyze
import relabel_paths
import train_decoder as td
from training import bootstrap, devsplit

NEAR_TIE = 0.05
CANDIDATE_TOTAL = 0.10
MAX_SET_REGRESSION = 0.25
EPS = 1e-9
T_STAR = "granite-speech-5.0-2b"
CONTRASTS = {"L": ("C", "A"), "R": ("O", "C"), "total": ("O", "A")}


def _json(path: Path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def records(model_dir: Path, name: str) -> list[dict]:
    return json.loads((Path(model_dir) / "eval-dev" / f"{name}.json").read_text())["records"]


def select(arm: str, split: devsplit.Split) -> dict:
    pending = [td.LR_NAMES[x] for x in td.LRS if not (td.run_dir(arm, x) / "summary.json").exists()]
    if pending:
        return {"arm": arm, "selected": None, "pending": pending}
    runs = {lr: td.part_wer(td.run_dir(arm, lr), split, "selection")["h4"] for lr in td.LRS}
    best = min(runs.values())
    chosen = min(lr for lr, v in runs.items() if v <= best + NEAR_TIE + EPS)
    return {"arm": arm, "selected": chosen, "selected_name": td.LR_NAMES[chosen],
            "selection_h4": {td.LR_NAMES[lr]: v for lr, v in runs.items()},
            "rule": f"lowest selection-part H4; within {NEAR_TIE} points of it, the lowest learning rate"}


def eval_tables(models: dict[str, list], split: devsplit.Split, part: str = "evaluation") -> dict:
    """{set: SetTable} of the given per-set records ({model: {set: records}}) on one part of the split."""
    out = {}
    for name in devsplit.H4:
        out[name] = bootstrap.SetTable.from_records({m: recs[name] for m, recs in models.items()},
                                                    cluster_of=lambda i, n=name: devsplit.cluster_of(n, i),
                                                    ids=split.ids(name, part))
    return out


def stage1_records(name: str) -> dict[str, list]:
    b0 = json.loads((relabel_paths.STUDENT_RECORDS["B0"] / f"{name}.json").read_text())["records"]
    ts = json.loads((relabel_paths.DIAGNOSTIC / T_STAR / f"{name}.json").read_text())["records"]
    return {"B0": b0, "T*": ts}


def claims(res: dict) -> dict:
    """The preregistered claims from a bootstrap result with contrasts L, R and total (points; see the docstring)."""
    L, total = res["mean"]["L"], res["mean"]["total"]
    worst = min(res["sets"][n]["total"]["point"] for n in res["mean_sets"])
    return {"labels_help_through_decoder": L["point"] > EPS and L["ci"][0] > EPS,
            "decoder_only_candidate": (total["point"] >= CANDIDATE_TOTAL - EPS and total["ci"][0] > EPS
                                       and -worst <= MAX_SET_REGRESSION + EPS),
            "worst_set_total_points": worst,
            "rules": {"claim1": "L(H4) > 0 and its 90% interval excludes 0",
                      "claim2": f"total(H4) >= {CANDIDATE_TOTAL}, interval excludes 0, no H4 set more than "
                                f"{MAX_SET_REGRESSION} points worse than the original student"}}


def student_result(s: str, split: devsplit.Split, selections: dict) -> dict:
    t_arm, c_arm = f"{s}-T", f"{s}-0"
    out = {"student": s, "name": td.STUDENTS[s]["name"], "t_arm": t_arm, "control_arm": c_arm}
    o_dir = td.original_dir(s)
    if not _json(o_dir / "scores.json"):
        return out | {"status": "original student not scored"}
    if selections[t_arm]["selected"] is None or selections[c_arm]["selected"] is None:
        return out | {"status": "runs pending"}
    dirs = {"O": o_dir, "C": td.run_dir(c_arm, selections[c_arm]["selected"]),
            "A": td.run_dir(t_arm, selections[t_arm]["selected"])}
    recs = {m: {n: records(d, n) for n in devsplit.H4} for m, d in dirs.items()}
    res = bootstrap.paired(eval_tables(recs, split), CONTRASTS, mean_sets=list(devsplit.H4))
    stage1 = {n: stage1_records(n) for n in devsplit.H4}
    gap_models = {"B0": {n: stage1[n]["B0"] for n in devsplit.H4}, "T*": {n: stage1[n]["T*"] for n in devsplit.H4},
                  "O": recs["O"], "A": recs["A"]}
    gt = eval_tables(gap_models, split)
    gap = {}
    for n in list(devsplit.H4) + ["H4"]:
        if n == "H4":
            w = {m: sum(gt[x].wer(m) for x in devsplit.H4) / len(devsplit.H4) for m in gap_models}
        else:
            w = {m: gt[n].wer(m) for m in gap_models}
        g = w["B0"] - w["T*"]
        gap[n] = {"wer": w, "gap_points": g, "closed_share": (w["O"] - w["A"]) / g if abs(g) > EPS else None}
    style = {}
    for n in devsplit.H4:
        ids = split.ids(n, "evaluation")
        sub = {m: [r for r in recs[m][n] if r["id"] in ids] for m in recs}
        sub |= {m: [r for r in stage1[n][m] if r["id"] in ids] for m in ("B0", "T*")}
        style[n] = analyze.style(sub)
    domain = {m: _json(d / "domain" / "summary.json") for m, d in dirs.items()}
    proxy = {m: _json(d / "proxy" / "summary.json") for m, d in dirs.items()}
    yodas = {m: json.loads((d / "eval-dev" / "yodas_dev.json").read_text())["wer"]["wer"] * 100 for m, d in dirs.items()}
    return out | {"status": "complete", "dirs": {m: str(d.relative_to(relabel_paths.ARTIFACTS)) for m, d in dirs.items()},
                  "selected": {"A": selections[t_arm]["selected_name"], "C": selections[c_arm]["selected_name"]},
                  "bootstrap": res, "claims": claims(res),
                  "gap": gap, "style": style, "yodas_dev_full_wer": yodas,
                  "domain": {m: (d or {}).get("sets") for m, d in domain.items()},
                  "domain_parts": {m: {p: {"wer": e["wer"], "term_recall": e.get("term_recall")}
                                       for p, e in (d or {}).get("parts", {}).items()} for m, d in domain.items()},
                  "proxy": {m: p for m, p in proxy.items()}}


def run_table() -> list[dict]:
    rows = []
    for arm in td.ARMS:
        for lr in td.LRS:
            s = _json(td.run_dir(arm, lr) / "summary.json")
            if s is None:
                rows.append({"arm": arm, "lr": td.LR_NAMES[lr], "status": "pending"})
                continue
            d = s.get("decoder_only", {})
            rows.append({"arm": arm, "lr": td.LR_NAMES[lr], "status": "done", "steps": s["steps"],
                         "consumed_batches": d.get("consumed_batches"),
                         "fully_rejected_batches": d.get("fully_rejected_batches"),
                         "selection_h4": d.get("selection_part", {}).get("h4"),
                         "dev_mean_wer_full": 100 * s["dev_mean_wer"], "audio_hours": s.get("audio_hours_seen"),
                         "encoder_bit_identical": d.get("encoder_bit_identical"),
                         "proxy": d.get("proxy"), "resumes": s.get("resumes")})
    return rows


def analyze_all() -> dict:
    split = devsplit.load(td.split_path(), devsplit.dev_manifests())
    selections = {arm: select(arm, split) for arm in td.ARMS}
    return {"plan": "decoder-only plan", "split": {"sha256": split.sha256, "counts": split.counts(),
                                                        "rule": split.data["rule"]},
            "selection": selections, "students": {s: student_result(s, split, selections) for s in td.STUDENTS},
            "runs": run_table(),
            "equivalence": {s: _json(td.root() / "equivalence" / f"{s}.json") for s in td.STUDENTS},
            "probe": {s: _json(td.root() / "probe" / f"{s}.json") for s in td.STUDENTS},
            "originals": {s: (_json(td.original_dir(s) / "scores.json") or {}).get("crosscheck") for s in td.STUDENTS}}


def _f(x, nd=2):
    return "–" if x is None else f"{x:.{nd}f}"


def _ci(e):
    return f"{e['point']:+.2f} [{e['ci'][0]:+.2f}, {e['ci'][1]:+.2f}]"


def markdown(r: dict) -> str:
    L = ["# Stage 2b results: decoder-only fine-tuning of Kea-B and Kea-T", "",
         "Generated by `train_decoder.py report` from `$RELABEL_ROOT/decoder-only/` (decoder-only plan). "
         "WER in points; intervals are 90% paired cluster-bootstrap intervals on the evaluation part "
         "(10,000 replicates). These are development results (stage 1 already used full dev to choose T*).", "",
         "## Dev split", "", "| Set | Part | Clusters | Utterances | Reference words |", "|---|---|---:|---:|---:|"]
    for n, parts in r["split"]["counts"].items():
        for p, c in parts.items():
            L.append(f"| {n} | {p} | {c['cluster_count']} | {c['utterances']} | {c['reference_words']} |")
    L += ["", f"Split sha256 `{r['split']['sha256'][:16]}`. AMI has 18 meetings: its interval is wide.", "",
          "## Learning-rate selection (selection part, H4)", "", "| Arm | lr1e-5 | lr3e-5 | lr1e-4 | Selected |",
          "|---|---:|---:|---:|---|"]
    for arm, sel in r["selection"].items():
        h = sel.get("selection_h4", {})
        L.append(f"| {arm} | " + " | ".join(_f(h.get(td.LR_NAMES[lr]), 3) for lr in td.LRS)
                 + f" | {sel.get('selected_name') or 'pending: ' + ', '.join(sel.get('pending', []))} |")
    for s, res in r["students"].items():
        L += ["", f"## {res['name']} ({res['t_arm']} against {res['control_arm']})", ""]
        if res.get("status") != "complete":
            L.append(f"Status: {res.get('status')}.")
            continue
        b = res["bootstrap"]
        L += [f"Selected: T* arm {res['selected']['A']}, control {res['selected']['C']}.", "",
              "| Set | WER O | WER C | WER A | L = C − A | R = O − C | L + R = O − A |", "|---|---:|---:|---:|---:|---:|---:|"]
        for n in list(devsplit.H4) + ["H4"]:
            w = b["wer"]["mean"] if n == "H4" else b["wer"][n]
            e = b["mean"] if n == "H4" else b["sets"][n]
            L.append(f"| {n} | {w['O']:.2f} | {w['C']:.2f} | {w['A']:.2f} | {_ci(e['L'])} | {_ci(e['R'])} | "
                     f"{_ci(e['total'])} |")
        c = res["claims"]
        L += ["", f"- Claim 1 (the new labels help through the decoder): **{'yes' if c['labels_help_through_decoder'] else 'no'}** "
                  f"({c['rules']['claim1']}).",
              f"- Claim 2 (decoder-only accuracy candidate, pending latency and energy): "
              f"**{'yes' if c['decoder_only_candidate'] else 'no'}** ({c['rules']['claim2']}; worst set total "
              f"{c['worst_set_total_points']:+.2f}).", "",
              "Domain sets (score_domain.py settings): mean WER (unweighted over parts) and mean critical-term recall.", "",
              "| Model | dev-medical WER | dev-medical term recall | dev-technical WER |", "|---|---:|---:|---:|"]
        for m in ("O", "C", "A"):
            d = res["domain"].get(m) or {}
            med, tech = d.get("dev-medical", {}), d.get("dev-technical", {})
            L.append(f"| {m} | {_f(med.get('mean_wer'))} | {_f(100 * med['mean_term_recall'] if med.get('mean_term_recall') is not None else None, 1)} "
                     f"| {_f(tech.get('mean_wer'))} |")
        L += ["", "Decoder-step proxy on full dev (pooled over the five stage 1 dev sets):", "",
              "| Model | Tokens / s audio | Decoder iterations / s audio | Mean predicted duration (frames) | NeMo token agreement |",
              "|---|---:|---:|---:|---:|"]
        for m in ("O", "C", "A"):
            p = (res["proxy"].get(m) or {}).get("pooled") or {}
            L.append(f"| {m} | {_f(p.get('tokens_per_second'), 3)} | {_f(p.get('iterations_per_second'), 3)} | "
                     f"{_f(p.get('mean_predicted_duration'), 3)} | {p.get('nemo_token_agreement')}/{p.get('utterances')} |")
        L += ["", "Share of the stage 1 B0-to-T* gap closed by A (evaluation part): "
              + ", ".join(f"{n} {_f(g['closed_share'] if g['closed_share'] is None else 100 * g['closed_share'], 0)}%"
                          for n, g in res["gap"].items()) + ".", "",
              "Style (share of scored hypotheses with punctuation / with an uppercase letter, evaluation part):", "",
              "| Set | O | C | A | B0 | T* |", "|---|---|---|---|---|---|"]
        for n, st in res["style"].items():
            L.append(f"| {n} | " + " | ".join(f"{100 * st[m]['punctuation_share']:.0f} / {100 * st[m]['uppercase_share']:.0f}"
                                               for m in ("O", "C", "A", "B0", "T*")) + " |")
        L += ["", "YODAS dev (full set, informational): " + ", ".join(f"{m} {v:.2f}" for m, v in res["yodas_dev_full_wer"].items()) + "."]
    L += ["", "## Runs", "", "| Arm | lr | Steps | Consumed batches | Fully rejected | Selection H4 | Full dev mean |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for row in r["runs"]:
        L.append(f"| {row['arm']} | {row['lr']} | {row.get('steps', '–')} | {row.get('consumed_batches', '–')} | "
                 f"{row.get('fully_rejected_batches', '–')} | {_f(row.get('selection_h4'), 3)} | "
                 f"{_f(row.get('dev_mean_wer_full'), 3)} |")
    return "\n".join(L) + "\n"


def main(out: Path | None = None) -> int:
    relabel_paths.require_mount()
    r = analyze_all()
    td.write_json(td.root() / "results.json", r)
    path = Path(out) if out else td.root() / "RESULTS-decoder-only.md"
    path.write_text(markdown(r))
    print(f"wrote {td.root() / 'results.json'} and {path}")
    return 0
