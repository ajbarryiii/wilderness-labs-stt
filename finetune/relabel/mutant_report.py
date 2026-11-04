"""Mutant (encoder probe 2) analysis and preregistered readout (mutant plan "Evaluation" and "Readout"): CPU only,
from records and the runs' held-out curves.

    ./python mutant_report.py [--smoke-shards N] [--out PATH] [--replicates N] [--seed S]

Inputs: each arm's records at its final stage (decoder/<arm>/<set>.json after the main run, decoder/<arm>/extension/
after the extension; every arm at the same stage, since the extension applies to all), B0's own records (H4 + YODAS:
relabel_paths.STUDENT_RECORDS["B0"]; T5: $RELABEL_ROOT/devsets/scores/B0 and scores/eval-stem/B0), and each arm's
held-out evaluations (runs/<arm>/main/summary.json, runs/<arm>/extension/summary.json).

Population: per set, records scored (scored = true) in every model; the ids, normalized references and flags must be
identical in all four (else AnalysisError), every record's S, D, I must recompute from ref_norm and hyp_norm
(analyze.realign), and no record file may be a --limit subset. WER is word-weighted within a set, in points.

Panels: H4 (the four human-transcribed dev sets; its mean is unweighted); YODAS reported only; T5 (tech-dev-1,
tech-dev-2, stem-chem, stem-cs, stem-math; unweighted mean of the five) and, v3.1, T5 without tech-dev-2 (a sibling of
T*'s training data; the mean of the other four), both point estimates for every comparison.

Comparisons (CONTRASTS, points): Delta_X = WER(D-B) - WER(D-X) for X in {D-T, D-Q} (positive: X better); WER(D-Q) -
WER(D-T); WER(arm) - WER(B0) for each arm.

Intervals: training/bootstrap.py's paired cluster bootstrap with its defaults (10,000 replicates, 90%, its seed); one
draw per set serves every model. H4: clusters training/devsplit.cluster_of over the full dev sets, the H4 mean
recomputed per replicate. T5: clusters training/t5clusters.cluster_of (the talk, call or lecture), a set's own
interval only with at least 10 clusters (tech-dev-1 has no talk id: none); the T5 means get no interval; YODAS none.
Conditional on the trained checkpoints, one training seed per arm: training-seed variation is not measured. Two
primary comparisons (D-T, D-Q), each at the same margin and interval, no multiplicity correction.

Readout (readout()):
- plateau: the held-out WER improved by at most 0.10 points from round(75% of S) to S (main), or from S to the
  extension's end (extended);
- control recovery: H4 WER(D-B) <= 5.90;
- after the main runs, a failed recovery or any arm not plateaued calls for the single extension of every arm
  ("extension required", no verdict yet);
- with recovery and every arm plateaued (after the main runs, or after the extension): A if Delta_X >= +0.30 with its
  90% interval excluding 0, C if Delta_X <= -0.30 with its interval excluding 0, B otherwise; after the extension,
  anything else is "unresolved", naming the arms;
- each A or C is labeled "consistent on T5" if T5's five-set mean difference WER_T5(D-B) - WER_T5(D-X) has the same sign
  and is at least 0.30 in size, else "H4 only".
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

import analyze as diagnostic
import mutant_data as md
import mutant_train as mt
import relabel_paths
from analyze import AnalysisError
from training import bootstrap, devsplit, t5clusters

ARMS = ("D-B", "D-T", "D-Q")
B0 = "B0"
MODELS = ARMS + (B0,)
PRIMARY = ("D-T", "D-Q")
CONTRASTS = {"D-T": ("D-B", "D-T"), "D-Q": ("D-B", "D-Q"), "D-Q_minus_D-T": ("D-Q", "D-T"),
             "D-B_minus_B0": ("D-B", B0), "D-T_minus_B0": ("D-T", B0), "D-Q_minus_B0": ("D-Q", B0)}
THRESHOLD = 0.30
PLATEAU_MAX_GAIN = 0.10
RECOVERY_MAX = 5.90           # B0's H4 5.60 + 0.30
EPS = 1e-9
H4 = list(relabel_paths.H4)
YODAS = relabel_paths.YODAS
T5 = list(t5clusters.SETS)
ALL_SETS = H4 + [YODAS] + T5
REPLICATES, SEED, CI = bootstrap.REPLICATES, bootstrap.SEED, bootstrap.CI


# --- rules ------------------------------------------------------------------------------------------------------

def _wer_at(evals: list[dict], step: int) -> float:
    for e in evals:
        if e["step"] == step:
            return float(e["wer"])
    raise AnalysisError(f"no held-out evaluation at step {step}")


def plateau(curve: dict, extended: bool) -> dict:
    """curve: {"S": S, "main": evals, "extension": evals or None}."""
    s = int(curve["S"])
    if extended:
        if not curve.get("extension"):
            raise AnalysisError("extended, but no extension evaluation")
        a, b = s, int(curve["extension"][-1]["step"])
        wa, wb = _wer_at(curve["main"], a), _wer_at(curve["extension"], b)
    else:
        a, b = round(mt.KEEP_FRACTION * s), s
        wa, wb = _wer_at(curve["main"], a), _wer_at(curve["main"], b)
    gain = wa - wb
    return {"from_step": a, "to_step": b, "from_wer": wa, "to_wer": wb, "improvement": gain,
            "plateaued": gain <= PLATEAU_MAX_GAIN + EPS}


def recovered(h4_db: float) -> bool:
    return h4_db <= RECOVERY_MAX + EPS


def verdict(delta: float, ci: list[float]) -> str:
    excludes = ci[0] > 0 or ci[1] < 0
    if delta >= THRESHOLD - EPS and excludes:
        return "A"
    if delta <= -THRESHOLD + EPS and excludes:
        return "C"
    return "B"


def t5_label(label: str, t5_delta: float) -> str | None:
    if label not in ("A", "C"):
        return None
    same_sign = (t5_delta > 0) if label == "A" else (t5_delta < 0)
    return "consistent on T5" if same_sign and abs(t5_delta) >= THRESHOLD - EPS else "H4 only"


MEANING = {"A": "X's features give a materially better Parakeet decoder on H4 (this recipe and budget)",
           "B": "no material difference on H4 under this recipe and budget",
           "C": "X's features give a materially worse Parakeet decoder on H4 (this recipe and budget)",
           "unresolved": "control recovery or a plateau failed after the extension"}


def readout(state: dict) -> dict:
    """state: extended, recovery, plateaued {arm: bool}, delta {X: (point, ci)}, t5_delta {X: point}."""
    failed = {"control_recovery": [] if state["recovery"] else ["D-B"],
              "plateau": [a for a in ARMS if not state["plateaued"][a]]}
    ok = not failed["control_recovery"] and not failed["plateau"]
    if not ok and not state["extended"]:
        return {"status": "extension required", "verdicts": None, "failed": failed,
                "note": "apply the single extension (1 epoch at a constant 10% of each arm's rate) to every arm",
                "inputs": state}
    verdicts = {}
    for x in PRIMARY:
        point, ci = state["delta"][x]
        label = verdict(point, ci) if ok else "unresolved"
        verdicts[x] = {"label": label, "meaning": MEANING[label], "delta_h4": point, "ci90": ci,
                       "t5_delta": state["t5_delta"][x], "t5": t5_label(label, state["t5_delta"][x])}
    return {"status": "final" if ok else "unresolved", "verdicts": verdicts, "failed": failed, "inputs": state,
            "note": "A or C labeled 'H4 only' goes to AJ with the per-set numbers; it does not justify a distillation "
                    "plan by itself"}


# --- inputs -----------------------------------------------------------------------------------------------------

def load_records(dirs: dict[str, Path], b0_dirs: dict[str, Path]) -> tuple[dict, dict]:
    models, files = {m: {} for m in MODELS}, {m: {} for m in MODELS}
    for m in MODELS:
        for s in ALL_SETS:
            path = Path(b0_dirs[s] if m == B0 else dirs[m]) / f"{s}.json"
            if not path.exists():
                raise AnalysisError(f"missing records {path}")
            body = json.loads(path.read_text())
            if body.get("limit") is not None:
                raise AnalysisError(f"{path} is a --limit subset")
            models[m][s] = body["records"]
            files[m][s] = {"path": str(path), "sha256": diagnostic.sha256_file(path)}
    for s in ALL_SETS:
        key = [(r["id"], r["ref_norm"], bool(r["scored"])) for r in models[B0][s]]
        if len({k[0] for k in key}) != len(key):
            raise AnalysisError(f"B0 {s}: duplicate ids")
        for m in ARMS:
            if [(r["id"], r["ref_norm"], bool(r["scored"])) for r in models[m][s]] != key:
                raise AnalysisError(f"{m} {s}: ids, ref_norm or scored differ from B0's records")
        for m in MODELS:
            diagnostic.realign(f"{m} {s}", models[m][s], check_recorded=True)
    return models, files


def counts(rows: list[dict]) -> tuple[int, int]:
    scored = [r for r in rows if r["scored"]]
    return sum(r["S"] + r["D"] + r["I"] for r in scored), sum(len(r["ref_norm"].split()) for r in scored)


def _cluster_fn(s: str):
    if s in H4:
        return lambda i: devsplit.cluster_of(s, i)
    return lambda i: t5clusters.cluster_of(s, i)


def _ci(x) -> list[float] | None:
    return None if x is None else [float(v) for v in x]


# --- analysis ---------------------------------------------------------------------------------------------------

def analyze(dirs: dict[str, Path], b0_dirs: dict[str, Path], curves: dict[str, dict], extended: bool,
            replicates: int = REPLICATES, seed: int = SEED) -> dict:
    models, files = load_records(dirs, b0_dirs)
    wer = {s: {} for s in ALL_SETS}
    for s in ALL_SETS:
        for m in MODELS:
            e, w = counts(models[m][s])
            wer[s][m] = 100.0 * e / w
    h4_tabs = {s: bootstrap.SetTable.from_records({m: models[m][s] for m in MODELS}, cluster_of=_cluster_fn(s))
               for s in H4}
    h4_boot = bootstrap.paired(h4_tabs, CONTRASTS, H4, replicates=replicates, seed=seed)
    t5_info = t5clusters.interval_sets({s: [r["id"] for r in models[B0][s] if r["scored"]] for s in T5})
    t5_boot_sets = [s for s in T5 if t5_info[s]["interval"]]
    t5_boot = None
    if t5_boot_sets:
        tabs = {s: bootstrap.SetTable.from_records({m: models[m][s] for m in MODELS}, cluster_of=_cluster_fn(s))
                for s in t5_boot_sets}
        t5_boot = bootstrap.paired(tabs, CONTRASTS, t5_boot_sets, replicates=replicates, seed=seed)

    sets = {}
    for s in ALL_SETS:
        boot = h4_boot if s in H4 else (t5_boot if s in t5_boot_sets else None)
        clusters = (len(h4_tabs[s].clusters) if s in H4 else t5_info[s]["clusters"] if s in T5 else None)
        e, w = counts(models[B0][s])
        sets[s] = {"wer": wer[s], "utterances": len(models[B0][s]),
                   "scored": sum(r["scored"] for r in models[B0][s]), "reference_words": w, "clusters": clusters,
                   "interval_rule": ("H4: devsplit clusters" if s in H4 else
                                     (t5_info[s]["reason"] or "T5: at least 10 clusters") if s in T5 else
                                     "YODAS: no cluster definition, no interval"),
                   "contrasts": {c: {"point": wer[s][a] - wer[s][b],
                                     "ci90": _ci(boot["sets"][s][c]["ci"]) if boot else None}
                                 for c, (a, b) in CONTRASTS.items()}}

    def panel(names: list[str], boot=None) -> dict:
        w = {m: float(np.mean([wer[s][m] for s in names])) for m in MODELS}
        return {"sets": list(names), "wer": w,
                "contrasts": {c: {"point": w[a] - w[b], "ci90": _ci(boot["mean"][c]["ci"]) if boot else None}
                              for c, (a, b) in CONTRASTS.items()}}

    h4 = panel(H4, h4_boot)
    t5 = panel(T5)
    t5_wo = panel(list(t5clusters.SETS_WITHOUT))
    plateaus = {a: plateau(curves[a], extended) for a in ARMS}
    state = {"extended": extended, "recovery": recovered(h4["wer"]["D-B"]), "h4_db": h4["wer"]["D-B"],
             "plateaued": {a: plateaus[a]["plateaued"] for a in ARMS},
             "delta": {x: (h4["contrasts"][x]["point"], h4["contrasts"][x]["ci90"]) for x in PRIMARY},
             "t5_delta": {x: t5["contrasts"][x]["point"] for x in PRIMARY}}
    return {
        "plan": md.PLAN, "created": dt.datetime.now(dt.timezone.utc).isoformat(), "inputs": files,
        "bootstrap": {"replicates": replicates, "seed": seed, "ci": CI, "method": "training/bootstrap.py paired",
                      "h4_clusters": {s: len(h4_tabs[s].clusters) for s in H4},
                      "t5_clusters": {s: t5_info[s]["clusters"] for s in T5}, "t5_interval_sets": t5_boot_sets,
                      "per_set_stream": "SeedSequence([seed, index of the set in the call's sorted set names])",
                      "conditional": "on the trained checkpoints; one training seed per arm"},
        "sets": sets, "h4": h4, "ami": sets["ami_dev"], "tech_dev_2": sets["tech-dev-2"], "yodas": sets[YODAS],
        "t5": t5, "t5_without_tech_dev_2": t5_wo, "plateau": plateaus,
        "control_recovery": {"h4_db": h4["wer"]["D-B"], "max": RECOVERY_MAX, "recovered": state["recovery"]},
        "readout": readout(state),
        "multiplicity": "two primary comparisons at the same margin and interval, no correction",
    }


# --- production inputs ------------------------------------------------------------------------------------------

def b0_dirs() -> dict[str, Path]:
    scores = relabel_paths.ARTIFACTS / "devsets" / "scores"
    out = {s: relabel_paths.STUDENT_RECORDS[B0] for s in H4 + [YODAS]}
    out.update({"tech-dev-1": scores / "B0", "tech-dev-2": scores / "B0"})
    out.update({s: scores / "eval-stem" / "B0" for s in ("stem-chem", "stem-cs", "stem-math")})
    return out


def curves_of(lay: md.Layout) -> tuple[dict, bool]:
    out, extended = {}, []
    for a in ARMS:
        main = mt.run_dir(lay, a, "main") / "summary.json"
        if not main.exists():
            raise AnalysisError(f"{a}: the main run is not complete")
        cfg = json.loads((mt.run_dir(lay, a, "main") / "config.json").read_text())["config"]
        ext = mt.run_dir(lay, a, "extension") / "summary.json"
        out[a] = {"S": cfg["schedule"]["S"], "main": json.loads(main.read_text())["evals"],
                  "extension": json.loads(ext.read_text())["evals"] if ext.exists() else None}
        extended.append(ext.exists())
    if len(set(extended)) != 1:
        raise AnalysisError("the extension applies to every arm together: some arms are extended, some not")
    return out, extended[0]


def pilots(lay: md.Layout) -> dict:
    out = {}
    for a in ARMS:
        try:
            res = mt.pilot_results(lay, a)
            out[a] = {"heldout_at_end": {mt.LR_NAMES[k]: v for k, v in res.items()},
                      "selected": mt.LR_NAMES[mt.select_lr(res)]}
        except SystemExit as exc:
            out[a] = {"missing": str(exc)}
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke-shards", type=int)
    p.add_argument("--out", type=Path)
    p.add_argument("--replicates", type=int, default=REPLICATES)
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args(argv)
    lay = md.layout(a.smoke_shards)
    out = a.out or (lay.records / "report.json")
    production = not lay.smoke and a.replicates == REPLICATES and a.seed == SEED
    if out.resolve() == (md.layout().records / "report.json").resolve():
        relabel_paths.require_mount()
        if not production:
            raise SystemExit("refusing to write the production report from smoke inputs or other settings")
    try:
        curves, extended = curves_of(lay)
        stage = "extension" if extended else "main"
        dirs = {arm: mt.records_dir(lay, arm, stage) for arm in ARMS}
        report = analyze(dirs, b0_dirs(), curves, extended, a.replicates, a.seed)
    except AnalysisError as exc:
        print(f"ANALYSIS STOPPED: {exc}", file=sys.stderr)
        return 2
    report["pilots"] = pilots(lay)
    report["stage"] = stage
    report["held_out_curves"] = curves
    probe1 = relabel_paths.ARTIFACTS / "encoder-probe" / "ctc" / "report.json"
    if probe1.exists():
        r1 = json.loads(probe1.read_text())
        report["probe1"] = {"readout": r1.get("readout"), "h4_delta_enc": r1.get("h4", {}).get("delta_enc")}
    md.write_json(out, report)
    r = report["readout"]
    print(f"readout: {r['status']}; " + (", ".join(f"{x}: {v['label']} ({v['t5']})" for x, v in r["verdicts"].items())
                                          if r["verdicts"] else json.dumps(r["failed"])))
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
