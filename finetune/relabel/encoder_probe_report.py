"""Analysis of encoder probe 1 (encoder-probe-ctc plan "Analysis" v2.1 and "Readout"): CPU only, from records.

    ./python encoder_probe_report.py [--probe DIR] [--b0 DIR] [--tstar DIR] [--out PATH]
                                     [--replicates N] [--seed S]

Reads the probe's records (encoder_probe_ctc.py: <probe>/<head>/<set>.json, a complete run whose implementation
checks 1-3 passed), B0's dev records and T*'s full-system records from the teacher diagnostic, and writes
report.json next to the probe's records.

Population: records with scored = true (identical in every model: same ids, normalized references and flags;
checked), and their reference words. Per set and on the H4 mean (unweighted over the four human-transcribed sets):

    WER(m)       word-weighted, points
    Delta_enc(h) WER(B0) - WER(h); positive: T*'s encoder alone, through head h, beats B0's full system
    s(h)         (WER(h) - WER(T*)) / (WER(B0) - WER(T*)), descriptive (it does not isolate the projector and LLM:
                 the probe encodes one clip per pass, the T* records padded batches); on H4 always, per set only
                 where |WER(B0) - WER(T*)| >= 0.30

Intervals: the paired cluster bootstrap of training/bootstrap.py with its defaults (10,000 replicates, 90%, its seed),
clusters from training/devsplit.cluster_of (LibriSpeech speaker, AMI meeting, VoxPopuli plenary session), drawn over
the full dev sets. One draw per set serves every model; the H4 mean is recomputed per replicate from the same draws;
s is computed per replicate from that replicate's WERs (replicate_wers repeats bootstrap.paired's seeded draws, which
the contract tests check). YODAS has no clusters: point values only.

Checks: 4 (librispeech_dev_clean WER at most 10% for bpe and 25% for the half-depth char-mid head; otherwise the
readout is withheld and 20 example hypotheses are reported) and 5 (every record's S, D, I recompute from ref_norm and
hyp_norm, analyze.realign; a mismatch stops the analysis). Checks 1-3 come from the probe's summary.

Readout (preregistered, primary head bpe, H4): A if Delta_enc >= +0.30 with its 90% interval excluding 0; C if
Delta_enc <= -0.30 with its interval excluding 0; B otherwise.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

import analyze as diagnostic  # the teacher diagnostic's analysis (load_model, teacher_state, realign, ...)
import encoder_probe_ctc as ep
import relabel_paths
from analyze import AnalysisError
from training import bootstrap, devsplit

B0, TSTAR = "B0", "T*"
HEADS = ep.HEADS
MODELS = (B0, TSTAR) + HEADS
PRIMARY = "bpe"
THRESHOLD = 0.30        # readout margin, points (the diagnostic's teacher margin)
S_MIN_GAIN = 0.30       # per-set s only where |WER(B0) - WER(T*)| >= this
CHAR, BPE = ep.CHAR, ep.BPE
GROSS_WER_MAX = {BPE: 10.0, CHAR: 25.0}  # check 4, librispeech_dev_clean, points (plan v2.2: char-mid is half depth)
GROSS_SET = "librispeech_dev_clean"
EXAMPLES = 20
EPS = 1e-9
REPLICATES, SEED, CI = bootstrap.REPLICATES, bootstrap.SEED, bootstrap.CI
REPORT = ep.OUT / "report.json"
TSTAR_RECORDS = relabel_paths.DIAGNOSTIC / ep.TEACHER
DEFINITIONS = {
    "population": "records with scored=true (identical across models); reference words of those records",
    "wer": "points: 100 * (S+D+I) / reference words, word-weighted within a set; H4 the unweighted mean of its sets",
    "delta_enc": "points: WER(B0) - WER(head); positive means the encoder alone through that head beats B0",
    "s": ("(WER(head) - WER(T*)) / (WER(B0) - WER(T*)); descriptive: the probe encodes one clip per pass, the T* "
          "records padded batches, so s does not isolate the projector and LLM; per set only where "
          "|WER(B0) - WER(T*)| >= 0.30"),
    "char_mid_minus_bpe": "points: WER(char-mid) - WER(bpe)",
    "char-mid": "the character head on the middle-layer logits (8 of 16 layers): a half-depth lower bound, no gate",
    "rates": "points: 100 * S, D or I / reference words",
    "empty_hypothesis_rate": "share of scored records whose normalized hypothesis is empty",
    "intervals": ("90% paired cluster bootstrap (training/bootstrap.py defaults), clusters from "
                  "training/devsplit.cluster_of, full dev sets; one draw per set for every model; H4 and s per "
                  "replicate; YODAS none"),
}


# --- inputs ---------------------------------------------------------------------------------------------------

def probe_state(directory: Path, sets: list[str]) -> dict:
    """The probe's summary, if it is a complete, non-smoke run whose checks 1-3 passed (else AnalysisError)."""
    path = Path(directory) / "summary.json"
    if not path.exists():
        raise AnalysisError(f"no {path}")
    summary = json.loads(path.read_text())
    if summary.get("status") != "scored" or not summary.get("complete"):
        raise AnalysisError(f"probe run status {summary.get('status')!r}, not a complete scored run")
    config = summary.get("config") or {}
    if config.get("limit") is not None:
        raise AnalysisError("probe run is a --limit smoke run")
    if not (summary.get("checks") or {}).get("passed"):
        raise AnalysisError("probe implementation checks 1-3 did not pass")
    for head in HEADS:
        for name in sets:
            f = Path(directory) / head / f"{name}.json"
            if not f.exists():
                raise AnalysisError(f"missing {f}")
            if json.loads(f.read_text()).get("config") != config:
                raise AnalysisError(f"{f} was written with another configuration than summary.json")
    return summary


def load_all(dirs: dict[str, Path], sets: list[str]) -> tuple[dict, dict, dict]:
    summary = probe_state(dirs["probe"], sets)
    ok, reason, _ = diagnostic.teacher_state(Path(dirs[TSTAR]), sets)
    if not ok:
        raise AnalysisError(f"T* records: {reason}")
    models, inputs = {}, {}
    for m, directory in ((B0, dirs[B0]), (TSTAR, dirs[TSTAR])):
        models[m], inputs[m] = diagnostic.load_model(Path(directory), sets)
    for head in HEADS:
        models[head], inputs[head] = diagnostic.load_model(Path(dirs["probe"]) / head, sets)
    diagnostic.check_consistent(models, sets)
    return models, inputs, summary


# --- quantities -----------------------------------------------------------------------------------------------

def counts(records: list[dict]) -> dict:
    scored = [r for r in records if r["scored"]]
    words = sum(len(r["ref_norm"].split()) for r in scored)
    return {"utterances": len(scored), "words": words, "S": sum(r["S"] for r in scored),
            "D": sum(r["D"] for r in scored), "I": sum(r["I"] for r in scored),
            "empty": sum(not r["hyp_norm"].split() for r in scored)}


def _wer(c: dict) -> float:
    return 100.0 * (c["S"] + c["D"] + c["I"]) / c["words"]


def s_value(head: float, tstar: float, b0: float) -> float:
    return (head - tstar) / (b0 - tstar)


def s_eligible(b0: float, tstar: float) -> bool:
    return abs(b0 - tstar) >= S_MIN_GAIN - EPS


def tables(models: dict[str, dict[str, list[dict]]], sets: list[str]) -> dict[str, bootstrap.SetTable]:
    """bootstrap.SetTable per set on the full dev population, clusters as training/devsplit.cluster_of."""
    return {n: bootstrap.SetTable.from_records({m: models[m][n] for m in models},
                                               cluster_of=lambda i, n=n: devsplit.cluster_of(n, i)) for n in sets}


def replicate_wers(tabs: dict[str, bootstrap.SetTable], models: list[str], replicates: int = REPLICATES,
                   seed: int = SEED) -> dict[str, dict[str, np.ndarray]]:
    """{set: {model: [replicates] WER}} from bootstrap.paired's draws (same per-set seeded stream, same order)."""
    out = {}
    for s_index, name in enumerate(sorted(tabs)):
        t = tabs[name]
        rng = np.random.default_rng(np.random.SeedSequence([seed, s_index]))
        n = len(t.clusters)
        draws = np.empty((replicates, n), dtype=np.float64)
        for b in range(replicates):
            draws[b] = np.bincount(rng.integers(0, n, n), minlength=n)
        words = draws @ t.words
        out[name] = {m: 100.0 * (draws @ t.errors[m]) / words for m in models}
    return out


def _ci(samples: np.ndarray) -> list[float] | None:
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return None
    return bootstrap._ci(finite, CI)


def readout(delta: float, ci: list[float]) -> str:
    """The preregistered rule on Delta_enc(bpe) over H4 (plan "Readout")."""
    lo, hi = ci
    excludes_zero = lo > 0 or hi < 0
    if delta >= THRESHOLD - EPS and excludes_zero:
        return "A"
    if delta <= -THRESHOLD + EPS and excludes_zero:
        return "C"
    return "B"


READOUT_TEXT = {"A": "T*'s encoder alone beats B0: encoder distillation looks promising",
                "B": "inconclusive",
                "C": "greedy BPE CTC underperforms B0 (says nothing on encoder vs decoder; probe 2 holds the decoder "
                     "fixed)"}


def gross_sanity(models: dict, sets: list[str]) -> dict:
    """Check 4: each head's librispeech_dev_clean WER is at most 10% (a gross alarm, not a validation)."""
    out = {"set": GROSS_SET, "max_wer": dict(GROSS_WER_MAX), "wer": {}, "heads": {}}
    for head in HEADS:
        rows = models[head][GROSS_SET]
        w = _wer(counts(rows))
        out["wer"][head] = w
        out["heads"][head] = {"passed": w <= GROSS_WER_MAX[head] + EPS, "examples": [
            {"id": r["id"], "ref": r["ref"], "hyp": r["hyp"]} for r in [r for r in rows if r["scored"]][:EXAMPLES]]}
    out["passed"] = all(v["passed"] for v in out["heads"].values())
    return out


def analyze(dirs: dict[str, Path], replicates: int = REPLICATES, seed: int = SEED) -> dict:
    """report.json's content from the probe, B0 and T* record directories ({"B0", "T*", "probe"})."""
    sets = list(relabel_paths.SETS)
    models, inputs, summary = load_all(dirs, sets)
    checked = {m: {} for m in MODELS}
    for m in MODELS:
        for name in sets:
            diagnostic.realign(m, models[m][name], check_recorded=True)  # check 5 (AnalysisError on a mismatch)
            checked[m][name] = len(models[m][name])
    h4_sets = [n for n in relabel_paths.H4 if n in sets]
    tabs = tables(models, h4_sets)
    contrasts = {CHAR: (B0, CHAR), BPE: (B0, BPE), "char_mid_minus_bpe": (CHAR, BPE)}
    paired = bootstrap.paired(tabs, contrasts, h4_sets, replicates=replicates, seed=seed)
    reps = replicate_wers(tabs, list(MODELS), replicates, seed)

    per_set = {}
    for name in sets:
        c = {m: counts(models[m][name]) for m in MODELS}
        wer = {m: _wer(c[m]) for m in MODELS}
        boot = name in tabs
        entry = {"utterances": len(models[B0][name]), "utterances_scored": c[B0]["utterances"],
                 "reference_words": c[B0]["words"], "clusters": len(tabs[name].clusters) if boot else None,
                 "wer": wer,
                 "rates": {m: {"substitution": 100.0 * c[m]["S"] / c[m]["words"],
                               "deletion": 100.0 * c[m]["D"] / c[m]["words"],
                               "insertion": 100.0 * c[m]["I"] / c[m]["words"]} for m in MODELS},
                 "empty_hypothesis_rate": {m: c[m]["empty"] / c[m]["utterances"] for m in MODELS},
                 "delta_enc": {}, "s": {}}
        eligible = s_eligible(wer[B0], wer[TSTAR])
        for head in HEADS:
            entry["delta_enc"][head] = {"point": wer[B0] - wer[head],
                                        "ci90": paired["sets"][name][head]["ci"] if boot else None}
            s_ci = None
            if boot and eligible:
                s_ci = _ci(s_value(reps[name][head], reps[name][TSTAR], reps[name][B0]))
            entry["s"][head] = {"eligible": eligible, "point": s_value(wer[head], wer[TSTAR], wer[B0])
                                if eligible else None, "ci90": s_ci}
        entry["char_mid_minus_bpe"] = {"point": wer[CHAR] - wer[BPE],
                                       "ci90": paired["sets"][name]["char_mid_minus_bpe"]["ci"] if boot else None}
        per_set[name] = entry

    mean = lambda f: float(np.mean([f(per_set[n]) for n in h4_sets]))  # noqa: E731
    h4_wer = {m: mean(lambda e, m=m: e["wer"][m]) for m in MODELS}
    h4_reps = {m: np.mean([reps[n][m] for n in h4_sets], axis=0) for m in MODELS}
    h4 = {"sets": h4_sets, "wer": h4_wer,
          "rates": {m: {k: mean(lambda e, m=m, k=k: e["rates"][m][k]) for k in ("substitution", "deletion",
                                                                               "insertion")} for m in MODELS},
          "empty_hypothesis_rate": {m: mean(lambda e, m=m: e["empty_hypothesis_rate"][m]) for m in MODELS},
          "delta_enc": {h: {"point": h4_wer[B0] - h4_wer[h], "ci90": paired["mean"][h]["ci"]} for h in HEADS},
          "s": {h: {"eligible": s_eligible(h4_wer[B0], h4_wer[TSTAR]),
                    "point": s_value(h4_wer[h], h4_wer[TSTAR], h4_wer[B0]),
                    "ci90": _ci(s_value(h4_reps[h], h4_reps[TSTAR], h4_reps[B0])),
                    "nonfinite_replicates": int((~np.isfinite(s_value(h4_reps[h], h4_reps[TSTAR],
                                                                       h4_reps[B0]))).sum())} for h in HEADS},
          "char_mid_minus_bpe": {"point": h4_wer[CHAR] - h4_wer[BPE],
                                 "ci90": paired["mean"]["char_mid_minus_bpe"]["ci"]},
          "teacher_gain": h4_wer[B0] - h4_wer[TSTAR]}

    c4 = gross_sanity(models, sets)
    checks = {k: summary["checks"][k] for k in ("hooks_passive", "pre_feedback", "determinism")}
    checks["gross_sanity"] = c4
    checks["records_consistent"] = {"passed": True, "rule": "analyze.realign(check_recorded=True) on every record",
                                    "records_checked": checked}
    delta, ci = h4["delta_enc"][PRIMARY]["point"], h4["delta_enc"][PRIMARY]["ci90"]
    failed = [k for k, v in checks.items() if not v.get("passed")]
    label = None if failed else readout(delta, ci)
    decision = {"head": PRIMARY, "scope": "H4", "delta_enc": delta, "ci90": ci, "threshold": THRESHOLD,
                "ci_excludes_zero": ci[0] > 0 or ci[1] < 0, "label": label,
                "meaning": READOUT_TEXT.get(label), "withheld": f"checks failed: {failed}" if failed else None,
                "s_h4": h4["s"][PRIMARY]}
    return {
        "plan": ep.PLAN, "created": dt.datetime.now(dt.timezone.utc).isoformat(), "inputs": inputs,
        "definitions": DEFINITIONS,
        "probe": {k: summary.get(k) for k in ("revision", "precision", "conventions_sha256", "timings",
                                              "peak_memory", "heads")},
        "checks": checks,
        "bootstrap": {"replicates": replicates, "seed": seed, "ci": CI, "method": "training/bootstrap.py paired",
                      "clusters": {n: len(tabs[n].clusters) for n in h4_sets},
                      "per_set_stream": "SeedSequence([seed, index of the set in sorted H4 set names])"},
        "sets": per_set, "h4": h4, "ami": per_set.get(relabel_paths.AMI), "yodas": per_set.get(relabel_paths.YODAS),
        "secondary": {"ami_unscored_nonempty": diagnostic.excluded_reference_output(
            {m: models[m][relabel_paths.AMI] for m in MODELS})},
        "readout": decision,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", type=Path, default=ep.OUT)
    parser.add_argument("--b0", type=Path, default=relabel_paths.STUDENT_RECORDS[B0])
    parser.add_argument("--tstar", type=Path, default=TSTAR_RECORDS)
    parser.add_argument("--out", type=Path, default=REPORT)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    production = (args.probe.resolve() == ep.OUT.resolve()
                  and args.b0.resolve() == relabel_paths.STUDENT_RECORDS[B0].resolve()
                  and args.tstar.resolve() == TSTAR_RECORDS.resolve()
                  and args.replicates == REPLICATES and args.seed == SEED)
    if args.out.resolve() == REPORT.resolve():
        relabel_paths.require_mount()
        if not production:
            raise SystemExit("refusing to write the production report from non-production inputs or settings")
    try:
        report = analyze({B0: args.b0, TSTAR: args.tstar, "probe": args.probe}, args.replicates, args.seed)
    except AnalysisError as exc:
        print(f"ANALYSIS STOPPED: {exc}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(args.out)
    h4 = report["h4"]
    for name in list(relabel_paths.SETS) + ["H4"]:
        e = h4 if name == "H4" else report["sets"][name]
        w = "  ".join(f"{m} {e['wer'][m]:6.2f}" for m in MODELS)
        d = "  ".join(f"d_{h} {e['delta_enc'][h]['point']:+.2f}" + (
            f" [{e['delta_enc'][h]['ci90'][0]:+.2f}, {e['delta_enc'][h]['ci90'][1]:+.2f}]"
            if e["delta_enc"][h]["ci90"] else "") for h in HEADS)
        print(f"{name:22s} {w}  {d}")
    r = report["readout"]
    print(f"readout: {r['label']} ({r['meaning']}); withheld: {r['withheld']}")
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
