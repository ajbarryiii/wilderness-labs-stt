"""Step 2 of diagnostic plan: CPU-only analysis of errors shared with B0, from per-utterance records.

    ./python analyze.py [--teachers-root DIR] [--b0 DIR] [--m1 DIR] [--mb1 DIR] [--out PATH]
                        [--resamples 1000] [--seed 20261021]

Reads the B0, M1 and MB1 dev records (relabel_paths.STUDENT_RECORDS) and the teacher records written by
score_teacher.py (<teachers-root>/<teacher>/<set>.json; only teachers whose summary.json says
status "scored" and complete), builds the CPU-only "vote" teacher, and writes report.json.

Population: records with scored = true (identical in every model: same manifest, same normalizer;
checked). Unscored records are used only by excluded_reference_output.

Check (hard failure): every record of every model is realigned with relabel_align.align (the backtrace
of evaluate.edit_counts) and the recomputed S, D, I must equal the recorded ones; each alignment's
event multiset must have S + D + I events.

Quantities per set (WER, Delta, F, N, insertion rate in points of reference words; c and o as fractions):

    WER(m)   word-weighted error rate                        c(s)  |E_s & E_B0| / |E_B0|
    o(s)     |E_s - E_B0| / |E_s|                            Delta(t)  WER(B0) - WER(t)
    F(s, t)  events of E_s & E_B0 at positions t fully corrects (ref position aligned as a match by t;
             insertion gap where t inserts nothing), per reference word
    N(t)     |E_t - E_B0| per reference word (descriptive)

(& and - are multiset intersection and difference.) H4 is the unweighted mean over the four human-
transcribed sets; YODAS (pseudo-labelled reference) and AMI (the only conversational set) are also
reported on their own. Bootstrap: per set, 1,000 resamples of the scored utterances with replacement,
every quantity of one resample from the same resampled utterances; H4 intervals average the per-set
resample b across sets; 90% intervals are the 5th and 95th percentiles.

The decision (diagnostic plan "Decision") is computed mechanically from DECISION.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np

import relabel_paths
from relabel_align import align, fixed_by, only_in, shared, size

SEED = 20261021
RESAMPLES = 1000
STUDENTS = ["M1", "MB1"]
BASE = "B0"
RECORDED = [BASE] + STUDENTS  # models whose recorded S, D, I must be reproduced (plan "Check")
DECISION = {"delta_min_points": 0.30, "F_MB1_min_points": 0.15, "near_tie_points": 0.05, "ci": 0.90,
            "student": "MB1"}
EPS = 1e-9  # float tolerance on the preregistered thresholds
# Open ASR Leaderboard private conversational WER (diagnostic plan "Candidate teachers"); vote uses Qwen3's.
PRIVATE_CONVERSATIONAL_WER = {"qwen3-asr-1.7b": 12.14, "granite-speech-5.0-2b": 13.79, "canary-qwen-2.5b": 13.41,
                              "vote": 12.14}
DEFINITIONS = {
    "population": "records with scored=true (identical across models); reference words of those records",
    "wer": "points: 100 * (S+D+I) / reference words",
    "c": "fraction: |E_s & E_B0| / |E_B0|, the share of B0's error events the student also makes (exact multiset match)",
    "o": "fraction: |E_s - E_B0| / |E_s|, the share of the student's error events B0 does not make",
    "delta": "points: WER(B0) - WER(t)",
    "F": ("points: events of E_s & E_B0 (errors the student shares with B0) at positions t fully corrects "
          "(ref position aligned as a match by t; insertion gap where t inserts nothing), per reference word"),
    "N": "points: |E_t - E_B0| per reference word, the teacher's error events B0 does not make (descriptive)",
    "insertion_rate": "points: 100 * I / reference words",
    "events": ("(utterance, sub|del, ref index, hyp word) and (utterance, ins, gap index, hyp word) from the "
               "edit_counts backtrace; multisets; shared events are exact matches, so the same mistake aligned "
               "differently by two models is not counted as shared (a slight undercount, stated, not corrected)"),
}


class AnalysisError(RuntimeError):
    pass


# --- inputs --------------------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_model(directory: Path, sets: list[str]) -> tuple[dict, dict]:
    """({set: records}, {set: {path, sha256}}) from DIR/<set>.json."""
    records, files = {}, {}
    for name in sets:
        path = Path(directory) / f"{name}.json"
        if not path.exists():
            raise AnalysisError(f"missing records {path}")
        data = json.loads(path.read_text())
        if data.get("limit") is not None:
            raise AnalysisError(f"{path} is a --limit subset")
        records[name] = data["records"]
        files[name] = {"path": str(path), "sha256": sha256_file(path)}
    return records, files


def teacher_state(directory: Path, sets: list[str]) -> tuple[bool, str, dict | None]:
    """(scorable, reason, summary) for a teacher output directory."""
    summary_path = Path(directory) / "summary.json"
    if not summary_path.exists():
        return False, f"no {summary_path}", None
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "scored" or not summary.get("complete"):
        return False, f"status {summary.get('status')!r}: {summary.get('reason', 'incomplete')}", summary
    if summary.get("config", {}).get("limit") is not None:
        return False, "smoke (--limit) run", summary
    missing = [n for n in sets if not (Path(directory) / f"{n}.json").exists()]
    if missing:
        return False, f"missing sets {missing}", summary
    if set(summary.get("sets", {})) != set(sets):
        return False, f"summary covers {sorted(summary.get('sets', {}))}, not every set", summary
    for name in sets:
        config = json.loads((Path(directory) / f"{name}.json").read_text()).get("config")
        if config != summary.get("config"):
            return False, f"{name}.json was written with another configuration than summary.json", summary
    return True, "", summary


def check_consistent(models: dict[str, dict], sets: list[str]) -> None:
    """Same ids in the same order, same ref_norm and scored flag, in every model and set."""
    for name in sets:
        base = models[BASE][name]
        key = [(r["id"], r["ref_norm"], bool(r["scored"])) for r in base]
        if len({k[0] for k in key}) != len(key):
            raise AnalysisError(f"{BASE} {name}: duplicate ids")
        for model, data in models.items():
            other = [(r["id"], r["ref_norm"], bool(r["scored"])) for r in data[name]]
            if other != key:
                raise AnalysisError(f"{model} {name}: ids, ref_norm or scored differ from {BASE}")


def realign(model: str, records: list[dict], check_recorded: bool) -> list:
    """Alignments of every record; raises AnalysisError if S, D, I differ from the record (check_recorded)
    or if an alignment's event multiset does not hold S + D + I events."""
    out, mismatches = [], []
    for r in records:
        a = align(r["ref_norm"].split(), r["hyp_norm"].split())
        if size(a.events) != a.errors:
            raise AnalysisError(f"{model} {r['id']}: {size(a.events)} events != S+D+I {a.errors}")
        if check_recorded and (a.S, a.D, a.I) != (r["S"], r["D"], r["I"]):
            mismatches.append((r["id"], (r["S"], r["D"], r["I"]), (a.S, a.D, a.I)))
        out.append(a)
    if mismatches:
        raise AnalysisError(f"{model}: recomputed S, D, I differ from the records for {len(mismatches)} "
                            f"utterances, first {mismatches[:3]}")
    return out


# --- vote ----------------------------------------------------------------------------------------

def build_vote(teacher_records: dict[str, list[dict]]) -> tuple[list[dict], dict]:
    """Vote records of one set from the three real teachers' records (relabel_paths.REAL_TEACHERS order).

    Per utterance: the normalized hypothesis at least two teachers produce; if all three differ, the
    first teacher's (Qwen3-ASR). Raw text from the first teacher, in order, whose normalized hypothesis
    is the chosen one. No reference is read for the choice; S, D, I are recomputed for the chosen text.
    """
    import evaluate

    order = relabel_paths.REAL_TEACHERS
    columns = [teacher_records[t] for t in order]
    out, sources = [], {t: 0 for t in order}
    for rows in zip(*columns):
        if len({r["id"] for r in rows}) != 1:
            raise AnalysisError("teacher records are not aligned by id")
        norms = [r["hyp_norm"] for r in rows]
        chosen = next((n for n in norms if norms.count(n) >= 2), norms[0])
        k = norms.index(chosen)
        src = rows[k]
        s, d, i = evaluate.edit_counts(src["ref_norm"].split(), chosen.split())
        out.append({"id": src["id"], "ref": src["ref"], "hyp": src["hyp"], "ref_norm": src["ref_norm"],
                    "hyp_norm": chosen, "S": s, "D": d, "I": i, "scored": src["scored"],
                    "duration": src["duration"], "vote_source": order[k]})
        sources[order[k]] += 1
    return out, sources


# --- per-set counts and metrics ------------------------------------------------------------------

def count_table(alignments: dict[str, list], scored: list[bool], ref_words: list[int],
                teachers: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    """Per scored utterance, the integer counts every metric is a ratio of sums of (columns by name)."""
    models = list(alignments)
    names = ["ref"] + [f"err:{m}" for m in models] + [f"ins:{m}" for m in models]
    names += [f"shared:{s}" for s in STUDENTS] + [f"fix:{s}:{t}" for s in STUDENTS for t in teachers]
    names += [f"new:{t}" for t in teachers]
    col = {n: k for k, n in enumerate(names)}
    rows = []
    for u, keep in enumerate(scored):
        if not keep:
            continue
        a = {m: alignments[m][u] for m in models}
        row = [0] * len(names)
        row[col["ref"]] = ref_words[u]
        for m in models:
            row[col[f"err:{m}"]] = a[m].errors
            row[col[f"ins:{m}"]] = a[m].I
        base = a[BASE].events
        for s in STUDENTS:
            common = shared(a[s].events, base)
            row[col[f"shared:{s}"]] = size(common)
            for t in teachers:
                row[col[f"fix:{s}:{t}"]] = sum(c for ev, c in common.items() if fixed_by(ev, a[t]))
        for t in teachers:
            row[col[f"new:{t}"]] = size(only_in(a[t].events, base))
        rows.append(row)
    return np.asarray(rows, dtype=np.int64).reshape(-1, len(names)), col


def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.asarray(b) == 0, np.nan, np.asarray(a, dtype=float) / np.where(np.asarray(b) == 0, 1, b))


def metrics(sums: np.ndarray, col: dict[str, int], models: list[str], teachers: list[str]) -> dict:
    """All Step 2 quantities from column sums (last axis = columns; leading axes = resamples)."""
    g = lambda name: sums[..., col[name]]  # noqa: E731
    ref = g("ref")
    wer = {m: 100 * _div(g(f"err:{m}"), ref) for m in models}
    out = {"ref_words": ref, "wer": wer,
           "insertion_rate": {m: 100 * _div(g(f"ins:{m}"), ref) for m in models},
           "c": {s: _div(g(f"shared:{s}"), g(f"err:{BASE}")) for s in STUDENTS},
           "o": {s: _div(g(f"err:{s}") - g(f"shared:{s}"), g(f"err:{s}")) for s in STUDENTS},
           "delta": {t: wer[BASE] - wer[t] for t in teachers},
           "F": {s: {t: 100 * _div(g(f"fix:{s}:{t}"), ref) for t in teachers} for s in STUDENTS},
           "N": {t: 100 * _div(g(f"new:{t}"), ref) for t in teachers}}
    return out


def bootstrap_sums(table: np.ndarray, resamples: int, seed: int, set_index: int) -> np.ndarray:
    """[resamples, columns] column sums over utterances drawn with replacement (fixed per-set stream)."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, set_index]))
    n = table.shape[0]
    out = np.empty((resamples, table.shape[1]), dtype=np.float64)
    values = table.astype(np.float64)
    for b in range(resamples):
        weights = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        out[b] = weights @ values
    return out


def _tolist(x):
    if isinstance(x, dict):
        return {k: _tolist(v) for k, v in x.items()}
    value = float(x)
    return None if np.isnan(value) else value


def _ci(samples: np.ndarray) -> list[float]:
    lo, hi = np.percentile(samples, [100 * (1 - DECISION["ci"]) / 2, 100 * (1 + DECISION["ci"]) / 2])
    return [float(lo), float(hi)]


def _mean(values: list):
    if isinstance(values[0], dict):
        return {k: _mean([v[k] for v in values]) for k in values[0]}
    return float(np.mean(values))


# --- secondary -----------------------------------------------------------------------------------

def excluded_reference_output(records: dict[str, list[dict]]) -> dict:
    """Per model, among records with scored = false: count and share with non-empty normalized output."""
    out = {}
    for model, rows in records.items():
        excluded = [r for r in rows if not r["scored"]]
        if excluded:
            nonempty = sum(bool(r["hyp_norm"].split()) for r in excluded)
            out[model] = {"excluded": len(excluded), "nonempty_output": nonempty,
                          "nonempty_share": nonempty / len(excluded)}
    return out


def style(records: dict[str, list[dict]]) -> dict:
    """Per model, share of scored raw hypotheses with any punctuation character and with any uppercase letter."""
    out = {}
    for model, rows in records.items():
        rows = [r for r in rows if r["scored"]]
        punct = sum(any(unicodedata.category(ch).startswith("P") for ch in r["hyp"]) for r in rows)
        upper = sum(any(ch.isupper() for ch in r["hyp"]) for r in rows)
        out[model] = {"punctuation_share": punct / len(rows), "uppercase_share": upper / len(rows)}
    return out


def agreement(records: dict[str, list[dict]], alignments: dict[str, list], teachers: list[str]) -> dict:
    """For each teacher a and other model b in teachers + B0: share of scored utterances with identical
    normalized hypotheses, and a's WER on the agreeing and disagreeing subsets."""
    out = {}
    scored = [r["scored"] for r in records[BASE]]
    refs = [len(r["ref_norm"].split()) for r in records[BASE]]
    hours = [r["duration"] for r in records[BASE]]
    for a in teachers:
        for b in teachers + [BASE]:
            if a == b:
                continue
            agree = {"n": 0, "err": 0, "ref": 0, "s": 0.0}
            differ = {"n": 0, "err": 0, "ref": 0, "s": 0.0}
            for u, keep in enumerate(scored):
                if not keep:
                    continue
                bucket = agree if records[a][u]["hyp_norm"] == records[b][u]["hyp_norm"] else differ
                bucket["n"] += 1; bucket["err"] += alignments[a][u].errors; bucket["ref"] += refs[u]
                bucket["s"] += hours[u]
            total = agree["n"] + differ["n"]
            out[f"{a}|{b}"] = {
                "agree_share": agree["n"] / total,
                "agree_audio_share": agree["s"] / (agree["s"] + differ["s"]),
                "wer_first_on_agree": 100 * agree["err"] / agree["ref"] if agree["ref"] else None,
                "wer_first_on_disagree": 100 * differ["err"] / differ["ref"] if differ["ref"] else None,
                "agree_utterances": agree["n"], "disagree_utterances": differ["n"]}
    return out


# --- decision ------------------------------------------------------------------------------------

def decide(h4: dict, h4_ci: dict, teachers: list[str], throughput: dict[str, float | None]) -> dict:
    """The preregistered rule on H4 means (diagnostic plan "Decision")."""
    s = DECISION["student"]
    rows = {}
    for t in teachers:
        delta, F = h4["delta"][t], h4["F"][s][t]
        lo, hi = h4_ci["delta"][t]
        excludes_zero = lo > 0 or hi < 0
        gate1 = delta is not None and delta >= DECISION["delta_min_points"] - EPS and excludes_zero
        gate2 = F is not None and F >= DECISION["F_MB1_min_points"] - EPS
        rows[t] = {"delta": delta, "delta_ci90": [lo, hi], "delta_ci_excludes_zero": excludes_zero,
                   f"F_{s}": F, "gate_delta": bool(gate1), f"gate_F_{s}": bool(gate2), "passes": bool(gate1 and gate2)}
    passing = [t for t in teachers if rows[t]["passes"]]
    result = {"rule": DECISION, "teachers": rows, "passing": passing}
    if not passing:
        result.update(write_labeling_plan=False, chosen=None,
                      verdict="no teacher passes both gates: stop and report")
        return result
    best = max(rows[t][f"F_{s}"] for t in passing)
    near = [t for t in passing if best - rows[t][f"F_{s}"] <= DECISION["near_tie_points"] + EPS]

    def key(t):
        rate = throughput.get(t)
        return (t == relabel_paths.VOTE, PRIVATE_CONVERSATIONAL_WER[t], -(rate or 0.0), -rows[t][f"F_{s}"])

    chosen = sorted(near, key=key)[0]  # near holds the largest F; with one entry, that teacher
    result.update(write_labeling_plan=True, chosen=chosen, near_ties=near,
                  tie_break=("single teacher over vote, then private-conversational WER, then throughput"
                             if len(near) > 1 else None),
                  verdict=f"write the labeling plan with {chosen}")
    return result


# --- driver --------------------------------------------------------------------------------------

def analyze(student_dirs: dict[str, Path], teachers_root: Path, resamples: int = RESAMPLES,
            seed: int = SEED, sets: list[str] | None = None) -> dict:
    sets = sets or relabel_paths.SETS
    models, inputs = {}, {}
    for m in RECORDED:
        models[m], inputs[m] = load_model(student_dirs[m], sets)
    teachers, not_scored, summaries = [], {}, {}
    for t in relabel_paths.REAL_TEACHERS:
        ok, reason, summary = teacher_state(Path(teachers_root) / t, sets)
        if not ok:
            not_scored[t] = reason
            continue
        models[t], inputs[t] = load_model(Path(teachers_root) / t, sets)
        teachers.append(t)
        summaries[t] = {"teacher": summary["teacher"], "throughput": summary["throughput"], "config": summary["config"]}
    vote_sources = None
    if not not_scored:
        models[relabel_paths.VOTE], vote_sources = {}, {}
        for name in sets:
            models[relabel_paths.VOTE][name], vote_sources[name] = build_vote({t: models[t][name] for t in teachers})
        teachers.append(relabel_paths.VOTE)
    else:
        not_scored[relabel_paths.VOTE] = f"not computed: teachers not scored {sorted(not_scored)}"
    check_consistent(models, sets)

    throughput = {t: summaries[t]["throughput"]["audio_hours_per_gpu_hour"] for t in teachers if t in summaries}
    if relabel_paths.VOTE in teachers:
        rates = [throughput[t] for t in relabel_paths.REAL_TEACHERS]
        throughput[relabel_paths.VOTE] = 1 / sum(1 / r for r in rates) if all(rates) else None

    model_names = list(models)
    per_set, boot, checked = {}, {}, {}
    secondary = {"excluded_reference_output": {}, "style": {}, "agreement": {}}
    for k, name in enumerate(sets):
        alignments = {m: realign(m, models[m][name], check_recorded=True) for m in model_names}
        checked[name] = {m: len(models[m][name]) for m in model_names}
        base = models[BASE][name]
        scored = [bool(r["scored"]) for r in base]
        refs = [len(r["ref_norm"].split()) for r in base]
        table, col = count_table(alignments, scored, refs, teachers)
        point = metrics(table.sum(axis=0), col, model_names, teachers)
        samples = metrics(bootstrap_sums(table, resamples, seed, relabel_paths.SETS.index(name)), col,
                          model_names, teachers)
        boot[name] = samples
        per_set[name] = {"utterances": len(base), "utterances_scored": int(table.shape[0]), **_tolist(point),
                         "ci90": {"delta": {t: _ci(samples["delta"][t]) for t in teachers},
                                  "F": {s: {t: _ci(samples["F"][s][t]) for t in teachers} for s in STUDENTS}}}
        secondary["excluded_reference_output"][name] = excluded_reference_output(
            {m: models[m][name] for m in model_names})
        secondary["style"][name] = style({m: models[m][name] for m in model_names})
        secondary["agreement"][name] = agreement({m: models[m][name] for m in model_names}, alignments, teachers)

    h4_sets = [n for n in relabel_paths.H4 if n in sets]
    keys = ["wer", "insertion_rate", "c", "o", "delta", "F", "N"]
    h4 = {key: _mean([per_set[n][key] for n in h4_sets]) for key in keys}
    h4_ci = {"delta": {t: _ci(np.mean([boot[n]["delta"][t] for n in h4_sets], axis=0)) for t in teachers},
             "F": {s: {t: _ci(np.mean([boot[n]["F"][s][t] for n in h4_sets], axis=0)) for t in teachers}
                   for s in STUDENTS}}
    h4["ci90"] = h4_ci
    h4["sets"] = h4_sets
    secondary["agreement_h4_mean_share"] = {
        pair: float(np.mean([secondary["agreement"][n][pair]["agree_share"] for n in h4_sets]))
        for pair in secondary["agreement"][sets[0]]} if teachers else {}

    decision = decide(h4, h4_ci, teachers, throughput) if teachers else {
        "rule": DECISION, "write_labeling_plan": False, "chosen": None, "verdict": "no teacher scored"}
    return {
        "plan": str(relabel_paths.PLAN), "created": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inputs": inputs, "definitions": DEFINITIONS,
        "check": {"sdi_reproduced": True, "event_counts_equal_sdi": True, "records_checked": checked,
                  "recorded_models": RECORDED},
        "teachers": teachers, "not_scored": not_scored, "teacher_summaries": summaries,
        "vote_sources": vote_sources, "throughput_audio_hours_per_gpu_hour": throughput,
        "private_conversational_wer": PRIVATE_CONVERSATIONAL_WER,
        "bootstrap": {"resamples": resamples, "seed": seed, "interval": DECISION["ci"],
                      "per_set_stream": "SeedSequence([seed, index of the set in relabel_paths.SETS])"},
        "sets": per_set, "h4": h4, "ami": per_set.get(relabel_paths.AMI), "yodas": per_set.get(relabel_paths.YODAS),
        "secondary": secondary, "decision": decision,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teachers-root", type=Path, default=relabel_paths.DIAGNOSTIC)
    parser.add_argument("--b0", type=Path, default=relabel_paths.STUDENT_RECORDS["B0"])
    parser.add_argument("--m1", type=Path, default=relabel_paths.STUDENT_RECORDS["M1"])
    parser.add_argument("--mb1", type=Path, default=relabel_paths.STUDENT_RECORDS["MB1"])
    parser.add_argument("--out", type=Path, default=relabel_paths.REPORT)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dirs = {"B0": args.b0, "M1": args.m1, "MB1": args.mb1}
    production_inputs = (args.teachers_root.resolve() == relabel_paths.DIAGNOSTIC.resolve()
                         and all(Path(dirs[m]).resolve() == relabel_paths.STUDENT_RECORDS[m].resolve() for m in dirs)
                         and args.resamples == RESAMPLES and args.seed == SEED)
    if args.out.resolve() == relabel_paths.REPORT.resolve():
        relabel_paths.require_mount()
        if not production_inputs:
            raise SystemExit("refusing to write the production report from non-production inputs or settings")
    try:
        report = analyze(dirs, args.teachers_root, args.resamples, args.seed)
    except AnalysisError as exc:
        print(f"ANALYSIS STOPPED: {exc}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(args.out)
    d, h4 = report["decision"], report["h4"]
    for t in report["teachers"]:
        lo, hi = h4["ci90"]["delta"][t]
        print(f"{t:24s} H4 WER {h4['wer'][t]:.2f} delta {h4['delta'][t]:+.2f} [{lo:+.2f}, {hi:+.2f}] "
              f"F(MB1) {h4['F']['MB1'][t]:.3f} F(M1) {h4['F']['M1'][t]:.3f} N {h4['N'][t]:.2f}")
    for t, reason in report["not_scored"].items():
        print(f"{t:24s} not scored: {reason}")
    print(f"decision: {d['verdict']}")
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
