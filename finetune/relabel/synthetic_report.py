"""Results of stage 2c (synthetic plan "Results (preregistered)"): results-synthetic.json and the stage 2c section of
RESULTS-decoder-only.md (written after stage 2b's, decoder_report.markdown, in the same file).

    ./python train_synthetic.py report [--out PATH]      (CPU; reads $RELABEL_ROOT/decoder-only/ and the term split)

Models, per student: A, the selected T* run of stage 2b (decoder-only/selection/<s>-T.json), and TS and TSn, the
student's synthetic arms at A's learning rate.

Term recall (real speech only): eval-stem (decoder-only/eval-stem/<arm>/<lr>/) and dev-technical (each run's
domain/), per (domain, side) of the frozen term split (synthetic/terms/split.json; vocab.Split.matcher), each
occurrence as score_domain.term_occurrences decides it (the aligned-occurrence definition with the spelling-variant
rule: vocab.recall's counts). Groups (groups()): "targeted" and "heldout", pooled over the domains of the pooled
claim (synthetic/terms/coverage.json, counted before training: a domain with fewer than coverage.MIN_TARGETED
targeted occurrences over both sets is underpowered and left out); "all-domains/<side>"; "<domain>/<side>". Recall
in points; a contrast (a, b) is recall(a) - recall(b), positive when a reproduces more.

Intervals: 90% paired cluster bootstraps, 10,000 replicates (training/bootstrap.py's constants). Term recall and
domain WER: clusters are recordings (recording_of: the talk, call or lecture of a segment id; a tech-dev-1 clip is
its own recording, since its source names no talk), drawn with replacement within each part from a stream seeded by
(seed, part index in sorted order), the same draw for every model; pooled recall is recomputed per replicate from
the drawn counts. H4: decoder_report's tables on the evaluation part of the frozen dev split (bootstrap.paired).
WER contrasts (a, b) are WER(a) - WER(b), positive when a is worse.

Claims, per student, mechanically (claims()):
    1. synthetic speech helps the targeted vocabulary: recall TS - A on "targeted" > 0 with its 90% interval
       excluding 0, and H4 TS - A <= H4_TOLERANCE (0.10 points, the point estimate);
    2. augmentation matters: recall TS - TSn on "targeted" > 0 with its interval excluding 0.
Reported without a claim: held-out recall, per-domain recall, domain WER per part, per set and over both sets
(unweighted means of the parts' WERs, as score_domain.py), the decoder-step proxy, whether TSn is worse than A on H4
(TSn - A > 0 with its interval excluding 0), the learning rate used and why, and the runs' counts.
"""
from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import numpy as np

import decoder_report as drep
import relabel_paths
import train_decoder as td
import train_synthetic as ts
from synthetic import layout, vocab
from training import bootstrap, devsplit

DOMAINS = layout.DOMAINS
SIDES = vocab.SIDES
TERM_SETS = {"eval-stem": ("stem-chem", "stem-math", "stem-cs"), "dev-technical": ("tech-dev-1", "tech-dev-2")}
H4_TOLERANCE = 0.10
EPS = 1e-9
CONTRASTS = {"TS-A": ("TS", "A"), "TS-TSn": ("TS", "TSn"), "TSn-A": ("TSn", "A")}
_SEGMENT = re.compile(r"(?P<part>[^:]+):(?P<recording>.+?)-\d{4}(?:-\d{2})?")


def recording_of(item_id: str) -> str:
    """The recording of a domain-set segment id "<part>:<recording>-<k:04d>[-<j:02d>]" (devsets/segment.py)."""
    m = _SEGMENT.fullmatch(str(item_id))
    if m is None:
        raise ValueError(f"{item_id!r} is not a domain-set segment id")
    return f"{m['part']}:{m['recording']}"


def check_recordings(part: str) -> int:
    """recording_of agrees with the manifest's audio directories (<part>/<recording>/<segment>.flac); returns the
    number of recordings."""
    from devsets import layout as dlayout
    seen = set()
    with open(dlayout.Layout.production().manifest(part), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                want = f"{part}:{Path(r['audio_filepath']).parent.name}"
                if recording_of(r["id"]) != want:
                    raise RuntimeError(f"{r['id']}: recording {recording_of(r['id'])} != its audio directory {want}")
                seen.add(want)
    return len(seen)


# ----------------------------------------------------------------------------- term recall

@dataclasses.dataclass
class RecallTable:
    clusters: list[str]                              # sorted cluster names
    occurrences: dict[tuple, np.ndarray]             # (domain, side) -> [clusters] reference occurrences
    reproduced: dict[str, dict[tuple, np.ndarray]]   # model -> (domain, side) -> [clusters] reproduced
    utterances: int

    @classmethod
    def from_records(cls, records: dict[str, list[dict]], split: vocab.Split, cluster_of=recording_of) -> "RecallTable":
        """Every model's records (evaluate.py's per-utterance layout) of one part: the same utterances and
        references, scored records only."""
        import score_domain
        models = sorted(records)
        keyed = {m: {r["id"]: r for r in records[m]} for m in models}
        first = keyed[models[0]]
        for m in models[1:]:
            if set(keyed[m]) != set(first):
                raise ValueError(f"model {m} does not cover the same utterances as {models[0]}")
        chosen = sorted(i for i in first if first[i].get("scored", True))
        for m in models:
            for i in chosen:
                if not keyed[m][i].get("scored", True) or keyed[m][i]["ref_norm"] != first[i]["ref_norm"]:
                    raise ValueError(f"{i}: scored or reference differs between {models[0]} and {m}")
        names = sorted({cluster_of(i) for i in chosen})
        index = {c: j for j, c in enumerate(names)}
        keys = [(d, s) for d in DOMAINS for s in SIDES]
        occ = {k: np.zeros(len(names)) for k in keys}
        rep = {m: {k: np.zeros(len(names)) for k in keys} for m in models}
        matcher = split.matcher()
        for m in models:
            for i in chosen:
                r, c = keyed[m][i], index[cluster_of(i)]
                ref = r["ref_norm"].split()
                for start, end, ok in score_domain.term_occurrences(r["ref_norm"], r["hyp_norm"], matcher):
                    key = split.classify(tuple(ref[start:end]))
                    if m == models[0]:
                        occ[key][c] += 1
                    rep[m][key][c] += int(ok)
        return cls(names, occ, rep, len(chosen))


def groups(underpowered) -> dict[str, set[tuple[str, str]]]:
    """The recall groups (module docstring); "targeted" and "heldout" leave the underpowered domains out."""
    claim = [d for d in DOMAINS if d not in set(underpowered)]
    out = {side: {(d, side) for d in claim} for side in SIDES}
    out |= {f"all-domains/{side}": {(d, side) for d in DOMAINS} for side in SIDES}
    out |= {f"{d}/{side}": {(d, side)} for d in DOMAINS for side in SIDES}
    return out


def _ci(samples: np.ndarray, level: float) -> list[float]:
    lo, hi = np.percentile(samples, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return [float(lo), float(hi)]


def recall_bootstrap(tables: dict[str, RecallTable], contrasts: dict[str, tuple[str, str]],
                     groups: dict[str, set], replicates: int = bootstrap.REPLICATES, seed: int = bootstrap.SEED,
                     ci: float = bootstrap.CI) -> dict:
    """Point recall per group and model, and every contrast's paired cluster-bootstrap interval (module docstring)."""
    models = sorted({m for pair in contrasts.values() for m in pair})
    names = sorted(tables)
    counts = {}
    for p_index, name in enumerate(names):
        n = len(tables[name].clusters)
        rng = np.random.default_rng(np.random.SeedSequence([seed, p_index]))
        c = np.empty((replicates, n), dtype=np.float64)
        for b in range(replicates):
            c[b] = np.bincount(rng.integers(0, n, n), minlength=n)
        counts[name] = c
    out = {"replicates": replicates, "seed": seed, "ci_level": ci, "clusters": {n: len(tables[n].clusters) for n in names},
           "utterances": {n: tables[n].utterances for n in names}, "groups": {}}
    for g, keys in groups.items():
        occ = {n: sum((tables[n].occurrences[k] for k in keys), np.zeros(len(tables[n].clusters))) for n in names}
        rep = {m: {n: sum((tables[n].reproduced[m][k] for k in keys), np.zeros(len(tables[n].clusters)))
                   for n in names} for m in models}
        total = float(sum(v.sum() for v in occ.values()))
        got = {m: float(sum(v.sum() for v in rep[m].values())) for m in models}
        entry = {"keys": sorted(f"{d}/{s}" for d, s in keys), "occurrences": int(total),
                 "reproduced": {m: int(v) for m, v in got.items()},
                 "recall": {m: (100.0 * got[m] / total if total else None) for m in models}, "contrasts": {}}
        if total:
            occ_r = sum(counts[n] @ occ[n] for n in names)
            with np.errstate(invalid="ignore", divide="ignore"):
                rec_r = {m: 100.0 * sum(counts[n] @ rep[m][n] for n in names) / occ_r for m in models}
            for c, (a, b) in contrasts.items():
                diff = rec_r[a] - rec_r[b]
                finite = diff[np.isfinite(diff)]
                entry["contrasts"][c] = {"point": entry["recall"][a] - entry["recall"][b], "ci": _ci(finite, ci),
                                         "replicates_without_occurrences": int(len(diff) - len(finite))}
        out["groups"][g] = entry
    return out


def claims(h4: dict, recall: dict) -> dict:
    """The preregistered claims from the H4 bootstrap and the recall bootstrap (points; module docstring)."""
    t = recall["groups"]["targeted"]["contrasts"]
    rules = {"claim1": f"recall TS - A (pooled targeted) > 0 with its 90% interval excluding 0, and H4 TS - A <= "
                       f"{H4_TOLERANCE} points", "claim2": "recall TS - TSn (pooled targeted) > 0 with its 90% "
                                                          "interval excluding 0"}
    h4_ts_a, tsn_a = h4["mean"]["TS-A"]["point"], h4["mean"]["TSn-A"]
    if not t:
        return {"synthetic_helps_targeted": None, "augmentation_matters": None,
                "tsn_worse_than_t_star_on_h4": tsn_a["point"] > EPS and tsn_a["ci"][0] > EPS,
                "h4_ts_minus_a_points": h4_ts_a, "rules": rules, "note": "no targeted occurrence in the claim domains"}
    ts_a, ts_tsn = t["TS-A"], t["TS-TSn"]
    return {"synthetic_helps_targeted": (ts_a["point"] > EPS and ts_a["ci"][0] > EPS
                                         and h4_ts_a <= H4_TOLERANCE + EPS),
            "augmentation_matters": ts_tsn["point"] > EPS and ts_tsn["ci"][0] > EPS,
            "tsn_worse_than_t_star_on_h4": tsn_a["point"] > EPS and tsn_a["ci"][0] > EPS,
            "h4_ts_minus_a_points": h4_ts_a, "rules": rules}


# ----------------------------------------------------------------------------- per student

def coverage_info(split: vocab.Split) -> dict:
    """The frozen coverage (synthetic/terms/coverage.json) and its underpowered domains (None if it is missing or
    not complete)."""
    path = layout.terms_dir() / "coverage.json"
    data = json.loads(path.read_text()) if path.exists() else None
    ok = bool(data and data.get("complete") and data.get("split_entries") == len(split.entries))
    return {"path": str(path.relative_to(layout.relabel_root())), "complete": ok,
            "underpowered": data["underpowered"] if ok else None, "totals": (data or {}).get("totals"),
            "min_targeted": (data or {}).get("min_targeted")}


def part_records(model_dirs: dict[str, Path], stem_dirs: dict[str, Path]) -> dict[str, dict[str, list[dict]]]:
    """{part: {model: records}} of eval-stem (stem_dirs) and dev-technical (each run's domain/)."""
    out = {}
    for name, parts in TERM_SETS.items():
        for part in parts:
            out[part] = {m: json.loads(((stem_dirs[m] if name == "eval-stem" else model_dirs[m] / "domain")
                                        / f"{part}.json").read_text())["records"] for m in model_dirs}
    return out


def reference_counts(tables: dict[str, RecallTable]) -> dict:
    return {d: {s: int(sum(t.occurrences[(d, s)].sum() for t in tables.values())) for s in SIDES} for d in DOMAINS}


def student_result(s: str, split: devsplit.Split, term_split: vocab.Split, cov: dict) -> dict:
    t_arm = f"{s}-T"
    arms = {"TS": f"{s}-TS", "TSn": f"{s}-TSn"}
    out = {"student": s, "name": td.STUDENTS[s]["name"], "t_arm": t_arm, "arms": arms}
    lr = ts.selected_lr(t_arm)
    if lr is None:
        return out | {"status": f"the {t_arm} learning rate is not selected yet"}
    dirs = {"A": td.run_dir(t_arm, lr), "TS": td.run_dir(arms["TS"], lr), "TSn": td.run_dir(arms["TSn"], lr)}
    stems = {"A": ts.stem_dir(t_arm, lr), "TS": ts.stem_dir(arms["TS"], lr), "TSn": ts.stem_dir(arms["TSn"], lr)}
    missing = [f"{m} run" for m, d in dirs.items() if not (d / "summary.json").exists()]
    missing += [f"{m} eval-stem" for m, d in stems.items() if not (d / "summary.json").exists()]
    out["lr"] = td.LR_NAMES[lr]
    if missing:
        return out | {"status": f"pending: {', '.join(missing)}"}
    if cov["underpowered"] is None:
        return out | {"status": "coverage.json is missing, incomplete or of another split (./python -m synthetic "
                                "coverage)"}
    recs = {m: {n: drep.records(d, n) for n in devsplit.H4} for m, d in dirs.items()}
    h4 = bootstrap.paired(drep.eval_tables(recs, split), CONTRASTS, mean_sets=list(devsplit.H4))
    parts = part_records(dirs, stems)
    tables = {p: RecallTable.from_records(r, term_split) for p, r in parts.items()}
    recall = recall_bootstrap(tables, CONTRASTS, groups(cov["underpowered"]))
    wer_tables = {p: bootstrap.SetTable.from_records(r, cluster_of=recording_of) for p, r in parts.items()}
    wer = {"parts": bootstrap.paired(wer_tables, CONTRASTS, mean_sets=sorted(wer_tables))}
    for name, members in TERM_SETS.items():
        wer[name] = bootstrap.paired({p: wer_tables[p] for p in members}, CONTRASTS, mean_sets=list(members))
    counts = reference_counts(tables)
    summaries = {m: json.loads((d / "summary.json").read_text()) for m, d in dirs.items()}
    return out | {"status": "complete", "dirs": {m: str(d.relative_to(td.root())) for m, d in dirs.items()},
                  "lr_selection": json.loads(ts.selection_path(t_arm).read_text()),
                  "h4": h4, "recall": recall, "domain_wer": wer, "claims": claims(h4, recall),
                  "reference_occurrences": counts,
                  "coverage_matches_scored_references": counts == {d: (cov["totals"] or {}).get(d) for d in DOMAINS},
                  "proxy": {m: (drep._json(d / "proxy" / "summary.json") or {}).get("pooled") for m, d in dirs.items()},
                  "seconds_by_source": {m: x.get("seconds_by_source") for m, x in summaries.items()}}


def run_table() -> list[dict]:
    rows = []
    for arm, t_arm in ts.BASE.items():
        lr = ts.selected_lr(t_arm)
        s = drep._json(td.run_dir(arm, lr) / "summary.json") if lr is not None else None
        if s is None:
            rows.append({"arm": arm, "lr": td.LR_NAMES[lr] if lr is not None else None, "status": "pending"})
            continue
        syn = s.get("synthetic", {})
        by_source = s.get("seconds_by_source") or {}
        rows.append({"arm": arm, "lr": td.LR_NAMES[lr], "status": "done", "steps": s["steps"],
                     "planned_batches": s.get("batches_read"), "synthetic_rows": syn.get("synthetic_rows"),
                     "synthetic_hours": by_source.get("synthetic", 0.0) / 3600,
                     "real_hours": sum(v for k, v in by_source.items() if k != "synthetic") / 3600,
                     "selection_h4": s.get("decoder_only", {}).get("selection_part", {}).get("h4"),
                     "encoder_bit_identical": s.get("decoder_only", {}).get("encoder_bit_identical"),
                     "resumes": s.get("resumes")})
    return rows


def analyze_all() -> dict:
    split = devsplit.load(td.split_path(), devsplit.dev_manifests())
    term_split = vocab.Split.load()
    cov = coverage_info(term_split)
    recordings = {}
    for parts in TERM_SETS.values():
        for p in parts:
            recordings[p] = check_recordings(p)
    return {"plan": "synthetic plan (stage 2c)", "split_sha256": split.sha256,
            "term_split": {"path": str(layout.split_path().relative_to(layout.relabel_root())),
                           "entries": len(term_split.entries)},
            "coverage": cov, "recordings": recordings,
            "selection": {t: drep._json(ts.selection_path(t)) for t in dict.fromkeys(ts.BASE.values())},
            "students": {s: student_result(s, split, term_split, cov) for s in td.STUDENTS},
            "runs": run_table(),
            "probes": {s: drep._json(td.root() / "synthetic-probe" / f"{s}.json") for s in td.STUDENTS}}


# ----------------------------------------------------------------------------- markdown

def _f(x, nd=2):
    return "–" if x is None else f"{x:.{nd}f}"


def _fmt(e):
    return "–" if not e else f"{e['point']:+.2f} [{e['ci'][0]:+.2f}, {e['ci'][1]:+.2f}]"


def markdown(r: dict) -> str:
    L = ["# Stage 2c results: synthetic technical speech in decoder-only training", "",
         "Generated by `train_synthetic.py report` (synthetic plan). A is the student's selected stage 2b T* run; TS "
         "and TSn are its synthetic arms (augmented and unaugmented synthetic clips) at A's learning rate. Recall and "
         "WER in points; intervals are 90% paired cluster-bootstrap intervals (10,000 replicates): recordings (talks, "
         "calls, lectures; a tech-dev-1 clip is its own recording) for term recall and domain WER, the stage 2b "
         "clusters on the evaluation part for H4. Recall contrasts are positive when the first model reproduces more; "
         "WER contrasts are positive when the first model is worse. These are development results.", ""]
    cov = r["coverage"]
    L += [f"Underpowered domains (coverage.json, out of the pooled claim): "
          f"{', '.join(cov['underpowered']) if cov['underpowered'] else 'none' if cov['complete'] else 'coverage missing'}."
          " Recordings per part: " + ", ".join(f"{p} {n}" for p, n in r["recordings"].items()) + ".", ""]
    for s, res in r["students"].items():
        L += [f"## {res['name']} ({res['arms']['TS']} and {res['arms']['TSn']} against {res['t_arm']})", ""]
        if res.get("status") != "complete":
            L.append(f"Status: {res.get('status')}.")
            continue
        sel = res["lr_selection"]
        L += [f"Learning rate {res['lr']}: the {res['t_arm']} selection (selection-part H4 "
              + ", ".join(f"{k} {v:.3f}" for k, v in sel["selection_h4"].items()) + f"; {sel['rule']}).", "",
              "Critical-term recall on eval-stem and dev-technical:", "",
              "| Group | Occurrences | A | TS | TSn | TS − A | TS − TSn | TSn − A |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for g, e in res["recall"]["groups"].items():
            if not e["occurrences"]:
                continue
            c = e["contrasts"]
            L.append(f"| {g} | {e['occurrences']} | {_f(e['recall']['A'], 1)} | {_f(e['recall']['TS'], 1)} | "
                     f"{_f(e['recall']['TSn'], 1)} | {_fmt(c.get('TS-A'))} | {_fmt(c.get('TS-TSn'))} | {_fmt(c.get('TSn-A'))} |")
        h = res["h4"]
        L += ["", "H4 (evaluation part) and its sets:", "", "| Set | WER A | WER TS | WER TSn | TS − A | TS − TSn | TSn − A |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for n in list(devsplit.H4) + ["H4"]:
            w = h["wer"]["mean"] if n == "H4" else h["wer"][n]
            e = h["mean"] if n == "H4" else h["sets"][n]
            L.append(f"| {n} | {w['A']:.2f} | {w['TS']:.2f} | {w['TSn']:.2f} | {_fmt(e['TS-A'])} | {_fmt(e['TS-TSn'])} | "
                     f"{_fmt(e['TSn-A'])} |")
        c = res["claims"]
        yes = {True: "yes", False: "no", None: "not evaluable"}
        L += ["", f"- Claim 1 (synthetic speech helps the targeted vocabulary): **{yes[c['synthetic_helps_targeted']]}** "
                  f"({c['rules']['claim1']}; H4 TS − A {c['h4_ts_minus_a_points']:+.2f}).",
              f"- Claim 2 (augmentation matters): **{yes[c['augmentation_matters']]}** ({c['rules']['claim2']}).",
              f"- TSn worse than A on H4 (no claim; a sign of overfitting to synthetic audio): "
              f"{yes[c['tsn_worse_than_t_star_on_h4']]}.", "",
              "Domain WER (descriptive; per part, and unweighted means of the parts per set and over both sets):", "",
              "| Part | WER A | WER TS | WER TSn | TS − A | TS − TSn |", "|---|---:|---:|---:|---:|---:|"]
        dw = res["domain_wer"]
        for p in dw["parts"]["sets"]:
            w = dw["parts"]["wer"][p]
            e = dw["parts"]["sets"][p]
            L.append(f"| {p} | {w['A']:.2f} | {w['TS']:.2f} | {w['TSn']:.2f} | {_fmt(e['TS-A'])} | {_fmt(e['TS-TSn'])} |")
        for name in (*TERM_SETS, "parts"):
            w, e = dw[name]["wer"]["mean"], dw[name]["mean"]
            label = "both sets" if name == "parts" else name
            L.append(f"| {label} (mean) | {w['A']:.2f} | {w['TS']:.2f} | {w['TSn']:.2f} | {_fmt(e['TS-A'])} | "
                     f"{_fmt(e['TS-TSn'])} |")
        L += ["", "Decoder-step proxy on full dev (pooled):", "",
              "| Model | Tokens / s audio | Decoder iterations / s audio | Mean predicted duration (frames) |",
              "|---|---:|---:|---:|"]
        for m in ("A", "TS", "TSn"):
            p = res["proxy"].get(m) or {}
            L.append(f"| {m} | {_f(p.get('tokens_per_second'), 3)} | {_f(p.get('iterations_per_second'), 3)} | "
                     f"{_f(p.get('mean_predicted_duration'), 3)} |")
        if not res["coverage_matches_scored_references"]:
            L += ["", "Note: the scored references' term occurrences differ from coverage.json's counts "
                      "(reference_occurrences in results-synthetic.json)."]
    L += ["", "## Stage 2c runs", "", "| Arm | lr | Steps | Planned batches | Synthetic rows | Real h | Synthetic h | "
          "Selection H4 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in r["runs"]:
        L.append(f"| {row['arm']} | {row.get('lr') or '–'} | {row.get('steps', '–')} | {row.get('planned_batches', '–')} | "
                 f"{row.get('synthetic_rows', '–')} | {_f(row.get('real_hours'), 1)} | {_f(row.get('synthetic_hours'), 1)} | "
                 f"{_f(row.get('selection_h4'), 3)} |")
    return "\n".join(L) + "\n"


def main(out: Path | None = None) -> int:
    relabel_paths.require_mount()
    r2b = drep.analyze_all()
    r = analyze_all()
    td.write_json(td.root() / "results.json", r2b)
    td.write_json(td.root() / "results-synthetic.json", r)
    path = Path(out) if out else td.root() / "RESULTS-decoder-only.md"
    path.write_text(drep.markdown(r2b) + "\n" + markdown(r))
    print(f"wrote {td.root() / 'results.json'}, {td.root() / 'results-synthetic.json'} and {path}")
    return 0
