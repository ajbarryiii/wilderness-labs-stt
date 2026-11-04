"""Step 1 of B0' v2 (b0prime-v2 plan, "Output style: verbatim or clean (Step 1, CPU, before any label is built)"): the
dev-WER effect of each output-style class, measured on B0''s saved dev hypotheses, and the preregistered rule.

    ./python b0v2_step1.py [--out DIR] [--replicates N] [--workers N]     (CPU; default $RELABEL_ROOT/b0prime-v2/step1/)

Measurement. For each class c (training/style.py: E, F, R, D) and dev set s, Delta_c(s) = WER(T_c applied to B0''s saved
hypotheses) - WER(B0''s hypotheses), in points, with the pinned scorer: evaluate.normalize of the transformed
hypothesis and edit_counts against the stored normalized reference (stage 1's scorer). Every stored record must be
reproduced by that scorer (its hyp_norm and S, D, I), else the set is refused. Unscored records (excluded references)
are left out. T_c's guard applies: a hypothesis it would empty is scored unchanged.

Sets. Decision sets (DECISION_SETS): H4's four sets, YODAS dev and the five domain parts, weighted equally. Reported,
not used (REPORTED_SETS): eval-stem's three parts, the cv27-en dev sample and AMI SDM dev. Convention groups (GROUPS)
are the plan's convention table over the decision sets.

Intervals. training/bootstrap.paired, 10,000 replicates, 90%: the ten decision sets form one bootstrap (each set's
clusters drawn from SeedSequence([seed, k]), k its index in sorted order; the same draws for B0''s and every class's
records), and each mean (the ten sets, H4, each group) is recomputed per replicate from those draws. Clusters
(cluster_fn): H4 as training/devsplit.py; YODAS by recording (the source video); the domain parts and eval-stem by
recording (synthetic_report.recording_of), except med-dev-2 by note (its conditions together); AMI SDM dev by meeting.
The reported sets with clusters form a second bootstrap; the cv27-en dev sample (no cluster definition) has a point
only.

Rule (preregistered). A class is adopted (cleaned) if the mean over the decision sets of Delta_c is below 0 and the
upper bound of its 90% interval is below 0; otherwise it stays as T* writes it (verbatim). Classes are decided
separately. If H4's mean has the opposite sign with its interval excluding 0, it is flagged (both are shown to AJ).
AJ approves the result before any label is built.

Also reported: Delta_c per set and per group, H4 alone, the reported sets, and Delta_c on Kea-B''s hypotheses (stage 4's
deployed x1 records) as a check; per class and set, the hypotheses changed and examples.

Inputs (dev records only; score_path refuses any other set): B0' = stage 3's b100,
$RELABEL_ROOT/b0prime/runs/b100/<eval-dev|domain|eval-stem|cv-dev>/<set>.json, and stage 4's re-score for AMI SDM dev,
$RELABEL_ROOT/keab-prime/scores/B0p/fp32/ami-sdm-dev.json; Kea-B' = $RELABEL_ROOT/keab-prime/scores/b100-main/x1/<set>.json
(the files keab_report.records reads). Outputs: step1.json (with every input's sha256 and the code's identity) and
RESULTS-step1.md, in $RELABEL_ROOT/b0prime-v2/step1/ or, with --out, a directory under it or under $RELABEL_ROOT/tmp/
(output_dir; the outputs hold dev transcripts, so nothing is written anywhere else).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from labeling import layout as llayout
from training import bootstrap, devsplit, style

H4_SETS = tuple(devsplit.H4)
DECISION_SETS = H4_SETS + ("yodas_dev", "med-dev-1", "med-dev-2", "med-dev-3", "tech-dev-1", "tech-dev-2")
STEM_SETS = ("stem-chem", "stem-cs", "stem-math")
REPORTED_SETS = STEM_SETS + ("cv27-en-dev", "ami-sdm-dev")
GROUPS = {"verbatim": ("ami_dev", "med-dev-1", "tech-dev-2"),
          "captions": ("yodas_dev",),
          "edited": ("voxpopuli_dev",),
          "scripted": ("med-dev-2",),
          "read": ("librispeech_dev_clean", "librispeech_dev_other", "med-dev-3", "tech-dev-1")}
GROUP_NOTES = {"verbatim": "verbatim references (AMI dev, med-dev-1, tech-dev-2)",
               "captions": "captions, mostly verbatim, abridged (YODAS dev)",
               "edited": "edited, fluent speech (VoxPopuli dev)",
               "scripted": "scripted synthetic dictation (med-dev-2)",
               "read": "read or scripted (LibriSpeech dev, med-dev-3, tech-dev-1)"}
REPLICATES = bootstrap.REPLICATES
CI = bootstrap.CI
SEED = bootstrap.SEED
MODELS = ("B0p", "Kea-B'")
EXAMPLES = 3
HERE = Path(__file__).resolve().parent

_YODAS = re.compile(r"^yodas:(en\d+_\d+_.{11})_")
_SEGMENT = re.compile(r"(?P<part>[^:]+):(?P<recording>.+?)-\d{4}(?:-\d{2})?")   # synthetic_report.recording_of
_NOTE = re.compile(r"(n\d+)-[^-]+")
_SDM = re.compile(r"^ami-sdm:AMI_([A-Za-z0-9]+)_")


def _recording(item_id: str) -> str:
    m = _SEGMENT.fullmatch(str(item_id))
    if m is None:
        raise ValueError(f"{item_id!r} is not a domain-set segment id")
    return f"{m['part']}:{m['recording']}"


def _yodas(item_id: str) -> str:
    m = _YODAS.match(item_id)
    if m is None:
        raise ValueError(f"cannot read the YODAS recording of {item_id!r}")
    return m.group(1)


def _note(item_id: str) -> str:
    part, _, recording = _recording(item_id).partition(":")
    m = _NOTE.fullmatch(recording)
    if m is None:
        raise ValueError(f"cannot read the med-dev-2 note of {item_id!r}")
    return f"{part}:{m.group(1)}"


def _sdm(item_id: str) -> str:
    m = _SDM.match(item_id)
    if m is None:
        raise ValueError(f"cannot read the AMI SDM meeting of {item_id!r}")
    return m.group(1)


def cluster_fn(name: str):
    """The cluster of an utterance id of a set, or None for a set without a cluster definition (cv27-en dev)."""
    if name in H4_SETS:
        return lambda i: devsplit.cluster_of(name, i)
    if name == "yodas_dev":
        return _yodas
    if name == "med-dev-2":
        return _note
    if name in DECISION_SETS or name in STEM_SETS:
        return _recording
    if name == "ami-sdm-dev":
        return _sdm
    if name == "cv27-en-dev":
        return None
    raise ValueError(f"no cluster definition for {name}")


def score_path(model: str, name: str) -> Path:
    """The saved dev records of B0' or Kea-B' on a decision or reported set; any other set or model is refused."""
    if name not in DECISION_SETS + REPORTED_SETS:
        raise ValueError(f"{name} is not a Step 1 dev set (decision {DECISION_SETS}, reported {REPORTED_SETS})")
    root = llayout.relabel_root()
    if model == "B0p":
        if name == "ami-sdm-dev":
            return root / "keab-prime" / "scores" / "B0p" / "fp32" / f"{name}.json"
        sub = ("eval-dev" if name in H4_SETS + ("yodas_dev",) else "eval-stem" if name in STEM_SETS
               else "cv-dev" if name == "cv27-en-dev" else "domain")
        return root / "b0prime" / "runs" / "b100" / sub / f"{name}.json"
    if model == "Kea-B'":
        return root / "keab-prime" / "scores" / "b100-main" / "x1" / f"{name}.json"
    raise ValueError(f"unknown model {model!r} (models: {MODELS})")


# ------------------------------------------------------------------------------------------------ scoring
def check_records(name: str, records: list[dict]) -> None:
    """Every scored record is reproduced by the pinned scorer (hyp_norm, and S, D, I against ref_norm)."""
    import evaluate
    for r in records:
        if not r.get("scored", True):
            continue
        hn = evaluate.normalize(r.get("hyp") or "")
        counts = tuple(evaluate.edit_counts(r["ref_norm"].split(), hn.split()))
        if hn != r["hyp_norm"] or counts != (r["S"], r["D"], r["I"]):
            raise ValueError(f"{name} {r['id']}: the stored record is not reproduced by the pinned scorer "
                             f"(hyp_norm {r['hyp_norm']!r} vs {hn!r}, S D I {(r['S'], r['D'], r['I'])} vs {counts})")


def rescore(records: list[dict], cls: str, st: style.Style) -> list[dict]:
    """The records with T_c applied to each hypothesis and rescored by the pinned scorer; "changed" and "edits" added
    (an unchanged hypothesis keeps its stored scores)."""
    import evaluate
    out = []
    for r in records:
        hyp, edits = st.edits(r.get("hyp") or "", cls)
        if hyp == (r.get("hyp") or ""):
            out.append({**r, "changed": False, "edits": 0})
            continue
        hn = evaluate.normalize(hyp)
        s, d, i = evaluate.edit_counts(r["ref_norm"].split(), hn.split())
        out.append({**r, "hyp": hyp, "hyp_norm": hn, "S": s, "D": d, "I": i, "changed": True, "edits": len(edits)})
    return out


def _prepare(job: tuple) -> tuple:
    name, records, st, classes = job
    check_records(name, records)
    scored = [r for r in records if r.get("scored", True)]
    out = {"base": records}
    changed, examples = {}, {}
    for c in classes:
        new = rescore(records, c, st)
        out[c] = new
        rows = [(r, t) for r, t in zip(records, new) if r.get("scored", True) and t["changed"]]
        changed[c] = {"hypotheses": len(rows), "edits": sum(t["edits"] for _, t in rows),
                      "errors_change": sum((t["S"] + t["D"] + t["I"]) - (r["S"] + r["D"] + r["I"]) for r, t in rows)}
        examples[c] = [{"id": r["id"], "ref": r.get("ref"), "hyp": r.get("hyp"), "after": t["hyp"],
                        "errors": [r["S"] + r["D"] + r["I"], t["S"] + t["D"] + t["I"]]}
                       for r, t in sorted(rows, key=lambda x: x[0]["id"])[:EXAMPLES]]
    return name, out, changed, examples, len(scored)


def _entry(e: dict) -> dict:
    return {"point": float(e["point"]), "ci": [float(x) for x in e["ci"]]}


def decide(mean: dict, h4: dict) -> dict:
    """The preregistered rule on the decision-set mean, and the H4 flag."""
    adopted = mean["point"] < 0 and mean["ci"][1] < 0
    h4_excludes_zero = h4["ci"][0] > 0 or h4["ci"][1] < 0
    return {"adopted": bool(adopted), "verdict": "cleaned" if adopted else "verbatim",
            "h4_opposite": bool(h4_excludes_zero and h4["point"] * mean["point"] < 0),
            "rule": "adopted if the decision-set mean of Delta_c is below 0 and the upper bound of its 90% interval is "
                    "below 0; H4 flagged if its mean has the opposite sign with its interval excluding 0"}


def measure(records: dict[str, list[dict]], st: style.Style, classes=style.CLASSES, replicates: int = REPLICATES,
            seed: int = SEED, workers: int = 1) -> dict:
    """Delta_c per set, per mean and per group, with intervals and the rule, on one model's records ({set: records})."""
    missing = [n for n in DECISION_SETS if n not in records]
    if missing:
        raise ValueError(f"decision sets missing: {', '.join(missing)}")
    unknown = sorted(set(records) - set(DECISION_SETS + REPORTED_SETS))
    if unknown:
        raise ValueError(f"not Step 1 sets: {', '.join(unknown)}")
    jobs = [(n, records[n], st, tuple(classes)) for n in DECISION_SETS + REPORTED_SETS if n in records]
    if workers > 1:
        from multiprocessing import Pool
        with Pool(min(workers, len(jobs))) as pool:
            prepared = pool.map(_prepare, jobs)
    else:
        prepared = [_prepare(j) for j in jobs]
    models = {name: out for name, out, _, _, _ in prepared}
    res = {"replicates": replicates, "seed": seed, "ci_level": CI, "classes": {c: {} for c in classes},
           "wer": {}, "utterances": {}, "clusters": {}, "reference_words": {},
           "changed": {c: {} for c in classes}, "examples": {c: {} for c in classes}}
    for name, _, changed, examples, n in prepared:
        res["utterances"][name] = n
        for c in classes:
            res["changed"][c][name] = changed[c]
            res["examples"][c][name] = examples[c]
    tables = {}
    for name, out in models.items():
        fn = cluster_fn(name)
        if fn is None:
            continue
        tables[name] = bootstrap.SetTable.from_records(out, cluster_of=fn)
        res["clusters"][name] = len(tables[name].clusters)
        res["reference_words"][name] = int(tables[name].words.sum())
        res["wer"][name] = {m: tables[name].wer(m) for m in ["base", *classes]}
    contrasts = {c: (c, "base") for c in classes}
    decision_tables = {n: tables[n] for n in DECISION_SETS}
    means = {"mean": list(DECISION_SETS), "h4": list(H4_SETS), **{f"group:{g}": list(m) for g, m in GROUPS.items()}}
    for key, sets in means.items():
        b = bootstrap.paired(decision_tables, contrasts, mean_sets=sets, replicates=replicates, seed=seed, ci=CI)
        for c in classes:
            e = res["classes"][c]
            if key == "mean":
                e["sets"] = {n: _entry(b["sets"][n][c]) for n in DECISION_SETS}
            if key.startswith("group:"):
                e.setdefault("groups", {})[key[6:]] = _entry(b["mean"][c])
            else:
                e[key] = _entry(b["mean"][c])
    reported = {n: tables[n] for n in REPORTED_SETS if n in tables}
    if reported:
        stem = [n for n in STEM_SETS if n in reported]
        b = bootstrap.paired(reported, contrasts, mean_sets=stem or sorted(reported), replicates=replicates, seed=seed,
                             ci=CI)
        for c in classes:
            res["classes"][c]["reported"] = {n: _entry(b["sets"][n][c]) for n in reported}
            if stem == list(STEM_SETS):
                res["classes"][c]["eval_stem"] = _entry(b["mean"][c])
    for name, out in models.items():
        if name in tables:
            continue
        rows = {m: [r for r in recs if r.get("scored", True)] for m, recs in out.items()}
        words = sum(len(r["ref_norm"].split()) for r in rows["base"])
        res["reference_words"][name] = words
        res["wer"][name] = {m: 100.0 * sum(r["S"] + r["D"] + r["I"] for r in rs) / words for m, rs in rows.items()}
        for c in classes:
            res["classes"][c].setdefault("reported", {})[name] = {
                "point": res["wer"][name][c] - res["wer"][name]["base"], "ci": None}
    for c in classes:
        e = res["classes"][c]
        e["decision"] = decide(e["mean"], e["h4"])
    return res


# ------------------------------------------------------------------------------------------------ the run
def _sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load(model: str) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """A model's records of every decision set (required) and reported set (if saved), with each file's sha256."""
    records, files = {}, {}
    for name in DECISION_SETS + REPORTED_SETS:
        p = score_path(model, name)
        if not p.exists():
            if name in DECISION_SETS:
                raise FileNotFoundError(f"{model}: {p} is missing")
            continue
        records[name] = json.loads(p.read_text())["records"]
        files[name] = {"path": str(p), "sha256": _sha256(p)}
    return records, files


def identity(st: style.Style) -> dict:
    import evaluate
    try:
        commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(HERE), "status", "--porcelain", "--", "."], capture_output=True,
                                    text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"style": st.identity(), "normalizer_sha256": evaluate.NORMALIZER_SHA256,
            "sources": {k: _sha256(HERE / k) for k in ("b0v2_step1.py", "training/style.py", "training/bootstrap.py")},
            "git": {"commit": commit, "dirty": dirty}}


def _ci(e: dict) -> str:
    if e.get("ci") is None:
        return f"{e['point']:+.3f}"
    return f"{e['point']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]"


def markdown(r: dict) -> str:
    b = r["models"]["B0p"]
    classes = list(b["classes"])
    L = ["# B0′ v2 Step 1: output style, verbatim or clean", "",
         "Generated by `b0v2_step1.py` (b0prime-v2 plan, \"Output style\"). Δ_c = WER(T_c(B0′'s saved dev hypotheses)) − "
         "WER(B0′'s hypotheses), points, pinned scorer; 90% paired cluster-bootstrap intervals, "
         f"{b['replicates']:,} replicates, the ten-set mean recomputed per replicate. **Dev-informed: AJ approves the "
         "result before any label is built.**", "",
         "## Decision (preregistered rule)", "",
         "| Class | Mean Δ over the 10 decision sets | H4 Δ | Rule | H4 opposite sign |", "|---|---:|---:|---|---|"]
    for c in classes:
        e = b["classes"][c]
        d = e["decision"]
        L.append(f"| {c} | {_ci(e['mean'])} | {_ci(e['h4'])} | **{d['verdict']}** | {'yes' if d['h4_opposite'] else 'no'} |")
    L += ["", f"Rule: {b['classes'][classes[0]]['decision']['rule']}.", "", "## Per decision set (B0′)", "",
          "| Set | Words | Clusters | WER | " + " | ".join(f"Δ_{c}" for c in classes) + " |",
          "|---|---:|---:|---:|" + "---:|" * len(classes)]
    for n in DECISION_SETS:
        L.append(f"| {n} | {b['reference_words'][n]:,} | {b['clusters'][n]} | {b['wer'][n]['base']:.2f} | "
                 + " | ".join(_ci(b["classes"][c]["sets"][n]) for c in classes) + " |")
    L += ["", "## Per convention group (B0′)", "", "| Group | " + " | ".join(f"Δ_{c}" for c in classes) + " |",
          "|---|" + "---:|" * len(classes)]
    for g in GROUPS:
        L.append(f"| {GROUP_NOTES[g]} | " + " | ".join(_ci(b["classes"][c]["groups"][g]) for c in classes) + " |")
    L += ["", "## Reported sets, not used (B0′)", "", "| Set | WER | " + " | ".join(f"Δ_{c}" for c in classes) + " |",
          "|---|---:|" + "---:|" * len(classes)]
    for n in REPORTED_SETS:
        if n in b["wer"]:
            L.append(f"| {n} | {b['wer'][n]['base']:.2f} | "
                     + " | ".join(_ci(b["classes"][c]["reported"][n]) for c in classes) + " |")
    if "eval_stem" in b["classes"][classes[0]]:
        L.append("| eval-stem mean | | " + " | ".join(_ci(b["classes"][c]["eval_stem"]) for c in classes) + " |")
    k = r["models"].get("Kea-B'")
    if k:
        L += ["", "## Check: the same transforms on Kea-B′'s hypotheses (deployed x1)", "",
              "| Class | Mean Δ (10 sets) | H4 Δ | Rule on Kea-B′ |", "|---|---:|---:|---|"]
        for c in classes:
            e = k["classes"][c]
            L.append(f"| {c} | {_ci(e['mean'])} | {_ci(e['h4'])} | {e['decision']['verdict']} |")
    L += ["", "## Hypotheses changed (B0′; scored utterances)", "",
          "| Set | Utterances | " + " | ".join(f"{c}: changed / Δerrors" for c in classes) + " |",
          "|---|---:|" + "---:|" * len(classes)]
    for n in DECISION_SETS + REPORTED_SETS:
        if n in b["utterances"]:
            L.append(f"| {n} | {b['utterances'][n]:,} | " + " | ".join(
                f"{b['changed'][c][n]['hypotheses']} / {b['changed'][c][n]['errors_change']:+d}" for c in classes) + " |")
    L += ["", "## Examples (B0′, first changed hypotheses by id)", ""]
    for c in classes:
        L += [f"### {c}", ""]
        for n in DECISION_SETS:
            for x in b["examples"][c][n][:2]:
                L.append(f"- {n} `{x['id']}` (errors {x['errors'][0]} → {x['errors'][1]}): “{x['hyp']}” → “{x['after']}”"
                         f" (ref “{x['ref']}”)")
        L.append("")
    return "\n".join(L) + "\n"


def output_dir(arg: Path | None) -> Path:
    """$RELABEL_ROOT/b0prime-v2/step1 by default; a given directory must resolve under it or under $RELABEL_ROOT/tmp."""
    root = llayout.relabel_root().resolve()
    default = root / "b0prime-v2" / "step1"
    if arg is None:
        return default
    p = Path(arg).resolve()
    for allowed in (default, root / "tmp"):
        if p == allowed or allowed in p.parents:
            return p
    raise ValueError(f"{arg}: Step 1 writes only under {default} or {root / 'tmp'}")


def _write(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="b0v2_step1.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=None, help="output directory (default $RELABEL_ROOT/b0prime-v2/step1)")
    ap.add_argument("--replicates", type=int, default=REPLICATES)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    llayout.require_mount()
    out = output_dir(args.out)
    out.mkdir(parents=True, exist_ok=True)
    st = style.Style.load()
    result = {"plan": "b0prime-v2 plan, Step 1", "identity": identity(st), "inputs": {}, "models": {}}
    for model in MODELS:
        records, files = load(model)
        result["inputs"][model] = files
        result["models"][model] = measure(records, st, replicates=args.replicates, workers=args.workers)
        print(f"{model}: " + ", ".join(f"{c} {_ci(e['mean'])} -> {e['decision']['verdict']}"
                                       for c, e in result["models"][model]["classes"].items()), flush=True)
    result["decision"] = {c: e["decision"] for c, e in result["models"]["B0p"]["classes"].items()}
    _write(out / "step1.json", json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    _write(out / "RESULTS-step1.md", markdown(result))
    print(f"wrote {out / 'step1.json'} and {out / 'RESULTS-step1.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
