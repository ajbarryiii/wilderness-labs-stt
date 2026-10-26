"""Term recall of the saved baseline hypotheses, before and after the 2026-10-24 spelling-variant rule (CPU only).

    ./python term_rescore.py --old-lists DIR [--old-split FILE] [--out NAME]

Reads the per-part score files score_domain.py wrote ($RELABEL_ROOT/devsets/scores/<model>/ for dev-medical and
dev-technical, scores/<set>/<model>/ for eval-ext and eval-stem) and never writes to them. No decoding: the saved
"ref_norm" / "hyp_norm" of every scored record are rescored.

- old: the term lists in DIR (a snapshot with terms/'s layout: DIR/medical, DIR/technical) and the rule before
  the change (score_domain.term_occurrences(variants=False)). Where a part file has a stored term_recall, old must
  equal it exactly (checked; a mismatch fails the run).
- new: the committed lists and the current rule (listed spelling variants are one term).
Term list per set: devsets.TERM_KIND, and the technical list for dev-technical (which score_domain.py does not
score for terms). Stage 2c's split recall (vocab.recall: per domain and side, pooled) on eval-stem and
dev-technical uses --old-split (old, with the old rule) and the current split (new).

Output: $RELABEL_ROOT/devsets/term-rescore/<NAME>/results.json (per model, set and part: occurrences, reproduced,
recall, old and new; per set the unweighted mean of part recalls and the pooled counts), and a markdown table on
stdout.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import devsets
import relabel_paths
import score_domain
import terms

MODELS = score_domain.MODELS
SETS = ("dev-medical", "dev-technical", "eval-ext", "eval-stem")
KIND = dict(devsets.TERM_KIND) | {"dev-technical": "technical"}
SPLIT_SETS = ("eval-stem", "dev-technical")


def scores_dir(model: str, name: str) -> Path:
    base = relabel_paths.ARTIFACTS / "devsets" / "scores"
    return base / model if name in devsets.DEFAULT_SETS else base / name / model


def recall(records: list[dict], matcher, variants: bool) -> dict:
    occ = rep = 0
    for r in records:
        if not r.get("scored", True):
            continue
        found = score_domain.term_occurrences(r["ref_norm"], r["hyp_norm"], matcher, variants=variants)
        occ += len(found)
        rep += sum(t[2] for t in found)
    return {"occurrences": occ, "reproduced": rep, "recall": rep / occ if occ else None}


def set_entry(parts: dict) -> dict:
    rs = [p["recall"] for p in parts.values()]
    occ = sum(p["occurrences"] for p in parts.values())
    rep = sum(p["reproduced"] for p in parts.values())
    return {"mean_recall": sum(rs) / len(rs) if all(r is not None for r in rs) else None,
            "pooled": {"occurrences": occ, "reproduced": rep, "recall": rep / occ if occ else None}}


def split_recall(records: list[dict], split, variants: bool) -> dict:
    """vocab.recall with the old or the new occurrence rule (pooled over domains, per side)."""
    from synthetic import vocab
    acc = {s: [0, 0] for s in vocab.SIDES}
    per = {}
    m = split.matcher()
    for r in records:
        if not r.get("scored", True):
            continue
        for start, end, ok in score_domain.term_occurrences(r["ref_norm"], r["hyp_norm"], m, variants=variants):
            domain, side = split.classify(tuple(r["ref_norm"].split()[start:end]))
            acc[side][0] += 1
            acc[side][1] += int(ok)
            c = per.setdefault(domain, {s: [0, 0] for s in vocab.SIDES})[side]
            c[0] += 1
            c[1] += int(ok)
    fmt = lambda n, k: {"occurrences": n, "reproduced": k, "recall": k / n if n else None}  # noqa: E731
    return {"pooled": {s: fmt(*acc[s]) for s in acc},
            "domains": {d: {s: fmt(*v[s]) for s in v} for d, v in sorted(per.items())}}


def run(old_lists: Path, old_split: Path | None) -> dict:
    from synthetic import vocab
    new_m = {k: terms.matcher(k) for k in terms.KINDS}
    old_m = {k: terms.Matcher(terms.load(k, base=old_lists), _normalize()) for k in terms.KINDS}
    splits = {"new": vocab.Split.load()}
    if old_split is not None:
        splits["old"] = vocab.Split(json.loads(Path(old_split).read_text()))
    out = {"models": {}, "old_lists": str(old_lists.name), "rule_new": "listed spelling variants are one term"}
    for model in MODELS:
        mo = out["models"][model] = {}
        for name in SETS:
            d = scores_dir(model, name)
            parts, records = {}, []
            for part in devsets.SETS[name]:
                f = d / f"{part}.json"
                if not f.exists():
                    raise SystemExit(f"{f} is missing (score {model} on {name} first)")
                res = json.loads(f.read_text())
                recs = res["records"]
                kind = KIND[name]
                old = recall(recs, old_m[kind], variants=False)
                stored = res.get("term_recall")
                if stored is not None and (stored["occurrences"], stored["reproduced"]) != (old["occurrences"],
                                                                                            old["reproduced"]):
                    raise SystemExit(f"{f}: the old rule gives {old}, the stored term_recall is {stored}")
                parts[part] = {"old": old, "new": recall(recs, new_m[kind], variants=True),
                               "stored_matches_old": stored is not None}
                records += recs
            entry = {"kind": KIND[name], "parts": parts,
                     "old": set_entry({p: v["old"] for p, v in parts.items()}),
                     "new": set_entry({p: v["new"] for p, v in parts.items()})}
            if name in SPLIT_SETS:
                entry["split"] = {k: split_recall(records, s, variants=(k == "new")) for k, s in splits.items()}
            mo[name] = entry
    return out


def _normalize():
    import evaluate
    return evaluate.normalize


def table(res: dict) -> str:
    def pct(x):
        return "-" if x is None else f"{100 * x:.2f}"
    lines = ["| model | set | list | old pooled (occ) | new pooled (occ) | old mean | new mean |",
             "|---|---|---|---|---|---|---|"]
    for model, sets in res["models"].items():
        for name, e in sets.items():
            o, n = e["old"], e["new"]
            lines.append(f"| {model} | {name} | {e['kind']} | {pct(o['pooled']['recall'])} ({o['pooled']['occurrences']}) "
                         f"| {pct(n['pooled']['recall'])} ({n['pooled']['occurrences']}) | {pct(o['mean_recall'])} "
                         f"| {pct(n['mean_recall'])} |")
    lines += ["", "| model | set | split | targeted (occ) | held out (occ) |", "|---|---|---|---|---|"]
    for model, sets in res["models"].items():
        for name, e in sets.items():
            for k, v in e.get("split", {}).items():
                t, h = v["pooled"]["targeted"], v["pooled"]["heldout"]
                lines.append(f"| {model} | {name} | {k} | {pct(t['recall'])} ({t['occurrences']}) "
                             f"| {pct(h['recall'])} ({h['occurrences']}) |")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--old-lists", type=Path, required=True)
    p.add_argument("--old-split", type=Path)
    p.add_argument("--out", default="variants-2026-10-24")
    args = p.parse_args(argv)
    relabel_paths.require_mount()
    res = run(args.old_lists, args.old_split)
    out = relabel_paths.ARTIFACTS / "devsets" / "term-rescore" / args.out / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(res, indent=1) + "\n")
    os.replace(tmp, out)
    print(table(res))
    print(f"\nwritten {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
