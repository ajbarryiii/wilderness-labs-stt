"""Reference coverage of the split's terms in the real evaluation sets (synthetic plan "Evaluation: Coverage").

Before any training, the number of reference occurrences of targeted and held-out terms is counted per domain in
eval-stem and dev-technical (every part of each set present under $RELABEL_ROOT/devsets/manifests/), with the
split's matcher (vocab.Split.matcher: the scorer's greedy longest-match, non-overlapping occurrences of every term
of every domain and side). Only references are read (the manifests' "text", normalized as the scorer does).

A domain with fewer than MIN_TARGETED targeted-term occurrences over eval-stem and dev-technical together is
UNDERPOWERED and stays out of the pooled claim. Missing parts are listed; the result is final only when every part
of both sets is present ("complete").
Output: $RELABEL_ROOT/synthetic/terms/coverage.json.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

from synthetic import layout, vocab

SETS = ("eval-stem", "dev-technical")
MIN_TARGETED = 100


def count(split: vocab.Split, texts_by_part: dict[str, list[str]]) -> dict:
    per_part = {}
    for part, texts in texts_by_part.items():
        acc = defaultdict(lambda: {"targeted": 0, "heldout": 0})
        terms = defaultdict(set)
        for t in texts:
            for o in split.occurrences(t):
                acc[o["domain"]][o["side"]] += 1
                terms[(o["domain"], o["side"])].add(o["phrase"])
        per_part[part] = {d: {**acc[d], "distinct_targeted": len(terms[(d, "targeted")]),
                              "distinct_heldout": len(terms[(d, "heldout")])} for d in layout.DOMAINS}
    totals = {d: {s: sum(per_part[p][d][s] for p in per_part) for s in ("targeted", "heldout")} for d in layout.DOMAINS}
    return {"parts": per_part, "totals": totals,
            "underpowered": [d for d in layout.DOMAINS if totals[d]["targeted"] < MIN_TARGETED],
            "min_targeted": MIN_TARGETED}


def run(split: vocab.Split | None = None) -> dict:
    import devsets
    from devsets import layout as dlayout
    layout.require_mount()
    split = split or vocab.Split.load()
    lay = dlayout.Layout.production()
    texts, missing = {}, []
    for name in SETS:
        for part in devsets.SETS[name]:
            p = lay.manifest(part)
            if not p.exists():
                missing.append(part)
                continue
            with open(p, encoding="utf-8") as f:
                texts[part] = [json.loads(line)["text"] for line in f if line.strip()]
    res = count(split, texts) | {"missing_parts": missing, "complete": not missing,
                                 "split_entries": len(split.entries)}
    out = layout.terms_dir() / "coverage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(res, indent=1) + "\n")
    os.replace(tmp, out)
    return res
