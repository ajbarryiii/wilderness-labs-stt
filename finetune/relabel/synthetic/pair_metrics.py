"""Metrics of the confusable pairs (synthetic plan v5, "Metrics"): CPU re-scoring of saved hypotheses.

    ./python -m synthetic pairs-metrics --model NAME TECH_DIR STEM_DIR [--model ...] [--contrast A B ...] --out NAME

Records: evaluate.py's per-utterance records ({"id", "ref_norm", "hyp_norm", "scored"}) of every part of
dev-technical (TECH_DIR/<part>.json) and eval-stem (STEM_DIR/<part>.json), as synthetic_report reads them. Occurrences
are score_domain's aligned occurrences (term_occurrences: the split's matcher, greedy longest match, the spelling-variant
rule); v2 and v3 have the same terms and groups, so the occurrences are the same under both and only sides differ.

Per occurrence (occurrences()):
- span: the hypothesis words aligned to the occurrence (relabel_align: matches and substitutions of its reference
  words) and inserted in its gaps, the gaps at both ends included (score_domain._variant_at's window);
- partner substitution, per listed pair of split v3 ("pairs": the accepted pairs, held-out-held-out ones included,
  and the seed) of which the occurrence's group is a member: the span holds the partner's paired token and not the
  term's own paired token, for one of the pair's token pairs ("hits") whose own token is in the occurrence's
  reference words. Tokens and words are compared on normalized letters (the evaluation normalizer, then letters
  only). A pair none of whose own tokens is in the occurrence is not measured there;
- near form, at occurrences where a partner substitution is measured: not reproduced (not a listed form), no partner
  substitution, and the edit distance of the term's and the span's letters (normalized, joined without spaces or
  hyphens) at most max(2, ceil(0.3 x the term's letter count)).

Groups (analyze; recall or share in points, each from the counts of its keys):
    stable-heldout                v2's held-out groups minus the moved ones (= v3's held-out groups)
    pair-recall                   the moved groups; pair-recall-no-seed without the dev-informed seed's
    targeted-v3, v3/<d>/<side>    v3's groups, all domains and per domain
    v2/<side>, v2/<d>/<side>      v2's groups (continuity; equal to vocab.recall's counts)
    partner-substitution[/<pair>] the share of measured occurrences with a partner substitution, pooled (an
                                  occurrence counted once) and per pair
    near-forms[/<pair>]           the same for near forms
Intervals: synthetic_report.recall_bootstrap (v1's paired cluster bootstrap over recordings, 10,000 replicates, 90%)
for each requested contrast (a, b) = value(a) - value(b).
Output: $RELABEL_ROOT/synthetic/v5/metrics/<NAME>.json.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from functools import partial
from pathlib import Path

from synthetic import layout, vocab

TECH_PARTS = ("tech-dev-1", "tech-dev-2")
STEM_PARTS = ("stem-chem", "stem-math", "stem-cs")


def _letters(word: str) -> str:
    from synthetic.confusable import fold
    return "".join(c for c in fold(word) if "a" <= c <= "z")


def _normal_token(token: str) -> str:
    import evaluate
    return _letters(evaluate.normalize(token))


def moved_groups(v2: vocab.Split, v3: vocab.Split) -> set[str]:
    """v3's recorded moves (checked: exactly v2's held-out groups that v3 targets)."""
    moved = {m["group"] for m in v3.data.get("moved", [])}
    side3 = {e.group: e.side for e in v3.entries}
    by_side = {e.group for e in v2.heldout() if side3[e.group] == "targeted"}
    if moved != by_side:
        raise RuntimeError("split v3's recorded moves differ from its sides")
    return moved


def stable_heldout_groups(v2: vocab.Split, v3: vocab.Split) -> set[str]:
    return {e.group for e in v2.heldout()} - moved_groups(v2, v3)


class _Pairs:
    """Split v3's listed pairs per member group: (pair id, [(own token, partner token)]) in letters."""

    def __init__(self, v3: vocab.Split):
        self.by_group = defaultdict(list)
        for p in v3.data.get("pairs", []):
            for k, g in enumerate(p["groups"]):
                hits = [(_normal_token(h["tokens"][k]), _normal_token(h["tokens"][1 - k])) for h in p["hits"]]
                self.by_group[g].append((p["id"], hits))


def _span(alignment, hyp: list[str], start: int, end: int) -> list[str]:
    window = [j for op, i, j in alignment.pairs if op in ("eq", "sub") and start <= i < end]
    window += [j for op, g, j in alignment.pairs if op == "ins" and start <= g <= end]
    return hyp[min(window):max(window) + 1] if window else []


def occurrences(ref_norm: str, hyp_norm: str, v3: vocab.Split, pairs: _Pairs | None = None) -> list[dict]:
    """Every term occurrence of a scored record (module docstring): {start, end, phrase, group, domain, side (v3),
    ok, span, substituted: {pair id: bool} for the pairs measured there, near}."""
    import relabel_align
    import score_domain
    from synthetic.confusable import levenshtein
    pairs = pairs or _Pairs(v3)
    found = score_domain.term_occurrences(ref_norm, hyp_norm, v3.matcher())
    if not found:
        return []
    ref, hyp = ref_norm.split(), hyp_norm.split()
    a = relabel_align.align(ref, hyp)
    out = []
    for s, e, ok in found:
        phrase = tuple(ref[s:e])
        entry = v3.by_phrase[phrase]
        span = _span(a, hyp, s, e)
        in_ref = {_letters(w) for w in phrase}
        in_span = {_letters(w) for w in span}
        sub = {}
        for pid, hits in pairs.by_group.get(entry.group, ()):
            measured = [(own, other) for own, other in hits if own in in_ref]
            if measured:
                sub[pid] = any(other in in_span and own not in in_span for own, other in measured)
        near = False
        if sub and not ok and not any(sub.values()):
            t, h = "".join(_letters(w) for w in phrase), "".join(_letters(w) for w in span)
            cap = max(2, math.ceil(0.3 * len(t)))
            near = levenshtein(t, h, cap) <= cap
        out.append({"start": s, "end": e, "phrase": " ".join(phrase), "group": entry.group, "domain": entry.domain,
                    "side": entry.side, "ok": bool(ok), "span": " ".join(span), "substituted": sub, "near": near})
    return out


def _keys(o: dict, v2_side: dict, stable: set, moved: set, seed: set) -> list[tuple[tuple[str, str], bool]]:
    """(table key, counts as reproduced) for one occurrence. Keys are (kind, name) pairs, as
    synthetic_report.recall_bootstrap reports them ("kind/name")."""
    g = o["group"]
    out = [(("v2", f"{o['domain']}/{v2_side[g]}"), o["ok"]), (("v3", f"{o['domain']}/{o['side']}"), o["ok"])]
    if g in stable:
        out.append((("stable", "heldout"), o["ok"]))
    if g in moved:
        out.append((("moved", "seed" if g in seed else "curated"), o["ok"]))
    if o["substituted"]:
        out += [(("sub", pid), v) for pid, v in o["substituted"].items()]
        out += [(("near", pid), o["near"]) for pid in o["substituted"]]
        out += [(("sub", "any"), any(o["substituted"].values())), (("near", "any"), o["near"])]
    return out


def tables(parts: dict[str, dict[str, list[dict]]], v2: vocab.Split, v3: vocab.Split) -> dict:
    """{part: synthetic_report.RecallTable} over this module's keys (every model's records of a part: the same
    utterances and references, scored records only; clusters are recordings)."""
    import numpy as np
    import synthetic_report as srep
    v2_side = {e.group: e.side for e in v2.entries}
    stable, moved = stable_heldout_groups(v2, v3), moved_groups(v2, v3)
    seed = set(v3.data.get("dev_informed_groups", []))
    pairs = _Pairs(v3)
    counted = {}
    for part, records in parts.items():
        models = sorted(records)
        keyed = {m: {r["id"]: r for r in records[m]} for m in models}
        first = keyed[models[0]]
        for m in models[1:]:
            if set(keyed[m]) != set(first):
                raise ValueError(f"{part}: model {m} does not cover the same utterances as {models[0]}")
        chosen = sorted(i for i in first if first[i].get("scored", True))
        for m in models:
            for i in chosen:
                if not keyed[m][i].get("scored", True) or keyed[m][i]["ref_norm"] != first[i]["ref_norm"]:
                    raise ValueError(f"{i}: scored or reference differs between {models[0]} and {m}")
        names = sorted({srep.recording_of(i) for i in chosen})
        index = {c: j for j, c in enumerate(names)}
        zeros = partial(np.zeros, len(names))       # this part's size: a key filled in later is this part's length
        occ = defaultdict(zeros)
        rep = {m: defaultdict(zeros) for m in models}
        for m in models:
            for i in chosen:
                r, c = keyed[m][i], index[srep.recording_of(i)]
                for o in occurrences(r["ref_norm"], r["hyp_norm"], v3, pairs):
                    for key, hit in _keys(o, v2_side, stable, moved, seed):
                        if m == models[0]:
                            occ[key][c] += 1
                        rep[m][key][c] += int(hit)
        counted[part] = (names, occ, rep, len(chosen))
    universe = sorted({k for _, occ, rep, _ in counted.values() for k in (*occ, *(k for r in rep.values() for k in r))})
    return {part: srep.RecallTable(names, {k: occ[k] for k in universe},
                                   {m: {k: r[k] for k in universe} for m, r in rep.items()}, n)
            for part, (names, occ, rep, n) in counted.items()}


def metric_groups(v3: vocab.Split, keys: set[tuple]) -> dict[str, set[tuple]]:
    """The reported groups (module docstring) over the keys present."""
    pair_ids = [p["id"] for p in v3.data.get("pairs", [])]
    out = {"stable-heldout": {("stable", "heldout")}, "pair-recall": {("moved", "curated"), ("moved", "seed")},
           "pair-recall-no-seed": {("moved", "curated")},
           "targeted-v3": {("v3", f"{d}/targeted") for d in layout.DOMAINS}}
    out |= {f"v3/{d}/{s}": {("v3", f"{d}/{s}")} for d in layout.DOMAINS for s in vocab.SIDES}
    out |= {f"v2/{s}": {("v2", f"{d}/{s}") for d in layout.DOMAINS} for s in vocab.SIDES}
    out |= {f"v2/{d}/{s}": {("v2", f"{d}/{s}")} for d in layout.DOMAINS for s in vocab.SIDES}
    out |= {"partner-substitution": {("sub", "any")}, "near-forms": {("near", "any")}}
    out |= {f"partner-substitution/{pid}": {("sub", pid)} for pid in pair_ids}
    out |= {f"near-forms/{pid}": {("near", pid)} for pid in pair_ids}
    return {name: {k for k in ks if k in keys} for name, ks in out.items()}


def analyze(parts: dict[str, dict[str, list[dict]]], v2: vocab.Split, v3: vocab.Split,
            contrasts: dict[str, tuple[str, str]] | None = None, replicates: int | None = None,
            seed: int | None = None) -> dict:
    """Point values of every group for every model, and the contrasts' paired cluster-bootstrap intervals."""
    import synthetic_report as srep
    from training import bootstrap
    tabs = tables(parts, v2, v3)
    models = sorted({m for t in tabs.values() for m in t.reproduced})
    contrasts = dict(contrasts or {})
    # recall_bootstrap reports the models of its contrasts: each model also enters as a contrast with itself, removed
    # from the output (its point and interval are 0)
    own = {f"{m}-{m}": (m, m) for m in models}
    keys = {k for t in tabs.values() for k in t.occurrences}
    groups = metric_groups(v3, keys)          # a group without occurrences sums to zeros: recall None, no interval
    res = srep.recall_bootstrap(tabs, own | contrasts, groups, replicates or bootstrap.REPLICATES,
                                bootstrap.SEED if seed is None else seed)
    for entry in res["groups"].values():
        entry["contrasts"] = {k: v for k, v in entry["contrasts"].items() if k not in own or k in contrasts}
    res |= {"plan": "synthetic plan v5, Confusable pairs: Metrics", "models": models,
            "contrasts_requested": {k: list(v) for k, v in contrasts.items()},
            "stable_heldout_groups": len(stable_heldout_groups(v2, v3)),
            "moved_groups": sorted(moved_groups(v2, v3)),
            "dev_informed_groups": sorted(v3.data.get("dev_informed_groups", [])),
            "pairs": [{"id": p["id"], "terms": p["terms"], "sides_v2": p["sides_v2"], "trained": p["trained"],
                       "seed": p["seed"]} for p in v3.data.get("pairs", [])]}
    return res


def load_parts(models: dict[str, tuple[Path, Path]]) -> dict[str, dict[str, list[dict]]]:
    """{part: {model: records}} from each model's dev-technical and eval-stem directories of part files."""
    out = {}
    for parts, k in ((TECH_PARTS, 0), (STEM_PARTS, 1)):
        for part in parts:
            out[part] = {m: json.loads((Path(dirs[k]) / f"{part}.json").read_text())["records"]
                         for m, dirs in models.items()}
    return out


def main(models: dict[str, tuple[Path, Path]], contrasts: dict[str, tuple[str, str]], name: str) -> dict:
    v2 = vocab.Split.load(layout.split_version_path(2))
    v3 = vocab.Split.load(layout.split_version_path(3))
    res = analyze(load_parts(models), v2, v3, contrasts)
    res["inputs"] = {m: {"dev-technical": str(d[0]), "eval-stem": str(d[1])} for m, d in models.items()}
    out = layout.v5_dir() / "metrics" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    return res
