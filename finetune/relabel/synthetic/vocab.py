"""Stage 2c term lists and their frozen targeted / held-out split (synthetic plan "Terms").

Inputs: the committed lists terms/synthetic/<domain>.txt (one term per line, '#' comments), domains
layout.DOMAINS. build_split() turns them into the frozen split:

1. Normalization: each term is normalized as the scorer normalizes references (the evaluation normalizer,
   finetune/parakeet-ternary/evaluate.py normalize) and split into words; a term that normalizes to nothing or
   to one character is dropped, as terms.Matcher drops it.
2. Medical / TCCC exclusion: a term whose normalized words contain an occurrence of the medical term lists
   (terms.matcher("medical"): drugs, anatomy, procedures, conditions, dosages and units, TCCC) is dropped.
3. Cross-domain de-duplication: a normalized phrase listed in several domains is kept in the first domain of
   layout.DOMAINS that lists it (and recorded). Spelling variants (terms.variant_key, 2026-10-24: listed phrases
   whose words joined without spaces are equal, e.g. "dataset" / "data set") are one term: every variant takes the
   domain of the first domain that lists any of them (recorded under "variant_domain").
4. Variant groups (union-find over every kept phrase of every domain): two phrases are one group if they are
   equal after reducing each word to its singular form (plural_key), or if they differ only in a final "s" / "es"
   on the last word and both are listed (plurals plural_key leaves alone, e.g. acronyms such as "apis", "gpus"),
   or if one is a contiguous sub-phrase of the other in plural_key form (a term nested inside a longer term),
   or if they are spelling variants (step 3). Groups can span domains.
5. Split: a group is TARGETED iff the first 8 bytes of sha256(SPLIT_SALT + "\\n" + group key), read as a
   big-endian integer, are below TARGETED_SHARE * 2**64; otherwise HELD OUT. The group key is the smallest of
   its members' canonical keys (canonical_key, 2026-10-24: the words joined WITHOUT spaces, last word singular; a
   function of the letters only), so a spelling variant shares its term's letter key. Adding one moves no group
   unless it also creates a new nesting link to another group (a variant such as "co-variance" whose words nest in
   another listed phrase), as adding any phrase can; the 2026-10-24 additions create none (split-changes.json). A term's side is its group's side, so a term
   and its variants are always on the same side.

The split is written once to $RELABEL_ROOT/synthetic/terms/split.json (frozen: a later build refuses to replace
it unless the lists and every rule give the same result) with the SHA-256 of each list; Split.load() refuses
lists that no longer match it.

Matching (frozen before training; the scorer's own definition):
- Split.matcher(): one terms.Matcher over every kept phrase of every domain and side. Its greedy longest-match,
  left-to-right, non-overlapping occurrences (score_domain.term_occurrences) are each attributed to the phrase's
  (domain, side). Overlapping phrases are therefore counted exactly as score_domain counts them.
- Split.heldout_matcher(): only the held-out phrases (any domain), as the scorer matches them.
- Split.heldout_guard(words) / has_heldout(text): the held-out filter of hygiene, a leak guard wider than the
  scorer: any run of consecutive normalized words that, joined without spaces, is a surface form
  (plural_surfaces: the canonical key, or its -s, -es, -ies or IRREGULAR plural) of a held-out phrase, at any
  length. So unlisted joined, split and hyphenated forms of a held-out term, and their regular and listed irregular
  plurals, are caught. It may reject a few extra sentences; it is never used for scoring.
- Split.targeted(domain): the targeted terms used to write prompts.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import terms as committed_terms   # finetune/relabel/terms (Part A API)
from synthetic import layout

LISTS_DIR = committed_terms.HERE / "synthetic"
SIDES = ("targeted", "heldout")
VERSION = 2          # 2 (2026-10-24): spelling variants are one term (one group, one domain)

# Irregular plurals common in the four domains (plural -> singular).
IRREGULAR = {"matrices": "matrix", "vertices": "vertex", "indices": "index", "indexes": "index",
             "appendices": "appendix", "radii": "radius", "nuclei": "nucleus", "foci": "focus", "loci": "locus",
             "criteria": "criterion", "phenomena": "phenomenon", "polyhedra": "polyhedron", "data": "datum",
             "analyses": "analysis", "hypotheses": "hypothesis", "theses": "thesis", "bases": "basis",
             "axes": "axis", "syntheses": "synthesis", "parentheses": "parenthesis", "ellipses": "ellipsis",
             "spectra": "spectrum", "maxima": "maximum", "minima": "minimum", "optima": "optimum",
             "formulae": "formula", "lemmata": "lemma", "schemata": "schema", "corpora": "corpus",
             "genera": "genus", "media": "medium", "quanta": "quantum", "moduli": "modulus", "tori": "torus",
             "simplices": "simplex", "apices": "apex", "children": "child", "men": "man", "women": "woman",
             "mice": "mouse", "feet": "foot", "teeth": "tooth", "geese": "goose", "people": "person",
             "gases": "gas", "lenses": "lens", "biases": "bias", "buses": "bus", "viruses": "virus",
             "statuses": "status", "aliases": "alias", "atlases": "atlas", "canvases": "canvas"}
_NO_STRIP = ("ss", "us", "is", "ics", "sis", "xis")


def plural_key(word: str) -> str:
    """A word's singular form for grouping (plural forms of one word share a key; over-merging is harmless:
    it only puts more terms on the same side)."""
    w = word
    if w in IRREGULAR:
        return IRREGULAR[w]
    if len(w) <= 3 or w.endswith(_NO_STRIP):
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "sses", "xes", "zes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith("'s"):
        return w[:-1]
    return w


def phrase_key(phrase: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(plural_key(w) for w in phrase)


def _contains(big: tuple[str, ...], small: tuple[str, ...]) -> bool:
    n, m = len(big), len(small)
    return m <= n and any(big[i:i + m] == small for i in range(n - m + 1))


def side_of(group_key: str, share: float = layout.TARGETED_SHARE, salt: str = layout.SPLIT_SALT) -> str:
    digest = hashlib.sha256(f"{salt}\n{group_key}".encode()).digest()
    return "targeted" if int.from_bytes(digest[:8], "big") < share * 2 ** 64 else "heldout"


def list_path(domain: str) -> Path:
    if domain not in layout.DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; expected one of {layout.DOMAINS}")
    return LISTS_DIR / f"{domain}.txt"


def read_lists(lists_dir: Path | None = None) -> dict[str, list[str]]:
    base = LISTS_DIR if lists_dir is None else Path(lists_dir)
    return {d: committed_terms.read_file(base / f"{d}.txt") for d in layout.DOMAINS}


def default_normalize() -> Callable[[str], str]:
    import evaluate  # finetune/parakeet-ternary
    return evaluate.normalize


def medical_matcher(normalize: Callable[[str], str]) -> committed_terms.Matcher:
    return committed_terms.Matcher(committed_terms.load("medical"), normalize)


class _UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra


def group_phrases(phrases: Iterable[tuple[str, ...]]) -> dict[tuple[str, ...], str]:
    """phrase -> group key (see the module docstring, step 4)."""
    phrases = sorted(set(phrases))
    keys = {p: phrase_key(p) for p in phrases}
    uf = _UnionFind(phrases)
    by_key = defaultdict(list)
    for p, k in keys.items():
        by_key[k].append(p)
    for members in by_key.values():
        for other in members[1:]:
            uf.union(members[0], other)
    present_raw = set(phrases)
    for p in phrases:                      # listed plural pairs plural_key does not reduce (acronyms)
        last = p[-1]
        for cut in (1, 2):
            if len(last) > cut + 1 and last.endswith("s" if cut == 1 else "es"):
                q = p[:-1] + (last[:-cut],)
                if q in present_raw:
                    uf.union(p, q)
    # Nested phrases: index every contiguous sub-phrase (in key form) of every phrase.
    distinct = sorted(by_key)
    present = set(distinct)
    for k in distinct:
        n = len(k)
        for i in range(n):
            for j in range(i + 1, n + 1):
                sub = k[i:j]
                if sub != k and sub in present:
                    uf.union(by_key[k][0], by_key[sub][0])
    by_variant = defaultdict(list)          # spelling variants and plural forms sharing a canonical key
    for p in phrases:
        by_variant[canonical_key(p)].append(p)
    for members in by_variant.values():
        for other in members[1:]:
            uf.union(members[0], other)
    groups = defaultdict(list)
    for p in phrases:
        groups[uf.find(p)].append(p)
    out = {}
    for members in groups.values():
        key = min(canonical_key(p) for p in members)
        for p in members:
            out[p] = key
    return out


def singular_tail(s: str) -> str:
    """The singular form of a string's last word, found from the string alone (no word boundaries): an irregular
    plural ending (IRREGULAR, longest first) is replaced, otherwise plural_key's rules apply to the string."""
    for plural, single in _IRREGULAR_BY_LENGTH:
        if s.endswith(plural):
            return s[:-len(plural)] + single
    if s.endswith(_SINGULAR_S):           # "noblegas", "convexlens": already singular
        return s
    return plural_key(s)


_IRREGULAR_BY_LENGTH = sorted(IRREGULAR.items(), key=lambda kv: (-len(kv[0]), kv[0]))
_SINGULAR_S = tuple(sorted({v for v in IRREGULAR.values() if v.endswith("s")}))


def canonical_key(phrase: tuple[str, ...]) -> str:
    """The words joined without spaces, last word singular (singular_tail): a function of the letters only, so a
    term's spelling variants always share it ("data warehouse" / "datawarehouse"), and so do most plural forms."""
    return singular_tail("".join(phrase))


def plural_surfaces(key: str) -> set[str]:
    """Joined surface forms of a canonical key: itself and its regular (-s, -es, -ies) and IRREGULAR plurals."""
    out = {key, key + "s", key + "es"}
    if key.endswith("y"):
        out.add(key[:-1] + "ies")
    for plural, single in IRREGULAR.items():
        if key.endswith(single):
            out.add(key[:-len(single)] + plural)
    return out


@dataclasses.dataclass(frozen=True)
class Entry:
    term: str                    # as written in the list
    phrase: tuple[str, ...]      # normalized words
    domain: str
    group: str
    side: str


def build_split(lists: dict[str, list[str]], normalize: Callable[[str], str],
                medical: committed_terms.Matcher, share: float = layout.TARGETED_SHARE,
                salt: str = layout.SPLIT_SALT) -> dict:
    """The split object (see the module docstring) from {domain: terms as written}."""
    unknown = sorted(set(lists) - set(layout.DOMAINS))
    if unknown:
        raise ValueError(f"unknown domains {unknown}")
    dropped, medical_excluded, duplicates = [], [], []
    owner: dict[tuple[str, ...], tuple[str, str]] = {}   # phrase -> (domain, first written form)
    for domain in layout.DOMAINS:
        for term in lists.get(domain, []):
            phrase = tuple(normalize(term).split())
            if not phrase or (len(phrase) == 1 and len(phrase[0]) == 1):
                dropped.append({"term": term, "domain": domain, "reason": "normalizes to nothing or one character"})
                continue
            if medical.find(list(phrase)):
                medical_excluded.append({"term": term, "domain": domain, "phrase": " ".join(phrase)})
                continue
            if phrase in owner:
                if owner[phrase][0] != domain:
                    duplicates.append({"term": term, "domain": domain, "kept_in": owner[phrase][0],
                                       "phrase": " ".join(phrase)})
                continue
            owner[phrase] = (domain, term)
    key_domain: dict[str, str] = {}
    for phrase, (domain, _) in owner.items():
        k = committed_terms.variant_key(phrase)
        if k not in key_domain or layout.DOMAINS.index(domain) < layout.DOMAINS.index(key_domain[k]):
            key_domain[k] = domain
    variant_domain = []
    for phrase, (domain, term) in list(owner.items()):
        kd = key_domain[committed_terms.variant_key(phrase)]
        if kd != domain:
            variant_domain.append({"term": term, "listed_in": domain, "domain": kd, "phrase": " ".join(phrase)})
            owner[phrase] = (kd, term)
    groups = group_phrases(owner)
    entries = []
    for phrase in sorted(owner, key=lambda p: (layout.DOMAINS.index(owner[p][0]), p)):
        domain, term = owner[phrase]
        g = groups[phrase]
        entries.append({"term": term, "phrase": " ".join(phrase), "domain": domain, "group": g,
                        "side": side_of(g, share, salt)})
    counts = {d: {s: sum(1 for e in entries if e["domain"] == d and e["side"] == s) for s in SIDES}
              for d in layout.DOMAINS}
    group_sides = {}
    for e in entries:
        group_sides[e["group"]] = e["side"]
    return {"version": VERSION, "salt": salt, "targeted_share": share,
            "rule": "group targeted iff int(sha256(salt + '\\n' + group key)[:8], big-endian) < share * 2**64",
            "normalizer": "finetune/parakeet-ternary/evaluate.py normalize",
            "entries": entries, "counts": counts,
            "groups": {"total": len(group_sides),
                       "targeted": sum(1 for s in group_sides.values() if s == "targeted")},
            "dropped": dropped, "medical_excluded": medical_excluded, "duplicates": duplicates,
            "variant_domain": variant_domain,
            "variants": sorted([" ".join(p) for p in v] for v in by_key_groups(owner))}


def by_key_groups(phrases) -> list[list[tuple[str, ...]]]:
    """Spelling-variant groups (terms.variant_key) with more than one listed phrase."""
    out = defaultdict(list)
    for p in sorted(phrases):
        out[committed_terms.variant_key(p)].append(p)
    return [v for v in out.values() if len(v) > 1]


def lists_record(lists_dir: Path | None = None) -> dict:
    base = LISTS_DIR if lists_dir is None else Path(lists_dir)
    out = {}
    for d in layout.DOMAINS:
        p = base / f"{d}.txt"
        out[d] = {"file": f"terms/synthetic/{d}.txt", "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    return out


def changes(old: dict, new: dict) -> dict:
    """Terms whose side or domain differs between two splits, and terms only in one of them (by phrase)."""
    o = {e["phrase"]: e for e in old["entries"]}
    n = {e["phrase"]: e for e in new["entries"]}
    moved = [{"phrase": p, "term": n[p]["term"], "old": [o[p]["domain"], o[p]["side"]],
              "new": [n[p]["domain"], n[p]["side"]]}
             for p in sorted(set(o) & set(n)) if (o[p]["domain"], o[p]["side"]) != (n[p]["domain"], n[p]["side"])]
    return {"changed": moved, "added": [{"phrase": p, "domain": n[p]["domain"], "side": n[p]["side"]}
                                        for p in sorted(set(n) - set(o))],
            "removed": sorted(set(o) - set(n)), "old_counts": old["counts"], "new_counts": new["counts"]}


def freeze(path: Path | None = None, lists_dir: Path | None = None, normalize=None,
           replace_reason: str | None = None) -> dict:
    """Build the split from the committed lists and write it once (refuses to change a frozen split). With a
    replace_reason (a plan decision), a differing frozen split is archived as split.v<version>.json next to it
    and replaced, and the changes are written to split-changes.json."""
    path = layout.split_path() if path is None else Path(path)
    normalize = normalize or default_normalize()
    split = build_split(read_lists(lists_dir), normalize, medical_matcher(normalize))
    split["lists"] = lists_record(lists_dir)
    if path.exists():
        old = json.loads(path.read_text())
        if old == split:
            return old
        if not replace_reason:
            raise RuntimeError(f"{path} is frozen and differs from a fresh build (lists or rules changed); "
                               "a new split needs a plan decision")
        archive = path.with_name(f"split.v{old.get('version', 1)}.json")
        if archive.exists() and json.loads(archive.read_text()) != old:
            raise RuntimeError(f"{archive} exists and differs from the split being replaced")
        archive.write_text(path.read_text())
        diff = changes(old, split) | {"reason": replace_reason, "archived": archive.name}
        tmp = path.with_name(f"split-changes.json.tmp{os.getpid()}")
        tmp.write_text(json.dumps(diff, indent=1, ensure_ascii=False) + "\n")
        os.replace(tmp, path.with_name("split-changes.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(split, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return split


class Split:
    """A loaded split: entries, lookups and the matchers (module docstring, "Matching")."""

    def __init__(self, data: dict, normalize: Callable[[str], str] | None = None):
        self.data = data
        self.normalize = normalize or default_normalize()
        self.entries = [Entry(e["term"], tuple(e["phrase"].split()), e["domain"], e["group"], e["side"])
                        for e in data["entries"]]
        self.by_phrase = {e.phrase: e for e in self.entries}
        self._matchers: dict = {}

    @classmethod
    def load(cls, path: Path | None = None, check_lists: bool = True, lists_dir: Path | None = None,
             normalize=None) -> "Split":
        path = layout.split_path() if path is None else Path(path)
        if not path.exists():
            raise RuntimeError(f"{path} is missing (run: ./python -m synthetic split)")
        data = json.loads(path.read_text())
        if check_lists and data.get("lists") != lists_record(lists_dir):
            raise RuntimeError("terms/synthetic/*.txt changed after the split was frozen")
        return cls(data, normalize)

    def targeted(self, domain: str) -> list[Entry]:
        return [e for e in self.entries if e.domain == domain and e.side == "targeted"]

    def heldout(self) -> list[Entry]:
        return [e for e in self.entries if e.side == "heldout"]

    def _identity_matcher(self, phrases) -> committed_terms.Matcher:
        # Phrases are already normalized: match them as written (identity normalizer).
        return committed_terms.Matcher([" ".join(p) for p in phrases], lambda s: s)

    def matcher(self) -> committed_terms.Matcher:
        if "all" not in self._matchers:
            self._matchers["all"] = self._identity_matcher(self.by_phrase)
        return self._matchers["all"]

    def heldout_matcher(self) -> committed_terms.Matcher:
        if "heldout" not in self._matchers:
            self._matchers["heldout"] = self._identity_matcher(e.phrase for e in self.heldout())
        return self._matchers["heldout"]

    def classify(self, phrase: tuple[str, ...]) -> tuple[str, str]:
        e = self.by_phrase[tuple(phrase)]
        return e.domain, e.side

    def occurrences(self, text: str) -> list[dict]:
        """Every term occurrence in a text, as the scorer finds them: [{phrase, domain, side, start, end}]."""
        words = self.normalize(text).split()
        out = []
        for occ in self.matcher().find(words):
            domain, side = self.classify(occ.phrase)
            out.append({"phrase": " ".join(occ.phrase), "domain": domain, "side": side, "start": occ.start,
                        "end": occ.end})
        return out

    def heldout_guard(self, words) -> bool:
        """Whether any run of consecutive words is a surface form of a held-out term (module docstring)."""
        if "guard" not in self._matchers:
            surfaces = set()
            for e in self.heldout():
                surfaces |= plural_surfaces(canonical_key(e.phrase))
            prefixes = {k[:i] for k in surfaces for i in range(1, len(k) + 1)}
            self._matchers["guard"] = (surfaces, prefixes)
        surfaces, prefixes = self._matchers["guard"]
        for i in range(len(words)):
            acc = ""
            for w in words[i:]:
                acc += w
                if acc not in prefixes:
                    break
                if acc in surfaces:
                    return True
        return False

    def has_heldout(self, text: str) -> bool:
        return self.heldout_guard(self.normalize(text).split())


def recall(records: Iterable[dict], split: Split) -> dict:
    """Critical-term recall per (domain, side) with score_domain's aligned-occurrence definition, over records
    with "ref_norm"/"hyp_norm" (evaluate.py's per-utterance records; unscored records are skipped). Returns
    {domain: {side: {occurrences, reproduced, recall}}} plus "pooled" per side over all domains."""
    import score_domain
    acc = {d: {s: [0, 0] for s in SIDES} for d in layout.DOMAINS}
    m = split.matcher()
    for r in records:
        if not r.get("scored", True):
            continue
        for start, end, ok in score_domain.term_occurrences(r["ref_norm"], r["hyp_norm"], m):
            domain, side = split.classify(tuple(r["ref_norm"].split()[start:end]))
            acc[domain][side][0] += 1
            acc[domain][side][1] += int(ok)
    out = {d: {s: {"occurrences": n, "reproduced": k, "recall": k / n if n else None}
               for s, (n, k) in sides.items()} for d, sides in acc.items()}
    out["pooled"] = {}
    for s in SIDES:
        n = sum(acc[d][s][0] for d in layout.DOMAINS)
        k = sum(acc[d][s][1] for d in layout.DOMAINS)
        out["pooled"][s] = {"occurrences": n, "reproduced": k, "recall": k / n if n else None}
    return out
