"""Hygiene of the generated sentences (synthetic plan "Text": form, hygiene; "Terms": held-out and medical rules).

Every generated output goes through these rules in order; the first failing rule is its reason (all rules are
evaluated and recorded as flags):

    no_sentence      no usable line (empty output, or the generator's reasoning markup)
    form_symbols     anything but letters, spaces and . , ; : ? ! ' " - (no digits, symbols, formulas or code)
    form_length      fewer than MIN_WORDS or more than MAX_WORDS normalized words
    tokenizer        the label does not encode with B0's SentencePiece model without the unknown id
    no_target        none of the prompt's requested targeted terms (or a variant in its group) occurs
    medical          any occurrence of the medical / TCCC term lists (terms.matcher("medical"))
    heldout_term     any surface form of a held-out term of any domain (vocab.Split.heldout_guard: listed, joined,
                     split, hyphenated or plural forms)
    denylist         Part B's 8-gram containment rule against every held-out text (pool/denylist.py build(),
                     each sentence taken as one training recording; boilerplate 8-grams counted over the
                     sentences)
    duplicate        same normalized text as an earlier kept sentence
    near_duplicate   word-bigram Jaccard >= NEAR_JACCARD with an earlier kept sentence (MinHash candidates,
                     exact Jaccard check), in sentence-id order

The spoken text is the sentence after labeling.filters.map_text (NFKC, curly quotes and dashes mapped); the
pronunciation layer renders it for the TTS engines (synthetic/pronounce.py). The label is the spoken text with
label forms applied (synthetic/forms.py: diacritics folded to ASCII, listed terms in their conventional case and
hyphenation), which never changes the evaluation normalizer's words. Every rule below runs on the label. Numbers are
written as words, which the evaluation normalizer maps to digits for scoring.

Held-out texts: every per-utterance manifest the global denylist reads (pool/__main__.heldout_manifests: the
parakeet-ternary dev and test references, text only, and every manifest under $RELABEL_ROOT/devsets/, which
holds dev-medical, dev-technical, eval-ext and eval-stem). The sets used are recorded with their SHA-256; a run
without eval-ext and eval-stem is marked provisional and cannot be used for training (mixing.ClipPool refuses it).
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from synthetic import layout

REASONS = ("no_sentence", "form_symbols", "form_length", "tokenizer", "no_target", "medical", "heldout_term",
           "denylist", "duplicate", "near_duplicate")
REQUIRED_HELDOUT_PARTS = ("medical-b", "lectures-a", "stem-chem", "stem-math", "stem-cs")
NEAR_JACCARD = 0.7
MINHASH_BANDS, MINHASH_ROWS = 10, 3
_ALLOWED = re.compile(r"[A-Za-z .,;:?!'\"-]+")
_PRIME = (1 << 61) - 1


def extract_sentence(raw: str | None) -> str | None:
    """The sentence of a generator output: first non-empty line, without wrapping quotes or markdown."""
    if not raw:
        return None
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    if "<think>" in text:
        return None
    for line in text.splitlines():
        line = line.strip().strip("*").strip()
        if line:
            if len(line) >= 2 and line[0] in "\"'“" and line[-1] in "\"'”":
                line = line[1:-1].strip()
            return line or None
    return None


def bigrams(words: Sequence[str]) -> set[str]:
    if len(words) < 2:
        return {" ".join(words)}
    return {f"{a} {b}" for a, b in zip(words, words[1:])}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


class NearDuplicates:
    """MinHash (bands x rows) candidate index of kept sentences' word-bigram sets, exact Jaccard check."""

    def __init__(self, threshold: float = NEAR_JACCARD, bands: int = MINHASH_BANDS, rows: int = MINHASH_ROWS):
        self.threshold, self.bands, self.rows = threshold, bands, rows
        rng = np.random.default_rng(20261022)
        n = bands * rows
        self.a = [int(x) for x in rng.integers(1, _PRIME - 1, n)]
        self.b = [int(x) for x in rng.integers(0, _PRIME - 1, n)]
        self.buckets: dict[tuple, list[int]] = defaultdict(list)
        self.sets: list[set[str]] = []

    @staticmethod
    def _h(s: str) -> int:
        return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")

    def _signature(self, shingles: set[str]) -> list[int]:
        hs = [self._h(s) for s in shingles]
        return [min((a * h + b) % _PRIME for h in hs) for a, b in zip(self.a, self.b)]

    def check_add(self, words: Sequence[str]) -> bool:
        """True if near-duplicate of an earlier added sentence; otherwise adds it and returns False."""
        sh = bigrams(words)
        sig = self._signature(sh)
        keys = [(i, tuple(sig[i * self.rows:(i + 1) * self.rows])) for i in range(self.bands)]
        seen = set()
        for k in keys:
            for j in self.buckets.get(k, ()):
                if j not in seen:
                    seen.add(j)
                    if jaccard(sh, self.sets[j]) >= self.threshold:
                        return True
        idx = len(self.sets)
        self.sets.append(sh)
        for k in keys:
            self.buckets[k].append(idx)
        return False


def heldout_sets(extra: Iterable[tuple[str, object]] = ()) -> tuple[list, dict]:
    """(pool.heldout sets, record) for every held-out manifest of the global denylist, plus `extra`."""
    from pool import heldout as ho
    from pool.__main__ import heldout_manifests
    paths, skipped = heldout_manifests([])
    paths = list(paths) + list(extra)
    sets = ho.load_sets(paths)
    names = [s.name for s in sets]
    have_parts = {n.rsplit("/", 1)[-1] for n in names}
    missing = [p for p in REQUIRED_HELDOUT_PARTS if p not in have_parts]
    record = {"sets": [{"name": s.name, "sha256": s.sha256, "utterances": len(s.ids)} for s in sets],
              "skipped_jsonl": skipped, "missing_required_parts": missing, "provisional": bool(missing)}
    return sets, record


def containment(sentences: Sequence[tuple[str, list[str]]], sets: list) -> tuple[set[str], dict]:
    """Ids of sentences denylisted by Part B's containment rule ([(id, normalized words)], held-out sets)."""
    from pool import common, denylist
    from pool.heldout import Index
    from pool.scan import ScanResult
    index = Index(sets)
    hasher = common.WordHasher()
    chunks = []
    for gid, (_, words) in enumerate(sentences):
        if len(words) < common.NGRAM or not len(index.hashes):
            continue
        grams = np.unique(common.ngram_hashes(hasher(words)))
        pos = np.searchsorted(index.hashes, grams)
        ok = pos < len(index.hashes)
        ok[ok] = index.hashes[pos[ok]] == grams[ok]
        if ok.any():
            chunks.append((np.uint64(gid) << np.uint64(32)) | pos[ok].astype(np.uint64))
    pairs = np.unique(np.concatenate(chunks)) if chunks else np.empty(0, np.uint64)
    keys = [("synthetic", sid) for sid, _ in sentences]
    scan = ScanResult(keys=keys, stats=[{} for _ in keys], locators=[{} for _ in keys], pairs=pairs, sources={})
    obj, details = denylist.build(scan, index, reserved=[])
    hit = set(obj["recordings"].get("synthetic", []))
    kept_details = [d for d in details if d["source"] == "synthetic"]
    return hit, {"denylisted": len(hit), "boilerplate_8grams": obj["counts"]["boilerplate_8grams"],
                 "details": kept_details[:200]}


def normalized_rows(records: Sequence[dict], forms=None) -> list[dict]:
    """[{id, domain, register, requested, sentence, spoken, label, form_subs, folded}] in id order (module docstring)."""
    from labeling.filters import map_text
    from synthetic import forms as label_forms
    forms = label_forms.Forms.load() if forms is None else forms
    out = []
    for r in sorted(records, key=lambda r: r["id"]):
        sent = extract_sentence(r.get("raw"))
        spoken = map_text(sent) if sent else ""
        label, subs = forms.apply(spoken)
        out.append({"id": r["id"], "domain": r["domain"], "register": r.get("register"),
                    "requested": list(r["requested"]), "sentence": sent, "spoken": spoken, "label": label,
                    "form_subs": subs, "folded": label_forms.fold(spoken) != spoken})
    return out


def run(records: Sequence[dict], split, medical, normalize: Callable[[str], str], coverage: Callable[[str], bool],
        sets: list, heldout_record: dict, forms=None) -> tuple[list[dict], dict, list[dict]]:
    """(kept sentences, report, every row with its flags and reason) from generator records {id, domain, register,
    requested, raw} in id order. forms: synthetic.forms.Forms (default: the committed files)."""
    from synthetic import forms as label_forms
    forms = label_forms.Forms.load() if forms is None else forms
    records = normalized_rows(records, forms)
    group_of = {e.phrase: e.group for e in split.entries}
    req_groups_cache: dict[tuple, set] = {}
    rows = []
    for r in records:
        flags = {k: False for k in REASONS}
        sent, label = r["sentence"], r["label"]
        norm = normalize(label).split() if label else []
        if not sent:
            flags["no_sentence"] = True
        else:
            flags["form_symbols"] = not _ALLOWED.fullmatch(label)
            flags["form_length"] = not layout.MIN_WORDS <= len(norm) <= layout.MAX_WORDS
            flags["tokenizer"] = not coverage(label)
            key = tuple(r["requested"])
            if key not in req_groups_cache:
                req_groups_cache[key] = {group_of[tuple(normalize(t).split())] for t in r["requested"]}
            occ = split.matcher().find(norm)
            flags["no_target"] = not any(group_of[o.phrase] in req_groups_cache[key] for o in occ)
            flags["medical"] = bool(medical.find(norm))
            flags["heldout_term"] = split.heldout_guard(norm)
        rows.append({"id": r["id"], "domain": r["domain"], "register": r.get("register"),
                     "requested": list(r["requested"]), "label": label, "spoken": r["spoken"],
                     "norm": " ".join(norm), "form_subs": [f"{x['written']} -> {x['form']}" for x in r["form_subs"]],
                     "folded": r["folded"], "flags": flags})
    # Containment against held-out texts, over the sentences that passed the per-sentence rules.
    cand = [(x["id"], x["norm"].split()) for x in rows if not any(x["flags"].values())]
    deny, deny_report = containment(cand, sets)
    exact, near = set(), NearDuplicates()
    kept = []
    for x in rows:
        f = x["flags"]
        if x["id"] in deny:
            f["denylist"] = True
        if not any(f.values()):
            if x["norm"] in exact:
                f["duplicate"] = True
            elif near.check_add(x["norm"].split()):
                f["near_duplicate"] = True
            else:
                exact.add(x["norm"])
        x["reason"] = next((k for k in REASONS if f[k]), None)
        if x["reason"] is None:
            kept.append({k: x[k] for k in ("id", "domain", "register", "requested", "label", "spoken", "norm")})
    report = {"generated": len(rows), "kept": len(kept),
              "by_domain": {d: {"generated": sum(1 for x in rows if x["domain"] == d),
                                "kept": sum(1 for x in kept if x["domain"] == d),
                                "reasons": dict(Counter(x["reason"] for x in rows if x["domain"] == d and x["reason"]))}
                            for d in layout.DOMAINS},
              "reasons": dict(Counter(x["reason"] for x in rows if x["reason"])),
              "flags": {k: sum(1 for x in rows if x["flags"][k]) for k in REASONS},
              "denylist": deny_report, "heldout": heldout_record,
              "forms": forms.record() | {
                  "sentences_changed": sum(1 for x in rows if x["label"] != x["spoken"]),
                  "kept_sentences_changed": sum(1 for x in kept if x["label"] != x["spoken"]),
                  "folded_sentences": sum(1 for x in rows if x["folded"]),
                  "substitutions": dict(sorted(Counter(t for x in rows for t in x["form_subs"]).items())),
                  "kept_substitutions": dict(sorted(Counter(t for x in rows if x["reason"] is None
                                                            for t in x["form_subs"]).items()))},
              "rules": {"min_words": layout.MIN_WORDS, "max_words": layout.MAX_WORDS, "near_jaccard": NEAR_JACCARD,
                        "allowed_characters": _ALLOWED.pattern}}
    return kept, report, rows
