"""Committed term lists (labeling plan): medical critical-term recall (Part A) and domain selection (Part B).

Layout: terms/<kind>/<category>.txt, plain text, one term per line. Blank lines and lines starting with
"#" are comments. Terms were written from general knowledge; everyday words with common non-domain senses
are left out on purpose (each file's header says so).

    medical    drugs, anatomy, procedures, conditions, dosages_units, tccc
    technical  computing, engineering, science, finance

Stable API (imported by Part B):

    KINDS                               ("medical", "technical")
    categories(kind) -> list[str]       the category files of a kind, in a fixed order
    load(kind, only=None)               the terms as written, de-duplicated (case-insensitive), in category
                                        then file order
    Matcher(terms, normalize)           .find(tokens) -> [Occurrence(start, end, phrase)]: greedy longest
                                        match, left to right, non-overlapping, over normalized word tokens
                                        (`end` exclusive); .count(tokens) -> int; .variants(phrase) and
                                        .groups(): the listed spelling variants of a term (below)
    matcher(kind, normalize=None)       a Matcher over load(kind); `normalize` defaults to the evaluation
                                        normalizer (finetune/parakeet-ternary/evaluate.py normalize)

A term is matched in its normalized form (the same normalizer as the references), so spelling variants the
normalizer unifies (British/American spellings, number words, a hyphen versus a space) match the same tokens.

Spelling variants (2026-10-24): listed terms whose normalized words, joined without spaces, are equal
(variant_key: "dataset" and "data set", "ecommerce" and "e commerce") are variants of ONE term. Only listed forms
are variants: nothing is joined or split automatically, so an ordinary word pair ("a part", "test is") never
matches a listed single word, and an unlisted joined form never matches a listed phrase. Each list therefore
lists every attested form it wants scored (reviewed for collisions). score_domain.term_occurrences counts an
occurrence once, whichever listed form it has, and accepts any listed form of the same term in the hypothesis. Terms that normalize to
nothing or to a single character are dropped by the Matcher (reported in Matcher.dropped).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

HERE = Path(__file__).resolve().parent
KINDS = ("medical", "technical")
_CATEGORIES = {
    "medical": ("drugs", "anatomy", "procedures", "conditions", "dosages_units", "tccc"),
    "technical": ("computing", "engineering", "science", "finance"),
}


def categories(kind: str) -> list[str]:
    if kind not in _CATEGORIES:
        raise ValueError(f"unknown term kind {kind!r}; expected one of {KINDS}")
    return list(_CATEGORIES[kind])


def read_file(path: Path) -> list[str]:
    """Terms of one file: stripped lines that are neither blank nor comments."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        term = line.strip()
        if term and not term.startswith("#"):
            out.append(term)
    return out


def load(kind: str, only: Iterable[str] | None = None, base: Path | None = None) -> list[str]:
    """Terms of `kind` (optionally `only` some categories), case-insensitively de-duplicated, in order. `base` is
    another directory with the same layout (e.g. a snapshot of earlier lists); default the committed lists."""
    names = categories(kind) if only is None else list(only)
    unknown = [c for c in names if c not in _CATEGORIES[kind]]
    if unknown:
        raise ValueError(f"unknown {kind} categories {unknown}")
    seen, out = set(), []
    for name in names:
        for term in read_file((HERE if base is None else Path(base)) / kind / f"{name}.txt"):
            key = term.casefold()
            if key not in seen:
                seen.add(key)
                out.append(term)
    return out


def variant_key(phrase: tuple[str, ...]) -> str:
    """The key that makes listed phrases variants of one term: its normalized words joined without spaces."""
    return "".join(phrase)


@dataclass(frozen=True)
class Occurrence:
    start: int   # first token index
    end: int     # one past the last token index
    phrase: tuple[str, ...]


class Matcher:
    """Greedy longest-match finder of normalized term phrases in a normalized token sequence."""

    def __init__(self, terms: Iterable[str], normalize: Callable[[str], str]) -> None:
        self.phrases: dict[tuple[str, ...], str] = {}
        self.dropped: list[str] = []
        for term in terms:
            phrase = tuple(normalize(term).split())
            if not phrase or (len(phrase) == 1 and len(phrase[0]) == 1):  # nothing, or one character
                self.dropped.append(term)
                continue
            self.phrases.setdefault(phrase, term)
        by_first: dict[str, set[int]] = {}
        for phrase in self.phrases:
            by_first.setdefault(phrase[0], set()).add(len(phrase))
        # Candidate lengths per first token, longest first.
        self._lengths = {first: sorted(lengths, reverse=True) for first, lengths in by_first.items()}
        by_key: dict[str, list[tuple[str, ...]]] = {}
        for phrase in self.phrases:
            by_key.setdefault(variant_key(phrase), []).append(phrase)
        self._variants = {k: tuple(sorted(v)) for k, v in by_key.items()}

    def variants(self, phrase: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
        """Every listed phrase that is a variant of `phrase` (itself included), sorted."""
        return self._variants.get(variant_key(tuple(phrase)), ())

    def groups(self) -> dict[str, tuple[tuple[str, ...], ...]]:
        """The variant groups with more than one listed form: {variant key: phrases}."""
        return {k: v for k, v in sorted(self._variants.items()) if len(v) > 1}

    def find(self, tokens: list[str] | tuple[str, ...]) -> list[Occurrence]:
        out, i, n = [], 0, len(tokens)
        while i < n:
            for length in self._lengths.get(tokens[i], ()):
                if i + length <= n and tuple(tokens[i:i + length]) in self.phrases:
                    out.append(Occurrence(i, i + length, tuple(tokens[i:i + length])))
                    i += length
                    break
            else:
                i += 1
        return out

    def count(self, tokens: list[str] | tuple[str, ...]) -> int:
        return len(self.find(tokens))


def matcher(kind: str, normalize: Callable[[str], str] | None = None) -> Matcher:
    if normalize is None:
        import evaluate  # finetune/parakeet-ternary/evaluate.py (on sys.path via ./python)

        normalize = evaluate.normalize
    return Matcher(load(kind), normalize)
