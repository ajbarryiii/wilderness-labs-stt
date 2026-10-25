"""Term occurrences in normalized text (labeling plan Part B "Score").

The committed term lists and the matching rule are Part A's (finetune/relabel/terms): `terms.load(kind)` for
kind "medical" or "technical", and `terms.Matcher`: each term normalized with the evaluation normalizer,
matched as a whole-word sequence, left to right, longest first, without overlap (so "blood pressure" is one
occurrence, not also "blood"); terms that normalize to nothing or to one character are dropped. Density is
occurrences per 1,000 normalized words of the recording.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence

from pool import common

KINDS = ("medical", "technical")


class TermMatcher:
    def __init__(self, term_list: Iterable[str], normalize: Callable[[str], str] = common.normalize):
        import terms  # Part A
        term_list = list(term_list)
        for term in term_list:
            if not isinstance(term, str):
                raise TypeError(f"term lists must contain strings, got {type(term).__name__}")
        self._m = terms.Matcher(term_list, normalize)
        if not self._m.phrases:
            raise ValueError("empty term list")
        self.terms = sorted(self._m.phrases)
        self.dropped = list(self._m.dropped)
        self.sha256 = hashlib.sha256("\n".join(" ".join(t) for t in self.terms).encode()).hexdigest()

    def count(self, words: Sequence[str]) -> int:
        return self._m.count(words)

    def __getstate__(self):   # picklable for worker processes (the normalizer is not needed after init)
        return {"terms": self.terms, "dropped": self.dropped, "sha256": self.sha256}

    def __setstate__(self, state):
        import terms
        self.__dict__.update(state)
        self._m = terms.Matcher([" ".join(t) for t in self.terms], lambda s: s)


def density(count: int, words: int) -> float:
    return 1000.0 * count / words if words else 0.0


def load_terms(kind: str) -> list[str]:
    """Part A's committed term list for `kind`."""
    if kind not in KINDS:
        raise ValueError(f"unknown term kind {kind!r}")
    import terms
    return list(terms.load(kind))


def matchers(loader: Callable[[str], Iterable[str]] = load_terms) -> dict[str, TermMatcher]:
    return {kind: TermMatcher(loader(kind)) for kind in KINDS}
