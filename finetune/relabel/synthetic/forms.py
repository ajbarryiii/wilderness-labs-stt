"""Label forms (synthetic plan "Text", hygiene): the conventional written form of listed terms in the label.

Files: terms/synthetic/forms/<domain>.tsv (committed, curated by hand; '#' comments, blank lines), one
"written<TAB>conventional" pair per line. Both sides are ASCII letters, apostrophes, hyphens and single spaces, and
they may differ only in letter case and in a hyphen versus a space (checked at load). The evaluation normalizer
lowercases and treats a hyphen as a space, so a form never changes the scorer's normalized words: term matching,
held-out and medical rules, the denylist and duplicate checks see the same words with or without forms. The domain
files are merged (a sentence can use another domain's terms); one written form with two conventional forms is an
error.

apply(text) -> (text, substitutions):
1. fold: Latin letters with diacritics become their ASCII letter (NFKD, combining marks dropped), e.g.
   "Schrodinger", "L'Hopital"; nothing else is touched.
2. forms: every occurrence of a written form, compared case-insensitively, bounded on both sides by a character that
   is not a letter or digit (or the text's start or end), is replaced by its conventional form. Matching is left to
   right, and at each position the longest written form wins ("non gaap" before "gaap"); replaced text is not
   matched again. Everything else is copied unchanged.
Every conventional form maps to itself, so apply is idempotent (checked at load).

The TTS input is not derived from the normalized label: hygiene keeps the text before apply as "spoken", and the
pronunciation layer renders that (case and hyphens change how the phonemizer joins words, so the engines' input
stays exactly as it was).
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

import terms as committed_terms
from synthetic import layout

FORMS_DIR = committed_terms.HERE / "synthetic" / "forms"
_SIDE = re.compile(r"[A-Za-z']+(?:[ -][A-Za-z']+)*")


def _key(s: str) -> str:
    return s.lower().replace("-", " ")


def fold(text: str) -> str:
    out = []
    for ch in text:
        if ord(ch) < 128:
            out.append(ch)
            continue
        base = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
        out.append(base if len(base) == 1 and base.isascii() and base.isalpha() else ch)
    return "".join(out)


class Forms:
    def __init__(self, table: dict[str, str], files: dict[str, str]):
        self.table, self.files = table, files          # lower-case written form -> conventional form
        if table:
            alts = sorted(table, key=lambda w: (-len(w), w))
            self._re = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(re.escape(w) for w in alts) + r")(?![A-Za-z0-9])",
                                  re.IGNORECASE)
        else:
            self._re = None
        for written, form in table.items():
            if self.apply(form)[0] != form:
                raise ValueError(f"{form!r} (the form of {written!r}) is changed again by another entry")

    @classmethod
    def parse(cls, texts: dict[str, str]) -> "Forms":
        """From {domain: file text}."""
        table: dict[str, str] = {}
        where: dict[str, str] = {}
        for domain in sorted(texts):
            for n, line in enumerate(texts[domain].splitlines(), 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                written, _, form = (p.strip() for p in line.partition("\t"))
                if not form:
                    raise ValueError(f"{domain}:{n}: no conventional form for {written!r}")
                for side in (written, form):
                    if not side.isascii() or not _SIDE.fullmatch(side):
                        raise ValueError(f"{domain}:{n}: {side!r} must be ASCII letters, apostrophes, hyphens and "
                                         "single spaces")
                if _key(written) != _key(form):
                    raise ValueError(f"{domain}:{n}: {written!r} -> {form!r} changes more than case and hyphens")
                w = written.lower()
                if w in table and table[w] != form:
                    raise ValueError(f"{domain}:{n}: {w!r} is {table[w]!r} in {where[w]} and {form!r} here")
                table[w], where[w] = form, domain
        files = {d: hashlib.sha256(t.encode()).hexdigest() for d, t in sorted(texts.items())}
        return cls(table, files)

    @classmethod
    def load(cls, forms_dir: Path | None = None) -> "Forms":
        base = FORMS_DIR if forms_dir is None else Path(forms_dir)
        texts = {}
        for d in layout.DOMAINS:
            p = base / f"{d}.tsv"
            if not p.exists():
                raise RuntimeError(f"label forms file {p.name} is missing")
            texts[d] = p.read_text(encoding="utf-8")
        return cls.parse(texts)

    def record(self) -> dict:
        return {"files": {d: {"file": f"terms/synthetic/forms/{d}.tsv", "sha256": s} for d, s in self.files.items()}}

    def pairs(self) -> list[tuple[str, str]]:
        return sorted(self.table.items())

    def apply(self, text: str) -> tuple[str, list[dict]]:
        text = fold(text)
        if self._re is None:
            return text, []
        subs = []

        def repl(m: re.Match) -> str:
            w = m.group(1).lower()
            form = self.table[w]
            if m.group(1) != form:
                subs.append({"written": w, "form": form, "start": m.start(1), "found": m.group(1)})
            return form
        return self._re.sub(repl, text), subs
