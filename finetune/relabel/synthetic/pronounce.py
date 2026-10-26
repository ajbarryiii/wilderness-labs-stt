"""Pronunciation layer (synthetic plan "Speech", Pronunciation): the text a TTS engine is given (tts_text), kept
separate from the label, which never changes.

Lexicon: terms/synthetic/pronunciation/<domain>.tsv (committed, curated by hand; '#' comments, blank lines). A line
is a token alone (spelled out with letter names) or "token<TAB>respelling" (said as that word or words). Tokens are
single runs of letters and digits, compared case-insensitively; spelled tokens are letters only. The domain files
are merged; one token with two different renderings is an error.

render(label) -> (tts_text, substitutions). The label is split into tokens (maximal runs of [A-Za-z0-9]); each token
is replaced, in this order of rules, if
  "lexicon"         its lower-case form is a lexicon token: the respelling, or the spelled letters;
  "lexicon_plural"  it is a spelled lexicon token plus "s" (any case): the spelled letters, last one plural;
  "caps"            it is all capitals, two or more letters, and not a Roman numeral of I, V and X: spelled;
  "caps_plural"     it is such a token followed by a lower-case "s": spelled, last letter plural.
A rendering equal to the token is not a substitution. Every other character (other tokens, punctuation, spacing) is
copied unchanged. Letter names are words (LETTERS)
because the engines' phonemizer reads a lone capital "A" in a spaced sequence as the article.

Acronym terms (report split): is_acronym_term(term as listed, normalized phrase, lexicon) is true iff a word of the
phrase is a lexicon token whose rendering differs from the word (or a spelled token's plural), or the term as listed
has an all-capital token of two or more letters.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

import terms as committed_terms
from synthetic import layout

LEXICON_DIR = committed_terms.HERE / "synthetic" / "pronunciation"
LETTERS = {"a": "eh", "b": "bee", "c": "see", "d": "dee", "e": "ee", "f": "ef", "g": "jee", "h": "aitch",
           "i": "eye", "j": "jay", "k": "kay", "l": "el", "m": "em", "n": "en", "o": "oh", "p": "pee", "q": "cue",
           "r": "ar", "s": "ess", "t": "tee", "u": "you", "v": "vee", "w": "double you", "x": "ex", "y": "why",
           "z": "zee"}
TOKEN = re.compile(r"[A-Za-z0-9]+")
CAPS = re.compile(r"[A-Z]{2,}")
CAPS_PLURAL = re.compile(r"([A-Z]{2,})s")
ROMAN = re.compile(r"X{0,3}(IX|IV|V?I{0,3})")
RESPELLING = re.compile(r"[a-z]+(?:['\- ][a-z]+)*")


def _plural(name: str) -> str:
    return name + ("es" if name.endswith(("s", "x", "ch")) else "s")


def spell(letters: str, plural: bool = False) -> str:
    names = [LETTERS[c] for c in letters.lower()]
    if plural:
        names[-1] = _plural(names[-1])
    return " ".join(names)


def _is_roman(token: str) -> bool:
    return bool(ROMAN.fullmatch(token)) and len(token) >= 2


class Lexicon:
    """Token (lower case) -> respelling, or None for spelled out."""

    def __init__(self, entries: dict[str, str | None], files: dict[str, str]):
        self.entries, self.files = entries, files

    @classmethod
    def parse(cls, texts: dict[str, str]) -> "Lexicon":
        """From {domain: file text}."""
        entries: dict[str, str | None] = {}
        where: dict[str, str] = {}
        for domain in sorted(texts):
            for n, line in enumerate(texts[domain].splitlines(), 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                token, _, spoken = (p.strip() for p in line.partition("\t"))
                if not TOKEN.fullmatch(token):
                    raise ValueError(f"{domain}:{n}: {token!r} is not a single token")
                key = token.lower()
                if not spoken:
                    if not key.isalpha():
                        raise ValueError(f"{domain}:{n}: spelled token {token!r} must be letters only")
                    value = None
                elif RESPELLING.fullmatch(spoken):
                    value = spoken
                else:
                    raise ValueError(f"{domain}:{n}: respelling {spoken!r} must be lower-case words")
                if key in entries and entries[key] != value:
                    raise ValueError(f"{domain}:{n}: {key!r} is {entries[key]!r} in {where[key]} and {value!r} here")
                entries[key], where[key] = value, domain
        files = {d: hashlib.sha256(t.encode()).hexdigest() for d, t in sorted(texts.items())}
        return cls(entries, files)

    @classmethod
    def load(cls, lexicon_dir: Path | None = None) -> "Lexicon":
        base = LEXICON_DIR if lexicon_dir is None else Path(lexicon_dir)
        texts = {}
        for d in layout.DOMAINS:
            p = base / f"{d}.tsv"
            if not p.exists():
                raise RuntimeError(f"pronunciation lexicon {p.name} is missing")
            texts[d] = p.read_text(encoding="utf-8")
        return cls.parse(texts)

    def record(self) -> dict:
        return {"files": {d: {"file": f"terms/synthetic/pronunciation/{d}.tsv", "sha256": s}
                          for d, s in self.files.items()},
                "letters": dict(LETTERS)}

    def _token(self, tok: str) -> tuple[str, str] | None:
        low = tok.lower()
        if low in self.entries:
            v = self.entries[low]
            return (spell(low) if v is None else v), "lexicon"
        if low.endswith("s") and len(low) > 2 and low[:-1] in self.entries and self.entries[low[:-1]] is None:
            return spell(low[:-1], plural=True), "lexicon_plural"
        if CAPS.fullmatch(tok) and not _is_roman(tok):
            return spell(tok), "caps"
        m = CAPS_PLURAL.fullmatch(tok)
        if m and not _is_roman(m.group(1)):
            return spell(m.group(1), plural=True), "caps_plural"
        return None

    def render(self, text: str) -> tuple[str, list[dict]]:
        out, subs, last = [], [], 0
        for m in TOKEN.finditer(text):
            hit = self._token(m.group())
            if hit is None or hit[0] == m.group():      # e.g. "acid" said as itself: nothing to substitute
                continue
            spoken, rule = hit
            out.append(text[last:m.start()] + spoken)
            last = m.end()
            subs.append({"token": m.group(), "start": m.start(), "end": m.end(), "spoken": spoken, "rule": rule})
        out.append(text[last:])
        return "".join(out), subs

    def covers(self, word: str) -> bool:
        """Whether a lower-case word is a lexicon token or a spelled token's plural."""
        w = word.lower()
        return w in self.entries or (w.endswith("s") and self.entries.get(w[:-1], "") is None and w[:-1] in self.entries)

    def changes(self, word: str) -> bool:
        """Whether the lexicon renders a lower-case word differently from itself."""
        w = word.lower()
        if w in self.entries:
            return self.entries[w] != w
        return self.covers(w)


def is_acronym_term(term: str, phrase: tuple[str, ...], lexicon: Lexicon) -> bool:
    if any(lexicon.changes(w) for w in phrase):
        return True
    return any(CAPS.fullmatch(t) or CAPS_PLURAL.fullmatch(t) for t in TOKEN.findall(term))


def apply(plan: list[dict], lexicon: Lexicon, spoken: dict[str, str] | None = None) -> tuple[list[dict], dict]:
    """The plan with each row's tts_text (labels untouched), and the run's pronunciation summary. tts_text renders
    the sentence's spoken text (spoken[id], hygiene's text before label forms; synthetic/forms.py), or the label for
    sentences without one (runs made before label forms, where the two are the same)."""
    out, by_rule, tokens = [], Counter(), Counter()
    changed = 0
    for r in plan:
        source = r["label"] if spoken is None or r["id"] not in spoken else spoken[r["id"]]
        text, subs = lexicon.render(source)
        out.append(dict(r) | {"tts_text": text})
        changed += text != r["label"]
        for s in subs:
            by_rule[s["rule"]] += 1
            if s["rule"].startswith("caps"):
                tokens[s["token"]] += 1
    summary = {"lexicon": lexicon.record(), "clips": len(plan), "clips_changed": changed,
               "substitutions_by_rule": dict(sorted(by_rule.items())),
               "default_rule_tokens": dict(tokens.most_common())}
    return out, summary
