"""Output-style transforms (b0prime-v2 plan, "Output style: verbatim or clean (Step 1)"): four classes of verbatim
content, each a deterministic text transform T_c, used both to measure (on hypotheses: b0v2_step1.py) and, for the
classes AJ adopts, to clean labels (cleaning version 4's style layer, a later module). Their definitions use no dev data.

    st = Style.load()                       the committed short-word table (SHORT_WORDS)
    st.transform(text, cls) -> str          T_c(text) for cls in CLASSES
    st.edits(text, cls) -> (str, [Edit])    the output and every changed span (offsets into the input)
    st.identity() -> {classes, short_words_sha256}

Tokens are whitespace-separated. A token is leading non-word characters, a core and trailing non-word characters (as
cleaning.py's layer 0); a word is a token with a non-empty core. A token's key is its core case-folded, with "’" read as
"'": inner apostrophes and periods are kept, so "we're" never equals "were" and "U.S." never equals "us".

Protected tokens. A letter run is 3 or more consecutive single-letter tokens (core one letter, with or without dots:
"F", "F.", "F,"), unless every one is "I" ("S A A B" is a run; the stutter "I I think" is not). A number token has a digit
in its core, or a core whose hyphen-separated parts are all NUMBER_WORDS (the pinned normalizer writes them as digits:
"one one" -> "11", "oh oh" -> "00"). No transform removes or changes a letter-run token, and none removes a number token
or changes its core (E may remove an ellipsis attached to one, D the comma after one).

E, ellipsis. An ellipsis is a maximal run of "." and "…" that holds a "…" or three or more characters, except a run with a
digit directly on both sides ("1...10") and a run inside or attached to a letter-run token. At the end of the text
(nothing after it but closing quotes, closing brackets and spaces) it becomes "."; it is dropped instead, with the spaces
before it, when the last character before it (spaces skipped) is ".", "?" or "!" ("Really?..." -> "Really?"). Anywhere
else it is removed: with the spaces after it when spaces surround it or it starts the text ("but ... nah" -> "but nah");
with the spaces before it when it follows a space and precedes punctuation ("word ...? yes" -> "word? yes"); replaced by
one space when it sits between two non-space characters and the next is a word character or an opening quote or
bracket ("a…b" -> "a b"); else alone ("but... nah" -> "but nah", "2.5… billion" -> "2.5 billion"). No word changes.

F, fragments. A word that is not protected is a fragment when
  - it ends in "-" (its last character), unless the next token's key is "and", "or", "to" or "nor" (a suspended hyphen:
    "perso- and"), or
  - it is 1-3 letters and nothing else (no punctuation: "M." is an initial), its key is not in the short-word table,
    and the next token is a word whose key strictly starts with its key ("w we", "th these").
Fragments are removed, until none is left ("th perso- these" -> "these").

R, repetitions. An immediate repeat is n = 1, 2 or 3 words followed by n words with the same keys; the first copy is
removed and the last kept ("in a, in a" -> "in a"), leftmost first and shortest n first, until none is left. Never
when a token of either copy is protected or is a single letter other than "I" or "a", or when the copy is one
REAL_DOUBLES key, once or repeated ("no no" and "No, no, no, no" stay).

D, discourse markers. "you know" or "I mean": two tokens with keys you+know or i+mean, the first bare (no leading or
trailing characters), the second with no leading characters and a trailing "," (followed by a comma), or the last
token of the text with any closing marks (followed by the end); preceded by the start of the text or by a token ending
in ",". Markers are found on the text as it stands and removed together, until none is left. The comma before a
removed block is dropped ("so, you know, the" -> "so the"), except on a letter-run token, which is never changed; a block
that ends the text puts its last token's closing marks, other than ",", in place of that comma ("it's good, you know?"
-> "it's good?"). Undelimited uses ("do you know", "as you know") are left alone.

Case (F, R, D). When a removed block starts the text or follows a token ending in ".", "?" or "!", and the block's first
letter is uppercase, the first character of the next kept word's core is uppercased if it is a lowercase letter and the
word is not protected ("You know, it's" -> "It's", "The the cat" -> "The cat", "Th- the" -> "The").

Guard (every class). A text with a word whose transform would leave no word is returned unchanged ("I mean",
"you know,"): no transform empties a text.

Every class is applied until a pass changes nothing, so T_c(T_c(x)) = T_c(x) (one pass suffices for E).

Short-word table (SHORT_WORDS, terms/cleaning/short_words.tsv; one lowercase word per line, "#" comments): English words
of 1-3 letters from a standard dictionary, SCOWL's English and American word lists up to size 60 (build_short_words;
`./python -m training.style short-words <SCOWL share/scowl directory>` prints the table). Of the single letters only
"a" and "i" are listed: SCOWL lists every letter, and a letter is not a word here ("w we" is a fragment). SCOWL's
copyright notices and conditions are kept verbatim next to the table (SCOWL_COPYRIGHT, the distribution's Copyright
file), as its licence requires.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import re
import sys
from pathlib import Path

CLASSES = ("E", "F", "R", "D")
SHORT_WORDS = Path(__file__).resolve().parents[1] / "terms" / "cleaning" / "short_words.tsv"
SCOWL_COPYRIGHT = SHORT_WORDS.with_name("short_words_scowl_copyright.txt")   # SCOWL's notices, verbatim

NUMBER_WORDS = frozenset(
    "zero oh one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
    "eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion "
    "trillion".split())
REAL_DOUBLES = frozenset({"had", "that", "very", "really", "no", "yeah", "bye", "ha", "so"})
SUSPENDED_BEFORE = frozenset({"and", "or", "to", "nor"})
MARKERS = (("you", "know"), ("i", "mean"))
REPEAT_MAX = 3
FRAGMENT_MAX = 3
SENTENCE_MARKS = ".?!"
SCOWL_LISTS = ("english-words", "american-words")
SCOWL_SIZE = 60

_TOKEN = re.compile(r"\S+")
_PARTS = re.compile(r"^(\W*)(.*?)(\W*)$", re.S)
_ELLIPSIS = re.compile(r"[.…]+")
_END = re.compile(r"[\s\"'”’»›)\]}]*")
_OPENERS = "\"'“‘«‹([{¿¡"
_LETTERS = re.compile(r"[^\W\d_]{1,%d}" % FRAGMENT_MAX)


@dataclasses.dataclass(frozen=True)
class Edit:
    """One changed span of the input: text[start:end] (= before) becomes after."""
    cls: str
    start: int
    end: int
    before: str
    after: str


class _Tok:
    """A token: its position among the input's tokens, its offsets in the input and its text as it stands (D may drop
    its comma; the case rule may uppercase its first letter), split into lead, core and trail."""
    __slots__ = ("index", "start", "end", "_text", "lead", "core", "trail", "key")

    def __init__(self, index: int, start: int, end: int, text: str):
        self.index, self.start, self.end = index, start, end
        self.text = text

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        self._text = value
        self.lead, self.core, self.trail = _PARTS.match(value).groups()
        self.key = self.core.casefold().replace("’", "'")

    @property
    def word(self) -> bool:
        return bool(self.core)


def _single_letter(t: _Tok) -> bool:
    c = t.core
    return len(c) == 1 and c.isalpha()


def _number(t: _Tok) -> bool:
    c = t.core
    return bool(c) and (any(ch.isdigit() for ch in c) or all(p in NUMBER_WORDS for p in t.key.split("-")))


def _letter_runs(toks: list[_Tok]) -> set[int]:
    """Indices (into toks) of the tokens of letter runs."""
    out, i = set(), 0
    while i < len(toks):
        j = i
        while j < len(toks) and _single_letter(toks[j]):
            j += 1
        if j - i >= 3 and any(t.key != "i" for t in toks[i:j]):
            out.update(range(i, j))
        i = max(j, i + 1)
    return out


def _has_word(text: str) -> bool:
    return any(_PARTS.match(m.group()).group(2) for m in _TOKEN.finditer(text))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_short_words(path: Path) -> list[str]:
    """The words of a short-word table, in file order."""
    words = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            words.append(line)
    return words


class Style:
    def __init__(self, short_words: frozenset[str], short_words_sha256: str | None = None):
        self.short_words = frozenset(short_words)
        self.short_words_sha256 = short_words_sha256 or hashlib.sha256(
            "\n".join(sorted(self.short_words)).encode()).hexdigest()

    @classmethod
    def load(cls, path: Path = SHORT_WORDS) -> "Style":
        return cls(frozenset(read_short_words(path)), sha256_file(path))

    def identity(self) -> dict:
        return {"classes": list(CLASSES), "short_words_sha256": self.short_words_sha256}

    def transform(self, text: str, cls: str) -> str:
        return self.edits(text, cls)[0]

    def edits(self, text: str, cls: str) -> tuple[str, list[Edit]]:
        if cls not in CLASSES:
            raise ValueError(f"unknown style class {cls!r} (classes: {', '.join(CLASSES)})")
        edits = _ellipsis_edits(text) if cls == "E" else self._token_edits(text, cls)
        out = _apply(text, edits)
        if _has_word(text) and not _has_word(out):
            return text, []
        return out, edits

    # ---------------------------------------------------------------------------------------- F, R, D
    def _token_edits(self, text: str, cls: str) -> list[Edit]:
        toks = [_Tok(k, m.start(), m.end(), m.group()) for k, m in enumerate(_TOKEN.finditer(text))]
        step = {"F": self._fragments, "R": _repeat, "D": _markers}[cls]
        cur = list(toks)
        while True:
            nxt = step(cur)
            if nxt is None:
                break
            cur = nxt
        kept = {t.index for t in cur}
        _capitalize(toks, kept)
        return _token_diff(text, toks, kept, cls)

    def _fragments(self, cur: list[_Tok]) -> list[_Tok] | None:
        runs = _letter_runs(cur)
        drop = set()
        for k, t in enumerate(cur):
            if not t.word or k in runs or _number(t):
                continue
            nxt = cur[k + 1] if k + 1 < len(cur) else None
            if t.text.endswith("-"):
                if nxt is None or nxt.key not in SUSPENDED_BEFORE:
                    drop.add(k)
            elif _LETTERS.fullmatch(t.text) and t.key not in self.short_words and nxt is not None and nxt.word \
                    and len(nxt.key) > len(t.key) and nxt.key.startswith(t.key):
                drop.add(k)
        return [t for k, t in enumerate(cur) if k not in drop] if drop else None


def _repeat(cur: list[_Tok]) -> list[_Tok] | None:
    runs = _letter_runs(cur)
    ok = [t.word and k not in runs and not _number(t) and not (_single_letter(t) and t.key not in ("i", "a"))
          for k, t in enumerate(cur)]
    for i in range(len(cur)):
        for n in range(1, REPEAT_MAX + 1):
            if i + 2 * n > len(cur):
                break
            if not all(ok[i:i + 2 * n]) or any(cur[i + j].key != cur[i + n + j].key for j in range(n)):
                continue
            if cur[i].key in REAL_DOUBLES and all(cur[i + j].key == cur[i].key for j in range(n)):
                continue
            return cur[:i] + cur[i + n:]
    return None


def _markers(cur: list[_Tok]) -> list[_Tok] | None:
    runs = _letter_runs(cur)
    drop, k = set(), 0
    while k + 1 < len(cur):
        a, b = cur[k], cur[k + 1]
        last = k + 1 == len(cur) - 1
        if (a.key, b.key) in MARKERS and not a.lead and not a.trail and not b.lead \
                and (b.trail == "," or last) and k not in runs and k + 1 not in runs \
                and (k == 0 or cur[k - 1].text.endswith(",")):
            drop.update((k, k + 1))
            k += 2
        else:
            k += 1
    if not drop:
        return None
    for start, stop in _blocks(sorted(drop)):
        if start == 0:
            continue
        before = cur[start - 1]
        if start - 1 in runs or start - 1 in drop:
            continue
        marks = cur[stop - 1].trail if stop == len(cur) else ""
        before.text = before.text[:-1] + ("" if marks in ("", ",") else marks)
    return [t for k, t in enumerate(cur) if k not in drop]


def _blocks(indices: list[int]) -> list[tuple[int, int]]:
    """Maximal runs of consecutive integers, as [start, stop)."""
    out = []
    for i in indices:
        if out and out[-1][1] == i:
            out[-1] = (out[-1][0], i + 1)
        else:
            out.append((i, i + 1))
    return out


def _capitalize(toks: list[_Tok], kept: set[int]) -> None:
    """The case rule over the removed blocks (indices into the input's tokens)."""
    removed = [t.index for t in toks if t.index not in kept]
    survivors = [t for t in toks if t.index in kept]
    runs = {survivors[k].index for k in _letter_runs(survivors)}
    for start, stop in _blocks(removed):
        if stop >= len(toks):
            continue
        prev, nxt = (toks[start - 1] if start else None), toks[stop]
        if prev is not None and not prev.text.endswith(tuple(SENTENCE_MARKS)):
            continue
        first = next((ch for t in toks[start:stop] for ch in t.text if ch.isalpha()), "")
        if not first.isupper() or not nxt.word or nxt.index in runs or _number(nxt):
            continue
        lead, c = nxt.lead, nxt.core
        if c[0].islower():
            nxt.text = lead + c[0].upper() + nxt.text[len(lead) + 1:]


def _token_diff(text: str, toks: list[_Tok], kept: set[int], cls: str) -> list[Edit]:
    """Edits of a token-level result: removed blocks (with the spaces they leave) and changed kept tokens."""
    edits = []
    removed = [t.index for t in toks if t.index not in kept]
    for start, stop in _blocks(removed):
        if stop < len(toks):
            s, e = toks[start].start, toks[stop].start
        elif start > 0:
            s, e = toks[start - 1].end, toks[stop - 1].end
        else:
            continue  # every token removed: the guard returns the input
        edits.append(Edit(cls, s, e, text[s:e], ""))
    for t in toks:
        if t.index in kept and t.text != text[t.start:t.end]:
            edits.append(Edit(cls, t.start, t.end, text[t.start:t.end], t.text))
    return sorted(edits, key=lambda x: (x.start, x.end))


# -------------------------------------------------------------------------------------------- E
def _ellipsis_edits(text: str) -> list[Edit]:
    toks = [_Tok(k, m.start(), m.end(), m.group()) for k, m in enumerate(_TOKEN.finditer(text))]
    runs = _letter_runs(toks)
    guarded = [(toks[k].start, toks[k].end) for k in runs]
    edits, floor = [], 0
    for m in _ELLIPSIS.finditer(text):
        s, e = m.span()
        if "…" not in m.group() and e - s < 3:
            continue
        if s > 0 and e < len(text) and text[s - 1].isdigit() and text[e].isdigit():
            continue
        if any(a <= s and e <= b for a, b in guarded):
            continue
        left = text[s - 1] if s > 0 else ""
        right = text[e] if e < len(text) else ""
        if _END.fullmatch(text, e):
            before = text[:s].rstrip()
            if before and before[-1] in SENTENCE_MARKS:
                edit = (len(before), e, "")
            else:
                edit = (s, e, ".")
        elif (not left or left.isspace()) and right.isspace():
            stop = e
            while stop < len(text) and text[stop].isspace():
                stop += 1
            edit = (s, stop, "")
        elif left.isspace() and not (right.isalnum() or right == "_" or right in _OPENERS):
            begin = s
            while begin > floor and text[begin - 1].isspace():
                begin -= 1
            edit = (begin, e, "")
        elif left and not left.isspace() and (right.isalnum() or right == "_" or right in _OPENERS):
            edit = (s, e, " ")
        else:
            edit = (s, e, "")
        a, b, after = edit
        a = max(a, floor)
        edits.append(Edit("E", a, b, text[a:b], after))
        floor = b
    return edits


def _apply(text: str, edits: list[Edit]) -> str:
    out, pos = [], 0
    for e in edits:
        out += [text[pos:e.start], e.after]
        pos = e.end
    out.append(text[pos:])
    return "".join(out)


# ---------------------------------------------------------------------------- the short-word table
def build_short_words(scowl_dir: Path, size: int = SCOWL_SIZE) -> list[str]:
    """SCOWL's English and American word lists of sizes up to `size` (files <list>.<size>, ISO-8859-1): lowercase
    ASCII words of 1-3 letters; of the single letters only "a" and "i"."""
    words = set()
    for p in sorted(Path(scowl_dir).iterdir()):
        name, _, level = p.name.rpartition(".")
        if name in SCOWL_LISTS and level.isdigit() and int(level) <= size:
            for w in p.read_text(encoding="latin-1").split("\n"):
                w = w.strip()
                if re.fullmatch(r"[a-z]{1,%d}" % FRAGMENT_MAX, w) and (len(w) > 1 or w in ("a", "i")):
                    words.add(w)
    return sorted(words)


SCOWL_NOTICE = """\
SCOWL notice (the word lists this table is selected from):
  Copyright 2000-2018 by Kevin Atkinson

  Permission to use, copy, modify, distribute and sell these word
  lists, the associated scripts, the output created from the scripts,
  and its documentation for any purpose is hereby granted without fee,
  provided that the above copyright notice appears in all copies and
  that both that copyright notice and this permission notice appear in
  supporting documentation. Kevin Atkinson makes no representations
  about the suitability of this array for any purpose. It is provided
  "as is" without express or implied warranty."""


def short_words_table(words: list[str], source: str) -> str:
    head = [f"Short English words for the output-style class F (training/style.py; b0prime-v2 plan, \"Output style\"):",
            "a 1-3-letter token that is a strict prefix of the next word is a fragment unless it is listed here.",
            f"Source: {source}; the {' and '.join(SCOWL_LISTS)} lists of sizes up to {SCOWL_SIZE}, lowercase ASCII",
            "entries of 1-3 letters; of the single letters only \"a\" and \"i\" (SCOWL lists every letter). Built by",
            "    ./python -m training.style short-words <SCOWL share/scowl directory> > terms/cleaning/short_words.tsv",
            f"SCOWL's complete copyright notices and conditions, verbatim: {SCOWL_COPYRIGHT.name} (next to this table).",
            "", *SCOWL_NOTICE.splitlines()]
    return "".join(f"# {line}".rstrip() + "\n" for line in head) + "".join(f"{w}\n" for w in words)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="training.style")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sw = sub.add_parser("short-words", help="print the short-word table built from a SCOWL share/scowl directory")
    sw.add_argument("scowl_dir", type=Path)
    sw.add_argument("--source", default="SCOWL 2020.12.07 (http://wordlist.aspell.net/)")
    args = ap.parse_args(argv)
    sys.stdout.write(short_words_table(build_short_words(args.scowl_dir), args.source))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
