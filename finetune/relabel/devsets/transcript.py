"""Transcript text handling for segmentation: markup tags, alignment units and sentence ends.

- Markup tags (`<...>` and `[...]`, e.g. "<UNIN/>", "<UNSURE>word</UNSURE>", "<inaudible>", "[laughter]") are
  removed from references; the evaluation normalizer removes them too, so this changes no score.
- Each whitespace token of the transcript is one alignment unit. Its forced-alignment characters are the
  aligner's alphabet (lowercase a-z and the apostrophe) after accent folding, with digits, "%", "&", "+" and
  currency signs spelled out (alignment only; the reference keeps the source text). A token with markup
  only (an untranscribed sound or word) or with no Latin letters becomes a wildcard unit (chars None),
  which the aligner matches to any audio. A token without letters, digits or tags (a dash) is no unit.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

TAG = re.compile(r"<[^<>]*>|\[[^\[\]]*\]")
ALPHABET = frozenset("abcdefghijklmnopqrstuvwxyz'")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?(?:st|nd|rd|th|s)?")
_SENTENCE_END = re.compile(r"[.?!][\"')\]]*$")
_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
         "sixteen seventeen eighteen nineteen").split()
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_SCALES = ((10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (10**3, "thousand"))
_ORDINAL = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
            "nine": "ninth", "twelve": "twelfth"}
_CURRENCY = {"$": "dollars", "€": "euros", "£": "pounds"}


def strip_tags(text: str) -> str:
    """Text without markup tags, whitespace collapsed."""
    return " ".join(TAG.sub(" ", text).split())


def _below_thousand(n: int) -> list[str]:
    words = []
    if n >= 100:
        words += [_ONES[n // 100], "hundred"]
        n %= 100
    if n >= 20:
        words.append(_TENS[n // 10])
        n %= 10
        if n:
            words.append(_ONES[n])
    elif n or not words:
        words.append(_ONES[n])
    return words


def cardinal(n: int) -> list[str]:
    if n >= 10**15:
        return [_ONES[int(d)] for d in str(n)]
    if n < 1000:
        return _below_thousand(n)
    words = []
    for value, name in _SCALES:
        if n >= value:
            words += _below_thousand(n // value) + [name]
            n %= value
    if n:
        words += _below_thousand(n)
    return words


def number_words(token: str) -> str:
    """Spoken-form approximation of a numeric token ("1,234.5", "1990s", "21st", "2020") for alignment."""
    suffix = ""
    match = re.fullmatch(r"([\d,.]+?)(st|nd|rd|th|s)?", token)
    body, suffix = (match.group(1), match.group(2) or "") if match else (token, "")
    integer, _, decimals = body.partition(".")
    digits = integer.replace(",", "")
    if not digits.isdigit():
        return " ".join(_ONES[int(c)] for c in token if c.isdigit())
    n = int(digits)
    if not decimals and "," not in integer and len(digits) == 4 and (1100 <= n <= 1999 or 2010 <= n <= 2099):
        words = _below_thousand(n // 100) + (_below_thousand(n % 100) if n % 100 else ["hundred"])
    else:
        words = cardinal(n)
    if decimals.isdigit():
        words += ["point"] + [_ONES[int(c)] for c in decimals]
    if suffix in ("st", "nd", "rd", "th"):
        last = words[-1]
        words[-1] = _ORDINAL.get(last, last[:-1] + "ieth" if last.endswith("y") else last + "th")
    elif suffix == "s":
        last = words[-1]
        words[-1] = last[:-1] + "ies" if last.endswith("y") else last + "s"
    return " ".join(words)


def alignment_chars(word: str) -> str | None:
    """The aligner characters of one tag-free token, or None if it has no Latin letter or digit."""
    w = word.replace("’", "'").replace("‘", "'")
    w = "".join(c for c in unicodedata.normalize("NFKD", w) if not unicodedata.combining(c)).lower()
    for sign, name in _CURRENCY.items():
        if sign in w:
            w = w.replace(sign, " ") + " " + name
    w = w.replace("&", " and ").replace("%", " percent ").replace("+", " plus ")
    w = _NUMBER.sub(lambda m: " " + number_words(m.group()) + " ", w)
    chars = "".join(c for c in w if c in ALPHABET).strip("'")
    return chars or None


@dataclass(frozen=True)
class Token:
    text: str          # reference text of the token (tags removed); "" for a markup-only token
    chars: str | None  # aligner characters; None = wildcard unit


def tokens(text: str) -> list[Token]:
    """Alignment units of a transcript, in order (see the module docstring)."""
    out = []
    for raw in text.split():
        clean = TAG.sub("", raw).strip()
        if not clean:
            if TAG.search(raw):
                out.append(Token("", None))
            continue
        if not any(c.isalnum() for c in clean):
            continue
        out.append(Token(clean, alignment_chars(clean)))
    return out


def ends_sentence(token: Token, boundary_words: frozenset[str] = frozenset()) -> bool:
    """Sentence end: the token ends with . ? or ! (closing quotes allowed), or is a dictated boundary word."""
    if _SENTENCE_END.search(token.text):
        return True
    return bool(boundary_words) and token.text.lower().strip(".,;:!?\"'") in boundary_words
