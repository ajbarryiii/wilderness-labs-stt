"""Label cleaning (stage 2a, cleaning plan): a versioned, deterministic read-time layer over the training labels.

    spec = Cleaning.load(choices_dir=..., version=V)   committed tables + the layer 2 choices artifact of one label run
    spec.clean(text, item_id, label_set) -> str    the cleaned training text of one kept label
    spec.edits(text, item_id, label_set)           (cleaned text, [Edit]) with every changed span
    spec.spans(text) -> [Span]                     layer 2 spans of a label (what the choices artifact decides)
    spec.identity() -> {version, tables_sha256, choices_sha256}

training/labels.py LabelStore.open(..., cleaning=spec) applies clean() to every kept label of either label set;
cleaning=None reproduces the uncleaned texts and identity byte for byte.

Versions (VERSIONS; the version is in the identity and the artifact's manifest, so a run's resume check refuses a
checkpoint of another version): 1 = layer 2 + the tables of layer 1 (stages 2b and 2c; production() defaults to it);
2 = version 1 + spelled-out acronyms in layer 1, before the tables (cleaning plan v5), + layer 0, B0's punctuation
transferred into T*'s labels (cleaning plan v6; T* only; its rule constants LAYER0 enter the identity); stage 3 and
later. Version 1's texts are unchanged by version 2's code.

Layer 0 (version 2, T* only; clean/edits take the id's B0 label as `donor`, as stored in `b0_raw`; rule "union"):
tokens are whitespace-separated; each is leading non-word characters, a core and trailing non-word characters, and a
pure-punctuation token joins the trail of the word before it. A word's key is its core lowercased with non-word
characters removed; a word's mark is the last of PUNCT_MARKS in its trail. The keys of T* and B0 are aligned by a
minimum-edit Levenshtein alignment (backtrace from the end preferring the diagonal, then a T* word without a B0 word);
aligned = matched with equal keys. At an aligned pair (i, j), B0's mark is inserted right after T*'s word's core when
T*'s word has no mark, B0's word has one, T*'s trail holds only marks and PUNCT_CLOSERS (so a false start "the-" is
left alone), B0's core has no "." (abbreviations), and the boundary condition holds: (i + 1, j + 1) is aligned too, or
i and j are both the last words. The mark goes right after the core, before any closing quote or bracket (as the
analysis prototype). If the inserted mark is one of PUNCT_ENDS, (i + 1, j + 1) is aligned and B0's word j + 1 starts
with a capital letter, T*'s word i + 1, if it starts with a lowercase letter, takes it upper-cased. Guard: nothing is done
when B0 has no word or the word edit rate of B0's keys against T*'s exceeds PUNCT_GUARD. Layer 0 never edits inside a
layer 2 span (an insertion strictly inside one, a capital within one), so layer 2's spans stay as layer 2 wrote them.
No mark is removed or changed, no word added, removed or respelled; an unchanged label is byte-identical. Its edits
are layer 0 edits of table PUNCT_TABLE (group "mark": an insertion, empty original; group "capital").

Tables (terms/cleaning/, committed; UTF-8; blank lines and lines starting with "#" are ignored; fields are separated
by one TAB; every written form is ASCII):

    forms.tsv     written<TAB>conventional             casing and hyphenation only: the two sides may differ only in
                                                       letter case and a hyphen versus a space (checked), so the
                                                       scorer's normalized words never change. No diacritic folding.
    variants.tsv  canonical<TAB>variant[<TAB>...]      a pure spelling-variant group (the same word in every
                                                       context): every variant becomes the canonical (first)
                                                       form, the form T* writes most often (cleaning plan v4).
                                                       Members differ by more than case. A canonical form with
                                                       capitals is also enforced on its own spelling ("ptsd" ->
                                                       "PTSD"); an all-lowercase one leaves its own casing as
                                                       written ("Dataset").
    dialect.tsv   dialect<TAB>standard                 a dialect or reduced spelling of a common word becomes its
                                                       standard spelling. A dialect form may begin or end with an
                                                       apostrophe ('cause, 'em).
    choices.tsv   member<TAB>member[<TAB>...]          one layer 2 set: spellings of the same spoken word or words,
                                                       offered to the model in this order. A member ending in "*" is
                                                       option-only: offered, but it does not start a span (the
                                                       standard spelling of a dialect set: "get*", "you*").

Matching (both layers): a written form matches case-insensitively on whole tokens. A
token is a run of letters and digits (any script) joined by internal apostrophes or periods ("y'all", "don't",
"index.html", "config.xml" are one token each), so a form never matches part of a token ("ya" is not found in
"ya'll", "git" not in "Git's", "html" not in "index.html"); an edge apostrophe belongs to a form only where the form
lists it. Spaces and hyphens inside a form match literally. Scanning is left to right; at each position the longest
form wins; matches never overlap.

Quote guard (dialect forms with an edge apostrophe): a form ending in an apostrophe is not applied when an unclosed
single quote precedes it (the apostrophe is then the closing quote: with a form "goin'", "'goin'" and "said 'I was
goin'" stay as written),
and a form beginning with an apostrophe is not applied when the token is followed by an apostrophe ("'em'"), so a
quote mark is never eaten. An opening single quote is an apostrophe before a token that is not the start of a listed
leading-apostrophe form; the next token-final apostrophe closes it.

Capitalization (every replacement, both layers): when the first letter of the original span is a capital and the
replacement is all lowercase, the replacement's first letter is capitalized ("Gonna" -> "Going to", "Maybe" ->
"May be", "Double check" -> "Double-check"); a replacement with any capital letter is written as listed ("GAAP",
"Git", "OK", "I'm going to").

Spelled-out acronyms (version 2, layer 1, before the tables; table "acronyms", group "spelled-out"): a run of two or
more capitals A-Z, each a single-letter token, separated by ". ", "." or " ", becomes the compact acronym ("A. P. I.",
"A.P.I.", "A P I" -> "API"). A dotted run (any separator with ".") takes one final "." after its last letter; an
undotted run's following "." is the sentence's and stays. A plural "s" or "'s" (after the last letter or the final
".", optionally after one space: "G.P.U.s", "G. P. U. s", "G P U's", "G P Us") becomes "s"; a possessive reads as a
plural. A dotted run takes no "s" attached directly to its last letter ("hard R. Is that": the word "Is").
Whole tokens: the run is not joined on either side to a letter or digit, directly or by "'", "." or "-" (hyphens join,
they do not separate: "A-V-E-N-S-A-R" and "C T X-ray" keep the hyphenated token; a last capital that starts a word,
"A B Cat", is not part of the run). Left as written, before anything else: two letters dotted with no space ("U.S.")
or undotted ("T V"); a run of only "I" ("I I think"); initials, a run followed by one space and a word that starts
with a capital and is not all capitals as layer 1 will write it ("J. A. Mellon", also "...the A. P. I. We"). The
final "." is kept only when it is the label's last character ("...the A. P. I." -> "...the API."), else dropped
("the A.P.I. and" -> "the API and"). The tables then apply to the result ("H T M L" -> "HTML", "M A C O S" -> "macOS",
one edit), and the two repeat until a round changes nothing (a table edit can split a chain left as written: "P T S
D. M R Is" -> "PTSD. MRIs"), so layer 1 is idempotent. Runs overlapping a layer 2 span are left as written.

Load-time checks: every side has the token syntax above; written forms are unique (case-insensitively) across the
three layer 1 tables; layer 1 is idempotent (each output is left unchanged by layer 1, and no output, as listed or
with its first letter capitalized, has a single capital letter as a token that could join an acronym run); a layer 2
set has two or more distinct members and at least one that starts spans; no member key is shared by two sets; the
layers are disjoint: no layer 1 form matches inside a member, and no member matches inside a layer 1 form.

Read-time order (clean): the empty text stays empty. Layer 2: the spans of the original text (members that start
spans, matched as above) take the recorded choice of the artifact, if one was chosen (else they are left as
written); the replacement must be a member of the span's set, and the text outside the spans is verified
byte-identical. Then (version 2, T*) layer 0 on the result. Then layer 1 (version 2: acronyms, then the tables),
which never touches a layer 2 span. A non-empty text never becomes empty. Edits of layers 0 and 1 that overlap become
one edit of layer 1's table and group.
Without a choices artifact (choices_dir None) layer 2 leaves every span as written (layer 1 only; used by checks).

The choices artifact ($RELABEL_ROOT/cleaning/<label run>/v<version>/, written by training/cleaning_run.py):
choices.parquet (CHOICE_COLUMNS: one row per span of a kept label of either label set) and manifest.json
(version, model alias and revision, template sha256, threshold, tables sha256, the label run's contract sha256,
row count, file sha256). Cleaning.load refuses an artifact whose file, row count, version, template, threshold or
tables differ; bind_run refuses one of another label run contract; clean refuses a label whose spans differ from
the recorded ones or whose text at [start, end) is not the recorded `original`, and a recorded choice that is not
a member of the span's set.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import terms as committed_terms

VERSION = 2                    # the latest cleaning version (the default of Cleaning and the runner)
VERSIONS = (1, 2)              # 1: stages 2b and 2c; 2: version 1 + spelled-out acronyms (plan v5), stage 3 on
TABLES_DIR = committed_terms.HERE / "cleaning"
TABLE_FILES = ("forms.tsv", "variants.tsv", "dialect.tsv", "choices.tsv")
THRESHOLD = 0.9
CANT_TELL = "can't tell"
LETTERS = "ABCDEFGHIJ"
MARK_OPEN, MARK_CLOSE = "[[", "]]"
OPTION_ONLY = "*"
CHOICE_COLUMNS = ("label_set", "id", "span", "start", "end", "original", "set", "options", "probs", "chosen")

SYSTEM = "You are a careful copy editor of speech transcripts."
USER = """Below is a transcript of speech. One place in it is marked with {open} and {close}. Every option below is a spelling of what the speaker said at that place. Choose the spelling the speaker meant, given the context. Prefer the standard spelling of an ordinary word over a dialect spelling. If the context does not decide between the spellings, choose "{cant_tell}".

Transcript: {marked}

Options:
{options}

Answer with the letter of one option only."""
TEMPLATE_SHA256 = hashlib.sha256(json.dumps(
    {"system": SYSTEM, "user": USER, "open": MARK_OPEN, "close": MARK_CLOSE, "cant_tell": CANT_TELL,
     "letters": LETTERS}, sort_keys=True).encode()).hexdigest()

_WORD = r"[^\W_]+(?:'[^\W_]+)*"
_PHRASE = re.compile(rf"{_WORD}(?:[ -]{_WORD})*")
_DIALECT = re.compile(rf"'?{_WORD}(?:[ -]{_WORD})*'?")
_STANDARD = re.compile(rf"{_WORD}(?:[ -]{_WORD})*")
_LEFT = r"(?<![^\W_])(?<![^\W_]['.])"
_RIGHT = r"(?![^\W_])(?!['.][^\W_])"
# spelled-out acronyms (version 2): a chain of capitals A-Z separated by ". ", "." or " ", starting a token (not
# joined to a token on its left by "'", "." or "-"); _acronym cuts it into a run
_CHAIN = re.compile(r"(?<![^\W_])(?<![^\W_]['.\-])[A-Z](?:(?:\. ?| )[A-Z])+")
_JOINED = re.compile(r"[^\W_]|['.\-][^\W_]")       # at a position: the token goes on
_PLURAL = re.compile(r" ?'?s(?![^\W_])(?!['.\-][^\W_])")
_NEXT_WORD = re.compile(r" (\S+)")
_LONE_CAPITAL = re.compile(r"(?<![^\W_])(?<![^\W_]['.\-])[A-Z](?![^\W_])(?!['.\-][^\W_])")
ACRONYM_TABLE, ACRONYM_GROUP = "acronyms", "spelled-out"
LAYER1_ROUNDS = 8              # version 2: rounds of (acronyms, tables) allowed to reach a text they leave unchanged
# layer 0 (version 2, cleaning plan v6): B0's punctuation transferred into T*'s labels
PUNCT_MARKS = ".,?!:;"
PUNCT_ENDS = ".?!"
PUNCT_CLOSERS = "\"')]}”’"
PUNCT_GUARD = 0.3
PUNCT_LABEL_SET = "T*"
PUNCT_TABLE = "punctuation"
LAYER0 = {"name": "punctuation transfer", "rule": "union", "version": 1, "label_set": PUNCT_LABEL_SET,
          "donor": "b0_raw", "marks": PUNCT_MARKS, "sentence_marks": PUNCT_ENDS, "closers": PUNCT_CLOSERS,
          "guard": PUNCT_GUARD, "boundary": "next words aligned, or both last", "alignment": "levenshtein, diagonal "
          "first, then a T* word without a B0 word", "abbreviation": "no '.' in B0's core"}
_TOKEN = re.compile(r"\S+")
_PARTS = re.compile(r"^(\W*)(.*?)(\W*)$", re.S)
_NONWORD = re.compile(r"\W")


class ArtifactMismatch(RuntimeError):
    """The choices artifact does not belong to these tables, this label run or this label."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _form_key(s: str) -> str:
    return s.lower().replace("-", " ")


def cased(replacement: str, original: str) -> str:
    """The capitalization rule (module docstring)."""
    first = next((c for c in original if c.isalpha()), "")
    if first.isupper() and replacement == replacement.lower():
        for k, c in enumerate(replacement):
            if c.isalpha():
                return replacement[:k] + c.upper() + replacement[k + 1:]
    return replacement


def _compile(keys) -> re.Pattern | None:
    """Case-insensitive whole-token alternation of the (lowercase) keys, longest first."""
    keys = sorted(set(keys), key=lambda w: (-len(w), w))
    if not keys:
        return None
    return re.compile(_LEFT + "(?:" + "|".join(re.escape(k) for k in keys) + ")" + _RIGHT, re.IGNORECASE)


def _lines(text: str, name: str):
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = [f.strip() for f in line.split("\t")]
        if any(not f for f in fields):
            raise ValueError(f"{name}:{n}: empty field (fields are separated by one TAB)")
        yield n, fields


def _check_side(side: str, pattern: re.Pattern, where: str) -> None:
    if not side.isascii() or not pattern.fullmatch(side):
        raise ValueError(f"{where}: {side!r} is not a whole-token ASCII form")


@dataclass(frozen=True)
class Entry:
    """One layer 1 written form."""
    key: str           # the written form, lowercase
    output: str
    table: str         # "forms" | "variants" | "dialect"
    group: str         # the group it is screened under (the conventional / canonical form, the dialect form)


@dataclass(frozen=True)
class Edit:
    """One changed span of a cleaned label: [start, end) of the original text, [out_start, out_end) of the result."""
    layer: int
    table: str         # "choices" | "forms" | "variants" | "dialect"
    group: str
    start: int
    end: int
    original: str
    replacement: str
    out_start: int = -1
    out_end: int = -1
    run: str | None = None   # an acronym edit: the spelled-out run as written (original may add merged table words)


@dataclass(frozen=True)
class ChoiceSet:
    index: int
    members: tuple[str, ...]
    option_only: frozenset

    @property
    def name(self) -> str:
        return "|".join(self.members)

    @property
    def options(self) -> tuple[str, ...]:
        return self.members + (CANT_TELL,)

    @property
    def triggers(self) -> set[str]:
        return {m.lower() for m in self.members if m not in self.option_only}


@dataclass(frozen=True)
class Span:
    index: int         # position among the label's spans
    start: int
    end: int
    original: str
    set: ChoiceSet


class Layer1:
    def __init__(self, entries: list[Entry]):
        self.entries: dict[str, Entry] = {}
        for e in entries:
            if e.key in self.entries:
                other = self.entries[e.key]
                raise ValueError(f"{e.key!r} is listed in {other.table}.tsv and {e.table}.tsv")
            self.entries[e.key] = e
        self._re = _compile(self.entries)
        self.leading = tuple(sorted(k for k in self.entries if k.startswith("'")))
        for e in self.entries.values():
            out, edits = self.apply(e.output)
            if out != e.output:
                raise ValueError(f"layer 1 is not idempotent: {e.output!r} (from {e.key!r}) becomes {out!r}")
            # a lone capital letter in an output could join a spelled-out acronym on a second pass
            for written in (e.output, cased(e.output, "X")):
                if _LONE_CAPITAL.search(written):
                    raise ValueError(f"{e.table}.tsv: the output {written!r} (from {e.key!r}) has a single "
                                     "capital letter as a token")

    def search(self, text: str) -> str | None:
        """The first layer 1 form matched in text (whether or not it would change it), or None."""
        m = None if self._re is None else self._re.search(text)
        return None if m is None else m.group(0)

    def _open_quote(self, text: str, pos: int) -> bool:
        """Whether an unclosed single quote precedes pos (module docstring, quote guard)."""
        open_ = False
        j = text.find("'", 0, pos)
        while j != -1:
            before = text[j - 1].isalnum() if j > 0 else False
            after = text[j + 1].isalnum() if j + 1 < len(text) else False
            if after and not before:
                tail = text[j:j + 32].lower()
                if not any(tail.startswith(k) and not (len(tail) > len(k) and tail[len(k)].isalnum())
                           for k in self.leading):
                    open_ = True
            elif before and not after:
                open_ = False
            j = text.find("'", j + 1, pos)
        return open_

    def apply(self, text: str, protect: list[tuple[int, int]] = ()) -> tuple[str, list[Edit]]:
        """Layer 1 over text; matches overlapping a protected [start, end) are left as written. Edits are in the
        coordinates of `text` (start, end) and of the result (out_start, out_end)."""
        if self._re is None or not text:
            return text, []
        out, edits, pos, delta = [], [], 0, 0
        for m in self._re.finditer(text):
            rep = self._replacement(text, m, protect)
            if rep is None:
                continue
            s, e, found = m.start(), m.end(), m.group(0)
            entry = self.entries[found.lower()]
            out.append(text[pos:s])
            out.append(rep)
            edits.append(Edit(1, entry.table, entry.group, s, e, found, rep, s + delta, s + delta + len(rep)))
            delta += len(rep) - (e - s)
            pos = e
        if not edits:
            return text, []
        out.append(text[pos:])
        return "".join(out), edits

    def _replacement(self, text: str, m: re.Match, protect) -> str | None:
        """What apply writes for the table match m of text (None: left as written)."""
        s, e, found = m.start(), m.end(), m.group(0)
        if any(s < pe and ps < e for ps, pe in protect):
            return None
        entry = self.entries[found.lower()]
        if found.endswith("'") and self._open_quote(text, e - 1):
            return None
        if found.startswith("'") and text[e:e + 1] == "'":
            return None
        rep = cased(entry.output, found)
        return None if rep == found else rep

    def acronyms(self, text: str, protect: list[tuple[int, int]] = ()) -> tuple[str, list[Edit]]:
        """Spelled-out acronyms (cleaning version 2; module docstring), before apply; runs overlapping a protected
        [start, end) are left as written. Edits as in apply."""
        if not text:
            return text, []
        out, edits, pos, delta = [], [], 0, 0
        for m in _CHAIN.finditer(text):
            run = self._acronym(text, m, protect)
            if run is None:
                continue
            s, e, rep = run
            if s < pos or any(s < pe and ps < e for ps, pe in protect):
                continue
            out.append(text[pos:s])
            out.append(rep)
            edits.append(Edit(1, ACRONYM_TABLE, ACRONYM_GROUP, s, e, text[s:e], rep, s + delta, s + delta + len(rep),
                              text[s:e]))
            delta += len(rep) - (e - s)
            pos = e
        if not edits:
            return text, []
        out.append(text[pos:])
        return "".join(out), edits

    def _acronym(self, text: str, m: re.Match, protect) -> tuple[int, int, str] | None:
        """The run of the chain m as (start, end, compact form), or None (fewer than two letters, not a whole
        token, or excluded)."""
        letters = [m.start() + k for k, c in enumerate(m.group(0)) if "A" <= c <= "Z"]
        while len(letters) >= 2:
            e = letters[-1] + 1
            seps = [text[a + 1:b] for a, b in zip(letters, letters[1:])]
            dotted = any("." in sep for sep in seps)
            final = dotted and text.startswith(".", e)          # an undotted run's period is the sentence's
            q = e + final
            plural = _PLURAL.match(text, q)
            if plural and dotted and not final and plural.group(0) == "s":
                plural = None                                   # "the R. Is that": a word, not "R" + "I" + "s"
            if plural:
                q = plural.end()
            if _JOINED.match(text, q):
                if final or plural or not _JOINED.match(text, e):
                    return None                                 # the run goes on as another token ("A.P.I.x")
                letters.pop()                                   # the last capital starts a word ("A B Cat")
                continue
            word = self._next_word(text, q, protect)
            if (len(letters) == 2 and seps[0] in (".", " ")                 # "U.S.", "T V"
                    or all(text[p] == "I" for p in letters)                 # "I I think"
                    or word and word[0].isupper() and word != word.upper()):    # initials: "J. A. Mellon"
                return None
            compact = "".join(text[p] for p in letters) + ("s" if plural else "")
            return letters[0], q, compact + ("." if final and not plural and q == len(text) else "")
        return None

    def _next_word(self, text: str, q: int, protect) -> str | None:
        """The token (up to the next space) directly after one space at q, as the tables will write it, or None.
        Read the same way on a cleaned text, so the initials exclusion does not change on a second pass."""
        nm = _NEXT_WORD.match(text, q)
        if nm is None:
            return None
        out, edits = self.apply(text, protect)
        if not edits:
            return nm.group(1)
        return re.match(r"\S*", out[q + 1 + _shift(edits, q + 1):]).group(0)


class Tables:
    """The four committed tables, parsed and checked (module docstring)."""

    def __init__(self, texts: dict[str, str]):
        missing = [f for f in TABLE_FILES if f not in texts]
        if missing:
            raise RuntimeError(f"cleaning tables {missing} are missing")
        self.files = {f: _sha256(texts[f].encode("utf-8")) for f in TABLE_FILES}
        self.sha256 = _sha256(json.dumps(self.files, sort_keys=True).encode())
        entries = self._forms(texts["forms.tsv"]) + self._variants(texts["variants.tsv"]) + \
            self._dialect(texts["dialect.tsv"])
        self.layer1 = Layer1(entries)
        self.sets = self._choices(texts["choices.tsv"])
        self.by_key: dict[str, ChoiceSet] = {}
        member_set: dict[str, int] = {}
        for cs in self.sets:
            for m in cs.members:
                k = m.lower()
                if member_set.setdefault(k, cs.index) != cs.index:
                    raise ValueError(f"choices.tsv: {m!r} is in two sets")
            for k in cs.triggers:
                self.by_key[k] = cs
        self._span_re = _compile(self.by_key)
        members_re = _compile(member_set)
        for cs in self.sets:
            for m in cs.members:
                hit = self.layer1.search(m)
                if hit:
                    raise ValueError(f"layers are not disjoint: the layer 1 form {hit!r} matches in the member "
                                     f"{m!r} of {cs.name!r}")
        if members_re is not None:
            for k in self.layer1.entries:
                hit = members_re.search(k)
                if hit:
                    raise ValueError(f"layers are not disjoint: the layer 1 form {k!r} contains the member "
                                     f"{hit.group(0)!r}")

    @staticmethod
    def _forms(text: str) -> list[Entry]:
        out = []
        for n, (written, *rest) in _lines(text, "forms.tsv"):
            if len(rest) != 1:
                raise ValueError(f"forms.tsv:{n}: expected written<TAB>conventional")
            form = rest[0]
            for side in (written, form):
                _check_side(side, _PHRASE, f"forms.tsv:{n}")
            if _form_key(written) != _form_key(form):
                raise ValueError(f"forms.tsv:{n}: {written!r} -> {form!r} changes more than case and hyphens")
            out.append(Entry(written.lower(), form, "forms", form))
        merged: dict[str, Entry] = {}
        for e in out:
            if e.key in merged and merged[e.key].output != e.output:
                raise ValueError(f"forms.tsv: {e.key!r} has two conventional forms")
            merged[e.key] = e
        return list(merged.values())

    @staticmethod
    def _variants(text: str) -> list[Entry]:
        out = []
        for n, members in _lines(text, "variants.tsv"):
            if len(members) < 2:
                raise ValueError(f"variants.tsv:{n}: a group needs a canonical form and at least one variant")
            for m in members:
                _check_side(m, _PHRASE, f"variants.tsv:{n}")
            if len({m.lower() for m in members}) != len(members):
                raise ValueError(f"variants.tsv:{n}: members must differ by more than case")
            canonical = members[0]
            # the canonical spelling itself is enforced only when it has capitals ("ok" -> "OK"); a lowercase
            # canonical leaves its own casing as written ("Dataset", "DATASET")
            listed = members if canonical != canonical.lower() else members[1:]
            out += [Entry(m.lower(), canonical, "variants", canonical) for m in listed]
        return out

    @staticmethod
    def _dialect(text: str) -> list[Entry]:
        out = []
        for n, fields in _lines(text, "dialect.tsv"):
            if len(fields) != 2:
                raise ValueError(f"dialect.tsv:{n}: expected dialect<TAB>standard")
            dialect, standard = fields
            _check_side(dialect, _DIALECT, f"dialect.tsv:{n}")
            _check_side(standard, _STANDARD, f"dialect.tsv:{n}")
            if dialect.lower() == standard.lower():
                raise ValueError(f"dialect.tsv:{n}: {dialect!r} is its own standard spelling")
            out.append(Entry(dialect.lower(), standard, "dialect", dialect.lower()))
        return out

    @staticmethod
    def _choices(text: str) -> list[ChoiceSet]:
        sets = []
        for n, fields in _lines(text, "choices.tsv"):
            members, option_only = [], set()
            for f in fields:
                m = f[:-1] if f.endswith(OPTION_ONLY) else f
                _check_side(m, _PHRASE, f"choices.tsv:{n}")
                if f.endswith(OPTION_ONLY):
                    option_only.add(m)
                members.append(m)
            if len(members) < 2 or len(set(members)) != len(members):
                raise ValueError(f"choices.tsv:{n}: a set needs two or more distinct members")
            if len(members) + 1 > len(LETTERS):
                raise ValueError(f"choices.tsv:{n}: more members than option letters")
            cs = ChoiceSet(len(sets), tuple(members), frozenset(option_only))
            if not cs.triggers:
                raise ValueError(f"choices.tsv:{n}: every member is option-only, so the set never starts a span")
            sets.append(cs)
        return sets

    @classmethod
    def load(cls, tables_dir: Path | None = None) -> "Tables":
        base = TABLES_DIR if tables_dir is None else Path(tables_dir)
        texts = {}
        for f in TABLE_FILES:
            p = base / f
            if not p.exists():
                raise RuntimeError(f"cleaning table {p} is missing")
            texts[f] = p.read_text(encoding="utf-8")
        return cls(texts)

    def spans(self, text: str) -> list[Span]:
        if self._span_re is None or not text:
            return []
        return [Span(k, m.start(), m.end(), m.group(0), self.by_key[m.group(0).lower()])
                for k, m in enumerate(self._span_re.finditer(text))]


def _shift(edits: list[Edit], p: int) -> int:
    """The length change made by the edits (start, end in their input) that end at or before p."""
    return sum(len(e.replacement) - (e.end - e.start) for e in edits if e.end <= p)


def _compose(text: str, out: str, first: list[Edit], second: list[Edit], second_leads: bool = False) -> list[Edit]:
    """Two passes' edits (first: text -> mid, second: mid -> out) as edits from text to out. Edits that overlap in
    mid become one: an acronym edit if one of them is ("M A C O S" -> "MACOS" -> "macOS" is one acronym edit, with the
    acronym's run), else of the first pass's table and group (second_leads: of the second pass's, if one of them is
    from it: layer 0 then layer 1), with the lead edit's layer."""
    if not first:
        return second
    items = sorted([(e.out_start, e.out_end, 0, e) for e in first] + [(e.start, e.end, 1, e) for e in second],
                   key=lambda x: (x[0], x[1], x[2]))
    clusters: list[list] = []
    for a, b, k, e in items:
        if clusters and a < clusters[-1][1]:
            clusters[-1][1] = max(clusters[-1][1], b)
            clusters[-1][2].append((k, e))
        else:
            clusters.append([a, b, [(k, e)]])
    res = []
    for a, b, members in clusters:
        # mid -> text: undo the first pass's edits before; mid -> out: apply the second pass's edits before
        s = a - sum(len(e.replacement) - (e.end - e.start) for e in first if e.out_end <= a)
        t = b - sum(len(e.replacement) - (e.end - e.start) for e in first if e.out_end <= b)
        o, p = a + _shift(second, a), b + _shift(second, b)
        members.sort(key=lambda m: m[0])
        lead = next((e for k, e in members if e.table == ACRONYM_TABLE), None)
        if lead is None and second_leads:
            lead = next((e for k, e in members if k == 1), None)
        lead = lead or members[0][1]
        res.append(Edit(lead.layer, lead.table, lead.group, s, t, text[s:t], out[o:p], o, p, lead.run))
    return res


# ----------------------------------------------------------------------------- layer 0 (version 2)

@dataclass
class _Word:
    key: str
    core: str
    core_start: int
    core_end: int
    trail: str


def _words(text: str) -> list[_Word]:
    """The word tokens of text (module docstring, "Layer 0"): pure-punctuation tokens join the trail of the word
    before them."""
    out: list[_Word] = []
    for m in _TOKEN.finditer(text):
        lead, core, trail = _PARTS.match(m.group(0)).groups()
        if not core:
            if out:
                out[-1].trail += m.group(0)
            continue
        key = _NONWORD.sub("", core.lower())
        if key:
            s = m.start() + len(lead)
            out.append(_Word(key, core, s, s + len(core), trail))
    return out


def _mark(trail: str) -> str:
    for ch in reversed(trail):
        if ch in PUNCT_MARKS:
            return ch
    return ""


_LEV: dict = {}


def _numba_kernels():
    """The alignment kernels (numba, compiled once per process)."""
    if not _LEV:
        import numba
        import numpy as np

        @numba.njit(cache=True)
        def lev(a, b):
            n, m = a.shape[0], b.shape[0]
            if n == 0:
                return m
            if m == 0:
                return n
            prev = np.arange(m + 1)
            cur = np.empty(m + 1, dtype=prev.dtype)
            for i in range(1, n + 1):
                cur[0] = i
                for j in range(1, m + 1):
                    c = prev[j - 1] + (0 if a[i - 1] == b[j - 1] else 1)
                    d = prev[j] + 1
                    if d < c:
                        c = d
                    d = cur[j - 1] + 1
                    if d < c:
                        c = d
                    cur[j] = c
                prev, cur = cur, prev
            return prev[m]

        @numba.njit(cache=True)
        def align(a, b):
            n, m = a.shape[0], b.shape[0]
            cost = np.zeros((n + 1, m + 1), dtype=np.int32)
            for i in range(n + 1):
                cost[i, 0] = i
            for j in range(m + 1):
                cost[0, j] = j
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    c = cost[i - 1, j - 1] + (0 if a[i - 1] == b[j - 1] else 1)
                    d = cost[i - 1, j] + 1
                    if d < c:
                        c = d
                    d = cost[i, j - 1] + 1
                    if d < c:
                        c = d
                    cost[i, j] = c
            out = np.empty((min(n, m), 2), dtype=np.int64)
            k = 0
            i, j = n, m
            while i > 0 or j > 0:
                if i > 0 and j > 0 and cost[i, j] == cost[i - 1, j - 1] + (0 if a[i - 1] == b[j - 1] else 1):
                    if a[i - 1] == b[j - 1]:
                        out[k, 0] = i - 1
                        out[k, 1] = j - 1
                        k += 1
                    i -= 1
                    j -= 1
                elif i > 0 and cost[i, j] == cost[i - 1, j] + 1:
                    i -= 1
                else:
                    j -= 1
            return out[:k][::-1]

        _LEV.update(lev=lev, align=align, np=np)
    return _LEV


def _codes(a: list[str], b: list[str]):
    np = _numba_kernels()["np"]
    vocab: dict[str, int] = {}
    return (np.array([vocab.setdefault(w, len(vocab)) for w in a], dtype=np.int64),
            np.array([vocab.setdefault(w, len(vocab)) for w in b], dtype=np.int64))


def key_errors(a: list[str], b: list[str]) -> int:
    """The word edit distance (substitutions, deletions, insertions) of key sequences a and b."""
    if a == b:
        return 0
    if not a or not b:
        return len(a) + len(b)
    return int(_numba_kernels()["lev"](*_codes(a, b)))


def key_matches(a: list[str], b: list[str]) -> list[tuple[int, int]]:
    """The matched pairs (i, j), a[i] == b[j], of the minimum-edit alignment with the fixed tie-break (module
    docstring, "Layer 0")."""
    if a == b:
        return [(k, k) for k in range(len(a))]
    if not a or not b:
        return []
    return [(int(i), int(j)) for i, j in _numba_kernels()["align"](*_codes(a, b))]


def punctuation_transfer(text: str, donor: str, protect=()) -> tuple[str, list[Edit]]:
    """Layer 0 (module docstring): T*'s text with B0's marks transferred from donor, and its edits (start, end in
    text; out_start, out_end in the result). `protect`: layer 2's spans [start, end) in text, never edited inside."""
    tw = _words(text)
    bw = _words(donor) if donor else []
    if not tw or not bw:
        return text, []
    a, b = [w.key for w in tw], [w.key for w in bw]
    if key_errors(a, b) / len(a) > PUNCT_GUARD:
        return text, []
    pairs = key_matches(a, b)
    nxt = dict(pairs)

    def inside(p: int, cap: bool) -> bool:
        return any((ps <= p < pe) if cap else (ps < p < pe) for ps, pe in protect)

    ops: list[tuple[int, str, str]] = []                  # (position in text, original, replacement)
    for i, j in pairs:
        t, w = tw[i], bw[j]
        mark = _mark(w.trail)
        if _mark(t.trail) or not mark or "." in w.core:
            continue
        if not all(ch in PUNCT_MARKS or ch in PUNCT_CLOSERS for ch in t.trail):
            continue
        if not (nxt.get(i + 1) == j + 1 or (i == len(tw) - 1 and j == len(bw) - 1)):
            continue
        if inside(t.core_end, False):
            continue
        ops.append((t.core_end, "", mark))
        if mark in PUNCT_ENDS and nxt.get(i + 1) == j + 1:
            b_next, p = bw[j + 1].core[0], tw[i + 1].core_start
            if b_next.isalpha() and b_next.isupper() and text[p].isalpha() and text[p].islower() \
                    and not inside(p, True):
                ops.append((p, text[p], text[p].upper()))
    if not ops:
        return text, []
    ops.sort(key=lambda x: (x[0], len(x[1])))
    pieces, edits, pos, delta = [], [], 0, 0
    for p, original, rep in ops:
        pieces.append(text[pos:p])
        pieces.append(rep)
        o = p + delta
        edits.append(Edit(0, PUNCT_TABLE, "mark" if not original else "capital", p, p + len(original), original, rep,
                          o, o + len(rep)))
        delta += len(rep) - len(original)
        pos = p + len(original)
    pieces.append(text[pos:])
    return "".join(pieces), edits


def marked(text: str, start: int, end: int) -> str:
    return text[:start] + MARK_OPEN + text[start:end] + MARK_CLOSE + text[end:]


def prompt_user(text: str, start: int, end: int, options: tuple[str, ...]) -> str:
    """The user message of one layer 2 prompt (the system message is SYSTEM)."""
    listed = "\n".join(f"{LETTERS[k]}. {o}" for k, o in enumerate(options))
    return USER.format(open=MARK_OPEN, close=MARK_CLOSE, cant_tell=CANT_TELL, marked=marked(text, start, end),
                       options=listed)


def prompt_key(text: str, start: int, end: int, options: tuple[str, ...]) -> str:
    """The dedup key: hash of (template, label text, span, the span's options)."""
    return _sha256(json.dumps([TEMPLATE_SHA256, text, start, end, list(options)]).encode())


def decide(probs: list[float], options: tuple[str, ...], threshold: float = THRESHOLD) -> str | None:
    """The chosen spelling: the most probable option if it is not "can't tell" and has probability >= threshold
    (probs renormalized over the options, "can't tell" included); else None (left as written)."""
    if len(probs) != len(options) or options[-1] != CANT_TELL:
        raise ValueError("probabilities do not match the options")
    total = float(sum(probs))
    if not total > 0:
        return None
    best = max(range(len(probs)), key=lambda k: (probs[k], -k))
    if best == len(options) - 1 or probs[best] / total < threshold:
        return None
    return options[best]


class Choices:
    """A verified layer 2 choices artifact: {(label set, id): [(span, start, end, original, set name, chosen)]}."""

    def __init__(self, directory: Path, manifest: dict, rows: dict):
        self.dir, self.manifest, self.rows = Path(directory), manifest, rows
        self.file_sha256 = manifest["file_sha256"]

    @classmethod
    def load(cls, directory: Path) -> "Choices":
        import pyarrow.parquet as pq
        directory = Path(directory)
        mpath, ppath = directory / "manifest.json", directory / "choices.parquet"
        if not mpath.exists() or not ppath.exists():
            raise ArtifactMismatch(f"{directory}: no choices artifact (manifest.json and choices.parquet)")
        manifest = json.loads(mpath.read_text())
        if _sha256(ppath.read_bytes()) != manifest.get("file_sha256"):
            raise ArtifactMismatch(f"{ppath}: SHA-256 differs from its manifest")
        t = pq.read_table(ppath, columns=["label_set", "id", "span", "start", "end", "original", "set",
                                          "chosen"]).to_pydict()
        if len(t["id"]) != manifest.get("rows"):
            raise ArtifactMismatch(f"{ppath}: {len(t['id'])} rows, the manifest says {manifest.get('rows')}")
        rows: dict = {}
        for ls, i, k, s, e, o, name, c in zip(t["label_set"], t["id"], t["span"], t["start"], t["end"],
                                              t["original"], t["set"], t["chosen"]):
            rows.setdefault((ls, i), []).append((k, s, e, o, name, c))
        for v in rows.values():
            v.sort()
            if [r[0] for r in v] != list(range(len(v))):
                raise ArtifactMismatch("choices.parquet: a label's span indices are not 0..n-1")
        return cls(directory, manifest, rows)


class Cleaning:
    """A loaded cleaning spec: the tables and (optionally) one label run's choices artifact."""

    def __init__(self, tables: Tables, choices: Choices | None = None, version: int = VERSION):
        if version not in VERSIONS:
            raise ValueError(f"cleaning version {version!r} is not one of {VERSIONS}")
        self.tables, self.choices, self.version = tables, choices, version
        if choices is not None:
            m = choices.manifest
            for key, want in (("version", version), ("tables_sha256", tables.sha256),
                              ("template_sha256", TEMPLATE_SHA256), ("threshold", THRESHOLD)):
                if m.get(key) != want:
                    raise ArtifactMismatch(f"choices artifact {choices.dir}: {key} {m.get(key)!r}, these tables and "
                                           f"this code have {want!r}")
        self._sets = {cs.name: cs for cs in tables.sets}

    @classmethod
    def load(cls, tables_dir: Path | None = None, choices_dir: Path | None = None,
             version: int = VERSION) -> "Cleaning":
        tables = Tables.load(tables_dir)
        return cls(tables, None if choices_dir is None else Choices.load(choices_dir), version)

    def identity(self) -> dict:
        out = {"version": self.version, "tables_sha256": self.tables.sha256,
               "choices_sha256": None if self.choices is None else self.choices.file_sha256}
        if self.version >= 2:
            out["layer0"] = dict(LAYER0)
        return out

    def needs_donor(self, label_set: str) -> bool:
        """Whether clean/edits of this label set read the id's B0 label (layer 0, version 2, T* only)."""
        return self.version >= 2 and label_set == PUNCT_LABEL_SET

    def bind_run(self, contract_sha256: str) -> None:
        """Refuse a choices artifact of another label run contract."""
        if self.choices is not None and self.choices.manifest.get("contract_sha256") != contract_sha256:
            raise ArtifactMismatch(f"choices artifact {self.choices.dir} is of label run contract "
                                   f"{self.choices.manifest.get('contract_sha256')}, not {contract_sha256}")

    def spans(self, text: str) -> list[Span]:
        return self.tables.spans(text)

    def _recorded(self, spans: list[Span], text: str, item_id: str, label_set: str) -> list[str | None]:
        if self.choices is None:
            return [None] * len(spans)
        rows = self.choices.rows.get((label_set, item_id), [])
        if len(rows) != len(spans):
            raise ArtifactMismatch(f"{label_set} {item_id}: {len(spans)} spans, the choices artifact records "
                                   f"{len(rows)}")
        out = []
        for sp, (k, s, e, original, name, chosen) in zip(spans, rows):
            if text[s:e] != original:
                raise ArtifactMismatch(f"{label_set} {item_id}: the label at [{s}, {e}) is {text[s:e]!r}, the "
                                       f"artifact records {original!r}")
            if (s, e, name) != (sp.start, sp.end, sp.set.name):
                raise ArtifactMismatch(f"{label_set} {item_id}: span {k} is [{sp.start}, {sp.end}) of "
                                       f"{sp.set.name!r}, the artifact records [{s}, {e}) of {name!r}")
            if chosen is not None and chosen not in sp.set.members:
                raise ArtifactMismatch(f"{label_set} {item_id}: {chosen!r} is not a member of {name!r}")
            out.append(chosen)
        return out

    def edits(self, text: str, item_id: str, label_set: str, donor: str | None = None) -> tuple[str, list[Edit]]:
        """The cleaned text and its changed spans (module docstring, read-time order); donor: the id's B0 label as
        stored, read by layer 0 (needs_donor); None reads as no B0 label (layer 0 does nothing). The training path
        (training/labels.py LabelStore) and the runner always pass the record's b0_raw."""
        if text == "":
            self._recorded([], text, item_id, label_set)      # no spans: the artifact must record none either
            return "", []
        spans = self.spans(text)
        chosen = self._recorded(spans, text, item_id, label_set)
        pieces, protect, l2, pos, delta = [], [], [], 0, 0
        for sp, c in zip(spans, chosen):
            rep = sp.original if c is None else cased(c, sp.original)
            pieces.append(text[pos:sp.start])
            o = sp.start + delta
            pieces.append(rep)
            protect.append((o, o + len(rep)))
            if rep != sp.original:
                l2.append(Edit(2, "choices", sp.set.name, sp.start, sp.end, sp.original, rep, o, o + len(rep)))
            delta += len(rep) - (sp.end - sp.start)
            pos = sp.end
        pieces.append(text[pos:])
        mid = "".join(pieces) if l2 else text
        # outside the spans, the text is byte-identical
        kept_in, kept_out, p_in, p_out = [], [], 0, 0
        for sp, (ps, pe) in zip(spans, protect):
            kept_in.append(text[p_in:sp.start])
            kept_out.append(mid[p_out:ps])
            p_in, p_out = sp.end, pe
        kept_in.append(text[p_in:])
        kept_out.append(mid[p_out:])
        if kept_in != kept_out:
            raise RuntimeError(f"{label_set} {item_id}: layer 2 changed text outside its spans")
        layer1 = self.tables.layer1
        l0 = []
        if self.needs_donor(label_set):
            mid0, l0 = punctuation_transfer(mid, donor or "", protect)
            protect0 = [(ps + _shift(l0, ps), pe + _shift(l0, ps)) for ps, pe in protect]
        else:
            mid0, protect0 = mid, protect
        if self.version >= 2:
            # spelled-out acronyms, then the tables on their result (the protected spans shifted accordingly), until
            # nothing changes: a table edit can split a chain left as written and expose a run ("P T S D. M R Is")
            cur, prot, l1 = mid0, list(protect0), []
            for k in range(LAYER1_ROUNDS):
                nxt, acr = layer1.acronyms(cur, prot)
                if k and not acr:
                    break
                prot = [(ps + _shift(acr, ps), pe + _shift(acr, ps)) for ps, pe in prot]
                out, tab = layer1.apply(nxt, prot)
                prot = [(ps + _shift(tab, ps), pe + _shift(tab, ps)) for ps, pe in prot]
                l1 = _compose(mid0, out, l1, _compose(cur, out, acr, tab))
                cur = out
            else:
                raise RuntimeError(f"{label_set} {item_id}: layer 1 did not settle in {LAYER1_ROUNDS} rounds")
        else:
            out, l1 = layer1.apply(mid, protect)
        if l0:
            l1 = _compose(mid, out, l0, l1, second_leads=True)      # layers 0 and 1, from the layer 2 result
        edits = self._merge(l2, l1, protect, spans)
        if out == "":
            raise RuntimeError(f"{label_set} {item_id}: cleaning made a non-empty label empty")
        return out, edits

    @staticmethod
    def _merge(l2: list[Edit], l1: list[Edit], protect, spans) -> list[Edit]:
        """Every edit in original coordinates (start, end) and final coordinates (out_start, out_end)."""
        if not l1:
            return l2
        # layer 1 positions (in the layer 2 result) back to the original text: subtract the shifts of the
        # layer 2 spans before them (layer 1 never overlaps a span)
        shifts = [(pe, (pe - ps) - (sp.end - sp.start)) for (ps, pe), sp in zip(protect, spans)]
        out = []
        for e in l1:
            back = sum(d for pe, d in shifts if pe <= e.start)
            out.append(Edit(e.layer, e.table, e.group, e.start - back, e.end - back, e.original, e.replacement,
                            run=e.run))
        allx = sorted(l2 + out, key=lambda x: x.start)
        final, delta = [], 0
        for e in allx:
            o = e.start + delta
            final.append(Edit(e.layer, e.table, e.group, e.start, e.end, e.original, e.replacement, o,
                              o + len(e.replacement), e.run))
            delta += len(e.replacement) - (e.end - e.start)
        return final

    def clean(self, text: str, item_id: str, label_set: str, donor: str | None = None) -> str:
        return self.edits(text, item_id, label_set, donor)[0]


def production_dir(label_run: str, version: int = VERSION) -> Path:
    """$RELABEL_ROOT/cleaning/<label run>/v<version> (the choices artifact and its work files)."""
    from labeling import layout
    return layout.relabel_root() / "cleaning" / label_run / f"v{version}"


def audit_dir(version: int = VERSION) -> Path:
    """The architect's sheets (never committed): $RELABEL_ROOT/cleaning/audit for version 1,
    $RELABEL_ROOT/cleaning/audit/v<version> for later versions."""
    from labeling import layout
    base = layout.relabel_root() / "cleaning" / "audit"
    return base if version == 1 else base / f"v{version}"


def production(label_run: str, version: int = 1) -> Cleaning:
    """The committed tables with the label run's choices artifact of `version`. The default, version 1, is what
    stages 2b and 2c read (train_decoder.label_cleaning); stage 3 and later pass version=2 (cleaning plan v5)."""
    return Cleaning.load(choices_dir=production_dir(label_run, version), version=version)
