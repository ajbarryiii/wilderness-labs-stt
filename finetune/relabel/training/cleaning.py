"""Label cleaning (stage 2a, cleaning plan): a versioned, deterministic read-time layer over the training labels.

    spec = Cleaning.load(choices_dir=...)          committed tables + the layer 2 choices artifact of one label run
    spec.clean(text, item_id, label_set) -> str    the cleaned training text of one kept label
    spec.edits(text, item_id, label_set)           (cleaned text, [Edit]) with every changed span
    spec.spans(text) -> [Span]                     layer 2 spans of a label (what the choices artifact decides)
    spec.identity() -> {version, tables_sha256, choices_sha256}

training/labels.py LabelStore.open(..., cleaning=spec) applies clean() to every kept label of either label set;
cleaning=None reproduces the uncleaned texts and identity byte for byte.

Tables (terms/cleaning/, committed; UTF-8; blank lines and lines starting with "#" are ignored; fields are separated
by one TAB; every written form is ASCII):

    forms.tsv     written<TAB>conventional             casing and hyphenation only: the two sides may differ only in
                                                       letter case and a hyphen versus a space (checked), so the
                                                       scorer's normalized words never change. No diacritic folding.
    variants.tsv  canonical<TAB>variant[<TAB>...]      a pure spelling-variant group (the same word in every
                                                       context): every variant becomes the canonical (first)
                                                       form. Members differ by more than case. A canonical form
                                                       with capitals is also enforced on its own spelling ("ok"
                                                       -> "OK"); an all-lowercase one leaves its own casing as
                                                       written ("Dataset").
    dialect.tsv   dialect<TAB>standard[<TAB>lower]     a dialect or reduced spelling of a common word becomes its
                                                       standard spelling. "lower": the dialect form matches only as
                                                       written in lowercase (the g-dropped allowlist: "nothin'" but
                                                       never the name "Dunkin'"). A dialect form may begin or end with
                                                       an apostrophe ('cause, 'em, nothin').
    choices.tsv   member<TAB>member[<TAB>...]          one layer 2 set: spellings of the same spoken word or words,
                                                       offered to the model in this order. A member ending in "*" is
                                                       option-only: offered, but it does not start a span (the
                                                       standard spelling of a dialect set: "get*", "you*").

Matching (both layers): a written form matches case-insensitively (except "lower" dialect forms) on whole tokens. A
token is a run of letters and digits (any script) joined by internal apostrophes or periods ("y'all", "don't",
"index.html", "config.xml" are one token each), so a form never matches part of a token ("ya" is not found in
"ya'll", "git" not in "Git's", "html" not in "index.html"); an edge apostrophe belongs to a form only where the form
lists it. Spaces and hyphens inside a form match literally. Scanning is left to right; at each position the longest
form wins; matches never overlap.

Quote guard (dialect forms with an edge apostrophe): a form ending in an apostrophe is not applied when an unclosed
single quote precedes it (the apostrophe is then the closing quote: "'goin'" and "said 'I was goin'" stay as written),
and a form beginning with an apostrophe is not applied when the token is followed by an apostrophe ("'em'"), so a
quote mark is never eaten. An opening single quote is an apostrophe before a token that is not the start of a listed
leading-apostrophe form; the next token-final apostrophe closes it.

Capitalization (every replacement, both layers): when the first letter of the original span is a capital and the
replacement is all lowercase, the replacement's first letter is capitalized ("Gonna" -> "Going to", "Maybe" ->
"May be", "Double check" -> "Double-check"); a replacement with any capital letter is written as listed ("GAAP",
"Git", "OK", "I'm going to").

Load-time checks: every side has the token syntax above; written forms are unique (case-insensitively) across the
three layer 1 tables; layer 1 is idempotent (each output is left unchanged by layer 1); a layer 2 set has two or
more distinct members and at least one that starts spans; no member key is shared by two sets; the layers are
disjoint: no layer 1 form matches inside a member, and no member matches inside a layer 1 form.

Read-time order (clean): the empty text stays empty. Layer 2: the spans of the original text (members that start
spans, matched as above) take the recorded choice of the artifact, if one was chosen (else they are left as
written); the replacement must be a member of the span's set, and the text outside the spans is verified
byte-identical. Then layer 1 on the result, which never touches a layer 2 span. A non-empty text never becomes empty.
Without a choices artifact (choices_dir None) layer 2 leaves every span as written (layer 1 only; used by checks).

The choices artifact ($RELABEL_ROOT/cleaning/<label run>/v<VERSION>/, written by training/cleaning_run.py):
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

VERSION = 1
TABLES_DIR = committed_terms.HERE / "cleaning"
TABLE_FILES = ("forms.tsv", "variants.tsv", "dialect.tsv", "choices.tsv")
THRESHOLD = 0.9
CANT_TELL = "can't tell"
LETTERS = "ABCDEFGHIJ"
MARK_OPEN, MARK_CLOSE = "[[", "]]"
OPTION_ONLY = "*"
LOWER_FLAG = "lower"
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


def _compile(keys: dict[str, bool]) -> re.Pattern | None:
    """Whole-token alternation of keys {lowercase key: lowercase-only}, longest first."""
    if not keys:
        return None
    alts = []
    for k in sorted(keys, key=lambda w: (-len(w), w)):
        alts.append(f"(?-i:{re.escape(k)})" if keys[k] else re.escape(k))
    return re.compile(_LEFT + "(?:" + "|".join(alts) + ")" + _RIGHT, re.IGNORECASE)


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
    lower_only: bool = False


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
        self._re = _compile({k: e.lower_only for k, e in self.entries.items()})
        self.leading = tuple(sorted(k for k in self.entries if k.startswith("'")))
        for e in self.entries.values():
            out, edits = self.apply(e.output)
            if out != e.output:
                raise ValueError(f"layer 1 is not idempotent: {e.output!r} (from {e.key!r}) becomes {out!r}")

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
            s, e, found = m.start(), m.end(), m.group(0)
            if any(s < pe and ps < e for ps, pe in protect):
                continue
            entry = self.entries[found.lower()]
            if found.endswith("'") and self._open_quote(text, e - 1):
                continue
            if found.startswith("'") and text[e:e + 1] == "'":
                continue
            rep = cased(entry.output, found)
            if rep == found:
                continue
            out.append(text[pos:s])
            out.append(rep)
            edits.append(Edit(1, entry.table, entry.group, s, e, found, rep, s + delta, s + delta + len(rep)))
            delta += len(rep) - (e - s)
            pos = e
        if not edits:
            return text, []
        out.append(text[pos:])
        return "".join(out), edits


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
        self._span_re = _compile({k: False for k in self.by_key})
        members_re = _compile({k: False for k in member_set})
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
            if len(fields) not in (2, 3) or (len(fields) == 3 and fields[2] != LOWER_FLAG):
                raise ValueError(f"dialect.tsv:{n}: expected dialect<TAB>standard[<TAB>{LOWER_FLAG}]")
            dialect, standard = fields[:2]
            lower = len(fields) == 3
            _check_side(dialect, _DIALECT, f"dialect.tsv:{n}")
            _check_side(standard, _STANDARD, f"dialect.tsv:{n}")
            if lower and dialect != dialect.lower():
                raise ValueError(f"dialect.tsv:{n}: a lowercase-only form must be written in lowercase")
            if dialect.lower() == standard.lower():
                raise ValueError(f"dialect.tsv:{n}: {dialect!r} is its own standard spelling")
            out.append(Entry(dialect.lower(), standard, "dialect", dialect.lower(), lower))
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

    def __init__(self, tables: Tables, choices: Choices | None = None):
        self.tables, self.choices = tables, choices
        if choices is not None:
            m = choices.manifest
            for key, want in (("version", VERSION), ("tables_sha256", tables.sha256),
                              ("template_sha256", TEMPLATE_SHA256), ("threshold", THRESHOLD)):
                if m.get(key) != want:
                    raise ArtifactMismatch(f"choices artifact {choices.dir}: {key} {m.get(key)!r}, these tables and "
                                           f"this code have {want!r}")
        self._sets = {cs.name: cs for cs in tables.sets}

    @classmethod
    def load(cls, tables_dir: Path | None = None, choices_dir: Path | None = None) -> "Cleaning":
        tables = Tables.load(tables_dir)
        return cls(tables, None if choices_dir is None else Choices.load(choices_dir))

    def identity(self) -> dict:
        return {"version": VERSION, "tables_sha256": self.tables.sha256,
                "choices_sha256": None if self.choices is None else self.choices.file_sha256}

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

    def edits(self, text: str, item_id: str, label_set: str) -> tuple[str, list[Edit]]:
        """The cleaned text and its changed spans (module docstring, read-time order)."""
        if text == "":
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
        out, l1 = self.tables.layer1.apply(mid, protect)
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
            out.append(Edit(1, e.table, e.group, e.start - back, e.end - back, e.original, e.replacement))
        allx = sorted(l2 + out, key=lambda x: x.start)
        final, delta = [], 0
        for e in allx:
            o = e.start + delta
            final.append(Edit(e.layer, e.table, e.group, e.start, e.end, e.original, e.replacement, o,
                              o + len(e.replacement)))
            delta += len(e.replacement) - (e.end - e.start)
        return final

    def clean(self, text: str, item_id: str, label_set: str) -> str:
        return self.edits(text, item_id, label_set)[0]


def production_dir(label_run: str) -> Path:
    """$RELABEL_ROOT/cleaning/<label run>/v<VERSION> (the choices artifact and its work files)."""
    from labeling import layout
    return layout.relabel_root() / "cleaning" / label_run / f"v{VERSION}"


def audit_dir() -> Path:
    """$RELABEL_ROOT/cleaning/audit (architect's sheets; never committed)."""
    from labeling import layout
    return layout.relabel_root() / "cleaning" / "audit"


def production(label_run: str) -> Cleaning:
    """The committed tables with the label run's choices artifact (what training reads by default)."""
    return Cleaning.load(choices_dir=production_dir(label_run))
