"""Style-neutral WER and formatted (case and punctuation) metrics (b0prime plan v4 "Reporting"), computed from scored
per-utterance records (parakeet-ternary evaluate.py's layout: id, ref, hyp, scored). CPU only.

Style-neutral WER (variants): canonicalizers applied identically to the reference and the hypothesis, then the
standard normalizer (evaluate.normalize) and the edit distance:
    std  the standard normalizer only (the gate's WER)
    acr  acronym runs (cleaning plan v7 "Scoring changes" 2, replacing stage 3's counting rule): after the normalizer,
         identically on both sides, every maximal run of 2 or more letter tokens is joined into one token. A letter
         token is a single letter a-z, or the normalizer's "0" / "00" (a spoken "O" / "O O") inside a run that has a
         letter, read as "o" / "oo". A run made only of "i" (a stutter) is not joined. So a spelled run counts as one
         reference word and a wrong letter as one error per run
    cmp  compounds: the bigrams of terms/style/compounds.tsv (hand-filtered) joined on normalized tokens
    dis  disfluencies: "uh-huh" and marked false starts ("the- ") removed before the normalizer (bound prefixes kept);
         extra fillers removed and immediate repeats of 1-3 word n-grams collapsed after it
    unit dose units (cleaning plan v7): after a number token (terms.number_token: digits, or the normalizer's "one"
         for a lone "1", v7.2), a unit form of terms/cleaning/units.tsv
         and its abbreviation become one token (the abbreviation, normalized), and "per" between two listed units is
         dropped: "500 milligrams per kilogram" and "500 mg/kg" both give "500 mg kg"
    all  acr, cmp, dis and unit together
A group's contribution is WER(std) - WER(group). Every canonicalizer reads one side only, so the canonical reference
words do not depend on the hypothesis. utterance_errors gives every variant's (errors, reference words) per utterance,
for paired bootstraps.

Formatted metrics (formatted_counts per utterance, formatted_summary over a set): tokens are words (case kept,
inner apostrophes kept, numbers with inner . , : whole) and clause marks . , ? ! ; : (a run of marks is its first).
Words are aligned case-insensitively (cleaning.key_matches); over aligned words: the case-error rate (a different
spelling of case), and each word's following mark: any-mark precision, recall and F1, and per mark (. , ?) the same.
Also the case- and punctuation-sensitive WER (marks as tokens, case kept) and the case-sensitive WER without marks.
"""
from __future__ import annotations

import collections
import re
from pathlib import Path

VARIANTS = {"std": (), "acr": ("acr",), "cmp": ("cmp",), "dis": ("dis",), "unit": ("unit",),
            "all": ("acr", "cmp", "dis", "unit")}
COMPOUNDS_FILE = Path(__file__).resolve().parent / "terms" / "style" / "compounds.tsv"
UNITS_FILE = Path(__file__).resolve().parent / "terms" / "cleaning" / "units.tsv"
EXTRA_FILLERS = {"er", "erm", "ah", "ahh", "ahm", "eh", "hm", "uhm", "umm", "uhh", "uhhh", "ummm", "mhmm", "mmhmm"}
PREFIXES = {"non", "pre", "post", "anti", "re", "co", "sub", "multi", "self", "semi", "inter", "intra", "over",
            "under", "mid", "ex", "pro", "de", "bi", "tri", "mini", "micro", "macro", "mono", "poly", "cross"}
MARKS = ".,?!;:"
PER_MARK = ".,?"
_UH_HUH = re.compile(r"\buh-huh\b", re.IGNORECASE)
_FALSE_START = re.compile(r"(?<![\w'-])([a-z']+|[A-Z][a-z']*)-(?=\s|$)")
_FMT = re.compile(r"\d+(?:[.,:]\d+)+|\w+(?:['’]\w+)*|[.,?!;:]+")
_COMPOUNDS: dict = {}
_UNITS: dict = {}


def compounds(path: Path = COMPOUNDS_FILE) -> dict[tuple[str, str], str]:
    """The compound table: {(first, second): joined} (comment lines start with "#")."""
    key = str(path)
    if key not in _COMPOUNDS:
        out = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                a, b = line.split("\t")
                out[(a, b)] = a + b
        _COMPOUNDS[key] = out
    return _COMPOUNDS[key]


def errors(a: list[str], b: list[str]) -> int:
    from training import cleaning
    return cleaning.key_errors(a, b)


def normalize(text: str) -> list[str]:
    import evaluate  # finetune/parakeet-ternary
    return evaluate.normalize(text).split()


def _false_start(m: re.Match) -> str:
    return m.group(0) if m.group(1).lower() in PREFIXES else ""


def pre(text: str, groups) -> str:
    if "dis" in groups:
        text = _FALSE_START.sub(_false_start, _UH_HUH.sub(" ", text))
    return text


def _letter(tok: str) -> bool:
    return len(tok) == 1 and "a" <= tok <= "z"


def _o(tok: str) -> bool:
    return tok in ("0", "00")                                  # the normalizer writes a spoken "O" as "0"


def join_letter_runs(tokens: list[str]) -> list[str]:
    """acr (module docstring): every maximal run of 2 or more letter tokens joined into one token."""
    out, i, n = [], 0, len(tokens)
    while i < n:
        if not (_letter(tokens[i]) or _o(tokens[i])):
            out.append(tokens[i])
            i += 1
            continue
        j = i
        while j < n and (_letter(tokens[j]) or _o(tokens[j])):
            j += 1
        run = tokens[i:j]
        if len(run) >= 2 and any(_letter(t) for t in run) and any(t != "i" for t in run):
            out.append("".join("o" * len(t) if _o(t) else t for t in run))
        else:
            out.extend(run)
        i = j
    return out


def units() -> dict[tuple[str, ...], str]:
    """{normalized unit form: normalized abbreviation} of terms/cleaning/units.tsv."""
    if not _UNITS:
        for line in UNITS_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                written, abbr = (f.strip() for f in line.split("\t"))
                _UNITS[tuple(normalize(written))] = " ".join(normalize(abbr))
    return _UNITS


def _unit_at(tokens: list[str], i: int) -> tuple[int, str] | None:
    table = units()
    for length in sorted({len(k) for k in table}, reverse=True):
        key = tuple(tokens[i:i + length])
        if len(key) == length and key in table:
            return length, table[key]
    return None


def join_units(tokens: list[str]) -> list[str]:
    """unit (module docstring): after a number token, a unit form becomes its abbreviation; "per" between two listed
    units is dropped."""
    from terms import number_token
    out, i, n = [], 0, len(tokens)
    while i < n:
        out.append(tokens[i])
        i += 1
        if not number_token(out[-1]) or i >= n:
            continue
        hit = _unit_at(tokens, i)
        if hit is None:
            continue
        out.append(hit[1])
        i += hit[0]
        if i < n and tokens[i] == "per":
            second = _unit_at(tokens, i + 1)
            if second is not None:
                out.append(second[1])
                i += 1 + second[0]
    return out


def collapse_repetitions(tokens: list[str]) -> list[str]:
    changed = True
    while changed:
        changed = False
        out, i, n = [], 0, len(tokens)
        while i < n:
            hit = next((k for k in (3, 2, 1) if i + 2 * k <= n and tokens[i:i + k] == tokens[i + k:i + 2 * k]), 0)
            if hit:
                out.extend(tokens[i:i + hit])
                i += 2 * hit
                changed = True
            else:
                out.append(tokens[i])
                i += 1
        tokens = out
    return tokens


def _compound(tokens: list[str]) -> list[str]:
    table = compounds()
    out, i, n = [], 0, len(tokens)
    while i < n:
        if i + 1 < n and (tokens[i], tokens[i + 1]) in table:
            out.append(table[(tokens[i], tokens[i + 1])])
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def post(ref: list[str], hyp: list[str], groups) -> tuple[list[str], list[str]]:
    if "acr" in groups:
        ref, hyp = join_letter_runs(ref), join_letter_runs(hyp)
    if "unit" in groups:
        ref, hyp = join_units(ref), join_units(hyp)
    if "cmp" in groups:
        ref, hyp = _compound(ref), _compound(hyp)
    if "dis" in groups:
        ref = collapse_repetitions([t for t in ref if t not in EXTRA_FILLERS])
        hyp = collapse_repetitions([t for t in hyp if t not in EXTRA_FILLERS])
    return ref, hyp


def utterance_errors(ref: str, hyp: str) -> dict[str, tuple[int, int]]:
    """{variant: (errors, reference words)} of one scored utterance."""
    rn, hn = normalize(ref or ""), normalize(hyp or "")
    out = {}
    for v, groups in VARIANTS.items():
        if not groups:
            rt, ht = rn, hn
        else:
            pr, ph = pre(ref or "", groups), pre(hyp or "", groups)
            rt, ht = post(rn if pr == ref else normalize(pr), hn if ph == hyp else normalize(ph), groups)
        out[v] = (errors(rt, ht), len(rt))
    return out


def fmt_tokens(text: str) -> list[str]:
    return [t[0] if t[0] in MARKS else t for t in _FMT.findall((text or "").replace("’", "'"))]


def words_with_marks(tokens: list[str]) -> tuple[list[str], list[str]]:
    """(words, the mark after each word or "")."""
    words, marks = [], []
    for t in tokens:
        if t in MARKS:
            if words and not marks[-1]:
                marks[-1] = t
        else:
            words.append(t)
            marks.append("")
    return words, marks


def formatted_counts(ref: str, hyp: str) -> collections.Counter:
    from training import cleaning
    rft, hft = fmt_tokens(ref), fmt_tokens(hyp)
    rw, rm = words_with_marks(rft)
    hw, hm = words_with_marks(hft)
    c = collections.Counter()
    c["F_err"] += errors(rft, hft)
    c["F_ref"] += len(rft)
    c["C_err"] += errors(rw, hw)
    c["C_ref"] += len(rw)
    for i, j in cleaning.key_matches([w.lower() for w in rw], [w.lower() for w in hw]):
        c["matched"] += 1
        c["case_err"] += rw[i] != hw[j]
        a, b = rm[i], hm[j]
        c["ref_any"] += bool(a)
        c["hyp_any"] += bool(b)
        c["tp_any"] += bool(a) and bool(b)
        for m in PER_MARK:
            c[f"ref_{m}"] += a == m
            c[f"hyp_{m}"] += b == m
            c[f"tp_{m}"] += a == m and b == m
    return c


def _prf(tp: int, ref: int, hyp: int) -> dict:
    """Precision (None without predicted marks), recall (None without reference marks) and F1 = 2 tp / (ref + hyp)
    (0 when the model writes no mark where the references have some; None when neither side has one)."""
    return {"precision": tp / hyp if hyp else None, "recall": tp / ref if ref else None,
            "f1": 2 * tp / (ref + hyp) if ref + hyp else None, "ref": ref, "hyp": hyp, "tp": tp}


def formatted_summary(c: collections.Counter, punct_only: bool = False) -> dict:
    out = {"any_mark": _prf(c["tp_any"], c["ref_any"], c["hyp_any"]),
           "per_mark": {m: _prf(c[f"tp_{m}"], c[f"ref_{m}"], c[f"hyp_{m}"]) for m in PER_MARK}}
    if not punct_only:
        out |= {"case_error_rate": c["case_err"] / c["matched"] if c["matched"] else None,
                "formatted_wer": 100 * c["F_err"] / c["F_ref"] if c["F_ref"] else None,
                "case_sensitive_wer": 100 * c["C_err"] / c["C_ref"] if c["C_ref"] else None}
    return out


def score_records(records: list[dict]) -> dict:
    """Per scored utterance (in record order): its id, every variant's (errors, words), and the formatted counts."""
    ids, per, fmt = [], {v: [] for v in VARIANTS}, collections.Counter()
    for r in records:
        if not r.get("scored", True):
            continue
        ids.append(r["id"])
        for v, ew in utterance_errors(r.get("ref") or "", r.get("hyp") or "").items():
            per[v].append(ew)
        fmt.update(formatted_counts(r.get("ref") or "", r.get("hyp") or ""))
    wer = {v: 100 * sum(e for e, _ in x) / max(1, sum(w for _, w in x)) for v, x in per.items()}
    return {"ids": ids, "per": per, "wer": wer, "contribution": {v: wer["std"] - wer[v] for v in VARIANTS},
            "formatted": fmt}
