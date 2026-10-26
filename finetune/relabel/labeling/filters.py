"""Filters 1-4 of the labeling pass (labeling plan Part C "Filters"); an utterance failing any one has no label.

1. 1-30 s, and a non-empty label (`duration_out_of_range`, `empty_label`).
2. Where a human transcript exists (reference_kind "human", non-empty): normalized WER(human, T*) <= 0.5
   (`human_wer_gt_0.5`; evaluate.utterance_wer, as parakeet-ternary teacher.drop_reason does today).
3. Hallucination guard, every source (normalized label words, the evaluation normalizer):
   - more than 6 words per second of audio (`words_per_second_gt_6`), unless an independent transcript confirms the
     label: normalized WER(confirmer, T*) <= 0.2, the confirmer being the human transcript where there is one, else
     B0's label (values "rate_confirmed"; version 2, labeling plan Part C "Filters", amended 2026-10-25);
   - a repeated n-gram loop: any 3-gram occurring >= 4 times when the label has <= 30 words, or the same word
     repeated >= 5 times in a row (`ngram_loop`);
   - for utterances without a human transcript: normalized WER(B0 greedy, T*) > 0.8, B0's label as the
     reference (`b0_wer_gt_0.8`).
4. Tokenizer coverage: after NFKC and the fixed mapping of curly quotes and dashes (TEXT_MAP), the label must
   tokenize with B0's SentencePiece model without the unknown id (`tokenizer_coverage`). Version 3 (labeling plan
   Part C "Filters", amended 2026-10-26) extends the mapping, after TEXT_MAP, with characters B0's vocabulary lacks
   (PUNCT_MAP): `" *` are removed, `( ) [ ] { } _` become a space, `;` becomes `,`, `& + = @ ° §` become the words
   " and ", " plus ", " equals ", " at ", " degrees ", " section "; then runs of whitespace become one space, a space
   before `, . ? ! :` is removed and the text is stripped. Every other character is unchanged (so `# < > \\ | ~`,
   backtick, non-Latin scripts and diacritics B0 lacks still fail), and a non-empty label that maps to empty text
   fails this rule.

Every rule is evaluated (flags record each failure); the disposition is "kept" iff no rule fails, with the
failing reasons in filter order. The stored training text is the mapped label (`text`).

What filters 2-3 score ("T*" above, and `norm`, the normalized words): versions 1 and 2 the raw label; version 3 the
mapped text `text` (labeling plan Part C "Filters", amended 2026-10-26). The evaluation normalizer deletes bracketed
text, so scoring the raw label would leave the words of a parenthetical, which version 3 keeps as training text,
unchecked; scored from `text`, they are compared with the reference or B0's label and counted in words per second.
Mapped symbols ("plus", "degrees") are scored as words, as B0, whose vocabulary has no symbols, writes them. Filter 1
always looks at the raw label (`empty_label` iff it is empty); a raw label that maps to empty text is scored as an
empty hypothesis and fails filter 4.

Versions (run.json filters_version; every caller names one, of `check` and of `map_text`): 1 has no confirmation of the
words-per-second rule, 2 has it, 3 adds the punctuation mapping and scores filters 2-3 on the mapped text (versions 1
and 2 map alike). C2's merge applies its run's version, so a run labeled under version 1 stays consistent; c2.refilter
re-applies the current version to a finished run's stored outputs, the stored text and normalized words included.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

MIN_SECONDS, MAX_SECONDS = 1.0, 30.0
MAX_HUMAN_WER = 0.5
MAX_WORDS_PER_SECOND = 6.0
LOOP_NGRAM, LOOP_COUNT, LOOP_MAX_WORDS = 3, 4, 30
REPEAT_RUN = 5
MAX_B0_WER = 0.8
CONFIRM_WER = 0.2
TEXT_MAP = {"‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
            "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
            "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-"}
REASONS = ("duration_out_of_range", "empty_label", "human_wer_gt_0.5", "words_per_second_gt_6", "ngram_loop",
           "b0_wer_gt_0.8", "tokenizer_coverage")
FILTER_OF = {"duration_out_of_range": 1, "empty_label": 1, "human_wer_gt_0.5": 2, "words_per_second_gt_6": 3,
             "ngram_loop": 3, "b0_wer_gt_0.8": 3, "tokenizer_coverage": 4}
HALLUCINATION = tuple(r for r, f in FILTER_OF.items() if f == 3)
PUNCT_MAP = {'"': "", "*": "", "(": " ", ")": " ", "[": " ", "]": " ", "{": " ", "}": " ", "_": " ", ";": ",",
             "&": " and ", "+": " plus ", "=": " equals ", "@": " at ", "°": " degrees ", "§": " section "}
_SPACE_BEFORE_PUNCT = re.compile(r" ([,.?!:])")
VERSIONS = (1, 2, 3)
VERSION = 3


def map_text(text: str, *, version: int) -> str:
    """The stored training text of a raw label under filters `version` (module docstring, filter 4)."""
    if version not in VERSIONS:
        raise ValueError(f"unknown filters version {version!r}")
    text = "".join(TEXT_MAP.get(c, c) for c in unicodedata.normalize("NFKC", text))
    if version >= 3:
        text = " ".join("".join(PUNCT_MAP.get(c, c) for c in text).split())
        text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return text.strip()


def ngram_loop(words: list[str]) -> bool:
    run = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a == b else 1
        if run >= REPEAT_RUN:
            return True
    if len(words) <= LOOP_MAX_WORDS:
        grams = Counter(tuple(words[i:i + LOOP_NGRAM]) for i in range(len(words) - LOOP_NGRAM + 1))
        if grams and max(grams.values()) >= LOOP_COUNT:
            return True
    return False


class Coverage:
    """B0's tokenizer: a label is covered iff it encodes without the unknown id."""

    def __init__(self, model_file: Path):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=str(model_file))
        self.unk = self.sp.unk_id()

    def __call__(self, text: str) -> bool:
        return self.unk not in self.sp.encode(text)

    def uncovered(self, text: str) -> list[str]:
        return sorted({c for c in text if not c.isspace() and self.unk in self.sp.encode(c)})


def b0_tokenizer_file(cache_dir: Path) -> Path:
    """B0's SentencePiece model, extracted once from the pinned .nemo (parakeet-ternary paths.MODEL_FILE)."""
    import tarfile
    import paths  # finetune/parakeet-ternary
    cache_dir = Path(cache_dir)
    out = cache_dir / "b0_tokenizer.model"
    if out.exists():
        return out
    with tarfile.open(paths.MODEL_FILE) as tar:
        members = [m for m in tar.getmembers() if m.name.endswith("_tokenizer.model")]
        if len(members) != 1:
            raise RuntimeError(f"expected one tokenizer model in {paths.MODEL_FILE}, found {len(members)}")
        data = tar.extractfile(members[0]).read()
    from pool import common
    common.atomic_write_bytes(out, data)
    return out


def rate_confirmed(words_per_second: float | None, reasons) -> bool:
    """Whether a stored record's words-per-second rule was waived by a confirming transcript (from its columns)."""
    fast = words_per_second is None or words_per_second > MAX_WORDS_PER_SECOND   # None: infinite (zero duration)
    return fast and "words_per_second_gt_6" not in reasons


def label_texts(label: str | None, *, version: int) -> dict:
    """A label's raw (stripped), stored training (`text`) and normalized (`norm`) forms under filters `version`, and
    `scored`, what filters 2-3 score: the raw label under versions 1-2, the mapped text under 3 (module docstring)."""
    import evaluate  # finetune/parakeet-ternary
    raw = (label or "").strip()
    text = map_text(raw, version=version)
    scored = text if version >= 3 else raw
    return {"raw": raw, "text": text, "scored": scored, "norm": evaluate.normalize(scored).strip() if scored else ""}


def check(duration: float, label: str | None, reference: str | None, reference_kind: str | None,
          b0_label: str | None, coverage, *, version: int) -> dict:
    """Filter flags and disposition of one utterance (T*'s raw label; B0's raw label) under filters `version`."""
    import evaluate  # finetune/parakeet-ternary
    if version not in VERSIONS:
        raise ValueError(f"unknown filters version {version!r}")
    forms = label_texts(label, version=version)
    raw, text, scored, norm = forms["raw"], forms["text"], forms["scored"], forms["norm"]
    words = norm.split()
    reasons = []
    values: dict = {"words": len(words), "words_per_second": len(words) / duration if duration > 0 else math.inf}
    if duration < MIN_SECONDS or duration > MAX_SECONDS:
        reasons.append("duration_out_of_range")
    if not raw:
        reasons.append("empty_label")
    human = reference_kind == "human" and isinstance(reference, str) and bool(reference.strip())
    if human:
        values["human_wer"] = evaluate.utterance_wer(reference, scored)
        if values["human_wer"] > MAX_HUMAN_WER:
            reasons.append("human_wer_gt_0.5")
    if not human:
        values["b0_wer"] = evaluate.utterance_wer(b0_label or "", scored)
    if values["words_per_second"] > MAX_WORDS_PER_SECOND:
        if version >= 2:
            # an infinite WER (B0 empty, or a reference that normalizes to nothing) never confirms
            values["rate_confirmed"] = values["human_wer" if human else "b0_wer"] <= CONFIRM_WER
        if not values.get("rate_confirmed"):
            reasons.append("words_per_second_gt_6")
    if ngram_loop(words):
        reasons.append("ngram_loop")
    if not human and values["b0_wer"] > MAX_B0_WER:
        reasons.append("b0_wer_gt_0.8")
    if raw:
        # version 3: a label the mapping empties (e.g. a lone quotation mark) has no training text
        covered = bool(text) and coverage(text) if version >= 3 else coverage(text)
        if not covered:
            reasons.append("tokenizer_coverage")
    values = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in values.items()}
    if "b0_wer" in values and values["b0_wer"] is None:
        values["b0_wer_infinite"] = True
    return {"raw": raw, "text": text, "norm": norm, "disposition": "rejected" if reasons else "kept",
            "reasons": reasons, "flags": {r: r in reasons for r in REASONS}, "values": values}
