"""Filters 1-4 of the labeling pass (labeling plan Part C "Filters"); an utterance failing any one has no label.

1. 1-30 s, and a non-empty label (`duration_out_of_range`, `empty_label`).
2. Where a human transcript exists (reference_kind "human", non-empty): normalized WER(human, T*) <= 0.5
   (`human_wer_gt_0.5`; evaluate.utterance_wer, as parakeet-ternary teacher.drop_reason does today).
3. Hallucination guard, every source (normalized label words, the evaluation normalizer):
   - more than 6 words per second of audio (`words_per_second_gt_6`);
   - a repeated n-gram loop: any 3-gram occurring >= 4 times when the label has <= 30 words, or the same word
     repeated >= 5 times in a row (`ngram_loop`);
   - for utterances without a human transcript: normalized WER(B0 greedy, T*) > 0.8, B0's label as the
     reference (`b0_wer_gt_0.8`).
4. Tokenizer coverage: after NFKC and the fixed mapping of curly quotes and dashes (TEXT_MAP), the label must
   tokenize with B0's SentencePiece model without the unknown id (`tokenizer_coverage`).

Every rule is evaluated (flags record each failure); the disposition is "kept" iff no rule fails, with the
failing reasons in filter order. The stored training text is the mapped label (`text`).
"""
from __future__ import annotations

import math
import unicodedata
from collections import Counter
from pathlib import Path

MIN_SECONDS, MAX_SECONDS = 1.0, 30.0
MAX_HUMAN_WER = 0.5
MAX_WORDS_PER_SECOND = 6.0
LOOP_NGRAM, LOOP_COUNT, LOOP_MAX_WORDS = 3, 4, 30
REPEAT_RUN = 5
MAX_B0_WER = 0.8
TEXT_MAP = {"‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
            "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
            "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-"}
REASONS = ("duration_out_of_range", "empty_label", "human_wer_gt_0.5", "words_per_second_gt_6", "ngram_loop",
           "b0_wer_gt_0.8", "tokenizer_coverage")
FILTER_OF = {"duration_out_of_range": 1, "empty_label": 1, "human_wer_gt_0.5": 2, "words_per_second_gt_6": 3,
             "ngram_loop": 3, "b0_wer_gt_0.8": 3, "tokenizer_coverage": 4}
HALLUCINATION = tuple(r for r, f in FILTER_OF.items() if f == 3)
VERSION = 1


def map_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return "".join(TEXT_MAP.get(c, c) for c in text).strip()


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


def check(duration: float, label: str | None, reference: str | None, reference_kind: str | None,
          b0_label: str | None, coverage) -> dict:
    """Filter flags and disposition of one utterance (T*'s raw label; B0's raw label)."""
    import evaluate  # finetune/parakeet-ternary
    raw = (label or "").strip()
    text = map_text(raw)
    norm = evaluate.normalize(raw).strip() if raw else ""
    words = norm.split()
    reasons = []
    values: dict = {"words": len(words), "words_per_second": len(words) / duration if duration > 0 else math.inf}
    if duration < MIN_SECONDS or duration > MAX_SECONDS:
        reasons.append("duration_out_of_range")
    if not raw:
        reasons.append("empty_label")
    human = reference_kind == "human" and isinstance(reference, str) and bool(reference.strip())
    if human:
        values["human_wer"] = evaluate.utterance_wer(reference, raw)
        if values["human_wer"] > MAX_HUMAN_WER:
            reasons.append("human_wer_gt_0.5")
    if values["words_per_second"] > MAX_WORDS_PER_SECOND:
        reasons.append("words_per_second_gt_6")
    if ngram_loop(words):
        reasons.append("ngram_loop")
    if not human:
        values["b0_wer"] = evaluate.utterance_wer(b0_label or "", raw)
        if values["b0_wer"] > MAX_B0_WER:
            reasons.append("b0_wer_gt_0.8")
    if raw and not coverage(text):
        reasons.append("tokenizer_coverage")
    values = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in values.items()}
    if "b0_wer" in values and values["b0_wer"] is None:
        values["b0_wer_infinite"] = True
    return {"raw": raw, "text": text, "norm": norm, "disposition": "rejected" if reasons else "kept",
            "reasons": reasons, "flags": {r: r in reasons for r in REASONS}, "values": values}
