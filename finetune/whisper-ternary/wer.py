"""Whisper-normalized, corpus-level word error rate with exact edit counts.

Both reference and hypothesis go through the Whisper EnglishTextNormalizer
built from the checkpoint's normalizer.json. WER = (S + D + I) / reference
words summed over the corpus; it is not an average of per-utterance rates and
can exceed 1.0.
"""
from __future__ import annotations

import json
from pathlib import Path

from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

import paths

NORMALIZER = "whisper EnglishTextNormalizer from normalizer.json"


class Normalizer:
    def __init__(self, model_dir: Path = paths.MODEL_DIR) -> None:
        spellings = json.loads((model_dir / "normalizer.json").read_text())
        self._normalize = EnglishTextNormalizer(spellings)

    def __call__(self, text: str) -> str:
        return self._normalize(text)


def edit_counts(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """(substitutions, deletions, insertions) of one minimum-cost Levenshtein alignment."""
    rows, cols = len(ref) + 1, len(hyp) + 1
    cost = [[i + j if i == 0 or j == 0 else 0 for j in range(cols)] for i in range(rows)]
    for i in range(1, rows):
        for j in range(1, cols):
            cost[i][j] = min(cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
                             cost[i - 1][j] + 1, cost[i][j - 1] + 1)
    subs = dels = ins = 0
    i, j = rows - 1, cols - 1
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            subs += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            dels, i = dels + 1, i - 1
        else:
            ins, j = ins + 1, j - 1
    return subs, dels, ins


def corpus_wer(records: list[dict]) -> dict:
    """Aggregate edit counts over records carrying normalized 'ref_norm' and 'hyp_norm'."""
    subs = dels = ins = ref_words = hyp_words = 0
    for record in records:
        ref, hyp = record["ref_norm"].split(), record["hyp_norm"].split()
        s, d, i = edit_counts(ref, hyp)
        subs, dels, ins = subs + s, dels + d, ins + i
        ref_words, hyp_words = ref_words + len(ref), hyp_words + len(hyp)
    if not ref_words:
        raise ValueError("no reference words to score")
    return {"wer": (subs + dels + ins) / ref_words, "substitutions": subs, "deletions": dels,
            "insertions": ins, "ref_words": ref_words, "hyp_words": hyp_words,
            "utterances": len(records)}
