"""Cut a recording into 1-30 s evaluation segments with stable ids (labeling plan Part A; reused by Part B).

    segment(audio, recording_id, transcript, aligner=None, boundary_words=(), normalize=None) -> Result

`audio` is 16 kHz mono float32. `transcript` is either

- a list of Utterance(start, end, text): the source's own utterance timestamps. Each utterance is one
  segment, cut exactly at its timestamps, with id "<recording_id>-<k>" (k = its index, 4 digits, in
  time order). An utterance longer than MAX_SECONDS is split with the aligner as below (ids
  "<recording_id>-<k>-<j>"); without an aligner it is dropped ("too_long"). Shorter than MIN_SECONDS:
  dropped ("too_short").
- a str: a recording without usable timestamps. Its units (transcript.tokens) are forced-aligned
  (align.MMSAligner, independent of every evaluated model), split into sentences (a token ending in . ? !,
  or one of `boundary_words` for dictated punctuation), sentences longer than MAX_SECONDS are split at
  their largest pauses, consecutive sentences are packed into segments of at most TARGET_SECONDS, and
  a segment still shorter than MIN_SECONDS is merged with its nearer neighbour when the result fits.
  Each segment is cut PAD_SECONDS outside its first and last unit, but never past the midpoint of the
  pause to the neighbouring unit. Ids are "<recording_id>-<k>", k = the segment's index before any drop,
  so a dropped segment never renumbers the others.

A segment is dropped, with its reason counted, if its reference normalizes to nothing ("empty_reference"),
its alignment failed ("alignment_failed": the mean unit score is below the aligner's min_score, no unit
could be aligned, or an aligned word spans more than MAX_WORD_SECONDS), or it lies outside 1-30 s after
cutting ("too_short", "too_long"). References are the source text without markup tags
(transcript.strip_tags). Drop reasons are counted per segment, with their audio seconds.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from devsets import transcript as tx

SAMPLE_RATE = 16000
MIN_SECONDS = 1.0
MAX_SECONDS = 30.0
TARGET_SECONDS = 15.0
PAD_SECONDS = 0.25
MAX_WORD_SECONDS = 3.0
DROP_REASONS = ("empty_reference", "alignment_failed", "too_short", "too_long")
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class Utterance:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Segment:
    id: str
    start: float       # seconds in the recording
    end: float
    text: str          # reference: source text without markup tags
    aligned: bool      # boundaries from the aligner (False: the source's own timestamps)
    untranscribed: bool = False  # the source marked untranscribed audio inside it (a markup-only token)


@dataclass
class Result:
    segments: list[Segment] = field(default_factory=list)
    drops: Counter = field(default_factory=Counter)
    dropped_seconds: Counter = field(default_factory=Counter)

    def drop(self, reason: str, seconds: float) -> None:
        assert reason in DROP_REASONS, reason
        self.drops[reason] += 1
        self.dropped_seconds[reason] += max(0.0, seconds)


def safe_id(recording_id: str) -> str:
    """A recording id usable in segment ids and file names (characters outside [A-Za-z0-9._-] -> "_")."""
    out = _SAFE.sub("_", str(recording_id)).strip("._")
    if not out:
        raise ValueError(f"recording id {recording_id!r} has no usable characters")
    return out


def _default_normalize(text: str) -> str:
    import evaluate  # finetune/parakeet-ternary/evaluate.py

    return evaluate.normalize(text)


def samples(seconds: float) -> int:
    return int(round(seconds * SAMPLE_RATE))


def cut_seconds(start: float, end: float) -> float:
    """Duration of the cut audio[samples(start):samples(end)], as written."""
    return (samples(end) - samples(start)) / SAMPLE_RATE


def segment(audio: np.ndarray, recording_id: str, transcript, aligner=None, boundary_words=(),
            normalize: Callable[[str], str] | None = None) -> Result:
    normalize = normalize or _default_normalize
    prefix = safe_id(recording_id)
    words = frozenset(w.lower() for w in boundary_words)
    duration = len(audio) / SAMPLE_RATE
    result = Result()
    if isinstance(transcript, str):
        _from_text(audio, 0.0, prefix, transcript, aligner, words, normalize, result)
        return result
    utterances = sorted(transcript, key=lambda u: (u.start, u.end))
    for k, utt in enumerate(utterances):
        sid = f"{prefix}-{k:04d}"
        start, end = max(0.0, utt.start), min(duration, utt.end)
        text = tx.strip_tags(utt.text)
        seconds = cut_seconds(start, end) if end > start else 0.0
        if not normalize(text).strip():
            result.drop("empty_reference", seconds)
        elif seconds < MIN_SECONDS:
            result.drop("too_short", seconds)
        elif seconds > MAX_SECONDS:
            if aligner is None:
                result.drop("too_long", seconds)
            else:
                piece = audio[samples(start):samples(end)]
                _from_text(piece, samples(start) / SAMPLE_RATE, sid, utt.text, aligner, words, normalize, result,
                           index_width=2)
        else:
            untranscribed = any(not t.text for t in tx.tokens(utt.text))
            result.segments.append(Segment(sid, start, end, text, False, untranscribed))
    return result


def _sentences(tokens: list[tx.Token], words: frozenset[str]) -> list[tuple[int, int]]:
    out, a = [], 0
    for i, tok in enumerate(tokens):
        if tx.ends_sentence(tok, words):
            out.append((a, i + 1))
            a = i + 1
    if a < len(tokens):
        out.append((a, len(tokens)))
    return out


def _split_long(a: int, b: int, spans, limit: float) -> list[tuple[int, int]]:
    """Split units [a, b) at their largest pauses until every piece spans at most `limit` seconds."""
    if spans[b - 1].end - spans[a].start <= limit or b - a < 2:
        return [(a, b)]
    cut = max(range(a + 1, b), key=lambda u: (spans[u].start - spans[u - 1].end, -abs(2 * u - a - b)))
    return _split_long(a, cut, spans, limit) + _split_long(cut, b, spans, limit)


def _pack(pieces: list[tuple[int, int]], spans, target: float, limit: float) -> list[tuple[int, int]]:
    packed: list[list[int]] = []
    for a, b in pieces:
        if packed and spans[b - 1].end - spans[packed[-1][0]].start <= target:
            packed[-1][1] = b
        else:
            packed.append([a, b])
    # Merge a piece still under MIN_SECONDS into its nearer neighbour if the result fits.
    k = 0
    while k < len(packed):
        a, b = packed[k]
        if spans[b - 1].end - spans[a].start + 2 * PAD_SECONDS >= MIN_SECONDS or len(packed) == 1:
            k += 1
            continue
        gap_prev = spans[a].start - spans[packed[k - 1][1] - 1].end if k > 0 else float("inf")
        gap_next = spans[packed[k + 1][0]].start - spans[b - 1].end if k + 1 < len(packed) else float("inf")
        order = [k - 1, k + 1] if gap_prev <= gap_next else [k + 1, k - 1]
        for other in order:
            if 0 <= other < len(packed):
                lo, hi = min(other, k), max(other, k)
                if spans[packed[hi][1] - 1].end - spans[packed[lo][0]].start <= limit:
                    packed[lo] = [packed[lo][0], packed[hi][1]]
                    del packed[hi]
                    k = lo
                    break
        else:
            k += 1
    return [(a, b) for a, b in packed]


def _from_text(audio: np.ndarray, offset: float, prefix: str, text: str, aligner, words: frozenset[str],
               normalize, result: Result, index_width: int = 4) -> None:
    tokens = tx.tokens(text)
    duration = len(audio) / SAMPLE_RATE
    if not tokens or not normalize(tx.strip_tags(text)).strip():
        result.drop("empty_reference", cut_seconds(0.0, duration))
        return
    if aligner is None:
        raise ValueError("a transcript without timestamps needs an aligner")
    spans = aligner.align(audio, [t.chars for t in tokens])
    min_score = aligner.min_score
    limit = MAX_SECONDS - 2 * PAD_SECONDS
    pieces = [p for a, b in _sentences(tokens, words) for p in _split_long(a, b, spans, limit)]
    for k, (a, b) in enumerate(_pack(pieces, spans, TARGET_SECONDS, limit)):
        left = spans[a - 1].end if a > 0 else 0.0
        right = spans[b].start if b < len(spans) else duration
        start = max(0.0, spans[a].start - min(PAD_SECONDS, max(0.0, spans[a].start - left) / 2))
        end = min(duration, spans[b - 1].end + min(PAD_SECONDS, max(0.0, right - spans[b - 1].end) / 2))
        seconds = cut_seconds(start, end) if end > start else 0.0
        ref = " ".join(t.text for t in tokens[a:b] if t.text)
        scored = [(spans[u], tokens[u]) for u in range(a, b) if tokens[u].chars is not None]
        if not normalize(ref).strip():
            result.drop("empty_reference", seconds)
        elif (not scored or sum(s.score for s, _ in scored) / len(scored) < min_score
              or any(s.end - s.start > MAX_WORD_SECONDS for s, _ in scored)):
            result.drop("alignment_failed", seconds)
        elif seconds < MIN_SECONDS:
            result.drop("too_short", seconds)
        elif seconds > MAX_SECONDS:
            result.drop("too_long", seconds)
        else:
            untranscribed = any(not t.text for t in tokens[a:b])
            result.segments.append(Segment(f"{prefix}-{k:0{index_width}d}", offset + start, offset + end, ref, True,
                                           untranscribed))
