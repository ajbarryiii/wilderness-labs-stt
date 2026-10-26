"""Opt-in native engine for the project's WER edit counts (finetune/whisper-ternary/wer.py), run by the Bend binary.

    edit_counts(ref, hyp)            -> (S, D, I), as wer.edit_counts
    edit_counts_many(pairs)          -> [(S, D, I), ..], one call of the binary per chunk
    corpus_wer(records)              -> the dict wer.corpus_wer returns, for the same records

Words become u32 ids (equal strings, equal ids; one table per call), written as IDS and OFFSETS files for
build/bend-wer, which writes S, D, I per utterance. The binary is $BEND_WER, else build/bend-wer beside this file
(`nix develop <repo> -c bend scorer.bend -o build/bend-wer` from this directory). Nothing in the project uses this
module unless asked to: wer.py stays the scorer of record.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# Word ids and utterances per binary call. A call on 3.9M ids peaked at 125 MB. The binary builds some lists by
# plain recursion, one native stack frame per utterance, so the utterance count is capped as well.
CHUNK_IDS = 1 << 22
CHUNK_UTTS = 1 << 20
U32_MAX = 2**32 - 1


class BendWerError(RuntimeError):
    """The binary failed: its exit code and stderr."""


def binary() -> Path:
    path = Path(os.environ["BEND_WER"]) if os.environ.get("BEND_WER") else HERE / "build" / "bend-wer"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: build it with `nix develop <repo> -c bend scorer.bend -o "
                                f"build/bend-wer` in {HERE}")
    return path


def encode(pairs: Sequence[tuple[Sequence[str], Sequence[str]]]) -> tuple[np.ndarray, np.ndarray]:
    """IDS and OFFSETS for the (ref, hyp) word lists: utterance u is ref ids[o[2u]:o[2u+1]], hyp ids[o[2u+1]:o[2u+2]]."""
    table: dict[str, int] = {}
    ids: list[int] = []
    offsets = [0]
    for ref, hyp in pairs:
        for words in (ref, hyp):
            ids.extend(table.setdefault(w, len(table)) for w in words)
            offsets.append(len(ids))
    if len(ids) > U32_MAX or len(table) > U32_MAX:
        raise ValueError("more than 2^32 - 1 words in one call")
    return np.asarray(ids, dtype="<u4"), np.asarray(offsets, dtype="<u4")


def run(ids: np.ndarray, offsets: np.ndarray, threads: int | None = None, tmp: str | os.PathLike | None = None) \
        -> np.ndarray:
    """The binary on raw arrays: a (k, 3) uint32 array of S, D, I."""
    with tempfile.TemporaryDirectory(dir=tmp, prefix="bend-wer-") as d:
        ip, op, cp = (Path(d) / name for name in ("ids.u32", "offsets.u32", "counts.u32"))
        np.asarray(ids, dtype="<u4").tofile(ip)
        np.asarray(offsets, dtype="<u4").tofile(op)
        argv = [str(binary()), str(ip), str(op), str(cp)]
        if threads is not None:
            argv += ["--threads", str(int(threads))]
        proc = subprocess.run(argv, capture_output=True, text=True)
        if proc.returncode != 0:
            raise BendWerError(f"bend-wer exited {proc.returncode}: {proc.stderr.strip()}")
        return np.fromfile(cp, dtype="<u4").reshape(-1, 3)


def _chunks(pairs: Sequence[tuple[Sequence[str], Sequence[str]]], limit: int, utts: int) -> Iterable[Sequence]:
    start, size = 0, 0
    for k, (ref, hyp) in enumerate(pairs):
        n = len(ref) + len(hyp)
        if k > start and (size + n > limit or k - start >= utts):
            yield pairs[start:k]
            start, size = k, 0
        size += n
    if start < len(pairs) or not pairs:
        yield pairs[start:]


def edit_counts_many(pairs: Sequence[tuple[Sequence[str], Sequence[str]]], threads: int | None = None,
                     chunk_ids: int = CHUNK_IDS, chunk_utts: int = CHUNK_UTTS, tmp: str | os.PathLike | None = None) \
        -> list[tuple[int, int, int]]:
    """wer.edit_counts(ref, hyp) for every (ref, hyp) pair, in order."""
    pairs = list(pairs)
    out: list[tuple[int, int, int]] = []
    for chunk in _chunks(pairs, chunk_ids, chunk_utts):
        out.extend(tuple(map(int, row)) for row in run(*encode(chunk), threads=threads, tmp=tmp))
    if len(out) != len(pairs):
        raise BendWerError(f"bend-wer returned {len(out)} counts for {len(pairs)} utterances")
    return out


def edit_counts(ref: Sequence[str], hyp: Sequence[str]) -> tuple[int, int, int]:
    """(substitutions, deletions, insertions) of one minimum-cost Levenshtein alignment, as wer.edit_counts."""
    return edit_counts_many([(ref, hyp)])[0]


def corpus_wer(records: list[dict], threads: int | None = None) -> dict:
    """wer.corpus_wer: edit counts aggregated over records carrying normalized 'ref_norm' and 'hyp_norm'."""
    pairs = [(r["ref_norm"].split(), r["hyp_norm"].split()) for r in records]
    subs = dels = ins = 0
    for s, d, i in edit_counts_many(pairs, threads=threads):
        subs, dels, ins = subs + s, dels + d, ins + i
    ref_words = sum(len(ref) for ref, _ in pairs)
    hyp_words = sum(len(hyp) for _, hyp in pairs)
    if not ref_words:
        raise ValueError("no reference words to score")
    return {"wer": (subs + dels + ins) / ref_words, "substitutions": subs, "deletions": dels,
            "insertions": ins, "ref_words": ref_words, "hyp_words": hyp_words,
            "utterances": len(records)}
