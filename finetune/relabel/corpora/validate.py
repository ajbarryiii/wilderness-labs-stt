"""Re-validation of the confidence threshold for the training aligner (corpora/ctc_align.py), with Part A's
protocol (devsets/align.py docstring): consecutive dev clips are concatenated into long recordings and
aligned long-form, with every 7th clip's text replaced by another clip's; the clip-mean unit scores of correct
and replaced clips must be separated by the threshold, and aligned clip edges are compared with the true ones.

Inputs are Part A's eval-only dev manifests (audio is read for this measurement only; nothing is trained on).
Output: $RELABEL_ROOT/corpora/aligner-validation.json.

Results (2026-10-22, this protocol, 129 correct + 21 replaced clips per part; tech-dev-1 / med-dev-1):

| aligner (threshold 0.35)            | correct q01  | replaced max | passing replaced | passing, word >1 s into a neighbour |
|-------------------------------------|--------------|--------------|------------------|-------------------------------------|
| chosen: wav2vec2 character CTC       | 0.66 / 1e-4  | 0.27 / 0.17  | 0 / 0            | 1 / 3                               |
| Part A's MMS (reference; eval only)  | 0.23 / 4e-4  | 0.46 / 1.00  | >=1 / >=1        | 3 / 4                               |
| rejected: BPE CTC (80 ms frames)     | 1e-4 / 1e-5  | 0.14 / 1.00  | 0 / >=1          | drift up to 20 s                    |

With the chosen model, 0.35 separates correct from replaced clips on tech-dev-1 (correct min 0.50, replaced max
0.27), as Part A found for MMS; on med-dev-1 (single-channel consultation turns with bleed from the other
channel) no replaced clip passes, 83% of correct clips pass, and the misplacements of passing clips are as rare
as MMS's. Edge errors of correct clips (wav2vec2): median 0.2 / 0.35 s, mostly the clips' own silence margins.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from pool import common

PARTS = ("tech-dev-1", "med-dev-1")
GROUP = 50          # clips per concatenated recording
GROUPS = 3          # recordings per part
REPLACE_EVERY = 7   # every 7th clip gets another clip's text


def _quantiles(x: Sequence[float]) -> dict:
    a = np.asarray(sorted(x), dtype=float)
    if not len(a):
        return {}
    return {"n": int(len(a)), "min": float(a.min()), "q01": float(np.quantile(a, 0.01)),
            "q05": float(np.quantile(a, 0.05)), "median": float(np.median(a)), "q99": float(np.quantile(a, 0.99)),
            "max": float(a.max())}


def build_recordings(rows: list[dict]) -> list[dict]:
    """GROUPS recordings of GROUP consecutive clips; clip k % 7 == 6 takes the text of the same clip index in
    the next group (another place of the dev set)."""
    import soundfile as sf
    rows = sorted(rows, key=lambda r: r["id"])[:GROUP * GROUPS]
    if len(rows) < GROUP * GROUPS:
        raise RuntimeError(f"need {GROUP * GROUPS} clips, have {len(rows)}")
    groups = [rows[g * GROUP:(g + 1) * GROUP] for g in range(GROUPS)]
    out = []
    for g, clips in enumerate(groups):
        audio, edges, texts, replaced = [], [], [], []
        t = 0.0
        for k, r in enumerate(clips):
            x, sr = sf.read(r["audio_filepath"], dtype="float32")
            if sr != 16000 or x.ndim != 1:
                raise RuntimeError(f"{r['id']}: expected 16 kHz mono")
            audio.append(x)
            edges.append((t, t + len(x) / 16000))
            t += len(x) / 16000
            swap = k % REPLACE_EVERY == REPLACE_EVERY - 1
            texts.append(groups[(g + 1) % GROUPS][k]["text"] if swap else r["text"])
            replaced.append(swap)
        out.append({"audio": np.concatenate(audio), "edges": edges, "texts": texts, "replaced": replaced})
    return out


MISPLACED_SECONDS = 1.0   # a correct clip whose first word starts this much before, or last word ends this
                          # much after, the clip's true edges (into a neighbour's audio)


def measure(aligner, recordings: list[dict]) -> dict:
    """Clip-mean scores of correct and replaced clips, edge errors of correct clips, and the safety counts: a
    clip passing the threshold while replaced, or while a word of it is placed more than MISPLACED_SECONDS into
    a neighbouring clip's audio (its segment would carry audio its text does not describe). Words placed inside
    the clip's own silence margins are not counted."""
    from devsets import transcript as tx
    correct, swapped, start_err, end_err = [], [], [], []
    passing_replaced = passing_into_neighbour = 0
    for rec in recordings:
        units, owner = [], []
        for k, text in enumerate(rec["texts"]):
            for tok in tx.tokens(text):
                units.append(tok.chars)
                owner.append(k)
        spans = aligner.align(rec["audio"], units)
        for k, (a, b) in enumerate(rec["edges"]):
            mine = [s for s, o, u in zip(spans, owner, units) if o == k and u is not None]
            if not mine:
                continue
            score = sum(s.score for s in mine) / len(mine)
            passing = score >= aligner.min_score
            (swapped if rec["replaced"][k] else correct).append(score)
            if rec["replaced"][k]:
                passing_replaced += passing
            else:
                start_err.append(abs(mine[0].start - a))
                end_err.append(abs(mine[-1].end - b))
                into = a - mine[0].start > MISPLACED_SECONDS or mine[-1].end - b > MISPLACED_SECONDS
                passing_into_neighbour += passing and into
    return {"correct": _quantiles(correct), "replaced": _quantiles(swapped),
            "start_error_s": _quantiles(start_err), "end_error_s": _quantiles(end_err),
            "separable": bool(correct and swapped and max(swapped) < min(correct)),
            "passing_replaced": int(passing_replaced), "passing_into_neighbour": int(passing_into_neighbour),
            "correct_passing_share": float(np.mean([c >= aligner.min_score for c in correct])) if correct else 0.0}


def run(aligner, manifests: dict[str, Path], out: Path) -> dict:
    result = {"protocol": f"{GROUPS} recordings x {GROUP} consecutive clips per part, every {REPLACE_EVERY}th "
                          "clip's text replaced; clip score = mean unit score",
              "aligner": getattr(aligner, "identity", {}), "min_score": aligner.min_score, "parts": {}}
    for part, path in manifests.items():
        result["parts"][part] = measure(aligner, build_recordings(common.read_jsonl(path)))
        common.log(f"aligner validation {part}: {json.dumps(result['parts'][part])}")
    result["created"] = common.now()
    common.atomic_write_json(out, result)
    return result
