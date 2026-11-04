"""The dev sample (b0prime plan "CV dev (informational)"): reported like YODAS dev, never used for decisions.

Eligible: rows of the release's dev split whose path is not an ESB test path, whose client_id is not an ESB speaker
in either id form (ESB's `id`, or this release's client_id of an ESB path; heldout.HeldOut.esb_speakers), and whose
sentence matches no ESB test sentence in either normalized form (heldout.sentence_keys). The dev split's speakers, clips and sentences are all held out of
training (heldout.py), so the sample never overlaps the training clips.

Sample: the eligible rows sorted by path, shuffled by random.Random(DEV_SEED), then taken in that order with at most
DEV_PER_SPEAKER per client_id until DEV_SIZE rows.

Manifest (stage 1's dev manifest format, evaluate.read_manifest / evaluate_set): one JSON line per clip
{"audio_filepath", "duration", "text", "id", "source"}, sorted by id, text = the read sentence as written (the
reference; the scorer normalizes it), duration = decoded samples / 16,000, plus <set>.meta.json.
"""
from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Iterable

from cv27 import heldout, layout


def eligible(dev_rows: Iterable[dict], held: heldout.HeldOut, normalize: Callable[[str], str]) -> list[dict]:
    return [r for r in dev_rows if r["path"] not in held.esb_paths and r["client_id"] not in held.esb_speakers
            and not any(k in held.esb_sentences for k in heldout.sentence_keys(r["sentence"], normalize))]


def sample(rows: list[dict], n: int = layout.DEV_SIZE, per_speaker: int = layout.DEV_PER_SPEAKER,
           seed: int = layout.DEV_SEED) -> list[dict]:
    order = sorted(rows, key=lambda r: r["path"])
    random.Random(seed).shuffle(order)
    taken, per = [], Counter()
    for r in order:
        if per[r["client_id"]] >= per_speaker:
            continue
        per[r["client_id"]] += 1
        taken.append(r)
        if len(taken) == n:
            break
    return taken


def manifest_rows(sample_rows: list[dict], records: dict[str, dict]) -> list[dict]:
    """The scorer manifest of the decoded sample (records: clip id -> store.decode_clip record)."""
    out = []
    for r in sample_rows:
        cid = heldout.clip_id(r["path"])
        rec = records[cid]
        out.append({"audio_filepath": rec["audio_filepath"], "duration": rec["samples"] / layout.SAMPLE_RATE,
                    "text": r["sentence"], "id": cid, "source": layout.ALIAS})
    return sorted(out, key=lambda r: r["id"])
