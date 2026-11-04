"""The selection: every validated clip with the first rule that drops it (b0prime plan "Extra sources").

Rules, in order; a clip gets the first one that applies (REASONS), and the rest of a stage sees only clips left. The
text rules 1-3 test both normalized forms of the sentence, as written (`norm`) and as the training label (`label_norm`,
the filters version 3 text; heldout.sentence_keys), and drop the clip if either matches:
1. held out (cv27/heldout.py rule 1): speaker, clip, sentence;
2. the global denylist (rule 2): denylist_identity, denylist_containment;
3. the talk text (rule 3): talk_text_8gram;
4. no_duration: the clip has no clip_durations.tsv entry;
5. caps, in one seeded order: the clips left, sorted by id, permuted by numpy default_rng(seed); walking that order
   a clip is dropped if its speaker already has SPEAKER_CAP kept clips (speaker_cap), else if its normalized
   sentence already has SENTENCE_CAP (sentence_cap); otherwise it is kept and counts toward both;
6. duration: clip_durations.tsv outside 1-30 s (inclusive);
7. labels, filters version 3 (labeling/filters.py rule 4): the training label is map_text(sentence, version=3)
   (NFKC, the curly-quote/dash map, the punctuation map); empty_label if that is empty, tokenizer_coverage if B0's
   tokenizer needs its unknown id for it. Filters 1-3 (they judge a teacher's label against audio) do not apply to
   a human-read sentence.
Cleaning (cleaning plan): the stored label is NOT cleaned; layer 1 alone is applied when the label is read, at the
reading run's cleaning version (store.label_layer1); layers 0 and 2 never apply to this source.
The decoded audio is checked again later: decoded_duration (samples outside 1-30 s), undecodable (store.py), and
B0's confirmation (confirm.py).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from cv27 import heldout, layout

REASONS = heldout.REASONS + ("denylist_identity", "denylist_containment", "talk_text_8gram", "no_duration",
                             "speaker_cap", "sentence_cap", "duration", "empty_label", "tokenizer_coverage")
STAGES = ("validated", "after_heldout", "after_denylist", "after_talk_text", "after_duration_lookup",
          "after_caps", "after_duration", "selected")
COLUMNS = ("id", "path", "client_id", "sentence", "accents", "ms", "norm", "label", "label_norm", "reason", "detail")


@dataclasses.dataclass(slots=True)
class Clip:
    id: str
    path: str
    client_id: str
    sentence: str
    accents: str
    ms: int | None
    norm: str                      # the normalized sentence as written (the sentence cap's key)
    label: str | None = None       # the filters version 3 text (the training label)
    label_norm: str | None = None  # its normalized form
    reason: str | None = None
    detail: str | None = None

    def keys(self) -> tuple[str, ...]:
        return (self.norm,) if self.label_norm in (None, self.norm) else (self.norm, self.label_norm)


def apply_caps(clips: list[Clip], seed: int = layout.CAP_SEED, speaker_cap: int = layout.SPEAKER_CAP,
               sentence_cap: int = layout.SENTENCE_CAP) -> None:
    """Rule 5 on the clips whose reason is None (in place)."""
    live = sorted((c for c in clips if c.reason is None), key=lambda c: c.id)
    order = np.random.default_rng(seed).permutation(len(live))
    per_speaker, per_sentence = Counter(), Counter()
    for i in order.tolist():
        c = live[i]
        if per_speaker[c.client_id] >= speaker_cap:
            c.reason = "speaker_cap"
        elif per_sentence[c.norm] >= sentence_cap:
            c.reason = "sentence_cap"
        else:
            per_speaker[c.client_id] += 1
            per_sentence[c.norm] += 1


def duration_ok(ms: int) -> bool:
    return layout.MIN_MS <= ms <= layout.MAX_MS


def label_of(sentence: str, coverage) -> tuple[str, str | None]:
    """(training label, reason or None): filters version 3's mapping and tokenizer coverage."""
    from labeling import filters
    text = filters.map_text(sentence or "", version=layout.FILTERS_VERSION)
    if not text:
        return text, "empty_label"
    if not coverage(text):
        return text, "tokenizer_coverage"
    return text, None


def _summary(clips: Iterable[Clip]) -> dict:
    clips = list(clips)
    return {"clips": len(clips), "hours": sum(c.ms or 0 for c in clips) / 3.6e6,
            "speakers": len({c.client_id for c in clips})}


def select(validated: Iterable[dict], held: heldout.HeldOut, durations: dict[str, int], talk: frozenset,
           containment, coverage, normalize: Callable[[str], str], seed: int = layout.CAP_SEED) -> tuple[list[Clip], dict]:
    """(every validated clip with its reason, counts) by the rules of the module docstring."""
    clips: list[Clip] = []
    seen = set()
    for r in validated:
        path = r["path"]
        if path in seen:
            raise RuntimeError(f"validated clip {path!r} is listed twice")
        seen.add(path)
        cid = sys.intern(r["client_id"])
        label = heldout.training_text(r["sentence"])
        clips.append(Clip(heldout.clip_id(path), path, cid, r["sentence"], sys.intern(r.get("accents") or ""),
                          durations.get(path), normalize(r["sentence"]), label, normalize(label)))
    stages = {"validated": _summary(clips)}

    for c in clips:
        c.reason = held.reason(c.client_id, c.path, c.keys())
    stages["after_heldout"] = _summary(c for c in clips if c.reason is None)

    live = [c for c in clips if c.reason is None]
    for c in live:
        if containment is not None and containment.identity_hit(c.id):
            c.reason = "denylist_identity"
    live = [c for c in live if c.reason is None]
    if containment is not None:
        hits = containment.contained([c.norm.split() for c in live])
        alt = [i for i, c in enumerate(live) if len(c.keys()) > 1]
        for j, h in containment.contained([live[i].label_norm.split() for i in alt]).items():
            hits.setdefault(alt[j], h)
        for pos, h in hits.items():
            live[pos].reason = "denylist_containment"
            live[pos].detail = json.dumps(h[:5], ensure_ascii=False)
    stages["after_denylist"] = _summary(c for c in clips if c.reason is None)

    for c in clips:
        if c.reason is None and any(heldout.shares_gram(k.split(), talk) for k in c.keys()):
            c.reason = "talk_text_8gram"
    stages["after_talk_text"] = _summary(c for c in clips if c.reason is None)

    for c in clips:
        if c.reason is None and c.ms is None:
            c.reason = "no_duration"
    stages["after_duration_lookup"] = _summary(c for c in clips if c.reason is None)

    apply_caps(clips, seed)
    stages["after_caps"] = _summary(c for c in clips if c.reason is None)

    for c in clips:
        if c.reason is None and not duration_ok(c.ms):
            c.reason = "duration"
    stages["after_duration"] = _summary(c for c in clips if c.reason is None)

    uncovered = Counter()
    for c in clips:
        if c.reason is None:
            _, c.reason = label_of(c.sentence, coverage)
            if c.reason == "tokenizer_coverage" and hasattr(coverage, "uncovered"):
                uncovered.update(coverage.uncovered(c.label))
    stages["selected"] = _summary(c for c in clips if c.reason is None)

    by_reason = Counter(c.reason for c in clips if c.reason is not None)
    hours = Counter()
    for c in clips:
        if c.reason is not None:
            hours[c.reason] += (c.ms or 0) / 3.6e6
    counts = {"heldout": held.counts, "stages": stages,
              "dropped": {r: {"clips": by_reason[r], "hours": hours[r]} for r in REASONS},
              "uncovered_characters": dict(uncovered.most_common(50))}
    return clips, counts


def write_table(path: Path, clips: list[Clip]) -> str:
    """The selection table (COLUMNS, sorted by id) written atomically; returns its SHA-256."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pool import common
    clips = sorted(clips, key=lambda c: c.id)
    cols = {k: [getattr(c, k) for c in clips] for k in COLUMNS}
    table = pa.table({k: pa.array(v, pa.int64() if k == "ms" else pa.string()) for k, v in cols.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    common.atomic_write_bytes(path, tmp.read_bytes())
    tmp.unlink()
    return common.sha256_file(path)


def read_selected(paths: layout.Paths) -> tuple[list[dict], dict]:
    """(selected rows sorted by id, selection meta); refuses a table whose SHA-256 differs from the meta."""
    import pyarrow.parquet as pq
    from pool import common
    meta = json.loads(paths.selection_meta.read_text())
    if common.sha256_file(paths.selection) != meta["table_sha256"]:
        raise RuntimeError(f"{paths.selection} does not match {paths.selection_meta}")
    t = pq.read_table(paths.selection, columns=["id", "path", "client_id", "sentence", "accents", "ms", "label",
                                                 "reason"]).to_pylist()
    rows = [r for r in t if r["reason"] is None]
    return sorted(rows, key=lambda r: r["id"]), meta


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
