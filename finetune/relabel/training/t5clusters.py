"""T5, the technical held-out-source panel of the mutant plan (encoder probe 2), and its bootstrap clusters.

T5 is five domain sets by alias: tech-dev-1 (talks), tech-dev-2 (calls), stem-chem, stem-cs and stem-math (lectures).
Their manifests are $RELABEL_ROOT/devsets/manifests/<set>.jsonl; ids are "<set>:<recording>-<NNNN>", the recording
being the source item each segment was cut from. T5's mean is the unweighted mean of the five set WERs; v3.1 also
reports it without tech-dev-2 (a sibling of one teacher's training data), as a point estimate.

Cluster rule (plan "Intervals", T5): the cluster is a talk, call or lecture, read from the item id's prefix (the
recording, without the segment number). A set gets its own interval only with at least MIN_CLUSTERS clusters.

tech-dev-1's id prefix is the source's row index within a split ("dev-<n>", "eval-<n>"): one segment per prefix, not
a talk, and the source has no talk column. The prefix rule would give one cluster per segment there, an utterance
bootstrap that ignores within-talk correlation, so tech-dev-1 is listed in NO_TALK_ID: it has no cluster
definition and gets a point estimate only (reported to the architect; not a plan choice made here).
"""
from __future__ import annotations

import re

SETS = ("tech-dev-1", "tech-dev-2", "stem-chem", "stem-cs", "stem-math")
WITHOUT = "tech-dev-2"                                   # v3.1: T5 is also reported without this set
SETS_WITHOUT = tuple(s for s in SETS if s != WITHOUT)
MIN_CLUSTERS = 10
NO_TALK_ID = {"tech-dev-1": "the id prefix is a per-segment row index (one segment per prefix), not a talk"}
_ID = re.compile(r"^(?P<set>[a-z0-9-]+):(?P<recording>.+)-(?P<segment>\d{4})$")


def cluster_of(set_name: str, item_id: str) -> str:
    """The recording (talk, call or lecture) of a T5 item: the id's prefix without the segment number."""
    if set_name not in SETS:
        raise ValueError(f"{set_name!r} is not a T5 set")
    if set_name in NO_TALK_ID:
        raise ValueError(f"{set_name}: no cluster definition ({NO_TALK_ID[set_name]})")
    m = _ID.match(item_id)
    if m is None or m.group("set") != set_name:
        raise ValueError(f"{set_name}: cannot read the recording of {item_id!r}")
    return m.group("recording")


def clusters(set_name: str, ids) -> set[str] | None:
    """The set's clusters among `ids`, or None for a set without a cluster definition."""
    if set_name in NO_TALK_ID:
        return None
    return {cluster_of(set_name, i) for i in ids}


def interval_sets(ids_by_set: dict[str, list[str]]) -> dict[str, dict]:
    """Per T5 set: its cluster count (None without a definition) and whether it gets its own interval."""
    out = {}
    for name in SETS:
        found = clusters(name, ids_by_set[name])
        count = None if found is None else len(found)
        out[name] = {"clusters": count, "interval": count is not None and count >= MIN_CLUSTERS,
                     "reason": NO_TALK_ID.get(name) if count is None else
                     (None if count >= MIN_CLUSTERS else f"fewer than {MIN_CLUSTERS} clusters")}
    return out


def mean(values: dict[str, float], sets=SETS) -> float:
    """The unweighted mean over `sets` (T5: all five; T5 without tech-dev-2: SETS_WITHOUT)."""
    missing = [s for s in sets if s not in values]
    if missing:
        raise ValueError(f"T5 mean: missing sets {missing}")
    return sum(values[s] for s in sets) / len(sets)
