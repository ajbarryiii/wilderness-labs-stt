"""The global denylist (labeling plan Part B), from a scan and the held-out index.

Rules, each recorded per recording:
- reserved: the reserved YODAS development recordings (parakeet-ternary yodas_reserved_shards.json);
- identity: recordings named by held-out ids (heldout.IDENTITY);
- containment: a held-out utterance with >= 8 words whose distinct non-boilerplate 8-grams occur at least
  half in a recording's transcript; boilerplate = an 8-gram in more than 100 training recordings (all routes).
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from pool import common
from pool.heldout import Index
from pool.scan import ScanResult

RULES = ("reserved", "identity", "containment")
DETAIL_HELDOUT = 20   # held-out utterances named per recording in the details file


def _containment(scan: ScanResult, index: Index, boiler: np.ndarray, nb: np.ndarray,
                 chunk: int = 4_000_000) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(gid, utterance, matched, of) for every contained (recording, held-out utterance) pair."""
    mask = np.uint64(0xFFFFFFFF)
    gid_all = (scan.pairs >> np.uint64(32)).astype(np.int64)
    h_all = (scan.pairs & mask).astype(np.int64)
    keep = ~boiler[h_all]
    gid_all, h_all = gid_all[keep], h_all[keep]
    need = np.ceil(common.CONTAIN_FRACTION * nb).astype(np.int64)
    out = [], [], [], []
    start = 0
    while start < len(gid_all):   # chunks end at a gid change (pairs are sorted by gid)
        end = min(len(gid_all), start + chunk)
        while end < len(gid_all) and gid_all[end] == gid_all[end - 1]:
            end += 1
        g, h = gid_all[start:end], h_all[start:end]
        lo, hi = index.csr_ptr[h], index.csr_ptr[h + 1]
        lens = hi - lo
        rep_g = np.repeat(g, lens)
        offs = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
        utt = index.csr_utts[np.repeat(lo, lens) + offs]
        keys, counts = np.unique((rep_g << 32) | utt, return_counts=True)
        kg, ku = keys >> 32, keys & 0xFFFFFFFF
        ok = (nb[ku] > 0) & (counts >= need[ku])
        for lst, arr in zip(out, (kg[ok], ku[ok], counts[ok], nb[ku[ok]])):
            lst.append(arr)
        start = end
    return tuple(np.concatenate(x) if x else np.empty(0, np.int64) for x in out)


def build(scan: ScanResult, index: Index, reserved: Iterable[str]) -> tuple[dict, list[dict]]:
    """(denylist object without provenance fields, details rows)."""
    nh = len(index.hashes)
    h_all = (scan.pairs & np.uint64(0xFFFFFFFF)).astype(np.int64)
    df = np.bincount(h_all, minlength=nh) if nh else np.zeros(0, np.int64)
    boiler = df > common.BOILERPLATE_MAX
    nb = np.array([int((~boiler[g]).sum()) for g in index.utt_grams], dtype=np.int64) \
        if len(index) else np.zeros(0, np.int64)
    c_gid, c_utt, c_cnt, c_of = _containment(scan, index, boiler, nb)

    rules: dict[tuple[str, str], set[str]] = defaultdict(set)
    heldout_hits: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in reserved:
        rules[("yodas", rec)].add("reserved")
    set_identity = defaultdict(set)
    for k, s in enumerate(index.sets):
        for key in s.identity:
            rules[key].add("identity")
            set_identity[k].add(key)
    set_contained_utts = defaultdict(set)
    set_recordings = defaultdict(set)
    for g, u, cnt, of in zip(c_gid.tolist(), c_utt.tolist(), c_cnt.tolist(), c_of.tolist()):
        key = scan.keys[g]
        rules[key].add("containment")
        k = int(index.utt_set[u])
        set_contained_utts[k].add(u)
        set_recordings[k].add(key)
        if len(heldout_hits[key]) < DETAIL_HELDOUT:
            heldout_hits[key].append({"set": index.sets[k].name, "id": index.utt_id[u], "matched": cnt, "of": of})

    per_set, offset = [], 0
    for k, s in enumerate(index.sets):
        n = len(s.ids)
        words = index.utt_words[offset:offset + n]
        nbs = nb[offset:offset + n]
        short = int((words < common.NGRAM).sum())
        empty = int(((words >= common.NGRAM) & (nbs == 0)).sum())
        recs_by_source = defaultdict(int)
        for src, _ in set_recordings[k] | set_identity[k]:
            recs_by_source[src] += 1
        per_set.append({"name": s.name, "path": s.path, "sha256": s.sha256, "utterances": n,
                        "checked": n - short - empty, "skipped_short": short, "skipped_empty": empty,
                        "contained_utterances": len(set_contained_utts[k]),
                        "containment_recordings": len(set_recordings[k]),
                        "identity_recordings": len(set_identity[k]),
                        "denylisted_recordings_by_source": dict(sorted(recs_by_source.items()))})
        offset += n

    by_source: dict[str, list[str]] = defaultdict(list)
    details = []
    for key in sorted(rules):
        by_source[key[0]].append(key[1])
        details.append({"source": key[0], "recording": key[1], "rules": [r for r in RULES if r in rules[key]],
                        "heldout": heldout_hits.get(key, [])})
    counts = {"by_rule": {r: sum(1 for v in rules.values() if r in v) for r in RULES},
              "by_source": {s: len(v) for s, v in sorted(by_source.items())},
              "total": len(rules), "boilerplate_8grams": int(boiler.sum()),
              "heldout_8grams": int(nh), "training_recordings_scanned": len(scan.keys),
              "contained_pairs": int(len(c_gid))}
    obj = {"params": {"ngram": common.NGRAM, "boilerplate_if_recordings_gt": common.BOILERPLATE_MAX,
                      "contain_fraction": common.CONTAIN_FRACTION,
                      "denominator": "distinct non-boilerplate 8-grams of the held-out utterance",
                      "normalizer": "finetune/parakeet-ternary/evaluate.py normalize",
                      "recording_keys": {"yodas": "original_audio_id",
                                         "peoples_speech": "row id without its '_<segment>.flac' suffix",
                                         "local corpora": "the manifest's `recording` field"}},
           "heldout": per_set, "counts": counts,
           "recordings": {s: sorted(v) for s, v in sorted(by_source.items())}}
    return obj, details
