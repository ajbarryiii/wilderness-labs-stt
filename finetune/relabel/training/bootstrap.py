"""The paired cluster bootstrap (b0prime plan "Intervals", decoder-only plan "Results").

Per set, the utterances of every model are aligned by id (the same utterances, scored records only); a
SetTable holds per cluster the reference words and each model's errors (S + D + I). One replicate draws the
set's clusters with replacement (as many as it has), sums words and errors over the drawn clusters, and computes
each model's corpus WER; the mean over `mean_sets` (H4) is recomputed per replicate from the same draws. A
contrast (a, b) is WER(a) - WER(b) in points; its interval is the central `ci` percentile range over the
replicates. Clusters, not utterances, are resampled (utterances of a speaker or meeting are correlated); the same
draw serves both models of a contrast (paired). Each set has its own seeded stream (SeedSequence([seed, set
index]) in sorted set order), so results do not depend on dictionary order.
"""
from __future__ import annotations

import dataclasses

import numpy as np

REPLICATES = 10_000
CI = 0.90
SEED = 20261022


@dataclasses.dataclass
class SetTable:
    clusters: list[str]                  # sorted cluster names
    words: np.ndarray                    # [clusters] reference words
    errors: dict[str, np.ndarray]        # model -> [clusters] errors
    utterances: int

    @classmethod
    def from_records(cls, records: dict[str, list[dict]], cluster_of, ids: set[str] | None = None) -> "SetTable":
        """Records (evaluate.py per-utterance layout) of each model on one set, restricted to `ids` if given."""
        models = sorted(records)
        keyed = {m: {r["id"]: r for r in records[m]} for m in models}
        first = keyed[models[0]]
        for m in models[1:]:
            if set(keyed[m]) != set(first):
                raise ValueError(f"model {m} does not cover the same utterances as {models[0]}")
        chosen = sorted(i for i in first if (ids is None or i in ids) and first[i].get("scored", True))
        for m in models:
            for i in chosen:
                if not keyed[m][i].get("scored", True):
                    raise ValueError(f"{i}: scored for {models[0]} but not for {m}")
                if keyed[m][i]["ref_norm"] != first[i]["ref_norm"]:
                    raise ValueError(f"{i}: references differ between models")
        names = sorted({cluster_of(i) for i in chosen})
        index = {c: j for j, c in enumerate(names)}
        idx = np.array([index[cluster_of(i)] for i in chosen], dtype=np.int64)
        words = np.bincount(idx, weights=[len(first[i]["ref_norm"].split()) for i in chosen], minlength=len(names))
        errors = {m: np.bincount(idx, weights=[keyed[m][i]["S"] + keyed[m][i]["D"] + keyed[m][i]["I"] for i in chosen],
                                 minlength=len(names)) for m in models}
        return cls(names, words, errors, len(chosen))

    def wer(self, model: str) -> float:
        return 100.0 * float(self.errors[model].sum()) / float(self.words.sum())


def _ci(samples: np.ndarray, level: float) -> list[float]:
    lo, hi = np.percentile(samples, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return [float(lo), float(hi)]


def paired(tables: dict[str, SetTable], contrasts: dict[str, tuple[str, str]], mean_sets: list[str],
           replicates: int = REPLICATES, seed: int = SEED, ci: float = CI) -> dict:
    """Point estimates and intervals of every contrast per set and for the mean over mean_sets (in points)."""
    models = sorted({m for pair in contrasts.values() for m in pair})
    names = sorted(tables)
    if not set(mean_sets) <= set(names):
        raise ValueError(f"mean sets {mean_sets} not all in {names}")
    rep: dict[str, dict[str, np.ndarray]] = {}
    for s_index, name in enumerate(names):
        t = tables[name]
        rng = np.random.default_rng(np.random.SeedSequence([seed, s_index]))
        n = len(t.clusters)
        counts = np.empty((replicates, n), dtype=np.float64)
        for b in range(replicates):
            counts[b] = np.bincount(rng.integers(0, n, n), minlength=n)
        words = counts @ t.words
        rep[name] = {m: 100.0 * (counts @ t.errors[m]) / words for m in models}
    out = {"replicates": replicates, "seed": seed, "ci_level": ci, "mean_sets": list(mean_sets),
           "wer": {name: {m: tables[name].wer(m) for m in models} for name in names},
           "clusters": {name: len(tables[name].clusters) for name in names},
           "utterances": {name: tables[name].utterances for name in names},
           "reference_words": {name: int(tables[name].words.sum()) for name in names},
           "sets": {name: {} for name in names}, "mean": {}}
    out["wer"]["mean"] = {m: float(np.mean([out["wer"][s][m] for s in mean_sets])) for m in models}
    for c, (a, b) in contrasts.items():
        for name in names:
            out["sets"][name][c] = {"point": out["wer"][name][a] - out["wer"][name][b],
                                    "ci": _ci(rep[name][a] - rep[name][b], ci)}
        diff = np.mean([rep[s][a] - rep[s][b] for s in mean_sets], axis=0)
        out["mean"][c] = {"point": out["wer"]["mean"][a] - out["wer"]["mean"][b], "ci": _ci(diff, ci)}
    return out
