"""Consumed batches of the pinned stream: train.Batcher's sequence, synchronously and with whole items.

finetune/parakeet-ternary/train.py Batcher (the main runs' batching) collates items into padded tensors and
keeps only audio, texts, ids and sources. The labeling pass also needs each item's reference and its kind,
so this module reproduces Batcher's sequence on the item level: pools of at least pool_batches x
batch_seconds of 1-30 s audio, read from the stream with the stream state recorded before each pool;
train.cut_pool; batch order permuted with default_rng([seed, 2, pool]); positions in Batcher's format
{"pool_state", "pool", "consumed"}, so a position from either resumes the other. The contract test
(tests/test_labeling_batches.py) checks that the id sequence and the positions equal train.Batcher's.
"""
from __future__ import annotations

import copy
from collections.abc import Iterator

import numpy as np


def batch_settings():
    import train  # finetune/parakeet-ternary
    return train.cut_pool, train.MIN_SECONDS, train.MAX_SECONDS, train.POOL_BATCHES


def iter_batches(stream, batch_seconds: float, seed: int, position: dict | None = None,
                 pool_batches: int | None = None, prefiltered: dict | None = None) -> Iterator[tuple[list[dict], dict]]:
    """Yields (items of one consumed batch, position after it), forever, in train.Batcher's order."""
    cut_pool, lo, hi, default_pool = batch_settings()
    pool_seconds = (pool_batches or default_pool) * batch_seconds
    pool, skip = 0, 0
    if position is not None:
        stream.load_state_dict(copy.deepcopy(position["pool_state"]))
        pool, skip = position["pool"], position["consumed"]
    iterator = iter(stream)
    while True:
        pool_state = copy.deepcopy(stream.state_dict())
        items, seconds = [], 0.0
        while seconds < pool_seconds:
            item = next(iterator)
            d = float(item["duration"])
            if d < lo or d > hi:
                if prefiltered is not None:
                    key = "short" if d < lo else "long"
                    prefiltered[key] = prefiltered.get(key, 0) + 1
                continue
            items.append(item)
            seconds += d
        next_state = copy.deepcopy(stream.state_dict())
        cut = cut_pool([float(it["duration"]) for it in items], batch_seconds)
        order = np.random.default_rng([seed, 2, pool]).permutation(len(cut))
        batches = [cut[j] for j in order]
        for j, idx in enumerate(batches):
            if j < skip:
                continue
            last = j + 1 == len(batches)
            after = ({"pool_state": next_state, "pool": pool + 1, "consumed": 0} if last
                     else {"pool_state": pool_state, "pool": pool, "consumed": j + 1})
            yield [items[i] for i in idx], after
        pool, skip = pool + 1, 0
        del items
