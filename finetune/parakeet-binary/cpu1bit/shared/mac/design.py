"""Williams designs for the block schedule (plans/cpu-benchmark.md "Schedule", carried over by
plans/cpu-benchmark-mac.md "Design"). A replicate is one complete Williams design for the actual number
of arms k: k sequences if k is even, 2k if odd (the second half the mirror images), each sequence one
block per arm. Every arm appears once per position per replicate (row-complete for even k) and every
ordered pair of distinct arms is adjacent equally often (first-order carryover balance). The design is
generated and recorded before measurement; every block carries replicate, sequence and position IDs.
Sequence order within a replicate is shuffled with a frozen seed per replicate (recorded).
"""
from __future__ import annotations

import random


def williams_sequences(k: int) -> list[list[int]]:
    """Williams (1949) square(s) over arm indices 0..k-1."""
    if k < 1:
        raise ValueError("k >= 1")
    if k == 1:
        return [[0]]
    first, lo, hi = [0], 1, k - 1
    take_lo = True
    while len(first) < k:
        if take_lo:
            first.append(lo)
            lo += 1
        else:
            first.append(hi)
            hi -= 1
        take_lo = not take_lo
    rows = [[(x + i) % k for x in first] for i in range(k)]
    if k % 2 == 1:
        rows += [list(reversed(r)) for r in rows]
    return rows


def design(arms: list[str], replicates: int, seed: int, stage: str, condition: dict) -> list[dict]:
    """Blocks in execution order with their IDs."""
    seqs = williams_sequences(len(arms))
    blocks = []
    for r in range(replicates):
        order = list(range(len(seqs)))
        random.Random(seed + r).shuffle(order)
        for s_pos, s in enumerate(order):
            for p, a in enumerate(seqs[s]):
                blocks.append({"stage": stage, "replicate": r, "sequence": s, "sequence_order": s_pos, "position": p,
                               "arm": arms[a], "condition": condition,
                               "block_id": f"{stage}-r{r}-s{s}-p{p}-{arms[a]}"})
    return blocks


def balance(seqs: list[list[int]], k: int) -> dict:
    """Position counts per arm and adjacency counts per ordered pair (for tests and records)."""
    pos = [[0] * k for _ in range(k)]
    adj = [[0] * k for _ in range(k)]
    for s in seqs:
        for p, a in enumerate(s):
            pos[a][p] += 1
        for a, b in zip(s, s[1:]):
            adj[a][b] += 1
    return {"position_counts": pos, "adjacency_counts": adj}
