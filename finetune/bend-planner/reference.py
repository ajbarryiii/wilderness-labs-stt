"""Reference batch planner (DESIGN.md "Contract"): the behaviour the Bend planner must reproduce byte for byte.

    order, offsets = plan(samples, pools, budget)
    write_u32(path, order)

samples[i] is occurrence i's length in samples; pools holds n_pools + 1 start offsets. Each pool is stable-sorted
by (samples, sequence number) and cut greedily: a new batch starts before clip i when the current batch is
non-empty and len(current) + 1 > budget // samples[i] (the overflow-free form of (len + 1) * samples > budget).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
U32_MAX = 2**32 - 1


def validate(samples: np.ndarray, pools: np.ndarray, budget: int) -> None:
    n = len(samples)
    if not 0 <= budget <= U32_MAX:
        raise ValueError(f"budget {budget} is not a u32")
    if len(pools) < 1 or pools[0] != 0 or pools[-1] != n:
        raise ValueError(f"pools must start at 0 and end at n={n}")
    if np.any(np.diff(pools.astype(np.int64)) < 0):
        raise ValueError("pool offsets decrease")


def plan(samples: np.ndarray, pools: np.ndarray, budget: int) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(samples, dtype=np.uint32)
    pools = np.asarray(pools, dtype=np.uint32)
    validate(samples, pools, budget)
    order: list[int] = []
    offsets = [0]
    for lo, hi in zip(pools[:-1].tolist(), pools[1:].tolist()):
        seq = sorted(range(lo, hi), key=lambda i: (int(samples[i]), i))
        count = 0
        for i in seq:
            s = int(samples[i])
            if count and s > 0 and count + 1 > budget // s:
                offsets.append(len(order))
                count = 0
            order.append(i)
            count += 1
        if count:
            offsets.append(len(order))
    return np.asarray(order, dtype=np.uint32), np.asarray(offsets, dtype=np.uint32)


def padding(samples: np.ndarray, order: np.ndarray, offsets: np.ndarray) -> float:
    """1 - real samples / padded samples (count x longest), over all batches."""
    real = padded = 0
    for lo, hi in zip(offsets[:-1].tolist(), offsets[1:].tolist()):
        batch = samples[order[lo:hi]].astype(np.int64)
        real += int(batch.sum())
        padded += len(batch) * int(batch.max())
    return 1.0 - real / padded if padded else 0.0


def read_u32(path: str | Path) -> np.ndarray:
    raw = Path(path).read_bytes()
    if len(raw) % 4:
        raise ValueError(f"{path}: length {len(raw)} is not a multiple of 4")
    return np.frombuffer(raw, dtype="<u4").astype(np.uint32)


def write_u32(path: str | Path, values) -> None:
    Path(path).write_bytes(np.asarray(values, dtype="<u4").tobytes())


def journal_samples(journal_paths: list[str | Path]) -> tuple[np.ndarray, np.ndarray]:
    """(samples, consumed batch index) per C1 journal occurrence, in (batch, pos) order across the given shards.

    Refuses durations that are not whole samples, so the integer key loses nothing."""
    import pyarrow.parquet as pq
    samples, batches = [], []
    for path in journal_paths:
        t = pq.read_table(path, columns=["batch", "pos", "duration"]).to_pandas().sort_values(["batch", "pos"])
        exact = t["duration"].to_numpy() * SAMPLE_RATE
        rounded = np.rint(exact)
        if np.any(np.abs(exact - rounded) > 1e-6):
            raise ValueError(f"{path}: a duration is not a whole number of samples")
        samples.append(rounded.astype(np.uint32))
        batches.append(t["batch"].to_numpy().astype(np.int64))
    if not samples:
        return np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.int64)
    batch = np.concatenate(batches)
    if np.any(np.diff(batch) < 0):
        raise ValueError("journal shards are not in consumed-batch order")
    return np.concatenate(samples), batch


def pools_by_batches(batch: np.ndarray, pool_batches: int) -> np.ndarray:
    """Pool offsets grouping consumed batches [k * pool_batches, (k + 1) * pool_batches), counted from batch[0]."""
    if len(batch) == 0:
        return np.zeros(1, dtype=np.uint32)
    pool = (batch - batch[0]) // pool_batches
    starts = np.flatnonzero(np.diff(pool)) + 1
    return np.concatenate([[0], starts, [len(batch)]]).astype(np.uint32)


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print("usage: reference.py SAMPLES POOLS BUDGET ORDER OFFSETS", file=sys.stderr)
        return 2
    try:
        samples, pools = read_u32(argv[0]), read_u32(argv[1])
        if not (argv[2].isascii() and argv[2].isdigit()):
            raise ValueError(f"budget {argv[2]!r} is not ASCII decimal digits")
        order, offsets = plan(samples, pools, int(argv[2]))
    except ValueError as exc:
        print(f"reference.py: {exc}", file=sys.stderr)
        return 1
    write_u32(argv[3], order)
    write_u32(argv[4], offsets)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
