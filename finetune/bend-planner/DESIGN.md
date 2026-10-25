# Bend 2 batch planner (exploration)

An offline batch planner for the relabel training replay, written in Bend 2
(https://github.com/bendlang/bend), with a Python reference that defines its
behaviour. It is an exploration of Bend and sits off the critical path: stage
2b uses the Python re-cut in `finetune/relabel/training/replay.py`. The Bend
planner only replaces it if it reproduces the Python plan byte for byte.

## Why

Everything after C1 reads C1's local shards, and a clip's label does not depend
on its batch, so batches can be re-cut once, before training. The current
8-batch sorting pool leaves 14.0% of each batch as padding, measured on 2,000 C1
batches. Pools of 128 batches cut that to about 1.3%. Planning is a pure
function from durations to batches, which suits Bend: pools are independent
(parallel), and the guarantees we care about can be stated as laws.

## Contract

The plan-file format is owned by the relabel thread and may change; this
document tracks it. Version 1 is final as of 2026-10-24, after a clean plan
review and a clean Codex code review, and is implemented in
`finetune/relabel/training/batchplan.py`.
- Plans are written to
  `$RELABEL_ROOT/training/batch-plan/<C1 config id>-p128-n<pools>/`.
- The shared contract is the cut below (`samples.u32`, `pools.u32` → `order.u32`,
  `offsets.u32`). The directory also holds files the Bend planner does not
  produce:
  - `batch_pools.u32` (`n_pools + 1`): the first batch of each pool, derivable
    from the cut.
  - `shuffle.u32` (`n_batches`): the Python-only seeded batch order, numpy
    `default_rng([20261030, 2, p]).permutation` per pool.
  - `plan.json`: the SHA-256 of every file, plus the journal shards' SHA-256s.
- Stage 2b's plan (`w12-7ff17f270fd6-p128-n156`): 156 pools, 19,968 consumed
  batches → 16,515 batches, 1.57% padding. `reference.py` reproduces its
  `order.u32` and `offsets.u32` byte for byte, and
  `test_reproduces_stage_2b_plan` checks the binary against them.
- v2 (final 2026-10-24, synthetic arms only; v1 is unchanged and still used by
  the no-synthetic arms): stage 2c's synthetic clips join each pool before the
  cut, rather than being appended after it. Appending them inflated padded
  audio 1.73x on average. Details:
  - Each pool lists its real journal occurrences in sequence order, then its
    synthetic draw in draw order.
  - `entries.u32` maps each entry to a source: `v < 2^31` is journal sequence
    number v, and `v >= 2^31` is synthetic clip `v - 2^31`.
  - `samples`, `pools`, `order` and `offsets` index entries instead of
    sequence numbers.
  - A synthetic key is `ceil(samples * 10 / 9)`, its longest speed-perturbed
    length.
  - The cut is the v1 cut, with ties broken by entry index. The Bend planner
    needs no change: it never interprets an index, and it never reads
    `entries.u32` or the keys' origin.
  - No v2 plan exists on disk yet. `test_reproduces_every_plan_on_disk` covers
    every v1 or v2 plan once one is written.

Its decisions that sit outside the planner:
- Pools are P = 128 consumed batches, counted from batch 0. Only complete pools
  are included.
- Labeling-rejected rows are dropped after the cut, at read time. The plan is
  therefore independent of the teacher, and all arms share identical batches.
- v1 appends synthetic extras per planned batch, after the cut; v2 moves them
  into the pools before the cut.
- A repeated id is a separate journal occurrence and appears once per
  occurrence, by sequence number.
- If `round(duration * 16000)` is not exact for some journal row, the key
  becomes C1's stored sample count. `samples.u32` keeps its format either way.

**Inputs** (little-endian u32 arrays):
- `samples.u32`: one entry per journal occurrence, in (batch, pos) order across
  shards. The entry is `round(duration * 16000)`, the clip length in samples.
  Its index is the occurrence's sequence number.
- `pools.u32`: `n_pools + 1` pool start offsets into the sequence. It must start
  at 0, never decrease, and end at `n`. Pool k holds sequence numbers
  `[pools[k], pools[k+1])`.
- The budget, in samples (`600 * 16000 = 9,600,000` for training).

**Cut**, for each pool independently:
1. Stable-sort the pool's sequence numbers by `(samples, sequence number)`.
2. Walk them in that order. Start a new batch before clip `i` whenever the
   current batch is non-empty and `(len(current) + 1) * samples[i] > budget`.
   This is `train.cut_pool`, computed in integers instead of floats.
3. A clip longer than the budget therefore forms its own batch. An empty pool
   produces no batches.

Batches are emitted pool by pool, in cut order (shortest clips first within a
pool). The seeded shuffle of batch order is a separate Python step and not part
of this contract, because numpy's generator cannot be reproduced in Bend.

**Outputs** (little-endian u32 arrays):
- `order.u32`: the `n` sequence numbers in planned order.
- `offsets.u32`: `n_batches + 1` batch start offsets into `order`, from 0 to `n`.

**CLI**: `planner SAMPLES POOLS BUDGET ORDER OFFSETS`.
- Success: exit 0, both output files written.
- Malformed input (a file length that is not a multiple of 4, bad pool offsets,
  a budget that is not one or more ASCII digits `0-9` with a value below 2^32):
  non-zero exit with a message on stderr.
- Limit: an input file of 4 GiB or more is refused with a non-zero exit, because
  Bend 2.0.37's `File.size` is u32. That's about 1.07 billion occurrences; full
  C1 is about 6 million, so `samples.u32` is about 24 MB.

Overflow: `(len + 1) * samples[i]` can exceed u32. Many short clips followed by
one long clip is enough. Both implementations therefore evaluate the
equivalent test `samples[i] > 0 and len(current) + 1 > budget // samples[i]`,
which never overflows. For positive `s`, `(n + 1) * s > B` holds exactly when
`n + 1 > floor(B / s)`. A zero-sample clip never starts a batch.

## Laws (Bend side)

Stated in `LAWS.bend` and proven in `PROOF.bend` where feasible:
1. **Exact cover**: `order` is a permutation of `0 .. n-1`, and each pool's
   entries stay inside that pool.
2. **Budget**: every batch has one clip, or `count * max(samples) <= budget`
   (computed in Nat, or with the division form).
3. **Greedy**: within a pool, batches are in non-decreasing sample order, and
   adding the next batch's first clip to a batch would break the budget.

Laws 1 to 3 together pin the output to the reference. The contract tests below
are the acceptance criteria. A law the checker cannot yet prove is listed as
unproven in the README, not weakened. Foreign file effects make `--verdict`
fail, so the I/O lives in its own module and no file that holds laws imports it.

## Acceptance (contract tests, `tests/test_contract.py`)

1. **Reference equality**: the binary's output files are byte-identical to
   `reference.py` on seeded random inputs (varied n, pool sizes and duration
   distributions, ties included) and on the edge cases: n = 0, one clip, all
   durations equal, a clip longer than the budget, empty pools.
2. **Invariants**: laws 1 to 3, checked in Python on every output.
3. **Rejection**: each malformed input exits non-zero and writes no output.
4. **Real data** (needs `$RELABEL_ROOT`; skipped when it doesn't hold the data):
   - On the first 40 C1 shards, the reference with 128-batch pools leaves under
     2% padding (1.61% measured, the same in both threads), and the binary
     matches the reference byte for byte.
   - On stage 2b's v1 plan, the binary reproduces `order.u32` and `offsets.u32`
     byte for byte, after the inputs and outputs are checked against
     `plan.json`'s SHA-256s.
   - Padding is `1 - sum(samples) / sum(count * max)` over all batches.

Reported but not a contract: wall time and peak memory for the full C1 journal,
Bend native (`--threads N` on 2.0.37; `+RTS --threads N -RTS` from 2.0.38)
against the Python reference. Measured 2026-10-24 (README "Measurements").

## Build

`nix develop` (repo-root flake: nixpkgs-unstable `bend2`), then
`bend planner.bend -o build/planner`. `build/` is gitignored.
