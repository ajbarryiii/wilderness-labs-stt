# Bend 2 batch planner

The relabel batch planner of `DESIGN.md`, written in Bend 2 (bend2 2.0.37 from the repo flake) as an
exploration of the language. It reproduces `reference.py` byte for byte, runs in parallel on the CPU,
and proves ten of its eleven laws in Bend's checker.

| File | What it is |
| --- | --- |
| `plan.bend` | The pure planner: validation, budget parsing, the per-pool sort and greedy cut, the parallel driver, and writing the plan into arrays. It imports no foreign code. |
| `planner.bend` | The CLI. It reads the inputs, validates everything, and only then opens and writes the outputs. |
| `effs/u32io.c`, `effs/u32io.js` | Two foreign effects, `U32s.read` and `U32s.write`. They move little-endian u32 words straight between a file and an `Array<U32>`, with no byte list. |
| `LAWS.bend` | The laws, stated over `plan.bend` in their own vocabulary. |
| `PROOF.bend` | The proofs, including the remaining exact-cover reduction. |
| `u32.bend` | General pure-Bend U32, Word and Nat arithmetic lemmas, with explicit overflow and positivity hypotheses. |
| `reference.py`, `tests/`, `DESIGN.md` | The contract, which is not modified here. |

## Build and run

```bash
nix develop <repo> -c bend planner.bend -o build/planner     # from finetune/bend-planner
build/planner SAMPLES POOLS BUDGET ORDER OFFSETS --threads 8 # bend 2.0.37: runtime options anywhere
build/planner +RTS --threads 8 -RTS SAMPLES ...              # bend >= 2.0.38
```

The runtime's option syntax changed in 2.0.38. On 2.0.37 the runtime consumes `--threads N`
wherever it appears, but the bare `+RTS` and `-RTS` tokens reach `IO.args`, so with `+RTS` the
planner prints its usage message. The installed guide (`bend guide`) shows the 2.0.37 form;
upstream's current guide shows `+RTS`.

On success the planner exits 0 and writes both files. On malformed input it exits 1 with
`planner: <reason>` on stderr, and writes nothing. Malformed means a length that is not a multiple
of 4, bad pool offsets, or a budget that is not one or more ASCII digits (leading zeros allowed)
with a value of at most 4294967295. A wrong argument count exits 2.
`bend planner.bend -o build/planner.js` builds the JS lane, whose output matched the native lane
on a small case.

Each input file must be under 4 GiB. Base's `File.size` in 2.0.37 answers a u32 and fails past
that with `EOVERFLOW`, so the planner exits 75. `DESIGN.md` records this limit. A 4 GiB input holds
2^30 entries. At the measured 56 bytes per clip, that is about 60 GB, all of this machine's
memory. Stage 2b's `samples.u32` is 4.8 MB, and full C1's would be about 24 MB.

One pool can hold at most about 60 million clips. The cut, the pool split and Base's `List.sort`
build their lists by plain (non-tail) recursion, and 2.0.37's native runtime keeps a frame of about
32 bytes per element on a fixed 2 GiB stack. The third review worked this out from the runtime
source: a single pool of 2^26 (67M) clips would hit the guard page, and single pools of 16M clips
passed. Real pools hold about 7,600 clips (128 batches), and all of C1 is about 6M.

## Tests

```
$ finetune/relabel/python -m unittest discover -s finetune/bend-planner/tests -v
test_c1_journal_128_batch_pools (test_contract.ContractTest.test_c1_journal_128_batch_pools) ... ok
test_edges_match_reference (test_contract.ContractTest.test_edges_match_reference) ... ok
test_random_matches_reference (test_contract.ContractTest.test_random_matches_reference) ... ok
test_rejects_malformed_input (test_contract.ContractTest.test_rejects_malformed_input) ... ok
test_reproduces_every_plan_on_disk (test_contract.ContractTest.test_reproduces_every_plan_on_disk)
Every v1 or v2 plan under batch-plan/ (v2 indexes entries rather than sequence numbers; same cut). ... ok
test_reproduces_stage_2b_plan (test_contract.ContractTest.test_reproduces_stage_2b_plan)
The relabel thread's v1 plan (training/batchplan.py): the cut files, checked against plan.json first. ... ok

Ran 6 tests in 0.606s

OK
```

## Design

The algorithm works on lists, so that the laws can talk about it. The steps are:

1. `planner.bend` reads each file with `U32s.read` into an `Array<U32>`. An array holds 2^d cells,
   with d the least depth that fits.
2. `Plan.list` turns each array into a list.
3. `Plan.check` validates the pool offsets.
4. `Plan.pools` cuts the numbered clips into pools in a single pass.
5. `Plan.tree` arranges the pools in a tree, halving them by count at each level.
6. `Plan.solve` sorts and cuts each leaf (`Plan.pool`) and handles the two halves of every node
   with a parallel let.
7. `Plan.run` walks the tree in order, writing `order` and `offsets` into arrays.
8. `U32s.write` writes the arrays out.

The cut test is the overflow-free form from `DESIGN.md`: `c > 0`, `s > 0` and `c + 1 > b / s`,
with c held as a `Nat`. A native `Nat` costs the same as a `U32` here. A micro-benchmark of the
test ran 100M iterations in 0.10 s with either counter.

An index is opaque to the planner: in plan format v1 it is a journal sequence number, and in v2 an
entry index (`DESIGN.md`). The planner never reads `entries.u32`.

## What is proven

All eleven laws are proven. Both commands print `ALL PROOFS CHECK`:

```bash
nix develop -c bend finetune/bend-planner/PROOF.bend
nix develop -c bend finetune/bend-planner/PROOF.bend --verdict
```

The arithmetic module checks on its own as well:

```bash
nix develop -c bend finetune/bend-planner/u32.bend
nix develop -c bend finetune/bend-planner/u32.bend --verdict
```

`--verdict` rechecks with the flake's `$BENDTT`, built from bend2 2.0.37's `bendtt.lean` with
exactly Lean 4.34.0. The tool also prints `bend 2.0.39 is available: run bend update`; that
message does not change the pinned 2.0.37 toolchain. `lean/` holds an independent Lean check
of the arithmetic and of the `Small` rewording below (`lean/PHASE2.md`, `lean/PHASE3.md`).

| `DESIGN.md` law | Law in `LAWS.bend` | Status |
| --- | --- | --- |
| 1, exact cover (per pool) | `pool_cover`: each clip occurs in a pool's batches exactly as often as in the pool | proven (`cut_cover` + `sort_perm`) |
| 1, entries stay in their pool | `pools_cover`: the pools, joined, are the clips 0 .. n-1 in sequence order | proven (round trip + symbolic `Small` bound) |
| | `pools_sizes`: pool k holds `ps[k+1] - ps[k]` clips | proven (validated subtraction + exact consumption) |
| 2, budget | `pool_budget`: a batch has one clip, or every clip x has `x.s = 0` or `len <= b / x.s` | proven (last-clip fit + sortedness + division antitonicity) |
| 3, greedy (order) | `pool_sorted`: the batches, joined, ascend in (samples, sequence) order | proven (`cut_cover` + `sort_sorted`) |
| 3, greedy (cut) | `pool_greedy`: the next batch's first clip x has `x.s > 0` and `len + 1 > b / x.s` | proven, for a cut of any list |
| (offsets increase) | `pool_filled`: no batch is empty | proven |
| (parallelism) | `solve_spec`: `leaves(solve(b, tree(d, k, xss))) == pools.cut(b, xss)` | proven |
| lemmas | `cut_cover`, `sort_perm`, `sort_sorted` (about Base's `List.sort`, fuel included) | proven |

`pool_budget` now follows from a cut invariant about each batch's last clip. Sortedness and
`U32.div_antitone` extend that fit to each earlier clip. The arithmetic proof establishes exact
quotient/remainder decomposition for Base's long division, including its high-bit shift. It
requires a positive divisor; Base defines `b / 0` to be zero. No multiplication of U32 batch
counts is used, so U32 multiplication overflow cannot invalidate the budget proof.

`pools_sizes` follows from monotone offsets, subtraction without underflow, a telescoping sum of
sizes, and exact `take`/`runs` consumption. It needs only
`to_nat(from_nat(n)) <= n`, which holds even when conversion wraps. The proof does not use `Small`, so the law
would also hold without it (`lean/PHASE2.md`); the exact-cover law does need it.

`pools_cover` is `pools_cover_roundtrip` (the planner reasoning, assuming the sample length
survives conversion to U32) plus `pools_cover_bounded`, which proves that condition when
`len(ss) <= Word.max_nat(32n)`. Here `Word.max_nat(0n) = 0n` and
`Word.max_nat(1n+p) = 1n + Nat.double(Word.max_nat(p))`, i.e. 2^n - 1.

**`Small` now states its bound as `Word.max_nat(32n)` instead of `4294967295n`.** This is the
only change to a law's wording since the first version, and it keeps the meaning:
`lean/BendModel/Phase3.lean` proves `small_propositions_equal` (the two conditions are the same
proposition) and `cover_contract_equivalent` (the quantified law holds with one exactly when it
holds with the other). The literal could not stay. bend2 keeps Nat unary, so any proof
relating `4294967295n` to the symbolic bound overflows the checker's stack; even
`{Word.max_nat(16n) == 65535n : Nat}` does. That is bendlang/bend#957, open and parked
upstream. The proof also avoids having the checker convert `Small`'s evidence into an equality,
which would normalize the closed bound. `small_bound` uses equality elimination
(`%h : {... == _ : Bool}` then `{==}`), so both sides stay the same symbolic term.

The exact-cover size restriction is real: 2^32 samples and offsets `[0, 0]` pass validation after
conversion wraps, yet produce one empty pool. That violates exact cover, while the pool's stated
size is still correctly zero.

The new module imports only Base and `nat.bend` from pure-Bend bend-mathlib, pinned to
`0xcd4682caa53da19986acd472c2181473`. Its public lemmas cover comparison and order, subtraction
with explicit borrow/no-underflow conditions, symbolic width bounds, increment/conversion with
and without wraparound, modular addition, exact shifts including overflow, long-division
correctness, and quotient antitonicity. Each law has a proof; helper definitions contain no
holes, unsafe code or foreign imports. Their full statements are listed below.

No law covers the array and list glue (`Plan.list`, `out.*`), the effects, budget parsing or the
validation messages. The contract tests check those.

## Measurements

The benchmark has two inputs:

- **Stage 2b's v1 plan** (`relabel/training/batch-plan/w12-7ff17f270fd6-p128-n156`, read-only):
  n = 1,196,222, 156 pools, 16,515 batches. This is the production input.
- **All C1 journals present on 2026-10-24**: 1,289 shards, n = 3,844,374, 504 pools of 128
  batches. These inputs are `$DATA_ROOT/bend-planner/c1-all.*.u32`, built with
  `journal_samples` and `pools_by_batches(..., 128)`.

Each time is the median of 3 runs, and peak RSS comes from GNU time. The reference times include
Python and numpy start-up (about 0.1 s). The machine has 32 threads, and a C1 job was running
alongside. Each of the 36 Bend runs was byte-identical to its comparison: on stage 2b, to the
plan's own `order.u32` and `offsets.u32` (which `reference.py` also reproduced), and on the
journals, to `reference.py`.

| Program | Threads | Stage 2b wall | Stage 2b RSS | All journals wall | All journals RSS |
| --- | --- | --- | --- | --- | --- |
| `reference.py` | 1 | 0.40 s | 91 MB | 1.16 s | 227 MB |
| Bend native | 1 | 0.31 s | 75 MB | 1.00 s | 217 MB |
| Bend native | 2 | 0.25 s | 75 MB | 0.70 s | 217 MB |
| Bend native | 4 | 0.15 s | 76 MB | 0.50 s | 217 MB |
| Bend native | 8 | 0.11 s | 78 MB | 0.35 s | 219 MB |
| Bend native | 16 | 0.10 s | 82 MB | 0.34 s | 223 MB |
| Bend native | 32 | 0.07 s | 87 MB | 0.37 s | 230 MB |

On the first 40 shards (n = 117,974, 16 pools), the reference took 0.09 s and 38 MB, and Bend took
0.02 s and 9 MB. Those runs were also byte-identical.

On all journals the speed-up stops at about 3x by 8 threads (4.4x by 32 threads on stage 2b),
because of the serial phases. These are reading the
arrays into lists (0.05 s at 3.7M clips), splitting the pools and writing the output. The sort is
about 1.0 s of single-thread time, and Bend runs it in parallel. An earlier split, which used
`List.take`/`List.drop` on a shared list, cost 0.19 s. The one-pass `take`/`runs` split replaced
it, which took the single-thread time from 1.2 s to 1.0 s and the 8-thread time from 0.50 s to
0.36 s. A hand-written merge sort, choosing its branch with `pick` on shared lists, ran about 3x
slower than Base's `List.sort`.

## Bend pain points

- **`match` takes only a parameter or a pattern-bound variable.** A `let` may not come before a
  match on a parameter, and destructuring a computed pair, as in `(a, b) = f(x)`, counts as a
  match. Mutual recursion is refused too, so a recursive def cannot hand its recursive result to a
  helper that recurses. The workarounds are to pass the computed pair to the next call as a
  parameter (`take`/`runs`, `list.go`), or to choose the next state with small `Bool` helpers
  (`cut.next`/`cut.step`).
- **Evaluation is strict, and there is no `if`.** Passing both branches as arguments computes both.
- **Order matters.** A def used before its definition is "an unfilled law", which happened often
  in `PROOF.bend`. Pairs (`A & B`) are `Type`, so they cannot be `+`. `Holds` had to return `Data`
  before a proof could be reused.
- **Rewrites run one way.** `%e : P` with `e : {a == b}` replaces b with a, so most lemmas were
  stated in both orientations or wrapped in `Equal.sym`.
- **Import paths cannot pass through a symlink whose target contains a dot** (`github.com`). A
  scratch file that imports `plan.bend` therefore needs a copy, not a link.
- **A foreign effect that the program never calls has no `CID`.** Each registration needs
  `#ifdef CID(...)`. The JS twin is a few lines with `require("fs")`.
- **Upstream's docs describe a newer runtime** than the flake's 2.0.37, notably the runtime
  options (see above).
- **`--verdict` builds its kernel only with exactly Lean 4.34.0**, which nixpkgs never shipped
  (4.30.0, then 4.34.1). Lean's bundled `leanc` also links for generic Linux, so a kernel bend
  builds itself does not start on NixOS. The flake packages the 4.34.0 release binary, builds and
  patches the kernel, and sets `$BENDTT`.
- **Proofs about Base's `List.sort` need a fuel argument.** The proof shows that n passes and
  merges with fuel n are enough, which takes about 500 lines.

## Contract notes

- The budget is ASCII digits only. `reference.py` used to accept non-ASCII decimal digits such as
  `٣` (through `str.isdigit()` and `int()`). The first review found this, and the contract owner
  tightened `reference.py` and `DESIGN.md` to ASCII. Two REJECT cases now cover it: a non-ASCII
  digit and an empty budget.
- The laws need n < 2^32 (`Small` above). This is a fact of the file format, not a defect.
- `DESIGN.md` now records the 4 GiB limit on each input file.

## Arithmetic lemma statements

All 41 new public laws in `u32.bend` are stated here verbatim. `N` is the hash-pinned Nat
library; its `le` and `lt` are true Boolean comparisons. `Nat.bit`, `Nat.sub_ok`,
`Nat.sub_carry`, `Word.Shift`, `U32.DivSpec` and the other helper
definitions are documented in `u32.bend`. `Word.max_nat(n)` is the symbolic 2^n - 1 bound;
no equality between its width-32 instance and the concrete maximum literal is assumed.

```bend
law Nat.cmp_bit:
  for x: Nat
  for y: Nat
  for +a: Bool
  for +b: Bool
  {Nat.cmp(Nat.bit(a, x), Nat.bit(b, y)) == Word.cmp.fin(a, b, Nat.cmp(x, y)) : Cmp}

law Word.cmp_to_nat:
  for +n: Nat
  for a: Word(n)
  for b: Word(n)
  {Word.cmp(n, a, b) == Nat.cmp(Word.to_nat(n, a), Word.to_nat(n, b)) : Cmp}

law U32.cmp_to_nat:
  for a: U32
  for b: U32
  {U32.cmp(a, b) == Nat.cmp(U32.to_nat(a), U32.to_nat(b)) : Cmp}

law U32.le_trans:
  for +a: U32
  for +b: U32
  for +c: U32
  for ab: {U32.is_le(a, b) == True{} : Bool}
  for bc: {U32.is_le(b, c) == True{} : Bool}
  {U32.is_le(a, c) == True{} : Bool}

law Nat.sub_bit:
  for x: Nat
  for y: Nat
  for +a: Bool
  for +b: Bool
  for +c: Bool
  for h: Nat.sub_ok(Bool.sub_carry(a, b, c), x, y)
  {Nat.bit(Bool.sub_low(a, b, c), Nat.sub_carry(Bool.sub_carry(a, b, c), x, y))
    == Nat.sub_carry(c, Nat.bit(a, x), Nat.bit(b, y)) : Nat}

law Word.to_nat_sub_carry:
  for +n: Nat
  for +a: Word(n)
  for +b: Word(n)
  for +c: Bool
  for +h: Nat.sub_ok(c, Word.to_nat(n, a), Word.to_nat(n, b))
  {Word.to_nat(n, Word.adc(n, a, b, True{}, c))
    == Nat.sub_carry(c, Word.to_nat(n, a), Word.to_nat(n, b)) : Nat}

law U32.to_nat_sub:
  for +a: U32
  for +b: U32
  for h: {U32.is_le(b, a) == True{} : Bool}
  {U32.to_nat(U32.sub(a, b)) == Nat.sub(U32.to_nat(a), U32.to_nat(b)) : Nat}

law Word.to_nat_bound:
  for +n: Nat
  for a: Word(n)
  {Nat.is_le(Word.to_nat(n, a), Word.max_nat(n)) == True{} : Bool}

law Nat.cmp_double:
  for x: Nat
  for y: Nat
  {Nat.cmp(x, y) == Nat.cmp(Nat.double(x), Nat.double(y)) : Cmp}

law Word.to_nat_inc:
  for +n: Nat
  for +a: Word(n)
  for h: {Nat.is_lt(Word.to_nat(n, a), Word.max_nat(n)) == True{} : Bool}
  {Word.to_nat(n, Word.inc(n, a)) == 1n+Word.to_nat(n, a) : Nat}

law U32.to_nat_bound:
  for +a: U32
  {Nat.is_le(U32.to_nat(a), Word.max_nat(32n)) == True{} : Bool}

law U32.to_nat_inc:
  for +a: U32
  for h: {Nat.is_lt(U32.to_nat(a), Word.max_nat(32n)) == True{} : Bool}
  {U32.to_nat(U32.inc(a)) == 1n+U32.to_nat(a) : Nat}

law Word.to_nat_zero:
  for n: Nat
  {Word.to_nat(n, Word.zero(n)) == 0n : Nat}

law Word.to_nat_from_nat:
  for +m: Nat
  for +n: Nat
  for +h: {Nat.is_le(m, Word.max_nat(n)) == True{} : Bool}
  {Word.to_nat(n, Word.from_nat(m, n)) == m : Nat}

law U32.from_nat_word:
  for +n: Nat
  {U32{Word.from_nat(n, 32n)} == U32.from_nat(n) : U32}

law U32.to_nat_from_nat:
  for +n: Nat
  for h: {Nat.is_le(n, Word.max_nat(32n)) == True{} : Bool}
  {U32.to_nat(U32.from_nat(n)) == n : Nat}

law Word.adc_carry:
  for +n: Nat
  for a: Word(n)
  for b: Word(n)
  {Word.adc(n, a, b, False{}, True{}) == Word.inc(n, Word.adc(n, a, b, False{}, False{})) : Word(n)}

law Word.add_inc:
  for +n: Nat
  for +a: Word(n)
  for +b: Word(n)
  {Word.add(n, a, Word.inc(n, b)) == Word.inc(n, Word.add(n, a, b)) : Word(n)}

law Word.add_zero:
  for +n: Nat
  for +a: Word(n)
  {Word.add(n, a, Word.zero(n)) == a : Word(n)}

law U32.add_one:
  for +a: U32
  {U32.add(a, 1) == U32.inc(a) : U32}

law U32.add_inc:
  for +a: U32
  for +b: U32
  {U32.add(a, U32.inc(b)) == U32.inc(U32.add(a, b)) : U32}

law U32.add_zero:
  for +a: U32
  {U32.add(a, 0) == a : U32}

law U32.add_from_nat_succ:
  for +a: U32
  for +n: Nat
  {U32.add(a, U32.from_nat(1n+n)) == U32.add(U32.add(a, 1), U32.from_nat(n)) : U32}

law Word.to_nat_inc_le:
  for +n: Nat
  for +a: Word(n)
  {Nat.is_le(Word.to_nat(n, Word.inc(n, a)), 1n+Word.to_nat(n, a)) == True{} : Bool}

law U32.to_nat_inc_le:
  for +a: U32
  {Nat.is_le(U32.to_nat(U32.inc(a)), 1n+U32.to_nat(a)) == True{} : Bool}

law U32.to_nat_from_nat_le:
  for +n: Nat
  {Nat.is_le(U32.to_nat(U32.from_nat(n)), n) == True{} : Bool}

law U32.to_nat_le:
  for +a: U32
  for +b: U32
  for h: {U32.is_le(a, b) == True{} : Bool}
  {Nat.is_le(U32.to_nat(a), U32.to_nat(b)) == True{} : Bool}

law U32.to_nat_eq:
  for +a: U32
  for +b: U32
  for h: {U32.is_eq(a, b) == True{} : Bool}
  {U32.to_nat(a) == U32.to_nat(b) : Nat}

law Nat.bit_add:
  for c: Bool
  for +x: Nat
  for +y: Nat
  {Nat.bit(c, Nat.add(x, y)) == Nat.add(Nat.bit(c, x), Nat.double(y)) : Nat}

law Word.shift_spec:
  for +n: Nat
  for +c: Bool
  for +w: Word(n)
  Word.Shift(n, c, w, Word.shl.out(n, c, w))

law Word.max_nat_mono:
  for +n: Nat
  for +m: Nat
  for h: N.le(n, m)
  N.le(Word.max_nat(n), Word.max_nat(m))

law U32.of_nat_le:
  for +a: U32
  for +b: U32
  for h: N.le(U32.to_nat(a), U32.to_nat(b))
  {U32.is_le(a, b) == True{} : Bool}

law U32.ge_nat:
  for +a: U32
  for +b: U32
  {U32.is_ge(a, b) == Nat.is_le(U32.to_nat(b), U32.to_nat(a)) : Bool}

law U32.divmod_spec:
  for +m: Nat
  for +a: Word(m)
  for +b: U32
  for +width: N.le(m, 32n)
  for positive: N.lt(0n, U32.to_nat(b))
  U32.DivSpec(m, a, b, U32.divmod.go(m, a, b))

law U32.zero_nat:
  for +a: U32
  {U32.is_zero(a) == Nat.is_eq(U32.to_nat(a), 0n) : Bool}

law U32.to_nat_div:
  for +a: U32
  for +b: U32
  for +positive: N.lt(0n, U32.to_nat(b))
  {U32.to_nat(U32.div(a, b)) == Nat.div(U32.to_nat(a), U32.to_nat(b)) : Nat}

law Nat.div_antitone:
  for +a: Nat
  for +s: Nat
  for +t: Nat
  for st: N.le(s, t)
  for positive: N.lt(0n, s)
  N.le(Nat.div(a, t), Nat.div(a, s))

law U32.to_nat_lt:
  for +a: U32
  for +b: U32
  for h: {U32.is_lt(a, b) == True{} : Bool}
  N.lt(U32.to_nat(a), U32.to_nat(b))

law U32.div_antitone:
  for +a: U32
  for +s: U32
  for +t: U32
  for st: {U32.is_le(s, t) == True{} : Bool}
  for positive: {U32.is_lt(0, s) == True{} : Bool}
  {U32.is_le(U32.div(a, t), U32.div(a, s)) == True{} : Bool}

law U32.positive_of_nonzero:
  for +a: U32
  for h: {U32.is_zero(a) == False{} : Bool}
  N.lt(0n, U32.to_nat(a))

law U32.zero_add:
  for +a: U32
  {U32.add(0, a) == a : U32}
```
