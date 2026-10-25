# Independent Lean verification: Phase 1

The latest [Phase 3 report](PHASE3.md) reviews the completed eleven-law proof,
the exact Small/Holds equivalence, and the new kernel mutation controls.

The subsequent [Phase 2 report](PHASE2.md) verifies all 41 new arithmetic laws,
records the mutations and exact Nat dependencies, and reports the remaining
planner-cover blocker. This page preserves the Phase 1 results.

AJ, the structural Lean model agrees with both Bend execution lanes on every
completed numeric comparison. The requested full-width checker conversion
coverage remains incomplete because the checker overflows its stack or exceeds
the probe time limit. These failures are recorded, not counted as passes.

This review uses Bend **2.0.37**, Lean/Lake **4.34.0**, and the installed Base.
It uses core Lean and Std, with no Mathlib and no imported Bend proofs.
All review files are confined to this directory; generated Bend fixtures and
large raw outputs are under `$BEND_LEAN_SCRATCH/`. No planner files were
edited and nothing was committed, pushed, or published.

Machine-specific paths in the committed reports and `results/` are replaced with
placeholders: `<repo>` for the checkout, `<nix-store>/` for a store path,
`<scratch>` for `$BEND_LEAN_SCRATCH` (default: the system temp dir's
`bend-lean-review`), `<tmp>` for the temp dir, `<data-root>` for `$DATA_ROOT` and
`~` for the home directory. The scripts read these from the environment, and Base
from the `bend` on `PATH`.

## Semantics

Let `M = 2^32`, and let `x,y` denote unsigned natural values in `[0,M)`.
The Lean theorems below quantify over every structural Word/U32 value, not just
the test vectors.

| Operation | Installed Bend Base behavior | Lean model / theorem | UInt32 / BitVec 32 |
| --- | --- | --- | --- |
| `add` | `(x+y) % M` | Word carry chain; `Word.toNat_add`, `U32.add_toUInt32` | Equal |
| `sub` | `(M+x-y) % M`; underflow wraps | Complement and carry; `Word.toNat_sub`, `U32.sub_toUInt32` | Equal |
| `mul` | `(x*y) % M` | Shift/add loop; `Word.toNat_mul`, `U32.mul_toUInt32` | Equal |
| `div` | `x / y`; `x / 0 = 0` | Base's long division, including the high-bit branch; `U32.div_spec`, `U32.div_toUInt32` | Equal, including zero divisor |
| `mod` | `x % y`; `x % 0 = x` | Same long division; `U32.mod_spec`, `U32.mod_toUInt32` | Equal, including zero divisor |
| `eq`, `ne`, `lt`, `le`, `gt`, `ge` | Unsigned comparisons via high bits before low bits | `Word.cmp_spec`, six U32 comparison refinements | Equal |
| `to_nat` | Exact unsigned value `x` | `Word.toNat`, `Word.toNat_lt`, `U32.toUInt32_toNat` | Equal |
| `from_nat(n)` | Repeated increment; value `n % M` | Faithful recursive definition; `U32.fromNat_spec`, `U32.fromNat_toUInt32` | Equal to `UInt32.ofNat n` |
| U32 literal `n` | Accepted only for `0 <= n < M`; otherwise rejected | `parseLiteral_ok`, `parseLiteral_reject` | Accepted literals agree; Lean's `UInt32.ofNat` wraps larger naturals |
| Nat literal `n` | Same upper bound; above 256 expands to `U32.to_nat(n)` | `parseNatLiteral_ok`, `parseNatLiteral_reject` | Accepted literals have their mathematical value |
| `inc`, `not`, `and`, `or`, `xor` | Ordinary wrapping/bit operations | Structural definitions and universal BitVec/UInt32 refinements | Equal |
| `shl`, `shr` | Shift by one, zero fill | `Word.toBitVec_shl`, `Word.toBitVec_shr` | Equal |
| `shln(a,n)`, `shrn(a,n)` | Repeated one-bit shifts; zero when `n >= 32` | `shlN_toBitVec`, `shrN_toBitVec`, `shlN_large`, `shrN_large` | Equal to BitVec shifts with a **Nat** count. UInt32 masks its count to `n % 32`; see exact theorems below |
| `min`, `max` | Unsigned minimum/maximum | `U32.min_spec`, `U32.max_spec` | Same unsigned values |
| `log2` | Floor log2 for positive values; zero at zero | Independently transcribed five-stage algorithm; `U32.log2_spec`, `U32.log2_toUInt32` | Equal |

The subtraction refinements also explicitly distinguish the two branches:

```text
y <= x  -> toNat(sub x y) = x-y
x < y   -> toNat(sub x y) = M+x-y
```

The shift difference is exact, not an unexplained counterexample:

```text
Bend shln/shrn:  BitVec Nat-count shift by n
UInt32 shift:   BitVec Nat-count shift by n % 32
```

`U32.uint32_shiftLeft_exact` and `U32.uint32_shiftRight_exact` prove these UInt32
formulas for every natural count, including conversion of that count to UInt32.
For example, Bend `U32.shln(1,32n)` is 0, while Lean
`((1 : UInt32) <<< 32)` is 1. Counts below 32 agree by `shlN_small`/`shrN_small`.
The runtime oracle handles this difference explicitly.

The mathematical model uses unbounded Lean Nat to match Base's inductive Nat.
The compiled Bend runtime separately limits Nat arithmetic to `2^48-1`.
Every conversion input in the corpus is within that runtime bound, including
`to_nat(a) * 65536 <= (2^32-1)*65536 < 2^48`.

## Fidelity results

The seed is **20261024**. Random pairs use xorshift32 with shifts `(13,17,5)`;
each pair consumes two consecutive generator states. The native program
generates those inputs itself and echoes both values, which Lean checks against
the independently generated input file. The checker receives literal copies of
the same input pairs, rather than performing I/O or compiling an IO main.

The 14 distinct edge values are the specified anchors and each in-range neighbor:

```text
0, 1, 2, 3, 65535, 65536, 65537,
2147483646, 2147483647, 2147483648, 2147483649,
4294967293, 4294967294, 4294967295
```

| Lane / comparison | Edge pairs | Seeded random pairs | Total pairs | Numeric mismatches |
| --- | ---: | ---: | ---: | ---: |
| Structural Lean model against independent UInt32/Nat oracle | 196 | 200,000 | 200,196 | 0 |
| Bend compiled native against Lean | 196 | 200,000 | 200,196 | 0 |
| Bend pure-main checker normalization against Lean | 196 | 4,096 | 4,292 | 0 |

Each common row contains two input values and 25 results: add/sub/mul/div/mod;
six comparisons; inc/not/and/or/xor; four shifts (variable counts span 0..64);
min/max; is_zero/is_even; and log2. The native row additionally exposes six
conversion results: `to_nat(a)`, `to_nat(b)`, and conversion back from `a`, `a+b`,
`a+1`, and `a*65536`. The efficient Lean conversion decoder used by the runner
is proved equal to Base's recursive `from_nat` by `fromNat_eq_literal`.

The comparison checked **6,606,468 native numeric fields** and **115,884 checker
numeric fields**. These field counts include the two echoed inputs. The common
fields of the two Bend lanes agree through their comparisons with the same Lean
results. [Machine-readable results](results/fidelity.json) include the input and
output SHA-256 values and per-lane elapsed times.

## Checker conversion findings

These are resource/observability failures, not value counterexamples:

- A pure `main() -> Nat: U32.to_nat(x)` fails with
  `Error: the machine stack overflowed (a deep recursion, or a literal too large to expand)`
  for every tested edge value from 65535 upward: 65535, 65536, 65537,
  2147483646, 2147483647, 2147483648, 2147483649, 4294967293, 4294967294,
  and 4294967295. The four tested values 0, 1, 2, 3 normalize correctly.
- A pure `main() -> U32: U32.from_nat(xn)` normalizes correctly for
  0, 1, 2, 3, 65535, 65536, and 65537. Each of the seven tested edges from
  2147483646 upward exceeded its individual **five-second** timeout.
- Separate checker `to_nat(from_nat(n))` probes pass for **all 257 values 0..256**.
- A U32 literal `4294967296` is rejected with
  `a u32 literal up to 4294967295 (got 4294967296)`, matching the source inspection.

Every conversion probe's input, return status, raw diagnostics, and elapsed time
are retained in [results/conversions.json](results/conversions.json). Full-width
conversion results were compared on the native corpus; they were **not** obtained
for the full checker corpus. Full-width checker conversion fidelity must not be
claimed from this run.

## What Lean has established

- `BendModel/Word.lean`: a low-bit-first dependent Word datatype; carry arithmetic,
  complement/subtraction, shifts including outgoing high bit, comparison,
  shift/add multiplication, bounded value decoding, and BitVec refinements.
- `BendModel/U32.lean`: width-32 operations, repeated-increment Nat conversion,
  and Base's long-division recursion. `divStep_spec` accounts for the outgoing
  high bit; quotient/remainder correctness then follows by induction and Euclidean
  division uniqueness. No assumption discards the overflowing branch.
- `BendModel/Refinement.lean`: UInt32 agreement and the exact differences in
  literal acceptance and shift counts.
- `BendModel/Log2.lean`: a separate arithmetic characterization of Base's log2,
  reduced to 32 interval cases checked with `omega`.
- `Audit.lean`: transitively checks every BendModel declaration for unsafe code
  and unapproved axioms. The completed audit covers 445 declarations, including
  294 theorems (these counts include compiler-generated declarations). Only
  `propext`, `Quot.sound`, and `Classical.choice` are permitted. There is no
  `sorryAx`, `Lean.ofReduceBool`, or custom arithmetic axiom.

`lake build` and `lake env lean Audit.lean` both pass with Lean 4.34.0. Package
builds treat warnings as errors. The final executable was replayed on the whole
200,196-pair input file and its output checked byte-for-byte against the stored
Lean results after the final model changes.

This is a handwritten independent transcription, with formal proofs **inside
Lean** and differential evidence **between implementations**. It does not prove
Bend's parser, Bend-to-BendTT translation, C compiler, or Lean's compiler correct.
In particular, testing operations does not establish that any later Bend law
means its intended Lean proposition. That is the separate Phase 2 obligation.
No Phase 2 lemma or planner-law verification is claimed here.

## Reproduction and provenance

From this directory, use the parent repo's Nix shell and a Python 3 executable:

```bash
nix develop -c lake build
nix develop -c lake env lean Audit.lean

# The devshell supplies Bend/Lean, not Python. Set PYTHON to an installed Python 3.
PYTHON=/path/to/python3 nix develop -c ./verify.sh
```

For a durable run on this machine, invoke the last command through `job start`
with `--mem 4G --cpus 2 --timeout 25m`; the actual complete fidelity run peaked
well below its 4 GiB cap. `fidelity.py` accepts `--scratch`, `--random`,
`--checker-random`, and `--chunk`; defaults reproduce the counts above.
`conversion_probes.py` has its own `--scratch` option and five-second limits for
individual edge probes. On NixOS the harness invokes the official Lean
executable with the working dynamic loader reported by `ldd`.

The installed Base SHA-256 is
`3238272964daa05438a8a99311008802fd495963ac78dbbf51f6d5e6bdcee77c`.
[results/provenance.json](results/provenance.json) records the precise Nix paths
and hashes for Base, the parser/frontend, compiler, and review source files.
The upstream source checkout was not used as a substitute for the installed
Base; it contains unrelated changes from a later revision.

The observed worktree branch is `bend-planner`, with parent commit
`555d25163f1b0bb3ea601bc39921f6e5681374de`. Other-agent edits in `PROOF.bend`,
the planner README, and `u32.bend` were already present and remain outside this
review's write scope.
