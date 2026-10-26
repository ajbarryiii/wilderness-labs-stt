# Independent Lean verification: Phase 2

AJ, **all 41 public arithmetic laws are true in the validated structural model and
have independent Lean proofs**. All 41 are generic arithmetic, upstreamable in
scope; none is planner-specific. This is not yet a complete eleven-law planner
verification: `pools_cover` remains unproved, and the requested symbolic `Small`
experiment also stack-overflows.

The surviving planner mutation removes `Small` from `pools_sizes`. Its unchanged
proof body still passes Bend and BendTT, establishing that the hypothesis is
unnecessary for that law. Lean also proves the width bound in `U32.divmod_spec`
unnecessary. `U32.to_nat_div` now has no positive-divisor hypothesis in either
Bend or its Lean mapping: both quotients are zero when the divisor is zero.

The original audit wrote only under `lean/` or `/tmp` and left the real Bend
files unchanged. The subsequent `U32.to_nat_div` generalization updates
`u32.bend` and its documentation; the recorded `results/*.json` remain historical
snapshots. No commits, pushes, PRs, or publishing.

## Independent translation and scope

The inventory came from **u32.bend itself**, not the README. Source statements are
preserved in [phase2_laws.json](results/phase2_laws.json). Lean uses the Phase 1
structural Word/U32 operations and faithfully transcribes the new `Nat.bit`,
subtraction helpers, repeated-increment `Word.from_nat`, `Word.Shift`, and
`U32.DivSpec`. `Nat.double` is explicitly modeled by its recursive Base definition
and proved to equal multiplication by two. No Bend proof or bend-mathlib theorem
is assumed as a Lean axiom.

Quantities (`+`, `-`) concern Bend's use of evidence; the Lean statements express
the mathematical contracts over the same values and Boolean relations. Every
current mathematical hypothesis is retained in the corresponding theorem.
`U32.to_nat_div` and `u32_to_nat_div` both quantify over every divisor, with the
positivity hypothesis removed. Other stronger forms have separate theorems.

`lake build` and `lake env lean Audit.lean` pass on Lean 4.34.0, without Mathlib.
The final audit covers **569 declarations, including 398 generated/user theorems**,
and permits only `propext`, `Quot.sound`, and `Classical.choice`. No unsafe model
definitions, `sorryAx`, compiler-trust axiom, or custom arithmetic axiom occurs.

## Classification, Lean mapping, and proof methods

All rows are **generic**: their statements mention only Bool, Nat, Word, U32,
comparison, and ordinary arithmetic. Even `add_from_nat_succ` contains no planner
concepts. “Upstreamable in scope” does not mean a drop-in Base patch: the pinned
third-party Nat dependencies and the checker resource failures still need to be
handled in an upstream implementation.

| # | Bend law | Lean theorem (`BendModel.Phase2.` prefix) | Classification | Proof method | Generality |
| ---: | --- | --- | --- | --- | --- |
| 1 | `Nat.cmp_bit` | [nat_cmp_bit](BendModel/Phase2.lean#L67) | Generic | Boolean cases, Nat comparison reflection, omega | Exact intended form; no unnecessary hypothesis identified. |
| 2 | `Word.cmp_to_nat` | [word_cmp_to_nat](BendModel/Phase2.lean#L83) | Generic | Phase 1 structural Word comparison refinement | Exact intended form; no unnecessary hypothesis identified. |
| 3 | `U32.cmp_to_nat` | [u32_cmp_to_nat](BendModel/Phase2.lean#L85) | Generic | Width-32 instance of structural comparison | Width-32 specialization of #2. |
| 4 | `U32.le_trans` | [u32_le_trans](BendModel/Phase2.lean#L87) | Generic | Boolean reflection and Nat order transitivity | Exact intended form; no unnecessary hypothesis identified. |
| 5 | `Nat.sub_bit` | [nat_sub_bit](BendModel/Phase2.lean#L93) | Generic | Eight bit cases and omega, with valid high-tail borrow | Exact intended form; no unnecessary hypothesis identified. |
| 6 | `Word.to_nat_sub_carry` | [word_to_nat_sub_carry](BendModel/Phase2.lean#L101) | Generic | Full-adder conservation, word bounds, carry cases, omega | Conditional non-wrapping form; a total modular ADC value formula is more general. |
| 7 | `U32.to_nat_sub` | [u32_to_nat_sub](BendModel/Phase2.lean#L116) | Generic | Phase 1 wrapping subtraction under no underflow | Conditional non-wrapping form; Phase 1 gives both subtraction branches. |
| 8 | `Word.to_nat_bound` | [word_to_nat_bound](BendModel/Phase2.lean#L122) | Generic | Word value bound and symbolic maximum/power identity | Exact intended form; no unnecessary hypothesis identified. |
| 9 | `Nat.cmp_double` | [nat_cmp_double](BendModel/Phase2.lean#L129) | Generic | Positive scaling preserves both Nat comparison tests | Exact intended form; no unnecessary hypothesis identified. |
| 10 | `Word.to_nat_inc` | [word_to_nat_inc](BendModel/Phase2.lean#L133) | Generic | Wrapping increment formula plus strict maximum bound | Conditional successor form; Phase 1 gives exact value modulo 2^n. |
| 11 | `U32.to_nat_bound` | [u32_to_nat_bound](BendModel/Phase2.lean#L140) | Generic | Width-32 instance of #8 | Width-32 specialization of #8. |
| 12 | `U32.to_nat_inc` | [u32_to_nat_inc](BendModel/Phase2.lean#L142) | Generic | Width-32 instance of #10 | Width-32 specialization of #10; exact modular form is stronger. |
| 13 | `Word.to_nat_zero` | [word_to_nat_zero](BendModel/Phase2.lean#L145) | Generic | Phase 1 structural induction on zero words | Exact intended form; no unnecessary hypothesis identified. |
| 14 | `Word.to_nat_from_nat` | [word_to_nat_from_nat](BendModel/Phase2.lean#L147) | Generic | Repeated-increment modulo invariant and representability bound | Bounded identity; wordFromNat_mod proves the total modular formula. |
| 15 | `U32.from_nat_word` | [u32_from_nat_word](BendModel/Phase2.lean#L154) | Generic | Word value injectivity and the same modulo decoding | Width-32 adapter for the generic Word conversion. |
| 16 | `U32.to_nat_from_nat` | [u32_to_nat_from_nat](BendModel/Phase2.lean#L158) | Generic | #14 and wrapper conversion #15 | Bounded identity; Phase 1 fromNat_spec is total and exact. |
| 17 | `Word.adc_carry` | [word_adc_carry](BendModel/Phase2.lean#L163) | Generic | Adder conservation modulo 2^n and increment formula | Can quantify over subtraction flag f too; word_adc_carry_general proves this. |
| 18 | `Word.add_inc` | [word_add_inc](BendModel/Phase2.lean#L168) | Generic | Modular addition and increment algebra | Exact intended form; no unnecessary hypothesis identified. |
| 19 | `Word.add_zero` | [word_add_zero](BendModel/Phase2.lean#L172) | Generic | Modular zero identity and word bound | Exact intended form; no unnecessary hypothesis identified. |
| 20 | `U32.add_one` | [u32_add_one](BendModel/Phase2.lean#L176) | Generic | Addition/increment value refinements and injectivity | Width-32 adapter; the same modular fact holds at arbitrary Word width. |
| 21 | `U32.add_inc` | [u32_add_inc](BendModel/Phase2.lean#L180) | Generic | Width-32 instance of #18 | Width-32 specialization of #18. |
| 22 | `U32.add_zero` | [u32_add_zero](BendModel/Phase2.lean#L182) | Generic | Modular zero identity and injectivity | Width-32 specialization of #19. |
| 23 | `U32.add_from_nat_succ` | [u32_add_from_nat_succ](BendModel/Phase2.lean#L186) | Generic | Modulo congruence and Nat addition associativity | Useful step form; derivable from ordinary modular addition laws. |
| 24 | `Word.to_nat_inc_le` | [word_to_nat_inc_le](BendModel/Phase2.lean#L196) | Generic | Exact wrapping increment and remainder <= dividend | One-sided bound; exact modular increment formula is stronger. |
| 25 | `U32.to_nat_inc_le` | [u32_to_nat_inc_le](BendModel/Phase2.lean#L200) | Generic | Width-32 instance of #24 | Width-32 one-sided bound; exact modular increment formula is stronger. |
| 26 | `U32.to_nat_from_nat_le` | [u32_to_nat_from_nat_le](BendModel/Phase2.lean#L202) | Generic | Exact modulo conversion and remainder <= dividend | One-sided bound; exact n mod 2^32 formula is stronger. |
| 27 | `U32.to_nat_le` | [u32_to_nat_le](BendModel/Phase2.lean#L207) | Generic | Phase 1 Boolean order reflection | One direction; u32_le_iff proves equivalence. |
| 28 | `U32.to_nat_eq` | [u32_to_nat_eq](BendModel/Phase2.lean#L210) | Generic | Phase 1 Boolean equality reflection | One direction; u32_eq_iff proves equivalence. |
| 29 | `Nat.bit_add` | [nat_bit_add](BendModel/Phase2.lean#L213) | Generic | Faithful Nat doubling refinement and distributivity | Exact intended form; no unnecessary hypothesis identified. |
| 30 | `Word.shift_spec` | [word_shift_spec](BendModel/Phase2.lean#L216) | Generic | Structural outgoing-bit shift conservation, including width 0 | Exact intended form; no unnecessary hypothesis identified. |
| 31 | `Word.max_nat_mono` | [word_max_nat_mono](BendModel/Phase2.lean#L222) | Generic | Maximum/power identity and monotonicity of powers of two | One direction; maxNat_mono_iff also reflects the width order. |
| 32 | `U32.of_nat_le` | [u32_of_nat_le](BendModel/Phase2.lean#L229) | Generic | Converse Boolean order reflection | Converse of #27; u32_le_iff combines both directions. |
| 33 | `U32.ge_nat` | [u32_ge_nat](BendModel/Phase2.lean#L232) | Generic | Phase 1 greater-or-equal reflection | Exact intended form; no unnecessary hypothesis identified. |
| 34 | `U32.divmod_spec` | [u32_divmod_spec](BendModel/Phase2.lean#L234) | Generic | Phase 1 long-division invariant; width hypothesis unused | Width bound m <= 32 is unnecessary; proved without it. Positivity remains necessary for strict remainder bound. |
| 35 | `U32.zero_nat` | [u32_zero_nat](BendModel/Phase2.lean#L239) | Generic | Phase 1 exact zero-test reflection | Exact intended form; no unnecessary hypothesis identified. |
| 36 | `U32.to_nat_div`: `U32.to_nat(U32.div(a, b)) == Nat.div(U32.to_nat(a), U32.to_nat(b))` for all `a, b` | [u32_to_nat_div](BendModel/Phase2.lean#L241) | Generic | Phase 1 total division refinement | Positive-divisor hypothesis removed in Bend and Lean; includes b=0. |
| 37 | `Nat.div_antitone` | [nat_div_antitone](BendModel/Phase2.lean#L244) | Generic | Core Lean Nat.div_le_div_left with positive smaller divisor | Exact intended form; no unnecessary hypothesis identified. |
| 38 | `U32.to_nat_lt` | [u32_to_nat_lt](BendModel/Phase2.lean#L249) | Generic | Phase 1 strict-order reflection | One direction; u32_lt_iff proves equivalence. |
| 39 | `U32.div_antitone` | [u32_div_antitone](BendModel/Phase2.lean#L252) | Generic | Division refinement, order reflection, and #37 | Exact intended form; no unnecessary hypothesis identified. |
| 40 | `U32.positive_of_nonzero` | [u32_positive_of_nonzero](BendModel/Phase2.lean#L260) | Generic | Zero-test reflection and Nat nonzero iff positive | One direction; u32_positive_iff proves equivalence. |
| 41 | `U32.zero_add` | [u32_zero_add](BendModel/Phase2.lean#L265) | Generic | Modular left-zero identity and word bound | Width-32 adapter; the left-zero identity holds for every Word width. |


The most concrete stronger forms are:

- `u32_divmod_spec_without_width`: the same quotient/remainder certificate for
  **every** prefix width, provided the divisor is positive. Phase 1's proof handles
  the high-bit overflow arm directly. The Bend proof instead uses width ≤32 to
  show that this arm cannot arise at its dividend prefixes.
- `u32_to_nat_div_without_positive`: exact quotient conversion for **all** U32
  divisors, since both Nat and U32 define division by zero as zero. This is now
  also the mapped `u32_to_nat_div` statement and the public Bend law.
- `word_adc_carry_general`: the carry-in/increment identity for both values of the
  add/subtract flag, rather than only addition.
- `u32_le_iff`, `u32_lt_iff`, `u32_eq_iff`, `u32_positive_iff`, and
  `maxNat_mono_iff`: equivalences strengthening the one-direction transports.
- The Phase 1 modular formulas strengthen conditional conversion identities and
  one-sided increment/conversion bounds. Their range hypotheses are still needed
  when the conclusion is the **unwrapped** identity. Explicit Lean counterexamples
  show that borrow, no-wrap, and antitonicity positivity conditions cannot simply
  all be removed.

## Mutation results

Two mutants were run for **each of the 41 laws**: **82 arithmetic mutants** total.
The baseline `u32.bend` passes both checkers. Each mutant is a full scratch copy
with the original proof bodies and a `PROOF.bend` importing it. Mutations negate
equalities, make an order strict, shift a result, flip an incoming bit, or remove
a proof requirement. For dropped arithmetic hypotheses, `Unit` retains the binder
without imposing a mathematical condition; this avoids an undefined-variable
parse failure.

| Arithmetic outcome | Count |
| --- | ---: |
| Normal `SOME PROOFS FAIL` rejection | 76 |
| Frontend `RangeError: Maximum call stack size exceeded` | 6 |
| Accepted mutants | 0 |

The six frontend crashes are both mutants of `U32.to_nat_bound`,
`U32.to_nat_inc`, and `U32.to_nat_from_nat`. These are **not** counted as normal
proof rejections. They occur in the frontend normalization/error path, with stack
frames in `term_higher` / `term_snf`; a source mutant does not reach BendTT after
that failure.

To separate this from kernel acceptance, I exported the valid original module
to BendTT, confirmed that the kernel accepts it, and independently applied the
**same changed propositions** to those six emitted theorem headers. The kernel
normally rejects **all six**, naming the corresponding theorem. For inequality,
the kernel text uses a live function from equality to `Empty`, matching Bend's
`!=` type. These direct-kernel controls do not repair or conceal the six failed
frontend runs.

| Planner control | Result |
| --- | --- |
| `pool_budget`: claim budget zero; claim budget minus one | Both normally rejected |
| `pools_sizes`: negate equality; prepend an extra zero-size pool | Both normally rejected |
| `pools_sizes`: remove `Small` and unused `w` binder | **Passes `--verdict`**; stronger law |
| Generic `pools_cover`: negate equality; shift expected indices by one | Both fail with the pre-existing TODO; **not valid mutation kills** |
| Concrete `pools_cover` instance, samples `[7]`, offsets `[0,1]` | Baseline passes; both corresponding mutants normally rejected |

The other ten planner laws form the closed baseline for the budget/sizes mutants;
only the unfilled `pools_cover` declaration is omitted in that scratch baseline.
For the concrete cover control, pool validation is also proved. This finite
control does not establish the generic cover law.

A rejection of an unchanged proof body does **not** prove that the mutated
statement is false or that every original hypothesis is necessary. The width and
positive-divisor generalizations above illustrate this: their original Bend
proof bodies use assumptions that the independently proved mathematics does not
need. No proof bodies were rewritten to make arithmetic mutants pass.

Every mutant's exact diff, command, result, and diagnostics are in
[phase2_mutations.json](results/phase2_mutations.json). The separate kernel controls
are in [phase2_kernel_controls.json](results/phase2_kernel_controls.json).

The exact passing `pools_sizes` mutation is:

```diff
--- a/LAWS.bend
+++ b/LAWS.bend
@@ -226,7 +226,6 @@
 law pools_sizes:
   for +ss: List<&2, U32>
   for +ps: List<&2, U32>
-  for w  : Small(ss)
   for v  : Valid(ss, ps)
   {lens(Plan.pools(ss, ps)) == offsets.gaps(ps) : List<&2, Nat>}
 
--- a/PROOF.bend
+++ b/PROOF.bend
@@ -1199,7 +1199,7 @@
 
 # Fill the original law verbatim. Small is unnecessary for lengths, but is kept
 # in the original interface; it remains essential for exact cover.
-def Laws.pools_sizes(ss, ps, w, v):
+def Laws.pools_sizes(ss, ps, v):
   match ps:
     case Nil{}:
       Empty.absurd({Nil{} == Nil{} : List<&2, Nat>}, false_true(v))
```

## Exact Nat dependencies

The module directly names **37** members of the pinned Nat library:
**34 public laws**, two predicate definitions (`le`, `lt`), and one helper
(`internal_false_ne_true`). The complete direct law list is:

```text
add_assoc, add_comm, add_left_comm, add_lt_add_iff_left, add_mul_div_right,
add_sub_of_le, add_succ, add_zero, div_eq_zero_of_le, div_mul_le_self, double_add,
double_eq_add, double_le_double, double_mul, eq_of_is_eq, is_ge_eq_is_le,
le_add_left, le_div_iff_mul_le, le_of_lt, le_of_lt_succ, le_refl, le_succ, le_trans,
lt_of_le_of_lt, lt_of_lt_of_le, lt_of_not_le, lt_of_succ_le, lt_succ_self,
mul_double, mul_le_mul_left, not_le_of_lt, one_mul, sub_zero, zero_le
```

The three other directly used names are `internal_false_ne_true, le, lt`. I traced resolved
global names through the emitted BendTT types and bodies, starting from all 41
new laws. Their complete **72-name transitive Nat dependency closure**, including
internal helpers and predicate definitions, is:

```text
add_add_add_comm, add_assoc, add_comm, add_div_left, add_le_add, add_le_add_left,
add_le_add_right, add_left_comm, add_lt_add_iff_left, add_mul, add_mul_div_right,
add_sub_of_le, add_succ, add_zero, div_eq_zero_of_le, div_mod_eq, div_mul_le_self,
double_add, double_eq_add, double_le_double, double_mul, eq_of_is_eq,
internal_div_add_zero_r, internal_div_block_start, internal_div_block_step,
internal_div_go_eq, internal_div_go_run, internal_div_go_shift,
internal_div_go_small, internal_div_gt_add, internal_div_le_add_cancel,
internal_div_le_big, internal_div_le_small, internal_div_le_step,
internal_div_le_wit, internal_div_not_succ_le, internal_div_true_ne_false,
internal_div_wit_succ, internal_div_zero_le, internal_div_zero_of_wit,
internal_false_ne_true, is_ge_eq_is_le, is_lt_eq_succ_le, le, le_add_left,
le_add_right, le_div_iff_mul_le, le_of_add_eq, le_of_lt, le_of_lt_succ, le_refl,
le_succ, le_trans, lt, lt_of_le_of_lt, lt_of_lt_of_le, lt_of_not_le, lt_of_succ_le,
lt_succ_iff, lt_succ_self, mul_add, mul_comm, mul_double, mul_le_mul_left,
mul_le_mul_right, mul_succ, mul_zero, not_le_of_lt, one_mul, sub_add_cancel,
sub_zero, zero_le
```

The pinned `nat.bend` imports only Base. Its SHA-256 is
`6ce3175202287430ee1f41b53ae9f959a0983081d1ba7e53191489115c3de37a`. Full directly referenced law statements and the
resolved closure are recorded in [phase2_audit.json](results/phase2_audit.json).
This distinguishes actual proof dependencies from unrelated imported Nat laws.

## The pools_cover blocker

The faithful Lean maximum is defined by:

```lean
def maxNat : Nat → Nat
  | 0 => 0
  | n+1 => 1 + natDouble (maxNat n)
```

The following all check in Lean:

```lean
maxNat_succ_eq_pow (n) : maxNat n + 1 = 2^n
maxNat_eq_pow_sub_one (n) : maxNat n = 2^n - 1
maxNat_32_literal : maxNat 32 = 4294967295
maxNat_32_pow : maxNat 32 = 2^32 - 1
small_bounds_equivalent (n) :
  decide (n ≤ 4294967295) = decide (n ≤ maxNat 32)
```

Thus the proposed bound substitution preserves the mathematical condition. That
does not mean the current Bend checker can process it.

The single requested scratch experiment imports `u32.bend` into `LAWS.bend`,
changes only the literal inside `Small`, and appends the minimal adapter calling
the existing `pools_cover_bounded` proof. **Both normal checking and `--verdict`
fail** with:

```text
SOME PROOFS FAIL
Error: the machine stack overflowed (a deep recursion, or a literal too large to expand)
```

It does **not** pass all eleven laws. The experiment fails before a successful
kernel verdict; no conclusion about the mathematical falsity of cover follows.
Here is the exact diff, also saved as
[phase2_symbolic_small.diff](results/phase2_symbolic_small.diff):

```diff
--- a/LAWS.bend
+++ b/LAWS.bend
@@ -17,6 +17,7 @@
 # The lemmas sort_perm and sort_sorted, and cut_cover, are the halves pool_cover and pool_sorted
 # split into.
 import Base
+import ./u32.bend as W
 import ./plan.bend as Plan
 
 # Vocabulary
@@ -170,7 +171,7 @@
 # ss has at most 2^32 - 1 samples, as a u32 file does (its length is a u32 count of bytes); past
 # that, Valid compares the pools with the length mod 2^32, and the pool laws need not hold
 def Small(+ss: List<&2, U32>) -> Type:
-  Holds(Nat.is_le(List.length(&2, U32, ss), 4294967295n))
+  Holds(Nat.is_le(List.length(&2, U32, ss), W.Word.max_nat(32n)))
 
 # The pool offsets pass the planner's validation for the samples ss
 def Valid(+ss: List<&2, U32>, +ps: List<&2, U32>) -> Type:
--- a/PROOF.bend
+++ b/PROOF.bend
@@ -1576,3 +1576,7 @@
   bounded: {Nat.is_le(List.length(&2, U32, ss), W.Word.max_nat(32n)) == True{} : Bool}, v: Laws.Valid(ss, ps)) ->
   {Laws.flat(Plan.pools(ss, ps)) == Plan.clips(ss, 0) : List<&2, Plan.Clip>}:
   pools_cover_roundtrip(ss, ps, W.U32.to_nat_from_nat(List.length(&2, U32, ss), bounded), v)
+
+
+def Laws.pools_cover(ss, ps, w, v):
+  pools_cover_bounded(ss, ps, w, v)
```

## Real files and complete reachable-file scan

I independently reran the real files:

| Real command | Result |
| --- | --- |
| `bend u32.bend` | ALL PROOFS CHECK |
| `bend u32.bend --verdict` | ALL PROOFS CHECK |
| `bend PROOF.bend` | SOME PROOFS FAIL; 1 TODO |
| `bend PROOF.bend --verdict` | SOME PROOFS FAIL; 1 TODO |

The scratch ten-law subset passes both checkers. Its original statements are
retained, including `Small` on `pools_sizes`. The only unfilled project law is
`pools_cover`; there are no explicit `?TODO` tokens, unsafe markers, question-mark
definitions, or foreign imports in `LAWS.bend`, `PROOF.bend`, `plan.bend`,
`u32.bend`, or the pinned `nat.bend`. The scan resolves the ten `Laws.*` proof
definitions across files, rather than treating every separate law declaration
as an unresolved hole.

**Base is not free of those features.** The installed Base contains two unsafe
definitions (`Array.fork`, `Array.join`), 112 foreign import lines in 56 effect
definitions, and 52 opaque/native type or operation declarations. The resolved
dependency closures of the arithmetic proofs and all ten closed planner-law
proofs reference **none** of the two unsafe definitions or the foreign effect
definitions. These standard Base facilities are listed in the audit rather than
silently excluded from the scan. Base's native opaque declarations are not
counted as unfinished project proofs.

Real source hashes, still unchanged after the experiments:

```text
LAWS.bend  366ee9e9b198ad219013a9ad3b777df6e21e55524c2c8af29982b5d3b41bb361
PROOF.bend be4224828ccffa8ea225171fb13460b6dfbab6f94ac8ad381070a8889ff82ee2
u32.bend   e81b2d072ed68b08866c21fe95a973aba7a01ad96f3b2b1b3c813316e9c22067
plan.bend  3f8ff4fef8345e6718c120122d07264f1ca837ba0e157664cef530a91f93dd58
```

## What remains unverified

- The **complete planner algorithms and three planner laws have not been ported
  to Lean**. The independent Lean result covers all 41 arithmetic lemmas and the
  symbolic/concrete maximum bridge. Budget/sizes have BendTT verdicts and negative
  controls, which still rely on the unproved Bend translation.
- Generic `pools_cover` has no completed proof in the real files. Its two generic
  mutation failures are masked by that existing hole; only the finite control is
  non-vacuous. The symbolic-bound experiment does not close it.
- Six frontend negative checks crash. Corresponding direct kernel controls reject,
  but this is not 82 clean frontend-plus-kernel negative verdicts.
- The **72 imported Nat definitions/laws have not all been independently ported
  as separate Lean theorems**. They are audited and checked by BendTT; the 41 new
  Lean proofs are independent of their truth, using Lean/core facts instead.
- No proof equates the Bend parser/translator or compiler with the Lean model.
  Header inspection, independent mathematics, Phase 1 differential tests, and
  negative controls reduce that gap but do not constitute a verified translator.

## Reproduction

All commands below run from `lean/` in the parent repository's Nix devshell.
The devshell supplies Bend, Lean, Lake, and BENDTT; select an installed Python 3.

```bash
nix develop -c lake build
nix develop -c lake env lean Audit.lean

# First run snapshots the real Bend files under $BEND_LEAN_SCRATCH/phase2/original:
nix develop -c /path/to/python3 phase2_bend.py
nix develop -c /path/to/python3 phase2_mutations.py
nix develop -c bend $BEND_LEAN_SCRATCH/phase2/original/u32.bend -o $BEND_LEAN_SCRATCH/phase2/u32.bendtt
nix develop -c bend $BEND_LEAN_SCRATCH/phase2/ten-laws/PROOF.bend -o $BEND_LEAN_SCRATCH/phase2/planner-ten.bendtt
nix develop -c /path/to/python3 phase2_kernel_controls.py
/path/to/python3 phase2_audit.py
/path/to/python3 phase2_report.py
```

Run long checks through `job` with an explicit 2–4 GiB memory cap. Individual
Bend checks have 180-second timeouts; supplemental kernel controls have
15-second limits and a 128 MiB Lean stack limit. All completed within those limits.
Raw source copies, exports, and mutant directories remain under
`$BEND_LEAN_SCRATCH/phase2/`. Machine-readable evidence is under `results/`.
