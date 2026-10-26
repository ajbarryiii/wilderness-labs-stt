# Phase 3: completed pools_cover review

AJ, **both real checkers now pass all eleven planner laws**, and the exported
module passes BendTT directly. The new `Small` condition is propositionally
identical to the original condition in the faithful Lean model. The new
`small_bound` and `pools_cover` proofs have no holes, unsafe dependencies, or
foreign dependencies.

**One requested negative check remains unresolved:** the width-33 header mutant
also stack-overflows the kernel. It is not a clean rejection. The other three
kernel mutants are normally rejected, with diagnostics naming `LAWS.pools_cover`.

## Reviewed revision and baseline

HEAD: `f4f5e113df4b9c2a02a9f4773b8455de8f17e22d`. The reviewed changes were uncommitted.
The [exact reviewed diff](results/phase3_reviewed.diff) confirms that the only
mathematical change to LAWS.bend is `Small`'s bound and its new import. PROOF.bend
adds the equality-elimination adapter `small_bound` and fills `Laws.pools_cover`.

| Command | Result |
| --- | --- |
| Real `bend PROOF.bend` | ALL PROOFS CHECK |
| Real `bend PROOF.bend --verdict` | ALL PROOFS CHECK |
| `bend` export of an identical snapshot to BendTT | Success |
| Unmodified export, checked by `$BENDTT` | ALL PROOFS CHECK |

The supplied `lean/minimal.bend` was copied and independently
rerun with `--verdict`. It still fails with `the machine stack overflowed`.
This supports the need for the new adapter; it does not invalidate its proof.

## Mutation controls

Every kernel mutant changes **only the emitted pools_cover theorem header**;
the proof body and every other emitted declaration remain unchanged.

| Mutant | Source `--verdict` | Direct BendTT kernel |
| --- | --- | --- |
| (a) `Plan.clips(ss, 0)` → `Plan.clips(ss, 1)` | Frontend RangeError | **Normally rejected**: expected start 1, observed start 0 |
| (b) Conclusion `==` → `!=` | Frontend RangeError | **Normally rejected**: expected equality → Empty, observed equality |
| (c) `Small` width 32 → 33 | Machine stack overflow | **Stack overflow; aborted with signal 6** |
| (d) My additional mutant: drop `Valid(ss, ps)` | Frontend RangeError | **Normally rejected**: expected Valid, observed Unit |

No mutant is accepted. Only three have clean kernel rejections. Resource failures
are recorded separately from proof rejections, including for (c).

For (b), Bend's inequality is encoded as a live function from the original
equality to `Empty`. For (d), `Unit` retains the original proof binder while
removing its mathematical restriction; this avoids an undefined-variable error.

For (c), the exported header originally uses `LAWS.Small`. I expanded that binder
to its actual `LAWS.Holds(Nat.is_le(length(ss), max_nat(width)))` body and changed
only width 32 to 33. The unary constructor terms and full old/new headers are
saved in the raw results. A **positive control with exactly the same expansion
at width 32 passes** the kernel. This checks that the width-33 failure is not
caused merely by expanding the alias incorrectly. The source mutation changes
the shared `Small` definition, as in AJ's experiment; the kernel control isolates
its resulting proposition at `pools_cover`.

The width-33 kernel run used the same installed BendTT binary as the successful
baseline, a 128 MiB Lean stack limit, and a 2 GiB job memory cap. It aborted after
about 2.3 seconds with exactly:

```text
Stack overflow detected. Aborting.
```

I cannot report a clean kernel rejection for that mutant. I did not modify the
kernel, reduce its fuel, or substitute a different proposition to manufacture one.

## Exact Lean equivalence for Small and Holds

[BendModel/Phase3.lean](BendModel/Phase3.lean) models `Holds(b)` as `b = true`,
exactly the equality condition in LAWS.bend, and uses actual `List.length` over
the validated U32 model:

```lean
def Holds (b : Bool) : Prop := b = true
def SmallLiteral (ss : List U32) : Prop :=
  Holds (decide (ss.length ≤ 4294967295))
def SmallSymbolic (ss : List U32) : Prop :=
  Holds (decide (ss.length ≤ maxNat 32))
```

The following statements all pass Lean:

| Theorem (`BendModel.Phase3.` prefix) | What it establishes | Method |
| --- | --- | --- |
| `small_propositions_equal` | `SmallLiteral ss = SmallSymbolic ss` | Congruence of Holds over Phase 2's proved Boolean equality |
| `small_iff` | Evidence transports in both directions | Rewrite by proposition equality |
| `holds_eliminate` | `Holds p → p = true` | Equality elimination |
| `small_bound` | The new Bend adapter's exact mathematical type | Apply `holds_eliminate` |
| `cover_contract_equivalent` | The complete quantified contract is equivalent with either Small condition | Transport Small evidence; arbitrary unchanged Valid and Result |
| `width33_changes_condition` | Length 2³² fails the width-32 bound and satisfies width 33 | Proved maximum/power identity and kernel computation |
| `width33_roundtrip_counterexample` | At length 2³², width 33 permits the length but U32 roundtrip changes it to 0 | Faithful conversion refinement and kernel computation |

`cover_contract_equivalent` quantifies over arbitrary `Valid` and `Result`
predicates. It therefore covers exactly how `Small` is used as a premise in the
planner laws, without assuming the planner's result. The rewording changes no
mathematical admissible input. Bend's quantity annotations affect use of evidence;
the Lean statement verifies the logical condition and its transport.

The adapter requires its `Small` evidence. Equality elimination exposes the
equation that evidence already contains; it does not prove arbitrary bounds.
The two width-33 counterexamples show why changing the width is a real weakening,
not a spelling change. They establish the arithmetic conversion failure, rather
than claiming a separately ported full-planner counterexample.

`lake build` and `lake env lean Audit.lean` pass on Lean 4.34.0 without Mathlib.
The updated audit covers **579 declarations and 405 theorems**, allowing only
`propext`, `Quot.sound`, and `Classical.choice`, with no unsafe declarations,
`sorryAx`, or custom axiom.

## Complete reachable-source and dependency rescan

The recursive import scan reaches six files: real LAWS.bend, PROOF.bend,
plan.bend, u32.bend, pinned bend-mathlib nat.bend, and installed Base. This includes
the newly introduced LAWS → u32 import. The project and pinned Nat source have no
holes, question-mark definitions, unsafe markers, or foreign imports. All eleven
law declarations have real proof definitions; no planner law remains open.

Base still contains two unsafe definitions (`Array.fork`, `Array.join`) and
112 foreign import lines in 56 effect definitions. The resolved exported
dependency closures of `small_bound` (**24 declarations**), `pools_cover`
(**213 declarations**), and every other planner law contain **none** of these
unsafe or foreign definitions. Both new proof bodies were also inspected in
the emitted kernel text: `small_bound` is equality elimination followed by
reflexivity; `pools_cover` applies the existing bounded proof with that evidence.

Real source hashes before and after the review:

```text
LAWS.bend  cf98af471d69253a7189e3de5f5ca9becd0c20d9af0e478a1bbc9350f84cd73e
PROOF.bend  0e1586d75dea176dce9d57da2787597c786dcfbf9c8898cf67208c101156ea27
u32.bend  e81b2d072ed68b08866c21fe95a973aba7a01ad96f3b2b1b3c813316e9c22067
plan.bend  3f8ff4fef8345e6718c120122d07264f1ca837ba0e157664cef530a91f93dd58
```

Only files under `lean/` and `/tmp` were written. No real Bend files were changed;
no commits, pushes, or PRs were made.

## Remaining limits and evidence

The full planner algorithms and generic planner cover law have **not** been
independently ported to Lean. Phase 3 verifies the exact Small/Holds equivalence
in Lean and the completed planner proofs in BendTT, with the negative controls
above. The parser/translator correctness gap remains; the width-33 kernel
resource failure also remains. This supersedes Phase 2's open-cover status,
without changing its historical results.

Machine-readable records:

- [Real checks, source scans and complete resolved closures](results/phase3_bend_checks.json)
- [All source/kernel mutations, exact headers, diffs and diagnostics](results/phase3_mutations.json)
- [Lean build and axiom audit](results/phase3_lean_check.json)
- [Source hashes and toolchain provenance](results/phase3_provenance.json)

Reproduce from `lean/` using the parent Nix devshell and an installed Python 3:

```bash
nix develop -c /path/to/python3 phase3_bend.py
nix develop -c /path/to/python3 phase3_mutations.py
nix develop -c lake build
nix develop -c lake env lean Audit.lean
```

The snapshot and complete kernel/source mutants remain under
`$BEND_LEAN_SCRATCH/phase3/`. Checks ran through agent-jobs with explicit 2 GiB
memory caps and per-command timeouts. The raw mutation records distinguish normal
rejection, abort, timeout, and acceptance; nonzero exit alone is not called a kill.
