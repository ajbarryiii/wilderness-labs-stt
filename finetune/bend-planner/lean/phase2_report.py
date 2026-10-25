#!/usr/bin/env python3
"""Assemble the review tables from source inventory and recorded tool results."""
import hashlib
import html
import json
from pathlib import Path
import re
import textwrap
from phase2_bend import HERE, ORIGINAL, REAL, SCRATCH

METHODS=[
 'Boolean cases, Nat comparison reflection, omega',
 'Phase 1 structural Word comparison refinement',
 'Width-32 instance of structural comparison',
 'Boolean reflection and Nat order transitivity',
 'Eight bit cases and omega, with valid high-tail borrow',
 'Full-adder conservation, word bounds, carry cases, omega',
 'Phase 1 wrapping subtraction under no underflow',
 'Word value bound and symbolic maximum/power identity',
 'Positive scaling preserves both Nat comparison tests',
 'Wrapping increment formula plus strict maximum bound',
 'Width-32 instance of #8',
 'Width-32 instance of #10',
 'Phase 1 structural induction on zero words',
 'Repeated-increment modulo invariant and representability bound',
 'Word value injectivity and the same modulo decoding',
 '#14 and wrapper conversion #15',
 'Adder conservation modulo 2^n and increment formula',
 'Modular addition and increment algebra',
 'Modular zero identity and word bound',
 'Addition/increment value refinements and injectivity',
 'Width-32 instance of #18',
 'Modular zero identity and injectivity',
 'Modulo congruence and Nat addition associativity',
 'Exact wrapping increment and remainder <= dividend',
 'Width-32 instance of #24',
 'Exact modulo conversion and remainder <= dividend',
 'Phase 1 Boolean order reflection',
 'Phase 1 Boolean equality reflection',
 'Faithful Nat doubling refinement and distributivity',
 'Structural outgoing-bit shift conservation, including width 0',
 'Maximum/power identity and monotonicity of powers of two',
 'Converse Boolean order reflection',
 'Phase 1 greater-or-equal reflection',
 'Phase 1 long-division invariant; width hypothesis unused',
 'Phase 1 exact zero-test reflection',
 'Phase 1 total division refinement; positivity unused',
 'Core Lean Nat.div_le_div_left with positive smaller divisor',
 'Phase 1 strict-order reflection',
 'Division refinement, order reflection, and #37',
 'Zero-test reflection and Nat nonzero iff positive',
 'Modular left-zero identity and word bound',
]
NOTES={
 3:'Width-32 specialization of #2.',
 6:'Conditional non-wrapping form; a total modular ADC value formula is more general.',
 7:'Conditional non-wrapping form; Phase 1 gives both subtraction branches.',
 10:'Conditional successor form; Phase 1 gives exact value modulo 2^n.',
 11:'Width-32 specialization of #8.',
 12:'Width-32 specialization of #10; exact modular form is stronger.',
 14:'Bounded identity; wordFromNat_mod proves the total modular formula.',
 15:'Width-32 adapter for the generic Word conversion.',
 16:'Bounded identity; Phase 1 fromNat_spec is total and exact.',
 17:'Can quantify over subtraction flag f too; word_adc_carry_general proves this.',
 20:'Width-32 adapter; the same modular fact holds at arbitrary Word width.',
 21:'Width-32 specialization of #18.',
 22:'Width-32 specialization of #19.',
 23:'Useful step form; derivable from ordinary modular addition laws.',
 24:'One-sided bound; exact modular increment formula is stronger.',
 25:'Width-32 one-sided bound; exact modular increment formula is stronger.',
 26:'One-sided bound; exact n mod 2^32 formula is stronger.',
 27:'One direction; u32_le_iff proves equivalence.',
 28:'One direction; u32_eq_iff proves equivalence.',
 31:'One direction; maxNat_mono_iff also reflects the width order.',
 32:'Converse of #27; u32_le_iff combines both directions.',
 34:'Width bound m <= 32 is unnecessary; proved without it. Positivity remains necessary for strict remainder bound.',
 36:'Positive-divisor hypothesis is unnecessary, including b=0.',
 38:'One direction; u32_lt_iff proves equivalence.',
 40:'One direction; u32_positive_iff proves equivalence.',
 41:'Width-32 adapter; the left-zero identity holds for every Word width.',
}


def main():
    laws=json.loads((HERE/'results/phase2_laws.json').read_text())
    audit=json.loads((HERE/'results/phase2_audit.json').read_text())
    muts=json.loads((HERE/'results/phase2_mutations.json').read_text())
    kernel=json.loads((HERE/'results/phase2_kernel_controls.json').read_text())
    baselines=json.loads((HERE/'results/phase2_bend_checks.json').read_text())
    lean=(HERE/'BendModel/Phase2.lean').read_text()
    assert len(laws)==len(METHODS)==41
    mapping=[]
    for i,(law,method) in enumerate(zip(laws,METHODS),1):
        name=law['name'].lower().replace('.','_')
        m=re.search(r'^theorem '+name+r'\b',lean,re.M);assert m,name
        mm=[r for r in muts['arithmetic'] if r['law']==law['name']]
        assert len(mm)==2
        statuses=['rejected' if r['rejected'] else 'frontend crash' if 'RangeError' in r['stderr'] else 'passed' if r['passed'] else 'other failure' for r in mm]
        mapping.append(dict(index=i,bend=law['name'],lean='BendModel.Phase2.'+name,
            lean_line=lean.count('\n',0,m.start())+1,classification='Generic / upstreamable in scope',
            method=method,generality=NOTES.get(i,'Exact intended form; no unnecessary hypothesis identified.'),
            mutation_results=statuses))
    (HERE/'results/phase2_mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
    direct=[r['name'] for r in audit['direct_nat_references'] if r['kind']=='law']
    helper=[r['name'] for r in audit['direct_nat_references'] if r['kind']!='law']
    table='| # | Bend law | Lean theorem (`BendModel.Phase2.` prefix) | Classification | Proof method | Generality |\n| ---: | --- | --- | --- | --- | --- |\n'
    for r in mapping:
        label=r['lean'].split('.')[-1]
        table+=f"| {r['index']} | `{r['bend']}` | [{label}](BendModel/Phase2.lean#L{r['lean_line']}) | Generic | {r['method']} | {r['generality']} |\n"
    diff=(HERE/'results/phase2_symbolic_small.diff').read_text()
    survivor=next(r for r in muts['planner'] if r['law']=='pools_sizes' and r['mutant']==3)
    assert survivor['passed']
    assert kernel[0]['passed'] and all(r['rejected'] for r in kernel[1:])
    assert len(muts['arithmetic'])==82
    assert sum(r['rejected'] for r in muts['arithmetic'])==76
    assert not any(r['passed'] for r in muts['arithmetic'])
    report=f'''# Independent Lean verification: Phase 2

AJ, **all 41 public arithmetic laws are true in the validated structural model and
have independent Lean proofs**. All 41 are generic arithmetic, upstreamable in
scope; none is planner-specific. This is not yet a complete eleven-law planner
verification: `pools_cover` remains unproved, and the requested symbolic `Small`
experiment also stack-overflows.

The surviving planner mutation removes `Small` from `pools_sizes`. Its unchanged
proof body still passes Bend and BendTT, establishing that the hypothesis is
unnecessary for that law. Two other unnecessary restrictions are proved in Lean:
the width bound in `U32.divmod_spec`, and positivity in `U32.to_nat_div`.

Everything written is under `lean/` or `/tmp`. The real `LAWS.bend`, `PROOF.bend`,
`u32.bend`, and `plan.bend` were not changed. No commits, pushes, PRs, or publishing.

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
original mathematical hypothesis is retained in the corresponding theorem,
including the two demonstrated redundant ones. The stronger forms have separate
theorems, so they do not replace or weaken the acceptance statements.

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

{table}

The most concrete stronger forms are:

- `u32_divmod_spec_without_width`: the same quotient/remainder certificate for
  **every** prefix width, provided the divisor is positive. Phase 1's proof handles
  the high-bit overflow arm directly. The Bend proof instead uses width ≤32 to
  show that this arm cannot arise at its dividend prefixes.
- `u32_to_nat_div_without_positive`: exact quotient conversion for **all** U32
  divisors, since both Nat and U32 define division by zero as zero.
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
{survivor['diff'].rstrip()}
```

## Exact Nat dependencies

The module directly names **37** members of the pinned Nat library:
**34 public laws**, two predicate definitions (`le`, `lt`), and one helper
(`internal_false_ne_true`). The complete direct law list is:

```text
{textwrap.fill(', '.join(direct),84)}
```

The three other directly used names are `{', '.join(helper)}`. I traced resolved
global names through the emitted BendTT types and bodies, starting from all 41
new laws. Their complete **72-name transitive Nat dependency closure**, including
internal helpers and predicate definitions, is:

```text
{textwrap.fill(', '.join(audit['transitive_nat_dependencies']),84)}
```

The pinned `nat.bend` imports only Base. Its SHA-256 is
`{baselines['nat_sha256']}`. Full directly referenced law statements and the
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
{diff.rstrip()}
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
LAWS.bend  {baselines['source_hashes']['LAWS.bend']}
PROOF.bend {baselines['source_hashes']['PROOF.bend']}
u32.bend   {baselines['source_hashes']['u32.bend']}
plan.bend  {baselines['source_hashes']['plan.bend']}
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
'''
    (HERE/'PHASE2.md').write_text(report)
    # A compact interactive presentation; detailed evidence remains in Markdown/JSON.
    rows=''.join('<tr>'+''.join(f'<td>{html.escape(v)}</td>' for v in
        [r['bend'],r['lean'].split('.')[-1],r['method'],r['generality']])+'</tr>' for r in mapping)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Phase 2 independent verification</title>
<style>body{margin:0;font:14px/1.48 var(--font-sans,system-ui);color:var(--foreground)}p{margin:0 0 14px}.lead{font-size:20px;font-weight:650;line-height:1.35}.muted{color:var(--muted-foreground)}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}.stat{border-top:2px solid var(--success,#438761);padding-top:8px}.stat b{display:block;font-size:26px}.small{font-size:12px}.note{padding:16px;border-radius:var(--radius,8px);background:var(--warning-surface,var(--secondary));margin:16px 0}.note p:last-child{margin:0}code{font:12px var(--font-mono,monospace)}table{border-collapse:collapse;width:100%;font-size:12px;margin:12px 0}th,td{padding:9px 7px;text-align:left;vertical-align:top;border-bottom:1px solid var(--border);overflow-wrap:anywhere}th:first-child,td:first-child{padding-left:0}th:last-child,td:last-child{padding-right:0}summary{cursor:pointer;font-weight:650;margin:18px 0 10px}input{box-sizing:border-box;width:100%;padding:9px 12px;background:var(--background);color:var(--foreground);border:1px solid var(--border);border-radius:var(--radius,7px)}@media(max-width:500px){.stats{gap:8px}.stat b{font-size:22px}#mapping th:nth-child(3),#mapping td:nth-child(3){display:none}table{font-size:11px}}</style>
<body><p class="lead">AJ, all 41 arithmetic laws have independent Lean proofs.</p><p class="muted">Each is generic arithmetic and upstreamable in scope. The full eleven-law planner verification remains incomplete.</p>
<div class="stats"><div class="stat"><b>41 / 41</b><span class="small">Lean statements proved<br>No Mathlib or Bend axioms</span></div><div class="stat"><b>82</b><span class="small">Arithmetic mutants<br>76 rejections · 6 frontend crashes</span></div><div class="stat"><b>1</b><span class="small">Surviving planner change<br>pools_sizes needs no Small</span></div></div>
<table><thead><tr><th>Independent finding</th><th>Result</th></tr></thead><tbody>
<tr><td>Remove Small from pools_sizes</td><td>Passes --verdict after deleting the unused proof parameter; proof body unchanged.</td></tr>
<tr><td>U32.divmod_spec width ≤32</td><td>Unnecessary. Lean proves the certificate for every prefix width and positive divisor.</td></tr>
<tr><td>U32.to_nat_div positivity</td><td>Unnecessary. Exact agreement also holds at divisor zero.</td></tr>
<tr><td>Word.adc_carry restricted to addition</td><td>Lean also proves it for the subtraction flag.</td></tr>
<tr><td>Symbolic maximum bridge</td><td>Lean proves maxNat 32 = 4,294,967,295 = 2³²−1.</td></tr>
<tr><td>Replace literal Small bound in /tmp</td><td>Still stack-overflows, with the adapter to pools_cover_bounded. All eleven laws do not pass.</td></tr>
<tr><td>Imported Nat dependencies</td><td>34 directly used laws, 2 predicates, 1 helper; 72 resolved transitive dependencies.</td></tr>
</tbody></table>
<div class="note"><p><strong>Negative-check limits are explicit.</strong> Both source mutants of to_nat_bound, to_nat_inc, and to_nat_from_nat crash the frontend. The same six changed propositions are normally rejected when applied directly to kernel theorem headers; the unmodified kernel export passes.</p><p>Generic pools_cover mutants fail at its existing TODO, so those failures are not mutation kills. A valid one-clip baseline passes and both concrete mutations are rejected. The complete planner laws have not been independently proved in Lean.</p></div>
<details><summary>All 41 mappings, proof methods, and generality notes</summary><p class="small muted">Classification: generic for every row. Lean theorem prefix: BendModel.Phase2.</p><input id="search" aria-label="Filter lemma table" placeholder="Filter by lemma, theorem, or restriction…"><table id="mapping"><thead><tr><th>Bend law</th><th>Lean theorem</th><th>Method</th><th>Generality</th></tr></thead><tbody>'''+rows+'''</tbody></table></details>
<p class="small muted">The real Bend files are unchanged. The final Lean audit finds no unsafe code or axioms beyond propext, Quot.sound, and Classical.choice. Base's unsafe array helpers and foreign effects were scanned and are absent from the proof dependency closures. Parser/translation correctness remains unproved.</p>
<script>document.getElementById('search').addEventListener('input',e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('#mapping tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));});console.log('Mappings:',document.querySelectorAll('#mapping tbody tr').length);</script></body></html>'''
    (HERE/'phase2.html').write_text(page)
    # Preserve current fingerprints separately from the historical Phase 1 record.
    paths=[p for p in HERE.rglob('*') if p.is_file() and p.suffix in {'.lean','.py','.sh'} and '.lake' not in p.parts and '__pycache__' not in p.parts]
    hashes={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    assert all(hashlib.sha256((REAL/n).read_bytes()).hexdigest()==h for n,h in baselines['source_hashes'].items())
    (HERE/'results/phase2_provenance.json').write_text(json.dumps(dict(review_sources=hashes,
        real_sources=baselines['source_hashes'],lean_version='4.34.0',bend_version='2.0.37',
        audit_declarations=569,audit_theorems_including_generated=398),indent=2)+'\n')
    print('Wrote PHASE2.md, phase2.html, 41-row mapping, and Phase 2 provenance.')


if __name__=='__main__': main()
