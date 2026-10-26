# Bend 2 WER edit counts

Every WER figure in this project comes from one dynamic program: `edit_counts(ref, hyp)` in
`finetune/whisper-ternary/wer.py`, a unit-cost Levenshtein table and a backtrace from its corner that
prefers the diagonal (match or substitution), then a deletion, then an insertion. This directory
reimplements it in Bend 2 (bend2 2.0.37 from the repo flake). The Bend version returns the same S, D
and I, tie-break included, proves its laws in Bend's checker, and runs natively as a parallel batch
scorer.

It is an opt-in engine. `wer.py`, `parakeet-ternary/evaluate.py` and `relabel/relabel_align.py` are
unchanged, and no production scorer calls it.

| File | What it is |
| --- | --- |
| `wer.bend` | The pure program: the table and walk, the utterance segmentation, offset validation and the parallel driver. It imports no foreign code. |
| `scorer.bend` | The CLI. It reads the inputs, validates everything, and only then opens and writes the output. |
| `LAWS.bend` | The laws, stated over `wer.bend` with their own vocabulary (a textbook Levenshtein recurrence, alignments as edit scripts). |
| `PROOF.bend` | The proofs of all nine laws. |
| `bend_wer.py` | The Python wrapper: `edit_counts`, `edit_counts_many` and `corpus_wer`, with `wer.py`'s contract. |
| `tests/` | Contract tests (fidelity, malformed input) and mutation tests. |

It reuses three things from `../bend-planner`: the u32 file effects (`effs/u32io.c`, `.js`),
`Plan.list` (array to list), and the `U32.cmp_to_nat` lemma of `u32.bend`. They are imported by
relative path, so nothing is copied.

## Build and run

```bash
nix develop <repo> -c bend scorer.bend -o build/bend-wer    # from finetune/bend-wer
build/bend-wer IDS OFFSETS COUNTS --threads 16               # bend 2.0.37: runtime options anywhere
```

All three files are little-endian u32 arrays:

- `IDS` holds the word ids of every utterance, ref then hyp.
- `OFFSETS` holds 2k + 1 offsets into `IDS` for k utterances. Utterance u has ref
  `IDS[o[2u] : o[2u+1]]` and hyp `IDS[o[2u+1] : o[2u+2]]`.
- `COUNTS` receives 3k words: S, D, I of each utterance, in order.

Only id equality matters. The wrapper gives equal words equal ids within one call.

On success the binary exits 0 and writes `COUNTS`. On malformed input it exits 1 with
`bend-wer: <reason>` on stderr and writes nothing. Malformed means any of these:

- a file length that is not a multiple of 4;
- an empty or even-length `OFFSETS`;
- offsets that do not start at 0, decrease, or do not end at the length of `IDS`;
- an input that cannot be opened.

A wrong argument count exits 2. As in bend-planner, an input over 4 GiB makes Base's `File.size`
fail with `EOVERFLOW` (exit 75).

From Python (the repo's `finetune/relabel/python`, or any interpreter with numpy):

```python
import bend_wer                                   # finetune/bend-wer on sys.path
bend_wer.edit_counts("a b c".split(), "a c d".split())    # (2, 0, 0), as wer.edit_counts
bend_wer.corpus_wer(records, threads=16)          # the dict wer.corpus_wer returns
```

The wrapper writes the arrays to a temporary directory and calls `build/bend-wer` (or `$BEND_WER`).
It sends at most 4M ids and 1M utterances per call. A call on 3.9M ids (79k utterances) peaked at
125 MB. The binary builds some of its lists by plain recursion, one native stack frame per
utterance or word, on the runtime's fixed 2 GiB stack (bend-planner's README), and the chunks stay
far below that.

## Design

`edit_counts` fills `cost[i][j] = d(ref[:i], hyp[:j])` and walks back from `cost[n][m]`. Reversing
both word lists turns those prefixes into suffixes: the same table, read from the opposite corner.
`Wer.counts(ref, hyp)` reverses the lists and fills the suffix table row by row, from the last ref
word up.

Each cell holds its cost and the counts (M, S, D, I) of the walk that starts there. A walk step
always lands on a cell that is already filled (below, diagonally below, or to the right), so the
walk is computed with the table. That takes one pass and keeps only two rows. The counts of the
corner cell are the answer.

`Wer.batch` cuts the ids into utterances in one pass. `Wer.tree` arranges them in a balanced tree
that halves them by count. `Wer.solve.tree` solves the two halves of every node with a parallel let.

## What is proven

All nine laws are proven. Both commands print `ALL PROOFS CHECK`:

```bash
nix develop <repo> -c bend finetune/bend-wer/PROOF.bend
nix develop <repo> -c bend finetune/bend-wer/PROOF.bend --verdict
```

`--verdict` rechecks with the flake's `$BENDTT`, the kernel built from bend2 2.0.37's `bendtt.lean`.
The proofs use no `@unsafe` def, no foreign code and no hole.

`distance(ref, hyp)` is `lev(rev(ref), rev(hyp))`, where `lev` is the textbook recurrence on the first
words with its own unit costs. That is the value of `edit_counts`' own table: its `cost[i][j]` is
`lev(rev(ref[:i]), rev(hyp[:j]))`.

| Law in `LAWS.bend` | Statement | Status |
| --- | --- | --- |
| `total_is_distance` | S + D + I = distance(ref, hyp) | proven |
| `ref_length` | M + S + D = len(ref) | proven |
| `hyp_length` | M + S + I = len(hyp) | proven |
| `identical` | counts(ws, ws) = (M = len(ws), S = D = I = 0) | proven |
| `lower_bound` | \|len(ref) - len(hyp)\| <= S + D + I | proven |
| `upper_bound` | S + D + I <= max(len(ref), len(hyp)) | proven |
| `optimal` | every edit script (keep, substitute, drop, add) of the words it reads and writes costs at least S + D + I | proven (stretch) |
| `symmetric` | S + D + I is unchanged when ref and hyp swap | proven (stretch) |
| `solve_spec` | the parallel driver returns each utterance's counts, in order | proven |

The proof goes through `Spec.cell(xs, ys)`, the suffix table written as a plain exponential
recursion over the two lists with `wer.bend`'s own step. It shows these facts in turn:

- the program's rows are `Spec.cell`'s, cell by cell;
- a cell's cost is `lev`;
- a cell's counts add up to its cost, because each step pays exactly what its cell costs (when
  neither the diagonal nor the deletion matches the minimum of three, the insertion does);
- a cell's M + S + D and M + S + I are its lengths.

`optimal` reduces to `lev` being no greater than any script's cost, and to reversing a script.
`symmetric` reduces to `lev(xs, ys) = lev(ys, xs)`.

What the laws do not pin down:

- **The tie-break.** Any minimum-cost walk satisfies every law here. The fidelity tests pin S, D and
  I individually.
- **The glue.** No law covers the segmentation of `IDS` by `OFFSETS`, the validation, the array
  and list glue, the effects, or the u32 output (`U32.from_nat` of counts below 2^32). The contract
  tests check those.

### Mutation tests

`tests/test_mutants.py` copies the sources, makes one edit to `wer.bend`, and requires both the
proof gate and the fidelity tests to reject the result. The inputs are 4,434 pairs: the edge cases,
every pair of words over {a, b} up to length 5, and random pairs. "Laws broken" lists the laws whose
statement fails numerically on some input. The length laws are checked in their M-free form, since
the binary does not output M.

| Mutant | `PROOF.bend` | Mismatches with `edit_counts` | Laws broken |
| --- | --- | --- | --- |
| tie-break: insertion before deletion | fails | 5 / 4,434 | none |
| tie-break: deletion before diagonal | fails | 479 / 4,434 | none |
| off by one: an insertion costs 2 | fails | 630 / 4,434 | `total_is_distance` |
| a match counted as a substitution | fails | 4,042 / 4,434 | `total_is_distance`, `identical`, bounds |

A tie-break change keeps every law true, so only the fidelity tests catch it. For the tie-breaks,
`PROOF.bend` fails only because its proof script follows the code's branch structure. Swapping the
insertion and deletion preference changes the counts rarely. The 5 cases come from the exhaustive
{a, b} pairs (for example ref `a b a b`, hyp `b a a b a`), and none of 450 random pairs hit one.

## Fidelity

The permanent tests (`finetune/relabel/python -m unittest discover -s finetune/bend-wer/tests -v`)
compare the Bend counts with `wer.edit_counts` and with `relabel_align.align`. The inputs are:

- edge cases: empty ref, empty hyp, both empty, repeats, swaps, shifts and unicode words;
- every pair over {a, b} up to length 5 (3,969 pairs);
- random batches over vocabularies of 1 to 50 words;
- a 400 x 350 utterance;
- runs at 1 to 8 threads and with forced chunking.

They also check that `corpus_wer` returns `wer.corpus_wer`'s dict, including its `ValueError`, and
that the binary rejects malformed input.

A one-off check scored every saved per-utterance score file of the project: the relabel dev sets
(B0, M1, MB1, eval-ext, eval-stem) and the parakeet-ternary and parakeet-binary eval and run
results. That was 1,083 files, 1,838,153 utterances, 28.4M reference words and 702M table cells.

| Comparison | Mismatches |
| --- | --- |
| Bend vs `wer.edit_counts` | 0 |
| `relabel_align.align` vs `wer.edit_counts` | 0 |
| S, D, I stored in the files vs `wer.edit_counts` | 0 |

## Measurements

The input is the largest saved score set: a 39,341-utterance test set with 969,328 reference words
and 27.0M table cells. Each time is the median of 5 runs (3 for the wrapper), and peak RSS comes from
GNU time. The machine has 32 threads, and other work was light (load average 2 to 4). Every Bend
run was byte-identical to `edit_counts`' counts.

| Program | Threads | Wall | Peak RSS | vs Python |
| --- | --- | --- | --- | --- |
| `wer.edit_counts` loop (Python) | 1 | 2.90 s | 150 MB | 1.0x |
| `bend-wer` native | 1 | 2.39 s | 62 MB | 1.2x |
| `bend-wer` native | 2 | 1.24 s | 62 MB | 2.3x |
| `bend-wer` native | 4 | 0.65 s | 62 MB | 4.5x |
| `bend-wer` native | 8 | 0.36 s | 63 MB | 8.1x |
| `bend-wer` native | 16 | 0.21 s | 63 MB | 14x |
| `bend_wer.edit_counts_many` | 1 | 2.63 s | 179 MB | 1.1x |
| `bend_wer.edit_counts_many` | 8 | 0.60 s | 179 MB | 4.8x |
| `bend_wer.edit_counts_many` | 16 | 0.46 s | 180 MB | 6.3x |

The rows measure different things:

- The Python row times only the loop over pre-split word lists. Its RSS is the interpreter plus
  those lists.
- The native rows are whole processes: read, cut, score and write.
- The wrapper rows add the Python side: ids for 1.94M words, the temporary files and reading the
  counts back. That is about 0.25 s here.

Single-threaded, a cell costs about 88 ns in Bend and 107 ns in Python. A cost-only version of the
same table, with no counts in the cells, ran at about 47 ns a cell. The counts make each cell a
heap node rather than an unboxed number, and that accounts for the difference.

## Bend pain points

These come on top of bend-planner's list (`../bend-planner/README.md`).

- **A parallel call whose arguments still need computing kept only two cores busy.** The first
  driver, `a b = solve(p, h, List.take(us, h)) solve(p, k - h, List.drop(us, h))`, ran in 1.3 s
  at 2, 4, 8 and 16 threads, with constant user time. Building the tree first and then forking over
  its nodes (as bend-planner does) scaled to 0.21 s at 16 threads.
- **A hypothesis is not refined by a match.** A lemma whose hypotheses mention a matched value must
  take them after the match, in its return type (`-> {..} -> {..}`, then `h => ..` in each case).
- **`%e : P` rewrites only the goal.** Turning one hypothesis into another needs a small transport
  def (`is_eq_false.rw` in `PROOF.bend`).
- **`Nat.cmp(0n, c)` does not reduce until c is a constructor.** A multi-scrutinee proof with a
  `case 0n _` branch therefore needs a separate lemma (`le_zero`) for `0n <= c`.
- **Lexicographic recursion works.** `lev(x <> xt, yt)` passes the first list rebuilt from its own
  pattern, and the checker (and `--verdict`) counts that as unchanged. A two-list recurrence needs
  no fuel.
- **Runtime `Nat` is native.** A cost-only table ran at the same speed with `Nat` and `U32` cells
  (1.89 s and 1.90 s on the same 40M cells), so the program and its proofs both use `Nat`.
