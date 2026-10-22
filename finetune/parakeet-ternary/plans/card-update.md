# Plan: restructure the Kea-T model card (docs-only Hub update)

Written 2026-10-21. Contract for `hf/stage_card.py` and `tests/test_stage_card.py`.
AJ asked: "Break it apart and lead with headline, size, accuracy, and speed relative to our
best for the original. Add other files which contain the necessary info."

## 1. What changes

The live card of [rajb3/kea-t](https://huggingface.co/rajb3/kea-t) at Hub revision
`db7d4198687023a8db3b0c9eb4969dc639d684d9` is the rendering of `hf/MODEL_CARD.md` at
`origin/main` 4e06caa (SHA-256 of the rendered README
`9d52c83748ee101c95d9be9fdb7498f39c7b43f6a4d6ee4e3c2da8685337c893`). It is about 360 lines.
It is replaced by a short card in the style of the Kea-B card, plus three detail documents
that carry everything moved out of it:

| Hub file | Repository template | Content |
| --- | --- | --- |
| `README.md` | `hf/MODEL_CARD.md` (rewritten) | Title and 2-3 sentences, ending with one sentence on the 1-bit sibling Kea-B (size and test mean); an at-a-glance table of size, test mean WER and 10 s speed, energy and process VRAM against the original (speed against our optimized BF16 build); Results (one WER table); How it was trained; Use (download folder `kea-t/`, as in Kea-B's card); Files; Limitations; Used in (LocalFlow); License and attribution, which states that the card and its detail documents (`RUNTIME.md`, `EVALUATION.md`, `TRAINING.md`) are CC-BY-4.0; links to the three documents. |
| `RUNTIME.md` | `hf/RUNTIME.md` (new) | Every runtime configuration (ternary expanded, ternary compact, original BF16 optimized by us, the off-the-shelf ONNX row) at 3, 10 and 30.04 s; the method (latency scope, medians, process VRAM, NVML energy, the 400 W limit); the compact/expanded agreement check; requirements; links to the GitHub runtime guide and evidence. |
| `EVALUATION.md` | `hf/EVALUATION.md` (new) | The full WER table (NVIDIA published, our FP32 scoring, the no-training row, this model, difference and ratio rows); test sets and revision; mean of 7; Common Voice; normalizer; empty references; strict FP32; scored once on the rebuilt export; rounding; the evaluation limitations (AMI empty outputs, greedy only, timestamps, TED-LIUM, YODAS/GigaSpeech overlap, no FP32 control, input length). |
| `TRAINING.md` | `hf/TRAINING.md` (new) | The full training recipe, data mix and pilot; training data with licenses and references; the full file-format description (`export.safetensors`, `manifest.json`, `reconstruction.json`, tokenizer); what `load()` checks and does. |

The front matter (YAML, including model-index) is byte-identical to the live card's. Wording
changes only where moving text requires it; no new claims beyond the sourced 3 s and 30.04 s
runtime rows and the facts they need (power draw per row, the graph-cache effect on VRAM),
the Kea-B sibling sentence and the CC-BY-4.0 statement for the detail documents (both decided
by the coordinator on 2026-10-21).

## 2. What is NOT changed or re-uploaded

`export.safetensors`, `manifest.json`, `reconstruction.json`, `tokenizer/` (three files),
`load_ternary.py`, `LICENSE`, `NOTICE` and `.gitattributes`. They are not staged, so the
upload cannot touch them. `hf/load_ternary.py` (SHA-pinned by other evidence), `hf/NOTICE`,
`hf/LICENSE` and `hf/stage.py` are not edited. `hf/stage.py` stays the record of the full
2026-10-04 upload; its card checks describe the 2026-10-04 card, so it is not rerun for this
update and would need updating for a full re-stage. `NOTICE` keeps its file list (it names
`README.md` as CC-BY-4.0); the card itself extends CC-BY-4.0 to the three detail documents.

## 3. The staging script and its named checks

`hf/stage_card.py` (CPU only, no model load, no GPU, so it does not need `./heavy`; run with
`./python hf/stage_card.py stage` from `finetune/parakeet-ternary`). Every check is recorded by
name; the set of names is fixed in advance (`required_checks()`), success is the conjunction of
all of them, and a missing or unexpected name fails the run.

**A. Live Hub repository (read-only, network).**

- `hub main revision == pinned db7d419` (if the Hub moved, re-review before uploading).
- `hub file set == stage report files + .gitattributes`.
- `hub <file> SHA-256 and size == stage report` for each of `LICENSE`, `export.safetensors`,
  `load_ternary.py`, `manifest.json`, `reconstruction.json`, `tokenizer/tokenizer.model`,
  `tokenizer/tokenizer.vocab`, `tokenizer/vocab.txt`. LFS files are compared through the Hub's
  LFS SHA-256 and size; the others are downloaded at the pinned revision into a temporary
  directory under `/mnt/hd/wilderness-labs-stt/parakeet-ternary/tmp/` and hashed.
- `hub NOTICE SHA-256 and size == hf/NOTICE (pinned, post-rename)`. The one known difference
  from the stage report: the 2026-10-11 rename to Kea-T changed NOTICE's first line.
- `hub NOTICE differs from the stage report's NOTICE only in its first line`, using the
  2026-10-04 staged copy, whose SHA-256 must equal the report's.
- `hub .gitattributes SHA-256 == pinned` and `hub README.md SHA-256 == pinned live card`.
- `new document names are not on the hub`.
- `hub rajb3/kea-b is public; export.safetensors LFS SHA-256 and size == Kea-B stage report`
  (the card links Kea-B and states its size).

**B. Sources bound to the published export.** `results/test.json` and
`inference/results/rtx5090.json` name the export SHA-256 of the stage report; the run's
`manifest.json` and `reconstruction.json` equal the stage report's hashes; `summary.json`'s
reconstruction and byte accounting equal them; `runs/pilot-selection.json` selected the run's
recipe and learning rate; Kea-B's stage report
(`/mnt/hd/wilderness-labs-stt/parakeet-binary/hf-staging/rajb3__kea-b.stage.json`) passed and
its export SHA-256 equals `finetune/parakeet-binary/results/test.json` `mb1_weights_sha256`;
that `test.json` names this export as Kea-T (`m1_export_sha256`); `results/TEST.md` rows equal `test.json`
rounded; the `inference/README.md` results table equals `rtx5090.json` rounded.

**C. Rendering.** No unrendered placeholder in any document; the card's YAML parses
(`huggingface_hub.ModelCard`) with model_name `kea-t`, license, base_model and library_name;
the card's front matter is byte-identical to the live card's; the detail documents have no
front matter.

**D. Numbers.** Each document's required numbers, computed from their sources (section 6), are
present (`<doc> computed numbers == sources`). A phrase only matches at token boundaries:
the character before it may not be a word character, `.`, `,`, `+`, `-` or `−`, and the text
after it may not continue the number (no word character, no `.d` or `,d`). So a sign or digit
added next to a checked number (`-63.1% lower`) breaks the match, and the number scanner, which
reads signed tokens, reports it as unchecked. Every comparison between durations is bound to the
source values at that duration, and its sentence carries the duration (`At 10 s, ...`). The card's WER table is exactly the TEST.md B0
and M1 rows; EVALUATION.md's is all six TEST.md rows; RUNTIME.md's results table is exactly
the `inference/README.md` table. Phrases carried verbatim from the live card are present in it
(`carried phrases appear in the live card`). Coverage (`<doc> every number is checked`): after
removing fenced code, link targets, checked tables, computed phrases and carried phrases, no
number remains in any document. Loss (`every number in the live card appears in the new
documents`): every number in the live card's body appears somewhere in the new set, except
the explicit `DROPPED` list (empty unless justified in the report).

**E. Links.** Relative links resolve to a staged document and, if they have one, an existing
heading; the card links all three documents and each document links back to the card; every
`github.com/ajbarryiii/wilderness-labs-stt/{tree,blob}/main/<path>` link names a path on
`origin/main`; every other absolute URL appears in the live card, except Kea-B's page
`https://huggingface.co/rajb3/kea-b`, which the Kea-B Hub check covers.

**F. Staging.** The staged file set is exactly `README.md`, `RUNTIME.md`, `EVALUATION.md`,
`TRAINING.md`, and each staged file's SHA-256 equals its rendering.

Files are written to a temporary sibling folder and renamed to
`/mnt/hd/wilderness-labs-stt/parakeet-ternary/hf-staging/rajb3__kea-t.card-update` only if
every check passes; the report goes to `rajb3__kea-t.card-update.stage.json` (each staged file
with SHA-256 and size, the Hub snapshot, every check, the upload command). Stale outputs are
removed first; any failure, exception or exit leaves no promoted and no temporary folder, and
only `rajb3__kea-t.card-update.stage.FAILED.json`. It writes only under the staging directory
and the temporary download directory, and never uploads.

**Publishing subcommands** (`hf/stage_card.py`):

- `upload --approved-readme-sha256 HEX` is the only Hub write in the script. It refuses, with no
  write, unless the stage report is PASSED, the staged folder holds exactly the four documents
  with the report's SHA-256 and sizes, `HEX` (recorded by AJ after reading the staged README)
  equals the staged README's SHA-256, and main is still the pinned revision. It then makes one
  `HfApi().create_commit(..., create_pr=True, parent_commit=PINNED_REVISION)` with exactly the
  four documents, so the PR is based on the reviewed revision, never on a newer main. The PR
  number, URL and commit go to `rajb3__kea-t.card-update.upload.json`.
- `verify refs/pr/N` (read-only): the stage report PASSED; main is still the pinned revision;
  the PR is exactly one commit whose parent is the pinned revision; its file set is the pinned
  set plus the three documents; every file other than the four documents is unchanged from the
  pinned revision (size, LFS SHA-256, git blob id); the four documents equal the stage report.
  The PR head commit is recorded in `...verify-refs_pr_N.json`.
- `pre-merge refs/pr/N` (read-only), run immediately before AJ merges: that verify report
  PASSED, main is still the pinned revision, and the PR head is still the verified commit.
- `verify main` (read-only, after the merge): the same content checks on main, and the commits
  on main since the pinned revision are only the card update (one commit, or a merge commit
  whose other commit is the verified PR head).

**No other Hub write to `rajb3/kea-t`, by anyone, is allowed between staging and the merge.**
If one happens, the checks above fail; restage against the new main after reviewing it.

## 4. Tests (`tests/test_stage_card.py`, CPU only, fast)

Run against the real templates and sources with a fake Hub and a fake `origin/main` listing:
the real documents pass every check; each check family has a failure path (a tampered number,
an uncovered number, a dropped live-card number, a placeholder, a broken relative link or
anchor, a GitHub link to a missing path, an unknown external URL, a front-matter change, a
changed Hub file, a moved Hub revision, a doc name already on the Hub, an extra table row, a
reversed sign on a checked number, a `+`/`-` swap on a Limitations delta, a removed duration
qualifier);
staging promotes atomically and every failure, exception or `SystemExit` leaves no folder; the
check-name set is enforced; the Kea-B sentence fails on a changed Kea-B source or Hub file, or a private Kea-B repository; `upload` refuses without a write on a failed report, a changed staged file, a
wrong approval hash or a moved main, and otherwise creates one PR on the pinned parent with
exactly the four documents; `verify` passes on an unchanged PR and fails on a changed non-doc
file, a moved main (Codex r1's scenario: a README correction on main after the PR) and a PR
based on a newer main; `pre-merge` fails on a moved main or PR head; `verify main` fails on an
intervening commit; the script's only Hub write is that one `create_commit`; the card body
(below the front matter) is at most 135 lines and leads with the at-a-glance table.

## 5. Upload procedure (AJ only, after review)

1. Codex reviews this plan, the templates, `hf/stage_card.py`, the tests and the staged folder,
   iterated until clean.
2. Rerun `./python hf/stage_card.py stage` immediately before uploading (it fails if the Hub
   moved). AJ reads the staged `README.md` and the three documents and records the README's
   SHA-256 (`sha256sum .../rajb3__kea-t.card-update/README.md`).
3. From `finetune/parakeet-ternary`, open the Hub pull request (never a direct write to main):

       ./python hf/stage_card.py upload --approved-readme-sha256 <the README SHA-256 AJ approved>

   It uploads only the four documents, as one PR commit on the pinned parent.
4. `./python hf/stage_card.py verify refs/pr/N` must pass; AJ reviews the rendering at
   `https://huggingface.co/rajb3/kea-t/blob/refs%2Fpr%2FN/README.md` and the three documents.
   Relative links open main's copies, so they are checked after the merge.
5. Immediately before merging: `./python hf/stage_card.py pre-merge refs/pr/N` must pass; then
   AJ merges the PR on the Hub at once.
6. `./python hf/stage_card.py verify main` must pass; AJ checks that the card's relative links
   and anchors open the documents on the Hub.
7. Commit the templates and script to GitHub only with AJ's go-ahead.

## 6. Source of each claim

| Claim | Source |
| --- | --- |
| 180.8 MB, 2,472 MB | `results/test.json` `m1_export_mb`, `original_nemo_mb` |
| Kea-B: 105.3 MB, 7.05% test mean | Kea-B stage report `export.safetensors` bytes (equal to Kea-B `test.json` `export_mb.MB1` and the Hub file), Kea-B `test.json` `rows.MB1.mean` |
| Test WERs, means, differences, ratios, NVIDIA's published row, TED-LIUM 3.38 | `results/TEST.md` (rows checked against `results/test.json`) |
| Reproduction within 0.09 points per set | `test.json`: max over the seven sets of abs(B0 - published) |
| AMI empty outputs 1.43% and 3.12% | `test.json` `empty_hypothesis_fraction` |
| Every latency, process VRAM, energy and power value; the 3, 10 and 30.04 s rows | `inference/results/rtx5090.json` (medians), equal to the `inference/README.md` table |
| 23.2%, 13.5%, 17.1%, 41.8%, 9.3%, within 2%, 63.1%, compact at 30.04 s higher | computed from `rtx5090.json` |
| 400 W limit, 600 W default, 2,048 utterances, eight sets, onnx-asr 0.12.0, onnxruntime-gpu 1.24.4, runtime versions | `rtx5090.json` |
| 603,979,776 parameters and bytes, 264 matrices, 24 layers, byte accounting, 98%, code histogram, 1,024 tokens, NeMo/torch versions, base revision | export `manifest.json` (SHA-256 = stage report) |
| about 5.5e-4, four dev-clean clips, 264 layers exact | export `reconstruction.json` (SHA-256 = stage report) |
| 250,000 steps, 62,500, 25%, 600 s, 5e-4, 2% warmup, seed, 69.3 h, 2.9 days, 33,248 h, per-source hours and shares, 1 to 30 s, weight 1.0, dev 6.09% | run `summary.json` and `config.json` |
| Pilot: 12,000 steps, 7.05% to 6.85%, 0.004 points, 2e-4/5e-4/1e-3 | `runs/pilot-selection.json` |
| Development FP32 original 5.58% | `eval/B0-dev/summary.json` |
| Everything else with a number (AdamW betas, weight decay, clipping, quantizer epsilon, about 106 s, 50% filter, LocalFlow's macOS 26, licenses and versions of datasets) | carried verbatim from the live card (each phrase checked to be in it) |

## 7. Review log

**Codex r1 (2026-10-21): NOT CLEAN, 0 blockers, 3 majors, 1 minor**
(`/mnt/hd/wilderness-labs-stt/parakeet-ternary/review/card-update/r1.md`). Fixes:

1. [major] RUNTIME.md generalized "the ternary rows draw as much power or more and use less
   energy" and the 63.1% BF16-versus-ONNX latency to every duration; at 30.04 s compact is
   slower and uses more energy, and expanded draws less power. Both sentences now begin
   "At 10 s, ..." and are computed phrases bound to the 10 s values (the power sentence only
   renders if, at 10 s, both ternary modes draw at least BF16's power and use less energy and
   time). Tests remove each qualifier.
2. [major] A concurrent edit on main could be overwritten: the CLI upload bases the PR on the
   current main, and verify did not check main or the PR's parent. The upload is now the
   `upload` subcommand (`create_commit`, `create_pr=True`, `parent_commit` = the pin, approval
   hash required); `verify refs/pr/N` checks main and the PR parent; `pre-merge` re-checks main
   and the PR head right before the merge; `verify main` checks the commits since the pin; no
   other Hub writes are allowed between staging and the merge. Tests include Codex's scenario.
3. [major] `-63.1% lower latency` passed: substring matching ignored a leading sign. Phrases now
   match only at token boundaries and the scanner reads signed tokens. Regression tests for that
   mutation and for `+4.06` to `-4.06` in Limitations. Contract change: `numbers()` returns
   signed tokens (`+0.36`, not `0.36`).
4. [minor] The license paragraph's "all other tensors are stored in FP16 (feature-extraction
   constants FP32)" omitted the integer BatchNorm counters; it now says "the remaining learned
   weights are stored in FP16". The Files bullet already says "every other learned tensor".

**Publication (2026-10-22).** AJ approved the staged documents (README SHA-256
`87a6e5fdf316899cb9b9b1e41ad3ca7a738c6e951320e3f44f02823e376bdd95`). `stage` was rerun and
passed 53/53 at the pin; `upload` opened Hub PR #1
(https://huggingface.co/rajb3/kea-t/discussions/1, commit
`97e07f7c016b660980f25652910d05274ab7af7a`, parent `db7d419`); `verify refs/pr/1` passed 18/18;
`pre-merge refs/pr/1` passed 3/3 and the PR was merged at once; main is
`c0f28965998b0bbeef45cd6e6ab8aaeaf6bd0a79` (a squash, one commit on the pin), and `verify main`
passed 17/17. The model page and the three documents return HTTP 200 on main, and the card's
links and anchors (`EVALUATION.md#limitations`, `TRAINING.md#training-data-and-licenses`)
resolve. Reports: `hf-staging/rajb3__kea-t.card-update.{stage,upload,verify-refs_pr_1,pre-merge-refs_pr_1,verify-main}.json`.
`PINNED_REVISION` now names the pre-update revision, so `stage` fails by design; a later card
update starts a new plan against the then-current main.
