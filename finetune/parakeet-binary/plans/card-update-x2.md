# Plan: add the INT8x2 GPU runtime results to the Kea-B and Kea-T cards (docs-only Hub updates)

Written 2026-10-22. This plan is the contract for `finetune/parakeet-ternary/hf/cardkit.py` (the shared
engine), `finetune/parakeet-ternary/hf/x2facts.py` (the shared x2 sources),
`finetune/parakeet-ternary/hf/stage_card.py` (Kea-T), `finetune/parakeet-binary/hf/stage_card.py`
(Kea-B) and their tests. It reuses the design of the Kea-T card update
(`finetune/parakeet-ternary/plans/card-update.md`, Codex reviews r1 and r2), including the three r1
fixes: duration scope, the pinned-parent upload and signed, token-bounded number matching.

**AJ approves every card text before any upload.** Nothing here uploads, writes to the Hub, pushes to
GitHub or runs on the GPU. The upload procedure (section 7) is run by AJ, or on AJ's instruction.

## 1. What changes

| Repository | Live main (pinned) | Staged (uploaded) | Unchanged on the Hub |
| --- | --- | --- | --- |
| `rajb3/kea-b` | `5e7c298bc0eee174394a3fba88d789dc7c53ee4d` | `README.md` (rewritten from `hf/MODEL_CARD.md`), `RUNTIME.md` (new, `hf/RUNTIME.md`) | `export.safetensors`, `manifest.json`, `reconstruction.json`, `tokenizer/` (3 files), `load_binary.py`, `LICENSE`, `NOTICE`, `.gitattributes` |
| `rajb3/kea-t` | `c0f28965998b0bbeef45cd6e6ab8aaeaf6bd0a79` | `README.md`, `RUNTIME.md` (both rewritten) | everything else, including `EVALUATION.md` and `TRAINING.md` |

**Decisions (2026-10-22, coordinator and AJ).** gpu1bit is published as a **research runtime**:
- Its code is published for reference and is not packaged for general use. MB1-x2's kernel files
  load from /mnt/hd, and the wrappers, the NVRTC/cubin paths and the GPU lock are fixed to one
  machine.
- The documents give no "how to run" line for it. Instead they say: "Speed measured with our
  research runtime on an RTX 5090; the runtime code is published for reference and is not
  packaged for general use."
- They link `finetune/parakeet-binary/gpu1bit/README.md` and `gpu1bit/RESULTS.md`. Both go live
  only when the x2 thread's PR merges.

**Kea-B.**
- **At a glance**, after the intro: weight file and test WER mean of 7, then latency and
  approximate GPU energy at 10 s against our optimized BF16 build. These are research-runtime
  numbers, labelled so in the note below the table. There is no VRAM row (AJ).
- **Weight-footprint row.** Deferred (section 10).
- **The note.** It gives the setting, links the research runtime, and says the code is published
  for reference and not packaged for general use. It adds one cross-session sentence and one
  approximate-energy sentence, and points to `RUNTIME.md`.
- **Results.** One sentence: the test WERs come from FP32 decoding, and the runtime was checked
  on the development sets only, where its mean WER equals FP32 decoding's.
- **Use.** The sentence "For now, the small file only saves download time and disk space. We are
  working on native 1-bit kernels, but they are not part of this release" is replaced. The new
  text says `load_binary.py` still rebuilds dense FP32, and the speed comes from the research
  runtime, which is published for reference and not packaged for general use.
- Files, Limitations and License each gain one line.
- **`RUNTIME.md` (new):**
  - the 3, 10 and 30.04 s table (original FP32 NeMo, original BF16 optimized, Kea-B x3, Kea-B x2:
    latency, process VRAM, energy, power);
  - the paired x2-vs-x3 result and x2 recognition on dev;
  - the method, the cross-session comparison, the research-runtime sentence;
  - the requirements: SM120 (RTX 5090) only, measured with torch 2.11.0+cu128, triton 3.6.0,
    nemo_toolkit 3.0.0 and Python 3.13;
  - links.

**Kea-T** (coordinator's decision; AJ can override at approval).
- **At a glance.** The table stays on the published, runnable x3 runtime (expanded mode of
  `parakeet-ternary/inference`), byte for byte as live.
- **One sentence under the table.** "Our research runtime with two-part INT8 activations is
  faster still": 10 s latency and approximate energy against the BF16 build, "measured in a later
  session", with a link to the RUNTIME.md section. The compact-mode sentence stays.
- **Results.** One sentence: on the development sets, both GPU runtimes give the same mean WER
  as FP32 decoding (6.09%).
- **`RUNTIME.md`.** The live document is kept and gains a section "Research runtime with INT8x2
  activations", placed after Requirements. The section holds:
  - the research-runtime sentence and the requirements;
  - the 3/10/30.04 s table: FP32, BF16, Kea-T expanded x3 (b5, this session), Kea-T expanded x2
    (b4);
  - the paired b4−b5 result, recognition on dev, the method, and the cross-session comparison.
  The x3 paragraph now says "three residual INT8 components ("x3")" and that the research
  runtime uses two. The R-hybrid runtime is not named.

Front matter (YAML, model-index) is byte-identical to the live cards. No claim beyond the sources in
section 6 is added.

## 2. What is NOT changed

- **Weights, loaders and legal files.** No weights, loader, LICENSE or NOTICE is staged, so the
  upload cannot touch them.
- **Kea-T's EVALUATION.md and TRAINING.md** are not staged. Their rendered templates must equal
  the live Hub bytes, and the links in the new documents are resolved against them.
- **The release records stay as they are.** `parakeet-binary/hf/stage.py` (the 2026-10-21 release)
  and `parakeet-ternary/hf/stage.py` (2026-10-04) remain the records of the full uploads; they
  are not rerun. Two tests read release-time inputs instead of the working tree:
  - `parakeet-binary/tests/test_hf_release.py` `CardTest` now reads the card template at the
    release commit `ca4bc02`, so it keeps testing the released card;
  - the new card is tested by the new contract tests.
- **The previous Kea-T card-update records are not overwritten.** These are the staging folder
  and the reports `rajb3__kea-t.card-update.*`. The new outputs are named `*.card-update-x2*`.

## 3. Definitions that the cards rely on

- **x2 / x3.** Each encoder activation enters the INT8 Tensor Core multiplies as two (x2) or three
  (x3) residual INT8 parts, with int32 accumulation and FP32 outputs. The weights are the same.
  - Kea-B x2 is the B-hybrid runtime (`gpu1bit.runtime`, roster arm b1); Kea-B x3 is b2.
  - Kea-T x2 is the expanded runtime under `gpu1bit.runtime_m1` with two parts (b4). Kea-T x3 is
    the same expanded runtime with three parts (b5), the configuration the live card publishes.
    The opt-in hybrid x3 runtime `ptx_row_best` is b6.
- **Per-clip values (this session).** Each arm ran in two fresh processes, in opposite orders
  (`clips1`, `clips2`). Each process gives the median of three 5 s windows per clip. The card
  value is the median of the two processes, which equals their mean.
  - The original's rows (FP32 from `comparison/run1`, BF16 from `comparison/run4`, both measured
    2026-10-05) used one process each, with the same harness design, clips, window protocol, GPU
    (same UUID) and driver.
- **Process VRAM.** The same quantity as Kea-T's published "process VRAM": the warm windows'
  sampled peak of NVML process memory (`max(windows[].process_sampled_peak_bytes)`), in decimal GB,
  excluding the first call and graph capture.
  - x2 session: the larger of the two processes. They agree exactly, and the three windows of
    each process are equal.
  - Not used: `report-test.json` `p_memory`. It is `process_sampled_peak_including_first_bytes`,
    which includes the first call and graph capture: 1.12 instead of 1.01 GB for MB1 at 3 s.
  - Cross-session check: Kea-T's x3 expanded runtime (b5) reproduces its published process VRAM
    (`run4/packed_expanded.json`, equal to `rtx5090.json`) byte for byte at all three lengths.
    VRAM against the BF16 build is therefore stated without "approximate".
- **Paired results.** These are medians of per-utterance paired changes on dev_24 (24 development
  utterances, 8 per target length), at full scope (waveform to text) and encoder scope, from
  `report-dev.json`, with the count of utterances where x2 was lower.
  - The same pairs on the 24 historical test utterances (`report-test.json`) are stated only as
    "agree within N points". N is the ceiling, to 0.1, of the largest absolute difference over
    the pairs and metrics that document shows.
- **Cross-session reproducibility.** Two runtimes measured in both sessions on the same clips bound
  how well numbers carry over:
  - Kea-T's x3 expanded runtime (b5 against `run4/packed_expanded.json`, the BF16 row's session);
  - the opt-in hybrid (b6 against `cuda-iteration-20261007/packed_cuda_best.json`, as embedded
    in `report-test.json` `m1_recorded` with its SHA-256).
    b6's earlier record used 0.3 s windows, not 5 s; it is kept as the more conservative reference.
  The documents call both "reference runtimes" (Kea-T's RUNTIME.md: "Kea-T's earlier runtimes")
  and do not name the hybrid. The cards state the ceiling of the largest absolute latency
  difference (1%) and of the largest absolute energy difference (8%) over both runtimes and all
  three clips. This is why the
  cross-session energy figures are labelled approximate, and their percentages are rounded to
  whole numbers with "about".

## 4. Kea-T lead decision

**Final: lead with x3; x2 is one sentence.** This is the coordinator's decision. My first draft
led with x2; it was replaced, for these reasons:
- Kea-T's glance numbers come from `parakeet-ternary/inference`, which people can run from the
  repository.
- M1-x2 runs only in the research runtime.
- Leading with a runtime that readers cannot run would put unreproducible numbers in the
  headline.

The cross-session x2 claims hold:
- Latency carries over within 1%, so the 25.9% figure stands.
- Energy is labelled approximate. An 8% shift would not reverse "about 30% less".

**Correction to the brief (confirmed by the x2 thread).**
- "−6.7% energy, latency unchanged" is b4 against b6 (R-hybrid), which Kea-T never published.
- Every Kea-T x2-vs-x3 statement therefore uses b4−b5: 10.1% less energy and 3.0% lower
  latency, full scope, dev, 24 of 24 each.
- Dev and test agree within 0.4 points for this pair. Across all M1 pairs the gap reaches 0.64
  (b4−b6), but the documents state the 0.4-point bound only for b4−b5.

## 5. Staging: the shared engine and the named checks

`cardkit.py` holds the engine. Each model's `stage_card.py` holds its configuration and its
source-specific checks. `x2facts.py` loads and checks the shared x2 sources and computes the shared
runtime phrases. The CLI is identical:

    ./python hf/stage_card.py stage | upload --approved-bundle-sha256 HEX | verify refs/pr/N | pre-merge refs/pr/N | verify main

It runs from `finetune/parakeet-ternary` (Kea-T) or `finetune/parakeet-binary` (Kea-B), on the CPU
only, with no model load and a few read-only Hub calls.

Every check is recorded by name. The set of names is fixed per model (`required_checks()`).
Success is the conjunction of all of them, and a missing or unexpected name fails the run.

**A. Hub (read-only).**
- `hub main revision == pinned <7 hex>`.
- `hub file set == pinned file set` (release record files, `.gitattributes`, live detail
  documents).
- `hub <file> SHA-256 and size == release record` for every unchanged non-document file:
  - Kea-B: its 2026-10-21 stage report;
  - Kea-T: its 2026-10-04 stage report, with NOTICE as the pinned post-rename `hf/NOTICE`.
  LFS files are compared through the Hub's LFS SHA-256 and size; small files are downloaded at
  the pin into a temporary directory and hashed.
- `hub .gitattributes SHA-256 == pinned`.
- `hub <doc> SHA-256 == pinned live document` for each live document:
  - Kea-B: README.md `48b6a7da…`;
  - Kea-T: README.md `87a6e5fd…` and the three detail documents, from the card-update stage
    report, whose `verify main` passed at c0f2896.
- `new document names are not on the hub` (Kea-B: `RUNTIME.md`).
- Per model:
  - Kea-T keeps its NOTICE first-line check and the Kea-B public-export check.
  - Kea-B gains `hub rajb3/kea-t is public; export.safetensors LFS SHA-256 and size == Kea-T
    release record`, because its card links Kea-T and states its numbers.

**B. Sources.**
- Kea-T keeps every existing source check.
- Kea-B: its stage report PASSED; manifest and test.json are bound to the published export;
  TEST.md rows equal test.json.
- Both models: the shared x2 checks.
  - `x2 sources SHA-256 == pinned`: report-dev, report-test, gates, the four gate-arm summaries,
    the ten per-clip test artifacts, both full-precision JSONs and `run4/packed_expanded.json`.
    The x2 thread may regenerate reports; a changed file fails until reviewed and re-pinned.
  - `x2 arms are the published exports and the gated identities`:
    - the report labels' export SHA-256 equals the stage report's (MB1 `9873d2f1…`, M1
      `a287e977…`);
    - the report identities equal the gate identities g1, g3, g4 and g5 and the summaries'.
  - `x2 dev gates passed`: gates `pass`, `mb1.pass`, `m1.pass`, every summary complete.
  - `x2 clip artifacts == report-test`:
    - each arm's two per-clip artifacts carry the arm's identity;
    - power limit 400 W, the same GPU UUID and driver, args `seconds` 5 and `repeats` 3;
    - their per-clip summaries and transcripts equal report-test's `p_clips`.
  - `full-precision rows: same clips, GPU, driver, power limit and windows`:
    - FP32 and BF16 have the same clip IDs and durations as report-test;
    - the same GPU UUID and driver;
    - 400 W and the same window protocol;
    - BF16's summaries equal `rtx5090.json`'s published BF16 row.
  - `FP32 dev WER == the run summaries`: the x3 summaries' `cross_check` names the run summary,
    and its mean equals the run summary's.
  - `x2 runtime environment == the gate identities`:
    - the four gate identities' `versions` (torch, triton, nemo_toolkit) agree with each other and
      with the per-clip artifacts' `variant_facts`;
    - the wrappers they hashed (`finetune/parakeet-{ternary,binary}/python`) equal the wrappers
      running this staging, which exec the repository's `.training-runtime` python3. Python's
      version is read from the staging interpreter.
  - `x2 runtime published on origin/main (gated selection, runtime README, measured kernels)`.
    On `origin/main` (`git show`), all of these must hold:
    - `gpu1bit/shared/selected.json` has SHA-256 `de3c18c4…`, the gated x2 selection;
    - `gpu1bit/README.md` and `gpu1bit/RESULTS.md` exist;
    - `gpu1bit/measured-x2-parts/` exists, and its file hashes equal `actprec/parts.py`'s
      `PINNED`;
    - `actprec/parts.py`'s SHA-256 equals its entry in `freeze-x2b-20261021.sha256`, which is
      pinned.
    Until the x2 thread's PR merges, this check fails, and so does the GitHub-link check on
    `gpu1bit/README.md`. Neither card can be uploaded before then.

**C. Rendering.**
- `no unrendered placeholder or TODO marker`. It stays as a guard; the drafts have no TODO
  markers left.
- The card YAML check (per model).
- `card front matter == live card front matter`.
- `detail documents have no front matter`.
- `unstaged documents == live hub documents` (Kea-T EVALUATION.md and TRAINING.md; vacuously
  true for Kea-B).

**D. Numbers.**
- **Tables.** Each table is exactly its rows computed from the sources:
  - at-a-glance;
  - card WER;
  - Kea-T's EVALUATION.md WER and its x3 runtime table (equal to `inference/README.md`);
  - the x2 runtime table and the paired table in RUNTIME.md.
- **Computed phrases.** `<doc> computed numbers == sources`: every phrase computed from its
  source (section 6) is present at token boundaries. The phrase may not be preceded by a word
  character, `.`, `,`, `+`, `-` or `−`, and may not be followed by a word character, or by `.`
  or `,` and a digit.
  - A phrase whose condition fails at the source is `None` and always fails, for example
    "roughly constant" or "grows with clip length".
  - Every comparison between durations carries its duration inside the phrase.
- **Carried phrases.** Phrases carried verbatim from the live documents must appear in them.
- **Coverage.** `<doc> every number is checked`: after the checked tables, computed phrases and
  carried phrases are removed, no number (signed token) remains.
- **Loss.** `every number in the live documents appears in the new documents`: every number of each
  live document being replaced appears in the new document set, except an explicit `DROPPED`
  list (empty).

**E. Links.**
- `relative links resolve to documents and headings`: against the full document set, including
  unstaged live documents.
- `card links every detail document; each links back`.
- `GitHub links point at paths on origin/main`, read from the local `origin/main` ref (refreshed
  with `git fetch` before staging).
- `other absolute URLs appear in the live documents or are allowed`. The allowed list is the
  sibling model's page.

**F. Staging.**
- `staged file set == the changed documents`.
- `staged <doc> SHA-256 == rendered`.

**Promotion.** Files are written to a temporary sibling folder and promoted atomically to
`<staging>/<repo>.card-update-x2` only if every check passes. Stale outputs are removed first. On
any failure, exception or exit, no folder is left, only `...card-update-x2.stage.FAILED.json`.

**Publishing.** The subcommands are unchanged from the Kea-T plan, section 3, parameterized by
model:
- **The approval token is a bundle digest** (Codex r1). It is the SHA-256 of a canonical manifest:
  `repo <id>`, `parent <pinned revision>`, then `file <name> <SHA-256> <bytes>` for every uploaded
  document in sorted order, one per line. `stage` records the manifest and the digest in the report
  (`bundle_manifest`, `bundle_sha256`) and prints both.
- `upload --approved-bundle-sha256 HEX` is the script's only Hub write. It creates one
  `create_commit(create_pr=True, parent_commit=<pin>)` with exactly the staged documents. It
  refuses, with no write, unless all of these hold:
  - the report PASSED;
  - the staged bytes equal the report;
  - HEX equals the bundle digest recomputed from the exact bytes about to be uploaded, and the
    report's `bundle_sha256`;
  - main is still the pin.
  A restage that changes any uploaded file, RUNTIME.md alone included, therefore invalidates an
  earlier approval.
- `verify refs/pr/N`:
  - main is still the pin, and the PR is one commit on it;
  - the file set is the pinned set plus new documents;
  - every other file, the unstaged documents included, is unchanged (size, LFS SHA-256, blob id);
  - the staged documents equal the report.
- `pre-merge refs/pr/N`: the verify report PASSED, main is still the pin, and the PR head is the
  verified head.
- `verify main`: the same content checks, and the commits since the pin are only the card update.

**No other Hub write to either repository is allowed between staging and the merge.**

**Kea-T re-pin.**
- `PINNED_REVISION` becomes c0f2896.
- `OLD_CARD_COMMIT` becomes `b579a94`, whose templates render to the live documents.
- The staging name becomes `rajb3__kea-t.card-update-x2`.

## 6. Source of each new claim

All x2 paths are under `/mnt/hd/wilderness-labs-stt/parakeet-binary/gpu1bit/H/`.

| Claim | Source |
| --- | --- |
| Per-clip latency, energy and power, this session (MB1 x2/x3, M1 x2/x3) | `x2/bench/x2b-20261021/report-test.json` `p_clips`, median of the two orders; equal to `bench/x2b-20261021-test-clips{1,2}/clips-<arm>.json` |
| Process VRAM, this session | per-clip artifacts `max(windows[].process_sampled_peak_bytes)`, the larger of the two orders (which agree exactly) |
| Original FP32 rows | `/mnt/hd/wilderness-labs-stt/parakeet-ternary/comparison/run1/nemo_fp32.json` `summary`, `max(windows[].process_sampled_peak_bytes)` |
| Original BF16 rows | `.../comparison/run4/nemo_bf16_graphs_fused_decoder.json`, equal to `rtx5090.json`'s BF16 row |
| 10 s percentages against BF16 (latency and VRAM one decimal; energy whole, "about") | computed from the two rows above |
| Latency saving per transcription at each length; energy saving grows with length | computed; the phrase renders only if max/min latency saving ≤ 1.25 and the energy saving strictly increases |
| Paired x2 vs x3, full and encoder, counts | `report-dev.json` `full`/`encoder` `.pairs.{b1-b2, b4-b5}` |
| dev_24 composition (24; 8 per 3, 10 and 30 s target) | `report-dev.json` pair `n` and `by_target` |
| Agreement with the test utterances within N points | max absolute difference of report-dev against report-test over the shown pairs, ceiling to 0.1 |
| Average power x2 / x3 | `report-dev.json` `absolute_median.full.<arm>.average_watts` |
| Process VRAM the same for x2 and x3 (Kea-B only); the expanded runtime's VRAM identical across sessions | per-clip artifacts; `run4/packed_expanded.json` |
| Dev WER x2, x3 and FP32; 22,418 utterances | `x2/gates-x2g-20261021b.json`; `fulldev/x2g-20261021b/<arm>/summary.json`; the run `summary.json` `dev_mean_wer` (named by the x3 summary's `cross_check`) |
| Cross-session: latency within 1%, energy within 8% | b5 against `run4/packed_expanded.json`, and b6 against `report-test.json` `m1_recorded.clips.packed_cuda_best`, per clip; ceiling of the max absolute differences |
| 400 W limit, board default 600 W, 5 s windows, three windows | per-clip artifacts `gpu`, `args`; equal in the full-precision JSONs |
| torch 2.11.0+cu128, triton 3.6.0, nemo_toolkit 3.0.0 (research runtime) | `fulldev/x2g-20261021b/<arm>/identity.json` `versions` (pinned), equal to the per-clip `variant_facts` |
| Python 3.13 | the staging interpreter, whose wrappers equal the ones hashed in the identities (both exec `.training-runtime/bin/python3` → python3.13) |
| SM120 (RTX 5090) | per-clip artifacts `variant_facts.capability` and `gpu` |
| Kea-T research sentence: 4.30 ms, 25.9% lower, about 1.33 J, about 30% less | b4 against the BF16 row at 10 s (same values as the x2 table) |
| Kea-T numbers already published (x3 table, glance-era percentages) | unchanged sources of the Kea-T plan, section 6 |
| Kea-B training and size numbers | `parakeet-binary/hf/stage.py` `card_facts` (test.json, run summary, manifest, stream sources) |

## 7. Upload procedure (AJ only, after review)

1. Codex reviews this plan, the templates, the scripts, the tests and both staged folders. The
   review is iterated until clean, with `review/SCOPE-RULES.md` appended.
2. Wait for the x2 thread's PR to merge into `origin/main`, then `git fetch`.
   - If the README path differs from `x2facts.MAIN_RUNTIME_README`, correct it.
   - Re-pin the x2 sources if the thread regenerated any report.
   - Rerun both `./python hf/stage_card.py stage`. Both must PASS with no failed check, the TODO,
     publication and GitHub-link checks included.
3. AJ reads each staged folder, README.md and RUNTIME.md, and approves its bundle: the
   `bundle_sha256` printed by `stage` and recorded in
   `<staging>/rajb3__kea-{b,t}.card-update-x2.stage.json`. The manifest lists every file's SHA-256.
4. For each model, from its package directory, open the Hub PR (never a direct write to main):

       ./python hf/stage_card.py upload --approved-bundle-sha256 <the bundle_sha256 AJ approved>

5. `./python hf/stage_card.py verify refs/pr/N` must pass. AJ reviews the rendering at
   `https://huggingface.co/rajb3/kea-<b|t>/blob/refs%2Fpr%2FN/README.md` and `RUNTIME.md`.
6. Immediately before merging, `./python hf/stage_card.py pre-merge refs/pr/N` must pass. AJ
   then merges at once.
7. `./python hf/stage_card.py verify main` must pass. AJ then checks that the relative links and
   anchors open on the Hub.
8. Commit the templates, scripts, tests and this plan to GitHub only with AJ's go-ahead.

## 8. Tests (CPU only, seconds)

`finetune/parakeet-ternary/tests/test_stage_card.py` (Kea-T, updated) and
`finetune/parakeet-binary/tests/test_kea_b_card.py` (new). Both run against the real templates and
sources, with a fake Hub built from the release records and the live documents, and the local
`origin/main` listing. They check:
- The real documents fail exactly two checks: the GitHub-link check on `gpu1bit/README.md` and
  the x2 publication check. In the published variant, `origin/main` carries the gated selection,
  the README, RESULTS.md and the pinned measured kernels, and the documents pass every check.
  Without the README on the listing, only the GitHub-link check fails.
- Each check family has a failure path:
  - **numbers:** a tampered number, a reversed sign, a dropped "about" or "(approximate)", a
    removed duration qualifier, a swapped x2/x3 label, an uncovered number, a dropped live number;
  - **tables:** a tampered table cell or an extra row;
  - **sources:** a source tampering;
  - **rendering:** a placeholder or TODO, a front-matter change, an unstaged document that
    differs from the Hub;
  - **links:** a broken relative link or anchor, a missing GitHub path, an unknown URL;
  - **Hub:** a changed Hub file, a moved revision, a document name already taken, a private
    sibling.
- Staging promotes atomically, and every failure, exception or `SystemExit` leaves no folder.
- The check-name set is enforced.
- `upload` refuses without a write on a failed report, a changed staged file, a wrong approval or a
  moved main. Otherwise it creates one PR on the pinned parent with exactly the staged documents.
- Approval is checked against the bundle:
  - Codex r1's regression: approve, restage with only RUNTIME.md changed, and the old approval
    fails with no write; the new bundle passes. On Kea-B it uses Codex's exact edit, dropping "not"
    from "were not re-scored".
  - A README-only hash is refused.
  - The manifest is canonical: order-independent, and covering the repository, the parent and the
    sizes.
- `verify`, `pre-merge` and `verify main`:
  - `verify` passes on an unchanged PR;
  - `verify` fails on a changed unstaged document or non-document file, and on a moved main;
  - `pre-merge` fails on a moved main or PR head;
  - `verify main` fails on an intervening commit.
- The scripts' only Hub write is that one `create_commit`.
- Both card bodies are at most 135 lines and lead with "At a glance" then "Results".
- Kea-T: the existing guarantees and tests are kept. Changed expectations are listed in section 9.

## 9. Contract changes to existing tests

- **Kea-T `test_stage_card.py`:**
  - **At-a-glance contents.** These are unchanged from live: the x3 runnable runtime. The tests
    add the research-runtime sentence and the dev-WER sentence for both runtimes, and check that
    the card names neither gpu1bit nor the hybrid.
  - **Real-document test.** It now asserts the two publication blockers (the README link and the
    publication check), plus a published-variant pass. Before, it asserted a full pass. Staging, upload and verify tests use the published
    variant.
  - **Loss-test example.** The dropped-live-number example moved from EVALUATION.md, which is now
    unstaged and guarded by `unstaged documents == live hub documents`, to RUNTIME.md
    (`41.8% less`).
  - **Pins.** `OLD_CARD_COMMIT`, `PINNED_REVISION` and the live-document pins are moved to
    c0f2896. The FakeHub's pinned revision gains the three live detail documents.
  - **Staging name.** It is now `rajb3__kea-t.card-update-x2`.
  - **Check names.** These generalize: `hub main revision == pinned c0f2896`,
    `hub file set == pinned file set`, `relative links resolve to documents and headings`,
    `staged file set == the changed documents`, and the live-document wording of the carried,
    loss and URL checks. Each family keeps its failure tests.
  - **Upload set.** Upload and verify cover the two staged documents. `EVALUATION.md` and
    `TRAINING.md` move to the unchanged set.
  - **Test infrastructure.** The `GITATTRIBUTES_SHA256` patch now targets the configuration
    object.
- **After PR #6 merged (554986c), 2026-10-22.**
  - **Real-document tests.** They now require every check to pass against the real `origin/main`.
    The two-blocker behaviour is tested on a synthesized pre-merge state (`cardfake.unpublished`:
    the x3 selection pointer, no runtime README, no measured kernels).
  - **`x2 runtime environment == the gate identities`.** Main's commit 101b314 added an opt-in
    `PARAKEET_PYTHON` override to `finetune/parakeet-ternary/python`. The gate identities hashed
    the earlier wrapper, so the check now requires three things:
    - the identities' wrapper hashes equal the wrappers at the pinned commit b579a94;
    - the staging interpreter resolves to the repository's `.training-runtime/bin/python3`;
    - `PARAKEET_PYTHON` is unset.
    New failure tests cover the other interpreter and the override.
- **Kea-B `test_hf_release.py`.** `CardTest` reads the template at the release commit `ca4bc02`.
  Its expectations are unchanged.

## 10. Open items

**Deferred: Kea-B weight-footprint row.** AJ decided on 2026-10-22 to skip it for now. Kea-B's glance
table has the weight file, test WER, latency and energy rows.

**The x2 thread's PR.** It brings the README, RESULTS.md, the code and `measured-x2-parts/`. When
it merges, `git fetch` and restage. The GitHub-link and publication checks must then pass.

**Resolved with the x2 thread.**
- Clips: the same as `rtx5090.json`, verified.
- Memory: warm process VRAM, verified. M1-x3 reproduces the published value byte for byte.
- Comparator: b5.
- Pins: they stand.
- Dates: the full-precision rows are 2026-10-05. The documents use no dates.
- Test: not re-scored (AJ).

**Not added.** The 512-utterance exact-match sentence for Kea-B's RUNTIME.md stays out unless the x2
thread provides its source record.

## 11. Review log

**Codex r1 (2026-10-22): NOT CLEAN, 0 blockers, 1 major, 1 minor**
(`/mnt/hd/wilderness-labs-stt/parakeet-binary/review/card-update-x2/r1.md`). Fixes, contract test
first:

1. **[major]** The approval bound only README.md. A restage that changed only RUNTIME.md could reuse
   it; Codex reproduced this on Kea-B by dropping "not" from "were not re-scored", and 47/47
   preparation checks plus 4/4 upload checks passed.
   - The approval is now the bundle digest (section 5, Publishing). `--approved-readme-sha256` is
     replaced by `--approved-bundle-sha256`, and the upload check is renamed to "approved bundle
     SHA-256 == bundle of the bytes to upload". Procedure steps 3 and 4 are updated.
   - New tests:
     - the runtime-only restage regression, on both models;
     - an edited folder with a rewritten report bundle is refused;
     - canonical manifest.
   - This is a contract change to the upload API. The Kea-T card-update path of
     plans/card-update.md is historical and was not rerun; its staging script is the configuration
     replaced by this plan.
2. **[minor]** "measured in both sessions" conflated two historical reference sessions: b5 against
   the 2026-10-05 run4 record, b6 against the cuda-iteration record. The wording is now
   "remeasured against their earlier records" in Kea-B MODEL_CARD.md and RUNTIME.md and in Kea-T
   RUNTIME.md. The bounds (1% / 8%) and the approximate-energy qualifier are unchanged.

**Codex r2 (2026-10-22): CLEAN.**

**Publication (2026-10-22).** AJ approved both staged bundles and said "Submit". The bundles were:
- Kea-B: `acac42f0e9e27796ba1badceca82a6d46e711c25b1b8686843dac61e97ebdc83` (README `d51659df…`,
  RUNTIME.md `565aa379…`);
- Kea-T: `330504853d922fbb3e3ec473b2fc8e2457c75449269ea02592d7ef048de86f7b` (README `c623c1c8…`,
  RUNTIME.md `b5abd82b…`).

Before the upload, both stages had passed against origin/main 554986c (Kea-B 50/50, Kea-T 66/66),
and both Hub mains were at their pins.

**Kea-B, first.**
- `upload` 4/4: Hub PR #1 (https://huggingface.co/rajb3/kea-b/discussions/1), commit
  `4d9f18b4b1a6fb494f88e71017842f89d9193f9d` on parent `5e7c298`.
- `verify refs/pr/1` 16/16, `pre-merge refs/pr/1` 3/3. Merged at once with
  `HfApi().merge_pull_request`.
- Main is `cfb1353488655282683ba699c402ae3b16b6de30`, one squash commit on the pin.
  `verify main` 15/15.
- **Pages:** the model page and README.md/RUNTIME.md (blob and raw) return HTTP 200.
- **Links and anchors:** all 9 link targets return 200. The anchors `use` (card) and
  `comparing-across-sessions` (RUNTIME.md) are present in the rendered pages.

**Kea-T, second.**
- `upload` 4/4: Hub PR #2 (https://huggingface.co/rajb3/kea-t/discussions/2), commit
  `7ace49ca203cb781b1f3fbbf04e40f945920dbee` on parent `c0f2896`.
- `verify refs/pr/2` 18/18, `pre-merge refs/pr/2` 3/3. Merged at once.
- Main is `c4a1f10ad23015ea43e8fa0802c30795e0442c4f`, one squash commit on the pin.
  `verify main` 17/17.
- **Pages:** the model page and all four documents (blob and raw) return 200.
- **Links and anchors:** all 27 link targets return 200. These anchors are present in the
  rendered pages:
  - `use` (card);
  - `limitations` (EVALUATION.md);
  - `training-data-and-licenses` (TRAINING.md);
  - in RUNTIME.md: `research-runtime-with-int8x2-activations`, `configurations`,
    `method-and-comparing-across-sessions`.

**Reports:** `<staging>/rajb3__kea-{b,t}.card-update-x2.{stage,upload,verify-refs_pr_N,pre-merge-refs_pr_N,verify-main}.json`.

`PINNED_REVISION` in both `stage_card.py` files now names the pre-update revision, so `stage`
fails by design. A later card update starts a new plan against the then-current main.
