# Kea-B release: Hugging Face kit for MB1 (`rajb3/kea-b`)

**Status: 2026-10-21, PUBLISHED at https://huggingface.co/rajb3/kea-b (Hub commit
`5e7c298bc0eee174394a3fba88d789dc7c53ee4d`). Re-staged 61/61 PASSED; AJ approved README `48b6a7da…3e91`;
uploaded private; fresh-download verification passed; then made public (section 9).**

This is the contract for publishing the binary main run MB1 as **Kea-B**. It reuses the
design of the published ternary sibling Kea-T (`finetune/parakeet-ternary/hf/stage.py`): a
fixed set of named checks, success only when all of them pass, a temporary folder promoted
atomically, a JSON report, and no upload from the staging script. Nothing here trains or
re-scores a model. The test set was scored once (plans/test-scoring.md); this plan only
packages that exact artifact and the numbers it produced.

## 1. What is staged

Source: `/mnt/hd/wilderness-labs-stt/parakeet-binary/runs/main-binB-m1-lr2e-4/export/`
(MB1, select final, step 250,000, format `parakeet-binary-v1`).

| Staged file | Source | Licence |
| --- | --- | --- |
| `export.safetensors` (105,299,804 bytes) | export dir | CC-BY-4.0 |
| `manifest.json`, `reconstruction.json` | export dir | CC-BY-4.0 |
| `tokenizer/tokenizer.model`, `tokenizer/tokenizer.vocab`, `tokenizer/vocab.txt` | export dir | CC-BY-4.0 |
| `README.md` | `hf/MODEL_CARD.md` rendered with the repo id | CC-BY-4.0 |
| `load_binary.py` | `hf/load_binary.py` | MIT |
| `LICENSE` | `hf/LICENSE`, byte-identical to Kea-T's | - |
| `NOTICE` | `hf/NOTICE` | - |

Exactly these 10 files; any other file in the staged folder fails the run. Output, and the only
place `hf/stage.py` writes: `/mnt/hd/wilderness-labs-stt/parakeet-binary/hf-staging/`:
`rajb3__kea-b/` (the folder), `rajb3__kea-b.stage.json` on success or
`rajb3__kea-b.stage.FAILED.json` on failure, and a temporary `.rajb3__kea-b.tmp-<pid>/` that
never survives the run.

## 2. The loader contract (`hf/load_binary.py`)

- MIT header as in `load_ternary.py`. Imports only the standard library, `torch`, `numpy`,
  `safetensors`, `soundfile`, `nemo` and `omegaconf`; no repository module.
- `unpack_codes(packed, in_features)`: the inverse of `binary_export.pack_codes` (bit `j % 8`
  of byte `j // 8`, LSB first, bit 1 = +1, bit 0 = -1). Rejects a wrong dtype or shape and
  nonzero tail bits.
- `load(folder, device)`: refuses another format, a weights file other than
  `export.safetensors` (`manifest["file"]`), a manifest whose `files` is not exactly
  `export.safetensors` plus the three tokenizer files, or `files[export.safetensors] != sha256`;
  checks every listed SHA-256; checks each scale is FP32 `[out]` and each layer kind is
  `linear` or `pointwise_conv1d`; rebuilds `codes * scale` per row in FP32, upcasts the other
  floating tensors to FP32, builds `EncDecRNNTBPEModel` from the stored config with the bundled
  tokenizer, `load_state_dict(strict=True)`, and returns it in eval mode on `device`.
  Its state_dict must equal `binary_export.load_export`'s tensor for tensor (same keys, same
  order, same dtype, `torch.equal`).
- `transcribe(model, inputs)`: the decoding of `evaluate.transcribe_records`, which the test
  scoring used: preprocessor, encoder, `rnnt_decoder_predictions_tensor` with the model's own
  greedy TDT config, strict FP32 (autocast off, TF32 off for matmul and cuDNN), dither 0,
  pad_to 0, inputs under 30 ms zero-padded to 30 ms, batches longest first. On CUDA it also puts
  NeMo's greedy decoder in the CUDA-graph mode `no_while_loops`, the mode the test scoring
  pinned (`binary_paths.DECODER_GRAPHS_MODE`), when NeMo exposes it.
- CLI: `python load_binary.py clip.wav [...]`.

## 3. Named checks (`hf/stage.py`)

Success means every name in `required_checks()` is recorded and true, and no other name is
recorded. The names are fixed in the code, so no check can be skipped silently. Constants pinned in
`stage.py`: run `main-binB-m1-lr2e-4`; select final; step 250,000; lr 2e-4; format
`parakeet-binary-v1`; 105,299,804 bytes; weights SHA-256 `9873d2f1…9651`; manifest.json SHA-256
`f3737a53…1dbd`; reconstruction.json SHA-256 `efa75346…d234`; the CC-BY-4.0 legal code SHA-256
`9ba9550a…9411` (the value Kea-T pinned). The manifest and reconstruction hashes were computed
after the provenance cross-checks below passed on the real files, in a CPU-only dry run
(section 6). They also equal the values pinned before test scoring in
`binary_test_paths.PINNED_FILES`.

**A. Provenance** (reads files only; CPU)

1. manifest format is parakeet-binary-v1 and file is export.safetensors
2. manifest identity is the MB1 final export (run, select, step, lr)
3. export.safetensors is 105,299,804 bytes (file and manifest)
4. export SHA-256 == manifest sha256 == manifest files[export.safetensors]
5. export SHA-256 == pinned MB1 weights == binary_test_paths.WEIGHTS_SHA256
6. export directory holds exactly the pinned entries
7. precheck passed (pass true, every check ok) and recorded these weights
8. export SHA-256 == results/test.json mb1_weights_sha256
9. test scoring: attempt 1 complete, all 8 sets, source is this export
10. test scoring librispeech_clean.json SHA-256 == its set_done record and summary per_set_sha256
11. manifest format/file/sha256/files/extra/accounting/bytes == those recorded by the test scoring
12. results/test.json MB1 row == test scoring summary (8 sets and mean)
13. results/TEST.md B0, M1 and MB1 rows == results/test.json rounded to 2 decimals
14. manifest identity and config_sha256 == run summary.json
15. run summary export_dir, export_bytes and code histogram == manifest (the summary records the
    manifest's histogram without its per_module entries, so those are compared)
16. code histogram is binary (no zero codes; -1 and +1 fractions sum to 1), overall, per family
    and for each of the 264 modules
17. manifest quantized layers: 264, kinds linear or pointwise_conv1d, no bias
18. manifest lists exactly export.safetensors and the three tokenizer files
19. tokenizer files match the manifest SHA-256
20. manifest base_model is the pinned base model
21. base .nemo SHA-256 == lock.json == test scoring base_model
22. manifest config == base .nemo model_config.yaml (resolved)
23. tokenizer files == base .nemo tokenizer members
24. reconstruction.json == run summary reconstruction
25. reconstruction exact: codes, scales, greedy hyps, all 264 layers
26. manifest.json SHA-256 == pinned == binary_test_paths
27. reconstruction.json SHA-256 == pinned == binary_test_paths

Items 7 and 9 to 12 bind the export to the one test scoring
(`eval/MB1-precheck.json`, `eval/MB1-test/`, `eval/MB1-test.attempts.jsonl`). Items 12 and 13
bind the committed results files to that scoring.

**B. Files**

28. LICENSE is the pinned CC-BY-4.0 legal code
29. load_binary.py imports only stdlib and documented dependencies
30. load_binary.py carries the MIT copyright and permission notice
31. NOTICE names the base model revision and every training source with its revision
    (the repos and revisions in MB1's stream record, `stream-records/seed-20261017/stream/sources.json`)

**C. Model card** (the rendered `README.md`)

32. card has no unrendered placeholder
33. card YAML: model_name, license cc-by-4.0, base_model, library_name nemo
34. card YAML datasets == MB1's training sources
35. card model-index: one entry per test set, value == round(test.json MB1, 2)
36. card results table: exactly the TEST.md B0, M1 and MB1 rows, in TEST.md column order
37. card numbers == their sources (see section 5)

**D. Staging**

38. staged file set is exactly the expected set
39-48. staged `<file>` SHA-256 == source, one per staged file (README.md against the rendered card)

**E. Standalone** (GPU, under `paths.gpu_lock`)

The staged `load_binary.py` is imported by file path from the temporary folder and loads that
folder. `binary_export.load_export` loads the source export. Then:

49. state_dict: same keys in the same order
50. state_dict: every tensor torch.equal
51. state_dict: every floating tensor FP32
52. staged model in eval mode

Both models then go to CUDA. The reference model is decoded the way the test scoring decoded:
`binary_train.pin_decoder_graphs_mode`, then `evaluate.transcribe_records`. The staged model is
decoded with `load_binary.transcribe`. The four clips are LibriSpeech test-clean
`1089-134686-0000` to `-0003`, the same clips Kea-T used.

53. decoder CUDA-graph mode no_while_loops for both models (as the test scoring)
54-61. per clip: staged transcript == repository transcript; staged transcript == test-scoring
    hypothesis (from `eval/MB1-test/librispeech_clean.json`, hash-bound by item 10)

The total is 61 named checks. Only when all pass is the temporary folder renamed (atomically, in
the same directory) to `rajb3__kea-b`, its listing re-hashed and compared with the validated
listing, and the success report written. On any failure or exception, including SystemExit, the
temporary and final folders are removed and only the FAILED report is written. Stale outputs
are removed before any check runs, so a failed run never leaves an old folder or an old success
report behind.

## 4. Staging command, resources and pass criteria

```sh
cd finetune/parakeet-binary
systemctl --user list-units 'parakeet-*'     # one heavy job at a time
./heavy kea-b-stage --mem-max 24G --runtime 30min --wait -- python hf/stage.py rajb3/kea-b
```

- `stage.py` refuses to run outside a systemd unit (no `INVOCATION_ID`), before touching any
  output, and then checks the `/mnt/hd` mount. Everything after that runs inside the staging
  lifecycle. That includes the runtime preflight: importing `binary_train` first, as
  `binary_test.py` did, so the process has the test scoring's CUDA settings, then torch, then
  refusing if CUDA is unavailable. Stale outputs are removed before the preflight, so a failed
  restage, CUDA-less or otherwise, leaves no folder, no PASSED report and a FAILED report.
- Memory: MemoryMax 24G (two FP32 models of about 2.5 GB each plus tensors; Kea-T staged
  in the same cap). Runtime cap 30 min. The GPU is used only inside
  `paths.gpu_lock("kea-b-stage")`, held from moving the models to CUDA until decoding ends.
- Log: `/mnt/hd/wilderness-labs-stt/parakeet-binary/logs/kea-b-stage.log`.
- Pass criteria: exit 0, `hf-staging/rajb3__kea-b.stage.json` with `"status": "PASSED"` and
  `"checks_passed": "61/61"`, no FAILED report, and the folder holding exactly the 10 files.
- A failure is investigated and fixed, and staging is rerun. Nothing is ever uploaded from a
  failed run, and a check is never weakened to make it pass.

## 5. What the card claims, and where each number comes from

The card (`hf/MODEL_CARD.md`) is short by design. In order: an intro, one results table, five
bullets on how it was trained, use, files, limitations, licence. Check 37 requires each of these strings to
appear, computed from its source at staging time:

| Claim | Value | Source |
| --- | --- | --- |
| export size | 105.3 MB | `results/test.json` `export_mb.MB1` |
| original .nemo | 2,472 MB | `results/test.json` `export_mb.original`, no decimals |
| Kea-T size | 180.8 MB | `results/test.json` `export_mb.M1` |
| mean WER original / Kea-T / Kea-B | 6.45% / 6.84% / 7.05% | `results/test.json` rows, 2 decimals |
| Common Voice original / Kea-B | 8.50% / 13.42% | `results/test.json` rows, 2 decimals |
| results table | 3 rows | `results/TEST.md` rows B0, M1, MB1, verbatim cells |
| model-index | 8 values | `results/test.json` MB1, rounded to 2 decimals |
| file bytes | 105,299,804 | manifest `bytes.file_bytes` |
| training steps | 250,000 | run `summary.json` `steps` |
| ramp | 62,500 steps | run `summary.json` `ramp_steps` |
| peak learning rate | 2e-4 | run `summary.json` `lr` (= 0.0002) |
| training time | 68.6 hours | run `summary.json` `finished_utc - started_utc` |
| audio seen | 33,246 hours | run `summary.json` `audio_hours_seen` |
| batch size | up to 600 s of audio | run `summary.json` `batch_seconds` |
| training clip lengths | 1 to 30 s | `sources.json` `filters.duration_seconds` |
| tested environment | NeMo 3.0.0, torch 2.11.0+cu128 | manifest `versions` |
| datasets (YAML) | 5 repos | run `summary.json` `seconds_by_source`, mapped by `sources.json` |

The card says the weights outside the binary matrices are "the remaining learned weights ...
in FP16". That follows `binary_export.export_model` and the manifest's `bytes.fp16_float_bytes`.
The FP32 feature constants and int64 BatchNorm counters are not learned weights; NOTICE and
`load_binary.py` name them.

Three statements are not numbers in a results file and are stated as facts in prose. The GPU is
one RTX 5090 (the project machine; README.md). Native 1-bit kernels are in progress
(`gpu1bit/`, `cpu1bit/`) and not part of this release. "Kea-T" is `rajb3/kea-t`. The card makes
no speed, memory or energy claim: `load_binary.py` rebuilds a dense FP32 model.

## 6. Pinning procedure (done once, recorded here)

On 2026-10-21, `stage.make_prepare("rajb3/kea-b")` ran on the real files. That covers checks A to C
(gather, provenance_from, file_checks, card_checks). It was a CPU-only, read-only run: no model
load, no GPU (`CUDA_VISIBLE_DEVICES=`), no writes, about 50 s including the base .nemo read.
All 37 checks passed. It computed these hashes:

- export.safetensors `9873d2f1a53127c673d0201db98cd3ba859ba432f334fef55e9e24fd057b9651`
- manifest.json `f3737a538da81939aabe04acd5334afd0e3cd4160ec113bd889e771b67991dbd`
- reconstruction.json `efa75346d366fbb92a49b1de72d569894c6338e84199271cbfc97d2f2696d234`
- base .nemo `d99e39955c9d3d0350d8fb7c75e40c64a2b2eaeb003883d7c941fd2e8747b28c`

The constants in `stage.py` are these values, and they equal `binary_test_paths.PINNED_FILES`,
which was recorded before test scoring. The constants were first copied from that file; this run
then confirmed them, after the cross-checks had passed. At that point the rendered card had
SHA-256 `642cae3f7c140a81fe6159cd7100bc494d2f546b59d66c6e459634b6475649f0`; it changes with any card
edit, and the staging report records the staged one. After AJ's card edit (section 9) it is
`48b6a7da9ac62afb6ab3de25c99af42a77c9d7757b973d292858638ede463e91`. Only the first dry run, before one fix,
had a failing check: the run summary stores the code histogram without `per_module`, and check 15
now compares that recorded form.

## 7. Upload procedure (after a PASSED staging run; never automated)

1. **AJ reads and approves the rendered card**, the staged
   `hf-staging/rajb3__kea-b/README.md`, before any upload, including the private one. If AJ
   asks for edits to the card, the card is edited in `hf/MODEL_CARD.md` and staging is rerun.
   The card is hash-checked in the staging report (`staged README.md SHA-256 == source`), so an
   edited card is never uploaded from an earlier staging.
2. Create the repo private, then upload the staged folder (from `finetune/parakeet-binary`):
   ```sh
   ./python -m huggingface_hub.cli.hf repos create rajb3/kea-b --type model --private
   ./python -c "from huggingface_hub import HfApi; assert HfApi().model_info('rajb3/kea-b').private"
   ./python -m huggingface_hub.cli.hf upload rajb3/kea-b \
       /mnt/hd/wilderness-labs-stt/parakeet-binary/hf-staging/rajb3__kea-b . --repo-type model
   ```
   Record the commit SHA that the upload prints.
3. Verify from a fresh download of that commit into a new, empty folder:
   ```sh
   d=/mnt/hd/wilderness-labs-stt/parakeet-binary/hf-staging/verify-$(date -u +%Y%m%dT%H%M%SZ)
   ./python -m huggingface_hub.cli.hf download rajb3/kea-b --revision <commit> --local-dir "$d/kea-b"
   ./heavy kea-b-verify --mem-max 16G --runtime 20min --wait -- \
       python hf/verify_download.py "$d/kea-b" /mnt/hd/wilderness-labs-stt/parakeet-binary/hf-staging/rajb3__kea-b.stage.json
   ```
   `verify_download.py` passes only if (a) every file in the stage report is present with the
   same SHA-256 and size, and the only other entries are the Hub's `.gitattributes` and the
   download cache `.cache/`; and (b) the downloaded `load_binary.py`, imported by file path,
   loads the downloaded folder and transcribes the four clips on CUDA, under the GPU lock,
   exactly as recorded in the stage report (which equal the test-scoring hypotheses). It writes
   `"$d/verify.json"` and exits 0 only if verification passed.
4. Only after verification passes, make the repo public:
   ```sh
   ./python -c "from huggingface_hub import HfApi; HfApi().update_repo_settings('rajb3/kea-b', private=False)"
   ```
   Then check the public page renders the card and the model-index. If verification fails, the
   repo stays private and the cause is fixed, then staging and verification are rerun.

No model file is committed to git. Only the scripts, the card template, LICENSE, NOTICE and this
plan are.

## 8. Tests

`tests/test_hf_release.py` runs on the CPU in seconds, with synthetic tensors and fake NeMo
classes. Run it from `finetune/parakeet-binary` with
`./python -m unittest discover -s tests -p test_hf_release.py`. (`./python -m unittest
tests.test_hf_release` does not work: `tests` resolves to another package on the path.) It covers:

- pack/unpack equivalence with `binary_export` and numpy, and rejection of bad packs;
- both loaders on a synthetic export giving identical state_dicts;
- the loader's refusals (hash, file set, format);
- the import allow-list and MIT notice;
- card rendering and each card check, including negative cases;
- the provenance check logic on synthetic documents;
- the required check-name set;
- the stage driver's success and failure paths (no final folder, no temporary folder, FAILED
  report only), and stale-output removal;
- a failed runtime preflight (CUDA unavailable, or a runtime import error) after an earlier
  success: the old folder and PASSED report are gone and a FAILED report exists;
- the refusal outside a systemd unit;
- the download-verification comparison;
- LICENSE identity with Kea-T's.

The real-model test is the staging run itself (section 4), which these tests do not replace.

## 9. Review log

- **Round 1** (Codex, 2026-10-21): NOT CLEAN.
  - Major: `main()` imported the runtime and checked CUDA before `stage()` removed stale outputs.
    A failed restage would therefore have left the old folder and PASSED report and written no
    FAILED report. Fixed: `runtime_preflight` now runs inside `stage()` after stale removal. The
    contract test `test_failed_runtime_preflight_invalidates_a_previous_success` covers it.
  - Minor: the card said every non-binary tensor is FP16. Fixed: the card now says "the remaining
    learned weights" (section 5).
  - Also fixed: the documented test command.
- **Round 2** (Codex, 2026-10-21): both round-1 findings resolved; 0 blockers, 0 majors, 1 minor.
  - Minor: `load_binary.load` built its expected file set from `manifest["file"]`, so a manifest
    that consistently renamed the weights file was accepted, contrary to section 2. Fixed test-first:
    `test_refuses_a_consistently_renamed_weights_file`, then the loader requires
    `manifest["file"] == "export.safetensors"`. The staging check (item 1) already pinned the name,
    so the release binding was unaffected.
- **User card edit, re-staged** (2026-10-21): at AJ's request, `## How it was made` was renamed
  `## How it was trained` and the `## Use` section was moved, unchanged, to follow it. The new
  order is Results, How it was trained, Use, Files, Limitations, License and attribution. Nothing
  else changed. The first staging (61/61 PASSED, README `642cae3f…49f0`) was replaced by a
  re-stage with the same command. The new rendered card SHA-256 is `48b6a7da9ac62afb6ab3de25c99af42a77c9d7757b973d292858638ede463e91`.
- **Upload** (2026-10-21, authorized by the coordinator after AJ approved README
  `48b6a7da9ac62afb6ab3de25c99af42a77c9d7757b973d292858638ede463e91`). Preconditions all held: the
  report was PASSED 61/61, the README hash equalled the approved one, the folder equalled the
  report's 10 files, and `rajb3/kea-b` did not exist.
  - Created the repo private and confirmed it private, then uploaded commit
    `5e7c298bc0eee174394a3fba88d789dc7c53ee4d`.
  - Downloaded that commit fresh into `/mnt/hd/wilderness-labs-stt/parakeet-binary/hf-staging/verify-20261021T223011Z/kea-b`, then ran `./heavy kea-b-verify ...
    hf/verify_download.py`. It passed: no file differences, and all 4 transcripts equal the
    staged and test-scoring ones (`/mnt/hd/wilderness-labs-stt/parakeet-binary/hf-staging/verify-20261021T223011Z/verify.json`).
  - Then made the repo public. The anonymous page returns 200, the public README hash equals the
    approved one, and the model-index has 8 values. The 5 links in the README all return 200.
