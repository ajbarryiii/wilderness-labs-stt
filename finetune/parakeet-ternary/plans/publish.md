# Plan: publish the ternary Parakeet model on Hugging Face

Written 2026-10-04 after Phase 3 (DESIGN.md "Main-run and test results").
User approved publication as `rajb3/parakeet-tdt-0.6b-v2-ternary`.

## What is published

Exactly the M1 final export that the test scores describe: run
`main-M1-P2-lr5e-4`, step 250,000, `export.safetensors` 180,797,564 bytes,
SHA-256 equal to `eval/M1-precheck.json` `export_sha256` (precheck passed
2026-10-04). Plus: `manifest.json`, `reconstruction.json`, the three tokenizer
files exported with it, `load_ternary.py` (standalone loader and CLI), `README.md`
(model card), `NOTICE` (attribution and changes) and `LICENSE` (CC-BY-4.0 legal
code, fetched from creativecommons.org). Nothing else; no checkpoints, no
training state, no datasets, no evaluation records.

## License and attribution

The base model is CC-BY-4.0 (NVIDIA). The derivative is released under
CC-BY-4.0 with: attribution to NVIDIA and the base revision, a statement that
it is modified and what changed (ternary encoder projections and pointwise
convolutions via QAT; storage format), the licence text, and the training-data
sources with their licences. No endorsement by NVIDIA is implied.

## Claims the model card may make (all must match the cited artifact)

- Test WERs and differences: `results/TEST.md` / `results/test.json`,
  identical rounding.
- Size: 180.8 MB versus 2,472 MB (`test.json` `m1_export_mb`,
  `original_nemo_mb`).
- Module set and parameter counts: export `manifest.json`.
- Training recipe and budget: DESIGN.md "Main run (M1)" and the run's
  `summary.json` (steps, audio hours).
- Limitations: English only; activations floating point, so no speed or energy
  claim; greedy decoding only; Common Voice gap; possible YouTube overlap
  between YODAS training data and GigaSpeech test, equally affecting the
  original.

## Steps

1. `hf/stage.py rajb3/parakeet-tdt-0.6b-v2-ternary` assembles the folder under
   `/mnt/hd/wilderness-labs-stt/parakeet-ternary/hf-staging/` and refuses unless
   the export is the precheck-verified M1 final export. It validates the card's
   YAML and placeholders and runs a standalone check: the staged loader (no
   repository imports) must produce state_dict tensors identical to the
   repository's `export.load_export`, and identical greedy transcripts on four
   fixed LibriSpeech test-clean clips. Model loading runs in a memory-capped
   `./heavy` unit.
2. Clean-environment check: the staged `load_ternary.py` transcribes a test
   clip with only NeMo's dependencies importable.
3. Codex (gpt-6-astra, xhigh) reviews this plan, `hf/` and the staged folder,
   iterated until clean (no open blocker or major).
4. Create the Hub repository as PRIVATE and upload there:
   `hf repo create rajb3/parakeet-tdt-0.6b-v2-ternary --type model --private`, then
   `hf upload rajb3/parakeet-tdt-0.6b-v2-ternary <staging> . --repo-type model`.
5. Verify the private repository: file list equals the staged report; the LFS
   SHA-256 of `export.safetensors` equals the precheck hash; every other file's
   hash equals the staging report; card metadata parses; model-index present.
6. Round trip from the private repository (authenticated): download into an
   empty directory and run the standalone CLI on a test clip in the clean
   environment; the transcript must match the staged check for that clip.
   Only after 5 and 6 pass, make the repository public
   (`HfApi().update_repo_settings(repo_id, private=False)`) and re-check that
   it is reachable anonymously.
7. Link the Hub model from `finetune/parakeet-ternary/README.md` and the
   top-level README (local edits; committing and pushing to GitHub only with the
   user's go-ahead).

## Code link

The user approved pushing the experiment code; it is on GitHub `main` (commit
4aa2845, 2026-10-04), so the card's link to `finetune/parakeet-ternary` resolves.
The `hf/` folder is pushed after the upload.

## Resource caps

Model loading only through `./heavy` (24 GB, 30 min). Upload size about 182 MB.
