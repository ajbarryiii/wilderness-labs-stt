# Plan: test-set scoring of the binary main run MB1

**Status: 2026-10-20: written; plan review round 1 NOT CLEAN (5 major, 2 minor),
all addressed below ("Plan review round 1"); round 2 NOT CLEAN (2 major, 1
minor), addressed; round 3 NOT CLEAN (1 major, 1 minor), addressed ("Plan review
round 3"); round 4 CLEAN with no new findings. Implemented (221 tests); code review rounds 1-3 NOT CLEAN, all fixed, round 4 CLEAN (review/test-scoring/c4.md). Dev check PASS (+0.000 points) and test scored once 2026-10-20 (attempt 1 complete); results in results/TEST.md: seven-set mean MB1 7.05% (M1 6.84, B0 6.45). binary_test_paths.py SHA-256 `8e28c86f1ccb842d3eb8047dd3e38205a7cf995a39c9d745cc7a44693ed17b1f`.**

MB1 (`main-binB-m1-lr2e-4`, plans/main.md) finished 2026-10-20 05:23: full-dev
mean 6.311961686% (M1 6.086%, B0 5.575%, pilot B2 7.021%), export 105.3 MB.
This plan scores MB1 once on the test sets, the same way M1 was scored
(finetune/parakeet-ternary/plans/test-scoring.md, completed 2026-10-04), and
builds a table B0 / M1 / MB1. It is a measurement; nothing is selected on it.

## What is scored

| Arm | Model | Test sets | Status |
| --- | --- | --- | --- |
| B0 | FP32 original, pinned revision | all 8 | done 2026-09-30, `parakeet-ternary/eval/b0-pretrained`; reused |
| M1 | ternary main run, rebuilt from its export (SHA-256 `a287e977...`) | all 8 | done 2026-10-04, `parakeet-ternary/eval/M1-test`; reused |
| MB1 | binary main run, final checkpoint (step 250,000, f = 1), rebuilt from its export | all 8 | to do |

Pilot arms (A, B5, B2, C) are not scored on test; they were selection arms on dev.

## Pinned MB1 identity (recorded 2026-10-20, before any MB1 test decoding)

`binary_test_paths.py` (new) holds these constants; every check below compares
files on disk against them, not against other recorded hashes.

| File under `runs/main-binB-m1-lr2e-4/` | SHA-256 |
| --- | --- |
| `export/export.safetensors` | `9873d2f1a53127c673d0201db98cd3ba859ba432f334fef55e9e24fd057b9651` |
| `export/manifest.json` | `f3737a538da81939aabe04acd5334afd0e3cd4160ec113bd889e771b67991dbd` |
| `export/reconstruction.json` | `efa75346d366fbb92a49b1de72d569894c6338e84199271cbfc97d2f2696d234` |
| `export/tokenizer/tokenizer.model` | `5a3a82c48998709f9bc9f5db2924d99a899d20bf82550ca52272da7c7557b0a0` |
| `export/tokenizer/tokenizer.vocab` | `dbe7ec6bd066c4526080940e1614e6582b9161515e856c2c39a1d49267c88cbc` |
| `export/tokenizer/vocab.txt` | `70b25deed256d79365a3c290d6191c31cc2315d952faf18d5c3a334f46e4866e` |
| `config.json` | `3cf1478ce33e3126a288e57cb9d82b386639d5327f56067308cff0781ba8c6b8` (= summary `config_sha256`) |
| `summary.json` | `1dcecfcd1a08ec54094babfe58261d79c1df9c022b9b80a53eed716d20fb425a` |

The manifest must also have `format` `parakeet-binary-v1`, `file`
`export.safetensors` (the only accepted value), and `sha256` and
`files["export.safetensors"]` equal to the pinned weight hash. Because the
manifest itself is pinned, the NeMo config and preprocessing settings it carries
cannot change unnoticed. The export directory must contain exactly these files
(no extra or missing files).

The recorded full-dev mean is pinned as the fraction `0.06311961686131834`, and
the per-set dev WERs as `summary.json["dev_wer"]`.

## Decoding and scoring (identical to M1's test scoring)

The frozen `parakeet-ternary/evaluate.py` (SHA-256 `b3136a80...`, the hash M1's
test outputs recorded) does all decoding and scoring: its `expand_sets`,
`resolve_manifest`, `evaluate_set` and `summarize`, the same manifests,
greedy-batch TDT with the model's own decoding config, strict FP32 (TF32 and
autocast off), Whisper normalizer, empty-reference rule, corpus WER, **batch
size 64 and max_batch_seconds 1200** (M1-test's recorded values, fixed in code,
not CLI options in production). evaluate.py cannot load a `parakeet-binary-v1`
export, so a new thin entry point does only the loading:

- **`binary_test.py`** (new, `finetune/parakeet-binary/`): loads the MB1
  export with `binary_export.load_export` (the loader MB1's full-dev score used)
  on CUDA; pins the greedy TDT decoder's CUDA-graph mode to
  `binary_paths.DECODER_GRAPHS_MODE` ("no_while_loops") with `binary_train`'s
  pin/check helpers and checks it before and after every set. M1-test decoded in
  this mode too: its log (`parakeet-ternary/logs/test-m1.log`) shows NeMo's full
  CUDA graph compilation failing and falling back to native PyTorch CUDA graphs.
- **Numerical settings** follow M1-test, not training: evaluate.py's
  `strict_fp32` and `inference_settings`, with no extra cuDNN deterministic or
  benchmark flags set by binary_test.py. Importing `binary_train` sets CUDA/numba
  environment variables (as for MB1's dev scoring); binary_test.py records
  `torch.backends.cudnn.deterministic`, `.benchmark`, the relevant environment
  variables and the torch/NeMo/CUDA versions in every summary so the dev check
  and the test run are shown to use identical settings.
- The source record is evaluate.py's `describe_source("export", ...)` shape with
  description "binary export, rebuilt with dequantized FP32 weights", plus the
  binary manifest and the pinned identity.
- It holds the shared GPU lock (`paths.gpu_lock`) for the whole run.

## Scorer provenance

After code review is CLEAN and before the dev check, the SHA-256 of each scoring
source is pinned in `binary_test_paths.SCORER_HASHES`: `binary_test.py`,
`binary_report_test.py`, `binary_export.py`, `binary_train.py`,
`binary_paths.py`, `binary_quant.py`, `parakeet-ternary/{evaluate,export,quant,
paths,train}.py`, `whisper-ternary/wer.py`, the loaded Transformers
`transformers/models/whisper/english_normalizer.py` (the normalizer
implementation evaluate.py imports; its installed path and digest, recorded by
the implementer before the dev check) and the Whisper `normalizer.json`, and `binary_test_paths.py` itself is
pinned by its hash in this plan's Status line once frozen. At start-up
binary_test.py verifies each pinned file's hash and that the imported module's
`__file__` is that exact path (no shadowing), and records the hashes in every
output. The report refuses outputs whose recorded scorer hashes differ from the
pinned ones or from each other (dev check vs test).

## Dev cross-check (a launch prerequisite for test)

`binary_test.py --dev-check` decodes the five dev sets at batch 64 into
`eval/MB1-dev-check/` and writes `gate.json` last, with the per-set and mean
WER, the recorded values, the differences, the pinned identity, scorer hashes
and numerical settings. It passes only if:

- the mean is within **0.02 points** (0.0002 as a fraction) of
  0.06311961686131834, and
- every set is within **0.10 points** (0.0010) of its recorded dev WER (so
  opposite per-set errors cannot cancel inside the mean tolerance).

MB1's recorded number came from train.DevEvaluator at batch 32; batch 64 can
move greedy results slightly through padding. If the gate fails, the test run
cannot start; the next step is a batch-32 dev reproduction
(`--dev-check --dev-batch-size 32`, dev only, into `eval/MB1-dev-check-b32`) to
locate the difference, then a report to the user. Tolerances are not loosened
after seeing a result. This uses dev only, so no test number influences anything.

`--test` refuses to start unless `eval/MB1-dev-check/gate.json` exists, says
pass, is not a `--limit` run, has batch 64, and records the same pinned
identity, scorer hashes, manifest hashes and numerical settings as the current
process.

## Preconditions (`binary_test.py --precheck`, read-only, no GPU, must print ALL PASS)

Writes `eval/MB1-precheck.json`; `--dev-check` and `--test` re-run the precheck
in-process and refuse unless it passes.

1. Unit success from durable evidence: the user journal of
   `parakeet-binary-binary-main.service` has no "Failed with result" line;
   `logs/binary-main.log` ends with the main phase's `[binary_sweep ...] done`;
   no `parakeet-*` unit other than the caller's own is active.
2. Final-MB1 identity: every pinned file hash matches; the export directory's
   file set is exact; manifest `file`, `format`, `sha256` and
   `files["export.safetensors"]` are as above; summary run_name
   `main-binB-m1-lr2e-4`, seed 20261017, lr 2e-4, select final,
   selected_step = steps = max_steps = 250,000, resumes empty, scored_on "model
   rebuilt from export", `binary.quantizer` binary, `binary.init` m1;
   reconstruction.json has codes_exact, scales_exact, greedy_hyps_equal true and
   equals summary `reconstruction`.
3. Code: every source hash MB1's summary records equals the current file (so the
   loader and quantizer are the ones MB1 was exported and scored with); every
   `SCORER_HASHES` entry matches.
4. Corpus: for each of the 8 test sets, the current manifest's SHA-256 (from
   evaluate.py's `resolve_manifest`) equals the `manifest_sha256` recorded in
   both M1-test's and b0-pretrained's per-set outputs, and its utterance IDs are
   unique.
5. Normalizer compatibility: the current normalizer (item 3's pinned module),
   applied to every saved raw `ref` and `hyp` in M1-test's and b0-pretrained's
   per-set records, reproduces their saved `ref_norm` and `hyp_norm` exactly
   (so the normalizer MB1 is scored with is the one the baselines were scored
   with, in behaviour, not only by name).
6. Reused arms: M1-test and b0-pretrained have all 8 sets, no `--limit`, the same
   decoding and normalizer records; M1-test's source SHA-256 is `a287e977...`;
   b0-pretrained's source kind is "pretrained" with the pinned base-model
   revision and .nemo SHA-256 (`base_model` in M1-test's source record).

## Test-set hygiene and attempts

- MB1 is scored on test exactly once. `--limit` is accepted only with
  `--dev-check` (smoke tests use dev, writing `eval/smoke-MB1-dev-limit<N>`);
  combined with `--test` it is rejected before any model loads.
- The production test output directory is fixed in code
  (`eval/MB1-test`), not a CLI argument.
- **Attempt record** `eval/MB1-test.attempts.jsonl` (append-only): before
  decoding, `--test` appends `{"attempt": n, "state": "started", ...}`; after
  each set, `{"set": name, "state": "set_done", "sha256": file hash}`; at the
  end `{"state": "complete"}`. `--test` refuses if any `complete` entry exists,
  or if a `started` attempt has no `failed` entry (a crash without a recorded
  failure is investigated by hand first, then recorded with
  `--record-failure "<evidence>"`, which renames the partial directory to
  `MB1-test.failed-<time>`).
- A failure after every set is `set_done` (for example while writing the
  summary) does not re-decode: `--summarize` rebuilds `summary.json` from the
  complete per-set files, checking their hashes against the attempt record, and
  then appends `complete` (idempotently: a second `--summarize` appends nothing).
  Once an attempt has all eight `set_done` entries, `--record-failure` and any
  new decoding attempt are refused; only `--summarize` is allowed.
- No decision depends on a test number.

## Report (`binary_report_test.py`, no GPU)

Writes `finetune/parakeet-binary/results/TEST.md` and `results/test.json`:
per-set WER for NVIDIA published / B0 / M1 / MB1, the seven-set mean (LS clean,
LS other, AMI, Earnings-22, GigaSpeech, SPGISpeech, VoxPopuli; TED-LIUM not in
the bundle), Common Voice separately, MB1 - M1 and MB1 - B0 per set (points and
relative), empty-hypothesis rate per set for all three, export sizes (MB1
105.3 MB, M1 180.8 MB, original 2472 MB), the dev-check result, and the label:
MB1 - M1 is the net change from converting M1 to binary and training 250,000
more steps on a new data order, not a causal binary cost (plans/main.md
"Reported").

It refuses if: an MB1 set records a source SHA-256 other than the pinned weight
hash; M1-test's summary or any of its per-set outputs records a source SHA-256
other than M1's pinned export hash `a287e97719c451b785be2cd01ecc861fcaa010ebaa4ff2841783ec78fcd61503`;
the MB1 per-set files' hashes do not all equal the `set_done` hashes of one single
`complete` attempt (mixed attempts or a replaced file), or MB1's summary.json
disagrees with its per-set files; any arm's manifest SHA-256 differs between arms or from the current file;
decoding or normalizer metadata differ from M1-test's; B0 is not the pinned
pretrained model; any output is partial, a set is missing or duplicated, or an
utterance ID is duplicated or missing relative to the manifest; recorded scorer
hashes differ from the pinned ones; the attempt record is not `complete`.

Sanity checks (pause the write-up for investigation, not a re-score): any set
with MB1 WER > 3x B0, or a set with more than 1% empty MB1 hypotheses and more
empty hypotheses than B0 on that set (M1's AMI flag was closed on that basis).

## Commands (from `finetune/parakeet-binary/`, sequential, one capped unit each)

```sh
./python binary_test.py --precheck
./heavy mb1-dev-check --mem-max 24G --runtime 2h --wait -- python binary_test.py --dev-check \
  && ./heavy mb1-test --mem-max 24G --runtime 2h --wait -- python binary_test.py --test \
  && ./python binary_report_test.py
```

The `&&` chain is a convenience; the gate is enforced by `--test` itself.
Expected about 15 minutes per GPU step (M1-test took 14.5 minutes).

## Contract tests (written before the implementation; `tests/test_binary_test.py`)

Using small fixtures (fake export directories, summaries, manifests, attempt
records, eval outputs); no GPU:

- Precheck fails, naming the item, for: each pinned file with a changed hash; an
  extra or missing export file; manifest `file` not `export.safetensors`; manifest
  `sha256` or `files` entry inconsistent; summary run_name / selected_step /
  seed / resumes wrong; reconstruction not exact; a changed recorded source;
  a changed scorer file; a module imported from a different path; a test
  manifest hash differing from M1-test or B0; duplicate utterance IDs; B0 not
  pretrained or wrong revision; an active `parakeet-*` unit; a journal with
  "Failed with result"; a log not ending in `done`.
- Gate: `--test` refuses with no gate.json, a failing gate, a `--limit` gate, a
  batch-32 gate, or a gate recording different identity / scorer hashes /
  settings. The gate fails when the mean is off by more than 0.0002, and when the
  mean is within tolerance but one set is off by more than 0.0010; it passes at
  exactly the tolerance. Points-to-fraction conversion is checked.
- Hygiene: `--test --limit` is rejected before loading; `--test` refuses after a
  `complete` record, and after a `started` attempt with no `failed` record;
  `--record-failure` renames the directory and appends; `--summarize` rebuilds
  the summary from complete set files, refuses a hash mismatch, appends
  `complete` once, and makes zero decoding calls, after which the report
  succeeds; with eight `set_done` entries, `--record-failure` and a new attempt
  are refused.
- Normalizer: a changed normalizer module digest fails the precheck and the
  gate binding; a fixture record whose saved `hyp_norm` differs from the current
  normalizer's output fails precheck item 5.
- Scoring path: `--test` and the production `--dev-check` call evaluate.py's
  `evaluate_set` with batch 64 and 1200 s; the diagnostic `--dev-check
  --dev-batch-size 32` is accepted, writes only `eval/MB1-dev-check-b32`, and its
  gate.json is rejected as a test-launch gate; with `binary_export.load_export`'s model, while holding
  `paths.gpu_lock`, and refuse a decoder mode other than `no_while_loops` before
  or after a set; summary.json and gate.json are written last.
- Report: refuses each condition listed under "Report", including M1 outputs
  substituted by another arm's, M1 summary and per-set source hashes disagreeing,
  an MB1 per-set file replaced after its `set_done` record, files from two
  attempts, and an MB1 summary inconsistent with its per-set files; computes the seven-set
  mean with Common Voice apart and the per-set differences, checked against a
  fixture with known values.

## Code changes

New files only: `binary_test.py`, `binary_test_paths.py`,
`binary_report_test.py`, `tests/test_binary_test.py`. Nothing in
`parakeet-ternary/` or the frozen binary pilot/main files changes.

## Resource caps

24 GB memory per unit, 2 h runtime limit, GPU idle (MB1 has exited). Noisy-dev
scoring waits for the user's listening check and is not part of this plan.

## Plan review round 1 (2026-10-20)

Codex gpt-6-astra xhigh, `/mnt/hd/wilderness-labs-stt/parakeet-binary/review/test-scoring/p1.md`.

1. (major) Test smoke runs break the one-time rule: `--limit` is dev-only;
   `--test --limit` is rejected before loading.
2. (major) No corpus or B0 identity checks: precheck item 4 (manifest hashes equal
   both baselines, unique IDs) and item 5 (B0 pinned pretrained); the report
   repeats them.
3. (major) The verified file need not be the loaded model: all export files,
   config.json and summary.json are pinned by hash; manifest `file` must be
   `export.safetensors`; the file set is exact.
4. (major) The dev gate was not enforced: `--test` requires a passing,
   bound gate.json.
5. (major) New scoring code outside provenance: SCORER_HASHES pinned after code
   review, module paths verified, hashes recorded in and compared across outputs.
6. (minor) Dev comparison: exact recorded value, per-set tolerance 0.10 points
   added, numerical settings follow M1-test and are recorded; batch-32
   reproduction is the investigation path.
7. (minor) Retry rules: append-only attempt record, fixed output path,
   `--record-failure` and `--summarize`.

## Plan review round 2 (2026-10-20)

`review/test-scoring/p2.md`. All round-1 findings resolved; all pins verified on disk.

1. (major) The report did not validate M1's identity: it now requires M1's summary
   and every per-set source hash to equal M1's pinned export hash.
2. (major) Completion was not bound to the reported files: the report checks all
   MB1 per-set hashes against one `complete` attempt's `set_done` records and the
   summary's consistency with them.
3. (minor) The batch-32 diagnostic contradicted the batch-64 contract: the contract
   now covers the production path, with separate tests for the diagnostic.

## Plan review round 3 (2026-10-20)

`review/test-scoring/p3.md`. Round-2 findings resolved; pins verified.

1. (major) The normalizer implementation was outside provenance: the loaded
   Transformers normalizer module and normalizer.json are in SCORER_HASHES, and
   precheck item 5 re-normalizes every saved baseline record and requires exact
   agreement.
2. (minor) Summary recovery had no completion transition: `--summarize` appends
   `complete` idempotently, and once all eight sets are done no failure record or
   new attempt is allowed.

## Implementation notes (2026-10-20)

Interpretations where the plan was silent (implementation in `binary_test.py`,
`binary_report_test.py`, `binary_test_paths.py`; contract tests in
`tests/test_binary_test.py`, run with `unittest`, pytest is not installed).

1. **Normalizer pinned.** The module evaluate.py imports is
   `/mnt/hd/wilderness-labs-stt/stt-distillation/deps/transformers/models/whisper/english_normalizer.py`
   (Transformers 5.17.0), SHA-256
   `e0fc3dc9373209dbecd5881dab6a9834f76cf21d8031914c8f4fa1c3a1288461`;
   `normalizer.json` (`finetune/stt/models/whisper-tiny.en/`) is
   `bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd`, equal to
   evaluate.py's own `NORMALIZER_SHA256`. Both are `SCORER_HASHES` entries; the
   module is also in `SCORER_MODULES`, so its `__file__` is checked.
2. **binary_test_paths.py.** Its hash must appear in this plan's bold Status
   paragraph (precheck item 3 checks it there), and every output records it
   (`scorer.binary_test_paths_sha256`), so the gate binding and the report
   compare it too.
3. **Tolerance edge.** A difference passes when `round(|diff|, 12) <= tolerance`,
   so values exactly at 0.0002 / 0.0010 pass despite float noise (1e-12 is far
   below one word in the dev sets).
4. **Precheck details.** Item 1: every journal message of the unit is checked (the
   journal must be readable and non-empty); the log's last `[binary_sweep` line
   must end in `] done` and a `phase main;` line must exist; the caller's own unit
   is taken from `/proc/self/cgroup`. Item 2 also checks `dev_limit` null,
   `config_sha256` = the pinned config.json hash, `dev_mean_wer` = the pinned
   fraction, `dev_wer` covering the five dev sets; "resumes empty" is
   `summary.resumes == 0` and no non-empty `resumes.jsonl`; reconstruction
   equality is the whole record. Item 6 also requires M1-test's decoding at batch
   64 / 1200 s, the normalizer string equal to evaluate.py's, and M1-test's
   `base_model` equal to the pinned one. The in-process precheck of `--dev-check`
   and `--test` does not rewrite `eval/MB1-precheck.json`; its failing checks are
   in the refusal message.
5. **Dev-check outputs.** Every dev check writes `gate.json` last: `pass` =
   within tolerance and no `--limit`; `test_launch_gate` = no `--limit` and batch
   64. `--test` requires `out_dir` = `eval/MB1-dev-check`, `test_launch_gate`,
   `pass`, then equal identity, scorer, manifests (all 5 dev + 8 test) and
   settings. A production dev-check directory with a `gate.json` is never
   overwritten (moved aside by hand); one without it (unfinished) has its stale
   JSON files removed. A `--limit` smoke run at batch 32 writes
   `eval/smoke-MB1-dev-limit<N>-b32`. Exit status: 1 if a production or batch-32
   gate fails, 0 for smoke runs.
6. **Numerical settings** recorded: `cudnn.deterministic`, `cudnn.benchmark`,
   `CUBLAS_WORKSPACE_CONFIG`, `PYTORCH_CUDA_ALLOC_CONF`, `CUDA_HOME`,
   `NUMBA_FORCE_CUDA_CC`, `CUDA_VISIBLE_DEVICES`, `LD_LIBRARY_PATH`,
   `NVIDIA_TF32_OVERRIDE`, `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`, `OMP_NUM_THREADS`,
   and torch / torch CUDA / cuDNN / NeMo / Transformers / Python versions. They are
   captured once before the model loads and must be unchanged before every set.
7. **Attempt record.** Every entry carries `attempt`; `started` also records the
   pid, unit and gate SHA-256 and is appended after the GPU lock is acquired,
   before the model loads. An exception during `--test` is not recorded as
   `failed` automatically (failures are recorded by hand with
   `--record-failure`). `--test` refusal order: argument errors, attempt record,
   precheck, gate, existing `eval/MB1-test`, CUDA, export weight hash (the source
   record is built and checked before the lock), then lock, `started`, load.
   `--record-failure` refuses empty evidence and renames to
   `MB1-test.failed-<UTC %Y%m%dT%H%M%SZ>` (`renamed_to` null if the directory was
   never created). `--summarize` also requires the current scorer files and each
   per-set record's scorer hashes to equal `SCORER_HASHES`; with a `complete`
   entry present it does nothing and exits 0.
8. **Report.** Relative differences are `100 (MB1 - ref) / ref` percent; empty
   hypotheses are counted over scored utterances; sizes come from the recorded
   export manifests (MB1 and M1-test sources) and MB1's `original_nemo_bytes`.
   It also refuses when the current scorer files differ from `SCORER_HASHES` or
   the gate is not a passing batch-64 test-launch gate. Sanity flags write the
   files with a banner and exit 3, as `parakeet-ternary/report_test.py` does.

## Code review round 1 (2026-10-20)

Codex gpt-6-astra xhigh, `review/test-scoring/c1.md`: NOT CLEAN (2 major, 1 minor).
Each fix's failing tests were written first; no existing test was weakened.

1. (major) Attempt transitions were not synchronized: the ledger was read before
   the blocking GPU lock, recovery commands took no lock, and the GPU lock ended
   before the summary and `complete`. Now `--test`, `--record-failure` and
   `--summarize` run under the attempt lock `eval/MB1-test.attempts.lock`
   (`binary_test_paths.ATTEMPTS_LOCK`, flock) and read the ledger only after
   acquiring it. `--test` waits for it; the recovery commands refuse at once
   while it is held (a holder means an attempt may be running). Lock order:
   attempt lock, then the shared GPU lock, never the reverse. Only binary_test
   takes the attempt lock, and the gpu1bit agents take only the GPU lock, so
   waiting on the GPU lock cannot deadlock. `--test` (and `--dev-check`) hold the
   GPU lock through summary.json, the `complete` record (gate.json for the dev
   check) and `release_model` (synchronize, empty the CUDA cache). Refusal order
   for `--test` is now: arguments, attempt lock, attempt record, precheck, gate,
   and so on (implementation note 7). Tests: a queued second `--test` in a
   thread waits on the lock, then refuses as complete without loading; a ledger
   changed by the previous holder is seen after acquisition; `--record-failure`
   and `--summarize` issued during the eighth set are refused and the attempt
   completes; the GPU lock is held at the summary write, `complete` and the
   release; and the lock order is checked.
2. (major) The report did not bind the dev gate to the completed attempt. It now
   requires `gate.json`'s SHA-256 to equal the complete attempt's
   `started.gate_sha256`, and the gate's identity, numerical settings, test
   manifest hashes and decoder mode to equal the MB1 outputs'. Tests cover a
   replacement gate with unchanged scorer hashes, a started record without the
   gate hash, and a rebound gate differing in each binding.
3. (minor) Summary consistency was partial. The report now compares MB1's
   summary.json with the full projection of its per-set files: per set wer,
   utterances, scored_utterances, excluded_empty_reference, padded_short,
   audio_hours, wall_s and limit; totals (wall_s, padded_short, mean_wer,
   mean_sets); source, scorer, settings, binary_test record and decoder mode.
   Each per-set file must also record `no_while_loops` before and after, the same
   settings as the others, and the production test-run record (batch 64, 1200 s,
   no limit). There is a test for each field.

Re-pinned: `SCORER_HASHES` (`binary_test.py`, `binary_report_test.py`) and the
binary_test_paths.py hash in the Status line. Full suite 203 tests OK; the real
`--precheck` is ALL PASS.

## Code review round 2 (2026-10-20)

`review/test-scoring/c2.md`: NOT CLEAN (2 major). Round-1 findings 2 and 3 are
resolved. Each fix's failing tests were written first.

1. (major) GPU cleanup was ineffective: the callers kept `model` alive through the
   GPU unlock, so `release_model` freed nothing. Now the model and its decoder
   exist only inside `_load_and_score`, called by `decode_and_release`. On an
   exception, that function clears the frames of the exception's traceback chain
   (`traceback.clear_frames`; those frames hold the model) before it propagates.
   On every path it then runs `gc.collect()`, `torch.cuda.synchronize()` and
   `torch.cuda.empty_cache()`, all while the caller still holds the GPU lock.
   The round-1 timing test is replaced: `test_the_gpu_lock_is_held_through_finalization`
   keeps the summary and `complete` checks, and `ModelLifetimeTest` watches the
   fake model through a weak reference only, with stubs that record nothing. The
   model must be dead when the GPU lock is released for `--dev-check` and
   `--test`, on the normal path, an exception during decoding, and a decoder-mode
   refusal.
2. (major) Gate TOCTOU: the gate was validated before the blocking GPU lock but
   hashed after it. Now `require_gate` reads the gate bytes once, validates those
   bytes and returns their SHA-256. `--test` calls it again under the GPU lock and
   records that digest in `started`, and also re-checks `eval/MB1-test` and the
   export weight hash there. `--dev-check` checks for an existing `gate.json` and
   creates and cleans its output directory under the GPU lock. The pre-lock
   checks remain as early refusals only. Tests (`GateRaceTest`):
   - a gate replaced by a failing one during the GPU-lock wait is refused with no
     load, ledger entry or output;
   - a gate removed during the wait is refused;
   - the recorded digest is that of the bytes validated under the lock;
   - a gate changed after it was read leaves the record unchanged, and the
     report refuses;
   - an output directory appearing during the wait is refused;
   - a second dev check whose wait spans a completed first one refuses without
     loading and leaves that gate untouched;
   - a smoke rerun's stale files are removed only under the lock.

Re-pinned: `SCORER_HASHES` (`binary_test.py`) and the binary_test_paths.py hash in
the Status line. Full suite 215 tests OK; the real `--precheck` is ALL PASS (104
checks).

## Code review round 3 (2026-10-20)

`review/test-scoring/c3.md`: NOT CLEAN (1 major). Both round-2 findings are
resolved. The failing tests were written first.

1. (major) The report parsed the gate in `check_scorer` and hashed the path again
   in `check_gate`. A gate swapped in between and then restored would be
   reported under the launch gate's digest. Now every report input is read once,
   through `read_bytes`. `snapshot` and `snapshot_json` return the parsed content
   and the SHA-256 of exactly those bytes, and the checks receive both; no path
   is opened twice. This applies to the gate (read once in `main`, passed to
   `check_scorer` and `check_gate`), and in the same way to the attempt record,
   the per-set outputs (their digests are compared with `set_done`) and the test
   manifests (`read_manifests`: one read gives the hash and the utterance IDs,
   used by `load_arm` and `check_comparable`). Hash-only inputs (the scorer
   sources) are unaffected. Tests (ReportTest, hooking `read_bytes`):
   - a gate changed right after it is read is neither reported nor changes the
     digest checked (the launch gate's dev result is reported);
   - a later gate read and the launch gate restored before any hashing is refused;
   - an MB1 per-set file restored after it is read is refused;
   - a ledger changed after it is read does not change the checks;
   - a manifest restored after it is read is refused;
   - each gate, ledger, per-set and manifest file is read exactly once.

Re-pinned: `SCORER_HASHES` (`binary_report_test.py`) and the binary_test_paths.py
hash in the Status line. Full suite 221 tests OK; the real `--precheck` is ALL
PASS (104 checks).
