# kea-b1 encoder benchmark

Chooses the 1-bit Core ML encoding of kea-b for LocalFlow. The goal is **lowest latency**, with the constraint
that the encoder uses **less memory than the shipped ternary model (kea-t)**. The harness measures latency, load
time, memory and accuracy for any number of LocalFlow-format bundles (`bundle.json`, multifunction
`Encoder.mlmodelc` with b4/b8/b15/b30, native decoder, front end, vocabulary). Each bundle can run on any Core ML
compute units, and an external (non-Core-ML) encoder can be plugged in through a command.

It times LocalFlow's **deployed runtime**: the driver compiles `Sources/Parakeet/*.swift` and
`Sources/Dictionary/*.swift` from a LocalFlow checkout (default `/Users/ajbarry/workspace/github.com/LocalFlow-kea-b`)
together with `kb1bench.swift`. Nothing is copied from LocalFlow, and its checkout is only read.

| File | Role |
| --- | --- |
| `kb1bench.py` | Driver on the Mac (`/usr/bin/python3`, standard library only): plan, phases, macguard, report |
| `kb1lib.py` | Pure logic: statistics, pairing, counterbalancing, WER, encoder error, bundle verification |
| `kb1bench.swift` | Measurement tool (modes `fixtures features load prepare encoder encode service decode`) |
| `tests/test_kb1lib.py` | Contract tests of `kb1lib` (invented data) |

## Run

On the Mac, as a jobs-mac job (`priority: normal`, `full_machine: true`, `quiet_host: true`). Each heavy step is
already wrapped in macguard (`--rss-cap 4G`). Refusals (lock held, free memory below 40%) are retried for up to
`--guard-wait` seconds.

```bash
cd /Users/ajbarry/workspace/github.com/wilderness-labs-stt-binary-coreml/finetune/parakeet-binary-coreml/bench
A=/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios
/usr/bin/python3 kb1bench.py run --out $A/kea-b1/bench/final-$(date +%Y%m%d-%H%M) \
  --bundle kea-t=$A/localflow-b30/bundle --bundle kea-b=$A/kea-b/bundle \
  --bundle B1L=$A/kea-b1/B1L/bundle --bundle B1S=$A/kea-b1/B1S/bundle ... \
  --units ane \
  --baseline kea-t@ane --reference kea-b@ane --ternary kea-t@ane
```

- `--units ane,gpu,cpu,all` runs every bundle on each compute-unit setting (`ane` = cpuAndNeuralEngine,
  `gpu` = cpuAndGPU, `cpu` = cpuOnly, `all` = all). Each setting becomes its own variant (`B1L@gpu`), with its
  own bundle copy, first load, rows and ratios. `--variants kea-t@ane,B1L@ane,B1L@gpu` picks variants explicitly.
- `--external NAME=COMMAND` adds the variant `NAME@ext` (see "External encoders" below).
- `--asr-per-split N` (default 150) sets the LibriSpeech subset size: N utterances each from test-clean and
  test-other, sampled with seed 0. Other size flags: `--rounds` (2), `--encoder-per-bucket` (6),
  `--encoder-warmups/--encoder-reps` (3/10 per round), `--e2e-warmups/--e2e-reps` (2/5 per round),
  `--encdiff-per-bucket` (8), `--e2e-lengths` (2,4,7,14,18,28,33).
- `--phases build,inputs,load,latency,accuracy,report` selects phases. Finished steps are skipped, so rerunning
  the same command resumes an interrupted run. A different plan in the same `--out` is refused.
- `--copies-from RUN_DIR` reuses the bundle copies of an earlier run, which are already specialized. This skips
  the first-load phase (minutes per ANE variant) when only latency or accuracy needs rerunning.
- `kb1bench.py report RUN_DIR` rebuilds `results.json`, `summary.md` and `summary.txt` from the raw records.
  `kb1bench.py verify BUNDLE` prints a bundle's integrity check and sizes.

Outputs go to `RUN_DIR/` in the artifact area. `plan.json` records the configuration, machine, LocalFlow commit,
dirty state of its runtime sources, source and executable SHA-256s, bundle sizes and the round orders. Raw JSON
lines go to `raw/<variant>/`, and `results.json`, `summary.md` and `summary.txt` hold the report. The inputs
(`inputs/`) are synthetic `say` speech (invented text, voice Samantha) cut into 16 kHz 16-bit WAV clips, plus
public LibriSpeech. Bundle copies are APFS clones (`cp -c`) under `copies/`. Source bundles are never written.

Tests (NixOS or Mac, Python 3.9 or later):

```bash
cd finetune/parakeet-binary-coreml/bench && python3 -m unittest discover -s tests
```

## Phases and what each number means

Every measurement runs in its own fresh process, built from the same executable.

**load**: per Core ML variant, in this order:
1. `load-new-path`: the first-ever load of a fresh copy at a new path. Core ML specializes the model for the
   device here (this took about 6 min for the ternary C6s8 four functions on ANE in earlier runs). For each
   function, in `--load-order` (default 15,4,8,30, because the app prepares the 15 s function first), the
   harness records `MLModel(contentsOf:configuration:)` time (`load_ms`), the first prediction on zero features
   (`first_prediction_ms`, the runtime's warm-up) and the cumulative time (`ready_since_start_ms`).
   `all_ready_ms` is the total for all four functions. The run does not delete caches, and other processes may
   have shared work. "New path" therefore means a new path and inode for Core ML's cache. It does not prove the
   model was uncached system-wide.
2. `prepare`: a fresh process timing `LocalParakeetService(startupStrategy: .allBuckets).prepare(directory:)`.
   This is the deployed path: SHA-256 integrity check of every bundle file, front end, decoder, and the four
   functions loaded and warmed in order 4, 8, 15, 30. Its `all_ready_ms` is "all four ready" as the app
   experiences it on a later launch.
3. `load-same-path`: a fresh-process reload with the same per-function breakdown as step 1, now cached.
- `new_path_cache_added_bytes`: bytes added to this tool's own Core ML cache directories during step 1. Those
  are `~/Library/Caches/kb1bench/com.apple.e5rt.e5bundlecache` plus the `kb1bench` directories under
  `getconf DARWIN_USER_CACHE_DIR` and `DARWIN_USER_TEMP_DIR`. Core ML names these directories after the
  executable, so no other process writes to them. Caches that system daemons keep (aned) are not visible.
  These caches are left in place; delete `~/Library/Caches/kb1bench` by hand to reclaim the space.

**memory**: `TASK_VM_INFO` sampled every 5 ms on a background thread, tagged by phase.
- `phys_footprint`: the process's own footprint (what Activity Monitor and jetsam use). For an ANE model it
  **excludes** the Neural Engine buffers.
- `neural_nofootprint` (+ `_compressed`): the kernel's neural ledger of this process, which holds those buffers.
  The earlier final-memory study (`final-memory-20261005`) measured about 0.9-1.2 GB here for C6s8 versus about
  0.27 GB of `phys_footprint`.
- `footprint_plus_neural` = phys_footprint + neural_nofootprint + neural_nofootprint_compressed. **This is the
  number for "less memory than ternary"**: memory attributable to this process, model and runtime included.
  Metal buffers of GPU variants are booked to phys_footprint (unified memory), so the same total compares ANE,
  GPU and CPU variants.
- `all_four_loaded` is the last sample after `prepare` has made all four functions ready and the process has
  idled for 1 s. `transcribe_peak_sampled` is the maximum over the end-to-end transcription phase. Peaks are
  sampled, so transients shorter than 5 ms can be missed.
- `memory_vs_ternary` compares each variant with `--ternary` on loaded footprint_plus_neural, loaded
  phys_footprint, transcription peak and weight.bin size, giving ratio and `less_than_ternary`.
- Not included: memory in other processes (aned, ANECompilerService, MTLCompilerService), device-wide file
  cache, and GPU driver allocations that are not booked to the process. `daemons_kib` records the daemons' RSS
  before and after loading. `system_wired_*` records system-wide wired memory. Both are indicative only, because
  the Mac is shared and other jobs change them.
- Sizes: `encoder_weight_bytes` (Encoder.mlmodelc/weights/weight.bin), `encoder_bytes` (all listed
  Encoder.mlmodelc files), `listed_bytes` (every file in bundle.json, what the app ships and loads) and
  `directory_bytes` (everything in the bundle directory, extras such as `conversion/` included).

**latency** runs in `--rounds` rounds. Round 2k runs the variants rotated by k, and round 2k+1 runs them in
reverse, so with an even number of rounds every variant's mean position in the session is the same. Each round
starts two fresh processes per variant:
- `encoder`: loads the four functions (cached), then times `MLModel.prediction(from:)` alone, warm, on real
  model inputs. The inputs are the deployed front end's features of the chosen chunks: `--encoder-per-bucket` per
  bucket, from LibriSpeech and the synthetic clips. For each clip there are warm-ups, then timed calls back to
  back. This is the raw encoder cost per function. The input arrays are prebuilt, so preprocessing is excluded.
- `service` (e2e): times `LocalParakeetService.transcribe(fileURL:directory:)` on synthetic clips of each length
  (3 different cuts per length). Each timing covers file decode, silence compaction, 30 s chunking with 2 s
  overlap, front end, encoder, native decoder and transcript assembly. Passes are interleaved: every pass
  transcribes every clip once. The buckets each clip used are recorded (e.g. 33 s uses [30, 8]). The app now
  transcribes while recording, so after release a user waits only for the final partial chunk. The batch call
  is therefore an upper bound on felt latency for long dictations, and equal to it for clips under 30 s.
- Estimates: `typical_ms` = median over clips of each clip's median call. `p95_ms` = p95 (linear interpolation)
  of the pooled calls. `vs_baseline` = paired ratio against `--baseline` on the same clips. Its
  `typical_ratio` = median over clips of (variant clip median / baseline clip median), with a 95% percentile
  bootstrap interval that resamples clips, keeping each clip's variant and baseline calls together. `p95_ratio`
  works the same way. Calls from all rounds are pooled per clip.
- `thermal`: `ProcessInfo.thermalState` at the start and end of each bucket block (encoder) and each pass (e2e).

**accuracy**
- Transcripts from the deployed `service` path (`transcribe(fileURL:directory:)`, .allBuckets, no dictionary)
  of the LibriSpeech subset, for every variant.
- WER against the LibriSpeech transcripts, per split and pooled. It uses the normalizer and alignment of
  LocalFlow's 2026-10-15 LibriSpeech comparison (`apple-speech-2026-10-15`): lowercase, no punctuation,
  numbers, currency and titles spelled out, and British-to-American spelling via Whisper's `english.json`
  (pinned SHA-256). `plain_wer` uses case and punctuation normalization only. The interval is a bootstrap over
  utterances.
- `vs_reference` compares each variant with `--reference` (the kea-b C6s8 bundle, i.e. the same effective FP16
  weights as every exact binary candidate). It reports exact transcript match rate, match rate after plain
  normalization, the list of differing utterances, the candidate's word error against the reference model's
  words, and the paired WER difference (candidate minus reference, bootstrap over utterances). The synthetic e2e
  clips get the same exact-match comparison (`e2e_fixture_agreement`).
- `encoder_error_vs_reference`: on `--encdiff-per-bucket` LibriSpeech chunks per bucket, the valid encoder
  frames ([T, 1024], float32) of the variant vs the reference on identical features. It reports pooled relative
  L2 (sqrt of the sum of squared errors over the sum of squared reference values), median/p95/max per-chunk
  relative L2, max absolute error and min cosine.
- `decode_path_self_check`: the reference's encoder outputs, dumped and decoded by the harness's `decode` mode
  (the runtime's chunking, native projection, greedy TDT and assembly). This must match the service transcripts
  exactly (100%). It validates the external-encoder seam below.

## External encoders (MLX / Metal / CPU prototypes)

`--external NAME=COMMAND` adds the variant `NAME@ext`. COMMAND is split like a shell command, and these
placeholders are substituted: `{mode}` (`time` or `encode`), `{features}`, `{keys}`, `{out}`, `{warmups}`,
`{reps}`. Example: `--external mlx1="/path/runner --mode {mode} --features {features} --keys {keys} --out {out}
--warmups {warmups} --reps {reps}"`. The command runs under macguard.

Inputs: `{features}/manifest.jsonl` has one row per chunk: `key`, `id`, `chunk`, `bucket`, `frames`, `valid`,
`silent`. `{features}/<key>.mel` holds float32 little-endian feature-major `[128, frames]` values. The model
input is that array zero-padded to `[1, 128, 100 * bucket + 1]` with `mel_length = valid`, exactly as LocalFlow
feeds Core ML. `{keys}` is a text file of chunk keys, one per line.

- `mode=time`: for each key, run `{warmups}` untimed and `{reps}` timed encoder calls. Write
  `{out}/timings.jsonl` rows `{"bucket": b, "clip": key, "rep": r, "warmup": bool, "ms": x}`. Optionally write
  `{out}/memory.json` (any JSON, reported verbatim). It is run once per round in the counterbalanced order.
- `mode=encode`: for each key, write `{out}/<key>.f32`, the valid encoder frames as float32 time-major
  `[frames, 1024]`, and `{out}/manifest.jsonl` rows `{"key", "frames"}`.

The harness decodes those outputs with the reference bundle's native decoder through LocalFlow's chunking and
assembly. This gives WER, agreement and encoder error on the same footing as Core ML variants. External
variants get encoder latency and accuracy, but no e2e or load rows. Their memory is whatever they report.

## Limits

- The Mac is a shared 16 GB M1 Pro. Timings are only comparable within one run, which is why they are paired
  and counterbalanced. A quiet-host job reduces but does not remove interference. Check `thermal` before
  trusting small ratios.
- First-load times depend on what else the ANE compiler is doing, and on whether Core ML found shared work.
- Memory figures cover the measuring process (including its neural ledger), not daemons or the device-wide
  file cache.
- The LibriSpeech subset is a few hundred utterances. Its WER intervals are wide compared with differences
  between exact binary encodings. Exact-match rate and encoder error against kea-b C6s8 are the sharper
  checks there.

## Runtime estimate

Measured in `validate-20261021` (defaults, 2 ANE variants):
- First load at a new path: about 413 s per ANE variant (about 100 s per function, for both C6s8 bundles).
- Cached prepare and reload: about 10 s.
- Both latency rounds: about 1.5 min per variant.
- Accuracy on 300 utterances: about 25 s per variant, plus about 20 s once for the reference.

Eight ANE bundles therefore take about 8 x (7 + 2) = about 75 min, of which about 55 min is first loads. To
stay under an hour, take the first loads early in their own run (`--phases build,inputs,load`, as candidates
land, which needs every bundle up front), then run the comparison with `--copies-from` that run (about 20 min
for 8). Alternatively, accept the 75 min. A failing variant (load timeout, RSS cap) is recorded in
`plan.json` `failures` and in the summary, then skipped. The other variants still run. `--retry-failed`
retries failed variants on resume.

**Core ML GPU (`cpuAndGPU`) for these bundles is impractical under the 4 GB guard.**
- kea-b C6s8 on GPU took 1149 s to load at a new path (about 285 s per function).
- It wrote 4.77 GB to the tool's Core ML cache.
- It held 3.3 GB `phys_footprint` once loaded (sampled peak 4.9 GB).
- Its `prepare` was killed by macguard (`rss 4399648KB > cap 4194304KB`).

Measuring Core ML GPU rows needs a higher `--rss-cap`, which needs the user's approval. CPU variants will be
much slower per call; lower `--encoder-reps` for them.
