# Parakeet-TDT 0.6B v2 on iPhone: inference benchmark

Design: [DESIGN.md](DESIGN.md). Nothing here is a model or audio; generated weights, golden
outputs and logs live outside Git (NixOS `/mnt/hd/wilderness-labs-stt/parakeet-ios/`, Mac
`/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios/`).

## S0 tooling: reference, surrogates, golden outputs, Mac guard

| File | What |
| --- | --- |
| `reference.py` | Pure-PyTorch FP32 reference of the v2 architecture (front end, FastConformer encoder, TDT prediction and joint networks, NeMo-equivalent greedy TDT with a complete step trace, forced replay returning logits and LSTM states). NeMo state_dict names; ternary modules load from int8 codes + FP32 row scales. |
| `weight_stats.py` → `weight_stats.json` | Statistics of the selected pilot export (P2, lr 5e-4): per-module code histograms, 257 row-scale quantiles, 129 value quantiles per floating tensor, the pinned model config, provenance. No weights. |
| `randomweights.py` | Seeded surrogate models from `weight_stats.json` (LayerNorm biases 0), bit-identical on Linux and macOS (per-tensor SHA-256 manifest), streamed to safetensors. |
| `models.py` | The benchmark models by name as reference models: `b0` (NVIDIA's .nemo), `mp2` (M_P2, the pilot P2 export, the primary model), `seed0`-`seed2` (surrogates). |
| `golden.py` | NeMo FP32 golden outputs of a full-depth benchmark model (`--model seed0` or `mp2`, NixOS) and the gate-1 comparison `compare()`. |
| `macguard` (sh front end) + `macguard.py` | Wrapper for every Mac job (Python, the Mac's `/usr/bin/python3`, stdlib only): start conditions, flock lock held through cleanup, a sentinel and a gated launch handshake (the job registers itself before it can run), own session and process group, RSS cap, timeout, memory and swap aborts, sentinel-loss abort, fail-closed probes and finalization, TERM→KILL until the group is verifiably empty, hand-over to a lock-holding watcher otherwise, logging. The job runs in the caller's exact environment (the front end undoes what the python3 shim injects, e.g. SDKROOT). Exit: the job's status; 124 if the guard aborted it; 130 interrupted; 3 refused; 2 usage; 125 machinery failure (never 0). **Accepted residual risk:** if both the supervisor and the sentinel are SIGKILLed and the job closes inherited descriptors, containment is lost. `tests/macguard_tests.sh` asserts every behaviour (exit status = number of failed checks), including fault injection via `MACGUARD_TEST_FAULT`. |
| `artifacts.py` | Allowed artifact locations; every writer refuses paths off `/mnt/hd` (Linux), outside the artifacts directory (Mac) or inside the repository. |

Commands (NixOS from `finetune/parakeet-ternary/`, one memory-capped unit at a time; `R` is the
repository root; every unit runs CPU only):

```sh
# weight statistics (no NeMo, < 1 GB, may run directly)
CUDA_VISIBLE_DEVICES= ./python ios/weight_stats.py
# a surrogate (no NeMo, streams one tensor at a time, < 0.5 GB)
CUDA_VISIBLE_DEVICES= ./python ios/randomweights.py --seed 0 --out /mnt/hd/wilderness-labs-stt/parakeet-ios/random/seed0
# reference vs NeMo: real B0 weights at full depth, and a 2-layer surrogate (gate 1)
./heavy ios-wp1-nemo --mem-max 12G --runtime 40min --wait -- env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 $R/finetune/parakeet-ternary/python $R/finetune/parakeet-ternary/ios/tests/test_reference_nemo.py -v
# NeMo golden outputs of B0, M_P2 and surrogate seed 0 (one unit each), then the reference against them (gate 1)
./heavy ios-wp1-golden --mem-max 12G --runtime 40min --wait -- env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 $R/finetune/parakeet-ternary/python $R/finetune/parakeet-ternary/ios/golden.py --model seed0  # or mp2, b0
./heavy ios-wp1-golden-check --mem-max 10G --runtime 40min --wait -- env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 $R/finetune/parakeet-ternary/python $R/finetune/parakeet-ternary/ios/tests/test_reference_golden.py -v
# surrogate determinism, fidelity and a full-depth forward pass
./heavy ios-wp1-random --mem-max 8G --runtime 30min --wait -- env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 $R/finetune/parakeet-ternary/python $R/finetune/parakeet-ternary/ios/tests/test_randomweights.py -v
```

On the Mac (repository `finetune/parakeet-ternary/`), every job goes through `macguard`, which
logs to `<artifacts>/logs/macguard.log`:

```sh
ios/macguard --rss-cap 6G --timeout 1800 -- ios/pyenv/.venv/bin/python ios/tests/test_randomweights.py -v
```

Both machines write `<artifacts>/random/manifests/seed{0,1,2}.json`; `tests/compare_manifests.py`
compares them. Verified: seeds 0-2, all 989 tensors identical between Linux x86-64 (NumPy 2.4) and
macOS 27 arm64 (NumPy 2.2.6), recorded in `results/wp1_manifest_crosscheck.json`. Other platforms
or NumPy versions are not covered by that check (NumPy's Generator stream is only guaranteed per
version); re-run the comparison after changing either.

## WP2: clips, replay traces, C0 pipeline, harness

| File | What |
| --- | --- |
| `clips.py` → `clips.json` | 82 clips from LibriSpeech dev-clean, read from `dev-clean.tar.gz` (MD5 `42e2234b…` = OpenSLR's published value; SHA-256 recorded). 16 natural utterances per 2/4/8/15 s bucket (distinct speakers first), 16 boundary crops (bucket edges `N_e`, `N_e + 1`, and `N = 160 M` with `M = 8k - 1, 8k, 8k + 1`), a silence and an impulse clip. Per-clip SHA-256 of the float32 PCM. The frame formulas are in the docstring. |
| `traces.py` → `traces.json` | B0 (`models.b0`, the FP32 reference) decodes every clip. Records the complete greedy TDT trace: frame, prediction input, token, duration, emitted, pred_updated, symbols_at_frame, forced_advance, advance. Also the transcript and WER. 114 KB. |
| `c0.py` → `c0.json` | C0 pin: `FluidInference/parakeet-tdt-0.6b-v2-coreml` at `ee09c56`. Only the files FluidAudio 0.7.8 loads for v2: `Preprocessor`/`Encoder`/`Decoder`/`JointDecision` `.mlmodelc`, `parakeet_vocab.json`, `config.json`. SHA-256 and I/O schema of every file. |
| `references.md` | Pinned permalinks and verbatim quotes behind DESIGN.md "Prior evidence", plus the C0 artifact table. |
| `bench/` | `ParakeetBench` Swift package (tools 6.0; macOS 15+ / iOS 26+), with `BenchCore` and the `parakeet-bench` CLI. C0 pipeline: Preprocessor → Encoder → per-step Decoder/JointDecision calls replicating FluidAudio 0.7.8's `TdtDecoderV3` (cited in `C0Pipeline.swift`). Also free and replay modes, the warm-up/timed protocol, `mach_absolute_time` stage times, `os_signpost` intervals, and `MLComputePlan` per-op dumps. Timed calls keep the same minimal bookkeeping in both modes: tokens, timestamps and call counters. `--diag-dir` adds one separate **untimed** diagnostic call per clip; see "Replay diagnostics and C0's limits" below. Replay validates every trace against `clips.json` before loading models: hashes, lengths and all of the reference loop's invariants. Unknown modes or options, and output paths outside the Mac artifact root or inside Git, are refused. |
| `c0report.py` (now `armreport.py`, WP4) | Scores `parakeet-bench` JSON lines against `clips.json` and the B0 traces, with the parent experiment's Whisper-normalized WER. It enforces repetition completeness per clip and reports, per bucket and stage, typical latency (median over clips of each clip's median) and the Harrell–Davis p95 over pooled calls. A run is labelled `baseline-eligible` only with at least 10 timed calls per clip (`--min-timed`); fewer requires `--smoke` and is labelled smoke. |
| `g0probe.py` | G0 feasibility: parses C0's `Encoder.mlmodelc` (MIL text and blob file), cross-checks with coremltools, and fingerprints the tensors against B0. |
| `results/smoke/c0_free_natural_smoke.summary.json`, `results/g0_probe.summary.json` | WP2 results. They hold no weights or audio. The C0 run is a **smoke** measurement: 1 timed call per clip. |

Every artifact writer refuses paths outside the machine's artifact area or inside the repository: `artifacts.check()` in `clips.py materialize`, `c0.py download`, `g0probe.py extract`, and `traces.py`/`c0report.py` when `--out` is not their committed text output; `ArtifactPath.check` in the Swift CLI. Raw `*.f32` files are git-ignored.

Commands (`A` is the machine's artifact directory, `R` the repository root):

```sh
# NixOS
CUDA_VISIBLE_DEVICES= ./python ios/clips.py select              # writes ios/clips.json
CUDA_VISIBLE_DEVICES= ./python ios/clips.py materialize --out $A/clips
./heavy ios-wp2-traces --mem-max 10G --runtime 40min --wait -- env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 $R/finetune/parakeet-ternary/python $R/finetune/parakeet-ternary/ios/traces.py
CUDA_VISIBLE_DEVICES= ./python ios/c0.py pin                    # writes ios/c0.json
# Mac (repository finetune/parakeet-ternary/)
ios/macguard --rss-cap 1G --timeout 3600 -- sh -c 'curl -fsSL -o $A/data/dev-clean.tar.gz https://www.openslr.org/resources/12/dev-clean.tar.gz \
    && ios/pyenv/.venv/bin/python ios/c0.py download --out $A/c0'
ios/macguard --rss-cap 1G --timeout 600 -- ios/pyenv/.venv/bin/python ios/clips.py materialize --out $A/clips
(cd ios/bench && ../macguard --rss-cap 4G --timeout 900 -- swift build -c release)  # job sees the caller's env
B=ios/bench/.build/release/parakeet-bench
ios/macguard --rss-cap 4G --timeout 1800 -- $B run --mode free --models $A/c0 --clips ios/clips.json --pcm $A/clips \
    --kinds natural --warmups 3 --timed 1 --out $A/results/c0-free-natural.jsonl   # smoke; the protocol default is --timed 10
ios/macguard --rss-cap 4G --timeout 900 -- $B run --mode replay --traces ios/traces.json --models $A/c0 \
    --clips ios/clips.json --pcm $A/clips --ids ID1,ID2 --warmups 1 --timed 1 \
    --out $A/results/c0-replay-diag-smoke.jsonl --diag-dir $A/results/diag   # untimed diagnostic pass per clip
ios/macguard --rss-cap 4G --timeout 900 -- $B plan --models $A/c0 --out $A/results/computeplan-c0
ios/macguard --rss-cap 4G --timeout 900 -- ios/pyenv/.venv/bin/python ios/g0probe.py extract \
    --model $A/c0/Encoder.mlmodelc --out $A/results/g0probe.json
# NixOS, on copies of the Mac outputs
CUDA_VISIBLE_DEVICES= ./python ios/c0report.py c0-free-natural.jsonl --smoke --plan summary.cpuAndNeuralEngine.json \
    --out ios/results/smoke/c0_free_natural_smoke.summary.json
./heavy ios-wp2-g0 --mem-max 8G --runtime 10min --wait -- env CUDA_VISIBLE_DEVICES= \
    $R/finetune/parakeet-ternary/python $R/finetune/parakeet-ternary/ios/g0probe.py compare --probe g0probe.json
```

Results (2026-10-02; Mac = M1 Pro, macOS 27.0, Swift 6.3.1; Mac timings are informational because the Mac is shared):

- **Clips.** The PCM of all 82 clips is bit-identical on NixOS and the Mac. Both machines report `all_sha256_match`, and the Mac's own OpenSLR download has the same MD5.
- **B0 traces.** 3,025 steps and 2,624 tokens. The max-symbols rule fires once (`b04-N32001`, a crop that ends mid-word and loops). Replaying the trace on the same model reproduces every decision. Whisper-normalized WER on the 64 natural clips is **1.77%** (17 errors / 959 words).
- **C0 free decoding: smoke run**, 64 natural clips, 3 warm-up and **1 timed call per clip**, `cpuAndNeuralEngine` (preprocessor `cpuOnly`). This is a functional check, not the comparison baseline. The baseline (10 timed calls per clip, typical and Harrell–Davis p95 latency) runs in S1, alongside every arm.
  - WER **1.67%** vs LibriSpeech (16 / 959) and **0.21%** vs B0's transcripts. 53 of 64 token sequences equal B0's.
  - Typical total latency per bucket, 2/4/8/15 s: 62.1 / 67.6 / 80.1 / 99.4 ms. These come from one timed call per clip; Harrell–Davis p95 over 16 calls per bucket is in the summary.
  - Encoder 41.3-42.7 ms in every bucket (fixed 15 s window).
  - Preprocessor 13.8-16.5 ms.
  - Decoder 0.51-0.54 ms per call, joint 0.136 ms per call.
  - Physical calls: 1 preprocessor + 1 encoder per utterance. Decoder 1,951 calls and joint 2,147 calls, against B0's 1,955 prediction-net runs and 2,159 logical steps.
- **Load.** The first Encoder load took 29.4 s and later loads 109 ms. This is only `MLModel(contentsOf:)` wall time. It is consistent with device compilation followed by cache hits, but is **not established**: that needs the Instruments Core ML trace's "prepare and cache" vs "cached" events (DESIGN.md "Load"), which S1 records. `phys_footprint` stays at 28-62 MB, because the model memory is not attributed to the process. The macguard group RSS peak was 0.5 GB.
- **Compute plan** (`cpuAndNeuralEngine`):
  - Encoder: 1,379 of 1,385 placed ops prefer the ANE (99.95% of estimated cost). The 6 CPU ops are 4 `cast`, 1 `expand_dims` and 1 `less`, i.e. input and length handling.
  - Decoder (24 ops) and JointDecision (21 ops) prefer the CPU entirely, as does the preprocessor.
  - The plan is not proof of placement (DESIGN.md gate 6).
- **Replay, diagnostic smoke** (`results/smoke/c0_replay_diag_smoke.summary.json`; raw records and arrays in the artifact directories). Run on 2 clips, `n15-2412-153947-0005` and `b04-N32001`, with 3 warm-up calls, 1 timed call and the untimed diagnostic pass:
  - All 82 traces validated before the run.
  - Physical calls equal the trace's logical work: 129 joint calls = 129 steps, and 127 decoder calls = 127 prediction-net runs.
  - Frame difference: C0's encoder_length is B0's frames + 1 for both clips, and the effective frames differ by 0 or 1, as allowed.
  - Agreement of C0's own argmax with B0's decisions:
    - natural 15 s clip: 98/99 tokens and 90/99 durations;
    - `b04-N32001`, the crop that hits the max-symbols loop: 23/30 tokens and 23/30 durations.
  - Diagnostic arrays: SHA-256 verified on NixOS, all finite, one distinct h state per decoder call.
  - An early 3-clip replay smoke run was not kept as raw evidence; this run supersedes it.

C0 deviates from NeMo/B0 in three places. These are properties of the published pipeline, reproduced here, not harness bugs:
1. **One extra valid mel frame.** C0's preprocessor reports `mel_length = N // 160 + 1`, where NeMo has `N // 160`. With FluidAudio's `ceil(N / 1280)`, 5 of the 64 natural clips decode one more encoder frame than B0.
2. **Final token dropped at the end.** FluidAudio emits a token only if `t + duration` is still inside the utterance. This drops a final token whose duration reaches the end: 2 of 3 clips lose final punctuation this way, and the third is a decision difference.
3. **Different joint model.** JointDecision is FP16 with an in-model argmax and per-step encoder projection, so durations differ more often than tokens.

### Replay diagnostics and C0's limits

DESIGN.md's replay returns logits and LSTM states. For C0 (the product baseline, not a gated arm), **raw logits cannot be obtained from the published models**. FluidAudio 0.7.8 uses `JointDecision.mlmodelc`, which computes the 1,030 logits internally and outputs only three values (see its `model.mil`):
- `token_id`: argmax over the 1,025 token and blank logits;
- `token_prob`: softmax probability of that token over the same 1,025;
- `duration`: argmax bin of the 5 duration logits.

So C0 cannot go through gate 4's logit and margin checks. The untimed diagnostic pass (`--diag-dir`) exposes what is obtainable:
- per joint step: frame, token id, token probability, duration bin, and which decoder call fed it;
- per decoder call: input token, the `decoder` output (640), and copies of `h_out`/`c_out` [2, 1, 640] taken right after the call;
- the encoder output [encoder_length, 1024].

Arrays go to `<clip>.<mode>.diag.f32`, whose layout and SHA-256 are in the `diagnostic` JSON record. Replay diagnostics also record C0's argmax agreement with the trace's decisions.

Allowed frame difference: C0's effective frames, `min(encoder_length, ceil(N / 1280))`, may exceed the trace's `num_frames` by 0 or 1 (deviation 1 below). Replay follows the trace's frames; any other difference is an error.

G0 feasibility: **practical.**
- `model.mil` holds 294 `constexpr_lut_to_dense` ops. Each has `indices` = packed 6-bit uint8 blob and `lut` = fp16[64], one LUT per tensor, plus 320 dense fp16 consts. That accounts for all 908 blobs in `weight.bin`.
- The blob format is a 64-byte header (count 908, version 2), then per blob 64 bytes of metadata (sentinel `0xDEADBEEF`, dtype, size, data offset) and 64-aligned data.
- An independent parser agrees exactly with coremltools 9.0's `_BlobStorageReader` on all 294 tensors and with `constexpr_lut_to_dense.decompress` on the 30 tensors checked in full.
- All 294 tensors map onto B0 weights. Median fingerprint relative error is 2.8% (max 7.8%), and RMS ratios are 0.9986-1.0005. These are 6-bit k-means of B0.
- Two catches for G0's graph:
  - the 24 depthwise convolutions have BatchNorm folded in (with separate fp16 bias consts);
  - `linear_pos` is not stored. Each layer has a folded position table `[1, 8, 128, 375]` for the 188-frame window. Shorter buckets need the middle `2T - 1` columns, a slice that is exact because the projection has no bias.

## WP4: generic arms, front end A, decode loops F0/F1/F2, paired reporting

| File | What |
| --- | --- |
| `bench/Sources/BenchCore/Arm.swift` | `ArmPipeline`: an arm is {front end, encoder package and length variant, decode loop, compute units}. The bucket is the smallest of 2/4/8/15 s holding the clip. Stages: preprocess, encoder (output materialized), preprojection, decode, total, plus time and physical calls per decode component. Timed calls keep the same minimal bookkeeping. Gate mode reads the encoder output from `<id>.f32` files instead. |
| `bench/Sources/BenchCore/Encoder.swift` | Encoder variants: `fixed15` (single function), `multifunction` (functions `b2`..`b15` via `MLModelConfiguration.functionName`) and `enumerated` (one model, input `[1, 128, F_b]`). WP3's contract: `mel` / `mel_length` → `encoder` / `encoder_length`, with F_b = 201/401/801/1501. A `.mlpackage` is compiled once into `<artifacts>/compiled/`, and its compile time is recorded. |
| `bench/Sources/BenchCore/FrontEnd.swift` | Front end A (Accelerate) implements the full NeMo feature contract with the model's stored window and filterbank. Features are computed on the valid audio, normalized over the valid frames, and zero-padded to F_b, with `mel_length` = N // 160. The `c0pre` option uses C0's Core ML Preprocessor for comparison (its `mel_length` is one higher). |
| `bench/Sources/BenchCore/Decode.swift` | `LabelLoop` follows NeMo's greedy_batch semantics, as `reference.run_steps` does, for free decoding and replay. Engines: <br>• F0: per-step `Decoder` + `JointDecision` Core ML calls with C0's contract. <br>• F1: fused `DecoderJoint`, with the pending token re-run from the state before it on each step. <br>• F2: native CPU loop in FP32. The encoder-side joint projection of all frames is one `cblas_sgemm`; the layer-0 input table is precomputed; the 2-layer LSTM uses `cblas_sgemv`; the prediction net runs only after a non-blank emission; one joint `sgemv` and the two-head argmax per step. <br>• `f1native`: a test double with F1's call structure on F2's math. <br>Untimed diagnostics: F2 exposes raw logits [1030] and h/c [2, 640] per step; F0/F1 expose their argmax outputs and states. |
| `bench/Tests/BenchCoreTests` | XCTest checks: vDSP DFT vs a naive DFT; native LSTM and joint vs naive doubles; F2 vs the fused call structure (same decisions, expected physical calls); free-decode traces passing `TraceFile.validate`; replay of a decode's own trace; mutated traces rejected. |
| `native.py` | Exports, under the artifact area: front-end constants, decoder/joint weights (with an exact FP16 transfer copy when possible), and the reference's encoder outputs, replay logits/states and greedy tokens (`reference`, heavy unit). Also runs `gate-frontend` and `gate-f2`. |
| `macpush.py` | Copies artifacts NixOS → Mac through the SSH helper as hash-checked chunks of 90 KB; each call carries at most one 128 KiB argument. |
| `armreport.py` | Arm-agnostic summary (renamed from `c0report.py`): completeness, typical latency and Harrell–Davis p95 per stage, physical calls, and token agreement with B0 or with the model's reference (`--ref`). `--baseline C0.jsonl` adds paired comparisons: per-clip median ratios, HD-p95 ratios, and a percentile bootstrap that resamples clips with each clip's pairs kept together. |
| `results/wp4/` | Gate and smoke summaries, and the macguard suite outputs. Numbers and LibriSpeech texts only. |

Commands:

```sh
# NixOS (heavy unit for the reference), then copy to the Mac
CUDA_VISIBLE_DEVICES= ./python ios/native.py frontend --model mp2 --out $A/native/mp2
CUDA_VISIBLE_DEVICES= ./python ios/native.py weights --model mp2 --out $A/native/mp2
./heavy ios-wp4-reference --mem-max 10G --runtime 40min --wait -- env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 \
    $R/finetune/parakeet-ternary/python $R/finetune/parakeet-ternary/ios/native.py reference --model mp2 --out $A/native/mp2
CUDA_VISIBLE_DEVICES= ./python ios/macpush.py $A/native/mp2/{frontend.json,frontend.f32bin,decoder_joint.json,decoder_joint.f16bin} \
    --dest $MAC_A/native/mp2
CUDA_VISIBLE_DEVICES= ./python ios/macpush.py $A/native/mp2/enc --dest $MAC_A/native/mp2/enc
# Mac
ios/macguard --rss-cap 1G --timeout 300 -- ios/pyenv/.venv/bin/python ios/native.py widen --dir $A/native/mp2
(cd ios/bench && ../macguard --rss-cap 4G --timeout 1200 -- swift test)
ios/macguard --rss-cap 2G --timeout 600 -- $B features --frontend-constants $A/native/mp2 --clips ios/clips.json --pcm $A/clips \
    --out $A/results/frontend-a
ios/macguard --rss-cap 3G --timeout 900 -- ios/pyenv/.venv/bin/python ios/native.py gate-frontend --frontend $A/native/mp2 \
    --swift $A/results/frontend-a --pcm $A/clips
F2="--arm custom --decode f2 --native-weights $A/native/mp2 --encoder-input $A/native/mp2/enc --vocab $A/c0/parakeet_vocab.json"
ios/macguard --rss-cap 2G --timeout 900 -- $B run --clips ios/clips.json --pcm $A/clips --kinds natural $F2 --mode replay \
    --traces ios/traces.json --warmups 1 --timed 1 --out $A/results/f2-gate/replay.jsonl --diag-dir $A/results/f2-gate/diag
ios/macguard --rss-cap 2G --timeout 900 -- $B run --clips ios/clips.json --pcm $A/clips --kinds natural $F2 --mode free \
    --warmups 1 --timed 1 --out $A/results/f2-gate/free.jsonl
# an end-to-end arm: front end A + C0's encoder + F2 with B0's decoder/joint (native.py weights --model b0)
ios/macguard --rss-cap 4G --timeout 900 -- $B run --clips ios/clips.json --pcm $A/clips --ids ... --arm custom --frontend vdsp \
    --frontend-constants $A/native/b0 --encoder $A/c0/Encoder.mlmodelc --encoder-variant fixed15 --decode f2 \
    --native-weights $A/native/b0 --models $A/c0 --out $A/results/wp4-smoke/vdsp-c0enc-f2.jsonl
# NixOS
CUDA_VISIBLE_DEVICES= ./python ios/native.py gate-f2 --ref $A/native/mp2 --results replay+free.jsonl --diag diag
CUDA_VISIBLE_DEVICES= ./python ios/armreport.py ARM.jsonl --baseline C0.jsonl --smoke
```

Results (2026-10-02; Mac = M1 Pro, macOS 27.0, Swift 6.3.1, Mac shared; all latencies are smoke and informational):

- **macguard** (sh front end + `macguard.py`) passes all of `tests/macguard_tests.sh`: 22 checks on Linux, 23 on the Mac (11b is Mac-only). That includes the job environment equalling the caller's, and the r3 fault cases (sentinel SIGKILL, cleanup PermissionError, supervisor death before the ACK, end-log failure). Under macguard, `swift build` no longer needs an SDKROOT pin.
- **Unit tests:** 4/4 pass. The loop test hit 134 emissions and 8 forced advances.
- **Front end A gate** vs `reference.Featurizer` (same stored constants, which are identical for B0 and M_P2) on all 82 clips (`results/wp4/frontend_a_gate.json`):
  - **rel ≤ 1e-5 holds on all 81 non-silence clips** (max 7.5e-6, median 1.2e-6).
  - **abs ≤ 1e-4 fails on 23 clips** (max 4.3e-4). An exact FP64 implementation of the reference algorithm is itself up to 3.2e-4 abs from the FP32 reference on 19 clips, worst in mel band 0, where pre-emphasis leaves little energy. FP32 front end A is a comparable distance from FP64 (4.8e-4 abs, 6.0e-6 rel). The abs ceiling is therefore below the FP32 reference's own rounding.
  - **The silence clip fails by construction:** every feature is constant, and the std guard makes the reference amplify the rounding of its mean by 1e5. Front end A returns the exact answer, 0.
  - These two outcomes are reported, not resolved; the gate definition is the design's to change.
  - Front end A takes 0.4 ms (2 s) to 1.8 ms (15 s) per call, against 13.7–16.1 ms for C0's Core ML Preprocessor.
- **F2 gate** for `mp2` on all 64 natural clips: the same FP32 reference encoder output is fed in, the B0 trace is replayed, and the result is compared with the reference's replay (`results/wp4/f2_gate_mp2.json`). **64/64 pass:**
  - logits rel ≤ 1.4e-7, abs ≤ 8.9e-7 (duration logits 3.8e-7 / 3.0e-6);
  - h rel 1.0e-6 / abs 2.9e-5, c rel 8.2e-7 / abs 3.4e-5;
  - every token and duration argmax equals the reference's;
  - free decoding: all 64 token sequences equal the reference's greedy tokens.

  M_P2's FP32 WER on these 64 clips is 2.71% (B0: 1.77%).
- **End-to-end smoke** (8 natural clips, 2 per bucket; 3 warm-ups + 3 timed calls; paired against C0 on the same clips, ratios with bootstrap 95% CI): arms that differ from C0 in front end and decode loop, all on C0's encoder.

  | Arm | Total typical vs C0 | Notes |
  | --- | --- | --- |
  | vDSP + C0 encoder + F0 | 0.80× (0.79× / 0.81× / 0.82× / 0.77× for 2/4/8/15 s) | encoder 1.00×, decode 0.96× |
  | vDSP + C0 encoder + F2 (B0 decoder/joint weights) | 0.77× | decode 0.80×, encoder 1.00× |
  | c0pre + C0 encoder + F0 | 1.00× | |

  - F2's FP32 prediction step costs 0.57 ms per call; Core ML's Decoder costs 0.51 ms. Its joint costs 0.069 ms per step, and the pre-projection 0.35 ms per 15 s utterance.
  - Transcripts of all three arms: WER 0 on these 8 clips, as C0's.
- **Replay smoke** of the custom F0 and F2 arms (2 clips, with diagnostics): physical calls equal the trace's logical work (129 joint steps, 127 prediction runs), and 97.7% of C0's own argmax tokens agree with the trace.
- **Not exercised yet:** F1 and the multifunction/enumerated encoder variants, because WP3's models do not exist yet. Their code paths compile, and F1's call structure is unit-tested through its native test double.

## WP3: Core ML encoder arms, decoder models, probes and gates (S1 core)

The encoder arms are authored directly as MIL programs from NumPy weights (coremltools 9.0 builder),
in the **plain** layout, which mirrors `reference.py` op for op. The ANE layout (DESIGN.md D) is not built
yet.

| File | What |
| --- | --- |
| `mil/weights.py` | Tensor sources: M_P2 (`ExportSource`, int8 codes + FP32 row scales), surrogates (verified safetensors), and C0's `Encoder.mlmodelc` tensors for G0 (g0probe's parser: 294 palettized + 320 dense FP16 tensors mapped to our roles). |
| `mil/encodings.py` | One constexpr chain per arm (C1, C3, C4, C7, C8, C6s(2/4/8), C6d(4/8), C5), the byte accounting, and the effective matrices for gate 2. |
| `mil/encoder.py` | The plain FastConformer graph in FP16. `linear_pos` is folded per bucket into FP32-computed position tables, then cast. BatchNorm is folded into the depthwise conv. Masking follows `mil/MASKING.md`. Providers: `ArmProvider` (C arms, plus the FP32 diagnostic `F32`) and `G0Provider`. |
| `mil/decoder.py` | `Decoder`, `JointDecision` (C0's contracts), `JointLogits` and the fused `DecoderJoint`, in FP16 from the same model's weights. |
| `mil/build.py` | Build → convert → save → compile (`.mlmodelc`) → per-function `MLComputePlan` → manifest. Fixed / multifunction (`b2,b4,b8,b15`, weights shared by cross-function constant deduplication) / enumerated variants. |
| `mil/computeplan.swift` | Standalone `MLComputePlan` dumper that sets `functionName`; coremltools' Python API only reports a model's default function. |
| `mil/probes.py` | One- and two-layer probes per encoding (memory, time, size, extrapolation, gate 2), the C7 folding probe, and the C7/C8 stress probe. |
| `mil/refcache.py` | Cached FP32 reference outputs per clip, at FP32 and FP16-rounded scales. Gate 3. C5 activation calibration. Probe inputs. |
| `mil/gates.py` | Gates 4 (encoder and TDT heads) and 5, G0 vs C0, decoder-only heads. Summary table. |
| `mil/diag.py` | The same graph in FP32 vs FP16 vs the FP32 reference, at reduced depth. This is a diagnostic, not a gate. |
| `mil/report.py` | `results/wp3_summary.json` and `results/wp3_summary_table.txt` from all result files. |
| `mil/macrun.sh` | Mac step runner: one macguard job per step, retry on refusal. It deletes each step's Core ML cache entries and stops below 60 GB free. |
| `mil/contract.json`, `mil/MASKING.md` | I/O contract for the Swift harness; padding and masking contract. |

Results (committed, text only): `results/builds/<model>-<arm>-<variant>.json` (build manifests),
`results/gate2/`, `results/gates/` (per arm × variant × compute units, per clip and bucket),
`results/probes/` (memory/time/size with extrapolations, C7 folding with the saved MIL, stress),
`results/calibration/`, `results/diag/`, `results/wp3_summary.json` and `results/wp3_summary_table.txt`. Models live in
`<artifacts>/arms/<model>/<arm>/{fixed,multi,enum}.mlmodelc` and `<artifacts>/arms/<model>/decoder/`.

Commands (Mac, from `finetune/parakeet-ternary/ios`, each through `macguard` or `mil/macrun.sh`):

```sh
P=pyenv/.venv/bin/python
$P -m mil.refcache run --model mp2                          # 6G cap; peak 2.9 GB, 197 s
$P -m mil.probes memory; $P -m mil.probes folding; $P -m mil.probes stress; $P -m mil.probes summary
$P -m mil.build decoder --model mp2
$P -m mil.build encoder --model mp2 --arm C4 --variant fixed          # then gate 2 on the package:
$P -m mil.probes gate2 --package <A>/arms/mp2/C4/fixed.mlpackage --model mp2 --arm C4 --out results/gate2/mp2-C4.json
$P -m mil.build encoder --model mp2 --arm C4 --variant multi --drop-package
$P -m mil.gates encoder --model mp2 --arm C4 --variant multi --units cpuAndNeuralEngine
$P -m mil.gates decoder --model mp2 --units cpuAndNeuralEngine; $P -m mil.gates g0
$P -m mil.diag depth --layers 1,2,4,8,12
python3 mil/report.py                                       # any machine
```

Results (2026-10-02; Mac = M1 Pro, macOS 27.0, coremltools 9.0, Python 3.12. The Mac was shared and
heavily loaded, so load times are informational. Full table: `results/wp3_summary_table.txt`.)

- **Deployment target.** Every C arm and every decoder model targets **iOS26**, with the iOS18 op
  definitions; G0 targets iOS17, as C0 does.
  - Why: at an iOS18 target, coremltools' `common::canonicalize_quantized_lut_pattern` rewrites C4's
    `lut_to_dense → blockwise_shift_scale` into a per-row LUT (`shift_scale(LUT) → lut_to_dense`). The
    pass's own comment says the LUT-first order is only supported from iOS26.
  - At iOS26 every chain is kept exactly as DESIGN.md lists it. This is checked in each build manifest's
    constexpr literals.
- **Reference cache and gate 3** (M_P2, 82 clips, peak 2.9 GB, 197 s). FP16-rounded vs FP32 row scales:
  - encoder: rel ≤ 7.8e-4 (median 2.6e-4), abs ≤ 0.012;
  - token logits: rel ≤ 1.7e-4; duration logits: rel ≤ 6.2e-4;
  - h and c: identical, because in forced replay the prediction network never sees the encoder;
  - greedy decodes: identical on 82/82 clips.
- **Probes** (`results/probes/`). Built with 1 and 2 layers of M_P2, fixed and multifunction, and
  extrapolated as v24 = v1 + 23 (v2 − v1).
  - Projected 24-layer peak RSS: 0.9–2.4 GB, so every full build fits the 6 GB cap with a 2× margin.
  - Actual full-build peaks: 1.0–2.5 GB, including compile and compute plan (C1 is the highest).
  - Conversion: 5–10 s fixed, 30–77 s multifunction. Compile to `.mlmodelc`: 0.3–1.4 s.
- **Gate 2: bit-exact for every arm.** Checked on 10 modules at 1 layer and 20 at 2 layers, then on all
  240 ternary modules of every full fixed build (C1, C3, C4, C5's C3 weights, C7, C8, all five C6
  variants). The check rebuilds each effective matrix with coremltools' own constexpr decompression
  and compares FP16 bit patterns.
- **Encoded sizes** (240 encoder matmuls, 552.6M weights; `weight.bin` adds about 27 MB for
  subsampling, norms and the 15 s position tables):

  | | C1 | C3 / C5 | C4 / C7 / C8 | C6s2 | C6s4 | C6s8 | C6d4 | C6d8 |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | encoded MB | 1104 | 553 | 139 | 164 | 211 | 257 | 279 | 420 |
  | fixed `.mlmodelc` MB | 1132 | 580 | 166 | 192 | 238 | 285 | 307 | 448 |

  - Multifunction packages are 17–19 MB larger than the fixed ones. That difference is exactly the
    b2/b4/b8 position tables (17 MB), so the matmul weights are stored once.
  - Not measured: whether the weights also stay resident once when several functions are loaded.
    That needs the Swift harness.
- **C7 folding probe** (`results/probes/c7_folding.json`, with the saved MIL text). With the pinned
  `DEFAULT` pipeline:
  - The per-row `mul` after a LUT-weight `linear` and after a LUT-weight 1×1 `conv` survives
    conversion. It is still present in the compiled `model.mil`.
  - Control: with a dense const weight, `common::fuse_conv_scale` absorbs it for the conv, but not for
    the linear.
  - So no pass is excluded for C7.
  - Device compiler: on CPU_AND_NE, C4 and C7 give **identical** per-case errors on all 282 fixed and
    multifunction cases. On CPU_ONLY they differ. This strongly suggests that the ANE compilation folds
    C7's scale, making C7 the same as C4 on the ANE. It cannot be proven from outside the compiler
    (DESIGN.md), so it is recorded as evidence, not as established.
- **C7/C8 stress** (`results/probes/stress.json`). Setup:
  - modules: layer 0 FF1 linear1 and linear2, and layer 23 FF2 linear2;
  - adversarial rows: all +1, all −1, runs of 256, and half +1 / half −1;
  - inputs: the clip activation ×1, ×8, ×64, and |x| ×64;
  - compute units: CPU_ONLY and CPU_AND_NE.

  Outcome:
  - **C7 and C8 produce inf, so they fail the preregistered rule.** Layer-0 FF1 linear2 overflows at
    ×8 on both units; layer 23 overflows at ×64 on the ANE.
  - C1 and C4 overflow too: on the ANE at ×8, and on the CPU at ×64. At ×8 the true layer-0 output is
    51,291, close to the FP16 maximum of 65,504. The stress levels therefore exceed this model's FP16
    headroom for every encoding.
  - C7 and C8 still overflow strictly earlier: at ×8 on CPU_ONLY, C1/C4 stay finite and C7/C8 do not.
  - At ×1 everything is finite.
  - C8 also loses precision on the CPU at ×1, because P and N are each about 21,900: rel 0.0136 vs
    0.0038 for C1.
  - The rule is the design's; it is reported here, not applied as a drop.
- **Compute plans, CPU_AND_NE** (`MLComputePlan` per function, through `mil/computeplan.swift`):
  - **C3, C4, C5, C6\*, C7, C8: 99.1–99.8% of estimated cost on the ANE in every function.** The
    shares are b2 0.998, b4 0.995, b8 0.991, b15 0.994. The 17–24 CPU ops are length arithmetic, mask
    casts and `identity`.
  - **C1 (dense FP16, 1.13 GB): every op on the CPU, in every function.** The ANE does not take this
    model on the M1 Pro. Its CPU_AND_NE gate run was aborted by the guard on its first load (swap +1.44
    GB in 16 s), so C1 is gated on CPU_ONLY only (multifunction). It has an unresolved CPU fallback on the Mac; the
    phone (A17 Pro) is to be checked in S2.
  - Decoder, JointDecision, JointLogits and DecoderJoint: all CPU, as C0's are.
- **FP16 execution vs the gate-4 encoder ceiling** (`results/diag/fp16_depth.json`).
  - **The FP32 build of the same graph matches the FP32 reference to about 1e-6 rel at every depth
    (1–12 layers),** so the graph, masking, rel-pos folding and BatchNorm folding are exact.
  - The FP16 error of C4 grows with depth on the ANE: median rel 0.0011 at 1 layer, 0.0075 at 8, 0.010
    at 12 and 0.012 at 24. On the CPU it stays around 0.0045.
  - M_P2's layer-0 FF outputs reach 2,700–6,400, a tenth of the FP16 range.
  - The decoder/joint models fed with the reference encoder output pass gate 4 on both units: token
    rel 0.0022, duration 0.0051, h 0.0076, c 0.0052; 100% agreement on 96–98% decisive steps.
- **Gates 4 and 5 per arm** (`results/gates/`, all 82 clips in every bucket they fit):
  - **CPU_AND_NE, fixed and multifunction: every exact arm fails gate 4's encoder ceiling on 7 of 82
    clips at 15 s** (16–19 cases over all buckets).
    - rel max 0.028–0.036, abs max 0.40–0.50, median rel about 0.012;
    - worst clip `n02-2428-83699-0004` (rel 0.034);
    - by kind: silence 0.021, boundary `b04-N32160` 0.022.
  - Duration logits exceed rel 0.02 on 3–5 clips (max 0.039). Token logits (≤ 0.011), h and c pass,
    and **token and duration argmax agree on 100% of decisive steps** (92–97% of steps decisive). So
    the heads gate fails only on the duration-logit rel ceiling.
  - **Gate 5 passes for every multifunction arm.** Buckets vs 15 s on the boundary, silence and
    impulse clips: rel ≤ 0.0031; over all clips ≤ 0.0083. On the ANE, b4 and b8 give exactly the 15 s
    output (rel 0) on the valid frames in all 900 cases; only b2 differs slightly.
  - Equal outputs: C4 = C7, and C6s2 = C6s4 = C6s8 = C6d4 = C6d8. C3 differs.
  - **CPU_ONLY, multifunction:**
    - C7 and C6s8 pass every gate (rel max 0.0112 and 0.0115);
    - C1, C3 and C4 (identical dense FP16 on the CPU) fail on one clip (rel 0.0215, abs 0.31);
    - C8 fails on 27 cases and in the heads, from the cancellation in P − N.
  - **Enumerated shapes, CPU_AND_NE: C4 and C7 pass gates 4 and 5** (rel max 0.0109 and 0.0067),
    with an error profile unlike the fixed/multifunction ANE runs (C4 ≠ C7 here).
    - Why: **the enumerated models fall back entirely to the CPU.** Their compute plan places all ops
      (1,638 for C4, 1,878 for C7) on the CPU.
    - Our shape-generic graph needs `shape`, `gather`, `range_1d`, dynamic `slice_by_index` and
      `concat` to derive T, the masks and the position-table slice. Core ML does not place it on the
      ANE on this Mac.
    - So the enumerated control currently measures a CPU encoder, not ANE shape specialization. Making
      it ANE-eligible (shape-free masks, a static position table per enumerated shape) is an S3 item.
    - Their CPU-only compute plans fail: Core ML rejects `functionName = "main"` for these models;
      fixed in `build.py` for later builds.
  - **C5 (W8A8, exploratory) is not usable as calibrated** (rel 0.82–2.4). A per-tensor max/127 int8
    scale cannot represent activations up to |x| = 76 (layer 0, `conv_mid`); it needs its own
    calibration or QAT study (DESIGN.md).
- **G0 vs C0** (`results/gates/c0-G0-fixed-cpuAndNeuralEngine.json`). G0 is C0's own 294 palettized
  and 320 dense tensors in our graph, targeting iOS17. Its `weight.bin` is 445,187,200 bytes, the size
  of C0's. Its compute plan puts 99.4% of cost on the ANE.
  - **On all 25 clips with M mod 8 ∈ {0, 7}, and only on those, G0's output equals C0's exactly** (rel 0). On the other 57
    it differs (rel 0.0026–0.27; median 0.016 over all clips).
  - Cause: **C0's subsampling has no masking.** Its `model.mil` runs conv → relu → conv → conv → relu
    with no `mul` or `select`, and computes the lengths in FP16 floats. NeMo's `MaskedConvSequential`,
    which our reference and G0 follow (gate 1), zeroes padded frames before each stage.
  - The masking only matters when an intermediate length L1 or L2 is odd: then a stride-2 conv reaches
    a padded frame that C0 leaves at `relu(bias)`. M mod 8 ∈ {0, 7} are exactly the residues where L1
    and L2 are both even.
  - So C0's encoder lets its 15 s window padding into the last valid frames, which attention then
    spreads. This is a fourth C0 deviation from NeMo; it belongs to the product baseline.
  - G0 fails gate 4's ceilings against C0 because the two graphs mask differently, not because the
    weights or op layout differ. Where masking is moot, the two are identical.
- **Surrogate seed 0** (weight-independence builds: C1, C4, C7, fixed and multifunction; gates on
  CPU_AND_NE, and CPU_ONLY for C1).
  - Gate 2 is bit-exact on all 240 modules of each fixed build.
  - Gate 3: encoder rel ≤ 0.0021, logits rel ≤ 2.9e-4; free decodes identical on 79/82 clips.
  - The encoder behaves as on M_P2 (C4 = C7 on the ANE: rel median 0.015, max 0.021, failing on 17
    of 82 clips; gate 5 passes). C1 multifunction on CPU_ONLY: rel max 0.029, 40 failing cases.
  - **The heads fail for every seed-0 model, including the decoder alone:** h and c rel about 1.0,
    logits about 0.25, decisive fraction 0.11–0.21.
  - This is the surrogate, not the models. Its i.i.d. LSTM weights are chaotic (‖W_hh‖₂ = 27 and 36,
    hidden units saturated at ±1). Rounding them to FP16, even with FP32 arithmetic, makes h diverge
    about 1.5× per step, reaching rel ≈ 1 after about 15 of the 109 steps of the longest trace.
  - Surrogates measure execution cost only (DESIGN.md); their heads cannot be gated in FP16.
- **Mac resources** (lessons for later stages):
  - Core ML keeps a device-specialized copy of every model loaded in
    `~/Library/Caches/<process>/com.apple.e5rt.e5bundlecache`, GBs per load: 28 GB for `python` and
    24 GB for the plan tool after the first runs. These are now purged after every load.
  - coremltools leaves temporary `.mlpackage` directories in `$TMPDIR`; these are deleted too.
  - Three jobs were aborted by macguard's swap-growth limit while the user's own work loaded the Mac
    (load up to 16): C1's ANE load, C6s2's CPU-only multifunction gate, and C8's enumerated CPU plan.
    `mil/macrun.sh` now retries such aborts once the Mac is calm (at most twice), with the limits
    unchanged.
  - CPU_ONLY gates were limited to C1, C3, C4, C7, C8 and C6s8, and CPU-only compute plans of later
    builds were skipped (all-CPU by construction).
  - Compiled models are archived to `/mnt/hd/wilderness-labs-stt/parakeet-ios/arms-archive/` with
    SHA-256 lists, and removed from the Mac once gated (`mil/archive.py`; restore with
    `archive.py back MODEL ARM [names]`). C4 fixed/multi and the decoder models stay on the Mac for the
    harness.

WP3 deviations from the task or the design, and open problems:

1. **Gate 4's encoder ceiling (rel ≤ 2e-2, abs ≤ 0.25) is narrowly missed by FP16 execution of M_P2
   on the ANE, for every exact arm** (7 of 82 clips at 15 s). So is the duration-logit ceiling (3–5
   clips).
   - The FP32 graph is exact (about 1e-6), and decisions agree 100% on decisive steps. The failure is
     FP16 arithmetic on a model whose layer-0 FF outputs reach a tenth of the FP16 range.
   - Under DESIGN.md every exact arm is therefore dropped on the Mac (only C7 and C6s8 pass, on
     CPU_ONLY). Thresholds were not touched; the gate definition (e.g. ceilings for FP16 execution, or
     an FP16-emulating reference) is the design's to revisit.
2. **The C7/C8 stress rule fails C7 and C8**, but the same stress levels overflow C1 and C4 too. The
   rule is reported; whether it drops C7/C8 is for the design.
3. **C1 has no ANE placement on the M1 Pro**, and its CPU_AND_NE load was guard-aborted. It is gated on
   CPU_ONLY only.
4. **The enumerated-shape variant runs on the CPU** (see above).
   - Built and gated: C4, C7, C8, C6s2 and C6s4. All pass gates 4 and 5 on that CPU fallback (rel max
     0.0067–0.0128).
   - **Not built: the enumerated C1, C3, C5, C6s8, C6d4 and C6d8.** They were stopped because the
     shared Mac was under heavy load (macguard swap aborts, waits for calm) and they add no new
     information. The step lists for them are in `mil/` (`mil.build encoder --variant enum`, then
     `replan`, then `gates`).
5. **G0** has C0's fixed window only, at iOS17 (DESIGN.md: same shapes and target as C0). It is gated
   against C0, not an FP32 reference: C0 ships `linear_pos` only folded, so no FP32 reference exists.
   G0 ≠ C0 traces to C0's missing subsampling masking.
6. **seed0:** fixed and multifunction only (no enumerated builds). Its heads cannot be gated in FP16
   (chaotic surrogate LSTM).
7. **CPU_ONLY coverage:** gates for C1, C3, C4, C7, C8 and C6s8 (multifunction) only, plus the
   decoders. CPU-only compute plans were skipped for later builds (all-CPU by construction).
8. **Not done in WP3** (these need the Swift harness or the phone):
   - the ANE layout (DESIGN.md D);
   - Instruments placement traces (gate 6);
   - "prepare and cache" load events;
   - resident memory with several functions loaded.
9. **C5's int8 activation calibration** (per-tensor max/127) is unusable for accuracy. Speed-only, as
   designed.
