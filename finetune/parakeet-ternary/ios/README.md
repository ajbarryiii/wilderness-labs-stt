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
