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
| `macguard` | Wrapper for every Mac job (Python, the Mac's `/usr/bin/python3`, stdlib only): start conditions, flock lock held through cleanup, own session and process group, RSS cap, timeout, memory and swap aborts, fail-closed probes, TERM→KILL until the group is verifiably empty, logging. Exit: the job's status; 124 if the guard aborted it; 130 interrupted; 3 refused; 2 usage; 125 internal. |
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
| `c0report.py` | Scores `parakeet-bench` JSON lines against `clips.json` and the B0 traces, with the parent experiment's Whisper-normalized WER. It enforces repetition completeness per clip and reports, per bucket and stage, typical latency (median over clips of each clip's median) and the Harrell–Davis p95 over pooled calls. A run is labelled `baseline-eligible` only with at least 10 timed calls per clip (`--min-timed`); fewer requires `--smoke` and is labelled smoke. |
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
# macguard runs under the /usr/bin/python3 shim, which exports SDKROOT = the Command Line Tools SDK (Swift 6.4 on
# this Mac) into the job; pin Xcode's SDK so it matches Xcode's Swift 6.3.1 compiler
S=$(xcrun --sdk macosx --show-sdk-path)
(cd ios/bench && ../macguard --rss-cap 4G --timeout 900 -- env SDKROOT=$S swift build -c release)
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
