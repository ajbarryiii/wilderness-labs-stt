# kea-b (MB1) 1-bit Core ML encoder arms

Goal (user): optimize the binary Parakeet model's encoder for **latency**, with an encoder **memory
footprint smaller than the shipped ternary model** (kea-t, `weight.bin` 345,404,096 bytes). This directory
holds the 1-bit encoder arms' screening, gate 2, placement records and the LocalFlow bundle builder.
Final timing comparisons belong to the benchmark harness (`bench/`, another agent) and the GPU/CPU kernel
work to `gpu-cpu/` (a third agent); the timings here are informational screening only (shared M1 Pro,
Python `predict` calls, synthetic mel input).

## What is here

| Path | What |
| --- | --- |
| `../parakeet-ternary/ios/mil/binary_export.py` (new) | Native `parakeet-binary-v1` reader: verifies format and export SHA-256 (manifest `sha256` and `files`), unpacks LSB-first 1-bit codes to int8 ±1, FP32 row scales, FP16 floats widened exactly, `raw()` for stored dtypes, `verify_tokenizer()`. numpy only. |
| `../parakeet-ternary/ios/mil/weights.py` | `Source("mb1")` (default path: Mac `…/cpu1bit/data/exports/MB1`, NixOS the canonical run export) with the same `(codes, scale)` interface the encoder build uses, so no 2-bit re-encoding is needed. |
| `../parakeet-ternary/ios/mil/encodings.py` | 1-bit arms `BINARY_ARMS` (below), registered in `ENCODER_ARMS`/`EXACT`/`CHAINS`; existing arms unchanged. `describe()` now names `int4` as such (only binary arms use it). |
| `../parakeet-ternary/ios/mil/probes.py` | gate 2 and the stress program treat B1P's post-matmul scale like C7's. |
| `localflow.py` | LocalFlow's geometry: functions b4/b8/b15/b30, rel-pos tables folded for 30 s (376 frames). |
| `prepare_bundle.py` | LocalFlow bundle for one 1-bit arm (derived from LocalFlow's `scripts/prepare-parakeet.py`; reads MB1 natively, pins its SHA-256). |
| `screen.py` | gate 2 per module, probe builds (sizes, gate 2, compute plans for ANE/GPU/CPU), round-robin probe timings, B1P headroom (activations, stress), bundle gate 2 / plans / load times. |
| `guarded.sh` | Mac job wrapper: upstream `macguard` (6 GB cap, 4 threads), artifact-area `TMPDIR`, `keab1py` interpreter alias (own Core ML cache, the only one purged); retries macguard refusals. |
| `tests/` | Contract tests (invented data only): reader, arms (gate 2 before and after conversion + save/reload, chain as authored, byte accounting), bundle manifest. |
| `results/` | Text results (JSON). |

## Arms (exact MIL chains; iOS26 target, FP16, plain layout, coremltools 9.0)

All 240 arm-encoded projections per model (10 per layer; the 24 `linear_pos` are folded into FP16 position
tables as before). `c` = ±1 code, `s` = FP16(row scale).

| Arm | Chain | Stored bits/weight |
| --- | --- | --- |
| **B1Q** | `constexpr_lut_to_dense(indices uint1 [out,in] = [c>0], lut int8 [1,1,2,1] = {-1,+1})` → int8 dense → `constexpr_blockwise_shift_scale(scale fp16 [out,1])` → linear / 1×1 conv | 1 |
| **B1S** | as B1Q with an fp16 LUT {-1,+1} (C4 analogue) | 1 |
| B1L | `constexpr_lut_to_dense(uint1, lut fp16 [out,1,2,1] = {-s_r,+s_r})` (per-row LUT) | 1 |
| B1P | `constexpr_lut_to_dense(uint1, fp16 {-1,+1})` → matmul → `mul(s)` per row (C7 analogue) | 1 |
| Bg2/Bg4/Bg8 | `constexpr_lut_to_dense(uint2/3/4 = 2k+[c<0], lut fp16 [out/g,1,2g,1] = {+s,-s per row})` (C6d analogue, no zero) | 2/3/4 |
| Bi4 | `constexpr_blockwise_shift_scale(data int4 ±1, scale fp16 [out,1])` (C3 analogue) | 4 |
| B1V2/B1V4 (added) | `constexpr_lut_to_dense(uintV [out/V,in] = Σ_t [c_(gV+t)>0]<<t, vector_axis=0, lut fp16 [out/V,1,2^V,V])`: vector palettization over V rows, scales inside the LUT | 1 |

Every chain stays as authored after conversion (build manifests record the literals, e.g. B1Q:
`constexpr_lut_to_dense: indices uint1 [4096,1024], lut int8 [1,1,2,1]` and
`constexpr_blockwise_shift_scale: data int8 [4096,1024], scale fp16 [4096,1]`; nothing is rewritten into
a per-row LUT at iOS26).

## Results (2026-10-21, M1 Pro, macOS 27, shared and heavily loaded)

**Gate 2: bit-exact for every arm.**
- Per module (`results/gate2-modules.json`): all **264** MB1 projections (incl. `linear_pos`, 603,979,776
  weights), coremltools' constexpr decompression vs codes × FP16(s), FP16 bit patterns, for B1L, B1S, B1Q,
  B1P, Bg2, Bg4, Bg8, Bi4. B1V2/B1V4 are covered by the contract tests and the 4-layer package gate.
- Packages: 4-layer multifunction probes for all 11 arms (+C6s8 reference) exact in all 4 functions;
  full B1Q and B1S bundle packages exact on **240 modules × 4 functions** each (`results/gate2-package-B1{Q,S}.json`).

**Sizes.** `weight.bin` = encoded weights + 70.19 MB per-function position tables + 11.4 MB other FP16
tensors (subsampling, norms, depthwise). This overhead is the same in every arm, so 4-bit arms land at
the size of kea-b C6s8. In that baseline the all-ones sparsity masks deduplicate to five blobs, so it is
effectively 4-bit dense plus a LUT. 24-layer sizes below are measured for B1Q/B1S. For the other arms
they are extrapolated from the 4-layer probes (base + 24 × per-layer; the extrapolation reproduces B1Q
exactly).

| | weight.bin (bytes) | vs ternary 345,404,096 | vs kea-b C6s8 372,690,944 |
| --- | --- | --- | --- |
| **B1Q, B1S** (measured) | 153,226,944 | −55.6% | −58.9% |
| B1P | 153.2 M | −55.6% | |
| B1L | 154.1 M | −55.4% | |
| B1V2 / B1V4 | 155.7 M / 165.8 M | | |
| Bg2 | 226.4 M | −34% | |
| Bg4 | 298.8 M | −13.5% | |
| Bg8 / Bi4 | 371.1 M / 370.3 M | **+7% (larger)** | ≈ |

**Placement (MLComputePlan per function).**
- 4-layer probes, `cpuAndNeuralEngine`: every scalar-LUT / shift-scale arm (B1L, B1S, B1Q, B1P, Bg*,
  Bi4, and the C6s8 reference) is placed identically. All 62 matmul/conv ops are on the ANE; 17–19 CPU ops
  (input cast, length arithmetic, `less` masks, `identity`, one `relu`/`transpose`); ANE cost share
  0.950–0.975.
- **B1V2/B1V4 fall off the ANE.** 40–56 of 62 matmuls are planned on the CPU, and only 0.8–5.5% of
  estimated cost is on the ANE.
- **Full B1Q and B1S bundles** (`results/plans/B1Q.json`, `B1S.json`; identical plans):
  - ANE: 1,374–1,376 ops, all 342 matmul/conv ops; 17–19 CPU ops; ANE cost share 0.990–0.995 (b4 0.9952,
    b8 0.9909, b15 0.9944, b30 0.9898).
  - `cpuAndGPU`: only 6–9 ops on the GPU (subsampling convs, one linear; 0.2–0.4% of cost). Every encoder
    matmul is planned on the **CPU**, so the WP6a finding holds for the 1-bit LUT arms.
  - `cpuOnly`: all 1,393 ops on the CPU.

**Probe timings** (4 layers of MB1, LocalFlow buckets, `cpuAndNeuralEngine`, round-robin over arms,
3 rounds × 10 calls, median of per-round medians, ms; `results/probes/bench-4L-*.json`). Informational
only: shared Mac (load 25–70 from other experiments), Python predict, synthetic mel.

| Arm | b4 | b8 | b15 | b30 | first load (4 L) s |
| --- | --- | --- | --- | --- | --- |
| C6s8 (kea-b's chain, reference) | 4.45 / 4.65 | 5.92 / 5.89 | 11.2 / 11.0 | 29.2 / 29.1 | 16.5–17.4 |
| **B1Q** | 4.78 / 4.83 | 6.17 / 6.27 | 11.1 / 11.3 | **27.5 / 27.5** | 3.7–4.6 |
| **B1S** | 4.73 / 4.81 | 6.51 / 6.29 | 11.1 / 11.3 | 27.7 / 27.7 | 3.8–5.0 |
| B1P | 4.73 | 6.35 | 11.1 | 27.8 | 4.0–5.0 |
| Bi4 | 4.47 / 4.47 | 6.06 / 5.83 | 11.2 / 11.1 | 27.9 / 27.8 | 2.7–3.7 |
| Bg8 | 4.30 / 4.45 | 6.01 / 5.93 | 11.0 / 11.0 | 29.1 / 29.0 | 3.7–4.6 |
| Bg4 | 4.34 | 5.85 | 10.9 | 32.3 | 6.0–6.8 |
| Bg2 | 4.61 | 8.35 | 13.3 | 43.2 | 7.5–8.4 |
| B1L | 7.06 | 12.75 | 20.5 | 66.4 | 13.6–14.4 |
| B1V2 / B1V4 | 10.9 / 10.2 | 16.7 / 16.8 | 28.3 / 26.2 | 50.3 / 53.3 | 1.0–4.0 |

(two values = the two independent round-robin runs.) Interpretation:
- On the ANE, a per-tensor LUT followed by a per-row shift/scale (B1Q/B1S) runs like the 4-bit
  C6s8/Bg8/Bi4 chains. These arms are slightly faster at 30 s (−5%), within noise at 15 s, and up to
  ~0.3 ms per 4 layers slower at 4 s.
- Many small LUTs are slow: per-row (B1L), 2 rows (Bg2), and 4 rows at 30 s (Bg4).
- First ANE compile is ~4× faster than C6s8.

Full-depth bundles (`Encoder.mlmodelc`, 5 calls after 2 warm-ups, same caveats; B1S: 17.7 / 22.2 / 43.7 / 127.5 ms, first loads 23.8–28.8 s). B1Q:
- 18.0 / 21.9 / 45.8 / 123.9 ms (b4/b8/b15/b30);
- first ANE load 22.8 / 26.3 / 26.2 / 29.0 s per function, then 0.32–0.34 s; own Core ML cache after
  the first load 0.1 MB.

Paired full-depth comparisons against kea-t/kea-b are the harness's job.

**B1P FP16 headroom** (`results/b1p-activations.json`, `results/b1p-stress.json`).
- FP32 reference of MB1 on 9 public LibriSpeech dev-clean clips (2 per bucket + the longest 15 s clip):
  - the worst pre-scale B1P matmul output |x·c| is **3,731** (layer 0 FF1 linear2): 17.6× below FP16
    max on real inputs;
  - the true scaled output there is 1,649; the pre-scale/scaled ratio is median 5.4, max 9.5;
  - no module exceeds FP16 at ×1 or ×8 in exact arithmetic; at ×64, 2 modules overflow pre-scale and
    1 overflows even scaled;
  - max ‖x‖₁ is 8,553, so an all +1 row reaches 68k at ×8.
- Core ML stress (adversarial ±1 rows, inputs ×1/×8/×64/|x|×64, CPU/GPU/ANE):
  - everything is finite at ×1;
  - **on the ANE, B1S and B1P both go non-finite at ×8** (layer 0 FF1 linear2) and at |x|×64
    (layer 23), where **B1Q, B1L and Bg2 stay finite**. On CPU/GPU all five behave alike.
  - So the ANE evidently applies B1S's fp16-LUT shift/scale after the matmul, exactly like B1P, while
    B1Q's int8-LUT form keeps scaled weights. Under WP3's relative rule (reference = B1S, the C4
    analogue), B1P "passes", but it has no advantage.

**Decisions.**
- **B1Q: primary candidate, bundle built.** Smallest (153.2 MB, −56% vs ternary), fastest or tied on
  the ANE, all-ANE placement, more FP16 headroom on the ANE than B1S/B1P.
- **B1S: bundle built** as the C4-analogue comparison (same size and speed; less ANE headroom).
- Dropped:
  - B1P: dominated by B1S (same size and speed; the ANE already treats B1S like B1P) plus the
    documented post-scale overflow risk.
  - B1L, Bg2, B1V2, B1V4: clearly slower (1.5–2.4×). B1V falls off the ANE.
  - Bg4: 13.5% smaller than ternary but 2× B1Q's size and slower at 30 s.
  - Bg8, Bi4: not smaller than ternary (≈370 MB, the size rule).

## Bundles

`/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios/kea-b1/<arm>/bundle`. The conversion package is
in `../conversion/mb1/<arm>/` and the build record in `../build-record.json`; copies are in
`results/bundles/`. `bundle.json`: model `localflow`, display_name `LocalFlow`, base_model `kea-b`,
training_step 250000, export_sha256 `9873d2f1…7b9651` (the 1-bit export), encoder = arm, layout plain,
compute_units cpuAndNeuralEngine (compute-unit agnostic; the harness may run other units), decoder
native-fp32, buckets 4/8/15/30, per-file SHA-256.

| Arm | bundle.json SHA-256 | weight.bin | Encoder.mlmodelc | bundle files total | convert / compile s | peak RSS | gate 2 | first ANE load / fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1Q | `3f7a89ae75f3880e9391071afd5f713631853412665790d8e4ac57ff0dc75d7b` | 153,226,944 | 156,015,052 | 191,954,926 | 31.1 / 1.0 | 1.40 GB | exact, 240 × 4 | 22.8–29.0 s |
| B1S | `410ffedee59b33a1c75fd580f5850dc01f6d73a669f8f622af0dd452cba91aab` | 153,226,944 | 156,024,652 | 191,964,526 | 31.4 / 0.97 | 1.26 GB | exact, 240 × 4 | 23.8–28.8 s |

- `frontend.f32bin`, `decoder_joint.f32bin`, `decoder_joint.f16bin` and `vocabulary.json` are
  byte-identical to the baseline kea-b bundle's.
- `frontend.json` and `decoder_joint.json` differ only in `provenance.export_sha256`: the 1-bit export
  instead of the 2-bit re-encoding.
- The tokenizer used is `freeflow-main/export/tokenizer/`; its three files match the MB1 manifest's
  `files` hashes (verified in the build).

## Commands

Mac, through the jobs system (`priority normal`, `full_machine`); `guarded.sh` wraps macguard:

```sh
cd finetune/parakeet-binary-coreml
./guarded.sh 900  sh -c '$PY -m unittest discover -s tests -v'            # contract tests
./guarded.sh 5400 sh -c '$PY screen.py gate2-modules'
./guarded.sh 5400 sh -c '$PY screen.py probe --arms B1Q,B1S,C6s8 --layers 4 --units "" --keep'
./guarded.sh 5400 sh -c '$PY screen.py bench --arms B1Q,B1S,C6s8 --layers 4 --rounds 3'
./guarded.sh 3600 sh -c '$PY screen.py activations' && ./guarded.sh 3600 sh -c '$PY screen.py stress'
K=/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios/kea-b1/B1Q
./guarded.sh 3600 sh -c "\$PY prepare_bundle.py --arm B1Q --output $K/bundle"
./guarded.sh 3600 sh -c "\$PY screen.py gate2-package --arm B1Q --package $K/conversion/mb1/B1Q/multi-localflow.mlpackage"
./guarded.sh 5400 sh -c "\$PY screen.py plan --arm B1Q --path $K/bundle/Encoder.mlmodelc"
./guarded.sh 5400 sh -c "\$PY screen.py loadtimes --arm B1Q --path $K/bundle/Encoder.mlmodelc"
```

Tests: `Ran 16 tests … OK` (reader 7, arms 5 covering all 10 binary arms at rank 2 and 3, bundle 4).

## Open risks

- **No end-to-end accuracy gate was run** on the 1-bit bundles: no gates 3–5, no WER, no decode
  comparison. Gate 2 proves the weights are exactly codes × FP16(s), the same values the kea-b C6s8
  bundle encodes. FP16 execution, however, is backend-defined per chain. On the ANE, B1S behaves like a
  post-scale arm, so B1Q and B1S outputs can differ from C6s8's. The harness should compare transcripts
  and encoder outputs against kea-b C6s8 / FP32.
- Timings are screening only: 4 layers, Python, synthetic input, shared Mac under heavy load. The ±0.3 ms
  short-bucket differences at 4 layers are near noise.
- The position tables are 70.2 MB (46% of a 1-bit `weight.bin`), stored separately per function. Sharing
  one 30 s table across functions could shrink every arm further. Not attempted, because constant folding
  would rematerialize slices.
- `cpuAndGPU` runs these encoders on the CPU, so a GPU path needs custom kernels (the gpu-cpu agent's
  scope).
- B1P/B1S overflow at ×8 under adversarial rows on the ANE. On real inputs the headroom is 17.6×.
