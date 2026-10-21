# References and provenance (S0)

Pinned sources behind `DESIGN.md` "Prior evidence (with limits)", with verbatim quotes for each claim.

Retrieved 2026-10-02. Method: GitHub sources were resolved with `gh api` to the current `main` commit of
each repository (and, separately, the last commit touching each file), then read at that SHA through the
contents API. Line numbers (`#Lx-Ly`) refer to the pinned SHA. The Hugging Face card was read at the `sha`
reported by `/api/models/...`. Apple and MLX pages have no revisions: the URL and retrieval date are given,
and quotes come from the page HTML (WWDC transcripts are on the page; Apple docs via their `.md` variant).
Tags were dereferenced to commits (`git/ref/tags/...`; both tags used here are lightweight).

## Sources

| # | Source | Pinned permalink | Revision |
|---|---|---|---|
| 1 | mobius `parakeet-redux` Core ML README | https://github.com/FluidInference/mobius/blob/864ef8050f2f281d0761de26e3a03108f9f1ce73/models/stt/parakeet-redux/coreml/README.md | mobius main `864ef80` (2026-09-26); file last changed `c41b7da` (2026-09-24) |
| 2 | FluidAudio `Phonon2.md` | https://github.com/FluidInference/FluidAudio/blob/0b1f46289fe27d95b5e66ad8be46e64f5ee02ae7/Documentation/ASR/Phonon2.md | FluidAudio main `0b1f462` (2026-10-01); file last changed `a8f1482` (2026-10-01) |
| 3 | HF `FluidInference/phonon-2-coreml` card | https://huggingface.co/FluidInference/phonon-2-coreml/blob/a812a0dfef205660787ef6234317ab79acf4d5d6/README.md | HF sha `a812a0dfef205660787ef6234317ab79acf4d5d6` (lastModified 2026-10-01T16:55:55Z) |
| 4 | FluidAudio `EncoderComputePlacement.md` | https://github.com/FluidInference/FluidAudio/blob/0b1f46289fe27d95b5e66ad8be46e64f5ee02ae7/Documentation/ASR/EncoderComputePlacement.md | main `0b1f462`; file last changed `d371275` (2026-06-05) |
| 5 | FluidAudio `ANE_Profiler.md` | https://github.com/FluidInference/FluidAudio/blob/0b1f46289fe27d95b5e66ad8be46e64f5ee02ae7/Documentation/ANE_Profiler.md | main `0b1f462`; file last changed `3ca01bb` (2026-09-23) |
| 6 | FluidAudio `Benchmarks.md` | https://github.com/FluidInference/FluidAudio/blob/0b1f46289fe27d95b5e66ad8be46e64f5ee02ae7/Documentation/Benchmarks.md | main `0b1f462`; file last changed `a8f1482` (2026-10-01) |
| 7 | mobius `parakeet-tdt-v2-0.6b/coreml` (directory) | https://github.com/FluidInference/mobius/tree/864ef8050f2f281d0761de26e3a03108f9f1ce73/models/stt/parakeet-tdt-v2-0.6b/coreml | main `864ef80`; tree `e44cec3`; directory last changed `6bf7833` (2025-09-26) |
| 8 | coremltools guide: Optimization overview | https://apple.github.io/coremltools/docs-guides/source/opt-overview.html | retrieved 2026-10-02; gh-pages `c8ede10` (see note) |
| 9 | coremltools guide: Palettization overview | https://apple.github.io/coremltools/docs-guides/source/opt-palettization-overview.html | retrieved 2026-10-02; gh-pages `c8ede10` |
| 10 | coremltools guide: Flexible input shapes | https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html | retrieved 2026-10-02; gh-pages `c8ede10` |
| 11 | coremltools guide: Multifunction models | https://apple.github.io/coremltools/docs-guides/source/multifunction-models.html | retrieved 2026-10-02; gh-pages `c8ede10` |
| 12 | coremltools 9.0 `iOS18/compression.py` | https://github.com/apple/coremltools/blob/428d4b2658dfc44194f27f4f36870751be402ff7/coremltools/converters/mil/mil/ops/defs/iOS18/compression.py | tag `9.0` -> commit `428d4b2658dfc44194f27f4f36870751be402ff7` (released 2025-11-10) |
| 13 | coremltools 9.0 `_quantization_passes.py` | https://github.com/apple/coremltools/blob/428d4b2658dfc44194f27f4f36870751be402ff7/coremltools/optimize/coreml/_quantization_passes.py | tag `9.0` -> `428d4b2` |
| 13a | coremltools 9.0 `optimize/_utils.py` (supporting) | https://github.com/apple/coremltools/blob/428d4b2658dfc44194f27f4f36870751be402ff7/coremltools/optimize/_utils.py | tag `9.0` -> `428d4b2` |
| 13b | coremltools 9.0 `optimize/coreml/_config.py` (supporting) | https://github.com/apple/coremltools/blob/428d4b2658dfc44194f27f4f36870751be402ff7/coremltools/optimize/coreml/_config.py | tag `9.0` -> `428d4b2` |
| 14 | WWDC23 10047 "Use Core ML Tools for machine learning model compression" | https://developer.apple.com/videos/play/wwdc2023/10047/ | retrieved 2026-10-02 |
| 15 | WWDC23 10049 "Improve Core ML integration with async prediction" | https://developer.apple.com/videos/play/wwdc2023/10049/ | retrieved 2026-10-02 |
| 16 | Apple: Measuring your app's power use with Power Profiler | https://developer.apple.com/documentation/xcode/measuring-your-app-s-power-use-with-power-profiler | retrieved 2026-10-02 |
| 17 | WWDC25 226 "Profile and optimize power usage in your app" | https://developer.apple.com/videos/play/wwdc2025/226/ | retrieved 2026-10-02 |
| 17a | WWDC25 227 "Finish tasks in the background" (added: source of the background-GPU claim) | https://developer.apple.com/videos/play/wwdc2025/227/ | retrieved 2026-10-02 |
| 17b | Apple entitlement: Background GPU Access (added) | https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.background-tasks.continued-processing.gpu | retrieved 2026-10-02 |
| 18 | MLX `mlx.core.quantize` | https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html | retrieved 2026-10-02; page title "MLX 0.32.3 documentation"; tag `v0.32.3` -> `64ea011cb65f14d9ce2737e60db9a4ae91ed7441` |
| 19 | macmon | https://github.com/vladkens/macmon/tree/6919d7781b6c55a6e3bedff83a210435837e1dfe | latest release `v0.8.2` (2026-08-04) -> commit `6919d7781b6c55a6e3bedff83a210435837e1dfe` (also current main) |

Notes on revisions:
- Full SHAs: mobius main `864ef8050f2f281d0761de26e3a03108f9f1ce73`; FluidAudio main
  `0b1f46289fe27d95b5e66ad8be46e64f5ee02ae7`.
- **coremltools docs (8-11).** The pages state no coremltools version (footer "© Copyright 2024, Apple Inc.").
  All four HTML files were last changed on the `gh-pages` branch by commit
  `c8ede10b25b36fd9a136e372381b7e0fe6df1ed2` ("Update Documentation (#2581)", 2025-08-12). That date is after
  `9.0b1` (2025-07-28) and before `9.0` (2025-11-10), so the guide corresponds to the 9.0 beta line, not to the
  9.0 tag. Pinned copy, for example:
  https://github.com/apple/coremltools/blob/c8ede10b25b36fd9a136e372381b7e0fe6df1ed2/docs-guides/source/flexible-inputs.html
- **macmon.** DESIGN.md names 0.8.2; tag `v0.8.2` exists (with the `v` prefix; `0.8.2` does not resolve) and
  is the latest release.
- **Model bases.** Sources 1-6 measure **Parakeet TDT v3**-family models (redux and Phonon-2 are re-trainings of
  `parakeet-tdt-0.6b-v3`), not v2. Source 7 is the v2 conversion that C0 comes from.

### Source 7: key files (blob permalinks at mobius `864ef80`; all last changed in `6bf7833`)

Base: `https://github.com/FluidInference/mobius/blob/864ef8050f2f281d0761de26e3a03108f9f1ce73/models/stt/parakeet-tdt-v2-0.6b/coreml/`

- `README.md`, `convert-parakeet.py` (export), `quantize_coreml.py` (quantization variants),
  `compile_modelc.py`, `individual_components.py`, `compare-components.py`,
  `speech_to_text_streaming_infer_rnnt.py`, `pyproject.toml`, `uv.lock`, `agents.md`; directories `audio/`,
  `context/`, `plots/`.
- `quantize_coreml.py#L734-L745`, the encoder 6-bit k-means variant used for the G0 fallback:
  `name="enc6bit-palettize"`, `global_config=OpPalettizerConfig(mode="kmeans", nbits=6)` (L740),
  `whitelist=["encoder"]`. No `granularity` is given, so coremltools' default applies (`"per_tensor"
  (default)`, source 13b `_config.py#L668-L671`), which gives one LUT per tensor.
- `README.md#L17`: "Shapes and I/O match the fixed 15‑second window contract." `README.md#L26`: "Minimum
  deployment target: iOS 17."

## Claims in DESIGN.md "Prior evidence"

### 1. parakeet-redux: FP16 at 20.8 ms on the ANE

DESIGN.md: "(M-series Mac, 15 s window): FP16 ran at 20.8 ms on the ANE."

- Source 1 `#L52`: "## Results (M-series Mac, macOS 27, coremltools 9.0b1)"; `#L54`: "Encoder, one 15 s window".
- `#L73`: "| fp16, iOS18 | 8.7 s | 20.8 ms | 1381 ANE, 4 CPU |" (`#L72` has the iOS17 target at 21.2 ms).
- `#L90`: "The same graph with plain fp16 weights compiles in 9 s and runs at 20.8 ms".

Verdict: **confirmed** (M-series Mac, macOS 27, coremltools 9.0b1, v3-based redux encoder, iOS18 target;
the iOS17 target measured 21.2 ms).

### 2. parakeet-redux: per-row 2-bit diagnostic at 70.4 ms, 45 s first load

DESIGN.md: "A per-row 2-bit *diagnostic* encoding (not exact) ran at 70.4 ms with a 45 s first load."

- Source 1 `#L77`: "| 2-bit, per-row scale (`joint-perchannel`) | 44.9 s | 70.4 ms | 1381 ANE, 4 CPU |"
- `#L78`: "| 2-bit, per-row LUT (`lut-perrow`) | 45.2 s | 70.6 ms | 1381 ANE, 4 CPU |"
- `#L85-L88`: "Dropping to one scale per row restores the fp16 placement exactly (1381 ANE, 4 CPU) and a 45 s compile,
  but lands at 70 ms — 3× the fp16 encoder. Both per-row variants are deliberately inexact (`cos 0.25`)".

Verdict: **confirmed**. The table gives 44.9 s first load; the prose rounds to "45 s". Same hardware as claim 1.

### 3. parakeet-redux: int8 per-channel at 27.5 ms

DESIGN.md: "int8 per-channel ran at 27.5 ms."

- Source 1 `#L150-L154`: "iOS 17 gets data-free int8 per-channel quantization of the iOS 17 fp16 export
  (`coremltools.optimize.coreml.linear_quantize_weights`, symmetric per-channel ...) It was not in the sweep
  above, and it runs well on the ANE: 1381 ANE ops, ~14 s first compile, 27.5 ms/window ANE, 16.3 ms GPU".

Verdict: **confirmed** (ANE, iOS 17 target, 15 s window). The section does not restate hardware; it sits in the
same README as the "M-series Mac, macOS 27" results, but "not in the sweep above".

### 4. parakeet-redux: GPU ran 2-bit weights at 21 ms

DESIGN.md: "The GPU ran 2-bit weights at 21 ms."

- Source 1 `#L92`: "The GPU, by contrast, decompresses in-kernel: 21 ms and a 0.6 s load, matching fp16."
- `#L60-L61` (table): `joint` 2-bit "0.6 s / 21.3 ms"; `blockwise-lut` 2-bit "0.6 s / 21.1 ms" (GPU column).

Verdict: **confirmed**. Note: those GPU numbers are for the exact per-(row, 128-block) 2-bit encodings, which on the
ANE ran at 52.2 / 44.8 ms with 368 s / 320 s first loads (`#L74-L75`).

### 5. Phonon-2: grouped sparse-palette at 18.6 ms; dense grouped-palette variant ties it

DESIGN.md: "a grouped sparse-palette encoding ran at 18.6 ms on the ANE. Its dense grouped-palette variant
tied it, so the evidence favours grouped palettes, not sparsity as such."

- Source 2 `#L16-L18`: "a sparsity mask (51 % of the weights are zero) plus fp16 palettes over the non-zeros
  (iOS 18 `constexpr_lut_to_sparse` + `constexpr_sparse_to_dense`, one palette per 8 output rows)".
- Source 2 `#L41`: "| phonon2 `Encoder.mlmodelc` (sparse, 8 rows/palette) | 321 MB | **18.6 ms** | **159×** | ..."
- Source 2 `#L44`: "| `Encoder_lut6.mlmodelc` (dense) | 470 MB | 18.6 ms | 155× | 16 ms, 0.6 s load |"
- Source 2 `#L47-L48`: "The Neural Engine's palette cost grows with the number of palettes, not their bit width, which is
  why 8 rows per palette beats v3's encoder while per-row palettes are 3× slower."
- Source 3 `#L43`: "One 15 s window on an M5 Pro (macOS 27)"; `#L48`: "sparse mask + 6-bit palette per 8 rows |
  **18.6 ms**"; `#L51`: "`Encoder_lut6.mlmodelc` | 470 MB | dense 6-bit palette per 8 rows | 18.6 ms".
- Source 3 `#L50`, `#L52`: per-row palettes, sparse 2-bit "70 ms" and dense 3-bit "72 ms"; v3 6-bit reference
  "23.5 ms" (`#L53`).
- Source 6 `#L69-L79` corroborates at whole-pipeline level: "M5 Pro, macOS 27, default encoder compute units
  (ANE)"; `phonon2` test-clean RTFx "**159×**".

Verdict: **confirmed** (M5 Pro, macOS 27, one 15 s window, ANE). Both are 6-bit palettes per 8 output rows. The
sparse variant leads on end-to-end test-clean RTFx (159× vs 155×). The two are tied on per-window encoder latency
only.

### 6. coremltools 9: per-tensor `enable_per_channel_scale` gives one shared LUT plus `constexpr_blockwise_shift_scale`

DESIGN.md: "`enable_per_channel_scale` with per-tensor granularity produces one shared LUT plus
`constexpr_blockwise_shift_scale`; it does *not* produce one palette per row."

Code at tag 9.0 (`428d4b2`):
- Source 13b `_config.py#L668-L671`: "granularity: str / Granularity for quantization. / * ``"per_tensor"``
  (default) / * ``"per_grouped_channel"``"; `#L689-L691`: "enable_per_channel_scale: bool / * When set to True,
  weights are normalized along the output channels using per channel scales before being palettized."
- Source 13a `_utils.py#L509-L512`: "block_sizes = [0] * len(weight_to_compress.shape) / if
  op_config.granularity == CompressionGranularity.PER_TENSOR: / input_channel_block_size = 0 /
  output_channel_block_size = 0".
- Source 13 `_quantization_passes.py#L862-L863`: "# Per-tensor compression, just need to pick a dummy axis. /
  channel_axis = 0"; `#L918-L919`: "if channel_group_size == 0: / channel_group_size = channel_num" (that is,
  one group spanning all channels, so one LUT).
- Source 13 `#L1067-L1071`: "if op_config.enable_per_channel_scale: / # Normalize by per channel scales before
  doing palettization. / per_channel_scale = np.max(np.abs(weight_to_compress), axis=channel_axis,
  keepdims=True)".
- Source 13 `#L1136-L1148`: "if op_config.enable_per_channel_scale: / if not
  is_current_opset_version_compatible_with(AvailableTarget.iOS18): ... / new_var =
  mb.constexpr_blockwise_shift_scale( / data=new_var, / scale=per_channel_scale, / offset=None,".
- Source 12 `compression.py#L35`: "Generic expression: output = scale * (data - offset)";
  `#L173-L180`: "supports block-wise / vector palettization. / LUT's rank is K + 2 ... e.g., when indices_shape
  = [2, 3, 4], lut_shape[:3] = [1, 1, 2], it means that there are two lookup tables over the last axis."

Docs (source 9): "The figure above shows what is referred to as per_tensor granularity, where the entire tensor
shares a single LUT." and "Starting with iOS18/macOS15, a mode called per_grouped_channel is available. It
allows a group of channels, specified by the parameter group_size, to share a single LUT ... a weight of shape
(1024, 1024), with group_size=16, will have 64 LUTs." "Per-channel scale: When this mode is enabled, weights are
normalized along the output channels using per-channel scales before being palettized."

Verdict: **confirmed** by reading the code at the 9.0 tag. The constexpr chain is one per-tensor
`constexpr_lut_to_dense` feeding `constexpr_blockwise_shift_scale` with a per-output-channel scale, and it
requires iOS18. The claim was not run. Related (source 1 `#L49-L50`): "`palettize_weights` cannot express a
per-(row, block) LUT ("general block-wise palettization is not supported")". This matches source 13
`#L851-L853`.

### 7. Apple: just-in-time decompression on the ANE from iOS 17, with gains for memory-bound models

DESIGN.md: "compressed weights are decompressed just in time on the ANE (from iOS 17), with gains for
memory-bound models."

- Source 14 (transcript): "This step of decompression takes place ahead of time in the iOS 16 runtime." ...
  "However, in iOS 17, in certain scenarios, the weights are decompressed just in time, just before the operation
  is executed. This has the advantage of loading smaller bit weights from the memory at the cost of doing
  decompression in every inference call. For certain compute units, such as the Neural Engine, and certain types
  of models that are memory bound, this could lead to inference gains."
- Source 14: "These are the range of speedups for 4-bit palettized models on iPhone 14 Pro Max. The improvements
  vary between 5% to 30%."
- Source 8: "some compiler backends may choose to decompress the weights fully before runtime, leading to a
  latency identical to that of the float16 model. In other cases, decompression may happen “on the fly”" ...
  "Because the decompression strategy varies per hardware and compute unit, is highly recommended to test".

Verdict: **confirmed, with a qualifier the summary drops.** Apple says "in certain scenarios" and "could lead to"
gains; it does not say decompression is always just in time on the ANE. The speedup figures were measured on an
iPhone 14 Pro Max with 4-bit palettized models.

### 8. Apple: W8A8 faster int8 compute on A17 Pro and M4

DESIGN.md: "W8A8 uses faster int8 compute on A17 Pro and M4."

- Source 8: "8-bit activation plus weight quantization, also referred to as the W8A8 mode, can lead to
  considerable latency benefits on the Neural Engine by leveraging the faster int8-int8 compute path supported
  in newer hardware (A17 pro, M4)." This sits under "As of iOS18/macOS15, here are a few high level
  recommendations".
- Source 14: "in iOS 17, 8-bit activation quantized models can also be executed."

Verdict: **confirmed** for the positive examples (Neural Engine, A17 Pro and M4, as of iOS18/macOS15). The M1 Pro
remark in DESIGN.md ("Our M1 Pro has neither") is **unverified**. No listed source states that the M1 Pro lacks the
faster int8-int8 path or just-in-time decompression; the M1 Pro is merely absent from Apple's examples, and absence
from a list of examples is not evidence of absence. Treat it as an open assumption until measured, e.g. a W8A8 vs
FP16 probe on the Mac.

### 9. Shapes: per-shape compilation; flexible shapes on the ANE from iOS 17.4 with `reshapeFrequency = .infrequent`

DESIGN.md: "fixed and enumerated shapes are compiled per shape. Flexible shapes can run on the ANE from iOS 17.4
with `reshapeFrequency = .infrequent`. No shape choice guarantees ANE placement, because `cpuAndNeuralEngine`
allows CPU fallback."

- Source 10: "Setting the Reshape Frequency Optimization Hint to Infrequent can allow flexible shaped models to
  run on the Neural Engine, with iOS 17.4 or later." Code: `optimization_hints={"reshapeFrequency":
  ct.ReshapeFrequency.Infrequent}`.
- Source 10: "Use EnumeratedShapes for best performance. During compilation the model can be optimized on the
  device for the finite set of input shapes. You can provide up to 128 different shapes."
- Source 15 (transcript): "It then segments the chain of operations for specific compute devices based on the
  estimated performance and hardware availability. This segmentation is then cached." Also: "If the configuration
  was not found in the cache, it then triggers a device-specialized compilation for it."
- Source 11 (multifunction, for arm B): "Starting with iOS18/macOS15, you can produce an mlprogram with multiple
  functions in it." "During the process of merging, Core ML Tools deduplicates shared weights by calculating the
  hash of the weight values."

Verdict: **iOS 17.4 / Infrequent confirmed.** The ANE wording is "can allow", not "will". **"Compiled per shape" was
not found verbatim.** Source 10 says only that enumerated models "can be optimized on the device for the finite
set of input shapes". CPU fallback under `cpuAndNeuralEngine` is consistent with source 15's per-device
segmentation, but none of these sources states it in those words.

### 10. FluidAudio: decoder and joint on the CPU at 0.1-0.5 ms per call

DESIGN.md: "FluidAudio reports the decoder and joint running on the CPU with 0.1-0.5 ms per call."

- Source 5 `#L5-L7`: "**Measured** | 2026-06-05", "**Machine** | MacBook, Apple Silicon M5, macOS (Darwin 25.x)",
  "`computeUnits = .cpuAndNeuralEngine`".
- Source 5 `#L64-L71` (Parakeet TDT v3, "measured on real audio (7.8 s clip, production config, 5-run average)"):
  "| Decoder | 0% | 0% | 100% | 24 | 23 MB | 9 ms (40× @ 0.23) |" and "| Joint | 0% | 0% | 100% | 24 | 13 MB |
  23 ms (49× @ 0.46) |".
- Source 4 `#L55-L57`: "The decoder (LSTM prediction net) and joint network are called many times per window
  (~40 and ~60 respectively) and are dispatch-bound at ~0.1 ms and ~0.22 ms per call — GPU does not help them and
  would add per-call dispatch overhead, so they stay on ANE."
- Source 7 `README.md#L72-L73` (v2, host-side, CPU+NE, 15 s clip): "Joint: 28.34 ms → 22.66 ms" and "Decoder
  (U=1): 7.51 ms → 4.32 ms". These numbers are not per-call figures in the same sense.

Verdict: **partly confirmed; the sources differ.** The per-call range is 0.1-0.46 ms across sources 4 and 5,
both measured on M-series Macs with v3. Source 5 (`MLComputePlan`, M5) places the decoder and joint 100% on the
CPU at 0.23 / 0.46 ms. Source 4 gives ~0.1 / ~0.22 ms and says "they stay on ANE", which means the
compute-units setting, not a measured placement. Not measured on iPhone. DESIGN.md already treats this as
"to be confirmed".

### 11. Background: iOS 26 background GPU through continued-processing tasks, with an entitlement

DESIGN.md: "iOS 26 can allow background GPU work through continued-processing tasks, with an entitlement."

- Source 17a (WWDC25 227 transcript): "In iPadOS and iOS 26, your continued processing tasks can also benefit from
  background GPU access on supported devices. To take advantage of this, make sure you add the background GPU
  capability in your Xcode project settings."
- Source 17b: "# Background GPU Access / The entitlement the system requires for a continuous background task to
  use the GPU." Availability "iOS: 26.0.0 -"; "This entitlement works with ... BGContinuedProcessingTask".
- Source 17 (WWDC25 226) does **not** cover this. It is about Power Profiler (claim 12).

Verdict: **confirmed** ("on supported devices"), but by sources 17a and 17b, which were added. None of the 19
listed sources supports it. Not relevant to measurements (foreground only).

### 12. Supporting: on-device power measurement (used in Measurements, not in "Prior evidence")

- Source 16: "Power Profiler is available on iPhone with iOS 26 or later, and iPad with iPadOS 26 or later."
  "Alternatively, record a performance trace using Power Profiler on your device while you’re away from your desk".
  "Xcode keeps Apple silicon awake while it’s paired with your device, so Instruments only shows sleep and wake
  events when you collect a performance trace on a device that isn’t connected with Xcode."
- Source 17 (transcript): "The Power Profiler is also available on-device" and "Once Performance Trace is enabled,
  there’s the option to enable Power Profiler."

### 13. Supporting: FluidAudio "compute-bound" on M-series (Hypothesis section of DESIGN.md)

- Source 4 `#L80-L82`: "The encoder is compute-bound, not weight-bandwidth-bound, so fewer weight bits buy no
  speed." `#L31`: "LibriSpeech `test-clean`, 100 files, M-series Mac, 6-bit palettized encoder". `#L44-L48`:
  `.cpuAndGPU` 17.8 ms, `.cpuAndNeuralEngine` 23.5 ms, `.cpuOnly` 85.4 ms for one 15 s window.

Verdict: **confirmed** as FluidAudio's statement (M-series Mac, v3, comparing 6-bit with INT4). This is their
inference, not a bandwidth measurement.

### 14. Supporting: MLX affine quantization (arm E encoding)

- Source 18: "every group_size elements in a row of w are quantized together." Modes table: "affine | 32, 64 * ,
  128 | 2, 3, 4 * , 5, 6, 8 | same as input | yes" ("* indicates the default value when unspecified"). "To
  dequantize the elements of w , we also save \(s\) and \(\beta\) which are the returned scales and biases
  respectively."

Relevance: 2-bit affine with group sizes 32/64/128 and a per-group bias allows the explicit q = C + 1 with bias -s
construction (MLX 0.32.3 docs). The docs state the forward formula w_hat = round((w - beta)/s). The
dequantization w = s * w_hat + beta is implied by "we also save s and beta", but the page does not print it.

## C0 artifact

Pinned in `c0.json` (written by `c0.py pin`). Repository `FluidInference/parakeet-tdt-0.6b-v2-coreml` at revision `ee09c569f73759e6d44c9bd16766f477b2b36d39` (last modified 2025-09-25T22:47:39.000Z): https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v2-coreml/tree/ee09c569f73759e6d44c9bd16766f477b2b36d39.

Files are the ones FluidAudio 0.7.8 (tag `v0.7.8` -> `8136bd0642e7c5ce1f6f5b2931890266aeecb08c`) loads for v2. `ModelNames.ASR` lists `Preprocessor`, `Encoder`, `Decoder` and `JointDecision` `.mlmodelc` plus `parakeet_vocab.json` (https://github.com/FluidInference/FluidAudio/blob/8136bd0642e7c5ce1f6f5b2931890266aeecb08c/Sources/FluidAudio/ModelNames.swift). `DownloadUtils.downloadRepo` also fetches root `*.json`/`*.txt` (`config.json`) (https://github.com/FluidInference/FluidAudio/blob/8136bd0642e7c5ce1f6f5b2931890266aeecb08c/Sources/FluidAudio/DownloadUtils.swift). Not downloaded: `Melspectogram.mlmodelc`, `Melspectrogram_v2.mlmodelc`, `ParakeetDecoder.mlmodelc`, `ParakeetEncoder.mlmodelc`, `ParakeetEncoder_4bit_par.mlmodelc`, `ParakeetEncoder_v2.mlmodelc`, `RNNTJoint.mlmodelc`.

22 files, 464,413,250 bytes. SHA-256 is the HF LFS oid for LFS files, and computed after a git-blob-id check for the rest:

| File | Bytes | SHA-256 |
|---|---:|---|
| `Decoder.mlmodelc/analytics/coremldata.bin` | 243 | `46de1a6fe2e49d19a2125bc91acf020df7f2aea84ba821532aade8427a440b05` |
| `Decoder.mlmodelc/coremldata.bin` | 554 | `d200ca07694a347f6d02a3886a062ae839831e094e443222f2e48a14945966a8` |
| `Decoder.mlmodelc/metadata.json` | 3,427 | `90a279b822496316458febc0ce761ab05954fadd9d66aa97bea077a35fc8f2b2` |
| `Decoder.mlmodelc/model.mil` | 13,106 | `7b95a5a6b672c652000348a67b6d4d92bb8e176b978c6666fe73c28a4d7ec579` |
| `Decoder.mlmodelc/weights/weight.bin` | 14,429,952 | `27d26890221d82322c1092fd99d7b40578e435d5cf4b83c887c42603caf97aba` |
| `Encoder.mlmodelc/analytics/coremldata.bin` | 243 | `42e638870d73f26b332918a3496ce36793fbb413a81cbd3d16ba01328637a105` |
| `Encoder.mlmodelc/coremldata.bin` | 485 | `4def7aa848599ad0e17a8b9a982edcdbf33cf92e1f4b798de32e2ca0bc74b030` |
| `Encoder.mlmodelc/metadata.json` | 2,926 | `58222fbc48c13c49d9715567803cd50cb9c23e4360462e0f8ffcea59a2c73c63` |
| `Encoder.mlmodelc/model.mil` | 959,769 | `ed7b19156ca29fa7dfd6891deb9fda4b0e8893f68597c985d135736546a43808` |
| `Encoder.mlmodelc/weights/weight.bin` | 445,187,200 | `4adc7ad44f9d05e1bffeb2b06d3bb02861a5c7602dff63a6b494aed3bf8a6c3e` |
| `JointDecision.mlmodelc/analytics/coremldata.bin` | 243 | `f1183ba213bb94a918c8d2cad19ab045320618f97f6ca662245b3936d7b090f7` |
| `JointDecision.mlmodelc/coremldata.bin` | 534 | `e2c6752f1c8cf2d3f6f26ec93195c9bfa759ad59edf9f806696a138154f96f11` |
| `JointDecision.mlmodelc/metadata.json` | 2,936 | `ba8d309417b9acd4a175fdb15687de6a941db2f5b06666a60e7cf3cc8e2d3c3c` |
| `JointDecision.mlmodelc/model.mil` | 9,722 | `93bf82042235127cb81ab537dcae47a1c2e7e242ce4ffdaf772981b45eedc4f0` |
| `JointDecision.mlmodelc/weights/weight.bin` | 3,453,388 | `ca22a65903a05e64137677da608077578a8606090a598abf4875fa6199aaa19d` |
| `Preprocessor.mlmodelc/analytics/coremldata.bin` | 243 | `03ab3c1327a054c54c07a40325db967ec574f2c91dcc8192bfa44aa561bcf2d8` |
| `Preprocessor.mlmodelc/coremldata.bin` | 494 | `d88ea1fc349459c9e100d6a96688c5b29a1f0d865f544be103001724b986b6d6` |
| `Preprocessor.mlmodelc/metadata.json` | 2,974 | `fb16c581ff5e1b962e7cb2181ed892cd32f9f84c12b6e80ff3e089f28e35bcbb` |
| `Preprocessor.mlmodelc/model.mil` | 27,166 | `3e06d16fd061294c8a75be68c43a3b1ed1f593d4a9c35249e9cdbccadc59721e` |
| `Preprocessor.mlmodelc/weights/weight.bin` | 298,880 | `a5f7df6c7f47147ae9486fe18cc7792f9a44d093ec3c6a11e91ef2dc363c48dc` |
| `config.json` | 3 | `ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356` |
| `parakeet_vocab.json` | 18,762 | `57019fe3c745772ca83a1b048a4bb951cd51329504ea33d4d83316b96e279a97` |

Model I/O (from each `metadata.json`; full schemas in `c0.json`):

- **Decoder** (Parakeet decoder (RNNT prediction network); storage Float16; coremltools 9.0b1): in `targets` Int32 [1, 1], `target_length` Int32 [1], `h_in` Float32 [2, 1, 640], `c_in` Float32 [2, 1, 640]; out `decoder` Float32 [1, 640, 1], `h_out` Float32 [2, 1, 640], `c_out` Float32 [2, 1, 640].
- **Encoder** (enc6bit-palettize quantized - encoder; storage Mixed (Float16, Palettized (6 bits)); coremltools 9.0b1): in `mel` Float32 [1, 128, 1501], `mel_length` Int32 [1]; out `encoder` Float32 [1, 1024, 188], `encoder_length` Int32 [1].
- **JointDecision** (Parakeet single-step joint decision (current frame); storage Float16; coremltools 9.0b1): in `encoder_step` Float32 [1, 1024, 1], `decoder_step` Float32 [1, 640, 1]; out `token_id` Int32 [1, 1, 1], `token_prob` Float32 [1, 1, 1], `duration` Int32 [1, 1, 1].
- **Preprocessor** (int8-linear quantized - preprocessor; storage Int8; coremltools 9.0b1): in `audio_signal` Float32 1 × 1...240000, `audio_length` Int32 [1]; out `mel` Float32 [], `mel_length` Int32 [1].
