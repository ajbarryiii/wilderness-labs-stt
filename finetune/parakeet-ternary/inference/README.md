# CUDA runtime for ternary Parakeet-TDT-0.6B-v2

This directory runs the published ternary checkpoint
[rajb3/parakeet-tdt-0.6b-v2-ternary](https://huggingface.co/rajb3/parakeet-tdt-0.6b-v2-ternary)
on NVIDIA SM120 GPUs (target: RTX 5090) directly from its packed weights. Unlike the `load_ternary.py` loader in the
Hugging Face download, it does not rebuild a dense FP32 model, and it lives only in this
repository (MIT). All numbers come from one RTX 5090 at its configured 400 W power limit,
measuring warm batch-one transcription from a CPU waveform to CPU text. The evidence, including
p95 latency, every window, the GPU clocks and the hashes of the runtime sources, is in
[`results/rtx5090.json`](results/rtx5090.json).

## Results

| Clip | Configuration | Latency (ms) | Process VRAM (GB) | GPU energy (J) | Avg GPU power (W) |
| --- | --- | ---: | ---: | ---: | ---: |
| 3 s | Ternary expanded — optimized by Wilderness Labs | 2.73 | 1.74 | 0.76 | 279 |
| 3 s | Ternary compact — optimized by Wilderness Labs | 3.16 | 1.12 | 0.91 | 288 |
| 3 s | Original BF16 — optimized by Wilderness Labs | 4.25 | 2.11 | 1.15 | 271 |
| 3 s | ONNX ASR / ORT CUDA — off the shelf | 7.87 | 4.13 | 1.97 | 249 |
| 10 s | Ternary expanded — optimized by Wilderness Labs | 4.46 | 1.89 | 1.57 | 351 |
| 10 s | Ternary compact — optimized by Wilderness Labs | 5.27 | 1.27 | 1.86 | 352 |
| 10 s | Original BF16 — optimized by Wilderness Labs | 5.81 | 2.18 | 1.90 | 326 |
| 10 s | ONNX ASR / ORT CUDA — off the shelf | 15.74 | 4.14 | 3.75 | 237 |
| 30.04 s | Ternary expanded — optimized by Wilderness Labs | 10.56 | 2.23 | 3.99 | 376 |
| 30.04 s | Ternary compact — optimized by Wilderness Labs | 12.21 | 1.62 | 4.74 | 386 |
| 30.04 s | Original BF16 — optimized by Wilderness Labs | 11.56 | 2.35 | 4.48 | 386 |
| 30.04 s | ONNX ASR / ORT CUDA — off the shelf | 36.38 | 4.19 | 9.00 | 248 |

Compared with our optimized BF16 build of the original:

- **Expanded** is lower on latency, process VRAM and GPU energy at all three lengths. At 10 s it
  has 23.2% lower latency, 13.5% lower process VRAM and 17.1% less GPU energy.
- **Compact** has the lowest process VRAM of the four configurations at every length (41.8% less
  than BF16 at 10 s), and 9.3% lower latency at 10 s. Its 10 s energy is 2% lower, which is too
  small a difference to count as a gain. At 30.04 s it has 5.6% higher latency and uses 5.8% more
  energy than BF16.
- **Expanded vs compact** at 10 s: 15.3% lower latency and 15.4% less energy, at the cost of
  about 617 MB more process VRAM.

Expanded had the lowest latency and GPU energy of the four configurations at every length.

Compared with the off-the-shelf ONNX row at 10 s, expanded has 71.6% lower latency, 54.4% lower
process VRAM and 58.0% less GPU energy. Compact is 66.5%, 69.3% and 50.4% lower. Our BF16 build
of the original also has 63.1% lower latency than that row; the configurations differ in
precision and runtime.

## Two modes, one checkpoint

Both modes load the same trained export. In both, the 264 ternary encoder matrices run on INT8
Tensor Cores: activations enter those multiplies as three residual INT8 components, with int32
accumulation and FP32 outputs. This is not ordinary single-INT8 activation quantization. The
modes differ in how the weights are held:

- **Compact** (`--optimized`) keeps the matrices on the GPU as 2-bit codes (four per byte) with
  per-row FP32 scales, and unpacks the codes inside the kernels.
- **Expanded** (`--optimized --encoder-storage expanded`) also caches an exact INT8 copy of the
  same codes (603,979,776 bytes, about 604 MB), which the kernels read directly.

In both modes the non-ternary tensors stay in floating point (stored as FP16 in the export and
rebuilt as FP32), and encoder operations are fused. The prediction and joint networks run in FP32
through a fused decoder, and NeMo's greedy TDT loop runs as conditional CUDA graphs. Graphs are
cached per input shape, with a graph allocation fix.

## The comparison rows

**Original BF16 — optimized by Wilderness Labs** is NVIDIA's original checkpoint (revision
`ae9ad07059c7c739ffaf932226a8fe64ae2620b0`) with a BF16 encoder, plus our CUDA graphs, fused
FP32 decoder and graph allocation fix. It is not stock NeMo; we built it so that the ternary rows
are compared against an optimized original. The checkpoints still differ: the original's
non-encoder weights are FP32, while the ternary export stores its float tensors as FP16, and the
two emit different transcripts and therefore do different amounts of decoder work. This row
compares complete runtimes; it does not isolate ternarization or the encoder.

**ONNX ASR / ORT CUDA — off the shelf** is the unmodified
[onnx-asr](https://github.com/istupakov/onnx-asr) 0.12.0 package with onnxruntime-gpu 1.24.4.
It runs the published FP32 export
[istupakov/parakeet-tdt-0.6b-v2-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx)
at revision `0bbb45a3365852604aef28b538a8f066f4ccaa85` on the CUDA execution provider, with the
normal CPU fallback, default provider and allocator options, 4 intra-op CPU threads and 1
inter-op thread. It uses neither NeMo nor our kernels, and ran on the same GPU, at the same power
limit, with the same protocol and clips. Its weights are FP32, and ONNX Runtime's defaults leave
CUDA graphs off. It is one common way to run the model, and this is not a survey of optimized
engines: we did not benchmark
[sherpa-onnx](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/index.html),
which also supports this model, or other ONNX Runtime settings. These results do not establish
the best achievable off-the-shelf performance.

## Method

- **Timed path:** a CPU FP32 16 kHz waveform in, CPU text out, at batch one. Timing includes
  host-device transfers and feature extraction, and excludes file I/O, model loading,
  compilation and the first graph capture.
- **Workloads:** one real LibriSpeech test-clean clip per length: 3.0, 10.0 and 30.04 s (record
  IDs are in the JSON).
- **Runs:** each configuration ran in its own fresh process, one after another, never
  concurrently. Each length got three 5 s timed windows, each after a 2 s warm-up; reported
  values are medians. One process runs the lengths in order (3, 10, 30.04 s) and its graph cache
  holds up to four shapes, so process VRAM at the longer lengths includes graphs already
  captured for the shorter ones.
- **Process VRAM:** the highest sampled GPU memory of the warm process, in decimal GB. It
  includes the CUDA context, weights, graphs and workspace, so it is neither the weight-file size
  nor allocator statistics alone.
- **GPU energy:** NVML's GPU-board energy counter per transcription. It excludes CPU, host and
  wall-outlet energy.
- **Power:** the configured 400 W limit (board default 600 W) was not changed for any row. At
  10 s the ternary rows draw as much power as BF16 or more (expanded 351 W, compact 352 W, BF16
  326 W, ONNX 237 W). They use less energy because they finish sooner; this is not a
  lower-wattage result.

## Quality validation

- **Compact vs expanded:** text, tokens and timestamps were identical on 2,048 of 2,048
  utterances (256 from each of eight test sets, 3.9 h), with encoders at batch sizes up to 8 and
  the decoder at batch one. In a further 256 of 256 utterances run in fresh batch-one
  processes, the transcripts matched. The floating-point outputs are not bit-identical: the
  largest absolute encoder difference was 1.34e-6.
- **What this does not show:** agreement between the two modes is not timestamp accuracy, and it
  does not show that accuracy is unchanged from NVIDIA's model. Full-corpus accuracy for this
  checkpoint is in [`../results/TEST.md`](../results/TEST.md): the mean over seven leaderboard
  sets is 6.84%, against 6.45% for the FP32 original, and Common Voice is 12.56% against 8.50%.
- **Benchmark subset:** on the benchmark's 256-utterance quality subset (32 per set, 4,721
  reference words), both ternary modes made 263 word edits and the off-the-shelf ONNX original
  made 245. These are subset counts, not leaderboard WER.

## Usage

Run from the repository root. The input is 16 kHz mono audio.

```sh
P=finetune/parakeet-ternary
EXPORT=/mnt/hd/wilderness-labs-stt/parakeet-ternary/exports/parakeet-v2
mountpoint -q /mnt/hd || exit 1       # the project disk must be mounted
export HF_HOME=/mnt/hd/wilderness-labs-stt/parakeet-ternary/caches/huggingface   # any cache on the mounted disk
export HF_HUB_CACHE="$HF_HOME/hub"
hf download rajb3/parakeet-tdt-0.6b-v2-ternary --local-dir "$EXPORT"

$P/python -m inference --export "$EXPORT" --optimized clip.wav                              # compact
$P/python -m inference --export "$EXPORT" --optimized --encoder-storage expanded clip.wav   # expanded
```

`$P/python` is a wrapper pinned to this machine's NixOS runtime
(`custom/autoresearch/.training-runtime`). It finds the project's dependencies and caches on
`/mnt/hd` and locates the NVRTC library that the full conditional graphs need. To use another
prepared interpreter, set `PARAKEET_PYTHON=/path/to/python`; it must already have NeMo 3.0,
PyTorch 2.11 (cu128), Triton 3.6, cuda-python bindings (`cuda-bindings` 13.4.1), NVRTC 12.9 and the Hugging Face
dependencies. The versions behind the published results are recorded in the results JSON. There
is no installer.

From Python, `enable_optimizations(model, encoder_storage="packed")` (compact) or
`encoder_storage="expanded"` switches a loaded model to the optimized runtime in place.

- Disable the optimizations before editing weights, moving the model to another device or
  training it.
- Calls on one model are serialized. Autocast is not supported, and transcription is offline
  (whole utterances) only.
- Up to four input shapes are cached as CUDA graphs. The first call at a new eligible shape
  captures a graph, so warm up the lengths you need. No cold-start latency is claimed.

## Reproducing the benchmark and tests

```sh
OUT=/mnt/hd/wilderness-labs-stt/parakeet-ternary/comparison/new-run
for variant in packed_optimized packed_expanded nemo_bf16_graphs_fused_decoder onnx_cuda; do
    $P/python -m inference.compare_v2 --variant "$variant" --out "$OUT/$variant" --quality-per-set 32
done

PARAKEET_KERNEL_EXPORT="$EXPORT" $P/python -m unittest discover \
    -s finetune/parakeet-ternary/inference -t finetune/parakeet-ternary -p 'test_*.py'
```

`inference.compare_v2` runs one variant per invocation, so the loop runs the four published
configurations one after another, each in its own process: `packed_optimized` (compact),
`packed_expanded`, `nemo_bf16_graphs_fused_decoder` and `onnx_cuda`. It does not read `$EXPORT`; it uses fixed
paths under `/mnt/hd/wilderness-labs-stt/parakeet-ternary/`, which a fresh checkout must
prepare:

- the trained export at `runs/main-M1-P2-lr5e-4/export`;
- the hash-locked local copy of NVIDIA's checkpoint, for the BF16 row;
- the test-set manifests and audio from `../testsets.py`, for the clips and the quality subset;
- `comparison/deps` (the separately installed ONNX packages) and `comparison/onnx-v2` (the ONNX
  export at the pinned revision).

Run one GPU job at a time; the benchmark checks for the 400 W limit before it starts.

## Limitations

- Measured on one RTX 5090, warm, at batch one, with one clip per length up to 30.04 s. Other
  GPUs, larger batches, cold starts, longer inputs, streaming and host energy were not measured.
- The environment is specific to this machine, as described under Usage.
