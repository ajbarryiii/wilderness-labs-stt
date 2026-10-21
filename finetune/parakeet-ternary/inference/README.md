# Packed Parakeet inference on RTX 5090

This runtime executes the existing `parakeet-ternary-v1` export with packed
2-bit weights for all 264 quantized encoder layers. It supports linear layers
and kernel-1 convolutions, including NeMo's transposed input views. The base path
uses existing NeMo code for decoder, frontend, attention and normalization.
It requires the project's PyTorch 2.11 / Triton 3.6 runtime and SM120 GPU.

The optional optimized implementation adds residual INT8x3 encoder arithmetic,
fused operations, and FP32 decoder kernels while retaining NeMo's model and
greedy decoding algorithm. It now includes the third optimization pass;
see [EXPERIMENTS3.md](EXPERIMENTS3.md) for its experiments and validation, and
[COMPONENTS.md](COMPONENTS.md) for isolated encoder/decoder measurements.
The earlier second-pass ledger is in [EXPERIMENTS.md](EXPERIMENTS.md).
See [COMPARISON.md](COMPARISON.md) for isolated process VRAM, latency, energy,
and quality measurements against original Parakeet v2 in NeMo and ONNX Runtime,
including a graph-enhanced dense BF16 baseline.
The third pass reduces warmed batch-one latency by **29–43%** and GPU energy
by **19–27%** relative to the second pass on the tested 3/10/30-second clips.
The 10-second result is **5.27 ms, 1.89 J, 1.27 GB process VRAM**. All **2,048**
validation transcripts match the second-pass control exactly. These
are measured workload results, not a cold-start or full-corpus quality guarantee.
Decoder fusions also accelerate dense NVIDIA v2; they are not a ternary-only gain.

The fourth pass adds an optional expanded-weight encoder. It retains the trained
ternary values but caches them as column-major INT8, using about 604 MB more
weight storage for faster kernels. It also fuses convolution and residual
operations. See [EXPERIMENTS4.md](EXPERIMENTS4.md) for the weight/sign audit,
candidate screens, rejected experiments, and validation. The 400 W GPU power
limit is retained throughout these experiments.
On the tested 10-second clip, it reduces the encoder from 3.74 to 2.96 ms and
full transcription from **5.27 to 4.46 ms**, using **1.57 J** and **1.89 GB
process VRAM**. Across the three clips, end-to-end latency falls 13.5–15.3%
and GPU energy 15.4–16.1%, with about 617–619 MB additional process VRAM.
All 2,048 comparison transcripts, token sequences and timestamps match the
third-pass compact runtime; the 12 regression tests pass.

On our RTX 5090 at its 400 W cap, the opt-in hybrid CUDA/Triton ternary runtime (`--experimental-cuda`) had 3.43% lower median full-transcription latency and 3.01% lower GPU energy per transcription than the Expanded Triton runtime. Process VRAM was essentially unchanged. Text, token IDs and timestamps matched on 2,048 sampled utterances. [CUDA_IMPROVEMENTS.md](CUDA_IMPROVEMENTS.md) has the results and their limits. [CUDA_RESULTS.md](CUDA_RESULTS.md) covers the older hand-written CUDA and cuBLAS trials and the decoder token-cache ablation.

From the repository root:

```sh
P=finetune/parakeet-ternary
EXPORT=/mnt/hd/wilderness-labs-stt/parakeet-ternary/runs/main-M1-P2-lr5e-4/export
$P/python -m inference --export "$EXPORT" clip.wav
# Faster path for repeated audio shapes:
$P/python -m inference --export "$EXPORT" --optimized clip.wav
# Trade additional VRAM for the fourth-pass encoder:
$P/python -m inference --export "$EXPORT" --optimized --encoder-storage expanded clip.wav
```

Audio must be 16 kHz. The export contains the tokenizer and configuration; the
runtime needs no model download or original `.nemo` file. Load once and reuse:

```python
from inference import load_packed, enable_optimizations
from hf.load_ternary import transcribe

model = load_packed("/mnt/hd/wilderness-labs-stt/parakeet-ternary/runs/main-M1-P2-lr5e-4/export")
# Fused encoder/decoder plus exact-shape frontend/encoder graph cache.
enable_optimizations(model, max_graphs=4)
print(transcribe(model, ["clip.wav"], batch_size=1))
```

Use the `python` wrapper: it verifies `/mnt/hd` is mounted and places all
model/compiler caches and temporary files on that disk. The first encounter
with a new shape compiles kernels. Warm up the intended shapes before measuring
latency or energy. No weights are trained or requantized by this runtime.

The wrapper now exposes the existing NVRTC 12.9 wheel to CUDA-Python. This lets
NeMo use its full conditional GPU graph decoder instead of falling back to
host-controlled loops. No dependency is downloaded. Outside this wrapper,
ensure CUDA-Python can find NVRTC; otherwise NeMo may use the slower fallback.

The optional graph cache stores the first four exact shape/stream combinations;
additional shapes run eagerly. It copies current features and lengths on every
call and returns owned output tensors. No audio bucketing or padding is added.
Capture needs an initial warm-up, so it benefits repeated shapes rather than
single cold calls. CLI equivalent: `--encoder-graphs`. Enable it after moving
the model to its final device; call `disable_encoder_graphs(model)` before
changing weights or device. Serialize calls per model across host threads;
concurrent calls on the same model are unsupported. Graph pools consume
additional activation memory.

`enable_optimizations` uses a graph around preprocessing plus the encoder;
`enable_encoder_graphs` remains available for the original encoder-only cache.
These caches are mutually exclusive. The optimized path requires offline FP32
inference without autocast, with frontend `dither=0` and `pad_to=0` for graph
reuse (`transcribe` sets these). Other frontend settings run eagerly. Streaming
encoder calls are unsupported by the optimized transforms. Pass `max_graphs=0`
to use the fusions eagerly. Call `disable_optimizations(model)` before changing
weights/device, saving a state dict, or training; it restores the original
modules and releases graphs. Load the normal export before re-enabling.

The optimized arithmetic keeps the two-bit weights and represents each input
as three scaled INT8 residual vectors. Integer products accumulate in INT32,
then combine with FP32 scales and outputs. Fused normalization, packed Q/K/V,
SiLU, relative attention, BatchNorm folding and fixed positional projections
reduce work and launches. This is a high-accuracy approximation of the original
floating activations, not ordinary single-pass INT8 inference. The code remains
inference-only and targets finite speech-model activations.

Pass `encoder_storage="expanded"` to `enable_optimizations` for the faster
column-major path. Expansion is exact and happens once during warmup. The cache
remains owned by the optimized model until disable; it is not an exported model
format or a further quantization step. The original packed weights remain
available for restoration. Tile tuning targets this 5090 and the tested speech
lengths; other workloads can have different tradeoffs.

The optimized decoder keeps FP32 weights, with a precomputed first-layer token
projection and fused recurrent/joint operations. Batch-one control kernels fuse
argmax, time advancement and hypothesis storage inside NeMo's conditional graph.
Larger batches and unsupported recording modes use the original control stages.
Conditional-body memory pools are retained for graph safety and released on
disable. `decoder=False, position_dot=False` selects second-pass arithmetic
while keeping the allocation-lifetime correction.

## Kernel design

At load time, four export bytes become an int32 word and the packed matrix is
transposed to `[ceil(K/16), N]`. This is a layout change, with exactly the same
`00=0`, `01=+1`, `10=-1` codes. There is one resident packed orientation. Each
kernel in the default packed path decodes only its active tile; no dense weight matrix is allocated on the
GPU. Scales and optional biases remain FP32.

- For 1–4 input rows, a SIMT reduction multiplies FP32 activations by exact
  ternary coefficients. Long reductions split across GPU blocks.
- Larger inputs use tiled Tensor Core multiplication. The default decomposes
  each FP32 activation into three BF16 components **in registers**, then computes
  their products with the exact ternary coefficients. High and residual sums
  accumulate separately in FP32, avoiding the loss of small contributions when
  repeatedly adding them to a large accumulator. Output scaling happens once.
- Shape-dependent tiles and split-K counts were selected offline on this 5090.
  Split-K uses an FP32 workspace and deterministic reduction, without atomics.
  There is no inference-time autotuning or CPU synchronization in the kernels.
- Pointwise convolutions read `[B,C,T]` strides and write that layout directly,
  avoiding im2col or transpose copies. Linear inputs with more than three
  dimensions may require a reshape copy if their layout cannot be flattened.

`mode="tf32x3"` supplies an alternative high-accuracy Tensor Core path for
comparison. The normal module/loader interface uses FP32 inputs and outputs;
`matmul(..., mode="fp16")` is an explicit low-level option requiring FP16 inputs.
It keeps FP32 row scales, so its reference is the FP32-weight result rounded to
FP16, not a matmul with weights independently rounded to FP16.

These operations approximate the FP32 computation; they are not bit-identical
to cuBLAS. Reduction order and moving the common row scale outside the sum also
change rounding. This is not ordinary BF16 activation inference: the residual
components are retained. The supported numerical domain is finite speech-model
activations; NaN, infinity, and preservation of subnormal values are not promised.

`PackedLinear` and `PackedPointwiseConv1d` hold buffers rather than trainable
weights. Use `torch.inference_mode()`. Calling `.half()` on the model is not
supported: scales must stay FP32. Gradients through the kernels are unsupported
and gradient-bearing inputs are rejected. CPU model execution is unsupported.

## Validation and reproduction

```sh
P=finetune/parakeet-ternary
$P/python -m unittest inference.test_kernels -v
$P/python -m unittest inference.test_experiments inference.test_bitwise -v
PARAKEET_KERNEL_EXPORT="$EXPORT" $P/python -m unittest inference.test_decoder -v
$P/python -m inference.benchmark micro
$P/python -m inference.benchmark tune --rows 16 64 128 384
$P/python -m inference.benchmark model
$P/python -m inference.benchmark2
$P/python -m inference.quality2 --variants baseline optimized --utterances-per-set 256
$P/python -m inference.quality3 --position-dot --utterances-per-set 256
```

The numerical suite covers zero rows, cancellation, wide activation ranges,
ragged dimensions, biases, convolution and linear strides, empty inputs, output
reuse, inference-only enforcement, alternate streams, and CUDA graph replay
with changed inputs. The model benchmark compares the same exported weights
against `export.load_export`, using strict FP32 cuBLAS/cuDNN settings for that
control. It samples 64 utterances evenly from each of eight test sets.

The benchmark writes source hashes, export hash, configuration, raw results,
transcripts, and telemetry to a new directory under
`/mnt/hd/wilderness-labs-stt/parakeet-ternary/kernel-runs/`. It takes the project's
GPU job lock and uses the existing `custom/inference-efficiency/energy.py` meter
to reject competing compute processes. Energy measurement requires
`nvidia-ml-py`; on this machine the benchmark reuses that dependency from the
existing inference-efficiency installation on `/mnt/hd`.

Microbenchmarks repeatedly use a single weight matrix, which can stay in L2;
they establish kernel latency, not model energy. The model benchmark separately
measures CUDA-graph encoder inference (mel features already on the GPU) and
eager preprocessing + encoder + greedy TDT decoding (audio already on the GPU).
It also measures full transcription with the encoder graph cache enabled for
**both** variants. Full transcription includes text construction. None includes loading, compilation,
audio disk I/O, or host energy. The shared desktop remains active. Three
alternating 10-second windows per variant follow a two-second warm-up each.
NVML cumulative board-energy counters provide joules per completed inference.
No power limits or clocks are changed.

These comparisons are against the repository's existing strict-FP32 runtime.
They do not establish a win over TensorRT, an FP16/BF16 deployment, or every
possible optimized dense implementation. Subset transcript agreement does not
replace full-corpus accuracy evaluation or field validation.

See [MEASUREMENTS.md](MEASUREMENTS.md) for the measured results and iteration record.

The isolated `bitwise.py` / `bitwise_benchmark.py` experiments compare FP32
lookup and SIMT paths without BF16 or activation quantization. They are not
selected by model inference: the tested encoder lookup path was slower and
used more energy than packed Tensor Core arithmetic. Fully quantized activation
popcount inference has not been accuracy-validated for this Parakeet model.
