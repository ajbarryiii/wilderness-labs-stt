# Inference efficiency experiment

Test whether packed binary/ternary Whisper inference on the RTX 5090 merits an
accuracy experiment. The decision requires at least **2× lower GPU energy per
identical workload, with no worse p95 latency**, against the better optimized
control. Random models do not establish recognition quality or phone efficiency.

The six variants are pretrained Whisper `medium.en` in CTranslate2 FP16 and
INT8/FP16, plus seeded binary and ternary models using either dense FP16 or
packed weights. Candidates share the Whisper graph, tied embedding/output
weights, SDPA attention, static KV caches, and CUDA graph replay. Their dense
twins contain identical numerical weights and isolate packed-kernel effects.
CTranslate2 is a separate optimized external control; the candidate kernels
are integrated into PyTorch rather than CTranslate2. This runtime difference
remains explicit in every report.

Each workload processes one of 16 frozen 30-second speech clips, runs the full
encoder, and replays exactly 128 decoder steps using the same predetermined
token stream. Frontend computation, transfers, scaling, and completed GPU work
are timed. Model loading, packing, compilation, and graph capture occur first.

From `custom/`, prepare and validate:

```sh
./inference-efficiency/setup
./inference-efficiency/python inference-efficiency/prepare.py models
./inference-efficiency/python inference-efficiency/prepare.py data
./inference-efficiency/python -m unittest discover -s inference-efficiency -p 'test_*.py'
./inference-efficiency/python inference-efficiency/runtime_checks.py --cuda
./inference-efficiency/python -m kernels.verify --full-shapes
./inference-efficiency/python inference-efficiency/benchmark.py smoke
```

Data preparation defaults to the existing LibriSpeech `train-clean-100` data
on `/mnt/hd`; use `prepare.py data --source /mnt/hd/PATH` for another local
LibriSpeech directory. Model revisions, package versions, speech provenance,
and artifact hashes are recorded. Benchmark workers use the prepared files
offline. `setup` uses this machine's existing `autoresearch/.training-runtime`
Python/PyTorch installation and installs pinned supplementary dependencies.

Run the complete experiment when the GPU has no other compute processes:

```sh
./inference-efficiency/python inference-efficiency/benchmark.py run
```

Defaults run all six variants in rotating order, with five windows per variant,
at least 60 seconds and a full 16-clip pass per window. CUDA graphs are enabled
for both candidate implementations. The GPU guard refuses other compute PIDs
before or during measurement. Smoke checks and kernel timings are not energy
results.

The meter prefers NVML cumulative energy and otherwise integrates timestamped
power samples. Reports include GPU joules/audio-second, total and incremental
energy, average watts, p50/p95 latency, real-time factor, memory, GPU state, and
window variation. CPU energy is excluded. The conservative decision compares
the candidate's highest-energy window with the selected control's lowest.
It also requires a measurable improvement over the identical dense candidate;
an external-control win caused only by runtime differences is labelled separately.

All downloads, weights, caches, prepared audio, and outputs live under
`/mnt/hd/wilderness-labs-stt/inference-efficiency/`; the wrapper refuses to run
if `/mnt/hd` is unmounted. Each measurement run writes a new `runs/` directory with
frozen `config.json`, workload manifest, raw per-window JSON, `summary.json`,
and `REPORT.md`. Smoke runs write explicitly unqualified smoke records.
No model files belong in Git.

Implementation limits are recorded in model metadata: activations/output are
FP16, accumulation is FP32, and per-row scales are stored FP32 but rounded to
FP16 on access. Learned biases and normalization vectors use random codes in
FP16 storage; fixed sinusoidal positions retain their normal values. Packed
convolutions include explicit im2col costs. The installed CTranslate2 4.8.2
wheel rejects Flash Attention 2 on SM120, so the control uses its supported
attention fallback. See [kernel details](kernels/README.md) and
[existing-kernel research](RESEARCH.md).

The agent-led RTX 5090 sprint adds vectorized CUDA GEMV and fused decoder
operations. See [optional installation](OPTIMIZED_KERNELS.md),
[tested avenues](SPRINT_AVENUES.md), [target research](SM120_SPRINT_RESEARCH.md),
and the [independent measurement audit](SPRINT_AUDIT.md). The
[per-step measurement ledger](/mnt/hd/wilderness-labs-stt/inference-efficiency/agent-kernel-sprint/20260912T160024Z/MEASUREMENTS.md)
reports fresh paired energy, watts, latency, and throughput measurements.
Its gains compare with the previous packed runtime; this sprint does not update
the six-way dense/CTranslate2 comparison. The optional installer requires a
source-bound confirmation receipt. The standard benchmark command above retains
the original kernels for reproducibility.
