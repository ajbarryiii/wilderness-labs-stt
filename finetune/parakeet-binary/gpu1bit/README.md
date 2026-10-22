# gpu1bit: native 1-bit GPU research runtime for Kea-B (and the Kea-T x2 loader)

**Status: research runtime.** This is the code that produced the measurements recorded in
[RESULTS.md](RESULTS.md). It is published for reference and is not packaged for use outside the
machine it was developed on. It depends on files and paths that exist only there; see "Portability
limits" below. To run the models elsewhere, use the standalone loaders published with them.

`gpu1bit` runs the binary-weight Parakeet MB1 (published as
[rajb3/kea-b](https://huggingface.co/rajb3/kea-b)) on the RTX 5090 straight from its 1-bit export:
- the 264 quantized layers stay resident at 1 bit per weight (75,497,472 B of codes);
- the projections run on custom INT8 tensor-core kernels (`int8_mma`) that expand the ±1 weights in
  registers;
- the stack around them is the ternary runtime's hybrid pipeline, with CUDA graphs and NeMo's fused
  greedy TDT decoder.

Activations are split into **two INT8 parts (x2) by default**. Before 2026-10-22 they were split into
three (x3). On full dev, x2 recognises like x3 and uses less energy.

The ternary sibling M1 ([rajb3/kea-t](https://huggingface.co/rajb3/kea-t)) has the same x2 default.
It runs through `gpu1bit.runtime_m1` on its own expanded runtime.

Plans: [../plans/gpu-kernel.md](../plans/gpu-kernel.md) and
[../plans/x2-default.md](../plans/x2-default.md).

## Portability limits

The code has run only on the development machine, and it hard-codes that machine's layout:
- **x2 part files loaded from `/mnt/hd`.** MB1's x2 kernel files (`kernels_parts.py`, `layer_parts.py`,
  `parts_run.py`) are included only as reference copies (below). `gpu1bit/actprec/parts.py` loads them, SHA-256-pinned,
  from `/mnt/hd/wilderness-labs-stt/parakeet-binary/gpu1bit/B/wip/parts/`. MB1 at x3 and M1 at x2 or x3
  do not use them.
  - **Reference copy:** [`measured-x2-parts/`](measured-x2-parts/) holds the exact bytes the measured
    MB1-x2 runs loaded: all nine files of that directory plus `parts-sha256.txt`. The hashes are the
    ones pinned in `actprec/parts.py`. To verify, run `sha256sum -c parts-sha256.txt` inside it.
  - Reference copies are included, but the runtime loader still reads the pinned `/mnt/hd` files.
- **`/mnt/hd` mount checks.** Both interpreter wrappers (`gpu1bit/shared/python`, which runs
  `finetune/parakeet-ternary/python`) refuse to start without the mount, and so does
  `int8_mma/cuda.py`. The wrappers also put caches and temporary files there.
- **Fixed NVRTC and cubin paths.** `int8_mma/cuda.py` loads NVRTC from
  `/mnt/hd/wilderness-labs-stt/inference-efficiency/deps/nvidia/cuda_nvrtc/lib/libnvrtc.so.12`. It
  caches compiled cubins under `/mnt/hd/wilderness-labs-stt/parakeet-binary/gpu1bit/B/caches/`.
- **The GPU lock.** The `gpu1bit.runtime` CLI holds the machine's GPU lock,
  `/mnt/hd/wilderness-labs-stt/parakeet-ternary/gpu.lock` (`paths.gpu_lock`), for its whole run. On
  that machine, Python callers were required to hold it themselves.
- **The interpreter** behind the wrapper, `custom/autoresearch/.training-runtime/bin/python3`, is not
  in the repository. `PARAKEET_PYTHON` overrides it.

## Measurement environment

- **GPU:** one GeForce RTX 5090 (32 GB), driver 595.71.05, 400 W power limit.
- **Device check:** both loaders refuse any device whose compute capability is not 12.0 (SM120),
  raising `ValueError: this runtime targets NVIDIA SM120 (RTX 5090)`.
- **Interpreter:** the one `gpu1bit/shared/python` runs. Versions were read from it on 2026-10-22:

| Package | Version |
| --- | --- |
| Python | 3.13.13 |
| torch | 2.11.0+cu128 (CUDA 12.8) |
| triton | 3.6.0 |
| nemo_toolkit | 3.0.0 |
| cuda-bindings | 13.4.1 |
| safetensors | 0.8.0 |
| omegaconf | 2.3.0 |
| numpy | 2.4.4 |
| soundfile | 0.13.1 |
| NVRTC | 12.9 (`nvidia-cuda-nvrtc-cu12` 12.9.86 wheel, on the library path) |

NVRTC compiles the int8_mma kernels at run time, and NeMo's full CUDA-graph decoder also uses it.

## Exports

The recorded runs loaded the exports from the development machine's run directories:
- MB1: `/mnt/hd/wilderness-labs-stt/parakeet-binary/runs/main-binB-m1-lr2e-4/export`;
- M1: `/mnt/hd/wilderness-labs-stt/parakeet-ternary/runs/main-M1-P2-lr5e-4/export`.

The published repositories hold the same export files plus their model cards and loaders.

**The export checks accept the downloaded layouts** (`hf download rajb3/kea-b`, `hf download
rajb3/kea-t`). This was checked on the CPU, with copies of the export files plus the extra files of
each repository:
- the manifest's format;
- the SHA-256 of every file the manifest lists;
- the 264 quantized layers;
- for MB1, every layer's packed codes and scales.

Extra files (model card, `LICENSE`, `NOTICE`, the standalone loader, `.gitattributes`, `.cache/`) are
ignored, and a ternary export given to the binary loader is refused, and the reverse too. Passing the
checks says nothing about running elsewhere; the limits above still apply.

## How the measurements were run

Everything ran from `finetune/parakeet-binary/` through the wrapper `gpu1bit/shared/python`, which puts
`gpu1bit` and the ternary project's modules on the import path. `$MB1` and `$M1` below are the export
directories above.

**MB1, `gpu1bit.runtime`.** The CLI as used for spot transcriptions:

```sh
gpu1bit/shared/python -m gpu1bit.runtime --export $MB1 clip.wav            # x2 (the selection's default)
gpu1bit/shared/python -m gpu1bit.runtime --export $MB1 --parts 3 clip.wav  # x3
gpu1bit/shared/python -m gpu1bit.runtime --export $MB1 --tokens a.wav b.wav
```

- `--tokens` prints JSON lines with the text, token IDs, frame timestamps, `parts`, the arm identity
  and the selection document's SHA-256.
- `--parts {2,3}` sets the activation parts.
- `--selection PATH` names the selection document. The default is `$GPU1BIT_SELECTION`, else
  `gpu1bit/shared/selected.json`.
- `--runtime`, `--kernel` and `--label` were for experiments and lock labels.
- A load or transcription error exits with status 1.

The gates and the benchmark built the B-hybrid arms through this loader, as
`load_binary_packed(MB1, runtime="hybrid", parts=n)`, with `GPU1BIT_SELECTION` set to
`shared/selected-x2.json`. The equivalent calls under the promoted pointer:

```python
from gpu1bit.runtime import load_binary_packed, transcribe, close_binary_packed
model = load_binary_packed(MB1)              # as b1 / g1: x2, int8_mma:x2 layers, HybridProjectionX2
model = load_binary_packed(MB1, parts=3)     # as b2 / g3: x3, int8_mma layers, hybrid_exact stack
text = transcribe(model, "clip.wav")         # batch 1 under torch.inference_mode()
close_binary_packed(model)
```

A call that names a kernel gets that kernel's own part count: `kernel="int8_mma"` gives x3, and
`kernel="int8_mma:x2"` gives x2. A kernel at another part count is refused.

**M1, `gpu1bit.runtime_m1`.** There is no command-line module.
- The gate and benchmark arms b4 / g4 (x2) and b5 / g5 (x3) were built by `runtime_m1.build_stack(M1, n)`.
  That is the stack `load_ternary_packed` wraps: `inference.runtime.load_packed`, then
  `enable_optimizations(encoder_storage="expanded")`, with the provider wrapped in
  `PartCount(provider, 2)` at x2.
- The deployment entry point:

```python
from gpu1bit.runtime_m1 import load_ternary_packed, transcribe, close_ternary_packed
model = load_ternary_packed(M1)              # x2 (m1_default_parts of the selection in use)
model = load_ternary_packed(M1, parts=3)     # x3
```

`loaded(...)` is its context-manager form.

**Selection document.** The part-count default comes from the document in use: `default_parts` for
MB1 and `m1_default_parts` for M1. Each is 3 if absent.
- `shared/selected.json` has been the x2 document since 2026-10-22.
- The previous x3 document (SHA-256 103d96f4…) is in git history.
- Identities, the hashes that records are bound to, hash the document's content; see RESULTS.md.

## x2 at a glance (RTX 5090, batch 1, dev)

| | MB1-x2 | M1-x2 |
| --- | ---: | ---: |
| Dev mean WER, x2 (x3) | 6.3122% (6.3120%) | 6.0853% (6.0856%) |
| Energy per full transcription, x2 vs x3, same stack | −11.4% | −10.1% |
| Latency, x2 vs x3, same stack | −6.8% | −3.0% |
| Peak process VRAM, fresh process, 3 / 10 / 30 s dev clips | 1.12 / 1.16 / 1.50 GB | 1.74 / 1.89 / 2.23 GB |

- **Energy and latency** are paired medians over 24 dev utterances. x2 was lower on all 24 in all four
  cells.
- **MB1-x2 against M1-x2:** −10.3% energy, −10.5% latency.
- **MB1-x2 against M1's previously deployed hybrid x3 runtime:** −16.6% energy, −10.5% latency.
- **Test WER was not re-scored for x2, by decision.** The published test means (MB1 7.05%, M1 6.84%)
  were scored on the FP32 NeMo reconstructions of the exports, not on this runtime.
- **More in [RESULTS.md](RESULTS.md), "INT8x2 deployment default":** full tables, the historical test
  clips and a cross-session comparison with full-precision NeMo.
