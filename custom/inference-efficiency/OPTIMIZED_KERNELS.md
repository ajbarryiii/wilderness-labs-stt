# Optional RTX 5090 inference kernels

`kernels.optimized_runtime.install` enables an explicitly selected kernel policy
for the packed Whisper runtime. Default kernels remain unchanged so previous
benchmarks are reproducible. A confirmation receipt is now configured for both
models after three independent baseline/candidate pairs each. Receipts bind
the selected policy to the measured source hashes.

Run with `./inference-efficiency/python` so dependencies and caches use the
project runtime:

```python
import numpy as np
import torch
from benchmark import workload_manifest
from kernels.optimized_runtime import install
from runtime_model import ReplayWhisper

workload = workload_manifest()  # Verifies the prepared clips and token schedule.
mel = torch.from_numpy(np.load(workload["clips"][0]["mel"], allow_pickle=False))
mel = mel.to("cuda", dtype=torch.float16)
forced_tokens = workload["replay_tokens"]
distribution = "ternary"  # or "binary"
with install(distribution) as selected:
    model = ReplayWhisper(distribution=distribution, implementation="packed")
    model.capture(mel, forced_tokens)
    logits = model.replay_graph(mel)
    torch.cuda.synchronize()
    print(selected.metadata["confirmed"])
```

Keep the context around **model creation, graph capture, and every inference**.
Exiting restores Python dispatch. Captured CUDA graphs retain their kernels;
recreate or recapture the model when switching policies, including a return to
the baseline. Only one installation may be active in a process. Compilation
caches default to `/mnt/hd/wilderness-labs-stt/inference-efficiency/optimized-runtime/`;
the mounted data disk is required. An explicit `plugins=[...]` override is
experimental and is recorded as unconfirmed.

The independently confirmed policies are:

| Distribution | Plugins, in installation order |
| --- | --- |
| Ternary | `kernels.sprint_cuda:ownedword`, `kernels.sprint_fused_vector:qkv_fusion_owned4` |
| Binary | `kernels.sprint_cuda:hybrid`, `kernels.sprint_fused_vector:qkv_fusion_hybrid` |

These combine vectorized activation loads with fused decoder GEMV epilogues.
Ternary uses direct reads of each lane's packed weight words. Binary uses paired
loads at K=1024 and four-value loads at K=4096. GELU and residual addition execute
inside the relevant projection kernels, while QKV projection writes K/V directly
to the original cache slice. Misaligned activation views retain scalar loads.

Weights, architecture, token schedule, attention, cache layout, and cache update
schedule remain the same. Activations and outputs remain FP16, with FP32 accumulation. Each fused
linear result is explicitly rounded to FP16 **before** GELU or residual addition.
Vectorization can reorder FP32 sums; full-model checks therefore allow at most
0.5% normalized RMS error and 2% normalized maximum error, and record prediction
agreement separately. These tolerances are numerical checks, not an accuracy
claim for the random binary/ternary models.

Measurements use the architecture-matched 762-million-parameter Whisper medium.en
runtime, 16 fixed 30-second clips, and 128 forced decoder outputs on an RTX 5090
at a 400 W power limit. Energy covers the GPU board, excluding host energy.
Paired comparisons measure improvements over the previous packed implementation;
they do not establish equivalent gains over a newly optimized dense runtime,
CTranslate2, or another device. Read the
[measurement ledger](/mnt/hd/wilderness-labs-stt/inference-efficiency/agent-kernel-sprint/20260912T160024Z/MEASUREMENTS.md)
and [final comparison](/mnt/hd/wilderness-labs-stt/inference-efficiency/agent-kernel-sprint/20260912T160024Z/final-comparison.json)
for screening versus independently confirmed results.

Final confirmation found 15.8% lower ternary GPU energy and 13.6% lower binary
GPU energy, with mean window-p95 latency reductions of 17.5% and 14.4%.
Average board watts increased 2.3% and 1.0%, respectively. All full-model
numerical checks and the configured-receipt CPU installer checks passed.
The additional installer GPU integration check is queued behind a separate
popcount benchmark using the shared GPU lock. Its result will be written to
`/mnt/hd/wilderness-labs-stt/inference-efficiency/agent-kernel-sprint/20260912T160024Z/installer-smoke.json`;
until that file reports success, that additional check remains pending.
