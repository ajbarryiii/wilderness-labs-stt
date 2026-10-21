"""Fixed-cost probe (agent A): per-kernel graph-replay cost of an empty kernel, a 1-CTA store, and N
dependent launches; tells how much of a small-M op is launch/drain overhead. Screening only."""
import json, sys
import torch, triton, triton.language as tl
from gpu1bit.triton_bf16.gpulock import locked
from gpu1bit.shared import env
from gpu1bit.triton_bf16.bench import time_graph

@triton.jit
def _nop(Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + i, 1.0, i < N)

def main():
    out = {}
    with locked("A:overhead"), torch.inference_mode():
        y = torch.empty(1 << 20, device="cuda")
        for ctas in (1, 170, 340, 1360):
            out[f"store_kernel_{ctas}ctas_us"] = round(time_graph(lambda: _nop[(ctas,)](y, ctas * 128, 128))[0], 3)
        x = torch.randn(126, 1024, device="cuda")
        out["torch_empty_126x1024_us"] = round(time_graph(lambda: torch.empty(126, 1024, device="cuda"))[0], 3)
        json.dump(out, sys.stdout, indent=1); print()

if __name__ == "__main__":
    main()
