"""Per-iteration cost of the 1-bit decode variants (agent A): reshape decode vs elementwise decode."""
import json
import torch
from gpu1bit.triton_bf16.gpulock import locked
from gpu1bit.shared import env
from gpu1bit.triton_bf16.probe_loop import kernel_us
from gpu1bit.triton_bf16.layer import TritonBF16Layer

def main():
    m, n = 38, 1024
    res = {}
    with locked("A:probe-decode"), torch.inference_mode():
        for kd in (1024, 4096):
            codes = torch.randint(0, 256, (n, kd // 8), dtype=torch.uint8)
            x = torch.randn(1, m, kd, device="cuda")
            for mode in ("fp16x2p", "fp16x2", "bf16x3"):
                for ew in (0, 1):
                    for stages in (3, 4):
                        layer = TritonBF16Layer.prepare(codes, torch.rand(n), None, kd, "cuda", mode=mode,
                                                        configs={m: (16, 32, 64, 1, 4, stages, "serial", 1, ew)})
                        y = layer.linear(x)
                        ref = layer.linear(x)
                        t = kernel_us(lambda: layer.linear(x))
                        res[(mode, ew, stages, kd)] = max(t.values())
        for (mode, ew, stages, kd), v in res.items():
            if kd == 1024:
                per = (res[(mode, ew, stages, 4096)] - v) / 48 * 1000
                print(f"{mode:8s} ew={ew} s={stages}: K1024 {v:6.2f} us, per-iteration {per:6.1f} ns")

if __name__ == "__main__":
    main()
