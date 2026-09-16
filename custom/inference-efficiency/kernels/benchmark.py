"""Short fixed-shape kernel diagnostics, separate from the energy decision.

Repeated weights can reside in L2. These timings do not predict whole-model
energy and are not valid decision measurements on a shared/busy GPU.
"""

import argparse
import json
import math
import statistics

import torch

from .packed import PackedWeight


def graph_latency_ms(operation, repeats: int = 32) -> float:
    """CUDA-event median per call, with Python dispatch removed using a graph."""
    for _ in range(3):
        operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            operation()
    samples = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(3):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / repeats)
    return statistics.median(samples)


def run_case(m: int, n: int, k: int, bits: int) -> dict:
    """Identical weights/activations; preparation and compilation excluded."""
    codes = (torch.randint(0, 2, (n, k), device="cuda", dtype=torch.int8) * 2 - 1
             if bits == 1 else torch.randint(-1, 2, (n, k), device="cuda", dtype=torch.int8))
    packed = PackedWeight.from_codes(codes, bits=bits, scale=1 / math.sqrt(k))
    dense = packed.dense()
    del codes
    x = torch.randn((m, k), device="cuda", dtype=torch.float16)
    out = torch.empty((m, n), device="cuda", dtype=torch.float16)
    expected = torch.mm(x, dense.T)
    torch.testing.assert_close(packed.linear(x, out=out), expected, atol=0.003, rtol=0.003)
    packed_ms = graph_latency_ms(lambda: packed.linear(x, out=out))
    dense_ms = graph_latency_ms(lambda: torch.mm(x, dense.T, out=out))
    return {"m": m, "n": n, "k": k, "bits": bits,
            "packed_ms": packed_ms, "dense_fp16_ms": dense_ms,
            "dense_over_packed_latency": dense_ms / packed_ms,
            "packed_bytes_including_scales": packed.storage_bytes,
            "dense_fp16_bytes": dense.numel() * dense.element_size()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", nargs="+", type=int, choices=[1, 2], default=[1, 2])
    args = parser.parse_args()
    torch.manual_seed(972)
    rows = []
    for bits in args.bits:
        for n, k in ((1024, 1024), (4096, 1024), (1024, 4096), (51864, 1024)):
            for m in ((1,) if n == 51864 else (1, 1500)):
                rows.append(run_case(m, n, k, bits))
    print(json.dumps({"diagnostic_only": True,
                      "warning": "Repeated weights may fit in L2; no energy or investment conclusion.",
                      "gpu": torch.cuda.get_device_name(), "rows": rows}, indent=2))


if __name__ == "__main__":
    with torch.inference_mode():
        main()
