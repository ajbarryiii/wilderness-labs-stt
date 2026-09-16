"""Numeric/packing checks; optional short timing diagnostics, never power claims.

Run from inference-efficiency: ./python -m kernels.verify
"""

import argparse
import json
import math

import torch
import torch.nn.functional as F

from .packed import PackedWeight


def check_close(actual, expected, name):
    torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.003, msg=name)
    return float((actual.float() - expected.float()).abs().max()) if actual.numel() else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", action="store_true",
                        help="Also report short CUDA-event timings (not energy results).")
    parser.add_argument("--full-shapes", action="store_true",
                        help="Also check full 1500-frame encoder and 51864-token projection.")
    args = parser.parse_args()
    torch.manual_seed(971)
    device = "cuda"
    results = []
    for bits in (1, 2):
        # Ragged sizes exercise word padding, output masks, and K-tile masks.
        shapes = [(19, 73), (1024, 1024), (4096, 1024), (1024, 4096)]
        if args.full_shapes:
            shapes.append((51864, 1024))
        for n, k in shapes:
            codes = (torch.randint(0, 2, (n, k), dtype=torch.int8) * 2 - 1
                     if bits == 1 else torch.randint(-1, 2, (n, k), dtype=torch.int8))
            scales = torch.linspace(0.5, 1.5, n) / math.sqrt(k)
            cpu = PackedWeight.from_codes(codes, bits=bits, scale=scales)
            packed = PackedWeight.from_codes(codes.to(device), bits=bits,
                                             scale=scales.to(device))
            # Unused final-word bits are unspecified; compare all logical values.
            dense = (codes.half() * scales.half()[:, None]).to(device)
            torch.testing.assert_close(cpu.dense().to(device), dense, atol=0, rtol=0)
            torch.testing.assert_close(packed.dense(), dense, atol=0, rtol=0)
            bias = torch.randn(n, device=device, dtype=torch.float16) * 0.1
            ids = torch.tensor([[0, n - 1], [n // 2, 1]], device=device)
            check_close(packed.embedding(ids), F.embedding(ids, dense), "embedding")
            rows = [1] if n == 51864 else [1, 3, 7, 33]
            if args.full_shapes and n != 51864:
                rows.append(1500)
            for m in rows:
                # Noncontiguous input is deliberately accepted; its copy is timed.
                x = torch.randn(k, m, device=device, dtype=torch.float16).T
                for with_bias in (False, True):
                    b = bias if with_bias else None
                    out = torch.empty((m, n), dtype=torch.float16, device=device)
                    y = packed.linear(x, b, out=out)
                    assert y is out
                    error = check_close(y, F.linear(x, dense, b), "linear")
                    results.append({"bits": bits, "m": m, "n": n, "k": k,
                                    "bias": with_bias, "max_absolute_error": error})
            # Empty output and rank-one linear are supported too.
            assert packed.linear(torch.empty((0, k), device=device, dtype=torch.float16)).shape == (0, n)
            check_close(packed.linear(x[0]), F.linear(x[0], dense), "rank-one")
            if args.timing:
                import triton
                a = torch.randn((1, k), device=device, dtype=torch.float16)
                out = torch.empty((1, n), device=device, dtype=torch.float16)
                packed_ms = triton.testing.do_bench(lambda: packed.linear(a, out=out),
                                                     warmup=25, rep=100)
                dense_ms = triton.testing.do_bench(lambda: torch.mm(a, dense.T, out=out),
                                                    warmup=25, rep=100)
                results.append({"diagnostic_only": True, "bits": bits, "n": n, "k": k,
                                "packed_ms": packed_ms, "dense_ms": dense_ms})
        codes = (torch.randint(0, 2, (11, 7 * 3), dtype=torch.int8, device=device) * 2 - 1
                 if bits == 1 else torch.randint(-1, 2, (11, 7 * 3), dtype=torch.int8, device=device))
        packed = PackedWeight.from_codes(codes, bits=bits, scale=0.2)
        x = torch.randn((2, 7, 39), device=device, dtype=torch.float16)
        bias = torch.randn(11, device=device, dtype=torch.float16)
        for stride in (1, 2):
            check_close(packed.conv1d(x, 3, bias, stride=stride, padding=1),
                        F.conv1d(x, packed.dense().reshape(11, 7, 3), bias,
                                 stride=stride, padding=1), "conv1d")
    for codes, bits in ((torch.tensor([[0]], dtype=torch.int8), 1),
                        (torch.tensor([[2]], dtype=torch.int8), 2)):
        try:
            PackedWeight.from_codes(codes, bits=bits)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid alphabet accepted")
    torch.cuda.synchronize()
    print(json.dumps({"status": "passed", "device": torch.cuda.get_device_name(),
                      "capability": torch.cuda.get_device_capability(),
                      "torch_version": torch.__version__, "cases": results}, indent=2))


if __name__ == "__main__":
    with torch.inference_mode():
        main()
