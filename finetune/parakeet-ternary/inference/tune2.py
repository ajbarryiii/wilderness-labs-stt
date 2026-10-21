"""Broader offline kernel sweep; all candidates are checked against FP32."""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import triton
from safetensors import safe_open

import paths
from export import unpack_codes
from .benchmark import DEFAULT_EXPORT, energy_meter, metadata, save
from .kernels import matmul, configuration
from .runtime import PackedLinear


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    p.add_argument("--rows", type=int, nargs="+", default=[38, 126, 376])
    p.add_argument("--modes", nargs="+", default=["bf16x3", "fp16x2"])
    args = p.parse_args()
    paths.require_mount()
    out = paths.ARTIFACTS / "kernel-runs" / f"{time.strftime('%Y%m%dT%H%M%S')}-tune2"
    out.mkdir(); print("RESULTS", out, flush=True)
    manifest = json.loads((args.export / "manifest.json").read_text())
    torch.manual_seed(571)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows, seen = [], set()
    with paths.gpu_lock("expanded kernel sweep"), torch.inference_mode(), energy_meter() as meter:
        meter._guard()
        save(out, "metadata.json", metadata(args))
        with safe_open(args.export / manifest["file"], framework="pt") as f:
            for name, layer in manifest["quantized_layers"].items():
                n, k = layer["shape"]
                if (n, k) in seen: continue
                seen.add((n, k))
                codes, scales = f.get_tensor(name+".codes"), f.get_tensor(name+".scale")
                mod = PackedLinear(codes, scales, k).cuda()
                dense = (unpack_codes(codes, k).float()*scales[:, None]).cuda()
                for m in args.rows:
                    x = torch.randn(m, k, device="cuda")
                    ref = torch.nn.functional.linear(x, dense)
                    for mode in args.modes:
                        configs = [configuration(m, n, k)]
                        configs += [(bm, bn, bk, split, warps, stages)
                                    for bm, bn, bk, warps in [(16, 64, 64, 2), (32, 64, 64, 4),
                                                              (32, 64, 128, 4), (32, 128, 64, 4),
                                                              (64, 64, 64, 4), (64, 64, 128, 4),
                                                              (64, 128, 64, 4), (32, 32, 64, 2)]
                                    for split in [1, 2, 4, 8] for stages in [2, 3]]
                        results = []
                        for config in dict.fromkeys(configs):
                            try:
                                fn = lambda: matmul(x, mod.packed_t, mod.scale, None, k, mode=mode, config=config)
                                y = fn()
                                torch.testing.assert_close(y, ref, atol=1e-4, rtol=2e-5)
                                us = triton.testing.do_bench_cudagraph(fn, rep=15)*1000
                                results.append(dict(config=config, us=us, max_error=float((y-ref).abs().max())))
                            except Exception as exc:
                                results.append(dict(config=config, error=repr(exc)))
                        good = sorted((r for r in results if "us" in r), key=lambda r:r["us"])
                        # Recheck finalists to avoid choosing a timing outlier.
                        for candidate in good[:4]:
                            config = candidate["config"]
                            fn = lambda: matmul(x, mod.packed_t, mod.scale, None, k, mode=mode, config=config)
                            candidate["recheck_us"] = [triton.testing.do_bench_cudagraph(fn, rep=50)*1000 for _ in range(3)]
                        finalists = sorted(good[:4], key=lambda r:statistics.median(r["recheck_us"]))
                        row = dict(m=m, n=n, k=k, mode=mode, baseline=results[0], best=finalists[0], candidates=results)
                        rows.append(row); save(out, "tune.json", rows)
                        print(json.dumps({k:v for k,v in row.items() if k != "candidates"}), flush=True)


if __name__ == "__main__":
    main()
