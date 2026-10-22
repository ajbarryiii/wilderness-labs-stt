"""compute-sanitizer for the P-part kernels (x1 / x2, +-1 and b01) with exact allocation bounds.

The child (one gpu_lock, PYTORCH_NO_CUDA_MEMORY_CACHING=1 so every tensor is its own cudaMalloc and an
out-of-bounds access is reported; test_parts' graph capture needs the pool and is covered by H's child
through parts_run instead) runs dot_parts for P = 1, 2, both operand forms, every test_parts.CONFIGS
entry (all exchange modes, WK, cluster, PF), N x K in (1024 x 1024, 1024 x 4096, 1001 x 1088), M = 1, 37,
130, plus project / linear / conv of the four registered variants (incl. the MB1-layout fallback).

    shared/python sanitize_parts.py --tool memcheck|racecheck
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, "/mnt/hd/wilderness-labs-stt/parakeet-binary/gpu1bit/B/wip/parts")
import wip  # noqa: F401,E402

from gpu1bit.shared import env, sanitize  # noqa: E402


def child() -> None:
    import itertools
    import torch
    from gpu1bit.int8_mma import gpulock
    from gpu1bit.int8_mma.layer_parts import IMPLEMENTATIONS, _aligned, dot_parts, m1_quantize_parts, variant
    from gpu1bit.int8_mma.test_parts import CONFIGS, activations, weights
    with gpulock.locked("B:sanitizer x1/x2"), torch.inference_mode():
        launches = 0
        for (n, k), m, parts, b01 in itertools.product([(1024, 1024), (1024, 4096), (1001, 1088)], (1, 37, 130), (1, 2), (False, True)):
            codes, scale, b = weights(n, k, n + m + parts, bias=True)
            layer = variant(parts, b01).prepare(codes, scale, b, k, "cuda")
            x = activations(1, m, k, m)
            q, qs = m1_quantize_parts(x, parts)
            q, ldq = _aligned(q, k)
            for cfg in CONFIGS:
                if cfg.cluster > -(-k // 128):
                    continue
                y = torch.empty(m, n, device="cuda")
                dot_parts(layer, q, qs, ldq, m, y, config=cfg, parts=parts, b01=b01, silu=bool(launches % 2))
                launches += 1
        for name, cls in IMPLEMENTATIONS.items():
            for (n, k) in ((1024, 1024), (33, 1001), (37, 1001)):
                codes, scale, b = weights(n, k, 5, bias=True)
                layer = cls.prepare(codes, scale, b, k, "cuda")
                x = activations(2, 23, k, 9)
                layer.linear(x)
                layer.conv(x.transpose(1, 2))
                if k % 16 == 0:
                    layer.project(x, components=1)
                    layer.project(x, components=2, activation="silu")
                launches += 4
        torch.cuda.synchronize()
        print(f"SANITIZE-CHILD-DONE launches={launches} caching={os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING') != '1'}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tool", choices=["memcheck", "racecheck", "synccheck", "initcheck"], default="memcheck")
    p.add_argument("--child", action="store_true")
    args = p.parse_args()
    if args.child:
        return child()
    out = env.ARTIFACTS_ROOT / "B" / "sanitizer"
    env.require_mount()
    out.mkdir(parents=True, exist_ok=True)
    log = out / f"{time.strftime('%Y%m%dT%H%M%S')}-{args.tool}-parts-x1x2.log"
    cmd = ["timeout", "600", str(sanitize.SANITIZER), "--injection-path", str(sanitize.SANITIZER.parent),
           "--launch-timeout", "0", "--tool", args.tool, "--report-api-errors", "no", "--target-processes", "all",
           sys.executable, os.path.abspath(__file__), "--child", "--tool", args.tool]
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTORCH_NO_CUDA_MEMORY_CACHING": "1"}).returncode
    text = log.read_text()
    summary = [line for line in text.splitlines()
               if "ERROR SUMMARY" in line or "RACECHECK SUMMARY" in line or "SANITIZE-CHILD-DONE" in line]
    print("\n".join(summary) or text[-2000:])
    print(f"exit {rc}; log {log}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
