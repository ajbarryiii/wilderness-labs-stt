"""compute-sanitizer on the project() paths H's shared/sanitize.py does not reach (agent A, A8):
LayerNorm fusion (_rowstats + normalizing _bgemm; _split_rows with NORM), input/output SiLU, conv
views, split-K finish, with PDL, eagerly and under CUDA-graph replay, M = 1, 5, 38, 126.

    gpu1bit/shared/python -m gpu1bit.triton_bf16.sanitize_project --tool memcheck|racecheck
"""
import argparse
import subprocess
import sys
import time
import types

from gpu1bit.shared import env
from gpu1bit.shared.sanitize import SANITIZER


def child():
    import torch
    from gpu1bit.shared import api
    from gpu1bit.triton_bf16.gpulock import locked
    from gpu1bit.triton_bf16.layer import TritonBF16Layer
    with locked("A:sanitize-project"), torch.inference_mode():
        g = torch.Generator().manual_seed(5)
        for n, k, conv, use_norm, ia, act in [(1024, 1024, False, True, "", ""), (4096, 1024, False, True, "", "silu"),
                                              (2048, 1024, True, True, "", ""), (1024, 1024, True, False, "silu", ""),
                                              (37, 1001, False, True, "silu", "silu")]:
            signs = (torch.randint(0, 2, (n, k), generator=g) * 2 - 1).to(torch.int8)
            layer = TritonBF16Layer.prepare(api.pack_signs(signs), torch.rand(n, generator=g), None, k, "cuda")
            norm = types.SimpleNamespace(normalized_shape=(k,), eps=1e-5, weight=torch.rand(k, device="cuda") + .5,
                                         bias=torch.randn(k, device="cuda")) if use_norm else None
            for m in (1, 5, 38, 126):
                x = torch.randn(1, k, m, device="cuda") if conv else torch.randn(1, m, k, device="cuda")
                fn = lambda: layer.project(x, conv=conv, norm=norm, activation=act, input_activation=ia)
                fn()
                graph = torch.cuda.CUDAGraph()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    with torch.cuda.graph(graph):
                        fn()
                torch.cuda.current_stream().wait_stream(s)
                graph.replay()
                torch.cuda.synchronize()
                del graph
        print("SANITIZE-PROJECT-DONE", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tool", default="memcheck")
    p.add_argument("--child", action="store_true")
    args = p.parse_args()
    if args.child:
        return child()
    out = env.ARTIFACTS_ROOT / "A" / "sanitizer"
    out.mkdir(parents=True, exist_ok=True)
    log = out / f"{time.strftime('%Y%m%dT%H%M%S')}-{args.tool}-project.log"
    cmd = ["timeout", "600", str(SANITIZER), "--injection-path", str(SANITIZER.parent), "--tool", args.tool,
           "--report-api-errors", "no", "--target-processes", "all", sys.executable, "-m",
           "gpu1bit.triton_bf16.sanitize_project", "--child", "--tool", args.tool]
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"exit {rc}; log {log}")
    print(open(log).read()[-600:])


if __name__ == "__main__":
    main()
