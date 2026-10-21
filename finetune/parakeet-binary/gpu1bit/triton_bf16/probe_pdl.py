"""Does intra-op programmatic dependent launch help? (agent A) Graph-replay latency of A's ops with
kernels.PDL off / on, alternating windows, real MB1 layer-0 weights, provider semantics (tune.SEMANTICS).

    timeout 600 gpu1bit/shared/python -m gpu1bit.triton_bf16.probe_pdl [--ms 38 126]
"""
import argparse
import json
import statistics
import types

import torch
from safetensors import safe_open

from gpu1bit.shared import env
from gpu1bit.triton_bf16 import kernels
from gpu1bit.triton_bf16.bench import SHAPES, load_weights, make_input, time_graph
from gpu1bit.triton_bf16.gpulock import locked
from gpu1bit.triton_bf16.layer import TritonBF16Layer
from gpu1bit.triton_bf16.tune import SEMANTICS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", type=int, nargs="+", default=[38, 126])
    args = p.parse_args()
    with locked("A:probe-pdl"), torch.inference_mode():
        with safe_open(str(env.MB1_EXPORT / "export.safetensors"), framework="pt") as f:
            for label, n, k, kind, names in SHAPES:
                codes, scale = load_weights(f, names)
                layer = TritonBF16Layer.prepare(codes, scale, None, k, "cuda")
                use_norm, act, in_act = SEMANTICS[label]
                norm = types.SimpleNamespace(normalized_shape=(k,), eps=1e-5, weight=torch.rand(k, device="cuda") + .5,
                                             bias=torch.randn(k, device="cuda") * .1) if use_norm else None
                for m in args.ms:
                    x = make_input(label, kind, m, k, seed=m)
                    fn = lambda: layer.project(x, conv=kind == "conv", activation=act, norm=norm, input_activation=in_act)
                    res = {False: [], True: []}
                    for rep in range(4):
                        for pdl in ((False, True) if rep % 2 == 0 else (True, False)):
                            kernels.PDL = pdl
                            res[pdl].append(time_graph(fn, reps=20, windows=5, inner=10)[0])
                    kernels.PDL = False
                    off, on = statistics.median(res[False]), statistics.median(res[True])
                    print(json.dumps({"op": label, "M": m, "pdl_off_us": round(off, 3), "pdl_on_us": round(on, 3),
                                      "change": round(on / off - 1, 4)}), flush=True)


if __name__ == "__main__":
    main()
