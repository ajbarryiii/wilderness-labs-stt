"""Gluon register-direct kernel: correctness vs FP64 and per-K-iteration cost vs the Triton kernel."""
import argparse, json
import torch
from gpu1bit.shared import api
from gpu1bit.triton_bf16.gpulock import locked
from gpu1bit.triton_bf16 import gluon_kernels as G
from gpu1bit.triton_bf16.layer import TritonBF16Layer
from gpu1bit.triton_bf16.probe_loop import kernel_us


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", default=["16,32,64,1,1,4", "16,64,64,1,1,4", "32,32,64,1,1,4", "16,32,64,4,1,4", "16,64,64,6,1,4", "16,32,128,1,1,4"])
    p.add_argument("--modes", nargs="+", default=["bf16x3", "fp16x2"])
    p.add_argument("--m", type=int, default=38)
    p.add_argument("--n", type=int, default=1024)
    args = p.parse_args()
    with locked("A:probe-gluon"), torch.inference_mode():
        m, n = args.m, args.n
        for kd in (1024, 4096):
            g = torch.Generator().manual_seed(kd)
            codes = torch.randint(0, 256, (n, kd // 8), dtype=torch.uint8, generator=g)
            scale = torch.rand(n, generator=g) * 0.05
            layer = TritonBF16Layer.prepare(codes, scale, None, kd, "cuda")
            x = (torch.randn(1, m, kd, generator=g) * torch.exp(torch.randn(1, m, kd, generator=g))).cuda()
            ref = api.reference_linear(x, codes, scale, None, kd)
            for mode in args.modes:
                for cfg in args.configs:
                    c = tuple(int(v) for v in cfg.split(","))
                    y = torch.empty(1, m, n, device="cuda")
                    try:
                        fn = lambda: G.rd_gemm(x, layer.bits_t, layer.scale, None, n, kd, y, (m * n, n, 1), mode=mode, config=c)
                        fn(); torch.cuda.synchronize()
                        err = api.relative_error(y, ref)
                        t = kernel_us(fn)
                        print(json.dumps({"K": kd, "mode": mode, "cfg": cfg, "rel_err": err, "kernels_us": {k: round(v, 3) for k, v in t.items()}}), flush=True)
                    except Exception as exc:
                        print(json.dumps({"K": kd, "mode": mode, "cfg": cfg, "error": repr(exc)[:400]}), flush=True)


if __name__ == "__main__":
    main()
