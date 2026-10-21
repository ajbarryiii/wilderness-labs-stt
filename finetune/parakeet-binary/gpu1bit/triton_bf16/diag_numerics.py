"""Where does a variant's worst elementwise-guard error come from? (agent A diagnostic)

On one captured dev input (fixture utterance U, layer L): per-element |y - y_ref| / (alpha_j ||x_i||_1)
for A's variants and M1's R-bf16x3 / R-expanded, the worst (i, j), and the row's statistics.

    timeout 600 gpu1bit/shared/python -m gpu1bit.triton_bf16.diag_numerics [--layer ...] [--utt 0]
"""
from __future__ import annotations

import argparse

import torch

from gpu1bit.shared import api, fixtures
from gpu1bit.triton_bf16.gpulock import locked
from gpu1bit.triton_bf16.layer import TritonBF16Layer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", default="encoder.layers.0.self_attn.linear_pos")
    p.add_argument("--utt", type=int, default=0)
    p.add_argument("--modes", nargs="+", default=["bf16x3", "bf16x3l", "fp16x2", "fp16x2p"])
    p.add_argument("--configs", nargs="*", default=[])
    args = p.parse_args()
    with locked("A:diag"), torch.inference_mode():
        torch.backends.cuda.matmul.allow_tf32 = False
        w = fixtures.mb1_layers([args.layer])[args.layer]
        x = fixtures.utterance_inputs(args.utt)[args.layer]
        kind, k = w["kind"], w["K"]
        ref = api.reference(kind, x, w["codes"], w["scale"], w["bias"], k)
        x3 = (x.transpose(1, 2) if kind == "conv" else x).reshape(-1, k).double()
        l1 = x3.abs().sum(1)
        print("x", tuple(x.shape), tuple(x.stride()), "M", x3.shape[0], "row |x|_1 min %.3g max %.3g" % (l1.min(), l1.max()))
        alpha = w["scale"].to(x.device).double()
        variants = {}
        for mode in args.modes:
            layer = TritonBF16Layer.prepare(w["codes"], w["scale"], w["bias"], k, "cuda", mode=mode)
            variants[f"A-{mode}"] = api.call(layer, kind, x)
        for spec in args.configs:
            mode, _, cfg = spec.partition("@")
            m = x3.shape[0]
            layer = TritonBF16Layer.prepare(w["codes"], w["scale"], w["bias"], k, "cuda", mode=mode,
                                            configs={m: tuple(int(v) if v.lstrip("-").isdigit() else v for v in cfg.split(","))})
            variants[f"A-{spec}"] = api.call(layer, kind, x)
        from gpu1bit.triton_bf16.bench import baselines
        base, keep = baselines(w["codes"], w["scale"], k, kind)
        for name in ("R-bf16x3", "R-expanded"):
            variants[name] = base[name](x)
        for name, y in variants.items():
            yy = (y.transpose(1, 2) if kind == "conv" else y).reshape(-1, y.shape[1] if kind == "conv" else y.shape[-1]).double()
            rr = (ref.transpose(1, 2) if kind == "conv" else ref).reshape(yy.shape)
            err = (yy - rr).abs()
            guard = err / (alpha[None, :] * l1[:, None] + 1e-300)
            i, j = divmod(int(guard.argmax()), guard.shape[1])
            rows_bad = (guard > 1.5e-7).any(1).nonzero().flatten().tolist()
            print(f"{name:22s} frob {api.relative_error(yy, rr):.3e} guard {guard.max():.3e} at (i={i}, j={j}) "
                  f"y_ref {rr[i, j]:.6g} err {yy[i, j] - rr[i, j]:.3e} | rows with guard>1.5e-7: {rows_bad[:12]}"
                  f"{'...' if len(rows_bad) > 12 else ''} ({len(rows_bad)})")
        i = int(((variants["A-bf16x3"].reshape(-1, ref.shape[-1]).double() - ref.reshape(-1, ref.shape[-1])).abs()
                 / (alpha[None, :] * l1[:, None])).max(1).values.argmax())
        row = x3[i]
        print("worst row", i, "max|x| %.4g min nonzero |x| %.4g zeros %d mean %.4g" %
              (row.abs().max(), row[row != 0].abs().min(), int((row == 0).sum()), row.mean()))


if __name__ == "__main__":
    main()
