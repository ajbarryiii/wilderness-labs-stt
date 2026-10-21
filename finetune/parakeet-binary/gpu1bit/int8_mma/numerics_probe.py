"""CPU emulation of INT8x3 activation-quantization schemes on H's real dev fixtures (no GPU, no lock).

For each layer x utterance it computes, against the FP64 reference, api.errors of:
  m1        per-row dynamic residual scales over the full K (M1 / b1_dot exact INT32), one rounding
  static    per-row STATIC residual scales: s1 = s0 / 254, s2 = s1 / 254 (|residual| <= s / 2), so
            one max reduction suffices (single pass), full K, one rounding
  slice<S>  per-(row, K-slice) static scales over S equal K slices, an FP32 partial per slice
            (P = fma(c2, s2, fma(c0, s0, c1 * s1)) emulated in FP64 then rounded to FP32) summed in
            FP32 in slice order, times alpha: what an in-kernel quantization per cluster rank gives
  dslice<S> the same with per-slice DYNAMIC residual scales
and the ratio of each metric to H's per-layer limit (1.5x the worse of M1's kernels, numerics report
int8_mma.json "limits"). Integer dot products are exact (FP64 matmul of integers < 2^53).
CPU division is IEEE (GPU uses div.full, <= 2 ulp), so this is representative, not bitwise.

    CUDA_VISIBLE_DEVICES= gpu1bit/shared/python -m gpu1bit.int8_mma.numerics_probe --utterances 2 5 8
"""
from __future__ import annotations

import argparse
import collections
import json

import torch

from gpu1bit.shared import api, env


def quantize(x, static: bool):
    """x FP32 [M, K] -> (q [3, M, K] float64 integers, s [3, M] FP32)."""
    qs, ss = [], []
    r = x.clone()
    s = torch.clamp(r.abs().amax(dim=1) / 127.0, min=1e-30)
    for part in range(3):
        if part and not static:
            s = torch.clamp(r.abs().amax(dim=1) / 127.0, min=1e-30)
        elif part:
            s = (s / 254.0).float()
        q = torch.round(r / s[:, None])
        qs.append(q.double())
        ss.append(s)
        r = (r - (q * s[:, None]).float()).float()
    return torch.stack(qs), torch.stack(ss)


def scheme(x, signs_t, alpha, name):
    m, k = x.shape
    if name in ("m1", "static"):
        q, s = quantize(x, name == "static")
        acc = [q[c] @ signs_t for c in range(3)]
        y = acc[0] * s[0].double()[:, None] + acc[1] * s[1].double()[:, None] + acc[2] * s[2].double()[:, None]
        return (y.float().double() * alpha.double()).float()
    if name.startswith("xslice"):  # b1_dot xchg 2: M1's full-row scales, FP32 partial per cluster rank
        parts = int(name.removeprefix("xslice"))
        q, s = quantize(x, False)
        bounds = [round(k * i / parts) for i in range(parts + 1)]
        total = torch.zeros(m, signs_t.shape[1], dtype=torch.float32)
        for a, b in zip(bounds, bounds[1:]):
            acc = [q[c][:, a:b] @ signs_t[a:b] for c in range(3)]
            p = (acc[0] * s[0].double()[:, None] + acc[1] * s[1].double()[:, None] + acc[2] * s[2].double()[:, None]).float()
            total = (total + p).float()
        return (total.double() * alpha.double()).float()
    static = name.startswith("slice")
    parts = int(name.removeprefix("dslice").removeprefix("slice"))
    bounds = [round(k * i / parts) for i in range(parts + 1)]
    total = torch.zeros(m, signs_t.shape[1], dtype=torch.float32)
    for a, b in zip(bounds, bounds[1:]):
        q, s = quantize(x[:, a:b].contiguous(), static)
        acc = [q[c] @ signs_t[a:b] for c in range(3)]
        p = (acc[0] * s[0].double()[:, None] + acc[1] * s[1].double()[:, None] + acc[2] * s[2].double()[:, None]).float()
        total = (total + p).float()
    return (total.double() * alpha.double()).float()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--utterances", type=int, nargs="+", default=[2, 5, 8])
    p.add_argument("--schemes", nargs="+", default=["m1", "static", "slice2", "slice4", "slice8", "dslice4", "dslice8"])
    p.add_argument("--layers", default="", help="substring filter on layer names")
    args = p.parse_args()
    import os
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise SystemExit('run with CUDA_VISIBLE_DEVICES="" (CPU only)')
    from gpu1bit.shared import fixtures
    torch.set_num_threads(24)
    limits = {}
    for r in json.loads((env.ARTIFACTS / "numerics" / "int8_mma.json").read_text())["rows"]:
        limits[r["utterance"], r["layer"]] = r["limits"]
    layers = fixtures.mb1_layers()
    worst = collections.defaultdict(lambda: collections.defaultdict(lambda: (0.0, "")))
    over = collections.Counter()
    count = 0
    for u in args.utterances:
        inputs = fixtures.utterance_inputs(u, device="cpu")
        key = f"{u:02d}.safetensors"
        for name, w in layers.items():
            if args.layers and args.layers not in name:
                continue
            x = inputs[name]
            kind = "conv" if w["kind"] == "conv" else "linear"
            rows = (x.transpose(1, 2) if kind == "conv" else x).reshape(-1, w["K"]).float().contiguous()
            signs_t = api.unpack_signs(w["codes"], w["K"]).double().t().contiguous()
            ref = rows.double() @ signs_t * w["scale"].double()
            lim = limits[key, name]
            count += 1
            for sc in args.schemes:
                y = scheme(rows, signs_t, w["scale"], sc)
                e = api.errors("linear", rows, y, ref, w["scale"])
                for metric in ("relative_frobenius", "elementwise_guard"):
                    ratio = e[metric] / lim[metric]
                    if ratio > worst[sc][metric][0]:
                        worst[sc][metric] = (ratio, f"{key}:{name}")
                    if ratio > 1:
                        over[sc, metric] += 1
        print(json.dumps({"utterance": u, "evaluations": count,
                          "worst": {sc: {m: [round(v[0], 3), v[1]] for m, v in d.items()} for sc, d in worst.items()},
                          "over_limit": {f"{a}:{b}": c for (a, b), c in over.items()}}), flush=True)


if __name__ == "__main__":
    main()
