"""One-off diagnostic: how binary-ready are M1's ternary latents vs the FP32 original?

Per quantized matrix W (row scale s = mean|W|):
  ternary T(W) = clamp(round(W/s), -1, 1) * s     binary B(W) = sign(W) * mean|W|
Reports, weighted by parameter count, per module family:
  zero      fraction of ternary codes that are 0
  near0     fraction with |W|/s < 0.1 (sign essentially arbitrary)
  errT/errB relative error ||W - Q(W)|| / ||W||
  B_vs_T    ||B(W) - T(W)|| / ||T(W)||: how far binarizing moves the weights the network actually uses
  flip      fraction of nonzero ternary codes whose sign differs from sign(W) (sanity, should be 0)
"""
import json
import re
import sys
import tarfile
from collections import defaultdict

import torch

PATTERN = re.compile(r"encoder\.layers\.\d+\.(feed_forward[12]\.linear[12]|self_attn\.linear_(q|k|v|out|pos)"
                     r"|conv\.pointwise_conv[12])\.weight")


def family(key):
    if "feed_forward" in key:
        return "ff"
    if "self_attn" in key:
        return "attn"
    return "pwconv"


def stats(sd):
    acc = defaultdict(lambda: defaultdict(float))
    keys = sorted(k for k in sd if PATTERN.fullmatch(k.removeprefix("student.")))
    assert len(keys) == 264, len(keys)
    for k in keys:
        w = sd[k].float().reshape(sd[k].shape[0], -1)
        s = w.abs().mean(1, keepdim=True).clamp_min(1e-8)
        r = w / s
        codes = r.round().clamp(-1, 1)
        t = codes * s
        b = torch.sign(w) * s  # mean|W| is the L2-optimal binary scale
        n = w.numel()
        nonzero = codes != 0
        for fam in (family(k), "all"):
            a = acc[fam]
            a["n"] += n
            a["zero"] += (codes == 0).sum().item()
            a["near0"] += (r.abs() < 0.1).sum().item()
            a["near0_25"] += (r.abs() < 0.25).sum().item()
            a["w2"] += w.pow(2).sum().item()
            a["eT"] += (w - t).pow(2).sum().item()
            a["eB"] += (w - b).pow(2).sum().item()
            a["t2"] += t.pow(2).sum().item()
            a["bt"] += (b - t).pow(2).sum().item()
            a["flip"] += (nonzero & (torch.sign(w) != codes)).sum().item()
            a["nz"] += nonzero.sum().item()
    out = {}
    for fam, a in acc.items():
        out[fam] = {"params": int(a["n"]), "zero": a["zero"] / a["n"], "near0_lt0.1": a["near0"] / a["n"],
                    "near0_lt0.25": a["near0_25"] / a["n"], "errT": (a["eT"] / a["w2"]) ** .5,
                    "errB": (a["eB"] / a["w2"]) ** .5, "B_vs_T": (a["bt"] / a["t2"]) ** .5,
                    "flip": a["flip"] / a["nz"]}
    return out


def main(m1_ckpt, nemo, out):
    m1 = torch.load(m1_ckpt, map_location="cpu", mmap=True, weights_only=False)["student"]
    with tarfile.open(nemo) as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("model_weights.ckpt"))
        fp32 = torch.load(tar.extractfile(member), map_location="cpu", weights_only=True)
    result = {"M1_latent": stats(m1), "FP32_original": stats(fp32)}
    json.dump(result, open(out, "w"), indent=1)
    for name, fams in result.items():
        print(name)
        for fam, v in fams.items():
            print(f"  {fam:7s}", " ".join(f"{k}={v[k]:.4f}" if isinstance(v[k], float) else f"{k}={v[k]}"
                                          for k in v))


if __name__ == "__main__":
    main(*sys.argv[1:])
