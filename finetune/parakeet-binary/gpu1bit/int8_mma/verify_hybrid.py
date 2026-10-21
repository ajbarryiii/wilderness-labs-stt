"""Coordinator condition 3 for B-hybrid (screening): identical context stack to R-hybrid apart from the
provider, the provider serves every projection, and no M1 weight kernel executes.

On dev clip 0 (eager encoder forward under each pipeline, same features):
  * call sequence: R-hybrid's PtxCudaProjection and B's HybridProjection are wrapped to record every
    call (N, K, conv, activation, norm module name, input_activation, deferred, prequantized); the
    two sequences must be equal call for call (24 x 9 = 216 calls with linear_pos in eager mode);
  * stack: the set of (module path, attribute) overridden in instance __dict__ (forward,
    forward_qkv, ... as patched by encoder_transform / layer_fusions / row_fusions / decoder
    transforms) must be equal for both models;
  * kernels: H's count_m1_weight_kernels (gpu1bit/tests/test_provider.py: _matmul, _gemv, _int8_dot,
    _stacked_dot) plus PtxCudaProjection native launches must be 0 in B-hybrid and > 0 in R-hybrid.
Writes /mnt/hd/.../gpu1bit/B/verify/<time>-hybrid.json.

    timeout 600 gpu1bit/shared/python -m gpu1bit.int8_mma.verify_hybrid
"""
from __future__ import annotations

import contextlib
import json
import time

import torch

from gpu1bit.int8_mma import gpulock


def patched_attributes(model) -> set:
    out = set()
    for name, mod in model.named_modules():
        for attr, value in vars(mod).items():
            if attr in ("forward", "forward_qkv", "_packed_positional_capture") or (callable(value) and not attr.startswith("_")
                                                                                   and not isinstance(value, torch.nn.Module)):
                out.add(f"{name}:{attr}")
    return out


def main():
    with gpulock.locked("B:verify B-hybrid stack"), torch.inference_mode(), contextlib.ExitStack() as stack:
        import evaluate as ev
        from nemo.utils import logging
        import inference.ptx_cuda as ptx
        from inference.row_fusion import DeferredProjection
        from gpu1bit.int8_mma import hybrid, paired  # noqa: F401
        from gpu1bit.shared import bench, env, workloads
        from gpu1bit.tests.test_provider import count_m1_weight_kernels
        logging.setLevel(logging.ERROR)
        stack.enter_context(ev.strict_fp32("cuda"))
        native = {"calls": 0}
        original_call = ptx.PtxCudaProjection.__call__

        def counted(self, *a, **kw):
            native["calls"] += 1
            return original_call(self, *a, **kw)
        ptx.PtxCudaProjection.__call__ = counted
        stack.callback(setattr, ptx.PtxCudaProjection, "__call__", original_call)
        models, info = {}, {}
        for n in ("R-hybrid", "B-hybrid"):
            models[n], info[n] = stack.enter_context(bench.variant_model(n, max_graphs=4))
        names = {n: {id(mm): k for k, mm in models[n].named_modules()} for n in models}
        records = {}

        owners = {id(info[n]["projection"]): n for n in models}
        for n in models:
            records[n] = []

        def recording(call):
            def wrapped(provider, x, mod, conv=False, components=3, activation="", norm=None, config=None,
                        input_activation=""):
                name = owners.get(id(provider))
                if name is None:
                    return call(provider, x, mod, conv, components, activation, norm, config, input_activation)
                xx = x.transpose(1, 2) if conv else x
                xx3 = xx if xx.dim() == 3 else xx.unsqueeze(0)
                hit = norm is not None and (xx3.data_ptr(), xx3.shape[0] * xx3.shape[1], xx3.shape[-1], id(norm)) in provider.prequantized
                y = call(provider, x, mod, conv, components, activation, norm, config, input_activation)
                records[name].append({"N": int(mod.packed_t.shape[1] if name == "B-hybrid" else mod.out_features),
                                      "K": int(xx3.shape[-1]), "M": int(xx3.shape[0] * xx3.shape[1]), "conv": bool(conv),
                                      "activation": activation, "norm": names[name].get(id(norm)) if norm is not None else None,
                                      "input_activation": input_activation, "prequantized": bool(hit),
                                      "deferred": isinstance(y, DeferredProjection)})
                return y
            return wrapped
        for cls in {type(info[n]["projection"]) for n in models}:
            stack.callback(setattr, cls, "__call__", cls.__call__)
            cls.__call__ = recording(cls.__call__)
        stacks = {n: patched_attributes(m) for n, m in models.items()}
        clip = workloads.load()["dev_clips"][0]
        audio, length = ev.audio_batch([clip])
        feats, flen = models["R-hybrid"].preprocessor(input_signal=audio.cuda(), length=length.cuda())
        counts, outs = {}, {}
        for n in ("R-hybrid", "B-hybrid"):
            native["calls"] = 0
            records[n].clear()
            with count_m1_weight_kernels() as c:
                outs[n] = models[n].encoder(audio_signal=feats, length=flen)
                torch.cuda.synchronize()
            counts[n] = {**c, "PtxCudaProjection.__call__": native["calls"]}
        t = int(outs["R-hybrid"][1][0])
        diff = outs["B-hybrid"][0][..., :t] - outs["R-hybrid"][0][..., :t]
        report = {
            "clip": clip["id"], "M": int(flen[0]),
            "calls": {n: len(r) for n, r in records.items()},
            "call_sequences_equal": records["R-hybrid"] == records["B-hybrid"],
            "first_difference": next(({"index": i, "R-hybrid": a, "B-hybrid": b} for i, (a, b) in
                                      enumerate(zip(records["R-hybrid"], records["B-hybrid"])) if a != b), None),
            "deferred_calls": {n: sum(r["deferred"] for r in rec) for n, rec in records.items()},
            "prequantized_calls": {n: sum(r["prequantized"] for r in rec) for n, rec in records.items()},
            "patched_attributes_equal": stacks["R-hybrid"] == stacks["B-hybrid"],
            "patched_attribute_count": {n: len(s) for n, s in stacks.items()},
            "patched_only_in": {n: sorted(stacks[n] - stacks[o])[:20] for n, o in (("R-hybrid", "B-hybrid"), ("B-hybrid", "R-hybrid"))},
            "m1_weight_kernel_launches": counts,
            "encoder_vs_R-hybrid": {"lengths_equal": bool(torch.equal(outs["B-hybrid"][1], outs["R-hybrid"][1])),
                                    "rel_frobenius_valid": (diff.norm() / outs["R-hybrid"][0][..., :t].norm()).item(),
                                    "max_abs_valid": diff.abs().max().item()},
            "sequence": records["B-hybrid"],
        }
        report["ok"] = (report["call_sequences_equal"] and report["patched_attributes_equal"]
                        and sum(counts["B-hybrid"].values()) == 0 and sum(counts["R-hybrid"].values()) > 0
                        and report["calls"]["B-hybrid"] == 24 * 9)
        out = env.ARTIFACTS_ROOT / "B" / "verify"
        out.mkdir(parents=True, exist_ok=True)
        path = out / (time.strftime("%Y%m%dT%H%M%S") + "-hybrid.json")
        path.write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "sequence"}, indent=1))
        print("RESULTS", path)
        if not report["ok"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
