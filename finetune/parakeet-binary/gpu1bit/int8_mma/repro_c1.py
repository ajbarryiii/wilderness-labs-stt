"""Review c1 finding 1 on the GPU: split-boundary cancellation through every deferred / split path.

Case A (Codex): M = 126, N = 1024, K = 4096, every weight +1, scale 1, no bias; each row x[0] = 16256,
x[128] = 2^-12, x[1024] = -16256, else 0 (exact result 2^-12 per output).
Case B (M1's interleaved split): the same with the cancelling value at x[BK] for R-hybrid's own BK, and
the small one at x[1], so it shares M1 split part 0 with x[0] while -16256 lands in part 1.
Paths: FP64 reference; M1's two A3 references (R-bf16x3, R-expanded INT8x3, gpu1bit.shared.m1.LayerOps);
R-hybrid's PtxCudaProjection('native', row_warps='adaptive') deferred (DeferredProjection +
row_fusion.finish_residual) and plain (its _finish); frozen B-hybrid deferred (XCHG 4 + M1's row
epilogue), B's exact in-kernel path, the new exact deferred path (XCHG 5 + _row_epilogue_exact), and
the b01 forms. Per path: distinct output values, api.errors, pass/fail against 1.5 x the worse M1
reference (the A3 rule). One process, one gpu_lock.

    timeout 600 gpu1bit/shared/python -m gpu1bit.int8_mma.repro_c1
"""
from __future__ import annotations

import json
import types

import torch

from gpu1bit.int8_mma import configs, gpulock, hybrid_exact
from gpu1bit.int8_mma.layer import Int8MmaLayer
from gpu1bit.int8_mma.layer_b01 import Int8MmaB01Layer
from gpu1bit.shared import api, env


def r_hybrid(x, module, deferred):
    from inference.ptx_cuda import PtxCudaProjection
    from inference.row_fusion import finish_residual
    proj = PtxCudaProjection(tuning="native", row_warps="adaptive")
    proj.defer_epilogue = deferred
    m, n, k = x.shape[0] * x.shape[1], module.out_features, x.shape[-1]
    value = proj(x, module)
    config = proj.configuration(m, n, k)
    if deferred:
        y = finish_residual(proj, value, torch.zeros(*x.shape[:-1], n, device="cuda"), 1.0)
        return y, {"config(bm,bn,bk,split,warps,stages)": config[0], "variant": config[1], "groups": config[2],
                   "deferred_split": value.split}
    return value, {"config(bm,bn,bk,split,warps,stages)": config[0], "variant": config[1], "groups": config[2]}


def main():
    env.require_shared_wrapper()
    out = env.ARTIFACTS_ROOT / "B" / "c1"
    out.mkdir(parents=True, exist_ok=True)
    report = {"cases": {}}
    with gpulock.locked("B:c1 repro") as lock, torch.inference_mode():
        from inference.encoder4 import expanded_configuration
        from gpu1bit.shared import m1
        torch.backends.cuda.matmul.allow_tf32 = False
        m, n, k = 126, 1024, 4096
        codes = api.pack_signs(torch.ones(n, k, dtype=torch.int8))
        scale = torch.ones(n)
        ops = m1.LayerOps(codes, scale, None, k)
        layers = {"pm1": Int8MmaLayer.prepare(codes, scale, None, k, "cuda"),
                  "b01": Int8MmaB01Layer.prepare(codes, scale, None, k, "cuda")}
        _, rcfg = r_hybrid(torch.zeros(1, m, k, device="cuda"), ops.module, True)
        bk = rcfg["config(bm,bn,bk,split,warps,stages)"][2]
        report["r_hybrid_config"] = rcfg
        report["r_expanded_expanded_configuration(bm,bn,bk,split,warps,stages)"] = list(expanded_configuration(m, n, k))
        report["b_deferred_frozen_config"] = configs.choose_deferred(m, n, k).label()
        report["b_exact_inkernel_config"] = configs.choose(m, n, k).label()
        cfg_new = hybrid_exact.choose_exact_deferred(m, n, k)
        report["b_exact_deferred_config"] = cfg_new.label() if cfg_new else None
        cases = {"A_codex": {0: 16256.0, 128: 2.0**-12, 1024: -16256.0},
                 "B_interleaved": {0: 16256.0, 1: 2.0**-12, bk: -16256.0}}
        for case, entries in cases.items():
            x = torch.zeros(1, m, k, device="cuda")
            for col, v in entries.items():
                x[..., col] = v
            ref = api.reference("linear", x, codes, scale, None, k)
            outs = {"R-bf16x3": ops.bf16x3(x), "R-expanded int8x3": ops.int8x3(x)}
            outs["R-hybrid deferred (ptx_row_best)"], _ = r_hybrid(x, ops.module, True)
            outs["R-hybrid plain (ptx_row_best, _finish)"], _ = r_hybrid(x, ops.module, False)
            for tag, layer in layers.items():
                outs[f"B-hybrid frozen deferred XCHG 4 [{tag}]"] = hybrid_exact.deferred_project(layer, x, exact=False)
                outs[f"B exact in-kernel [{tag}]"] = layer.linear(x)
                outs[f"B-hybrid-exact deferred XCHG 5 [{tag}]"] = hybrid_exact.deferred_project(layer, x, exact=True)
            torch.cuda.synchronize()
            errs = {name: api.errors("linear", x, y, ref, scale) for name, y in outs.items()}
            limits = api.numerics_thresholds({"r_bf16x3": errs["R-bf16x3"], "r_expanded_int8x3": errs["R-expanded int8x3"]})
            rows = {}
            for name, y in outs.items():
                values = torch.unique(y.float()).tolist()
                rows[name] = {"distinct_values": values[:6], "errors": errs[name], "A3_pass": api.within(errs[name], limits),
                              "equal_to_B_exact_inkernel_bitwise": bool(torch.equal(y, outs["B exact in-kernel [pm1]"]))}
            report["cases"][case] = {"x_nonzero": {str(c): v for c, v in entries.items()},
                                     "reference_distinct": torch.unique(ref).tolist()[:4], "limits": limits, "paths": rows}
            print(case, json.dumps({n_: (r["distinct_values"], r["A3_pass"]) for n_, r in rows.items()}, indent=0), flush=True)
        report["lock"] = {"label": lock["label"], "held_s": env.check_hold(lock, margin_s=0)}
        (out / "repro.json").write_text(json.dumps(report, indent=1) + "\n")
        print("RESULTS", out / "repro.json")


if __name__ == "__main__":
    main()
