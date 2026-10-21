"""A3 for DEFERRED providers (Codex code review c1, finding 1): the R-hybrid-protocol providers whose
N = 1024, no-activation projections return a DeferredProjection that M1's row epilogue
(inference.row_fusion._row_epilogue) finishes: B-hybrid's HybridProjection (and its b01 twin) and
R-hybrid's own PtxCudaProjection (the latter reported for information only).

The value under test is exactly what the pipeline adds to the residual stream: the provider is
called with defer_epilogue as R-hybrid's row_fusions sets it, and a DeferredProjection is finished by
M1's own `finish_residual(provider, value, residual=0, factor=1.0)`. With a zero residual and factor 1
that kernel's output is the projection value bit for bit (0 + v * 1.0 == v), so the split-slice
FP32 summation, scaling and bias of the real epilogue are what is measured.

REAL: all 264 layers x the 15 frozen dev fixture utterances, each layer deferred exactly when the
pipeline defers it (self_attn.linear_out, feed_forward{1,2}.linear2, conv.pointwise_conv2; the rest go
through the provider's non-deferred path), input as captured. pointwise_conv2 is called with
input_activation="" on its captured (post-SiLU) input, as M1's per-layer references are. Limits are
the per-layer, per-utterance A3 limits of fixtures.m1_thresholds() (1.5 x the worse of R-bf16x3 and
R-expanded INT8x3; both metrics; no floor).
SYNTHETIC (split-boundary cancellation; "discriminating" = both M1 references exact): for the deferred shapes (1024 x 1024 linear and conv,
1024 x 4096 linear) at M = 38, 126, 368: rows carrying +a at position i, -a at position j and a small
d at position r, placed across contiguous slice boundaries K*s/S (S = 2, 3, 4, 8) and across
interleaved tile boundaries (BK = 32, 64, 128 with SPLIT = 2..8), with all-+1 weights (every output
cancels) and with random +-1 weights and real-like scales; an M1-exact family for K = 4096 (+a / -a
1024 apart, i.e. in the same split part of both M1 references, small term 128 away, so both
references are exact while the pair crosses contiguous slice boundaries); Codex's counterexample exactly
(M = 126, N = 1024, K = 4096, weights +1, scale 1, no bias, x[0] = 16256, x[128] = 2^-12,
x[1024] = -16256). Limits: 1.5 x the worse of M1's two references on the same input, computed live.

PROVIDERS (the interface agreed with B in NOTES): built in, "R-hybrid" (information only),
"B-hybrid@frozen" (gpu1bit.int8_mma.hybrid.HybridProjection over int8_mma) and "B-hybrid-b01@frozen"
(the same over int8_mma:b01, registered by gpu1bit.int8_mma.b01_run). External: a module given with
--module exposing DEFERRED_PROVIDERS = {name: {"kernel": <registry name>, "factory":
callable(table, names) -> provider}} (any patch modules it needs are imported by --patches).

    timeout 600 gpu1bit/shared/python -m gpu1bit.shared.dispatch --patches gpu1bit.int8_mma.b01_run \\
        deferred qualify --provider B-hybrid@frozen
Report: /mnt/hd/wilderness-labs-stt/parakeet-binary/gpu1bit/H/deferred/[<run>/]<provider>.json

PASSES (Codex GPU code review c3): `passes` is True only for the COMPLETE scope (`full_scope`: every
fixture utterance, real and synthetic) with zero failures. A partial run (--no-real, --no-synthetic,
an utterance subset) is marked partial and is informational only; `checks_passed` says whether what
ran passed. The report records its scope, the provider identity (registry kernel and factory) and a
fingerprint that includes the scope. Selection validates reports with selection.deferred_a3_check.
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
import types

import torch

from gpu1bit.shared import api, env

DEFERRED_SUFFIXES = ("self_attn.linear_out", "feed_forward1.linear2", "feed_forward2.linear2", "conv.pointwise_conv2")
CODEX = {"name": "codex_c1_counterexample", "M": 126, "N": 1024, "K": 4096, "kind": "linear", "weights": "plus_one",
         "entries": [(0, 16256.0), (128, 2.0 ** -12), (1024, -16256.0)]}


def pipeline_defers(name: str) -> bool:
    return name.endswith(DEFERRED_SUFFIXES)


# --- providers -----------------------------------------------------------------------------------

class _Ptx:
    """R-hybrid's PtxCudaProjection (one instance shared by all layers, as in the pipeline: its weight
    and kernel caches are keyed per layer) over the derived 2-bit module of one layer."""
    _shared = None

    def __init__(self, w, kind):
        from inference.ptx_cuda import PtxCudaProjection
        from gpu1bit.shared import m1
        if _Ptx._shared is None:
            _Ptx._shared = PtxCudaProjection(None, tuning="native", weight_bits=8, row_warps="adaptive")
        self.provider = _Ptx._shared
        self.mod = m1.derived_packed_linear(w["codes"], w["scale"], w["bias"], w["K"], conv=kind == "conv")


class _Binary:
    """A BinaryLayer-table provider (B's) over one prepared layer."""

    def __init__(self, w, kind, impl, factory):
        from gpu1bit.shared.provider import Table, _handle
        layer = impl.prepare(w["codes"], w["scale"], w["bias"], w["K"], "cuda")
        table = Table()
        table.add("layer", layer, kind, w["N"], w["K"])
        self.provider = factory(table, {})
        self.mod = types.SimpleNamespace(packed_t=_handle(0, w["N"]))


def provider_spec(name: str, module: str | None = None, components: int = 3):
    """(builder(w, kind) -> object with .provider/.mod, informational?). An external provider's
    registered part count (entry "components", default 3) must equal `components`."""
    from gpu1bit.shared import registry
    if components not in HARNESS_COMPONENTS:
        raise ValueError(f"part count {components} is not one of {HARNESS_COMPONENTS}")
    if name == "R-hybrid":
        if components != 3:
            raise ValueError("R-hybrid serves part count 3 only")
        return (lambda w, kind: _Ptx(w, kind)), True
    if name in ("B-hybrid@frozen", "B-hybrid-b01@frozen"):
        if components != 3:
            raise ValueError(f"{name} serves part count 3 only")
        from gpu1bit.int8_mma.hybrid import HybridProjection
        impl = registry.get("int8_mma" if name == "B-hybrid@frozen" else "int8_mma:b01")
        return (lambda w, kind: _Binary(w, kind, impl, HybridProjection)), False
    if module is None:
        raise KeyError(f"unknown deferred provider {name!r} (external ones need --module)")
    entry = importlib.import_module(module).DEFERRED_PROVIDERS[name]
    if int(entry.get("components", 3)) != components:
        raise ValueError(f"{name}: registered part count {entry.get('components', 3)} is not the requested {components}")
    if ":x" in entry["kernel"]:   # a part-count layer variant: B's hash-pinned files (plans/x2-default.md F5)
        from gpu1bit.actprec import parts as AP
        AP.load_parts()
    impl = registry.get(entry["kernel"])
    return (lambda w, kind: _Binary(w, kind, impl, entry["factory"])), False


def run_layer(obj, kind: str, x: torch.Tensor, defer: bool, components: int = 3):
    """(y in the layer's output layout, whether the provider deferred)."""
    from inference.row_fusion import DeferredProjection, finish_residual
    provider, conv = obj.provider, kind == "conv"
    provider.defer_epilogue = defer
    try:
        v = provider(x, obj.mod, conv, components, "", None, None, "")
    finally:
        provider.defer_epilogue = False
    if isinstance(v, DeferredProjection):
        residual = torch.zeros(v.shape, device=x.device, dtype=torch.float32)
        y = finish_residual(provider, v, residual, 1.0)          # [B, T, N] = 0 + value * 1.0
        if conv:
            return y.transpose(1, 2), True
        return y.reshape(*x.shape[:-1], y.shape[-1]), True
    return v, False


# --- cases ---------------------------------------------------------------------------------------

def synthetic_cases() -> list[dict]:
    cases = [CODEX]
    for (n, k) in ((1024, 1024), (1024, 4096)):
        bounds = set()
        for s_count in (2, 3, 4, 8):
            for s in range(1, s_count):
                bounds.add((k * s) // s_count)
        for bk in (32, 64, 128):
            for split in (2, 4, 8):
                if bk * split < k:
                    bounds.add(bk * split)
                bounds.add(bk)
        bounds = sorted(b for b in bounds if 1 < b < k - 1)
        for m in (38, 126, 368):
            for idx, b in enumerate(bounds):
                for weights in ("plus_one", "random"):
                    i, j = (0, b) if idx % 2 == 0 else (b - 1, b)
                    r = min(i + 1 if i + 1 != j else i + 2, k - 1)
                    cases.append({"name": f"cancel_n{n}_k{k}_m{m}_b{b}_{weights}", "M": m, "N": n, "K": k,
                                  "kind": "linear", "weights": weights,
                                  "entries": [(i, 16256.0), (r, 2.0 ** -12), (j, -16256.0)]})
            if k == 1024:
                cases.append({"name": f"cancel_conv_n{n}_k{k}_m{m}", "M": m, "N": n, "K": k, "kind": "conv",
                              "weights": "plus_one", "entries": [(0, 16256.0), (128, 2.0 ** -12), (512, -16256.0)]})
            if k == 4096:
                # M1-exact family (Codex's construction generalised): +a and -a 1024 apart fall in the same
                # split part of BOTH M1 references (expanded INT8x3 interleaves BK 128 x SPLIT 8, BF16x3
                # BK 32 x SPLIT 8, periods dividing 1024), the small term 128 away lands in another part,
                # so both references are exact while the pair crosses contiguous K-slice boundaries.
                for i in (0, 64, 384, 896):
                    for t in (1, 2, 3):
                        j = i + 1024 * t
                        if j >= k:
                            continue
                        for weights in ("plus_one", "random"):
                            cases.append({"name": f"m1exact_n{n}_k{k}_m{m}_i{i}_j{j}_{weights}", "M": m, "N": n, "K": k,
                                          "kind": "linear", "weights": weights,
                                          "entries": [(i, 16256.0), (i + 128, 2.0 ** -12), (j, -16256.0)]})
    return cases


def case_data(case: dict, seed: int):
    g = torch.Generator().manual_seed(seed)
    n, k, m = case["N"], case["K"], case["M"]
    if case["weights"] == "plus_one":
        signs = torch.ones(n, k, dtype=torch.int8)
        scale = torch.ones(n)
    else:
        signs = (torch.randint(0, 2, (n, k), generator=g, dtype=torch.int8) * 2 - 1).to(torch.int8)
        scale = 0.01 + 0.04 * torch.rand(n, generator=g)
    w = {"codes": api.pack_signs(signs), "scale": scale.float().contiguous(), "bias": None, "N": n, "K": k}
    rows = torch.zeros(m, k)
    for pos, val in case["entries"]:
        rows[:, pos] = val
    if case["kind"] == "conv":
        x = rows.t().contiguous().unsqueeze(0).cuda()           # [1, K, T]: pointwise_conv2's contiguous layout
    else:
        x = rows.unsqueeze(0).cuda()                              # [1, M, K]
    return w, x


# --- qualification -------------------------------------------------------------------------------

def _exact_x2(obj, kind, x, expect_defer: bool):
    """x2 criterion (plans/x2-default.md F3, Z5): the deferred value, finished by finish_residual(0, 1.0),
    equals the same provider's non-deferred x2 value bitwise; deferral where the pipeline defers."""
    y_def, deferred = run_layer(obj, kind, x, expect_defer, components=2)
    y_ref, ref_deferred = run_layer(obj, kind, x, False, components=2)
    exact = (not ref_deferred) and y_def.shape == y_ref.shape and bool(torch.equal(y_def, y_ref))
    return {"deferred": deferred, "exact": exact, "finite": bool(torch.isfinite(y_def).all()),
            "max_abs_difference": (y_def.double() - y_ref.double()).abs().max().item() if y_def.shape == y_ref.shape else None}


def qualify(name: str, module: str | None = None, utterances: list[int] | None = None, real: bool = True,
            synthetic: bool = True, components: int = 3) -> dict:
    if components == 2:
        return _qualify_x2(name, module, utterances, real, synthetic)
    from gpu1bit.shared import fixtures, m1
    build, informational = provider_spec(name, module, 3)
    out = {"provider": name, "module": module, "informational": informational, "real": None, "synthetic": None}
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode():
        if synthetic:
            rows = []
            for c_i, case in enumerate(synthetic_cases()):
                _Ptx._shared = None  # a fresh R-hybrid provider per case: its INT8 cache is keyed by data_ptr
                w, x = case_data(case, 9000 + c_i)
                kind = case["kind"]
                ref = api.reference(kind, x, w["codes"], w["scale"], w["bias"], w["K"])
                ops = m1.LayerOps(w["codes"], w["scale"], w["bias"], w["K"])
                m1_err = {o: api.errors(kind, x, f(x, kind == "conv"), ref, w["scale"]) for o, f in ops.ops().items()}
                limits = api.numerics_thresholds(m1_err)
                y, deferred = run_layer(build(w, kind), kind, x, True)
                got = api.errors(kind, x, y, ref, w["scale"])
                rows.append({"case": case["name"], "M": case["M"], "N": case["N"], "K": case["K"], "kind": kind,
                             "weights": case["weights"], "deferred": deferred, **got, "m1": m1_err, "limits": limits,
                             "ok": api.within(got, limits) and deferred})
                del ops
            out["synthetic"] = {"cases": len(rows), "failures": sum(not r["ok"] for r in rows),
                                "not_deferred": sum(not r["deferred"] for r in rows), "rows": rows}
        if real:
            thresholds = fixtures.m1_thresholds()
            idx = fixtures.index()
            weights = fixtures.mb1_layers()
            _Ptx._shared = None
            objs = {n: build(w, w["kind"]) for n, w in weights.items()}  # all alive: no address reuse
            rows = []
            for i in (utterances if utterances is not None else range(len(idx["utterances"]))):
                entry = idx["utterances"][i]
                inputs = fixtures.utterance_inputs(i)
                for n, w in weights.items():
                    x = inputs[n]
                    defer = pipeline_defers(n)
                    y, deferred = run_layer(objs[n], w["kind"], x, defer)
                    got = api.errors(w["kind"], x, y, api.reference(w["kind"], x, w["codes"], w["scale"], w["bias"], w["K"]),
                                     w["scale"])
                    lim = thresholds[entry["file"]][n]
                    rows.append({"utterance": entry["file"], "layer": n, "M": entry["layers"][n]["M"],
                                 "pipeline_defers": defer, "deferred": deferred, **got, "limits": lim,
                                 "ok": api.within(got, lim) and deferred == defer})
            def worst(sel, metric):
                return max((r[metric] / r["limits"][metric] if r["limits"][metric] > 0 else (0.0 if r[metric] == 0 else float("inf"))
                            for r in sel), default=None)
            d_rows = [r for r in rows if r["pipeline_defers"]]
            out["real"] = {"evaluations": len(rows), "failures": sum(not r["ok"] for r in rows),
                           "deferred_evaluations": len(d_rows), "deferred_failures": sum(not r["ok"] for r in d_rows),
                           "deferral_mismatches": sum(r["deferred"] != r["pipeline_defers"] for r in rows),
                           "worst_ratio": {m: worst(rows, m) for m in ("relative_frobenius", "elementwise_guard")},
                           "worst_ratio_deferred": {m: worst(d_rows, m) for m in ("relative_frobenius", "elementwise_guard")},
                           "rows": rows}
    if out["synthetic"]:
        disc = [r for r in out["synthetic"]["rows"] if max(v["relative_frobenius"] for v in r["m1"].values()) == 0.0]
        out["synthetic"]["discriminating_cases"] = len(disc)
        out["synthetic"]["discriminating_failures"] = sum(not r["ok"] for r in disc)
    return out


DEFAULT_MODULE = "gpu1bit.int8_mma.deferred"


HARNESS_COMPONENTS = (2, 3)   # x3: M1's A3 limits; x2: bitwise deferred == in-kernel (plans/x2-default.md Z5)
CRITERIA = {3: "x3-relative A3 limits", 2: "bitwise deferred == in-kernel"}


def full_scope(components: int = 3) -> dict:
    """The complete qualification scope: every fixture utterance, real and synthetic, at a part count,
    with that part count's criterion (Z5)."""
    from gpu1bit.shared import fixtures
    return {"utterances": list(range(len(fixtures.index()["utterances"]))), "real": True, "synthetic": True,
            "components": components, "criterion": CRITERIA[components]}


def expected_coverage() -> dict:
    """What a complete qualification evaluates (Codex GPU code review c3)."""
    from gpu1bit.shared import fixtures
    names = fixtures.layer_names()
    n = len(fixtures.index()["utterances"])
    return {"utterances": n, "real_evaluations": len(names) * n,
            "deferred_evaluations": sum(pipeline_defers(x) for x in names) * n, "synthetic_cases": len(synthetic_cases())}


def finalize(out: dict, scope: dict) -> dict:
    """passes is True only for a COMPLETE scope with zero failures (Codex GPU c3: a partial run, e.g.
    --no-real --no-synthetic or an utterance subset, is informational and never passes)."""
    out["scope"] = scope
    out["partial"] = scope != full_scope(scope.get("components", 3) if scope.get("components") in CRITERIA else 3)
    out["checks_passed"] = all(part is None or part["failures"] == 0 for part in (out.get("real"), out.get("synthetic")))
    out["passes"] = (not out["partial"]) and out["checks_passed"] and out.get("real") is not None \
        and out.get("synthetic") is not None
    return out


def provider_identity(provider: str, module: str | None) -> dict:
    """The provider as resolved: registry kernel and factory (external), or the built-in definition."""
    if provider == "R-hybrid":
        return {"builtin": "R-hybrid PtxCudaProjection"}
    if provider in ("B-hybrid@frozen", "B-hybrid-b01@frozen"):
        return {"builtin": "gpu1bit.int8_mma.hybrid:HybridProjection",
                "kernel": "int8_mma" if provider == "B-hybrid@frozen" else "int8_mma:b01"}
    entry = importlib.import_module(module).DEFERRED_PROVIDERS[provider]
    f = entry["factory"]
    return {"kernel": entry["kernel"], "factory": f"{getattr(f, '__module__', '?')}:{getattr(f, '__qualname__', repr(f))}",
            "components": int(entry.get("components", 3))}


def _qualify_x2(name: str, module: str | None, utterances: list[int] | None, real: bool, synthetic: bool) -> dict:
    """The x2 qualification: the same real and synthetic case set as x3, criterion `_exact_x2`."""
    from gpu1bit.shared import fixtures
    build, informational = provider_spec(name, module, 2)
    out = {"provider": name, "module": module, "informational": informational, "real": None, "synthetic": None,
           "criterion": "bitwise deferred == in-kernel"}
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode():
        if synthetic:
            rows = []
            for c_i, case in enumerate(synthetic_cases()):
                w, x = case_data(case, 9000 + c_i)
                r = _exact_x2(build(w, case["kind"]), case["kind"], x, True)
                rows.append({"case": case["name"], "M": case["M"], "N": case["N"], "K": case["K"], "kind": case["kind"],
                             "weights": case["weights"], **r, "ok": r["exact"] and r["deferred"]})
            out["synthetic"] = {"cases": len(rows), "failures": sum(not r["ok"] for r in rows),
                                "not_deferred": sum(not r["deferred"] for r in rows),
                                "inexact": sum(not r["exact"] for r in rows), "rows": rows}
        if real:
            idx = fixtures.index()
            weights = fixtures.mb1_layers()
            objs = {n: build(w, w["kind"]) for n, w in weights.items()}
            rows = []
            for i in (utterances if utterances is not None else range(len(idx["utterances"]))):
                entry = idx["utterances"][i]
                inputs = fixtures.utterance_inputs(i)
                for n, w in weights.items():
                    defer = pipeline_defers(n)
                    r = _exact_x2(objs[n], w["kind"], inputs[n], defer)
                    rows.append({"utterance": entry["file"], "layer": n, "M": entry["layers"][n]["M"],
                                 "pipeline_defers": defer, **r, "ok": r["exact"] and r["deferred"] == defer})
            d_rows = [r for r in rows if r["pipeline_defers"]]
            out["real"] = {"evaluations": len(rows), "failures": sum(not r["ok"] for r in rows),
                           "deferred_evaluations": len(d_rows), "deferred_failures": sum(not r["ok"] for r in d_rows),
                           "deferral_mismatches": sum(r["deferred"] != r["pipeline_defers"] for r in rows),
                           "inexact": sum(not r["exact"] for r in rows), "rows": rows}
    return out


def record_fingerprint(provider: str, module: str | None, scope: dict | None = None) -> dict:
    """The fingerprint a qualification report is made under (Codex GPU code review c2), including the
    scope that was run (c3)."""
    from gpu1bit.shared import fingerprint as F
    return F.make("deferred A3", provider=provider, module=module, scope=scope)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    q = sub.add_parser("qualify")
    q.add_argument("--provider", required=True)
    q.add_argument("--module", default=None)
    q.add_argument("--utterances", default="0-14")
    q.add_argument("--no-real", action="store_true")
    q.add_argument("--no-synthetic", action="store_true")
    q.add_argument("--run", default=None, help="write the report to deferred/<run>/ (re-runs under new fingerprints)")
    q.add_argument("--components", type=int, choices=HARNESS_COMPONENTS, default=3,
                   help="activation part count; 2 uses the bitwise deferred == in-kernel criterion (Z5)")
    args = p.parse_args()
    env.require_shared_wrapper()
    a, b = (int(v) for v in args.utterances.split("-"))
    started = time.time()
    with env.gpu_lock(f"H:deferred A3 {args.provider}") as lock:
        utterances = list(range(a, b + 1))
        report = qualify(args.provider, args.module, utterances, not args.no_real, not args.no_synthetic, args.components)
        scope = {"utterances": utterances, "real": not args.no_real, "synthetic": not args.no_synthetic,
                 "components": args.components, "criterion": CRITERIA[args.components]}
        finalize(report, scope)
        report.update(lock_held_s=env.check_hold(lock, margin_s=0), wall_s=time.time() - started, wrapper=env.wrapper(),
                      sources={"deferred_a3.py": env.sha256_file(env.SHARED / "deferred_a3.py")}, module=args.module,
                      provider_identity=provider_identity(args.provider, args.module),
                      fingerprint=record_fingerprint(args.provider, args.module, scope))
        where = ("deferred", args.run) if args.run else ("deferred",)
        out = env.artifacts(*where) / f"{args.provider.replace(':', '_').replace('@', '_at_')}.json"
        out.write_text(json.dumps(report) + "\n")
    summary = {k: report[k] for k in ("provider", "informational", "partial", "checks_passed", "passes")}
    for part in ("synthetic", "real"):
        if report[part]:
            summary[part] = {k: v for k, v in report[part].items() if k != "rows"}
            summary[part]["first_failures"] = [{k: r[k] for k in r if k not in ("m1", "limits")} | {"limits": r["limits"]}
                                               for r in report[part]["rows"] if not r["ok"]][:3]
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
