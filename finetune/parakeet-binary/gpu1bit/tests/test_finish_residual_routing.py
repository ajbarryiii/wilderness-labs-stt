"""The exact deferred path must be finished by B's router in every code path of the deployed stack
(coordinator 2026-10-20, B's warning of 13:11): no module, closure or default argument may hold an
import-time reference to M1's ORIGINAL inference.row_fusion.finish_residual, except B's router module
(gpu1bit.int8_mma.row_exact), which delegates non-exact values to it.

For the hybrid stack named by EXACT_SPEC (B's fixed stack; also the deployed one once selected.json
points at it) and, if different, the deployed selected.json stack: build gpu1bit.runtime's hybrid model,
transcribe the three dev clips twice (eager warm-up + capture, then replay), then
  - the router is installed: inference.row_fusion.finish_residual is not M1's original;
  - gc.get_referrers(original) holds nothing outside row_exact (module globals of any other module,
    closure cells, defaults, containers): each offender is named;
  - the provider actually returned ExactDeferredProjection values (whose .partial raises, so a
    successful transcription proves that the router, not M1's original, finished them).
    timeout 600 gpu1bit/shared/python -m gpu1bit.tests.run --label H:routing gpu1bit.tests.test_finish_residual_routing
"""
from __future__ import annotations

import gc
import types
import unittest

import torch

from gpu1bit import runtime
from gpu1bit.shared import env

EXACT_SPEC = {"provider": "gpu1bit.int8_mma.hybrid_exact:HybridProjectionExact",
              "pipeline": "gpu1bit.int8_mma.hybrid_exact:exact_hybrid_pipeline",
              "imports": ["gpu1bit.int8_mma.row_exact"]}
ROUTER_MODULE = "gpu1bit.int8_mma.row_exact"


def m1_original():
    """M1's finish_residual function object, identified by its code (defined in inference/row_fusion.py)."""
    found = [o for o in gc.get_objects() if isinstance(o, types.FunctionType) and o.__name__ == "finish_residual"
             and o.__code__.co_filename.endswith("inference/row_fusion.py")]
    if len(found) != 1:
        raise AssertionError(f"expected exactly one M1 finish_residual, found {len(found)}")
    return found[0]


def offenders(original) -> list[str]:
    import sys
    router = sys.modules.get(ROUTER_MODULE)
    router_dict = vars(router) if router else None
    router_cells = set()
    if router:
        for f in vars(router).values():
            if isinstance(f, types.FunctionType) and f.__closure__:
                router_cells.update(id(c) for c in f.__closure__)
    module_of = {id(vars(m)): name for name, m in list(sys.modules.items()) if m is not None and hasattr(m, "__dict__")}
    out = []
    for ref in gc.get_referrers(original):
        if ref is router_dict or id(ref) in router_cells or isinstance(ref, types.FrameType):
            continue
        if isinstance(ref, dict) and id(ref) in module_of:
            out.append(f"module global of {module_of[id(ref)]}")
        elif type(ref).__name__ == "cell":
            owners = [f"{g.__module__}.{g.__qualname__}" for t in gc.get_referrers(ref) if isinstance(t, tuple)
                      for g in gc.get_referrers(t) if isinstance(g, types.FunctionType)]
            if owners and all(o.startswith(ROUTER_MODULE + ".") for o in owners):
                continue  # the router's own closure (e.g. a dispatcher built inside install())
            out.append(f"closure cell of {owners or 'unknown function'}")
        elif isinstance(ref, tuple):
            owners = [f"{g.__module__}.{g.__qualname__}" for g in gc.get_referrers(ref) if isinstance(g, types.FunctionType)]
            if owners and all(o.startswith(ROUTER_MODULE + ".") for o in owners):
                continue
            out.append(f"defaults/closure tuple of {owners or 'unknown'}")
        elif isinstance(ref, dict) and any(isinstance(f, types.FunctionType) and f.__dict__ is ref
                                           for f in gc.get_referrers(ref)):
            owner = next(f for f in gc.get_referrers(ref) if isinstance(f, types.FunctionType) and f.__dict__ is ref)
            if owner.__module__ == ROUTER_MODULE:
                continue  # an attribute of the router function itself (it delegates non-exact values)
            out.append(f"attribute {[k for k, v in ref.items() if v is original]} of function "
                       f"{owner.__module__}.{owner.__qualname__}")
        elif isinstance(ref, (list, dict)) and not any(isinstance(r, types.FrameType) for r in gc.get_referrers(ref)):
            keys = [k for k, v in ref.items() if v is original] if isinstance(ref, dict) else []
            holders = []
            for r in gc.get_referrers(ref):
                if hasattr(r, "__dict__") and getattr(r, "__dict__", None) is ref:
                    holders.append(f"__dict__ of {type(r).__module__}.{type(r).__qualname__}")
                elif isinstance(r, types.FunctionType):
                    holders.append(f"function {r.__module__}.{r.__qualname__}")
                elif isinstance(r, dict) and id(r) in module_of:
                    holders.append(f"global of {module_of[id(r)]} as {[k for k, v in r.items() if v is ref]}")
                else:
                    holders.append(type(r).__name__)
            out.append(f"container {type(ref).__name__} keys {keys} held by {holders[:4]}")
    return out


class Routing:
    spec: dict = {}

    def test_router_finishes_every_exact_deferred_value(self):
        from gpu1bit.shared import workloads
        import evaluate as ev
        from nemo.utils import logging
        logging.setLevel(logging.ERROR)
        model = runtime.load_binary_packed(env.MB1_EXPORT, runtime="hybrid", hybrid=self.spec, parts=3)   # the x3 stack
        try:
            prov = runtime.runtime_of(model).provider
            from gpu1bit.int8_mma.row_exact import ExactDeferredProjection
            counts = {"exact": 0, "other_deferred": 0}
            call = type(prov).__call__

            def counting(self_, *a, **k):
                v = call(self_, *a, **k)
                if isinstance(v, ExactDeferredProjection):
                    counts["exact"] += 1
                elif type(v).__name__ == "DeferredProjection":
                    counts["other_deferred"] += 1
                return v
            type(prov).__call__ = counting
            try:
                texts = []
                for rec in workloads.load()["dev_clips"]:
                    model.forward.entries.clear()
                    texts.append([runtime.transcribe(model, rec["audio_filepath"]) for _ in range(2)])
            finally:
                type(prov).__call__ = call
            import inference.row_fusion as rf
            original = m1_original()
            self.assertIsNot(rf.finish_residual, original, "the router is not installed in inference.row_fusion")
            bad = offenders(original)
            self.assertEqual(bad, [], f"import-time references to M1's original finish_residual: {bad}")
            self.assertGreater(counts["exact"], 0, "no ExactDeferredProjection was produced: routing not exercised")
            # Not vacuous: a planted stale module global and a planted closure must both be reported.
            import sys
            stale = types.ModuleType("gpu1bit_stale_reference_probe")
            stale.finish_residual = original
            sys.modules[stale.__name__] = stale

            def make(f):
                return lambda *a: f(*a)
            planted = make(original)
            try:
                found = offenders(original)
                self.assertTrue(any("gpu1bit_stale_reference_probe" in o for o in found), found)
                self.assertTrue(any("closure" in o for o in found), found)
            finally:
                del sys.modules[stale.__name__], stale, planted
            self.assertTrue(all(t[0] == t[1] and t[0] for t in texts))
        finally:
            runtime.close_binary_packed(model)
            del model
            gc.collect()
            torch.cuda.empty_cache()


class RoutingFixedStack(Routing, unittest.TestCase):
    spec = EXACT_SPEC


_deployed = __import__("gpu1bit.shared.seldoc", fromlist=["x"]).entry(runtime.selected(), 3).get("hybrid")   # the x3 entry
if _deployed and _deployed != EXACT_SPEC:
    RoutingDeployedStack = type("RoutingDeployedStack", (Routing, unittest.TestCase), {"spec": _deployed})

if __name__ == "__main__":
    unittest.main()
