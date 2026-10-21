"""A2 provider seam and A5 whole-model storage audit, per registered implementation.

1. Dispatch coverage and call-sequence parity: MB1 through `load_binary_model` +
   `enable_binary_optimizations` (eager, max_graphs=0) on a dev clip makes exactly the same sequence
   of projection calls as MB1 (derived export) on R-expanded (layer, N, K, conv, components,
   activation, norm, input_activation, input shape and strides); every call reaches the candidate;
   no M1 weight kernel (R-bf16x3 _matmul/_gemv, INT8 _int8_dot, ExpandedProjection, int8_matmul)
   executes, while the same instrumentation sees them in R-expanded; no fused-QKV member is called
   alone; encoder output finite, lengths equal, rel. Frobenius <= 1e-3 on valid frames (A12 bound).
2. A5 audit after warm-up AND CUDA-graph capture of the full pipeline on the three dev clips:
   weight-code storage <= 83,047,219 B; scales reported separately. R-expanded's audit (sanity)
   must exceed the budget.
"""
from __future__ import annotations

import contextlib
import gc
import json
import unittest

import torch

from gpu1bit.shared import audit, env, registry

KEYS = ("layer", "N", "K", "conv", "components", "activation", "norm", "input_activation", "x_shape", "x_stride")


@contextlib.contextmanager
def count_m1_weight_kernels():
    """Counts launches of M1's weight-reading kernels/ops while active."""
    import inference.encoder4 as e4
    import inference.fused_kernels as fk
    import inference.kernels as k
    counts = {}
    saved = []

    def wrap_jit(name, fn):
        original = fn.run

        def run(*a, **kw):
            counts[name] = counts.get(name, 0) + 1
            return original(*a, **kw)
        fn.run = run
        saved.append((fn, "run", None))

    for name, fn in (("kernels._matmul", k._matmul), ("kernels._gemv", k._gemv), ("fused_kernels._int8_dot", fk._int8_dot),
                     ("encoder4._stacked_dot", e4._stacked_dot)):
        wrap_jit(name, fn)
    try:
        yield counts
    finally:
        for fn, attr, _ in saved:
            del fn.__dict__[attr]


def module_scales(model):
    from gpu1bit.shared import fixtures
    return {n: w["scale"] for n, w in fixtures.mb1_layers().items()}


class ProviderSeam:
    impl_name = ""
    impl = None

    def setUp(self):
        if isinstance(self.impl, Exception):
            self.fail(str(self.impl))

    def test_call_sequence_and_dispatch(self):
        import evaluate as ev
        from nemo.utils import logging
        from inference.optimized import enable_optimizations, disable_optimizations
        from gpu1bit.shared import m1, provider as P, workloads
        logging.setLevel(logging.ERROR)
        clip = workloads.load()["dev_clips"][0]
        with torch.inference_mode(), ev.strict_fp32("cuda"):
            ref = m1.load_derived()
            cand, table = P.load_binary_model(env.MB1_EXPORT, self.impl)
            audio, length = ev.audio_batch([clip])
            audio, length = audio.cuda(), length.cuda()
            with ev.inference_settings(ref), ev.inference_settings(cand):
                enable_optimizations(ref, max_graphs=0, encoder_storage="expanded")
                prov = P.BinaryProjection(table, P.module_names(cand))
                P.enable_binary_optimizations(cand, prov, max_graphs=0)
                try:
                    projections = [o for o in gc.get_objects() if type(o).__name__ == "ExpandedProjection"]
                    self.assertEqual(len(projections), 1)
                    rec = P.r_expanded_recorder(ref, projections[0])
                    with count_m1_weight_kernels() as ref_counts:
                        f, fl = ref.preprocessor(input_signal=audio, length=length)
                        e_ref, l_ref = ref.encoder(audio_signal=f, length=fl)
                        torch.cuda.synchronize()
                    prov.record = []
                    with count_m1_weight_kernels() as cand_counts:
                        f2, fl2 = cand.preprocessor(input_signal=audio, length=length)
                        e_c, l_c = cand.encoder(audio_signal=f2, length=fl2)
                        torch.cuda.synchronize()
                finally:
                    disable_optimizations(ref)
                    disable_optimizations(cand)
        self.assertGreater(sum(ref_counts.values()), 0, "instrumentation saw no M1 kernel in R-expanded")
        self.assertEqual(cand_counts, {}, f"M1 weight kernels ran in the candidate: {cand_counts}")
        self.assertEqual(len(prov.record), len(rec))
        self.assertEqual(len(rec), 24 * 9)
        for i, (a, b) in enumerate(zip(rec, prov.record)):
            self.assertEqual({k: a[k] for k in KEYS}, {k: b[k] for k in KEYS}, f"call {i}")
        self.assertEqual(sum(getattr(m, "unfused_calls", 0) for m in cand.modules()), 0)
        self.assertTrue(torch.equal(l_ref, l_c))
        self.assertTrue(torch.isfinite(e_c).all())
        t = int(l_ref[0])
        rel = ((e_c[..., :t] - e_ref[..., :t]).norm() / e_ref[..., :t].norm()).item()
        self.assertLessEqual(rel, 1e-3)
        out = env.artifacts("provider") / f"{self.impl_name.replace(':', '_')}-sequence.json"
        out.write_text(json.dumps({"impl": self.impl_name, "calls": len(rec), "fused_op_calls":
                                   sum(r["fused_op"] for r in prov.record), "composed_calls": prov.composed_calls,
                                   "encoder_rel_frobenius_vs_r_expanded": rel,
                                   "encoder_max_abs_vs_r_expanded": (e_c[..., :t] - e_ref[..., :t]).abs().max().item(),
                                   "r_expanded_kernel_counts": ref_counts, "clip": clip["id"]}, indent=1) + "\n")

    def test_whole_model_storage_audit(self):
        import evaluate as ev
        from nemo.utils import logging
        from inference.optimized import disable_optimizations
        from gpu1bit.shared import fixtures, provider as P, workloads
        logging.setLevel(logging.ERROR)
        clips = workloads.load()["dev_clips"]
        scales = {n: w["scale"] for n, w in fixtures.mb1_layers().items()}
        with torch.inference_mode(), ev.strict_fp32("cuda"):
            cand, table = P.load_binary_model(env.MB1_EXPORT, self.impl)
            with ev.inference_settings(cand):
                prov = P.BinaryProjection(table, P.module_names(cand))
                P.enable_binary_optimizations(cand, prov, max_graphs=4)
                try:
                    cand.decoding.decoding.decoding_computer.force_cuda_graphs_mode("full_graph")
                    for clip in clips:
                        audio, length = ev.audio_batch([clip])
                        for _ in range(2):  # first call warms and captures; second replays
                            cand(input_signal=audio.cuda(), input_signal_length=length.cuda())
                    torch.cuda.synchronize()
                    self.assertEqual(len(cand.forward.entries), len(clips), "pipeline graphs not captured")
                    names = [n for n in scales]
                    report = audit.audit(cand, names, {"provider": prov}, scales=scales)
                finally:
                    disable_optimizations(cand)
        out = env.artifacts("audit") / f"{self.impl_name.replace(':', '_')}.json"
        out.write_text(json.dumps(report, indent=1, default=str) + "\n")
        self.assertTrue(report["within_budget"], f"weight-code storage {report['weight_code_bytes']} > "
                        f"{audit.CODE_BUDGET_BYTES}; largest: {report['weights'][:5]}")
        self.assertGreater(report["weight_code_bytes"], 0)


class AuditSanity(unittest.TestCase):
    """The audit must see R-expanded's 2-bit buffers, fused-QKV copies and INT8 cache (over budget)."""

    def test_r_expanded_is_over_budget(self):
        import evaluate as ev
        from nemo.utils import logging
        from gpu1bit.shared import fixtures, m1, workloads
        logging.setLevel(logging.ERROR)
        clip = workloads.load()["dev_clips"][0]
        scales = {n: w["scale"] for n, w in fixtures.mb1_layers().items()}
        with torch.inference_mode(), ev.strict_fp32("cuda"):
            model = m1.load_derived()
            with ev.inference_settings(model), m1.baseline("R-expanded", model):
                model.decoding.decoding.decoding_computer.force_cuda_graphs_mode("full_graph")
                audio, length = ev.audio_batch([clip])
                for _ in range(2):
                    model(input_signal=audio.cuda(), input_signal_length=length.cuda())
                torch.cuda.synchronize()
                projection = [o for o in gc.get_objects() if type(o).__name__ == "ExpandedProjection"][0]
                report = audit.audit(model, list(scales), {"provider": projection}, scales=scales)
        out = env.artifacts("audit") / "R-expanded.json"
        out.write_text(json.dumps(report, indent=1, default=str) + "\n")
        self.assertFalse(report["within_budget"])
        # 2-bit packed_t (150,994,944) + fused QKV 2-bit copies (24 x 786,432) + INT8 cache (603,979,776)
        self.assertGreaterEqual(report["weight_code_bytes"], 150_994_944 + 603_979_776)
        self.assertGreaterEqual(report["scale_bytes"], 1_769_472)


def _register():
    for name, impl in registry.discover().items():
        cls_name = "ProviderSeam_" + "".join(c if c.isalnum() else "_" for c in name)
        globals()[cls_name] = type(cls_name, (ProviderSeam, unittest.TestCase), {"impl_name": name, "impl": impl})


_register()

if __name__ == "__main__":
    unittest.main()
