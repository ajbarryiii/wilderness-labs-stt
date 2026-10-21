"""Contract tests of gpu1bit/runtime.py, the integrated MB1 runtime (phase 3). In-process, one lock
(gpu1bit.tests.run). Fresh-process checks (VRAM, no .nemo / 2-bit files opened): test_runtime_fresh.

Refusals (before any CUDA use): 2-bit parakeet-ternary-v1 exports (M1's and the derived MB1 layout),
a wrong format field, a file whose bytes differ from its manifest SHA-256, a layer with nonzero tail
bits, a missing layer, an unknown kernel, a kernel without B's dot op for the hybrid stack.
End to end, for runtime "hybrid" (the winner) and "provider", with the selected kernel:
  - text, token IDs and timestamps equal to MB1 on R-expanded on all dev timing workloads (A12/A13);
  - no M1 weight kernel executes (instrumented; the same instrumentation sees them in R-expanded and,
    for the hybrid stack, PtxCudaProjection calls in R-hybrid);
  - pipeline CUDA graphs captured for the three dev clip shapes, replayed in both orders and after
    allocator churn: encoder outputs bitwise equal to the captured call, decodes identical, eager
    within M1's capture tolerance (inference.benchmark.capture: atol 1e-6, rtol 1e-5);
  - A5 whole-model audit after warm-up and capture: weight-code storage <= 83,047,219 B.
Override the kernel with GPU1BIT_RUNTIME_KERNEL=<registry name> (default: the selection document's part-3
entry). These are the x3 runtime's contracts and load at parts 3 explicitly, before and after promotion.
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import unittest

import torch

from gpu1bit import runtime
from gpu1bit.shared import env

# The historical x3 contracts: the selection document's part-3 entry, loaded at parts 3 explicitly, so they
# keep testing x3 after promotion (Codex x2 c1, finding 7); the x2 default's are tests/test_x2_*.
KERNEL = os.environ.get("GPU1BIT_RUNTIME_KERNEL") or __import__("gpu1bit.shared.seldoc", fromlist=["x"]).entry(
    runtime.selected(), 3)["kernel"]


def _craft(name: str, *, edit=None, corrupt: str | None = None) -> str:
    """A copy of MB1's export under H/tmp: symlinks to the real files, an edited manifest.json, and
    optionally one file replaced by a copy with one byte appended (its manifest hash kept)."""
    src = env.MB1_EXPORT
    d = env.artifacts("tmp", "runtime-tests", f"{name}-{os.getpid()}")
    shutil.rmtree(d)
    (d / "tokenizer").mkdir(parents=True)
    manifest = json.loads((src / "manifest.json").read_text())
    for rel in manifest["files"]:
        if rel == corrupt:
            data = (src / rel).read_bytes() + b"\n"
            (d / rel).write_bytes(data)
        else:
            (d / rel).symlink_to(src / rel)
    if edit is not None:
        edit(manifest)
    (d / "manifest.json").write_text(json.dumps(manifest))
    return str(d)


class Refusals(unittest.TestCase):
    def test_refuses_ternary_layout_exports(self):
        for d in (env.DERIVED_EXPORT, env.M1_EXPORT):
            with self.subTest(export=str(d)), self.assertRaisesRegex(ValueError, "parakeet-ternary-v1|format"):
                runtime.load_binary_packed(d)

    def test_refuses_wrong_format_field(self):
        d = _craft("format", edit=lambda m: m.update(format="parakeet-binary-v2"))
        with self.assertRaisesRegex(ValueError, "format"):
            runtime.load_binary_packed(d)

    def test_refuses_hash_mismatch(self):
        for rel in ("tokenizer/vocab.txt", "tokenizer/tokenizer.model"):
            with self.subTest(file=rel):
                d = _craft("hash", corrupt=rel)
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    runtime.load_binary_packed(d)

    def test_refuses_tail_bits(self):
        from safetensors import safe_open
        name = "encoder.layers.0.self_attn.linear_out"
        with safe_open(env.MB1_EXPORT / "export.safetensors", framework="pt") as f:
            codes = f.get_tensor(f"{name}.codes")
        # Precondition: claiming K = 1020 turns bits 1020..1023 into tail padding; they are not all 0.
        self.assertTrue(bool(((codes[:, -1] >> 4) != 0).any()))
        d = _craft("tail", edit=lambda m: m["quantized_layers"][name].update(shape=[1024, 1020]))
        with self.assertRaisesRegex(ValueError, "tail"):
            runtime.load_binary_packed(d)

    def test_refuses_missing_layer(self):
        d = _craft("missing", edit=lambda m: m["quantized_layers"].pop("encoder.layers.23.conv.pointwise_conv2"))
        with self.assertRaisesRegex(ValueError, "264"):
            runtime.load_binary_packed(d)

    def test_refuses_unknown_kernel_and_hybrid_without_dot(self):
        with self.assertRaises(KeyError):
            runtime.load_binary_packed(env.MB1_EXPORT, kernel="no-such-kernel")
        with self.assertRaisesRegex(ValueError, "dot"):
            runtime.load_binary_packed(env.MB1_EXPORT, runtime="hybrid", kernel="h_reference")

    def test_validate_export_accepts_mb1(self):
        self.assertEqual(runtime.validate_export(env.MB1_EXPORT)["sha256"], env.MB1_SHA256)


def _decode_item(model, path):
    """(text, tokens, timestamps) of one batch-1 transcription through M1's transcribe."""
    from hf.load_ternary import transcribe as m1_transcribe
    with torch.inference_mode():
        text = m1_transcribe(model, path, batch_size=1)
    return {"text": text, **runtime.decoded(model)}


def _encode(model, rec):
    """model.forward (the pipeline graph cache) under the same strict-FP32 flags as transcribe: the
    cache key includes the TF32 / cuDNN-TF32 flags, so other flags would capture extra entries."""
    import evaluate as ev
    audio, length = ev.audio_batch([rec])
    with torch.inference_mode(), ev.strict_fp32("cuda"):
        e, l = model.forward(input_signal=audio.cuda(), input_signal_length=length.cuda())
        torch.cuda.synchronize()
        return e.clone(), l.clone(), audio, length


class EndToEnd:
    runtime_name = ""

    @classmethod
    def setUpClass(cls):
        from nemo.utils import logging
        from gpu1bit.shared import m1, workloads
        from inference.optimized import enable_optimizations
        logging.setLevel(logging.ERROR)
        doc = workloads.load()
        cls.clips = doc["dev_clips"]
        cls.timing = list({r["id"]: r for r in doc["dev_clips"] + doc["dev_24"]}.values())
        with torch.inference_mode():
            cls.ref = m1.load_derived()   # the REFERENCE only: MB1 on R-expanded (derived 2-bit layout)
            cls.ref.decoding.decoding.decoding_computer.force_cuda_graphs_mode("full_graph")
            enable_optimizations(cls.ref, max_graphs=4, encoder_storage="expanded")
        cls.model = runtime.load_binary_packed(env.MB1_EXPORT, runtime=cls.runtime_name, kernel=KERNEL, parts=3)

    @classmethod
    def tearDownClass(cls):
        from inference.optimized import disable_optimizations
        runtime.close_binary_packed(cls.model)
        assert not hasattr(cls.model, "_gpu1bit_runtime")
        assert cls.model.preprocessor.featurizer.dither == 1e-5
        with torch.inference_mode():
            disable_optimizations(cls.ref)
        del cls.model, cls.ref
        gc.collect()
        torch.cuda.empty_cache()

    def _out(self, name):
        return env.artifacts("runtime-tests") / f"{self.runtime_name}-{KERNEL.replace(':', '_')}-{name}.json"

    def test_dev_timing_workloads_equal_r_expanded(self):
        rows, bad = [], []
        for rec in self.timing:
            got = {}
            for label, m in (("reference", self.ref), ("runtime", self.model)):
                m.forward.entries.clear()
                got[label] = _decode_item(m, rec["audio_filepath"])
            rows.append({"id": rec["id"], **got, "identical": got["reference"] == got["runtime"]})
            if not rows[-1]["identical"]:
                bad.append(rec["id"])
        self._out("timing").write_text(json.dumps({"kernel": KERNEL, "rows": rows}, indent=1) + "\n")
        self.assertEqual(len(rows), 24)
        self.assertEqual(bad, [], "text/tokens/timestamps differ from MB1 on R-expanded")
        api_texts = runtime.transcribe(self.model, [r["audio_filepath"] for r in self.clips])
        self.assertEqual(api_texts, [next(x["runtime"]["text"] for x in rows if x["id"] == r["id"]) for r in self.clips])
        self.assertIsInstance(runtime.transcribe(self.model, self.clips[0]["audio_filepath"]), str)

    def test_no_m1_weight_kernel_executes(self):
        from gpu1bit.tests.test_provider import count_m1_weight_kernels
        import inference.encoder4 as e4
        import inference.ptx_cuda as ptx
        calls = {}
        saved = []
        for owner, cls_name in ((e4, "ExpandedProjection"), (ptx, "PtxCudaProjection")):
            klass = getattr(owner, cls_name)
            original = klass.__call__

            def wrapped(self, *a, _o=original, _n=cls_name, **k):
                calls[_n] = calls.get(_n, 0) + 1
                return _o(self, *a, **k)
            klass.__call__ = wrapped
            saved.append((klass, original))
        try:
            self.model.forward.entries.clear()
            with count_m1_weight_kernels() as counts:
                for rec in self.clips:
                    for _ in range(2):  # capture (eager warm-ups + capture), then replay
                        _decode_item(self.model, rec["audio_filepath"])
            runtime_counts = (dict(counts), dict(calls))
            calls.clear()
            self.ref.forward.entries.clear()
            with count_m1_weight_kernels() as ref_counts:
                _decode_item(self.ref, self.clips[0]["audio_filepath"])
            control = {"r_expanded": (dict(ref_counts), dict(calls))}
            if self.runtime_name == "hybrid":
                from gpu1bit.shared import m1
                calls.clear()
                with torch.inference_mode():
                    hybrid = m1.load_derived()
                    hybrid.decoding.decoding.decoding_computer.force_cuda_graphs_mode("full_graph")
                    with m1.baseline("R-hybrid", hybrid):
                        _decode_item(hybrid, self.clips[0]["audio_filepath"])
                control["r_hybrid"] = dict(calls)
                del hybrid
        finally:
            for klass, original in saved:
                klass.__call__ = original
        self._out("m1-kernels").write_text(json.dumps({"runtime": runtime_counts, "controls": control}, indent=1) + "\n")
        self.assertEqual(runtime_counts, ({}, {}), f"M1 weight kernels ran in the runtime: {runtime_counts}")
        self.assertGreater(sum(control["r_expanded"][0].values()), 0)
        self.assertGreater(control["r_expanded"][1].get("ExpandedProjection", 0), 0)
        if self.runtime_name == "hybrid":
            self.assertGreater(control["r_hybrid"].get("PtxCudaProjection", 0), 0)

    def test_graph_capture_and_replay_three_shapes(self):
        m = self.model
        m.forward.entries.clear()
        first = {}
        for rec in self.clips:
            e, l, audio, length = _encode(m, rec)
            first[rec["id"]] = (e, l, _decode_item(m, rec["audio_filepath"]))
        self.assertEqual(len(m.forward.entries), 3)
        shapes = sorted(k[0][1] for k in m.forward.entries)
        self.assertEqual(shapes, sorted(r["samples"] for r in self.clips))
        for order in (self.clips[::-1], self.clips):
            junk = torch.empty(256 * 1024 * 1024 // 4, device="cuda")  # allocator churn between replays
            del junk
            for rec in order:
                e, l, audio, length = _encode(m, rec)
                e0, l0, d0 = first[rec["id"]]
                self.assertTrue(torch.equal(e, e0) and torch.equal(l, l0), f"replay differs: {rec['id']}")
                self.assertEqual(_decode_item(m, rec["audio_filepath"]), d0)
        self.assertEqual(len(m.forward.entries), 3)
        import evaluate as ev
        with torch.inference_mode(), ev.strict_fp32("cuda"):
            for rec in self.clips:
                e0, l0, _ = first[rec["id"]]
                audio, length = ev.audio_batch([rec])
                eager, el = m.forward.model_forward(input_signal=audio.cuda(), input_signal_length=length.cuda())
                torch.testing.assert_close(eager, e0, atol=1e-6, rtol=1e-5)
                self.assertTrue(torch.equal(el, l0))

    def test_whole_model_storage_audit(self):
        from gpu1bit.shared import audit, fixtures
        from gpu1bit.shared.selection import reclassify_decoder_side
        m = self.model
        m.forward.entries.clear()
        for rec in self.clips:
            for _ in range(2):
                _decode_item(m, rec["audio_filepath"])
        self.assertEqual(len(m.forward.entries), 3)
        scales = {n: w["scale"] for n, w in fixtures.mb1_layers().items()}
        report = audit.audit(m, list(scales), {"provider": runtime.runtime_of(m).provider}, scales=scales)
        reclassify_decoder_side(report)
        self._out("audit").write_text(json.dumps(report, indent=1, default=str) + "\n")
        self.assertTrue(report["within_budget"], f"{report['weight_code_bytes']} > {audit.CODE_BUDGET_BYTES}")
        self.assertGreaterEqual(report["scale_bytes"], 1_769_472)


for _name in runtime.RUNTIMES:
    _cls = f"EndToEnd_{_name}"
    globals()[_cls] = type(_cls, (EndToEnd, unittest.TestCase), {"runtime_name": _name})

if __name__ == "__main__":
    unittest.main()
