"""Contract tests (CPU): the stage 5 x1 loader training/keat_deployed.py (keat-prime plan "Scoring and gate", "The
loader").

gpu1bit.runtime_m1.build_stack runs for real; only what needs the RTX 5090 is faked: inference.runtime.load_packed
(returns a stand-in model with a featurizer and a decoding computer) and inference.optimized.enable_optimizations /
disable_optimizations (the stand-in constructs the stack's one ExpandedProjection('triton_col_tuned') as the real
transform does, through build_stack's factory, and keeps it as the stack's provider).
- x1: exactly one provider, a fulldev.PartCount that applies n = 1 on every call, including the stack's hard-coded
  components = 3, with the calls recorded; anything else is refused;
- x3: the same builder at 3 parts with no adapter;
- dither 0 and pad_to 0 while loaded, restored on close (also when disabling the stack fails);
- an arbitrary export directory is passed to the builder as given (resolved);
- an interpreter without the gpu1bit wrapper or without NVRTC is refused before any model work;
- scoring is stage 4's deployed.score_records (one scorer for both stages).
The real stack on M1's export is in test_keat_deployed_gpu.py.
"""
from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from training import deployed  # puts finetune/parakeet-binary (gpu1bit) on sys.path
from training import keat_deployed as tdep

import inference.encoder4 as e4  # noqa: E402  finetune/parakeet-ternary/inference
import inference.optimized  # noqa: E402
import inference.runtime  # noqa: E402
from gpu1bit.shared import env, fulldev  # noqa: E402

_MADE: list[Path] = []
GOOD = {"wrapper": deployed.WRAPPER, "nvrtc_on_library_path": True, "executable": "python"}


class Featurizer:
    def __init__(self):
        self.dither, self.pad_to = 1e-5, 16


class Computer:
    cuda_graphs_mode = None

    def force_cuda_graphs_mode(self, mode):
        self.cuda_graphs_mode = mode


class StandIn(torch.nn.Module):
    """What load_packed returns, reduced to what the loader and build_stack touch."""

    def __init__(self, directory: Path):
        super().__init__()
        self.directory = directory
        self.preprocessor = SimpleNamespace(featurizer=Featurizer())
        self.decoding = SimpleNamespace(decoding=SimpleNamespace(decoding_computer=Computer()))
        self.stack_provider = None
        self.disabled = 0
        self.eval()


class Recorder:
    """A probe: stands in for the ExpandedProjection's arithmetic and records the components it is called with."""

    def __init__(self, inner):
        self.inner, self.components = inner, []

    def __call__(self, x, mod, conv=False, components=3, activation="", norm=None, config=None, input_activation=""):
        self.components.append(components)
        return x


class Fakes:
    def __init__(self, providers: int = 1, fail_disable: bool = False):
        self.loaded, self.providers, self.fail_disable = [], providers, fail_disable
        self.probes: list[Recorder] = []

    def load_packed(self, directory, device="cuda", **kw):
        self.loaded.append(Path(directory))
        return StandIn(Path(directory))

    def enable(self, model, *, max_graphs=4, encoder_storage="packed", **kw):
        if (max_graphs, encoder_storage) != (4, "expanded"):
            raise AssertionError("the deployed stack is enable_optimizations(max_graphs=4, encoder_storage='expanded')")
        for _ in range(self.providers):
            model.stack_provider = e4.ExpandedProjection("triton_col_tuned")
        model._packed_optimization = True

    def disable(self, model):
        model.disabled += 1
        if self.fail_disable:
            raise RuntimeError("synchronize failed")

    def probe(self, inner):
        rec = Recorder(inner)
        self.probes.append(rec)
        return rec

    @contextlib.contextmanager
    def active(self, facts=GOOD):
        with mock.patch.object(inference.runtime, "load_packed", self.load_packed), \
                mock.patch.object(inference.optimized, "enable_optimizations", self.enable), \
                mock.patch.object(inference.optimized, "disable_optimizations", self.disable), \
                mock.patch.object(env, "wrapper", lambda: dict(facts)):
            yield


def export_dir() -> Path:
    """An export directory at an arbitrary path, with a manifest (the stand-in load_packed reads no weights)."""
    import json
    import relabel_paths
    base = relabel_paths.ARTIFACTS / "tmp"
    base.mkdir(parents=True, exist_ok=True)
    top = Path(tempfile.mkdtemp(prefix="keat-x1-", dir=base))
    _MADE.append(top)
    d = top / "any" / "export"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"format": "parakeet-ternary-v1", "sha256": "f" * 64}))
    return d


class X1(unittest.TestCase):
    def test_one_partcount_applying_one_part_on_every_call(self):
        fakes = Fakes()
        with fakes.active():
            model = tdep.load_x1(export_dir(), probe=fakes.probe)
            try:
                prov = model.stack_provider
                self.assertIs(tdep.provider(model), prov)
                self.assertIsInstance(prov, fulldev.PartCount)
                self.assertEqual(prov.n, 1)
                self.assertEqual(len(fakes.probes), 1)
                x = torch.zeros(1, 4, 8)
                for requested in (3, 3, 2, 1):            # 3: layer_fusions4's hard-coded pointwise_conv2 count
                    prov(x, None, components=requested)
                self.assertEqual(fakes.probes[0].components, [1, 1, 1, 1])
                self.assertEqual(tdep.part_calls(model), {"1->1": 1, "2->1": 1, "3->1": 2})
                facts = tdep.info(model)
                self.assertEqual((facts["parts"], facts["provider"]), (1, "PartCount"))
                self.assertEqual(facts["part_calls"], {"1->1": 1, "2->1": 1, "3->1": 2})
                self.assertEqual(facts["decoder_cuda_graphs_mode"], "full_graph")
                self.assertEqual(facts["wrapper"]["wrapper"], deployed.WRAPPER)
            finally:
                tdep.close(model)

    def test_check_refuses_a_build_that_is_not_one_partcount_at_one(self):
        from gpu1bit import runtime_m1
        fakes = Fakes()

        def built(adapter):
            def build_stack(export, parts, wrap=None):
                model = fakes.load_packed(export)
                inner = e4.ExpandedProjection("triton_col_tuned")
                hooked = wrap(inner)
                return model, (inner, adapter(hooked))
            return build_stack

        for adapter in (lambda p: p, lambda p: fulldev.PartCount(p, 2), lambda p: fulldev.PartCount(p, 3)):
            with fakes.active(), mock.patch.object(runtime_m1, "build_stack", built(adapter)):
                with self.assertRaises(RuntimeError):
                    tdep.load_x1(export_dir())
        with fakes.active(), mock.patch.object(runtime_m1, "build_stack", built(lambda p: fulldev.PartCount(p, 1))):
            tdep.close(tdep.load_x1(export_dir()))
            with self.assertRaises(RuntimeError):
                tdep.load_x3(export_dir())

    def test_two_providers_are_refused(self):
        fakes = Fakes(providers=2)
        with fakes.active():
            with self.assertRaises((RuntimeError, AssertionError)):
                tdep.load_x1(export_dir())

    def test_arbitrary_export_directory(self):
        fakes = Fakes()
        d = export_dir()
        with fakes.active():
            model = tdep.load_x1(d)
            try:
                self.assertEqual(fakes.loaded, [d.resolve()])
                self.assertEqual(tdep.info(model)["export"], str(d.resolve()))
                self.assertEqual(tdep.info(model)["export_sha256"], "f" * 64)
            finally:
                tdep.close(model)


class X3(unittest.TestCase):
    def test_same_builder_at_three_without_the_adapter(self):
        fakes = Fakes()
        with fakes.active():
            model = tdep.load_x3(export_dir(), probe=fakes.probe)
            try:
                prov = model.stack_provider
                self.assertIs(tdep.provider(model), prov)
                self.assertNotIsInstance(prov, fulldev.PartCount)
                self.assertIs(prov, fakes.probes[0])
                prov(torch.zeros(1, 2, 8), None, components=3)
                self.assertEqual(fakes.probes[0].components, [3])
                facts = tdep.info(model)
                self.assertEqual(facts["parts"], 3)
                self.assertIsNone(facts["part_calls"])
            finally:
                tdep.close(model)


class Frontend(unittest.TestCase):
    def test_dither_and_pad_to_zero_while_loaded_and_restored_on_close(self):
        for loader in (tdep.load_x1, tdep.load_x3):
            fakes = Fakes()
            with fakes.active():
                model = loader(export_dir())
                f = model.preprocessor.featurizer
                self.assertEqual((f.dither, f.pad_to), (0.0, 0))
                self.assertEqual(tdep.info(model)["featurizer_saved"], {"dither": 1e-5, "pad_to": 16})
                tdep.close(model)
                self.assertEqual((f.dither, f.pad_to), (1e-5, 16))
                self.assertEqual(model.disabled, 1)

    def test_restored_even_when_disabling_the_stack_fails(self):
        fakes = Fakes(fail_disable=True)
        with fakes.active():
            model = tdep.load_x1(export_dir())
            with self.assertRaises(RuntimeError):
                tdep.close(model)
            f = model.preprocessor.featurizer
            self.assertEqual((f.dither, f.pad_to), (1e-5, 16))


class Interpreter(unittest.TestCase):
    def test_without_nvrtc_or_the_wrapper_is_refused_before_any_model_work(self):
        for facts in ({**GOOD, "nvrtc_on_library_path": False}, {**GOOD, "wrapper": "unknown"}):
            for loader in (tdep.load_x1, tdep.load_x3):
                fakes = Fakes()
                with self.subTest(facts=facts, loader=loader.__name__), fakes.active(facts):
                    with self.assertRaises(RuntimeError):
                        loader(export_dir())
                    self.assertEqual(fakes.loaded, [])

    def test_one_scorer_for_both_stages(self):
        self.assertIs(tdep.score_records, deployed.score_records)
        self.assertIs(tdep.require_wrapper, deployed.require_wrapper)


def tearDownModule():
    import shutil
    for d in _MADE:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
