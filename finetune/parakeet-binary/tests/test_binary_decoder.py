"""Review round 1, finding 2 (plans/pilot.md "Review round 1"): every model the pilot decodes with
(teacher, student, rebuilt export) uses NeMo's greedy TDT decoder in the CUDA-graph mode M1 used,
"no_while_loops" (native PyTorch CUDA graphs, M1's fallback from the full CUDA-graph decoder),
and anything else is refused; the interpreter wrapper drops NVRTC from the library path."""
from __future__ import annotations

import os
import subprocess
import unittest

import binary_fixtures as fx
import torch

import binary_paths
import binary_train


def computer(model):
    return model.decoding.decoding.decoding_computer


class DecoderModeTest(unittest.TestCase):
    def test_pin_sets_and_verifies_the_native_pytorch_graph_mode(self) -> None:
        self.assertEqual(binary_train.DECODER_GRAPHS_MODE, "no_while_loops")
        model = fx.tiny_model(seed=61)
        computer(model).force_cuda_graphs_mode("full_graph")  # as NeMo picks when NVRTC is found
        self.assertEqual(binary_train.pin_decoder_graphs_mode(model, "teacher"), "no_while_loops")
        self.assertEqual(computer(model).cuda_graphs_mode.value, "no_while_loops")
        self.assertEqual(binary_train.require_decoder_graphs_mode(model, "teacher"), "no_while_loops")

    def test_any_other_mode_is_refused(self) -> None:
        model = fx.tiny_model(seed=62)
        binary_train.pin_decoder_graphs_mode(model, "student")
        for mode in ("full_graph", "no_graphs", None):
            with self.subTest(mode=mode):
                computer(model).force_cuda_graphs_mode(mode)
                with self.assertRaises(binary_train.DecoderModeError):
                    binary_train.require_decoder_graphs_mode(model, "student")
        no_graphs = fx.tiny_model(seed=63, cuda_graph_decoder=False)  # not M1's decoder config
        with self.assertRaises(binary_train.DecoderModeError):
            binary_train.pin_decoder_graphs_mode(no_graphs, "teacher")

    def test_evaluator_pins_every_model_it_decodes_and_records_the_mode(self) -> None:
        model = fx.tiny_model(seed=64)
        computer(model).force_cuda_graphs_mode("full_graph")
        evaluator = binary_train.DecoderPinnedEvaluator(fx.StubEvaluator())
        self.assertIs(evaluator.subsets, evaluator.inner.subsets)  # train.reconstruction reads these
        evaluator.subset(model)
        self.assertEqual(computer(model).cuda_graphs_mode.value, "no_while_loops")
        evaluator.full(model)
        self.assertEqual(evaluator.modes, {"dev-subset": "no_while_loops", "full-dev": "no_while_loops"})

    def test_teacher_mode_is_verified_around_every_labeling(self) -> None:
        teacher = fx.tiny_model(seed=65)
        binary_train.pin_decoder_graphs_mode(teacher, "teacher")
        calls = []

        def label(model, *args, **kwargs):
            calls.append(1)
            return "labels"

        checked = binary_train.verified_label_fn(label)
        self.assertEqual(checked(teacher, torch.zeros(1)), "labels")
        computer(teacher).force_cuda_graphs_mode("full_graph")  # something re-enabled the full graph
        with self.assertRaises(binary_train.DecoderModeError):
            checked(teacher, torch.zeros(1))
        self.assertEqual(len(calls), 1)  # refused before labeling in the other mode


class WrapperTest(unittest.TestCase):
    def test_python_wrapper_drops_nvrtc_from_the_library_path(self) -> None:
        nvrtc, plain = fx.TEST_ROOT / "fake-nvrtc", fx.TEST_ROOT / "plain-lib"
        for d in (nvrtc, plain):
            d.mkdir(parents=True, exist_ok=True)
        (nvrtc / "libnvrtc.so.12").write_bytes(b"")
        env = {**os.environ, "LD_LIBRARY_PATH": f"{nvrtc}:{plain}"}
        out = subprocess.run([str(binary_paths.HERE / "python"), "-c",
                              "import os; print(os.environ['LD_LIBRARY_PATH'])"],
                             env=env, capture_output=True, text=True, check=True)
        entries = out.stdout.strip().split(":")
        self.assertNotIn(str(nvrtc), entries)
        self.assertIn(str(plain), entries)
        self.assertFalse([e for e in entries if e and any(f.startswith("libnvrtc")
                                                          for f in (os.listdir(e) if os.path.isdir(e) else []))])


if __name__ == "__main__":
    unittest.main()
