"""The zero-step points and the G-B gate (plans/pilot.md "Zero-step points") on the CPU: a tiny
NeMo model as the pretrained model, a checkpoint of it as M1, a stub dev evaluator. G-B passes
only if its full-dev mean is within 0.02 points of M1's recorded mean and its encoder equals the
ternary M1 model's; Z-B is scored on the model rebuilt from its 1-bit export."""
from __future__ import annotations

import contextlib
import json
import shutil
import unittest
from unittest import mock

import binary_fixtures as fx
import torch

import binary_eval
import binary_paths
import binary_train
import paths
import train


class PointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = fx.TEST_ROOT / "points"
        shutil.rmtree(cls.root, ignore_errors=True)
        cls.fp32_state = {k: v.clone() for k, v in fx.tiny_model(seed=51).state_dict().items()}
        cls.m1_state = fx.perturbed_state(fx.tiny_model(seed=51), seed=52)
        cls.m1_path = cls.root / "m1.pt"
        cls.m1_sha = fx.write_checkpoint(cls.m1_path, cls.m1_state)
        cls.m1_wer = json.loads((binary_paths.M1_RUN / "summary.json").read_text())["dev_wer"]

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root, ignore_errors=True)

    def pretrained(self, device: str = "cpu"):
        model = fx.tiny_model(seed=53)
        model.load_state_dict(self.fp32_state)
        return model

    def score(self, point: str, values: dict | None, limit: int | None = None, extra=()) -> dict:
        with contextlib.ExitStack() as stack:
            for cm in (fx.pinned_init(self.m1_path, self.m1_sha), fx.tiny_module_count(),
                       mock.patch.object(binary_paths, "RUNS", self.root / "runs"),
                       mock.patch.object(binary_paths, "EVAL", self.root / "eval"),
                       mock.patch.object(paths, "gpu_lock", lambda label: contextlib.nullcontext()),
                       mock.patch.object(train, "verify_base", return_value={"base_sha256": "stub"}),
                       mock.patch.object(train, "load_pretrained", side_effect=self.pretrained),
                       mock.patch.object(train, "DevEvaluator",
                                         side_effect=lambda *a: fx.StubEvaluator(tuple(paths.DEV_SETS), values))):
                stack.enter_context(cm)
            for cm in extra:
                stack.enter_context(cm)
            binary_eval.main(["--point", point, "--device", "cpu", "--overwrite",
                              *(["--limit", str(limit)] if limit else [])])
            name = point if limit is None else f"smoke-{point}-limit{limit}"
            return json.loads((self.root / "eval" / name / "summary.json").read_text())

    def test_gate_passes_only_within_tolerance_with_an_identical_ternary_twin(self) -> None:
        summary = self.score("G-B", self.m1_wer)
        self.assertEqual(summary["weight_fraction"], 0.0)
        self.assertEqual((summary["init"]["sha256"], summary["init"]["tensors"]), (self.m1_sha, len(self.m1_state)))
        self.assertTrue(summary["ternary_twin"]["encoder_equal"])
        self.assertEqual(summary["decoder_graphs"], {"student": "no_while_loops", "full-dev": "no_while_loops"})
        self.assertTrue(summary["gate"]["pass"])
        self.assertAlmostEqual(summary["gate"]["difference_points"], 0.0, places=9)
        off = {k: v + 0.0003 for k, v in self.m1_wer.items()}  # 0.03 points worse on every set
        summary = self.score("G-B", off)
        self.assertAlmostEqual(summary["gate"]["difference_points"], 0.03, places=6)
        self.assertFalse(summary["gate"]["pass"])
        near = {k: v + 0.0001 for k, v in self.m1_wer.items()}  # 0.01 points
        self.assertTrue(self.score("G-B", near)["gate"]["pass"])

    def test_zero_step_points_are_scored_on_the_rebuilt_export(self) -> None:
        for point, init in (("Z-A", "fp32"), ("Z-B", "m1")):
            with self.subTest(point):
                summary = self.score(point, None)
                self.assertEqual(summary["init"]["init"], init)
                self.assertEqual(summary["weight_fraction"], 1.0)
                self.assertEqual(summary["scored_on"], "model rebuilt from export")
                self.assertTrue(summary["reconstruction"]["codes_exact"])
                self.assertTrue(summary["reconstruction"]["scales_exact"])
                self.assertEqual(summary["decoder_graphs"], {"qat": "no_while_loops", "rebuilt": "no_while_loops",
                                                             "full-dev": "no_while_loops"})
                manifest = json.loads((self.root / "runs" / point / "export" / "manifest.json").read_text())
                self.assertEqual(manifest["format"], "parakeet-binary-v1")
                self.assertEqual(set(summary["dev_wer"]), set(paths.DEV_SETS))
                self.assertNotIn("gate", summary)
                for name in paths.DEV_SETS:
                    self.assertTrue((self.root / "eval" / point / f"{name}.json").exists())


    def test_limited_scoring_never_occupies_the_production_path(self) -> None:
        """Review round 1, finding 4: --limit writes eval/smoke-<point>-limit<N> (and its export
        under runs/smoke-<point>-limit<N>), carries no gate verdict, and leaves eval/<point> alone."""
        shutil.rmtree(self.root / "eval", ignore_errors=True)
        shutil.rmtree(self.root / "runs", ignore_errors=True)
        summary = self.score("G-B", self.m1_wer, limit=16)
        self.assertEqual(summary["limit"], 16)
        self.assertIsNone(summary["gate"]["pass"])
        self.assertFalse((self.root / "eval" / "G-B").exists())
        summary = self.score("Z-B", None, limit=16)
        self.assertTrue((self.root / "runs" / "smoke-Z-B-limit16" / "export" / "manifest.json").exists())
        self.assertFalse((self.root / "eval" / "Z-B").exists() or (self.root / "runs" / "Z-B").exists())
        self.assertEqual(binary_eval.output_name("G-B", None), "G-B")
        self.assertEqual(binary_eval.output_name("Z-A", 16), "smoke-Z-A-limit16")

    def test_inexact_reconstruction_stops_a_zero_step_point_before_scoring(self) -> None:
        """Review round 1, finding 1, for the zero-step points."""
        import binary_export
        load = binary_export.load_export

        def tampered(out_dir, device="cpu"):
            model = load(out_dir, device)
            with torch.no_grad():
                model.encoder.layers[0].feed_forward1.linear1.weight[0].neg_()
            return model

        shutil.rmtree(self.root / "eval" / "Z-A", ignore_errors=True)
        with self.assertRaises(binary_train.ReconstructionError):
            self.score("Z-A", None, extra=(mock.patch.object(binary_export, "load_export", tampered),))
        self.assertFalse((self.root / "eval" / "Z-A" / "summary.json").exists())
        self.assertFalse(list((self.root / "eval" / "Z-A").glob("*.json")))


if __name__ == "__main__":
    unittest.main()
