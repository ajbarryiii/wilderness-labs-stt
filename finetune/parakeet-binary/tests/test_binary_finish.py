"""The binary seam end to end on the CPU, without training: train.build_context (replaced by
binary_train's) builds an arm, train.finish (wrapped) exports, rebuilds, checks and scores it,
and summary.json's "binary" block reports the code balance and the latent signs changed since
the arm's start. Arm B5's 1-bit export and control C's 2-bit ternary export (written under the
binary artifacts) are both exercised. The pretrained model is a tiny NeMo model, M1 a
checkpoint of it, the dev evaluator a stub over four noise clips; no CUDA, no loss.

Also: the Hub stream that binary_train opens for the pilot has the ternary pilot's recorded
configuration hash, i.e. arm A sees ternary pilot P2's data in the same order (offline; the
stream's file listings are cached on the data disk and no worker is started).
"""
from __future__ import annotations

import contextlib
import json
import shutil
import unittest
from pathlib import Path
from unittest import mock

import binary_fixtures as fx
import numpy as np
import torch

import binary_paths
import binary_train
import train

TRAIN_ARGS = ["--arm", "P2", "--run-name", "finish-test", "--max-steps", "8", "--stream", "local-librispeech",
              "--batch-seconds", "8", "--eval-every", "4", "--no-powerlog"]


class ArmFixture:
    """An m1 arm built through the seam's build_context and finished with train.finish."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = fx.TEST_ROOT / "finish"
        shutil.rmtree(cls.root, ignore_errors=True)
        cls.fp32_state = {k: v.clone() for k, v in fx.tiny_model(seed=41).state_dict().items()}
        cls.m1_state = fx.perturbed_state(fx.tiny_model(seed=41), seed=42)
        cls.m1_path = cls.root / "m1.pt"
        cls.m1_sha = fx.write_checkpoint(cls.m1_path, cls.m1_state)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root, ignore_errors=True)

    def finish_arm(self, quantizer: str, flips: int, tamper=None) -> tuple[dict, dict, train.Context]:
        """Build the arm (m1 init), flip `flips` latent signs, then train.finish. `tamper(module)`
        returns patches applied while finishing (to break the export)."""
        run = self.root / f"run-{quantizer}"
        shutil.rmtree(run, ignore_errors=True)
        run.mkdir(parents=True)
        with fx.stubbed_build(self.fp32_state, self.m1_path, self.m1_sha), binary_train.seam(quantizer):
            args = binary_train.parse_args(["--init", "m1", "--quantizer", quantizer, "--lr", "2e-4", *TRAIN_ARGS])
            ctx, info = train.build_context(args, run, device="cpu")
            self.assertTrue(info["binary"]["init"]["loaded"])
            self.assertEqual(info["binary"]["init"]["sha256"], self.m1_sha)
            self.assertEqual(info["binary"]["decoder_graphs"],
                             {"teacher": "no_while_loops", "student": "no_while_loops"})
            self.assertTrue((run / binary_train.START_SIGNS).exists())
            self.assertEqual(len(info["quantized_modules"]), fx.TINY_MODULES)
            # Flip the latent sign of `flips` weights of one feed-forward matrix ("training").
            name = next(n for n in info["quantized_modules"] if ".feed_forward" in n)
            with torch.no_grad():
                matrix = ctx.student.get_submodule(name).matrix()
                matrix[0, :flips] = torch.where(matrix[0, :flips] >= 0, -1.0, 1.0) * (matrix[0, :flips].abs() + 0.1)
            ctx.state.step = args.max_steps
            ctx.state.final_subset = {"dev_a": 0.1, "dev_b": 0.2}
            ctx.save()  # the final checkpoint, as train_loop writes it
            ctx.teacher = None
            with contextlib.ExitStack() as stack:
                for cm in (tamper(name) if tamper else ()):
                    stack.enter_context(cm)
                result = train.finish(ctx, "config-sha")
            return result, info, ctx



class FinishTest(ArmFixture, unittest.TestCase):
    def check_common(self, result: dict, info: dict, flips: int) -> None:
        export_dir = Path(result["export_dir"])
        manifest = json.loads((export_dir / "manifest.json").read_text())
        self.assertEqual(manifest["sha256"], fx.sha256_file(export_dir / manifest["file"]))
        self.assertTrue(result["reconstruction"]["codes_exact"])
        self.assertTrue(result["reconstruction"]["scales_exact"])
        self.assertEqual(result["scored_on"], "model rebuilt from export")
        self.assertEqual(set(result["dev_wer"]), {"dev_a", "dev_b"})
        change = result["binary"]["latent_sign_change"]
        weights = sum(int(np.prod(manifest["quantized_layers"][n]["shape"])) for n in info["quantized_modules"])
        self.assertEqual(change["weights"], weights)
        self.assertEqual(change["changed"], flips)
        self.assertAlmostEqual(change["fraction"], flips / weights)
        self.assertGreater(change["per_family"]["feed_forward"], 0.0)
        self.assertEqual(change["per_family"]["self_attn"], 0.0)
        self.assertEqual(change["per_family"]["pointwise_conv"], 0.0)
        balance = result["binary"]["code_balance"]
        self.assertAlmostEqual(balance["minus_one"] + balance["zero"] + balance["plus_one"], 1.0, places=9)
        self.assertEqual(set(balance["per_family"]), {"feed_forward", "self_attn", "pointwise_conv"})
        graphs = result["binary"]["decoder_graphs"]
        self.assertEqual(graphs, {"teacher": "no_while_loops", "student": "no_while_loops",
                                  "rebuilt": "no_while_loops", "dev-subset": None, "full-dev": "no_while_loops"})
        timing = result["binary"]["timing"]
        self.assertEqual((timing["step"], timing["sessions"]), (8, 1))
        self.assertGreater(timing["train_wall_s"], 0.0)

    def test_binary_arm_finishes_with_a_1_bit_export(self) -> None:
        result, info, ctx = self.finish_arm("binary", flips=7)
        self.assertEqual(ctx.ramp, train.ramp_steps(8))
        self.assertTrue(result["export_dir"].startswith(str(binary_paths.ARTIFACTS)))
        manifest = json.loads((ctx.run / "export" / "manifest.json").read_text())
        self.assertEqual(manifest["format"], "parakeet-binary-v1")
        self.check_common(result, info, flips=7)
        self.assertEqual(result["binary"]["code_balance"]["zero"], 0.0)
        self.assertEqual((result["binary"]["init"], result["binary"]["quantizer"], result["binary"]["ramp_kind"]),
                         ("m1", "binary", "ternary"))

    def test_control_c_finishes_with_the_ternary_export(self) -> None:
        result, info, ctx = self.finish_arm("ternary", flips=5)
        self.assertEqual(ctx.ramp, 0)
        self.assertTrue(result["export_dir"].startswith(str(binary_paths.ARTIFACTS)))
        manifest = json.loads((ctx.run / "export" / "manifest.json").read_text())
        self.assertEqual(manifest["format"], "parakeet-ternary-v1")
        self.check_common(result, info, flips=5)
        self.assertGreater(result["binary"]["code_balance"]["zero"], 0.0)
        self.assertEqual((result["binary"]["quantizer"], result["binary"]["ramp_kind"]), ("ternary", "none"))


class ReconstructionFailureTest(ArmFixture, unittest.TestCase):
    """Review round 1, finding 1: an inexact reconstruction stops finishing before any scoring,
    for the binary arms and for control C."""

    def test_inexact_reconstruction_raises_before_scoring(self) -> None:
        import binary_export
        import export as ternary_export
        for quantizer, module in (("binary", binary_export), ("ternary", ternary_export)):
            with self.subTest(quantizer=quantizer):
                load = module.load_export

                def tampered(out_dir, device="cpu", _load=load):
                    """The rebuilt model with one row's signs flipped (a broken export)."""
                    model = _load(out_dir, device)
                    with torch.no_grad():
                        model.encoder.layers[0].feed_forward1.linear1.weight[0].neg_()
                    return model

                def tamper(name, _module=module, _tampered=tampered):
                    return (mock.patch.object(_module, "load_export", _tampered),)

                with self.assertRaisesRegex(binary_train.ReconstructionError, "codes_exact"):
                    self.finish_arm(quantizer, flips=3, tamper=tamper)
                run = self.root / f"run-{quantizer}"
                self.assertTrue((run / "export" / "manifest.json").exists())
                self.assertFalse((run / "eval-dev.json").exists())  # nothing was scored
                self.assertFalse((run / "eval-dev").exists())


class StreamTest(unittest.TestCase):
    def test_pilot_hub_stream_is_the_ternary_pilots(self) -> None:
        import stream
        recorded = json.loads((binary_paths.TERNARY_PILOT_RUN / "summary.json").read_text())["stream"]
        args = binary_train.parse_args(["--init", "fp32", "--quantizer", "binary", "--lr", "5e-4", "--arm", "P2",
                                        "--run-name", "stream-test", "--max-steps", "8", "--stream", "hub"])
        with mock.patch.object(stream, "record_sources"):  # writes the ternary data dir otherwise
            opened = binary_train.open_stream(args)
            self.assertEqual(opened.config_hash, recorded["config_hash"])
            self.assertEqual(opened.state_dict()["partitions"], 6)
            opened.close()


if __name__ == "__main__":
    unittest.main()
