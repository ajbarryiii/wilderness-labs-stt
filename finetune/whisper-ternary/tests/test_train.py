"""Revision 2 training logic: quantization ramp, distillation loss, checkpoint rule, training splits.

CPU only, on a tiny random Whisper config; no checkpoint download and no GPU.
"""
from __future__ import annotations

import contextlib
import copy
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import WhisperConfig, WhisperForConditionalGeneration

import data
import paths
import quant
import train

HAS_RAMP = hasattr(quant, "set_weight_fraction") and hasattr(quant, "weight_fraction")


def tiny_model(seed: int = 0) -> WhisperForConditionalGeneration:
    torch.manual_seed(seed)
    config = WhisperConfig(
        vocab_size=100, d_model=32, encoder_layers=2, decoder_layers=2, encoder_attention_heads=2,
        decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64, num_mel_bins=8,
        max_source_positions=50, max_target_positions=20, pad_token_id=99, bos_token_id=98,
        eos_token_id=99, decoder_start_token_id=97, suppress_tokens=[], begin_suppress_tokens=[])
    return WhisperForConditionalGeneration(config)


def tiny_batch() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    features = torch.randn(3, 8, 100, generator=generator)
    labels = torch.randint(0, 97, (3, 9), generator=generator)
    labels[0, 6:] = -100
    labels[2, 4:] = -100
    return features, labels


def spec_kd(student: torch.Tensor, teacher: torch.Tensor, labels: torch.Tensor, t: float) -> torch.Tensor:
    """The brief's formula verbatim: KL over the full [B, L, V], masked after the vocabulary sum."""
    mask = labels != -100
    s, q = student.float() / t, teacher.float() / t
    kl = F.kl_div(F.log_softmax(s, -1), F.log_softmax(q, -1), log_target=True, reduction="none")
    return kl.sum(-1)[mask].mean() * t * t


class RampTest(unittest.TestCase):
    def test_ramp_steps(self) -> None:
        self.assertEqual(train.ramp_steps(0.0, 4000), 0)
        self.assertEqual(train.ramp_steps(0.25, 12000), 3000)
        self.assertEqual(train.ramp_steps(0.5, 24), 12)
        self.assertEqual(train.ramp_steps(1.0, 24), 24)
        self.assertEqual(train.ramp_steps(1e-6, 100), 1)  # a positive fraction never means "no ramp"
        for bad in (-0.1, 1.5, float("nan")):
            with self.assertRaises(ValueError):
                train.ramp_steps(bad, 100)

    def test_fraction_schedule(self) -> None:
        steps = train.ramp_steps(0.25, 12000)
        f = train.ramp_weight_fraction
        self.assertEqual(f(1, steps), 1 / 3000)
        self.assertEqual(f(steps // 2, steps), 0.5)
        self.assertEqual(f(steps, steps), 1.0)
        self.assertEqual(f(steps + 1, steps), 1.0)
        self.assertEqual(f(12000, steps), 1.0)
        values = [f(k, steps) for k in range(1, 12001)]
        self.assertTrue(all(b > a for a, b in zip(values[:steps], values[1:steps])))
        self.assertTrue(all(v == 1.0 for v in values[steps - 1:]))
        with self.assertRaises(ValueError):
            f(0, steps)  # updates are 1-based

    def test_smoke_schedule(self) -> None:
        steps = train.ramp_steps(0.5, 24)
        values = [train.ramp_weight_fraction(k, steps) for k in range(1, 25)]
        self.assertEqual(values[:12], [k / 12 for k in range(1, 13)])
        self.assertEqual(values[12:], [1.0] * 12)
        self.assertLess(values[5], 1.0)  # a step-6 evaluation would not be selectable

    def test_no_ramp_is_always_one(self) -> None:
        steps = train.ramp_steps(0.0, 4000)
        self.assertEqual({train.ramp_weight_fraction(k, steps) for k in (1, 2, 200, 3999, 4000)}, {1.0})


class KDLossTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(0)
        self.student = torch.randn(2, 5, 11, generator=generator)
        self.teacher = torch.randn(2, 5, 11, generator=generator)
        self.labels = torch.randint(0, 11, (2, 5), generator=generator)
        self.labels[0, 3:] = -100
        self.labels[1, 4] = -100

    def test_zero_for_identical_logits(self) -> None:
        for t in (1.0, 2.0, 0.5):
            kd = train.kd_loss(self.student, self.student.clone(), self.labels, t)
            self.assertLess(abs(kd.item()), 1e-6)

    def test_positive_otherwise(self) -> None:
        self.assertGreater(train.kd_loss(self.student, self.teacher, self.labels, 1.0).item(), 1e-3)

    def test_matches_kl_definition_and_spec_formula(self) -> None:
        mask = self.labels != -100
        p_t = F.softmax(self.teacher, -1)
        kl = (p_t * (F.log_softmax(self.teacher, -1) - F.log_softmax(self.student, -1))).sum(-1)
        kd = train.kd_loss(self.student, self.teacher, self.labels, 1.0)
        torch.testing.assert_close(kd, kl[mask].mean())
        for t in (1.0, 2.0):
            torch.testing.assert_close(train.kd_loss(self.student, self.teacher, self.labels, t),
                                       spec_kd(self.student, self.teacher, self.labels, t))

    def test_masked_positions_excluded(self) -> None:
        kd = train.kd_loss(self.student, self.teacher, self.labels, 1.0)
        student, teacher = self.student.clone(), self.teacher.clone()
        student[0, 4] += 100 * torch.randn(11)
        teacher[1, 4] = -teacher[1, 4]
        self.assertTrue(torch.equal(train.kd_loss(student, teacher, self.labels, 1.0), kd))
        student[0, 0, 3] += 1.0  # a label position does count (one logit: a uniform shift would not)
        self.assertFalse(torch.equal(train.kd_loss(student, teacher, self.labels, 1.0), kd))

    def test_temperature_scaling(self) -> None:
        t = 2.0
        kd = train.kd_loss(self.student, self.teacher, self.labels, t)
        unscaled = train.kd_loss(self.student / t, self.teacher / t, self.labels, 1.0)
        torch.testing.assert_close(kd, unscaled * t * t)
        self.assertFalse(torch.allclose(kd, train.kd_loss(self.student, self.teacher, self.labels, 1.0)))

    def test_fp32_and_gradient_to_student_only(self) -> None:
        student = self.student.to(torch.bfloat16).requires_grad_(True)
        kd = train.kd_loss(student, self.teacher.to(torch.bfloat16), self.labels, 1.0)
        self.assertEqual(kd.dtype, torch.float32)
        kd.backward()
        self.assertTrue(student.grad[self.labels != -100].abs().sum() > 0)
        self.assertTrue((student.grad[self.labels == -100] == 0).all())


class CombineLossTest(unittest.TestCase):
    def test_zero_weight_is_ce_exactly(self) -> None:
        ce, kd = torch.tensor(2.3456789), torch.tensor(0.7)
        self.assertIs(train.combine_loss(ce, kd, 0.0), ce)
        self.assertIs(train.combine_loss(ce, None, 0.0), ce)

    def test_mixture(self) -> None:
        ce, kd = torch.tensor(2.0), torch.tensor(1.0)
        self.assertEqual(train.combine_loss(ce, kd, 0.5).item(), 1.5)
        self.assertEqual(train.combine_loss(ce, kd, 1.0).item(), 1.0)
        self.assertAlmostEqual(train.combine_loss(ce, kd, 0.25).item(), 1.75)


class ComputeLossTest(unittest.TestCase):
    """compute_loss on a tiny model on CPU (BF16 autocast, like the GPU training path)."""

    def setUp(self) -> None:
        self.student = tiny_model().train()
        self.teacher = copy.deepcopy(self.student).eval().requires_grad_(False)
        self.features, self.labels = tiny_batch()

    def test_no_teacher_is_the_v1_loss(self) -> None:
        loss, ce, kd = train.compute_loss(self.student, None, self.features, self.labels, 0.0, 1.0)
        self.assertIsNone(kd)
        self.assertIs(loss, ce)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            v1 = self.student(input_features=self.features, labels=self.labels).loss
        self.assertTrue(torch.equal(loss, v1))

    def test_identical_teacher_gives_zero_kd(self) -> None:
        loss, ce, kd = train.compute_loss(self.student, self.teacher, self.features, self.labels,
                                          0.5, 1.0)
        self.assertLess(abs(kd.item()), 1e-6)
        torch.testing.assert_close(loss, 0.5 * ce + 0.5 * kd)

    def test_quantized_student_gradients_and_frozen_teacher(self) -> None:
        quant.quantize_model(self.student, include_embedding=False)
        loss, ce, kd = train.compute_loss(self.student, self.teacher, self.features, self.labels,
                                          0.5, 2.0)
        self.assertGreater(kd.item(), 1e-4)
        self.assertTrue(torch.equal(loss, 0.5 * ce + 0.5 * kd))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in self.teacher.parameters()))
        self.assertIsNotNone(self.student.model.encoder.layers[0].fc1.weight.grad)
        self.assertFalse(any(isinstance(m, (quant.TernaryLinear, quant.TernaryEmbedding))
                             for m in self.teacher.modules()))

    @unittest.skipUnless(HAS_RAMP, "quant.set_weight_fraction not available yet")
    def test_ramp_endpoints_through_compute_loss(self) -> None:
        quant.quantize_model(self.student, include_embedding=False)
        quant.set_weight_fraction(self.student, 0.0)  # f = 0: the float weights, like the teacher
        _, _, kd = train.compute_loss(self.student, self.teacher, self.features, self.labels, 0.5, 1.0)
        self.assertLess(abs(kd.item()), 1e-6)
        kds = []
        for k in (1, 6, 12):
            quant.set_weight_fraction(self.student, train.ramp_weight_fraction(k, 12))
            kds.append(train.compute_loss(self.student, self.teacher, self.features, self.labels,
                                          0.5, 1.0)[2].item())
        self.assertEqual(quant.weight_fraction(self.student), 1.0)
        self.assertLess(kds[0], kds[2])  # more quantization, further from the FP32 teacher


class BestCheckpointRuleTest(unittest.TestCase):
    @staticmethod
    def select(evaluations: list[tuple[int, float, float | None]]) -> tuple[int | None, float]:
        best_step, best = None, float("inf")
        for step, wer, fraction in evaluations:
            if train.improves(wer, fraction, best):
                best_step, best = step, wer
        return best_step, best

    def test_ignores_partial_quantization(self) -> None:
        evaluations = [(6, 0.05, 0.5), (12, 0.40, 1.0), (18, 0.30, 1.0), (24, 0.35, 1.0)]
        self.assertEqual(self.select(evaluations), (18, 0.30))
        self.assertEqual(self.select([(6, 0.05, 0.5), (12, 0.90, 1.0)]), (12, 0.90))
        self.assertEqual(self.select([(6, 0.05, 0.999999)]), (None, float("inf")))

    def test_fp32_and_no_ramp(self) -> None:
        self.assertEqual(self.select([(500, 0.2, None), (1000, 0.1, None)]), (1000, 0.1))
        self.assertTrue(train.selectable(train.ramp_weight_fraction(1, 0)))
        self.assertFalse(train.selectable(train.ramp_weight_fraction(6, 12)))
        self.assertTrue(train.selectable(train.ramp_weight_fraction(12, 12)))

    def test_strictly_better_only(self) -> None:
        self.assertEqual(self.select([(500, 0.2, 1.0), (1000, 0.2, 1.0)]), (500, 0.2))


class ParseArgsTest(unittest.TestCase):
    BASE = ["--arm", "ternary", "--lr", "3e-4", "--run-name", "x"]

    def test_defaults_are_v1(self) -> None:
        args = train.parse_args(self.BASE)
        self.assertEqual((args.train_splits, args.quant_ramp_fraction, args.distill_weight,
                          args.distill_temperature, args.max_steps, args.warmup, args.eval_every),
                         (["train-clean-100"], 0.0, 0.0, 1.0, 4000, 200, 500))

    def test_values_and_rejections(self) -> None:
        args = train.parse_args(self.BASE + ["--train-splits", "train-clean-100, train-clean-360",
                                             "--quant-ramp-fraction", "0.25",
                                             "--distill-weight", "0.5", "--distill-temperature", "2"])
        self.assertEqual((args.train_splits, args.quant_ramp_fraction, args.distill_weight,
                          args.distill_temperature),
                         (["train-clean-100", "train-clean-360"], 0.25, 0.5, 2.0))
        for bad in (["--train-splits", "train-clean-100,dev-clean"],
                    ["--train-splits", "train-clean-100,train-clean-100"],
                    ["--quant-ramp-fraction", "1.5"], ["--distill-weight", "-0.1"],
                    ["--distill-temperature", "0"]):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                train.parse_args(self.BASE + bad)


class TrainingManifestTest(unittest.TestCase):
    SPLITS = {"train-fake-a": {"19-198-0001": "HELLO", "19-198-0000": "WORLD", "20-7-0000": "ONE"},
              "train-fake-b": {"5-1-0001": "TWO THREE", "5-1-0000": "FOUR"},
              "dev-fake": {"7-7-0000": "HELD OUT"}}

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp(dir=paths.ARTIFACTS / "tmp"))
        cls.roots = {}
        for split, lines in cls.SPLITS.items():
            root = cls.tmp / "LibriSpeech" / split
            chapters: dict[Path, list[str]] = {}
            for uid, text in lines.items():
                speaker, chapter, _ = uid.split("-")
                folder = root / speaker / chapter
                folder.mkdir(parents=True, exist_ok=True)
                t = np.arange(paths.SAMPLE_RATE // 4) / paths.SAMPLE_RATE
                sf.write(folder / f"{uid}.flac", (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32),
                         paths.SAMPLE_RATE, format="FLAC")
                chapters.setdefault(folder, []).append(f"{uid} {text}")
            for folder, rows in chapters.items():
                (folder / f"{folder.parent.name}-{folder.name}.trans.txt").write_text("\n".join(rows) + "\n")
            cls.roots[split] = root
        (cls.tmp / "download.log").write_text("done 2026-09-29T21:03:01Z\n")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp)

    def patched(self, counts: dict[str, int] | None = None):
        stack = contextlib.ExitStack()
        stack.enter_context(patch.dict(paths.SPLITS, self.roots))
        stack.enter_context(patch.dict(paths.EXPECTED_UTTERANCES,
                                       counts or {s: len(v) for s, v in self.SPLITS.items()}))
        manifests = self.tmp / f"manifests-{self.id().rsplit('.', 1)[-1]}"
        stack.enter_context(patch.object(paths, "MANIFESTS", manifests))
        stack.enter_context(patch.object(paths, "DATA", self.tmp))
        return stack

    def test_concatenation_order_and_counts(self) -> None:
        with self.patched():
            manifest = data.build_training_manifest(["train-fake-b", "train-fake-a"])
            self.assertEqual([r["id"] for r in manifest],
                             ["5-1-0000", "5-1-0001", "19-198-0000", "19-198-0001", "20-7-0000"])
            self.assertEqual([r["split"] for r in manifest], ["train-fake-b"] * 2 + ["train-fake-a"] * 3)
            self.assertEqual({s: sum(r["split"] == s for r in manifest) for s in ("train-fake-a", "train-fake-b")},
                             {"train-fake-a": 3, "train-fake-b": 2})
            for record in manifest:
                self.assertEqual(record["text"], self.SPLITS[record["split"]][record["id"]])
                self.assertTrue(Path(record["path"]).is_file())
            reordered = data.build_training_manifest(["train-fake-a", "train-fake-b"])
            self.assertEqual([r["id"] for r in reordered][:3], ["19-198-0000", "19-198-0001", "20-7-0000"])

    def test_single_split_matches_build_manifest(self) -> None:
        with self.patched():
            single = data.build_training_manifest(["train-fake-a"])
            plain = data.build_manifest("train-fake-a")
            self.assertEqual([{k: v for k, v in r.items() if k != "split"} for r in single], plain)

    def test_each_split_count_verified(self) -> None:
        with self.patched({"train-fake-a": 3, "train-fake-b": 3}):
            with self.assertRaisesRegex(ValueError, "train-fake-b: found 2 utterances.*expected 3"):
                data.build_training_manifest(["train-fake-a", "train-fake-b"])

    def test_rejections(self) -> None:
        with self.patched():
            with self.assertRaisesRegex(ValueError, "duplicate"):
                data.build_training_manifest(["train-fake-a", "train-fake-b", "train-fake-a"])
            for held_out in ("dev-fake", "dev-clean", "test-clean", "test-other"):
                with self.assertRaisesRegex(ValueError, "not training splits"):
                    data.build_training_manifest(["train-fake-a", held_out])
            with self.assertRaisesRegex(ValueError, "unknown training splits"):
                data.build_training_manifest(["train-other-500"])
            with self.assertRaises(ValueError):
                data.build_training_manifest([])
            with self.assertRaises(TypeError):
                data.build_training_manifest("train-fake-a")
            self.assertFalse((self.tmp / "manifests-test_rejections").exists())  # nothing scanned

    def test_sampler_covers_concatenation(self) -> None:
        with self.patched():
            manifest = data.build_training_manifest(["train-fake-a", "train-fake-b"])
        sampler = data.EpochSampler(len(manifest), seed=paths.SEED)
        order = list(sampler)
        self.assertEqual(sorted(order), list(range(5)))
        self.assertEqual(order, torch.randperm(5, generator=torch.Generator().manual_seed(paths.SEED)).tolist())


if __name__ == "__main__":
    unittest.main()
