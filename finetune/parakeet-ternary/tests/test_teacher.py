"""In-loop teacher labeling: filters, agreement with the evaluation decode path, output shapes (real model, CPU)."""
from __future__ import annotations

import unittest

import torch

import evaluate
import paths
import teacher


class DropReasonTest(unittest.TestCase):
    def test_filters(self) -> None:
        dr = teacher.drop_reason
        self.assertIsNone(dr(5.0, "Hello world.", "hello world"))
        self.assertEqual(dr(5.0, "  ", "hello world"), "empty_teacher_text")
        self.assertEqual(dr(5.0, "", None), "empty_teacher_text")
        self.assertEqual(dr(5.0, "x y z d", "a b c d"), "teacher_human_wer_gt_0.5")
        self.assertIsNone(dr(5.0, "a b x y", "a b c d"))  # exactly 0.5 is kept
        self.assertIsNone(dr(5.0, "no human text", None))
        self.assertIsNone(dr(5.0, "blank human text", "   "))
        self.assertEqual(dr(0.5, "a", "a"), "duration_out_of_range")
        self.assertEqual(dr(31.0, "a", "a"), "duration_out_of_range")
        self.assertIsNone(dr(31.0, "a", "a", min_s=None, max_s=None))


class TeacherBatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        manifest = paths.MANIFESTS / "test_librispeech_clean.jsonl"
        if not manifest.exists():
            raise unittest.SkipTest("run testsets.py first")
        records = sorted(evaluate.read_manifest(manifest), key=lambda r: r["duration"])
        cls.records = [records[100], records[2000]]
        cls.audio, cls.lengths = evaluate.audio_batch(cls.records)
        cls.model = evaluate.load_pretrained("cpu").requires_grad_(False)

    def test_fp32_matches_evaluation_path_and_filters(self) -> None:
        expected = evaluate.transcribe_records(self.model, self.records, batch_size=2, device="cpu", workers=0)
        counts: dict = {}
        human = [self.records[0]["text"], "completely unrelated words here for this clip and more"]
        texts, keep, enc, enc_len = teacher.teacher_label_batch(self.model, self.audio, self.lengths, human,
                                                                precision="fp32", counts=counts)
        self.assertEqual(texts, expected)
        self.assertEqual(keep.tolist(), [True, False])
        self.assertEqual(counts, {"kept": 1, "teacher_human_wer_gt_0.5": 1})
        self.assertEqual((enc.dtype, enc.shape[0], enc.shape[1]), (torch.float32, 2, 1024))
        self.assertEqual(int(enc_len.max()), enc.shape[2])
        self.assertFalse(self.model.training)

    def test_bf16_runs_and_returns_fp32_encoder_output(self) -> None:
        texts, keep, enc, _ = teacher.teacher_label_batch(self.model, self.audio, self.lengths, [None, None])
        self.assertEqual(enc.dtype, torch.float32)
        self.assertTrue(all(texts))
        self.assertEqual(keep.tolist(), [True, True])


if __name__ == "__main__":
    unittest.main()
