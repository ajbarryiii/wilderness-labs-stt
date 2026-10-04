"""Export format tests: packing parity with the Whisper format, refusal below fraction 1, and a full
export -> load_export round trip of the real (PTQ) model on CPU with two real test clips."""
from __future__ import annotations

import ast
import math
import shutil
import unittest
from pathlib import Path
from unittest import mock

import torch

import evaluate
import export
import paths
import quant


def whisper_pack_functions() -> dict:
    """pack_codes / unpack_codes executed from finetune/whisper-ternary/export.py's own source."""
    source = (quant.WHISPER_DIR / "export.py").read_text()
    tree = ast.parse(source)
    keep = [node for node in tree.body
            if (isinstance(node, ast.FunctionDef) and node.name in ("pack_codes", "unpack_codes"))
            or (isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "_SHIFTS" for t in node.targets))]
    namespace = {"torch": torch, "math": math, "Tensor": torch.Tensor}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "whisper_export", "exec"), namespace)
    return namespace


class PackingTest(unittest.TestCase):
    def test_round_trip_and_parity_with_whisper_format(self) -> None:
        whisper = whisper_pack_functions()
        generator = torch.Generator().manual_seed(0)
        for cols in (1, 4, 7, 1024, 4097):
            codes = (torch.randint(0, 3, (5, cols), generator=generator) - 1).to(torch.int8)
            packed = export.pack_codes(codes)
            self.assertEqual(tuple(packed.shape), (5, math.ceil(cols / 4)))
            self.assertTrue(torch.equal(packed, whisper["pack_codes"](codes)))
            self.assertTrue(torch.equal(export.unpack_codes(packed, cols), codes))

    def test_unpack_rejects_invalid(self) -> None:
        with self.assertRaises(ValueError):
            export.unpack_codes(torch.full((1, 1), 0b11, dtype=torch.uint8), 4)
        with self.assertRaises(ValueError):
            export.unpack_codes(torch.full((1, 1), 0b01000000, dtype=torch.uint8), 3)  # nonzero tail


class RealExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        manifest = paths.MANIFESTS / "test_librispeech_clean.jsonl"
        if not manifest.exists():
            raise unittest.SkipTest("run testsets.py first (needs test_librispeech_clean.jsonl)")
        records = sorted(evaluate.read_manifest(manifest), key=lambda r: r["duration"])
        clips = [records[100], records[2000]]  # two real clips, about 2 s and 5 s
        cls.batch = (*evaluate.audio_batch(clips), [r["text"] for r in clips])
        cls.model = evaluate.load_pretrained("cpu")
        quant.quantize_parakeet(cls.model)
        cls.out_dir = paths.ARTIFACTS / "tmp" / "test-export"
        shutil.rmtree(cls.out_dir, ignore_errors=True)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.out_dir, ignore_errors=True)

    def test_1_refuses_fraction_below_1_and_writes_nothing(self) -> None:
        refused = paths.ARTIFACTS / "tmp" / "test-export-refused"
        shutil.rmtree(refused, ignore_errors=True)
        try:
            quant.set_weight_fraction(self.model, 0.999)
            with self.assertRaisesRegex(ValueError, "fraction < 1"):
                export.export_model(self.model, refused, extra={})
            self.assertFalse(refused.exists())
        finally:
            quant.set_weight_fraction(self.model, 1.0)

    def test_2_refuses_outside_artifacts(self) -> None:
        with self.assertRaises(ValueError):
            export.export_model(self.model, Path("/tmp/parakeet-export-test"), extra={})

    def test_3_round_trip(self) -> None:
        manifest = export.export_model(self.model, self.out_dir, extra={"test": True})
        self.assertEqual(manifest["parameter_accounting"]["total_parameters"], quant.TOTAL_PARAMETERS)
        self.assertEqual(len(manifest["quantized_layers"]), 264)
        self.assertEqual(sorted(manifest["files"]), sorted(["export.safetensors", "tokenizer/tokenizer.model",
                                                            "tokenizer/vocab.txt", "tokenizer/tokenizer.vocab"]))
        sizes = manifest["bytes"]
        self.assertEqual(sizes["packed_code_bytes"], manifest["parameter_accounting"]["ternary_parameters"] // 4)
        # The loader must not need the original .nemo.
        with mock.patch.object(paths, "MODEL_FILE", Path("/nonexistent/model.nemo")):
            rebuilt = export.load_export(self.out_dir, "cpu")
        self.assertFalse(quant.quantized_module_names(rebuilt))
        self.assertFalse(rebuilt.training)
        self.assertTrue(all(p.dtype == torch.float32 for p in rebuilt.parameters()))
        self.model.eval()
        check = export.reconstruction_check(self.model, rebuilt, self.batch)
        self.assertTrue(check["codes_exact"])
        self.assertTrue(check["scales_exact"])
        # Untrained PTQ decodes these clips to empty strings, so hyp equality alone is weak here;
        # the teacher-forced joint comparison over the reference tokens covers the decoder path.
        self.assertTrue(check["greedy_hyps_equal"], check)
        self.assertEqual(check["joint_targets"], "reference texts")
        self.assertGreater(check["joint_target_tokens"], 10)
        self.assertLess(check["joint_output_max_abs_diff"], 0.1)
        self.assertLess(check["encoder_max_abs_diff"], 0.05 * check["encoder_max_abs"])
        print(f"\nreconstruction (CPU): {({k: v for k, v in check.items() if 'hyps' not in k})}")

        # A tampered file is rejected.
        vocab = self.out_dir / "tokenizer" / "vocab.txt"
        original = vocab.read_bytes()
        try:
            vocab.write_bytes(original + b"x")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                export.load_export(self.out_dir, "cpu")
        finally:
            vocab.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
