"""Contract tests 6-7 (plans/pilot.md "Contract tests"): 1-bit packing and the binary export.

Packing is checked against numpy.packbits(bitorder="little"), an independent implementation of
the same layout (eight codes per byte, least significant bit first, bit 1 = +1). The export is
a full export -> load_export round trip of a tiny NeMo model (the pinned config and tokenizer)
on the CPU.
"""
from __future__ import annotations

import math
import shutil
import unittest
from pathlib import Path
from unittest import mock

import binary_fixtures as fx
import numpy as np
import torch

import binary_export
import binary_quant
import paths


class PackingTest(unittest.TestCase):
    def test_round_trip_and_numpy_parity(self) -> None:
        generator = torch.Generator().manual_seed(0)
        for cols in (1, 7, 8, 9, 15, 1023, 1024, 4097):
            with self.subTest(cols=cols):
                codes = (2 * torch.randint(0, 2, (5, cols), generator=generator) - 1).to(torch.int8)
                packed = binary_export.pack_codes(codes)
                self.assertEqual((packed.dtype, tuple(packed.shape)), (torch.uint8, (5, math.ceil(cols / 8))))
                reference = np.packbits((codes.numpy() == 1).astype(np.uint8), axis=1, bitorder="little")
                self.assertTrue(np.array_equal(packed.numpy(), reference))
                self.assertTrue(torch.equal(binary_export.unpack_codes(packed, cols), codes))

    def test_bit_convention(self) -> None:
        def pack(row: list[int]) -> list[int]:
            return binary_export.pack_codes(torch.tensor([row], dtype=torch.int8))[0].tolist()

        self.assertEqual(pack([1] + [-1] * 7), [0b00000001])  # code 0 -> least significant bit
        self.assertEqual(pack([-1, 1] + [-1] * 6), [0b00000010])
        self.assertEqual(pack([-1] * 7 + [1]), [0b10000000])
        self.assertEqual(pack([1] * 8), [0xFF])
        self.assertEqual(pack([-1] * 8), [0x00])
        self.assertEqual(pack([1] * 9), [0xFF, 0b00000001])  # tail bits are 0
        self.assertEqual(pack([-1] * 8 + [1, -1, 1]), [0x00, 0b00000101])
        unpacked = binary_export.unpack_codes(torch.tensor([[0b00000101]], dtype=torch.uint8), 3)
        self.assertEqual(unpacked.tolist(), [[1, -1, 1]])

    def test_rejects_invalid(self) -> None:
        with self.assertRaises(ValueError):  # a ternary zero has no binary code
            binary_export.pack_codes(torch.tensor([[1, 0, -1]], dtype=torch.int8))
        with self.assertRaises(ValueError):
            binary_export.pack_codes(torch.tensor([1, -1], dtype=torch.int8))
        with self.assertRaises(ValueError):
            binary_export.pack_codes(torch.tensor([[1, -1]], dtype=torch.int32))
        with self.assertRaises(ValueError):  # nonzero tail padding
            binary_export.unpack_codes(torch.tensor([[0b00001000]], dtype=torch.uint8), 3)
        with self.assertRaises(ValueError):  # wrong width
            binary_export.unpack_codes(torch.zeros(2, 2, dtype=torch.uint8), 20)
        with self.assertRaises(ValueError):
            binary_export.unpack_codes(torch.zeros(2, 1, dtype=torch.int8), 8)


class ExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        paths.require_mount()
        cls.model = fx.tiny_model(seed=11)
        with fx.tiny_module_count():
            cls.names = binary_quant.quantize_parakeet(cls.model, "ternary")
        cls.out_dir = fx.TEST_ROOT / "export"
        cls.refused = fx.TEST_ROOT / "export-refused"
        for d in (cls.out_dir, cls.refused):
            shutil.rmtree(d, ignore_errors=True)
        generator = torch.Generator().manual_seed(12)
        cls.batch = (0.1 * torch.randn(2, 16000 * 2, generator=generator), torch.tensor([32000, 20000]))

    @classmethod
    def tearDownClass(cls) -> None:
        for d in (cls.out_dir, cls.refused):
            shutil.rmtree(d, ignore_errors=True)

    def test_1_refuses_fraction_below_1_and_writes_nothing(self) -> None:
        try:
            binary_quant.set_weight_fraction(self.model, 0.999)
            with self.assertRaisesRegex(ValueError, "fraction"):
                binary_export.export_model(self.model, self.refused, extra={})
            self.assertFalse(self.refused.exists())
            binary_quant.set_weight_fraction(self.model, 1.0)
            self.model.get_submodule(self.names[3]).fraction = 0.5  # one module behind
            with self.assertRaisesRegex(ValueError, "fraction"):
                binary_export.export_model(self.model, self.refused, extra={})
            self.assertFalse(self.refused.exists())
        finally:
            binary_quant.set_weight_fraction(self.model, 1.0)

    def test_2_refuses_outside_the_binary_artifacts(self) -> None:
        for bad in (Path("/tmp/parakeet-binary-export-test"), paths.ARTIFACTS / "tmp" / "binary-export-test"):
            with self.subTest(out_dir=bad), self.assertRaises(ValueError):
                binary_export.export_model(self.model, bad, extra={})
            self.assertFalse(bad.exists())

    def test_3_round_trip_exact_codes_and_scales(self) -> None:
        manifest = binary_export.export_model(self.model, self.out_dir, extra={"test": True})
        self.assertEqual(manifest["format"], "parakeet-binary-v1")
        self.assertEqual(sorted(manifest["quantized_layers"]), self.names)
        self.assertEqual(manifest["extra"], {"test": True})
        # The manifest's SHA-256 is the file's, for every file.
        self.assertEqual(manifest["sha256"], fx.sha256_file(self.out_dir / manifest["file"]))
        self.assertEqual(sorted(manifest["files"]), sorted(["export.safetensors", "tokenizer/tokenizer.model",
                                                            "tokenizer/vocab.txt", "tokenizer/tokenizer.vocab"]))
        for rel, digest in manifest["files"].items():
            self.assertEqual(digest, fx.sha256_file(self.out_dir / rel), rel)
        expected_bytes = sum(layer["shape"][0] * math.ceil(layer["shape"][1] / 8)
                             for layer in manifest["quantized_layers"].values())
        self.assertEqual(manifest["bytes"]["packed_code_bytes"], expected_bytes)
        self.assertEqual(manifest["parameter_accounting"]["binary_parameters"],
                         sum(math.prod(l["shape"]) for l in manifest["quantized_layers"].values()))

        with mock.patch.object(paths, "MODEL_FILE", Path("/nonexistent/model.nemo")):  # not needed to load
            rebuilt = binary_export.load_export(self.out_dir, "cpu")
        self.assertFalse(binary_quant.quantized_module_names(rebuilt))
        self.assertFalse(rebuilt.training)
        self.assertTrue(all(p.dtype == torch.float32 for p in rebuilt.parameters()))
        for name in self.names:
            codes, scale = self.model.get_submodule(name).quantized_weight()
            weight = rebuilt.get_submodule(name).weight.detach()
            weight = weight.squeeze(2) if weight.dim() == 3 else weight
            self.assertTrue(torch.equal(weight, codes.float() * scale[:, None]), name)
        self.model.eval()
        check = binary_export.reconstruction_check(self.model, rebuilt, self.batch)
        self.assertTrue(check["codes_exact"])
        self.assertTrue(check["scales_exact"])
        self.assertEqual(check["checked_layers"], len(self.names))
        self.assertIn("joint_output_max_abs_diff", check)  # the logit difference is recorded
        self.assertLess(check["encoder_max_abs_diff"], 0.05 * check["encoder_max_abs"] + 1e-6)

        # A tampered file is rejected.
        vocab = self.out_dir / "tokenizer" / "vocab.txt"
        original = vocab.read_bytes()
        try:
            vocab.write_bytes(original + b"x")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                binary_export.load_export(self.out_dir, "cpu")
        finally:
            vocab.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
