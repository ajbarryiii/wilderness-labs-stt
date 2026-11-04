"""Contract of the native parakeet-binary-v1 reader (mil/binary_export.py) and its weights.Source binding.

Invented tiny exports only (tests/common.py). Contracts:
- codes unpack LSB first, bit 1 = +1, bit 0 = -1, as int8 [out, in]; nonzero tail bits, a wrong packed
  width or dtype are refused;
- the export file must match the manifest SHA-256 (and its "files" entry); the format must be
  parakeet-binary-v1;
- scales stay FP32, FP16 floats are widened to FP32 exactly, raw() keeps the stored dtype;
- tokenizer files are checked against the manifest's "files" hashes;
- weights.Source("mb1", path) yields the same (codes, scale) interface the encoder build uses, and the
  codes equal what the 2-bit parakeet-ternary-v1 reader gives for the same model.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import common  # noqa: F401  (paths)
from common import MODULES, write_binary_export, write_ternary_export


class UnpackTests(unittest.TestCase):
    def test_lsb_first_and_sign(self):
        from mil.binary_export import unpack_binary_codes

        # row 0: bits 0 and 2 of byte 0 set, bit 0 of byte 1 set -> +1 -1 +1 -1 -1 -1 -1 -1 +1
        packed = np.array([[0b00000101, 0b00000001], [0xFF, 0b00000000]], dtype=np.uint8)
        codes = unpack_binary_codes(packed, 9)
        self.assertEqual(codes.dtype, np.int8)
        np.testing.assert_array_equal(codes[0], [1, -1, 1, -1, -1, -1, -1, -1, 1])
        np.testing.assert_array_equal(codes[1], [1, 1, 1, 1, 1, 1, 1, 1, -1])

    def test_refuses_tail_bits_width_and_dtype(self):
        from mil.binary_export import unpack_binary_codes

        with self.assertRaises(ValueError):  # bit 1 of byte 1 is beyond in = 9
            unpack_binary_codes(np.array([[0, 0b10]], dtype=np.uint8), 9)
        with self.assertRaises(ValueError):  # ceil(9 / 8) = 2 bytes, not 3
            unpack_binary_codes(np.zeros((1, 3), dtype=np.uint8), 9)
        with self.assertRaises(ValueError):
            unpack_binary_codes(np.zeros((1, 2), dtype=np.int8), 9)


class ExportSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "export"

    def tearDown(self):
        self.tmp.cleanup()

    def test_codes_scales_and_floats(self):
        from mil.binary_export import BinaryExportSource

        fx = write_binary_export(self.dir)
        src = BinaryExportSource(self.dir)
        self.assertEqual(src.manifest["sha256"], fx["sha256"])
        for name in MODULES:
            codes, scale = src.binary(name)
            self.assertEqual(codes.dtype, np.int8)
            np.testing.assert_array_equal(codes, fx["codes"][name])
            self.assertEqual(scale.dtype, np.float32)
            np.testing.assert_array_equal(scale, fx["scales"][name])
            np.testing.assert_array_equal(src[f"{name}.codes"], fx["codes"][name])
        for name, value in fx["floats"].items():
            self.assertEqual(src[name].dtype, np.float32)
            np.testing.assert_array_equal(src[name], value.astype(np.float32))
            self.assertEqual(src.raw(name).dtype, value.dtype)
            np.testing.assert_array_equal(src.raw(name), value)
        self.assertIn("encoder.layers.0.norm_out.weight", set(src))

    def test_refuses_hash_mismatch_and_format(self):
        from mil.binary_export import BinaryExportSource

        write_binary_export(self.dir)
        data = bytearray((self.dir / "export.safetensors").read_bytes())
        data[-1] ^= 1
        (self.dir / "export.safetensors").write_bytes(bytes(data))
        with self.assertRaises(ValueError):
            BinaryExportSource(self.dir)
        write_binary_export(self.dir)
        m = json.loads((self.dir / "manifest.json").read_text())
        m["files"]["export.safetensors"] = "0" * 64
        (self.dir / "manifest.json").write_text(json.dumps(m))
        with self.assertRaises(ValueError):
            BinaryExportSource(self.dir)
        write_binary_export(self.dir)
        m = json.loads((self.dir / "manifest.json").read_text())
        m["format"] = "parakeet-ternary-v1"
        (self.dir / "manifest.json").write_text(json.dumps(m))
        with self.assertRaises(ValueError):
            BinaryExportSource(self.dir)

    def test_refuses_nonzero_tail_bits_in_file(self):
        from mil.binary_export import BinaryExportSource

        def corrupt(t):  # pointwise_conv1 has in = 9: bit 7 of the last byte is padding
            t["encoder.layers.0.conv.pointwise_conv1.codes"][0, -1] |= 0x80

        write_binary_export(self.dir, corrupt=corrupt)
        src = BinaryExportSource(self.dir)
        with self.assertRaises(ValueError):
            src.binary("encoder.layers.0.conv.pointwise_conv1")

    def test_tokenizer_verification(self):
        from mil.binary_export import BinaryExportSource

        tok = {"tokenizer/tokenizer.vocab": b"<unk>\t0\n", "tokenizer/tokenizer.model": b"invented"}
        write_binary_export(self.dir, tokenizer=tok)
        tdir = Path(self.tmp.name) / "tok"
        tdir.mkdir()
        for rel, data in tok.items():
            (tdir / Path(rel).name).write_bytes(data)
        src = BinaryExportSource(self.dir)
        checked = src.verify_tokenizer(tdir)
        self.assertEqual(set(checked), set(tok))
        (tdir / "tokenizer.model").write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            src.verify_tokenizer(tdir)

    def test_weights_source_binding_and_ternary_equivalence(self):
        import reference
        from mil.weights import Source

        fx = write_binary_export(self.dir)
        src = Source("mb1", self.dir)
        self.assertEqual(src.provenance["export_sha256"], fx["sha256"])
        self.assertEqual(src.provenance["format"], "parakeet-binary-v1")
        tdir = Path(self.tmp.name) / "ternary"
        write_ternary_export(tdir, fx["codes"], fx["scales"], fx["floats"])
        tern = reference.ExportSource(tdir)
        for name in MODULES:
            codes, scale = src.ternary(name)
            self.assertEqual(codes.dtype, np.int8)
            self.assertEqual(set(np.unique(codes)), {-1, 1})
            np.testing.assert_array_equal(codes, tern[f"{name}.codes"].numpy())
            np.testing.assert_array_equal(scale, tern[f"{name}.scale"].numpy())
        key = "encoder.layers.0.norm_out.weight"
        self.assertTrue(src.has(key))
        np.testing.assert_array_equal(src.floating(key), tern[key].numpy())


if __name__ == "__main__":
    unittest.main()
