"""Exactness contract of the derived 2-bit export (plans/benchmark.md stage 1, step 1). CPU only."""
from __future__ import annotations

import json
import unittest

import torch

from gpu1bit.shared import env


class DerivedExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from safetensors.torch import load_file
        cls.src_manifest = json.loads((env.MB1_EXPORT / "manifest.json").read_text())
        cls.dst_manifest = json.loads((env.DERIVED_EXPORT / "manifest.json").read_text())
        cls.src = load_file(env.MB1_EXPORT / cls.src_manifest["file"])
        cls.dst = load_file(env.DERIVED_EXPORT / cls.dst_manifest["file"])

    def test_file_hashes_and_provenance(self):
        import binary_export
        self.assertEqual(self.dst_manifest["format"], "parakeet-ternary-v1")
        for rel, digest in self.dst_manifest["files"].items():
            self.assertEqual(env.sha256_file(env.DERIVED_EXPORT / rel), digest, rel)
        self.assertEqual(self.dst_manifest["sha256"], self.dst_manifest["files"]["export.safetensors"])
        d = self.dst_manifest["derived_from"]
        self.assertEqual(d["files"], self.src_manifest["files"])
        self.assertEqual(d["files"]["export.safetensors"], env.MB1_SHA256)
        self.assertEqual(d["manifest_sha256"], env.sha256_file(env.MB1_EXPORT / binary_export.MANIFEST))
        self.assertEqual(d["script_sha256"], env.sha256_file(env.SHARED / "derive.py"),
                         "derive.py changed since the export was written: rerun it")

    def test_codes_are_the_same_signs(self):
        import binary_export
        import export as ternary_export
        layers = self.src_manifest["quantized_layers"]
        self.assertEqual(layers, self.dst_manifest["quantized_layers"])
        total = torch.zeros(3, dtype=torch.long)
        for name, info in layers.items():
            n, k = info["shape"]
            src = binary_export.unpack_codes(self.src[f"{name}.codes"], k)
            dst_packed = self.dst[f"{name}.codes"]
            self.assertEqual(tuple(dst_packed.shape), (n, (k + 3) // 4), name)
            dst = ternary_export.unpack_codes(dst_packed, k)  # also rejects 11 fields and tail bits
            self.assertTrue(torch.equal(src, dst), name)
            total += torch.bincount(dst.flatten().long() + 1, minlength=3)
        self.assertEqual(total[1].item(), 0)
        self.assertEqual(total.tolist(), self.dst_manifest["code_histogram"]["counts"])
        self.assertEqual(self.dst_manifest["code_histogram"]["zero"], 0.0)

    def test_every_other_tensor_byte_identical(self):
        layers = self.src_manifest["quantized_layers"]
        code_keys = {f"{n}.codes" for n in layers}
        self.assertEqual(set(self.src), set(self.dst))
        for key, value in self.src.items():
            if key in code_keys:
                continue
            other = self.dst[key]
            self.assertEqual((value.dtype, value.shape), (other.dtype, other.shape), key)
            self.assertTrue(torch.equal(value.reshape(-1).contiguous().view(torch.uint8), other.reshape(-1).contiguous().view(torch.uint8)), key)
        for name in layers:
            self.assertTrue(torch.equal(self.src[f"{name}.scale"].view(torch.int32), self.dst[f"{name}.scale"].view(torch.int32)))

    def test_tokenizer_identical(self):
        for rel, digest in self.src_manifest["files"].items():
            if rel.startswith("tokenizer/"):
                self.assertEqual(self.dst_manifest["files"][rel], digest)


if __name__ == "__main__":
    unittest.main()
