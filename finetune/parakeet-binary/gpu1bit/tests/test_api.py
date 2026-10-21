"""CPU tests of H's shared definitions: bit layout, reference, metrics, budgets, shapes, workloads."""
from __future__ import annotations

import json
import math
import unittest

import torch

from gpu1bit.shared import api, env


class BitLayout(unittest.TestCase):
    def test_pack_unpack_match_binary_export(self):
        import binary_export
        g = torch.Generator().manual_seed(1)
        for n, k in [(1, 1), (3, 7), (5, 8), (4, 9), (7, 33), (2, 1025)]:
            signs = (torch.randint(0, 2, (n, k), generator=g, dtype=torch.int8) * 2 - 1).to(torch.int8)
            codes = api.pack_signs(signs)
            self.assertTrue(torch.equal(codes, binary_export.pack_codes(signs)))
            self.assertTrue(torch.equal(api.unpack_signs(codes, k), binary_export.unpack_codes(codes, k)))
            self.assertTrue(torch.equal(api.unpack_signs(codes, k), signs))

    def test_lsb_first_bit_one_is_plus_one(self):
        codes = torch.tensor([[0b00000101, 0b1]], dtype=torch.uint8)  # K = 9: +1 at j = 0, 2, 8
        expected = torch.tensor([[1, -1, 1, -1, -1, -1, -1, -1, 1]], dtype=torch.int8)
        self.assertTrue(torch.equal(api.unpack_signs(codes, 9), expected))

    def test_tail_bits_rejected(self):
        codes = torch.zeros(2, 2, dtype=torch.uint8)
        codes[1, 1] = 0b10  # bit 9 of a K = 9 row
        with self.assertRaises(ValueError):
            api.unpack_bits(codes, 9)
        with self.assertRaises(ValueError):
            api.validate_weights(codes, torch.ones(2), None, 9)

    def test_validate_weights_rejections(self):
        c = torch.zeros(3, 2, dtype=torch.uint8)
        s = torch.ones(3)
        api.validate_weights(c, s, None, 16)
        for args in [(c.to(torch.int8), s, None, 16), (c, s, None, 17), (c, s, None, 8), (c, s.double(), None, 16),
                     (c, torch.ones(4), None, 16), (c, torch.tensor([1., math.inf, 1.]), None, 16),
                     (c, s, torch.ones(2), 16), (c, s, torch.tensor([0., math.nan, 0.]), 16), (c, s, None, 0)]:
            with self.assertRaises(ValueError, msg=str(args[3])):
                api.validate_weights(*args)


class Reference(unittest.TestCase):
    def test_fp64_reference_matches_explicit_sum(self):
        g = torch.Generator().manual_seed(2)
        n, k = 5, 13
        signs = (torch.randint(0, 2, (n, k), generator=g, dtype=torch.int8) * 2 - 1).to(torch.int8)
        codes = api.pack_signs(signs)
        scale = torch.rand(n, generator=g) + 0.5
        bias = torch.randn(n, generator=g)
        x = torch.randn(2, 3, k, generator=g)
        y = api.reference_linear(x, codes, scale, bias, k)
        for b in range(2):
            for t in range(3):
                for j in range(n):
                    s = sum(float(signs[j, i]) * float(x[b, t, i]) for i in range(k))
                    self.assertAlmostEqual(y[b, t, j].item(), float(scale[j]) * s + float(bias[j]), places=12)
        xc = torch.randn(2, k, 4, generator=g)
        yc = api.reference_conv(xc, codes, scale, None, k)
        self.assertTrue(torch.allclose(yc, api.reference_linear(xc.transpose(1, 2), codes, scale, None, k).transpose(1, 2)))

    def test_metrics(self):
        x = torch.tensor([[1., -2., 3.]])
        ref = torch.tensor([[2., 0.]], dtype=torch.float64)
        y = torch.tensor([[2.5, 0.25]])
        e = api.errors("linear", x, y, ref, torch.tensor([1., 0.5]))
        self.assertTrue(e["finite"])
        self.assertAlmostEqual(e["relative_frobenius"], math.hypot(0.5, 0.25) / 2.0, places=12)
        self.assertAlmostEqual(e["elementwise_guard"], max(0.5 / 6.0, 0.25 / 3.0), places=12)
        zero = api.errors("linear", x, torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.float64), torch.ones(2))
        self.assertEqual(zero["relative_frobenius"], 0.0)
        self.assertFalse(api.errors("linear", x, torch.tensor([[math.nan, 0.]]), ref, torch.ones(2))["finite"])
        th = api.numerics_thresholds({"r_bf16x3": {"relative_frobenius": 1e-7, "elementwise_guard": 4e-8},
                                      "r_expanded_int8x3": {"relative_frobenius": 3e-7, "elementwise_guard": 2e-8}})
        self.assertAlmostEqual(th["relative_frobenius"], 4.5e-7)
        self.assertAlmostEqual(th["elementwise_guard"], 6e-8)

    def test_budgets(self):
        self.assertEqual(api.code_budget_bytes(1024, 1024), (1024 * 128 * 11) // 10)
        self.assertEqual(api.code_budget_bytes(3, 9), (3 * 2 * 11) // 10)
        self.assertEqual(api.weight_budget_bytes(1024, 4096, True), (1024 * 512 * 11) // 10 + 8192)
        from gpu1bit.shared import audit
        self.assertEqual(audit.CODE_BUDGET_BYTES, 83_047_219)


class PublishedFiles(unittest.TestCase):
    def test_shapes_json_matches_manifest(self):
        shapes = json.loads((env.SHARED / "shapes.json").read_text())
        manifest = json.loads((env.MB1_EXPORT / "manifest.json").read_text())
        self.assertEqual(shapes["export_sha256"], manifest["sha256"])
        layers = {l["name"]: l for l in shapes["layers"]}
        self.assertEqual(len(layers), 264)
        for name, info in manifest["quantized_layers"].items():
            self.assertEqual([layers[name]["N"], layers[name]["K"]], info["shape"])
        self.assertEqual(sum(l["N"] * ((l["K"] + 7) // 8) for l in layers.values()), 75_497_472)
        self.assertEqual([c["encoder_frames_T"] for c in shapes["clips"]], [38, 126, 376])

    def test_workloads_are_dev_only_and_frozen(self):
        from gpu1bit.shared import workloads
        doc = workloads.load()
        self.assertEqual(len(doc["dev_clips"]), 3)
        self.assertEqual(len(doc["dev_24"]), 24)
        self.assertEqual(len(doc["dev_sample"]), 512)
        self.assertEqual(len({r["id"] for r in doc["dev_sample"]}), 512)
        self.assertEqual(len({r["id"] for r in doc["dev_24"]}), 24)
        for target in workloads.TARGETS:
            group = [r for r in doc["dev_24"] if r["target_seconds"] == target]
            self.assertEqual(len(group), 8)
        for r in doc["dev_clips"] + doc["dev_24"] + doc["dev_sample"] + doc["fixture_utterances"]:
            self.assertIn(r["corpus"], workloads.DEV_CORPORA)
            self.assertNotIn("test", r["audio_filepath"].lower().replace("/mnt/hd/wilderness-labs-stt", ""))
        # the frame-count formula reproduces the measured M of the historical clips
        self.assertEqual([workloads._frames(s)["M"] for s in (48000, 160000, 480640)], [38, 126, 376])


if __name__ == "__main__":
    unittest.main()
