"""reference.py against saved NeMo FP32 golden outputs at full depth (golden.py): DESIGN.md gate 1.

No NeMo import. For each benchmark model with goldens (B0, M_P2, surrogate seed0), the reference
is built by models.load (B0 from the .nemo, M_P2 from the export as codes + FP32 scales, seed0
regenerated from weight_stats.json); its identity must equal the one golden.py recorded (.nemo
SHA-256, export SHA-256, surrogate manifest digest), the clip list and depth must be the expected
ones, and golden.compare gates every clip end to end: features, subsampling output, every layer,
encoder output (also fed the golden features), token and duration logits, LSTM h and c (rel <=
1e-5 and abs <= 1e-4), identical greedy decisions, finite outputs, and input sensitivity against
the reference's own unperturbed output (finite rel >= 1e-2 and >= 100x the encoder parity
error). Golden root: $PARAKEET_IOS_GOLDEN or <artifacts>/golden. Full depth (about 4 GB per model,
one at a time), so on NixOS through ../heavy:

  ../heavy ios-wp1-golden-check --mem-max 10G --runtime 40min --wait -- env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 \
      MKL_NUM_THREADS=4 <repo>/finetune/parakeet-ternary/python <repo>/finetune/parakeet-ternary/ios/tests/test_reference_golden.py -v
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import golden  # noqa: E402
import models  # noqa: E402


class GoldenGate:
    name = ""

    @classmethod
    def setUpClass(cls) -> None:
        torch.set_grad_enabled(False)
        cls.dir = Path(os.environ.get("PARAKEET_IOS_GOLDEN", common.artifacts_dir() / "golden")) / cls.name
        cls.meta = json.loads((cls.dir / "meta.json").read_text())
        cls.ref = models.load(cls.name)

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.ref

    def test_same_model(self) -> None:
        prov = self.ref.provenance
        if self.name == "mp2":
            self.assertEqual(prov["export_sha256"], self.meta["export_sha256"])
        elif self.name == "b0":
            self.assertEqual(prov["nemo_sha256"], self.meta["nemo_sha256"])
        else:
            self.assertEqual(prov["digest"], self.meta["model_digest"])
        self.assertEqual(prov["load"]["ternary_modules"], 0 if self.name == "b0" else 264)

    def test_gate(self) -> None:
        self.assertEqual(tuple(c["id"] for c in self.meta["clips"]), common.EXPECTED_CLIP_IDS)
        self.assertEqual(self.meta["layers"], 24)
        self.assertEqual(self.ref.cfg.n_layers, 24)
        checked = 0
        for clip in self.meta["clips"]:
            with np.load(self.dir / clip["file"]) as data:
                g = {k: data[k] for k in data.files}
            self.assertEqual(common.sha256_array(g["audio"]), clip["audio_sha256"])
            metrics, failures = golden.compare(self.ref, g)
            common.report(f"golden_{self.name}_gate", clip=clip["id"], failures=failures, **metrics)
            self.assertEqual(failures, [], clip["id"])
            checked += 1
        self.assertEqual(checked, len(common.EXPECTED_CLIP_IDS))


class B0(GoldenGate, unittest.TestCase):
    name = "b0"


class MP2(GoldenGate, unittest.TestCase):
    name = "mp2"


class Seed0(GoldenGate, unittest.TestCase):
    name = f"seed{common.GOLDEN_SEED}"


def tearDownModule() -> None:
    common.report("test_reference_golden_peak", peak_rss_mb=common.peak_rss_mb())


if __name__ == "__main__":
    unittest.main()
