"""reference.py against saved NeMo FP32 golden outputs at full depth (golden.py): DESIGN.md gate 1.

No NeMo import. For each benchmark model with goldens (surrogate seed0, M_P2), the reference is
built by models.load (seed0 regenerated from weight_stats.json, M_P2 read from the export as codes +
FP32 scales); its identity must equal the one golden.py recorded (surrogate manifest digest, export
SHA-256), and golden.compare gates every clip: features, subsampling output, every layer, encoder
output, token and duration logits, LSTM h and c (rel <= 1e-5 and abs <= 1e-4), identical greedy
decisions, finite outputs, and input sensitivity (rel >= 1e-2 and >= 100x the encoder parity
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
        else:
            self.assertEqual(prov["digest"], self.meta["model_digest"])
        self.assertEqual(prov["load"]["ternary_modules"], 264)

    def test_gate(self) -> None:
        for clip in self.meta["clips"]:
            with np.load(self.dir / clip["file"]) as data:
                g = {k: data[k] for k in data.files}
            self.assertEqual(common.sha256_array(g["audio"]), clip["audio_sha256"])
            metrics, failures = golden.compare(self.ref, g)
            common.report(f"golden_{self.name}_gate", clip=clip["id"], failures=failures, **metrics)
            self.assertEqual(failures, [], clip["id"])


class MP2(GoldenGate, unittest.TestCase):
    name = "mp2"


class Seed0(GoldenGate, unittest.TestCase):
    name = f"seed{common.GOLDEN_SEED}"


def tearDownModule() -> None:
    common.report("test_reference_golden_peak", peak_rss_mb=common.peak_rss_mb())


if __name__ == "__main__":
    unittest.main()
