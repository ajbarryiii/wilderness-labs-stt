"""reference.py against the saved NeMo FP32 golden outputs of surrogate seed 0 at full depth: DESIGN.md gate 1.

No NeMo import: the surrogate is regenerated from weight_stats.json (its manifest digest must equal
the one golden.py recorded), loaded into the reference as int8 codes + FP32 scales, and
golden.compare gates every clip (features, subsampling output, every layer, encoder output, token
and duration logits, LSTM h and c: rel <= 1e-5 and abs <= 1e-4; identical greedy decisions;
finite; input sensitivity). Golden directory: $PARAKEET_IOS_GOLDEN or <artifacts>/golden/seed0.
Full depth (about 4 GB), so on NixOS through ../heavy:

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
import randomweights as rw  # noqa: E402
import reference  # noqa: E402


class GoldenSeed0(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_grad_enabled(False)
        cls.dir = Path(os.environ.get("PARAKEET_IOS_GOLDEN", common.artifacts_dir() / "golden" / f"seed{common.GOLDEN_SEED}"))
        cls.meta = json.loads((cls.dir / "meta.json").read_text())
        stats = rw.load_stats()
        tensors, cls.digest = golden.tensor_source(stats, cls.meta["seed"])
        cls.ref = reference.build(reference.Config.from_model_config(stats["model_config"]))
        cls.load = reference.load_weights(cls.ref, tensors)
        del tensors

    def test_same_model(self) -> None:
        self.assertEqual(self.digest, self.meta["model_digest"])
        self.assertEqual(self.load["ternary_modules"], 264)

    def test_gate(self) -> None:
        for clip in self.meta["clips"]:
            with np.load(self.dir / clip["file"]) as data:
                g = {k: data[k] for k in data.files}
            self.assertEqual(common.sha256_array(g["audio"]), clip["audio_sha256"])
            metrics, failures = golden.compare(self.ref, g)
            common.report("golden_seed0_gate", clip=clip["id"], failures=failures, **metrics)
            self.assertEqual(failures, [], clip["id"])


def tearDownModule() -> None:
    common.report("test_reference_golden_peak", peak_rss_mb=common.peak_rss_mb())


if __name__ == "__main__":
    unittest.main()
