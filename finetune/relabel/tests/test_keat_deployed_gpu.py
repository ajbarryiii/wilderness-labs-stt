"""Contract tests (CUDA; M1's real export on the real x1 stack): the stage 5 x1 loader training/keat_deployed.py
(keat-prime plan "Scoring and gate", "The loader").

- x1 on M1's export: one provider, a fulldev.PartCount at n = 1; scoring a few dev utterances records projection calls,
  every one at one part, the stack's hard-coded three-part pointwise_conv2 calls included;
- x3: the same builder at 3 with no adapter;
- dither and pad_to are 0 while loaded and restored on close;
- an arbitrary export directory (a copy of M1's export elsewhere under $RELABEL_ROOT) is accepted and decodes the
  same tokens as M1's own path.
Run inside a GPU job under ./python-gpu1bit with CUBLAS_WORKSPACE_CONFIG unset (training/keat_launch.sh gpu-tests).
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

import keab_step0 as k0
import relabel_paths
from training import keat_deployed as tdep
from training import keat_m1 as km1


def wrapper_ok() -> bool:
    try:
        tdep.require_wrapper()
        return True
    except RuntimeError:
        return False


HAS = torch.cuda.is_available() and km1.m1_export().exists() and wrapper_ok() \
    and not os.environ.get("CUBLAS_WORKSPACE_CONFIG")
SET, N = "librispeech_dev_clean", 4


def tokens(model) -> list[list[int]]:
    recs = k0.deployed_records(SET, "dev", N)
    out = tdep.score_records(model, recs, SET, "keat-loader-test")
    return [r["tokens"] for r in out["records"]]


@unittest.skipUnless(HAS, "needs CUDA, M1's export and ./python-gpu1bit with CUBLAS_WORKSPACE_CONFIG unset")
class RealStack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = relabel_paths.ARTIFACTS / "tmp"
        base.mkdir(parents=True, exist_ok=True)
        cls.tmp = Path(tempfile.mkdtemp(prefix="keat-x1-gpu-", dir=base))
        cls.reference = None

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def m1_tokens(self):
        if RealStack.reference is None:
            model = tdep.load_x1(km1.m1_export())
            try:
                RealStack.reference = tokens(model)
            finally:
                tdep.close(model)
        return RealStack.reference

    def test_x1_one_partcount_at_one_part_on_every_call(self):
        from gpu1bit.shared import fulldev
        model = tdep.load_x1(km1.m1_export())
        try:
            featurizer = model.preprocessor.featurizer
            saved = dict(tdep.info(model)["featurizer_saved"])
            self.assertEqual((featurizer.dither, featurizer.pad_to), (0.0, 0))
            facts = tdep.info(model)
            self.assertEqual((facts["parts"], facts["provider"]), (1, "PartCount"))
            self.assertEqual(facts["export_sha256"], km1.M1_EXPORT_SHA256)
            self.assertEqual(facts["decoder_cuda_graphs_mode"], "full_graph")
            rows = tokens(model)
            self.assertEqual(len(rows), N)
            calls = tdep.part_calls(model)
            print(f"\nx1 projection calls: {calls}")
            self.assertTrue(calls, "the provider was called")
            self.assertEqual({k.split("->")[1] for k in calls}, {"1"})
            self.assertIn("3->1", calls, "the stack's three-part calls run at one part")
            self.assertIsInstance(tdep.provider(model), fulldev.PartCount)
            self.assertEqual(tdep.provider(model).n, 1)
        finally:
            tdep.close(model)
        self.assertEqual({k: getattr(featurizer, k) for k in saved}, saved)
        self.assertNotEqual(saved["dither"], 0.0, "NeMo's own dither is restored")

    def test_x3_without_the_adapter(self):
        from gpu1bit.shared import fulldev
        model = tdep.load_x3(km1.m1_export())
        try:
            self.assertNotIsInstance(tdep.provider(model), fulldev.PartCount)
            self.assertEqual(tdep.info(model)["parts"], 3)
            self.assertEqual(len(tokens(model)), N)
        finally:
            tdep.close(model)

    def test_arbitrary_export_directory(self):
        copy = self.tmp / "elsewhere" / "export"
        shutil.copytree(km1.m1_export(), copy)
        model = tdep.load_x1(copy)
        try:
            self.assertEqual(tdep.info(model)["export"], str(copy.resolve()))
            got = tokens(model)
        finally:
            tdep.close(model)
        self.assertEqual(got, self.m1_tokens())


if __name__ == "__main__":
    unittest.main()
