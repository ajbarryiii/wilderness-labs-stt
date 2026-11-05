"""Opt-in contract test on the real Kea-B 0.1 files (release-kea-b-0.1 plan "Loader contract"): load_kea_b.py rebuilds
the same FP32 state, bit for bit, from stage 4's bundle, from its scoring view, and as binary_export.load_export
(the K-fp test procedure) does from that view. CPU, about 10 GB of memory; run it as a job:

    RELABEL_HF_REAL_TEST=1 ./python -m unittest discover -s tests -p test_hf_release_real.py
"""
from __future__ import annotations

import os
import unittest

import torch

import keab_test_paths as tp
from hf import load_kea_b as lk
from training import keab_export  # noqa: F401  (puts finetune/parakeet-binary on sys.path)

import binary_export  # noqa: E402


@unittest.skipUnless(os.environ.get("RELABEL_HF_REAL_TEST") == "1", "set RELABEL_HF_REAL_TEST=1 (needs ~10 GB)")
class RealFiles(unittest.TestCase):
    def test_bundle_view_and_load_export_states_are_identical(self):
        self.assertEqual(lk.detect_format(tp.BUNDLE), lk.BUNDLE_FORMAT)
        self.assertEqual(lk.detect_format(tp.VIEW), lk.VIEW_FORMAT)
        with torch.no_grad():
            bundle = lk.load(tp.BUNDLE).state_dict()
            view = lk.load(tp.VIEW).state_dict()
            self.assertEqual(list(bundle), list(view))
            self.assertEqual([k for k in bundle if not torch.equal(bundle[k], view[k])
                              or bundle[k].dtype != view[k].dtype], [])
            del view
            reference = binary_export.load_export(tp.VIEW, "cpu").state_dict()
            self.assertEqual(list(bundle), list(reference))
            self.assertEqual([k for k in bundle if not torch.equal(bundle[k], reference[k])
                              or bundle[k].dtype != reference[k].dtype], [])


if __name__ == "__main__":
    unittest.main()
