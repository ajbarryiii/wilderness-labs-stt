"""eval-stem contracts (eval-only STEM lectures): one part per subject, outside the default dev sets, with
technical critical-term recall.

Run from finetune/relabel:  ./python -m unittest discover -s tests -p "test_devsets_*" -v
"""
from __future__ import annotations

import unittest

import devsets


class SetsTest(unittest.TestCase):
    def test_eval_stem_has_one_part_per_subject(self) -> None:
        self.assertEqual(devsets.SETS["eval-stem"], ("stem-chem", "stem-math", "stem-cs"))
        self.assertNotIn("eval-stem", devsets.DEFAULT_SETS)
        self.assertEqual(devsets.TERM_KIND["eval-stem"], "technical")
        self.assertEqual(devsets.TERM_KIND["dev-medical"], "medical")
        self.assertEqual(devsets.set_of("stem-cs"), "eval-stem")
        self.assertEqual(len(set(devsets.PARTS)), len(devsets.PARTS))


if __name__ == "__main__":
    unittest.main()
