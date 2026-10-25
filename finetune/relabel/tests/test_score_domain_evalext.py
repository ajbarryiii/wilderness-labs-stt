"""score_domain contracts for eval-ext: the default run scores exactly the two domain dev sets (the queued
B0/M1/MB1 jobs), and eval-ext is scored only on request, into its own directory, with the same ami_dev
cross-check and with critical-term recall.

Run from finetune/relabel:  ./python -m unittest discover -s tests -p "test_score_domain_*" -v
"""
from __future__ import annotations

import json
import unittest

import devsets
import score_domain as sd
from devsets import layout
from test_score_domain_contracts import RunFixture


class DefaultsTest(unittest.TestCase):
    def test_default_sets_and_paths_are_unchanged(self) -> None:
        self.assertEqual(sd.parse_args(["--model", "B0"]).sets, ["dev-medical", "dev-technical"])
        self.assertEqual(sd.output_dir("B0", None, None), layout.scores_root() / "B0")
        self.assertEqual(sd.output_dir("B0", None, None, ["dev-medical", "dev-technical"]), layout.scores_root() / "B0")
        ext = sd.output_dir("B0", None, None, ["eval-ext"])
        self.assertEqual(ext, layout.scores_root() / "eval-ext" / "B0")
        self.assertNotEqual(sd.output_dir("B0", 4, None, ["eval-ext"]), sd.output_dir("B0", 4, None))
        self.assertEqual(sd.parse_args(["--model", "M1", "--sets", "eval-ext"]).sets, ["eval-ext"])


class EvalExtRunTest(RunFixture):
    def test_eval_ext_scored_separately_with_crosscheck_and_term_recall(self) -> None:
        self.assertEqual(sd.score("MB1", ["eval-ext"], None, None, "cuda", 5, False), 0)
        out = layout.scores_root() / "eval-ext" / "MB1"
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(sorted(summary["parts"]), sorted(devsets.SETS["eval-ext"]))
        self.assertIn("crosscheck", summary)
        self.assertAlmostEqual(summary["sets"]["eval-ext"]["mean_term_recall"], 2 / 3)
        self.assertFalse((layout.scores_root() / "MB1").exists())
        self.assertEqual(sd.score("MB1", list(devsets.DEFAULT_SETS), None, None, "cuda", 5, False), 0)
        default = json.loads((layout.scores_root() / "MB1" / "summary.json").read_text())
        self.assertEqual(sorted(default["sets"]), ["dev-medical", "dev-technical"])
        self.assertEqual(json.loads((out / "summary.json").read_text()), summary)


if __name__ == "__main__":
    unittest.main()
