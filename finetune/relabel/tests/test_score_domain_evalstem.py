"""score_domain contracts for eval-stem: scored only on request, into scores/eval-stem/<model>, with the ami_dev
cross-check, and critical-term recall measured with the TECHNICAL term list (not the medical one).

Run from finetune/relabel:  ./python -m unittest discover -s tests -p "test_score_domain_*" -v
"""
from __future__ import annotations

import hashlib
import json
import unittest

import devsets
import score_domain as sd
from devsets import layout
from test_score_domain_contracts import RunFixture


class EvalStemRunTest(RunFixture):
    def test_eval_stem_uses_the_technical_term_list(self) -> None:
        for part in devsets.SETS["eval-stem"]:
            manifest = self.lay.manifest(part)
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            manifest.write_text("".join(json.dumps({**r, "text": f"the eigenvalue and ketamine {i}"}) + "\n"
                                        for i, r in enumerate(rows)))
            meta = json.loads(self.lay.meta(part).read_text())
            meta["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            self.lay.meta(part).write_text(json.dumps(meta))
        self.assertEqual(sd.score("B0", ["eval-stem"], None, None, "cuda", 5, False), 0)
        out = layout.scores_root() / "eval-stem" / "B0"
        summary = json.loads((out / "summary.json").read_text())
        self.assertIn("crosscheck", summary)
        for part in devsets.SETS["eval-stem"]:
            # eigenvalue (technical) in every row, reproduced; ketamine (medical) must not be counted
            self.assertEqual(summary["parts"][part]["term_recall"],
                             {"occurrences": 3, "reproduced": 3, "recall": 1.0})
        self.assertEqual(summary["sets"]["eval-stem"]["mean_term_recall"], 1.0)
        self.assertFalse((layout.scores_root() / "B0").exists())


if __name__ == "__main__":
    unittest.main()
