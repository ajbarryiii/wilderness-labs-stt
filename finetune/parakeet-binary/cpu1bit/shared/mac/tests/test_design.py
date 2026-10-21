"""Williams design properties for every k used (screen k = 2..9, confirm k = 2, tuning k = 6)."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import design as D  # noqa: E402


class Williams(unittest.TestCase):
    def test_balance(self):
        for k in range(2, 10):
            seqs = D.williams_sequences(k)
            self.assertEqual(len(seqs), k if k % 2 == 0 else 2 * k)
            for s in seqs:
                self.assertEqual(sorted(s), list(range(k)))
            b = D.balance(seqs, k)
            reps = 1 if k % 2 == 0 else 2
            for a in range(k):
                self.assertEqual(b["position_counts"][a], [reps] * k, k)
                for c in range(k):
                    if a != c:
                        self.assertEqual(b["adjacency_counts"][a][c], reps, (k, a, c))

    def test_design_ids_and_determinism(self):
        arms = ["R-MB1", "V"]
        d1 = D.design(arms, 6, 20261021, "confirm", {"threads": 8, "components": 2})
        d2 = D.design(arms, 6, 20261021, "confirm", {"threads": 8, "components": 2})
        self.assertEqual(d1, d2)
        self.assertEqual(len(d1), 6 * 2 * 2)
        self.assertEqual(len({b["block_id"] for b in d1}), len(d1))
        for r in range(6):
            blocks = [b for b in d1 if b["replicate"] == r]
            self.assertEqual(sorted(b["arm"] for b in blocks), ["R-MB1", "R-MB1", "V", "V"])


if __name__ == "__main__":
    unittest.main()
