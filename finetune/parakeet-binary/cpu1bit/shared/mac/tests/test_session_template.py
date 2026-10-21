"""The session template covers the specified condition matrix and arms (Codex code review c1 finding 9)."""
from __future__ import annotations

import json
import os
import unittest

TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "session-template.json")


class Template(unittest.TestCase):
    def setUp(self):
        with open(TEMPLATE) as f:
            self.t = json.load(f)

    def test_condition_matrix_and_descriptive_arms(self):
        screens = {k: v for k, v in self.t["stages"].items() if k.startswith("screen")}
        conds = {(v["condition"]["threads"], v["condition"]["components"]) for v in screens.values()}
        self.assertEqual(conds, {(8, 2), (4, 2), (8, 3), (4, 3)})
        for k, v in screens.items():
            self.assertIn("B1T-MB1", v["arms"], k)
            self.assertIn("R-MB1", v["arms"], k)
            self.assertEqual(v["candidates"], ["V", "T", "P"], k)
            self.assertNotIn("B1T-MB1", v["candidates"])
        self.assertEqual(self.t["stages"]["screen"]["condition"], {"threads": 8, "components": 2})

    def test_arms_exist_and_command_stage(self):
        for k, v in self.t["stages"].items():
            for a in v.get("arms", []):
                if a != "CANDIDATE":
                    self.assertIn(a, self.t["arms"], (k, a))
        isa = self.t["stages"]["p-isa"]
        self.assertEqual(isa["type"], "command")
        self.assertEqual(isa["argv"][1:], ["--qos-ui", "isa"])          # c2 finding 4
        self.assertEqual((isa["qos_readback"]["class"], isa["qos_readback"]["rel"]), ("0x21", 0))
        import re
        self.assertEqual(re.search(isa["qos_readback"]["regex"],
                                   "# p-ubench isa (x); main-thread qos class 0x21 rel 0").groups(), ("0x21", "0"))
        self.assertIn("p_cluster_min_rise", isa["placement"])
        self.assertEqual(self.t["stages"]["confirm"]["replicates"], 6)
        self.assertEqual(len(self.t["stages"]["tune-bneon"]["arms"]), 6)


if __name__ == "__main__":
    unittest.main()
