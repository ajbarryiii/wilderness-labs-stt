"""Contract tests (CPU): the stage 4 report's preregistered gate and its comparisons (keab-prime plan "Scoring and
gate"), on synthetic per-utterance records in the scorer's format.

- the gate is computed on the deployed-mode (x1) records of b100-main (Kea-B') and b100-control: delta = H4(control) -
  H4(Kea-B') with a paired cluster-bootstrap interval over the H4 sets (clusters as training/devsplit.py); per H4 set
  WER(Kea-B') - WER(control); the decision follows training/stage4.gate;
- kernel fidelity's token-sequence mismatch is matched by utterance id;
- a section whose inputs do not exist is pending, not an error.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import keab_report as kr
import train_b0prime as tb


def recs(name: str, errors: list[int], words: int = 10) -> list[dict]:
    """Records of one H4 set, one utterance per cluster (speaker or meeting ids in devsplit's form)."""
    out = []
    for k, e in enumerate(errors):
        cluster = {"librispeech_dev_clean": f"{100 + k}-1-{k}", "librispeech_dev_other": f"{200 + k}-1-{k}",
                   "ami_dev": f"ami:AMI_ES20{k:02d}a_H00_FEE00{k}_0000000_0000100",
                   "voxpopuli_dev": f"voxpopuli:2020{k:04d}-0900-PLENARY-11-en_{k}"}[name]
        out.append({"id": cluster, "ref_norm": " ".join(["w"] * words), "hyp_norm": "", "S": e, "D": 0, "I": 0,
                    "scored": True, "ref": "", "hyp": "", "tokens": [e]})
    return out


class Gate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, model: str, mode: str, errors: dict[str, list[int]]):
        d = self.root / model / mode
        d.mkdir(parents=True, exist_ok=True)
        for name, e in errors.items():
            (d / f"{name}.json").write_text(json.dumps({"records": recs(name, e)}))

    def dirs(self):
        return {"Kea-B'": {m: self.root / "b100-main" / m for m in ("x1", "x3", "torch")},
                "control": {m: self.root / "b100-control" / m for m in ("x1", "x3", "torch")}}

    def test_the_gate_on_deployed_records(self):
        n = 30
        control = {s: [2] * n for s in tb.H4}
        main = {s: [1] * n for s in tb.H4}                                  # 10 points better on every set
        self.write("b100-main", "x1", main)
        self.write("b100-control", "x1", control)
        self.write("b100-main", "torch", control)                         # never read by the gate
        with mock.patch.object(kr, "score_dirs", self.dirs), mock.patch.object(kr, "domain_means", lambda m: {}):
            g = kr.gate_section()
        self.assertEqual(g["status"], "complete")
        self.assertAlmostEqual(g["delta"]["h4"]["point"], 10.0)
        self.assertGreater(g["delta"]["h4"]["ci"][0], 0)
        self.assertTrue(all(abs(v + 10.0) < 1e-9 for v in g["per_set_regression"].values()))
        self.assertTrue(g["decision"]["pass"])

    def test_a_worse_set_fails_the_gate(self):
        n = 30
        control = {s: [3] * n for s in tb.H4}
        main = {s: [1] * n for s in tb.H4}
        main["ami_dev"] = [7] * n                                         # 40 points worse on AMI dev
        self.write("b100-main", "x1", main)
        self.write("b100-control", "x1", control)
        with mock.patch.object(kr, "score_dirs", self.dirs), mock.patch.object(kr, "domain_means", lambda m: {}):
            g = kr.gate_section()
        self.assertFalse(g["decision"]["no_large_regression"])
        self.assertFalse(g["decision"]["pass"])

    def test_pending_without_scores(self):
        with mock.patch.object(kr, "score_dirs", self.dirs):
            self.assertTrue(kr.gate_section()["status"].startswith("pending"))

    def test_mismatch_by_id(self):
        a = [{"id": "x", "tokens": [1, 2]}, {"id": "y", "tokens": [3]}, {"id": "z", "tokens": [4]}]
        b = [{"id": "y", "tokens": [3]}, {"id": "x", "tokens": [1, 5]}]
        self.assertEqual(kr.mismatch(a, b), {"utterances": 2, "mismatches": 1, "rate": 0.5})


if __name__ == "__main__":
    unittest.main()
