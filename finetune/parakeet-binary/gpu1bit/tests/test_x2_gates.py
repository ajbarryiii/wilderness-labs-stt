"""Contract tests of the x2 dev gates (gpu1bit/x2/gates.py; plans/x2-default.md "Gates", X4, Y2; CPU):
  - roster g0 to g5: model, export, stack, parts, fulldev arm and role as in X4/Y2. The order is
    references first: MB1 g0, g2, g1, g3; then M1 g5, g4.
  - launch: require_launch refuses any arm whose reference chain is not in place (g2 before g0 has
    passed R2, g1/g3 before g2, g4 before g5). The drive calls it before every arm.
  - the MB1 gate (g1 against g2), frozen bounds:
    - mean within 0.02 points and every set within 0.10 points, both directions;
    - token mismatches at most 0.5% pooled and 1.0% per set;
    - encoder relative Frobenius at most 1e-3 per utterance on the 512 sample;
    - R5: finite, decoded, encoder lengths equal to g3's.
    Every mismatching utterance is listed.
  - the M1 gate: g5 passes R2; g4 against g5 is recognition-equivalent (the activation-precision
    reading); g4 is R5-valid.
  - an exact-equality row comparison: identical rows pass; each criterion fails on its own.
  - drive order: require_launch then the arm, for g0, g2, g1, g3, g5, g4; the two encoder comparisons; the
    evaluation; then the Z4 descriptive comparisons with ap-20261020r4 (g2 with MB1-int8_mma-x2, g4 with
    M1-x2), which never gate: a refused recorded binding is written down and the gate record returned
    by drive is the evaluation's, unchanged.

    gpu1bit/shared/python -m gpu1bit.tests.run --cpu gpu1bit.tests.test_x2_gates
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import unittest

from gpu1bit.shared import fulldev
from gpu1bit.x2 import gates as G

SETS = fulldev.DEV_SETS


def _rows(n_per_set=50, tokens=(1, 2, 3), enc_len=10):
    out = {}
    for s in SETS:
        out[s] = [{"id": f"{s}:{i}", "tokens": list(tokens), "hyp": "a b", "enc_len": enc_len, "finite": True,
                   "decode_error": None, "timestamps": [0, 1], "S": 0, "D": 0, "I": 0, "N": 2, "scored": True,
                   "ref_norm": "a b"} for i in range(n_per_set)]
    return out


def _summary(wers, valid=True, cross=None):
    return {"dev_wer_percent": dict(zip(SETS, wers)), "dev_mean_wer_percent": sum(wers) / len(wers),
            "structurally_valid": valid, "cross_check": cross, "empty_hypotheses": {s: 0 for s in SETS}}


ENC_OK = {"utterances": 512, "max_rel_frobenius_valid": 2e-4, "not_finite": [], "length_differences": []}


class Roster(unittest.TestCase):
    def test_rows_and_order(self):
        r = {g["id"]: g for g in G.GATE_ROSTER}
        self.assertEqual(sorted(r), ["g0", "g1", "g2", "g3", "g4", "g5"])
        self.assertEqual((r["g0"]["model"], r["g0"]["arm"], r["g0"]["parts"]), ("MB1", "MB1-int8_mma-x3", 3))
        self.assertEqual((r["g1"]["arm"], r["g1"]["parts"], r["g1"]["reference"]), ("MB1-B-hybrid-x2", 2, "g2"))
        self.assertEqual((r["g2"]["arm"], r["g2"]["reference"]), ("MB1-int8_mma-x2", "g0"))
        self.assertEqual((r["g3"]["arm"], r["g3"]["parts"]), ("MB1-B-hybrid-x3", 3))
        self.assertEqual((r["g4"]["arm"], r["g4"]["reference"]), ("M1-x2", "g5"))
        self.assertEqual(r["g5"]["arm"], "M1-x3")
        self.assertEqual(G.ORDER, ["g0", "g2", "g1", "g3", "g5", "g4"])
        for g in G.GATE_ROSTER:
            self.assertEqual(fulldev.parse_arm(g["arm"])["model"], g["model"])


class Launch(unittest.TestCase):
    def setUp(self):
        from gpu1bit.actprec import parts
        self.run = f"test-x2-gates-{os.getpid()}"
        self.ctx = parts.installed()
        self.ctx.__enter__()
        self.counts = {n: len(fulldev.subset_records(n, "full")) for n in SETS}

    def tearDown(self):
        self.ctx.__exit__(None, None, None)
        shutil.rmtree(fulldev.ROOT / self.run, ignore_errors=True)

    def done(self, arm, **over):
        h = fulldev.freeze(self.run, arm, "full")
        s = {"run": self.run, "arm": arm, "subset": "full", "identity": h, "complete": True, "utterances": dict(self.counts),
             "structurally_valid": True, "spec": fulldev.parse_arm(arm),
             "cross_check": {"pass": True} if arm in ("M1-x3", "MB1-int8_mma-x3") else None}
        s.update(over)
        (fulldev.arm_dir(self.run, arm, "full") / "summary.json").write_text(json.dumps(s))

    def test_references_first(self):
        G.require_launch(self.run, "g0")
        G.require_launch(self.run, "g5")
        for gid in ("g2", "g1", "g3", "g4"):
            with self.subTest(gid=gid), self.assertRaisesRegex(SystemExit, "R2"):
                G.require_launch(self.run, gid)
        self.done("MB1-int8_mma-x3")
        G.require_launch(self.run, "g2")
        for gid in ("g1", "g3"):
            with self.assertRaisesRegex(SystemExit, "R2"):
                G.require_launch(self.run, gid)
        self.done("MB1-int8_mma-x2")
        G.require_launch(self.run, "g1")
        G.require_launch(self.run, "g3")
        self.done("M1-x3", cross_check={"pass": False})
        with self.assertRaisesRegex(SystemExit, "R2"):
            G.require_launch(self.run, "g4")
        self.done("M1-x3")
        G.require_launch(self.run, "g4")

    def test_drive_checks_before_every_arm(self):
        import inspect
        src = inspect.getsource(G.drive)
        self.assertIn("require_launch(", src)


class MB1Gate(unittest.TestCase):
    def base(self):
        rows = _rows()
        return dict(g1=_summary([2.0, 4.0, 10.0, 6.0, 8.0]), g2=_summary([2.0, 4.0, 10.0, 6.0, 8.0]),
                    g1_rows=rows, g2_rows=copy.deepcopy(rows), g3_rows=copy.deepcopy(rows), enc=dict(ENC_OK))

    def test_identical_rows_pass(self):
        out = G.mb1_gate(**self.base())
        self.assertTrue(out["pass"], out)
        self.assertEqual(out["token_mismatches"]["pooled"], 0)

    def test_each_criterion(self):
        cases = {
            "mean +0.021": lambda b: b.update(g1=_summary([2.105, 4.0, 10.0, 6.0, 8.0])),
            "mean -0.021 (two-sided)": lambda b: b.update(g1=_summary([1.895, 4.0, 10.0, 6.0, 8.0])),
            "a set better by 0.11 (two-sided)": lambda b: b.update(g1=_summary([1.89, 4.05, 10.05, 6.0, 8.0])),
            "churn in every set (2% each)": lambda b: self._churn(b, per_set=1, sets=SETS),
            "one set churn 1.2% (pooled 0.24%)": lambda b: self._churn(b, per_set=1, sets=[SETS[0]], n_frac=0.012),
            "encoder 1.1e-3": lambda b: b["enc"].update(max_rel_frobenius_valid=1.1e-3),
            "encoder not finite": lambda b: b["enc"].update(not_finite=["x"]),
            "R5 lengths vs g3": lambda b: b["g3_rows"][SETS[1]][3].update(enc_len=11),
            "R5 not finite": lambda b: b["g1_rows"][SETS[2]][0].update(finite=False),
            "R5 decode error": lambda b: b["g1_rows"][SETS[2]][0].update(decode_error="boom"),
        }
        for name, edit in cases.items():
            with self.subTest(case=name):
                b = self.base()
                edit(b)
                self.assertFalse(G.mb1_gate(**b)["pass"], name)

    def _churn(self, b, per_set, sets, n_frac=0.006):
        for s in sets:
            k = max(1, round(n_frac * len(b["g1_rows"][s])))
            for r in b["g1_rows"][s][:k]:
                r["tokens"] = [9]

    def test_mismatches_are_listed(self):
        b = self.base()
        b["g1_rows"][SETS[4]][7]["tokens"] = [5]
        out = G.mb1_gate(**b)
        self.assertEqual(out["mismatch_ids"], [f"{SETS[4]}:7"])


class M1Gate(unittest.TestCase):
    def test_reading(self):
        rows = _rows()
        cmp = G.row_comparison(rows, copy.deepcopy(rows))
        g5 = _summary([2.0, 4.0, 10.0, 6.0, 8.0], cross={"pass": True})
        ok = G.m1_gate(g5, _summary([2.0, 4.0, 10.0, 6.0, 8.0]), cmp, dict(ENC_OK))
        self.assertTrue(ok["pass"], ok)
        self.assertFalse(G.m1_gate(dict(g5, cross_check={"pass": False}), _summary([2.0, 4.0, 10.0, 6.0, 8.0]), cmp,
                                   dict(ENC_OK))["pass"])
        self.assertFalse(G.m1_gate(g5, _summary([2.03, 4.03, 10.03, 6.03, 8.03]), cmp, dict(ENC_OK))["pass"])
        self.assertFalse(G.m1_gate(g5, _summary([2.0, 4.0, 10.0, 6.0, 8.0], valid=False), cmp, dict(ENC_OK))["pass"])


class DriveOrderAndRecorded(unittest.TestCase):
    def test_order_and_descriptive_recorded_comparisons(self):
        import subprocess
        import tempfile
        from pathlib import Path
        from gpu1bit.actprec import parts
        from gpu1bit.x2 import recorded_arm as RA
        calls = []
        record = {"run": "R", "pass": True}
        saved = (G.require_launch, parts.drive, parts.install, subprocess.run, G.evaluate, RA.compare_with_recorded, G.out_dir,
                 os.environ.get("GPU1BIT_SELECTION"))
        tmp = Path(tempfile.mkdtemp())

        class _P:
            returncode = 0

        def compare(run, arm, rec_run, rec_arm, pins=None, **kw):
            calls.append(("recorded", arm, rec_run, rec_arm))
            if arm == "M1-x2":
                raise SystemExit("recorded arm refused: modified row file")
            return {"descriptive_only": True, "pooled": {"utterances": 10, "differing": 0}}
        G.require_launch = lambda run, gid: calls.append(("require", gid))
        parts.drive = lambda args: calls.append(("drive", args.arm))
        parts.install = lambda *a, **k: None
        subprocess.run = lambda cmd, **kw: (calls.append(("encoder", cmd[cmd.index("--arm") + 1])), _P())[1]
        G.evaluate = lambda run: (calls.append(("evaluate",)), dict(record))[1]
        RA.compare_with_recorded = compare
        G.out_dir = lambda: tmp
        try:
            out = G.drive("R")
        finally:
            (G.require_launch, parts.drive, parts.install, subprocess.run, G.evaluate, RA.compare_with_recorded, G.out_dir,
             sel) = saved
            if sel is None:
                os.environ.pop("GPU1BIT_SELECTION", None)
            else:
                os.environ["GPU1BIT_SELECTION"] = sel
        order = [c for c in calls if c[0] in ("require", "drive")]
        want = []
        for gid in G.ORDER:
            want += [("require", gid), ("drive", G.ARM[gid])]
        self.assertEqual(order, want)
        kinds = [c[0] for c in calls]
        self.assertLess(max(i for i, k in enumerate(kinds) if k == "drive"), kinds.index("encoder"))
        self.assertLess(max(i for i, k in enumerate(kinds) if k == "encoder"), kinds.index("evaluate"))
        self.assertLess(kinds.index("evaluate"), kinds.index("recorded"))
        self.assertEqual(sorted(c[1:] for c in calls if c[0] == "recorded"),
                         [("M1-x2", "ap-20261020r4", "M1-x2"), ("MB1-int8_mma-x2", "ap-20261020r4", "MB1-int8_mma-x2")])
        self.assertEqual(out, record, "the recorded comparisons never change the gate record")
        doc = json.loads((tmp / "recorded-R.json").read_text())
        self.assertTrue(doc["descriptive_only"])
        self.assertIn("refused", doc["comparisons"]["g4"]["error"])
        self.assertEqual(doc["comparisons"]["g2"]["pooled"]["differing"], 0)
        shutil.rmtree(tmp, ignore_errors=True)
