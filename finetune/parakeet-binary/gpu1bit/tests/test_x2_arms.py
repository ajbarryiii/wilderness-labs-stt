"""Contract tests of the x2 arms, loaders and gate references (plans/x2-default.md X1, X3, X4, Y1, Y2; CPU):
  - PartCount forwards attribute WRITES to the inner provider for every name except its own
    (provider, n, record); reads keep delegating (X1).
  - B-hybrid arms (MB1-B-hybrid-x2, MB1-B-hybrid-x3) are activation-precision arms:
    - enabled at x2 and x3 only;
    - built by gpu1bit.runtime.load_binary_packed(runtime="hybrid", parts=n) under the selection in use;
    - their identity hashes the actprec patch files, B's pinned part-count files at x2, the hybrid
      loader and the selection content.
  - Gate references (Y2), with g2 (MB1-int8_mma-x2) and g4 (M1-x2) referring to x3 arms that must pass
    R2, and g1/g3 (MB1-B-hybrid-*) referring to g2, whose own chain must hold. require_reference
    refuses a reference that is missing, incomplete, of another identity, of another model, or failed
    R2 (directly or down the chain).
  - The drive's x3 R2 stop applies to the R2 reference arms (M1-x3, MB1-int8_mma-x3), not to g3.
  - Loaders (X3, Y1):
    - load_binary_packed and runtime_m1.load_ternary_packed take parts= and selection=;
    - the default part count comes from the selection document in use (default_parts,
      m1_default_parts; absent = 3);
    - any part count other than 2 and 3 is refused before any model work;
    - parakeet-ternary/README.md points to runtime_m1.

    gpu1bit/shared/python -m gpu1bit.tests.run --cpu gpu1bit.tests.test_x2_arms
"""
from __future__ import annotations

import inspect
import json
import os
import shutil
import tempfile
import types
import unittest
from pathlib import Path

from gpu1bit.shared import env, fulldev, seldoc


class _Inner:
    def __init__(self):
        self.defer_epilogue = False
        self.no_defer_weights = set()
        self.calls = []

    def __call__(self, *a):
        self.calls.append(a)
        return "y"


class PartCountWrites(unittest.TestCase):
    def test_writes_reach_the_inner_provider(self):
        inner = _Inner()
        pc = fulldev.PartCount(inner, 2)
        pc.defer_epilogue = True
        pc.no_defer_weights = {1, 2}
        self.assertIs(inner.defer_epilogue, True)
        self.assertEqual(inner.no_defer_weights, {1, 2})
        self.assertNotIn("defer_epilogue", vars(pc))
        pc.record = []
        self.assertEqual(pc.record, [])
        self.assertFalse(hasattr(inner, "record"), "the adapter's own names stay on the adapter")
        self.assertIs(pc.provider, inner)
        self.assertEqual(pc.n, 2)
        self.assertIs(pc.defer_epilogue, True)          # reads delegate
        pc(object(), object())
        self.assertEqual(inner.calls[0][3], 2)


class BHybridArms(unittest.TestCase):
    def setUp(self):
        from gpu1bit.actprec import parts
        self.parts = parts

    def test_enabled_and_references(self):
        P = self.parts
        with P.installed():
            for arm in ("MB1-B-hybrid-x2", "MB1-B-hybrid-x3"):
                fulldev.check_arm_enabled(fulldev.parse_arm(arm))
            with self.assertRaises(RuntimeError):
                fulldev.check_arm_enabled(fulldev.parse_arm("MB1-B-hybrid-x1"))
        self.assertEqual(P.reference_arm("MB1-B-hybrid-x2"), "MB1-int8_mma-x2")
        self.assertEqual(P.reference_arm("MB1-int8_mma-x2"), "MB1-int8_mma-x3")
        self.assertEqual(P.reference_arm("M1-x2"), "M1-x3")
        self.assertTrue(P.is_r2_reference("M1-x3") and P.is_r2_reference("MB1-int8_mma-x3"))
        self.assertFalse(P.is_r2_reference("MB1-B-hybrid-x3"))

    def test_identity(self):
        P = self.parts
        with P.installed():
            d2 = fulldev.identity("MB1-B-hybrid-x2", "full")
            d3 = fulldev.identity("MB1-B-hybrid-x3", "full")
        for d in (d2, d3):
            self.assertIn("finetune/parakeet-binary/gpu1bit/actprec/parts.py", d["sources"])
            self.assertIn("load_binary_packed", d["stack"]["loader"])
            self.assertIn("selection", d)
        self.assertEqual(d2["part_count_patch"]["layer_variant"], "int8_mma:x2")
        self.assertEqual({k: v["sha256"] for k, v in d2["part_count_patch"]["wip_parts"].items()}, P.PINNED)
        self.assertNotIn("part_count_patch", d3)
        self.assertNotEqual(d2["hash"], d3["hash"])

    def test_builder_routing(self):
        P = self.parts
        calls = []
        saved = P._build_b_hybrid
        P._build_b_hybrid = lambda n: calls.append(n) or ("model", {"partcount": None})
        try:
            with P.installed():
                fulldev.build_arm(fulldev.parse_arm("MB1-B-hybrid-x2"))
                fulldev.build_arm(fulldev.parse_arm("MB1-B-hybrid-x3"))
        finally:
            P._build_b_hybrid = saved
        self.assertEqual(calls, [2, 3])


class GateReferences(unittest.TestCase):
    def setUp(self):
        from gpu1bit.actprec import parts
        self.P = parts
        self.run = f"test-x2-gateref-{os.getpid()}"
        self.ctx = parts.installed()
        self.ctx.__enter__()
        self.counts = {n: len(fulldev.subset_records(n, "full")) for n in fulldev.DEV_SETS}

    def tearDown(self):
        self.ctx.__exit__(None, None, None)
        shutil.rmtree(fulldev.ROOT / self.run, ignore_errors=True)

    def summary(self, arm, **over):
        h = fulldev.freeze(self.run, arm, "full")
        s = {"run": self.run, "arm": arm, "subset": "full", "identity": h, "complete": True, "utterances": dict(self.counts),
             "structurally_valid": True, "spec": fulldev.parse_arm(arm),
             "cross_check": {"pass": True} if arm.endswith("x3") else None}
        s.update(over)
        (fulldev.arm_dir(self.run, arm, "full") / "summary.json").write_text(json.dumps(s))
        return s

    def test_chain_and_refusals(self):
        P = self.P
        with self.assertRaisesRegex(SystemExit, "R2"):
            P.require_reference(self.run, "MB1-B-hybrid-x2")                    # g2 missing
        self.summary("MB1-int8_mma-x3")
        self.summary("MB1-int8_mma-x2")
        P.require_reference(self.run, "MB1-int8_mma-x2")                       # g2 may launch: g0 passed R2
        P.require_reference(self.run, "MB1-B-hybrid-x2")                       # g1 may launch: g2 complete, chain holds
        cases = {
            "g2 incomplete": ("MB1-int8_mma-x2", {"complete": False}),
            "g2 other identity": ("MB1-int8_mma-x2", {"identity": "0" * 64}),
            "g2 structurally invalid": ("MB1-int8_mma-x2", {"structurally_valid": False}),
            "g2 of another model": ("MB1-int8_mma-x2", {"spec": fulldev.parse_arm("M1-x2")}),
            "g0 failed R2 (chain)": ("MB1-int8_mma-x3", {"cross_check": {"pass": False}}),
            "g0 of another model": ("MB1-int8_mma-x3", {"spec": fulldev.parse_arm("M1-x3")}),
        }
        for name, (arm, over) in cases.items():
            with self.subTest(case=name):
                self.summary(arm, **over)
                with self.assertRaisesRegex(SystemExit, "R2"):
                    P.require_reference(self.run, "MB1-B-hybrid-x2")
                self.summary(arm)
        P.require_reference(self.run, "MB1-B-hybrid-x2")
        self.summary("M1-x3", cross_check={"pass": False})
        with self.assertRaisesRegex(SystemExit, "R2"):
            P.require_reference(self.run, "M1-x2")                             # g4 refused: g5 failed R2

    def test_reference_identity_changed_after_its_gate(self):
        self.summary("MB1-int8_mma-x3")
        self.P.require_reference(self.run, "MB1-int8_mma-x2")
        idp = fulldev.arm_dir(self.run, "MB1-int8_mma-x3", "full") / "identity.json"
        doc = json.loads(idp.read_text())
        doc["sources"]["finetune/parakeet-binary/gpu1bit/shared/api.py"] = "0" * 64   # frozen under other sources
        doc["hash"] = "f" * 64
        idp.write_text(json.dumps(doc))
        with self.assertRaisesRegex(SystemExit, "R2"):
            self.P.require_reference(self.run, "MB1-int8_mma-x2")


class Loaders(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=str(env.ARTIFACTS_ROOT / "H")))
        self.saved = (seldoc.SELECTED, os.environ.get("GPU1BIT_SELECTION"))
        os.environ.pop("GPU1BIT_SELECTION", None)

    def tearDown(self):
        seldoc.SELECTED, sel = self.saved
        if sel is not None:
            os.environ["GPU1BIT_SELECTION"] = sel
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_signatures_and_refusals(self):
        from gpu1bit import runtime, runtime_m1
        for fn in (runtime.load_binary_packed, runtime_m1.load_ternary_packed):
            sig = inspect.signature(fn)
            self.assertIn("parts", sig.parameters)
            self.assertIn("selection", sig.parameters)
            for bad in (1, 4):
                with self.subTest(fn=fn.__name__, parts=bad), self.assertRaisesRegex(ValueError, "part count"):
                    fn(env.MB1_EXPORT if fn is runtime.load_binary_packed else env.M1_EXPORT, parts=bad)

    def test_default_parts_follow_the_document_in_use(self):
        from gpu1bit import runtime, runtime_m1
        doc = json.loads((env.SHARED / "selected.json").read_text())    # x3 before promotion, the candidate after
        x3 = {k: v for k, v in doc.items() if k not in ("default_parts", "m1_default_parts", "parts")}
        p3 = self.tmp / "x3.json"
        p3.write_text(json.dumps(x3))
        x2 = {**x3, "default_parts": 2, "m1_default_parts": 2,
              "parts": {"3": {"kernel": "int8_mma", "imports": [], "hybrid": x3["hybrid"]},
                        "2": {"kernel": "int8_mma:x2", "imports": [], "hybrid": x3["hybrid"]}}}
        p2 = self.tmp / "x2.json"
        p2.write_text(json.dumps(x2))
        self.assertEqual(runtime.resolve_parts(None, p3), 3)
        self.assertEqual(runtime.resolve_parts(None, p2), 2)
        self.assertEqual(runtime.resolve_parts(3, p2), 3)
        self.assertEqual(runtime_m1.resolve_parts(None, p3), 3)
        self.assertEqual(runtime_m1.resolve_parts(None, p2), 2)
        seldoc.SELECTED = p3
        self.assertEqual(runtime.resolve_parts(None, None), 3, "unqualified before promotion: x3")
        shutil.copy(p2, p3)
        self.assertEqual(runtime.resolve_parts(None, None), 2, "unqualified after promotion: x2")
        self.assertEqual(runtime_m1.resolve_parts(None, None), 2)

    def test_cli_and_readme(self):
        from gpu1bit import runtime
        src = inspect.getsource(runtime.main)
        self.assertIn("--parts", src)
        self.assertIn("--selection", src)
        readme = (env.TERNARY_DIR / "README.md").read_text()
        self.assertIn("gpu1bit.runtime_m1.load_ternary_packed", readme)
