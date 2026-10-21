"""Contract tests of selection documents for the INT8x2 default (plans/x2-default.md Y1, Z2; CPU):
  - documents: shared/seldoc.py loads a selection document and hashes its CONTENT. It resolves the
    entry for a part count (x2 documents carry "parts"; the x3 document is the part-3 entry). It
    reads default_parts and m1_default_parts (absent means 3), and refuses part counts other than 2 and
    3. The document in use is GPU1BIT_SELECTION if set, else shared/selected.json.
  - pointer (Z2): selected.json is accepted only byte-identical to the pinned x3 document or to the
    candidate selected-x2.json; any third state is refused. `promote` copies the candidate's bytes.
  - fingerprints and identities hash the content of the document in use, not the path of
    selected.json:
    - shared/selected.json is not a fingerprint source;
    - the same bytes under another path give the same fingerprint and identity;
    - promotion leaves the b1 (MB1-B-hybrid-x2) and b4 (M1-x2) identities and the gate fingerprint
      unchanged;
    - a different document changes them.
  - freeze (Z2): a session freeze lists shared/selected-x2.json, not shared/selected.json. Its
    verification passes before and after promotion and refuses a third pointer state.

    gpu1bit/shared/python -m gpu1bit.tests.run --cpu gpu1bit.tests.test_x2_selection
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from gpu1bit.shared import env, fingerprint as F, fulldev, seldoc, selection

X3 = {"about": "x3", "kernel": "int8_mma", "runtime": "hybrid", "imports": [],
      "hybrid": {"provider": "gpu1bit.int8_mma.hybrid_exact:HybridProjectionExact",
                 "pipeline": "gpu1bit.int8_mma.hybrid_exact:exact_hybrid_pipeline", "imports": ["gpu1bit.int8_mma.row_exact"]},
      "selection": {"session": "sel-20261020b", "variant": "B-hybrid"}}
X2 = {**X3, "about": "x2 candidate", "default_parts": 2, "m1_default_parts": 2,
      "parts": {"3": {"kernel": "int8_mma", "imports": [], "hybrid": X3["hybrid"]},
                "2": {"kernel": "int8_mma:x2", "imports": [],
                      "hybrid": {"provider": "gpu1bit.int8_mma.hybrid_x2:HybridProjectionX2",
                                 "pipeline": "gpu1bit.int8_mma.hybrid_x2:exact_hybrid_pipeline_x2", "imports": []}}},
      "arms": {"default": "MB1-B-hybrid-x2", "override_x3": "MB1-B-hybrid-x3", "m1_default": "M1-x2"}}


class _Temp(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=str(env.ARTIFACTS_ROOT / "H")))
        self.ptr, self.cand = self.tmp / "selected.json", self.tmp / "selected-x2.json"
        self.ptr.write_text(json.dumps(X3, indent=1) + "\n")
        self.cand.write_text(json.dumps(X2, indent=1) + "\n")
        self.saved = (seldoc.SELECTED, seldoc.CANDIDATE, os.environ.get("GPU1BIT_SELECTION"))
        seldoc.SELECTED, seldoc.CANDIDATE = self.ptr, self.cand
        os.environ.pop("GPU1BIT_SELECTION", None)

    def tearDown(self):
        seldoc.SELECTED, seldoc.CANDIDATE, sel = self.saved
        if sel is None:
            os.environ.pop("GPU1BIT_SELECTION", None)
        else:
            os.environ["GPU1BIT_SELECTION"] = sel
        shutil.rmtree(self.tmp, ignore_errors=True)


class Documents(_Temp):
    def test_load_and_entries(self):
        d = seldoc.load(self.cand)
        self.assertEqual(d["sha256"], env.sha256_file(self.cand))
        self.assertEqual(seldoc.default_parts(d["doc"]), 2)
        self.assertEqual(seldoc.m1_default_parts(d["doc"]), 2)
        self.assertEqual(seldoc.entry(d["doc"], 2)["kernel"], "int8_mma:x2")
        self.assertEqual(seldoc.entry(d["doc"], 3)["hybrid"], X3["hybrid"])
        x3 = seldoc.load(self.ptr)["doc"]
        self.assertEqual((seldoc.default_parts(x3), seldoc.m1_default_parts(x3)), (3, 3))
        self.assertEqual(seldoc.entry(x3, 3)["kernel"], "int8_mma")
        with self.assertRaisesRegex(ValueError, "part"):
            seldoc.entry(x3, 2)                                  # the x3 document has no x2 entry
        for bad in (1, 4, "2"):
            with self.subTest(parts=bad), self.assertRaises(ValueError):
                seldoc.check_parts(bad)

    def test_document_in_use(self):
        self.assertEqual(seldoc.current_path(), self.ptr)
        os.environ["GPU1BIT_SELECTION"] = str(self.cand)
        self.assertEqual(seldoc.current_path(), self.cand)


class Pointer(_Temp):
    def test_states(self):
        pin = env.sha256_file(self.ptr)
        self.assertEqual(seldoc.pointer_state(pin), "x3")
        seldoc.promote()
        self.assertEqual(self.ptr.read_bytes(), self.cand.read_bytes())
        self.assertEqual(seldoc.pointer_state(pin), "promoted")
        self.ptr.write_text(json.dumps({**X2, "default_parts": 3}))
        with self.assertRaisesRegex(SystemExit, "third state"):
            seldoc.pointer_state(pin)


class ContentHashing(_Temp):
    def test_selected_json_is_not_a_fingerprint_source(self):
        real = env.SHARED / "selected.json"
        self.assertNotIn(real, F.source_files())
        self.assertNotIn("finetune/parakeet-binary/gpu1bit/shared/selected.json", fulldev.identity("M1-x3", "full")["sources"])

    def test_promotion_invariance(self):
        from gpu1bit.actprec import parts
        from gpu1bit.shared import gate
        with parts.installed():
            os.environ["GPU1BIT_SELECTION"] = str(self.cand)               # qualified: the candidate, explicitly
            qual = {a: fulldev.identity(a, "full")["hash"] for a in ("MB1-B-hybrid-x2", "M1-x2")}
            gate_q = gate.current_fingerprint("int8_mma", [])["hash"]
            del os.environ["GPU1BIT_SELECTION"]
            pre = {a: fulldev.identity(a, "full")["hash"] for a in ("MB1-B-hybrid-x2", "M1-x2")}
            self.assertNotEqual(pre, qual, "an unqualified caller before promotion uses the x3 document")
            seldoc.promote()
            post = {a: fulldev.identity(a, "full")["hash"] for a in ("MB1-B-hybrid-x2", "M1-x2")}
            self.assertEqual(post, qual, "after promotion an unqualified caller has the qualified identities")
            self.assertEqual(gate.current_fingerprint("int8_mma", [])["hash"], gate_q)
            os.environ["GPU1BIT_SELECTION"] = str(self.cand)
            self.assertEqual({a: fulldev.identity(a, "full")["hash"] for a in qual}, qual, "qualified identities unchanged")
            other = self.tmp / "other.json"
            other.write_text(json.dumps({**X2, "about": "different"}))
            os.environ["GPU1BIT_SELECTION"] = str(other)
            self.assertNotEqual(fulldev.identity("M1-x2", "full")["hash"], qual["M1-x2"])
            same = self.tmp / "copy-elsewhere.json"
            shutil.copy(self.cand, same)
            os.environ["GPU1BIT_SELECTION"] = str(same)
            self.assertEqual(fulldev.identity("M1-x2", "full")["hash"], qual["M1-x2"], "content, not path")


class FreezePointer(_Temp):
    def setUp(self):
        super().setUp()
        self.run = f"test-x2-freeze-{os.getpid()}"

    def tearDown(self):
        shutil.rmtree(selection.sel_dir(self.run), ignore_errors=True)
        for p in (selection.freeze_path(self.run), selection.pointer_pin_path(self.run)):
            if p.exists():
                p.unlink()
        super().tearDown()

    def test_freeze_and_verify_across_promotion(self):
        args = type("A", (), {"run": self.run, "variants": ["R-expanded", "int8_mma"], "patches": "", "phase_pair": [],
                              "stack_baseline": []})()
        selection.make_roster(args)
        listed = selection.freeze_path(self.run).read_text()
        self.assertNotIn("gpu1bit/shared/selected.json", listed)
        if (env.SHARED / "selected-x2.json").exists():
            self.assertIn("gpu1bit/shared/selected-x2.json", listed)
        self.assertTrue(selection.verify_freeze(self.run)["ok"])
        self.assertEqual(selection.verify_freeze(self.run)["pointer"], "x3")
        shutil.copy(self.cand, self.ptr)
        self.assertEqual(selection.verify_freeze(self.run)["pointer"], "promoted")
        self.ptr.write_text("{}")
        with self.assertRaisesRegex(SystemExit, "third state"):
            selection.verify_freeze(self.run)


class LoaderPartCounts(unittest.TestCase):
    """X3: the deployment loaders take parts 2 or 3 only, refused before any model work. The activation-
    precision x1 research arm reaches the provider stack only through the private `_research_parts`, which
    needs an explicit kernel on the provider runtime (never the selection's deployment stack)."""

    def test_deployment_part_counts(self):
        from gpu1bit import runtime, runtime_m1
        for bad in (1, 4):
            with self.subTest(parts=bad):
                with self.assertRaisesRegex(ValueError, "part count"):
                    runtime.load_binary_packed(env.MB1_EXPORT, parts=bad)
                with self.assertRaisesRegex(ValueError, "part count"):
                    runtime_m1.load_ternary_packed(parts=bad)

    def test_research_part_count_is_provider_only(self):
        from gpu1bit import runtime
        for kw in ({}, {"runtime": "hybrid", "kernel": "int8_mma:x1"}, {"runtime": "provider"},
                   {"runtime": "provider", "kernel": "int8_mma:x1", "parts": 2}):
            with self.subTest(kw=kw), self.assertRaisesRegex(ValueError, "_research_parts"):
                runtime.load_binary_packed(env.MB1_EXPORT, _research_parts=1, **kw)
        with self.assertRaisesRegex(ValueError, "_research_parts"):
            runtime.load_binary_packed(env.MB1_EXPORT, runtime="provider", kernel="int8_mma:x1", _research_parts=4)


class AfterPromotion(_Temp):
    """Codex x2 c1 findings 7 and 8: after promotion (the pointer is the candidate's bytes) only a canonical
    deployment call (no kernel, no hybrid spec, no parts) takes default_parts 2. A caller naming a kernel or a
    hybrid stack gets that kernel's own part count (int8_mma and int8_mma:b01 are x3, int8_mma:x2 is x2), and
    an x3-only kernel at 2 parts is refused before any model work. fulldev's base arms pass their parsed part
    count. The historical x3 selection record (session, variant) resolves from either document."""

    def setUp(self):
        super().setUp()
        seldoc.promote()
        self.doc = seldoc.load()["doc"]

    def test_load_resolution(self):
        from gpu1bit import runtime
        r = runtime.resolve_load(self.doc)
        self.assertEqual((r["parts"], r["kernel"], r["runtime"]), (2, "int8_mma:x2", "hybrid"))
        self.assertEqual(r["hybrid"]["provider"], X2["parts"]["2"]["hybrid"]["provider"])
        self.assertTrue(r["canonical"])
        r = runtime.resolve_load(self.doc, parts=3)
        self.assertEqual((r["parts"], r["kernel"], r["hybrid"]["provider"]), (3, "int8_mma", X3["hybrid"]["provider"]))
        for kw, want in (({"runtime": "provider", "kernel": "int8_mma"}, (3, "int8_mma")),
                         ({"runtime": "hybrid", "kernel": "int8_mma"}, (3, "int8_mma")),
                         ({"runtime": "hybrid", "kernel": "int8_mma:b01"}, (3, "int8_mma:b01")),
                         ({"runtime": "provider", "kernel": "int8_mma:x2"}, (2, "int8_mma:x2")),
                         ({"runtime": "hybrid", "hybrid": X3["hybrid"]}, (3, "int8_mma"))):
            with self.subTest(kw=kw):
                r = runtime.resolve_load(self.doc, **kw)
                self.assertEqual((r["parts"], r["kernel"]), want)
                self.assertFalse(r["canonical"])
                if kw.get("runtime") == "hybrid":
                    self.assertEqual(r["hybrid"]["provider"], X3["hybrid"]["provider"])
        with self.assertRaisesRegex(ValueError, "part"):
            runtime.resolve_load(self.doc, runtime="provider", kernel="int8_mma", parts=2)
        with self.assertRaisesRegex(ValueError, "part"):
            runtime.resolve_load(self.doc, runtime="provider", kernel="int8_mma:x2", parts=3)

    def test_fulldev_base_arm_passes_its_part_count(self):
        from gpu1bit import runtime
        from gpu1bit.actprec import parts
        seen = []

        class _Stop(Exception):
            pass

        def stub(export, **kw):
            seen.append(kw)
            raise _Stop
        saved = runtime.load_binary_packed
        runtime.load_binary_packed = stub
        try:
            with self.assertRaises(_Stop):
                parts.BASE["build_arm"](fulldev.parse_arm("MB1-int8_mma-x3"))
        finally:
            runtime.load_binary_packed = saved
        self.assertEqual(seen, [{"runtime": "provider", "kernel": "int8_mma", "parts": 3}])

    def test_x3_selection_record(self):
        for path in (env.SHARED / "selected.json", env.SHARED / "selected-x2.json"):
            with self.subTest(document=path.name):
                rec = seldoc.x3_selection_record(json.loads(path.read_text()))
                self.assertEqual((rec["session"], rec["variant"]), ("sel-20261020b", "B-hybrid"))
        self.assertEqual(seldoc.x3_selection_record(self.doc)["session"], "sel-20261020b")
        with self.assertRaisesRegex(ValueError, "x3 selection"):
            seldoc.x3_selection_record({"selection": {}})
