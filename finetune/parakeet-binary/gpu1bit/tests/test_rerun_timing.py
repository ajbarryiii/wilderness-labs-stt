"""Contract tests of gpu1bit/rerun_timing.py's roster-file mode (activation-precision timing; CPU):
  - the steps of a roster file are the selection session's protocols with the roster's variants and
    patches:
    - P-paired at encoder and full scope, 6 chunks of 4 dev_24 utterances each;
    - P-clips/P-memory, one fresh process per variant, forward then reverse order;
    - phases only if the roster names a pair;
    - every command goes through gpu1bit.shared.dispatch with the roster's patches, and every output
      lands under bench/<roster run>;
  - the guard before every process re-verifies each variant's frozen identity in the roster's
    identity run (fulldev.check_identity with the activation-precision patch installed), and refuses
    a mismatch.
  - resume (review c2):
    - the guard runs before any step is skipped;
    - a step is reused only if its accepted sidecar is bound to the same roster, protocol (the exact
      command) and identity map, and its output is byte-identical to the accepted one (SHA-256);
    - the output itself must name the requested protocol, run, variants and identities;
    - otherwise the step is pending again, and the stale output is moved out of the bench directory.
  - recognition-run binding (review c3):
    - every worker gets the roster's identity_run explicitly (GPU1BIT_ACTPREC_RUN), overriding any
      inherited value;
    - the guard also gates that run's references (parts.require_reference for every variant), so
      references are verified before any reuse;
    - every paired / clip artifact must name that identity_run and each variant's reference arm and
      identity; a foreign run or wrong reference metadata is refused, and a step recorded under
      another identity_run is not reused.

    gpu1bit/shared/python -m gpu1bit.tests.run --cpu gpu1bit.tests.test_rerun_timing
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import unittest
from pathlib import Path

from gpu1bit.shared import env, fulldev

sys.path.insert(0, str(env.GPU1BIT.parent))
import gpu1bit.rerun_timing as T  # noqa: E402

ARMS = ["M1-x3", "M1-x2", "M1-x1", "MB1-int8_mma-x3", "MB1-int8_mma-x2", "MB1-int8_mma-x1"]


class RosterFile(unittest.TestCase):
    def setUp(self):
        self.run = f"test-rerun-timing-{os.getpid()}"
        self.roster = {"run": self.run, "variants": ARMS, "patches": ["gpu1bit.actprec.bench"], "phase_pair": None,
                       "identity_run": self.run}

    def tearDown(self):
        from gpu1bit.actprec import parts
        parts.uninstall()
        shutil.rmtree(fulldev.ROOT / self.run, ignore_errors=True)

    def test_steps(self):
        steps = T.timing_steps(self.run, self.roster)
        names = [s["name"] for s in steps]
        self.assertEqual(len([n for n in names if n.startswith("paired-encoder")]), 6)
        self.assertEqual(len([n for n in names if n.startswith("paired-full")]), 6)
        self.assertEqual([n for n in names if n.startswith("clips1-")], [f"clips1-{a}" for a in ARMS])
        self.assertEqual([n for n in names if n.startswith("clips2-")], [f"clips2-{a}" for a in ARMS[::-1]])
        self.assertFalse(any(n.startswith("phases") for n in names))
        for s in steps:
            cmd = s["cmd"]
            self.assertEqual(cmd[1:5], ["-m", "gpu1bit.shared.dispatch", "--patches", "gpu1bit.actprec.bench"])
            self.assertTrue(str(s["output"]).startswith(str(env.ARTIFACTS / "bench" / self.run)))
            if "paired" in s["name"]:
                i = cmd.index("--variants")
                self.assertEqual(cmd[i + 1:i + 1 + len(ARMS)], ARMS)

    def write_reference(self, arm, h, **overrides):
        summary = {"run": self.run, "arm": arm, "subset": "full", "identity": h, "complete": True,
                   "utterances": {n: len(fulldev.subset_records(n, "full")) for n in fulldev.DEV_SETS},
                   "structurally_valid": True, "cross_check": {"pass": True},
                   "spec": fulldev.parse_arm(arm)}                        # as fulldev.finish writes it
        summary.update(overrides)
        (fulldev.arm_dir(self.run, arm, "full") / "summary.json").write_text(json.dumps(summary))

    def test_child_env_carries_the_identity_run(self):
        os.environ["GPU1BIT_ACTPREC_RUN"] = "some-other-run"
        try:
            e = T.child_env(self.roster)
            self.assertEqual(e["GPU1BIT_ACTPREC_RUN"], self.run, "the roster's run, not the inherited one")
            self.assertNotIn("CUDA_VISIBLE_DEVICES", e)
            self.assertNotIn("CUBLAS_WORKSPACE_CONFIG", e)
            selection_roster = {k: v for k, v in self.roster.items() if k != "identity_run"}
            self.assertNotIn("GPU1BIT_ACTPREC_RUN", T.child_env(selection_roster))
        finally:
            del os.environ["GPU1BIT_ACTPREC_RUN"]

    def test_identity_guard(self):
        from gpu1bit.actprec import parts
        with parts.installed():
            hashes = {a: fulldev.freeze(self.run, a, "full") for a in ARMS}
        with self.assertRaisesRegex(SystemExit, "R2"):               # identities verify, but no references yet
            T.identity_guard(self.roster)
        for ref in ("M1-x3", "MB1-int8_mma-x3"):
            self.write_reference(ref, hashes[ref])
        guard = T.identity_guard(self.roster)
        self.assertTrue(guard["ok"])
        self.assertEqual(guard["identities"], hashes)
        self.assertEqual(guard["identity_run"], self.run)
        self.assertEqual(guard["references"]["MB1-int8_mma-x1"], {"arm": "MB1-int8_mma-x3", "identity": hashes["MB1-int8_mma-x3"]})
        self.assertEqual(guard["references"]["M1-x3"], {"arm": "M1-x3", "identity": hashes["M1-x3"]})
        self.assertEqual((guard["mismatched"], guard["added"]), ([], []))
        self.write_reference("M1-x3", hashes["M1-x3"], cross_check={"pass": False})
        with self.assertRaisesRegex(SystemExit, "R2"):
            T.identity_guard(self.roster)
        self.write_reference("M1-x3", hashes["M1-x3"])
        path = fulldev.arm_dir(self.run, "MB1-int8_mma-x1", "full") / "identity.json"
        doc = json.loads(path.read_text())
        doc["parts"], doc["hash"] = 3, "e" * 64
        path.write_text(json.dumps(doc))
        with self.assertRaisesRegex(SystemExit, "identity mismatch"):
            T.identity_guard(self.roster)


class Resume(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.run = f"test-rerun-resume-{os.getpid()}"
        self.tmp = Path(tempfile.mkdtemp(prefix="rerun-resume-", dir=str(env.ARTIFACTS_ROOT / "H")))
        self.roster = {"run": self.run, "variants": ARMS, "patches": ["gpu1bit.actprec.bench"], "phase_pair": None,
                       "identity_run": self.run}
        self.ids = {a: f"{k:064x}" for k, a in enumerate(ARMS, start=1)}

    def guard(self, ids=None, identity_run=None):
        from gpu1bit.actprec import parts
        ids = dict(ids or self.ids)
        refs = {a: {"arm": parts.reference_arm(a), "identity": ids[parts.reference_arm(a)]} for a in ARMS}
        return {"files": 6, "mismatched": [], "added": [], "ok": True, "identities": ids, "references": refs,
                "identity_run": identity_run or self.run}

    def bind(self, ids=None, identity_run=None):
        return T.make_binding(self.roster, self.guard(ids, identity_run))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        for d in (self.run, f"{self.run}-clips1", f"{self.run}-clips2"):
            shutil.rmtree(env.ARTIFACTS / "bench" / d, ignore_errors=True)

    def info(self, v, ids, identity_run=None):
        g = self.guard(ids, identity_run)
        return {"arm": v, "identity": ids[v], "identity_run": g["identity_run"], "reference": g["references"][v]}

    def doc_for(self, step, ids=None, identity_run=None, **overrides):
        ids = ids or self.ids
        if step["kind"] == "clips":
            doc = {"protocol": "P-clips+P-memory", "variant": step["variant"], "args": {"run": f"{self.run}-{step['label']}"},
                   "variant_info": self.info(step["variant"], ids, identity_run), "workloads": [{}, {}, {}],
                   "memory_samples_10ms": []}
        else:
            a, b = (int(v) for v in step["range"].split("-"))
            doc = {"protocol": "P-paired", "scope": step["scope"], "variants": list(self.roster["variants"]),
                   "args": {"run": self.run, "utterances": step["range"], "scope": step["scope"]},
                   "variants_info": {v: self.info(v, ids, identity_run) for v in self.roster["variants"]},
                   "rows": [{"index": k} for k in range(a, b + 1)]}
        doc.update(overrides)
        return doc

    def accept(self, step, doc, roster=None, ids=None, identity_run=None):
        bind = T.make_binding(roster or self.roster, self.guard(ids, identity_run))
        step["output"].parent.mkdir(parents=True, exist_ok=True)
        step["output"].write_text(json.dumps(doc))
        rec = {"step": step["name"], "accepted": True,
               "attempts": [{"accepted": True, "binding": T.step_binding(step, bind), "output_sha256": env.sha256_file(step["output"])}]}
        (self.tmp / "host").mkdir(exist_ok=True)
        (self.tmp / "host" / f"{step['name']}.json").write_text(json.dumps(rec))

    def pending(self, roster=None, ids=None, identity_run=None):
        return [s["name"] for s in T.pending_steps((roster or self.roster)["run"], roster or self.roster, self.tmp,
                                                   lambda: self.guard(ids, identity_run))[0]]

    def steps(self):
        return {s["name"]: s for s in T.timing_steps(self.run, self.roster)}

    def test_guard_runs_before_any_skip(self):
        steps = self.steps()
        for name in ("paired-encoder-0-3", "clips1-M1-x3"):
            self.accept(steps[name], self.doc_for(steps[name]))

        def refuse():
            raise SystemExit("identity mismatch for M1-x2@full")
        with self.assertRaisesRegex(SystemExit, "identity mismatch"):
            T.pending_steps(self.run, self.roster, self.tmp, refuse)

    def test_bound_steps_are_reused_and_others_are_not(self):
        steps = self.steps()
        for name in ("paired-encoder-0-3", "clips1-M1-x3"):
            self.accept(steps[name], self.doc_for(steps[name]))
        pending = self.pending()
        self.assertNotIn("paired-encoder-0-3", pending)
        self.assertNotIn("clips1-M1-x3", pending)
        self.assertEqual(len(pending), len(steps) - 2)

    def test_changed_roster_invalidates_reuse(self):
        steps = self.steps()
        self.accept(steps["clips1-M1-x3"], self.doc_for(steps["clips1-M1-x3"]))
        changed = dict(self.roster, patches=["gpu1bit.actprec.bench", "gpu1bit.int8_mma.b01_run"])
        self.assertIn("clips1-M1-x3", self.pending(roster=changed))
        self.assertFalse(steps["clips1-M1-x3"]["output"].exists(), "the stale output is moved out")
        self.assertTrue(list((self.tmp / "dropped").glob("clips1-M1-x3*")))

    def test_replaced_output_invalidates_reuse(self):
        steps = self.steps()
        step = steps["paired-encoder-0-3"]
        self.accept(step, self.doc_for(step))
        doc = self.doc_for(step)
        doc["rows"][0]["summary"] = "replaced"
        step["output"].write_text(json.dumps(doc))
        self.assertIn("paired-encoder-0-3", self.pending())

    def test_changed_identities_invalidate_reuse(self):
        steps = self.steps()
        self.accept(steps["clips1-M1-x3"], self.doc_for(steps["clips1-M1-x3"]))
        other = dict(self.ids, **{"M1-x3": "9" * 64})
        self.assertIn("clips1-M1-x3", self.pending(ids=other))

    def test_artifact_must_match_protocol_run_variants_identities(self):
        steps = self.steps()
        paired, clips = steps["paired-full-4-7"], steps["clips2-MB1-int8_mma-x1"]
        self.assertTrue(T.artifact_bound(paired, self.doc_for(paired), self.roster, self.bind()))
        self.assertTrue(T.artifact_bound(clips, self.doc_for(clips), self.roster, self.bind()))
        for bad in ({"protocol": "phases"}, {"scope": "encoder"}, {"variants": ARMS[::-1]}, {"args": {"run": "x", "utterances": "4-7"}},
                    {"variants_info": {v: dict(self.info(v, self.ids), identity="0" * 64) for v in ARMS}}):
            with self.subTest(bad=bad):
                self.assertFalse(T.artifact_bound(paired, self.doc_for(paired, **bad), self.roster, self.bind()))
        for bad in ({"variant": "M1-x3"}, {"variant_info": {}}, {"variant_info": dict(self.info("MB1-int8_mma-x1", self.ids), identity="0" * 64)},
                    {"args": {"run": f"{self.run}-clips1"}}, {"protocol": "P-paired"}):
            with self.subTest(bad=bad):
                self.assertFalse(T.artifact_bound(clips, self.doc_for(clips, **bad), self.roster, self.bind()))
        # a sidecar that claims an artifact of other identities is not reused either
        self.accept(clips, self.doc_for(clips, ids=dict(self.ids, **{"MB1-int8_mma-x1": "7" * 64})))
        self.assertIn("clips2-MB1-int8_mma-x1", self.pending())

    def test_foreign_identity_run_and_wrong_reference_are_refused(self):
        """Review c3: the recognition run and each variant's reference are part of the artifact binding."""
        steps = self.steps()
        paired, clips = steps["paired-encoder-8-11"], steps["clips1-MB1-int8_mma-x2"]
        for step in (paired, clips):
            with self.subTest(step=step["name"]):
                self.assertTrue(T.artifact_bound(step, self.doc_for(step), self.roster, self.bind()))
                self.assertFalse(T.artifact_bound(step, self.doc_for(step, identity_run="ap-older-run"), self.roster, self.bind()),
                                 "an artifact recorded against another recognition run")
                self.assertFalse(T.artifact_bound(step, self.doc_for(step), self.roster, self.bind(identity_run="ap-older-run")))
        wrong_arm = self.doc_for(clips)
        wrong_arm["variant_info"]["reference"] = {"arm": "MB1-int8_mma-x2", "identity": self.ids["MB1-int8_mma-x2"]}
        self.assertFalse(T.artifact_bound(clips, wrong_arm, self.roster, self.bind()))
        wrong_hash = self.doc_for(clips)
        wrong_hash["variant_info"]["reference"] = {"arm": "MB1-int8_mma-x3", "identity": "5" * 64}
        self.assertFalse(T.artifact_bound(clips, wrong_hash, self.roster, self.bind()))
        no_ref = self.doc_for(paired)
        del no_ref["variants_info"]["M1-x1"]["reference"]
        self.assertFalse(T.artifact_bound(paired, no_ref, self.roster, self.bind()))
        # a step accepted under another recognition run is measured again
        self.accept(clips, self.doc_for(clips, identity_run="ap-older-run"), identity_run="ap-older-run")
        self.assertIn(clips["name"], self.pending())
        self.accept(clips, self.doc_for(clips))
        self.assertNotIn(clips["name"], self.pending())

