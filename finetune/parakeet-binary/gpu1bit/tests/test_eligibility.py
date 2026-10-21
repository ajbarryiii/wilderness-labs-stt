"""Contract tests for selection eligibility and decision provenance (Codex GPU code review c3; CPU only):

  deferred A3 (finding 1)
    - shared/deferred_a3.py: a run of partial scope (--no-real, --no-synthetic, or a subset of the
      fixture utterances) never reports passes=True. It is marked partial, and its fingerprint
      includes the scope.
    - selection.deferred_a3_check, the one validator both reporting paths use, accepts a report
      only if it:
      - is of the candidate's provider and module, with the provider identity (registry kernel,
        factory) of the current DEFERRED_PROVIDERS entry;
      - is not informational;
      - has the full scope, with the full expected coverage: 264 layers x every fixture utterance
        real, with exactly the pipeline's deferred layers deferred, and every synthetic case
        including discriminating ones;
      - has zero failures and passes=True;
      - is bound to the current fingerprint.
      Each refusal is tested. Failing results are a quality outcome (ineligible); everything else is
      a provenance problem.
  shared eligibility (finding 1)
    - selection.evaluate_eligibility is used by both `report` and `regenerate`. A hybrid-stack
      candidate is eligible only with a valid, passing deferred-A3 record; provenance problems are
      listed separately from quality failures.
  decision writes (finding 3)
    - a decision with provenance problems is never written. An existing accepted decision file is
      left byte-identical, and the refusal names the problems.
    - the check-run name is derived from the current source state, so a source change gives a new
      name, and records of the old name are never reused.
  general arms (plans/x2-default.md relies on it)
    - the validator is parametrized by candidate, provider module and part count (roster keys
      deferred_module, deferred_components); the part count is in the recorded scope.
  A15 (finding 2)
    - the winner rule's text is plan amendment A15 in plans/gpu-kernel.md, and the plan is part of
      every fingerprint, so a measurement binds to the rule text.

    gpu1bit/shared/python -m gpu1bit.tests.run --cpu gpu1bit.tests.test_eligibility
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from gpu1bit.shared import deferred_a3, env, fingerprint as F, selection

MODULE = "gpu1bit.int8_mma.deferred"


def _valid_report(provider: str = "B-hybrid") -> dict:
    cov = deferred_a3.expected_coverage()
    scope = deferred_a3.full_scope()
    return {"provider": provider, "module": MODULE, "informational": False,
            "provider_identity": deferred_a3.provider_identity(provider, MODULE),
            "scope": scope, "partial": False,
            "real": {"evaluations": cov["real_evaluations"], "failures": 0, "deferred_evaluations": cov["deferred_evaluations"],
                     "deferred_failures": 0, "deferral_mismatches": 0},
            "synthetic": {"cases": cov["synthetic_cases"], "failures": 0, "not_deferred": 0,
                          "discriminating_cases": 37, "discriminating_failures": 0},
            "passes": True, "fingerprint": deferred_a3.record_fingerprint(provider, MODULE, scope)}


class DeferredA3Scope(unittest.TestCase):
    def test_partial_runs_never_pass(self):
        full = deferred_a3.full_scope()
        ok_part = {"failures": 0}
        for real, synthetic, utt in ((False, False, full["utterances"]), (True, False, full["utterances"]),
                                     (False, True, full["utterances"]), (True, True, full["utterances"][:3])):
            with self.subTest(real=real, synthetic=synthetic, utterances=len(utt)):
                scope = {"utterances": list(utt), "real": real, "synthetic": synthetic}
                out = {"real": ok_part if real else None, "synthetic": dict(ok_part) if synthetic else None}
                deferred_a3.finalize(out, scope)
                self.assertIs(out["passes"], False)
                self.assertTrue(out["partial"])
                self.assertTrue(out["checks_passed"])
        out = {"real": {"failures": 0}, "synthetic": {"failures": 0}}
        deferred_a3.finalize(out, full)
        self.assertIs(out["passes"], True)
        self.assertFalse(out["partial"])
        out = {"real": {"failures": 1}, "synthetic": {"failures": 0}}
        deferred_a3.finalize(out, full)
        self.assertIs(out["passes"], False)

    def test_fingerprint_includes_scope(self):
        full = deferred_a3.full_scope()
        part = dict(full, synthetic=False)
        self.assertNotEqual(deferred_a3.record_fingerprint("B-hybrid", MODULE, full)["hash"],
                            deferred_a3.record_fingerprint("B-hybrid", MODULE, part)["hash"])

    def test_expected_coverage(self):
        cov = deferred_a3.expected_coverage()
        self.assertEqual(cov["utterances"], 15)
        self.assertEqual(cov["real_evaluations"], 264 * 15)
        self.assertEqual(cov["deferred_evaluations"], 4 * 24 * 15)
        self.assertEqual(cov["synthetic_cases"], len(deferred_a3.synthetic_cases()))


class DeferredA3Validator(unittest.TestCase):
    def check(self, doc, candidate="B-hybrid"):
        return selection.deferred_a3_check(doc, candidate, MODULE)

    def test_valid_report_is_accepted(self):
        res = self.check(_valid_report())
        self.assertTrue(res["ok"], res)
        self.assertTrue(res["passes"])
        self.assertEqual(res["provenance"], [])

    def test_each_refusal(self):
        cov = deferred_a3.expected_coverage()
        full = deferred_a3.full_scope()
        cases = {
            "missing": None,
            "other provider": lambda d: d.update(provider="B-hybrid-b01"),
            "other module": lambda d: d.update(module="gpu1bit.int8_mma.other"),
            "informational": lambda d: d.update(informational=True),
            "provider identity": lambda d: d["provider_identity"].update(kernel="int8_mma:b01"),
            "no real": lambda d: d.update(real=None, scope=dict(full, real=False)),
            "no synthetic": lambda d: d.update(synthetic=None, scope=dict(full, synthetic=False)),
            "utterance subset": lambda d: d.update(scope=dict(full, utterances=full["utterances"][:5])),
            "partial flag": lambda d: d.update(partial=True),
            "real count": lambda d: d["real"].update(evaluations=cov["real_evaluations"] - 1),
            "deferred count": lambda d: d["real"].update(deferred_evaluations=cov["deferred_evaluations"] - 15),
            "deferral mismatch": lambda d: d["real"].update(deferral_mismatches=1),
            "synthetic count": lambda d: d["synthetic"].update(cases=cov["synthetic_cases"] - 1),
            "no discriminating case": lambda d: d["synthetic"].update(discriminating_cases=0),
            "stale fingerprint": lambda d: d.update(fingerprint=dict(d["fingerprint"], hash="0" * 64)),
            "fingerprint of a partial scope": lambda d: d.update(
                fingerprint=deferred_a3.record_fingerprint("B-hybrid", MODULE, dict(full, synthetic=False))),
        }
        for name, edit in cases.items():
            with self.subTest(refusal=name):
                doc = None
                if edit is not None:
                    doc = copy.deepcopy(_valid_report())
                    edit(doc)
                res = self.check(doc)
                self.assertFalse(res["ok"])
                self.assertTrue(res["provenance"], f"{name}: must be a provenance problem")

    def test_failures_are_quality_not_provenance(self):
        for name, edit in {"real failure": lambda d: d["real"].update(failures=1, deferred_failures=1),
                           "synthetic failure": lambda d: d["synthetic"].update(failures=2),
                           "discriminating failure": lambda d: d["synthetic"].update(failures=1, discriminating_failures=1)}.items():
            with self.subTest(case=name):
                doc = copy.deepcopy(_valid_report())
                edit(doc)
                doc["passes"] = False
                res = self.check(doc)
                self.assertFalse(res["ok"])
                self.assertFalse(res["passes"])
                self.assertEqual(res["provenance"], [])
        doc = copy.deepcopy(_valid_report())
        doc["passes"] = False                                  # inconsistent with zero failures
        self.assertFalse(self.check(doc)["ok"])


class SharedEligibility(unittest.TestCase):
    """evaluate_eligibility on a synthetic check run with real fingerprints."""

    def setUp(self):
        self.chk = f"test-elig-{os.getpid()}"
        self.roster = {"run": "x", "variants": ["R-expanded", "R-hybrid", "c-prov", "B-hybrid"], "reference": "R-expanded",
                       "candidates": ["c-prov", "B-hybrid"], "patches": [],
                       "stack_baseline": {"c-prov": "R-expanded", "B-hybrid": "R-hybrid"}}
        self.saved = selection._gate_verdict
        selection._gate_verdict = lambda chk, c, patches: {"passes": True, "problems": [], "complete": True}
        audit_dir = selection.sel_dir(self.chk) / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        for v in self.roster["variants"]:
            (audit_dir / f"{v}.json").write_text(json.dumps({"within_budget": v not in ("R-expanded", "R-hybrid"),
                                                            "fingerprint": selection.audit_fingerprint(v, [])}))
        self.enc = {v: {"ok": True} for v in self.roster["variants"]}

    def tearDown(self):
        selection._gate_verdict = self.saved
        shutil.rmtree(selection.sel_dir(self.chk), ignore_errors=True)
        shutil.rmtree(env.ARTIFACTS / "deferred" / self.chk, ignore_errors=True)

    def write_deferred(self, doc):
        d = env.artifacts("deferred", self.chk)
        (d / "B-hybrid.json").write_text(json.dumps(doc))

    def test_hybrid_candidate_needs_a_valid_passing_deferred_record(self):
        res = selection.evaluate_eligibility(self.roster, self.chk, self.enc)
        self.assertTrue(res["eligibility"]["c-prov"]["eligible"])
        self.assertNotIn("deferred_a3", res["eligibility"]["c-prov"])
        self.assertFalse(res["eligibility"]["B-hybrid"]["eligible"])
        self.assertTrue(any("B-hybrid" in p and "deferred" in p for p in res["provenance_problems"]))
        partial = _valid_report()
        partial.update(partial=True, synthetic=None, scope=dict(partial["scope"], synthetic=False), passes=True)
        self.write_deferred(partial)
        res = selection.evaluate_eligibility(self.roster, self.chk, self.enc)
        self.assertFalse(res["eligibility"]["B-hybrid"]["eligible"])
        self.assertTrue(res["provenance_problems"])
        self.write_deferred(_valid_report())
        res = selection.evaluate_eligibility(self.roster, self.chk, self.enc)
        self.assertTrue(res["eligibility"]["B-hybrid"]["eligible"], res)
        self.assertEqual(res["provenance_problems"], [])
        failing = _valid_report()
        failing["synthetic"].update(failures=3)
        failing["passes"] = False
        self.write_deferred(failing)
        res = selection.evaluate_eligibility(self.roster, self.chk, self.enc)
        self.assertFalse(res["eligibility"]["B-hybrid"]["eligible"])
        self.assertEqual(res["provenance_problems"], [])

    def test_both_reporting_paths_use_it(self):
        import inspect
        for fn in (selection.report, selection.regenerate):
            self.assertIn("evaluate_eligibility(", inspect.getsource(fn), fn.__name__)

    def test_unbound_or_missing_audit_and_gate_problems_are_provenance(self):
        (selection.sel_dir(self.chk) / "audit" / "c-prov.json").write_text(json.dumps({"within_budget": True}))
        self.write_deferred(_valid_report())
        res = selection.evaluate_eligibility(self.roster, self.chk, self.enc)
        self.assertTrue(any("c-prov" in p and "audit" in p for p in res["provenance_problems"]))
        selection._gate_verdict = lambda chk, c, patches: {"passes": False, "problems": ["sources/exports/workloads changed"],
                                                           "complete": True}
        res = selection.evaluate_eligibility(self.roster, self.chk, self.enc)
        self.assertTrue(any("gate" in p for p in res["provenance_problems"]))


class DecisionWrites(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=str(env.ARTIFACTS_ROOT / "H")))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_provenance_problems_never_write(self):
        path = self.tmp / "report-corrected.json"
        accepted = {"winner_rule": {"winner": "B-hybrid"}, "provenance_problems": []}
        path.write_text(json.dumps(accepted))
        before = path.read_bytes()
        with self.assertRaisesRegex(SystemExit, "provenance"):
            selection.write_decision(path, {"winner_rule": {"winner": None}, "provenance_problems": ["gate x: stale"]})
        self.assertEqual(path.read_bytes(), before)
        fresh = self.tmp / "new.json"
        with self.assertRaisesRegex(SystemExit, "provenance"):
            selection.write_decision(fresh, {"winner_rule": {"winner": "a"}, "provenance_problems": ["audit a: missing"]})
        self.assertFalse(fresh.exists())

    def test_clean_decision_writes_and_keeps_the_previous_one(self):
        path = self.tmp / "report-corrected.json"
        path.write_text(json.dumps({"winner_rule": {"winner": "B-hybrid"}, "provenance_problems": []}))
        selection.write_decision(path, {"winner_rule": {"winner": "B-hybrid"}, "provenance_problems": []})
        self.assertEqual(json.loads(path.read_text())["winner_rule"]["winner"], "B-hybrid")
        self.assertEqual(len(list(self.tmp.glob("report-corrected.previous-*.json"))), 1)

    def test_check_run_name_follows_the_source_state(self):
        name = selection.checks_run_name("sel-x", [])
        self.assertTrue(name.startswith("sel-x-chk-"))
        self.assertEqual(selection.checks_run_name("sel-x", []), name)
        probe = env.SHARED / f"_name_probe_{os.getpid()}.py"
        try:
            probe.write_text("# temporary\n")
            self.assertNotEqual(selection.checks_run_name("sel-x", []), name)
        finally:
            probe.unlink()
        self.assertEqual(selection.checks_run_name("sel-x", []), name)


class PlanAmendment(unittest.TestCase):
    def test_a15_is_the_rule_and_is_fingerprinted(self):
        plan = env.BINARY_DIR / "plans" / "gpu-kernel.md"
        text = plan.read_text()
        self.assertIn("### A15.", text)
        a15 = text[text.index("### A15."):]
        for phrase in ("A*", "anchored", "16 of 24", "1%", "latency", "name", "top two", "retrospective"):
            self.assertIn(phrase, a15)
        self.assertIn(plan, F.source_files())
        self.assertIn("A15", selection.__doc__)


class GeneralArms(unittest.TestCase):
    """The validator is parametrized by candidate, provider module and part count, so a further hybrid
    arm (e.g. a B-hybrid-x2 under plans/x2-default.md) is validated by the same code: the scope records
    the part count, and a report of another part count is refused as a provenance problem."""

    def test_part_count_is_part_of_the_scope(self):
        self.assertEqual(deferred_a3.full_scope()["components"], 3)
        self.assertEqual(deferred_a3.full_scope(2)["components"], 2)
        doc = _valid_report()
        self.assertTrue(selection.deferred_a3_check(doc, "B-hybrid", MODULE)["ok"])
        res = selection.deferred_a3_check(doc, "B-hybrid", MODULE, components=2)
        self.assertFalse(res["ok"])
        self.assertTrue(any("scope" in p for p in res["provenance"]))

    def test_roster_parameters_reach_the_validator(self):
        roster = {"candidates": ["X-hybrid"], "stack_baseline": {"X-hybrid": "R-hybrid"},
                  "deferred_module": "m.n", "deferred_components": {"X-hybrid": 2}}
        self.assertEqual(selection.hybrid_candidates(roster), ["X-hybrid"])
        self.assertEqual(selection.deferred_parameters(roster, "X-hybrid"), ("m.n", 2))
        self.assertEqual(selection.deferred_parameters({"candidates": []}, "B-hybrid"), (MODULE, 3))


class CriterionCoverage(unittest.TestCase):
    """plans/x2-default.md Z5: deferred-exactness coverage is specific to each criterion.
      - x3: x3-relative A3 limits; complete coverage including discriminating cases.
      - x2: bitwise deferred == in-kernel x2 on every case; the same expected real and synthetic case set
        (cancellation and boundary cases included); no discriminating-case requirement.
      An x2 report missing any expected case, or with one inexact case, is refused. The provider's
      registered part count must equal the requested one."""

    def x2_report(self):
        cov = deferred_a3.expected_coverage()
        scope = deferred_a3.full_scope(2)
        ident = {"kernel": "int8_mma:x2", "factory": "gpu1bit.int8_mma.hybrid_x2:HybridProjectionX2", "components": 2}
        return {"provider": "B-hybrid-x2", "module": "gpu1bit.int8_mma.deferred_x2", "informational": False,
                "provider_identity": ident, "scope": scope, "partial": False,
                "real": {"evaluations": cov["real_evaluations"], "failures": 0, "deferred_evaluations": cov["deferred_evaluations"],
                         "deferred_failures": 0, "deferral_mismatches": 0, "inexact": 0},
                "synthetic": {"cases": cov["synthetic_cases"], "failures": 0, "not_deferred": 0, "inexact": 0},
                "passes": True, "fingerprint": deferred_a3.record_fingerprint("B-hybrid-x2", "gpu1bit.int8_mma.deferred_x2", scope)}

    def check(self, doc):
        saved = deferred_a3.provider_identity
        deferred_a3.provider_identity = lambda p, m: {"kernel": "int8_mma:x2", "components": 2,
                                                      "factory": "gpu1bit.int8_mma.hybrid_x2:HybridProjectionX2"}
        try:
            return selection.deferred_a3_check(doc, "B-hybrid-x2", "gpu1bit.int8_mma.deferred_x2", components=2)
        finally:
            deferred_a3.provider_identity = saved

    def test_scopes_name_their_criterion(self):
        self.assertEqual(deferred_a3.full_scope(3)["criterion"], "x3-relative A3 limits")
        self.assertEqual(deferred_a3.full_scope(2)["criterion"], "bitwise deferred == in-kernel")
        self.assertIn(2, deferred_a3.HARNESS_COMPONENTS)

    def test_complete_exact_x2_report_is_accepted_without_discriminating_cases(self):
        res = self.check(self.x2_report())
        self.assertTrue(res["ok"], res)

    def test_x2_refusals(self):
        cov = deferred_a3.expected_coverage()
        for name, edit in {"a missing synthetic case": lambda d: d["synthetic"].update(cases=cov["synthetic_cases"] - 1),
                           "a missing real evaluation": lambda d: d["real"].update(evaluations=cov["real_evaluations"] - 1),
                           "one inexact case": lambda d: (d["synthetic"].update(failures=1, inexact=1), d.update(passes=False)),
                           "an x3-criterion scope": lambda d: d.update(scope=deferred_a3.full_scope(3))}.items():
            with self.subTest(refusal=name):
                doc = copy.deepcopy(self.x2_report())
                edit(doc)
                self.assertFalse(self.check(doc)["ok"])

    def test_x3_still_needs_discriminating_cases(self):
        doc = _valid_report()
        doc["synthetic"]["discriminating_cases"] = 0
        self.assertFalse(selection.deferred_a3_check(doc, "B-hybrid", MODULE)["ok"])

    def test_registered_part_count_must_match(self):
        import types
        fake = types.ModuleType("fake_deferred_parts")
        fake.DEFERRED_PROVIDERS = {"X": {"kernel": "int8_mma:x2", "factory": object, "components": 2}}
        import sys
        sys.modules["fake_deferred_parts"] = fake
        try:
            self.assertEqual(deferred_a3.provider_identity("X", "fake_deferred_parts")["components"], 2)
            with self.assertRaisesRegex(ValueError, "part count"):
                deferred_a3.provider_spec("X", "fake_deferred_parts", components=3)
        finally:
            del sys.modules["fake_deferred_parts"]
