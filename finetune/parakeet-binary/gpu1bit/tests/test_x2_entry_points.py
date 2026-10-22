"""Every x2 entry point works under the candidate selection with the live pointer still at x3 (Codex x2 c4,
major; the class of c1 findings 7 and 8). The gates, the preflight and its duration budget, the benchmark
session and test scoring are all defined under shared/selected-x2.json; before promotion the pointer
shared/selected.json is the x3 document, and GPU1BIT_SELECTION may be unset. CPU:
  - preflight.budget, called with GPU1BIT_SELECTION unset, verifies a candidate-identity gate run (a fixture
    run frozen under the candidate) instead of refusing with a selection mismatch;
  - each CLI action of x2.gates, x2.preflight, x2.bench_session and x2.testscore reaches its function with
    the candidate selected (GPU1BIT_SELECTION = selected-x2.json), whatever was set before.

    gpu1bit/shared/python -m gpu1bit.tests.run --cpu gpu1bit.tests.test_x2_entry_points
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from gpu1bit.shared import env, fulldev, seldoc


@contextlib.contextmanager
def _env_selection(value):
    saved = os.environ.get("GPU1BIT_SELECTION")
    if value is None:
        os.environ.pop("GPU1BIT_SELECTION", None)
    else:
        os.environ["GPU1BIT_SELECTION"] = str(value)
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("GPU1BIT_SELECTION", None)
        else:
            os.environ["GPU1BIT_SELECTION"] = saved


class BudgetWithPointerAtX3(unittest.TestCase):
    def setUp(self):
        if env.sha256_file(seldoc.SELECTED) == env.sha256_file(seldoc.CANDIDATE):
            self.skipTest("promoted: the pointer is the candidate")
        from gpu1bit.actprec import parts
        from gpu1bit.x2 import preflight as PF
        self.PF = PF
        parts.install()
        self.run = f"test-budget-{os.getpid()}"
        self.arm = "M1-x2"
        self.tmp = Path(tempfile.mkdtemp(dir=str(env.ARTIFACTS_ROOT / "H")))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.addCleanup(shutil.rmtree, fulldev.ROOT / self.run, True)
        with _env_selection(seldoc.CANDIDATE):                      # the gate run is frozen under the candidate
            self.frozen = fulldev.freeze(self.run, self.arm, "full")
        dev = [r for n in fulldev.DEV_SETS for _, r in fulldev.subset_records(n, "full")]
        chunk = {"identity": self.frozen, "rows": {"all": int(0.6 * len(dev)) + 1},
                 "audio_s": {"all": 0.6 * sum(r["duration"] for r in dev) + 1}, "work_s": 100.0}
        fulldev.chunks_path(self.run, self.arm, "full").write_text(json.dumps(chunk) + "\n")
        pf = self.tmp / f"preflight-{self.arm}.json"
        pf.write_text(json.dumps({"identity": self.frozen, "startup_s": 5.0, "capture_s": 10.0, "lease_bytes": 2**33}))
        saved = (PF.preflight_path, PF.budget_path)
        PF.preflight_path = lambda arm: self.tmp / f"preflight-{arm}.json"
        PF.budget_path = lambda gate_run, arm: self.tmp / f"budget-{gate_run}-{arm}.json"
        self.addCleanup(lambda: (setattr(PF, "preflight_path", saved[0]), setattr(PF, "budget_path", saved[1])))

    def test_budget_selects_the_candidate(self):
        with _env_selection(None):                                   # the live pointer, x3
            self.assertNotEqual(seldoc.load()["sha256"], env.sha256_file(seldoc.CANDIDATE))
            out = self.PF.budget(self.arm, self.run)
        self.assertEqual(out["identity"], self.frozen)
        self.assertGreater(out["timeout_s"], 0)
        self.assertTrue((self.tmp / f"budget-{self.run}-{self.arm}.json").exists())


class CliSelectsTheCandidate(unittest.TestCase):
    """Each CLI action runs with the candidate selected, even when GPU1BIT_SELECTION points elsewhere."""

    def dispatch(self, module, argv: list[str], patched: dict[str, object]):
        seen = []

        class _Stop(Exception):
            pass

        def stub(*a, **k):
            seen.append(os.environ.get("GPU1BIT_SELECTION"))
            raise _Stop
        saved = {k: getattr(module, k) for k in patched}
        for k in patched:
            setattr(module, k, stub)
        old_argv = sys.argv
        sys.argv = [module.__name__, *argv]
        try:
            with _env_selection(seldoc.SELECTED):                  # set to the x3 pointer explicitly
                with self.assertRaises(_Stop):
                    module.main()
        finally:
            sys.argv = old_argv
            for k, v in saved.items():
                setattr(module, k, v)
        self.assertEqual(seen, [str(seldoc.CANDIDATE)], f"{module.__name__} {argv}")

    def test_gates(self):
        from gpu1bit.x2 import gates as G
        for argv, fn in ((["evaluate", "--run", "R"], "evaluate"), (["recorded", "--run", "R"], "recorded"),
                         (["encoder", "--run", "R", "--arm", "g1", "--ref", "g2"], "encoder_compare"),
                         (["drive", "--run", "R"], "require_launch")):
            with self.subTest(action=argv[0]):
                self.dispatch(G, argv, {fn: None})

    def test_preflight(self):
        from gpu1bit.x2 import preflight as PF
        for argv, fn in ((["--arm", "M1-x2"], "run"), (["--arm", "M1-x2", "--budget-from-gate-run", "G"], "budget")):
            with self.subTest(argv=argv):
                self.dispatch(PF, argv, {fn: None})

    def test_testscore(self):
        from gpu1bit.x2 import testscore as T
        for argv, fn in ((["precheck", "--arm", "M1-x2"], "precheck"),
                         (["pin", "--arm", "M1-x2", "--attempt", "1", "--benchmark-run", "R", "--gates-run", "G"], "pin"),
                         (["summarize", "--arm", "M1-x2"], "recover_summary"),
                         (["score", "--arm", "M1-x2"], "score")):
            with self.subTest(action=argv[0]):
                saved = os.environ.pop("GPU1BIT_TESTSCORE_FAKE", None)
                try:
                    self.dispatch(T, argv, {fn: None})
                finally:
                    if saved is not None:
                        os.environ["GPU1BIT_TESTSCORE_FAKE"] = saved

    def test_bench_session(self):
        from gpu1bit.x2 import bench_session as BS
        self.dispatch(BS, ["roster", "--run", "R", "--gate-run", "G"], {"write_roster": None})


if __name__ == "__main__":
    unittest.main()
