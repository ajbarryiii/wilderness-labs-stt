"""Contract tests (CPU): the stage 5 step 0 driver keat_step0.py and M1's init check training/keat_m1.py (keat-prime plan
"Inputs", "Step 0", "Scoring and gate").

- M1's init check: the SHA-256 before unpickling, then complete, train.py's format, step 250,000, weight fraction 1,
  args arm M1 with recipe P2 and no quantizer; MB1's checkpoint is refused (the real files when present);
- the M1 check: every dev set within 0.02 points of run ap-20261020r4's M1-x1 arm, whose WERs are pinned;
- the rule: per model as stage 4's, the largest scope passing on M1 and on B0' chosen, B0''s rows read from stage
  4's step 0 report (read only, refused unless its SHA-256 is the pinned one);
- the report on synthetic scorings: M1's rows against "none", B0''s rows, the chosen scope, smoke never a decision;
- outputs only under $RELABEL_ROOT/keat-prime (smoke apart from production), the write guard's roots exclude stage
  4's keab-prime; the production run's variants and set groups; the recorded code identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import keab_step0 as k0
import keat_step0 as t0
import relabel_paths
from test_keab_step0 import records, result
from training import keat_export as tke
from training import keat_m1 as km1
from training import write_guard


def tmpdir(prefix: str) -> Path:
    relabel_paths.require_mount()
    base = relabel_paths.ARTIFACTS / "tmp"
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=base))


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


M1_ARGS = {"arm": "M1", "recipe": "P2", "run_name": "main-M1-P2-lr5e-4", "lr": 5e-4}


def payload(**changes) -> dict:
    p = {"step": km1.M1_STEP, "weight_fraction": 1.0, "format": 2, "complete": True, "args": dict(M1_ARGS),
         "student": {"encoder.w": torch.arange(4.0)}}
    for k, v in changes.items():
        if k.startswith("args."):
            p["args"][k[5:]] = v
        else:
            p[k] = v
    return p


class InitCheck(unittest.TestCase):
    def setUp(self):
        self.dir = tmpdir("keat-init-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.production_cache = km1.sha_cache_dir()
        patch = mock.patch.object(km1, "sha_cache_dir", lambda: self.dir / "sha-cache")   # tmp files stay out of it
        patch.start()
        self.addCleanup(patch.stop)

    def save(self, name: str, data: dict) -> Path:
        path = self.dir / name
        torch.save(data, path)
        return path

    def test_m1_like_checkpoint_is_accepted(self):
        path = self.save("m1.pt", payload())
        state, facts = km1.load_m1_state(path, expected_sha256=sha(path))
        self.assertTrue(torch.equal(state["encoder.w"], torch.arange(4.0)))
        self.assertEqual((facts["step"], facts["weight_fraction"], facts["arm"], facts["recipe"]),
                         (km1.M1_STEP, 1.0, "M1", "P2"))
        self.assertEqual(facts["sha256"], sha(path))

    def test_each_deviation_is_refused(self):
        bad = {"step": 249_999, "weight_fraction": 0.5, "format": 1, "complete": False, "args.arm": "P2",
               "args.recipe": "P3", "args.quantizer": "binary"}
        for key, value in bad.items():
            with self.subTest(key=key):
                path = self.save(f"bad-{key}.pt", payload(**{key: value}))
                with self.assertRaises(RuntimeError):
                    km1.load_m1_state(path, expected_sha256=sha(path))

    def test_mb1_like_checkpoint_is_refused(self):
        mb1 = payload(**{"args.quantizer": "binary", "args.ramp_kind": "fp32", "args.arm": "binB",
                         "args.init": "m1"})
        path = self.save("mb1.pt", mb1)
        with self.assertRaises(RuntimeError):
            km1.load_m1_state(path, expected_sha256=sha(path))

    def test_sha_mismatch_is_refused_before_unpickling(self):
        path = self.save("other.pt", payload())
        with mock.patch.object(torch, "load", side_effect=AssertionError("unpickled")):
            with self.assertRaises(RuntimeError):
                km1.load_m1_state(path, expected_sha256="0" * 64)

    def test_pins(self):
        import binary_paths
        self.assertEqual(km1.M1_SHA256, binary_paths.M1_SHA256)
        self.assertTrue(km1.M1_SHA256.startswith("003b1289"))
        self.assertTrue(km1.M1_EXPORT_SHA256.startswith("a287e977"))
        self.assertEqual(km1.m1_checkpoint(), relabel_paths.PARAKEET_TERNARY / "runs" / "main-M1-P2-lr5e-4" / "ckpt"
                         / "latest.pt")
        self.assertEqual(km1.m1_export(), relabel_paths.PARAKEET_TERNARY / "runs" / "main-M1-P2-lr5e-4" / "export")
        self.assertEqual(km1.init_record()["sha256"], km1.M1_SHA256)

    def test_sha_cache_is_under_keat_prime_and_follows_the_file(self):
        self.assertEqual(self.production_cache, tke.keat_root() / "caches" / "sha256")
        path = self.save("cached.pt", payload())
        self.assertEqual(km1.cached_sha256(path), sha(path))
        self.assertEqual(len(list((self.dir / "sha-cache").glob("*.json"))), 1)
        self.save("cached.pt", payload(step=1))                  # rewritten: hashed again
        self.assertEqual(km1.cached_sha256(path), sha(path))


@unittest.skipUnless(km1.m1_checkpoint().exists() and k0.mb1_checkpoint().exists(), "needs M1's and MB1's checkpoints")
class RealCheckpoints(unittest.TestCase):
    def test_m1_accepted_and_mb1_refused(self):
        state, facts = km1.load_m1_state()
        self.assertEqual(facts["sha256"], km1.M1_SHA256)
        self.assertEqual((facts["step"], facts["weight_fraction"], facts["format"]), (250_000, 1.0, 2))
        self.assertEqual(len([k for k in state if k.startswith("encoder.layers.")]) > 0, True)
        del state
        with self.assertRaises(RuntimeError):
            km1.load_m1_state(k0.mb1_checkpoint())


class M1Check(unittest.TestCase):
    def test_within_002_per_set(self):
        same = t0.check_against_recorded(dict(t0.RECORDED_X1))
        self.assertTrue(same["pass"])
        self.assertAlmostEqual(same["h4"], t0.RECORDED_X1_H4, places=12)
        moved = dict(t0.RECORDED_X1, ami_dev=t0.RECORDED_X1["ami_dev"] + 0.021)
        self.assertFalse(t0.check_against_recorded(moved)["pass"])
        moved = dict(t0.RECORDED_X1, yodas_dev=t0.RECORDED_X1["yodas_dev"] - 0.021)
        self.assertFalse(t0.check_against_recorded(moved)["pass"], "YODAS is checked too")
        edge = dict(t0.RECORDED_X1, librispeech_dev_clean=t0.RECORDED_X1["librispeech_dev_clean"] + 0.0199)
        self.assertTrue(t0.check_against_recorded(edge)["pass"])

    def test_recorded_values_are_the_plans(self):
        plan = {"librispeech_dev_clean": 1.9891, "librispeech_dev_other": 4.2365, "ami_dev": 11.7323,
                "voxpopuli_dev": 6.0089, "yodas_dev": 6.5834}
        self.assertEqual({k: round(v, 4) for k, v in t0.RECORDED_X1.items()}, plan)
        self.assertEqual(round(t0.RECORDED_X1_H4, 3), 5.992)
        self.assertEqual((t0.RECORDED_RUN, t0.RECORDED_ARM), ("ap-20261020r4", "M1-x1"))

    @unittest.skipUnless(t0.recorded_summary_path().exists(), "needs the recorded x1 arm")
    def test_recorded_values_are_the_arms_summary(self):
        s = json.loads(t0.recorded_summary_path().read_text())
        self.assertEqual((s["run"], s["arm"], s["complete"], s["spec"]["parts"]), ("ap-20261020r4", "M1-x1", True, 1))
        self.assertEqual(s["dev_wer_percent"], t0.RECORDED_X1)


def stage4_report(path: Path, s1: bool = True, s2: bool = True, decision: bool = True) -> Path:
    rows = {"S1-int8": {"h4_delta": 0.003, "h4_ci": [-0.001, 0.007], "rule": {"pass": s1, "worst_set_delta": 0.007}},
            "S2-int8": {"h4_delta": 0.016, "h4_ci": [0.004, 0.028], "rule": {"pass": s2, "worst_set_delta": 0.054}}}
    path.write_text(json.dumps({"limit": None if decision else 20, "decision": decision,
                                "passes": {"MB1": {"S1": True, "S2": True}, "B0p": {"S1": s1, "S2": s2}},
                                "models": {"B0p": {"none": {"h4": 5.127}, "variants": rows}}}))
    return path


class B0pRows(unittest.TestCase):
    def setUp(self):
        self.dir = tmpdir("keat-s4r-")
        self.addCleanup(shutil.rmtree, self.dir, True)

    def test_read_only_and_checked_by_hash(self):
        path = stage4_report(self.dir / "report.json", s1=True, s2=False)
        with self.assertRaises(SystemExit):
            t0.stage4_b0p(path)
        with mock.patch.object(t0, "STAGE4_REPORT_SHA256", sha(path)):
            got = t0.stage4_b0p(path)
        self.assertEqual(got["passes"], {"S1": True, "S2": False})
        self.assertEqual(got["sha256"], sha(path))
        self.assertAlmostEqual(got["rows"]["S2-int8"]["h4_delta"], 0.016)

    def test_a_smoke_report_is_refused(self):
        path = stage4_report(self.dir / "smoke.json", decision=False)
        with mock.patch.object(t0, "STAGE4_REPORT_SHA256", sha(path)), self.assertRaises(SystemExit):
            t0.stage4_b0p(path)

    @unittest.skipUnless(t0.stage4_report_path().exists(), "needs stage 4's step 0 report")
    def test_stage_4s_real_rows(self):
        got = t0.stage4_b0p()
        self.assertEqual(got["passes"], {"S1": True, "S2": True})
        self.assertEqual(round(got["rows"]["S1-int8"]["h4_delta"], 3), 0.003)
        self.assertEqual(round(got["rows"]["S2-int8"]["h4_delta"], 3), 0.016)
        self.assertTrue(t0.stage4_report_path().is_relative_to(relabel_paths.ARTIFACTS / "keab-prime"))


class Rule(unittest.TestCase):
    def test_largest_scope_passing_on_m1_and_b0p(self):
        both = {"S1": True, "S2": True}
        self.assertEqual(t0.choose({"M1": both, "B0p": both}), "S2")
        self.assertEqual(t0.choose({"M1": {"S1": True, "S2": False}, "B0p": both}), "S1")
        self.assertEqual(t0.choose({"M1": both, "B0p": {"S1": True, "S2": False}}), "S1")
        self.assertEqual(t0.choose({"M1": {"S1": False, "S2": True}, "B0p": both}), "S2")
        self.assertEqual(t0.choose({"M1": {"S1": False, "S2": False}, "B0p": both}), "none")

    def test_thresholds_are_stage_4s(self):
        self.assertIs(t0.rule, k0.rule)
        self.assertEqual((k0.COST_MAX, k0.CI_UPPER_MAX, k0.SET_MAX), (0.03, 0.06, 0.10))


SCORER = "5" * 64


def write_scores(root: Path, variant: str, ami_errors: list[int], n: int = 12, scorer: str = SCORER) -> None:
    d = root / "scores" / "M1" / variant
    d.mkdir(parents=True, exist_ok=True)
    for s in k0.DEV_SETS:
        res = result(records(s, ami_errors if s == "ami_dev" else [1] * n))
        res["wer_percent"] = 100 * res["wer"]["wer"]
        res["config"] = {"variant": variant, "scorer": {"code_sha256": scorer}}
        (d / f"{s}.json").write_text(json.dumps(res))


def write_check(root: Path, passed: bool = True, scorer: str = SCORER, limit=None) -> None:
    d = root / "m1-x1-check"
    d.mkdir(parents=True, exist_ok=True)
    summary = t0.check_against_recorded(dict(t0.RECORDED_X1))
    summary.update({"pass": passed, "limit": limit, "scorer_sha256": scorer})
    (d / "summary.json").write_text(json.dumps(summary))


class Report(unittest.TestCase):
    def setUp(self):
        self.dir = tmpdir("keat-report-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.s4 = stage4_report(self.dir / "s4.json")
        self.patches = [mock.patch.object(t0, "out_root", lambda limit: self.dir / ("prod" if limit is None else
                                                                                    f"smoke{limit}")),
                        mock.patch.object(t0, "step0_root", lambda: self.dir / "prod"),
                        mock.patch.object(t0, "stage4_report_path", lambda: self.s4),
                        mock.patch.object(t0, "STAGE4_REPORT_SHA256", sha(self.s4))]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def run_report(self, limit=None) -> dict:
        with mock.patch("builtins.print"):
            t0.report(argparse.Namespace(limit=limit))
        return json.loads((t0.out_root(limit) / "report.json").read_text())

    def test_chosen_scope_from_m1_and_b0p(self):
        n = 12
        write_scores(self.dir / "prod", "none", [1] * n)
        write_scores(self.dir / "prod", "S1-int8", [1] * n)
        write_scores(self.dir / "prod", "S2-int8", [1] * (n - 1) + [2])          # +0.83 on AMI: S2 fails on M1
        write_check(self.dir / "prod")
        r = self.run_report()
        self.assertTrue(r["m1_check_gate"]["pass"])
        self.assertTrue(r["decision"])
        self.assertEqual(r["passes"]["M1"], {"S1": True, "S2": False})
        self.assertEqual(r["passes"]["B0p"], {"S1": True, "S2": True})
        self.assertEqual(r["chosen_scope"], "S1")
        self.assertEqual(r["b0p"]["sha256"], sha(self.s4))
        m1 = r["models"]["M1"]["variants"]
        self.assertAlmostEqual(m1["S2-int8"]["delta"]["ami_dev"], 100 / 120)
        self.assertEqual(m1["S1-int8"]["h4_delta"], 0.0)
        self.assertTrue((self.dir / "prod" / "report.md").exists())

    def test_no_decision_without_a_passing_m1_check_of_the_same_scorer(self):
        for v in t0.VARIANTS:
            write_scores(self.dir / "prod", v, [1] * 12)
        cases = {"absent": None, "failed": dict(passed=False), "other scorer": dict(scorer="6" * 64),
                 "smoke check": dict(limit=8)}
        for case, check in cases.items():
            with self.subTest(case=case):
                shutil.rmtree(self.dir / "prod" / "m1-x1-check", ignore_errors=True)
                if check is not None:
                    write_check(self.dir / "prod", **check)
                r = self.run_report()
                self.assertEqual(r["passes"]["M1"], {"S1": True, "S2": True})
                self.assertFalse(r["m1_check_gate"]["pass"])
                self.assertIsNone(r["chosen_scope"])
        write_check(self.dir / "prod")
        write_scores(self.dir / "prod", "S2-int8", [1] * 12, scorer="7" * 64)     # one variant by another scorer
        r = self.run_report()
        self.assertFalse(r["m1_check_gate"]["pass"])
        self.assertIsNone(r["chosen_scope"])

    def test_incomplete_or_smoke_is_never_a_decision(self):
        write_scores(self.dir / "prod", "none", [1] * 12)
        write_scores(self.dir / "prod", "S1-int8", [1] * 12)
        r = self.run_report()
        self.assertIsNone(r["chosen_scope"])
        for v in t0.VARIANTS:
            write_scores(self.dir / "smoke8", v, [1] * 12)
        r = self.run_report(limit=8)
        self.assertFalse(r["decision"])
        self.assertIsNone(r["chosen_scope"])
        self.assertEqual(r["slice_rule_outcome"], "S2")


class RequireCheck(unittest.TestCase):
    def setUp(self):
        self.dir = tmpdir("keat-req-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        for p in (mock.patch.object(t0, "step0_root", lambda: self.dir),
                  mock.patch.object(t0, "require_x1_process", lambda: None)):
            p.start()
            self.addCleanup(p.stop)

    def test_production_scoring_needs_the_passing_check_of_this_scorer(self):
        scorer = t0.scorer_identity()["sha256"]
        with self.assertRaises(SystemExit):
            t0.require_m1_check(scorer)
        for kw in (dict(passed=False, scorer=scorer), dict(scorer="6" * 64), dict(scorer=scorer, limit=8)):
            write_check(self.dir, **kw)
            with self.subTest(kw=kw), self.assertRaises(SystemExit):
                t0.require_m1_check(scorer)
        write_check(self.dir, scorer=scorer)
        self.assertTrue(t0.require_m1_check(scorer)["pass"])

    def test_run_refuses_before_any_scoring(self):
        scored = mock.Mock(return_value=None)
        args = argparse.Namespace(variant="S1-int8", sets=["dev"], limit=None, lock_wait=0.0, no_gpu_lock=False)
        with mock.patch.object(t0, "checked_weights", lambda v: (self.dir, {"view": "v"})), \
                mock.patch.object(t0, "_scored", scored), mock.patch.object(t0, "scores_dir", lambda v, l: self.dir):
            with self.assertRaises(SystemExit):
                t0.run(args)
            scored.assert_not_called()
            t0.run(argparse.Namespace(**{**vars(args), "limit": 8}))            # a smoke needs no check
            scored.assert_called_once()

    def test_set_config_names_the_scorer(self):
        ident = t0.scorer_identity()
        cfg = t0.set_config("none", "librispeech_dev_clean", "dev", 4, {"export": "x"})
        self.assertEqual(cfg["scorer"]["code_sha256"], ident["sha256"])
        for key in ("relabel/training/keat_deployed.py", "relabel/training/deployed.py", "relabel/keab_step0.py",
                    "relabel/python-gpu1bit", "parakeet-ternary/evaluate.py", "whisper-ternary/wer.py",
                    "parakeet-binary/gpu1bit/runtime_m1.py", "parakeet-ternary/inference/encoder4.py"):
            self.assertIn(key, ident["files"])
        self.assertNotIn("relabel/keat_step0.py", ident["files"], "report-side code does not invalidate the check")


class FakeModels:
    facts = {"sha256": "c" * 64, "step": km1.M1_STEP}

    def __call__(self):
        model = mock.Mock()
        model.to.return_value = model
        return model


class PrepareA2(unittest.TestCase):
    """The none view must equal M1's export (A2); a failure is refused on every retry and no other view is built."""

    def setUp(self):
        self.dir = tmpdir("keat-prep-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.built: list[str] = []
        self.none_sha = "0" * 64

        def derive(bdir, vdir, extra=None):
            sha_ = self.none_sha if extra["variant"] == "none" else hashlib.sha256(str(vdir).encode()).hexdigest()
            self.built.append(extra["variant"])
            Path(vdir).mkdir(parents=True, exist_ok=True)
            (Path(vdir) / "manifest.json").write_text(json.dumps({"sha256": sha_}))
            return {"sha256": sha_, "bytes": {"file_bytes": 10, "header_bytes": 2}}

        from training import keat_eightbit as teb
        for p in (mock.patch.object(t0, "exports_root", lambda: self.dir / "exports"),
                  mock.patch.object(teb, "apply_keat_scope", lambda m, s: {"scope": s}),
                  mock.patch.object(tke, "write_bundle", lambda m, d, extra=None: {"sha256": "b", "bytes": {
                      "tensor_bytes": 8}}),
                  mock.patch.object(tke, "derive_view", derive)):
            p.start()
            self.addCleanup(p.stop)

    def prepare(self, variants=t0.VARIANTS) -> int:
        index_path = self.dir / "exports" / "index.json"
        index = json.loads(index_path.read_text()) if index_path.exists() else {"variants": {}}
        args = argparse.Namespace(variants=list(variants), force=False, no_gpu_lock=True, lock_wait=0.0)
        with mock.patch.object(t0, "log"):
            return t0._prepare(args, FakeModels(), index, index_path)

    def test_failure_is_refused_on_every_retry(self):
        self.assertEqual(self.prepare(), t0.EXIT_CHECK)
        self.assertEqual(self.built, ["none"])
        self.assertEqual(self.prepare(), t0.EXIT_CHECK, "the retry must not skip the failed none view")
        self.assertEqual(self.prepare(("S1-int8", "S2-int8")), t0.EXIT_CHECK, "no view without a passing none")
        self.assertNotIn("S1-int8", self.built)

    def test_views_after_a_passing_none(self):
        self.none_sha = km1.M1_EXPORT_SHA256
        self.assertEqual(self.prepare(("S2-int8", "none", "S1-int8")), 0)
        self.assertEqual(self.built, ["none", "S1-int8", "S2-int8"], "none first, whatever the order asked")
        self.built.clear()
        self.assertEqual(self.prepare(), 0)
        self.assertEqual(self.built, [], "existing views are kept")


class Paths(unittest.TestCase):
    def test_outputs_under_keat_prime_smoke_apart(self):
        prod, smoke = t0.out_root(None), t0.out_root(20)
        self.assertTrue(prod.is_relative_to(tke.keat_root()))
        self.assertTrue(smoke.is_relative_to(tke.keat_root()))
        self.assertFalse(smoke.is_relative_to(prod))
        self.assertTrue(t0.scores_dir("S1-int8", 20).is_relative_to(smoke))
        self.assertTrue(t0.scores_dir("S2-int8", None).is_relative_to(prod))
        self.assertTrue(t0.view_dir("S2-int8").is_relative_to(prod))
        self.assertTrue(t0.bundle_dir("S1-int8").is_relative_to(prod))
        with self.assertRaises(ValueError):
            t0.out_root(0)
        with self.assertRaises(ValueError):
            t0.scores_dir("S2-palette", None)

    def test_none_is_m1s_own_export(self):
        self.assertEqual(t0.weights_of("none")[0], km1.m1_export())
        self.assertEqual(t0.weights_of("S2-int8")[0], t0.view_dir("S2-int8"))

    def test_write_guard_roots_exclude_stage_4(self):
        roots = t0.allowed_roots()
        keab = relabel_paths.ARTIFACTS / "keab-prime"
        self.assertIn(tke.keat_root(), roots)
        self.assertFalse(any(keab.is_relative_to(r) or Path(r).is_relative_to(keab) for r in roots))
        target = keab / f"no-such-dir-{id(self)}" / "x.json"
        with write_guard.active(roots, files=[relabel_paths.GPU_LOCK]):
            with self.assertRaises(PermissionError):
                open(target, "w")
        self.assertFalse(target.parent.exists())

    def test_production_plan(self):
        plan = t0.full_plan()
        self.assertEqual(plan, [("none", ("dev", "domain")), ("S1-int8", ("dev", "domain")),
                                ("S2-int8", ("dev", "domain"))])


class Lease(unittest.TestCase):
    def test_the_jobs_lease_environment_is_recorded(self):
        import os
        env = {"AGENT_JOBS_LEASES": '[{"resource": "gpu", "compute": 100}]', "AGENT_JOBS_VRAM_FRACTION": "0.2",
               "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "100", "UNRELATED": "x"}
        with mock.patch.dict(os.environ, env):
            got = t0.lease_record()
        for key in ("AGENT_JOBS_LEASES", "AGENT_JOBS_VRAM_FRACTION", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"):
            self.assertEqual(got["env"][key], env[key])
        self.assertNotIn("UNRELATED", got["env"])


class CodeIdentity(unittest.TestCase):
    def test_covers_this_stages_code_the_x1_trees_and_the_wrappers(self):
        ident = t0.code_identity()
        files = ident["files"]
        for key in ("relabel/keat_step0.py", "relabel/training/keat_deployed.py", "relabel/training/keat_export.py",
                    "relabel/training/keat_eightbit.py", "relabel/training/keat_m1.py", "relabel/training/deployed.py",
                    "relabel/python", "relabel/python-gpu1bit", "parakeet-binary/gpu1bit/runtime_m1.py",
                    "parakeet-binary/gpu1bit/shared/fulldev.py", "parakeet-binary/gpu1bit/shared/python",
                    "parakeet-ternary/inference/encoder4.py", "parakeet-ternary/evaluate.py"):
            self.assertIn(key, files)
        self.assertEqual(ident["sha256"], hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())
        self.assertEqual(files["relabel/keat_step0.py"], sha(Path(t0.__file__)))
        self.assertRegex(ident["git_head"] or "", r"^([0-9a-f]{40})?$")


if __name__ == "__main__":
    unittest.main()
