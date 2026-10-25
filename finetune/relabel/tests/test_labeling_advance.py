"""Contract tests: labeling.advance, the driver that queues the checks after the GPU probes and C2 after the checks.

CPU only: fake probe, consistency and equivalence results in a temporary $RELABEL_ROOT, a fake submitter capturing
`job start` argv lists, a fake job registry and a fake c2-init.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from labeling import advance, c2, checks, layout
from labeling_fixtures import relabel_root

TOTAL = 31.84
RATES = {("vllm", 0): 10.0, ("vllm", 1): 30.0, ("vllm", 2): 20.0,
         ("transformers", 0): 5.0, ("transformers", 1): 4.0, ("transformers", 2): 8.0}
PEAKS = {("vllm", 0): 13.43, ("vllm", 1): 16.6, ("vllm", 2): 22.9,
         ("transformers", 0): 9.3, ("transformers", 1): 14.0, ("transformers", 2): 21.07}


def run_dict(engine: str, i: int, declared: float) -> dict:
    spec = (checks.vllm_engine(checks.VLLM_SETTINGS[i], TOTAL) if engine == "vllm"
            else checks.tf_engine(checks.TRANSFORMERS_SETTINGS[i]))
    return {"teacher": layout.TEACHER, "teacher_revision": "rev", "pool_dir": "pool-audio/x",
            "group_shards": checks.GROUP_SHARDS, "engine": spec, "b0": {"batch_seconds": checks.B0_BATCH_SECONDS},
            "declared_vram_gb": declared, "filters_version": 1}


def probe(engine: str, i: int, rate: float | None = None, peak: float | None = None, rss: float = 10.0,
          cgroup: float | None = None, repeat: int = 0) -> dict:
    rate = RATES[(engine, i)] if rate is None else rate
    peak = PEAKS[(engine, i)] if peak is None else peak
    run = run_dict(engine, i, 29.0)
    mon = lambda r, c: {"wall_s": 600.0, "peak_tree_rss_gb": r, "peak_cgroup_memory_gb": c, "peak_gpu_gb": peak}
    return {"engine": run["engine"], "contract": c2.contract(run), "audio_hours_per_gpu_hour": rate,
            "peak_vram_gb": peak, "declared_vram_gb_for_c2": peak + layout.VRAM_MARGIN_GB,
            "teacher_monitor": mon(rss, cgroup), "b0_monitor": mon(rss / 2, None),
            "split": {"signals": {"repeat_changed": repeat, "repeat_changed_share": repeat / 512}}}


def write_probes(**over) -> None:
    for e, i in advance.PROBES:
        p = over.get(f"{e}{i}", probe(e, i))
        if p is not None:
            c2.write_json(advance.probe_path(e, i), p)


def flag(argv: list[str], name: str) -> str | None:
    head = argv[:argv.index("--")]
    return head[head.index(name) + 1] if name in head else None


def command(argv: list[str]) -> list[str]:
    return argv[argv.index("--") + 1:]


class Jobs:
    """Fake `job start` (captures argv) and `job status` (names started)."""

    def __init__(self, fail_at: int | None = None):
        self.argvs: list[list[str]] = []
        self.fail_at = fail_at

    def submit(self, argv: list[str]) -> None:
        if self.fail_at is not None and len(self.argvs) == self.fail_at:
            self.fail_at = None
            raise RuntimeError("job start failed")
        assert argv[:2] == ["job", "start"], argv
        self.argvs.append(list(argv))

    def exists(self, name: str) -> bool:
        return name in self.names

    @property
    def names(self) -> list[str]:
        return [a[2] for a in self.argvs]

    def by_name(self, name: str) -> list[str]:
        return next(a for a in self.argvs if a[2] == name)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ctx = relabel_root(self.root)
        self.ctx.__enter__()
        self.jobs = Jobs()
        self.out: list[str] = []
        self.inits: list[list[str]] = []
        self.c1 = self.root / "pool-audio" / "w12-test"
        c2.write_json(self.c1 / "status.json", {"complete": True})
        c2.write_json(self.c1 / "manifest.json", {"complete": True, "distinct_hours": 1000.0})

    def tearDown(self):
        self.ctx.__exit__(None, None, None)
        self.tmp.cleanup()

    def fake_init(self, args: list[str]) -> None:
        self.inits.append(list(args))
        sel = self.selected_contract
        run = {"teacher": layout.TEACHER, "teacher_revision": sel["teacher_revision"], "pool_dir": "pool-audio/w12-test",
               "group_shards": sel["group_shards"], "engine": dict(sel["engine"]) | {"total_vram_gb": TOTAL},
               "b0": {"batch_seconds": sel["b0_batch_seconds"]}, "declared_vram_gb": sel["declared_vram_gb"],
               "filters_version": sel["filters_version"]}
        c2.write_json(layout.labels_dir(layout.TEACHER, advance.RUN) / "run.json", run)

    def run_step(self, step: str, *extra: str, jobs: Jobs | None = None) -> int:
        jobs = jobs or self.jobs
        return advance.main([step, *extra], submit=jobs.submit, exists=jobs.exists, init=self.fake_init,
                            c1_dir=lambda: self.c1, out=self.out.append)

    def files(self) -> dict[str, bytes]:
        return {str(p.relative_to(self.root)): p.read_bytes() for p in sorted(self.root.rglob("*")) if p.is_file()}

    def record(self, step: str) -> dict:
        return json.loads(advance.record_path(step).read_text())


class Checks(Base):
    def test_picks_fastest_setting_per_engine(self):
        write_probes()
        self.assertEqual(self.run_step("checks"), 0)
        self.assertIn("relabel-consistency-vllm-1", self.jobs.names)
        self.assertIn("relabel-consistency-transformers-2", self.jobs.names)
        eq = command(self.jobs.by_name("relabel-equivalence"))
        self.assertEqual(eq[eq.index("--vllm-setting") + 1], "1")
        self.assertEqual(eq[eq.index("--tf-setting") + 1], "2")
        self.assertEqual(sorted(self.jobs.names), sorted(
            ["relabel-consistency-prepare", "relabel-consistency-vllm-1", "relabel-consistency-transformers-2",
             "relabel-equivalence-prepare", "relabel-equivalence", "relabel-advance-c2"]))
        rec = self.record("checks")
        self.assertEqual(rec["exit"], 0)
        self.assertEqual(rec["submitted"], self.jobs.names)

    def test_declared_vram_is_rounded_up_to_a_tenth_and_qualifies(self):
        write_probes()
        self.assertEqual(self.run_step("checks"), 0)
        for name, key in (("relabel-consistency-vllm-1", ("vllm", 1)),
                          ("relabel-consistency-transformers-2", ("transformers", 2))):
            cmd = command(self.jobs.by_name(name))
            v = float(cmd[cmd.index("--declared-vram-gb") + 1])
            want = PEAKS[key] + layout.VRAM_MARGIN_GB
            self.assertGreaterEqual(v, want)
            self.assertLess(v - want, 0.1 + 1e-9)
            self.assertAlmostEqual(v * 10, round(v * 10), places=6)
            # a consistency result at that declared VRAM qualifies under select_production
            e, i = key
            cons = {"passed": True, "full_population": True, "limit": None,
                    "contract": c2.contract(run_dict(e, i, v)), "sets": {}}
            eq = {"passed": True, "contract": cons["contract"]} if e == "vllm" else None
            sel = checks.select_production([probe(e, i)], [cons], eq)["selected"]
            self.assertIsNotNone(sel)
            self.assertEqual(sel["contract"]["declared_vram_gb"], v)
        # the equivalence check runs at the vLLM consistency's declared VRAM (the C2 cap)
        cv = command(self.jobs.by_name("relabel-consistency-vllm-1"))
        eq = command(self.jobs.by_name("relabel-equivalence"))
        self.assertEqual(eq[eq.index("--declared-vram-gb") + 1], cv[cv.index("--declared-vram-gb") + 1])
        self.assertEqual(cv[cv.index("--declared-vram-gb") + 1], "18.6")

    def test_tenth_rounding_edges(self):
        for x in (15.43, 15.4, 15.3, 0.3, 24.900000000000002, 13.399999999, 20.0):
            r = advance.tenth_up(x)
            self.assertGreaterEqual(r, x)
            self.assertLess(r - x, 0.1 + 1e-9)
            self.assertEqual(float(repr(r)), r)
            self.assertEqual(round(r, 1), r)
        self.assertEqual(advance.tenth_up(15.4), 15.4)
        self.assertEqual(advance.tenth_up(15.43), 15.5)

    def test_lease_and_memory_flags(self):
        write_probes(transformers1=probe("transformers", 1, rss=12.0, cgroup=20.3))
        self.assertEqual(self.run_step("checks"), 0)
        gpu = ("relabel-consistency-vllm-1", "relabel-consistency-transformers-2", "relabel-equivalence")
        for name in gpu:
            a = self.jobs.by_name(name)
            v = float(command(a)[command(a).index("--declared-vram-gb") + 1])
            self.assertEqual(flag(a, "--needs"), "gpu:shared")
            self.assertEqual(flag(a, "--vram"), f"{math.ceil(v) + 1}G")
            self.assertEqual(flag(a, "--mem"), "26G")     # ceil(20.3 x 1.25)
            self.assertEqual(flag(a, "--stall"), "30m")
            self.assertEqual(flag(a, "--env"), "HF_HUB_OFFLINE=1")
            self.assertEqual(flag(a, "--cwd"), str(c2._here()))
            self.assertEqual(command(a)[:3], ["./python", "-m", "labeling"])
        self.assertEqual(flag(self.jobs.by_name("relabel-consistency-vllm-1"), "--vram"), "20G")   # 18.6 -> 19 + 1
        self.assertEqual(flag(self.jobs.by_name("relabel-consistency-vllm-1"), "--timeout"), "2h")
        self.assertEqual(flag(self.jobs.by_name("relabel-equivalence"), "--timeout"), "3h")
        for name, timeout, cmd in (("relabel-consistency-prepare", "1h", "consistency-prepare"),
                                   ("relabel-equivalence-prepare", "2h", "equivalence-prepare")):
            a = self.jobs.by_name(name)
            self.assertIsNone(flag(a, "--needs"))
            self.assertIsNone(flag(a, "--vram"))
            self.assertEqual((flag(a, "--mem"), flag(a, "--timeout")), ("16G", timeout))
            self.assertEqual(command(a), ["./python", "-m", "labeling", cmd])
        a = self.jobs.by_name("relabel-advance-c2")
        self.assertIsNone(flag(a, "--needs"))
        self.assertEqual((flag(a, "--mem"), flag(a, "--timeout")), ("4G", "30m"))
        self.assertEqual(command(a), ["./python", "-m", "labeling.advance", "c2"])
        self.assertEqual(self.record("checks")["decisions"]["mem_gb"], 26)

    def test_memory_floor_and_cap(self):
        self.assertEqual(advance.job_mem_gb(5.0), 12)
        self.assertEqual(advance.job_mem_gb(20.3), 26)
        self.assertEqual(advance.job_mem_gb(40.0), 36)
        self.assertEqual(advance.host_peak_gb(probe("vllm", 0, rss=9.0, cgroup=11.5)), 11.5)
        self.assertEqual(advance.host_peak_gb(probe("vllm", 0, rss=9.0, cgroup=None)), 9.0)

    def test_gpu_jobs_are_serialised(self):
        write_probes()
        self.assertEqual(self.run_step("checks"), 0)
        after = {n: set(filter(None, (flag(self.jobs.by_name(n), "--after") or "").split(","))) for n in self.jobs.names}
        self.assertEqual(after["relabel-consistency-prepare"], set())
        self.assertEqual(after["relabel-consistency-vllm-1"], {"relabel-consistency-prepare"})
        self.assertEqual(after["relabel-consistency-transformers-2"], {"relabel-consistency-vllm-1"})
        self.assertEqual(after["relabel-equivalence-prepare"], {"relabel-consistency-prepare"})
        self.assertEqual(after["relabel-equivalence"], {"relabel-consistency-transformers-2", "relabel-equivalence-prepare"})
        self.assertEqual(after["relabel-advance-c2"], {"relabel-equivalence"})
        for k, n in enumerate(self.jobs.names):        # every dependency is submitted before its dependant
            self.assertTrue(after[n] <= set(self.jobs.names[:k]), n)

    def test_missing_probe_exits_2_without_submission(self):
        write_probes(transformers1=None)
        self.assertEqual(self.run_step("checks"), 2)
        self.assertEqual(self.jobs.argvs, [])
        rec = self.record("checks")
        self.assertEqual(rec["exit"], 2)
        self.assertTrue(any("transformers-1" in r for r in rec["reasons"]), rec["reasons"])

    def test_zero_rate_probe_exits_2(self):
        write_probes(vllm2=probe("vllm", 2, rate=0.0))
        self.assertEqual(self.run_step("checks"), 2)
        self.assertEqual(self.jobs.argvs, [])

    def test_vllm_repeat_changed_stops_with_3(self):
        write_probes(vllm0=probe("vllm", 0, repeat=3))
        self.assertEqual(self.run_step("checks"), 3)
        self.assertEqual(self.jobs.argvs, [])
        rec = self.record("checks")
        self.assertEqual(rec["exit"], 3)
        self.assertTrue(any("vllm-0" in r for r in rec["reasons"]), rec["reasons"])

    def test_rerun_is_idempotent(self):
        write_probes()
        self.assertEqual(self.run_step("checks"), 0)
        n = len(self.jobs.argvs)
        before = self.record("checks")
        self.assertEqual(self.run_step("checks"), 0)
        self.assertEqual(len(self.jobs.argvs), n)
        self.assertEqual(self.record("checks")["decisions"], before["decisions"])

    def test_rerun_with_other_decisions_is_refused(self):
        write_probes()
        self.assertEqual(self.run_step("checks"), 0)
        n = len(self.jobs.argvs)
        before = advance.record_path("checks").read_bytes()
        write_probes(vllm2=probe("vllm", 2, rate=50.0))
        self.assertEqual(self.run_step("checks"), 4)
        self.assertEqual(len(self.jobs.argvs), n)
        self.assertEqual(advance.record_path("checks").read_bytes(), before)

    def test_rerun_refuses_when_recorded_jobs_are_gone(self):
        write_probes()
        self.assertEqual(self.run_step("checks"), 0)
        self.assertEqual(self.run_step("checks", jobs=Jobs()), 4)

    def test_failed_submission_resumes_without_resubmitting(self):
        write_probes()
        jobs = Jobs(fail_at=3)
        self.assertEqual(self.run_step("checks", jobs=jobs), 1)
        self.assertEqual(len(jobs.argvs), 3)
        self.assertEqual(self.run_step("checks", jobs=jobs), 0)
        self.assertEqual(len(jobs.names), 6)
        self.assertEqual(len(set(jobs.names)), 6)

    def test_interrupted_final_write_is_completed_on_rerun(self):
        write_probes()
        self.assertEqual(self.run_step("checks"), 0)
        rec = self.record("checks")
        c2.write_json(advance.record_path("checks"), rec | {"status": "submitting", "exit": None})
        self.assertEqual(self.run_step("checks"), 0)
        self.assertEqual(len(self.jobs.argvs), 6)
        self.assertEqual(self.record("checks")["status"], "submitted")
        self.assertEqual(self.record("checks")["submitted"], rec["submitted"])

    def test_interruption_inside_job_start_is_refused_on_rerun(self):
        write_probes()
        jobs = Jobs()

        def killed_after_start(argv):
            jobs.submit(argv)
            if len(jobs.argvs) == 2:
                raise KeyboardInterrupt      # the job exists, its name is not yet recorded as submitted
        with self.assertRaises(KeyboardInterrupt):
            advance.main(["checks"], submit=killed_after_start, exists=jobs.exists, out=self.out.append)
        self.assertEqual(self.run_step("checks", jobs=jobs), 4)
        self.assertEqual(len(jobs.argvs), 2)
        self.assertEqual(len(set(jobs.names)), 2)

    def test_dry_run_submits_and_writes_nothing(self):
        write_probes()
        before = self.files()
        self.assertEqual(self.run_step("checks", "--dry-run"), 0)
        self.assertEqual(self.jobs.argvs, [])
        self.assertEqual(self.files(), before)
        printed = json.loads(self.out[-1])
        self.assertEqual(len(printed["submit"]), 6)
        self.assertTrue(all(a[:2] == ["job", "start"] for a in printed["submit"]))

    def test_dry_run_with_missing_probes_exits_2_and_writes_nothing(self):
        before = self.files()
        self.assertEqual(self.run_step("checks", "--dry-run"), 2)
        self.assertEqual(self.files(), before)
        self.assertEqual(self.jobs.argvs, [])


class C2(Base):
    def setUp(self):
        super().setUp()
        write_probes()
        self.assertEqual(self.run_step("checks"), 0)
        self.checks_jobs = list(self.jobs.names)
        self.declared = {}
        for e in advance.ENGINES:
            name = next(n for n in self.checks_jobs if n.startswith(f"relabel-consistency-{e}-"))
            cmd = command(self.jobs.by_name(name))
            self.declared[e] = (int(cmd[cmd.index("--setting") + 1]), float(cmd[cmd.index("--declared-vram-gb") + 1]))

    def write_checks(self, vllm_passed: bool = True, eq_passed: bool = True, tf_passed: bool = True) -> None:
        for e, ok in (("vllm", vllm_passed), ("transformers", tf_passed)):
            i, v = self.declared[e]
            c2.write_json(advance.consistency_result_path(e, i),
                          {"passed": ok, "full_population": True, "limit": None, "sets": {},
                           "contract": c2.contract(run_dict(e, i, v))})
        i, v = self.declared["vllm"]
        c2.write_json(advance.equivalence_result_path(),
                      {"passed": eq_passed, "contract": c2.contract(run_dict("vllm", i, v))})

    @property
    def selected_contract(self) -> dict:
        return advance.load_selection()["result"]["selected"]["contract"]

    def c2_jobs(self) -> list[str]:
        return [n for n in self.jobs.names if n not in self.checks_jobs]

    def test_selects_vllm_and_queues_chunks_and_gate(self):
        self.write_checks()
        self.assertEqual(self.run_step("c2"), 0)
        i, v = self.declared["vllm"]
        self.assertEqual(self.inits, [["c2-init", "--run", "main", "--engine", "vllm", "--setting", str(i),
                                       "--declared-vram-gb", repr(v), "--group-shards", str(checks.GROUP_SHARDS)]])
        # 1000 distinct hours at 30 audio h per GPU-h = 33.3 GPU-h; 110-minute chunks -> 19, plus 2 spare
        names = self.c2_jobs()
        self.assertEqual(names, [f"relabel-c2-main-{k}" for k in range(1, 22)] + ["relabel-gate-main"])
        rd = layout.labels_dir(layout.TEACHER, "main")
        mem = f"{self.record('checks')['decisions']['mem_gb']}G"
        for k in range(1, 22):
            a = self.jobs.by_name(f"relabel-c2-main-{k}")
            self.assertEqual(flag(a, "--after"), None if k == 1 else f"relabel-c2-main-{k - 1}")
            self.assertEqual(flag(a, "--needs"), "gpu:shared")
            self.assertEqual(flag(a, "--vram"), f"{math.ceil(v) + 1}G")
            self.assertEqual(flag(a, "--mem"), mem)
            self.assertEqual(flag(a, "--stall"), "30m")
            self.assertEqual(flag(a, "--env"), "HF_HUB_OFFLINE=1")
            self.assertEqual(flag(a, "--timeout"), "2h30m")
            self.assertEqual(command(a), ["./python", "-m", "labeling.c2", "chunk", "--run-dir", str(rd),
                                          "--budget-minutes", "110"])
        g = self.jobs.by_name("relabel-gate-main")
        self.assertEqual(flag(g, "--after"), "relabel-c2-main-21")
        self.assertEqual((flag(g, "--mem"), flag(g, "--cpus"), flag(g, "--timeout")), ("36G", "12", "6h"))
        self.assertIsNone(flag(g, "--needs"))
        self.assertEqual(command(g), ["./python", "-m", "labeling", "gate", "--run", "main",
                                      "--consistency", str(advance.consistency_result_path("vllm", i)),
                                      "--equivalence", str(advance.equivalence_result_path())])
        rec = self.record("c2")
        self.assertEqual(rec["decisions"]["chunks"], 21)
        self.assertAlmostEqual(rec["decisions"]["projection"]["c2_gpu_hours"], 1000 / 30)

    def test_transformers_fallback_gate_has_no_equivalence(self):
        self.write_checks(eq_passed=False)
        self.assertEqual(self.run_step("c2"), 0)
        i, v = self.declared["transformers"]
        self.assertEqual(self.inits[0][3:8], ["--engine", "transformers", "--setting", str(i), "--declared-vram-gb"])
        # 1000 h at 8 h per GPU-h = 125 GPU-h -> 69 chunks, plus 2
        self.assertEqual(len([n for n in self.c2_jobs() if n.startswith("relabel-c2-main-")]), 71)
        g = command(self.jobs.by_name("relabel-gate-main"))
        self.assertNotIn("--equivalence", g)
        self.assertEqual(g[g.index("--consistency") + 1], str(advance.consistency_result_path("transformers", i)))
        self.assertEqual(flag(self.jobs.by_name("relabel-c2-main-1"), "--vram"), f"{math.ceil(v) + 1}G")

    def test_no_selection_exits_3(self):
        self.write_checks(vllm_passed=False, tf_passed=False)
        self.assertEqual(self.run_step("c2"), 3)
        self.assertEqual(self.c2_jobs(), [])
        self.assertEqual(self.inits, [])
        self.assertEqual(self.record("c2")["exit"], 3)

    def test_selection_matches_the_select_command(self):
        from labeling.__main__ import main as cli
        self.write_checks(eq_passed=False)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli(["select"])
        self.assertEqual(json.loads(buf.getvalue()), json.loads(json.dumps(advance.load_selection()["result"])))

    def test_existing_run_with_the_same_contract_is_accepted(self):
        self.write_checks()
        self.fake_init([])
        self.inits.clear()
        self.assertEqual(self.run_step("c2"), 0)
        self.assertEqual(self.inits, [])
        self.assertIn("relabel-gate-main", self.jobs.names)

    def test_existing_run_with_another_contract_is_refused(self):
        self.write_checks()
        i, v = self.declared["transformers"]
        c2.write_json(layout.labels_dir(layout.TEACHER, "main") / "run.json", run_dict("transformers", i, v))
        self.assertEqual(self.run_step("c2"), 4)
        self.assertEqual(self.c2_jobs(), [])
        self.assertEqual(self.inits, [])

    def test_rerun_is_idempotent(self):
        self.write_checks()
        self.assertEqual(self.run_step("c2"), 0)
        n = len(self.jobs.argvs)
        self.assertEqual(self.run_step("c2"), 0)
        self.assertEqual(len(self.jobs.argvs), n)
        self.assertEqual(len(self.inits), 1)

    def test_without_c1_manifest_exits_2(self):
        self.write_checks()
        (self.c1 / "manifest.json").unlink()
        self.assertEqual(self.run_step("c2"), 2)
        self.assertEqual(self.c2_jobs(), [])
        self.assertEqual(self.inits, [])

    def test_dry_run_submits_inits_and_writes_nothing(self):
        self.write_checks()
        before = self.files()
        self.assertEqual(self.run_step("c2", "--dry-run"), 0)
        self.assertEqual(self.c2_jobs(), [])
        self.assertEqual(self.inits, [])
        self.assertEqual(self.files(), before)
        printed = json.loads(self.out[-1])
        self.assertEqual(len(printed["submit"]), 22)


if __name__ == "__main__":
    unittest.main()
