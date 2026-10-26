"""Contract tests: the batch plan's cut engines (training/batchplan.py, "Cut engines"). The bend engine runs
finetune/bend-planner's binary and keeps its cut only when numpy's cut is byte-identical, so a plan directory,
plan.json included, is the same whichever engine built it, and a refused cut leaves no plan anywhere. The tests of
the real binary skip when it has not been built (finetune/bend-planner/README.md); the others use fake planners."""
from __future__ import annotations

import os
import shlex
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from labeling_fixtures import relabel_root
from training import batchplan as bp
from training_fixtures import LabeledWorld

P = 4                    # fixture pools: batches 0-3, 4-7, 8-11, 12-13
N_POOLS = 4
BUDGET = 160_000
SEED = 7
SYN_SEED, SHARE = 11, 0.1
KW = dict(pool_batches=P, budget=BUDGET, seed=SEED)


def plan_files(d: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(Path(d).iterdir())}


def entries(d: Path) -> list[str]:
    return sorted(p.name for p in Path(d).iterdir()) if Path(d).exists() else []


class Engines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.ctx = relabel_root(cls.root)
        cls.ctx.__enter__()
        cls.w = LabeledWorld(cls.root)
        from test_synthetic_mixing import make_pool
        cls.clips = make_pool(cls.root / "syn", n=40)
        cls.syn_kw = dict(KW, syn_seed=SYN_SEED, share=SHARE)
        cls.numpy_plan = bp.build(cls.w.pool, cls.root / "plans" / "numpy", N_POOLS, **KW)
        cls.numpy_syn = bp.build(cls.w.pool, cls.root / "plans" / "numpy-syn", N_POOLS, clips=cls.clips, **cls.syn_kw)
        _, cls.data = bp.compute(cls.w.pool, N_POOLS, **KW)
        cls.fakes = cls.root / "fakes"
        cls.fakes.mkdir()
        cls.n = 0

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)
        cls.tmp.cleanup()

    def scratch(self) -> Path:
        type(self).n += 1
        d = self.fakes / f"s{self.n}"
        d.mkdir()
        return d

    def fake(self, body: str) -> Path:
        """A planner script: `body` in /bin/sh with $1 .. $5 the documented arguments."""
        path = self.scratch() / "planner"
        path.write_text("#!/bin/sh\n" + body + "\n")
        path.chmod(0o755)
        return path

    def answering(self, order: bytes, offsets: bytes, record: Path | None = None) -> Path:
        """A fake planner that writes the given order and offsets (and records its arguments and inputs)."""
        d = self.scratch()
        (d / "order").write_bytes(order)
        (d / "offsets").write_bytes(offsets)
        q = shlex.quote
        rec = ""
        if record is not None:
            rec = (f'printf "%s\\n" "$@" > {q(str(record / "args"))}; cp "$1" {q(str(record / "samples"))}; '
                   f'cp "$2" {q(str(record / "pools"))}; ')
        return self.fake(f'{rec}cat {q(str(d / "order"))} > "$4"; cat {q(str(d / "offsets"))} > "$5"')

    def tmp_area(self) -> list[str]:
        return entries(self.root / "tmp")

    def assert_refused(self, planner: Path, out: Path, **kw):
        before = self.tmp_area()
        with self.assertRaises(bp.PlanError) as cm:
            bp.build(self.w.pool, out, N_POOLS, engine="bend", planner=planner, **KW, **kw)
        self.assertFalse(out.exists(), "a refused cut wrote a plan directory")
        self.assertEqual(entries(out.parent), [], "a refused cut left a temporary plan next to the output")
        self.assertEqual(self.tmp_area(), before, "a refused cut left temporary files")
        return cm.exception


class Default(Engines):
    def test_numpy_is_the_default_and_runs_no_planner(self):
        with mock.patch("subprocess.run", side_effect=AssertionError("the numpy engine ran a process")):
            default = bp.compute(self.w.pool, N_POOLS, **KW)
            self.assertEqual(default, bp.compute(self.w.pool, N_POOLS, engine="numpy", **KW))
            out = self.scratch() / "plan"
            self.assertEqual(plan_files(bp.build(self.w.pool, out, N_POOLS, **KW).dir), plan_files(self.numpy_plan.dir))

    def test_cli_default_and_bend_options(self):
        calls = []

        def build(pool, out, n_pools, **kw):
            calls.append(kw)
            return SimpleNamespace(dir=Path(out), identity=lambda: {}, meta={"stats": {}})
        out = str(self.root / "smoke" / "cli" / "plan")
        with mock.patch.object(bp, "_c1_and_config", return_value=(self.w.pool, "cid", {})), \
                mock.patch.object(bp, "build", side_effect=build), mock.patch("sys.stdout"):
            self.assertEqual(bp.main(["build", "--pools", "2", "--out", out]), 0)
            self.assertEqual(bp.main(["build", "--pools", "2", "--out", out, "--cut", "bend", "--planner", "/x/planner",
                                      "--threads", "3"]), 0)
            with mock.patch("sys.stderr"):
                for bad in (["--planner", "/x/planner"], ["--threads", "3"], ["--cut", "fast"]):
                    with self.assertRaises(SystemExit) as cm:
                        bp.main(["build", "--pools", "2", "--out", out, *bad])
                    self.assertNotIn(cm.exception.code, (0, None))
        self.assertEqual(calls[0].get("engine", "numpy"), "numpy")
        self.assertIsNone(calls[0].get("planner"))
        self.assertEqual((calls[1]["engine"], calls[1]["planner"], calls[1]["threads"]), ("bend", Path("/x/planner"), 3))
        self.assertEqual(len(calls), 2)

    def test_unknown_engine_is_refused(self):
        with self.assertRaises(bp.PlanError):
            bp.compute(self.w.pool, N_POOLS, engine="fast", **KW)


class FakePlanner(Engines):
    def test_a_planner_that_agrees_with_numpy_gives_the_same_plan_directory(self):
        record = self.scratch()
        planner = self.answering(self.data["order.u32"], self.data["offsets.u32"], record)
        before = self.tmp_area()
        out = self.scratch() / "plan"
        plan = bp.build(self.w.pool, out, N_POOLS, engine="bend", planner=planner, threads=3, **KW)
        self.assertEqual(plan_files(plan.dir), plan_files(self.numpy_plan.dir))     # plan.json included
        self.assertEqual(plan.sha256, self.numpy_plan.sha256)
        self.assertEqual(self.tmp_area(), before)
        args = (record / "args").read_text().splitlines()
        self.assertEqual(len(args), 7)
        self.assertEqual(args[2], str(BUDGET))
        self.assertEqual(args[5:], ["--threads", "3"])
        for a in (args[0], args[1], args[3], args[4]):     # temporary work in $RELABEL_ROOT's tmp area, not the plan's
            self.assertTrue(Path(a).is_relative_to(self.root / "tmp"), a)
        self.assertEqual((record / "samples").read_bytes(), self.data["samples.u32"])
        self.assertEqual((record / "pools").read_bytes(), self.data["pools.u32"])

    def test_a_cut_that_differs_from_numpy_is_refused_with_nothing_written(self):
        order, offsets = bytearray(self.data["order.u32"]), self.data["offsets.u32"]
        order[0:4], order[4:8] = order[4:8], order[0:4]
        moved = bytearray(offsets)
        moved[4:8] = (int.from_bytes(moved[4:8], "little") + 1).to_bytes(4, "little")
        for name, planner in (("order", self.answering(bytes(order), offsets)),
                              ("offsets", self.answering(self.data["order.u32"], bytes(moved)))):
            with self.subTest(name):
                e = self.assert_refused(planner, self.scratch() / "out" / "plan")
                self.assertIn("differs", str(e))

    def test_a_failing_planner_is_refused(self):
        planner = self.fake('echo "planner: bad pool offsets" >&2; exit 1')
        e = self.assert_refused(planner, self.scratch() / "out" / "plan")
        self.assertIn("1", str(e))
        self.assertIn("bad pool offsets", str(e))

    def test_wrong_output_length_or_no_output_is_refused(self):
        order, offsets = self.data["order.u32"], self.data["offsets.u32"]
        for name, planner in (("empty order", self.answering(b"", offsets)),
                              ("short offsets", self.answering(order, offsets[:-4])),
                              ("no output", self.fake("exit 0"))):
            with self.subTest(name):
                self.assert_refused(planner, self.scratch() / "out" / "plan")

    def test_a_missing_planner_says_how_to_build_it(self):
        e = self.assert_refused(self.root / "no-such-planner", self.scratch() / "out" / "plan")
        self.assertIn("bend planner.bend", str(e))

    def test_refused_without_the_data_mount(self):
        record = self.scratch()
        planner = self.answering(self.data["order.u32"], self.data["offsets.u32"], record)
        with mock.patch.dict(os.environ, {"DATA_MOUNT": str(self.root / "not-a-mount")}):
            with self.assertRaises(RuntimeError):
                bp.build(self.w.pool, self.scratch() / "plan", N_POOLS, engine="bend", planner=planner, **KW)
        self.assertFalse((record / "args").exists(), "the planner ran without the data mount")

    def test_a_relative_planner_path_is_not_looked_up_on_path(self):
        planner = self.answering(self.data["order.u32"], self.data["offsets.u32"])
        cwd = os.getcwd()
        os.chdir(planner.parent)
        try:
            plan = bp.build(self.w.pool, self.scratch() / "plan", N_POOLS, engine="bend", planner=Path("./planner"), **KW)
        finally:
            os.chdir(cwd)
        self.assertEqual(plan.sha256, self.numpy_plan.sha256)

    def test_a_synthetic_plan_is_refused_on_a_differing_entry_cut(self):
        """The planner gets the base plan's cut right and the version 2 entry cut wrong."""
        _, syn = bp.compute_synthetic(self.w.pool, N_POOLS, self.clips, **self.syn_kw)
        order = bytearray(syn["order.u32"])
        order[0:4], order[4:8] = order[4:8], order[0:4]
        d, q = self.scratch(), shlex.quote
        for name, b in (("order1", self.data["order.u32"]), ("offsets1", self.data["offsets.u32"]),
                        ("order2", bytes(order)), ("offsets2", syn["offsets.u32"])):
            (d / name).write_bytes(b)
        planner = self.fake(f'if [ -e {q(str(d / "called"))} ]; then k=2; else k=1; : > {q(str(d / "called"))}; fi; '
                            f'cat {q(str(d))}/order$k > "$4"; cat {q(str(d))}/offsets$k > "$5"; echo $k >> {q(str(d / "log"))}')
        before = self.tmp_area()
        out = self.scratch() / "out" / "plan"
        with self.assertRaises(bp.PlanError) as cm:
            bp.build(self.w.pool, out, N_POOLS, clips=self.clips, engine="bend", planner=planner, **self.syn_kw)
        self.assertIn("differs", str(cm.exception))
        self.assertEqual((d / "log").read_text().split(), ["1", "2"])     # refused at the entry cut, not the base
        self.assertFalse(out.exists())
        self.assertEqual(entries(out.parent), [])
        self.assertEqual(self.tmp_area(), before)

    def test_cli_exits_nonzero_and_writes_no_plan(self):
        _, data = bp.compute(self.w.pool, 1, config_id="cid")       # the CLI's plan: production P, budget and seed
        order = bytearray(data["order.u32"])
        order[0:4], order[4:8] = order[4:8], order[0:4]
        before = self.tmp_area()
        for name, planner in (("mismatch", self.answering(bytes(order), data["offsets.u32"])),
                              ("failure", self.fake("exit 3"))):
            with self.subTest(name):
                out = self.root / "smoke" / f"cli-{name}" / "plan"
                with mock.patch.object(bp, "_c1_and_config", return_value=(self.w.pool, "cid", {})), \
                        mock.patch("sys.stdout"), mock.patch("sys.stderr"):
                    with self.assertRaises(SystemExit) as cm:
                        bp.main(["build", "--pools", "1", "--out", str(out), "--cut", "bend", "--planner", str(planner)])
                self.assertNotIn(cm.exception.code, (0, None))
                self.assertFalse((out / "plan.json").exists())
                self.assertEqual(entries(out.parent), [])
                self.assertEqual(self.tmp_area(), before)


@unittest.skipUnless(bp.BEND_PLANNER.exists(), "the Bend planner is not built (finetune/bend-planner/README.md)")
class RealPlanner(Engines):
    def test_v1_plan_is_byte_identical_to_numpys(self):
        for threads in (None, 1, 4):
            with self.subTest(threads=threads):
                plan = bp.build(self.w.pool, self.scratch() / "plan", N_POOLS, engine="bend", threads=threads, **KW)
                self.assertEqual(plan_files(plan.dir), plan_files(self.numpy_plan.dir))

    def test_v2_plan_is_byte_identical_to_numpys(self):
        for threads in (None, 4):
            with self.subTest(threads=threads):
                plan = bp.build(self.w.pool, self.scratch() / "plan", N_POOLS, clips=self.clips, engine="bend",
                                threads=threads, **self.syn_kw)
                self.assertEqual(plan_files(plan.dir), plan_files(self.numpy_syn.dir))
                self.assertEqual(plan.version, bp.SYNTHETIC_VERSION)

    def test_leaves_no_temporary_files(self):
        before = self.tmp_area()
        bp.compute(self.w.pool, N_POOLS, engine="bend", **KW)
        self.assertEqual(self.tmp_area(), before)


if __name__ == "__main__":
    unittest.main()
