"""Contract tests of the recorded x2 benchmark session (gpu1bit/x2/bench_session.py), orchestrated by the
selection-session machinery of gpu1bit/rerun_timing.py in roster-file mode (plans/x2-default.md "Recorded
GPU benchmark", X5, Y4, Z2; A13 of plans/gpu-kernel.md for the historical test workloads; CPU):
  - roster: b1 to b6 exactly as X5, R-expanded (b3) first as the P-paired reference, the frozen pairs,
    the x2 patch, the gate run and its record, the candidate selection document (path and content hash);
    written once, a different roster for the same run is refused;
  - steps, dev phase: P-paired encoder and full scope over dev_24 in 6 chunks with all six arms
    resident, P-clips + P-memory per arm in roster order then reverse order (fresh processes), one
    run-scoped P-kernel step (int8_mma and int8_mma:x2 layers plus the M1 baselines). Historical phase:
    the same P-paired and P-clips protocols on M1's historical test workloads, under separate run
    names; refused until the dev results are written and bound to this roster (A13);
  - guard before every process: the session freeze (sources, and the pointer in one of its two Z2
    states), the selection in use equal to the candidate, the gate record passed under the candidate,
    and each fulldev arm's identity (b1 g1, b2 g3, b4 g4, b5 g5) verified in the gate run and equal to
    the gate record's;
  - binding: an artifact is accepted only if every variant's recorded roster id, model, complete export
    identity, selection hash and (fulldev arms) identity match; a swapped export, wrong roster id,
    another identity, another selection or another workload set is refused;
  - M1's arms b4 to b6 load M1's export (not MB1's derived layout), b1 and b2 MB1's 1-bit export, b3 H's
    derived layout;
  - P-kernel times each layer variant at its own part count (an int8_mma:x2 layer called with
    components 3 would time the x3 path);
  - historical workloads: M1's own rules reproduce M1's recorded workloads (the LibriSpeech test-clean
    clips of compare_v2 and the 24-utterance multi-corpus set of cuda_experiments);
  - dev report: labels verified, the frozen pairs with median paired energy and latency separately (Y4,
    plain statements where b4 is worse than b6, no decision), P-memory per arm; regenerated
    byte-identically after promotion, refused in a third pointer state (Z2); it marks the dev results
    written, which the historical phase requires;
  - acceptance (Codex x2 c1, finding 5): the reports and the historical authorization require every step's
    accepted sidecar from the orchestrator, bound to this roster, protocol and identities, with the output
    bytes it accepted (rerun_timing.reusable); a missing or rejected sidecar, changed output bytes or
    another binding are refused.

    gpu1bit/shared/python -m gpu1bit.tests.run --cpu gpu1bit.tests.test_x2_bench_session
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from gpu1bit.shared import env, fulldev, seldoc, selection
from gpu1bit.x2 import bench_roster as BR

VARIANTS = ["R-expanded", "MB1-B-hybrid-x2", "MB1-B-hybrid-x3", "M1-x2", "M1-x3", "M1-R-hybrid-x3"]


def _bs():
    from gpu1bit.x2 import bench_session
    return bench_session


class _Tmp(unittest.TestCase):
    """A temporary session root, bench root and gate record; the real selection documents."""

    def setUp(self):
        self.BS = _bs()
        self.tmp = Path(tempfile.mkdtemp(dir=str(env.ARTIFACTS_ROOT / "H")))
        self.saved = (self.BS.ROOT, self.BS.BENCH, os.environ.get("GPU1BIT_SELECTION"))
        self.addCleanup(self._restore)                   # also when a subclass's setUp skips or fails
        self.BS.ROOT, self.BS.BENCH = self.tmp / "sessions", self.tmp / "bench"
        self.run = f"test-x2bench-{os.getpid()}"
        self.gate_record = self.tmp / "gates-G.json"
        self.gate_ids = {"g1": "id-b1", "g3": "id-b2", "g4": "id-b4", "g5": "id-b5"}
        self.gate_record.write_text(json.dumps({"run": "G", "pass": True, "identities": self.gate_ids,
                                                "selection": {"sha256": env.sha256_file(seldoc.CANDIDATE)}}))

    def _restore(self):
        self.BS.ROOT, self.BS.BENCH, sel = self.saved
        if sel is None:
            os.environ.pop("GPU1BIT_SELECTION", None)
        else:
            os.environ["GPU1BIT_SELECTION"] = sel
        shutil.rmtree(self.tmp, ignore_errors=True)

    def roster(self):
        return self.BS.roster_doc(self.run, "G", gate_record=self.gate_record)


class Roster(_Tmp):
    def test_rows_pairs_and_binding_fields(self):
        r = self.roster()
        self.assertEqual(r["kind"], "x2-benchmark")
        self.assertEqual(r["variants"], VARIANTS)
        self.assertEqual({v: BR.VARIANT_TO_ID[v] for v in r["variants"]},
                         {"R-expanded": "b3", "MB1-B-hybrid-x2": "b1", "MB1-B-hybrid-x3": "b2", "M1-x2": "b4",
                          "M1-x3": "b5", "M1-R-hybrid-x3": "b6"})
        self.assertEqual(r["roster"], BR.BENCH_ROSTER)
        self.assertEqual([tuple(p[:2]) for p in r["pairs"]], [(a, b) for a, b, _ in BR.PAIRS])
        self.assertEqual(r["patches"], ["gpu1bit.x2.bench_patch"])
        self.assertEqual(r["kernel_impls"], ["int8_mma", "int8_mma:x2"])
        self.assertEqual((r["gate_run"], r["gate_record"]), ("G", str(self.gate_record)))
        self.assertEqual(r["selection"], {"path": str(seldoc.CANDIDATE), "sha256": env.sha256_file(seldoc.CANDIDATE)})
        self.assertNotIn("identity_run", r, "the actprec recognition-run guard is not used for this roster")

    def test_written_once(self):
        root = self.tmp / "r"
        a = self.BS.write_roster(self.roster(), root=root)
        self.assertEqual(self.BS.write_roster(self.roster(), root=root), a)
        other = {**self.roster(), "gate_run": "H"}
        with self.assertRaisesRegex(SystemExit, "different roster"):
            self.BS.write_roster(other, root=root)


class Steps(_Tmp):
    def test_dev_phase(self):
        st = self.BS.steps(self.roster(), "dev")
        paired = [s for s in st if s["kind"] == "paired"]
        clips = [s for s in st if s["kind"] == "clips"]
        kernel = [s for s in st if s["kind"] == "kernel"]
        self.assertEqual(len(st), 12 + 12 + 1)
        self.assertEqual([(s["scope"], s["range"]) for s in paired],
                         [(sc, r) for sc in ("encoder", "full") for r in ("0-3", "4-7", "8-11", "12-15", "16-19", "20-23")])
        for s in paired:
            self.assertEqual(s["cmd"][s["cmd"].index("--variants") + 1:s["cmd"].index("--utterances")], VARIANTS)
            self.assertIn("gpu1bit.x2.bench_patch", s["cmd"])
            self.assertEqual(s["workloads"], "dev")
            self.assertEqual(s["output"], self.BS.BENCH / self.run / f"paired-{s['scope']}-{s['range'].split('-')[0].zfill(2)}-"
                             f"{s['range'].split('-')[1].zfill(2)}.json")
        self.assertEqual([s["variant"] for s in clips], VARIANTS + VARIANTS[::-1])
        self.assertEqual([s["label"] for s in clips], ["clips1"] * 6 + ["clips2"] * 6)
        k = kernel[0]
        self.assertEqual(k["output"], self.BS.BENCH / self.run / "kernel.json")
        i = k["cmd"].index("--impl")
        self.assertEqual(k["cmd"][i + 1:i + 3], ["int8_mma", "int8_mma:x2"])
        self.assertIn("--baselines", k["cmd"])
        self.assertEqual(k["cmd"][k["cmd"].index("--run") + 1], self.run)

    def test_historical_phase_only_after_dev_results(self):
        """A13 and Codex x2 c1 finding 5: the historical phase needs the dev report's marker bound to this roster
        AND every dev step's accepted measurement (verified again, not trusted from the marker). The
        positive path is DevReport.test_report_pairs_labels_memory_and_marker."""
        r = self.roster()
        with self.assertRaisesRegex(SystemExit, "A13"):
            self.BS.steps(r, "historical-test")
        marker = self.BS.ROOT / self.run / "report-dev.json"    # what report_dev writes, bound to this roster
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"dev_results_written": False, "roster_sha256": self.BS.roster_sha(r)}))
        with self.assertRaisesRegex(SystemExit, "A13"):
            self.BS.steps(r, "historical-test")
        marker.write_text(json.dumps({"dev_results_written": True, "roster_sha256": self.BS.roster_sha(r)}))
        with self.assertRaisesRegex(SystemExit, "A13"):          # a marker without accepted measurements
            self.BS.steps(r, "historical-test")
        other = {**r, "gate_run": "H"}                  # a marker of another roster does not count
        with self.assertRaisesRegex(SystemExit, "A13"):
            self.BS.steps(other, "historical-test")


class Guard(_Tmp):
    def setUp(self):
        super().setUp()
        os.environ["GPU1BIT_SELECTION"] = str(seldoc.CANDIDATE)
        self.ids = {"MB1-B-hybrid-x2": "id-b1", "MB1-B-hybrid-x3": "id-b2", "M1-x2": "id-b4", "M1-x3": "id-b5"}
        self.patches = contextlib.ExitStack()
        self.addCleanup(self.patches.close)
        fz = {"freeze": "f", "files": 9, "mismatched": [], "added": [], "ok": True, "pointer": "x3"}
        self.patch(selection, "verify_freeze", lambda run: dict(fz))
        freeze = self.tmp / "freeze.sha256"                       # the stubbed freeze exists
        freeze.write_text("")
        self.patch(selection, "freeze_path", lambda run: freeze)
        self.patch(fulldev, "check_identity", lambda run, arm, subset: {"hash": self.ids[arm]})

    def patch(self, obj, name, value):
        old = getattr(obj, name)
        setattr(obj, name, value)
        self.patches.callback(setattr, obj, name, old)

    def test_accepts_and_reports_identities_and_labels(self):
        g = self.BS.guard(self.roster())
        self.assertTrue(g["ok"])
        self.assertEqual(g["identities"], {**self.ids, "R-expanded": None, "M1-R-hybrid-x3": None})
        self.assertEqual(g["pointer"], "x3")
        for v in VARIANTS:
            bid = BR.VARIANT_TO_ID[v]
            self.assertEqual(g["labels"][v], {"roster_id": bid, "model": BR.ROW[bid]["model"],
                                              "export": fulldev.export_identity(BR.export_dir(bid))})
        self.assertEqual(g["labels"]["M1-R-hybrid-x3"]["export"], fulldev.export_identity(env.M1_EXPORT))

    def test_refusals(self):
        cases = {
            "gate failed": lambda: self.gate_record.write_text(json.dumps({"pass": False, "identities": self.gate_ids,
                                                                           "selection": {"sha256": env.sha256_file(seldoc.CANDIDATE)}})),
            "gate under another selection": lambda: self.gate_record.write_text(json.dumps(
                {"pass": True, "identities": self.gate_ids, "selection": {"sha256": "0" * 64}})),
            "identity differs from the gate record": lambda: self.ids.update({"M1-x2": "id-other"}),
            "selection in use is not the candidate": lambda: os.environ.update(GPU1BIT_SELECTION=str(self.other_selection())),
        }
        for name, edit in cases.items():
            with self.subTest(case=name):
                saved = (self.gate_record.read_text(), dict(self.ids), os.environ["GPU1BIT_SELECTION"])
                edit()
                try:
                    with self.assertRaises(SystemExit):
                        self.BS.guard(self.roster())
                finally:
                    self.gate_record.write_text(saved[0])
                    self.ids.clear()
                    self.ids.update(saved[1])
                    os.environ["GPU1BIT_SELECTION"] = saved[2]

    def other_selection(self) -> Path:
        """A selection document other than the candidate (the pointer may already be the promoted candidate)."""
        p = self.tmp / "other-selection.json"
        p.write_text(json.dumps({"about": "not the candidate", "kernel": "int8_mma", "runtime": "hybrid"}))
        return p

    def test_freeze_failure_propagates(self):
        def bad(run):
            raise SystemExit("freeze verification FAILED")
        self.patch(selection, "verify_freeze", bad)
        with self.assertRaisesRegex(SystemExit, "freeze"):
            self.BS.guard(self.roster())


class Binding(Guard):
    def doc(self, step, roster, expected):
        infos = {v: {**expected["labels"][v], "selection_sha256": roster["selection"]["sha256"],
                     **({"identity": expected["identities"][v]} if expected["identities"][v] else {})} for v in VARIANTS}
        if step["kind"] == "paired":
            return {"protocol": "P-paired", "scope": step["scope"], "variants": VARIANTS, "variants_info": infos,
                    "args": {"run": step["run"], "utterances": step["range"], "workloads": step["workloads"]}}
        return {"protocol": "P-clips+P-memory", "variant": step["variant"], "variant_info": infos[step["variant"]],
                "args": {"run": step["run"], "workloads": step["workloads"]}}

    def test_accept_and_refuse(self):
        from gpu1bit import rerun_timing as RT
        roster = {**self.roster(), "phase": "dev"}
        expected = RT.make_binding(roster, self.BS.guard(roster))
        st = self.BS.steps(roster, "dev")
        paired, clip = st[0], next(s for s in st if s["kind"] == "clips" and s["variant"] == "M1-x2")
        for step in (paired, clip):
            self.assertTrue(RT.artifact_bound(step, self.doc(step, roster, expected), roster, expected), step["name"])
        edits = {
            "swapped export": lambda i: i.update(export=fulldev.export_identity(env.DERIVED_EXPORT)),
            "wrong roster id": lambda i: i.update(roster_id="b5"),
            "another identity": lambda i: i.update(identity="id-x"),
            "another selection": lambda i: i.update(selection_sha256="0" * 64),
        }
        for name, edit in edits.items():
            with self.subTest(case=name):
                d = self.doc(clip, roster, expected)
                edit(d["variant_info"])
                self.assertFalse(RT.artifact_bound(clip, d, roster, expected))
                d = self.doc(paired, roster, expected)
                edit(d["variants_info"]["M1-x2"])
                self.assertFalse(RT.artifact_bound(paired, d, roster, expected))
        d = self.doc(clip, roster, expected)
        d["args"]["workloads"] = "historical-test"
        self.assertFalse(RT.artifact_bound(clip, d, roster, expected), "another workload set")

    def test_kernel_step_complete_and_bound(self):
        from gpu1bit import rerun_timing as RT
        roster = {**self.roster(), "phase": "dev"}
        expected = RT.make_binding(roster, self.BS.guard(roster))
        k = next(s for s in self.BS.steps(roster, "dev") if s["kind"] == "kernel")
        k["output"].parent.mkdir(parents=True, exist_ok=True)
        doc = {"protocol": "P-kernel", "args": {"run": self.run, "impl": ["int8_mma", "int8_mma:x2"], "baselines": True,
                                                "M": [38, 126, 368], "ops": ["qkv", "out"]},
               "rows": [{}] * 6, "power_limit_w_end": 400.0}
        k["output"].write_text(json.dumps(doc))
        self.assertTrue(RT.output_complete(k))
        self.assertTrue(RT.artifact_bound(k, doc, roster, expected))
        doc["rows"] = [{}] * 5
        k["output"].write_text(json.dumps(doc))
        self.assertFalse(RT.output_complete(k))
        self.assertFalse(RT.artifact_bound(k, {**doc, "args": {**doc["args"], "impl": ["int8_mma"]}}, roster, expected))

    def test_child_environment(self):
        from gpu1bit import rerun_timing as RT
        e = RT.child_env(self.roster())
        self.assertEqual(e["GPU1BIT_SELECTION"], str(seldoc.CANDIDATE))
        self.assertNotIn("GPU1BIT_ACTPREC_RUN", e)


class _Stop(Exception):
    pass


class Exports(unittest.TestCase):
    """b4-b6 load M1's export, b1/b2 MB1's 1-bit export, b3 H's derived builder (no model is built)."""

    def test_loader_paths(self):
        from gpu1bit import runtime, runtime_m1
        from gpu1bit.x2 import bench_patch as BP
        import inference.runtime as ir
        seen = {}
        saved = (runtime_m1.build_stack, ir.load_packed, runtime.load_binary_packed, BP._previous,
                 os.environ.get("GPU1BIT_SELECTION"))

        def rec(name):
            def f(path, *a, **k):
                seen.setdefault(name, []).append(Path(path).resolve())
                raise _Stop
            return f

        @contextlib.contextmanager
        def previous(name, *, max_graphs=4):
            seen.setdefault("previous", []).append(name)
            raise _Stop
            yield
        runtime_m1.build_stack, ir.load_packed = rec("build_stack"), rec("load_packed")
        runtime.load_binary_packed, BP._previous = rec("load_binary_packed"), previous
        os.environ["GPU1BIT_SELECTION"] = str(seldoc.CANDIDATE)
        got = {}
        try:
            for v in VARIANTS:
                seen.clear()
                with self.assertRaises(_Stop):
                    with BP.variant_model(v):
                        pass
                got[v] = {k: list(x) for k, x in seen.items()}
        finally:
            runtime_m1.build_stack, ir.load_packed, runtime.load_binary_packed, BP._previous, sel = saved
            if sel is None:
                os.environ.pop("GPU1BIT_SELECTION", None)
            else:
                os.environ["GPU1BIT_SELECTION"] = sel
        m1, mb1 = env.M1_EXPORT.resolve(), env.MB1_EXPORT.resolve()
        self.assertEqual(got["M1-x2"], {"build_stack": [m1]})
        self.assertEqual(got["M1-x3"], {"build_stack": [m1]})
        self.assertEqual(got["M1-R-hybrid-x3"], {"load_packed": [m1]})
        self.assertEqual(got["MB1-B-hybrid-x2"], {"load_binary_packed": [mb1]})
        self.assertEqual(got["MB1-B-hybrid-x3"], {"load_binary_packed": [mb1]})
        self.assertEqual(got["R-expanded"], {"previous": ["R-expanded"]})
        self.assertNotEqual(m1, env.DERIVED_EXPORT.resolve())


class KernelParts(unittest.TestCase):
    def test_candidate_op_uses_the_variant_part_count(self):
        from gpu1bit.shared import bench, provider
        calls = []

        class Layer:
            def project(self, *a, **k):
                pass

        class Impl:
            def prepare(self, codes, scale, bias, k, device):
                return Layer()
        saved = provider.BinaryProjection.__call__

        def spy(self, x, mod, conv=False, components=3, *a, **k):
            calls.append(components)
        provider.BinaryProjection.__call__ = spy
        try:
            w = {"codes": None, "scale": None, "bias": None}
            for name, parts in (("int8_mma", 3), ("int8_mma:x2", 2), ("int8_mma:x1", 1), ("int8_mma:x2:b01", 2)):
                calls.clear()
                fn, _, _ = bench.candidate_op(Impl(), "out", w, None, components=bench.part_count(name))
                fn(None)
                self.assertEqual(calls, [parts], name)
        finally:
            provider.BinaryProjection.__call__ = saved


class Historical(unittest.TestCase):
    def test_m1_rules_reproduce_m1_recorded_workloads(self):
        from gpu1bit.x2 import historical as HI
        sel = HI.select()
        self.assertEqual([r["id"] for r in sel["clips"]], list(HI.RECORDED_CLIPS))
        self.assertEqual([r["id"] for r in sel["utterances"]], list(HI.RECORDED_24))
        self.assertEqual(len(sel["utterances"]), 24)
        self.assertEqual([(r["corpus"], r["target_seconds"]) for r in sel["utterances"]][:3],
                         [("librispeech_clean", 3), ("librispeech_clean", 10), ("librispeech_clean", 30)])

    def test_records_refused_before_dev_results(self):
        from gpu1bit.x2 import historical as HI
        with self.assertRaisesRegex(SystemExit, "A13"):
            HI.records(f"no-such-run-{os.getpid()}")


def _paired_rows(scope: str) -> list[dict]:
    e = {"R-expanded": 1.10, "MB1-B-hybrid-x2": 0.90, "MB1-B-hybrid-x3": 1.00, "M1-x2": 1.05, "M1-x3": 1.15,
         "M1-R-hybrid-x3": 1.00}
    lat = {"R-expanded": 4.6, "MB1-B-hybrid-x2": 4.0, "MB1-B-hybrid-x3": 4.1, "M1-x2": 4.3, "M1-x3": 4.5,
           "M1-R-hybrid-x3": 4.2}
    return [{"index": i, "id": f"u{i}", "target_seconds": (3, 10, 30)[i // 8],
             "summary": {v: {"joules_per_iteration": e[v] * (1 + 0.001 * (i % 3)), "median_ms": lat[v], "average_watts": 300.0}
                         for v in VARIANTS},
             "checks": {v: {"finite": True, "lengths_equal": True, "rel_frobenius_valid": 0.0} for v in VARIANTS}}
            for i in range(24)]


class DevReport(_Tmp):
    """Report regeneration across promotion (Z2) with temporary selection documents and a real freeze."""

    def setUp(self):
        if seldoc.SELECTED.read_bytes() == seldoc.CANDIDATE.read_bytes():
            self.skipTest("promoted already: the x3 pointer bytes this test promotes are not available")
        super().setUp()
        self.sel_tmp = self.tmp / "sel"
        self.sel_tmp.mkdir()
        self.ptr, self.cand = self.sel_tmp / "selected.json", self.sel_tmp / "selected-x2.json"
        shutil.copy(seldoc.SELECTED, self.ptr)
        shutil.copy(seldoc.CANDIDATE, self.cand)
        self.saved_sel = (seldoc.SELECTED, seldoc.CANDIDATE)
        seldoc.SELECTED, seldoc.CANDIDATE = self.ptr, self.cand
        self.addCleanup(self._restore_dev)
        os.environ["GPU1BIT_SELECTION"] = str(self.cand)
        self.ids = {"MB1-B-hybrid-x2": "id-b1", "MB1-B-hybrid-x3": "id-b2", "M1-x2": "id-b4", "M1-x3": "id-b5"}
        self.gate_record.write_text(json.dumps({"run": "G", "pass": True, "identities": self.gate_ids,
                                                "selection": {"sha256": env.sha256_file(self.cand)}}))
        self.old_check = fulldev.check_identity
        fulldev.check_identity = lambda run, arm, subset: {"hash": self.ids[arm]}
        self.addCleanup(setattr, fulldev, "check_identity", self.old_check)
        roster = self.BS.roster_doc(self.run, "G", gate_record=self.gate_record)
        self.roster = self.BS.write_roster(roster)
        selection.write_freeze(self.run)
        g = self.BS.guard(self.roster)
        infos = {v: {**g["labels"][v], "selection_sha256": self.roster["selection"]["sha256"],
                     **({"identity": g["identities"][v]} if g["identities"][v] else {})} for v in VARIANTS}
        d = self.BS.BENCH / self.run
        d.mkdir(parents=True)
        for scope in ("encoder", "full"):
            rows = _paired_rows(scope)
            for c in range(6):
                (d / f"paired-{scope}-{4 * c:02d}-{4 * c + 3:02d}.json").write_text(json.dumps(
                    {"protocol": "P-paired", "scope": scope, "variants": VARIANTS, "variants_info": infos,
                     "args": {"run": self.run, "utterances": f"{4 * c}-{4 * c + 3}", "workloads": "dev"},
                     "rows": rows[4 * c:4 * c + 4]}))
        for label in ("clips1", "clips2"):
            dd = self.BS.BENCH / f"{self.run}-{label}"
            dd.mkdir(parents=True)
            for v in VARIANTS:
                peak = 1000 if v.startswith("MB1") else 2000
                (dd / f"clips-{v}.json").write_text(json.dumps(
                    {"protocol": "P-clips+P-memory", "variant": v, "variant_info": infos[v],
                     "args": {"run": f"{self.run}-{label}", "workloads": "dev"}, "memory_samples_10ms": [],
                     "workloads": [{"record": {"id": c, "M": 1}, "hypothesis": "t",
                                    "summary": {"median_ms": 1.0, "joules_per_iteration": 1.0, "average_watts": 1.0},
                                    "process_sampled_peak_including_first_bytes": peak + (label == "clips2")}
                                   for c in ("c3", "c10", "c30")]}))
        (d / "kernel.json").write_text(json.dumps({"protocol": "P-kernel", "args": {"run": self.run, "impl": ["int8_mma", "int8_mma:x2"],
                                                                                    "baselines": True, "M": [38], "ops": ["out"]},
                                                   "rows": [{"op": "out", "M": 38, "variants": {"int8_mma": {"warm_us": 2.0, "cold_us": 3.0},
                                                                                                "int8_mma:x2": {"warm_us": 1.5, "cold_us": 2.5}}}],
                                                   "power_limit_w_end": 400.0}))
        self.accept("dev")

    def accept(self, phase):
        """Accepted sidecars for every step of `phase`, as rerun_timing's orchestrator writes them."""
        from gpu1bit import rerun_timing as RT
        rp = {**self.roster, "phase": phase}
        bind = RT.make_binding(rp, self.BS.guard(rp))
        host = self.BS.ROOT / self.run / "host"
        host.mkdir(parents=True, exist_ok=True)
        for st in self.BS.steps(rp, phase):
            rec = {"step": st["name"], "run": self.run, "accepted": True,
                   "attempts": [{"attempt": 0, "rc": 0, "accepted": True, "binding": RT.step_binding(st, bind),
                                 "output_sha256": env.sha256_file(st["output"])}]}
            (host / f"{st['name']}.json").write_text(json.dumps(rec) + "\n")

    def _restore_dev(self):
        seldoc.SELECTED, seldoc.CANDIDATE = self.saved_sel
        for p in (selection.freeze_path(self.run), selection.pointer_pin_path(self.run)):
            if p.exists():
                p.unlink()
        shutil.rmtree(env.ARTIFACTS / "selection" / self.run, ignore_errors=True)

    def test_report_pairs_labels_memory_and_marker(self):
        rep = self.BS.report_dev(self.run)
        self.assertEqual(set(rep["full"]["pairs"]), {"b1-b2", "b4-b5", "b1-b6", "b4-b6", "b1-b4", "b1-b3"})
        self.assertTrue(any("b4" in s and "more energy" in s for s in rep["full"]["statements"]))
        self.assertNotIn("decision", rep)
        self.assertEqual(rep["p_memory"]["MB1-B-hybrid-x2"]["c3"], 1001)              # max over the two orders
        self.assertEqual(rep["labels"]["M1-x2"]["roster_id"], "b4")
        self.assertEqual(rep["kernel"]["int8_mma:x2"]["out"]["38"]["warm_us"], 1.5)
        self.assertTrue(rep["dev_results_written"])
        self.assertEqual(len(self.BS.steps(self.roster, "historical-test")), 24, "A13 marker accepted")

    def test_regenerated_after_promotion_and_refused_in_a_third_state(self):
        self.BS.report_dev(self.run)
        path = self.BS.ROOT / self.run / "report-dev.json"
        before = path.read_bytes()
        seldoc.promote()
        os.environ.pop("GPU1BIT_SELECTION", None)                     # the promoted pointer is the candidate
        self.BS.report_dev(self.run)
        self.assertEqual(path.read_bytes(), before, "promotion rewrites no recorded evidence")
        self.ptr.write_text("{}")
        with self.assertRaisesRegex(SystemExit, "third state"):
            self.BS.report_dev(self.run)
        self.assertEqual(path.read_bytes(), before)

    def test_unaccepted_measurements_are_refused(self):
        """Finding 5: the report and the historical authorization require each dev step's accepted sidecar,
        bound to this roster, protocol and identities, with the output bytes it accepted."""
        self.BS.report_dev(self.run)                    # accepted: written; the historical phase is authorized
        self.assertEqual(len(self.BS.steps(self.roster, "historical-test")), 24)
        host = self.BS.ROOT / self.run / "host"
        clip = self.BS.BENCH / f"{self.run}-clips1" / "clips-M1-x2.json"

        def missing():
            (host / "kernel.json").unlink()

        def rejected():
            p = host / "paired-full-0-3.json"
            d = json.loads(p.read_text())
            d["accepted"] = False
            d["attempts"][-1]["accepted"] = False
            p.write_text(json.dumps(d))

        def bytes_changed():
            clip.write_text(clip.read_text() + "\n")

        def other_binding():
            p = host / "clips2-M1-R-hybrid-x3.json"
            d = json.loads(p.read_text())
            d["attempts"][-1]["binding"]["roster_sha256"] = "0" * 64
            p.write_text(json.dumps(d))
        for name, edit in {"no sidecar": missing, "not accepted": rejected, "output changed after acceptance": bytes_changed,
                           "accepted under another binding": other_binding}.items():
            with self.subTest(case=name):
                saved = {p: p.read_bytes() for p in list(host.glob("*.json")) + [clip]}
                edit()
                try:
                    with self.assertRaisesRegex(SystemExit, "accepted"):
                        self.BS.report_dev(self.run)
                    with self.assertRaisesRegex(SystemExit, "A13"):
                        self.BS.steps(self.roster, "historical-test")
                finally:
                    for p, b in saved.items():
                        p.write_bytes(b)

    def test_swapped_export_in_an_artifact_is_refused(self):
        p = self.BS.BENCH / self.run / "paired-full-08-11.json"
        d = json.loads(p.read_text())
        d["variants_info"]["M1-R-hybrid-x3"]["export"] = fulldev.export_identity(env.DERIVED_EXPORT)
        p.write_text(json.dumps(d))
        with self.assertRaisesRegex(SystemExit, "b6"):
            self.BS.report_dev(self.run)
        self.assertFalse((self.BS.ROOT / self.run / "report-dev.json").exists())


if __name__ == "__main__":
    unittest.main()
