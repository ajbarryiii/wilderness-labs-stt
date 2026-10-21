"""Contract tests for the Codex code review c1 corrections (CPU only):
  finding 2  gate records bound to fingerprints: foreign records refused, verdict recomputed (a stale
             passing verdict.json is ignored), a source change after the gate fails the verdict;
  finding 3  the A14 winner rule against a common reference with the paired candidate-vs-candidate
             tie test (the sel-20261020 scenario, a 0.9% head-to-head tie, a 15/24-vs-9/24 tie);
  GPU c2     the anchored tie set for three or more candidates (third-ranked fastest tie member,
             nontransitive ties, name and latency tie-breaks); the export fingerprint covers the
             complete export (a metadata-only change invalidates gate records);
  finding 4  A12 encoder criteria over the union of the dev sample and the timing workloads;
  finding 5  decoder-side reclassification conserves every category total and keeps the raw audit;
  plus the session freeze (a changed file and an added file are both refused).
    gpu1bit/shared/python -m gpu1bit.tests.run --cpu gpu1bit.tests.test_selection_rules
"""
from __future__ import annotations

import json
import os
import shutil
import unittest
from pathlib import Path

from gpu1bit.shared import env, fingerprint as F, gate, selection


def _rows(energy: dict[str, list[float]], latency: dict[str, list[float]] | None = None) -> list[dict]:
    n = len(next(iter(energy.values())))
    latency = latency or {k: [1.0] * n for k in energy}
    return [{"index": i, "target_seconds": (3, 10, 30)[i // 8],
             "summary": {v: {"joules_per_iteration": energy[v][i], "median_ms": latency[v][i], "average_watts": 300.0}
                         for v in energy}} for i in range(n)]


class WinnerRule(unittest.TestCase):
    """A14 with the anchored tie set (Codex GPU code review c2; CONTRACT CHANGE, see the selection.py
    docstring): A* = lowest (median full-scope energy change vs R-expanded, name); S = A* plus every
    eligible candidate whose head-to-head energy with A* is an A14 tie (|median| < 1%, or neither lower
    on >= 16 of 24); within S the lowest median full-transcription latency vs R-expanded wins, then the
    lower head-to-head latency, then the name. The previous implementation compared only the top two by
    energy; the assertions on its "top_two" / "a_lower_count" fields are replaced by the anchor and
    tie-set fields. The two-candidate outcomes of the old tests are unchanged."""

    def test_no_tie_anchor_wins_by_energy(self):
        # sel-20261020-like: a is -3.99% head to head on 23/24.
        ref = [1.0] * 24
        b = [0.92] * 24
        a = [0.92 * 0.9601] * 23 + [0.93]
        rule = selection.winner_rule(_rows({"R-expanded": ref, "a": a, "b": b}), ["a", "b"])
        self.assertEqual(rule["ranking_vs_reference"][0][0], "a")
        self.assertEqual(rule["anchor"], "a")
        self.assertEqual(rule["tie_set"], ["a"])
        self.assertFalse(rule["head_to_head"]["b"]["tie"])
        self.assertEqual(rule["head_to_head"]["b"]["anchor_lower"], 23)
        self.assertEqual(rule["winner"], "a")
        self.assertTrue(rule["decided_by"].startswith("energy"))

    def test_paired_difference_under_one_percent_is_a_tie_decided_by_latency(self):
        ref = [1.0] * 24
        a = [0.910] * 24
        b = [0.918] * 24                                    # a is 0.87% lower head to head
        lat = {"R-expanded": [1.0] * 24, "a": [1.05] * 24, "b": [1.00] * 24}
        rule = selection.winner_rule(_rows({"R-expanded": ref, "a": a, "b": b}, lat), ["a", "b"])
        self.assertTrue(rule["head_to_head"]["b"]["tie"])
        self.assertEqual(rule["tie_set"], ["a", "b"])
        self.assertEqual(rule["winner"], "b")
        self.assertTrue(rule["decided_by"].startswith("latency"))

    def test_neither_lower_on_16_of_24_is_a_tie(self):
        ref = [1.0] * 24
        b = [0.90] * 24
        a = [0.85] * 15 + [0.95] * 9                       # median -5.6% but only 15/24 lower; b lower 9/24
        lat = {"R-expanded": [1.0] * 24, "a": [0.99] * 24, "b": [1.0] * 24}
        rule = selection.winner_rule(_rows({"R-expanded": ref, "a": a, "b": b}, lat), ["a", "b"])
        h = rule["head_to_head"]["b"]
        self.assertEqual((h["anchor_lower"], h["candidate_lower"]), (15, 9))
        self.assertTrue(h["tie"])
        self.assertEqual(rule["winner"], "a")

    def test_ranking_is_not_reversed_by_baselines(self):
        ref = [1.0] * 24
        rule = selection.winner_rule(_rows({"R-expanded": ref, "c": [0.95] * 24, "d": [0.90] * 24}), ["c", "d"])
        self.assertEqual(rule["winner"], "d")

    # --- three or more candidates -----------------------------------------------------------------

    def test_third_ranked_fastest_tie_member_wins(self):
        """The sel-20261020b pattern: A* ties with the 2nd AND the 3rd by energy; the 3rd is fastest."""
        ref = [1.0] * 24
        e = {"R-expanded": ref, "a": [0.880] * 24, "b": [0.885] * 24,          # b: A* -0.56% -> tie
             "c": [0.87] * 9 + [0.89] * 15,                                    # c: A* lower on 15, c on 9 -> tie
             "d": [0.92] * 24}                                                 # d: A* -4.3% on 24/24 -> no tie
        lat = {"R-expanded": [1.0] * 24, "a": [0.97] * 24, "b": [0.95] * 24, "c": [0.92] * 24, "d": [0.80] * 24}
        rule = selection.winner_rule(_rows(e, lat), ["a", "b", "c", "d"])
        self.assertEqual(rule["anchor"], "a")
        self.assertEqual(rule["tie_set"], ["a", "b", "c"])     # ranking order; d excluded although fastest
        self.assertEqual(rule["winner"], "c")
        self.assertTrue(rule["decided_by"].startswith("latency"))

    def test_nontransitive_ties_are_anchored_not_chained(self):
        """a ~ b and b ~ c, but a and c are not tied: c is not in S even though it is the fastest."""
        ref = [1.0] * 24
        e = {"R-expanded": ref, "a": [0.880] * 24, "b": [0.886] * 24, "c": [0.893] * 24}   # a-b 0.68%, b-c 0.78%, a-c 1.46%
        lat = {"R-expanded": [1.0] * 24, "a": [0.99] * 24, "b": [0.97] * 24, "c": [0.90] * 24}
        rows = _rows(e, lat)
        self.assertLess(abs(selection.paired(rows, "b", "c", "joules_per_iteration")["median_pct"]), 1.0)
        rule = selection.winner_rule(rows, ["a", "b", "c"])
        self.assertEqual(rule["tie_set"], ["a", "b"])
        self.assertFalse(rule["head_to_head"]["c"]["tie"])
        self.assertEqual(rule["winner"], "b")

    def test_nontransitive_tie_member_beats_another_head_to_head(self):
        """b and c both tie with A* (neither side lower on 16), while c beats b head to head by energy
        (-3.3%, 24/24). The rule still decides within S by latency vs R-expanded (b), as frozen; the
        pairwise relations inside S are recorded."""
        ref = [1.0] * 24
        e = {"R-expanded": ref, "a": [0.880] * 24, "b": [0.92] * 13 + [0.87] * 11, "c": [0.89] * 13 + [0.86] * 11}
        lat = {"R-expanded": [1.0] * 24, "a": [0.99] * 24, "b": [0.93] * 24, "c": [0.96] * 24}
        rows = _rows(e, lat)
        rule = selection.winner_rule(rows, ["a", "b", "c"])
        self.assertEqual(rule["anchor"], "a")
        self.assertEqual(rule["tie_set"], ["a", "c", "b"])
        self.assertEqual(rule["winner"], "b")
        cb = rule["within_tie_set_energy"]["c vs b"]
        self.assertLess(cb["median_pct"], -1.0)
        self.assertEqual(cb["lower"], 24)

    def test_anchor_by_name_on_equal_energy(self):
        ref = [1.0] * 24
        rule = selection.winner_rule(_rows({"R-expanded": ref, "z": [0.9] * 24, "m": [0.9] * 24}), ["z", "m"])
        self.assertEqual(rule["anchor"], "m")
        self.assertEqual(rule["tie_set"], ["m", "z"])

    def test_latency_tie_then_head_to_head_then_name(self):
        ref = [1.0] * 24
        e = {"R-expanded": ref, "a": [0.900] * 24, "b": [0.903] * 24, "c": [0.905] * 24}   # all tie with a
        lat = {"R-expanded": [1.0] * 24, "a": [0.99] * 24,
               "b": [0.94] * 11 + [0.95] * 2 + [0.96] * 11,      # median latency change vs R-expanded: -5%
               "c": [0.93] * 11 + [0.95] * 2 + [0.95] * 11}      # also exactly -5%; c lower than b on 22/24
        rows = _rows(e, lat)
        self.assertEqual(selection.paired(rows, "b", "R-expanded", "median_ms")["median_pct"],
                         selection.paired(rows, "c", "R-expanded", "median_ms")["median_pct"])
        self.assertLess(selection.paired(rows, "c", "b", "median_ms")["median_pct"], 0)
        rule = selection.winner_rule(rows, ["a", "b", "c"])
        self.assertEqual(rule["tie_set"], ["a", "b", "c"])
        self.assertEqual(rule["winner"], "c")
        self.assertTrue(rule["decided_by"].startswith("head-to-head latency"))
        lat["c"] = list(lat["b"])                                # exactly equal everywhere: the name decides
        rule = selection.winner_rule(_rows(e, lat), ["a", "b", "c"])
        self.assertEqual(rule["winner"], "b")
        self.assertTrue(rule["decided_by"].startswith("name"))

    def test_legacy_amendment_matches_codex(self):
        out = selection.amend_legacy(type("A", (), {"run": "sel-20261020"})())
        self.assertEqual(out["corrected"]["winner"], "B-hybrid")
        self.assertTrue(out["corrected"]["decided_by"].startswith("energy"))
        h = out["corrected"]["head_to_head"]["int8_mma"]
        self.assertAlmostEqual(h["energy_anchor_vs_candidate"]["median_pct"], -3.988, places=2)
        self.assertEqual(h["anchor_lower"], 23)


class Reclassification(unittest.TestCase):
    def test_bytes_move_to_dense_and_totals_are_conserved(self):
        report = {"dense_bytes": 1000, "activation_bytes": 50, "weight_code_bytes": 900, "scale_bytes": 40, "bias_bytes": 0,
                  "storages": {"dense": 3, "activation": 1, "weights": 4},
                  "weights": [{"bytes": 500, "path": "provider.x", "class": "code"},
                              {"bytes": 300, "path": "decoder.predict.<closure table>", "class": "code"},
                              {"bytes": 100, "path": "joint.joint_after_projection.<closure comp>.state.a", "class": "code"},
                              {"bytes": 40, "path": "provider.s", "class": "scale"}]}
        before = sum(report[k] for k in selection._TOTAL_KEYS)
        selection.reclassify_decoder_side(report)
        self.assertEqual(sum(report[k] for k in selection._TOTAL_KEYS), before)
        self.assertEqual(report["weight_code_bytes"], 500)
        self.assertEqual(report["dense_bytes"], 1400)
        self.assertEqual(report["storages"], {"dense": 5, "activation": 1, "weights": 2})
        self.assertEqual([w["path"] for w in report["weights"]], ["provider.x", "provider.s"])
        self.assertEqual(report["reclassified_decoder_side_bytes"], 400)
        self.assertEqual(report["raw"]["weight_code_bytes"], 900)
        self.assertEqual(report["raw"]["dense_bytes"], 1000)
        self.assertEqual(len(report["raw"]["weights"]), 4)
        snapshot = json.dumps(report, sort_keys=True)
        selection.reclassify_decoder_side(report)                # idempotent
        self.assertEqual(json.dumps(report, sort_keys=True), snapshot)


class GateBinding(unittest.TestCase):
    def setUp(self):
        self.run = f"test-gate-{os.getpid()}"
        self.impl = "h_reference"
        self.dir = gate.gate_dir("H", self.run, self.impl)
        self.fp = gate.bind("H", self.run, self.impl, [])
        from gpu1bit.shared import workloads
        self.doc = workloads.load()

    def tearDown(self):
        shutil.rmtree(self.dir.parent, ignore_errors=True)

    def _row(self, rec, part, **over):
        side = {"text": "x", "tokens": [1, 2], "timestamps": [0, 1],
                "score": {"scored": True, "ref_norm": "a b", "S": 0, "D": 0, "I": 0}}
        row = {"id": rec["id"], "part": part, "fingerprint": self.fp["hash"], "gpu": {"device": "RTX 5090"},
               "finite": True, "lengths_equal": True, "rel_frobenius_valid": 1e-7, "max_abs_valid": 1e-7,
               "reference": dict(side), "candidate": dict(side)}
        if part == "timing":
            row["identical"] = True
        row.update(over)
        return row

    def _write(self, sample_over=None, timing_over=None):
        sample = [self._row(r, "sample") for r in self.doc["dev_sample"]]
        timing_ids = list({r["id"]: r for r in self.doc["dev_clips"] + self.doc["dev_24"]}.values())
        timing = [self._row(r, "timing") for r in timing_ids]
        for i, over in (sample_over or {}).items():
            sample[i].update(over)
        for i, over in (timing_over or {}).items():
            timing[i].update(over)
        (self.dir / "sample.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sample))
        (self.dir / "timing.jsonl").write_text("".join(json.dumps(r) + "\n" for r in timing))

    def test_complete_records_pass(self):
        self._write()
        v = gate.compute_verdict("H", self.run, self.impl, [])
        self.assertTrue(v["passes"], v)
        self.assertEqual(v["encoder"]["utterances"], 512 + 24)

    def test_stale_verdict_is_never_trusted(self):
        (self.dir / "verdict.json").write_text(json.dumps({"passes": True, "complete": True}))
        v = gate.compute_verdict("H", self.run, self.impl, [])
        self.assertFalse(v["passes"])
        self.assertFalse(json.loads((self.dir / "verdict.json").read_text())["passes"])

    def test_foreign_fingerprint_records_are_refused(self):
        self._write(sample_over={5: {"fingerprint": "0" * 64}})
        with self.assertRaisesRegex(ValueError, "another fingerprint"):
            gate.compute_verdict("H", self.run, self.impl, [])

    def test_source_change_after_the_gate_fails_it(self):
        self._write()
        probe = env.SHARED / f"_gate_probe_{os.getpid()}.py"
        try:
            probe.write_text("# temporary\n")
            v = gate.compute_verdict("H", self.run, self.impl, [])
            self.assertFalse(v["passes"])
            self.assertTrue(any("changed since" in p for p in v["problems"]))
            with self.assertRaises(SystemExit):
                gate.bind("H", self.run, self.impl, [])
        finally:
            probe.unlink()
        self.assertTrue(gate.compute_verdict("H", self.run, self.impl, [])["passes"])

    def test_encoder_criteria_cover_timing_workloads(self):
        for label, over in (("length", {"lengths_equal": False}), ("finite", {"finite": False}),
                            ("frobenius", {"rel_frobenius_valid": 2e-3}), ("missing fields", None)):
            with self.subTest(case=label):
                self._write(timing_over={3: over} if over else None)
                if over is None:
                    rows = [json.loads(l) for l in (self.dir / "timing.jsonl").read_text().splitlines()]
                    for k in ("finite", "lengths_equal", "rel_frobenius_valid"):
                        rows[3].pop(k)
                    (self.dir / "timing.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
                v = gate.compute_verdict("H", self.run, self.impl, [])
                self.assertTrue(v["timing"]["ok"])            # recognition identical, yet...
                self.assertFalse(v["encoder"]["ok"])           # ...the encoder criteria fail it
                self.assertFalse(v["passes"])

    def test_two_gpu_identities_fail(self):
        self._write(timing_over={0: {"gpu": {"device": "another GPU"}}})
        v = gate.compute_verdict("H", self.run, self.impl, [])
        self.assertFalse(v["passes"])


class ExportFingerprint(unittest.TestCase):
    """Codex GPU code review c2: the fingerprint binds the COMPLETE export directories (every file:
    weights, manifest.json with the model configuration, tokenizer files), not only the weights digest.
    A metadata-only change (config.preprocessor.normalize) changes the fingerprint and fails the
    verdict of gate records bound before it; restoring the file restores both."""

    def test_metadata_only_change_invalidates(self):
        import tempfile
        for attr in ("MB1_EXPORT", "DERIVED_EXPORT"):
            real = getattr(env, attr)
            with tempfile.TemporaryDirectory(dir=str(env.ARTIFACTS_ROOT / "H")) as tmp, self.subTest(export=attr):
                copy = os.path.join(tmp, "export")
                shutil.copytree(real, copy)
                setattr(env, attr, type(real)(copy))
                try:
                    before = F.exports()
                    name = "MB1" if attr == "MB1_EXPORT" else "derived"
                    self.assertIn("manifest.json", before[name]["files"])
                    self.assertTrue(any(k.startswith("tokenizer/") for k in before[name]["files"]))
                    self.assertEqual(before[name]["sha256"], json.loads((real / "manifest.json").read_text())["sha256"])
                    fp = F.make("test")
                    path = os.path.join(copy, "manifest.json")
                    doc = json.loads(open(path).read())
                    self.assertEqual(doc["config"]["preprocessor"]["normalize"], "per_feature")
                    doc["config"]["preprocessor"]["normalize"] = "all_features"
                    with open(path, "w") as f:
                        json.dump(doc, f)
                    after = F.exports()
                    self.assertEqual(after[name]["sha256"], before[name]["sha256"], "the weights digest alone is unchanged")
                    self.assertNotEqual(after[name]["config_sha256"], before[name]["config_sha256"])
                    self.assertNotEqual(F.make("test")["hash"], fp["hash"])
                    self.assertIn(f"exports.{name}", F.diff(fp, F.make("test")))
                    shutil.copy(real / "manifest.json", path)
                    self.assertEqual(F.make("test")["hash"], fp["hash"])
                finally:
                    setattr(env, attr, real)

    def test_gate_bound_before_a_metadata_change_fails(self):
        import tempfile
        run, impl = f"test-gate-export-{os.getpid()}", "h_reference"
        real = env.MB1_EXPORT
        with tempfile.TemporaryDirectory(dir=str(env.ARTIFACTS_ROOT / "H")) as tmp:
            copy = os.path.join(tmp, "export")
            shutil.copytree(real, copy)
            env.MB1_EXPORT = type(real)(copy)
            try:
                gate.bind("H", run, impl, [])
                self.assertFalse(any("changed" in p for p in gate.compute_verdict("H", run, impl, [])["problems"]))
                path = os.path.join(copy, "manifest.json")
                doc = json.loads(open(path).read())
                doc["config"]["preprocessor"]["normalize"] = "all_features"
                with open(path, "w") as f:
                    json.dump(doc, f)
                v = gate.compute_verdict("H", run, impl, [])
                self.assertTrue(any("changed since the gate ran" in p and "exports.MB1" in p for p in v["problems"]), v["problems"])
                self.assertFalse(v["passes"])
            finally:
                env.MB1_EXPORT = real
                shutil.rmtree(gate.gate_dir("H", run, impl).parent, ignore_errors=True)


class TimingBinding(unittest.TestCase):
    """Codex GPU code review c2: a session's recorded timing may be re-used for a corrected decision
    only if its binding still verifies (selection.verify_timing_binding):
      - the archived sources equal the freeze manifest file for file, with nothing missing or extra;
      - every timing step has an accepted sidecar attempt whose freeze check passed (files listed,
        none mismatched, none added) and whose load stayed <= 2;
      - each output is unchanged since that attempt ended (inode change time not later than the end
        of the attempt), and it names the step's protocol, run, variants and range;
      - the roster, the freeze manifest and every file of the exports were last changed before the
        first accepted timing attempt started."""

    def setUp(self):
        import tarfile
        import tempfile
        import time
        self.tmp = tempfile.mkdtemp(dir=str(env.ARTIFACTS_ROOT / "H"))
        self.run = f"test-binding-{os.getpid()}"
        self.roster = {"run": self.run, "variants": ["R-expanded", "x"], "patches": [], "phase_pair": []}
        src = os.path.join(self.tmp, "src", "finetune", "a")
        os.makedirs(src)
        for name in ("one.py", "two.py"):
            with open(os.path.join(src, name), "w") as f:
                f.write(f"# {name}\n")
        self.freeze = os.path.join(self.tmp, "freeze.sha256")
        with open(self.freeze, "w") as f:
            for name in ("one.py", "two.py"):
                f.write(f"{env.sha256_file(Path(src) / name)}  finetune/a/{name}\n")
        self.archive = os.path.join(self.tmp, "archive.tar.gz")
        with tarfile.open(self.archive, "w:gz") as t:
            for name in ("one.py", "two.py"):
                t.add(os.path.join(src, name), arcname=f"finetune/a/{name}")
        self.host = Path(self.tmp) / "host"
        self.host.mkdir()
        self.export = Path(self.tmp) / "export"
        self.export.mkdir()
        (self.export / "manifest.json").write_text("{}")
        self.roster_file = Path(self.tmp) / "roster.json"
        self.roster_file.write_text(json.dumps(self.roster))
        import gpu1bit.rerun_timing as T
        self.T = T
        self.steps = T.timing_steps(self.run, self.roster)
        time.sleep(1.1)                    # sidecar start times have 1 s resolution; inputs come strictly before
        self.attempt_start = time.time()
        for step in self.steps:
            self.write(step)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        for d in (self.run, f"{self.run}-clips1", f"{self.run}-clips2"):
            shutil.rmtree(env.ARTIFACTS / "bench" / d, ignore_errors=True)

    def write(self, step, doc=None, accepted=True, freeze_ok=True, load=1.2, end=None):
        import time
        if doc is None:
            if step["kind"] == "clips":
                doc = {"protocol": "P-clips+P-memory", "variant": step["variant"], "args": {"run": f"{self.run}-{step['label']}"},
                       "workloads": [{}, {}, {}], "memory_samples_10ms": []}
            else:
                a, b = (int(v) for v in step["range"].split("-"))
                doc = {"protocol": "P-paired", "scope": step["scope"], "variants": self.roster["variants"],
                       "args": {"run": self.run, "utterances": step["range"]}, "rows": [{"index": k} for k in range(a, b + 1)]}
        step["output"].parent.mkdir(parents=True, exist_ok=True)
        step["output"].write_text(json.dumps(doc))
        start = self.attempt_start if end is None else end - 2
        start = float(int(start))                                   # as the recorded string resolves it
        attempt = {"accepted": accepted, "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start)),
                   "seconds": (time.time() + 2 - start) if end is None else 2.0, "load1_max": load,
                   "freeze": {"files": 2, "mismatched": [], "added": [], "ok": freeze_ok}}
        (self.host / f"{step['name']}.json").write_text(json.dumps({"step": step["name"], "accepted": accepted, "attempts": [attempt]}))

    def verify(self):
        return selection.verify_timing_binding(self.run, self.roster, host=self.host, archive=Path(self.archive),
                                               freeze=Path(self.freeze), roster_file=self.roster_file,
                                               inputs=[self.export], max_load=2.0)

    def test_consistent_binding_verifies(self):
        out = self.verify()
        self.assertTrue(out["ok"])
        self.assertEqual(out["archive"]["files"], 2)
        self.assertEqual(out["steps"], len(self.steps))

    def test_archive_mismatch_is_refused(self):
        with open(self.freeze, "a") as f:
            f.write("0" * 64 + "  finetune/a/three.py\n")
        with self.assertRaisesRegex(SystemExit, "archive"):
            self.verify()

    def test_unaccepted_or_unfrozen_or_loaded_step_is_refused(self):
        step = self.steps[3]
        for kw in ({"accepted": False}, {"freeze_ok": False}, {"load": 2.4}):
            with self.subTest(kw=kw):
                self.write(step, **kw)
                with self.assertRaisesRegex(SystemExit, step["name"]):
                    self.verify()
        self.write(step)
        self.verify()

    def test_output_changed_after_acceptance_is_refused(self):
        import time
        step = self.steps[0]
        self.write(step, end=time.time() - 30)        # the sidecar says the attempt ended 30 s ago
        with self.assertRaisesRegex(SystemExit, "after its accepted attempt"):
            self.verify()

    def test_output_naming_other_variants_is_refused(self):
        step = self.steps[1]
        a, b = (int(v) for v in step["range"].split("-"))
        self.write(step, doc={"protocol": "P-paired", "scope": step["scope"], "variants": ["R-expanded", "y"],
                              "args": {"run": self.run, "utterances": step["range"]}, "rows": [{"index": k} for k in range(a, b + 1)]})
        with self.assertRaisesRegex(SystemExit, "not bound"):
            self.verify()

    def test_input_changed_after_timing_started_is_refused(self):
        import time
        first = self.verify()["first_attempt_started"]
        time.sleep(1.1)
        (self.export / "tokenizer.model").write_text("new")
        with self.assertRaisesRegex(SystemExit, "changed after"):
            selection.verify_timing_binding(self.run, self.roster, host=self.host, archive=Path(self.archive),
                                            freeze=Path(self.freeze), roster_file=self.roster_file,
                                            inputs=[self.export], max_load=2.0, inputs_before=first)


class RecordFingerprint(unittest.TestCase):
    """A5 audit and deferred-A3 records carry the fingerprint they were made under (Codex GPU c2: re-run
    under the new fingerprints); selection.record_bound accepts a record only if it matches now."""

    def test_record_bound(self):
        fp = F.make("A5 audit", variant="x", patches=[])
        self.assertTrue(selection.record_bound({"fingerprint": fp}, F.make("A5 audit", variant="x", patches=[])))
        self.assertFalse(selection.record_bound({"fingerprint": fp}, F.make("A5 audit", variant="y", patches=[])))
        self.assertFalse(selection.record_bound({}, fp))
        stale = dict(fp, exports={"MB1": "old"})
        self.assertFalse(selection.record_bound({"fingerprint": stale}, fp))


class SessionFreeze(unittest.TestCase):
    def test_changed_and_added_files_are_refused(self):
        run = f"test-freeze-{os.getpid()}"
        fz = selection.freeze_path(run)
        try:
            lines = [f"{env.sha256_file(p)}  {p.relative_to(env.REPO)}" for p in F.source_files()]
            fz.write_text("\n".join(lines) + "\n")
            self.assertTrue(selection.verify_freeze(run)["ok"])
            probe = env.SHARED / f"_freeze_probe_{os.getpid()}.py"
            try:
                probe.write_text("# added\n")
                with self.assertRaisesRegex(SystemExit, "added"):
                    selection.verify_freeze(run)
            finally:
                probe.unlink()
            tampered = lines[:]
            tampered[0] = "0" * 64 + tampered[0][64:]
            fz.write_text("\n".join(tampered) + "\n")
            with self.assertRaisesRegex(SystemExit, "mismatched"):
                selection.verify_freeze(run)
        finally:
            fz.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
