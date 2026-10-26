"""Contract tests: the equivalence check, the consistency gate, engine selection and the gate to stage 3."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import devsets
from labeling import c2, checks, config, filters, replay, shards
from labeling_fixtures import World, relabel_root
from test_labeling_c2 import RefB0, RefTeacher, table

WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike".split()


def clip_row(ident: str, source: str, ref: str | None, kind: str | None, seconds: float = 2.0) -> dict:
    rng = np.random.default_rng([sum(map(ord, ident)), len(ident)])
    pcm = rng.integers(-2000, 2000, int(seconds * 16000), dtype=np.int16)
    return {"id": ident, "source": source, "duration": pcm.size / 16000, "samples": pcm.size, "reference": ref,
            "reference_kind": kind, "pcm_sha256": shards.pcm_sha256(pcm), "flac": shards.encode_flac(pcm)}


def write_outputs(run_dir: Path, pool: Path, labels: dict[str, str], b0: dict[str, str] | None = None,
                  signals: dict[str, dict] | None = None) -> None:
    run = c2.load_run(run_dir)
    for g, ks in enumerate(c2.groups(pool, run["group_shards"])):
        ids = [r["id"] for k in ks for r in shards.read_audio(shards.paths(pool, k)["audio"], columns=("id",))]
        sig = lambda i: c2.signal_row((signals or {}).get(i))
        for step, rows in (("teacher", [{"id": i, "raw": labels[i], "hit_cap": False} | sig(i) for i in ids]),
                           ("b0", [{"id": i, "b0": (b0 or {}).get(i, labels[i])} for i in ids] if b0 is not None else None)):
            if rows is None:
                continue
            p = run_dir / "work" / step / f"{c2.gname(g)}.parquet"
            sha = c2._write_parquet(p, rows, c2.teacher_schema() if step == "teacher" else None)
            c2.write_json(p.with_suffix(".json"), {"group": g, "shards": ks, "clips": len(ids), "sha256": sha,
                                                   "engine": {"engine": "fake"}, "audio_seconds": 1.0, "read_s": 0.0,
                                                   "decode_and_engine_s": 1.0, "load_s": 0.0})


class Equivalence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ctx = relabel_root(self.root)
        self.ctx.__enter__()
        self.d = self.root / "eq"
        rows, self.ids = [], {}
        for src, kind in (("srcA", "pseudo"), ("srcB", "human")):
            self.ids[src] = [f"{src}:{i:03d}" for i in range(100)]
            rows += [clip_row(i, src, " ".join(WORDS[j % 13] for j in range(n, n + 6)), kind)
                     for n, i in enumerate(self.ids[src])]
        self.parts = [p for s in ("dev-medical", "dev-technical") for p in devsets.SETS[s]]
        for part in self.parts:
            rows += [clip_row(f"{part}:{i}", part, " ".join(WORDS[(i + j) % 13] for j in range(10)), "human")
                     for i in range(20)]
        checks.write_rows_as_shards(rows, self.d / "pool", 60)
        c2.write_json(self.d / "pool" / "materialized.json", {})
        c2.write_json(self.d / "set.json", {"c1_ids": self.ids})
        self.refs = {r["id"]: r["reference"] for r in rows}
        for name, g in (("b32", 100), ("bmax", 2), ("vllm", 2), ("vllm-resumed", 2)):
            c2.init_run(self.d / name, pool_dir=self.d / "pool",
                        engine={"kind": "fake", "name": name.replace("-resumed", "")},
                        group_shards=g, b0_batch_seconds=60.0, declared_vram_gb=None)

    def tearDown(self):
        self.ctx.__exit__(None, None, None)
        self.tmp.cleanup()

    def changed(self, ids, n, how="word"):
        out = dict(self.refs)
        for i in ids[:n]:
            out[i] = "zulu " + self.refs[i] if how == "word" else "nothing like it at all whatsoever here"
        return out

    def outcome(self, bmax, vllm, resumed=None, signals=None, resumed_signals=None):
        write_outputs(self.d / "b32", self.d / "pool", self.refs, b0=self.refs)
        write_outputs(self.d / "bmax", self.d / "pool", bmax)
        write_outputs(self.d / "vllm", self.d / "pool", vllm, signals=signals)
        write_outputs(self.d / "vllm-resumed", self.d / "pool", resumed or vllm,
                      signals=resumed_signals if resumed_signals is not None else signals)
        return checks.evaluate_equivalence(self.d, coverage=lambda t: True)

    def signals(self, shift=0.0, at=-30.0, ids_change=False):
        out = {}
        for i in self.refs:
            n = len(self.refs[i].split()) + 1
            lp = [-0.25] * (n - 1) + [at]
            out[i] = {"token_ids": list(range(n)) if not (ids_change and i == self.ids["srcA"][0]) else list(range(1, n + 1)),
                      "logprob": [x + shift for x in lp], "runner_up": [x - 3.0 + shift for x in lp]}
        return out

    def test_identical_resumed_signals_pass(self):
        r = self.outcome(dict(self.refs), dict(self.refs), signals=self.signals(), resumed_signals=self.signals())
        self.assertTrue(r["vllm_runs_identical"])
        self.assertTrue(r["vllm_logprobs_identical"])
        self.assertEqual(r["vllm_logprob_max_abs_diff"], 0.0)
        self.assertTrue(r["passed"], r["sources"])

    def test_resumed_log_probabilities_must_be_bitwise_identical(self):
        """labeling plan 2026-10-24 (deterministic vLLM): any difference in a stored log-probability fails, down to one
        float16 step (2**-11 at 0.25-0.5 nats)."""
        base = self.signals()
        other = self.signals()
        i = self.ids["srcB"][7]
        other[i] = dict(other[i], logprob=[other[i]["logprob"][0] - 2 ** -11] + other[i]["logprob"][1:])
        r = self.outcome(dict(self.refs), dict(self.refs), signals=base, resumed_signals=other)
        self.assertTrue(r["vllm_runs_identical"])
        self.assertFalse(r["vllm_logprobs_identical"])
        self.assertGreater(r["vllm_logprob_max_abs_diff"], 0.0)
        self.assertFalse(r["passed"])
        other = self.signals()
        other[i] = dict(other[i], runner_up=other[i]["runner_up"][:-1] + [other[i]["runner_up"][-1] - 0.5])
        r = self.outcome(dict(self.refs), dict(self.refs), signals=base, resumed_signals=other)
        self.assertFalse(r["vllm_logprobs_identical"])
        self.assertFalse(r["passed"])

    def test_resumed_token_ids_must_be_identical(self):
        r = self.outcome(dict(self.refs), dict(self.refs), signals=self.signals(),
                         resumed_signals=self.signals(ids_change=True))
        self.assertFalse(r["vllm_runs_identical"])
        self.assertFalse(r["passed"])

    def test_vllm_runs_with_signals_on_must_have_them_for_every_clip(self):
        sig = self.signals()
        partial = dict(sig)
        partial.pop(self.ids["srcB"][3])
        c2.init_run(self.d / "vllm-sig", pool_dir=self.d / "pool", engine={"kind": "fake", "confidence": True},
                    group_shards=2, b0_batch_seconds=60.0, declared_vram_gb=None)
        write_outputs(self.d / "vllm-sig", self.d / "pool", self.refs, signals=partial)
        with self.assertRaises(RuntimeError):
            checks.teacher_signals(self.d / "vllm-sig")

    def test_passes_at_the_limit(self):
        bmax = self.changed(self.ids["srcA"], 2)
        vllm = self.changed(self.ids["srcA"][50:], 5)        # 5% <= 2 x 2% + 1 point
        r = self.outcome(bmax, vllm)
        self.assertTrue(r["passed"], r["sources"])
        self.assertAlmostEqual(r["sources"]["srcA"]["limit"], 0.05)

    def test_fails_above_the_label_limit(self):
        r = self.outcome(self.changed(self.ids["srcA"], 2), self.changed(self.ids["srcA"][50:], 6))
        self.assertFalse(r["passed"])
        self.assertFalse(r["sources"]["srcA"]["label_ok"])

    def test_fails_when_the_resumed_run_differs(self):
        vllm = dict(self.refs)
        r = self.outcome(dict(self.refs), vllm, resumed=self.changed(self.ids["srcA"], 1))
        self.assertFalse(r["vllm_runs_identical"])
        self.assertFalse(r["passed"])

    def test_fails_on_changed_dispositions(self):
        bmax = self.changed(self.ids["srcB"], 3)
        vllm = self.changed(self.ids["srcB"][50:], 2, how="all")   # human WER > 0.5: rejected (b32: kept)
        r = self.outcome(bmax, vllm)
        self.assertTrue(r["sources"]["srcB"]["label_ok"])
        self.assertAlmostEqual(r["sources"]["srcB"]["disposition_changes"], 0.02)
        self.assertFalse(r["passed"])

    def test_fails_on_dev_wer(self):
        part = devsets.SETS["dev-technical"][0]
        vllm = self.changed([f"{part}:{i}" for i in range(20)], 1)
        r = self.outcome(dict(self.refs), vllm)
        self.assertFalse(r["dev"]["dev-technical"]["ok"])
        self.assertTrue(r["dev"]["dev-medical"]["ok"])
        self.assertFalse(r["passed"])


class Consistency(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ctx = relabel_root(self.root)
        self.ctx.__enter__()
        self.d, self.s1 = self.root / "cons", self.root / "stage1"
        rows, info, summary = [], {"sets": {}}, {"sets": {}}
        self.hyps = {}
        for name in ("ami_dev", "voxpopuli_dev"):
            recs = []
            for i in range(30):
                ref = " ".join(WORDS[(i + j) % 13] for j in range(8))
                hyp = ref if i % 10 else "zulu " + ref
                recs.append({"id": f"{name}:{i:03d}", "ref": ref, "hyp": hyp})
                rows.append(clip_row(f"{name}:{i:03d}", name, ref, "human"))
                self.hyps[f"{name}:{i:03d}"] = hyp
            wer = checks.corpus_wer([(r["ref"], r["hyp"]) for r in recs])
            summary["sets"][name] = {"wer": wer}
            info["sets"][name] = {"manifest_sha256": "m"}
            self.s1.mkdir(parents=True, exist_ok=True)
            (self.s1 / f"{name}.json").write_text(json.dumps({"records": recs, "manifest_sha256": "m"}))
        (self.s1 / "summary.json").write_text(json.dumps(summary))
        checks.write_rows_as_shards(rows, self.d / "pool", 25)
        c2.write_json(self.d / "pool" / "materialized.json", {})
        c2.write_json(self.d / "set.json", info)
        c2.init_run(self.d / "run", pool_dir=self.d / "pool", engine={"kind": "fake"}, group_shards=2,
                    b0_batch_seconds=60.0, declared_vram_gb=None)

    def tearDown(self):
        self.ctx.__exit__(None, None, None)
        self.tmp.cleanup()

    def test_reproduces_stage_one(self):
        write_outputs(self.d / "run", self.d / "pool", self.hyps)
        r = checks.evaluate_consistency(self.d, stage1_dir=self.s1)
        self.assertTrue(r["passed"])
        self.assertAlmostEqual(r["sets"]["ami_dev"]["diff"], 0.0)

    def test_fails_beyond_five_hundredths(self):
        hyps = dict(self.hyps)
        hyps["ami_dev:001"] = "zulu " + hyps["ami_dev:001"]
        write_outputs(self.d / "run", self.d / "pool", hyps)
        r = checks.evaluate_consistency(self.d, stage1_dir=self.s1)
        self.assertFalse(r["sets"]["ami_dev"]["ok"])
        self.assertTrue(r["sets"]["voxpopuli_dev"]["ok"])
        self.assertFalse(r["passed"])

    def test_population_must_be_stage_ones(self):
        data = json.loads((self.s1 / "ami_dev.json").read_text())
        data["records"].append({"id": "ami_dev:999", "ref": "x", "hyp": "x"})
        (self.s1 / "ami_dev.json").write_text(json.dumps(data))
        write_outputs(self.d / "run", self.d / "pool", self.hyps)
        with self.assertRaises(RuntimeError):
            checks.evaluate_consistency(self.d, stage1_dir=self.s1)

    # --- hallucination-guard exclusion (labeling plan "Gate to stage 3" item 1, amended 2026-10-25) ---

    LOOP = "alpha " * 40                                       # 20 words per second and a repeated word: the guard

    def stage1_hyps(self, name: str, hyps: dict[str, str]) -> None:
        """Replace stage 1's transcripts of some clips (records and summary WER, as stage 1 wrote them)."""
        data = json.loads((self.s1 / f"{name}.json").read_text())
        for r in data["records"]:
            r["hyp"] = hyps.get(r["id"], r["hyp"])
        (self.s1 / f"{name}.json").write_text(json.dumps(data))
        summary = json.loads((self.s1 / "summary.json").read_text())
        summary["sets"][name]["wer"] = checks.corpus_wer([(r["ref"], r["hyp"]) for r in data["records"]])
        (self.s1 / "summary.json").write_text(json.dumps(summary))

    def test_stage_one_loop_is_excluded_and_unfiltered_numbers_reported(self):
        self.stage1_hyps("ami_dev", {"ami_dev:005": self.LOOP})
        write_outputs(self.d / "run", self.d / "pool", self.hyps)   # the run transcribes ami_dev:005 correctly
        r = checks.evaluate_consistency(self.d, stage1_dir=self.s1)
        s = r["sets"]["ami_dev"]
        recs = json.loads((self.s1 / "ami_dev.json").read_text())["records"]
        kept = [x for x in recs if x["id"] != "ami_dev:005"]
        self.assertEqual((s["excluded"], s["guard_rejected_stage1"], s["guard_rejected_run"]), (1, 1, 0))
        self.assertAlmostEqual(s["wer"], checks.corpus_wer([(x["ref"], self.hyps[x["id"]]) for x in kept]))
        self.assertAlmostEqual(s["stage1_wer"], checks.corpus_wer([(x["ref"], x["hyp"]) for x in kept]))
        self.assertAlmostEqual(s["diff"], 0.0)
        self.assertAlmostEqual(s["wer_unfiltered"], checks.corpus_wer([(x["ref"], self.hyps[x["id"]]) for x in recs]))
        self.assertAlmostEqual(s["stage1_wer_unfiltered"], checks.corpus_wer([(x["ref"], x["hyp"]) for x in recs]))
        self.assertAlmostEqual(s["diff_unfiltered"], s["wer_unfiltered"] - s["stage1_wer_unfiltered"])
        self.assertLess(s["diff_unfiltered"], -checks.WER_POINTS)   # the unfiltered rule would fail this set
        self.assertEqual(s["guard_allowance"], 2)
        self.assertTrue(s["ok"])
        self.assertTrue(r["passed"])
        self.assertEqual(r["gate_version"], 2)
        self.assertIsInstance(r["criterion"], str)
        for k in ("full_population", "contract", "limit", "run", "engine"):
            self.assertIn(k, r)

    def test_a_loop_on_both_sides_is_excluded_once(self):
        self.stage1_hyps("ami_dev", {"ami_dev:005": self.LOOP})
        write_outputs(self.d / "run", self.d / "pool", self.hyps | {"ami_dev:005": self.LOOP + "bravo"})
        s = checks.evaluate_consistency(self.d, stage1_dir=self.s1)["sets"]["ami_dev"]
        self.assertEqual((s["excluded"], s["guard_rejected_stage1"], s["guard_rejected_run"]), (1, 1, 1))
        self.assertTrue(s["ok"])

    def test_run_hallucinating_beyond_the_allowance_fails(self):
        loops = {f"ami_dev:{i:03d}": self.LOOP for i in (1, 2, 3)}
        write_outputs(self.d / "run", self.d / "pool", self.hyps | loops)
        r = checks.evaluate_consistency(self.d, stage1_dir=self.s1)
        s = r["sets"]["ami_dev"]
        self.assertEqual((s["excluded"], s["guard_rejected_stage1"], s["guard_rejected_run"]), (3, 0, 3))
        self.assertAlmostEqual(s["diff"], 0.0)                   # the kept clips match stage 1 exactly
        self.assertFalse(s["ok"])
        self.assertTrue(r["sets"]["voxpopuli_dev"]["ok"])
        self.assertFalse(r["passed"])

    def test_run_hallucinating_within_the_allowance_passes(self):
        loops = {f"ami_dev:{i:03d}": self.LOOP for i in (1, 2)}
        write_outputs(self.d / "run", self.d / "pool", self.hyps | loops)
        r = checks.evaluate_consistency(self.d, stage1_dir=self.s1)
        s = r["sets"]["ami_dev"]
        self.assertEqual((s["guard_rejected_run"], s["guard_allowance"]), (2, 2))
        self.assertGreater(abs(s["diff_unfiltered"]), checks.WER_POINTS)
        self.assertTrue(s["ok"])
        self.assertTrue(r["passed"])

    def test_allowance_counts_against_stage_ones_rejections(self):
        self.stage1_hyps("ami_dev", {"ami_dev:004": self.LOOP})
        loops = {f"ami_dev:{i:03d}": self.LOOP for i in (1, 2, 3)}   # 3 run rejections <= 1 + 2
        write_outputs(self.d / "run", self.d / "pool", self.hyps | loops)
        s = checks.evaluate_consistency(self.d, stage1_dir=self.s1)["sets"]["ami_dev"]
        self.assertEqual((s["excluded"], s["guard_rejected_stage1"], s["guard_rejected_run"]), (4, 1, 3))
        self.assertTrue(s["ok"])

    def test_human_wer_filter_does_not_exclude(self):
        write_outputs(self.d / "run", self.d / "pool", self.hyps | {"ami_dev:001": "nothing like it at all"})
        s = checks.evaluate_consistency(self.d, stage1_dir=self.s1)["sets"]["ami_dev"]
        self.assertEqual(s["excluded"], 0)
        self.assertFalse(s["ok"])

    def test_a_set_with_nothing_left_to_compare_fails(self):
        d = self.root / "cons-one"                              # a --limit 1 smoke
        rows = [clip_row(f"{name}:000", name, "x", "human") for name in ("ami_dev", "voxpopuli_dev")]
        checks.write_rows_as_shards(rows, d / "pool", 25)
        c2.write_json(d / "pool" / "materialized.json", {})
        c2.write_json(d / "set.json", {"sets": {n: {"manifest_sha256": "m"} for n in ("ami_dev", "voxpopuli_dev")},
                                       "limit": 1})
        c2.init_run(d / "run", pool_dir=d / "pool", engine={"kind": "fake"}, group_shards=2, b0_batch_seconds=60.0,
                    declared_vram_gb=None)
        write_outputs(d / "run", d / "pool", {"ami_dev:000": self.LOOP, "voxpopuli_dev:000": self.hyps["voxpopuli_dev:000"]})
        r = checks.evaluate_consistency(d, stage1_dir=self.s1)
        s = r["sets"]["ami_dev"]
        self.assertEqual((s["excluded"], s["guard_rejected_run"]), (1, 1))
        self.assertIsNone(s["wer"])
        self.assertIsNone(s["diff"])
        self.assertGreater(s["diff_unfiltered"], checks.WER_POINTS)
        self.assertFalse(s["ok"])
        self.assertTrue(r["sets"]["voxpopuli_dev"]["ok"])
        self.assertFalse(r["passed"])
        self.assertTrue((d / "result-run.json").exists())

    def test_the_guard_applies_the_runs_filters_version(self):
        """ami_dev:000 at 1 s: both transcripts are over 6 words per second and within WER 0.2 of the reference, so the
        guard rejects them under filters version 1 and accepts them under version 2 (labeling plan "Filters",
        amended 2026-10-25)."""
        for version, excluded in ((1, 1), (2, 0)):
            d = self.root / f"cons-filters-v{version}"
            rows = [clip_row(i, name, ref, "human", seconds=1.0 if i == "ami_dev:000" else 2.0)
                    for name in ("ami_dev", "voxpopuli_dev")
                    for i, ref in sorted((x["id"], x["ref"]) for x in
                                         json.loads((self.s1 / f"{name}.json").read_text())["records"])[:10]]
            checks.write_rows_as_shards(rows, d / "pool", 25)
            c2.write_json(d / "pool" / "materialized.json", {})
            c2.write_json(d / "set.json", {"sets": {n: {"manifest_sha256": "m"} for n in ("ami_dev", "voxpopuli_dev")},
                                           "limit": 10})
            c2.init_run(d / "run", pool_dir=d / "pool", engine={"kind": "fake"}, group_shards=2,
                        b0_batch_seconds=60.0, declared_vram_gb=None, extra={"filters_version": version})
            write_outputs(d / "run", d / "pool", {r["id"]: r["reference"] for r in rows})
            s = checks.evaluate_consistency(d, stage1_dir=self.s1)["sets"]["ami_dev"]
            self.assertEqual((s["excluded"], s["guard_rejected_run"], s["guard_rejected_stage1"]),
                             (excluded, excluded, excluded), version)

    def test_the_guard_scores_the_mapped_text_under_version_3(self):
        """guard_rejects passes the run's version to filters.check, so under version 3 a parenthetical's words, which
        the normalizer deletes from the raw label, count toward words per second."""
        label = "patient stable (alpha beta gamma delta epsilon zeta)"
        for version, rejected in ((1, False), (2, False), (3, True)):
            self.assertEqual(checks.guard_rejects(1.0, label, "patient stable", version), rejected, version)

    def test_allowance(self):
        self.assertEqual([checks.guard_allowance(n) for n in (1, 30, 400, 401, 2000, 13098)], [2, 2, 2, 3, 10, 66])

    def test_limited_population_is_the_first_ids(self):
        d = self.root / "cons-limit"
        rows = [clip_row(i, name, ref, "human")
                for name in ("ami_dev", "voxpopuli_dev")
                for i, ref in sorted((x["id"], x["ref"]) for x in
                                     json.loads((self.s1 / f"{name}.json").read_text())["records"])[:10]]
        checks.write_rows_as_shards(rows, d / "pool", 25)
        c2.write_json(d / "pool" / "materialized.json", {})
        c2.write_json(d / "set.json", {"sets": {n: {"manifest_sha256": "m"} for n in ("ami_dev", "voxpopuli_dev")},
                                       "limit": 10})
        c2.init_run(d / "run", pool_dir=d / "pool", engine={"kind": "fake"}, group_shards=2, b0_batch_seconds=60.0,
                    declared_vram_gb=None)
        self.stage1_hyps("ami_dev", {"ami_dev:005": self.LOOP})
        write_outputs(d / "run", d / "pool", {r["id"]: self.hyps[r["id"]] for r in rows})
        r = checks.evaluate_consistency(d, stage1_dir=self.s1)
        s = r["sets"]["ami_dev"]
        recs = sorted(json.loads((self.s1 / "ami_dev.json").read_text())["records"], key=lambda x: x["id"])[:10]
        self.assertAlmostEqual(s["stage1_wer_unfiltered"], checks.corpus_wer([(x["ref"], x["hyp"]) for x in recs]))
        self.assertEqual(s["excluded"], 1)
        self.assertTrue(s["ok"])
        self.assertFalse(r["full_population"])
        self.assertEqual(r["limit"], 10)


def contract(engine, group_shards=4, declared=20.0):
    return {"engine": engine, "group_shards": group_shards, "b0_batch_seconds": 1200.0, "declared_vram_gb": declared,
            "teacher_revision": "r", "filters_version": 1}


def cons(contract_, passed=True, full=True):
    return {"passed": passed, "full_population": full, "limit": None if full else 10, "contract": contract_}


class Selection(unittest.TestCase):
    v1 = {"kind": "vllm", "max_num_seqs": 64, "gpu_memory_utilization": 0.4}
    v2 = {"kind": "vllm", "max_num_seqs": 256, "gpu_memory_utilization": 0.7}
    t1 = {"kind": "transformers", "batch_size": 64, "max_batch_seconds": 1200.0}

    def probes(self):
        return [{"contract": contract(self.v1, declared=26.0), "audio_hours_per_gpu_hour": 900, "declared_vram_gb_for_c2": 15},
                {"contract": contract(self.v2, declared=26.0), "audio_hours_per_gpu_hour": 1500, "declared_vram_gb_for_c2": 25},
                {"contract": contract(self.t1, declared=26.0), "audio_hours_per_gpu_hour": 400, "declared_vram_gb_for_c2": 12}]

    def test_fastest_setting_qualified_under_the_production_contract(self):
        c = [cons(contract(self.v1, declared=15.0)), cons(contract(self.t1, declared=12.0))]
        eq = {"passed": True, "contract": contract(self.v1, declared=15.0)}
        sel = checks.select_production(self.probes(), c, eq)["selected"]
        self.assertEqual(sel["contract"], contract(self.v1, declared=15.0))
        c.append(cons(contract(self.v2, declared=25.0)))
        self.assertEqual(checks.select_production(self.probes(), c, eq)["selected"]["contract"]["engine"], self.v1)
        eq2 = {"passed": True, "contract": contract(self.v2, declared=25.0)}
        self.assertEqual(checks.select_production(self.probes(), c, eq2)["selected"]["contract"]["engine"], self.v2)
        eq_fail = {"passed": False, "contract": contract(self.v1, declared=15.0)}
        self.assertEqual(checks.select_production(self.probes(), c, eq_fail)["selected"]["contract"]["engine"], self.t1)
        self.assertIsNone(checks.select_production(self.probes(), [], eq)["selected"])

    def test_checks_below_the_probe_peak_or_limited_or_other_groups_do_not_qualify(self):
        eq = {"passed": True, "contract": contract(self.v1, declared=14.0)}
        c = [cons(contract(self.v1, declared=14.0))]                        # below peak + 2 GB (15)
        self.assertIsNone(checks.select_production(self.probes()[:1], c, eq)["selected"])
        eq = {"passed": True, "contract": contract(self.v1, declared=15.0)}
        c = [cons(contract(self.v1, declared=15.0), full=False)]            # a smoke result
        self.assertIsNone(checks.select_production(self.probes()[:1], c, eq)["selected"])
        c = [cons(contract(self.v1, group_shards=1, declared=15.0))]       # other batch composition
        self.assertIsNone(checks.select_production(self.probes()[:1], c, eq)["selected"])

    def test_a_check_without_batch_invariance_does_not_qualify_a_run_with_it(self):
        on = contract(self.v1 | {"confidence": True, "batch_invariant": True}, declared=21.0)
        off = contract(self.v1 | {"confidence": True}, declared=21.0)
        self.assertFalse(checks.qualifies(on, cons(off), {"passed": True, "contract": on})[0])
        self.assertFalse(checks.qualifies(on, cons(on), {"passed": True, "contract": off})[0])
        self.assertTrue(checks.qualifies(on, cons(on), {"passed": True, "contract": on})[0])

    def test_a_check_without_signals_does_not_qualify_a_run_with_them(self):
        on = contract(self.v1 | {"confidence": True}, declared=15.0)
        off = contract(self.v1, declared=15.0)
        self.assertFalse(checks.qualifies(on, cons(off), {"passed": True, "contract": on})[0])
        self.assertFalse(checks.qualifies(on, cons(on), {"passed": True, "contract": off})[0])
        self.assertTrue(checks.qualifies(on, cons(on), {"passed": True, "contract": on})[0])

    def test_checks_qualify_the_engine_whatever_the_filters_version(self):
        """The checks qualify what produces the labels; filters only decide dispositions and are re-applied on CPU
        (c2.refilter), so a run re-filtered under a later version is still qualified by checks run under an earlier one."""
        k1 = contract(self.v1, declared=15.0)
        k2 = k1 | {"filters_version": 2}
        self.assertTrue(checks.qualifies(k2, cons(k1), {"passed": True, "contract": k1})[0])
        self.assertFalse(checks.qualifies(k2 | {"group_shards": 2}, cons(k1), {"passed": True, "contract": k1})[0])

    def test_qualifies(self):
        k = contract(self.v1, declared=15.0)
        self.assertTrue(checks.qualifies(k, cons(k), {"passed": True, "contract": k})[0])
        self.assertFalse(checks.qualifies(k, cons(k), None)[0])
        self.assertFalse(checks.qualifies(k, cons(k), {"passed": True, "contract": contract(self.v1, 2, 15.0)})[0])
        self.assertFalse(checks.qualifies(k, cons(k, full=False), {"passed": True, "contract": k})[0])
        t = contract(self.t1, declared=12.0)
        self.assertTrue(checks.qualifies(t, cons(t), None)[0])
        self.assertFalse(checks.qualifies(t, cons(contract(self.t1, declared=13.0)), None)[0])


class GpuProbe(unittest.TestCase):
    """run_gpu_probe with the steps run in-process (fake engines): fresh measurements and consistent accounting."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ctx = relabel_root(self.root)
        self.ctx.__enter__()
        rows = [clip_row(f"s:{i:03d}", "s", "alpha bravo", "human") for i in range(40)]
        checks.write_rows_as_shards(rows, self.root / "pool", 10)
        c2.write_json(self.root / "pool" / "materialized.json", {})
        self.refs = {r["pcm_sha256"]: r["reference"] for r in rows}
        self.calls = []
        from unittest import mock
        self.io = mock.patch.object(checks, "IO_START_MAX", 101.0)   # the host's real disk load must not gate tests
        self.io.start()

    def tearDown(self):
        self.io.stop()
        self.ctx.__exit__(None, None, None)
        self.tmp.cleanup()

    def runner(self, cmd, cwd=None):
        import types
        run_dir = Path(cmd[cmd.index("--run-dir") + 1])
        step = cmd[cmd.index("-m") + 2]
        self.calls.append(step)
        if step == "teacher":
            c2.teacher_step(run_dir, engine=RefTeacher(self.refs))
        elif step == "b0":
            c2.b0_step(run_dir, labeler=RefB0(self.refs))
        else:
            c2.write_json(run_dir / "split.json", {"fake": True})
        return types.SimpleNamespace(returncode=0)

    def probe(self, out):
        from unittest import mock
        import relabel_paths
        with mock.patch.object(relabel_paths, "GPU_LOCK", self.root / "gpu.lock"):
            return checks.run_gpu_probe(self.root / "pool", out, {"kind": "fake"}, 5.0, 10.0, group_shards=2,
                                        runner=self.runner,
                                        merge=lambda d: c2.merge_step(d, coverage=lambda t: True))

    def test_interrupted_probe_is_measured_afresh(self):
        out = self.root / "probe"
        c2.init_run(out, pool_dir=self.root / "pool", engine={"kind": "fake"}, group_shards=2,
                    b0_batch_seconds=checks.B0_BATCH_SECONDS, declared_vram_gb=10.0)
        c2.teacher_step(out, engine=RefTeacher(self.refs), stop_after_groups=1)   # an interrupted earlier attempt
        meta = out / "work" / "teacher" / "00000.json"
        old = json.loads(meta.read_text())
        old["group_s"] = 1e6                       # a long-running earlier invocation's group time
        meta.write_text(json.dumps(old))
        r = self.probe(out)
        self.assertEqual(r["groups"], [0, 1])
        self.assertGreaterEqual(r["startup_s_per_chunk"], 0.0)
        self.assertGreater(r["audio_hours_per_gpu_hour"], 0)
        self.assertEqual(self.calls, ["teacher", "b0", "split"])

    def test_steady_rate_excludes_the_first_groups_compile_and_warm_up(self):
        """labeling plan 2026-10-24: the first group of each step's process carries compilation; the steady rate comes from
        the later groups and the first group's excess counts as per-chunk start-up."""
        import time as _time

        class Compiling(RefTeacher):
            first = True

            def run(self, clips):
                if Compiling.first:
                    Compiling.first = False
                    _time.sleep(0.6)               # compile / warm-up on the first call of the process
                return super().run(clips)

        def runner(cmd, cwd=None):
            import types
            run_dir = Path(cmd[cmd.index("--run-dir") + 1])
            step = cmd[cmd.index("-m") + 2]
            if step == "teacher":
                c2.teacher_step(run_dir, engine=Compiling(self.refs))
            elif step == "b0":
                c2.b0_step(run_dir, labeler=RefB0(self.refs))
            else:
                c2.write_json(run_dir / "split.json", {"fake": True})
            return types.SimpleNamespace(returncode=0)

        self.runner = runner
        out = self.root / "probe-compile"
        r = self.probe(out)
        self.assertEqual(r["groups"], [0, 1])
        self.assertEqual(r["steady_groups"], [1])
        metas = {s: {m["group"]: m for m in (json.loads(p.read_text()) for p in (out / "work" / s).glob("*.json"))}
                 for s in ("teacher", "b0")}
        merged = {g: json.loads((out / "groups" / f"{c2.gname(g)}.json").read_text()) for g in (0, 1)}
        g1_s = metas["teacher"][1]["group_s"] + metas["b0"][1]["group_s"] + merged[1]["merge_s"]
        g1_h = metas["teacher"][1]["audio_seconds"] / 3600
        self.assertAlmostEqual(r["steady_audio_hours_per_gpu_hour"], g1_h / (g1_s / 3600), places=6)
        self.assertGreaterEqual(r["first_group_excess_s"], 0.5)
        self.assertGreaterEqual(r["startup_s_per_chunk"], r["first_group_excess_s"])
        self.assertGreater(r["audio_hours_per_gpu_hour"], 0)

    def test_probe_records_io_pressure_during_each_step(self):
        r = self.probe(self.root / "probe-io")
        io = r["io_pressure"]
        for step in ("teacher", "b0", "split"):
            self.assertIn("some", io[step])
            self.assertGreaterEqual(io[step]["some"], 0.0)
        self.assertIn("some", io["start"])

    def test_probe_waits_for_low_io_pressure_and_refuses_if_it_stays_high(self):
        """labeling plan 2026-10-24: the probe starts only while the data disk's IO pressure is low."""
        from unittest import mock
        high = {"some": {"total_us": 0, "avg10": 50.0, "avg60": 40.0}, "full": {"total_us": 0, "avg10": 40.0, "avg60": 30.0}}
        low = {"some": {"total_us": 0, "avg10": 1.0, "avg60": 2.0}, "full": {"total_us": 0, "avg10": 1.0, "avg60": 1.0}}
        readings = iter([high, high, low] + [low] * 50)
        with mock.patch.object(checks, "io_pressure", lambda: next(readings)), \
                mock.patch.object(checks, "IO_POLL_S", 0.01), mock.patch.object(checks, "IO_START_MAX", 10.0):
            r = self.probe(self.root / "probe-wait")
        self.assertEqual(self.calls, ["teacher", "b0", "split"])
        self.assertGreaterEqual(r["io_pressure"]["waited_s"], 0.0)
        self.assertLessEqual(r["io_pressure"]["start"]["some"]["avg60"], 10.0)
        self.calls.clear()
        with mock.patch.object(checks, "io_pressure", lambda: high), mock.patch.object(checks, "IO_POLL_S", 0.01), \
                mock.patch.object(checks, "IO_WAIT_S", 0.05), mock.patch.object(checks, "IO_START_MAX", 10.0):
            with self.assertRaises(RuntimeError):
                self.probe(self.root / "probe-busy")
        self.assertEqual(self.calls, [])

    def test_io_pressure_is_rechecked_once_the_gpu_lock_is_held(self):
        """Pressure that rises while the probe waits for the GPU lock: it releases the lock, waits again, and starts only
        on a low reading taken under the lock (recorded as the start reading)."""
        from unittest import mock
        import score_teacher
        lv = lambda pct: {"some": {"total_us": 0, "avg10": pct, "avg60": pct}, "full": {"total_us": 0, "avg10": 0.0,
                                                                                     "avg60": 0.0}}
        readings = iter([lv(2.0), lv(40.0), lv(3.0), lv(4.0)] + [lv(1.0)] * 50)
        real, locks = score_teacher.gpu_lock, []

        def counting(label, wait):
            locks.append(label)
            return real(label, wait)

        with mock.patch.object(checks, "io_pressure", lambda: next(readings)), \
                mock.patch.object(checks, "IO_POLL_S", 0.01), mock.patch.object(checks, "IO_START_MAX", 10.0), \
                mock.patch.object(score_teacher, "gpu_lock", counting):
            r = self.probe(self.root / "probe-recheck")
        self.assertEqual(len(locks), 2)
        self.assertEqual(r["io_pressure"]["start"]["some"]["avg60"], 4.0)
        self.assertEqual(self.calls, ["teacher", "b0", "split"])

    def test_io_share_is_the_stall_time_over_the_wall(self):
        a = {"some": {"total_us": 1_000_000, "avg10": 0.0}, "full": {"total_us": 0, "avg10": 0.0}}
        b = {"some": {"total_us": 3_000_000, "avg10": 9.0}, "full": {"total_us": 1_000_000, "avg10": 1.0}}
        self.assertEqual(checks.io_share(a, b, 4.0), {"some": 0.5, "full": 0.25})

    def test_completed_probe_is_returned_unchanged(self):
        out = self.root / "probe"
        first = self.probe(out)
        again = self.probe(out)
        self.assertEqual(first, again)
        self.assertEqual(self.calls, ["teacher", "b0", "split"])


class SignalCost(unittest.TestCase):
    """The probe's time split also decodes its clips without the signals: their cost and the share of changed labels."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ctx = relabel_root(self.root)
        self.ctx.__enter__()
        rows = [clip_row(f"s:{i:03d}", "s", "alpha bravo", "human", seconds=1.0 + i / 10) for i in range(12)]
        checks.write_rows_as_shards(rows, self.root / "pool", 6)
        c2.write_json(self.root / "pool" / "materialized.json", {})
        c2.init_run(self.root / "run", pool_dir=self.root / "pool", engine={"kind": "fake", "confidence": True},
                    group_shards=2, b0_batch_seconds=60.0, declared_vram_gb=None)

    def tearDown(self):
        self.ctx.__exit__(None, None, None)
        self.tmp.cleanup()

    def test_split_reports_the_signal_cost_and_changed_labels(self):
        out = checks.split_step(self.root / "run")
        sc = out["signals"]
        self.assertEqual(sc["clips"], 12)
        self.assertEqual(sc["labels_changed"], 0)
        self.assertEqual(sc["labels_changed_share"], 0.0)
        self.assertEqual(sc["repeat_changed"], 0)
        self.assertGreater(sc["with_s"], 0)
        self.assertGreater(sc["without_s"], 0)
        self.assertIn("cost_share", sc)
        self.assertTrue((self.root / "run" / "split.json").exists())

    def test_split_warms_up_on_all_its_clips_before_timing(self):
        from unittest import mock
        orig, calls = c2.FakeTeacher.run, []

        def run(self_, clips, max_tokens=None, confidence=None):
            calls.append((len(clips), self_.confidence if confidence is None else confidence))
            return orig(self_, clips, max_tokens=max_tokens, confidence=confidence)

        rows = [clip_row(f"w:{i:03d}", "w", "alpha", "human", seconds=1.0 + i / 20) for i in range(40)]
        checks.write_rows_as_shards(rows, self.root / "pool40", 20)
        c2.write_json(self.root / "pool40" / "materialized.json", {})
        c2.init_run(self.root / "run40", pool_dir=self.root / "pool40", engine={"kind": "fake", "confidence": True},
                    group_shards=2, b0_batch_seconds=60.0, declared_vram_gb=None)
        with mock.patch.object(c2.FakeTeacher, "run", run):
            checks.split_step(self.root / "run40")
        first_timed = next(k for k, (n, _) in enumerate(calls) if n == 40)
        self.assertGreaterEqual(sum(n for n, conf in calls[:first_timed] if conf), 40)
        self.assertTrue(any(not conf for _, conf in calls[:first_timed]))

    def test_split_counts_labels_that_change_without_the_signals(self):
        from unittest import mock
        orig = c2.FakeTeacher.run

        def run(self_, clips, max_tokens=None, confidence=None):
            out = orig(self_, clips, max_tokens=max_tokens, confidence=confidence)
            on = self_.confidence if confidence is None else confidence
            return out if on else [("changed", h) + tuple(rest) for _, h, *rest in out[:1]] + out[1:]

        with mock.patch.object(c2.FakeTeacher, "run", run):
            sc = checks.split_step(self.root / "run")["signals"]
        self.assertEqual(sc["labels_changed"], 1)
        self.assertAlmostEqual(sc["labels_changed_share"], 1 / 12)
        self.assertEqual(sc["repeat_changed"], 0)

    def test_split_reports_run_to_run_noise_at_the_production_setting(self):
        from unittest import mock
        orig, calls = c2.FakeTeacher.run, []

        def run(self_, clips, max_tokens=None, confidence=None):
            out = orig(self_, clips, max_tokens=max_tokens, confidence=confidence)
            on = self_.confidence if confidence is None else confidence
            if on and max_tokens is None and len(clips) == 12:
                calls.append(1)
                if len(calls) == 2:                       # the second full pass with the signals differs on 2 clips
                    return [("noise",) + tuple(o[1:]) for o in out[:2]] + out[2:]
            return out

        with mock.patch.object(c2.FakeTeacher, "run", run):
            sc = checks.split_step(self.root / "run")["signals"]
        self.assertEqual(sc["repeat_changed"], 2)
        self.assertEqual(sc["labels_changed"], 0)


class Drops(unittest.TestCase):
    def test_thresholds_by_ids_and_occurrences(self):
        labels = {i: ("s", True, ()) for i in range(95)}
        labels |= {100 + i: ("s", False, ("human_wer_gt_0.5",)) for i in range(5)}
        t = checks.drop_table(labels, {i: 1 for i in labels})
        self.assertTrue(t["s"]["ok"])
        self.assertAlmostEqual(t["s"]["drop_rate_ids"], 0.05)
        occ = {i: 1 for i in labels} | {100: 2}       # by occurrences: 6/101 > 5%
        self.assertFalse(checks.drop_table(labels, occ)["s"]["ok"])
        labels = {i: ("s", True, ()) for i in range(97)} | {200 + i: ("s", False, ("ngram_loop",)) for i in range(3)}
        t = checks.drop_table(labels, {i: 1 for i in labels})
        self.assertAlmostEqual(t["s"]["hallucination_rate_ids"], 0.03)
        self.assertFalse(t["s"]["ok"])

    # --- per-source hallucination limit (labeling plan "Gate to stage 3" item 2, amended 2026-10-26) ---

    @staticmethod
    def source(src: str, n: int, hallucination: int, other: int = 0, base: int = 0) -> dict:
        """n ids of src: `hallucination` rejected by the guard (B0 rule), `other` by another filter, the rest kept."""
        out = {}
        for j in range(n):
            reasons = ("b0_wer_gt_0.8",) if j < hallucination else \
                ("tokenizer_coverage",) if j < hallucination + other else ()
            out[base + j] = (src, not reasons, reasons)
        return out

    def test_the_limits(self):
        self.assertEqual((checks.MAX_DROP, checks.MAX_HALLUCINATION), (0.05, 0.02))
        self.assertEqual(checks.HALLUCINATION_LIMIT_BY_SOURCE, {"yodas": 0.04})

    def test_yodas_hallucination_limit_is_4_percent(self):
        labels = self.source("yodas", 1000, 36)                                    # 3.6%
        labels |= self.source("yodas-tech", 1000, 21, base=10_000)                 # 2.1%
        labels |= self.source("yodas-med", 1000, 36, base=20_000)                  # 3.6%
        labels |= self.source("librispeech", 1000, 19, base=30_000)                # 1.9%
        labels |= self.source("ami", 1000, 21, base=40_000)                        # 2.1%
        t = checks.drop_table(labels, {i: 1 for i in labels})
        self.assertAlmostEqual(t["yodas"]["hallucination_rate_ids"], 0.036)
        self.assertEqual({s: c["ok"] for s, c in t.items()},
                         {"yodas": True, "yodas-tech": False, "yodas-med": False, "librispeech": True, "ami": False})
        self.assertEqual({s: c["hallucination_limit"] for s, c in t.items()},
                         {"yodas": 0.04, "yodas-tech": 0.02, "yodas-med": 0.02, "librispeech": 0.02, "ami": 0.02})
        self.assertFalse(checks.drop_table(self.source("yodas", 1000, 41), {})["yodas"]["ok"])          # 4.1%
        self.assertTrue(checks.drop_table(self.source("yodas", 1000, 40), {})["yodas"]["ok"])           # 4.0%

    def test_yodas_limit_by_occurrences_too(self):
        labels = self.source("yodas", 1000, 36)
        occ = {i: 1 for i in labels} | {0: 6}                       # by occurrences: 41/1005 = 4.08%
        t = checks.drop_table(labels, occ)
        self.assertGreater(t["yodas"]["hallucination_rate_occurrences"], 0.04)
        self.assertFalse(t["yodas"]["ok"])

    def test_yodas_total_drop_stays_5_percent(self):
        t = checks.drop_table(self.source("yodas", 1000, 30, other=21), {})       # guard 3.0%, total 5.1%
        self.assertAlmostEqual(t["yodas"]["drop_rate_ids"], 0.051)
        self.assertFalse(t["yodas"]["ok"])
        t = checks.drop_table(self.source("yodas", 1000, 30, other=20), {})       # total 5.0%
        self.assertTrue(t["yodas"]["ok"])


class Gate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.ctx = relabel_root(cls.root)
        cls.ctx.__enter__()
        cls.world = World(cls.root)
        cls.pin = cls.world.pinned(workers=1, batch_seconds=10.0)
        cls.make = staticmethod(lambda: config.make_stream(cls.pin, 0))
        cls.pool = cls.root / "pool-audio"
        replay.replay(cls.pin, cls.pool, total_batches=12, shard_batches=4, stream_factory=cls.make)
        refs = {}
        for k in shards.committed(cls.pool):
            for r in shards.read_audio(shards.paths(cls.pool, k)["audio"]):
                refs[r["pcm_sha256"]] = r["reference"] or ""
        cls.labels = cls.root / "labels"
        engine = {"kind": "fake"}
        c2.init_run(cls.labels, pool_dir=cls.pool, engine=engine, group_shards=2, b0_batch_seconds=60.0,
                    declared_vram_gb=None)
        c2.teacher_step(cls.labels, engine=RefTeacher(refs))
        c2.b0_step(cls.labels, labeler=RefB0(refs))
        c2.merge_step(cls.labels, coverage=lambda t: True)
        cls.cons = cls.root / "cons.json"
        cls.cons.write_text(json.dumps(cons(c2.contract(c2.load_run(cls.labels))) | {"sets": {}}))

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)
        cls.tmp.cleanup()

    def gate(self, run=None, out="gate"):
        return checks.gate(self.pin, self.pool, run or self.labels, self.cons, self.root / out, total_batches=12,
                           stream_factory=self.make)

    def test_passes_on_a_complete_consistent_pass(self):
        r = self.gate()
        self.assertTrue(r["passed"], (r["items"], r["consistency"]))
        self.assertTrue(r["independent_replay"]["same_sequence"])
        th = r["thresholds"]
        self.assertEqual((th["drop"], th["hallucination"], th["hallucination_by_source"]), (0.05, 0.02, {"yodas": 0.04}))
        self.assertTrue(r["drops"])
        for src, c in r["drops"].items():
            self.assertEqual(c["hallucination_limit"], 0.04 if src == "yodas" else 0.02, src)
        files = sorted((self.root / "gate" / "spotcheck").glob("*.txt"))
        self.assertTrue(files)
        text = files[0].read_text()
        self.assertIn("T*: ", text)
        self.assertIn("B0: ", text)

    def test_missing_record_is_reported(self):
        import shutil
        run = self.root / "labels-missing"
        shutil.copytree(self.labels, run)
        p = sorted((run / "shards").glob("*.labels.parquet"))[1]
        rows = table(p)[1:]
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(rows), p)
        r = self.gate(run, "gate-missing")
        self.assertFalse(r["items"]["completeness"])
        self.assertTrue(any("without a record" in x for x in r["completeness"]["problems"]))

    def test_independent_replay_detects_a_different_sequence(self):
        import shutil
        pool = self.root / "pool-tamper"
        shutil.copytree(self.pool, pool)
        p = shards.paths(pool, 0)["journal"]
        rows = shards.read_journal(p)
        rows[0], rows[1] = rows[1], rows[0]
        shards.write_journal(p, rows)
        r = checks.independent_replay(self.pin, pool, 4, stream_factory=self.make)
        self.assertFalse(r["ok"])
        self.assertTrue(checks.independent_replay(self.pin, self.pool, 4, stream_factory=self.make)["ok"])

    def test_spot_check_shows_a_confidence_summary(self):
        from test_labeling_c2 import SigTeacher
        run = self.root / "labels-sig"
        c2.init_run(run, pool_dir=self.pool, engine={"kind": "fake", "confidence": True}, group_shards=2,
                    b0_batch_seconds=60.0, declared_vram_gb=None)
        refs = {}
        for k in shards.committed(self.pool):
            for r in shards.read_audio(shards.paths(self.pool, k)["audio"]):
                refs[r["pcm_sha256"]] = r["reference"] or ""
        c2.teacher_step(run, engine=SigTeacher(refs))
        c2.b0_step(run, labeler=RefB0(refs))
        c2.merge_step(run, coverage=lambda t: True)
        checks.spot_check(run, self.root / "s-sig", per_source=2)
        recs = {r["id"]: r for p in sorted((run / "shards").glob("*.labels.parquet")) for r in table(p)}
        files = sorted((self.root / "s-sig" / "spotcheck").glob("*.txt"))
        self.assertTrue(files)
        for f in files:
            blocks = f.read_text().split("\n## ")[1:]
            self.assertTrue(blocks)
            for b in blocks:
                r = recs[b.split(" | ")[0]]
                lp, ru = np.array(r["t_logprob"], np.float64), np.array(r["t_runner_up"], np.float64)
                want = checks.confidence_summary(r)
                self.assertEqual(want["tokens"], len(r["t_token_ids"]))
                self.assertAlmostEqual(want["mean_logprob"], lp.mean())
                self.assertAlmostEqual(want["min_logprob"], lp.min())
                self.assertAlmostEqual(want["mean_margin"], (lp - ru).mean())
                self.assertIn(f"conf: {want['tokens']} tokens", b)
        # labels without signals: no summary line
        checks.spot_check(self.labels, self.root / "s-nosig", per_source=2)
        for f in (self.root / "s-nosig" / "spotcheck").glob("*.txt"):
            self.assertNotIn("conf:", f.read_text())

    def test_spot_check_is_seeded_and_capped(self):
        a = checks.spot_check(self.labels, self.root / "s1", per_source=2)
        b = checks.spot_check(self.labels, self.root / "s2", per_source=2)
        self.assertEqual(a, b)
        self.assertTrue(all(n <= 2 for n in a.values()))
        for f in (self.root / "s1" / "spotcheck").glob("*.txt"):
            self.assertEqual(f.read_text(), (self.root / "s2" / "spotcheck" / f.name).read_text())

    def test_consistency_under_another_contract_or_limited_does_not_count(self):
        k = c2.contract(c2.load_run(self.labels))
        for i, bad in enumerate((cons(k | {"group_shards": 1}), cons(k, full=False), cons(k, passed=False))):
            other = self.root / f"cons-other-{i}.json"
            other.write_text(json.dumps(bad | {"sets": {}}))
            r = checks.gate(self.pin, self.pool, self.labels, other, self.root / f"gate-other-{i}", total_batches=12,
                            stream_factory=self.make, audio_hash=False)
            self.assertFalse(r["items"]["consistency"])
            self.assertFalse(r["passed"])

    def test_a_refiltered_run_passes_and_a_run_under_older_filters_does_not(self):
        """A run labeled under filters version 1, with its checks run under version 1 (as production's), fails the gate
        only on item "filters"; re-filtered (c2.refilter) it passes under the same checks."""
        refs = {}
        for k in shards.committed(self.pool):
            for r in shards.read_audio(shards.paths(self.pool, k)["audio"]):
                refs[r["pcm_sha256"]] = r["reference"] or ""
        v1 = self.root / "labels-v1"
        c2.init_run(v1, pool_dir=self.pool, engine={"kind": "fake"}, group_shards=2, b0_batch_seconds=60.0,
                    declared_vram_gb=None, extra={"filters_version": 1})
        c2.teacher_step(v1, engine=RefTeacher(refs))
        c2.b0_step(v1, labeler=RefB0(refs))
        c2.merge_step(v1, coverage=lambda t: True)
        self.assertTrue(c2.finish(v1)["complete"])
        cons_v1 = self.root / "cons-v1.json"
        cons_v1.write_text(json.dumps(cons(c2.contract(c2.load_run(v1))) | {"sets": {}}))
        old = checks.gate(self.pin, self.pool, v1, cons_v1, self.root / "gate-v1", total_batches=12,
                          stream_factory=self.make, audio_hash=False)
        self.assertEqual({k for k, v in old["items"].items() if not v}, {"filters"})
        self.assertFalse(old["passed"])
        self.assertEqual((old["filters"]["run"], old["filters"]["current"]), (1, filters.VERSION))
        out = self.root / f"labels-v1-f{filters.VERSION}"
        c2.refilter(v1, out, coverage=lambda t: True)
        r = checks.gate(self.pin, self.pool, out, cons_v1, self.root / "gate-v1-refiltered", total_batches=12,
                        stream_factory=self.make, audio_hash=False)
        self.assertTrue(r["passed"], (r["items"], r["consistency"]))
        self.assertTrue(r["completeness"]["ok"])
        self.assertEqual(r["filters"]["run"], filters.VERSION)
        self.assertEqual(r["filters"]["refilter"]["source_filters_version"], 1)

    def test_tampered_c1_audio_fails_completeness(self):
        import shutil
        pool = self.root / "pool-audio-tamper"
        shutil.copytree(self.pool, pool)
        p = shards.paths(pool, 1)["audio"]
        data = bytearray(p.read_bytes())
        data[-100] ^= 0xFF
        p.write_bytes(bytes(data))
        res, _, _ = checks.completeness(pool, self.labels, 12)
        self.assertFalse(res["ok"])
        self.assertTrue(any("SHA-256" in x for x in res["problems"]))


if __name__ == "__main__":
    unittest.main()
