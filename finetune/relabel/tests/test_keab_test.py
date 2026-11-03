"""Contract tests (CPU): Kea-B′'s test-set scoring (keab-prime-test plan v1.2 "Code and review").

- each precheck item's refusal (stage 4 finished, identity, code, corpus, normalizer, reused arms, training-data
  hygiene: the Common Voice store bound to the run and the selection, a held-out speaker with a changed id, a held-out
  sentence and clip path, the AMI test meetings);
- the dev gate's tolerances (mean off by more than 0.02 points fails; mean within but one set off by more than 0.10
  fails; exactly at the tolerance passes) and each mode's dev check (x1 against stage 4's recorded x1 scores, fp
  validated on MB1 first); a structurally invalid row fails the gate;
- --test refusals (no gate, failing gate, --limit gate, a gate with other identity, hashes, manifests or settings) and
  --test --limit rejected before any model loads;
- resume: a set_done set is not decoded again, a set without one is (also after its file was written), a changed
  done file refuses, complete refuses; --summarize makes no decoding calls;
- the report: its arithmetic against known values, its refusals, the structurally invalid row, the sanity checks.

No GPU and no real decoding: keab_test_fixtures.World builds the inputs and fakes the model loads and decoders.
"""
from __future__ import annotations

import keab_test_fixtures as fx  # noqa: I001  first: the fixture world (imports keab_test, train_keab)

import contextlib
import io
import json
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

import binary_paths
import evaluate
import keab_report_test as kr
import keab_test as kt
import keab_test_paths as tp
import paths
import train_keab as tk

TEST, DEV = fx.TEST, fx.DEV
X1 = "Kea-B' K-x1"
FP = "Kea-B' K-fp"


def run_main(argv: list[str]) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        code = kt.main(argv)
    return code, out.getvalue()


def run_report() -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = kr.main()
    return code, out.getvalue()


class WorldCase(unittest.TestCase):
    def setUp(self) -> None:
        self.w = fx.World()
        self.addCleanup(self.w.close)
        self.enterContext(self.w.installed())

    def refused(self, argv: list[str], word: str) -> SystemExit:
        with self.assertRaises(SystemExit) as caught:
            run_main(argv)
        self.assertIn(word, str(caught.exception))
        return caught.exception

    def report_refused(self, word: str) -> None:
        with self.assertRaises(SystemExit) as caught:
            run_report()
        self.assertIn("refused", str(caught.exception))
        self.assertIn(word, str(caught.exception))

    def failing(self) -> list[tuple[int, str]]:
        result = kt.precheck()
        return [(c["item"], c["check"]) for c in result["checks"] if not c["ok"]]

    def assert_item_fails(self, item: int, word: str) -> None:
        result = kt.precheck()
        failing = [(c["item"], c["check"]) for c in result["checks"] if not c["ok"]]
        self.assertFalse(result["pass"])
        self.assertTrue(any(i == item and word in name for i, name in failing),
                        f"no failing item {item} check naming {word!r}: {failing}")

    def assert_only_items_fail(self, items: set[int]) -> None:
        self.assertEqual({i for i, _ in self.failing()}, items)

    def gate(self, mode: str) -> dict:
        return fx.read_json(tp.dev_check_dir(mode) / tp.GATE)

    def dev_check(self, mode: str, *extra: str) -> int:
        return run_main(["--dev-check", mode, *extra])[0]

    def complete(self, *modes: str) -> None:
        for mode in modes:
            self.assertEqual(self.dev_check(mode), 0)
            self.assertEqual(run_main(["--test", mode])[0], 0)

    def edit_gate(self, mode: str, fn) -> None:
        path = tp.dev_check_dir(mode) / tp.GATE
        gate = fx.read_json(path)
        fn(gate)
        fx.write_json(path, gate)


# ============================================================================ pinned constants

class PinnedConstantsTest(unittest.TestCase):
    def test_identity_pins_are_the_plans(self) -> None:
        self.assertEqual(tp.RUN_NAME, "b100-main")
        self.assertTrue(tp.VIEW_SHA256.startswith("ec05526e"))
        self.assertTrue(tp.BUNDLE_SHA256.startswith("c00230f8"))
        self.assertTrue(tp.CV_CLIP_POOL.startswith("8566b62d"))
        self.assertTrue(tp.MB1_EXPORT_SHA256.startswith("9873d2f1"))
        self.assertTrue(tp.M1_EXPORT_SHA256.startswith("a287e977"))
        self.assertTrue(tp.EVALUATE_SHA256.startswith("b3136a80"))
        self.assertEqual({k: tp.EXPECT_SUMMARY[k] for k in ("run_name", "steps", "max_steps", "lr", "seed")},
                         {"run_name": "b100-main", "steps": 97137, "max_steps": 97137, "lr": 2e-5, "seed": 20261031})
        for rel in ("export/view/export.safetensors", "export/view/manifest.json", "export/bundle/bundle.safetensors",
                    "export/bundle/manifest.json", "model.pt", "summary.json", "config.json", *[
                        f"export/view/{t}" for t in tp.VIEW_TOKENIZER]):
            self.assertRegex(tp.PINNED_FILES[rel], r"^[0-9a-f]{64}$")
        self.assertEqual(set(tp.DEV_X1_FILES), set(DEV))
        self.assertAlmostEqual(tp.MB1_DEV_MEAN * 100, 6.3120, places=4)
        self.assertEqual(tp.BUNDLE_TENSOR_BYTES, 91_236_520)

    def test_fixed_output_locations(self) -> None:
        root = tp.KEAB / "test"
        self.assertEqual(tp.TEST_ROOT, root)
        self.assertEqual(tp.test_dir("x1"), root / "x1")
        self.assertEqual(tp.test_dir("fp"), root / "fp")
        self.assertEqual(tp.dev_check_dir("x1"), root / "x1-dev-check")
        self.assertEqual(tp.dev_check_dir("fp"), root / "fp-dev-check")
        self.assertEqual(tp.attempts_path("x1"), root / "x1.attempts.jsonl")
        smoke = tp.dev_check_dir("x1", 5)
        for production in (tp.test_dir("x1"), tp.test_dir("fp"), tp.dev_check_dir("x1"), tp.dev_check_dir("fp")):
            self.assertFalse(smoke.is_relative_to(production))
        with self.assertRaises(ValueError):
            tp.test_dir("x3")
        self.assertEqual(tp.RESULTS_MD.name, "RESULTS-keab-prime-test.md")
        self.assertEqual(tp.TEST_JSON, root / "test.json")

    def test_scoring_constants(self) -> None:
        self.assertEqual(tp.DECODER_GRAPHS_MODE, binary_paths.DECODER_GRAPHS_MODE)
        self.assertEqual((tp.BATCH_SIZE, tp.MAX_BATCH_SECONDS), (64, 1200.0))
        self.assertEqual(tp.MEAN_TOLERANCE, tp.points_to_fraction(tp.MEAN_TOLERANCE_POINTS))
        self.assertEqual(tp.SET_TOLERANCE, tp.points_to_fraction(tp.SET_TOLERANCE_POINTS))
        self.assertEqual((tp.MEAN_TOLERANCE_POINTS, tp.SET_TOLERANCE_POINTS), (0.02, 0.10))

    def test_scorer_files_cover_the_plans_sources(self) -> None:
        keys = set(tp.scorer_keys())
        for key in ("relabel/keab_test.py", "relabel/keab_report_test.py", "relabel/train_keab.py",
                    "relabel/training/deployed.py", "relabel/training/eightbit.py", "relabel/keab_step0.py",
                    "parakeet-binary/binary_export.py", "parakeet-binary/binary_train.py",
                    "parakeet-ternary/evaluate.py", "whisper-ternary/wer.py", tp.NORMALIZER_MODULE, tp.NORMALIZER_JSON,
                    "parakeet-binary/gpu1bit/runtime.py", "parakeet-binary/gpu1bit/shared/fulldev.py",
                    "parakeet-binary/gpu1bit/actprec/parts.py", "parakeet-binary/gpu1bit/int8_mma/cuda.py",
                    *tp.WIP_PARTS):
            self.assertIn(key, keys)
            self.assertTrue(tp.scorer_path(key).is_file(), key)
        self.assertNotIn("relabel/keab_test_paths.py", keys)     # its hash is in the plan's Status line
        self.assertFalse(any("/tests/" in k for k in keys))

    def test_scorer_hashes_are_pinned_after_review(self) -> None:
        self.assertTrue(tp.SCORER_HASHES)          # pinned after code review (keab_test.py --scorer-hashes)

    def test_committed_files_hold_no_machine_paths(self) -> None:
        here = Path(kt.__file__).resolve().parent
        for f in (here / "keab_test.py", here / "keab_test_paths.py", here / "keab_report_test.py",
                  here / "tests" / "test_keab_test.py", here / "tests" / "keab_test_fixtures.py"):
            text = f.read_text()
            self.assertIsNone(re.search("/" + "mnt/|/" + "home/", text), f.name)   # split: this file is checked too


# ============================================================================ precheck

class PrecheckTest(WorldCase):
    def test_the_fixture_world_passes_and_writes_its_record(self) -> None:
        result = kt.precheck()
        self.assertTrue(result["pass"], [c for c in result["checks"] if not c["ok"]])
        self.assertEqual({c["item"] for c in result["checks"]}, set(range(1, 8)))
        code, out = run_main(["--precheck"])
        self.assertEqual(code, 0)
        self.assertIn("ALL PASS", out)
        self.assertTrue(fx.read_json(tp.PRECHECK)["pass"])
        self.assertEqual(result["facts"]["hygiene"]["cv_heldout"], fx.CV_COUNTS)

    def test_a_failing_precheck_exits_nonzero(self) -> None:
        self.w.jobs = ["relabel-keab-score-06"]
        code, out = run_main(["--precheck"])
        self.assertEqual(code, 1)
        self.assertIn("PRECHECK FAILED", out)

    # item 1
    def test_a_pending_stage4_chain_item(self) -> None:
        self.w.pending = ["score b100-main x3"]
        self.assert_item_fails(1, "chain items")

    def test_stage4_gate_not_pass(self) -> None:
        self.w.edit_json(self.w.keab / "results.json", lambda r: r["gate"]["decision"].__setitem__("pass", False))
        self.assert_item_fails(1, "decision PASS")

    def test_stage4_report_missing(self) -> None:
        (self.w.plans / "RESULTS-keab-prime.md").unlink()
        self.assert_item_fails(1, "RESULTS-keab-prime.md")

    def test_an_active_stage4_job(self) -> None:
        self.w.jobs = ["relabel-keab-score-06"]
        self.assert_item_fails(1, "job active")

    def test_a_failed_job_query(self) -> None:
        self.w.jobs = None
        self.assert_item_fails(1, "job active")

    # item 2
    def test_each_pinned_file_with_a_changed_hash(self) -> None:
        for rel in tp.PINNED_FILES:
            with self.subTest(rel=rel):
                saved = dict(self.w.pins["PINNED_FILES"])
                self.w.pins["PINNED_FILES"][rel] = "0" * 64
                self.assert_item_fails(2, rel)
                self.w.pins["PINNED_FILES"].clear()
                self.w.pins["PINNED_FILES"].update(saved)

    def test_summary_identity_fields(self) -> None:
        for key, bad in (("steps", 97136), ("max_steps", 97138), ("lr", 2e-4), ("seed", 1), ("run_name", "b50-main")):
            with self.subTest(key=key):
                good = fx.read_json(self.w.run / "summary.json")[key]
                self.w.edit_json(self.w.run / "summary.json", lambda s: s.__setitem__(key, bad), repin=False)
                with mock.patch.object(self.w, "write_summary", lambda: None):
                    self.w.repin()
                self.assert_item_fails(2, f"summary.{key}")
                self.w.edit_json(self.w.run / "summary.json", lambda s: s.__setitem__(key, good))

    def test_the_view_must_record_its_bundle_and_model(self) -> None:
        self.w.edit_json(self.w.view / "manifest.json",
                         lambda m: m["extra"]["keab"].__setitem__("bundle_sha256", "f" * 64))
        self.assert_item_fails(2, "records its bundle")
        self.w.edit_json(self.w.view / "manifest.json", lambda m: m["extra"].__setitem__("model_sha256", "f" * 64))
        self.assert_item_fails(2, "training model")

    def test_an_extra_file_in_the_view(self) -> None:
        (self.w.view / "notes.txt").write_text("x")
        self.assert_item_fails(2, "file set exact")

    def test_runtime_record_of_another_view(self) -> None:
        self.w.edit_json(self.w.scores_x1 / "runtime.json",
                         lambda r: r["runtime"].__setitem__("export_sha256", "f" * 64))
        self.assert_item_fails(2, "loaded this view")

    def test_a_changed_recorded_x1_dev_score(self) -> None:
        self.w.pins["DEV_X1_FILES"]["ami_dev"] = "0" * 64
        self.assert_item_fails(2, "ami_dev.json is the pinned file")

    def test_a_recorded_x1_dev_score_of_a_limit_run(self) -> None:
        self.w.edit_json(self.w.scores_x1 / "yodas_dev.json", lambda d: d["config"].__setitem__("limit", 2))
        self.assert_item_fails(2, "full set")

    # item 3
    def test_scorer_hashes_not_pinned(self) -> None:
        with mock.patch.object(tp, "SCORER_HASHES", {}):
            result = kt.precheck()
        failing = [c for c in result["checks"] if not c["ok"]]
        self.assertEqual({c["item"] for c in failing}, {3})
        self.assertTrue(any("not pinned yet" in c["detail"] for c in failing))

    def test_a_changed_scorer_file(self) -> None:
        tp.SCORER_HASHES["relabel/keab_test.py"] = "0" * 64
        self.assert_item_fails(3, "every scorer file matches")

    def test_scorer_hashes_missing_a_file(self) -> None:
        del tp.SCORER_HASHES["parakeet-binary/gpu1bit/runtime.py"]
        self.assert_item_fails(3, "covers exactly")

    def test_a_module_imported_from_another_path(self) -> None:
        fake = type(sys)("evaluate")
        fake.__file__ = str(tp.FINETUNE / "parakeet-binary" / "binary_paths.py")
        with mock.patch.dict(sys.modules, {"evaluate": fake}):
            self.assert_item_fails(3, "module evaluate")

    def test_a_gpu1bit_module_outside_the_pinned_files(self) -> None:
        stray = type(sys)("gpu1bit.stray")
        stray.__file__ = str(self.w.root / "stray.py")
        with mock.patch.dict(sys.modules, {"gpu1bit.stray": stray}):
            self.assert_item_fails(3, "gpu1bit / inference")

    def test_paths_hash_missing_from_the_plan_status(self) -> None:
        self.w.write_plan("0" * 64)
        self.assert_item_fails(3, "Status line")

    # item 4
    def test_a_test_manifest_differing_from_the_reused_arms(self) -> None:
        manifest = self.w.manifests / "test_earnings22.jsonl"
        manifest.write_text(manifest.read_text().replace("Mister", "Mr"))
        self.assert_item_fails(4, "earnings22: manifest SHA-256")

    def test_duplicate_utterance_ids(self) -> None:
        manifest = self.w.manifests / "test_gigaspeech.jsonl"
        lines = manifest.read_text().splitlines()
        manifest.write_text("\n".join(lines + [lines[0]]) + "\n")
        self.assert_item_fails(4, "IDs unique")

    # item 5
    def test_a_saved_hyp_norm_the_current_normalizer_does_not_reproduce(self) -> None:
        self.w.edit_json(self.w.arms["MB1"] / "voxpopuli.json",
                         lambda d: d["records"][0].__setitem__("hyp_norm", "something else"))
        self.w.write_mb1_attempts()
        self.w.repin()
        self.assert_item_fails(5, "voxpopuli")

    # item 6
    def test_a_reused_arm_file_changed(self) -> None:
        self.w.pins["REUSED_FILES"]["M1"]["ami.json"] = "0" * 64
        self.assert_item_fails(6, "M1: the pinned output files")

    def test_a_reused_arm_missing_a_set(self) -> None:
        (self.w.arms["B0"] / "spgispeech.json").unlink()
        self.assert_item_fails(6, "B0: all 8 test sets")

    def test_a_reused_arm_limit_run(self) -> None:
        self.w.edit_json(self.w.arms["M1"] / "ami.json", lambda d: d.__setitem__("limit", 2))
        self.assert_item_fails(6, "M1: all 8 test sets")

    def test_mb1_test_scored_another_export(self) -> None:
        self.w.edit_json(self.w.arms["MB1"] / "summary.json", lambda s: s["source"].__setitem__("sha256", "f" * 64))
        self.assert_item_fails(6, "MB1's pinned export")

    def test_m1_test_scored_another_export(self) -> None:
        self.w.edit_json(self.w.arms["M1"] / "common_voice.json", lambda d: d["source"].__setitem__("sha256", "f" * 64))
        self.assert_item_fails(6, "M1's pinned export")

    def test_b0_not_the_pinned_pretrained_model(self) -> None:
        self.w.edit_json(self.w.arms["B0"] / "ami.json", lambda d: d["source"].__setitem__("kind", "export"))
        self.assert_item_fails(6, "pretrained model")

    def test_mb1_attempt_not_complete(self) -> None:
        lines = self.w.mb1_attempts.read_text().splitlines()[:-1]
        self.w.mb1_attempts.write_text("\n".join(lines) + "\n")
        self.w.repin()
        self.assert_item_fails(6, "one complete attempt")

    # item 7: Common Voice
    def test_cv_store_not_the_runs_clip_pool(self) -> None:
        store = self.w.cv / "store" / "clips.jsonl"
        store.write_text(store.read_text() + json.dumps({"id": "cv27-en:v1", "sentence": "x"}) + "\n")
        self.assert_item_fails(7, "clip_pool")

    def test_cv_store_meta_bound_to_another_selection(self) -> None:
        self.w.edit_json(self.w.cv / "store" / "clips.meta.json", lambda m: m.__setitem__("selection_sha256", "f" * 64))
        self.assert_item_fails(7, "selection_sha256")

    def test_a_store_clip_that_is_not_an_accepted_selection_row(self) -> None:
        self.w.write_selection(accepted=("v1.mp3", "v2.mp3"))
        store = self.w.cv / "store" / "clips.jsonl"
        store.write_text(store.read_text() + json.dumps({"id": "cv27-en:o9", "sentence": "x"}) + "\n")
        self.w.edit_json(self.w.cv / "store" / "clips.meta.json",
                         lambda m: m.__setitem__("clips_sha256", fx.sha(store)), repin=False)
        self.w.repin()
        self.assert_item_fails(7, "accepted selection row")

    def test_a_store_sentence_other_than_the_selection_rows(self) -> None:
        self.w.write_selection(store_sentence={"cv27-en:v1": "A sentence the selection never judged."})
        self.w.repin()
        self.assert_item_fails(7, "same sentence")

    def test_a_held_out_speaker_with_a_changed_id(self) -> None:
        """v3.mp3 is read by spkNew, the client_id this release records for ESB clip e1.mp3, whose ESB id is esbOld."""
        self.w.write_selection(accepted=("v1.mp3", "v2.mp3", "v3.mp3"))
        self.w.repin()
        result = kt.precheck()
        failing = [(c["item"], c["check"]) for c in result["checks"] if not c["ok"]]
        self.assertEqual(failing, [(7, "no Common Voice store clip with a held-out speaker")])
        self.assertEqual(result["facts"]["hygiene"]["cv_heldout"]["esb_paths_with_changed_id"], 1)

    def test_a_held_out_speaker_of_the_releases_test_split(self) -> None:
        self.w.write_selection(accepted=("v1.mp3", "v2.mp3", "v5.mp3"))
        self.w.repin()
        self.assertEqual(self.failing(), [(7, "no Common Voice store clip with a held-out speaker")])

    def test_a_held_out_sentence_through_its_filters_v3_form(self) -> None:
        """v4's sentence normalizes without its bracketed word; its filters v3 text keeps it and is ESB's sentence."""
        self.assertNotEqual(evaluate.normalize("(Have) many acquaintances, but few friends."),
                            evaluate.normalize(fx.ESB[0][1]))
        self.w.write_selection(accepted=("v1.mp3", "v2.mp3", "v4.mp3"))
        self.w.repin()
        self.assertEqual(self.failing(), [(7, "no Common Voice store clip with a held-out sentence")])

    def test_a_held_out_clip_path(self) -> None:
        self.w.write_selection(accepted=("v1.mp3", "v2.mp3", "t1.mp3"))
        self.w.repin()
        self.assertIn((7, "no Common Voice store clip with a held-out clip"), self.failing())

    def test_recomputed_held_out_counts_must_equal_the_recorded_ones(self) -> None:
        self.w.write_selection(counts={**fx.CV_COUNTS, "held_speakers": 5})
        self.w.repin()
        self.assert_item_fails(7, "held-out counts")

    def test_a_release_tsv_changed_since_the_selection(self) -> None:
        tsv = self.w.raw / "other.tsv"
        tsv.write_text(tsv.read_text() + "spkZ\to2.mp3\tLate addition.\t\n")
        self.assert_item_fails(7, "six tsvs")

    # item 7: AMI
    def test_an_ami_test_meeting_in_the_pair_store(self) -> None:
        self.w.build_ami(meetings=("ES2002a", "EN2002a"))
        self.w.repin()
        self.assertEqual(self.failing(), [(7, "no ESB AMI test meeting in the AMI SDM pair store")])

    def test_ami_store_not_the_runs_clip_pool(self) -> None:
        self.w.ami_store.write_text(self.w.ami_store.read_text() + "\n")
        self.assert_item_fails(7, "AMI SDM store")


REAL_ACTIVE = kt.active_stage4_jobs


class JobQueryTest(unittest.TestCase):
    def test_only_active_stage4_jobs_other_than_its_own_run(self) -> None:
        runs = [{"name": "relabel-keab-score-01", "state": "succeeded", "run_dir": "/x/1"},
                {"name": "relabel-keab-test-x1", "state": "running", "run_dir": "/x/own"},
                {"name": "other-job", "state": "running", "run_dir": "/x/2"},
                {"name": "relabel-keab-main-31", "state": "queued", "run_dir": "/x/3"}]
        done = mock.Mock(returncode=0, stdout=json.dumps(runs))
        with mock.patch.object(kt.subprocess, "run", return_value=done), \
                mock.patch.dict(kt.os.environ, {"AGENT_JOBS_RUN_DIR": "/x/own"}):
            self.assertEqual(REAL_ACTIVE(), ["relabel-keab-main-31"])
        with mock.patch.object(kt.subprocess, "run", return_value=mock.Mock(returncode=1, stdout="")):
            self.assertIsNone(REAL_ACTIVE())


# ============================================================================ gate tolerances

class GateVerdictTest(unittest.TestCase):
    REC = dict(fx.X1_DEV)

    def test_identical_values_pass(self) -> None:
        v = kt.gate_verdict(dict(self.REC), self.REC)
        self.assertTrue(v["pass"])
        self.assertEqual(v["mean_diff"], 0.0)

    def test_mean_off_by_more_than_0_02_points_fails(self) -> None:
        v = kt.gate_verdict({n: w + 0.00021 for n, w in self.REC.items()}, self.REC)
        self.assertFalse(v["mean_within"])
        self.assertFalse(v["pass"])
        self.assertTrue(all(e["within"] for e in v["per_set"].values()))

    def test_one_set_off_by_more_than_0_10_points_fails_with_the_mean_within(self) -> None:
        moved = dict(self.REC, ami_dev=self.REC["ami_dev"] + 0.00101, yodas_dev=self.REC["yodas_dev"] - 0.00101)
        v = kt.gate_verdict(moved, self.REC)
        self.assertTrue(v["mean_within"])
        self.assertFalse(v["per_set"]["ami_dev"]["within"])
        self.assertFalse(v["pass"])

    def test_exactly_at_the_tolerances_passes(self) -> None:
        mean_edge = {n: w + 0.0002 for n, w in self.REC.items()}
        self.assertTrue(kt.gate_verdict(mean_edge, self.REC)["pass"])
        set_edge = dict(self.REC, ami_dev=self.REC["ami_dev"] + 0.0010, yodas_dev=self.REC["yodas_dev"] - 0.0010)
        v = kt.gate_verdict(set_edge, self.REC)
        self.assertTrue(v["per_set"]["ami_dev"]["within"] and v["per_set"]["yodas_dev"]["within"])
        self.assertTrue(v["pass"])

    def test_differences_in_points_and_the_tolerances_recorded(self) -> None:
        v = kt.gate_verdict(dict(self.REC, ami_dev=self.REC["ami_dev"] + 0.0005), self.REC)
        self.assertAlmostEqual(v["per_set"]["ami_dev"]["diff_points"], 0.05, places=9)
        self.assertAlmostEqual(v["mean_diff_points"], 0.01, places=9)
        self.assertEqual(v["tolerances"], {"mean_points": 0.02, "set_points": 0.10, "mean_fraction": 0.0002,
                                           "set_fraction": 0.0010})


# ============================================================================ dev checks

class DevCheckTest(WorldCase):
    def test_x1_dev_check_reproduces_the_recorded_scores(self) -> None:
        self.assertEqual(self.dev_check("x1"), 0)
        self.assertEqual([c["name"] for c in self.w.calls], DEV)
        self.assertTrue(all(c["lock_held"] and c["mode"] == "x1" for c in self.w.calls))
        self.assertEqual(len(self.w.loads), 1)
        self.assertEqual(self.w.loads[0]["path"].resolve(), self.w.view.resolve())
        self.assertTrue(self.w.loads[0]["lock_held"])
        self.assertEqual(self.w.closed, 1)
        gate = self.gate("x1")
        self.assertTrue(gate["pass"])
        self.assertTrue(gate["test_launch_gate"])
        self.assertIsNone(gate["limit"])
        self.assertEqual(gate["token_identity"]["rate"], 1.0)
        self.assertTrue(all(gate["same_records"].values()))
        self.assertEqual(gate["scorer"]["hashes"], fx.CURRENT_SCORER)
        self.assertEqual(gate["identity"]["view_sha256"], tp.VIEW_SHA256)
        out = tp.dev_check_dir("x1")
        for name in DEV:
            doc = fx.read_json(out / f"{name}.json")
            self.assertEqual(doc["mode"], "x1")
            self.assertEqual(doc["settings"], gate["settings"])
            self.assertTrue(doc["structurally_valid"])
            self.assertEqual(doc["manifest_sha256"], fx.sha(self.w.manifests / f"{name}.jsonl"))
        self.assertIsNone(gate["settings"]["env"]["CUBLAS_WORKSPACE_CONFIG"])

    def test_x1_dev_check_writes_the_gate_last(self) -> None:
        events = []
        real = kt.write_json
        with mock.patch.object(kt, "write_json", side_effect=lambda p, d: (events.append(Path(p).name), real(p, d))):
            self.assertEqual(self.dev_check("x1"), 0)
        self.assertEqual(events, [f"{n}.json" for n in DEV] + ["gate.json"])

    def test_x1_dev_wer_outside_the_tolerance_fails(self) -> None:
        self.w.x1_dev_wer["voxpopuli_dev"] += 0.0011
        self.assertEqual(self.dev_check("x1"), 1)
        gate = self.gate("x1")
        self.assertFalse(gate["pass"])
        self.assertFalse(gate["per_set"]["voxpopuli_dev"]["within"])

    def test_a_structurally_invalid_row_fails_the_x1_gate(self) -> None:
        self.w.invalid.add(("x1", "librispeech_dev_other"))
        self.assertEqual(self.dev_check("x1"), 1)
        gate = self.gate("x1")
        self.assertTrue(gate["within_tolerance"])
        self.assertFalse(gate["structurally_valid"])
        self.assertFalse(gate["pass"])

    def test_fp_dev_check_validates_on_mb1_first_then_scores_keab(self) -> None:
        self.assertEqual(self.dev_check("fp"), 0)
        self.assertEqual([load["label"] for load in self.w.loads], ["MB1", "keab"])
        self.assertTrue(all(load["lock_held"] for load in self.w.loads))
        self.assertEqual([(c["label"], c["name"]) for c in self.w.calls],
                         [("MB1", n) for n in DEV] + [("keab", n) for n in DEV])
        for call in self.w.calls:
            self.assertEqual((call["batch_size"], call["max_batch_seconds"], call["limit"]), (64, 1200.0, None))
            self.assertEqual(call["graphs"], "no_while_loops")
            self.assertTrue(call["lock_held"])
        gate = self.gate("fp")
        self.assertTrue(gate["pass"])
        self.assertTrue(gate["mb1"]["pass"])
        self.assertEqual(gate["keab"]["wer"], fx.KFP_DEV)
        self.assertEqual(gate["keab"]["pytorch_fake_quant"], fx.TORCH_DEV)
        self.assertIsNone(gate["keab"]["threshold"])
        doc = fx.read_json(tp.dev_check_dir("fp") / "keab" / "ami_dev.json")
        self.assertEqual(doc["decoder_graphs_mode"], {"before": "no_while_loops", "after": "no_while_loops"})
        self.assertEqual(doc["source"]["sha256"], tp.VIEW_SHA256)

    def test_fp_mb1_validation_outside_the_tolerance_fails_and_keab_is_not_scored(self) -> None:
        self.w.mb1_dev_wer = {n: w + 0.00025 for n, w in self.w.mb1_dev_wer.items()}
        self.assertEqual(self.dev_check("fp"), 1)
        self.assertEqual([load["label"] for load in self.w.loads], ["MB1"])
        gate = self.gate("fp")
        self.assertFalse(gate["pass"])
        self.assertIsNone(gate["keab"])

    def test_a_limit_dev_check_is_a_smoke_never_a_gate(self) -> None:
        self.assertEqual(self.dev_check("x1", "--limit", "2"), 0)
        self.assertTrue(all(c["n"] == 2 for c in self.w.calls))
        gate = fx.read_json(tp.dev_check_dir("x1", 2) / tp.GATE)
        self.assertFalse(gate["pass"])
        self.assertFalse(gate["test_launch_gate"])
        self.assertFalse((tp.dev_check_dir("x1") / tp.GATE).exists())

    def test_the_production_dev_check_is_not_rerun(self) -> None:
        self.assertEqual(self.dev_check("x1"), 0)
        self.w.loads.clear()
        self.refused(["--dev-check", "x1"], "exists")
        self.assertEqual(self.w.loads, [])

    def test_dev_check_refuses_without_a_passing_precheck(self) -> None:
        self.w.jobs = ["relabel-keab-score-06"]
        self.refused(["--dev-check", "fp"], "precheck failed")
        self.assertEqual(self.w.loads, [])

    def test_x1_refuses_a_process_started_with_cublas_workspace_config(self) -> None:
        with mock.patch.object(tk, "_CUBLAS_AT_START", ":4096:8"), \
                mock.patch.dict(kt.os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}):
            self.refused(["--dev-check", "x1"], "CUBLAS_WORKSPACE_CONFIG")
        self.assertEqual(self.w.loads, [])

    def test_fp_refuses_the_gpu1bit_wrapper(self) -> None:
        with mock.patch.dict(kt.os.environ, {"GPU1BIT_WRAPPER": "gpu1bit/shared/python"}):
            self.refused(["--dev-check", "fp"], "python-gpu1bit")
        self.assertEqual(self.w.loads, [])

    def test_fp_refuses_cudnn_flags_other_than_mb1_tests(self) -> None:
        import torch
        saved = torch.backends.cudnn.benchmark      # a module property: mock.patch cannot restore it
        torch.backends.cudnn.benchmark = True
        try:
            self.refused(["--dev-check", "fp"], "cuDNN")
        finally:
            torch.backends.cudnn.benchmark = saved
        self.assertEqual(self.w.loads, [])

    def test_fp_refuses_versions_other_than_mb1_tests(self) -> None:
        self.w.edit_json(self.w.arms["MB1"] / "summary.json",
                         lambda s: s["settings"]["versions"].__setitem__("nemo", "2.9.0"), repin=False)
        self.refused(["--test", "fp"], "versions")
        self.assertEqual(self.w.loads, [])


# ======================================================================= K-fp structural validity (real evaluate_set)

class RealFpPathTest(WorldCase):
    """K-fp through the REAL evaluate.evaluate_set (TinyModel, real audio files): the encoder output of every row is
    checked for finiteness over its valid frames during the forward; nothing in evaluate_set itself does."""

    def setUp(self) -> None:
        super().setUp()
        self.enterContext(self.w.real_fp())

    def doc(self, *parts: str) -> dict:
        return fx.read_json(tp.dev_check_dir("fp").joinpath(*parts))

    def test_clean_dev_check_passes_and_records_every_rows_finiteness(self) -> None:
        """TinyModel fills every row's padding frames with NaN: only valid frames are checked."""
        self.assertEqual(self.dev_check("fp"), 0)
        gate = self.gate("fp")
        self.assertTrue(gate["pass"], gate.get("structural"))
        for part in ("MB1", "keab"):
            for name in DEV:
                doc = self.doc(part, f"{name}.json")
                self.assertTrue(all(r["finite"] is True for r in doc["records"]), (part, name))
                check = doc["encoder_finiteness"]
                self.assertEqual(check["calls"], check["batches"])
                self.assertEqual(check["not_finite"], [])
                self.assertTrue(doc["structurally_valid"])
                self.assertEqual(doc["wer"]["wer"], 0.0)

    def test_non_finite_encoder_output_fails_the_fp_gate_with_the_evidence_kept(self) -> None:
        self.w.write_audio("ami_dev", 1, fx.POISON)
        self.w.poison_models = {"keab"}             # MB1's encoder output stays finite; Kea-B''s does not
        self.assertEqual(self.dev_check("fp"), 1)
        gate = self.gate("fp")
        self.assertTrue(gate["mb1"]["pass"])
        self.assertFalse(gate["structurally_valid"])
        self.assertFalse(gate["pass"])
        doc = self.doc("keab", "ami_dev.json")
        self.assertFalse(doc["structurally_valid"])
        self.assertEqual({r["id"]: r["finite"] for r in doc["records"]},
                         {"ami_dev:0": True, "ami_dev:1": False, "ami_dev:2": True})
        # one batch, longest first: ami_dev:2 (3 s), ami_dev:1 (2 s), ami_dev:0 (1 s)
        self.assertEqual(doc["encoder_finiteness"]["not_finite"], [{"batch": 0, "row": 1, "id": "ami_dev:1"}])
        self.assertEqual(doc["structural"]["not_finite"], ["ami_dev:1"])
        self.assertTrue(all(r["finite"] for r in self.doc("MB1", "ami_dev.json")["records"]))

    def test_an_encoder_call_count_other_than_the_batches_is_structurally_invalid(self) -> None:
        self.w.extra_call = True
        self.assertEqual(self.dev_check("fp"), 1)
        gate = self.gate("fp")
        self.assertFalse(gate["structurally_valid"])
        doc = self.doc("MB1", "librispeech_dev_clean.json")
        self.assertNotEqual(doc["encoder_finiteness"]["calls"], doc["encoder_finiteness"]["batches"])
        self.assertFalse(doc["structurally_valid"])

    def test_a_non_finite_test_utterance_is_kept_as_evidence_and_the_report_refuses(self) -> None:
        self.assertEqual(self.dev_check("fp"), 0)
        self.w.write_audio("voxpopuli", 1, fx.POISON)
        self.w.poison_models = {"keab"}
        code, out = run_main(["--test", "fp"])
        self.assertEqual(code, 0)
        self.assertIn("structurally invalid", out)
        self.assertEqual(self.w.ledger("fp")[-1]["state"], "complete")
        doc = fx.read_json(tp.test_dir("fp") / "voxpopuli.json")
        self.assertFalse(doc["structurally_valid"])
        self.assertEqual([r["id"] for r in doc["records"] if r["finite"] is False], ["voxpopuli:1"])
        self.assertFalse(fx.read_json(tp.test_dir("fp") / "summary.json")["structurally_valid"])
        self.complete("x1")
        self.report_refused("structurally invalid")

    def test_a_decoder_exception_crashes_and_leaves_the_attempt_open(self) -> None:
        self.assertEqual(self.dev_check("fp"), 0)
        self.w.write_audio("earnings22", 0, fx.CRASH)
        with self.assertRaises(RuntimeError):
            run_main(["--test", "fp"])
        ledger = self.w.ledger("fp")
        self.assertEqual([e["state"] for e in ledger], ["started", "set_done", "set_done", "set_done"])
        self.assertEqual([e["set"] for e in ledger[1:]], TEST[:3])
        self.assertFalse((tp.test_dir("fp") / "earnings22.json").exists())
        self.refused(["--summarize", "fp"], "all eight")
        self.w.write_audio("earnings22", 0)          # investigated and fixed: the open attempt resumes
        self.assertEqual(run_main(["--test", "fp"])[0], 0)
        self.assertEqual([e["state"] for e in self.w.ledger("fp")][-2:], ["set_done", "complete"])


# ============================================================================ --test refusals and runs

class TestLaunchTest(WorldCase):
    def assert_no_test_decoding(self) -> None:
        self.assertFalse(any(c["name"] in TEST for c in self.w.calls))
        self.assertFalse(tp.test_dir("x1").exists() and any(tp.test_dir("x1").glob("*.json")))

    def test_no_gate(self) -> None:
        self.refused(["--test", "x1"], "no dev-check gate")
        self.assertEqual(self.w.loads, [])

    def test_a_failing_gate(self) -> None:
        self.w.x1_dev_wer["ami_dev"] += 0.002
        self.assertEqual(self.dev_check("x1"), 1)
        self.w.loads.clear()
        self.refused(["--test", "x1"], "did not pass")
        self.assertEqual(self.w.loads, [])

    def test_a_limit_gate(self) -> None:
        self.assertEqual(self.dev_check("x1"), 0)
        self.edit_gate("x1", lambda g: g.__setitem__("limit", 2))
        self.w.loads.clear()
        self.refused(["--test", "x1"], "--limit")
        self.assertEqual(self.w.loads, [])

    def test_a_gate_with_other_identity_hashes_manifests_or_settings(self) -> None:
        self.assertEqual(self.dev_check("x1"), 0)
        cases = {"identity": lambda g: g["identity"].__setitem__("view_sha256", "f" * 64),
                 "scorer": lambda g: g["scorer"]["hashes"].__setitem__("relabel/keab_test.py", "0" * 64),
                 "manifests": lambda g: g["manifests"].__setitem__("ami", "0" * 64),
                 "settings": lambda g: g["settings"]["versions"].__setitem__("torch", "0.0")}
        path = tp.dev_check_dir("x1") / tp.GATE
        original = path.read_text()
        for key, fn in cases.items():
            with self.subTest(key=key):
                path.write_text(original)
                self.edit_gate("x1", fn)
                self.w.loads.clear()
                self.refused(["--test", "x1"], f"records other {key}")
                self.assertEqual(self.w.loads, [])

    def test_a_scorer_file_changed_after_the_gate(self) -> None:
        self.assertEqual(self.dev_check("fp"), 0)
        tp.SCORER_HASHES["relabel/keab_test.py"] = "0" * 64      # the precheck sees a changed scorer file
        self.w.loads.clear()
        self.refused(["--test", "fp"], "precheck failed")
        self.assertEqual(self.w.loads, [])

    def test_test_with_limit_is_rejected_before_anything_loads(self) -> None:
        self.assertEqual(self.dev_check("x1"), 0)
        self.w.loads.clear()
        with mock.patch.object(kt, "precheck", side_effect=AssertionError("no precheck either")):
            with self.assertRaises(SystemExit):
                run_main(["--test", "x1", "--limit", "5"])
            with self.assertRaises(SystemExit):
                run_main(["--summarize", "x1", "--limit", "5"])
        self.assertEqual(self.w.loads, [])
        self.assert_no_test_decoding()

    def test_x1_test_run(self) -> None:
        self.assertEqual(self.dev_check("x1"), 0)
        self.w.calls.clear()
        self.w.loads.clear()
        self.assertEqual(run_main(["--test", "x1"])[0], 0)
        self.assertEqual([c["name"] for c in self.w.calls], TEST)
        self.assertTrue(all(c["lock_held"] for c in self.w.calls))
        self.assertEqual(len(self.w.loads), 1)
        ledger = self.w.ledger("x1")
        self.assertEqual([e["state"] for e in ledger], ["started"] + ["set_done"] * 8 + ["complete"])
        self.assertEqual(ledger[0]["gate_sha256"], fx.sha(tp.dev_check_dir("x1") / tp.GATE))
        out = tp.test_dir("x1")
        self.assertEqual({e["set"]: e["sha256"] for e in ledger if e["state"] == "set_done"},
                         {n: fx.sha(out / f"{n}.json") for n in TEST})
        summary = fx.read_json(out / "summary.json")
        self.assertEqual(set(summary["sets"]), set(TEST))
        self.assertAlmostEqual(summary["mean_wer"], 100 * sum(fx.KX1_WER[n] for n in paths.MEAN_SETS) / 7, places=9)
        self.assertEqual(summary["per_set_sha256"], {n: fx.sha(out / f"{n}.json") for n in TEST})
        doc = fx.read_json(out / "common_voice.json")
        self.assertEqual([r["i"] for r in doc["records"]], [0, 1, 2])
        self.assertEqual(doc["phase"], "test")

    def test_fp_test_run(self) -> None:
        self.assertEqual(self.dev_check("fp"), 0)
        self.w.calls.clear()
        self.w.loads.clear()
        self.assertEqual(run_main(["--test", "fp"])[0], 0)
        self.assertEqual([(c["label"], c["name"]) for c in self.w.calls], [("keab", n) for n in TEST])
        self.assertTrue(all(c["graphs"] == "no_while_loops" and c["batch_size"] == 64 for c in self.w.calls))
        self.assertEqual([e["state"] for e in self.w.ledger("fp")], ["started"] + ["set_done"] * 8 + ["complete"])

    def test_the_modes_are_independent(self) -> None:
        self.complete("x1")
        self.refused(["--test", "fp"], "no dev-check gate")


# ============================================================================ attempts and resume

class ResumeTest(WorldCase):
    def setUp(self) -> None:
        super().setUp()
        self.assertEqual(self.dev_check("x1"), 0)
        self.w.calls.clear()
        self.w.loads.clear()

    def crash_at(self, name: str) -> None:
        self.w.fail_on["x1"] = name
        with self.assertRaises(RuntimeError):
            run_main(["--test", "x1"])
        del self.w.fail_on["x1"]
        self.w.calls.clear()

    def test_a_set_done_set_is_not_decoded_again_and_the_rest_are(self) -> None:
        self.crash_at("earnings22")
        out = tp.test_dir("x1")
        first = {n: fx.sha(out / f"{n}.json") for n in TEST[:3]}
        self.assertEqual([e["state"] for e in self.w.ledger("x1")], ["started"] + ["set_done"] * 3)
        self.assertEqual(run_main(["--test", "x1"])[0], 0)
        self.assertEqual([c["name"] for c in self.w.calls], TEST[3:])
        self.assertEqual({n: fx.sha(out / f"{n}.json") for n in TEST[:3]}, first)
        ledger = self.w.ledger("x1")
        self.assertEqual([e["state"] for e in ledger],
                         ["started"] + ["set_done"] * 3 + ["resumed"] + ["set_done"] * 5 + ["complete"])
        self.assertEqual(ledger[4]["gate_sha256"], ledger[0]["gate_sha256"])

    def test_a_set_written_but_not_recorded_is_decoded_again_from_the_start(self) -> None:
        real = kt.append_attempt

        def append(mode, entry):
            if entry.get("set") == "ami":
                raise OSError("power lost before the set_done record")
            real(mode, entry)

        with mock.patch.object(kt, "append_attempt", side_effect=append):
            with self.assertRaises(OSError):
                run_main(["--test", "x1"])
        self.assertTrue((tp.test_dir("x1") / "ami.json").exists())
        self.w.calls.clear()
        self.assertEqual(run_main(["--test", "x1"])[0], 0)
        self.assertEqual([c["name"] for c in self.w.calls], TEST[2:])
        resumed = [e for e in self.w.ledger("x1") if e["state"] == "resumed"]
        self.assertEqual(resumed[0]["replayed"], ["ami"])

    def test_a_done_file_that_no_longer_matches_refuses(self) -> None:
        self.crash_at("gigaspeech")
        path = tp.test_dir("x1") / "ami.json"
        path.write_text(path.read_text().replace("Mister", "Mr"))
        self.w.loads.clear()
        self.refused(["--test", "x1"], "set_done hash")
        self.assertEqual(self.w.loads, [])
        self.assertEqual(self.w.calls, [])

    def test_a_resume_with_another_gate_refuses(self) -> None:
        self.crash_at("gigaspeech")
        self.edit_gate("x1", lambda g: g.__setitem__("utc", "2099-01-01T00:00:00Z"))
        self.w.loads.clear()
        self.refused(["--test", "x1"], "not the gate this attempt started with")
        self.assertEqual(self.w.loads, [])

    def test_complete_refuses(self) -> None:
        self.assertEqual(run_main(["--test", "x1"])[0], 0)
        self.w.loads.clear()
        self.w.calls.clear()
        self.refused(["--test", "x1"], "complete")
        self.assertEqual((self.w.loads, self.w.calls), ([], []))

    def test_output_without_an_attempt_record_refuses(self) -> None:
        tp.test_dir("x1").mkdir(parents=True)
        self.refused(["--test", "x1"], "without an attempt record")
        self.assertEqual(self.w.loads, [])

    def test_summarize_makes_no_decoding_calls(self) -> None:
        real = kt.append_attempt

        def append(mode, entry):
            if entry.get("state") == "complete":
                raise OSError("crash before the complete record")
            real(mode, entry)

        with mock.patch.object(kt, "append_attempt", side_effect=append):
            with self.assertRaises(OSError):
                run_main(["--test", "x1"])
        (tp.test_dir("x1") / "summary.json").unlink()
        self.refused(["--test", "x1"], "only --summarize")
        self.w.loads.clear()
        self.w.calls.clear()
        lock_events = list(self.w.lock_events)
        with mock.patch.object(kt, "precheck", side_effect=AssertionError("no precheck needed")):
            self.assertEqual(run_main(["--summarize", "x1"])[0], 0)
        self.assertEqual((self.w.loads, self.w.calls), ([], []))
        self.assertEqual(self.w.lock_events, lock_events)            # no GPU lock taken
        self.assertEqual(self.w.ledger("x1")[-1]["state"], "complete")
        self.assertTrue((tp.test_dir("x1") / "summary.json").exists())
        self.assertEqual(run_main(["--summarize", "x1"])[0], 0)      # already complete: nothing to do

    def test_summarize_refuses_without_eight_done_sets_or_with_a_changed_file(self) -> None:
        self.crash_at("gigaspeech")
        self.refused(["--summarize", "x1"], "all eight")

    def test_summarize_refuses_a_hash_mismatch(self) -> None:
        real = kt.append_attempt
        with mock.patch.object(kt, "append_attempt",
                               side_effect=lambda m, e: (_ for _ in ()).throw(OSError()) if e.get("state") == "complete"
                               else real(m, e)):
            with self.assertRaises(OSError):
                run_main(["--test", "x1"])
        path = tp.test_dir("x1") / "voxpopuli.json"
        path.write_text(path.read_text().replace("Mister", "Mr"))
        self.refused(["--summarize", "x1"], "set_done hash")

    def test_an_invalid_row_on_test_is_kept_as_evidence_not_rescored(self) -> None:
        self.w.invalid.add(("x1", "voxpopuli"))
        code, out = run_main(["--test", "x1"])
        self.assertEqual(code, 0)
        self.assertIn("structurally invalid", out)
        doc = fx.read_json(tp.test_dir("x1") / "voxpopuli.json")
        self.assertFalse(doc["structurally_valid"])
        self.assertEqual([c["name"] for c in self.w.calls].count("voxpopuli"), 1)
        self.assertFalse(fx.read_json(tp.test_dir("x1") / "summary.json")["structurally_valid"])


# ============================================================================ the report

class ReportTest(WorldCase):
    def setUp(self) -> None:
        super().setUp()
        self.complete("x1", "fp")

    def test_known_values(self) -> None:
        code, _ = run_report()
        self.assertEqual(code, 0)
        result = fx.read_json(tp.TEST_JSON)
        cols = list(paths.MEAN_SETS) + ["mean", "common_voice"]
        self.assertEqual(result["columns"], cols)
        rows = result["rows"]
        self.assertEqual(list(rows), ["NVIDIA published", "B0", "M1", "MB1", FP, X1])
        for label, wer in (("B0", fx.B0_WER), ("M1", fx.M1_WER), ("MB1", fx.MB1_WER), (FP, fx.KFP_WER),
                           (X1, fx.KX1_WER)):
            for n in TEST:
                self.assertAlmostEqual(rows[label][n], 100 * wer[n], places=9, msg=(label, n))
            self.assertAlmostEqual(rows[label]["mean"], 100 * sum(wer[n] for n in paths.MEAN_SETS) / 7, places=9)
        self.assertAlmostEqual(rows["NVIDIA published"]["mean"],
                               sum(paths.PUBLISHED_WER[n] for n in paths.MEAN_SETS) / 7, places=9)
        self.assertNotIn("common_voice", rows["NVIDIA published"])
        d = result["differences"]
        self.assertAlmostEqual(d[f"{X1} - MB1"]["ami"]["points"], 100 * (0.100 - 0.110), places=9)
        self.assertAlmostEqual(d[f"{X1} - MB1"]["ami"]["relative_percent"], 100 * (0.100 - 0.110) / 0.110, places=9)
        self.assertAlmostEqual(d[f"{FP} - MB1"]["common_voice"]["points"], 100 * (0.071 - 0.090), places=9)
        mean_x1 = 100 * sum(fx.KX1_WER[n] for n in paths.MEAN_SETS) / 7
        mean_mb1 = 100 * sum(fx.MB1_WER[n] for n in paths.MEAN_SETS) / 7
        mean_m1 = 100 * sum(fx.M1_WER[n] for n in paths.MEAN_SETS) / 7
        self.assertAlmostEqual(d[f"{X1} - MB1"]["mean"]["points"], mean_x1 - mean_mb1, places=9)
        self.assertAlmostEqual(d[f"{X1} - M1"]["mean"]["points"], mean_x1 - mean_m1, places=9)
        self.assertAlmostEqual(d[f"{X1} - M1"]["mean"]["relative_percent"], 100 * (mean_x1 - mean_m1) / mean_m1,
                               places=9)
        self.assertIn(f"{FP} - M1", d)
        empty = result["empty_hypothesis_fraction"]
        self.assertAlmostEqual(empty["B0"]["ami"], 1 / 3, places=12)
        self.assertEqual(empty[X1]["ami"], 0.0)
        self.assertEqual(result["empty_hypotheses"]["B0"]["ami"], 1)
        self.assertEqual(result["sizes_mb"], {"Kea-B'": 91.24, "MB1": 105.3, "M1": 180.8, "B0": 2472.2})
        self.assertEqual(result["sanity_flags"], [])
        self.assertTrue(result["dev_check"]["x1"]["pass"] and result["dev_check"]["fp"]["pass"])
        text = tp.RESULTS_MD.read_text()
        self.assertIn("net change of stage 4's whole package", text)
        self.assertIn("teacher-package effect", text)
        self.assertIn("Common Voice", text)
        self.assertIn("No intervals", text)
        self.assertNotIn("SANITY", text)

    def test_empty_rates_use_mb1s_definition_not_the_scorers_count(self) -> None:
        self.w.tamper("x1", "ami", lambda d: d.__setitem__("empty_hypotheses", 99))
        run_report()
        self.assertEqual(fx.read_json(tp.TEST_JSON)["empty_hypotheses"][X1]["ami"], 0)

    # refusals
    def test_scorer_files_other_than_the_pinned_ones(self) -> None:
        tp.SCORER_HASHES["relabel/keab_test.py"] = "0" * 64
        self.report_refused("SCORER_HASHES")

    def test_recorded_scorer_hashes_other_than_the_pinned_ones(self) -> None:
        self.w.tamper("fp", "ami", lambda d: d["scorer"]["hashes"].__setitem__("relabel/keab_test.py", "0" * 64))
        self.report_refused("scorer")

    def test_identity_other_than_the_pinned_one(self) -> None:
        self.w.tamper("x1", "gigaspeech", lambda d: d["identity"].__setitem__("view_sha256", "f" * 64))
        self.report_refused("identity")

    def test_a_mode_not_complete(self) -> None:
        entries = self.w.ledger("fp")[:-1]
        self.w.write_ledger("fp", entries)
        self.report_refused("complete")

    def test_a_per_set_file_not_matching_its_set_done_hash(self) -> None:
        self.w.tamper("x1", "earnings22", lambda d: d["wer"].__setitem__("wer", 0.01), rehash=False)
        self.report_refused("set_done")

    def test_a_manifest_differing_from_the_current_file(self) -> None:
        manifest = self.w.manifests / "test_voxpopuli.jsonl"
        manifest.write_text(manifest.read_text().replace("Mister", "Mr"))
        self.report_refused("manifest")

    def test_a_manifest_differing_between_arms(self) -> None:
        self.w.tamper("fp", "spgispeech", lambda d: d.__setitem__("manifest_sha256", "0" * 64))
        self.report_refused("manifest")

    def test_a_set_missing(self) -> None:
        (tp.test_dir("x1") / "librispeech_other.json").unlink()
        self.report_refused("librispeech_other")

    def test_a_duplicated_set(self) -> None:
        doc = fx.read_json(tp.test_dir("x1") / "ami.json")
        self.w.tamper("x1", "earnings22", lambda d: d.update(doc))
        self.report_refused("earnings22")

    def test_a_partial_set(self) -> None:
        self.w.tamper("fp", "gigaspeech", lambda d: d.__setitem__("limit", 2))
        self.report_refused("partial")

    def test_an_utterance_id_missing_or_duplicated(self) -> None:
        self.w.tamper("x1", "voxpopuli", lambda d: d["records"].pop())
        self.report_refused("utterance IDs")
        self.w.tamper("x1", "voxpopuli", lambda d: d["records"].append(dict(d["records"][0])))
        self.report_refused("utterance IDs")

    def test_fp_decoding_metadata_differing_from_mb1_tests(self) -> None:
        self.w.tamper("fp", "ami", lambda d: d["decoding"].__setitem__("batch_size", 32))
        self.report_refused("decoding")

    def test_fp_versions_differing_from_mb1_tests(self) -> None:
        for name in TEST:
            self.w.tamper("fp", name, lambda d: d["settings"]["versions"].__setitem__("nemo", "2.9.0"))
        self.report_refused("versions")

    def test_normalizer_metadata_differing_from_mb1_tests(self) -> None:
        self.w.tamper("x1", "ami", lambda d: d.__setitem__("normalizer", "another normalizer"))
        self.report_refused("normalizer")

    def test_a_gate_other_than_the_one_the_attempt_started_with(self) -> None:
        self.edit_gate("fp", lambda g: g.__setitem__("utc", "2099-01-01T00:00:00Z"))
        self.report_refused("gate")

    def test_an_output_without_the_finiteness_record_is_refused(self) -> None:
        self.w.tamper("fp", "earnings22", lambda d: [r.pop("finite") for r in d["records"]])
        self.report_refused("finite")

    def test_an_fp_output_without_its_encoder_check_is_refused(self) -> None:
        self.w.tamper("fp", "gigaspeech", lambda d: d.pop("encoder_finiteness"))
        self.report_refused("encoder")

    def test_an_fp_output_whose_encoder_check_missed_batches_is_refused(self) -> None:
        self.w.tamper("fp", "gigaspeech", lambda d: d["encoder_finiteness"].__setitem__("calls", 0))
        self.report_refused("encoder")

    def test_a_structurally_invalid_row_refuses_a_numerical_result(self) -> None:
        self.w.tamper("fp", "common_voice", lambda d: d["records"][1].__setitem__("finite", False))
        self.report_refused("structurally invalid")

    # sanity checks
    def test_a_set_above_three_times_b0_pauses_the_write_up(self) -> None:
        self.w.set_test_wer("x1", "gigaspeech", 3 * fx.B0_WER["gigaspeech"] + 0.001)
        code, out = run_report()
        self.assertEqual(code, 3)
        self.assertIn("SANITY", tp.RESULTS_MD.read_text())
        self.assertTrue(any("gigaspeech" in f for f in fx.read_json(tp.TEST_JSON)["sanity_flags"]))

    def test_exactly_three_times_b0_is_not_flagged(self) -> None:
        self.w.set_test_wer("x1", "gigaspeech", 3 * fx.B0_WER["gigaspeech"])
        self.assertEqual(run_report()[0], 0)

    def test_empty_hypotheses_above_one_percent_and_more_than_b0_pause_the_write_up(self) -> None:
        def empty_one(d):
            d["records"][0]["hyp"] = ""
            d["records"][0]["hyp_norm"] = ""
        self.w.tamper("fp", "voxpopuli", empty_one)      # B0 has none on voxpopuli
        code, _ = run_report()
        self.assertEqual(code, 3)
        self.assertTrue(any("voxpopuli" in f for f in fx.read_json(tp.TEST_JSON)["sanity_flags"]))

    def test_empty_hypotheses_not_more_than_b0_are_not_flagged(self) -> None:
        def empty_one(d):
            d["records"][0]["hyp"] = ""
            d["records"][0]["hyp_norm"] = ""
        self.w.tamper("x1", "ami", empty_one)            # B0 also has one on ami
        self.assertEqual(run_report()[0], 0)


if __name__ == "__main__":
    unittest.main()
