"""Contract tests 1-8 of the binary main run MB1 (plans/main.md "Contract tests", "Implementation
notes (2026-10-17)"): the sweep's main phase (command, selection check, gate, validation), the
MB1 entry point binary_main.py (stream pins, seed refusal, the ternary stream record), resume
refusal, and that the finished pilot keeps validating (its binary sources are unchanged).

Some tests read the real pilot artifacts (read only), as test_provenance.py reads M1's summary.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest import mock

import binary_fixtures as fx

import binary_main
import binary_main_paths
import binary_paths
import binary_sweep
import binary_train
import data
import paths
import stream
import sweep
import train

SETS = list(paths.DEV_SETS)
MB1_HASH = "bad909114fbe3e50505434635c49406f751f0673663e275b460ebdb82612badf"
MB1_ARGS = ["--init", "m1", "--quantizer", "binary", "--arm", "P2", "--lr", "2e-4",
            "--run-name", "main-binB-m1-lr2e-4", "--train", "main", "--max-steps", "250000",
            "--seed", "20261017", "--batch-seconds", "600", "--eval-every", "12500", "--select", "final",
            "--stream", "hub", "--grad-checkpointing", "--no-powerlog"]
B2_ARGS = ["--init", "m1", "--quantizer", "binary", "--arm", "P2", "--lr", "2e-4",
           "--run-name", "pilot-binB-m1-lr2e-4", "--train", "pilot", "--max-steps", "12000",
           "--batch-seconds", "600", "--eval-every", "1200", "--select", "final", "--stream", "hub",
           "--grad-checkpointing", "--no-powerlog"]


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextlib.contextmanager
def fake_init():
    """binary_paths' pinned M1 checkpoint pointed at a small test file (only its hash is read)."""
    path = fx.TEST_ROOT / "main-fake-m1.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not M1")
    with fx.pinned_init(path, sha(path)):
        yield


class FakeStream:
    def __init__(self, config_hash: str) -> None:
        self.config_hash, self.closed = config_hash, False

    def close(self) -> None:
        self.closed = True


# ----------------------------------------------------------------------------- contract 1

class MainCommandTest(unittest.TestCase):
    def test_mb1_arm_and_command(self) -> None:
        arm = binary_sweep.MAIN_ARM
        self.assertEqual((arm.name, arm.init, arm.quantizer, arm.ramp_kind, arm.lr, arm.run_name, arm.train,
                          arm.seed), ("MB1", "m1", "binary", "ternary", "2e-4", "main-binB-m1-lr2e-4", "main",
                                      20261017))
        cmd = binary_sweep.train_command(binary_sweep.main_plan(), arm)
        self.assertEqual(cmd[:2], [str(binary_paths.HERE / "python"), str(binary_paths.HERE / "binary_main.py")])
        self.assertEqual(cmd[2:], ["--init", "m1", "--quantizer", "binary", "--arm", "P2", "--lr", "2e-4",
                                   "--run-name", "main-binB-m1-lr2e-4", "--train", "main", "--max-steps", "250000",
                                   "--seed", "20261017", "--batch-seconds", "600", "--eval-every", "12500",
                                   "--select", "final", "--stream", "hub", "--grad-checkpointing"])

    def test_pilot_commands_are_unchanged(self) -> None:
        for arm in binary_sweep.ARMS.values():
            cmd = binary_sweep.train_command(binary_sweep.Plan(), arm)
            self.assertEqual(cmd[1], str(binary_paths.HERE / "binary_train.py"))
            self.assertNotIn("--seed", cmd)
            self.assertEqual(cmd[cmd.index("--train") + 1], "pilot")

    def test_dry_run_prints_only_the_mb1_command_and_writes_nothing(self) -> None:
        root = fx.TEST_ROOT / "main-dry-run"
        shutil.rmtree(root, ignore_errors=True)
        runs = root / "runs"
        out = io.StringIO()
        with mock.patch.object(binary_sweep, "require_selection", return_value={"decision": {}}), \
                mock.patch.object(binary_sweep, "require_gate", return_value={"gate": {"pass": True}}), \
                mock.patch.object(binary_paths, "RUNS", runs), contextlib.redirect_stdout(out):
            binary_sweep.main(["--phase", "main", "--dry-run", "--allow-unconfined"])
        lines = [line for line in out.getvalue().splitlines() if "/python " in line]
        self.assertEqual(len(lines), 1, out.getvalue())
        expected = binary_sweep.train_command(binary_sweep.main_plan(), binary_sweep.MAIN_ARM)
        self.assertTrue(lines[0].endswith(" ".join(expected)), lines[0])
        self.assertFalse(runs.exists())
        shutil.rmtree(root, ignore_errors=True)


# ----------------------------------------------------------------------------- contract 2

class SelectionCheckTest(unittest.TestCase):
    """--phase main requires the pilot's selection: proposal "binary main run", start m1, winner B2,
    made from pilot arms that still validate (the real pilot artifacts, read only)."""

    @classmethod
    def setUpClass(cls) -> None:
        paths.require_mount()
        cls.real = binary_paths.RUNS / binary_sweep.SELECTION_NAME
        cls.selection = json.loads(cls.real.read_text())
        cls.root = fx.TEST_ROOT / "main-selection"
        shutil.rmtree(cls.root, ignore_errors=True)
        cls.root.mkdir(parents=True)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root, ignore_errors=True)

    def write(self, name: str, selection: dict) -> Path:
        path = self.root / name
        path.write_text(json.dumps(selection, indent=1) + "\n")
        return path

    def refused(self, path: Path, word: str) -> None:
        with self.assertRaises(SystemExit) as caught:
            binary_sweep.require_selection(path)
        self.assertIn(word, str(caught.exception))

    def test_the_finished_pilot_selection_is_accepted(self) -> None:
        selection = binary_sweep.require_selection()
        self.assertEqual(selection["decision"]["recommendation"]["proposal"], "binary main run")
        self.assertEqual(selection["decision"]["starting_point"]["winner"], "B2")

    def test_missing_selection_is_refused(self) -> None:
        self.refused(self.root / "absent.json", "missing")

    def test_mixed_pilot_proposal_is_refused(self) -> None:
        s = copy.deepcopy(self.selection)
        s["decision"]["recommendation"]["proposal"] = "mixed pilot"
        self.refused(self.write("mixed.json", s), "proposal")

    def test_other_winner_or_start_is_refused(self) -> None:
        s = copy.deepcopy(self.selection)
        s["decision"]["starting_point"]["winner"] = "B5"
        s["decision"]["recommendation"]["start"] = "B5"
        self.refused(self.write("winner.json", s), "winner")
        s = copy.deepcopy(self.selection)
        s["decision"]["starting_point"]["start"] = "fp32"
        self.refused(self.write("start.json", s), "start")

    def test_stale_selection_is_refused(self) -> None:
        # An arm that no longer validates under the current sources.
        real_validate = binary_sweep.validate_arm

        def validate(run, expected, sources):
            if expected.binary.name == "B5":
                raise RuntimeError(f"run {run} cannot be reused: parakeet-binary/binary_train.py changed")
            return real_validate(run, expected, sources)

        with mock.patch.object(binary_sweep, "validate_arm", side_effect=validate):
            self.refused(self.real, "stale")
        # A decision that the validated arms no longer produce.
        s = copy.deepcopy(self.selection)
        s["decision"]["starting_point"]["A_minus_B_points"] += 0.01
        self.refused(self.write("decision.json", s), "stale")


# ----------------------------------------------------------------------------- contract 3

class GateTest(unittest.TestCase):
    def test_no_training_unless_the_gate_passed(self) -> None:
        failed = {"point": "G-B", "gate": {"pass": False, "difference_points": 0.5,
                                           "ternary_twin_encoder_equal": True}}
        with mock.patch.object(binary_sweep, "require_selection", return_value={"decision": {}}), \
                mock.patch.object(binary_sweep, "ensure_point", return_value=failed), \
                mock.patch.object(sweep, "run_train") as run_train, \
                mock.patch.object(binary_sweep, "ensure_arm") as ensure_arm:
            with self.assertRaises(SystemExit):
                binary_sweep.phase(binary_sweep.main_plan(), "main", dry=False)
            run_train.assert_not_called()
            ensure_arm.assert_not_called()

    def test_selection_is_checked_before_the_gate(self) -> None:
        with mock.patch.object(binary_sweep, "require_selection", side_effect=SystemExit("no selection")), \
                mock.patch.object(binary_sweep, "require_gate") as gate, \
                mock.patch.object(sweep, "run_train") as run_train:
            with self.assertRaises(SystemExit):
                binary_sweep.phase(binary_sweep.main_plan(), "main", dry=False)
            gate.assert_not_called()
            run_train.assert_not_called()


# ----------------------------------------------------------------------------- contract 4

class StreamPinTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = fx.TEST_ROOT / "main-stream-pin"
        shutil.rmtree(self.root, ignore_errors=True)
        patch = mock.patch.object(binary_main_paths, "STREAM_RECORD_ROOT", self.root / "records")
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(shutil.rmtree, self.root, True)

    @staticmethod
    def hub(train_phase: str, seed: int) -> argparse.Namespace:
        return argparse.Namespace(stream="hub", train=train_phase, seed=seed)

    def test_pins(self) -> None:
        self.assertEqual(binary_main_paths.stream_pin("main", 20261017), (8, MB1_HASH))
        self.assertEqual(binary_main_paths.stream_pin("pilot", paths.SEED),
                         (binary_paths.STREAM_WORKERS["pilot"], binary_paths.STREAM_CONFIG_HASH["pilot"]))
        for phase, seed in (("main", paths.SEED), ("main", 1), ("pilot", 20261017), ("main", 20261018)):
            with self.subTest(phase=phase, seed=seed), self.assertRaises(ValueError):
                binary_main_paths.stream_pin(phase, seed)

    def test_mb1_opens_8_workers_and_requires_the_pinned_hash(self) -> None:
        calls = []

        def make(phase, seed, num_workers):
            calls.append((phase, seed, num_workers))
            return FakeStream(MB1_HASH)

        with mock.patch.object(stream, "make_training_stream", side_effect=make):
            opened = binary_main.open_stream(self.hub("main", 20261017))
        self.assertEqual(calls, [("main", 20261017, 8)])
        self.assertEqual(opened.config_hash, MB1_HASH)

        wrong = FakeStream("0" * 64)
        with mock.patch.object(stream, "make_training_stream", return_value=wrong), \
                self.assertRaises(RuntimeError):
            binary_main.open_stream(self.hub("main", 20261017))
        self.assertTrue(wrong.closed)

    def test_unpinned_seeds_are_refused_before_opening(self) -> None:
        with mock.patch.object(stream, "make_training_stream", side_effect=AssertionError("opened")):
            for phase, seed in (("main", paths.SEED), ("main", 12345), ("pilot", 20261017)):
                with self.subTest(phase=phase, seed=seed), self.assertRaises(RuntimeError):
                    binary_main.open_stream(self.hub(phase, seed))
            with self.assertRaises(RuntimeError):  # a local stream at an unpinned seed as well
                binary_main.open_stream(argparse.Namespace(stream="local-librispeech", train="main", seed=7))

    def test_argument_parsing_refuses_an_unpinned_seed(self) -> None:
        with fake_init():
            args = binary_main.parse_args(MB1_ARGS)
            self.assertEqual((args.train, args.seed, args.max_steps), ("main", 20261017, 250000))
            for seed in ("20260930", "12345"):
                argv = list(MB1_ARGS)
                argv[argv.index("--seed") + 1] = seed
                with self.subTest(seed=seed), self.assertRaises(SystemExit) as caught, \
                        contextlib.redirect_stderr(io.StringIO()):
                    binary_main.parse_args(argv)
                self.assertEqual(caught.exception.code, 2)
            self.assertEqual(binary_main.parse_args(B2_ARGS).seed, paths.SEED)

    def test_pilot_stream_behaviour_is_unchanged(self) -> None:
        pilot_hash = binary_paths.STREAM_CONFIG_HASH["pilot"]
        for opener in (binary_train.open_stream, binary_main.open_stream):
            calls = []

            def make(phase, seed, num_workers):
                calls.append((phase, seed, num_workers))
                return FakeStream(pilot_hash)

            with self.subTest(opener.__module__), \
                    mock.patch.object(stream, "make_training_stream", side_effect=make):
                opener(self.hub("pilot", paths.SEED))
                self.assertEqual(calls, [("pilot", paths.SEED, 6)])
            with self.subTest(opener.__module__, hash="wrong"), self.assertRaises(RuntimeError), \
                    mock.patch.object(stream, "make_training_stream", return_value=FakeStream("1" * 64)):
                opener(self.hub("pilot", paths.SEED))


# ----------------------------------------------------------------------------- contract 5

class SeedIndependenceTest(unittest.TestCase):
    """The held-out data does not depend on the training seed, and opening the MB1 stream leaves
    the ternary stream record byte-identical (it is written under the binary artifacts instead)."""

    def setUp(self) -> None:
        paths.require_mount()
        self.root = fx.TEST_ROOT / "main-records"
        shutil.rmtree(self.root, ignore_errors=True)
        patch = mock.patch.object(binary_main_paths, "STREAM_RECORD_ROOT", self.root)
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(shutil.rmtree, self.root, True)

    @staticmethod
    def held_out_files() -> list[Path]:
        names = SETS + [f"{name}_{data.DEV_SUBSET}" for name in SETS]
        return [data.manifest_path(n) for n in names] + [data.reserved_path()]

    def test_held_out_data_and_the_ternary_record_are_unchanged(self) -> None:
        ternary = [paths.DATA / "stream" / "sources.json", paths.DATA / "stream" / "yodas_train_files.json"]
        self.assertEqual(list(binary_main_paths.TERNARY_STREAM_RECORDS), ternary)
        protected = ternary + self.held_out_files()
        before = {p: sha(p) for p in protected}
        reserved = sorted(data.reserved_recordings())
        original_bytes = {p: p.read_bytes() for p in ternary}

        def restore_ternary_record() -> None:  # M1's provenance record, should the redirect ever fail
            for path, content in original_bytes.items():
                if path.read_bytes() != content:
                    data.atomic_write_text(path, content.decode())
        self.addCleanup(restore_ternary_record)

        mb1 = binary_main.open_stream(argparse.Namespace(stream="hub", train="main", seed=20261017))
        pilot = binary_main.open_stream(argparse.Namespace(stream="hub", train="pilot", seed=paths.SEED))
        try:
            self.assertEqual(mb1.config_hash, MB1_HASH)
            self.assertEqual(mb1.num_workers, 8)
            self.assertEqual(mb1.cfg.seed, 20261017)
            self.assertEqual(list(mb1.cfg.exclude_groups), reserved)
            self.assertEqual(mb1.cfg.exclude_groups, pilot.cfg.exclude_groups)
            self.assertEqual(mb1.cfg.sources, pilot.cfg.sources)  # same files, shares and slots
        finally:
            mb1.close()
            pilot.close()
        self.assertEqual({p: sha(p) for p in protected}, before)
        self.assertEqual(sorted(data.reserved_recordings()), reserved)

        # The record is written under the binary artifacts, per seed; only the seed differs.
        record = json.loads((self.root / "seed-20261017" / "stream" / "sources.json").read_text())
        original = json.loads(ternary[0].read_text())
        self.assertEqual(record["seed"], 20261017)
        self.assertEqual({k: v for k, v in record.items() if k != "seed"},
                         {k: v for k, v in original.items() if k != "seed"})

    def test_a_changed_ternary_record_stops_the_run(self) -> None:
        fake = [self.root / "ternary" / "sources.json", self.root / "ternary" / "yodas_train_files.json"]
        for path in fake:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n")

        def make(phase, seed, num_workers):
            fake[0].write_text('{"seed": 20261017}\n')  # what an unredirected record would do
            return FakeStream(MB1_HASH)

        with mock.patch.object(binary_main_paths, "TERNARY_STREAM_RECORDS", tuple(fake)), \
                mock.patch.object(stream, "make_training_stream", side_effect=make), \
                self.assertRaises(RuntimeError):
            binary_main.open_stream(argparse.Namespace(stream="hub", train="main", seed=20261017))


# ----------------------------------------------------------------------------- contract 6

MODE = "no_while_loops"


class MainValidationTest(unittest.TestCase):
    """A finished MB1 run is reused only if it is exactly the planned run."""

    @classmethod
    def setUpClass(cls) -> None:
        paths.require_mount()
        cls.root = fx.TEST_ROOT / "main-validation"
        cls.init = mock.patch.object(binary_paths, "init_sha256", side_effect=lambda init: f"{init:0>64}")
        cls.init.start()
        cls.plan, cls.arm = binary_sweep.main_plan(), binary_sweep.MAIN_ARM
        cls.reference = binary_sweep.main_reference()
        cls.expected = binary_sweep.BinaryExpected(cls.plan, cls.arm, cls.reference)
        cls.sources = binary_sweep.current_sources(binary_sweep.MAIN_TRAINING_SOURCES)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.init.stop()
        shutil.rmtree(cls.root, ignore_errors=True)

    def records(self) -> tuple[dict, dict]:
        """(summary.json, config.json) of a finished MB1 run as train.py and the seam write them."""
        args = self.expected.args()
        stream_id = {"kind": "hub", "phase": "main", "seed": 20261017, "config_hash": MB1_HASH,
                     "stream_py_sha256": "s" * 64, "teacher_py_sha256": self.reference["stream"]["teacher_py_sha256"]}
        graphs = {"teacher": MODE, "student": MODE, "rebuilt": MODE, "dev-subset": MODE, "full-dev": MODE}
        summary = {"run_name": "main-binB-m1-lr2e-4", "arm": "P2", "recipe": "P2", "lr": 2e-4,
                   "max_steps": 250000, "steps": 250000, "select": "final", "selected_step": 250000,
                   "eval_every": 12500, "grad_checkpointing": True, "source_hashes": dict(self.sources),
                   "dev_subset_mean_wer": 0.07, "dev_mean_wer": 0.065, "dev_wer": {s: 0.065 for s in SETS},
                   "seed": 20261017, "batch_seconds": 600.0, "train": "main", "stream": stream_id,
                   "quantized": True, "encoder_matching": True, "train_pred_joint": False, "ramp_steps": 62500,
                   "warmup_steps": 5000, "audio_seconds_seen": 1.0e8, "smoke": False, "dev_limit": None,
                   "started_utc": "2026-10-17T12:00:00Z", "finished_utc": "2026-10-20T12:00:00Z",
                   "quantized_modules_sha256": self.reference["quantized_modules_sha256"],
                   "reconstruction": {"codes_exact": True, "scales_exact": True},
                   "binary": {"init": "m1", "quantizer": "binary", "ramp_kind": "ternary",
                              "code_balance": {"minus_one": 0.5, "zero": 0.0, "plus_one": 0.5},
                              "decoder_graphs": graphs, "timing": {"train_wall_s": 2.4e5, "sessions": 1}}}
        config = {"args": dict(args), "recipe": {"encoder_matching": True, "train_pred_joint": False,
                                                 "quantized": True},
                  "trainable": {"trainable_modules": ["encoder"]}, "training_source_hashes": dict(self.sources),
                  "stream": stream_id, "quantized_modules_sha256": self.reference["quantized_modules_sha256"],
                  "binary": {"decoder_graphs": {"teacher": MODE, "student": MODE}, "stream_workers": 8}}
        return summary, config

    def write(self, name: str, summary: dict, config: dict) -> Path:
        run = self.root / name
        shutil.rmtree(run, ignore_errors=True)
        (run / "export").mkdir(parents=True)
        (run / "summary.json").write_text(json.dumps(summary))
        (run / "config.json").write_text(json.dumps(config))
        (run / "eval-dev.json").write_text(json.dumps({"dev_wer": summary["dev_wer"], "limit": None,
                                                       "selected_step": summary["selected_step"]}))
        (run / "export" / "manifest.json").write_text("{}")
        (run / "export" / "export.safetensors").write_bytes(b"")
        return run

    def test_reference_is_the_pilot_reference_with_the_mb1_stream(self) -> None:
        pilot = binary_sweep.arm_reference()
        self.assertEqual(self.reference["stream"]["config_hash"], MB1_HASH)
        self.assertEqual(self.reference["stream"]["teacher_py_sha256"], pilot["stream"]["teacher_py_sha256"])
        self.assertEqual(self.reference["quantized_modules_sha256"], pilot["quantized_modules_sha256"])
        self.assertEqual(self.expected.args()["seed"], 20261017)
        for name in ("parakeet-binary/binary_main.py", "parakeet-binary/binary_main_paths.py",
                     *binary_paths.BINARY_SOURCES, *sweep.TRAINING_SOURCES):
            self.assertIn(name, self.sources)

    def test_the_planned_run_validates_and_any_difference_fails(self) -> None:
        summary, config = self.records()
        run = self.write("ok", summary, config)
        self.assertEqual(binary_sweep.validate_arm(run, self.expected, self.sources)["run_name"],
                         "main-binB-m1-lr2e-4")
        main_py, main_paths_py = "parakeet-binary/binary_main.py", "parakeet-binary/binary_main_paths.py"
        pilot_sources = {k: v for k, v in self.sources.items() if k not in (main_py, main_paths_py)}
        # label -> edits, each (record, key path, value)
        cases = {
            "steps": [("s", ("max_steps",), 125000), ("s", ("steps",), 125000), ("s", ("selected_step",), 125000),
                      ("c", ("args", "max_steps"), 125000)],
            "summary steps only": [("s", ("steps",), 249999)],
            "config steps only": [("c", ("args", "max_steps"), 125000)],
            "learning rate": [("s", ("lr",), 5e-4)],
            "config learning rate": [("c", ("args", "lr"), 5e-4)],
            "init": [("s", ("binary", "init"), "fp32")],
            "config init": [("c", ("args", "init"), "fp32")],
            "quantizer": [("s", ("binary", "quantizer"), "ternary")],
            "config quantizer": [("c", ("args", "quantizer"), "ternary")],
            "seed": [("s", ("seed",), paths.SEED)],
            "stream seed": [("s", ("stream", "seed"), paths.SEED)],
            "config seed": [("c", ("args", "seed"), paths.SEED)],
            "stream phase": [("s", ("stream", "phase"), "pilot"), ("s", ("train",), "pilot")],
            "stream phase only": [("s", ("stream", "phase"), "pilot")],
            "stream hash (M1's order)": [("s", ("stream", "config_hash"), binary_paths.STREAM_CONFIG_HASH["main"])],
            "config stream hash": [("c", ("stream", "config_hash"), "0" * 64)],
            "source hash": [("s", ("source_hashes", main_py), "0" * 64)],
            "config source hash": [("c", ("training_source_hashes", main_paths_py), "0" * 64)],
            "pilot source set": [("s", ("source_hashes",), pilot_sources)],
        }
        for label, edits in cases.items():
            s, c = copy.deepcopy(summary), copy.deepcopy(config)
            for record, keys, value in edits:
                node = s if record == "s" else c
                for key in keys[:-1]:
                    node = node[key]
                node[keys[-1]] = value
            with self.subTest(label):
                bad = self.write("bad", s, c)
                with self.assertRaises(RuntimeError):
                    binary_sweep.validate_arm(bad, self.expected, self.sources)


class MainReportTest(unittest.TestCase):
    """plans/main.md "Reported": MB1 against M1, B0 and pilot B2, mean and per set (numbers only)."""

    def test_report_differences(self) -> None:
        paths.require_mount()
        refs = binary_sweep.references()
        summary = {"run_name": "main-binB-m1-lr2e-4", "lr": 2e-4, "dev_mean_wer": 0.065, "seed": 20261017,
                   "steps": 250000, "dev_wer": {s: 0.06 + 0.002 * i for i, s in enumerate(SETS)},
                   "started_utc": "2026-10-17T12:00:00Z", "finished_utc": "2026-10-20T12:00:00Z",
                   "binary": {"timing": {"train_wall_s": 2.4e5, "sessions": 2}}}
        root = fx.TEST_ROOT / "main-report"
        shutil.rmtree(root, ignore_errors=True)
        (root / "runs").mkdir(parents=True)
        for name in (binary_sweep.SELECTION_NAME, f"{binary_sweep.ARMS['B2'].run_name}/summary.json"):
            (root / "runs" / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(binary_paths.RUNS / name, root / "runs" / name)
        b2 = json.loads((binary_paths.RUNS / binary_sweep.ARMS["B2"].run_name / "summary.json").read_text())
        with mock.patch.object(binary_paths, "RUNS", root / "runs"):
            report = binary_sweep.main_report(summary, refs, {"gate": {"pass": True}}, binary_sweep.main_plan(),
                                              binary_sweep.main_reference())
        shutil.rmtree(root, ignore_errors=True)
        json.dumps(report)  # serializable
        for key, ref in (("MB1_minus_M1", refs["m1"]), ("MB1_minus_B0", refs["b0"]), ("MB1_minus_pilot_B2", b2)):
            diff = report["comparisons"][key]
            self.assertAlmostEqual(diff["mean"], 0.065 - ref["dev_mean_wer"])
            self.assertEqual(set(diff["per_set"]), set(SETS))
            for s in SETS:
                self.assertAlmostEqual(diff["per_set"][s], summary["dev_wer"][s] - ref["dev_wer"][s])
        self.assertEqual(report["mb1"]["train_wall_s"], 2.4e5)
        # Review round 1, finding 2: MB1 - M1 is a net change, not a causal binary cost.
        self.assertEqual(report["readings"]["MB1_minus_M1"],
                         "net change after binary conversion and 250,000 further steps from M1 (not a causal "
                         "binary cost; see plans/main.md 'Reported')")
        self.assertEqual(report["stream_pin"], {"train": "main", "seed": 20261017, "workers": 8,
                                                "config_hash": MB1_HASH})
        self.assertEqual(report["protocol"]["seed"], 20261017)
        self.assertEqual(report["protocol"]["steps"], 250000)
        self.assertEqual(report["protocol"]["eval_every"], 12500)


# ----------------------------------------------------------------------------- contract 7

class MainResumeTest(unittest.TestCase):
    """A run directory holding a checkpoint of a pilot arm, another seed or another max_steps is
    never resumed as MB1 (train.py exit 78); MB1's own checkpoint is."""

    def test_only_mb1_checkpoints_resume_as_mb1(self) -> None:
        root = fx.TEST_ROOT / "main-resume"
        shutil.rmtree(root, ignore_errors=True)
        with fake_init(), binary_main.seam("binary"):
            mb1 = train.parse_args(MB1_ARGS)
            self.assertEqual((mb1.seed, mb1.train, mb1.max_steps), (20261017, "main", 250000))
            others = {"pilot arm B2": train.parse_args(B2_ARGS)}
            for field, value in (("seed", paths.SEED), ("seed", 20261018), ("max_steps", 125000),
                                 ("eval_every", 1200)):
                other = copy.deepcopy(mb1)
                setattr(other, field, value)
                others[f"{field} {value}"] = other
            for label, other in [("MB1 itself", mb1), *others.items()]:
                run = root / label.replace(" ", "-")
                train.save_checkpoint(run, {"step": 1, "args": vars(other), "resume_key": train.resume_key(other)})
                # A started run records its sources at the first start (review round 1, finding 1).
                (run / "config.json").write_text(json.dumps({"training_source_hashes": train.training_source_hashes()}))
                with self.subTest(label):
                    if other is mb1:
                        self.assertEqual(train.load_resume_payload(run, mb1)["step"], 1)
                        continue
                    with self.assertRaises(SystemExit) as caught:
                        train.load_resume_payload(run, mb1)
                    self.assertEqual(caught.exception.code, train.EXIT_RESUME_MISMATCH)
        shutil.rmtree(root, ignore_errors=True)


class Reached(Exception):
    """Raised by the stubbed build_context: the resume got past every refusal."""


class SourceProvenanceResumeTest(unittest.TestCase):
    """Review round 1, finding 1: binary_main.py refuses (exit EXIT_SOURCES_CHANGED, which
    sweep.run_train does not retry) to resume a run whose training sources, binary_main.py and
    binary_main_paths.py included, differ from those its config.json recorded at the first start;
    checked before any checkpoint is restored, on a direct rerun and on run_train's exit-75 retries."""

    def setUp(self) -> None:
        self.root = fx.TEST_ROOT / "main-provenance"
        shutil.rmtree(self.root, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.root, True)
        self.changed: set[str] = set()
        real_sha = train.sha256

        def sha256(path):  # a source file "edited" on disk: its hash differs from now on
            return "0" * 64 if Path(path).name in self.changed else real_sha(path)

        self.reached: list = []

        def build_context(args, run, device="cuda"):
            self.reached.append(run)
            raise Reached(str(run))

        for cm in (fake_init(), mock.patch.object(binary_paths, "RUNS", self.root / "runs"),
                   mock.patch.object(train, "sha256", side_effect=sha256),
                   mock.patch.object(train.torch.cuda, "is_available", return_value=True),
                   mock.patch.object(train, "deterministic_mode", return_value={}),
                   mock.patch.object(paths, "gpu_lock", side_effect=lambda label: contextlib.nullcontext()),
                   mock.patch.object(binary_train, "build_context", side_effect=build_context)):
            cm.__enter__()
            self.addCleanup(cm.__exit__, None, None, None)
        # A started MB1 run: config.json written at the first start, one checkpoint.
        with binary_main.seam("binary"):
            args = train.parse_args(MB1_ARGS)
            self.recorded = train.training_source_hashes()
        self.run_dir = self.root / "runs" / args.run_name
        train.save_checkpoint(self.run_dir, {"step": 1, "args": vars(args), "resume_key": None})
        (self.run_dir / "config.json").write_text(json.dumps({"args": vars(args),
                                                              "training_source_hashes": self.recorded}))

    def launch(self) -> int:
        """binary_main.py MB1_ARGS in this process: its exit code, or 0 once build_context is reached."""
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                binary_main.main(MB1_ARGS)
        except SystemExit as stop:
            return stop.code
        except Reached:
            return 0
        return 0

    def test_the_main_sources_are_recorded(self) -> None:
        for name in ("parakeet-binary/binary_main.py", "parakeet-binary/binary_main_paths.py",
                     "parakeet-binary/binary_train.py", "parakeet-ternary/train.py"):
            self.assertIn(name, self.recorded)
        self.assertNotIn(binary_main.EXIT_SOURCES_CHANGED, (sweep.EXIT_STREAM_FAILURE, train.EXIT_RESUME_MISMATCH,
                                                            sweep.EXIT_INTERRUPTED, sweep.EXIT_UNREADABLE_CHECKPOINT, 0))

    def test_unchanged_sources_resume(self) -> None:
        self.assertEqual(self.launch(), 0)
        self.assertEqual(self.reached, [self.run_dir])

    def test_a_changed_source_is_refused_on_a_direct_resume(self) -> None:
        for name in ("binary_main.py", "binary_main_paths.py", "binary_train.py", "stream.py"):
            with self.subTest(name):
                self.changed = {name}
                with mock.patch.object(train.Context, "restore", side_effect=AssertionError("restored")):
                    self.assertEqual(self.launch(), binary_main.EXIT_SOURCES_CHANGED)
                self.assertEqual(self.reached, [])
                self.assertTrue((self.run_dir / "ckpt" / "latest.pt").exists())  # nothing discarded

    def test_a_checkpoint_without_its_recorded_sources_is_refused(self) -> None:
        (self.run_dir / "config.json").unlink()
        self.assertEqual(self.launch(), binary_main.EXIT_SOURCES_CHANGED)
        self.assertEqual(self.reached, [])

    def test_a_source_changed_between_exit_75_retries_is_refused_and_not_retried(self) -> None:
        calls = []

        def runner(cmd):
            calls.append(cmd)
            code = self.launch()
            if len(calls) == 1:  # this session reached training, then the stream failed
                self.assertEqual(code, 0)
                self.changed = {"binary_main.py"}  # edited during the outage
                code = sweep.EXIT_STREAM_FAILURE
            return mock.Mock(returncode=code)

        cmd = binary_sweep.train_command(binary_sweep.main_plan(), binary_sweep.MAIN_ARM)
        with self.assertRaises(RuntimeError) as caught:
            sweep.run_train(cmd, dry=False, sleep=lambda s: None, runner=runner)
        self.assertIn(str(binary_main.EXIT_SOURCES_CHANGED), str(caught.exception))
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.reached, [self.run_dir])  # only the first session


# ----------------------------------------------------------------------------- launch script

LAUNCH_MAIN = binary_paths.ARTIFACTS / "launch-main.sh"
NVIDIA_SMI_STORE = "/nix/store/42mxdz5ml46z3svc72q3jalk8359bi6i-nvidia-x11-595.71.05-bin/bin/nvidia-smi"


class LaunchScriptTest(unittest.TestCase):
    """Review round 1, finding 4: launch-main.sh starts the sweep only with a
    positive numeric GPU power limit of at most 400 W. A stub nvidia-smi (NVIDIA_SMI) and a stub
    ./python in the working directory stand in; the real GPU is not queried or changed."""

    def setUp(self) -> None:
        self.root = fx.TEST_ROOT / "launch-main"
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root, True)
        python = self.root / "python"
        python.write_text('#!/usr/bin/env bash\necho "STUB python $*"\n')
        python.chmod(0o755)

    def launch(self, output: str | None, status: int = 0):
        smi = self.root / "nvidia-smi"
        smi.write_text("#!/usr/bin/env bash\n" + (f"printf '%s\\n' {output!r}\n" if output is not None else "")
                       + f"exit {status}\n")
        smi.chmod(0o755)
        return subprocess.run(["bash", str(LAUNCH_MAIN)], cwd=self.root, capture_output=True, text=True,
                              env={"PATH": "/run/current-system/sw/bin:/usr/bin:/bin", "NVIDIA_SMI": str(smi)})

    def test_default_is_the_system_nvidia_smi(self) -> None:
        self.assertIn(f'NVIDIA_SMI:-{NVIDIA_SMI_STORE}', LAUNCH_MAIN.read_text())

    def test_limits_at_or_below_400_w_launch_the_main_phase(self) -> None:
        for value in ("400.00", "400", "350.5", " 400.00 "):
            with self.subTest(value):
                done = self.launch(value)
                self.assertEqual(done.returncode, 0, done.stderr)
                self.assertEqual(done.stdout.strip(), "STUB python binary_sweep.py --phase main")

    def test_unknown_or_too_high_limits_are_refused(self) -> None:
        for value in ("", "N/A", "[N/A]", "abc", "0", "0.00", "-5", "400W", "450.00", "400.01", "1e3"):
            with self.subTest(value):
                done = self.launch(value)
                self.assertNotEqual(done.returncode, 0)
                self.assertNotIn("STUB python", done.stdout)
                self.assertIn("launch-main:", done.stderr)
        for output, status in ((None, 0), ("400.00", 9)):  # no output; the query itself failing
            with self.subTest(output=output, status=status):
                done = self.launch(output, status)
                self.assertNotEqual(done.returncode, 0)
                self.assertNotIn("STUB python", done.stdout)
                self.assertIn("launch-main:", done.stderr)


# ----------------------------------------------------------------------------- contract 8

class PilotUnchangedTest(unittest.TestCase):
    """The finished pilot keeps validating: the binary sources its arms and the gate recorded are
    unchanged, and only MB1 records the two new files."""

    def test_binary_sources_match_every_pilot_record(self) -> None:
        paths.require_mount()
        now = binary_sweep.current_sources(binary_sweep.TRAINING_SOURCES)
        records = {arm.run_name: binary_paths.RUNS / arm.run_name / "summary.json"
                   for arm in binary_sweep.ARMS.values()}
        records["G-B"] = binary_paths.EVAL / "G-B" / "summary.json"
        for label, path in records.items():
            recorded = json.loads(path.read_text())["source_hashes"]
            for name in binary_paths.BINARY_SOURCES:
                with self.subTest(label, source=name):
                    self.assertEqual(recorded[name], now[name])

    def test_source_sets(self) -> None:
        new = ("parakeet-binary/binary_main.py", "parakeet-binary/binary_main_paths.py")
        self.assertEqual(binary_sweep.TRAINING_SOURCES, tuple(sweep.TRAINING_SOURCES) + binary_paths.BINARY_SOURCES)
        self.assertEqual(binary_sweep.MAIN_TRAINING_SOURCES, binary_sweep.TRAINING_SOURCES + new)
        with binary_train.seam("binary"):
            self.assertFalse(set(new) & set(train.training_source_hashes()))
        with binary_main.seam("binary"):
            recorded = train.training_source_hashes()
        for name in new:
            self.assertEqual(recorded[name], sha(paths.REPO / "finetune" / name))
        self.assertEqual(binary_train.BINARY_SOURCES, binary_paths.BINARY_SOURCES)  # restored on exit
        self.assertEqual(binary_train.open_stream.__module__, "binary_train")
        self.assertIs(train.parse_args, binary_train._TRAIN["parse_args"])


if __name__ == "__main__":
    unittest.main()
