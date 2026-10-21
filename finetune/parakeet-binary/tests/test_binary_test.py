"""Contract tests of the MB1 test-set scoring (plans/test-scoring.md "Contract tests"): the precheck,
the dev-check gate and its binding, test-set hygiene (attempt record, --record-failure,
--summarize), the scoring path and the report.

No GPU and no real decoding: every test builds a small fake world (export directory, summaries,
manifests, baseline outputs, attempt records) under binary_paths.TMP/tests/test-scoring and
points binary_test_paths and paths.MANIFESTS at it. The model load (binary_export.load_export),
evaluate.evaluate_set and paths.gpu_lock are stubbed where the scoring path is asserted; the
decoder-mode helpers of binary_train run for real against a stand-in decoding computer. The
normalizer is the real one (evaluate.normalize).
"""
from __future__ import annotations

import binary_fixtures as fx  # noqa: I001  first: CPU only

import contextlib
import copy
import hashlib
import inspect
import io
import json
import shutil
import threading
import time
import types
import unittest
import weakref
from pathlib import Path
from unittest import mock

import binary_export
import binary_report_test
import binary_test
import binary_test_paths as tp
import binary_train
import evaluate
import paths

SETS = list(paths.TEST_SETS)
DEV = list(paths.DEV_SETS)
ROOT = fx.TEST_ROOT / "test-scoring"
REAL_EVALUATE_SET = evaluate.evaluate_set
# MB1's recorded full-dev WERs (summary.json "dev_wer"), whose mean is tp.DEV_MEAN.
DEV_WER = {"librispeech_dev_clean": 0.021551881455527572, "librispeech_dev_other": 0.04686772240335352,
           "ami_dev": 0.11952913728193436, "voxpopuli_dev": 0.06063573022450931,
           "yodas_dev": 0.06701361294126695}
# Test-set WER fractions of the fixture arms (known values for the report).
B0_WER = {"librispeech_clean": 0.017, "librispeech_other": 0.032, "ami": 0.111, "earnings22": 0.112,
          "gigaspeech": 0.097, "spgispeech": 0.022, "voxpopuli": 0.060, "common_voice": 0.060}
M1_WER = {"librispeech_clean": 0.019, "librispeech_other": 0.040, "ami": 0.104, "earnings22": 0.120,
          "gigaspeech": 0.100, "spgispeech": 0.030, "voxpopuli": 0.065, "common_voice": 0.080}
MB1_WER = {"librispeech_clean": 0.020, "librispeech_other": 0.044, "ami": 0.110, "earnings22": 0.125,
           "gigaspeech": 0.102, "spgispeech": 0.031, "voxpopuli": 0.066, "common_voice": 0.090}
# References exercising the normalizer (numbers, spelling, punctuation, contractions, fillers).
TEXTS = ["Mister Smith's 2 dogs ate the colour-coded biscuits!", "Uh, we're going at 5:30 p.m., OK?",
         "It cost $3.50 — about twenty percent less."]
HYPS = ["Mr. Smith's two dogs ate the color coded biscuits.", "We are going at five thirty PM okay",
        "It cost three dollars fifty, about 20% less."]
DECODING_CONFIG = {"strategy": "greedy_batch", "model_type": "tdt", "durations": [0, 1, 2, 3, 4],
                   "greedy": {"max_symbols": 10},
                   "beam": {"beam_size": 2, "return_best_hypothesis": False, "score_norm": True,
                            "tsd_max_sym_exp": 50, "alsd_max_target_len": 2.0}}


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decoding(batch_size: int = 64, max_batch_seconds: float = 1200.0) -> dict:
    return {"method": "preprocessor -> encoder -> decoding.rnnt_decoder_predictions_tensor (transcribe() path)",
            "config": copy.deepcopy(DECODING_CONFIG),
            "precision": "float32, TF32 off (matmul and cuDNN), autocast disabled",
            "batch_size": batch_size, "max_batch_seconds": max_batch_seconds, "batch_order": "descending duration"}


CURRENT_SCORER = {key: sha(tp.scorer_path(key)) for key in tp.SCORER_HASHES}
PINNED_KEYS = list(tp.PINNED_FILES)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n")


def read_json(path: Path):
    return json.loads(Path(path).read_text())


class FakeComputer:
    """Stands in for NeMo's label-looping decoding computer (binary_train's mode helpers)."""

    allow_cuda_graphs = True

    def __init__(self) -> None:
        self.cuda_graphs_mode = types.SimpleNamespace(value="full_graph")  # NeMo's default pick

    def force_cuda_graphs_mode(self, mode) -> None:
        self.cuda_graphs_mode = types.SimpleNamespace(value=str(mode))


class FakeModel:
    """Stands in for the rebuilt NeMo model: model.decoding.decoding.decoding_computer (weak-referenceable,
    so the lifetime tests can watch it die)."""

    def __init__(self) -> None:
        self.computer = FakeComputer()
        self.decoding = types.SimpleNamespace(decoding=types.SimpleNamespace(decoding_computer=self.computer))


def fake_model():
    return FakeModel()


class World:
    """A complete fake set of inputs: MB1 run and export, training log and journal, test and dev
    manifests, M1-test and b0-pretrained outputs, and empty output locations."""

    def __init__(self, name: str) -> None:
        self.root = ROOT / name
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True)
        self.manifests = self.root / "manifests"
        self.run = self.root / "runs" / tp.RUN_NAME
        self.export = self.run / "export"
        self.m1 = self.root / "ternary-eval" / "M1-test"
        self.b0 = self.root / "ternary-eval" / "b0-pretrained"
        self.eval = self.root / "eval"
        self.results = self.root / "results"
        self.log = self.root / "logs" / "binary-main.log"
        self.plan = self.root / "plan.md"
        self.journal: list[str] | None = [
            "Started [systemd-run] /mnt/hd/wilderness-labs-stt/parakeet-binary/launch-main.sh.",
            "parakeet-binary-binary-main.service: Consumed 3d 36min CPU time, 32.2G memory peak."]
        self.units: list[str] | None = []
        self.own_unit: str | None = None
        self.dev_wer = dict(DEV_WER)  # what the stubbed evaluate_set returns per dev set
        self.mb1_wer = dict(MB1_WER)  # ... and per test set
        self.empty_sets: set[str] = set()  # test sets whose stubbed MB1 hypotheses are all empty
        self.calls: list[dict] = []
        self.loads: list[dict] = []
        self.lock_held = False
        self.lock_events: list[str] = []  # "gpu+", "gpu-" (the fake shared GPU lock)
        self.on_evaluate = None
        self.model = None
        self.pins: dict[str, str] = {}
        self.build()

    # ------------------------------------------------------------------ building
    def build(self) -> None:
        for name in SETS:
            self.write_manifest(self.manifests / f"test_{name}.jsonl", name)
        for name in DEV:
            self.write_manifest(self.manifests / f"{name}.jsonl", name)
        b0_source = {"kind": "pretrained", "description": "pretrained Parakeet-TDT-0.6B-v2, FP32, no training",
                     "path": str(paths.MODEL_FILE), "weights_file": str(paths.MODEL_FILE),
                     "sha256": tp.BASE_MODEL["nemo_sha256"], "base_model": dict(tp.BASE_MODEL)}
        m1_source = {"kind": "export", "description": "ternary export, rebuilt with dequantized FP32 weights",
                     "path": "/m1/export", "weights_file": "/m1/export/export.safetensors",
                     "sha256": tp.M1_EXPORT_SHA256, "base_model": dict(tp.BASE_MODEL),
                     "export_manifest": {"format": "parakeet-ternary-v1", "sha256": tp.M1_EXPORT_SHA256,
                                         "bytes": {"file_bytes": 180797564, "original_nemo_bytes": 2472222720}}}
        self.write_arm(self.b0, b0_source, B0_WER, empty={"ami": [2]})
        self.write_arm(self.m1, m1_source, M1_WER, empty={})
        export = self.export
        (export / "tokenizer").mkdir(parents=True)
        (export / "export.safetensors").write_bytes(b"fake MB1 packed weights")
        for name in ("tokenizer.model", "tokenizer.vocab", "vocab.txt"):
            (export / "tokenizer" / name).write_bytes(f"fake {name}".encode())
        files = {rel: sha(export / rel) for rel in ("export.safetensors", "tokenizer/tokenizer.model",
                                                    "tokenizer/tokenizer.vocab", "tokenizer/vocab.txt")}
        write_json(export / "manifest.json", {
            "format": tp.FORMAT, "file": "export.safetensors", "sha256": files["export.safetensors"],
            "files": files, "base_model": {"id": tp.BASE_MODEL["id"], "revision": tp.BASE_MODEL["revision"]},
            "config": {"decoding": copy.deepcopy(DECODING_CONFIG)},
            "parameter_accounting": {"binary_parameters": 603979776},
            "bytes": {"file_bytes": 105299804, "original_nemo_bytes": 2472222720},
            "extra": {"run_name": tp.RUN_NAME, "selected_step": 250000}})
        recon = {"codes_exact": True, "scales_exact": True, "greedy_hyps_equal": True,
                 "encoder_max_abs_diff": 0.0005, "format": tp.FORMAT}
        write_json(export / "reconstruction.json", recon)
        write_json(self.run / "config.json", {"args": {"run_name": tp.RUN_NAME, "seed": 20261017}})
        summary = {**tp.EXPECT_SUMMARY, "arm": "P2", "recipe": "P2",
                   "config_sha256": sha(self.run / "config.json"),
                   "source_hashes": {k: sha(paths.REPO / "finetune" / k)
                                     for k in ("parakeet-ternary/evaluate.py", "parakeet-binary/binary_export.py",
                                               "parakeet-binary/python")},
                   "reconstruction": dict(recon), "dev_wer": dict(DEV_WER), "dev_mean_wer": tp.DEV_MEAN,
                   "binary": {"init": "m1", "quantizer": "binary", "ramp_kind": "ternary"}}
        write_json(self.run / "summary.json", summary)
        self.repin()
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.log.write_text("[binary_sweep 08:46:11] systemd unit: parakeet-binary-binary-main.service\n"
                            "[binary_sweep 08:46:11] phase main; protocol {}; arms ['MB1']\n"
                            "[train 05:23:12] main-binB-m1-lr2e-4: selected (final) step 250000\n"
                            "[binary_sweep 05:23:18] MB1 full dev 6.312%\n"
                            "[binary_sweep 05:23:18] done\n")
        self.write_plan()

    def write_plan(self, digest: str | None = None) -> None:
        digest = sha(Path(tp.__file__)) if digest is None else digest
        self.plan.write_text(f"# Plan\n\n**Status: 2026-10-20: frozen; binary_test_paths.py SHA-256 `{digest}`.**\n\n"
                             "Body.\n")

    @staticmethod
    def write_manifest(path: Path, name: str, ids: list[str] | None = None) -> None:
        ids = ids or [f"{name}:{i}" for i in range(3)]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps({"id": uid, "audio_filepath": f"/nonexistent/{uid}.flac",
                                            "duration": 1.0 + i, "text": TEXTS[i % 3]}) + "\n"
                                for i, uid in enumerate(ids)))

    def write_arm(self, out: Path, source: dict, wer: dict, empty: dict) -> None:
        sets = {}
        for name in SETS:
            manifest = self.manifests / f"test_{name}.jsonl"
            records = []
            for i, record in enumerate(evaluate.read_manifest(manifest)):
                hyp = "" if i in empty.get(name, []) else HYPS[i % 3]
                records.append({"id": record["id"], "ref": record["text"], "hyp": hyp,
                                **evaluate.score(record["text"], hyp), "duration": record["duration"]})
            write_json(out / f"{name}.json", {
                "source": source, "set": name, "manifest": str(manifest), "manifest_sha256": sha(manifest),
                "utterances": len(records), "limit": None, "audio_hours": 0.001, "decoding": decoding(),
                "normalizer": evaluate.NORMALIZER,
                "wer": {"wer": wer[name], "utterances": len(records), "excluded_empty_reference": 0},
                "wall_s": 1.0, "records": records})
            sets[name] = {"wer": 100 * wer[name], "utterances": len(records), "limit": None}
        write_json(out / "summary.json", {"source": source, "sets": sets, "wall_s": 8.0})

    def repin(self) -> None:
        """Pin the fixture's current files (in place: the installed tp.PINNED_FILES is this dict)."""
        pins = {rel: sha(self.run / rel) for rel in PINNED_KEYS}
        self.pins.clear()
        self.pins.update(pins)

    def edit_json(self, path: Path, fn, repin: bool = True) -> None:
        data = read_json(path)
        fn(data)
        write_json(path, data)
        if repin:
            self.repin()

    # ------------------------------------------------------------------ stubs
    @contextlib.contextmanager
    def gpu_lock(self, label: str):
        self.lock_events.append("gpu+")
        self.lock_held = True
        try:
            yield
        finally:
            self.lock_held = False
            self.lock_events.append("gpu-")

    def load_export(self, out_dir, device="cpu"):
        self.loads.append({"path": Path(out_dir), "device": str(device), "lock_held": self.lock_held})
        self.model = fake_model()
        return self.model

    def evaluate_set(self, *args, **kwargs):
        bound = inspect.signature(REAL_EVALUATE_SET).bind(*args, **kwargs)
        bound.apply_defaults()
        a = dict(bound.arguments)
        self.calls.append({**a, "lock_held": self.lock_held, "mode": a["model"].computer.cuda_graphs_mode.value})
        if self.on_evaluate:
            self.on_evaluate(a)
        return self.result(a)

    def result(self, a: dict) -> dict:
        """The stubbed evaluate_set output for bound arguments `a` (keeps no reference to the model)."""
        name, manifest = a["name"], Path(a["manifest"])
        records = evaluate.read_manifest(manifest)[:a["limit"]]
        out = []
        for r in records:
            hyp = "" if name in self.empty_sets else r["text"]
            out.append({"id": r["id"], "ref": r["text"], "hyp": hyp, **evaluate.score(r["text"], hyp),
                        "duration": r["duration"]})
        wer = (self.dev_wer if name in DEV else self.mb1_wer)[name]
        return {"source": a["source"], "set": name, "manifest": str(manifest),
                "manifest_sha256": evaluate.sha256_file(manifest), "utterances": len(out), "limit": a["limit"],
                "audio_hours": sum(r["duration"] for r in out) / 3600,
                "decoding": decoding(a["batch_size"], a["max_batch_seconds"]), "normalizer": evaluate.NORMALIZER,
                "wer": {"wer": wer, "substitutions": 0, "deletions": 0, "insertions": 0, "ref_words": 10,
                        "hyp_words": 10, "utterances": len(out), "excluded_empty_reference": 0},
                "wall_s": 1.0, "padded_short": 0, "padded_short_ids": [], "min_decode_seconds": 0.03,
                "rtfx": 1.0, "peak_gpu_memory_gb": None, "records": out}

    @contextlib.contextmanager
    def installed(self):
        attrs = {"RUN": self.run, "EXPORT": self.export, "PINNED_FILES": self.pins,
                 "WEIGHTS_SHA256": self.pins["export/export.safetensors"], "M1_TEST": self.m1, "B0_TEST": self.b0,
                 "EVAL": self.eval, "PRECHECK": self.eval / "MB1-precheck.json",
                 "DEV_CHECK": self.eval / "MB1-dev-check", "DEV_CHECK_B32": self.eval / "MB1-dev-check-b32",
                 "TEST_OUT": self.eval / "MB1-test", "ATTEMPTS": self.eval / "MB1-test.attempts.jsonl",
                 "ATTEMPTS_LOCK": self.eval / "MB1-test.attempts.lock",
                 "RESULTS": self.results, "MAIN_LOG": self.log, "PLAN": self.plan,
                 "SCORER_HASHES": dict(CURRENT_SCORER)}
        with contextlib.ExitStack() as stack:
            for key, value in attrs.items():
                stack.enter_context(mock.patch.object(tp, key, value))
            stack.enter_context(mock.patch.object(paths, "MANIFESTS", self.manifests))
            stack.enter_context(mock.patch.object(binary_test, "journal_messages", lambda unit: self.journal))
            stack.enter_context(mock.patch.object(binary_test, "active_units", lambda: self.units))
            stack.enter_context(mock.patch.object(binary_test, "own_unit", lambda: self.own_unit))
            stack.enter_context(mock.patch.object(binary_test, "DEVICE", "cpu"))
            stack.enter_context(mock.patch.object(paths, "gpu_lock", self.gpu_lock))
            stack.enter_context(mock.patch.object(binary_export, "load_export", side_effect=self.load_export))
            stack.enter_context(mock.patch.object(evaluate, "evaluate_set", side_effect=self.evaluate_set))
            yield

    # ------------------------------------------------------------------ ledger and outputs
    def ledger(self) -> list[dict]:
        path = self.eval / "MB1-test.attempts.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []

    def write_ledger(self, entries: list[dict]) -> None:
        path = self.eval / "MB1-test.attempts.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(e) + "\n" for e in entries))

    def tamper_mb1(self, name: str, fn, rehash: bool = True) -> None:
        """Edit MB1's per-set output; with rehash, update the complete attempt's set_done hash and the
        summary's per-set hash so that only the edited content is wrong."""
        path = self.eval / "MB1-test" / f"{name}.json"
        doc = read_json(path)
        fn(doc)
        write_json(path, doc)
        if rehash:
            digest = sha(path)
            entries = self.ledger()
            for e in entries:
                if e.get("state") == "set_done" and e.get("set") == name:
                    e["sha256"] = digest
            self.write_ledger(entries)
            self.edit_json(self.eval / "MB1-test" / "summary.json",
                           lambda s: s["per_set_sha256"].__setitem__(name, digest), repin=False)


def run_main(argv: list[str]) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        code = binary_test.main(argv)
    return code, out.getvalue()


def run_report() -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = binary_report_test.main()
    return code, out.getvalue()


class WorldCase(unittest.TestCase):
    def setUp(self) -> None:
        paths.require_mount()
        self.w = World(self.id().rsplit(".", 2)[-2] + "-" + self._testMethodName)
        self.enterContext(self.w.installed())
        self.addCleanup(shutil.rmtree, self.w.root, True)

    def refused(self, argv: list[str], word: str) -> SystemExit:
        with self.assertRaises(SystemExit) as caught:
            run_main(argv)
        self.assertIn(word, str(caught.exception))
        return caught.exception

    def report_refused(self, word: str) -> None:
        with self.assertRaises(SystemExit) as caught:
            run_report()
        self.assertIn(word, str(caught.exception))

    def failed(self, result: dict) -> list[tuple[int, str]]:
        return [(c["item"], c["check"]) for c in result["checks"] if not c["ok"]]

    def assert_item_fails(self, item: int, word: str) -> None:
        result = binary_test.precheck()
        failing = self.failed(result)
        self.assertFalse(result["pass"])
        self.assertTrue(any(i == item and word in name for i, name in failing),
                        f"no failing item {item} check naming {word!r}: {failing}")

    def dev_check(self, *extra: str) -> int:
        return run_main(["--dev-check", *extra])[0]

    def complete(self) -> None:
        """A passing dev check and a complete test run."""
        self.assertEqual(self.dev_check(), 0)
        self.assertEqual(run_main(["--test"])[0], 0)


# ============================================================================ pinned constants

class PinnedConstantsTest(unittest.TestCase):
    def test_identity_pins_are_the_plans(self) -> None:
        self.assertEqual(tp.PINNED_FILES, {
            "export/export.safetensors": "9873d2f1a53127c673d0201db98cd3ba859ba432f334fef55e9e24fd057b9651",
            "export/manifest.json": "f3737a538da81939aabe04acd5334afd0e3cd4160ec113bd889e771b67991dbd",
            "export/reconstruction.json": "efa75346d366fbb92a49b1de72d569894c6338e84199271cbfc97d2f2696d234",
            "export/tokenizer/tokenizer.model": "5a3a82c48998709f9bc9f5db2924d99a899d20bf82550ca52272da7c7557b0a0",
            "export/tokenizer/tokenizer.vocab": "dbe7ec6bd066c4526080940e1614e6582b9161515e856c2c39a1d49267c88cbc",
            "export/tokenizer/vocab.txt": "70b25deed256d79365a3c290d6191c31cc2315d952faf18d5c3a334f46e4866e",
            "config.json": "3cf1478ce33e3126a288e57cb9d82b386639d5327f56067308cff0781ba8c6b8",
            "summary.json": "1dcecfcd1a08ec54094babfe58261d79c1df9c022b9b80a53eed716d20fb425a"})
        self.assertEqual(tp.WEIGHTS_SHA256, tp.PINNED_FILES["export/export.safetensors"])
        self.assertEqual(tp.DEV_MEAN, 0.06311961686131834)
        self.assertEqual(tp.M1_EXPORT_SHA256, "a287e97719c451b785be2cd01ecc861fcaa010ebaa4ff2841783ec78fcd61503")
        self.assertEqual(tp.EVALUATE_SHA256, "b3136a80558df15441b915568aca7199fbff878a392a2eed272f85fea9120a20")
        self.assertEqual(tp.BASE_MODEL["revision"], paths.MODEL_REVISION)
        self.assertEqual(tp.BASE_MODEL["id"], paths.MODEL_ID)
        self.assertEqual((tp.BATCH_SIZE, tp.MAX_BATCH_SECONDS), (64, 1200.0))
        self.assertEqual(tp.RUN, Path("/mnt/hd/wilderness-labs-stt/parakeet-binary/runs/main-binB-m1-lr2e-4"))

    def test_fixed_output_paths(self) -> None:
        eval_dir = Path("/mnt/hd/wilderness-labs-stt/parakeet-binary/eval")
        self.assertEqual(tp.EVAL, eval_dir)
        self.assertEqual(tp.PRECHECK, eval_dir / "MB1-precheck.json")
        self.assertEqual(tp.DEV_CHECK, eval_dir / "MB1-dev-check")
        self.assertEqual(tp.DEV_CHECK_B32, eval_dir / "MB1-dev-check-b32")
        self.assertEqual(tp.TEST_OUT, eval_dir / "MB1-test")
        self.assertEqual(tp.ATTEMPTS, eval_dir / "MB1-test.attempts.jsonl")
        self.assertEqual(tp.smoke_dir(16), eval_dir / "smoke-MB1-dev-limit16")
        self.assertEqual(tp.RESULTS, Path(binary_test.__file__).resolve().parent / "results")

    def test_points_to_fraction(self) -> None:
        self.assertEqual(tp.points_to_fraction(0.02), 0.0002)
        self.assertEqual(tp.points_to_fraction(0.10), 0.0010)
        self.assertEqual(tp.points_to_fraction(tp.MEAN_TOLERANCE_POINTS), tp.MEAN_TOLERANCE)
        self.assertEqual(tp.points_to_fraction(tp.SET_TOLERANCE_POINTS), tp.SET_TOLERANCE)
        self.assertEqual((tp.MEAN_TOLERANCE_POINTS, tp.SET_TOLERANCE_POINTS), (0.02, 0.10))

    def test_scorer_hashes_cover_the_plans_files_and_match_them(self) -> None:
        expected = {"parakeet-binary/binary_test.py", "parakeet-binary/binary_report_test.py",
                    "parakeet-binary/binary_export.py", "parakeet-binary/binary_train.py",
                    "parakeet-binary/binary_paths.py", "parakeet-binary/binary_quant.py",
                    "parakeet-ternary/evaluate.py", "parakeet-ternary/export.py", "parakeet-ternary/quant.py",
                    "parakeet-ternary/paths.py", "parakeet-ternary/train.py", "whisper-ternary/wer.py",
                    str(tp.NORMALIZER_MODULE), "stt/models/whisper-tiny.en/normalizer.json"}
        self.assertEqual(set(tp.SCORER_HASHES), expected)
        for key, digest in tp.SCORER_HASHES.items():
            with self.subTest(key):
                self.assertEqual(digest, CURRENT_SCORER[key])
        self.assertEqual(tp.SCORER_HASHES["parakeet-ternary/evaluate.py"], tp.EVALUATE_SHA256)

    def test_the_pinned_normalizer_is_the_one_evaluate_loads(self) -> None:
        import transformers.models.whisper.english_normalizer as module
        self.assertEqual(Path(module.__file__).resolve(), tp.NORMALIZER_MODULE.resolve())
        self.assertEqual(evaluate.NORMALIZER_JSON.resolve(), tp.scorer_path(tp.NORMALIZER_JSON).resolve())


# ============================================================================ precheck

class PrecheckTest(WorldCase):
    def test_the_fixture_world_passes_and_writes_its_record(self) -> None:
        code, out = run_main(["--precheck"])
        self.assertEqual(code, 0, out)
        self.assertIn("ALL PASS", out)
        record = read_json(self.w.eval / "MB1-precheck.json")
        self.assertTrue(record["pass"])
        self.assertEqual({c["item"] for c in record["checks"]}, {1, 2, 3, 4, 5, 6})
        self.assertEqual(self.w.loads, [])  # no model, no GPU

    def test_a_failing_precheck_exits_nonzero_and_says_so(self) -> None:
        self.w.units = ["parakeet-binary-other.service"]
        code, out = run_main(["--precheck"])
        self.assertEqual(code, 1)
        self.assertIn("PRECHECK FAILED", out)
        self.assertFalse(read_json(self.w.eval / "MB1-precheck.json")["pass"])

    # item 1
    def test_journal_with_failed_result(self) -> None:
        self.w.journal = self.w.journal + ["parakeet-binary-binary-main.service: Failed with result 'exit-code'."]
        self.assert_item_fails(1, "journal")

    def test_unreadable_or_empty_journal(self) -> None:
        for journal in (None, []):
            with self.subTest(journal=journal):
                self.w.journal = journal
                self.assert_item_fails(1, "journal")

    def test_log_not_ending_in_done(self) -> None:
        self.w.log.write_text(self.w.log.read_text() + "[binary_sweep 05:24:00] MB1 failed: stream\n")
        self.assert_item_fails(1, "done")

    def test_an_active_parakeet_unit(self) -> None:
        self.w.units = ["parakeet-binary-mb1-dev-check.service", "parakeet-train.service"]
        self.w.own_unit = "parakeet-binary-mb1-dev-check.service"
        self.assert_item_fails(1, "parakeet-*")

    def test_the_callers_own_unit_is_allowed(self) -> None:
        self.w.units = ["parakeet-binary-mb1-dev-check.service"]
        self.w.own_unit = "parakeet-binary-mb1-dev-check.service"
        self.assertTrue(binary_test.precheck()["pass"])

    def test_a_failed_unit_query(self) -> None:
        self.w.units = None
        self.assert_item_fails(1, "parakeet-*")

    # item 2
    def test_each_pinned_file_with_a_changed_hash(self) -> None:
        for rel in tp.PINNED_FILES:
            with self.subTest(rel):
                path = self.w.run / rel
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                try:
                    self.assert_item_fails(2, rel)
                finally:
                    path.write_bytes(original)
        self.assertTrue(binary_test.precheck()["pass"])

    def test_an_extra_or_missing_export_file(self) -> None:
        extra = self.w.export / "tokenizer" / "extra.txt"
        extra.write_text("x")
        self.assert_item_fails(2, "file set")
        extra.unlink()
        (self.w.export / "notes").mkdir()
        self.assert_item_fails(2, "file set")
        (self.w.export / "notes").rmdir()
        (self.w.export / "tokenizer" / "vocab.txt").unlink()
        self.assert_item_fails(2, "file set")

    def test_manifest_file_must_be_export_safetensors(self) -> None:
        self.w.edit_json(self.w.export / "manifest.json", lambda m: m.update(file="other.safetensors"))
        self.assert_item_fails(2, "manifest file ==")

    def test_manifest_sha256_or_files_entry_inconsistent(self) -> None:
        self.w.edit_json(self.w.export / "manifest.json", lambda m: m.update(sha256="0" * 64))
        self.assert_item_fails(2, "manifest sha256")
        self.w.edit_json(self.w.export / "manifest.json",
                         lambda m: m.update(sha256=self.w.pins["export/export.safetensors"]))
        self.w.edit_json(self.w.export / "manifest.json", lambda m: m["files"].update({"export.safetensors": "1" * 64}))
        self.assert_item_fails(2, "manifest files[")

    def test_manifest_format(self) -> None:
        self.w.edit_json(self.w.export / "manifest.json", lambda m: m.update(format="parakeet-ternary-v1"))
        self.assert_item_fails(2, "format")

    def test_summary_identity_fields(self) -> None:
        for key, value in (("run_name", "pilot-binB-m1-lr2e-4"), ("selected_step", 237500), ("seed", 20260930),
                           ("resumes", 1), ("steps", 249999), ("lr", 5e-4), ("select", "best"),
                           ("scored_on", "QAT model")):
            with self.subTest(key):
                original = read_json(self.w.run / "summary.json")[key]
                self.w.edit_json(self.w.run / "summary.json", lambda s: s.update({key: value}))
                self.assert_item_fails(2, f"summary.{key}")
                self.w.edit_json(self.w.run / "summary.json", lambda s: s.update({key: original}))
        self.w.edit_json(self.w.run / "summary.json", lambda s: s["binary"].update(init="fp32"))
        self.assert_item_fails(2, "summary.binary.init")

    def test_a_resumes_record_with_entries(self) -> None:
        (self.w.run / "resumes.jsonl").write_text('{"step": 100}\n')
        self.assert_item_fails(2, "resumes")

    def test_reconstruction_not_exact(self) -> None:
        for key in ("codes_exact", "scales_exact", "greedy_hyps_equal"):
            with self.subTest(key):
                def inexact(r, key=key):
                    r[key] = False
                self.w.edit_json(self.w.export / "reconstruction.json", inexact)
                self.w.edit_json(self.w.run / "summary.json", lambda s, key=key: s["reconstruction"].update({key: False}))
                self.assert_item_fails(2, "reconstruction")
                self.w.edit_json(self.w.export / "reconstruction.json", lambda r, key=key: r.update({key: True}))
                self.w.edit_json(self.w.run / "summary.json", lambda s, key=key: s["reconstruction"].update({key: True}))

    def test_reconstruction_disagreeing_with_the_summary(self) -> None:
        self.w.edit_json(self.w.run / "summary.json", lambda s: s["reconstruction"].update(encoder_max_abs_diff=1.0))
        self.assert_item_fails(2, "summary.reconstruction")

    # item 3
    def test_a_changed_recorded_source(self) -> None:
        self.w.edit_json(self.w.run / "summary.json",
                         lambda s: s["source_hashes"].update({"parakeet-binary/binary_export.py": "0" * 64}))
        self.assert_item_fails(3, "recorded")

    def test_a_changed_scorer_file(self) -> None:
        for key in ("parakeet-binary/binary_test.py", "parakeet-ternary/evaluate.py", "whisper-ternary/wer.py",
                    tp.NORMALIZER_JSON):
            with self.subTest(key), mock.patch.dict(tp.SCORER_HASHES, {key: "0" * 64}):
                self.assert_item_fails(3, key)

    def test_a_changed_normalizer_module_digest(self) -> None:
        with mock.patch.dict(tp.SCORER_HASHES, {str(tp.NORMALIZER_MODULE): "0" * 64}):
            self.assert_item_fails(3, "english_normalizer")

    def test_a_module_imported_from_another_path(self) -> None:
        import transformers.models.whisper.english_normalizer as normalizer
        for module in (evaluate, binary_export, normalizer):
            with self.subTest(module.__name__), mock.patch.object(module, "__file__", "/elsewhere/shadow.py"):
                self.assert_item_fails(3, module.__name__)

    def test_paths_hash_missing_from_the_plan_status(self) -> None:
        self.w.write_plan("0" * 64)
        self.assert_item_fails(3, "binary_test_paths.py")

    # item 4
    def test_a_test_manifest_differing_from_the_baselines(self) -> None:
        manifest = self.w.manifests / "test_ami.jsonl"
        manifest.write_text(manifest.read_text() + "\n")
        self.assert_item_fails(4, "ami")

    def test_a_baseline_recording_another_manifest(self) -> None:
        for arm in (self.w.m1, self.w.b0):
            with self.subTest(arm.name):
                path = arm / "voxpopuli.json"
                original = path.read_bytes()
                self.w.edit_json(path, lambda d: d.update(manifest_sha256="0" * 64), repin=False)
                self.assert_item_fails(4, "voxpopuli")
                path.write_bytes(original)

    def test_duplicate_utterance_ids(self) -> None:
        manifest = self.w.manifests / "test_earnings22.jsonl"
        World.write_manifest(manifest, "earnings22", ["earnings22:0", "earnings22:1", "earnings22:1"])
        digest = sha(manifest)
        for arm in (self.w.m1, self.w.b0):
            self.w.edit_json(arm / "earnings22.json", lambda d: d.update(manifest_sha256=digest), repin=False)
        result = binary_test.precheck()
        failing = self.failed(result)
        self.assertTrue(any(i == 4 and "unique" in n and "earnings22" in n for i, n in failing), failing)

    # item 5
    def test_a_saved_hyp_norm_the_current_normalizer_does_not_reproduce(self) -> None:
        def change(d):
            d["records"][0]["hyp_norm"] = d["records"][0]["hyp_norm"] + " extra"
        self.w.edit_json(self.w.b0 / "gigaspeech.json", change, repin=False)
        self.assert_item_fails(5, "gigaspeech")

    def test_a_saved_ref_norm_the_current_normalizer_does_not_reproduce(self) -> None:
        def change(d):
            d["records"][1]["ref_norm"] = d["records"][1]["ref"]  # raw, unnormalized
        self.w.edit_json(self.w.m1 / "librispeech_other.json", change, repin=False)
        self.assert_item_fails(5, "librispeech_other")

    # item 6
    def test_b0_not_pretrained_or_wrong_revision(self) -> None:
        for change in (lambda s: s.update(kind="export"),
                       lambda s: s["base_model"].update(revision="0" * 40),
                       lambda s: s.update(sha256="0" * 64)):
            with self.subTest(change=change):
                saved = {p.name: p.read_bytes() for p in self.w.b0.glob("*.json")}
                for path in self.w.b0.glob("*.json"):
                    self.w.edit_json(path, lambda d: change(d["source"]), repin=False)
                self.assert_item_fails(6, "b0-pretrained")
                for name, data in saved.items():
                    (self.w.b0 / name).write_bytes(data)

    def test_m1_test_scored_another_export(self) -> None:
        self.w.edit_json(self.w.m1 / "ami.json", lambda d: d["source"].update(sha256="0" * 64), repin=False)
        self.assert_item_fails(6, "M1-test")

    def test_a_baseline_with_a_missing_or_partial_set(self) -> None:
        (self.w.m1 / "spgispeech.json").unlink()
        self.assert_item_fails(6, "M1-test")

    def test_a_baseline_limit_run(self) -> None:
        self.w.edit_json(self.w.b0 / "ami.json", lambda d: d.update(limit=100), repin=False)
        self.assert_item_fails(6, "b0-pretrained")

    def test_baselines_with_different_decoding(self) -> None:
        self.w.edit_json(self.w.b0 / "ami.json", lambda d: d["decoding"].update(batch_size=32), repin=False)
        self.assert_item_fails(6, "decoding")


# ============================================================================ dev-check gate

class GateVerdictTest(unittest.TestCase):
    def verdict(self, offsets: list[float]) -> dict:
        measured = {name: DEV_WER[name] + off for name, off in zip(DEV, offsets)}
        return binary_test.gate_verdict(measured, DEV_WER, tp.DEV_MEAN)

    def test_identical_values_pass(self) -> None:
        verdict = self.verdict([0, 0, 0, 0, 0])
        self.assertTrue(verdict["pass"])
        self.assertAlmostEqual(verdict["mean"], tp.DEV_MEAN, places=15)

    def test_mean_off_by_more_than_0_02_points_fails(self) -> None:
        verdict = self.verdict([0.00021] * 5)  # every set within 0.10 points
        self.assertFalse(verdict["pass"])
        self.assertFalse(verdict["mean_within"])
        self.assertTrue(all(s["within"] for s in verdict["per_set"].values()))
        self.assertFalse(self.verdict([-0.00021] * 5)["pass"])

    def test_one_set_off_by_more_than_0_10_points_fails_with_the_mean_in_tolerance(self) -> None:
        verdict = self.verdict([0.0011, -0.0011, 0, 0, 0])  # mean unchanged
        self.assertTrue(verdict["mean_within"])
        self.assertFalse(verdict["pass"])
        self.assertFalse(verdict["per_set"][DEV[0]]["within"])

    def test_exactly_at_the_tolerances_passes(self) -> None:
        verdict = self.verdict([0.0010, -0.0010, 0.0010, -0.0010, 0.0010])  # mean off by exactly 0.0002
        self.assertTrue(verdict["pass"], verdict)
        self.assertTrue(self.verdict([0.0002] * 5)["pass"])
        self.assertTrue(self.verdict([-0.0002] * 5)["pass"])

    def test_differences_are_reported_in_points_and_fractions(self) -> None:
        verdict = self.verdict([0.0005, 0, 0, 0, 0])
        entry = verdict["per_set"][DEV[0]]
        self.assertAlmostEqual(entry["diff"], 0.0005, places=12)
        self.assertAlmostEqual(entry["diff_points"], 0.05, places=10)
        self.assertEqual(verdict["tolerances"], {"mean_points": 0.02, "set_points": 0.10,
                                                 "mean_fraction": 0.0002, "set_fraction": 0.0010})


class GateBindingTest(WorldCase):
    def assert_test_refused(self, word: str) -> None:
        loads = len(self.w.loads)  # a dev check before may have loaded
        self.refused(["--test"], word)
        self.assertEqual(len(self.w.loads), loads)
        self.assertEqual(self.w.ledger(), [])

    def edit_gate(self, fn) -> None:
        self.w.edit_json(self.w.eval / "MB1-dev-check" / "gate.json", fn, repin=False)

    def test_a_passing_gate_lets_the_test_start(self) -> None:
        self.assertEqual(self.dev_check(), 0)
        gate = read_json(self.w.eval / "MB1-dev-check" / "gate.json")
        self.assertTrue(gate["pass"])
        self.assertIsNone(gate["limit"])
        self.assertEqual(gate["batch_size"], 64)
        self.assertEqual(run_main(["--test"])[0], 0)

    def test_no_gate(self) -> None:
        self.assert_test_refused("gate")

    def test_a_failing_gate(self) -> None:
        self.w.dev_wer["ami_dev"] += 0.0011
        self.assertEqual(self.dev_check(), 1)
        self.assertFalse(read_json(self.w.eval / "MB1-dev-check" / "gate.json")["pass"])
        self.assert_test_refused("gate")

    def test_a_limit_gate(self) -> None:
        self.assertEqual(self.dev_check("--limit", "2"), 0)
        smoke = self.w.eval / "smoke-MB1-dev-limit2"
        self.assertTrue((smoke / "gate.json").exists())
        self.assertFalse((self.w.eval / "MB1-dev-check").exists())
        (self.w.eval / "MB1-dev-check").mkdir()
        shutil.copy(smoke / "gate.json", self.w.eval / "MB1-dev-check" / "gate.json")
        self.assert_test_refused("limit")
        self.edit_gate(lambda g: g.update({"pass": True}))
        self.assert_test_refused("limit")

    def test_a_batch_32_gate(self) -> None:
        self.assertEqual(self.dev_check("--dev-batch-size", "32"), 0)
        b32 = self.w.eval / "MB1-dev-check-b32"
        (self.w.eval / "MB1-dev-check").mkdir()
        shutil.copy(b32 / "gate.json", self.w.eval / "MB1-dev-check" / "gate.json")
        self.assert_test_refused("batch")
        self.edit_gate(lambda g: g.update(out_dir=str(self.w.eval / "MB1-dev-check")))
        self.assert_test_refused("batch")

    def test_a_gate_recording_another_identity(self) -> None:
        self.assertEqual(self.dev_check(), 0)
        self.edit_gate(lambda g: g["identity"].update(weights_sha256="0" * 64))
        self.assert_test_refused("identity")

    def test_a_gate_recording_other_scorer_hashes(self) -> None:
        self.assertEqual(self.dev_check(), 0)
        self.edit_gate(lambda g: g["scorer"]["hashes"].update({"parakeet-binary/binary_test.py": "0" * 64}))
        self.assert_test_refused("scorer")

    def test_a_gate_recording_another_normalizer_module_digest(self) -> None:
        self.assertEqual(self.dev_check(), 0)
        self.edit_gate(lambda g: g["scorer"]["hashes"].update({str(tp.NORMALIZER_MODULE): "0" * 64}))
        self.assert_test_refused("scorer")

    def test_a_gate_recording_other_settings(self) -> None:
        self.assertEqual(self.dev_check(), 0)
        self.edit_gate(lambda g: g["settings"].update(cudnn_benchmark=not g["settings"]["cudnn_benchmark"]))
        self.assert_test_refused("settings")
        self.edit_gate(lambda g: g["settings"].update(cudnn_benchmark=not g["settings"]["cudnn_benchmark"]))
        self.edit_gate(lambda g: g["settings"]["versions"].update(torch="0.0"))
        self.assert_test_refused("settings")

    def test_a_gate_recording_other_manifests(self) -> None:
        self.assertEqual(self.dev_check(), 0)
        self.edit_gate(lambda g: g["manifests"].update(ami="0" * 64))
        self.assert_test_refused("manifest")

    def test_a_changed_scorer_after_the_gate(self) -> None:
        self.assertEqual(self.dev_check(), 0)
        with mock.patch.dict(tp.SCORER_HASHES, {str(tp.NORMALIZER_MODULE): "0" * 64}):
            self.assert_test_refused("precheck")

    def test_the_production_dev_check_is_not_overwritten(self) -> None:
        self.assertEqual(self.dev_check(), 0)
        before = sha(self.w.eval / "MB1-dev-check" / "gate.json")
        self.refused(["--dev-check"], "gate.json")
        self.assertEqual(sha(self.w.eval / "MB1-dev-check" / "gate.json"), before)

    def test_dev_check_refuses_without_a_passing_precheck(self) -> None:
        self.w.units = ["parakeet-train.service"]
        self.refused(["--dev-check"], "precheck")
        self.assertEqual(self.w.loads, [])
        self.assertFalse((self.w.eval / "MB1-dev-check").exists())


# ============================================================================ scoring path

class ScoringPathTest(WorldCase):
    def assert_production_calls(self, names: list[str]) -> None:
        self.assertEqual([c["name"] for c in self.w.calls], names)
        for call in self.w.calls:
            self.assertEqual(call["batch_size"], 64)
            self.assertEqual(call["max_batch_seconds"], 1200.0)
            self.assertIsNone(call["limit"])
            self.assertIs(call["model"], self.w.model)
            self.assertTrue(call["lock_held"])
            self.assertEqual(call["mode"], "no_while_loops")
            self.assertEqual(Path(call["manifest"]), evaluate.resolve_manifest(call["name"])[1])
            self.assertEqual(call["source"]["sha256"], self.w.pins["export/export.safetensors"])
            self.assertEqual(call["source"]["kind"], "export")
            self.assertEqual(call["source"]["description"], "binary export, rebuilt with dequantized FP32 weights")
        self.assertEqual(len(self.w.loads), 1)
        self.assertEqual(self.w.loads[0]["path"].resolve(), self.w.export.resolve())
        self.assertTrue(self.w.loads[0]["lock_held"])

    def recording_writes(self) -> list[str]:
        events: list[str] = []
        real_write, real_append = binary_test.write_json, binary_test.append_attempt

        def write(path, data):
            events.append(Path(path).name)
            real_write(path, data)

        def append(entry):
            events.append(f"ledger:{entry['state']}")
            real_append(entry)

        self.enterContext(mock.patch.object(binary_test, "write_json", side_effect=write))
        self.enterContext(mock.patch.object(binary_test, "append_attempt", side_effect=append))
        return events

    def test_production_dev_check(self) -> None:
        events = self.recording_writes()
        self.assertEqual(self.dev_check(), 0)
        self.assert_production_calls(DEV)
        self.assertEqual(events, [f"{n}.json" for n in DEV] + ["summary.json", "gate.json"])
        out = self.w.eval / "MB1-dev-check"
        gate = read_json(out / "gate.json")
        self.assertEqual(gate["decoder_graphs_mode"], "no_while_loops")
        self.assertEqual(gate["scorer"]["hashes"], CURRENT_SCORER)
        for name in DEV:
            doc = read_json(out / f"{name}.json")
            self.assertEqual(doc["decoder_graphs_mode"], {"before": "no_while_loops", "after": "no_while_loops"})
            self.assertEqual(doc["scorer"], gate["scorer"])
            self.assertEqual(doc["settings"], gate["settings"])
        self.assertEqual(gate["per_set_sha256"], {n: sha(out / f"{n}.json") for n in DEV})
        summary = read_json(out / "summary.json")
        self.assertIn("dev_mean_wer", summary)
        self.assertEqual(summary["source"]["pinned_identity"], gate["identity"])
        self.assertEqual(summary["source"]["export_manifest"]["format"], tp.FORMAT)

    def test_test_run(self) -> None:
        self.assertEqual(self.dev_check(), 0)
        self.w.calls.clear()
        self.w.loads.clear()
        events = self.recording_writes()
        self.assertEqual(run_main(["--test"])[0], 0)
        self.assert_production_calls(SETS)
        expected = ["ledger:started"]
        for name in SETS:
            expected += [f"{name}.json", "ledger:set_done"]
        self.assertEqual(events, expected + ["summary.json", "ledger:complete"])
        ledger = self.w.ledger()
        self.assertEqual([e["state"] for e in ledger], ["started"] + ["set_done"] * 8 + ["complete"])
        self.assertTrue(all(e["attempt"] == 1 for e in ledger))
        out = self.w.eval / "MB1-test"
        self.assertEqual({e["set"]: e["sha256"] for e in ledger if e["state"] == "set_done"},
                         {n: sha(out / f"{n}.json") for n in SETS})
        summary = read_json(out / "summary.json")
        self.assertEqual(set(summary["sets"]), set(SETS))
        self.assertIn("mean_wer", summary)

    def test_diagnostic_batch_32_dev_check(self) -> None:
        self.assertEqual(self.dev_check("--dev-batch-size", "32"), 0)
        self.assertEqual([c["batch_size"] for c in self.w.calls], [32] * 5)
        self.assertEqual({c["max_batch_seconds"] for c in self.w.calls}, {1200.0})
        self.assertEqual(sorted(p.name for p in self.w.eval.iterdir()), ["MB1-dev-check-b32"])
        gate = read_json(self.w.eval / "MB1-dev-check-b32" / "gate.json")
        self.assertEqual(gate["batch_size"], 32)
        self.assertFalse(gate["test_launch_gate"])
        self.refused(["--test"], "gate")

    def test_other_dev_batch_sizes_and_test_batch_options_are_rejected(self) -> None:
        for argv in (["--dev-check", "--dev-batch-size", "16"], ["--test", "--dev-batch-size", "32"],
                     ["--test", "--batch-size", "32"], ["--precheck", "--limit", "3"]):
            with self.subTest(argv), self.assertRaises(SystemExit):
                run_main(argv)
        self.assertEqual(self.w.calls, [])

    def test_smoke_dev_check_with_limit(self) -> None:
        self.assertEqual(self.dev_check("--limit", "2"), 0)
        self.assertEqual({c["limit"] for c in self.w.calls}, {2})
        self.assertEqual(sorted(p.name for p in self.w.eval.iterdir()), ["smoke-MB1-dev-limit2"])
        gate = read_json(self.w.eval / "smoke-MB1-dev-limit2" / "gate.json")
        self.assertEqual(gate["limit"], 2)
        self.assertFalse(gate["pass"])

    def test_a_decoder_mode_changed_during_a_set_is_refused(self) -> None:
        def flip(a):
            if a["name"] == DEV[1]:
                a["model"].computer.force_cuda_graphs_mode("full_graph")
        self.w.on_evaluate = flip
        with self.assertRaises(binary_train.DecoderModeError):
            run_main(["--dev-check"])
        self.assertEqual(len(self.w.calls), 2)
        self.assertFalse((self.w.eval / "MB1-dev-check" / "gate.json").exists())
        self.assertFalse((self.w.eval / "MB1-dev-check" / "summary.json").exists())

    def test_a_decoder_mode_wrong_before_a_set_is_refused(self) -> None:
        with mock.patch.object(binary_train, "pin_decoder_graphs_mode", return_value="no_while_loops"):
            with self.assertRaises(binary_train.DecoderModeError):
                run_main(["--dev-check"])
        self.assertEqual(self.w.calls, [])
        self.assertFalse((self.w.eval / "MB1-dev-check" / "gate.json").exists())

    def test_a_decoder_mode_change_during_the_test_is_refused_and_not_completed(self) -> None:
        self.assertEqual(self.dev_check(), 0)
        self.w.calls.clear()

        def flip(a):
            if a["name"] == "earnings22":
                a["model"].computer.force_cuda_graphs_mode("no_graphs")
        self.w.on_evaluate = flip
        with self.assertRaises(binary_train.DecoderModeError):
            run_main(["--test"])
        states = [e["state"] for e in self.w.ledger()]
        self.assertEqual(states, ["started", "set_done", "set_done", "set_done"])
        self.assertFalse((self.w.eval / "MB1-test" / "summary.json").exists())
        self.refused(["--test"], "--record-failure")

    def test_a_source_other_than_the_pinned_weights_is_refused(self) -> None:
        self.assertEqual(self.dev_check(), 0)
        self.w.calls.clear()
        with mock.patch.object(evaluate, "describe_source",
                               return_value={"kind": "export", "sha256": "0" * 64, "base_model": {}}):
            with self.assertRaises(SystemExit):
                run_main(["--test"])
        self.assertEqual(self.w.calls, [])


# ============================================================================ hygiene

class HygieneTest(WorldCase):
    def test_test_with_limit_is_rejected_before_loading(self) -> None:
        self.assertEqual(self.dev_check(), 0)
        self.w.loads.clear()
        with self.assertRaises(SystemExit):
            run_main(["--test", "--limit", "5"])
        self.assertEqual(self.w.loads, [])
        self.assertEqual(self.w.ledger(), [])
        self.assertFalse((self.w.eval / "MB1-test").exists())

    def test_test_refuses_after_a_complete_record(self) -> None:
        self.complete()
        self.w.loads.clear()
        self.refused(["--test"], "complete")
        self.assertEqual(self.w.loads, [])
        self.assertEqual(sum(e["state"] == "started" for e in self.w.ledger()), 1)

    def test_test_refuses_after_a_started_attempt_without_a_failure_record(self) -> None:
        self.assertEqual(self.dev_check(), 0)
        self.w.loads.clear()
        self.w.write_ledger([{"attempt": 1, "state": "started"}])
        self.refused(["--test"], "--record-failure")
        self.assertEqual(self.w.loads, [])

    def test_ledger_refusals_come_before_the_precheck_and_any_load(self) -> None:
        self.w.write_ledger([{"attempt": 1, "state": "started"}])
        self.w.units = ["parakeet-train.service"]  # the precheck would fail too
        self.refused(["--test"], "--record-failure")
        self.assertEqual(self.w.loads, [])

    def test_record_failure_renames_the_partial_directory_and_appends(self) -> None:
        self.assertEqual(self.dev_check(), 0)

        def crash(a):
            if a["name"] == "gigaspeech":
                raise RuntimeError("CUDA error: an illegal memory access")
        self.w.on_evaluate = crash
        with self.assertRaises(RuntimeError):
            run_main(["--test"])
        partial = self.w.eval / "MB1-test"
        self.assertTrue((partial / "ami.json").exists())
        self.assertEqual(run_main(["--record-failure", "illegal memory access in gigaspeech, log mb1-test.log"])[0], 0)
        self.assertFalse(partial.exists())
        renamed = [p for p in self.w.eval.iterdir() if p.name.startswith("MB1-test.failed-")]
        self.assertEqual(len(renamed), 1)
        self.assertTrue((renamed[0] / "ami.json").exists())
        last = self.w.ledger()[-1]
        self.assertEqual((last["attempt"], last["state"]), (1, "failed"))
        self.assertIn("illegal memory access", last["evidence"])
        self.assertEqual(Path(last["renamed_to"]), renamed[0])
        # A new attempt may now start; it decodes everything again.
        self.w.on_evaluate = None
        self.w.calls.clear()
        self.assertEqual(run_main(["--test"])[0], 0)
        self.assertEqual(len(self.w.calls), 8)
        self.assertEqual([e["attempt"] for e in self.w.ledger() if e["state"] == "started"], [1, 2])

    def test_record_failure_needs_an_open_attempt_and_evidence(self) -> None:
        self.refused(["--record-failure", "nothing ran"], "attempt")
        self.w.write_ledger([{"attempt": 1, "state": "started"}])
        self.refused(["--record-failure", "  "], "evidence")

    def crash_while_writing_the_summary(self) -> None:
        self.assertEqual(self.dev_check(), 0)
        with mock.patch.object(binary_test, "write_test_summary", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                run_main(["--test"])
        states = [e["state"] for e in self.w.ledger()]
        self.assertEqual(states, ["started"] + ["set_done"] * 8)

    def test_summarize_rebuilds_the_summary_once_without_decoding_then_the_report_succeeds(self) -> None:
        self.crash_while_writing_the_summary()
        self.w.calls.clear()
        self.w.loads.clear()
        self.assertEqual(run_main(["--summarize"])[0], 0)
        self.assertEqual(self.w.calls, [])
        self.assertEqual(self.w.loads, [])
        summary = read_json(self.w.eval / "MB1-test" / "summary.json")
        self.assertEqual(set(summary["sets"]), set(SETS))
        self.assertEqual([e["state"] for e in self.w.ledger()].count("complete"), 1)
        before = self.w.ledger()
        self.assertEqual(run_main(["--summarize"])[0], 0)
        self.assertEqual(self.w.ledger(), before)  # idempotent
        self.assertEqual(self.w.calls, [])
        self.assertEqual(run_report()[0], 0)

    def test_summarize_refuses_a_hash_mismatch(self) -> None:
        self.crash_while_writing_the_summary()
        self.w.edit_json(self.w.eval / "MB1-test" / "ami.json", lambda d: d["wer"].update(wer=0.01), repin=False)
        self.refused(["--summarize"], "ami")
        self.assertNotIn("complete", [e["state"] for e in self.w.ledger()])
        self.assertFalse((self.w.eval / "MB1-test" / "summary.json").exists())

    def test_summarize_needs_eight_done_sets(self) -> None:
        self.w.write_ledger([{"attempt": 1, "state": "started"},
                             {"attempt": 1, "set": "ami", "state": "set_done", "sha256": "0" * 64}])
        self.refused(["--summarize"], "set_done")

    def test_with_eight_done_sets_only_summarize_is_allowed(self) -> None:
        self.crash_while_writing_the_summary()
        self.w.loads.clear()
        self.refused(["--record-failure", "summary write failed"], "--summarize")
        self.refused(["--test"], "--summarize")
        self.assertEqual(self.w.loads, [])
        self.assertTrue((self.w.eval / "MB1-test").exists())
        self.assertNotIn("failed", [e["state"] for e in self.w.ledger()])


# ============================================================================ attempt transitions (code review round 1)

class AttemptLockTest(WorldCase):
    """Code review round 1, finding 1: launch and recovery transitions are serialized under the
    attempt lock (eval/MB1-test.attempts.lock), taken before the shared GPU lock and never after it;
    the ledger is re-read after the lock is acquired; the GPU lock is held through finalization and
    the model's release."""

    def setUp(self) -> None:
        super().setUp()
        self.assertEqual(self.dev_check(), 0)
        self.w.calls.clear()
        self.w.loads.clear()
        self.w.lock_events.clear()

    def test_a_queued_launch_rechecks_the_record_after_the_lock(self) -> None:
        """A second --test started while the first decodes waits for the attempt lock, then finds the
        first attempt complete and refuses without loading (its earlier view of the record is stale)."""
        second: dict = {}

        def launch_second() -> None:
            try:
                binary_test.main(["--test"])
                second["result"] = "ran"
            except SystemExit as error:
                second["result"] = str(error)

        thread = threading.Thread(target=launch_second)

        def during(a):
            if a["name"] == "ami" and not thread.is_alive() and "result" not in second:
                thread.start()
                time.sleep(0.5)
                self.assertTrue(thread.is_alive(), "the queued launch did not wait for the attempt lock")
        self.w.on_evaluate = during
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(binary_test.main(["--test"]), 0)
        thread.join(30)
        self.assertFalse(thread.is_alive())
        self.assertIn("complete", second["result"])
        self.assertEqual(len(self.w.loads), 1)
        self.assertEqual([e["state"] for e in self.w.ledger()], ["started"] + ["set_done"] * 8 + ["complete"])

    def test_the_record_is_reread_after_acquiring_the_lock(self) -> None:
        """State changed by the previous holder while this launch waited is seen: no stale "started"."""
        real = binary_test.attempt_lock

        @contextlib.contextmanager
        def lock_then_other_attempt_finished(*args, **kwargs):
            with real(*args, **kwargs):
                entries = [{"attempt": 1, "state": "started"}]
                entries += [{"attempt": 1, "set": n, "state": "set_done", "sha256": "0" * 64} for n in SETS]
                entries += [{"attempt": 1, "state": "complete"}]
                self.w.write_ledger(entries)
                yield

        with mock.patch.object(binary_test, "attempt_lock", side_effect=lock_then_other_attempt_finished):
            self.refused(["--test"], "complete")
        self.assertEqual(self.w.loads, [])
        self.assertEqual(sum(e["state"] == "started" for e in self.w.ledger()), 1)

    def test_record_failure_racing_the_eighth_set_done_is_refused(self) -> None:
        raced: dict = {}

        def during(a):
            if a["name"] == SETS[-1]:
                with self.assertRaises(SystemExit) as caught:
                    binary_test.main(["--record-failure", "looks hung"])
                raced["refusal"] = str(caught.exception)
        self.w.on_evaluate = during
        self.assertEqual(run_main(["--test"])[0], 0)
        self.assertIn("attempt lock", raced["refusal"])
        states = [e["state"] for e in self.w.ledger()]
        self.assertEqual(states, ["started"] + ["set_done"] * 8 + ["complete"])
        self.assertTrue((self.w.eval / "MB1-test").exists())
        self.refused(["--record-failure", "too late"], "attempt")  # nothing open any more
        self.assertEqual(run_report()[0], 0)

    def test_summarize_while_an_attempt_runs_is_refused(self) -> None:
        raced: dict = {}

        def during(a):
            if a["name"] == SETS[-1]:
                with self.assertRaises(SystemExit) as caught:
                    binary_test.main(["--summarize"])
                raced["refusal"] = str(caught.exception)
        self.w.on_evaluate = during
        self.assertEqual(run_main(["--test"])[0], 0)
        self.assertIn("attempt lock", raced["refusal"])
        self.assertEqual([e["state"] for e in self.w.ledger()].count("complete"), 1)

    def test_the_gpu_lock_is_held_through_finalization(self) -> None:
        seen: list[tuple[str, bool]] = []
        real_write, real_append = binary_test.write_json, binary_test.append_attempt

        def write(path, data):
            seen.append((Path(path).name, self.w.lock_held))
            real_write(path, data)

        def append(entry):
            seen.append((f"ledger:{entry['state']}", self.w.lock_held))
            real_append(entry)

        with mock.patch.object(binary_test, "write_json", side_effect=write), \
                mock.patch.object(binary_test, "append_attempt", side_effect=append):
            self.assertEqual(run_main(["--test"])[0], 0)
        self.assertEqual(seen[-2:], [("summary.json", True), ("ledger:complete", True)])
        self.assertTrue(all(held for _, held in seen))
        self.assertEqual(self.w.lock_events, ["gpu+", "gpu-"])

    def test_lock_order_attempt_lock_before_the_gpu_lock(self) -> None:
        order: list[str] = []
        real = binary_test.attempt_lock

        @contextlib.contextmanager
        def logged(*args, **kwargs):
            with real(*args, **kwargs):
                order.append("attempt+")
                yield
                order.append("attempt-")

        def gpu(label):
            @contextlib.contextmanager
            def cm():
                order.append("gpu+")
                with self.w.gpu_lock(label):
                    yield
                order.append("gpu-")
            return cm()

        with mock.patch.object(binary_test, "attempt_lock", side_effect=logged), \
                mock.patch.object(paths, "gpu_lock", side_effect=gpu):
            self.assertEqual(run_main(["--test"])[0], 0)
            self.assertEqual(order, ["attempt+", "gpu+", "gpu-", "attempt-"])
            order.clear()
            self.assertEqual(run_main(["--summarize"])[0], 0)  # already complete: nothing to do
            self.assertEqual(order, ["attempt+", "attempt-"])  # recovery never takes the GPU lock

    def test_recovery_commands_take_the_attempt_lock(self) -> None:
        """While another process holds the attempt lock, --record-failure and --summarize refuse at once."""
        import fcntl
        self.w.write_ledger([{"attempt": 1, "state": "started"}])
        with open(self.w.eval / "MB1-test.attempts.lock", "a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            self.refused(["--record-failure", "crashed"], "attempt lock")
            self.refused(["--summarize"], "attempt lock")
            fcntl.flock(handle, fcntl.LOCK_UN)
        self.assertEqual([e["state"] for e in self.w.ledger()], ["started"])
        self.assertEqual(run_main(["--record-failure", "crashed"])[0], 0)


# ============================================================================ code review round 2

class ModelLifetimeTest(WorldCase):
    """Code review round 2, finding 1: every model (and decoder) reference is gone, and GPU memory has
    been collected and flushed, when the shared GPU lock is released, on the normal and the exception
    path. The model is watched through a weak reference only; the stubs used here record nothing."""

    def setUp(self) -> None:
        super().setUp()
        self.refs: list[weakref.ref] = []
        self.alive_at_release: list[list[bool]] = []
        self.flushed: list[str] = []
        self.fail_at: str | None = None
        world = self.w

        def load_export(out_dir, device="cpu"):
            model = FakeModel()
            self.refs.append(weakref.ref(model))
            return model

        def evaluate_set(model, name, manifest, source, batch_size, limit, device, max_batch_seconds=1200.0):
            if name == self.fail_at:
                raise RuntimeError("CUDA error: an illegal memory access was encountered")
            return world.result({"name": name, "manifest": manifest, "source": source, "batch_size": batch_size,
                                 "limit": limit, "max_batch_seconds": max_batch_seconds})

        @contextlib.contextmanager
        def gpu_lock(label):
            world.lock_held = True
            try:
                yield
            finally:
                self.alive_at_release.append([r() is not None for r in self.refs])
                world.lock_held = False

        real_collect = binary_test.gc.collect

        def collect(*args):
            self.flushed.append("gc.collect")
            return real_collect(*args)

        for target, attr, new in ((binary_export, "load_export", load_export), (evaluate, "evaluate_set", evaluate_set),
                                  (paths, "gpu_lock", gpu_lock), (binary_test.gc, "collect", collect)):
            self.enterContext(mock.patch.object(target, attr, new))

    def check(self, argv: list[str], fail_at: str | None) -> None:
        self.fail_at = fail_at
        if fail_at is None:
            run_main(argv)
        else:
            with self.assertRaises(RuntimeError):
                run_main(argv)
        self.assertEqual(len(self.refs), 1)
        self.assertEqual(self.alive_at_release, [[False]], "the model outlived the GPU lock")
        self.assertIn("gc.collect", self.flushed)

    def test_dev_check_normal_path(self) -> None:
        self.check(["--dev-check"], None)
        self.assertTrue((self.w.eval / "MB1-dev-check" / "gate.json").exists())

    def test_dev_check_exception_path(self) -> None:
        self.check(["--dev-check"], "ami_dev")
        self.assertFalse((self.w.eval / "MB1-dev-check" / "gate.json").exists())

    def test_test_normal_path(self) -> None:
        self.check(["--dev-check"], None)
        self.refs.clear()
        self.alive_at_release.clear()
        self.check(["--test"], None)
        self.assertEqual([e["state"] for e in self.w.ledger()][-1], "complete")

    def test_test_exception_path(self) -> None:
        self.check(["--dev-check"], None)
        self.refs.clear()
        self.alive_at_release.clear()
        self.check(["--test"], "gigaspeech")
        self.assertNotIn("complete", [e["state"] for e in self.w.ledger()])

    def test_a_decoder_mode_refusal_also_releases_the_model(self) -> None:
        def flip_then_score(model, name, *args, **kwargs):
            model.computer.force_cuda_graphs_mode("full_graph")
            return self.w.result({"name": name, "manifest": args[0], "source": args[1], "batch_size": args[2],
                                  "limit": args[3], "max_batch_seconds": kwargs.get("max_batch_seconds", 1200.0)})
        with mock.patch.object(evaluate, "evaluate_set", flip_then_score):
            with self.assertRaises(binary_train.DecoderModeError):
                run_main(["--dev-check"])
        self.assertEqual(self.alive_at_release, [[False]])


class GateRaceTest(WorldCase):
    """Code review round 2, finding 2: the launch gate is read once under the shared GPU lock, those
    bytes are validated and their SHA-256 recorded; the dev-check output directory is checked and
    cleaned under the GPU lock."""

    def gate(self) -> Path:
        return self.w.eval / "MB1-dev-check" / "gate.json"

    @contextlib.contextmanager
    def during_gpu_wait(self, action):
        """paths.gpu_lock that runs `action` once while this process "waits" for the lock."""
        world, done = self.w, []

        @contextlib.contextmanager
        def gpu_lock(label):
            if not done:
                done.append(True)
                action()
            with world.gpu_lock(label):
                yield
        with mock.patch.object(paths, "gpu_lock", gpu_lock):
            yield

    def test_a_gate_replaced_by_a_failing_gate_during_the_wait_is_refused_without_loading(self) -> None:
        self.assertEqual(self.dev_check(), 0)
        self.w.loads.clear()
        with self.during_gpu_wait(lambda: self.w.edit_json(self.gate(), lambda g: g.update({"pass": False}),
                                                           repin=False)):
            self.refused(["--test"], "did not pass")
        self.assertEqual(self.w.loads, [])
        self.assertEqual(self.w.ledger(), [])
        self.assertFalse((self.w.eval / "MB1-test").exists())

    def test_a_gate_removed_during_the_wait_is_refused(self) -> None:
        self.assertEqual(self.dev_check(), 0)
        self.w.loads.clear()
        with self.during_gpu_wait(lambda: self.gate().unlink()):
            self.refused(["--test"], "gate")
        self.assertEqual(self.w.loads, [])
        self.assertEqual(self.w.ledger(), [])

    def test_the_recorded_digest_is_of_the_bytes_validated_under_the_lock(self) -> None:
        self.assertEqual(self.dev_check(), 0)
        replacement = {}

        def replace_with_a_passing_variant():
            self.w.edit_json(self.gate(), lambda g: g.update(utc="2026-10-20T23:59:59Z"), repin=False)
            replacement["sha"] = sha(self.gate())
        with self.during_gpu_wait(replace_with_a_passing_variant):
            self.assertEqual(run_main(["--test"])[0], 0)
        started = self.w.ledger()[0]
        self.assertEqual(started["gate_sha256"], replacement["sha"])
        self.assertEqual(started["gate_utc"], "2026-10-20T23:59:59Z")

    def test_a_gate_changed_after_it_was_read_does_not_change_the_record(self) -> None:
        self.assertEqual(self.dev_check(), 0)
        launched = sha(self.gate())

        def replace(a):
            if a["name"] == "ami":
                self.w.edit_json(self.gate(), lambda g: g.update({"pass": False}), repin=False)
        self.w.on_evaluate = replace
        self.assertEqual(run_main(["--test"])[0], 0)
        self.assertEqual(self.w.ledger()[0]["gate_sha256"], launched)
        self.report_refused("gate")  # the report sees the replaced gate

    def test_an_output_directory_appearing_during_the_wait_is_refused(self) -> None:
        self.assertEqual(self.dev_check(), 0)
        self.w.loads.clear()
        with self.during_gpu_wait(lambda: (self.w.eval / "MB1-test").mkdir()):
            self.refused(["--test"], "exists")
        self.assertEqual(self.w.loads, [])
        self.assertEqual(self.w.ledger(), [])

    def test_a_second_dev_check_does_not_overwrite_a_gate_completed_during_its_wait(self) -> None:
        """Two concurrent dev checks: the second passed its early checks, then waited for the GPU lock
        while the first completed; under the lock it finds gate.json and refuses, before loading and
        before touching the directory."""
        first: dict = {}

        def first_dev_check_completes():
            with mock.patch.object(paths, "gpu_lock", self.w.gpu_lock):
                first["code"] = run_main(["--dev-check"])[0]
            first["files"] = {p.name: sha(p) for p in (self.w.eval / "MB1-dev-check").iterdir()}
            first["loads"] = len(self.w.loads)
        with self.during_gpu_wait(first_dev_check_completes):
            self.refused(["--dev-check"], "gate.json")
        self.assertEqual(first["code"], 0)
        self.assertEqual(len(self.w.loads), first["loads"])  # the second never loaded
        self.assertEqual({p.name: sha(p) for p in (self.w.eval / "MB1-dev-check").iterdir()}, first["files"])

    def test_a_smoke_rerun_cleans_its_directory_only_under_the_lock(self) -> None:
        smoke = self.w.eval / "smoke-MB1-dev-limit2"
        smoke.mkdir(parents=True)
        (smoke / "stale.json").write_text("{}")
        seen = {}
        with self.during_gpu_wait(lambda: seen.update(stale=(smoke / "stale.json").exists())):
            self.assertEqual(self.dev_check("--limit", "2"), 0)
        self.assertTrue(seen["stale"])  # still there while waiting for the lock
        self.assertFalse((smoke / "stale.json").exists())


# ============================================================================ report

class ReportTest(WorldCase):
    def setUp(self) -> None:
        super().setUp()
        self.complete()

    def test_known_values(self) -> None:
        code, out = run_report()
        self.assertEqual(code, 0, out)
        result = read_json(self.w.results / "test.json")
        mean_sets = paths.MEAN_SETS
        self.assertEqual(result["mean_sets"], mean_sets)
        self.assertNotIn("common_voice", mean_sets)
        for arm, wer in (("B0", B0_WER), ("M1", M1_WER), ("MB1", MB1_WER)):
            row = result["rows"][arm]
            for name in SETS:
                self.assertAlmostEqual(row[name], 100 * wer[name], places=9)
            self.assertAlmostEqual(row["mean"], 100 * sum(wer[n] for n in mean_sets) / 7, places=9)
        # MB1 mean (7 sets) = (2.0 + 4.4 + 11.0 + 12.5 + 10.2 + 3.1 + 6.6) / 7 = 49.8 / 7
        self.assertAlmostEqual(result["rows"]["MB1"]["mean"], 49.8 / 7, places=9)
        self.assertAlmostEqual(result["rows"]["MB1"]["common_voice"], 9.0, places=9)
        diff = result["differences"]["MB1 - M1"]
        self.assertAlmostEqual(diff["librispeech_other"]["points"], 0.4, places=9)
        self.assertAlmostEqual(diff["librispeech_other"]["relative_percent"], 10.0, places=9)
        self.assertAlmostEqual(diff["common_voice"]["points"], 1.0, places=9)
        self.assertAlmostEqual(diff["mean"]["points"], (49.8 - 47.8) / 7, places=9)
        diff_b0 = result["differences"]["MB1 - B0"]
        self.assertAlmostEqual(diff_b0["spgispeech"]["points"], 0.9, places=9)
        self.assertAlmostEqual(diff_b0["spgispeech"]["relative_percent"], 100 * 0.9 / 2.2, places=9)
        self.assertAlmostEqual(result["empty_hypothesis_fraction"]["B0"]["ami"], 1 / 3, places=12)
        self.assertEqual(result["empty_hypothesis_fraction"]["MB1"]["ami"], 0.0)
        self.assertEqual(result["export_mb"], {"MB1": 105.3, "M1": 180.8, "original": 2472.2})
        self.assertTrue(result["dev_check"]["pass"])
        self.assertEqual(result["sanity_flags"], [])
        self.assertEqual(result["rows"]["NVIDIA published"]["librispeech_clean"], 1.69)
        text = (self.w.results / "TEST.md").read_text()
        self.assertIn("net change", text)
        self.assertIn("not a causal", text)
        self.assertIn("Common Voice", text)
        self.assertIn("105.3", text)

    def test_mb1_set_with_another_source_hash(self) -> None:
        self.w.tamper_mb1("ami", lambda d: d["source"].update(sha256="0" * 64))
        self.report_refused("source SHA-256")

    def test_m1_summary_or_set_with_another_source_hash(self) -> None:
        self.w.edit_json(self.w.m1 / "summary.json", lambda s: s["source"].update(sha256="0" * 64), repin=False)
        self.report_refused("M1")

    def test_m1_per_set_source_disagreeing_with_its_summary(self) -> None:
        self.w.edit_json(self.w.m1 / "gigaspeech.json", lambda d: d["source"].update(sha256="1" * 64), repin=False)
        self.report_refused("M1")

    def test_m1_outputs_substituted_by_another_arms(self) -> None:
        shutil.rmtree(self.w.m1)
        shutil.copytree(self.w.b0, self.w.m1)
        self.report_refused("M1")

    def test_mb1_file_replaced_after_its_set_done_record(self) -> None:
        self.w.tamper_mb1("voxpopuli", lambda d: d["wer"].update(wer=0.01), rehash=False)
        self.report_refused("attempt")

    def test_files_from_two_attempts(self) -> None:
        entries = self.w.ledger()
        path = self.w.eval / "MB1-test" / "earnings22.json"
        self.w.edit_json(path, lambda d: d.update(wall_s=2.0), repin=False)  # "attempt 0's" file
        earlier = [{"attempt": 0, "state": "started"},
                   {"attempt": 0, "set": "earnings22", "state": "set_done", "sha256": sha(path)},
                   {"attempt": 0, "state": "failed", "evidence": "fixture"}]
        self.w.write_ledger(earlier + entries)
        self.report_refused("attempt")

    def test_mb1_summary_inconsistent_with_its_per_set_files(self) -> None:
        self.w.edit_json(self.w.eval / "MB1-test" / "summary.json",
                         lambda s: s["sets"]["ami"].update(wer=1.0), repin=False)
        self.report_refused("summary")

    def test_manifest_differing_between_arms_or_from_the_current_file(self) -> None:
        self.w.edit_json(self.w.b0 / "spgispeech.json", lambda d: d.update(manifest_sha256="0" * 64), repin=False)
        self.report_refused("manifest")

    def test_manifest_changed_since_scoring(self) -> None:
        manifest = self.w.manifests / "test_spgispeech.jsonl"
        manifest.write_text(manifest.read_text() + "\n")
        self.report_refused("manifest")

    def test_decoding_or_normalizer_differing_from_m1(self) -> None:
        self.w.tamper_mb1("ami", lambda d: d["decoding"]["config"]["greedy"].update(max_symbols=5))
        self.report_refused("decoding")

    def test_normalizer_differing_from_m1(self) -> None:
        self.w.tamper_mb1("librispeech_clean", lambda d: d.update(normalizer="another normalizer"))
        self.report_refused("normalizer")

    def test_b0_not_the_pinned_pretrained_model(self) -> None:
        self.w.edit_json(self.w.b0 / "ami.json", lambda d: d["source"].update(kind="checkpoint"), repin=False)
        self.report_refused("B0")

    def test_partial_output(self) -> None:
        self.w.tamper_mb1("ami", lambda d: d.update(limit=10))
        self.report_refused("limit")

    def test_missing_set(self) -> None:
        (self.w.b0 / "common_voice.json").unlink()
        self.report_refused("missing")

    def test_set_missing_from_mb1_summary(self) -> None:
        self.w.edit_json(self.w.eval / "MB1-test" / "summary.json", lambda s: s["sets"].pop("voxpopuli"), repin=False)
        self.report_refused("voxpopuli")

    def test_duplicated_utterance_id(self) -> None:
        self.w.tamper_mb1("gigaspeech", lambda d: d["records"].__setitem__(1, dict(d["records"][0])))
        self.report_refused("utterance")

    def test_missing_utterance_id(self) -> None:
        self.w.edit_json(self.w.m1 / "earnings22.json", lambda d: d["records"].pop(), repin=False)
        self.report_refused("utterance")

    def test_recorded_scorer_hashes_differing_from_the_pinned_ones(self) -> None:
        self.w.tamper_mb1("ami", lambda d: d["scorer"]["hashes"].update({"parakeet-ternary/evaluate.py": "0" * 64}))
        self.report_refused("scorer")

    def test_dev_check_scorer_differing_from_the_tests(self) -> None:
        self.w.edit_json(self.w.eval / "MB1-dev-check" / "gate.json",
                         lambda g: g["scorer"].update(binary_test_paths_sha256="0" * 64), repin=False)
        self.report_refused("scorer")

    # code review round 1, finding 2: the dev gate bound to the completed attempt
    def started_gate(self, digest: str) -> None:
        entries = self.w.ledger()
        for e in entries:
            if e["state"] == "started":
                e["gate_sha256"] = digest
        self.w.write_ledger(entries)

    def test_a_replacement_gate_with_unchanged_scorer_hashes(self) -> None:
        """A later passing batch-64 dev check (other settings, same scorer) is not the launch gate."""
        gate = self.w.eval / "MB1-dev-check" / "gate.json"
        self.w.edit_json(gate, lambda g: g["settings"].update(cudnn_benchmark=not g["settings"]["cudnn_benchmark"]),
                         repin=False)
        self.report_refused("gate")

    def test_a_started_record_without_the_gate_hash(self) -> None:
        self.started_gate(None)
        self.report_refused("gate")

    def test_a_bound_gate_recording_other_bindings_than_the_outputs(self) -> None:
        gate = self.w.eval / "MB1-dev-check" / "gate.json"
        original = gate.read_bytes()
        for word, change in (("identity", lambda g: g["identity"].update(weights_sha256="0" * 64)),
                             ("settings", lambda g: g["settings"]["versions"].update(torch="0.0")),
                             ("manifest", lambda g: g["manifests"].update(ami="0" * 64)),
                             ("decoder", lambda g: g.update(decoder_graphs_mode="full_graph"))):
            with self.subTest(word):
                gate.write_bytes(original)
                self.w.edit_json(gate, change, repin=False)
                self.started_gate(sha(gate))  # bound by digest, so only the binding differs
                self.report_refused(word)

    # code review round 1, finding 3: the full summary projection
    def test_mb1_summary_projection_fields(self) -> None:
        summary = self.w.eval / "MB1-test" / "summary.json"
        original = summary.read_bytes()
        for key, value in (("scored_utterances", 99), ("excluded_empty_reference", 7), ("audio_hours", 12.5),
                           ("utterances", 99), ("padded_short", 3), ("wall_s", 0.5), ("limit", None)):
            with self.subTest(key):
                summary.write_bytes(original)
                if key == "limit":  # an entry missing a field is inconsistent too
                    self.w.edit_json(summary, lambda s: s["sets"]["gigaspeech"].pop("wall_s"), repin=False)
                else:
                    self.w.edit_json(summary, lambda s, key=key, value=value: s["sets"]["gigaspeech"].update({key: value}),
                                     repin=False)
                self.report_refused("summary")

    def test_mb1_summary_recorded_settings_and_decoder_mode(self) -> None:
        summary = self.w.eval / "MB1-test" / "summary.json"
        original = summary.read_bytes()
        for change in (lambda s: s["settings"].update(cudnn_deterministic=not s["settings"]["cudnn_deterministic"]),
                       lambda s: s["settings"]["env"].update(LD_LIBRARY_PATH="/elsewhere"),
                       lambda s: s.update(decoder_graphs_mode=["full_graph"]),
                       lambda s: s.update(binary_test={**s["binary_test"], "batch_size": 32}),
                       lambda s: s.update(padded_short=5), lambda s: s.update(wall_s=0.0)):
            with self.subTest(change=change):
                summary.write_bytes(original)
                self.w.edit_json(summary, change, repin=False)
                self.report_refused("summary")

    def test_mb1_per_set_decoder_mode_or_settings_other_than_the_runs(self) -> None:
        self.w.tamper_mb1("earnings22", lambda d: d["decoder_graphs_mode"].update(before="full_graph"))
        self.report_refused("decoder")

    def test_mb1_per_set_settings_differing_between_sets(self) -> None:
        self.w.tamper_mb1("ami", lambda d: d["settings"].update(cudnn_benchmark=not d["settings"]["cudnn_benchmark"]))
        self.report_refused("settings")

    # code review round 3: the report reads every file it both parses and hashes exactly once
    @contextlib.contextmanager
    def after_read(self, path: Path, action):
        """binary_report_test.read_bytes that runs `action` right after `path` has been read (once)."""
        real, done = binary_report_test.read_bytes, []
        reads: list[Path] = []

        def read(p):
            data = real(p)
            reads.append(Path(p))
            if Path(p) == path and not done:
                done.append(True)
                action()
            return data
        with mock.patch.object(binary_report_test, "read_bytes", side_effect=read):
            yield reads

    def test_a_gate_changed_after_it_is_read_is_not_reported(self) -> None:
        gate = self.w.eval / "MB1-dev-check" / "gate.json"
        launched = read_json(gate)

        def replace():  # a later passing dev check, same bindings, other results
            self.w.edit_json(gate, lambda g: g.update(mean=0.5, mean_diff_points=99.0), repin=False)
        with self.after_read(gate, replace) as reads:
            self.assertEqual(run_report()[0], 0)
        self.assertEqual(reads.count(gate), 1)
        dev = read_json(self.w.results / "test.json")["dev_check"]
        self.assertAlmostEqual(dev["mean_percent"], 100 * launched["mean"], places=12)
        self.assertEqual(dev["mean_diff_points"], launched["mean_diff_points"])

    def test_a_replacement_gate_restored_after_it_is_read_is_refused(self) -> None:
        """The report reads a later gate; the launch gate is put back before any hashing would happen: the
        digest checked is that of the bytes read, so the later gate is refused, not reported."""
        gate = self.w.eval / "MB1-dev-check" / "gate.json"
        original = gate.read_bytes()
        self.w.edit_json(gate, lambda g: g.update(mean=0.5), repin=False)
        with self.after_read(gate, lambda: gate.write_bytes(original)):
            self.report_refused("gate")

    def test_an_mb1_file_restored_after_it_is_read_is_refused(self) -> None:
        path = self.w.eval / "MB1-test" / "ami.json"
        original = path.read_bytes()
        self.w.tamper_mb1("ami", lambda d: d["wer"].update(wer=0.0123), rehash=False)
        with self.after_read(path, lambda: path.write_bytes(original)) as reads:
            self.report_refused("attempt")
        self.assertEqual(reads.count(path), 1)

    def test_a_ledger_changed_after_it_is_read_does_not_change_the_checks(self) -> None:
        ledger = self.w.eval / "MB1-test.attempts.jsonl"
        with self.after_read(ledger, lambda: self.w.write_ledger([])) as reads:
            self.assertEqual(run_report()[0], 0)
        self.assertEqual(reads.count(ledger), 1)

    def test_a_manifest_restored_after_it_is_read_is_refused(self) -> None:
        manifest = self.w.manifests / "test_voxpopuli.jsonl"
        original = manifest.read_bytes()
        manifest.write_bytes(original + b"\n")  # same utterances, another file than the arms scored
        with self.after_read(manifest, lambda: manifest.write_bytes(original)) as reads:
            self.report_refused("manifest")
        self.assertEqual(reads.count(manifest), 1)

    def test_each_parsed_and_hashed_file_is_read_once(self) -> None:
        with self.after_read(Path("/nonexistent"), lambda: None) as reads:
            self.assertEqual(run_report()[0], 0)
        once = [self.w.eval / "MB1-dev-check" / "gate.json", self.w.eval / "MB1-test.attempts.jsonl"]
        once += [self.w.eval / "MB1-test" / f"{n}.json" for n in SETS]
        once += [self.w.manifests / f"test_{n}.jsonl" for n in SETS]
        for path in once:
            with self.subTest(path.name):
                self.assertEqual(reads.count(path), 1)

    def test_attempt_record_not_complete(self) -> None:
        self.w.write_ledger([e for e in self.w.ledger() if e["state"] != "complete"])
        self.report_refused("complete")

    def test_sanity_flags_pause_the_write_up(self) -> None:
        shutil.rmtree(self.w.eval)
        self.w.mb1_wer["ami"] = 3.5 * B0_WER["ami"]
        self.w.empty_sets = {"voxpopuli"}
        self.complete()
        code, out = run_report()
        self.assertEqual(code, 3)
        flags = read_json(self.w.results / "test.json")["sanity_flags"]
        self.assertEqual(len(flags), 2, flags)
        self.assertTrue(any("ami" in f and "3x" in f for f in flags))
        self.assertTrue(any("voxpopuli" in f and "empty" in f for f in flags))
        self.assertIn("SANITY CHECK FAILED", (self.w.results / "TEST.md").read_text())


if __name__ == "__main__":
    unittest.main()
