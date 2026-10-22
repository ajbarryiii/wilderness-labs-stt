"""Contract tests of gpu1bit/shared/fulldev.py (plans/activation-precision.md "Implementation", R2-R5).

Run with the runner's CPU mode, so THIS process never touches the GPU; the GPU checks spawn
`timeout 600` children that each hold one lock:
    gpu1bit/shared/python -m gpu1bit.tests.run --cpu gpu1bit.tests.test_fulldev

CPU:
  - dev split asserted; test manifests, other manifests and non-dev paths refused;
  - scorer parity: fed the recorded evaluate.py hypotheses of M1's and MB1's full-dev evaluations,
    the scorer gives bitwise the pinned per-set WERs and dev means and the same per-utterance S/D/I;
  - completeness: every subset utterance exactly once (missing, duplicated, foreign ids refused);
  - resume: a working file with a torn last line resumes to exactly the one-shot result; a
    double-counted utterance is refused;
  - R4 identity: freezing, refusal after a source change (a new file in a hashed directory), refusal
    of rows of another identity;
  - R5 failure injection through the real per-utterance function (`process_one`) with a stub model:
    a NaN encoder output, a decoder error and a wrong encoder length each make the arm invalid;
  - R2 cross-check arithmetic: mean 0.02 and per-set 0.10 limits, both required;
  - audio prefetch (one utterance ahead): same order, same audio, same rows, errors in place;
  - held part-count arms refuse to build; arm names parse.
GPU (children; only with GPU1BIT_SPAWN_GPU_CHILDREN=1, inside an agent-jobs GPU lease):
  - lock rule + R3: a worker takes its one lock before CUDA, is visible in the lock file under its
    own pid, holds < 600 s; a worker killed mid-chunk (SIGKILL) and then resumed leaves every
    utterance exactly once, in manifest order; graph-cache overflow is exercised (4 entries, the rest
    eager) and a work budget stops a worker early.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import types
import unittest
from pathlib import Path

import torch

from gpu1bit.shared import env, fulldev

RECORDED = {"M1": Path("/mnt/hd/wilderness-labs-stt/parakeet-ternary/runs/main-M1-P2-lr5e-4"),
            "MB1": Path("/mnt/hd/wilderness-labs-stt/parakeet-binary/runs/main-binB-m1-lr2e-4")}
PY = str(env.SHARED / "python")


def _recorded(model: str, name: str) -> dict:
    return json.loads((RECORDED[model] / "eval-dev" / f"{name}.json").read_text())


def _fake_rows(model: str, name: str, identity: str | None = None) -> list[dict]:
    rec = _recorded(model, name)
    return [fulldev.make_row(i, r["id"], r["hyp"], tokens=[], padded_short=False, reference=r["ref"],
                             duration=r["duration"], enc_len=10, identity_hash=identity) for i, r in enumerate(rec["records"])]


class DevOnly(unittest.TestCase):
    def test_dev_sets_are_paths_dev_sets(self):
        import paths
        self.assertEqual(list(fulldev.DEV_SETS), list(paths.DEV_SETS))

    def test_dev_manifests_load_and_others_are_refused(self):
        records, meta = fulldev.manifest("voxpopuli_dev")
        self.assertEqual(len(records), 1753)
        self.assertEqual(meta["split"], "validation")
        for bad in ("test_librispeech_clean", "librispeech_clean", "ami", "librispeech_dev_clean_400",
                    "/mnt/hd/wilderness-labs-stt/parakeet-ternary/manifests/test_ami.jsonl"):
            with self.subTest(name=bad), self.assertRaises(ValueError):
                fulldev.manifest(bad)
        with self.assertRaises(ValueError):
            fulldev.parse_sets(["librispeech_dev_clean", "test_ami"])

    def test_sample_subset_is_the_frozen_512(self):
        from gpu1bit.shared import workloads
        n = sum(len(fulldev.subset_records(s, "sample")) for s in fulldev.DEV_SETS)
        self.assertEqual(n, len(workloads.load()["dev_sample"]))


class Scoring(unittest.TestCase):
    def test_scorer_parity_with_recorded_evaluations(self):
        for model, run in RECORDED.items():
            pinned = json.loads((run / "summary.json").read_text())
            per_set = {}
            for name in fulldev.DEV_SETS:
                rec = _recorded(model, name)
                summary = fulldev.summarize_set(name, _fake_rows(model, name))
                with self.subTest(model=model, set=name):
                    self.assertEqual(summary["wer"]["wer"], rec["wer"])
                    self.assertEqual(summary["wer"]["wer"], pinned["dev_wer"][name])
                    self.assertEqual(summary["utterances"], pinned["dev_utterances"][name])
                    for row, r in zip(summary["records"], rec["records"]):
                        self.assertEqual((row["S"], row["D"], row["I"], row["scored"]), (r["S"], r["D"], r["I"], r["scored"]))
                per_set[name] = summary
            with self.subTest(model=model, what="mean"):
                self.assertEqual(fulldev.dev_mean(per_set), pinned["dev_mean_wer"])
                xc = fulldev.cross_check(model, {n: s["wer_percent"] for n, s in per_set.items()},
                                         100 * fulldev.dev_mean(per_set))
                self.assertTrue(xc["pass"])
                self.assertEqual(xc["mean_difference_points"], 0.0)

    def test_cross_check_limits(self):
        pinned = json.loads(fulldev.PINNED["MB1"].read_text())
        sets = {n: 100 * pinned["dev_wer"][n] for n in fulldev.DEV_SETS}
        mean = 100 * pinned["dev_mean_wer"]
        self.assertTrue(fulldev.cross_check("MB1", sets, mean + 0.019)["pass"])
        self.assertFalse(fulldev.cross_check("MB1", sets, mean + 0.021)["pass"])
        shifted = dict(sets, ami_dev=sets["ami_dev"] + 0.11)
        self.assertFalse(fulldev.cross_check("MB1", shifted, mean)["pass"])   # one set beyond 0.10 fails even at equal mean
        shifted = dict(sets, ami_dev=sets["ami_dev"] + 0.09)
        self.assertTrue(fulldev.cross_check("MB1", shifted, mean)["pass"])

    def test_completeness_is_enforced(self):
        rows = _fake_rows("MB1", "voxpopuli_dev")
        fulldev.summarize_set("voxpopuli_dev", rows)
        cases = {"missing": rows[:100] + rows[101:], "duplicated": rows + [rows[7]],
                 "foreign": rows[:-1] + [dict(rows[-1], id="voxpopuli:not-in-manifest")]}
        for label, bad in cases.items():
            with self.subTest(case=label), self.assertRaises(ValueError):
                fulldev.summarize_set("voxpopuli_dev", bad)

    def test_resume_does_not_double_count(self):
        rows = _fake_rows("M1", "voxpopuli_dev")
        work = env.artifacts("tmp", "fulldev-tests") / f"resume-{os.getpid()}.jsonl"
        work.unlink(missing_ok=True)
        half = len(rows) // 2
        with open(work, "w") as f:
            for r in rows[:half]:
                f.write(json.dumps(r) + "\n")
            f.write(json.dumps(rows[half])[:40])
        done = fulldev.load_rows(work)
        self.assertEqual([r["id"] for r in done], [r["id"] for r in rows[:half]])
        self.assertEqual(work.read_text().count("\n"), half)
        with open(work, "a") as f:
            for r in rows[half:]:
                f.write(json.dumps(r) + "\n")
        resumed = fulldev.summarize_set("voxpopuli_dev", fulldev.load_rows(work))
        self.assertEqual(resumed["utterances"], len(rows))
        self.assertEqual(resumed["wer"], fulldev.summarize_set("voxpopuli_dev", rows)["wer"])
        with open(work, "a") as f:
            f.write(json.dumps(rows[3]) + "\n")
        with self.assertRaises(ValueError):
            fulldev.summarize_set("voxpopuli_dev", fulldev.load_rows(work))
        work.unlink()


class Identity(unittest.TestCase):
    def test_identity_binds_the_complete_export(self):
        """Review c1 major 2: the export's configuration (e.g. decoding.greedy.max_symbols) and every
        file of the export directory (tokenizer included) are part of the identity, not only the
        weights digest."""
        import shutil
        import tempfile
        for model, attr in (("M1", "M1_EXPORT"), ("MB1", "MB1_EXPORT")):
            arm = "M1-x3" if model == "M1" else "MB1-int8_mma-x3"
            real = getattr(env, attr)
            with tempfile.TemporaryDirectory() as tmp, self.subTest(model=model):
                copy = Path(tmp) / "export"
                shutil.copytree(real, copy)
                setattr(env, attr, copy)
                try:
                    base = fulldev.identity(arm, "full")
                    self.assertEqual(fulldev.identity(arm, "full")["hash"], base["hash"], "deterministic")
                    self.assertEqual(base["export"]["files"]["manifest.json"], env.sha256_file(copy / "manifest.json"))
                    self.assertIn("tokenizer/tokenizer.model", base["export"]["files"])
                    manifest = json.loads((copy / "manifest.json").read_text())
                    self.assertEqual(manifest["config"]["decoding"]["greedy"]["max_symbols"], 10)
                    manifest["config"]["decoding"]["greedy"]["max_symbols"] = 1
                    (copy / "manifest.json").write_text(json.dumps(manifest))
                    changed = fulldev.identity(arm, "full")
                    self.assertNotEqual(changed["hash"], base["hash"], "max_symbols 10 -> 1 must change the identity")
                    self.assertNotEqual(changed["export"]["config_sha256"], base["export"]["config_sha256"])
                    shutil.copy(real / "manifest.json", copy / "manifest.json")
                    self.assertEqual(fulldev.identity(arm, "full")["hash"], base["hash"])
                    tok = copy / "tokenizer" / "vocab.txt"
                    tok.write_bytes(tok.read_bytes() + b"x\n")
                    self.assertNotEqual(fulldev.identity(arm, "full")["hash"], base["hash"], "a tokenizer change must change it")
                finally:
                    setattr(env, attr, real)

    def test_identity_covers_sources_scorer_manifests_and_candidate_dirs(self):
        m1 = fulldev.identity("M1-x3", "full")
        mb1 = fulldev.identity("MB1-int8_mma-x3", "full")
        for doc in (m1, mb1):
            self.assertIn("finetune/parakeet-binary/gpu1bit/shared/fulldev.py", doc["sources"])
            self.assertIn("finetune/parakeet-ternary/inference/encoder4.py", doc["sources"])
            self.assertEqual(set(doc["manifests"]), set(fulldev.DEV_SETS))
            self.assertEqual(set(doc["scorer"]) >= {"evaluate.py", "wer.py", "normalizer.json", "english_normalizer.py"}, True)
        self.assertTrue(any(k.startswith("finetune/parakeet-binary/gpu1bit/int8_mma/") for k in mb1["sources"]))
        self.assertFalse(any(k.startswith("finetune/parakeet-binary/gpu1bit/int8_mma/") for k in m1["sources"]))
        on_disk = {str(q.relative_to(env.REPO)) for q in (env.GPU1BIT / "int8_mma").rglob("*")
                   if q.is_file() and "__pycache__" not in q.parts}
        self.assertTrue(on_disk <= set(mb1["sources"]), "every file in the candidate directory (b01 included) is hashed")
        self.assertNotEqual(m1["hash"], mb1["hash"])
        self.assertNotEqual(fulldev.identity("M1-x3", "sample")["hash"], m1["hash"])

    def test_refusal_after_a_source_change_and_of_mixed_rows(self):
        run = f"test-identity-{os.getpid()}"
        h = fulldev.freeze(run, "M1-x3", "first:5")
        self.assertEqual(fulldev.check_identity(run, "M1-x3", "first:5")["hash"], h)
        probe = env.SHARED / f"_identity_probe_{os.getpid()}.py"
        try:
            probe.write_text("# temporary file: a new source in a hashed directory\n")
            with self.assertRaisesRegex(SystemExit, "identity mismatch"):
                fulldev.check_identity(run, "M1-x3", "first:5")
            with self.assertRaisesRegex(SystemExit, "changed since it was frozen"):
                fulldev.freeze(run, "M1-x3", "first:5")
        finally:
            probe.unlink()
        self.assertEqual(fulldev.check_identity(run, "M1-x3", "first:5")["hash"], h)
        rows = _fake_rows("M1", "voxpopuli_dev", identity=h)
        fulldev.summarize_set("voxpopuli_dev", rows, "full", h)
        rows[11] = dict(rows[11], identity="0" * 64)
        with self.assertRaisesRegex(ValueError, "other arm identities"):
            fulldev.summarize_set("voxpopuli_dev", rows, "full", h)


class _StubDecoding:
    def __init__(self, fail):
        self.fail = fail

    def rnnt_decoder_predictions_tensor(self, encoder_output, encoded_lengths, return_hypotheses=False):
        if self.fail:
            raise RuntimeError("injected decoder failure")
        return [types.SimpleNamespace(text="hello world", y_sequence=torch.tensor([5, 7]), timestamp=torch.tensor([0, 3]))]


class _StubModel(torch.nn.Module):
    def __init__(self, nan=False, length=5, decode_fails=False):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.nan, self.length = nan, length
        self.decoding = _StubDecoding(decode_fails)

    def forward(self, input_signal, input_signal_length):
        enc = torch.zeros(1, 4, 6)
        if self.nan:
            enc[0, 1, 2] = float("nan")
        return enc, torch.tensor([self.length])


class StructuralGates(unittest.TestCase):
    """R5 failure injection through fulldev.process_one with a stub model (CPU)."""

    def setUp(self):
        self.i, self.rec = fulldev.subset_records("voxpopuli_dev", "first:1")[0]

    def test_clean_row(self):
        row = fulldev.process_one(_StubModel(), self.i, self.rec, "h")
        self.assertTrue(row["finite"])
        self.assertIsNone(row["decode_error"])
        self.assertEqual((row["tokens"], row["timestamps"], row["enc_len"], row["hyp"]), ([5, 7], [0, 3], 5, "hello world"))

    def test_nan_output_invalidates(self):
        row = fulldev.process_one(_StubModel(nan=True), self.i, self.rec, "h")
        self.assertFalse(row["finite"])
        self._assert_invalid(row)

    def test_decoder_error_invalidates(self):
        row = fulldev.process_one(_StubModel(decode_fails=True), self.i, self.rec, "h")
        self.assertIn("injected decoder failure", row["decode_error"])
        self._assert_invalid(row)

    def test_wrong_length_invalidates_against_x3(self):
        good = fulldev.process_one(_StubModel(length=5), self.i, self.rec, "h")
        bad = fulldev.process_one(_StubModel(length=4), self.i, self.rec, "h")
        self.assertEqual(fulldev.length_differences([bad], [good]), [bad["id"]])
        self.assertEqual(fulldev.length_differences([good], [good]), [])

    def _assert_invalid(self, row):
        rows = [dict(r) for r in _fake_rows("MB1", "voxpopuli_dev")]
        rows[0] = dict(row, ref=rows[0]["ref"], id=rows[0]["id"])
        summary = fulldev.summarize_set("voxpopuli_dev", rows)
        self.assertFalse(summary["structurally_valid"])


class Prefetch(unittest.TestCase):
    """One-utterance-ahead audio prefetch (user request 2026-10-20, to speed up full-dev runs):
      - prefetched() yields every (index, record) in the input order, with exactly the audio a
        sequential load_clip gives;
      - it loads at most one record ahead of the consumer, and all records once;
      - a loading error surfaces at its own record, after every earlier record was delivered;
      - process_one with prefetched audio gives the same row as with its own load."""

    def test_order_and_audio_unchanged(self):
        import numpy as np
        pairs = fulldev.subset_records("voxpopuli_dev", "first:12") + fulldev.subset_records("ami_dev", "first:6")
        got = list(fulldev.prefetched(pairs))
        self.assertEqual([(i, r["id"]) for i, r, _ in got], [(i, r["id"]) for i, r in pairs])
        for (i, rec, (clip, padded)) in got:
            want, want_padded = fulldev.load_clip(rec)
            self.assertEqual(padded, want_padded)
            self.assertEqual(clip.dtype, want.dtype)
            self.assertTrue(np.array_equal(clip, want), rec["id"])
        self.assertEqual(list(fulldev.prefetched([])), [])

    def test_one_ahead_and_errors_in_place(self):
        import threading
        events, lock = [], threading.Lock()

        def loader(rec):
            with lock:
                events.append(("load", rec["k"]))
            if rec["k"] == 7:
                raise OSError("injected read failure")
            return rec["k"] * 10, False
        items = [(k, {"k": k}) for k in range(10)]
        consumed = []
        with self.assertRaisesRegex(OSError, "injected"):
            for i, rec, audio in fulldev.prefetched(items, loader):
                with lock:
                    loads = [k for e, k in events if e == "load"]
                self.assertLessEqual(max(loads), i + 1, "at most one record ahead")
                self.assertEqual(audio, (i * 10, False))
                consumed.append(i)
        self.assertEqual(consumed, list(range(7)), "every record before the failing one is delivered")
        events.clear()
        self.assertEqual([i for i, _, _ in fulldev.prefetched(items[:3], lambda r: (r["k"], False))], [0, 1, 2])

    def test_process_one_with_prefetched_audio(self):
        i, rec = fulldev.subset_records("voxpopuli_dev", "first:1")[0]
        own = fulldev.process_one(_StubModel(), i, rec, "h")
        pre = fulldev.process_one(_StubModel(), i, rec, "h", audio=fulldev.load_clip(rec))
        self.assertEqual(own, pre)


class Arms(unittest.TestCase):
    def test_arm_names(self):
        self.assertEqual(fulldev.parse_arm("M1-x3")["parts"], 3)
        self.assertEqual(fulldev.parse_arm("MB1-int8_mma-x3")["kernel"], "int8_mma")
        self.assertEqual(fulldev.parse_arm("MB1-int8_mma:b01-x1")["kernel"], "int8_mma:b01")
        self.assertEqual(fulldev.parse_arm("MB1-fp32")["kind"], "fp32")
        for bad in ("M1-x4", "M2-x3", "MB1-x3", "M1-int8_mma-x3", "MB1-int8_mma-fp32"):
            with self.subTest(arm=bad), self.assertRaises(ValueError):
                fulldev.parse_arm(bad)

    def test_part_count_arms_are_held(self):
        for arm in ("M1-x2", "M1-x1", "MB1-int8_mma-x2", "MB1-int8_mma-x1"):
            with self.subTest(arm=arm), self.assertRaisesRegex(RuntimeError, "held"):
                fulldev.check_arm_enabled(fulldev.parse_arm(arm))
        fulldev.check_arm_enabled(fulldev.parse_arm("M1-x3"))
        fulldev.check_arm_enabled(fulldev.parse_arm("M1-fp32"))


class Workers(unittest.TestCase):
    """GPU children; this process stays off the GPU (gpu1bit.tests.run --cpu)."""

    def setUp(self):
        if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
            self.skipTest("run via `gpu1bit.tests.run --cpu`: the parent must not touch the GPU")
        if os.environ.get("GPU1BIT_SPAWN_GPU_CHILDREN") != "1":
            self.skipTest("spawns GPU children; run inside an agent-jobs GPU lease with GPU1BIT_SPAWN_GPU_CHILDREN=1")

    @staticmethod
    def _env():
        e = dict(os.environ)
        e.pop("CUDA_VISIBLE_DEVICES", None)
        e.pop("CUBLAS_WORKSPACE_CONFIG", None)
        return e

    def test_lock_rule_kill_resume_overflow_budget(self):
        import paths
        run, arm, subset, name = f"test-workers-{os.getpid()}", "M1-x3", "first:120", "voxpopuli_dev"
        fulldev.freeze(run, arm, subset)
        cmd = ["timeout", "600", PY, "-m", "gpu1bit.shared.fulldev", "run", "--run", run, "--arm", arm,
               "--subset", subset, "--sets", name]
        work = fulldev.work_path(run, arm, name, subset)
        # 1. a worker killed mid-chunk
        child = subprocess.Popen(cmd, env=self._env(), cwd=str(env.BINARY_DIR), stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, start_new_session=True)
        seen = None
        deadline = time.time() + 580          # within the worker's own `timeout 600`, lock wait included (busy shared GPU)
        while time.time() < deadline and child.poll() is None:
            text = paths.GPU_LOCK.read_text() if paths.GPU_LOCK.exists() else ""
            if "H:fulldev" in text:
                seen = text.strip()
            if len(fulldev.load_rows(work)) >= 30:
                break
            time.sleep(0.2)
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
        killed_rows = fulldev.load_rows(work)
        self.assertGreaterEqual(len(killed_rows), 30)
        self.assertLess(len(killed_rows), 120)
        self.assertIsNotNone(seen)
        # 2. a worker with a small work budget stops early
        rc = subprocess.run(cmd + ["--work-budget", "3"], env=self._env(), cwd=str(env.BINARY_DIR), capture_output=True, text=True)
        self.assertEqual(rc.returncode, 0, rc.stdout[-2000:] + rc.stderr[-2000:])
        budget = fulldev.chunks(run, arm, subset)[-1]
        self.assertTrue(budget["stopped_by_budget"])
        # 3. resume to completion: fresh workers until done (R3). On a busy shared GPU one worker's work budget
        # may end before the last utterance; every such worker must have stopped by its budget, not failed.
        for _ in range(6):
            rc = subprocess.run(cmd, env=self._env(), cwd=str(env.BINARY_DIR), capture_output=True, text=True)
            self.assertEqual(rc.returncode, 0, rc.stdout[-2000:] + rc.stderr[-2000:])
            if len(fulldev.load_rows(work)) >= 120:
                break
            self.assertTrue(fulldev.chunks(run, arm, subset)[-1]["stopped_by_budget"], "an incomplete resume stopped by budget")
        last = fulldev.chunks(run, arm, subset)[-1]
        rows = fulldev.load_rows(work)
        self.assertEqual([r["i"] for r in rows], list(range(120)))
        self.assertEqual(len({r["id"] for r in rows}), 120)
        self.assertTrue(all(r["identity"] == last["identity"] for r in rows))
        self.assertTrue(all(r["finite"] and r["decode_error"] is None for r in rows))
        done = fulldev.chunks(run, arm, subset)
        self.assertGreaterEqual(len(done), 2)                 # the killed worker wrote no chunk record:
        self.assertTrue(done[0]["stopped_by_budget"])         # the budget worker first, then the resume worker(s)
        self.assertTrue(all(c["stopped_by_budget"] for c in done[1:-1]))
        for c in done:
            self.assertEqual(set(c["audio_s"]), set(c["rows"]))           # timed coverage per set (X7 budget, c3)
            self.assertTrue(all(v > 0 for v in c["audio_s"].values()))
            self.assertTrue(c["lock_before_cuda"])
            self.assertEqual(c["locks_taken"], 1)
            self.assertLess(c["held_s"], 600)
            self.assertEqual(c["label"], f"H:fulldev {arm}@{subset} {run}")
        pid, label = seen.split(" ", 1)                       # the killed worker's lock line: its pid, our label
        self.assertNotEqual(int(pid), os.getpid())
        self.assertEqual(label, f"H:fulldev {arm}@{subset} {run}")
        self.assertEqual(last["graph_entries"], 4)            # cache full: later shapes ran on the eager overflow path
        self.assertGreater(len({r["enc_frames"] for r in rows}), 4)


if __name__ == "__main__":
    unittest.main()
