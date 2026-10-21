"""Fresh-process contracts of the deployment loaders at a selectable part count (plans/x2-default.md X3, Y1).
Run through the runner's CPU mode, so THIS process never touches the GPU; every child is its own
`timeout 600` GPU process holding the lock (MB1: the gpu1bit.runtime CLI, i.e. load_binary_packed with
no arguments but the export; M1: tests/x2_loader_child.py calling runtime_m1.load_ternary_packed):

    job start h-x2-loaders --needs gpu:shared --vram 4G -- env GPU1BIT_SPAWN_GPU_CHILDREN=1 \
        gpu1bit/shared/python -m gpu1bit.tests.run --cpu gpu1bit.tests.test_x2_loaders_fresh

For each model (MB1: gpu1bit.runtime.load_binary_packed; M1: gpu1bit.runtime_m1.load_ternary_packed):
  - pointer (no selection argument, GPU1BIT_SELECTION unset): before promotion the pointer
    shared/selected.json is the x3 document, so the call gets x3 (Y1), with the fulldev identity of
    MB1-B-hybrid-x3 / M1-x3 under the pointer's content;
  - candidate (selection=shared/selected-x2.json): x2 by default, with an identity equal to the gate
    and benchmark arm's (g1 = b1 = MB1-B-hybrid-x2; g4 = b4 = M1-x2 under the candidate) (X3);
  - candidate, parts=3: the named override, under the x3 arm identity of the candidate (g3 = b2; g5 = b5);
  - simulated promotion (GPU1BIT_SELECTION = a byte-identical copy of the candidate at another path, no
    argument): an unqualified call gets x2 with the identity of b1 / b4, because identities hash the
    CONTENT of the selection document (Y1's post-promotion contract); same transcript as the candidate.
  At parts 2 the provider is fulldev.PartCount(·, 2) around the part-2 provider; at 3 it is unwrapped.
  The frontend is set for deployment (dither 0, pad_to 0) and restored by close (M1).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import unittest

from gpu1bit.shared import env

PY = str(env.SHARED / "python")


def _child_env(selection=None) -> dict:
    e = dict(os.environ)
    e.pop("CUDA_VISIBLE_DEVICES", None)
    e.pop("CUBLAS_WORKSPACE_CONFIG", None)
    e.pop("GPU1BIT_SELECTION", None)
    if selection is not None:
        e["GPU1BIT_SELECTION"] = str(selection)
    return e


def _expected(arm: str, selection) -> str:
    from gpu1bit.actprec import parts
    return parts.identity(arm, "full", selection=selection)["hash"]


class _Cases:
    model = ""

    @classmethod
    def setUpClass(cls):
        if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
            raise unittest.SkipTest("run via `gpu1bit.tests.run --cpu`: the parent must not touch the GPU")
        if os.environ.get("GPU1BIT_SPAWN_GPU_CHILDREN") != "1":
            raise unittest.SkipTest("spawns GPU children; run inside an agent-jobs GPU lease with GPU1BIT_SPAWN_GPU_CHILDREN=1")
        from gpu1bit.shared import seldoc
        cls.seldoc = seldoc
        cls.tmp = env.artifacts("tmp", "x2-loaders", f"{cls.model}-{os.getpid()}")
        shutil.rmtree(cls.tmp)
        cls.tmp.mkdir(parents=True)
        cls.promoted = cls.tmp / "selected.json"
        shutil.copyfile(seldoc.CANDIDATE, cls.promoted)
        cls.results = {}

    def run_case(self, case: str) -> dict:
        raise NotImplementedError

    def check(self, case, parts, arm, selection_path, sha_path):
        got = self.run_case(case)
        self.results[case] = got
        self.assertEqual(got["parts"], parts)
        self.assertEqual(got["selection_sha256"], env.sha256_file(sha_path))
        self.assertEqual(got["identity"], _expected(arm, selection_path), f"{case}: not the {arm} arm identity")
        self.assertTrue(got["text"].strip())
        return got

    def test_1_pointer_is_x3_before_promotion(self):
        s = self.seldoc
        if env.sha256_file(s.SELECTED) == env.sha256_file(s.CANDIDATE):
            self.skipTest("promoted: the pointer is the candidate (covered by the simulated promotion)")
        self.assertEqual(json.loads(s.SELECTED.read_text()).get("default_parts", 3), 3)
        self.check("pointer", 3, f"{self.model}-{self.x3_arm}", s.SELECTED, s.SELECTED)

    def test_2_candidate_default_is_x2(self):
        s = self.seldoc
        self.check("candidate", 2, f"{self.model}-{self.x2_arm}", s.CANDIDATE, s.CANDIDATE)

    def test_3_candidate_parts_3_override(self):
        s = self.seldoc
        self.check("candidate-x3", 3, f"{self.model}-{self.x3_arm}", s.CANDIDATE, s.CANDIDATE)

    def test_4_simulated_promotion_equals_the_qualified_arm(self):
        s = self.seldoc
        got = self.check("promoted", 2, f"{self.model}-{self.x2_arm}", s.CANDIDATE, self.promoted)
        self.assertEqual(got["identity"], _expected(f"{self.model}-{self.x2_arm}", s.CANDIDATE))
        cand = self.results.get("candidate") or self.run_case("candidate")
        for k in ("text", "tokens", "timestamps"):
            if k in cand:
                self.assertEqual(got[k], cand[k], k)


class MB1(_Cases, unittest.TestCase):
    model, x2_arm, x3_arm = "MB1", "B-hybrid-x2", "B-hybrid-x3"

    def run_case(self, case):
        from gpu1bit.shared import workloads
        s = self.seldoc
        clip = workloads.load()["dev_clips"][0]
        args = {"pointer": [], "candidate": ["--selection", str(s.CANDIDATE)],
                "candidate-x3": ["--selection", str(s.CANDIDATE), "--parts", "3"], "promoted": []}[case]
        proc = subprocess.run(["timeout", "600", PY, "-m", "gpu1bit.runtime", "--export", str(env.MB1_EXPORT), "--tokens",
                               "--label", f"H:x2 MB1 loader {case}", *args, clip["audio_filepath"]],
                              env=_child_env(self.promoted if case == "promoted" else None), cwd=str(env.BINARY_DIR),
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
        return json.loads(proc.stdout.strip().splitlines()[-1])


class M1(_Cases, unittest.TestCase):
    model, x2_arm, x3_arm = "M1", "x2", "x3"

    def run_case(self, case):
        s = self.seldoc
        out = self.tmp / f"{case}.json"
        args = {"pointer": [], "candidate": ["--selection", str(s.CANDIDATE)],
                "candidate-x3": ["--selection", str(s.CANDIDATE), "--parts", "3"], "promoted": []}[case]
        proc = subprocess.run(["timeout", "600", PY, "-m", "gpu1bit.tests.x2_loader_child", "--out", str(out), *args],
                              env=_child_env(self.promoted if case == "promoted" else None), cwd=str(env.BINARY_DIR),
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
        got = json.loads(out.read_text())
        self.assertEqual(got["export"], str(env.M1_EXPORT.resolve()))
        if got["parts"] == 2:
            self.assertEqual((got["provider"], got["partcount_n"]), ("PartCount", 2))
        else:
            self.assertIsNone(got["partcount_n"])
        self.assertEqual(got["inner"], "ExpandedProjection")
        self.assertEqual(got["frontend_during"], {"dither": 0.0, "pad_to": 0})
        self.assertEqual(got["frontend_after"], {"dither": 1e-5, "pad_to": 0})
        self.assertTrue(got["record_removed"])
        return got


if __name__ == "__main__":
    unittest.main()
