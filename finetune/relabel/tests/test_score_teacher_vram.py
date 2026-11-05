"""score_teacher's VRAM lease (heldout-panel plan "Fix first"): under a shared GPU lease (AGENT_JOBS_VRAM_FRACTION)
the allocator cap goes to an indexed device, the device's own index or, for an index-less "cuda", the current device.
torch.cuda.set_per_process_memory_fraction refuses an index-less device; the stand-in below validates its device
argument with torch's own check, so the old call fails here as it failed under a lease.

CPU only. Run from finetune/relabel:  ./python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import shutil
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402
from torch.cuda import _utils as cuda_utils  # noqa: E402

import relabel_paths  # noqa: E402
import score_teacher  # noqa: E402
import teachers  # noqa: E402

TMP = relabel_paths.ARTIFACTS / "tmp" / "tests" / "score-teacher-vram"


class FractionRecorder:
    """set_per_process_memory_fraction's argument contract: an integer index or an indexed torch.device."""

    def __init__(self):
        self.calls = []

    def __call__(self, fraction, device=None):
        index = cuda_utils._get_device_index(device)  # raises ValueError for torch.device("cuda")
        self.calls.append((fraction, index))


class VramLeaseTest(unittest.TestCase):
    def lease(self, device, env):
        recorder = FractionRecorder()
        with mock.patch.dict(os.environ, env, clear=False), \
             mock.patch.object(torch.cuda, "set_per_process_memory_fraction", recorder), \
             mock.patch.object(torch.cuda, "current_device", return_value=3):
            if "AGENT_JOBS_VRAM_FRACTION" not in env:
                os.environ.pop("AGENT_JOBS_VRAM_FRACTION", None)
            score_teacher.vram_lease(device)
        return recorder.calls

    def test_index_less_cuda_uses_the_current_device(self) -> None:
        self.assertEqual(self.lease(torch.device("cuda"), {"AGENT_JOBS_VRAM_FRACTION": "0.3125"}), [(0.3125, 3)])

    def test_an_indexed_device_keeps_its_index(self) -> None:
        self.assertEqual(self.lease(torch.device("cuda:1"), {"AGENT_JOBS_VRAM_FRACTION": "0.5"}), [(0.5, 1)])

    def test_no_lease_and_cpu_set_nothing(self) -> None:
        self.assertEqual(self.lease(torch.device("cuda"), {}), [])
        self.assertEqual(self.lease(torch.device("cpu"), {"AGENT_JOBS_VRAM_FRACTION": "0.5"}), [])


class StandIn(teachers.Teacher):
    name = relabel_paths.REAL_TEACHERS[0]
    hf_id = "stand-in"
    revision = "0" * 40
    license = "none"
    library = "test"

    def _load(self) -> None:
        raise AssertionError("not loaded in this test")


class ScorerUnderLeaseTest(unittest.TestCase):
    """score_teacher.main --device cuda under a lease gets past the allocator cap (to the host-memory refusal, which
    this test forces so nothing is loaded)."""

    def setUp(self) -> None:
        relabel_paths.require_mount()
        self.root = TMP / "entry"
        if self.root.exists():
            shutil.rmtree(self.root)
        self.patches = [mock.patch.object(relabel_paths, "SMOKE", self.root / "smoke"),
                        mock.patch.object(relabel_paths, "DIAGNOSTIC", self.root / "diagnostic"),
                        mock.patch.object(relabel_paths, "GPU_LOCK", self.root / "gpu.lock")]
        for p in self.patches:
            p.start()

    def tearDown(self) -> None:
        for p in self.patches:
            p.stop()

    def test_cuda_run_under_a_lease_does_not_crash_on_the_cap(self) -> None:
        recorder = FractionRecorder()
        with mock.patch.dict(os.environ, {"AGENT_JOBS_VRAM_FRACTION": "0.25"}), \
             mock.patch.object(teachers, "get", lambda name: StandIn()), \
             mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda, "current_device", return_value=0), \
             mock.patch.object(torch.cuda, "set_per_process_memory_fraction", recorder), \
             mock.patch.object(score_teacher, "available_memory_gb", return_value=1.0):
            rc = score_teacher.main(["--teacher", relabel_paths.REAL_TEACHERS[0], "--limit", "1",
                                     "--sets", "yodas_dev", "--lock-wait", "1", "--min-available-gb", "16"])
        self.assertEqual(rc, score_teacher.LOW_MEMORY_EXIT)
        self.assertEqual(recorder.calls, [(0.25, 0)])


if __name__ == "__main__":
    unittest.main()
