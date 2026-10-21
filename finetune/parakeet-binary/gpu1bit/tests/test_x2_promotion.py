"""Promotion end to end (Codex x2 code review c1, findings 7 and 8; plans/x2-default.md Y1, Z2).

A scratch copy of the source trees (finetune/parakeet-binary, parakeet-ternary and whisper-ternary, without
caches; the interpreter, finetune/stt and custom/inference-efficiency linked) is made under
H/scratch/promotion-<cpu|gpu>, and its pointer shared/selected.json is
PROMOTED there with its own seldoc.promote (a byte-identical copy of selected-x2.json). Then the existing x3
and x2 suites run in the scratch copy, unchanged:
  - Promoted.test_cpu_suites: every CPU module of the regular suite (CPU_SUITE), one runner process;
  - PromotedGPU (GPU1BIT_PROMOTION_GPU=1, inside an agent-jobs shared GPU lease): each in-process GPU module
    (test_x2_runtime per class) in its own locked runner process (GPU_INPROCESS), so each hold stays under the
    runner's 570 s budget on a shared, busy GPU, the deferred-A3 contracts through dispatch (x3 module and
    the x2 module), and the GPU-children modules with GPU1BIT_SPAWN_GPU_CHILDREN=1 (GPU_CHILDREN).
test_runtime_fresh's fresh-process children need a GPU without foreign processes (the EnergyMeter guard,
P-memory); they are listed in EXCLUSIVE and run only with GPU1BIT_PROMOTION_EXCLUSIVE=1 under an exclusive
lease. Its CPU part (the x3 selection record, finding 8) is in CPU_SUITE.
The real tree's pointer is never touched; each run's summary is written to H/scratch/promotion-<cpu|gpu>.json.

    gpu1bit/shared/python -m gpu1bit.tests.run --cpu gpu1bit.tests.test_x2_promotion.Promoted
    job start h-x2-promoted-gpu --needs gpu:shared --vram 8G -- env GPU1BIT_PROMOTION_GPU=1 \\
        gpu1bit/shared/python -m gpu1bit.tests.run --cpu gpu1bit.tests.test_x2_promotion.PromotedGPU
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path

from gpu1bit.shared import env

TREES = ("finetune/parakeet-binary", "finetune/parakeet-ternary", "finetune/whisper-ternary")
# read-only dependencies outside the copied trees, linked: the interpreter, the normalizer model files the
# scorer hashes (evaluate.NORMALIZER_JSON) and M1's energy meter (inference.benchmark.energy_meter)
LINKS = ("custom/autoresearch/.training-runtime", "finetune/stt", "custom/inference-efficiency")
CPU_SUITE = ["gpu1bit.tests.test_api", "gpu1bit.tests.test_derived", "gpu1bit.tests.test_eligibility",
             "gpu1bit.tests.test_selection_rules", "gpu1bit.tests.test_actprec_report", "gpu1bit.tests.test_actprec_bench",
             "gpu1bit.tests.test_rerun_timing", "gpu1bit.tests.test_partcount", "gpu1bit.tests.test_fulldev",
             "gpu1bit.tests.test_x2_arms", "gpu1bit.tests.test_x2_bench", "gpu1bit.tests.test_x2_bench_session",
             "gpu1bit.tests.test_x2_gates", "gpu1bit.tests.test_x2_preflight_runtime", "gpu1bit.tests.test_x2_recorded_arm",
             "gpu1bit.tests.test_x2_selection", "gpu1bit.tests.test_x2_testscore", "gpu1bit.tests.test_x2_loaders_fresh",
             "gpu1bit.tests.test_runtime_fresh"]
GPU_INPROCESS = ["gpu1bit.tests.test_x2_runtime.DeployedX1", "gpu1bit.tests.test_x2_runtime.EncoderLengthM1",
                 "gpu1bit.tests.test_x2_runtime.KernelOpX2", "gpu1bit.tests.test_partcount",
                 "gpu1bit.tests.test_finish_residual_routing", "gpu1bit.tests.test_runtime"]   # one hold each (<= 570 s)
GPU_CHILDREN = ["gpu1bit.tests.test_x2_loaders_fresh", "gpu1bit.tests.test_fulldev.Workers"]
EXCLUSIVE = ["gpu1bit.tests.test_runtime_fresh"]
# The outer timeout of a locked runner covers its wait for the shared gpu lock as well as its hold; the hold
# itself is limited by the runner (570 s budget, A9).
WAIT_AND_HOLD_S = 3000


def scratch_root(kind: str) -> Path:
    return env.ARTIFACTS / "scratch" / f"promotion-{kind}"


def make_scratch(kind: str) -> Path:
    """A fresh copy of the source trees (one per kind, so the CPU and GPU runs never share one) with the
    pointer promoted by the copy's own seldoc.promote."""
    env.require_mount()
    root = scratch_root(kind)
    shutil.rmtree(root, ignore_errors=True)
    for tree in TREES:
        shutil.copytree(env.REPO / tree, root / tree, symlinks=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
    for rel in LINKS:
        link = root / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(env.REPO / rel)
    binary = root / "finetune" / "parakeet-binary"
    py = str(binary / "gpu1bit" / "shared" / "python")
    rc = subprocess.run([py, "-c", "from gpu1bit.shared import seldoc; seldoc.promote(); print(seldoc.current_path())"],
                        cwd=str(binary), env=_env(), capture_output=True, text=True)
    if rc.returncode != 0:
        raise AssertionError(rc.stderr[-2000:])
    shared = binary / "gpu1bit" / "shared"
    if (shared / "selected.json").read_bytes() != (env.SHARED / "selected-x2.json").read_bytes():
        raise AssertionError("the scratch pointer is not the candidate's bytes after promote()")
    return binary


def _env(**extra) -> dict:
    e = {k: v for k, v in os.environ.items() if k not in ("GPU1BIT_SELECTION", "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG",
                                                          "GPU1BIT_SPAWN_GPU_CHILDREN", "PYTHONPATH")}
    e.update(extra)
    return e


def _run(binary: Path, args: list[str], label: str, timeout: int, **extra) -> dict:
    py = str(binary / "gpu1bit" / "shared" / "python")
    proc = subprocess.run(["timeout", str(timeout), py, *args], cwd=str(binary), env=_env(**extra),
                          capture_output=True, text=True)
    summary = None
    for line in (proc.stdout + "\n" + proc.stderr).splitlines()[::-1]:   # the runner's JSON summary line
        at = line.find('{"label"')
        if at >= 0:
            try:
                summary = json.loads(line[at:])
                break
            except ValueError:
                continue
    log = binary.parents[1] / f"run-{label.replace(' ', '-')}.log"
    log.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    return {"label": label, "args": args, "rc": proc.returncode, "summary": summary, "log": str(log),
            "tail": (proc.stdout + proc.stderr)[-4000:] if proc.returncode or summary is None else ""}


def _record(name: str, runs: list[dict]) -> None:
    (env.ARTIFACTS / "scratch").mkdir(parents=True, exist_ok=True)
    (env.ARTIFACTS / "scratch" / f"promotion-{name}.json").write_text(json.dumps(runs, indent=1) + "\n")


class Promoted(unittest.TestCase):
    def test_cpu_suites(self):
        binary = make_scratch("cpu")
        r = _run(binary, ["-m", "gpu1bit.tests.run", "--cpu", "--label", "H:promoted-cpu", *CPU_SUITE], "cpu", 3000)
        _record("cpu", [r])
        self.assertEqual(r["rc"], 0, r["tail"])
        self.assertTrue(r["summary"] and r["summary"]["ok"], r["summary"])


class PromotedGPU(unittest.TestCase):
    def setUp(self):
        if os.environ.get("GPU1BIT_PROMOTION_GPU") != "1":
            self.skipTest("GPU: run inside an agent-jobs GPU lease with GPU1BIT_PROMOTION_GPU=1")

    def test_gpu_suites(self):
        binary = make_scratch("gpu")
        runs = [_run(binary, ["-m", "gpu1bit.tests.run", "--label", f"H:promoted-{m.split('.', 2)[-1]}", m],
                     f"gpu {m.split('.', 2)[-1]}", WAIT_AND_HOLD_S) for m in GPU_INPROCESS]   # one locked process per entry
        runs += [_run(binary, ["-m", "gpu1bit.shared.dispatch", "--patches", "gpu1bit.int8_mma.b01_run,gpu1bit.int8_mma.session_patch",
                              "tests", "--label", "H:promoted-deferred", "gpu1bit.tests.test_deferred_a3"], "deferred x3",
                       WAIT_AND_HOLD_S),
                _run(binary, ["-m", "gpu1bit.tests.run", "--label", "H:promoted-deferred-x2",
                              "gpu1bit.tests.test_deferred_a3.DeferredProviders"], "deferred x2", WAIT_AND_HOLD_S,
                     GPU1BIT_DEFERRED_MODULE="gpu1bit.int8_mma.deferred_x2"),
                 _run(binary, ["-m", "gpu1bit.tests.run", "--cpu", "--label", "H:promoted-children", *GPU_CHILDREN],
                      "gpu children", 3000, GPU1BIT_SPAWN_GPU_CHILDREN="1")]
        if os.environ.get("GPU1BIT_PROMOTION_EXCLUSIVE") == "1":
            runs.append(_run(binary, ["-m", "gpu1bit.tests.run", "--cpu", "--label", "H:promoted-exclusive", *EXCLUSIVE],
                             "exclusive children", 3000, GPU1BIT_SPAWN_GPU_CHILDREN="1"))
        _record("gpu", runs)
        for r in runs:
            with self.subTest(run=r["label"]):
                self.assertEqual(r["rc"], 0, r["tail"])
                self.assertTrue(r["summary"] and r["summary"]["ok"], r["summary"])


if __name__ == "__main__":
    unittest.main()
