"""Fresh-process contract tests of gpu1bit/runtime.py. Run through the runner's CPU mode, so THIS
process never touches the GPU; every child is its own `timeout 600` GPU process holding the lock:

    GPU1BIT_SPAWN_GPU_CHILDREN=1 gpu1bit/shared/python -m gpu1bit.tests.run --cpu gpu1bit.tests.test_runtime_fresh

These are the historical x3 runtime's contracts: they resolve the preserved x3 selection record
(seldoc.x3_selection_record: the x3 document's own, or the candidate's selection.x3_selection) and load at
parts 3 explicitly, before and after promotion (Codex x2 c1, finding 8). The x2 default's memory contract
against the new benchmark's b1 is tests/test_x2_runtime_fresh.py.
1. Fresh-process VRAM (P-memory method: NVML process memory every 10 ms from before loading), per
   runtime: per dev clip, the peak including first capture is at most 1.02 x the selection
   session's P-memory peak of the same variant (hybrid = B-hybrid, provider = int8_mma; the max of its
   forward- and reverse-order runs) and at least 500 MB below R-expanded's; transcripts equal the
   selection session's. (The selection bound applies to the selected kernel only.)
2. No .nemo and no 2-bit export is opened: the CLI (`python -m gpu1bit.runtime --export MB1 ...`)
   under `strace -f -e trace=%file` opens MB1's export files (positive control) and never a path
   ending in .nemo, M1's model directory, M1's export or H's derived 2-bit export.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import unittest
from pathlib import Path

from gpu1bit.shared import env

PY = str(env.SHARED / "python")


def _selection() -> tuple[Path, dict]:
    """(report.json of the session that selected the deployed configuration, {runtime: variant name})
    from shared/selected.json: "selection.session", "selection.variant" (the hybrid winner's roster
    name) and "kernel" (the provider-stack variant is the kernel's registry name)."""
    from gpu1bit import runtime
    from gpu1bit.shared import seldoc
    sel = runtime.selected()
    rec = seldoc.x3_selection_record(sel)          # x3 document or the candidate's preserved record (c1, finding 8)
    return (env.ARTIFACTS / "selection" / rec["session"] / "report.json",
            {"hybrid": rec.get("variant", "B-hybrid"), "provider": seldoc.entry(sel, 3)["kernel"]})


def _child_env() -> dict:
    """The deployment environment the selection's P-memory used: the runner's CUDA hiding and its
    CUBLAS_WORKSPACE_CONFIG=:4096:8 (deterministic cuBLAS for the in-process contract tests) are
    removed. Measured: that setting alone adds 25.2 MB after load and 37.7 / 83.9 / 117.4 MB at the
    3 / 10 / 29.4 s dev clips (cuBLAS workspaces per stream, graph-capture streams included)."""
    e = dict(os.environ)
    e.pop("CUDA_VISIBLE_DEVICES", None)
    e.pop("CUBLAS_WORKSPACE_CONFIG", None)
    return e


def _selected_kernel() -> str:
    from gpu1bit import runtime
    from gpu1bit.shared import seldoc
    return os.environ.get("GPU1BIT_RUNTIME_KERNEL") or seldoc.entry(runtime.selected(), 3)["kernel"]


class SelectionRecord(unittest.TestCase):
    """CPU (finding 8): the historical x3 contracts resolve the preserved x3 selection record and the part-3
    kernel from the document in use, before and after promotion."""

    def test_x3_record_and_kernel(self):
        report, variant_of = _selection()
        self.assertEqual(report, env.ARTIFACTS / "selection" / "sel-20261020b" / "report.json")
        self.assertTrue(report.exists())
        self.assertEqual(variant_of, {"hybrid": "B-hybrid", "provider": "int8_mma"})
        if not os.environ.get("GPU1BIT_RUNTIME_KERNEL"):
            self.assertEqual(_selected_kernel(), "int8_mma")


class FreshProcess(unittest.TestCase):
    def setUp(self):
        if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
            self.skipTest("run via `gpu1bit.tests.run --cpu`: the parent must not touch the GPU")
        if os.environ.get("GPU1BIT_SPAWN_GPU_CHILDREN") != "1":
            self.skipTest("spawns GPU children (EnergyMeter guard: no foreign GPU process); run inside an agent-jobs "
                          "GPU lease with GPU1BIT_SPAWN_GPU_CHILDREN=1")

    def _vram(self, rt: str):
        kernel = _selected_kernel()
        out = env.artifacts("runtime-tests") / f"fresh-vram-{rt}-{kernel.replace(':', '_')}.json"
        if out.exists():
            out.unlink()
        proc = subprocess.run(["timeout", "600", PY, "-m", "gpu1bit.tests.runtime_child", "--runtime", rt,
                               "--kernel", kernel, "--parts", "3", "--out", str(out)], env=_child_env(), cwd=str(env.BINARY_DIR),
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
        got = json.loads(out.read_text())
        report, variant_of = _selection()
        session = json.loads(report.read_text())["clips"]
        ref = {c["id"]: max(r["clips"][i]["process_peak_incl_first_bytes"] for r in session["R-expanded"])
               for i, c in enumerate(session["R-expanded"][0]["clips"])}
        mine = session[variant_of[rt]]
        expected = {c["id"]: (max(r["clips"][i]["process_peak_incl_first_bytes"] for r in mine), c["text"])
                    for i, c in enumerate(mine[0]["clips"])}
        for clip in got["clips"]:
            with self.subTest(runtime=rt, clip=clip["id"]):
                peak = clip["process_peak_including_first_bytes"]
                self.assertLessEqual(peak, ref[clip["id"]] - 500_000_000, "not >= 500 MB below R-expanded")
                if kernel == _selection()[1]["provider"]:
                    bound, text = expected[clip["id"]]
                    self.assertLessEqual(peak, 1.02 * bound, f"{peak} B > 1.02 x selection P-memory {bound} B")
                    self.assertEqual(set(clip["texts"]), {text})
        return got

    def test_fresh_process_vram_hybrid(self):
        self._vram("hybrid")

    def test_fresh_process_vram_provider(self):
        self._vram("provider")

    def test_no_nemo_or_two_bit_files_opened(self):
        from gpu1bit.shared import workloads
        import paths
        clip = workloads.load()["dev_clips"][0]
        trace = env.artifacts("runtime-tests") / "strace-files.txt"
        proc = subprocess.run(["strace", "-f", "-qq", "-e", "trace=%file", "-o", str(trace),
                               "timeout", "600", PY, "-m", "gpu1bit.runtime", "--export", str(env.MB1_EXPORT),
                               "--parts", "3", "--tokens", "--label", "H:runtime strace", clip["audio_filepath"]],
                              env=_child_env(), cwd=str(env.BINARY_DIR), capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
        line = json.loads(proc.stdout.strip().splitlines()[-1])
        report, variant_of = _selection()
        session = json.loads(report.read_text())["clips"][variant_of["hybrid"]][0]["clips"][0]
        self.assertEqual(line["text"], session["text"])
        text = trace.read_text()
        self.assertIn(str(env.MB1_EXPORT / "export.safetensors"), text)
        self.assertIn(str(env.MB1_EXPORT / "tokenizer" / "tokenizer.model"), text)
        forbidden = {"a .nemo file": re.compile(r'\.nemo"'),
                     "M1's model directory": re.compile(re.escape(str(paths.MODEL_DIR))),
                     "M1's export": re.compile(re.escape(str(env.M1_EXPORT))),
                     "H's derived 2-bit export": re.compile(re.escape(str(env.DERIVED_EXPORT)))}
        for label, pattern in forbidden.items():
            hits = [l for l in text.splitlines() if pattern.search(l)]
            self.assertEqual(hits[:5], [], f"opened {label}")


if __name__ == "__main__":
    unittest.main()
