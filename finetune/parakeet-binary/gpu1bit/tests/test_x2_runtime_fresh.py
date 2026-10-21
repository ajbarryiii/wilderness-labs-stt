"""Z6/Z8 memory contract of the x2 runtime against the recorded benchmark (plans/x2-default.md Z6, Z8). Runs
after the recorded benchmark and before promotion and test scoring, on the unchanged frozen sources, with
the GPU exclusive (the child's NVML process-memory sampling refuses foreign GPU processes, as P-memory):

    GPU1BIT_X2_BENCH_RUN=<benchmark run> gpu1bit/shared/python -m gpu1bit.tests.run --cpu \\
        gpu1bit.tests.test_x2_runtime_fresh

THIS process never touches the GPU; the child (tests/runtime_child.py, its own `timeout 600` GPU process
holding the lock) loads gpu1bit.runtime.load_binary_packed(MB1, runtime hybrid, selection=selected-x2.json)
with the default part count (2), in the deployment environment, and samples NVML process memory every
10 ms from before loading. test_runtime_fresh's contract with its reference replaced (Z6), via
x2/runtime_contracts.py (fixture-tested in test_x2_preflight_runtime):
  - per dev clip, the peak including first capture is at most 1.02 x b1's recorded P-memory peak (the max
    over the benchmark's P-clips runs <run>-clips1 / <run>-clips2, both required, each with all three
    frozen clips), and at least 500 MB below R-expanded's (b3);
  - transcripts equal b1's;
  - b1's records carry the runtime's identity (the child's), and the child runs at parts 2 under the
    candidate's content hash.
On failure nothing is promoted or scored, and the user is told (Z8). The record
(H/x2/runtime-fresh-compare-<run>.json: ok, bench_run, identity) is what the test-scoring precheck requires.
Without GPU1BIT_X2_BENCH_RUN the module fails (never skipped silently).
"""
from __future__ import annotations

import json
import os
import subprocess
import unittest

from gpu1bit.shared import env

PY = str(env.SHARED / "python")
RUN = os.environ.get("GPU1BIT_X2_BENCH_RUN")
LABELS = ("clips1", "clips2")


def clips_docs(run: str, variant: str) -> dict:
    """{label: recorded P-clips document or None} for both orders (runtime_contracts requires both)."""
    out = {}
    for label in LABELS:
        p = env.ARTIFACTS / "bench" / f"{run}-{label}" / f"clips-{variant.replace(':', '_')}.json"
        out[label] = json.loads(p.read_text()) if p.exists() else None
    return out


class PMemoryAgainstB1(unittest.TestCase):
    def test_fresh_process_vram_x2(self):
        if not RUN:
            self.fail("set GPU1BIT_X2_BENCH_RUN to the recorded benchmark run (this contract needs b1's P-memory)")
        if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
            self.fail("run via `gpu1bit.tests.run --cpu`: the parent must not touch the GPU")
        from gpu1bit.shared import seldoc
        from gpu1bit.x2 import runtime_contracts as RC
        out = env.artifacts("x2") / f"runtime-fresh-{RUN}.json"
        if out.exists():
            out.unlink()
        e = dict(os.environ)
        for k in ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "GPU1BIT_SELECTION"):
            e.pop(k, None)
        proc = subprocess.run(["timeout", "600", PY, "-m", "gpu1bit.tests.runtime_child", "--runtime", "hybrid",
                               "--selection", str(seldoc.CANDIDATE), "--out", str(out)], env=e, cwd=str(env.BINARY_DIR),
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
        got = json.loads(out.read_text())
        self.assertEqual(got["parts"], 2)
        self.assertEqual(got["selection_sha256"], env.sha256_file(seldoc.CANDIDATE))
        ref = RC.pmemory_reference(clips_docs(RUN, "MB1-B-hybrid-x2"), runtime_identity=got["identity"])
        rexp = RC.r_expanded_reference(clips_docs(RUN, "R-expanded"))
        fresh = {c["id"]: {"peak": c["process_peak_including_first_bytes"], "texts": c["texts"]} for c in got["clips"]}
        result = RC.compare_pmemory(fresh, ref, rexp)
        record = {**result, "bench_run": RUN, "identity": got["identity"], "parts": got["parts"],
                  "selection_sha256": got["selection_sha256"], "fresh_report": str(out)}
        (env.artifacts("x2") / f"runtime-fresh-compare-{RUN}.json").write_text(json.dumps(record, indent=1) + "\n")
        self.assertTrue(result["ok"], result["rows"])


if __name__ == "__main__":
    unittest.main()
