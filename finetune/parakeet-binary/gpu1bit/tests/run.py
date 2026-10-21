"""Run gpu1bit contract tests in ONE process holding ONE GPU lock (plans/gpu-kernel.md A9).

    timeout 600 finetune/parakeet-binary/gpu1bit/shared/python -m gpu1bit.tests.run \\
        [--label A:contract] [--impl NAME[,NAME]] [--cpu] [MODULE_OR_TEST ...]

MODULE_OR_TEST defaults to every test module (gpu1bit.tests.test_*). --impl restricts the
implementations under test (registry names or directory names; h_reference is H's known-good one).
--cpu runs only CPU tests (test_api, test_derived) with CUDA hidden and no lock.
The lock is taken before CUDA is initialised and released at exit; if the hold reaches 570 s the
run stops with a failure naming the remaining tests (narrow the selection and rerun). A JSON
summary goes to /mnt/hd/wilderness-labs-stt/parakeet-binary/gpu1bit/H/test-runs/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import unittest

CPU_MODULES = ["gpu1bit.tests.test_api", "gpu1bit.tests.test_derived"]
GPU_MODULES = ["gpu1bit.tests.test_meta", "gpu1bit.tests.test_contract", "gpu1bit.tests.test_numerics_real", "gpu1bit.tests.test_provider"]


class _Budget(unittest.TextTestResult):
    lock_state: dict | None = None
    limit_s = 570.0

    def startTest(self, test):
        if self.lock_state is not None:
            held = time.monotonic() - self.lock_state["acquired"]
            if held > self.limit_s:
                self.stop()
                self.addError(test, (RuntimeError, RuntimeError(
                    f"GPU lock held {held:.0f} s; stopping before the 600 s limit (narrow the selection)"), None))
                return
        super().startTest(test)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", default="H:contract-tests")
    p.add_argument("--impl", default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("tests", nargs="*")
    args = p.parse_args()
    if args.impl:
        os.environ["GPU1BIT_IMPLEMENTATIONS"] = args.impl
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS across streams
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from gpu1bit.shared import env
    env.require_mount()
    names = args.tests or (CPU_MODULES if args.cpu else CPU_MODULES + GPU_MODULES)
    started = time.time()
    with (env.gpu_lock(args.label) if not args.cpu else _nullcontext({"acquired": time.monotonic(), "waited_s": 0})) as lock:
        _Budget.lock_state = None if args.cpu else lock
        suite = unittest.defaultTestLoader.loadTestsFromNames(names)
        runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1, resultclass=_Budget)
        result = runner.run(suite)
        held = time.monotonic() - lock["acquired"]
    summary = {"label": args.label, "tests": names, "impl": os.environ.get("GPU1BIT_IMPLEMENTATIONS"),
               "ran": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
               "skipped": len(result.skipped), "ok": result.wasSuccessful(), "lock_waited_s": lock["waited_s"],
               "lock_held_s": held, "wall_s": time.time() - started, "wrapper": env.wrapper(),
               "failed": [t.id() for t, _ in result.failures + result.errors],
               "skips": [(t.id(), r) for t, r in result.skipped]}
    out = env.artifacts("test-runs") / f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}.json"
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("skips",)}), file=sys.stderr)
    return 0 if result.wasSuccessful() else 1


class _nullcontext:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    sys.exit(main())
