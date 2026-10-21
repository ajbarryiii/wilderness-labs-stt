"""Whole-process GPU lock for agent B's tools (plans/gpu-kernel.md A9).

`locked(label)` takes gpu1bit.shared.env.gpu_lock before the caller's first CUDA call and never
releases it while this process can still own a CUDA context: on leaving the block (normally or by
an exception) it synchronizes, flushes stdout/stderr and ends the process with os._exit INSIDE the
lock, so no context outlives the hold (the race H reported at 08:22 for a lock released before
interpreter teardown). The exit code is 0, or 1 after printing the traceback.
"""
from __future__ import annotations

import contextlib
import os
import sys
import traceback


@contextlib.contextmanager
def locked(label: str):
    from gpu1bit.shared import env
    code = 0
    with env.gpu_lock(label) as state:
        try:
            yield state
        except BaseException:  # noqa: BLE001 - reported, then the process ends inside the lock
            traceback.print_exc()
            code = 1
        finally:
            try:
                import torch
                if torch.cuda.is_initialized():
                    torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                code = code or 1
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(code)


def run_unittests(label: str, modules: list[str]) -> None:
    """unittest for `modules` with the lock held for the whole process (own suite runner)."""
    import unittest
    with locked(label):
        suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)


if __name__ == "__main__":
    run_unittests("B:own tests", sys.argv[1:] or ["gpu1bit.int8_mma.test_int8_mma"])
