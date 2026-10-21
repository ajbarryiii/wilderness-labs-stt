"""Lock scope for agent A's GPU processes (plans/gpu-kernel.md A9).

`run_locked(label, main)` takes gpu_lock before the first CUDA call and never releases it while
the process is alive: after main() returns (or raises) it flushes output and calls os._exit while
still holding the lock, so the kernel drops the CUDA context and the flock together at exit and no
later lock holder can see this process on the GPU.
"""
from __future__ import annotations

import os
import sys
import traceback

from gpu1bit.shared import env


def run_locked(label: str, main) -> None:
    code = 1
    with env.gpu_lock(label) as state:
        try:
            result = main(state)
            code = 0 if result is None else int(result)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
        except BaseException:
            traceback.print_exc()
            code = 1
        finally:
            try:
                import torch
                if torch.cuda.is_initialized():
                    torch.cuda.synchronize()
            except Exception:
                pass
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(code)


class locked:
    """`with locked(label) as state:` -- the body runs holding the GPU lock and the process then
    exits (os._exit) still holding it: code after the with-block never runs. Exit code 0, or 1 if the
    body raised (traceback printed); a SystemExit's code is kept."""

    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self._cm = env.gpu_lock(self.label)
        return self._cm.__enter__()

    def __exit__(self, exc_type, exc, tb):
        code = 0
        if exc_type is SystemExit:
            code = exc.code if isinstance(exc.code, int) else 1
        elif exc_type is not None:
            traceback.print_exception(exc_type, exc, tb)
            code = 1
        try:
            import torch
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
        except Exception:
            code = code or 1
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
