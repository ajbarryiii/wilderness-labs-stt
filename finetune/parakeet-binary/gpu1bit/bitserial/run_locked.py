"""Run a module or script while this process holds paths.gpu_lock once (plans/gpu-kernel.md A9).

  timeout 600 finetune/parakeet-binary/gpu1bit/shared/python \
      finetune/parakeet-binary/gpu1bit/bitserial/run_locked.py C:<what> -m unittest gpu1bit.bitserial.test_bitserial -v
  timeout 600 .../shared/python .../bitserial/run_locked.py C:<what> path/to/script.py args...

Refuses to start unless the power limit is exactly 400 W and no other compute process is on the GPU.
"""
from __future__ import annotations

import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # finetune/parakeet-binary (gpu1bit namespace package)

from gpu1bit.shared import env  # noqa: E402  (also puts parakeet-ternary on sys.path)


def gpu_preflight() -> None:
    limit = subprocess.run(['nvidia-smi', '--query-gpu=power.limit', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, check=True).stdout.strip()
    if float(limit) != env.POWER_LIMIT_W:
        raise RuntimeError(f'power limit {limit} W != {env.POWER_LIMIT_W} W')
    deadline = time.monotonic() + 60  # the previous lock holder may still be tearing down its context
    while True:
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                              capture_output=True, text=True, check=True).stdout.split()
        foreign = [p for p in apps if p.strip() and int(p) != os.getpid()]
        if not foreign:
            return
        if time.monotonic() > deadline:
            raise RuntimeError(f'foreign compute processes on the GPU: {foreign}')
        time.sleep(1)


def main() -> None:
    if len(sys.argv) < 3 or not sys.argv[1].startswith('C:'):
        raise SystemExit(__doc__)
    label, rest = sys.argv[1], sys.argv[2:]
    with env.gpu_lock(label) as state:
        gpu_preflight()
        start = time.monotonic()
        os.environ['C_LOCK_ACQUIRED_MONOTONIC'] = repr(state['acquired'])
        print(f'LOCK {label} acquired after {state["waited_s"]:.1f} s, wrapper {env.wrapper()["wrapper"]}', flush=True)
        try:
            if rest[0] == '-m':
                sys.argv = [rest[1]] + rest[2:]
                runpy.run_module(rest[1], run_name='__main__', alter_sys=True)
            else:
                sys.argv = rest
                runpy.run_path(rest[0], run_name='__main__')
        finally:
            print(f'LOCK {label} released after {time.monotonic() - start:.1f} s', flush=True)


if __name__ == '__main__':
    main()
