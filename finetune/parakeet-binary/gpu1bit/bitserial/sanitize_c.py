"""H's compute-sanitizer run (gpu1bit/shared/sanitize.py, unchanged) with --launch-timeout 0.

H's outer command uses the sanitizer's default attach timeout; its child takes the GPU lock before
creating a CUDA context, so when another job holds the lock the sanitizer gives up attaching
("No attachable process found") and the child then runs uninstrumented. This wrapper starts the same
child, the same tool and options, but waits for attachment indefinitely (bounded by `timeout`).

  finetune/parakeet-binary/gpu1bit/shared/python -m gpu1bit.bitserial.sanitize_c --tool memcheck|racecheck
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

from gpu1bit.shared import env
from gpu1bit.shared.sanitize import SANITIZER


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tool', choices=['memcheck', 'racecheck', 'synccheck', 'initcheck'], default='memcheck')
    p.add_argument('--impl', default='bitserial')
    args = p.parse_args()
    out = env.ARTIFACTS_ROOT / 'C' / 'sanitizer'
    env.require_mount()
    out.mkdir(parents=True, exist_ok=True)
    log = out / f'{time.strftime("%Y%m%dT%H%M%S")}-{args.tool}-{args.impl}-launch-timeout-0.log'
    cmd = ['timeout', '1500', str(SANITIZER), '--injection-path', str(SANITIZER.parent), '--launch-timeout', '0',
           '--tool', args.tool, '--report-api-errors', 'no', '--target-processes', 'all',
           sys.executable, '-m', 'gpu1bit.shared.sanitize', '--child', '--impl', args.impl,
           '--tool', args.tool, '--agent', 'C']
    with open(log, 'w') as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    text = log.read_text()
    summary = [l for l in text.splitlines() if 'ERROR SUMMARY' in l or 'RACECHECK SUMMARY' in l
               or 'SANITIZE-CHILD-DONE' in l or 'Error' in l]
    print('\n'.join(summary) or text[-2000:])
    print(f'exit {rc}; log {log}')
    sys.exit(rc)


if __name__ == '__main__':
    main()
