"""compute-sanitizer for int8_mma (plans/gpu-kernel.md A8): H's gpu1bit.shared.sanitize child, unchanged,
under the same Compute Sanitizer 12.9.79 command line plus `--launch-timeout 0`. Without it the tool's
attach timeout can expire while the child waits for the GPU lock, and the child then runs
uninstrumented with no summary (C's 09:22 note). Logs: /mnt/hd/.../gpu1bit/B/sanitizer/.

    gpu1bit/shared/python -m gpu1bit.int8_mma.sanitize_b --tool memcheck|racecheck [--impl int8_mma]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

from gpu1bit.shared import env, sanitize


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", default="int8_mma")
    p.add_argument("--tool", choices=["memcheck", "racecheck", "synccheck", "initcheck"], default="memcheck")
    args = p.parse_args()
    out = env.ARTIFACTS_ROOT / "B" / "sanitizer"
    env.require_mount()
    out.mkdir(parents=True, exist_ok=True)
    log = out / f"{time.strftime('%Y%m%dT%H%M%S')}-{args.tool}-{args.impl.replace(':', '_')}.log"
    cmd = ["timeout", "600", str(sanitize.SANITIZER), "--injection-path", str(sanitize.SANITIZER.parent),
           "--launch-timeout", "0", "--tool", args.tool, "--report-api-errors", "no", "--target-processes", "all",
           sys.executable, "-m", "gpu1bit.shared.sanitize", "--child", "--impl", args.impl, "--tool", args.tool,
           "--agent", "B"]
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    text = log.read_text()
    summary = [line for line in text.splitlines()
               if "ERROR SUMMARY" in line or "RACECHECK SUMMARY" in line or "SANITIZE-CHILD-DONE" in line]
    print("\n".join(summary) or text[-2000:])
    print(f"exit {rc}; log {log}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
