"""compute-sanitizer memcheck / racecheck on one small shape per kernel path of an implementation
(plans/gpu-kernel.md A8). Uses the Compute Sanitizer M1 used (12.9, under parakeet-ternary/kernel-tools).

    gpu1bit/shared/python -m gpu1bit.shared.sanitize --impl NAME --tool memcheck|racecheck [--agent H]

Run the outer command under shared/python (so the environment is the wrapper's); it starts
`timeout 600 compute-sanitizer --injection-path <lib> --tool T --report-api-errors no
--target-processes all <sys.executable> -m gpu1bit.shared.sanitize --child ...` with that
environment inherited (M1 also ran the sanitizer on the interpreter, not a wrapper); the child (the only
GPU process) holds the gpu lock for its life (A9) and runs linear and conv at M = 1, 5, 38 for
(1024, 1024), (3072, 1024), (1024, 4096), (2048, 1024) and a ragged (37, 1001), each eager and once
under CUDA-graph replay. API-error reporting is off, as in M1's memory-only runs (EXPERIMENTS3.md);
the log is kept under .../gpu1bit/<agent>/sanitizer/.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from gpu1bit.shared import env

# The extracted 12.9.79 archive (interpreters patched for NixOS, with TreeLauncherSubreaper); the
# pip wheel copy under kernel-tools/nvidia lacks TreeLauncherSubreaper and cannot launch targets.
SANITIZER = Path("/mnt/hd/wilderness-labs-stt/parakeet-ternary/kernel-tools/cuda_sanitizer_api-linux-x86_64-12.9.79-archive/compute-sanitizer/compute-sanitizer")
SHAPES = [(1024, 1024), (3072, 1024), (1024, 4096), (2048, 1024), (37, 1001)]


def child(impl_name: str, label: str) -> None:
    import torch
    from gpu1bit.shared import api, registry
    from gpu1bit.tests.test_contract import capture_and_replay, make_input, make_weights
    impl = registry.get(impl_name)
    with env.gpu_lock(label), torch.inference_mode():
        for i, (n, k) in enumerate(SHAPES):
            w = make_weights(n, k, 7000 + i, bias=(n == 37))
            layer = impl.prepare(w["codes"], w["scale"], w["bias"], k, "cuda")
            for m in (1, 5, 38):
                for kind, layout, shape in (("linear", "contiguous", (1, m, k)), ("conv", "transposed", (1, k, m)),
                                            ("conv", "contiguous", (1, k, m))):
                    x = make_input(kind, layout, shape, k, 8000 + m)
                    api.call(layer, kind, x)
                    capture_and_replay(lambda t: api.call(layer, kind, t), x, [x.clone()])
            torch.cuda.synchronize()
        print("SANITIZE-CHILD-DONE", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", required=True)
    p.add_argument("--tool", choices=["memcheck", "racecheck", "synccheck", "initcheck"], default="memcheck")
    p.add_argument("--agent", default="H")
    p.add_argument("--child", action="store_true")
    args = p.parse_args()
    label = f"{args.agent}:sanitizer {args.tool} {args.impl}"
    if args.child:
        child(args.impl, label)
        return
    if not SANITIZER.exists():
        raise SystemExit(f"compute-sanitizer not found at {SANITIZER}")
    out = env.ARTIFACTS_ROOT / args.agent / "sanitizer"
    env.require_mount()
    out.mkdir(parents=True, exist_ok=True)
    log = out / f"{time.strftime('%Y%m%dT%H%M%S')}-{args.tool}-{args.impl.replace(':', '_')}.log"
    cmd = ["timeout", "600", str(SANITIZER), "--injection-path", str(SANITIZER.parent),
           "--tool", args.tool, "--report-api-errors", "no", "--target-processes", "all",
           sys.executable, "-m", "gpu1bit.shared.sanitize", "--child", "--impl", args.impl,
           "--tool", args.tool, "--agent", args.agent]
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    text = log.read_text()
    summary = [l for l in text.splitlines() if "ERROR SUMMARY" in l or "RACECHECK SUMMARY" in l or "SANITIZE-CHILD-DONE" in l]
    print("\n".join(summary) or text[-2000:])
    print(f"exit {rc}; log {log}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
