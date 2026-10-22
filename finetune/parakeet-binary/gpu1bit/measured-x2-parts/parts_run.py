"""H's tools, unchanged, with the activation-part-count variants registered (patch lives only in int8_mma/).

On top of b01_run.py (int8_mma:b01, B-hybrid-b01), registry.discover also returns
    int8_mma:x2, int8_mma:x1, int8_mma:x2:b01, int8_mma:x1:b01      (layer_parts.IMPLEMENTATIONS)
and registry.source_path points those at layer_parts.py. Every Int8MmaPartsLayer's `project` honours
`components` in {1, 2, 3} (H's PartCount wrapper, plans/activation-precision.md R1); a variant's own part
count only sets linear / conv. Subcommands as in b01_run.py:
    gpu1bit/shared/python -m gpu1bit.int8_mma.parts_run tests [H's gpu1bit.tests.run args]
    gpu1bit/shared/python -m gpu1bit.int8_mma.parts_run bench [H's gpu1bit.shared.bench args]
    gpu1bit/shared/python -m gpu1bit.int8_mma.parts_run gate [H's gpu1bit.shared.gate args]
    gpu1bit/shared/python -m gpu1bit.int8_mma.parts_run sanitize --tool memcheck|racecheck --impl int8_mma:x1
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from gpu1bit.int8_mma import b01_run
from gpu1bit.shared import registry

NAMES = ("int8_mma:x2", "int8_mma:x1", "int8_mma:x2:b01", "int8_mma:x1:b01")
_discover = registry.discover  # b01_run's patched discover


def discover(include_reference: bool = True):
    found = _discover(include_reference)
    wanted = os.environ.get("GPU1BIT_IMPLEMENTATIONS")
    keys = [w.strip() for w in wanted.split(",") if w.strip()] if wanted else None
    try:
        from gpu1bit.int8_mma.layer_parts import IMPLEMENTATIONS
        classes = {f"int8_mma:{k}": v for k, v in IMPLEMENTATIONS.items()}
    except Exception as exc:  # noqa: BLE001 - reported per candidate, as registry does
        classes = {n: ImportError(f"gpu1bit/int8_mma/layer_parts.py failed to import: {exc!r}") for n in NAMES}
    for name in NAMES:
        if keys is None or any(name == w or name.split(":")[0] == w for w in keys):
            found[name] = classes[name]
    return found


registry.discover = discover
_source_path = registry.source_path


def source_path(name: str):
    if name in NAMES:
        return Path(__file__).resolve().parent / "layer_parts.py"
    return _source_path(name)


registry.source_path = source_path


def main() -> None:
    command = sys.argv[1]
    if command in ("sanitize", "sanitize-child"):
        import argparse
        import subprocess
        import time
        from gpu1bit.shared import env
        from gpu1bit.shared import sanitize as S
        p = argparse.ArgumentParser()
        p.add_argument("--impl", default="int8_mma:x1")
        p.add_argument("--tool", choices=["memcheck", "racecheck", "synccheck", "initcheck"], default="memcheck")
        args = p.parse_args(sys.argv[2:])
        if command == "sanitize-child":
            S.child(args.impl, f"B:sanitizer {args.tool} {args.impl}")
            return
        out = env.ARTIFACTS_ROOT / "B" / "sanitizer"
        env.require_mount()
        out.mkdir(parents=True, exist_ok=True)
        log = out / f"{time.strftime('%Y%m%dT%H%M%S')}-{args.tool}-{args.impl.replace(':', '_')}.log"
        here = Path(__file__).resolve().parent
        child = ([sys.executable, "-m", "gpu1bit.int8_mma.parts_run"] if here.name == "int8_mma"
                 else [sys.executable, str(here / "wip_main.py")])  # while the modules live outside int8_mma/
        cmd = ["timeout", "600", str(S.SANITIZER), "--injection-path", str(S.SANITIZER.parent), "--launch-timeout", "0",
               "--tool", args.tool, "--report-api-errors", "no", "--target-processes", "all", *child,
               "sanitize-child", "--impl", args.impl, "--tool", args.tool]
        with open(log, "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        text = log.read_text()
        print("\n".join(l for l in text.splitlines() if "SUMMARY" in l or "SANITIZE-CHILD-DONE" in l) or text[-2000:])
        print(f"exit {rc}; log {log}")
        sys.exit(rc)
    b01_run.main()


if __name__ == "__main__":
    main()
