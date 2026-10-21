"""One entry point for patched variants: import the roster's patch modules, then run an H tool.

    gpu1bit/shared/python -m gpu1bit.shared.dispatch --patches mod1,mod2 <tool> [tool args]
tools: bench (shared/bench.py), gate (shared/gate.py; the patch list is passed on for its fingerprint),
       audit (selection.py audit), phases (shared/phases.py), deferred (shared/deferred_a3.py),
       tests (tests/run.py).
Patch modules (B's, in gpu1bit/int8_mma/) register variants by patching registry.discover /
bench.variant_model, as int8_mma/paired.py and b01_run.py do; they are imported in the given order,
so a later module can override an earlier one's variant name.
"""
from __future__ import annotations

import importlib
import sys


def main() -> None:
    argv = sys.argv[1:]
    patches: list[str] = []
    if argv[:1] == ["--patches"]:
        patches = [m for m in argv[1].split(",") if m]
        argv = argv[2:]
    for module in patches:
        importlib.import_module(module)
    if not argv:
        raise SystemExit(__doc__)
    tool, rest = argv[0], argv[1:]
    if tool == "bench":
        from gpu1bit.shared import bench
        sys.argv = ["bench", *rest]
        bench.main()
    elif tool == "gate":
        from gpu1bit.shared import gate
        sys.argv = ["gate", *rest, "--patches", ",".join(patches)]
        gate.main()
    elif tool == "audit":
        from gpu1bit.shared import selection
        sys.argv = ["selection", "audit", *rest, "--patches", ",".join(patches)]
        selection.main()
    elif tool == "phases":
        from gpu1bit.shared import phases
        sys.argv = ["phases", *rest]
        phases.main()
    elif tool == "deferred":
        from gpu1bit.shared import deferred_a3
        sys.argv = ["deferred_a3", *rest]
        deferred_a3.main()
    elif tool == "tests":
        from gpu1bit.tests import run
        sys.argv = ["run", *rest]
        sys.exit(run.main())
    else:
        raise SystemExit(f"unknown tool {tool!r}")


if __name__ == "__main__":
    main()
