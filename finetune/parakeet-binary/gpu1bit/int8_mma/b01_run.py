"""H's tools, unchanged, with the post-selection variant registered (patches live only in int8_mma/).

* registry.discover additionally returns "int8_mma:b01" (layer_b01.IMPLEMENTATIONS["b01"]); the frozen
  layer.py keeps exposing only IMPLEMENTATION, so "int8_mma" itself is untouched.
* bench.variant_model (already extended by the frozen paired.py with "B-hybrid") additionally builds
  "B-hybrid-b01": R-hybrid's context stack (frozen hybrid.hybrid_pipeline) with the frozen
  HybridProjection over load_binary_model(MB1, Int8MmaB01Layer).
Subcommands (each GPU run its own `timeout 600` process holding one gpu_lock, as H's tools do):
    gpu1bit/shared/python -m gpu1bit.int8_mma.b01_run tests [H's gpu1bit.tests.run args]
    gpu1bit/shared/python -m gpu1bit.int8_mma.b01_run bench [H's gpu1bit.shared.bench args]
    gpu1bit/shared/python -m gpu1bit.int8_mma.b01_run gate [H's gpu1bit.shared.gate args]
    gpu1bit/shared/python -m gpu1bit.int8_mma.b01_run sanitize --tool memcheck|racecheck [--impl int8_mma:b01]
"""
from __future__ import annotations

import contextlib
import os
import sys

from gpu1bit.int8_mma import paired  # noqa: F401  (frozen: patches bench.variant_model for "B-hybrid")
from gpu1bit.shared import bench, registry

NAME = "int8_mma:b01"
_discover = registry.discover


def discover(include_reference: bool = True):
    found = _discover(include_reference)
    wanted = os.environ.get("GPU1BIT_IMPLEMENTATIONS")
    keys = [w.strip() for w in wanted.split(",") if w.strip()] if wanted else None
    if keys is None or any(NAME == w or NAME.split(":")[0] == w for w in keys):
        try:
            from gpu1bit.int8_mma.layer_b01 import IMPLEMENTATIONS
            found[NAME] = IMPLEMENTATIONS["b01"]
        except Exception as exc:  # noqa: BLE001 - reported per candidate, as registry does
            found[NAME] = ImportError(f"gpu1bit/int8_mma/layer_b01.py failed to import: {exc!r}")
    return found


registry.discover = discover
_source_path = registry.source_path


def source_path(name: str):
    """Provenance (numerics reports hash this file): the variant's own module, not the frozen layer.py."""
    if name == NAME:
        from pathlib import Path
        return Path(__file__).resolve().parent / "layer_b01.py"
    return _source_path(name)


registry.source_path = source_path


@contextlib.contextmanager
def b_hybrid_b01_model(max_graphs: int = 4):
    """hybrid.b_hybrid_model with the b01 layer class (same stack, same provider class)."""
    import evaluate as ev
    from gpu1bit.int8_mma.hybrid import HybridProjection, hybrid_pipeline
    from gpu1bit.int8_mma.layer_b01 import Int8MmaB01Layer
    from gpu1bit.shared import env, provider as P
    if max_graphs != 4:
        raise ValueError("R-hybrid's pipeline fixes max_graphs=4")
    model, table = P.load_binary_model(env.MB1_EXPORT, Int8MmaB01Layer)
    with ev.inference_settings(model):
        model.decoding.decoding.decoding_computer.force_cuda_graphs_mode("full_graph")
        provider = HybridProjection(table, P.module_names(model))
        with hybrid_pipeline(model, provider):
            yield model, {"kind": "candidate", "stack": "experimental_pipeline(ptx_row_best) contexts + HybridProjection (b01)",
                          "projection": provider, "weight_bytes": table.weight_bytes()}


_variant_model = bench.variant_model


@contextlib.contextmanager
def variant_model(name: str, *, max_graphs: int = 4):
    if name == "B-hybrid-b01":
        with b_hybrid_b01_model(max_graphs=4) as pair:
            yield pair
    else:
        with _variant_model(name, max_graphs=max_graphs) as pair:
            yield pair


bench.variant_model = variant_model


def sanitize() -> None:
    """int8_mma/sanitize_b.py's command line with this module as the (patched) child."""
    import argparse
    import subprocess
    import time
    from gpu1bit.shared import env
    from gpu1bit.shared import sanitize as S
    p = argparse.ArgumentParser()
    p.add_argument("--impl", default=NAME)
    p.add_argument("--tool", choices=["memcheck", "racecheck", "synccheck", "initcheck"], default="memcheck")
    args = p.parse_args(sys.argv[2:])
    out = env.ARTIFACTS_ROOT / "B" / "sanitizer"
    env.require_mount()
    out.mkdir(parents=True, exist_ok=True)
    log = out / f"{time.strftime('%Y%m%dT%H%M%S')}-{args.tool}-{args.impl.replace(':', '_')}.log"
    cmd = ["timeout", "600", str(S.SANITIZER), "--injection-path", str(S.SANITIZER.parent), "--launch-timeout", "0",
           "--tool", args.tool, "--report-api-errors", "no", "--target-processes", "all", sys.executable, "-m",
           "gpu1bit.int8_mma.b01_run", "sanitize-child", "--impl", args.impl, "--tool", args.tool]
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    text = log.read_text()
    print("\n".join(l for l in text.splitlines() if "SUMMARY" in l or "SANITIZE-CHILD-DONE" in l) or text[-2000:])
    print(f"exit {rc}; log {log}")
    sys.exit(rc)


def main() -> None:
    command = sys.argv[1]
    if command == "tests":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from gpu1bit.tests import run
        sys.exit(run.main())
    elif command == "bench":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        bench.main()
    elif command == "gate":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from gpu1bit.shared import gate
        gate.main()
    elif command == "sanitize":
        sanitize()
    elif command == "sanitize-child":
        import argparse
        from gpu1bit.shared import sanitize as S
        p = argparse.ArgumentParser()
        p.add_argument("--impl", default=NAME)
        p.add_argument("--tool", default="memcheck")
        args = p.parse_args(sys.argv[2:])
        S.child(args.impl, f"B:sanitizer {args.tool} {args.impl}")
    else:
        raise SystemExit(f"unknown command {command!r}")


if __name__ == "__main__":
    main()
