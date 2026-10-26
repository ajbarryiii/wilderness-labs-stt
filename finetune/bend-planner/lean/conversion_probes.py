#!/usr/bin/env python3
"""Resource-bounded checker conversion probes; every failure is retained verbatim."""
import argparse
import json
import os
import tempfile
from pathlib import Path
import subprocess
import time

from fidelity import vectors, bend_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, default=Path(os.environ.get("BEND_LEAN_SCRATCH", Path(tempfile.gettempdir()) / "bend-lean-review")) / "conversions")
    args = ap.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    results = []
    edges, _, _ = vectors(0)
    for name, ty, expression in [
        ("to_nat", "Nat", "U32.to_nat({x})"),
        ("from_nat", "U32", "U32.from_nat({x}n)"),
    ]:
        for x in edges:
            p = args.scratch / f"{name}-{x}.bend"
            p.write_text(f"import Base\n\ndef main() -> {ty}:\n  {expression.format(x=x)}\n")
            started = time.monotonic()
            try:
                got = subprocess.run(["bend", str(p)], capture_output=True, text=True, timeout=5)
                record = dict(op=name, input=x, returncode=got.returncode,
                              stdout=got.stdout, stderr=got.stderr)
                if got.returncode == 0:
                    assert int(got.stdout.strip().removesuffix("n")) == x
            except subprocess.TimeoutExpired as exc:
                record = dict(op=name, input=x, timed_out_seconds=5,
                              stdout=(exc.stdout or b"").decode(), stderr=(exc.stderr or b"").decode())
            record["elapsed_seconds"] = time.monotonic() - started
            results.append(record)
            print(json.dumps(record), flush=True)
    small_count = 0
    for offset in range(0, 257, 16):
        xs = range(offset, min(offset+16, 257))
        p = args.scratch / f"small-{offset:03}.bend"
        row = ", ".join(f"U32.to_nat(U32.from_nat({x}n))" for x in xs)
        p.write_text(f"import Base\n\ndef main() -> List<&2, Nat>:\n  [{row}]\n")
        got = subprocess.run(["bend", str(p)], capture_output=True, text=True, timeout=30)
        if got.returncode:
            raise RuntimeError(got.stderr)
        assert bend_list(got.stdout) == list(xs)
        small_count += len(list(xs))
    summary = dict(edge_probes=results, small_roundtrips_passed=small_count,
                   limitation="Checker conversion results could not be normalized for all full-width inputs; see individual resource failures.")
    (args.scratch / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Checker small roundtrips passed: {small_count}", flush=True)


if __name__ == "__main__":
    main()
