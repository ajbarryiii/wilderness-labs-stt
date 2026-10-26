#!/usr/bin/env python3
"""Reproducible differential test of the installed Bend lanes and Lean model.

Run from the Lean package inside `nix develop`, using an installed Python 3.
All generated Bend files and raw vectors/results go under --scratch (default /tmp).
The checker fixture has a pure main; no I/O compilation or foreign effects occur.
"""
import argparse
import hashlib
import itertools
import json
import os
import tempfile
from pathlib import Path
import re
import subprocess
import time

MASK = (1 << 32) - 1
SEED = 20261024
OPS = [
    "a", "b", "add", "sub", "mul", "div", "mod", "eq", "ne", "lt", "le", "gt", "ge",
    "inc", "not", "and", "or", "xor", "shl", "shr", "shln", "shrn", "min", "max",
    "is_zero", "is_even", "log2",
]
CONVS = ["to_nat_a", "to_nat_b", "from_nat_a", "from_nat_sum", "from_nat_succ", "from_nat_wide"]

ROW = '''import Base

def row(+a: U32, +b: U32) -> List<&2, U32>:
  [a, b, U32.add(a, b), U32.sub(a, b), U32.mul(a, b), U32.div(a, b), U32.mod(a, b),
   Bool.to_u32(U32.is_eq(a, b)), Bool.to_u32(U32.is_ne(a, b)),
   Bool.to_u32(U32.is_lt(a, b)), Bool.to_u32(U32.is_le(a, b)),
   Bool.to_u32(U32.is_gt(a, b)), Bool.to_u32(U32.is_ge(a, b)),
   U32.inc(a), U32.not(a), U32.and(a, b), U32.or(a, b), U32.xor(a, b),
   U32.shl(a), U32.shr(a), U32.shln(a, U32.to_nat(U32.mod(b, 65))),
   U32.shrn(a, U32.to_nat(U32.mod(b, 65))), U32.min(a, b), U32.max(a, b),
   Bool.to_u32(U32.is_zero(a)), Bool.to_u32(U32.is_even(a)), U32.from_nat(U32.log2(a))]
'''

NATIVE = '''
def nat_rows(xs: List<&2, U32>) -> List<&2, Nat>:
  match xs:
    case Nil{}:
      Nil{}
    case h <> t:
      U32.to_nat(h) <> nat_rows(t)

def native_row(+a: U32, +b: U32) -> List<&2, Nat>:
  List.append(&2, Nat, nat_rows(row(a, b)),
    [U32.to_nat(a), U32.to_nat(b), U32.to_nat(U32.from_nat(U32.to_nat(a))),
     U32.to_nat(U32.from_nat(Nat.add(U32.to_nat(a), U32.to_nat(b)))),
     U32.to_nat(U32.from_nat(1n+U32.to_nat(a))),
     U32.to_nat(U32.from_nat(Nat.mul(U32.to_nat(a), 65536n)))])

def next(+x: U32) -> U32:
  +y = U32.xor(x, U32.shln(x, 13n))
  +z = U32.xor(y, U32.shrn(y, 17n))
  U32.xor(z, U32.shln(z, 5n))

def random_rows(n: Nat, state: U32) -> List<&2, List<&2, Nat>>:
  match n:
    case 0n:
      Nil{}
    case 1n+p:
      +a = next(state)
      +b = next(a)
      native_row(a, b) <> random_rows(p, b)
'''


def next_u32(x):
    x ^= (x << 13) & MASK
    x ^= x >> 17
    x ^= (x << 5) & MASK
    return x & MASK


def vectors(count):
    anchors = [0, 1, 2, 1 << 16, (1 << 31) - 1, 1 << 31, MASK - 1, MASK]
    edges = sorted({v + d for v in anchors for d in [-1, 0, 1] if 0 <= v + d <= MASK})
    pairs = list(itertools.product(edges, repeat=2))
    edge_count = len(pairs)
    state = SEED
    for _ in range(count):
        a = next_u32(state)
        b = next_u32(a)
        pairs.append((a, b))
        state = b
    return edges, edge_count, pairs


def run(cmd, *, timeout, output=None):
    started = time.monotonic()
    if output:
        with output.open("w") as f:
            p = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True, timeout=timeout)
        stdout = None
    else:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        stdout = p.stdout
    if p.returncode:
        raise RuntimeError(f"{cmd}: exit={p.returncode}\n{p.stderr}\n{stdout or ''}")
    return stdout, time.monotonic() - started


def bend_list(text):
    return json.loads(re.sub(r"(?<=\d)n\b", "", text.strip()))


def check_rows(label, actual, expected, pairs, names, findings):
    if len(actual) != len(expected):
        raise RuntimeError(f"{label}: {len(actual)} rows, expected {len(expected)}")
    for index, (got, want) in enumerate(zip(actual, expected)):
        if len(got) != len(want):
            raise RuntimeError(f"{label}: row {index}: {len(got)} columns, expected {len(want)}")
        for name, g, w in zip(names, got, want):
            if g != w:
                findings.append(dict(lane=label, index=index, inputs=pairs[index], op=name, got=g, expected=w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, default=Path(os.environ.get("BEND_LEAN_SCRATCH", Path(tempfile.gettempdir()) / "bend-lean-review")) / "fidelity")
    ap.add_argument("--random", type=int, default=200000)
    ap.add_argument("--checker-random", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    scratch = args.scratch
    scratch.mkdir(parents=True, exist_ok=True)
    edges, nedge, pairs = vectors(args.random)
    subset = pairs[:nedge + args.checker_random]
    inp = scratch / "pairs.tsv"
    inp.write_text("".join(f"{a}\t{b}\n" for a,b in pairs))
    edge_rows = ",\n   ".join(f"native_row({a}, {b})" for a,b in pairs[:nedge])
    native = scratch / "native.bend"
    native.write_text(ROW + NATIVE + f'''
def main() -> List<&2, List<&2, Nat>>:
  List.append(&2, List<&2, Nat>, [{edge_rows}], random_rows({args.random}n, {SEED}))
''')
    for offset in range(0, len(subset), args.chunk):
        rows = ",\n   ".join(f"row({a}, {b})" for a,b in subset[offset:offset+args.chunk])
        (scratch / f"checker-{offset:05}.bend").write_text(ROW + f'''
def main() -> List<&2, List<&2, U32>>:
  [{rows}]
''')
    report = dict(seed=SEED, generator="xorshift32(13,17,5); consecutive nonoverlapping pairs",
                  edges=edges, edge_pairs=nedge, random_pairs=args.random,
                  checker_random_pairs=args.checker_random, total=len(pairs), checker_total=len(subset),
                  operations=OPS, conversion_operations=CONVS,
                  input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest())
    (scratch / "vectors.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if args.prepare_only:
        return
    lean_out = scratch / "lean.tsv"
    lean_cmd = [".lake/build/bin/fidelity", str(inp)]
    if Path("/etc/NIXOS").exists():
        deps = subprocess.check_output(["ldd", lean_cmd[0]], text=True)
        loader = re.search(r"/lib64/ld-linux-x86-64.so.2 => (\S+)", deps)
        if loader:
            lean_cmd.insert(0, loader[1])
    _, report["lean_seconds"] = run(lean_cmd, timeout=900, output=lean_out)
    expected = [[int(x) for x in l.split()] for l in lean_out.read_text().splitlines()]
    exe = scratch / "native"
    _, report["native_build_seconds"] = run(["bend", str(native), "-o", str(exe)], timeout=180)
    native_out = scratch / "native.txt"
    _, report["native_seconds"] = run([str(exe), "--threads", "1"], timeout=300, output=native_out)
    actual = bend_list(native_out.read_text())
    findings = []
    check_rows("native-vs-Lean", actual, expected, pairs, OPS+CONVS, findings)
    print(f"Native compared: {len(actual)} rows, {len(findings)} mismatches", flush=True)
    del actual
    checker_rows = []
    checker_seconds = 0
    for offset in range(0, len(subset), args.chunk):
        file = scratch / f"checker-{offset:05}.bend"
        out, elapsed = run(["bend", str(file)], timeout=120)
        (file.with_suffix(".out")).write_text(out)
        checker_rows.extend(bend_list(out))
        checker_seconds += elapsed
        if offset % (args.chunk * 8) == 0:
            print(f"Checker compared through {len(checker_rows)} / {len(subset)}", flush=True)
    check_rows("checker-vs-Lean", checker_rows, [r[:len(OPS)] for r in expected[:len(subset)]],
               subset, OPS, findings)
    report["checker_seconds"] = checker_seconds
    report["mismatches"] = findings
    report["scalar_comparisons_native"] = len(pairs) * len(OPS+CONVS)
    report["scalar_comparisons_checker"] = len(subset) * len(OPS)
    report["outputs_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [lean_out, native_out]}
    (scratch / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
