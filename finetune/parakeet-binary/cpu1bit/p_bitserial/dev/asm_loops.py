"""Hot-loop census of P's emitted aarch64 assembly (cargo rustc --release -p p_bitserial --lib --
--emit asm). For every loop (a backward branch to a label of the same function) that contains
CNT, it counts the instruction classes per iteration and every stack access ([sp ...]), the
signature of a register spill.

    finetune/parakeet-binary/python cpu1bit/p_bitserial/dev/asm_loops.py FILE.s
"""
from __future__ import annotations

import re
import sys
from collections import Counter

FUNC = re.compile(r"^(_[\w$.]+):")
LABEL = re.compile(r"^(L[\w$.]+):")
BRANCH = re.compile(r"^\s+(b(?:\.\w+)?|cbn?z|tbn?z)\s+(?:[^,]+,\s*)*(L[\w$.]+)\s*$")


def classify(ins: str) -> str | None:
    full = ins.split()[0]
    op = full.split(".")[0]
    if op in ("udot", "sdot", "uadalp", "cnt", "addp", "umlal", "umlal2", "umull", "umull2", "ld1r", "ld4", "fmla"):
        return op
    if op == "eor" and (".16b" in full or ".16b" in ins):
        return "eor.16b"
    if op in ("ldr", "ldp", "ldur") and re.search(r"\bq\d+", ins):
        return "ldq"
    if op in ("str", "stp", "stur") and re.search(r"\bq\d+", ins):
        return "stq"
    return None


def main(path: str) -> None:
    lines = open(path).read().splitlines()
    funcs: list[tuple[str, int, int]] = []
    start, name = None, None
    for i, line in enumerate(lines):
        m = FUNC.match(line)
        if m:
            if name is not None:
                funcs.append((name, start, i))
            name, start = m.group(1), i
    if name is not None:
        funcs.append((name, start, len(lines)))
    for fname, a, b in funcs:
        labels = {}
        for i in range(a, b):
            m = LABEL.match(lines[i])
            if m:
                labels[m.group(1)] = i
        loops = []
        for i in range(a, b):
            m = BRANCH.match(lines[i])
            if m and m.group(2) in labels and labels[m.group(2)] < i:
                loops.append((labels[m.group(2)], i, m.group(2)))
        # Innermost loops only: no other backward branch strictly inside.
        for lo, i, target in loops:
            if any(lo <= l2 and i2 < i and (l2, i2) != (lo, i) for l2, i2, _ in loops):
                continue
            body = [l.strip() for l in lines[lo + 1:i + 1] if l.startswith("\t") and not l.strip().startswith((".", ";", "//"))]
            counts = Counter(c for c in map(classify, body) if c)
            if not counts.get("cnt"):
                continue
            stack = sum(1 for l in body if "[sp" in l)
            inst = re.sub(r"^.*3run", "run", fname)
            print(f"{inst:48s} loop {target:>10s} ins={len(body):4d} stack={stack:3d} " +
                  " ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main(sys.argv[1])
