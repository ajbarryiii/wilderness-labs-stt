"""Spill inspection of emitted aarch64 assembly (plans/cpu-benchmark-mac.md M7).

    python3 asm_spills.py FILE.s [SUBSTRING ...]      (default substring: "micro")

For every function whose (demangled-ish) symbol contains a substring: the number of SDOT instructions,
stack-relative SIMD/FP loads and stores ([sp ...] with a q/d/s/v register operand) in the whole function
and inside the hot region (from the first to the last SDOT), and callee-saved d8-d15 saves. A spill in the
hot region means the register blocking exceeds the 32 vector registers.
"""
from __future__ import annotations

import json
import re
import sys

LABEL = re.compile(r"^([A-Za-z_.$][\w.$]*):")
SDOT = re.compile(r"^\s*sdot\s")
SP_VEC = re.compile(r"^\s*(ldr|str|ldp|stp|ldur|stur)\s+([qdsv]\d+|[qdsv]\d+,\s*[qdsv]\d+).*\[sp")


def functions(lines: list[str]):
    name, body = None, []
    for line in lines:
        m = LABEL.match(line)
        if m and not m.group(1).startswith(("L", ".L", "ltmp", "Ltmp", "lCPI", "LBB", "Lfunc", "Lloh")):
            if name:
                yield name, body
            name, body = m.group(1), []
        elif name:
            body.append(line)
    if name:
        yield name, body


def inspect(path: str, subs: list[str]) -> list[dict]:
    lines = open(path).read().splitlines()
    out = []
    for name, body in functions(lines):
        if not any(s in name for s in subs):
            continue
        ins = [l for l in body if l.startswith("\t") and not l.strip().startswith((".", "//", ";"))]
        sdot_idx = [i for i, l in enumerate(ins) if SDOT.match(l)]
        sp_idx = [i for i, l in enumerate(ins) if SP_VEC.match(l)]
        lo, hi = (sdot_idx[0], sdot_idx[-1]) if sdot_idx else (0, -1)
        hot_spills = [ins[i].strip() for i in sp_idx if lo <= i <= hi]
        out.append({"function": name, "instructions": len(ins), "sdot": len(sdot_idx), "sp_vector_accesses": len(sp_idx),
                    "hot_region_instructions": hi - lo + 1 if sdot_idx else 0, "hot_region_spills": len(hot_spills),
                    "hot_region_spill_examples": hot_spills[:4]})
    return out


if __name__ == "__main__":
    res = inspect(sys.argv[1], sys.argv[2:] or ["micro"])
    print(json.dumps({"functions": len(res), "with_hot_spills": sum(1 for r in res if r["hot_region_spills"]),
                      "detail": res}, indent=1))
