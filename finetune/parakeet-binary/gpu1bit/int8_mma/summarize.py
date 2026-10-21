"""CPU-only tables from H's harness outputs for NOTES (no GPU).

    gpu1bit/shared/python -m gpu1bit.int8_mma.summarize kernel <kernel.json>
    gpu1bit/shared/python -m gpu1bit.int8_mma.summarize paired <run dir> <scope> <reference> <candidate>
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

BLOCK = {"qkv": 1, "out": 1, "ff1": 2, "ff2": 2, "pw1": 1, "pw2": 1}


def kernel(path):
    d = json.loads(Path(path).read_text())
    rows = d["rows"]
    variants = list(rows[0]["variants"])
    print("op M | warm us: " + " ".join(variants) + " | cold us: " + " ".join(variants))
    for r in rows:
        v = r["variants"]
        print(f"{r['op']} {r['M']} | " + " ".join(f"{v[x]['warm_us']:.2f}" for x in variants) + " | " +
              " ".join(f"{v[x]['cold_us']:.2f}" for x in variants))
    for m in sorted({r["M"] for r in rows}):
        tot = {x: [0.0, 0.0] for x in variants}
        for r in rows:
            if r["M"] == m and r["op"] in BLOCK:
                for x in variants:
                    tot[x][0] += BLOCK[r["op"]] * r["variants"][x]["warm_us"]
                    tot[x][1] += BLOCK[r["op"]] * r["variants"][x]["cold_us"]
        print(f"block M={m}: " + "; ".join(f"{x} {a:.1f}/{b:.1f}" for x, (a, b) in tot.items()))


def paired(run, scope, ref, cand):
    rows = []
    for f in sorted(Path(run).glob(f"paired-{scope}-*.json")):
        rows += json.loads(f.read_text())["rows"]
    lat, en, pw = [], [], []
    lines = []
    for r in rows:
        a, b = r["summary"][ref], r["summary"][cand]
        dl = 100 * (b["median_ms"] / a["median_ms"] - 1)
        de = 100 * (b["joules_per_iteration"] / a["joules_per_iteration"] - 1)
        dp = 100 * (b["average_watts"] / a["average_watts"] - 1)
        lat.append(dl), en.append(de), pw.append(dp)
        ok = r["checks"].get(cand, {})
        flag = ok.get("identical_to_" + ref, ok.get("lengths_equal"))
        lines.append(f"{r['index']:2d} {r['corpus'][:6]:6s} {r['duration']:6.2f}s M={r['M']:3d} | {ref} {a['median_ms']:.3f} ms "
                     f"{a['joules_per_iteration']:.4f} J {a['average_watts']:.0f} W | {cand} {b['median_ms']:.3f} ms "
                     f"{b['joules_per_iteration']:.4f} J {b['average_watts']:.0f} W | {dl:+.2f}% {de:+.2f}% | check {flag}")
    print("\n".join(lines))
    if rows and scope == "encoder":
        c = [r["checks"][cand] for r in rows]
        print(f"encoder checks vs {ref}: lengths equal {sum(x['lengths_equal'] for x in c)}/{len(c)}, finite "
              f"{sum(x['finite'] for x in c)}/{len(c)}, max rel Frobenius (valid frames) "
              f"{max(x['rel_frobenius_valid'] for x in c):.3e}, max abs {max(x['max_abs_valid'] for x in c):.3e}")
    if rows and scope == "full":
        c = [r["checks"][cand] for r in rows]
        same = sum(x.get("identical_to_" + ref, False) for x in c)
        print(f"full checks vs {ref}: text+tokens+timestamps identical {same}/{len(c)}")
    if rows:
        print(f"n={len(rows)} median latency {statistics.median(lat):+.2f}% ({sum(x < 0 for x in lat)}/{len(lat)} lower), "
              f"energy {statistics.median(en):+.2f}% ({sum(x < 0 for x in en)}/{len(en)} lower), "
              f"avg power {statistics.median(pw):+.2f}%")
        for target in sorted({r["target_seconds"] for r in rows}):
            sel = [i for i, r in enumerate(rows) if r["target_seconds"] == target]
            print(f"  target {target}s: latency {statistics.median(lat[i] for i in sel):+.2f}% "
                  f"({sum(lat[i] < 0 for i in sel)}/{len(sel)}), energy {statistics.median(en[i] for i in sel):+.2f}% "
                  f"({sum(en[i] < 0 for i in sel)}/{len(sel)})")


if __name__ == "__main__":
    if sys.argv[1] == "kernel":
        kernel(sys.argv[2])
    else:
        paired(*sys.argv[2:6])
