"""Turn tune.py sweep results into configs.TUNED (CPU only; prints the Python table to paste).

Selection per (N, K, tuned M): lowest L2-cold time among the configs timed cold (the encoder streams
each layer's weights once per forward), ties within 1% broken by hot time.

    CUDA_VISIBLE_DEVICES= gpu1bit/shared/python -m gpu1bit.int8_mma.make_configs <tag-prefix>
"""
from __future__ import annotations

import json
import sys

from gpu1bit.int8_mma.tune import ARTIFACTS


def main():
    prefix = sys.argv[1]
    best = {}
    for d in sorted(ARTIFACTS.glob(f"*-{prefix}*")):
        f = d / "results.json"
        if not f.is_file():
            continue
        for r in json.loads(f.read_text()):
            timed = [x for x in r["rows"] if "cold_us" in x]
            if not timed:
                continue
            fastest = min(x["cold_us"] for x in timed)
            pick = min((x for x in timed if x["cold_us"] <= 1.01 * fastest), key=lambda x: x["hot_us"])
            best[(r["n"], r["k"], r["m"])] = (pick, r["m1_expanded_dot"], d.name)
    print("TUNED_M = {")
    for (n, k, m), (pick, m1, src) in sorted(best.items()):
        print(f"    ({n}, {k}, {m}): DotConfig(*{pick['cfg']}),  # {pick['config']} hot {pick['hot_us']:.2f} "
              f"cold {pick['cold_us']:.2f} us; R-expanded dot {m1['hot_us']:.2f} / {m1['cold_us']:.2f}")
    print("}")


if __name__ == "__main__":
    main()
