"""Compare surrogate manifests written on two machines (tests/test_randomweights.py) and record the result.

    python tests/compare_manifests.py DIR_A DIR_B --out results/wp1_manifest_crosscheck.json

DIR_A and DIR_B hold seed{0,1,2}.json. For each seed: tensor count, how many per-tensor SHA-256
entries are equal, both digests and the machines' platform/numpy versions. Exit 1 unless every
tensor of every seed is equal. The output holds hashes and versions only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mil import evidence

SEEDS = (0, 1, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {"seeds": {}}
    ok = True
    for seed in SEEDS:
        a, b = (json.loads((d / f"seed{seed}.json").read_text()) for d in (args.a, args.b))
        names = sorted(set(a["tensors"]) | set(b["tensors"]))
        equal = sum(a["tensors"].get(n) == b["tensors"].get(n) for n in names)
        same = equal == len(names) and a["digest"] == b["digest"] and a["weight_stats_sha256"] == b["weight_stats_sha256"]
        ok &= same
        result["seeds"][str(seed)] = {
            "tensors": len(names), "tensors_equal": equal, "identical": same,
            "weight_stats_sha256": a["weight_stats_sha256"],
            "machines": [{"platform": m["machine"]["platform"], "numpy": m["machine"]["numpy"],
                          "python": m["machine"]["python"], "digest": m["digest"]} for m in (a, b)]}
    result["all_identical"] = ok
    evidence.write_text(args.out, json.dumps(result, indent=1) + "\n")
    print(json.dumps({s: (r["tensors_equal"], r["tensors"], r["identical"]) for s, r in result["seeds"].items()}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
