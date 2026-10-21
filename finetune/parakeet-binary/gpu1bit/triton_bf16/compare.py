"""Tabulate a shared/bench.py P-kernel JSON: candidate vs baselines, warm and cold, and the fraction
of the BF16x3 tensor-core floor (3 x M*N*K MACs at 127 T MAC/s: C's register-resident mma.sync
m16n8k16 BF16/FP32-acc probe, 254 TFLOPS, NOTES 2026-10-20 07:49). CPU only.

    finetune/parakeet-binary/python -m gpu1bit.triton_bf16.compare <kernel.json> [candidate]
"""
import json
import sys

BF16_MACS = 254e12 / 2


def main():
    d = json.load(open(sys.argv[1]))
    rows = d["rows"]
    names = list(rows[0]["variants"])
    cand = sys.argv[2] if len(sys.argv) > 2 else next(n for n in names if not n.startswith("r_"))
    print(f"{'op':5s} {'M':>4s} | warm us: " + " ".join(f"{n[:12]:>12s}" for n in names)
          + " | cand/r_bf16x3 cand/best_M1 | cold: cand  best_M1 | floor% | rel_frob")
    totals = {}
    for r in rows:
        v = r["variants"]
        warm = {n: v[n].get("warm_us") for n in names}
        cold = {n: v[n].get("cold_us") for n in names}
        m1 = {n: w for n, w in warm.items() if n.startswith("r_") and n != "r_bf16x3" and w}
        best = min(m1.values()) if m1 else float("nan")
        bestc = min(c for n, c in cold.items() if n.startswith("r_") and n != "r_bf16x3" and c)
        floor = 3 * r["M"] * r["N"] * r["K"] / BF16_MACS * 1e6
        c = warm[cand]
        print(f"{r['op']:5s} {r['M']:4d} | " + " ".join(f"{(warm[n] or float('nan')):12.2f}" for n in names)
              + f" | {c / warm['r_bf16x3']:12.2f} {c / best:11.2f} | {cold[cand]:10.2f} {bestc:8.2f} | "
              f"{100 * floor / c:5.1f} | {v[cand].get('rel_frobenius_vs_fp64', float('nan')):.2e}")
        if r["op"] != "pos":
            mult = 2 if r["op"] in ("ff1", "ff2") else 1
            for n in names:
                totals.setdefault(r["M"], {}).setdefault(n, 0.0)
                totals[r["M"]][n] += mult * (warm[n] or 0)
    print("per Conformer block (qkv+out+2ff1+2ff2+pw1+pw2), warm us:")
    for m, t in totals.items():
        print(f"  M={m:4d} " + " ".join(f"{n}={t[n]:.1f}" for n in names))


if __name__ == "__main__":
    main()
