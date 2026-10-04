"""Eligibility record and summary of the MLX GPU arm (NixOS: needs the parent experiment's scorer and SentencePiece).

  ./python ios/mlxarm/record.py --results DIR --refcache-index FILE

DIR holds the Mac's results/mlxarm outputs (gate2, 4a, 4b, stress, timing .json, f2_swift.jsonl); FILE is WP3's
reference cache index (refcache/mp2/index.json, with the FP32 reference's free-decoding tokens). Writes:
- ios/results/mlxarm/{gate2,4a,4b,stress,timing}.json (copies), f2_swift.json (Swift F2 on the MLX encoder
  outputs: token sequences vs the gate's F2-equivalent decode) and summary.json (incl. the HD p95 per bucket);
- ios/results/eligibility/mp2-MLX-multi-gpu-mlx.json in mil/eligibility.py's record format (design revision 8),
  so mil.eligibility.check("mp2", "MLX", "multi", "gpu-mlx") answers for it. Fields beyond WP3's: "runtime"
  (MLX version), "encoding" (the 2-bit affine construction) and "deployed_decoder" ("F2").
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mil import evidence

IOS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IOS))
sys.path.insert(0, str(IOS.parent))

OUT = IOS / "results" / "mlxarm"
REC = IOS / "results" / "eligibility" / "mp2-MLX-multi-gpu-mlx.json"
BEST_ANE = {"2": 14.0, "4": 15.4, "8": 19.6, "15": 40.2}  # WP5 C6s8 multi (ANE) encoder typical ms


def sha(p: Path) -> str:
    return hashlib.sha256(evidence.read_bytes(p)).hexdigest()


def main() -> None:
    from armreport import harrell_davis
    from mil.eligibility import WER_POINTS, _tokenizer, _wer, check

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--refcache-index", required=True)
    args = ap.parse_args()
    src = Path(args.results)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("gate2", "4a", "4b", "stress", "timing"):
        evidence.write_text(OUT / f"{name}.json", evidence.read_text(src / f"{name}.json"))
    g = {n: json.loads(evidence.read_text((OUT / f"{n}.json"))) for n in ("gate2", "4a", "4b", "stress", "timing")}
    clips = {c["id"]: c for c in json.loads(evidence.read_text((IOS / "clips.json")))["clips"]}
    ref_tokens = {c: v["fp16s_free_tokens"] for c, v in json.loads(evidence.read_text(Path(args.refcache_index)))["free_tokens"].items()}
    arm_tokens = g["4b"]["free_tokens"]
    swift = {json.loads(l)["clip"]: json.loads(l)["result"]["tokens"] for l in evidence.read_text((src / "f2_swift.jsonl")).splitlines()
             if l.strip() and '"result"' in l}
    f2 = {"what": "Swift parakeet-bench F2 (native FP32 CPU decode loop) free decoding of the MLX arm's own-bucket encoder "
                  "outputs (external-encoder mode), vs the gate's F2-equivalent (reference.py FP32 decoder/joint) tokens",
          "clips": len(arm_tokens), "identical": sum(swift.get(c) == t for c, t in arm_tokens.items()),
          "identical_to_reference": sum(swift.get(c) == ref_tokens[c] for c in arm_tokens)}
    evidence.write_text((OUT / "f2_swift.json"), json.dumps(f2, indent=1) + "\n")
    sp, sp_path = _tokenizer()
    ref_w = _wer({c: ref_tokens[c] for c in arm_tokens}, clips, sp)
    arm_w = _wer(arm_tokens, clips, sp)
    sw_w = _wer({c: swift[c] for c in arm_tokens}, clips, sp)
    timing = g["timing"]
    per = {}
    for b, v in timing["per_bucket"].items():
        calls = [r["ms"] for r in timing["calls"] if str(r["bucket"]) == b]
        per[b] = {**v, "p95_hd_ms": harrell_davis(calls), "best_ane_c6s8_ms": BEST_ANE[b],
                  "ratio_to_best_ane": v["typical_ms"] / BEST_ANE[b]}
    b4 = g["4b"]
    checks = {
        "scope": {"pass": True, "note": "DESIGN.md E: MLX GPU encoder (2-bit affine, exact)"},
        "gate2": {"pass": g["gate2"]["pass"], "modules": g["gate2"]["modules"]},
        "gate4a": {"pass": g["4a"]["pass"], "source": "results/mlxarm/4a.json", "own_fp32_build": True,
                   "rel_max": g["4a"]["max_rel"], "abs_max": g["4a"]["max_abs"]},
        "gate4a_decoder": {"pass": True, "note": "F2 gated in WP4 (results/wp4/f2_gate_mp2.json)"},
        "gate4b": {"pass": b4["encoder"]["pass"] and b4["decisions"]["pass"] and b4["free_decoding"]["pass"],
                   "result_design_revision": b4["design_revision"], "decoder_precision": "fp32 (F2)",
                   "encoder_rel_max": b4["encoder"]["max_rel"],
                   "encoder_failing": b4["encoder"]["runs"] - b4["encoder"]["passed"],
                   "heads": {"f2": {h: {"decisive_fraction": v["decisive_fraction"], "agreement_on_decisive": v["agree_decisive"],
                                        "agreement_all_steps": v["agree_all"]} for h, v in b4["decisions"].items() if h in ("token", "duration")}},
                   "sequence_identity": {"f2": b4["free_decoding"]["identity"]},
                   "rev5_diagnostic_encoder_pass": b4["encoder"]["rev5_passed"] == b4["encoder"]["runs"]},
        "gate5": {"pass": b4["gate5"]["pass"], "applicable": True, "gated_rel_max": b4["gate5"]["max_rel"]},
        "wer": {"pass": arm_w["wer"] <= ref_w["wer"] + WER_POINTS / 100, "reference_wer_pct": round(100 * ref_w["wer"], 3),
                "f2_wer_pct": round(100 * arm_w["wer"], 3), "swift_f2_wer_pct": round(100 * sw_w["wer"], 3),
                "clips": len(arm_tokens), "limit_points": WER_POINTS},
        "stress": {"pass": g["stress"]["pass"], "detail": {k: g["stress"][k] for k in ("nonfinite_where_c4_finite", "nonfinite_at_x1")},
                   "c4_reference": "MLX dense FP16 matmul with C4's effective weights on the same GPU"},
        "f2_swift": {"pass": f2["identical"] == f2["clips"], "identical": f2["identical"], "clips": f2["clips"]},
    }
    reasons = [k for k, v in checks.items() if not v["pass"]]
    inputs = {str(p.relative_to(IOS)): sha(p) for p in sorted(evidence.glob(OUT, "*.json")) if p.name != "summary.json"}
    record = {"design_revision": 8, "code_version": "wp6b-mlxarm-record-1", "model": "mp2", "arm": "MLX", "variant": "multi",
              "backend": "gpu-mlx", "compute_units": "mlx-gpu", "decoder_precision": "fp32", "deployed_decoder": "F2",
              "runtime": {"mlx": b4["environment"]["mlx"], "device": b4["environment"]["device"]},
              "encoding": "q = codes + 1 in {0,1,2} packed 2-bit (LSB first), scales FP16(s), biases -FP16(s) per group of "
                          f"{b4['environment']['group_size']}; mx.quantized_matmul bits 2",
              "eligible": not reasons, "timing_allowed": not reasons, "selection_eligible": False,
              "selection_note": "exploration arm (S3): no Swift/device implementation; not a deployment candidate yet",
              "reasons": reasons, "checks": checks, "inputs": inputs, "tokenizer_sha256": sha(sp_path),
              "built": time.strftime("%Y-%m-%d %H:%M")}
    evidence.write_text(REC, json.dumps(record, indent=1) + "\n")
    summary = {"informational_timing": "Python MLX on the shared Mac (M1 Pro GPU); encoder only, mx.eval + synchronize "
                                       "in the timed region; 3 warm-up + 10 timed per natural clip",
               "timing_per_bucket": per, "peak_memory_mb": timing["peak_memory_mb"],
               "weights_load_s": timing["weights_load_s"],
               "step3_condition": "eligible and typical encoder ms <= 1.5x the best Core ML ANE arm in every bucket",
               "step3_triggered": (not reasons) and all(v["ratio_to_best_ane"] <= 1.5 for v in per.values()),
               "eligibility": {k: record[k] for k in ("eligible", "timing_allowed", "selection_eligible", "reasons")},
               "gates": {k: v["pass"] for k, v in checks.items()}, "wer": checks["wer"], "f2_swift": f2}
    evidence.write_text((OUT / "summary.json"), json.dumps(summary, indent=1) + "\n")
    rec = check("mp2", "MLX", "multi", "gpu-mlx")
    print(json.dumps({"eligibility_check": {k: rec.get(k) for k in ("timing_allowed", "eligible")}, **summary}, indent=1))


if __name__ == "__main__":
    main()
