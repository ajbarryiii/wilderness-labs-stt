"""Gates, stress and informational timing of the MLX GPU encoder arm (macOS; through ios/macguard).

  cd ios && python -m mlxarm.gates gate2              # exact 2-bit affine round trip, all 264 ternary modules
  cd ios && python -m mlxarm.gates 4a                 # FP32 path, every clip in every bucket it fits
  cd ios && python -m mlxarm.gates 4b                 # FP16 on the GPU: encoder, decisions (F2-equivalent), free decoding, gate 5
  cd ios && python -m mlxarm.gates stress             # relative stress rule on the GPU (MLX dense FP16 = the C4 analogue)
  cd ios && python -m mlxarm.gates time               # informational: 3 warm-up + 10 timed per natural clip

Thresholds are WP3's (mil/gates7.py, DESIGN.md revisions 7-8):
- 4a: rel <= 1e-5, abs <= 1e-4 (tau 1e-6) on the valid frames vs the FP32 reference with FP16-rounded scales
  (WP3's reference cache, refcache "fp16s"), finite, encoder_length exact.
- 4b: finite; encoder valid frames rel <= 0.1 (tau 1e-3) on every clip and bucket; encoder_length exact. Decisions
  through the deployed decode path (F2: the native FP32 CPU loop; here its reference equivalent, reference.py's
  FP32 decoder/joint with M_P2's weights, which F2 matches to 1.4e-7, WP4) by forced replay of every trace on the
  clip's own-bucket output: per head (token incl. blank, duration), steps with a reference raw-logit top-1 margin
  >= 1.0 are decisive; pooled agreement >= 99.5% on decisive steps and >= 99% on all, >= 50% decisive. Free
  decoding of the 64 natural clips: token sequences identical to the reference's on >= 95% (WER: record.py).
  Gate 5: every bucket vs the arm's 15 s output, rel <= 0.1 on the boundary, silence and impulse clips.
  The revision 2-5 ceilings (rel <= 2e-2, abs <= 0.25; heads rel <= 2e-2) are reported as diagnostics.
- stress: the relative rule (revision 6): non-finite output at a level where the C4 analogue stays finite on the
  same device fails, and x1 must be finite. C4's Core ML GPU results are WP3's; the analogue here is MLX's dense
  FP16 matmul with C4's effective weights (codes x FP16(s)) on the same GPU.
Results: <artifacts>/results/mlxarm/<gate>.json (copied into ios/results/mlxarm/ on NixOS); the arm's own-bucket
encoder outputs (valid frames, time-major float32) go to <artifacts>/mlxarm/enc/<id>.f32 for the Swift F2 check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "mlxarm"

import numpy as np

from . import IOS  # noqa: E402
from .encoder import BUCKETS, GROUP, Encoder, Weights, bucket_of, encoder_frames, unpack_q  # noqa: E402

DESIGN_REVISION = 8
CODE_VERSION = "wp6b-mlxarm-1"
G4A = {"rel": 1e-5, "abs": 1e-4, "tau": 1e-6}
G4B = {"encoder_rel": 0.1, "tau": 1e-3, "decisive_margin": 1.0, "agree_decisive": 0.995, "agree_all": 0.99,
       "decisive_min": 0.5, "sequence_identity": 0.95}
REV5 = {"encoder_rel": 2e-2, "encoder_abs": 0.25, "heads_rel": 2e-2}
GATE5_KINDS = ("boundary", "silence", "impulse")
N_DUR = 5


def art() -> Path:
    import artifacts

    return artifacts.root()


def errors(a, r, tau):
    from mil.refcache import errors as e

    return e(a, r, tau)


def environment() -> dict:
    import mlx.core as mx

    return {"mlx": mx.__version__, "device": str(mx.default_device()), "platform": platform.platform(),
            "python": platform.python_version(), "group_size": GROUP, "bits": 2,
            "encoder_py_sha256": hashlib.sha256((IOS / "mlxarm" / "encoder.py").read_bytes()).hexdigest(),
            "gates_py_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def write(name: str, doc: dict) -> None:
    from mil.gates7 import jsonable

    out = art() / "results" / "mlxarm"
    out.mkdir(parents=True, exist_ok=True)
    doc = {"design_revision": DESIGN_REVISION, "code_version": CODE_VERSION, "arm": "MLX", "model": "mp2",
           "variant": "multi", "backend": "gpu-mlx", "date": time.strftime("%Y-%m-%d %H:%M"),
           "environment": environment(), **doc}
    (out / f"{name}.json").write_text(json.dumps(jsonable(doc), indent=1, allow_nan=False) + "\n")


def manifest() -> list[dict]:
    import clips as clipmod

    return clipmod.load_manifest()["clips"]


def refcache_index() -> dict:
    from mil import refcache

    return refcache.validate("mp2")


def clip_ref(cid: str) -> dict:
    from mil import refcache

    return refcache.load_clip("mp2", cid)


# --- gate 2 ---------------------------------------------------------------------------------------------------

def gate2(args) -> int:
    import mlx.core as mx

    from mil.weights import effective_fp16, fp16_scale

    W = Weights("mp2", "fp16")
    rows, ok = {}, True
    for key, qt in W.packed.items():
        codes, scale = W.source.ternary(f"encoder.{key}")
        codes = codes.reshape(codes.shape[0], -1)
        s16 = fp16_scale(scale)
        q = unpack_q(qt["wq"], codes.shape[1])
        groups = codes.shape[1] // GROUP
        checks = {
            "unpacked_q_minus_1_equals_codes": bool(np.array_equal(q.astype(np.int16) - 1, codes.astype(np.int16))),
            "scales_equal_fp16_s": bool(np.array_equal(qt["scales"].view(np.uint16), np.repeat(s16[:, None], groups, 1).view(np.uint16))),
            "biases_equal_minus_fp16_s": bool(np.array_equal(qt["biases"].view(np.uint16), np.repeat((-s16)[:, None], groups, 1).view(np.uint16))),
        }
        deq = np.array(mx.dequantize(mx.array(qt["wq"]), mx.array(qt["scales"]), mx.array(qt["biases"]),
                                     group_size=GROUP, bits=2))
        checks["mx_dequantize_fp16_equals_codes_x_fp16_s_bitwise"] = bool(
            np.array_equal(deq.astype(np.float16).view(np.uint16), effective_fp16(codes, scale).view(np.uint16))
            and deq.dtype == np.float16)
        deq32 = np.array(mx.dequantize(mx.array(qt["wq"]), mx.array(qt["scales"].astype(np.float32)),
                                       mx.array(qt["biases"].astype(np.float32)), group_size=GROUP, bits=2))
        checks["mx_dequantize_fp32_equals_codes_x_fp16_s_bitwise"] = bool(
            np.array_equal(deq32, codes.astype(np.float32) * s16.astype(np.float32)[:, None]))
        rows[key] = checks
        ok &= all(checks.values())
    write("gate2", {"gate": 2, "pass": ok, "modules": len(rows),
                    "construction": "q = codes + 1 packed 16 per uint32 (LSB first); scales = FP16(s), biases = "
                                    f"-FP16(s), repeated over every group of {GROUP}; mx.dequantize = q s - s",
                    "failures": {k: v for k, v in rows.items() if not all(v.values())}, "per_module": rows})
    print(json.dumps({"gate2_pass": ok, "modules": len(rows)}))
    return 0 if ok else 10


# --- decode (F2-equivalent FP32) --------------------------------------------------------------------------------

def decoder_model():
    import types

    import torch

    import reference
    from mil.weights import Source

    torch.set_grad_enabled(False)
    cfg = reference.Config()
    dec, joint = reference.Decoder(cfg), reference.Joint(cfg)
    src = Source("mp2")
    for mod, prefix in ((dec, "decoder."), (joint, "joint.")):
        mod.load_state_dict({k: torch.from_numpy(src.floating(prefix + k)) for k in mod.state_dict()})
    return types.SimpleNamespace(cfg=cfg, decoder=dec.eval(), joint=joint.eval())


def heads_margin(logits: np.ndarray):
    tok, dur = logits[:, :-N_DUR], logits[:, -N_DUR:]
    def margin(x):
        s = np.sort(x, axis=1)
        return s[:, -1] - s[:, -2]
    return (tok.argmax(1), margin(tok)), (dur.argmax(1), margin(dur))


# --- 4a / 4b ----------------------------------------------------------------------------------------------------

def run_buckets(enc: Encoder, clips: list[dict], tau: float, ceiling: dict, keep_own: bool):
    """Every clip in every bucket it fits: errors of the valid frames vs the reference, lengths, finiteness."""
    rows, outputs = [], {}
    for clip in clips:
        ref = clip_ref(clip["id"])
        feats, m = ref["features"], int(ref["mel_length"])
        e = clip["encoder_frames"]
        for b in [b for b in BUCKETS if clip["length"] <= 16000 * b]:
            out, length = enc(enc.mel_input(feats, b), m, b)
            a = np.array(out)[:, :e]
            rel, ab = errors(a, ref["fp16s_enc"], tau)
            row = {"clip": clip["id"], "kind": clip["kind"], "bucket": b, "rel": rel, "abs": ab,
                   "length_ok": length == e == encoder_frames(m), "finite": bool(np.isfinite(np.array(out)).all())}
            if ceiling.get("abs") is not None:
                row["pass"] = rel <= ceiling["rel"] and ab <= ceiling["abs"] and row["length_ok"] and row["finite"]
            else:
                row["pass"] = rel <= ceiling["rel"] and row["length_ok"] and row["finite"]
                row["rev5_pass"] = rel <= REV5["encoder_rel"] and ab <= REV5["encoder_abs"]
            rows.append(row)
            if keep_own and b == bucket_of(clip["length"]):
                outputs[clip["id"]] = a
            if clip["kind"] in GATE5_KINDS:
                outputs[(b, clip["id"])] = a
    return rows, outputs


def gate4a(args) -> int:
    index = refcache_index()
    W = Weights("mp2", "fp32")
    enc = Encoder(W)
    rows, _ = run_buckets(enc, manifest(), G4A["tau"], G4A, keep_own=False)
    ok = all(r["pass"] for r in rows)
    write("4a", {"gate": "4a", "pass": ok, "precision": "fp32 (FP32 activations; scales/biases = FP16(s) in FP32; "
                 "same packed 2-bit weights and quantized_matmul kernel)", "ceilings": G4A,
                 "reference": {"cache": "refcache mp2 fp16s", "provenance": index["provenance"]},
                 "runs": len(rows), "passed": sum(r["pass"] for r in rows),
                 "max_rel": max(r["rel"] for r in rows), "max_abs": max(r["abs"] for r in rows), "rows": rows})
    print(json.dumps({"4a_pass": ok, "runs": len(rows), "max_rel": max(r["rel"] for r in rows),
                      "max_abs": max(r["abs"] for r in rows)}))
    return 0 if ok else 10


def gate4b(args) -> int:
    import torch

    import reference
    import traces as tracemod

    index = refcache_index()
    W = Weights("mp2", "fp16")
    enc = Encoder(W)
    clips = manifest()
    rows, outputs = run_buckets(enc, clips, G4B["tau"], {"rel": G4B["encoder_rel"]}, keep_own=True)
    dec = decoder_model()
    traces = {t["id"]: t for t in json.loads((IOS / "traces.json").read_text())["clips"]}
    enc_dir = art() / "mlxarm" / "enc"
    enc_dir.mkdir(parents=True, exist_ok=True)
    tok_all = tok_dec = tok_dec_ok = dur_all = dur_dec = dur_dec_ok = tok_ok = dur_ok = 0
    head_rows, free_rows = [], []
    for clip in clips:
        a = outputs[clip["id"]]                                            # [1024, E] own bucket
        a.T.astype("<f4").tofile(enc_dir / f"{clip['id']}.f32")
        ref = clip_ref(clip["id"])
        e = a.shape[1]
        x = torch.from_numpy(a.astype(np.float32))[None]
        trace = tracemod.record_to_trace(traces[clip["id"]])
        steps = reference.replay(dec, x, e, trace)
        logits = steps.logits.numpy()
        (ta, _), (da, _) = heads_margin(logits)
        (tr, tm), (dr, dm) = heads_margin(ref["fp16s_logits"])
        tdec, ddec = tm >= G4B["decisive_margin"], dm >= G4B["decisive_margin"]
        tok_all += len(tr); dur_all += len(dr)
        tok_ok += int((ta == tr).sum()); dur_ok += int((da == dr).sum())
        tok_dec += int(tdec.sum()); dur_dec += int(ddec.sum())
        tok_dec_ok += int((ta == tr)[tdec].sum()); dur_dec_ok += int((da == dr)[ddec].sum())
        nd = N_DUR
        head_rows.append({"clip": clip["id"], "token_logits_rel": errors(logits[:, :-nd], ref["fp16s_logits"][:, :-nd], 1e-3)[0],
                          "duration_logits_rel": errors(logits[:, -nd:], ref["fp16s_logits"][:, -nd:], 1e-3)[0],
                          "h_rel": errors(steps.h.numpy(), ref["fp16s_h"], 1e-3)[0],
                          "c_rel": errors(steps.c.numpy(), ref["fp16s_c"], 1e-3)[0]})
        if clip["kind"] == "natural":
            free = reference.greedy_decode(dec, x, torch.tensor([e]))[0].tokens
            want = index["free_tokens"][clip["id"]]["fp16s_free_tokens"]
            free_rows.append({"clip": clip["id"], "tokens": free, "equal": free == want})
    decisions = {
        "token": {"steps": tok_all, "agree_all": tok_ok / tok_all, "decisive_fraction": tok_dec / tok_all,
                  "agree_decisive": tok_dec_ok / max(tok_dec, 1)},
        "duration": {"steps": dur_all, "agree_all": dur_ok / dur_all, "decisive_fraction": dur_dec / dur_all,
                     "agree_decisive": dur_dec_ok / max(dur_dec, 1)}}
    decisions_pass = all(h["agree_decisive"] >= G4B["agree_decisive"] and h["agree_all"] >= G4B["agree_all"]
                         and h["decisive_fraction"] >= G4B["decisive_min"] for h in decisions.values())
    identity = sum(r["equal"] for r in free_rows) / len(free_rows)
    gate5 = []
    for clip in [c for c in clips if c["kind"] in GATE5_KINDS]:
        full = outputs[(15, clip["id"])]
        for b in [b for b in BUCKETS if clip["length"] <= 16000 * b and b != 15]:
            rel, ab = errors(outputs[(b, clip["id"])], full, G4B["tau"])
            gate5.append({"clip": clip["id"], "bucket": b, "rel": rel, "abs": ab, "pass": rel <= G4B["encoder_rel"],
                          "rev5_pass": rel <= REV5["encoder_rel"] and ab <= REV5["encoder_abs"]})
    enc_pass = all(r["pass"] for r in rows)
    ok = enc_pass and decisions_pass and identity >= G4B["sequence_identity"] and all(g["pass"] for g in gate5)
    write("4b", {
        "gate": "4b", "pass": ok, "precision": "fp16 (FP16 activations and scales), MLX GPU",
        "deployed_decoder": "F2 (native FP32 CPU loop); evaluated with reference.py's FP32 decoder/joint and M_P2's "
                            "weights (F2 = reference within 1.4e-7, WP4); results/mlxarm/f2_swift.json checks F2 itself",
        "thresholds": G4B, "rev5_diagnostic_ceilings": REV5,
        "reference": {"cache": "refcache mp2 fp16s", "provenance": index["provenance"]},
        "encoder": {"pass": enc_pass, "runs": len(rows), "passed": sum(r["pass"] for r in rows),
                    "max_rel": max(r["rel"] for r in rows), "max_abs": max(r["abs"] for r in rows),
                    "rev5_passed": sum(r["rev5_pass"] for r in rows)},
        "decisions": {"pass": decisions_pass, **decisions},
        "free_decoding": {"clips": len(free_rows), "identical": sum(r["equal"] for r in free_rows),
                          "identity": identity, "pass": identity >= G4B["sequence_identity"]},
        "gate5": {"pass": all(g["pass"] for g in gate5), "runs": len(gate5),
                  "max_rel": max(g["rel"] for g in gate5), "rev5_passed": sum(g["rev5_pass"] for g in gate5), "rows": gate5},
        "heads_diagnostic": {k: max(r[k] for r in head_rows) for k in ("token_logits_rel", "duration_logits_rel", "h_rel", "c_rel")},
        "free_tokens": {r["clip"]: r["tokens"] for r in free_rows},
        "encoder_rows": rows, "head_rows": head_rows})
    print(json.dumps({"4b_pass": ok, "encoder_max_rel": max(r["rel"] for r in rows), "decisions": decisions,
                      "free_identity": identity, "gate5_max_rel": max(g["rel"] for g in gate5)}))
    return 0 if ok else 10


# --- stress ---------------------------------------------------------------------------------------------------------

def stress(args) -> int:
    import mlx.core as mx

    from mil import refcache
    from mil.probes import STRESS_MODULES, adversarial
    from mil.weights import Source, fp16_scale

    from .encoder import quantized

    src = Source("mp2")
    probe = np.load(refcache.default_out("mp2") / "probe_inputs.npz")
    results, fails, x1_bad = {}, [], []
    for module, site in STRESS_MODULES:
        codes, scale = src.ternary(f"encoder.{module}")
        codes = adversarial(codes)
        layer = module.split(".")[1]
        x0 = probe[f"layers_{layer}_{site}"].astype(np.float32)
        qt = quantized(codes, scale, np.float16)
        w16 = (codes.astype(np.float16) * fp16_scale(scale)[:, None])
        w64 = codes.astype(np.float64) * fp16_scale(scale).astype(np.float64)[:, None]
        wq, sc, bi, wd = mx.array(qt["wq"]), mx.array(qt["scales"]), mx.array(qt["biases"]), mx.array(w16)
        for level, x in {"x1": x0, "x8": 8 * x0, "x64": 64 * x0, "abs_x64": 64 * np.abs(x0)}.items():
            xm = mx.array(x.astype(np.float16))
            arm = np.array(mx.quantized_matmul(xm, wq, sc, bi, transpose=True, group_size=GROUP, bits=2)).astype(np.float64)
            c4 = np.array(xm @ wd.T).astype(np.float64)
            ref = x.astype(np.float64) @ w64.T
            row = {"arm_finite": bool(np.isfinite(arm).all()), "c4_analogue_finite": bool(np.isfinite(c4).all()),
                   "arm_max_abs": float(np.nanmax(np.abs(arm))) if np.isfinite(arm).any() else None,
                   "c4_max_abs": float(np.nanmax(np.abs(c4))) if np.isfinite(c4).any() else None,
                   "reference_max_abs": float(np.abs(ref).max()), "input_finite_fp16": bool(np.isfinite(x.astype(np.float16)).all())}
            if row["arm_finite"]:
                row["arm_rel_err"], row["arm_abs_err"] = errors(arm, ref, 1e-3)
            results.setdefault(module, {})[level] = row
            if level == "x1" and not row["arm_finite"]:
                x1_bad.append(module)
            if not row["arm_finite"] and row["c4_analogue_finite"]:
                fails.append((module, level))
    ok = not fails and not x1_bad
    write("stress", {"probe": "stress (revision 6/7 relative rule) on the MLX GPU", "pass": ok,
                     "rule": "fail on a non-finite output where the C4 analogue (MLX dense FP16 matmul with codes x "
                             "FP16(s), same GPU) stays finite; x1 must be finite",
                     "modules": [m for m, _ in STRESS_MODULES], "nonfinite_where_c4_finite": fails,
                     "nonfinite_at_x1": x1_bad, "results": results})
    print(json.dumps({"stress_pass": ok, "fails": fails, "x1_bad": x1_bad}))
    return 0 if ok else 10


# --- informational timing --------------------------------------------------------------------------------------------

def timing(args) -> int:
    import mlx.core as mx

    W = Weights("mp2", "fp16")
    enc = Encoder(W)
    clips = [c for c in manifest() if c["kind"] == "natural"]
    reset = getattr(mx, "reset_peak_memory", None) or mx.metal.reset_peak_memory
    peak = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
    active = getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
    reset()
    rows = []
    first = {}
    for clip in clips:
        ref = clip_ref(clip["id"])
        b = bucket_of(clip["length"])
        mel = enc.mel_input(ref["features"], b)
        mx.eval(mel)
        for rep in range(args.warmups + args.timed):
            t0 = time.perf_counter()
            out, length = enc(mel, int(ref["mel_length"]), b)          # mx.eval inside
            mx.synchronize()
            ms = 1000 * (time.perf_counter() - t0)
            if b not in first:
                first[b] = ms
            if rep >= args.warmups:
                rows.append({"clip": clip["id"], "bucket": b, "rep": rep, "ms": ms})
    from armreport import harrell_davis

    per = {}
    for b in BUCKETS:
        rs = [r for r in rows if r["bucket"] == b]
        meds = {}
        for r in rs:
            meds.setdefault(r["clip"], []).append(r["ms"])
        per[str(b)] = {"clips": len(meds), "typical_ms": statistics.median(statistics.median(v) for v in meds.values()),
                       "p95_hd_ms": harrell_davis([r["ms"] for r in rs]), "first_call_ms_bucket": first.get(b)}
    doc = {"informational": "Python, shared Mac; encoder only (mel in GPU memory -> encoder output evaluated), "
                            "mx.eval + mx.synchronize inside the timed region; front end and decoding excluded",
           "protocol": {"warmups": args.warmups, "timed": args.timed, "clips": len(clips)},
           "weights_load_s": W.load_seconds, "per_bucket": per,
           "peak_memory_mb": peak() / 2 ** 20, "active_memory_mb": active() / 2 ** 20, "calls": rows}
    write("timing", doc)
    print(json.dumps({"per_bucket": per, "peak_memory_mb": doc["peak_memory_mb"]}))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in (("gate2", gate2), ("4a", gate4a), ("4b", gate4b), ("stress", stress)):
        sub.add_parser(name).set_defaults(func=fn)
    p = sub.add_parser("time"); p.add_argument("--warmups", type=int, default=3); p.add_argument("--timed", type=int, default=10)
    p.set_defaults(func=timing)
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
