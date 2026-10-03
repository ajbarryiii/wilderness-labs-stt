"""Correctness gates of DESIGN.md revision 7: 4a (FP32 graph), 4b (FP16 execution per backend), 5 (buckets).

  cd ios && python -m mil.gates7 4a --model mp2 --arm C4 --variant multi          # <arm>-fp32 build, CPU_ONLY
  cd ios && python -m mil.gates7 4a-decoder --model mp2                            # decoder-fp32 builds, CPU_ONLY
  cd ios && python -m mil.gates7 4b --model mp2 --arm C4 --variant multi --units cpuAndNeuralEngine

Exit status: 0 every condition the command can evaluate passed; 10 a gate ran and failed (the result file is
written either way); anything else is an error. Every result records the design revision and gate-code version.
Results: ios/results/gates/v7/<model>-<label>-<variant>-<units>.json. Eligibility (mil/eligibility.py) is
derived from these, gate 2 and the stress probe; the WER part of 4b is evaluated there (it needs the
SentencePiece tokenizer and the parent experiment's scorer).

4a (implementation correctness): the arm's graph built with FP32 compute (mil.build --precision fp32: the
same constexpr chain with FP32 LUT/scale values equal to the FP16-rounded scales), run with CPU_ONLY at full
depth on every clip in every bucket it fits; valid frames vs the FP32 reference (FP16-rounded scales):
rel <= 1e-5 and abs <= 1e-4 (tau 1e-6), finite, encoder_length exact. 4a-decoder: the decoder/joint models
built in FP32, forced replay of every trace with the reference encoder output: JointLogits logits and Decoder
h/c within the same ceilings, JointDecision and DecoderJoint decisions equal to the reference argmax on every
step, and free decoding through both deployed paths token-identical to the reference on every clip.

4b (FP16 execution on one backend), the arm's FP16 build:
- every output finite; encoder valid frames rel <= 0.1 vs the reference on every clip and bucket; encoder_length
  exact. The revision 2-5 ceilings (rel <= 2e-2 and abs <= 0.25; heads rel <= 2e-2) are reported as
  diagnostics.
- decisions through the deployed paths: forced replay of traces.json (all 82 clips, encoder output of the
  clip's own bucket; fixed: 15 s) with (a) Decoder + JointDecision (C0's contract) and (b) DecoderJoint. Per
  head (token incl. blank, duration): a step is decisive when the FP32 reference's raw-logit top-1 margin is
  >= 1.0; pooled over clips, agreement >= 99.5% on decisive steps and >= 99% on all steps, with >= 50% of
  steps decisive. JointLogits is run alongside for the reported logit errors (diagnostic).
- free decoding through both deployed paths on the 64 natural clips (own bucket): token sequence identical to
  the FP32 reference's on >= 95% of clips (evaluated here); WER within +0.2 points of the reference
  (evaluated in eligibility.py from the recorded token sequences).
- gate 5 (multifunction, enumerated): valid frames of each bucket vs the same arm's 15 s output within the
  4b encoder ceiling (rel <= 0.1) on the boundary, silence and impulse clips; the revision 2-5 ceilings are
  reported.
F2 (WP4's native CPU decode loop) is not exercised here; it is WP4's deployed path and is gated there.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "mil"

import numpy as np

from . import IOS, refcache  # noqa: E402
from .gates import BUCKETS, GATE5_KINDS, arms_root, clips_manifest, padded, purge_cache, traces_by_id, units_of

DESIGN_REVISION = 7
GATE_CODE_VERSION = "wp3-gates7-1"
OUT = IOS / "results" / "gates" / "v7"
EXIT_FAIL = 10
TAU16, TAU32 = 1e-3, 1e-6
G4A = {"rel": 1e-5, "abs": 1e-4, "tau": TAU32}
G4B = {"encoder_rel": 0.1, "tau": TAU16, "decisive_margin": 1.0, "agree_decisive": 0.995, "agree_all": 0.99,
       "decisive_min": 0.5, "sequence_identity": 0.95, "wer_points": 0.2}
REV5 = {"encoder_rel": 2e-2, "encoder_abs": 0.25, "heads_rel": 2e-2}
BLANK, N_DUR, H = 1024, 5, 640
DURATIONS = (0, 1, 2, 3, 4)
MAX_SYMBOLS = 10
BACKEND = {"cpuAndNeuralEngine": "ane", "cpuOnly": "cpu"}
TOPOLOGY = {"C1": "dense", "C3": "dense", "C4": "dense", "C6s2": "dense", "C6s4": "dense", "C6s8": "dense",
            "C6d4": "dense", "C6d8": "dense", "C7": "post_scale", "C8": "planes"}


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def header(kind: str, **kw) -> dict:
    return {"design_revision": DESIGN_REVISION, "gate_code_version": GATE_CODE_VERSION, "gate": kind,
            "gates7_py_sha256": sha(Path(__file__)), "date": time.strftime("%Y-%m-%d %H:%M"), **kw}


def jsonable(x):
    """Strict-JSON form: non-finite floats as the strings "nan", "inf", "-inf"; numpy scalars as Python."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if np.isfinite(x) else ("nan" if np.isnan(x) else ("inf" if x > 0 else "-inf"))
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def write(doc: dict, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(jsonable(doc), indent=0, allow_nan=False) + "\n")
    return path


def finish(doc: dict, name: str) -> int:
    write(doc, name)
    print(json.dumps(jsonable({"result": name, "pass": doc["pass"], "summary": doc.get("summary")}), default=str)[:3000], flush=True)
    return 0 if doc["pass"] else EXIT_FAIL


def build_manifest(model: str, label: str, variant: str) -> dict:
    return json.loads((arms_root() / model / label / f"manifest-{variant}.json").read_text())


def bucket_runs(model: str, label: str, variant: str, units: str, clips: list[dict]):
    """Yield (clip, bucket, encoder [1024, T], encoder_length) for every clip in every bucket it fits."""
    import coremltools as ct

    path = arms_root() / model / label / f"{variant}.mlmodelc"
    cu = units_of(units)
    buckets = [15] if variant == "fixed" else [15, 8, 4, 2]
    m = None
    for b in buckets:
        fn = f"b{b}" if variant == "multi" else None
        if m is None or variant == "multi":
            if m is not None:
                del m
                purge_cache()
            t0 = time.time()
            m = ct.models.CompiledMLModel(str(path), compute_units=cu, function_name=fn)
            yield None, b, None, round(time.time() - t0, 2)  # load-time marker
        for clip in clips:
            if clip["bucket"] > b:
                continue
            ref = refcache.load_clip(model, clip["id"])
            pred = m.predict({"mel": padded(ref["features"], BUCKETS[b]),
                              "mel_length": np.array([ref["mel_length"]], dtype=np.int32)})
            yield clip, b, np.asarray(pred["encoder"], dtype=np.float32)[0], int(np.asarray(pred["encoder_length"]).reshape(-1)[0])
    del m
    purge_cache()


# --- 4a ------------------------------------------------------------------------------------------------

def gate4a(args) -> int:
    t0 = time.time()
    label = f"{args.arm}-fp32"
    man = build_manifest(args.model, label, args.variant)
    if man.get("precision") != "fp32":
        raise ValueError(f"{label}/{args.variant} is not an FP32 build")
    refcache.validate(args.model, man["provenance"])
    clips = clips_manifest()
    rows, loads = {}, {}
    for clip, b, enc, length in bucket_runs(args.model, label, args.variant, "cpuOnly", clips):
        if clip is None:
            loads[f"b{b}"] = length
            continue
        ref = refcache.load_clip(args.model, clip["id"])["fp16s_enc"]
        e = clip["encoder_frames"]
        valid = enc[:, :e]
        rel, ab = refcache.errors(valid, ref, TAU32)
        ok = bool(np.isfinite(valid).all()) and length == e and rel <= G4A["rel"] and ab <= G4A["abs"]
        rows[f"{clip['id']}@{b}"] = {"bucket": b, "kind": clip["kind"], "rel": rel, "abs": ab, "length_ok": length == e,
                                     "finite": bool(np.isfinite(valid).all()), "pass": ok}
    fails = [k for k, r in rows.items() if not r["pass"]]
    doc = header("4a", model=args.model, arm=args.arm, label=label, variant=args.variant, units="cpuOnly",
                 topology=TOPOLOGY.get(args.arm), thresholds=G4A, build_provenance=man["provenance"],
                 build_chain=man["encoding"].get("chain"))
    doc["pass"] = not fails and len(rows) > 0
    doc["summary"] = {"cases": len(rows), "failing_cases": len(fails), "failing_examples": fails[:20],
                      "rel_max": max(r["rel"] for r in rows.values()), "abs_max": max(r["abs"] for r in rows.values()),
                      "rel_median": float(np.median([r["rel"] for r in rows.values()])),
                      "buckets": sorted({r["bucket"] for r in rows.values()})}
    doc.update(cases=rows, load_s=loads, seconds=round(time.time() - t0, 1))
    return finish(doc, f"{args.model}-{label}-{args.variant}-cpuOnly-4a.json")


# --- decoder paths ------------------------------------------------------------------------------------------

class Paths:
    """The deployed decoder models of one model/precision, loaded for one compute-units setting."""

    def __init__(self, model: str, units: str, precision: str = "fp16") -> None:
        import coremltools as ct

        base = arms_root() / model / ("decoder" if precision == "fp16" else f"decoder-{precision}")
        cu = units_of(units)
        self.manifest = json.loads((base / "manifest.json").read_text())
        load = lambda n: ct.models.CompiledMLModel(str(base / f"{n}.mlmodelc"), compute_units=cu)
        self.dec, self.jd, self.jl, self.dj = (load(n) for n in ("Decoder", "JointDecision", "JointLogits", "DecoderJoint"))

    def decoder(self, token: int, state):
        out = self.dec.predict({"targets": np.array([[token]], dtype=np.int32), "target_length": np.array([1], dtype=np.int32),
                                "h_in": state[0], "c_in": state[1]})
        return (np.asarray(out["decoder"], dtype=np.float32).reshape(1, H, 1),
                (np.asarray(out["h_out"], dtype=np.float32), np.asarray(out["c_out"], dtype=np.float32)))

    def joint_decision(self, f, g):
        out = self.jd.predict({"encoder_step": f, "decoder_step": g})
        return (int(np.asarray(out["token_id"]).reshape(-1)[0]), int(np.asarray(out["duration"]).reshape(-1)[0]),
                float(np.asarray(out["token_prob"]).reshape(-1)[0]))

    def joint_logits(self, f, g):
        out = self.jl.predict({"encoder_step": f, "decoder_step": g})
        return np.concatenate([np.asarray(out["token_logits"], dtype=np.float32).reshape(-1),
                               np.asarray(out["duration_logits"], dtype=np.float32).reshape(-1)])

    def decoder_joint(self, token: int, state, f):
        out = self.dj.predict({"targets": np.array([[token]], dtype=np.int32), "h_in": state[0], "c_in": state[1],
                               "encoder_step": f})
        return (int(np.asarray(out["token_id"]).reshape(-1)[0]), int(np.asarray(out["duration"]).reshape(-1)[0]),
                (np.asarray(out["h_out"], dtype=np.float32), np.asarray(out["c_out"], dtype=np.float32)),
                float(np.asarray(out["token_prob"]).reshape(-1)[0]))


def zero_state():
    return (np.zeros((2, 1, H), np.float32), np.zeros((2, 1, H), np.float32))


def frame(enc: np.ndarray, t: int) -> np.ndarray:
    return np.ascontiguousarray(enc[:, t].reshape(1, -1, 1), dtype=np.float32)


def replay(paths: Paths, enc: np.ndarray, rec: dict) -> dict:
    """Forced replay of one trace through (a) Decoder + JointDecision (+ JointLogits) and (b) DecoderJoint."""
    steps = rec["steps"]
    a_tok, a_dur, b_tok, b_dur = (np.zeros(steps, np.int64) for _ in range(4))
    logits = np.zeros((steps, BLANK + 1 + N_DUR), np.float32)
    hs, cs = np.zeros((steps, 2, H), np.float32), np.zeros((steps, 2, H), np.float32)
    finite = True
    g, state = paths.decoder(BLANK, zero_state())
    last, s_prev = BLANK, zero_state()
    for i in range(steps):
        f = frame(enc, rec["frame"][i])
        a_tok[i], a_dur[i], prob = paths.joint_decision(f, g)
        logits[i] = paths.joint_logits(f, g)
        hs[i], cs[i] = state[0][:, 0], state[1][:, 0]
        b_tok[i], b_dur[i], s_out, prob_b = paths.decoder_joint(last, s_prev, f)
        finite &= bool(np.isfinite(prob) and np.isfinite(prob_b) and np.isfinite(g).all())
        if rec["emitted"][i]:
            tok = rec["token"][i]
            g, state = paths.decoder(tok, state)
            last, s_prev = tok, s_out
    finite &= bool(np.isfinite(logits).all() and np.isfinite(hs).all() and np.isfinite(cs).all())
    return {"jd": (a_tok, a_dur), "dj": (b_tok, b_dur), "logits": logits, "h": hs, "c": cs, "finite": finite}


def label_loop(length: int, joint_at, emit) -> list[int]:
    """NeMo greedy_batch label looping (reference.run_steps): joint_at(t) -> (token, duration bin)."""
    t, last_nb, lasts, tokens = 0, -1, 0, []
    guard = 0
    while t < length:
        guard += 1
        if guard > 20 * (length + 1) * MAX_SYMBOLS:
            raise RuntimeError("decode loop does not terminate")
        tok, dbin = joint_at(t)
        dur = DURATIONS[dbin]
        emitted = tok != BLANK
        advance = 1 if (not emitted and dur == 0) else dur
        if emitted:
            lasts = lasts + 1 if last_nb == t else 1
            last_nb = t
            tokens.append(tok)
            emit(tok)
            if t + advance < length and lasts >= MAX_SYMBOLS and last_nb == t + advance:
                advance += 1
        t += advance
    return tokens


def free_decode(paths: Paths, enc: np.ndarray, length: int) -> dict:
    """Greedy TDT through (a) Decoder + JointDecision and (b) DecoderJoint; token sequences."""
    st = {"g": None, "state": None}
    st["g"], st["state"] = paths.decoder(BLANK, zero_state())

    def jd_at(t):
        tok, dbin, _ = paths.joint_decision(frame(enc, t), st["g"])
        return tok, dbin

    def jd_emit(tok):
        st["g"], st["state"] = paths.decoder(tok, st["state"])

    a = label_loop(length, jd_at, jd_emit)
    fz = {"last": BLANK, "prev": zero_state(), "out": None}

    def dj_at(t):
        tok, dbin, s_out, _ = paths.decoder_joint(fz["last"], fz["prev"], frame(enc, t))
        fz["out"] = s_out
        return tok, dbin

    def dj_emit(tok):
        fz["last"], fz["prev"] = tok, fz["out"]

    b = label_loop(length, dj_at, dj_emit)
    return {"jd": a, "dj": b}


def head_stats(rep: dict, ref: dict) -> dict:
    """Per clip: decision bookkeeping for both paths (arm-independent decisive set) and logit/state errors."""
    r = ref["fp16s_logits"]
    out = {"steps": int(r.shape[0]), "finite": rep["finite"]}
    for head, sl in (("token", slice(0, BLANK + 1)), ("duration", slice(BLANK + 1, None))):
        rr = r[:, sl]
        srt = np.sort(rr, axis=1)
        margin = srt[:, -1] - srt[:, -2]
        decisive = margin >= G4B["decisive_margin"]
        ref_arg = rr.argmax(axis=1)
        entry = {"decisive": int(decisive.sum()), "margin_q05_q50_q95": np.quantile(margin, [.05, .5, .95]).tolist()}
        for path in ("jd", "dj"):
            arg = rep[path][0 if head == "token" else 1]
            agree = arg == ref_arg
            entry[path] = {"agree_all": int(agree.sum()), "agree_decisive": int((agree & decisive).sum())}
        rel, ab = refcache.errors(rep["logits"][:, sl], rr, TAU16)
        entry["logits_rel"], entry["logits_abs"] = rel, ab
        out[head] = entry
    for q in ("h", "c"):
        rel, ab = refcache.errors(rep[q], ref[f"fp16s_{q}"], TAU16)
        out[q] = {"rel": rel, "abs": ab}
    return out


def pool(per_clip: dict) -> dict:
    total = sum(v["steps"] for v in per_clip.values())
    res = {"clips": len(per_clip), "steps": total, "all_finite": all(v["finite"] for v in per_clip.values()), "paths": {}}
    ok = res["all_finite"] and total > 0
    for path in ("jd", "dj"):
        pr = {}
        for head in ("token", "duration"):
            dec = sum(v[head]["decisive"] for v in per_clip.values())
            agd = sum(v[head][path]["agree_decisive"] for v in per_clip.values())
            aga = sum(v[head][path]["agree_all"] for v in per_clip.values())
            h = {"decisive_fraction": dec / total if total else 0.0, "agreement_on_decisive": agd / dec if dec else 0.0,
                 "agreement_all_steps": aga / total if total else 0.0, "disagreements_all": total - aga,
                 "disagreements_decisive": dec - agd}
            h["pass"] = (h["decisive_fraction"] >= G4B["decisive_min"] and h["agreement_on_decisive"] >= G4B["agree_decisive"]
                         and h["agreement_all_steps"] >= G4B["agree_all"])
            ok &= h["pass"]
            pr[head] = h
        res["paths"][path] = pr
    res["rev5_diagnostic"] = {q: {"rel_max": float(max(v[q]["logits_rel"] if q in ("token", "duration") else v[q]["rel"]
                                                       for v in per_clip.values()))}
                              for q in ("token", "duration", "h", "c")}
    for q in res["rev5_diagnostic"].values():
        q["pass_rev5"] = q["rel_max"] <= REV5["heads_rel"]
    res["pass"] = bool(ok)
    return res


def free_stats(per_clip: dict) -> dict:
    out = {"clips": len(per_clip)}
    ok = len(per_clip) > 0
    for path in ("jd", "dj"):
        same = sum(1 for v in per_clip.values() if v[path] == v["reference"])
        frac = same / len(per_clip) if per_clip else 0.0
        out[path] = {"identical": same, "identity_fraction": frac, "pass_identity": frac >= G4B["sequence_identity"]}
        ok &= out[path]["pass_identity"]
    out["pass_identity"] = bool(ok)
    out["wer"] = "evaluated by mil/eligibility.py from the token sequences (within +0.2 points of the reference)"
    return out


# --- 4b --------------------------------------------------------------------------------------------------

def gate4b(args) -> int:
    t0 = time.time()
    label = args.arm
    man = build_manifest(args.model, label, args.variant)
    index = refcache.validate(args.model, man["provenance"])
    clips = clips_manifest()
    rows, loads, full15, own = {}, {}, {}, {}
    for clip, b, enc, length in bucket_runs(args.model, label, args.variant, args.units, clips):
        if clip is None:
            loads[f"b{b}"] = length
            continue
        ref = refcache.load_clip(args.model, clip["id"])
        e = clip["encoder_frames"]
        valid = enc[:, :e]
        rel, ab = refcache.errors(valid, ref["fp16s_enc"], TAU16)
        fin = bool(np.isfinite(valid).all())
        row = {"bucket": b, "kind": clip["kind"], "rel": rel, "abs": ab, "finite": fin, "length_ok": length == e,
               "finite_padded": bool(np.isfinite(enc).all()),
               "pass": fin and length == e and rel <= G4B["encoder_rel"],
               "pass_rev5": fin and length == e and rel <= REV5["encoder_rel"] and ab <= REV5["encoder_abs"]}
        if b == 15:
            full15[clip["id"]] = valid
        else:
            r5, a5 = refcache.errors(valid, full15[clip["id"]], TAU16)
            row["vs_15s"] = {"rel": r5, "abs": a5, "pass": fin and r5 <= G4B["encoder_rel"],
                             "pass_rev5": fin and r5 <= REV5["encoder_rel"] and a5 <= REV5["encoder_abs"]}
        if b == clip["bucket"] or args.variant == "fixed":
            own[clip["id"]] = valid
        rows[f"{clip['id']}@{b}"] = row
        print(f"{clip['id']}@{b} rel {rel:.4g}", flush=True) if not row["pass"] else None
    print(f"encoder done at {time.time() - t0:.0f}s", flush=True)
    paths = Paths(args.model, args.units)
    traces = traces_by_id()
    heads, free = {}, {}
    for clip in clips:
        ref = refcache.load_clip(args.model, clip["id"])
        heads[clip["id"]] = head_stats(replay(paths, own[clip["id"]], traces[clip["id"]]), ref)
        if clip["kind"] == "natural":
            fd = free_decode(paths, own[clip["id"]], clip["encoder_frames"])
            free[clip["id"]] = {**fd, "reference": index["free_tokens"][clip["id"]]["fp16s_free_tokens"]}
    del paths
    purge_cache()
    enc_fail = [k for k, r in rows.items() if not r["pass"]]
    g5 = [r for r in rows.values() if "vs_15s" in r]
    g5_gated = [r for r in g5 if r["kind"] in GATE5_KINDS]
    heads_pool, free_pool = pool(heads), free_stats(free)
    summary = {
        "encoder_4b": {"pass": not enc_fail, "cases": len(rows), "failing_cases": len(enc_fail), "failing_examples": enc_fail[:20],
                       "rel_max": max(r["rel"] for r in rows.values()), "abs_max": max(r["abs"] for r in rows.values()),
                       "rel_median": float(np.median([r["rel"] for r in rows.values()])),
                       "all_finite_valid": all(r["finite"] for r in rows.values()),
                       "all_finite_padded_frames": all(r["finite_padded"] for r in rows.values()),
                       "note": "gated on the valid frames; padded frames are unspecified (MASKING.md) and reported"},
        "encoder_rev5_diagnostic": {"pass": all(r["pass_rev5"] for r in rows.values()),
                                    "failing_cases": sum(1 for r in rows.values() if not r["pass_rev5"])},
        "gate5": ({"pass": all(r["vs_15s"]["pass"] for r in g5_gated), "gated_cases": len(g5_gated),
                   "gated_rel_max": max((r["vs_15s"]["rel"] for r in g5_gated), default=None),
                   "all_cases": len(g5), "all_rel_max": max((r["vs_15s"]["rel"] for r in g5), default=None),
                   "pass_rev5_gated": all(r["vs_15s"]["pass_rev5"] for r in g5_gated)}
                  if g5 else {"pass": None, "note": "fixed 15 s window: no smaller buckets"}),
        "heads_4b": heads_pool,
        "free_decoding": free_pool,
    }
    passed = (summary["encoder_4b"]["pass"] and heads_pool["pass"] and free_pool["pass_identity"]
              and summary["gate5"]["pass"] is not False)
    doc = header("4b", model=args.model, arm=args.arm, label=label, variant=args.variant, units=args.units,
                 backend=BACKEND[args.units], thresholds=G4B, rev5_diagnostic_thresholds=REV5,
                 build_provenance=man["provenance"], build_chain=man["encoding"].get("chain"),
                 decoder_models=paths_manifest(args.model))
    doc["pass"] = bool(passed)
    doc["pass_scope"] = "all 4b conditions except WER (eligibility.py) and gate 5 where applicable"
    doc["summary"] = summary
    doc.update(encoder_cases=rows, heads_per_clip=heads, free_decoding_tokens=free, load_s=loads,
               seconds=round(time.time() - t0, 1), peak_rss_mb=_peak_mb())
    return finish(doc, f"{args.model}-{label}-{args.variant}-{args.units}.json")


def paths_manifest(model: str, precision: str = "fp16") -> dict:
    base = arms_root() / model / ("decoder" if precision == "fp16" else f"decoder-{precision}")
    m = json.loads((base / "manifest.json").read_text())
    return {"dir": str(base), "provenance": m["provenance"], "precision": m.get("precision", "fp16")}


def _peak_mb() -> float:
    import resource

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / 2 ** 20 if sys.platform == "darwin" else r / 1024, 1)


# --- 4a decoder -------------------------------------------------------------------------------------------

def gate4a_decoder(args) -> int:
    t0 = time.time()
    pm = paths_manifest(args.model, "fp32")
    index = refcache.validate(args.model, pm["provenance"])
    paths = Paths(args.model, "cpuOnly", "fp32")
    traces = traces_by_id()
    rows, fails = {}, []
    for clip in clips_manifest():
        ref = refcache.load_clip(args.model, clip["id"])
        enc = ref["fp16s_enc"]
        rep = replay(paths, enc, traces[clip["id"]])
        r = ref["fp16s_logits"]
        row = {"finite": rep["finite"]}
        for q, a, b in (("logits", rep["logits"], r), ("h", rep["h"], ref["fp16s_h"]), ("c", rep["c"], ref["fp16s_c"])):
            rel, ab = refcache.errors(a, b, TAU32)
            row[q] = {"rel": rel, "abs": ab, "pass": rel <= G4A["rel"] and ab <= G4A["abs"]}
        ref_tok, ref_dur = r[:, :BLANK + 1].argmax(1), r[:, BLANK + 1:].argmax(1)
        for path in ("jd", "dj"):
            row[f"{path}_decisions_equal"] = bool((rep[path][0] == ref_tok).all() and (rep[path][1] == ref_dur).all())
        fd = free_decode(paths, enc, clip["encoder_frames"])
        reference = index["free_tokens"][clip["id"]]["fp16s_free_tokens"]
        row["free_identical"] = {p: fd[p] == reference for p in ("jd", "dj")}
        row["pass"] = (row["finite"] and all(row[q]["pass"] for q in ("logits", "h", "c"))
                       and row["jd_decisions_equal"] and row["dj_decisions_equal"] and all(row["free_identical"].values()))
        rows[clip["id"]] = row
        if not row["pass"]:
            fails.append(clip["id"])
    del paths
    purge_cache()
    doc = header("4a-decoder", model=args.model, label="decoder-fp32", units="cpuOnly", thresholds=G4A,
                 build_provenance=pm["provenance"])
    doc["pass"] = not fails
    doc["summary"] = {"clips": len(rows), "failing": fails[:20], "failing_clips": len(fails),
                      **{f"{q}_rel_max": max(r[q]["rel"] for r in rows.values()) for q in ("logits", "h", "c")},
                      **{f"{q}_abs_max": max(r[q]["abs"] for r in rows.values()) for q in ("logits", "h", "c")}}
    doc.update(per_clip=rows, seconds=round(time.time() - t0, 1))
    return finish(doc, f"{args.model}-decoder-fp32-cpuOnly-4a.json")


def gate4b_decoder(args) -> int:
    """Decoder-only 4b (diagnostic for attribution): the FP16 deployed decoder paths fed with the FP32
    reference's own encoder output; same decision and free-decoding conditions as 4b."""
    t0 = time.time()
    pm = paths_manifest(args.model)
    index = refcache.validate(args.model, pm["provenance"])
    paths = Paths(args.model, args.units)
    traces = traces_by_id()
    heads, free = {}, {}
    for clip in clips_manifest():
        ref = refcache.load_clip(args.model, clip["id"])
        heads[clip["id"]] = head_stats(replay(paths, ref["fp16s_enc"], traces[clip["id"]]), ref)
        if clip["kind"] == "natural":
            fd = free_decode(paths, ref["fp16s_enc"], clip["encoder_frames"])
            free[clip["id"]] = {**fd, "reference": index["free_tokens"][clip["id"]]["fp16s_free_tokens"]}
    del paths
    purge_cache()
    hp, fp = pool(heads), free_stats(free)
    doc = header("4b-decoder", model=args.model, label="decoder", units=args.units, backend=BACKEND[args.units],
                 thresholds=G4B, decoder_models=pm, encoder_input="FP32 reference encoder output (FP16-rounded scales)")
    doc["pass"] = bool(hp["pass"] and fp["pass_identity"])
    doc["summary"] = {"heads_4b": hp, "free_decoding": fp}
    doc.update(heads_per_clip=heads, free_decoding_tokens=free, seconds=round(time.time() - t0, 1))
    return finish(doc, f"{args.model}-decoder-{args.units}-4b.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("4a")
    p.add_argument("--model", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--variant", required=True, choices=("fixed", "multi", "enum"))
    p = sub.add_parser("4a-decoder")
    p.add_argument("--model", required=True)
    p = sub.add_parser("4b-decoder")
    p.add_argument("--model", required=True)
    p.add_argument("--units", required=True, choices=tuple(BACKEND))
    p = sub.add_parser("4b")
    p.add_argument("--model", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--variant", required=True, choices=("fixed", "multi", "enum"))
    p.add_argument("--units", required=True, choices=tuple(BACKEND))
    args = parser.parse_args()
    sys.exit({"4a": gate4a, "4a-decoder": gate4a_decoder, "4b": gate4b, "4b-decoder": gate4b_decoder}[args.cmd](args))


if __name__ == "__main__":
    main()
