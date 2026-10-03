"""Deployed-pipeline gates and records (WP7; DESIGN.md revision 10, gate 4b applied to the deployed combinations).

A deployed pipeline = front end A (vDSP) -> encoder arm (model, arm, variant, backend) -> decode loop (F2 native
FP32; F0 / F1 with the FP32 decoder models). Gated on all 82 clips, run in Swift (`parakeet-bench gate`):
- encoder: the arm's output from front end A's features (own bucket) vs the FP32 reference encoder (FP16-rounded
  scales, the weights every exact arm encodes) on FP64-evaluated reference features (DESIGN.md rev. 9, gate 4,
  "Deployed-pipeline references"; `ref64` computes it once on the Mac): rel <= 0.1 (tau 1e-3) on every clip, finite,
  encoder_length = clips.json. The comparison against the FP32-feature reference (WP3's refcache) is a diagnostic;
- every output finite: the encoder output and every decoder output captured in free decoding and in replay
  (logits, LSTM state, decoder outputs, token probabilities; counted in Swift and recounted here);
- decisions: forced replay of every trace through the pipeline's decode loop; per head (token incl. blank,
  duration), steps where the FP32-feature reference's raw-logit top-1 margin is >= 1.0 are decisive; pooled
  agreement with the reference argmax >= 99.5% on decisive steps, >= 99% on all, >= 50% decisive;
- free decoding of the 64 natural clips: token sequences identical to the reference's on >= 61, and WER within
  +0.2 points of the reference (scored on NixOS with the parent experiment's scorer);
- coverage: every clip of clips.json, every trace replayed in full, every decode present; else the gate fails.
Reported (DESIGN.md rev. 10, gate 4b): head errors for token logits (incl. blank) and duration logits for F2
(directly) and F0 (JointLogits on the same joint inputs); for F1 they are UNAVAILABLE (DecoderJoint outputs only
decisions, a probability and its state), and a labelled informational PROXY (FP32 Decoder + JointLogits evaluated on
F1's own inputs) is reported separately with its consistency to F1 (state, token and duration argmax agreement),
never as F1's head error. LSTM h and c errors (all three, F1's own h_out/c_out), token-probability differences, and
margin distributions (the reference's, the pipeline's own for F2/F0, and the reference margins of disagreeing
steps). Every required diagnostic section, decision and probability array and capture count must have its exact
dimensions and valid values before anything is scored.

Identity (review WP7 r1 findings 2, 4): records bind every component (parakeet-bench recomputes and compares them
before timing, executable SHA-256 included); the FP64-feature reference is valid only while its front-end manifest,
reference code, model and refcache identities are unchanged; an evaluation is accepted only from this exact
pipegate.py, this design revision, and a clean Mac checkout at the commit NixOS is on.

Records: ios/results/eligibility/pipelines/<model>-<arm>-<variant>-<backend>-vdsp-<decode>.json (WP3's record format
plus "kind": "pipeline", "front_end", "decode", "decoder_precision", "encoder_record", "components", "build").

  python -m pipegate ref64                             # Mac, once per reviewed commit, inside macguard (6G)
  python -m pipegate evaluate --gate-dir DIR           # Mac, inside pipegate_job.sh (exit 10 on a failed condition)
  ./python ios/pipegate.py run [--only NAME,...]        # NixOS: every eligible encoder record x {f2, f0, f1}
  ./python ios/pipegate.py record --evaluation FILE     # NixOS: WER, records (run calls it per combination)
  ./python ios/pipegate.py table
Exit codes of run: 0 every requested pipeline gated and passed; 10 all gated, some failed; 1 an orchestration or
job failure (nothing for that combination is published; its earlier records were withdrawn first).
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

IOS = Path(__file__).resolve().parent
sys.path.insert(0, str(IOS))
sys.path.insert(0, str(IOS.parent))

DESIGN_REVISION = 10         # pipeline records (DESIGN.md revision 10)
ENCODER_RECORD_REVISION = 8  # WP3's encoder records
CODE_VERSION = "wp7-pipegate-5"
T = {"encoder_rel": 0.1, "tau": 1e-3, "decisive_margin": 1.0, "agree_decisive": 0.995, "agree_all": 0.99,
     "decisive_min": 0.5, "identical_min": 61, "wer_points": 0.2, "diag_head_rel": 2e-2}
DURATIONS = (0, 1, 2, 3, 4)
N_DUR = 5
DECODES = ("f2", "f0", "f1")
STATE_SECTIONS = {"f2": ("h", "c"), "f0": ("h_step", "c_step"), "f1": ("h_out", "c_out")}
# replay diagnostics every decode loop must deliver (review WP7 r2 finding 3): name -> (row shape, rows per clip:
# "steps" = trace steps, "predictions" = 1 + emitted tokens). F1's proxy_* sections are the informational proxy of
# DESIGN.md rev. 10 (FP32 Decoder + JointLogits on F1's own inputs); F1's own logits are unavailable.
REQUIRED_SECTIONS = {
    "f2": {"logits": ([1030], "steps"), "h": ([2, 640], "steps"), "c": ([2, 640], "steps"),
           "pred_g": ([640], "predictions"), "pred_h": ([2, 640], "predictions"), "pred_c": ([2, 640], "predictions")},
    "f0": {"logits": ([1030], "steps"), "h_step": ([2, 640], "steps"), "c_step": ([2, 640], "steps"),
           "decoder_out": ([640], "predictions"), "h": ([2, 640], "predictions"), "c": ([2, 640], "predictions")},
    "f1": {"h_out": ([2, 640], "steps"), "c_out": ([2, 640], "steps"),
           "proxy_logits": ([1030], "steps"), "proxy_h": ([2, 640], "steps"), "proxy_c": ([2, 640], "steps")},
}
TRUE_HEADS = ("f2", "f0")    # DESIGN.md rev. 10: F0 and F2 report true head errors
PROB_DECODES = ("f0", "f1")  # their decision models output token_prob
UNAVAILABLE = "unavailable"
PROXY_LABEL = ("informational proxy (DESIGN.md rev. 10), not F1's own head error: FP32 Decoder + JointLogits evaluated "
               "on F1's own inputs (pending token, F1's input state h_in/c_in, the same encoder frame)")


def expected_values(dec: str, steps: int, predictions: int) -> int:
    """Float values one DiagSink captures for a pass: every required section plus token_prob per step (F0, F1)."""
    n = 0
    for shape, rows in REQUIRED_SECTIONS[dec].values():
        size = 1
        for k in shape:
            size *= k
        n += size * (steps if rows == "steps" else predictions)
    return n + (steps if dec in PROB_DECODES else 0)


def record_problems(dec: str, x: dict, sec: dict, steps: int, predictions: int) -> list[str]:
    """Exact lengths and valid values of the decision/probability arrays, and capture counts consistent with the
    loop's own step counts, before anything is scored (review WP7 r3 finding 2)."""
    out = []

    def ints(key: str, valid) -> None:
        v = x.get(key)
        if not isinstance(v, list) or len(v) != steps:
            out.append(f"{key}: {len(v) if isinstance(v, list) else v!r} entries != {steps} steps")
        elif not all(isinstance(t, int) and valid(t) for t in v):
            out.append(f"{key}: invalid values")

    ints("argmax_token", lambda t: 0 <= t <= 1024)
    ints("argmax_duration", lambda t: t in DURATIONS)
    if dec in PROB_DECODES:
        pr = x.get("token_prob")
        if not isinstance(pr, list) or len(pr) != steps:
            out.append(f"token_prob: {len(pr) if isinstance(pr, list) else pr!r} entries != {steps} steps")
        elif not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in pr):
            out.append("token_prob: values outside [0, 1] or not finite")
    if x.get("replay_predictions") != predictions:
        out.append(f"replay_predictions {x.get('replay_predictions')} != {predictions}")
    if x.get("replay_values") != expected_values(dec, steps, predictions):
        out.append(f"replay_values {x.get('replay_values')} != {expected_values(dec, steps, predictions)}")
    fs, ft = x.get("free_steps"), x.get("tokens")
    if not isinstance(fs, int) or fs < 1 or not isinstance(ft, list):
        out.append(f"free decoding: steps {fs!r}")
    elif x.get("free_values") != expected_values(dec, fs, 1 + len(ft)):
        out.append(f"free_values {x.get('free_values')} != {expected_values(dec, fs, 1 + len(ft))}")
    for key in ("free_nonfinite", "replay_nonfinite"):
        if not isinstance(x.get(key), int) or x[key] < 0:
            out.append(f"{key}: {x.get(key)!r}")
    return out


def section_problems(dec: str, sec: dict, steps: int, predictions: int) -> list[str]:
    out = []
    for name, (shape, rows) in REQUIRED_SECTIONS[dec].items():
        want = [steps if rows == "steps" else predictions, *shape]
        if name not in sec:
            out.append(f"section {name} missing")
        elif list(sec[name].shape) != want:
            out.append(f"section {name} shape {list(sec[name].shape)} != {want}")
    return out
MAC_REPO = "/Users/ajbarry/workspace/github.com/wilderness-labs-stt"
MAC_IOS = MAC_REPO + "/finetune/parakeet-ternary/ios"
MAC_A = "/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios"
LOCAL = Path("/mnt/hd/wilderness-labs-stt/parakeet-ios/results/pipegate")
SUMMARIES = IOS / "results" / "pipegates"
RECORDS = IOS / "results" / "eligibility" / "pipelines"
EXCLUDED = {"G0": "graph control with C0's weights (not deployable; WP3 control record)",
            "MLX": "no Swift implementation (WP6b prototype)"}


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def disk_preflight(need_gb: float, floor_gb: float = 30) -> None:
    """Refuse (SystemExit 5) unless the artifact volume has the 30 GB floor + need_gb free (review r2 finding 6)."""
    import artifacts

    try:
        st = os.statvfs(artifacts.root())
        free = st.f_bavail * st.f_frsize / 2 ** 30
    except OSError as exc:
        raise SystemExit(f"refusing: cannot read free disk space ({exc})") from None
    if free < floor_gb + need_gb:
        print(f"refusing: {free:.1f} GB free < {floor_gb} GB floor + {need_gb} GB", file=sys.stderr)
        raise SystemExit(5)


# --- ref64 (Mac) -------------------------------------------------------------------------------------------------

def ref64_dir(model: str) -> Path:
    import artifacts

    return artifacts.root() / "refcache" / f"{model}-ref64"


def reference_code_sha256() -> str:
    """Identity of the code that defines the FP64-feature reference: native.fp64_features, this module's ref64
    computation, refcache's scale handling and model loader, and the reference model files (models.py,
    reference.py)."""
    import native
    from mil import refcache

    parts = [inspect.getsource(native.fp64_features), inspect.getsource(cmd_ref64), inspect.getsource(refcache.set_scales),
             inspect.getsource(refcache.load_model), sha(IOS / "models.py"), sha(IOS / "reference.py")]
    return sha_text("\n".join(parts))


def cmd_ref64(args) -> int:
    """The FP32 reference encoder (FP16-rounded scales, as WP3's refcache fp16s variant) on the FP64 evaluation of
    the reference front end (native.fp64_features, the stored constants), every clip: <dir>/<id>.npy [1024, E]."""
    import numpy as np
    import torch

    import clips as clipmod
    from mil import refcache
    from mil.weights import Source
    from native import fp64_features, load_blob

    disk_preflight(1)  # 82 reference arrays, about 30 MB
    torch.set_grad_enabled(False)
    t0 = time.time()
    index = refcache.validate(args.model)
    consts_dir = Path(args.frontend)
    consts = load_blob(consts_dir, "frontend")
    out = ref64_dir(args.model)
    out.mkdir(parents=True, exist_ok=True)
    source = Source(args.model)
    model = refcache.load_model(args.model)
    refcache.set_scales(model, source, "fp16s")
    rows = {}
    for clip in clipmod.load_manifest()["clips"]:
        pcm = clipmod.read_pcm(refcache.default_pcm(), clip)
        if clipmod.pcm_sha256(pcm) != clip["sha256"]:
            raise ValueError(f"{clip['id']}: PCM SHA-256 differs from clips.json")
        f64 = fp64_features(pcm, consts["window"], consts["fb"])
        feats = torch.from_numpy(f64.astype(np.float32))[None]
        m = len(pcm) // 160
        enc, enc_len = model.encoder(feats, torch.tensor([m]))
        e = int(enc_len[0])
        if e != clip["encoder_frames"]:
            raise ValueError(f"{clip['id']}: {e} encoder frames != clips.json")
        a = enc[0, :, :e].numpy().astype(np.float32)
        np.save(out / f"{clip['id']}.npy", a)
        old = refcache.load_clip(args.model, clip["id"])
        rows[clip["id"]] = {"kind": clip["kind"], "sha256": hashlib.sha256(a.astype("<f4").tobytes()).hexdigest(),
                            "fp32_vs_fp64_features_rel_abs": [float(x) for x in refcache.errors(old["features"], f64, 1e-6)],
                            "enc_refcache_vs_ref64_rel_abs": [float(x) for x in refcache.errors(old["fp16s_enc"], a, T["tau"])]}
    doc = {"model": args.model, "scales": "fp16s", "features": "native.fp64_features (FP64 reference front end), cast to FP32",
           "frontend_manifest_sha256": sha(consts_dir / "frontend.json"), "refcache_provenance": index["provenance"],
           "model_provenance": source.provenance, "clips_json_sha256": sha(IOS / "clips.json"),
           "reference_code_sha256": reference_code_sha256(), "torch": torch.__version__,
           "seconds": round(time.time() - t0, 1), "clips": rows}
    (out / "index.json").write_text(json.dumps(doc, indent=1) + "\n")
    worst = sorted(rows.items(), key=lambda kv: -kv[1]["enc_refcache_vs_ref64_rel_abs"][0])[:3]
    print(json.dumps({"clips": len(rows), "seconds": doc["seconds"],
                      "largest refcache-vs-ref64 encoder differences": {k: v["enc_refcache_vs_ref64_rel_abs"] for k, v in worst}}))
    return 0


def validate_ref64(model: str, frontend_manifest_sha256: str, refcache_index: dict) -> tuple[Path, dict, list[str]]:
    """The FP64-feature reference and every reason it is stale (review WP7 r1 finding 4)."""
    from mil.weights import Source

    d = ref64_dir(model)
    r64 = json.loads((d / "index.json").read_text())
    problems = []
    checks = (("clips_json_sha256", sha(IOS / "clips.json"), "clips.json"),
              ("frontend_manifest_sha256", frontend_manifest_sha256, "front-end constants manifest (as loaded by the gate)"),
              ("reference_code_sha256", reference_code_sha256(), "reference code"),
              ("refcache_provenance", refcache_index["provenance"], "refcache provenance"),
              ("model_provenance", Source(model).provenance, "model source identity"))
    for key, current, label in checks:
        if r64.get(key) != current:
            problems.append(f"FP64-feature reference is stale: {label} changed ({key})")
    if sorted(r64.get("clips", {})) != sorted(c["id"] for c in json.loads((IOS / "clips.json").read_text())["clips"]):
        problems.append("FP64-feature reference does not cover every clip")
    return d, r64, problems


# --- evaluate (Mac) ----------------------------------------------------------------------------------------------

def quantiles(x) -> dict:
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {}
    q = np.percentile(x, [0, 1, 5, 25, 50])
    return {"n": int(x.size), "min": round(float(q[0]), 4), "p1": round(float(q[1]), 4), "p5": round(float(q[2]), 4),
            "p25": round(float(q[3]), 4), "median": round(float(q[4]), 4)}


def load_diag(gate_dir: Path, entry: dict) -> dict:
    """Sections of one replay diagnostic file (after checking its SHA-256): name -> array [rows, *shape]."""
    import numpy as np

    raw = (gate_dir / entry["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError(f"{entry['file']}: SHA-256 differs from the gate record")
    out = {}
    for s in entry["sections"]:
        out[s["name"]] = np.frombuffer(raw, "<f4", count=s["bytes"] // 4, offset=s["offset"]).reshape(s["shape"])
    return out


def cmd_evaluate(args) -> int:
    import numpy as np

    from mil import refcache

    d = Path(args.gate_dir)
    lines = [json.loads(l) for l in (d / "gate.jsonl").read_text().splitlines() if l.strip()]
    header = next(l for l in lines if l.get("record") == "header")
    rows = {l["clip"]: l for l in lines if l.get("record") == "clip"}
    clips = {c["id"]: c for c in json.loads((IOS / "clips.json").read_text())["clips"]}
    traces = {t["id"]: t for t in json.loads((IOS / "traces.json").read_text())["clips"]}
    build = json.loads((d / "build.json").read_text())
    index = refcache.validate(header["model"])
    problems = []
    fe_manifests = {p["components"]["front_end"].get("manifest_sha256") for p in header["pipelines"].values()
                    if isinstance(p, dict) and "components" in p}
    if len(fe_manifests) != 1:
        problems.append("pipelines disagree on the front-end constants manifest")
    r64dir, r64, stale = validate_ref64(header["model"], next(iter(fe_manifests)), index)
    problems += stale
    if not build.get("clean") or build.get("executable_sha256") != header.get("executable_sha256"):
        problems.append("the gate binary is not the clean reviewed build (build.json: modified tracked files, or not built by build_reviewed.sh from this commit)")
    if sorted(rows) != sorted(clips) or sorted(header["clip_ids"]) != sorted(clips):
        problems.append(f"coverage: {len(rows)} of {len(clips)} clips")
    if header["clips_json_sha256"] != sha(IOS / "clips.json"):
        problems.append("clips.json changed since the gate run")
    if set(header["decodes"]) != set(DECODES):
        problems.append(f"decodes {header['decodes']} != {list(DECODES)}")
    enc_rows = []
    for cid, r in rows.items():
        ref = refcache.load_clip(header["model"], cid)
        a = np.fromfile(d / "enc" / f"{cid}.f32", "<f4").reshape(-1, 1024).T
        if hashlib.sha256(a.T.astype("<f4").tobytes()).hexdigest() != r["enc_sha256"]:
            problems.append(f"{cid}: encoder output file differs from the gate record")
        ref64 = np.load(r64dir / f"{cid}.npy")
        if hashlib.sha256(ref64.astype("<f4").tobytes()).hexdigest() != r64["clips"][cid]["sha256"]:
            problems.append(f"{cid}: FP64-feature reference file differs from its index")
        ok_len = r["encoder_length"] == clips[cid]["encoder_frames"] == a.shape[1]
        finite = bool(r["finite"]) and bool(np.isfinite(a).all())
        rel, ab = refcache.errors(a, ref64, T["tau"]) if ok_len else (math.inf, math.inf)
        rel32, ab32 = refcache.errors(a, ref["fp16s_enc"], T["tau"]) if ok_len else (math.inf, math.inf)
        enc_rows.append({"clip": cid, "bucket": r["bucket"], "rel": rel, "abs": ab, "length_ok": ok_len, "finite": finite,
                         "pass": ok_len and finite and rel <= T["encoder_rel"],
                         "diag_vs_reference_on_fp32_features": {"rel": rel32, "abs": ab32}})
    encoder = {"pass": all(x["pass"] for x in enc_rows) and len(enc_rows) == len(clips), "clips": len(enc_rows),
               "passed": sum(x["pass"] for x in enc_rows), "max_rel": max(x["rel"] for x in enc_rows),
               "max_abs": max(x["abs"] for x in enc_rows),
               "reference": "FP32 reference encoder (fp16s scales) on FP64-evaluated reference features (ref64)",
               "diag_vs_reference_on_fp32_features": {
                   "max_rel": max(x["diag_vs_reference_on_fp32_features"]["rel"] for x in enc_rows),
                   "over_ceiling": [x["clip"] for x in enc_rows if x["diag_vs_reference_on_fp32_features"]["rel"] > T["encoder_rel"]]}}
    per_decode = {}
    for dec in header["decodes"]:
        tok = {"all": 0, "ok": 0, "dec": 0, "dec_ok": 0, "ref_margin": [], "own_margin": [], "disagree_ref_margin": []}
        dur = {"all": 0, "ok": 0, "dec": 0, "dec_ok": 0, "ref_margin": [], "own_margin": [], "disagree_ref_margin": []}
        head_err = {"token_logits": [], "duration_logits": [], "h": [], "c": []}
        prob_diff = []
        proxy = {"token_logits": [], "duration_logits": [], "state_rel": [], "steps": 0, "token_equal": 0, "duration_equal": 0}
        nonfinite = {"free": 0, "replay": 0, "replay_recount": 0, "values": 0}
        missing, free_rows = [], {}
        for cid, r in rows.items():
            x = r.get(dec)
            if x is None or "replay_error" in x or x.get("replay_steps") != traces[cid]["steps"] or "diag" not in x:
                missing.append(cid)
                continue
            steps = traces[cid]["steps"]
            predictions = 1 + sum(traces[cid]["pred_updated"])
            sec = load_diag(d, x["diag"])
            # validation before any scoring (review r2 finding 3, r3 finding 2): sections, decisions, probabilities
            # and capture counts must all have their exact dimensions and valid values
            bad = section_problems(dec, sec, steps, predictions) + record_problems(dec, x, sec, steps, predictions)
            if bad:
                missing.append(f"{cid}: " + "; ".join(bad))
                continue
            nonfinite["free"] += x["free_nonfinite"]
            nonfinite["replay"] += x["replay_nonfinite"]
            nonfinite["values"] += x["free_values"] + x["replay_values"]
            nonfinite["replay_recount"] += sum(int((~np.isfinite(v)).sum()) for v in sec.values())
            ref = refcache.load_clip(header["model"], cid)
            logits = ref["fp16s_logits"]
            at_tok, at_dur = np.asarray(x["argmax_token"]), np.asarray(x["argmax_duration"])
            own = sec["logits"] if dec in TRUE_HEADS else None   # F1: DecoderJoint exposes no logits (rev. 10)
            for head, at, sl, vals in ((tok, at_tok, slice(0, -N_DUR), None), (dur, at_dur, slice(-N_DUR, None), DURATIONS)):
                lg = logits[:, sl]
                srt = np.sort(lg, axis=1)
                margin = srt[:, -1] - srt[:, -2]
                ref_arg = lg.argmax(1) if vals is None else np.asarray(vals)[lg.argmax(1)]
                agree = at == ref_arg
                decisive = margin >= T["decisive_margin"]
                head["all"] += len(agree); head["ok"] += int(agree.sum())
                head["dec"] += int(decisive.sum()); head["dec_ok"] += int(agree[decisive].sum())
                head["ref_margin"] += margin.tolist()
                head["disagree_ref_margin"] += margin[~agree].tolist()
                if own is not None:
                    so = np.sort(own[:, sl], axis=1)
                    head["own_margin"] += (so[:, -1] - so[:, -2]).tolist()
            if own is not None:
                head_err["token_logits"].append(refcache.errors(own[:, :-N_DUR], logits[:, :-N_DUR], T["tau"]))
                head_err["duration_logits"].append(refcache.errors(own[:, -N_DUR:], logits[:, -N_DUR:], T["tau"]))
            hn, cn = STATE_SECTIONS[dec]
            head_err["h"].append(refcache.errors(sec[hn].reshape(ref["fp16s_h"].shape), ref["fp16s_h"], T["tau"]))
            head_err["c"].append(refcache.errors(sec[cn].reshape(ref["fp16s_c"].shape), ref["fp16s_c"], T["tau"]))
            if dec == "f1":  # informational proxy only (DESIGN.md rev. 10): never reported as F1's head error
                pl = sec["proxy_logits"]
                proxy["token_logits"].append(refcache.errors(pl[:, :-N_DUR], logits[:, :-N_DUR], T["tau"]))
                proxy["duration_logits"].append(refcache.errors(pl[:, -N_DUR:], logits[:, -N_DUR:], T["tau"]))
                proxy["state_rel"].append(max(refcache.errors(sec["proxy_h"], sec["h_out"], T["tau"])[0],
                                              refcache.errors(sec["proxy_c"], sec["c_out"], T["tau"])[0]))
                proxy["steps"] += steps
                proxy["token_equal"] += int((pl[:, :-N_DUR].argmax(1) == at_tok).sum())
                proxy["duration_equal"] += int((np.asarray(DURATIONS)[pl[:, -N_DUR:].argmax(1)] == at_dur).sum())
            # probability of the pipeline's chosen token vs the reference softmax at that token (F0, F1: their own
            # token_prob output; F2: from its logits)
            tl = logits[:, :-N_DUR].astype(np.float64)
            p_ref = np.exp(tl - tl.max(1, keepdims=True))
            p_ref /= p_ref.sum(1, keepdims=True)
            if dec in PROB_DECODES:
                p_own = np.asarray(x["token_prob"], dtype=np.float64)
            else:
                ol = own[:, :-N_DUR].astype(np.float64)
                e = np.exp(ol - ol.max(1, keepdims=True))
                p_own = (e / e.sum(1, keepdims=True))[np.arange(steps), at_tok]
            prob_diff.append(float(np.abs(p_own - p_ref[np.arange(steps), at_tok]).max()))
            if clips[cid]["kind"] == "natural":
                free_rows[cid] = x["tokens"]
        heads = {}
        for name, h in (("token", tok), ("duration", dur)):
            heads[name] = {"steps": h["all"], "agree_all": h["ok"] / max(h["all"], 1), "decisive_fraction": h["dec"] / max(h["all"], 1),
                           "agree_decisive": h["dec_ok"] / max(h["dec"], 1),
                           "reference_margin": quantiles(h["ref_margin"]),
                           "pipeline_margin": quantiles(h["own_margin"]) if dec in TRUE_HEADS else UNAVAILABLE,
                           "reference_margin_of_disagreements": sorted(round(v, 4) for v in h["disagree_ref_margin"])[:50]}
            heads[name]["pass"] = (heads[name]["agree_decisive"] >= T["agree_decisive"] and heads[name]["agree_all"] >= T["agree_all"]
                                   and heads[name]["decisive_fraction"] >= T["decisive_min"])

        def summarize(v):
            rels = [e[0] for e in v]
            return {"clips": len(v), "rel_max": max(rels), "rel_median": float(np.median(rels)), "abs_max": max(e[1] for e in v),
                    "clips_over_diag_ceiling": sum(r > T["diag_head_rel"] for r in rels)} if v else None

        errs = {k: summarize(v) for k, v in head_err.items()}
        informational_proxy = None
        if dec not in TRUE_HEADS:
            errs["token_logits"] = errs["duration_logits"] = UNAVAILABLE
            errs["note"] = ("DESIGN.md rev. 10: DecoderJoint outputs only decisions, a probability and its state, so F1's token "
                            "and duration logit errors are unavailable; h and c are F1's own (h_out, c_out)")
            informational_proxy = {
                "label": PROXY_LABEL,
                "token_logits_vs_reference": summarize(proxy["token_logits"]),
                "duration_logits_vs_reference": summarize(proxy["duration_logits"]),
                "consistency_with_f1": {"state_rel_max_vs_h_out_c_out": max(proxy["state_rel"]) if proxy["state_rel"] else None,
                                        "token_argmax_agreement": proxy["token_equal"] / max(proxy["steps"], 1),
                                        "duration_argmax_agreement": proxy["duration_equal"] / max(proxy["steps"], 1),
                                        "steps": proxy["steps"]}}
        ref_tokens = {c: index["free_tokens"][c]["fp16s_free_tokens"] for c in free_rows}
        identical = sum(free_rows[c] == ref_tokens[c] for c in free_rows)
        n_nat = sum(1 for c in clips.values() if c["kind"] == "natural")
        finite_ok = nonfinite["free"] == 0 and nonfinite["replay"] == 0 and nonfinite["replay_recount"] == 0
        per_decode[dec] = {
            "coverage": {"pass": not missing and len(free_rows) == n_nat, "missing_or_incomplete": missing},
            "finite": {"pass": finite_ok, **nonfinite},
            "decisions": {"pass": all(h["pass"] for h in heads.values()), **heads},
            "head_errors": {"reference": "FP32-feature reference (refcache fp16s), forced replay; tau 1e-3", **errs},
            "token_prob_max_abs_diff_vs_reference_softmax": max(prob_diff) if prob_diff else None,
            "free_decoding": {"pass": identical >= T["identical_min"] and len(free_rows) == n_nat, "clips": len(free_rows),
                              "identical": identical},
            "free_tokens": free_rows, "reference_tokens": ref_tokens}
        if informational_proxy is not None:
            per_decode[dec]["informational_proxy"] = informational_proxy
        per_decode[dec]["pass_before_wer"] = (encoder["pass"] and not problems and per_decode[dec]["coverage"]["pass"]
                                             and finite_ok and per_decode[dec]["decisions"]["pass"]
                                             and per_decode[dec]["free_decoding"]["pass"])
    doc = {"design_revision": DESIGN_REVISION, "code_version": CODE_VERSION, "thresholds": T, "header": header, "build": build,
           "refcache_provenance": index["provenance"], "ref64_index_sha256": sha(r64dir / "index.json"), "problems": problems,
           "encoder": encoder, "encoder_rows": enc_rows, "decodes": per_decode, "pipegate_py_sha256": sha(Path(__file__)),
           "evaluated": time.strftime("%Y-%m-%d %H:%M")}
    (d / "evaluation.json").write_text(json.dumps(doc, indent=1) + "\n")
    ok = all(v["pass_before_wer"] for v in per_decode.values())
    print(json.dumps({"encoder": {k: encoder[k] for k in ("pass", "passed", "max_rel")}, "problems": problems[:10],
                      **{k: {"pass": v["pass_before_wer"], "finite": v["finite"]["pass"], "identical": v["free_decoding"]["identical"],
                             "token_decisive": round(v["decisions"]["token"]["agree_decisive"], 5),
                             "duration_decisive": round(v["decisions"]["duration"]["agree_decisive"], 5)}
                         for k, v in per_decode.items()}}))
    return 0 if ok else 10


# --- record (NixOS) --------------------------------------------------------------------------------------------------

def local_commit() -> str:
    return subprocess.run(["git", "-C", str(IOS), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def cmd_record(args) -> int:
    from mil.eligibility import _tokenizer, _wer

    ev = json.loads(Path(args.evaluation).read_text())
    # an evaluation is accepted only from this exact evaluator, design revision and reviewed build (finding 4)
    stale = []
    if ev.get("design_revision") != DESIGN_REVISION or ev.get("code_version") != CODE_VERSION:
        stale.append(f"evaluation is revision {ev.get('design_revision')} / {ev.get('code_version')}")
    if ev.get("pipegate_py_sha256") != sha(Path(__file__)):
        stale.append("evaluation was made by a different pipegate.py")
    b = ev.get("build") or {}
    if not b.get("clean") or b.get("commit") != local_commit():
        stale.append(f"gate build {b.get('commit')} (clean {b.get('clean')}) is not this checkout's commit {local_commit()}")
    if stale:
        print(json.dumps({"refused": stale}))
        return 1
    h = ev["header"]
    clips = {c["id"]: c for c in json.loads((IOS / "clips.json").read_text())["clips"]}
    backend = {"cpuAndNeuralEngine": "ane", "cpuOnly": "cpu", "cpuAndGPU": "gpu"}[h["compute_units"]]
    base = f"{h['model']}-{h['arm']}-{h['variant']}-{backend}"
    enc_rec_path = IOS / "results" / "eligibility" / f"{base}.json"
    enc_rec = json.loads(enc_rec_path.read_text())
    sp, sp_path = _tokenizer()
    SUMMARIES.mkdir(parents=True, exist_ok=True)
    RECORDS.mkdir(parents=True, exist_ok=True)
    summary_path = SUMMARIES / f"{base}-vdsp.json"
    wers = {}
    for dec, v in ev["decodes"].items():
        ref_w = _wer(v["reference_tokens"], clips, sp)
        arm_w = _wer(v["free_tokens"], clips, sp)
        wers[dec] = {"pass": arm_w["wer"] <= ref_w["wer"] + T["wer_points"] / 100, "reference_wer_pct": round(100 * ref_w["wer"], 3),
                     "pipeline_wer_pct": round(100 * arm_w["wer"], 3), "clips": len(v["free_tokens"]), "limit_points": T["wer_points"]}
    summary = {k: ev[k] for k in ("design_revision", "code_version", "thresholds", "header", "build", "refcache_provenance",
                                  "ref64_index_sha256", "problems", "encoder", "evaluated", "pipegate_py_sha256")}
    summary["decodes"] = {dec: {k: v[k] for k in ("coverage", "finite", "decisions", "head_errors",
                                                   "token_prob_max_abs_diff_vs_reference_softmax", "free_decoding",
                                                   "informational_proxy") if k in v}
                          | {"wer": wers[dec]} for dec, v in ev["decodes"].items()}
    summary["encoder_rows"] = ev["encoder_rows"]
    summary_path.write_text(json.dumps(summary, indent=1) + "\n")
    out = {}
    for dec, v in summary["decodes"].items():
        pipe = h["pipelines"][dec]
        comps = pipe["components"]
        precision = comps["decode"].get("precision")
        checks = {"encoder_record": {"pass": bool(enc_rec.get("timing_allowed")) and enc_rec.get("design_revision") == ENCODER_RECORD_REVISION,
                                     "record": enc_rec_path.name},
                  "decoder_precision": {"pass": precision == "fp32", "precision": precision},
                  "problems": {"pass": not ev["problems"], "list": ev["problems"]},
                  "encoder": ev["encoder"], "coverage": v["coverage"], "finite": v["finite"], "decisions": v["decisions"],
                  "free_decoding": v["free_decoding"], "wer": v["wer"]}
        reasons = [k for k, c in checks.items() if not c["pass"]]
        rec = {"design_revision": DESIGN_REVISION, "code_version": CODE_VERSION, "kind": "pipeline",
               "model": h["model"], "arm": h["arm"], "variant": h["variant"], "backend": backend, "compute_units": h["compute_units"],
               "front_end": "vdsp", "decode": dec, "decoder_precision": precision,
               "encoder_record": {"file": enc_rec_path.name, "sha256": sha(enc_rec_path)},
               "components": comps, "build": ev["build"], "eligible": not reasons, "timing_allowed": not reasons,
               "selection_eligible": (not reasons) and bool(enc_rec.get("selection_eligible")),
               "reasons": reasons, "checks": checks,
               "inputs": {str(summary_path.relative_to(IOS)): sha(summary_path), str(enc_rec_path.relative_to(IOS)): sha(enc_rec_path)},
               "tokenizer_sha256": sha(sp_path), "built": time.strftime("%Y-%m-%d %H:%M")}
        path = RECORDS / pipe["record"]
        path.write_text(json.dumps(rec, indent=1) + "\n")
        out[dec] = {"record": path.name, "timing_allowed": rec["timing_allowed"], "reasons": reasons}
    print(json.dumps(out))
    return 0 if all(v["timing_allowed"] for v in out.values()) else 10


# --- run (NixOS) ---------------------------------------------------------------------------------------------------------

def encoder_records() -> list[dict]:
    out = []
    for p in sorted((IOS / "results" / "eligibility").glob("*.json")):
        r = json.loads(p.read_text())
        if r.get("design_revision") != ENCODER_RECORD_REVISION or not r.get("timing_allowed") or r.get("kind") == "pipeline":
            continue
        if r.get("decoder_precision") == "fp16" or r["arm"] in EXCLUDED:
            continue
        out.append(r)
    return out


def combo_name(r: dict) -> str:
    return f"{r['model']}-{r['arm']}-{r['variant']}-{r['backend']}"


def mac(cmd: str, timeout: int = 900) -> str:
    from mil.archive import local_config

    res = subprocess.run([*local_config("WP3_MAC_RUN").split(), "--repo", MAC_IOS, "--", "sh", "-c", cmd],
                         capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(f"Mac command failed ({res.returncode}): {cmd[:120]}: {res.stderr.strip()[-300:]}")
    return res.stdout


def check_deployment() -> str:
    """The Mac checkout must be this commit, with no modified tracked files (the binary is built from it)."""
    commit = local_commit()
    remote = mac("git rev-parse HEAD; git status --porcelain --untracked-files=no | wc -l").split()
    if remote[0] != commit or remote[1] != "0":
        raise RuntimeError(f"Mac checkout {remote[0]} (modified files: {remote[1]}) is not the reviewed commit {commit}")
    return commit


def disk_need_gb(r: dict) -> int:
    """Measured headroom: the package's size on disk (measured on the Mac) plus its Core ML cache estimate: WP5
    measured 4.6 GB for C1 multi (dense FP16, four functions), 18-54 MB for every compressed arm; dense C1 gets
    4 x its package size, every other arm 1 GB; + 1 GB for the gate's outputs."""
    pkg = f"{MAC_A}/arms/{r['model']}/{r['arm']}/{r['variant']}.mlmodelc"
    kb = mac(f"du -sk {pkg} | cut -f1").strip()
    if not kb.isdigit():
        raise RuntimeError(f"cannot measure {pkg}")
    gb = int(kb) / 2 ** 20
    cache = 4 * gb if r["arm"] == "C1" else 1.0
    return math.ceil(gb + cache + 1)


def withdraw(name: str) -> list[str]:
    """Remove a combination's records and summary before it is re-gated, so a failed attempt leaves nothing that
    could pass for fresh evidence (git history keeps the old files)."""
    gone = []
    for p in [*RECORDS.glob(f"{name}-vdsp-*.json"), SUMMARIES / f"{name}-vdsp.json"]:
        if p.exists():
            p.unlink()
            gone.append(p.name)
    return gone


def cmd_run(args) -> int:
    from wp5sweep import ensure_model

    every = {combo_name(r): r for r in encoder_records()}
    only = args.only.split(",") if args.only else list(every)
    unknown = [n for n in only if n not in every]
    if unknown:
        print(f"unknown or ineligible combinations: {unknown}", file=sys.stderr)
        return 1
    commit = check_deployment()
    recs = [every[n] for n in only]
    restored: set = set()
    log, failures, gate_failures = [], [], []
    for i, r in enumerate(recs):
        name = combo_name(r)
        entry = {"combo": name, "commit": commit, "withdrawn": withdraw(name)}
        try:
            ensure_model({"model": r["model"], "arm": r["arm"], "variant": r["variant"]}, restored)
            need = disk_need_gb(r)
            run_id = time.strftime("%Y%m%d-%H%M%S")
            out = f"{MAC_A}/results/pipegate/{name}/{run_id}"
            gate = ["--eligibility", f"{r['model']}:{r['arm']}", "--encoder", f"{MAC_A}/arms/{r['model']}/{r['arm']}/{r['variant']}.mlmodelc",
                    "--encoder-variant", {"fixed": "fixed15", "multi": "multifunction", "enum": "enumerated"}[r["variant"]],
                    "--compute-units", r["compute_units"], "--frontend-constants", f"{MAC_A}/native/mp2",
                    "--native-weights", f"{MAC_A}/native/mp2", "--decoder-models", f"{MAC_A}/arms/mp2/decoder-fp32",
                    "--decodes", ",".join(DECODES), "--clips", f"{MAC_IOS}/clips.json", "--pcm", f"{MAC_A}/clips",
                    "--traces", f"{MAC_IOS}/traces.json"]
            # 6G: dense C1 (WP5) and the GPU and CPU backends, whose decompressed / GPU-visible model memory counts in
            # the job's RSS (C3 multi with the F0/F1 decoder models exceeded 4.19 GB at load on cpuAndGPU and on
            # cpuOnly); macguard's system-memory and swap aborts still apply
            cap = "6G" if r["arm"] == "C1" or r["backend"] != "ane" else "4G"
            q = " ".join(shlex.quote(x) for x in gate)
            line = (f"mkdir -p {out}; i=0; while :; do ./macguard --rss-cap {cap} --timeout 5400 -- sh pipegate_job.sh {need} {out} -- {q} "
                    f"> {out}/job.log 2>&1; s=$?; [ $s -ne 3 ] && break; i=$((i+1)); [ $i -gt 120 ] && break; sleep 60; done; "
                    f"echo $s > {out}/STATUS")
            mac(f"nohup sh -c {shlex.quote(line)} > /dev/null 2>&1 < /dev/null & echo started")
            entry.update({"run": run_id, "need_gb": need, "cap": cap})
            t0, status = time.time(), ""
            while not status and time.time() - t0 < 6 * 3600:  # 120 lock retries + the 5400 s job, with margin
                time.sleep(60)
                try:
                    status = mac(f"cat {out}/STATUS 2>/dev/null || true").strip()
                except Exception as exc:  # a transient SSH failure must not abandon a running job
                    print(f"poll: {exc}", file=sys.stderr, flush=True)
            if not status.lstrip("-").isdigit():
                raise RuntimeError(f"no status from {out} (job still running or lost)")
            entry.update({"status": int(status), "minutes": round((time.time() - t0) / 60, 1)})
            if int(status) not in (0, 10):
                raise RuntimeError(f"gate job exit {status}")
            # complete retrieval into a temporary directory, then an atomic rename (no partial or stale results)
            final = LOCAL / name / run_id
            tmp = LOCAL / name / f".{run_id}.part"
            tmp.mkdir(parents=True, exist_ok=False)
            for f in ("evaluation.json", "gate.jsonl", "job.log", "cache.log", "build.json"):
                (tmp / f).write_text(mac(f"cat {out}/{f}"))
            json.loads((tmp / "evaluation.json").read_text())  # must parse
            os.replace(tmp, final)
            res = subprocess.run([str(IOS.parent / "python"), str(Path(__file__)), "record", "--evaluation", str(final / "evaluation.json")],
                                 capture_output=True, text=True)
            entry["record_exit"] = res.returncode
            entry["record"] = res.stdout.strip()[-600:] or res.stderr.strip()[-600:]
            if res.returncode == 10:
                gate_failures.append(name)
            elif res.returncode != 0:
                withdraw(name)
                raise RuntimeError(f"record failed ({res.returncode})")
            published = sorted(p.name for p in RECORDS.glob(f"{name}-vdsp-*.json"))
            if len(published) != len(DECODES):
                withdraw(name)
                raise RuntimeError(f"{len(published)} records published, expected {len(DECODES)}")
        except Exception as exc:  # recorded; nothing of this combination is published; later combinations continue
            entry["error"] = str(exc)[-600:]
            failures.append(name)
        log.append(entry)
        print(json.dumps(entry), flush=True)
        later = {(b["model"], b["arm"]) for b in recs[i + 1:]}
        for key in sorted(restored - later):
            subprocess.run([str(IOS.parent / "python"), str(IOS / "mil" / "archive.py"), "out", *key], check=True)
            restored.discard(key)
    LOCAL.mkdir(parents=True, exist_ok=True)
    summary = {"commit": commit, "requested": only, "failures": failures, "gate_failures": gate_failures, "log": log}
    (LOCAL / f"run-{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({"requested": len(only), "orchestration_failures": failures, "gate_failures": gate_failures}))
    return 1 if failures else 10 if gate_failures else 0


def cmd_table(args) -> int:
    rows = []
    for p in sorted(RECORDS.glob("*.json")):
        r = json.loads(p.read_text())
        c = r["checks"]
        rows.append(f"| {r['model']} | {r['arm']} | {r['variant']} | {r['backend']} | {r['decode']} | "
                    f"{'yes' if r['timing_allowed'] else 'no'} | {', '.join(r['reasons']) or '-'} | "
                    f"{c['encoder'].get('max_rel', float('nan')):.4f} | {c['free_decoding']['identical']}/64 | "
                    f"{c['decisions']['token']['agree_decisive']:.4f} / {c['decisions']['duration']['agree_decisive']:.4f} | "
                    f"{c['decisions']['token']['agree_all']:.4f} / {c['decisions']['duration']['agree_all']:.4f} | "
                    f"{c['wer']['pipeline_wer_pct']:.3f} / {c['wer']['reference_wer_pct']:.3f} |")
    text = ("Deployed-pipeline records (front end A + encoder arm + decode loop; DESIGN.md revision 10 gate 4b conditions).\n\n"
            "| model | arm | variant | backend | decode | timing allowed | failed | encoder rel max | identical | "
            "decisive agreement token / duration | all-step agreement token / duration | WER pipeline / ref (%) |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
    (RECORDS / "table.txt").write_text(text)
    print(text)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ref64"); p.add_argument("--model", default="mp2")
    p.add_argument("--frontend", default=f"{MAC_A}/native/mp2")
    p.set_defaults(func=cmd_ref64)
    p = sub.add_parser("evaluate"); p.add_argument("--gate-dir", required=True); p.set_defaults(func=cmd_evaluate)
    p = sub.add_parser("record"); p.add_argument("--evaluation", required=True); p.set_defaults(func=cmd_record)
    p = sub.add_parser("run"); p.add_argument("--only"); p.set_defaults(func=cmd_run)
    sub.add_parser("table").set_defaults(func=cmd_table)
    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
