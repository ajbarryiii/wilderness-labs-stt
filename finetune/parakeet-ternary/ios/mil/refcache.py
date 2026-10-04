"""FP32 reference outputs for the gates, cached per clip (DESIGN.md gates 3-4), plus C5 calibration and probe inputs.

  cd ios && python -m mil.refcache run --model mp2 [--pcm DIR] [--out DIR]

One reference model in memory at a time (models.load: about 3-4 GB peak for mp2); no Core ML model is
loaded in this process. For every clip of clips.json (PCM SHA-256 verified):
- features: the reference preprocessor's log-mel [128, M + 1] (M = N // 160 valid frames, the extra frame
  zero) and M, i.e. the arms' mel / mel_length before bucket padding;
- two scale variants of the same model, "fp32" (the export's FP32 row scales) and "fp16s" (every ternary
  module's scales rounded to FP16: W = codes x FP16(s), the weights every exact arm encodes):
  encoder output on the valid frames [1024, E], and the forced replay of the clip's traces.json trace:
  raw joint logits [S, 1030] and the LSTM state (h, c) [S, 2, 640] that produced each step's prediction;
- the reference's own free greedy decode (tokens) for both variants (informational).
Writes <out>/<clip id>.npz and <out>/index.json; gate 3 (fp16s vs fp32 per output) goes to
ios/results/gates/<model>-gate3.json. For mp2 also:
- C5 calibration: per layer and site, max |x| of the input of every ternary matmul over 16 natural clips
  (4 per bucket, the first by id), FP32-scale model; scale = max / 127 -> ios/results/calibration/;
- probe inputs: layer 0 and layer 23 matmul inputs on the longest natural 15 s clip (<out>/probe_inputs.npz).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mil import evidence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "mil"

import numpy as np

from . import IOS  # noqa: E402

GATES = IOS / "results" / "gates"
CALIB = IOS / "results" / "calibration"
SITE_MODULES = {"ff1_in": "feed_forward1.linear1", "ff1_mid": "feed_forward1.linear2", "att_in": "self_attn.linear_q",
                "att_out": "self_attn.linear_out", "conv_in": "conv.pointwise_conv1", "conv_mid": "conv.pointwise_conv2",
                "ff2_in": "feed_forward2.linear1", "ff2_mid": "feed_forward2.linear2"}
TAU_FP32 = 1e-6


def errors(a, r, tau: float) -> tuple[float, float]:
    """DESIGN.md: rel = ||a - r|| / max(||r||, tau sqrt(n)); abs = max|a - r| / max(RMS(r), tau)."""
    a = np.asarray(a, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    if a.shape != r.shape:
        raise ValueError(f"shape {a.shape} != {r.shape}")
    if r.size == 0:
        return 0.0, 0.0
    d = a - r
    rel = float(np.linalg.norm(d) / max(float(np.linalg.norm(r)), tau * np.sqrt(r.size)))
    rms = float(np.sqrt(np.mean(r * r)))
    return rel, float(np.abs(d).max() / max(rms, tau))


def default_out(model: str) -> Path:
    import artifacts

    return artifacts.root() / "refcache" / model


def default_pcm() -> Path:
    import artifacts

    return artifacts.root() / "clips"


def load_model(name: str):
    import models

    from .weights import default_path

    return models.load(name, default_path(name))


def set_scales(model, source, rounding: str) -> int:
    """Rewrite every ternary weight in place as codes x s (rounding "fp32": the export's scales; "fp16s":
    FP16-rounded scales). Returns the number of modules."""
    import torch

    from .weights import TERNARY_SUFFIXES, fp16_scale

    n = 0
    with torch.no_grad():
        for i in range(model.cfg.n_layers):
            for suffix in TERNARY_SUFFIXES:
                key = f"encoder.layers.{i}.{suffix}"
                codes, scale = source.ternary(key)
                s = scale if rounding == "fp32" else fp16_scale(scale).astype(np.float32)
                target = model.get_parameter(key + ".weight")
                target.view(target.shape[0], -1).copy_(torch.from_numpy(codes.astype(np.float32) * s[:, None]))
                n += 1
    return n


def run(args) -> None:
    import torch

    import clips as clipmod
    import reference
    import traces as tracemod

    from .weights import Source

    torch.set_grad_enabled(False)
    t0 = time.time()
    out = Path(args.out or default_out(args.model))
    import artifacts

    out = artifacts.check(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = clipmod.load_manifest()
    trace_doc = json.loads(evidence.read_text((IOS / "traces.json")))
    traces = {r["id"]: r for r in trace_doc["clips"]}
    source = Source(args.model)
    model = load_model(args.model)
    print(f"loaded {args.model} in {time.time() - t0:.1f}s", flush=True)
    pcm_dir = Path(args.pcm or default_pcm())
    clips = manifest["clips"] if not args.ids else [c for c in manifest["clips"] if c["id"] in args.ids.split(",")]
    results: dict[str, dict] = {c["id"]: {} for c in clips}
    store: dict[str, dict] = {c["id"]: {} for c in clips}
    for variant in ("fp32", "fp16s"):
        n = set_scales(model, source, variant)
        print(f"{variant}: {n} ternary modules set", flush=True)
        for clip in clips:
            pcm = clipmod.read_pcm(pcm_dir, clip)
            if clipmod.pcm_sha256(pcm) != clip["sha256"]:
                raise ValueError(f"{clip['id']}: PCM SHA-256 differs from clips.json")
            audio = torch.from_numpy(pcm.astype(np.float32))[None]
            feats, feat_len = model.preprocessor(audio, torch.tensor([len(pcm)]))
            enc, enc_len = model.encoder(feats, feat_len)
            m, e = int(feat_len[0]), int(enc_len[0])
            if m != clip["mel_frames"] or e != clip["encoder_frames"]:
                raise ValueError(f"{clip['id']}: frames {m}/{e} differ from clips.json")
            trace = tracemod.record_to_trace(traces[clip["id"]])
            steps = reference.replay(model, enc, e, trace)
            free = reference.greedy_decode(model, enc, enc_len)[0]
            s = store[clip["id"]]
            if variant == "fp32":
                s["features"] = feats[0].numpy().astype(np.float32)
                s["mel_length"] = np.int32(m)
            else:
                if not np.array_equal(s["features"], feats[0].numpy()):
                    raise ValueError("features depend on the scale variant")
            s[f"{variant}_enc"] = enc[0, :, :e].numpy().astype(np.float32)
            s[f"{variant}_logits"] = steps.logits.numpy().astype(np.float32)
            s[f"{variant}_h"] = steps.h.numpy().astype(np.float32)
            s[f"{variant}_c"] = steps.c.numpy().astype(np.float32)
            results[clip["id"]][f"{variant}_free_tokens"] = free.tokens
            results[clip["id"]][f"{variant}_replay_argmax_equals_trace"] = (
                steps.argmax_token.tolist() == trace.token and steps.argmax_duration.tolist() == trace.duration)
        print(f"{variant} done at {time.time() - t0:.0f}s", flush=True)
    gate3 = {}
    for clip in clips:
        s = store[clip["id"]]
        nd = 5
        row = {}
        for key, a, r in (("encoder", s["fp16s_enc"], s["fp32_enc"]),
                          ("token_logits", s["fp16s_logits"][:, :-nd], s["fp32_logits"][:, :-nd]),
                          ("duration_logits", s["fp16s_logits"][:, -nd:], s["fp32_logits"][:, -nd:]),
                          ("h", s["fp16s_h"], s["fp32_h"]), ("c", s["fp16s_c"], s["fp32_c"])):
            rel, ab = errors(a, r, TAU_FP32)
            row[key] = {"rel": rel, "abs": ab}
        row["free_tokens_equal"] = results[clip["id"]]["fp16s_free_tokens"] == results[clip["id"]]["fp32_free_tokens"]
        gate3[clip["id"]] = row
        np.savez(out / f"{clip['id']}.npz", **s)
    index = {"model": args.model, "provenance": source.provenance, "clips": [c["id"] for c in clips],
             "variants": {"fp32": "export FP32 row scales", "fp16s": "row scales rounded to FP16 (the arms' weights)"},
             "arrays": {"features": "[128, M + 1] float32", "mel_length": "M int32", "<v>_enc": "[1024, E] valid frames",
                        "<v>_logits": "[S, 1030] raw joint logits (forced replay of traces.json)",
                        "<v>_h / <v>_c": "[S, 2, 640] LSTM state producing each step's prediction"},
             "free_tokens": {k: {kk: vv for kk, vv in v.items() if "free" in kk} for k, v in results.items()},
             "traces_json_sha256": hashlib.sha256(evidence.read_bytes((IOS / "traces.json"))).hexdigest(),
             "reference_py_sha256": hashlib.sha256(evidence.read_bytes((IOS / "reference.py"))).hexdigest(),
             "clips_json_sha256": hashlib.sha256(evidence.read_bytes((IOS / "clips.json"))).hexdigest(),
             "torch": torch.__version__, "seconds": round(time.time() - t0, 1)}
    evidence.write_text((out / "index.json"), json.dumps(index) + "\n")
    summary = {}
    for key in ("encoder", "token_logits", "duration_logits", "h", "c"):
        rels = [v[key]["rel"] for v in gate3.values()]
        abss = [v[key]["abs"] for v in gate3.values()]
        summary[key] = {"rel_median": float(np.median(rels)), "rel_max": float(np.max(rels)),
                        "abs_median": float(np.median(abss)), "abs_max": float(np.max(abss))}
    GATES.mkdir(parents=True, exist_ok=True)
    doc = {"gate": 3, "model": args.model, "description": "FP32 reference with FP16-rounded row scales vs FP32 row "
           "scales (same clips, same forced replay); the price of FP16 scales, separate from any encoding",
           "error_definitions": "rel = ||a-r||/max(||r||, 1e-6 sqrt(n)); abs = max|a-r|/max(RMS(r), 1e-6)",
           "summary": summary,
           "free_decode_tokens_equal_clips": sum(v["free_tokens_equal"] for v in gate3.values()),
           "clips": len(gate3), "per_clip": gate3}
    evidence.write_text((GATES / f"{args.model}-gate3.json"), json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"summary": summary, "free_equal": doc["free_decode_tokens_equal_clips"]}), flush=True)
    if args.model == "mp2" and not args.ids:
        set_scales(model, source, "fp32")
        calibrate(model, manifest, pcm_dir, out)
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({"seconds": round(time.time() - t0, 1),
                      "peak_rss_mb": round(rss / 2 ** 20 if sys.platform == "darwin" else rss / 1024)}), flush=True)


def calibrate(model, manifest, pcm_dir: Path, out: Path) -> None:
    """C5 activation scales (max |x| / 127 per site) and the probe inputs."""
    import torch

    import clips as clipmod

    naturals = [c for c in manifest["clips"] if c["kind"] == "natural"]
    chosen = []
    for b in (2, 4, 8, 15):
        chosen += sorted((c for c in naturals if c["bucket"] == b), key=lambda c: c["id"])[:4]
    maxima: dict[str, float] = {}
    captured: dict[str, np.ndarray] = {}
    probe_clip = max((c for c in naturals if c["bucket"] == 15), key=lambda c: (c["length"], c["id"]))
    current = {"probe": False}
    hooks = []
    for i, layer in enumerate(model.encoder.layers):
        for site, suffix in SITE_MODULES.items():
            module = layer.get_submodule(suffix)

            def hook(mod, inputs, i=i, site=site):
                x = inputs[0]
                key = f"layers.{i}.{site}"
                maxima[key] = max(maxima.get(key, 0.0), float(x.abs().max()))
                if current["probe"] and i in (0, len(model.encoder.layers) - 1):
                    v = x[0] if x.dim() == 3 else x
                    captured[key] = (v.transpose(0, 1) if suffix.startswith("conv.") else v).numpy().astype(np.float32)
            hooks.append(module.register_forward_pre_hook(hook))
    with torch.no_grad():
        for clip in chosen + [probe_clip]:
            current["probe"] = clip is probe_clip
            if current["probe"]:
                snapshot = dict(maxima)
            pcm = clipmod.read_pcm(pcm_dir, clip)
            feats, feat_len = model.preprocessor(torch.from_numpy(pcm.astype(np.float32))[None], torch.tensor([len(pcm)]))
            model.encoder(feats, feat_len)
    for h in hooks:
        h.remove()
    maxima = snapshot  # the probe clip is not part of the calibration set
    CALIB.mkdir(parents=True, exist_ok=True)
    doc = {"model": "mp2", "description": "C5 int8 activation calibration: per-tensor symmetric, zero point 0, "
           "scale = max|x| / 127 over the calibration clips (FP32 reference, FP32 scales, unpadded clips); x is "
           "the input of the site's ternary matmul(s) (att_in feeds q, k and v)",
           "clips": [c["id"] for c in chosen],
           "sites": {k: {"max_abs": v, "scale": v / 127.0} for k, v in sorted(maxima.items())}}
    evidence.write_text((CALIB / "mp2-c5-activations.json"), json.dumps(doc, indent=1) + "\n")
    np.savez(out / "probe_inputs.npz", **{k.replace(".", "_"): v for k, v in captured.items()})
    evidence.write_text((out / "probe_inputs.json"), json.dumps({"clip": probe_clip["id"], "keys": sorted(captured)}) + "\n")
    print(f"calibrated {len(maxima)} sites on {len(chosen)} clips; probe inputs from {probe_clip['id']}", flush=True)


IDENTITY_KEYS = ("export_sha256", "digest", "file_sha256")


def validate(model: str, build_provenance: dict | None = None, root: Path | None = None) -> dict:
    """The cache's index, after checking that it still describes this model and these inputs (review finding 8):
    clips.json, traces.json (and reference.py, when recorded) hash as when the cache was written, the model's
    current source identity (export SHA-256 / surrogate digest) equals the cached one, and, if given, the
    tested build's recorded provenance names the same weights. ValueError otherwise."""
    from .weights import Source

    root = Path(root or default_out(model))
    index = json.loads(evidence.read_text((root / "index.json")))
    problems = []
    for key, name in (("clips_json_sha256", "clips.json"), ("traces_json_sha256", "traces.json"),
                      ("reference_py_sha256", "reference.py")):
        if key in index and index[key] != hashlib.sha256(evidence.read_bytes((IOS / name))).hexdigest():
            problems.append(f"{name} changed since the cache was written")
    current = Source(model).provenance
    for prov, label in ((current, "current model source"), (build_provenance, "tested build")):
        if prov is None:
            continue
        shared = [k for k in IDENTITY_KEYS if k in prov and k in index["provenance"]]
        if not shared:
            problems.append(f"{label}: no comparable identity key")
        for k in shared:
            if prov[k] != index["provenance"][k]:
                problems.append(f"{label}: {k} {prov[k]} != cached {index['provenance'][k]}")
    if index.get("model") != model:
        problems.append(f"cache is for {index.get('model')}, not {model}")
    if problems:
        raise ValueError(f"reference cache {root} is stale: " + "; ".join(problems))
    return index


def load_clip(model: str, clip_id: str, root: Path | None = None) -> dict:
    with np.load(Path(root or default_out(model)) / f"{clip_id}.npz") as z:
        return {k: z[k] for k in z.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--model", required=True)
    p.add_argument("--pcm")
    p.add_argument("--out")
    p.add_argument("--ids", help="comma-separated clip ids (development)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
