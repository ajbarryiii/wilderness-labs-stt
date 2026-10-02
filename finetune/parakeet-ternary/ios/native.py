"""Native-arm data: front-end constants, decoder/joint weights, reference outputs, and the gates (WP4).

DESIGN.md arms A (front end on the CPU with Accelerate/vDSP) and F2 (native CPU decode loop). Files are
little-endian float32 blobs with a JSON manifest (shapes, byte offsets, per-tensor and file SHA-256), written
only under the machine's artifact area (artifacts.check).

  python native.py frontend --model mp2 --out DIR      # window [400], fb [128, 257] as the model stores them
  python native.py weights --model mp2 --out DIR       # decoder + joint, FP32 (and an exact FP16 copy if possible)
  python native.py widen --dir DIR                     # FP16 copy -> FP32 blob, checked against the manifest
  python native.py reference --model mp2 --out DIR [--kinds natural]
        # per clip: the reference's encoder output (enc/<id>.f32 [T, 1024], the F2 gate's input), its replay of
        # the B0 trace (logits [S, 1030], h, c [S, 2, 640]) and its own greedy decode (tokens) -> ref/<id>.npz
  python native.py gate-frontend --frontend DIR --swift DIR [--clips-dir PCM]
        # parakeet-bench `features` output vs reference.Featurizer with the same constants (DESIGN.md errors)
  python native.py gate-f2 --ref DIR --results JSONL --diag DIR
        # parakeet-bench F2 diagnostic replay vs ref/<id>.npz; F2 free decoding vs the reference's greedy tokens

Models: "mp2" (the pilot P2 export via reference.ExportSource: the same tensors models.mp2() loads, read
lazily), "seed0".."seed2" (randomweights.generate, as models.surrogate), "b0" (models.b0's checkpoint).
`reference` loads the full model with models.load() (NixOS: through ../heavy, about 4 GB).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FRONTEND_KEYS = ("preprocessor.featurizer.window", "preprocessor.featurizer.fb")
DECODER_KEYS = (  # native order; shapes are NeMo's
    "decoder.prediction.embed.weight",
    "decoder.prediction.dec_rnn.lstm.weight_ih_l0", "decoder.prediction.dec_rnn.lstm.weight_hh_l0",
    "decoder.prediction.dec_rnn.lstm.bias_ih_l0", "decoder.prediction.dec_rnn.lstm.bias_hh_l0",
    "decoder.prediction.dec_rnn.lstm.weight_ih_l1", "decoder.prediction.dec_rnn.lstm.weight_hh_l1",
    "decoder.prediction.dec_rnn.lstm.bias_ih_l1", "decoder.prediction.dec_rnn.lstm.bias_hh_l1",
    "joint.enc.weight", "joint.enc.bias", "joint.pred.weight", "joint.pred.bias",
    "joint.joint_net.2.weight", "joint.joint_net.2.bias",
)
REL_CEILING, ABS_CEILING, TAU = 1e-5, 1e-4, 1e-6


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def errors(a: np.ndarray, r: np.ndarray, tau: float = TAU) -> tuple[float, float]:
    """DESIGN.md: rel = ||a - r|| / max(||r||, tau sqrt(n)); abs = max|a - r| / max(RMS(r), tau)."""
    a, r = np.asarray(a, np.float64), np.asarray(r, np.float64)
    if a.shape != r.shape:
        raise ValueError(f"shape {a.shape} != {r.shape}")
    if r.size == 0:
        return 0.0, 0.0
    d = a - r
    rel = float(np.linalg.norm(d) / max(float(np.linalg.norm(r)), tau * r.size ** 0.5))
    return rel, float(np.abs(d).max() / max(float(np.sqrt(np.mean(r ** 2))), tau))


# --- tensors of a named model ----------------------------------------------------------------------------

def source_tensors(model: str, keys) -> tuple[dict[str, np.ndarray], dict]:
    """The named model's tensors `keys` as stored (dtype kept), plus provenance; no full model is built."""
    keys = set(keys)
    if model == "mp2":
        import models
        import reference

        src = reference.ExportSource(models._defaults()["mp2"])
        raw = {}
        for k in keys:  # stored dtype (FP16 or FP32); ExportSource would widen to FP32
            raw[k] = src._file.get_tensor(k).numpy()
        return raw, {"model": "mp2", "export_dir": str(src.dir), "export_sha256": src.manifest["sha256"]}
    if model.startswith("seed"):
        import randomweights as rw

        stats, seed, out = rw.load_stats(), int(model[4:]), {}
        for name, value in rw.generate(stats, seed):
            if name in keys:
                out[name] = np.asarray(value)
        return out, {"model": model, "seed": seed, "weight_stats_sha256": stats["_sha256"]}
    if model == "b0":
        import models
        import torch

        m = models.b0()
        state = m.state_dict()
        return ({k: state[k].numpy() for k in keys}, {"model": "b0", **{k: v for k, v in m.provenance.items() if k != "load"}})
    raise KeyError(model)


def write_blob(out: Path, stem: str, tensors: list[tuple[str, np.ndarray]], meta: dict) -> dict:
    """FP32 blob <stem>.f32bin + <stem>.json; also an FP16 copy <stem>.f16bin if every value is FP16-exact."""
    import artifacts

    out = artifacts.check(out)
    out.mkdir(parents=True, exist_ok=True)
    entries, chunks, offset = [], [], 0
    exact16 = True
    for name, value in tensors:
        v32 = np.ascontiguousarray(value, dtype="<f4")
        exact16 &= bool(np.array_equal(v32.astype("<f2").astype("<f4"), v32))
        data = v32.tobytes()
        entries.append({"name": name, "shape": list(v32.shape), "offset": offset, "bytes": len(data), "sha256": sha256(data)})
        chunks.append(data)
        offset += len(data)
    blob = b"".join(chunks)
    (out / f"{stem}.f32bin").write_bytes(blob)
    manifest = {**meta, "dtype": "float32 little-endian", "file": f"{stem}.f32bin", "bytes": len(blob),
                "sha256": sha256(blob), "tensors": entries, "fp16_exact": exact16}
    if exact16:
        half = b"".join(np.ascontiguousarray(t, dtype="<f4").astype("<f2").tobytes() for _, t in tensors)
        (out / f"{stem}.f16bin").write_bytes(half)
        manifest["fp16_copy"] = {"file": f"{stem}.f16bin", "bytes": len(half), "sha256": sha256(half),
                                 "note": "exact: every FP32 value is FP16-representable; `widen` rebuilds the FP32 blob"}
    (out / f"{stem}.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def cmd_frontend(args) -> None:
    import reference

    raw, prov = source_tensors(args.model, FRONTEND_KEYS)
    cfg = reference.Config()
    window = raw["preprocessor.featurizer.window"].astype(np.float32).reshape(-1)
    fb = raw["preprocessor.featurizer.fb"].astype(np.float32).reshape(cfg.features, cfg.n_fft // 2 + 1)
    meta = {"kind": "frontend", "provenance": prov, "config": {
        "sample_rate": cfg.sample_rate, "n_fft": cfg.n_fft, "hop_length": cfg.hop_length, "win_length": cfg.win_length,
        "features": cfg.features, "preemph": cfg.preemph, "log_guard": cfg.log_guard, "norm_constant": reference.NORM_CONSTANT,
        "pad_value": cfg.pad_value, "window": "stored symmetric Hann [400], centred in n_fft (offset 56)",
        "fb": "stored Slaney mel filterbank [128, 257]",
        "differs_from_computed": {"window_max_abs": float(np.abs(window - reference.hann_window(cfg.win_length)).max()),
                                  "fb_max_abs": float(np.abs(fb - reference.mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.features)).max())}}}
    m = write_blob(Path(args.out), "frontend", [("window", window), ("fb", fb)], meta)
    print(json.dumps({k: m[k] for k in ("file", "bytes", "sha256")}))


def cmd_weights(args) -> None:
    raw, prov = source_tensors(args.model, DECODER_KEYS)
    tensors = [(k, raw[k]) for k in DECODER_KEYS]
    stored = sorted({str(raw[k].dtype) for k in DECODER_KEYS})
    import reference

    cfg = reference.Config()
    meta = {"kind": "native-decoder-joint", "provenance": prov, "stored_dtypes": stored, "config": {
        "vocab_size": cfg.vocab_size, "blank": cfg.blank, "pred_hidden": cfg.pred_hidden, "pred_rnn_layers": cfg.pred_rnn_layers,
        "joint_hidden": cfg.joint_hidden, "d_model": cfg.d_model, "num_outputs": cfg.num_outputs,
        "durations": list(cfg.durations), "max_symbols": cfg.max_symbols, "lstm_gate_order": "i, f, g, o (PyTorch)"}}
    m = write_blob(Path(args.out), "decoder_joint", tensors, meta)
    print(json.dumps({k: m[k] for k in ("file", "bytes", "sha256", "fp16_exact")}))


def cmd_widen(args) -> None:
    """Rebuild the FP32 blob from its exact FP16 copy and check every SHA-256 (after a transfer)."""
    import artifacts

    d = artifacts.check(args.dir)
    for manifest_path in sorted(d.glob("*.json")):
        m = json.loads(manifest_path.read_text())
        if "fp16_copy" not in m:
            continue
        half = (d / m["fp16_copy"]["file"]).read_bytes()
        if sha256(half) != m["fp16_copy"]["sha256"]:
            raise ValueError(f"{m['fp16_copy']['file']}: SHA-256 mismatch")
        full = np.frombuffer(half, "<f2").astype("<f4").tobytes()
        if sha256(full) != m["sha256"]:
            raise ValueError(f"widened {m['file']} does not match the manifest")
        (d / m["file"]).write_bytes(full)
        print(json.dumps({"file": m["file"], "sha256_ok": True, "bytes": len(full)}))


# --- reference outputs (NixOS, heavy) ------------------------------------------------------------------------

def cmd_reference(args) -> None:
    import artifacts
    import torch

    import clips as clipmod
    import models
    import reference

    torch.set_grad_enabled(False)
    out = artifacts.check(args.out)
    (out / "enc").mkdir(parents=True, exist_ok=True)
    (out / "ref").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = models.load(args.model)
    raw, _ = source_tensors(args.model, DECODER_KEYS)
    state = model.state_dict()
    same = all(np.array_equal(state[k].numpy(), raw[k].astype(np.float32)) for k in DECODER_KEYS)
    manifest = clipmod.load_manifest()
    traces = {t["id"]: t for t in json.loads((HERE / "traces.json").read_text())["clips"]}
    import traces as tracemod

    kinds = set(args.kinds.split(","))
    index = {"model": args.model, "provenance": {k: v for k, v in model.provenance.items() if k != "load"},
             "decoder_joint_equal_to_weights_export": same, "clips": {}}
    for clip in manifest["clips"]:
        if clip["kind"] not in kinds:
            continue
        pcm = clipmod.read_pcm(Path(args.pcm), clip)
        if clipmod.pcm_sha256(pcm) != clip["sha256"]:
            raise ValueError(clip["id"])
        enc, enc_len = model(torch.from_numpy(pcm.astype(np.float32))[None], torch.tensor([len(pcm)]))
        length = int(enc_len[0])
        frames = enc[0, :, :length].T.contiguous().numpy().astype("<f4")  # [T, 1024]
        data = frames.tobytes()
        (out / "enc" / f"{clip['id']}.f32").write_bytes(data)
        trace = tracemod.record_to_trace(traces[clip["id"]])
        steps = reference.replay(model, enc, length, trace)
        greedy = reference.greedy_decode(model, enc, enc_len, record=True)[0]
        g_trace, g_steps = greedy
        np.savez(out / "ref" / f"{clip['id']}.npz", logits=steps.logits.numpy(), h=steps.h.numpy(), c=steps.c.numpy(),
                 argmax_token=steps.argmax_token.numpy(), argmax_duration=steps.argmax_duration.numpy(),
                 greedy_tokens=np.asarray(g_trace.tokens, np.int64), greedy_steps=len(g_trace))
        index["clips"][clip["id"]] = {"frames": length, "enc_sha256": sha256(data), "replay_steps": len(trace),
                                      "greedy_tokens": len(g_trace.tokens)}
    (out / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    peak = __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss / 1024
    print(json.dumps({"clips": len(index["clips"]), "decoder_joint_equal": same, "seconds": round(time.time() - t0, 1),
                      "peak_rss_mb": round(peak)}))


# --- gates ---------------------------------------------------------------------------------------------------

def load_blob(d: Path, stem: str) -> dict[str, np.ndarray]:
    m = json.loads((d / f"{stem}.json").read_text())
    blob = (d / m["file"]).read_bytes()
    if sha256(blob) != m["sha256"]:
        raise ValueError(f"{m['file']}: SHA-256 mismatch")
    return {t["name"]: np.frombuffer(blob, "<f4", count=int(np.prod(t["shape"])), offset=t["offset"]).reshape(t["shape"])
            for t in m["tensors"]}


def cmd_gate_frontend(args) -> None:
    """Swift features (<id>.mel.f32 [128, frames] + features.jsonl) vs reference.Featurizer, same constants."""
    import torch

    import clips as clipmod
    import reference

    consts = load_blob(Path(args.frontend), "frontend")
    cfg = reference.Config()
    feat = reference.Featurizer(cfg)
    feat.window = torch.from_numpy(consts["window"].copy())
    feat.fb = torch.from_numpy(consts["fb"].copy())[None]
    sw = Path(args.swift)
    records = {json.loads(l)["clip"]: json.loads(l) for l in (sw / "features.jsonl").read_text().splitlines() if l.strip()}
    rows, worst = [], None
    for clip in clipmod.load_manifest()["clips"]:
        if clip["id"] not in records:
            continue
        pcm = clipmod.read_pcm(Path(args.pcm), clip)
        with torch.no_grad():
            ref, ref_len = feat(torch.from_numpy(pcm.astype(np.float32))[None], torch.tensor([len(pcm)]))
        ref = ref[0].numpy()
        r = records[clip["id"]]
        mine = np.fromfile(sw / f"{clip['id']}.mel.f32", "<f4").reshape(128, -1)
        rel, ab = errors(mine, ref)
        row = {"clip": clip["id"], "kind": clip["kind"], "frames": int(ref.shape[1]), "mel_length": r["mel_length"],
               "mel_length_ok": r["mel_length"] == int(ref_len[0]), "rel": rel, "abs": ab,
               "pass": rel <= REL_CEILING and ab <= ABS_CEILING and r["mel_length"] == int(ref_len[0])
               and bool(np.isfinite(mine).all())}
        rows.append(row)
        if worst is None or rel > worst["rel"]:
            worst = row
    summary = {"gate": "front end A (vDSP) vs reference.Featurizer, same stored constants",
               "ceilings": {"rel": REL_CEILING, "abs": ABS_CEILING}, "clips": len(rows),
               "passed": sum(r["pass"] for r in rows), "max_rel": max(r["rel"] for r in rows),
               "max_abs": max(r["abs"] for r in rows), "median_rel": float(np.median([r["rel"] for r in rows])),
               "worst": worst, "failures": [r for r in rows if not r["pass"]]}
    print(json.dumps(summary, indent=1))
    if args.out:
        import artifacts
        dest = Path(args.out).resolve()
        if not dest.is_relative_to((HERE / "results").resolve()):
            dest = artifacts.check(dest)
        dest.write_text(json.dumps({**summary, "rows": rows}, indent=1) + "\n")


def cmd_gate_f2(args) -> None:
    """F2 diagnostic replay (sections logits, h, c) vs the reference's replay, and F2 free tokens vs greedy."""
    ref_dir, diag_dir = Path(args.ref), Path(args.diag)
    lines = [json.loads(l) for l in Path(args.results).read_text().splitlines() if l.strip()]
    diags = [l for l in lines if l.get("record") == "diagnostic"]
    rows = []
    for d in diags:
        ref = np.load(ref_dir / "ref" / f"{d['clip']}.npz")
        blob = (diag_dir / d["arrays"]["file"]).read_bytes()
        if sha256(blob) != d["arrays"]["sha256"]:
            raise ValueError(d["arrays"]["file"])
        sec = {s["name"]: np.frombuffer(blob, "<f4", count=int(np.prod(s["shape"])), offset=s["offset"]).reshape(s["shape"])
               for s in d["arrays"]["sections"]}
        row = {"clip": d["clip"], "mode": d["mode"], "steps": int(ref["logits"].shape[0])}
        if d["mode"] == "replay":
            logits = sec["logits"]
            row["logits"] = errors(logits, ref["logits"])
            row["token_logits"] = errors(logits[:, :1025], ref["logits"][:, :1025])
            row["duration_logits"] = errors(logits[:, 1025:], ref["logits"][:, 1025:])
            row["h"] = errors(sec["h"].reshape(ref["h"].shape), ref["h"])
            row["c"] = errors(sec["c"].reshape(ref["c"].shape), ref["c"])
            row["argmax_token_equal"] = bool(np.array_equal(np.asarray(d["steps"]["argmax_token"]), ref["argmax_token"]))
            row["argmax_duration_equal"] = bool(np.array_equal(np.asarray(d["steps"]["argmax_duration"]), ref["argmax_duration"]))
            row["argmax_token_mismatches"] = int((np.asarray(d["steps"]["argmax_token"]) != ref["argmax_token"]).sum())
            row["pass"] = all(row[k][0] <= REL_CEILING and row[k][1] <= ABS_CEILING
                              for k in ("token_logits", "duration_logits", "h", "c")) and bool(np.isfinite(logits).all())
        rows.append(row)
    free = {l["clip"]: l for l in lines if "result" in l and l.get("mode") == "free" and not l["warmup"]}
    free_rows = []
    for cid, l in free.items():
        ref = np.load(ref_dir / "ref" / f"{cid}.npz")
        free_rows.append({"clip": cid, "tokens_equal": l["result"]["tokens"] == ref["greedy_tokens"].tolist(),
                          "f2_tokens": len(l["result"]["tokens"]), "ref_tokens": int(ref["greedy_tokens"].size)})
    rep = [r for r in rows if r["mode"] == "replay"]
    summary = {"gate": "F2 (native CPU decode loop, FP32) vs reference replay of the B0 trace, same encoder output",
               "ceilings": {"rel": REL_CEILING, "abs": ABS_CEILING}, "replay_clips": len(rep),
               "replay_passed": sum(r["pass"] for r in rep)}
    for k in ("logits", "token_logits", "duration_logits", "h", "c"):
        if rep:
            summary[f"max_{k}"] = {"rel": max(r[k][0] for r in rep), "abs": max(r[k][1] for r in rep)}
    if rep:
        summary["argmax_equal_all_clips"] = {"token": all(r["argmax_token_equal"] for r in rep),
                                             "duration": all(r["argmax_duration_equal"] for r in rep)}
    if free_rows:
        summary["free_decoding"] = {"clips": len(free_rows), "token_sequences_equal": sum(r["tokens_equal"] for r in free_rows),
                                    "differing": [r for r in free_rows if not r["tokens_equal"]]}
    print(json.dumps(summary, indent=1))
    if args.out:
        dest = Path(args.out).resolve()
        if not dest.is_relative_to((HERE / "results").resolve()):
            import artifacts
            dest = artifacts.check(dest)
        dest.write_text(json.dumps({**summary, "rows": rows, "free_rows": free_rows}, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in (("frontend", cmd_frontend), ("weights", cmd_weights)):
        p = sub.add_parser(name); p.add_argument("--model", default="mp2"); p.add_argument("--out", required=True)
        p.set_defaults(func=fn)
    p = sub.add_parser("widen"); p.add_argument("--dir", required=True); p.set_defaults(func=cmd_widen)
    p = sub.add_parser("reference"); p.add_argument("--model", default="mp2"); p.add_argument("--out", required=True)
    p.add_argument("--pcm", default="/mnt/hd/wilderness-labs-stt/parakeet-ios/clips"); p.add_argument("--kinds", default="natural")
    p.set_defaults(func=cmd_reference)
    p = sub.add_parser("gate-frontend"); p.add_argument("--frontend", required=True); p.add_argument("--swift", required=True)
    p.add_argument("--pcm", required=True); p.add_argument("--out"); p.set_defaults(func=cmd_gate_frontend)
    p = sub.add_parser("gate-f2"); p.add_argument("--ref", required=True); p.add_argument("--results", required=True)
    p.add_argument("--diag", required=True); p.add_argument("--out"); p.set_defaults(func=cmd_gate_f2)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
