"""Deployed-pipeline gates and records (WP7; DESIGN.md revision 8, gate 4b applied to the deployed combinations).

A deployed pipeline = front end A (vDSP) -> encoder arm (model, arm, variant, backend) -> decode loop (F2 native
FP32; F0 / F1 with the FP32 decoder models). Gated on all 82 clips, run in Swift (`parakeet-bench gate`):
- encoder: the arm's output from front end A's features (own bucket) vs the FP32 reference encoder (FP16-rounded
  scales, the weights every exact arm encodes) on the reference front end's features: rel <= 0.1 (tau 1e-3) on
  every clip, finite, encoder_length = ceil(M / 8). The reference features are the reference front end evaluated in
  FP64 (DESIGN.md rev. 5, gate 5: the FP32 evaluation is rounding noise on the silence clip, where the exact
  features are 0 and front end A returns 0); `ref64` computes that reference encoder output once (Mac). The
  comparison against WP3's refcache (reference encoder on FP32-evaluated features) is kept as a diagnostic;
- decisions: forced replay of every trace through the pipeline's decode loop; per head (token incl. blank,
  duration), steps where the reference's raw-logit top-1 margin is >= 1.0 are decisive; pooled agreement with the
  reference argmax >= 99.5% on decisive steps, >= 99% on all, >= 50% decisive;
- free decoding of the 64 natural clips: token sequences identical to the reference's on >= 61, and WER within
  +0.2 points of the reference (evaluated on NixOS, the parent experiment's scorer);
- coverage: every clip of clips.json, every trace replayed in full, every decode present; else the gate fails.
Records: ios/results/eligibility/pipelines/<model>-<arm>-<variant>-<backend>-vdsp-<decode>.json (WP3's record format
plus "kind": "pipeline", "front_end", "decode", "decoder_precision", "encoder_record" and "components": the SHA-256
and configuration of every loaded component, which parakeet-bench run recomputes and compares before timing).

  python -m pipegate ref64                             # Mac, once, inside macguard (6G): FP64-feature reference
  python -m pipegate evaluate --gate-dir DIR           # Mac, inside pipegate_job.sh (exit 10 on a failed condition)
  ./python ios/pipegate.py run [--only NAME,...]        # NixOS: every eligible encoder record x {f2, f0, f1}
  ./python ios/pipegate.py record --evaluation FILE     # NixOS: WER, records (run calls it per combination)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import sys
import time
from pathlib import Path

IOS = Path(__file__).resolve().parent
sys.path.insert(0, str(IOS))
sys.path.insert(0, str(IOS.parent))

DESIGN_REVISION = 8
CODE_VERSION = "wp7-pipegate-2"
T = {"encoder_rel": 0.1, "tau": 1e-3, "decisive_margin": 1.0, "agree_decisive": 0.995, "agree_all": 0.99,
     "decisive_min": 0.5, "identical_min": 61, "wer_points": 0.2}
DURATIONS = (0, 1, 2, 3, 4)
N_DUR = 5
DECODES = ("f2", "f0", "f1")
MAC_IOS = "/Users/ajbarry/workspace/github.com/wilderness-labs-stt/finetune/parakeet-ternary/ios"
MAC_A = "/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios"
LOCAL = Path("/mnt/hd/wilderness-labs-stt/parakeet-ios/results/pipegate")
SUMMARIES = IOS / "results" / "pipegates"
RECORDS = IOS / "results" / "eligibility" / "pipelines"
EXCLUDED = {"G0": "graph control with C0's weights (not deployable; WP3 control record)",
            "MLX": "no Swift implementation (WP6b prototype)"}


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- ref64 (Mac) -------------------------------------------------------------------------------------------------

def ref64_dir(model: str) -> Path:
    import artifacts

    return artifacts.root() / "refcache" / f"{model}-ref64"


def cmd_ref64(args) -> int:
    """The FP32 reference encoder (FP16-rounded scales, as WP3's refcache fp16s variant) on the FP64 evaluation of
    the reference front end (native.fp64_features, the stored constants), every clip: <dir>/<id>.npy [1024, E]."""
    import numpy as np
    import torch

    import clips as clipmod
    from mil import refcache
    from mil.weights import Source
    from native import fp64_features, load_blob

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
        rows[clip["id"]] = {"kind": clip["kind"], "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                            "fp32_vs_fp64_features_rel_abs": [float(x) for x in refcache.errors(old["features"], f64, 1e-6)],
                            "enc_refcache_vs_ref64_rel_abs": [float(x) for x in refcache.errors(old["fp16s_enc"], a, T["tau"])]}
    doc = {"model": args.model, "scales": "fp16s", "features": "native.fp64_features (FP64 reference front end), cast to FP32",
           "frontend_manifest_sha256": sha(consts_dir / "frontend.json"), "refcache_provenance": index["provenance"],
           "model_provenance": source.provenance, "clips_json_sha256": sha(IOS / "clips.json"),
           "pipegate_py_sha256": sha(Path(__file__)), "torch": torch.__version__, "seconds": round(time.time() - t0, 1),
           "clips": rows}
    (out / "index.json").write_text(json.dumps(doc, indent=1) + "\n")
    worst = sorted(rows.items(), key=lambda kv: -kv[1]["enc_refcache_vs_ref64_rel_abs"][0])[:3]
    print(json.dumps({"clips": len(rows), "seconds": doc["seconds"],
                      "largest refcache-vs-ref64 encoder differences": {k: v["enc_refcache_vs_ref64_rel_abs"] for k, v in worst}}))
    return 0


# --- evaluate (Mac) ----------------------------------------------------------------------------------------------

def cmd_evaluate(args) -> int:
    import numpy as np

    from mil import refcache

    d = Path(args.gate_dir)
    lines = [json.loads(l) for l in (d / "gate.jsonl").read_text().splitlines() if l.strip()]
    header = next(l for l in lines if l.get("record") == "header")
    rows = {l["clip"]: l for l in lines if l.get("record") == "clip"}
    clips = {c["id"]: c for c in json.loads((IOS / "clips.json").read_text())["clips"]}
    traces = {t["id"]: t for t in json.loads((IOS / "traces.json").read_text())["clips"]}
    index = refcache.validate(header["model"])
    r64dir = ref64_dir(header["model"])
    r64 = json.loads((r64dir / "index.json").read_text())
    problems = []
    if r64["clips_json_sha256"] != sha(IOS / "clips.json") or r64["refcache_provenance"] != index["provenance"]:
        problems.append("FP64-feature reference is stale (clips.json or model provenance changed)")
    if sorted(rows) != sorted(clips) or sorted(header["clip_ids"]) != sorted(clips):
        problems.append(f"coverage: {len(rows)} of {len(clips)} clips")
    if header["clips_json_sha256"] != sha(IOS / "clips.json"):
        problems.append("clips.json changed since the gate run")
    enc_rows = []
    for cid, r in rows.items():
        ref = refcache.load_clip(header["model"], cid)
        a = np.fromfile(d / "enc" / f"{cid}.f32", "<f4").reshape(-1, 1024).T
        if hashlib.sha256(a.T.astype("<f4").tobytes()).hexdigest() != r["enc_sha256"]:
            problems.append(f"{cid}: encoder output file differs from the gate record")
        e = clips[cid]["encoder_frames"]
        ok_len = r["encoder_length"] == e == a.shape[1]
        ref64 = np.load(r64dir / f"{cid}.npy")
        if hashlib.sha256(ref64.astype("<f4").tobytes()).hexdigest() != r64["clips"][cid]["sha256"]:
            problems.append(f"{cid}: FP64-feature reference file differs from its index")
        rel, ab = refcache.errors(a, ref64, T["tau"]) if ok_len else (math.inf, math.inf)
        rel32, ab32 = refcache.errors(a, ref["fp16s_enc"], T["tau"]) if ok_len else (math.inf, math.inf)
        enc_rows.append({"clip": cid, "bucket": r["bucket"], "rel": rel, "abs": ab, "length_ok": ok_len, "finite": r["finite"],
                         "pass": ok_len and r["finite"] and rel <= T["encoder_rel"],
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
        tok = {"all": 0, "ok": 0, "dec": 0, "dec_ok": 0}
        dur = {"all": 0, "ok": 0, "dec": 0, "dec_ok": 0}
        missing, free_rows = [], {}
        for cid, r in rows.items():
            x = r.get(dec)
            if x is None or "replay_error" in x or x.get("replay_steps") != traces[cid]["steps"]:
                missing.append(cid)
                continue
            logits = refcache.load_clip(header["model"], cid)["fp16s_logits"]
            for head, at, sl, vals in ((tok, x["argmax_token"], slice(0, -N_DUR), None), (dur, x["argmax_duration"], slice(-N_DUR, None), DURATIONS)):
                lg = logits[:, sl]
                srt = np.sort(lg, axis=1)
                margin = srt[:, -1] - srt[:, -2]
                ref_arg = lg.argmax(1) if vals is None else np.asarray(vals)[lg.argmax(1)]
                agree = np.asarray(at) == ref_arg
                decisive = margin >= T["decisive_margin"]
                head["all"] += len(agree); head["ok"] += int(agree.sum())
                head["dec"] += int(decisive.sum()); head["dec_ok"] += int(agree[decisive].sum())
            if clips[cid]["kind"] == "natural":
                free_rows[cid] = x["tokens"]
        heads = {}
        for name, h in (("token", tok), ("duration", dur)):
            heads[name] = {"steps": h["all"], "agree_all": h["ok"] / max(h["all"], 1), "decisive_fraction": h["dec"] / max(h["all"], 1),
                           "agree_decisive": h["dec_ok"] / max(h["dec"], 1)}
            heads[name]["pass"] = (heads[name]["agree_decisive"] >= T["agree_decisive"] and heads[name]["agree_all"] >= T["agree_all"]
                                   and heads[name]["decisive_fraction"] >= T["decisive_min"])
        ref_tokens = {c: index["free_tokens"][c]["fp16s_free_tokens"] for c in free_rows}
        identical = sum(free_rows[c] == ref_tokens[c] for c in free_rows)
        n_nat = sum(1 for c in clips.values() if c["kind"] == "natural")
        per_decode[dec] = {
            "coverage": {"pass": not missing and len(free_rows) == n_nat, "missing_or_incomplete": missing},
            "decisions": {"pass": all(h["pass"] for h in heads.values()), **heads},
            "free_decoding": {"pass": identical >= T["identical_min"] and len(free_rows) == n_nat, "clips": len(free_rows),
                              "identical": identical},
            "free_tokens": free_rows, "reference_tokens": ref_tokens}
        per_decode[dec]["pass_before_wer"] = (encoder["pass"] and not problems and per_decode[dec]["coverage"]["pass"]
                                             and per_decode[dec]["decisions"]["pass"] and per_decode[dec]["free_decoding"]["pass"])
    doc = {"design_revision": DESIGN_REVISION, "code_version": CODE_VERSION, "thresholds": T, "header": header,
           "refcache_provenance": index["provenance"], "ref64_index_sha256": sha(r64dir / "index.json"), "problems": problems, "encoder": encoder, "encoder_rows": enc_rows,
           "decodes": per_decode, "pipegate_py_sha256": sha(Path(__file__)), "evaluated": time.strftime("%Y-%m-%d %H:%M")}
    (d / "evaluation.json").write_text(json.dumps(doc, indent=1) + "\n")
    ok = all(v["pass_before_wer"] for v in per_decode.values())
    print(json.dumps({"encoder": {k: encoder[k] for k in ("pass", "passed", "max_rel")}, "problems": problems,
                      **{k: {"pass": v["pass_before_wer"], "identical": v["free_decoding"]["identical"],
                             "token_decisive": round(v["decisions"]["token"]["agree_decisive"], 5),
                             "duration_decisive": round(v["decisions"]["duration"]["agree_decisive"], 5)}
                         for k, v in per_decode.items()}}))
    return 0 if ok else 10


# --- record (NixOS) --------------------------------------------------------------------------------------------------

def cmd_record(args) -> int:
    from mil.eligibility import _tokenizer, _wer

    ev = json.loads(Path(args.evaluation).read_text())
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
    summary = {k: ev[k] for k in ("design_revision", "code_version", "thresholds", "header", "refcache_provenance",
                                  "ref64_index_sha256", "problems",
                                  "encoder", "evaluated", "pipegate_py_sha256")}
    summary["decodes"] = {dec: {k: v[k] for k in ("coverage", "decisions", "free_decoding")} | {"wer": wers[dec]}
                          for dec, v in ev["decodes"].items()}
    summary["encoder_rows"] = ev["encoder_rows"]
    summary_path.write_text(json.dumps(summary, indent=1) + "\n")
    out = {}
    for dec, v in summary["decodes"].items():
        pipe = h["pipelines"][dec]
        comps = pipe["components"]
        precision = comps["decode"].get("precision")
        checks = {"encoder_record": {"pass": bool(enc_rec.get("timing_allowed")) and enc_rec.get("design_revision") == DESIGN_REVISION,
                                     "record": enc_rec_path.name},
                  "decoder_precision": {"pass": precision == "fp32", "precision": precision},
                  "problems": {"pass": not ev["problems"], "list": ev["problems"]},
                  "encoder": ev["encoder"], "coverage": v["coverage"], "decisions": v["decisions"],
                  "free_decoding": v["free_decoding"], "wer": v["wer"]}
        reasons = [k for k, c in checks.items() if not c["pass"]]
        rec = {"design_revision": DESIGN_REVISION, "code_version": CODE_VERSION, "kind": "pipeline",
               "model": h["model"], "arm": h["arm"], "variant": h["variant"], "backend": backend, "compute_units": h["compute_units"],
               "front_end": "vdsp", "decode": dec, "decoder_precision": precision,
               "encoder_record": {"file": enc_rec_path.name, "sha256": sha(enc_rec_path)},
               "components": comps, "eligible": not reasons, "timing_allowed": not reasons,
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
        if r.get("design_revision") != DESIGN_REVISION or not r.get("timing_allowed") or r.get("kind") == "pipeline":
            continue
        if r.get("decoder_precision") == "fp16" or r["arm"] in EXCLUDED:
            continue
        out.append(r)
    return out


def combo_name(r: dict) -> str:
    return f"{r['model']}-{r['arm']}-{r['variant']}-{r['backend']}"


def mac(cmd: str, timeout: int = 900) -> str:
    import subprocess

    from mil.archive import local_config

    res = subprocess.run([*local_config("WP3_MAC_RUN").split(), "--repo", MAC_IOS, "--", "sh", "-c", cmd],
                         capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(f"Mac command failed ({res.returncode}): {cmd[:120]}: {res.stderr.strip()[-300:]}")
    return res.stdout


def disk_need_gb(r: dict) -> int:
    """Measured headroom: the package's size on disk (measured on the Mac) plus its Core ML cache estimate: WP5
    measured 4.6 GB for C1 multi (dense FP16, four functions), 18-54 MB for every compressed arm; dense C1 gets
    4 x its package size, every other arm 1 GB."""
    pkg = f"{MAC_A}/arms/{r['model']}/{r['arm']}/{r['variant']}.mlmodelc"
    kb = mac(f"du -sk {pkg} | cut -f1").strip()
    if not kb.isdigit():
        raise RuntimeError(f"cannot measure {pkg}")
    gb = int(kb) / 2 ** 20
    cache = 4 * gb if r["arm"] == "C1" else 1.0
    return math.ceil(gb + cache + 1)


def cmd_run(args) -> int:
    import subprocess

    from wp5sweep import ensure_model

    only = set(args.only.split(",")) if args.only else None
    recs = [r for r in encoder_records() if not only or combo_name(r) in only]
    restored: set = set()
    log = []
    for i, r in enumerate(recs):
        name = combo_name(r)
        a = {"model": r["model"], "arm": r["arm"], "variant": r["variant"]}
        ensure_model(a, restored)
        need = disk_need_gb(r)
        run_id = time.strftime("%Y%m%d-%H%M%S")
        out = f"{MAC_A}/results/pipegate/{name}/{run_id}"
        units = r["compute_units"]
        gate = ["--eligibility", f"{r['model']}:{r['arm']}", "--encoder", f"{MAC_A}/arms/{r['model']}/{r['arm']}/{r['variant']}.mlmodelc",
                "--encoder-variant", {"fixed": "fixed15", "multi": "multifunction", "enum": "enumerated"}[r["variant"]],
                "--compute-units", units, "--frontend-constants", f"{MAC_A}/native/mp2", "--native-weights", f"{MAC_A}/native/mp2",
                "--decoder-models", f"{MAC_A}/arms/mp2/decoder-fp32", "--decodes", ",".join(DECODES),
                "--clips", f"{MAC_IOS}/clips.json", "--pcm", f"{MAC_A}/clips", "--traces", f"{MAC_IOS}/traces.json"]
        cap = "6G" if r["arm"] == "C1" else "4G"
        q = " ".join(shlex.quote(x) for x in gate)
        line = (f"mkdir -p {out}; i=0; while :; do ./macguard --rss-cap {cap} --timeout 5400 -- sh pipegate_job.sh {need} {out} -- {q} "
                f"> {out}/job.log 2>&1; s=$?; [ $s -ne 3 ] && break; i=$((i+1)); [ $i -gt 120 ] && break; sleep 60; done; "
                f"echo $s > {out}/STATUS")
        mac(f"nohup sh -c {shlex.quote(line)} > /dev/null 2>&1 < /dev/null & echo started")
        t0 = time.time()
        status = ""
        while not status and time.time() - t0 < 6 * 3600:  # 120 lock retries + the 5400 s job, with margin
            time.sleep(60)
            try:
                status = mac(f"cat {out}/STATUS 2>/dev/null || true").strip()
            except Exception as exc:  # a transient SSH failure must not abandon a running job
                print(f"poll: {exc}", file=sys.stderr, flush=True)
        if not status.lstrip("-").isdigit():
            raise RuntimeError(f"{name}: no status from {out} (job still running or lost); stopping")
        entry = {"combo": name, "status": int(status), "minutes": round((time.time() - t0) / 60, 1), "run": run_id, "need_gb": need, "cap": cap}
        if int(status) in (0, 10):
            # complete retrieval into a temporary directory, then an atomic rename (no partial or stale results)
            final = LOCAL / name / run_id
            tmp = LOCAL / name / f".{run_id}.part"
            tmp.mkdir(parents=True, exist_ok=False)
            for f in ("evaluation.json", "gate.jsonl", "job.log", "cache.log"):
                (tmp / f).write_text(mac(f"cat {out}/{f}"))
            json.loads((tmp / "evaluation.json").read_text())  # must parse
            os.replace(tmp, final)
            (LOCAL / name / "LATEST").write_text(run_id + "\n")
            res = subprocess.run([str(IOS.parent / "python"), str(Path(__file__)), "record", "--evaluation", str(final / "evaluation.json")],
                                 capture_output=True, text=True)
            entry["record"] = res.stdout.strip()[-400:] or res.stderr.strip()[-400:]
        log.append(entry)
        print(json.dumps(entry), flush=True)
        later = {(b["model"], b["arm"]) for b in recs[i + 1:]}
        for key in sorted(restored - later):
            subprocess.run([str(IOS.parent / "python"), str(IOS / "mil" / "archive.py"), "out", *key], check=True)
            restored.discard(key)
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / f"run-{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(log, indent=1) + "\n")
    return 0


def cmd_table(args) -> int:
    rows = []
    for p in sorted(RECORDS.glob("*.json")):
        r = json.loads(p.read_text())
        c = r["checks"]
        rows.append(f"| {r['model']} | {r['arm']} | {r['variant']} | {r['backend']} | {r['decode']} | "
                    f"{'yes' if r['timing_allowed'] else 'no'} | {', '.join(r['reasons']) or '-'} | "
                    f"{c['encoder'].get('max_rel', float('nan')):.4f} | {c['free_decoding']['identical']}/64 | "
                    f"{c['decisions']['token']['agree_decisive']:.4f} / {c['decisions']['duration']['agree_decisive']:.4f} | "
                    f"{c['wer']['pipeline_wer_pct']:.3f} / {c['wer']['reference_wer_pct']:.3f} |")
    text = ("Deployed-pipeline records (front end A + encoder arm + decode loop; DESIGN.md revision 8 gate 4b conditions).\n\n"
            "| model | arm | variant | backend | decode | timing allowed | failed | encoder rel max | identical | "
            "decisive agreement token / duration | WER pipeline / ref (%) |\n|---|---|---|---|---|---|---|---|---|---|---|\n"
            + "\n".join(rows) + "\n")
    (RECORDS / "table.txt").write_text(text)
    print(text)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ref64"); p.add_argument("--model", default="mp2")
    p.add_argument("--frontend", default="/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios/native/mp2")
    p.set_defaults(func=cmd_ref64)
    p = sub.add_parser("evaluate"); p.add_argument("--gate-dir", required=True); p.set_defaults(func=cmd_evaluate)
    p = sub.add_parser("record"); p.add_argument("--evaluation", required=True); p.set_defaults(func=cmd_record)
    p = sub.add_parser("run"); p.add_argument("--only"); p.set_defaults(func=cmd_run)
    sub.add_parser("table").set_defaults(func=cmd_table)
    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
