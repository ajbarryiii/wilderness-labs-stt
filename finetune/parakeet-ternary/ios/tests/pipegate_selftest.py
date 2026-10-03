"""Self-test of `pipegate.py evaluate` on synthetic gate outputs (Mac; needs WP3's refcache, runs no model).

A perfect pipeline is mocked from the reference itself: encoder output = refcache fp16s_enc (the FP64-feature
reference is replaced by the same arrays), replay diagnostics = the reference logits / LSTM states, argmax = the
reference argmax, free tokens = the reference tokens. Expected: every condition passes, head errors 0. Then three
faults, each of which must fail the evaluation (exit 10) for the stated reason:
  nan     one NaN in one clip's F1 replay state       -> f1 finite fails, f2/f0 pass
  missing one clip dropped                             -> coverage fails (problems)
  flip    5% of F0 token decisions flipped              -> f0 decisions fail

  cd ios && ios/macguard --rss-cap 2G --timeout 900 -- pyenv/.venv/bin/python tests/pipegate_selftest.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

IOS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IOS))
sys.path.insert(0, str(IOS.parent))

import artifacts  # noqa: E402
import pipegate  # noqa: E402
from mil import refcache  # noqa: E402

MODEL = "mp2"


def build(root: Path, fault: str | None) -> Path:
    d = root / (fault or "clean")
    shutil.rmtree(d, ignore_errors=True)
    (d / "enc").mkdir(parents=True)
    clips = json.loads((IOS / "clips.json").read_text())["clips"]
    traces = {t["id"]: t for t in json.loads((IOS / "traces.json").read_text())["clips"]}
    index = refcache.validate(MODEL)
    ids = [c["id"] for c in clips]
    if fault == "missing":
        ids = ids[1:]
    comps = {"front_end": {"manifest_sha256": "mock"}}
    header = {"record": "header", "model": MODEL, "arm": "MOCK", "variant": "multi", "compute_units": "cpuAndNeuralEngine",
              "decodes": list(pipegate.DECODES), "clip_ids": ids, "clips_json_sha256": pipegate.sha(IOS / "clips.json"),
              "executable_sha256": "mock", "pipelines": {k: {"record": f"mock-{k}.json", "components": comps} for k in pipegate.DECODES}}
    lines = [header]
    rng = np.random.default_rng(0)
    for c in clips:
        if c["id"] not in ids:
            continue
        ref = refcache.load_clip(MODEL, c["id"])
        enc = ref["fp16s_enc"].T.astype("<f4")
        (d / "enc" / f"{c['id']}.f32").write_bytes(enc.tobytes())
        rec = {"record": "clip", "clip": c["id"], "kind": c["kind"], "bucket": c["bucket"], "encoder_length": enc.shape[0],
               "finite": True, "enc_sha256": hashlib.sha256(enc.tobytes()).hexdigest()}
        lg = ref["fp16s_logits"]
        tok = lg[:, :-5].argmax(1)
        dur = np.asarray(pipegate.DURATIONS)[lg[:, -5:].argmax(1)]
        for dec in pipegate.DECODES:
            hn, cn = pipegate.STATE_SECTIONS[dec]
            secs = {hn: ref["fp16s_h"].astype("<f4").copy(), cn: ref["fp16s_c"].astype("<f4").copy()}
            if dec != "f1":
                secs["logits"] = lg.astype("<f4")
            if fault == "nan" and dec == "f1" and c["id"] == ids[0]:
                secs[hn][0, 0, 0] = np.nan
            t = tok.copy()
            if fault == "flip" and dec == "f0":
                k = rng.random(t.size) < 0.05
                t[k] = (t[k] + 1) % 1025
            data, index_ = b"", []
            for name, arr in secs.items():
                index_.append({"name": name, "shape": list(arr.shape), "offset": len(data), "bytes": arr.nbytes})
                data += arr.tobytes()
            (d / "diag" / dec).mkdir(parents=True, exist_ok=True)
            (d / "diag" / dec / f"{c['id']}.f32").write_bytes(data)
            nonfinite = int(sum((~np.isfinite(a)).sum() for a in secs.values()))
            rec[dec] = {"tokens": index["free_tokens"][c["id"]]["fp16s_free_tokens"], "free_steps": int(lg.shape[0]),
                        "free_values": 1, "free_nonfinite": 0, "argmax_token": t.tolist(), "argmax_duration": dur.tolist(),
                        "replay_steps": traces[c["id"]]["steps"], "replay_predictions": 0, "replay_values": 1,
                        "replay_nonfinite": nonfinite,
                        "diag": {"file": f"diag/{dec}/{c['id']}.f32", "sha256": hashlib.sha256(data).hexdigest(), "sections": index_}}
        lines.append(rec)
    (d / "gate.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines))
    (d / "build.json").write_text(json.dumps({"commit": "mock", "clean": True, "executable_sha256": "mock"}))
    return d


def main() -> int:
    root = artifacts.check(artifacts.root() / "scratch" / "pipegate-selftest")
    # the FP64-feature reference is replaced by the FP32-feature one for the mock (same arrays as the mock encoder)
    ref_dir = root / "ref"
    shutil.rmtree(ref_dir, ignore_errors=True)
    ref_dir.mkdir(parents=True)
    rows = {}
    for c in json.loads((IOS / "clips.json").read_text())["clips"]:
        a = refcache.load_clip(MODEL, c["id"])["fp16s_enc"].astype(np.float32)
        np.save(ref_dir / f"{c['id']}.npy", a)
        rows[c["id"]] = {"sha256": hashlib.sha256(a.astype("<f4").tobytes()).hexdigest()}
    (ref_dir / "index.json").write_text(json.dumps({"clips": rows}))
    pipegate.validate_ref64 = lambda model, fe, idx: (ref_dir, {"clips": rows}, [])
    failures = []
    for fault, expect in ((None, {"f2": True, "f0": True, "f1": True}), ("nan", {"f2": True, "f0": True, "f1": False}),
                          ("missing", {"f2": False, "f0": False, "f1": False}), ("flip", {"f2": True, "f0": False, "f1": True})):
        d = build(root, fault)

        class A:
            gate_dir = str(d)
        code = pipegate.cmd_evaluate(A)
        ev = json.loads((d / "evaluation.json").read_text())
        got = {k: v["pass_before_wer"] for k, v in ev["decodes"].items()}
        ok = got == expect and code == (0 if all(expect.values()) else 10)
        if fault is None:
            e = ev["decodes"]["f2"]["head_errors"]
            ok = ok and e["token_logits"]["rel_max"] == 0 and e["h"]["rel_max"] == 0 and ev["decodes"]["f1"]["head_errors"]["token_logits"] is None
        if fault == "nan":
            ok = ok and not ev["decodes"]["f1"]["finite"]["pass"] and ev["decodes"]["f1"]["finite"]["replay_recount"] == 1
        if fault == "flip":
            ok = ok and not ev["decodes"]["f0"]["decisions"]["pass"]
        print(f"{fault or 'clean'}: exit {code}, pass {got}, expected {expect}: {'PASS' if ok else 'FAIL'}", flush=True)
        failures += [] if ok else [fault or "clean"]
    print(f"failed checks: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
