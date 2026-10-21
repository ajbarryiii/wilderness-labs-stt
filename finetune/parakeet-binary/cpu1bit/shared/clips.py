"""Freeze the cpu1bit dev workloads (plans/cpu-benchmark-mac.md "Workloads", M5) into shared/clips.json.

DEV MANIFESTS ONLY, through gpu1bit/shared/workloads.py's checked reader (`_manifest`: the file is
one of paths.DEV_SETS, its metadata split is a dev/validation split, its SHA-256 equals dev.done.json).
Test manifests are never opened. Rules, fixed before any capture or timing:
- timed (8): targets 16 s then 28 s, 4 clips each; corpora in DEV_CORPORA order, round-robin
  continuing across targets (16 s: librispeech_dev_clean, librispeech_dev_other, ami_dev,
  voxpopuli_dev; 28 s: yodas_dev, librispeech_dev_clean, librispeech_dev_other, ami_dev), each the
  corpus's closest unused utterance to the target (|duration - target|, then id).
- exactness (16): 4 + 3 + 3 + 3 + 3 in DEV_CORPORA order, evenly spaced over each full dev manifest
  (index i * (len - 1) // (count - 1)), skipping timed ids and zero durations by taking the next index.
Frames: M as workloads._frames (NeMo's encoder TENSOR frames: mel frames samples // 160 + 1, then three
ceil(L / 2)) and T = LocalFlow's frames_for_samples (lf-asr encoder.rs:176: out_len^3(samples // 160),
out_len(n) = ceil(n / 2)), which equals NeMo's encoded length. The replay uses T (LocalFlow runs each
utterance at its exact length); when M = T + 1 NeMo's last tensor frame is padding and is not captured.
Capture bytes = T * 344,064 floats * 4 (schedule.json capture_floats_per_frame).
(v2, 2026-10-21: T added and used for capture bytes; the v1 freeze an hour earlier used M; same ids.)

    finetune/parakeet-binary/python cpu1bit/shared/clips.py freeze|check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # finetune/parakeet-binary, for gpu1bit.shared
FROZEN = HERE / "clips.json"
TIMED_TARGETS = (16, 28)
TIMED_PER_TARGET = 4
EXACTNESS = 16
FLOATS_PER_FRAME = 344_064


def localflow_frames(samples: int) -> int:
    """lf-asr encoder.rs frames_for_samples: out_len(out_len(out_len(samples / 160))), out_len(n) = (n - 1) / 2 + 1."""
    n = samples // 160
    for _ in range(3):
        n = 0 if n == 0 else (n - 1) // 2 + 1
    return n


def select() -> dict:
    import soundfile as sf
    from gpu1bit.shared import workloads as W
    manifests, meta = {}, {}
    for name in W.DEV_CORPORA:
        manifests[name], meta[name] = W._manifest(name)
    used: set = set()
    timed, turn = [], 0
    for target in TIMED_TARGETS:
        for _ in range(TIMED_PER_TARGET):
            corpus = W.DEV_CORPORA[turn % len(W.DEV_CORPORA)]
            r = W._closest(manifests[corpus], target, used)
            used.add(r["id"])
            timed.append(W._slim(r, target_seconds=target))
            turn += 1
    exact = []
    for i, corpus in enumerate(W.DEV_CORPORA):
        count = EXACTNESS // len(W.DEV_CORPORA) + (1 if i < EXACTNESS % len(W.DEV_CORPORA) else 0)
        source = manifests[corpus]
        for j in range(count):
            idx = j * (len(source) - 1) // (count - 1)
            while source[idx]["id"] in used or source[idx]["duration"] <= 0:
                idx += 1
            used.add(source[idx]["id"])
            exact.append(W._slim(source[idx]))
    for r in timed + exact:
        info = sf.info(r["audio_filepath"])
        if info.samplerate != 16000 or info.channels != 1:
            raise ValueError(f"{r['id']}: {info.samplerate} Hz, {info.channels} channels")
        r.update(W._frames(int(info.frames)))
        r["T"] = localflow_frames(int(info.frames))
        if r["M"] not in (r["T"], r["T"] + 1):
            raise AssertionError(f"{r['id']}: NeMo tensor frames {r['M']} vs LocalFlow T {r['T']}")
        r["capture_bytes"] = r["T"] * FLOATS_PER_FRAME * 4
        if r["corpus"] not in W.DEV_CORPORA or "/test" in r["audio_filepath"].lower():
            raise ValueError(f"{r['id']}: not a dev record")
    return {"about": __doc__.split("\n\n")[0], "rules": __doc__.split("\n\n")[1], "manifests": meta,
            "timed": timed, "exactness": exact,
            "budget": {"timed_capture_bytes_per_model": sum(r["capture_bytes"] for r in timed),
                       "exactness_capture_bytes_per_model": sum(r["capture_bytes"] for r in exact),
                       "largest_clip_bytes": max(r["capture_bytes"] for r in timed + exact)},
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def load() -> dict:
    doc = json.loads(FROZEN.read_text())
    body = {k: v for k, v in doc.items() if k != "content_sha256"}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != doc["content_sha256"]:
        raise ValueError(f"{FROZEN} content hash mismatch")
    return doc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("action", choices=["freeze", "check"])
    a = p.parse_args()
    if a.action == "freeze":
        if FROZEN.exists():
            raise SystemExit(f"{FROZEN} is frozen; `check` verifies it")
        doc = select()
        doc["frozen_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        doc["content_sha256"] = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
        tmp = FROZEN.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(doc, indent=1) + "\n")
        os.replace(tmp, FROZEN)
    doc = load()
    if a.action == "check":
        fresh = select()
        for group in ("timed", "exactness"):
            if [r["id"] for r in fresh[group]] != [r["id"] for r in doc[group]]:
                raise SystemExit(f"{group}: the rule no longer reproduces the frozen ids")
    print(json.dumps({"timed": [(r["target_seconds"], r["corpus"], r["id"], r["duration"], r["M"], r["T"]) for r in doc["timed"]],
                      "exactness": [(r["corpus"], r["id"], r["duration"], r["M"], r["T"]) for r in doc["exactness"]],
                      "budget_GB": {k: round(v / 1e9, 3) for k, v in doc["budget"].items()},
                      "content_sha256": doc["content_sha256"]}, indent=1))


if __name__ == "__main__":
    main()
