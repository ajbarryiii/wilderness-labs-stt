"""Frozen dev-only workloads for screening, P-kernel inputs, fixtures and the gate (plans/gpu-kernel.md
A4, A12, A13). DEV MANIFESTS ONLY: every record is asserted to come from one of paths.DEV_SETS'
manifests, whose metadata split is a dev/validation split. Test manifests are never read here.

Selection rules (fixed before any candidate timing; `freeze` writes workloads.json once and
refuses to change it):
- dev_clips: LibriSpeech dev-clean utterances closest to 3, 10 and 30 s (ties by id), distinct.
- dev_24: for each target 3, 10, 30 s, round-robin over the five dev corpora in the order of
  DEV_CORPORA, each corpus contributing its next-closest unused utterance (|duration - target|,
  then id), until 8 are chosen (2 + 2 + 2 + 1 + 1); no id is used twice in the set.
- dev_sample (A4/A12 recognition gate): 512 utterances, 103 + 103 + 102 + 102 + 102 in DEV_CORPORA
  order, evenly spaced over each full dev manifest (index i * (len - 1) // (count - 1)).
- fixture_utterances (A3 captured layer inputs): for each dev clip target, one per corpus: the
  dev_24 members of that target from distinct corpora (5 per target, 15 in all), so captured
  inputs cover all five dev corpora at the three benchmark lengths.

    CUDA_VISIBLE_DEVICES= gpu1bit/shared/python -m gpu1bit.shared.workloads freeze
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from gpu1bit.shared import env

HERE = Path(__file__).resolve().parent
FROZEN = HERE / "workloads.json"
DEV_CORPORA = ("librispeech_dev_clean", "librispeech_dev_other", "ami_dev", "voxpopuli_dev", "yodas_dev")
DEV_SPLITS = {"librispeech_dev_clean": ["dev-clean"], "librispeech_dev_other": ["dev-other"],
              "ami_dev": "validation", "voxpopuli_dev": "validation", "yodas_dev": "asr_only"}
TARGETS = (3, 10, 30)
SAMPLE_SIZE = 512


def _manifest(name: str) -> tuple[list[dict], dict]:
    import paths
    if name not in paths.DEV_SETS or name not in DEV_CORPORA:
        raise ValueError(f"{name} is not a dev set")
    path = paths.MANIFESTS / f"{name}.jsonl"
    meta = json.loads((paths.MANIFESTS / f"{name}.meta.json").read_text())
    if Path(meta["path"]) != path or meta["split"] != DEV_SPLITS[name] or "test" in path.name:
        raise ValueError(f"{path}: unexpected split {meta['split']!r}")
    done = json.loads((paths.MANIFESTS / "dev.done.json").read_text())["manifests"]
    digest = env.sha256_file(path)
    if done[name] != digest:
        raise ValueError(f"{path} does not match dev.done.json")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for r in records:
        r["corpus"] = name
    return records, {"path": str(path), "sha256": digest, "split": meta["split"], "utterances": len(records)}


def _closest(records: list[dict], target: float, used: set) -> dict:
    return min((r for r in records if r["id"] not in used and r["duration"] > 0),
               key=lambda r: (abs(r["duration"] - target), r["id"]))


def _slim(r: dict, **extra) -> dict:
    return {"id": r["id"], "corpus": r["corpus"], "duration": r["duration"], "audio_filepath": r["audio_filepath"],
            "text": r["text"], **extra}


def _frames(samples: int) -> dict:
    """Encoder tensor frames (= projection rows M at batch 1), as shapes.json measured on MB1:
    mel tensor frames floor(s/160) + 1, then three stride-2 convolutions ceil(L/2) each."""
    feature = samples // 160 + 1
    t = feature
    for _ in range(3):
        t = (t + 1) // 2
    return {"samples": samples, "feature_tensor_frames": feature, "M": t, "M_linear_pos": 2 * t - 1}


def select() -> dict:
    import soundfile as sf
    manifests, meta = {}, {}
    for name in DEV_CORPORA:
        manifests[name], meta[name] = _manifest(name)
    used: set = set()
    clips = []
    for target in TARGETS:
        r = _closest(manifests["librispeech_dev_clean"], target, used)
        used.add(r["id"])
        clips.append(_slim(r, target_seconds=target))
    dev24, used24 = [], set()
    for target in TARGETS:
        chosen, turn = [], 0
        while len(chosen) < 8:
            corpus = DEV_CORPORA[turn % len(DEV_CORPORA)]
            r = _closest(manifests[corpus], target, used24)
            used24.add(r["id"])
            chosen.append(_slim(r, target_seconds=target))
            turn += 1
        dev24.extend(chosen)
    sample = []
    for i, corpus in enumerate(DEV_CORPORA):
        count = SAMPLE_SIZE // len(DEV_CORPORA) + (1 if i < SAMPLE_SIZE % len(DEV_CORPORA) else 0)
        source = manifests[corpus]
        idx = sorted({j * (len(source) - 1) // (count - 1) for j in range(count)})
        if len(idx) != count:
            raise AssertionError(f"{corpus}: duplicate sample indices")
        sample.extend(_slim(source[j]) for j in idx)
    fixtures = []
    for target in TARGETS:
        seen = set()
        for r in dev24:
            if r["target_seconds"] == target and r["corpus"] not in seen:
                seen.add(r["corpus"])
                fixtures.append(r)
    for group in (clips, dev24, sample):
        for r in group:
            info = sf.info(r["audio_filepath"])
            if info.samplerate != 16000 or info.channels != 1:
                raise ValueError(f"{r['id']}: {info.samplerate} Hz, {info.channels} channels")
            r.update(_frames(int(info.frames)))
    by_id = {r["id"]: r for r in dev24}
    fixtures = [dict(by_id[r["id"]]) for r in fixtures]
    return {"about": __doc__.split("\n\n")[0], "rules": __doc__.split("Selection rules")[1].split("\n\n")[0],
            "manifests": meta, "dev_clips": clips, "dev_24": dev24, "dev_sample": sample,
            "fixture_utterances": fixtures,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def load() -> dict:
    """The frozen workloads; raises if workloads.json is missing or was edited by hand."""
    doc = json.loads(FROZEN.read_text())
    body = {k: v for k, v in doc.items() if k != "content_sha256"}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != doc["content_sha256"]:
        raise ValueError(f"{FROZEN} content hash mismatch")
    for group in ("dev_clips", "dev_24", "dev_sample", "fixture_utterances"):
        for r in doc[group]:
            if r["corpus"] not in DEV_CORPORA or "/test" in r["audio_filepath"].lower():
                raise ValueError(f"{r['id']}: not a dev record")
    return doc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["freeze", "check", "show"])
    args = p.parse_args()
    if args.action == "freeze":
        if FROZEN.exists():
            raise SystemExit(f"{FROZEN} is frozen; `check` verifies it")
        doc = select()
        doc["frozen_at"] = __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z")
        doc["content_sha256"] = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
        tmp = FROZEN.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(doc, indent=1) + "\n")
        os.replace(tmp, FROZEN)
    doc = load()
    if args.action == "check":
        fresh = select()
        for group in ("dev_clips", "dev_24", "dev_sample", "fixture_utterances"):
            if [r["id"] for r in fresh[group]] != [r["id"] for r in doc[group]]:
                raise SystemExit(f"{group}: selection rule no longer reproduces the frozen ids")
    print(json.dumps({"dev_clips": [(r["id"], r["duration"], r["M"]) for r in doc["dev_clips"]],
                      "dev_24": [(r["target_seconds"], r["corpus"], r["id"], r["duration"], r["M"]) for r in doc["dev_24"]],
                      "dev_sample": len(doc["dev_sample"]),
                      "fixtures": [(r["corpus"], r["duration"], r["M"]) for r in doc["fixture_utterances"]],
                      "content_sha256": doc["content_sha256"]}, indent=1))


if __name__ == "__main__":
    main()
