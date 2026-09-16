"""CPU-only inventory, leakage checks and feature preparation. Never trains."""

import argparse
import collections
import hashlib
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from common import ART, digest, encode, norm, save
from prepare import feature
from recovery_core import SOURCE_RUN, artifact, read
from scaling_core import ROOT, DOMAINS, groups_for_seed, rank, tier_ids

HERE = Path(__file__).resolve().parent
KNOWN = SOURCE_RUN / "analysis/broad-investigation/unseen-known-speakers.json"


def pcm(path):
    x, sr = sf.read(artifact(path), dtype="float32")
    assert sr == 16000 and x.ndim == 1 and np.isfinite(x).all(), path
    return x, hashlib.sha256(np.asarray(x, dtype="<f4").tobytes()).hexdigest()


def prepare(output):
    output = artifact(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = read(HERE / "scaling.json")
    original = read(SOURCE_RUN / "manifest.json")
    subsets = read(SOURCE_RUN / "subsets.json")
    known = read(KNOWN)["rows"]
    anchors = set(subsets["gate"] + subsets["monitor"])
    baseline = {r["id"]: r for r in original["rows"]}
    speakers = {str(r["speaker"]) for r in original["rows"] if r["split"] == "train" and r["domain"] == "general"}
    dev_speakers = {str(r["speaker"]) for r in original["rows"] if r["split"] == "development" and r["domain"] == "general"}
    held = [dict(r) for r in original["rows"] if r["split"] != "train"] + [dict(r) for r in known]
    held_ids = {r["id"] for r in held}
    held_text = {norm(r["text"]) for r in held}
    held_pcm = set()
    for r in held:
        x, h = pcm(r["audio"])
        r.update(pcm_sha256=h, frames=feature(x).shape[-1] if "frames" not in r else r["frames"])
        assert digest(r["features"]) == r["feature_sha256"]
        held_pcm.add(h)
    retained = [dict(r) for r in original["rows"] if r["split"] == "train" and r["domain"] != "general"]
    seen_pcm = set()
    for r in retained:
        x, h = pcm(r["audio"])
        assert h not in held_pcm, f"Existing training/holdout waveform leakage: {r['id']}"
        if r["domain"] == "medical_symptoms":
            assert norm(r["text"]) not in held_text
        assert digest(r["features"]) == r["feature_sha256"]
        if r["domain"] == "digits":
            assert not x[:3200].any()
        r["pcm_sha256"] = h
        seen_pcm.add(h)
    catalog_path = output / "catalog.json"
    source = ART / "datasets/libri/LibriSpeech/train-clean-100"
    if catalog_path.exists():
        catalog = read(catalog_path)["rows"]
    else:
        catalog = []
        for file in sorted(source.glob("*/*/*.trans.txt")):
            speaker = file.parent.parent.name
            if speaker not in speakers | dev_speakers:
                continue
            for line in file.read_text().splitlines():
                name, text = line.split(" ", 1)
                path = file.parent / (name + ".flac")
                info = sf.info(path)
                if cfg["duration_edges_seconds"][0] <= info.duration <= cfg["duration_edges_seconds"][-1]:
                    catalog.append(dict(id="libri-" + name, domain="general", speaker=speaker,
                        split="train" if speaker in speakers else "long_development",
                        audio=str(path), text=norm(text), seconds=info.duration, frames=info.frames // 160,
                        source="LibriSpeech train-clean-100"))
        save(catalog_path, dict(rows=catalog))
    # An additional frozen long-clip diagnostic covers the newly admitted range.
    # Existing development/calibration recordings and exact transcripts stay reserved.
    long_rows = sorted((r for r in catalog if r["split"] == "long_development" and r["seconds"] > 12
                        and r["id"] not in held_ids and r["text"] not in held_text), key=lambda r: rank(1309, r["id"]))
    long_selected, texts = [], set(held_text)
    # Round robin over all twelve held-out general-speech speakers.
    queues = {s: [r for r in long_rows if r["speaker"] == s] for s in sorted(dev_speakers)}
    while len(long_selected) < 64 and any(queues.values()):
        for queue in queues.values():
            if queue and len(long_selected) < 64:
                r = queue.pop(0)
                if r["text"] not in texts:
                    long_selected.append(r)
                    texts.add(r["text"])
    assert len(long_selected) == 64
    rejected = collections.Counter()
    all_train = sorted((r for r in catalog if r["split"] == "train"),
                       key=lambda r: (r["id"] not in anchors, rank(8011, r["id"])))
    prepared = []
    feat_dir = output / "features"
    feat_dir.mkdir(exist_ok=True)
    seen_text = set(texts)
    for index, candidate in enumerate(long_selected + all_train):
        r = dict(candidate)
        if r["id"] in held_ids or (r["split"] == "train" and r["text"] in seen_text):
            rejected["reserved_or_duplicate_transcript"] += 1
            continue
        x, h = pcm(r["audio"])
        if h in held_pcm or h in seen_pcm:
            rejected["reserved_or_duplicate_pcm"] += 1
            continue
        y = encode(r["text"])
        needed = len(y) + sum(a == b for a, b in zip(y, y[1:]))
        # Reserve onset hold frames too; CTC is never allowed to hide infeasibility.
        if not y or (r["frames"] + 1) // 2 - 7 < needed:
            rejected["ctc_infeasible"] += 1
            continue
        if r["id"] in baseline:
            r.update({k: baseline[r["id"]][k] for k in ("features", "feature_sha256")})
            assert digest(r["features"]) == r["feature_sha256"]
            f = np.load(r["features"], allow_pickle=False)
        else:
            path = feat_dir / (r["id"] + ".npy")
            sidecar = path.with_suffix(".json")
            if sidecar.exists() and path.exists() and read(sidecar)["pcm_sha256"] == h:
                assert digest(path) == read(sidecar)["feature_sha256"]
                f = np.load(path, allow_pickle=False)
            else:
                f = feature(x)
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as stream:
                    np.save(stream, f, allow_pickle=False)
                temporary.replace(path)
                save(sidecar, dict(pcm_sha256=h, feature_sha256=digest(path)))
            r.update(features=str(path), feature_sha256=digest(path))
        assert f.shape == (80, r["frames"]) and np.isfinite(f).all()
        assert f.max() < 2, "Upper clipping prevents exact gain changes in feature space"
        r.update(pcm_sha256=h, audio_sha256=digest(r["audio"]))
        prepared.append(r)
        if r["split"] == "train":
            seen_text.add(r["text"])
            seen_pcm.add(h)
        else:
            held_pcm.add(h)
        if index % 500 == 0:
            print(f"Prepared {index}/{len(long_selected) + len(all_train)} candidates", flush=True)
    train = retained + [r for r in prepared if r["split"] == "train"]
    assert anchors <= {r["id"] for r in train}
    assert len([r for r in prepared if r["split"] == "long_development"]) == 64
    rows = train + held + [r for r in prepared if r["split"] == "long_development"]
    assert len(rows) == len({r["id"] for r in rows})
    by_id = {r["id"]: r for r in rows}
    designs, summaries = {}, {}
    for seed in cfg["seeds"]:
        groups, dropped = groups_for_seed(train, anchors, seed, cfg)
        designs[str(seed)] = dict(groups=groups, dropped_incomplete_cell_rows=dropped)
        summaries[str(seed)] = {}
        for size, k in cfg["sizes"].items():
            chosen = [by_id[i] for i in tier_ids(groups, k)]
            summaries[str(seed)][size] = {d: dict(recordings=sum(r["domain"] == d for r in chosen),
                hours=sum(r["seconds"] for r in chosen if r["domain"] == d) / 3600,
                speakers=len({r["speaker"] for r in chosen if r["domain"] == d and r["speaker"] is not None})) for d in DOMAINS}
    # Gain arithmetic is checked against the waveform implementation, including
    # the digit prefix boundary. The worker uses this only for nonpositive gains.
    parity = []
    for d in DOMAINS:
        examples = sorted((r for r in train if r["domain"] == d), key=lambda r: r["seconds"])
        for r in (examples[0], examples[len(examples) // 2], examples[-1]):
            x, _ = pcm(r["audio"])
            f = np.load(r["features"], allow_pickle=False)
            if d == "digits":
                x, f = x[3200:], f[:, 20:]
            for gain in cfg["gains_db"]:
                error = float(np.max(np.abs(feature(x * 10 ** (gain / 20)) - np.maximum(-2, f + gain * math.log(10) / 60))))
                assert error < 1e-4, (r["id"], gain, error)
                parity.append(error)
    save(output / "manifest.json", dict(rows=rows))
    save(output / "designs.json", designs)
    save(output / "subsets.json", dict(common_train_monitor=subsets["monitor"], gate=subsets["gate"],
        development=[r["id"] for r in rows if r["split"] == "development"],
        long_development=[r["id"] for r in rows if r["split"] == "long_development"],
        known_speaker=[r["id"] for r in known]))
    save(output / "ready.json", dict(status="data_prepared_no_training", summary=summaries,
        rejection_counts=dict(rejected), training_speakers=len(speakers), max_gain_parity_error=max(parity),
        source_manifest_sha256=digest(SOURCE_RUN / "manifest.json"), known_manifest_sha256=digest(KNOWN),
        config_sha256=digest(HERE / "scaling.json"),
        files={n: digest(output / n) for n in ("manifest.json", "designs.json", "subsets.json")},
        scope="More utterances from the same training speakers/source; fixed medical/digit pools. No final test data used."))
    print(read(output / "ready.json")["summary"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "data/v1")
    prepare(p.parse_args().output)
