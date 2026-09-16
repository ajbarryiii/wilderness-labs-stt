"""Full-training decoding and unseen-utterance checks on known Libri speakers.

The new diagnostic subset uses downloaded train-clean-100 audio excluded from
the pilot manifest, 64 distinct training speakers, and development-matched
durations. Selection does not inspect model predictions. No official test used.
"""

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

RUN = Path(
    "/mnt/hd/wilderness-labs-stt/stt-distillation/runs/pilot-4h-20260910T180602Z"
)
sys.path.insert(0, str(RUN / "code"))
from common import ART, digest, norm, save, storage
from prepare import feature


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "evaluate"])
    args = p.parse_args()
    storage()
    out = RUN / "analysis/broad-investigation"
    rows = json.loads((RUN / "manifest.json").read_text())["rows"]
    if args.mode == "prepare":
        root = ART / "datasets/libri/LibriSpeech/train-clean-100"
        used = {r["id"] for r in rows}
        train_speakers = {
            r["speaker"]
            for r in rows
            if r["split"] == "train" and r["domain"] == "general"
        }
        dev = [
            r for r in rows if r["split"] == "development" and r["domain"] == "general"
        ]
        rank = lambda s: hashlib.sha256(
            ("broad-audit-20260910:" + s).encode()
        ).hexdigest()
        speakers = sorted(train_speakers, key=rank)[: len(dev)]
        prepared = []
        folder = out / "unseen-known-speakers"
        folder.mkdir(exist_ok=False)
        for speaker, reference in zip(speakers, dev):
            candidates = []
            for transcript in (root / speaker).rglob("*.trans.txt"):
                for line in transcript.read_text().splitlines():
                    name, text = line.split(" ", 1)
                    ident = "libri-" + name
                    if ident in used:
                        continue
                    audio_path = transcript.parent / (name + ".flac")
                    duration = sf.info(audio_path).duration
                    if 3 <= duration <= 12:
                        candidates.append(
                            (
                                abs(duration - reference["seconds"]),
                                rank(ident),
                                ident,
                                text,
                                audio_path,
                            )
                        )
            assert candidates, speaker
            _, _, ident, text, audio_path = min(candidates)
            audio, sr = sf.read(audio_path, dtype="float32")
            assert sr == 16000 and audio.ndim == 1
            fp = folder / (ident + ".npy")
            np.save(fp, feature(audio))
            prepared.append(
                {
                    "id": ident,
                    "text": norm(text),
                    "speaker": speaker,
                    "domain": "general",
                    "split": "diagnostic_unseen_known_speaker",
                    "seconds": len(audio) / sr,
                    "audio": str(audio_path),
                    "features": str(fp),
                    "feature_sha256": digest(fp),
                    "audio_sha256": digest(audio_path),
                    "duration_matching_development_id": reference["id"],
                    "duration_difference_seconds": abs(
                        len(audio) / sr - reference["seconds"]
                    ),
                }
            )
        save(
            out / "unseen-known-speakers.json",
            {
                "rows": prepared,
                "selection": __doc__,
                "source_sha256": digest(__file__),
                "median_duration_difference_seconds": float(
                    np.median([r["duration_difference_seconds"] for r in prepared])
                ),
                "max_duration_difference_seconds": max(
                    r["duration_difference_seconds"] for r in prepared
                ),
            },
        )
        print(
            f"Prepared {len(prepared)} previously unused recordings from {len(speakers)} known training speakers."
        )
        return

    import torch
    from pilot4_train import configured_model, evaluate

    torch.set_num_threads(4)
    unseen = json.loads((out / "unseen-known-speakers.json").read_text())["rows"]
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "development"]
    features = {
        r["id"]: torch.from_numpy(np.load(r["features"])) for r in train + dev + unseen
    }
    result = {}
    for name in ["best", "latest"]:
        cp = torch.load(
            RUN / "train/ternary_matched" / (name + ".pt"),
            mmap=True,
            map_location="cpu",
            weights_only=False,
        )
        model = configured_model(cp["config"])(
            cp["config"]["model"], cp["arm"]["precision"]
        )
        model.load_state_dict(cp["model"], strict=True)
        model.cuda().eval()
        step = cp["step"]
        del cp
        result[name] = {}
        for group, selected in [
            ("all_training", train),
            ("unseen_known_speakers", unseen),
            ("development", dev),
        ]:
            result[name][group] = evaluate(model, selected, features, step)
            m = result[name][group]["metrics"]
            print(
                json.dumps(
                    {
                        "checkpoint": name,
                        "step": step,
                        "group": group,
                        "domains": {
                            d: {k: m[d][k] for k in ["wer", "cer", "utterances"]}
                            for d in ["general", "medical_symptoms", "digits"]
                        },
                    }
                ),
                flush=True,
            )
            save(out / "generalization.json", result)
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
