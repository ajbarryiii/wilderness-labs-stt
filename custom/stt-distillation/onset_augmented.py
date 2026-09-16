"""Fresh onset repair with leading-silence and gain augmentation on training clips."""

import argparse
import json
import shutil

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from common import ART, decode, digest, norm, save, storage
from onset_model import OnsetModel
from onset_repair import prepare, audit
from pilot4_train import worker
from prepare import feature


class Augmentation:
    def __init__(self, run):
        cfg = json.loads((run / "config.json").read_text())
        manifest = json.loads((run / "manifest.json").read_text())["rows"]
        gate = set(json.loads((run / "subsets.json").read_text())["gate"])
        self.rows = [r for r in manifest if r["id"] in gate]
        self.rng = np.random.default_rng(cfg["seed"] + 171)
        self.cache = {}
        for row in self.rows:
            assert (
                row["split"] == "train" and digest(row["audio"]) == row["audio_sha256"]
            )
            audio, sr = sf.read(row["audio"], dtype="float32")
            assert sr == 16000
            if row["domain"] == "digits":
                assert not np.any(audio[:3200])
                audio = audio[3200:]
            for gain in [0, -6, -12, -18, -24]:
                self.cache[row["id"], gain] = torch.from_numpy(
                    feature(audio * 10 ** (gain / 20))
                )
        self.trace = __import__("hashlib").sha256()

    def get(self, row, prefix_frames, gain):
        return F.pad(self.cache[row["id"], gain], (prefix_frames, 0), value=-2)

    def __call__(self, rows):
        result = []
        for row in rows:
            if self.rng.random() < 0.25:
                prefix = 20 if row["domain"] == "digits" else 0
                gain = 0
            else:
                prefix = int(self.rng.integers(0, 101))
                gain = int(self.rng.choice([0, -6, -12, -18, -24]))
            self.trace.update(f"{row['id']} {prefix} {gain}\n".encode())
            result.append(self.get(row, prefix, gain))
        return result

    def gate(self, model):
        was_training = model.training
        model.eval()
        predictions = []
        with torch.inference_mode():
            for row in self.rows:
                if row["domain"] != "digits":
                    continue
                for prefix in [0, 60]:
                    for gain in [0, -24]:
                        x = self.get(row, prefix, gain)[None].cuda()
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            pred = decode(model(x)[0].argmax(-1).tolist())
                        predictions.append(
                            {
                                "id": row["id"],
                                "prefix_ms": prefix * 10,
                                "gain_db": gain,
                                "reference": norm(row["text"]),
                                "prediction": pred,
                                "exact": pred == norm(row["text"]),
                            }
                        )
        model.train(was_training)
        exact = sum(p["exact"] for p in predictions)
        return {
            "passed": exact == len(predictions),
            "exact": exact,
            "total": len(predictions),
            "predictions": predictions,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "train", "audit"])
    parser.add_argument("name")
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    storage()
    run = ART / "training-repair" / args.name
    if args.mode == "prepare":
        prepare(args.name, 10, args.seed)
        shutil.copy2(__file__, run / "source/onset_augmented.py")
        cfg = json.loads((run / "config.json").read_text())
        cfg["augmentation"] = {
            "prefix_ms": "uniform integer 10 ms hops, 0..1000",
            "gain_db": [0, -6, -12, -18, -24],
            "original_probability": 0.25,
            "seed": args.seed + 171,
            "scope": "All 32 gate training clips. Only the known synthetic 200 ms zero prefix is removed from digit clips before adding a new prefix.",
        }
        cfg["extra_gate"] = (
            "All eight digit clips exact at prefix 0/600 ms x gain 0/-24 dB, two consecutive checks. These are training augmentation conditions."
        )
        save(run / "config.json", cfg)
        provenance = json.loads((run / "provenance.json").read_text())
        provenance["sources"]["onset_augmented.py"] = digest(
            run / "source/onset_augmented.py"
        )
        save(run / "provenance.json", provenance)
    elif args.mode == "train":
        augmentation = Augmentation(run)
        worker(
            run,
            "fp_control",
            "gate",
            12000,
            1100,
            model_class=OnsetModel,
            batch_transform=augmentation,
            extra_gate=augmentation.gate,
        )
        save(
            run / "augmentation-state.json",
            {
                "trace_sha256": augmentation.trace.hexdigest(),
                "rng_state": augmentation.rng.bit_generator.state,
                "resume_scope": "Augmentation state is supplemental; this trial always starts fresh.",
            },
        )
    else:
        audit(run)


if __name__ == "__main__":
    main()
