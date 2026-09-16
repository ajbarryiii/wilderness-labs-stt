"""Apply the validated onset augmentation to all pilot training clips."""

import json

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from common import digest
from onset_augmented import Augmentation
from prepare import feature


class PilotAugmentation(Augmentation):
    def __init__(self, run):
        cfg = json.loads((run / "config.json").read_text())
        aug = cfg["augmentation"]
        # Inherit the exact RNG draw order and transformations of the validated
        # fresh FP diagnostic. Do not silently accept unsupported recipes.
        assert aug["gain_db"] == [0, -6, -12, -18, -24]
        assert aug["original_probability"] == 0.25
        assert aug["seed"] == cfg["seed"] + 171
        super().__init__(run)
        self.training_ids = {
            r["id"]
            for r in json.loads((run / "manifest.json").read_text())["rows"]
            if r["split"] == "train"
        }

    def get(self, row, prefix_frames, gain):
        assert row["split"] == "train" and row["id"] in self.training_ids
        key = row["id"], gain
        if key not in self.cache:
            assert digest(row["audio"]) == row["audio_sha256"]
            audio, sr = sf.read(row["audio"], dtype="float32")
            assert sr == 16000 and audio.ndim == 1
            if row["domain"] == "digits":
                assert not np.any(audio[:3200])
                audio = audio[3200:]
            self.cache[key] = torch.from_numpy(feature(audio * 10 ** (gain / 20)))
        return F.pad(self.cache[key], (prefix_frames, 0), value=-2)

    def state_dict(self):
        return {
            "rng_state": self.rng.bit_generator.state,
            "trace_sha256": self.trace.hexdigest(),
        }
