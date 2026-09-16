"""Bounded broad-data probe changing only FP initialization to gate weights.

Original run artifacts remain immutable. Sampling, augmentation, optimizer and
the original 27,000-step LR schedule are reused for 3,000 diagnostic updates.
The gate optimizer is deliberately not restored. This is not a qualified model.
"""

import json
import shutil
import sys
from pathlib import Path

RUN = Path(
    "/mnt/hd/wilderness-labs-stt/stt-distillation/runs/pilot-4h-20260910T180602Z"
)
sys.path.insert(0, str(RUN / "code"))

import torch
from common import digest, save, storage
from onset_model import OnsetModel
import pilot4_train


def main():
    storage()
    out = RUN / "analysis/broad-investigation/warmstart-probe"
    out.mkdir(parents=True, exist_ok=False)
    for name in ["config.json", "manifest.json", "subsets.json"]:
        shutil.copy2(RUN / name, out / name)
    shutil.copy2(__file__, out / Path(__file__).name)
    save(
        out / "probe-provenance.json",
        {
            "source_run": str(RUN),
            "initial_weights": str(RUN / "gate/fp_control/latest.pt"),
            "gate_step": 4700,
            "optimizer": "fresh AdamW, identical to original broad training",
            "requested_probe_updates": 3000,
            "lr_schedule_total_updates": 27000,
            "probe_sha256": digest(__file__),
            "training_source_hashes": json.loads((RUN / "provenance.json").read_text())[
                "source_hashes"
            ],
        },
    )

    class WarmStartModel(OnsetModel):
        def __init__(self, cfg, precision):
            super().__init__(cfg, precision)
            saved = torch.load(
                RUN / "gate/fp_control/latest.pt",
                map_location="cpu",
                mmap=True,
                weights_only=False,
            )
            self.load_state_dict(saved["model"], strict=True)

    original_schedule = pilot4_train.schedule

    def original_broad_schedule(arm, cfg, step, steps, gate=False):
        return original_schedule(arm, cfg, step, 27000, gate=gate)

    pilot4_train.schedule = original_broad_schedule
    pilot4_train.worker(
        out, "fp_control", "train", 3000, 900, model_class=WarmStartModel
    )


if __name__ == "__main__":
    main()
