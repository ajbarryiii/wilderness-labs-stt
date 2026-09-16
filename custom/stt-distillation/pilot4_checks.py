"""Functional contracts for startup silence, causality, padding and pilot schedules."""

import json
from pathlib import Path

import numpy as np
import torch

from common import ART, digest, encode, read_manifest, save, storage
from model import Conv, Linear
from onset_model import OnsetModel as PilotModel, startup_mask
from pilot4_train import schedule


def main():
    storage()
    torch.set_num_threads(2)
    cfg = {
        "width": 32,
        "depth": 2,
        "heads": 4,
        "context": 16,
        "dropout": 0.0,
        "checkpoint": False,
        "initial_blank_bias": 2,
        "startup_floor": -1.95,
    }
    for precision in ["fp", "ternary"]:
        torch.manual_seed(12)
        model = PilotModel(cfg, precision).eval()
        # A nonconstant head makes the causal and padding tests informative.
        torch.nn.init.xavier_uniform_(model.head.weight)
        x = torch.randn(1, 80, 80)
        x[..., :8] = -2
        future = x.clone()
        future[..., 40:] = torch.randn_like(future[..., 40:]) * 2
        with torch.no_grad():
            y = model(x)
            altered = model(future)
            assert torch.allclose(y[:, :20], altered[:, :20], atol=1e-5, rtol=1e-5)
            assert (y[:, :4].argmax(-1) == 0).all()
            silent = model(torch.full_like(x, -2))
            assert torch.isfinite(silent).all() and (silent.argmax(-1) == 0).all()
            short = x[..., :60]
            padded = torch.nn.functional.pad(short, (0, 20))
            combined = model(torch.cat([padded, x], dim=0))
            assert torch.allclose(model(short), combined[:1, :30], atol=2e-5, rtol=2e-5)
        assert isinstance(model.head, torch.nn.Linear) and model.head.bias is not None
        assert all(m.precision == "fp" for m in model.modules() if isinstance(m, Conv))
        assert all(
            m.precision == precision
            for b in model.blocks
            for m in b.modules()
            if isinstance(m, Linear)
        )
    config = json.loads(Path(__file__).with_name("pilot4.json").read_text())
    for arm in config["arms"]:
        lr1, wd1, stage1 = schedule(arm, config, 500, 2000)
        lr2, wd2, stage2 = schedule(arm, config, 1500, 2000)
        assert lr1 > 0 and lr2 > 0 and stage1 == 1
        if arm["schedule"] == "two_stage":
            assert stage2 == 2 and wd2 == 0 and wd1 > 0
        else:
            assert stage2 == 1 and wd1 == wd2
    count = 0
    for row in read_manifest()["rows"]:
        x = np.load(row["features"])
        seen = startup_mask(torch.from_numpy(x)[None])[0, ::2]
        y = encode(row["text"])
        needed = len(y) + sum(a == b for a, b in zip(y, y[1:]))
        assert int(seen.sum()) >= needed, (
            f"Startup gate leaves insufficient CTC frames: {row['id']}"
        )
        count += 1
    from onset_checks import main as check_onset
    from pilot4_aug_checks import main as check_augmentation

    check_onset()
    check_augmentation()
    save(
        ART / "preflight/pilot4-checks.json",
        {
            "passed": True,
            "manifest_rows": count,
            "contracts": [
                "causality",
                "startup_silence",
                "all_silence_finite",
                "padding_invariance",
                "precision_placement",
                "optimizer_stages",
                "CTC_alignment_after_onset",
            ],
            "source_hashes": {
                name: digest(Path(__file__).with_name(name))
                for name in [
                    "pilot4_model.py", "onset_model.py", "model.py",
                    "pilot4_train.py", "pilot4_control.py", "pilot4.json",
                    "pilot4_augmentation.py", "onset_augmented.py",
                    "onset_repair.py", "prepare.py", "common.py", "pilot4_probe.py",
                    "medical_terms.json", "python", "pilot4_checks.py",
                    "onset_checks.py", "pilot4_aug_checks.py",
                ]
            },
        },
    )
    print(
        f"Passed functional contracts; all {count} rows remain CTC-feasible after startup silence gating."
    )


if __name__ == "__main__":
    main()
