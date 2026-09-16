"""Check signal onset timing, retained audio, causality and CTC feasibility."""

import json

import numpy as np
import torch

from common import ART, encode, read_manifest, save, storage
from onset_model import OnsetModel, startup_mask


def reference(active, confirmation=3, hold=10):
    """Independent single-frame state machine, including chunk-persistent state."""
    streak = 0
    opened = None
    result = []
    for index, value in enumerate(active):
        streak = streak + 1 if value else 0
        if opened is None and streak >= confirmation:
            opened = index + hold
        result.append(opened is not None and index >= opened)
    return result


def main():
    storage()
    torch.set_num_threads(2)
    rng = np.random.default_rng(31)
    signals = rng.random((30, 80)) > 0.3
    x = (
        torch.from_numpy(np.where(signals[:, None], 0.0, -2.0))
        .float()
        .expand(-1, 80, -1)
    )
    assert startup_mask(x).tolist() == [reference(a) for a in signals]
    # Known floor, one-frame transient, then persistent signal. The transient
    # cannot open the gate. Confirmation at 22 + 10 hold hops => frame 32.
    x = torch.full((1, 80, 80), -2.0)
    x[..., 5] = 0
    x[..., 20:] = 0
    allow = startup_mask(x)
    assert not allow[0, :32].any() and allow[0, 32:].all()
    assert not startup_mask(torch.full_like(x, -2)).any()
    for end in [1, 6, 20, 22, 23, 32, 33, 60]:
        assert torch.equal(startup_mask(x[..., :end]), allow[:, :end])
    cfg = dict(
        width=32,
        depth=2,
        heads=4,
        context=16,
        dropout=0,
        checkpoint=False,
        startup_confirmation_frames=3,
        startup_hold_frames=10,
    )
    for precision in ["fp", "ternary"]:
        torch.manual_seed(19)
        model = OnsetModel(cfg, precision).eval()
        torch.nn.init.xavier_uniform_(model.head.weight)
        x = torch.randn(1, 80, 100)
        x[..., :20] = -2
        future = x.clone()
        future[..., 60:] += torch.randn_like(future[..., 60:])
        with torch.no_grad():
            z = model(x)
            assert torch.allclose(
                z[:, :30], model(future)[:, :30], atol=2e-5, rtol=2e-5
            )
            assert not z[:, :16].argmax(-1).any()
            assert torch.isfinite(model(torch.full_like(x, -2))).all()
            # Perturb audio inside the held region: later outputs must change,
            # proving that the hold suppresses emissions without deleting audio.
            changed = x.clone()
            changed[..., 24:28] += torch.randn_like(changed[..., 24:28])
            assert not torch.allclose(z[:, 16:25], model(changed)[:, 16:25])
    counts = {}
    misses = []
    for row in read_manifest()["rows"]:
        x = torch.from_numpy(np.load(row["features"]))[None]
        allowed = startup_mask(x)[0, ::2]
        target = encode(row["text"])
        needed = len(target) + sum(a == b for a, b in zip(target, target[1:]))
        if int(allowed.sum()) < needed:
            misses.append(row["id"])
        counts[row["domain"]] = counts.get(row["domain"], 0) + 1
    result = {
        "contracts_passed": True,
        "manifest_counts": counts,
        "ctc_infeasible": misses,
        "scope": "Digital-floor onset only; no labeled speech/noise boundary benchmark.",
    }
    save(ART / "preflight/onset-checks.json", result)
    print(json.dumps(result))
    assert not misses


if __name__ == "__main__":
    main()
