"""Check diagnostic equivalence and training-wide augmentation feasibility."""

import json
from pathlib import Path

import numpy as np
import torch

from common import ART, digest, encode, read_manifest, save, storage
from onset_augmented import Augmentation
from onset_model import startup_mask
from pilot4_augmentation import PilotAugmentation
from pilot4_probe import select_rows


def main():
    storage()
    torch.set_num_threads(2)
    run = ART / "preflight/pilot4-augmentation"
    run.mkdir(exist_ok=True)
    cfg = json.loads(Path(__file__).with_name("pilot4.json").read_text())
    manifest = read_manifest()
    gate = select_rows(32)
    save(run / "config.json", cfg)
    save(run / "manifest.json", manifest)
    save(run / "subsets.json", {"gate": [r["id"] for r in gate]})
    reference = Augmentation(run)
    actual = PilotAugmentation(run)
    for _ in range(8):
        expected = reference(gate)
        observed = actual(gate)
        assert all(torch.equal(a, b) for a, b in zip(expected, observed))
    assert reference.trace.hexdigest() == actual.trace.hexdigest()
    assert reference.rng.bit_generator.state == actual.rng.bit_generator.state
    count = 0
    for row in manifest["rows"]:
        if row["split"] != "train":
            continue
        original = torch.from_numpy(np.load(row["features"]))
        prefix = 20 if row["domain"] == "digits" else 0
        assert torch.equal(original, actual.get(row, prefix, 0)), row["id"]
        target = encode(row["text"])
        needed = len(target) + sum(a == b for a, b in zip(target, target[1:]))
        # Lowest gain produces the latest threshold crossing. Check both
        # feature/output-stride parities without any extra silence allowance.
        for prefix in [0, 1]:
            x = actual.get(row, prefix, -24)[None]
            allowed = startup_mask(x)[0, ::2]
            assert int(allowed.sum()) >= needed, row["id"]
        count += 1
        if count % 500 == 0:
            print(f"Augmentation audited {count} training rows", flush=True)
    result = {
        "passed": True,
        "training_rows": count,
        "diagnostic_equivalent_draws": 256,
        "original_features_exact": True,
        "minimum_gain_ctc_feasible_both_stride_parities": True,
        "manifest_sha256": digest(run / "manifest.json"),
    }
    save(ART / "preflight/pilot4-augmentation-checks.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
