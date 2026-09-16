"""Measure real accumulated training throughput before freezing the timed budget."""

import json, os, shutil, statistics, subprocess
from pathlib import Path
from common import ART, save, digest

old = Path(json.loads((ART / "latest.json").read_text())["run"])
out = ART / "preflight/throughput"
out.mkdir(parents=True, exist_ok=True)
for f in ["config.json", "targets.json", "manifest.json"]:
    shutil.copy2(old / f, out / f)
cfg = json.loads((out / "config.json").read_text())
cfg["max_steps"] = 100000
save(out / "config.json", cfg)
env = dict(os.environ, WILDERNESS_STT_MANIFEST=str(out / "manifest.json"))
here = Path(__file__).parent
subprocess.run(
    [
        str(here / "python"),
        str(here / "train.py"),
        str(out),
        "ternary_single",
        "--steps",
        "200",
        "--seconds",
        "300",
    ],
    env=env,
    check=True,
)
for precision, folder in [
    ("fp", old / "fp_single"),
    ("ternary", out / "ternary_single"),
]:
    rows = [json.loads(s) for s in (folder / "metrics.jsonl").read_text().splitlines()]
    steady = rows[25:]
    result = json.loads((folder / "result.json").read_text())
    overhead = max(0, result["elapsed_seconds"] - sum(x["step_seconds"] for x in rows))
    checkpoints = len(rows) // 500 + 1
    checkpoint_seconds = overhead / checkpoints
    measured = statistics.mean(x["step_seconds"] for x in steady)
    rate = (measured + checkpoint_seconds / 500) * 1.10
    path = ART / (precision + "-profile.json")
    profile = json.loads(path.read_text())
    profile.update(
        real_data_calibration=dict(
            run=str(folder),
            steps=len(rows),
            mean_update_seconds=measured,
            mean_checkpoint_seconds=checkpoint_seconds,
            safety_factor=1.10,
            manifest_sha256=digest(out / "manifest.json"),
        ),
        calibrated_update_seconds=rate,
    )
    save(path, profile)
    print(
        json.dumps(
            dict(
                precision=precision,
                calibrated_update_seconds=rate,
                measurement=profile["real_data_calibration"],
            )
        ),
        flush=True,
    )
save(
    ART / "preflight/throughput-calibration.json",
    dict(
        superseded_run=str(old),
        scope="Throughput only; no development-score tuning; preserve original test deadline 2026-09-10T12:38:34Z",
    ),
)
