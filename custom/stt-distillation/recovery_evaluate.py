"""Run frozen acoustic/decoder diagnostics within the experiment's report reserve."""

import argparse
import subprocess
import time
from pathlib import Path

from common import save
from recovery_core import SOURCE_RUN, artifact, read


def main(run, seconds):
    run = artifact(run)
    end = time.monotonic() + seconds
    here = Path(__file__).resolve().parent
    outputs = []
    selected = None
    if (run / "winner.json").exists():
        selected = read(run / "winner.json")["job"]
    elif (run / "selection.json").exists():
        selected = next(iter(read(run / "selection.json")["selected"]), None)
    if selected:
        budget = min(480, max(120, (end - time.monotonic()) / 3))
        try:
            subprocess.run(
                [
                    str(here / "python"),
                    str(here / "recovery_audit.py"),
                    str(run),
                    selected,
                    "--seconds",
                    str(budget),
                ],
                check=True,
                timeout=budget + 30,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            outputs.append(
                dict(
                    name="generalization-audit",
                    status="failed_or_incomplete",
                    error=str(exc),
                )
            )
    cases = [
        ("reference", None),
        ("ternary14k", SOURCE_RUN / "train/ternary_matched/best.pt"),
    ]
    if selected:
        cases.insert(
            0, ("recovery-" + selected, run / "training" / selected / "best.pt")
        )
    cases.append(("failed-fp", SOURCE_RUN / "train/fp_control/latest.pt"))
    for index, (name, checkpoint) in enumerate(cases):
        remaining = end - time.monotonic()
        if remaining < 150:
            break
        budget = remaining / (len(cases) - index)
        cache = run / "evaluation" / name / "logits"
        args = [
            str(here / "python"),
            str(here / "cache_ctc.py"),
            str(cache),
            "--seconds",
            str(min(300, budget / 2)),
        ]
        if checkpoint:
            args += ["--checkpoint", str(checkpoint)]
        result = dict(
            name=name,
            checkpoint=str(checkpoint) if checkpoint else "pretrained reference",
        )
        try:
            subprocess.run(args, check=True, timeout=min(360, budget / 2 + 60))
            args = [
                str(here / "decoder-python"),
                str(here / "decode_ctc.py"),
                str(cache),
                str(run / "evaluation" / name / "decoding"),
                "--seconds",
                str(max(30, min(budget / 2, end - time.monotonic() - 30))),
            ]
            if name == "failed-fp":
                args += ["--quick"]
            subprocess.run(
                args,
                check=True,
                timeout=max(60, min(budget / 2 + 60, end - time.monotonic())),
            )
            result["status"] = "completed"
        except (subprocess.SubprocessError, OSError) as exc:
            result.update(status="failed_or_incomplete", error=str(exc))
        outputs.append(result)
        save(
            run / "evaluation/summary.json",
            dict(
                results=outputs,
                selected_acoustic_job=selected,
                fresh_confirmation=False,
                energy_measured=False,
            ),
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--seconds", type=float, default=1500)
    a = p.parse_args()
    main(a.run, a.seconds)
