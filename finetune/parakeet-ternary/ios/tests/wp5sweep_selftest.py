"""Self-test of the sweep report's run validation (WP7 review r2 finding 5); NixOS, no Mac, no timing.

Copies WP5's C4-multi-vdsp-f2 run into a throwaway results tree twice: once under its own arm name and once under
C3-multi-vdsp-f2 (another arm's records under this arm's name), plus a copy whose paired C0 ran on cpuOnly. Legacy
manifests name them. verified_runs must accept the first and refuse the other two before any summary is written.

  CUDA_VISIBLE_DEVICES= ./python ios/tests/wp5sweep_selftest.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

IOS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IOS))
sys.path.insert(0, str(IOS.parent))

import wp5sweep  # noqa: E402

SRC = Path("/mnt/hd/wilderness-labs-stt/parakeet-ios/results/wp5/C4-multi-vdsp-f2/wp5-20261003")
ROOT = Path("/mnt/hd/wilderness-labs-stt/parakeet-ios/scratch/wp5sweep-selftest")


def place(name: str, edit=None) -> dict:
    d = ROOT / name / "wp5-20261003"
    shutil.copytree(SRC, d)
    if edit:
        edit(d)
    return {"all": {"run": d.name, "status": 0, "dir": str(d)}}


def c0_on_cpu(d: Path) -> None:
    lines = (d / "c0.jsonl").read_text().splitlines()
    load = json.loads(lines[0])
    load["compute_units"] = "cpuOnly"
    (d / "c0.jsonl").write_text("\n".join([json.dumps(load), *lines[1:]]) + "\n")


def main() -> int:
    shutil.rmtree(ROOT, ignore_errors=True)
    wp5sweep.LOCAL = ROOT
    natural = [c["id"] for c in json.loads((IOS / "clips.json").read_text())["clips"] if c["kind"] == "natural"]
    base = {"legacy": "selftest", "build": None, "halves": {"all": natural}}
    cases = [("right arm", "C4-multi-vdsp-f2", None, True),
             ("another arm's records", "C3-multi-vdsp-f2", None, False),
             ("C0 on cpuOnly", "C4-multi-vdsp-f2", c0_on_cpu, False)]
    fails = 0
    for label, name, edit, expect_ok in cases:
        shutil.rmtree(ROOT / name, ignore_errors=True)
        manifest = {**base, "arms": [name], "runs": {name: place(name, edit)}}
        try:
            wp5sweep.verified_runs(manifest, name)
            ok, why = True, ""
        except SystemExit as exc:
            ok, why = False, str(exc)[:160]
        good = ok == expect_ok
        fails += not good
        print(f"{'PASS' if good else 'FAIL'} {label}: accepted={ok} {why}")
    shutil.rmtree(ROOT, ignore_errors=True)
    print(f"failed checks: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
