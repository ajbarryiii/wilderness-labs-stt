"""Self-test of the sweep report's run validation (WP7 reviews r2 finding 5, r3 finding 3); NixOS, no Mac, no timing.

Builds throwaway results trees from WP5's C4-multi-vdsp-f2 run (64 natural clips) and checks verified_runs:
- legacy manifests (WP5 record format): the right arm is accepted; another arm's records under this arm's name and
  a C0 run on cpuOnly are refused;
- non-legacy WP7 manifests (the load records rewritten to the WP7 format: pipeline-record eligibility, executable,
  settle, pairing block naming the arm): the right arm is accepted; refused are a cached run in replay mode, on the
  wrong clip, paired with C0, header-only, with two calls, and an arm run without its end record.
The trees are removed before the run and in a `finally`.

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
EXE = "e" * 64


def lines(d: Path, f: str) -> list[dict]:
    return [json.loads(l) for l in (d / f).read_text().splitlines() if l.strip()]


def write(d: Path, f: str, recs: list[dict]) -> None:
    (d / f).write_text("".join(json.dumps(r) + "\n" for r in recs))


def edit_load(d: Path, f: str, fn) -> None:
    write(d, f, [fn(r) if r.get("record") == "load" else r for r in lines(d, f)])


def place(name: str, run: str, edit=None) -> dict:
    d = ROOT / name / run
    shutil.copytree(SRC, d)
    (d / "MIGRATED").unlink(missing_ok=True)
    if edit:
        edit(d)
    return {"A" if run != "wp5-20261003" else "all": {"run": d.name, "status": 0, "dir": str(d)}}


def to_wp7(d: Path) -> None:
    """WP5 records -> the WP7 load-record format the runner now writes."""
    a = wp5sweep.spec_of("C4-multi-vdsp-f2")
    elig = {"pipeline_record": wp5sweep.pipeline_record(a).name,
            "components": {"compute_units": "cpuAndNeuralEngine", "encoder": {"model": "mp2", "arm": "C4", "variant": "multi"}}}
    for f in ("arm.jsonl", "cached.jsonl"):
        edit_load(d, f, lambda r: {**r, "eligibility": elig, "settle_ms": 500, "executable_sha256": EXE})
    edit_load(d, "c0.jsonl", lambda r: {**r, "settle_ms": 500, "executable_sha256": EXE,
                                        "pairing": {**(r.get("pairing") or {}), "arm": "C4-multi-vdsp-f2"}})
    edit_load(d, "arm.jsonl", lambda r: {**r, "pairing": {**(r.get("pairing") or {}), "arm": "C4-multi-vdsp-f2"}})


def c0_on_cpu(d: Path) -> None:
    edit_load(d, "c0.jsonl", lambda r: {**r, "compute_units": "cpuOnly"})


def faults():
    def cached_replay(d): edit_load(d, "cached.jsonl", lambda r: {**r, "mode": "replay"})
    def cached_clip(d): edit_load(d, "cached.jsonl", lambda r: {**r, "clip_ids": ["n02-1272-141231-0016"]})
    def cached_paired(d): edit_load(d, "cached.jsonl", lambda r: {**r, "pairing": {"paired_with": "C0", "arm": "x"}})
    def cached_header_only(d): write(d, "cached.jsonl", [r for r in lines(d, "cached.jsonl") if r.get("record") == "load"])
    def cached_two_calls(d):
        recs = lines(d, "cached.jsonl")
        write(d, "cached.jsonl", recs[:2] + [recs[1]] + recs[2:])
    def arm_no_end(d): write(d, "arm.jsonl", [r for r in lines(d, "arm.jsonl") if r.get("record") != "end"])
    return [("cached run in replay mode", cached_replay), ("cached run on another clip", cached_clip),
            ("cached run paired with C0", cached_paired), ("cached run header-only", cached_header_only),
            ("cached run with two calls", cached_two_calls), ("arm run without end record", arm_no_end)]


def main() -> int:
    shutil.rmtree(ROOT, ignore_errors=True)  # also clears what an interrupted earlier run left behind
    try:
        return run()
    finally:  # disposable, removed on every exit path Python sees (review r3 finding 5)
        shutil.rmtree(ROOT, ignore_errors=True)


def check(label: str, manifest: dict, name: str, expect_ok: bool) -> int:
    try:
        wp5sweep.verified_runs(manifest, name)
        ok, why = True, ""
    except SystemExit as exc:
        ok, why = False, str(exc)[:150]
    good = ok == expect_ok
    print(f"{'PASS' if good else 'FAIL'} {label}: accepted={ok} {why}")
    return 0 if good else 1


def run() -> int:
    wp5sweep.LOCAL = ROOT
    natural = [c["id"] for c in json.loads((IOS / "clips.json").read_text())["clips"] if c["kind"] == "natural"]
    fails = 0
    legacy = {"legacy": "selftest", "build": None, "halves": {"all": natural}}
    for label, name, edit, expect_ok in (("legacy: right arm", "C4-multi-vdsp-f2", None, True),
                                         ("legacy: another arm's records", "C3-multi-vdsp-f2", None, False),
                                         ("legacy: C0 on cpuOnly", "C4-multi-vdsp-f2", c0_on_cpu, False)):
        shutil.rmtree(ROOT / name, ignore_errors=True)
        fails += check(label, {**legacy, "arms": [name], "runs": {name: place(name, "wp5-20261003", edit)}}, name, expect_ok)
    name = "C4-multi-vdsp-f2"
    wp7 = {"build": {"executable_sha256": EXE, "commit": "c"}, "settle_ms": 500, "warmups": 3, "timed": 10,
           "halves": {"A": natural}, "arms": [name]}
    for i, (label, fault) in enumerate([("WP7: right arm", None), *faults()]):
        run_id = f"wp7-{i}"
        runs = place(name, run_id, lambda d, f=fault: (to_wp7(d), f and f(d)))
        fails += check(label, {**wp7, "runs": {name: runs}}, name, fault is None)
    print(f"failed checks: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
