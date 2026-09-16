"""Exercise the real-size warm-start and accumulated-batch workers before launch."""

import datetime
import json
from pathlib import Path

from common import save
from recovery_control import prepare
from recovery_core import ROOT, read, source_hashes
from recovery_train import run_worker


def main():
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = prepare(ROOT / "setup" / ("fullsize-" + stamp))
    results = {}
    for job in ["R1", "R3"]:
        run_worker(run, job, 16, 600)
        r = read(run / "training" / job / "result.json")
        assert r["status"] == "completed" and r["presented"] == 16
        results[job] = dict(
            status=r["status"], presented=r["presented"], updates=r["updates"]
        )
    save(
        ROOT / "setup/fullsize-checks.json",
        dict(
            passed=True,
            run=str(run),
            jobs=results,
            source_hashes=source_hashes(Path(__file__).parent),
        ),
    )
    print(json.dumps(dict(passed=True, run=str(run))), flush=True)


if __name__ == "__main__":
    main()
