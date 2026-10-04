"""Preconditions for Phase 3 test scoring (plans/test-scoring.md). Exits 1 on any failure.

Reads only, except that it writes its result to eval/M1-precheck.json (pass or fail).

    python precheck_scoring.py            # writes eval/M1-precheck.json, prints PASS/FAIL per check

Proves the export about to be scored is M1's final checkpoint, intact, built by the frozen code,
and that the training unit finished successfully. Uses no GPU.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import paths

RUN = paths.RUNS / "main-M1-P2-lr5e-4"
EXPORT = RUN / "export"
FROZEN = paths.ARTIFACTS / "frozen-code-20261001T0609Z" / "SHA256SUMS"
EXPECT = {"run_name": "main-M1-P2-lr5e-4", "arm": "M1", "recipe": "P2", "lr": 5e-4,
          "select": "final", "selected_step": 250000}
IDENTITY_KEYS = ("run_name", "arm", "recipe", "lr", "select", "selected_step", "config_sha256")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def last_invocation(unit: str) -> tuple[bool, str]:
    """(succeeded, detail) for the unit's most recent invocation, from the user journal.

    A transient service that exits 0 logs only "Started" and "<unit>: Consumed ..." for its
    invocation; a failure adds "Failed with result" (and a non-zero "Main process exited").
    Earlier invocations are ignored so an earlier failure cannot disqualify a later success.
    """
    out = subprocess.run(["journalctl", "--user", "-u", unit, "--no-pager", "-o", "json"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return False, f"journalctl failed: {out.stderr.strip()[:200]}"
    entries = [json.loads(line) for line in out.stdout.splitlines() if line.strip()]
    ids = [e.get("USER_INVOCATION_ID") or e.get("INVOCATION_ID") or e.get("_SYSTEMD_INVOCATION_ID") for e in entries]
    ids = [i for i in ids if i]
    if not ids:
        return False, "no journal entries with an invocation id"
    last = ids[-1]
    msgs = [str(e.get("MESSAGE", "")) for e in entries
            if (e.get("USER_INVOCATION_ID") or e.get("INVOCATION_ID") or e.get("_SYSTEMD_INVOCATION_ID")) == last]
    failed = any("Failed with result" in m or ("Main process exited" in m and "status=0/" not in m) for m in msgs)
    finished = any(": Consumed " in m for m in msgs)
    return finished and not failed, f"invocation {last}: " + " | ".join(m[:80] for m in msgs[-3:])


def active_parakeet_units() -> list[str] | None:
    """Active parakeet-* units, or None if the query itself failed."""
    out = subprocess.run(["systemctl", "--user", "list-units", "--state=active", "--no-legend",
                          "--plain", "parakeet-*"], capture_output=True, text=True)
    if out.returncode != 0:
        return None
    return [line.split()[0] for line in out.stdout.splitlines() if line.strip()]


def main() -> int:
    checks: list[tuple[str, bool, str]] = []
    actual = None

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    try:
        actual = run_checks(check)
    except Exception as error:  # a missing or unreadable artifact is a failed precondition
        check("required artifacts present and readable", False, f"{type(error).__name__}: {error}")
    return report(checks, actual)


def run_checks(check) -> str:
    ok, detail = last_invocation("parakeet-main.service")
    check("main unit's last invocation finished successfully (journal)", ok, detail)
    sweep_lines = [l for l in (paths.ARTIFACTS / "logs" / "main.log").read_text(errors="replace").splitlines()
                   if l.startswith("[sweep")]
    check("the last sweep log line is 'done'", bool(sweep_lines) and sweep_lines[-1].rstrip().endswith("done"),
          sweep_lines[-1][:120] if sweep_lines else "no sweep lines")
    units = active_parakeet_units()
    check("no parakeet-* unit active (query succeeded)", units == [],
          "systemctl query failed" if units is None else ", ".join(units))

    summary = json.loads((RUN / "summary.json").read_text())
    manifest = json.loads((EXPORT / "manifest.json").read_text())
    extra = manifest.get("extra", {})
    for key, want in EXPECT.items():
        check(f"summary.{key} == {want!r}", summary.get(key) == want, repr(summary.get(key)))
    check("summary steps == max_steps == 250000",
          summary.get("steps") == summary.get("max_steps") == 250000,
          f"{summary.get('steps')} / {summary.get('max_steps')}")
    check("summary.quantized", summary.get("quantized") is True)
    check("summary.scored_on is the rebuilt export", summary.get("scored_on") == "model rebuilt from export",
          repr(summary.get("scored_on")))
    for key in IDENTITY_KEYS:
        check(f"manifest.extra.{key} == summary.{key}", extra.get(key) == summary.get(key),
              f"{extra.get(key)!r} vs {summary.get(key)!r}")
    weights = EXPORT / manifest["file"]
    actual = sha256(weights)
    check("export.safetensors SHA-256 matches manifest", actual == manifest["sha256"], actual)
    recon = json.loads((EXPORT / "reconstruction.json").read_text())
    check("reconstruction codes and scales exact", recon.get("codes_exact") and recon.get("scales_exact"))
    srecon = summary.get("reconstruction") or {}
    check("summary.reconstruction matches reconstruction.json",
          all(srecon.get(k) == recon.get(k) for k in ("codes_exact", "scales_exact", "encoder_max_abs_diff",
                                                     "joint_output_max_abs_diff", "greedy_hyps_equal")))
    root = paths.REPO / "finetune"
    mismatched = [k for k, v in (summary.get("source_hashes") or {}).items() if sha256(root / k) != v]
    check("training/scoring sources unchanged since M1 launch", not mismatched and summary.get("source_hashes"),
          ", ".join(mismatched))
    frozen = dict(reversed(line.split()) for line in FROZEN.read_text().splitlines() if line.strip())
    check("evaluate.py equals the frozen snapshot", frozen.get("evaluate.py") == sha256(paths.HERE / "evaluate.py"))
    return actual


def report(checks: list[tuple[str, bool, str]], actual: str | None) -> int:
    result = {"pass": bool(checks) and all(ok for _, ok, _ in checks), "export_sha256": actual,
              "checks": [{"check": n, "ok": ok, "detail": d} for n, ok, d in checks]}
    out = paths.EVAL / "M1-precheck.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    for n, ok, d in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {n}" + (f"  [{d}]" if d and not ok else ""))
    print("ALL PASS" if result["pass"] else "PRECHECK FAILED", "->", out)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
