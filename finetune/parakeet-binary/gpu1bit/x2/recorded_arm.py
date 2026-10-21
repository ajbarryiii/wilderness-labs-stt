"""Recorded-arm binding (plans/x2-default.md Z4, Z7): read a historical fulldev arm as DATA, without
recomputing its identity from current sources, for descriptive comparisons only. It never gates.

`verify_recorded_arm(run, arm, pins)` accepts the arm only if:
  - identity.json's hash equals summary.json's identity and every row's identity (no mixed rows);
  - the summary names the run and arm, the full subset, complete per-set counts equal to the dev
    manifests' counts, every utterance exactly once;
  - its x3 reference's recorded summary is complete, structurally valid and passed R2;
  - every pinned file's SHA-256 equals the hash pinned in the plan before the run (PINS block);
  - every row file's ctime is earlier than the arm's last chunk `finished` + 1 s (exclusive; `finished`
    is truncated to whole seconds, Z7).
`compare_with_recorded(run, arm, rec_run, rec_arm, pins)` compares the current run's arm (identity
verified by fulldev.check_identity) row by row with the recorded arm and writes a descriptive
comparison storing both runs and both identities.

The pins are read from plans/x2-default.md, the fenced block tagged `pinned-ap-20261020r4`
("<sha256>  <run>/<arm>/<file>" per line), so the plan is the single source.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

from gpu1bit.shared import env, fulldev

PLAN = env.BINARY_DIR / "plans" / "x2-default.md"
PIN_BLOCK = "pinned-ap-20261020r4"


def plan_pins(plan: Path = PLAN, block: str = PIN_BLOCK) -> dict[str, str]:
    """{"<run>/<arm>/<file>": sha256} from the plan's fenced pin block."""
    text = plan.read_text()
    m = re.search(rf"```{re.escape(block)}\n(.*?)```", text, re.S)
    if not m:
        raise ValueError(f"{plan}: no ```{block} block")
    pins = {}
    for line in m.group(1).splitlines():
        if line.strip():
            digest, rel = line.split(None, 1)
            pins[rel.strip()] = digest
    return pins


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _refuse(run: str, arm: str, why: str):
    raise ValueError(f"recorded arm {run}/{arm} refused: {why}")


def verify_recorded_arm(run: str, arm: str, pins: dict[str, str], *, root: Path | None = None) -> dict:
    root = Path(root or fulldev.ROOT)
    d = root / run / arm
    ident = json.loads((d / "identity.json").read_text())
    summary = json.loads((d / "summary.json").read_text())
    h = ident.get("hash")
    if summary.get("identity") != h:
        _refuse(run, arm, f"summary identity {str(summary.get('identity'))[:12]} != identity.json {str(h)[:12]}")
    want = {n: len(fulldev.subset_records(n, "full")) for n in fulldev.DEV_SETS}
    if (summary.get("run"), summary.get("arm"), summary.get("subset")) != (run, arm, "full") or summary.get("complete") is not True \
            or summary.get("utterances") != want:
        _refuse(run, arm, "summary is not the complete full-subset summary of this arm")
    rows_by_set, row_files = {}, []
    for n in fulldev.DEV_SETS:
        p = d / f"{n}.jsonl"
        row_files.append(p)
        rows = fulldev.load_rows(p) if p.exists() else []
        ids = [r["id"] for r in rows]
        if len(rows) != want[n] or len(set(ids)) != len(ids):
            _refuse(run, arm, f"{n}: {len(rows)} rows ({len(set(ids))} distinct), expected {want[n]}")
        mixed = {r.get("identity") for r in rows} - {h}
        if mixed:
            _refuse(run, arm, f"{n}: rows of other identities {sorted(map(str, mixed))[:2]}")
        rows_by_set[n] = rows
    rel = lambda p: f"{run}/{arm}/{p.name}"  # noqa: E731
    pinned_here = {k: v for k, v in pins.items() if k.startswith(f"{run}/{arm}/")}
    for p in row_files:
        if rel(p) not in pinned_here:
            _refuse(run, arm, f"{p.name} is not pinned in the plan")
    for k, digest in pinned_here.items():
        p = root / k
        if not p.exists() or _sha(p) != digest:
            _refuse(run, arm, f"{k}: SHA-256 differs from the plan's pin")
    chunks = [json.loads(l) for l in (d / "chunks.jsonl").read_text().splitlines() if l.strip()]
    if not chunks or any(c.get("identity") != h for c in chunks):
        _refuse(run, arm, "chunks missing or of another identity")
    last = max(time.mktime(time.strptime(c["finished"], "%Y-%m-%dT%H:%M:%S")) for c in chunks)
    late = [p.name for p in row_files if os.stat(p).st_ctime >= last + 1.0]
    if late:
        _refuse(run, arm, f"row files changed after the last chunk (+1 s): {late}")
    ref = fulldev.reference_arm(arm)
    if ref is None:
        if not ((summary.get("cross_check") or {}).get("pass") and summary.get("structurally_valid") is True):
            _refuse(run, arm, "this x3 arm did not pass R2")
    else:
        for k, digest in pins.items():
            if k.startswith(f"{run}/{ref}/") and (not (root / k).exists() or _sha(root / k) != digest):
                _refuse(run, arm, f"{k} (its x3 reference): SHA-256 differs from the plan's pin")
        rs = json.loads((root / run / ref / "summary.json").read_text())
        if (rs.get("run"), rs.get("arm"), rs.get("subset"), rs.get("complete")) != (run, ref, "full", True) \
                or rs.get("structurally_valid") is not True or not (rs.get("cross_check") or {}).get("pass"):
            _refuse(run, arm, f"its x3 reference {ref} did not pass R2 or is incomplete")
        rid = json.loads((root / run / ref / "identity.json").read_text())["hash"]
        if rs.get("identity") != rid:
            _refuse(run, arm, f"its x3 reference {ref}: summary identity differs from identity.json")
    return {"run": run, "arm": arm, "identity": h, "rows": rows_by_set, "last_chunk": last,
            "pinned_files": sorted(pinned_here), "binding": "recorded (not recomputed from current sources; Z4)"}


def compare_with_recorded(run: str, arm: str, rec_run: str, rec_arm: str, pins: dict[str, str] | None = None,
                          *, out_dir: Path | None = None) -> dict:
    """Descriptive row-by-row comparison of a current arm with a recorded arm (never gates)."""
    pins = plan_pins() if pins is None else pins
    rec = verify_recorded_arm(rec_run, rec_arm, pins)
    cur_id = fulldev.check_identity(run, arm, "full")["hash"]
    out = {"descriptive_only": True, "current": {"run": run, "arm": arm, "identity": cur_id},
           "recorded": {"run": rec_run, "arm": rec_arm, "identity": rec["identity"], "binding": rec["binding"]},
           "sets": {}}
    tot = diff = 0
    for n in fulldev.DEV_SETS:
        cur = {r["id"]: r for r in fulldev.summarize_set(n, fulldev.load_rows(fulldev.work_path(run, arm, n)), "full", cur_id)["records"]}
        old = {r["id"]: r for r in rec["rows"][n]}
        if set(cur) != set(old):
            raise ValueError(f"{n}: the two arms cover different utterances")
        keys = ("tokens", "hyp", "enc_len", "timestamps", "finite", "decode_error")
        d = [u for u in sorted(cur) if any(cur[u].get(k) != old[u].get(k) for k in keys)]
        out["sets"][n] = {"utterances": len(cur), "differing": len(d), "differing_ids": d[:100],
                          "token_differences": sum(cur[u]["tokens"] != old[u]["tokens"] for u in cur)}
        tot += len(cur)
        diff += len(d)
    out["pooled"] = {"utterances": tot, "differing": diff}
    dest = Path(out_dir or (fulldev.ROOT / run / "compare"))
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{arm}--vs--recorded-{rec_run}-{rec_arm}.json").write_text(json.dumps(out, indent=1) + "\n")
    return out
