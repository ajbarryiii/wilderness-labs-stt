"""Quiet-host rerun of a selection session's TIMING protocols, plus the session's remaining gates and
audits (coordinator decision 2026-10-20 14:38 PDT, after sel-20261020b's timing was contaminated by host
load ~50 from 13:35; the contaminated outputs are under bench/contaminated/sel-20261020b-hostload50/).

Orchestration only. Every measurement is the session's own frozen command (gpu1bit.shared.dispatch with
the roster's patches), in the session's order: P-paired encoder then full scope chunks, P-clips/P-memory
forward then reverse roster order, phases chunks; then the gate (timing, sample) and audit steps exactly
as selection.session runs them (selection._run: freeze verified before every process, session ledger).
This file is outside the session freeze; nothing it defines is imported by a measurement; its SHA-256 is
recorded in every sidecar.

Timing steps: before EVERY process the host must be quiet: load1 (/proc/loadavg) < 2, no agent-jobs job
holds cpu-heavy (`job queue --json`), no compute process on the GPU (nvidia-smi), and the freeze
verifies. The check repeats every 30 s (no tight loop). While the process runs, a sampler records once
per second load1/5/15, runnable/total tasks, busy cores of this orchestrator's descendants (own) and of
every other process (foreign), and every 30 s the agent-jobs cpu-heavy holders. A chunk is ACCEPTED only
if it exits 0, its output is complete and bound (artifact_bound: protocol, run, variants, identities)
and load1 never exceeded 2.0; otherwise its output is moved to
selection/<run>/dropped/ and the step is retried at the next quiet moment.
RECOGNITION RUN (review c3): every worker gets the roster's identity_run as GPU1BIT_ACTPREC_RUN
(child_env), the guard gates that run's references, and artifacts must name the run and each
variant's reference.
RESUME (review c2): on (re)start the guard runs first; a step is skipped only if `reusable`: its
accepted sidecar is bound to the same roster, protocol (exact command) and identity map, its output is
byte-identical to the accepted one, and the output itself is bound. Any other existing output is moved
to dropped/ as stale and the step is measured again. RERUN COMPLETE requires every step reusable. Each step's sidecar with all
attempts: selection/<run>/host/<step>.json.
X2 BENCHMARK (plans/x2-default.md; roster "kind": "x2-benchmark", written by gpu1bit/x2/bench_session.py): the
same orchestration and acceptance for the six-arm roster b1 to b6. Its guard is bench_session.guard (the
session freeze with the Z2 pointer check, the candidate selection in use, the passed gate record and each
fulldev arm's gate-run identity), its binding adds each variant's roster id, model and complete export, and
--phase historical-test runs M1's historical test workloads only after the dev results are written (A13).
Gate and audit steps are numerics, not timing; they run while the host is busy (they do not wait).
--timing-only skips them (all done for sel-20261020b). Since 2026-10-20 16:40 the rerun runs as ONE
agent-jobs run (`job start gpu1bit-timing-selb --needs gpu --quiet-host --full-machine --priority normal`,
the user having granted full GPU control); the checks above stay as guards; --metadata records the job.

    gpu1bit/shared/python gpu1bit/rerun_timing.py --run sel-20261020b
    gpu1bit/shared/python gpu1bit/rerun_timing.py --roster-file <dir>/roster.json   (activation precision)
Roster-file mode (activation precision): the same protocols for the roster's variants and patches. The
guard before every process is identity_guard: every variant's frozen identity in the roster's
identity run must still verify. This file is hashed into the activation-precision identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from gpu1bit.shared import env  # noqa: E402

LOAD_LIMIT = 2.0
CHECK_EVERY_S = 30.0
PY = str(env.SHARED / "python")
CHUNKS = ["0-3", "4-7", "8-11", "12-15", "16-19", "20-23"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def loadavg() -> dict:
    a = Path("/proc/loadavg").read_text().split()
    running, total = a[3].split("/")
    return {"load1": float(a[0]), "load5": float(a[1]), "load15": float(a[2]), "running": int(running), "tasks": int(total)}


def agent_jobs() -> dict:
    """cpu-heavy and gpu holders from `job queue --json`; an unreadable queue counts as a holder (not quiet)."""
    try:
        doc = json.loads(subprocess.run(["job", "queue", "--json"], capture_output=True, text=True, timeout=30).stdout)
    except Exception as exc:  # noqa: BLE001
        return {"cpu_heavy": [{"error": repr(exc)}], "gpu": []}
    out = {"cpu_heavy": [], "gpu": []}
    for r in doc.get("resources", []):
        key = {"cpu-heavy": "cpu_heavy", "gpu": "gpu"}.get(r.get("name"))
        if key:
            out[key] = [{"name": h.get("name"), "owner": (h.get("owner") if isinstance(h.get("owner"), str) else None),
                         "held_s": h.get("held_s")} for h in r.get("holders", [])] + \
                       [{"foreign": f} for f in r.get("foreign", []) or []]
    return out


def gpu_compute_apps() -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception as exc:  # noqa: BLE001
        return [f"nvidia-smi failed: {exc!r}"]
    return [line.strip() for line in out.splitlines() if line.strip()]


def quiet_state() -> dict:
    return {"time": now(), "load": loadavg(), "agent_jobs": agent_jobs(), "gpu_compute_apps": gpu_compute_apps()}


def is_quiet(state: dict) -> bool:
    return (state["load"]["load1"] < LOAD_LIMIT and not state["agent_jobs"]["cpu_heavy"]
            and not state["gpu_compute_apps"])


def _proc_table() -> dict[int, tuple[int, int]]:
    """{pid: (ppid, utime+stime ticks)} for every process."""
    table = {}
    for d in os.scandir("/proc"):
        if d.name.isdigit():
            try:
                with open(f"/proc/{d.name}/stat") as f:
                    fields = f.read().rsplit(")", 1)[1].split()
                table[int(d.name)] = (int(fields[1]), int(fields[11]) + int(fields[12]))
            except (OSError, IndexError, ValueError):
                continue
    return table


def _descendants(table: dict[int, tuple[int, int]], root: int) -> set[int]:
    children: dict[int, list[int]] = {}
    for pid, (ppid, _) in table.items():
        children.setdefault(ppid, []).append(pid)
    out, stack = {root}, [root]
    while stack:
        for c in children.get(stack.pop(), []):
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.samples, self.jobs = [], []
        self.hz = os.sysconf("SC_CLK_TCK")

    def run(self):
        last, last_t, last_jobs = _proc_table(), time.time(), 0.0
        while not self.stop.wait(1.0):
            table, t = _proc_table(), time.time()
            mine = _descendants(table, os.getpid())
            own = foreign = 0
            for pid, (_, ticks) in table.items():
                if pid in last:
                    delta = ticks - last[pid][1]
                    if pid in mine:
                        own += delta
                    else:
                        foreign += delta
            dt = max(t - last_t, 1e-3)
            self.samples.append({"t": round(t, 2), **loadavg(), "own_busy_cores": round(own / self.hz / dt, 3),
                                 "foreign_busy_cores": round(foreign / self.hz / dt, 3)})
            last, last_t = table, t
            if t - last_jobs > 30:
                self.jobs.append({"t": round(t, 2), **agent_jobs()})
                last_jobs = t


def _x2(roster: dict) -> bool:
    """The recorded x2 benchmark roster (gpu1bit/x2/bench_session.py; plans/x2-default.md)."""
    return roster.get("kind") == "x2-benchmark"


def timing_steps(run: str, roster: dict) -> list[dict]:
    if _x2(roster):
        from gpu1bit.x2 import bench_session
        return bench_session.steps(roster, roster.get("phase", "dev"))
    disp = [PY, "-m", "gpu1bit.shared.dispatch", "--patches", ",".join(roster["patches"])]
    bench = env.ARTIFACTS / "bench"
    out = []
    for scope in ("encoder", "full"):
        for rng in CHUNKS:
            a, b = (int(v) for v in rng.split("-"))
            out.append({"name": f"paired-{scope}-{rng}", "output": bench / run / f"paired-{scope}-{a:02d}-{b:02d}.json",
                        "kind": "paired", "scope": scope, "range": rng, "cmd": disp + ["bench", "paired", "--agent", "H", "--run", run, "--scope", scope,
                                                     "--variants", *roster["variants"], "--utterances", rng]})
    for order, label in ((roster["variants"], "clips1"), (roster["variants"][::-1], "clips2")):
        for v in order:
            safe = v.replace(":", "_")
            out.append({"name": f"{label}-{safe}", "output": bench / f"{run}-{label}" / f"clips-{safe}.json", "clips": True,
                        "kind": "clips", "label": label, "variant": v,
                        "cmd": disp + ["bench", "clips", "--agent", "H", "--run", f"{run}-{label}", "--variant", v]})
    if roster.get("phase_pair"):
        for rng in CHUNKS:
            a, b = (int(v) for v in rng.split("-"))
            out.append({"name": f"phases-{rng}", "output": bench / run / f"phases-{a:02d}-{b:02d}.json", "range": rng,
                        "kind": "phases",
                        "cmd": disp + ["phases", "--agent", "H", "--run", run, "--variants", *roster["phase_pair"],
                                       "--utterances", rng]})
    return out


def output_complete(step: dict) -> bool:
    p = step["output"]
    if not p.exists():
        return False
    try:
        doc = json.loads(p.read_text())
    except ValueError:
        return False
    if step.get("clips"):
        return len(doc.get("workloads", [])) == 3 and "memory_samples_10ms" in doc
    if step.get("kind") == "kernel":            # P-kernel: every op at every M, finished (power check at the end)
        a = doc.get("args") or {}
        return (doc.get("protocol") == "P-kernel" and "power_limit_w_end" in doc
                and len(doc.get("rows", [])) == len(a.get("ops") or []) * len(a.get("M") or []))
    a, b = (int(v) for v in step["range"].split("-"))
    return {r["index"] for r in doc.get("rows", [])} >= set(range(a, b + 1))


# --- resume binding (review c2 of the activation-precision code) ------------------------------------

def make_binding(roster: dict, guard_result: dict) -> dict:
    """What every reused artifact must have been recorded under: the roster, the identity map, the
    recognition run and each variant's reference (roster-file mode; None in selection mode, where the
    freeze guard stands for the sources)."""
    bind = {"roster_sha256": hashlib.sha256(json.dumps(roster, sort_keys=True).encode()).hexdigest(),
            "identities": guard_result.get("identities"), "references": guard_result.get("references"),
            "identity_run": guard_result.get("identity_run"), "guard_files": guard_result.get("files")}
    if guard_result.get("labels") is not None:   # x2 benchmark: each variant's roster id, model and export
        bind["labels"] = guard_result["labels"]
    return bind


def step_binding(step: dict, bind: dict) -> dict:
    """A step's binding: the run binding plus its protocol, i.e. the exact command."""
    return {**bind, "cmd": [str(c) for c in step["cmd"]]}


def _info_bound(info: dict, variant: str, expected: dict) -> bool:
    """A variant's recorded info names its identity, the recognition run and its reference (arm, identity)."""
    return (info.get("identity") == expected["identities"].get(variant) and info.get("identity_run") == expected["identity_run"]
            and info.get("reference") == (expected.get("references") or {}).get(variant))


def artifact_bound(step: dict, doc: dict, roster: dict, expected: dict | None) -> bool:
    """The output names the requested protocol, run and variants, and (given a binding with identities,
    i.e. roster-file mode) each variant's identity, recognition run and reference."""
    if _x2(roster):
        from gpu1bit.x2 import bench_session
        return bench_session.artifact_bound(step, doc, roster, expected or {})
    run = roster["run"]
    args = doc.get("args") or {}
    identities = (expected or {}).get("identities")
    if step["kind"] == "paired":
        ok = (doc.get("protocol") == "P-paired" and doc.get("scope") == step["scope"] and doc.get("variants") == list(roster["variants"])
              and args.get("run") == run and args.get("utterances") == step["range"])
        if ok and identities is not None:
            info = doc.get("variants_info") or {}
            ok = all(_info_bound(info.get(v) or {}, v, expected) for v in roster["variants"])
        return ok
    if step["kind"] == "clips":
        ok = (doc.get("protocol") == "P-clips+P-memory" and doc.get("variant") == step["variant"]
              and args.get("run") == f"{run}-{step['label']}")
        if ok and identities is not None:
            ok = _info_bound(doc.get("variant_info") or {}, step["variant"], expected)
        return ok
    return (doc.get("protocol") == "phases" and doc.get("variants") == list(roster.get("phase_pair") or [])
            and args.get("run") == run and args.get("utterances") == step["range"])


def reusable(step: dict, record: dict, bind: dict, roster: dict) -> bool:
    """An accepted step whose sidecar binding, output bytes and output content all still match."""
    accepted = [a for a in record.get("attempts", []) if a.get("accepted")]
    if not record.get("accepted") or not accepted:
        return False
    last = accepted[-1]
    if last.get("binding") != step_binding(step, bind) or not step["output"].exists():
        return False
    if env.sha256_file(step["output"]) != last.get("output_sha256") or not output_complete(step):
        return False
    try:
        doc = json.loads(step["output"].read_text())
    except ValueError:
        return False
    return artifact_bound(step, doc, roster, bind)


def _sidecar(root: Path, step: dict) -> dict:
    path = root / "host" / f"{step['name']}.json"
    return json.loads(path.read_text()) if path.exists() else {"step": step["name"], "attempts": []}


def pending_steps(run: str, roster: dict, root: Path, guard) -> tuple[list[dict], dict]:
    """(steps to (re)measure, binding). The guard runs FIRST, before any step is considered reusable;
    a step that is not reusable has its stale output moved to <root>/dropped/."""
    bind = make_binding(roster, guard())
    pending = []
    for step in timing_steps(run, roster):
        if reusable(step, _sidecar(root, step), bind, roster):
            continue
        if step["output"].exists():
            (root / "dropped").mkdir(parents=True, exist_ok=True)
            shutil.move(str(step["output"]), root / "dropped" / f"{step['name']}-stale-{time.strftime('%Y%m%dT%H%M%S')}.json")
        pending.append(step)
    return pending, bind


def gate_steps(run: str, roster: dict) -> list[dict]:
    """selection.session's gate and audit steps (same commands, logs, completion predicates, retries)."""
    from gpu1bit.shared import selection, workloads
    disp = [PY, "-m", "gpu1bit.shared.dispatch", "--patches", ",".join(roster["patches"])]
    doc = workloads.load()
    n_timing = len({r["id"] for r in doc["dev_clips"] + doc["dev_24"]})
    out = []
    for impl in roster["candidates"]:
        safe = impl.replace(":", "_")
        out.append({"name": f"gate-timing-{safe}", "retries": 1,
                    "cmd": disp + ["gate", "timing", "--agent", "H", "--run", run, "--impl", impl],
                    "done": lambda i=impl: selection._gate_count(run, i, "timing") >= n_timing})
        out.append({"name": f"gate-sample-{safe}", "retries": 12,
                    "cmd": disp + ["gate", "sample", "--agent", "H", "--run", run, "--impl", impl, "--range", "0-511"],
                    "done": lambda i=impl: selection._gate_count(run, i, "sample") >= 512})
    for v in roster["variants"]:
        path = selection.sel_dir(run) / "audit" / f"{v.replace(':', '_')}.json"
        out.append({"name": f"audit-{v.replace(':', '_')}", "retries": 1,
                    "cmd": disp + ["audit", "--run", run, "--variant", v], "done": lambda p=path: p.exists()})
    return out


def identity_guard(roster: dict) -> dict:
    """Roster-file mode's guard before every process: each variant's identity, recomputed with the
    activation-precision patch installed, must equal the one frozen in the roster's identity run
    (fulldev.check_identity raises SystemExit otherwise). Same keys as selection.verify_freeze. The x2
    benchmark roster has its own guard (bench_session.guard: freeze, pointer, gate record, identities)."""
    if _x2(roster):
        from gpu1bit.x2 import bench_session
        return bench_session.guard(roster)
    from gpu1bit.actprec import parts
    from gpu1bit.shared import fulldev
    parts.install()
    run = roster["identity_run"]
    hashes = {arm: fulldev.check_identity(run, arm, "full")["hash"] for arm in roster["variants"]}
    refs = {}
    for arm in roster["variants"]:   # review c3: the requested run's references are gated before any reuse
        s = parts.require_reference(run, arm)
        refs[arm] = {"arm": s["arm"], "identity": s["identity"]}
    return {"files": len(hashes), "mismatched": [], "added": [], "ok": True, "identities": hashes,
            "references": refs, "identity_run": run}


def child_env(roster: dict) -> dict:
    """A worker's environment: CUDA visible, no CUBLAS workspace override, and the roster's recognition
    run passed explicitly (overriding any inherited value; removed in selection mode). Review c3."""
    e = dict(os.environ)
    e.pop("CUBLAS_WORKSPACE_CONFIG", None)
    e.pop("CUDA_VISIBLE_DEVICES", None)
    e.pop("GPU1BIT_ACTPREC_RUN", None)
    if roster.get("identity_run"):
        e["GPU1BIT_ACTPREC_RUN"] = roster["identity_run"]
    if _x2(roster):                              # the candidate selection document, explicitly (Y1)
        from gpu1bit.x2 import bench_session
        e.update(bench_session.child_env_updates(roster))
    return e


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=None, help="selection run (roster and freeze of selection/<run>)")
    p.add_argument("--roster-file", default=None,
                   help="roster JSON instead of a selection run: run, variants, patches, phase_pair, identity_run; "
                        "the guard is identity_guard, sidecars go next to the file, no gate steps")
    p.add_argument("--give-up-h", type=float, default=12.0, help="exit if no timing step could start for this long")
    p.add_argument("--max-attempts", type=int, default=6, help="per timing step")
    p.add_argument("--timing-only", action="store_true", help="only the timing steps (the session's gates and audits are done)")
    p.add_argument("--metadata", default="{}", help="JSON object recorded in every sidecar attempt (e.g. the supervising job)")
    p.add_argument("--phase", default="dev", choices=("dev", "historical-test"),
                   help="x2 benchmark rosters only: the dev workloads, or M1's historical test workloads (A13)")
    args = p.parse_args()
    meta = json.loads(args.metadata)
    meta["agent_jobs_env"] = {k: v for k, v in os.environ.items() if k.startswith("AGENT_JOBS")}
    meta["pid"], meta["nice"] = os.getpid(), os.nice(0)
    from gpu1bit.shared import selection
    if (args.run is None) == (args.roster_file is None):
        p.error("give exactly one of --run and --roster-file")
    if args.roster_file:
        roster = json.loads(Path(args.roster_file).read_text())
        run, root = roster["run"], Path(args.roster_file).resolve().parent
        if _x2(roster):
            roster["phase"] = args.phase
            os.environ["GPU1BIT_SELECTION"] = roster["selection"]["path"]   # the guard checks the document in use
        elif args.phase != "dev":
            p.error("--phase applies to x2 benchmark rosters only")
        meta["roster"] = roster
        guard = lambda: identity_guard(roster)  # noqa: E731
    else:
        run = args.run
        roster = selection.load_roster(run)
        root = selection.sel_dir(run)
        guard = lambda: selection.verify_freeze(run)  # noqa: E731
    for d in ("host", "dropped", "logs"):
        (root / d).mkdir(parents=True, exist_ok=True)
    ledger_path = root / "ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    me = env.sha256_file(Path(__file__))
    log_file = open(root / "rerun_timing.log", "a")

    def log(msg: str) -> None:
        line = f"{now()} {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    def sidecar(step):
        path = root / "host" / f"{step['name']}.json"
        return path, (json.loads(path.read_text()) if path.exists() else {"step": step["name"], "run": run, "attempts": []})

    timing, bind = pending_steps(run, roster, root, guard)   # the guard runs before any skip (review c2)
    gates = [] if (args.timing_only or args.roster_file) else [g for g in gate_steps(run, roster) if not g["done"]()]
    log(f"start: orchestrator {me[:12]}, {len(timing)} timing steps, {len(gates)} gate/audit steps pending, "
        f"metadata {json.dumps(meta)}")
    last_progress, last_wait_log = time.time(), 0.0
    while timing or gates:
        state = quiet_state()
        if timing and is_quiet(state):
            step = timing[0]
            path, record = sidecar(step)
            freeze = guard()
            e = child_env(roster)
            started = time.time()
            sampler = Sampler()
            with open(root / "logs" / f"rerun-{step['name']}.log", "a") as out:
                out.write(f"\n### {now()} {' '.join(step['cmd'])}\n")
                out.flush()
                sampler.start()
                rc = subprocess.run(["timeout", "600", *step["cmd"]], stdout=out, stderr=subprocess.STDOUT,
                                    cwd=str(env.BINARY_DIR), env=e).returncode
                sampler.stop.set()
                sampler.join()
            s = sampler.samples
            load_max = max([x["load1"] for x in s] + [state["load"]["load1"]])
            complete = output_complete(step)
            try:
                bound = complete and artifact_bound(step, json.loads(step["output"].read_text()), roster, bind)
            except ValueError:
                bound = False
            accepted = rc == 0 and complete and bound and load_max <= LOAD_LIMIT
            entry = {"attempt": len(record["attempts"]), "rc": rc, "output_complete": complete, "output_bound": bound,
                     "binding": step_binding(step, bind),
                     "output_sha256": env.sha256_file(step["output"]) if step["output"].exists() else None,
                     "started": time.strftime(
                         "%Y-%m-%dT%H:%M:%S", time.localtime(started)), "seconds": round(time.time() - started, 1),
                     "pre_check": state, "freeze": {k: freeze[k] for k in ("files", "mismatched", "added", "ok", "identities", "references", "identity_run") if k in freeze},
                     "post_check": quiet_state(), "load1_max": load_max,
                     "own_busy_cores_max": max((x["own_busy_cores"] for x in s), default=None),
                     "foreign_busy_cores_max": max((x["foreign_busy_cores"] for x in s), default=None),
                     "agent_jobs_during": sampler.jobs, "samples": s, "accepted": accepted,
                     "rule": f"accepted iff rc 0, output complete and bound to the roster, protocol and identities, and "
                             f"load1 <= {LOAD_LIMIT} at the pre-check and every 1 s sample",
                     "orchestrator_sha256": me, "run_metadata": meta}
            if not accepted and step["output"].exists():
                dest = root / "dropped" / f"{step['name']}-attempt{entry['attempt']}-{time.strftime('%H%M%S')}.json"
                shutil.move(str(step["output"]), dest)
                entry["moved_to"] = str(dest)
            record["attempts"].append(entry)
            record["accepted"] = accepted
            path.write_text(json.dumps(record) + "\n")
            log(f"TIMING {step['name']} attempt {entry['attempt']}: rc {rc}, complete {complete}, load1 max {load_max}, "
                f"own cores max {entry['own_busy_cores_max']}, foreign cores max {entry['foreign_busy_cores_max']}, "
                f"{'ACCEPTED' if accepted else 'DROPPED'}")
            last_progress = time.time()
            if accepted:
                timing.pop(0)
            elif len(record["attempts"]) >= args.max_attempts:
                log(f"STOP {step['name']}: no clean attempt in {args.max_attempts}")
                raise SystemExit(2)
            continue
        if gates:
            g = gates[0]
            if g["done"]():
                gates.pop(0)
                continue
            g["tries"] = g.get("tries", 0) + 1
            if g["tries"] > g["retries"] + 1:
                log(f"STOP gate step failed: {g['name']}")
                raise SystemExit(3)
            rc = selection._run(["timeout", "600", *g["cmd"]], root / "logs" / f"{g['name']}.log", ledger, run)
            ledger_path.write_text(json.dumps(ledger, indent=1) + "\n")
            log(f"GATE {g['name']} try {g['tries']}: rc {rc}, done {g['done']()}")
            if g["done"]():
                gates.pop(0)
            continue
        if time.time() - last_progress > args.give_up_h * 3600:
            log(f"STOP host not quiet for {args.give_up_h} h: {json.dumps(state)}")
            raise SystemExit(4)
        if time.time() - last_wait_log > 600:
            log(f"WAIT host busy: load1 {state['load']['load1']}, cpu-heavy {[h.get('name') for h in state['agent_jobs']['cpu_heavy']]}, "
                f"gpu apps {state['gpu_compute_apps']}")
            last_wait_log = time.time()
        time.sleep(CHECK_EVERY_S)
    unbound = [s["name"] for s in timing_steps(run, roster) if not reusable(s, _sidecar(root, s), bind, roster)]
    if unbound:
        log(f"STOP outputs not bound to this roster / protocol / identities at the end: {unbound}")
        raise SystemExit(5)
    log("RERUN COMPLETE")


if __name__ == "__main__":
    main()
