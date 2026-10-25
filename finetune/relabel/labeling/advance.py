"""Queue the next jobs of stage 2 Part C when the previous ones finish, so that labeling starts without a human between
jobs (labeling plan "GPU probe", "Equivalence check", "Lease and chunking", "Gate to stage 3").

    ./python -m labeling.advance checks [--dry-run]   after the six GPU probes: the consistency gate at the fastest
                                                      setting of each engine and the equivalence check, then `c2`
    ./python -m labeling.advance c2 [--dry-run]       after the checks: the selection, c2-init --run main, the chain
                                                      of C2 chunk jobs and the gate to stage 3

checks: all six probes (probes/gpu/<engine>-<i>/probe.json) must exist with a positive audio_hours_per_gpu_hour, else
exit 2. A vLLM probe whose two passes at the production setting differed (split.signals.repeat_changed > 0) stops
with exit 3: the plan has a non-zero share reported to the architect before C2. Otherwise V and T are the fastest vLLM
and transformers settings (ties: the lower index). Each one's declared VRAM is its probe's declared_vram_gb_for_c2
rounded up to 0.1 GB, never below it, so select_production accepts the consistency result; the equivalence check
runs at V's (the cap C2 will use), and vLLM's memory fraction must fit under it (c2.init_run's check). Host memory of
every GPU job (consistency, equivalence, C2 chunks; standing rule: measured peak + 25%): the largest host peak in the
six probes' teacher and b0 monitors, the larger of the process tree's summed RSS and the cgroup's memory.current
(which counts page cache, so this errs high; the split step and merge are not monitored), x MEM_HEADROOM, rounded
up to a whole GB, within [MEM_FLOOR_GB, MEM_CAP_GB] (launch.sh's cap). The jobs carry launch.sh's flags, lease rule
(ceil(VRAM) + 1 GB) and timeouts, chained so that one GPU job runs at a time:

    relabel-consistency-prepare -> relabel-consistency-vllm-V -> relabel-consistency-transformers-T --+
    relabel-consistency-prepare -> relabel-equivalence-prepare --------------------------------------+-> ...
    ... -> relabel-equivalence -> relabel-advance-c2

The transformers consistency is the fallback should vLLM fail the equivalence check. The consistency and equivalence
commands exit 0 whether or not their check passes (they fail only when a decode step fails); `select` decides.

c2: select_production over the inputs `./python -m labeling select` loads, loaded the same way (load_selection);
nothing selected: exit 3. The selected contract maps to exactly one engine setting (else exit 2); c2-init --run main
with it through the CLI's own code path (a run.json already under the same contract is accepted, under another one
refused). N = ceil(c2_cost GPU-hours / CHUNK_MINUTES hours) + SPARE_CHUNKS chunk jobs: the probe's rate already
amortizes the per-chunk start-up over a CHUNK_MINUTES chunk, and a chunk with nothing left takes the GPU lock, runs
two steps that find no group, and rewrites the manifest. The chain is launch.sh `c2` with the checks' memory cap; the
gate (launch.sh `gate`) follows the last chunk, with the selected setting's consistency result and, for vLLM, the
equivalence result.

Each step writes labeling/advance/<step>.json (inputs, decisions, the jobs and the names submitted so far) before and
while it submits; --dry-run prints the decisions and the argv lists it would submit and writes nothing. A re-run under
the same decisions submits only what is not yet submitted (nothing after a complete submission, provided every
recorded job still exists); a record of submitted jobs under other decisions is refused. The name being submitted is
recorded (in_flight) before its `job start`; if the driver is killed inside it, a re-run cannot tell whether the job
started, so it refuses until the record is fixed by hand.

Exit codes: 0 submitted, or nothing left to submit; 1 a submission or c2-init failed (a re-run resumes); 2 an input is
missing or invalid; 3 a stop (vLLM repeat noise, or no qualifying setting); 4 refused (a record or run.json under
other decisions, a recorded job that no longer exists, or an interrupted submission).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from labeling import c2, checks, layout

RUN = "main"
ENGINES = ("vllm", "transformers")
PROBES = tuple((e, i) for e in ENGINES for i in range(3))
MONITORS = ("teacher_monitor", "b0_monitor")
HOST_FIELDS = ("peak_tree_rss_gb", "peak_cgroup_memory_gb")
MEM_HEADROOM, MEM_FLOOR_GB, MEM_CAP_GB = 1.25, 12, 36
SPARE_CHUNKS = 2
GPU_FLAGS = ("--stall", "30m", "--env", "HF_HUB_OFFLINE=1")
OK, FAILED, INPUT, STOP, REFUSED = 0, 1, 2, 3, 4


# --- paths ---------------------------------------------------------------------------------------------

def record_path(step: str) -> Path:
    return layout.advance_dir() / f"{step}.json"


def probe_path(engine: str, i: int) -> Path:
    return layout.probes_dir() / "gpu" / f"{engine}-{i}" / "probe.json"


def consistency_result_path(engine: str, i: int) -> Path:
    return layout.consistency_dir("dev") / f"result-{engine}-{i}.json"


def equivalence_result_path() -> Path:
    return layout.equivalence_dir("main") / "result.json"


def _load(path: Path) -> dict | None:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


# --- rules ---------------------------------------------------------------------------------------------

def tenth_up(x: float) -> float:
    """x rounded up to a multiple of 0.1 (GB), never below x; the result prints as its tenth."""
    r = math.ceil(round(x * 10, 6)) / 10
    while r < x:
        r = round(r + 0.1, 1)
    return r


def lease(vram_gb: float) -> str:
    """launch.sh's lease: ceil(VRAM) + 1 GB (the supervisor's fraction is over the nominal size; engines.vram_fraction)."""
    return f"{math.ceil(vram_gb) + 1}G"


def gb_arg(v: float) -> str:
    """A GB value as a CLI argument that parses back to the same float."""
    return repr(float(v))


def host_peak_gb(probe: dict) -> float | None:
    """The largest host memory peak in a probe's teacher and b0 monitors (tree RSS or cgroup memory, the larger)."""
    vals = [m[k] for m in (probe.get(n) or {} for n in MONITORS) for k in HOST_FIELDS if m.get(k) is not None]
    return max(vals) if vals else None


def job_mem_gb(peak_gb: float) -> int:
    """Host memory cap of a GPU job: peak x MEM_HEADROOM, up to a whole GB, within [MEM_FLOOR_GB, MEM_CAP_GB]."""
    return min(MEM_CAP_GB, max(MEM_FLOOR_GB, math.ceil(round(peak_gb * MEM_HEADROOM, 6))))


def job_start(name: str, cmd: list[str], *, mem: str, timeout: str, after=(), vram: float | None = None,
              cpus: int | None = None) -> list[str]:
    """A `job start` argv in launch.sh's flag order; vram: a GPU job (shared lease, stall and offline flags)."""
    argv = ["job", "start", name, "--cwd", str(c2._here())]
    if after:
        argv += ["--after", ",".join(after)]
    if vram is not None:
        argv += ["--needs", "gpu:shared", "--vram", lease(vram)]
    argv += ["--mem", mem]
    if cpus is not None:
        argv += ["--cpus", str(cpus)]
    if vram is not None:
        argv += list(GPU_FLAGS)
    return argv + ["--timeout", timeout, "--", *cmd]


def labeling_cmd(*args) -> list[str]:
    return ["./python", "-m", "labeling", *map(str, args)]


def _plain(obj):
    return json.loads(json.dumps(obj))


def _result(exit_code: int, status: str, reasons: list[str], inputs: dict, decisions: dict | None = None, **extra) -> dict:
    return {"exit": exit_code, "status": status, "reasons": reasons, "inputs": _plain(inputs),
            "decisions": None if decisions is None else _plain(decisions), **extra}


# --- step `checks` -------------------------------------------------------------------------------------

def plan_checks(probes: dict[tuple[str, int], dict | None]) -> dict:
    """Decisions of step `checks` from the six probe results ((engine, setting) -> probe.json or None)."""
    inputs, reasons = {}, []
    for e, i in PROBES:
        key, p = f"{e}-{i}", probes.get((e, i))
        if p is None:
            reasons.append(f"probe {key} is missing ({probe_path(e, i).name} not written)")
            continue
        sig = (p.get("split") or {}).get("signals") or {}
        inputs[key] = {"audio_hours_per_gpu_hour": p.get("audio_hours_per_gpu_hour"), "peak_vram_gb": p.get("peak_vram_gb"),
                       "declared_vram_gb_for_c2": p.get("declared_vram_gb_for_c2"), "host_peak_gb": host_peak_gb(p),
                       "repeat_changed": sig.get("repeat_changed"), "repeat_changed_share": sig.get("repeat_changed_share")}
        rate = inputs[key]["audio_hours_per_gpu_hour"]
        if not isinstance(rate, (int, float)) or rate <= 0:
            reasons.append(f"probe {key} has no positive audio_hours_per_gpu_hour ({rate})")
        if not inputs[key]["declared_vram_gb_for_c2"]:
            reasons.append(f"probe {key} has no declared_vram_gb_for_c2")
        if inputs[key]["host_peak_gb"] is None:
            reasons.append(f"probe {key} recorded no host memory peak")
        if e == "vllm" and inputs[key]["repeat_changed"] is None:
            reasons.append(f"probe {key} has no split.signals.repeat_changed")
    if reasons:
        return _result(INPUT, "missing", reasons, inputs)
    repeat = [f"probe vllm-{i}: {inputs[f'vllm-{i}']['repeat_changed']} of the split's clips differ between two passes "
              f"at the production setting" for i in range(3) if inputs[f"vllm-{i}"]["repeat_changed"] > 0]
    if repeat:
        return _result(STOP, "stopped", repeat + ['a non-zero repeat share is reported to the architect before C2 '
                                                  '(labeling plan "GPU probe"); nothing submitted'], inputs)
    best = {e: max(range(3), key=lambda i, e=e: (probes[(e, i)]["audio_hours_per_gpu_hour"], -i)) for e in ENGINES}
    vram = {e: tenth_up(probes[(e, best[e])]["declared_vram_gb_for_c2"]) for e in ENGINES}
    v, t = best["vllm"], best["transformers"]
    total = (probes[("vllm", v)].get("engine") or {}).get("total_vram_gb")
    util = checks.VLLM_SETTINGS[v]["gpu_memory_utilization"]
    if total and util * total > vram["vllm"] + 1e-6:
        return _result(INPUT, "invalid", [f"vLLM setting {v}'s memory fraction ({util} x {total:.2f} GB) exceeds its "
                                          f"declared VRAM {vram['vllm']} GB (c2.init_run would refuse)"], inputs)
    peak = max(x["host_peak_gb"] for x in inputs.values())
    mem_gb = job_mem_gb(peak)
    mem = f"{mem_gb}G"
    prep, eq_prep = "relabel-consistency-prepare", "relabel-equivalence-prepare"
    cons_v, cons_t = f"relabel-consistency-vllm-{v}", f"relabel-consistency-transformers-{t}"
    jobs = [
        (prep, job_start(prep, labeling_cmd("consistency-prepare"), mem="16G", timeout="1h")),
        (cons_v, job_start(cons_v, labeling_cmd("consistency", "--engine", "vllm", "--setting", v,
                                                "--declared-vram-gb", gb_arg(vram["vllm"])),
                           after=[prep], vram=vram["vllm"], mem=mem, timeout="2h")),
        (cons_t, job_start(cons_t, labeling_cmd("consistency", "--engine", "transformers", "--setting", t,
                                                "--declared-vram-gb", gb_arg(vram["transformers"])),
                           after=[cons_v], vram=vram["transformers"], mem=mem, timeout="2h")),
        (eq_prep, job_start(eq_prep, labeling_cmd("equivalence-prepare"), after=[prep], mem="16G", timeout="2h")),
        ("relabel-equivalence", job_start("relabel-equivalence", labeling_cmd(
            "equivalence", "--vllm-setting", v, "--tf-setting", t, "--declared-vram-gb", gb_arg(vram["vllm"])),
            after=[cons_t, eq_prep], vram=vram["vllm"], mem=mem, timeout="3h")),
        ("relabel-advance-c2", job_start("relabel-advance-c2", ["./python", "-m", "labeling.advance", "c2"],
                                         after=["relabel-equivalence"], mem="4G", timeout="30m")),
    ]
    decisions = {"vllm_setting": v, "transformers_setting": t, "declared_vram_gb": vram,
                 "lease": {e: lease(vram[e]) for e in ENGINES}, "mem_gb": mem_gb,
                 "mem_basis": {"host_peak_gb": peak, "headroom": MEM_HEADROOM, "floor_gb": MEM_FLOOR_GB,
                               "cap_gb": MEM_CAP_GB, "capped": math.ceil(peak * MEM_HEADROOM) > MEM_CAP_GB,
                               "note": "largest teacher/b0 monitor peak of the six probes (tree RSS or cgroup memory, "
                                       "the larger; cgroup memory counts page cache)"},
                 "jobs": [{"name": n, "argv": a} for n, a in jobs]}
    return _result(OK, "planned", [], inputs, decisions)


# --- step `c2` -----------------------------------------------------------------------------------------

def load_selection() -> dict:
    """The inputs of `./python -m labeling select`, loaded as it loads them, and select_production over them."""
    from labeling import config
    probes = [config.load_json(p) for p in sorted((layout.probes_dir() / "gpu").glob("*/probe.json"))]
    cons = [config.load_json(p) for p in sorted(layout.consistency_dir("dev").glob("result-*.json"))]
    eq_path = layout.equivalence_dir("main") / "result.json"
    eq = config.load_json(eq_path) if eq_path.exists() else None
    return {"probes": probes, "consistency": cons, "equivalence": eq,
            "result": checks.select_production(probes, cons, eq)}


def _strip(contract: dict) -> dict:
    return {k: v for k, v in contract.items() if k != "declared_vram_gb"}


def setting_of(contract: dict) -> tuple[str, int]:
    """(engine, setting index) of a contract's engine spec; raises unless exactly one production setting matches."""
    specs = {("vllm", i): checks.vllm_engine(s) for i, s in enumerate(checks.VLLM_SETTINGS)} | \
            {("transformers", i): checks.tf_engine(s) for i, s in enumerate(checks.TRANSFORMERS_SETTINGS)}
    hits = [k for k, s in specs.items() if {a: b for a, b in s.items() if a != "total_vram_gb"} == contract["engine"]]
    if len(hits) != 1:
        raise ValueError(f"the selected engine spec matches {len(hits)} production settings: {contract['engine']}")
    return hits[0]


def plan_c2(selection: dict, checks_record: dict | None, c1_status: dict | None, c1_manifest: dict | None,
            consistency: dict[tuple[str, int], dict | None], run: dict | None) -> dict:
    """Decisions of step `c2`. selection: load_selection(); consistency: (engine, setting) -> result or None; run: the
    existing labels/<T*>/main/run.json or None."""
    inputs = {"table": selection["result"]["table"], "selected": selection["result"]["selected"],
              "c1_distinct_hours": (c1_manifest or {}).get("distinct_hours")}
    cd = (checks_record or {}).get("decisions")
    if not cd or (checks_record or {}).get("status") != "submitted":
        return _result(INPUT, "missing", ["step checks has not submitted its jobs (no memory cap recorded)"], inputs)
    inputs["checks"] = {k: cd[k] for k in ("vllm_setting", "transformers_setting", "declared_vram_gb", "mem_gb")}
    sel = selection["result"]["selected"]
    if sel is None:
        return _result(STOP, "stopped", ["no probed setting qualifies (consistency and, for vLLM, equivalence); "
                                         "nothing submitted"], inputs)
    contract = sel["contract"]
    try:
        engine, i = setting_of(contract)
    except ValueError as exc:
        return _result(INPUT, "invalid", [str(exc)], inputs)
    matched = [p for p in selection["probes"] if _strip(p["contract"]) == _strip(contract)]
    if len(matched) != 1:
        return _result(INPUT, "invalid", [f"{len(matched)} probes match the selected contract"], inputs)
    reasons = []
    if not (c1_status or {}).get("complete") or not (c1_manifest or {}).get("distinct_hours"):
        reasons.append("C1 is not complete (no manifest with distinct_hours)")
    cons = consistency.get((engine, i))
    if cons is None or cons.get("contract") != contract:
        reasons.append(f"{consistency_result_path(engine, i).name} is missing or under another contract (the gate "
                       f"reads it)")
    if reasons:
        return _result(INPUT, "missing", reasons, inputs)
    if run is not None and c2.contract(run) != contract:
        return _result(REFUSED, "refused", [f"labels/{layout.TEACHER}/{RUN}/run.json exists under another contract"],
                       inputs | {"run_contract": c2.contract(run)})
    vram, group_shards = contract["declared_vram_gb"], contract["group_shards"]
    cost = checks.c2_cost(matched[0], c1_manifest["distinct_hours"])
    chunk_h = layout.CHUNK_MINUTES / 60
    needed = math.ceil(round(cost["c2_gpu_hours"] / chunk_h, 9))
    n = needed + SPARE_CHUNKS
    mem = f"{cd['mem_gb']}G"
    run_dir = layout.labels_dir(layout.TEACHER, RUN)
    jobs, prev = [], None
    for k in range(1, n + 1):
        name = f"relabel-c2-{RUN}-{k}"
        jobs.append((name, job_start(name, ["./python", "-m", "labeling.c2", "chunk", "--run-dir", str(run_dir),
                                            "--budget-minutes", f"{layout.CHUNK_MINUTES:g}"],
                                     after=[prev] if prev else (), vram=vram, mem=mem, timeout="2h30m")))
        prev = name
    gate = labeling_cmd("gate", "--run", RUN, "--consistency", consistency_result_path(engine, i))
    if engine == "vllm":
        gate += ["--equivalence", str(equivalence_result_path())]
    jobs.append((f"relabel-gate-{RUN}", job_start(f"relabel-gate-{RUN}", gate, after=[prev], mem="36G", cpus=12,
                                                  timeout="6h")))
    init_args = ["c2-init", "--run", RUN, "--engine", engine, "--setting", str(i), "--declared-vram-gb", gb_arg(vram),
                 "--group-shards", str(group_shards)]
    decisions = {"engine": engine, "setting": i, "group_shards": group_shards, "declared_vram_gb": vram,
                 "contract": contract, "mem_gb": cd["mem_gb"], "chunks": n, "init_args": init_args,
                 "projection": cost | {"chunk_hours": chunk_h, "chunks_needed": needed, "spare_chunks": SPARE_CHUNKS,
                                       "chunks": n, "lease": lease(vram)},
                 "jobs": [{"name": nm, "argv": a} for nm, a in jobs]}
    return _result(OK, "planned", [], inputs, decisions, init=None if run is not None else init_args)


# --- submission ----------------------------------------------------------------------------------------

def job_submit(argv: list[str]) -> None:
    subprocess.run(argv, check=True)


def job_exists(name: str) -> bool:
    return subprocess.run(["job", "status", name], capture_output=True).returncode == 0


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def execute(step: str, plan: dict, *, submit, exists, dry_run: bool, out, prepare=None) -> int:
    """Record and submit a plan (see the module docstring); prepare(args): run once before the first submission when
    plan["init"] is set (c2-init)."""
    path = record_path(step)
    old = _load(path)
    done: list[str] = []
    if old and old.get("in_flight"):
        out(json.dumps({"step": step, "exit": REFUSED, "reason": f"an earlier run was interrupted while submitting "
                        f"{old['in_flight']}; check `job status {old['in_flight']}`, then fix {path.name} by hand "
                        f"(add the name to submitted if the job exists, and clear in_flight)"}, indent=1))
        return REFUSED
    if old and old.get("submitted"):
        gone = [n for n in old["submitted"] if not exists(n)] if old.get("decisions") == plan["decisions"] else []
        if old.get("decisions") != plan["decisions"] or gone:
            why = (f"recorded jobs no longer exist: {gone}" if gone else
                   f"{path.name} records submitted jobs under other decisions; resolve by hand")
            out(json.dumps({"step": step, "exit": REFUSED, "reason": why, "now": {k: plan[k] for k in
                            ("exit", "status", "reasons", "decisions")}, "recorded": old.get("decisions")}, indent=1))
            return REFUSED
        done = list(old["submitted"])
    jobs = (plan["decisions"] or {}).get("jobs", [])
    pending = [j for j in jobs if j["name"] not in done]
    report = {"step": step, "exit": plan["exit"], "status": plan["status"], "reasons": plan["reasons"],
              "decisions": plan["decisions"], "already_submitted": done}
    if dry_run:
        out(json.dumps(report | {"dry_run": True, "init": plan.get("init"), "submit": [j["argv"] for j in pending]},
                       indent=1))
        return plan["exit"]
    rec = {"step": step, "status": plan["status"], "exit": plan["exit"], "reasons": plan["reasons"],
           "inputs": plan["inputs"], "decisions": plan["decisions"], "submitted": done, "updated": _now()}
    if plan["exit"] != OK:
        c2.write_json(path, rec)
        out(json.dumps(report, indent=1))
        return plan["exit"]
    if not pending:
        if old and old.get("status") != "submitted":   # every job recorded, the final write interrupted
            c2.write_json(path, old | {"status": "submitted", "exit": OK, "updated": _now()})
        out(json.dumps(report | {"note": "every job is already submitted; nothing to do"}, indent=1))
        return OK
    rec |= {"status": "submitting", "exit": None, "in_flight": None}
    c2.write_json(path, rec)
    try:
        if plan.get("init") is not None and prepare is not None:
            prepare(plan["init"])
        for j in pending:
            # in_flight survives only a hard interruption inside `job start`; a re-run then refuses (above)
            rec |= {"in_flight": j["name"], "updated": _now()}
            c2.write_json(path, rec)
            submit(j["argv"])
            done.append(j["name"])
            rec |= {"submitted": done, "in_flight": None, "updated": _now()}
            c2.write_json(path, rec)
    except Exception as exc:   # `job start` failed (nothing started): recorded; a re-run resumes
        rec |= {"status": "failed", "exit": FAILED, "error": f"{type(exc).__name__}: {exc}", "in_flight": None,
                "updated": _now()}
        c2.write_json(path, rec)
        out(json.dumps(report | {"exit": FAILED, "error": rec["error"], "submitted": done}, indent=1))
        return FAILED
    rec |= {"status": "submitted", "exit": OK, "updated": _now()}
    c2.write_json(path, rec)
    out(json.dumps(report | {"status": "submitted", "submitted": done}, indent=1))
    return OK


def run_checks(*, submit, exists, dry_run: bool, out) -> int:
    probes = {(e, i): _load(probe_path(e, i)) for e, i in PROBES}
    return execute("checks", plan_checks(probes), submit=submit, exists=exists, dry_run=dry_run, out=out)


def run_c2(*, submit, exists, dry_run: bool, out, init=None, c1_dir=None) -> int:
    if init is None or c1_dir is None:
        from labeling import __main__ as cli
        init = init or cli.main
        c1_dir = c1_dir or cli.c1_dir
    status = manifest = None
    try:
        pool = c1_dir()
        status, manifest = _load(pool / "status.json"), _load(pool / "manifest.json")
    except (SystemExit, RuntimeError, OSError) as exc:   # no pinned configuration
        out(f"C1 directory: {exc}")
    selection = load_selection()
    consistency = {(e, i): _load(consistency_result_path(e, i)) for e, i in PROBES}
    run_dir = layout.labels_dir(layout.TEACHER, RUN)
    run = _load(run_dir / "run.json")
    plan = plan_c2(selection, _load(record_path("checks")), status, manifest, consistency, run)

    def prepare(args: list[str]) -> None:
        init(args)
        made = c2.load_run(run_dir)
        if c2.contract(made) != plan["decisions"]["contract"]:
            raise RuntimeError(f"c2-init wrote {run_dir / 'run.json'} under another contract than the selected one")

    return execute("c2", plan, submit=submit, exists=exists, dry_run=dry_run, out=out, prepare=prepare)


def main(argv: list[str] | None = None, *, submit=None, exists=None, init=None, c1_dir=None, out=print) -> int:
    parser = argparse.ArgumentParser(prog="labeling.advance", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("checks", "c2"))
    parser.add_argument("--dry-run", action="store_true", help="print the decisions and argv lists; write nothing")
    args = parser.parse_args(argv)
    layout.require_mount()
    kw = {"submit": submit or job_submit, "exists": exists or job_exists, "dry_run": args.dry_run, "out": out}
    if args.step == "checks":
        return run_checks(**kw)
    return run_c2(**kw, init=init, c1_dir=c1_dir)


if __name__ == "__main__":
    sys.exit(main())
