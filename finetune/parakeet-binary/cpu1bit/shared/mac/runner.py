"""cpu1bit timed-block runner on the Mac (plans/cpu-benchmark-mac.md "Measurement", "Assumptions A1-A3 and
validation", "Design and selection"; Codex code reviews c1-c3, plan reviews p4-p10).

Timed stages run ONLY as (user authorization for overnight timed runs, recorded in every record):
  job start cpu1bit-<stage> --host mac --quiet-host --full-machine --priority normal -- \
      /usr/bin/python3 -I <ws>/shared/mac/runner.py SESSION.json STAGE
Stages:
- validate: the A1-A3 tests and the instrument checks (a 10-minute capture with an idle window and 20
  one-thread spins: band, rate and J, idle reference, energy-interval test, spin power; 20 pre-spawn spin
  trials: A2 and delta_A2; an R-MB1 latency block, then one R-MB1 instrument block). Never compared.
- validate-plumbing / plumbing: the same code at background priority, shortened; never a measurement.
- sizing stages ({"type": "sizing", "arms", "condition"}): one block per arm with one replay per clip;
  N = ceil(12 s / (0.95 x fastest latency)) per clip, the A_MAX preflight, and a write-once sizing record
  that every later stage of that condition reads (frozen N, its SHA-256 in the stage fingerprint).
- dry-run: no timing; designs, expected hours, file checks.
- command stages ({"type": "command", "argv": [...]}, e.g. P's `p-ubench isa`): one non-comparison run
  of a command under the full monitoring, recorded separately like validate.
- design stages ({"arms", "replicates", "condition"}): Williams blocks with the frozen N.

Records (c1 finding 7): <ROOT>/results/<session>/<stage>/ holds stage.json (the immutable fingerprint of
the stage, its configuration, every arm's binary / export / captures hash, the harness sources, the
validation record, the frozen N and the frozen constants) and design.json, both written once; a resumed
stage must reproduce both exactly or it refuses. Every sequence attempt gets a new number n (never
reused, even if interrupted): seq-r<r>-s<s>.att<n>.start.json, one <block_id>.att<n>.json per block (plus
its raw powermetrics stream and stderr) and seq-r<r>-s<s>.att<n>.end.json; every file is created once.

Per block: wait until quiet (HIDIdleTime >= 600 s, AC + battery 100%, thermal nominal, Low Power Mode off,
load < 1.5, memory pressure normal), then under the EXCLUSIVE Mac lock and caffeinate -i -s: s =
CLOCK_UPTIME_RAW just before spawning powermetrics (permitted command only), the 1 s state sampler, a 10 s
idle window (idle reference; P-clusters busy <= P_IDLE_LIMIT), the arm's `block` process (load, warmup,
storage audit, then per clip: untimed replay, S + G, N replays, G + S', window events), then reading
until every window is bracketed, and stopping powermetrics by closing its pipe.
Validity: stream framing (A1), the per-block band (A1) and rate (A3) checks, the host state over the whole
span from spawn to the end of the capture, the storage audit, QoS, the frozen N and the dispatch audit;
per window: guard.window_bounds (offset interval [-delta_A2, U], coverage, idle edges, work present),
no event read inside [t0 - G, t1 + G], and the state over the padded interval extended by U + L.
An invalid window invalidates its block attempt and the whole sequence attempt is re-measured: up to
MAX_ATTEMPTS per pass and PASSES passes, then the stage exits 2, explicitly INCOMPLETE, and analysis.py
refuses it until a later resume completes it.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import design  # noqa: E402
import guard  # noqa: E402
import macstate  # noqa: E402
import pm  # noqa: E402

# The Mac artifacts root: $CPU1BIT_ROOT, default $HOME/wilderness-labs-stt-artifacts/cpu1bit. Session files
# may write paths as "$CPU1BIT_ROOT/..." (or "~/..."); main() expands them (expand_session).
ROOT = os.environ.get("CPU1BIT_ROOT") or os.path.join(os.path.expanduser("~"), "wilderness-labs-stt-artifacts", "cpu1bit")
LOCK = ROOT + "/lock/active.lock"
AUTHORIZATION = ("User-authorized overnight timed runs on the Mac as `job start --host mac --quiet-host --full-machine "
                 "--priority normal` (plans/cpu-benchmark-mac.md user decisions; agent H task brief 2026-10-21); "
                 "stage started by the coordinator.")
IDLE_S = 10.0
P_IDLE_LIMIT = 0.10          # max P-cluster busy share in the pre-block idle window (frozen before timing)
E_FLAG_PP = 0.10             # E-cluster busy share > idle mean + 10 points flags a window (descriptive)
MIN_IDLE_SAMPLES = 20        # samples wholly inside the idle window for every admissible offset
WINDOW_TARGET_S = 12.0       # sizing target; shorter windows are flagged, never invalid
SIZING_MARGIN = 0.95
PREFLIGHT_SLACK_S = 60.0     # every planned block <= A_MAX - 60 s
BRACKET_TIMEOUT_S = 15.0
CALLS_PER_REPLAY, QUANTIZATIONS_PER_REPLAY = 240, 192
MAX_ATTEMPTS = 4             # per sequence per pass
PASSES = 2
QUIET_POLL_S = 30
QUIET_MAX_WAIT_S = 4 * 3600
# validate
VALIDATE_CAPTURE_S = 610.0   # the 10-minute capture (A1 band, A3 rate, J)
VALIDATE_SPINS = 20          # energy-interval test and spin power
SPIN_S, SPIN_GAP_S = 1.5, 3.0     # spins longer than U_MAX + 2 L hold whole samples; gaps leave the look-back idle
PRESPAWN_TRIALS = 20         # A2
PRESPAWN_SPIN_S = 0.6        # >= 500 ms, ending immediately before the spawn
PRESPAWN_READ_S = 2.0
MAX_RELATIVE_WIDTH = 0.01    # validate: median (E_upper - E_lower) / E_est
HARNESS_FILES = ("runner.py", "pm.py", "guard.py", "macstate.py", "design.py")
CAFFEINATE = ["caffeinate", "-i", "-s"]
STATE_PROBES: dict = {}      # keyword arguments for macstate.Sampler (tests inject fake probes)
STATE_WAIT_S = 5.0           # for a completed state sample before the spawn and after the capture


def frozen_constants() -> dict:
    return {"IDLE_S": IDLE_S, "P_IDLE_LIMIT": P_IDLE_LIMIT, "E_FLAG_PP": E_FLAG_PP, "MIN_IDLE_SAMPLES": MIN_IDLE_SAMPLES,
            "WINDOW_TARGET_S": WINDOW_TARGET_S, "SIZING_MARGIN": SIZING_MARGIN, "PREFLIGHT_SLACK_S": PREFLIGHT_SLACK_S,
            "MAX_ATTEMPTS": MAX_ATTEMPTS, "PASSES": PASSES,
            "guard": {k: getattr(guard, k) for k in ("G_NS", "S_NS", "S_POST_NS", "K", "L_MAX_NS", "A_MAX_NS", "U_MAX_NS",
                                                     "RHO", "TAU_BAND_NS", "Q_MJ")},
            "macstate.FOREIGN_LIMIT_CORES": macstate.FOREIGN_LIMIT_CORES,
            "macstate.MAX_STATE_GAP_NS": macstate.MAX_STATE_GAP_NS, "macstate.HID_QUIET_S": macstate.HID_QUIET_S}


def now_ns() -> int:
    return pm.now_ns()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(doc) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()


def write_new(path: str, doc) -> None:
    """Creates `path` once and atomically (c1 finding 7, c2 finding 3): the JSON is written to a
    `.tmp-` file in the same directory and hard-linked into place, which fails if `path` exists, so
    evidence is never overwritten and a reader (pull.py) never sees a partial record."""
    d, base = os.path.split(path)
    tmp = os.path.join(d, f".tmp-{base}.{os.getpid()}.{time.time_ns()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=1, default=lambda o: o.__dict__)
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)
    finally:
        os.unlink(tmp)


def read_json(path: str):
    with open(path) as f:
        return json.load(f)


class ExclusiveLock:
    """LOCK_EX on the Mac lock for timed blocks; plumbing runs (not measurements) take LOCK_SH."""

    def __init__(self, shared: bool = False):
        self.mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX

    def __enter__(self):
        os.makedirs(os.path.dirname(LOCK), exist_ok=True)
        self.f = open(LOCK, "a")
        t0 = time.monotonic()
        fcntl.flock(self.f, self.mode)
        self.waited_s = time.monotonic() - t0
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def wait_quiet(log) -> dict:
    t0 = time.monotonic()
    while True:
        q = macstate.quiet_now()
        if q["ok"]:
            q["waited_s"] = time.monotonic() - t0
            return q
        if time.monotonic() - t0 > QUIET_MAX_WAIT_S:
            raise SystemExit(f"host not quiet for {QUIET_MAX_WAIT_S} s: {q}")
        log({"event": "waiting_for_quiet", "checks": q["checks"], "errors": q["errors"], "hid_idle_s": q.get("hid_idle_s")})
        time.sleep(QUIET_POLL_S)


# --- one-thread spins (validate: energy-interval test, spin power, pre-spawn trials) -----------------

def spin_code() -> str:
    """Child: lead idle, then per spin an idle gap and a one-thread spin at QOS_CLASS_USER_INTERACTIVE,
    printing each spin's [on, off] in CLOCK_UPTIME_RAW ns; exits at once after the last spin (os._exit),
    so a pre-spawn trial's spin ends immediately before the runner spawns powermetrics."""
    return r'''
import ctypes, json, os, sys, time
lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
lib.pthread_set_qos_class_self_np(0x21, 0)
clk = lambda: time.clock_gettime_ns(time.CLOCK_UPTIME_RAW)
n, spin_s, gap_s, lead_s = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
time.sleep(lead_s)
for i in range(n):
    if gap_s:
        time.sleep(gap_s)
    on = clk(); end = on + int(spin_s * 1e9)
    x = 0
    while clk() < end:
        x += 1
    print(json.dumps({"on_ns": on, "off_ns": clk()}), flush=True)
os._exit(0)
'''


def run_spins(n: int, spin_s: float, gap_s: float, lead_s: float, active: list, timeout_s: float = 300) -> list[dict]:
    p = subprocess.Popen([sys.executable, "-I", "-c", spin_code(), str(n), str(spin_s), str(gap_s), str(lead_s)],
                         stdout=subprocess.PIPE, text=True, stdin=subprocess.DEVNULL)
    active.append(p.pid)
    try:
        out, _ = p.communicate(timeout=timeout_s)
    finally:
        active.remove(p.pid)
    if p.returncode != 0:
        raise RuntimeError(f"spin child exit {p.returncode}")
    return [json.loads(line) for line in out.splitlines() if line.strip()]


# --- analysis of one block (pure) --------------------------------------------------------------------

def cluster_groups(samples) -> tuple[list[str], list[str]]:
    names = sorted({n for s in samples if s.ok for n in s.clusters})
    return [n for n in names if n.startswith("E")], [n for n in names if n.startswith("P")]


def command_checks(cfg: dict, events: list[dict], samples: list, phases: dict, idle_busy: dict,
                   e_names: list[str], p_names: list[str]) -> tuple[list[str], dict]:
    """The QoS gate of a command stage (Codex code review c2 finding 4). The command must print its
    thread QoS readback (cfg["qos_readback"]: regex with groups (class, relative priority), the
    expected class and relative priority, the minimum number of matching lines; every match must
    agree); the child must not be darwin-background (getpriority(PRIO_DARWIN_PROCESS, pid) sampled
    after spawn and at every output line); placement evidence (descriptive per M4, no per-thread proof):
    over the command's run some P-cluster's busy share rises by >= cfg["placement"]["p_cluster_min_rise"]
    above the idle window and no E-cluster rises by more than E_FLAG_PP."""
    reasons = []
    q = cfg.get("qos_readback")
    if not q:
        return ["command stage has no qos_readback specification"], {}
    rx = re.compile(q["regex"])
    found = [m.groups() for e in events for m in [rx.search(e.get("line", ""))] if m]
    bad = [g for g in found if g[0].lower() != q["class"].lower() or int(g[1]) != int(q["rel"])]
    if len(found) < q.get("min_matches", 1):
        reasons.append(f"QoS readback: {len(found)} readback line(s), need {q.get('min_matches', 1)}")
    if bad:
        reasons.append(f"QoS readback mismatch: {bad[:3]} (expected class {q['class']} rel {q['rel']})")
    child_bg = phases.get("child_bg", [])
    observed = [v for v in child_bg if v in (0, 1)]   # -1: the child had already exited
    if not observed or any(v != 0 for v in observed):
        reasons.append(f"child darwin-background state {child_bg[:5]} (must be observed and 0)")
    t0, t1 = phases.get("proc_start"), phases.get("proc_end")
    busy = pm.cluster_busy(samples, t0, t1) if t0 and t1 else {}
    rise = {n: (busy.get(n) or 0.0) - (idle_busy.get(n) or 0.0) for n in busy}
    pl = cfg.get("placement", {})
    p_rise = max((rise.get(n, 0.0) for n in p_names), default=0.0)
    e_rise = max((rise.get(n, 0.0) for n in e_names), default=0.0)
    if p_rise < pl.get("p_cluster_min_rise", 0.10):
        reasons.append(f"placement evidence: P-cluster busy rise {p_rise:.2f} < {pl.get('p_cluster_min_rise', 0.10):.2f}")
    if e_rise > E_FLAG_PP:
        reasons.append(f"placement evidence: E-cluster busy rise {e_rise:.2f} > {E_FLAG_PP:.2f}")
    return reasons, {"qos_readback_lines": found, "child_darwin_process_bg": child_bg, "cluster_busy": busy,
                     "cluster_busy_rise": rise}


def leading_run(samples) -> list:
    """The mapped records: the leading run of parsed records in one timeline segment (c_k is unknown after
    a malformed or lost record), as guard's mapping uses them. Call pm.build_timeline first."""
    out = []
    for x in samples:
        if not x.ok or (out and getattr(x, "segment", None) != getattr(out[0], "segment", None)):
            break
        out.append(x)
    return out


def u_bound_ns(samples, spawn_ns: int) -> int | None:
    """U = min_k (d_k - s - c_k (1 - rho)) over the block's mapped records (plan "Mapping")."""
    c, u = 0, None
    for x in leading_run(samples):
        c += x.elapsed_ns
        v = x.arrival_ns - spawn_ns - c * (1 - guard.RHO)
        u = v if u is None else min(u, v)
    return None if u is None else int(u)


def bracketed(samples, spawn_ns: int, t_ns: int, delta_a2_ns: int = 0) -> bool:
    """A record has arrived whose earliest admissible start exceeds t_ns (plan A1 "Bracketing")."""
    c = 0
    for x in leading_run(samples):
        if spawn_ns - delta_a2_ns + c * (1 - guard.RHO) > t_ns:
            return True
        c += x.elapsed_ns
    return False


def analyse_block(sess: dict, events: list[dict], samples: list, state: list, phases: dict, exit_code: int,
                  mode: str = "block", command_cfg: dict | None = None, replays: list[int] | None = None,
                  delta_a2_ns: int = 0) -> dict:
    """Validity and measurements of one block from its raw evidence. `phases`: CLOCK_UPTIME_RAW ns of
    spawn_ns (s), idle0, idle1, proc_start, proc_end and span1 (end of the capture), plus `stream`
    (pm.check_stream's verdict) and, for commands, child_bg. `replays`: the frozen N per window in clip
    order (block mode). mode "command" skips the replay-specific checks."""
    pm.build_timeline(samples)
    e_names, p_names = cluster_groups(samples)
    spawn = phases["spawn_ns"]
    span1 = phases["span1"]
    reasons = []
    if exit_code != 0:
        reasons.append(f"process exit {exit_code}")
    stream = phases.get("stream") or {"ok": False, "reasons": ["no stream verdict"]}
    if not stream.get("ok"):
        reasons += [f"powermetrics stream (A1): {r}" for r in (stream.get("reasons") or ["not ok"])]
    band = guard.band_check(samples)
    if not band["valid"]:
        reasons += [f"powermetrics band (A1): {r}" for r in (band.get("reasons") or ["band too wide"])]
    rate = guard.rate_check(samples)
    if not rate["valid"]:
        reasons += [f"powermetrics rate (A3): {r}" for r in (rate.get("reasons") or ["rate outside rho"])]
    span_state = macstate.window_state(state, spawn, span1)
    if not span_state["valid"]:
        reasons += [f"block span: {r}" for r in span_state["reasons"]]
    t_i0, t_i1 = phases["idle0"], phases["idle1"]
    idle_samples = guard.samples_within(samples, spawn, t_i0, t_i1, delta_a2_ns)
    ref = guard.idle_reference(idle_samples) if len(idle_samples) >= MIN_IDLE_SAMPLES else None
    if ref is None:
        reasons.append(f"idle window: {len(idle_samples)} samples wholly inside, need {MIN_IDLE_SAMPLES}")
    idle_busy = pm.cluster_busy(samples, t_i0, t_i1)
    p_idle_busy = max((idle_busy.get(n) or 0.0) for n in p_names) if p_names else None
    if p_idle_busy is None or p_idle_busy > P_IDLE_LIMIT:
        reasons.append(f"P-clusters not idle before the block (busy {p_idle_busy})")
    windows, storage = [], None
    if mode == "block":
        qos = [e for e in events if e.get("event") in ("qos", "qos_end")]
        if len(qos) != 2 or not all(e.get("ok") for e in qos):
            reasons.append("QoS readback mismatch or missing")
        st_ev = [e for e in events if e.get("event") == "storage"]
        storage = st_ev[-1] if st_ev else None
        if storage is None:
            reasons.append("storage audit: missing")
        elif not storage.get("ok"):
            reasons.append(f"storage audit failed: {storage.get('reasons')}")
        if replays is None:
            reasons.append("no frozen replay count for this block")
        g, l_max = guard.G_NS, guard.L_MAX_NS
        u_block = u_bound_ns(samples, spawn)
        starts = {e["window"]: e for e in events if e.get("event") == "window_start"}
        for end in (e for e in events if e.get("event") == "window_end"):
            w = end["window"]
            st = starts.get(w)
            w_reasons = list(reasons)
            if st is None:
                w_reasons.append("window_start event missing")
                st = {"t_ns": end["t_ns"], "clip": None, "id": None, "T": None}
            t0, t1, n = st["t_ns"], end["t_ns"], end["replays"]
            if replays is not None and (w >= len(replays) or n != replays[w]):
                w_reasons.append(f"replay count {n} != frozen N {replays[w] if w < len(replays) else None}")
            if n < 1:
                w_reasons.append("no replays")
            n_div = max(n, 1)
            if end.get("gemm_calls") != CALLS_PER_REPLAY * n or end.get("quantizations") != QUANTIZATIONS_PER_REPLAY * n:
                w_reasons.append(f"dispatch audit: calls != {CALLS_PER_REPLAY} / quantizations != "
                                 f"{QUANTIZATIONS_PER_REPLAY} per replay")
            early = [e.get("event") for e in events
                     if e.get("received_ns") is not None and t0 - g <= e["received_ns"] <= t1 + g]
            if early:
                w_reasons.append(f"event read inside the padded interval [t0 - G, t1 + G]: {early[:3]}")
            flags = []
            if ref is not None:
                wb = guard.window_bounds(samples, spawn, t0, t1, ref, delivery_jitter_ns=rate.get("jitter_ns"),
                                         delta_a2_ns=delta_a2_ns)
                if not wb["valid"]:
                    w_reasons += [f"energy: {r}" for r in wb["reasons"]]
                flags += list(wb.get("flags") or [])
                parts = (wb.get("e_lower_j"), wb.get("e_est_j"), wb.get("e_upper_j"))
                if any(v is None for v in parts):          # an invalid window may carry no bounds
                    energy = energy_window = None
                else:
                    energy = {"lower": parts[0] / n_div, "est": parts[1] / n_div, "upper": parts[2] / n_div}
                    energy_window = {"lower": parts[0], "est": parts[1], "upper": parts[2]}
                guard_j = wb.get("guard_j")
                u_ns, x_count = wb.get("u_ns"), wb.get("x_count")
                bounds = {k: v for k, v in wb.items() if k not in ("reasons", "flags")}
            else:
                energy = energy_window = guard_j = None
                u_ns, x_count, bounds = u_block, None, None
            dur_s = (t1 - t0) / 1e9
            if dur_s < WINDOW_TARGET_S:
                flags.append(f"short window: {dur_s:.2f} s < {WINDOW_TARGET_S:.0f} s (flagged, not invalid)")
            ext = (u_ns if u_ns is not None and u_ns > 0 else 0) + l_max
            ws = macstate.window_state(state, t0 - g - ext, t1 + g + ext)
            w_reasons += ws["reasons"]
            busy = pm.cluster_busy(samples, t0, t1)
            windows.append({
                "window": w, "clip": st["clip"], "id": st["id"], "T": st["T"], "t0_ns": t0, "t1_ns": t1,
                "replays": n, "duration_s": dur_s, "latency_s_per_replay": dur_s / n_div, "replay_ns": end.get("replay_ns"),
                "energy_j_per_replay": energy, "energy_j_window": energy_window,
                "guard_j_per_replay": None if guard_j is None else guard_j / n_div,
                "u_ns": u_ns, "x_count": x_count, "bounds": bounds, "flags": flags,
                "cluster_busy": busy,
                "e_cluster_flag": any((busy.get(nm) or 0.0) > (idle_busy.get(nm) or 0.0) + E_FLAG_PP for nm in e_names),
                "both_p_clusters_active": all((busy.get(nm) or 0.0) > 0.5 for nm in p_names),
                "foreign_busy_cores_max": ws["foreign_busy_cores_max"], "foreign_busy_cores_mean": ws["foreign_busy_cores_mean"],
                "valid": not w_reasons, "invalid_reasons": list(dict.fromkeys(w_reasons)),
            })
        expected = len(sess["clips"])
        if len(windows) != expected:
            reasons.append(f"block produced {len(windows)} of {expected} windows")
            for w in windows:
                w["valid"] = False
                w["invalid_reasons"].append(reasons[-1])
        valid = not reasons and len(windows) == expected and all(w["valid"] for w in windows)
        command = None
    else:
        cmd_reasons, command = command_checks(command_cfg or {}, events, samples, phases, idle_busy, e_names, p_names)
        reasons += cmd_reasons
        valid = not reasons
    out_cmd = {"command": command} if mode == "command" else {}
    return {**out_cmd, "span_state": span_state, "stream": stream, "band": band, "rate": rate,
            "u_ns": u_bound_ns(samples, spawn), "delta_a2_ns": delta_a2_ns, "storage": storage,
            "idle": {"t0_ns": t_i0, "t1_ns": t_i1, "samples": len(idle_samples), "reference": ref,
                     "cluster_busy": idle_busy, "p_cluster_busy_max": p_idle_busy},
            "chain": pm.chain_stats(samples), "windows": windows, "block_invalid_reasons": list(dict.fromkeys(reasons)),
            "valid": valid}


# --- one block on the Mac ----------------------------------------------------------------------------

def darwin_process_bg(pid: int) -> int:
    import ctypes
    return ctypes.CDLL("/usr/lib/libSystem.B.dylib").getpriority(4, pid)  # PRIO_DARWIN_PROCESS


def run_block(sess: dict, argv: list[str], stem: str, log, plumbing: bool = False, mode: str = "block",
              command_cfg: dict | None = None, replays: list[int] | None = None, delta_a2_ns: int = 0) -> dict:
    """Runs one fresh process (`argv`) under the full monitoring; raw evidence goes to stem.* (created once)."""
    rec = {"argv": argv, "authorization": AUTHORIZATION, "plumbing": plumbing, "mode": mode,
           "started_unix": time.time(), "quiet": None if plumbing else wait_quiet(log)}
    active: list = []
    with ExclusiveLock(shared=plumbing) as lk:
        rec["lock_waited_s"] = lk.waited_s
        rec["lock_mode"] = "shared" if plumbing else "exclusive"
        caff = subprocess.Popen(CAFFEINATE)
        raw_path = stem + ".powermetrics.plist0"
        if os.path.exists(raw_path):
            raise FileExistsError(raw_path)
        own = lambda: [os.getpid(), caff.pid] + list(active)  # noqa: E731
        # Lifecycle (c4 blocker): the state sampler runs from before the spawn to after the capture ends,
        # so the block span [spawn, span1] has a completed state sample on each side.
        sampler = macstate.Sampler(own, **STATE_PROBES).start()
        reader = None
        events, exit_code, phases = [], None, {}
        try:
            if not sampler.wait_after(now_ns(), STATE_WAIT_S):
                raise RuntimeError("state sampler produced no sample before the spawn")
            reader = pm.Reader(raw_path)
            phases["spawn_ns"] = reader.spawn_ns
            if not reader.wait_for(5, 60):
                raise RuntimeError("powermetrics produced no samples")
            phases["idle0"] = now_ns()
            time.sleep(IDLE_S)
            phases["idle1"] = now_ns()
            err_path = stem + ".stderr"
            fd = os.open(err_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(fd, "wb") as err:
                phases["proc_start"] = now_ns()
                p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=err, stdin=subprocess.DEVNULL)
                active.append(p.pid)
                phases["child_bg"] = [darwin_process_bg(p.pid)] if mode == "command" else []
                for line in p.stdout:
                    t = now_ns()
                    if mode == "command":   # -1 once the child has exited (ESRCH): ignored by the gate
                        phases["child_bg"].append(darwin_process_bg(p.pid))
                    if mode == "block":
                        try:
                            ev = json.loads(line)
                        except ValueError:
                            ev = {"event": "unparsed", "line": line.decode(errors="replace")[:500]}
                    else:
                        ev = {"event": "stdout", "line": line.decode(errors="replace").rstrip("\n")}
                    ev["received_ns"] = t
                    events.append(ev)
                exit_code = p.wait()
                p.stdout.close()
                active.remove(p.pid)
            phases["proc_end"] = now_ns()
            # Bracketing (A1): keep reading until a record starts, under every offset, after the last
            # window's t1 + G (or after the process end for commands).
            last = max([e["t_ns"] for e in events if e.get("event") == "window_end"] or [phases["proc_end"]])
            deadline = time.monotonic() + BRACKET_TIMEOUT_S
            while (not bracketed(pm.build_timeline(reader.snapshot()), phases["spawn_ns"], last + guard.G_NS, delta_a2_ns)
                   and time.monotonic() < deadline):
                time.sleep(0.1)
        finally:
            if reader is not None:
                rec["powermetrics"] = reader.stop()
            phases["span1"] = now_ns()                       # the capture has ended
            rec["state_after_end"] = sampler.wait_after(phases["span1"], STATE_WAIT_S)
            sampler.stop()
            caff.terminate()
            caff.wait(timeout=10)
    phases["stream"] = rec["powermetrics"].get("stream")
    samples = reader.snapshot()
    state = sampler.snapshot()
    rec.update(events=events, exit_code=exit_code, phases=phases, replays=replays, delta_a2_ns=delta_a2_ns)
    rec["powermetrics"]["raw_sha256"] = sha256_file(raw_path)
    rec["powermetrics"]["arrivals_ns"] = [s.arrival_ns for s in samples]
    rec["state_samples"] = [s.__dict__ for s in state]
    rec["analysis"] = analyse_block(sess, events, samples, state, phases, exit_code, mode, command_cfg,
                                    replays=replays, delta_a2_ns=delta_a2_ns)
    if plumbing:
        rec["analysis"]["valid"] = False
        rec["analysis"]["block_invalid_reasons"].append("plumbing run: not a measurement")
    rec["ended_unix"] = time.time()
    return rec


def cond_key(cond: dict) -> str:
    return f"threads={cond['threads']},components={cond['components']}"


def sizing_key(sess: dict, stage: str | None, cond: dict) -> str:
    """The sizing record a stage uses: its "sizing_key" (e.g. the tuning grids, sized together with
    their own roster) or, by default, its condition's key."""
    st = (sess.get("stages") or {}).get(stage) or {}
    if st.get("type") == "sizing":
        return st.get("key") or cond_key(cond)
    return st.get("sizing_key") or cond_key(cond)


def block_argv(sess: dict, blk: dict, plumbing: bool = False, replays: list[int] | None = None) -> list[str]:
    """The arm's `block` command. `replays` (one N per clip, in the session's clip order) defaults to the
    frozen sizing of the block's condition; plumbing runs use one replay per clip."""
    arm = sess["arms"][blk["arm"]]
    cond = blk["condition"]
    if replays is None:
        if plumbing:      # never a measurement; enough replays per window to exercise the energy gates
            replays = [int(sess.get("plumbing_replays", 1))] * len(sess["clips"])
        else:
            n = frozen_n(sess, sizing_key(sess, blk.get("stage"), cond))
            replays = [int(n[c] if c in n else n[str(c)]) for c in sess["clips"]]
    argv = [arm["binary"], "block", "--export", arm["export"], "--captures", arm["captures"],
            "--clips", ",".join(str(c) for c in sess["clips"]), "--threads", str(cond["threads"]),
            "--components", str(cond["components"]), "--replays", ",".join(str(r) for r in replays),
            "--guard-s", repr(guard.G_NS / 1e9), "--settle-s", repr(guard.S_NS / 1e9),
            "--settle-post-s", repr(guard.S_POST_NS / 1e9), "--block-id", blk["block_id"]]
    if arm.get("tuning") is not None:
        argv += ["--tuning", json.dumps(arm["tuning"])]
    if plumbing:
        argv += ["--allow-background", "1"]
    return argv


def argv_replays(argv: list[str]) -> list[int]:
    return [int(x) for x in argv[argv.index("--replays") + 1].split(",")]


# --- fixed replay count: sizing, preflight, freezing (plan "Fixed N and sizing") ----------------------

def size_condition(latency_s: dict, plan: dict) -> dict:
    """latency_s: {arm: {clip: per-replay s}}; plan: {"idle_s", "warmup_s": {arm: s}, "clip_load_s"}.
    N per clip = ceil(12 s / (0.95 x the fastest arm's latency)), the same for every arm; the planned
    block of each arm = idle + warmup + sum over clips of (N x latency + S + 2G + S' + clip load + one
    untimed replay); feasible iff every block <= A_MAX - 60 s."""
    arms = sorted(latency_s)
    clips = list(latency_s[arms[0]])
    g, s_pre, s_post = guard.G_NS / 1e9, guard.S_NS / 1e9, guard.S_POST_NS / 1e9
    n = {}
    for c in clips:
        fastest = min(latency_s[a][c] for a in arms)
        if not (fastest > 0):
            raise ValueError(f"clip {c}: non-positive latency")
        n[c] = int(math.ceil(WINDOW_TARGET_S / (SIZING_MARGIN * fastest)))
    block_s = {}
    for a in arms:
        per_clip = sum(n[c] * latency_s[a][c] + s_pre + 2 * g + s_post + plan["clip_load_s"] + latency_s[a][c]
                       for c in clips)
        block_s[a] = plan["idle_s"] + plan["warmup_s"][a] + per_clip
    limit = guard.A_MAX_NS / 1e9 - PREFLIGHT_SLACK_S
    reasons = [f"{a}: planned block {block_s[a]:.1f} s > A_MAX - {PREFLIGHT_SLACK_S:.0f} s = {limit:.1f} s"
               for a in arms if block_s[a] > limit]
    return {"N": n, "feasible": not reasons, "block_s": block_s, "reasons": reasons, "latency_s": latency_s,
            "plan": plan, "limit_s": limit}


def _canonical(doc):
    """JSON-canonical form (dict keys as strings) so an in-memory and a reloaded record hash alike."""
    return json.loads(json.dumps(doc, sort_keys=True, default=str))


def _sizing_sha(rec: dict) -> str:
    return sha256_json(_canonical({k: rec.get(k) for k in ("N", "latency_s", "feasible", "block_s")}))


def freeze_sizing(session: dict, condition: str, sizing: dict) -> dict:
    out = copy.deepcopy(session)
    rec = {"N": copy.deepcopy(sizing["N"]), "latency_s": copy.deepcopy(sizing.get("latency_s")),
           "feasible": bool(sizing["feasible"]), "block_s": copy.deepcopy(sizing.get("block_s")),
           "reasons": list(sizing.get("reasons") or [])}
    rec["sha256"] = _sizing_sha(rec)
    out.setdefault("sizing", {})[condition] = rec
    return out


def frozen_n(session: dict, condition: str) -> dict:
    rec = (session.get("sizing") or {}).get(condition)
    if rec is None:
        raise SystemExit(f"no frozen sizing for {condition}: run that condition's sizing stage first")
    if rec.get("sha256") != _sizing_sha(rec):
        raise SystemExit(f"the frozen sizing of {condition} does not match its SHA-256")
    if not rec.get("feasible"):
        raise SystemExit(f"{condition} is infeasible under A_MAX ({rec.get('reasons')}); fixing it is a plan change")
    return rec["N"]


def short_windows(windows: list[dict], min_s: float = WINDOW_TARGET_S) -> list[dict]:
    return [w for w in windows if (w["t1_ns"] - w["t0_ns"]) / 1e9 < min_s]


def load_frozen_sizing(sess: dict) -> dict:
    """Merges every condition's write-once sizing record (results/<session>/<sizing stage>/sizing.json)."""
    for name, st in sess.get("stages", {}).items():
        if st.get("type") != "sizing":
            continue
        path = os.path.join(ROOT, "results", sess["session"], name, "sizing.json")
        if os.path.exists(path):
            rec = read_json(path)
            sess.setdefault("sizing", {})[rec["condition"]] = {k: rec[k] for k in
                                                               ("N", "latency_s", "feasible", "block_s", "reasons", "sha256")}
    return sess


def sizing_from_blocks(records: dict, clips: list) -> dict:
    """Latencies, warmup and clip-load times from the sizing blocks ({arm: block record})."""
    lat, warm, load = {}, {}, []
    pre = (guard.S_NS + guard.G_NS) / 1e9
    for arm, rec in records.items():
        ev = rec["events"]
        t = {e["event"]: e["t_ns"] for e in ev if "t_ns" in e and "window" not in e}
        warm[arm] = (t["warmup_done"] - t["start"]) / 1e9 + 5.0     # + process start, the storage audit, slack
        ends = sorted((e for e in ev if e.get("event") == "window_end"), key=lambda e: e["window"])
        starts = {e["window"]: e for e in ev if e.get("event") == "window_start"}
        lat[arm] = {}
        prev = t.get("storage", t["warmup_done"])
        for e in ends:
            c = clips[e["window"]]
            lat[arm][c] = (e["t_ns"] - starts[e["window"]]["t_ns"]) / 1e9 / e["replays"]
            t0 = starts[e["window"]]["t_ns"]
            load.append(max(0.0, (t0 - prev) / 1e9 - pre - lat[arm][c]))
            prev = e["t_ns"] + guard.G_NS + guard.S_POST_NS
    return {"latency_s": lat, "plan": {"idle_s": IDLE_S + 3.0, "warmup_s": warm, "clip_load_s": max(load) + 0.5 if load else 1.0}}


# --- immutable stage records and execution (pure; findings 3, 4, 7) ----------------------------------

def stage_fingerprint(sess: dict, stage: str, validation_path: str | None, harness_dir: str = HERE) -> dict:
    st = {k: v for k, v in sess["stages"][stage].items() if k != "about"}
    arms = {}
    for name in st.get("arms", []):
        a = sess["arms"][name]
        arms[name] = {"binary": a["binary"], "binary_sha256": sha256_file(a["binary"]),
                      "export": a["export"], "export_manifest_sha256": sha256_file(os.path.join(a["export"], "manifest.json")),
                      "captures": a["captures"], "captures_index_sha256": sha256_file(os.path.join(a["captures"], "index.json")),
                      "tuning": a.get("tuning")}
    sizing = None
    if "condition" in st:
        rec = (sess.get("sizing") or {}).get(sizing_key(sess, stage, st["condition"]))
        sizing = None if rec is None else {"N": _canonical(rec["N"]), "sha256": rec.get("sha256")}
    return {"session": sess["session"], "stage": stage, "stage_config": st,
            "session_config": {k: sess.get(k) for k in ("seed", "clips")}, "sizing": sizing,
            "arms": arms, "harness_sha256": {f: sha256_file(os.path.join(harness_dir, f)) for f in HARNESS_FILES},
            "validation": None if validation_path is None else {"path": validation_path, "sha256": sha256_file(validation_path)},
            "constants": frozen_constants()}


def ensure_stage(out_dir: str, fp: dict, blocks: list[dict]) -> str:
    """Writes stage.json and design.json once; a resume must reproduce both exactly (finding 7)."""
    fp_sha = sha256_json(fp)
    stage_doc = {"fingerprint": fp, "fingerprint_sha256": fp_sha, "authorization": AUTHORIZATION}
    design_doc = {"fingerprint_sha256": fp_sha, "blocks": blocks}
    for name, doc in (("stage.json", stage_doc), ("design.json", design_doc)):
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            old = read_json(path)
            if name == "stage.json" and old["fingerprint_sha256"] != fp_sha:
                diff = sorted(k for k in fp if old["fingerprint"].get(k) != json.loads(json.dumps(fp[k], default=str)))
                raise SystemExit(f"refusing to resume: the stage fingerprint changed ({', '.join(diff)}); "
                                 f"use a new session or stage name")
            if name == "design.json" and sha256_json(old) != sha256_json(design_doc):
                raise SystemExit("refusing to resume: design.json differs from the stage's design")
        else:
            write_new(path, doc)
    return fp_sha


def sequences(blocks: list[dict]) -> list[list[dict]]:
    out: list[list[dict]] = []
    for b in blocks:
        if not out or (out[-1][0]["replicate"], out[-1][0]["sequence"]) != (b["replicate"], b["sequence"]):
            out.append([])
        out[-1].append(b)
    return out


def seq_prefix(seq: list[dict]) -> str:
    return f"seq-r{seq[0]['replicate']}-s{seq[0]['sequence']}"


def attempt_numbers(out_dir: str, seq: list[dict]) -> list[int]:
    pre = seq_prefix(seq) + ".att"
    nums = set()
    for f in os.listdir(out_dir):
        if f.startswith(pre) and f.endswith(".start.json"):
            nums.add(int(f[len(pre):-len(".start.json")]))
    return sorted(nums)


def valid_attempt(out_dir: str, seq: list[dict], fp_sha: str) -> int | None:
    """The lowest attempt number whose end record is valid and whose every block record exists, matches
    the design block, the attempt and the fingerprint, and is valid (finding 3)."""
    for n in attempt_numbers(out_dir, seq):
        end = os.path.join(out_dir, f"{seq_prefix(seq)}.att{n}.end.json")
        if not os.path.exists(end) or not read_json(end).get("valid"):
            continue
        ok = True
        for b in seq:
            path = os.path.join(out_dir, f"{b['block_id']}.att{n}.json")
            if not os.path.exists(path):
                ok = False
                break
            r = read_json(path)
            if (r.get("block") != b or r.get("attempt") != n or r.get("fingerprint_sha256") != fp_sha
                    or not r["analysis"]["valid"]):
                ok = False
                break
        if ok:
            return n
    return None


def execute_stage(blocks: list[dict], out_dir: str, fp_sha: str, run_one, log,
                  max_attempts: int = MAX_ATTEMPTS, passes: int = PASSES) -> dict:
    """Runs every sequence until one attempt is wholly valid (M6). run_one(blk, stem) -> block record
    (raw evidence written under stem); this function writes the record and the attempt ledgers."""
    seqs = sequences(blocks)
    for pas in range(passes):
        pending = [s for s in seqs if valid_attempt(out_dir, s, fp_sha) is None]
        if not pending:
            break
        for seq in pending:
            for _ in range(max_attempts):
                nums = attempt_numbers(out_dir, seq)
                n = (nums[-1] + 1) if nums else 0
                pre = os.path.join(out_dir, f"{seq_prefix(seq)}.att{n}")
                write_new(pre + ".start.json", {"replicate": seq[0]["replicate"], "sequence": seq[0]["sequence"],
                                                "attempt": n, "pass": pas, "blocks": [b["block_id"] for b in seq],
                                                "fingerprint_sha256": fp_sha, "started_unix": time.time()})
                valid, files = True, []
                for b in seq:
                    stem = os.path.join(out_dir, f"{b['block_id']}.att{n}")
                    rec = run_one(b, stem)
                    rec.update(block=b, attempt=n, fingerprint_sha256=fp_sha)
                    write_new(stem + ".json", rec)
                    files.append(os.path.basename(stem) + ".json")
                    valid = valid and bool(rec["analysis"]["valid"])
                write_new(pre + ".end.json", {"attempt": n, "valid": valid, "files": files, "ended_unix": time.time()})
                log({"event": "sequence_attempt", "replicate": seq[0]["replicate"], "sequence": seq[0]["sequence"],
                     "attempt": n, "pass": pas, "valid": valid})
                if valid:
                    break
    missing = [{"replicate": s[0]["replicate"], "sequence": s[0]["sequence"]} for s in seqs
               if valid_attempt(out_dir, s, fp_sha) is None]
    status = {"status": "complete" if not missing else "incomplete", "missing_sequences": missing,
              "sequences": len(seqs), "fingerprint_sha256": fp_sha, "at_unix": time.time()}
    write_new(os.path.join(out_dir, f"run-{time.strftime('%Y%m%dT%H%M%S')}-{time.time_ns()}.status.json"), status)
    return status


# --- validation: the A1-A3 tests and the instrument checks (plan "Assumptions A1-A3 and validation") ----

def _capture(out_dir: str, stamp: str, tag: str, log, plumbing: bool, body) -> dict:
    """One powermetrics capture under the lock and caffeinate; body(reader, active, marks) runs inside."""
    active: list = []
    marks: dict = {}
    with ExclusiveLock(shared=plumbing):
        caff = subprocess.Popen(CAFFEINATE)
        reader = pm.Reader(os.path.join(out_dir, f"validation-{stamp}-{tag}.powermetrics.plist0"))
        try:
            body(reader, active, marks)
        finally:
            info = reader.stop()
            caff.terminate()
            caff.wait(timeout=10)
    samples = reader.snapshot()
    pm.build_timeline(samples)
    return {"spawn_ns": reader.spawn_ns, "samples": samples, "powermetrics": info, "marks": marks}


def validate_capture(cap: dict) -> dict:
    """A1 (stream, band), A3 (rate, J), idle reference, energy-interval test and spin power of the long
    capture (pure; cap = {"spawn_ns", "samples", "powermetrics", "marks": {"idle0", "idle1", "spins"}})."""
    s, spawn, marks = cap["samples"], cap["spawn_ns"], cap["marks"]
    good = [x for x in s if x.ok]
    stream = cap["powermetrics"].get("stream") or {"ok": False, "reasons": ["no stream verdict"]}
    band = guard.band_check(s)
    rate = guard.rate_check(s)
    idle = guard.samples_within(s, spawn, marks["idle0"], marks["idle1"])
    ref = guard.idle_reference(idle) if len(idle) >= MIN_IDLE_SAMPLES else None
    spins = [(b["on_ns"], b["off_ns"]) for b in marks.get("spins", [])]
    interval = guard.energy_interval_check(s, spawn, spins, ref) if ref is not None else {"valid": False, "reasons": ["no idle reference"]}
    spin_w = guard.spin_excess_power_w(s, spawn, spins, ref) if ref is not None and spins else None
    u = u_bound_ns(s, spawn)
    units = [abs(x.cpu_power_mw * x.elapsed_ns / 1e9 - x.cpu_energy_mj) for x in good]
    idle_busy = pm.cluster_busy(s, marks["idle0"], marks["idle1"])
    reasons = []
    for name, ok, why in (("stream", stream.get("ok"), stream.get("reasons")), ("band (A1)", band["valid"], band.get("reasons")),
                          ("rate (A3)", rate["valid"], rate.get("reasons")),
                          ("energy-interval test", interval["valid"], interval.get("reasons"))):
        if not ok:
            reasons.append(f"{name}: {why}")
    if ref is None:
        reasons.append(f"idle window: {len(idle)} samples, need {MIN_IDLE_SAMPLES}")
    if u is None or u < 0 or u > guard.U_MAX_NS:
        reasons.append(f"U = {u} outside [0, U_MAX]")
    if spin_w is None or not math.isfinite(spin_w) or spin_w <= 0:
        reasons.append(f"spin excess power {spin_w} not measurable")
    capture_s = (good[-1].arrival_ns - spawn) / 1e9 if good else 0.0
    return {"ok": not reasons, "reasons": reasons, "stream": stream, "band": band, "rate": rate,
            "jitter_ns": rate.get("jitter_ns"), "u_ns": u, "capture_s": capture_s,
            "first_sample_bound_ns": (good[0].arrival_ns - spawn - good[0].elapsed_ns) if good else None,
            "idle_reference": ref, "idle_samples": len(idle), "idle_cluster_busy": idle_busy,
            "p_cluster_idle_share": max(((v or 0.0) for k, v in idle_busy.items() if k.startswith("P")), default=None),
            "energy_interval": interval, "spin_excess_power_w": spin_w, "spins": marks.get("spins", []),
            "samples": len(s), "malformed": sum(1 for x in s if not x.ok),
            "elapsed_ns": _stats([x.elapsed_ns for x in good]),
            "arrival_interval_ns": _stats([b.arrival_ns - a.arrival_ns for a, b in zip(good, good[1:])]),
            "unit_check_mj": {"max_abs_power_x_elapsed_minus_energy": max(units) if units else None}}


def delta_a2_ns(tol_mj: float | None, spin_excess_w: float | None) -> int | None:
    """delta_A2 = tol / spin excess power (mJ / W = ms)."""
    vals = (tol_mj, spin_excess_w)
    if any(v is None or not math.isfinite(v) or v <= 0 for v in vals):
        return None
    return int(math.ceil(tol_mj / spin_excess_w * 1e6))


def instrument_ok(analysis: dict) -> tuple[bool, list[str], float | None]:
    """Validate's instrument pass rule: every window valid, U <= U_MAX, median relative width <= 1%."""
    ws = analysis["windows"]
    reasons = []
    if not analysis["valid"]:
        reasons.append("instrument block invalid: " + "; ".join(
            analysis["block_invalid_reasons"][:4] + sorted({r for w in ws for r in w["invalid_reasons"]})[:4]))
    widths = [(w["energy_j_window"]["upper"] - w["energy_j_window"]["lower"]) / w["energy_j_window"]["est"]
              for w in ws if w.get("energy_j_window") and w["energy_j_window"]["est"]]
    med = statistics.median(widths) if widths else None
    if med is None or med > MAX_RELATIVE_WIDTH:
        reasons.append(f"median relative bound width {med} > {MAX_RELATIVE_WIDTH}")
    if any(w.get("u_ns") is None or w["u_ns"] > guard.U_MAX_NS for w in ws):
        reasons.append("U above U_MAX in a window")
    return not reasons, reasons, med


def validate(sess: dict, out_dir: str, log, plumbing: bool = False) -> dict:
    """The A1-A3 tests and the instrument checks. A failure stops all timing (no ok record is written)."""
    stamp = time.strftime("%Y%m%dT%H%M%S")
    rec = {"authorization": AUTHORIZATION, "started_unix": time.time(), "plumbing": plumbing,
           "quiet": None if plumbing else wait_quiet(log), "stamp": stamp}
    capture_s = 40.0 if plumbing else VALIDATE_CAPTURE_S
    n_spins = 3 if plumbing else VALIDATE_SPINS
    n_trials = 2 if plumbing else PRESPAWN_TRIALS

    def long_body(reader, active, marks):
        if not reader.wait_for(5, 60):
            raise RuntimeError("powermetrics produced no samples")
        marks["idle0"] = now_ns()
        time.sleep(IDLE_S)
        marks["idle1"] = now_ns()
        marks["spins"] = run_spins(n_spins, SPIN_S, SPIN_GAP_S, 0.0, active)
        rest = capture_s - (now_ns() - reader.spawn_ns) / 1e9
        if rest > 0:
            time.sleep(rest)

    cap = _capture(out_dir, stamp, "capture", log, plumbing, long_body)
    capr = validate_capture(cap)
    rec["capture"] = {k: v for k, v in capr.items()}
    # A record per raw stream (pull.py copies a stream once its record exists).
    write_new(os.path.join(out_dir, f"validation-{stamp}-capture.json"),
              {"spawn_ns": cap["spawn_ns"], "marks": cap["marks"], "powermetrics": cap["powermetrics"],
               "arrivals_ns": [x.arrival_ns for x in cap["samples"]], "analysis": capr})
    log({"event": "validate_capture", "ok": capr["ok"], "reasons": capr["reasons"]})
    ref = capr["idle_reference"]
    trials = []
    for i in range(n_trials):
        time.sleep(3.0)

        def trial_body(reader, active, marks):
            reader.wait_for(int(PRESPAWN_READ_S * 9), PRESPAWN_READ_S + 5)

        # The spin runs in a child that exits right after it; the reader (and s) follow immediately.
        active: list = []
        with ExclusiveLock(shared=plumbing):
            spin = run_spins(1, PRESPAWN_SPIN_S, 0.0, 0.0, active)
            caff = subprocess.Popen(CAFFEINATE)
            reader = pm.Reader(os.path.join(out_dir, f"validation-{stamp}-prespawn{i:02d}.powermetrics.plist0"))
            try:
                trial_body(reader, active, {})
            finally:
                info = reader.stop()
                caff.terminate()
                caff.wait(timeout=10)
        samples = reader.snapshot()
        pm.build_timeline(samples)
        r = guard.prespawn_trial(samples, reader.spawn_ns, ref) if ref is not None else {"valid": False, "reasons": ["no idle reference"]}
        trials.append({"trial": i, "spin": spin[0] if spin else None, "spawn_ns": reader.spawn_ns,
                       "spin_end_to_spawn_ns": (reader.spawn_ns - spin[0]["off_ns"]) if spin else None,
                       "stream": info.get("stream"), **{k: r.get(k) for k in ("valid", "first_excess_mj", "reasons")}})
        write_new(os.path.join(out_dir, f"validation-{stamp}-prespawn{i:02d}.json"),
                  dict(trials[-1], powermetrics=info, arrivals_ns=[x.arrival_ns for x in samples]))
    a2_ok = bool(trials) and all(t["valid"] and (t.get("stream") or {}).get("ok") for t in trials)
    d_a2 = delta_a2_ns(ref["tol_mj"] if ref else None, capr["spin_excess_power_w"])
    rec["prespawn"] = {"trials": trials, "ok": a2_ok, "delta_a2_ns": d_a2}
    log({"event": "validate_prespawn", "ok": a2_ok, "delta_a2_ns": d_a2})
    rec["delta_a2_ns"] = d_a2
    rec["jitter_ns"] = capr["jitter_ns"]
    inst = sess.get("instrument_block")
    rec["instrument"] = None
    inst_ok = False
    if inst and capr["ok"] and a2_ok and d_a2 is not None or (inst and plumbing):
        blk = {"stage": "validate", "replicate": 0, "sequence": 0, "sequence_order": 0, "position": 0,
               "arm": inst["arm"], "condition": inst["condition"], "instrument_only": True}
        lat_blk = dict(blk, block_id=f"validate-latency-{inst['arm']}")
        stem = os.path.join(out_dir, f"{lat_blk['block_id']}-{stamp}")
        ones = [1] * len(sess["clips"])
        b = run_block(sess, block_argv(sess, lat_blk, plumbing, replays=ones), stem, log, plumbing,
                      replays=ones, delta_a2_ns=d_a2 or 0)
        b["block"] = lat_blk
        write_new(stem + ".json", b)
        lat = sizing_from_blocks({inst["arm"]: b}, sess["clips"])["latency_s"].get(inst["arm"], {})
        n = [1] * len(sess["clips"]) if plumbing else [int(math.ceil(WINDOW_TARGET_S / (SIZING_MARGIN * lat[c])))
                                                       for c in sess["clips"]]
        ins_blk = dict(blk, block_id=f"validate-instrument-{inst['arm']}")
        stem = os.path.join(out_dir, f"{ins_blk['block_id']}-{stamp}")
        b = run_block(sess, block_argv(sess, ins_blk, plumbing, replays=n), stem, log, plumbing,
                      replays=n, delta_a2_ns=d_a2 or 0)
        b["block"] = ins_blk
        write_new(stem + ".json", b)
        inst_ok, inst_reasons, med = instrument_ok(b["analysis"])
        rec["instrument"] = {"ok": inst_ok, "reasons": inst_reasons, "median_relative_width": med, "replays": n,
                             "latency_s": lat, "summary": instrument_summary(b)}
    elif inst:
        rec["instrument"] = {"ok": False, "reasons": ["skipped: the A1-A3 tests did not pass"]}
    rec["reasons"] = (capr["reasons"] + ([] if a2_ok else ["A2 pre-spawn trials failed"])
                      + ([] if d_a2 is not None else ["delta_A2 not measurable"])
                      + ([] if inst_ok else ["instrument block: " + "; ".join((rec["instrument"] or {}).get("reasons", ["missing"]))]))
    rec["ok"] = (not plumbing) and capr["ok"] and a2_ok and d_a2 is not None and inst_ok
    rec["ended_unix"] = time.time()
    path = os.path.join(out_dir, f"validation-{stamp}.json")
    write_new(path, rec)
    log({"event": "validated", "ok": rec["ok"], "path": path, "reasons": rec["reasons"][:6]})
    return rec


def instrument_summary(b: dict) -> dict:
    """Cadence and gate figures of one block record (instrument validation)."""
    ev = b["events"]
    t = {e["event"]: e["t_ns"] for e in ev if "t_ns" in e and "window" not in e}
    a = b["analysis"]
    ws = a["windows"]
    ph = b["phases"]
    phases = {"quiet_wait_s": (b.get("quiet") or {}).get("waited_s"), "lock_wait_s": b.get("lock_waited_s"),
              "block_wall_s": b["ended_unix"] - b["started_unix"],
              "spawn_to_idle_s": (ph["idle0"] - ph["spawn_ns"]) / 1e9, "idle_s": (ph["idle1"] - ph["idle0"]) / 1e9,
              "process_start_to_loaded_s": (t["loaded"] - t["start"]) / 1e9 if "loaded" in t and "start" in t else None,
              "warmup_s": (t["warmup_done"] - t["loaded"]) / 1e9 if "warmup_done" in t and "loaded" in t else None,
              "windows_span_s": (max(w["t1_ns"] for w in ws) - min(w["t0_ns"] for w in ws)) / 1e9 if ws else None,
              "windows_timed_s": sum((w["t1_ns"] - w["t0_ns"]) / 1e9 for w in ws),
              "block_age_s": (ph["span1"] - ph["spawn_ns"]) / 1e9}
    return {"phases": phases, "valid": a["valid"], "block_invalid_reasons": a["block_invalid_reasons"],
            "window_invalid_reasons": sorted({r for w in ws for r in w["invalid_reasons"]}),
            "u_ns": a.get("u_ns"), "band": a.get("band"), "rate": a.get("rate"), "storage": a.get("storage"),
            "idle": a["idle"],
            "latency_s_per_replay": [round(w["latency_s_per_replay"], 5) for w in ws],
            "energy_j_per_replay": [w["energy_j_per_replay"] for w in ws],
            "guard_j_per_replay": [w["guard_j_per_replay"] for w in ws],
            "x_count": [w["x_count"] for w in ws], "flags": [w["flags"] for w in ws],
            "replays": [w["replays"] for w in ws], "chain": a["chain"],
            "foreign_busy_cores_max": [w["foreign_busy_cores_max"] for w in ws],
            "e_cluster_flag": [w["e_cluster_flag"] for w in ws], "both_p_clusters_active": [w["both_p_clusters_active"] for w in ws],
            "qos": [e for e in ev if e.get("event") in ("qos", "qos_end")]}


def _stats(v):
    if not v:
        return None
    v = sorted(v)
    return {"n": len(v), "min": v[0], "max": v[-1], "median": v[len(v) // 2], "mean": sum(v) / len(v),
            "sd": statistics.pstdev(v)}


def latest_ok_validation(sess: dict) -> str | None:
    """The newest completed non-plumbing validation record, if it passed; None otherwise. A newer failure
    revokes permission to time: it is never skipped in favour of an older success (c4 major 3). Plumbing
    records and the per-stream records (validation-<stamp>-capture.json, -prespawnNN.json) are ignored."""
    d = os.path.join(ROOT, "results", sess["session"], "validate")
    if not os.path.isdir(d):
        return None
    paths = sorted(p for p in os.listdir(d) if re.fullmatch(r"validation-\d{8}T\d{6}\.json", p))
    for p in reversed(paths):
        rec = read_json(os.path.join(d, p))
        if rec.get("plumbing"):
            continue
        return os.path.join(d, p) if rec.get("ok") is True else None
    return None


def blocks_for(sess: dict, stage: str) -> list[dict]:
    st = sess["stages"][stage]
    return design.design(st["arms"], st["replicates"], sess["seed"] + hash_stage(stage), stage, st["condition"])


def hash_stage(stage: str) -> int:
    return int(hashlib.sha256(stage.encode()).hexdigest()[:6], 16)


ENERGY_REASON_PREFIX = "energy: "    # the window energy checks (guard.window_bounds), from analyse_block


def sizing_block_ok(rec: dict, n_clips: int) -> tuple[bool, list[str]]:
    """Whether a sizing block's latencies may be used (c4 major 2): every block-level gate (process exit,
    stream, band, rate, host state over the whole span, idle, QoS, storage) and every window-level gate
    (frozen N, dispatch, event timing, host state over the window) must pass. Only the window energy
    checks are exempt: a one-replay window cannot show the energy the bounds need."""
    reasons = []
    if rec.get("exit_code") != 0:
        reasons.append(f"process exit {rec.get('exit_code')}")
    a = rec.get("analysis") or {}
    reasons += list(a.get("block_invalid_reasons") or [])
    ws = a.get("windows") or []
    if len(ws) != n_clips:
        reasons.append(f"{len(ws)} of {n_clips} windows")
    block = set(a.get("block_invalid_reasons") or [])
    for w in ws:
        reasons += [f"window {w.get('window')}: {r}" for r in w.get("invalid_reasons") or []
                    if r not in block and not r.startswith(ENERGY_REASON_PREFIX)]
    reasons = list(dict.fromkeys(reasons))
    return not reasons, reasons


def run_sizing(sess: dict, stage: str, out_dir: str, val: dict, log, run=None) -> dict:
    """One block per arm with one replay per clip (recorded like validate, never compared); only blocks
    passing sizing_block_ok give latencies, rejected blocks are retried (up to MAX_ATTEMPTS x PASSES per
    arm); then N, the preflight and the write-once sizing record. run(blk, argv, stem, replays) -> block
    record (default: run_block)."""
    st = sess["stages"][stage]
    path = os.path.join(out_dir, "sizing.json")
    if os.path.exists(path):
        raise SystemExit(f"{path} exists: a condition is sized once (a new sizing is a new session)")
    d_a2 = val.get("delta_a2_ns") or 0
    if run is None:
        def run(blk, argv, stem, replays):
            return run_block(sess, argv, stem, log, replays=replays, delta_a2_ns=d_a2)
    ones = [1] * len(sess["clips"])
    records, stems = {}, {}
    for arm in st["arms"]:
        blk = {"stage": stage, "replicate": 0, "sequence": 0, "sequence_order": 0, "position": 0, "arm": arm,
               "condition": st["condition"], "block_id": f"{stage}-{arm}", "sizing_only": True}
        for attempt in range(MAX_ATTEMPTS * PASSES):
            stem = os.path.join(out_dir, f"{blk['block_id']}.att{attempt}")
            if os.path.exists(stem + ".json"):
                continue
            b = run(blk, block_argv(sess, blk, replays=ones), stem, ones)
            ok, reasons = sizing_block_ok(b, len(sess["clips"]))
            b.update(block=blk, attempt=attempt, sizing_accepted=ok, sizing_reasons=reasons)
            write_new(stem + ".json", b)
            log({"event": "sizing_block", "arm": arm, "attempt": attempt, "accepted": ok, "reasons": reasons[:4]})
            if ok:
                records[arm] = b
                stems[arm] = os.path.basename(stem) + ".json"
                break
        if arm not in records:
            raise SystemExit(f"sizing: no block of {arm} passed the gates in {MAX_ATTEMPTS * PASSES} attempts")
    inputs = sizing_from_blocks(records, sess["clips"])
    sizing = size_condition(inputs["latency_s"], inputs["plan"])
    key = sizing_key(sess, stage, st["condition"])
    frozen = freeze_sizing({}, key, sizing)["sizing"][key]
    doc = {"condition": key, "stage_condition": st["condition"], **frozen, "plan": inputs["plan"],
           "limit_s": sizing["limit_s"], "blocks": stems, "at_unix": time.time()}
    write_new(path, doc)
    log({"event": "sized", "condition": doc["condition"], "feasible": doc["feasible"], "N": doc["N"],
         "block_s": doc["block_s"], "reasons": doc["reasons"]})
    return doc


def expand_path(p: str) -> str:
    return os.path.expanduser(os.path.expandvars(p))


def expand_session(sess: dict) -> dict:
    """Expands "$CPU1BIT_ROOT" (= ROOT) and "~" in the arms' paths and the command stages' argv."""
    os.environ.setdefault("CPU1BIT_ROOT", ROOT)
    for arm in sess.get("arms", {}).values():
        for k in ("binary", "export", "captures"):
            if isinstance(arm.get(k), str):
                arm[k] = expand_path(arm[k])
    for st in sess.get("stages", {}).values():
        if isinstance(st.get("argv"), list):
            st["argv"] = [expand_path(a) if isinstance(a, str) else a for a in st["argv"]]
    return sess


def main() -> None:
    sess_path, stage = sys.argv[1], sys.argv[2]
    plumbing = stage in ("plumbing", "validate-plumbing")
    sess = expand_session(read_json(sess_path))
    sess_sha = sha256_file(sess_path)
    load_frozen_sizing(sess)
    out_dir = os.path.join(ROOT, "results", sess["session"], stage)  # plumbing never lands in validate/
    os.makedirs(out_dir, exist_ok=True)
    # One append-only log per invocation (c2 finding 3): a resume never rewrites an earlier log.
    log_path = os.path.join(out_dir, f"runner-{time.strftime('%Y%m%dT%H%M%S')}-{time.time_ns()}.log.jsonl")
    logf = os.fdopen(os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644), "a")

    def log(d):
        d = {"t_unix": time.time(), **d}
        logf.write(json.dumps(d, default=str) + "\n")
        logf.flush()
        print(json.dumps(d, default=str), flush=True)

    log({"event": "start", "stage": stage, "session_sha256": sess_sha, "argv": sys.argv, "pid": os.getpid(),
         "python": sys.version, "authorization": AUTHORIZATION, "sizing": sess.get("sizing")})
    if stage not in ("plumbing", "dry-run", "validate-plumbing"):
        # Timed and instrument stages: the job must be `--priority normal` (nice 0, no darwin-background clamp).
        import ctypes
        bg = ctypes.CDLL("/usr/lib/libSystem.B.dylib").getpriority(4, 0)  # PRIO_DARWIN_PROCESS
        log({"event": "priority", "nice": os.nice(0), "darwin_process_bg": bg})
        if os.nice(0) != 0 or bg != 0:
            raise SystemExit("timed stages must run as `job start ... --quiet-host --full-machine --priority normal`")
    if stage in ("validate", "validate-plumbing"):
        rec = validate(sess, out_dir, log, plumbing=stage == "validate-plumbing")
        if not rec["ok"] and not rec["plumbing"]:
            sys.exit(2)
        return
    if stage == "dry-run":
        for name, st in sess["stages"].items():
            if st.get("type") in ("command", "sizing") or name == "plumbing":
                log({"event": st.get("type") or "plumbing", "stage": name, "argv": st.get("argv"), "arms": st.get("arms")})
                continue
            b = blocks_for(sess, name)
            key = sizing_key(sess, name, st["condition"])
            log({"event": "design", "stage": name, "blocks": len(b), "arms": st["arms"], "condition": st["condition"],
                 "sized": key in (sess.get("sizing") or {}),
                 "expected_hours": round(len(b) * sess.get("expected_block_s", 175) / 3600, 2)})
        for name, arm in sess["arms"].items():
            missing = [k for k in ("binary", "export", "captures") if not os.path.exists(arm[k])]
            log({"event": "arm", "arm": name, "missing": missing})
        return
    if plumbing:
        st = sess["stages"]["plumbing"]
        blk = design.design(st["arms"], 1, sess["seed"], "plumbing", st["condition"])[0]
        stem = os.path.join(out_dir, f"{blk['block_id']}-{time.strftime('%Y%m%dT%H%M%S')}")
        argv = block_argv(sess, blk, True)
        rec = run_block(sess, argv, stem, log, plumbing=True, replays=argv_replays(argv))
        rec["block"] = blk
        write_new(stem + ".json", rec)
        log({"event": "plumbing_done", "reasons": rec["analysis"]["block_invalid_reasons"][:8]})
        return
    st = sess["stages"][stage]
    val_path = latest_ok_validation(sess)
    if val_path is None:
        raise SystemExit("run the validate stage first: no ok validation record")
    val = read_json(val_path)
    d_a2 = val.get("delta_a2_ns")
    if d_a2 is None:
        raise SystemExit(f"{val_path} has no delta_a2_ns")
    if st.get("type") == "sizing":
        run_sizing(sess, stage, out_dir, val, log)
        return
    if st.get("type") == "command":
        stem = os.path.join(out_dir, f"{stage}-{time.strftime('%Y%m%dT%H%M%S')}")
        rec = run_block(sess, st["argv"], stem, log, mode="command", command_cfg=st, delta_a2_ns=d_a2)
        rec.update(stage=stage, stage_config=st, validation=val_path, non_comparison=True,
                   argv_sha256={a: sha256_file(a) for a in st["argv"][:1]})
        write_new(stem + ".json", rec)
        log({"event": "command_done", "valid": rec["analysis"]["valid"], "reasons": rec["analysis"]["block_invalid_reasons"],
             "stdout_lines": len(rec["events"])})
        return
    blocks = blocks_for(sess, stage)
    fp = stage_fingerprint(sess, stage, val_path)
    fp_sha = ensure_stage(out_dir, fp, blocks)

    def run_one(b, stem):
        argv = block_argv(sess, b)
        return run_block(sess, argv, stem, log, replays=argv_replays(argv), delta_a2_ns=d_a2)
    status = execute_stage(blocks, out_dir, fp_sha, run_one, log)
    log({"event": "stage_" + status["status"], **status})
    if status["status"] != "complete":
        sys.exit(2)


if __name__ == "__main__":
    main()
