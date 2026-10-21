"""Tests of the runner's block analysis and stage records (Codex code review c1 findings 1, 4, 5, 7;
plan review p8 major 1).

Block-analysis contract (plans/cpu-benchmark-mac.md "Block protocol", "Host-state gates", "Energy",
"Assumptions A1-A3", "Storage and workspace"; written before the implementation):
  runner.analyse_block(sess, events, samples, state, phases, exit_code, mode="block", command_cfg=None,
                       replays=None) -> {"valid", "block_invalid_reasons", "windows": [...], ...}
    phases (CLOCK_UPTIME_RAW ns): spawn_ns (s, read just before spawning powermetrics), idle0, idle1,
      proc_start, proc_end, span1 (end of the capture); stream = pm.check_stream's verdict
      {"ok", "reasons"}; child_bg (command mode)
    replays: the frozen N per window, in clip order (block mode)
    per window: "valid", "invalid_reasons", "flags", "energy_j_per_replay" {"lower", "est", "upper"},
      "guard_j_per_replay", "duration_s"; no idle-corrected (net) energy (plan review p9)
  Block mode: windows are measured with guard.window_bounds against the spawn time; the per-block band
  (A1) and rate (A3) checks, the stream framing, the state over the whole span from spawn to span1, the
  storage audit event, QoS, dispatch, the frozen replay count and the event timing (no window event may
  arrive inside [t0 - G, t1 + G]) all gate. There are no calibration bursts and no absolute anchor.

p8 contract change (from M14/M15): the retired fixture ran work across the padded edges (leading-edge
excess about 1.7 J against a 5 mJ tolerance) and required calibration bursts; it is replaced by the
specified waits and the spawn-based mapping. The warmup, QoS, dispatch and exit assertions are kept.

    python3 tests/test_runner.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import guard  # noqa: E402
import macstate  # noqa: E402
import pm  # noqa: E402
import runner  # noqa: E402
from test_pm import MS, S, synthetic  # noqa: E402

T0 = 10**12                    # s: the runner's clock just before spawning powermetrics
SESS = {"clips": [0, 1]}
REPLAY_NS = 250 * MS
N = 8                          # frozen replay count per window: 2 s windows (short: flagged, not invalid)
REPLAYS = [N, N]
BG_W, WORK_W, LOAD_W, START_W = 0.4, 6.0, 3.0, 2.0


class Block:
    """A complete synthetic block under the specified protocol: powermetrics spawned at T0 (first sample
    300 ms later), a 10 s idle window, the arm process (start, load, warmup), then per clip: clip load,
    one untimed replay, wait S + G, the window of exactly N replays, wait G + S', and only then the
    window events. Power is background plus the process's work; `continuous_work` keeps the process busy
    from start to exit (the retired fixture's profile, which crosses the padded edges)."""

    def __init__(self, seed=1, warmup_disturbance=None, continuous_work=False, drop_storage=False,
                 storage_ok=True, offset_ms=300):
        g, s_pre, s_post = guard.G_NS, guard.S_NS, guard.S_POST_NS
        self.idle0, self.idle1 = T0 + 1 * S, T0 + 11 * S
        self.start = self.idle1 + 100 * MS
        self.loaded = self.start + 1 * S
        self.warm = self.loaded + 2 * S
        self.work = [(self.start, self.loaded, START_W), (self.loaded, self.warm, WORK_W)]
        self.windows, emit = [], []
        t = self.warm
        for _ in SESS["clips"]:
            self.work.append((t, t + 300 * MS, LOAD_W))                  # clip load
            u0 = t + 300 * MS
            u1 = u0 + REPLAY_NS                                         # one untimed replay
            self.work.append((u0, u1, WORK_W))
            t0 = u1 + s_pre + g
            t1 = t0 + N * REPLAY_NS
            self.work.append((t0, t1, WORK_W))
            self.windows.append((t0, t1))
            t = t1 + g + s_post                                         # window events emitted here
            emit.append(t)
        self.proc_end = t + 50 * MS
        self.span1 = self.proc_end + 1500 * MS
        self.continuous_work = continuous_work
        n = int((self.span1 + 2 * S - T0) / (107 * MS))
        self.samples, self.truth_iv = synthetic(self.power, n, seed, latency_max_ns=2 * MS, noise_mj=1.0,
                                                start_ns=T0 + offset_ms * MS)
        for x in self.samples:
            busy = 0.9 if self.start <= x.arrival_ns <= self.proc_end + 200 * MS else 0.02
            x.clusters = {"E-Cluster": {"idle_ratio": 0.9, "freq_hz": 1e9, "cpus": {0: 0.9, 1: 0.9}},
                          "P0-Cluster": {"idle_ratio": 1 - busy, "freq_hz": 3e9, "cpus": {c: 1 - busy for c in range(2, 6)}},
                          "P1-Cluster": {"idle_ratio": 1 - busy, "freq_hz": 3e9, "cpus": {c: 1 - busy for c in range(6, 10)}}}
        ev = [{"event": "start", "t_ns": self.start}, {"event": "qos", "ok": True, "t_ns": self.start},
              {"event": "loaded", "t_ns": self.loaded}, {"event": "warmup_done", "t_ns": self.warm}]
        if not drop_storage:
            ev.append({"event": "storage", "t_ns": self.warm, "ok": storage_ok, "weight_bytes": 72_400_000,
                       "cap_bytes": 79_586_918, "workspace_bytes": 1 << 20, "workspace_cap_bytes": 32 << 20})
        for i, ((t0, t1), e) in enumerate(zip(self.windows, emit)):
            ev.append({"event": "window_start", "window": i, "clip": f"0{i}", "id": f"c{i}", "T": 200, "t_ns": t0})
            ev.append({"event": "window_end", "window": i, "t_ns": t1, "replays": N, "replay_ns": [REPLAY_NS] * N,
                       "gemm_calls": 240 * N, "quantizations": 192 * N})
        ev += [{"event": "qos_end", "ok": True, "t_ns": self.proc_end}, {"event": "end", "t_ns": self.proc_end}]
        for e in ev:                                                    # read by the runner when written
            w = e.get("window")
            e["received_ns"] = (emit[w] if w is not None else e["t_ns"]) + 1 * MS
        self.events = ev
        self.state = []
        t = T0
        while t <= self.span1 + 2 * S:
            kw = dict(ac=True, battery_pct=100, therm_nominal=True, lowpower=0, hid_idle_ns=10**13, load1=0.5,
                      pressure=1, pageouts=10, swapouts=20, host_busy_cores=1.0, own_busy_cores=1.0,
                      foreign_busy_cores=0.0)
            if warmup_disturbance and self.loaded < t <= self.warm:
                kw.update(warmup_disturbance)
            self.state.append(macstate.StateSample(t, **kw))
            t += 1 * S
        self.phases = {"spawn_ns": T0, "idle0": self.idle0, "idle1": self.idle1, "proc_start": self.start,
                       "proc_end": self.proc_end, "span1": self.span1, "stream": {"ok": True, "reasons": []}}

    def power(self, t):
        if self.continuous_work and self.start <= t < self.proc_end:
            return BG_W + WORK_W
        return BG_W + sum(w for a, b, w in self.work if a <= t < b)

    def truth_j(self, i):
        """The measurand of window i: the integral of power over [t0 - G, t1 + G]."""
        t0, t1 = self.windows[i]
        return sum(self.power(t) for t in range(t0 - guard.G_NS, t1 + guard.G_NS, MS)) / 1000.0

    def analyse(self, exit_code=0, events=None, replays=REPLAYS, **kw):
        return runner.analyse_block(SESS, self.events if events is None else events, self.samples, self.state,
                                    self.phases, exit_code, replays=replays, **kw)


def reasons_of(a):
    return list(a["block_invalid_reasons"]) + [r for w in a["windows"] for r in w["invalid_reasons"]]


class BlockAnalysis(unittest.TestCase):
    def test_clean_block_is_valid_and_contains_the_measurand(self):
        b = Block()
        a = b.analyse()
        self.assertTrue(a["valid"], reasons_of(a))
        self.assertNotIn("anchor", a)                                   # retired with the calibration bursts
        self.assertEqual(len(a["windows"]), 2)
        for i, w in enumerate(a["windows"]):
            e = w["energy_j_per_replay"]
            truth = b.truth_j(i) / N
            self.assertLessEqual(e["lower"], truth + 1e-9, (i, e, truth))
            self.assertGreaterEqual(e["upper"], truth - 1e-9, (i, e, truth))
            self.assertLessEqual(e["lower"], e["est"] + 1e-12)
            self.assertLessEqual(e["est"], e["upper"] + 1e-12)
            self.assertTrue(any("short" in f for f in w["flags"]), w["flags"])   # 2 s < 12 s: flag only

    def test_window_columns_gross_guard_and_duration_without_net_energy(self):
        """Plan review p9: no idle-corrected energy. Each window reports the gross padded energy per replay,
        the guard energy per replay (mean power of the samples wholly inside a guard for every admissible
        offset, times 2G, divided by N) and the window duration as separate columns."""
        b = Block()
        a = b.analyse()
        self.assertTrue(a["valid"], reasons_of(a))
        g = guard.G_NS
        for i, w in enumerate(a["windows"]):
            t0, t1 = b.windows[i]
            self.assertFalse([k for k in w if k.startswith(("e_net", "net_")) or "idle_corrected" in k], sorted(w))
            self.assertAlmostEqual(w["duration_s"], (t1 - t0) / 1e9, places=9)
            true_guard = (sum(b.power(t) for t in range(t0 - g, t0, MS))
                          + sum(b.power(t) for t in range(t1, t1 + g, MS))) / 1000.0 / N
            self.assertAlmostEqual(w["guard_j_per_replay"], true_guard, delta=0.03 * true_guard)

    def test_work_across_the_padded_edges_is_invalid(self):
        """p8 major 1: the retired fixture's profile (work from process start to exit) contaminates the
        edges of every padded interval; acceptance would admit contaminated windows."""
        a = Block(continuous_work=True).analyse()
        self.assertFalse(a["valid"])
        self.assertTrue(all(not w["valid"] for w in a["windows"]), reasons_of(a))

    def test_codex_repro_warmup_disturbances_invalidate(self):
        """Review c1 finding 5: critical memory pressure, thermal throttling and four foreign busy
        cores during warmup only."""
        for dist, why in ((dict(pressure=4), "memory pressure"), (dict(therm_nominal=False), "thermal"),
                          (dict(foreign_busy_cores=4.0), "foreign busy"),
                          (dict(pressure=4, therm_nominal=False, foreign_busy_cores=4.0), "block span")):
            a = Block(warmup_disturbance=dist).analyse()
            self.assertFalse(a["valid"], dist)
            self.assertTrue(any(why in r for r in a["block_invalid_reasons"]), (dist, a["block_invalid_reasons"]))
            self.assertTrue(all(not w["valid"] for w in a["windows"]))

    def test_qos_dispatch_replay_count_and_exit_checks(self):
        b = Block()
        self.assertFalse(b.analyse(events=[dict(e, ok=False) if e["event"] == "qos_end" else e for e in b.events])["valid"])
        self.assertFalse(b.analyse(events=[dict(e, gemm_calls=240 * N - 1) if e["event"] == "window_end" else e
                                           for e in b.events])["valid"])
        self.assertFalse(b.analyse(exit_code=3)["valid"])
        self.assertFalse(b.analyse(replays=[N + 1, N])["valid"])           # not the frozen N

    def test_stream_framing_failure_invalidates_block(self):
        """A1, deterministic part: a chunk that does not parse, or EOF before the runner's stop."""
        b = Block()
        b.phases["stream"] = {"ok": False, "reasons": ["EOF before the runner stopped powermetrics"]}
        a = b.analyse()
        self.assertFalse(a["valid"])
        self.assertTrue(any("stream" in r for r in a["block_invalid_reasons"]), a["block_invalid_reasons"])

    def test_producer_skip_outside_every_window_invalidates_block(self):
        """A1, per-block band: one record lost during warmup, far from every padded interval."""
        b = Block(seed=2)
        cut = next(i for i, (a0, _, _) in enumerate(b.truth_iv) if a0 >= b.loaded + 500 * MS)
        del b.samples[cut]
        a = b.analyse()
        self.assertFalse(a["valid"])
        self.assertTrue(any("band" in r for r in a["block_invalid_reasons"]), a["block_invalid_reasons"])

    def test_rate_error_invalidates_block(self):
        """A3: elapsed_ns running 1000 ppm fast against CLOCK_UPTIME_RAW over the block."""
        b = Block(seed=3)
        for x in b.samples:
            x.elapsed_ns = int(x.elapsed_ns * (1 + 1e-3))
        a = b.analyse()
        self.assertFalse(a["valid"])
        self.assertTrue(any("rate" in r for r in a["block_invalid_reasons"]), a["block_invalid_reasons"])

    def test_window_event_inside_the_padded_interval_is_invalid(self):
        """Window events are emitted after the post-window wait; one read inside [t0 - G, t1 + G] makes
        runner activity time-locked to the window."""
        b = Block()
        t0, _ = b.windows[1]
        ev = [dict(e, received_ns=t0 + 1 * S) if e["event"] == "window_start" and e["window"] == 1 else e
              for e in b.events]
        a = b.analyse(events=ev)
        self.assertFalse(a["valid"])
        self.assertTrue(a["windows"][0]["valid"], a["windows"][0]["invalid_reasons"])
        self.assertFalse(a["windows"][1]["valid"])
        self.assertTrue(any("event" in r for r in a["windows"][1]["invalid_reasons"]), a["windows"][1]["invalid_reasons"])

    def test_storage_audit_is_required(self):
        for blk, why in ((Block(drop_storage=True), "missing"), (Block(storage_ok=False), "failed")):
            a = blk.analyse()
            self.assertFalse(a["valid"], why)
            self.assertTrue(any("storage" in r for r in a["block_invalid_reasons"]), (why, a["block_invalid_reasons"]))


ISA_CFG = {"type": "command", "argv": ["p-ubench", "--qos-ui", "isa"],
           "qos_readback": {"regex": r"main-thread qos class (0x[0-9a-fA-F]+) rel (-?[0-9]+)", "class": "0x21", "rel": 0,
                            "min_matches": 1},
           "placement": {"p_cluster_min_rise": 0.10}}


def command_block(line: str, child_bg=(0, 0), e_busy_during=0.1):
    """A Block's evidence turned into a command run: stdout lines instead of replay events."""
    b = Block()
    for x in b.samples:
        if b.start <= x.arrival_ns <= b.proc_end + 200 * MS:
            x.clusters["E-Cluster"]["cpus"] = {0: 1 - e_busy_during, 1: 1 - e_busy_during}
    ph = dict(b.phases, child_bg=list(child_bg))
    events = [{"event": "stdout", "line": "# p-ubench isa (overnight, USER_INTERACTIVE); " + line, "received_ns": b.start + S},
              {"event": "stdout", "line": "eor 4.00/cycle", "received_ns": b.start + 2 * S}]
    return events, b.samples, b.state, ph


class CommandStage(unittest.TestCase):
    """Codex code review c2 finding 4: command stages are gated on QoS readback and placement evidence."""

    def analyse(self, *args, cfg=ISA_CFG):
        ev, s, st, ph = args
        return runner.analyse_block(SESS, ev, s, st, ph, 0, mode="command", command_cfg=cfg)

    def test_ui_readback_with_p_cluster_rise_is_valid(self):
        a = self.analyse(*command_block("main-thread qos class 0x21 rel 0"))
        self.assertTrue(a["valid"], a["block_invalid_reasons"])
        self.assertEqual(a["command"]["qos_readback_lines"], [("0x21", "0")])

    def test_missing_or_wrong_readback_is_invalid(self):
        for line, why in (("no readback here", "readback line"),
                          ("main-thread qos class 0x15 rel 0", "mismatch"),     # inherited / utility class
                          ("main-thread qos class 0x21 rel -5", "mismatch")):
            a = self.analyse(*command_block(line))
            self.assertFalse(a["valid"], line)
            self.assertTrue(any(why in r for r in a["block_invalid_reasons"]), (line, a["block_invalid_reasons"]))

    def test_background_child_or_e_cluster_activity_is_invalid(self):
        a = self.analyse(*command_block("main-thread qos class 0x21 rel 0", child_bg=(1, -1)))
        self.assertTrue(any("darwin-background" in r for r in a["block_invalid_reasons"]))
        a = self.analyse(*command_block("main-thread qos class 0x21 rel 0", child_bg=(-1,)))   # never observed
        self.assertFalse(a["valid"])
        a = self.analyse(*command_block("main-thread qos class 0x21 rel 0", e_busy_during=0.6))
        self.assertTrue(any("E-cluster" in r for r in a["block_invalid_reasons"]), a["block_invalid_reasons"])

    def test_stage_without_qos_specification_is_invalid(self):
        a = self.analyse(*command_block("main-thread qos class 0x21 rel 0"), cfg={"type": "command", "argv": ["x"]})
        self.assertFalse(a["valid"])


def blocks(n_seq=2, k=2):
    out = []
    for s in range(n_seq):
        for p in range(k):
            arm = ["B", "C"][(p + s) % k]
            out.append({"stage": "x", "replicate": 0, "sequence": s, "sequence_order": s, "position": p, "arm": arm,
                        "condition": {"threads": 8, "components": 2}, "block_id": f"x-r0-s{s}-p{p}-{arm}"})
    return out


class StageRecords(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def files(self):
        return sorted(os.listdir(self.dir))

    def test_write_new_never_overwrites(self):
        p = os.path.join(self.dir, "a.json")
        runner.write_new(p, {"x": 1})
        with self.assertRaises(FileExistsError):
            runner.write_new(p, {"x": 2})
        self.assertEqual(runner.read_json(p), {"x": 1})

    def test_retry_exhaustion_is_explicitly_incomplete(self):
        """Review c1 finding 4: four invalid attempts per pass must not shrink the design."""
        bl = blocks()
        calls = []

        def run_one(b, stem):
            calls.append((b["block_id"], stem))
            return {"analysis": {"valid": b["sequence"] == 0}}
        st = runner.execute_stage(bl, self.dir, "fp", run_one, lambda d: None, max_attempts=4, passes=2)
        self.assertEqual(st["status"], "incomplete")
        self.assertEqual(st["missing_sequences"], [{"replicate": 0, "sequence": 1}])
        self.assertEqual(runner.attempt_numbers(self.dir, runner.sequences(bl)[1]), list(range(8)))
        self.assertEqual(len([c for c in calls if "-s1-" in c[0]]), 16)
        # A later resume that succeeds completes the stage with NEW attempt numbers; nothing overwritten.
        before = {}
        for f in self.files():
            with open(os.path.join(self.dir, f)) as h:
                before[f] = h.read()
        st2 = runner.execute_stage(bl, self.dir, "fp", lambda b, stem: {"analysis": {"valid": True}}, lambda d: None)
        self.assertEqual(st2["status"], "complete")
        self.assertEqual(runner.valid_attempt(self.dir, runner.sequences(bl)[1], "fp"), 8)
        for f, txt in before.items():
            with open(os.path.join(self.dir, f)) as h:
                self.assertEqual(h.read(), txt)

    def test_interrupted_attempt_keeps_evidence_and_gets_a_new_id(self):
        bl = blocks()
        seq = runner.sequences(bl)[0]
        pre = os.path.join(self.dir, "seq-r0-s0.att0")
        runner.write_new(pre + ".start.json", {"attempt": 0})
        runner.write_new(os.path.join(self.dir, f"{seq[0]['block_id']}.att0.json"),
                         {"block": seq[0], "attempt": 0, "fingerprint_sha256": "fp", "analysis": {"valid": True}})
        self.assertIsNone(runner.valid_attempt(self.dir, seq, "fp"))   # no end record: never selected
        st = runner.execute_stage(bl, self.dir, "fp", lambda b, stem: {"analysis": {"valid": True}}, lambda d: None)
        self.assertEqual(st["status"], "complete")
        self.assertEqual(runner.valid_attempt(self.dir, seq, "fp"), 1)
        self.assertIn(f"{seq[0]['block_id']}.att0.json", self.files())

    def test_valid_attempt_requires_matching_records(self):
        bl = blocks()
        runner.execute_stage(bl, self.dir, "fp", lambda b, stem: {"analysis": {"valid": True}}, lambda d: None)
        seq = runner.sequences(bl)[0]
        self.assertEqual(runner.valid_attempt(self.dir, seq, "fp"), 0)
        self.assertIsNone(runner.valid_attempt(self.dir, seq, "other-fingerprint"))

    def fake_session(self, tuning=1):
        d = self.dir
        for name in ("bin", "export", "cap"):
            os.makedirs(os.path.join(d, name), exist_ok=True)
        for rel, text in (("bin/replay", "binary v1"), ("export/manifest.json", "{}"), ("cap/index.json", "{}"),
                          ("validation.json", '{"ok": true}')):
            with open(os.path.join(d, rel), "w") as f:
                f.write(text)
        arm = {"binary": os.path.join(d, "bin", "replay"), "export": os.path.join(d, "export"),
               "captures": os.path.join(d, "cap"), "tuning": {"nb": tuning}}
        return {"session": "t", "seed": 1, "window_s": 12, "clips": [0], "arms": {"B": arm, "C": dict(arm)},
                "stages": {"x": {"arms": ["B", "C"], "replicates": 1, "condition": {"threads": 8, "components": 2},
                                 "about": "free text"}}}

    def test_resume_refuses_changed_configuration_or_build(self):
        """Review c1 finding 7: tuning, binaries, or the design changing under the same stage."""
        out = os.path.join(self.dir, "out")
        os.makedirs(out)
        sess = self.fake_session()
        val = os.path.join(self.dir, "validation.json")
        fp = runner.stage_fingerprint(sess, "x", val)
        runner.ensure_stage(out, fp, blocks())
        runner.ensure_stage(out, runner.stage_fingerprint(sess, "x", val), blocks())  # identical: fine
        sess["stages"]["x"]["about"] = "edited prose"                                    # not part of the identity
        runner.ensure_stage(out, runner.stage_fingerprint(sess, "x", val), blocks())
        changed = self.fake_session(tuning=2)
        with self.assertRaises(SystemExit):
            runner.ensure_stage(out, runner.stage_fingerprint(changed, "x", val), blocks())
        rebuilt = self.fake_session()
        with open(rebuilt["arms"]["B"]["binary"], "w") as f:
            f.write("binary v2")                                                      # same path, new build
        with self.assertRaises(SystemExit):
            runner.ensure_stage(out, runner.stage_fingerprint(rebuilt, "x", val), blocks())
        with self.assertRaises(SystemExit):
            runner.ensure_stage(out, runner.stage_fingerprint(self.fake_session(), "x", val), blocks(n_seq=1))
        self.assertEqual(sorted(os.listdir(out)), ["design.json", "stage.json"])
        self.assertEqual(len(runner.read_json(os.path.join(out, "design.json"))["blocks"]), 4)


class ValidationSelection(unittest.TestCase):
    """Codex code review c4 major 3 (contract change from c1): the NEWEST completed non-plumbing validation
    must pass; a failure revokes permission to time and is never skipped in favour of an older success.
    Plumbing records are never used. Per-stream records (validation-<stamp>-capture.json, ...prespawnNN.json)
    are not validation records."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.old = runner.ROOT
        runner.ROOT = self.root
        self.d = os.path.join(self.root, "results", "s", "validate")
        os.makedirs(self.d)
        self.sess = {"session": "s"}

    def tearDown(self):
        runner.ROOT = self.old
        shutil.rmtree(self.root)

    def put(self, stamp, ok, plumbing=False):
        runner.write_new(os.path.join(self.d, f"validation-{stamp}.json"), {"ok": ok, "plumbing": plumbing})

    def test_plumbing_validation_is_never_used(self):
        self.assertIsNone(runner.latest_ok_validation(self.sess))
        self.put("20261021T010000", True, plumbing=True)
        self.assertIsNone(runner.latest_ok_validation(self.sess))
        self.put("20261021T020000", True)
        self.put("20261021T030000", True, plumbing=True)                 # newer, but plumbing: ignored
        self.assertTrue(runner.latest_ok_validation(self.sess).endswith("validation-20261021T020000.json"))
        runner.write_new(os.path.join(self.d, "validation-20261021T040000-capture.json"), {"analysis": {"ok": False}})
        self.assertTrue(runner.latest_ok_validation(self.sess).endswith("validation-20261021T020000.json"))

    def test_success_then_failure_revokes(self):
        self.put("20261021T010000", True)
        self.put("20261021T020000", False)                                # e.g. instrument width or A2 failed
        self.assertIsNone(runner.latest_ok_validation(self.sess))
        self.put("20261021T030000", True)                                 # a newer success restores it
        self.assertTrue(runner.latest_ok_validation(self.sess).endswith("validation-20261021T030000.json"))


class SessionPaths(unittest.TestCase):
    """No user name in the source: session paths may be "$CPU1BIT_ROOT/..." (the Mac artifacts root) or "~/..."."""

    def test_expand_session(self):
        old = os.environ.get("CPU1BIT_ROOT")
        os.environ["CPU1BIT_ROOT"] = "/Users/mac/root"
        try:
            sess = runner.expand_session({"arms": {"A": {"binary": "$CPU1BIT_ROOT/target/frozen/release/replay-x",
                                                         "export": "~/e", "captures": "/abs/c"}},
                                          "stages": {"p": {"type": "command", "argv": ["$CPU1BIT_ROOT/bin/p", "--qos-ui"]}}})
        finally:
            if old is None:
                os.environ.pop("CPU1BIT_ROOT")
            else:
                os.environ["CPU1BIT_ROOT"] = old
        self.assertEqual(sess["arms"]["A"]["binary"], "/Users/mac/root/target/frozen/release/replay-x")
        self.assertEqual(sess["arms"]["A"]["export"], os.path.expanduser("~/e"))
        self.assertEqual(sess["arms"]["A"]["captures"], "/abs/c")
        self.assertEqual(sess["stages"]["p"]["argv"], ["/Users/mac/root/bin/p", "--qos-ui"])


if __name__ == "__main__":
    unittest.main()
