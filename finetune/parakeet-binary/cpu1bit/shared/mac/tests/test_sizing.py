"""Contract tests for the fixed replay count and its sizing (plans/cpu-benchmark-mac.md M15.2, M16.5; Codex
plan review p6 minor 2). Written before the implementation, pending plan review p7.

Contract (shared/mac/runner.py; constants from guard/runner, not hard-coded here):
  runner.run_sizing(sess, stage, out_dir, validation, log, run=None) -> the sizing record; run(blk, argv, stem,
      replays) -> block record (default: runner.run_block); every attempt is recorded with sizing_accepted
      and sizing_reasons; only accepted blocks give latencies (c4 major 2); no clean block: SystemExit
  runner.sizing_block_ok(record, n_clips) -> (ok, reasons): every block-level and window-level gate
      except the window energy checks ("energy: ..."), which cannot apply to one-replay windows
  runner.size_condition(latency_s, plan) -> {"N": {clip: n}, "feasible": bool, "block_s": {arm: s}, "reasons"}
      latency_s: {arm: {clip: per-replay seconds}} from the sizing stage
      plan: {"idle_s", "warmup_s": {arm: s}, "clip_load_s"}; N = ceil(12 s / (0.95 * fastest latency));
      block_s = idle + warmup + sum over clips of (N*lat + S + 2G + S' + clip load + one untimed replay);
      feasible iff every arm's block_s <= A_MAX - 60 s
  runner.freeze_sizing(session, condition, sizing) -> session with session["sizing"][condition] frozen
      ({"N", "latency_s", "sha256"})
  runner.frozen_n(session, condition) -> {clip: n}; SystemExit if missing or its sha256 does not match
  runner.block_argv(session, block) -> argv including "--replays" with N in the session's clip order
  runner.stage_fingerprint(...) covers the frozen N (a changed N refuses to resume, c1 finding 7)
  runner.short_windows(windows, min_s=12) -> windows shorter than min_s (flagged, never invalid)

    python3 tests/test_sizing.py
"""
from __future__ import annotations

import copy
import math
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import guard  # noqa: E402
import runner  # noqa: E402

CLIPS = [0, 1, 2, 3, 4, 5, 6, 7]
PLAN = {"idle_s": 10.0, "warmup_s": {"FAST": 30.0, "SLOW": 60.0}, "clip_load_s": 0.5}


class Sizing(unittest.TestCase):
    def test_codex_p6_infeasible_condition(self):
        """Fastest replay 100 ms gives N = 127; an arm at 600 ms needs 609.6 s for its eight windows."""
        lat = {"FAST": {c: 0.1 for c in CLIPS}, "SLOW": {c: 0.6 for c in CLIPS}}
        r = runner.size_condition(lat, PLAN)
        self.assertEqual(set(r["N"].values()), {127})
        self.assertFalse(r["feasible"])
        self.assertGreater(r["block_s"]["SLOW"], 609.6)
        self.assertTrue(any("SLOW" in x for x in r["reasons"]), r["reasons"])

    def test_feasible_condition_and_window_target(self):
        lat = {"FAST": {c: 1.0 + 0.1 * c for c in CLIPS}, "SLOW": {c: 2.5 + 0.2 * c for c in CLIPS}}
        r = runner.size_condition(lat, PLAN)
        self.assertTrue(r["feasible"], r["reasons"])
        for c in CLIPS:
            fastest = min(lat[a][c] for a in lat)
            self.assertEqual(r["N"][c], math.ceil(12.0 / (0.95 * fastest)))
            self.assertGreaterEqual(r["N"][c] * fastest, 12.0)              # fastest arm's window >= 12 s
        self.assertLessEqual(max(r["block_s"].values()), guard.A_MAX_NS / 1e9 - 60)


class Persistence(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        b = os.path.join(self.d, "replay")
        with open(b, "w") as f:
            f.write("binary")
        for sub in ("export", "cap"):
            os.makedirs(os.path.join(self.d, sub))
        with open(os.path.join(self.d, "export", "manifest.json"), "w") as f:
            f.write("{}")
        with open(os.path.join(self.d, "cap", "index.json"), "w") as f:
            f.write("{}")
        with open(os.path.join(self.d, "validation.json"), "w") as f:
            f.write('{"ok": true}')
        arm = {"binary": b, "export": os.path.join(self.d, "export"), "captures": os.path.join(self.d, "cap"), "tuning": None}
        cond = {"threads": 8, "components": 2}
        self.sess = {"session": "t", "seed": 1, "window_s": 12, "clips": CLIPS, "arms": {"B": arm, "C": dict(arm)},
                     "stages": {"screen": {"arms": ["B", "C"], "replicates": 1, "condition": cond},
                                "confirm": {"arms": ["B", "C"], "replicates": 6, "condition": cond}}}
        lat = {"B": {c: 1.0 for c in CLIPS}, "C": {c: 0.8 for c in CLIPS}}
        self.sizing = runner.size_condition(lat, dict(PLAN, warmup_s={"B": 30.0, "C": 30.0}))
        self.cond_key = "threads=8,components=2"

    def tearDown(self):
        shutil.rmtree(self.d)

    def test_frozen_n_is_used_by_every_later_stage(self):
        sess = runner.freeze_sizing(self.sess, self.cond_key, self.sizing)
        n = runner.frozen_n(sess, self.cond_key)
        self.assertEqual(n, self.sizing["N"])
        for stage in ("screen", "confirm"):
            blk = runner.blocks_for(sess, stage)[0]
            argv = runner.block_argv(sess, blk)
            self.assertIn("--replays", argv)
            self.assertEqual(argv[argv.index("--replays") + 1], ",".join(str(n[c]) for c in CLIPS))

    def test_plumbing_blocks_use_the_session_plumbing_count(self):
        blk = runner.blocks_for(self.sess, "screen")[0]
        argv = runner.block_argv(self.sess, blk, plumbing=True)
        self.assertEqual(argv[argv.index("--replays") + 1], ",".join("1" for _ in CLIPS))
        argv = runner.block_argv(dict(self.sess, plumbing_replays=10), blk, plumbing=True)
        self.assertEqual(argv[argv.index("--replays") + 1], ",".join("10" for _ in CLIPS))
        self.assertIn("--allow-background", argv)

    def test_missing_or_tampered_sizing_refuses(self):
        with self.assertRaises(SystemExit):
            runner.frozen_n(self.sess, self.cond_key)
        sess = runner.freeze_sizing(self.sess, self.cond_key, self.sizing)
        bad = copy.deepcopy(sess)
        bad["sizing"][self.cond_key]["N"][0] += 1
        with self.assertRaises(SystemExit):
            runner.frozen_n(bad, self.cond_key)

    def test_changed_n_refuses_to_resume(self):
        sess = runner.freeze_sizing(self.sess, self.cond_key, self.sizing)
        out = os.path.join(self.d, "out")
        os.makedirs(out)
        val = os.path.join(self.d, "validation.json")
        runner.ensure_stage(out, runner.stage_fingerprint(sess, "screen", val), runner.blocks_for(sess, "screen"))
        other = dict(self.sizing, N={c: v + 1 for c, v in self.sizing["N"].items()})
        sess2 = runner.freeze_sizing(self.sess, self.cond_key, other)
        with self.assertRaises(SystemExit):
            runner.ensure_stage(out, runner.stage_fingerprint(sess2, "screen", val), runner.blocks_for(sess2, "screen"))


class GuardBand(unittest.TestCase):
    def test_short_windows_are_flagged_not_invalid(self):
        S = 10**9
        windows = [{"window": 0, "t0_ns": 0, "t1_ns": 13 * S, "valid": True},
                   {"window": 1, "t0_ns": 20 * S, "t1_ns": 31 * S, "valid": True}]     # 11 s: faster than sized
        flagged = runner.short_windows(windows, min_s=12)
        self.assertEqual([w["window"] for w in flagged], [1])
        self.assertTrue(all(w["valid"] for w in windows))


def sizing_record(lat_s: float, block_reasons=(), window_reasons=(), exit_code=0, clips=(0, 1)):
    """A sizing block record as runner.run_block returns it (one replay per clip)."""
    S, MS = 10**9, 10**6
    g, sp, spost = guard.G_NS, guard.S_NS, guard.S_POST_NS
    ev = [{"event": "start", "t_ns": 0}, {"event": "loaded", "t_ns": 1 * S}, {"event": "warmup_done", "t_ns": 20 * S},
          {"event": "storage", "t_ns": 20 * S, "ok": True}]
    t, windows = 20 * S, []
    for i, _ in enumerate(clips):
        t0 = t + 300 * MS + int(lat_s * S) + sp + g
        t1 = t0 + int(lat_s * S)
        ev += [{"event": "window_start", "window": i, "t_ns": t0},
               {"event": "window_end", "window": i, "t_ns": t1, "replays": 1}]
        windows.append({"window": i, "t0_ns": t0, "t1_ns": t1, "replays": 1, "valid": not (block_reasons or window_reasons),
                        "invalid_reasons": list(block_reasons) + list(window_reasons)})
        t = t1 + g + spost
    return {"events": ev, "exit_code": exit_code,
            "analysis": {"valid": not (block_reasons or window_reasons), "block_invalid_reasons": list(block_reasons),
                         "windows": windows}}


class SizingGates(unittest.TestCase):
    """Codex code review c4 major 2: sizing latencies come only from blocks that pass the applicable
    host-state, dispatch and storage gates; rejected blocks are retried. Only the energy checks, which
    cannot apply to one-replay windows, are exempt."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        arm = {"binary": "/nonexistent/replay", "export": "x", "captures": "y", "tuning": None}
        self.sess = {"session": "t", "seed": 1, "clips": [0, 1], "arms": {"A": arm, "B": dict(arm)},
                     "stages": {"size": {"type": "sizing", "arms": ["A", "B"], "condition": {"threads": 8, "components": 2}}}}

    def tearDown(self):
        shutil.rmtree(self.d)

    def test_contaminated_sizing_block_is_retried_not_used(self):
        queue = {"A": [sizing_record(0.6, block_reasons=["block span: foreign busy 1.20 (+0.10 tick margin) cores"]),
                       sizing_record(0.4)],
                 "B": [sizing_record(0.5, window_reasons=["energy: work present: no sample inside W exceeds tol"])]}
        calls = []

        def run(blk, argv, stem, replays):
            calls.append((blk["arm"], os.path.basename(stem), replays))
            return queue[blk["arm"]].pop(0)
        doc = runner.run_sizing(self.sess, "size", self.d, {"delta_a2_ns": 0}, lambda x: None, run=run)
        self.assertEqual([c[0] for c in calls], ["A", "A", "B"])
        self.assertTrue(all(c[2] == [1, 1] for c in calls))
        self.assertAlmostEqual(doc["latency_s"]["A"][0], 0.4)
        self.assertAlmostEqual(runner.read_json(os.path.join(self.d, "sizing.json"))["latency_s"]["A"]["0"], 0.4)
        self.assertTrue(os.path.exists(os.path.join(self.d, "size-A.att0.json")))   # the rejected attempt is kept
        self.assertTrue(os.path.exists(os.path.join(self.d, "sizing.json")))
        rej = runner.read_json(os.path.join(self.d, "size-A.att0.json"))
        self.assertFalse(rej["sizing_accepted"])
        self.assertTrue(any("foreign busy" in r for r in rej["sizing_reasons"]), rej["sizing_reasons"])

    def test_every_applicable_gate_rejects(self):
        for bad in (dict(block_reasons=["block span: thermal state not nominal"]),
                    dict(block_reasons=["storage audit failed: ['weight bytes']"]),
                    dict(window_reasons=["dispatch audit: calls != 240 / quantizations != 192 per replay"]),
                    dict(window_reasons=["user input during the interval"]),
                    dict(exit_code=3)):
            ok, reasons = runner.sizing_block_ok(sizing_record(0.4, **bad), 2)
            self.assertFalse(ok, bad)
            self.assertTrue(reasons, bad)
        ok, reasons = runner.sizing_block_ok(sizing_record(0.4, window_reasons=["energy: coverage: P_out not covered"]), 2)
        self.assertTrue(ok, reasons)

    def test_no_clean_block_means_no_sizing(self):
        def run(blk, argv, stem, replays):
            return sizing_record(0.4, block_reasons=["block span: memory pressure level 4"])
        with self.assertRaises(SystemExit):
            runner.run_sizing(self.sess, "size", self.d, {"delta_a2_ns": 0}, lambda x: None, run=run)
        self.assertFalse(os.path.exists(os.path.join(self.d, "sizing.json")))


if __name__ == "__main__":
    unittest.main()
