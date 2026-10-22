"""Contract tests of the one-time test-scoring entry point (gpu1bit/x2/testscore.py; plans/x2-default.md
"One-time test scoring", X6, X8, Y3, Y5, Z1; CPU).

They run the REAL orchestrator, worker and summary code in FAKE mode (GPU1BIT_TESTSCORE_FAKE). Fake mode
has synthetic sets, a stub model and an output root separate from the fixed real one. Real test audio is
never touched; in real mode the output directory is fixed in code. Covered:
  X6 resume and refusals
    - concurrent launch is refused;
    - a kill at each boundary (mid-utterance, between chunks, between sets, during the summary)
      resumes to a complete result, with each utterance exactly once;
    - a resume with a changed identity is refused;
    - a modified per-set file is refused through its hash;
    - summary-only recovery decodes nothing.
  Y5 the attempt lock survives orchestrator death
    - workers refuse to start without the inherited attempt lock;
    - kill the orchestrator while a worker is mid-chunk: reacquisition is refused until the worker
      exits, then the resume is complete with each utterance exactly once.
  Z1 only a structurally valid utterance counts
    - injected decoder exception, NaN encoder output and wrong encoder length are written to the
      failure log, not as rows;
    - a transient failure recovers in a fresh worker; a persistent one stops (Y3);
    - set_done, complete and the report refuse while unresolved failures remain.
  Y3 attempts
    - scored once: complete refuses another score;
    - a stopped attempt refuses a new attempt without replacement evidence; every referenced record is
      opened and bound (Codex x2 c1, finding 2): approval, dev-only reproducer, CLEAN review, a new freeze,
      passing gates and a replacement benchmark equal to the new attempt's own pins, all after the stop;
      template placeholders, another identity and the old records are refused;
    - every attempt is disclosed (status, reason) and the utterances two attempts share are compared.
  qualification (finding 1): `pin` writes per-attempt pins only for a passing qualification (complete and
    accepted benchmark, gates, preflights, Z6/Z8 runtime records bound to the identities); a new attempt
    starts only if the precheck of its pins passes; missing or failing records refuse it.
  recovery (findings 3 and 4): torn JSONL tails are repaired under the attempt lock before any append
    (complete records kept); retry workers continue until every failed utterance has had its one retry.
  hygiene
    - no --limit with test;
    - the real output root is fixed in code;
    - fake mode can never write under it.
  precheck logic (pure, with fixtures): identity equal to the pinned and benchmark identity; benchmark
    complete; scorer pins; gates passing for the same identity under the candidate; preflights and runtime
    records passing, bound and unchanged; pointer state.
  X8 report rows
    - "MB1, runtime x2" and "M1, runtime x2", with per set and seven-set mean, Common Voice apart;
    - runtime minus its own NeMo row, empty-hypothesis rates, and the X8 label;
    - the section is written into results/TEST.md and gpu1bit/RESULTS.md, each with the X8 label,
      replaced (not duplicated) when the report is regenerated, the rest of each file untouched.
  precheck wiring (the real precheck with its inputs stubbed): the plan's items 4 to 6 of
    plans/test-scoring.md (each test manifest's SHA-256 equals M1-test's and b0-pretrained's, the
    normalizer reproduces their saved ref_norm / hyp_norm, the reused arms are complete), the pinned
    benchmark identities, scorer hashes, gate record and the Z2 pointer state; any failing item fails it.

    gpu1bit/shared/python -m gpu1bit.tests.run --cpu gpu1bit.tests.test_x2_testscore
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path

from gpu1bit.shared import env

PY = str(env.SHARED / "python")
ARM = "M1-x2"


class _Fake(unittest.TestCase):
    def setUp(self):
        self.name = f"t{os.getpid()}-{self.id().rsplit('.', 1)[-1][:40]}"
        self.env = {k: v for k, v in os.environ.items() if k not in ("CUBLAS_WORKSPACE_CONFIG",)}
        self.env.update(GPU1BIT_TESTSCORE_FAKE=self.name, GPU1BIT_TESTSCORE_FAKE_IDENTITY="id-A",
                        GPU1BIT_TESTSCORE_CHUNK="7", CUDA_VISIBLE_DEVICES="")
        from gpu1bit.x2 import testscore as T
        self.T = T
        self.root = T.fake_root(self.name)
        self.qualify()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def qualify(self, ident="id-A", bench="R1", gates="G1", attempt=1, edit=None, pin=True, mtime=None):
        """Fake mode's stand-ins for the qualification records (the same file names under the fake root as
        under H/x2 in real mode): a complete accepted benchmark, passing gates, preflights and Z6/Z8 runtime
        records, all bound to `ident`; then `pin` for `attempt` through the entry point."""
        from gpu1bit.shared import seldoc
        b = self.T.base_dir(self.name)
        b.mkdir(parents=True, exist_ok=True)
        docs = {f"bench-{bench}.json": {"complete": True, "problems": [], "identities": {a: ident for a in self.T.ARMS}},
                f"gates-{gates}.json": {"pass": True, "identities": {"g1": ident, "g4": ident},
                                        "selection": {"sha256": env.sha256_file(seldoc.CANDIDATE)}},
                **{f"preflight-{a}.json": {"pass": True, "identity": ident, "lease_bytes": 9 * 2**30} for a in self.T.ARMS},
                f"runtime-recognition-{gates}.json": {"ok": True, "identity": ident, "gate_run": gates},
                f"runtime-fresh-compare-{bench}.json": {"ok": True, "identity": ident, "bench_run": bench},
                **{f"budget-{gates}-{a}.json": {"arm": a, "gate_run": gates, "preflight_identity": ident, "timeout_s": 4321,
                                                "lease_bytes": 7 * 2**30} for a in self.T.ARMS}}
        if edit:
            edit(docs)
        for f, d in docs.items():
            (b / f).write_text(json.dumps(d))
        (b / f"freeze-{bench}.sha256").write_text("stand-in freeze\n")
        if mtime is not None:
            for f in list(docs) + [f"freeze-{bench}.sha256"]:
                os.utime(b / f, (mtime, mtime))
        if pin:
            return self.cmd("pin", "--arm", ARM, "--attempt", str(attempt), "--benchmark-run", bench, "--gates-run", gates,
                            extra={"GPU1BIT_TESTSCORE_FAKE_IDENTITY": ident}, check=False)

    def cmd(self, *args, extra=None, check=True, timeout=300):
        e = dict(self.env, **(extra or {}))
        p = subprocess.run([PY, "-m", "gpu1bit.x2.testscore", *args], env=e, cwd=str(env.BINARY_DIR),
                           capture_output=True, text=True, timeout=timeout)
        if check and p.returncode != 0:
            raise AssertionError(f"{args} rc {p.returncode}\n{p.stdout[-2000:]}\n{p.stderr[-3000:]}")
        return p

    def score(self, **kw):
        return self.cmd("score", "--arm", ARM, **kw)

    def rows(self):
        out = {}
        for d in sorted((self.root / ARM).glob("attempt-*")):
            for f in d.glob("*.jsonl"):
                if f.name in ("failures.jsonl",):
                    continue
                for line in f.read_text().splitlines():
                    if line.strip():
                        r = json.loads(line)
                        out.setdefault(r["id"], []).append(r)
        return out

    def ledger(self):
        p = self.root / ARM / "ledger.jsonl"
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []

    def assert_complete_once(self):
        rows = self.rows()
        want = [r["id"] for s in self.T.test_sets(self.name) for r in self.T.records(s, self.name)]
        self.assertEqual(sorted(rows), sorted(want))
        self.assertTrue(all(len(v) == 1 for v in rows.values()), "each utterance exactly once")
        self.assertEqual(self.ledger()[-1]["event"], "complete")


class Hygiene(_Fake):
    def test_real_root_fixed_and_fake_separate(self):
        self.assertEqual(self.T.REAL_ROOT, env.ARTIFACTS_ROOT / "H" / "testscore")
        self.assertFalse(str(self.root).startswith(str(self.T.REAL_ROOT) + "/"))
        self.assertEqual(self.T.test_sets(None), ["librispeech_clean", "librispeech_other", "ami", "earnings22",
                                                  "gigaspeech", "spgispeech", "voxpopuli", "common_voice"])

    def test_no_limit_with_test(self):
        p = self.cmd("score", "--arm", ARM, "--limit", "5", check=False)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("--limit", p.stdout + p.stderr)


class ResumeAndOnce(_Fake):
    def test_clean_run_then_scored_once(self):
        self.score()
        self.assert_complete_once()
        p = self.cmd("score", "--arm", ARM, check=False)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("scored once", p.stdout + p.stderr)

    def test_kill_at_each_boundary_resumes_exactly_once(self):
        for point in ("mid-utterance", "between-chunks", "between-sets", "during-summary"):
            with self.subTest(point=point):
                shutil.rmtree(self.root, ignore_errors=True)
                self.qualify()
                p = self.cmd("score", "--arm", ARM, extra={"GPU1BIT_TESTSCORE_KILL": point}, check=False)
                self.assertNotEqual(p.returncode, 0, f"{point}: the injected kill must interrupt the run")
                if point == "during-summary":            # every set done: only summary-only recovery (X6)
                    q = self.cmd("score", "--arm", ARM, check=False)
                    self.assertIn("only `summarize`", q.stdout + q.stderr)
                    self.cmd("summarize", "--arm", ARM)
                else:
                    self.score()
                self.assert_complete_once()

    def test_changed_identity_is_refused(self):
        self.cmd("score", "--arm", ARM, extra={"GPU1BIT_TESTSCORE_KILL": "between-sets"}, check=False)
        p = self.cmd("score", "--arm", ARM, extra={"GPU1BIT_TESTSCORE_FAKE_IDENTITY": "id-B"}, check=False)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("identity", p.stdout + p.stderr)

    def test_modified_set_file_is_refused(self):
        self.cmd("score", "--arm", ARM, extra={"GPU1BIT_TESTSCORE_KILL": "during-summary"}, check=False)
        per_set = sorted((self.root / ARM).glob("attempt-1/*.json"))
        target = [p for p in per_set if p.name != "summary.json"][0]
        target.write_text(target.read_text().replace('"set"', '"set" ', 1))
        p = self.cmd("summarize", "--arm", ARM, check=False)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("hash", p.stdout + p.stderr)

    def test_summary_only_recovery_decodes_nothing(self):
        self.cmd("score", "--arm", ARM, extra={"GPU1BIT_TESTSCORE_KILL": "during-summary"}, check=False)
        before = self.rows()
        p = self.cmd("summarize", "--arm", ARM)
        self.assertIn("decoded 0", p.stdout)
        self.assertEqual(self.rows(), before)
        self.assertEqual(self.ledger()[-1]["event"], "complete")
        p = self.cmd("summarize", "--arm", ARM)                 # idempotent
        self.assertEqual(sum(e["event"] == "complete" for e in self.ledger()), 1)


class Lock(_Fake):
    def test_concurrent_launch_is_refused(self):
        slow = dict(self.env, GPU1BIT_TESTSCORE_SLEEP="0.2")
        first = subprocess.Popen([PY, "-m", "gpu1bit.x2.testscore", "score", "--arm", ARM], env=slow,
                                 cwd=str(env.BINARY_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self._wait_for_worker()
            p = self.cmd("score", "--arm", ARM, check=False)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("attempt lock", p.stdout + p.stderr)
        finally:
            first.wait(timeout=600)

    def test_worker_refuses_without_the_lock(self):
        p = self.cmd("worker", "--arm", ARM, "--attempt", "1", "--set", "fake_a", check=False)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("attempt lock", p.stdout + p.stderr)

    def test_orchestrator_death_keeps_the_lock_until_the_worker_exits(self):
        slow = dict(self.env, GPU1BIT_TESTSCORE_SLEEP="0.15")
        orch = subprocess.Popen([PY, "-m", "gpu1bit.x2.testscore", "score", "--arm", ARM], env=slow,
                                cwd=str(env.BINARY_DIR), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                start_new_session=False)
        worker_pid = self._wait_for_worker()
        orch.kill()                                              # SIGKILL: the orchestrator dies mid-chunk
        orch.wait()
        self.assertTrue(self.T.pid_alive(worker_pid), "the worker is still running")
        p = self.cmd("score", "--arm", ARM, check=False)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("attempt lock", p.stdout + p.stderr)
        deadline = time.time() + 300
        while self.T.pid_alive(worker_pid) and time.time() < deadline:
            time.sleep(0.2)
        self.score()
        self.assert_complete_once()

    def _wait_for_worker(self):
        p = self.root / ARM / "workers.jsonl"
        deadline = time.time() + 120
        while time.time() < deadline:
            if p.exists():
                lines = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
                if lines:
                    return lines[-1]["pid"]
            time.sleep(0.1)
        raise AssertionError("no worker started")


class Failures(_Fake):
    def inject(self, spec):
        return {"GPU1BIT_TESTSCORE_INJECT": json.dumps(spec)}

    def test_injected_failures_are_logged_not_rows(self):
        ids = [r["id"] for r in self.T.records("fake_a", self.name)]
        spec = {"fail": {ids[1]: "decode", ids[2]: "nan", ids[3]: "length"}, "times": {ids[1]: 1, ids[2]: 1, ids[3]: 1}}
        self.score(extra=self.inject(spec))
        self.assert_complete_once()
        fails = [json.loads(l) for l in (self.root / ARM / "attempt-1" / "failures.jsonl").read_text().splitlines()]
        self.assertEqual({f["kind"] for f in fails}, {"decode_error", "not_finite", "length"})
        self.assertEqual({f["id"] for f in fails}, set(ids[1:4]))
        self.assertTrue(all(f["identity"] for f in fails))

    def test_persistent_failure_stops_and_blocks_completion(self):
        ids = [r["id"] for r in self.T.records("fake_b", self.name)]
        p = self.score(extra=self.inject({"fail": {ids[0]: "nan"}, "times": {ids[0]: 99}}), check=False)
        self.assertNotEqual(p.returncode, 0)
        self.assertEqual(self.ledger()[-1]["event"], "stopped")
        self.assertNotIn("fake_b", [e.get("set") for e in self.ledger() if e["event"] == "set_done"])
        for args in (("summarize", "--arm", ARM), ("score", "--arm", ARM)):
            q = self.cmd(*args, check=False)
            self.assertNotEqual(q.returncode, 0, args)
        with self.assertRaisesRegex(SystemExit, "unresolved"):
            self.T.set_done(ARM, 1, "fake_b", self.name)
        q = self.cmd("report", check=False)                       # Z1: the report refuses too
        self.assertNotEqual(q.returncode, 0)
        self.assertRegex(q.stdout + q.stderr, "report refused")
        with self.assertRaisesRegex(SystemExit, "unresolved"):
            self.T.collect(self.name, arms=[ARM])

    def test_new_attempt_needs_bound_evidence(self):
        """Y3 and Codex x2 c1 finding 2: a new attempt needs every replacement record opened and bound: the user's
        approval for this arm and identity, a dev-only reproducer of a stopped attempt, a CLEAN review, a new
        freeze, passing gates and a replacement benchmark, the last three equal to the new attempt's own pins
        (the old pins cannot pass a legitimate fix) and all written after the stop."""
        ids = [r["id"] for r in self.T.records("fake_a", self.name)]
        self.score(extra=self.inject({"fail": {ids[0]: "decode"}, "times": {ids[0]: 99}}), check=False)
        stop = [e for e in self.ledger() if e["event"] == "stopped"][-1]
        import calendar
        after = calendar.timegm(time.strptime(stop["utc"], "%Y-%m-%dT%H:%M:%SZ")) + 30
        p = self.cmd("score", "--arm", ARM, "--new-attempt", "/nonexistent.json", check=False)
        self.assertNotEqual(p.returncode, 0)
        ev = self.root / "evidence.json"
        ev.write_text(json.dumps(self.T.evidence_template(ARM, "id-B")))       # placeholders only
        p = self.cmd("score", "--arm", ARM, "--new-attempt", str(ev), extra={"GPU1BIT_TESTSCORE_FAKE_IDENTITY": "id-B"},
                     check=False)
        self.assertNotEqual(p.returncode, 0, "template placeholders must not pass (finding 2)")
        self.assertNotIn("started", [e["event"] for e in self.ledger() if e["attempt"] == 2])
        # the replacement qualification under the new identity, pinned for attempt 2
        self.assertEqual(self.qualify(ident="id-B", bench="R2", gates="G2", attempt=2, mtime=after).returncode, 0)
        d = self.root / "evidence"
        d.mkdir()
        recs = {"user_approval": {"kind": "user_approval", "arm": ARM, "identity": "id-B", "approved": True},
                "dev_reproducer": {"kind": "dev_reproducer", "arm": ARM, "stopped_attempt": 1,
                                   "sets": ["librispeech_dev_clean"], "result": "reproduced"}}
        for k, v in recs.items():
            (d / f"{k}.json").write_text(json.dumps(v))
        (d / "review.md").write_text("1. findings fixed\n\nCLEAN (no blocker or major)\n")   # the requested review format
        for f in d.iterdir():
            os.utime(f, (after, after))
        b = self.T.base_dir(self.name)
        good = {"arm": ARM, "identity": "id-B", "user_approval": str(d / "user_approval.json"),
                "dev_reproducer": str(d / "dev_reproducer.json"), "review_clean": str(d / "review.md"),
                "freeze": str(b / "freeze-R2.sha256"), "gates_record": str(b / "gates-G2.json"), "benchmark": "R2"}
        bad = {"another identity": {"identity": "id-OTHER"},
               "approval for another identity": "approval",
               "reproducer on test data": "reproducer",
               "review not CLEAN": "review",
               "old benchmark": {"benchmark": "R1"},
               "old gates record": {"gates_record": str(b / "gates-G1.json")},
               "old freeze": {"freeze": str(b / "freeze-R1.sha256")}}
        for name, change in bad.items():
            with self.subTest(case=name):
                doc = dict(good)
                saved = {f: f.read_bytes() for f in d.iterdir()}
                if isinstance(change, dict):
                    doc.update(change)
                elif change == "approval":
                    (d / "user_approval.json").write_text(json.dumps({**recs["user_approval"], "identity": "id-A"}))
                elif change == "reproducer":
                    (d / "dev_reproducer.json").write_text(json.dumps({**recs["dev_reproducer"], "sets": ["librispeech_clean"]}))
                else:
                    (d / "review.md").write_text("1. findings\n\nNOT CLEAN\n")
                for f in d.iterdir():
                    os.utime(f, (after, after))
                ev.write_text(json.dumps(doc))
                p = self.cmd("score", "--arm", ARM, "--new-attempt", str(ev), extra={"GPU1BIT_TESTSCORE_FAKE_IDENTITY": "id-B"},
                             check=False)
                self.assertNotEqual(p.returncode, 0, name)
                self.assertNotIn("started", [e["event"] for e in self.ledger() if e["attempt"] == 2], name)
                for f, data in saved.items():
                    f.write_bytes(data)
                    os.utime(f, (after, after))
        ev.write_text(json.dumps(good))
        p = self.cmd("score", "--arm", ARM, "--new-attempt", str(ev), extra={"GPU1BIT_TESTSCORE_FAKE_IDENTITY": "id-B"})
        self.assertEqual(self.ledger()[-1]["event"], "complete")
        self.assertEqual(max(e["attempt"] for e in self.ledger()), 2)
        self.assertTrue((self.root / ARM / "attempt-1").exists(), "the stopped attempt's rows are kept")
        d2 = self.T.attempts_disclosure(ARM, self.name)            # Y3: every attempt reported, shared rows compared
        self.assertEqual([a["status"] for a in d2["attempts"]], ["stopped", "complete"])
        self.assertGreater(d2["shared"]["utterances"], 0)
        self.assertIn("differing", d2["shared"])

    def test_retry_reaches_every_failed_utterance(self):
        """Finding 4: retry workers continue (bounded chunks) until every failed utterance has had its one retry;
        only an actual second failure stops. Eight transient failures with a 7-utterance chunk complete."""
        ids = [r["id"] for r in self.T.records("fake_a", self.name)][:8]
        self.score(extra=self.inject({"fail": {u: "decode" for u in ids}, "times": {u: 1 for u in ids}}))
        self.assert_complete_once()
        fails = [json.loads(l) for l in (self.root / ARM / "attempt-1" / "failures.jsonl").read_text().splitlines()]
        self.assertEqual(sorted(f["id"] for f in fails), sorted(ids), "each failed once and recovered on its retry")


class TornTails(_Fake):
    """Finding 3: an interrupted append leaves an incomplete last line. Under the attempt lock, before anything is
    appended, the tail is repaired: a complete record without its newline is kept (terminated), a partial one is
    dropped; every complete record is preserved and every file stays valid JSON Lines after resuming."""

    def strict(self):
        for f in sorted((self.root / ARM).rglob("*.jsonl")):
            data = f.read_text()
            self.assertTrue(data == "" or data.endswith("\n"), f.name)
            for line in data.splitlines():
                json.loads(line)                                  # raises on a malformed line

    def test_repair_unit(self):
        f = self.root / "unit.jsonl"
        f.write_text('{"a": 1}\n{"b": 2}\n{"c": ')
        self.T.repair_tail(f)
        self.assertEqual(f.read_text(), '{"a": 1}\n{"b": 2}\n')
        f.write_text('{"a": 1}\n{"b": 2}')
        self.T.repair_tail(f)
        self.assertEqual(f.read_text(), '{"a": 1}\n{"b": 2}\n')
        self.T.append_jsonl(f, {"d": 4})
        self.assertEqual([json.loads(l) for l in f.read_text().splitlines()], [{"a": 1}, {"b": 2}, {"d": 4}])
        f.write_text('{"a": 1}\n{"x"')
        self.T.append_jsonl(f, {"d": 4})
        self.assertEqual([json.loads(l) for l in f.read_text().splitlines()], [{"a": 1}, {"d": 4}])

    def test_resume_after_torn_tails(self):
        p = self.cmd("score", "--arm", ARM, extra={"GPU1BIT_TESTSCORE_KILL": "between-chunks"}, check=False)
        self.assertNotEqual(p.returncode, 0)
        rows = self.root / ARM / "attempt-1" / "fake_a.jsonl"
        last = json.loads(rows.read_text().splitlines()[-1])
        done = {json.loads(l)["id"] for l in rows.read_text().splitlines()}
        nxt = next(r["id"] for r in self.T.records("fake_a", self.name) if r["id"] not in done)
        with open(rows, "a") as f:                                # a complete row whose newline never landed
            f.write(json.dumps({**last, "id": nxt}))
        for name in ("ledger.jsonl", "workers.jsonl"):
            with open(self.root / ARM / name, "a") as f:          # a torn, partial record
                f.write('{"event": "set_do')
        self.score()
        self.assert_complete_once()
        self.strict()


class Precheck(unittest.TestCase):
    """The decidable precheck items (pure; Codex x2 c1, findings 1 and 2): before an attempt starts, the arm's
    identity equals its pinned and benchmark identity; the benchmark (both phases) is complete and accepted;
    the scorer hashes, gates, preflights and the Z6/Z8 runtime records pass, are bound to the current
    identities and runs, and are unchanged since pinned; the pointer is in a Z2 state."""

    def setUp(self):
        from gpu1bit.x2 import testscore as T
        self.T = T
        ids = {"MB1-B-hybrid-x2": "i1", "M1-x2": "i4"}
        self.ok = {"arm": "M1-x2", "identities": dict(ids),
                   "pins": {"arm": "M1-x2", "benchmark_run": "R", "gates_run": "G", "identities": dict(ids),
                            "scorer": {"evaluate.py": "s1"}, "candidate_sha256": "c"},
                   "bench": {"complete": True, "problems": [], "identities": dict(ids)},
                   "docs": {"gates": {"pass": True, "identities": {"g1": "i1", "g4": "i4"}, "selection": {"sha256": "c"}},
                            "preflight:MB1-B-hybrid-x2": {"pass": True, "identity": "i1"},
                            "preflight:M1-x2": {"pass": True, "identity": "i4"},
                            "recognition": {"ok": True, "identity": "i1", "gate_run": "G"},
                            "memory": {"ok": True, "identity": "i1", "bench_run": "R"},
                            "budget:MB1-B-hybrid-x2": {"gate_run": "G", "preflight_identity": "i1", "timeout_s": 100},
                            "budget:M1-x2": {"gate_run": "G", "preflight_identity": "i4", "timeout_s": 100}},
                   "unchanged": {"gates": True, "preflight:MB1-B-hybrid-x2": True, "preflight:M1-x2": True,
                                 "recognition": True, "memory": True, "budget:MB1-B-hybrid-x2": True, "budget:M1-x2": True},
                   "scorer_now": {"evaluate.py": "s1"}, "pointer": "x3"}

    def checks(self, **over):
        import copy
        kw = copy.deepcopy(self.ok)
        for k, v in over.items():
            if callable(v):
                v(kw)
            else:
                kw[k] = v
        return self.T.qualification_checks(**kw)

    def test_all_pass(self):
        c = self.checks()
        self.assertTrue(all(x["ok"] for x in c), [x for x in c if not x["ok"]])

    def test_each_failure(self):
        def doc(key, **kv):
            return lambda kw: kw["docs"][key].update(kv)
        cases = {
            "identity differs from the pinned benchmark identity": {"identities": {"MB1-B-hybrid-x2": "i1", "M1-x2": "i9"}},
            "scorer changed": {"scorer_now": {"evaluate.py": "s2"}},
            "gates failed": doc("gates", **{"pass": False}),
            "gates of another identity": doc("gates", identities={"g1": "i1", "g4": "i5"}),
            "gates under another selection": doc("gates", selection={"sha256": "x"}),
            "pointer third state": {"pointer": None},
            "benchmark incomplete or unaccepted": {"bench": {"complete": False, "problems": ["no accepted sidecar"], "identities": {}}},
            "benchmark of another identity": lambda kw: kw["bench"]["identities"].update({"M1-x2": "i9"}),
            "preflight missing": lambda kw: kw["docs"].update({"preflight:M1-x2": None}),
            "preflight failed": doc("preflight:M1-x2", **{"pass": False}),
            "preflight of another identity": doc("preflight:M1-x2", identity="i9"),
            "Z6 recognition failed": doc("recognition", ok=False),
            "Z6 recognition of another identity": doc("recognition", identity="i9"),
            "Z6 recognition of another gate run": doc("recognition", gate_run="G9"),
            "Z8 memory missing": lambda kw: kw["docs"].update({"memory": None}),
            "Z8 memory failed": doc("memory", ok=False),
            "Z8 memory of another identity": doc("memory", identity="i9"),
            "Z8 memory of another benchmark": doc("memory", bench_run="R9"),
            "a record changed since pinned": lambda kw: kw["unchanged"].update({"memory": False}),
            "duration budget missing": lambda kw: kw["docs"].update({"budget:M1-x2": None}),
            "duration budget of another identity": doc("budget:M1-x2", preflight_identity="i9"),
            "duration budget of another gate run": doc("budget:M1-x2", gate_run="G9"),
            "duration budget without a timeout": doc("budget:M1-x2", timeout_s=0),
            "duration budget changed since pinned": lambda kw: kw["unchanged"].update({"budget:M1-x2": False}),
            "pins of another arm": lambda kw: kw["pins"].update({"arm": "MB1-B-hybrid-x2"}),
        }
        for name, over in cases.items():
            with self.subTest(case=name):
                c = self.checks(_=over) if callable(over) else self.checks(**over)
                self.assertFalse(all(x["ok"] for x in c), name)

    def test_no_pins_fails(self):
        c = self.T.qualification_checks(arm="M1-x2", identities={"MB1-B-hybrid-x2": "i1", "M1-x2": "i4"}, pins=None,
                                        bench={}, docs={}, unchanged={}, scorer_now={}, pointer="x3")
        self.assertFalse(all(x["ok"] for x in c))


class Qualification(_Fake):
    """Finding 1 through the entry point: `pin` writes an attempt's pins only for a passing qualification, and a
    new attempt starts only if the precheck of its pins passes; the benchmark comes from the benchmark
    qualification (complete, accepted), never from a single chunk."""

    def started(self):
        return [e for e in self.ledger() if e["event"] == "started"]

    def test_missing_or_failing_records_refuse_the_attempt(self):
        def unset(key, **kv):
            return lambda docs: docs[next(k for k in docs if k.startswith(key))].update(kv)
        cases = {"no pins": None,
                 "benchmark incomplete": unset("bench-", complete=False),
                 "preflight failed": unset(f"preflight-{ARM}", **{"pass": False}),
                 "Z8 memory failed": unset("runtime-fresh-compare-", ok=False),
                 "Z6 recognition of another identity": unset("runtime-recognition-", identity="id-X"),
                 "gates failed": unset("gates-", **{"pass": False})}
        for name, edit in cases.items():
            with self.subTest(case=name):
                shutil.rmtree(self.root, ignore_errors=True)
                if edit is None:
                    self.qualify(pin=False)
                else:
                    self.assertEqual(self.qualify().returncode, 0)          # pinned while passing ...
                    self.qualify(edit=edit, pin=False)                       # ... then a record fails
                p = self.cmd("score", "--arm", ARM, check=False)
                self.assertNotEqual(p.returncode, 0, name)
                self.assertIn("precheck", p.stdout + p.stderr)
                self.assertEqual(self.started(), [], name)

    def test_pin_refuses_a_failing_qualification(self):
        shutil.rmtree(self.root, ignore_errors=True)
        p = self.qualify(edit=lambda docs: docs[f"preflight-{ARM}.json"].update({"pass": False}))
        self.assertNotEqual(p.returncode, 0)
        self.assertFalse((self.root / ARM / "pins-1.json").exists())
        self.assertEqual(self.qualify().returncode, 0)
        self.assertNotEqual(self.qualify().returncode, 0, "pins are written once per attempt")

    def test_job_arguments_come_from_the_pinned_records(self):
        """Codex x2 c3 minor: the scoring job's timeout is the pinned duration budget's and its lease the pinned
        preflight's; a budget changed after pinning is refused."""
        p = self.cmd("job-args", "--arm", ARM, "--attempt", "1")
        got = json.loads(p.stdout.strip().splitlines()[-1])
        self.assertEqual(got["timeout_s"], 4321)
        self.assertIn("--timeout 4321s", got["command"])
        self.assertIn("--needs gpu:shared", got["command"])
        self.assertIn("--vram 9G", got["command"], "the lease is the pinned preflight's, not the budget's copy (c4)")
        self.assertTrue(got["command"].endswith(f"score --arm {ARM}"))
        b = self.T.base_dir(self.name) / f"budget-G1-{ARM}.json"
        b.write_text(json.dumps({**json.loads(b.read_text()), "timeout_s": 99}))
        q = self.cmd("job-args", "--arm", ARM, "--attempt", "1", check=False)
        self.assertNotEqual(q.returncode, 0)
        self.assertIn("changed", q.stdout + q.stderr)

    def test_job_arguments_for_an_exceptional_retry_and_a_resume(self):
        """c4 minor and c5 minor: the helper forwards --new-attempt <evidence> whenever it is supplied and never
        requires it; score() decides from the ledger. A new attempt after a stop needs the evidence, while an
        open attempt is resumed without it, and score refuses evidence for an open attempt."""
        self.assertEqual(self.qualify(ident="id-A", bench="R2", gates="G2", attempt=2).returncode, 0)
        p = self.cmd("job-args", "--arm", ARM, "--attempt", "2", "--new-attempt", "/x/evidence.json")
        got = json.loads(p.stdout.strip().splitlines()[-1])
        self.assertTrue(got["command"].endswith(f"score --arm {ARM} --new-attempt /x/evidence.json"))
        p = self.cmd("job-args", "--arm", ARM, "--attempt", "2")                 # resuming an open attempt 2
        got = json.loads(p.stdout.strip().splitlines()[-1])
        self.assertTrue(got["command"].endswith(f"score --arm {ARM}"))

    def test_open_attempt_resumes_without_evidence(self):
        """The resume path the helper's plain command runs: an interrupted open attempt resumes with
        `score --arm A` and completes; `--new-attempt` evidence for an open attempt is refused by score()."""
        p = self.cmd("score", "--arm", ARM, extra={"GPU1BIT_TESTSCORE_KILL": "between-sets"}, check=False)
        self.assertNotEqual(p.returncode, 0)
        q = self.cmd("score", "--arm", ARM, "--new-attempt", "/x/evidence.json", check=False)
        self.assertNotEqual(q.returncode, 0)
        self.assertIn("resume it", q.stdout + q.stderr)
        self.score()
        self.assertEqual(self.ledger()[-1]["event"], "complete")
        self.assertEqual(max(e["attempt"] for e in self.ledger()), 1)

    def test_real_benchmark_qualification_reads_the_session(self):
        q = self.T.bench_qualification(f"no-such-run-{os.getpid()}", None)
        self.assertFalse(q["complete"])
        self.assertTrue(q["problems"])


class ReportRows(unittest.TestCase):
    def test_rows_and_label(self):
        from gpu1bit.x2 import testscore as T
        sets = T.TEST_COLUMNS
        nemo = {"MB1": {s: 5.0 for s in sets}, "M1": {s: 4.0 for s in sets}}
        rt = {"MB1-B-hybrid-x2": {s: 5.1 for s in sets}, "M1-x2": {s: 4.05 for s in sets}}
        empty = {"MB1-B-hybrid-x2": {s: 0.01 for s in sets}, "M1-x2": {s: 0.0 for s in sets}}
        md = T.report_markdown(rt, nemo, empty)
        self.assertIn("| MB1, runtime x2 |", md)
        self.assertIn("| M1, runtime x2 |", md)
        self.assertIn("MB1, runtime x2 - MB1 NeMo FP32 (points)", md)
        self.assertIn("+0.10", md)
        self.assertIn("+0.05", md)
        for phrase in ("INT8x2 activations", "integer and fused arithmetic", "batch 1", "batch 64",
                       "full CUDA-graph", "not an x2-versus-x3 test comparison"):
            self.assertIn(phrase, md)
        self.assertIn("common_voice", md.splitlines()[[i for i, l in enumerate(md.splitlines()) if l.startswith("| Arm")][0]])
        self.assertIn("Empty hypotheses", md)


class ReportFiles(unittest.TestCase):
    def test_sections_written_replaced_and_labelled(self):
        import tempfile
        from gpu1bit.x2 import testscore as T
        d = Path(tempfile.mkdtemp(dir=str(env.ARTIFACTS_ROOT / "H")))
        try:
            test_md, results_md = d / "TEST.md", d / "RESULTS.md"
            test_md.write_text("# Test\n\nFP32 rows\n")
            results_md.write_text("# Results\n\n## Earlier\n\nkept\n")
            sets = T.TEST_COLUMNS
            md = T.report_markdown({a: {s: 5.0 for s in sets} for a in T.ARMS}, {"MB1": {s: 5.0 for s in sets},
                                   "M1": {s: 4.0 for s in sets}}, {a: {s: 0.0 for s in sets} for a in T.ARMS})
            for _ in range(2):
                T.write_sections(md, test_md=test_md, results_md=results_md)
            for f, keep in ((test_md, "FP32 rows"), (results_md, "## Earlier")):
                text = f.read_text()
                self.assertIn(keep, text)
                self.assertIn(T.X8_LABEL, text)
                self.assertEqual(text.count("| MB1, runtime x2 |"), md.count("| MB1, runtime x2 |"), f.name)
            self.assertIn(T.RESULTS_SECTION, results_md.read_text())
        finally:
            shutil.rmtree(d, ignore_errors=True)


class PrecheckWiring(unittest.TestCase):
    """The real precheck (real mode) with its inputs stubbed: the plan's items 4 to 6 of plans/test-scoring.md
    (test manifest SHA-256 = M1-test's and b0-pretrained's, the normalizer reproduces their saved ref_norm /
    hyp_norm, the reused arms complete), the qualification of the attempt's own pins and the Z2 pointer state;
    any failing item fails it."""

    def setUp(self):
        import tempfile
        import binary_test
        from gpu1bit.shared import seldoc, selection
        from gpu1bit.x2 import testscore as T
        self.T, self.bt, self.sel = T, binary_test, selection
        self.d = Path(tempfile.mkdtemp(dir=str(env.ARTIFACTS_ROOT / "H")))
        base = self.d / "x2"
        base.mkdir()
        ids = {"MB1-B-hybrid-x2": "i1", "M1-x2": "i4"}
        cand = env.sha256_file(seldoc.CANDIDATE)
        docs = {"gates-G.json": {"pass": True, "identities": {"g1": "i1", "g4": "i4"}, "selection": {"sha256": cand}},
                "preflight-MB1-B-hybrid-x2.json": {"pass": True, "identity": "i1"},
                "preflight-M1-x2.json": {"pass": True, "identity": "i4"},
                "runtime-recognition-G.json": {"ok": True, "identity": "i1", "gate_run": "G"},
                "runtime-fresh-compare-B.json": {"ok": True, "identity": "i1", "bench_run": "B"},
                "budget-G-MB1-B-hybrid-x2.json": {"gate_run": "G", "preflight_identity": "i1", "timeout_s": 100},
                "budget-G-M1-x2.json": {"gate_run": "G", "preflight_identity": "i4", "timeout_s": 100}}
        for f, doc in docs.items():
            (base / f).write_text(json.dumps(doc))
        self.saved = (T.REAL_ROOT, T.base_dir, T.identity, T.scorer_now, T.bench_qualification, T.freeze_file,
                      binary_test._baseline_checks, selection.verify_freeze)
        T.REAL_ROOT = self.d / "testscore"
        T.base_dir = lambda name=None: base
        T.identity = lambda arm, name=None: ids[arm]
        T.scorer_now = lambda: {"s": "1"}
        T.bench_qualification = lambda run, name=None: {"complete": True, "problems": [], "identities": dict(ids)}
        T.freeze_file = lambda run, name=None: base / "gates-G.json"
        selection.verify_freeze = lambda run: {"ok": True, "pointer": "x3"}
        self.items = {4: True, 5: True, 6: True}

        def baseline(c4, c5, c6):
            for item, c in ((4, c4), (5, c5), (6, c6)):
                c(f"stub item {item}", self.items[item])
        binary_test._baseline_checks = baseline
        T.pin("M1-x2", 1, "B", "G")

    def tearDown(self):
        (self.T.REAL_ROOT, self.T.base_dir, self.T.identity, self.T.scorer_now, self.T.bench_qualification, self.T.freeze_file,
         self.bt._baseline_checks, self.sel.verify_freeze) = self.saved
        shutil.rmtree(self.d, ignore_errors=True)

    def test_all_items_and_failures(self):
        r = self.T.precheck("M1-x2", 1)
        self.assertTrue(r["pass"], [c for c in r["checks"] if not c["ok"]])
        names = [c["check"] for c in r["checks"]]
        for item in (4, 5, 6):
            self.assertTrue(any(n.startswith(f"item {item}:") for n in names), item)
        self.assertTrue(any("pointer" in n for n in names))
        for item in (4, 5, 6):
            with self.subTest(item=item):
                self.items[item] = False
                self.assertFalse(self.T.precheck("M1-x2", 1)["pass"])
                self.items[item] = True

        def third(run):
            raise SystemExit("third state")
        self.sel.verify_freeze = third
        self.assertFalse(self.T.precheck("M1-x2", 1)["pass"])
        self.sel.verify_freeze = lambda run: {"ok": True, "pointer": "x3"}
        self.assertFalse(self.T.precheck("M1-x2", 2)["pass"], "attempt 2 has no pins and no evidence")


class ReviewVerdict(unittest.TestCase):
    """Codex x2 c2 minor: a review's verdict (its last non-empty line) is CLEAN in either supported form,
    `CLEAN` or `CLEAN (no blocker or major)`; `NOT CLEAN` in any form, or anything else, is not."""

    def test_forms(self):
        from gpu1bit.x2 import testscore as T
        for text, want in (("findings\n\nCLEAN\n", True), ("findings\n\nCLEAN (no blocker or major)\n  \n", True),
                           ("NOT CLEAN\n", False), ("findings\nNOT CLEAN (no blocker or major)\n", False),
                           ("CLEAN\nlater text\n", False), ("", False), ("CLEANISH\n", False),
                           ("CLEAN (one major)\n", False)):
            with self.subTest(text=text):
                self.assertEqual(T.review_is_clean(text), want)
        review = Path("/mnt/hd/wilderness-labs-stt/parakeet-binary/review/x2-default")
        if (review / "c2.md").exists():
            self.assertTrue(T.review_is_clean((review / "c2.md").read_text()))
            self.assertFalse(T.review_is_clean((review / "c1.md").read_text()))
