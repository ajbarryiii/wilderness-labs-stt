"""One-time test scoring of the INT8x2 defaults through their runtimes (plans/x2-default.md "One-time test
scoring", X6, X8, Y3, Y5, Z1, Y6; hygiene of plans/test-scoring.md). A separate entry point; the
production output directory is fixed in code (REAL_ROOT).

ARMS: MB1-B-hybrid-x2 (MB1 on B-hybrid-x2) and M1-x2 (M1 on R-expanded x2), built exactly as the gate and
benchmark arms (gpu1bit.actprec.parts.build) under the candidate selection document. All 8 test sets,
every utterance, batch 1. No --limit.

ATTEMPTS (append-only ledger per arm; <root>/<arm>/ledger.jsonl):
  started {attempt, identity}, set_done {set, sha256 of the per-set file}, complete {summary sha256},
  stopped {evidence}.
- An arm is scored once; after `complete` every further score is refused.
- Same-attempt resume: workers resume from the first unfinished utterance under the same attempt and
  identity. A resume with another identity is refused.
- After every set is done, only summary-only recovery (`summarize`, zero decoding) is allowed. It checks
  the per-set files against their set_done hashes.
- An attempt that cannot resume is `stopped` with evidence (automatically for a persistent utterance
  failure, by hand with `record-failure`). Its rows stay in place, read-only, and scoring stops (Y3).
  A new attempt needs `--new-attempt EVIDENCE.json` bound to the new identity: user approval, a
  dev-only reproducer, a CLEAN review, a new freeze, gates and replacement benchmark (checked by the
  precheck in real mode).

QUALIFICATION (Codex x2 c1, findings 1 and 2): an attempt starts only if the precheck of its OWN pins
(<root>/<arm>/pins-<attempt>.json, written once by `pin`, only for a passing qualification) passes: the
recorded benchmark complete with every step an accepted measurement (both phases; bench_session), the gates
passing under the candidate, the arm's Y6/Z3 preflight and the Z6/Z8 runtime records passing, all bound to the
current identities and runs and unchanged since pinned; the scorer hashes; the Z2 pointer; and in real mode
items 4 to 6 of plans/test-scoring.md (test manifests = M1-test's and b0's, normalizer, reused arms). An
exceptional retry's evidence is opened record by record (evidence_checks) and must point to the new attempt's
own pins.
RECOVERY (findings 3 and 4): torn JSONL tails are repaired under the attempt lock before any append; retry
workers continue until every failed utterance has had its one retry.

ATTEMPT LOCK (X6, Y5): a flock on <root>/<arm>/attempt.lock, taken by the orchestrator and passed to every
worker as an inherited open file description (pass_fds, GPU1BIT_ATTEMPT_LOCK_FD). The lock holds while any
holder lives. A new orchestrator or recovery command needs the lock and refuses while any recorded worker
(pid and start time) is still running. A worker refuses to start without the inherited lock.

UTTERANCES (Z1): an utterance is completed only if decoding returned without an exception, the encoder
output is finite, and the encoder length equals the frontend's expected length. A failure goes to the
attempt's failures.jsonl (exception, kind, finiteness, identity), not to the rows. Each failed utterance is
retried once in a fresh worker. A second failure is unrecoverable: the attempt is stopped (Y3) and the user
is told. set_done, complete and the report refuse while unresolved failures exist.

FAKE MODE (contract tests only): GPU1BIT_TESTSCORE_FAKE=<name> uses synthetic sets, a stub model and the
separate root <H>/testscore-fake/<name>. Injection variables: GPU1BIT_TESTSCORE_INJECT, _KILL, _SLEEP,
_CHUNK, _FAKE_IDENTITY.

    gpu1bit/shared/python -m gpu1bit.x2.testscore pin --arm A --attempt 1 --benchmark-run R --gates-run G
        (after the benchmark, the Z6/Z8 runtime tests and the preflights; refused unless they all pass)
    gpu1bit/shared/python -m gpu1bit.x2.testscore precheck --arm A [--attempt N --evidence E]   (read-only; ALL PASS)
    gpu1bit/shared/python -m gpu1bit.x2.testscore score --arm MB1-B-hybrid-x2   (job --needs gpu:shared)
    gpu1bit/shared/python -m gpu1bit.x2.testscore summarize --arm A | record-failure --arm A --evidence T | report
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from gpu1bit.shared import env

REAL_ROOT = env.ARTIFACTS_ROOT / "H" / "testscore"
ARMS = {"MB1-B-hybrid-x2": {"model": "MB1", "label": "MB1, runtime x2", "nemo": "MB1", "bench": "b1", "gate": "g1"},
        "M1-x2": {"model": "M1", "label": "M1, runtime x2", "nemo": "M1", "bench": "b4", "gate": "g4"}}
MEAN_SETS = ["librispeech_clean", "librispeech_other", "ami", "earnings22", "gigaspeech", "spgispeech", "voxpopuli"]
TEST_COLUMNS = MEAN_SETS + ["common_voice"]
FAKE_SETS = ["fake_a", "fake_b", "fake_c"]
FAKE_N = 20
WORK_BUDGET_S, HOLD_CAP_S = 480.0, 540.0
EVIDENCE_FIELDS = ("user_approval", "dev_reproducer", "review_clean", "freeze", "gates_record", "benchmark")
X8_LABEL = ("runtime x2 - NeMo FP32 is the combined difference of the deployment path: INT8x2 activations; the "
            "runtime's integer and fused arithmetic; batch 1 against the historical batch 64; the runtime's decoder "
            "execution (full CUDA-graph mode against NeMo's historical mode). It is not an x2-versus-x3 test comparison; "
            "no x3 runtime arm is scored on test. No decision depends on a test number.")


# --- modes, sets, paths ------------------------------------------------------------------------------

def fake_name() -> str | None:
    return os.environ.get("GPU1BIT_TESTSCORE_FAKE") or None


def fake_root(name: str) -> Path:
    return env.ARTIFACTS_ROOT / "H" / "testscore-fake" / name


def root(name: str | None = None) -> Path:
    name = fake_name() if name is None else name
    return fake_root(name) if name else REAL_ROOT


def test_sets(name: str | None = None) -> list[str]:
    if name:
        return list(FAKE_SETS)
    import paths
    return list(paths.TEST_SETS)


def records(set_name: str, name: str | None = None) -> list[dict]:
    if name:
        return [{"id": f"{set_name}:{i:03d}", "text": "hello world", "duration": 1.0, "audio_filepath": None}
                for i in range(FAKE_N)]
    import evaluate as ev
    return ev.read_manifest(ev.resolve_manifest(set_name)[1])


def arm_dir(arm: str, name=None) -> Path:
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; test arms are {sorted(ARMS)}")
    d = root(name) / arm
    env.require_mount()
    d.mkdir(parents=True, exist_ok=True)
    return d


def attempt_dir(arm: str, attempt: int, name=None) -> Path:
    d = arm_dir(arm, name) / f"attempt-{attempt}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    data = p.read_bytes()
    keep = data.rfind(b"\n") + 1
    return [json.loads(l) for l in data[:keep].decode().splitlines() if l.strip()]


def repair_tail(p: Path) -> dict | None:
    """Codex x2 c1, finding 3: an interrupted append leaves an incomplete last line, and a later append
    would glue a record to it. Called by the attempt-lock holder before appending: a last line without its
    newline that is a complete JSON object is kept (terminated); otherwise it is dropped. Every complete
    record before it is preserved. Returns what was done (None if the file ends cleanly)."""
    if not p.exists():
        return None
    data = p.read_bytes()
    if not data or data.endswith(b"\n"):
        return None
    cut = data.rfind(b"\n") + 1
    tail = data[cut:]
    try:
        ok = isinstance(json.loads(tail.decode()), dict)
    except ValueError:
        ok = False
    with open(p, "r+b") as f:
        if ok:
            f.seek(0, os.SEEK_END)
            f.write(b"\n")
        else:
            f.truncate(cut)
        f.flush()
        os.fsync(f.fileno())
    return {"file": str(p), "action": "terminated" if ok else "dropped", "bytes": len(tail)}


def append_jsonl(p: Path, entry: dict) -> None:
    """Append one record (callers hold the attempt lock); a torn tail is repaired first."""
    repair_tail(p)
    with open(p, "a") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()
        os.fsync(f.fileno())


def repair_all(arm: str, name=None) -> list[dict]:
    """Under the attempt lock, before anything else: repair the tails of every writable JSONL file of the arm
    (ledger, workers, the open attempts' rows and failure logs); stopped attempts' files are read-only."""
    d = arm_dir(arm, name)
    done = []
    for f in [d / "ledger.jsonl", d / "workers.jsonl"] + sorted(d.glob("attempt-*/*.jsonl")):
        if f.exists() and os.access(f, os.W_OK):
            r = repair_tail(f)
            if r:
                done.append(r)
    for r in done:
        append_jsonl(d / "repairs.jsonl", {**r, "utc": utc()})
    return done


# --- identity ------------------------------------------------------------------------------------------

def identity(arm: str, name=None) -> str:
    if name:
        return os.environ.get("GPU1BIT_TESTSCORE_FAKE_IDENTITY", "fake-identity")
    from gpu1bit.actprec import parts
    from gpu1bit.shared import seldoc
    return parts.identity(arm, "full", selection=seldoc.CANDIDATE)["hash"]


# --- ledger ----------------------------------------------------------------------------------------------

def ledger(arm: str, name=None) -> list[dict]:
    return read_jsonl(arm_dir(arm, name) / "ledger.jsonl")


def state(arm: str, name=None) -> dict:
    """{attempt: {"started": entry, "set_done": {set: entry}, "complete": entry|None, "stopped": entry|None}}."""
    out: dict[int, dict] = {}
    for e in ledger(arm, name):
        a = out.setdefault(e["attempt"], {"started": None, "set_done": {}, "complete": None, "stopped": None})
        if e["event"] == "started":
            a["started"] = e
        elif e["event"] == "set_done":
            a["set_done"][e["set"]] = e
        elif e["event"] in ("complete", "stopped"):
            a[e["event"]] = e
    return out


# --- lock and workers (X6, Y5) ---------------------------------------------------------------------------

def pid_alive(pid: int, start_ticks: int | None = None) -> bool:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except OSError:
        return False
    if fields[0] == "Z":
        return False
    return start_ticks is None or int(fields[19]) == start_ticks


def _start_ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])


def acquire_lock(arm: str, name=None) -> int:
    """The attempt lock for an orchestrator or a recovery command; SystemExit if any holder lives or a
    recorded worker still runs (Y5)."""
    d = arm_dir(arm, name)
    fd = os.open(d / "attempt.lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise SystemExit(f"{arm}: the attempt lock is held by a live process (an orchestrator or one of its workers)")
    alive = [w for w in read_jsonl(d / "workers.jsonl") if pid_alive(w["pid"], w["start_ticks"])]
    if alive:
        os.close(fd)
        raise SystemExit(f"{arm}: attempt lock refused: recorded workers still running {[w['pid'] for w in alive]}")
    return fd


def require_inherited_lock(arm: str, name=None) -> int:
    """A worker runs only with the orchestrator's open lock description (Y5)."""
    raw = os.environ.get("GPU1BIT_ATTEMPT_LOCK_FD")
    if not raw:
        raise SystemExit(f"{arm}: worker refused: no inherited attempt lock (GPU1BIT_ATTEMPT_LOCK_FD)")
    fd = int(raw)
    lock = arm_dir(arm, name) / "attempt.lock"
    try:
        if os.fstat(fd).st_ino != os.stat(lock).st_ino:
            raise SystemExit(f"{arm}: worker refused: the inherited descriptor is not the attempt lock")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, BlockingIOError):
        raise SystemExit(f"{arm}: worker refused: the inherited attempt lock is not held") from None
    return fd


# --- the model (real or stub) -------------------------------------------------------------------------

class _StubModel:
    """Fake mode: deterministic tokens, expected length 10, injected failures by id."""

    def __init__(self, arm, attempt, name):
        self.inject = json.loads(os.environ.get("GPU1BIT_TESTSCORE_INJECT", "{}"))
        self.failures = attempt_dir(arm, attempt, name) / "failures.jsonl"

    def transcribe(self, rec):
        sleep = float(os.environ.get("GPU1BIT_TESTSCORE_SLEEP", "0"))
        if sleep:
            time.sleep(sleep)
        uid = rec["id"]
        kind = self.inject.get("fail", {}).get(uid)
        prior = sum(1 for f in read_jsonl(self.failures) if f["id"] == uid)
        if kind and prior < int(self.inject.get("times", {}).get(uid, 1)):
            if kind == "decode":
                raise RuntimeError("injected decoder exception")
            return {"hyp": "x", "tokens": [1], "timestamps": [0], "enc_len": 9 if kind == "length" else 10,
                    "expected_len": 10, "finite": kind != "nan"}
        return {"hyp": "hello world", "tokens": [int(h, 16) for h in hashlib.sha256(uid.encode()).hexdigest()[:4]],
                "timestamps": [0, 1], "enc_len": 10, "expected_len": 10, "finite": True}


def expected_encoder_length(model, n_samples: int) -> int:
    """The frontend's expected encoder length: NeMo's feature length (FilterbankFeatures.get_seq_len)
    through the encoder's subsampling (calc_length with pre_encode's parameters)."""
    import torch
    from nemo.collections.asr.parts.submodules.subsampling import calc_length
    feat = model.preprocessor.featurizer.get_seq_len(torch.tensor([n_samples]))
    pe = model.encoder.pre_encode
    return int(calc_length(feat, all_paddings=pe._left_padding + pe._right_padding, kernel_size=pe._kernel_size,
                           stride=pe._stride, ceil_mode=pe._ceil_mode, repeat_num=pe._sampling_num)[0])


class _RealModel:
    def __init__(self, arm):
        import torch
        from gpu1bit.actprec import parts
        from gpu1bit.shared import fulldev
        parts.install()
        spec = fulldev.parse_arm(arm)
        with torch.inference_mode():
            self.model, self.info = parts.build(spec["model"], spec["parts"], kernel=spec["kernel"] or "int8_mma")

    def transcribe(self, rec, audio=None):
        import torch
        import evaluate as ev
        from gpu1bit.shared import fulldev
        clip, padded = audio if audio is not None else fulldev.load_clip(rec)
        x = torch.from_numpy(clip).unsqueeze(0).cuda()
        n = torch.tensor([len(clip)], dtype=torch.long).cuda()
        with torch.inference_mode(), ev.strict_fp32("cuda"), ev.inference_settings(self.model):
            enc, enc_len = self.model.forward(input_signal=x, input_signal_length=n)
            finite = bool(torch.isfinite(enc).all())
            hyps = self.model.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc, encoded_lengths=enc_len,
                                                                        return_hypotheses=True)
            h = (hyps[0] if isinstance(hyps, tuple) else hyps)[0]
            seq = getattr(h, "y_sequence", None)
            ts = getattr(h, "timestamp", None)
        return {"hyp": h.text, "tokens": [int(t) for t in (seq.tolist() if hasattr(seq, "tolist") else (seq or []))],
                "timestamps": [int(t) for t in ts.tolist()] if hasattr(ts, "tolist") else [],
                "enc_len": int(enc_len[0]), "expected_len": expected_encoder_length(self.model, len(clip)),
                "finite": finite, "padded_short": padded}


# --- worker ---------------------------------------------------------------------------------------------

def done_ids(arm: str, attempt: int, set_name: str, name=None) -> set[str]:
    return {r["id"] for r in read_jsonl(attempt_dir(arm, attempt, name) / f"{set_name}.jsonl")}


def failures(arm: str, attempt: int, name=None) -> list[dict]:
    return read_jsonl(attempt_dir(arm, attempt, name) / "failures.jsonl")


def unresolved(arm: str, attempt: int, name=None) -> dict[str, int]:
    """{id: failure count} of failed utterances without a row."""
    done = set()
    for s in test_sets(name):
        done |= done_ids(arm, attempt, s, name)
    out: dict[str, int] = {}
    for f in failures(arm, attempt, name):
        if f["id"] not in done:
            out[f["id"]] = out.get(f["id"], 0) + 1
    return out


def worker(arm: str, attempt: int, set_name: str, retry: list[str] | None, name=None) -> int:
    require_inherited_lock(arm, name)
    st = state(arm, name).get(attempt)
    ident = identity(arm, name)
    if not st or not st["started"] or st["started"]["identity"] != ident or st["complete"] or st["stopped"]:
        raise SystemExit(f"{arm}: worker refused: attempt {attempt} is not an open attempt of the current identity")
    append_jsonl(arm_dir(arm, name) / "workers.jsonl", {"pid": os.getpid(), "start_ticks": _start_ticks(os.getpid()),
                                                        "attempt": attempt, "set": set_name, "utc": utc()})
    if name:
        return _work(arm, attempt, set_name, retry, ident, name, _StubModel(arm, attempt, name), None)
    with env.gpu_lock(f"H:x2 test {arm} {set_name}") as lock:
        model = _RealModel(arm)
        code = _work(arm, attempt, set_name, retry, ident, name, model, lock)
        sys.stdout.flush()
        os._exit(code)


def _work(arm, attempt, set_name, retry, ident, name, model, lock) -> int:
    from gpu1bit.shared import fulldev
    d = attempt_dir(arm, attempt, name)
    recs = records(set_name, name)
    done = done_ids(arm, attempt, set_name, name)
    failed = {f["id"] for f in failures(arm, attempt, name)}
    if retry is not None:
        want = set(retry)
        todo = [(i, r) for i, r in enumerate(recs) if r["id"] in want and r["id"] not in done]
    else:
        todo = [(i, r) for i, r in enumerate(recs) if r["id"] not in done and r["id"] not in failed]
    chunk = int(os.environ.get("GPU1BIT_TESTSCORE_CHUNK", "0")) if name else 0
    kill = os.environ.get("GPU1BIT_TESTSCORE_KILL")
    start, slowest, n = time.monotonic(), 0.0, 0
    stream = ((i, r, None) for i, r in todo) if name else fulldev.prefetched(todo)
    repair_tail(d / f"{set_name}.jsonl")                   # the inherited attempt lock is held (finding 3)
    with open(d / f"{set_name}.jsonl", "a") as sink:
        for i, rec, audio in stream:
            if chunk and n >= chunk:
                break
            if lock is not None:
                now = time.monotonic()
                if now - start + 3 * slowest + 1 > WORK_BUDGET_S or now - lock["acquired"] + 3 * slowest + 1 > HOLD_CAP_S:
                    break
            t0 = time.monotonic()
            if kill == "mid-utterance" and set_name == test_sets(name)[0] and n == 2:
                os._exit(9)                                  # killed mid-utterance: no row for this one
            try:
                out = model.transcribe(rec) if name else model.transcribe(rec, audio)
                kind = None if out["finite"] else "not_finite"
                if kind is None and out["enc_len"] != out["expected_len"]:
                    kind = "length"
                detail = None if kind is None else {"finite": out["finite"], "enc_len": out["enc_len"],
                                                    "expected_len": out["expected_len"]}
            except Exception as exc:  # noqa: BLE001 - Z1: a failure record, not a row
                kind, detail, out = "decode_error", repr(exc), None
            if kind:
                append_jsonl(d / "failures.jsonl", {"id": rec["id"], "set": set_name, "kind": kind, "detail": detail,
                                                    "identity": ident, "worker_pid": os.getpid(), "utc": utc()})
            else:
                row = fulldev.make_row(i, rec["id"], out["hyp"], tokens=out["tokens"], timestamps=out["timestamps"],
                                       padded_short=bool(out.get("padded_short")), reference=rec["text"],
                                       duration=rec["duration"], enc_len=out["enc_len"], finite=True, identity_hash=ident)
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
            n += 1
            slowest = max(slowest, time.monotonic() - t0)
        sink.flush()
        os.fsync(sink.fileno())
    print(json.dumps({"worker": os.getpid(), "set": set_name, "processed": n}), flush=True)
    return 0


# --- set summaries, completion ----------------------------------------------------------------------------

def set_done(arm: str, attempt: int, set_name: str, name=None) -> dict:
    """Write the per-set summary and append set_done; refuses with unresolved failures (Z1)."""
    import evaluate as ev
    bad = unresolved(arm, attempt, name)
    if bad:
        raise SystemExit(f"{arm}: set_done refused: unresolved failures {sorted(bad)[:5]}")
    d = attempt_dir(arm, attempt, name)
    rows = read_jsonl(d / f"{set_name}.jsonl")
    want = [r["id"] for r in records(set_name, name)]
    got = [r["id"] for r in rows]
    if sorted(got) != sorted(want) or len(set(got)) != len(got):
        raise SystemExit(f"{arm} {set_name}: rows are not every utterance exactly once")
    by = {r["id"]: r for r in rows}
    ordered = [by[u] for u in want]
    summary = {"set": set_name, "arm": arm, "attempt": attempt, "identity": state(arm, name)[attempt]["started"]["identity"],
               "utterances": len(ordered), "wer": ev.wer_summary(ordered), "empty_hypotheses": sum(r["empty"] for r in ordered),
               "rows_sha256": sha(d / f"{set_name}.jsonl"), "records": ordered}
    if not name:
        summary["manifest_sha256"] = sha(ev.resolve_manifest(set_name)[1])
    p = d / f"{set_name}.json"
    p.write_text(json.dumps(summary, ensure_ascii=False) + "\n")
    entry = {"event": "set_done", "attempt": attempt, "set": set_name, "sha256": sha(p), "utc": utc()}
    append_jsonl(arm_dir(arm, name) / "ledger.jsonl", entry)
    return entry


def summarize(arm: str, attempt: int, name=None) -> dict:
    """Summary-only completion (zero decoding): every set done, files unchanged, no unresolved failure."""
    st = state(arm, name)[attempt]
    if st["stopped"]:
        raise SystemExit(f"{arm}: attempt {attempt} is stopped")
    if unresolved(arm, attempt, name):
        raise SystemExit(f"{arm}: complete refused: unresolved failures")
    d = attempt_dir(arm, attempt, name)
    if set(st["set_done"]) != set(test_sets(name)):
        raise SystemExit(f"{arm}: not every set is done: {sorted(set(test_sets(name)) - set(st['set_done']))}")
    sets = {}
    for s, e in st["set_done"].items():
        p = d / f"{s}.json"
        if not p.exists() or sha(p) != e["sha256"]:
            raise SystemExit(f"{arm} {s}: per-set file hash differs from its set_done hash (modified or replaced)")
        doc = json.loads(p.read_text())
        if sha(d / f"{s}.jsonl") != doc["rows_sha256"]:
            raise SystemExit(f"{arm} {s}: rows file hash differs from the per-set summary")
        sets[s] = {k: doc[k] for k in ("utterances", "wer", "empty_hypotheses")}
    out = {"arm": arm, "attempt": attempt, "identity": st["started"]["identity"], "sets": sets,
           "recovered_failures": failures(arm, attempt, name)}
    p = d / "summary.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    if not st["complete"]:
        append_jsonl(arm_dir(arm, name) / "ledger.jsonl", {"event": "complete", "attempt": attempt, "sha256": sha(p),
                                                           "utc": utc()})
    return out


def _stop(arm, attempt, name, reason, evidence):
    append_jsonl(arm_dir(arm, name) / "ledger.jsonl", {"event": "stopped", "attempt": attempt, "reason": reason,
                                                       "evidence": evidence, "utc": utc()})
    d = attempt_dir(arm, attempt, name)
    for p in d.glob("*.jsonl"):
        os.chmod(p, 0o444)                                    # rows kept in place, read-only (Y3)
    raise SystemExit(f"{arm}: attempt {attempt} STOPPED (Y3): {reason}. Scoring stops; tell the user.")


# --- orchestrator ---------------------------------------------------------------------------------------

def score(arm: str, new_attempt: Path | None = None, name=None) -> dict:
    name = fake_name() if name is None else name
    if not name:
        os.environ["GPU1BIT_SELECTION"] = str(__import__("gpu1bit.shared.seldoc", fromlist=["x"]).CANDIDATE)
    fd = acquire_lock(arm, name)
    repair_all(arm, name)
    st = state(arm, name)
    ident = identity(arm, name)
    if any(a["complete"] for a in st.values()):
        raise SystemExit(f"{arm} was scored once (an attempt is complete); only `report` remains")
    open_attempts = [k for k, a in st.items() if not a["stopped"]]
    if open_attempts:
        attempt = max(open_attempts)
        if st[attempt]["started"]["identity"] != ident:
            raise SystemExit(f"{arm}: resume refused: attempt {attempt} has identity "
                             f"{st[attempt]['started']['identity'][:12]}, the current identity is {ident[:12]} (Y3)")
        if set(st[attempt]["set_done"]) == set(test_sets(name)):
            raise SystemExit(f"{arm}: every set is done; only `summarize` (summary-only recovery) is allowed")
        if new_attempt:
            raise SystemExit(f"{arm}: attempt {attempt} is open; resume it instead of a new attempt")
    else:
        attempt = (max(st) + 1) if st else 1
        ev = None
        if st:   # every earlier attempt is stopped: a new attempt is an exceptional retry (Y3)
            if not new_attempt or not Path(new_attempt).exists():
                raise SystemExit(f"{arm}: a new attempt is refused without replacement evidence (Y3)")
            ev = json.loads(Path(new_attempt).read_text())
        result = precheck(arm, attempt, ev, name)      # the attempt's own pins and, for a retry, its evidence
        if not result["pass"]:
            raise SystemExit(f"{arm}: precheck failed for attempt {attempt}; nothing is decoded: "
                             f"{[c['check'] for c in result['checks'] if not c['ok']][:6]}")
        append_jsonl(arm_dir(arm, name) / "ledger.jsonl", {"event": "started", "attempt": attempt, "identity": ident,
                                                           "evidence": str(new_attempt) if new_attempt else None, "utc": utc()})
    kill = os.environ.get("GPU1BIT_TESTSCORE_KILL")
    e = {k: v for k, v in os.environ.items() if k not in ("CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES")}
    if name:
        e["CUDA_VISIBLE_DEVICES"] = ""
    e["GPU1BIT_ATTEMPT_LOCK_FD"] = str(fd)
    if kill == "mid-utterance":
        e["GPU1BIT_TESTSCORE_KILL"] = kill
    else:
        e.pop("GPU1BIT_TESTSCORE_KILL", None)
    chunks = 0

    def launch(set_name, retry=None):
        nonlocal chunks
        cmd = ["timeout", "600", sys.executable, "-m", "gpu1bit.x2.testscore", "worker", "--arm", arm,
               "--attempt", str(attempt), "--set", set_name]
        if retry is not None:
            cmd += ["--retry", ",".join(retry)]
        rc = subprocess.run(cmd, env=e, pass_fds=(fd,), cwd=str(env.BINARY_DIR)).returncode
        chunks += 1
        if rc != 0:
            raise SystemExit(f"{arm}: worker for {set_name} failed (rc {rc}); resume under the same attempt")
        if kill == "between-chunks" and chunks == 1:
            os._exit(9)

    for set_name in test_sets(name):
        if set_name in state(arm, name)[attempt]["set_done"]:
            continue
        ids = [r["id"] for r in records(set_name, name)]
        for _ in range(1000):
            done = done_ids(arm, attempt, set_name, name)
            failed = {f["id"] for f in failures(arm, attempt, name)}
            if all(u in done or u in failed for u in ids):
                break
            launch(set_name)
        while True:   # Z1: one retry per failed utterance, in fresh (bounded) workers until each has had it
            bad = {u: c for u, c in unresolved(arm, attempt, name).items() if u in set(ids)}
            persistent = sorted(u for u, c in bad.items() if c >= 2)
            if persistent:                                    # an actual second failure (finding 4)
                _stop(arm, attempt, name, f"utterances failed twice {persistent[:5]}",
                      [f for f in failures(arm, attempt, name) if f["id"] in persistent])
            if not bad:
                break
            before = (len(done_ids(arm, attempt, set_name, name)), len(failures(arm, attempt, name)))
            launch(set_name, retry=sorted(bad))
            if (len(done_ids(arm, attempt, set_name, name)), len(failures(arm, attempt, name))) == before:
                raise SystemExit(f"{arm}: a retry worker for {set_name} made no progress; resume under the same attempt")
        set_done(arm, attempt, set_name, name)
        if kill == "between-sets":
            os._exit(9)
    if kill == "during-summary":
        os._exit(9)
    out = summarize(arm, attempt, name)
    os.close(fd)
    return out


def recover_summary(arm: str, name=None) -> dict:
    name = fake_name() if name is None else name
    fd = acquire_lock(arm, name)
    repair_all(arm, name)
    st = state(arm, name)
    open_attempts = [k for k, a in st.items() if not a["stopped"]]
    if not open_attempts:
        raise SystemExit(f"{arm}: no open attempt")
    attempt = max(open_attempts)
    if st[attempt]["started"]["identity"] != identity(arm, name):
        raise SystemExit(f"{arm}: summary refused: identity changed")
    out = summarize(arm, attempt, name)
    os.close(fd)
    print("decoded 0 utterances (summary-only recovery)")
    return out


def record_failure(arm: str, evidence: str, name=None) -> None:
    name = fake_name() if name is None else name
    fd = acquire_lock(arm, name)
    try:
        repair_all(arm, name)
        st = state(arm, name)
        open_attempts = [k for k, a in st.items() if not a["stopped"] and not a["complete"]]
        if not open_attempts:
            raise SystemExit(f"{arm}: no open attempt to record")
        attempt = max(open_attempts)
        if set(st[attempt]["set_done"]) == set(test_sets(name)):
            raise SystemExit(f"{arm}: every set is done; only summary-only recovery is allowed")
        _stop(arm, attempt, name, "recorded by hand", evidence)       # appended under the lock
    finally:
        os.close(fd)


# --- qualification, per-attempt pins and the precheck (Codex x2 c1, findings 1 and 2) --------------------

NEMO_MB1_TEST = Path("/mnt/hd/wilderness-labs-stt/parakeet-binary/eval/MB1-test")
SCORER_FILES = ("finetune/parakeet-ternary/evaluate.py", "finetune/whisper-ternary/wer.py",
                "finetune/parakeet-binary/gpu1bit/x2/testscore.py", "finetune/parakeet-binary/gpu1bit/shared/fulldev.py")
RECORD_KEYS = ("gates", "preflight:MB1-B-hybrid-x2", "preflight:M1-x2", "recognition", "memory")


def scorer_now() -> dict:
    import evaluate as ev
    import transformers.models.whisper.english_normalizer as en
    out = {k: env.sha256_file(env.REPO / k) for k in SCORER_FILES}
    out["normalizer.json"] = env.sha256_file(Path(ev.NORMALIZER_JSON))
    out["english_normalizer.py"] = env.sha256_file(Path(en.__file__))
    return out


def base_dir(name=None) -> Path:
    """Where the qualification records live: H/x2 (gates, preflight, runtime records) in real mode; the fake
    root's stand-ins (same file names) in fake mode."""
    return fake_root(name) if name else env.ARTIFACTS / "x2"


def pins_path(arm: str, attempt: int, name=None) -> Path:
    return arm_dir(arm, name) / f"pins-{attempt}.json"


def record_paths(benchmark_run: str, gates_run: str, name=None) -> dict[str, Path]:
    b = base_dir(name)
    return {"gates": b / f"gates-{gates_run}.json", **{f"preflight:{a}": b / f"preflight-{a}.json" for a in ARMS},
            "recognition": b / f"runtime-recognition-{gates_run}.json", "memory": b / f"runtime-fresh-compare-{benchmark_run}.json"}


def freeze_file(benchmark_run: str, name=None) -> Path:
    if name:
        return base_dir(name) / f"freeze-{benchmark_run}.sha256"
    from gpu1bit.shared import selection
    return selection.freeze_path(benchmark_run)


def bench_record(benchmark_run: str, name=None) -> Path:
    if name:
        return base_dir(name) / f"bench-{benchmark_run}.json"
    from gpu1bit.x2 import bench_session as BS
    return BS.ROOT / benchmark_run / "report-dev.json"


def bench_qualification(benchmark_run: str, name=None) -> dict:
    """The recorded six-arm benchmark as a whole: {"complete", "problems", "identities", ...}. Real mode: both
    phases' reports written and every step an accepted, bound measurement (bench_session._collect, i.e.
    rerun_timing.reusable), never a single chunk. Fake mode: the stand-in record."""
    if name:
        p = bench_record(benchmark_run, name)
        return json.loads(p.read_text()) if p.exists() else {"complete": False, "problems": [f"no benchmark record {p}"],
                                                             "identities": {}}
    from gpu1bit.x2 import bench_session as BS
    try:
        roster = BS.load_roster(benchmark_run)
        BS.require_dev_results(benchmark_run, roster)
        for f in ("report-dev.json", "report-test.json"):
            if not (BS.ROOT / benchmark_run / f).exists():
                raise SystemExit(f"{benchmark_run}: {f} is not written")
        BS._collect(roster, "historical-test")
        dev = BS._collect(roster, "dev")
        return {"complete": True, "problems": [], "identities": dev["guard"]["identities"], "gate_run": roster["gate_run"],
                "freeze_file": roster["freeze_file"]}
    except (SystemExit, OSError, ValueError, KeyError) as exc:
        return {"complete": False, "problems": [str(exc)[:300]], "identities": {}}


def qualification_checks(arm: str, identities: dict, pins: dict | None, bench: dict, docs: dict, unchanged: dict,
                         scorer_now: dict, pointer: str | None) -> list[dict]:
    """Every precheck item about the attempt's qualification (pure). identities: the current {arm: hash};
    pins: the attempt's pins; bench: bench_qualification; docs: {record key: loaded record or None};
    unchanged: {record key: equal to its pinned SHA-256}."""
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": str(detail)[:300]})
    if not pins:
        check(f"{arm}: the attempt has qualification pins (`pin`)", False)
        return checks
    mb1 = identities.get("MB1-B-hybrid-x2")
    ident = identities.get(arm)
    check(f"{arm}: the pins are this arm's", pins.get("arm") == arm, pins.get("arm"))
    check(f"{arm}: identity equals the pinned identity", ident == (pins.get("identities") or {}).get(arm), ident)
    check("the benchmark (both phases) is complete and every step an accepted measurement",
          bench.get("complete") is True, bench.get("problems"))
    check(f"{arm}: identity equals the benchmark identity ({ARMS[arm]['bench']})", (bench.get("identities") or {}).get(arm) == ident)
    if bench.get("gate_run") is not None:
        check("the benchmark is bound to the pinned gate run", bench.get("gate_run") == pins.get("gates_run"), bench.get("gate_run"))
    check("scorer hashes equal the pinned ones", scorer_now == pins.get("scorer"),
          sorted(k for k in set(scorer_now) | set(pins.get("scorer") or {}) if scorer_now.get(k) != (pins.get("scorer") or {}).get(k)))
    g = docs.get("gates") or {}
    check("both gates recorded as passing", g.get("pass") is True)
    check(f"{arm}: the gate record is for the same identity", (g.get("identities") or {}).get(ARMS[arm]["gate"]) == ident)
    check("the gates ran under the candidate selection", (g.get("selection") or {}).get("sha256") == pins.get("candidate_sha256"))
    pf = docs.get(f"preflight:{arm}") or {}
    check(f"{arm}: the Y6/Z3 preflight passed", pf.get("pass") is True)
    check(f"{arm}: the preflight is of the same identity", pf.get("identity") == ident)
    rec, mem = docs.get("recognition") or {}, docs.get("memory") or {}
    check("the Z6 runtime recognition contract passed", rec.get("ok") is True)
    check("the Z6 record is of the MB1 runtime identity and the pinned gate run",
          rec.get("identity") == mb1 and rec.get("gate_run") == pins.get("gates_run"))
    check("the Z8 runtime memory contract passed", mem.get("ok") is True)
    check("the Z8 record is of the MB1 runtime identity and the pinned benchmark",
          mem.get("identity") == mb1 and mem.get("bench_run") == pins.get("benchmark_run"))
    for k in RECORD_KEYS:
        check(f"record {k} present and unchanged since pinned", unchanged.get(k) is True)
    check("the deployment pointer is x3 as pinned or the promoted candidate (Z2)", pointer in ("x3", "promoted"), pointer)
    return checks


def _load_records(pins: dict) -> tuple[dict, dict]:
    docs, unchanged = {}, {}
    for k in RECORD_KEYS:
        r = (pins.get("records") or {}).get(k) or {}
        path = Path(r["path"]) if r.get("path") else None
        ok = bool(path and path.exists())
        docs[k] = json.loads(path.read_text()) if ok else None
        unchanged[k] = ok and sha(path) == r.get("sha256")
    return docs, unchanged


def _pointer(benchmark_run: str, name=None) -> str | None:
    if name:
        return "x3"                                             # fake mode has no deployment pointer
    from gpu1bit.shared import selection
    try:
        return selection.verify_freeze(benchmark_run).get("pointer")
    except (SystemExit, OSError):
        return None


def _utc_epoch(text: str) -> float:
    import calendar
    return float(calendar.timegm(time.strptime(text, "%Y-%m-%dT%H:%M:%SZ")))


def evidence_template(arm: str, ident: str) -> dict:
    """The fields a new attempt's evidence carries (Y3): the arm, the new identity and a path (or run name)
    per record. Placeholders never pass: every record is opened and checked (evidence_checks)."""
    return {"arm": arm, "identity": ident, **{k: f"<{k}>" for k in EVIDENCE_FIELDS}}


CLEAN_VERDICTS = ("CLEAN", "CLEAN (no blocker or major)")


def review_is_clean(text: str) -> bool:
    """A Codex review's verdict, its last non-empty line, is one of the supported CLEAN forms (Codex x2 c2
    minor); NOT CLEAN in any form, or anything else, is not."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return bool(lines) and lines[-1] in CLEAN_VERDICTS


def evidence_checks(ev: dict | None, arm: str, ident: str, st: dict, pins: dict | None, old_pins: list[dict],
                    name=None) -> list[str]:
    """Problems with an exceptional-retry evidence document (Y3; Codex x2 c1, finding 2), [] if none. Every
    referenced record is opened: the user's approval of this arm and identity; a dev-only reproducer of one of
    the arm's stopped attempts; a Codex review whose verdict line is CLEAN; and the new freeze, gates record
    and benchmark, which must be the new attempt's own pins (not an earlier attempt's) and, like every
    record, be written after the last stop."""
    from gpu1bit.shared import fulldev
    if not ev:
        return ["no replacement evidence"]
    out = [f"missing {k}" for k in EVIDENCE_FIELDS if not ev.get(k)]
    if ev.get("arm") != arm:
        out.append(f"evidence is for arm {ev.get('arm')}")
    if ev.get("identity") != ident:
        out.append(f"evidence is bound to identity {str(ev.get('identity'))[:12]}, not the current {str(ident)[:12]}")
    stops = {k: a["stopped"] for k, a in st.items() if a["stopped"]}
    if not stops:
        return out + ["no stopped attempt to replace"]
    after = max(_utc_epoch(e["utc"]) for e in stops.values()) + 1

    def newer(path: Path, what: str) -> bool:
        if not path.exists():
            out.append(f"{what}: {path} does not exist")
            return False
        if path.stat().st_mtime <= after:
            out.append(f"{what}: {path} is not newer than the last stop")
            return False
        return True

    def load(field: str):
        path = Path(str(ev.get(field) or ""))
        if not newer(path, field):
            return None
        try:
            return json.loads(path.read_text())
        except ValueError:
            out.append(f"{field}: {path} is not a JSON record")
            return None
    a = load("user_approval")
    if a is not None and not (a.get("kind") == "user_approval" and a.get("approved") is True and a.get("arm") == arm
                              and a.get("identity") == ident):
        out.append("user_approval: not the user's approval of this arm and identity")
    r = load("dev_reproducer")
    if r is not None and not (r.get("kind") == "dev_reproducer" and r.get("arm") == arm and r.get("stopped_attempt") in stops
                              and r.get("sets") and all(x in fulldev.DEV_SETS for x in r["sets"])):
        out.append("dev_reproducer: not a dev-only reproducer of a stopped attempt of this arm")
    review = Path(str(ev.get("review_clean") or ""))
    if newer(review, "review_clean") and not review_is_clean(review.read_text()):
        out.append("review_clean: the review's verdict is not CLEAN")
    if not pins:
        return out + ["the new attempt has no pins (`pin`)"]
    if Path(str(ev.get("freeze"))) != freeze_file(pins["benchmark_run"], name):
        out.append("freeze: not the new attempt's pinned benchmark freeze")
    else:
        newer(Path(str(ev["freeze"])), "freeze")
    gates_path = Path((pins.get("records") or {}).get("gates", {}).get("path", ""))
    if Path(str(ev.get("gates_record"))) != gates_path:
        out.append("gates_record: not the new attempt's pinned gate record")
    else:
        newer(gates_path, "gates_record")
    if ev.get("benchmark") != pins.get("benchmark_run"):
        out.append("benchmark: not the new attempt's pinned replacement benchmark")
    else:
        newer(bench_record(pins["benchmark_run"], name), "benchmark")
    for old in old_pins:
        if old.get("benchmark_run") == pins.get("benchmark_run") or old.get("gates_run") == pins.get("gates_run"):
            out.append(f"the new pins reuse attempt {old.get('attempt')}'s benchmark or gates")
    return out


def _qualify(arm: str, attempt: int, pins: dict | None, name=None) -> list[dict]:
    idents = {a: identity(a, name) for a in ARMS}
    docs, unchanged = _load_records(pins) if pins else ({}, {})
    bench = bench_qualification(pins["benchmark_run"], name) if pins else {}
    return qualification_checks(arm, idents, pins, bench, docs, unchanged, scorer_now(),
                                _pointer(pins["benchmark_run"], name) if pins else None)


def pin(arm: str, attempt: int, benchmark_run: str, gates_run: str, name=None) -> dict:
    """The attempt's qualification pins (written once, per arm and attempt, only if the qualification passes):
    the benchmark run, gate run, freeze, every qualification record's path and SHA-256, the identities, the
    scorer hashes and the candidate's content hash. A legitimate fix (Y3) gets its own new pins."""
    from gpu1bit.shared import seldoc
    name = fake_name() if name is None else name
    if not name:
        os.environ["GPU1BIT_SELECTION"] = str(seldoc.CANDIDATE)
    path = pins_path(arm, attempt, name)
    if path.exists():
        raise SystemExit(f"{path} exists; an attempt's pins are written once")
    recs = record_paths(benchmark_run, gates_run, name)
    doc = {"arm": arm, "attempt": attempt, "benchmark_run": benchmark_run, "gates_run": gates_run,
           "freeze_file": str(freeze_file(benchmark_run, name)),
           "records": {k: {"path": str(p), "sha256": sha(p) if p.exists() else None} for k, p in recs.items()},
           "identities": {a: identity(a, name) for a in ARMS}, "scorer": scorer_now(),
           "candidate_sha256": env.sha256_file(seldoc.CANDIDATE), "utc": utc()}
    bad = [c for c in _qualify(arm, attempt, doc, name) if not c["ok"]]
    if bad:
        raise SystemExit(f"pin refused: the qualification fails: {[c['check'] for c in bad][:8]}")
    path.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def precheck(arm: str, attempt: int | None = None, evidence: dict | None = None, name=None) -> dict:
    """Read-only; ALL PASS required before an attempt starts: the qualification of the attempt's own pins,
    for an exceptional retry its evidence, and in real mode items 4 to 6 of plans/test-scoring.md."""
    from gpu1bit.shared import seldoc
    name = fake_name() if name is None else name
    if not name:
        os.environ["GPU1BIT_SELECTION"] = str(seldoc.CANDIDATE)
    st = state(arm, name)
    attempt = attempt or ((max(st) + 1) if st else 1)
    pp = pins_path(arm, attempt, name)
    pins = json.loads(pp.read_text()) if pp.exists() else None
    checks = _qualify(arm, attempt, pins, name)
    if attempt > 1:
        old = [json.loads(pins_path(arm, k, name).read_text()) for k in range(1, attempt) if pins_path(arm, k, name).exists()]
        probs = evidence_checks(evidence, arm, identity(arm, name), st, pins, old, name)
        checks.append({"check": "exceptional-retry evidence opened and bound (Y3)", "ok": not probs, "detail": str(probs)[:300]})
    if not name:
        import binary_test

        def mk(item):
            def c(nm, ok, detail=""):
                checks.append({"check": f"item {item}: {nm}", "ok": bool(ok), "detail": str(detail)[:300]})
            return c
        binary_test._baseline_checks(mk(4), mk(5), mk(6))
    result = {"pass": bool(checks) and all(c["ok"] for c in checks), "arm": arm, "attempt": attempt, "utc": utc(),
              "checks": checks}
    print("ALL PASS" if result["pass"] else "PRECHECK FAILED")
    for c in checks:
        if not c["ok"]:
            print("  FAIL", c["check"], c["detail"])
    return result


# --- report (X8) ------------------------------------------------------------------------------------------

def report_markdown(runtime: dict[str, dict[str, float]], nemo: dict[str, dict[str, float]],
                    empty: dict[str, dict[str, float]]) -> str:
    """The TEST.md section: runtime x2 rows, runtime minus its own NeMo FP32 row, empty-hypothesis rates."""
    def mean(d):
        return sum(d[s] for s in MEAN_SETS) / len(MEAN_SETS)
    head = "| Arm | " + " | ".join(MEAN_SETS) + " | mean | common_voice |"
    sep = "| --- |" + " ---: |" * (len(MEAN_SETS) + 2)
    lines = ["## Runtime x2 (plans/x2-default.md)", "", f"**{X8_LABEL}**", "", head, sep]
    for arm, cfg in ARMS.items():
        r = runtime[arm]
        lines.append(f"| {cfg['label']} | " + " | ".join(f"{r[s]:.2f}" for s in MEAN_SETS) + f" | {mean(r):.2f} | {r['common_voice']:.2f} |")
    for arm, cfg in ARMS.items():
        r, n = runtime[arm], nemo[cfg["nemo"]]
        lines.append(f"| {cfg['label']} - {cfg['nemo']} NeMo FP32 (points) | "
                     + " | ".join(f"{r[s] - n[s]:+.2f}" for s in MEAN_SETS)
                     + f" | {mean(r) - mean(n):+.2f} | {r['common_voice'] - n['common_voice']:+.2f} |")
    lines += ["", "### Empty hypotheses, runtime x2 (fraction of scored utterances)", "",
              "| Arm | " + " | ".join(TEST_COLUMNS) + " |", "| --- |" + " ---: |" * len(TEST_COLUMNS)]
    for arm, cfg in ARMS.items():
        lines.append(f"| {cfg['label']} | " + " | ".join(f"{100 * empty[arm][s]:.2f}%" for s in TEST_COLUMNS) + " |")
    return "\n".join(lines) + "\n"


TEST_SECTION = "## Runtime x2 (plans/x2-default.md)"
RESULTS_SECTION = "## Test scoring of the x2 defaults (plans/x2-default.md)"


def attempts_disclosure(arm: str, name=None) -> dict:
    """Y3: every attempt with its status and reason, and the utterances the attempts share compared (text)."""
    st = state(arm, name)
    out = {"attempts": [], "shared": None}
    rows = {}
    for k in sorted(st):
        a = st[k]
        status = "complete" if a["complete"] else "stopped" if a["stopped"] else "open"
        out["attempts"].append({"attempt": k, "status": status, "identity": (a["started"] or {}).get("identity"),
                                "reason": (a["stopped"] or {}).get("reason")})
        rows[k] = {r["id"]: r for p in attempt_dir(arm, k, name).glob("*.jsonl") if p.name != "failures.jsonl"
                   for r in read_jsonl(p)}
    if len(rows) > 1:
        ks = sorted(rows)
        shared = set.intersection(*(set(rows[k]) for k in ks))
        out["shared"] = {"attempts": ks, "utterances": len(shared),
                         "differing": sum(len({json.dumps(rows[k][u].get("hyp")) for k in ks}) > 1 for u in shared)}
    return out


def collect(name=None, arms=None) -> tuple[dict, dict, dict]:
    """(runtime WER %, empty-hypothesis fraction, attempts disclosure) per arm, from the single complete
    attempt; SystemExit ("report refused") while it is missing or any unresolved failure remains (Z1)."""
    runtime, empty, disclosure = {}, {}, {}
    for arm in (arms or list(ARMS)):
        st = state(arm, name)
        done = [k for k, a in st.items() if a["complete"]]
        if len(done) != 1:
            latest = max(st) if st else None
            if latest is not None and unresolved(arm, latest, name):
                raise SystemExit(f"{arm}: report refused: unresolved failures in attempt {latest}")
            raise SystemExit(f"{arm}: report refused: no single complete attempt")
        if unresolved(arm, done[0], name):
            raise SystemExit(f"{arm}: report refused: unresolved failures")
        summary = attempt_dir(arm, done[0], name) / "summary.json"
        if sha(summary) != st[done[0]]["complete"]["sha256"]:
            raise SystemExit(f"{arm}: report refused: summary.json differs from its complete entry")
        s = json.loads(summary.read_text())
        runtime[arm] = {k: 100 * v["wer"]["wer"] for k, v in s["sets"].items()}
        empty[arm] = {k: v["empty_hypotheses"] / v["utterances"] for k, v in s["sets"].items()}
        disclosure[arm] = attempts_disclosure(arm, name)
    return runtime, empty, disclosure


def _replace_section(text: str, marker: str, body: str) -> str:
    """`body` (starting with `marker`) in place of the section `marker` heads (up to the next "## "
    heading), or appended."""
    if marker not in text:
        return text.rstrip() + "\n\n" + body.rstrip() + "\n"
    before, after = text.split(marker, 1)
    rest = after.split("\n## ", 1)
    tail = "\n## " + rest[1] if len(rest) == 2 else ""
    return before.rstrip() + "\n\n" + body.rstrip() + "\n" + tail


def write_sections(md: str, test_md: Path | None = None, results_md: Path | None = None) -> None:
    """The X8-labelled section into results/TEST.md and gpu1bit/RESULTS.md (replaced, never duplicated)."""
    test_md = Path(test_md or env.BINARY_DIR / "results" / "TEST.md")
    results_md = Path(results_md or env.GPU1BIT / "RESULTS.md")
    test_md.write_text(_replace_section(test_md.read_text(), TEST_SECTION, md))
    body = RESULTS_SECTION + "\n\n" + md.replace(TEST_SECTION, "### Test WER of the defaults through their runtimes "
                                                                 "(also in results/TEST.md)", 1)
    results_md.write_text(_replace_section(results_md.read_text(), RESULTS_SECTION, body))


def report(name=None) -> str:
    """From the single complete attempt per arm with no unresolved failures (Z1); writes the X8-labelled
    section into results/TEST.md and gpu1bit/RESULTS.md (real mode only; fake mode returns it)."""
    runtime, empty, disclosure = collect(name)
    import binary_test_paths as tp
    nemo = {}
    for model, d in (("M1", tp.M1_TEST), ("MB1", NEMO_MB1_TEST)):
        doc = json.loads((d / "summary.json").read_text())
        nemo[model] = {k: float(v["wer"]) for k, v in doc["sets"].items()}   # percent, as recorded
    md = report_markdown(runtime, nemo, empty)
    lines = ["", "### Attempts (plans/x2-default.md Y3)", ""]
    for arm, d in disclosure.items():
        att = "; ".join(f"attempt {a['attempt']} {a['status']}" + (f" ({a['reason']})" if a["reason"] else "")
                        for a in d["attempts"])
        shared = (f"; {d['shared']['utterances']} shared utterances, {d['shared']['differing']} with differing text"
                  if d["shared"] else "")
        lines.append(f"- {ARMS[arm]['label']}: {att}{shared}")
    md = md.rstrip() + "\n" + "\n".join(lines) + "\n"
    if not name:
        write_sections(md)
    return md


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    s = sub.add_parser("score")
    s.add_argument("--arm", required=True)
    s.add_argument("--new-attempt", type=Path, default=None)
    s.add_argument("--limit", type=int, default=None, help="refused: smoke tests use dev only")
    s = sub.add_parser("worker")
    s.add_argument("--arm", required=True)
    s.add_argument("--attempt", type=int, required=True)
    s.add_argument("--set", required=True)
    s.add_argument("--retry", default=None)
    s = sub.add_parser("summarize")
    s.add_argument("--arm", required=True)
    s = sub.add_parser("record-failure")
    s.add_argument("--arm", required=True)
    s.add_argument("--evidence", required=True)
    s = sub.add_parser("precheck")
    s.add_argument("--arm", required=True)
    s.add_argument("--attempt", type=int, default=None)
    s.add_argument("--evidence", type=Path, default=None)
    s = sub.add_parser("pin")
    s.add_argument("--arm", required=True)
    s.add_argument("--attempt", type=int, required=True)
    s.add_argument("--benchmark-run", required=True)
    s.add_argument("--gates-run", required=True)
    sub.add_parser("report")
    args = p.parse_args()
    if args.action == "score":
        if args.limit is not None:
            raise SystemExit("--limit is refused for test scoring; smoke tests use dev only")
        print(json.dumps({k: v for k, v in score(args.arm, args.new_attempt).items() if k in ("arm", "attempt")}))
    elif args.action == "worker":
        sys.exit(worker(args.arm, args.attempt, args.set, args.retry.split(",") if args.retry else None, fake_name()))
    elif args.action == "summarize":
        recover_summary(args.arm)
    elif args.action == "record-failure":
        record_failure(args.arm, args.evidence)
    elif args.action == "precheck":
        ev = json.loads(args.evidence.read_text()) if args.evidence else None
        sys.exit(0 if precheck(args.arm, args.attempt, ev)["pass"] else 1)
    elif args.action == "pin":
        print(json.dumps(pin(args.arm, args.attempt, args.benchmark_run, args.gates_run), indent=1))
    else:
        print(report(fake_name()))


if __name__ == "__main__":
    main()
