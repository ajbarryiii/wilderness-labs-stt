"""A fake arm process for the runner's end-to-end test (tests/test_e2e.py): it accepts the real `block`
arguments (runner.block_argv) and emits the bench protocol's JSON events with the same timing rules
(load, warmup, storage audit, then per clip: clip load, one untimed replay, sleep S + G, exactly N
replays, sleep G + S', and only then window_start / window_end), on the runner's clock. "Work" is
logged to $CPU1BIT_FAKE_WORKLOG for the fake powermetrics producer; nothing burns CPU.

    python3 -I fake_block.py block --clips 0,1 --replays 8,8 --guard-s 1.0 --settle-s 1.2 --settle-post-s 1.0 ...
"""
import json
import os
import sys
import time

CLK = getattr(time, "CLOCK_UPTIME_RAW", None)
if CLK is None:
    CLK = time.CLOCK_MONOTONIC_RAW
REPLAY_S = 0.25


def now():
    return time.clock_gettime_ns(CLK)


def emit(d):
    sys.stdout.write(json.dumps(d) + "\n")
    sys.stdout.flush()


def main():
    args = sys.argv[1:]
    assert args[0] == "block", args
    kv = dict(zip(args[1::2], args[2::2]))
    clips = kv["--clips"].split(",")
    replays = [int(x) for x in kv["--replays"].split(",")]
    g, s_pre, s_post = float(kv["--guard-s"]), float(kv["--settle-s"]), float(kv["--settle-post-s"])
    log = open(os.environ["CPU1BIT_FAKE_WORKLOG"], "a")

    def work(seconds):
        t0 = now()
        log.write(f"S {t0}\n")
        log.flush()
        time.sleep(seconds)
        t1 = now()
        log.write(f"E {t1}\n")
        log.flush()
        return t0, t1

    emit({"event": "start", "t_ns": now(), "pid": os.getpid(), "kernel": "fake", "args": kv})
    emit({"event": "qos", "t_ns": now(), "records": [], "ok": True, "plumbing": False})
    work(0.3)
    emit({"event": "loaded", "t_ns": now()})
    work(1.0)
    emit({"event": "warmup_done", "t_ns": now()})
    emit({"event": "storage", "t_ns": now(), "ok": True, "reasons": [], "weight_bytes": 72_400_000})
    for wi, (clip, n) in enumerate(zip(clips, replays)):
        work(0.2)                                   # clip load
        work(REPLAY_S)                              # one untimed replay
        time.sleep(s_pre + g)
        t0, t1 = work(n * REPLAY_S)                 # exactly N replays
        time.sleep(g + s_post)
        emit({"event": "window_start", "window": wi, "clip": clip, "id": f"c{clip}", "T": 200, "t_ns": t0})
        emit({"event": "window_end", "window": wi, "t_ns": t1, "replays": n, "replay_ns": [(t1 - t0) // n] * n,
              "gemm_calls": 240 * n, "quantizations": 192 * n})
    emit({"event": "qos_end", "t_ns": now(), "records": [], "ok": True})
    emit({"event": "end", "t_ns": now()})


if __name__ == "__main__":
    main()
