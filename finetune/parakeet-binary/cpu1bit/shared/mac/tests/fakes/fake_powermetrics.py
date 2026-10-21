"""A fake `powermetrics --samplers cpu_power -i 100 -f plist` for the runner's end-to-end test
(tests/test_e2e.py). It writes NUL-terminated plist records like the real producer: contiguous
intervals of 100-110 ms on a fixed schedule from its own start (after the spawn: A2), `elapsed_ns`,
an integer-mJ `cpu_energy` from a power model, the 1 s wall-clock `timestamp`, and cluster residency.
Power: IDLE_W, plus WORK_W while the fake block process reports work in WORKLOG ("S <t_ns>" / "E <t_ns>"
lines on the same clock). It exits when the reader closes the pipe.

    python3 -I fake_powermetrics.py WORKLOG [SEED]
"""
import datetime
import plistlib
import random
import sys
import time

CLK = getattr(time, "CLOCK_UPTIME_RAW", None)
if CLK is None:
    CLK = time.CLOCK_MONOTONIC_RAW
IDLE_W, WORK_W = 0.4, 6.0


def now():
    return time.clock_gettime_ns(CLK)


def work_intervals(path):
    ivs, start = [], None
    try:
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) != 2:
                    continue
                if parts[0] == "S":
                    start = int(parts[1])
                elif parts[0] == "E" and start is not None:
                    ivs.append((start, int(parts[1])))
                    start = None
    except FileNotFoundError:
        pass
    if start is not None:
        ivs.append((start, None))
    return ivs


def overlap(ivs, a, b):
    tot = 0
    for s, e in ivs:
        e = b if e is None else e
        tot += max(0, min(b, e) - max(a, s))
    return tot


def main():
    worklog = sys.argv[1]
    rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    a = now()
    wall0 = time.time() - (now() - a) / 1e9
    t_start = a
    out = sys.stdout.buffer
    while True:
        el = 100_000_000 + rng.randrange(0, 10_000_000)
        b = a + el
        while True:
            d = b - now()
            if d <= 0:
                break
            time.sleep(d / 1e9)
        w = overlap(work_intervals(worklog), a, b)
        e_mj = IDLE_W * el / 1e6 + WORK_W * w / 1e6
        busy = 0.9 * w / el
        wall = datetime.datetime.fromtimestamp(int(wall0 + (b - t_start) / 1e9), datetime.timezone.utc).replace(tzinfo=None)
        rec = {"elapsed_ns": el, "is_delta": True, "timestamp": wall,
               "processor": {"cpu_energy": int(round(e_mj)), "cpu_power": e_mj / (el / 1e9),
                             "combined_power": e_mj / (el / 1e9),
                             "clusters": [{"name": "E-Cluster", "idle_ratio": 0.9, "freq_hz": 1e9,
                                           "cpus": [{"cpu": 0, "idle_ratio": 0.9}, {"cpu": 1, "idle_ratio": 0.9}]},
                                          {"name": "P0-Cluster", "idle_ratio": 1 - busy, "freq_hz": 3e9,
                                           "cpus": [{"cpu": c, "idle_ratio": 1 - busy} for c in range(2, 6)]},
                                          {"name": "P1-Cluster", "idle_ratio": 1 - busy, "freq_hz": 3e9,
                                           "cpus": [{"cpu": c, "idle_ratio": 1 - busy} for c in range(6, 10)]}]}}
        try:
            out.write(plistlib.dumps(rec) + b"\0")
            out.flush()
        except (BrokenPipeError, OSError, ValueError):
            return
        a = b


if __name__ == "__main__":
    main()
