"""Memory and duration preflight of the test arms (plans/x2-default.md Y6, Z3; agent H). One process per arm
(g1 and g4 builds, under the candidate selection), in this order, with full decoding:
  1. four distinct near-maximum shapes, 104, 100, 96 and 92 s, which fill the 4-entry pipeline graph cache;
  2. the 105.7 s maximum, which is not cached, on the eager fallback while all four graphs are held.
The cache must hold 4 entries before the fifth input, and the fifth input's route must be eager (no new entry;
the eager fallback is called once). Otherwise the preflight fails.
Recorded: the peak reserved VRAM of the process (lease = peak + max(25%, 1 GiB)), the cold start-up and
capture times, and the longest single-utterance time. Time check: cold start-up + capture + 2 x longest
within the worker's hold cap, and capture + 2 x longest within R3's work budget (at least one utterance per
chunk at 2x margin). If it fails, scoring does not start and the chunk budget is replanned under review.
Inputs are dev audio (dev_24 utterances concatenated, cut to the exact sample count).

    gpu1bit/shared/python -m gpu1bit.x2.preflight --arm MB1-B-hybrid-x2   (job --needs gpu:shared --vram <measured>)
    gpu1bit/shared/python -m gpu1bit.x2.preflight --arm MB1-B-hybrid-x2 --budget-from-gate-run G   (CPU, after the gates)
Duration budget (X7): compute_budget extrapolates the gate run's measured batch-1 full-dev work time of the arm
to the test sets (per audio second and per utterance, the larger kept), adds a cold start-up and capture per
chunk, and sets the scoring job's timeout to twice that.
"""
from __future__ import annotations

import argparse
import json
import time

from gpu1bit.shared import env

CACHED_SECONDS = (104.0, 100.0, 96.0, 92.0)
EAGER_SECONDS = 105.7
SAMPLE_RATE = 16000
HOLD_CAP_S, WORK_BUDGET_S = 540.0, 480.0


def lease_bytes(peak: int) -> int:
    return int(peak + max(0.25 * peak, 2**30))


def time_ok(startup_s: float, capture_s: float, longest_s: float) -> dict:
    hold = startup_s + capture_s + 2 * longest_s
    work = capture_s + 2 * longest_s
    return {"ok": hold <= HOLD_CAP_S and work <= WORK_BUDGET_S, "startup_capture_2x_longest_s": hold,
            "capture_2x_longest_s": work, "hold_cap_s": HOLD_CAP_S, "work_budget_s": WORK_BUDGET_S}


def compute_budget(work_s: list[float], dev_utterances: int, dev_audio_s: float, test_utterances: int,
                   test_audio_s: float, startup_s: float, capture_s: float) -> dict:
    """X7's job duration budget (pure): the gate run's full-dev work time of the arm (its worker chunks'
    work_s, batch 1) per audio second and per utterance, each extrapolated to the test sets and the larger
    kept; plus a cold start-up and capture (this preflight) for each chunk the work budget implies. The job
    timeout is twice that."""
    import math
    if not work_s or dev_utterances <= 0 or dev_audio_s <= 0 or test_utterances <= 0 or test_audio_s <= 0:
        raise ValueError("the duration budget needs the gate run's chunk work times and the dev and test totals")
    work = float(sum(work_s))
    per_s, per_utt = work / dev_audio_s, work / dev_utterances
    predicted_work = max(per_s * test_audio_s, per_utt * test_utterances)
    chunks = math.ceil(predicted_work / WORK_BUDGET_S)
    predicted = predicted_work + chunks * (startup_s + capture_s)
    return {"dev_work_s": work, "dev_utterances": dev_utterances, "dev_audio_s": dev_audio_s,
            "per_audio_second_s": per_s, "per_utterance_s": per_utt, "test_utterances": test_utterances,
            "test_audio_s": test_audio_s, "predicted_work_s": predicted_work, "chunks": chunks,
            "startup_capture_s": startup_s + capture_s, "predicted_s": predicted, "timeout_s": int(2 * predicted)}


def budget(arm: str, gate_run: str) -> dict:
    """compute_budget from the gate run's chunks of `arm` (the g1 / g4 build), the full dev and test
    manifests' utterance counts and durations, and this arm's preflight record; written next to it."""
    from gpu1bit.shared import fulldev
    from gpu1bit.x2 import testscore
    pf = json.loads((env.artifacts("x2") / f"preflight-{arm}.json").read_text())
    chunks = [json.loads(l) for l in fulldev.chunks_path(gate_run, arm, "full").read_text().splitlines() if l.strip()]
    dev = [r for n in fulldev.DEV_SETS for _, r in fulldev.subset_records(n, "full")]
    test = [r for n in testscore.test_sets(None) for r in testscore.records(n, None)]
    out = compute_budget([c["work_s"] for c in chunks], len(dev), sum(r["duration"] for r in dev), len(test),
                         sum(r["duration"] for r in test), pf["startup_s"], pf["capture_s"])
    out.update(arm=arm, gate_run=gate_run, preflight_identity=pf["identity"], lease_bytes=pf["lease_bytes"])
    (env.artifacts("x2") / f"budget-{arm}.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def fifth_route(cache, call) -> dict:
    """Assert the cache holds max_graphs (4) entries, then run `call` (the fifth input) and assert its route
    was eager: no new entry and exactly one eager fallback call."""
    before = len(cache.entries)
    if before != 4:
        raise AssertionError(f"the cache holds {before} entries before the fifth input, expected 4 entries")
    original = cache.original_forward
    calls = []

    def counting(*a, **k):
        calls.append(1)
        return original(*a, **k)
    cache.original_forward = counting
    try:
        call()
    finally:
        cache.original_forward = original
    after = len(cache.entries)
    if after != before or len(calls) != 1:
        raise AssertionError(f"the fifth input's route is not eager (entries {before} -> {after}, "
                             f"{len(calls)} original_forward calls)")
    return {"route": "eager", "entries": after}


def dev_audio(seconds: float):
    """Dev audio of exactly `seconds` (dev_24 utterances concatenated in order, cut)."""
    import numpy as np
    from gpu1bit.shared import fulldev, workloads
    need = int(round(seconds * SAMPLE_RATE))
    parts, have = [], 0
    for rec in workloads.load()["dev_24"]:
        clip, _ = fulldev.load_clip(rec)
        parts.append(clip)
        have += len(clip)
        if have >= need:
            break
    return np.concatenate(parts)[:need].astype("float32")


def run(arm: str) -> dict:
    import torch
    import evaluate as ev
    from nemo.utils import logging
    from gpu1bit.actprec import parts
    from gpu1bit.shared import fulldev, seldoc
    from gpu1bit.x2 import testscore
    logging.setLevel(logging.ERROR)
    env.require_shared_wrapper()
    import os
    os.environ.setdefault("GPU1BIT_SELECTION", str(seldoc.CANDIDATE))
    with env.gpu_lock(f"H:x2 preflight {arm}") as lock:
        t0 = time.monotonic()
        parts.install()
        spec = fulldev.parse_arm(arm)
        with torch.inference_mode():
            model, _ = parts.build(spec["model"], spec["parts"], kernel=spec["kernel"] or "int8_mma")
        startup = time.monotonic() - t0
        cache = model.forward
        rows, capture = [], 0.0

        def one(seconds):
            clip = dev_audio(seconds)
            x = torch.from_numpy(clip).unsqueeze(0).cuda()
            n = torch.tensor([len(clip)], dtype=torch.long).cuda()
            t = time.monotonic()
            with torch.inference_mode(), ev.strict_fp32("cuda"), ev.inference_settings(model):
                enc, enc_len = model.forward(input_signal=x, input_signal_length=n)
                model.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc, encoded_lengths=enc_len, return_hypotheses=False)
            torch.cuda.synchronize()
            dt = time.monotonic() - t
            return {"seconds": seconds, "wall_s": dt, "enc_len": int(enc_len[0]),
                    "expected_len": testscore.expected_encoder_length(model, len(clip)), "finite": bool(torch.isfinite(enc).all())}
        cache.entries.clear()
        for s in CACHED_SECONDS:
            first = one(s)
            capture += first["wall_s"]
            rows.append({**first, "route": "captured", "replay": one(s)})
        route = fifth_route(cache, lambda: rows.append({**one(EAGER_SECONDS), "route": "eager"}))
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_reserved()
        longest = max([r["wall_s"] for r in rows] + [r["replay"]["wall_s"] for r in rows if "replay" in r])
        held = time.monotonic() - lock["acquired"]
        parts.close(model)
    timing = time_ok(startup, capture, longest)
    out = {"arm": arm, "rows": rows, "route": route, "peak_reserved_bytes": peak, "lease_bytes": lease_bytes(peak),
           "startup_s": startup, "capture_s": capture, "longest_utterance_s": longest, "time": timing,
           "lengths_ok": all(r["enc_len"] == r["expected_len"] and r["finite"] for r in rows), "lock_held_s": held,
           "selection_sha256": seldoc.load()["sha256"], "identity": parts.identity(arm, "full")["hash"]}
    out["pass"] = timing["ok"] and out["lengths_ok"]
    (env.artifacts("x2") / f"preflight-{arm}.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=["MB1-B-hybrid-x2", "M1-x2"])
    p.add_argument("--budget-from-gate-run", default=None,
                   help="CPU: the job duration budget from this gate run's chunks and the preflight record")
    args = p.parse_args()
    if args.budget_from_gate_run:
        out = budget(args.arm, args.budget_from_gate_run)
        print(json.dumps({k: out[k] for k in ("arm", "predicted_s", "chunks", "timeout_s", "lease_bytes")}))
        return
    out = run(args.arm)
    print(json.dumps({k: out[k] for k in ("arm", "pass", "peak_reserved_bytes", "lease_bytes", "longest_utterance_s")}))


if __name__ == "__main__":
    main()
