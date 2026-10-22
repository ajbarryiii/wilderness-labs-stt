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
Duration budget (X7; Codex x2 c3): compute_budget takes the gate run's measured batch-1 throughput of the arm
only over the utterances and audio seconds its recorded chunk timings cover (at least half of dev, the arm's
identity), extrapolates it to the test sets (per audio second and per utterance, the larger kept), adds a cold
start-up and capture per chunk, and sets the scoring job's timeout to twice that. The record
(H/x2/budget-<gate run>-<arm>.json) is written once, pinned by testscore's qualification, and the scoring job
command takes its timeout from it (`testscore job-args`).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

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


MIN_COVERAGE = 0.5


def compute_budget(chunks: list[dict], identity: str, dev_utterances: int, dev_audio_s: float, test_utterances: int,
                   test_audio_s: float, startup_s: float, capture_s: float) -> dict:
    """X7's job duration budget (pure; Codex x2 c3). Throughput is taken only over what recorded chunk timings
    cover: the chunks of `identity` that ended without an error, their rows and their processed audio seconds
    (fulldev chunk records "rows" and "audio_s"). A killed worker keeps its rows but writes no chunk, so its
    share is excluded, never divided over. The covered share must be at least MIN_COVERAGE of dev in
    utterances and in audio seconds, or ValueError. Per audio second and per utterance, each extrapolated to
    the test sets, the larger kept; plus a cold start-up and capture per chunk the work budget implies; the
    job timeout is twice that."""
    import math
    if dev_utterances <= 0 or dev_audio_s <= 0 or test_utterances <= 0 or test_audio_s <= 0:
        raise ValueError("the duration budget needs the dev and test totals")
    other = [c.get("identity") for c in chunks if c.get("identity") != identity]
    if other:
        raise ValueError(f"chunk timings of another identity {str(other[0])[:12]} (expected {identity[:12]})")
    timed = [c for c in chunks if not c.get("error") and c.get("work_s") is not None]
    if any("audio_s" not in c for c in timed):
        raise ValueError("a chunk record has no audio_s: its covered audio is unknown")
    rows = sum(sum(c["rows"].values()) for c in timed)
    audio = sum(sum(c["audio_s"].values()) for c in timed)
    if rows < MIN_COVERAGE * dev_utterances or audio < MIN_COVERAGE * dev_audio_s:
        raise ValueError(f"timing coverage {rows}/{dev_utterances} utterances, {audio:.0f}/{dev_audio_s:.0f} s is below "
                         f"{MIN_COVERAGE:.0%} of dev; no budget")
    work = float(sum(c["work_s"] for c in timed))
    per_s, per_utt = work / audio, work / rows
    predicted_work = max(per_s * test_audio_s, per_utt * test_utterances)
    n_chunks = math.ceil(predicted_work / WORK_BUDGET_S)
    predicted = predicted_work + n_chunks * (startup_s + capture_s)
    return {"identity": identity, "timed_chunks": len(timed), "covered_utterances": rows, "covered_audio_s": audio,
            "dev_utterances": dev_utterances, "dev_audio_s": dev_audio_s, "covered_work_s": work,
            "per_audio_second_s": per_s, "per_utterance_s": per_utt, "test_utterances": test_utterances,
            "test_audio_s": test_audio_s, "predicted_work_s": predicted_work, "chunks": n_chunks,
            "startup_capture_s": startup_s + capture_s, "predicted_s": predicted, "timeout_s": int(2 * predicted)}


def budget_path(gate_run: str, arm: str) -> Path:
    return env.ARTIFACTS / "x2" / f"budget-{gate_run}-{arm}.json"


def preflight_path(arm: str) -> Path:
    return env.ARTIFACTS / "x2" / f"preflight-{arm}.json"


def write_once(path: Path, doc: dict) -> dict:
    """A record written once: the same content again is kept; different content is refused (Codex x2 c3)."""
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise SystemExit(f"{path} is written once; a different budget is refused (delete nothing, ask)")
        return doc
    env.require_mount()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)
    return doc


def budget(arm: str, gate_run: str) -> dict:
    """compute_budget from the gate run's chunks of `arm` (the g1 / g4 build; their identity is the gate run's
    frozen identity and must equal the preflight's), the full dev and test manifests' utterance counts and
    durations, and the arm's preflight record; written once per gate run and arm (budget_path)."""
    from gpu1bit.actprec import parts
    from gpu1bit.shared import fulldev, seldoc
    from gpu1bit.x2 import testscore
    seldoc.use_candidate()                                 # the gate run's arms are candidate arms (c4)
    parts.install()
    pf = json.loads(preflight_path(arm).read_text())
    frozen = fulldev.check_identity(gate_run, arm, "full")["hash"]
    if frozen != pf["identity"]:
        raise SystemExit(f"the gate run's {arm} identity {frozen[:12]} is not the preflight's {pf['identity'][:12]}")
    chunks = [json.loads(l) for l in fulldev.chunks_path(gate_run, arm, "full").read_text().splitlines() if l.strip()]
    dev = [r for n in fulldev.DEV_SETS for _, r in fulldev.subset_records(n, "full")]
    test = [r for n in testscore.test_sets(None) for r in testscore.records(n, None)]
    out = compute_budget(chunks, frozen, len(dev), sum(r["duration"] for r in dev), len(test),
                         sum(r["duration"] for r in test), pf["startup_s"], pf["capture_s"])
    out.update(arm=arm, gate_run=gate_run, preflight_identity=pf["identity"], lease_bytes=pf["lease_bytes"])
    return write_once(budget_path(gate_run, arm), out)


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
    seldoc.use_candidate()                                 # the test arms are candidate arms (c4)
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
    env.artifacts("x2")
    preflight_path(arm).write_text(json.dumps(out, indent=1) + "\n")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=["MB1-B-hybrid-x2", "M1-x2"])
    p.add_argument("--budget-from-gate-run", default=None,
                   help="CPU: the job duration budget from this gate run's chunks and the preflight record")
    args = p.parse_args()
    from gpu1bit.shared import seldoc
    seldoc.use_candidate()
    if args.budget_from_gate_run:
        out = budget(args.arm, args.budget_from_gate_run)
        print(json.dumps({k: out[k] for k in ("arm", "predicted_s", "chunks", "timeout_s", "lease_bytes")}))
        return
    out = run(args.arm)
    print(json.dumps({k: out[k] for k in ("arm", "pass", "peak_reserved_bytes", "lease_bytes", "longest_utterance_s")}))


if __name__ == "__main__":
    main()
