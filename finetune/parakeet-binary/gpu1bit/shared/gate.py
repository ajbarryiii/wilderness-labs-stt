"""A12 recognition gate (plans/gpu-kernel.md A12, frozen criteria): candidate vs MB1 on R-expanded.

Criteria (all must hold; otherwise the candidate is a quality trade-off, not a winner):
  encoder   on EVERY gate utterance, i.e. the union of the 512-utterance dev sample and the timing
            workloads (Codex c1 finding 4): candidate output finite everywhere; output lengths equal
            the reference's; on valid frames only per-utterance ||E - E_ref|| / ||E_ref|| <= 1e-3
            (maximum absolute difference reported); no decoding error
  sample    on the frozen 512-utterance dev sample: token-ID sequences differ on at most 0.5% of
            utterances (<= 2 of 512), and pooled dev-sample WER within 0.02 points of the reference's
  timing    dev timing workloads (dev_clips and dev_24, A13): text, token IDs and timestamps
            identical, through the full pipeline with CUDA graphs (as P-paired runs it)

RECORD BINDING (Codex c1 finding 2). Records live in ARTIFACTS_ROOT/<agent>/gate/<run>/<impl>/ and
carry a fingerprint (gpu1bit.shared.fingerprint: every source a run can import, both exports, the
workloads, package versions, the implementation, the reference and the patch modules). The first
process of a run freezes it in fingerprint.json; every later process recomputes it and refuses a
mismatch; a records file holding rows of another fingerprint is refused, never resumed; every row
also records the GPU identity (device, driver), which must be the same for all rows. `verdict`
always recomputes from the current run's rows (an existing verdict.json is never read; it is output).

Parts run in separate processes, each <= 10 min with one lock (A9), appending one JSON line per
utterance (resumable; done ids of the current fingerprint are skipped):
    timeout 600 gpu1bit/shared/python -m gpu1bit.shared.gate sample --agent H --run R --impl NAME
    timeout 600 gpu1bit/shared/python -m gpu1bit.shared.gate timing --agent H --run R --impl NAME
    gpu1bit/shared/python -m gpu1bit.shared.gate verdict --agent H --run R --impl NAME     (CPU)
Patched variants (e.g. B-hybrid) are run through gpu1bit.shared.dispatch with --patches, whose
module names are part of the fingerprint.
Sample part: eager transformed encoders (max_graphs=0; the same arithmetic the graphs replay),
batch 1, then each model's own fused decoder; both models see identical features. Timing part: each
model's own full pipeline (pipeline graph cache, cleared per utterance), encoder output taken from
model.forward and compared with the reference's.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path

import torch

from gpu1bit.shared import env, fingerprint as F

REFERENCE = "R-expanded"
ENCODER_REL_LIMIT = 1e-3
MISMATCH_FRACTION = 0.005
WER_POINTS = 0.02


def gate_dir(agent: str, run: str, impl: str) -> Path:
    if not run:
        raise ValueError("a gate run name is required (records are bound to a run)")
    env.require_mount()
    d = env.ARTIFACTS_ROOT / agent / "gate" / run / impl.replace(":", "_")
    d.mkdir(parents=True, exist_ok=True)
    return d


def current_fingerprint(impl: str, patches: list[str]) -> dict:
    return F.make("A12 gate", impl=impl, reference=REFERENCE, patches=list(patches))


def bind(agent: str, run: str, impl: str, patches: list[str]) -> dict:
    """Freeze (first call) or verify (later calls) the run's fingerprint; refuse a mismatch."""
    return F.freeze(gate_dir(agent, run, impl) / "fingerprint.json", current_fingerprint(impl, patches))


def read_rows(path: Path, fp_hash: str) -> list[dict]:
    """Rows of the current fingerprint; any row of another fingerprint makes the file unusable."""
    if not path.exists():
        return []
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    foreign = [r.get("id") for r in rows if r.get("fingerprint") != fp_hash]
    if foreign:
        raise ValueError(f"{path}: {len(foreign)} record(s) of another fingerprint (e.g. {foreign[:3]}); refused")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicated records")
    return rows


def _encoder_metrics(e_c, l_c, e_r, l_r) -> dict:
    t = int(l_r[0])
    finite = bool(torch.isfinite(e_c).all())
    lengths_equal = bool(torch.equal(l_r, l_c))
    if t and finite and e_c.shape == e_r.shape:
        diff = e_c[..., :t] - e_r[..., :t]
        rel = (diff.norm() / e_r[..., :t].norm()).item()
        mx = diff.abs().max().item()
    else:
        rel, mx = (0.0, 0.0) if not t else (float("inf"), float("inf"))
    return {"finite": finite, "lengths_equal": lengths_equal, "rel_frobenius_valid": rel, "max_abs_valid": mx}


def _decode(model, encoded, lengths):
    from gpu1bit.shared.bench import decoded_state
    result = model.decoding.rnnt_decoder_predictions_tensor(encoded, lengths, return_hypotheses=False)
    if isinstance(result, tuple):
        result = result[0]
    text = result[0].text if hasattr(result[0], "text") else str(result[0])
    return {"text": text, **decoded_state(model)}


def _worker_setup(args):
    import evaluate as ev
    from nemo.utils import logging
    env.require_shared_wrapper()
    logging.setLevel(logging.ERROR)
    fp = bind(args.agent, args.run, args.impl, args.patches)
    return ev, fp


def sample(args) -> None:
    from gpu1bit.shared import bench, workloads
    ev, fp = _worker_setup(args)
    records = workloads.load()["dev_sample"]
    a, b = (int(v) for v in args.range.split("-"))
    out = gate_dir(args.agent, args.run, args.impl) / "sample.jsonl"
    with env.gpu_lock(f"{args.agent}:A12 gate sample {args.impl} {args.run}") as lock, torch.inference_mode(), \
            contextlib.ExitStack() as stack:
        gpu = F.gpu_identity()
        done = {r["id"] for r in read_rows(out, fp["hash"])}
        todo = [r for r in records[a:b + 1] if r["id"] not in done]
        stack.enter_context(ev.strict_fp32("cuda"))
        ref, _ = stack.enter_context(bench.variant_model(REFERENCE, max_graphs=0))
        cand, _ = stack.enter_context(bench.variant_model(args.impl, max_graphs=0))
        with open(out, "a") as sink:
            for rec in todo:
                if env.check_hold(lock, margin_s=20) > env.LOCK_HOLD_LIMIT_S - 60:
                    break
                audio, length = ev.audio_batch([rec])
                f, fl = ref.preprocessor(input_signal=audio.cuda(), length=length.cuda())
                e_r, l_r = ref.encoder(audio_signal=f, length=fl)
                e_c, l_c = cand.encoder(audio_signal=f, length=fl)
                row = {"id": rec["id"], "corpus": rec["corpus"], "duration": rec["duration"], "part": "sample",
                       "fingerprint": fp["hash"], "gpu": gpu, **_encoder_metrics(e_c, l_c, e_r, l_r)}
                row["reference"] = _decode(ref, e_r, l_r)
                row["candidate"] = _decode(cand, e_c, l_c)
                row["reference"]["score"] = ev.score(rec["text"], row["reference"]["text"])
                row["candidate"]["score"] = ev.score(rec["text"], row["candidate"]["text"])
                sink.write(json.dumps(row) + "\n")
                sink.flush()
            os.fsync(sink.fileno())
    print(json.dumps({"impl": args.impl, "run": args.run, "file": str(out), "done": len(read_rows(out, fp["hash"]))}))


def timing(args) -> None:
    from gpu1bit.shared import bench, workloads
    ev, fp = _worker_setup(args)
    doc = workloads.load()
    records = list({r["id"]: r for r in doc["dev_clips"] + doc["dev_24"]}.values())
    out = gate_dir(args.agent, args.run, args.impl) / "timing.jsonl"
    with env.gpu_lock(f"{args.agent}:A12 gate timing {args.impl} {args.run}") as lock, torch.inference_mode(), \
            contextlib.ExitStack() as stack:
        gpu = F.gpu_identity()
        done = {r["id"] for r in read_rows(out, fp["hash"])}
        stack.enter_context(ev.strict_fp32("cuda"))
        models = {n: stack.enter_context(bench.variant_model(n, max_graphs=4))[0] for n in (REFERENCE, args.impl)}
        with open(out, "a") as sink:
            for rec in records:
                if rec["id"] in done:
                    continue
                env.check_hold(lock)
                audio, length = ev.audio_batch([rec])
                got, enc = {}, {}
                for n, m in models.items():
                    m.forward.entries.clear()
                    e, l = m.forward(input_signal=audio.cuda(), input_signal_length=length.cuda())
                    enc[n] = (e.clone(), l.clone())
                    got[n] = _decode(m, e, l)
                metrics = _encoder_metrics(*enc[args.impl], *enc[REFERENCE])
                row = {"id": rec["id"], "corpus": rec["corpus"], "duration": rec["duration"], "part": "timing",
                       "fingerprint": fp["hash"], "gpu": gpu, **metrics,
                       "identical": got[REFERENCE] == got[args.impl], "reference": got[REFERENCE], "candidate": got[args.impl]}
                sink.write(json.dumps(row) + "\n")
                sink.flush()
            os.fsync(sink.fileno())
    print(json.dumps({"impl": args.impl, "run": args.run, "file": str(out), "done": len(read_rows(out, fp["hash"]))}))


def compute_verdict(agent: str, run: str, impl: str, patches: list[str] | None = None, *, check_sources: bool = True) -> dict:
    """The A12 verdict from the run's own records (never from an existing verdict.json)."""
    from gpu1bit.shared import workloads
    d = gate_dir(agent, run, impl)
    frozen = json.loads((d / "fingerprint.json").read_text()) if (d / "fingerprint.json").exists() else None
    problems = []
    if frozen is None:
        problems.append("no fingerprint.json: no records of this run")
        fp_hash = None
    else:
        fp_hash = frozen["hash"]
        if check_sources:
            now = current_fingerprint(impl, patches if patches is not None else frozen["extra"]["patches"])
            if now["hash"] != fp_hash:
                problems.append(f"sources/exports/workloads changed since the gate ran: {F.diff(frozen, now)[:8]}")
    doc = workloads.load()
    sample_rows = {r["id"]: r for r in read_rows(d / "sample.jsonl", fp_hash)} if fp_hash else {}
    timing_rows = {r["id"]: r for r in read_rows(d / "timing.jsonl", fp_hash)} if fp_hash else {}
    want_sample = [r["id"] for r in doc["dev_sample"]]
    want_timing = list({r["id"]: r for r in doc["dev_clips"] + doc["dev_24"]})
    missing = [i for i in want_sample if i not in sample_rows] + [i for i in want_timing if i not in timing_rows]
    gpus = {json.dumps(r.get("gpu"), sort_keys=True) for r in list(sample_rows.values()) + list(timing_rows.values())}
    if len(gpus) > 1:
        problems.append(f"records from {len(gpus)} different GPU identities")
    rows = [sample_rows[i] for i in want_sample if i in sample_rows]
    union = rows + [timing_rows[i] for i in want_timing if i in timing_rows]

    def pooled(side):
        scored = [r[side]["score"] for r in rows if r[side]["score"]["scored"]]
        words = sum(len(s["ref_norm"].split()) for s in scored)
        return 100.0 * sum(s["S"] + s["D"] + s["I"] for s in scored) / max(words, 1), words
    wer_ref, words = pooled("reference") if rows else (0.0, 0)
    wer_cand, _ = pooled("candidate") if rows else (0.0, 0)
    mismatches = [r["id"] for r in rows if r["reference"]["tokens"] != r["candidate"]["tokens"]]
    encoder_bad = [r["id"] for r in union if not (r.get("finite") is True and r.get("lengths_equal") is True
                                                    and r.get("rel_frobenius_valid", float("inf")) <= ENCODER_REL_LIMIT)]
    timing_bad = [i for i in want_timing if i in timing_rows and not timing_rows[i]["identical"]]
    result = {
        "impl": impl, "run": run, "reference": REFERENCE, "fingerprint": fp_hash, "problems": problems,
        "complete": not missing, "missing": missing[:20], "missing_count": len(missing),
        "encoder": {"ok": not encoder_bad and bool(union), "failing": encoder_bad, "utterances": len(union),
                    "scope": "union of the dev sample and the timing workloads",
                    "max_rel_frobenius_valid": max((r.get("rel_frobenius_valid", float("inf")) for r in union), default=None),
                    "max_abs_valid": max((r.get("max_abs_valid", float("inf")) for r in union), default=None)},
        "sample": {"utterances": len(rows), "token_mismatches": len(mismatches), "mismatch_ids": mismatches,
                   "mismatch_limit": int(MISMATCH_FRACTION * len(want_sample)),
                   "wer_reference": wer_ref, "wer_candidate": wer_cand, "wer_delta_points": wer_cand - wer_ref,
                   "scored_words": words,
                   "ok": bool(rows) and len(mismatches) <= MISMATCH_FRACTION * len(want_sample) and abs(wer_cand - wer_ref) <= WER_POINTS},
        "timing": {"utterances": len(timing_rows), "not_identical": timing_bad, "ok": bool(timing_rows) and not timing_bad},
    }
    result["passes"] = (not problems and result["complete"] and result["encoder"]["ok"] and result["sample"]["ok"]
                        and result["timing"]["ok"])
    tmp = d / f"verdict.json.tmp{os.getpid()}"
    tmp.write_text(json.dumps(result, indent=1) + "\n")
    os.replace(tmp, d / "verdict.json")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("part", choices=["sample", "timing", "verdict"])
    p.add_argument("--agent", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--impl", required=True)
    p.add_argument("--range", default="0-511")
    p.add_argument("--patches", default="", help="comma-separated patch modules imported by gpu1bit.shared.dispatch")
    args = p.parse_args()
    args.patches = [m for m in args.patches.split(",") if m]
    if args.part == "verdict":
        print(json.dumps(compute_verdict(args.agent, args.run, args.impl, args.patches or None), indent=1))
    else:
        {"sample": sample, "timing": timing}[args.part](args)


if __name__ == "__main__":
    main()
