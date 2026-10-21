"""Selection sessions (plans/gpu-kernel.md A12, A14; Codex code review c1 findings 2-5).

A session is defined by a ROSTER frozen before it starts (selection/<run>/roster.json): the variants
(reference R-expanded first), each candidate's stack baseline (information), the patch modules that
register patched variants (imported by gpu1bit.shared.dispatch), the phase pair, and a NEW FREEZE FILE
(ARTIFACTS_ROOT/freeze-<run>.sha256) listing every file fingerprint.source_files() covers (H's shared/,
runtime.py, every file of every candidate directory, M1's inference/, the ternary/binary modules, the
wrappers). `verify` checks that every listed file is unchanged AND that no file was added to a covered
directory; the session runs it before every process.

    gpu1bit/shared/python -m gpu1bit.shared.selection roster --run sel-X --variants R-expanded R-hybrid ... \\
        --patches gpu1bit.int8_mma.b01_run,gpu1bit.int8_mma.session_patch --phase-pair B-hybrid B-hybrid-b01 \\
        --stack-baseline B-hybrid=R-hybrid B-hybrid-b01=R-hybrid
    gpu1bit/shared/python -m gpu1bit.shared.selection session --run sel-X     (spawns one locked
        `timeout 600` dispatch process per chunk; resumable; verifies the freeze before each)
    gpu1bit/shared/python -m gpu1bit.shared.selection report --run sel-X      (CPU)
    gpu1bit/shared/python -m gpu1bit.shared.selection checks --run sel-X      (one locked process per
        step) the A12 gates, A5 audits and deferred-A3 qualifications of the roster under the CURRENT
        source state, in check run `checks_run_name` = sel-X-chk-<fingerprint hash>: a source change
        gives a new name, so records of another source state are never reused (Codex GPU c3).
    gpu1bit/shared/python -m gpu1bit.shared.selection regenerate --run sel-X  (CPU)
        a corrected decision from the session's RECORDED timing (binding re-verified: verify_timing_binding)
        and the checks of the current source state; writes report-corrected.json, report.json stays as
        recorded.
Decisions (report.json, report-corrected.json) are written only without provenance problems
(`write_decision`): stale, missing or partial check records refuse the write and leave an accepted
decision untouched. Eligibility has ONE validator for both paths (`evaluate_eligibility`, with
`deferred_a3_check` for hybrid-stack candidates; Codex GPU c3).

Session: P-paired encoder and full scope over dev_24 with all variants resident in each chunk process
(balanced forward/reverse order per window, bench.p_paired); P-clips + P-memory once per variant in
forward order and once in reverse order (fresh process each); the per-phase energy breakdown with
power / SM-clock / memory-clock traces for the phase pair (shared/phases.py); the A12 gate (512 dev
sample + timing workloads, run-scoped and fingerprinted records) for every candidate; the A5 audit for
every variant after warm-up and pipeline-graph capture of the three dev clips.

ELIGIBILITY (report): the A12 verdict RECOMPUTED from this run's fingerprinted gate records
(gate.compute_verdict; an existing verdict.json is never trusted; Codex c1 finding 2), whose encoder
criteria cover the union of the dev sample and the timing workloads (finding 4); the A5 audit within
budget; and the P-paired encoder checks against R-expanded on all 24 dev utterances (finite, equal
lengths, rel. Frobenius <= 1e-3 on valid frames; finding 4).

WINNER RULE (A14; its multi-candidate tie handling is plan amendment A15). The rule text:
among candidates that pass A12, the lowest median paired
change in GPU energy per full transcription on the dev 24-utterance set; if two candidates differ by
less than 1% or neither is lower on at least 16 of 24 utterances, the lower median full-transcription
latency decides.
Implementation, corrected twice:
- Codex c1 finding 3: sel-20261020 ranked each candidate against its own stack baseline; now every
  candidate is ranked against ONE common reference, R-expanded, and "differ by less than 1%" / "lower on
  at least 16 of 24" are evaluated on the paired candidate-vs-candidate change per utterance,
  (E_a - E_b) / E_b.
- Codex GPU code review c2 (2026-10-21): A14's text is written for two candidates, and the c1
  implementation compared only the top two by energy, so a third candidate that also ties with the
  first was never considered. This is resolved by plan amendment A15 (plans/gpu-kernel.md), a
  RETROSPECTIVE amendment written after sel-20261020b's results were known (Codex GPU c3): the frozen
  GPU implementation was top-two. A15 adapts the anchored tie set the CPU plan already used
  (plans/cpu-benchmark.md review round 2; full text in review/cpu-benchmark/p4.log, 2026-10-20 08:43)
  to A14's tie definition; that earlier text shows the method existed, not a prior commitment for the
  GPU. A15 changes the sel-20261020b winner from B-hybrid-b01 to B-hybrid, and is frozen (the plan is
  in every fingerprint) before any new measurement:
  1. A* = the eligible candidate with the lowest (median full-scope energy change vs R-expanded, name).
  2. S = A* plus every eligible candidate whose head-to-head energy with A* is an A14 tie (`a14_tie`).
     The set is anchored on A*, not closed under ties (pairwise ties are nontransitive).
  3. Within S, the lowest median full-transcription latency change vs R-expanded wins; if exactly
     equal, the candidate lower in the head-to-head latency comparison against every other equal one;
     then the name that sorts first. If S = {A*}, A* wins by energy.
  With two candidates this gives the c1 outcome, except that a tie is now decided by latency vs
  R-expanded first (A14's "the lower median full-transcription latency") and head to head second.
  Applied to sel-20261020 (`amend-legacy`), B-hybrid still wins by energy (-3.99% head to head, 23/24).

CODEX REVIEW ITEM 2 (all variants resident in each P-paired process). This is M1's own paired
method (cuda_experiments.multi_encoder / multi_pipeline; CUDA_IMPROVEMENTS.md "Both models stay
resident, so this process is not used for deployment VRAM"). It is acceptable for latency and GPU
energy because each 1 s window executes exactly one variant's work with every other model idle: no
concurrent kernels, the same CUDA context, the same clocks and thermal history, and windows of all
variants interleave per utterance in balanced forward/reverse order, so drift is shared rather than
attributed. Resident but idle allocations draw no extra board power (all GDDR7 is refreshed
regardless of allocation) and do not occupy L2 during another variant's window; the previous
variant's L2 lines are displaced by the 0.3 s warm-up that precedes every window. It is NOT
acceptable for VRAM: NVML process memory is the sum over all resident models, graph pools and
caching-allocator reservations, so per-variant memory comes only from P-memory (one fresh process per
variant, sampled every 10 ms from before loading).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

from gpu1bit.shared import env, fingerprint as F

REFERENCE = "R-expanded"
BASELINES = ("R-expanded", "R-hybrid")
PY = str(env.SHARED / "python")
CHUNKS = ["0-3", "4-7", "8-11", "12-15", "16-19", "20-23"]
ENCODER_REL_LIMIT = 1e-3
TIE_PERCENT, TIE_COUNT = 1.0, 16
LEGACY = {"sel-20261020": {"variants": ["R-expanded", "R-hybrid", "int8_mma", "B-hybrid", "triton_bf16:bf16x3", "bitserial"],
                           "stack_baseline": {"int8_mma": "R-expanded", "B-hybrid": "R-hybrid",
                                              "triton_bf16:bf16x3": "R-expanded", "bitserial": "R-expanded"}}}


def sel_dir(run: str) -> Path:
    env.require_mount()
    d = env.ARTIFACTS / "selection" / run
    d.mkdir(parents=True, exist_ok=True)
    return d


def freeze_path(run: str) -> Path:
    return env.ARTIFACTS_ROOT / f"freeze-{run}.sha256"


def pointer_pin_path(run: str) -> Path:
    """The pointer pin of a freeze (plans/x2-default.md Z2): the SHA-256 of the pre-promotion pointer."""
    return env.ARTIFACTS_ROOT / f"freeze-{run}.pointer.json"


# --- roster and freeze ---------------------------------------------------------------------------

def make_roster(args) -> dict:
    variants = list(args.variants)
    if variants[0] != REFERENCE or len(set(variants)) != len(variants):
        raise SystemExit(f"the roster must start with {REFERENCE} and list each variant once")
    stack = dict(s.split("=", 1) for s in args.stack_baseline)
    candidates = [v for v in variants if v not in BASELINES]
    doc = {"run": args.run, "variants": variants, "reference": REFERENCE, "candidates": candidates,
           "stack_baseline": {c: stack.get(c, REFERENCE) for c in candidates},
           "patches": [m for m in args.patches.split(",") if m], "phase_pair": list(args.phase_pair or []),
           "freeze_file": str(freeze_path(args.run)), "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    path = sel_dir(args.run) / "roster.json"
    if path.exists():
        old = json.loads(path.read_text())
        same = {k: v for k, v in old.items() if k != "frozen_at"} == {k: v for k, v in doc.items() if k != "frozen_at"}
        if not same:
            raise SystemExit(f"{path} is frozen with a different roster")
        doc = old
    else:
        path.write_text(json.dumps(doc, indent=1) + "\n")
    write_freeze(args.run)
    return doc


def write_freeze(run: str) -> dict:
    """The session freeze, written once: every fingerprint source file with its SHA-256, and the pointer pin
    (Z2: the pre-promotion pointer's SHA-256). Verified (SystemExit on any change) and returned."""
    fz = freeze_path(run)
    if not fz.exists():
        from gpu1bit.shared import seldoc
        env.require_mount()
        lines = [f"{env.sha256_file(p)}  {p.relative_to(env.REPO)}" for p in F.source_files()]
        fz.write_text("\n".join(lines) + "\n")
        pointer_pin_path(run).write_text(json.dumps({
            "pointer": str(seldoc.SELECTED), "x3_sha256": env.sha256_file(seldoc.SELECTED),
            "candidate": str(seldoc.CANDIDATE), "rule": "plans/x2-default.md Z2"}, indent=1) + "\n")
    return verify_freeze(run)


def verify_freeze(run: str) -> dict:
    """Every listed file unchanged and no file added to a covered directory; SystemExit otherwise. The
    deployment pointer shared/selected.json is not listed; with a pointer pin (freezes since x2-default
    Z2) it must be the pinned x3 document or a byte-identical copy of the candidate."""
    fz = freeze_path(run)
    listed, bad = set(), []
    for line in fz.read_text().splitlines():
        if line.strip():
            digest, rel = line.split(None, 1)
            rel = rel.strip()
            listed.add(rel)
            p = env.REPO / rel
            if not p.exists() or env.sha256_file(p) != digest:
                bad.append(rel)
    now = {str(p.relative_to(env.REPO)) for p in F.source_files()}
    added = sorted(now - listed)
    result = {"freeze": str(fz), "files": len(listed), "mismatched": bad, "added": added, "ok": not bad and not added}
    if not result["ok"]:
        raise SystemExit(f"freeze verification FAILED: {json.dumps(result)}")
    pin = pointer_pin_path(run)
    if pin.exists():   # Z2: the pointer is checked separately (x3 as pinned, or the promoted candidate)
        from gpu1bit.shared import seldoc
        result["pointer"] = seldoc.pointer_state(json.loads(pin.read_text())["x3_sha256"])
    return result


def load_roster(run: str) -> dict:
    if run in LEGACY:
        legacy = LEGACY[run]
        return {"run": run, "variants": legacy["variants"], "reference": REFERENCE,
                "candidates": [v for v in legacy["variants"] if v not in BASELINES],
                "stack_baseline": legacy["stack_baseline"], "patches": ["gpu1bit.int8_mma.paired"], "phase_pair": [],
                "legacy": True}
    return json.loads((env.ARTIFACTS / "selection" / run / "roster.json").read_text())


# --- A5 audit for any variant --------------------------------------------------------------------

def audit_variant(args) -> None:
    import gc
    import torch
    import evaluate as ev
    from nemo.utils import logging
    from gpu1bit.shared import audit, bench, fixtures, workloads
    env.require_shared_wrapper()
    logging.setLevel(logging.ERROR)
    clips = workloads.load()["dev_clips"]
    scales = {n: w["scale"] for n, w in fixtures.mb1_layers().items()}
    out_dir = sel_dir(args.run) / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    with env.gpu_lock(f"H:A5 audit {args.variant} {args.run}") as lock, torch.inference_mode(), ev.strict_fp32("cuda"):
        with bench.variant_model(args.variant, max_graphs=4) as (model, info):
            for clip in clips:
                audio, length = ev.audio_batch([clip])
                for _ in range(2):  # first call warms and captures the pipeline graph; second replays
                    bench.recognize(model, audio, length)
            torch.cuda.synchronize()
            projection = info.get("projection")
            if not hasattr(projection, "__call__") or isinstance(projection, tuple):
                found = [o for o in gc.get_objects() if type(o).__name__ in ("ExpandedProjection", "PtxCudaProjection")]
                projection = found[0] if len(found) == 1 else found
            entries = len(model.forward.entries) if hasattr(model.forward, "entries") else None
            report = audit.audit(model, list(scales), {"provider": projection}, scales=scales)
            held = env.check_hold(lock, margin_s=0)
    reclassify_decoder_side(report)
    patches = [m for m in (args.patches or "").split(",") if m]
    report.update(variant=args.variant, graph_entries=entries, lock_held_s=held, wrapper=env.wrapper(), patches=patches,
                  fingerprint=audit_fingerprint(args.variant, patches))
    (out_dir / f"{args.variant.replace(':', '_')}.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps({k: report[k] for k in ("variant", "weight_code_bytes", "dense_bytes", "scale_bytes",
                                              "within_budget", "graph_entries")}))


def audit_fingerprint(variant: str, patches: list[str]) -> dict:
    return F.make("A5 audit", variant=variant, patches=list(patches))


def record_bound(record: dict, now: dict) -> bool:
    """A record (A5 audit, deferred A3) made under exactly the current fingerprint `now`."""
    fp = record.get("fingerprint")
    return isinstance(fp, dict) and fp.get("hash") == now["hash"] and not F.diff(fp, now)


DECODER_SIDE_PREFIXES = ("decoder.", "joint.", "decoding")
_TOTAL_KEYS = ("dense_bytes", "activation_bytes", "weight_code_bytes", "scale_bytes", "bias_bytes")


def reclassify_decoder_side(report: dict) -> dict:
    """CODEX REVIEW ITEM 1 / c1 finding 5. Post-process an audit.audit report for stacks whose
    transform contexts are not stored in model._packed_optimization (R-hybrid's experimental_pipeline,
    B-hybrid's hybrid_pipeline, gpu1bit.runtime's hybrid runtime keep them in suspended generator
    frames).

    audit.audit walks, as weight roots, the functions patched onto every module's __dict__. M1's
    decoder_transform / control_transform patch decoder.* and joint.* methods with closures that hold
    the fused decoder's FP32 state: the precomputed first-layer token table [1025, 2560] (10,496,000 B),
    the label-looping computer's state (encoder_output_projected [1, T, 640], batched_hyps transcript
    and timestamps, decoder states, _packed_decoder_keepalive). With R-expanded / provider stacks the
    same objects are reached first through model._packed_optimization's decoder_transform /
    control_transform generators and classified dense by audit.audit itself. With the hybrid stacks
    they are first reached via those patched decoder-side functions and land in "weights" as code.

    Rule: a "code" storage whose first path starts with decoder., joint. or decoding is MOVED to dense:
    its entry leaves report["weights"] and is listed in report["reclassified_decoder_side"] (class
    "dense: decoder-side, reclassified"), its bytes leave weight_code_bytes and are ADDED to dense_bytes,
    and the storage counts move from weights to dense, so every category total stays consistent (the sum
    of all categories is unchanged). report["raw"] keeps the unmodified audit.audit totals, counts and
    weights list. These storages are derived from the dense FP32 decoder/joint weights and decoder
    activations, not from the 264 quantized encoder layers. In sel-20261020 this moved 11,562,815 B for
    R-hybrid and B-hybrid alike (same decoder stack), 0 B for the other variants; the adjusted B-hybrid
    code storage is exactly MB1's 75,497,472 packed code bytes.
    """
    from gpu1bit.shared import audit
    if "raw" in report:
        return report
    report["raw"] = {k: report.get(k) for k in _TOTAL_KEYS} | {"storages": dict(report.get("storages", {})),
                                                              "weights": copy.deepcopy(report.get("weights", []))}
    moved = [w for w in report["weights"] if w["class"] == "code" and w["path"].startswith(DECODER_SIDE_PREFIXES)]
    moved_ids = {id(w) for w in moved}
    report["weights"] = [w for w in report["weights"] if id(w) not in moved_ids]
    moved_bytes = sum(w["bytes"] for w in moved)
    report["weight_code_bytes"] -= moved_bytes
    report["dense_bytes"] += moved_bytes
    storages = dict(report.get("storages", {}))
    storages["weights"] = storages.get("weights", 0) - len(moved)
    storages["dense"] = storages.get("dense", 0) + len(moved)
    report["storages"] = storages
    report["reclassified_decoder_side"] = [dict(w, **{"class": "dense: decoder-side, reclassified"}) for w in moved]
    report["reclassified_decoder_side_bytes"] = moved_bytes
    report["raw_weight_code_bytes"] = report["raw"]["weight_code_bytes"]
    report["within_budget"] = report["weight_code_bytes"] <= audit.CODE_BUDGET_BYTES
    return report


# --- the winner rule (A14; implementation corrected per Codex c1 finding 3) ----------------------

def paired(rows: list[dict], a: str, b: str, key: str) -> dict:
    """Median of per-utterance (a - b) / b in percent, with counts and per-target medians."""
    ch = [(r["summary"][a][key] - r["summary"][b][key]) / r["summary"][b][key] * 100 for r in rows]
    by = {}
    for t in sorted({r["target_seconds"] for r in rows}):
        sel = [c for c, r in zip(ch, rows) if r["target_seconds"] == t]
        by[str(t)] = {"median_pct": statistics.median(sel), "lower": sum(c < 0 for c in sel), "n": len(sel)}
    return {"median_pct": statistics.median(ch), "lower": sum(c < 0 for c in ch),
            "higher": sum(c > 0 for c in ch), "n": len(ch), "by_target": by}


def a14_tie(e: dict) -> bool:
    """A14's tie condition on a paired candidate-vs-candidate energy comparison: the median differs by
    less than 1%, or neither side is lower on at least 16 of 24 utterances."""
    return abs(e["median_pct"]) < TIE_PERCENT or (e["lower"] < TIE_COUNT and e["higher"] < TIE_COUNT)


def winner_rule(full_rows: list[dict], eligible: list[str], reference: str = REFERENCE) -> dict:
    """A14 with the anchored tie set (see the module docstring, "WINNER RULE")."""
    out = {"implementation": "anchored tie set (Codex GPU code review c2, 2026-10-21; adopted from the project's "
                             "pre-existing deterministic rule, plans/cpu-benchmark.md review round 2); common "
                             "reference and paired comparisons as corrected per Codex c1 finding 3",
           "reference": reference, "eligible": list(eligible)}
    energy = {c: paired(full_rows, c, reference, "joules_per_iteration")["median_pct"] for c in eligible}
    ranking = sorted(((c, energy[c]) for c in eligible), key=lambda t: (t[1], t[0]))
    out["ranking_vs_reference"] = ranking
    if not ranking:
        return out | {"winner": None, "decided_by": "no eligible candidate"}
    anchor = ranking[0][0]
    h2h = {}
    for c, _ in ranking[1:]:
        e = paired(full_rows, anchor, c, "joules_per_iteration")
        h2h[c] = {"energy_anchor_vs_candidate": e, "latency_anchor_vs_candidate": paired(full_rows, anchor, c, "median_ms"),
                  "anchor_lower": e["lower"], "candidate_lower": e["higher"], "tie": a14_tie(e)}
    tie_set = [anchor] + [c for c, _ in ranking[1:] if h2h[c]["tie"]]
    out |= {"anchor": anchor, "head_to_head": h2h, "tie_set": tie_set}
    if len(tie_set) == 1:
        if len(ranking) == 1:
            return out | {"winner": anchor, "decided_by": "energy (only eligible candidate)"}
        return out | {"winner": anchor, "decided_by": "energy (A* has no A14 tie with any eligible candidate)"}
    out["within_tie_set_energy"] = {f"{x} vs {y}": paired(full_rows, x, y, "joules_per_iteration")
                                    for i, x in enumerate(tie_set) for y in tie_set[i + 1:]}
    latency = {c: paired(full_rows, c, reference, "median_ms")["median_pct"] for c in tie_set}
    out["tie_set_latency_vs_reference"] = latency
    best = min(latency.values())
    fastest = sorted(c for c in tie_set if latency[c] == best)
    if len(fastest) == 1:
        return out | {"winner": fastest[0], "decided_by": "latency vs R-expanded within the A14 tie set"}
    beats = [c for c in fastest if all(paired(full_rows, c, d, "median_ms")["median_pct"] < 0 for d in fastest if d != c)]
    if len(beats) == 1:
        return out | {"winner": beats[0], "decided_by": "head-to-head latency (equal median latency vs R-expanded)"}
    return out | {"winner": fastest[0], "decided_by": "name (equal median latency vs R-expanded and head to head)"}


# --- session driver (no CUDA in this process) ----------------------------------------------------

def _run(cmd: list[str], log: Path, ledger: list, run: str, precheck=None) -> int:
    (precheck or (lambda: verify_freeze(run)))()
    started = time.time()
    with open(log, "a") as f:
        f.write(f"\n### {time.strftime('%Y-%m-%dT%H:%M:%S')} {' '.join(cmd)}\n")
        f.flush()
        e = dict(os.environ)
        e.pop("CUBLAS_WORKSPACE_CONFIG", None)
        e.pop("CUDA_VISIBLE_DEVICES", None)
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(env.BINARY_DIR), env=e).returncode
    ledger.append({"cmd": cmd, "rc": rc, "seconds": round(time.time() - started, 1), "log": str(log),
                   "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
    print(json.dumps(ledger[-1]), flush=True)
    return rc


def _chunk_done(path: Path, rng: str) -> bool:
    if not path.exists():
        return False
    a, b = (int(v) for v in rng.split("-"))
    rows = {r["index"] for r in json.loads(path.read_text())["rows"]}
    return rows >= set(range(a, b + 1))


def _gate_count(run: str, impl: str, part: str) -> int:
    from gpu1bit.shared import gate
    d = env.ARTIFACTS / "gate" / run / impl.replace(":", "_")
    if not (d / "fingerprint.json").exists():
        return 0
    return len(gate.read_rows(d / f"{part}.jsonl", json.loads((d / "fingerprint.json").read_text())["hash"]))


# --- checks: A12 gates, A5 audits, deferred A3 (shared by session and `checks`) -------------------

DEFERRED_MODULE = "gpu1bit.int8_mma.deferred"


def deferred_parameters(roster: dict, candidate: str) -> tuple[str, int]:
    """(provider module, part count) a hybrid-stack candidate is qualified with: roster keys
    deferred_module and deferred_components[candidate]; defaults DEFERRED_MODULE and 3."""
    return roster.get("deferred_module", DEFERRED_MODULE), int(roster.get("deferred_components", {}).get(candidate, 3))


def hybrid_candidates(roster: dict) -> list[str]:
    """Candidates on a deferred-path (hybrid) stack: their stack baseline is not the reference."""
    return [c for c in roster["candidates"] if roster["stack_baseline"][c] != REFERENCE]


def deferred_report_path(record_run: str, candidate: str) -> Path:
    return env.ARTIFACTS / "deferred" / record_run / f"{candidate.replace(':', '_').replace('@', '_at_')}.json"


def check_steps(record_run: str, roster: dict) -> list[dict]:
    """The quality checks every candidate needs, recorded under `record_run`: deferred A3 (full scope)
    for hybrid-stack candidates, the A12 gate (timing workloads and the 512 dev sample) for every
    candidate, the A5 audit for every variant. `done` means a valid record exists (pass or fail)."""
    from gpu1bit.shared import workloads
    doc = workloads.load()
    n_timing = len({r["id"] for r in doc["dev_clips"] + doc["dev_24"]})
    dispatch = [PY, "-m", "gpu1bit.shared.dispatch", "--patches", ",".join(roster["patches"])]
    out = []
    for c in hybrid_candidates(roster):
        module, components = deferred_parameters(roster, c)

        def done(c=c, module=module, components=components):
            p = deferred_report_path(record_run, c)
            return p.exists() and not deferred_a3_check(json.loads(p.read_text()), c, module, components)["provenance"]
        out.append({"name": f"deferred-{c}", "log": f"deferred-{c.replace(':', '_')}.log", "retries": 1, "done": done,
                    "cmd": dispatch + ["deferred", "qualify", "--provider", c, "--module", module, "--run", record_run]})
    for impl in roster["candidates"]:
        safe = impl.replace(":", "_")
        out.append({"name": f"gate-timing-{safe}", "log": f"gate-timing-{safe}.log", "retries": 1,
                    "cmd": dispatch + ["gate", "timing", "--agent", "H", "--run", record_run, "--impl", impl],
                    "done": lambda i=impl: _gate_count(record_run, i, "timing") >= n_timing})
        out.append({"name": f"gate-sample-{safe}", "log": f"gate-sample-{safe}.log", "retries": 12,
                    "cmd": dispatch + ["gate", "sample", "--agent", "H", "--run", record_run, "--impl", impl, "--range", "0-511"],
                    "done": lambda i=impl: _gate_count(record_run, i, "sample") >= 512})
    for v in roster["variants"]:
        path = sel_dir(record_run) / "audit" / f"{v.replace(':', '_')}.json"
        out.append({"name": f"audit-{v}", "log": f"audit-{v.replace(':', '_')}.log", "retries": 1,
                    "cmd": dispatch + ["audit", "--run", record_run, "--variant", v],
                    "done": lambda p=path, v=v: p.exists() and record_bound(json.loads(p.read_text()),
                                                                           audit_fingerprint(v, roster["patches"]))})
    return out


def checks_run_name(run: str, patches: list[str]) -> str:
    """The check-run name for the CURRENT source state (sources, exports, workloads, versions, patches):
    a source change gives a new name, so records of another state are never reused (Codex GPU c3)."""
    return f"{run}-chk-{F.make('selection checks', run=run, patches=list(patches))['hash'][:12]}"


def run_checks(args) -> None:
    """`checks`: run check_steps for a session's roster under checks_run_name (one `timeout 600`
    locked process per step; the source state is re-verified before each)."""
    run = args.run
    roster = load_roster(run)
    chk = checks_run_name(run, roster["patches"])
    root = sel_dir(chk)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    ledger_path = root / "ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []

    def same_state():
        if checks_run_name(run, roster["patches"]) != chk:
            raise SystemExit(f"the source state changed during the checks of {chk}; start again (new name)")
    print(json.dumps({"checks_run": chk, "session": run}), flush=True)
    for st in check_steps(chk, roster):
        for _ in range(st["retries"] + 1):
            if st["done"]():
                break
            _run(["timeout", "600", *st["cmd"]], logs / st["log"], ledger, chk, precheck=same_state)
            ledger_path.write_text(json.dumps(ledger, indent=1) + "\n")
        if not st["done"]():
            raise SystemExit(f"check step failed: {st['name']} (see {logs / st['log']})")
    print(f"CHECKS COMPLETE {chk}", flush=True)


def deferred_a3_check(doc: dict | None, candidate: str, module: str = DEFERRED_MODULE, components: int = 3) -> dict:
    """The one deferred-A3 validator (Codex GPU code review c3), used by every reporting path.
    provenance: anything that makes the record not a complete, current qualification of this provider
    (missing, other provider / module / identity, informational, partial scope or coverage, stale
    fingerprint). quality: failures. ok = no provenance problem and the qualification passes.
    Coverage is criterion-specific (plans/x2-default.md Z5): at x3 discriminating cases are required; at
    x2 every case must be bitwise exact (deferred == in-kernel) and there is no discriminating-case
    requirement."""
    from gpu1bit.shared import deferred_a3
    prov, quality = [], []
    if not doc:
        return {"ok": False, "passes": False, "provenance": [f"{candidate}: no deferred-A3 report"], "quality": []}
    full, cov = deferred_a3.full_scope(components), deferred_a3.expected_coverage()
    if doc.get("provider") != candidate:
        prov.append(f"provider {doc.get('provider')!r} is not {candidate!r}")
    if doc.get("module") != module:
        prov.append(f"module {doc.get('module')!r} is not {module!r}")
    if doc.get("informational"):
        prov.append("informational provider")
    try:
        identity = deferred_a3.provider_identity(candidate, module)
    except Exception as exc:  # noqa: BLE001 - an unresolvable provider is a provenance problem
        identity = {"error": repr(exc)}
        prov.append(f"provider identity unresolvable: {exc!r}")
    if doc.get("provider_identity") != identity:
        prov.append(f"provider identity {doc.get('provider_identity')} is not the current {identity}")
    if doc.get("scope") != full or doc.get("partial") is not False:
        prov.append(f"partial scope {doc.get('scope')} (informational only)")
    real, syn = doc.get("real") or {}, doc.get("synthetic") or {}
    if real.get("evaluations") != cov["real_evaluations"] or real.get("deferred_evaluations") != cov["deferred_evaluations"]:
        prov.append(f"real coverage {real.get('evaluations')}/{real.get('deferred_evaluations')} is not "
                    f"{cov['real_evaluations']}/{cov['deferred_evaluations']}")
    if real.get("deferral_mismatches") != 0:
        prov.append(f"deferral mismatches {real.get('deferral_mismatches')}")
    if syn.get("cases") != cov["synthetic_cases"] or (components == 3 and not syn.get("discriminating_cases")):
        prov.append(f"synthetic coverage {syn.get('cases')} cases ({syn.get('discriminating_cases')} discriminating), "
                    f"expected {cov['synthetic_cases']}" + (" with discriminating cases" if components == 3 else ""))
    if not record_bound(doc, deferred_a3.record_fingerprint(candidate, module, full)):
        prov.append("fingerprint is not the current one for the full scope")
    for part, d in (("real", real), ("synthetic", syn)):
        if d.get("failures", 1) != 0:
            quality.append(f"{part} failures {d.get('failures')}")
    if components == 3 and syn.get("discriminating_failures", 1) != 0:
        quality.append(f"discriminating failures {syn.get('discriminating_failures')}")
    if components == 2 and (real.get("inexact", 1) != 0 or syn.get("inexact", 1) != 0):
        quality.append(f"inexact cases: real {real.get('inexact')}, synthetic {syn.get('inexact')}")
    passes = doc.get("passes") is True and not quality
    if doc.get("passes") is not True and not quality:
        quality.append("report does not pass")
    prov = [f"{candidate}: deferred A3 {x}" for x in prov]
    return {"ok": not prov and passes, "passes": passes, "provenance": prov, "quality": quality}


def _gate_verdict(checks_run: str, candidate: str, patches: list[str]) -> dict:
    from gpu1bit.shared import gate
    return gate.compute_verdict("H", checks_run, candidate, patches)


def evaluate_eligibility(roster: dict, checks_run: str, encoder: dict[str, dict]) -> dict:
    """Eligibility from the records of `checks_run` (Codex GPU c3: the one validator of both `report`
    and `regenerate`). encoder: the P-paired encoder checks per variant. A candidate is eligible iff
    its A12 verdict passes, its A5 audit is bound and within budget, its P-paired encoder checks pass,
    and (hybrid stacks) its deferred-A3 record is valid and passes. Provenance problems (stale or
    missing records, partial scope) are listed separately from quality failures."""
    patches = roster["patches"]
    out = {"checks_run": checks_run, "gate": {}, "audit": {}, "deferred_a3": {}, "eligibility": {}, "provenance_problems": []}
    prov = out["provenance_problems"]
    for v in roster["variants"]:
        p = sel_dir(checks_run) / "audit" / f"{v.replace(':', '_')}.json"
        d = json.loads(p.read_text()) if p.exists() else {}
        bound = bool(d) and record_bound(d, audit_fingerprint(v, patches))
        out["audit"][v] = {k: d.get(k) for k in ("weight_code_bytes", "raw_weight_code_bytes", "dense_bytes", "scale_bytes",
                                                 "activation_bytes", "within_budget", "reclassified_decoder_side_bytes")}
        out["audit"][v]["fingerprint_bound"] = bound
    hybrid = hybrid_candidates(roster)
    for c in roster["candidates"]:
        g = _gate_verdict(checks_run, c, patches)
        out["gate"][c] = g
        if g.get("problems") or not g.get("complete"):
            prov.append(f"{c}: gate {g.get('problems') or 'incomplete records'}")
        if not out["audit"][c]["fingerprint_bound"]:
            prov.append(f"{c}: audit missing or not bound to the current fingerprint")
        e = {"gate": bool(g.get("passes")), "audit": bool(out["audit"][c].get("within_budget")) and out["audit"][c]["fingerprint_bound"],
             "p_paired_encoder": bool(encoder[c]["ok"])}
        if c in hybrid:
            p = deferred_report_path(checks_run, c)
            chk = deferred_a3_check(json.loads(p.read_text()) if p.exists() else None, c, *deferred_parameters(roster, c))
            out["deferred_a3"][c] = {"report": str(p), **chk}
            prov += chk["provenance"]
            e["deferred_a3"] = chk["ok"]
        e["eligible"] = all(e.values())
        out["eligibility"][c] = e
    return out


def write_decision(path: Path, out: dict) -> None:
    """Write a decision only if it has no provenance problems (Codex GPU c3); an existing decision is
    kept as <stem>.previous-<time>.json first. Otherwise nothing is written and SystemExit names them."""
    if out.get("provenance_problems"):
        raise SystemExit(f"refusing to write {path}: provenance problems: {out['provenance_problems'][:8]}")
    if path.exists():
        import shutil
        shutil.copy2(path, path.with_name(f"{path.stem}.previous-{time.strftime('%Y%m%dT%H%M%S')}.json"))
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(out, indent=1) + "\n")
    os.replace(tmp, path)


def session(args) -> None:
    run = args.run
    roster = load_roster(run)
    if roster.get("legacy"):
        raise SystemExit("legacy runs are not re-run")
    root = sel_dir(run)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    ledger_path = root / "ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    (root / "freeze-check.json").write_text(json.dumps(verify_freeze(run), indent=1) + "\n")
    bench_dir = env.ARTIFACTS / "bench" / run
    dispatch = [PY, "-m", "gpu1bit.shared.dispatch", "--patches", ",".join(roster["patches"])]
    variants = roster["variants"]

    def step(cmd, log, done, retries=1):
        for _ in range(retries + 1):
            if done():
                return
            rc = _run(["timeout", "600", *cmd], logs / log, ledger, run)
            ledger_path.write_text(json.dumps(ledger, indent=1) + "\n")
            if rc == 0 and done():
                return
        if not done():
            raise SystemExit(f"step failed: {cmd} (see {logs / log})")

    for scope in ("encoder", "full"):
        for rng in CHUNKS:
            a, b = (int(v) for v in rng.split("-"))
            path = bench_dir / f"paired-{scope}-{a:02d}-{b:02d}.json"
            step(dispatch + ["bench", "paired", "--agent", "H", "--run", run, "--scope", scope, "--variants", *variants,
                             "--utterances", rng], f"paired-{scope}-{rng}.log", lambda p=path, r=rng: _chunk_done(p, r))
    for order, label in ((variants, "clips1"), (variants[::-1], "clips2")):
        for v in order:
            path = env.ARTIFACTS / "bench" / f"{run}-{label}" / f"clips-{v.replace(':', '_')}.json"
            step(dispatch + ["bench", "clips", "--agent", "H", "--run", f"{run}-{label}", "--variant", v],
                 f"{label}-{v.replace(':', '_')}.log",
                 lambda p=path: p.exists() and len(json.loads(p.read_text()).get("workloads", [])) == 3
                 and "memory_samples_10ms" in json.loads(p.read_text()))
    if roster["phase_pair"]:
        for rng in CHUNKS:
            a, b = (int(v) for v in rng.split("-"))
            path = env.ARTIFACTS / "bench" / run / f"phases-{a:02d}-{b:02d}.json"
            step(dispatch + ["phases", "--agent", "H", "--run", run, "--variants", *roster["phase_pair"], "--utterances", rng],
                 f"phases-{rng}.log", lambda p=path, r=rng: _chunk_done(p, r))
    for st in check_steps(run, roster):
        step(st["cmd"], st["log"], st["done"], retries=st["retries"])
    print("SESSION COMPLETE", flush=True)


# --- report --------------------------------------------------------------------------------------

def load_rows(run: str, scope: str, variants: list[str]) -> list[dict]:
    rows = []
    for f in sorted((env.ARTIFACTS / "bench" / run).glob(f"paired-{scope}-*.json")):
        if "merged" in f.name:
            continue
        doc = json.loads(f.read_text())
        if doc["variants"] != variants:
            raise ValueError(f"{f}: variants {doc['variants']} != roster {variants}")
        rows += doc["rows"]
    rows.sort(key=lambda r: r["index"])
    if [r["index"] for r in rows] != list(range(24)):
        raise ValueError(f"{scope}: utterances {[r['index'] for r in rows]}")
    return rows


def encoder_checks(rows: list[dict], variant: str) -> dict:
    """P-paired encoder-scope checks against the first variant (R-expanded) on all utterances."""
    bad = [r["id"] for r in rows if not (r["checks"][variant]["finite"] and r["checks"][variant]["lengths_equal"]
                                         and r["checks"][variant]["rel_frobenius_valid"] <= ENCODER_REL_LIMIT)]
    return {"ok": not bad, "failing": bad, "utterances": len(rows),
            "max_rel_frobenius_valid": max(r["checks"][variant]["rel_frobenius_valid"] for r in rows)}


def report(args) -> dict:
    run = args.run
    roster = load_roster(run)
    if roster.get("legacy"):
        raise SystemExit("use amend-legacy for sel-20261020")
    root = sel_dir(run)
    variants, candidates = roster["variants"], roster["candidates"]
    keys = ("median_ms", "joules_per_iteration", "average_watts")
    out = {"run": run, "roster": roster, "freeze": verify_freeze(run), "scopes": {}}
    for scope in ("encoder", "full"):
        rows = load_rows(run, scope, variants)
        s = {"absolute_median": {v: {k: statistics.median(r["summary"][v][k] for r in rows) for k in keys} for v in variants},
             "vs_reference": {v: {k: paired(rows, v, REFERENCE, k) for k in keys} for v in variants if v != REFERENCE},
             "vs_stack_baseline": {c: {k: paired(rows, c, roster["stack_baseline"][c], k) for k in keys} for c in candidates
                                   if roster["stack_baseline"][c] != REFERENCE}}
        if scope == "encoder":
            s["checks"] = {v: encoder_checks(rows, v) for v in variants}
        else:
            s["checks"] = {v: {"identical_to_reference": sum(r["checks"][v][f"identical_to_{REFERENCE}"] for r in rows)}
                           for v in variants}
        out["scopes"][scope] = s
    out["clips"] = {}
    for label in ("clips1", "clips2"):
        for v in variants:
            p = env.ARTIFACTS / "bench" / f"{run}-{label}" / f"clips-{v.replace(':', '_')}.json"
            if p.exists():
                d = json.loads(p.read_text())
                out["clips"].setdefault(v, []).append({
                    "order": label, "clips": [{"id": w["record"]["id"], "M": w["record"]["M"],
                                               **{k: w["summary"][k] for k in ("median_ms", "joules_per_iteration", "average_watts")},
                                               "process_peak_incl_first_bytes": w["process_sampled_peak_including_first_bytes"],
                                               "text": w["hypothesis"]} for w in d["workloads"]]})
    if roster["phase_pair"]:
        from gpu1bit.shared import phases
        out["phases"] = phases.summarize(run, roster["phase_pair"])
    out |= evaluate_eligibility(roster, run, out["scopes"]["encoder"]["checks"])
    eligible = [c for c in candidates if out["eligibility"][c]["eligible"]]
    out["winner_rule"] = winner_rule(load_rows(run, "full", variants), eligible)
    write_decision(root / "report.json", out)
    print(json.dumps({"eligibility": out["eligibility"], "winner_rule": {k: v for k, v in out["winner_rule"].items()
                                                                          if not k.startswith("paired_")}}, indent=1))
    return out


# --- corrected decision from recorded timing (Codex GPU code review c2) ----------------------------

def _ctime(p: Path) -> float:
    return os.stat(p).st_ctime


def verify_archive(archive: Path, freeze: Path) -> dict:
    """The archived sources equal the freeze manifest: same file set, every SHA-256 equal."""
    import hashlib
    import tarfile
    listed = {}
    for line in freeze.read_text().splitlines():
        if line.strip():
            digest, rel = line.split(None, 1)
            listed[rel.strip()] = digest
    got = {}
    with tarfile.open(archive) as t:
        for m in t.getmembers():
            if m.isfile():
                got[m.name] = hashlib.sha256(t.extractfile(m).read()).hexdigest()
    missing, extra = sorted(set(listed) - set(got)), sorted(set(got) - set(listed))
    bad = sorted(r for r in set(listed) & set(got) if listed[r] != got[r])
    if missing or extra or bad:
        raise SystemExit(f"archive {archive} does not match {freeze}: missing {missing[:5]}, extra {extra[:5]}, changed {bad[:5]}")
    return {"archive": str(archive), "archive_sha256": env.sha256_file(archive), "freeze": str(freeze), "files": len(listed)}


def verify_timing_binding(run: str, roster: dict, *, host: Path, archive: Path, freeze: Path, roster_file: Path,
                          inputs: list[Path], max_load: float = 2.0, inputs_before: float | None = None,
                          slack_s: float = 2.0) -> dict:
    """The recorded timing of `run` is still what the session measured (see tests TimingBinding):
    the archive equals the freeze manifest; every timing step has an accepted, freeze-verified,
    load-checked sidecar attempt; each output is unchanged since that attempt ended and names its
    protocol, run, variants and range; the roster, the freeze manifest and every file under `inputs`
    (the exports) were last changed before the first accepted attempt started. SystemExit otherwise."""
    import sys
    sys.path.insert(0, str(env.GPU1BIT.parent))
    import gpu1bit.rerun_timing as T
    out = {"archive": verify_archive(archive, freeze)}
    starts, checked = [], []
    for step in T.timing_steps(run, roster):
        side = host / f"{step['name']}.json"
        rec = json.loads(side.read_text()) if side.exists() else {}
        acc = [a for a in rec.get("attempts", []) if a.get("accepted")]
        if not rec.get("accepted") or not acc:
            raise SystemExit(f"{step['name']}: no accepted attempt")
        a = acc[-1]
        fz = a.get("freeze") or {}
        if not (fz.get("ok") is True and not fz.get("mismatched") and not fz.get("added") and fz.get("files")):
            raise SystemExit(f"{step['name']}: the accepted attempt has no passing freeze check ({fz})")
        if a.get("load1_max") is None or a["load1_max"] > max_load:
            raise SystemExit(f"{step['name']}: the accepted attempt's load1 max {a.get('load1_max')} > {max_load}")
        start = time.mktime(time.strptime(a["started"], "%Y-%m-%dT%H:%M:%S"))
        end = start + float(a["seconds"])
        if not step["output"].exists():
            raise SystemExit(f"{step['name']}: output {step['output']} missing")
        if _ctime(step["output"]) > end + slack_s:
            raise SystemExit(f"{step['name']}: output changed after its accepted attempt ended "
                             f"(ctime {_ctime(step['output']):.0f} > end {end:.0f})")
        doc = json.loads(step["output"].read_text())
        if not (T.output_complete(step) and T.artifact_bound(step, doc, roster, None)):
            raise SystemExit(f"{step['name']}: output not bound to the roster's protocol, run, variants and range")
        starts.append(start)
        checked.append({"step": step["name"], "attempt_started": a["started"], "seconds": a["seconds"],
                        "load1_max": a["load1_max"], "freeze_files": fz.get("files"),
                        "output_sha256": env.sha256_file(step["output"])})
    first = min(starts)
    limit = first if inputs_before is None else inputs_before
    files = [roster_file, freeze] + [q for d in inputs for q in sorted(Path(d).rglob("*"))] + [Path(d) for d in inputs]
    late = [str(q) for q in files if _ctime(q) > limit]
    if late:
        raise SystemExit(f"inputs changed after the first accepted timing attempt started: {late[:5]}")
    return out | {"ok": True, "steps": len(checked), "first_attempt_started": first, "inputs_checked": len(files),
                  "inputs_last_changed": max(_ctime(q) for q in files), "rule": "outputs: ctime <= accepted attempt end + "
                  f"{slack_s} s; inputs (roster, freeze manifest, export files and directories): ctime <= first accepted start",
                  "step_records": checked}


def regenerate(args) -> dict:
    """A corrected decision for a session from its RECORDED timing (binding re-verified) and the A12
    gates, A5 audits and deferred-A3 checks re-run under the current fingerprints in run `--checks`.
    Writes selection/<run>/report-corrected.json; report.json stays as recorded."""
    run = args.run
    roster = load_roster(run)
    checks = args.checks or checks_run_name(run, roster["patches"])
    root = sel_dir(run)
    archive = root / f"frozen-sources-{run}.tar.gz"
    binding = verify_timing_binding(run, roster, host=root / "host", archive=archive, freeze=freeze_path(run),
                                    roster_file=root / "roster.json", inputs=[env.MB1_EXPORT, env.DERIVED_EXPORT])
    variants, candidates = roster["variants"], roster["candidates"]
    keys = ("median_ms", "joules_per_iteration", "average_watts")
    out = {"run": run, "checks_run": checks, "roster": roster, "timing_binding": {k: v for k, v in binding.items()
                                                                                    if k != "step_records"},
           "timing_step_records": binding["step_records"], "scopes": {}}
    for scope in ("encoder", "full"):
        rows = load_rows(run, scope, variants)
        sc = {"absolute_median": {v: {k: statistics.median(r["summary"][v][k] for r in rows) for k in keys} for v in variants},
              "vs_reference": {v: {k: paired(rows, v, REFERENCE, k) for k in keys} for v in variants if v != REFERENCE}}
        if scope == "encoder":
            sc["checks"] = {v: encoder_checks(rows, v) for v in variants}
        out["scopes"][scope] = sc
    out |= evaluate_eligibility(roster, checks, out["scopes"]["encoder"]["checks"])
    elig = out["eligibility"]
    eligible = [c for c in candidates if elig[c]["eligible"]]
    out["winner_rule"] = winner_rule(load_rows(run, "full", variants), eligible)
    recorded = json.loads((root / "report.json").read_text())["winner_rule"]
    out["as_recorded"] = {"winner": recorded.get("winner"), "decided_by": recorded.get("decided_by"),
                          "implementation": recorded.get("implementation")}
    out["fingerprint_now"] = {"sources": F.sources(), "exports": F.exports()}
    out["rule"] = {"plan": "plans/gpu-kernel.md A15", "plan_sha256": env.sha256_file(env.BINARY_DIR / "plans" / "gpu-kernel.md")}
    write_decision(root / "report-corrected.json", out)
    print(json.dumps({"eligibility": elig, "as_recorded": out["as_recorded"],
                      "winner_rule": {k: v for k, v in out["winner_rule"].items()
                                      if k in ("ranking_vs_reference", "anchor", "tie_set", "tie_set_latency_vs_reference",
                                               "winner", "decided_by")}}, indent=1))
    return out


def amend_legacy(args) -> dict:
    """sel-20261020 with the corrected winner-rule implementation (P-paired data as recorded; that
    session's gate records predate fingerprinting and are taken as recorded in its report.json)."""
    run = args.run
    roster = load_roster(run)
    old = json.loads((sel_dir(run) / "report.json").read_text())
    eligible = old["winner_rule"]["eligible"]
    rule = winner_rule(load_rows(run, "full", roster["variants"]), eligible)
    out = {"run": run, "as_recorded": {k: old["winner_rule"].get(k) for k in ("winner", "decided_by", "tie_condition",
                                                                             "difference_points")},
           "corrected": rule, "note": "gate eligibility as recorded (pre-fingerprint records); ranking recomputed"}
    (sel_dir(run) / "report-amended.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"as_recorded": out["as_recorded"], "corrected": {k: v for k, v in rule.items() if not k.startswith("paired_")}},
                     indent=1))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["roster", "verify", "session", "audit", "report", "amend-legacy", "regenerate",
                                      "checks", "checks-name"])
    p.add_argument("--run", required=True)
    p.add_argument("--checks", help="regenerate: the check run (default: checks_run_name of the current source state)")
    p.add_argument("--variant")
    p.add_argument("--variants", nargs="+")
    p.add_argument("--patches", default="")
    p.add_argument("--phase-pair", nargs="*")
    p.add_argument("--stack-baseline", nargs="*", default=[])
    args = p.parse_args()
    if args.action == "roster":
        print(json.dumps(make_roster(args), indent=1))
    elif args.action == "verify":
        print(json.dumps(verify_freeze(args.run), indent=1))
    elif args.action == "session":
        session(args)
    elif args.action == "audit":
        audit_variant(args)
    elif args.action == "report":
        report(args)
    elif args.action == "regenerate":
        regenerate(args)
    elif args.action == "checks":
        run_checks(args)
    elif args.action == "checks-name":
        print(checks_run_name(args.run, load_roster(args.run)["patches"]))
    else:
        amend_legacy(args)


if __name__ == "__main__":
    main()
