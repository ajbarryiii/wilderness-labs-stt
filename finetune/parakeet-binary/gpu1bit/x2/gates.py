"""The INT8x2 dev gates (plans/x2-default.md "Gates", X4, Y2; agent H). Dev only; accuracy only.

ROSTER (X4, Y2), all under the candidate selection document (shared/selected-x2.json, GPU1BIT_SELECTION):
  g0  MB1  provider int8_mma        x3  reference for g2; R2 against MB1's pinned FP32-NeMo dev WERs
  g2  MB1  provider int8_mma:x2     x2  reference for g1
  g1  MB1  B-hybrid                 x2  candidate (MB1 default)
  g3  MB1  B-hybrid                 x3  R5 lengths, x3 context
  g5  M1   R-expanded               x3  reference for g4; R2 against M1's pinned FP32-NeMo dev WERs
  g4  M1   R-expanded               x2  candidate (M1 default)
ORDER: references first for each model: g0, g2, g1, g3, then g5, g4. `require_launch` runs before every
arm. It refuses g2 / g4 unless g0 / g5 is complete, identity-verified, of the same model and passed R2,
and g1 / g3 unless g2 passes the same check (its own chain included). This uses
gpu1bit.actprec.parts.require_reference, which the workers also call; no caller bypasses it.

GATES (frozen):
  MB1 (g1 against g2):
    - mean WER within 0.02 points and every set within 0.10 points (both directions);
    - token mismatches at most 0.5% pooled and 1.0% per set;
    - encoder relative Frobenius on valid frames at most 1e-3 per utterance on the 512 dev sample (A12);
    - R5: finite encoder outputs, decoding completes, encoder lengths equal to g3's, on every dev
      utterance.
    Every mismatching utterance is listed.
  M1:
    - g5 passes R2;
    - g4 against g5 is recognition-equivalent under the activation-precision reading (0.02 mean, no set
      worse by more than 0.10, at most 0.5% pooled and 1.0% per set token mismatches, R5 valid);
    - the 512-sample encoder comparison of g4 against g5 is reported.
  On failure: no timing and no test scoring; the user is told. Bounds are not loosened.

    gpu1bit/shared/python -m gpu1bit.x2.gates drive --run x2g-<date>          (CPU driver; locked workers)
    gpu1bit/shared/python -m gpu1bit.x2.gates encoder --run R --arm g1 --ref g2   (one locked GPU process)
    gpu1bit/shared/python -m gpu1bit.x2.gates evaluate --run R                (CPU; writes the gate record)
    gpu1bit/shared/python -m gpu1bit.x2.gates recorded --run R                (CPU; Z4, descriptive only)
Gate record: .../gpu1bit/H/x2/gates-<run>.json. After the evaluation, drive writes the Z4 descriptive
comparisons of g2 and g4 with the recorded ap-20261020r4 arms (recorded-<run>.json); they never gate.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from gpu1bit.shared import env, fulldev

GATE_ROSTER = [
    {"id": "g0", "model": "MB1", "export": "MB1 1-bit export", "stack": "provider int8_mma", "parts": 3,
     "arm": "MB1-int8_mma-x3", "role": "reference for g2; R2", "reference": None},
    {"id": "g2", "model": "MB1", "export": "MB1 1-bit export", "stack": "provider int8_mma:x2", "parts": 2,
     "arm": "MB1-int8_mma-x2", "role": "reference for g1", "reference": "g0"},
    {"id": "g1", "model": "MB1", "export": "MB1 1-bit export", "stack": "B-hybrid", "parts": 2,
     "arm": "MB1-B-hybrid-x2", "role": "candidate (MB1 default)", "reference": "g2"},
    {"id": "g3", "model": "MB1", "export": "MB1 1-bit export", "stack": "B-hybrid", "parts": 3,
     "arm": "MB1-B-hybrid-x3", "role": "R5 lengths, x3 context", "reference": "g2"},
    {"id": "g5", "model": "M1", "export": "M1 export", "stack": "R-expanded", "parts": 3,
     "arm": "M1-x3", "role": "reference for g4; R2", "reference": None},
    {"id": "g4", "model": "M1", "export": "M1 export", "stack": "R-expanded", "parts": 2,
     "arm": "M1-x2", "role": "candidate (M1 default)", "reference": "g5"},
]
ORDER = ["g0", "g2", "g1", "g3", "g5", "g4"]
ARM = {g["id"]: g["arm"] for g in GATE_ROSTER}
MEAN_POINTS, SET_POINTS, POOLED_MISMATCH, SET_MISMATCH, ENCODER_REL = 0.02, 0.10, 0.005, 0.010, 1e-3
EPS = 1e-9
SETS = fulldev.DEV_SETS


def out_dir() -> Path:
    return env.artifacts("x2")


def require_launch(run: str, gid: str) -> None:
    """SystemExit unless gid's reference chain is in place (Y2)."""
    from gpu1bit.actprec import parts
    g = {x["id"]: x for x in GATE_ROSTER}[gid]
    if g["reference"] is None:
        return
    parts.require_reference(run, g["arm"])


# --- comparisons and gates (pure) ------------------------------------------------------------------

def row_comparison(a_rows: dict[str, list[dict]], b_rows: dict[str, list[dict]]) -> dict:
    """Per set and pooled: token mismatches (all ids listed), text mismatches, encoder-length differences;
    the fulldev.compare format, so the activation-precision reading applies to it."""
    out, tot, mism = {"sets": {}}, 0, 0
    for s in SETS:
        a = {r["id"]: r for r in a_rows[s]}
        b = {r["id"]: r for r in b_rows[s]}
        if set(a) != set(b):
            raise ValueError(f"{s}: the two arms cover different utterances")
        ids = sorted(a)
        m = [u for u in ids if a[u]["tokens"] != b[u]["tokens"]]
        out["sets"][s] = {"utterances": len(ids), "token_mismatches": len(m), "mismatch_rate": len(m) / max(len(ids), 1),
                          "mismatch_ids": m, "encoder_length_differences": [u for u in ids if a[u]["enc_len"] != b[u]["enc_len"]],
                          "text_mismatches": sum(a[u]["hyp"] != b[u]["hyp"] for u in ids)}
        tot += len(ids)
        mism += len(m)
    out["pooled"] = {"utterances": tot, "token_mismatches": mism, "mismatch_rate": mism / max(tot, 1)}
    return out


def _r5(rows: dict[str, list[dict]]) -> dict:
    return {"not_finite": [r["id"] for s in SETS for r in rows[s] if r.get("finite") is not True],
            "decode_errors": [r["id"] for s in SETS for r in rows[s] if r.get("decode_error")]}


def mb1_gate(g1: dict, g2: dict, g1_rows: dict, g2_rows: dict, g3_rows: dict, enc: dict) -> dict:
    cmp = row_comparison(g1_rows, g2_rows)
    lengths = row_comparison(g1_rows, g3_rows)
    r5 = _r5(g1_rows)
    len_diff = [u for s in SETS for u in lengths["sets"][s]["encoder_length_differences"]]
    deltas = {s: g1["dev_wer_percent"][s] - g2["dev_wer_percent"][s] for s in SETS}
    mean_delta = g1["dev_mean_wer_percent"] - g2["dev_mean_wer_percent"]
    criteria = {
        "mean_within_0.02": abs(mean_delta) <= MEAN_POINTS + EPS,
        "every_set_within_0.10": all(abs(v) <= SET_POINTS + EPS for v in deltas.values()),
        "pooled_mismatch_le_0.5%": cmp["pooled"]["mismatch_rate"] <= POOLED_MISMATCH + EPS,
        "each_set_mismatch_le_1.0%": all(cmp["sets"][s]["mismatch_rate"] <= SET_MISMATCH + EPS for s in SETS),
        "encoder_rel_frobenius_le_1e-3": (enc.get("utterances") == 512 and enc.get("max_rel_frobenius_valid") is not None
                                          and enc["max_rel_frobenius_valid"] <= ENCODER_REL and not enc.get("not_finite")
                                          and not enc.get("length_differences")),
        "r5_finite_decoded": g1.get("structurally_valid") is True and not r5["not_finite"] and not r5["decode_errors"],
        "r5_lengths_equal_g3": not len_diff,
    }
    return {"pass": all(criteria.values()), "criteria": criteria, "mean_delta_points": mean_delta, "set_delta_points": deltas,
            "token_mismatches": {"pooled": cmp["pooled"]["token_mismatches"], **{s: cmp["sets"][s]["token_mismatches"] for s in SETS}},
            "mismatch_rate": {"pooled": cmp["pooled"]["mismatch_rate"], **{s: cmp["sets"][s]["mismatch_rate"] for s in SETS}},
            "mismatch_ids": [u for s in SETS for u in cmp["sets"][s]["mismatch_ids"]],
            "r5": {**r5, "length_differences_vs_g3": len_diff}, "encoder": enc}


def m1_gate(g5: dict, g4: dict, cmp45: dict, enc: dict) -> dict:
    import gpu1bit.actprec_report as AR
    r2 = bool((g5.get("cross_check") or {}).get("pass")) and g5.get("structurally_valid") is True
    reading = AR.reading(g4, g5, cmp45, {"not_finite": enc.get("not_finite", []),
                                         "length_differences": enc.get("length_differences", [])})
    return {"pass": r2 and reading["verdict"] == "recognition-equivalent", "g5_r2": r2, "reading": reading,
            "encoder_reported": enc}


# --- GPU: encoder comparison on the 512 dev sample ---------------------------------------------------

def encoder_compare(run: str, arm_id: str, ref_id: str) -> dict:
    """Relative Frobenius (valid frames, ref lengths) of arm against ref on the 512-utterance dev sample;
    both identities verified; one locked GPU process; reference built first."""
    import torch
    import evaluate as ev
    from nemo.utils import logging
    from gpu1bit.actprec import parts
    env.require_shared_wrapper()
    logging.setLevel(logging.ERROR)
    parts.install()
    require_launch(run, arm_id)
    arm, ref = ARM[arm_id], ARM[ref_id]
    ids = {a: fulldev.check_identity(run, a, "full")["hash"] for a in (arm, ref)}
    records = [(n, i, rec) for n in SETS for i, rec in fulldev.subset_records(n, "sample")]
    rows, store = {}, {}
    with env.gpu_lock(f"H:x2 encoder {arm_id} vs {ref_id} {run}") as lock:
        for a in (ref, arm):
            with torch.inference_mode():
                model, _ = fulldev.build_arm(fulldev.parse_arm(a))
            try:
                for n, i, rec in records:
                    env.check_hold(lock)
                    clip, _ = fulldev.load_clip(rec)
                    audio = torch.from_numpy(clip).unsqueeze(0).cuda()
                    length = torch.tensor([len(clip)], dtype=torch.long).cuda()
                    with torch.inference_mode(), ev.strict_fp32("cuda"), ev.inference_settings(model):
                        enc, enc_len = model.forward(input_signal=audio, input_signal_length=length)
                        t_n, finite = int(enc_len[0]), bool(torch.isfinite(enc).all())
                        if a == ref:
                            store[rec["id"]] = (enc[..., :t_n].double().cpu(), t_n)
                            rows[rec["id"]] = {"set": n, "id": rec["id"], "ref_len": t_n, "ref_finite": finite}
                        else:
                            r, t = store[rec["id"]]
                            y = enc[..., :t].double().cpu()
                            rows[rec["id"]].update(finite=finite, length_equal=t_n == t,
                                                   rel_frobenius_valid=((y - r).norm() / r.norm()).item() if t_n == t else None)
            finally:
                parts.close(model)
                del model
                gc.collect()
                torch.cuda.empty_cache()
        held = time.monotonic() - lock["acquired"]
    rows = list(rows.values())
    rels = [r["rel_frobenius_valid"] for r in rows if r["rel_frobenius_valid"] is not None]
    doc = {"run": run, "arm": arm, "ref": ref, "arm_id": arm_id, "ref_id": ref_id, "identities": ids, "subset": "sample",
           "utterances": len(rows), "max_rel_frobenius_valid": max(rels) if rels else None,
           "not_finite": [r["id"] for r in rows if not r["finite"]], "ref_not_finite": [r["id"] for r in rows if not r["ref_finite"]],
           "length_differences": [r["id"] for r in rows if not r["length_equal"]], "above_1e-3": sum(v > ENCODER_REL for v in rels),
           "lock_held_s": held, "wrapper": env.wrapper(), "rows": rows}
    (out_dir() / f"encoder-{run}-{arm_id}-vs-{ref_id}.json").write_text(json.dumps(doc) + "\n")
    return doc


def _encoder_doc(run: str, arm_id: str, ref_id: str, ids: dict) -> dict:
    p = out_dir() / f"encoder-{run}-{arm_id}-vs-{ref_id}.json"
    doc = json.loads(p.read_text())
    if (doc.get("run"), doc.get("subset"), doc.get("identities")) != (run, "sample", {ARM[arm_id]: ids[arm_id], ARM[ref_id]: ids[ref_id]}):
        raise ValueError(f"{p}: not bound to run {run} and the verified identities")
    if doc.get("ref_not_finite"):
        doc["not_finite"] = list(doc.get("not_finite", [])) + [f"ref:{u}" for u in doc["ref_not_finite"]]
    return doc


# --- evaluation and driver ---------------------------------------------------------------------------

def evaluate(run: str) -> dict:
    """Both gates from identity-bound, complete records; writes the gate record."""
    import gpu1bit.actprec_report as AR
    from gpu1bit.actprec import parts
    from gpu1bit.shared import seldoc
    parts.install()
    counts = {n: len(fulldev.subset_records(n, "full")) for n in SETS}
    sums, ids, rows = {}, {}, {}
    for gid in ORDER:
        sums[gid], ids[gid] = AR.verified_summary(run, ARM[gid], counts)
        rows[gid] = {n: fulldev.summarize_set(n, fulldev.load_rows(fulldev.work_path(run, ARM[gid], n)), "full", ids[gid])["records"]
                     for n in SETS}
    enc1 = _encoder_doc(run, "g1", "g2", ids)
    enc4 = _encoder_doc(run, "g4", "g5", ids)
    mb1 = mb1_gate(sums["g1"], sums["g2"], rows["g1"], rows["g2"], rows["g3"],
                   {k: enc1.get(k) for k in ("utterances", "max_rel_frobenius_valid", "not_finite", "length_differences")})
    g0_r2 = bool((sums["g0"].get("cross_check") or {}).get("pass")) and sums["g0"].get("structurally_valid") is True
    mb1["g0_r2"] = g0_r2
    mb1["pass"] = mb1["pass"] and g0_r2
    m1 = m1_gate(sums["g5"], sums["g4"], row_comparison(rows["g4"], rows["g5"]),
                 {k: enc4.get(k) for k in ("utterances", "max_rel_frobenius_valid", "not_finite", "length_differences")})
    sel = seldoc.load()
    record = {"run": run, "selection": {"path": sel["path"], "sha256": sel["sha256"]}, "identities": ids,
              "arms": ARM, "mb1": mb1, "m1": m1, "pass": mb1["pass"] and m1["pass"],
              "rule": "plans/x2-default.md Gates, X4, Y2 (frozen bounds; not loosened after seeing a result)"}
    (out_dir() / f"gates-{run}.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


def drive(run: str, selection: Path | None = None) -> dict:
    """g0, g2, g1, g3, g5, g4 in order (require_launch before every arm), then the encoder comparisons and
    the evaluation. CPU driver; every worker is one locked GPU process."""
    from gpu1bit.actprec import parts
    from gpu1bit.shared import seldoc
    os.environ["GPU1BIT_SELECTION"] = str(Path(selection or seldoc.CANDIDATE).resolve())
    parts.install()
    for gid in ORDER:
        require_launch(run, gid)
        args = argparse.Namespace(run=run, arm=ARM[gid], subset="full", sets=None, work_budget=fulldev.WORK_BUDGET_S,
                                  max_processes=60)
        parts.drive(args)
    for arm_id, ref_id in (("g1", "g2"), ("g4", "g5")):
        rc = subprocess.run(["timeout", "600", fulldev.PY, "-m", "gpu1bit.x2.gates", "encoder", "--run", run, "--arm", arm_id,
                             "--ref", ref_id], cwd=str(env.BINARY_DIR),
                            env={k: v for k, v in os.environ.items() if k not in ("CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES")}).returncode
        if rc != 0:
            raise SystemExit(f"encoder comparison {arm_id} vs {ref_id} failed (rc {rc})")
    record = evaluate(run)
    try:   # Z4: descriptive only; nothing here can change the gate record
        recorded(run)
    except Exception as exc:  # noqa: BLE001
        print(f"recorded-arm comparisons not written: {exc!r}", file=sys.stderr)
    return record


RECORDED_RUN = "ap-20261020r4"
RECORDED = [("g2", "MB1-int8_mma-x2"), ("g4", "M1-x2")]


def recorded(run: str) -> dict:
    """Z4/Z7: g2 and g4 compared row by row with the recorded ap-20261020r4 arms of the same name, through
    the recorded-arm binding (x2/recorded_arm.py: pinned row-file hashes, frozen identity, completeness,
    R2 of the x3 reference, ctime). Descriptive only: a refused binding is written down, never gates."""
    from gpu1bit.x2 import recorded_arm as RA
    doc = {"run": run, "descriptive_only": True, "recorded_run": RECORDED_RUN,
           "rule": "plans/x2-default.md Z4, Z7: a descriptive reference that never gates", "comparisons": {}}
    for gid, rec_arm in RECORDED:
        entry = {"arm": ARM[gid], "recorded": f"{RECORDED_RUN}/{rec_arm}"}
        try:
            out = RA.compare_with_recorded(run, ARM[gid], RECORDED_RUN, rec_arm)
            entry.update(pooled=out.get("pooled"), current=out.get("current"), recorded_identity=(out.get("recorded") or {}).get("identity"),
                         sets={n: {k: v for k, v in d.items() if k != "differing_ids"} for n, d in (out.get("sets") or {}).items()})
        except (SystemExit, ValueError, OSError, KeyError) as exc:
            entry["error"] = f"recorded binding refused: {exc}"
        doc["comparisons"][gid] = entry
    (out_dir() / f"recorded-{run}.json").write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    s = sub.add_parser("drive")
    s.add_argument("--run", required=True)
    s.add_argument("--selection", type=Path, default=None)
    s = sub.add_parser("encoder")
    s.add_argument("--run", required=True)
    s.add_argument("--arm", required=True, choices=["g1", "g4"])
    s.add_argument("--ref", required=True, choices=["g2", "g5"])
    s = sub.add_parser("evaluate")
    s.add_argument("--run", required=True)
    s = sub.add_parser("recorded", help="Z4 descriptive comparisons with ap-20261020r4 (never gates)")
    s.add_argument("--run", required=True)
    args = p.parse_args()
    from gpu1bit.shared import seldoc
    if args.action != "drive":
        seldoc.use_candidate()        # every gate action is under the candidate, also with the pointer at x3 (c4)
    if args.action == "drive":
        print(json.dumps({k: v for k, v in drive(args.run, args.selection).items() if k in ("pass",)}))
    elif args.action == "encoder":
        doc = encoder_compare(args.run, args.arm, args.ref)
        print(json.dumps({k: doc[k] for k in ("utterances", "max_rel_frobenius_valid", "above_1e-3")}))
        sys.stdout.flush()
        os._exit(0)
    elif args.action == "recorded":
        print(json.dumps({g: c.get("pooled") or c.get("error") for g, c in recorded(args.run)["comparisons"].items()}))
    else:
        print(json.dumps({k: v for k, v in evaluate(args.run).items() if k == "pass"}))


if __name__ == "__main__":
    main()
