"""Zero-step points and the G-B gate of the binary pilot, scored like an arm's final model. See plans/pilot.md "Zero-step points".

    python binary_eval.py --point {Z-A,Z-B,G-B} [--limit N] [--overwrite]

- Z-A: the FP32 original binarized (binary post-training quantization, f = 1): exported,
  rebuilt from the export, reconstruction-checked, scored on the full development sets.
- Z-B: M1 binarized (M1's student state, binary modules at f = 1): exported, rebuilt,
  checked, scored.
- G-B: arm B's model before any update (M1's student state, ternary-to-binary ramp at f = 0),
  scored as the QAT model itself (f < 1 cannot be exported). The gate passes only if (1) its
  full-dev mean WER equals M1's recorded full-dev mean (M1 summary.json "dev_mean_wer", scored
  on M1's rebuilt export) within GATE_POINTS = 0.02 WER points, and (2) on the reconstruction
  batch its encoder output equals the ternary M1 model's bit for bit (that ternary model is
  also control C's start). binary_sweep.py runs nothing else unless the gate passed.

Initialization and quantization are binary_train's (read_init_state, apply_init_state: pinned
SHA-256, step 250,000, f = 1, exact key set) and binary_quant's. Scoring is
train.DevEvaluator.full with train.py's default evaluation batch size (32): the decoding,
normalizer, sets and batching of every arm's final full-dev score and of M1's. Writes
EVAL/<point>/<set>.json (per-utterance records) and EVAL/<point>/summary.json last; exports
go to RUNS/<point>/export. Takes the (shared) GPU lock. --limit is for smoke tests only: it
writes EVAL/smoke-<point>-limit<N> (and RUNS/smoke-<point>-limit<N>/export), never the
production path, and carries no gate verdict.

Every model is decoded with binary_train's pinned NeMo decoder mode (recorded per model in
"decoder_graphs"), and a zero-step point whose rebuilt export does not reproduce the codes and
scales exactly raises binary_train.ReconstructionError before it is scored (review round 1).
"""
from __future__ import annotations

import binary_train  # noqa: I001  imports train first (CUDA/numba configuration)

import argparse
import json
import sys
import time

import torch

import binary_export
import binary_paths
import binary_quant
import paths
import quant as ternary_quant
import train

POINTS = {
    "Z-A": {"init": "fp32", "ramp": "fp32", "fraction": 1.0, "export": True,
            "description": "FP32 original binarized (binary PTQ, no training), rebuilt from its export"},
    "Z-B": {"init": "m1", "ramp": "ternary", "fraction": 1.0, "export": True,
            "description": "M1 binarized (no retraining), rebuilt from its export"},
    "G-B": {"init": "m1", "ramp": "ternary", "fraction": 0.0, "export": False,
            "description": "arm B before any update: M1, ternary-to-binary ramp at f = 0 (QAT model)"},
}
GATE_POINTS = 0.02  # WER points, plans/pilot.md "Gate G-B"
EVAL_BATCH_SIZE = 32  # train.py --eval-batch-size default, used for every arm's (and M1's) full dev
EVAL_SOURCES = binary_paths.BINARY_SOURCES + ("parakeet-binary/binary_eval.py",)


def log(msg: str) -> None:
    print(f"[binary_eval {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def source_hashes() -> dict[str, str]:
    """SHA-256 of train.py's TRAINING_SOURCES (as M1 recorded them), the binary training sources
    and this script; binary_sweep.py compares them with the files on disk before reuse."""
    root = paths.REPO / "finetune"
    names = tuple(binary_train._TRAIN["TRAINING_SOURCES"]) + EVAL_SOURCES
    return {name: train.sha256(root / name) for name in names}


def output_name(point: str, limit: int | None) -> str:
    """EVAL/<name> (and RUNS/<name>/export): the point itself for a full evaluation, a separate
    smoke-<point>-limit<N> for a limited one, which never occupies the production path."""
    return point if limit is None else f"smoke-{point}-limit{limit}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--point", choices=list(POINTS), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--limit", type=int, help="first N utterances per dev set (smoke only; no verdict)")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


@torch.no_grad()
def ternary_twin_check(evaluator, model, twin) -> dict:
    """Encoder outputs of the binary model and its ternary twin on the reconstruction batch (the
    first utterances of the first dev subset), strict FP32, inference settings."""
    ev = evaluator.evaluate
    records = next(iter(evaluator.subsets.values()))[:train.RECONSTRUCTION_UTTERANCES]
    audio, lengths = ev.audio_batch(records)
    device = next(model.parameters()).device
    outputs = []
    with ev.strict_fp32(device):
        for m in (model, twin):
            with ev.inference_settings(m):
                outputs.append(m.forward(input_signal=audio.to(device), input_signal_length=lengths.to(device)))
    (enc_b, len_b), (enc_t, len_t) = outputs
    equal = torch.equal(len_b, len_t) and torch.equal(enc_b, enc_t)
    return {"encoder_equal": bool(equal),
            "encoder_max_abs_diff": float((enc_b - enc_t).abs().max()) if enc_b.shape == enc_t.shape else None,
            "utterance_ids": [r["id"] for r in records]}


def score_point(point: str, device: str, batch_size: int, limit: int | None) -> dict:
    """Build, (export, rebuild, check,) and score one point; returns the summary record."""
    spec = POINTS[point]
    base = train.verify_base()
    model = train.load_pretrained("cpu")
    init: dict = {"init": spec["init"], "sha256": base["base_sha256"]}
    state = None
    if spec["init"] == "m1":
        state, init = binary_train.read_init_state()
        init = {"init": "m1", **init, "tensors": binary_train.apply_init_state(model, state)}
    names = binary_quant.quantize_parakeet(model, spec["ramp"])
    binary_quant.set_weight_fraction(model, spec["fraction"])
    model.to(device).eval()
    evaluator = binary_train.DecoderPinnedEvaluator(train.DevEvaluator(limit, batch_size, device))
    record = {"point": point, "description": spec["description"], "init": init, "ramp": spec["ramp"],
              "weight_fraction": spec["fraction"], "quantized_modules": len(names)}
    graphs: dict[str, str | None] = {}
    if spec["export"]:
        export = binary_train.FinishingExport(binary_export)  # pinned rebuild, exact reconstruction
        graphs["qat"] = binary_train.pin_decoder_graphs_mode(model, "QAT model")  # decodes in the check
        run = binary_paths.run_dir(output_name(point, limit))
        manifest = export.export_model(model, run / "export", extra={
            "point": point, "limit": limit, "description": spec["description"], "init": init,
            "source_hashes": source_hashes()})
        train.drop_tree_cache(run / "export")
        scorer = export.load_export(run / "export", device)
        graphs["rebuilt"] = export.rebuilt_mode
        check = train.reconstruction(export, evaluator, model, scorer)  # ReconstructionError if inexact
        train.write_json(run / "export" / "reconstruction.json", check)
        record.update(export_dir=str(run / "export"), export_bytes=manifest["bytes"],
                      code_histogram={k: v for k, v in manifest["code_histogram"].items() if k != "per_module"},
                      reconstruction=check, scored_on="model rebuilt from export")
        del model
    else:
        twin = train.load_pretrained("cpu")
        binary_train.apply_init_state(twin, state)
        ternary_quant.quantize_parakeet(twin)
        twin.to(device).eval()
        record["ternary_twin"] = ternary_twin_check(evaluator, model, twin)
        del twin
        scorer = model
        graphs["student"] = binary_train.pin_decoder_graphs_mode(model, "student")
        record["scored_on"] = "QAT model at weight fraction 0 (not exportable)"
    del state
    if device != "cpu":
        torch.cuda.empty_cache()
    scorer.eval()
    with torch.no_grad():
        full = evaluator.full(scorer)
    graphs["full-dev"] = evaluator.modes["full-dev"]
    record.update(decoder_graphs=graphs, dev_wer={n: r["wer"] for n, r in full.items()},
                  dev_utterances={n: r["utterances"] for n, r in full.items()},
                  dev_mean_wer=train.mean_wer(full), limit=limit, batch_size=batch_size)
    record["_records"] = full
    return record


def gate_verdict(record: dict) -> dict:
    """G-B: full-dev mean within GATE_POINTS of M1's recorded value and the ternary twin equal."""
    m1 = json.loads((binary_paths.M1_RUN / "summary.json").read_text())
    diff = 100 * (record["dev_mean_wer"] - m1["dev_mean_wer"])
    full = record["limit"] is None
    passed = abs(diff) <= GATE_POINTS + 1e-12 and record["ternary_twin"]["encoder_equal"]
    return {"reference": f"{binary_paths.M1_RUN / 'summary.json'} dev_mean_wer (full dev, M1's rebuilt export)",
            "m1_dev_mean_wer": m1["dev_mean_wer"], "m1_dev_wer": m1["dev_wer"],
            "difference_points": diff, "tolerance_points": GATE_POINTS,
            "per_set_difference_points": {n: 100 * (record["dev_wer"][n] - m1["dev_wer"][n])
                                          for n in record["dev_wer"]} if full else None,
            "ternary_twin_encoder_equal": record["ternary_twin"]["encoder_equal"],
            "pass": bool(passed) if full else None}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    binary_paths.require_mount()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("CUDA is required")
    name = output_name(args.point, args.limit)
    out = binary_paths.EVAL / name
    if (out / "summary.json").exists() and not args.overwrite:
        sys.exit(f"{out / 'summary.json'} exists; pass --overwrite to rescore {name}")
    started = train.utc()
    with paths.gpu_lock(f"binary_eval {name}"):
        out = binary_paths.eval_dir(name)
        for stale in out.glob("*.json"):  # a rescore replaces every file; summary.json first
            stale.unlink()
        train.deterministic_mode()
        record = score_point(args.point, args.device, args.batch_size, args.limit)
    full = record.pop("_records")
    for name, r in full.items():
        train.write_json(out / f"{name}.json", {"set": name, "wer": r["wer"], "utterances": r["utterances"],
                                                "records": r["records"], "source": record["scored_on"]})
    if args.point == "G-B":
        record["gate"] = gate_verdict(record)
    record.update(source_hashes=source_hashes(), versions=train.versions(), git_head=train.git_head(),
                  started_utc=started, finished_utc=train.utc())
    train.write_json(out / "summary.json", record)  # last write
    log(f"{args.point}: full dev mean WER {100 * record['dev_mean_wer']:.3f}% "
        f"({', '.join(f'{n} {100 * w:.2f}' for n, w in record['dev_wer'].items())})")
    if "gate" in record:
        gate = record["gate"]
        log(f"G-B gate: {gate['difference_points']:+.3f} points vs M1 (tolerance {GATE_POINTS}), ternary twin "
            f"{'equal' if gate['ternary_twin_encoder_equal'] else 'DIFFERENT'}: "
            f"{ {True: 'PASS', False: 'FAIL', None: 'no verdict (limited)'}[gate['pass']] }")


if __name__ == "__main__":
    main()
