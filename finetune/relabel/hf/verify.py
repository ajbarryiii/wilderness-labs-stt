"""Verify a download (or any local copy) of the Kea-B 0.1 repository against the PASSED stage report
(release-kea-b-0.1 plan "Upload": checks a-c). Never uploads.

    job start kea-b-0.1-verify --needs gpu:shared --vram 14G --mem 14G --priority normal \\
        --env HF_HUB_OFFLINE=1 -- ./python -m hf.verify FOLDER          (from finetune/relabel)
    ./python -m hf.verify FOLDER --listing-only                          (a: listing and hashes only; CPU)

(a) Listing: every file of the stage report is present with its SHA-256 and size, and the only other entries are
    the Hub's .gitattributes and the download cache .cache/.
(b) The folder's own load_kea_b.py, imported by file path, loads the bundle (FOLDER) and the compat view
    (FOLDER/compat) to the same state, bit for bit, and the bundle model transcribes the report's check clips on
    CUDA, under the shared GPU lock and per set as the stage did, exactly as recorded (the staged transcripts, which
    equal K-fp's hypotheses).
(c) That model, with the repository's evaluator (evaluate.evaluate_set at K-fp's batch 64 and 1200 s, decoder pinned
    to no_while_loops), reproduces K-fp's LibriSpeech test-clean result exactly: the unrounded WER and its edit
    counts, compared with K-fp's recorded set file.
Measured 2026-11-04 on the staging folder: host peak 10.8 GB, GPU 12.0 GB allocated (K-fp's batch 64), 40 s once
the shared GPU lock is free. The result goes to --out (default: verify/<folder name>-<UTC>.json beside the stage report); exit 0 only if every
check passed. Running it on the staging folder itself is the dry run.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import sys
from pathlib import Path

from hf import kit

KFP_SET = "librispeech_clean"


def listing_check(report: dict, folder: Path, checks: dict, result: dict) -> None:
    problems = kit.compare_listing(report["files"], folder)
    result["file_problems"] = problems
    checks["a listing and hashes == stage report"] = not problems


def model_checks(report: dict, folder: Path, checks: dict, result: dict) -> None:
    import train_keab as tk   # first, as keab_test (train.py's CUDA/numba settings before torch)
    import torch

    import binary_train
    import evaluate
    import keab_step0 as k0
    import keab_test_paths as tp
    from hf import stage

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required: the recorded transcripts and K-fp were decoded on CUDA")
    spec = importlib.util.spec_from_file_location("downloaded_load_kea_b", folder / kit.LOADER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result["loader_file"] = module.__file__
    with torch.no_grad():
        model = module.load(folder, device="cpu")
        compat = module.load(folder / kit.COMPAT, device="cpu")
        diff = stage.compare_states(model.state_dict(), compat.state_dict())
        del compat
    checks["b bundle state == compat state, bit for bit"] = not diff
    result["state_differing"] = diff[:10]

    clips = report["clips"]
    kfp_file = tp.test_dir("fp") / f"{KFP_SET}.json"
    kfp_summary = json.loads((tp.test_dir("fp") / "summary.json").read_text())
    kfp = json.loads(kfp_file.read_text())
    checks["c K-fp set file == its summary's SHA-256"] = (
        kit.sha256_file(kfp_file) == kfp_summary["per_set_sha256"][KFP_SET])
    tk.vram_lease()
    device = torch.device("cuda")
    with k0.waiting_lock("kea-b-0.1-verify", 6 * 3600.0):
        model.to(device)
        texts = {}
        for name in dict.fromkeys(c["set"] for c in clips):
            group = [c for c in clips if c["set"] == name]
            for c, text in zip(group, module.transcribe(model, [c["audio_filepath"] for c in group]), strict=True):
                texts[(name, c["id"])] = text
        binary_train.pin_decoder_graphs_mode(model, "downloaded model")
        source = {"kind": "export", "description": "Kea-B 0.1 bundle, downloaded, rebuilt by its load_kea_b.py",
                  "path": str(folder), "sha256": report["files"][kit.BUNDLE_WEIGHTS]["sha256"]}
        before = binary_train.require_decoder_graphs_mode(model, "downloaded model before")
        scored = evaluate.evaluate_set(model, KFP_SET, tp.manifest_path(KFP_SET), source, tp.BATCH_SIZE, None, device,
                                       max_batch_seconds=tp.MAX_BATCH_SECONDS)
        after = binary_train.require_decoder_graphs_mode(model, "downloaded model after")
        del model
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    result["clips"] = [{"set": c["set"], "id": c["id"], "downloaded": texts[(c["set"], c["id"])],
                        "staged": c["staged"], "kfp": c["kfp"]} for c in clips]
    for c in result["clips"]:
        checks[f"b clip {c['set']}/{c['id']}: transcript == staged == K-fp"] = c["downloaded"] == c["staged"] == c["kfp"]
    checks["b at least one check clip"] = bool(clips)
    hyps = {r["id"]: r["hyp"] for r in kfp["records"]}
    differing = [r["id"] for r in scored["records"] if hyps.get(r["id"]) != r["hyp"]]
    result["kfp_reproduction"] = {"set": KFP_SET, "wer": scored["wer"], "kfp_wer": kfp["wer"],
                                  "wer_percent": 100 * scored["wer"]["wer"],
                                  "kfp_summary_percent": kfp_summary["sets"][KFP_SET]["wer"],
                                  "differing_hypotheses": len(differing), "first_differing": differing[:10],
                                  "decoder_graphs_mode": {"before": before, "after": after},
                                  "peak_gpu_memory_gb": scored.get("peak_gpu_memory_gb")}
    checks["c LibriSpeech test-clean WER == K-fp's, unrounded (with S, D, I and word counts)"] = (
        scored["wer"] == kfp["wer"] and 100 * scored["wer"]["wer"] == kfp_summary["sets"][KFP_SET]["wer"])
    checks["c every LibriSpeech test-clean hypothesis == K-fp's"] = not differing and len(scored["records"]) == len(hyps)


def main(argv: list[str] | None = None) -> None:
    import relabel_paths
    from hf import stage

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--report", type=Path, default=stage.REPORT)
    parser.add_argument("--listing-only", action="store_true", help="check (a) only (e.g. the anonymous re-download)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    relabel_paths.require_mount()
    folder = args.folder.resolve()
    report = json.loads(args.report.read_text())
    if report.get("status") != "PASSED":
        sys.exit(f"{args.report} is not a PASSED stage report")
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or args.report.parent / "verify" / f"{folder.name}-{stamp}{'-listing' if args.listing_only else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    checks: dict[str, bool] = {}
    result: dict = {"folder": str(folder), "report": str(args.report.resolve()),
                    "report_sha256": kit.sha256_file(args.report), "bundle_sha256": report.get("bundle_sha256"),
                    "listing_only": args.listing_only, "started_utc": stamp}
    try:
        listing_check(report, folder, checks, result)
        if not args.listing_only and checks["a listing and hashes == stage report"]:
            model_checks(report, folder, checks, result)
        elif not args.listing_only:
            result["error"] = "listing differs; the model checks were not run"
    except Exception as error:  # noqa: BLE001  recorded, then the run fails
        result["error"] = f"{type(error).__name__}: {error}"
    result["checks"] = checks
    result["passed"] = bool(checks) and all(checks.values()) and "error" not in result and (
        args.listing_only or len(checks) > 3)
    kit.write_json(out, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("clips",)}, indent=1, default=str))
    print(f"\nVERIFY {'PASSED' if result['passed'] else 'FAILED'} -> {out}")
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
