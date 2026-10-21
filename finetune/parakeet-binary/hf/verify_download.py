"""Verify a fresh download of the Kea-B repo against its PASSED stage report (plans/release.md section 7, step 3).

    ./heavy kea-b-verify --mem-max 16G --runtime 20min --wait -- \
        python hf/verify_download.py DOWNLOAD_DIR /mnt/hd/wilderness-labs-stt/parakeet-binary/hf-staging/rajb3__kea-b.stage.json

Passes only if (a) every file in the stage report is present with the same SHA-256 and size and
the only other entries are the Hub's .gitattributes and the download cache .cache/, and (b) the
downloaded load_binary.py, imported by file path, loads the downloaded folder and transcribes
the report's four clips on CUDA, under the shared GPU lock, exactly as the report recorded
(the staged transcripts, which equal the test-scoring hypotheses). The result is written to
verify.json in DOWNLOAD_DIR's parent, which must be under the staging root. Never uploads.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import binary_paths
import paths

STAGING = binary_paths.ARTIFACTS / "hf-staging"
IGNORED_FILES = {".gitattributes"}
IGNORED_DIRS = {".cache"}


def file_sha256(path: Path) -> str:
    return binary_paths.sha256_file(path)


def compare_listing(report_files: dict, folder: Path) -> list[str]:
    """Differences between the stage report's {rel: {"sha256", "bytes"}} and the folder (empty if identical)."""
    folder = Path(folder)
    found = {}
    for p in sorted(folder.rglob("*")):
        rel = p.relative_to(folder)
        if not p.is_file() or rel.parts[0] in IGNORED_DIRS or rel.as_posix() in IGNORED_FILES:
            continue
        found[rel.as_posix()] = p
    problems = [f"missing {rel}" for rel in sorted(set(report_files) - set(found))]
    problems += [f"unexpected {rel}" for rel in sorted(set(found) - set(report_files))]
    for rel in sorted(set(found) & set(report_files)):
        want = report_files[rel]
        if found[rel].stat().st_size != want["bytes"] or file_sha256(found[rel]) != want["sha256"]:
            problems.append(f"{rel} differs from the staged file")
    return problems


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="the fresh download (hf download ... --local-dir)")
    parser.add_argument("report", type=Path, help="the PASSED stage report")
    args = parser.parse_args(argv)
    if not os.environ.get("INVOCATION_ID"):
        sys.exit("run through ./heavy (loads the model, decodes on CUDA): ./heavy kea-b-verify --mem-max 16G "
                 "--runtime 20min --wait -- python hf/verify_download.py DOWNLOAD_DIR STAGE_REPORT")
    import binary_train  # noqa: F401  configures CUDA as for the test scoring and the staging run
    import torch

    import evaluate

    binary_paths.require_mount()
    folder = args.folder.resolve()
    if not folder.parent.is_relative_to(STAGING.resolve()):
        sys.exit(f"the download must be under {STAGING}")
    report = json.loads(args.report.read_text())
    if report.get("status") != "PASSED":
        sys.exit(f"{args.report} is not a PASSED stage report")
    result = {"folder": str(folder), "report": str(args.report.resolve()),
              "report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest()}
    result["file_problems"] = compare_listing(report["files"], folder)
    clips = report["clips"]
    if not result["file_problems"]:
        if not torch.cuda.is_available():
            sys.exit("CUDA is required: the recorded transcripts were decoded on CUDA")
        spec = importlib.util.spec_from_file_location("downloaded_load_binary", folder / "load_binary.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.load(folder, device="cpu")
        records = {r["id"]: r for r in evaluate.read_manifest(evaluate.resolve_manifest("librispeech_clean")[1])}
        with paths.gpu_lock("kea-b-verify"):
            model.to(torch.device("cuda"))
            hyps = module.transcribe(model, [records[c["id"]]["audio_filepath"] for c in clips])
            del model
            torch.cuda.empty_cache()
        result["clips"] = [{"id": c["id"], "downloaded": h, "staged": c["staged"], "test_scoring": c["test_scoring"],
                            "equal": h == c["staged"] == c["test_scoring"]} for c, h in zip(clips, hyps, strict=True)]
    result["passed"] = (not result["file_problems"] and len(clips) == 4
                        and all(c["equal"] for c in result.get("clips", [])) and len(result.get("clips", [])) == 4)
    out = folder.parent / "verify.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    print(f"\nVERIFY {'PASSED' if result['passed'] else 'FAILED'} -> {out}")
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
