"""Upload the staged Kea-B 0.1 folder to a new private Hub repository (release-kea-b-0.1 plan "Approval", "Upload").

    ./python -m hf.upload --approved-bundle-sha256 <the stage report's bundle_sha256>      (from finetune/relabel)

Refuses, before any Hub call, unless the stage report is PASSED, the given digest is its bundle_sha256 and the
staging folder still hashes to it (kit.approval_problems). Then: creates the repository private (it must not exist
yet), checks that the Hub reports it private, uploads the folder as one commit and records the commit in
upload.json beside the stage report. Making the repository public is a separate step, after hf/verify.py passes on a
fresh download of that commit.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

from hf import kit


class UploadRefused(RuntimeError):
    pass


def upload(report: dict, folder: Path, approved: str | None, api, *, commit_message: str | None = None) -> dict:
    """Create report["repo_id"] private through api (an HfApi or a test double), upload folder, return the record.
    UploadRefused, with nothing written to the Hub, if the approval does not hold."""
    problems = kit.approval_problems(report, folder, approved)
    if problems:
        raise UploadRefused("; ".join(problems))
    repo_id = report["repo_id"]
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=False)
    info = api.model_info(repo_id)
    if getattr(info, "private", None) is not True:
        raise UploadRefused(f"{repo_id} is not private on the Hub after creation; nothing uploaded")
    # The bytes uploaded are the bytes approved: the folder is hashed again immediately before the upload.
    if kit.bundle_sha256(repo_id, kit.listing(folder)) != approved:
        raise UploadRefused("the staged folder changed after the approval check; nothing uploaded")
    commit = api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(folder),
                               commit_message=commit_message or f"Kea-B 0.1 (bundle {approved[:12]})")
    return {"repo_id": repo_id, "private": True, "commit": getattr(commit, "oid", None),
            "commit_url": getattr(commit, "commit_url", None), "bundle_sha256": approved,
            "files": report["files"], "uploaded_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}


def main(argv: list[str] | None = None) -> None:
    import relabel_paths
    from hf import stage

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--approved-bundle-sha256", required=True)
    parser.add_argument("--report", type=Path, default=stage.REPORT)
    args = parser.parse_args(argv)
    relabel_paths.require_mount()
    report = json.loads(args.report.read_text())
    from huggingface_hub import HfApi

    try:
        record = upload(report, Path(report.get("folder", stage.STAGING)), args.approved_bundle_sha256, HfApi())
    except UploadRefused as error:
        sys.exit(f"upload refused: {error}")
    out = args.report.parent / "upload.json"
    kit.write_json(out, record)
    print(json.dumps({k: v for k, v in record.items() if k != "files"}, indent=1))
    print(f"\nUploaded privately; record: {out}\nNext: download commit {record['commit']} into a new empty folder "
          "and run ./python -m hf.verify on it.")


if __name__ == "__main__":
    main()
