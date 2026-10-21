"""Assemble a Hugging Face upload folder for the M1 ternary export, then check it standalone.

    ./heavy hf-stage --mem-max 24G --runtime 30min --wait -- python hf/stage.py USER/NAME [--export DIR]

(run from finetune/parakeet-ternary; it loads the model twice, so it refuses to run outside a
systemd unit such as the one ./heavy creates.)

Every check is recorded by name in the report, and success is the conjunction of all of them
(the set of names is fixed in advance, so none can be skipped silently):

1. Provenance: the export is the M1 final export the model card describes, and every file both
   loaders consume is bound to the evaluated artifact. export.safetensors (180,797,564 bytes)
   has the SHA-256 recorded in manifest.json, eval/M1-precheck.json (which passed),
   results/test.json and the test scoring (eval/M1-test/summary.json). manifest.json's format,
   sha256, accounting, bytes and extra equal what the test scoring recorded; its identity and
   config_sha256 equal runs/<run>/summary.json; its NeMo config equals the pinned base .nemo's
   model_config.yaml; its tokenizer hashes match the files, which equal the base .nemo's
   tokenizer members. reconstruction.json equals summary.json's and is exact. manifest.json and
   reconstruction.json also match SHA-256 values pinned here after those checks.
2. LICENSE (pinned CC-BY-4.0 legal code), load_ternary.py imports (stdlib + documented deps).
3. Model card: valid YAML (huggingface_hub.ModelCard), no placeholders, license, model name,
   model-index values = results/test.json, results table rows = results/TEST.md, file sizes.
4. Staging: files are written to a temporary sibling folder; every staged file's SHA-256 must
   equal its source and the file set must be exactly the expected one.
5. Standalone: the staged load_ternary.py (imported by file path; no repository module) and the
   repository's export.load_export, both loading the same export, give identical state_dicts
   (torch.equal for every key, all FP32, eval mode) and identical greedy transcripts on four fixed
   LibriSpeech test-clean clips, which must also equal the hypotheses recorded by the test scoring.

Only if every check passes is the temporary folder renamed to STAGING/USER__NAME and the report
written to STAGING/USER__NAME.stage.json (every staged file with SHA-256 and size). Any stale
folder or report is removed first; on failure no final folder exists and the report is written
to STAGING/USER__NAME.stage.FAILED.json. Writes only under STAGING. Never uploads; the upload
command is printed.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tarfile
import traceback
from pathlib import Path

import torch

import evaluate
import export
import paths

HERE = Path(__file__).resolve().parent
STAGING = paths.ARTIFACTS / "hf-staging"
RUN_NAME = "main-M1-P2-lr5e-4"
DEFAULT_EXPORT = paths.RUNS / RUN_NAME / "export"
RUN_SUMMARY = paths.RUNS / RUN_NAME / "summary.json"
PRECHECK = paths.EVAL / "M1-precheck.json"
TEST_SCORING = paths.EVAL / "M1-test"
TEST_JSON = HERE.parent / "results" / "test.json"
TEST_MD = HERE.parent / "results" / "TEST.md"
# MODEL_CARD.md states this export's numbers; only it may be staged.
EXPECTED = {"run_name": RUN_NAME, "arm": "M1", "select": "final", "selected_step": 250000}
FILE_BYTES = 180797564
IDENTITY_KEYS = ("run_name", "arm", "recipe", "lr", "select", "selected_step", "config_sha256")
# Pinned 2026-10-04 after the cross-checks in provenance_checks passed on these exact files.
MANIFEST_SHA256 = "5c8b4bea0393db4072351e07b0d2206079373ca52646efdec4bb30876e0c8c61"
RECONSTRUCTION_SHA256 = "6aa71b9e7226747ccff3a02ae503c7407f381dc6f773f3b9739a1d9abbed6636"
LICENSE_SHA256 = "9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411"  # legalcode.txt, fetched 2026-10-04
CARD_LICENSE = "cc-by-4.0"
# Card table columns, in TEST.md order (paths.MEAN_SETS, mean, common_voice).
CARD_HEADER = ("| Model | LS clean | LS other | AMI | Earnings-22 | GigaSpeech | SPGISpeech | VoxPopuli "
               "| Mean of 7 | Common Voice |")
PLACEHOLDERS = ("{repo_id}", "{repo_name}", "{code_histogram}")
CHECK_MANIFEST = paths.MANIFESTS / "test_librispeech_clean.jsonl"
CHECK_IDS = ("1089-134686-0000", "1089-134686-0001", "1089-134686-0002", "1089-134686-0003")
# Modules the staged loader may import: stdlib plus the documented dependencies.
LOADER_IMPORTS = {"__future__", "contextlib", "hashlib", "json", "math", "sys", "pathlib", "numpy",
                  "torch", "safetensors", "soundfile", "nemo", "omegaconf"}
TOKENIZER_RELS = tuple(f"{export.TOKENIZER_DIR}/{f}" for f in export.TOKENIZER_FILES.values())
EXPORT_RELS = (export.FILE, *TOKENIZER_RELS, export.MANIFEST, "reconstruction.json")
HF_FILES = ("load_ternary.py", "LICENSE", "NOTICE")
STAGED_FILES = tuple(sorted((*EXPORT_RELS, *HF_FILES, "README.md")))


class StageFailed(Exception):
    pass


def sha256(path: Path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def base_nemo_members(manifest: dict) -> tuple[dict, dict[str, bytes]]:
    """(resolved model_config.yaml, {tokenizer rel path: bytes}) from the pinned base .nemo."""
    from omegaconf import OmegaConf

    refs = {Path(str(manifest["config"]["tokenizer"][key]).removeprefix("nemo:")).name: rel
            for key, rel in zip(export.TOKENIZER_FILES, TOKENIZER_RELS)}
    config, tokenizer = None, {}
    with tarfile.open(paths.MODEL_FILE, "r:*") as tar:
        for member in tar:
            name = Path(member.name).name
            if member.isfile() and name == "model_config.yaml":
                config = OmegaConf.to_container(OmegaConf.create(tar.extractfile(member).read().decode()),
                                                resolve=True)
            elif member.isfile() and name in refs:
                tokenizer[refs[name]] = tar.extractfile(member).read()
    return config, tokenizer


def provenance_checks(export_dir: Path, checks: dict, info: dict) -> dict:
    """Record provenance checks; return the manifest."""
    manifest = read_json(export_dir / export.MANIFEST)
    recon = read_json(export_dir / "reconstruction.json")
    precheck, test, summary = read_json(PRECHECK), read_json(TEST_JSON), read_json(RUN_SUMMARY)
    scoring = read_json(TEST_SCORING / "summary.json")["source"]
    digest = sha256(export_dir / export.FILE)
    extra = manifest["extra"]
    info["artifact_sha256"] = {rel: sha256(export_dir / rel) for rel in EXPORT_RELS}
    info["export_sha256"] = digest

    checks["manifest format is parakeet-ternary-v1 and file is export.safetensors"] = (
        manifest["format"] == export.FORMAT and manifest["file"] == export.FILE)
    checks["manifest identity is the M1 final export (run, arm, select, step)"] = (
        {k: extra.get(k) for k in EXPECTED} == EXPECTED)
    checks["export.safetensors is 180,797,564 bytes (file and manifest)"] = (
        (export_dir / export.FILE).stat().st_size == FILE_BYTES == manifest["bytes"]["file_bytes"])
    checks["export SHA-256 == manifest sha256 == manifest files[export.safetensors]"] = (
        digest == manifest["sha256"] == manifest["files"].get(export.FILE))
    checks["precheck passed (pass true, every check ok)"] = (
        precheck.get("pass") is True and bool(precheck["checks"]) and all(c["ok"] for c in precheck["checks"]))
    checks["export SHA-256 == precheck export_sha256"] = precheck.get("export_sha256") == digest
    checks["export SHA-256 == results/test.json m1_export_sha256"] = test.get("m1_export_sha256") == digest
    checks["export SHA-256 == test scoring source sha256"] = (
        scoring.get("kind") == "export" and scoring.get("sha256") == digest
        and Path(scoring.get("path", "")).resolve() == export_dir)
    recorded = scoring.get("export_manifest", {})
    checks["manifest format/sha256/accounting/bytes/extra == those recorded by the test scoring"] = (
        set(recorded) == {"format", "sha256", "parameter_accounting", "bytes", "extra"}
        and all(manifest[k] == v for k, v in recorded.items()))
    checks["manifest identity and config_sha256 == run summary.json"] = all(
        extra.get(k) == summary.get(k) for k in IDENTITY_KEYS)
    checks["run summary export_dir, export_bytes and code histogram == manifest"] = (
        Path(summary["export_dir"]).resolve() == export_dir and summary["export_bytes"] == manifest["bytes"]
        and summary["code_histogram"] == {k: manifest["code_histogram"][k] for k in ("minus_one", "zero", "plus_one")})
    checks["manifest lists exactly export.safetensors and the three tokenizer files"] = (
        set(manifest["files"]) == {export.FILE, *TOKENIZER_RELS})
    checks["tokenizer files match the manifest SHA-256"] = all(
        info["artifact_sha256"][rel] == manifest["files"].get(rel) for rel in TOKENIZER_RELS)
    checks["manifest base_model is the pinned base model"] = manifest["base_model"] == {
        "id": paths.MODEL_ID, "revision": paths.MODEL_REVISION, "license": "CC-BY-4.0"}
    lock = read_json(paths.MODEL_DIR / "lock.json")
    nemo_digest = sha256(paths.MODEL_FILE)
    checks["base .nemo SHA-256 == lock.json == test scoring base_model"] = (
        lock["revision"] == paths.MODEL_REVISION
        and nemo_digest == lock["files"][paths.MODEL_FILE.name] == scoring["base_model"]["nemo_sha256"])
    nemo_config, nemo_tokenizer = base_nemo_members(manifest)
    checks["manifest config == base .nemo model_config.yaml (resolved)"] = nemo_config == manifest["config"]
    checks["tokenizer files == base .nemo tokenizer members"] = (
        set(nemo_tokenizer) == set(TOKENIZER_RELS)
        and all((export_dir / rel).read_bytes() == data for rel, data in nemo_tokenizer.items()))
    checks["reconstruction.json == run summary reconstruction"] = recon == summary.get("reconstruction")
    checks["reconstruction exact: codes, scales, greedy hyps, all 264 layers"] = (
        recon["codes_exact"] is True and recon["scales_exact"] is True and recon["greedy_hyps_equal"] is True
        and recon["checked_layers"] == len(manifest["quantized_layers"]) == 264)
    checks["manifest.json SHA-256 == pinned"] = info["artifact_sha256"][export.MANIFEST] == MANIFEST_SHA256
    checks["reconstruction.json SHA-256 == pinned"] = (
        info["artifact_sha256"]["reconstruction.json"] == RECONSTRUCTION_SHA256)
    return manifest


def file_checks(checks: dict) -> None:
    text = (HERE / "LICENSE").read_text()
    checks["LICENSE is the pinned CC-BY-4.0 legal code"] = (
        sha256(HERE / "LICENSE") == LICENSE_SHA256 and text.startswith("Attribution 4.0 International")
        and "Section 1 -- Definitions." in text and "Section 8 -- Interpretation." in text)
    names = set()
    for node in ast.walk(ast.parse((HERE / "load_ternary.py").read_text())):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add("." * node.level + (node.module or "") if node.level else node.module.split(".")[0])
    checks["load_ternary.py imports only stdlib and documented dependencies"] = names <= LOADER_IMPORTS
    checks["load_ternary.py carries the MIT copyright and permission notice"] = all(
        s in (HERE / "load_ternary.py").read_text() for s in (
            "# MIT License", "# Copyright (c) 2026 ajbarryiii", "# Permission is hereby granted, free of charge",
            "# The above copyright notice and this permission notice shall be included in all"))


def render_card(repo_id: str, manifest: dict) -> str:
    hist = manifest["code_histogram"]
    summary = (f"{100 * hist['minus_one']:.1f}% -1, {100 * hist['zero']:.1f}% 0, "
               f"{100 * hist['plus_one']:.1f}% +1")
    text = (HERE / "MODEL_CARD.md").read_text()
    for key, value in {"{repo_id}": repo_id, "{repo_name}": repo_id.split("/")[1],
                       "{code_histogram}": summary}.items():
        text = text.replace(key, value)
    return text


def card_checks(card: str, repo_id: str, checks: dict, info: dict) -> None:
    from huggingface_hub import ModelCard

    checks["card has no unrendered placeholder"] = not any(p in card for p in PLACEHOLDERS)
    data = ModelCard(card).data  # raises on invalid YAML front matter
    checks["card YAML: model_name, license cc-by-4.0, base_model, library_name nemo"] = (
        data.model_name == repo_id.split("/")[1] and data.license == CARD_LICENSE
        and data.base_model == paths.MODEL_ID and data.library_name == "nemo")
    m1 = read_json(TEST_JSON)["rows"]["M1"]
    by_dataset = {v: k for k, v in paths.TEST_SETS.items()}
    seen, index_ok = {}, True
    for result in data.eval_results or []:
        key = by_dataset.get((result.dataset_config, result.dataset_split))
        index_ok &= (key is not None and key not in seen and result.dataset_type == paths.ESB_REPO
                     and result.dataset_revision == paths.ESB_REVISION and result.metric_type == "wer"
                     and result.metric_value == round(m1[key], 2))
        seen[key] = result.metric_value
    checks["card model-index: one entry per test set, value == round(test.json M1, 2)"] = (
        index_ok and set(seen) == set(paths.TEST_SETS))
    rows = [line for line in TEST_MD.read_text().splitlines() if line.startswith("| ")]
    header = [c.strip() for c in rows[0].strip("|").split("|")]
    matched, missing = 0, []
    for line in rows[1:]:
        cells = [c.strip() for c in line.strip("|").split("|")][1:]
        if set("".join(cells)) <= set("-: "):
            continue
        if "| " + " | ".join(cells) + " |" in card:
            matched += 1
        else:
            missing.append(line.split("|")[1].strip())
    checks["card results table reproduces all 6 TEST.md rows in TEST.md column order"] = (
        header[1:] == paths.MEAN_SETS + ["mean", "common_voice"] and CARD_HEADER in card
        and matched == 6 and not missing)
    test = read_json(TEST_JSON)
    checks["card file sizes == test.json (180.8 MB, 2,472 MB)"] = all(
        s in card for s in (f"**{test['m1_export_mb']:.1f} MB**", f"{test['original_nemo_mb']:,.0f} MB"))
    info["card"] = {"model_name": data.model_name, "license": data.license, "base_model": data.base_model,
                    "model_index_results": len(seen), "test_md_rows_matched": matched, "missing_rows": missing}


@torch.no_grad()
def standalone_checks(folder: Path, reference_export: Path, checks: dict, info: dict) -> None:
    spec = importlib.util.spec_from_file_location("staged_load_ternary", folder / "load_ternary.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    staged = module.load(folder, device="cpu")
    reference = export.load_export(reference_export, device="cpu")
    a, b = staged.state_dict(), reference.state_dict()
    differing = [k for k in a if k not in b or a[k].dtype != b[k].dtype or a[k].shape != b[k].shape
                 or not torch.equal(a[k], b[k])]
    checks["state_dict: same keys in the same order"] = list(a) == list(b)
    checks["state_dict: every tensor torch.equal"] = not differing and len(a) == len(b) > 0
    checks["state_dict: every floating tensor FP32"] = all(
        v.dtype == torch.float32 for v in a.values() if v.is_floating_point())
    checks["staged model in eval mode"] = not staged.training
    info["loader_module_file"] = module.__file__
    info["state_dict_keys"] = len(a)
    info["state_dict_differing"] = differing[:10]
    del a, b
    records = {r["id"]: r for r in evaluate.read_manifest(CHECK_MANIFEST)}
    records = [records[i] for i in CHECK_IDS]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lock = paths.gpu_lock("hf-stage") if device.type == "cuda" else contextlib.nullcontext()
    with lock:
        staged.to(device)
        reference.to(device)
        hyps_staged = module.transcribe(staged, [r["audio_filepath"] for r in records])
        hyps_reference = evaluate.transcribe_records(reference, records, batch_size=len(records),
                                                     device=device, workers=0)
    scored = {r["id"]: r["hyp"] for r in read_json(TEST_SCORING / "librispeech_clean.json")["records"]}
    for r, hs, hr in zip(records, hyps_staged, hyps_reference, strict=True):
        checks[f"clip {r['id']}: staged transcript == repository transcript"] = hs == hr
        checks[f"clip {r['id']}: staged transcript == test-scoring hypothesis"] = hs == scored.get(r["id"])
    info["device"] = device.type
    info["clips"] = [{"id": r["id"], "reference": r["text"], "staged": hs, "repository": hr,
                      "test_scoring": scored.get(r["id"])}
                     for r, hs, hr in zip(records, hyps_staged, hyps_reference)]


def required_checks() -> set[str]:
    """Every check name a successful run must record (any missing name fails the run)."""
    names = {
        "manifest format is parakeet-ternary-v1 and file is export.safetensors",
        "manifest identity is the M1 final export (run, arm, select, step)",
        "export.safetensors is 180,797,564 bytes (file and manifest)",
        "export SHA-256 == manifest sha256 == manifest files[export.safetensors]",
        "precheck passed (pass true, every check ok)",
        "export SHA-256 == precheck export_sha256",
        "export SHA-256 == results/test.json m1_export_sha256",
        "export SHA-256 == test scoring source sha256",
        "manifest format/sha256/accounting/bytes/extra == those recorded by the test scoring",
        "manifest identity and config_sha256 == run summary.json",
        "run summary export_dir, export_bytes and code histogram == manifest",
        "manifest lists exactly export.safetensors and the three tokenizer files",
        "tokenizer files match the manifest SHA-256",
        "manifest base_model is the pinned base model",
        "base .nemo SHA-256 == lock.json == test scoring base_model",
        "manifest config == base .nemo model_config.yaml (resolved)",
        "tokenizer files == base .nemo tokenizer members",
        "reconstruction.json == run summary reconstruction",
        "reconstruction exact: codes, scales, greedy hyps, all 264 layers",
        "manifest.json SHA-256 == pinned",
        "reconstruction.json SHA-256 == pinned",
        "LICENSE is the pinned CC-BY-4.0 legal code",
        "load_ternary.py imports only stdlib and documented dependencies",
        "load_ternary.py carries the MIT copyright and permission notice",
        "card has no unrendered placeholder",
        "card YAML: model_name, license cc-by-4.0, base_model, library_name nemo",
        "card model-index: one entry per test set, value == round(test.json M1, 2)",
        "card results table reproduces all 6 TEST.md rows in TEST.md column order",
        "card file sizes == test.json (180.8 MB, 2,472 MB)",
        "staged file set is exactly the expected set",
        "state_dict: same keys in the same order",
        "state_dict: every tensor torch.equal",
        "state_dict: every floating tensor FP32",
        "staged model in eval mode",
    }
    names |= {f"staged {rel} SHA-256 == source" for rel in STAGED_FILES}
    for i in CHECK_IDS:
        names |= {f"clip {i}: staged transcript == repository transcript",
                  f"clip {i}: staged transcript == test-scoring hypothesis"}
    return names


def staged_listing(folder: Path) -> dict:
    return {str(p.relative_to(folder)): {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(folder.rglob("*")) if p.is_file()}


def write_json(path: Path, data) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=1) + "\n")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo_id", help="USER/NAME on huggingface.co")
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", args.repo_id):
        parser.error("repo_id must be USER/NAME")
    if not os.environ.get("INVOCATION_ID"):
        sys.exit("run through ./heavy (loads the model twice): "
                 "./heavy hf-stage --mem-max 24G --runtime 30min --wait -- python hf/stage.py USER/NAME")
    paths.require_mount()
    STAGING.mkdir(parents=True, exist_ok=True)
    name = args.repo_id.replace("/", "__")
    folder = STAGING / name
    tmp = STAGING / f".{name}.tmp-{os.getpid()}"
    report_ok, report_failed = STAGING / f"{name}.stage.json", STAGING / f"{name}.stage.FAILED.json"
    # Remove stale outputs first, so a failure below leaves no final folder and no success report.
    for stale in (report_ok, report_failed):
        stale.unlink(missing_ok=True)
    for stale in (folder, *STAGING.glob(f".{name}.tmp-*")):
        if stale.exists():
            shutil.rmtree(stale)

    export_dir = args.export.resolve()
    checks: dict[str, bool] = {}
    info: dict = {"repo_id": args.repo_id, "export": str(export_dir)}
    try:
        manifest = provenance_checks(export_dir, checks, info)
        file_checks(checks)
        card = render_card(args.repo_id, manifest)
        card_checks(card, args.repo_id, checks, info)
        if not all(checks.values()):
            raise StageFailed("provenance, file or card checks failed; not staging")

        (tmp / export.TOKENIZER_DIR).mkdir(parents=True)
        sources = {rel: export_dir / rel for rel in EXPORT_RELS} | {n: HERE / n for n in HF_FILES}
        for rel, src in sources.items():
            shutil.copy2(src, tmp / rel)
        (tmp / "README.md").write_text(card)
        listing = staged_listing(tmp)
        checks["staged file set is exactly the expected set"] = tuple(sorted(listing)) == STAGED_FILES
        for rel in STAGED_FILES:
            want = (hashlib.sha256(card.encode()).hexdigest() if rel == "README.md"
                    else info["artifact_sha256"].get(rel) or sha256(sources[rel]))
            checks[f"staged {rel} SHA-256 == source"] = listing.get(rel, {}).get("sha256") == want

        standalone_checks(tmp, export_dir, checks, info)

        missing = sorted(required_checks() - set(checks))
        unexpected = sorted(set(checks) - required_checks())
        if missing or unexpected:
            raise StageFailed(f"check set mismatch: missing {missing}, unexpected {unexpected}")
        failed = sorted(k for k, ok in checks.items() if ok is not True)
        if failed:
            raise StageFailed(f"{len(failed)} check(s) failed: {failed}")

        validated = listing
        os.rename(tmp, folder)  # atomic promotion within STAGING
        listing = staged_listing(folder)
        if listing != validated:
            raise StageFailed("staged files changed between validation and promotion")
        report = {"status": "PASSED", "folder": str(folder), **info, "files": listing,
                  "total_bytes": sum(v["bytes"] for v in listing.values()),
                  "checks_passed": f"{len(checks)}/{len(checks)}", "checks": checks}
        write_json(report_ok, report)
    except BaseException as error:  # noqa: BLE001 -- any failure, including SystemExit, must leave no final folder
        for leftover in (tmp, folder):
            if leftover.exists():
                shutil.rmtree(leftover)
        report = {"status": "FAILED", "error": "".join(traceback.format_exception_only(error)).strip(),
                  **info, "checks": checks,
                  "failed_checks": sorted(k for k, ok in checks.items() if ok is not True)}
        write_json(report_failed, report)
        print(json.dumps(report, indent=1))
        print(f"\nSTAGING FAILED; no upload folder was left. Report: {report_failed}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(report, indent=1))
    print(f"\nAll {len(checks)} checks passed (report: {report_ok}). Upload, from the repository root, with:\n"
          f"  PYTHONPATH=finetune/stt/deps custom/autoresearch/.training-runtime/bin/python3 "
          f"-m huggingface_hub.cli.hf upload {args.repo_id} {folder} . --repo-type model")


if __name__ == "__main__":
    main()
