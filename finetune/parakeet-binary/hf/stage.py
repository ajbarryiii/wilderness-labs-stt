"""Assemble the Hugging Face upload folder for Kea-B (the MB1 binary export), then check it standalone.

    ./heavy kea-b-stage --mem-max 24G --runtime 30min --wait -- python hf/stage.py rajb3/kea-b

Run it from finetune/parakeet-binary. It loads the model twice and decodes on CUDA, so it refuses
to run outside a systemd unit such as the one ./heavy creates. The contract is plans/release.md.

Every check is recorded by name, and success is the conjunction of all of them. The set of names
is fixed in advance (required_checks), so none can be skipped silently:

A. Provenance (PROVENANCE_CHECKS, CPU reads). The export is MB1's final export, pinned by run,
   step, size and the SHA-256 of its weights, manifest and reconstruction record. It is bound to
   the one test scoring: the precheck, the attempt record, the scoring summary and the
   librispeech_clean hypotheses. results/test.json and results/TEST.md are bound to that scoring,
   and the manifest to the run summary. The config and tokenizer are bound to the pinned base
   .nemo.
B. Files (FILE_CHECKS): the pinned CC-BY-4.0 LICENSE (Kea-T's), load_binary.py's import
   allow-list and MIT notice, and NOTICE's attribution of the base model and training sources.
C. Model card (CARD_CHECKS): valid YAML, no placeholder, datasets = MB1's stream sources,
   model-index = results/test.json, the one results table = TEST.md rows, every stated number
   computed from its source.
D. Staging (staging_checks): files are copied to a temporary sibling folder; the file set must be
   exactly STAGED_FILES and every file's SHA-256 must equal its source.
E. Standalone (STANDALONE_CHECKS, GPU under paths.gpu_lock). The staged load_binary.py (imported
   by file path, no repository module) and binary_export.load_export give identical state_dicts.
   On four fixed LibriSpeech test-clean clips, the staged transcripts equal those of the
   repository model decoded as the test scoring decoded, and the test-scoring hypotheses.

Only if every check passes is the temporary folder renamed to STAGING/USER__NAME and the report
written to STAGING/USER__NAME.stage.json. Stale outputs are removed first; on any failure no final
folder exists and the report goes to STAGING/USER__NAME.stage.FAILED.json. It writes only under
STAGING and never uploads (upload procedure: plans/release.md section 7).
"""
from __future__ import annotations

import argparse
import ast
import datetime
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

import binary_main_paths
import binary_paths
import binary_test_paths as tp
import paths

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
STAGING = binary_paths.ARTIFACTS / "hf-staging"
RUN_NAME = "main-binB-m1-lr2e-4"
DEFAULT_EXPORT = binary_paths.RUNS / RUN_NAME / "export"
RUN_SUMMARY = binary_paths.RUNS / RUN_NAME / "summary.json"
PRECHECK = binary_paths.EVAL / "MB1-precheck.json"
TEST_SCORING = binary_paths.EVAL / "MB1-test"
ATTEMPTS = binary_paths.EVAL / "MB1-test.attempts.jsonl"
SOURCES_JSON = binary_main_paths.record_dir(binary_main_paths.SEED) / "stream" / "sources.json"
TEST_JSON = PKG / "results" / "test.json"
TEST_MD = PKG / "results" / "TEST.md"

# The card states this export's numbers; only it may be staged.
EXPECTED = {"run_name": RUN_NAME, "select": "final", "selected_step": 250_000, "lr": 0.0002}
FILE_BYTES = 105_299_804
QUANTIZED_LAYERS = 264
LAYER_KINDS = ("linear", "pointwise_conv1d")
IDENTITY_KEYS = ("run_name", "arm", "recipe", "lr", "select", "selected_step", "config_sha256")
# Manifest fields binary_test.source_record copied into the test scoring's source record.
SCORING_MANIFEST_KEYS = ("format", "file", "sha256", "files", "extra", "parameter_accounting", "bytes")
WEIGHTS_SHA256 = "9873d2f1a53127c673d0201db98cd3ba859ba432f334fef55e9e24fd057b9651"
# Pinned 2026-10-21 after the provenance cross-checks passed on these exact files in a CPU-only
# read (plans/release.md section 6); equal to binary_test_paths.PINNED_FILES, recorded before test scoring.
MANIFEST_SHA256 = "f3737a538da81939aabe04acd5334afd0e3cd4160ec113bd889e771b67991dbd"
RECONSTRUCTION_SHA256 = "efa75346d366fbb92a49b1de72d569894c6338e84199271cbfc97d2f2696d234"
LICENSE_SHA256 = "9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411"  # legalcode.txt, as Kea-T
CARD_LICENSE = "cc-by-4.0"
CARD_HEADER = ("| Model | LS clean | LS other | AMI | Earnings-22 | GigaSpeech | SPGISpeech | VoxPopuli "
               "| Mean of 7 | Common Voice |")
TABLE_ARMS = ("B0", "M1", "MB1")
CARD_ROW_LABELS = {"B0": "Original, FP32 (our scoring)", "M1": "[Kea-T](https://huggingface.co/rajb3/kea-t)",
                   "MB1": "**Kea-B (this model)**"}
PLACEHOLDERS = ("{repo_id}", "{repo_name}")
CHECK_IDS = ("1089-134686-0000", "1089-134686-0001", "1089-134686-0002", "1089-134686-0003")
# Modules the staged loader may import: stdlib plus the documented dependencies.
LOADER_IMPORTS = {"__future__", "contextlib", "hashlib", "json", "math", "sys", "pathlib", "numpy",
                  "torch", "safetensors", "soundfile", "nemo", "omegaconf"}
MIT_LINES = ("# MIT License", "# Copyright (c) 2026 ajbarryiii", "# Permission is hereby granted, free of charge",
             "# The above copyright notice and this permission notice shall be included in all")
# binary_export's layout constants, restated so that importing this module does not import torch:
# main() imports binary_train (which sets CUDA environment variables before torch) first, as
# binary_test.py did. tests/test_hf_release.py checks they equal binary_export's.
FORMAT = "parakeet-binary-v1"
EXPORT_FILE = "export.safetensors"
MANIFEST = "manifest.json"
RECONSTRUCTION = "reconstruction.json"
TOKENIZER_DIR = "tokenizer"
TOKENIZER_FILES = {"model_path": "tokenizer.model", "vocab_path": "vocab.txt", "spe_tokenizer_vocab": "tokenizer.vocab"}
TOKENIZER_RELS = tuple(f"{TOKENIZER_DIR}/{f}" for f in TOKENIZER_FILES.values())
EXPORT_RELS = (EXPORT_FILE, *TOKENIZER_RELS, MANIFEST, RECONSTRUCTION)
HF_FILES = ("load_binary.py", "LICENSE", "NOTICE")
STAGED_FILES = tuple(sorted((*EXPORT_RELS, *HF_FILES, "README.md")))

# ----------------------------------------------------------------------------- check names
PROVENANCE_CHECKS = frozenset({
    "manifest format is parakeet-binary-v1 and file is export.safetensors",
    "manifest identity is the MB1 final export (run, select, step, lr)",
    "export.safetensors is 105,299,804 bytes (file and manifest)",
    "export SHA-256 == manifest sha256 == manifest files[export.safetensors]",
    "export SHA-256 == pinned MB1 weights == binary_test_paths.WEIGHTS_SHA256",
    "export directory holds exactly the pinned entries",
    "precheck passed (pass true, every check ok) and recorded these weights",
    "export SHA-256 == results/test.json mb1_weights_sha256",
    "test scoring: attempt 1 complete, all 8 sets, source is this export",
    "test scoring librispeech_clean.json SHA-256 == its set_done record and summary per_set_sha256",
    "manifest format/file/sha256/files/extra/accounting/bytes == those recorded by the test scoring",
    "results/test.json MB1 row == test scoring summary (8 sets and mean)",
    "results/TEST.md B0, M1 and MB1 rows == results/test.json rounded to 2 decimals",
    "manifest identity and config_sha256 == run summary.json",
    "run summary export_dir, export_bytes and code histogram == manifest",
    "code histogram is binary (no zero codes; -1 and +1 fractions sum to 1)",
    "manifest quantized layers: 264, kinds linear or pointwise_conv1d, no bias",
    "manifest lists exactly export.safetensors and the three tokenizer files",
    "tokenizer files match the manifest SHA-256",
    "manifest base_model is the pinned base model",
    "base .nemo SHA-256 == lock.json == test scoring base_model",
    "manifest config == base .nemo model_config.yaml (resolved)",
    "tokenizer files == base .nemo tokenizer members",
    "reconstruction.json == run summary reconstruction",
    "reconstruction exact: codes, scales, greedy hyps, all 264 layers",
    "manifest.json SHA-256 == pinned == binary_test_paths",
    "reconstruction.json SHA-256 == pinned == binary_test_paths",
})
NOTICE_CHECK = "NOTICE names the base model revision and every training source with its revision"
FILE_CHECKS = frozenset({
    "LICENSE is the pinned CC-BY-4.0 legal code",
    "load_binary.py imports only stdlib and documented dependencies",
    "load_binary.py carries the MIT copyright and permission notice",
    NOTICE_CHECK,
})
CARD_CHECKS = frozenset({
    "card has no unrendered placeholder",
    "card YAML: model_name, license cc-by-4.0, base_model, library_name nemo",
    "card YAML datasets == MB1's training sources",
    "card model-index: one entry per test set, value == round(test.json MB1, 2)",
    "card results table: exactly the TEST.md B0, M1 and MB1 rows, in TEST.md column order",
    "card numbers == their sources",
})
DECODER_CHECK = "decoder CUDA-graph mode no_while_loops for both models (as the test scoring)"
STANDALONE_CHECKS = frozenset({
    "state_dict: same keys in the same order",
    "state_dict: every tensor torch.equal",
    "state_dict: every floating tensor FP32",
    "staged model in eval mode",
    DECODER_CHECK,
    *(f"clip {i}: staged transcript == repository transcript" for i in CHECK_IDS),
    *(f"clip {i}: staged transcript == test-scoring hypothesis" for i in CHECK_IDS),
})


def staging_checks() -> frozenset[str]:
    return frozenset({"staged file set is exactly the expected set",
                      *(f"staged {rel} SHA-256 == source" for rel in STAGED_FILES)})


def required_checks() -> set[str]:
    """Every check name a successful run must record (a missing or extra name fails the run)."""
    return set(PROVENANCE_CHECKS | FILE_CHECKS | CARD_CHECKS | staging_checks() | STANDALONE_CHECKS)


class StageFailed(Exception):
    pass


def sha256(path: Path) -> str:
    return binary_paths.sha256_file(Path(path))


def read_json(path: Path):
    return json.loads(Path(path).read_text())


# ----------------------------------------------------------------------------- A. provenance

def base_nemo_members(manifest: dict) -> tuple[dict, dict[str, bytes]]:
    """(resolved model_config.yaml, {tokenizer rel path: bytes}) from the pinned base .nemo."""
    from omegaconf import OmegaConf

    refs = {Path(str(manifest["config"]["tokenizer"][key]).removeprefix("nemo:")).name: rel
            for key, rel in zip(TOKENIZER_FILES, TOKENIZER_RELS)}
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


def gather(export_dir: Path) -> dict:
    """Every input of provenance_from, read from disk (CPU only; reads the 2.5 GB base .nemo)."""
    export_dir = Path(export_dir).resolve()
    manifest = read_json(export_dir / MANIFEST)
    nemo_config, nemo_tokenizer = base_nemo_members(manifest)
    return {
        "export_dir": export_dir, "manifest": manifest, "recon": read_json(export_dir / RECONSTRUCTION),
        "summary": read_json(RUN_SUMMARY), "precheck": read_json(PRECHECK), "test": read_json(TEST_JSON),
        "test_md": TEST_MD.read_text(), "scoring": read_json(TEST_SCORING / "summary.json"),
        "attempts": [json.loads(line) for line in ATTEMPTS.read_text().splitlines() if line.strip()],
        "entries": {p.relative_to(export_dir).as_posix() for p in export_dir.rglob("*")},
        "digests": {rel: sha256(export_dir / rel) for rel in EXPORT_RELS},
        "export_size": (export_dir / EXPORT_FILE).stat().st_size,
        "ls_clean_sha256": sha256(TEST_SCORING / "librispeech_clean.json"),
        "lock": read_json(paths.MODEL_DIR / "lock.json"), "nemo_sha256": sha256(paths.MODEL_FILE),
        "nemo_config": nemo_config, "nemo_tokenizer": nemo_tokenizer,
        "tokenizer_bytes": {rel: (export_dir / rel).read_bytes() for rel in TOKENIZER_RELS},
    }


def first_table(text: str) -> tuple[list[str], dict[str, list[str]]]:
    """(header cells, {row label: cells}) of the first Markdown table in text."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| "))
    table = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        table.append([c.strip() for c in line.strip().strip("|").split("|")])
    rows = {cells[0]: cells[1:] for cells in table[2:]}
    return table[0], rows


def rounded_row(row: dict, columns: list[str]) -> list[str]:
    return [f"{row[c]:.2f}" if c in row else "n/a" for c in columns]


def provenance_from(d: dict, checks: dict, info: dict) -> dict:
    """Record the PROVENANCE_CHECKS on gathered inputs; return the manifest."""
    manifest, recon, summary, test, scoring = d["manifest"], d["recon"], d["summary"], d["test"], d["scoring"]
    export_dir = Path(d["export_dir"]).resolve()
    digest = d["digests"][EXPORT_FILE]
    extra = manifest.get("extra", {})
    source = scoring.get("source", {})
    info["artifact_sha256"] = dict(d["digests"])
    info["export_sha256"] = digest
    info["test_scoring_librispeech_clean_sha256"] = d["ls_clean_sha256"]

    checks["manifest format is parakeet-binary-v1 and file is export.safetensors"] = (
        manifest.get("format") == FORMAT == tp.FORMAT and manifest.get("file") == EXPORT_FILE)
    checks["manifest identity is the MB1 final export (run, select, step, lr)"] = (
        {k: extra.get(k) for k in EXPECTED} == EXPECTED)
    checks["export.safetensors is 105,299,804 bytes (file and manifest)"] = (
        d["export_size"] == FILE_BYTES == manifest.get("bytes", {}).get("file_bytes"))
    checks["export SHA-256 == manifest sha256 == manifest files[export.safetensors]"] = (
        digest == manifest.get("sha256") == manifest.get("files", {}).get(EXPORT_FILE))
    checks["export SHA-256 == pinned MB1 weights == binary_test_paths.WEIGHTS_SHA256"] = (
        digest == WEIGHTS_SHA256 == tp.WEIGHTS_SHA256)
    checks["export directory holds exactly the pinned entries"] = set(d["entries"]) == set(tp.EXPORT_ENTRIES)
    precheck = d["precheck"]
    checks["precheck passed (pass true, every check ok) and recorded these weights"] = (
        precheck.get("pass") is True and bool(precheck.get("checks"))
        and all(c.get("ok") is True for c in precheck["checks"]) and precheck.get("weights_sha256") == digest)
    checks["export SHA-256 == results/test.json mb1_weights_sha256"] = test.get("mb1_weights_sha256") == digest

    states = [(e.get("attempt"), e.get("state"), e.get("set")) for e in d["attempts"]]
    expected_states = ([(1, "started", None)] + [(1, "set_done", n) for n in paths.TEST_SETS]
                       + [(1, "complete", None)])
    sets = scoring.get("sets", {})
    checks["test scoring: attempt 1 complete, all 8 sets, source is this export"] = (
        sorted(states, key=str) == sorted(expected_states, key=str) and scoring.get("attempt") == 1
        and set(sets) == set(paths.TEST_SETS) and all(s.get("limit") is None for s in sets.values())
        and source.get("kind") == "export" and source.get("sha256") == digest
        and Path(source.get("path", "/nonexistent")).resolve() == export_dir)
    set_done = {e.get("set"): e.get("sha256") for e in d["attempts"] if e.get("state") == "set_done"}
    checks["test scoring librispeech_clean.json SHA-256 == its set_done record and summary per_set_sha256"] = (
        d["ls_clean_sha256"] == set_done.get("librispeech_clean")
        == scoring.get("per_set_sha256", {}).get("librispeech_clean"))
    recorded = source.get("export_manifest", {})
    checks["manifest format/file/sha256/files/extra/accounting/bytes == those recorded by the test scoring"] = (
        set(recorded) == set(SCORING_MANIFEST_KEYS) and all(manifest.get(k) == v for k, v in recorded.items()))
    mb1 = test.get("rows", {}).get("MB1", {})
    checks["results/test.json MB1 row == test scoring summary (8 sets and mean)"] = (
        set(mb1) == {*paths.TEST_SETS, "mean"} and all(mb1[n] == sets[n].get("wer") for n in paths.TEST_SETS)
        and mb1["mean"] == scoring.get("mean_wer"))
    header, rows = first_table(d["test_md"])
    columns = test.get("columns", [])
    checks["results/TEST.md B0, M1 and MB1 rows == results/test.json rounded to 2 decimals"] = (
        header[1:] == columns == [*paths.MEAN_SETS, "mean", "common_voice"]
        and all(rows.get(arm) == rounded_row(test["rows"][arm], columns) for arm in TABLE_ARMS))

    checks["manifest identity and config_sha256 == run summary.json"] = all(
        k in extra and extra.get(k) == summary.get(k) for k in IDENTITY_KEYS)
    checks["run summary export_dir, export_bytes and code histogram == manifest"] = (
        Path(summary.get("export_dir", "/nonexistent")).resolve() == export_dir
        and summary.get("export_bytes") == manifest.get("bytes")
        # the run summary records the manifest's histogram without its per_module entries
        and summary.get("code_histogram") == {k: v for k, v in manifest.get("code_histogram", {}).items()
                                              if k != "per_module"})
    hist = manifest.get("code_histogram", {})
    layers = manifest.get("quantized_layers", {})
    parts = [hist, *hist.get("per_family", {}).values(), *hist.get("per_module", {}).values()]
    checks["code histogram is binary (no zero codes; -1 and +1 fractions sum to 1)"] = (
        bool(hist.get("per_family")) and set(hist.get("per_module", {})) == set(layers)
        and all(p.get("zero") == 0 and abs(p.get("minus_one", 0) + p.get("plus_one", 0) - 1) < 1e-9 for p in parts))
    checks["manifest quantized layers: 264, kinds linear or pointwise_conv1d, no bias"] = (
        len(layers) == QUANTIZED_LAYERS and all(v.get("kind") in LAYER_KINDS and v.get("bias") is False
                                                for v in layers.values()))
    checks["manifest lists exactly export.safetensors and the three tokenizer files"] = (
        set(manifest.get("files", {})) == {EXPORT_FILE, *TOKENIZER_RELS})
    checks["tokenizer files match the manifest SHA-256"] = all(
        d["digests"].get(rel) == manifest.get("files", {}).get(rel) for rel in TOKENIZER_RELS)
    checks["manifest base_model is the pinned base model"] = manifest.get("base_model") == {
        "id": paths.MODEL_ID, "revision": paths.MODEL_REVISION, "license": "CC-BY-4.0"}
    checks["base .nemo SHA-256 == lock.json == test scoring base_model"] = (
        d["lock"].get("revision") == paths.MODEL_REVISION == tp.BASE_MODEL["revision"]
        and d["nemo_sha256"] == d["lock"].get("files", {}).get(paths.MODEL_FILE.name)
        == tp.BASE_MODEL["nemo_sha256"] and source.get("base_model") == tp.BASE_MODEL)
    checks["manifest config == base .nemo model_config.yaml (resolved)"] = (
        d["nemo_config"] is not None and d["nemo_config"] == manifest.get("config"))
    checks["tokenizer files == base .nemo tokenizer members"] = (
        set(d["nemo_tokenizer"]) == set(TOKENIZER_RELS)
        and all(d["tokenizer_bytes"].get(rel) == data for rel, data in d["nemo_tokenizer"].items()))
    checks["reconstruction.json == run summary reconstruction"] = recon == summary.get("reconstruction")
    checks["reconstruction exact: codes, scales, greedy hyps, all 264 layers"] = (
        recon.get("codes_exact") is True and recon.get("scales_exact") is True
        and recon.get("greedy_hyps_equal") is True and recon.get("checked_layers") == len(layers) == QUANTIZED_LAYERS)
    checks["manifest.json SHA-256 == pinned == binary_test_paths"] = (
        d["digests"].get(MANIFEST) == MANIFEST_SHA256 == tp.PINNED_FILES["export/manifest.json"])
    checks["reconstruction.json SHA-256 == pinned == binary_test_paths"] = (
        d["digests"].get(RECONSTRUCTION) == RECONSTRUCTION_SHA256 == tp.PINNED_FILES["export/reconstruction.json"])
    return manifest


# ----------------------------------------------------------------------------- B. files

def loader_imports(text: str) -> set[str]:
    """Top-level module names imported anywhere in the source (relative imports keep their dots)."""
    names = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add("." * node.level + (node.module or "") if node.level else node.module.split(".")[0])
    return names


def file_checks(checks: dict, sources: dict, hf_dir: Path = HERE) -> None:
    text = (hf_dir / "LICENSE").read_text()
    checks["LICENSE is the pinned CC-BY-4.0 legal code"] = (
        sha256(hf_dir / "LICENSE") == LICENSE_SHA256 and text.startswith("Attribution 4.0 International")
        and "Section 1 -- Definitions." in text and "Section 8 -- Interpretation." in text)
    loader = (hf_dir / "load_binary.py").read_text()
    checks["load_binary.py imports only stdlib and documented dependencies"] = loader_imports(loader) <= LOADER_IMPORTS
    checks["load_binary.py carries the MIT copyright and permission notice"] = all(s in loader for s in MIT_LINES)
    notice = (hf_dir / "NOTICE").read_text()
    entries = list(sources.get("sources", {}).values())
    checks[NOTICE_CHECK] = (
        paths.MODEL_ID in notice and paths.MODEL_REVISION in notice and "load_binary.py" in notice
        and len(entries) == 5 and all(e["repo"] in notice and e["revision"] in notice for e in entries))


# ----------------------------------------------------------------------------- C. card

def render_card(template: str, repo_id: str) -> str:
    return template.replace("{repo_id}", repo_id).replace("{repo_name}", repo_id.split("/")[1])


def _utc(text: str) -> datetime.datetime:
    return datetime.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")


def card_facts(test: dict, summary: dict, manifest: dict, sources: dict) -> dict:
    """The strings the card must contain, computed from their sources (plans/release.md section 5),
    and the training datasets. ValueError if the stream record is not MB1's."""
    if sources.get("seed") != summary.get("seed"):
        raise ValueError(f"stream record seed {sources.get('seed')} != run seed {summary.get('seed')}")
    used = [k for k, seconds in summary["seconds_by_source"].items() if seconds > 0]
    missing = [k for k in used if k not in sources.get("sources", {})]
    if missing:
        raise ValueError(f"sources {missing} are not in the stream record")
    rows, mb = test["rows"], test["export_mb"]
    mantissa, exponent = f"{summary['lr']:.0e}".split("e")
    hours = (_utc(summary["finished_utc"]) - _utc(summary["started_utc"])).total_seconds() / 3600
    strings = {
        "export size": f"{mb['MB1']:.1f} MB",
        "original .nemo size": f"{mb['original']:,.0f} MB",
        "Kea-T size": f"{mb['M1']:.1f} MB",
        "original mean": f"{rows['B0']['mean']:.2f}%",
        "Kea-T mean": f"{rows['M1']['mean']:.2f}%",
        "Kea-B mean": f"{rows['MB1']['mean']:.2f}%",
        "original Common Voice": f"{rows['B0']['common_voice']:.2f}%",
        "Kea-B Common Voice": f"{rows['MB1']['common_voice']:.2f}%",
        "file bytes": f"{manifest['bytes']['file_bytes']:,}",
        "steps": f"{summary['steps']:,}",
        "ramp steps": f"{summary['ramp_steps']:,}",
        "peak learning rate": f"{mantissa}e{int(exponent)}",
        "training time": f"{hours:.1f} hours",
        "audio hours": f"{summary['audio_hours_seen']:,.0f} hours",
        "batch audio": f"up to {summary['batch_seconds']:.0f} s of audio",
        "clip lengths": "{:g} to {:g} s".format(*sources["filters"]["duration_seconds"]),
        "NeMo version": f"NeMo {manifest['versions']['nemo']}",
        "torch version": f"torch {manifest['versions']['torch']}",
    }
    return {"strings": strings, "datasets": {sources["sources"][k]["repo"] for k in used}}


def card_table(card: str) -> list[tuple[str, list[str]]] | None:
    """[(label, cells)] of the table under CARD_HEADER, None if the header is absent or not followed by a separator."""
    lines = card.splitlines()
    if lines.count(CARD_HEADER) != 1:
        return None
    i = lines.index(CARD_HEADER)
    if i + 1 >= len(lines) or not lines[i + 1].startswith("| ---"):
        return None
    rows = []
    for line in lines[i + 2:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append((cells[0], cells[1:]))
    return rows


def card_checks(card: str, repo_id: str, test: dict, test_md: str, facts: dict, checks: dict, info: dict) -> None:
    from huggingface_hub import ModelCard

    checks["card has no unrendered placeholder"] = not any(p in card for p in PLACEHOLDERS)
    data = ModelCard(card).data  # raises on invalid YAML front matter
    checks["card YAML: model_name, license cc-by-4.0, base_model, library_name nemo"] = (
        data.model_name == repo_id.split("/")[1] and data.license == CARD_LICENSE
        and data.base_model == paths.MODEL_ID and data.library_name == "nemo")
    datasets = list(data.datasets or [])
    checks["card YAML datasets == MB1's training sources"] = (
        set(datasets) == facts["datasets"] and len(datasets) == len(facts["datasets"]))
    mb1 = test["rows"]["MB1"]
    by_dataset = {v: k for k, v in paths.TEST_SETS.items()}
    seen, index_ok = {}, True
    for result in data.eval_results or []:
        key = by_dataset.get((result.dataset_config, result.dataset_split))
        index_ok &= (key is not None and key not in seen and result.dataset_type == paths.ESB_REPO
                     and result.dataset_revision == paths.ESB_REVISION and result.metric_type == "wer"
                     and result.metric_value == round(mb1[key], 2))
        seen[key] = result.metric_value
    checks["card model-index: one entry per test set, value == round(test.json MB1, 2)"] = (
        index_ok and set(seen) == set(paths.TEST_SETS))
    header, md_rows = first_table(test_md)
    want = [(CARD_ROW_LABELS[arm], md_rows.get(arm)) for arm in TABLE_ARMS]
    got = card_table(card)
    checks["card results table: exactly the TEST.md B0, M1 and MB1 rows, in TEST.md column order"] = (
        header[1:] == [*paths.MEAN_SETS, "mean", "common_voice"] and got == want)
    missing = sorted(label for label, s in facts["strings"].items() if s not in card)
    checks["card numbers == their sources"] = not missing
    info["card"] = {"model_name": data.model_name, "license": data.license, "base_model": data.base_model,
                    "datasets": datasets, "model_index_results": len(seen), "table_rows": got,
                    "numbers": facts["strings"], "missing_numbers": missing,
                    "sha256": hashlib.sha256(card.encode()).hexdigest()}


# ----------------------------------------------------------------------------- A-C together

def make_prepare(repo_id: str):
    def prepare(export_dir: Path, checks: dict, info: dict) -> tuple[str, dict[str, Path]]:
        docs = gather(export_dir)
        manifest = provenance_from(docs, checks, info)
        sources = read_json(SOURCES_JSON)
        file_checks(checks, sources)
        facts = card_facts(docs["test"], docs["summary"], manifest, sources)
        card = render_card((HERE / "MODEL_CARD.md").read_text(), repo_id)
        card_checks(card, repo_id, docs["test"], docs["test_md"], facts, checks, info)
        export_dir = Path(export_dir).resolve()
        return card, {**{rel: export_dir / rel for rel in EXPORT_RELS}, **{n: HERE / n for n in HF_FILES}}
    return prepare


# ----------------------------------------------------------------------------- E. standalone

def standalone_checks(folder: Path, export_dir: Path, checks: dict, info: dict) -> None:
    """Load the staged folder with its own load_binary.py and the source export with the repository
    loader; compare state_dicts on the CPU, then transcripts on CUDA under the shared GPU lock."""
    import torch

    import binary_export
    import binary_train
    import evaluate

    spec = importlib.util.spec_from_file_location("staged_load_binary", folder / "load_binary.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with torch.no_grad():
        staged = module.load(folder, device="cpu")
        reference = binary_export.load_export(export_dir, device="cpu")
        a, b = staged.state_dict(), reference.state_dict()
        differing = [k for k in a if k not in b or a[k].dtype != b[k].dtype or a[k].shape != b[k].shape
                     or not torch.equal(a[k], b[k])]
        checks["state_dict: same keys in the same order"] = list(a) == list(b)
        checks["state_dict: every tensor torch.equal"] = not differing and len(a) == len(b) > 0
        checks["state_dict: every floating tensor FP32"] = all(
            v.dtype == torch.float32 for v in a.values() if v.is_floating_point())
        checks["staged model in eval mode"] = not staged.training
        info.update(loader_module_file=module.__file__, state_dict_keys=len(a), state_dict_differing=differing[:10])
        del a, b

    records = {r["id"]: r for r in evaluate.read_manifest(evaluate.resolve_manifest("librispeech_clean")[1])}
    records = [records[i] for i in CHECK_IDS]
    data = (TEST_SCORING / "librispeech_clean.json").read_bytes()
    if hashlib.sha256(data).hexdigest() != info["test_scoring_librispeech_clean_sha256"]:
        raise StageFailed("eval/MB1-test/librispeech_clean.json changed since the provenance checks")
    scored = {r["id"]: r["hyp"] for r in json.loads(data)["records"]}
    device = torch.device("cuda")
    with paths.gpu_lock("kea-b-stage"):
        staged.to(device)
        reference.to(device)
        binary_train.pin_decoder_graphs_mode(reference, "repository model")  # as binary_test.load_model
        hyps_staged = module.transcribe(staged, [r["audio_filepath"] for r in records])
        hyps_reference = evaluate.transcribe_records(reference, records, batch_size=len(records),
                                                     device=device, workers=0)
        modes = {"staged": binary_train.decoder_graphs_mode(staged, "staged model"),
                 "repository": binary_train.decoder_graphs_mode(reference, "repository model")}
        del staged, reference
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    checks[DECODER_CHECK] = set(modes.values()) == {binary_paths.DECODER_GRAPHS_MODE}
    for r, hs, hr in zip(records, hyps_staged, hyps_reference, strict=True):
        checks[f"clip {r['id']}: staged transcript == repository transcript"] = hs == hr
        checks[f"clip {r['id']}: staged transcript == test-scoring hypothesis"] = hs == scored.get(r["id"])
    info["device"] = device.type
    info["decoder_graphs_mode"] = modes
    info["clips"] = [{"id": r["id"], "audio_filepath": r["audio_filepath"], "reference": r["text"], "staged": hs,
                      "repository": hr, "test_scoring": scored.get(r["id"])}
                     for r, hs, hr in zip(records, hyps_staged, hyps_reference)]


# ----------------------------------------------------------------------------- runtime preflight

def load_runtime():
    """Import binary_train first (it sets CUDA environment variables before torch is imported, as
    binary_test.py relied on), then torch; return torch."""
    import binary_train  # noqa: F401
    import torch

    return torch


def runtime_preflight(info: dict) -> None:
    """The process's runtime: StageFailed unless CUDA is available (the clip checks compare with
    hypotheses decoded on CUDA). Runs inside stage(), after stale outputs are removed."""
    torch = load_runtime()
    if not torch.cuda.is_available():
        raise StageFailed("CUDA is required: the clip checks compare with hypotheses decoded on CUDA")
    info["cuda_device"] = torch.cuda.get_device_name(0)


# ----------------------------------------------------------------------------- driver

def staged_listing(folder: Path) -> dict:
    return {p.relative_to(folder).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(folder.rglob("*")) if p.is_file()}


def write_json(path: Path, data) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def stage(repo_id: str, export_dir: Path, staging: Path, prepare, standalone,
          preflight=None) -> tuple[bool, dict, Path]:
    """Run every check and stage repo_id's folder under staging; (passed, report, report path).

    preflight(info), if given, runs first after stale outputs are removed (the runtime imports and
    the CUDA check), so its failure is recorded like any other. prepare(export_dir, checks, info) records the provenance, file and card checks and returns
    (rendered card, {staged rel path: source file}); standalone(folder, export_dir, checks, info)
    records the standalone checks on the temporary folder. Any failure, including an exception
    or SystemExit, leaves no final and no temporary folder, and only the FAILED report.
    """
    name = repo_id.replace("/", "__")
    folder = staging / name
    tmp = staging / f".{name}.tmp-{os.getpid()}"
    report_ok, report_failed = staging / f"{name}.stage.json", staging / f"{name}.stage.FAILED.json"
    staging.mkdir(parents=True, exist_ok=True)
    # Stale outputs go first, so a failure below leaves no final folder and no success report.
    for stale in (report_ok, report_failed):
        stale.unlink(missing_ok=True)
    for stale in (folder, *staging.glob(f".{name}.tmp-*")):
        if stale.exists():
            shutil.rmtree(stale)

    checks: dict[str, bool] = {}
    info: dict = {"repo_id": repo_id, "export": str(export_dir)}
    try:
        if preflight is not None:
            preflight(info)
        card, sources = prepare(export_dir, checks, info)
        if any(ok is not True for ok in checks.values()):
            raise StageFailed("provenance, file or card checks failed; not staging")
        tmp.mkdir()
        for rel, src in sources.items():
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, tmp / rel)
        (tmp / "README.md").write_text(card)
        listing = staged_listing(tmp)
        checks["staged file set is exactly the expected set"] = tuple(sorted(listing)) == STAGED_FILES
        for rel in STAGED_FILES:
            want = (hashlib.sha256(card.encode()).hexdigest() if rel == "README.md"
                    else info.get("artifact_sha256", {}).get(rel) or (sha256(sources[rel]) if rel in sources else None))
            checks[f"staged {rel} SHA-256 == source"] = want is not None and listing.get(rel, {}).get("sha256") == want
        if any(ok is not True for ok in checks.values()):
            raise StageFailed("staged files differ from the expected set or their sources; not loading")

        standalone(tmp, export_dir, checks, info)

        required = required_checks()
        missing, unexpected = sorted(required - set(checks)), sorted(set(checks) - required)
        if missing or unexpected:
            raise StageFailed(f"check set mismatch: missing {missing}, unexpected {unexpected}")
        failed = sorted(k for k, ok in checks.items() if ok is not True)
        if failed:
            raise StageFailed(f"{len(failed)} check(s) failed: {failed}")

        validated = listing
        os.rename(tmp, folder)  # atomic promotion within staging
        listing = staged_listing(folder)
        if listing != validated:
            raise StageFailed("staged files changed between validation and promotion")
        report = {"status": "PASSED", "folder": str(folder), **info, "files": listing,
                  "total_bytes": sum(v["bytes"] for v in listing.values()),
                  "checks_passed": f"{len(checks)}/{len(checks)}", "checks": checks}
        write_json(report_ok, report)
        return True, report, report_ok
    except BaseException as error:  # noqa: BLE001 -- any failure, including SystemExit, must leave no final folder
        for leftover in (tmp, folder):
            if leftover.exists():
                shutil.rmtree(leftover)
        report_ok.unlink(missing_ok=True)
        report = {"status": "FAILED", "error": "".join(traceback.format_exception_only(error)).strip(),
                  **info, "checks": checks, "failed_checks": sorted(k for k, ok in checks.items() if ok is not True)}
        write_json(report_failed, report)
        return False, report, report_failed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo_id", help="USER/NAME on huggingface.co (rajb3/kea-b)")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", args.repo_id):
        parser.error("repo_id must be USER/NAME")
    if not os.environ.get("INVOCATION_ID"):
        sys.exit("run through ./heavy (loads the model twice, decodes on CUDA): ./heavy kea-b-stage --mem-max 24G "
                 "--runtime 30min --wait -- python hf/stage.py USER/NAME")
    binary_paths.require_mount()
    # Everything after the mount check, including the runtime imports and the CUDA check
    # (runtime_preflight), runs inside stage(): stale outputs are removed first and any failure
    # leaves only a FAILED report.
    passed, report, path = stage(args.repo_id, Path(DEFAULT_EXPORT).resolve(), STAGING, make_prepare(args.repo_id),
                                 standalone_checks, preflight=runtime_preflight)
    print(json.dumps(report, indent=1, default=str))
    if not passed:
        print(f"\nSTAGING FAILED; no upload folder was left. Report: {path}", file=sys.stderr)
        sys.exit(1)
    print(f"\nAll {len(report['checks'])} checks passed (report: {path}). Folder: {report['folder']}\n"
          "Next: AJ reads and approves README.md in that folder; then upload privately and verify from a fresh "
          "download as in plans/release.md section 7. Nothing was uploaded.")


if __name__ == "__main__":
    main()
