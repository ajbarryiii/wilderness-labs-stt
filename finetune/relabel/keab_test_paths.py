"""Pinned inputs and fixed locations of Kea-B′'s test-set scoring (keab-prime-test plan v1.2), without torch so the
report can import them.

Every check in keab_test.py and keab_report_test.py compares files on disk against these constants, not against other
recorded hashes. The identity constants were filled in from the files on 2026-11-03, before any Kea-B′ test decoding.
SCORER_HASHES is pinned after code review (keab_test.py --scorer-hashes prints it); until then precheck item 3 fails.
This file cannot hold its own hash: its SHA-256 is recorded in the plan's Status line once frozen (precheck item 3
checks it there).

Roots come from the environment as for the rest of finetune/relabel (relabel_paths; ./python fills them from
private/local.env). A scorer file outside the repository is named "$ROOT/relative/path" and resolved from that
environment variable (scorer_path); repository files are named relative to finetune/.
"""
from __future__ import annotations

import os
from pathlib import Path

import paths  # finetune/parakeet-ternary/paths.py
import relabel_paths

HERE = relabel_paths.HERE
FINETUNE = HERE.parent
PLAN = relabel_paths.ARTIFACTS / "plans" / "keab-prime-test.md"   # private; its Status line holds this file's hash

# ----------------------------------------------------------------------------- Kea-B′ identity (plan "Pinned identity")
KEAB = relabel_paths.ARTIFACTS / "keab-prime"
RUN_NAME = "b100-main"
RUN = KEAB / "runs" / RUN_NAME
VIEW = RUN / "export" / "view"
BUNDLE = RUN / "export" / "bundle"
PINNED_FILES = {  # relative to RUN
    "export/view/export.safetensors": "ec05526ed7aa7c742ee970c09724b1b8ccfaef6eb10f2ff1a11b031e9dfc4995",
    "export/view/manifest.json": "cfe771d540d4ccdd05a629a4cff7034e0fdb6ba52aca63dc796952661613ccd6",
    "export/view/tokenizer/tokenizer.model": "5a3a82c48998709f9bc9f5db2924d99a899d20bf82550ca52272da7c7557b0a0",
    "export/view/tokenizer/tokenizer.vocab": "dbe7ec6bd066c4526080940e1614e6582b9161515e856c2c39a1d49267c88cbc",
    "export/view/tokenizer/vocab.txt": "70b25deed256d79365a3c290d6191c31cc2315d952faf18d5c3a334f46e4866e",
    "export/bundle/bundle.safetensors": "c00230f8d325dd78a020fc6c6299c83fce874c1518135ece5a252f563f535f81",
    "export/bundle/manifest.json": "bf1b668bbbd2500791acaadfb3486ac6462257d3a4df9fb90245d8c1c0b3a803",
    "model.pt": "b5240c1357c91903eaed9a1232ccce4503cdbd8ccc642dd8ae537c194527207a",
    "model.json": "800290f93ed87a74bf2f0abbad356e36777cb63b3a58289d92b1718b65cda254",
    "summary.json": "a4265e65238914cfd1cf3868bc8acfc2f0a17e04cb7eb84917a928da974b1f81",
    "config.json": "e0488a909d2a139e8c50bbc23ff7ff2883e3b3fef01ac3f58dd199fbe863b930",
}
VIEW_SHA256 = PINNED_FILES["export/view/export.safetensors"]
BUNDLE_SHA256 = PINNED_FILES["export/bundle/bundle.safetensors"]
MODEL_SHA256 = PINNED_FILES["model.pt"]
CONFIG_SHA256 = PINNED_FILES["config.json"]
VIEW_FORMAT = "parakeet-binary-v1"
VIEW_FILE = "export.safetensors"
BUNDLE_FORMAT = "keab-8bit-v1"
BUNDLE_TENSOR_BYTES = 91_236_520          # bundle manifest bytes.tensor_bytes: the shipped 91.24 MB
VIEW_TOKENIZER = ("tokenizer/tokenizer.model", "tokenizer/tokenizer.vocab", "tokenizer/vocab.txt")
VIEW_ENTRIES = frozenset({VIEW_FILE, "manifest.json", "tokenizer", *VIEW_TOKENIZER})
EXPECT_SUMMARY = {"run_name": RUN_NAME, "steps": 97_137, "max_steps": 97_137, "lr": 2e-5, "seed": 20261031,
                  "smoke": False, "dev_limit": None}
VIEW_DESCRIPTION = "Kea-B' scoring view (parakeet-binary-v1), rebuilt with dequantized FP32 weights"

# Stage 4's recorded deployed x1 dev scores of this view (scores/b100-main/x1): file hashes, and runtime.json (which
# records the view SHA-256 the x1 runtime loaded).
SCORES_X1 = KEAB / "scores" / RUN_NAME / "x1"
DEV_X1_FILES = {
    "librispeech_dev_clean": "8ab71caed8cea0faaf0a078b6cdcf4980df632b98f909c488867f99700fb0623",
    "librispeech_dev_other": "e962674811f7c11ddd8a4cb5941e5cdc489922f3ca45a0cdbdfc17a4f9babd84",
    "ami_dev": "86edfafce46aeb7745d714a6c6df683b8a361307f9bb80bff4e5e04531bfb98e",
    "voxpopuli_dev": "6616521c6bb4af5eb147eebc67d7f1728e06abe920ebdcef7be9e3b0539ef20b",
    "yodas_dev": "0c5b19c8485a1d870c879264849d04c2f4681825b73e656e07eacc74f6b69ee9",
}
RUNTIME_X1_SHA256 = "f88b66cad0a252544a06ae381acf9a5e936693b205612ea48f2ffdb049e6a352"
STAGE4_RESULTS_JSON = KEAB / "results.json"
STAGE4_RESULTS_MD = relabel_paths.ARTIFACTS / "plans" / "RESULTS-keab-prime.md"
JOB_PREFIX = "relabel-keab-"              # stage 4's jobs (training/keab_train.sh): none may be active

# ----------------------------------------------------------------------------- reused arms (plan "What is scored")
TERNARY_EVAL = relabel_paths.PARAKEET_TERNARY / "eval"
BINARY_EVAL = relabel_paths.PARAKEET_BINARY / "eval"
B0_TEST = TERNARY_EVAL / "b0-pretrained"
M1_TEST = TERNARY_EVAL / "M1-test"
MB1_TEST = BINARY_EVAL / "MB1-test"
MB1_ATTEMPTS = BINARY_EVAL / "MB1-test.attempts.jsonl"
REUSED = {"B0": B0_TEST, "M1": M1_TEST, "MB1": MB1_TEST}
REUSED_FILES = {  # {arm: {file in its directory: SHA-256}}, recorded 2026-11-03
    "B0": {"summary.json": "6a1a3a7ac7e9d61a0fc40f8626fc51695a69db1443d30fe7e8640982fab8a30f",
           "librispeech_clean.json": "e3ee200fea38b102de83fca16bb685497d18aa31efcc3b0687422d974f05f32b",
           "librispeech_other.json": "222dbb77c28702b6cd75c0b8362f5f04e0aea9b6e9e57140945cc5c8489c835a",
           "ami.json": "a4b2d83840376cbf4cb3de4a69128c51cc82a71a4db61f7e40c9a6f0f6e55d87",
           "earnings22.json": "fcaa50bfad0e579156e4807aaebbdc8fc2a4c04adccade974af4da0e0aa03278",
           "gigaspeech.json": "a275336542dd92241669ae10c7dfa0ff22b925a40ed2cdaaa643dec971346300",
           "spgispeech.json": "7ab7e3d49132bf23a6a152372be6549741a6d3c6237b3d8dc8e170df4b3be5df",
           "voxpopuli.json": "ea3c0c03ac2ba394487766856b4e02d615e0bf1b7249de7699319de635cb4653",
           "common_voice.json": "5f4073723ae70ede64a43cf4ec99ec95f799cef883e7b2b99fb70f8f2343953f"},
    "M1": {"summary.json": "4cec73e07dcf73fe754177f947ae019a74f8b37e20c21a73d3c89e793b6b3b43",
           "librispeech_clean.json": "b2499c69c4420ebce524726dfc74f191b78b4cccf401a1ac29b2d90bbac09436",
           "librispeech_other.json": "08497ea766b8f0e6f9eead573969515eb0f240b563a835814fafe55690e28010",
           "ami.json": "79e4ceaacedd30d4c95b6066b27f6ec26060f403d7654c5783948930f359ea5e",
           "earnings22.json": "f7337c5187a637eb4ba0304363749a06de3cab3d672c38efdba972385a8af0b5",
           "gigaspeech.json": "486da1ec73f7b940039f9a673f2259bb20f71f781371974730679d4c9c22487f",
           "spgispeech.json": "8bf5db55ee312208a954802586b44d2c0454fcd72d9e7bae18b6ed2fcadf1ce5",
           "voxpopuli.json": "f2391bc97269fb1880b83243b81c3aa329ed97a4e76a46893c458df81a5b4da1",
           "common_voice.json": "02cb4f4cde2d8eeaf6d214515088170d304f2f558382c78b78b13835fa13e798"},
    "MB1": {"summary.json": "9e5bfb62bac67854a817dc683de9bde9f705bb650de01998b29c24a965cfab11",
            "librispeech_clean.json": "dabf12753f74a28fae871490ed6da1efc6c9b24d9228bfb00668847e6fd98668",
            "librispeech_other.json": "38f3be4a468bae03f74a565caa90f8ade65d2f8055244cce79b96e70a803a618",
            "ami.json": "8c37a3d42b384f622d8c31464d73677efce4733d453dd73b63e667213c446497",
            "earnings22.json": "d8279e56ed6c0185ae46b28b472a9ff54c0bc294fed8b5dab2465376e54e3cb2",
            "gigaspeech.json": "cebbed828c464c2eb1d6d14677746f672a3256bceb2173510c7c403627943643",
            "spgispeech.json": "491147ce82f4d05d48e074b0baf4ba5e6f655ae0d2059a54d3e0f2b2371d4657",
            "voxpopuli.json": "912241ff95f3cd5a0ce0ae28f2749da71e8ee27722d3b323e3d8a65c6ae03f67",
            "common_voice.json": "6a7fb21db5ce1cad54a72609c4265cd99a56a5c500f68b55c6449495b21a97fb"},
}
MB1_ATTEMPTS_SHA256 = "92c87a78cb7e30af01803e58b2475cd47a9b870ddcf51256b8da3dbba5ad4383"
MB1_EXPORT_SHA256 = "9873d2f1a53127c673d0201db98cd3ba859ba432f334fef55e9e24fd057b9651"
M1_EXPORT_SHA256 = "a287e97719c451b785be2cd01ecc861fcaa010ebaa4ff2841783ec78fcd61503"
BASE_MODEL = {"id": "nvidia/parakeet-tdt-0.6b-v2", "revision": "ae9ad07059c7c739ffaf932226a8fe64ae2620b0",
              "nemo_sha256": "d99e39955c9d3d0350d8fb7c75e40c64a2b2eaeb003883d7c941fd2e8747b28c"}
MB1_EXPORT = relabel_paths.PARAKEET_BINARY / "runs" / "main-binB-m1-lr2e-4" / "export"
MB1_DESCRIPTION = "binary export, rebuilt with dequantized FP32 weights"   # binary_test_paths.SOURCE_DESCRIPTION
# MB1's recorded dev check (the K-fp path's validation target, plan "Dev cross-check").
MB1_DEV_CHECK_GATE = BINARY_EVAL / "MB1-dev-check" / "gate.json"
MB1_DEV_CHECK_GATE_SHA256 = "d44027715ca15defe11f7f980ccf54f1810933fdff6ff445085380576ece4046"
MB1_DEV_MEAN = 0.06311961686131834

# ----------------------------------------------------------------------------- training-data hygiene (precheck item 7)
CV27 = relabel_paths.ARTIFACTS / "corpora" / "cv27-en"
CV_ALIAS = "cv27-en"
CV_STORE_CLIPS = CV27 / "store" / "clips.jsonl"
CV_STORE_META = CV27 / "store" / "clips.meta.json"
CV_SELECTION = CV27 / "selection" / "selection.parquet"
CV_SELECTION_META = CV27 / "selection" / "selection.json"
CV_CLIP_POOL = "8566b62d5a2b8950b8b2b11e25e237dd8fbd82a695b22a3a4647d94aa042b24f"     # config extra_sources cv27-en:
CV_SELECTION_SHA256 = "7dc8081d4b299258d1a7bf3f628e2dfd86c38e9540ad2f0a13faade68daa77c8"
CV_SELECTION_META_SHA256 = "05cda97df8eebf205244747441c078d701bfde7c72080911069ed99372ceb72c"
CV_STORE_CLIPS_COUNT = 937_683
# The production held-out rules (stage 3's cv27/heldout.py, reimplemented in keab_test.py): the release's six tsvs,
# the held splits, filters version 3.
CV_TSVS = ("validated.tsv", "other.tsv", "invalidated.tsv", "train.tsv", "dev.tsv", "test.tsv")
CV_HELD_SPLITS = ("test.tsv", "dev.tsv")
CV_FILTERS_VERSION = 3
AMI_STORE_CLIPS = relabel_paths.ARTIFACTS / "corpora" / "ami-sdm" / "store" / "clips.jsonl"
AMI_CLIP_POOL = "a9d6609f004bcb52be62c376101eb3c99ca9a453c0ccb2bb5642bf08badad95d"    # config extra_sources ami-sdm:
AMI_TEST_SET = "ami"

# ----------------------------------------------------------------------------- scoring
MODES = ("x1", "fp")
MODEL_LABEL = "Kea-B'"
EVALUATE_SHA256 = "b3136a80558df15441b915568aca7199fbff878a392a2eed272f85fea9120a20"   # frozen evaluate.py
BATCH_SIZE = 64                   # K-fp: MB1-test's evaluate_set batch
MAX_BATCH_SECONDS = 1200.0
DECODER_GRAPHS_MODE = "no_while_loops"     # binary_paths.DECODER_GRAPHS_MODE (K-fp)
X1_KERNEL, X1_PARTS = "int8_mma:x1", 1     # training/deployed.py
FP_VERSION_KEYS = ("torch", "torch_cuda", "cudnn", "nemo", "transformers", "python")   # must equal MB1-test's
# Dev cross-check tolerances: plan values in WER points, compared as fractions.
MEAN_TOLERANCE_POINTS = 0.02
SET_TOLERANCE_POINTS = 0.10
MEAN_TOLERANCE = 0.0002
SET_TOLERANCE = 0.0010
# Sanity checks (plan "Report"): pause the write-up, never re-score.
SANITY_B0_RATIO = 3.0
SANITY_EMPTY_FRACTION = 0.01
# Sizes (plan "Report"), in MB: Kea-B′'s bundle tensors; the others from their records.
SIZES_MB = {"Kea-B'": 91.24, "MB1": 105.3, "M1": 180.8, "B0": 2472.2}


def points_to_fraction(points: float) -> float:
    return points / 100.0


# ----------------------------------------------------------------------------- outputs (fixed in code)
TEST_ROOT = KEAB / "test"
PRECHECK = TEST_ROOT / "precheck.json"
GATE = "gate.json"
TEST_JSON = TEST_ROOT / "test.json"
RESULTS_MD = relabel_paths.ARTIFACTS / "plans" / "RESULTS-keab-prime-test.md"


def check_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return mode


def dev_check_dir(mode: str, limit: int | None = None) -> Path:
    """keab-prime/test/<mode>-dev-check, or a smoke directory for a --limit run (never the production one)."""
    check_mode(mode)
    if limit is None:
        return TEST_ROOT / f"{mode}-dev-check"
    if int(limit) < 1:
        raise ValueError("--limit must be positive")
    return TEST_ROOT / "smoke" / f"{mode}-dev-check-limit{int(limit)}"


def test_dir(mode: str) -> Path:
    return TEST_ROOT / check_mode(mode)


def attempts_path(mode: str) -> Path:
    return TEST_ROOT / f"{check_mode(mode)}.attempts.jsonl"


def attempts_lock(mode: str) -> Path:
    """Serializes --test and --summarize of one mode. Lock order: this lock, then the shared GPU lock."""
    return TEST_ROOT / f"{check_mode(mode)}.attempts.lock"


# ----------------------------------------------------------------------------- scorer provenance (pinned after review)
NORMALIZER_MODULE = "$DATA_ROOT/stt-distillation/deps/transformers/models/whisper/english_normalizer.py"
NORMALIZER_JSON = "stt/models/whisper-tiny.en/normalizer.json"        # evaluate.NORMALIZER_JSON
WIP_PARTS = tuple(f"$PARAKEET_BINARY_ROOT/gpu1bit/B/wip/parts/{n}.py" for n in ("kernels_parts", "layer_parts",
                                                                                "parts_run"))
SCORER_FIXED = (
    "relabel/keab_test.py", "relabel/keab_report_test.py", "relabel/train_keab.py", "relabel/keab_step0.py",
    "relabel/training/deployed.py", "relabel/training/eightbit.py", "relabel/training/keab_export.py",
    "relabel/relabel_paths.py", "relabel/relabel_boot.py", "relabel/python", "relabel/python-gpu1bit",
    "relabel/ami_sdm.py", "relabel/labeling/filters.py", "relabel/corpora/registry.py",
    "parakeet-binary/binary_export.py", "parakeet-binary/binary_train.py", "parakeet-binary/binary_paths.py",
    "parakeet-binary/binary_quant.py", "parakeet-binary/binary_report_test.py", "parakeet-binary/binary_test_paths.py",
    "parakeet-ternary/evaluate.py", "parakeet-ternary/export.py", "parakeet-ternary/quant.py",
    "parakeet-ternary/paths.py", "parakeet-ternary/train.py", "parakeet-ternary/testsets.py",
    "parakeet-ternary/python", "whisper-ternary/wer.py", NORMALIZER_MODULE, NORMALIZER_JSON, *WIP_PARTS)
# The x1 runtime's source trees (training/deployed.load_x1 -> gpu1bit.runtime, its registry, selection documents and
# kernels, and M1's inference/ stack): every file of these, enumerated by scorer_keys().
X1_TREES = ("parakeet-binary/gpu1bit", "parakeet-ternary/inference")
X1_TREE_SUFFIXES = (".py", ".json")
X1_TREE_EXTRA = ("parakeet-binary/gpu1bit/shared/python",)
# Imported module -> its scorer key: the module's __file__ must be that exact file (precheck item 3).
SCORER_MODULES = {
    "keab_test": "relabel/keab_test.py",
    "train_keab": "relabel/train_keab.py",
    "keab_step0": "relabel/keab_step0.py",
    "training.deployed": "relabel/training/deployed.py",
    "training.eightbit": "relabel/training/eightbit.py",
    "training.keab_export": "relabel/training/keab_export.py",
    "relabel_paths": "relabel/relabel_paths.py",
    "ami_sdm": "relabel/ami_sdm.py",
    "labeling.filters": "relabel/labeling/filters.py",
    "corpora.registry": "relabel/corpora/registry.py",
    "binary_export": "parakeet-binary/binary_export.py",
    "binary_train": "parakeet-binary/binary_train.py",
    "binary_paths": "parakeet-binary/binary_paths.py",
    "binary_quant": "parakeet-binary/binary_quant.py",
    "evaluate": "parakeet-ternary/evaluate.py",
    "export": "parakeet-ternary/export.py",
    "quant": "parakeet-ternary/quant.py",
    "paths": "parakeet-ternary/paths.py",
    "train": "parakeet-ternary/train.py",
    "testsets": "parakeet-ternary/testsets.py",
    "whisper_ternary_wer": "whisper-ternary/wer.py",          # quant.load_whisper_module("wer")
    "transformers.models.whisper.english_normalizer": NORMALIZER_MODULE,
}
X1_PACKAGES = ("gpu1bit", "inference")     # every imported module of these must be a pinned file
SCORER_HASHES: dict[str, str] = {}         # pinned after code review: keab_test.py --scorer-hashes


def identity() -> dict:
    """The pinned Kea-B′ identity, recorded in every output and every gate (keab_test) and checked by the report."""
    return {"model": MODEL_LABEL, "run_name": RUN_NAME, "run": str(RUN), "pinned_files": dict(PINNED_FILES),
            "view_sha256": VIEW_SHA256, "bundle_sha256": BUNDLE_SHA256, "model_sha256": MODEL_SHA256,
            "dev_x1_files": dict(DEV_X1_FILES)}


def scorer_path(key: str) -> Path:
    """The file a scorer key names: "$ROOT/rel" under that environment root, else relative to finetune/."""
    if key.startswith("$"):
        var, _, rel = key[1:].partition("/")
        value = os.environ.get(var)
        if not value:
            raise RuntimeError(f"{var} is not set (environment or finetune/relabel/private/local.env)")
        return Path(value) / rel
    return FINETUNE / key


def scorer_keys() -> list[str]:
    """Every scorer file key: SCORER_FIXED plus every .py and .json file (and the wrapper) of the x1 trees."""
    keys = set(SCORER_FIXED) | set(X1_TREE_EXTRA)
    for tree in X1_TREES:
        for p in (FINETUNE / tree).rglob("*"):
            rel = p.relative_to(FINETUNE)
            if p.is_file() and p.suffix in X1_TREE_SUFFIXES and "__pycache__" not in rel.parts \
                    and "tests" not in rel.parts:
                keys.add(rel.as_posix())
    return sorted(keys)


def manifest_path(name: str) -> Path:
    """evaluate.resolve_manifest's file for a test or dev set (without importing evaluate)."""
    return paths.MANIFESTS / (f"test_{name}.jsonl" if name in paths.TEST_SETS else f"{name}.jsonl")
