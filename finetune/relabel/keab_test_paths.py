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
# Pinned 2026-11-03 after code review (sol r2 CLEAN on fa5cb96), from keab_test.py --scorer-hashes.
SCORER_HASHES: dict[str, str] = {
    "$DATA_ROOT/stt-distillation/deps/transformers/models/whisper/english_normalizer.py": "e0fc3dc9373209dbecd5881dab6a9834f76cf21d8031914c8f4fa1c3a1288461",
    "$PARAKEET_BINARY_ROOT/gpu1bit/B/wip/parts/kernels_parts.py": "6e8cb12deeedd76cc9a627c1fb361a4e9372c0f6f0289fd30890448b51d271b7",
    "$PARAKEET_BINARY_ROOT/gpu1bit/B/wip/parts/layer_parts.py": "4dc1a2e78d81885c533c42de83af10a4db0d65a55bdf884e598f17b601739786",
    "$PARAKEET_BINARY_ROOT/gpu1bit/B/wip/parts/parts_run.py": "8e308ce5117c9d2e8e06592712144d02e0f47346e343fc2ff580134da42d2d29",
    "parakeet-binary/binary_export.py": "490f185b725a95bfcdbcbc871e68d1cd44970ed0a1c9141f2d4031382f6c83cb",
    "parakeet-binary/binary_paths.py": "f6c95c2914950c6477c83620097c7724df3b7d5c5251bd76dacf67407c253d79",
    "parakeet-binary/binary_quant.py": "3da0acbb3b64fc4ddb3a14fbee45ccec352274502f85891c857d80541720f6d5",
    "parakeet-binary/binary_report_test.py": "6f6d396859a9e8ceae02b8170da2e0cacf37737bde33e450c291e755f07c8731",
    "parakeet-binary/binary_test_paths.py": "8e28c86f1ccb842d3eb8047dd3e38205a7cf995a39c9d745cc7a44693ed17b1f",
    "parakeet-binary/binary_train.py": "77a0012288e84b3596c0d892c2888f2a1ec9afd1e639d53374086169f921166e",
    "parakeet-binary/gpu1bit/actprec/__init__.py": "b6c3c0a3a761eb6380fb862bc5572041bf3b552ddbe37ad02a3b6caf8f3d5d82",
    "parakeet-binary/gpu1bit/actprec/bench.py": "31050ea784f4ed2f0ed79108e47982dc469824b1d4baea7e54c07e08fd410fae",
    "parakeet-binary/gpu1bit/actprec/parts.py": "befaf14c4950a9bf9fff0359aae29c3026423e1c35682373061ea92f44072ffb",
    "parakeet-binary/gpu1bit/actprec_report.py": "4e74f31b82e1ab539931ead554981ae3c8c8e7616a20dfad57de0d1f47c23abd",
    "parakeet-binary/gpu1bit/bitserial/__init__.py": "74093397a54f0f0c8641713ac20084fd68a0477d7c45a25f51ac335939e1d6b0",
    "parakeet-binary/gpu1bit/bitserial/kernels.py": "5a3c2f6a73d985717f5352da8ec4f36127735e04677bd7f5a27f3b1678605c81",
    "parakeet-binary/gpu1bit/bitserial/layer.py": "d37497e63cbb61135502662176f3b5ed91d6b26faea72bf5ced97104772784d5",
    "parakeet-binary/gpu1bit/bitserial/mma_probe.py": "a9f12631f35b43f2b06558d897ff3fbb1fe0f99bdb5a46698f91c7bf1b6a463a",
    "parakeet-binary/gpu1bit/bitserial/numerics_probe.py": "3c6612f6a61ab4c67bae3215cf9ca36e6709bf649b4f404c91aede1b82d449d5",
    "parakeet-binary/gpu1bit/bitserial/nvrtc.py": "e82c0a5a9db1c4a1cc3cd0c4bde1e03ceb24a286826fc71a857fa8ae2ba3912e",
    "parakeet-binary/gpu1bit/bitserial/run_locked.py": "f55502bc079844975e9fbfcf1548d750e3b0af07bbb952df35f4002db7b0057f",
    "parakeet-binary/gpu1bit/bitserial/sanitize_c.py": "c62555a44eb427253fa1942fd4579ccc95a2e936c07a8fb81521db01f16f5a97",
    "parakeet-binary/gpu1bit/bitserial/test_bitserial.py": "40eb02418858d932f784f225592b294641a488c14332c0da8500dbd72237cdd3",
    "parakeet-binary/gpu1bit/bitserial/tune.py": "0b9867bffba2a3711066a76f664104941dc8aa1ba426549c80376fc7f91a8d1e",
    "parakeet-binary/gpu1bit/int8_mma/__init__.py": "a5d6556ef6e47fedcfad080715474ca13fc2af32854c3ed9e31ecb8fe9522250",
    "parakeet-binary/gpu1bit/int8_mma/audit_x2.py": "37393a88426bc2c28258c18b7557f53a665ab728364312f2a4be5be39adebb3b",
    "parakeet-binary/gpu1bit/int8_mma/b01_run.py": "bf3c60d6ffc0ff232b7eef092d22580d33c22b051806084424541b5c24790032",
    "parakeet-binary/gpu1bit/int8_mma/bench.py": "44fb1e2c9ead745a2d0fb2856c0a9ec4b8267ba63c21a4d24e6791b9d6ed094d",
    "parakeet-binary/gpu1bit/int8_mma/configs.py": "ee8d7cf0a92b2e7638fc77487f6eebb8bf7d5acc2f39ac6d6368d4f8b5b14f06",
    "parakeet-binary/gpu1bit/int8_mma/cuda.py": "4a517ab863e59fe6357a5493fc43645d25f75904b9a5e5c748184ec9bd803185",
    "parakeet-binary/gpu1bit/int8_mma/deferred.py": "41b372606fcf6c8ad741887649508e0b1e67483cc732d56a1199617a97a28022",
    "parakeet-binary/gpu1bit/int8_mma/deferred_x2.py": "e5b32e7a6cc87898ac5aabd9d8d3e50eeead0cb4b50cdd882bcec889dc76ac71",
    "parakeet-binary/gpu1bit/int8_mma/gpulock.py": "309615c7f36411301921f1c9aede5a977219a646b91f633b3a8c98530b1645b4",
    "parakeet-binary/gpu1bit/int8_mma/hybrid.py": "90d559b4f2d23928c87e4efa2ab74ec0ef6399edbd6d85b3aa4bffe7ea1b7fb6",
    "parakeet-binary/gpu1bit/int8_mma/hybrid_exact.py": "09139b96382ec0fe01b3acbdd208cf83b20925b2aba54791200e0f1c47017ff1",
    "parakeet-binary/gpu1bit/int8_mma/hybrid_x2.py": "e35cad215de5181b8562b5cf883eece3b0f5c25e72ddf987ff8e4b8de9dbf08b",
    "parakeet-binary/gpu1bit/int8_mma/kernels.py": "ae57a8e86eb0b161ab031a7e3cf56a8a5a3d96bb9053586106c71d21e785b72d",
    "parakeet-binary/gpu1bit/int8_mma/kernels_b01.py": "75583a19bd301d720fbd1cca105b1e789c870e927eddaead1e40b18c54c9206b",
    "parakeet-binary/gpu1bit/int8_mma/kernels_exact.py": "8b8f805b3269411668f9bf5b89d21eef8548f01b69d4ebe13fceac331c20f8bc",
    "parakeet-binary/gpu1bit/int8_mma/kernels_exact_x2.py": "b32bb1007fdfbf4ba91e7726e1206356fbd399660c5071d614afd5f20260d40f",
    "parakeet-binary/gpu1bit/int8_mma/layer.py": "a8200dd28eb9853574eb0efd14e2b258ac11254cd243a59b82363433a4f38f05",
    "parakeet-binary/gpu1bit/int8_mma/layer_b01.py": "a76a5c7789b2dafd7b4f77ab163283ce20039a8dbab843bc969b433521667e78",
    "parakeet-binary/gpu1bit/int8_mma/make_configs.py": "6fc1057165c80093a1a74d0076ba1528d1a2bd1dc50a550c0300f36f6c1c23c6",
    "parakeet-binary/gpu1bit/int8_mma/numerics_probe.py": "9b84b2636d6d47f04b55f522c732d3125e14e93e04a69cdea1ba2c8a1fa4c0ed",
    "parakeet-binary/gpu1bit/int8_mma/paired.py": "f1ca5b42c0fc2c1fee4531e04d70af4f6360e7ab7da0e29cc79551854df92144",
    "parakeet-binary/gpu1bit/int8_mma/phase_b01.py": "025c889f1273c5e66d8f8c56121040c67a21a7f2be4bd46c2304ff8048621974",
    "parakeet-binary/gpu1bit/int8_mma/profile_ops.py": "c3c8f945ad6d86bc8ff302c21b6d9d789a6e399b98d31dd7b7ec4e10d5527c9a",
    "parakeet-binary/gpu1bit/int8_mma/repro_c1.py": "d30eafed1db1e296e522376ccf535ad968015a2006b53b49f41b7212bd356df4",
    "parakeet-binary/gpu1bit/int8_mma/row_exact.py": "3bb1e768938e3405f632d2757713db0b89dc40a67079c9361ca4b3072b3725bc",
    "parakeet-binary/gpu1bit/int8_mma/row_exact_x2.py": "6f6efa0787bfa9890802b2c2ddcadd1e1b670a27f835e0b7ed559bfc4d381afa",
    "parakeet-binary/gpu1bit/int8_mma/sanitize_b.py": "a2103dc6da54147624ba644cfd01a57201861d6c68d82caee47008bfb6b480a2",
    "parakeet-binary/gpu1bit/int8_mma/sanitize_exact.py": "1d7ea5ca057118eb33043e23cd8aa2574c26a30bac291131791daad150f67822",
    "parakeet-binary/gpu1bit/int8_mma/sanitize_x2.py": "fe3c5ce22a1373627ee8030d27b42cc5173cdaf9a80a89c47021692dd807915c",
    "parakeet-binary/gpu1bit/int8_mma/sass_b01.py": "748722c6e119c783db84b733478465c3485070a157a61720ad9ddc33b7235f17",
    "parakeet-binary/gpu1bit/int8_mma/session_patch.py": "315adac684699b26e7f6ec6e8f46ac81b550503deccc2182026155d583b39073",
    "parakeet-binary/gpu1bit/int8_mma/session_patch_x2.py": "0fc470782f3878db77f6564f9de13732b14987f72a1b649177b5681cf9668b22",
    "parakeet-binary/gpu1bit/int8_mma/smoke_hybrid.py": "3ef21c993f5228cb1cf3ebadfd83f4c1e3268720fb7ef4ba8c3c8b3972d00f37",
    "parakeet-binary/gpu1bit/int8_mma/summarize.py": "9432c4e52a878ee4cab6525486b1b24c639228ed5ed22b7c2816591db23468e2",
    "parakeet-binary/gpu1bit/int8_mma/test_b01.py": "be911b2f93c2ebc45cb22c550da1b65c2edb1edaf864b0d3cb59e4f69d349887",
    "parakeet-binary/gpu1bit/int8_mma/test_exact.py": "6d08c963126b717a09644d9b9e5f96f3db22d89bab70c6bf7ce886c44711baf2",
    "parakeet-binary/gpu1bit/int8_mma/test_exact_pipeline.py": "c0c5995cf7fca1f06ec664f2a37d73c047f5e9a9220fa3b651071e0066259cea",
    "parakeet-binary/gpu1bit/int8_mma/test_exact_pipeline_x2.py": "666b41bd50b4bba927d82495fa338f34ef05031a8c1847b733e1b24c9952d86e",
    "parakeet-binary/gpu1bit/int8_mma/test_exact_x2.py": "1c40633f9314c21400beeefd576223bde6b61bc6369e6fd9b9d70088caa95f9d",
    "parakeet-binary/gpu1bit/int8_mma/test_int8_mma.py": "8c465fdf9acb76ba2fb02ec9b56c51543967057cc6c90382e18af3c5ede29db5",
    "parakeet-binary/gpu1bit/int8_mma/test_x2_imports.py": "b416277666df16c318d074607d8e3e9ceed810f31448b18e4fcb7133e205e3f8",
    "parakeet-binary/gpu1bit/int8_mma/tune.py": "42e6b00141abf20093cfa920701468203a45f95a4ceeef5911ac4d189cdd009f",
    "parakeet-binary/gpu1bit/int8_mma/tune_exact.py": "5fc8c4ab3333fdf6b3cd6886ef2fece619ca74793d7bb9c8cedeb54c0a2a5b37",
    "parakeet-binary/gpu1bit/int8_mma/verify_hybrid.py": "3d5cd4099c0c4dcf38a076832cc46d21d84e6b8ccdae2d45ba7d88e572d39865",
    "parakeet-binary/gpu1bit/measured-x2-parts/kernels_parts.py": "6e8cb12deeedd76cc9a627c1fb361a4e9372c0f6f0289fd30890448b51d271b7",
    "parakeet-binary/gpu1bit/measured-x2-parts/layer_parts.py": "4dc1a2e78d81885c533c42de83af10a4db0d65a55bdf884e598f17b601739786",
    "parakeet-binary/gpu1bit/measured-x2-parts/parts_run.py": "8e308ce5117c9d2e8e06592712144d02e0f47346e343fc2ff580134da42d2d29",
    "parakeet-binary/gpu1bit/measured-x2-parts/precompile.py": "02dac89308ece796f962c9c7ddb733f2f89c03fa6daca666b953e61982dda9a0",
    "parakeet-binary/gpu1bit/measured-x2-parts/run.py": "b558530395c74cdb6382190cf1556721df245dadd6f86ee9b6863336d5e9af66",
    "parakeet-binary/gpu1bit/measured-x2-parts/sanitize_parts.py": "627563dec1ca49479f0b397816b30a852fa8698516d7b8482593504dea77435b",
    "parakeet-binary/gpu1bit/measured-x2-parts/test_parts.py": "a435fed64ac6d8e1904c881f6fa689b052b62cc5e14d2bd336acad0c9c076e9f",
    "parakeet-binary/gpu1bit/measured-x2-parts/wip.py": "0c5415138d79946bfd7d370badd166bda15697b810adb56b1786dde11c5bb4cb",
    "parakeet-binary/gpu1bit/measured-x2-parts/wip_main.py": "1f1e1eefedb1e5e95ca7b57cd342a9e11909e2433851cb8ab4cfeffb9a8afc14",
    "parakeet-binary/gpu1bit/rerun_timing.py": "0f82a90477479ba38dfe2c2a9e1042f7ad807a9a996a8cb6838349e8b2d92d36",
    "parakeet-binary/gpu1bit/runtime.py": "a3a5c60bde463254acaf5d7dbe16233eabef9589fe451aa95041fd884ce0f387",
    "parakeet-binary/gpu1bit/runtime_m1.py": "bbe9ab7f4a797ec8cca2b04d59ba9435b27c8a4197899a0db4a1d80cd607186f",
    "parakeet-binary/gpu1bit/shared/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "parakeet-binary/gpu1bit/shared/api.py": "eb150c3239cb220a7dbb7fd685519d51cc263b77a5a792dcb41677a87bcba72b",
    "parakeet-binary/gpu1bit/shared/audit.py": "3c0eec2a01fbfe97758081243bfa79400e31a53026c4a4c7eedb69ed433fa7ef",
    "parakeet-binary/gpu1bit/shared/bench.py": "da50840d0697e4994d3bda0ae3d2e57b1cb51b5438be0b97b8931006f7d57ded",
    "parakeet-binary/gpu1bit/shared/deferred_a3.py": "d3702a492ce9a44466b3b7ab3cdbfe5072c48738c0a436dc58b0cbf6d7cff5b5",
    "parakeet-binary/gpu1bit/shared/derive.py": "cec8c48432635e086dac930f2abc7de03e88bc92ff98c2b750d27f3ec4dc5a78",
    "parakeet-binary/gpu1bit/shared/dispatch.py": "fa043899d85e8ec33332c99b48b48f94aa84d3a0a583accf029d051190f71988",
    "parakeet-binary/gpu1bit/shared/env.py": "02427539e7d952fe85e478930f96bdc2dc579bc176680f3edd639e6e22d7dc02",
    "parakeet-binary/gpu1bit/shared/fingerprint.py": "fe0db289ca5f5c57b2b1bc7abd8ca2a350d8a01c2765294d1cc207807f27dec9",
    "parakeet-binary/gpu1bit/shared/fixtures.py": "274de718c4e5c83d3c049fd9f35df113ccb7395a61949aaa62a170b62b19ca68",
    "parakeet-binary/gpu1bit/shared/fulldev.py": "736b00f80c0969261bb7c6a421ebfecb794c3fcd151cbb086530e5de91770fe2",
    "parakeet-binary/gpu1bit/shared/gate.py": "3f4b3e1325c0fed28bdf842bfc3b3b7c7ab01c078ac87a3c1b7b72e85bf88181",
    "parakeet-binary/gpu1bit/shared/m1.py": "9b8c7f8530b2b8ee2d0d9363a3f71b442aca3c8360fd697223df1e23b6630f89",
    "parakeet-binary/gpu1bit/shared/make_shapes.py": "60f18a202a0fbfd5baf80fb84e1a72c1b7958df71337a9ade1ff59bf8f5632be",
    "parakeet-binary/gpu1bit/shared/phases.py": "6d005e3c274ecfb8d56cff700db9dd454c117931fa81198ecf433aca729bc7fa",
    "parakeet-binary/gpu1bit/shared/provider.py": "97774ece13aa798209d9c873f35f0c41805e4b6289632b090bcf759f8c4ee578",
    "parakeet-binary/gpu1bit/shared/python": "e7711b5ccc524861633238ec7d6227049dd107b6d62783f0ec2d4911301fc6df",
    "parakeet-binary/gpu1bit/shared/reference_layer.py": "4defa3b631a096a9afe1d0db9a68d6de39b6d7402731ba2fed5d42edad77b8f2",
    "parakeet-binary/gpu1bit/shared/registry.py": "1ce1a5779cf20db779ccc1c3ecc394594c551dd16f895d24b33b55f3b73e75bf",
    "parakeet-binary/gpu1bit/shared/sanitize.py": "bd4708f62197cd983f71bd11cf509413cfef801c40ae6ffbd02591cfb9b8ed4d",
    "parakeet-binary/gpu1bit/shared/seldoc.py": "86231fa6eb258cbde312deb5def0328f1c932b6ccb37bb1e0559716576f4c8fa",
    "parakeet-binary/gpu1bit/shared/selected-x2.json": "de3c18c4ab20d33b04abad970ae6ce9988dd7a4dc7cea597fce3ae1c9555ba2f",
    "parakeet-binary/gpu1bit/shared/selected.json": "de3c18c4ab20d33b04abad970ae6ce9988dd7a4dc7cea597fce3ae1c9555ba2f",
    "parakeet-binary/gpu1bit/shared/selection.py": "96fad61229c59cba50304e58d6cc4b741125c06b665ba52cd1042b3205adc1c0",
    "parakeet-binary/gpu1bit/shared/shapes.json": "fd5f56f998fdc720fba11024b79d1150088be7d132c4bbd2fefcb8da46be72d1",
    "parakeet-binary/gpu1bit/shared/workloads.json": "598c1d72f01f0ed26a0415da474a362231ad481ae0b788a95c509ab2f4ef908e",
    "parakeet-binary/gpu1bit/shared/workloads.py": "2b2d01e7604cb3df2eb20880f56393fc38af86a649c4b746e417c0336d203520",
    "parakeet-binary/gpu1bit/triton_bf16/bench.py": "774d108e30befe0de60acec4bc0caeeb491105983ee63967d711492dc24a1498",
    "parakeet-binary/gpu1bit/triton_bf16/compare.py": "98bb44dad745e8326e342acc7ebfd2c3fdd4ec7c624a11b6c15a5d4603c36988",
    "parakeet-binary/gpu1bit/triton_bf16/configs.py": "9c53daf0c588c870278b2dfb8447efb413e586982a223a0a202e140855cd4ef2",
    "parakeet-binary/gpu1bit/triton_bf16/diag_numerics.py": "1c16c99bbdb37090b03ced216f346c15205890ccf43355f3b0535fd1817fa848",
    "parakeet-binary/gpu1bit/triton_bf16/gen_configs.py": "a9c04231f8523c02737533dd6e5f4532a688ffe4674108b553b61aaa0867c44c",
    "parakeet-binary/gpu1bit/triton_bf16/gluon_kernels.py": "0ff15cfa6edf6cf7290145f281181b97c60b22258795d1ea1c4384e69399b329",
    "parakeet-binary/gpu1bit/triton_bf16/gpulock.py": "5e76f35f5fafd8e1d3ad15192884b8d25333dfb30e826912be4ec03e32e1e7fb",
    "parakeet-binary/gpu1bit/triton_bf16/inspect_kernel.py": "4f0814e797184912235a87b2643c4d8d86873ee756c8108e52713ba0876573d8",
    "parakeet-binary/gpu1bit/triton_bf16/kernels.py": "9f309b4d0c8b06097a79ae4fea5517bdad872259910e610b1536c0355edd2599",
    "parakeet-binary/gpu1bit/triton_bf16/layer.py": "176b7cd042e3af22235224b86ed718d87a213f88ad72444de38512b1bad3fed0",
    "parakeet-binary/gpu1bit/triton_bf16/probe_decode.py": "c87873869b7e94df0245fc2e7439e3b79b2d2181d5fe4ab48124e726d40ea7e2",
    "parakeet-binary/gpu1bit/triton_bf16/probe_gluon.py": "7b5e14a4f726b907f3ac61211e5f79ae9761a26cbd3a71fa5efb36bfe45ce00a",
    "parakeet-binary/gpu1bit/triton_bf16/probe_identity.py": "9d20f48b35017bc3f3dd2030bfc3e240137f0bd0f1171fb342dc2331f4e21f0f",
    "parakeet-binary/gpu1bit/triton_bf16/probe_int8.py": "d068d6bcb0a03c6e6bd180617a51afdc3984fbdd651acabf4b67f4c280652b7b",
    "parakeet-binary/gpu1bit/triton_bf16/probe_loop.py": "031a0b464496059c620517db89a8b4d727911d89891229527785efa5ef087f53",
    "parakeet-binary/gpu1bit/triton_bf16/probe_overhead.py": "e5b9112efc36714ed8ffaefd33b1d1784442d7ac9552207e7242cb15cf4df973",
    "parakeet-binary/gpu1bit/triton_bf16/probe_pdl.py": "0ee067638b57ae5579b445914a589e8dcb87f3743263dd0f7edf4cc5ba080950",
    "parakeet-binary/gpu1bit/triton_bf16/probe_peak.py": "18ac3ac31608506cbafbd0d3f2eb18c19cdf2d2479f8491aab3d96f2bc6643a0",
    "parakeet-binary/gpu1bit/triton_bf16/profile_ops.py": "e132eecb5c78b0dca35baa6c264cea870bd47a06b568f51068e9fd3b4362fc4e",
    "parakeet-binary/gpu1bit/triton_bf16/run_tests.py": "d55e9a598ad5ef4cc1ac5761ae013adcbda51c53accbad50d92626707c460957",
    "parakeet-binary/gpu1bit/triton_bf16/sanitize_project.py": "a41cd3e83a04d66ec256b6a5110591b593ef7edb8ce891cf0598a145af7aaf16",
    "parakeet-binary/gpu1bit/triton_bf16/test_layer.py": "0ab1b1c9a1ab33dff784abc3e9537c4a38e422567a40b53adcc8b7c88905afd9",
    "parakeet-binary/gpu1bit/triton_bf16/tune.py": "acaa98e13f1ea10c20d782ddc6cf80b7d4aea7d6878cb43fd2284765b5781228",
    "parakeet-binary/gpu1bit/x2/__init__.py": "e0a26a2e4c3b7197c7b3907391942db763df7565bceeafb91b532338e3c62c3e",
    "parakeet-binary/gpu1bit/x2/bench_patch.py": "910560444ade7b604b40760d2420eed2d932b72a34e9b10470f19c050bc78cc6",
    "parakeet-binary/gpu1bit/x2/bench_roster.py": "f42f57555a7e128687e4c16704627ff872736d4cf9ff9530b04da788b819abb9",
    "parakeet-binary/gpu1bit/x2/bench_session.py": "fc16b8ea599ac158789428a596722de23d006d4e59e8d4ceec660f2e552115f9",
    "parakeet-binary/gpu1bit/x2/gates.py": "f8754ec8ce21fcb9b460c53bcd5f691d275dba03d395131032f0841585dad13c",
    "parakeet-binary/gpu1bit/x2/historical.py": "df0574378f0b57ec0acb8160b4edda09ea6f3db398868c610fffce606e793641",
    "parakeet-binary/gpu1bit/x2/preflight.py": "058c308d39009dc17b74a5cd37bc22ec4f3b254ada0bf8ef8b36f2d5c9ded4dd",
    "parakeet-binary/gpu1bit/x2/recorded_arm.py": "21293f98f173630ff9d51a9b658644c538f1253ee8a8861601afaa48bf9a4dac",
    "parakeet-binary/gpu1bit/x2/runtime_contracts.py": "0882d062b0b95e5068bba5a516582311fb60f54b4492ed3172740c6ae4c99d55",
    "parakeet-binary/gpu1bit/x2/testscore.py": "36e96072fe22ecf325a30a40472bcd1af5c00ce5cdb42bb756d67ea25fa1b88f",
    "parakeet-ternary/evaluate.py": "b3136a80558df15441b915568aca7199fbff878a392a2eed272f85fea9120a20",
    "parakeet-ternary/export.py": "1cfeb049192d81b8cb993e95e603234f5fbe5dd7829af45693f5e28ae39d5065",
    "parakeet-ternary/inference/__init__.py": "0e9296dd1db393ce8e54846dd3ae33f59f249d8e784abbfd345466488ab02767",
    "parakeet-ternary/inference/__main__.py": "528f56ae28c35b774b710592b9d38265a4abd4124d3537b94e8953d3ed1539bf",
    "parakeet-ternary/inference/benchmark.py": "b017eca7d7cb07b89a1b947cb02cd983a61b1afc0eb0414a0b95163c2cee8785",
    "parakeet-ternary/inference/benchmark2.py": "3f8d0faceea38a19592c87cdd6aeae7bf3d448fceda26d71f08979cbaaa70cf4",
    "parakeet-ternary/inference/bitwise.py": "dacab7d29a3b71f80377b0f4bf873b5fd90845e2ec1a3880be0c6efed2f128f6",
    "parakeet-ternary/inference/bitwise_benchmark.py": "2efd05dab67cfb911a1db4057da3ddf55af863101b0c529c62e15dc622d964d0",
    "parakeet-ternary/inference/compare_v2.py": "e28e86929826d9a937d014ad20548ad0ab37171dca9cce709372105536f78dfb",
    "parakeet-ternary/inference/compare_v2_components.py": "4da9252f81cf91b3ca417e10ddbf665f4a6fd3f733f492be3999fc168ead4622",
    "parakeet-ternary/inference/compare_v2_report.py": "e09789521f8f2db1599ff94c19c7f33231df84e1a1e6397e15df4438e52a5cb9",
    "parakeet-ternary/inference/compressed_weights.py": "b9a4e75b7bd94743866b5b973bf318bf5b57a205d68b2415e94d650e416963d5",
    "parakeet-ternary/inference/cuda_cublas.py": "acfd3e0fffac55da2558733068c5ca5063abc9d55163594c75fa045cbe030b1e",
    "parakeet-ternary/inference/cuda_experiments.py": "2ceced1cf1edbf9ac2824d3cbeaa8913d04d1de6c9e6a45fa69fb58d9d538ea4",
    "parakeet-ternary/inference/cuda_int4.py": "512e32ef1d4a5c7c8afabe3784fc42048eb2050e248c63d9ee2353cd930871b4",
    "parakeet-ternary/inference/cuda_matmul.py": "1a7f2152dce80cbd4998ec75a0836fbc43f8e351d12c8a2e4291db8c11995337",
    "parakeet-ternary/inference/cuda_pipeline.py": "829dbb6a4d03a1d566ca8963af30105c9eeab151282acbe059d1d3e78033c0a1",
    "parakeet-ternary/inference/decoder_kernels.py": "333d7330062157f0e04b2f2cab7d5c209f087691bffa732c4b7feb11fa063c58",
    "parakeet-ternary/inference/encoder4.py": "924cfd2552744868247e375ae60082b7d82c2d87ff66d9fadc08fa833ffe93a3",
    "parakeet-ternary/inference/experiments.py": "21395f1d467c36e1ce8fdd254f83913c46bc945f043c77cb056c6c26103591ae",
    "parakeet-ternary/inference/fused_kernels.py": "79777e18d0a4961224447432ffa86ed8adf54f415eb98e967f8af1bfeb0e7163",
    "parakeet-ternary/inference/graph_memory.py": "c36061eff855968c0e7fd284911638a38ad2af1603423c6345b49970d39a00a2",
    "parakeet-ternary/inference/graphs.py": "a3eaef81b17c540e44e01d37b707e7fad962d679c65d30699ca551053e02e6dc",
    "parakeet-ternary/inference/grouped_dot.py": "4ed54231888b157125523699b661a7b8b1f165988cd4f05e552d8e51f32fff39",
    "parakeet-ternary/inference/kernels.py": "7b8d0e0e2da37b56f6f6562ce287bea090bc3edcbd9aa46e2831c2557ea474e7",
    "parakeet-ternary/inference/layer_fusions4.py": "e30f8d2035409333785bb9714c2bc53edd3222a1ace460d4cf91017fa5041b68",
    "parakeet-ternary/inference/native_groups.py": "bcf2d76a1002620639b070abcd87c4421a05f14ae334b0d29defd2f405b1951f",
    "parakeet-ternary/inference/optimized.py": "8401e907ea22c2795f3b8e13cd9e7bfa05c42ecc62ec41b4863aa9867af0a29e",
    "parakeet-ternary/inference/pipeline_probe.py": "92cd34b030846ddc903ed518ca23a01d2f856e54f5b32dcabc2710ab14ca046c",
    "parakeet-ternary/inference/profile3.py": "5b226a2d357fc552261b6da690924ed0119af35b458817c90296ca4a1884ba9d",
    "parakeet-ternary/inference/profile4.py": "b27f5e3bf01cde38c516d34933cd847e3fb7af04896c20fe8927bfb6161d6865",
    "parakeet-ternary/inference/ptx_cuda.py": "784be5f79270d2ff595d5d1808ecb82972b52dd90e628f918f4864d415789c4d",
    "parakeet-ternary/inference/ptx_tune.py": "62048064480792df8b9e84566401d5509b38e84f88ed868935441aa6a167ce42",
    "parakeet-ternary/inference/quality2.py": "3a5c7ca3c31b84d4000007b3f4b696ef79dbf69a6875687468ebf5216d988ef2",
    "parakeet-ternary/inference/quality3.py": "dd27bc7416ffe5e14b2e24e643083eba8fdac0873beffba4b038adf52456f06a",
    "parakeet-ternary/inference/quality4.py": "b1b06ac62f397cd72c52ebb7b0da6c09bc1d1e3ea02d62ef943f2556f98d2e18",
    "parakeet-ternary/inference/results/rtx5090.json": "76828cdbd9eb4bfa281cba4ad2ee95e9aaa3dfac28aee49a57a6708fc50db411",
    "parakeet-ternary/inference/row_fusion.py": "712d5823bb77ddd7385adf560a8010656769de2fe54e7681f91406f1e50c6386",
    "parakeet-ternary/inference/runtime.py": "f62de5bcc7aa850d8ca5920a8b187636859065e4d332edcff4851e16e70328da",
    "parakeet-ternary/inference/sprint3.py": "8e4f5531ca13f16f004be4bdd8583bc9c329d1ae5b6283d54d290c6b19d29fa5",
    "parakeet-ternary/inference/sprint4.py": "397335dd2416c4c705c70606fe60d33f2fd1d3280380364ae41b12f645e2da24",
    "parakeet-ternary/inference/test_bitwise.py": "cfcd9bd5c6efd500de03d342d079382a7abee0741742ce8a84e77e25af9165d0",
    "parakeet-ternary/inference/test_cuda_matmul.py": "960e4e085dc156212d16d5c6e06906e5cbd6be1fe2f4b4f72b73a8d40d3bf5f0",
    "parakeet-ternary/inference/test_decoder.py": "e5eee466121f27093365b9ffab52718fd275c87cc8a40f6d0c12b300edd3acf8",
    "parakeet-ternary/inference/test_encoder4.py": "f440e544900dfd8414358fad453f9a7caa3fab28b5c78a5c8c505d3706452923",
    "parakeet-ternary/inference/test_experiments.py": "7bc6d92cc5c00e3819ac8c54fec718b308a56442b112fef2a2ebaea071c60aba",
    "parakeet-ternary/inference/test_kernels.py": "683cedc8d3314036874c53a2aa3940a7e296872ddf589105631c13147717a997",
    "parakeet-ternary/inference/test_model_graphs.py": "c82b605cb19642aca071170b20a46c5d16c7a091f3a5c11664b154d4989cdd8f",
    "parakeet-ternary/inference/test_ptx_cuda.py": "eceda0368deedc828b09c5c4afacc47580200918f5512ef52c7d38f22ef88097",
    "parakeet-ternary/inference/test_row_fusion.py": "3b1a5228658a6ddd680fab099872805c1db6fb9f5c758f2a6d993ce681165251",
    "parakeet-ternary/inference/transforms.py": "be3ac9ddbcd3998526b9f9f1d0483e9f72ce2fb179eaed3e0edca99a6ca0690f",
    "parakeet-ternary/inference/tune2.py": "2b04e5adb3330c4e309b260f89538ec9aee72da7751cc0e4aebd71629a3f999d",
    "parakeet-ternary/inference/tune_expanded.py": "79407cfe242bc5b61919241b9c13d6cb666b4d8cc0b0b991ef25c31d1e504082",
    "parakeet-ternary/inference/tune_integer.py": "e0db253ff76d6bd7b77a6cb3c2b7137821eabb7b6b0fcb96238b937ef4a491c7",
    "parakeet-ternary/inference/weight_audit.py": "16d7b53ae9e7b06f89a0f3492779e0d7d84a9677a6112b1eb779c3c5c75b9f18",
    "parakeet-ternary/paths.py": "8ee2ed9298d434907cda9f055ef67282fd42ef73eae49c2cf0111c250cb33fa2",
    "parakeet-ternary/python": "18ea70d6bcd533d8c7fc829190158b90617c2a26184c9e67e64e8313e39942f5",
    "parakeet-ternary/quant.py": "b60a6118b8771b479a4356d380de9722a2a5d833dba5d2431911a5e18aa7d66b",
    "parakeet-ternary/testsets.py": "9668f573489cd31d25ec4e402eb8eb8fa95e1e6dd094da5628316a1dfc5fe195",
    "parakeet-ternary/train.py": "d44996b1b2c313ac8baf04228a1b8873ff0df12c99da9708459191ea7cf3c853",
    "relabel/ami_sdm.py": "ba3b572bf44033a0a50a2ff343d7020e60e2a183f4a0157e1c672c6f2908d7b0",
    "relabel/corpora/registry.py": "6396591db90d8683271b67abf37a284235df822dbd89ed1f5a7cf7719bda3970",
    "relabel/keab_report_test.py": "b7c8dd84f67a4ce20e7ded46a135b2557e5b6ee3ae502a7dd58196316fb20bc6",
    "relabel/keab_step0.py": "4b7268077573990f6a3af1363c4335bc921dbe9d6bfb633800f8d4ee00798c07",
    "relabel/keab_test.py": "1e30113cae122bdc05fcd4c789671932aa0a6a24b8357f2e4c4956ea3308dc4a",
    "relabel/labeling/filters.py": "d2b06f9f59b3d1aac2cabb9b6a01fed9672dfaf64f627acd78b6e4cd0c927a64",
    "relabel/python": "d8f4ac856cf3d17a91a676726045587f8ff4ce8b19950f03a91cd3ea145f2ae1",
    "relabel/python-gpu1bit": "8dc21f43e9888e56abdf5bf8df772e267fce639dce6cfb096fc574effa48c1f3",
    "relabel/relabel_boot.py": "75b94b6d52b3500c07b6b5867ffcfaec7b2ca3051757cc80fa1b23f5e91a9417",
    "relabel/relabel_paths.py": "53fabd35f220acbffde1bd05082d4d5dc2f80416e65cdaaee3920c760a84feb4",
    "relabel/train_keab.py": "daf70171524e65b9f65ba470ec70acdbd9184502ef5a7dae7b8b8ed6fabba3c2",
    "relabel/training/deployed.py": "8741c24e376b3d857bcfb1f7f71f6de47d7a01dba8e59d7f8e2e88a7ed42df8a",
    "relabel/training/eightbit.py": "aef5590892d5765c257d7f3a545f3fd2be31227745463e6124d1c0a2dac56e0b",
    "relabel/training/keab_export.py": "416babed62139eb6018766b939cf7d4fd2de0a7c8129e4adefaad9cbc561a9fd",
    "stt/models/whisper-tiny.en/normalizer.json": "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd",
    "whisper-ternary/wer.py": "f1ced52d0b5b0a738cc7e3e494bae3b634ee0f0edb5f521e2974a900f5e41ae4"
}


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
