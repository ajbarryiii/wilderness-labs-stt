"""Pinned inputs and fixed locations of the MB1 test-set scoring (plans/test-scoring.md), without
torch so the report can import them.

Every check in binary_test.py and binary_report_test.py compares files on disk against these
constants, not against other recorded hashes. This file cannot hold its own hash: its SHA-256 is
recorded in the plan's Status line once frozen (binary_test.py --precheck item 3 checks it there).
"""
from __future__ import annotations

from pathlib import Path

import binary_paths
import paths

HERE = binary_paths.HERE
FINETUNE = paths.REPO / "finetune"
PLAN = HERE / "plans" / "test-scoring.md"

# ----------------------------------------------------------------------------- MB1 identity
# Recorded 2026-10-20, before any MB1 test decoding (plan "Pinned MB1 identity").
RUN_NAME = "main-binB-m1-lr2e-4"
RUN = binary_paths.RUNS / RUN_NAME
EXPORT = RUN / "export"
PINNED_FILES = {  # relative to RUN
    "export/export.safetensors": "9873d2f1a53127c673d0201db98cd3ba859ba432f334fef55e9e24fd057b9651",
    "export/manifest.json": "f3737a538da81939aabe04acd5334afd0e3cd4160ec113bd889e771b67991dbd",
    "export/reconstruction.json": "efa75346d366fbb92a49b1de72d569894c6338e84199271cbfc97d2f2696d234",
    "export/tokenizer/tokenizer.model": "5a3a82c48998709f9bc9f5db2924d99a899d20bf82550ca52272da7c7557b0a0",
    "export/tokenizer/tokenizer.vocab": "dbe7ec6bd066c4526080940e1614e6582b9161515e856c2c39a1d49267c88cbc",
    "export/tokenizer/vocab.txt": "70b25deed256d79365a3c290d6191c31cc2315d952faf18d5c3a334f46e4866e",
    "config.json": "3cf1478ce33e3126a288e57cb9d82b386639d5327f56067308cff0781ba8c6b8",
    "summary.json": "1dcecfcd1a08ec54094babfe58261d79c1df9c022b9b80a53eed716d20fb425a",
}
WEIGHTS_SHA256 = PINNED_FILES["export/export.safetensors"]
CONFIG_SHA256 = PINNED_FILES["config.json"]  # = summary config_sha256
FORMAT = "parakeet-binary-v1"
WEIGHTS_FILE = "export.safetensors"  # the only accepted manifest "file"
# Everything the export directory holds, files and directories (no extra or missing entries).
EXPORT_ENTRIES = frozenset({"export.safetensors", "manifest.json", "reconstruction.json", "tokenizer",
                            "tokenizer/tokenizer.model", "tokenizer/tokenizer.vocab", "tokenizer/vocab.txt"})
SOURCE_DESCRIPTION = "binary export, rebuilt with dequantized FP32 weights"
# summary.json fields that identify the final MB1 checkpoint (precheck item 2).
EXPECT_SUMMARY = {"run_name": RUN_NAME, "seed": 20261017, "lr": 2e-4, "select": "final",
                  "selected_step": 250_000, "steps": 250_000, "max_steps": 250_000, "resumes": 0,
                  "scored_on": "model rebuilt from export", "dev_limit": None}
EXPECT_BINARY = {"quantizer": "binary", "init": "m1"}
# The recorded full-dev mean (a fraction); the per-set values are summary.json["dev_wer"].
DEV_MEAN = 0.06311961686131834

# ----------------------------------------------------------------------------- reused arms
TERNARY_EVAL = binary_paths.TERNARY_ARTIFACTS / "eval"
M1_TEST = TERNARY_EVAL / "M1-test"
B0_TEST = TERNARY_EVAL / "b0-pretrained"
M1_EXPORT_SHA256 = "a287e97719c451b785be2cd01ecc861fcaa010ebaa4ff2841783ec78fcd61503"
BASE_MODEL = {"id": "nvidia/parakeet-tdt-0.6b-v2", "revision": "ae9ad07059c7c739ffaf932226a8fe64ae2620b0",
              "nemo_sha256": "d99e39955c9d3d0350d8fb7c75e40c64a2b2eaeb003883d7c941fd2e8747b28c"}

# ----------------------------------------------------------------------------- scoring
EVALUATE_SHA256 = "b3136a80558df15441b915568aca7199fbff878a392a2eed272f85fea9120a20"  # frozen evaluate.py
BATCH_SIZE = 64  # M1-test's recorded value; not a CLI option in production
DIAGNOSTIC_DEV_BATCH = 32  # --dev-check --dev-batch-size 32 only (train.DevEvaluator's batch)
MAX_BATCH_SECONDS = 1200.0
DECODER_GRAPHS_MODE = binary_paths.DECODER_GRAPHS_MODE

# Dev cross-check tolerances: plan values in WER points, compared as fractions.
MEAN_TOLERANCE_POINTS = 0.02
SET_TOLERANCE_POINTS = 0.10
MEAN_TOLERANCE = 0.0002
SET_TOLERANCE = 0.0010


def points_to_fraction(points: float) -> float:
    """WER points (percent) -> fraction."""
    return points / 100.0


# ----------------------------------------------------------------------------- outputs (fixed)
EVAL = binary_paths.EVAL
PRECHECK = EVAL / "MB1-precheck.json"
DEV_CHECK = EVAL / "MB1-dev-check"
DEV_CHECK_B32 = EVAL / "MB1-dev-check-b32"
TEST_OUT = EVAL / "MB1-test"
ATTEMPTS = EVAL / "MB1-test.attempts.jsonl"
# Serializes --test, --record-failure and --summarize. Lock order: this lock first, then the shared GPU
# lock (paths.gpu_lock), never the reverse; nothing else takes this lock (code review round 1).
ATTEMPTS_LOCK = EVAL / "MB1-test.attempts.lock"
RESULTS = HERE / "results"  # TEST.md, test.json
GATE = "gate.json"


def smoke_dir(limit: int, batch_size: int = BATCH_SIZE) -> Path:
    """EVAL/smoke-MB1-dev-limit<N> (suffix -b32 for the diagnostic batch)."""
    suffix = "" if batch_size == BATCH_SIZE else f"-b{batch_size}"
    return EVAL / f"smoke-MB1-dev-limit{int(limit)}{suffix}"


# ----------------------------------------------------------------------------- training unit
MAIN_UNIT = "parakeet-binary-binary-main.service"
MAIN_LOG = binary_paths.LOGS / "binary-main.log"

# ----------------------------------------------------------------------------- scorer provenance
# The Transformers Whisper normalizer evaluate.py imports (installed path, recorded 2026-10-20).
NORMALIZER_MODULE = Path("/mnt/hd/wilderness-labs-stt/stt-distillation/deps/transformers/models/whisper/"
                         "english_normalizer.py")
NORMALIZER_JSON = "stt/models/whisper-tiny.en/normalizer.json"  # evaluate.NORMALIZER_JSON

# Keys are relative to finetune/ unless absolute (scorer_path). Pinned after code review.
SCORER_HASHES = {
    "parakeet-binary/binary_test.py": "5089cd4b4b9a40ec14d4d2c711efb02bc31ae481b37bdaad3be00a88561381e8",
    "parakeet-binary/binary_report_test.py": "6f6d396859a9e8ceae02b8170da2e0cacf37737bde33e450c291e755f07c8731",
    "parakeet-binary/binary_export.py": "490f185b725a95bfcdbcbc871e68d1cd44970ed0a1c9141f2d4031382f6c83cb",
    "parakeet-binary/binary_train.py": "77a0012288e84b3596c0d892c2888f2a1ec9afd1e639d53374086169f921166e",
    "parakeet-binary/binary_paths.py": "f6c95c2914950c6477c83620097c7724df3b7d5c5251bd76dacf67407c253d79",
    "parakeet-binary/binary_quant.py": "3da0acbb3b64fc4ddb3a14fbee45ccec352274502f85891c857d80541720f6d5",
    "parakeet-ternary/evaluate.py": "b3136a80558df15441b915568aca7199fbff878a392a2eed272f85fea9120a20",
    "parakeet-ternary/export.py": "1cfeb049192d81b8cb993e95e603234f5fbe5dd7829af45693f5e28ae39d5065",
    "parakeet-ternary/quant.py": "b60a6118b8771b479a4356d380de9722a2a5d833dba5d2431911a5e18aa7d66b",
    "parakeet-ternary/paths.py": "8ee2ed9298d434907cda9f055ef67282fd42ef73eae49c2cf0111c250cb33fa2",
    "parakeet-ternary/train.py": "d44996b1b2c313ac8baf04228a1b8873ff0df12c99da9708459191ea7cf3c853",
    "whisper-ternary/wer.py": "f1ced52d0b5b0a738cc7e3e494bae3b634ee0f0edb5f521e2974a900f5e41ae4",
    str(NORMALIZER_MODULE): "e0fc3dc9373209dbecd5881dab6a9834f76cf21d8031914c8f4fa1c3a1288461",
    NORMALIZER_JSON: "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd",
}
# Imported module -> its SCORER_HASHES key: the module's __file__ must be that exact file.
SCORER_MODULES = {
    "binary_export": "parakeet-binary/binary_export.py",
    "binary_train": "parakeet-binary/binary_train.py",
    "binary_paths": "parakeet-binary/binary_paths.py",
    "binary_quant": "parakeet-binary/binary_quant.py",
    "evaluate": "parakeet-ternary/evaluate.py",
    "export": "parakeet-ternary/export.py",
    "quant": "parakeet-ternary/quant.py",
    "paths": "parakeet-ternary/paths.py",
    "train": "parakeet-ternary/train.py",
    "whisper_ternary_wer": "whisper-ternary/wer.py",  # quant.load_whisper_module("wer")
    "transformers.models.whisper.english_normalizer": str(NORMALIZER_MODULE),
}


def scorer_path(key: str) -> Path:
    """The file a SCORER_HASHES key names."""
    path = Path(key)
    return path if path.is_absolute() else FINETUNE / key
