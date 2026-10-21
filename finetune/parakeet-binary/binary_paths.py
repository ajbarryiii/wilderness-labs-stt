"""Fixed locations and pinned inputs of the binary Parakeet pilot. See plans/pilot.md.

Everything the pilot writes lives under ARTIFACTS (/mnt/hd/wilderness-labs-stt/parakeet-binary).
The parakeet-ternary artifacts (pretrained model, dev manifests and audio, the M1 run, the
ternary pilot runs, Hugging Face caches) are read, never written.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import paths

HERE = Path(__file__).resolve().parent
ARTIFACTS = Path("/mnt/hd/wilderness-labs-stt/parakeet-binary")
RUNS = ARTIFACTS / "runs"
EVAL = ARTIFACTS / "eval"
POWER = ARTIFACTS / "power"
LOGS = ARTIFACTS / "logs"
TMP = ARTIFACTS / "tmp"
CACHES = ARTIFACTS / "caches"

# parakeet-ternary, read only (paths.MODEL_DIR is a constant; paths.ARTIFACTS is patched while
# binary_train runs, so it is not used here).
TERNARY_ARTIFACTS = paths.MODEL_DIR.parents[1]
TERNARY_RUNS = TERNARY_ARTIFACTS / "runs"
M1_RUN = TERNARY_RUNS / "main-M1-P2-lr5e-4"
M1_CHECKPOINT = M1_RUN / "ckpt" / "latest.pt"
M1_SHA256 = "003b1289b19114b58327bebb61af598412bdaca408db47a6949f7951d62d212a"
M1_STEP = 250_000
TERNARY_PILOT_RUN = TERNARY_RUNS / "pilot-P2-lr5e-4"  # ternary P2 at 5e-4, the cost reference for arm A

# The Hub stream's worker count changes its item order (stream.py partitions files by worker).
# The ternary pilot ran with 6 workers, M1 with 8 (DESIGN.md "Main run"); stream.py's default is
# now 8. The binary pilot uses the pilot's 6 so that arm A sees the ternary pilot's data, and
# checks the stream's configuration hash against the one those runs recorded.
STREAM_WORKERS = {"pilot": 6, "main": 8}
STREAM_CONFIG_HASH = {"pilot": "874bf7e4502f36eafe31d7f8663b912be25091a89a6f9e910a6edb21788f8217",
                      "main": "519aeeb581ef1161d54a28b7494b6bf0cb0a0c61e8234f733db07410f543ba00"}

# NeMo's greedy TDT decoder CUDA-graph mode (label-looping computer, CudaGraphsMode) that M1 and
# the ternary pilot decoded with: NeMo picked "full_graph", whose compilation failed without NVRTC
# ("Falling back to native PyTorch CUDA graphs", M1's log), leaving "no_while_loops" for every
# decode. binary_train pins it on every model it decodes with and refuses any other.
DECODER_GRAPHS_MODE = "no_while_loops"

# Provenance (here rather than in the modules they describe, so binary_sweep.py needs no torch).
# Binary sources whose change can alter what an arm trains or how it is scored, relative to
# finetune/; train.TRAINING_SOURCES is extended with them. The interpreter wrapper is one: it
# sets the library path, which decides NeMo's decoder backend (review round 1, finding 3), and
# so is ./heavy, which sets the job's limits. binary_eval.py (scoring of the zero-step points,
# which records its own hash) and binary_sweep.py (orchestration) are not.
BINARY_SOURCES = ("parakeet-binary/binary_train.py", "parakeet-binary/binary_quant.py",
                  "parakeet-binary/binary_export.py", "parakeet-binary/binary_paths.py",
                  "parakeet-binary/python", "parakeet-binary/heavy")
# Sources whose change alters a quantizer's forward, relative to finetune/ (quantizer_sha256).
QUANTIZER_SOURCES = {"binary": ("parakeet-binary/binary_quant.py", "parakeet-ternary/quant.py",
                                "whisper-ternary/quant.py"),
                     "ternary": ("parakeet-ternary/quant.py", "whisper-ternary/quant.py")}


def require_mount() -> None:
    paths.require_mount()


def _named_dir(root: Path, name: str) -> Path:
    if not paths._NAME.fullmatch(name) or name in {".", ".."}:
        raise ValueError(f"invalid name {name!r}")
    require_mount()
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dir(name: str) -> Path:
    """RUNS/name, created; the name may not contain path separators."""
    return _named_dir(RUNS, name)


def eval_dir(name: str) -> Path:
    """EVAL/name, created."""
    return _named_dir(EVAL, name)


def sha256_file(path: Path) -> str:
    """Streaming SHA-256; the file's pages are dropped from the page cache afterwards (multi-GB
    reads otherwise stay charged to the job's memory-capped unit)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
        try:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass
    return digest.hexdigest()


def quantizer_sha256(kind: str) -> str:
    """One SHA-256 over the sources of the "binary" or "ternary" quantizer (QUANTIZER_SOURCES):
    part of a run's resume key, so a quantizer change never resumes an old checkpoint."""
    if kind not in QUANTIZER_SOURCES:
        raise ValueError(f"unknown quantizer {kind!r}")
    finetune = paths.REPO / "finetune"
    lines = [f"{name} {hashlib.sha256((finetune / name).read_bytes()).hexdigest()}\n"
             for name in QUANTIZER_SOURCES[kind]]
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def init_sha256(init: str) -> str:
    """SHA-256 of an arm's initialization. "fp32": the pinned .nemo's, from lock.json
    (train.verify_base checks the file against it before training); "m1": the M1 checkpoint's,
    which must equal M1_SHA256 (ValueError otherwise)."""
    if init == "fp32":
        lock = json.loads((paths.MODEL_DIR / "lock.json").read_text())
        return lock["files"][paths.MODEL_FILE.name]
    if init != "m1":
        raise ValueError(f"unknown init {init!r}")
    digest = cached_sha256(M1_CHECKPOINT)
    if digest != M1_SHA256:
        raise ValueError(f"{M1_CHECKPOINT} has SHA-256 {digest}, not the pinned M1 {M1_SHA256}; "
                         "refusing to initialize from it")
    return digest


def cached_sha256(path: Path) -> str:
    """SHA-256 of a large file, cached in CACHES/sha256/ under the file's identity.

    The cache entry is keyed by the resolved path, size, mtime_ns, inode and device: any rewrite
    of the file (which changes mtime) is hashed again. Callers compare the result with a pinned
    value and refuse on mismatch; the cache only avoids re-reading 7 GB on every start.
    """
    path = Path(path).resolve()
    st = path.stat()
    identity = {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                "inode": st.st_ino, "device": st.st_dev}
    entry = CACHES / "sha256" / f"{hashlib.sha256(str(path).encode()).hexdigest()[:32]}.json"
    try:
        cached = json.loads(entry.read_text())
        if {k: cached.get(k) for k in identity} == identity and len(cached.get("sha256", "")) == 64:
            return cached["sha256"]
    except (OSError, ValueError):
        pass
    digest = sha256_file(path)
    if path.stat().st_mtime_ns == st.st_mtime_ns:  # unchanged while hashing
        require_mount()
        entry.parent.mkdir(parents=True, exist_ok=True)
        tmp = entry.with_name(entry.name + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps({**identity, "sha256": digest,
                                   "hashed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
        os.replace(tmp, entry)
    return digest
