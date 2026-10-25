"""Roots, plan constants and artifact locations of Part C (labeling plan "Part C: the labeling pass").

Roots come from the environment (./python fills it from private/local.env). Everything Part C writes is under
$RELABEL_ROOT:

    labeling/config/candidate.json            the stream configuration without W (input to the replay probe)
    labeling/config/pinned-w<W>.json          the pinned configuration (with W and its hash); exactly one
    pool-audio/<config id>/                   C1: shards/<k>.audio.parquet, <k>.journal.parquet, <k>.json,
                                              status.json, manifest.json
    labels/<teacher>/<run>/                   C2: run.json, work/{teacher,b0}/<group>.*, shards/<k>.labels.parquet
                                              and <k>.json, groups/<g>.json, manifest.json, report.json
    labeling/probes/                          replay probe (per W), GPU probe (per engine and setting)
    labeling/equivalence/<run>/               the equivalence check
    labeling/consistency/<run>/               the consistency gate (stage 1 dev ids, no filters)
    labeling/gate/<run>/                      the gate to stage 3 (drops, completeness, spot check)
    smoke/labeling/...                        smoke runs (tiny fixtures, --limit); never a production path

The private registry ($PRIVATE_REGISTRY/sources.json) maps aliases to sources; Part B's outputs are read
from $RELABEL_ROOT/pool/ and $RELABEL_ROOT/corpora/.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- plan constants (labeling plan Part C) ---------------------------------------------------------
SEED = 20261022                 # the new stream's seed (stream and batch order)
BATCH_SECONDS = 600.0           # the main runs' padded batch seconds
TOTAL_BATCHES = 100_000         # labels cover the first 100,000 consumed batches
SHARD_BATCHES = 50              # C1 shard = 50 consumed batches (~6-7 audio hours of occurrences)
WORKER_CANDIDATES = (8, 12, 16)  # replay probe
HOST_PEAK_LIMIT_GB = 30.0       # replay probe and training smoke: host peak at most 30G (under the 36G cap)
FASTER_BY = 0.10                # 12 or 16 replaces the candidate only if at least 10% faster
REPLAY_PROBE_MINUTES = 15.0
GPU_PROBE_MINUTES = 10.0
CHUNK_MINUTES = 110.0           # chunk jobs of at most about 2 hours
INDEPENDENT_REPLAY_BATCHES = 1_000
SPOT_CHECK_PER_SOURCE = 100
EQUIVALENCE_PER_SOURCE = 300
EQUIVALENCE_SEED = 20261023
SPOT_CHECK_SEED = 20261024
VRAM_MARGIN_GB = 2.0            # declared VRAM = probe peak + 2 GB

# Shares of audio hours (labeling plan Part C "Pool and stream"). The existing five sources share the
# remainder in their existing relative shares (finetune/parakeet-ternary/stream.py SHARES). A local corpus
# that Part B dropped returns its share to the existing mix.
EXISTING_SHARE = 0.79
DOMAIN_SHARES = {"medical": 0.10, "yodas-tech": 0.07}   # medical = yodas-med + ps-med by their hours
LOCAL_SHARES = {"syn-dialog": 0.03, "med-osce": 0.01}
MEDICAL_SUBSETS = ("yodas-med", "ps-med")
SUBSET_PARENT = {"yodas-med": "yodas", "ps-med": "peoples_speech", "yodas-tech": "yodas"}

TEACHER = "granite-speech-5.0-2b"    # T* (diagnostic results)


def root(key: str) -> Path:
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(f"{key} is not set (environment or finetune/relabel/private/local.env; "
                           "see finetune/relabel/local.env.example)")
    return Path(value)


def require_mount() -> None:
    mount = root("DATA_MOUNT")
    if not os.path.ismount(mount):
        raise RuntimeError(f"{mount} is not mounted; refusing artifact writes.")
    if not root("RELABEL_ROOT").is_relative_to(mount):
        raise RuntimeError("RELABEL_ROOT is not on DATA_MOUNT")


def relabel_root() -> Path:
    return root("RELABEL_ROOT")


def labeling_dir() -> Path:
    return relabel_root() / "labeling"


def config_dir() -> Path:
    return labeling_dir() / "config"


def candidate_path() -> Path:
    return config_dir() / "candidate.json"


def pinned_path(workers: int) -> Path:
    return config_dir() / f"pinned-w{int(workers)}.json"


def pool_dir() -> Path:
    return relabel_root() / "pool"


def corpora_dir() -> Path:
    return relabel_root() / "corpora"


def pool_audio_dir(config_id: str) -> Path:
    return relabel_root() / "pool-audio" / config_id


def labels_dir(teacher: str, run: str) -> Path:
    return relabel_root() / "labels" / teacher / run


def probes_dir() -> Path:
    return labeling_dir() / "probes"


def equivalence_dir(run: str) -> Path:
    return labeling_dir() / "equivalence" / run


def consistency_dir(run: str) -> Path:
    return labeling_dir() / "consistency" / run


def gate_dir(run: str) -> Path:
    return labeling_dir() / "gate" / run


def smoke_dir(name: str) -> Path:
    return relabel_root() / "smoke" / "labeling" / name


def is_production(path: Path) -> bool:
    """True for any path under $RELABEL_ROOT that is not under smoke/."""
    path = Path(path).resolve()
    rr = relabel_root().resolve()
    return path.is_relative_to(rr) and not path.is_relative_to(rr / "smoke")
