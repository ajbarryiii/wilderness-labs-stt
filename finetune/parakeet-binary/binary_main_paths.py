"""Fixed values of the binary main run MB1 (plans/main.md), without torch so binary_sweep.py can
import them.

They live here, not in binary_paths.py, because the finished pilot arms and the G-B gate
recorded the SHA-256 of binary_paths.py (and of every other BINARY_SOURCES file); changing it
would make the pilot unreusable (plans/main.md "Implementation notes (2026-10-17)", item 1).
"""
from __future__ import annotations

import binary_paths
import paths

RUN_NAME = "main-binB-m1-lr2e-4"
LR = "2e-4"  # pilot winner B2
STEPS = 250_000  # user decision 2026-10-17
EVAL_EVERY = 12_500  # 20 dev-subset points, as M1
SEED = 20261017  # user decision 2026-10-17 ("Data order"); not M1's paths.SEED

# Hub stream pins, keyed by (stream phase, seed): (worker count, configuration hash). The hash
# covers the file lists, mixture, filters, exclusions, seed and worker count (stream.StreamConfig
# .digest); a stream with any other hash is refused. The pilot's pin is binary_paths' (the
# ternary pilot's recorded hash). The MB1 hash was computed offline on 2026-10-17 (plans/main.md
# "Implementation notes", item 2). M1's own order (main, paths.SEED) is deliberately not pinned.
STREAM_PINS = {
    ("pilot", paths.SEED): (binary_paths.STREAM_WORKERS["pilot"], binary_paths.STREAM_CONFIG_HASH["pilot"]),
    ("main", SEED): (8, "bad909114fbe3e50505434635c49406f751f0673663e275b460ebdb82612badf"),
}

# stream.experiment_config rewrites these parakeet-ternary records (with the seed) on every open;
# binary_main redirects that write to STREAM_RECORD_ROOT/seed-<seed>/stream/ and refuses to train
# if they change anyway.
TERNARY_STREAM_RECORDS = (paths.DATA / "stream" / "sources.json", paths.DATA / "stream" / "yodas_train_files.json")
STREAM_RECORD_ROOT = binary_paths.ARTIFACTS / "stream-records"

# MB1's training sources: the pilot's plus the two main-run files (recorded by train.py under
# binary_main.seam and checked by the sweep for MB1 only).
MAIN_FILES = ("parakeet-binary/binary_main.py", "parakeet-binary/binary_main_paths.py")
MAIN_SOURCES = binary_paths.BINARY_SOURCES + MAIN_FILES


def stream_pin(phase: str, seed: int) -> tuple[int, str]:
    """(workers, configuration hash) pinned for this stream phase and seed; ValueError otherwise."""
    try:
        return STREAM_PINS[(phase, seed)]
    except KeyError:
        pinned = ", ".join(f"--train {p} --seed {s}" for p, s in STREAM_PINS)
        raise ValueError(f"--train {phase} --seed {seed} is not a pinned stream (pinned: {pinned}); "
                         "refusing to train on an unchecked data order") from None


def record_dir(seed: int):
    """Where binary_main writes the stream record (stream.record_sources' paths.DATA) for a seed."""
    return STREAM_RECORD_ROOT / f"seed-{int(seed)}"
