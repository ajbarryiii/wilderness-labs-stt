"""The synthetic clip source of the TS / TSn arms (synthetic plan "Mixing"), for training/replay.py's `extra`.

ClipPool.load(run): the kept clips of a generation run (runs/<run>/clips.jsonl, checked against clips.meta.json;
a provisional run, whose hygiene lacked eval-ext or eval-stem, is refused for training).

plan(real_seconds, clip_seconds, seed, share): the fixed, seeded assignment of synthetic clips to consumed batches.
- Clip order: a seeded permutation of the pool (pass 0), then a new seeded permutation for each further pass, so
  clips repeat only after the whole pool was used.
- Batch b's target is share x its REAL seconds plus the carry from earlier batches (error diffusion); clips are
  taken in order while adding the next one brings the batch's synthetic seconds closer to the target (adding stops
  at the first clip that would overshoot by more than it helps); the carry is target minus what was taken. So
  every batch gets about `share` of its real audio (within half a clip) and the running total stays within half a
  clip of `share` x the real seconds so far.
- REAL seconds count only the real rows that are trained on (keep(id): the label store has a label for the id);
  a batch with no kept real row gets no synthetic clip (it stays a consumed batch without a step, as in stage 2b).
The plan depends only on (the real batches' ids, durations and keep mask, the pool, seed, share): the TS and TSn
arms built from the same inputs get identical clips in identical batches, and a resumed run gets the same plan.

Since 2026-10-24 (synthetic plan "Mixing") the synthetic arms train on a version 2 batch plan
(training/batchplan.py compute_synthetic), whose draw is plan() over pools and whose clips are cut together with
the real rows; SyntheticSource (clips appended after the cut) is kept for version 1 replays only and is not used
by the synthetic arms. Their clips come from the version 2 replay itself (training/replay.py PlannedReplay with
`synthetic` = the ClipPool), and transform_for(arm) is its `synthetic_transform`: the augmentation for TS, None
for TSn (train_synthetic.py).

SyntheticSource(pool, real_batches, keep, seed, share, augment): the `extra` object of training/replay.py:
- items(index, real_items) -> the batch's synthetic items {"audio" float32, "duration", "id": "syn:<clip id>",
  "source": "synthetic", "label"}; the real items are only read (their ids and durations must equal the plan's,
  else it raises), never changed, and come first in the batch (the replay batcher appends these after them);
- transform: the augmentation (augment.Augmenter) for TS, None for TSn (no transform: audio as stored);
- identity: a dict naming the pool, plan and augmentation, recorded in the trainer's data state.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from synthetic import layout
from synthetic.textgen import read_jsonl

SOURCE = "synthetic"


class ClipPool:
    def __init__(self, clips: list[dict], identity: str, verify_audio: bool = True):
        self.clips, self.identity, self.verify_audio = clips, identity, verify_audio
        self.by_id = {c["id"]: c for c in clips}

    @classmethod
    def load(cls, run: str | Path, allow_provisional: bool = False, verify_audio: bool = True) -> "ClipPool":
        d = layout.run_dir(run) if isinstance(run, str) else Path(run)
        meta = json.loads((d / "clips.meta.json").read_text())
        data = (d / "clips.jsonl").read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if sha != meta["clips_sha256"]:
            raise RuntimeError(f"{d}/clips.jsonl does not match clips.meta.json")
        if meta.get("provisional") and not allow_provisional:
            raise RuntimeError(f"{d}: provisional run (hygiene without eval-ext/eval-stem); not usable for training")
        current = hashlib.sha256(layout.split_path().read_bytes()).hexdigest() if layout.split_path().exists() else None
        if (meta.get("hygiene_split_sha256") is None or meta["hygiene_split_sha256"] != current) and not allow_provisional:
            raise RuntimeError(f"{d}: hygiene did not run under the current term split (split_sha256 "
                               f"{meta.get('hygiene_split_sha256')} vs {current}); rerun hygiene and finalize")
        return cls(read_jsonl(d / "clips.jsonl"), identity=sha, verify_audio=verify_audio)

    def read(self, clip_id: str) -> np.ndarray:
        import soundfile as sf
        c = self.by_id[clip_id]
        raw = Path(c["audio_filepath"]).read_bytes()
        if self.verify_audio and hashlib.sha256(raw).hexdigest() != c["sha256"]:
            raise RuntimeError(f"{c['audio_filepath']}: SHA-256 differs from clips.jsonl")
        import io
        a, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        if rate != layout.SAMPLE_RATE:
            raise RuntimeError(f"{c['audio_filepath']}: not 16 kHz")
        return a[:, 0]


def clip_order(n: int, seed: int, passes: int, first: int = 0) -> list[int]:
    """Pass `first`, `first`+1, ...: a seeded permutation of range(n) each."""
    out = []
    for p in range(first, first + passes):
        h = hashlib.sha256(f"{seed}|clip-order|{p}".encode()).digest()
        out += np.random.default_rng(int.from_bytes(h[:8], "big")).permutation(n).tolist()
    return out


def plan(real_seconds: Sequence[float], clip_seconds: Sequence[float], seed: int,
         share: float = layout.SYNTHETIC_SHARE) -> list[list[int]]:
    """Per batch, the indices (into clip_seconds) of its synthetic clips (module docstring)."""
    n = len(clip_seconds)
    if n == 0:
        raise ValueError("empty clip pool")
    need = share * float(sum(real_seconds))
    passes = int(need / max(sum(clip_seconds), 1e-9)) + 2
    order = clip_order(n, seed, passes)
    cursor, carry, out = 0, 0.0, []
    for real in real_seconds:
        take: list[int] = []
        if real > 0:
            target = share * real + carry
            got = 0.0
            while True:
                if cursor >= len(order):
                    order += clip_order(n, seed, 1, first=len(order) // n)
                d = clip_seconds[order[cursor]]
                if got + d / 2.0 > target:      # adding would overshoot by more than it closes
                    break
                take.append(order[cursor])
                got += d
                cursor += 1
            carry = target - got
        out.append(take)
    return out


class SyntheticSource:
    def __init__(self, pool: ClipPool, real_batches: Sequence[Sequence[tuple[str, float]]],
                 keep: Callable[[str], bool] | None = None, seed: int = layout.SEED,
                 share: float = layout.SYNTHETIC_SHARE, augment=None):
        self.pool, self.seed, self.share = pool, int(seed), float(share)
        keep = keep or (lambda _id: True)
        self.real = [[(str(i), float(d)) for i, d in b] for b in real_batches]
        self.real_seconds = [sum(d for i, d in b if keep(i)) for b in self.real]
        self.clips = sorted(pool.clips, key=lambda c: c["id"])
        self.batches = plan(self.real_seconds, [c["duration"] for c in self.clips], self.seed, self.share)
        self.transform = augment
        h = hashlib.sha256(json.dumps(self.batches).encode()).hexdigest()
        self.identity = {"pool": pool.identity, "plan_sha256": h, "seed": self.seed, "share": self.share,
                         "augment": None if augment is None else getattr(augment, "identity", repr(type(augment)))}

    def clip_ids(self, index: int) -> list[str]:
        return [self.clips[k]["id"] for k in self.batches[index]]

    def synthetic_seconds(self, index: int) -> float:
        return float(sum(self.clips[k]["duration"] for k in self.batches[index]))

    def items(self, index: int, real_items: Sequence[dict]) -> list[dict]:
        if not 0 <= index < len(self.batches):
            raise IndexError(f"batch {index} outside the synthetic plan's {len(self.batches)} batches")
        got = [(str(it["id"]), round(float(it["duration"]), 6)) for it in real_items]
        want = [(i, round(d, 6)) for i, d in self.real[index]]
        if got != want:
            raise RuntimeError(f"batch {index}: the real rows differ from the ones the synthetic plan was built on")
        out = []
        for k in self.batches[index]:
            c = self.clips[k]
            audio = self.pool.read(c["id"])
            out.append({"audio": audio, "duration": len(audio) / layout.SAMPLE_RATE, "id": f"syn:{c['id']}",
                        "source": SOURCE, "label": c["label"]})
        return out


ARMS = ("TS", "TSn")


def transform_for(arm: str, seed: int = layout.SEED, augment_pool=None):
    """The synthetic rows' transform of arm "TS" (augment.Augmenter over the TRAIN noise/RIR pool, seeded) or
    "TSn" (None: the clips as stored). PlannedReplay refuses a transform that changes a label or makes a clip longer
    than its planned key."""
    from synthetic import augment
    if arm not in ARMS:
        raise ValueError(f"unknown synthetic arm {arm!r}; expected one of {ARMS}")
    if arm == "TSn":
        return None
    return augment.Augmenter(augment_pool or augment.Pool.load(), seed=seed)


def source_for(arm: str, replay, labels, stop: int, run: str = "full", seed: int = layout.SEED,
               share: float = layout.SYNTHETIC_SHARE, pool: ClipPool | None = None, augment_pool=None) -> SyntheticSource:
    """The `extra` object of a stage 2b replay for arm "TS" (augmented) or "TSn" (unaugmented).

    replay: training.replay.PlannedReplay (its .batches give each planned batch's (id, source, duration), in cut
    order; 2026-10-24) or JournalReplay (consumed batches);
    labels: the arm's label store (training.labels.LabelStore, T*); a real row counts as trained iff
    labels.text(id) is not None; stop: the run's stop batch (the plan covers batches [0, stop) of the replay).
    Both arms built from the same replay, labels, stop, run, seed and share get identical clips in identical batches.
    """
    from synthetic import augment
    if arm not in ARMS:
        raise ValueError(f"unknown synthetic arm {arm!r}; expected one of {ARMS}")
    pool = pool or ClipPool.load(run)
    real = [[(i, d) for i, _, d in replay.batches[b]] for b in range(stop)]
    aug = None
    if arm == "TS":
        aug = augment.Augmenter(augment_pool or augment.Pool.load(), seed=seed)
    return SyntheticSource(pool, real, keep=lambda i: labels.text(i) is not None, seed=seed, share=share, augment=aug)
