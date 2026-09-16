"""Replayable recovery experiment contracts; no model or decoder dependencies."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from common import ART, digest, save, storage

ROOT = ART / "recovery-and-decoding"
SOURCE_RUN = ART / "runs/pilot-4h-20260910T180602Z"
DOMAINS = ["general", "medical_symptoms", "digits"]


def artifact(path):
    storage()
    path = Path(path).resolve()
    if not path.is_relative_to(ART.resolve()):
        raise ValueError(f"Artifact outside mounted project storage: {path}")
    return path


def read(path):
    return json.loads(Path(path).read_text())


def lr_factor(presented, horizon=108000, warmup=800):
    return min(1.0, presented / warmup) * (
        0.2 + 0.4 * (1 + math.cos(math.pi * min(presented, horizon) / horizon))
    )


def tape(rows, seed, count, probabilities=(0.5, 0.3, 0.2), gate=False):
    """Same sampling/augmentation RNG order as the historical worker.

    Columns: manifest row index, prefix frames, original gain index. Storing
    the draws makes batch accumulation and resume independent of RNG replay.
    """
    rng = np.random.default_rng(seed)
    aug = np.random.default_rng(seed + 171)
    groups = [[i for i, r in enumerate(rows) if r["domain"] == d] for d in DOMAINS]
    pending = []
    result = np.empty((count, 3), dtype=np.int32)
    for j in range(count):
        if gate:
            if not pending:
                pending = rng.permutation(len(rows)).tolist()
            idx = pending.pop(0)
        else:
            domain = int(rng.choice(3, p=probabilities))
            idx = groups[domain][int(rng.integers(len(groups[domain])))]
        if aug.random() < 0.25:
            prefix, gain_index = (20 if rows[idx]["domain"] == "digits" else 0), 0
        else:
            prefix, gain_index = int(aug.integers(0, 101)), int(aug.choice(5))
        result[j] = idx, prefix, gain_index
    return result


def exposure(rows, draws, gains):
    sample, augmentation = hashlib.sha256(), hashlib.sha256()
    counts = {d: 0 for d in DOMAINS}
    hours = {d: 0.0 for d in DOMAINS}
    unique = set()
    for idx, prefix, gain_index in draws:
        row = rows[int(idx)]
        sample.update((row["id"] + "\n").encode())
        augmentation.update(
            f"{row['id']} {int(prefix)} {gains[int(gain_index)]}\n".encode()
        )
        counts[row["domain"]] += 1
        hours[row["domain"]] += row["seconds"] / 3600
        unique.add(row["id"])
    return dict(
        sample_hash=sample.hexdigest(),
        augmentation_hash=augmentation.hexdigest(),
        domain_presentations=counts,
        exposure_hours=hours,
        unique_recordings=len(unique),
    )


def selection(metric):
    return (metric["general"]["cer"] + metric["medical_symptoms"]["cer"]) / 2


def source_hashes(here):
    return {
        p.name: digest(p)
        for p in Path(here).iterdir()
        if p.is_file()
        and (
            p.suffix in {".py", ".json", ".md", ".txt"}
            or p.name in {"python", "decoder-python"}
        )
    }


def nested_subsets(rows, gate_ids, seed):
    rng = np.random.default_rng(seed + 401)
    chosen = list(gate_ids)
    out = {}
    # Round-robin duration bins and shuffled speaker groups avoid a shortest-clip gate.
    for size, counts in [(256, (128, 77, 51)), (1024, (512, 307, 205))]:
        for domain, target in zip(DOMAINS, counts):
            current = sum(r["id"] in chosen and r["domain"] == domain for r in rows)
            remaining = sorted(
                [r for r in rows if r["domain"] == domain and r["id"] not in chosen],
                key=lambda r: (r["seconds"], r["id"]),
            )
            bins = [
                list(x) for x in np.array_split(np.array(remaining, dtype=object), 4)
            ]
            for bucket in bins:
                rng.shuffle(bucket)
            while current < target:
                for bucket in bins:
                    if bucket and current < target:
                        chosen.append(bucket.pop()["id"])
                        current += 1
        out[str(size)] = list(chosen)
    return out


def write_tape(run, rows, seed, count, label="broad", gate=False):
    path = artifact(Path(run) / "tapes" / f"{label}-{seed}.npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    draws = tape(rows, seed, count, gate=gate)
    np.save(path, draws, allow_pickle=False)
    save(
        path.with_suffix(".json"),
        dict(
            seed=seed,
            count=count,
            sha256=digest(path),
            row_ids=[r["id"] for r in rows],
            gate=gate,
        ),
    )
    return path
