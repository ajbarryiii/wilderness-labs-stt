"""LibriSpeech manifests, download provenance, and the Whisper feature/label dataset.

Manifests must list exactly the published utterance count per split. Features
are the checkpoint's log-mel extractor padded or truncated to 30 s; labels are
the checkpoint tokenizer applied to the lowercased transcript (no other text
edits) with the leading decoder-start token removed. Nothing here resamples,
augments or filters utterances; clips over 30 s are flagged, not dropped.
Training may concatenate several training splits (DESIGN.md "Revision 2");
dev-* and test-* splits are never accepted as training data.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers.models.whisper.modeling_whisper import shift_tokens_right

import paths

URL = "https://www.openslr.org/resources/12/{split}.tar.gz"
LICENSE = "CC BY 4.0 (OpenSLR SLR12, Panayotov et al. 2015)"
MAX_SECONDS = 30.0


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    tmp.replace(path)


def scan(root: Path) -> list[dict]:
    """One record per transcript line under a LibriSpeech split root, sorted by id."""
    records = []
    for trans in sorted(Path(root).glob("*/*/*.trans.txt")):
        for line in trans.read_text().splitlines():
            if not line.strip():
                continue
            uid, text = line.split(" ", 1)
            flac = (trans.parent / f"{uid}.flac").absolute()
            info = sf.info(str(flac))
            if info.samplerate != paths.SAMPLE_RATE or info.channels != 1:
                raise ValueError(f"{flac}: expected {paths.SAMPLE_RATE} Hz mono, got "
                                 f"{info.samplerate} Hz x{info.channels}")
            records.append({"id": uid, "path": str(flac), "text": text.strip(),
                            "duration_s": info.frames / info.samplerate})
    return sorted(records, key=lambda r: r["id"])


def build_manifest(split: str, refresh: bool = False) -> list[dict]:
    """Cached, count-checked manifest for a split; building it also refreshes provenance.json."""
    path = paths.MANIFESTS / f"{split}.json"
    if path.exists() and not refresh:
        records = json.loads(path.read_text())
    else:
        paths.require_mount()
        records = scan(paths.SPLITS[split])
    expected = paths.EXPECTED_UTTERANCES[split]
    if len(records) != expected:
        raise ValueError(f"{split}: found {len(records)} utterances under {paths.SPLITS[split]}, "
                         f"expected {expected}; is the split fully extracted?")
    if refresh or not path.exists():
        write_json(path, records)
        record_provenance()
    return records


def check_training_splits(splits: Sequence[str]) -> list[str]:
    """The split names as a list; ValueError if empty, duplicated, unknown, or not a train-* split."""
    if isinstance(splits, str):
        raise TypeError("pass a sequence of split names, not one string")
    splits = list(splits)
    if not splits or any(not s for s in splits):
        raise ValueError(f"training splits must be non-empty names, got {splits}")
    duplicates = sorted({s for s in splits if splits.count(s) > 1})
    if duplicates:
        raise ValueError(f"duplicate training splits: {duplicates}")
    held_out = [s for s in splits if not s.startswith("train-")]
    if held_out:
        raise ValueError(f"not training splits: {held_out}; dev-* and test-* are never trained on")
    unknown = [s for s in splits if s not in paths.SPLITS or s not in paths.EXPECTED_UTTERANCES]
    if unknown:
        raise ValueError(f"unknown training splits: {unknown}; known: "
                         f"{sorted(s for s in paths.SPLITS if s.startswith('train-'))}")
    return splits


def build_training_manifest(splits: Sequence[str]) -> list[dict]:
    """Concatenated count-verified manifests of the given training splits, in the given order.

    Each split's records are sorted by id and tagged with "split". A single
    split yields exactly build_manifest(split) plus that tag, so the default
    train-clean-100 run sees the same records in the same order as v1.
    """
    manifest: list[dict] = []
    for split in check_training_splits(splits):
        records = sorted(build_manifest(split), key=lambda r: r["id"])
        manifest.extend({**r, "split": split} for r in records)
    return manifest


def record_provenance() -> dict:
    """Write DATA/provenance.json from download.log plus the manifests present."""
    log = paths.DATA / "download.log"
    splits: dict[str, dict] = {}
    finished = None
    for line in log.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[1] == "sha256" and parts[3] == "bytes":
            splits.setdefault(parts[0], {}).update(tarball_sha256=parts[2], tarball_bytes=int(parts[4]))
        elif len(parts) == 3 and parts[1] == "flac_count":
            splits.setdefault(parts[0], {})["flac_count"] = int(parts[2])
        elif len(parts) == 2 and parts[0] == "done":
            finished = parts[1]
    splits.setdefault("train-clean-100", {})["note"] = (
        "extracted earlier for the stt-distillation project; tarball hash not recorded here")
    for split, entry in splits.items():
        entry["url"] = URL.format(split=split)
        entry["extracted"] = str(paths.SPLITS.get(split, ""))
        manifest = paths.MANIFESTS / f"{split}.json"
        if manifest.exists():
            entry["manifest_utterances"] = len(json.loads(manifest.read_text()))
    provenance = {"license": LICENSE, "download_log": str(log), "download_finished": finished,
                  "splits": splits}
    write_json(paths.DATA / "provenance.json", provenance)
    return provenance


def dev_subset(manifest: list[dict], n: int = 400) -> list[dict]:
    """Deterministic selection subset: every k-th id-sorted utterance, k = len // n, first n."""
    ordered = sorted(manifest, key=lambda r: r["id"])
    if not 0 < n <= len(ordered):
        raise ValueError(f"subset size {n} not in 1..{len(ordered)}")
    return ordered[::len(ordered) // n][:n]


class LibriSpeechDataset(Dataset):
    """Log-mel features [80, 3000] and label ids for each manifest record."""

    def __init__(self, manifest: list[dict], feature_extractor, tokenizer,
                 decoder_start_token_id: int) -> None:
        self.manifest = manifest
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.start = decoder_start_token_id

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict:
        record = self.manifest[index]
        audio, rate = sf.read(record["path"], dtype="float32")
        if rate != paths.SAMPLE_RATE or audio.ndim != 1:
            raise ValueError(f"{record['path']}: expected {paths.SAMPLE_RATE} Hz mono")
        features = self.feature_extractor(audio, sampling_rate=paths.SAMPLE_RATE,
                                          return_tensors="np").input_features[0]
        ids = self.tokenizer(record["text"].lower()).input_ids
        if ids and ids[0] == self.start:
            ids = ids[1:]
        return {"input_features": torch.from_numpy(features), "labels": torch.tensor(ids),
                "id": record["id"], "truncated": record["duration_s"] > MAX_SECONDS}


def collate(batch: list[dict]) -> dict:
    labels = torch.full((len(batch), max(len(b["labels"]) for b in batch)), -100, dtype=torch.long)
    for row, item in enumerate(batch):
        labels[row, :len(item["labels"])] = item["labels"]
    return {"input_features": torch.stack([b["input_features"] for b in batch]).float(),
            "labels": labels, "ids": [b["id"] for b in batch],
            "truncated": [b["truncated"] for b in batch]}


class EpochSampler(Sampler[int]):
    """Permutation from a torch.Generator seeded with seed + epoch; identical for every arm."""

    def __init__(self, size: int, seed: int = paths.SEED) -> None:
        self.size, self.seed, self.epoch = size, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.size, generator=generator).tolist())

    def __len__(self) -> int:
        return self.size


def train_loader(dataset: Dataset, batch_size: int, workers: int = 8,
                 seed: int = paths.SEED) -> tuple[DataLoader, EpochSampler]:
    """Shuffled, drop_last training loader; call sampler.set_epoch(e) before each pass."""
    sampler = EpochSampler(len(dataset), seed)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers,
                        pin_memory=True, persistent_workers=workers > 0, drop_last=True,
                        collate_fn=collate)
    return loader, sampler


def reference_batch(processor, config, n: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed export-check batch: first n utterances of the 400-utterance dev subset.

    Returns (input_features, decoder_input_ids) with the labels shifted right
    exactly as the model does for teacher forcing.
    """
    manifest = dev_subset(build_manifest("dev-clean"))[:n]
    dataset = LibriSpeechDataset(manifest, processor.feature_extractor, processor.tokenizer,
                                 config.decoder_start_token_id)
    batch = collate([dataset[i] for i in range(len(dataset))])
    return batch["input_features"], shift_tokens_right(
        batch["labels"], config.pad_token_id, config.decoder_start_token_id)
