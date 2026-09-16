"""Pinned Hugging Face speech streaming with exact bounded-buffer checkpointing.

The baseline deliberately uses one iterator in the training process. Its raw
shuffle buffers are saved together with HF state *after* prefetch, avoiding the
documented loss of HF ``IterableDataset.shuffle`` buffers on resume. Do not put
this iterator in a multiprocessing DataLoader. Checkpoints must also save any
examples already consumed into the trainer's own pending batch.

Shard order is randomized from the first pass using a per-source, per-epoch
seed. This uses the buffer-free ``set_epoch(nonzero_seed)`` behavior verified
in datasets 5.0.1; a version guard prevents silent changes after an upgrade.
"""

from __future__ import annotations

from copy import deepcopy
import gc
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Callable

import torch

from .tokenizer import artifact_path, normalize_text


class DataPipelineError(RuntimeError):
    """An unusable dataset or upstream I/O failure must stop training visibly."""


class InvalidExample(ValueError):
    """A malformed individual row that counts toward rejection limits."""


def local_source_revision(source: dict) -> str:
    """Content pin for local Parquet smoke fixtures on the mounted artifact disk."""
    files = source.get("data_files")
    if isinstance(files, dict):
        files = files.get(source["split"])
    if isinstance(files, str):
        files = [files]
    if not isinstance(files, list) or not files:
        raise ValueError("Local Parquet sources require explicit data_files; globs are not supported")
    digest = hashlib.sha256()
    for filename in sorted(files):
        path = artifact_path(filename)
        if path.suffix != ".parquet" or not path.is_file():
            raise ValueError(f"Expected an existing local Parquet file: {path}")
        digest.update(str(path).encode() + b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def resolve_sources(sources: list[dict], *, token: str | bool | None = None) -> list[dict]:
    """Resolve each Hub branch/tag to immutable SHA without fetching audio.

    This resolves identity only. ``StreamingSpeechDataset`` subsequently checks
    config, split and required columns before yielding any examples.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    resolved = []
    for original in sources:
        source = deepcopy(original)
        if source["id"] == "parquet":
            source["revision"] = local_source_revision(source)
            resolved.append(source)
            continue
        info = api.dataset_info(source["id"], revision=source.get("revision") or "main")
        if not info.sha or not re.fullmatch(r"[0-9a-f]{40,64}", info.sha):
            raise DataPipelineError(f"Hub returned no immutable revision for {source['id']}")
        source["revision"] = info.sha
        resolved.append(source)
    return resolved


def _source_name(source: dict) -> str:
    return "/".join(str(source.get(key) or "default") for key in ("id", "config", "split"))


def _shard_epoch(source: dict, seed: int, epoch: int) -> int:
    """Derive a stable, strictly positive HF epoch/shard seed, independent of RNG use."""
    identity = [seed, epoch, source["id"], source.get("config"), source["split"], source["revision"]]
    digest = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).digest()
    # HF keeps its epoch in an int64 torch scalar. Zero disables shard shuffling.
    return int.from_bytes(digest[:8], "big") % ((1 << 63) - 1) + 1


def _synchronous_parquet_reader(stream, download_config) -> None:
    """Replace only this stream's async Arrow scanner with synchronous reads.

    Native DatasetScanner callbacks into a remote Python/fsspec file can remain
    active after generator.close() and deadlock during interpreter finalization.
    HF still owns source resolution, shuffling, Arrow iteration and checkpoints;
    this version-guarded adapter changes its Parquet batch reader only. No global
    monkeypatch or exit bypass is used. Our source catalog is entirely Parquet.
    """
    from datasets.builder import Key
    from datasets.iterable_dataset import ArrowExamplesIterable
    from datasets.packaged_modules.parquet.parquet import Parquet
    from datasets.utils.file_utils import xopen
    import pyarrow as pa
    import pyarrow.parquet as pq

    iterable = stream._ex_iterable
    builder = getattr(getattr(iterable, "generate_tables_fn", None), "__self__", None)
    if not isinstance(iterable, ArrowExamplesIterable) or not isinstance(builder, Parquet):
        raise DataPipelineError("The synchronous streaming adapter requires a datasets 5.0.1 Parquet source")
    if builder.config.filters is not None:
        raise DataPipelineError("Parquet predicate filters are not supported by the synchronous reader")

    def generate_tables(files, row_groups_list):
        for file_index, (filename, row_groups) in enumerate(zip(files, row_groups_list)):
            with xopen(filename, "rb", download_config=download_config) as handle:
                parquet = pq.ParquetFile(handle, pre_buffer=False)
                try:
                    if not parquet.metadata.num_row_groups:
                        continue
                    batch_size = builder.config.batch_size or parquet.metadata.row_group(0).num_rows
                    batches = parquet.iter_batches(batch_size=max(1, batch_size), row_groups=row_groups,
                                                    columns=builder.config.columns, use_threads=False)
                    try:
                        for batch_index, batch in enumerate(batches):
                            yield Key(file_index, batch_index), builder._cast_table(pa.Table.from_batches([batch]))
                    finally:
                        del batches
                finally:
                    parquet.close()

    iterable.generate_tables_fn = generate_tables


def _load_hf_source(source: dict, cache_dir: str | Path, epoch: int = 0, *, seed: int = 0,
                    text_only: bool = False):
    from datasets import Audio, DownloadConfig, load_dataset, __version__ as datasets_version
    import pyarrow.dataset as arrow_dataset

    if datasets_version != "5.0.1":
        raise DataPipelineError(
            "Exact shard-shuffle replay is verified for datasets==5.0.1; "
            f"found {datasets_version}. Revalidate replay before upgrading."
        )

    cache = artifact_path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    kwargs = dict(name=source.get("config"), split=source["split"], streaming=True,
                  cache_dir=str(cache), download_config=DownloadConfig(cache_dir=str(cache), max_retries=2),
                  fragment_scan_options=arrow_dataset.ParquetFragmentScanOptions(pre_buffer=False))
    if source["id"] == "parquet":
        if local_source_revision(source) != source["revision"]:
            raise DataPipelineError("Local Parquet data changed since its revision was pinned")
        kwargs["data_files"] = source["data_files"]
    else:
        if not re.fullmatch(r"[0-9a-f]{40,64}", source.get("revision", "")):
            raise ValueError("Dataset must be pinned to an immutable commit")
        kwargs["revision"] = source["revision"]
    if text_only:
        # Our catalog uses Parquet; projection avoids transferring audio columns.
        kwargs["columns"] = [source["text_column"]]
    stream = load_dataset(source["id"], **kwargs)
    _synchronous_parquet_reader(stream, kwargs["download_config"])
    required = [source["text_column"]] if text_only else [source["text_column"], source["audio_column"]]
    if stream.features is not None and any(column not in stream.features for column in required):
        raise DataPipelineError(f"Missing columns {required} in {_source_name(source)}")
    if not text_only:
        stream = stream.cast_column(source["audio_column"], Audio(decode=False))
    # In the pinned version, public set_epoch(nonzero) invokes
    # shuffle_data_sources(default_rng(epoch)) during iteration preparation,
    # without creating the lossy BufferShuffledExamplesIterable wrapper.
    # Keep our logical epoch separately in the cursor and reproduce this derived
    # epoch before load_state_dict, so HF restores the same shuffled shard index.
    stream.set_epoch(_shard_epoch(source, seed, epoch))
    return stream


def iter_source_texts(sources: list[dict], max_records: int = 1_000_000, *, seed: int = 0,
                      cache_dir: str | Path = "/mnt/hd/wilderness-labs-stt/binary-stt/cache/datasets",
                      casefold: bool = True):
    """Bounded weighted tokenizer corpus; Parquet column projection avoids audio.

    Exhausted sources are removed; this never repeats a source. Shard order is
    independently seeded for every source from the first example. Both malformed
    and accepted rows consume the finite ``max_records`` input budget.
    """
    if max_records <= 0:
        raise ValueError("max_records must be positive")
    rng = random.Random(seed)
    active = list(range(len(sources)))
    streams = {}
    try:
        for _ in range(max_records):
            while active:
                index = rng.choices(active, weights=[float(sources[i].get("weight", 1)) for i in active], k=1)[0]
                if index not in streams:
                    streams[index] = iter(_load_hf_source(sources[index], cache_dir, seed=seed, text_only=True))
                try:
                    row = next(streams[index])
                    break
                except StopIteration:
                    active.remove(index)
            else:
                return
            value = row.get(sources[index]["text_column"])
            if isinstance(value, str):
                text = normalize_text(value, casefold=casefold)
                if text:
                    yield text
    finally:
        for iterator in streams.values():
            close = getattr(iterator, "close", None)
            if close:
                close()
        streams.clear()
        # Release Arrow scans while Python's interpreter/GIL machinery is live.
        gc.collect()


class StreamingSpeechDataset(torch.utils.data.IterableDataset):
    """Weighted streaming mixer; sample weights are probabilities per utterance.

    Sources require ``id, split, revision, text_column, audio_column``. Optional
    fields are ``config, id_column, speaker_column, weight, license``. Revisions
    must be immutable Hub commit hashes. ``factory(source, epoch)`` is solely an
    injection point for stateful local fixtures; injected streams must implement
    ``state_dict`` and ``load_state_dict`` and may use revision ``fixture``.

    The iterator emits CPU float32 mono audio at 16 kHz, normalized transcripts,
    duration, source, canonical source-row ID, speaker and decoded-PCM content
    hash. IDs remain stable across repeats. Content hashes detect exact decoded
    duplicates, not overlapping crops or lossy re-encodings. No unbounded ID set
    is retained here. A downstream on-disk ledger can count unique exposure.

    Network/decoder backend failures stop immediately; individual malformed
    examples are rejected subject to global and per-source rejection limits.
    Cache and tokenizer/model artifacts must live on the mounted data drive.
    """

    STATE_VERSION = 3

    def __init__(
        self,
        sources: list[dict],
        seed: int = 0,
        shuffle_buffer: int = 32,
        min_seconds: float = 0.25,
        max_seconds: float = 20.0,
        *,
        repeat: bool = True,
        max_consecutive_bad: int = 100,
        max_rejection_fraction: float = 0.5,
        rejection_fraction_min_samples: int = 100,
        max_audio_bytes: int = 32 * 1024 * 1024,
        max_buffer_bytes: int = 256 * 1024 * 1024,
        max_text_chars: int = 4096,
        casefold: bool = True,
        cache_dir: str | Path = "/mnt/hd/wilderness-labs-stt/binary-stt/cache/datasets",
        factory: Callable[[dict, int], Any] | None = None,
    ):
        super().__init__()
        if not sources:
            raise ValueError("At least one source is required")
        if shuffle_buffer < 1 or not math.isfinite(max_seconds) or not 0 < min_seconds < max_seconds:
            raise ValueError("Require shuffle_buffer >= 1 and 0 < min_seconds < max_seconds")
        if min(max_consecutive_bad, rejection_fraction_min_samples, max_audio_bytes, max_buffer_bytes, max_text_chars) < 1:
            raise ValueError("Data safety budgets must be positive")
        if not 0 <= max_rejection_fraction <= 1:
            raise ValueError("max_rejection_fraction must be in [0, 1]")
        self.sources = deepcopy(sources)
        for source in self.sources:
            for key in ("id", "split", "revision", "text_column", "audio_column"):
                if not source.get(key):
                    raise ValueError(f"Source missing required field: {key}")
            revision = source["revision"]
            if not re.fullmatch(r"[0-9a-f]{40,64}", revision) and not (factory and revision == "fixture"):
                raise ValueError(f"Pin {source['id']} to a commit SHA with resolve_sources(), not {revision!r}")
            if source["id"] == "parquet" and local_source_revision(source) != revision:
                raise ValueError("Local Parquet revision does not match data_files content")
            weight = float(source.get("weight", 1.0))
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("Every source weight must be positive and finite")
            source["weight"] = weight
        names = [_source_name(source) for source in self.sources]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate dataset/config/split entries would double-count a source")
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        self.repeat = repeat
        self.max_consecutive_bad = max_consecutive_bad
        self.max_rejection_fraction = max_rejection_fraction
        self.rejection_fraction_min_samples = rejection_fraction_min_samples
        self.max_audio_bytes = max_audio_bytes
        self.max_buffer_bytes = max_buffer_bytes
        self.max_text_chars = max_text_chars
        self.casefold = casefold
        self.cache_dir = Path(cache_dir)
        self.factory = factory
        self.rng = random.Random(seed)
        self.stats = {"accepted": 0, "rejected": 0, "seconds": 0.0, "reasons": {}, "sources": {}}
        self.consecutive_bad = 0
        self._cursors: dict[int, dict] = {}
        self._active = list(range(len(sources)))
        self._buffer_bytes = 0
        self._closed = False

    def _fingerprint(self) -> str:
        settings = {name: getattr(self, name) for name in (
            "sources", "seed", "shuffle_buffer", "min_seconds", "max_seconds", "repeat",
            "max_consecutive_bad", "max_rejection_fraction", "rejection_fraction_min_samples",
            "max_audio_bytes", "max_buffer_bytes", "max_text_chars", "casefold",
        )}
        return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()

    def _new_cursor(self, index: int, epoch: int = 0) -> dict:
        source = self.sources[index]
        try:
            if self.factory:
                stream = self.factory(deepcopy(source), epoch)
            else:
                stream = _load_hf_source(source, self.cache_dir, epoch, seed=self.seed)
            if not all(hasattr(stream, name) for name in ("state_dict", "load_state_dict")):
                raise DataPipelineError("Stream must implement checkpointable state_dict/load_state_dict")
            return {"stream": stream, "iterator": iter(stream), "epoch": epoch, "buffer": [],
                    "exhausted": False, "rows_seen": 0, "accepted_in_epoch": 0}
        except Exception as error:
            raise DataPipelineError(f"Cannot open {_source_name(source)}: {error}") from error

    def _raw_row(self, index: int) -> tuple[dict, int] | None:
        cursor = self._cursors.setdefault(index, None)
        if cursor is None:
            cursor = self._cursors[index] = self._new_cursor(index)
        if cursor["exhausted"] and not cursor["buffer"]:
            if not cursor["rows_seen"] or not cursor["accepted_in_epoch"]:
                raise DataPipelineError(f"Empty or entirely rejected source: {_source_name(self.sources[index])}")
            if not self.repeat:
                self._active.remove(index)
                return None
            cursor = self._cursors[index] = self._new_cursor(index, cursor["epoch"] + 1)
        while len(cursor["buffer"]) < self.shuffle_buffer and not cursor["exhausted"]:
            try:
                row = next(cursor["iterator"])
            except StopIteration:
                cursor["exhausted"] = True
                break
            except Exception as error:
                raise DataPipelineError(f"Streaming read failed for {_source_name(self.sources[index])}: {error}") from error
            source = self.sources[index]
            columns = [source.get(key) for key in ("text_column", "audio_column", "id_column", "speaker_column")]
            if not isinstance(row, dict):
                row = {}
            row = {column: row[column] for column in columns if column in row}
            audio = row.get(source["audio_column"], {})
            size = len(audio.get("bytes") or b"") if isinstance(audio, dict) else 0
            size += len(str(row.get(source["text_column"], ""))) * 4
            if self._buffer_bytes + size > self.max_buffer_bytes:
                raise DataPipelineError("Raw shuffle buffer byte budget exceeded; reduce shuffle_buffer or clip sizes")
            cursor["buffer"].append((row, cursor["rows_seen"], size))
            cursor["rows_seen"] += 1
            self._buffer_bytes += size
        if not cursor["buffer"]:
            if not cursor["rows_seen"]:
                raise DataPipelineError(f"Empty source: {_source_name(self.sources[index])}")
            return self._raw_row(index)
        position = self.rng.randrange(len(cursor["buffer"]))
        row, ordinal, size = cursor["buffer"].pop(position)
        self._buffer_bytes -= size
        return row, ordinal

    def _decode(self, row: dict, source: dict, ordinal: int) -> dict:
        try:
            text = normalize_text(row[source["text_column"]], casefold=self.casefold)
        except (KeyError, TypeError) as error:
            raise InvalidExample("invalid_text") from error
        if not text or len(text) > self.max_text_chars:
            raise InvalidExample("empty_or_long_text")
        audio = row.get(source["audio_column"])
        if not isinstance(audio, dict):
            raise InvalidExample("invalid_audio")
        encoded = audio.get("bytes")
        path = audio.get("path")
        if encoded is None and path:
            # HF streaming archive paths can be virtual URLs; xopen resolves them
            # without writing a local copy. Bound compressed bytes before decode.
            try:
                from datasets.utils.file_utils import xopen

                with xopen(path, "rb") as handle:
                    encoded = handle.read(self.max_audio_bytes + 1)
            except Exception as error:
                raise DataPipelineError(f"Audio read failed for {_source_name(source)}: {error}") from error
        if not isinstance(encoded, (bytes, bytearray)) or not encoded:
            raise InvalidExample("missing_audio_bytes")
        if len(encoded) > self.max_audio_bytes:
            raise InvalidExample("encoded_audio_too_large")
        import soundfile as sf

        try:
            with sf.SoundFile(io.BytesIO(encoded)) as handle:
                rate = handle.samplerate
                if not 8000 <= rate <= 192000 or not 1 <= handle.channels <= 8:
                    raise InvalidExample("unsupported_audio_format")
                seconds = handle.frames / rate
                if not self.min_seconds <= seconds <= self.max_seconds:
                    raise InvalidExample("duration_out_of_range")
                samples = handle.read(dtype="float32", always_2d=True)
        except (sf.LibsndfileError, RuntimeError, ValueError) as error:
            if isinstance(error, InvalidExample):
                raise
            raise InvalidExample("audio_decode_error") from error
        waveform = torch.from_numpy(samples).mean(dim=1)
        if not bool(torch.isfinite(waveform).all()):
            raise InvalidExample("nonfinite_audio")
        if rate != 16000:
            from torchaudio.functional import resample

            waveform = resample(waveform, rate, 16000)
        waveform = waveform.contiguous()
        if not bool(torch.isfinite(waveform).all()):
            raise InvalidExample("nonfinite_resampled_audio")
        seconds = waveform.numel() / 16000
        if not self.min_seconds <= seconds <= self.max_seconds:
            raise InvalidExample("resampled_duration_out_of_range")
        pcm = waveform.numpy().astype("<f4", copy=False).tobytes()
        content_id = hashlib.sha256(b"pcm-f32le-mono-16000\0" + pcm).hexdigest()
        raw_id = row.get(source.get("id_column"))
        if raw_id is None:
            # Iteration ordinals change when shard order changes between epochs.
            # An exact-content fallback preserves identity for sources without IDs.
            raw_id = path or f"pcm-{content_id}"
        name = _source_name(source)
        identifier = json.dumps([source["id"], source.get("config"), source["split"], str(raw_id)], ensure_ascii=False, separators=(",", ":"))
        speaker = row.get(source.get("speaker_column"))
        return {"id": identifier, "source": name, "audio": waveform, "sample_rate": 16000,
                "text": text, "seconds": seconds, "speaker": None if speaker is None else str(speaker),
                "content_id": content_id}

    def __iter__(self):
        if torch.utils.data.get_worker_info() is not None:
            raise DataPipelineError("Exact replay requires num_workers=0")
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self) -> None:
        """Release suspended HF/Arrow file scans before interpreter shutdown.

        Call in ``finally`` (or use ``with``), including after bounded ``islice``
        and failure. Closing preserves the latest checkpoint state and buffers;
        ``load_state_dict`` may subsequently reopen the stream. It never drains
        the source or downloads the rest of a shard.
        """
        if self._closed:
            return
        self._closed = True
        for cursor in self._cursors.values():
            if cursor is None:
                continue
            iterator = cursor.get("iterator")
            cursor["iterator"] = None
            close = getattr(iterator, "close", None)
            if close:
                close()
            del iterator
        gc.collect()

    def __next__(self):
        if self._closed:
            raise StopIteration
        while self._active:
            index = self.rng.choices(self._active, weights=[self.sources[i]["weight"] for i in self._active], k=1)[0]
            raw = self._raw_row(index)
            if raw is None:
                continue
            source = self.sources[index]
            name = _source_name(source)
            source_stats = self.stats["sources"].setdefault(name, {"accepted": 0, "rejected": 0, "seconds": 0.0})
            try:
                example = self._decode(raw[0], source, raw[1])
            except InvalidExample as error:
                self.stats["rejected"] += 1
                source_stats["rejected"] += 1
                reason = str(error)
                self.stats["reasons"][reason] = self.stats["reasons"].get(reason, 0) + 1
                self.consecutive_bad += 1
                for counts in (self.stats, source_stats):
                    seen = counts["accepted"] + counts["rejected"]
                    excessive = seen >= self.rejection_fraction_min_samples and counts["rejected"] / seen > self.max_rejection_fraction
                    if self.consecutive_bad >= self.max_consecutive_bad or excessive:
                        raise DataPipelineError(f"Excessive rejected audio in {name}: {self.stats}") from error
                continue
            self._cursors[index]["accepted_in_epoch"] += 1
            self.stats["accepted"] += 1
            self.stats["seconds"] += example["seconds"]
            source_stats["accepted"] += 1
            source_stats["seconds"] += example["seconds"]
            self.consecutive_bad = 0
            return example
        self.close()
        raise StopIteration

    def state_dict(self) -> dict:
        cursors = {}
        for index, cursor in self._cursors.items():
            if cursor is None:
                continue
            state = {key: value for key, value in cursor.items() if key not in ("stream", "iterator")}
            state["stream_state"] = cursor["stream"].state_dict()
            cursors[index] = state
        return deepcopy({"version": self.STATE_VERSION, "fingerprint": self._fingerprint(),
                         "rng": self.rng.getstate(), "cursors": cursors, "active": self._active,
                         "stats": self.stats, "consecutive_bad": self.consecutive_bad,
                         "buffer_bytes": self._buffer_bytes})

    def load_state_dict(self, state: dict) -> None:
        state = deepcopy(state)
        if state.get("version") != self.STATE_VERSION or state.get("fingerprint") != self._fingerprint():
            raise DataPipelineError("Dataset checkpoint source/configuration mismatch")
        self.close()
        cursors = {}
        for index, saved in state["cursors"].items():
            index = int(index)
            cursor = self._new_cursor(index, saved["epoch"])
            cursor["stream"].load_state_dict(saved.pop("stream_state"))
            cursor["iterator"] = iter(cursor["stream"])
            cursor.update(saved)
            cursors[index] = cursor
        self._cursors = cursors
        self._active = state["active"]
        self._buffer_bytes = state["buffer_bytes"]
        self.rng.setstate(state["rng"])
        self.stats = state["stats"]
        self.consecutive_bad = state["consecutive_bad"]
        self._closed = False
