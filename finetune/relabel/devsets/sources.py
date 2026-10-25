"""Source readers for the domain dev sets: recordings with audio and a transcript, by alias.

A source is configured by its private registry entry (registry.py); the reader kind is its "layout":

- "textgrid-channels": one 16 kHz WAV per channel in raw/<audio_dir>, and per WAV a Praat TextGrid in
  raw/<transcript_dir>/<stem><transcript_suffix> holding that channel's utterances with timestamps. Each
  channel is its own recording (id = file stem), so a consultation's two channels stay separate utterances.
- "parquet-clips": one pre-cut utterance per Parquet row (audio column with "bytes", text column); the
  utterance spans the whole clip.
- "parquet-longform": one long recording per row with a plain transcript and no timestamps (aligned).
- "srt-longform": one media file per recording (suffix "media_suffix", any container ffmpeg reads, id = file
  stem) with a SubRip caption file of the same stem in the same directory. Caption cues are 2-4 s display
  units, not utterances, so their timestamps are not used: the transcript is the cues' text in order, joined,
  without speaker labels (an upper-case name and a colon opening a caption line, e.g. "PROFESSOR:"), and the
  recording is aligned long-form. Audio is decoded by ffmpeg ($FFMPEG, else
  $RELABEL_ROOT/deps/ffmpeg-headless-bin/bin/ffmpeg) straight to 16 kHz mono.

Parquet layouts read only the fetched ".parquet" files (other fetched files, such as the dataset card, are
documentation). Recording ids come from the registry's "id_column" (prefixed with the file's split name, the file name up
to its first "-", when "id_with_split" is set); "id_column": "__row__" (a source without an id column) names a
row by its file and position, "<file stem>-r<row in file, 5 digits>". Rows are listed in file order (the fetched file list) and row
order. Listing drops a recording whose id repeats ("duplicate_recording") or that is a test-set recording
("test_overlap": the registry's "exclude_test_set" names a test set whose recording ids, read from the
test manifest's "id" field only, the part must not contain; the recording id is the test id up to "/").
Audio is decoded with soundfile, averaged to mono and resampled to 16 kHz (torchaudio sinc resampling).
"""
from __future__ import annotations

import io
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from devsets.segment import SAMPLE_RATE, Utterance

LAYOUTS = ("textgrid-channels", "parquet-clips", "parquet-longform", "srt-longform")
ROW_ID = "__row__"
_SPEAKER = re.compile(r"^(?:[A-Z][A-Z0-9.'\- ]*[A-Z0-9.]|>>)\s*:\s+")
_SRT_TIME = re.compile(r"^\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->")
RECORDING_DROPS = ("duplicate_recording", "test_overlap", "unreadable_audio", "missing_transcript")
_INTERVAL = re.compile(r'intervals\s*\[\d+\]:\s*xmin\s*=\s*([-+\d.eE]+)\s*xmax\s*=\s*([-+\d.eE]+)\s*'
                       r'text\s*=\s*"((?:[^"]|"")*)"', re.S)


@dataclass(frozen=True)
class Item:
    id: str
    locator: tuple   # (path, row group, row) for Parquet; (wav path, transcript path) for TextGrid


@dataclass
class Recording:
    id: str
    audio: np.ndarray                  # 16 kHz mono float32
    transcript: list[Utterance] | str  # source timestamps, or a plain transcript to align


def parse_textgrid(text: str) -> list[Utterance]:
    """Non-empty intervals of every interval tier of a (long-format) Praat TextGrid, in time order."""
    out = [Utterance(float(a), float(b), t.replace('""', '"')) for a, b, t in _INTERVAL.findall(text)]
    return sorted((u for u in out if u.text.strip()), key=lambda u: (u.start, u.end))


def read_text(path: Path) -> str:
    data = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{path}: neither UTF-8 nor UTF-16")


def to_16k_mono(audio: np.ndarray, rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        import torch
        import torchaudio.functional as AF

        audio = AF.resample(torch.from_numpy(np.ascontiguousarray(audio)), rate, SAMPLE_RATE).numpy()
    return np.ascontiguousarray(audio, dtype=np.float32)


def decode(data: bytes) -> np.ndarray:
    audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return to_16k_mono(audio, rate)


def srt_text(text: str) -> str:
    """The caption text of a SubRip file, cues in file order, lines joined by spaces, speaker labels removed."""
    out = []
    for block in re.split(r"\r?\n\s*\r?\n", text.replace("\ufeff", "")):
        lines = [line.strip() for line in block.strip().splitlines()]
        if lines and lines[0].isdigit():
            lines = lines[1:]
        if not lines or not _SRT_TIME.match(lines[0]):
            continue
        for line in lines[1:]:
            line = _SPEAKER.sub("", line).strip()
            if line:
                out.append(line)
    return " ".join(" ".join(out).split())


def ffmpeg_path() -> str:
    """$FFMPEG, else the pinned ffmpeg under $RELABEL_ROOT/deps; RuntimeError if neither exists."""
    import os

    path = os.environ.get("FFMPEG") or str(Path(os.environ.get("RELABEL_ROOT", "/nonexistent")) / "deps" /
                                           "ffmpeg-headless-bin" / "bin" / "ffmpeg")
    if not os.access(path, os.X_OK):
        raise RuntimeError(f"no ffmpeg at {path} (set FFMPEG or build the pinned one into $RELABEL_ROOT/deps)")
    return path


def ffmpeg_env() -> dict:
    """The environment for ffmpeg: ours without LD_LIBRARY_PATH (the runtime's library path would shadow the
    libraries of ffmpeg's own closure)."""
    import os

    return {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}


def ffmpeg_version() -> str:
    import subprocess

    out = subprocess.run([ffmpeg_path(), "-version"], capture_output=True, text=True, check=True,
                         env=ffmpeg_env()).stdout
    return out.splitlines()[0] if out else "unknown"


def decode_media(path: Path) -> np.ndarray:
    """16 kHz mono float32 audio of any media file ffmpeg reads (video streams ignored)."""
    import subprocess

    cmd = [ffmpeg_path(), "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
           "-f", "f32le", "-"]
    done = subprocess.run(cmd, capture_output=True, check=True, env=ffmpeg_env())
    audio = np.frombuffer(done.stdout, dtype="<f4")
    if not len(audio):
        raise ValueError(f"{path}: no audio")
    return audio.astype(np.float32, copy=True)  # writable (np.frombuffer over bytes is read-only)


def test_set_recordings(name: str) -> set[str]:
    """Recording ids of a test set, from the "id" field of its manifest only (decontamination)."""
    import json

    import evaluate  # finetune/parakeet-ternary/evaluate.py: resolve_manifest

    import paths

    if name not in paths.TEST_SETS:
        raise ValueError(f"exclude_test_set {name!r} is not a test set")
    _, manifest = evaluate.resolve_manifest(name)
    ids = set()
    with open(manifest) as handle:
        for line in handle:
            if line.strip():
                ids.add(str(json.loads(line)["id"]).split("/")[0])
    if not ids:
        raise ValueError(f"no ids in {manifest}")
    return ids


class Source:
    def __init__(self, alias: str, cfg: dict, files: list[Path]) -> None:
        if cfg["layout"] not in LAYOUTS:
            raise ValueError(f"{alias}: unknown layout {cfg['layout']!r}")
        self.alias, self.cfg, self.files = alias, cfg, [Path(f) for f in files]
        self.boundary_words = tuple(cfg.get("boundary_words", ()))
        self._cache: tuple | None = None

    # -- listing -------------------------------------------------------------------------------
    def _raw_items(self) -> list[Item]:
        cfg = self.cfg
        if cfg["layout"] == "textgrid-channels":
            wavs = sorted(f for f in self.files if f.parent.name == cfg["audio_dir"] and f.suffix.lower() == ".wav")
            by_name = {(f.parent.name, f.name): f for f in self.files}
            items = []
            for wav in wavs:
                key = (cfg["transcript_dir"], wav.stem + cfg.get("transcript_suffix", ".TextGrid"))
                items.append(Item(wav.stem, (wav, by_name.get(key))))
            return items
        if cfg["layout"] == "srt-longform":
            suffix = cfg["media_suffix"].lower()
            captions = {f.with_suffix("").as_posix(): f for f in self.files if f.suffix.lower() == ".srt"}
            media = sorted((f for f in self.files if f.name.lower().endswith(suffix)), key=lambda f: f.name)
            return [Item(f.name[:-len(suffix)], (f, captions.get(f.as_posix()[:-len(suffix)]))) for f in media]
        import pyarrow.parquet as pq

        items = []
        for path in (f for f in self.files if f.suffix == ".parquet"):  # fetched docs are not data
            split = path.name.split("-")[0]
            pf = pq.ParquetFile(path)
            offset = 0
            for group in range(pf.metadata.num_row_groups):
                if cfg["id_column"] == ROW_ID:
                    ids = [f"{path.stem}-r{offset + row:05d}" for row in range(pf.metadata.row_group(group).num_rows)]
                else:
                    ids = pf.read_row_group(group, columns=[cfg["id_column"]]).column(0).to_pylist()
                offset += len(ids)
                for row, rid in enumerate(ids):
                    rid = f"{split}-{rid}" if cfg.get("id_with_split") else str(rid)
                    items.append(Item(rid, (path, group, row)))
        return items

    def items(self) -> tuple[list[Item], Counter]:
        """(recordings to build, recording drops by reason) in a fixed order."""
        drops: Counter = Counter()
        excluded = test_set_recordings(self.cfg["exclude_test_set"]) if self.cfg.get("exclude_test_set") else set()
        seen, out = set(), []
        for item in self._raw_items():
            if item.id in seen:
                drops["duplicate_recording"] += 1
            elif item.id in excluded:
                drops["test_overlap"] += 1
            else:
                seen.add(item.id)
                out.append(item)
        return out, drops

    # -- loading -------------------------------------------------------------------------------
    def _row(self, path: Path, group: int, row: int) -> dict:
        import pyarrow.parquet as pq

        if self._cache is None or self._cache[0] != (path, group):
            columns = [self.cfg["audio_column"], self.cfg["text_column"]]
            self._cache = ((path, group), pq.ParquetFile(path).read_row_group(group, columns=columns))
        table = self._cache[1]
        return {name: table.column(name)[row].as_py() for name in table.column_names}

    def load(self, item: Item) -> Recording:
        """The recording; raises ValueError("missing_transcript") / ("unreadable_audio") for a recording drop."""
        cfg = self.cfg
        if cfg["layout"] == "textgrid-channels":
            wav, grid = item.locator
            if grid is None:
                raise ValueError("missing_transcript")
            try:
                audio, rate = sf.read(wav, dtype="float32", always_2d=True)
            except Exception as error:  # noqa: BLE001  any decoder failure is a recording drop
                raise ValueError("unreadable_audio") from error
            return Recording(item.id, to_16k_mono(audio, rate), parse_textgrid(read_text(grid)))
        if cfg["layout"] == "srt-longform":
            media, captions = item.locator
            if captions is None:
                raise ValueError("missing_transcript")
            try:
                audio = decode_media(media)
            except Exception as error:  # noqa: BLE001
                raise ValueError("unreadable_audio") from error
            return Recording(item.id, audio, srt_text(read_text(captions)))
        row = self._row(*item.locator)
        cell = row[cfg["audio_column"]]
        try:
            audio = decode(cell["bytes"] if isinstance(cell, dict) else cell)
        except Exception as error:  # noqa: BLE001
            raise ValueError("unreadable_audio") from error
        text = row[cfg["text_column"]] or ""
        if cfg["layout"] == "parquet-clips":
            return Recording(item.id, audio, [Utterance(0.0, len(audio) / SAMPLE_RATE, text)])
        return Recording(item.id, audio, text)
