"""Fixture world for the test_labeling_* contract tests (no tests here).

A temporary $RELABEL_ROOT with:
- local parquet "Hub" files for the five existing sources (two files each, row groups of ROWS_PER_GROUP rows,
  16 kHz PCM16 FLAC audio of seeded noise; ids encode their recording: "<recording>-<k>");
- Part B outputs: pool/summary.json, pool/denylist.json, pool/subsets/*.jsonl with locators, and text-cache
  shards for the subsets' mean durations;
- a built `med-osce` corpus (manifest + FLAC files) and a dropped `syn-dialog`.

Recordings (yodas): yA reserved (denylisted), yG denylisted, yB and yH in yodas-med (yB listed in row group 1 of
file 0; two more yB rows sit in row groups no yodas-med locator lists: row group 0 of file 0, a file yodas-med
does read, and row group 1 of file 1), yD and yE in yodas-tech,
yC shares yB's row group and belongs to no subset, yF plain; peoples_speech: pA in ps-med, pB denylisted, pC and
pD plain (pD shares a listed ps-med row group).
"""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

ROWS_PER_GROUP = 4
SR = 16000


def flac(seed: str, seconds: float) -> bytes:
    rng = np.random.default_rng([ord(c) for c in seed])
    pcm = rng.integers(-3000, 3000, int(seconds * SR), dtype=np.int16)
    buf = io.BytesIO()
    sf.write(buf, pcm, SR, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def duration_for(ident: str) -> float:
    return 1.2 + (sum(map(ord, ident)) % 25) / 10.0     # 1.2 .. 3.6 s


def write_parquet(path: Path, groups: list[list[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    for rows in groups:
        t = pa.Table.from_pylist(rows)
        if writer is None:
            writer = pq.ParquetWriter(path, t.schema)
        writer.write_table(t)
    writer.close()


def _row(src: str, rec: str, k: int, short: bool = False) -> dict:
    ident = f"{rec}-{k}"
    d = 0.5 if short else duration_for(ident)
    audio = {"bytes": flac(ident, d), "path": None}
    if src == "yodas":
        return {"utt_id": ident, "text": f"pseudo {rec} {k}", "audio": audio, "original_audio_id": rec}
    if src == "peoples_speech":
        return {"id": f"{rec}_{k:05d}.flac", "text": f"human {rec} {k}", "audio": audio}
    if src == "voxpopuli":
        return {"audio_id": ident, "normalized_text": f"human {rec} {k}", "audio": audio}
    if src == "ami":
        return {"audio_id": ident, "text": f"human {rec} {k}", "audio": audio}
    return {"id": ident, "text": f"human {rec} {k}", "audio": audio}


YODAS_FILES = {  # file -> row groups -> list of (recording, k)
    "data/en000/asr_only/00000000.parquet": [[("yA", 0), ("yA", 1), ("yA", 2), ("yB", 8)],
                                             [("yB", 0), ("yB", 1), ("yC", 0), ("yC", 1)],
                                             [("yD", 0), ("yD", 1), ("yH", 0), ("yH", 1)]],
    "data/en000/asr_only/00000001.parquet": [[("yE", i) for i in range(4)],
                                             [("yF", 0), ("yF", 1), ("yF", 2), ("yB", 9)],
                                             [("yG", 0), ("yG", 1), ("yF", 4), ("yF", 5, "short")]],
}
PS_FILES = {
    "clean/train-00000-of-00002.parquet": [[("pA", 1), ("pA", 2), ("pC", 1), ("pC", 2)],
                                           [("pB", 1), ("pB", 2), ("pC", 3), ("pC", 4)]],
    "clean/train-00001-of-00002.parquet": [[("pA", 3), ("pD", 1), ("pD", 2), ("pD", 3)]],
}
SIMPLE = {"librispeech": "lsR", "voxpopuli": "vpR", "ami": "amR"}
HF_SOURCES = {"yodas": {"repo": "r/y", "revision": "1", "id_col": "utt_id", "text_col": "text"},
              "peoples_speech": {"repo": "r/p", "revision": "1", "id_col": "id", "text_col": "text"},
              "librispeech": {"repo": "r/l", "revision": "1", "id_col": "id", "text_col": "text"},
              "voxpopuli": {"repo": "r/v", "revision": "1", "id_col": "audio_id", "text_col": "normalized_text"},
              "ami": {"repo": "r/a", "revision": "1", "id_col": "audio_id", "text_col": "text"}}
MEAN_SECONDS = {"yodas": 2.4, "librispeech": 2.4, "peoples_speech": 2.4, "voxpopuli": 2.4, "ami": 2.4}
OSCE = [("oscA", k) for k in range(3)] + [("oscB", k) for k in range(2)]


class World:
    def __init__(self, root: Path, osce: str = "built", final: bool = True, deny_extra: dict | None = None):
        self.root = Path(root)
        self.hub = self.root / "hub"
        self.pool = self.root / "pool"
        self.corpora = self.root / "corpora"
        self.files: dict[str, list[str]] = {}
        self.rows: dict[str, dict] = {}       # id (as the stream builds it) -> source row
        self._hub(osce, final, deny_extra or {})

    def url(self, repo: str, revision: str, path: str) -> str:
        return str(self.hub / repo.replace("/", "_") / path)

    def _write_source(self, src: str, files: dict) -> None:
        spec = HF_SOURCES[src]
        out = []
        for path, groups in files.items():
            built = [[_row(src, g[0], g[1], len(g) > 2) for g in rows] for rows in groups]
            write_parquet(Path(self.url(spec["repo"], spec["revision"], path)), built)
            out.append(self.url(spec["repo"], spec["revision"], path))
            for rows in built:
                for r in rows:
                    self.rows[r[spec["id_col"]]] = r
        self.files[src] = sorted(out)

    def _hub(self, osce: str, final: bool, deny_extra: dict) -> None:
        self._write_source("yodas", YODAS_FILES)
        self._write_source("peoples_speech", PS_FILES)
        for src, rec in SIMPLE.items():
            self._write_source(src, {f"{src}/train-0.parquet": [[(rec, i) for i in range(3)]],
                                     f"{src}/train-1.parquet": [[(rec + "x", i) for i in range(3)]]})
        # Part B outputs
        subsets = {"yodas-med": [{"recording": "yB", "locators": [{"file": "data/en000/asr_only/00000000.parquet",
                                                                    "row_groups": [1]}]},
                                 {"recording": "yH", "locators": [{"file": "data/en000/asr_only/00000000.parquet",
                                                                    "row_groups": [2]}]}],
                   "yodas-tech": [{"recording": "yD", "locators": [{"file": "data/en000/asr_only/00000000.parquet",
                                                                     "row_groups": [2]}]},
                                  {"recording": "yE", "locators": [{"file": "data/en000/asr_only/00000001.parquet",
                                                                     "row_groups": [0]}]}],
                   "ps-med": [{"recording": "pA", "locators": [{"file": "clean/train-00000-of-00002.parquet",
                                                                 "row_groups": [0]},
                                                                {"file": "clean/train-00001-of-00002.parquet",
                                                                 "row_groups": [0]}]}]}
        for name, rows in subsets.items():
            p = self.pool / "subsets" / f"{name}.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        summary = {"final": final, "subsets": {"yodas-med": {"hours_1_30s": 3.0}, "ps-med": {"hours_1_30s": 1.0},
                                               "yodas-tech": {"hours_1_30s": 5.0}}}
        (self.pool / "summary.json").write_text(json.dumps(summary))
        deny = {"final": final, "recordings": {"yodas": ["yA", "yG"], "peoples_speech": ["pB"], **deny_extra}}
        (self.pool / "denylist.json").write_text(json.dumps(deny))
        # text cache (recording, duration) for the subsets' mean seconds
        from pool import fetch
        for parent, files in (("yodas", YODAS_FILES), ("peoples_speech", PS_FILES)):
            for path, groups in files.items():
                recs = [g[0] for rows in groups for g in rows]
                durs = [0.5 if len(g) > 2 else duration_for(f"{g[0]}-{g[1]}") for rows in groups for g in rows]
                cache, _ = fetch.cache_paths(self.pool / "text" / parent, {"path": path})
                cache.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.table({"recording": recs, "duration": pa.array(durs, pa.float32())}), cache)
        # local corpora
        sd = self.corpora / "syn-dialog"
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "status.json").write_text(json.dumps({"status": "dropped"}))
        cd = self.corpora / "med-osce"
        cd.mkdir(parents=True, exist_ok=True)
        if osce == "built":
            rows = []
            for rec, k in OSCE:
                ident = f"med-osce:{rec}-{k}"
                f = cd / "audio" / rec / f"{rec}-{k}.flac"
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(flac(ident, duration_for(ident)))
                rows.append({"audio_filepath": str(f), "duration": duration_for(ident), "text": f"osce {rec} {k}",
                             "id": ident, "source": "med-osce", "recording": rec})
            m = cd / "manifest.jsonl"
            m.write_text("".join(json.dumps(r) + "\n" for r in rows))
            from pool import common as c
            (cd / "status.json").write_text(json.dumps({"status": "built", "manifest": str(m), "sha256": c.sha256_file(m)}))
            for r in rows:
                self.rows[r["id"]] = r
        elif osce == "dropped":
            (cd / "status.json").write_text(json.dumps({"status": "dropped"}))

    def candidate(self, **kw) -> dict:
        from labeling import config
        args = dict(source_files=self.files, mean_seconds=MEAN_SECONDS, hf_sources=HF_SOURCES, reserved={"yA"},
                    url_fn=self.url)
        args.update(kw)
        return config.build_candidate(self.pool, self.corpora, **args)

    def pinned(self, workers: int = 1, batch_seconds: float = 10.0, **kw) -> dict:
        from labeling import config
        cand = self.candidate(**kw)
        cand["batch"]["batch_seconds"] = batch_seconds
        return config.pin(cand, workers)


def recording(item_id: str) -> str:
    """The recording a fixture id belongs to."""
    native = item_id.split(":", 1)[1]
    if native.endswith(".flac"):
        return native.split("_")[0]
    return native.rsplit("-", 1)[0]


@contextlib.contextmanager
def relabel_root(path: Path):
    old = os.environ.get("RELABEL_ROOT")
    os.environ["RELABEL_ROOT"] = str(path)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("RELABEL_ROOT", None)
        else:
            os.environ["RELABEL_ROOT"] = old


class FakeStream:
    """A deterministic resumable item stream (state: position) for batching and replay tests.
    conflict_at: an occurrence index whose item reuses an earlier id with different audio."""

    def __init__(self, n_ids: int = 37, seed: int = 0, conflict_at: int | None = None, durations=None):
        self.n_ids, self.seed, self.pos, self.conflict_at = n_ids, seed, 0, conflict_at
        self.durations = durations
        self.config_hash = f"fake-{n_ids}-{seed}"

    def state_dict(self) -> dict:
        return {"pos": self.pos, "config_hash": self.config_hash}

    def load_state_dict(self, state: dict) -> None:
        if state["config_hash"] != self.config_hash:
            raise ValueError("other configuration")
        self.pos = state["pos"]

    def item(self, i: int) -> dict:
        k = (i * 7 + self.seed) % self.n_ids
        if self.durations:
            d = self.durations(k)
        elif k % 11 == 3:
            d = 0.4 if k % 2 else 31.0          # outside 1-30 s: dropped before bucketing
        else:
            d = 1.0 + (k * 13 % 50) / 10.0
        audio_seed = 999 if self.conflict_at is not None and i == self.conflict_at else k
        rng = np.random.default_rng([audio_seed, 5])
        pcm = rng.integers(-2000, 2000, max(1, int(d * SR)), dtype=np.int16)
        return {"audio": pcm.astype(np.float32) / 32768.0, "duration": pcm.size / SR, "text": f"text {k}",
                "id": f"fake:{k}", "source": "fake" if k % 3 else "fake-b", "reference": f"text {k}",
                "reference_kind": "human" if k % 2 else "pseudo"}

    def __iter__(self):
        while True:
            it = self.item(self.pos)
            self.pos += 1
            yield it

    def close(self) -> None:
        pass

    def counters(self) -> dict:
        return {"items": self.pos}
