"""Unit tests for stream.py on synthetic local parquet sources (no network)."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402
import stream  # noqa: E402

DURATIONS = (0.5, 0.999, 1.0, 2.0, 30.0, 30.5)  # seconds; 0.5, 0.999 and 30.5 must be dropped


def _wav(seconds: float, seed: int) -> bytes:
    n = int(round(seconds * 16000))
    x = (np.random.default_rng(seed).standard_normal(n) * 1000).astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, x, 16000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _write_source(root: Path, name: str, files: int, rows: int, durations=(2.0,), group=None) -> list[str]:
    out = []
    for f in range(files):
        ids = [f"{name}-{f:02d}-{r:03d}" for r in range(rows)]
        durs = [durations[(f + r) % len(durations)] for r in range(rows)]
        table = pa.table({
            "id": ids,
            "text": [f"text {i}" for i in ids],
            "audio": [{"bytes": _wav(d, hash(i) % 2**32), "path": f"{i}.wav"} for i, d in zip(ids, durs)],
            "group": [group(f, r) if group else f"rec-{f}" for r in range(rows)],
        })
        path = root / f"{name}-{f:02d}.parquet"
        pq.write_table(table, path, row_group_size=7)
        out.append(str(path))
    return out


class StreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.files = {
            "a": _write_source(root, "a", 4, 30),
            "b": _write_source(root, "b", 3, 30),
            "c": _write_source(root, "c", 2, 20),
            "f": _write_source(root, "f", 2, 18, DURATIONS),
            "g": _write_source(root, "g", 2, 20, group=lambda f, r: "dev-rec" if r % 4 == 0 else f"rec-{f}"),
        }

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def cfg(self, names=("a", "b", "c"), shares=(0.65, 0.25, 0.10), **kw) -> stream.StreamConfig:
        sources = tuple(stream.SourceSpec(n, tuple(self.files[n]), "id", "text", s, slots=2, group_col="group")
                        for n, s in zip(names, shares))
        kw.setdefault("weighting", "utterances")
        kw.setdefault("batch_rows", 4)
        kw.setdefault("retry_base_seconds", 0.0)
        return stream.StreamConfig(sources=sources, seed=123, **kw)

    @staticmethod
    def take(s: stream.TrainingStream, n: int) -> list[dict]:
        out = []
        for item in s:
            out.append(item)
            if len(out) == n:
                break
        return out

    def test_item_format(self):
        item = self.take(stream.TrainingStream(self.cfg(), num_workers=0), 1)[0]
        self.assertEqual(set(item), {"audio", "duration", "text", "id", "source"})
        self.assertEqual(item["audio"].dtype, np.float32)
        self.assertEqual(item["audio"].ndim, 1)
        self.assertAlmostEqual(item["duration"], item["audio"].size / 16000)
        self.assertTrue(item["id"].startswith(item["source"] + ":"))
        cfg = stream.StreamConfig(sources=(stream.SourceSpec("a", tuple(self.files["a"]), "id", None, 1.0),),
                                  weighting="utterances")
        self.assertIsNone(self.take(stream.TrainingStream(cfg, num_workers=0), 1)[0]["text"])

    def test_interleave_proportions(self):
        s = stream.TrainingStream(self.cfg(), num_workers=0)
        items = self.take(s, 3000)
        frac = {n: sum(i["source"] == n for i in items) / len(items) for n in "abc"}
        for n, p in zip("abc", (0.65, 0.25, 0.10)):
            self.assertAlmostEqual(frac[n], p, delta=0.03, msg=frac)
        c = s.counters()
        self.assertEqual(c["items"], 3000)
        self.assertEqual(sum(c["sources"][n]["kept"] for n in "abc"), 3000)
        self.assertGreater(c["sources"]["c"]["passes_started"], 1)  # small source repeats

    def test_hours_weighting(self):
        cfg = self.cfg(weighting="hours", mean_seconds=(("a", 2.0), ("b", 4.0), ("c", 8.0)))
        p = cfg.probabilities()
        expected = np.array([0.65 / 2, 0.25 / 4, 0.10 / 8])
        np.testing.assert_allclose(p, expected / expected.sum())

    def test_duration_filter(self):
        s = stream.TrainingStream(self.cfg(("f",), (1.0,)), num_workers=0)
        items = self.take(s, 200)
        durations = {round(i["duration"], 3) for i in items}
        self.assertEqual(durations, {1.0, 2.0, 30.0})
        dropped = s.counters()["sources"]["f"]["dropped"]
        self.assertGreater(dropped["too_short"], 0)
        self.assertGreater(dropped["too_long"], 0)
        self.assertEqual(set(dropped), {"too_short", "too_long"})

    def test_reserved_exclusion(self):
        names = [f"data/en{i // 50:03d}/asr_only/{i % 50:08d}.parquet" for i in range(500)]
        reserved = [3, 77, 401]
        keep = stream.yodas_training_files(list(reversed(names)), reserved)
        self.assertEqual(len(keep), 497)
        self.assertFalse({sorted(names)[i] for i in reserved} & set(keep))
        cfg = self.cfg(("g",), (1.0,), exclude_groups=("dev-rec",))
        s = stream.TrainingStream(cfg, num_workers=0)
        items = self.take(s, 100)
        self.assertFalse(any(i["id"].endswith(("-000", "-004", "-008")) for i in items))
        self.assertGreater(s.counters()["sources"]["g"]["dropped"]["dev_recording_overlap"], 0)

    def test_deterministic(self):
        a = [i["id"] for i in self.take(stream.TrainingStream(self.cfg(), num_workers=0), 200)]
        b = [i["id"] for i in self.take(stream.TrainingStream(self.cfg(), num_workers=0), 200)]
        self.assertEqual(a, b)
        other = stream.StreamConfig(**{**self.cfg().__dict__, "seed": 124})
        c = [i["id"] for i in self.take(stream.TrainingStream(other, num_workers=0), 200)]
        self.assertNotEqual(a, c)

    def _resume(self, workers: int):
        s1 = stream.TrainingStream(self.cfg(), num_workers=workers)
        self.take(s1, 137)
        state = json.loads(json.dumps(s1.state_dict()))  # must survive JSON
        expected = self.take(s1, 60)
        s1.close()
        s2 = stream.TrainingStream(self.cfg(), num_workers=workers)
        s2.load_state_dict(state)
        got = self.take(s2, 60)
        s2.close()
        self.assertEqual([i["id"] for i in got], [i["id"] for i in expected])
        for x, y in zip(got, expected):
            np.testing.assert_array_equal(x["audio"], y["audio"])
        self.assertEqual(s2.state_dict()["items"], 197)

    def test_state_round_trip_in_process(self):
        self._resume(0)

    def test_state_round_trip_two_workers(self):
        self._resume(2)

    def test_state_rejects_other_config(self):
        s = stream.TrainingStream(self.cfg(), num_workers=0)
        self.take(s, 5)
        other = stream.TrainingStream(self.cfg(shares=(0.5, 0.4, 0.1)), num_workers=0)
        with self.assertRaises(ValueError):
            other.load_state_dict(s.state_dict())
        with self.assertRaises(ValueError):
            stream.TrainingStream(self.cfg(), num_workers=2).load_state_dict(s.state_dict())
        # Reader settings that change positions or were changed by the 2026-10-01 memory fix.
        fewer_slots = self.cfg()
        fewer_slots = stream.StreamConfig(**{**fewer_slots.__dict__, "sources": tuple(
            stream.SourceSpec(**{**src.__dict__, "slots": 1}) for src in fewer_slots.sources)})
        for cfg in (fewer_slots, self.cfg(block_size=32 * 2**20), self.cfg(mean_seconds=(("a", 1.0),))):
            with self.assertRaises(ValueError):
                stream.TrainingStream(cfg, num_workers=0).load_state_dict(s.state_dict())
        stream.TrainingStream(self.cfg(), num_workers=0).load_state_dict(s.state_dict())  # same config: fine

    def test_transient_errors_are_retried(self):
        real = stream._open_file
        calls = {"n": 0}

        def flaky(url, block_size):
            calls["n"] += 1
            if calls["n"] in (1, 2):
                raise OSError("simulated network error")
            return real(url, block_size)

        with mock.patch.object(stream, "_open_file", flaky), mock.patch.object(stream, "_log"):
            s = stream.TrainingStream(self.cfg(), num_workers=0)
            items = self.take(s, 50)
        self.assertEqual(len(items), 50)
        self.assertEqual(s.counters()["retries"], 2)

    def test_persistent_failure_names_source(self):
        def broken(url, block_size):
            if "/b-" in url:
                raise OSError("simulated outage")
            return open(url, "rb")

        with mock.patch.object(stream, "_open_file", broken), mock.patch.object(stream, "_log"):
            s = stream.TrainingStream(self.cfg(retry_attempts=3), num_workers=0)
            with self.assertRaises(stream.StreamError) as ctx:
                self.take(s, 500)
        self.assertEqual(ctx.exception.source, "b")

    def test_phase_validated(self):
        with self.assertRaises(ValueError):
            stream.make_training_stream("final")


def _rss_mb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("no VmRSS")


class MemoryRegressionTest(unittest.TestCase):
    """A worker reading one ~200 MB parquet file end to end must not hold what it has read.

    pyarrow's default pre_buffer=True kept every column chunk read so far alive while the file
    was open (the 2026-09-30 OOM); stream.py opens files with pre_buffer=False. The same read
    with pre_buffer forced back on is the control that shows the bound discriminates.
    Measured 2026-10-01 on a 205 MB file: +10 MB without pre-buffering, +92 MB with it.
    Takes ~2 s and ~0.6 GB peak; writes a temporary file under ARTIFACTS/tmp.
    """
    BOUND_MB = 45.0
    ROWS, ROW_SECONDS, ROW_GROUP = 640, 10.0, 100  # 640 x 320 kB WAV = ~205 MB, 7 row groups

    @classmethod
    def setUpClass(cls):
        tmp_root = paths.ARTIFACTS / "tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        cls._tmp = tempfile.TemporaryDirectory(dir=tmp_root, prefix="test-stream-mem-")
        cls.path = Path(cls._tmp.name) / "big.parquet"
        schema = pa.schema([("id", pa.string()), ("text", pa.string()),
                            ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())]))])
        with pq.ParquetWriter(cls.path, schema) as writer:
            for start in range(0, cls.ROWS, cls.ROW_GROUP):
                ids = [f"big-{i:04d}" for i in range(start, min(cls.ROWS, start + cls.ROW_GROUP))]
                writer.write_table(pa.table({"id": ids, "text": ids, "audio": [
                    {"bytes": _wav(cls.ROW_SECONDS, i), "path": f"{i}.wav"} for i in range(len(ids))]},
                    schema=schema))
        cls.size_mb = cls.path.stat().st_size / 1e6

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _growth(self) -> float:
        cfg = stream.StreamConfig(sources=(stream.SourceSpec("big", (str(self.path),), "id", "text", 1.0,
                                                             slots=1),), weighting="utterances")
        s = stream.TrainingStream(cfg, num_workers=0)
        it = iter(s)
        next(it)  # file open, first batch read
        base = peak = _rss_mb()
        for n, _ in enumerate(it, 2):
            if n % 20 == 0:
                peak = max(peak, _rss_mb())
            if n >= self.ROWS - 5:  # stay inside the first pass over the file
                break
        s.close()
        return peak - base

    def test_rss_bounded_without_pre_buffer(self):
        import pyarrow.parquet as pqm
        growth = self._growth()
        real = pqm.ParquetFile
        with mock.patch.object(pqm, "ParquetFile", lambda *a, **k: real(*a, **{**k, "pre_buffer": True})):
            control = self._growth()
        print(f"\n[memory] file {self.size_mb:.0f} MB: RSS growth {growth:.0f} MB with pre_buffer=False, "
              f"{control:.0f} MB with pre_buffer=True (bound {self.BOUND_MB:.0f} MB)", file=sys.stderr)
        self.assertLess(growth, self.BOUND_MB)
        self.assertGreater(control, self.BOUND_MB, "control did not grow; the test no longer discriminates")


if __name__ == "__main__":
    unittest.main()
