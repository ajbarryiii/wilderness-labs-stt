"""Build contracts (labeling plan Part A): manifests in the existing per-utterance schema plus "part",
16 kHz mono audio stored under the dev-set root, the two channels of a consultation kept as separate
utterances, per-part counts, hours and drops by reason, the source revision in the artifact metadata,
test-set recordings excluded, resumable builds, and smoke builds that never touch production paths.
The aligner's device: a build on the GPU (--device cuda, holding the parakeet GPU lock) resumes recordings
built on the CPU (the device is not part of the segmentation config); each recording's cache records the
device of the build that segmented it ("aligner_device"; caches without it were built on the CPU), and the
meta counts recordings per device ("aligner_devices") and names the aligner without a device.
Synthetic sources in a temporary $RELABEL_ROOT; a stand-in aligner; no network.

Run from finetune/relabel:  ./python -m unittest discover -s tests -p "test_devsets_*" -v
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import types
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

import score_teacher
from devsets import build, fetch, layout, segment, sources

REAL_ROOT = layout.root("RELABEL_ROOT")
RECORD_KEYS = {"audio_filepath", "duration", "text", "id", "source", "part"}


class EvenAligner:
    """Spreads the units evenly over the audio (0.9 score); counts calls."""

    min_score = 0.5

    def __init__(self, device: str | None = None) -> None:
        self.calls = 0
        if device is not None:
            self.device = device

    def align(self, audio, units):
        from devsets.align import UnitSpan

        self.calls += 1
        seconds = len(audio) / segment.SAMPLE_RATE
        step = seconds / (len(units) + 1)
        return [UnitSpan((i + 0.5) * step, (i + 1.2) * step, None if u is None else 0.9) for i, u in enumerate(units)]


class Refusing:
    min_score = 0.5

    def align(self, audio, units):
        raise AssertionError("the aligner must not run on a resumed build")


def tone(seconds: float, rate: int = 16000, channels: int = 1, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    x = 0.1 * np.sin(2 * np.pi * 220 * t) + 0.01 * rng.standard_normal(len(t))
    return np.stack([x] * channels, axis=1).astype(np.float32) if channels > 1 else x.astype(np.float32)


def flac_bytes(audio: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, rate, format="FLAC")
    return buf.getvalue()


def textgrid(intervals: list[tuple[float, float, str]], total: float) -> str:
    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", "xmin = 0", f"xmax = {total}",
             "tiers? <exists>", "size = 1", "item []:", "    item [1]:", '        class = "IntervalTier"',
             '        name = "Speaker"', "        xmin = 0", f"        xmax = {total}",
             f"        intervals: size = {len(intervals)}"]
    for i, (a, b, text) in enumerate(intervals, 1):
        lines += [f"        intervals [{i}]:", f"            xmin = {a}", f"            xmax = {b}",
                  f'            text = "{text}"']
    return "\n".join(lines) + "\n"


class BuildFixture(unittest.TestCase):
    def setUp(self) -> None:
        layout.require_mount()
        self.tmp = REAL_ROOT / "tmp" / "tests" / f"devsets-build-{uuid.uuid4().hex[:8]}"
        self.data = self.tmp / "data"
        self.data.mkdir(parents=True)
        self.env = mock.patch.dict(os.environ, {"RELABEL_ROOT": str(self.tmp / "root")})
        self.env.start()
        self.lay = layout.Layout.production()
        self.assertTrue(self.lay.base.is_relative_to(self.tmp))

    def tearDown(self) -> None:
        self.env.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def register(self, part: str, entry: dict, files: list[Path]) -> dict:
        entry = {"origin": "hf", "repo": f"test/{part}", "revision": "abc123", "files": ["*"], "license": "test",
                 "acquisition": "test", **entry}
        record = {"alias": part, "repo": entry["repo"], "revision": entry["revision"],
                  "files": [{"path": f.name, "local": str(f), "bytes": f.stat().st_size,
                             "sha256": fetch.sha256_file(f), "lfs": False} for f in files]}
        build.write_json(self.lay.fetch_record(part), record)
        return entry

    def channels(self) -> dict:
        audio_dir, grid_dir = self.data / "audio", self.data / "transcripts"
        audio_dir.mkdir()
        grid_dir.mkdir()
        files = []
        for stem, intervals in {
            "c01_doctor": [(0, 1.0, ""), (1.0, 4.0, "Hello, how can I help?"), (5.0, 5.5, "Okay."),
                           (6.0, 9.0, "Any <UNSURE>chest</UNSURE> pain?"), (9.0, 12.0, "")],
            "c01_patient": [(0, 4.2, ""), (4.2, 5.8, "Morning."), (10.0, 11.0, "<UNIN/>"), (11.0, 14.0, "Yes, a bit.")],
        }.items():
            sf.write(audio_dir / f"{stem}.wav", tone(15, seed=len(stem)), 16000, subtype="PCM_16")
            (grid_dir / f"{stem}.TextGrid").write_text(textgrid(intervals, 15.0))
            files += [audio_dir / f"{stem}.wav", grid_dir / f"{stem}.TextGrid"]
        return self.register("med-dev-1", {"origin": "github", "layout": "textgrid-channels", "audio_dir": "audio",
                                           "transcript_dir": "transcripts"}, files)

    def longform(self) -> dict:
        rows = [("1001", flac_bytes(tone(40, 24000, 2, 1), 24000), "Good morning everyone. " * 4 +
                 "Revenue grew nine percent. " * 4 + "Thank you."),
                ("1002", flac_bytes(tone(20, 16000, 1, 2), 16000), "This call is in the test set."),
                ("1001", flac_bytes(tone(10, 16000, 1, 3), 16000), "A duplicate id."),
                ("1003", b"not audio", "Unreadable.")]
        path = self.data / "full-00000.parquet"
        pq.write_table(pa.table({"file_id": [r[0] for r in rows],
                                 "audio": [{"bytes": r[1], "path": None} for r in rows],
                                 "transcription": [r[2] for r in rows]}), path)
        readme = self.data / "README.md"  # fetched documentation next to the data (registry "files")
        readme.write_text("# card\n")
        return self.register("tech-dev-2", {"layout": "parquet-longform", "id_column": "file_id",
                                            "audio_column": "audio", "text_column": "transcription",
                                            "exclude_test_set": "earnings22"}, [path, readme])

    def manifest(self, lay, part) -> list[dict]:
        return [json.loads(line) for line in lay.manifest(part).read_text().splitlines()]


class ChannelsTest(BuildFixture):
    def test_schema_audio_channels_counts_and_revision(self) -> None:
        entry = self.channels()
        meta = build.build_part("med-dev-1", entry, self.lay, EvenAligner(), None)
        records = self.manifest(self.lay, "med-dev-1")
        self.assertEqual([r["id"] for r in records], ["med-dev-1:c01_doctor-0000", "med-dev-1:c01_doctor-0002",
                                                      "med-dev-1:c01_patient-0000", "med-dev-1:c01_patient-0002"])
        for r in records:
            self.assertEqual(set(r), RECORD_KEYS)
            self.assertEqual((r["source"], r["part"]), ("med-dev-1", "med-dev-1"))
            path = Path(r["audio_filepath"])
            self.assertTrue(path.is_relative_to(self.lay.audio("med-dev-1")))
            info = sf.info(path)
            self.assertEqual((info.samplerate, info.channels), (16000, 1))
            self.assertAlmostEqual(info.frames / 16000, r["duration"])
            self.assertTrue(1.0 <= r["duration"] <= 30.0)
            channel = r["id"].split(":")[1].rsplit("-", 1)[0]
            self.assertEqual(path.parent.name, channel)  # each channel is its own recording
        by_id = {r["id"]: r for r in records}
        self.assertEqual(by_id["med-dev-1:c01_doctor-0002"]["text"], "Any chest pain?")
        self.assertAlmostEqual(by_id["med-dev-1:c01_patient-0000"]["duration"], 1.6)
        self.assertEqual(meta["utterances"], 4)
        self.assertAlmostEqual(meta["hours"], sum(r["duration"] for r in records) / 3600)
        self.assertEqual(meta["segments_dropped"]["too_short"]["count"], 1)
        self.assertEqual(meta["segments_dropped"]["empty_reference"]["count"], 1)
        self.assertEqual(meta["source"]["revision"], "abc123")
        self.assertEqual(meta["recordings"]["used"], 2)
        self.assertEqual(meta["manifest_sha256"], fetch.sha256_file(self.lay.manifest("med-dev-1")))
        self.assertTrue(meta["complete"])


class LongformTest(BuildFixture):
    def test_resampled_aligned_segments_and_recording_drops(self) -> None:
        entry = self.longform()
        aligner = EvenAligner()
        with mock.patch.object(sources, "test_set_recordings", return_value={"1002", "9999"}):
            meta = build.build_part("tech-dev-2", entry, self.lay, aligner, None)
        records = self.manifest(self.lay, "tech-dev-2")
        self.assertTrue(records and all(r["id"].startswith("tech-dev-2:1001-") for r in records))
        self.assertEqual(meta["recordings"]["dropped"],
                         {"test_overlap": 1, "duplicate_recording": 1, "unreadable_audio": 1})
        self.assertEqual(meta["recordings"]["used"], 1)
        self.assertAlmostEqual(meta["recordings"]["audio_hours"], 40 / 3600, places=4)
        self.assertEqual(meta["aligned_segments"], len(records))
        for r in records:
            info = sf.info(r["audio_filepath"])
            self.assertEqual((info.samplerate, info.channels), (16000, 1))
            self.assertTrue(1.0 <= r["duration"] <= 30.0)
        self.assertEqual(aligner.calls, 1)

    def test_resume_reuses_finished_recordings_and_config_change_recomputes(self) -> None:
        entry = self.longform()
        with mock.patch.object(sources, "test_set_recordings", return_value=set()):
            build.build_part("tech-dev-2", entry, self.lay, EvenAligner(), None)
            first = self.lay.manifest("tech-dev-2").read_bytes()
            build.build_part("tech-dev-2", entry, self.lay, Refusing(), None)
            self.assertEqual(self.lay.manifest("tech-dev-2").read_bytes(), first)
            again = EvenAligner()
            with mock.patch.object(segment, "TARGET_SECONDS", 8.0):
                build.build_part("tech-dev-2", entry, self.lay, again, None)
            self.assertGreater(again.calls, 0)
            self.assertNotEqual(self.lay.manifest("tech-dev-2").read_bytes(), first)


class VariantTest(BuildFixture):
    def test_acoustic_variants_reuse_the_reference_cuts(self) -> None:
        script = "Chart number one two three period " * 3 + "The patient is well period " * 3
        clean = tone(40, seed=5)
        padded = np.concatenate([clean, np.zeros(31, dtype=np.float32)])  # an MP3 decode's trailing frame padding
        rows = [("n0-clean", clean, script), ("n0-ward", clean + tone(40, seed=6), script), ("n0-mp3_low", padded, script),
                ("n0-fast", tone(32, seed=7), script), ("n0-hard", tone(32, seed=8), script),
                ("n1-clean", tone(40, seed=9), script), ("n1-ward", tone(41, seed=10), script),
                ("n2-ward", tone(40, seed=11), script)]
        path = self.data / "dev-00000.parquet"
        pq.write_table(pa.table({"id": [r[0] for r in rows],
                                 "audio": [{"bytes": flac_bytes(r[1], 16000), "path": None} for r in rows],
                                 "script": [r[2] for r in rows]}), path)
        entry = self.register("med-dev-2", {"layout": "parquet-longform", "id_column": "id", "audio_column": "audio",
                                            "text_column": "script", "boundary_words": ["period"],
                                            "timing_from": {"pattern": r"(?P<base>n\d+)-(?P<variant>\w+)",
                                                            "template": "{base}-{variant}",
                                                            "reference": {"ward": "clean", "mp3_low": "clean",
                                                                          "hard": "fast"}}}, [path])
        aligner = EvenAligner()
        meta = build.build_part("med-dev-2", entry, self.lay, aligner, None)
        self.assertEqual(aligner.calls, 5)  # n0-clean, n0-fast, n1-clean, n1-ward (length differs), n2-ward
        self.assertEqual(meta["recordings_with_reference_cuts"], 3)
        records = self.manifest(self.lay, "med-dev-2")
        cuts = {}
        for r in records:
            rec, k = r["id"].split(":")[1].rsplit("-", 1)
            cuts.setdefault(rec, {})[k] = (r["text"], r["duration"])
        self.assertEqual(cuts["n0-ward"], cuts["n0-clean"])
        self.assertEqual(cuts["n0-mp3_low"], cuts["n0-clean"])  # within the sample tolerance
        self.assertEqual(cuts["n0-hard"], cuts["n0-fast"])
        clean = json.loads((self.lay.work("med-dev-2") / "n0-clean.json").read_text())
        ward = json.loads((self.lay.work("med-dev-2") / "n0-ward.json").read_text())
        self.assertEqual(ward["timing_from"], "n0-clean")
        self.assertEqual([(s["start"], s["end"]) for s in ward["segments"]],
                         [(s["start"], s["end"]) for s in clean["segments"]])
        self.assertIsNone(json.loads((self.lay.work("med-dev-2") / "n1-ward.json").read_text())["timing_from"])
        self.assertEqual(meta["by_variant"]["ward"]["recordings"], 3)
        self.assertEqual(sum(v["utterances"] for v in meta["by_variant"].values()), meta["utterances"])


class DeviceTest(BuildFixture):
    def variants(self) -> dict:
        script = "Chart number one two three period " * 3 + "The patient is well period " * 3
        rows = [(f"n{k}-clean", tone(30, seed=20 + k), script) for k in range(4)]
        path = self.data / "dev-00000.parquet"
        pq.write_table(pa.table({"id": [r[0] for r in rows],
                                 "audio": [{"bytes": flac_bytes(r[1], 16000), "path": None} for r in rows],
                                 "script": [r[2] for r in rows]}), path)
        return self.register("med-dev-2", {"layout": "parquet-longform", "id_column": "id", "audio_column": "audio",
                                           "text_column": "script", "boundary_words": ["period"]}, [path])

    def test_gpu_build_resumes_cpu_recordings_and_counts_devices(self) -> None:
        entry = self.variants()
        cpu = EvenAligner("cpu")
        build.build_part("med-dev-2", entry, self.lay, cpu, 2)   # the first two recordings, on the CPU
        self.assertEqual(cpu.calls, 2)
        work = self.lay.work("med-dev-2")
        first = {p.name: p.read_bytes() for p in work.glob("*.json")}
        self.assertEqual(len(first), 2)
        gpu = EvenAligner("cuda")
        gpu._aligner = types.SimpleNamespace(identity={"aligner": "even", "sha256": "abc", "device": "cuda"})
        meta = build.build_part("med-dev-2", entry, self.lay, gpu, None)
        self.assertEqual(gpu.calls, 2)   # only the two recordings not built yet
        self.assertEqual({p.name: p.read_bytes() for p in work.glob("*.json") if p.name in first}, first)
        devices = {p.stem: json.loads(p.read_text())["aligner_device"] for p in work.glob("*.json")}
        self.assertEqual(devices, {"n0-clean": "cpu", "n1-clean": "cpu", "n2-clean": "cuda", "n3-clean": "cuda"})
        self.assertEqual(meta["aligner_devices"], {"cpu": 2, "cuda": 2})
        self.assertEqual(meta["aligner"], {"aligner": "even", "sha256": "abc"})
        # A cache written before devices were recorded counts as a CPU build; nothing is re-aligned.
        legacy = work / "n0-clean.json"
        cache = json.loads(legacy.read_text())
        del cache["aligner_device"]
        legacy.write_text(json.dumps(cache))
        meta = build.build_part("med-dev-2", entry, self.lay, Refusing(), None)
        self.assertEqual(meta["aligner_devices"], {"cpu": 2, "cuda": 2})

    def test_cli_device_and_gpu_lock(self) -> None:
        def run(argv: list[str]) -> tuple[list, mock.MagicMock]:
            events: list = []

            @contextlib.contextmanager
            def lock(label, wait_s, *a, **k):
                events.append(("lock", float(wait_s)))
                yield
                events.append(("unlock",))

            with mock.patch.object(score_teacher, "gpu_lock", lock), \
                    mock.patch.object(build, "build_part", side_effect=lambda part, *a, **k: events.append(("build", part))), \
                    mock.patch.object(build.align, "LazyAligner") as lazy, \
                    mock.patch.object(build.registry, "load", return_value={}), \
                    mock.patch.object(build.registry, "source", return_value={}):
                self.assertEqual(build.main(argv), 0)
            return events, lazy

        events, lazy = run(["build", "--parts", "med-dev-2", "lectures-a", "--device", "cuda", "--lock-wait", "7"])
        self.assertEqual(events, [("lock", 7.0), ("build", "med-dev-2"), ("build", "lectures-a"), ("unlock",)])
        self.assertEqual(lazy.call_args.kwargs.get("device"), "cuda")
        events, lazy = run(["build", "--parts", "med-dev-2"])
        self.assertEqual(events, [("build", "med-dev-2")])
        self.assertEqual(lazy.call_args.kwargs.get("device", "cpu"), "cpu")


class SmokeTest(BuildFixture):
    def test_smoke_build_never_writes_production_paths(self) -> None:
        entry = self.channels()
        smoke = layout.Layout.smoke(1)
        self.assertFalse(smoke.base.is_relative_to(layout.production_root()))
        meta = build.build_part("med-dev-1", entry, smoke, EvenAligner(), 1)
        self.assertEqual(meta["recordings"]["selected"], 1)
        self.assertTrue(smoke.manifest("med-dev-1").exists())
        for sub in ("audio", "manifests", "work", "scores"):
            self.assertFalse((layout.production_root() / sub).exists(), sub)

    def test_build_requires_the_pinned_fetch(self) -> None:
        entry = self.channels()
        with self.assertRaises(SystemExit):
            build.build_part("med-dev-1", {**entry, "revision": "other"}, self.lay, EvenAligner(), None)


class FetchHelpersTest(unittest.TestCase):
    def test_lfs_pointer_and_blob_hash(self) -> None:
        pointer = (b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64 + b"\nsize 1234\n")
        self.assertEqual(fetch.parse_lfs_pointer(pointer), ("a" * 64, 1234))
        self.assertIsNone(fetch.parse_lfs_pointer(b"File type = TextGrid"))
        self.assertEqual(fetch.git_blob_sha1(b"hello\n"), "ce013625030ba8dba906f756967f9e9ca394464a")
        self.assertTrue(fetch.matches("audio/x.wav", ["audio/*.wav"]))
        self.assertFalse(fetch.matches("notes/x.json", ["audio/*.wav", "transcripts/*.TextGrid"]))


class TextGridTest(unittest.TestCase):
    def test_parse(self) -> None:
        text = textgrid([(0, 1.5, ""), (1.5, 3.25, 'He said ""stop"".'), (3.25, 4, " ")], 4.0)
        self.assertEqual(sources.parse_textgrid(text), [segment.Utterance(1.5, 3.25, 'He said "stop".')])


if __name__ == "__main__":
    unittest.main()
