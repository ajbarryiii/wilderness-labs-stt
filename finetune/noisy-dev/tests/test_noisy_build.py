"""Build contracts on a synthetic end-to-end build (plans/build.md "Code and tests"):

1. Determinism: byte-identical FLACs and manifest rows with 1 and 4 workers and after an
   interrupted build resumed from its manifest.
2. Added-noise ratios within 0.1 dB of target, measured from the written FLACs; event window
   levels within 0.1 dB; event count and placement rules; the babble recipe.
5. The blocklist written by the build covers every source file the manifest references.
8. Each condition has exactly the dev sets' rows, IDs and clean references; short or silent
   utterances are flagged and keep their clean audio.
9. Adding the rotor conditions changes no other condition's files or rows.

Review round 2: the scorer validates this real builder output (build.json schema, lock, manifest
and audio hashes, expected paths); pool inputs are re-verified on every build and resume (a
mutated source after an interruption is refused); a corrupted completed output is redone; babble
talker normalization is measured independently; outputs are 24-bit; rows carry noise-source groups.

Review round 3: a completed manifest whose digest differs from build.json is refused, journal rows
are re-validated against the locked inputs (references, clusters, expected paths) and redone if
they differ, and the field condition and the impulsive events (truncated ones included) are
reconstructed independently from the recorded draws.
"""
from __future__ import annotations

import dataclasses
import json
import math
import shutil
import unittest
from collections import defaultdict
from pathlib import Path

import noisy_fixtures as fx
import numpy as np

import noisy_build
import noisy_paths
import noisy_score
import noisy_sources

SR = fx.SR
NOISE_CONDITIONS = ("eng10", "eng0", "wind10", "wind0", "crowd10", "crowd0", "amb10", "amb0", "bab10", "bab0")
BASE_CONDITIONS = ("clean", "gun0", "gun10", "eng10", "eng0", "wind10", "wind0", "bab10", "bab0", "crowd10",
                   "crowd0", "amb10", "amb0", "rev", "radio", "field")


def power_db(x) -> float:
    return 10 * math.log10(float(np.mean(np.square(x))))


def active_frames(x, gate_db=30.0, frame=320):
    """Independent re-implementation of the 20 ms, -30 dB-relative activity gate."""
    n = int(math.ceil(len(x) / frame))
    pad = np.zeros(n * frame)
    pad[: len(x)] = x
    energy = np.mean(pad.reshape(n, frame) ** 2, axis=1)
    with np.errstate(divide="ignore"):
        db = 10 * np.log10(energy)
    return db >= db.max() - gate_db


class BuildContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = fx.fresh_dir("build")
        cls.manifests = fx.make_dev_sets(cls.root / "dev")
        cls.pool_path = fx.make_pool(cls.root / "pool", rotor=True)
        cls.rotor_path = cls.root / "pool" / "rotor.json"
        cls.pool = noisy_sources.load_pool(cls.pool_path)
        cls.outs = {}
        for name, workers, rotor in (("w1", 1, False), ("w4", 4, False), ("rotor", 2, True)):
            cls.outs[name] = cls.root / name
            noisy_build.build(cls.config(cls.outs[name], workers, rotor))
        # An interrupted build (stopped after 6 utterances), then resumed with another worker count.
        cls.outs["resumed"] = cls.root / "resumed"
        try:
            noisy_build.build(cls.config(cls.outs["resumed"], 1, False, stop_after=6))
        except noisy_build.BuildInterrupted:
            pass
        else:
            raise AssertionError("stop_after did not interrupt the build")
        partial = cls.outs["resumed"] / "build_manifest.partial.jsonl"
        assert partial.exists()
        with open(partial, "a") as handle:
            handle.write('{"condition": "gun0", "trunc')  # a torn last line, as after a crash
        noisy_build.build(cls.config(cls.outs["resumed"], 3, False))
        cls.rows = fx.build_rows(cls.outs["w1"])
        cls.dev = {name: [json.loads(line) for line in (cls.manifests / f"{name}.jsonl").read_text().splitlines()]
                   for name in fx.DEV_IDS}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    @classmethod
    def config(cls, out: Path, workers: int, rotor: bool, stop_after: int | None = None):
        raw, verify = fx.source_dirs(cls.pool_path)
        return noisy_build.BuildConfig(out=out, manifests=cls.manifests, pool=cls.pool_path,
                                       rotor_pool=cls.rotor_path if rotor else None, workers=workers,
                                       voxpopuli_speakers=fx.VOX_SPEAKERS, stop_after=stop_after,
                                       raw=raw, verify_dir=verify, registry=fx.registry(cls.pool_path))

    def rows_of(self, condition: str, rows=None) -> list[dict]:
        return [r for r in (rows or self.rows) if r["condition"] == condition]

    def clean(self, row: dict) -> np.ndarray:
        src = next(r for r in self.dev[row["set"]] if r["id"] == row["id"])
        return fx.read_pcm(src["audio_filepath"])

    def written(self, out: Path, row: dict) -> np.ndarray:
        return fx.read_pcm(out / row["audio"])

    # 1 ---------------------------------------------------------------------------------------
    def test_byte_identical_across_workers_and_resume(self):
        reference = fx.flac_files(self.outs["w1"])
        manifest = (self.outs["w1"] / "build_manifest.jsonl").read_bytes()
        self.assertTrue(reference)
        for name in ("w4", "resumed"):
            self.assertEqual(fx.flac_files(self.outs[name]), reference, name)
            self.assertEqual((self.outs[name] / "build_manifest.jsonl").read_bytes(), manifest, name)
            self.assertFalse((self.outs[name] / "build_manifest.partial.jsonl").exists(), name)
        for row in self.rows:
            if row["condition"] != "clean":
                self.assertEqual(noisy_build.sha256_file(self.outs["w1"] / row["audio"]), row["sha256"])

    def test_lock_freezes_inputs(self):
        lock = json.loads((self.outs["w1"] / "build.lock.json").read_text())
        self.assertEqual(lock["build_seed"], 20261016)
        self.assertEqual(lock["rng"], "numpy.random.Generator(PCG64(int(SHA-256(utterance ID, condition ID, build seed))))")
        n = sum(len(v) for v in fx.DEV_IDS.values())
        self.assertEqual(sum(len(s["pcm_sha256"]) for s in lock["dev_sets"].values()), n)
        self.assertIn("delay_samples_8k", lock["codec2"])
        self.assertTrue(lock["codec2"]["store_path"].startswith("/nix/store/"))
        for lib in ("numpy", "scipy", "soundfile", "libsndfile", "torch"):
            self.assertIn(lib, lock["versions"])
        self.assertEqual(lock["pool"]["events"], self.pool["events"])
        self.assertIn("parameters", lock)

    def test_changed_input_refused_on_resume(self):
        out = self.root / "changed"
        shutil.copytree(self.outs["w1"], out)
        (out / "build_manifest.jsonl").unlink()
        cfg = self.config(out, 1, False)
        lock = json.loads((out / "build.lock.json").read_text())
        lock["build_seed"] = 1
        (out / "build.lock.json").write_text(json.dumps(lock))
        with self.assertRaises(noisy_build.LockMismatch):
            noisy_build.build(cfg)

    # 2 ---------------------------------------------------------------------------------------
    def test_added_noise_ratio_measured_from_written_flac(self):
        checked = 0
        for cond in NOISE_CONDITIONS:
            for row in self.rows_of(cond):
                if row["flagged"]:
                    continue
                speech = row["speech_scale"] * self.clean(row)
                noise = self.written(self.outs["w1"], row) - speech
                measured = power_db(speech) - power_db(noise)
                self.assertAlmostEqual(measured, row["ratio_target_db"], delta=0.1, msg=f"{cond} {row['id']}")
                self.assertAlmostEqual(row["ratio_measured_db"], measured, delta=0.01)
                checked += 1
        self.assertGreater(checked, 100)

    def test_event_count_placement_and_levels(self):
        single = 0
        for cond, ratio in (("gun0", 0.0), ("gun10", 10.0)):
            for row in self.rows_of(cond):
                if row["flagged"]:
                    continue
                clean = self.clean(row)
                n = len(clean)
                events = row["events"]
                self.assertEqual(len(events), min(6, math.ceil(n / (3 * SR))), row["id"])
                active = active_frames(clean)
                for e in events:
                    self.assertGreaterEqual(e["start"], 0)
                    self.assertLessEqual(e["start"] + e["length"], n)
                    if not e["shifted"]:
                        self.assertTrue(active[e["start"] // 320], (row["id"], e))
                    self.assertAlmostEqual(e["window_level_db"], 20 * math.log10(np.sqrt(np.mean(clean ** 2))) + ratio,
                                           delta=0.1)
                if len(events) == 1 and not events[0]["truncated"]:
                    e = events[0]
                    speech = row["speech_scale"] * clean
                    noise = self.written(self.outs["w1"], row) - speech
                    span = noise[e["start"]: e["start"] + e["length"]]
                    self.assertAlmostEqual(power_db(span) - power_db(speech), ratio, delta=0.1, msg=row["id"])
                    single += 1
                self.assertIn("event_active_overlap", row)
                self.assertEqual(row["event_to_speech_db"], ratio)
                self.assertNotIn("ratio_target_db", row)  # ratio_measured_db is the resulting utterance-level ratio
        self.assertGreater(single, 3)

    def test_babble_recipe(self):
        clips = {c["key"]: c for c in self.pool["clips"]["babble"]}
        for cond in ("bab10", "bab0"):
            for row in self.rows_of(cond):
                if row["flagged"]:
                    continue
                talkers = row["babble"]
                self.assertEqual(len(talkers), 5)
                self.assertEqual(len({t["key"] for t in talkers}), 5)  # five different files
                for t in talkers:
                    x = fx.read_pcm(clips[t["key"]]["path"])[t["offset"]: t["offset"] + t["length"]]
                    self.assertEqual(len(x), len(self.clean(row)))
                    self.assertGreaterEqual(float(np.mean(active_frames(x))), 0.6)

    def test_field_ratio_against_reverberated_speech(self):
        for row in self.rows_of("field"):
            if row["flagged"]:
                continue
            self.assertEqual(row["ratio_target_db"], 5.0)
            self.assertAlmostEqual(row["ratio_measured_db"], 5.0, delta=0.1)
            self.assertEqual(row["ratio_reference"], "reverberated speech, whole length, before the radio channel")
            self.assertEqual(row["samples"], len(self.clean(row)) + noisy_build.REVERB_TAIL)
            self.assertIn("rir", row)

    # 5 ---------------------------------------------------------------------------------------
    def test_blocklist_covers_every_referenced_source(self):
        block = json.loads((self.outs["w1"] / "eval_pool_blocklist.json").read_text())
        hashes = set(block["pcm_sha256"])
        by_key = {f["key"]: f for f in noisy_sources.pool_files(self.pool)}
        referenced = 0
        for row in self.rows:
            for key in noisy_build.referenced_sources(row):
                self.assertIn(by_key[key]["pcm_sha256"], hashes, key)
                referenced += 1
        self.assertGreater(referenced, 100)

    # 8 ---------------------------------------------------------------------------------------
    def test_each_condition_has_exactly_the_dev_rows(self):
        expected = sorted((name, r["id"], r["text"]) for name, rows in self.dev.items() for r in rows)
        by_condition = defaultdict(list)
        for row in self.rows:
            by_condition[row["condition"]].append((row["set"], row["id"], row["text"]))
        self.assertEqual(tuple(by_condition), BASE_CONDITIONS)
        for cond, got in by_condition.items():
            self.assertEqual(sorted(got), expected, cond)
        for cond in BASE_CONDITIONS:
            for name, rows in self.dev.items():
                manifest = self.outs["w1"] / "manifests" / cond / f"{name}.jsonl"
                lines = [json.loads(line) for line in manifest.read_text().splitlines()]
                self.assertEqual([(r["id"], r["text"]) for r in lines], [(r["id"], r["text"]) for r in rows])
                for r in lines:
                    self.assertTrue(Path(r["audio_filepath"]).is_absolute())
                    self.assertTrue(Path(r["audio_filepath"]).exists())
                    self.assertIn("cluster", r)

    def test_clean_rows_point_at_the_original_audio(self):
        for row in self.rows_of("clean"):
            src = next(r for r in self.dev[row["set"]] if r["id"] == row["id"])
            self.assertEqual(row["audio"], src["audio_filepath"])
            self.assertEqual(row["sha256"], noisy_build.sha256_file(Path(src["audio_filepath"])))
            self.assertIn("active_fraction", row)

    def test_short_and_silent_flagged_and_kept_clean(self):
        flagged = {(r["id"], r["flagged"]) for r in self.rows if r["flagged"]}
        self.assertEqual({f for _, f in flagged}, {"short", "silent"})
        self.assertEqual({i for i, _ in flagged}, {fx.SHORT_ID, fx.SILENT_ID})
        for row in self.rows:
            if row["id"] in (fx.SHORT_ID, fx.SILENT_ID) and row["condition"] != "clean":
                self.assertTrue(row["flagged"])
                pcm = self.written(self.outs["w1"], row)
                np.testing.assert_array_equal(pcm, self.clean(row))
        summary = json.loads((self.outs["w1"] / "build.json").read_text())
        self.assertEqual(summary["statistics"]["flagged"], {"short": 1, "silent": 1})

    def test_clusters(self):
        clusters = {(r["set"], r["id"]): r["cluster"] for r in self.rows_of("clean")}
        self.assertEqual(clusters[("librispeech_dev_clean", "librispeech:1272-128104-0000")], "speaker:1272")
        self.assertEqual(clusters[("ami_dev", "ami:AMI_ES2011a_H00_FEE041_0003427_0003714")], "meeting:ES2011a")
        self.assertEqual(clusters[("yodas_dev", "yodas:en000_00000316_gDJb_EmZIeQ_294_94_2_54")], "video:gDJb_EmZIeQ")
        self.assertEqual(clusters[("voxpopuli_dev", "voxpopuli:20180531-0900-PLENARY-15-en_20180531-15:00:20_3")],
                         "speaker:124989")
        self.assertEqual(clusters[("voxpopuli_dev", "voxpopuli:20160526-0900-PLENARY-3-en_20160526-10:18:50_2")],
                         "session:20160526-0900-PLENARY-3")

    # 9 ---------------------------------------------------------------------------------------
    def test_rotor_conditions_change_nothing_else(self):
        rotor_rows = fx.build_rows(self.outs["rotor"])
        conds = list(dict.fromkeys(r["condition"] for r in rotor_rows))
        self.assertEqual(conds, list(BASE_CONDITIONS[:5]) + ["rotor10", "rotor0"] + list(BASE_CONDITIONS[5:]))
        base = fx.flac_files(self.outs["w1"])
        with_rotor = fx.flac_files(self.outs["rotor"])
        self.assertEqual({k: v for k, v in with_rotor.items() if not k.startswith("audio/rotor")}, base)
        self.assertEqual([r for r in rotor_rows if not r["condition"].startswith("rotor")], self.rows)
        for row in self.rows_of("rotor10", rotor_rows):
            if not row["flagged"]:
                self.assertTrue(all(s["key"].startswith("freesound:") for s in row["noise"]))
        without = json.loads((self.outs["w1"] / "build.json").read_text())
        self.assertEqual(without["rotor"], "unmeasured (no licensed rotor material)")

    # review round 2 --------------------------------------------------------------------------
    def stub_hooks(self):
        import contextlib

        class Model:
            pass

        def decode(model, records, batch_size, device):
            return [{"id": r["id"], "ref": r["text"], "hyp": "", "ref_norm": r["text"].lower(), "hyp_norm": "",
                     "S": 1, "D": 0, "I": 0, "duration": r["duration"], "scored": True} for r in records]

        return noisy_score.Hooks(gpu_lock=lambda label: contextlib.nullcontext(), load_model=lambda spec, d: Model(),
                                 pin_decoder=lambda m, role: "no_while_loops",
                                 require_decoder=lambda m, role: "no_while_loops", decode=decode,
                                 deterministic=lambda: {})

    def test_build_json_separates_condition_ids_from_statistics(self):
        info = json.loads((self.outs["w1"] / "build.json").read_text())
        self.assertEqual(info["conditions"], list(BASE_CONDITIONS))
        self.assertEqual(sorted(info["statistics"]["conditions"]), sorted(BASE_CONDITIONS))

    def test_scorer_accepts_the_real_builder_output(self):
        rows = noisy_score.condition_rows(self.outs["w1"])
        self.assertEqual(list(rows), list(BASE_CONDITIONS))
        for sets in rows.values():
            self.assertEqual(list(sets), list(fx.DEV_IDS))
        spec = noisy_score.ModelSpec(kind="b0", path=None, name="stub")
        summary = noisy_score.score_model(spec, self.outs["w1"], self.root / "scores", hooks=self.stub_hooks(),
                                          device="cpu")
        self.assertEqual(list(summary["conditions"]), list(BASE_CONDITIONS))
        info = json.loads((self.outs["w1"] / "build.json").read_text())
        self.assertEqual(summary["build"]["lock_sha256"], info["lock_sha256"])
        self.assertEqual(summary["build"]["conditions"], list(BASE_CONDITIONS))

    def copy_build(self, name: str) -> Path:
        out = self.root / name
        shutil.rmtree(out, ignore_errors=True)
        shutil.copytree(self.outs["w1"], out)
        # Manifests hold absolute paths: point the copy's at the copy's own audio.
        for path in (out / "manifests").rglob("*.jsonl"):
            path.write_text(path.read_text().replace(str(self.outs["w1"].resolve()), str(out.resolve())))
        return out

    def test_scorer_refuses_substituted_audio_paths(self):
        out = self.copy_build("substituted")
        noisy_score.condition_rows(out)  # the faithful copy is accepted
        clean = {r["id"]: r["audio"] for r in self.rows_of("clean")}
        path = out / "manifests" / "radio" / "ami_dev.jsonl"
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        for line in lines:
            line["audio_filepath"] = clean[line["id"]]  # clean audio under the radio condition
        path.write_text("".join(json.dumps(line) + "\n" for line in lines))
        with self.assertRaisesRegex(ValueError, "audio_filepath"):
            noisy_score.condition_rows(out)

    def test_scorer_refuses_changed_audio_and_lock(self):
        out = self.copy_build("tampered-audio")
        row = self.rows_of("eng0")[3]
        target = out / row["audio"]
        data = bytearray(target.read_bytes())
        data[-10] ^= 0xFF
        target.write_bytes(bytes(data))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            noisy_score.condition_rows(out)
        out = self.copy_build("tampered-lock")
        lock = json.loads((out / "build.lock.json").read_text())
        lock["build_seed"] += 1
        (out / "build.lock.json").write_text(json.dumps(lock))
        with self.assertRaisesRegex(ValueError, "lock"):
            noisy_score.condition_rows(out)

    def test_completed_output_corruption_is_redone(self):
        out = self.copy_build("corrupted")
        row = self.rows_of("radio")[2]
        (out / row["audio"]).write_bytes(b"not a flac")
        cfg = self.config(out, 1, False)
        noisy_build.build(cfg)
        self.assertEqual(fx.flac_files(out), fx.flac_files(self.outs["w1"]))
        self.assertEqual((out / "build_manifest.jsonl").read_bytes(),
                         (self.outs["w1"] / "build_manifest.jsonl").read_bytes())

    def test_outputs_are_24_bit(self):
        import soundfile as sf
        for row in self.rows:
            if row["condition"] != "clean":
                self.assertEqual(sf.info(str(self.outs["w1"] / row["audio"])).subtype, "PCM_24", row["audio"])

    def test_babble_normalization_measured_independently(self):
        clips = {c["key"]: c for c in self.pool["clips"]["babble"]}
        checked = 0
        for cond in ("bab10", "bab0"):
            for row in self.rows_of(cond):
                if row["flagged"]:
                    continue
                clean = self.clean(row)
                total = np.zeros(len(clean))
                for t in row["babble"]:
                    seg = fx.read_pcm(clips[t["key"]]["path"])[t["offset"]: t["offset"] + t["length"]]
                    active = np.repeat(active_frames(seg), 320)[: len(seg)]
                    self.assertGreaterEqual(power_db(seg[active]), -50.0)  # absolute level of the speech
                    total += seg / np.sqrt(np.mean(seg ** 2))             # equal RMS per talker
                expected = row["headroom_scale"] * (clean + row["noise_gain"] * total)
                np.testing.assert_allclose(self.written(self.outs["w1"], row), expected, atol=2.0 ** -22)
                checked += 1
        self.assertGreater(checked, 20)

    def test_rows_carry_noise_source_groups(self):
        groups = {c["key"]: c["group"] for items in self.pool["clips"].values() for c in items}
        for row in self.rows:
            if row["flagged"] or row["condition"] in ("clean", "radio"):
                continue
            self.assertIn("groups", row, row["condition"])
            if row["condition"] in NOISE_CONDITIONS and row["condition"] not in ("bab10", "bab0"):
                self.assertEqual(row["groups"]["noise"], groups[row["noise"][0]["key"]])
            if row["condition"] in ("rev", "field"):
                self.assertEqual(row["groups"]["rir"], f"but-reverbdb:{row['rir']['room']}")
            if row["condition"] == "field":
                self.assertEqual(row["groups"]["noise"], groups[row["noise"][0]["key"]])


class ReconstructionTest(unittest.TestCase):
    """Review round 3, finding 5: independent reconstruction of field and impulsive outputs."""

    @classmethod
    def setUpClass(cls):
        cls.root = fx.fresh_dir("reconstruct")
        cls.out, cls.cfg = fx.tiny_build(cls.root)
        cls.rows = fx.build_rows(cls.out)
        cls.pool = noisy_sources.load_pool(cls.cfg.pool)
        cls.dev = {name: {r["id"]: r for r in map(json.loads, (cls.cfg.manifests / f"{name}.jsonl").read_text().splitlines())}
                   for name in fx.DEV_IDS}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def clean(self, row):
        return fx.read_pcm(self.dev[row["set"]][row["id"]]["audio_filepath"])

    @staticmethod
    def loop(clip, length, offset, crossfade=800):
        """Equal-gain raised-cosine looping (plans/build.md "Continuous noise segments"), re-implemented."""
        fade = 0.5 - 0.5 * np.cos(np.pi * (np.arange(crossfade) + 0.5) / crossfade)
        period = len(clip) - crossfade
        stream = clip.copy()
        while len(stream) < offset + length:
            nxt = clip.copy()
            nxt[:crossfade] *= fade
            stream[-crossfade:] *= 1 - fade
            stream = np.concatenate([stream[:len(stream) - crossfade], stream[len(stream) - crossfade:] + nxt[:crossfade],
                                     nxt[crossfade:]])
        assert period > 0
        return stream[offset: offset + length]

    def test_field_reconstructed_independently(self):
        from scipy import signal as ss
        import soundfile as sf

        import noisy_dsp
        rirs = {r["key"]: r for r in self.pool["rirs"]}
        clips = {c["key"]: c for c in self.pool["clips"]["engine"]}
        delay = json.loads((self.out / "build.json").read_text())["codec_delay_samples_8k"]
        codec = noisy_dsp.Codec2()
        checked = 0
        for row in (r for r in self.rows if r["condition"] == "field" and not r["flagged"]):
            clean = self.clean(row)
            n = len(clean)
            h = sf.read(rirs[row["rir"]["key"]]["path"], always_2d=True)[0][:, row["rir"]["channel"]]
            h = h[int(np.argmax(np.abs(h) > 0.1 * np.abs(h).max())):]
            rev = np.zeros(n + 8000)
            full = ss.fftconvolve(clean, h)[: n + 8000]
            rev[: len(full)] = full
            rev *= np.sqrt(np.mean(clean ** 2) / np.mean(rev[:n] ** 2))
            seg_info = row["noise"][0]
            x = fx.read_pcm(clips[seg_info["key"]]["path"])
            seg = (self.loop(x, n + 8000, seg_info["offset"]) if seg_info["looped"]
                   else x[seg_info["offset"]: seg_info["offset"] + n + 8000])
            gain = np.sqrt(np.mean(rev ** 2) / (np.mean(seg ** 2) * 10 ** 0.5))  # 5 dB against the reverberated speech
            self.assertAlmostEqual(row["noise_gain"] / gain, 1.0, places=9, msg=row["id"])
            y = noisy_dsp.radio(rev + gain * seg, codec, delay)
            y *= min(1.0, 0.99 / np.abs(y).max())
            np.testing.assert_allclose(fx.read_pcm(self.out / row["audio"]), y, atol=2.0 ** -21)
            checked += 1
        self.assertGreater(checked, 10)

    def test_impulsive_events_reconstructed_independently(self):
        events = {e["key"]: e for e in self.pool["events"]}
        clips = {c["key"]: c for c in self.pool["event_clips"]}
        truncated = checked = 0
        for row in (r for r in self.rows if r["condition"] in ("gun0", "gun10") and not r["flagged"]):
            clean = self.clean(row)
            n = len(clean)
            target = 10 * np.log10(np.mean(clean ** 2)) + (0.0 if row["condition"] == "gun0" else 10.0)
            noise = np.zeros(n)
            for e in row["events"]:
                ev = events[e["event"]]
                window = fx.read_pcm(clips[ev["clip"]]["path"])[ev["start"]: ev["start"] + ev["length"]]
                gain = 10 ** (target / 20) / np.sqrt(np.mean(window ** 2))  # window RMS = speech RMS + ratio
                self.assertAlmostEqual(e["gain"] / gain, 1.0, places=9)
                placed = window * gain
                if ev["length"] > n:  # truncated to the utterance with a 10 ms fade
                    placed = placed[:n].copy()
                    placed[-160:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(160) / 159)
                    self.assertEqual((e["start"], e["length"]), (0, n))
                    truncated += 1
                noise[e["start"]: e["start"] + len(placed)] += placed
            expected = row["headroom_scale"] * (clean + noise)
            np.testing.assert_allclose(fx.read_pcm(self.out / row["audio"]), expected, atol=2.0 ** -22)
            checked += 1
        self.assertGreater(checked, 20)
        self.assertGreater(truncated, 0)


class ResumeCertificationTest(unittest.TestCase):
    """Review round 3, finding 3: the completed manifest and every journal row are checked against
    the recorded digest and the current locked inputs, not only against their audio hashes."""

    @classmethod
    def setUpClass(cls):
        cls.root = fx.fresh_dir("certify")
        cls.reference, cls.cfg = fx.tiny_build(cls.root, out="reference")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def config(self, out, stop_after=None):
        return dataclasses.replace(self.cfg, out=self.root / out, stop_after=stop_after)

    def interrupted(self, name):
        with self.assertRaises(noisy_build.BuildInterrupted):
            noisy_build.build(self.config(name, stop_after=6))
        return self.root / name / "build_manifest.partial.jsonl"

    def edit_journal(self, partial, edit):
        lines = partial.read_text().splitlines()
        rows = [json.loads(line) for line in lines[1:]]
        target = next(k for k, r in enumerate(rows) if r["condition"] == "radio" and not r["flagged"])
        edit(rows[target], rows)
        partial.write_text("\n".join([lines[0]] + [json.dumps(r) for r in rows]) + "\n")

    def assert_like_reference(self, out):
        self.assertEqual(fx.flac_files(out), fx.flac_files(self.reference))
        self.assertEqual((out / "build_manifest.jsonl").read_bytes(),
                         (self.reference / "build_manifest.jsonl").read_bytes())

    def test_journal_row_with_edited_reference_is_redone(self):
        partial = self.interrupted("j-text")
        self.edit_journal(partial, lambda row, rows: row.update(text="an unrelated reference"))
        noisy_build.build(self.config("j-text"))
        self.assert_like_reference(self.root / "j-text")

    def test_journal_row_with_edited_cluster_is_redone(self):
        partial = self.interrupted("j-cluster")
        self.edit_journal(partial, lambda row, rows: row.update(cluster="speaker:9999"))
        noisy_build.build(self.config("j-cluster"))
        self.assert_like_reference(self.root / "j-cluster")

    def test_journal_row_pointing_at_another_valid_file_is_redone(self):
        partial = self.interrupted("j-path")

        def swap(row, rows):
            other = next(r for r in rows if r["condition"] == "radio" and r["id"] != row["id"])
            row.update(audio=other["audio"], sha256=other["sha256"])  # a real file with a matching hash

        self.edit_journal(partial, swap)
        noisy_build.build(self.config("j-path"))
        self.assert_like_reference(self.root / "j-path")

    def test_completed_manifest_with_edited_rows_refused(self):
        out = self.root / "m-text"
        shutil.copytree(self.reference, out)
        lines = (out / "build_manifest.jsonl").read_text().splitlines()
        row = json.loads(lines[40])
        row["text"] = "an unrelated reference"
        lines[40] = noisy_paths.canonical_json(row)
        (out / "build_manifest.jsonl").write_text("\n".join(lines) + "\n")
        with self.assertRaises(noisy_build.ManifestTampered):
            noisy_build.build(self.config("m-text"))
        self.assertTrue((out / "build_manifest.jsonl").exists())  # left for inspection, not converted


class InputIdentityTest(unittest.TestCase):
    """Review round 2, finding 2: pool contents and verification records are checked on every
    build and resume, so a source file changed after an interruption is refused."""

    def setUp(self):
        self.root = fx.fresh_dir("identity")
        self.manifests = fx.make_dev_sets(self.root / "dev")
        self.pool_path = fx.make_pool(self.root / "pool")
        self.raw, self.verify = fx.source_dirs(self.pool_path)
        self.pool = noisy_sources.load_pool(self.pool_path)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def config(self, stop_after=None):
        return noisy_build.BuildConfig(out=self.root / "out", manifests=self.manifests, pool=self.pool_path,
                                       workers=1, voxpopuli_speakers=fx.VOX_SPEAKERS, stop_after=stop_after,
                                       raw=self.raw, verify_dir=self.verify, registry=fx.registry(self.pool_path))

    def interrupt(self):
        with self.assertRaises(noisy_build.BuildInterrupted):
            noisy_build.build(self.config(stop_after=4))

    def test_noise_file_changed_after_interruption_refused(self):
        self.interrupt()
        clip = self.pool["clips"]["engine"][0]
        fx.write_audio(Path(clip["path"]), fx.noise_clip(9.0, 999, "brown"))  # same length, other audio
        with self.assertRaises(noisy_build.InputChanged):
            noisy_build.build(self.config())
        self.assertFalse((self.root / "out" / "build_manifest.jsonl").exists())

    def test_rir_file_changed_after_interruption_refused(self):
        self.interrupt()
        rir = self.pool["rirs"][1]
        x, _ = __import__("soundfile").read(rir["path"], always_2d=True)
        fx.write_audio(Path(rir["path"]), x * 0.5, subtype="FLOAT")
        with self.assertRaises(noisy_build.InputChanged):
            noisy_build.build(self.config())

    def test_verified_download_changed_after_interruption_refused(self):
        self.interrupt()
        (self.raw / "demand" / "demand.bin").write_bytes(b"something else entirely")
        with self.assertRaises(noisy_sources.VerificationError):
            noisy_build.build(self.config())

    def test_reverification_with_other_contents_refused(self):
        self.interrupt()
        import hashlib
        payload = b"a different archive"
        (self.raw / "musan" / "musan.bin").write_bytes(payload)
        registry = {"musan": {"dataset": "musan", "licence": "test", "checksum": "md5",
                              "files": {"musan.bin": hashlib.md5(payload).hexdigest()}}}
        noisy_sources.verify_source("musan", raw=self.raw, verify_dir=self.verify, registry=registry)
        # Round 5: the record now fails validation against the build's registry (its checksum is
        # not the registered one) before it could be compared with the pool's.
        with self.assertRaisesRegex(noisy_sources.VerificationError, "registry"):
            noisy_build.build(self.config())

    def test_pool_built_from_other_downloads_refused(self):
        self.interrupt()
        pool = json.loads(self.pool_path.read_text())
        pool["sources"]["demand"]["verification"]["files"][0]["sha256"] = "0" * 64
        self.pool_path.write_text(json.dumps(pool))
        with self.assertRaises(noisy_build.InputChanged):  # the record is valid, the pool names other files
            noisy_build.build(self.config())

    def test_unchanged_inputs_resume(self):
        self.interrupt()
        noisy_build.build(self.config())
        self.assertTrue((self.root / "out" / "build_manifest.jsonl").exists())


if __name__ == "__main__":
    unittest.main()


class BabbleTalkerTest(unittest.TestCase):
    """Review round 2, finding 7: silence, a stationary noise floor or speech below -50 dBFS
    active RMS never becomes a normalized babble talker."""

    @classmethod
    def setUpClass(cls):
        cls.root = fx.fresh_dir("babble-talkers")

    def view(self, good: int) -> noisy_build.PoolView:
        rng = np.random.default_rng(7)
        bad = {"silence": 1e-6 * rng.standard_normal(20 * SR),                     # -120 dBFS
               "floor": 0.01 * rng.standard_normal(20 * SR),                       # -40 dBFS, stationary: 100 % "active"
               "quiet": fx.speech_like(20.0, 77, -62.0, pauses=(0.02, 0.08))}      # speech, but too quiet
        clips = []
        for name, x in [*((f"good{k}", fx.speech_like(20.0, 60 + k, -24, pauses=(0.02, 0.08))) for k in range(good)),
                        *bad.items()]:
            path = fx.write_audio(self.root / f"{name}.wav", x)
            clips.append(noisy_sources.clip_record(path, key=f"musan:{name}", source="musan", category="babble",
                                                   original=path.name, licence="public domain"))
        pool = noisy_sources.make_pool(clips={"babble": clips}, event_clips=[], events=[], rirs=[],
                                       sources={"musan": {"dataset": "MUSAN", "verification": {"ok": True}}},
                                       reserved={})
        return noisy_build.PoolView(pool)

    def test_bad_segments_never_chosen(self):
        view = self.view(good=6)
        clean = fx.speech_like(2.5, 5)
        cond = noisy_build.Condition("bab0", "babble", "babble", 0.0)
        for seed in range(25):
            _, info = noisy_build.render_babble(view, clean, cond, np.random.Generator(np.random.PCG64(seed)))
            keys = {t["key"] for t in info["babble"]}
            self.assertEqual(len(keys), 5)
            self.assertTrue(all(k.startswith("musan:good") for k in keys), keys)
            for t in info["babble"]:
                self.assertGreaterEqual(t["active_level_db"], -50.0)

    def test_too_few_speech_files_refused(self):
        view = self.view(good=4)
        clean = fx.speech_like(2.5, 5)
        cond = noisy_build.Condition("bab0", "babble", "babble", 0.0)
        with self.assertRaises(noisy_build.BuildError):
            noisy_build.render_babble(view, clean, cond, np.random.Generator(np.random.PCG64(1)))
