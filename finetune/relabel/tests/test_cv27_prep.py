"""Contract tests: preparation of the read-speech extra source (b0prime plan "Extra sources", v4 contract tests).

- no selected clip whose speaker, clip or normalized sentence is held out, including a held-out clip whose speaker id
  changed between releases (the path join over all six tsvs);
- the global denylist's containment rule (boilerplate-aware) and identity rule, and the talk-text 8-gram rule;
- the caps (speaker and sentence, seeded order), the 1-30 s duration rule;
- labels through filters version 3 only when stored, and cleaning layer 1 alone when read;
- the B0 confirmation threshold and the per-accent report;
- the dev sample's exclusions and per-speaker limit;
- the store: 16 kHz mono PCM_16 FLAC with sample counts and PCM SHA-256, resumable chunks, the clip source in the
  synthetic store's format, loadable as a synthetic.mixing.ClipPool.

Run from finetune/relabel:  ./python -m unittest discover -s tests -p 'test_cv27_*' -v
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from cv27 import confirm, dev, heldout, layout, select, store
from labeling import filters

HEADER = ["client_id", "path", "sentence_id", "sentence", "sentence_domain", "up_votes", "down_votes", "age",
          "gender", "accents", "variant", "locale", "segment"]
_TOKENIZER: dict = {}


def coverage():
    if "cov" not in _TOKENIZER:
        _TOKENIZER["dir"] = tempfile.TemporaryDirectory()
        _TOKENIZER["cov"] = filters.Coverage(filters.b0_tokenizer_file(Path(_TOKENIZER["dir"].name)))
    return _TOKENIZER["cov"]


def write_tsv(path: Path, rows: list[dict]) -> None:
    lines = ["\t".join(HEADER)] + ["\t".join(str(r.get(h, "")) for h in HEADER) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def row(spk: str, path: str, sentence: str, accents: str = "") -> dict:
    return {"client_id": spk, "path": path, "sentence": sentence, "accents": accents, "locale": "en"}


class Release:
    """Six split tsvs plus clip_durations.tsv in a temporary directory, and the ESB rows."""

    def __init__(self, root: Path, tsvs: dict[str, list[dict]], esb: list[tuple[str, str, str]], ms: dict[str, int]):
        self.root, self.esb = root, esb
        for name in layout.TSVS:
            write_tsv(root / name, tsvs.get(name, []))
        (root / layout.DURATIONS_TSV).write_text("clip\tduration[ms]\n" + "".join(f"{k}\t{v}\n" for k, v in ms.items()))

    def tsv_rows(self, name):
        return heldout.read_tsv(self.root / name, ("client_id", "path", "sentence"))

    def validated(self):
        return heldout.read_tsv(self.root / layout.TRAIN_TSV, ("client_id", "path", "sentence", "accents"))


def build_release(tmp: Path) -> Release:
    """The held-out cases:
    - ESB clip c_e1.mp3 (ESB id "esbA") is in this release's other.tsv under client_id "spkX" (changed id);
      spkX's validated clip c_x2.mp3 must go by speaker;
    - ESB clip c_e2.mp3 (ESB id "esbB") is in invalidated.tsv under "spkY"; spkY's validated c_y1.mp3 goes;
    - ESB id "esbC" is also a client_id in validated.tsv (the plain id form): c_c1.mp3 goes;
    - test.tsv: spkT reads "Mr. Smith has two dogs." (c_t1.mp3); spkT's validated c_t2.mp3 goes by speaker,
      spkF's "mister smith has 2 dogs" (c_f1.mp3) goes by normalized sentence;
    - dev.tsv: spkD (c_d1.mp3, reserved) - spkD's validated c_d2.mp3 goes;
    - ESB text "An ESB sentence about rivers." - spkG's validated "an esb sentence about rivers" goes (sentence)."""
    tsvs = {
        "validated.tsv": [row("spkX", "c_x2.mp3", "A clip by a changed speaker."),
                          row("spkY", "c_y1.mp3", "Another changed speaker clip."),
                          row("esbC", "c_c1.mp3", "Plain id form clip."),
                          row("spkT", "c_t2.mp3", "The test speaker reads again."),
                          row("spkF", "c_f1.mp3", "mister smith has 2 dogs"),
                          row("spkD", "c_d2.mp3", "The dev speaker reads again."),
                          row("spkG", "c_g1.mp3", "an esb sentence about rivers"),
                          row("spkK", "c_k1.mp3", "A perfectly ordinary kept sentence.", "England English"),
                          row("spkK", "c_k2.mp3", "Yet another ordinary kept sentence.", "")],
        "other.tsv": [row("spkX", "c_e1.mp3", "An older recording.")],
        "invalidated.tsv": [row("spkY", "c_e2.mp3", "undefined")],
        "train.tsv": [row("spkK", "c_k1.mp3", "A perfectly ordinary kept sentence.")],
        "test.tsv": [row("spkT", "c_t1.mp3", "Mr. Smith has two dogs.")],
        "dev.tsv": [row("spkD", "c_d1.mp3", "A sentence reserved for dev.")],
    }
    esb = [("esbA", "An older recording.", "c_e1.mp3"), ("esbB", "Something else entirely.", "c_e2.mp3"),
           ("esbC", "An ESB sentence about rivers.", "c_c9.mp3")]
    ms = {r["path"]: 3000 for rows in tsvs.values() for r in rows}
    return Release(tmp, tsvs, esb, ms)


def normalizer():
    return heldout.Normalizer()


class HeldOutTest(unittest.TestCase):
    def test_no_selected_clip_is_held_out_and_changed_ids_are_joined(self):
        with tempfile.TemporaryDirectory() as t:
            rel = build_release(Path(t))
            norm = normalizer()
            held = heldout.build_heldout(rel.esb, rel.tsv_rows, norm)
            self.assertIn("spkX", held.speakers)        # joined from other.tsv by the ESB path
            self.assertIn("spkY", held.speakers)        # joined from invalidated.tsv
            self.assertEqual(held.counts["esb_paths_with_changed_id"], 2)
            durations = heldout.read_durations(rel.root / layout.DURATIONS_TSV)
            clips, counts = select.select(rel.validated(), held, durations, frozenset(), None, coverage(), norm)
            reason = {c.path: c.reason for c in clips}
            self.assertEqual(reason["c_x2.mp3"], "speaker")
            self.assertEqual(reason["c_y1.mp3"], "speaker")
            self.assertEqual(reason["c_c1.mp3"], "speaker")
            self.assertEqual(reason["c_t2.mp3"], "speaker")
            self.assertEqual(reason["c_d2.mp3"], "speaker")
            self.assertEqual(reason["c_f1.mp3"], "sentence")      # the evaluation normalizer: mister / 2
            self.assertEqual(reason["c_g1.mp3"], "sentence")
            self.assertIsNone(reason["c_k1.mp3"])
            self.assertIsNone(reason["c_k2.mp3"])
            for c in clips:
                if c.reason is None:
                    self.assertNotIn(c.client_id, held.speakers)
                    self.assertNotIn(c.path, held.paths)
                    self.assertNotIn(norm(c.sentence), held.sentences)
            self.assertEqual(counts["stages"]["selected"]["clips"], 2)

    def test_without_the_path_join_the_changed_speaker_would_leak(self):
        """The changed-id case is real: ESB's id alone does not name spkX."""
        with tempfile.TemporaryDirectory() as t:
            rel = build_release(Path(t))
            self.assertNotIn("spkX", {uid for uid, _, _ in rel.esb})

    def test_unsafe_clip_paths_are_refused(self):
        for bad in ("../x.mp3", "a/b.mp3", "x.wav", "", ".mp3"):
            with self.assertRaises(ValueError):
                heldout.clip_stem(bad)
        self.assertEqual(heldout.clip_id("abc_1.mp3"), f"{layout.ALIAS}:abc_1")


def heldout_manifest(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


class DenylistTest(unittest.TestCase):
    WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo".split()   # 11 words: 4 8-grams

    def _containment(self, tmp: Path, boiler_grams: list[list[str]], identity=None):
        from pool import common
        from pool import heldout as ho
        rows = [{"id": "u1", "text": " ".join(self.WORDS)}]
        if identity:
            rows.append({"id": "u2", "text": "short", "identity": identity})
        idx = ho.Index(ho.load_sets([("dev_x", heldout_manifest(tmp / "dev_x.jsonl", rows))]))
        h = common.WordHasher()
        boiler = np.array([common.ngram_hashes(h(g))[0] for g in boiler_grams], dtype=np.uint64)
        return heldout.Containment(idx, boiler)

    def test_containment_is_half_of_the_non_boilerplate_8grams(self):
        w = self.WORDS
        with tempfile.TemporaryDirectory() as t:
            plain = self._containment(Path(t), [])
            # 2 of 4 8-grams (words 0..8, 2..10) -> contained; 1 of 4 (words 0..7, 3..10) -> not
            self.assertEqual(set(plain.contained([w[0:9], w[0:8], w[2:11], w[3:11], ["x"] * 9])), {0, 2})
            # with the first two 8-grams as boilerplate the denominator is 2: one non-boilerplate 8-gram suffices
            boiled = self._containment(Path(t), [w[0:8], w[1:9]])
            self.assertEqual(set(boiled.contained([w[2:10], w[0:9]])), {0})
            hit = boiled.contained([w[2:10]])[0][0]
            self.assertEqual((hit["set"], hit["id"], hit["matched"], hit["of"]), ("dev_x", "u1", 1, 2))

    def test_identity_rule(self):
        with tempfile.TemporaryDirectory() as t:
            c = self._containment(Path(t), [], identity={layout.ALIAS: f"{layout.ALIAS}:c_k1"})
            self.assertTrue(c.identity_hit(f"{layout.ALIAS}:c_k1"))
            self.assertFalse(c.identity_hit(f"{layout.ALIAS}:c_k2"))

    def test_select_applies_the_denylist_and_talk_rules_after_the_held_out_rules(self):
        w = self.WORDS
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            rel_dir = tmp / "rel"
            rel_dir.mkdir()
            talk = "the quick brown fox jumps over the lazy dog again and again"
            tsvs = {"validated.tsv": [row("s1", "a1.mp3", " ".join(w[0:9])),             # contained
                                      row("s2", "a2.mp3", "And the quick brown fox jumps over the lazy dog."),  # talk
                                      row("s3", "a3.mp3", "Quick brown fox jumps over the lazy."),  # 7 words of it
                                      row("s4", "a4.mp3", "Kept by every rule here.")]}
            rel = Release(rel_dir, tsvs, [], {f"a{i}.mp3": 2000 for i in range(1, 5)})
            norm = normalizer()
            held = heldout.build_heldout([], rel.tsv_rows, norm)
            grams = heldout.talk_grams([talk], norm)
            clips, counts = select.select(rel.validated(), held, heldout.read_durations(rel_dir / layout.DURATIONS_TSV),
                                          grams, self._containment(tmp, []), coverage(), norm)
            reason = {c.path: c.reason for c in clips}
            self.assertEqual(reason, {"a1.mp3": "denylist_containment", "a2.mp3": "talk_text_8gram", "a3.mp3": None,
                                      "a4.mp3": None})
            self.assertEqual(counts["dropped"]["talk_text_8gram"]["clips"], 1)

    def test_reproduced_boilerplate_must_match_the_denylist_counts(self):
        """reproduce_boilerplate takes the 8-grams in more than 100 recordings, and refuses a scan that does not
        reproduce the recorded counts."""
        from types import SimpleNamespace
        from unittest import mock

        from pool import common
        from pool import heldout as ho
        w = self.WORDS
        with tempfile.TemporaryDirectory() as t:
            idx = ho.Index(ho.load_sets([("dev_x", heldout_manifest(Path(t) / "dev_x.jsonl",
                                                                    [{"id": "u1", "text": " ".join(w)}]))]))
            # hash index 0 in 101 recordings (boilerplate), index 1 in 100 (not)
            pairs = [(g << 32) | 0 for g in range(101)] + [(g << 32) | 1 for g in range(100)]
            res = SimpleNamespace(pairs=np.array(sorted(pairs), dtype=np.uint64), keys=[("yodas", str(i)) for i in range(101)],
                                  sources={"yodas": {"files_sha256": "f", "listing_sha256": "l"}})
            deny = {"counts": {"boilerplate_8grams": 1, "heldout_8grams": 4, "training_recordings_scanned": 101,
                               "contained_pairs": 0},
                    "training": {"yodas": {"files_sha256": "f", "listing_sha256": "l"}}}
            # u1 has 4 8-grams, 3 non-boilerplate; 100 recordings hold one each: 1 < ceil(1.5) -> no containment
            with mock.patch("pool.scan.scan", return_value=res):
                hashes, info = heldout.reproduce_boilerplate(deny, idx, {"yodas": Path(t)}, {}, 0)
                self.assertEqual(hashes.tolist(), [int(idx.hashes[0])])
                deny["counts"]["boilerplate_8grams"] = 2
                with self.assertRaises(RuntimeError):
                    heldout.reproduce_boilerplate(deny, idx, {"yodas": Path(t)}, {}, 0)
            path = Path(t) / "b" / "boilerplate.json"
            heldout.save_boilerplate(path, hashes, info, "deny-sha")
            self.assertEqual(heldout.load_boilerplate(path, "deny-sha").tolist(), hashes.tolist())
            with self.assertRaises(RuntimeError):
                heldout.load_boilerplate(path, "other-denylist")
            self.assertTrue(common.sha256_file(path.with_suffix(".u64")))


def clip(i: int, spk: str, norm: str, ms: int = 2000) -> select.Clip:
    return select.Clip(f"{layout.ALIAS}:c{i:04d}", f"c{i:04d}.mp3", spk, norm, "", ms, norm)


class CapsTest(unittest.TestCase):
    def test_plan_constants(self):
        self.assertEqual((layout.SPEAKER_CAP, layout.SENTENCE_CAP), (1000, 3))
        self.assertEqual((layout.MIN_MS, layout.MAX_MS), (1000, 30000))
        self.assertEqual(layout.B0_MAX_WER, 0.5)
        self.assertEqual((layout.DEV_SIZE, layout.DEV_PER_SPEAKER, layout.DEV_SEED), (2000, 2, 27))

    def _world(self):
        clips = []
        for i in range(40):                       # speaker "big": 40 clips, distinct sentences
            clips.append(clip(i, "big", f"sentence {i}"))
        for i in range(40, 52):                   # 12 speakers read the same sentence
            clips.append(clip(i, f"s{i}", "popular sentence"))
        for i in range(52, 60):
            clips.append(clip(i, f"t{i}", f"own {i}"))
        clips[59].reason = "speaker"              # already dropped clips are not touched
        return clips

    def test_caps_hold_and_are_maximal_and_seeded(self):
        clips = self._world()
        select.apply_caps(clips, seed=5, speaker_cap=10, sentence_cap=3)
        kept = [c for c in clips if c.reason is None]
        self.assertLessEqual(max(Counter(c.client_id for c in kept).values()), 10)
        self.assertLessEqual(max(Counter(c.norm for c in kept).values()), 3)
        self.assertEqual(sum(c.client_id == "big" for c in kept), 10)
        self.assertEqual(sum(c.norm == "popular sentence" for c in kept), 3)
        self.assertEqual(Counter(c.reason for c in clips), Counter({None: 10 + 3 + 7, "speaker_cap": 30,
                                                                     "sentence_cap": 9, "speaker": 1}))
        again = self._world()[::-1]               # input order does not matter
        select.apply_caps(again, seed=5, speaker_cap=10, sentence_cap=3)
        self.assertEqual({c.id: c.reason for c in clips}, {c.id: c.reason for c in again})
        other = self._world()
        select.apply_caps(other, seed=6, speaker_cap=10, sentence_cap=3)
        self.assertNotEqual({c.id for c in clips if c.reason is None}, {c.id for c in other if c.reason is None})

    def test_duration_bounds_are_inclusive(self):
        self.assertEqual([select.duration_ok(m) for m in (999, 1000, 30000, 30001)], [False, True, True, False])


class LabelTest(unittest.TestCase):
    def test_labels_are_filters_version_3_text(self):
        cov = coverage()
        text, reason = select.label_of('He said "yes"; then Q&A (later).', cov)
        self.assertEqual(text, filters.map_text('He said "yes"; then Q&A (later).', version=3))
        self.assertEqual((text, reason), ("He said yes, then Q and A later.", None))
        self.assertEqual(select.label_of("Tokyo 東京 trip.", cov)[1], "tokenizer_coverage")
        self.assertEqual(select.label_of('""', cov)[1], "empty_label")

    def test_stored_label_is_uncleaned_and_read_through_layer_1_alone(self):
        from training import cleaning
        label, reason = select.label_of("We gonna setup the acid base test.", coverage())
        self.assertIsNone(reason)
        self.assertEqual(label, "We gonna setup the acid base test.")     # no cleaning when stored
        clean, identity = store.label_layer1(cleaning.VERSION)
        out = clean(label, f"{layout.ALIAS}:x")
        self.assertEqual(out, cleaning.Cleaning.load(choices_dir=None).clean(label, f"{layout.ALIAS}:x", layout.ALIAS))
        self.assertIn("going to", out)              # layer 1 (dialect)
        self.assertIn("acid-base", out)             # layer 1 (forms)
        self.assertIn("setup", out)                 # a layer 2 member is left as written
        self.assertEqual(identity["layers"], [1])
        self.assertIsNone(identity["choices_sha256"])
        with self.assertRaises(ValueError):
            store.label_layer1(99)


class ConfirmTest(unittest.TestCase):
    def test_threshold(self):
        self.assertTrue(confirm.keep(0.5))
        self.assertFalse(confirm.keep(0.5000001))
        self.assertFalse(confirm.keep(math.inf))
        self.assertTrue(confirm.judged("alpha bravo charlie delta", "alpha bravo echo golf")["kept"])     # 0.5
        self.assertFalse(confirm.judged("alpha bravo charlie delta", "alpha kilo echo golf")["kept"])     # 0.75
        self.assertTrue(confirm.judged("Mr. Smith has two dogs.", "mister smith has 2 dogs")["kept"])
        j = confirm.judged("[noise]", "words")             # the reference normalizes to nothing
        self.assertEqual((j["wer"], j["wer_infinite"], j["kept"]), (None, True, False))

    def test_accent_report(self):
        rep = confirm.accent_report([("England English", False), ("England English", True), ("", True),
                                     ("Scottish English|England English", False)])
        self.assertEqual(rep["by_field"][""], {"checked": 1, "dropped": 1, "rate": 1.0})
        self.assertEqual(rep["by_field"]["England English"]["rate"], 0.5)
        self.assertEqual(rep["by_tag"]["England English"], {"checked": 3, "dropped": 1, "rate": 1 / 3})
        self.assertEqual(rep["by_tag"][""]["checked"], 1)

    def test_run_chunk_uses_the_cut_and_judges_every_clip(self):
        class Fake:
            def plan(self, durations):
                import train
                return train.cut_pool(durations, 3.0)

            def run(self, clips):
                return ["hello world" if len(c) > 16000 else "zzz qqq" for c in clips]

        recs = [{"id": "a", "status": "ok", "samples": 24000}, {"id": "b", "status": "undecodable"},
                {"id": "c", "status": "ok", "samples": 16000}]
        audio = {"a": np.zeros(24000, np.int16), "c": np.zeros(16000, np.int16)}
        out = confirm.run_chunk(Fake(), recs, {"a": "Hello, world!", "c": "Hello world."}, audio)
        self.assertEqual([o["id"] for o in out], ["a", "c"])
        self.assertEqual([o["kept"] for o in out], [True, False])
        self.assertEqual(out[1]["b0"], "zzz qqq")


class DevTest(unittest.TestCase):
    def test_exclusions_and_limits(self):
        with tempfile.TemporaryDirectory() as t:
            rel = build_release(Path(t))
            norm = normalizer()
            held = heldout.build_heldout(rel.esb, rel.tsv_rows, norm)
            rows = [row("esbA", "d1.mp3", "fresh one"),               # ESB id form
                    row("spkX", "d2.mp3", "fresh two"),               # this release's id of an ESB path
                    row("ok1", "c_e1.mp3", "fresh three"),            # an ESB path
                    row("ok2", "d4.mp3", "an esb sentence about rivers!")]   # an ESB sentence
            rows += [row(f"sp{i % 5}", f"e{i:03d}.mp3", f"sentence {i}") for i in range(40)]
            el = dev.eligible(rows, held, norm)
            self.assertEqual({r["path"] for r in el}, {f"e{i:03d}.mp3" for i in range(40)})
            s = dev.sample(el, n=8, per_speaker=2, seed=27)
            self.assertEqual(len(s), 8)
            self.assertLessEqual(max(Counter(r["client_id"] for r in s).values()), 2)
            self.assertEqual(s, dev.sample(el[::-1], n=8, per_speaker=2, seed=27))
            self.assertEqual(len(dev.sample(el, n=100, per_speaker=2)), 10)     # 5 speakers x 2


def mp3_bytes(seconds: float, rate: int = 48000, freq: float = 440.0) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    buf = io.BytesIO()
    sf.write(buf, (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), rate, format="MP3")
    return buf.getvalue()


class StoreTest(unittest.TestCase):
    def test_decode_is_16k_mono_pcm16_with_hashes(self):
        with tempfile.TemporaryDirectory() as t:
            src = Path(t) / "a.mp3"
            src.write_bytes(mp3_bytes(2.0))
            rec = store.decode_clip(("cv:a", str(src), str(Path(t) / "out" / "a.flac")))
            self.assertEqual(rec["status"], "ok")
            data = Path(rec["audio_filepath"]).read_bytes()
            info = sf.info(io.BytesIO(data))
            self.assertEqual((info.samplerate, info.channels, info.subtype, info.format), (16000, 1, "PCM_16", "FLAC"))
            pcm, _ = sf.read(io.BytesIO(data), dtype="int16")
            self.assertEqual(rec["samples"], len(pcm))
            self.assertAlmostEqual(rec["samples"] / 16000, 2.0, delta=0.1)
            self.assertEqual(rec["duration"], rec["samples"] / 16000)
            self.assertEqual(rec["sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(rec["pcm_sha256"], hashlib.sha256(pcm.astype("<i2").tobytes()).hexdigest())
            self.assertEqual(rec["source_rate"], 48000)

    def test_short_and_damaged_clips_get_no_file(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / "s.mp3").write_bytes(mp3_bytes(0.5))
            (Path(t) / "d.mp3").write_bytes(b"not audio at all" * 10)
            s = store.decode_clip(("cv:s", str(Path(t) / "s.mp3"), str(Path(t) / "o" / "s.flac")))
            d = store.decode_clip(("cv:d", str(Path(t) / "d.mp3"), str(Path(t) / "o" / "d.flac")))
            self.assertEqual((s["status"], d["status"]), ("decoded_duration", "undecodable"))
            self.assertFalse((Path(t) / "o").exists())

    def test_quantize_clips_and_counts(self):
        pcm, clipped = store.quantize(np.array([0.0, 0.5, -1.0, 1.0, 1.2], np.float32))
        self.assertEqual(pcm.tolist(), [0, 16384, -32768, 32767, 32767])
        self.assertEqual(clipped, 2)


def store_world(tmp: Path, n: int = 5, chunk: int = 2):
    raw = tmp / "raw" / "clips"
    raw.mkdir(parents=True)
    rows = []
    for i in range(n):
        name = f"x_{i}.mp3"
        (raw / name).write_bytes(mp3_bytes(1.2 + 0.1 * i, freq=300 + 50 * i))
        rows.append({"id": heldout.clip_id(name), "path": name, "client_id": f"s{i % 2}", "sentence": f"Sentence {i}.",
                     "accents": "" if i % 2 else "England English", "ms": 1200 + 100 * i,
                     "label": f"Sentence {i}.", "reason": None})
    rows.sort(key=lambda r: r["id"])
    paths = layout.Paths(tmp / "out")
    store.plan(paths, "sel-sha", [r["id"] for r in rows], chunk_clips=chunk)
    return paths, rows, raw


class StoreChunksTest(unittest.TestCase):
    def test_chunks_resume_and_refuse_another_selection(self):
        with tempfile.TemporaryDirectory() as t:
            paths, rows, raw = store_world(Path(t))
            first = store.decode_chunks(paths, rows, raw, [0, 1], workers=0, log=lambda m: None)
            self.assertEqual(first, {"decoded_chunks": 2, "already_done": 0})
            again = store.decode_chunks(paths, rows, raw, [0, 1, 2], workers=0, log=lambda m: None)
            self.assertEqual(again, {"decoded_chunks": 1, "already_done": 2})
            per, _ = store.chunk_records(paths, rows)
            self.assertEqual([r["id"] for p in per for r in p], [r["id"] for r in rows])
            with self.assertRaises(RuntimeError):
                store.plan(paths, "another-selection", [r["id"] for r in rows], chunk_clips=2)
            with self.assertRaises(RuntimeError):
                store.chunk_records(paths, rows[:-1])

    def test_finalize_writes_the_clip_source(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as t:
            paths, rows, raw = store_world(Path(t))
            store.decode_chunks(paths, rows, raw, [0, 1, 2], workers=0, log=lambda m: None)
            per, _ = store.chunk_records(paths, rows)
            # B0 reads the even clips correctly and the odd ones as something else entirely
            said = {r["samples"]: (f"sentence {i}" if i % 2 == 0 else "completely different words here")
                    for i, r in enumerate(x for p in per for x in p)}
            self.assertEqual(len(said), 5)

            class Fake:
                def plan(self, durations):
                    return [list(range(len(durations)))]

                def run(self, clips):
                    return [said[len(c)] for c in clips]

            fake_settings = {"batch_seconds": 10.0, "engine": {}, "model_sha256": "m", "rule": confirm.RULE,
                             "max_wer": 0.5, "normalizer": "n"}
            with mock.patch.object(confirm, "settings", return_value=fake_settings):
                out = confirm.confirm_chunks(paths, rows, Fake(), 10.0, [0, 1, 2], log=lambda m: None)
                self.assertEqual(out["confirmed_chunks"], 3)
                with mock.patch.object(confirm, "settings", return_value=fake_settings | {"batch_seconds": 20.0}):
                    with self.assertRaises(RuntimeError):
                        confirm.confirm_chunks(paths, rows, Fake(), 20.0, [0], log=lambda m: None)
            report = confirm.finalize(paths, rows, {"counts": {"stages": {}}})
            kept_ids = [json.loads(line)["id"] for line in paths.clips.read_text().splitlines()]
            self.assertEqual(kept_ids, [rows[i]["id"] for i in (0, 2, 4)])
            self.assertEqual(report["store"]["checked_by_b0"], 5)
            self.assertEqual(report["store"]["b0_wer_gt_0.5"], 2)
            self.assertEqual(report["accents"]["by_field"]["England English"]["dropped"], 0)
            self.assertEqual(report["accents"]["by_field"][""]["dropped"], 2)
            self.assertEqual(report["final"]["clips"], 3)
            import pyarrow.parquet as pq
            self.assertEqual(pq.read_table(paths.b0_transcripts).num_rows, 5)
            pool = store.ClipPool.load(paths.base)
            self.assertEqual([c["id"] for c in pool.clips], kept_ids)
            c = pool.clips[0]
            for key in ("id", "audio_filepath", "duration", "sha256", "label", "samples", "pcm_sha256"):
                self.assertIn(key, c)
            audio = pool.read(c["id"])
            self.assertEqual((audio.dtype, len(audio)), (np.float32, c["samples"]))
            from training import batchplan
            self.assertEqual(batchplan.clip_samples(pool), [c["samples"] for c in sorted(pool.clips, key=lambda c: c["id"])])
            Path(c["audio_filepath"]).write_bytes(Path(c["audio_filepath"]).read_bytes()[:-10] + b"0" * 10)
            with self.assertRaises(RuntimeError):
                pool.read(c["id"])
            meta = json.loads(paths.clips_meta.read_text())
            self.assertEqual(meta["label"]["filters_version"], 3)
            meta["clips_sha256"] = "0" * 64
            paths.clips_meta.write_text(json.dumps(meta))
            with self.assertRaises(RuntimeError):
                store.ClipPool.load(paths.base)


if __name__ == "__main__":
    unittest.main()
