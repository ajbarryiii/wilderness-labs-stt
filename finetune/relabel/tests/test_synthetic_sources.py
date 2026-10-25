"""Contract tests: TRAIN noise/RIR pool disjointness, the voice-provenance check, voice assignment, the round-trip
decision and the run report (synthetic plan "Augmentation", "Speech").

Run from finetune/relabel:  ./python -m unittest discover -s tests -p 'test_synthetic_*' -v
"""
from __future__ import annotations

import collections
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from synthetic import augment, layout, noise, roundtrip, tts, voicecheck
from synthetic_fixtures import make_split, roots, tone, write_clip
from synthetic.textgen import read_jsonl, write_jsonl

BLOCKLIST = {"units": [{"dataset": "rooms-db", "room": "R1"}, {"dataset": "ambient-db"},
                       {"dataset": "mix-db", "subset": "speech/gov"}],
             "excluded_families": {"nc-db": "CC BY-NC"},
             "freesound_ids": [11, 12], "freesound_uploaders": ["alice"], "pcm_sha256": ["a" * 64]}


def candidate(**kw):
    base = {"key": "k", "dataset": "events-db", "subset": None, "freesound_id": 99, "uploader": "bob",
            "pcm_sha256": "b" * 64}
    return base | kw


class TrainPoolCheck(unittest.TestCase):
    def test_a_clean_pool_passes(self):
        res = noise.check_train_pool([candidate(), candidate(key="m", dataset="mix-db", subset="noise/other",
                                                            freesound_id=None, uploader=None, pcm_sha256="c" * 64),
                                      candidate(key="r", dataset="rooms-db", room="R2", freesound_id=None,
                                                uploader=None, pcm_sha256="d" * 64)], BLOCKLIST)
        self.assertTrue(res["pass"], res)

    def test_each_kind_of_intersection_fails(self):
        cases = {
            "reserved unit": candidate(dataset="ambient-db"),
            "subset unit": candidate(dataset="mix-db", subset="speech/gov"),
            "room unit": candidate(dataset="rooms-db", room="R1"),
            "family": candidate(dataset="nc-db"),
            "sound id": candidate(freesound_id=12),
            "uploader": candidate(uploader="alice"),
            "pcm": candidate(pcm_sha256="a" * 64),
            "no pcm": candidate(pcm_sha256=None),
        }
        for name, c in cases.items():
            res = noise.check_train_pool([candidate(key="ok"), c | {"key": name}], BLOCKLIST)
            self.assertFalse(res["pass"], name)
            self.assertEqual([v["key"] for v in res["violations"]], [name])

    def test_license_blocks_of_the_noise_subset(self):
        text = """noise-x-0046
Title: Door Slam
License: Attribution 3.0
Recorded by someone
http://example.org/1-Door-Slam.html
========================================================================
noise-x-0003
Title: Rain On Roof
License: Attribution 3.0
Recorded by other
License: Attribution 3.0 
http://example.org/2-Rain.html
========================================================================
noise-x-0045
Title: Football Crowd
License: Public Domain
Recorded by third
http://example.org/3-Crowd.html
========================================================================
noise-x-0009
Title: Mixed
License: Public Domain
License: Attribution 3.0
http://example.org/4.html
"""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "LICENSE"
            p.write_text(text)
            b = noise.read_license_blocks(p, "noise-x-")
        self.assertEqual(sorted(b), ["noise-x-0003", "noise-x-0009", "noise-x-0045", "noise-x-0046"])
        self.assertEqual(b["noise-x-0046"]["license"], "attribution")
        self.assertEqual(b["noise-x-0003"]["license"], "attribution")
        self.assertEqual(b["noise-x-0045"]["license"], "public domain")
        self.assertIsNone(b["noise-x-0009"]["license"])                   # two kinds: not accepted
        self.assertEqual(b["noise-x-0046"]["recorded_by"], "someone")
        self.assertTrue(noise.voice_title(b["noise-x-0045"]["title"]))
        self.assertFalse(noise.voice_title(b["noise-x-0003"]["title"]))

    def test_pcm_hash_is_the_noisy_dev_definition(self):
        with tempfile.TemporaryDirectory() as tmp:
            x = tone(0.5)
            p = write_clip(Path(tmp) / "a.flac", x)
            import soundfile as sf
            a, rate = sf.read(str(p), dtype="float32", always_2d=True)
            expected = hashlib.sha256(f"{rate},1\n".encode() + np.ascontiguousarray(a, dtype="<f4").tobytes()).hexdigest()
            self.assertEqual(noise.pcm_sha256(p), expected)
            noise.noisy_dev_on_path()
            try:
                import noisy_sources
            except Exception:  # noqa: BLE001  (its roots may be unset here)
                return
            self.assertEqual(noise.pcm_sha256(p), noisy_sources.pcm_sha256(p))

    def test_the_augmenter_refuses_a_pool_that_failed_the_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "pool.json"
            p.write_text(json.dumps({"noise": [{"path": "n"}], "rirs": [{"path": "r"}], "train_check": {"pass": False}}))
            with self.assertRaises(RuntimeError):
                augment.Pool.load(p)
            p.write_text(json.dumps({"noise": [{"path": "n"}], "rirs": [{"path": "r"}], "train_check": {"pass": True}}))
            self.assertEqual(augment.Pool.load(p).noise_paths, ["n"])


REG = {"syn-tts-a": {"license": "apache-2.0", "outputs": "no terms", "voice_assets": "table",
                     "training_data": "studio corpus, CC BY 4.0", "voice_id_range": [225, 376]},
       "syn-tts-b": {"license": "mit", "outputs": "no terms", "voice_assets": "table",
                     "training_data": "train split, CC BY 4.0", "voice_split": "train-clean-360"}}
SPEAKERS = {"14": "train-clean-360", "16": "train-clean-360", "19": "train-clean-100", "84": "dev-clean",
            "61": "test-clean"}


class VoiceCheck(unittest.TestCase):
    def test_engine_b_passes_only_with_documented_split_speakers_outside_dev_and_test(self):
        with roots(REG):
            ok = voicecheck.check_engine("syn-tts-b", ["p14", "p16"], SPEAKERS, {"1272"})
            self.assertEqual(ok["verdict"], "pass", ok)
            for table, why in ((["p14", "p84"], "dev"), (["p14", "p61"], "test"), (["p14", "p19"], "split"),
                               (["p14", "p1272"], "manifest")):
                r = voicecheck.check_engine("syn-tts-b", table, SPEAKERS | {"1272": "train-clean-360"}, {"1272"})
                self.assertEqual(r["verdict"], "fail", why)

    def test_engine_a_undocumented_entries_are_excluded_and_out_of_range_ids_fail(self):
        with roots(REG):
            r = voicecheck.check_engine("syn-tts-a", ["ED\n", "p225", "p376"], {}, set())
            self.assertEqual(r["verdict"], "pass", r)
            self.assertEqual(r["excluded_entries"], ["ED\n"])
            self.assertEqual(r["voices"], 2)
            self.assertEqual(voicecheck.check_engine("syn-tts-a", ["p225", "p999"], {}, set())["verdict"], "fail")

    def test_restrictive_or_missing_licenses_fail(self):
        bad = json.loads(json.dumps(REG))
        bad["syn-tts-a"]["license"] = "CC BY-NC-ND 4.0"
        with roots(bad):
            self.assertEqual(voicecheck.check_engine("syn-tts-a", ["p225"], {}, set())["verdict"], "fail")
        bad = json.loads(json.dumps(REG))
        del bad["syn-tts-b"]["outputs"]
        with roots(bad):
            self.assertEqual(voicecheck.check_engine("syn-tts-b", ["p14"], SPEAKERS, set())["verdict"], "fail")

    def test_tts_jobs_refuse_an_engine_that_did_not_pass(self):
        with roots(REG):
            with self.assertRaises(RuntimeError):
                voicecheck.require_pass("syn-tts-a")
            out = layout.base() / "voice_check.json"
            out.parent.mkdir(parents=True)
            out.write_text(json.dumps({"engines": {"syn-tts-a": {"verdict": "fail"}, "syn-tts-b": {"verdict": "pass"}}}))
            with self.assertRaises(RuntimeError):
                voicecheck.require_pass("syn-tts-a")
            voicecheck.require_pass("syn-tts-b")


class Assignment(unittest.TestCase):
    def sentences(self, n):
        return [{"id": f"r-{i:06d}", "domain": layout.DOMAINS[i % 4], "label": f"Sentence {i}."} for i in range(n)]

    def test_every_voice_is_used_balanced_and_engines_alternate_per_domain(self):
        voices = {"syn-tts-a": [f"p{i}" for i in range(225, 235)], "syn-tts-b": [f"p{i}" for i in range(1, 8)]}
        plan = tts.assign(self.sentences(400), voices, seed=5)
        self.assertEqual(plan, tts.assign(self.sentences(400), voices, seed=5))
        for e, vs in voices.items():
            c = collections.Counter(r["voice"] for r in plan if r["engine"] == e)
            self.assertEqual(set(c), set(vs))
            self.assertLessEqual(max(c.values()) - min(c.values()), 1)
        for d in layout.DOMAINS:
            c = collections.Counter(r["engine"] for r in plan if r["domain"] == d)
            self.assertLessEqual(abs(c["syn-tts-a"] - c["syn-tts-b"]), 1)
        self.assertEqual(sorted(r["id"] for r in plan), sorted(s["id"] for s in self.sentences(400)))


class RoundTrip(unittest.TestCase):
    def test_keep_rule(self):
        text = "the rate constant doubles when the temperature rises by ten degrees"   # 12 words
        self.assertEqual(roundtrip.decide(text, text, False, 3.0)[:2], (True, None))
        three = "the rate constant halves when the temperature rises by ten degrees"     # 1 substitution
        self.assertTrue(roundtrip.decide(text, three, False, 3.0)[0])
        bad = "a great constant halved once the temperatures rose by tens of degrees"
        self.assertEqual(roundtrip.decide(text, bad, False, 3.0)[1], "wer_gt_0.25")
        self.assertEqual(roundtrip.decide(text, text, True, 3.0)[1], "hit_cap")
        self.assertEqual(roundtrip.decide(text, text, False, 0.5)[1], "duration")
        self.assertEqual(roundtrip.decide(text, text, False, 31.0)[1], "duration")
        # exactly 0.25 keeps (<= 0.25): 1 error in 4 words
        self.assertTrue(roundtrip.decide("alpha beta gamma delta", "alpha beta gamma omega", False, 2.0)[0])
        self.assertFalse(roundtrip.decide("alpha beta gamma delta", "alpha beta sigma omega", False, 2.0)[0])


class Finalize(unittest.TestCase):
    def world(self, t, split, n=80, cross=None):
        """A finished run of n clips; every fifth clip rejected by T*. cross: (index, extra term) puts a second,
        other-domain term into one clip's text that T* does not reproduce."""
        targeted = {d: split.targeted(d)[0].term for d in layout.DOMAINS}
        d = t / "run"
        prompts, sents, plan, rt = [], [], [], []
        for i in range(n):
            dom = layout.DOMAINS[i % 4]
            term = targeted[dom]
            pid = f"r-{i:06d}"
            label = f"Today the {term} came up again in our meeting."
            hyp = label
            if cross and cross[0] == i:
                label = f"Today the {term} and the {cross[1]} came up again in our meeting."
                hyp = f"Today the {term} and the something came up again in our meeting."
            prompts.append({"id": pid, "domain": dom, "register": "explanatory", "requested": [term]})
            sents.append({"id": pid, "domain": dom, "label": label, "requested": [term]})
            eng = layout.TTS_ENGINES[i % 2]
            path = write_clip(d / "audio" / f"{pid}.flac", tone(1.5 + (i % 3)))
            plan.append({"id": pid, "domain": dom, "label": label, "tts_text": label, "engine": eng, "voice": "p225",
                         "seed": i})
            write_jsonl(d / "tts" / eng / f"{i:05d}.jsonl", [plan[-1] | {
                "audio_filepath": str(path), "duration": 1.5 + (i % 3), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}])
            kept = i % 5 != 0
            rt.append({"id": pid, "hyp": hyp if kept else "something else entirely", "hit_cap": False,
                       "wer": 0.0 if kept else 1.0, "kept": kept, "reason": None if kept else "wer_gt_0.25"})
        write_jsonl(d / "prompts.jsonl", prompts)
        write_jsonl(d / "sentences.jsonl", sents)
        write_jsonl(d / "tts_plan.jsonl", plan)
        write_jsonl(d / "roundtrip" / "00000.jsonl", rt)
        (d / "hygiene.json").write_text(json.dumps({"generated": n, "kept": n, "reasons": {},
                                                    "heldout": {"provisional": True}}))
        layout.split_path().parent.mkdir(parents=True, exist_ok=True)
        layout.split_path().write_text(json.dumps(split.data))
        return d, sents

    def test_only_clips_kept_by_the_current_hygiene_are_published(self):
        split = make_split()
        with roots() as t:
            d, sents = self.world(t, split)
            # A hygiene rerun (e.g. with eval-stem present) drops two sentences and is no longer provisional.
            dropped = {sents[1]["id"], sents[2]["id"]}
            write_jsonl(d / "sentences.jsonl", [s for s in sents if s["id"] not in dropped])
            (d / "hygiene.json").write_text(json.dumps({"generated": 80, "kept": 78, "reasons": {"denylist": 2},
                                                        "heldout": {"provisional": False}}))
            rep = roundtrip.finalize(d, split, listen=False)
            ids = {c["id"] for c in read_jsonl(d / "clips.jsonl")}
            self.assertFalse(ids & dropped)
            self.assertEqual(len(ids), 64 - 2)
            self.assertEqual(rep["excluded_by_current_hygiene"], 2)
            meta = json.loads((d / "clips.meta.json").read_text())
            self.assertFalse(meta["provisional"])
            self.assertEqual(sorted(meta["excluded_by_current_hygiene"]), sorted(dropped))
            # A changed label is not published either.
            write_jsonl(d / "sentences.jsonl", [s | ({"label": s["label"] + " Really."} if s["id"] == sents[3]["id"] else {})
                                                for s in sents])
            roundtrip.finalize(d, split, listen=False)
            self.assertNotIn(sents[3]["id"], {c["id"] for c in read_jsonl(d / "clips.jsonl")})

    def test_term_agreement_is_counted_under_the_terms_domain(self):
        split = make_split()
        other = split.targeted("software_engineering")[0].term
        with roots() as t:
            d, _ = self.world(t, split, n=8, cross=(1, other))    # clip 1: a mathematics prompt, kept by T*
            rep = roundtrip.finalize(d, split, listen=False)
            self.assertEqual(rep["domains"]["mathematics"]["term_agreement"]["kept/targeted"]["rate"], 1.0)
            se = rep["domains"]["software_engineering"]["term_agreement"]["kept/targeted"]
            self.assertEqual((se["occurrences"], se["reproduced"]), (3, 2))   # clips 3 and 7, and clip 1's miss

    def test_report_clips_and_listening_sample(self):
        split = make_split()
        targeted = {d: split.targeted(d)[0].term for d in layout.DOMAINS}
        with roots() as t:
            d = t / "run"
            prompts, sents, plan, rt = [], [], [], []
            for i in range(80):
                dom = layout.DOMAINS[i % 4]
                term = targeted[dom]
                pid = f"r-{i:06d}"
                label = f"Today the {term} came up again in our meeting."
                prompts.append({"id": pid, "domain": dom, "register": "explanatory", "requested": [term]})
                sents.append({"id": pid, "domain": dom, "label": label, "requested": [term]})
                eng = layout.TTS_ENGINES[i % 2]
                path = write_clip(d / "audio" / f"{pid}.flac", tone(1.5 + (i % 3)))
                plan.append({"id": pid, "domain": dom, "label": label, "tts_text": label, "engine": eng, "voice": "p225",
                         "seed": i})
                write_jsonl(d / "tts" / eng / f"{i:05d}.jsonl", [plan[-1] | {
                    "audio_filepath": str(path), "duration": 1.5 + (i % 3), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}])
                kept = i % 5 != 0
                rt.append({"id": pid, "hyp": label if kept else "something else entirely", "hit_cap": False,
                           "wer": 0.0 if kept else 1.0, "kept": kept, "reason": None if kept else "wer_gt_0.25"})
            write_jsonl(d / "prompts.jsonl", prompts)
            write_jsonl(d / "sentences.jsonl", sents)
            write_jsonl(d / "tts_plan.jsonl", plan)
            write_jsonl(d / "roundtrip" / "00000.jsonl", rt)
            (d / "hygiene.json").write_text(json.dumps({"generated": 80, "kept": 80, "reasons": {},
                                                        "heldout": {"provisional": True}}))
            layout.split_path().parent.mkdir(parents=True)
            layout.split_path().write_text(json.dumps(split.data))
            rep = roundtrip.finalize(d, split)
            clips = read_jsonl(d / "clips.jsonl")
            self.assertEqual(len(clips), 64)
            self.assertTrue(all(c["id"] in {r["id"] for r in rt if r["kept"]} for c in clips))
            self.assertEqual(rep["total"]["kept"], 64)
            rend = sum(1.5 + (i % 3) for i in range(80))
            kept_s = sum(1.5 + (i % 3) for i in range(80) if i % 5)
            self.assertAlmostEqual(rep["total"]["retained_hours_per_generated_hour"], kept_s / rend)
            for dom in layout.DOMAINS:
                cov = rep["domains"][dom]["term_coverage"]
                self.assertEqual((cov["requested"], cov["in_kept_clips"]), (1, 1))
                ag = rep["domains"][dom]["term_agreement"]
                self.assertEqual(ag["kept/targeted"]["rate"], 1.0)
                self.assertEqual(ag["rejected/targeted"]["rate"], 0.0)
                kept_dir, rej_dir = d / "listening" / dom / "kept", d / "listening" / dom / "rejected"
                self.assertEqual(len(list(kept_dir.glob("*.flac"))), 16)        # all kept (fewer than 50)
                self.assertEqual(len(list(rej_dir.glob("*.flac"))), 4)
            meta = json.loads((d / "clips.meta.json").read_text())
            self.assertTrue(meta["provisional"])
            from synthetic import mixing
            with self.assertRaises(RuntimeError):
                mixing.ClipPool.load(d)
            self.assertEqual(len(mixing.ClipPool.load(d, allow_provisional=True).clips), 64)


if __name__ == "__main__":
    unittest.main()
