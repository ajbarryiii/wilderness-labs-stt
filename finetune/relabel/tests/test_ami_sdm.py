"""Contract tests: the AMI distant-microphone (SDM) pair store (keab-prime plan "Data", "AMI distant microphone (new)";
"Code and review", "AMI SDM").

- pairing: each SDM row with the IHM row of the same meeting, speaker, begin and end time; the rest unpaired, by
  reason, and counted;
- held out: no pair from an AMI dev or test meeting (meeting ids from the ami_dev manifest and the ESB AMI test
  manifest), counted; a train meeting that is held out stops the build;
- labels: a pair is kept only if its IHM id has a stored label kept under T*'s disposition; the label read for a
  row is its IHM id's (T* for the main arm, B0's under the same mask for the control); the store holds no label;
- equal encoder frame counts (NeMo's formula for Parakeet TDT 0.6B v2); unequal or undecodable pairs dropped and
  counted;
- the IHM PCM checked against C1's stored SHA-256;
- the store's read interface: SDM and IHM audio, sample counts and PCM SHA-256 verified, the clip-store format the
  batch plan reads;
- the AMI SDM dev manifest; the draws per clip from C1's journal.

Run from finetune/relabel:  ./python -m unittest discover -s tests -p 'test_ami_sdm*' -v
"""
from __future__ import annotations

import collections
import hashlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

import ami_sdm
from labeling import shards
from labeling_fixtures import relabel_root
from training import ami_store
from training import batchplan as bp

SR = 16000


def pcm_of(seed: int, n: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(-3000, 3000, n, dtype=np.int16)


def row(audio_id: str, meeting: str, speaker: str, begin: float, end: float, mic: str = "SDM1",
        text: str = "SOME WORDS") -> dict:
    return {"audio_id": audio_id, "meeting_id": meeting, "speaker_id": speaker, "begin_time": begin,
            "end_time": end, "microphone_id": mic, "text": text}


def sdm(meeting: str, speaker: str, begin: float, end: float, **kw) -> dict:
    return row(f"AMI_{meeting}_sdm_{speaker}_{round(begin * 100):07d}_{round(end * 100):07d}", meeting, speaker,
               begin, end, **kw)


def ihm(meeting: str, speaker: str, begin: float, end: float, mic: str = "H00", **kw) -> dict:
    return row(f"AMI_{meeting}_{mic}_{speaker}_{round(begin * 100):07d}_{round(end * 100):07d}", meeting, speaker,
               begin, end, mic=mic, **kw)


class Frames(unittest.TestCase):
    def test_encoder_frames_is_nemos_and_the_stage_3_trainers(self):
        """NeMo's own length functions with Parakeet TDT 0.6B v2's preprocessor (n_fft 512, hop 160, centered) and
        its 8x subsampling (three stride-2 convolutions, kernel 3, padding 1), and train_b0prime.encoder_frames."""
        import torch
        from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures
        from nemo.collections.asr.parts.submodules.subsampling import calc_length

        import train_b0prime as tb
        pre = types.SimpleNamespace(stft_pad_amount=None, n_fft=512, hop_length=160)
        for samples in (160, 16000, 16001, 16799, 16800, 20480, 123457, 480000):
            mel = FilterbankFeatures.get_seq_len(pre, torch.tensor([samples]))
            nemo = calc_length(mel.float(), all_paddings=2, kernel_size=3, stride=2, ceil_mode=False, repeat_num=3)
            self.assertEqual(ami_store.encoder_frames(samples), int(nemo.item()), samples)
        for samples in range(1, 600_000, 997):
            self.assertEqual(ami_store.encoder_frames(samples), tb.encoder_frames(samples), samples)
        self.assertEqual((ami_store.encoder_frames(16799), ami_store.encoder_frames(16800)), (13, 14))


class Meetings(unittest.TestCase):
    def test_meeting_of_pool_dev_and_esb_ids(self):
        self.assertEqual(ami_sdm.meeting_of("ami:AMI_ES2011a_H00_FEE041_0003427_0003714"), "ES2011a")
        self.assertEqual(ami_sdm.meeting_of("AMI_EN2002a_H00_MEE073_0000096_0000665"), "EN2002a")
        self.assertEqual(ami_sdm.meeting_of("AMI_TS3009a_sdm_MTD033PM_0103448_0104838"), "TS3009a")
        for bad in ("ES2011a_H00", "ami:XYZ_ES2011a_H00_F_1_2", ""):
            with self.assertRaises(ValueError):
                ami_sdm.meeting_of(bad)

    def test_held_out_meetings_from_the_dev_and_test_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            dev, test = Path(tmp) / "ami_dev.jsonl", Path(tmp) / "test_ami.jsonl"
            dev.write_text("".join(json.dumps({"id": f"ami:AMI_{m}_H00_FEE041_0000001_0000002"}) + "\n"
                                   for m in ("ES2011a", "ES2011b", "ES2011a")))
            test.write_text(json.dumps({"id": "AMI_EN2002a_H00_MEE073_0000096_0000665"}) + "\n")
            held = ami_sdm.held_out_meetings(dev, test)
        self.assertEqual(held, {"dev": {"ES2011a", "ES2011b"}, "test": {"EN2002a"}})

    def test_a_held_out_train_meeting_stops_the_build(self):
        held = {"dev": {"ES2011a"}, "test": {"EN2002a"}}
        ami_sdm.check_disjoint({"ES2002a", "TS3009a"}, held)
        for train in ({"ES2002a", "ES2011a"}, {"EN2002a"}):
            with self.assertRaises(RuntimeError):
                ami_sdm.check_disjoint(train, held)


class Pairing(unittest.TestCase):
    def setUp(self):
        self.ihm = [ihm("ES2002a", "FEE005", 23.18, 25.11), ihm("ES2002a", "MEE006", 23.18, 25.11, mic="H01"),
                    ihm("ES2002a", "FEE005", 30.0, 31.5), ihm("ES2002b", "FEE005", 40.0, 42.0),
                    ihm("TS3009a", "MTD033PM", 1.0, 2.0), ihm("TS3009a", "MTD033PM", 1.0, 2.0, mic="H01")]
        self.sdm = [sdm("ES2002a", "FEE005", 23.18, 25.11),     # pairs with the first IHM row (not MEE006's)
                    sdm("ES2002a", "MEE006", 23.18, 25.11),     # pairs with MEE006's row of the same times
                    sdm("ES2002a", "FEE005", 30.0, 31.6),       # same meeting and speaker, other end time
                    sdm("ES2002a", "FEO070", 23.18, 25.11),     # a speaker without IHM rows in that meeting
                    sdm("IS1009c", "FEE005", 40.0, 42.0),       # a meeting without IHM rows
                    sdm("TS3009a", "MTD033PM", 1.0, 2.0),       # two IHM rows with its key: ambiguous
                    sdm("ES2002b", "FEE005", 40.0, 42.0),
                    sdm("ES2002b", "FEE005", 40.0, 42.0, text="OTHER")]   # two SDM rows with one key: ambiguous

    def test_pairs_by_meeting_speaker_begin_and_end(self):
        pairs, unpaired = ami_sdm.pair(self.sdm, self.ihm)
        got = {p["sdm"]["audio_id"]: p["ihm"]["audio_id"] for p in pairs}
        self.assertEqual(got, {self.sdm[0]["audio_id"]: self.ihm[0]["audio_id"],
                               self.sdm[1]["audio_id"]: self.ihm[1]["audio_id"]})
        for p in pairs:
            self.assertEqual(ami_sdm.pair_key(p["sdm"]), ami_sdm.pair_key(p["ihm"]))

    def test_unpaired_rows_by_reason_and_counted(self):
        pairs, unpaired = ami_sdm.pair(self.sdm, self.ihm)
        self.assertEqual({k: sorted(v) for k, v in unpaired.items()}, {
            "no_ihm_time": [self.sdm[2]["audio_id"]],
            "no_ihm_speaker": [self.sdm[3]["audio_id"]],
            "no_ihm_meeting": [self.sdm[4]["audio_id"]],
            "ambiguous_ihm_key": [self.sdm[5]["audio_id"]],
            "ambiguous_sdm_key": sorted([self.sdm[6]["audio_id"], self.sdm[7]["audio_id"]])})
        self.assertEqual(len(pairs) + sum(map(len, unpaired.values())), len(self.sdm))
        self.assertEqual(set(unpaired), set(ami_sdm.UNPAIRED))

    def test_times_are_compared_exactly(self):
        a = sdm("ES2002a", "FEE005", 23.18, 25.11)
        b = dict(a, begin_time=float(np.nextafter(np.float32(23.18), np.float32(30))))
        pairs, unpaired = ami_sdm.pair([b], [ihm("ES2002a", "FEE005", 23.18, 25.11)])
        self.assertEqual((pairs, list(unpaired["no_ihm_time"])), ([], [b["audio_id"]]))


class Selection(unittest.TestCase):
    def setUp(self):
        rows = [("ES2002a", "FEE005", 1.0, 2.0), ("ES2002a", "FEE005", 3.0, 4.0), ("ES2002a", "FEE005", 5.0, 6.0),
                ("ES2011a", "FEE041", 1.0, 2.0), ("EN2002a", "MEE073", 1.0, 2.0), ("TS3009a", "MTD033PM", 1.0, 2.0)]
        self.pairs = [{"sdm": sdm(*r), "ihm": ihm(*r)} for r in rows]
        self.held = {"dev": {"ES2011a"}, "test": {"EN2002a"}}
        ids = [ami_sdm.ihm_pool_id(p["ihm"]["audio_id"]) for p in self.pairs]
        self.dispositions = {ids[0]: "kept", ids[1]: "rejected", ids[3]: "kept", ids[4]: "kept", ids[5]: "kept"}

    def test_held_out_meetings_and_labels_select_and_count(self):
        selected, drops = ami_sdm.select(self.pairs, self.held, self.dispositions)
        self.assertEqual([s["sdm_id"] for s in selected],
                         [self.pairs[0]["sdm"]["audio_id"], self.pairs[5]["sdm"]["audio_id"]])
        self.assertEqual(drops, {"held_out_dev": 1, "held_out_test": 1, "no_label_record": 1, "not_kept": 1})
        self.assertEqual(len(selected) + sum(drops.values()), len(self.pairs))
        for s in selected:
            self.assertNotIn(s["meeting_id"], self.held["dev"] | self.held["test"])
            self.assertEqual(self.dispositions[s["ihm_id"]], "kept")
            self.assertTrue(s["ihm_id"].startswith("ami:AMI_") and s["ihm_id"][4:] == s["ihm_audio_id"])

    def test_a_held_out_meeting_is_dropped_even_with_a_kept_label(self):
        selected, _ = ami_sdm.select(self.pairs, self.held, self.dispositions)
        self.assertNotIn("ES2011a", {s["meeting_id"] for s in selected})
        self.assertIn(ami_sdm.ihm_pool_id(self.pairs[3]["ihm"]["audio_id"]), self.dispositions)


def parquet_bytes(rows: list[dict]) -> bytes:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buf, row_group_size=2)
    return buf.getvalue()


def flac_of(pcm: np.ndarray) -> bytes:
    return shards.encode_flac(pcm)


def wav_of(pcm: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, pcm, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class Decoding(unittest.TestCase):
    """A parquet file in the Hub's AMI schema: its index rows and decoded clips are the pool's decoding."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.pcm = {"a": pcm_of(1, 20000), "b": pcm_of(2, 17000), "c": np.zeros(9000, np.int16)}
        audio = {"a": flac_of(self.pcm["a"]), "b": wav_of(self.pcm["b"]), "c": flac_of(self.pcm["c"]),
                 "d": b"not audio"}
        rows = [dict(sdm("ES2002a", "FEE005", i, i + 1.0), audio={"bytes": audio[k], "path": None})
                for i, k in enumerate("abcd")]
        self.ids = {k: r["audio_id"] for k, r in zip("abcd", rows)}
        self.path = self.root / "sdm-train-00000-of-00001.parquet"
        self.path.write_bytes(parquet_bytes(rows))

    def tearDown(self):
        self.tmp.cleanup()

    def test_index_rows(self):
        rows = ami_sdm.index_file(self.path, "sdm/train-00000-of-00001.parquet")
        self.assertEqual([r["audio_id"] for r in rows], [self.ids[k] for k in "abcd"])
        self.assertEqual([r["row"] for r in rows], [0, 1, 2, 3])
        self.assertEqual(rows[0]["meeting_id"], "ES2002a")
        self.assertEqual(rows[1]["begin_time"], float(np.float32(1.0)))
        self.assertNotIn("audio", rows[0])

    def test_decoded_clips_are_the_pools_samples(self):
        import data  # finetune/parakeet-ternary: the pool's decoding (labeling/pool_stream.py)
        out = self.root / "audio"
        recs = {r["id"]: r for r in ami_sdm.decode_file(self.path, [self.ids[k] for k in "abcd"], out)}
        for k in "ab":
            r = recs[self.ids[k]]
            self.assertEqual(r["status"], "ok")
            self.assertEqual(r["samples"], len(self.pcm[k]))
            self.assertEqual(r["pcm_sha256"], shards.pcm_sha256(self.pcm[k]))
            raw = Path(r["audio_filepath"]).read_bytes()
            self.assertEqual(r["sha256"], hashlib.sha256(raw).hexdigest())
            np.testing.assert_array_equal(shards.decode_flac(raw), self.pcm[k])
            src = (flac_of if k == "a" else wav_of)(self.pcm[k])
            np.testing.assert_array_equal(data.convert(src)[0], self.pcm[k])
        self.assertEqual((recs[self.ids["c"]]["status"], recs[self.ids["c"]]["reason"]), ("undecodable", "silent_audio"))
        self.assertEqual((recs[self.ids["d"]]["status"], recs[self.ids["d"]]["reason"]), ("undecodable", "decode_error"))
        self.assertFalse(any(Path(out).glob(f"{self.ids['d']}*")))

    def test_only_wanted_rows_are_decoded(self):
        recs = ami_sdm.decode_file(self.path, [self.ids["b"]], self.root / "audio")
        self.assertEqual([r["id"] for r in recs], [self.ids["b"]])
        with self.assertRaises(RuntimeError):
            ami_sdm.decode_file(self.path, [self.ids["b"], "AMI_XX0000a_sdm_X_0000000_0000001"], self.root / "audio")


def make_clips(root: Path, specs) -> list[dict]:
    """Clip records (the store's format) for (sdm id, ihm id, sdm pcm, ihm pcm) specs, with FLAC files."""
    clips = []
    for sid, hid, spcm, hpcm in specs:
        files = {}
        for side, pcm, name in (("sdm", spcm, sid), ("ihm", hpcm, hid.removeprefix("ami:"))):
            path = root / side / f"{name}.flac"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(flac_of(pcm))
            files[side] = {"audio_filepath": str(path), "samples": len(pcm),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "pcm_sha256": shards.pcm_sha256(pcm)}
        meeting = ami_sdm.meeting_of(sid)
        clips.append({"id": sid, **files["sdm"], "duration": len(spcm) / SR, "frames": ami_store.encoder_frames(len(spcm)),
                      "ihm_id": hid, **{f"ihm_{k}": v for k, v in files["ihm"].items()},
                      "meeting_id": meeting, "speaker_id": sid.split("_")[3], "begin_time": 1.0, "end_time": 2.0})
    return clips


class Assemble(unittest.TestCase):
    """Decoded pairs to clip records: equal encoder frames, both sides decodable, IHM PCM against C1's."""

    def rec(self, aid: str, n: int | None, seed: int = 0, status: str = "ok") -> dict:
        if status != "ok":
            return {"id": aid, "status": status, "reason": "decode_error"}
        pcm = pcm_of(seed, n)
        return {"id": aid, "status": "ok", "audio_filepath": f"/x/{aid}.flac", "samples": n,
                "sha256": "f" * 64, "pcm_sha256": shards.pcm_sha256(pcm)}

    def setUp(self):
        keys = [("ES2002a", "FEE005", float(i), i + 1.0) for i in range(5)]
        self.sel = [{"sdm_id": sdm(*k)["audio_id"], "ihm_audio_id": ihm(*k)["audio_id"],
                     "ihm_id": ami_sdm.ihm_pool_id(ihm(*k)["audio_id"]), "meeting_id": k[0], "speaker_id": k[1],
                     "begin_time": k[2], "end_time": k[3]} for k in keys]
        s, h = [x["sdm_id"] for x in self.sel], [x["ihm_audio_id"] for x in self.sel]
        self.sdm_rec = {s[0]: self.rec(s[0], 16000, 1), s[1]: self.rec(s[1], 16800, 2), s[2]: self.rec(s[2], 0, status="undecodable"),
                        s[3]: self.rec(s[3], 20000, 4), s[4]: self.rec(s[4], 30000, 5)}
        self.ihm_rec = {h[0]: self.rec(h[0], 16799, 11), h[1]: self.rec(h[1], 16799, 12), h[2]: self.rec(h[2], 16000, 13),
                        h[3]: self.rec(h[3], 0, status="undecodable"), h[4]: self.rec(h[4], 30000, 15)}
        self.c1 = {self.sel[0]["ihm_id"]: {"samples": 16799, "pcm_sha256": self.ihm_rec[h[0]]["pcm_sha256"]},
                   self.sel[4]["ihm_id"]: {"samples": 30000, "pcm_sha256": "0" * 64}}

    def test_equal_frames_kept_and_unequal_dropped_and_counted(self):
        clips, report = ami_sdm.assemble(self.sel, self.sdm_rec, self.ihm_rec, self.c1)
        self.assertEqual([c["id"] for c in clips], sorted([self.sel[0]["sdm_id"], self.sel[4]["sdm_id"]]))
        self.assertEqual(report["drops"], {"sdm_undecodable": 1, "ihm_undecodable": 1, "frames_differ": 1})
        for c in clips:
            self.assertEqual(c["frames"], ami_store.encoder_frames(c["samples"]))
            self.assertEqual(c["frames"], ami_store.encoder_frames(c["ihm_samples"]))
            self.assertEqual(c["duration"], c["samples"] / SR)
            self.assertNotIn("label", c)
            self.assertNotIn("text", c)
        first = next(c for c in clips if c["id"] == self.sel[0]["sdm_id"])
        self.assertEqual((first["samples"], first["ihm_samples"]), (16000, 16799))    # 13 frames each
        self.assertEqual(first["ihm_id"], self.sel[0]["ihm_id"])

    def test_c1_pcm_checked_and_mismatches_reported(self):
        _, report = ami_sdm.assemble(self.sel, self.sdm_rec, self.ihm_rec, self.c1)
        c1 = report["c1"]
        self.assertEqual((c1["checked"], c1["matched"], c1["not_in_c1"]), (2, 1, 0))
        self.assertEqual(c1["mismatched"], [self.sel[4]["ihm_id"]])
        with self.assertRaises(RuntimeError):
            ami_sdm.require_c1_match(report)
        del self.c1[self.sel[4]["ihm_id"]]
        _, report = ami_sdm.assemble(self.sel, self.sdm_rec, self.ihm_rec, self.c1)
        self.assertEqual((report["c1"]["checked"], report["c1"]["not_in_c1"], report["c1"]["mismatched"]), (1, 1, []))
        ami_sdm.require_c1_match(report)

    def test_c1_sample_count_is_part_of_the_check(self):
        self.c1[self.sel[0]["ihm_id"]]["samples"] = 16800
        _, report = ami_sdm.assemble(self.sel, self.sdm_rec, self.ihm_rec, self.c1)
        self.assertIn(self.sel[0]["ihm_id"], report["c1"]["mismatched"])


class Store(unittest.TestCase):
    """The pair store's read interface (training/ami_store.py), as the batch plan and the trainer read it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.pcm = {i: (pcm_of(i, 16000 + 37 * i), pcm_of(100 + i, 16000 + 37 * i)) for i in range(3)}
        self.specs = [(f"AMI_ES2002a_sdm_FEE005_{i:07d}_{i + 100:07d}", f"ami:AMI_ES2002a_H00_FEE005_{i:07d}_{i + 100:07d}",
                       *self.pcm[i]) for i in (2, 0, 1)]
        self.clips = make_clips(self.root / "audio", self.specs)
        self.dir = self.root / "store"
        ami_store.write_store(self.dir, self.clips, {"source": "fixture"})

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip(self):
        store = ami_store.PairStore.load(self.dir)
        self.assertEqual([c["id"] for c in store.clips], sorted(s[0] for s in self.specs))
        self.assertEqual(store.identity, hashlib.sha256((self.dir / "clips.jsonl").read_bytes()).hexdigest())
        meta = json.loads((self.dir / "clips.meta.json").read_text())
        self.assertEqual((meta["clips"], meta["complete"], meta["provisional"]), (3, True, False))
        self.assertAlmostEqual(meta["hours"], sum(len(s[2]) for s in self.specs) / SR / 3600)
        for sid, hid, spcm, hpcm in self.specs:
            a, t = store.read(sid), store.read_teacher(sid)
            self.assertEqual((a.dtype, t.dtype), (np.float32, np.float32))
            np.testing.assert_array_equal(a, spcm.astype(np.float32) / 32768.0)
            np.testing.assert_array_equal(t, hpcm.astype(np.float32) / 32768.0)
            self.assertEqual(store.ihm_id(sid), hid)

    def test_clip_store_format_for_the_batch_plan(self):
        store = ami_store.PairStore.load(self.dir)
        self.assertEqual(bp.clip_samples(store), [c["samples"] for c in store.clips])
        src = bp.Source("ami-sdm", ami_store.PREFIX, store, 1, 0.5, False)
        self.assertEqual([src.key(n) for n in bp.clip_samples(store)], [c["samples"] for c in store.clips])
        self.assertEqual(src.config()["clip_pool"], store.identity)
        for c in store.clips:
            self.assertNotIn("label", c)

    def test_a_changed_flac_is_refused(self):
        store = ami_store.PairStore.load(self.dir)
        sid = self.specs[0][0]
        clip = store.by_id[sid]
        Path(clip["audio_filepath"]).write_bytes(flac_of(pcm_of(999, clip["samples"])))
        with self.assertRaises(RuntimeError):
            store.read(sid)
        Path(clip["ihm_audio_filepath"]).write_bytes(flac_of(pcm_of(998, clip["ihm_samples"])))
        with self.assertRaises(RuntimeError):
            store.read_teacher(sid)

    def test_sample_count_and_pcm_sha256_are_verified(self):
        for field, value in (("samples", None), ("pcm_sha256", "0" * 64), ("ihm_samples", None),
                             ("ihm_pcm_sha256", "0" * 64)):
            clips = [dict(c) for c in self.clips]
            c = clips[0]
            c[field] = value if value is not None else c[field] + 1
            if field == "samples":
                c["duration"] = c["samples"] / SR
            d = self.root / f"bad-{field}"
            try:
                ami_store.write_store(d, clips, {})
            except ValueError:
                continue                                     # refused at write (frames no longer equal)
            store = ami_store.PairStore.load(d)
            read = store.read_teacher if field.startswith("ihm_") else store.read
            with self.assertRaises(RuntimeError, msg=field):
                read(c["id"])

    def test_a_damaged_or_incomplete_store_is_refused(self):
        path = self.dir / "clips.jsonl"
        data = path.read_bytes()
        path.write_bytes(data.replace(b'"meeting_id": "ES2002a"', b'"meeting_id": "ES2002b"', 1))
        with self.assertRaises(RuntimeError):
            ami_store.PairStore.load(self.dir)
        path.write_bytes(data)
        meta_path = self.dir / "clips.meta.json"
        meta = json.loads(meta_path.read_text())
        meta_path.write_text(json.dumps(meta | {"complete": False}))
        with self.assertRaises(RuntimeError):
            ami_store.PairStore.load(self.dir)

    def test_unequal_frames_are_refused(self):
        clips = [dict(c) for c in self.clips]
        sid, hid, spcm, _ = self.specs[0]
        bad = make_clips(self.root / "audio-bad", [(sid, hid, spcm, pcm_of(5, len(spcm) + 1600))])[0]
        clips = [bad if c["id"] == sid else c for c in clips]
        with self.assertRaises(ValueError):
            ami_store.write_store(self.root / "bad", clips, {})
        d = self.root / "forged"
        ami_store.write_store(d, self.clips, {})
        lines = (d / "clips.jsonl").read_text().splitlines()
        lines = [json.dumps(bad, ensure_ascii=False) if json.loads(x)["id"] == sid else x for x in lines]
        data = ("\n".join(lines) + "\n").encode()
        (d / "clips.jsonl").write_bytes(data)
        meta = json.loads((d / "clips.meta.json").read_text())
        (d / "clips.meta.json").write_text(json.dumps(meta | {"clips_sha256": hashlib.sha256(data).hexdigest()}))
        with self.assertRaises(RuntimeError):
            ami_store.PairStore.load(d)


class Labels(unittest.TestCase):
    """The label of a pair is its IHM id's stored label: T*'s for the main arm, B0's under T*'s disposition mask for
    the control (a real label store: C1 shards and C2 records of training_fixtures.LabeledWorld, whose T* rejects
    every id of one batch). The fixture has one AMI id, so the readers are also checked with prefix "" (every id)."""

    @classmethod
    def setUpClass(cls):
        from training_fixtures import LabeledWorld
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.ctx = relabel_root(cls.root)
        cls.ctx.__enter__()
        cls.w = LabeledWorld(cls.root)
        cls.through = 14

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)
        cls.tmp.cleanup()

    def stores(self):
        from training import labels as tlabels
        return {s: tlabels.LabelStore.open(self.w.labels_dir, self.w.pool, label_set=s, through_batch=self.through)
                for s in ("T*", "B0")}

    def test_dispositions_are_the_label_stores_mask(self):
        t = self.stores()["T*"]
        for prefix in ("ami:", ""):
            disp = ami_sdm.label_dispositions(self.w.labels_dir, self.w.pool, self.through, prefix=prefix)
            want = {i: r["disposition"] for i, r in self.w.records.items() if i.startswith(prefix)}
            self.assertTrue(want)
            self.assertEqual(disp, want)
            self.assertEqual({i for i, d in disp.items() if d == "kept"}, {i for i in want if t.text(i) is not None})
        self.assertTrue(any(d != "kept" for d in disp.values()))
        self.assertTrue(any(i.startswith("ami:") and d == "kept" for i, d in disp.items()))

    def test_dispositions_refuse_a_changed_label_shard(self):
        import shutil
        copy = self.root / "labels-copy"
        shutil.copytree(self.w.labels_dir, copy)
        shard = sorted((copy / "shards").glob("*.labels.parquet"))[0]
        shard.write_bytes(shard.read_bytes() + b"x")
        with self.assertRaises(RuntimeError):
            ami_sdm.label_dispositions(copy, self.w.pool, self.through)

    def test_pair_labels_read_the_ihm_ids_label_in_each_arm(self):
        kept = sorted(i for i, r in self.w.records.items() if r["disposition"] == "kept")
        rejected = sorted(i for i, r in self.w.records.items() if r["disposition"] != "kept")
        ami = [i for i in kept if i.startswith("ami:")]
        self.assertTrue(ami and rejected)
        chosen = ami[:1] + [i for i in kept if not i.startswith("ami:")][:1] + rejected[:1]
        specs = [(f"AMI_ES2002a_sdm_FEE005_{k:07d}_{k + 100:07d}", hid, pcm_of(k, 16000), pcm_of(50 + k, 16000))
                 for k, hid in enumerate(chosen)]
        # in memory: the store's own writer accepts only pool AMI ids, and the fixture has one
        store = ami_store.PairStore(make_clips(self.root / "pair-audio", specs), identity="fixture")
        arms = self.stores()
        main, control = (ami_store.PairLabels(store, arms[s]) for s in ("T*", "B0"))
        for sid, hid, _, _ in specs:
            row = ami_store.PREFIX + sid
            self.assertEqual(main.ihm_id(row), hid)
            self.assertEqual(main.text(row), arms["T*"].text(hid))
            self.assertEqual(control.text(row), arms["B0"].text(hid))
            if hid in kept:
                self.assertEqual(main.text(row), self.w.records[hid]["text"])
                self.assertEqual(control.text(row), self.w.records[hid]["b0_raw"])
                self.assertNotEqual(main.text(row), control.text(row))
            else:
                self.assertIsNone(main.text(row))
                self.assertIsNone(control.text(row))           # the same mask in both arms
        self.assertEqual(main.prefix, ami_store.PREFIX)
        self.assertEqual(main.identity["clip_pool"], "fixture")
        with self.assertRaises(KeyError):
            main.text(ami_store.PREFIX + "AMI_XX0000a_sdm_X_0000000_0000001")
        with self.assertRaises(KeyError):
            main.text(specs[0][0])                              # a row id carries the prefix

    def test_pair_labels_through_the_runs_reader(self):
        """Through label_sources.StoreLabels with a read-time reject list (stage 3's reader): a listed IHM id has no
        label, so its pair has none either."""
        from training import label_sources
        ami = sorted(i for i, r in self.w.records.items() if i.startswith("ami:") and r["disposition"] == "kept")
        specs = [("AMI_ES2002a_sdm_FEE005_0000001_0000101", ami[0], pcm_of(1, 16000), pcm_of(2, 16000))]
        d = self.root / "pair-store"
        ami_store.write_store(d, make_clips(self.root / "pair-audio-2", specs), {})
        store = ami_store.PairStore.load(d)
        t = self.stores()["T*"]
        row = ami_store.PREFIX + specs[0][0]
        self.assertEqual(ami_store.PairLabels(store, label_sources.StoreLabels(t)).text(row), t.text(ami[0]))
        listed = label_sources.RejectList([ami[0]], {"name": "fixture"})
        self.assertIsNone(ami_store.PairLabels(store, label_sources.StoreLabels(t, reject=listed)).text(row))

    def test_c1_ami_occurrences_and_pcm(self):
        audio = {r["id"]: r for k in shards.committed(self.w.pool)
                 for r in shards.read_audio(shards.paths(self.w.pool, k)["audio"], ["id", "samples", "pcm_sha256"])}
        for prefix in ("ami:", ""):
            c1 = ami_sdm.c1_ami(self.w.pool, self.through, prefix=prefix)
            occ = collections.Counter(r["id"] for r in self.w.journal if r["id"].startswith(prefix))
            self.assertTrue(occ)
            self.assertEqual({i: v["occurrences"] for i, v in c1.items()}, dict(occ))
            for i, v in c1.items():
                self.assertEqual((v["samples"], v["pcm_sha256"]), (audio[i]["samples"], audio[i]["pcm_sha256"]))
        self.assertGreater(max(collections.Counter(r["id"] for r in self.w.journal).values()), 1)   # repeats counted
        short = ami_sdm.c1_ami(self.w.pool, 4, prefix="")
        self.assertEqual(sum(v["occurrences"] for v in short.values()), sum(r["batch"] < 4 for r in self.w.journal))


class Draws(unittest.TestCase):
    def test_average_occurrences_rounded(self):
        self.assertEqual(ami_sdm.draws_per_clip([3, 4])["draws"], 4)          # 3.5 rounds up
        self.assertEqual(ami_sdm.draws_per_clip([7, 8, 8])["draws"], 8)
        self.assertEqual(ami_sdm.draws_per_clip([1, 1, 1, 2])["draws"], 1)    # 1.25
        got = ami_sdm.draws_per_clip([2, 5, 5])
        self.assertEqual((got["ids"], got["occurrences"], got["mean"]), (3, 12, 4.0))
        with self.assertRaises(ValueError):
            ami_sdm.draws_per_clip([])


class DevManifest(unittest.TestCase):
    def test_sdm_validation_rows_of_the_dev_meetings(self):
        rows = [sdm("ES2011a", "FEE041", 1.0, 3.0, text="HERE WE GO"), sdm("ES2011b", "FEE041", 1.0, 2.0),
                sdm("IS1008a", "FEE041", 1.0, 2.0), sdm("ES2011a", "FEE041", 5.0, 6.0, text="  "),
                sdm("ES2011a", "FEE041", 7.0, 8.0)]
        recs = {r["audio_id"]: {"id": r["audio_id"], "status": "ok", "audio_filepath": f"/abs/{r['audio_id']}.flac",
                                "samples": 32000} for r in rows}
        recs[rows[4]["audio_id"]] = {"id": rows[4]["audio_id"], "status": "undecodable", "reason": "silent_audio"}
        out, counts = ami_sdm.dev_rows(rows, recs, {"ES2011a", "ES2011b"})
        import data  # finetune/parakeet-ternary: the dev manifests' format
        for r in out:
            self.assertEqual(list(r), list(data.MANIFEST_KEYS))
            data._check_row(r)
        self.assertEqual([r["id"] for r in out], sorted(ami_store.PREFIX + rows[i]["audio_id"] for i in (0, 1)))
        first = next(r for r in out if r["id"].endswith(rows[0]["audio_id"]))
        self.assertEqual((first["text"], first["duration"], first["source"]), ("HERE WE GO", 2.0, ami_store.ALIAS))
        self.assertEqual(counts, {"rows": 5, "not_dev_meeting": 1, "empty_text": 1, "silent_audio": 1, "kept": 2})

    def test_wanted_dev_rows(self):
        rows = [sdm("ES2011a", "FEE041", 1.0, 3.0), sdm("IS1008a", "FEE041", 1.0, 2.0),
                sdm("ES2011a", "FEE041", 5.0, 6.0, text="")]
        self.assertEqual(ami_sdm.dev_wanted(rows, {"ES2011a"}), [rows[0]["audio_id"]])


if __name__ == "__main__":
    unittest.main()
