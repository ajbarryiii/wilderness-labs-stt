"""Source contracts (plans/build.md "Code and tests" 3 and 4, "Evaluation-pool sources"):

3. Licence filter: only CC0 or CC BY clips from the FSD50K eval split enter the pool; no NC,
   excluded or unverified source can enter any pool.
4. Download verification: a source with a missing or mismatching checksum is refused, and the
   pool refuses a source whose verification record is missing. BUT ReverbDB (no published
   checksum): HTTPS only, pinned exact size, SHA-256 trusted on first use and enforced after
   (review round 2, finding 4: this replaces the earlier "first-seen" contract).

Review round 2: extraction budgets (zip and tar bombs), FSD50K reservation from the complete
metadata (missing expected audio fails), and caches bound to their verified inputs.

Plus the extraction safety rules (path traversal, links, split zip archives) on synthetic archives.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import struct
import tarfile
import unittest
import zipfile
import zlib
from pathlib import Path
from unittest import mock

import noisy_fixtures as fx
import numpy as np

import noisy_extract
import noisy_paths
import noisy_sources

CC0 = "http://creativecommons.org/publicdomain/zero/1.0/"
BY3 = "http://creativecommons.org/licenses/by/3.0/"
BY4 = "https://creativecommons.org/licenses/by/4.0/"
NC = "http://creativecommons.org/licenses/by-nc/3.0/"
SAMPLING = "http://creativecommons.org/licenses/sampling+/1.0/"


def fsd50k_layout(root: Path) -> None:
    """A miniature extracted FSD50K: ground truth, clip info and eval audio."""
    gt, meta, audio = root / "FSD50K.ground_truth", root / "FSD50K.metadata", root / "FSD50K.eval_audio"
    for d in (gt, meta, audio):
        d.mkdir(parents=True)
    eval_rows = {  # fname: (labels, licence, uploader)
        "101": ("Gunshot_and_gunfire,Explosion", CC0, "alice"),
        "102": ("Explosion", BY3, "bob"),
        "103": ("Gunshot_and_gunfire,Explosion", NC, "carol"),          # NC: excluded
        "104": ("Engine,Vehicle,Car", BY4, "dave"),
        "105": ("Wind", SAMPLING, "erin"),                               # Sampling+: excluded
        "106": ("Wind", CC0, "frank"),
        "107": ("Crowd,Human_group_actions", BY3, "gina"),
        "108": ("Shout,Human_voice", CC0, "hank"),
        "109": ("Car,Vehicle,Male_speech_and_man_speaking,Speech,Human_voice", CC0, "ivan"),  # voice in engine
        "110": ("Dog,Animal", CC0, "judy"),                              # no category; still reserved
        "111": ("Fireworks,Explosion", CC0, "ken"),                      # Explosion (smeared label): gun
        "112": ("Truck,Vehicle", "", "lena"),                            # missing licence: excluded
    }
    with open(gt / "eval.csv", "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["fname", "labels", "mids"])
        for fname, (labels, _, _) in eval_rows.items():
            w.writerow([fname, labels, ""])
    with open(gt / "dev.csv", "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["fname", "labels", "mids", "split"])
        w.writerow(["201", "Gunshot_and_gunfire,Explosion", "", "train"])
    info = {f: {"title": f"clip {f}", "description": "", "tags": [], "license": lic, "uploader": up}
            for f, (_, lic, up) in eval_rows.items()}
    info["201"] = {"title": "dev clip", "license": CC0, "uploader": "zed"}  # a dev clip's info must not matter
    (meta / "eval_clips_info_FSD50K.json").write_text(json.dumps(info))
    for k, fname in enumerate([*eval_rows, "201"]):  # 201 is a dev clip planted in the eval audio dir
        fx.write_audio(audio / f"{fname}.wav", fx.noise_clip(1.5, 200 + k), sr=44100)


class LicenceFilterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = fx.fresh_dir("fsd50k-licence")
        fsd50k_layout(cls.root)
        cls.index = noisy_sources.index_fsd50k(fx.members_of(cls.root))

    def test_only_cc0_and_cc_by_eval_clips_enter_categories(self):
        used = {c["freesound_id"]: c for cat in self.index["clips"].values() for c in cat}
        self.assertEqual(sorted(used), [101, 102, 104, 106, 107, 108, 111])
        for clip in used.values():
            self.assertTrue(noisy_sources.licence_allowed(clip["licence_url"]), clip)
            self.assertNotIn("nc", clip["licence_url"].lower())
            self.assertEqual(clip["split"], "eval")
        cats = {cat: sorted(c["freesound_id"] for c in clips) for cat, clips in self.index["clips"].items()}
        self.assertEqual(cats["gun"], [101, 102, 111])
        self.assertEqual(cats["engine"], [104])
        self.assertEqual(cats["wind"], [106])
        self.assertEqual(cats["crowd"], [107, 108])

    def test_dev_split_clip_never_used_and_exclusions_recorded(self):
        used = {c["freesound_id"] for cat in self.index["clips"].values() for c in cat}
        self.assertNotIn(201, used)
        reasons = {e["fname"]: e["reason"] for e in self.index["excluded"]}
        self.assertEqual(reasons["103"], "licence")
        self.assertEqual(reasons["105"], "licence")
        self.assertEqual(reasons["112"], "licence")
        self.assertEqual(reasons["109"], "voice_label")
        self.assertEqual(reasons["201"], "not_eval_split")

    def test_whole_eval_split_reserved(self):
        reserved = self.index["reserved"]
        self.assertEqual(sorted(reserved["sound_ids"]), list(range(101, 113)))  # every eval clip, used or not
        self.assertIn("carol", reserved["uploaders"])
        self.assertNotIn("zed", reserved["uploaders"])
        self.assertEqual(len(reserved["pcm_sha256"]), 12)

    def test_reservation_comes_from_metadata_and_missing_audio_fails(self):
        root = fx.fresh_dir("fsd50k-missing")
        fsd50k_layout(root)
        (root / "FSD50K.eval_audio" / "110.wav").unlink()  # an eval clip in no category
        with self.assertRaisesRegex(FileNotFoundError, "110"):
            noisy_sources.index_fsd50k(fx.members_of(root))

    def test_incomplete_clip_info_fails(self):
        root = fx.fresh_dir("fsd50k-noinfo")
        fsd50k_layout(root)
        path = root / "FSD50K.metadata" / "eval_clips_info_FSD50K.json"
        info = json.loads(path.read_text())
        del info["110"]
        path.write_text(json.dumps(info))
        with self.assertRaisesRegex(noisy_sources.PoolError, "110"):
            noisy_sources.index_fsd50k(fx.members_of(root))

    def test_licence_predicate(self):
        for url in (CC0, BY3, BY4, "https://creativecommons.org/licenses/by/2.0"):
            self.assertTrue(noisy_sources.licence_allowed(url), url)
        for url in (NC, SAMPLING, "http://creativecommons.org/licenses/by-nc-sa/3.0/", "", None,
                    "http://creativecommons.org/licenses/by-nd/4.0/"):
            self.assertFalse(noisy_sources.licence_allowed(url), url)


class PoolAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.root = fx.fresh_dir("pool-admission")
        x = fx.noise_clip(1.0, 1)
        self.path = fx.write_audio(self.root / "a.wav", x)

    def pool(self, licence: str, source: str = "fsd50k", verification: dict | None = None):
        clip = noisy_sources.clip_record(self.path, key=f"{source}:1", source=source, category="engine",
                                         original="a.wav", licence=licence)
        sources = {source: {"dataset": source, "licence": licence,
                            "verification": {"ok": True} if verification is None else verification}}
        return noisy_sources.make_pool(clips={"engine": [clip]}, event_clips=[], events=[], rirs=[],
                                       sources=sources, reserved={})

    def test_accepts_allowed_licences(self):
        for licence in ("CC0-1.0", "CC-BY-4.0", "CC-BY-SA-3.0", "public domain"):
            self.pool(licence)

    def test_refuses_nc_and_unknown_licences(self):
        for licence in ("CC-BY-NC-3.0", "CC-BY-NC-4.0", "unverified", ""):
            with self.assertRaises(noisy_sources.PoolError, msg=licence):
                self.pool(licence)

    def test_refuses_excluded_sources(self):
        for source in ("esc50", "mit-ir-survey", "ir64-car", "urbansound8k"):
            with self.assertRaises(noisy_sources.PoolError, msg=source):
                self.pool("CC0-1.0", source=source)

    def test_refuses_unregistered_source_with_self_attested_verification(self):
        # Review round 4, finding 3: only registered sources and plug-ins enter, whatever they claim.
        for source in ("my-noise", "freesound", "fsd50k-copy"):
            with self.assertRaisesRegex(noisy_sources.PoolError, "registered", msg=source):
                self.pool("CC0-1.0", source=source, verification={"ok": True, "files": []})

    def test_refuses_unverified_source(self):
        with self.assertRaises(noisy_sources.PoolError):
            self.pool("CC0-1.0", verification={})
        with self.assertRaises(noisy_sources.PoolError):
            self.pool("CC0-1.0", verification={"ok": False})


class PoolInputCheckTest(unittest.TestCase):
    """Review round 4, finding 3: check_pool_inputs verifies every pooled source through its real
    mechanism (registry sources: their verification records; the rotor plug-in: sounds.json and
    the downloads it lists) and refuses any other source, whatever verification it claims."""

    def setUp(self):
        self.root = fx.fresh_dir("pool-inputs")
        self.pool_path = fx.make_pool(self.root, rotor=True)
        self.raw, self.verify = fx.source_dirs(self.pool_path)
        self.pool = noisy_sources.load_pool(self.pool_path, self.root / "rotor.json")

    def check(self, pool=None):
        return noisy_sources.check_pool_inputs(pool or self.pool, self.raw, self.verify,
                                               registry=fx.registry(self.pool_path))

    def test_forged_empty_record_and_entry_refused(self):
        # Review round 5, finding 2: an empty record matching an empty pool entry is no verification.
        pool = json.loads(json.dumps(self.pool))
        pool["sources"]["fsd50k"]["verification"]["files"] = []
        (self.verify / "fsd50k.json").write_text(json.dumps({"ok": True, "source": "fsd50k", "files": []}))
        with self.assertRaises(noisy_sources.VerificationError):
            self.check(pool)

    def test_production_registry_refuses_test_records(self):
        with self.assertRaises(noisy_sources.VerificationError):
            noisy_sources.check_pool_inputs(self.pool, self.raw, self.verify)  # default: production SOURCES

    def test_registered_sources_and_plug_in_are_checked(self):
        result = self.check()
        self.assertEqual(result["sources_checked"],
                         sorted(["but-reverbdb", "demand", "freesound-rotor", "fsd50k", "gunshots-ipod", "musan"]))
        self.assertNotIn("freesound:9003", {c["key"] for c in self.pool["clips"]["rotor"]})  # NC: never entered

    def test_unregistered_source_refused_even_if_self_attested(self):
        pool = json.loads(json.dumps(self.pool))
        pool["sources"]["my-noise"] = {"dataset": "my-noise", "licence": "CC0-1.0", "verification": {"ok": True}}
        with self.assertRaisesRegex(noisy_sources.VerificationError, "registered"):
            self.check(pool)

    def test_registered_source_without_its_record_refused(self):
        (self.verify / "musan.json").unlink()
        with self.assertRaises(noisy_sources.VerificationError):
            self.check()

    def test_plug_in_metadata_changed_refused(self):
        meta = self.raw / "freesound-rotor" / "sounds.json"
        meta.write_text(meta.read_text().replace("u-rotor2", "someone-else"))
        with self.assertRaises(noisy_sources.InputChanged):
            self.check()

    def test_plug_in_download_changed_refused(self):
        fx.write_audio(self.raw / "freesound-rotor" / "9002.wav", fx.noise_clip(0.9, 99))
        with self.assertRaises((noisy_sources.InputChanged, noisy_sources.VerificationError)):
            self.check()

    def test_plug_in_without_verification_files_refused(self):
        pool = json.loads(json.dumps(self.pool))
        pool["sources"]["freesound-rotor"]["verification"] = {"ok": True}
        with self.assertRaises(noisy_sources.VerificationError):
            self.check(pool)


class VerificationTest(unittest.TestCase):
    def setUp(self):
        self.root = fx.fresh_dir("verify")
        self.raw, self.out = self.root / "raw", self.root / "verify"
        (self.raw / "toy").mkdir(parents=True)
        (self.raw / "toy" / "a.bin").write_bytes(b"hello")
        (self.raw / "toy" / "b.bin").write_bytes(b"world")
        for name in ("a.bin", "b.bin"):  # download.sh's markers: the download finished
            (self.raw / "toy" / f"{name}.done").touch()
        self.registry = {"toy": {"dataset": "toy", "licence": "CC0-1.0", "checksum": "md5",
                                 "files": {"a.bin": "5d41402abc4b2a76b9719d911017c592",
                                           "b.bin": "7d793037a0760186574b0282f2f435e7"}},
                         "tofu": {"dataset": "fs", "licence": "CC-BY-4.0", "checksum": "tofu-sha256",
                                  "url": "https://example.org/c.bin", "sizes": {"c.bin": 3},
                                  "last_modified": "Wed, 05 Jun 2019 12:31:28 GMT", "files": {"c.bin": None}}}

    def verify(self, name):
        return noisy_sources.verify_source(name, raw=self.raw, verify_dir=self.out, registry=self.registry)

    def test_matching_checksums_verify_and_are_recorded(self):
        record = self.verify("toy")
        self.assertTrue(record["ok"])
        self.assertEqual(noisy_sources.require_verified("toy", self.out, raw=self.raw, registry=self.registry)["ok"], True)
        self.assertEqual({f["name"] for f in record["files"]}, {"a.bin", "b.bin"})
        for f in record["files"]:
            self.assertEqual(len(f["sha256"]), 64)

    def test_mismatch_refused(self):
        (self.raw / "toy" / "b.bin").write_bytes(b"w0rld")
        with self.assertRaises(noisy_sources.VerificationError):
            self.verify("toy")
        with self.assertRaises(noisy_sources.VerificationError):
            noisy_sources.require_verified("toy", self.out, raw=self.raw, registry=self.registry)

    def test_missing_file_refused(self):
        (self.raw / "toy" / "a.bin").unlink()
        with self.assertRaises(noisy_sources.VerificationError):
            self.verify("toy")

    def test_unfinished_download_not_read(self):
        (self.raw / "toy" / "b.bin.done").unlink()
        with mock.patch.object(noisy_sources, "_hashes", side_effect=AssertionError("read")):
            with self.assertRaisesRegex(noisy_sources.VerificationError, "not finished"):
                self.verify("toy")

    def test_missing_publisher_checksum_refused(self):
        self.registry["toy"]["files"]["a.bin"] = None
        with self.assertRaises(noisy_sources.VerificationError):
            self.verify("toy")

    def test_missing_record_refused(self):
        with self.assertRaises(noisy_sources.VerificationError):
            noisy_sources.require_verified("toy", self.out, raw=self.raw, registry=self.registry)

    def test_file_changed_after_verification_refused(self):
        self.verify("toy")
        (self.raw / "toy" / "a.bin").write_bytes(b"hellO")
        with self.assertRaises(noisy_sources.VerificationError):
            noisy_sources.require_verified("toy", self.out, raw=self.raw, registry=self.registry)

    def tofu_file(self, payload: bytes = b"abc") -> None:
        (self.raw / "tofu").mkdir(exist_ok=True)
        (self.raw / "tofu" / "c.bin").write_bytes(payload)
        (self.raw / "tofu" / "c.bin.done").touch()

    def test_tofu_sha256_recorded_over_https_then_enforced(self):
        self.tofu_file()
        record = self.verify("tofu")
        self.assertTrue(record["ok"])
        f = record["files"][0]
        self.assertEqual(f["checksum_kind"], "trust on first use: sha256 over https, pinned size")
        self.assertEqual(f["transport"], "https")
        self.assertEqual(f["last_modified"], "Wed, 05 Jun 2019 12:31:28 GMT")
        self.assertEqual(f["sha256"], hashlib.sha256(b"abc").hexdigest())
        self.verify("tofu")  # same contents: still ok
        self.tofu_file(b"abd")  # same size, other contents
        with self.assertRaises(noisy_sources.VerificationError):
            self.verify("tofu")

    def test_tofu_size_mismatch_refused(self):
        self.tofu_file(b"abcd")
        with self.assertRaisesRegex(noisy_sources.VerificationError, "size"):
            self.verify("tofu")

    def test_tofu_requires_https_source(self):
        self.tofu_file()
        self.registry["tofu"]["url"] = "http://example.org/c.bin"
        with self.assertRaisesRegex(noisy_sources.VerificationError, "https"):
            self.verify("tofu")

    def test_tofu_refuses_a_non_https_record(self):
        self.tofu_file()
        record = self.verify("tofu")
        record["files"][0]["transport"] = "http"
        (self.out / "tofu.json").write_text(json.dumps(record))
        with self.assertRaisesRegex(noisy_sources.VerificationError, "https"):
            self.verify("tofu")
        with self.assertRaisesRegex(noisy_sources.VerificationError, "https"):
            noisy_sources.require_verified("tofu", self.out, raw=self.raw, registry=self.registry)

    def test_tofu_pinned_digest_enforced_when_known(self):
        self.tofu_file()
        self.registry["tofu"]["sha256"] = {"c.bin": "0" * 64}
        with self.assertRaises(noisy_sources.VerificationError):
            self.verify("tofu")

    def test_real_but_entry(self):
        spec = noisy_sources.SOURCES["but-reverbdb"]
        self.assertEqual(spec["checksum"], "tofu-sha256")
        self.assertTrue(spec["url"].startswith("https://"))
        self.assertEqual(spec["sizes"], {"BUT_ReverbDB_rel_19_06_RIR-Only.tgz": 9308593693})
        self.assertEqual(spec["last_modified"], "Wed, 05 Jun 2019 12:31:28 GMT")

    def forge(self, name: str, mutate) -> None:
        record = self.verify(name)
        mutate(record)
        (self.out / f"{name}.json").write_text(json.dumps(record))

    def test_incomplete_or_forged_records_refused(self):
        # Review round 5, finding 2: the record must cover exactly the registered files with the
        # registered checksums.
        cases = {
            "no_files": lambda r: r.update(files=[]),
            "missing_file": lambda r: r.update(files=r["files"][:1]),
            "extra_file": lambda r: r["files"].append({**r["files"][0], "name": "c.bin"}),
            "other_md5": lambda r: r["files"][0].update(md5="0" * 32, expected_md5="0" * 32),
            "other_source": lambda r: r.update(source="musan"),
            "no_sha256": lambda r: r["files"][0].pop("sha256"),
            "not_ok": lambda r: r.update(ok="yes"),
        }
        for name, mutate in cases.items():
            self.forge("toy", mutate)
            with self.assertRaises(noisy_sources.VerificationError, msg=name):
                noisy_sources.require_verified("toy", self.out, raw=self.raw, registry=self.registry)
        (self.out / "toy.json").write_text(json.dumps({"ok": True, "files": []}))
        with self.assertRaises(noisy_sources.VerificationError):
            noisy_sources.require_verified("toy", self.out, raw=self.raw, registry=self.registry)

    def test_test_records_do_not_pass_under_production_names(self):
        record = self.verify("toy")
        (self.out / "musan.json").write_text(json.dumps({**record, "source": "musan"}))
        (self.raw / "musan").mkdir()
        for f in record["files"]:
            (self.raw / "musan" / f["name"]).write_bytes((self.raw / "toy" / f["name"]).read_bytes())
        with self.assertRaisesRegex(noisy_sources.VerificationError, "registry"):
            noisy_sources.require_verified("musan", self.out, raw=self.raw)  # production SOURCES

    def test_real_registry_has_publisher_checksums(self):
        for name, spec in noisy_sources.SOURCES.items():
            for fname, checksum in spec["files"].items():
                if spec["checksum"] == "md5":
                    self.assertRegex(checksum or "", r"^[0-9a-f]{32}$", f"{name}/{fname}")
        self.assertEqual(noisy_sources.SOURCES["musan"]["files"]["musan.tar.gz"], "0c472d4fc0c5141eca47ad1ffeb2a7df")
        for excluded in ("esc50", "mit-ir-survey", "ir64-car"):
            self.assertNotIn(excluded, noisy_sources.SOURCES)


def _split_zip(segments: list[Path], members: dict[str, bytes], split_at: int) -> None:
    """A two-disk split zip as Info-ZIP writes it (.z01 then .zip): per-disk offsets, central
    directory on the last disk, deflated members; the split signature starts disk 0."""
    data, local = bytearray(b"PK\x07\x08"), []
    for name, payload in members.items():
        comp = zlib.compressobj(9, zlib.DEFLATED, -15)
        body = comp.compress(payload) + comp.flush()
        crc = zlib.crc32(payload)
        local.append((name, len(data), crc, len(body), len(payload)))
        data += struct.pack("<IHHHHHIIIHH", 0x04034B50, 20, 0, 8, 0, 0, crc, len(body), len(payload),
                            len(name), 0) + name.encode() + body
    disk0, disk1 = bytes(data[:split_at]), bytearray(data[split_at:])
    cd_start = len(disk1)
    for name, off, crc, csize, usize in local:
        disk, rel = (0, off) if off < split_at else (1, off - split_at)
        disk1 += struct.pack("<IHHHHHHIIIHHHHHII", 0x02014B50, 20, 20, 0, 8, 0, 0, crc, csize, usize,
                             len(name), 0, 0, disk, 0, 0o100644 << 16, rel) + name.encode()
    cd_size = len(disk1) - cd_start
    disk1 += struct.pack("<IHHHHIIH", 0x06054B50, 1, 1, len(local), len(local), cd_size, cd_start, 0)
    segments[0].write_bytes(disk0)
    segments[1].write_bytes(bytes(disk1))


class ExtractionTest(unittest.TestCase):
    def setUp(self):
        self.root = fx.fresh_dir("extract")

    def test_member_name_checks(self):
        for bad in ("/etc/passwd", "../x", "a/../../x", "a\\b", "", "C:/x", "a/\x00b"):
            with self.assertRaises(noisy_extract.UnsafeArchive, msg=repr(bad)):
                noisy_extract.safe_member_name(bad)
        self.assertEqual(str(noisy_extract.safe_member_name("FSD50K.eval_audio/1.wav")), "FSD50K.eval_audio/1.wav")

    def test_zip_traversal_refused_and_nothing_written(self):
        archive = self.root / "evil.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("ok.txt", "fine")
            z.writestr("../escape.txt", "bad")
        with self.assertRaises(noisy_extract.UnsafeArchive):
            noisy_extract.extract_zip([archive], self.root / "out", lambda n: True)
        self.assertFalse((self.root / "escape.txt").exists())
        self.assertFalse((self.root / "out").exists())

    def test_zip_symlink_refused(self):
        archive = self.root / "link.zip"
        with zipfile.ZipFile(archive, "w") as z:
            info = zipfile.ZipInfo("link")
            info.external_attr = (0o120777 << 16)
            z.writestr(info, "/etc/passwd")
        with self.assertRaises(noisy_extract.UnsafeArchive):
            noisy_extract.extract_zip([archive], self.root / "out", lambda n: True)

    def test_tar_links_and_traversal_refused(self):
        for kind in ("symlink", "traversal", "hardlink"):
            archive = self.root / f"{kind}.tar.gz"
            with tarfile.open(archive, "w:gz") as t:
                info = tarfile.TarInfo("musan/speech/us-gov/a.wav" if kind != "traversal" else "../a.wav")
                if kind == "symlink":
                    info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
                    t.addfile(info)
                elif kind == "hardlink":
                    info.type, info.linkname = tarfile.LNKTYPE, "/etc/passwd"
                    t.addfile(info)
                else:
                    info.size = 3
                    t.addfile(info, io.BytesIO(b"abc"))
            with self.assertRaises(noisy_extract.UnsafeArchive, msg=kind):
                noisy_extract.extract_tar(archive, self.root / f"out-{kind}", lambda n: True)

    def test_tar_selected_members_extracted(self):
        archive = self.root / "ok.tar.gz"
        with tarfile.open(archive, "w:gz") as t:
            for name, payload in (("musan/speech/us-gov/a.wav", b"aaa"), ("musan/noise/b.wav", b"bb")):
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                t.addfile(info, io.BytesIO(payload))
        out = self.root / "out-tar"
        members = noisy_extract.extract_tar(archive, out, lambda n: n.startswith("musan/speech/us-gov/"))
        self.assertEqual([m["name"] for m in members], ["musan/speech/us-gov/a.wav"])
        self.assertEqual((out / "musan/speech/us-gov/a.wav").read_bytes(), b"aaa")
        self.assertFalse((out / "musan/noise/b.wav").exists())

    def test_tar_root_directory_entry_accepted(self):
        # GNU tar run as `tar czf x.tgz .` starts with a "./" directory entry (the BUT ReverbDB
        # release does). It names the archive root, carries no data and is skipped; "./"-prefixed
        # files extract under their normalized names, and "./../" stays refused.
        archive = self.root / "root.tar.gz"
        with tarfile.open(archive, "w:gz") as t:
            for name in ("./", "."):
                info = tarfile.TarInfo(name)
                info.type = tarfile.DIRTYPE
                t.addfile(info)
            info = tarfile.TarInfo("./RIR/a.wav")
            info.size = 3
            t.addfile(info, io.BytesIO(b"abc"))
        out = self.root / "out-root"
        members = noisy_extract.extract_tar(archive, out, lambda n: True)
        self.assertEqual([m["name"] for m in members], ["RIR/a.wav"])
        self.assertEqual((out / "RIR/a.wav").read_bytes(), b"abc")
        for name, kind in (("./../a.wav", tarfile.REGTYPE), ("./", tarfile.REGTYPE)):
            archive = self.root / "bad-root.tar.gz"
            with tarfile.open(archive, "w:gz") as t:
                info = tarfile.TarInfo(name)
                info.type, info.size = kind, 3
                t.addfile(info, io.BytesIO(b"abc"))
            with self.assertRaises(noisy_extract.UnsafeArchive, msg=name):
                noisy_extract.extract_tar(archive, self.root / "out-bad-root", lambda n: True)

    def test_zip_root_directory_entry_accepted(self):
        # The ZIP counterpart of the tar "./" root entry: skipped, still counted against the member
        # budget; a "." file entry (no trailing slash) or a root entry with data stays refused.
        for compression in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):  # deflated "./" is a 2-byte empty stream
            archive = self.root / f"root-{compression}.zip"
            with zipfile.ZipFile(archive, "w", compression) as z:
                z.writestr("./", b"")
                z.writestr("./RIR/a.wav", b"abc")
            out = self.root / f"out-zip-root-{compression}"
            members = noisy_extract.extract_zip([archive], out, lambda n: True)
            self.assertEqual([m["name"] for m in members], ["RIR/a.wav"], msg=compression)
            self.assertEqual((out / "RIR/a.wav").read_bytes(), b"abc")
        with self.assertRaises(noisy_extract.UnsafeArchive):
            noisy_extract.extract_zip([archive], self.root / "out-zip-root-budget", lambda n: True,
                                      noisy_extract.Budget(max_members=1, max_member_bytes=1 << 20,
                                                           max_total_bytes=1 << 20))
        for name, payload in ((".", b"abc"), ("./", b"abc")):
            bad = self.root / "bad-root.zip"
            with zipfile.ZipFile(bad, "w") as z:
                z.writestr(zipfile.ZipInfo(name), payload)
            with self.assertRaises(noisy_extract.UnsafeArchive, msg=name):
                noisy_extract.extract_zip([bad], self.root / "out-zip-bad-root", lambda n: True)

    def test_split_zip_members_spanning_disks(self):
        rng = np.random.default_rng(0)
        members = {"FSD50K.eval_audio/1.wav": rng.bytes(5000), "FSD50K.eval_audio/2.wav": rng.bytes(7000),
                   "FSD50K.eval_audio/3.wav": b"x" * 3000}
        segments = [self.root / "a.z01", self.root / "a.zip"]
        _split_zip(segments, members, split_at=6000)  # member 2 starts on disk 0 and ends on disk 1
        out = self.root / "out-split"
        got = noisy_extract.extract_zip(segments, out, lambda n: n.endswith(".wav"))
        self.assertEqual(sorted(m["name"] for m in got), sorted(members))
        for name, payload in members.items():
            self.assertEqual((out / name).read_bytes(), payload)

    def test_single_zip_selected_members(self):
        archive = self.root / "TBUS_16k.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("TBUS/ch01.wav", b"one")
            z.writestr("TBUS/ch02.wav", b"two")
            z.writestr("TBUS/readme.txt", b"x")
        out = self.root / "out-zip"
        got = noisy_extract.extract_zip([archive], out, lambda n: n.endswith(".wav"))
        self.assertEqual(sorted(m["name"] for m in got), ["TBUS/ch01.wav", "TBUS/ch02.wav"])
        self.assertFalse((out / "TBUS/readme.txt").exists())


def _zip_bytes(members: dict[str, bytes], method=zipfile.ZIP_DEFLATED) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", method) as z:
        for name, payload in members.items():
            z.writestr(name, payload)
    return buf.getvalue()


def _tar_gz(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as t:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            t.addfile(info, io.BytesIO(payload))
    return path


class BudgetTest(unittest.TestCase):
    """Review round 2, finding 5: member count, per-member and total expanded bytes are bounded,
    and a member that inflates past its declared size is refused (bounded decompressor output)."""

    SMALL = noisy_extract.Budget(max_members=10, max_member_bytes=1 << 20, max_total_bytes=1 << 21)

    def setUp(self):
        self.root = fx.fresh_dir("budget")

    def refused(self, fn, *args):
        with self.assertRaises(noisy_extract.UnsafeArchive):
            fn(*args, lambda n: True, self.SMALL)
        self.assertFalse((self.root / "out").exists())
        self.assertEqual(list(self.root.glob(".staging-*")), [])

    def test_zip_bomb_member(self):
        archive = self.root / "bomb.zip"
        archive.write_bytes(_zip_bytes({"a.wav": bytes(10 << 20)}))  # 10 MB of zeros, ~10 KB deflated
        self.assertLess(archive.stat().st_size, 100_000)
        self.refused(noisy_extract.extract_zip, [archive], self.root / "out")

    def test_zip_member_inflating_past_its_declared_size(self):
        data = bytearray(_zip_bytes({"a.wav": bytes(3 << 20)}))
        for sig, field in ((b"PK\x03\x04", 22), (b"PK\x01\x02", 24)):  # local header and central directory
            at = data.find(sig)
            data[at + field:at + field + 4] = struct.pack("<I", 1000)  # claims 1,000 bytes
        archive = self.root / "liar.zip"
        archive.write_bytes(bytes(data))
        with self.assertRaisesRegex(noisy_extract.UnsafeArchive, "declared"):
            noisy_extract.extract_zip([archive], self.root / "out", lambda n: True, self.SMALL)

    def test_zip_too_many_members(self):
        archive = self.root / "many.zip"
        archive.write_bytes(_zip_bytes({f"{k}.wav": b"x" for k in range(50)}))
        self.refused(noisy_extract.extract_zip, [archive], self.root / "out")

    def test_zip_total_expansion(self):
        archive = self.root / "total.zip"
        archive.write_bytes(_zip_bytes({f"{k}.wav": bytes(800_000) for k in range(5)}))
        self.refused(noisy_extract.extract_zip, [archive], self.root / "out")

    def test_tar_bomb_member(self):
        archive = _tar_gz(self.root / "bomb.tar.gz", {"musan/a.wav": bytes(10 << 20)})
        self.assertLess(archive.stat().st_size, 100_000)
        self.refused(noisy_extract.extract_tar, archive, self.root / "out")

    def test_tar_too_many_members_even_unselected(self):
        archive = _tar_gz(self.root / "many.tar.gz", {f"x/{k}.wav": b"x" for k in range(50)})
        with self.assertRaises(noisy_extract.UnsafeArchive):
            noisy_extract.extract_tar(archive, self.root / "out", lambda n: False, self.SMALL)

    def test_tar_total_expansion(self):
        archive = _tar_gz(self.root / "total.tar.gz", {f"x/{k}.wav": bytes(800_000) for k in range(5)})
        self.refused(noisy_extract.extract_tar, archive, self.root / "out")

    @staticmethod
    def raw_zip(name: str, body: bytes, crc: int, usize: int) -> bytes:
        """A one-member deflated zip whose compressed data is exactly `body` (declared as csize)."""
        n = name.encode()
        local = struct.pack("<IHHHHHIIIHH", 0x04034B50, 20, 0, 8, 0, 0, crc, len(body), usize, len(n), 0) + n
        cd = struct.pack("<IHHHHHHIIIHHHHHII", 0x02014B50, 20, 20, 0, 8, 0, 0, crc, len(body), usize, len(n), 0, 0, 0,
                         0, 0o100644 << 16, 0) + n
        eocd = struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, 1, 1, len(cd), len(local) + len(body), 0)
        return local + body + cd + eocd

    def test_trailing_data_after_the_deflate_stream_refused_immediately(self):
        # Review round 4, finding 1: one decoded byte, then 4 MiB of trailing data, under a 16-byte budget.
        comp = zlib.compressobj(9, zlib.DEFLATED, -15)
        stream = comp.compress(b"x") + comp.flush()
        archive = self.root / "trailing.zip"
        archive.write_bytes(self.raw_zip("a.wav", stream + os.urandom(4 << 20), zlib.crc32(b"x"), 1))
        budget = noisy_extract.Budget(max_members=10, max_member_bytes=16, max_total_bytes=16)
        read = []
        original = noisy_extract._Segments.read

        def spy(seg, pos, n):
            read.append(n)
            return original(seg, pos, n)

        with mock.patch.object(noisy_extract._Segments, "read", spy):
            with self.assertRaisesRegex(noisy_extract.UnsafeArchive, "trailing"):
                noisy_extract.extract_zip([archive], self.root / "out", lambda n: True, budget)
        data_reads = [n for n in read if n > 65557]  # beyond the end-record tail and headers: member data
        self.assertEqual(data_reads, [noisy_extract.CHUNK])  # stopped within the first compressed chunk
        self.assertLess(sum(read), 2 << 20)  # of the 4 MiB of trailing data
        self.assertFalse((self.root / "out").exists())

    def test_truncated_deflate_stream_refused(self):
        payload = os.urandom(50_000)
        comp = zlib.compressobj(9, zlib.DEFLATED, -15)
        stream = comp.compress(payload) + comp.flush()
        archive = self.root / "truncated.zip"
        archive.write_bytes(self.raw_zip("a.wav", stream[: len(stream) // 2], zlib.crc32(payload), len(payload)))
        with self.assertRaisesRegex(noisy_extract.UnsafeArchive, "truncated"):
            noisy_extract.extract_zip([archive], self.root / "out", lambda n: True, self.SMALL)

    def test_exact_deflate_stream_accepted(self):
        payload = b"hello world" * 100
        comp = zlib.compressobj(9, zlib.DEFLATED, -15)
        stream = comp.compress(payload) + comp.flush()
        archive = self.root / "exact.zip"
        archive.write_bytes(self.raw_zip("a.wav", stream, zlib.crc32(payload), len(payload)))
        noisy_extract.extract_zip([archive], self.root / "out", lambda n: True, self.SMALL)
        self.assertEqual((self.root / "out" / "a.wav").read_bytes(), payload)

    def test_within_budget_extracts(self):
        archive = self.root / "ok.zip"
        archive.write_bytes(_zip_bytes({"a.wav": bytes(500_000)}))
        got = noisy_extract.extract_zip([archive], self.root / "out", lambda n: True, self.SMALL)
        self.assertEqual(got[0]["size"], 500_000)

    def test_real_sources_have_finite_budgets(self):
        for name in ("fsd50k", "demand", "musan", "but-reverbdb"):
            for budget in noisy_sources.EXTRACTION_BUDGETS[name].values():
                self.assertIsInstance(budget, noisy_extract.Budget)
                self.assertLess(budget.max_total_bytes, 64 << 30)


class MetadataBudgetTest(unittest.TestCase):
    """Review round 3, finding 1: archive metadata is bounded before it is read. A ZIP whose end
    record declares a huge central directory (classic or ZIP64) is refused without a large read;
    a TAR's PAX or GNU extended header larger than the metadata budget is refused before parsing;
    extended headers count as members."""

    SMALL = noisy_extract.Budget(max_members=10, max_member_bytes=1 << 20, max_total_bytes=1 << 21,
                                 max_metadata_bytes=1024)

    def setUp(self):
        self.root = fx.fresh_dir("metadata-budget")

    def biggest_read(self, fn, *args):
        """Run fn and return the largest single read request made of the archive segments."""
        sizes = []
        original = noisy_extract._Segments.read

        def spy(seg, pos, n):
            sizes.append(n)
            return original(seg, pos, n)

        with mock.patch.object(noisy_extract._Segments, "read", spy):
            with self.assertRaises(noisy_extract.UnsafeArchive) as ctx:
                fn(*args)
        return max(sizes), str(ctx.exception)

    def test_huge_declared_central_directory(self):
        data = bytearray(_zip_bytes({"a.wav": b"abc"}))
        at = data.rfind(b"PK\x05\x06")
        data[at + 12:at + 16] = struct.pack("<I", 1 << 30)  # central directory "size": 1 GiB
        archive = self.root / "cd.zip"
        archive.write_bytes(bytes(data))
        biggest, message = self.biggest_read(noisy_extract.extract_zip, [archive], self.root / "out",
                                             lambda n: True, self.SMALL)
        self.assertLessEqual(biggest, 65557)
        self.assertIn("central directory", message)
        self.assertFalse((self.root / "out").exists())

    def zip64(self, cd_size: int | None = None, entries: int | None = None) -> Path:
        """A one-member zip with ZIP64 end records whose central-directory size or entry count lies."""
        data = bytearray(_zip_bytes({"a.wav": b"abc"}))
        at = data.rfind(b"PK\x05\x06")
        (_, _, _, n_disk, n, size, offset, _) = struct.unpack("<IHHHHIIH", data[at:at + 22])
        body = bytes(data[:at])
        z64 = struct.pack("<IQHHIIQQQQ", 0x06064B50, 44, 45, 45, 0, 0, entries or n, entries or n,
                          cd_size or size, offset)
        locator = struct.pack("<IIQI", 0x07064B50, 0, len(body), 1)
        eocd = struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, 0xFFFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0)
        archive = self.root / f"z64-{cd_size}-{entries}.zip"
        archive.write_bytes(body + z64 + locator + eocd)
        return archive

    def test_zip64_end_record_is_parsed(self):
        out = self.root / "ok64"
        got = noisy_extract.extract_zip([self.zip64()], out, lambda n: True, self.SMALL)
        self.assertEqual([m["name"] for m in got], ["a.wav"])

    def test_zip64_huge_central_directory(self):
        biggest, message = self.biggest_read(noisy_extract.extract_zip, [self.zip64(cd_size=1 << 40)],
                                             self.root / "out", lambda n: True, self.SMALL)
        self.assertLessEqual(biggest, 65557)
        self.assertIn("central directory", message)

    def test_zip64_huge_entry_count(self):
        biggest, message = self.biggest_read(noisy_extract.extract_zip, [self.zip64(entries=1 << 40)],
                                             self.root / "out", lambda n: True, self.SMALL)
        self.assertLessEqual(biggest, 65557)
        self.assertIn("members", message)

    def tar(self, name: str, members: dict, fmt=tarfile.PAX_FORMAT, pax: dict | None = None) -> Path:
        path = self.root / name
        with tarfile.open(path, "w:gz", format=fmt) as t:
            for member_name, payload in members.items():
                info = tarfile.TarInfo(member_name)
                info.size = len(payload)
                if pax:
                    info.pax_headers = dict(pax)
                t.addfile(info, io.BytesIO(payload))
        return path

    def test_large_pax_header_refused_before_parsing(self):
        archive = self.tar("pax.tar.gz", {"musan/a.wav": b"abc"}, pax={"comment": "x" * 65536})
        with self.assertRaisesRegex(noisy_extract.UnsafeArchive, "extended header"):
            noisy_extract.extract_tar(archive, self.root / "out", lambda n: True, self.SMALL)
        self.assertFalse((self.root / "out").exists())

    def test_large_gnu_long_name_refused(self):
        archive = self.tar("gnu.tar.gz", {"musan/" + "n" * 5000 + ".wav": b"abc"}, fmt=tarfile.GNU_FORMAT)
        with self.assertRaisesRegex(noisy_extract.UnsafeArchive, "extended header"):
            noisy_extract.extract_tar(archive, self.root / "out", lambda n: True, self.SMALL)

    def test_extended_headers_count_as_members(self):
        archive = self.tar("many-pax.tar.gz", {f"x/{k}.wav": b"x" for k in range(6)}, pax={"comment": "c"})
        with self.assertRaisesRegex(noisy_extract.UnsafeArchive, "members"):  # 6 PAX headers + 6 files > 10
            noisy_extract.extract_tar(archive, self.root / "out", lambda n: False, self.SMALL)

    def test_tar_parser_reads_long_names_and_skips_unselected(self):
        long_name = "musan/speech/us-gov/" + "d" * 120 + "/speech-us-gov-0001.wav"
        for fmt in (tarfile.PAX_FORMAT, tarfile.GNU_FORMAT, tarfile.USTAR_FORMAT):
            name = long_name if fmt != tarfile.USTAR_FORMAT else "musan/speech/us-gov/speech-us-gov-0001.wav"
            archive = self.tar(f"fmt{fmt}.tar.gz", {"musan/noise/n.wav": b"skip", name: b"payload" * 100}, fmt=fmt)
            out = self.root / f"out{fmt}"
            got = noisy_extract.extract_tar(archive, out, lambda n: n.startswith("musan/speech/"),
                                            noisy_extract.DEFAULT_BUDGET)
            self.assertEqual([m["name"] for m in got], [name], fmt)
            self.assertEqual((out / name).read_bytes(), b"payload" * 100)
            self.assertFalse((out / "musan/noise/n.wav").exists())

    def test_tar_header_checksum_checked(self):
        import gzip
        archive = self.tar("ok.tar.gz", {"a/b.wav": b"abc"}, fmt=tarfile.USTAR_FORMAT)
        raw = bytearray(gzip.decompress(archive.read_bytes()))
        raw[0] ^= 0x01  # corrupt the first header's name byte
        bad = self.root / "bad.tar.gz"
        bad.write_bytes(gzip.compress(bytes(raw)))
        with self.assertRaisesRegex(noisy_extract.UnsafeArchive, "checksum"):
            noisy_extract.extract_tar(bad, self.root / "out", lambda n: True, self.SMALL)

    @staticmethod
    def xz_huge_dictionary() -> bytes:
        """An XZ stream header and an LZMA2 block header announcing a ~4 GiB dictionary."""
        flags = b"\x00\x04"
        stream = b"\xfd7zXZ\x00" + flags + struct.pack("<I", zlib.crc32(flags))
        body = bytes([0x02, 0x00, 0x21, 0x01, 0x28, 0x00, 0x00, 0x00])  # size, flags, LZMA2, props, dict 0x28
        return stream + body + struct.pack("<I", zlib.crc32(body)) + bytes(64)

    def test_xz_with_a_huge_dictionary_refused_by_magic(self):
        # Review round 5, finding 4: rejected by magic before any decoder exists, whatever the name.
        import lzma
        data = self.xz_huge_dictionary()
        with self.assertRaises(lzma.LZMAError):  # the fixture really asks for more than 64 MiB
            lzma.LZMADecompressor(memlimit=64 << 20).decompress(data)
        archive = self.root / "evil.tgz"
        archive.write_bytes(data)
        with mock.patch("lzma.LZMAFile", side_effect=AssertionError("XZ decoder created")), \
                mock.patch("lzma.LZMADecompressor", side_effect=AssertionError("XZ decoder created")):
            with self.assertRaisesRegex(noisy_extract.UnsafeArchive, "compression"):
                noisy_extract.extract_tar(archive, self.root / "out", lambda n: True, self.SMALL)
        self.assertFalse((self.root / "out").exists())

    def test_tar_only_as_gzip(self):
        import bz2
        plain = self.root / "plain.tar"
        with tarfile.open(plain, "w") as t:
            info = tarfile.TarInfo("a.wav")
            info.size = 3
            t.addfile(info, io.BytesIO(b"abc"))
        bzipped = self.root / "b.tgz"
        bzipped.write_bytes(bz2.compress(plain.read_bytes()))
        for archive in (plain, bzipped):
            with self.assertRaisesRegex(noisy_extract.UnsafeArchive, "compression", msg=archive.name):
                noisy_extract.extract_tar(archive, self.root / f"out-{archive.name}", lambda n: True, self.SMALL)

    def test_real_budgets_bound_metadata(self):
        for name in ("fsd50k", "demand", "musan", "but-reverbdb"):
            for budget in noisy_sources.EXTRACTION_BUDGETS[name].values():
                self.assertLessEqual(budget.max_metadata_bytes, 256 << 20)


class CacheBindingTest(unittest.TestCase):
    """Review round 2, finding 2: prepared clips and extracted archives are reused only for the
    same verified input and processing parameters."""

    def setUp(self):
        self.root = fx.fresh_dir("cache-binding")

    def test_prepared_clip_follows_its_original(self):
        original = fx.write_audio(self.root / "orig.wav", fx.noise_clip(1.0, 1), sr=44100)
        first = noisy_sources.prepare(original, self.root / "prepared", "src", "clip")
        fx.write_audio(original, fx.noise_clip(1.0, 2), sr=44100)  # the original changed
        second = noisy_sources.prepare(original, self.root / "prepared", "src", "clip")
        self.assertNotEqual(first, second)
        expected = noisy_sources.prepare(fx.write_audio(self.root / "fresh.wav", fx.noise_clip(1.0, 2), sr=44100),
                                         self.root / "fresh", "src", "clip")
        self.assertEqual(noisy_sources.pcm_sha256(second), noisy_sources.pcm_sha256(expected))

    def test_tampered_prepared_clip_is_redone(self):
        original = fx.write_audio(self.root / "orig.wav", fx.noise_clip(1.0, 3), sr=44100)
        path = noisy_sources.prepare(original, self.root / "prepared", "src", "clip")
        digest = noisy_sources.pcm_sha256(path)
        fx.write_audio(path, fx.noise_clip(1.0, 4), subtype="FLOAT")
        again = noisy_sources.prepare(original, self.root / "prepared", "src", "clip")
        self.assertEqual(noisy_sources.pcm_sha256(again), digest)

    def test_extra_files_or_links_in_a_reused_cache_refused(self):
        # Review round 5, finding 3.
        archive = self.root / "b.zip"
        archive.write_bytes(_zip_bytes({"musan/speech/us-gov/a.wav": b"one"}))
        verification = {"files": [{"name": "b.zip", "sha256": noisy_paths.sha256_file(archive)}]}
        args = ("toy", "b", [archive], "zip", ("all", lambda n: True), BudgetTest.SMALL, verification,
                self.root / "extracted")
        first = noisy_sources.extract_archive(*args)
        dest = Path(first["dest"])
        (dest / "musan/speech/us-gov/extra.wav").write_bytes(b"planted")
        with self.assertRaisesRegex(noisy_sources.InputChanged, "unexpected"):
            noisy_sources.extract_archive(*args)
        (dest / "musan/speech/us-gov/extra.wav").unlink()
        (dest / "musan/link.wav").symlink_to(dest / "musan/speech/us-gov/a.wav")
        with self.assertRaisesRegex(noisy_sources.InputChanged, "unexpected"):
            noisy_sources.extract_archive(*args)
        (dest / "musan/link.wav").unlink()
        self.assertEqual(noisy_sources.extract_archive(*args)["dest"], first["dest"])

    def test_indexers_use_only_the_verified_member_list(self):
        root = self.root / "musan-x"
        for name in ("speech-us-gov-0000", "speech-us-gov-0001", "extra"):
            fx.write_audio(root / "musan/speech/us-gov" / f"{name}.wav", fx.speech_like(2.0, len(name)))
        members = [m for m in fx.members_of(root) if "extra" not in m[1]]
        keys = [c["key"] for c in noisy_sources.index_musan(members)]
        self.assertEqual(keys, ["musan:speech-us-gov-0000", "musan:speech-us-gov-0001"])

    def test_prepared_audio_bound_to_the_member(self):
        original = fx.write_audio(self.root / "m.wav", fx.noise_clip(1.0, 5), sr=44100)
        with self.assertRaises(noisy_sources.InputChanged):
            noisy_sources.prepare(original, self.root / "prepared", "src", "m", expected_sha256="0" * 64)
        path = noisy_sources.prepare(original, self.root / "prepared", "src", "m",
                                     expected_sha256=noisy_paths.sha256_file(original))
        self.assertTrue(path.exists())

    def test_extraction_bound_to_verified_archive(self):
        archive = self.root / "a.zip"
        archive.write_bytes(_zip_bytes({"TBUS/ch01.wav": b"one"}))
        verification = {"files": [{"name": "a.zip", "sha256": noisy_paths.sha256_file(archive)}]}
        args = ("toy", "a", [archive], "zip", ("all", lambda n: True), BudgetTest.SMALL)
        first = noisy_sources.extract_archive(*args, verification, self.root / "extracted")
        again = noisy_sources.extract_archive(*args, verification, self.root / "extracted")
        self.assertEqual(first["dest"], again["dest"])
        member = Path(first["dest"]) / "TBUS/ch01.wav"
        member.write_bytes(b"two")  # an extracted member changed
        with self.assertRaises(noisy_sources.InputChanged):
            noisy_sources.extract_archive(*args, verification, self.root / "extracted")
        other = {"files": [{"name": "a.zip", "sha256": "f" * 64}]}  # another verified archive
        third = noisy_sources.extract_archive(*args, other, self.root / "extracted")
        self.assertNotEqual(third["dest"], first["dest"])


if __name__ == "__main__":
    unittest.main()
