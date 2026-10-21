"""Leakage-control contracts (plans/build.md "Leakage controls", "Code and tests" 5): the blocklist
covers every reserved unit (datasets and campaigns, BUT rooms, every FSD50K eval-split sound ID and
uploader, the decoded-PCM SHA-256 of every pool file), and check_train_pool fails on each kind of
match while accepting a disjoint TRAIN candidate.

Review round 2, finding 6: a candidate with insufficient or ambiguous provenance is refused (no
dataset or PCM hash, a Freesound-derived row without sound ID and uploader, an FSD50K row without
a split, a MUSAN row naming a parent of a reserved subset), and reserved splits are recognized.

Review round 3, finding 2: every negative case is an otherwise valid candidate and asserts its
specific reason (the exact set of match kinds). MUSAN subsets must name a known group ("noise" is
ambiguous: it contains free-sound), a BUT room may not be empty or whitespace, collection wrappers
(DNS-Challenge, RIRS_NOISES/SLR28, SLR26) resolve to their canonical origins, whose rules then
apply, and an unknown dataset or wrapper subset fails closed.

Review round 4, finding 2: an explicit origin may not contradict a known wrapper mapping, may only
carry provenance fields (never a PCM hash), and cannot hide the candidate's own PCM hash or a
reserved identity stated at any link of the chain.

Review round 5 (finding 1; check_train_pool is provisional per plans/build.md "Scope of the TRAIN
check"): an explicit origin is refused whenever the candidate's dataset is a known wrapper or
MUSAN, so an origin can never override a mapped subset. This is a deliberate contract change: the
round-4 cases that resolved a wrapper through an explicit origin now fail closed as ambiguous,
while reserved identities stated on the candidate are still reported."""
from __future__ import annotations

import unittest

import noisy_fixtures as fx

import noisy_blocklist
import noisy_sources


class BlocklistTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = fx.fresh_dir("blocklist")
        cls.pool = noisy_sources.load_pool(fx.make_pool(root, rotor=True), root / "rotor.json")
        cls.block = noisy_blocklist.make_blocklist(cls.pool)

    def test_covers_every_reserved_unit(self):
        units = {(u["dataset"], u.get("subset"), u.get("room")) for u in self.block["units"]}
        for unit in (("zenodo-3997406", None, None), ("demand", None, None), ("musan", "speech/us-gov", None),
                     ("but-reverbdb", None, "Room0"), ("but-reverbdb", None, "Room1"),
                     ("but-reverbdb", None, "Room2")):
            self.assertIn(unit, units)
        self.assertEqual(set(self.block["freesound_ids"]), {1001, 1002, 2001, 3001, 4001, 4002, 9001, 9002})
        self.assertTrue({"u-other", "u-gun", "u-rotor", "u-rotor2"} <= set(self.block["freesound_uploaders"]))
        hashes = set(self.block["pcm_sha256"])
        for item in noisy_sources.pool_files(self.pool):
            self.assertIn(item["pcm_sha256"], hashes, item["key"])
        self.assertTrue(self.block["excluded_families"])

    def test_check_passes_on_disjoint_candidates(self):
        rows = [{"dataset": "fsd50k", "split": "dev", "freesound_id": 777, "uploader": "someone-else",
                 "pcm_sha256": "0" * 64},
                {"dataset": "musan", "subset": "noise/free-sound", "freesound_id": 778, "uploader": "x",
                 "pcm_sha256": "1" * 64},
                {"dataset": "but-reverbdb", "room": "Room9", "pcm_sha256": "2" * 64},
                {"dataset": "pyroomacoustics-simulated", "pcm_sha256": "3" * 64}]
        self.assertEqual(noisy_blocklist.check_train_pool(rows, self.block), rows)

    def kinds(self, row: dict) -> set[str]:
        """The exact set of match kinds for one candidate ({} if it passes)."""
        try:
            noisy_blocklist.check_train_pool([row], self.block)
        except noisy_blocklist.LeakageError as error:
            self.assertTrue(all(m["row"] == 0 for m in error.matches))
            return {m["kind"] for m in error.matches}
        return set()

    def test_check_fails_on_each_kind_of_match(self):
        some_hash = noisy_sources.pool_files(self.pool)[0]["pcm_sha256"]
        h = "4" * 64
        cases = {  # otherwise valid candidates, each with exactly one reason
            "dataset": ({"dataset": "DEMAND", "pcm_sha256": h}, {"unit"}),
            "campaign": ({"dataset": "zenodo-3997406", "pcm_sha256": h}, {"unit"}),
            "subset": ({"dataset": "musan", "subset": "speech/us-gov/speech-us-gov-0001.wav", "pcm_sha256": h},
                       {"unit"}),
            "room": ({"dataset": "but-reverbdb", "room": "Room1", "pcm_sha256": h}, {"unit"}),
            "sound_id": ({"dataset": "esc50", "freesound_id": 4002, "uploader": "fresh-uploader", "pcm_sha256": h},
                         {"freesound_id"}),  # reserved though unused
            "uploader": ({"dataset": "freesound", "freesound_id": 31337, "uploader": "u-other", "pcm_sha256": h},
                         {"uploader"}),
            "pcm": ({"dataset": "pyroomacoustics-simulated", "pcm_sha256": some_hash}, {"pcm_sha256"}),
            "reserved_split": ({"dataset": "fsd50k", "split": "eval", "freesound_id": 31337, "uploader": "fresh",
                                "pcm_sha256": h}, {"reserved_split"}),
        }
        for name, (row, expected) in cases.items():
            self.assertEqual(self.kinds(row), expected, name)

    def test_insufficient_or_ambiguous_provenance_refused(self):
        h = "6" * 64
        insufficient, ambiguous, unknown = {"insufficient_provenance"}, {"ambiguous_provenance"}, {"unknown_dataset"}
        cases = {
            "empty": ({}, insufficient),
            "no_dataset": ({"pcm_sha256": h}, insufficient),
            "no_hash": ({"dataset": "pyroomacoustics-simulated"}, insufficient),
            "fsd50k_no_split": ({"dataset": "fsd50k", "freesound_id": 31337, "uploader": "z", "pcm_sha256": h},
                                insufficient),
            "fsd50k_no_id": ({"dataset": "fsd50k", "split": "dev", "uploader": "z", "pcm_sha256": h}, insufficient),
            "freesound_no_uploader": ({"dataset": "freesound", "freesound_id": 31338, "pcm_sha256": h}, insufficient),
            "musan_parent_speech": ({"dataset": "musan", "subset": "speech", "pcm_sha256": h}, ambiguous),
            "musan_parent_noise": ({"dataset": "musan", "subset": "noise", "pcm_sha256": h}, ambiguous),
            "musan_unknown_group": ({"dataset": "musan", "subset": "other/x.wav", "pcm_sha256": h}, ambiguous),
            "musan_no_subset": ({"dataset": "musan", "pcm_sha256": h}, ambiguous),
            "musan_freesound_no_id": ({"dataset": "musan", "subset": "noise/free-sound/noise-free-sound-0001.wav",
                                       "pcm_sha256": h}, insufficient),
            "but_no_room": ({"dataset": "but-reverbdb", "pcm_sha256": h}, ambiguous),
            "but_empty_room": ({"dataset": "but-reverbdb", "room": "", "pcm_sha256": h}, ambiguous),
            "but_blank_room": ({"dataset": "but-reverbdb", "room": "   ", "pcm_sha256": h}, ambiguous),
            "unknown_dataset": ({"dataset": "some-noise-collection", "pcm_sha256": h}, unknown),
        }
        for name, (row, expected) in cases.items():
            self.assertEqual(self.kinds(row), expected, name)

    def test_collection_wrappers_resolve_to_their_origins(self):
        h = "8" * 64
        cases = {
            "dns_demand": ({"dataset": "DNS-Challenge", "subset": "DEMAND/TBUS/ch01.wav", "pcm_sha256": h}, {"unit"}),
            "dns_freesound_reserved_uploader": ({"dataset": "dns-challenge", "subset": "freesound/123.wav",
                                                 "freesound_id": 123, "uploader": "u-gun", "pcm_sha256": h},
                                                {"uploader"}),
            "dns_freesound_no_id": ({"dataset": "dns-challenge", "subset": "freesound/123.wav", "pcm_sha256": h},
                                    {"insufficient_provenance"}),
            "dns_unknown_subset": ({"dataset": "dns-challenge", "subset": "mystery/1.wav", "pcm_sha256": h},
                                   {"ambiguous_provenance"}),
            "dns_no_subset": ({"dataset": "dns-challenge", "pcm_sha256": h}, {"ambiguous_provenance"}),
            "slr28_pointsource_needs_origin": ({"dataset": "RIRS_NOISES", "subset": "pointsource_noises/n1.wav",
                                                "pcm_sha256": h}, {"ambiguous_provenance"}),
            "slr28_alias_with_explicit_origin_musan_usgov": (  # round 5: origins on wrappers are refused
                {"dataset": "slr28", "subset": "pointsource_noises/n1.wav", "pcm_sha256": h,
                 "origin": {"dataset": "musan", "subset": "speech/us-gov/speech-us-gov-0002.wav"}},
                {"ambiguous_provenance"}),
            "dns_slr26_nested": ({"dataset": "dns-challenge", "subset": "slr26/simulated_rirs/x.wav", "pcm_sha256": h},
                                 set()),
            "slr28_simulated": ({"dataset": "rirs_noises", "subset": "simulated_rirs/smallroom/x.wav",
                                 "pcm_sha256": h}, set()),
            "slr26_whole": ({"dataset": "SLR26", "subset": "simulated_rirs_8k/x.wav", "pcm_sha256": h}, set()),
        }
        for name, (row, expected) in cases.items():
            self.assertEqual(self.kinds(row), expected, name)

    def reason(self, row: dict, kind: str) -> str:
        try:
            noisy_blocklist.check_train_pool([row], self.block)
        except noisy_blocklist.LeakageError as error:
            return " ".join(str(m["value"]) for m in error.matches if m["kind"] == kind)
        return ""

    def test_explicit_origins_cannot_erase_provenance(self):
        reserved = noisy_sources.pool_files(self.pool)[0]["pcm_sha256"]
        h = "9" * 64
        conflicting = {"dataset": "DNS-Challenge", "subset": "DEMAND/TBUS/ch01.wav", "pcm_sha256": h,
                       "origin": {"dataset": "slr26-simulated"}}
        self.assertEqual(self.kinds(conflicting), {"ambiguous_provenance"})
        self.assertIn("conflicts", self.reason(conflicting, "ambiguous_provenance"))
        star_conflict = {"dataset": "SLR26", "subset": "x.wav", "pcm_sha256": h, "origin": {"dataset": "demand"}}
        self.assertEqual(self.kinds(star_conflict), {"ambiguous_provenance"})
        plain_conflict = {"dataset": "freesound", "freesound_id": 1, "uploader": "a", "pcm_sha256": h,
                          "origin": {"dataset": "rwcp"}}
        self.assertEqual(self.kinds(plain_conflict), {"ambiguous_provenance"})
        # The candidate's own hash is immutable: an origin cannot carry one, and a valid origin
        # does not hide a reserved outer hash.
        hash_in_origin = {"dataset": "rirs_noises", "subset": "pointsource_noises/n1.wav", "pcm_sha256": reserved,
                          "origin": {"dataset": "rwcp", "pcm_sha256": h}}
        self.assertEqual(self.kinds(hash_in_origin), {"ambiguous_provenance", "pcm_sha256"})
        self.assertIn("pcm_sha256", self.reason(hash_in_origin, "ambiguous_provenance"))
        wrapper_origin_reserved_hash = {"dataset": "rirs_noises", "subset": "pointsource_noises/n1.wav",
                                        "pcm_sha256": reserved, "origin": {"dataset": "rwcp"}}
        self.assertEqual(self.kinds(wrapper_origin_reserved_hash), {"ambiguous_provenance", "pcm_sha256"})
        # A reserved identity stated at an outer link is still checked after the origin replaces it.
        outer_uploader = {"dataset": "rirs_noises", "subset": "pointsource_noises/n1.wav", "pcm_sha256": h,
                          "freesound_id": 5, "uploader": "u-gun",
                          "origin": {"dataset": "freesound", "freesound_id": 31337, "uploader": "fresh"}}
        self.assertEqual(self.kinds(outer_uploader), {"ambiguous_provenance", "uploader"})
        outer_sound_id = {"dataset": "rirs_noises", "subset": "pointsource_noises/n1.wav", "pcm_sha256": h,
                          "freesound_id": 4002, "uploader": "fresh",
                          "origin": {"dataset": "freesound", "freesound_id": 31337, "uploader": "fresh"}}
        self.assertEqual(self.kinds(outer_sound_id), {"ambiguous_provenance", "freesound_id"})
        unknown_field = {"dataset": "slr28", "subset": "pointsource_noises/n1.wav", "pcm_sha256": h,
                         "origin": {"dataset": "rwcp", "note": "trust me"}}
        self.assertEqual(self.kinds(unknown_field), {"ambiguous_provenance"})
        wrapper_origin = {"dataset": "slr28", "subset": "real_rirs_isotropic_noises/x.wav", "pcm_sha256": h,
                          "origin": {"dataset": "rwcp"}}
        self.assertEqual(self.kinds(wrapper_origin), {"ambiguous_provenance"})  # round 5: fails closed
        same_dataset_origin = {"dataset": "freesound", "freesound_id": 31337, "uploader": "a", "pcm_sha256": h,
                               "origin": {"dataset": "freesound", "freesound_id": 31337, "uploader": "a"}}
        self.assertEqual(self.kinds(same_dataset_origin), set())

    def test_origins_cannot_override_mapped_subsets(self):
        # Review round 5, finding 1: the two bypasses from the review, each otherwise valid.
        h = "a" * 64
        musan = {"dataset": "musan", "subset": "noise/free-sound/noise-free-sound-0001.wav", "pcm_sha256": h,
                 "origin": {"dataset": "musan", "subset": "speech/librivox"}}
        self.assertEqual(self.kinds(musan), {"ambiguous_provenance"})
        self.assertIn("origin", self.reason(musan, "ambiguous_provenance"))
        dns = {"dataset": "dns-challenge", "subset": "slr28/pointsource_noises/n1.wav", "pcm_sha256": h,
               "origin": {"dataset": "rirs-noises", "subset": "simulated_rirs/smallroom/x.wav"}}
        self.assertEqual(self.kinds(dns), {"ambiguous_provenance"})
        self.assertIn("origin", self.reason(dns, "ambiguous_provenance"))

    def test_fsd50k_dev_split_row_with_full_provenance_passes(self):
        row = {"dataset": "FSD50K", "split": "dev", "freesound_id": 31337, "uploader": "zed", "pcm_sha256": "7" * 64}
        self.assertEqual(noisy_blocklist.check_train_pool([row], self.block), [row])


if __name__ == "__main__":
    unittest.main()
