"""syn-dialog voice-provenance contracts (labeling plan "New corpora"): the corpus is used only if every
reference voice is confirmed to be a training speaker of the voice corpus; if the check fails or cannot be
established, the corpus is marked dropped and nothing else is done for it.

Run from finetune/relabel:  ./python -m unittest discover -s tests -p 'test_corpora_*' -v
"""
from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from corpora import syn_dialog

TRAIN = ("train-clean-100", "train-clean-360", "train-other-500")


def bank(subsets: list[str]) -> list[dict]:
    return [{"identifier": str(i), "subset": s} for i, s in enumerate(subsets)]


def decide(b, voices, n=None, complete=True):
    n = len(voices) if n is None else n
    return syn_dialog.decide(b, "identifier", "subset", TRAIN, voices, n, complete)


class DecideTest(unittest.TestCase):
    def test_unnamed_voices_with_training_only_bank_pass(self):
        self.assertTrue(decide(bank(["train-clean-100", "train-other-500"]), [set(), set()])["passed"])

    def test_unnamed_voices_with_dev_test_speakers_in_bank_not_established(self):
        v = decide(bank(["train-clean-100", "dev-clean", "test-other"]), [set(), set()])
        self.assertFalse(v["passed"])
        self.assertEqual(v["verdict"], "not_established")
        self.assertEqual(v["bank_non_training_speakers"], 2)

    def test_named_training_voices_pass_only_when_complete(self):
        b = bank(["train-clean-100", "train-clean-360", "dev-clean"])
        self.assertTrue(decide(b, [{"0"}, {"1", "0"}])["passed"])
        self.assertFalse(decide(b, [{"0"}, {"1"}], complete=False)["passed"])
        self.assertFalse(decide(b, [{"0"}, set()])["passed"])

    def test_named_non_training_or_unknown_voice_fails(self):
        b = bank(["train-clean-100", "dev-clean"])
        self.assertEqual(decide(b, [{"0"}, {"1"}])["verdict"], "failed")
        self.assertEqual(decide(b, [{"0"}, {"99"}])["verdict"], "failed")


def tar_head(records: list[dict]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for k, rec in enumerate(records):
            for name, data in ((f"d{k}.opus", b"\0" * 3000), (f"d{k}.json", json.dumps(rec).encode())):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class RunCheckTest(unittest.TestCase):
    def test_records_without_voice_ids_and_mixed_bank_drop_the_corpus(self):
        recs = [{"personas": {"DOCTOR": {"name": "A", "gender": "F"}}, "turns": [{"speaker": "DOCTOR", "text": "hi"}]}] * 3
        csv = "identifier,subset\n1,train-clean-100\n2,dev-clean\n"
        reg = {"hf_id": "x", "revision": "r", "train_files": ["a.tar", "b.tar"],
               "voice_bank": {"hf_id": "v", "revision": "r", "metadata_file": "m.csv", "id_field": "identifier",
                              "subset_field": "subset", "training_subsets": list(TRAIN)}}

        full = tar_head(recs)
        reads = []

        def get(hf_id, revision, path, byte_range=None):
            if path == "m.csv":
                return csv.encode()
            reads.append(byte_range)
            return full[byte_range[0]:byte_range[1]]

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            v = syn_dialog.run_check(out, reg, get)
            self.assertFalse(v["passed"])
            self.assertEqual(v["records_inspected"], 6)   # 3 records per shard, 2 shards
            opus = [(m.offset_data, m.offset_data + m.size) for m in tarfile.open(fileobj=io.BytesIO(full))
                    if m.name.endswith(".opus")]
            for start, end in reads:   # no byte of an audio member is read
                self.assertFalse(any(start < e and s < end for s, e in opus), (start, end))
            status = json.loads((out / "status.json").read_text())
            self.assertEqual(status["status"], "dropped")
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["status.json", "voice_check.json"])

    def test_voice_keys_are_found(self):
        self.assertEqual(syn_dialog.record_voice_ids({"personas": {"D": {"voice": "1089", "speaker": "DOCTOR"}},
                                                      "tts": {"ref_audio": "x/14.wav"}}), {"1089", "x/14.wav"})


if __name__ == "__main__":
    unittest.main()
