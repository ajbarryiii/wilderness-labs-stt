"""Manifest parsing, dataset/label contract, collation, subsets, data order, and checkpoint lock."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperProcessor

import checkpoint
import data
import paths

LINES = {"19-198-0000": (0.5, "HELLO WORLD"), "19-198-0001": (1.0, "MISTER BROWN'S SECOND COLOR"),
         "19-198-0002": (31.0, "A VERY LONG UTTERANCE")}
LOG = ("start 2026-09-29T21:01:16Z\nfake sha256 abc123 bytes 42\nfake flac_count 3\n"
       "done 2026-09-29T21:03:01Z\n")


class DataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp())
        cls.root = cls.tmp / "LibriSpeech" / "fake"
        chapter = cls.root / "19" / "198"
        chapter.mkdir(parents=True)
        lines = []
        for uid, (seconds, text) in reversed(LINES.items()):
            t = np.arange(int(seconds * paths.SAMPLE_RATE)) / paths.SAMPLE_RATE
            sf.write(chapter / f"{uid}.flac", (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32),
                     paths.SAMPLE_RATE, format="FLAC")
            lines.append(f"{uid} {text}")
        (chapter / "19-198.trans.txt").write_text("\n".join(lines) + "\n")
        cls.processor = WhisperProcessor.from_pretrained(paths.MODEL_DIR, local_files_only=True)
        cls.manifest = data.scan(cls.root)
        cls.dataset = data.LibriSpeechDataset(cls.manifest, cls.processor.feature_extractor,
                                              cls.processor.tokenizer, 50257)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp)

    def test_scan(self) -> None:
        self.assertEqual([r["id"] for r in self.manifest], sorted(LINES))
        for record in self.manifest:
            seconds, text = LINES[record["id"]]
            self.assertEqual(record["text"], text)
            self.assertAlmostEqual(record["duration_s"], seconds)
            self.assertTrue(Path(record["path"]).is_absolute() and Path(record["path"]).is_file())

    def test_build_manifest_count_and_provenance(self) -> None:
        (self.tmp / "download.log").write_text(LOG)
        with patch.dict(paths.SPLITS, {"fake": self.root}), \
                patch.dict(paths.EXPECTED_UTTERANCES, {"fake": 3}), \
                patch.object(paths, "MANIFESTS", self.tmp / "manifests"), \
                patch.object(paths, "DATA", self.tmp):
            self.assertEqual(data.build_manifest("fake", refresh=True), self.manifest)
            self.assertEqual(data.build_manifest("fake"), self.manifest)
            provenance = json.loads((self.tmp / "provenance.json").read_text())
            fake = provenance["splits"]["fake"]
            self.assertEqual((fake["tarball_sha256"], fake["tarball_bytes"], fake["flac_count"],
                              fake["manifest_utterances"]), ("abc123", 42, 3, 3))
            self.assertEqual(fake["url"], "https://www.openslr.org/resources/12/fake.tar.gz")
            self.assertEqual(provenance["download_finished"], "2026-09-29T21:03:01Z")
            with patch.dict(paths.EXPECTED_UTTERANCES, {"fake": 4}):
                with self.assertRaisesRegex(ValueError, "found 3 utterances.*expected 4"):
                    data.build_manifest("fake", refresh=True)

    def test_item_contract(self) -> None:
        for index, record in enumerate(self.manifest):
            item = self.dataset[index]
            self.assertEqual(tuple(item["input_features"].shape), (80, 3000))
            self.assertEqual(item["input_features"].dtype, torch.float32)
            labels = item["labels"].tolist()
            self.assertEqual((labels[0], labels[-1]), (50362, 50256))
            self.assertNotIn(50257, labels)
            text = self.processor.tokenizer.decode(labels, skip_special_tokens=True)
            self.assertEqual(text, record["text"].lower())
            self.assertEqual(item["truncated"], record["id"] == "19-198-0002")

    def test_collate(self) -> None:
        items = [self.dataset[i] for i in range(3)]
        batch = data.collate(items)
        self.assertEqual(tuple(batch["input_features"].shape), (3, 80, 3000))
        self.assertEqual(batch["input_features"].dtype, torch.float32)
        self.assertEqual(batch["labels"].dtype, torch.long)
        self.assertEqual(batch["ids"], sorted(LINES))
        self.assertEqual(batch["truncated"], [False, False, True])
        for row, item in enumerate(items):
            n = len(item["labels"])
            self.assertTrue(torch.equal(batch["labels"][row, :n], item["labels"]))
            self.assertTrue((batch["labels"][row, n:] == -100).all())
        self.assertTrue((batch["labels"] == -100).any())

    def test_dev_subset(self) -> None:
        manifest = [{"id": f"{i:05d}"} for i in range(2703)]
        subset = data.dev_subset(list(reversed(manifest)), 400)
        self.assertEqual(len(subset), 400)
        self.assertEqual([r["id"] for r in subset], [f"{i:05d}" for i in range(0, 2400, 6)])
        self.assertEqual(subset, data.dev_subset(manifest, 400))
        self.assertEqual(len(data.dev_subset(manifest, 16)), 16)

    def test_epoch_order(self) -> None:
        sampler = data.EpochSampler(100, seed=paths.SEED)
        first = list(sampler)
        self.assertEqual(first, list(data.EpochSampler(100, seed=paths.SEED)))
        self.assertEqual(sorted(first), list(range(100)))
        sampler.set_epoch(1)
        self.assertNotEqual(list(sampler), first)

    def test_loader_reseeds_each_epoch_with_persistent_workers(self) -> None:
        items = [{"input_features": torch.zeros(1), "labels": torch.tensor([i]), "id": str(i),
                  "truncated": False} for i in range(10)]
        loader, sampler = data.train_loader(items, batch_size=3, workers=2)
        for epoch in range(2):
            sampler.set_epoch(epoch)
            order = torch.randperm(10, generator=torch.Generator().manual_seed(paths.SEED + epoch))
            seen = [i for batch in loader for i in batch["ids"]]
            self.assertEqual(seen, [str(i) for i in order.tolist()[:9]])


class LockTest(unittest.TestCase):
    def test_real_checkpoint_verifies(self) -> None:
        lock = checkpoint.verify_lock()
        self.assertEqual(lock["revision"], paths.MODEL_REVISION)

    def test_tampered_copy_fails(self) -> None:
        names = ["config.json", "generation_config.json", "normalizer.json"]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name in names + ["lock.json"]:
                shutil.copy(paths.MODEL_DIR / name, tmp / name)
            checkpoint.verify_lock(tmp, names)
            raw = bytearray((tmp / "config.json").read_bytes())
            raw[10] ^= 1
            (tmp / "config.json").write_bytes(bytes(raw))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                checkpoint.verify_lock(tmp, names)


if __name__ == "__main__":
    unittest.main()
