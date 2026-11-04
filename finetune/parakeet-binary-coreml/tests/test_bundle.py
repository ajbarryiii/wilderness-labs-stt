"""Contract of the LocalFlow bundle written by prepare_bundle.py (invented files only).

- bundle.json: model "localflow", display_name "LocalFlow", base_model "kea-b" and training_step 250000 for
  MB1 or "kea-b-prime" and 347137 for Kea-B' (b100-main), export_sha256 = the 1-bit export's, encoder = the arm id, layout "plain", compute_units
  "cpuAndNeuralEngine", decoder "native-fp32", buckets_seconds = the built functions ([15] by default: LocalFlow runs
  only b15; [4, 8, 15, 30] on request), sample_rate 16000, and
  "files": the SHA-256 of every bundle file (frontend, decoder/joint, vocabulary, every Encoder.mlmodelc file);
- only registered 1-bit arms and the corrected 0/1 arm Z1C are accepted (never the timing-only Z1Q/Z1S);
  required files must exist;
- localflow.apply_buckets: the default builds b15 alone with upstream's 15 s position tables (MAX_T 188); the
  four LocalFlow functions extend them to 30 s (MAX_T 376); bucket sets without b15 are refused;
- vocabulary.json: the 1024 SentencePiece pieces by id (tokenizer.vocab first column);
- the conversion refuses any export other than MB1 or Kea-B' (their SHA-256 are pinned).
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import common  # noqa: F401  (paths)

MB1_SHA = "9873d2f1a53127c673d0201db98cd3ba859ba432f334fef55e9e24fd057b9651"
KEAB_PRIME_SHA = "ec05526ed7aa7c742ee970c09724b1b8ccfaef6eb10f2ff1a11b031e9dfc4995"


def invented_bundle(root: Path) -> list[str]:
    files = {"frontend.json": b"{}", "frontend.f32bin": b"\0" * 8, "decoder_joint.json": b"{}",
             "decoder_joint.f32bin": b"\1" * 8, "vocabulary.json": b"{}",
             "Encoder.mlmodelc/model.mil": b"program", "Encoder.mlmodelc/coremldata.bin": b"c",
             "Encoder.mlmodelc/weights/weight.bin": b"w" * 64, "Encoder.mlmodelc/analytics/coremldata.bin": b"a"}
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    return sorted(files)


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_manifest_shape(self):
        import prepare_bundle

        names = invented_bundle(self.root)
        files = prepare_bundle.bundle_files(self.root)
        self.assertEqual(sorted(files), names)
        m = prepare_bundle.bundle_manifest(self.root, "B1L", MB1_SHA, files)
        expect = {"model": "localflow", "display_name": "LocalFlow", "base_model": "kea-b", "training_step": 250000,
                  "export_sha256": MB1_SHA, "encoder": "B1L", "layout": "plain", "compute_units": "cpuAndNeuralEngine",
                  "decoder": "native-fp32", "buckets_seconds": [15], "sample_rate": 16000}
        for k, v in expect.items():
            self.assertEqual(m[k], v, k)
        self.assertEqual(set(m["files"]), set(names))
        for rel, digest in m["files"].items():
            self.assertEqual(digest, hashlib.sha256((self.root / rel).read_bytes()).hexdigest(), rel)
        json.dumps(m)  # serializable as written

    def test_manifest_lists_requested_buckets(self):
        import prepare_bundle

        invented_bundle(self.root)
        files = prepare_bundle.bundle_files(self.root)
        m = prepare_bundle.bundle_manifest(self.root, "B1Q", MB1_SHA, files, buckets=(4, 8, 15, 30))
        self.assertEqual(m["buckets_seconds"], [4, 8, 15, 30])

    def test_apply_buckets(self):
        import localflow
        from mil import encoder

        saved = (encoder.BUCKETS, encoder.MAX_T, encoder.fold_positions)
        try:
            localflow.apply_buckets()
            self.assertEqual((encoder.BUCKETS, encoder.MAX_T), ({15: 1501}, 188))
            encoder.BUCKETS, encoder.MAX_T, encoder.fold_positions = saved
            localflow.apply_buckets((4, 8, 15, 30))
            self.assertEqual((encoder.BUCKETS, encoder.MAX_T), ({4: 401, 8: 801, 15: 1501, 30: 3001}, 376))
            encoder.BUCKETS, encoder.MAX_T, encoder.fold_positions = saved
            with self.assertRaises(ValueError):
                localflow.apply_buckets((4, 8, 30))
        finally:
            encoder.BUCKETS, encoder.MAX_T, encoder.fold_positions = saved

    def test_keab_prime_manifest_names_its_model(self):
        import prepare_bundle

        names = invented_bundle(self.root)
        m = prepare_bundle.bundle_manifest(self.root, "B1Q", KEAB_PRIME_SHA, prepare_bundle.bundle_files(self.root))
        self.assertEqual((m["base_model"], m["training_step"], m["export_sha256"], m["encoder"]),
                         ("kea-b-prime", 347137, KEAB_PRIME_SHA, "B1Q"))
        self.assertEqual(set(m["files"]), set(names))

    def test_manifest_accepts_corrected_arm(self):
        import prepare_bundle

        invented_bundle(self.root)
        m = prepare_bundle.bundle_manifest(self.root, "Z1C", KEAB_PRIME_SHA, prepare_bundle.bundle_files(self.root))
        self.assertEqual(m["encoder"], "Z1C")
        with self.assertRaises(ValueError):  # timing-only arms never make a bundle
            prepare_bundle.bundle_manifest(self.root, "Z1Q", KEAB_PRIME_SHA, prepare_bundle.bundle_files(self.root))

    def test_refuses_unknown_arm_and_missing_files(self):
        import prepare_bundle

        names = invented_bundle(self.root)
        with self.assertRaises(ValueError):
            prepare_bundle.bundle_manifest(self.root, "C6s8x", MB1_SHA, names)
        (self.root / "decoder_joint.f32bin").unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            prepare_bundle.bundle_files(self.root)

    def test_vocabulary(self):
        import prepare_bundle

        lines = ["<unk>\t0"] + [f"▁p{i}\t-{i}.5" for i in range(1, 1024)]
        (self.root / "tokenizer.vocab").write_text("\n".join(lines) + "\n")
        vocab = prepare_bundle.vocabulary(self.root)
        self.assertEqual(len(vocab), 1024)
        self.assertEqual(vocab[0], "<unk>")
        self.assertEqual(vocab[1023], "▁p1023")
        (self.root / "tokenizer.vocab").write_text("\n".join(lines[:-1]) + "\n")
        with self.assertRaises(ValueError):
            prepare_bundle.vocabulary(self.root)

    def test_export_pin(self):
        import prepare_bundle

        self.assertEqual(prepare_bundle.MB1_EXPORT_SHA256, MB1_SHA)
        self.assertEqual(prepare_bundle.check_export(MB1_SHA), ("kea-b", 250000))
        self.assertEqual(prepare_bundle.check_export(KEAB_PRIME_SHA), ("kea-b-prime", 347137))
        with self.assertRaises(ValueError):
            prepare_bundle.check_export("0e1db1b49548c2858a20c8c08ce920beccfbb849b8ac155829d4e95f87fed0d4")


if __name__ == "__main__":
    unittest.main()
