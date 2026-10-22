"""Contract tests of the Kea-B Hugging Face release kit (plans/release.md section 8).

CPU only, seconds, no model: synthetic tensors and a fake NeMo model class stand in for the
real export. The real-model test is the staging run itself (plans/release.md section 4).
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from omegaconf import OmegaConf  # imported before fake_nemo patches sys.modules (the patch drops new imports)
from safetensors.torch import save_file

import binary_export
import binary_test_paths
import paths

PKG = Path(__file__).resolve().parents[1]
HF = PKG / "hf"
TERNARY_HF = PKG.parent / "parakeet-ternary" / "hf"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


load_binary = _load("kea_b_load_binary", HF / "load_binary.py")
stage = _load("kea_b_stage", HF / "stage.py")
verify = _load("kea_b_verify_download", HF / "verify_download.py")

MODEL_TARGET = "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ----------------------------------------------------------------------------- fake NeMo

class FakeNeMo:
    """Stands in for EncDecRNNTBPEModel: keeps the strict-loaded state dict."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.training = True
        self._state: dict[str, torch.Tensor] = {}

    def load_state_dict(self, state, strict=True):
        assert strict is True
        self._state = dict(state)

    def to(self, device=None, dtype=None):
        self._state = {k: v.to(dtype) if dtype is not None and v.is_floating_point() else v
                       for k, v in self._state.items()}
        return self

    def eval(self):
        self.training = False
        return self

    def state_dict(self):
        return self._state


def fake_nemo():
    module = types.ModuleType("nemo.collections.asr.models")
    module.EncDecRNNTBPEModel = FakeNeMo
    return mock.patch.dict(sys.modules, {"nemo.collections.asr.models": module})


def make_export(root: Path, seed: int = 0) -> dict:
    """A tiny parakeet-binary-v1 export: two quantized layers (odd widths), FP16, FP32 and int tensors."""
    g = torch.Generator().manual_seed(seed)
    layers = {"encoder.layers.0.feed_forward1.linear1": ("linear", (3, 13)),
              "encoder.layers.0.conv.pointwise_conv1": ("pointwise_conv1d", (4, 9))}
    tensors = {}
    for name, (_, (rows, cols)) in layers.items():
        codes = (2 * torch.randint(0, 2, (rows, cols), generator=g) - 1).to(torch.int8)
        tensors[f"{name}.codes"] = binary_export.pack_codes(codes)
        tensors[f"{name}.scale"] = torch.rand(rows, generator=g) + 0.01
    tensors["encoder.layers.0.norm.weight"] = torch.randn(5, generator=g).half()
    tensors["preprocessor.featurizer.window"] = torch.randn(6, generator=g)
    tensors["encoder.layers.0.conv.batch_norm.num_batches_tracked"] = torch.tensor(7)
    root.mkdir(parents=True, exist_ok=True)
    save_file(tensors, root / "export.safetensors", metadata={"format": "parakeet-binary-v1"})
    (root / "tokenizer").mkdir(exist_ok=True)
    files = {"export.safetensors": sha((root / "export.safetensors").read_bytes())}
    for i, rel in enumerate(("tokenizer/tokenizer.model", "tokenizer/vocab.txt", "tokenizer/tokenizer.vocab")):
        (root / rel).write_bytes(f"tok{i}".encode())
        files[rel] = sha(f"tok{i}".encode())
    manifest = {
        "format": "parakeet-binary-v1", "file": "export.safetensors", "sha256": files["export.safetensors"],
        "files": files,
        "config": {"target": MODEL_TARGET, "train_ds": {"x": 1},
                   "tokenizer": {"dir": "nemo:", "model_path": "nemo:a_tokenizer.model",
                                 "vocab_path": "nemo:b_vocab.txt", "spe_tokenizer_vocab": "nemo:c_tokenizer.vocab"}},
        "quantized_layers": {n: {"kind": k, "shape": list(s), "bias": False} for n, (k, s) in layers.items()},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# ----------------------------------------------------------------------------- loader

class UnpackTest(unittest.TestCase):
    def test_matches_repository_unpack_and_numpy(self) -> None:
        g = torch.Generator().manual_seed(1)
        for cols in (1, 7, 8, 9, 15, 64, 1023, 1024, 4097):
            with self.subTest(cols=cols):
                codes = (2 * torch.randint(0, 2, (5, cols), generator=g) - 1).to(torch.int8)
                packed = binary_export.pack_codes(codes)
                ours = load_binary.unpack_codes(packed, cols)
                self.assertEqual(ours.dtype, torch.int8)
                self.assertTrue(torch.equal(ours, codes))
                self.assertTrue(torch.equal(ours, binary_export.unpack_codes(packed, cols)))
                bits = np.unpackbits(packed.numpy(), axis=1, bitorder="little")[:, :cols]
                self.assertTrue(np.array_equal(2 * bits.astype(np.int8) - 1, codes.numpy()))

    def test_rejects_bad_packs(self) -> None:
        good = binary_export.pack_codes(torch.ones(2, 9, dtype=torch.int8))
        tail = good.clone()
        tail[0, 1] |= 0b10  # bit 9 of row 0: beyond in_features
        for packed, cols in ((tail, 9), (good, 17), (good.to(torch.int16), 9), (good[0], 9)):
            with self.subTest(shape=tuple(packed.shape), dtype=packed.dtype, cols=cols), self.assertRaises(ValueError):
                load_binary.unpack_codes(packed, cols)


class LoaderRebuildTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "export"
        self.manifest = make_export(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_state_dict_identical_to_repository_loader(self) -> None:
        with fake_nemo():
            ours = load_binary.load(self.root, device="cpu")
            ref = binary_export.load_export(self.root, device="cpu")
        a, b = ours.state_dict(), ref.state_dict()
        self.assertEqual(list(a), list(b))
        for key in a:
            with self.subTest(key=key):
                self.assertEqual(a[key].dtype, b[key].dtype)
                self.assertEqual(a[key].shape, b[key].shape)
                self.assertTrue(torch.equal(a[key], b[key]))
        self.assertFalse(ours.training)
        self.assertTrue(all(v.dtype == torch.float32 for v in a.values() if v.is_floating_point()))
        self.assertEqual(a["encoder.layers.0.conv.pointwise_conv1.weight"].shape, (4, 9, 1))
        self.assertEqual(a["encoder.layers.0.conv.batch_norm.num_batches_tracked"].dtype, torch.int64)
        w = a["encoder.layers.0.feed_forward1.linear1.weight"]
        self.assertTrue(torch.equal(w.abs(), w.abs().amax(dim=1, keepdim=True).expand_as(w)))  # +-scale only
        self.assertEqual(ours.cfg.tokenizer.model_path, str(self.root.resolve() / "tokenizer" / "tokenizer.model"))
        self.assertIsNone(OmegaConf.to_container(ours.cfg)["train_ds"])

    def test_refuses_hash_mismatch(self) -> None:
        (self.root / "tokenizer" / "vocab.txt").write_bytes(b"tampered")
        with fake_nemo(), self.assertRaisesRegex(ValueError, "SHA-256"):
            load_binary.load(self.root)

    def test_refuses_a_consistently_renamed_weights_file(self) -> None:
        """The weights file name is fixed (plans/release.md section 2; Codex release review r2)."""
        manifest = copy.deepcopy(self.manifest)
        (self.root / "export.safetensors").rename(self.root / "renamed.safetensors")
        manifest["file"] = "renamed.safetensors"
        manifest["files"]["renamed.safetensors"] = manifest["files"].pop("export.safetensors")
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        with fake_nemo(), self.assertRaisesRegex(ValueError, "export.safetensors"):
            load_binary.load(self.root)

    def test_refuses_other_file_sets_and_formats(self) -> None:
        cases = {
            "extra file": lambda m: m["files"].update({"other.bin": "0" * 64}),
            "missing tokenizer": lambda m: m["files"].pop("tokenizer/vocab.txt"),
            "sha256 != files entry": lambda m: m.update(sha256="0" * 64),
            "format": lambda m: m.update(format="parakeet-ternary-v1"),
            "kind": lambda m: next(iter(m["quantized_layers"].values())).update(kind="conv2d"),
        }
        for label, edit in cases.items():
            with self.subTest(label):
                manifest = copy.deepcopy(self.manifest)
                edit(manifest)
                (self.root / "manifest.json").write_text(json.dumps(manifest))
                with fake_nemo(), self.assertRaises(ValueError):
                    load_binary.load(self.root)


class LoaderSourceTest(unittest.TestCase):
    def test_import_allow_list(self) -> None:
        names = stage.loader_imports((HF / "load_binary.py").read_text())
        self.assertTrue(names)
        self.assertLessEqual(names, stage.LOADER_IMPORTS)
        self.assertLessEqual(stage.LOADER_IMPORTS - set(sys.stdlib_module_names),
                             {"torch", "numpy", "safetensors", "soundfile", "nemo", "omegaconf"})

    def test_allow_list_catches_forbidden_imports(self) -> None:
        for source in ("import binary_export", "from paths import ARTIFACTS", "import requests",
                       "def f():\n    import evaluate", "from . import x"):
            with self.subTest(source=source):
                self.assertFalse(stage.loader_imports(source) <= stage.LOADER_IMPORTS)

    def test_file_checks_pass_on_the_kit(self) -> None:
        checks: dict = {}
        stage.file_checks(checks, SOURCES)
        self.assertEqual(set(checks), stage.FILE_CHECKS)
        self.assertTrue(all(checks.values()), checks)

    def test_notice_check_needs_every_source_revision(self) -> None:
        sources = copy.deepcopy(SOURCES)
        sources["sources"]["ami"]["revision"] = "f" * 40
        checks: dict = {}
        stage.file_checks(checks, sources)
        self.assertFalse(checks[stage.NOTICE_CHECK])

    def test_license_is_kea_t_license(self) -> None:
        data = (HF / "LICENSE").read_bytes()
        self.assertEqual(data, (TERNARY_HF / "LICENSE").read_bytes())
        self.assertEqual(sha(data), stage.LICENSE_SHA256)


# ----------------------------------------------------------------------------- card

# Values of the real run files as of 2026-10-21 (summary.json, manifest.json, sources.json);
# the staging run reads the files themselves.
SUMMARY = {"steps": 250000, "ramp_steps": 62500, "lr": 0.0002, "seed": 20261017, "batch_seconds": 600.0,
           "started_utc": "2026-10-17T15:46:12Z", "finished_utc": "2026-10-20T12:23:12Z",
           "audio_hours_seen": 33246.28231562482,
           "seconds_by_source": {"yodas": 79341827.6, "ami": 5409532.5, "librispeech": 11460230.6,
                                 "voxpopuli": 11342892.9, "peoples_speech": 12132132.8}}
MANIFEST = {"bytes": {"file_bytes": 105299804}, "versions": {"nemo": "3.0.0", "torch": "2.11.0+cu128"}}
SOURCES = {"seed": 20261017, "filters": {"duration_seconds": [1.0, 30.0]}, "sources": {
    "yodas": {"repo": "espnet/yodas-granary", "revision": "969944574ea3f37890beaf67ea651e160cfaf043"},
    "librispeech": {"repo": "openslr/librispeech_asr", "revision": "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"},
    "peoples_speech": {"repo": "MLCommons/peoples_speech", "revision": "f10597c5d3d3a63f8b6827701297c3afdf178272"},
    "voxpopuli": {"repo": "facebook/voxpopuli", "revision": "42f01879c780b4a2e90ec0b4f616c2ece526e4f1"},
    "ami": {"repo": "edinburghcstr/ami", "revision": "46f28f2503e2ec48f8867a84eef356c70476beab"}}}
REPO_ID = "rajb3/kea-b"


# The card as released on 2026-10-21 (Hub commit 5e7c298). hf/MODEL_CARD.md has since been updated by the
# docs-only card update (plans/card-update-x2.md), whose contract is tests/test_kea_b_card.py; these tests
# keep checking the released card that hf/stage.py staged.
RELEASE_CARD_COMMIT = "ca4bc02a7f8d11b2710b309336433548082ea3de"


def release_card_template() -> str:
    return subprocess.run(["git", "-C", str(paths.REPO), "show",
                           f"{RELEASE_CARD_COMMIT}:finetune/parakeet-binary/hf/MODEL_CARD.md"],
                          check=True, capture_output=True, text=True).stdout


class CardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.template = release_card_template()
        self.test = json.loads((PKG / "results" / "test.json").read_text())
        self.test_md = (PKG / "results" / "TEST.md").read_text()
        self.facts = stage.card_facts(self.test, SUMMARY, MANIFEST, SOURCES)

    def run_checks(self, card: str) -> dict:
        checks: dict = {}
        stage.card_checks(card, REPO_ID, self.test, self.test_md, self.facts, checks, {})
        self.assertEqual(set(checks), stage.CARD_CHECKS)
        return checks

    def test_render_replaces_placeholders(self) -> None:
        self.assertTrue(all(p in self.template for p in stage.PLACEHOLDERS))
        card = stage.render_card(self.template, REPO_ID)
        self.assertFalse(any(p in card for p in stage.PLACEHOLDERS))
        self.assertIn("hf download rajb3/kea-b", card)
        self.assertIn('name: "kea-b"', card)

    def test_real_card_passes(self) -> None:
        checks = self.run_checks(stage.render_card(self.template, REPO_ID))
        self.assertTrue(all(checks.values()), {k: v for k, v in checks.items() if not v})

    def test_card_is_short(self) -> None:
        body = self.template.split("\n---\n", 1)[1].strip("\n").splitlines()
        self.assertGreaterEqual(len(body), 80)
        self.assertLessEqual(len(body), 120)
        tables = [line for line in body if line.startswith("| ---")]
        self.assertEqual(len(tables), 1, "one results table")

    def test_facts_from_sources(self) -> None:
        strings = set(self.facts["strings"].values())
        for want in ("105.3 MB", "2,472 MB", "180.8 MB", "6.45%", "6.84%", "7.05%", "8.50%", "13.42%",
                     "105,299,804", "250,000", "62,500", "2e-4", "68.6 hours", "33,246 hours",
                     "NeMo 3.0.0", "torch 2.11.0+cu128", "up to 600 s of audio", "1 to 30 s"):
            self.assertIn(want, strings)
        self.assertEqual(self.facts["datasets"], {s["repo"] for s in SOURCES["sources"].values()})

    def test_negative_cases(self) -> None:
        card = stage.render_card(self.template, REPO_ID)
        mb1_row = "| 2.20 | 4.56 | 10.65 | 12.17 | 10.46 | 3.01 | 6.30 | 7.05 | 13.42 |"
        self.assertIn(mb1_row, card)
        cases = {
            "card has no unrendered placeholder": card + "\n{repo_id}\n",
            "card results table: exactly the TEST.md B0, M1 and MB1 rows, in TEST.md column order":
                card.replace(mb1_row, mb1_row.replace("4.56", "4.55")),
            "card model-index: one entry per test set, value == round(test.json MB1, 2)":
                card.replace("value: 13.42", "value: 13.4", 1),
            "card YAML datasets == MB1's training sources": card.replace("- edinburghcstr/ami\n", "", 1),
            "card numbers == their sources": card.replace("105.3 MB", "105 MB"),
            "card YAML: model_name, license cc-by-4.0, base_model, library_name nemo":
                card.replace("license: cc-by-4.0", "license: mit", 1),
        }
        for name, bad in cases.items():
            with self.subTest(name):
                self.assertNotEqual(bad, card)
                checks = self.run_checks(bad)
                self.assertFalse(checks[name])

    def test_extra_table_row_fails(self) -> None:
        card = stage.render_card(self.template, REPO_ID)
        row = next(line for line in card.splitlines() if line.startswith("| Original"))
        checks = self.run_checks(card.replace(row, row + "\n" + row))
        self.assertFalse(checks["card results table: exactly the TEST.md B0, M1 and MB1 rows, in TEST.md column order"])


# ----------------------------------------------------------------------------- provenance logic

def consistent_docs(root: Path) -> dict:
    """Synthetic, mutually consistent inputs of stage.provenance_from."""
    test = json.loads((PKG / "results" / "test.json").read_text())
    test_md = (PKG / "results" / "TEST.md").read_text()
    weights = stage.WEIGHTS_SHA256
    tok = {rel: f"tok-{rel}".encode() for rel in stage.TOKENIZER_RELS}
    files = {"export.safetensors": weights, **{rel: sha(b) for rel, b in tok.items()}}
    layers = {f"encoder.layers.{i}.m{j}": {"kind": "linear" if j % 2 else "pointwise_conv1d",
                                           "shape": [4, 4], "bias": False} for i in range(24) for j in range(11)}
    histogram = {"minus_one": 0.5000592884090211, "zero": 0.0, "plus_one": 0.49994071159097886,
                 "per_family": {"feed_forward": {"minus_one": 0.5, "zero": 0.0, "plus_one": 0.5}}}
    extra = {"run_name": stage.RUN_NAME, "arm": "P2", "recipe": "P2", "lr": 0.0002, "select": "final",
             "selected_step": 250000, "config_sha256": "c" * 64}
    histogram["per_module"] = {name: {"minus_one": 0.5, "zero": 0.0, "plus_one": 0.5} for name in layers}
    nbytes = {"file_bytes": stage.FILE_BYTES, "packed_code_bytes": 1}
    manifest = {"format": "parakeet-binary-v1", "file": "export.safetensors", "sha256": weights, "files": files,
                "base_model": {"id": paths.MODEL_ID, "revision": paths.MODEL_REVISION, "license": "CC-BY-4.0"},
                "config": {"target": MODEL_TARGET, "x": 1}, "quantized_layers": layers,
                "code_histogram": histogram, "parameter_accounting": {"binary_parameters": 1},
                "bytes": nbytes, "extra": extra}
    recon = {"codes_exact": True, "scales_exact": True, "greedy_hyps_equal": True, "checked_layers": 264}
    recorded_histogram = {k: v for k, v in histogram.items() if k != "per_module"}  # as binary_train records it
    summary = copy.deepcopy({**extra, "export_dir": str(root), "export_bytes": nbytes,
                             "code_histogram": recorded_histogram,
                             "reconstruction": recon})
    recorded = copy.deepcopy({k: manifest[k] for k in stage.SCORING_MANIFEST_KEYS})
    base = dict(binary_test_paths.BASE_MODEL)
    nemo = base["nemo_sha256"]
    ls_sha = "l" * 64
    scoring = {"source": {"kind": "export", "sha256": weights, "path": str(root), "export_manifest": recorded,
                          "base_model": base},
               "sets": {name: {"wer": test["rows"]["MB1"][name], "limit": None} for name in paths.TEST_SETS},
               "mean_wer": test["rows"]["MB1"]["mean"], "attempt": 1, "per_set_sha256": {"librispeech_clean": ls_sha}}
    attempts = ([{"attempt": 1, "state": "started"}]
                + [{"attempt": 1, "state": "set_done", "set": n, "sha256": ls_sha if n == "librispeech_clean" else "s"}
                   for n in paths.TEST_SETS] + [{"attempt": 1, "state": "complete"}])
    digests = {"export.safetensors": weights, **files, "manifest.json": stage.MANIFEST_SHA256,
               "reconstruction.json": stage.RECONSTRUCTION_SHA256}
    return {"export_dir": root, "manifest": manifest, "recon": recon, "summary": summary,
            "precheck": {"pass": True, "weights_sha256": weights, "checks": [{"ok": True}]},
            "test": test, "test_md": test_md, "scoring": scoring, "attempts": attempts,
            "entries": set(binary_test_paths.EXPORT_ENTRIES), "digests": digests, "export_size": stage.FILE_BYTES,
            "ls_clean_sha256": ls_sha, "lock": {"revision": paths.MODEL_REVISION,
                                                "files": {paths.MODEL_FILE.name: nemo}},
            "nemo_sha256": nemo, "nemo_config": copy.deepcopy(manifest["config"]), "nemo_tokenizer": dict(tok),
            "tokenizer_bytes": dict(tok)}


class ProvenanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.docs = consistent_docs(Path("/tmp/kea-b-export"))

    def run_checks(self, docs: dict) -> dict:
        checks: dict = {}
        stage.provenance_from(docs, checks, {})
        self.assertEqual(set(checks), stage.PROVENANCE_CHECKS)
        return checks

    def test_consistent_documents_pass(self) -> None:
        checks = self.run_checks(self.docs)
        self.assertTrue(all(checks.values()), [k for k, v in checks.items() if not v])

    def test_pins_agree_with_test_scoring_pins(self) -> None:
        pinned = binary_test_paths.PINNED_FILES
        self.assertEqual(stage.WEIGHTS_SHA256, pinned["export/export.safetensors"])
        self.assertEqual(stage.MANIFEST_SHA256, pinned["export/manifest.json"])
        self.assertEqual(stage.RECONSTRUCTION_SHA256, pinned["export/reconstruction.json"])
        self.assertEqual(stage.RUN_NAME, binary_test_paths.RUN_NAME)

    def test_each_tamper_fails_its_check(self) -> None:
        def mutate(path, value):
            def edit(d):
                target = d
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
            return edit

        cases = {
            "manifest format is parakeet-binary-v1 and file is export.safetensors":
                mutate(("manifest", "format"), "parakeet-ternary-v1"),
            "manifest identity is the MB1 final export (run, select, step, lr)":
                mutate(("manifest", "extra", "selected_step"), 237500),
            "export.safetensors is 105,299,804 bytes (file and manifest)": mutate(("export_size",), 1),
            "export SHA-256 == manifest sha256 == manifest files[export.safetensors]":
                mutate(("manifest", "sha256"), "0" * 64),
            "export SHA-256 == pinned MB1 weights == binary_test_paths.WEIGHTS_SHA256":
                mutate(("digests", "export.safetensors"), "0" * 64),
            "export directory holds exactly the pinned entries":
                lambda d: d["entries"].add("extra.bin"),
            "precheck passed (pass true, every check ok) and recorded these weights":
                mutate(("precheck", "checks"), [{"ok": True}, {"ok": False}]),
            "export SHA-256 == results/test.json mb1_weights_sha256":
                mutate(("test", "mb1_weights_sha256"), "0" * 64),
            "test scoring: attempt 1 complete, all 8 sets, source is this export":
                lambda d: d["attempts"].pop(),
            "test scoring librispeech_clean.json SHA-256 == its set_done record and summary per_set_sha256":
                mutate(("ls_clean_sha256",), "0" * 64),
            "manifest format/file/sha256/files/extra/accounting/bytes == those recorded by the test scoring":
                mutate(("scoring", "source", "export_manifest", "extra", "lr"), 0.0005),
            "results/test.json MB1 row == test scoring summary (8 sets and mean)":
                mutate(("scoring", "mean_wer"), 7.0),
            "results/TEST.md B0, M1 and MB1 rows == results/test.json rounded to 2 decimals":
                lambda d: d.update(test_md=d["test_md"].replace("| MB1 | 2.20 |", "| MB1 | 2.21 |")),
            "manifest identity and config_sha256 == run summary.json":
                mutate(("summary", "config_sha256"), "d" * 64),
            "run summary export_dir, export_bytes and code histogram == manifest":
                mutate(("summary", "export_dir"), "/elsewhere"),
            "code histogram is binary (no zero codes; -1 and +1 fractions sum to 1)":
                lambda d: next(iter(d["manifest"]["code_histogram"]["per_module"].values())).update(zero=0.1),
            "manifest quantized layers: 264, kinds linear or pointwise_conv1d, no bias":
                lambda d: next(iter(d["manifest"]["quantized_layers"].values())).update(bias=True),
            "manifest lists exactly export.safetensors and the three tokenizer files":
                lambda d: d["manifest"]["files"].update({"x": "0"}),
            "tokenizer files match the manifest SHA-256":
                mutate(("digests", "tokenizer/vocab.txt"), "0" * 64),
            "manifest base_model is the pinned base model": mutate(("manifest", "base_model", "revision"), "main"),
            "base .nemo SHA-256 == lock.json == test scoring base_model": mutate(("nemo_sha256",), "0" * 64),
            "manifest config == base .nemo model_config.yaml (resolved)": mutate(("nemo_config",), {"x": 2}),
            "tokenizer files == base .nemo tokenizer members":
                mutate(("nemo_tokenizer", "tokenizer/vocab.txt"), b"other"),
            "reconstruction.json == run summary reconstruction": mutate(("summary", "reconstruction"), {}),
            "reconstruction exact: codes, scales, greedy hyps, all 264 layers":
                lambda d: (d["recon"].update(greedy_hyps_equal=False), d["summary"].update(reconstruction=d["recon"])),
            "manifest.json SHA-256 == pinned == binary_test_paths": mutate(("digests", "manifest.json"), "0" * 64),
            "reconstruction.json SHA-256 == pinned == binary_test_paths":
                mutate(("digests", "reconstruction.json"), "0" * 64),
        }
        self.assertEqual(set(cases), stage.PROVENANCE_CHECKS)
        for name, edit in cases.items():
            with self.subTest(name):
                docs = copy.deepcopy(self.docs)
                edit(docs)
                self.assertFalse(self.run_checks(docs)[name])


# ----------------------------------------------------------------------------- check names, driver

class RequiredChecksTest(unittest.TestCase):
    def test_partition_and_count(self) -> None:
        required = stage.required_checks()
        parts = (stage.PROVENANCE_CHECKS, stage.FILE_CHECKS, stage.CARD_CHECKS, stage.staging_checks(),
                 stage.STANDALONE_CHECKS)
        self.assertEqual(sum(len(p) for p in parts), len(required))
        self.assertEqual(set().union(*parts), required)
        self.assertEqual(len(stage.PROVENANCE_CHECKS), 27)
        self.assertEqual(len(stage.FILE_CHECKS), 4)
        self.assertEqual(len(stage.CARD_CHECKS), 6)
        self.assertEqual(len(stage.staging_checks()), 1 + len(stage.STAGED_FILES))
        self.assertEqual(len(stage.STANDALONE_CHECKS), 5 + 2 * len(stage.CHECK_IDS))
        self.assertEqual(len(required), 61)

    def test_staged_files_and_clips(self) -> None:
        self.assertEqual(len(stage.STAGED_FILES), 10)
        self.assertEqual(set(stage.STAGED_FILES), {
            "export.safetensors", "manifest.json", "reconstruction.json", "tokenizer/tokenizer.model",
            "tokenizer/tokenizer.vocab", "tokenizer/vocab.txt", "README.md", "load_binary.py", "LICENSE", "NOTICE"})
        self.assertEqual(stage.CHECK_IDS, ("1089-134686-0000", "1089-134686-0001", "1089-134686-0002",
                                           "1089-134686-0003"))
        for i in stage.CHECK_IDS:
            self.assertIn(f"clip {i}: staged transcript == test-scoring hypothesis", stage.required_checks())

    def test_layout_constants_equal_binary_export(self) -> None:
        self.assertEqual(stage.FORMAT, binary_export.FORMAT)
        self.assertEqual(stage.EXPORT_FILE, binary_export.FILE)
        self.assertEqual(stage.MANIFEST, binary_export.MANIFEST)
        self.assertEqual(stage.TOKENIZER_DIR, binary_export.TOKENIZER_DIR)
        self.assertEqual(stage.TOKENIZER_FILES, binary_export.TOKENIZER_FILES)
        self.assertEqual(load_binary.FORMAT, binary_export.FORMAT)
        self.assertEqual(load_binary.TOKENIZER_FILES, binary_export.TOKENIZER_FILES)

    def test_stage_import_does_not_import_torch(self) -> None:
        code = ("import importlib.util, sys; spec = importlib.util.spec_from_file_location('s', sys.argv[1]); "
                "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
                "sys.exit(1 if 'torch' in sys.modules else 0)")
        for script in ("stage.py", "verify_download.py"):
            with self.subTest(script):
                done = subprocess.run([sys.executable, "-c", code, str(HF / script)], capture_output=True)
                self.assertEqual(done.returncode, 0, done.stderr.decode()[-500:])

    def test_staging_root_is_the_binary_artifacts(self) -> None:
        self.assertEqual(stage.STAGING, Path("/mnt/hd/wilderness-labs-stt/parakeet-binary/hf-staging"))


class StageDriverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.staging = base / "staging"
        self.export = base / "export"
        self.export.mkdir()
        self.sources = {}
        for rel in stage.STAGED_FILES:
            if rel == "README.md":
                continue
            path = (self.export if rel in stage.EXPORT_RELS else base / "hf") / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"content of {rel}".encode())
            self.sources[rel] = path
        self.final = self.staging / "rajb3__kea-b"
        self.ok_report = self.staging / "rajb3__kea-b.stage.json"
        self.failed_report = self.staging / "rajb3__kea-b.stage.FAILED.json"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def prepare(self, fail: str | None = None):
        def run(export_dir, checks, info):
            for name in stage.PROVENANCE_CHECKS | stage.FILE_CHECKS | stage.CARD_CHECKS:
                checks[name] = name != fail
            return "# card\n", dict(self.sources)
        return run

    def standalone(self, fail: str | None = None, error: BaseException | None = None):
        def run(folder, export_dir, checks, info):
            self.assertTrue(folder.name.startswith(".rajb3__kea-b.tmp-"))
            self.assertEqual(sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()),
                             sorted(stage.STAGED_FILES))
            if error is not None:
                raise error
            for name in stage.STANDALONE_CHECKS:
                checks[name] = name != fail
        return run

    def leftovers(self) -> list[str]:
        return sorted(p.name for p in self.staging.iterdir() if p.name.startswith("."))

    def test_success_promotes_atomically(self) -> None:
        ok, report, path = stage.stage(REPO_ID, self.export, self.staging, self.prepare(), self.standalone())
        self.assertTrue(ok)
        self.assertEqual(path, self.ok_report)
        self.assertTrue(self.final.is_dir())
        self.assertFalse(self.failed_report.exists())
        self.assertEqual(self.leftovers(), [])
        saved = json.loads(self.ok_report.read_text())
        self.assertEqual(saved["status"], "PASSED")
        self.assertEqual(saved["checks_passed"], "61/61")
        self.assertEqual(set(saved["files"]), set(stage.STAGED_FILES))
        self.assertEqual(saved["files"]["README.md"]["sha256"], sha(b"# card\n"))
        self.assertEqual((self.final / "NOTICE").read_bytes(), b"content of NOTICE")

    def assert_failed(self, ok: bool, path: Path) -> dict:
        self.assertFalse(ok)
        self.assertEqual(path, self.failed_report)
        self.assertFalse(self.final.exists())
        self.assertFalse(self.ok_report.exists())
        self.assertEqual(self.leftovers(), [])
        report = json.loads(self.failed_report.read_text())
        self.assertEqual(report["status"], "FAILED")
        return report

    def test_failed_check_before_staging_leaves_nothing(self) -> None:
        name = sorted(stage.CARD_CHECKS)[0]
        ok, _, path = stage.stage(REPO_ID, self.export, self.staging, self.prepare(fail=name), self.standalone())
        report = self.assert_failed(ok, path)
        self.assertEqual(report["failed_checks"], [name])

    def test_failed_standalone_check_leaves_nothing(self) -> None:
        name = f"clip {stage.CHECK_IDS[2]}: staged transcript == test-scoring hypothesis"
        ok, _, path = stage.stage(REPO_ID, self.export, self.staging, self.prepare(), self.standalone(fail=name))
        self.assertEqual(self.assert_failed(ok, path)["failed_checks"], [name])

    def test_exception_and_exit_leave_nothing(self) -> None:
        for error in (RuntimeError("CUDA error"), SystemExit("refused"), KeyboardInterrupt()):
            with self.subTest(type(error).__name__):
                ok, report, path = stage.stage(REPO_ID, self.export, self.staging, self.prepare(),
                                               self.standalone(error=error))
                self.assert_failed(ok, path)
                self.assertIn(type(error).__name__, report["error"])

    def test_missing_or_unexpected_check_name_fails(self) -> None:
        def short(folder, export_dir, checks, info):
            for name in sorted(stage.STANDALONE_CHECKS)[1:]:
                checks[name] = True

        def extra(folder, export_dir, checks, info):
            for name in stage.STANDALONE_CHECKS:
                checks[name] = True
            checks["an unplanned check"] = True

        for run in (short, extra):
            with self.subTest(run.__name__):
                ok, report, path = stage.stage(REPO_ID, self.export, self.staging, self.prepare(), run)
                self.assertIn("check set mismatch", self.assert_failed(ok, path)["error"])

    def test_non_boolean_true_is_a_failure(self) -> None:
        def truthy(folder, export_dir, checks, info):
            for name in stage.STANDALONE_CHECKS:
                checks[name] = 1
        ok, _, path = stage.stage(REPO_ID, self.export, self.staging, self.prepare(), truthy)
        self.assert_failed(ok, path)

    def test_stale_outputs_removed_even_on_failure(self) -> None:
        self.final.mkdir(parents=True)
        (self.final / "old").write_text("x")
        self.ok_report.write_text("{}")
        (self.staging / ".rajb3__kea-b.tmp-1").mkdir()
        ok, _, path = stage.stage(REPO_ID, self.export, self.staging, self.prepare(fail=sorted(stage.FILE_CHECKS)[0]),
                                  self.standalone())
        self.assert_failed(ok, path)

    def test_extra_staged_file_fails(self) -> None:
        def prepare(export_dir, checks, info):
            card, sources = self.prepare()(export_dir, checks, info)
            extra = self.export / "extra.txt"
            extra.write_text("x")
            return card, {**sources, "extra.txt": extra}
        ok, _, path = stage.stage(REPO_ID, self.export, self.staging, prepare, lambda *a: None)
        self.assert_failed(ok, path)

    def test_refuses_outside_a_systemd_unit(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "INVOCATION_ID"}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(stage.paths, "require_mount", side_effect=AssertionError("wrote")), \
                self.assertRaises(SystemExit) as raised:
            stage.main([REPO_ID])
        self.assertIn("./heavy", str(raised.exception.code))

    def test_failed_runtime_preflight_invalidates_a_previous_success(self) -> None:
        """A restage that fails before any check (CUDA unavailable, runtime import error) must still remove the
        previous promoted folder and PASSED report and record a FAILED report (Codex release review r1, item 1)."""
        no_cuda = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        failures = {"CUDA unavailable": mock.Mock(return_value=no_cuda),
                    "runtime import error": mock.Mock(side_effect=ImportError("no module named nemo"))}
        for label, load_runtime in failures.items():
            with self.subTest(label):
                self.final.mkdir(parents=True, exist_ok=True)
                (self.final / "README.md").write_text("old card")
                self.ok_report.write_text(json.dumps({"status": "PASSED"}))
                prepare = mock.Mock(side_effect=AssertionError("checks ran without a runtime"))
                with mock.patch.dict(os.environ, {"INVOCATION_ID": "test"}), \
                        mock.patch.object(stage, "STAGING", self.staging), \
                        mock.patch.object(stage, "DEFAULT_EXPORT", self.export), \
                        mock.patch.object(stage.binary_paths, "require_mount"), \
                        mock.patch.object(stage, "load_runtime", load_runtime), \
                        mock.patch.object(stage, "make_prepare", return_value=prepare), \
                        mock.patch("sys.stdout"), mock.patch("sys.stderr"), \
                        self.assertRaises(SystemExit) as raised:
                    stage.main([REPO_ID])
                self.assertEqual(raised.exception.code, 1)
                load_runtime.assert_called_once()
                prepare.assert_not_called()
                report = self.assert_failed(False, self.failed_report)
                self.assertIn("CUDA" if label == "CUDA unavailable" else "ImportError", report["error"])

    def test_rejects_bad_repo_id(self) -> None:
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            stage.main(["kea-b"])


# ----------------------------------------------------------------------------- download verification

class VerifyDownloadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name) / "kea-b"
        (self.folder / "tokenizer").mkdir(parents=True)
        self.report = {}
        for rel in ("README.md", "tokenizer/vocab.txt"):
            data = rel.encode()
            (self.folder / rel).write_bytes(data)
            self.report[rel] = {"sha256": sha(data), "bytes": len(data)}

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_identical_download_passes_hub_extras_ignored(self) -> None:
        (self.folder / ".gitattributes").write_text("*.safetensors filter=lfs\n")
        (self.folder / ".cache" / "huggingface").mkdir(parents=True)
        (self.folder / ".cache" / "huggingface" / "x.metadata").write_text("x")
        self.assertEqual(verify.compare_listing(self.report, self.folder), [])

    def test_differences_are_reported(self) -> None:
        (self.folder / "README.md").write_text("changed")
        (self.folder / "extra.bin").write_text("x")
        (self.folder / "tokenizer" / "vocab.txt").unlink()
        problems = verify.compare_listing(self.report, self.folder)
        self.assertEqual(len(problems), 3, problems)


if __name__ == "__main__":
    unittest.main()
