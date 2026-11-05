"""Contract tests (CPU, the tiny model): the Kea-B 0.1 release kit (release-kea-b-0.1 plan), finetune/relabel/hf/.

- load_kea_b.py, the standalone loader: on a tiny keab-8bit-v1 bundle and the parakeet-binary-v1 view derived from
  it, the bundle's FP32 state == the view's == binary_export.load_export(view)'s, bit for bit, also from release-form
  manifests; the format comes from the manifest; unknown formats, hash mismatches and missing files raise;
- release-form manifests: only machine paths under "extra" are removed and reported;
- scrubbing: machine paths, the host's name and private registry names are found in staged files; clean files pass;
- approval: kit.approval_problems and upload.upload refuse a wrong, missing or stale digest before any Hub call, and
  an accepted upload creates the repository private before uploading;
- documents: the templates render from the computed values with no placeholder left, and the card's YAML,
  model-index and table are the computed ones;
- verify's listing comparison: missing, unexpected and changed files are reported, Hub extras are ignored;
- provenance: unmapped or non-permissive training sources fail; a clean record maps every alias.
The real files are checked by test_hf_release_real.py (opt-in) and by the staging run itself.
"""
from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

import relabel_paths
from hf import kit
from hf import load_kea_b as lk
from hf import provenance as pv
from hf import stage
from hf import upload as up
from test_keab_export import training_model
from training import keab_export as ke

import binary_export  # noqa: E402  (training.eightbit puts finetune/parakeet-binary on sys.path)

HF = Path(stage.__file__).resolve().parent


class Tmp:
    def __init__(self, prefix: str):
        relabel_paths.require_mount()
        base = relabel_paths.ARTIFACTS / "tmp"
        base.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix=prefix, dir=base))

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


def states_equal(a: dict, b: dict) -> list[str]:
    if list(a) != list(b):
        return ["keys differ"]
    return [k for k in a if a[k].dtype != b[k].dtype or a[k].shape != b[k].shape or not torch.equal(a[k], b[k])]


class Loader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Tmp("hf-release-test-")
        cls.bundle, cls.view = cls.tmp.root / "bundle", cls.tmp.root / "view"
        ke.write_bundle(training_model("S2"), cls.bundle, root=cls.tmp.root,
                        extra={"run": "tiny", "teacher": {"name": "t", "path": "/mnt/hd/somewhere/model.nemo"}})
        ke.derive_view(cls.bundle, cls.view, root=cls.tmp.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.close()

    def copy(self, src: Path, name: str) -> Path:
        dst = self.tmp.root / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        return dst

    def test_format_comes_from_the_manifest(self):
        self.assertEqual(lk.detect_format(self.bundle), "keab-8bit-v1")
        self.assertEqual(lk.detect_format(self.view), "parakeet-binary-v1")

    def test_bundle_state_equals_view_state_equals_load_export_bit_for_bit(self):
        _, bundle_state = lk.load_state(self.bundle)
        _, view_state = lk.load_state(self.view)
        self.assertEqual(sorted(bundle_state), sorted(view_state))
        self.assertEqual(states_equal(dict(sorted(bundle_state.items())), dict(sorted(view_state.items()))), [])
        bundle_model = lk.load(self.bundle)
        view_model = lk.load(self.view)
        reference = binary_export.load_export(self.view, "cpu")
        a = bundle_model.state_dict()
        self.assertEqual(states_equal(a, view_model.state_dict()), [])
        self.assertEqual(states_equal(a, reference.state_dict()), [])
        self.assertTrue(all(v.dtype == torch.float32 for v in a.values() if v.is_floating_point()))
        self.assertFalse(bundle_model.training)

    def test_release_form_manifests_load_to_the_same_state(self):
        reference = lk.load(self.bundle).state_dict()
        for src, name in ((self.bundle, "rel-bundle"), (self.view, "rel-view")):
            d = self.copy(src, name)
            manifest = json.loads((d / "manifest.json").read_text())
            form, removed = kit.release_form(manifest)
            self.assertTrue(removed)
            (d / "manifest.json").write_text(kit.manifest_text(form))
            self.assertNotIn("/mnt/hd/somewhere", (d / "manifest.json").read_text())
            self.assertTrue(all(k.startswith("extra.") for k in removed))
            self.assertEqual(states_equal(lk.load(d).state_dict(), reference), [], name)

    def test_unknown_format_raises(self):
        d = self.copy(self.bundle, "unknown")
        manifest = json.loads((d / "manifest.json").read_text())
        manifest["format"] = "keab-8bit-v2"
        (d / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            lk.detect_format(d)
        with self.assertRaises(ValueError):
            lk.load(d)

    def test_hash_mismatch_raises(self):
        for src, rel in ((self.bundle, "bundle.safetensors"), (self.view, "export.safetensors"),
                         (self.bundle, "tokenizer/vocab.txt")):
            d = self.copy(src, "damaged")
            data = bytearray((d / rel).read_bytes())
            data[-1] ^= 1
            (d / rel).write_bytes(bytes(data))
            with self.subTest(rel=rel), self.assertRaises(ValueError):
                lk.load_state(d)

    def test_missing_files_raise(self):
        d = self.copy(self.bundle, "missing")
        (d / "tokenizer" / "tokenizer.model").unlink()
        with self.assertRaises(FileNotFoundError):
            lk.load_state(d)
        d = self.copy(self.bundle, "nomanifest")
        (d / "manifest.json").unlink()
        with self.assertRaises(FileNotFoundError):
            lk.load_state(d)

    def test_manifest_file_list_must_be_exact(self):
        d = self.copy(self.view, "extra-file")
        manifest = json.loads((d / "manifest.json").read_text())
        manifest["files"]["extra.bin"] = manifest["sha256"]
        (d / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            lk.load_state(d)


class ReleaseForm(unittest.TestCase):
    def test_only_machine_paths_under_extra_are_removed(self):
        manifest = {"format": "keab-8bit-v1", "config": {"tokenizer": {"dir": "/lustre/upstream/path"}},
                    "extra": {"run": "b100-main", "teacher": {"name": "B0p", "path": "/mnt/hd/x/model.nemo",
                                                              "sha256": "ab"},
                              "keab": {"bundle": "/home/u/bundle", "scope": "S2"}}}
        form, removed = kit.release_form(manifest)
        self.assertEqual(sorted(removed), ["extra.keab.bundle", "extra.teacher.path"])
        self.assertEqual(form["extra"], {"run": "b100-main", "teacher": {"name": "B0p", "sha256": "ab"},
                                         "keab": {"scope": "S2"}})
        self.assertEqual(form["config"], manifest["config"])          # outside extra: unchanged
        self.assertEqual(manifest["extra"]["teacher"]["path"], "/mnt/hd/x/model.nemo")   # input untouched


class Scrub(unittest.TestCase):
    REGISTRY = {
        "train-a": {"name": "Trained Corpus A", "license": "CC0", "url": "https://example.org/a"},
        "heldout-x": {"name": "SecretSet42 (someorg/secret-set), English config", "license": "CC BY-NC-SA",
                      "hf_id": "someorg/secret-set", "role": "eval"},
        "lectures-z": {"name": "Big University OpenLectures via an archive mirror", "license": "CC BY-NC-SA"},
    }

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hf-scrub-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_terms_cover_names_aliases_and_ids_of_sources_not_trained_on(self):
        terms = kit.scrub_terms(self.REGISTRY, {"train-a"}, ["extra-term"])
        for t in ("heldout-x", "someorg/secret-set", "SecretSet42", "lectures-z", "OpenLectures", "extra-term",
                  "eval-ext", "eval-stem", "ted-test-text"):
            self.assertIn(t, terms)
        self.assertNotIn("train-a", terms)
        self.assertNotIn("Trained Corpus A", terms)

    def test_findings(self):
        terms = kit.scrub_terms(self.REGISTRY, {"train-a"})
        (self.tmp / "README.md").write_text("Trained on Trained Corpus A.\n")
        (self.tmp / "sub").mkdir()
        (self.tmp / "sub" / "manifest.json").write_text(json.dumps({"extra": {"p": "/mnt/hd/x"}}))
        (self.tmp / "NOTICE").write_text("We also scored on secretset42 and on eval-stem.\n")
        (self.tmp / "TRAINING.md").write_text("built on host build-box-7\n")
        (self.tmp / ".gitattributes").write_text("/mnt/ ignored\n")
        found = kit.scrub_findings(self.tmp, terms, ["build-box-7"])
        got = {(f["file"], f["term"]) for f in found}
        self.assertIn(("sub/manifest.json", "/mnt/"), got)
        self.assertIn(("NOTICE", "SecretSet42"), got)
        self.assertIn(("NOTICE", "eval-stem"), got)
        self.assertIn(("TRAINING.md", "hostname"), got)
        self.assertNotIn("README.md", {f["file"] for f in found})
        self.assertNotIn(".gitattributes", {f["file"] for f in found})
        for p in ("NOTICE", "TRAINING.md", "sub/manifest.json"):
            (self.tmp / p).unlink()
        self.assertEqual(kit.scrub_findings(self.tmp, terms, ["build-box-7"]), [])

    def test_image_metadata_is_searched_for_every_kind_of_term(self):
        (self.tmp / "assets").mkdir()
        (self.tmp / "assets" / "chart.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00tEXtComment\x00secret-corpus on build-box-7")
        found = sorted(f["term"] for f in kit.scrub_findings(self.tmp, ["secret-corpus"], ["build-box-7"]))
        self.assertEqual(found, ["hostname", "secret-corpus"])

    def test_safetensors_header_is_searched(self):
        from safetensors.torch import save_file
        save_file({"w": torch.zeros(2)}, self.tmp / "x.safetensors", metadata={"path": "/home/someone/x"})
        self.assertEqual([f["term"] for f in kit.scrub_findings(self.tmp, [])], ["/home/"])


class FakeApi:
    def __init__(self, private=True):
        self.calls, self.private = [], private

    def create_repo(self, repo_id, **kw):
        self.calls.append(("create_repo", repo_id, kw))

    def model_info(self, repo_id):
        self.calls.append(("model_info", repo_id))
        return type("Info", (), {"private": self.private})()

    def create_commit(self, **kw):
        ops = {op.path_in_repo: op.path_or_fileobj for op in kw["operations"]}
        self.calls.append(("create_commit", {**kw, "operations": ops}))
        return type("Commit", (), {"oid": "c0ffee", "commit_url": "https://hub/commit/c0ffee"})()


class Approval(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp(prefix="hf-approval-"))
        self.addCleanup(shutil.rmtree, self.folder, True)
        (self.folder / "README.md").write_text("card\n")
        (self.folder / "compat").mkdir()
        (self.folder / "compat" / "manifest.json").write_text("{}\n")
        files = kit.listing(self.folder)
        self.report = {"status": "PASSED", "repo_id": "org/model", "folder": str(self.folder), "files": files,
                       "bundle_sha256": kit.bundle_sha256("org/model", files)}

    def test_digest_covers_repo_and_every_file(self):
        files = self.report["files"]
        self.assertNotEqual(kit.bundle_sha256("org/other", files), self.report["bundle_sha256"])
        changed = copy.deepcopy(files)
        changed["README.md"]["bytes"] += 1
        self.assertNotEqual(kit.bundle_sha256("org/model", changed), self.report["bundle_sha256"])

    def test_refusals(self):
        good = self.report["bundle_sha256"]
        self.assertEqual(kit.approval_problems(self.report, self.folder, good), [])
        for approved in (None, "", "abc", "0" * 64):
            self.assertTrue(kit.approval_problems(self.report, self.folder, approved), approved)
        failed = {**self.report, "status": "FAILED"}
        self.assertTrue(kit.approval_problems(failed, self.folder, good))
        (self.folder / "README.md").write_text("edited card\n")
        self.assertTrue(kit.approval_problems(self.report, self.folder, good))

    def test_upload_refuses_before_any_hub_call(self):
        api = FakeApi()
        with self.assertRaises(up.UploadRefused):
            up.upload(self.report, self.folder, "0" * 64, api)
        (self.folder / "extra.txt").write_text("x")
        with self.assertRaises(up.UploadRefused):
            up.upload(self.report, self.folder, self.report["bundle_sha256"], api)
        self.assertEqual(api.calls, [])

    def test_upload_creates_private_then_uploads_and_records_the_commit(self):
        api = FakeApi()
        record = up.upload(self.report, self.folder, self.report["bundle_sha256"], api)
        self.assertEqual([c[0] for c in api.calls], ["create_repo", "model_info", "create_commit"])
        self.assertEqual(api.calls[0][2], {"repo_type": "model", "private": True, "exist_ok": False})
        self.assertEqual(api.calls[2][1]["operations"], {"README.md": b"card\n", "compat/manifest.json": b"{}\n"})
        self.assertEqual(record["commit"], "c0ffee")
        self.assertEqual(record["bundle_sha256"], self.report["bundle_sha256"])

    def test_upload_stops_if_the_repo_is_not_private(self):
        api = FakeApi(private=False)
        with self.assertRaises(up.UploadRefused):
            up.upload(self.report, self.folder, self.report["bundle_sha256"], api)
        self.assertNotIn("create_commit", [c[0] for c in api.calls])

    def test_upload_sends_exactly_the_approved_files_and_bytes(self):
        (self.folder / ".gitattributes").write_text("* filter=lfs\n")   # ignored by the listing, never uploaded
        api = FakeApi()
        up.upload(self.report, self.folder, self.report["bundle_sha256"], api)
        self.assertEqual(sorted(api.calls[2][1]["operations"]), sorted(self.report["files"]))

    def test_upload_refuses_bytes_that_differ_from_the_report(self):
        api = FakeApi()
        report = copy.deepcopy(self.report)
        report["files"]["README.md"]["sha256"] = "0" * 64
        report["bundle_sha256"] = kit.bundle_sha256("org/model", report["files"])
        with self.assertRaises(up.UploadRefused):
            up.upload(report, self.folder, report["bundle_sha256"], api)
        self.assertNotIn("create_commit", [c[0] for c in api.calls])


class Listing(unittest.TestCase):
    def setUp(self):
        self.a = Path(tempfile.mkdtemp(prefix="hf-listing-"))
        self.addCleanup(shutil.rmtree, self.a, True)
        (self.a / "x.txt").write_text("x")
        (self.a / "d").mkdir()
        (self.a / "d" / "y.bin").write_bytes(b"\0\1")
        self.files = kit.listing(self.a)

    def test_identical_with_hub_extras_ignored(self):
        (self.a / ".gitattributes").write_text("*.bin filter=lfs\n")
        (self.a / ".cache" / "huggingface").mkdir(parents=True)
        (self.a / ".cache" / "huggingface" / "meta").write_text("m")
        self.assertEqual(kit.compare_listing(self.files, self.a), [])

    def test_missing_unexpected_and_changed(self):
        (self.a / "x.txt").write_text("X")
        (self.a / "d" / "y.bin").unlink()
        (self.a / "z.md").write_text("z")
        self.assertEqual(kit.compare_listing(self.files, self.a),
                         ["missing d/y.bin", "unexpected z.md", "x.txt differs from the staged file"])


# ----------------------------------------------------------------------------- provenance and documents

def provenance_inputs():
    registry = {
        "cv-x": {"name": "Read speech X", "license": "CC0-1.0", "url": "https://example.org/cv"},
        "tts-a": {"name": "TTS A", "license": "apache-2.0", "hf_id": "org/tts-a", "revision": "r1"},
        "text-a": {"name": "LLM", "license": "apache-2.0", "hf_id": "org/llm", "revision": "r2"},
        "terms-a": {"name": "KB", "license": "CC0 1.0", "url": "https://kb.example"},
        "rir-a": {"name": "RIRs", "license": "Apache 2.0", "url": "https://rir.example"},
        "noise-a": {"name": "Noise", "license": "per clip: CC0 or CC BY; CC BY-NC excluded", "url": "https://n.example"},
        "eval-y": {"name": "EvalSet", "license": "CC BY-NC-SA", "role": "eval"},
    }
    display = {a: {"name": f"Public {a}", "license": "CC0-1.0" if a != "noise-a" else "per clip"}
               for a in ("cv-x", "tts-a", "text-a", "terms-a", "rir-a", "noise-a")}
    return {"run": "r", "seconds_by_source": {"corp": 7200.0, "corp-sub": 3600.0, "cv-x": 1800.0, "synthetic": 900.0,
                                              "unused": 0.0},
            "plan_sources": [{"config": {"name": "synthetic"}, "n_clips": 3, "k": 2},
                             {"config": {"name": "cv-x"}, "n_clips": 10, "k": 1}],
            "corpora": {"corp": {"repo": "org/corp", "revision": "abc", "license": "CC-BY-4.0"}},
            "subsets": {"corp-sub": ("corp", "medical")}, "registry": registry, "display": display,
            "synthetic": {"clips": 3, "engines": {"tts-a": 3}, "noise": {"noise-a": {
                "http://creativecommons.org/publicdomain/zero/1.0/": 5}}, "rirs": {"rir-a": 2},
                "text": ["text-a"], "terms": ["terms-a"]},
            "tools": {"label teacher": {"name": "org/asr", "hf_id": "org/asr", "revision": "r3", "license": "apache-2.0"}}}


class Provenance(unittest.TestCase):
    def test_clean_record_maps_every_trained_alias(self):
        prov = pv.derive(**provenance_inputs())
        self.assertEqual([s["alias"] for s in prov["sources"]], ["corp", "corp-sub", "cv-x", "synthetic"])
        self.assertEqual(prov["sources"][1]["repo"], "org/corp")
        self.assertEqual(pv.card_datasets(prov), ["org/corp"])
        self.assertNotIn("eval-y", prov["allowed_aliases"])
        self.assertIn("noise-a", prov["allowed_aliases"])
        self.assertAlmostEqual(prov["hours_seen_total"], 3.75)

    def test_failures(self):
        cases = []
        d = provenance_inputs(); d["seconds_by_source"]["mystery"] = 1.0; cases.append(("unmapped", d))
        d = provenance_inputs(); d["seconds_by_source"]["eval-y"] = 1.0; cases.append(("eval source", d))
        d = provenance_inputs(); d["registry"]["cv-x"]["license"] = "CC BY-NC 4.0"; cases.append(("NC", d))
        d = provenance_inputs(); d["synthetic"]["noise"]["noise-a"]["CC BY-NC 3.0"] = 1; cases.append(("noise NC", d))
        d = provenance_inputs(); d["corpora"]["corp"]["license"] = "CC BY-SA 4.0"; cases.append(("SA", d))
        d = provenance_inputs(); del d["display"]["cv-x"]; cases.append(("no public name", d))
        d = provenance_inputs(); d["synthetic"]["clips"] = 4; cases.append(("clip count", d))
        for label, inputs in cases:
            with self.subTest(label), self.assertRaises(pv.ProvenanceError):
                pv.derive(**inputs)


def fake_result_inputs():
    cols = ["librispeech_clean", "librispeech_other", "ami", "earnings22", "gigaspeech", "spgispeech", "voxpopuli",
            "mean", "common_voice"]
    arms = ["NVIDIA published", "B0", "M1", "MB1", "Kea-B' K-fp", "Kea-B' K-x1"]
    rows = {a: {c: 1.0 + 0.1 * i + 0.01 * j for j, c in enumerate(cols)} for i, a in enumerate(arms)}
    del rows["NVIDIA published"]["common_voice"]
    diffs = {k: {c: {"points": 0.1, "relative_percent": 1.0} for c in cols}
             for k in ("Kea-B' K-fp - MB1", "Kea-B' K-x1 - MB1")}
    sets = ["librispeech_dev_clean", "librispeech_dev_other", "ami_dev", "voxpopuli_dev"]
    test = {"rows": rows, "columns": cols, "differences": diffs,
            "empty_hypothesis_fraction": {a: {c: 0.001 for c in cols if c != "mean"} for a in arms[1:]},
            "sizes_mb": {"Kea-B'": 91.24, "MB1": 105.3, "M1": 180.8, "B0": 2472.2},
            "dev_check": {"x1": {"mean_percent": 6.1, "recorded_mean_percent": 6.1, "token_identity_rate": 1.0},
                          "fp": {"mb1_mean_percent": 6.3, "keab_fp_dev_percent": {s: 2.0 for s in sets}}}}
    summary = {"steps": 97137, "lr": 2e-5, "batch_seconds": 600.0, "audio_hours_seen": 15602.4, "seed": 1,
               "started_utc": "2026-11-03T09:07:51Z", "finished_utc": "2026-11-03T12:21:47Z"}
    config = {"args": {"schedule": {"warmup": 1651, "cooldown": 14579}, "teacher": {"sha256": "ab" * 32}}}
    results = {"gate": {"delta": {"h4": {"point": 0.33, "ci": [0.23, 0.41]},
                                  "sets": {s: {"point": 0.1, "ci": [0.0, 0.2]} for s in sets},
                                  "wer": {s: {"a": 2.1, "b": 2.0} for s in sets}}}}
    sizes = {"bundle_file_bytes": 91_345_312, "bundle_tensor_bytes": 91_236_520, "view_file_bytes": 105_299_804,
             "view_tensor_bytes": 105_190_000}
    return {"test": test, "summary": summary, "config": config, "results": results,
            "prov": pv.derive(**provenance_inputs()), "sizes": sizes,
            "view_manifest": {"versions": {"nemo": "3.0.0", "torch": "2.11.0+cu128"}}, "repo_id": "org/kea-b-0.1",
            "assets": ["size-vs-wer.png"]}


class Documents(unittest.TestCase):
    def test_templates_render_without_placeholders_and_the_card_checks_pass(self):
        inputs = fake_result_inputs()
        vals = stage.values(**inputs)
        docs = {t: kit.render((HF / t).read_text(), vals) for t in ("MODEL_CARD.md", "EVALUATION.md", "TRAINING.md",
                                                                     "NOTICE")}
        for name, text in docs.items():
            self.assertEqual(kit.unfilled(text), [], name)
        checks, info = {}, {}
        stage.card_checks(docs["MODEL_CARD.md"], inputs["test"], inputs["prov"], inputs["repo_id"], checks, info)
        self.assertEqual(len(checks), 3)
        self.assertTrue(all(checks.values()), checks)
        self.assertTrue(stage.asset_links({"README.md": docs["MODEL_CARD.md"]}))
        self.assertIn("91.24 MB", docs["MODEL_CARD.md"])

    def test_asset_links_finds_every_relative_asset_image_and_link(self):
        docs = {"a": "![x](assets/one.png) and [see](assets/two.webp)", "b": "![y](https://e.org/assets/no.png) ![z](assets/one.png)"}
        self.assertEqual(stage.asset_links(docs), {"one.png", "two.webp"})

    def test_render_refuses_missing_values_and_leftovers(self):
        with self.assertRaises(kit.RenderError):
            kit.render("a {{x}} b", {})
        with self.assertRaises(kit.RenderError):
            kit.render("a {{x}} b TODO", {"x": "1"})
        self.assertEqual(kit.render("a {{ x }} b", {"x": "1"}), "a 1 b")

    def test_loader_file_is_the_shipped_one_with_its_mit_notice(self):
        text = (HF / kit.LOADER).read_text()
        for line in ("# MIT License", "# Copyright (c) 2026 ajbarryiii", "# Permission is hereby granted"):
            self.assertIn(line, text)
        self.assertEqual(kit.sha256_file(HF / "LICENSE"),
                         "9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411")


if __name__ == "__main__":
    unittest.main()
