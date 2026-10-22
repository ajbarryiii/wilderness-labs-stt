"""Contract tests of hf/stage_card.py (Kea-B): the docs-only card update that adds the INT8x2 runtime results
(plans/card-update-x2.md). The shared engine (finetune/parakeet-ternary/hf/cardkit.py) and the shared x2
checks are tested in depth by finetune/parakeet-ternary/tests/test_stage_card.py; these tests cover Kea-B's
configuration, documents and every check family on Kea-B's real data.

    ./python -m unittest discover -s tests -p test_kea_b_card.py      # from finetune/parakeet-binary

CPU only, seconds, no model and no network: the Hub is faked from the release folder and report, origin/main
is the local git listing.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import paths

PKG = Path(__file__).resolve().parents[1]
HF = PKG / "hf"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sc = _load("kea_b_stage_card", HF / "stage_card.py")
cardkit = sc.cardkit
cardfake = _load("cardfake", PKG.parent / "parakeet-ternary" / "tests" / "cardfake.py")
DOCS = ("README.md", "RUNTIME.md")
BLOCKERS = ["GitHub links point at paths on origin/main",
            "x2 runtime published on origin/main (gated selection, runtime README, measured kernels)"]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mounted() -> bool:
    return sc.STAGE_REPORT.is_file() and sc.KEA_T_STAGE_REPORT.is_file()


def live_card() -> str:
    template = subprocess.run(["git", "-C", str(paths.REPO), "show",
                               f"{sc.OLD_CARD_COMMIT}:finetune/parakeet-binary/hf/MODEL_CARD.md"],
                              check=True, capture_output=True, text=True).stdout
    return sc.render({"README.md": template}, {})["README.md"]


def fake_hub(live: str) -> cardfake.FakeHub:
    report = json.loads(sc.STAGE_REPORT.read_text())
    folder = Path(report["folder"])
    small = {rel: (folder / rel).read_bytes() for rel in report["files"] if rel not in sc.LFS_FILES and rel != "README.md"}
    small["README.md"] = live.encode()
    lfs = {rel: (report["files"][rel]["bytes"], report["files"][rel]["sha256"]) for rel in sc.LFS_FILES}
    kea_t = json.loads(sc.KEA_T_STAGE_REPORT.read_text())["files"]["export.safetensors"]
    siblings = {sc.KEA_T_REPO: {"export.safetensors": cardkit.FileMeta(kea_t["bytes"], kea_t["sha256"], "blob-kea-t")}}
    return cardfake.FakeHub(sc.PINNED_REVISION, small, lfs, siblings)


@unittest.skipUnless(mounted(), "needs /mnt/hd and the release records")
class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sources = sc.load_sources()
        cls.templates = sc.load_templates()
        cls.live = live_card()
        cls.main_files = sc.origin_main_files(paths.REPO)
        cls.pub_sources, cls.pub_templates, cls.pub_main = cardfake.published(cls.sources, cls.templates, cls.main_files)
        cls.unpub_sources, cls.unpub_main = cardfake.unpublished(cls.sources, cls.main_files)

    def setUp(self) -> None:
        self.hub = fake_hub(self.live)
        cfg = dataclasses.replace(sc.config(), gitattributes_sha256=sha(cardfake.GITATTRIBUTES))
        patcher = mock.patch.object(sc, "config", lambda: cfg)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_prepare(self, templates=None, sources=None, main_files=None, published=True):
        base_t, base_s, base_m = ((self.pub_templates, self.pub_sources, self.pub_main) if published
                                  else (self.templates, self.unpub_sources, self.unpub_main))
        checks, info = {}, {}
        docs = sc.prepare(self.hub, base_m if main_files is None else main_files,
                          base_s if sources is None else sources, base_t if templates is None else templates,
                          checks, info)
        return docs, checks, info

    def failed(self, checks: dict) -> list[str]:
        return sorted(k for k, ok in checks.items() if ok is not True)

    def tampered(self, rel: str, old: str, new: str) -> dict:
        templates = dict(self.pub_templates)
        self.assertIn(old, templates[rel], f"{old!r} not in {rel}")
        templates[rel] = templates[rel].replace(old, new, 1)
        return templates

    def assert_fails(self, templates, *names, **kwargs) -> None:
        failed = self.failed(self.run_prepare(templates=templates, **kwargs)[1])
        for name in names:
            self.assertIn(name, failed)


class RealDocuments(Base):
    def test_live_card_pin_is_the_release_template_rendered(self) -> None:
        self.assertEqual(sha(self.live.encode()), sc.LIVE_DOCS["README.md"])

    def test_real_documents_pass_every_check_on_the_real_origin_main(self) -> None:
        """After PR #6 merged: the real sources and the real origin/main listing, every check passes."""
        checks, info = {}, {}
        docs = sc.prepare(self.hub, self.main_files, self.sources, self.templates, checks, info)
        self.assertEqual(self.failed(checks), [])

    def test_real_documents_fail_only_on_the_unpublished_x2_runtime(self) -> None:
        docs, checks, _ = self.run_prepare(published=False)
        self.assertEqual(self.failed(checks), BLOCKERS)
        self.assertEqual(set(checks) | cardkit.staging_checks(sc.config()), sc.required_checks())
        self.assertEqual(sorted(docs), sorted(DOCS))

    def test_real_documents_pass_once_the_x2_runtime_is_published(self) -> None:
        _, checks, _ = self.run_prepare()
        self.assertEqual(self.failed(checks), [])

    def test_published_link_must_be_on_origin_main(self) -> None:
        _, checks, _ = self.run_prepare(main_files=self.unpub_main)
        self.assertEqual(self.failed(checks), ["GitHub links point at paths on origin/main"])

    def test_card_is_short_and_leads_with_the_glance(self) -> None:
        card = sc.render(self.templates, self.sources)["README.md"]
        body = card.split("\n---\n", 1)[1]
        self.assertLessEqual(len(body.strip().splitlines()), 135)
        headings = [line for line in body.splitlines() if line.startswith("## ")]
        self.assertEqual(headings, ["## At a glance", "## Results", "## How it was trained", "## Use", "## Files",
                                    "## Limitations", "## License and attribution"])
        glance = " ".join(body.split("## At a glance", 1)[1].split("## Results", 1)[0].split())
        for s in ("| Weight file | **105.3 MB** | 2,472 MB |", "| Test WER, mean of 7 sets | 7.05% | 6.45% |",
                  "| Latency, 10 s clip | **3.88 ms** (33.3% lower) | 5.81 ms |",
                  "| GPU energy, 10 s clip (approximate) | **1.20 J** (about 37% less) | 1.90 J |",
                  "The latency and energy rows are warm batch-one transcription",
                  "Kea-B on our native 1-bit [research runtime]",
                  "The runtime code is published for reference and is not packaged for general use.",
                  "optimized BF16 build of the original (not stock NeMo)",
                  "measured in an earlier session, not side by side with Kea-B",
                  "Kea-T's runtimes, remeasured against their earlier records, agreed within 1% on latency but "
                  "only within 8% on energy, so the energy figures are approximate",
                  "[RUNTIME.md](RUNTIME.md)"):
            self.assertIn(s, glance)
        self.assertNotIn("Process VRAM", glance)  # AJ: no VRAM row on Kea-B's glance; RUNTIME.md keeps the column
        self.assertNotIn("TODO", body)
        results = " ".join(body.split("## Results", 1)[1].split("## How it was trained", 1)[0].split())
        self.assertIn("The GPU runtime was not used for these test WERs; on our development sets its mean WER, "
                      "6.31%, is the same as FP32 decoding's.", results)
        use = " ".join(body.split("## Use", 1)[1].split("## Files", 1)[0].split())
        self.assertIn("`load_binary.py` still rebuilds a dense FP32 model", use)
        self.assertNotIn("not part of this release", body)
        self.assertNotIn("We are working on native 1-bit kernels", body)
        self.assertIn("The weights, this card and `RUNTIME.md` are licensed", " ".join(body.split()))

    def test_runtime_doc(self) -> None:
        runtime = " ".join(sc.render(self.templates, self.sources)["RUNTIME.md"].split())
        for s in ("two residual INT8 parts (\"x2\"); the earlier version of the runtime used three (\"x3\")",
                  "| 30.04 s | **Kea-B, x2 activations (current)** | 9.29 | 1.51 | 3.24 | 348 |",
                  "| 30.04 s | Kea-B, x3 activations | 10.05 | 1.51 | 3.58 | 356 |",
                  "| 3 s | Original FP32 — NeMo | 10.37 | 3.41 | 2.98 | 283 |",
                  "| Full transcription | 11.4% less (24 of 24) | 6.8% lower (24 of 24) |",
                  "| Encoder only | 11.9% less (24 of 24) | 11.0% lower (24 of 24) |",
                  "6.3122% with x2 and 6.3120% with x3", "The test sets were not re-scored with the runtime",
                  "Process VRAM** is the warm process's sampled peak, in decimal GB",
                  "Speed measured with our research runtime on an RTX 5090; the runtime code is published for "
                  "reference and is not packaged for general use.",
                  "The research runtime runs on SM120 GPUs (RTX 5090) only and was measured with torch 2.11.0+cu128, "
                  "triton 3.6.0, nemo_toolkit 3.0.0 and Python 3.13."):
            self.assertIn(s, runtime)
        self.assertNotIn("Peak GPU memory", runtime)

    def test_release_kit_files_unchanged(self) -> None:
        report = json.loads(sc.STAGE_REPORT.read_text())["files"]
        for rel in ("LICENSE", "NOTICE", "load_binary.py"):
            self.assertEqual(sha((HF / rel).read_bytes()), report[rel]["sha256"], rel)


class Checks(Base):
    def test_numbers(self) -> None:
        self.assert_fails(self.tampered("RUNTIME.md", "Kea-B with x2 has 33.3% lower", "Kea-B with x2 has -33.3% lower"),
                          "RUNTIME.md computed numbers == sources", "RUNTIME.md every number is checked")
        self.assert_fails(self.tampered("RUNTIME.md", "At 10 s, against our BF16 build, Kea-B", "Against our BF16 build, Kea-B"),
                          "RUNTIME.md computed numbers == sources")
        self.assert_fails(self.tampered("README.md", "(about 37% less)", "(37% less)"), "card at-a-glance table == sources")
        self.assert_fails(self.tampered("README.md", "within 8% on energy", "within 2% on energy"),
                          "README.md computed numbers == sources", "README.md every number is checked")
        self.assert_fails(self.tampered("README.md", "The weight file is 105.3 MB", "The weight file is 105.3 MB, 2x smaller"),
                          "README.md every number is checked")
        self.assert_fails(self.tampered("RUNTIME.md", "fell from 323 W to 304 W", "fell from 323 W to 314 W"),
                          "RUNTIME.md computed numbers == sources", "RUNTIME.md every number is checked")
        self.assert_fails(self.tampered("README.md", "It took 68.6 hours", "It took 69.6 hours"),
                          "README.md computed numbers == sources", "README.md every number is checked")

    def test_dropped_live_number_fails(self) -> None:
        self.assert_fails(self.tampered("README.md", "It took 68.6 hours on one RTX 5090.", "It took about three days."),
                          "every number in the live documents appears in the new documents",
                          "README.md computed numbers == sources")

    def test_tables(self) -> None:
        self.assert_fails(self.tampered("RUNTIME.md", "| 10 s | **Kea-B, x2 activations (current)** | 3.88 |",
                                        "| 10 s | **Kea-B, x2 activations (current)** | 4.13 |"),
                          "RUNTIME.md x2 results table == sources")
        self.assert_fails(self.tampered("RUNTIME.md", "| Encoder only | 11.9% less", "| Encoder only | 12.9% less"),
                          "RUNTIME.md paired table == report-dev")
        self.assert_fails(self.tampered("README.md", "| 7.05 | 13.42 |", "| 7.04 | 13.42 |"),
                          "card WER table == TEST.md B0, M1 and MB1 rows")

    def test_sources(self) -> None:
        sources = copy.deepcopy(self.pub_sources)
        for r in sources["x2"]["test"]["p_clips"]["MB1-B-hybrid-x2"]["1320-122617-0010"]:
            r["joules_per_iteration"] = 1.5
        failed = self.failed(self.run_prepare(sources=sources)[1])
        self.assertIn("card at-a-glance table == sources", failed)
        self.assertIn("x2 clip artifacts == report-test", failed)
        sources = copy.deepcopy(self.pub_sources)
        sources["test"]["m1_export_sha256"] = "0" * 64
        self.assertIn("Kea-B test.json Kea-T export SHA-256 == Kea-T release record",
                      self.failed(self.run_prepare(sources=sources)[1]))
        sources = copy.deepcopy(self.pub_sources)
        sources["stage_report"]["status"] = "FAILED"
        self.assertIn("Kea-B stage report PASSED and is the pinned release", self.failed(self.run_prepare(sources=sources)[1]))

    def test_render_and_links(self) -> None:
        self.assert_fails(self.tampered("RUNTIME.md", "## Limitations", "## Limitations\n\nTBD"),
                          "no unrendered placeholder or TODO marker")
        self.assert_fails(self.tampered("README.md", "value: 13.42", "value: 13.43"), "card front matter == live card front matter")
        self.assert_fails(self.tampered("RUNTIME.md", "(#comparing-across-sessions)", "(#elsewhere)"),
                          "relative links resolve to documents and headings")
        unlinked = self.tampered("RUNTIME.md", "[Kea-B model card](README.md)", "Kea-B model card")
        unlinked["RUNTIME.md"] = unlinked["RUNTIME.md"].replace("[Use](README.md#use)", "Use")
        self.assert_fails(unlinked, "card links every detail document; each links back")
        self.assert_fails(self.tampered("README.md", "(https://huggingface.co/rajb3/kea-t)", "(https://huggingface.co/rajb3/kea-x)"),
                          "other absolute URLs appear in the live documents or are allowed")

    def test_hub(self) -> None:
        self.hub.head_sha = "f" * 40
        self.assert_fails(None, "hub main revision == pinned 5e7c298")
        self.hub = fake_hub(self.live)
        self.hub.put("RUNTIME.md", b"# already here\n")
        self.assert_fails(None, "new document names are not on the hub", "hub file set == pinned file set")
        self.hub = fake_hub(self.live)
        self.hub.put("load_binary.py", self.hub.content["load_binary.py"] + b"\n")
        self.assert_fails(None, "hub load_binary.py SHA-256 and size == release record")
        self.hub = fake_hub(self.live)
        self.hub.put("README.md", self.live.encode() + b"\n")
        self.assert_fails(None, "hub README.md SHA-256 == pinned live document")
        self.hub = fake_hub(self.live)
        self.hub.siblings[sc.KEA_T_REPO] = None
        self.assert_fails(None, "hub rajb3/kea-t is public; export.safetensors LFS SHA-256 and size == Kea-T release record")


class StagingAndPublish(Base):
    def setUp(self) -> None:
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.staging = Path(self.tmp.name)

    def stage(self, published=True):
        t, s, m = ((self.pub_templates, self.pub_sources, self.pub_main) if published
                   else (self.templates, self.unpub_sources, self.unpub_main))
        return cardkit.stage(sc.config(), self.staging, lambda c, i: sc.prepare(self.hub, m, s, t, c, i))

    def test_unpublished_x2_runtime_stages_nothing(self) -> None:
        ok, report, path = self.stage(published=False)
        self.assertFalse(ok)
        self.assertEqual(report["failed_checks"], BLOCKERS)
        self.assertEqual(path.name, f"{sc.NAME}.stage.FAILED.json")
        self.assertEqual(sorted(p.name for p in self.staging.iterdir()), [path.name])

    def test_stage_upload_verify_premerge_main(self) -> None:
        ok, report, _ = self.stage()
        self.assertTrue(ok, report.get("failed_checks"))
        folder = Path(report["folder"])
        self.assertEqual(sorted(p.name for p in folder.iterdir()), sorted(DOCS))
        self.assertEqual(set(report["checks"]), sc.required_checks())
        cfg = sc.config()
        checks, info = {}, {}
        self.assertEqual(report["bundle_sha256"], cardkit.bundle_sha256(sc.REPO_ID, sc.PINNED_REVISION, report["files"]))
        self.assertIsNone(cardkit.upload(cfg, self.hub, folder, report, report["files"]["README.md"]["sha256"], checks, info))
        self.assertEqual(self.hub.created, [])
        checks, info = {}, {}
        pr = cardkit.upload(cfg, self.hub, folder, report, report["bundle_sha256"], checks, info)
        self.assertEqual(self.failed(checks), [])
        self.assertEqual(self.hub.created[0]["parent"], sc.PINNED_REVISION)
        self.assertEqual(sorted(self.hub.created[0]["files"]), sorted(DOCS))
        checks, info = {}, {}
        cardkit.verify(cfg, self.hub, pr["pr_revision"], report, checks, info)
        self.assertEqual(self.failed(checks), [])
        self.assertEqual(set(checks), cardkit.verify_checks(cfg, pr["pr_revision"]))
        verified = {"status": "PASSED", **info}
        checks, info = {}, {}
        cardkit.pre_merge(cfg, self.hub, pr["pr_revision"], verified, checks, info)
        self.assertEqual(self.failed(checks), [])
        self.hub.merge(pr["pr_revision"])
        checks, info = {}, {}
        cardkit.verify(cfg, self.hub, "main", report, checks, info, pr_head=verified["verify"]["revision_sha"])
        self.assertEqual(self.failed(checks), [])

    def test_old_approval_fails_after_a_runtime_only_restage(self) -> None:
        """Codex card-update-x2 r1 major, on Kea-B: Codex's exact edit (drop "not" from "were not re-scored")
        changes only RUNTIME.md; the old approval must fail with no write."""
        ok, first, _ = self.stage()
        self.assertTrue(ok)
        templates = dict(self.pub_templates)
        self.assertIn("The test sets were not\nre-scored", templates["RUNTIME.md"])
        templates["RUNTIME.md"] = templates["RUNTIME.md"].replace("The test sets were not\nre-scored",
                                                                  "The test sets were\nre-scored", 1)
        t, s, m = templates, self.pub_sources, self.pub_main
        ok, second, _ = cardkit.stage(sc.config(), self.staging, lambda c, i: sc.prepare(self.hub, m, s, t, c, i))
        self.assertTrue(ok, second.get("failed_checks"))
        self.assertEqual(first["files"]["README.md"], second["files"]["README.md"])
        checks = {}
        self.assertIsNone(cardkit.upload(sc.config(), self.hub, Path(second["folder"]), second, first["bundle_sha256"], checks, {}))
        self.assertIn("approved bundle SHA-256 == bundle of the bytes to upload", self.failed(checks))
        self.assertEqual(self.hub.created, [])

    def test_verify_fails_on_a_changed_release_file(self) -> None:
        ok, report, _ = self.stage()
        self.assertTrue(ok)
        cfg = sc.config()
        pr = cardkit.upload(cfg, self.hub, Path(report["folder"]), report, report["bundle_sha256"], {}, {})
        self.hub.put("NOTICE", b"changed\n", revision=pr["pr_revision"])
        checks = {}
        cardkit.verify(cfg, self.hub, pr["pr_revision"], report, checks, {})
        self.assertIn("revision NOTICE unchanged from pinned", self.failed(checks))


class Safety(unittest.TestCase):
    def test_names_and_pins(self) -> None:
        self.assertEqual(sc.NAME, "rajb3__kea-b.card-update-x2")
        self.assertEqual(sc.STAGING, Path("/mnt/hd/wilderness-labs-stt/parakeet-binary/hf-staging"))
        self.assertEqual(sc.PINNED_REVISION, "5e7c298bc0eee174394a3fba88d789dc7c53ee4d")

    def test_import_does_not_import_torch(self) -> None:
        code = (f"import sys, importlib.util; spec = importlib.util.spec_from_file_location('s', {str(HF / 'stage_card.py')!r}); "
                "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
                "sys.exit(1 if 'torch' in sys.modules else 0)")
        self.assertEqual(subprocess.run([sys.executable, "-c", code]).returncode, 0)


if __name__ == "__main__":
    unittest.main()
