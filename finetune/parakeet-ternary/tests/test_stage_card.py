"""Contract tests of hf/stage_card.py, the docs-only Kea-T card update (plans/card-update.md).

CPU only, seconds, no model and no network: the Hub is faked from the original stage report and
the real files it names, origin/main is the local git listing. The real-data test is the staging
run itself (plans/card-update.md section 5).
"""
from __future__ import annotations

import ast
import copy
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


sc = _load("kea_t_stage_card", HF / "stage_card.py")
DOCS = ("README.md", "RUNTIME.md", "EVALUATION.md", "TRAINING.md")
DETAIL = ("RUNTIME.md", "EVALUATION.md", "TRAINING.md")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mounted() -> bool:
    return Path(paths.ARTIFACTS).is_dir() and sc.STAGE_REPORT.is_file()


def live_card() -> str:
    """The live README: the card template at OLD_CARD_COMMIT, rendered for rajb3/kea-t."""
    template = subprocess.run(["git", "-C", str(paths.REPO), "show",
                               f"{sc.OLD_CARD_COMMIT}:finetune/parakeet-ternary/hf/MODEL_CARD.md"],
                              check=True, capture_output=True, text=True).stdout
    return sc.render({"README.md": template}, sc.load_sources())["README.md"]


class FakeHub:
    """The Hub: the pinned revision built from the original stage report and the files it names,
    plus main edits, pull requests (create_pr records every call) and merges."""

    GITATTRIBUTES = b"*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
    OLDER = ["bc1c883f6d967398f32750a01c0e17d2e710a496", "4cd127f33f46ec1e2833cd8571bee5fd195dcbd7"]

    def __init__(self, live: str) -> None:
        report = json.loads(sc.STAGE_REPORT.read_text())
        export = Path(report["export"])
        self.head_sha = sc.PINNED_REVISION
        self.content: dict[str, bytes] = {}
        self.meta: dict[str, sc.FileMeta] = {}
        for rel, entry in report["files"].items():
            if rel in sc.LFS_FILES:
                self.meta[rel] = sc.FileMeta(entry["bytes"], entry["sha256"], f"blob-{rel}")
                continue
            if rel == "README.md":
                data = live.encode()
            elif rel == "NOTICE":
                data = (HF / "NOTICE").read_bytes()
            elif rel in ("LICENSE", "load_ternary.py"):
                data = (HF / rel).read_bytes()
            else:
                data = (export / rel).read_bytes()
            self.put(rel, data)
        self.put(".gitattributes", self.GITATTRIBUTES)
        # Other revisions (main edits, pull requests); the pinned revision is meta/content.
        self.revisions: dict[str, tuple[dict, dict]] = {}
        self.log: dict[str, list[str]] = {sc.PINNED_REVISION: [sc.PINNED_REVISION, *self.OLDER]}
        self.created: list[dict] = []
        self.counter = 0
        kea_b = json.loads(sc.KEA_B_STAGE_REPORT.read_text())["files"]["export.safetensors"]
        self.kea_b: dict | None = {"export.safetensors": sc.FileMeta(kea_b["bytes"], kea_b["sha256"], "blob-kea-b")}

    def _commit_id(self) -> str:
        self.counter += 1
        return f"{self.counter:040x}"

    def _resolve(self, revision: str) -> str:
        return self.head_sha if revision == "main" else revision

    def put(self, rel: str, data: bytes, revision: str | None = None) -> None:
        meta = sc.FileMeta(len(data), None, "blob-" + sha(data)[:40])
        if revision is None:
            self.content[rel], self.meta[rel] = data, meta
        else:
            self.revisions[revision][0][rel] = meta
            self.revisions[revision][1][rel] = data

    def fork(self, revision: str, parent: str = sc.PINNED_REVISION) -> None:
        meta, content = self._at(parent)
        self.revisions[revision] = (dict(meta), dict(content))
        self.log[revision] = [self._commit_id(), *self.log[self._resolve(parent)]]

    def edit_main(self, rel: str, data: bytes) -> str:
        """Another editor's commit on main."""
        new = self._commit_id()
        meta, content = self._at("main")
        self.revisions[new] = (dict(meta), dict(content))
        self.put(rel, data, revision=new)
        self.log[new] = [new, *self.log[self.head_sha]]
        self.head_sha = new
        return new

    def add_pr_commit(self, revision: str, rel: str, data: bytes) -> None:
        self.put(rel, data, revision=revision)
        self.log[revision] = [self._commit_id(), *self.log[revision]]

    def merge(self, revision: str, merge_commit: bool = False) -> None:
        """Merge a PR into main: fast-forward when main is the PR's parent and merge_commit is False,
        otherwise a merge commit whose history also holds main's commits since the PR's parent
        (the PR's files win, as in a silent overwrite)."""
        meta, content = self.revisions[revision]
        pr_log = self.log[revision]
        if not merge_commit and self.head_sha == pr_log[1]:
            new, log = pr_log[0], pr_log
        else:
            new = self._commit_id()
            log = [new, *[c for c in self.log[self.head_sha] if c not in pr_log], *pr_log]
        self.revisions[new] = (dict(meta), dict(content))
        self.log[new] = log
        self.head_sha = new

    def head(self) -> str:
        return self.head_sha

    def _at(self, revision: str) -> tuple[dict, dict]:
        revision = self._resolve(revision)
        if revision in self.revisions:
            return self.revisions[revision]
        if revision != sc.PINNED_REVISION:
            raise KeyError(revision)
        return self.meta, self.content

    def files(self, revision: str) -> dict:
        return dict(self._at(revision)[0])

    def read(self, rel: str, revision: str) -> bytes:
        return self._at(revision)[1][rel]

    def commits(self, revision: str) -> list[str]:
        return list(self.log[self._resolve(revision)])

    def public_files(self, repo_id: str) -> dict | None:
        assert repo_id == sc.KEA_B_REPO
        return None if self.kea_b is None else dict(self.kea_b)

    def create_pr(self, files: dict, parent: str, message: str) -> dict:
        self.created.append({"files": dict(files), "parent": parent, "message": message})
        num = len(self.created)
        revision = f"refs/pr/{num}"
        self.fork(revision, parent)
        for rel, data in files.items():
            self.put(rel, data, revision=revision)
        return {"pr_num": num, "pr_url": f"https://huggingface.co/rajb3/kea-t/discussions/{num}",
                "commit": self.log[revision][0], "pr_revision": revision}


@unittest.skipUnless(mounted(), "needs /mnt/hd and the original stage report")
class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sources = sc.load_sources()
        cls.templates = sc.load_templates()
        cls.live = live_card()
        cls.main_files = sc.origin_main_files(paths.REPO)

    def setUp(self) -> None:
        self.hub = FakeHub(self.live)
        patcher = mock.patch.object(sc, "GITATTRIBUTES_SHA256", sha(FakeHub.GITATTRIBUTES))
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_prepare(self, templates=None, sources=None, hub=None, main_files=None):
        checks, info = {}, {}
        docs = sc.prepare(hub or self.hub, self.main_files if main_files is None else main_files,
                          self.sources if sources is None else sources,
                          self.templates if templates is None else templates, checks, info)
        return docs, checks, info

    def failed(self, checks: dict) -> list[str]:
        return sorted(k for k, ok in checks.items() if ok is not True)

    def tampered(self, rel: str, old: str, new: str, count: int = 1) -> dict:
        templates = dict(self.templates)
        self.assertIn(old, templates[rel], f"{old!r} not in {rel}")
        templates[rel] = templates[rel].replace(old, new, count)
        return templates


class RealDocuments(Base):
    def test_live_card_pin_is_the_old_template_rendered(self) -> None:
        self.assertEqual(sha(self.live.encode()), sc.LIVE_README_SHA256)

    def test_real_documents_pass_every_check(self) -> None:
        docs, checks, info = self.run_prepare()
        self.assertEqual(self.failed(checks), [])
        self.assertEqual(set(checks) | sc.staging_checks(), sc.required_checks())
        self.assertEqual(sorted(docs), sorted(DOCS))

    def test_card_is_short_and_leads_with_the_glance(self) -> None:
        card = sc.render(self.templates, self.sources)["README.md"]
        body = card.split("\n---\n", 1)[1]
        self.assertLessEqual(len(body.strip().splitlines()), 135)
        headings = [line for line in body.splitlines() if line.startswith("## ")]
        self.assertEqual(headings[:2], ["## At a glance", "## Results"])
        glance = body.split("## At a glance", 1)[1].split("## Results", 1)[0]
        for s in ("**180.8 MB**", "2,472 MB", "6.84%", "6.45%", "**4.46 ms**", "5.81 ms", "23.2% lower",
                  "**1.57 J**", "1.90 J", "17.1% less", "1.89 GB", "2.18 GB", "custom CUDA runtime",
                  "not `load_ternary.py`", "RUNTIME.md"):
            self.assertIn(s, glance)
        for heading in ("## How it was trained", "## Use", "## Files", "## Limitations", "## Used in",
                        "## License and attribution"):
            self.assertIn(heading, headings)
        self.assertIn("LocalFlow", body)
        intro = body.split("## At a glance", 1)[0]
        self.assertIn("Its 1-bit sibling [Kea-B](https://huggingface.co/rajb3/kea-b) is 105.3 MB with a 7.05% "
                      "test mean.", " ".join(intro.split()))
        self.assertIn("hf download rajb3/kea-t --local-dir kea-t", body)
        self.assertNotIn("parakeet-ternary/load_ternary.py", body)
        license_text = " ".join(body.split("## License and attribution", 1)[1].split())
        self.assertIn("this model card and its detail documents (`RUNTIME.md`, `EVALUATION.md`, `TRAINING.md`) "
                      "are licensed under", license_text)
        self.assertIn("the remaining learned weights are stored in FP16", license_text)
        self.assertNotIn("all other tensors", license_text)

    def test_published_files_untouched_in_the_repository(self) -> None:
        report = json.loads(sc.STAGE_REPORT.read_text())["files"]
        for rel in ("LICENSE", "load_ternary.py"):
            self.assertEqual(sha((HF / rel).read_bytes()), report[rel]["sha256"], rel)
        self.assertEqual(sha((HF / "NOTICE").read_bytes()), sc.NOTICE_SHA256)

    def test_render_replaces_placeholders(self) -> None:
        docs = sc.render(self.templates, self.sources)
        for rel, text in docs.items():
            self.assertNotRegex(text, r"\{(repo_id|repo_name|code_histogram)\}", rel)
        self.assertIn("hf download rajb3/kea-t --local-dir kea-t", docs["README.md"])
        self.assertIn("32.9% -1, 34.2% 0, 32.9% +1", docs["TRAINING.md"])


class NumberChecks(Base):
    def assert_fails(self, templates, *names) -> None:
        _, checks, _ = self.run_prepare(templates=templates)
        failed = self.failed(checks)
        for name in names:
            self.assertIn(name, failed)

    def test_reversed_sign_fails(self) -> None:
        """Codex r1 finding 3: a sign added in front of a checked number."""
        self.assert_fails(self.tampered("RUNTIME.md", "has 63.1% lower latency", "has -63.1% lower latency"),
                          "RUNTIME.md computed numbers == sources", "RUNTIME.md every number is checked")
        self.assert_fails(self.tampered("RUNTIME.md", "  23.2% lower latency, 13.5%", "  +23.2% lower latency, 13.5%"),
                          "RUNTIME.md computed numbers == sources", "RUNTIME.md every number is checked")

    def test_limitations_delta_sign_swap_fails(self) -> None:
        self.assert_fails(self.tampered("README.md", "Common Voice (+4.06 points)", "Common Voice (-4.06 points)"),
                          "README.md computed numbers == sources", "README.md every number is checked")
        self.assert_fails(self.tampered("EVALUATION.md", "test-other (+1.01)", "test-other (-1.01)"),
                          "EVALUATION.md computed numbers == sources", "EVALUATION.md every number is checked")

    def test_changed_word_next_to_number_fails(self) -> None:
        self.assert_fails(self.tampered("RUNTIME.md", "17.1% less GPU", "17.1% more GPU"),
                          "RUNTIME.md computed numbers == sources")
        self.assert_fails(self.tampered("TRAINING.md", "(150,994,944 bytes)", "(150,994,944.5 bytes)"),
                          "TRAINING.md computed numbers == sources", "TRAINING.md every number is checked")
        self.assert_fails(self.tampered("TRAINING.md", "sum to 33,247", "sum to 133,247"),
                          "TRAINING.md computed numbers == sources", "TRAINING.md every number is checked")

    def test_removed_duration_qualifier_fails(self) -> None:
        """Codex r1 finding 1: duration comparisons must name their duration."""
        self.assert_fails(self.tampered("RUNTIME.md", "At 10 s, our BF16 build also has", "Our BF16 build also has"),
                          "RUNTIME.md computed numbers == sources")
        self.assert_fails(self.tampered("RUNTIME.md", "At 10 s, compared with our BF16\n  build, the ternary rows",
                                        "Compared with our BF16\n  build, the ternary rows"),
                          "RUNTIME.md computed numbers == sources")

    def test_power_sentence_bound_to_10_s_values(self) -> None:
        sources = copy.deepcopy(self.sources)
        sources["rtx"]["variants"]["packed_optimized"]["workloads"][1]["average_watts"] = 300.0
        _, checks, _ = self.run_prepare(sources=sources)
        self.assertIn("RUNTIME.md computed numbers == sources", self.failed(checks))
        sources = copy.deepcopy(self.sources)
        sources["rtx"]["variants"]["packed_optimized"]["workloads"][1]["joules_per_iteration"] = 1.95
        _, checks, _ = self.run_prepare(sources=sources)
        self.assertIn("RUNTIME.md computed numbers == sources", self.failed(checks))

    def test_phrase_boundaries(self) -> None:
        self.assertTrue(sc.contains("x has 63.1% lower", "63.1% lower"))
        self.assertFalse(sc.contains("x has -63.1% lower", "63.1% lower"))
        self.assertFalse(sc.contains("x has +63.1% lower", "63.1% lower"))
        self.assertFalse(sc.contains("on 12,048 utterances", "2,048 utterances"))
        self.assertFalse(sc.contains("sum to 33,247.5 h", "sum to 33,247"))
        self.assertFalse(sc.contains("sum to 33,2470", "sum to 33,247"))
        self.assertTrue(sc.contains("sum to 33,247; the", "sum to 33,247"))
        self.assertTrue(sc.contains("[istupakov/parakeet-tdt-0.6b-v2-onnx]", "parakeet-tdt-0.6b-v2"))

    def test_changed_kea_b_number_fails(self) -> None:
        self.assert_fails(self.tampered("README.md", "is 105.3 MB with", "is 105.4 MB with"),
                          "README.md computed numbers == sources", "README.md every number is checked")

    def test_changed_glance_number_fails(self) -> None:
        self.assert_fails(self.tampered("README.md", "**4.46 ms**", "**4.36 ms**"), "card at-a-glance table == sources")

    def test_changed_prose_number_fails(self) -> None:
        self.assert_fails(self.tampered("TRAINING.md", "69.3 hours", "59.3 hours"),
                          "TRAINING.md computed numbers == sources", "TRAINING.md every number is checked")

    def test_uncovered_number_fails(self) -> None:
        self.assert_fails(self.tampered("README.md", "English only, like the base model.",
                                        "English only, like the base model, and 3x faster."),
                          "README.md every number is checked")

    def test_changed_carried_number_fails(self) -> None:
        self.assert_fails(self.tampered("TRAINING.md", "0.98, weight decay", "0.99, weight decay"),
                          "TRAINING.md every number is checked")

    def test_dropped_number_fails(self) -> None:
        templates = self.tampered("EVALUATION.md", "fewer than the FP32\n  original's 3.12%",
                                  "fewer than the FP32\n  original's")
        self.assert_fails(templates, "every number in the live card appears in the new documents",
                          "EVALUATION.md computed numbers == sources")

    def test_runtime_table_cell_fails(self) -> None:
        self.assert_fails(self.tampered("RUNTIME.md", "| 10.56 | 2.23 | 3.99 | 376 |", "| 10.56 | 2.23 | 3.98 | 376 |"),
                          "RUNTIME.md results table == inference/README.md table")

    def test_extra_wer_row_fails(self) -> None:
        row = "| **Kea-T (this model)** | 2.05 | 4.20 | 10.42 | 11.72 | 10.35 | 2.94 | 6.19 | 6.84 | 12.56 |"
        self.assert_fails(self.tampered("README.md", row, row + "\n" + row), "card WER table == TEST.md B0 and M1 rows")
        evaluation_row = "| **This model** | 2.05 | 4.20 | 10.42 | 11.72 | 10.35 | 2.94 | 6.19 | 6.84 | 12.56 |"
        self.assert_fails(self.tampered("EVALUATION.md", evaluation_row, evaluation_row.replace("2.05", "2.06")),
                          "EVALUATION.md WER table == all 6 TEST.md rows")

    def test_source_tampering_fails(self) -> None:
        sources = copy.deepcopy(self.sources)
        sources["test"]["rows"]["M1"]["mean"] = 6.9
        _, checks, _ = self.run_prepare(sources=sources)
        self.assertIn("results/TEST.md rows == results/test.json rounded", self.failed(checks))
        sources = copy.deepcopy(self.sources)
        sources["rtx"]["variants"]["packed_expanded"]["workloads"][1]["median_ms"] = 4.9
        _, checks, _ = self.run_prepare(sources=sources)
        self.assertIn("inference/README.md results table == rtx5090.json rounded", self.failed(checks))
        self.assertIn("card at-a-glance table == sources", self.failed(checks))
        sources = copy.deepcopy(self.sources)
        sources["kea_b_test"]["rows"]["MB1"]["mean"] = 7.2
        _, checks, _ = self.run_prepare(sources=sources)
        self.assertIn("README.md computed numbers == sources", self.failed(checks))
        sources = copy.deepcopy(self.sources)
        sources["kea_b_test"]["export_mb"]["MB1"] = 105.4
        _, checks, _ = self.run_prepare(sources=sources)
        self.assertIn("README.md computed numbers == sources", self.failed(checks))
        sources = copy.deepcopy(self.sources)
        sources["kea_b_test"]["mb1_weights_sha256"] = "0" * 64
        _, checks, _ = self.run_prepare(sources=sources)
        self.assertIn("Kea-B stage report PASSED and its export SHA-256 == Kea-B test.json", self.failed(checks))
        sources = copy.deepcopy(self.sources)
        sources["kea_b_test"]["m1_export_sha256"] = "0" * 64
        _, checks, _ = self.run_prepare(sources=sources)
        self.assertIn("Kea-B test.json Kea-T export SHA-256 == stage report", self.failed(checks))
        sources = copy.deepcopy(self.sources)
        sources["test"]["m1_export_sha256"] = "0" * 64
        _, checks, _ = self.run_prepare(sources=sources)
        self.assertIn("results/test.json export SHA-256 == stage report", self.failed(checks))

    def test_number_scan(self) -> None:
        # Signed tokens (contract change in r1: a sign is part of the number).
        self.assertEqual(sc.numbers("2,472 MB, 1.21x, 5.5e-4, ae9ad07 b6bdcd0beb FP32 v2 (+0.36) -63.1% TDT-0.6"),
                         ["2,472", "1.21", "5.5e-4", "+0.36", "-63.1", "0.6"])
        self.assertEqual(sc.numbers("[a 3 b](https://x.org/5/6) ```\n7\n``` 8"), ["3", "8"])


class RenderAndLinkChecks(Base):
    def assert_fails(self, templates, name, **kwargs) -> None:
        _, checks, _ = self.run_prepare(templates=templates, **kwargs)
        self.assertIn(name, self.failed(checks))

    def test_placeholder_fails(self) -> None:
        self.assert_fails(self.tampered("RUNTIME.md", "Detail for the", "Detail ({repo_slug}) for the"),
                          "no unrendered placeholder")

    def test_front_matter_change_fails(self) -> None:
        self.assert_fails(self.tampered("README.md", "value: 4.20", "value: 4.21"), "card front matter == live card front matter")
        self.assert_fails(self.tampered("TRAINING.md", "# Kea-T: training", "---\nlicense: mit\n---\n# Kea-T: training"),
                          "detail documents have no front matter")

    def test_broken_relative_links_fail(self) -> None:
        name = "relative links resolve to staged documents and headings"
        self.assert_fails(self.tampered("README.md", "(EVALUATION.md#limitations)", "(EVALUATION.md#nope)"), name)
        self.assert_fails(self.tampered("README.md", "(RUNTIME.md)", "(RUNTIMES.md)"), name)

    def test_missing_doc_link_fails(self) -> None:
        name = "card links every detail document; each links back"
        templates = self.tampered("README.md", "[TRAINING.md](TRAINING.md)", "TRAINING.md", count=-1)
        templates["README.md"] = templates["README.md"].replace("[TRAINING.md](TRAINING.md#training-data-and-licenses)",
                                                                "TRAINING.md")
        self.assertNotIn("](TRAINING.md", templates["README.md"])
        self.assert_fails(templates, name)
        self.assert_fails(self.tampered("EVALUATION.md", "[Kea-T model card](README.md)", "Kea-T model card"), name)

    def test_github_link_to_missing_path_fails(self) -> None:
        self.assert_fails(self.tampered("RUNTIME.md", "inference/results/rtx5090.json", "inference/results/rtx4090.json"),
                          "GitHub links point at paths on origin/main")
        files = set(self.main_files) - {"finetune/parakeet-ternary/results/TEST.md"}
        self.assert_fails(self.templates, "GitHub links point at paths on origin/main", main_files=files)

    def test_unknown_external_url_fails(self) -> None:
        self.assert_fails(self.tampered("README.md", "(https://github.com/zachlatta/freeflow)",
                                        "(https://github.com/zachlatta/freeflow2)"),
                          "other absolute URLs appear in the live card or are Kea-B's page")

    def test_slug(self) -> None:
        self.assertEqual(sc.slug("Training data and licenses"), "training-data-and-licenses")
        self.assertEqual(sc.slug("Kea-T runtime results (custom CUDA runtime on GitHub)"),
                         "kea-t-runtime-results-custom-cuda-runtime-on-github")


class HubChecks(Base):
    def assert_fails(self, name) -> None:
        _, checks, _ = self.run_prepare()
        self.assertIn(name, self.failed(checks))

    def test_moved_head_fails(self) -> None:
        self.hub.head_sha = "f" * 40
        self.assert_fails("hub main revision == pinned db7d419")

    def test_changed_lfs_file_fails(self) -> None:
        meta = self.hub.meta["export.safetensors"]
        self.hub.meta["export.safetensors"] = sc.FileMeta(meta.size, "0" * 64, meta.blob_id)
        self.assert_fails("hub export.safetensors SHA-256 and size == stage report")

    def test_changed_small_file_fails(self) -> None:
        self.hub.put("load_ternary.py", self.hub.content["load_ternary.py"] + b"\n")
        self.assert_fails("hub load_ternary.py SHA-256 and size == stage report")

    def test_changed_notice_fails(self) -> None:
        self.hub.put("NOTICE", self.hub.content["NOTICE"].replace(b"CC0-1.0", b"CC-BY-4.0"))
        _, checks, _ = self.run_prepare()
        self.assertIn("hub NOTICE SHA-256 and size == hf/NOTICE (pinned, post-rename)", self.failed(checks))
        self.assertIn("hub NOTICE differs from the stage report's NOTICE only in its first line", self.failed(checks))

    def test_changed_readme_fails(self) -> None:
        self.hub.put("README.md", self.live.encode() + b"\n")
        self.assert_fails("hub README.md SHA-256 == pinned live card")

    def test_kea_b_private_or_changed_fails(self) -> None:
        name = "hub rajb3/kea-b is public; export.safetensors LFS SHA-256 and size == Kea-B stage report"
        self.hub.kea_b = None
        self.assert_fails(name)
        self.hub = FakeHub(self.live)
        meta = self.hub.kea_b["export.safetensors"]
        self.hub.kea_b["export.safetensors"] = sc.FileMeta(meta.size + 1, meta.lfs_sha256, meta.blob_id)
        self.assert_fails(name)

    def test_extra_file_and_existing_doc_name_fail(self) -> None:
        self.hub.put("RUNTIME.md", b"# already here\n")
        _, checks, _ = self.run_prepare()
        self.assertIn("hub file set == stage report files + .gitattributes", self.failed(checks))
        self.assertIn("new document names are not on the hub", self.failed(checks))


class Staging(Base):
    def setUp(self) -> None:
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.staging = Path(self.tmp.name)
        self.final = self.staging / sc.NAME
        self.ok_report = self.staging / f"{sc.NAME}.stage.json"
        self.failed_report = self.staging / f"{sc.NAME}.stage.FAILED.json"

    def real_prepare(self):
        return lambda checks, info: sc.prepare(self.hub, self.main_files, self.sources, self.templates, checks, info)

    def fake_prepare(self, fail=None, error=None, extra=None, drop=None, value=True):
        def run(checks, info):
            if error is not None:
                raise error
            for name in sc.required_checks() - sc.staging_checks():
                if name != drop:
                    checks[name] = value if name == fail and value is not True else name != fail
            if extra:
                checks[extra] = True
            return {rel: f"# {rel}\n" for rel in DOCS}
        return run

    def leftovers(self) -> list[str]:
        return sorted(p.name for p in self.staging.iterdir() if p.name.startswith("."))

    def test_real_staging_promotes_exactly_the_four_documents(self) -> None:
        ok, report, path = sc.stage(self.staging, self.real_prepare())
        self.assertTrue(ok, report.get("failed_checks"))
        self.assertEqual(path, self.ok_report)
        self.assertEqual(sorted(p.relative_to(self.final).as_posix() for p in self.final.rglob("*") if p.is_file()),
                         sorted(DOCS))
        saved = json.loads(self.ok_report.read_text())
        self.assertEqual(saved["status"], "PASSED")
        self.assertEqual(set(saved["checks"]), sc.required_checks())
        self.assertEqual(sorted(saved["files"]), sorted(DOCS))
        for rel in DOCS:
            self.assertEqual(saved["files"][rel]["sha256"], sha((self.final / rel).read_bytes()))
        self.assertIn("hf/stage_card.py upload --approved-readme-sha256", saved["upload_command"])
        self.assertEqual(saved["files"]["README.md"]["sha256"], sha((self.final / "README.md").read_bytes()))
        self.assertFalse(self.failed_report.exists())
        self.assertEqual(self.leftovers(), [])

    def test_failed_check_leaves_nothing(self) -> None:
        ok, report, path = sc.stage(self.staging, self.fake_prepare(fail="card front matter == live card front matter"))
        self.assertFalse(ok)
        self.assertEqual(path, self.failed_report)
        self.assertFalse(self.final.exists())
        self.assertFalse(self.ok_report.exists())
        self.assertEqual(self.leftovers(), [])
        self.assertIn("card front matter == live card front matter", report["failed_checks"])

    def test_real_failure_leaves_nothing(self) -> None:
        self.hub.head_sha = "f" * 40
        ok, _, _ = sc.stage(self.staging, self.real_prepare())
        self.assertFalse(ok)
        self.assertFalse(self.final.exists())
        self.assertEqual(self.leftovers(), [])

    def test_exception_and_exit_leave_nothing(self) -> None:
        for error in (RuntimeError("boom"), SystemExit(3), KeyboardInterrupt()):
            ok, report, _ = sc.stage(self.staging, self.fake_prepare(error=error))
            self.assertFalse(ok)
            self.assertFalse(self.final.exists())
            self.assertEqual(self.leftovers(), [])
            self.assertEqual(report["status"], "FAILED")

    def test_check_set_is_enforced(self) -> None:
        for kwargs in ({"drop": "carried phrases appear in the live card"}, {"extra": "an unplanned check"}):
            ok, report, _ = sc.stage(self.staging, self.fake_prepare(**kwargs))
            self.assertFalse(ok, kwargs)
            self.assertIn("check set mismatch", report["error"])
            self.assertFalse(self.final.exists())

    def test_non_boolean_true_is_a_failure(self) -> None:
        ok, _, _ = sc.stage(self.staging, self.fake_prepare(fail="no unrendered placeholder", value=1))
        self.assertFalse(ok)
        self.assertFalse(self.final.exists())

    def test_stale_outputs_removed_even_on_failure(self) -> None:
        self.final.mkdir()
        (self.final / "README.md").write_text("stale")
        self.ok_report.write_text("{}")
        (self.staging / f".{sc.NAME}.tmp-1").mkdir()
        ok, _, _ = sc.stage(self.staging, self.fake_prepare(fail="no unrendered placeholder"))
        self.assertFalse(ok)
        self.assertFalse(self.final.exists())
        self.assertFalse(self.ok_report.exists())
        self.assertEqual(self.leftovers(), [])


class Publish(Base):
    """upload, verify refs/pr/N, pre-merge and verify main (plans/card-update.md section 3)."""

    def setUp(self) -> None:
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        ok, self.report, _ = sc.stage(Path(self.tmp.name),
                                      lambda c, i: sc.prepare(self.hub, self.main_files, self.sources,
                                                              self.templates, c, i))
        self.assertTrue(ok, self.report.get("failed_checks"))
        self.folder = Path(self.report["folder"])
        self.approved = self.report["files"]["README.md"]["sha256"]

    def run_upload(self, report=None, approved=None) -> tuple[dict | None, list[str]]:
        checks, info = {}, {}
        pr = sc.upload(self.hub, self.folder, self.report if report is None else report,
                       self.approved if approved is None else approved, checks, info)
        self.assertEqual(set(checks), sc.upload_checks())
        return pr, self.failed(checks)

    def uploaded(self) -> dict:
        pr, failed = self.run_upload()
        self.assertEqual(failed, [])
        return pr

    def run_verify(self, revision="refs/pr/1", report=None, pr_head=None) -> tuple[list[str], dict]:
        checks, info = {}, {}
        sc.verify(self.hub, revision, self.report if report is None else report, checks, info, pr_head=pr_head)
        self.assertEqual(set(checks), sc.verify_checks(revision))
        return self.failed(checks), info

    def verified(self) -> dict:
        failed, info = self.run_verify()
        self.assertEqual(failed, [])
        return {"status": "PASSED", **info}

    def run_pre_merge(self, verify_report) -> list[str]:
        checks, info = {}, {}
        sc.pre_merge(self.hub, "refs/pr/1", verify_report, checks, info)
        self.assertEqual(set(checks), sc.pre_merge_checks())
        return self.failed(checks)

    # upload
    def test_upload_creates_one_pr_on_the_pinned_parent(self) -> None:
        pr = self.uploaded()
        self.assertEqual(len(self.hub.created), 1)
        call = self.hub.created[0]
        self.assertEqual(call["parent"], sc.PINNED_REVISION)
        self.assertEqual(call["files"], {rel: (self.folder / rel).read_bytes() for rel in DOCS})
        self.assertEqual(call["message"], sc.COMMIT_MESSAGE)
        self.assertEqual(pr["pr_revision"], "refs/pr/1")

    def test_upload_refuses_without_writing(self) -> None:
        cases = {
            "card-update stage report PASSED": lambda: self.run_upload(report=dict(self.report, status="FAILED")),
            "approved README SHA-256 == staged README": lambda: self.run_upload(approved="0" * 64),
        }
        for name, run in cases.items():
            pr, failed = run()
            self.assertIsNone(pr)
            self.assertIn(name, failed)
        (self.folder / "RUNTIME.md").write_text("# edited after staging\n")
        pr, failed = self.run_upload()
        self.assertIsNone(pr)
        self.assertIn("staged files == stage report (exact set, SHA-256 and size)", failed)
        self.assertEqual(self.hub.created, [])

    def test_upload_refuses_after_a_main_edit(self) -> None:
        self.hub.edit_main("README.md", self.live.replace("6.84%", "6.85%").encode())
        pr, failed = self.run_upload()
        self.assertIsNone(pr)
        self.assertIn("main is still the pinned revision", failed)
        self.assertEqual(self.hub.created, [])

    def test_upload_refuses_an_extra_staged_file(self) -> None:
        (self.folder / "extra.md").write_text("x\n")
        pr, failed = self.run_upload()
        self.assertIsNone(pr)
        self.assertIn("staged files == stage report (exact set, SHA-256 and size)", failed)

    # verify refs/pr/N
    def test_unchanged_pr_passes_and_records_its_head(self) -> None:
        pr = self.uploaded()
        failed, info = self.run_verify()
        self.assertEqual(failed, [])
        self.assertEqual(info["verify"]["revision_sha"], pr["commit"])

    def test_main_edit_after_the_pr_fails_verify(self) -> None:
        """Codex r1 finding 2: a README correction on main after the PR was opened."""
        self.uploaded()
        self.hub.edit_main("README.md", self.live.replace("6.84%", "6.85%").encode())
        failed, _ = self.run_verify()
        self.assertIn("main is still the pinned revision", failed)

    def test_pr_based_on_a_newer_main_fails_verify(self) -> None:
        """A PR made the CLI way, from a main that moved after staging."""
        edited = self.hub.edit_main("README.md", self.live.replace("6.84%", "6.85%").encode())
        self.hub.create_pr({rel: (self.folder / rel).read_bytes() for rel in DOCS}, edited, "cli-style")
        failed, _ = self.run_verify()
        self.assertIn("main is still the pinned revision", failed)
        self.assertIn("revision is one commit on the pinned revision", failed)

    def test_changed_non_doc_file_fails(self) -> None:
        self.uploaded()
        self.hub.put("NOTICE", b"changed\n", revision="refs/pr/1")
        self.assertIn("revision NOTICE unchanged from pinned", self.run_verify()[0])

    def test_doc_mismatch_or_missing_fails(self) -> None:
        self.uploaded()
        self.hub.put("RUNTIME.md", b"# other\n", revision="refs/pr/1")
        self.assertIn("revision RUNTIME.md == staged", self.run_verify()[0])
        del self.hub.revisions["refs/pr/1"][0]["TRAINING.md"]
        failed, _ = self.run_verify()
        self.assertIn("revision file set == pinned files + detail documents", failed)
        self.assertIn("revision TRAINING.md == staged", failed)

    def test_extra_pr_commit_fails_verify(self) -> None:
        self.uploaded()
        self.hub.add_pr_commit("refs/pr/1", "EVALUATION.md", self.folder.joinpath("EVALUATION.md").read_bytes())
        self.assertIn("revision is one commit on the pinned revision", self.run_verify()[0])

    def test_unpassed_report_fails(self) -> None:
        self.uploaded()
        failed, _ = self.run_verify(report=dict(self.report, status="FAILED"))
        self.assertIn("card-update stage report PASSED", failed)

    # pre-merge
    def test_pre_merge_passes_when_nothing_moved(self) -> None:
        self.uploaded()
        self.assertEqual(self.run_pre_merge(self.verified()), [])

    def test_pre_merge_fails_on_a_moved_main_or_pr_head(self) -> None:
        self.uploaded()
        report = self.verified()
        self.hub.add_pr_commit("refs/pr/1", "RUNTIME.md", b"# late change\n")
        self.assertIn("PR head == the verified PR head", self.run_pre_merge(report))
        self.hub.edit_main("README.md", b"# edited\n")
        self.assertIn("main is still the pinned revision", self.run_pre_merge(report))
        self.assertIn("verify report for the PR PASSED", self.run_pre_merge(dict(report, status="FAILED")))

    # verify main
    def test_main_after_merge_passes(self) -> None:
        for merge_commit in (False, True):
            self.hub = FakeHub(self.live)
            self.uploaded()
            pr_head = self.verified()["verify"]["revision_sha"]
            self.hub.merge("refs/pr/1", merge_commit=merge_commit)
            failed, _ = self.run_verify("main", pr_head=pr_head)
            self.assertEqual(failed, [], merge_commit)

    def test_main_with_an_intervening_commit_fails(self) -> None:
        """Codex r1's scenario end to end: a README correction on main before the merge is
        overwritten by the merge; the content checks alone pass, the history check does not."""
        self.uploaded()
        pr_head = self.verified()["verify"]["revision_sha"]
        self.hub.edit_main("README.md", self.live.replace("6.84%", "6.85%").encode())
        self.hub.merge("refs/pr/1")
        failed, _ = self.run_verify("main", pr_head=pr_head)
        self.assertEqual(failed, ["commits since the pinned revision are only the card update"])

    def test_edit_after_the_merge_fails(self) -> None:
        self.uploaded()
        pr_head = self.verified()["verify"]["revision_sha"]
        self.hub.merge("refs/pr/1")
        self.hub.edit_main("README.md", b"# edited after the merge\n")
        failed, _ = self.run_verify("main", pr_head=pr_head)
        self.assertIn("revision README.md == staged", failed)


class Safety(unittest.TestCase):
    def test_upload_command_needs_an_approval_hash(self) -> None:
        self.assertIn("./python hf/stage_card.py upload --approved-readme-sha256", sc.upload_command())

    def test_only_hub_write_is_one_pr_commit_on_the_pinned_parent(self) -> None:
        tree = ast.parse((HF / "stage_card.py").read_text())
        writes = ("upload_folder", "upload_file", "create_commit", "create_pull_request", "merge_pull_request",
                  "delete_file", "delete_folder", "create_repo", "update_repo_settings", "create_branch",
                  "delete_repo", "move_repo", "change_discussion_status", "comment_discussion")
        calls = [(fn.name, node) for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef)
                 for node in ast.walk(fn) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        found = [(name, node) for name, node in calls if node.func.attr in writes]
        self.assertEqual([name for name, _ in found], ["create_pr"])
        keywords = {k.arg: k.value for k in found[0][1].keywords}
        self.assertIsInstance(keywords["create_pr"], ast.Constant)
        self.assertIs(keywords["create_pr"].value, True)
        self.assertIn("parent_commit", keywords)
        callers = sorted({name for name, node in calls if node.func.attr == "create_pr"})
        self.assertEqual(callers, ["upload"])

    def test_import_does_not_import_torch(self) -> None:
        code = (f"import sys, importlib.util; spec = importlib.util.spec_from_file_location('s', {str(HF / 'stage_card.py')!r}); "
                "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
                "sys.exit(1 if 'torch' in sys.modules else 0)")
        self.assertEqual(subprocess.run([sys.executable, "-c", code]).returncode, 0)

    def test_staging_root(self) -> None:
        self.assertEqual(sc.STAGING, paths.ARTIFACTS / "hf-staging")
        self.assertEqual(sc.NAME, "rajb3__kea-t.card-update")
        self.assertEqual(sc.REPO_ID, "rajb3/kea-t")


if __name__ == "__main__":
    unittest.main()
