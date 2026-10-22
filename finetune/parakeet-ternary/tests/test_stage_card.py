"""Contract tests of hf/stage_card.py (Kea-T) and the shared engine hf/cardkit.py: the docs-only card
update that adds the INT8x2 runtime results (finetune/parakeet-binary/plans/card-update-x2.md, which
generalizes plans/card-update.md; the contract changes from that plan are listed in its section 9).

CPU only, seconds, no model and no network: the Hub is faked from the release records, the real files
they name and the live documents; origin/main is the local git listing. The real-data test is the
staging run itself (plans/card-update-x2.md section 7).
"""
from __future__ import annotations

import ast
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


sc = _load("kea_t_stage_card", HF / "stage_card.py")
cardkit = sc.cardkit
cardfake = _load("cardfake", Path(__file__).resolve().parent / "cardfake.py")
DOCS = ("README.md", "RUNTIME.md", "EVALUATION.md", "TRAINING.md")
STAGED = ("README.md", "RUNTIME.md")
UNSTAGED = ("EVALUATION.md", "TRAINING.md")
BLOCKERS = ["GitHub links point at paths on origin/main",
            "x2 runtime published on origin/main (gated selection, runtime README, measured kernels)"]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mounted() -> bool:
    return Path(paths.ARTIFACTS).is_dir() and sc.STAGE_REPORT.is_file()


def live_documents() -> dict[str, str]:
    """The live documents: the templates at OLD_CARD_COMMIT, rendered for rajb3/kea-t."""
    templates = {rel: subprocess.run(["git", "-C", str(paths.REPO), "show",
                                      f"{sc.OLD_CARD_COMMIT}:finetune/parakeet-ternary/hf/{name}"],
                                     check=True, capture_output=True, text=True).stdout
                 for rel, name in sc.TEMPLATES.items()}
    return sc.render(templates, sc.load_sources())


def fake_hub(live: dict[str, str]) -> cardfake.FakeHub:
    report = json.loads(sc.STAGE_REPORT.read_text())
    export = Path(report["export"])
    small = {"NOTICE": (HF / "NOTICE").read_bytes(), "LICENSE": (HF / "LICENSE").read_bytes(),
             "load_ternary.py": (HF / "load_ternary.py").read_bytes(),
             **{rel: (export / rel).read_bytes() for rel in ("manifest.json", "reconstruction.json",
                                                              "tokenizer/tokenizer.vocab", "tokenizer/vocab.txt")},
             **{rel: text.encode() for rel, text in live.items()}}
    lfs = {rel: (report["files"][rel]["bytes"], report["files"][rel]["sha256"]) for rel in sc.LFS_FILES}
    kea_b = json.loads(sc.KEA_B_STAGE_REPORT.read_text())["files"]["export.safetensors"]
    siblings = {sc.KEA_B_REPO: {"export.safetensors": cardkit.FileMeta(kea_b["bytes"], kea_b["sha256"], "blob-kea-b")}}
    return cardfake.FakeHub(sc.PINNED_REVISION, small, lfs, siblings)


@unittest.skipUnless(mounted(), "needs /mnt/hd and the release records")
class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sources = sc.load_sources()
        cls.templates = sc.load_templates()
        cls.live = live_documents()
        cls.main_files = sc.origin_main_files(paths.REPO)
        cls.pub_sources, cls.pub_templates, cls.pub_main = cardfake.published(cls.sources, cls.templates, cls.main_files)
        cls.unpub_sources, cls.unpub_main = cardfake.unpublished(cls.sources, cls.main_files)

    def setUp(self) -> None:
        self.hub = fake_hub(self.live)
        cfg = dataclasses.replace(sc.config(), gitattributes_sha256=sha(cardfake.GITATTRIBUTES))
        patcher = mock.patch.object(sc, "config", lambda: cfg)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_prepare(self, templates=None, sources=None, hub=None, main_files=None, published=True):
        """prepare() on the published variant (x2 PR merged, TODO markers replaced) unless published=False."""
        base_t, base_s, base_m = ((self.pub_templates, self.pub_sources, self.pub_main) if published
                                  else (self.templates, self.unpub_sources, self.unpub_main))
        checks, info = {}, {}
        docs = sc.prepare(hub or self.hub, base_m if main_files is None else main_files,
                          base_s if sources is None else sources, base_t if templates is None else templates,
                          checks, info)
        return docs, checks, info

    def failed(self, checks: dict) -> list[str]:
        return sorted(k for k, ok in checks.items() if ok is not True)

    def tampered(self, rel: str, old: str, new: str, count: int = 1) -> dict:
        templates = dict(self.pub_templates)
        self.assertIn(old, templates[rel], f"{old!r} not in {rel}")
        templates[rel] = templates[rel].replace(old, new, count)
        return templates


class RealDocuments(Base):
    def test_live_documents_are_the_old_templates_rendered(self) -> None:
        self.assertEqual({rel: sha(text.encode()) for rel, text in self.live.items()}, sc.LIVE_DOCS)

    def test_real_documents_pass_every_check_on_the_real_origin_main(self) -> None:
        """After PR #6 merged: the real sources and the real origin/main listing, every check passes."""
        checks, info = {}, {}
        docs = sc.prepare(self.hub, self.main_files, self.sources, self.templates, checks, info)
        self.assertEqual(self.failed(checks), [])

    def test_real_documents_fail_only_on_the_unpublished_x2_runtime(self) -> None:
        """Until the x2 thread's PR merges: the unresolved research-runtime README link and the publication
        check, nothing else."""
        docs, checks, _ = self.run_prepare(published=False)
        self.assertEqual(self.failed(checks), BLOCKERS)
        self.assertEqual(set(checks) | sc.staging_checks(), sc.required_checks())
        self.assertEqual(sorted(docs), sorted(STAGED))

    def test_real_documents_pass_once_the_x2_runtime_is_published(self) -> None:
        docs, checks, _ = self.run_prepare()
        self.assertEqual(self.failed(checks), [])
        self.assertEqual(sorted(docs), sorted(STAGED))

    def test_published_link_must_be_on_origin_main(self) -> None:
        _, checks, _ = self.run_prepare(main_files=self.unpub_main)
        self.assertEqual(self.failed(checks), ["GitHub links point at paths on origin/main"])

    def test_card_is_short_and_leads_with_the_runnable_runtime(self) -> None:
        """The glance stays on the published, runnable x3 runtime; the x2 research runtime is one sentence."""
        card = sc.render(self.templates, self.sources)["README.md"]
        body = card.split("\n---\n", 1)[1]
        self.assertLessEqual(len(body.strip().splitlines()), 135)
        headings = [line for line in body.splitlines() if line.startswith("## ")]
        self.assertEqual(headings[:2], ["## At a glance", "## Results"])
        glance = " ".join(body.split("## At a glance", 1)[1].split("## Results", 1)[0].split())
        for s in ("**180.8 MB**", "2,472 MB", "6.84%", "6.45%", "**4.46 ms**", "5.81 ms", "23.2% lower",
                  "**1.57 J**", "1.90 J", "17.1% less", "1.89 GB", "2.18 GB", "custom CUDA runtime",
                  "not `load_ternary.py`", "RUNTIME.md",
                  "Our research runtime with two-part INT8 activations is faster still: at 10 s, 4.30 ms (25.9% lower "
                  "than the BF16 build) and about 1.33 J (about 30% less), measured in a later session, so the energy "
                  "comparison is approximate ([details](RUNTIME.md#research-runtime-with-int8x2-activations))."):
            self.assertIn(s, glance)
        self.assertNotIn("hybrid", body)
        self.assertNotIn("gpu1bit", body)
        results = " ".join(body.split("## Results", 1)[1].split("## How it was trained", 1)[0].split())
        self.assertIn("on our development sets, both GPU runtimes give the same mean WER as FP32 decoding (6.09%)",
                      results)
        for heading in ("## How it was trained", "## Use", "## Files", "## Limitations", "## Used in",
                        "## License and attribution"):
            self.assertIn(heading, headings)
        self.assertIn("LocalFlow", body)
        intro = body.split("## At a glance", 1)[0]
        self.assertIn("Its 1-bit sibling [Kea-B](https://huggingface.co/rajb3/kea-b) is 105.3 MB with a 7.05% "
                      "test mean.", " ".join(intro.split()))
        self.assertIn("hf download rajb3/kea-t --local-dir kea-t", body)
        license_text = " ".join(body.split("## License and attribution", 1)[1].split())
        self.assertIn("this model card and its detail documents (`RUNTIME.md`, `EVALUATION.md`, `TRAINING.md`) "
                      "are licensed under", license_text)
        self.assertIn("the remaining learned weights are stored in FP16", license_text)

    def test_runtime_doc_distinguishes_x2_and_x3(self) -> None:
        runtime = " ".join(sc.render(self.templates, self.sources)["RUNTIME.md"].split())
        for s in ('each activation split into three residual INT8 components ("x3")',
                  'Our research runtime uses two components ("x2")',
                  "Speed measured with our research runtime on an RTX 5090; the runtime code is published for "
                  "reference and is not packaged for general use.",
                  "It runs on SM120 GPUs (RTX 5090) only and was measured with torch 2.11.0+cu128, triton 3.6.0, "
                  "nemo_toolkit 3.0.0 and Python 3.13.",
                  "| Full transcription | 10.1% less (24 of 24) | 3.0% lower (24 of 24) |",
                  "The test sets were not re-scored with the runtime"):
            self.assertIn(s, runtime)
        self.assertNotIn("hybrid", runtime)
        self.assertNotIn("Peak GPU memory", runtime)
        self.assertLess(runtime.index("## Results"), runtime.index("## Research runtime with INT8x2 activations"))

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
    def assert_fails(self, templates, *names, **kwargs) -> None:
        _, checks, _ = self.run_prepare(templates=templates, **kwargs)
        failed = self.failed(checks)
        for name in names:
            self.assertIn(name, failed)

    def test_reversed_sign_fails(self) -> None:
        """Codex r1 finding 3: a sign added in front of a checked number."""
        self.assert_fails(self.tampered("RUNTIME.md", "has 63.1% lower latency", "has -63.1% lower latency"),
                          "RUNTIME.md computed numbers == sources", "RUNTIME.md every number is checked")
        self.assert_fails(self.tampered("RUNTIME.md", "  23.2% lower latency, 13.5%", "  +23.2% lower latency, 13.5%"),
                          "RUNTIME.md computed numbers == sources", "RUNTIME.md every number is checked")
        self.assert_fails(self.tampered("RUNTIME.md", "x2 has 25.9% lower latency", "x2 has -25.9% lower latency"),
                          "RUNTIME.md computed numbers == sources", "RUNTIME.md every number is checked")

    def test_limitations_delta_sign_swap_fails(self) -> None:
        self.assert_fails(self.tampered("README.md", "Common Voice (+4.06 points)", "Common Voice (-4.06 points)"),
                          "README.md computed numbers == sources", "README.md every number is checked")

    def test_changed_word_next_to_number_fails(self) -> None:
        self.assert_fails(self.tampered("RUNTIME.md", "17.1% less GPU", "17.1% more GPU"),
                          "RUNTIME.md computed numbers == sources")
        self.assert_fails(self.tampered("README.md", "(about 30% less)", "(about 30% more)"),
                          "README.md computed numbers == sources")

    def test_removed_duration_qualifier_fails(self) -> None:
        """Codex r1 finding 1: duration comparisons must name their duration."""
        self.assert_fails(self.tampered("RUNTIME.md", "At 10 s, our BF16 build also has", "Our BF16 build also has"),
                          "RUNTIME.md computed numbers == sources")
        self.assert_fails(self.tampered("RUNTIME.md", "At 10 s, compared with our BF16\n  build, the ternary rows",
                                        "Compared with our BF16\n  build, the ternary rows"),
                          "RUNTIME.md computed numbers == sources")
        self.assert_fails(self.tampered("RUNTIME.md", "At 10 s, against our BF16 build, x2 has",
                                        "Against our BF16 build, x2 has"),
                          "RUNTIME.md computed numbers == sources")

    def test_approximate_energy_qualifiers_are_checked(self) -> None:
        self.assert_fails(self.tampered("README.md", "(about 30% less)", "(30% less)"), "README.md computed numbers == sources")
        self.assert_fails(self.tampered("README.md", "energy comparison is approximate", "energy comparison is exact"),
                          "README.md computed numbers == sources")
        self.assert_fails(self.tampered("RUNTIME.md", "uses\nabout 30% less GPU energy", "uses\n30% less GPU energy"),
                          "RUNTIME.md computed numbers == sources")
        self.assert_fails(self.tampered("RUNTIME.md", "within 8% on energy", "within 3% on energy"),
                          "RUNTIME.md computed numbers == sources", "RUNTIME.md every number is checked")

    def test_swapped_x2_x3_label_fails(self) -> None:
        row = "| 10 s | **Kea-T expanded, x2 activations (research runtime)** | 4.30 |"
        self.assert_fails(self.tampered("RUNTIME.md", row, row.replace("4.30", "4.46")),
                          "RUNTIME.md x2 results table == sources")
        swapped = self.tampered("RUNTIME.md", "| 10 s | Kea-T expanded, x3 activations |", "| 10 s | KEEP |")
        swapped["RUNTIME.md"] = swapped["RUNTIME.md"].replace(
            "| 10 s | **Kea-T expanded, x2 activations (research runtime)** |", "| 10 s | Kea-T expanded, x3 activations |"
        ).replace("| 10 s | KEEP |", "| 10 s | **Kea-T expanded, x2 activations (research runtime)** |")
        self.assert_fails(swapped, "RUNTIME.md x2 results table == sources")

    def test_power_sentence_bound_to_10_s_values(self) -> None:
        sources = copy.deepcopy(self.pub_sources)
        sources["rtx"]["variants"]["packed_optimized"]["workloads"][1]["average_watts"] = 300.0
        _, checks, _ = self.run_prepare(sources=sources)
        self.assertIn("RUNTIME.md computed numbers == sources", self.failed(checks))
        sources = copy.deepcopy(self.pub_sources)
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
        self.assert_fails(self.tampered("RUNTIME.md", "6.0853%", "6.0753%"),
                          "RUNTIME.md computed numbers == sources", "RUNTIME.md every number is checked")

    def test_uncovered_number_fails(self) -> None:
        self.assert_fails(self.tampered("README.md", "English only, like the base model.",
                                        "English only, like the base model, and 3x faster."),
                          "README.md every number is checked")

    def test_changed_carried_number_fails(self) -> None:
        self.assert_fails(self.tampered("RUNTIME.md", "medians of three 5 s windows", "medians of three 6 s windows"),
                          "RUNTIME.md every number is checked")

    def test_dropped_live_number_fails(self) -> None:
        templates = self.tampered("RUNTIME.md", "our BF16 build also has 63.1% lower latency than this row",
                                  "our BF16 build also has lower latency than this row")
        self.assert_fails(templates, "every number in the live documents appears in the new documents",
                          "RUNTIME.md computed numbers == sources")

    def test_runtime_table_cell_fails(self) -> None:
        self.assert_fails(self.tampered("RUNTIME.md", "| 10.56 | 2.23 | 3.99 | 376 |", "| 10.56 | 2.23 | 3.98 | 376 |"),
                          "RUNTIME.md results table == inference/README.md table")
        self.assert_fails(self.tampered("RUNTIME.md", "| 10.1% less (24 of 24) |", "| 10.1% less (23 of 24) |"),
                          "RUNTIME.md paired table == report-dev")

    def test_extra_wer_row_fails(self) -> None:
        row = "| **Kea-T (this model)** | 2.05 | 4.20 | 10.42 | 11.72 | 10.35 | 2.94 | 6.19 | 6.84 | 12.56 |"
        self.assert_fails(self.tampered("README.md", row, row + "\n" + row), "card WER table == TEST.md B0 and M1 rows")
        evaluation_row = "| **This model** | 2.05 | 4.20 | 10.42 | 11.72 | 10.35 | 2.94 | 6.19 | 6.84 | 12.56 |"
        self.assert_fails(self.tampered("EVALUATION.md", evaluation_row, evaluation_row.replace("2.05", "2.06")),
                          "EVALUATION.md WER table == all 6 TEST.md rows", "unstaged documents == live hub documents")

    def test_source_tampering_fails(self) -> None:
        def failed_with(mutate) -> list[str]:
            sources = copy.deepcopy(self.pub_sources)
            mutate(sources)
            return self.failed(self.run_prepare(sources=sources)[1])

        def set_(path, value):
            def mutate(s):
                node = s
                for key in path[:-1]:
                    node = node[key]
                node[path[-1]] = value
            return mutate

        self.assertIn("results/TEST.md rows == results/test.json rounded",
                      failed_with(set_(["test", "rows", "M1", "mean"], 6.9)))
        self.assertIn("inference/README.md results table == rtx5090.json rounded",
                      failed_with(set_(["rtx", "variants", "packed_expanded", "workloads", 1, "median_ms"], 4.9)))
        self.assertIn("card at-a-glance table == sources",
                      failed_with(set_(["rtx", "variants", "packed_expanded", "workloads", 1, "median_ms"], 4.9)))
        self.assertIn("README.md computed numbers == sources",
                      failed_with(lambda s: [r.__setitem__("median_ms", 4.9)
                                             for r in s["x2"]["test"]["p_clips"]["M1-x2"]["1320-122617-0010"]]))
        self.assertIn("README.md computed numbers == sources", failed_with(set_(["kea_b_test", "rows", "MB1", "mean"], 7.2)))
        self.assertIn("Kea-B stage report PASSED and its export SHA-256 == Kea-B test.json",
                      failed_with(set_(["kea_b_test", "mb1_weights_sha256"], "0" * 64)))
        self.assertIn("results/test.json export SHA-256 == stage report",
                      failed_with(set_(["test", "m1_export_sha256"], "0" * 64)))
        self.assertIn("previous card update PASSED and verified at the pin",
                      failed_with(set_(["previous_update_main", "status"], "FAILED")))

    def test_number_scan(self) -> None:
        # Signed tokens (contract change in r1: a sign is part of the number).
        self.assertEqual(sc.numbers("2,472 MB, 1.21x, 5.5e-4, ae9ad07 b6bdcd0beb FP32 v2 (+0.36) -63.1% TDT-0.6"),
                         ["2,472", "1.21", "5.5e-4", "+0.36", "-63.1", "0.6"])
        self.assertEqual(sc.numbers("[a 3 b](https://x.org/5/6) ```\n7\n``` 8"), ["3", "8"])


class X2SourceChecks(Base):
    """The shared x2 checks (hf/x2facts.py), on Kea-T's sources."""

    def failed_with(self, mutate) -> list[str]:
        sources = copy.deepcopy(self.pub_sources)
        mutate(sources["x2"])
        return self.failed(self.run_prepare(sources=sources)[1])

    def test_regenerated_report_fails(self) -> None:
        self.assertIn("x2 sources SHA-256 == pinned",
                      self.failed_with(lambda x2: x2["sha256"].__setitem__(str(sc.x2facts.REPORT_TEST), "0" * 64)))

    def test_wrong_export_or_identity_fails(self) -> None:
        name = "x2 arms are the published exports and the gated identities"
        self.assertIn(name, self.failed_with(lambda x2: x2["dev"]["labels"]["M1-x2"].__setitem__("export_sha256", "0" * 64)))
        self.assertIn(name, self.failed_with(lambda x2: x2["gates"]["identities"].__setitem__("g4", "0" * 64)))

    def test_failed_gate_fails(self) -> None:
        self.assertIn("x2 dev gates passed", self.failed_with(lambda x2: x2["gates"]["m1"].__setitem__("pass", False)))

    def test_artifact_and_report_disagree(self) -> None:
        def mutate(x2):
            x2["clips"]["M1-x2", "clips2"]["workloads"][1]["summary"]["median_ms"] = 4.4
        self.assertIn("x2 clip artifacts == report-test", self.failed_with(mutate))
        self.assertIn("x2 clip artifacts == report-test",
                      self.failed_with(lambda x2: x2["clips"]["M1-x2", "clips1"]["gpu"].__setitem__("power_limit_w", 450.0)))

    def test_full_precision_rows_must_match_the_published_bf16_row(self) -> None:
        self.assertIn("full-precision rows: same clips, GPU, driver, power limit and windows",
                      self.failed_with(lambda x2: x2["bf16"]["workloads"][1]["summary"].__setitem__("median_ms", 5.9)))
        self.assertIn("full-precision rows: same clips, GPU, driver, power limit and windows",
                      self.failed_with(lambda x2: x2["fp32"]["gpu"].__setitem__("driver_version", "1")))

    def test_publication_requires_the_gated_selection_on_main(self) -> None:
        self.assertIn(cardfake.x2facts.PUBLISHED_CHECK,
                      self.failed_with(lambda x2: x2["main"].__setitem__("selection_sha256", "0" * 64)))
        self.assertIn(cardfake.x2facts.PUBLISHED_CHECK,
                      self.failed_with(lambda x2: x2["main"].__setitem__("runtime_readme_sha256", None)))
        self.assertIn(cardfake.x2facts.PUBLISHED_CHECK,
                      self.failed_with(lambda x2: x2["main"]["measured_parts"].__setitem__("kernels_parts.py", "9" * 64)))
        self.assertIn(cardfake.x2facts.PUBLISHED_CHECK,
                      self.failed_with(lambda x2: x2["main"].__setitem__("parts_py_sha256", "9" * 64)))
        self.assertIn(cardfake.x2facts.PUBLISHED_CHECK,
                      self.failed_with(lambda x2: x2["main"].__setitem__("measured_parts", {})))

    def test_environment_must_match_the_gate_identities(self) -> None:
        name = "x2 runtime environment == the gate identities"
        self.assertIn(name, self.failed_with(lambda x2: x2["identities"]["M1-x2"]["versions"].__setitem__("nemo_toolkit", "2.9")))
        self.assertIn(name, self.failed_with(lambda x2: x2["staging_env"]["wrappers"].__setitem__(
            "finetune/parakeet-ternary/python", "0" * 64)))
        self.assertIn(name, self.failed_with(lambda x2: x2["staging_env"].__setitem__("executable", "/usr/bin/python3")))
        self.assertIn(name, self.failed_with(lambda x2: x2["staging_env"].__setitem__("parakeet_python_override", "/x/py")))

    def test_rounding_and_ceilings(self) -> None:
        x2 = self.sources["x2"]
        rm = sc.x2facts.remeasure(x2)
        self.assertEqual((rm["latency_within"], rm["energy_within"], rm["expanded_vram_identical"]), (1, 8, True))
        self.assertAlmostEqual(sc.x2facts.test_agreement(x2, "M1"), 0.4)
        self.assertAlmostEqual(sc.x2facts.test_agreement(x2, "MB1"), 0.6)
        self.assertAlmostEqual(sc.x2facts.ceil_to(0.30000000000000004, 0.1), 0.3)  # float noise is not rounded up
        self.assertAlmostEqual(sc.x2facts.ceil_to(0.301, 0.1), 0.4)


class RenderAndLinkChecks(Base):
    def assert_fails(self, templates, name, **kwargs) -> None:
        _, checks, _ = self.run_prepare(templates=templates, **kwargs)
        self.assertIn(name, self.failed(checks))

    def test_placeholder_or_todo_fails(self) -> None:
        name = "no unrendered placeholder or TODO marker"
        self.assert_fails(self.tampered("RUNTIME.md", "Detail for the", "Detail ({repo_slug}) for the"), name)
        self.assert_fails(self.tampered("README.md", "## Use", "## Use\n\nTODO: usage."), name)

    def test_front_matter_change_fails(self) -> None:
        self.assert_fails(self.tampered("README.md", "value: 4.20", "value: 4.21"), "card front matter == live card front matter")
        self.assert_fails(self.tampered("TRAINING.md", "# Kea-T: training", "---\nlicense: mit\n---\n# Kea-T: training"),
                          "detail documents have no front matter")

    def test_unstaged_document_must_equal_the_hub(self) -> None:
        self.assert_fails(self.tampered("TRAINING.md", "seed", "Seed", count=1), "unstaged documents == live hub documents")

    def test_broken_relative_links_fail(self) -> None:
        name = "relative links resolve to documents and headings"
        self.assert_fails(self.tampered("README.md", "(EVALUATION.md#limitations)", "(EVALUATION.md#nope)"), name)
        self.assert_fails(self.tampered("README.md", "(RUNTIME.md)", "(RUNTIMES.md)"), name)
        self.assert_fails(self.tampered("RUNTIME.md", "(#method-and-comparing-across-sessions)", "(#comparing-sessions)"), name)

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
        files = set(self.pub_main) - {"finetune/parakeet-ternary/results/TEST.md"}
        self.assert_fails(None, "GitHub links point at paths on origin/main", main_files=files)

    def test_unknown_external_url_fails(self) -> None:
        self.assert_fails(self.tampered("README.md", "(https://github.com/zachlatta/freeflow)",
                                        "(https://github.com/zachlatta/freeflow2)"),
                          "other absolute URLs appear in the live documents or are allowed")

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
        self.assert_fails("hub main revision == pinned c0f2896")

    def test_changed_lfs_file_fails(self) -> None:
        meta = self.hub.meta["export.safetensors"]
        self.hub.meta["export.safetensors"] = cardkit.FileMeta(meta.size, "0" * 64, meta.blob_id)
        self.assert_fails("hub export.safetensors SHA-256 and size == release record")

    def test_changed_small_file_fails(self) -> None:
        self.hub.put("load_ternary.py", self.hub.content["load_ternary.py"] + b"\n")
        self.assert_fails("hub load_ternary.py SHA-256 and size == release record")

    def test_changed_notice_fails(self) -> None:
        self.hub.put("NOTICE", self.hub.content["NOTICE"].replace(b"CC0-1.0", b"CC-BY-4.0"))
        _, checks, _ = self.run_prepare()
        self.assertIn("hub NOTICE SHA-256 and size == release record", self.failed(checks))
        self.assertIn("hub NOTICE differs from the stage report's NOTICE only in its first line", self.failed(checks))

    def test_changed_live_document_fails(self) -> None:
        for rel in DOCS:
            self.hub = fake_hub(self.live)
            self.hub.put(rel, self.live[rel].encode() + b"\n")
            self.assert_fails(f"hub {rel} SHA-256 == pinned live document")

    def test_kea_b_private_or_changed_fails(self) -> None:
        name = "hub rajb3/kea-b is public; export.safetensors LFS SHA-256 and size == Kea-B stage report"
        self.hub.siblings[sc.KEA_B_REPO] = None
        self.assert_fails(name)
        self.hub = fake_hub(self.live)
        meta = self.hub.siblings[sc.KEA_B_REPO]["export.safetensors"]
        self.hub.siblings[sc.KEA_B_REPO]["export.safetensors"] = cardkit.FileMeta(meta.size + 1, meta.lfs_sha256, meta.blob_id)
        self.assert_fails(name)

    def test_extra_file_fails(self) -> None:
        self.hub.put("BENCHMARKS.md", b"# not ours\n")
        self.assert_fails("hub file set == pinned file set")


class Staging(Base):
    def setUp(self) -> None:
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.staging = Path(self.tmp.name)
        self.final = self.staging / sc.NAME
        self.ok_report = self.staging / f"{sc.NAME}.stage.json"
        self.failed_report = self.staging / f"{sc.NAME}.stage.FAILED.json"

    def real_prepare(self, published=True):
        t, s, m = ((self.pub_templates, self.pub_sources, self.pub_main) if published
                   else (self.templates, self.unpub_sources, self.unpub_main))
        return lambda checks, info: sc.prepare(self.hub, m, s, t, checks, info)

    def fake_prepare(self, fail=None, error=None, extra=None, drop=None, value=True):
        def run(checks, info):
            if error is not None:
                raise error
            for name in sc.required_checks() - sc.staging_checks():
                if name != drop:
                    checks[name] = value if name == fail and value is not True else name != fail
            if extra:
                checks[extra] = True
            return {rel: f"# {rel}\n" for rel in STAGED}
        return run

    def leftovers(self) -> list[str]:
        return sorted(p.name for p in self.staging.iterdir() if p.name.startswith("."))

    def test_real_staging_promotes_exactly_the_changed_documents(self) -> None:
        ok, report, path = sc.stage(self.staging, self.real_prepare())
        self.assertTrue(ok, report.get("failed_checks"))
        self.assertEqual(path, self.ok_report)
        self.assertEqual(sorted(p.relative_to(self.final).as_posix() for p in self.final.rglob("*") if p.is_file()),
                         sorted(STAGED))
        saved = json.loads(self.ok_report.read_text())
        self.assertEqual(saved["status"], "PASSED")
        self.assertEqual(set(saved["checks"]), sc.required_checks())
        for rel in STAGED:
            self.assertEqual(saved["files"][rel]["sha256"], sha((self.final / rel).read_bytes()))
        self.assertIn("hf/stage_card.py upload --approved-bundle-sha256", saved["upload_command"])
        self.assertEqual(saved["bundle_sha256"], cardkit.bundle_sha256(sc.REPO_ID, sc.PINNED_REVISION, saved["files"]))
        self.assertEqual(saved["bundle_manifest"], cardkit.bundle_manifest(sc.REPO_ID, sc.PINNED_REVISION, saved["files"]))
        self.assertFalse(self.failed_report.exists())
        self.assertEqual(self.leftovers(), [])

    def test_unpublished_x2_runtime_stages_nothing(self) -> None:
        ok, report, path = sc.stage(self.staging, self.real_prepare(published=False))
        self.assertFalse(ok)
        self.assertEqual(path, self.failed_report)
        self.assertEqual(report["failed_checks"], BLOCKERS)
        self.assertFalse(self.final.exists())
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
        for kwargs in ({"drop": "carried phrases appear in the live documents"}, {"extra": "an unplanned check"}):
            ok, report, _ = sc.stage(self.staging, self.fake_prepare(**kwargs))
            self.assertFalse(ok, kwargs)
            self.assertIn("check set mismatch", report["error"])
            self.assertFalse(self.final.exists())

    def test_non_boolean_true_is_a_failure(self) -> None:
        ok, _, _ = sc.stage(self.staging, self.fake_prepare(fail="no unrendered placeholder or TODO marker", value=1))
        self.assertFalse(ok)
        self.assertFalse(self.final.exists())

    def test_stale_outputs_removed_even_on_failure(self) -> None:
        self.final.mkdir()
        (self.final / "README.md").write_text("stale")
        self.ok_report.write_text("{}")
        (self.staging / f".{sc.NAME}.tmp-1").mkdir()
        ok, _, _ = sc.stage(self.staging, self.fake_prepare(fail="no unrendered placeholder or TODO marker"))
        self.assertFalse(ok)
        self.assertFalse(self.final.exists())
        self.assertFalse(self.ok_report.exists())
        self.assertEqual(self.leftovers(), [])


class Publish(Base):
    """upload, verify refs/pr/N, pre-merge and verify main (plans/card-update-x2.md section 5)."""

    def setUp(self) -> None:
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        ok, self.report, _ = sc.stage(Path(self.tmp.name),
                                      lambda c, i: sc.prepare(self.hub, self.pub_main, self.pub_sources,
                                                              self.pub_templates, c, i))
        self.assertTrue(ok, self.report.get("failed_checks"))
        self.folder = Path(self.report["folder"])
        self.approved = self.report["bundle_sha256"]

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
        self.assertEqual(call["files"], {rel: (self.folder / rel).read_bytes() for rel in STAGED})
        self.assertEqual(call["message"], sc.COMMIT_MESSAGE)
        self.assertEqual(pr["pr_revision"], "refs/pr/1")

    def test_upload_refuses_without_writing(self) -> None:
        cases = {
            "card-update stage report PASSED": lambda: self.run_upload(report=dict(self.report, status="FAILED")),
            "approved bundle SHA-256 == bundle of the bytes to upload": lambda: self.run_upload(approved="0" * 64),
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

    def test_old_approval_fails_after_a_runtime_only_restage(self) -> None:
        """Codex card-update-x2 r1 major: approve a bundle, restage with only RUNTIME.md changed (README.md
        unchanged); the old approval fails with no write, the new bundle's approval passes."""
        old_bundle, old_readme = self.report["bundle_sha256"], self.report["files"]["README.md"]["sha256"]
        templates = dict(self.pub_templates)
        templates["RUNTIME.md"] = templates["RUNTIME.md"].replace("Detail for the", "Further detail for the", 1)
        ok, report, _ = sc.stage(Path(self.tmp.name), lambda c, i: sc.prepare(self.hub, self.pub_main, self.pub_sources,
                                                                               templates, c, i))
        self.assertTrue(ok, report.get("failed_checks"))
        self.assertEqual(report["files"]["README.md"]["sha256"], old_readme)
        self.assertNotEqual(report["bundle_sha256"], old_bundle)
        self.report = report
        pr, failed = self.run_upload(approved=old_bundle)
        self.assertIsNone(pr)
        self.assertEqual(failed, ["approved bundle SHA-256 == bundle of the bytes to upload"])
        pr, failed = self.run_upload(approved=old_readme)
        self.assertIsNone(pr)
        self.assertEqual(self.hub.created, [])
        pr, failed = self.run_upload(approved=report["bundle_sha256"])
        self.assertEqual(failed, [])
        self.assertEqual(len(self.hub.created), 1)

    def test_approval_is_checked_against_the_bytes_not_the_report(self) -> None:
        """A report whose recorded bundle was rewritten to match an edited folder still fails: the digest is
        recomputed from the bytes about to be uploaded and must also equal the report's."""
        (self.folder / "RUNTIME.md").write_text("# edited\n")
        pr, failed = self.run_upload(report=dict(self.report, bundle_sha256="0" * 64), approved="0" * 64)
        self.assertIsNone(pr)
        self.assertIn("approved bundle SHA-256 == bundle of the bytes to upload", failed)
        self.assertEqual(self.hub.created, [])

    def test_upload_refuses_after_a_main_edit(self) -> None:
        self.hub.edit_main("README.md", self.live["README.md"].replace("6.84%", "6.85%").encode())
        pr, failed = self.run_upload()
        self.assertIsNone(pr)
        self.assertIn("main is still the pinned revision", failed)
        self.assertEqual(self.hub.created, [])

    def test_upload_refuses_an_extra_staged_file(self) -> None:
        (self.folder / "EVALUATION.md").write_text("x\n")
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
        self.hub.edit_main("README.md", self.live["README.md"].replace("6.84%", "6.85%").encode())
        failed, _ = self.run_verify()
        self.assertIn("main is still the pinned revision", failed)

    def test_pr_based_on_a_newer_main_fails_verify(self) -> None:
        edited = self.hub.edit_main("README.md", self.live["README.md"].replace("6.84%", "6.85%").encode())
        self.hub.create_pr({rel: (self.folder / rel).read_bytes() for rel in STAGED}, edited, "cli-style")
        failed, _ = self.run_verify()
        self.assertIn("main is still the pinned revision", failed)
        self.assertIn("revision is one commit on the pinned revision", failed)

    def test_changed_non_doc_or_unstaged_doc_fails(self) -> None:
        self.uploaded()
        self.hub.put("NOTICE", b"changed\n", revision="refs/pr/1")
        self.hub.put("EVALUATION.md", b"# changed\n", revision="refs/pr/1")
        failed, _ = self.run_verify()
        self.assertIn("revision NOTICE unchanged from pinned", failed)
        self.assertIn("revision EVALUATION.md unchanged from pinned", failed)

    def test_doc_mismatch_or_missing_fails(self) -> None:
        self.uploaded()
        self.hub.put("RUNTIME.md", b"# other\n", revision="refs/pr/1")
        self.assertIn("revision RUNTIME.md == staged", self.run_verify()[0])
        del self.hub.revisions["refs/pr/1"][0]["TRAINING.md"]
        failed, _ = self.run_verify()
        self.assertIn("revision file set == pinned files + new documents", failed)
        self.assertIn("revision TRAINING.md unchanged from pinned", failed)

    def test_extra_pr_commit_fails_verify(self) -> None:
        self.uploaded()
        self.hub.add_pr_commit("refs/pr/1", "RUNTIME.md", self.folder.joinpath("RUNTIME.md").read_bytes())
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
            self.hub = fake_hub(self.live)
            self.uploaded()
            pr_head = self.verified()["verify"]["revision_sha"]
            self.hub.merge("refs/pr/1", merge_commit=merge_commit)
            failed, _ = self.run_verify("main", pr_head=pr_head)
            self.assertEqual(failed, [], merge_commit)

    def test_main_with_an_intervening_commit_fails(self) -> None:
        """Codex r1's scenario end to end: a README correction on main before the merge is overwritten by
        the merge; the content checks alone pass, the history check does not."""
        self.uploaded()
        pr_head = self.verified()["verify"]["revision_sha"]
        self.hub.edit_main("README.md", self.live["README.md"].replace("6.84%", "6.85%").encode())
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
    WRITES = ("upload_folder", "upload_file", "create_commit", "create_pull_request", "merge_pull_request",
              "delete_file", "delete_folder", "create_repo", "update_repo_settings", "create_branch",
              "delete_repo", "move_repo", "change_discussion_status", "comment_discussion")

    def write_calls(self, path: Path) -> list[tuple[str, ast.Call]]:
        tree = ast.parse(path.read_text())
        calls = [(fn.name, node) for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef)
                 for node in ast.walk(fn) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        return [(name, node) for name, node in calls if node.func.attr in self.WRITES], calls

    def test_upload_command_needs_an_approval_hash(self) -> None:
        self.assertIn("./python hf/stage_card.py upload --approved-bundle-sha256", sc.upload_command())

    def test_bundle_manifest_is_canonical(self) -> None:
        """Codex card-update-x2 r1: the approval token covers the repository, the parent and every uploaded
        file's name, SHA-256 and size, in sorted order, independent of dict order."""
        files = {"RUNTIME.md": {"sha256": "b" * 64, "bytes": 2}, "README.md": {"sha256": "a" * 64, "bytes": 1}}
        manifest = cardkit.bundle_manifest("rajb3/kea-t", "c" * 40, files)
        self.assertEqual(manifest, "repo rajb3/kea-t\nparent " + "c" * 40 + "\nfile README.md " + "a" * 64
                         + " 1\nfile RUNTIME.md " + "b" * 64 + " 2\n")
        same = cardkit.bundle_sha256("rajb3/kea-t", "c" * 40, dict(reversed(list(files.items()))))
        self.assertEqual(cardkit.bundle_sha256("rajb3/kea-t", "c" * 40, files), same)
        for other in (cardkit.bundle_sha256("rajb3/kea-b", "c" * 40, files),
                      cardkit.bundle_sha256("rajb3/kea-t", "d" * 40, files),
                      cardkit.bundle_sha256("rajb3/kea-t", "c" * 40, {**files, "RUNTIME.md": {"sha256": "b" * 64, "bytes": 3}})):
            self.assertNotEqual(other, same)

    def test_only_hub_write_is_one_pr_commit_on_the_pinned_parent(self) -> None:
        found, calls = self.write_calls(HF / "cardkit.py")
        self.assertEqual([name for name, _ in found], ["create_pr"])
        keywords = {k.arg: k.value for k in found[0][1].keywords}
        self.assertIsInstance(keywords["create_pr"], ast.Constant)
        self.assertIs(keywords["create_pr"].value, True)
        self.assertIn("parent_commit", keywords)
        callers = sorted({name for name, node in calls if node.func.attr == "create_pr"})
        self.assertEqual(callers, ["upload"])
        for script in (HF / "stage_card.py", HF / "x2facts.py", PKG.parent / "parakeet-binary" / "hf" / "stage_card.py"):
            found, calls = self.write_calls(script)
            self.assertEqual(found, [], script)
            self.assertFalse([n for _, n in calls if n.func.attr == "create_pr"], script)

    def test_import_does_not_import_torch(self) -> None:
        code = (f"import sys, importlib.util; spec = importlib.util.spec_from_file_location('s', {str(HF / 'stage_card.py')!r}); "
                "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
                "sys.exit(1 if 'torch' in sys.modules else 0)")
        self.assertEqual(subprocess.run([sys.executable, "-c", code]).returncode, 0)

    def test_staging_root(self) -> None:
        self.assertEqual(sc.STAGING, paths.ARTIFACTS / "hf-staging")
        self.assertEqual(sc.NAME, "rajb3__kea-t.card-update-x2")
        self.assertEqual(sc.config().name, sc.NAME)
        self.assertEqual(sc.REPO_ID, "rajb3/kea-t")
        self.assertEqual(sc.PINNED_REVISION, "c0f28965998b0bbeef45cd6e6ab8aaeaf6bd0a79")


if __name__ == "__main__":
    unittest.main()
