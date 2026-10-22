"""Stage the docs-only update of the Kea-B model card (rajb3/kea-b) that adds the INT8x2 runtime results.
Never uploads by itself.

    ./python hf/stage_card.py stage                                   # from finetune/parakeet-binary
    ./python hf/stage_card.py upload --approved-bundle-sha256 HEX     # AJ, after review: opens the Hub PR
    ./python hf/stage_card.py verify refs/pr/N
    ./python hf/stage_card.py pre-merge refs/pr/N                     # immediately before AJ merges
    ./python hf/stage_card.py verify main                             # after the merge

The contract is plans/card-update-x2.md; the engine is finetune/parakeet-ternary/hf/cardkit.py and the
shared x2 sources finetune/parakeet-ternary/hf/x2facts.py. This file is Kea-B's configuration: the pins
(main 5e7c298, the 2026-10-21 release), the release record of every unchanged file (hf/stage.py's
report), the source checks, the checked tables and the computed phrases. The update replaces README.md
and adds RUNTIME.md; no other Hub file is staged, so none can change. CPU only, so it does not need
./heavy. hf/stage.py stays the record of the full release and is not rerun.
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

import binary_paths
import paths

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
TERNARY_HF = PKG.parent / "parakeet-ternary" / "hf"
sys.path.insert(0, str(TERNARY_HF))
sys.path.insert(0, str(HERE))
import cardkit  # noqa: E402
import stage as release  # noqa: E402  -- the 2026-10-21 release kit: its card facts and constants
import x2facts  # noqa: E402
from cardkit import Table, read_json, row, sha  # noqa: E402

REPO_ID = "rajb3/kea-b"
NAME = REPO_ID.replace("/", "__") + ".card-update-x2"
STAGING = binary_paths.ARTIFACTS / "hf-staging"
# The live Hub state this update is written against: the 2026-10-21 release (plans/release.md section 9).
PINNED_REVISION = "5e7c298bc0eee174394a3fba88d789dc7c53ee4d"
OLD_CARD_COMMIT = "ca4bc02a7f8d11b2710b309336433548082ea3de"  # hf/MODEL_CARD.md that renders to the live README
LIVE_DOCS = {"README.md": "48b6a7da9ac62afb6ab3de25c99af42a77c9d7757b973d292858638ede463e91"}
STAGE_REPORT = STAGING / "rajb3__kea-b.stage.json"  # the 2026-10-21 release
GITATTRIBUTES_SHA256 = "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361"  # the Hub default
# Kea-T, the ternary sibling the card links and quotes (its 2026-10-04 release record).
KEA_T_REPO = "rajb3/kea-t"
KEA_T_URL = f"https://huggingface.co/{KEA_T_REPO}"
KEA_T_STAGE_REPORT = paths.ARTIFACTS / "hf-staging" / "rajb3__parakeet-tdt-0.6b-v2-ternary.stage.json"

TEMPLATES = {"README.md": "MODEL_CARD.md", "RUNTIME.md": "RUNTIME.md"}
DOCS = tuple(TEMPLATES)
STAGED = DOCS
LFS_FILES = ("export.safetensors", "tokenizer/tokenizer.model")
COLUMNS = [*paths.MEAN_SETS, "mean", "common_voice"]
WER_HEADER = release.CARD_HEADER
GLANCE_HEADER = "| | Kea-B | Original |"
X2_LABELS = {"x3": "Kea-B, x3 activations", "x2": "**Kea-B, x2 activations (current)**"}
# Phrases with numbers carried verbatim from the live card; each must appear in it.
CARRIED = ("Parakeet-TDT-0.6B-v2", "parakeet-tdt-0.6b-v2", "(-1, 0, +1)", "+1 or -1", "1 = +1", "1-bit",
           "pieces of about 30 s", "on one RTX 5090.", "on an RTX 5090 in FP32", "CC-BY-4.0", "SHA-256")
# Numbers of the live card deliberately absent from the new documents (none).
DROPPED: tuple[str, ...] = ()

HUB_EXTRA_CHECKS = frozenset({
    "hub rajb3/kea-t is public; export.safetensors LFS SHA-256 and size == Kea-T release record",
})
SOURCE_CHECKS = frozenset({
    "Kea-B stage report PASSED and is the pinned release",
    "export manifest.json and run summary.json == the release",
    "results/test.json MB1 weights SHA-256 == stage report",
    "results/TEST.md B0, M1 and MB1 rows == results/test.json rounded",
    "Kea-B test.json Kea-T export SHA-256 == Kea-T release record",
    *x2facts.CHECKS,
})
TABLE_CHECKS = frozenset({
    "card at-a-glance table == sources",
    "card WER table == TEST.md B0, M1 and MB1 rows",
    "RUNTIME.md x2 results table == sources",
    "RUNTIME.md paired table == report-dev",
})
CARD_YAML_CHECK = "card YAML: model_name kea-b, license cc-by-4.0, base_model, library_name nemo"
COMMIT_MESSAGE = ("Model card: add the research runtime's speed and energy at a glance; add RUNTIME.md")
UPLOAD_COMMAND = ("./python hf/stage_card.py upload --approved-bundle-sha256 <the bundle_sha256 AJ "
                  "approved>   # from finetune/parakeet-binary")


# ----------------------------------------------------------------------------- inputs

def load_sources() -> dict:
    """Every source a claim is checked against (plans/release.md section 5, plans/card-update-x2.md
    section 6)."""
    return {
        "stage_report": read_json(STAGE_REPORT),
        "kea_t_report": read_json(KEA_T_STAGE_REPORT),
        "test": read_json(release.TEST_JSON),
        "test_md": release.TEST_MD.read_text(),
        "summary": read_json(release.RUN_SUMMARY),
        "manifest": read_json(release.DEFAULT_EXPORT / release.MANIFEST),
        "manifest_sha256": sha((release.DEFAULT_EXPORT / release.MANIFEST).read_bytes()),
        "stream_sources": read_json(release.SOURCES_JSON),
        "x2": x2facts.load(paths.REPO),
    }


def load_templates() -> dict[str, str]:
    return {rel: (HERE / name).read_text() for rel, name in TEMPLATES.items()}


def origin_main_files(repo: Path) -> set[str]:
    return cardkit.origin_main_files(repo)


def render(templates: dict[str, str], sources: dict) -> dict[str, str]:
    return {rel: release.render_card(text, REPO_ID) for rel, text in templates.items()}


# ----------------------------------------------------------------------------- A. Hub (Kea-B's own)

def hub_extra(hub, sources: dict, files: dict, checks: dict, info: dict) -> None:
    kea_t = hub.public_files(KEA_T_REPO)
    want = sources["kea_t_report"]["files"]["export.safetensors"]
    got = (kea_t or {}).get("export.safetensors")
    checks["hub rajb3/kea-t is public; export.safetensors LFS SHA-256 and size == Kea-T release record"] = (
        got is not None and got.lfs_sha256 == want["sha256"] and got.size == want["bytes"])
    info["hub"]["kea_t_export"] = got._asdict() if got is not None else None


# ----------------------------------------------------------------------------- B. sources

def test_md_rows(test_md: str) -> dict[str, list[str]]:
    header, rows = release.first_table(test_md)
    return rows if header[1:] == COLUMNS else {}


def check_sources(sources: dict, checks: dict, info: dict) -> None:
    rep, test, manifest, summary = sources["stage_report"], sources["test"], sources["manifest"], sources["summary"]
    files = rep.get("files", {})
    export_sha = files.get("export.safetensors", {}).get("sha256")
    checks["Kea-B stage report PASSED and is the pinned release"] = (
        rep.get("status") == "PASSED" and rep.get("repo_id") == REPO_ID and bool(rep.get("checks"))
        and all(v is True for v in rep["checks"].values())
        and rep.get("checks_passed") == f"{len(rep['checks'])}/{len(rep['checks'])}"
        and files.get("README.md", {}).get("sha256") == LIVE_DOCS["README.md"]
        and export_sha == release.WEIGHTS_SHA256 == x2facts.EXPORT_SHA["MB1"])
    checks["export manifest.json and run summary.json == the release"] = (
        sources["manifest_sha256"] == files.get("manifest.json", {}).get("sha256") == release.MANIFEST_SHA256
        and manifest.get("sha256") == export_sha
        and summary.get("export_bytes") == manifest.get("bytes") and summary.get("run_name") == release.RUN_NAME)
    checks["results/test.json MB1 weights SHA-256 == stage report"] = test.get("mb1_weights_sha256") == export_sha
    md = test_md_rows(sources["test_md"])
    checks["results/TEST.md B0, M1 and MB1 rows == results/test.json rounded"] = all(
        md.get(arm) == release.rounded_row(test["rows"][arm], COLUMNS) for arm in release.TABLE_ARMS)
    checks["Kea-B test.json Kea-T export SHA-256 == Kea-T release record"] = (
        test.get("m1_export_sha256") == sources["kea_t_report"]["files"]["export.safetensors"]["sha256"]
        == x2facts.EXPORT_SHA["M1"])
    x2facts.check(sources["x2"], checks, info)


def check_yaml(data) -> bool:
    return (data.model_name == REPO_ID.split("/")[1] and data.license == release.CARD_LICENSE
            and data.base_model == paths.MODEL_ID and data.library_name == "nemo")


# ----------------------------------------------------------------------------- D. numbers

def glance_rows(sources: dict) -> list[str] | None:
    test, mb = sources["test"], sources["test"]["export_mb"]
    x2_rows = x2facts.glance_rows(sources["x2"], "MB1", vram_row=False)  # AJ: no VRAM row on Kea-B's glance
    if len(x2_rows) != 2:
        return None
    return [
        row("Weight file", [f"**{mb['MB1']:.1f} MB**", f"{mb['original']:,.0f} MB"]),
        row(f"Test WER, mean of {len(paths.MEAN_SETS)} sets",
            [f"{test['rows']['MB1']['mean']:.2f}%", f"{test['rows']['B0']['mean']:.2f}%"]),
        *x2_rows,
    ]


def tables(docs: dict[str, str], sources: dict) -> list[Table]:
    md = test_md_rows(sources["test_md"])
    x2 = sources["x2"]
    return [
        Table("card at-a-glance table == sources", "README.md", GLANCE_HEADER, glance_rows(sources)),
        Table("card WER table == TEST.md B0, M1 and MB1 rows", "README.md", WER_HEADER,
              [row(label, md[arm]) for arm, label in release.CARD_ROW_LABELS.items()]
              if all(arm in md for arm in release.CARD_ROW_LABELS) else None),
        Table("RUNTIME.md x2 results table == sources", "RUNTIME.md", x2facts.TABLE_HEADER,
              x2facts.runtime_rows(x2, "MB1", X2_LABELS)),
        Table("RUNTIME.md paired table == report-dev", "RUNTIME.md", x2facts.PAIRED_HEADER,
              x2facts.paired_rows(x2, "MB1")),
    ]


def facts(sources: dict) -> dict[str, tuple[tuple[str, ...], str | None]]:
    """Every computed phrase: the release card's numbers (hf/stage.py card_facts), the layer count and the
    shared x2 phrases."""
    card, run_md = DOCS
    released = release.card_facts(sources["test"], sources["summary"], sources["manifest"], sources["stream_sources"])
    layers = len(sources["manifest"]["quantized_layers"])
    return {
        **{f"release: {label}": ((card,), phrase) for label, phrase in released["strings"].items()},
        "quantized matrices": ((card,), f"{layers} large encoder matrices"),
        **x2facts.facts(sources["x2"], "MB1", card, run_md, "Kea-B with x2"),
    }


# ----------------------------------------------------------------------------- configuration and CLI

@functools.lru_cache(maxsize=None)
def config() -> cardkit.Config:
    report = read_json(STAGE_REPORT)["files"]
    return cardkit.Config(
        repo_id=REPO_ID, staging=STAGING, tmp_root=binary_paths.ARTIFACTS / "tmp", pinned_revision=PINNED_REVISION,
        docs=DOCS, staged=STAGED, live_docs=dict(LIVE_DOCS),
        release={rel: v for rel, v in report.items() if rel != "README.md"}, lfs_files=LFS_FILES,
        gitattributes_sha256=GITATTRIBUTES_SHA256, allowed_urls=frozenset({KEA_T_URL}), carried=CARRIED,
        dropped=DROPPED, commit_message=COMMIT_MESSAGE, upload_command=UPLOAD_COMMAND,
        card_yaml_check=CARD_YAML_CHECK, hub_extra_checks=HUB_EXTRA_CHECKS, source_checks=SOURCE_CHECKS,
        table_checks=TABLE_CHECKS, check_hub_extra=hub_extra, check_sources=check_sources, check_yaml=check_yaml,
        tables=tables, facts=facts)


def required_checks() -> set[str]:
    return cardkit.required_checks(config())


def prepare(hub, main_files: set[str], sources: dict, templates: dict[str, str], checks: dict, info: dict) -> dict:
    """Record every check before staging; return the staged documents."""
    return cardkit.prepare(config(), hub, main_files, sources, render(templates, sources), checks, info)


def main(argv: list[str] | None = None) -> None:
    paths.require_mount()

    def prepare_stage(hub, main_files, checks, info):
        return prepare(hub, main_files, load_sources(), load_templates(), checks, info)

    cardkit.main(config(), prepare_stage, paths.REPO, argv, doc=__doc__)


if __name__ == "__main__":
    main()
