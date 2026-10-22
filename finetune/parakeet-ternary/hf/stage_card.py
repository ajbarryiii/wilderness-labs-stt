"""Stage the docs-only update of the Kea-T model card (rajb3/kea-t) and check it. Never uploads.

    ./python hf/stage_card.py stage                                   # from finetune/parakeet-ternary
    ./python hf/stage_card.py upload --approved-readme-sha256 HEX     # AJ, after review: opens the Hub PR
    ./python hf/stage_card.py verify refs/pr/N
    ./python hf/stage_card.py pre-merge refs/pr/N                     # immediately before AJ merges
    ./python hf/stage_card.py verify main                             # after the merge

CPU only: no model load, no GPU, a few small Hub downloads, so it does not need ./heavy. The
contract is plans/card-update.md. The update replaces README.md and adds RUNTIME.md,
EVALUATION.md and TRAINING.md; no other Hub file is staged, so none can change.

`stage` records every check by name; the set of names is fixed in advance (required_checks), and
success is the conjunction of all of them:

A. Hub (read-only): main is still the pinned revision; its file set and every non-document file
   equal the original 2026-10-04 stage report (NOTICE: the pinned post-rename copy, differing
   from the report's only in its first line); README.md is the pinned live card; the new
   document names are not taken.
B. Sources: test.json, rtx5090.json and the run's manifest and reconstruction record are bound
   to the published export; TEST.md and the inference/README.md table equal their JSON.
C. Rendering: no placeholder; the card's YAML parses; its front matter is byte-identical to the
   live card's (model-index unchanged); the detail documents have no front matter.
D. Numbers: the tables equal their sources; every computed phrase is present; every number in
   every document is inside a checked table, a computed phrase or a phrase carried verbatim from
   the live card; every number of the live card still appears in the new documents.
E. Links: relative links resolve to a staged document and heading; the card links every detail
   document and each links back; GitHub links name paths on origin/main; other URLs appear in
   the live card.
F. Staging: exactly the four documents, each equal to its rendering.

Files are written to a temporary sibling folder and renamed to STAGING/rajb3__kea-t.card-update
only if every check passes (report: STAGING/rajb3__kea-t.card-update.stage.json). Stale outputs
are removed first; on any failure no folder is left and the report goes to
...card-update.stage.FAILED.json. The upload, a Hub pull request, is printed, not run.

The card's one sentence on Kea-B is checked against Kea-B's stage report, its test.json and its
public Hub file.

`upload` is the only Hub write: one PR commit of exactly the four staged documents with
parent_commit = the pinned revision, refused unless the stage report passed, the staged bytes
equal it, AJ's approval hash equals the staged README and main is still the pin. `verify
refs/pr/N` (read-only) also requires main to be the pin and the PR to be one commit on it;
`pre-merge` re-checks main and the PR head immediately before the merge; `verify main` checks
the commits since the pin. No other Hub write is allowed between staging and the merge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from collections import namedtuple
from pathlib import Path

import paths

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REPO_ID = "rajb3/kea-t"
NAME = REPO_ID.replace("/", "__") + ".card-update"
STAGING = paths.ARTIFACTS / "hf-staging"
# The live Hub state this update is written against (2026-10-11 "Drop the former-name note").
PINNED_REVISION = "db7d4198687023a8db3b0c9eb4969dc639d684d9"
OLD_CARD_COMMIT = "596d29a6675957b908192fa44051dad4290710c9"  # hf/MODEL_CARD.md that renders to the live README
LIVE_README_SHA256 = "9d52c83748ee101c95d9be9fdb7498f39c7b43f6a4d6ee4e3c2da8685337c893"
NOTICE_SHA256 = "a63eeea2b250401da1b54679221d096f9fd188c354817740639c1da4cf4cddc0"  # hf/NOTICE since the rename
GITATTRIBUTES_SHA256 = "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361"  # the Hub default
STAGE_REPORT = STAGING / "rajb3__parakeet-tdt-0.6b-v2-ternary.stage.json"  # the 2026-10-04 full upload
OLD_STAGED_NOTICE = STAGING / "rajb3__parakeet-tdt-0.6b-v2-ternary" / "NOTICE"
# Kea-B, the 1-bit sibling the card links (its published export and test scoring).
KEA_B_REPO = "rajb3/kea-b"
KEA_B_URL = f"https://huggingface.co/{KEA_B_REPO}"
KEA_B_TEST = PKG.parent / "parakeet-binary" / "results" / "test.json"
KEA_B_STAGE_REPORT = paths.ARTIFACTS.parent / "parakeet-binary" / "hf-staging" / "rajb3__kea-b.stage.json"
RUN = paths.RUNS / "main-M1-P2-lr5e-4"
EXPORT = RUN / "export"

TEMPLATES = {"README.md": "MODEL_CARD.md", "RUNTIME.md": "RUNTIME.md", "EVALUATION.md": "EVALUATION.md",
             "TRAINING.md": "TRAINING.md"}
DOCS = tuple(TEMPLATES)
DETAIL_DOCS = DOCS[1:]
LFS_FILES = ("export.safetensors", "tokenizer/tokenizer.model")
REPORT_UNCHANGED = ("LICENSE", "export.safetensors", "load_ternary.py", "manifest.json", "reconstruction.json",
                    "tokenizer/tokenizer.model", "tokenizer/tokenizer.vocab", "tokenizer/vocab.txt")
UNCHANGED = (*REPORT_UNCHANGED, "NOTICE", ".gitattributes")
GITHUB = "https://github.com/ajbarryiii/wilderness-labs-stt/"
FileMeta = namedtuple("FileMeta", "size lfs_sha256 blob_id")

COLUMNS = [*paths.MEAN_SETS, "mean", "common_voice"]
TEST_MD_HEADER = "| Arm | " + " | ".join(COLUMNS) + " |"
WER_HEADER = ("| Model | LS clean | LS other | AMI | Earnings-22 | GigaSpeech | SPGISpeech | VoxPopuli "
              "| Mean of 7 | Common Voice |")
CARD_WER_LABELS = {"B0": "Original, FP32 (our scoring)", "M1": "**Kea-T (this model)**"}
EVAL_WER_LABELS = {"NVIDIA published": "Original, NVIDIA published", "B0": "Original, FP32, our scoring",
                   "B1": "Same ternary quantizer, no training", "M1": "**This model**",
                   "M1 - B0 (points)": "Difference from our FP32 scoring (points)",
                   "M1 / B0 (relative)": "Ratio to our FP32 scoring"}
RUNTIME_HEADER = ("| Clip | Configuration | Latency (ms) | Process VRAM (GB) | GPU energy (J) "
                  "| Avg GPU power (W) |")
RUNTIME_LABELS = {"packed_expanded": "Ternary expanded — optimized by Wilderness Labs",
                  "packed_optimized": "Ternary compact — optimized by Wilderness Labs",
                  "nemo_bf16_graphs_fused_decoder": "Original BF16 — optimized by Wilderness Labs",
                  "onnx_cuda": "ONNX ASR / ORT CUDA — off the shelf"}
ONNX_JSON_LABEL = "ONNX ASR / ONNX Runtime CUDA — off the shelf"
GLANCE_HEADER = "| | Kea-T | Original |"
GLANCE_SECONDS = 10.0
SET_NAMES = {"librispeech_clean": "LibriSpeech test-clean", "librispeech_other": "LibriSpeech test-other",
             "ami": "AMI", "earnings22": "Earnings-22", "gigaspeech": "GigaSpeech", "spgispeech": "SPGISpeech",
             "voxpopuli": "VoxPopuli", "common_voice": "Common Voice"}
SOURCE_NAMES = {"yodas": "YODAS-Granary English", "peoples_speech": "People's Speech `clean`",
                "librispeech": "LibriSpeech 960 h training sets", "voxpopuli": "VoxPopuli English train",
                "ami": "AMI IHM train"}
WORDS = {4: "four", 5: "five", 8: "eight"}
# Phrases with numbers carried verbatim from the live card; each must appear in it.
CARRIED = (
    "Parakeet-TDT-0.6B-v2", "parakeet-tdt-0.6b-v2", "{-1, 0, +1}", "(-1, 0, +1)",
    "clamp(round(W / s), -1, 1)", "max(mean|W|, 1e-8)", "(E_student - E_teacher)^2",
    "more than 50% normalized WER", "(betas 0.9, 0.98, weight decay 0.01 on matrices)", "gradient clipping 1.0",
    "LibriSpeech 960 h", "SHA-256", "dither 0", "segments of about 30 s or less", "up to about 106 s",
    "can differ by 0.01", "medians of three 5 s windows", "2-bit codes", "kernel-1 convolutions",
    "CC-BY-4.0", "CC-BY-3.0", "CC0-1.0", "(2.0, 2.5, 3.0 or 4.0)", "25 European Languages", "macOS 26",
    "NVIDIA RTX 5090 (CUDA, strict FP32)",
)
# Numbers of the live card deliberately absent from the new documents (none).
DROPPED: tuple[str, ...] = ()

HUB_CHECKS = frozenset({
    "hub main revision == pinned db7d419",
    "hub file set == stage report files + .gitattributes",
    *(f"hub {rel} SHA-256 and size == stage report" for rel in REPORT_UNCHANGED),
    "hub NOTICE SHA-256 and size == hf/NOTICE (pinned, post-rename)",
    "hub NOTICE differs from the stage report's NOTICE only in its first line",
    "hub .gitattributes SHA-256 == pinned",
    "hub README.md SHA-256 == pinned live card",
    "new document names are not on the hub",
    "hub rajb3/kea-b is public; export.safetensors LFS SHA-256 and size == Kea-B stage report",
})
SOURCE_CHECKS = frozenset({
    "original stage report PASSED",
    "results/test.json export SHA-256 == stage report",
    "rtx5090.json export SHA-256 == stage report",
    "run manifest.json and reconstruction.json SHA-256 == stage report",
    "run summary.json reconstruction and export_bytes == export",
    "pilot selection is the run's recipe and learning rate",
    "Kea-B stage report PASSED and its export SHA-256 == Kea-B test.json",
    "Kea-B test.json Kea-T export SHA-256 == stage report",
    "results/TEST.md rows == results/test.json rounded",
    "inference/README.md results table == rtx5090.json rounded",
})
RENDER_CHECKS = frozenset({
    "no unrendered placeholder",
    "card YAML: model_name kea-t, license cc-by-4.0, base_model, library_name nemo",
    "card front matter == live card front matter",
    "detail documents have no front matter",
})
NUMBER_CHECKS = frozenset({
    "card at-a-glance table == sources",
    "card WER table == TEST.md B0 and M1 rows",
    "EVALUATION.md WER table == all 6 TEST.md rows",
    "RUNTIME.md results table == inference/README.md table",
    *(f"{doc} computed numbers == sources" for doc in DOCS),
    "carried phrases appear in the live card",
    *(f"{doc} every number is checked" for doc in DOCS),
    "every number in the live card appears in the new documents",
})
LINK_CHECKS = frozenset({
    "relative links resolve to staged documents and headings",
    "card links every detail document; each links back",
    "GitHub links point at paths on origin/main",
    "other absolute URLs appear in the live card or are Kea-B's page",
})


class StageFailed(Exception):
    pass


def staging_checks() -> frozenset[str]:
    return frozenset({"staged file set == README.md + detail documents",
                      *(f"staged {rel} SHA-256 == rendered" for rel in DOCS)})


def required_checks() -> set[str]:
    return set(HUB_CHECKS | SOURCE_CHECKS | RENDER_CHECKS | NUMBER_CHECKS | LINK_CHECKS | staging_checks())


def verify_checks() -> set[str]:
    return {"card-update stage report PASSED", "revision file set == pinned files + detail documents",
            *(f"revision {rel} unchanged from pinned" for rel in UNCHANGED),
            *(f"revision {rel} == staged" for rel in DOCS)}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text())


# ----------------------------------------------------------------------------- inputs

def load_sources() -> dict:
    """Every source a claim is checked against (plans/card-update.md section 6)."""
    return {
        "test": read_json(PKG / "results" / "test.json"),
        "test_md": (PKG / "results" / "TEST.md").read_text(),
        "rtx": read_json(PKG / "inference" / "results" / "rtx5090.json"),
        "inference_md": (PKG / "inference" / "README.md").read_text(),
        "manifest": read_json(EXPORT / "manifest.json"),
        "manifest_sha256": sha((EXPORT / "manifest.json").read_bytes()),
        "reconstruction": read_json(EXPORT / "reconstruction.json"),
        "reconstruction_sha256": sha((EXPORT / "reconstruction.json").read_bytes()),
        "summary": read_json(RUN / "summary.json"),
        "config": read_json(RUN / "config.json"),
        "pilot": read_json(paths.RUNS / "pilot-selection.json"),
        "b0_dev": read_json(paths.EVAL / "B0-dev" / "summary.json"),
        "stage_report": read_json(STAGE_REPORT),
        "old_notice": OLD_STAGED_NOTICE.read_bytes(),
        "hf_notice": (HERE / "NOTICE").read_bytes(),
        "kea_b_test": read_json(KEA_B_TEST),
        "kea_b_report": read_json(KEA_B_STAGE_REPORT),
    }


def load_templates() -> dict[str, str]:
    return {rel: (HERE / name).read_text() for rel, name in TEMPLATES.items()}


def origin_main_files(repo: Path) -> set[str]:
    """Every file path on the local origin/main ref (not fetched here)."""
    out = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", "origin/main"],
                         check=True, capture_output=True, text=True).stdout
    return set(out.splitlines())


def code_histogram(manifest: dict) -> str:
    hist = manifest["code_histogram"]
    return (f"{100 * hist['minus_one']:.1f}% -1, {100 * hist['zero']:.1f}% 0, "
            f"{100 * hist['plus_one']:.1f}% +1")


def render(templates: dict[str, str], sources: dict) -> dict[str, str]:
    values = {"{repo_id}": REPO_ID, "{repo_name}": REPO_ID.split("/")[1],
              "{code_histogram}": code_histogram(sources["manifest"])}
    out = {}
    for rel, text in templates.items():
        for key, value in values.items():
            text = text.replace(key, value)
        out[rel] = text
    return out


# ----------------------------------------------------------------------------- text helpers

# A number, with its sign when the sign starts a token ("-63.1", "+4.06"; not "TDT-0.6").
NUMBER = re.compile(r"(?:(?<![\w.,])[-+\u2212])?(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?:e-?\d+)?")
LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]+)\)")
URL = re.compile(r"https?://[^\s)>\]]+")
FENCE = re.compile(r"```.*?```", re.S)
PLACEHOLDER = re.compile(r"\{[a-z_]+\}")


def front_matter(text: str) -> str | None:
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    return None if end < 0 else text[:end + 5]


def body(text: str) -> str:
    fm = front_matter(text)
    return text[len(fm):] if fm else text


def norm(text: str) -> str:
    """Prose for matching: no fenced code, links replaced by their text, no bare URLs, single spaces."""
    text = FENCE.sub(" ", text)
    text = LINK.sub(r"\1", text)
    text = URL.sub(" ", text)
    return " ".join(text.split())


def numbers(text: str) -> list[str]:
    return NUMBER.findall(norm(body(text)))


def _bounded(phrase: str) -> re.Pattern:
    """phrase as a whole token run: not preceded by a word character, '.', ',' or a sign, and not
    followed by a word character or by '.' or ',' and a digit (so '-63.1%' or '33,247.5' never
    match '63.1%' or '33,247')."""
    return re.compile(r"(?<![\w.,+\-\u2212])" + re.escape(phrase) + r"(?!\w)(?![.,]\d)")


def contains(text: str, phrase: str) -> bool:
    """norm(phrase) occurs in norm(text) at token boundaries."""
    return _bounded(norm(phrase)).search(norm(text)) is not None


def table(text: str, header: str) -> list[str] | None:
    """The row lines of the one table under header, None if absent, repeated or without a separator."""
    lines = text.splitlines()
    if lines.count(header) != 1:
        return None
    i = lines.index(header)
    if i + 1 >= len(lines) or not lines[i + 1].startswith("| ---"):
        return None
    rows = []
    for line in lines[i + 2:]:
        if not line.startswith("|"):
            break
        rows.append(line)
    return rows


def cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def row(label: str, values: list[str]) -> str:
    return "| " + " | ".join([label, *values]) + " |"


def headings(text: str) -> set[str]:
    return {slug(line.lstrip("#").strip()) for line in FENCE.sub("", text).splitlines() if line.startswith("#")}


def slug(heading: str) -> str:
    return re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")


# ----------------------------------------------------------------------------- A. Hub

class HubClient:
    """Read-only access to REPO_ID; downloads go to a temporary directory under tmp_root."""

    def __init__(self, tmp_root: Path) -> None:
        from huggingface_hub import HfApi

        tmp_root.mkdir(parents=True, exist_ok=True)
        self.api = HfApi()
        self.tmp = tempfile.TemporaryDirectory(dir=tmp_root, prefix="kea-t-card-")

    def head(self) -> str:
        return self.api.model_info(REPO_ID).sha

    def files(self, revision: str) -> dict[str, FileMeta]:
        info = self.api.model_info(REPO_ID, revision=revision, files_metadata=True)
        return {s.rfilename: FileMeta(s.size, s.lfs.sha256 if s.lfs else None, s.blob_id) for s in info.siblings}

    def public_files(self, repo_id: str) -> dict[str, FileMeta] | None:
        """The files at main of another public repository; None if it is private."""
        info = self.api.model_info(repo_id, files_metadata=True)
        if info.private:
            return None
        return {s.rfilename: FileMeta(s.size, s.lfs.sha256 if s.lfs else None, s.blob_id) for s in info.siblings}

    def read(self, rel: str, revision: str) -> bytes:
        from huggingface_hub import hf_hub_download

        local = Path(self.tmp.name) / re.sub(r"[^A-Za-z0-9._-]", "_", revision)
        return Path(hf_hub_download(REPO_ID, rel, revision=revision, local_dir=local)).read_bytes()

    def commits(self, revision: str) -> list[str]:
        return [c.commit_id for c in self.api.list_repo_commits(REPO_ID, revision=revision)]

    def create_pr(self, files: dict[str, bytes], parent: str, message: str) -> dict:
        """The script's only Hub write: one PR commit adding files, based on parent."""
        from huggingface_hub import CommitOperationAdd

        operations = [CommitOperationAdd(path_in_repo=rel, path_or_fileobj=data) for rel, data in sorted(files.items())]
        commit = self.api.create_commit(REPO_ID, operations=operations, commit_message=message, repo_type="model",
                                        create_pr=True, parent_commit=parent)
        return {"pr_num": commit.pr_num, "pr_url": commit.pr_url, "pr_revision": commit.pr_revision,
                "commit": commit.oid}

    def close(self) -> None:
        self.tmp.cleanup()


def hub_file_matches(hub, files: dict, rel: str, revision: str, want_sha: str, want_bytes: int) -> bool:
    meta = files.get(rel)
    if meta is None:
        return False
    if meta.lfs_sha256 is not None:
        return meta.lfs_sha256 == want_sha and meta.size == want_bytes
    data = hub.read(rel, revision)
    return sha(data) == want_sha and len(data) == want_bytes == meta.size


def hub_checks(hub, sources: dict, checks: dict, info: dict) -> str:
    """Record the A checks; return the live card's text."""
    report = sources["stage_report"]["files"]
    head = hub.head()
    files = hub.files(PINNED_REVISION)
    info["hub"] = {"head": head, "pinned_revision": PINNED_REVISION,
                   "files": {rel: meta._asdict() for rel, meta in sorted(files.items())}}
    checks["hub main revision == pinned db7d419"] = head == PINNED_REVISION
    checks["hub file set == stage report files + .gitattributes"] = set(files) == set(report) | {".gitattributes"}
    for rel in REPORT_UNCHANGED:
        checks[f"hub {rel} SHA-256 and size == stage report"] = hub_file_matches(
            hub, files, rel, PINNED_REVISION, report[rel]["sha256"], report[rel]["bytes"])

    def small(rel: str) -> bytes:
        meta = files.get(rel)
        return hub.read(rel, PINNED_REVISION) if meta is not None and meta.lfs_sha256 is None else b""

    notice, hf_notice, old = small("NOTICE"), sources["hf_notice"], sources["old_notice"]
    checks["hub NOTICE SHA-256 and size == hf/NOTICE (pinned, post-rename)"] = (
        sha(notice) == NOTICE_SHA256 == sha(hf_notice) and len(notice) == getattr(files.get("NOTICE"), "size", -1) == len(hf_notice))
    new_lines, old_lines = notice.split(b"\n"), old.split(b"\n")
    checks["hub NOTICE differs from the stage report's NOTICE only in its first line"] = (
        sha(old) == report["NOTICE"]["sha256"] and new_lines[1:] == old_lines[1:] and new_lines[0] != old_lines[0])
    checks["hub .gitattributes SHA-256 == pinned"] = sha(small(".gitattributes")) == GITATTRIBUTES_SHA256
    live = small("README.md")
    checks["hub README.md SHA-256 == pinned live card"] = sha(live) == LIVE_README_SHA256
    checks["new document names are not on the hub"] = not set(files) & set(DETAIL_DOCS)
    kea_b = hub.public_files(KEA_B_REPO)
    want = sources["kea_b_report"]["files"]["export.safetensors"]
    got = (kea_b or {}).get("export.safetensors")
    checks["hub rajb3/kea-b is public; export.safetensors LFS SHA-256 and size == Kea-B stage report"] = (
        got is not None and got.lfs_sha256 == want["sha256"] and got.size == want["bytes"])
    info["hub"]["kea_b_export"] = got._asdict() if got is not None else None
    return live.decode()


# ----------------------------------------------------------------------------- B. sources

def expected_test_md(test: dict) -> dict[str, list[str]]:
    rows, pub = test["rows"], test["published"]
    out = {"NVIDIA published": [f"{pub[c]:.2f}" if c in pub else "n/a" for c in COLUMNS]}
    for arm in ("B0", "B1", "M1"):
        out[arm] = [f"{rows[arm][c]:.2f}" for c in COLUMNS]
    out["M1 - B0 (points)"] = [f"{rows['M1'][c] - rows['B0'][c]:+.2f}" for c in COLUMNS]
    out["M1 / B0 (relative)"] = [f"{rows['M1'][c] / rows['B0'][c]:.2f}x" for c in COLUMNS]
    return out


def test_md_rows(test_md: str) -> dict[str, list[str]] | None:
    rows = table(test_md, TEST_MD_HEADER)
    return None if rows is None else {cells(line)[0]: cells(line)[1:] for line in rows}


def workload(rtx: dict, key: str, seconds: float) -> dict:
    (w,) = [w for w in rtx["variants"][key]["workloads"] if w["audio_seconds"] == seconds]
    return w


def lengths(rtx: dict) -> list[float]:
    return sorted({w["audio_seconds"] for v in rtx["variants"].values() for w in v["workloads"]})


def runtime_rows(rtx: dict) -> list[str]:
    out = []
    for s in lengths(rtx):
        for key, label in RUNTIME_LABELS.items():
            w = workload(rtx, key, s)
            out.append(row(f"{s:g} s", [label, f"{w['median_ms']:.2f}", f"{w['process_vram_gb']:.2f}",
                                        f"{w['joules_per_iteration']:.2f}", f"{w['average_watts']:.0f}"]))
    return out


def source_checks(sources: dict, checks: dict, info: dict) -> None:
    rep, rtx, s, m = sources["stage_report"], sources["rtx"], sources["summary"], sources["manifest"]
    files = rep["files"]
    export_sha = files["export.safetensors"]["sha256"]
    checks["original stage report PASSED"] = (
        rep.get("status") == "PASSED" and bool(rep.get("checks")) and all(v is True for v in rep["checks"].values())
        and rep.get("checks_passed") == f"{len(rep['checks'])}/{len(rep['checks'])}")
    checks["results/test.json export SHA-256 == stage report"] = sources["test"].get("m1_export_sha256") == export_sha
    checks["rtx5090.json export SHA-256 == stage report"] = (
        rtx["model_provenance"].get("ternary_export_sha256") == export_sha)
    checks["run manifest.json and reconstruction.json SHA-256 == stage report"] = (
        sources["manifest_sha256"] == files["manifest.json"]["sha256"] and m["sha256"] == export_sha
        and sources["reconstruction_sha256"] == files["reconstruction.json"]["sha256"])
    checks["run summary.json reconstruction and export_bytes == export"] = (
        s["reconstruction"] == sources["reconstruction"] and s["export_bytes"] == m["bytes"]
        and s["run_name"] == m["extra"]["run_name"] and s["config_sha256"] == m["extra"]["config_sha256"]
        and sources["config"]["args"]["run_name"] == s["run_name"])
    pilot = sources["pilot"]
    checks["pilot selection is the run's recipe and learning rate"] = (
        pilot["recipe"] == s["recipe"] and float(pilot["lr"]) == s["lr"])
    kb_report, kb_test = sources["kea_b_report"], sources["kea_b_test"]
    checks["Kea-B stage report PASSED and its export SHA-256 == Kea-B test.json"] = (
        kb_report.get("status") == "PASSED" and kb_report.get("repo_id") == KEA_B_REPO
        and bool(kb_report.get("checks")) and all(v is True for v in kb_report["checks"].values())
        and kb_report["files"]["export.safetensors"]["sha256"] == kb_test.get("mb1_weights_sha256"))
    checks["Kea-B test.json Kea-T export SHA-256 == stage report"] = kb_test.get("m1_export_sha256") == export_sha
    checks["results/TEST.md rows == results/test.json rounded"] = (
        test_md_rows(sources["test_md"]) == expected_test_md(sources["test"]))
    variants = rtx["variants"]
    checks["inference/README.md results table == rtx5090.json rounded"] = (
        set(variants) == set(RUNTIME_LABELS)
        and all(variants[k]["label"] == RUNTIME_LABELS[k] for k in RUNTIME_LABELS if k != "onnx_cuda")
        and variants["onnx_cuda"]["label"] == ONNX_JSON_LABEL
        and all(v["gpu"]["power_limit_w"] == rtx["power_limit_w"] for v in variants.values())
        and table(sources["inference_md"], RUNTIME_HEADER) == runtime_rows(rtx))


# ----------------------------------------------------------------------------- C. rendering

def render_checks(docs: dict[str, str], live: str, checks: dict, info: dict) -> None:
    from huggingface_hub import ModelCard

    checks["no unrendered placeholder"] = not any(PLACEHOLDER.search(t) for t in docs.values())
    card = docs["README.md"]
    data = ModelCard(card).data  # raises on invalid YAML front matter
    checks["card YAML: model_name kea-t, license cc-by-4.0, base_model, library_name nemo"] = (
        data.model_name == REPO_ID.split("/")[1] and data.license == "cc-by-4.0"
        and data.base_model == paths.MODEL_ID and data.library_name == "nemo")
    checks["card front matter == live card front matter"] = (
        front_matter(card) is not None and front_matter(card) == front_matter(live)
        and data.to_dict() == ModelCard(live).data.to_dict())
    checks["detail documents have no front matter"] = all(
        rel in docs and not docs[rel].lstrip().startswith("---") for rel in DETAIL_DOCS)
    info["card"] = {"model_name": data.model_name, "model_index_results": len(data.eval_results or []),
                    "body_lines": len(body(card).strip().splitlines())}


# ----------------------------------------------------------------------------- D. numbers

def pct_lower(a: float, b: float) -> float:
    return 100 * (1 - a / b)


def e_notation(x: float, digits: int) -> str:
    mantissa, exponent = f"{x:.{digits}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def mm(version: str) -> str:
    return ".".join(version.split("+")[0].split(".")[:2])


def glance_rows(sources: dict) -> list[str]:
    t, rtx = sources["test"], sources["rtx"]
    ex, bf = workload(rtx, "packed_expanded", GLANCE_SECONDS), workload(rtx, "nemo_bf16_graphs_fused_decoder",
                                                                        GLANCE_SECONDS)
    g = f"{GLANCE_SECONDS:g} s clip"
    return [
        row("Weight file", [f"**{t['m1_export_mb']:.1f} MB**", f"{t['original_nemo_mb']:,.0f} MB"]),
        row(f"Test WER, mean of {len(paths.MEAN_SETS)} sets",
            [f"{t['rows']['M1']['mean']:.2f}%", f"{t['rows']['B0']['mean']:.2f}%"]),
        row(f"Latency, {g}", [f"**{ex['median_ms']:.2f} ms** ({pct_lower(ex['median_ms'], bf['median_ms']):.1f}% lower)",
                              f"{bf['median_ms']:.2f} ms"]),
        row(f"GPU energy, {g}", [f"**{ex['joules_per_iteration']:.2f} J** "
                                 f"({pct_lower(ex['joules_per_iteration'], bf['joules_per_iteration']):.1f}% less)",
                                 f"{bf['joules_per_iteration']:.2f} J"]),
        row(f"Process VRAM, {g}", [f"**{ex['process_vram_gb']:.2f} GB** "
                                   f"({pct_lower(ex['process_vram_gb'], bf['process_vram_gb']):.1f}% lower)",
                                   f"{bf['process_vram_gb']:.2f} GB"]),
    ]


def facts(sources: dict) -> dict[str, tuple[tuple[str, ...], str | None]]:
    """{label: (documents that must contain it, phrase computed from its source, None if the source
    contradicts the claim)} (plans/card-update.md section 6)."""
    t, rtx, m, rec = sources["test"], sources["rtx"], sources["manifest"], sources["reconstruction"]
    s, cfg, pilot = sources["summary"], sources["config"], sources["pilot"]
    rows, pub, empty = t["rows"], t["published"], t["empty_hypothesis_fraction"]
    card, run_md, eval_md, train_md = DOCS
    acc, b = m["parameter_accounting"], m["bytes"]
    layers = dict(m["quantized_layers"])
    n_layers = len({k.split(".")[2] for k in layers})
    n_layers = n_layers if n_layers == m["config"]["encoder"]["n_layers"] else None
    gpu = rtx["hardware"].removeprefix("NVIDIA ")
    train_gpu = cfg["versions"]["gpu"].removeprefix("NVIDIA GeForce ")
    pl = rtx["power_limit_w"]
    (default_pl,) = {v["gpu"]["default_power_limit_w"] for v in rtx["variants"].values()}
    short, mid, long_ = lengths(rtx)
    w = {k: {x: workload(rtx, k, x) for x in (short, mid, long_)} for k in RUNTIME_LABELS}
    ex, co, bf, on = (w[k] for k in RUNTIME_LABELS)
    metrics = ("median_ms", "process_vram_gb", "joules_per_iteration")
    lo_s, hi_s = cfg["bucketing"]["prefilter_seconds"]
    steps, ramp, warm = s["steps"], s["ramp_steps"], s["warmup_steps"]
    lr = e_notation(s["lr"], 0)
    hours = s["last_session_train_time_s"] / 3600 if s["resumes"] == 0 else None
    total_h = s["audio_hours_seen"]
    by_source = {k: v / 3600 for k, v in s["seconds_by_source"].items()}
    diff = {c: rows["M1"][c] - rows["B0"][c] for c in COLUMNS if c != "mean"}
    top = sorted(diff, key=diff.get, reverse=True)[:3]
    gap = (f"The gap to the original is largest on {SET_NAMES[top[0]]} ({diff[top[0]]:+.2f} points), which is "
           f"outside the training mixture, followed by {SET_NAMES[top[1]]} ({diff[top[1]]:+.2f}) and "
           f"{SET_NAMES[top[2]]} ({diff[top[2]]:+.2f}).") if top[0] == "common_voice" else None
    p1, p2, p3 = (next(r for r in pilot["recipe_stage"] if r["arm"] == a) for a in ("P1", "P2", "P3"))
    pilot_steps = {r["selected_step"] for r in pilot["recipe_stage"] + pilot["lr_stage"]}
    lr_runs = sorted(pilot["lr_stage"], key=lambda r: r["lr"])
    best = min(lr_runs, key=lambda r: r["dev_mean_wer"])
    validation = rtx["runtime_validation"]
    env = rtx["publication_validation"]["environment"]
    cu = env["torch"].split("+cu")[1]
    nvrtc = next(v for k, v in env.items() if k.startswith("nvidia-cuda-nvrtc"))
    onnx = rtx["model_provenance"]
    repro = max(abs(rows["B0"][c] - pub[c]) for c in paths.MEAN_SETS)
    ex_beats = all(ex[x][k] < bf[x][k] for x in (short, mid, long_) for k in metrics)
    co_least = all(co[x]["process_vram_gb"] == min(w[k][x]["process_vram_gb"] for k in w) for x in (short, mid, long_))
    co_energy = pct_lower(co[mid]["joules_per_iteration"], bf[mid]["joules_per_iteration"])
    co_long_higher = (co[long_]["median_ms"] > bf[long_]["median_ms"]
                      and co[long_]["joules_per_iteration"] > bf[long_]["joules_per_iteration"])
    hist_ok = rec["codes_exact"] is True and rec["scales_exact"] is True and rec["greedy_hyps_equal"] is True
    rows_out = sum(shape["shape"][0] for shape in layers.values())
    kb_mb = sources["kea_b_report"]["files"]["export.safetensors"]["bytes"] / 1e6
    kb_mean = sources["kea_b_test"]["rows"]["MB1"]["mean"]

    def ok(cond: bool, phrase: str) -> str | None:
        return phrase if cond else None

    return {
        # card
        "Kea-B sibling": ((card,), ok(f"{kb_mb:.1f}" == f"{sources['kea_b_test']['export_mb']['MB1']:.1f}",
                                      f"Its 1-bit sibling Kea-B is {kb_mb:.1f} MB with a {kb_mean:.2f}% test mean.")),
        "ternary share of parameters": ((card,), f"{100 * acc['ternary_parameters'] / acc['total_parameters']:.0f}% "
                                                 "of the model's parameters"),
        "encoder layers": ((card,), ok(n_layers is not None, f"{n_layers}-layer FastConformer encoder")),
        "speed setting": ((card, run_md), f"one {gpu} at a {pl} W power limit"),
        "compact VRAM vs BF16 (card)": ((card,), f"mode uses {co[mid]['process_vram_gb']:.2f} GB of process VRAM, "
                                                 f"{pct_lower(co[mid]['process_vram_gb'], bf[mid]['process_vram_gb']):.1f}% "
                                                 "less than that build"),
        "other clip lengths": ((card, run_md), f"the {short:g} s and {long_:g} s clips"),
        "reproduction of NVIDIA's numbers": ((card, eval_md), f"within {repro:.2f} points per set"),
        "ternary matrices": ((card,), f"{len(layers)} ternary matrices"),
        "ramp": ((card, train_md), f"first {100 * ramp / steps:.0f}% of steps ({ramp:,})"),
        "steps and batch audio": ((card, train_md), f"{steps:,} steps of batches of up to {s['batch_seconds']:.0f} s of audio"),
        "peak learning rate (card)": ((card,), f"peak learning rate of {lr}"),
        "audio hours": ((card, train_md), f"{total_h:,.0f} hours of audio"),
        "training time (card)": ((card,), ok(hours is not None, f"{hours:.1f} hours on one {train_gpu}")),
        "trained clip lengths": ((card, eval_md), f"trained on {lo_s:g} to {hi_s:g} s utterances"),
        "export bytes": ((card, train_md), f"({b['file_bytes']:,} bytes)"),
        "largest gaps": ((card, eval_md), gap),
        "runtime setting (limitations)": ((card, run_md), f"one {gpu} at a {pl} W limit"),
        "loader versions": ((card, train_md), f"NeMo {m['versions']['nemo']} and torch {m['versions']['torch']}"),
        # RUNTIME.md
        "expanded INT8 copy": ((run_md,), f"({acc['ternary_parameters']:,} bytes, about "
                                          f"{acc['ternary_parameters'] / 1e6:.0f} MB)"),
        "expanded vs BF16": ((run_md,), f"at {mid:g} s, it has {pct_lower(ex[mid]['median_ms'], bf[mid]['median_ms']):.1f}% "
                                        f"lower latency, {pct_lower(ex[mid]['process_vram_gb'], bf[mid]['process_vram_gb']):.1f}% "
                                        f"lower process VRAM and "
                                        f"{pct_lower(ex[mid]['joules_per_iteration'], bf[mid]['joules_per_iteration']):.1f}% "
                                        "less GPU energy"),
        "expanded lower at every length": ((run_md,), ok(ex_beats, f"It is also lower on all three at the {short:g} s "
                                                                   f"and {long_:g} s clips")),
        "compact least VRAM": ((run_md,), ok(co_least, "It used the least process VRAM of the four configurations "
                                                       "at every length tested")),
        "compact vs BF16": ((run_md,), f"{pct_lower(co[mid]['process_vram_gb'], bf[mid]['process_vram_gb']):.1f}% less "
                                       f"than BF16 at {mid:g} s, with "
                                       f"{pct_lower(co[mid]['median_ms'], bf[mid]['median_ms']):.1f}% lower latency"),
        "compact energy within": ((run_md,), f"Its {mid:g} s energy is within {math.ceil(abs(co_energy))}% of BF16"),
        "compact longest clip": ((run_md,), ok(co_long_higher, f"at {long_:g} s its latency and energy are higher "
                                                               "than BF16's")),
        "ONNX packages": ((run_md,), f"{onnx['onnx_package'].replace('==', ' ')} package with "
                                     f"{onnx['onnx_runtime'].replace('==', ' ')}"),
        "BF16 vs ONNX": ((run_md,), f"At {mid:g} s, our BF16 build also has "
                                    f"{pct_lower(bf[mid]['median_ms'], on[mid]['median_ms']):.1f}% lower latency "
                                    "than this row"),
        "power and energy at 10 s": ((run_md,), ok(
            all(x[mid]["average_watts"] >= bf[mid]["average_watts"]
                and x[mid]["joules_per_iteration"] < bf[mid]["joules_per_iteration"]
                and x[mid]["median_ms"] < bf[mid]["median_ms"] for x in (ex, co)),
            f"At {mid:g} s, compared with our BF16 build, the ternary rows draw as much power or more "
            f"(expanded {ex[mid]['average_watts']:.0f} W, compact {co[mid]['average_watts']:.0f} W, "
            f"BF16 {bf[mid]['average_watts']:.0f} W) and use less energy because they finish sooner")),
        "length order": ((run_md,), "lengths in order (" + ", ".join(f"{x:g}" for x in (short, mid, long_)) + " s)"),
        "power limit": ((run_md,), f"configured {pl} W limit (board default {default_pl:.0f} W)"),
        "mode agreement": ((run_md,), ok(validation["text"] == validation["tokens"] == validation["timestamps"]
                                         == validation["utterances"],
                                         f"{validation['utterances']:,} utterances from "
                                         f"{WORDS[len(validation['per_set']['candidate'])]} test sets")),
        "runtime environment": ((run_md,), f"NeMo {mm(env['nemo_toolkit'])}, PyTorch {mm(env['torch'])} with CUDA "
                                           f"{cu[:-1]}.{cu[-1]}, Triton {mm(env['triton'])}, NVRTC {mm(nvrtc)}"),
        # EVALUATION.md
        "mean of 7": ((eval_md,), f"Mean of {len(paths.MEAN_SETS)}"),
        "TED-LIUM published": ((eval_md,), f"TED-LIUM result, {pub['tedlium']:.2f}, is excluded"),
        "Common Voice gap": ((eval_md,), f"largest gap ({diff['common_voice']:+.2f} points)"),
        "AMI empty outputs": ((eval_md,), ok(empty["M1"]["ami"] < empty["B0"]["ami"],
                                             f"{100 * empty['M1']['ami']:.2f}% of this model's AMI test hypotheses are "
                                             f"empty, fewer than the FP32 original's {100 * empty['B0']['ami']:.2f}%")),
        # TRAINING.md
        "base revision": ((train_md,), f"`{m['base_model']['revision']}`"),
        "encoder matching weight": ((train_md,), f"valid frames, weight {cfg['encoder_matching_weight']:.1f}"),
        "clip filter": ((train_md,), f"utterances outside {lo_s:g} to {hi_s:g} s"),
        "peak learning rate and warmup": ((train_md,), f"peak learning rate {lr} with {100 * warm / steps:.0f}% linear "
                                                       "warmup"),
        "seed": ((train_md,), f"seed {s['seed']}"),
        "training time": ((train_md,), ok(hours is not None, f"One {train_gpu}, {hours:.1f} hours of training "
                                                             f"(about {hours / 24:.1f} days)")),
        **{f"hours of {k}": ((train_md,), f"{SOURCE_NAMES[k]} ({h:,.0f} h, {100 * h / total_h:.1f}%)")
           for k, h in by_source.items()},
        "rounded hours sum": ((train_md,), f"sum to {sum(round(h) for h in by_source.values()):,}"),
        "unrounded hours": ((train_md,), f"unrounded total is {total_h:,.1f} h"),
        "pilot steps": ((train_md,), ok(len(pilot_steps) == 1, f"Pilot ({min(pilot_steps):,} steps per arm")),
        "pilot encoder matching": ((train_md,), ok(all(p2["dev_wer"][k] < p1["dev_wer"][k] for k in p1["dev_wer"]),
                                                   f"{100 * p1['dev_mean_wer']:.2f}% to {100 * p2['dev_mean_wer']:.2f}% "
                                                   f"and helped on all {WORDS[len(p1['dev_wer'])]} sets")),
        "pilot trainable decoder": ((train_md,), f"changed it by {100 * abs(p3['dev_mean_wer'] - p2['dev_mean_wer']):.3f} "
                                                 "points"),
        "pilot learning rates": ((train_md,), ok(e_notation(best["lr"], 0) == lr,
                                                 "Of learning rates " + ", ".join(e_notation(r["lr"], 0) for r in lr_runs[:-1])
                                                 + f" and {e_notation(lr_runs[-1]['lr'], 0)}, {lr} was best")),
        "development WER": ((train_md,), f"development mean WER is {100 * s['dev_mean_wer']:.2f}% (FP32 original "
                                         f"{sources['b0_dev']['dev_mean_wer']:.2f}%)"),
        "ternary layers and parameters": ((train_md,), f"{len(layers)} ternary weight matrices, "
                                                       f"{acc['ternary_parameters']:,} parameters"),
        "layers (files)": ((train_md,), ok(n_layers is not None, f"each of the {n_layers} encoder layers")),
        "packed code bytes": ((train_md,), f"({b['packed_code_bytes']:,} bytes)"),
        "scales": ((train_md,), ok(rows_out * 4 == b["scale_bytes"], f"({rows_out:,} rows, {b['scale_bytes']:,} bytes)")),
        "float parameters": ((train_md,), f"{acc['float_parameters']:,} parameters"),
        "FP16 bytes": ((train_md,), f"({b['fp16_float_bytes']:,} bytes)"),
        "FP32 constants": ((train_md,), f"`.fb`, {b['fp32_feature_constant_bytes']:,} bytes)"),
        "code histogram": ((train_md,), code_histogram(m)),
        "reconstruction layers": ((train_md,), ok(hist_ok and rec["checked_layers"] == len(layers),
                                                  f"exact for all {rec['checked_layers']} layers")),
        "reconstruction clips": ((train_md,), f"on {WORDS[len(rec['utterance_ids'])]} LibriSpeech dev-clean clips"),
        "reconstruction difference": ((train_md,), f"about {e_notation(rec['encoder_max_abs_diff'], 1)}"),
        "tokenizer size": ((train_md,), f"({m['config']['decoder']['vocab_size']:,} tokens)"),
    }


def number_checks(docs: dict[str, str], live: str, sources: dict, checks: dict, info: dict) -> None:
    card, run_md, eval_md = docs["README.md"], docs["RUNTIME.md"], docs["EVALUATION.md"]
    md = test_md_rows(sources["test_md"]) or {}
    glance = table(card, GLANCE_HEADER)
    checks["card at-a-glance table == sources"] = glance == glance_rows(sources)
    card_wer = table(card, WER_HEADER)
    checks["card WER table == TEST.md B0 and M1 rows"] = card_wer == [
        row(label, md.get(arm, [])) for arm, label in CARD_WER_LABELS.items()]
    eval_wer = table(eval_md, WER_HEADER)
    checks["EVALUATION.md WER table == all 6 TEST.md rows"] = list(md) == list(EVAL_WER_LABELS) and eval_wer == [
        row(label, md[arm]) for arm, label in EVAL_WER_LABELS.items()]
    runtime = table(run_md, RUNTIME_HEADER)
    checks["RUNTIME.md results table == inference/README.md table"] = (
        runtime is not None and runtime == table(sources["inference_md"], RUNTIME_HEADER))

    computed = facts(sources)
    missing = {}
    for rel in DOCS:
        text = norm(docs[rel])
        missing[rel] = sorted(label for label, (where, phrase) in computed.items()
                              if rel in where and (phrase is None or not contains(text, phrase)))
        checks[f"{rel} computed numbers == sources"] = not missing[rel]
    live_text = norm(live)
    carried_missing = sorted(p for p in CARRIED if not contains(live_text, p))
    checks["carried phrases appear in the live card"] = not carried_missing

    def table_lines(text: str, header: str, rows: list[str] | None) -> set[str]:
        lines = text.splitlines()
        return set() if rows is None else {header, lines[lines.index(header) + 1], *rows}

    masked = {"README.md": table_lines(card, GLANCE_HEADER, glance) | table_lines(card, WER_HEADER, card_wer),
              "RUNTIME.md": table_lines(run_md, RUNTIME_HEADER, runtime),
              "EVALUATION.md": table_lines(eval_md, WER_HEADER, eval_wer), "TRAINING.md": set()}
    phrases = sorted({norm(p) for _, p in computed.values() if p} | {norm(p) for p in CARRIED}, key=len, reverse=True)
    unchecked = {}
    for rel in DOCS:
        text = norm("\n".join(line for line in body(docs[rel]).splitlines() if line not in masked[rel]))
        for phrase in phrases:
            text = _bounded(phrase).sub(" \x00 ", text)
        unchecked[rel] = NUMBER.findall(text)
        checks[f"{rel} every number is checked"] = not unchecked[rel]
    new_numbers = {n for text in docs.values() for n in numbers(text)}
    lost = sorted(set(numbers(live)) - new_numbers - set(DROPPED))
    checks["every number in the live card appears in the new documents"] = not lost
    info["numbers"] = {"missing_computed": missing, "carried_not_in_live_card": carried_missing,
                       "unchecked": unchecked, "lost_from_live_card": lost, "dropped": list(DROPPED),
                       "computed": {label: phrase for label, (_, phrase) in computed.items()}}


# ----------------------------------------------------------------------------- E. links

def link_checks(docs: dict[str, str], live: str, main_files: set[str], checks: dict, info: dict) -> None:
    targets = {rel: [t for _, t in LINK.findall(FENCE.sub("", body(text)))] for rel, text in docs.items()}
    anchors = {rel: headings(text) for rel, text in docs.items()}
    bad_relative, github_missing, unknown = [], [], []
    live_urls = set(URL.findall(live))
    dirs = {"/".join(f.split("/")[:i]) for f in main_files for i in range(1, f.count("/") + 1)}
    for rel, links in targets.items():
        for target in links:
            if not re.match(r"[a-z]+:", target):
                path, _, anchor = target.partition("#")
                if path not in docs or (anchor and anchor not in anchors[path]):
                    bad_relative.append(f"{rel}: {target}")
        for url in URL.findall(FENCE.sub("", docs[rel])):
            if url.startswith(GITHUB):
                match = re.fullmatch(re.escape(GITHUB) + r"(tree|blob)/main/(.+?)/?", url)
                if not match or match[2] not in (dirs if match[1] == "tree" else main_files):
                    github_missing.append(f"{rel}: {url}")
            elif url not in live_urls and url != KEA_B_URL:
                unknown.append(f"{rel}: {url}")
    linked = {rel: {t.partition("#")[0] for t in links} for rel, links in targets.items()}
    checks["relative links resolve to staged documents and headings"] = not bad_relative
    checks["card links every detail document; each links back"] = (
        set(DETAIL_DOCS) <= linked["README.md"] and all("README.md" in linked[rel] for rel in DETAIL_DOCS))
    checks["GitHub links point at paths on origin/main"] = not github_missing
    checks["other absolute URLs appear in the live card or are Kea-B's page"] = not unknown
    info["links"] = {"bad_relative": bad_relative, "github_missing": github_missing, "unknown_urls": unknown,
                     "linked": {rel: sorted(v) for rel, v in linked.items()}}


# ----------------------------------------------------------------------------- A-E together

def prepare(hub, main_files: set[str], sources: dict, templates: dict[str, str], checks: dict, info: dict) -> dict:
    """Record every check before staging; return the rendered documents."""
    live = hub_checks(hub, sources, checks, info)
    source_checks(sources, checks, info)
    docs = render(templates, sources)
    render_checks(docs, live, checks, info)
    number_checks(docs, live, sources, checks, info)
    link_checks(docs, live, main_files, checks, info)
    return docs


# ----------------------------------------------------------------------------- F. staging

def staged_listing(folder: Path) -> dict:
    return {p.relative_to(folder).as_posix(): {"sha256": sha(p.read_bytes()), "bytes": p.stat().st_size}
            for p in sorted(folder.rglob("*")) if p.is_file()}


def write_json(path: Path, data) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=1, default=str) + "\n")
    os.replace(tmp, path)


COMMIT_MESSAGE = ("Model card: lead with size, accuracy and speed; move details to RUNTIME.md, EVALUATION.md "
                  "and TRAINING.md")


def upload_command() -> str:
    return ("./python hf/stage_card.py upload --approved-readme-sha256 <SHA-256 of the README.md AJ approved>"
            "   # from finetune/parakeet-ternary")


def stage(staging: Path, prepare_fn) -> tuple[bool, dict, Path]:
    """Run every check and stage the documents under staging; (passed, report, report path).

    prepare_fn(checks, info) records the A-E checks and returns {staged name: text}. Any failure,
    including an exception or SystemExit, leaves no final and no temporary folder, and only the
    FAILED report.
    """
    folder, tmp = staging / NAME, staging / f".{NAME}.tmp-{os.getpid()}"
    report_ok, report_failed = staging / f"{NAME}.stage.json", staging / f"{NAME}.stage.FAILED.json"
    staging.mkdir(parents=True, exist_ok=True)
    for stale in (report_ok, report_failed):
        stale.unlink(missing_ok=True)
    for stale in (folder, *staging.glob(f".{NAME}.tmp-*")):
        if stale.exists():
            shutil.rmtree(stale)
    checks: dict = {}
    info: dict = {"repo_id": REPO_ID, "pinned_revision": PINNED_REVISION}
    try:
        docs = prepare_fn(checks, info)
        failed = sorted(k for k, ok in checks.items() if ok is not True)
        if failed:
            raise StageFailed(f"{len(failed)} check(s) failed before staging: {failed}")
        tmp.mkdir()
        for rel, text in docs.items():
            (tmp / rel).write_bytes(text.encode())
        listing = staged_listing(tmp)
        checks["staged file set == README.md + detail documents"] = sorted(listing) == sorted(DOCS)
        for rel in DOCS:
            checks[f"staged {rel} SHA-256 == rendered"] = (
                rel in docs and listing.get(rel, {}).get("sha256") == sha(docs[rel].encode()))
        required = required_checks()
        missing, unexpected = sorted(required - set(checks)), sorted(set(checks) - required)
        if missing or unexpected:
            raise StageFailed(f"check set mismatch: missing {missing}, unexpected {unexpected}")
        failed = sorted(k for k, ok in checks.items() if ok is not True)
        if failed:
            raise StageFailed(f"{len(failed)} check(s) failed: {failed}")
        validated = listing
        os.rename(tmp, folder)  # atomic promotion within staging
        listing = staged_listing(folder)
        if listing != validated:
            raise StageFailed("staged files changed between validation and promotion")
        report = {"status": "PASSED", "folder": str(folder), **info, "files": listing,
                  "upload_command": upload_command(),
                  "checks_passed": f"{len(checks)}/{len(checks)}", "checks": checks}
        write_json(report_ok, report)
        return True, report, report_ok
    except BaseException as error:  # noqa: BLE001 -- any failure, including SystemExit, must leave no folder
        for leftover in (tmp, folder):
            if leftover.exists():
                shutil.rmtree(leftover)
        report_ok.unlink(missing_ok=True)
        report = {"status": "FAILED", "error": "".join(traceback.format_exception_only(error)).strip(), **info,
                  "checks": checks, "failed_checks": sorted(k for k, ok in checks.items() if ok is not True)}
        write_json(report_failed, report)
        return False, report, report_failed


# ----------------------------------------------------------------------------- publish

def upload_checks() -> set[str]:
    return {"card-update stage report PASSED", "staged files == stage report (exact set, SHA-256 and size)",
            "approved README SHA-256 == staged README", "main is still the pinned revision"}


def upload(hub, folder: Path, report: dict, approved_sha256: str, checks: dict, info: dict) -> dict | None:
    """Open the Hub PR, the script's only Hub write, if every upload check passes; else None.

    The bytes that are hashed are the bytes uploaded, as one commit of exactly the four documents
    on parent_commit = the pinned revision (never on a newer main)."""
    staged = report.get("files", {})
    checks["card-update stage report PASSED"] = (
        report.get("status") == "PASSED" and sorted(staged) == sorted(DOCS)
        and Path(report.get("folder", "")) == Path(folder))
    present = sorted(p.relative_to(folder).as_posix() for p in Path(folder).rglob("*") if p.is_file()) \
        if Path(folder).is_dir() else []
    data = {rel: (Path(folder) / rel).read_bytes() for rel in DOCS if rel in present}
    listing = {rel: {"sha256": sha(b), "bytes": len(b)} for rel, b in data.items()}
    checks["staged files == stage report (exact set, SHA-256 and size)"] = (
        present == sorted(DOCS) and listing == staged)
    checks["approved README SHA-256 == staged README"] = (
        bool(re.fullmatch(r"[0-9a-f]{64}", approved_sha256 or ""))
        and approved_sha256 == staged.get("README.md", {}).get("sha256") == listing.get("README.md", {}).get("sha256"))
    head = hub.head()
    checks["main is still the pinned revision"] = head == PINNED_REVISION
    info["upload"] = {"folder": str(folder), "main": head, "parent_commit": PINNED_REVISION, "files": listing}
    if set(checks) != upload_checks() or any(ok is not True for ok in checks.values()):
        return None
    pr = hub.create_pr(data, PINNED_REVISION, COMMIT_MESSAGE)
    info["upload"]["pr"] = pr
    return pr


def verify_checks(revision: str) -> set[str]:
    names = {"card-update stage report PASSED", "revision file set == pinned files + detail documents",
             *(f"revision {rel} unchanged from pinned" for rel in UNCHANGED),
             *(f"revision {rel} == staged" for rel in DOCS)}
    if revision.startswith("refs/pr/"):
        return names | {"main is still the pinned revision", "revision is one commit on the pinned revision"}
    return names | {"commits since the pinned revision are only the card update"}


def verify(hub, revision: str, report: dict, checks: dict, info: dict, pr_head: str | None = None) -> None:
    """Record verify_checks(revision) for refs/pr/N (before the merge) or main (after it; pr_head is
    the verified PR head commit, from the pre-merge report)."""
    staged = report.get("files", {})
    checks["card-update stage report PASSED"] = report.get("status") == "PASSED" and sorted(staged) == sorted(DOCS)
    commits = hub.commits(revision)
    since = commits[:commits.index(PINNED_REVISION)] if PINNED_REVISION in commits else None
    if revision.startswith("refs/pr/"):
        checks["main is still the pinned revision"] = hub.head() == PINNED_REVISION
        checks["revision is one commit on the pinned revision"] = since is not None and len(since) == 1
    else:
        # A fast-forward or squash adds one commit; a merge commit adds itself on top of the PR head.
        checks["commits since the pinned revision are only the card update"] = since is not None and (
            len(since) == 1 or (len(since) == 2 and pr_head is not None and since[1] == pr_head))
    pinned, now = hub.files(PINNED_REVISION), hub.files(revision)
    checks["revision file set == pinned files + detail documents"] = set(now) == set(pinned) | set(DOCS)
    for rel in UNCHANGED:
        checks[f"revision {rel} unchanged from pinned"] = rel in now and rel in pinned and now[rel] == pinned[rel]
    for rel in DOCS:
        want = staged.get(rel)
        data = hub.read(rel, revision) if rel in now else None
        checks[f"revision {rel} == staged"] = (want is not None and data is not None
                                               and sha(data) == want["sha256"] and len(data) == want["bytes"])
    info["verify"] = {"revision": revision, "revision_sha": commits[0] if commits else None,
                      "commits_since_pin": since, "pr_head": pr_head,
                      "files": {rel: meta._asdict() for rel, meta in sorted(now.items())}}


def pre_merge_checks() -> set[str]:
    return {"verify report for the PR PASSED", "main is still the pinned revision", "PR head == the verified PR head"}


def pre_merge(hub, revision: str, verify_report: dict, checks: dict, info: dict) -> None:
    """Run immediately before AJ merges: nothing moved since `verify revision` passed."""
    verified = verify_report.get("verify", {})
    checks["verify report for the PR PASSED"] = (
        verify_report.get("status") == "PASSED" and verified.get("revision") == revision
        and bool(verified.get("revision_sha")))
    head = hub.head()
    checks["main is still the pinned revision"] = head == PINNED_REVISION
    pr_head = hub.commits(revision)[0]
    checks["PR head == the verified PR head"] = pr_head == verified.get("revision_sha")
    info["pre_merge"] = {"revision": revision, "main": head, "pr_head": pr_head}


# ----------------------------------------------------------------------------- CLI

def _safe(revision: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", revision)


def _finish(path: Path, title: str, checks: dict, expected: set[str], info: dict) -> dict:
    failed = sorted(k for k, ok in checks.items() if ok is not True)
    out = {"status": "FAILED" if failed or set(checks) != expected else "PASSED", **info,
           "checks": checks, "failed_checks": failed}
    write_json(path, out)
    print(json.dumps(out, indent=1, default=str))
    print(f"\n{title}: {out['status']} (report: {path})")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("stage", help="check and stage the card update (no upload)")
    u = sub.add_parser("upload", help="open the Hub PR (the only Hub write), on the pinned parent")
    u.add_argument("--approved-readme-sha256", required=True,
                   help="SHA-256 of the staged README.md that AJ read and approved")
    v = sub.add_parser("verify", help="check refs/pr/N (before the merge) or main (after it)")
    v.add_argument("revision")
    m = sub.add_parser("pre-merge", help="run immediately before merging: nothing moved since verify")
    m.add_argument("revision")
    args = parser.parse_args(argv)
    paths.require_mount()
    hub = HubClient(paths.ARTIFACTS / "tmp")
    try:
        if args.command == "stage":
            def run(checks: dict, info: dict) -> dict:
                main_files = origin_main_files(paths.REPO)
                info["origin_main"] = subprocess.run(["git", "-C", str(paths.REPO), "rev-parse", "origin/main"],
                                                     check=True, capture_output=True, text=True).stdout.strip()
                return prepare(hub, main_files, load_sources(), load_templates(), checks, info)

            passed, report, path = stage(STAGING, run)
            print(json.dumps(report, indent=1, default=str))
            if not passed:
                print(f"\nSTAGING FAILED; no upload folder was left. Report: {path}", file=sys.stderr)
                sys.exit(1)
            print(f"\nAll {len(report['checks'])} checks passed (report: {path}). Nothing was uploaded.\n"
                  f"Staged README.md SHA-256: {report['files']['README.md']['sha256']}\n"
                  "After review and AJ's approval (plans/card-update.md section 5):\n"
                  f"  {report['upload_command']}")
            return
        report = read_json(STAGING / f"{NAME}.stage.json")
        checks: dict = {}
        info: dict = {"repo_id": REPO_ID, "pinned_revision": PINNED_REVISION}
        if args.command == "upload":
            pr = upload(hub, STAGING / NAME, report, args.approved_readme_sha256, checks, info)
            out = _finish(STAGING / f"{NAME}.upload.json", "upload", checks, upload_checks(), info)
            if pr is None or out["status"] != "PASSED":
                print("Refused; nothing was written to the Hub.", file=sys.stderr)
                sys.exit(1)
            print(f"Opened {pr['pr_url']}. Next: ./python hf/stage_card.py verify {pr['pr_revision']}")
            return
        if args.command == "verify":
            pr_head = None
            if not args.revision.startswith("refs/pr/"):
                records = sorted(STAGING.glob(f"{NAME}.pre-merge-*.json"), key=lambda p: p.stat().st_mtime)
                passed_records = [read_json(p) for p in records if read_json(p).get("status") == "PASSED"]
                pr_head = passed_records[-1]["pre_merge"]["pr_head"] if passed_records else None
            verify(hub, args.revision, report, checks, info, pr_head=pr_head)
            out = _finish(STAGING / f"{NAME}.verify-{_safe(args.revision)}.json", f"verify {args.revision}",
                          checks, verify_checks(args.revision), info)
        else:
            if not args.revision.startswith("refs/pr/"):
                parser.error("pre-merge takes refs/pr/N")
            verified = read_json(STAGING / f"{NAME}.verify-{_safe(args.revision)}.json")
            pre_merge(hub, args.revision, verified, checks, info)
            out = _finish(STAGING / f"{NAME}.pre-merge-{_safe(args.revision)}.json", "pre-merge", checks,
                          pre_merge_checks(), info)
            if out["status"] == "PASSED":
                print("Merge the PR on the Hub now; no other Hub writes until then. Then: "
                      "./python hf/stage_card.py verify main")
        if out["status"] != "PASSED":
            sys.exit(1)
    finally:
        hub.close()


if __name__ == "__main__":
    main()
