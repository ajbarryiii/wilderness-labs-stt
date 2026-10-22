"""Shared engine for docs-only updates of a published model card (Kea-T, Kea-B). Never uploads by itself.

Contract: finetune/parakeet-binary/plans/card-update-x2.md (generalizing
finetune/parakeet-ternary/plans/card-update.md). Each model's hf/stage_card.py builds a Config (pins,
documents, release record, per-model checks, tables and computed phrases) and calls main(). CPU only:
no model load, no GPU, a few read-only Hub calls.

`stage` records every check by name; the set of names is fixed per model (required_checks) and success
is the conjunction of all of them:

A. Hub (read-only): main is still the pin; its file set is the release record, .gitattributes and the
   live documents; every unchanged file equals its release record; each live document is its pinned
   SHA-256; new document names are not taken; plus the model's own Hub checks.
B. Sources: the model's own source checks (Config.check_sources).
C. Rendering: no placeholder or TODO; the card's YAML (Config.check_yaml); front matter byte-identical to
   the live card's; detail documents without front matter; unstaged documents equal the live ones.
D. Numbers: every checked table equals its rows computed from the sources; every computed phrase is
   present at token boundaries; every carried phrase is in the live documents; no number in any
   document is outside a checked table, a computed phrase or a carried phrase; every number of each
   live document being replaced still appears in the new documents.
E. Links: relative links resolve to documents and headings; the card links every detail document and
   each links back; GitHub links name paths on origin/main; other URLs are in the live documents or
   allowed.
F. Staging: exactly the changed documents, each equal to its rendering.

`upload` is the only Hub write: one PR commit of exactly the staged documents on parent_commit = the
pin. `verify refs/pr/N`, `pre-merge refs/pr/N` and `verify main` are read-only.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

FileMeta = namedtuple("FileMeta", "size lfs_sha256 blob_id")
GITHUB = "https://github.com/ajbarryiii/wilderness-labs-stt/"


class StageFailed(Exception):
    pass


# ----------------------------------------------------------------------------- text helpers

# A number, with its sign when the sign starts a token ("-63.1", "+4.06"; not "TDT-0.6").
NUMBER = re.compile(r"(?:(?<![\w.,])[-+−])?(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?:e-?\d+)?")
LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]+)\)")
URL = re.compile(r"https?://[^\s)>\]]+")
FENCE = re.compile(r"```.*?```", re.S)
PLACEHOLDER = re.compile(r"\{[a-z_]+\}")
TODO = re.compile(r"\bTODO\b|\bTBD\b|\bFIXME\b|XXX")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text())


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


def bounded(phrase: str) -> re.Pattern:
    """phrase as a whole token run: not preceded by a word character, '.', ',' or a sign, and not
    followed by a word character or by '.' or ',' and a digit (so '-63.1%' or '33,247.5' never
    match '63.1%' or '33,247')."""
    return re.compile(r"(?<![\w.,+\-−])" + re.escape(phrase) + r"(?!\w)(?![.,]\d)")


def contains(text: str, phrase: str) -> bool:
    """norm(phrase) occurs in norm(text) at token boundaries."""
    return bounded(norm(phrase)).search(norm(text)) is not None


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


def pct_lower(a: float, b: float) -> float:
    return 100 * (1 - a / b)


def e_notation(x: float, digits: int) -> str:
    mantissa, exponent = f"{x:.{digits}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def mm(version: str) -> str:
    return ".".join(version.split("+")[0].split(".")[:2])


def ceil_to(x: float, step: float) -> float:
    """The smallest multiple of step that is >= x (with a guard against float noise)."""
    return math.ceil(round(x / step, 9)) * step


# ----------------------------------------------------------------------------- configuration

@dataclass(frozen=True)
class Table:
    """A checked table: the rows under header in doc must equal expected (None: the source is missing).
    With section, the header is looked up only under that heading (up to the next heading of the same or a
    higher level), so two tables with one header can be checked separately."""
    check: str
    doc: str
    header: str
    expected: list[str] | None
    section: str | None = None


def section_text(text: str, heading: str) -> str:
    """The lines from heading (a whole line such as '## Results') to the next heading of the same or a higher
    level; '' if the heading is absent or repeated."""
    lines = text.splitlines()
    if lines.count(heading) != 1:
        return ""
    level = len(heading) - len(heading.lstrip("#"))
    start = lines.index(heading)
    out = [lines[start]]
    for line in lines[start + 1:]:
        if line.startswith("#") and len(line) - len(line.lstrip("#")) <= level:
            break
        out.append(line)
    return "\n".join(out)


@dataclass(frozen=True)
class Config:
    repo_id: str
    staging: Path
    tmp_root: Path
    pinned_revision: str
    docs: tuple[str, ...]                    # the full document set after the update; README.md first
    staged: tuple[str, ...]                  # the documents uploaded (new or changed)
    live_docs: dict                          # document on the Hub at the pin -> SHA-256
    release: dict                            # unchanged non-document file -> {"sha256", "bytes"}
    lfs_files: tuple[str, ...]
    gitattributes_sha256: str
    allowed_urls: frozenset
    carried: tuple[str, ...]
    dropped: tuple[str, ...]
    commit_message: str
    upload_command: str
    card_yaml_check: str
    hub_extra_checks: frozenset
    source_checks: frozenset
    table_checks: frozenset
    check_hub_extra: Callable                # (hub, sources, files, checks, info) -> None
    check_sources: Callable                  # (sources, checks, info) -> None
    check_yaml: Callable                     # (ModelCard data, checks) -> bool for card_yaml_check
    tables: Callable                         # (docs, sources) -> list[Table]
    facts: Callable                          # (sources) -> {label: (documents, phrase or None)}
    name_suffix: str = "card-update-x2"
    extra: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.repo_id.replace("/", "__") + "." + self.name_suffix

    @property
    def short_pin(self) -> str:
        return self.pinned_revision[:7]

    @property
    def detail_docs(self) -> tuple[str, ...]:
        return self.docs[1:]

    @property
    def new_docs(self) -> tuple[str, ...]:
        return tuple(d for d in self.docs if d not in self.live_docs)

    @property
    def unstaged(self) -> tuple[str, ...]:
        return tuple(d for d in self.docs if d not in self.staged)

    @property
    def unchanged(self) -> tuple[str, ...]:
        """Every Hub file a PR must leave as at the pin."""
        return (*self.release, ".gitattributes", *self.unstaged)


# ----------------------------------------------------------------------------- check-name sets

def hub_check_names(cfg: Config) -> set[str]:
    return {f"hub main revision == pinned {cfg.short_pin}", "hub file set == pinned file set",
            *(f"hub {rel} SHA-256 and size == release record" for rel in cfg.release),
            "hub .gitattributes SHA-256 == pinned",
            *(f"hub {doc} SHA-256 == pinned live document" for doc in cfg.live_docs),
            "new document names are not on the hub", *cfg.hub_extra_checks}


RENDER_CHECKS = ("no unrendered placeholder or TODO marker", "card front matter == live card front matter",
                 "detail documents have no front matter", "unstaged documents == live hub documents")


def number_check_names(cfg: Config) -> set[str]:
    return {*cfg.table_checks, *(f"{doc} computed numbers == sources" for doc in cfg.docs),
            "carried phrases appear in the live documents", *(f"{doc} every number is checked" for doc in cfg.docs),
            "every number in the live documents appears in the new documents"}


LINK_CHECKS = ("relative links resolve to documents and headings", "card links every detail document; each links back",
               "GitHub links point at paths on origin/main",
               "other absolute URLs appear in the live documents or are allowed")


def staging_checks(cfg: Config) -> set[str]:
    return {"staged file set == the changed documents", *(f"staged {rel} SHA-256 == rendered" for rel in cfg.staged)}


def required_checks(cfg: Config) -> set[str]:
    return (hub_check_names(cfg) | set(cfg.source_checks) | {*RENDER_CHECKS, cfg.card_yaml_check}
            | number_check_names(cfg) | set(LINK_CHECKS) | staging_checks(cfg))


def upload_checks() -> set[str]:
    return {"card-update stage report PASSED", "staged files == stage report (exact set, SHA-256 and size)",
            "approved bundle SHA-256 == bundle of the bytes to upload", "main is still the pinned revision"}


def verify_checks(cfg: Config, revision: str) -> set[str]:
    names = {"card-update stage report PASSED", "revision file set == pinned files + new documents",
             *(f"revision {rel} unchanged from pinned" for rel in cfg.unchanged),
             *(f"revision {rel} == staged" for rel in cfg.staged)}
    if revision.startswith("refs/pr/"):
        return names | {"main is still the pinned revision", "revision is one commit on the pinned revision"}
    return names | {"commits since the pinned revision are only the card update"}


def pre_merge_checks() -> set[str]:
    return {"verify report for the PR PASSED", "main is still the pinned revision", "PR head == the verified PR head"}


# ----------------------------------------------------------------------------- Hub client

class HubClient:
    """Read-only access to repo_id (plus create_pr, used only by upload); downloads go to a temporary
    directory under tmp_root."""

    def __init__(self, repo_id: str, tmp_root: Path) -> None:
        from huggingface_hub import HfApi

        tmp_root.mkdir(parents=True, exist_ok=True)
        self.repo_id = repo_id
        self.api = HfApi()
        self.tmp = tempfile.TemporaryDirectory(dir=tmp_root, prefix=repo_id.split("/")[1] + "-card-")

    def head(self) -> str:
        return self.api.model_info(self.repo_id).sha

    def files(self, revision: str) -> dict[str, FileMeta]:
        info = self.api.model_info(self.repo_id, revision=revision, files_metadata=True)
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
        return Path(hf_hub_download(self.repo_id, rel, revision=revision, local_dir=local)).read_bytes()

    def commits(self, revision: str) -> list[str]:
        return [c.commit_id for c in self.api.list_repo_commits(self.repo_id, revision=revision)]

    def create_pr(self, files: dict[str, bytes], parent: str, message: str) -> dict:
        """The engine's only Hub write: one PR commit adding files, based on parent."""
        from huggingface_hub import CommitOperationAdd

        operations = [CommitOperationAdd(path_in_repo=rel, path_or_fileobj=data) for rel, data in sorted(files.items())]
        commit = self.api.create_commit(self.repo_id, operations=operations, commit_message=message,
                                        repo_type="model", create_pr=True, parent_commit=parent)
        return {"pr_num": commit.pr_num, "pr_url": commit.pr_url, "pr_revision": commit.pr_revision,
                "commit": commit.oid}

    def close(self) -> None:
        self.tmp.cleanup()


def origin_main_files(repo: Path) -> set[str]:
    """Every file path on the local origin/main ref (refresh it with git fetch before staging)."""
    out = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", "origin/main"],
                         check=True, capture_output=True, text=True).stdout
    return set(out.splitlines())


def hub_file_matches(hub, files: dict, rel: str, revision: str, want_sha: str, want_bytes: int) -> bool:
    meta = files.get(rel)
    if meta is None:
        return False
    if meta.lfs_sha256 is not None:
        return meta.lfs_sha256 == want_sha and meta.size == want_bytes
    data = hub.read(rel, revision)
    return sha(data) == want_sha and len(data) == want_bytes == meta.size


def small(hub, files: dict, rel: str, revision: str) -> bytes:
    meta = files.get(rel)
    return hub.read(rel, revision) if meta is not None and meta.lfs_sha256 is None else b""


# ----------------------------------------------------------------------------- A. Hub

def hub_checks(cfg: Config, hub, sources: dict, checks: dict, info: dict) -> dict[str, str]:
    """Record the A checks; return the live documents' text."""
    head = hub.head()
    files = hub.files(cfg.pinned_revision)
    info["hub"] = {"head": head, "pinned_revision": cfg.pinned_revision,
                   "files": {rel: meta._asdict() for rel, meta in sorted(files.items())}}
    checks[f"hub main revision == pinned {cfg.short_pin}"] = head == cfg.pinned_revision
    checks["hub file set == pinned file set"] = set(files) == set(cfg.release) | {".gitattributes"} | set(cfg.live_docs)
    for rel, want in cfg.release.items():
        checks[f"hub {rel} SHA-256 and size == release record"] = hub_file_matches(
            hub, files, rel, cfg.pinned_revision, want["sha256"], want["bytes"])
    checks["hub .gitattributes SHA-256 == pinned"] = (
        sha(small(hub, files, ".gitattributes", cfg.pinned_revision)) == cfg.gitattributes_sha256)
    live = {}
    for doc, want in cfg.live_docs.items():
        data = small(hub, files, doc, cfg.pinned_revision)
        checks[f"hub {doc} SHA-256 == pinned live document"] = sha(data) == want
        live[doc] = data.decode()
    checks["new document names are not on the hub"] = not set(files) & set(cfg.new_docs)
    cfg.check_hub_extra(hub, sources, files, checks, info)
    return live


# ----------------------------------------------------------------------------- C. rendering

def render_checks(cfg: Config, docs: dict[str, str], live: dict[str, str], checks: dict, info: dict) -> None:
    from huggingface_hub import ModelCard

    checks["no unrendered placeholder or TODO marker"] = not any(
        PLACEHOLDER.search(t) or TODO.search(t) for t in docs.values())
    card = docs["README.md"]
    data = ModelCard(card).data  # raises on invalid YAML front matter
    checks[cfg.card_yaml_check] = cfg.check_yaml(data) is True
    live_card = live.get("README.md", "")
    checks["card front matter == live card front matter"] = (
        front_matter(card) is not None and front_matter(card) == front_matter(live_card)
        and data.to_dict() == ModelCard(live_card).data.to_dict())
    checks["detail documents have no front matter"] = all(
        rel in docs and not docs[rel].lstrip().startswith("---") for rel in cfg.detail_docs)
    checks["unstaged documents == live hub documents"] = all(
        rel in docs and rel in live and docs[rel] == live[rel] for rel in cfg.unstaged)
    info["card"] = {"model_name": data.model_name, "model_index_results": len(data.eval_results or []),
                    "body_lines": {rel: len(body(text).strip().splitlines()) for rel, text in docs.items()}}


# ----------------------------------------------------------------------------- D. numbers

def number_checks(cfg: Config, docs: dict[str, str], live: dict[str, str], sources: dict, checks: dict,
                  info: dict) -> None:
    masked: dict[str, set[str]] = {rel: set() for rel in cfg.docs}
    tables_info = {}
    for t in cfg.tables(docs, sources):
        text = docs.get(t.doc, "")
        scope = section_text(text, t.section) if t.section else text
        got = table(scope, t.header)
        checks[t.check] = t.expected is not None and got is not None and got == t.expected
        if got is not None:
            lines = scope.splitlines()
            masked[t.doc] |= {t.header, lines[lines.index(t.header) + 1], *got}
        tables_info[t.check] = {"doc": t.doc, "expected": t.expected, "got": got}

    computed = cfg.facts(sources)
    missing = {}
    for rel in cfg.docs:
        text = norm(docs.get(rel, ""))
        missing[rel] = sorted(label for label, (where, phrase) in computed.items()
                              if rel in where and (phrase is None or not contains(text, phrase)))
        checks[f"{rel} computed numbers == sources"] = not missing[rel]
    live_text = norm("\n".join(live.values()))
    carried_missing = sorted(p for p in cfg.carried if not contains(live_text, p))
    checks["carried phrases appear in the live documents"] = not carried_missing

    phrases = sorted({norm(p) for _, p in computed.values() if p} | {norm(p) for p in cfg.carried}, key=len,
                     reverse=True)
    unchecked = {}
    for rel in cfg.docs:
        text = norm("\n".join(line for line in body(docs.get(rel, "")).splitlines() if line not in masked[rel]))
        for phrase in phrases:
            text = bounded(phrase).sub(" \x00 ", text)
        unchecked[rel] = NUMBER.findall(text)
        checks[f"{rel} every number is checked"] = not unchecked[rel]
    new_numbers = {n for text in docs.values() for n in numbers(text)}
    replaced = [rel for rel in cfg.staged if rel in live]
    lost = sorted({n for rel in replaced for n in numbers(live[rel])} - new_numbers - set(cfg.dropped))
    checks["every number in the live documents appears in the new documents"] = not lost
    info["numbers"] = {"tables": tables_info, "missing_computed": missing, "carried_not_in_live": carried_missing,
                       "unchecked": unchecked, "lost_from_live": lost, "dropped": list(cfg.dropped),
                       "computed": {label: phrase for label, (_, phrase) in computed.items()}}


# ----------------------------------------------------------------------------- E. links

def link_checks(cfg: Config, docs: dict[str, str], live: dict[str, str], main_files: set[str], checks: dict,
                info: dict) -> None:
    targets = {rel: [t for _, t in LINK.findall(FENCE.sub("", body(text)))] for rel, text in docs.items()}
    anchors = {rel: headings(text) for rel, text in docs.items()}
    bad_relative, github_missing, unknown = [], [], []
    live_urls = {u for text in live.values() for u in URL.findall(text)}
    dirs = {"/".join(f.split("/")[:i]) for f in main_files for i in range(1, f.count("/") + 1)}
    for rel, links in targets.items():
        for target in links:
            if not re.match(r"[a-z]+:", target):
                path, _, anchor = target.partition("#")
                path = path or rel
                if path not in docs or (anchor and anchor not in anchors[path]):
                    bad_relative.append(f"{rel}: {target}")
        for url in URL.findall(FENCE.sub("", docs[rel])):
            if url.startswith(GITHUB):
                match = re.fullmatch(re.escape(GITHUB) + r"(tree|blob)/main/(.+?)/?", url)
                if not match or match[2] not in (dirs if match[1] == "tree" else main_files):
                    github_missing.append(f"{rel}: {url}")
            elif url not in live_urls and url not in cfg.allowed_urls:
                unknown.append(f"{rel}: {url}")
    linked = {rel: {t.partition("#")[0] for t in links} for rel, links in targets.items()}
    checks["relative links resolve to documents and headings"] = not bad_relative
    checks["card links every detail document; each links back"] = (
        set(cfg.detail_docs) <= linked.get("README.md", set())
        and all("README.md" in linked.get(rel, set()) for rel in cfg.detail_docs))
    checks["GitHub links point at paths on origin/main"] = not github_missing
    checks["other absolute URLs appear in the live documents or are allowed"] = not unknown
    info["links"] = {"bad_relative": bad_relative, "github_missing": github_missing, "unknown_urls": unknown,
                     "linked": {rel: sorted(v) for rel, v in linked.items()}}


# ----------------------------------------------------------------------------- A-E together

def prepare(cfg: Config, hub, main_files: set[str], sources: dict, docs: dict[str, str], checks: dict,
            info: dict) -> dict[str, str]:
    """Record every check before staging; return the staged documents {name: text}. docs is the full
    rendered document set."""
    live = hub_checks(cfg, hub, sources, checks, info)
    cfg.check_sources(sources, checks, info)
    render_checks(cfg, docs, live, checks, info)
    number_checks(cfg, docs, live, sources, checks, info)
    link_checks(cfg, docs, live, main_files, checks, info)
    return {rel: docs[rel] for rel in cfg.staged if rel in docs}


# ----------------------------------------------------------------------------- F. staging

def bundle_manifest(repo_id: str, parent: str, files: dict) -> str:
    """The canonical manifest AJ's approval covers: the repository, the pinned parent and every uploaded
    file's name, SHA-256 and size, in sorted order (Codex card-update-x2 r1)."""
    lines = [f"repo {repo_id}", f"parent {parent}"]
    lines += [f"file {rel} {files[rel]['sha256']} {files[rel]['bytes']}" for rel in sorted(files)]
    return "\n".join(lines) + "\n"


def bundle_sha256(repo_id: str, parent: str, files: dict) -> str:
    return sha(bundle_manifest(repo_id, parent, files).encode())


def staged_listing(folder: Path) -> dict:
    return {p.relative_to(folder).as_posix(): {"sha256": sha(p.read_bytes()), "bytes": p.stat().st_size}
            for p in sorted(folder.rglob("*")) if p.is_file()}


def write_json(path: Path, data) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def stage(cfg: Config, staging: Path, prepare_fn) -> tuple[bool, dict, Path]:
    """Run every check and stage the changed documents under staging; (passed, report, report path).

    prepare_fn(checks, info) records the A-E checks and returns {staged name: text}. Any failure,
    including an exception or SystemExit, leaves no final and no temporary folder, and only the
    FAILED report."""
    name = cfg.name
    folder, tmp = staging / name, staging / f".{name}.tmp-{os.getpid()}"
    report_ok, report_failed = staging / f"{name}.stage.json", staging / f"{name}.stage.FAILED.json"
    staging.mkdir(parents=True, exist_ok=True)
    for stale in (report_ok, report_failed):
        stale.unlink(missing_ok=True)
    for stale in (folder, *staging.glob(f".{name}.tmp-*")):
        if stale.exists():
            shutil.rmtree(stale)
    checks: dict = {}
    info: dict = {"repo_id": cfg.repo_id, "pinned_revision": cfg.pinned_revision}
    try:
        docs = prepare_fn(checks, info)
        failed = sorted(k for k, ok in checks.items() if ok is not True)
        if failed:
            raise StageFailed(f"{len(failed)} check(s) failed before staging: {failed}")
        tmp.mkdir()
        for rel, text in docs.items():
            (tmp / rel).write_bytes(text.encode())
        listing = staged_listing(tmp)
        checks["staged file set == the changed documents"] = sorted(listing) == sorted(cfg.staged)
        for rel in cfg.staged:
            checks[f"staged {rel} SHA-256 == rendered"] = (
                rel in docs and listing.get(rel, {}).get("sha256") == sha(docs[rel].encode()))
        required = required_checks(cfg)
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
                  "bundle_manifest": bundle_manifest(cfg.repo_id, cfg.pinned_revision, listing),
                  "bundle_sha256": bundle_sha256(cfg.repo_id, cfg.pinned_revision, listing),
                  "upload_command": cfg.upload_command,
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

def upload(cfg: Config, hub, folder: Path, report: dict, approved_bundle_sha256: str, checks: dict,
           info: dict) -> dict | None:
    """Open the Hub PR, the only Hub write, if every upload check passes; else None. The bytes that are
    hashed are the bytes uploaded, as one commit of exactly the staged documents on parent_commit = the pin
    (never on a newer main). AJ's approval is the bundle digest of those bytes (bundle_sha256), so a restage
    that changes any uploaded file invalidates it."""
    staged = report.get("files", {})
    checks["card-update stage report PASSED"] = (
        report.get("status") == "PASSED" and sorted(staged) == sorted(cfg.staged)
        and Path(report.get("folder", "")) == Path(folder))
    present = sorted(p.relative_to(folder).as_posix() for p in Path(folder).rglob("*") if p.is_file()) \
        if Path(folder).is_dir() else []
    data = {rel: (Path(folder) / rel).read_bytes() for rel in cfg.staged if rel in present}
    listing = {rel: {"sha256": sha(b), "bytes": len(b)} for rel, b in data.items()}
    checks["staged files == stage report (exact set, SHA-256 and size)"] = (
        present == sorted(cfg.staged) and listing == staged)
    bundle = bundle_sha256(cfg.repo_id, cfg.pinned_revision, listing)
    checks["approved bundle SHA-256 == bundle of the bytes to upload"] = (
        bool(re.fullmatch(r"[0-9a-f]{64}", approved_bundle_sha256 or ""))
        and approved_bundle_sha256 == bundle == report.get("bundle_sha256"))
    head = hub.head()
    checks["main is still the pinned revision"] = head == cfg.pinned_revision
    info["upload"] = {"folder": str(folder), "main": head, "parent_commit": cfg.pinned_revision, "files": listing,
                      "bundle_sha256": bundle, "bundle_manifest": bundle_manifest(cfg.repo_id, cfg.pinned_revision, listing)}
    if set(checks) != upload_checks() or any(ok is not True for ok in checks.values()):
        return None
    pr = hub.create_pr(data, cfg.pinned_revision, cfg.commit_message)
    info["upload"]["pr"] = pr
    return pr


def verify(cfg: Config, hub, revision: str, report: dict, checks: dict, info: dict, pr_head: str | None = None) -> None:
    """Record verify_checks(cfg, revision) for refs/pr/N (before the merge) or main (after it; pr_head
    is the verified PR head commit, from the pre-merge report)."""
    staged = report.get("files", {})
    pin = cfg.pinned_revision
    checks["card-update stage report PASSED"] = report.get("status") == "PASSED" and sorted(staged) == sorted(cfg.staged)
    commits = hub.commits(revision)
    since = commits[:commits.index(pin)] if pin in commits else None
    if revision.startswith("refs/pr/"):
        checks["main is still the pinned revision"] = hub.head() == pin
        checks["revision is one commit on the pinned revision"] = since is not None and len(since) == 1
    else:
        # A fast-forward or squash adds one commit; a merge commit adds itself on top of the PR head.
        checks["commits since the pinned revision are only the card update"] = since is not None and (
            len(since) == 1 or (len(since) == 2 and pr_head is not None and since[1] == pr_head))
    pinned, now = hub.files(pin), hub.files(revision)
    checks["revision file set == pinned files + new documents"] = set(now) == set(pinned) | set(cfg.docs)
    for rel in cfg.unchanged:
        checks[f"revision {rel} unchanged from pinned"] = rel in now and rel in pinned and now[rel] == pinned[rel]
    for rel in cfg.staged:
        want = staged.get(rel)
        data = hub.read(rel, revision) if rel in now else None
        checks[f"revision {rel} == staged"] = (want is not None and data is not None
                                               and sha(data) == want["sha256"] and len(data) == want["bytes"])
    info["verify"] = {"revision": revision, "revision_sha": commits[0] if commits else None,
                      "commits_since_pin": since, "pr_head": pr_head,
                      "files": {rel: meta._asdict() for rel, meta in sorted(now.items())}}


def pre_merge(cfg: Config, hub, revision: str, verify_report: dict, checks: dict, info: dict) -> None:
    """Run immediately before AJ merges: nothing moved since `verify revision` passed."""
    verified = verify_report.get("verify", {})
    checks["verify report for the PR PASSED"] = (
        verify_report.get("status") == "PASSED" and verified.get("revision") == revision
        and bool(verified.get("revision_sha")))
    head = hub.head()
    checks["main is still the pinned revision"] = head == cfg.pinned_revision
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


def main(cfg: Config, prepare_stage: Callable, repo: Path, argv: list[str] | None = None, doc: str = "") -> None:
    """The CLI. prepare_stage(hub, main_files, checks, info) runs prepare() on the model's sources."""
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("stage", help="check and stage the card update (no upload)")
    u = sub.add_parser("upload", help="open the Hub PR (the only Hub write), on the pinned parent")
    u.add_argument("--approved-bundle-sha256", required=True,
                   help="the stage report's bundle_sha256 (repository, parent and every staged file) that AJ approved")
    v = sub.add_parser("verify", help="check refs/pr/N (before the merge) or main (after it)")
    v.add_argument("revision")
    m = sub.add_parser("pre-merge", help="run immediately before merging: nothing moved since verify")
    m.add_argument("revision")
    args = parser.parse_args(argv)
    hub = HubClient(cfg.repo_id, cfg.tmp_root)
    name, staging = cfg.name, cfg.staging
    try:
        if args.command == "stage":
            def run(checks: dict, info: dict) -> dict:
                main_files = origin_main_files(repo)
                info["origin_main"] = subprocess.run(["git", "-C", str(repo), "rev-parse", "origin/main"],
                                                     check=True, capture_output=True, text=True).stdout.strip()
                return prepare_stage(hub, main_files, checks, info)

            passed, report, path = stage(cfg, staging, run)
            print(json.dumps(report, indent=1, default=str))
            if not passed:
                print(f"\nSTAGING FAILED; no upload folder was left. Report: {path}", file=sys.stderr)
                sys.exit(1)
            print(f"\nAll {len(report['checks'])} checks passed (report: {path}). Nothing was uploaded.\n"
                  + "".join(f"Staged {rel} SHA-256: {v['sha256']}\n" for rel, v in sorted(report["files"].items()))
                  + f"Bundle SHA-256 (the approval token): {report['bundle_sha256']}\n" + f"{report['bundle_manifest']}"
                  f"After review and AJ's approval:\n  {report['upload_command']}")
            return
        report = read_json(staging / f"{name}.stage.json")
        checks: dict = {}
        info: dict = {"repo_id": cfg.repo_id, "pinned_revision": cfg.pinned_revision}
        if args.command == "upload":
            pr = upload(cfg, hub, staging / name, report, args.approved_bundle_sha256, checks, info)
            out = _finish(staging / f"{name}.upload.json", "upload", checks, upload_checks(), info)
            if pr is None or out["status"] != "PASSED":
                print("Refused; nothing was written to the Hub.", file=sys.stderr)
                sys.exit(1)
            print(f"Opened {pr['pr_url']}. Next: ./python hf/stage_card.py verify {pr['pr_revision']}")
            return
        if args.command == "verify":
            pr_head = None
            if not args.revision.startswith("refs/pr/"):
                records = sorted(staging.glob(f"{name}.pre-merge-*.json"), key=lambda p: p.stat().st_mtime)
                passed_records = [read_json(p) for p in records if read_json(p).get("status") == "PASSED"]
                pr_head = passed_records[-1]["pre_merge"]["pr_head"] if passed_records else None
            verify(cfg, hub, args.revision, report, checks, info, pr_head=pr_head)
            out = _finish(staging / f"{name}.verify-{_safe(args.revision)}.json", f"verify {args.revision}",
                          checks, verify_checks(cfg, args.revision), info)
        else:
            if not args.revision.startswith("refs/pr/"):
                parser.error("pre-merge takes refs/pr/N")
            verified = read_json(staging / f"{name}.verify-{_safe(args.revision)}.json")
            pre_merge(cfg, hub, args.revision, verified, checks, info)
            out = _finish(staging / f"{name}.pre-merge-{_safe(args.revision)}.json", "pre-merge", checks,
                          pre_merge_checks(), info)
            if out["status"] == "PASSED":
                print("Merge the PR on the Hub now; no other Hub writes until then. Then: "
                      "./python hf/stage_card.py verify main")
        if out["status"] != "PASSED":
            sys.exit(1)
    finally:
        hub.close()
