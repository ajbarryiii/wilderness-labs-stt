"""Test helpers shared by the Kea-T and Kea-B card-update tests (plans/card-update-x2.md section 8): a fake
Hub repository and the "x2 runtime published" variant of the real sources and templates. No network."""
from __future__ import annotations

import copy
import hashlib
import re

import cardkit
import x2facts

GITATTRIBUTES = b"*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
OLDER = ["bc1c883f6d967398f32750a01c0e17d2e710a496", "4cd127f33f46ec1e2833cd8571bee5fd195dcbd7"]
TODO_MARKER = re.compile(r"\[TODO x2 PR:[^\]]*\]")
README_URL = cardkit.GITHUB + "blob/main/" + x2facts.MAIN_RUNTIME_README


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeHub:
    """A Hub repository at the pin, built from the release record's files (small ones as bytes, LFS ones as
    metadata) and the live documents, plus main edits, pull requests (create_pr records every call),
    merges and the public sibling repositories."""

    def __init__(self, pinned: str, small: dict[str, bytes], lfs: dict[str, tuple[int, str]],
                 siblings: dict[str, dict | None]) -> None:
        self.pinned = pinned
        self.head_sha = pinned
        self.content: dict[str, bytes] = {}
        self.meta: dict[str, cardkit.FileMeta] = {}
        for rel, (size, digest) in lfs.items():
            self.meta[rel] = cardkit.FileMeta(size, digest, f"blob-{rel}")
        for rel, data in small.items():
            self.put(rel, data)
        self.put(".gitattributes", GITATTRIBUTES)
        self.revisions: dict[str, tuple[dict, dict]] = {}
        self.log: dict[str, list[str]] = {pinned: [pinned, *OLDER]}
        self.created: list[dict] = []
        self.counter = 0
        self.siblings = siblings

    def _commit_id(self) -> str:
        self.counter += 1
        return f"{self.counter:040x}"

    def _resolve(self, revision: str) -> str:
        return self.head_sha if revision == "main" else revision

    def put(self, rel: str, data: bytes, revision: str | None = None) -> None:
        meta = cardkit.FileMeta(len(data), None, "blob-" + sha(data)[:40])
        if revision is None:
            self.content[rel], self.meta[rel] = data, meta
        else:
            self.revisions[revision][0][rel] = meta
            self.revisions[revision][1][rel] = data

    def fork(self, revision: str, parent: str) -> None:
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
        otherwise a merge commit whose history also holds main's commits since the PR's parent (the PR's
        files win, as in a silent overwrite)."""
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
        if revision != self.pinned:
            raise KeyError(revision)
        return self.meta, self.content

    def files(self, revision: str) -> dict:
        return dict(self._at(revision)[0])

    def read(self, rel: str, revision: str) -> bytes:
        return self._at(revision)[1][rel]

    def commits(self, revision: str) -> list[str]:
        return list(self.log[self._resolve(revision)])

    def public_files(self, repo_id: str) -> dict | None:
        files = self.siblings[repo_id]
        return None if files is None else dict(files)

    def create_pr(self, files: dict, parent: str, message: str) -> dict:
        self.created.append({"files": dict(files), "parent": parent, "message": message})
        num = len(self.created)
        revision = f"refs/pr/{num}"
        self.fork(revision, parent)
        for rel, data in files.items():
            self.put(rel, data, revision=revision)
        return {"pr_num": num, "pr_url": f"https://huggingface.co/fake/discussions/{num}",
                "commit": self.log[revision][0], "pr_revision": revision}


def published(sources: dict, templates: dict[str, str], main_files: set[str]) -> tuple[dict, dict, set[str]]:
    """The state after the x2 thread's PR merges: origin/main carries the gated selection, the runtime
    README, RESULTS.md and the measured kernels pinned by parts.py (whose hash is the freeze entry), and the
    runtime README is on the origin/main listing. (TODO markers, if any, are replaced by a link to it.)"""
    sources = copy.copy(sources)
    parts = {"kernels_parts.py": "1" * 64, "layer_parts.py": "2" * 64}
    sources["x2"] = dict(sources["x2"], main={"selection_sha256": x2facts.GATED_SELECTION_SHA256,
                                              "runtime_readme_sha256": "0" * 64, "results_sha256": "0" * 64,
                                              "parts_py_sha256": sources["x2"]["freeze_parts_py"],
                                              "parts_pinned": dict(parts), "measured_parts": dict(parts)})
    linked = {rel: TODO_MARKER.sub(f"([runtime README]({README_URL}))", text) for rel, text in templates.items()}
    return sources, linked, set(main_files) | {x2facts.MAIN_RUNTIME_README}


def unpublished(sources: dict, main_files: set[str]) -> tuple[dict, set[str]]:
    """origin/main as it was before the x2 thread's PR #6 merged: the x3 selection pointer, no runtime README
    and no measured kernels (RESULTS.md and parts.py were already there as a draft)."""
    sources = copy.copy(sources)
    main = dict(sources["x2"]["main"], selection_sha256="103d96f4defdf704fab09c4aa958053c8355194be6f531d4cf945be3f82af1cc",
                runtime_readme_sha256=None, measured_parts={})
    sources["x2"] = dict(sources["x2"], main=main)
    return sources, set(main_files) - {x2facts.MAIN_RUNTIME_README}
