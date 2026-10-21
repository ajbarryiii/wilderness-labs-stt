"""Pull a results directory from the Mac to NixOS for analysis (hash-checked; Codex code review c2 finding 3).

    finetune/parakeet-binary/python cpu1bit/shared/pull.py <session>/<stage>
-> /mnt/hd/wilderness-labs-stt/parakeet-binary/cpu1bit/results/<session>/<stage>/ . The Mac lists every
file's SHA-256 and streams a tar over SSH stdout (the helper gives SSH no stdin, but stdout streams);
the archive is unpacked into a fresh temporary directory and every SHA-256 verified; `merge` then
applies, per file:
- immutable evidence (everything else; the runner writes it once, atomically): copied if absent,
  accepted if identical, and a differing existing file ABORTS the pull (evidence is never replaced);
- append-only logs (`*.log.jsonl`, one per runner invocation): copied if absent, or extended if the
  pulled content starts with the existing content (a later pull of a resumed or still-running stage);
  anything else aborts;
- in-progress raw streams (`<stem>.powermetrics.plist0`, `<stem>.stderr` whose `<stem>.json` record does
  not exist yet) and runner temporaries (`.tmp-*`) are skipped; they arrive with a later pull.
Repeated pulls are therefore safe at any time (pull, resume, pull).
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

MAC_RESULTS = None  # the Mac's results dir: see mac_results() (no user name in the source)
NIX_RESULTS = Path("/mnt/hd/wilderness-labs-stt/parakeet-binary/cpu1bit/results")
RUN = Path.home() / ".claude/skills/macos-ssh/scripts/run.sh"
RAW_SUFFIXES = (".powermetrics.plist0", ".stderr")
LOG_SUFFIX = ".log.jsonl"


class PullConflict(Exception):
    pass


def sha256(p: Path) -> str:
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def classify(name: str, names: set[str]) -> str:
    base = os.path.basename(name)
    if base.startswith(".tmp-"):
        return "skip"
    if name.endswith(LOG_SUFFIX):
        return "log"
    for suf in RAW_SUFFIXES:
        if name.endswith(suf) and name[:-len(suf)] + ".json" not in names:
            return "skip"
    return "evidence"


def merge(src: Path, dest: Path, digests: dict[str, str]) -> dict:
    """Applies the rules above from an unpacked, digest-listed tree `src` to `dest`. Checks everything
    before changing anything, so a conflict leaves `dest` untouched."""
    names = set(digests)
    actions = []
    for name, digest in sorted(digests.items()):
        kind = classify(name, names)
        if kind == "skip":
            actions.append(("skip", name))
            continue
        s, d = src / name, dest / name
        if sha256(s) != digest:
            raise PullConflict(f"{name}: SHA-256 mismatch after transfer")
        if not d.exists():
            actions.append(("copy", name))
        elif sha256(d) == digest:
            actions.append(("same", name))
        elif kind == "log":
            old = d.read_bytes()
            if not s.read_bytes().startswith(old):
                raise PullConflict(f"{d}: log is not an extension of the pulled copy; refusing")
            actions.append(("extend", name))
        else:
            raise PullConflict(f"{d} exists with different content; refusing to replace evidence")
    for act, name in actions:
        if act in ("copy", "extend"):
            d = dest / name
            d.parent.mkdir(parents=True, exist_ok=True)
            tmp = d.with_name(f".pull-{d.name}.{os.getpid()}")
            shutil.copyfile(src / name, tmp)
            os.replace(tmp, d)
    counts: dict = {}
    for act, _ in actions:
        counts[act] = counts.get(act, 0) + 1
    return counts


def mac_results() -> PurePosixPath:
    """$CPU1BIT_MAC_ROOT/results, else the Mac's $HOME/wilderness-labs-stt-artifacts/cpu1bit/results."""
    global MAC_RESULTS
    if MAC_RESULTS is None:
        root = os.environ.get("CPU1BIT_MAC_ROOT")
        if not root:
            home = subprocess.run([str(RUN), "--", "sh", "-c", 'printf %s "$HOME"'], capture_output=True, text=True,
                                  check=True).stdout.strip()
            if not home.startswith("/"):
                raise RuntimeError(f"cannot read the Mac's $HOME ({home!r})")
            root = home + "/wilderness-labs-stt-artifacts/cpu1bit"
        MAC_RESULTS = PurePosixPath(root) / "results"
    return MAC_RESULTS


def fetch(rel: PurePosixPath, tmp: str) -> dict[str, str]:
    sums = subprocess.run([str(RUN), "--", "sh", "-c", 'cd "$1" && find . -type f -exec shasum -a 256 {} +', "sh",
                           str(mac_results() / rel)], capture_output=True, text=True, check=True).stdout
    want = {}
    for line in sums.splitlines():
        digest, name = line.split(None, 1)
        want[os.path.normpath(name)] = digest
    tar = subprocess.Popen([str(RUN), "--", "sh", "-c", 'cd "$1" && COPYFILE_DISABLE=1 tar cf - .', "sh",
                            str(mac_results() / rel)], stdout=subprocess.PIPE)
    un = subprocess.run(["tar", "xf", "-", "-C", tmp, "--warning=no-unknown-keyword"], stdin=tar.stdout)
    if tar.wait() != 0 or un.returncode != 0:
        raise SystemExit("tar stream failed")
    # Files still growing between the listing and the tar (raw streams, logs) are re-hashed here: an
    # evidence file must match its listed digest; growing files keep the listed prefix semantics.
    for name in list(want):
        p = Path(tmp) / name
        if not p.exists():
            raise SystemExit(f"{name}: listed but not in the archive")
        if classify(name, set(want)) in ("log", "skip"):
            want[name] = sha256(p)
    return want


def main() -> None:
    rel = PurePosixPath(sys.argv[1])
    if rel.is_absolute() or ".." in rel.parts or len(rel.parts) != 2:
        raise SystemExit("usage: pull.py <session>/<stage>")
    if not os.path.ismount("/mnt/hd"):
        raise SystemExit("/mnt/hd is not mounted")
    dest = NIX_RESULTS / rel
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=NIX_RESULTS) as tmp:
        want = fetch(rel, tmp)
        try:
            counts = merge(Path(tmp), dest, want)
        except PullConflict as e:
            raise SystemExit(str(e))
    print(f"{rel}: {len(want)} files listed, {counts} -> {dest}")


if __name__ == "__main__":
    main()
