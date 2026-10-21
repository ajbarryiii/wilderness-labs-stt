"""Hash-checked transfer from NixOS to the Mac's cpu1bit artifacts (plans/cpu-benchmark-mac.md M5).

Same mechanism as finetune/parakeet-ternary/ios/macpush.py (SHA-256 computed on NixOS, recomputed
on the Mac on a `.part` file, renamed into place only when equal) with the transport of
ios/mil/archive.py's restore: the macos-ssh helper gives SSH no stdin, so the Mac pulls each file
with curl from a one-shot, token-gated HTTP server bound to this machine's tailnet address. Refuses
unless the Mac volume keeps at least 20 GiB free after the transfer.

    python push.py code --agent H                 # stage the workspace -> MAC/ws/H (source tree)
    python push.py files SRC... --dest REL        # data files -> MAC/data/REL/<name> (dirs recursively)
    python push.py rm --dest REL                  # delete MAC/data/REL (streamed exactness batches)

Every transfer appends a JSON record to /mnt/hd/wilderness-labs-stt/parakeet-binary/cpu1bit/push-log.jsonl.
Configuration: CPU1BIT_SERVE_HOST (default: WP3_SERVE_HOST in the untracked ios/mil/local.json).
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import os
import secrets
import shlex
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage  # noqa: E402

MAC_ROOT = None  # the Mac artifacts root: see mac_root() (no user name in the source)
NIX_ROOT = Path("/mnt/hd/wilderness-labs-stt/parakeet-binary/cpu1bit")
RUN = Path.home() / ".claude/skills/macos-ssh/scripts/run.sh"
FLOOR = 20 * 2**30
AGENTS = ("H", "H-smoke", "V", "T", "P", "frozen")


def serve_host() -> str:
    if os.environ.get("CPU1BIT_SERVE_HOST"):
        return os.environ["CPU1BIT_SERVE_HOST"]
    cfg = HERE.parents[2] / "parakeet-ternary/ios/mil/local.json"
    host = json.loads(cfg.read_text()).get("WP3_SERVE_HOST") if cfg.exists() else None
    if not host:
        raise SystemExit("set CPU1BIT_SERVE_HOST to this machine's tailnet address")
    return host


def mac(script: str, *args: str) -> str:
    out = subprocess.run([str(RUN), "--", "sh", "-c", script, "sh", *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"Mac command failed ({out.returncode}): {out.stderr.strip()[-600:]}")
    return out.stdout


def mac_root() -> PurePosixPath:
    """$CPU1BIT_MAC_ROOT, else the Mac's $HOME + /wilderness-labs-stt-artifacts/cpu1bit (asked once)."""
    global MAC_ROOT
    if MAC_ROOT is None:
        env = os.environ.get("CPU1BIT_MAC_ROOT")
        home = None if env else mac('printf %s "$HOME"').strip()
        if not env and not (home or "").startswith("/"):
            raise RuntimeError(f"cannot read the Mac's $HOME ({home!r})")
        MAC_ROOT = PurePosixPath(env) if env else PurePosixPath(home) / "wilderness-labs-stt-artifacts" / "cpu1bit"
    return MAC_ROOT


def mac_free_bytes() -> int:
    kb = mac('mkdir -p "$1" && df -k "$1" | awk \'NR == 2 {print $4}\'', str(mac_root())).strip()
    if not kb.isdigit():
        raise RuntimeError(f"cannot read the Mac's free disk space ({kb!r})")
    return int(kb) * 1024


def require_space(nbytes: int) -> int:
    free = mac_free_bytes()
    if free - nbytes < FLOOR:
        raise SystemExit(f"refusing: Mac has {free / 2**30:.1f} GiB free; {nbytes / 2**30:.2f} GiB would leave "
                         f"less than the 20 GiB floor")
    return free


def sha256_file(p: Path) -> str:
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


class OneShot:
    """Serves the given {url name: local path} map under a random token, bound to the tailnet address."""

    def __init__(self, files: dict[str, Path]):
        self.token = secrets.token_hex(16)
        token, table = self.token, dict(files)

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                prefix = f"/{token}/"
                name = self.path[len(prefix):] if self.path.startswith(prefix) else None
                path = table.get(name)
                if path is None:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(path.stat().st_size))
                self.end_headers()
                with open(path, "rb") as f:
                    while chunk := f.read(1 << 20):
                        self.wfile.write(chunk)

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer((serve_host(), 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def url(self, name: str) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/{self.token}/{name}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def log(record: dict) -> None:
    if not os.path.ismount("/mnt/hd"):
        raise SystemExit("/mnt/hd is not mounted")
    NIX_ROOT.mkdir(parents=True, exist_ok=True)
    with open(NIX_ROOT / "push-log.jsonl", "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


# Fetch one file: curl to <dest>.part, SHA-256 on the Mac, rename only when equal.
FETCH = r'''
set -eu
url=$1; dest=$2; want=$3
mkdir -p "$(dirname "$dest")"
curl -fsS --retry 3 -o "$dest.part" "$url"
got=$(shasum -a 256 "$dest.part" | cut -d' ' -f1)
if [ "$got" != "$want" ]; then rm -f "$dest.part"; echo "sha256 mismatch for $dest: $got != $want" >&2; exit 7; fi
mv -f "$dest.part" "$dest"
echo "$got"
'''


def push_files(sources: list[Path], dest_rel: str, skip_existing: bool) -> list[dict]:
    dest_root = mac_root() / "data" / PurePosixPath(dest_rel)
    if ".." in PurePosixPath(dest_rel).parts or PurePosixPath(dest_rel).is_absolute():
        raise SystemExit("--dest must be a relative path without ..")
    pairs: list[tuple[Path, PurePosixPath]] = []
    for s in sources:
        s = s.resolve()
        if not str(s).startswith("/mnt/hd/"):
            raise SystemExit(f"{s}: sources must be artifacts under /mnt/hd")
        files = [s] if s.is_file() else sorted(p for p in s.rglob("*") if p.is_file())
        for f in files:
            rel = f.name if s.is_file() else f.relative_to(s.parent).as_posix()
            pairs.append((f, dest_root / rel))
    total = sum(f.stat().st_size for f, _ in pairs)
    free = require_space(total)
    digests = {str(f): sha256_file(f) for f, _ in pairs}
    server = OneShot({f"f{i}": f for i, (f, _) in enumerate(pairs)})
    out = []
    try:
        for i, (f, dest) in enumerate(pairs):
            want = digests[str(f)]
            t0 = time.time()
            if skip_existing:
                have = mac('[ -f "$1" ] && shasum -a 256 "$1" | cut -d" " -f1 || true', str(dest)).strip()
                if have == want:
                    out.append({"file": str(f), "dest": str(dest), "sha256": want, "bytes": f.stat().st_size, "skipped": True})
                    continue
            got = mac(FETCH, server.url(f"f{i}"), str(dest), want).strip()
            assert got == want
            rec = {"file": str(f), "dest": str(dest), "sha256": want, "bytes": f.stat().st_size,
                   "seconds": round(time.time() - t0, 2)}
            out.append(rec)
            print(json.dumps(rec), flush=True)
    finally:
        server.close()
    log({"kind": "files", "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "dest": str(dest_root),
         "mac_free_before": free, "files": out})
    return out


def push_code(agent: str, members: list[str] | None, root: Path | None = None) -> dict:
    if agent not in AGENTS:
        raise SystemExit(f"--agent must be one of {AGENTS}")
    NIX_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=NIX_ROOT, prefix="stage-") as tmp:
        tree = Path(tmp) / "tree"
        rec = stage.stage(tree, members, root)
        sums = "".join(f"{h}  {r}\n" for r, h in sorted(rec["files"].items()))
        (tree / "SHA256SUMS").write_text(sums)
        tar_path = Path(tmp) / "tree.tar"
        with tarfile.open(tar_path, "w") as t:
            t.add(tree, arcname=".")
        require_space(2 * tar_path.stat().st_size)
        want = sha256_file(tar_path)
        server = OneShot({"tree.tar": tar_path})
        ws = mac_root() / "ws" / agent
        try:
            script = r'''
set -eu
url=$1; ws=$2; want=$3
mkdir -p "$(dirname "$ws")"
tmp="$ws.incoming"; rm -rf "$tmp" "$tmp.tar"; mkdir -p "$tmp"
curl -fsS --retry 3 -o "$tmp.tar" "$url"
got=$(shasum -a 256 "$tmp.tar" | cut -d' ' -f1)
[ "$got" = "$want" ] || { echo "tar sha256 mismatch: $got != $want" >&2; exit 7; }
tar xf "$tmp.tar" -C "$tmp" && rm -f "$tmp.tar"
(cd "$tmp" && shasum -a 256 -c --quiet SHA256SUMS)
rm -rf "$ws.old"; [ -d "$ws" ] && mv "$ws" "$ws.old"; mv "$tmp" "$ws"; rm -rf "$ws.old"
chmod +x "$ws/shared/mac/"* 2>/dev/null || true
echo ok
'''
            mac(script, server.url("tree.tar"), str(ws), want)
        finally:
            server.close()
    out = {"kind": "code", "agent": agent, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "dest": str(ws),
           "tree_sha256": rec["tree_sha256"], "members": rec["members"], "dropped_members": rec["dropped_members"],
           "tar_sha256": want}
    log(out)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("code")
    c.add_argument("--agent", required=True)
    c.add_argument("--members", help="comma-separated subset (default: every member with a Cargo.toml)")
    c.add_argument("--root", type=Path, help="workspace root to stage (default: the live tree)")
    f = sub.add_parser("files")
    f.add_argument("sources", nargs="+", type=Path)
    f.add_argument("--dest", required=True)
    f.add_argument("--skip-existing", action="store_true")
    r = sub.add_parser("rm")
    r.add_argument("--dest", required=True)
    a = p.parse_args()
    if a.cmd == "code":
        print(json.dumps(push_code(a.agent, a.members.split(",") if a.members else None, a.root)))
    elif a.cmd == "files":
        push_files(a.sources, a.dest, a.skip_existing)
    else:
        rel = PurePosixPath(a.dest)
        if ".." in rel.parts or rel.is_absolute() or not rel.parts:
            raise SystemExit("--dest must be a non-empty relative path without ..")
        target = mac_root() / "data" / rel
        mac('rm -rf -- "$1" && echo removed', str(target))
        log({"kind": "rm", "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "dest": str(target)})


if __name__ == "__main__":
    main()
