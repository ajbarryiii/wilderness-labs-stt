"""Copy artifact files from NixOS to the Mac artifact directory through the local SSH helper (NixOS only).

The helper is configured outside Git: WP3_MAC_RUN in the environment or in the untracked mil/local.json (the
same setting mil/archive.py uses; `$WP3_MAC_RUN -- CMD...` runs CMD on the Mac). It has no stdin and passes the
remote command as one argument, which Linux limits to 128 KiB; files therefore travel as base64 chunks of CHUNK bytes appended to
<dest>.part, and are renamed into place only when the Mac's SHA-256 equals the local one. Sources must be under
the NixOS artifact area and destinations under the Mac artifact root (both checked). Appending decoded chunks is
I/O only; anything that computes on the Mac still goes through ios/macguard.

  python macpush.py SRC_DIR_OR_FILE... --dest MAC_DIR [--skip-existing]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import subprocess
import sys
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
MAC_ROOT = PurePosixPath("/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios")
CHUNK = 90_000


def mac(script: str, *args: str) -> str:
    from mil.archive import local_config

    out = subprocess.run([*local_config("WP3_MAC_RUN").split(), "--", "sh", "-c", script, "sh", *args],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"remote command failed ({out.returncode}): {out.stderr.strip()[:300]}")
    return out.stdout


def push(src: Path, dest: PurePosixPath, skip_existing: bool) -> dict:
    data = src.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    have = mac('[ -f "$1" ] && shasum -a 256 "$1" | cut -d" " -f1 || true', str(dest)).strip()
    if have == digest and skip_existing:
        return {"file": src.name, "bytes": len(data), "sha256": digest, "skipped": True}
    part = str(dest) + ".part"
    mac('mkdir -p "$(dirname "$1")" && : > "$1"', part)
    for i in range(0, len(data), CHUNK):
        mac('printf %s "$1" | base64 -d >> "$2"', base64.b64encode(data[i:i + CHUNK]).decode(), part)
    remote = mac('shasum -a 256 "$1" | cut -d" " -f1', part).strip()
    if remote != digest:
        raise RuntimeError(f"{src}: Mac SHA-256 {remote} != {digest}")
    mac('mv "$1" "$2"', part, str(dest))
    return {"file": src.name, "bytes": len(data), "sha256": digest, "chunks": -(-len(data) // CHUNK)}


def main() -> None:
    import artifacts

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sources", nargs="+")
    parser.add_argument("--dest", required=True)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    dest_root = PurePosixPath(args.dest)
    if not (dest_root == MAC_ROOT or MAC_ROOT in dest_root.parents) or ".." in dest_root.parts:
        raise SystemExit(f"destination must be under {MAC_ROOT}")
    for s in args.sources:
        src = artifacts.check(s)
        files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file())
        for f in files:
            rel = f.name if src.is_file() else f.relative_to(src).as_posix()
            result = push(f, dest_root / rel, args.skip_existing)
            print(" ".join(f"{k}={v}" for k, v in result.items()), flush=True)


if __name__ == "__main__":
    main()
