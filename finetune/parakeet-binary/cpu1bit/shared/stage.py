"""Stage the cpu1bit Cargo workspace into a clean directory (for a NixOS check build or a Mac push).

Copies the workspace root files (Cargo.toml, Cargo.lock if present, rust-toolchain.toml,
.cargo/config.toml) and every member directory listed in Cargo.toml that has a Cargo.toml (members
without one yet are dropped from the staged `members` list and recorded). Build outputs, Python
caches and large data never travel. Writes STAGED.json with the member list and the SHA-256 of
every staged file; its `tree_sha256` identifies the staged source.

    finetune/parakeet-binary/python cpu1bit/shared/stage.py OUT_DIR [--members a,b]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

CPU1BIT = Path(__file__).resolve().parent.parent
LIVE = CPU1BIT
ROOT_FILES = ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".cargo/config.toml"]
SKIP_DIRS = {"target", "__pycache__", ".pytest_cache", "data"}
SKIP_SUFFIXES = {".pyc", ".bin", ".f32", ".safetensors", ".plist", ".log"}
MAX_FILE = 4 << 20  # source files only


def listed_members(cargo_toml: str) -> list[str]:
    m = re.search(r"^members\s*=\s*\[(.*?)\]", cargo_toml, re.S | re.M)
    if not m:
        raise SystemExit("Cargo.toml: no members list")
    return re.findall(r'"([^"]+)"', m.group(1))


def registry_packages(lock: str) -> set[tuple[str, str]]:
    """(name, version) of every registry package in a Cargo.lock."""
    out = set()
    for block in lock.split("[[package]]")[1:]:
        name = re.search(r'^name = "([^"]+)"', block, re.M)
        ver = re.search(r'^version = "([^"]+)"', block, re.M)
        if name and ver and re.search(r'^source = "registry', block, re.M):
            out.add((name.group(1), ver.group(1)))
    return out


def stage(out: Path, only: list[str] | None = None, root: Path | None = None) -> dict:
    """Stage the workspace at `root` (default: the live cpu1bit tree)."""
    global CPU1BIT
    CPU1BIT = (root or LIVE).resolve()
    cargo = (CPU1BIT / "Cargo.toml").read_text()
    listed = listed_members(cargo)
    present = [m for m in listed if (CPU1BIT / m / "Cargo.toml").is_file() and (only is None or m in only)]
    if "shared/core" not in present:
        raise SystemExit("shared/core is required")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    staged_cargo = re.sub(r"^members\s*=\s*\[.*?\]", "members = [" + ", ".join(f'"{m}"' for m in present) + "]",
                          cargo, count=1, flags=re.S | re.M)
    files: dict[str, str] = {}

    def put(rel: str, data: bytes) -> None:
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        files[rel] = hashlib.sha256(data).hexdigest()

    for rel in ROOT_FILES:
        src = CPU1BIT / rel
        if src.is_file():
            put(rel, staged_cargo.encode() if rel == "Cargo.toml" else src.read_bytes())
    # shared/schedule.json is embedded by shared/core (include_str!).
    put("shared/schedule.json", (CPU1BIT / "shared/schedule.json").read_bytes())
    # Mac-side tools (lock, cargo wrapper, harness scripts).
    for p in sorted((CPU1BIT / "shared/mac").rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            put(f"shared/mac/{p.relative_to(CPU1BIT / 'shared/mac').as_posix()}", p.read_bytes())
    for member in present:
        base = CPU1BIT / member
        for p in sorted(base.rglob("*")):
            rel_parts = p.relative_to(base).parts
            if not p.is_file() or any(part in SKIP_DIRS for part in rel_parts) or p.suffix in SKIP_SUFFIXES:
                continue
            if p.stat().st_size > MAX_FILE:
                raise SystemExit(f"{p}: {p.stat().st_size} bytes; source files only")
            put(f"{member}/{'/'.join(rel_parts)}", p.read_bytes())
    # Members not in the pinned Cargo.lock are added minimally (cargo metadata, offline); registry
    # packages must stay exactly as pinned, so `cargo --locked` on the Mac builds the same deps.
    pinned = registry_packages((CPU1BIT / "Cargo.lock").read_text())
    subprocess.run(["cargo", "metadata", "--offline", "--format-version", "1"], cwd=out, check=True,
                   stdout=subprocess.DEVNULL)
    lock = (out / "Cargo.lock").read_bytes()
    if registry_packages(lock.decode()) - pinned:
        raise SystemExit(f"staged members need registry packages outside the pinned Cargo.lock: "
                         f"{sorted(registry_packages(lock.decode()) - pinned)}; ask H")
    files["Cargo.lock"] = hashlib.sha256(lock).hexdigest()
    tree = hashlib.sha256("".join(f"{h}  {r}\n" for r, h in sorted(files.items())).encode()).hexdigest()
    record = {"source": str(CPU1BIT), "staged_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "listed_members": listed,
              "members": present, "dropped_members": [m for m in listed if m not in present],
              "tree_sha256": tree, "files": files}
    (out / "STAGED.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return record


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out", type=Path)
    p.add_argument("--members", help="comma-separated subset of the listed members")
    p.add_argument("--root", type=Path, help="workspace root to stage (default: the live tree)")
    a = p.parse_args()
    rec = stage(a.out.resolve(), a.members.split(",") if a.members else None, a.root)
    print(json.dumps({k: rec[k] for k in ("members", "dropped_members", "tree_sha256")}))


if __name__ == "__main__":
    main()
