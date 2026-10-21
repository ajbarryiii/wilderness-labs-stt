"""Move compiled WP3 models off the shared Mac to NixOS storage and back (Mac disk floor 30 GB).

  ./python ios/mil/archive.py out MODEL ARM      # Mac <artifacts>/arms/MODEL/ARM/*.mlmodelc,*.mlpackage -> /mnt/hd
  ./python ios/mil/archive.py back MODEL ARM [fixed.mlmodelc,...]   # restore (all or some) to the Mac

Runs on NixOS. out: per-file SHA-256 on the Mac, a tar stream over SSH (each Mac side as a macguard job,
512 MB cap), extraction under /mnt/hd/wilderness-labs-stt/parakeet-ios/arms-archive/MODEL/ARM, verification
of every checksum, then deletion of the models on the Mac; the manifests stay and ARCHIVED.json records
what moved where. back: the Mac pulls a tar stream from a one-shot token-gated HTTP server on this machine
(the SSH wrapper gives no stdin), after checking inside the guarded job that 30 GB + the archive's size are free.

Local configuration (not in Git): environment variables, or the untracked file mil/local.json with the same keys:
  WP3_MAC_RUN    command that runs a remote shell command on the Mac: `$WP3_MAC_RUN --repo DIR -- CMD...`
  WP3_SERVE_HOST address of this machine as the Mac reaches it (the restore server binds to it)
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

FLOOR_GB = 30


def local_config(key: str) -> str:
    """WP3_* setting from the environment or the untracked mil/local.json (no host names in tracked code)."""
    if os.environ.get(key):
        return os.environ[key]
    path = Path(__file__).with_name("local.json")
    if path.exists() and json.loads(path.read_text()).get(key):
        return json.loads(path.read_text())[key]
    raise SystemExit(f"set {key} in the environment or in {path} (untracked)")


MAC_IOS = "/Users/ajbarry/workspace/github.com/wilderness-labs-stt/finetune/parakeet-ternary/ios"
MAC_ARMS = "/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios/arms"
LOCAL = Path("/mnt/hd/wilderness-labs-stt/parakeet-ios/arms-archive")


def guarded(cmd: str) -> list[str]:
    inner = f"./macguard --rss-cap 512M --timeout 3600 -- sh -c {shlex.quote(cmd)} 2>/dev/null"
    return [local_config("WP3_MAC_RUN"), "--repo", MAC_IOS, "--", "sh", "-c", inner]


def mac(cmd: str, tries: int = 40) -> str:
    """Run a guarded command on the Mac; a macguard refusal (exit 3) is retried every 180 s."""
    for _ in range(tries):
        proc = subprocess.run(guarded(cmd), capture_output=True, text=True)
        if proc.returncode != 3:
            proc.check_returncode()
            return proc.stdout
        time.sleep(180)
    raise SystemExit("macguard refused 40 times")


def out(model: str, arm: str) -> None:
    if not os.path.ismount("/mnt/hd"):
        raise SystemExit("/mnt/hd is not mounted")
    base = f"{MAC_ARMS}/{model}/{arm}"
    names = [n for n in mac(f"cd {base} && ls -d *.mlmodelc *.mlpackage 2>/dev/null || true").split() if n]
    if not names:
        raise SystemExit(f"nothing to archive in {base}")
    quoted = " ".join(shlex.quote(n) for n in names)
    sums = mac(f"cd {base} && find {quoted} -type f -exec shasum -a 256 {{}} +")
    dest = LOCAL / model / arm
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for _ in range(40):
        tar = subprocess.Popen(guarded(f"cd {base} && COPYFILE_DISABLE=1 tar cf - {quoted}"), stdout=subprocess.PIPE)
        local = subprocess.run(["tar", "xf", "-", "-C", str(dest), "--warning=no-unknown-keyword"], stdin=tar.stdout)
        status = tar.wait()
        if status == 3:  # refused by macguard before anything was sent
            time.sleep(180)
            continue
        if status != 0 or local.returncode != 0:
            raise SystemExit(f"tar stream failed ({status}, {local.returncode})")
        break
    else:
        raise SystemExit("macguard refused 40 times")
    bad, n = [], 0
    for line in sums.splitlines():
        digest, rel = line.split(None, 1)
        n += 1
        with open(dest / rel, "rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
                bad.append(rel)
    if bad or n == 0:
        raise SystemExit(f"checksum mismatch on {len(bad)} of {n} files; Mac copies kept")
    merged = {}
    if (dest / "SHA256SUMS").exists():  # an arm archived in several rounds (e.g. fixed/multi, then enum)
        for line in (dest / "SHA256SUMS").read_text().splitlines():
            merged[line.split(None, 1)[1]] = line
    for line in sums.splitlines():
        merged[line.split(None, 1)[1]] = line
    (dest / "SHA256SUMS").write_text("".join(merged[k] + "\n" for k in sorted(merged)))
    names = sorted({k.split("/", 1)[0] for k in merged})
    note = {"archived_to": f"nixos:{dest}", "models": names, "files": len(merged), "date": time.strftime("%Y-%m-%d %H:%M"),
            "restore": f"./python ios/mil/archive.py back {model} {arm}", "seconds": round(time.time() - t0, 1)}
    mac(f"cd {base} && rm -rf {quoted} && printf %s {shlex.quote(json.dumps(note))} > ARCHIVED.json")
    print(json.dumps(note))


def back(model: str, arm: str, only: list[str] | None = None) -> None:
    """Restore: the Mac pulls a tar stream from a one-shot, token-gated HTTP server bound to this machine's
    tailnet address (run.sh gives SSH no stdin), extracts it and verifies every SHA-256."""
    import http.server
    import secrets
    import threading

    src = LOCAL / model / arm
    sums = (src / "SHA256SUMS").read_text()
    names = sorted({line.split(None, 1)[1].split("/", 1)[0] for line in sums.splitlines()})
    if only:
        names = [n for n in names if n in only]
        sums = "".join(line + "\n" for line in sums.splitlines() if line.split(None, 1)[1].split("/", 1)[0] in names)
    token = secrets.token_hex(16)
    host = local_config("WP3_SERVE_HOST")
    need_gb = -(-sum(f.stat().st_size for n in names for f in (src / n).rglob("*") if f.is_file()) // 10 ** 9) + 1

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path != f"/{token}.tar":
                self.send_error(404)
                return
            self.send_response(200)
            self.end_headers()
            proc = subprocess.Popen(["tar", "cf", "-", "-C", str(src)] + names, stdout=subprocess.PIPE)
            while chunk := proc.stdout.read(1 << 20):
                self.wfile.write(chunk)
            proc.wait()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer((host, 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://{host}:{server.server_address[1]}/{token}.tar"
        base = f"{MAC_ARMS}/{model}/{arm}"
        mac(f"free=$(df -g {base} | awk 'NR == 2 {{print $4}}'); [ \"$free\" -ge {FLOOR_GB + need_gb} ] || "
            f"{{ echo 'refusing restore: '$free' GB free < {FLOOR_GB} + {need_gb} GB' >&2; exit 5; }}; "
            f"cd {base} && curl -fsS {url} | tar xf - && printf %s {shlex.quote(sums)} | shasum -a 256 -c --quiet")
    finally:
        server.shutdown()
    print(f"restored {names} to {MAC_ARMS}/{model}/{arm}")


if __name__ == "__main__":
    if sys.argv[1] == "back":
        back(sys.argv[2], sys.argv[3], sys.argv[4].split(",") if len(sys.argv) > 4 else None)
    else:
        out(sys.argv[2], sys.argv[3])
