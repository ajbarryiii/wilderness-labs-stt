"""Move compiled WP3 models off the shared Mac to NixOS storage and back (disk floor: Mac free > 60 GB).

  ./python ios/mil/archive.py out MODEL ARM      # Mac <artifacts>/arms/MODEL/ARM/*.mlmodelc,*.mlpackage -> /mnt/hd
  ./python ios/mil/archive.py back MODEL ARM     # restore them to the Mac (when the latency sweep needs them)

Runs on NixOS. out: per-file SHA-256 on the Mac, a tar stream over SSH (each Mac side as a macguard job,
512 MB cap), extraction under /mnt/hd/wilderness-labs-stt/parakeet-ios/arms-archive/MODEL/ARM, verification
of every checksum, then deletion of the models on the Mac; the manifests stay and ARCHIVED.json records
what moved where. back: the reverse (tar stream into the Mac, checksums verified there, ARCHIVED.json removed).
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

RUN = str(Path.home() / ".codex/skills/macos-ssh/scripts/run.sh")
MAC_IOS = "/Users/ajbarry/workspace/github.com/wilderness-labs-stt/finetune/parakeet-ternary/ios"
MAC_ARMS = "/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios/arms"
LOCAL = Path("/mnt/hd/wilderness-labs-stt/parakeet-ios/arms-archive")


def guarded(cmd: str) -> list[str]:
    inner = f"./macguard --rss-cap 512M --timeout 3600 -- sh -c {shlex.quote(cmd)} 2>/dev/null"
    return [RUN, "--repo", MAC_IOS, "--", "sh", "-c", inner]


def mac(cmd: str) -> str:
    return subprocess.run(guarded(cmd), check=True, capture_output=True, text=True).stdout


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
    tar = subprocess.Popen(guarded(f"cd {base} && COPYFILE_DISABLE=1 tar cf - {quoted}"), stdout=subprocess.PIPE)
    subprocess.run(["tar", "xf", "-", "-C", str(dest), "--warning=no-unknown-keyword"], stdin=tar.stdout, check=True)
    if tar.wait() != 0:
        raise SystemExit("tar stream failed")
    bad, n = [], 0
    for line in sums.splitlines():
        digest, rel = line.split(None, 1)
        n += 1
        with open(dest / rel, "rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
                bad.append(rel)
    if bad or n == 0:
        raise SystemExit(f"checksum mismatch on {len(bad)} of {n} files; Mac copies kept")
    (dest / "SHA256SUMS").write_text(sums)
    note = {"archived_to": f"nixos:{dest}", "models": names, "files": n, "date": time.strftime("%Y-%m-%d %H:%M"),
            "restore": f"./python ios/mil/archive.py back {model} {arm}", "seconds": round(time.time() - t0, 1)}
    mac(f"cd {base} && rm -rf {quoted} && printf %s {shlex.quote(json.dumps(note))} > ARCHIVED.json")
    print(json.dumps(note))


def back(model: str, arm: str, host: str = "100.81.222.117") -> None:
    """Restore: the Mac pulls a tar stream from a one-shot, token-gated HTTP server bound to this machine's
    tailnet address (run.sh gives SSH no stdin), extracts it and verifies every SHA-256."""
    import http.server
    import secrets
    import threading

    src = LOCAL / model / arm
    sums = (src / "SHA256SUMS").read_text()
    names = sorted({line.split(None, 1)[1].split("/", 1)[0] for line in sums.splitlines()})
    token = secrets.token_hex(16)

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
        mac(f"cd {base} && curl -fsS {url} | tar xf - && printf %s {shlex.quote(sums)} | shasum -a 256 -c --quiet "
            f"&& rm -f ARCHIVED.json")
    finally:
        server.shutdown()
    print(f"restored {names} to {MAC_ARMS}/{model}/{arm}")


if __name__ == "__main__":
    {"out": out, "back": back}[sys.argv[1]](sys.argv[2], sys.argv[3])
