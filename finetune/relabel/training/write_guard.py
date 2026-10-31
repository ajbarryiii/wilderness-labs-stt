"""Refuse writes outside the allowed roots (keab-prime plan "Outputs": every artifact under $RELABEL_ROOT).

A Python audit hook (sys.addaudithook) sees every Python-level file operation of the process and of its forked
workers: open for writing (builtin open, os.open), os.mkdir, os.rename / os.replace, os.remove / unlink, os.rmdir,
os.truncate, os.chmod, os.symlink, os.link, os.utime, shutil.rmtree / copyfile / copytree / move. While the guard is
active, an operation on a path outside the allowed roots and files raises PermissionError before it happens and is
recorded (refused()). Reads anywhere, mkdir of a directory that already exists (nothing is written), /dev and /proc
pass. Operations relative to a directory descriptor (dir_fd, e.g. inside shutil.rmtree) were checked through their
parent call. Writes from native code (safetensors' writer, CUDA and NVRTC caches) are not seen by the hook; their
targets are this stage's own output directories and the caches relabel_boot.py points at $RELABEL_ROOT.

The hook cannot be removed, so it is registered once and switched with install/deactivate (or `active`).
"""
from __future__ import annotations

import contextlib
import os
import sys

_STATE: dict = {"roots": None, "files": [], "refused": [], "registered": False}
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
_PASS = ("/dev/", "/proc/")
# event -> (indices of path arguments, index of its dir_fd argument or None)
_EVENTS = {
    "os.mkdir": ((0,), 2), "os.rename": ((0, 1), 2), "os.remove": ((0,), 1), "os.rmdir": ((0,), 1),
    "os.truncate": ((0,), None), "os.chmod": ((0,), 2), "os.symlink": ((1,), 2), "os.link": ((1,), 3),
    "os.utime": ((0,), 3), "shutil.rmtree": ((0,), 1), "shutil.copyfile": ((1,), None),
    "shutil.copytree": ((1,), None), "shutil.move": ((0, 1), None),
}


def _resolve(path) -> str | None:
    if isinstance(path, int) or path is None:
        return None
    return os.path.realpath(os.fsdecode(path))


def _allowed(path: str) -> bool:
    if path.startswith(_PASS) or path in _STATE["files"]:
        return True
    return any(path == root or path.startswith(root.rstrip("/") + "/") for root in _STATE["roots"])


def _refuse(event: str, path: str) -> None:
    _STATE["refused"].append({"event": event, "path": path})
    print(f"write_guard: refused {event} {path} (outside $RELABEL_ROOT)", file=sys.stderr, flush=True)
    raise PermissionError(f"write outside the allowed roots refused: {event} {path}")


def _hook(event: str, args) -> None:
    if _STATE["roots"] is None:
        return
    if event == "open":
        path, mode, flags = (tuple(args) + (None, None, None))[:3]
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
            (isinstance(flags, int) and bool(flags & _WRITE_FLAGS) and not isinstance(mode, str))
        paths = [path] if writing else []
    elif event in _EVENTS:
        indices, fd_index = _EVENTS[event]
        if fd_index is not None and fd_index < len(args) and args[fd_index] not in (None, -1):
            return
        paths = [args[i] for i in indices if i < len(args)]
        if event == "os.mkdir" and paths and _resolve(paths[0]) and os.path.isdir(_resolve(paths[0])):
            return
    else:
        return
    for p in paths:
        resolved = _resolve(p)
        if resolved is not None and not _allowed(resolved):
            _refuse(event, resolved)


def install(roots, files=()) -> None:
    """Activate the guard with these roots (directories) and single files allowed."""
    if not _STATE["registered"]:
        sys.addaudithook(_hook)
        _STATE["registered"] = True
    _STATE.update(roots=[os.path.realpath(os.fspath(r)) for r in roots],
                  files=[os.path.realpath(os.fspath(f)) for f in files], refused=[])


def deactivate() -> None:
    _STATE.update(roots=None, files=[])


def state() -> tuple[list[str] | None, list[str]]:
    return _STATE["roots"], list(_STATE["files"])


def refused() -> list[dict]:
    return list(_STATE["refused"])


@contextlib.contextmanager
def active(roots, files=()):
    install(roots, files)
    try:
        yield
    finally:
        deactivate()
