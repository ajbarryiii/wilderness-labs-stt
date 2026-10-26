"""Host and GPU resource sampling for the probes (a background thread; Linux /proc and nvidia-smi).

Per sample: the summed RSS of this process and all its descendants (conservative: shared pages count once per
process), their summed CPU seconds, the cgroup's memory.current (when the job runs in its own cgroup), and,
with gpu=True, the summed GPU memory of the process tree from `nvidia-smi --query-compute-apps`.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path


def descendants(pid: int) -> list[int]:
    children: dict[int, list[int]] = {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            ppid = int((d / "stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(d.name))
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        todo += children.get(p, [])
    return out


def rss_bytes(pids: list[int]) -> int:
    total = 0
    for pid in pids:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    break
        except OSError:
            pass
    return total


def cpu_seconds(pids: list[int]) -> float:
    tick = os.sysconf("SC_CLK_TCK")
    total = 0.0
    for pid in pids:
        try:
            f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            total += (int(f[11]) + int(f[12])) / tick
        except (OSError, IndexError, ValueError):
            pass
    return total


def cgroup_file(name: str) -> Path | None:
    try:
        rel = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
    except (OSError, IndexError):
        return None
    path = Path("/sys/fs/cgroup") / rel.lstrip("/") / name
    return path if path.exists() else None


def gpu_bytes(pids: set[int]) -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    total = 0
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) in pids and parts[1].isdigit():
            total += int(parts[1]) * 2**20
    return total


class Monitor:
    def __init__(self, interval: float = 1.0, gpu: bool = False):
        self.interval, self.gpu = interval, gpu
        self.peak_rss = self.peak_cgroup = self.peak_gpu = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="monitor", daemon=True)
        self._cg = cgroup_file("memory.current")
        self.start_cpu = cpu_seconds(descendants(os.getpid()))
        self.t0 = time.monotonic()

    def _sample(self) -> None:
        pids = descendants(os.getpid())
        self.peak_rss = max(self.peak_rss, rss_bytes(pids))
        if self._cg is not None:
            try:
                self.peak_cgroup = max(self.peak_cgroup, int(self._cg.read_text()))
            except (OSError, ValueError):
                pass
        if self.gpu:
            self.peak_gpu = max(self.peak_gpu, gpu_bytes(set(pids)))
        self.samples += 1

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def __enter__(self):
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=30)
        self._sample()

    def result(self) -> dict:
        wall = time.monotonic() - self.t0
        cpu = cpu_seconds(descendants(os.getpid())) - self.start_cpu
        return {"wall_s": wall, "peak_tree_rss_gb": self.peak_rss / 2**30,
                "peak_cgroup_memory_gb": self.peak_cgroup / 2**30 if self._cg is not None else None,
                "peak_gpu_gb": self.peak_gpu / 2**30 if self.gpu else None,
                "cpu_cores_mean": cpu / wall if wall > 0 else None, "samples": self.samples,
                "note": "tree RSS sums every process (shared pages counted per process); cpu covers live processes"}
