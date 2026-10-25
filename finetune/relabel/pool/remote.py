"""Column-projected parquet reads by byte range (no audio is fetched).

`read_projected(source, columns)` reads the 8-byte trailer and then exactly the parquet footer, computes
the byte range of every projected column chunk of every row group from the footer metadata, fetches exactly
those ranges (adjacent ranges are merged only when the gap holds no chunk of a never-fetched column such as
`audio`), and lets pyarrow decode the row groups from a strict sparse in-memory view of the file: a read
outside the fetched ranges raises instead of fetching (tests check that no audio byte is requested).

Range sources:
- LocalRangeSource(path): a local file (tests, local copies); records every requested range.
- HubRangeSource(repo, revision, path, size, sha256): a dataset file at a pinned revision. One HEAD on the
  Hub resolves the content URL; the pinned size and SHA-256 (the listing's LFS sha256 = X-Linked-ETag) are
  verified before any read, so content is exactly the pinned revision's. Ranges are fetched by a small
  thread pool with retries (exponential backoff, re-resolve on failure).
"""
from __future__ import annotations

import bisect
import concurrent.futures as cf
import io
import threading
import time
from collections.abc import Sequence
from pathlib import Path

MERGE_GAP = 64 * 1024       # merge ranges whose gap is at most this (and free of never-fetched chunks)
NEVER_FETCH = ("audio",)    # top-level columns whose bytes are never requested


class RemoteError(RuntimeError):
    pass


class LocalRangeSource:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.size = self.path.stat().st_size
        self.requests: list[tuple[int, int]] = []

    def fetch(self, ranges: Sequence[tuple[int, int]]) -> list[bytes]:
        out = []
        with open(self.path, "rb") as handle:
            for start, end in ranges:
                self.requests.append((start, end))
                handle.seek(start)
                out.append(handle.read(end - start))
        return out

    def close(self) -> None:
        pass


class HubRangeSource:
    """Byte ranges of one dataset file at a pinned revision, integrity-checked against the listing."""

    def __init__(self, repo: str, revision: str, path: str, size: int | None, sha256: str | None,
                 threads: int = 8, attempts: int = 8, base_wait: float = 5.0, max_wait: float = 120.0,
                 timeout: float = 60.0):
        self.repo, self.revision, self.path = repo, revision, path
        self.expected_size, self.expected_sha256 = size, sha256
        self.threads, self.attempts = threads, attempts
        self.base_wait, self.max_wait, self.timeout = base_wait, max_wait, timeout
        self.requests: list[tuple[int, int]] = []
        self.retries = 0
        self._local = threading.local()
        self._lock = threading.RLock()   # re-entrant: _resolve() (under the lock) calls _wait()
        self._url: str | None = None
        self.size = 0
        self._resolve()

    def _session(self):
        s = getattr(self._local, "session", None)
        if s is None:
            import requests
            s = requests.Session()
            self._local.session = s
        return s

    def _resolve(self) -> None:
        from huggingface_hub import get_token, hf_hub_url
        url = hf_hub_url(self.repo, self.path, repo_type="dataset", revision=self.revision)
        token = get_token()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        for attempt in range(1, self.attempts + 1):
            try:
                r = self._session().head(url, headers=headers, allow_redirects=False, timeout=self.timeout)
                if r.status_code in (301, 302, 303, 307, 308):
                    from urllib.parse import urljoin
                    location = urljoin(url, r.headers["location"])
                    size = int(r.headers.get("x-linked-size") or -1)
                    etag = (r.headers.get("x-linked-etag") or "").strip('"')
                elif r.status_code == 200:
                    location, size = url, int(r.headers.get("content-length") or -1)
                    etag = (r.headers.get("x-linked-etag") or r.headers.get("etag") or "").strip('"')
                else:
                    raise RemoteError(f"HEAD {self.path}: HTTP {r.status_code}")
                break
            except Exception as exc:  # network errors and transient HTTP failures
                if attempt == self.attempts:
                    raise RemoteError(f"resolve {self.path}: giving up after {attempt} attempts: {exc!r}") from exc
                self._wait(attempt, f"resolve {self.path}: {exc!r}")
        if self.expected_size is not None and size != self.expected_size:
            raise RemoteError(f"{self.path}: size {size} != pinned {self.expected_size}")
        if self.expected_sha256 is not None and etag != self.expected_sha256:
            raise RemoteError(f"{self.path}: content hash {etag!r} != pinned sha256 {self.expected_sha256}")
        self._url, self.size = location, size

    def _wait(self, attempt: int, what: str) -> None:
        wait = min(self.max_wait, self.base_wait * 2 ** (attempt - 1))
        with self._lock:
            self.retries += 1
        print(f"remote: {what}; retry {attempt} in {wait:.0f}s", flush=True)
        time.sleep(wait)

    def _get(self, rng: tuple[int, int]) -> bytes:
        start, end = rng
        for attempt in range(1, self.attempts + 1):
            try:
                r = self._session().get(self._url, headers={"Range": f"bytes={start}-{end - 1}"},
                                        timeout=self.timeout)
                if r.status_code != 206 or len(r.content) != end - start:
                    raise RemoteError(f"GET {self.path} [{start},{end}): HTTP {r.status_code}, "
                                      f"{len(r.content)} bytes")
                return r.content
            except Exception as exc:
                if attempt == self.attempts:
                    raise RemoteError(f"{self.path} [{start},{end}): giving up after {attempt} attempts: "
                                      f"{exc!r}") from exc
                self._wait(attempt, f"{self.path} [{start},{end}): {exc!r}")
                if attempt >= 2:
                    with self._lock:
                        self._resolve()   # the content URL may have expired
        raise AssertionError("unreachable")

    def fetch(self, ranges: Sequence[tuple[int, int]]) -> list[bytes]:
        self.requests.extend(ranges)
        if len(ranges) <= 1 or self.threads <= 1:
            return [self._get(r) for r in ranges]
        with cf.ThreadPoolExecutor(min(self.threads, len(ranges))) as pool:
            return list(pool.map(self._get, ranges))

    def close(self) -> None:
        s = getattr(self._local, "session", None)
        if s is not None:
            s.close()


class SparseFile(io.RawIOBase):
    """A read-only file of `size` bytes of which only the added segments exist; other reads raise OSError."""

    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self._starts: list[int] = []
        self._segs: dict[int, bytes] = {}
        self._pos = 0

    def add(self, start: int, data: bytes) -> None:
        if start in self._segs and len(self._segs[start]) >= len(data):
            return
        if start not in self._segs:
            bisect.insort(self._starts, start)
        self._segs[start] = data

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self._pos, 2: self.size}[whence]
        self._pos = base + offset
        return self._pos

    def _covered(self, start: int, end: int) -> bytes | None:
        for i in range(bisect.bisect_right(self._starts, start) - 1, -1, -1):
            s = self._starts[i]
            seg = self._segs[s]
            if s + len(seg) >= end:
                return seg[start - s:end - s]
        return None

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self._pos
        start, end = self._pos, min(self.size, self._pos + n)
        if start >= end:
            return b""
        data = self._covered(start, end)
        if data is None:
            raise OSError(f"sparse read [{start},{end}) outside the fetched ranges")
        self._pos = end
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def _chunk_range(cc) -> tuple[int, int]:
    start = cc.data_page_offset
    if cc.has_dictionary_page and cc.dictionary_page_offset:
        start = min(start, cc.dictionary_page_offset)
    return start, start + cc.total_compressed_size


def plan_ranges(meta, columns: Sequence[str], never: Sequence[str] = NEVER_FETCH,
                merge_gap: int = MERGE_GAP) -> list[tuple[int, int]]:
    """Merged byte ranges covering the projected columns' chunks of every row group."""
    wanted = set(columns)
    want, forbid = [], []
    for g in range(meta.num_row_groups):
        rg = meta.row_group(g)
        found = set()
        for i in range(rg.num_columns):
            cc = rg.column(i)
            top = cc.path_in_schema.split(".")[0]
            if top in wanted:
                want.append(_chunk_range(cc))
                found.add(top)
            elif top in never:
                forbid.append(_chunk_range(cc))
        if found != wanted:
            raise RemoteError(f"row group {g}: columns {sorted(wanted - found)} missing")
    want.sort()
    forbid.sort()
    merged: list[list[int]] = []
    for start, end in want:
        if merged:
            last = merged[-1]
            gap_ok = start - last[1] <= merge_gap and not any(fs < start and fe > last[1] for fs, fe in forbid)
            if start <= last[1] or gap_ok:
                last[1] = max(last[1], end)
                continue
        merged.append([start, end])
    return [(s, e) for s, e in merged]


def read_projected(source, columns: Sequence[str]) -> tuple[list[tuple[int, object]], dict]:
    """[(row group index, pyarrow.Table of `columns`)] for every row group, plus read statistics."""
    import pyarrow.parquet as pq
    size = source.size
    if size < 12:
        raise RemoteError("not a parquet file (too small)")
    end = source.fetch([(size - 8, size)])[0]
    if end[4:] != b"PAR1":
        raise RemoteError("not a parquet file (no trailing magic)")
    footer_len = int.from_bytes(end[:4], "little")
    if footer_len <= 0 or footer_len + 12 > size:
        raise RemoteError(f"bad parquet footer length {footer_len}")
    footer = source.fetch([(size - 8 - footer_len, size - 8)])[0]
    # Parse the footer on its own, so pyarrow never makes its speculative 64 KB tail read (which would
    # include the end of the last row group's audio chunk).
    meta = pq.read_metadata(io.BytesIO(b"PAR1" + footer + end))
    ranges = plan_ranges(meta, columns)
    sparse = SparseFile(size)   # strict: any read outside the planned ranges raises
    for (start, _), data in zip(ranges, source.fetch(ranges)):
        sparse.add(start, data)
    pf = pq.ParquetFile(sparse, metadata=meta, pre_buffer=False)
    tables = [(g, pf.read_row_group(g, columns=list(columns), use_threads=False))
              for g in range(meta.num_row_groups)]
    stats = {"file_bytes": size, "row_groups": meta.num_row_groups, "rows": meta.num_rows,
             "ranges": len(ranges), "bytes_fetched": 8 + footer_len + sum(e - s for s, e in ranges)}
    return tables, stats
