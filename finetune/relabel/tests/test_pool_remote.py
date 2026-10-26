"""Column projection contracts (labeling plan Part B "Text": text columns only, no audio).

read_projected returns exactly the projected columns of every row group, and none of the bytes it requests
belong to an audio column chunk. Over HTTP, the pinned size and SHA-256 are verified before any range read.

Run from finetune/relabel:  ./python -m unittest discover -s tests -p 'test_pool_*' -v
"""
from __future__ import annotations

import hashlib
import http.server
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import pyarrow.parquet as pq

from pool import remote
from test_pool_fixtures import source_parquet

YODAS_COLS = ("duration", "text", "original_audio_id", "original_audio_offset")


def audio_ranges(path: Path) -> list[tuple[int, int]]:
    meta = pq.ParquetFile(path).metadata
    out = []
    for g in range(meta.num_row_groups):
        rg = meta.row_group(g)
        for i in range(rg.num_columns):
            cc = rg.column(i)
            if cc.path_in_schema.startswith("audio"):
                start = min(cc.data_page_offset, cc.dictionary_page_offset or cc.data_page_offset)
                out.append((start, start + cc.total_compressed_size))
    return out


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


class ProjectionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "f.parquet"
        recs = [(f"rec{i}", [f"text {i} {k}" for k in range(5)]) for i in range(7)]
        self.rows = source_parquet(self.path, "yodas", recs, rows_per_group=4)

    def tearDown(self):
        self.tmp.cleanup()

    def test_rows_equal_full_read_and_no_audio_bytes(self):
        src = remote.LocalRangeSource(self.path)
        tables, stats = remote.read_projected(src, YODAS_COLS)
        self.assertEqual([g for g, _ in tables], list(range(pq.ParquetFile(self.path).metadata.num_row_groups)))
        got = [r for _, t in tables for r in t.to_pylist()]
        want = pq.read_table(self.path, columns=list(YODAS_COLS)).to_pylist()
        self.assertEqual(got, want)
        self.assertEqual(len(got), len(self.rows))
        audio = audio_ranges(self.path)
        self.assertTrue(audio)
        for req in src.requests:
            for a in audio:
                self.assertFalse(overlaps(req, a), f"requested {req} overlaps audio chunk {a}")
        self.assertLess(stats["bytes_fetched"], 0.2 * self.path.stat().st_size)

    def test_row_groups_match_rows(self):
        tables, _ = remote.read_projected(remote.LocalRangeSource(self.path), YODAS_COLS)
        pf = pq.ParquetFile(self.path)
        for g, t in tables:
            self.assertEqual(t.to_pylist(), pf.read_row_group(g, columns=list(YODAS_COLS)).to_pylist())

    def test_sparse_file_refuses_unfetched_reads(self):
        f = remote.SparseFile(100)
        f.add(10, b"x" * 20)
        f.seek(12)
        self.assertEqual(f.read(5), b"xxxxx")
        f.seek(25)
        with self.assertRaises(OSError):
            f.read(10)


class _Handler(http.server.BaseHTTPRequestHandler):
    blob: bytes = b""
    etag: str = ""
    gets: list = []
    fail_gets = 0    # the next N GETs answer 503
    fail_heads = 0   # the next N HEADs after a failed GET answer 503
    armed = False

    def log_message(self, *args):
        pass

    def do_HEAD(self):
        if type(self).armed and type(self).fail_heads > 0:
            type(self).fail_heads -= 1
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(302)
        self.send_header("Location", "/content")
        self.send_header("X-Linked-Size", str(len(self.blob)))
        self.send_header("X-Linked-ETag", f'"{self.etag}"')
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if type(self).fail_gets > 0:
            type(self).fail_gets -= 1
            type(self).armed = True
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        start, end = self.headers["Range"].split("=")[1].split("-")
        data = self.blob[int(start):int(end) + 1]
        type(self).gets.append((int(start), int(end) + 1))
        self.send_response(206)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class HubSourceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name) / "f.parquet"
        source_parquet(path, "peoples_speech", [("itemA_SLASH_a_DOT_mp3", ["one two", "three"])], rows_per_group=1)
        _Handler.blob = path.read_bytes()
        _Handler.etag = hashlib.sha256(_Handler.blob).hexdigest()
        _Handler.gets = []
        _Handler.fail_gets = _Handler.fail_heads = 0
        _Handler.armed = False
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{self.server.server_port}/resolve"
        self.patch = mock.patch("huggingface_hub.hf_hub_url", lambda *a, **k: url)
        self.patch.start()
        self.path = path

    def tearDown(self):
        self.patch.stop()
        self.server.shutdown()
        self.tmp.cleanup()

    def test_pinned_content_reads(self):
        src = remote.HubRangeSource("r", "rev", "f.parquet", len(_Handler.blob), _Handler.etag, threads=2,
                                    attempts=1)
        tables, _ = remote.read_projected(src, ("id", "duration_ms", "text"))
        self.assertEqual([r["text"] for _, t in tables for r in t.to_pylist()], ["one two", "three"])
        audio = audio_ranges(self.path)
        for req in _Handler.gets:
            for a in audio:
                self.assertFalse(overlaps(req, a))

    def test_recovers_after_transient_get_and_resolve_failures(self):
        src = remote.HubRangeSource("r", "rev", "f.parquet", len(_Handler.blob), _Handler.etag, threads=2,
                                    attempts=6, base_wait=0.01, max_wait=0.02, timeout=5)
        _Handler.fail_gets, _Handler.fail_heads = 2, 1   # two failed GETs, then the re-resolve HEAD fails once
        done = []
        t = threading.Thread(target=lambda: done.append(remote.read_projected(src, ("id", "duration_ms", "text"))),
                             daemon=True)
        t.start()
        t.join(timeout=30)
        self.assertFalse(t.is_alive(), "fetch did not recover (deadlock?)")
        tables, _ = done[0]
        self.assertEqual([r["text"] for _, tb in tables for r in tb.to_pylist()], ["one two", "three"])
        self.assertGreaterEqual(src.retries, 3)

    def test_hash_mismatch_refuses_before_reading(self):
        with self.assertRaises(remote.RemoteError):
            remote.HubRangeSource("r", "rev", "f.parquet", len(_Handler.blob), "0" * 64, attempts=1)
        self.assertEqual(_Handler.gets, [])

    def test_size_mismatch_refuses(self):
        with self.assertRaises(remote.RemoteError):
            remote.HubRangeSource("r", "rev", "f.parquet", len(_Handler.blob) + 1, _Handler.etag, attempts=1)


if __name__ == "__main__":
    unittest.main()
