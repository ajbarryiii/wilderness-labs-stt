"""Contract tests (CPU): nothing stage 4 step 0 writes lands outside $RELABEL_ROOT (training/write_guard.py) and
gpu1bit's cubin cache misses go to $RELABEL_ROOT/keab-prime (training/deployed.relabel_cubin_cache).

- with the guard active, Python-level writes outside the allowed roots are refused before they happen (open for
  writing, mkdir, rename/replace, remove, rmtree, tempfile), and recorded; reads anywhere, writes inside the roots,
  the allowed lock file, and mkdir of an existing directory pass;
- keab_step0's entry point runs every subcommand under the guard with $RELABEL_ROOT and the parakeet GPU lock file;
- the cubin seam reads gpu1bit's cache on a hit and compiles a miss into $RELABEL_ROOT/keab-prime/caches/cubin.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import relabel_paths
from training import write_guard as wg


class Guard(unittest.TestCase):
    def setUp(self):
        relabel_paths.require_mount()
        base = relabel_paths.ARTIFACTS / "tmp"
        base.mkdir(parents=True, exist_ok=True)
        self.inside = Path(tempfile.mkdtemp(prefix="keab-guard-in-", dir=base))
        self.outside = Path(tempfile.mkdtemp(prefix="keab-guard-out-", dir="/tmp"))
        (self.outside / "existing.txt").write_text("x")
        self.lock = self.outside / "gpu.lock"
        self.lock.write_text("")

    def tearDown(self):
        wg.deactivate()
        shutil.rmtree(self.inside, ignore_errors=True)
        shutil.rmtree(self.outside, ignore_errors=True)

    def test_writes_outside_are_refused_and_recorded(self):
        with wg.active([self.inside], files=[self.lock]):
            for attempt in (lambda: open(self.outside / "new.txt", "w"),
                            lambda: open(self.outside / "existing.txt", "a"),
                            lambda: os.open(self.outside / "flags.txt", os.O_WRONLY | os.O_CREAT),
                            lambda: (self.outside / "dir").mkdir(),
                            lambda: os.replace(self.inside / "nope", self.outside / "moved"),
                            lambda: os.remove(self.outside / "existing.txt"),
                            lambda: shutil.rmtree(self.outside),
                            lambda: tempfile.mkstemp(dir=self.outside)):
                with self.assertRaises(PermissionError):
                    attempt()
            refused = wg.refused()
        self.assertEqual(len(refused), 8)
        self.assertTrue((self.outside / "existing.txt").read_text() == "x")
        self.assertFalse((self.outside / "new.txt").exists())
        self.assertFalse((self.outside / "dir").exists())

    def test_reads_inside_writes_lock_file_and_existing_dirs_pass(self):
        with wg.active([self.inside], files=[self.lock]):
            self.assertEqual((self.outside / "existing.txt").read_text(), "x")
            with open(self.inside / "a.json", "w") as f:
                f.write("{}")
            os.replace(self.inside / "a.json", self.inside / "b.json")
            (self.inside / "sub" / "deeper").mkdir(parents=True)
            with open(self.lock, "a+") as f:
                f.write("1 label\n")
            self.outside.mkdir(parents=True, exist_ok=True)     # exists: nothing is written
            fd, name = tempfile.mkstemp(dir=self.inside)
            os.close(fd)
            shutil.rmtree(self.inside / "sub")
            self.assertEqual(wg.refused(), [])

    def test_inactive_guard_refuses_nothing(self):
        with wg.active([self.inside]):
            pass
        with open(self.outside / "after.txt", "w") as f:
            f.write("ok")

    def test_driver_runs_under_the_guard(self):
        import keab_step0 as k0
        seen = {}

        def fake_status(args):
            seen["roots"], seen["files"] = wg.state()
            return 0
        with mock.patch.dict(k0.ACTIONS, {"status": fake_status}):
            self.assertEqual(k0.main(["status"]), 0)
        self.assertEqual(seen["roots"], [str(relabel_paths.ARTIFACTS.resolve())])
        self.assertEqual(seen["files"], [str(Path(relabel_paths.GPU_LOCK).resolve())])
        self.assertEqual(wg.state(), (None, []))


class CubinSeam(unittest.TestCase):
    def setUp(self):
        relabel_paths.require_mount()
        base = relabel_paths.ARTIFACTS / "tmp"
        base.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(prefix="keab-cubin-", dir=base))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_hit_reads_shared_and_miss_compiles_into_relabel(self):
        from training import deployed
        from gpu1bit.int8_mma import cuda
        shared, own = self.tmp / "shared", self.tmp / "own"
        compiled = []

        def fake_original(source, name, arch="120a", options=cuda.OPTIONS):
            compiled.append(name)
            path = cuda.ARTIFACTS / "caches" / "cubin" / f"{deployed.cubin_key(source, name, arch, options, 'v')}.cubin"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"compiled " + name.encode())
            return path.read_bytes()

        key = deployed.cubin_key("src-hit", "k_hit", "120a", cuda.OPTIONS, "v")
        (shared / "caches" / "cubin").mkdir(parents=True)
        (shared / "caches" / "cubin" / f"{key}.cubin").write_bytes(b"shared")
        with mock.patch.object(cuda, "ARTIFACTS", shared), mock.patch.object(cuda, "setup", lambda: "v"), \
                mock.patch.object(cuda, "compile_cubin", fake_original):
            with deployed.relabel_cubin_cache(own):
                self.assertEqual(cuda.compile_cubin("src-hit", "k_hit"), b"shared")
                self.assertEqual(cuda.compile_cubin("src-miss", "k_miss"), b"compiled k_miss")
                self.assertEqual(cuda.compile_cubin("src-miss", "k_miss"), b"compiled k_miss")
                self.assertIs(cuda.ARTIFACTS, shared)
            self.assertIs(cuda.compile_cubin, fake_original)
        self.assertEqual(compiled, ["k_miss"], "a hit never compiles; a miss compiles once")
        miss = deployed.cubin_key("src-miss", "k_miss", "120a", cuda.OPTIONS, "v")
        self.assertTrue((own / "caches" / "cubin" / f"{miss}.cubin").is_file())
        self.assertFalse((shared / "caches" / "cubin" / f"{miss}.cubin").exists())


if __name__ == "__main__":
    unittest.main()
