"""F5 contracts of B-hybrid-x2 (plans/x2-default.md), CPU only, each check in a fresh interpreter:

1. Importing B's x3 modules, and the x2 modules without using them, loads no gpu1bit.actprec module
   and none of the part-count modules.
2. A tampered copy of any loaded part-count file (kernels_parts, layer_parts, parts_run) makes every x2
   entry point (exact_model_x2, HybridProjectionX2, dot_slices_x2) raise ValueError naming the file and
   both hashes, before any CUDA context exists. The copy is made in a scratch directory on /mnt/hd and
   H's loader is pointed at it with gpu1bit.actprec.parts.installed(wip_dir); B's real files under
   B/wip/parts are only read, and their hashes are verified unchanged afterwards.

    CUDA_VISIBLE_DEVICES="" gpu1bit/shared/python -m unittest gpu1bit.int8_mma.test_x2_imports
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from gpu1bit.shared import env

X3_MODULES = ("gpu1bit.int8_mma.layer", "gpu1bit.int8_mma.kernels", "gpu1bit.int8_mma.configs", "gpu1bit.int8_mma.hybrid",
              "gpu1bit.int8_mma.hybrid_exact", "gpu1bit.int8_mma.row_exact", "gpu1bit.int8_mma.kernels_exact",
              "gpu1bit.int8_mma.deferred", "gpu1bit.int8_mma.session_patch", "gpu1bit.int8_mma.b01_run",
              "gpu1bit.int8_mma.layer_b01")
X2_MODULES = ("gpu1bit.int8_mma.hybrid_x2", "gpu1bit.int8_mma.row_exact_x2", "gpu1bit.int8_mma.kernels_exact_x2",
              "gpu1bit.int8_mma.deferred_x2", "gpu1bit.int8_mma.session_patch_x2")
LOADED = ("kernels_parts", "layer_parts", "parts_run")
REPO = Path(__file__).resolve().parents[2]  # finetune/parakeet-binary


def child(code: str, *args: str) -> dict:
    env_ = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([sys.executable, "-c", code, *args], cwd=REPO, env=env_, capture_output=True, text=True, timeout=300)
    lines = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
    if r.returncode != 0 or not lines:
        raise AssertionError(f"child failed (rc {r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
    return json.loads(lines[-1][len("RESULT "):])


IMPORT_CHILD = r'''
import importlib, json, sys
mods = json.loads(sys.argv[1])
for m in mods:
    importlib.import_module(m)
bad = sorted(m for m in sys.modules if m.startswith("gpu1bit.actprec") or m.split(".")[-1] in ("kernels_parts", "layer_parts", "parts_run"))
import torch
print("RESULT " + json.dumps({"imported": mods, "bad": bad, "cuda_initialized": torch.cuda.is_initialized()}))
'''

TAMPER_CHILD = r'''
import json, sys, traceback
from pathlib import Path
wip, entry = Path(sys.argv[1]), sys.argv[2]
from gpu1bit.actprec import parts as AP
import torch
out = {"entry": entry}
with AP.installed(wip):
    from gpu1bit.int8_mma import hybrid_x2
    try:
        if entry == "exact_model_x2":
            with hybrid_x2.exact_model_x2():
                pass
        elif entry == "HybridProjectionX2":
            hybrid_x2.HybridProjectionX2(None, None)
        elif entry == "dot_slices_x2":
            hybrid_x2.dot_slices_x2(None, None, None, 0, 0, None, None)
        out["raised"] = None
    except Exception as exc:
        out["raised"] = type(exc).__name__
        out["message"] = str(exc)
out["cuda_initialized"] = torch.cuda.is_initialized()
out["actprec_wip_dir_during"] = str(wip)
print("RESULT " + json.dumps(out))
'''


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ImportIsolation(unittest.TestCase):
    def test_x3_imports_load_no_actprec(self):
        got = child(IMPORT_CHILD, json.dumps(list(X3_MODULES)))
        self.assertEqual(got["bad"], [], "x3 imports pulled in actprec / part-count modules")
        self.assertFalse(got["cuda_initialized"])

    def test_x2_imports_without_use_load_no_actprec(self):
        got = child(IMPORT_CHILD, json.dumps(list(X3_MODULES + X2_MODULES)))
        self.assertEqual(got["bad"], [], "importing the x2 modules pulled in actprec / part-count modules")
        self.assertFalse(got["cuda_initialized"])


class TamperedPartsRefused(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gpu1bit.actprec import parts as AP
        env.require_mount()
        cls.AP = AP
        cls.real = Path(AP.WIP_DIR)
        cls.before = {name: sha(cls.real / name) for name in AP.PINNED}
        scratch = env.ARTIFACTS_ROOT / "B" / "scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        cls.tmp = Path(tempfile.mkdtemp(prefix="x2-tamper-", dir=scratch))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_real_files_are_the_pinned_ones(self):
        self.assertEqual(self.before, dict(self.AP.PINNED))

    def test_tampered_file_makes_every_x2_entry_point_raise(self):
        for name in LOADED:
            d = self.tmp / name
            d.mkdir()
            for f in self.AP.PINNED:
                shutil.copyfile(self.real / f, d / f)
            target = d / f"{name}.py"
            data = target.read_bytes()
            target.write_bytes(data.replace(b'"""', b'""" ', 1))  # one byte more inside the first docstring
            tampered = sha(target)
            pinned = self.AP.PINNED[f"{name}.py"]
            self.assertNotEqual(tampered, pinned)
            for entry in ("exact_model_x2", "HybridProjectionX2", "dot_slices_x2"):
                with self.subTest(file=name, entry=entry):
                    got = child(TAMPER_CHILD, str(d), entry)
                    self.assertEqual(got["raised"], "ValueError", got)
                    self.assertIn(str(target), got["message"])
                    self.assertIn(tampered, got["message"])
                    self.assertIn(pinned, got["message"])
                    self.assertFalse(got["cuda_initialized"], "the refusal came after CUDA work")
        after = {f: sha(self.real / f) for f in self.AP.PINNED}
        self.assertEqual(after, self.before, "B's real part-count files changed")
        self.assertEqual(after, dict(self.AP.PINNED))


if __name__ == "__main__":
    unittest.main()
