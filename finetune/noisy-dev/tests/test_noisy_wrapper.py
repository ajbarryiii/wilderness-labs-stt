"""Interpreter isolation (review round 3, finding 6): ./python uses only its fixed, trusted
PYTHONPATH. An inherited PYTHONPATH (a download directory, a relative or an empty entry) never
reaches sys.path, and neither does the working directory (-P)."""
from __future__ import annotations

import json
import os
import subprocess
import unittest
from pathlib import Path

import noisy_fixtures as fx

HERE = Path(__file__).resolve().parents[1]


class WrapperTest(unittest.TestCase):
    def test_inherited_pythonpath_is_ignored(self):
        evil = fx.fresh_dir("wrapper-evil")
        (evil / "noisy_paths.py").write_text("raise SystemExit('imported from an untrusted directory')\n")
        env = {**os.environ, "PYTHONPATH": f"{evil}:relative/dir::"}
        out = subprocess.run([str(HERE / "python"), "-c", "import sys, json, noisy_paths; print(json.dumps(sys.path))"],
                             cwd=evil, env=env, capture_output=True, text=True, check=True)
        path = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertNotIn(str(evil), path)
        self.assertNotIn("relative/dir", path)
        self.assertNotIn("", path)
        self.assertTrue(all(os.path.isabs(p) for p in path), path)
        self.assertEqual(path[0], str(HERE))


if __name__ == "__main__":
    unittest.main()
