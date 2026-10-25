"""Contract tests: the checkpoint directory vLLM loads T* from (labeling/engines.py, VllmTeacher).

T*'s pinned snapshot holds, next to model.safetensors, an auxiliary CTC head (out_llm.safetensors, top-level
"weight" and "bias") that the stage 1 transformers decoder never loads. vLLM loads every *.safetensors file of
a local checkpoint directory that has no index, and fails on those names. So vLLM is given a view of the
snapshot without the auxiliary file:

- the view links every other entry of the snapshot (links resolve to the snapshot's files) and leaves out
  exactly engines.VLLM_EXCLUDED_FILES;
- it is reused when already correct, and a view that differs from the snapshot is refused (never silently
  used or rewritten);
- VllmTeacher.load passes the view as the model and tokenizer directory of vllm.LLM and records it and the
  excluded files in the engine identity, with the pinned hf_id and revision;
- the snapshot is resolved from the local Hugging Face cache only (C2 runs with HF_HUB_OFFLINE=1, and the cache
  holds only the files the stage 1 decoder fetched): the pinned revision's snapshot folder is used even when
  other repository files were never cached; a missing file that vLLM needs (engines.VLLM_REQUIRED_FILES) or an
  uncached revision is refused with an error naming it; nothing is downloaded.

No GPU and no vLLM: vllm and the tokenizer are stand-ins.
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from labeling import engines


def snapshot(root: Path) -> Path:
    snap = root / "snap"
    snap.mkdir()
    for name in ("config.json", "model.safetensors", "out_llm.safetensors", "tokenizer.json",
                 "preprocessor_config.json", "chat_template.jinja"):
        (snap / name).write_text(name)
    return snap


class ViewTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.snap = snapshot(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_view_links_everything_but_the_auxiliary_head(self):
        self.assertEqual(engines.VLLM_EXCLUDED_FILES, ("out_llm.safetensors",))
        view = engines.vllm_model_dir(self.snap, self.root / "views" / "t-rev")
        names = sorted(p.name for p in view.iterdir())
        self.assertEqual(names, sorted(p.name for p in self.snap.iterdir() if p.name != "out_llm.safetensors"))
        for p in view.iterdir():
            self.assertTrue(p.is_symlink())
            self.assertEqual(p.resolve(), (self.snap / p.name).resolve())
            self.assertEqual(p.read_text(), p.name)
        self.assertEqual(sorted(p.name for p in view.glob("*.safetensors")), ["model.safetensors"])

    def test_reused_when_correct_refused_when_different(self):
        target = self.root / "views" / "t-rev"
        first = engines.vllm_model_dir(self.snap, target)
        mtime = os.lstat(first / "config.json").st_mtime_ns
        self.assertEqual(engines.vllm_model_dir(self.snap, target), first)
        self.assertEqual(os.lstat(first / "config.json").st_mtime_ns, mtime)
        (first / "extra.safetensors").symlink_to(self.snap / "out_llm.safetensors")
        with self.assertRaises(RuntimeError):
            engines.vllm_model_dir(self.snap, target)


REV = "f" * 40


def hf_cache(root: Path, files=None, rev: str = REV) -> Path:
    """A Hugging Face hub cache holding snapshot `rev` of org/model with `files` (default: the required files and
    the auxiliary head, not the rest of the repository)."""
    cache = root / "hub"
    snap = cache / "models--org--model" / "snapshots" / rev
    snap.mkdir(parents=True)
    for name in (engines.VLLM_REQUIRED_FILES + ("out_llm.safetensors",) if files is None else files):
        (snap / name).write_text(name)
    return cache


class OfflineSnapshotTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_partial_cache_resolves_to_the_pinned_snapshot_without_downloading(self):
        self.assertIn("model.safetensors", engines.VLLM_REQUIRED_FILES)
        self.assertIn("config.json", engines.VLLM_REQUIRED_FILES)
        cache = hf_cache(self.root)
        import huggingface_hub
        with mock.patch.object(huggingface_hub, "snapshot_download", side_effect=AssertionError("download")), \
                mock.patch.object(huggingface_hub, "hf_hub_download", side_effect=AssertionError("download")):
            snap = engines.cached_snapshot("org/model", REV, cache_dir=cache)
        self.assertEqual(snap, cache / "models--org--model" / "snapshots" / REV)
        view = engines.vllm_model_dir(snap, self.root / "views" / "v")
        self.assertEqual(sorted(p.name for p in view.iterdir()), sorted(engines.VLLM_REQUIRED_FILES))

    def test_missing_required_file_is_named(self):
        files = tuple(f for f in engines.VLLM_REQUIRED_FILES if f != "model.safetensors")
        cache = hf_cache(self.root, files)
        with self.assertRaisesRegex(RuntimeError, r"model\.safetensors"):
            engines.cached_snapshot("org/model", REV, cache_dir=cache)

    def test_uncached_revision_is_named(self):
        cache = hf_cache(self.root, rev="a" * 40)
        with self.assertRaisesRegex(RuntimeError, REV):
            engines.cached_snapshot("org/model", REV, cache_dir=cache)
        with self.assertRaisesRegex(RuntimeError, "org/other"):
            engines.cached_snapshot("org/other", REV, cache_dir=cache)


class LoadTest(unittest.TestCase):
    def test_llm_gets_the_view_and_the_identity_records_it(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            snap = snapshot(root)
            calls = {}

            class LLM:
                def __init__(self, **kw):
                    calls.update(kw)

            fake_vllm = types.SimpleNamespace(LLM=LLM, SamplingParams=lambda **kw: types.SimpleNamespace(**kw),
                                              __version__="0.32.0")
            fake_envs = types.ModuleType("vllm.envs")     # reads the environment when accessed, as vllm.envs does
            fake_envs.environment_variables = {engines.VLLM_MODE_ENV: None}
            fake_envs.__getattr__ = lambda name: os.environ.get(name) == "1"
            tok = types.SimpleNamespace(apply_chat_template=lambda *a, **k: "PROMPT")
            fake_tf = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: tok))
            def no_download(*a, **k):
                raise AssertionError("load must not download (C2 runs offline)")

            import huggingface_hub
            hub = types.SimpleNamespace(snapshot_download=no_download, hf_hub_download=no_download,
                                        try_to_load_from_cache=huggingface_hub.try_to_load_from_cache)
            view_root = root / "views"
            with mock.patch.dict(sys.modules, {"vllm": fake_vllm, "vllm.envs": fake_envs, "transformers": fake_tf,
                                               "huggingface_hub": hub}), \
                    mock.patch.object(engines, "vllm_view_root", return_value=view_root), \
                    mock.patch.object(engines, "cached_snapshot",
                                      side_effect=lambda repo, rev, cache_dir=None: snap) as resolve, \
                    mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("AGENT_JOBS_VRAM_FRACTION", None)
                os.environ.pop(engines.VLLM_MODE_ENV, None)
                t = engines.VllmTeacher(64, 0.4, batch_invariant=True)
                t.load()
                info = t.info()
                self.assertEqual(os.environ[engines.VLLM_MODE_ENV], "1")
            import teachers
            from labeling import layout
            spec = teachers.REGISTRY[layout.TEACHER]
            resolve.assert_called_once()
            self.assertEqual(resolve.call_args.args[:2], (spec.hf_id, spec.revision))
            model_dir = Path(calls["model"])
            self.assertTrue(model_dir.is_relative_to(view_root))
            self.assertEqual(Path(calls["tokenizer"]), model_dir)
            self.assertFalse((model_dir / "out_llm.safetensors").exists())
            self.assertTrue((model_dir / "model.safetensors").exists())
            self.assertEqual(info["model_dir"], str(model_dir))
            self.assertEqual(info["excluded_files"], list(engines.VLLM_EXCLUDED_FILES))
            self.assertEqual((info["hf_id"], info["revision"]), (spec.hf_id, spec.revision))
            self.assertIs(info["batch_invariant"], True)
            self.assertEqual(calls["logprobs_mode"], engines.VLLM_LOGPROBS_MODE)


if __name__ == "__main__":
    unittest.main()
