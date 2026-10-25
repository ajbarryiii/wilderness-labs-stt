"""Contract tests: the per-clip Granite encoder for vLLM (labeling/vllm_granite_patch.py) and how the engine uses it
(labeling/engines.py VllmTeacher, checks.vllm_engine, c2.contract, labeling/vllm_env.sh).

- A clip's embedding does not depend on its batch-mates: with each clip's own length, a batch of clips of different
  lengths gives every clip what it gets alone, and alone the patched encoder computes what the unpatched one does
  (module stand-ins: transformers' Granite speech encoder and projector, the modules vLLM's mirror, tiny and random,
  float32 on CPU). The unpatched encoder in the same batch does not (the fixture can see the leak). Frames after a
  clip's end are zero in the patched encoder's output; the projector tokens per clip are the feature extractor's count,
  and a count that disagrees with the clip's length is refused.
- The patch identity (name, version, sha256 of the patch file) is in every production vLLM spec, so in c2.contract:
  a run, check or probe without it, or with another one, never shares a contract with one under it, and a run
  directory made under another spec is refused.
- Activation: register() does nothing (and imports no vLLM) unless the environment asks for this exact patch, refuses
  another sha256 or another vLLM version, and when active registers the patched model for T*'s architecture and the
  worker status method. VllmTeacher.load asks for the patch before vLLM starts, removes the request when its spec has
  none, and refuses to run unless the worker reports the expected identity and model class from another process; the
  identity and the report are in info().
- labeling/vllm_env.sh plugin installs, idempotently, a distribution whose vllm.general_plugins entry point is
  register().
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from labeling import c2, checks, engines
from labeling import vllm_granite_patch as P

HERE = Path(__file__).resolve().parents[1]


# --- the per-clip encoder --------------------------------------------------------------------------------------------

def tiny_modules(seed: int = 0):
    import torch
    from transformers.models.blip_2.configuration_blip_2 import Blip2QFormerConfig
    from transformers.models.granite_speech.configuration_granite_speech import (GraniteSpeechConfig,
                                                                                  GraniteSpeechEncoderConfig)
    from transformers.models.granite_speech.modeling_granite_speech import (GraniteSpeechCTCEncoder,
                                                                             GraniteSpeechEncoderProjector)
    torch.manual_seed(seed)
    enc_cfg = GraniteSpeechEncoderConfig(input_dim=6, num_layers=4, hidden_dim=16, feedforward_mult=2, num_heads=2,
                                         dim_head=8, output_dim=5, context_size=5, max_pos_emb=8, conv_kernel_size=5,
                                         conv_expansion_factor=2, dropout=0.0)
    q_cfg = Blip2QFormerConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=24,
                               encoder_hidden_size=16, cross_attention_frequency=1, hidden_dropout_prob=0.0,
                               attention_probs_dropout_prob=0.0)
    cfg = GraniteSpeechConfig(encoder_config=enc_cfg, projector_config=q_cfg, window_size=3, downsample_rate=1,
                              text_config={"model_type": "granite", "hidden_size": 12, "num_hidden_layers": 1,
                                           "num_attention_heads": 2, "num_key_value_heads": 2,
                                           "intermediate_size": 16, "vocab_size": 32})
    enc, proj = GraniteSpeechCTCEncoder(enc_cfg), GraniteSpeechEncoderProjector(cfg)
    with torch.no_grad():
        for m in enc.modules():
            if isinstance(m, torch.nn.BatchNorm1d):     # non-trivial running statistics
                m.running_mean.normal_()
                m.running_var.uniform_(0.5, 2.0)
            if isinstance(m, torch.nn.LayerNorm):
                m.weight.normal_(1.0, 0.2)
                m.bias.normal_(0.0, 0.2)
        for p in list(enc.parameters()) + list(proj.parameters()):
            if p.dim() > 1:
                p.normal_(0.0, 0.4)
    return enc.eval(), proj.eval()


def padded(feats):
    import torch
    n = max(f.shape[0] for f in feats)
    return torch.stack([torch.nn.functional.pad(f, (0, 0, 0, n - f.shape[0])) for f in feats])


class PerClipEncoder(unittest.TestCase):
    LENGTHS = [23, 7, 15, 11, 3]          # context 5, window 3: partial and whole blocks, partial and whole windows

    def setUp(self):
        import torch
        self.torch = torch
        self.enc, self.proj = tiny_modules()
        g = torch.Generator().manual_seed(1)
        self.feats = [torch.randn(n, 6, generator=g) for n in self.LENGTHS]

    def tokens(self, n):
        return P.projector_tokens(n, self.proj.window_size, self.proj.downsample_rate)

    def test_alone_is_the_unpatched_encoder(self):
        with self.torch.no_grad():
            for f in self.feats:
                ref = self.enc(f[None]).last_hidden_state
                mine = P.encode(self.enc, f[None], [f.shape[0]])
                self.torch.testing.assert_close(mine, ref, atol=1e-5, rtol=1e-5)
                ref_emb = self.proj(ref)[0, :self.tokens(f.shape[0])]
                (mine_emb,) = P.embed(self.enc, self.proj, f[None], [f.shape[0]], [self.tokens(f.shape[0])])
                self.torch.testing.assert_close(mine_emb, ref_emb, atol=1e-5, rtol=1e-5)

    def test_batched_clip_is_the_clip_alone(self):
        x = padded(self.feats)
        sizes = [self.tokens(n) for n in self.LENGTHS]
        with self.torch.no_grad():
            h = P.encode(self.enc, x, self.LENGTHS)
            batched = P.embed(self.enc, self.proj, x, self.LENGTHS, sizes)
            for b, f in enumerate(self.feats):
                n = f.shape[0]
                alone = P.encode(self.enc, f[None], [n])
                self.torch.testing.assert_close(h[b, :n], alone[0], atol=1e-5, rtol=1e-5)
                self.assertTrue(bool((h[b, n:] == 0).all()))           # nothing after the clip's end
                (alone_emb,) = P.embed(self.enc, self.proj, f[None], [n], [sizes[b]])
                self.assertEqual(tuple(batched[b].shape), tuple(alone_emb.shape))
                self.torch.testing.assert_close(batched[b], alone_emb, atol=1e-5, rtol=1e-5)

    def test_the_unpatched_encoder_leaks_in_this_fixture(self):
        x = padded(self.feats)
        with self.torch.no_grad():
            h = self.enc(x).last_hidden_state
            b = self.LENGTHS.index(7)
            alone = self.enc(self.feats[b][None]).last_hidden_state
            self.assertGreater(float((h[b, :7] - alone[0]).abs().max()), 1e-3)

    def test_token_counts_must_match_the_lengths(self):
        x = padded(self.feats[:2])
        sizes = [self.tokens(n) for n in self.LENGTHS[:2]]
        with self.assertRaisesRegex(RuntimeError, "token counts"):
            P.embed(self.enc, self.proj, x, self.LENGTHS[:2], [sizes[0], sizes[1] + 1])

    def test_unexpected_convolution_modules_are_refused(self):
        for layer in self.enc.layers:
            P.check_conv(layer.conv)
        conv = self.enc.layers[0].conv
        conv.depth_conv.conv.dilation = (2,)
        with self.assertRaises(RuntimeError):
            P.check_conv(conv)

    def test_feature_lengths(self):
        torch = self.torch
        self.assertEqual(P.feature_lengths([torch.zeros(4, 6), torch.zeros(9, 6)]), [4, 9])
        self.assertEqual(P.feature_lengths(torch.zeros(3, 8, 6)), [8, 8, 8])
        self.assertEqual(P.feature_lengths(torch.zeros(2, 1, 8, 6)), [8, 8])
        with self.assertRaises(ValueError):
            P.feature_lengths([torch.zeros(1, 4, 6)])


# --- identity and contract -------------------------------------------------------------------------------------------

class Identity(unittest.TestCase):
    def test_identity_is_the_patch_file(self):
        ident = P.identity()
        self.assertEqual(ident["sha256"], hashlib.sha256(Path(P.__file__).read_bytes()).hexdigest())
        self.assertEqual((ident["name"], ident["version"]), (P.NAME, P.VERSION))

    def test_every_production_vllm_spec_names_the_patch(self):
        for s in checks.VLLM_SETTINGS:
            self.assertEqual(checks.vllm_engine(s, 32.0)["encoder_patch"], P.identity())
        self.assertNotIn("encoder_patch", checks.tf_engine(checks.TRANSFORMERS_SETTINGS[0]))
        eng = c2.make_engine(checks.vllm_engine(checks.VLLM_SETTINGS[0], 32.0), {})
        self.assertEqual(eng.encoder_patch, P.identity())

    def test_the_patch_is_part_of_the_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = Path(tmp) / "pool"
            pool.mkdir()
            spec = checks.vllm_engine(checks.VLLM_SETTINGS[2], 32.0)
            runs = {"patched": spec, "unpatched": {k: v for k, v in spec.items() if k != "encoder_patch"},
                    "other": spec | {"encoder_patch": spec["encoder_patch"] | {"sha256": "0" * 64}}}
            k = {}
            for name, eng in runs.items():
                run = c2.init_run(Path(tmp) / name, pool_dir=pool, engine=eng, group_shards=4, b0_batch_seconds=1200.0,
                                  declared_vram_gb=29.0)
                k[name] = c2.contract(run)
            self.assertEqual(k["patched"]["engine"]["encoder_patch"], P.identity())
            self.assertNotEqual(k["patched"], k["unpatched"])
            self.assertNotEqual(k["patched"], k["other"])
            cons = {"passed": True, "full_population": True, "contract": k["unpatched"]}
            self.assertFalse(checks.qualifies(k["patched"], cons, {"passed": True, "contract": k["patched"]})[0])
            self.assertFalse(checks.qualifies(k["patched"], cons | {"contract": k["patched"]},
                                              {"passed": True, "contract": k["other"]})[0])
            with self.assertRaises(RuntimeError):     # a run directory made without the patch is not resumed with it
                c2.init_run(Path(tmp) / "unpatched", pool_dir=pool, engine=spec, group_shards=4,
                            b0_batch_seconds=1200.0, declared_vram_gb=29.0)


# --- activation ------------------------------------------------------------------------------------------------------

def fake_vllm(version="0.32.0"):
    """Stand-ins for the vLLM modules register() touches."""
    calls = {"registered": {}, "processors": []}

    class Base:
        def __init__(self, *a, **k):
            self.encoder = Encoder()

    class Encoder:
        layers = []

    class Registry:
        @staticmethod
        def register_model(arch, cls):
            calls["registered"][arch] = cls

    class MMRegistry:
        @staticmethod
        def register_processor(*a, **k):
            calls["processors"].append((a, k))
            return lambda cls: cls

    class WorkerBase:
        pass

    gs = types.SimpleNamespace(GraniteSpeechForConditionalGeneration=Base, GraniteSpeechCTCEncoder=Encoder,
                               GraniteSpeechMultiModalProcessor=object, GraniteSpeechProcessingInfo=object,
                               GraniteSpeechDummyInputsBuilder=object)
    mods = {"vllm": types.SimpleNamespace(__version__=version, ModelRegistry=Registry),
            "vllm.model_executor": types.SimpleNamespace(), "vllm.model_executor.models": types.SimpleNamespace(
                granite_speech=gs), "vllm.model_executor.models.granite_speech": gs,
            "vllm.multimodal": types.SimpleNamespace(MULTIMODAL_REGISTRY=MMRegistry),
            "vllm.v1": types.SimpleNamespace(), "vllm.v1.worker": types.SimpleNamespace(),
            "vllm.v1.worker.worker_base": types.SimpleNamespace(WorkerBase=WorkerBase)}
    return mods, calls, WorkerBase


class Register(unittest.TestCase):
    def setUp(self):
        self.state = dict(P._state)
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()

    def tearDown(self):
        self.env.stop()
        P._state.clear()
        P._state.update(self.state)
        P.__dict__.pop("PerClipGraniteSpeech", None)

    def test_not_requested_does_nothing_and_imports_no_vllm(self):
        os.environ.pop(P.ENV, None)
        with mock.patch.dict(sys.modules, {"vllm": None}):        # importing vllm would raise
            self.assertIsNone(P.register())
        self.assertFalse(P._state["active"])

    def test_another_patch_or_vllm_is_refused(self):
        os.environ[P.ENV] = "0" * 64
        with mock.patch.dict(sys.modules, {"vllm": None}), self.assertRaisesRegex(RuntimeError, "another patch"):
            P.register()
        os.environ[P.ENV] = P.source_sha256()
        mods, _, _ = fake_vllm("0.33.0")
        with mock.patch.dict(sys.modules, mods), self.assertRaisesRegex(RuntimeError, "0.33.0"):
            P.register()
        self.assertFalse(P._state["active"])

    def test_requested_registers_the_model_and_the_status(self):
        P.apply_env(P.identity())
        self.assertEqual(os.environ[P.ENV], P.source_sha256())
        mods, calls, worker_base = fake_vllm()
        with mock.patch.dict(sys.modules, mods):
            P.register()
        cls = calls["registered"][P.ARCH]
        self.assertEqual(f"{cls.__module__}.{cls.__qualname__}", f"{P.__name__}.PerClipGraniteSpeech")
        self.assertEqual(len(calls["processors"]), 1)
        model = cls(vllm_config=None)
        worker = types.SimpleNamespace(get_model=lambda: model)
        report = getattr(worker_base, P.STATUS_METHOD)(worker)
        self.assertEqual(report["pid"], os.getpid())
        self.assertEqual(P.check_status(P.identity(), [report], parent_pid=-1), [report])
        P.apply_env(None)
        self.assertNotIn(P.ENV, os.environ)


GOOD = {"pid": -1, "active": True, "identity": None, "model_class": f"{P.__name__}.PerClipGraniteSpeech"}


class Status(unittest.TestCase):
    def test_refusals(self):
        ident = P.identity()
        good = GOOD | {"identity": ident}
        self.assertEqual(P.check_status(ident, [good], parent_pid=os.getpid()), [good])
        bad = {"no report": None, "empty": [], "inactive": [good | {"active": False}],
               "other identity": [good | {"identity": ident | {"sha256": "0" * 64}}],
               "unpatched model": [good | {"model_class": "vllm.model_executor.models.granite_speech."
                                                          "GraniteSpeechForConditionalGeneration"}],
               "this process": [good | {"pid": os.getpid()}]}
        for why, reports in bad.items():
            with self.subTest(why), self.assertRaises(RuntimeError):
                P.check_status(ident, reports, parent_pid=os.getpid())


class Load(unittest.TestCase):
    """VllmTeacher.load with stand-ins for vLLM and the checkpoint (as test_labeling_vllm_weights)."""

    def load(self, patch, rpc):
        seen = {}

        class LLM:
            def __init__(self, **kw):
                seen["env_at_start"] = os.environ.get(P.ENV)

            def collective_rpc(self, method):
                seen["rpc"] = method
                return rpc()

        fake = types.SimpleNamespace(LLM=LLM, SamplingParams=lambda **kw: kw, __version__="0.32.0")
        envs = types.ModuleType("vllm.envs")
        envs.environment_variables = {engines.VLLM_MODE_ENV: None}
        envs.__getattr__ = lambda name: os.environ.get(name) == "1"
        tok = types.SimpleNamespace(apply_chat_template=lambda *a, **k: "PROMPT")
        tf = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: tok))
        with tempfile.TemporaryDirectory() as d:
            snap = Path(d) / "snap"
            snap.mkdir()
            for n in ("config.json", "model.safetensors"):
                (snap / n).write_text(n)
            with mock.patch.dict(sys.modules, {"vllm": fake, "vllm.envs": envs, "transformers": tf}), \
                    mock.patch.object(engines, "vllm_view_root", return_value=Path(d) / "views"), \
                    mock.patch.object(engines, "cached_snapshot", return_value=snap), \
                    mock.patch.dict(os.environ, {P.ENV: "stale"}, clear=False):
                os.environ.pop("AGENT_JOBS_VRAM_FRACTION", None)
                t = engines.VllmTeacher(64, 0.4, batch_invariant=True, encoder_patch=patch)
                try:
                    t.load()
                except RuntimeError as e:
                    return None, seen, e
                return t.info(), seen, None

    def test_expected_and_reported_by_the_engine_core(self):
        ident = P.identity()
        report = GOOD | {"identity": ident, "pid": os.getpid() + 1}
        info, seen, err = self.load(ident, lambda: [report])
        self.assertIsNone(err)
        self.assertEqual(seen["env_at_start"], ident["sha256"])
        self.assertEqual(seen["rpc"], P.STATUS_METHOD)
        self.assertEqual(info["encoder_patch"], ident)
        self.assertEqual(info["encoder_patch_status"], [report])

    def test_refused_unless_active_in_another_process(self):
        ident = P.identity()

        def missing():
            raise NotImplementedError(f"Method {P.STATUS_METHOD!r} is not implemented.")

        cases = {"plugin not loaded": missing, "this process": lambda: [GOOD | {"identity": ident,
                                                                                "pid": os.getpid()}],
                 "inactive": lambda: [GOOD | {"identity": ident, "active": False}],
                 "other patch": lambda: [GOOD | {"identity": ident | {"version": 0}}]}
        for why, rpc in cases.items():
            with self.subTest(why):
                _, _, err = self.load(ident, rpc)
                self.assertIsInstance(err, RuntimeError)

    def test_without_a_patch_the_request_is_removed(self):
        info, seen, err = self.load(None, lambda: self.fail("no status asked without a patch"))
        self.assertIsNone(err)
        self.assertIsNone(seen["env_at_start"])
        self.assertIsNone(info["encoder_patch"])
        self.assertIsNone(info["encoder_patch_status"])


# --- the plugin distribution ----------------------------------------------------------------------------------------

class PluginInstall(unittest.TestCase):
    def test_installs_the_entry_point_idempotently(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            site = root / "deps" / "vllm-0.32.0" / "lib" / "python3.13" / "site-packages"
            site.mkdir(parents=True)
            env = dict(os.environ, RELABEL_ROOT=str(root), DATA_MOUNT="/")

            def run():
                return subprocess.run([str(HERE / "labeling" / "vllm_env.sh"), "plugin"], env=env,
                                      capture_output=True, text=True)

            r = run()
            self.assertEqual(r.returncode, 0, r.stderr)
            dists = list(importlib.metadata.distributions(path=[str(site)]))
            self.assertEqual(len(dists), 1)
            eps = [e for e in dists[0].entry_points if e.group == "vllm.general_plugins"]
            self.assertEqual([(e.name, e.value) for e in eps], [(P.PLUGIN, P.ENTRY_POINT)])
            module, func = P.ENTRY_POINT.split(":")
            self.assertEqual(module, P.__name__)
            self.assertTrue(callable(getattr(P, func)))
            files = sorted(p for p in site.rglob("*") if p.is_file())
            stamp = [p.stat().st_mtime_ns for p in files]
            self.assertEqual(run().returncode, 0)
            self.assertEqual([p.stat().st_mtime_ns for p in files], stamp)
            self.assertEqual(sorted(p for p in site.rglob("*") if p.is_file()), files)


if __name__ == "__main__":
    unittest.main()
