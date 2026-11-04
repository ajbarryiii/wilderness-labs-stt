"""Contract tests (CPU, the tiny model): the export adapter training/keab_export.py (keab-prime plan "Export").

On the tiny Parakeet (decoder_fixtures, 22 binary modules) in each training form (eightbit.apply_keab_scope):
- the bundle's byte counts per part match its manifest, the file, and the sizes expected from the model's shapes;
- the INT8 codes and scales are preserved exactly (the frozen parts' load-time codes, pre_encode.out's from its
  latent weight);
- the scoring view's tensors equal the training forward weights bit for bit (W8; fp32(fp16(alpha)) under S2), and
  for pre_encode.out the forward weight W8 is exported, never the latent weight;
- FP16 scales widen exactly to the FP32 the runtime reads, and binary_export.load_export rebuilds those weights;
- with no scope in 8 bits, bundle -> view equals binary_export.export_model of the training model byte for byte
  (the real-model check against MB1's export.safetensors is in test_keab_export_gpu.py);
- refusals: a model whose state does not match its form, a damaged bundle, a view outside the artifact root.
"""
from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file

import relabel_paths
from decoder_fixtures import tiny_model
from training import eightbit as eb
from training import keab_export as ke

import binary_export  # noqa: E402  (training.eightbit puts finetune/parakeet-binary on sys.path)
import binary_quant  # noqa: E402

FLOAT_PARTS = ("prediction_network", "joint", "pre_encode_out")


def use_pinned_tokenizer_refs(model) -> None:
    """The tiny model's tokenizer entries as the pinned .nemo's ("nemo:<member>"), as in every real Parakeet config
    (the exporter copies the tokenizer files out of the pinned .nemo by these names)."""
    from omegaconf import open_dict
    with open_dict(model.cfg):
        for key in binary_export.TOKENIZER_FILES:
            model.cfg.tokenizer[key] = "nemo:" + Path(str(model.cfg.tokenizer[key])).name


def training_model(scope: str, seed: int = 11):
    model = tiny_model(seed=seed)
    use_pinned_tokenizer_refs(model)
    eb.binarize(model)
    eb.apply_keab_scope(model, scope)
    return model.eval()


def part_of(key: str) -> str | None:
    for part, prefix in eb.PART_PREFIX.items():
        if key.startswith(prefix):
            return part
    return None


def expected_part_bytes(model, scope: str) -> dict[str, int]:
    """The byte count of every part from the model's shapes alone (keab-prime plan "Export")."""
    in8 = set(eb.keys_of(eb.KEAB_SCOPES[scope]))
    names = binary_quant.quantized_module_names(model)
    covered = {f"{n}.weight" for n in names} | {f"{n}.bias" for n in names}
    out = dict.fromkeys(ke.PARTS, 0)
    for n in names:
        rows, cols = model.get_submodule(n).matrix().shape
        out["binary_codes"] += rows * math.ceil(cols / 8)
        out["binary_scales"] += rows * (2 if scope == "S2" else 4)
    for key, value in model.state_dict().items():
        if key in covered:
            continue
        part = part_of(key)
        if key in in8:
            rows = value.shape[0]
            out[part] += value.numel() + 4 * rows
        elif not value.is_floating_point():
            out["integer_buffers"] += value.numel() * value.element_size()
        elif key in binary_export.FP32_BUFFERS:
            out["fp32_feature_constants"] += 4 * value.numel()
        else:
            out[part or "other_fp16"] += 2 * value.numel()
    return out


class Tmp:
    def __init__(self):
        relabel_paths.require_mount()
        base = relabel_paths.ARTIFACTS / "tmp"
        base.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="keab-export-test-", dir=base))

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


class Bundle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Tmp()
        cls.models = {s: training_model(s) for s in eb.SCOPES}
        cls.bundles = {s: ke.write_bundle(cls.models[s], cls.tmp.root / s / "bundle", root=cls.tmp.root,
                                          extra={"test": s}) for s in eb.SCOPES}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.close()

    def test_byte_counts_per_part(self):
        for scope, manifest in self.bundles.items():
            with self.subTest(scope=scope):
                d = self.tmp.root / scope / "bundle"
                tensors = load_file(d / ke.FILE)
                listed = manifest["tensors"]
                self.assertEqual(set(listed), set(tensors))
                per_part = dict.fromkeys(ke.PARTS, 0)
                for name, t in tensors.items():
                    self.assertEqual(listed[name]["bytes"], t.numel() * t.element_size())
                    self.assertEqual(listed[name]["dtype"], str(t.dtype).removeprefix("torch."))
                    per_part[listed[name]["part"]] += t.numel() * t.element_size()
                b = manifest["bytes"]
                self.assertEqual(b["parts"], per_part)
                self.assertEqual(per_part, expected_part_bytes(self.models[scope], scope))
                self.assertEqual(b["tensor_bytes"], sum(per_part.values()))
                self.assertEqual(b["file_bytes"], (d / ke.FILE).stat().st_size)
                self.assertEqual(b["header_bytes"], b["file_bytes"] - b["tensor_bytes"])
                self.assertEqual(b["tokenizer_bytes"],
                                 sum((d / f).stat().st_size for f in manifest["files"] if f != ke.FILE))

    def test_int8_codes_and_scales_preserved_exactly(self):
        for scope in ("S1", "S2"):
            model, d = self.models[scope], self.tmp.root / scope / "bundle"
            tensors = load_file(d / ke.FILE)
            record = eb.frozen_int8(model)
            keys = eb.keys_of(eb.KEAB_SCOPES[scope])
            self.assertEqual(sorted(k.removesuffix(ke.INT8) for k in tensors if k.endswith(ke.INT8)), sorted(keys))
            for key in keys:
                q, s = record[key] if key in record else model.encoder.pre_encode.out.int8()
                self.assertTrue(torch.equal(tensors[key + ke.INT8], q.cpu()), key)
                self.assertTrue(torch.equal(tensors[key + ke.INT8_SCALE], s.cpu()), key)
                self.assertEqual(tensors[key + ke.INT8].dtype, torch.int8)
                self.assertEqual(tensors[key + ke.INT8_SCALE].dtype, torch.float32)
                self.assertNotIn(key, tensors, "an 8-bit weight is stored only as codes and scales")
            self.assertEqual(self.bundles[scope]["scope"], scope)

    def test_binary_scales_fp16_under_s2_only_and_other_tensors_as_binary_export(self):
        for scope in eb.SCOPES:
            tensors = load_file(self.tmp.root / scope / "bundle" / ke.FILE)
            model = self.models[scope]
            for n in binary_quant.quantized_module_names(model):
                codes, scale = model.get_submodule(n).quantized_weight()
                self.assertTrue(torch.equal(tensors[f"{n}.codes"], binary_export.pack_codes(codes)))
                want = scale.to(torch.float16) if scope == "S2" else scale
                self.assertTrue(torch.equal(tensors[f"{n}.scale"], want))
            for key, value in model.state_dict().items():
                if key in tensors:
                    if key in binary_export.FP32_BUFFERS:
                        self.assertTrue(torch.equal(tensors[key], value.float()))
                    elif value.is_floating_point():
                        self.assertTrue(torch.equal(tensors[key], value.to(torch.float16)))
                    else:
                        self.assertTrue(torch.equal(tensors[key], value))

    def test_damaged_bundle_is_refused(self):
        src = self.tmp.root / "S1" / "bundle"
        dst = self.tmp.root / "damaged" / "bundle"
        shutil.copytree(src, dst)
        data = bytearray((dst / ke.FILE).read_bytes())
        data[-1] ^= 1
        (dst / ke.FILE).write_bytes(bytes(data))
        with self.assertRaises(ValueError):
            ke.read_bundle(dst)

    def test_state_that_does_not_match_the_form_is_refused(self):
        model = training_model("S1")
        with torch.no_grad():
            model.get_parameter("decoder.prediction.embed.weight")[0, 0] += 1.0
        with self.assertRaises(ValueError):
            ke.write_bundle(model, self.tmp.root / "bad" / "bundle", root=self.tmp.root)
        self.assertFalse((self.tmp.root / "bad" / "bundle" / ke.FILE).exists())
        with self.assertRaises(ValueError):
            ke.write_bundle(training_model("none"), Path(tempfile.gettempdir()) / "outside-root", root=self.tmp.root)


class View(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Tmp()
        cls.models = {s: training_model(s) for s in eb.SCOPES}
        cls.views = {}
        for s in eb.SCOPES:
            ke.write_bundle(cls.models[s], cls.tmp.root / s / "bundle", root=cls.tmp.root)
            cls.views[s] = ke.derive_view(cls.tmp.root / s / "bundle", cls.tmp.root / s / "view", root=cls.tmp.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.close()

    def view_tensors(self, scope):
        return load_file(self.tmp.root / scope / "view" / binary_export.FILE)

    def test_view_equals_training_forward_weights_bit_for_bit(self):
        for scope in eb.SCOPES:
            with self.subTest(scope=scope):
                model, view = self.models[scope], self.view_tensors(scope)
                for n in binary_quant.quantized_module_names(model):
                    module = model.get_submodule(n)
                    forward = (module.forward_weight() if hasattr(module, "forward_weight")
                               else binary_quant._effective_weight(module.matrix(), 1.0, module.ramp)).detach()
                    rows, cols = forward.shape
                    rebuilt = binary_quant.dequantize(binary_export.unpack_codes(view[f"{n}.codes"], cols),
                                                      view[f"{n}.scale"])
                    self.assertEqual(view[f"{n}.scale"].dtype, torch.float32)
                    self.assertTrue(torch.equal(rebuilt, forward), n)
                for key in eb.keys_of(eb.KEAB_SCOPES[scope]):
                    if key == "encoder.pre_encode.out.weight":
                        forward = model.encoder.pre_encode.out.forward_weight().detach()
                        latent = model.encoder.pre_encode.out.weight.detach()
                        self.assertFalse(torch.equal(view[key], latent.to(torch.float16)), "never the latent")
                    else:
                        forward = model.get_parameter(key).detach()
                    self.assertEqual(view[key].dtype, torch.float16)
                    self.assertTrue(torch.equal(view[key].float(), forward), key)

    def test_fp16_scales_widen_exactly_to_the_runtime_fp32(self):
        bundle = load_file(self.tmp.root / "S2" / "bundle" / ke.FILE)
        view = self.view_tensors("S2")
        rebuilt = binary_export.load_export(self.tmp.root / "S2" / "view", "cpu")
        rebuilt_state = rebuilt.state_dict()
        for n in binary_quant.quantized_module_names(self.models["S2"]):
            self.assertEqual(bundle[f"{n}.scale"].dtype, torch.float16)
            self.assertTrue(torch.equal(view[f"{n}.scale"], bundle[f"{n}.scale"].to(torch.float32)))
            w = rebuilt_state[f"{n}.weight"]
            w = w.squeeze(2) if w.dim() == 3 else w
            codes = binary_export.unpack_codes(view[f"{n}.codes"], w.shape[1])
            self.assertTrue(torch.equal(w, codes.float() * bundle[f"{n}.scale"].float()[:, None]))

    def test_no_scope_view_equals_the_direct_export_byte_for_byte(self):
        direct = self.tmp.root / "direct" / "view"
        ke.write_view(self.models["none"], direct, root=self.tmp.root)
        a = (self.tmp.root / "none" / "view" / binary_export.FILE).read_bytes()
        b = (direct / binary_export.FILE).read_bytes()
        self.assertEqual(a, b)

    def test_view_is_a_parakeet_binary_export_with_the_bundle_config(self):
        for scope in eb.SCOPES:
            manifest = json.loads((self.tmp.root / scope / "view" / binary_export.MANIFEST).read_text())
            bundle = json.loads((self.tmp.root / scope / "bundle" / ke.MANIFEST).read_text())
            self.assertEqual(manifest["format"], binary_export.FORMAT)
            self.assertEqual(manifest["config"], bundle["config"])
            self.assertEqual(manifest["extra"]["keab"]["bundle_sha256"], bundle["sha256"])
            self.assertEqual(manifest["extra"]["keab"]["scope"], scope)
            self.assertEqual(self.views[scope]["sha256"], manifest["sha256"])

    def test_view_outside_the_root_is_refused(self):
        with self.assertRaises(ValueError):
            ke.derive_view(self.tmp.root / "S1" / "bundle", Path(tempfile.gettempdir()) / "keab-outside",
                           root=self.tmp.root / "S2")


if __name__ == "__main__":
    unittest.main()
