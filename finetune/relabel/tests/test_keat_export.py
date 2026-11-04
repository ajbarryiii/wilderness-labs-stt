"""Contract tests (CPU, the tiny model): the stage 5 export adapter training/keat_export.py (keat-prime plan "Export").

On the tiny Parakeet (decoder_fixtures, 22 ternary modules) in each training form (keat_eightbit.apply_keat_scope):
- the bundle (keat-8bit-v1): ternary codes packed as export.pack_codes, row scales FP32 or FP16 under S2, INT8
  codes with FP32 per-channel scales, every other tensor as export.export_model stores it; its byte counts per part
  match the manifest, the file and the sizes expected from the model's shapes (M1's real table is in
  test_keat_export_gpu.py);
- the scoring view (parakeet-ternary-v1), derived from the bundle alone and written by export.export_model through
  the paths.ARTIFACTS seam: it preserves the ternary codes and scales (FP16 widened exactly) and the INT8 W8 tensors
  exactly, so it equals the training forward weights bit for bit; every other floating tensor equals its FP16
  conversion; the FP32 feature constants and integer buffers are preserved exactly; its ternary modules answer
  quantized_weight() with the stored codes and scales (recomputing them from the dequantized weights would give
  another scale);
- with no scope in 8 bits, bundle -> view equals export.export_model of the training model byte for byte;
- the seam is restored after the export, also when it fails;
- refusals: a model whose state does not match its form, a damaged bundle, a bundle or view outside the root.
"""
from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import load_file

import export as ternary_export  # finetune/parakeet-ternary/export.py
import paths  # finetune/parakeet-ternary/paths.py
import quant
import relabel_paths
from test_keab_export import use_pinned_tokenizer_refs
from test_keat_eightbit import ternary_model
from training import eightbit as eb
from training import keat_eightbit as teb
from training import keat_export as tke


def training_model(scope: str, seed: int = 11):
    model = ternary_model(seed)
    use_pinned_tokenizer_refs(model)
    teb.apply_keat_scope(model, scope)
    return model.eval()


def part_of(key: str) -> str | None:
    for part, prefix in eb.PART_PREFIX.items():
        if key.startswith(prefix):
            return part
    return None


def expected_part_bytes(model, scope: str) -> dict[str, int]:
    """The byte count of every part from the model's shapes alone (keat-prime plan "Export")."""
    in8 = set(eb.keys_of(teb.KEAT_SCOPES[scope]))
    names = quant.quantized_module_names(model)
    covered = {f"{n}.weight" for n in names} | {f"{n}.bias" for n in names}
    out = dict.fromkeys(tke.PARTS, 0)
    for n in names:
        module = model.get_submodule(n)
        rows, cols = module.matrix().shape
        out["ternary_codes"] += rows * math.ceil(cols / 4)
        out["ternary_scales"] += rows * (2 if scope == "S2" else 4)
        if module.bias is not None:
            out["ternary_biases"] += 4 * module.bias.numel()
    for key, value in model.state_dict().items():
        if key in covered:
            continue
        part = part_of(key)
        if key in in8:
            out[part] += value.numel() + 4 * value.shape[0]
        elif not value.is_floating_point():
            out["integer_buffers"] += value.numel() * value.element_size()
        elif key in ternary_export.FP32_BUFFERS:
            out["fp32_feature_constants"] += 4 * value.numel()
        else:
            out[part or "other_fp16"] += 2 * value.numel()
    return out


class Tmp:
    def __init__(self):
        relabel_paths.require_mount()
        base = relabel_paths.ARTIFACTS / "tmp"
        base.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="keat-export-test-", dir=base))

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


class Bundle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Tmp()
        cls.models = {s: training_model(s) for s in eb.SCOPES}
        cls.bundles = {s: tke.write_bundle(cls.models[s], cls.tmp.root / s / "bundle", root=cls.tmp.root,
                                           extra={"test": s}) for s in eb.SCOPES}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.close()

    def test_byte_counts_per_part(self):
        for scope, manifest in self.bundles.items():
            with self.subTest(scope=scope):
                d = self.tmp.root / scope / "bundle"
                tensors = load_file(d / tke.FILE)
                listed = manifest["tensors"]
                self.assertEqual(set(listed), set(tensors))
                per_part = dict.fromkeys(tke.PARTS, 0)
                for name, t in tensors.items():
                    self.assertEqual(listed[name]["bytes"], t.numel() * t.element_size())
                    self.assertEqual(listed[name]["dtype"], str(t.dtype).removeprefix("torch."))
                    per_part[listed[name]["part"]] += t.numel() * t.element_size()
                b = manifest["bytes"]
                self.assertEqual(b["parts"], per_part)
                self.assertEqual(per_part, expected_part_bytes(self.models[scope], scope))
                self.assertEqual(b["tensor_bytes"], sum(per_part.values()))
                self.assertEqual(b["file_bytes"], (d / tke.FILE).stat().st_size)
                self.assertEqual(b["header_bytes"], b["file_bytes"] - b["tensor_bytes"])
                self.assertEqual(b["tokenizer_bytes"],
                                 sum((d / f).stat().st_size for f in manifest["files"] if f != tke.FILE))
                self.assertEqual(manifest["format"], tke.FORMAT)
                self.assertEqual(manifest["scope"], scope)
                self.assertEqual(manifest["extra"], {"test": scope})

    def test_ternary_codes_and_scales_fp16_under_s2_only(self):
        for scope in eb.SCOPES:
            tensors = load_file(self.tmp.root / scope / "bundle" / tke.FILE)
            model = self.models[scope]
            for n in quant.quantized_module_names(model):
                module = model.get_submodule(n)
                codes32, s32 = quant.ternary_quantize(module.matrix())
                self.assertTrue(torch.equal(tensors[f"{n}.codes"], ternary_export.pack_codes(codes32)), n)
                self.assertEqual(tensors[f"{n}.codes"].dtype, torch.uint8)
                want = s32.to(torch.float16) if scope == "S2" else s32
                self.assertEqual(tensors[f"{n}.scale"].dtype, want.dtype)
                self.assertTrue(torch.equal(tensors[f"{n}.scale"], want), n)
                self.assertNotIn(f"{n}.weight", tensors)

    def test_int8_codes_and_scales_preserved_exactly(self):
        for scope in ("S1", "S2"):
            model, d = self.models[scope], self.tmp.root / scope / "bundle"
            tensors = load_file(d / tke.FILE)
            record = eb.frozen_int8(model)
            keys = eb.keys_of(teb.KEAT_SCOPES[scope])
            self.assertEqual(sorted(k.removesuffix(tke.INT8) for k in tensors if k.endswith(tke.INT8)), sorted(keys))
            for key in keys:
                q, s = record[key] if key in record else model.encoder.pre_encode.out.int8()
                self.assertTrue(torch.equal(tensors[key + tke.INT8], q.cpu()), key)
                self.assertTrue(torch.equal(tensors[key + tke.INT8_SCALE], s.cpu()), key)
                self.assertEqual(tensors[key + tke.INT8].dtype, torch.int8)
                self.assertEqual(tensors[key + tke.INT8_SCALE].dtype, torch.float32)
                self.assertNotIn(key, tensors, "an 8-bit weight is stored only as codes and scales")

    def test_other_tensors_as_the_ternary_exporter_stores_them(self):
        for scope in eb.SCOPES:
            tensors = load_file(self.tmp.root / scope / "bundle" / tke.FILE)
            model = self.models[scope]
            in8 = set(eb.keys_of(teb.KEAT_SCOPES[scope]))
            names = quant.quantized_module_names(model)
            covered = {f"{n}.weight" for n in names} | {f"{n}.bias" for n in names} | in8
            for key, value in model.state_dict().items():
                if key in covered:
                    continue
                if key in ternary_export.FP32_BUFFERS:
                    self.assertEqual(tensors[key].dtype, torch.float32)
                    self.assertTrue(torch.equal(tensors[key], value.float()), key)
                elif value.is_floating_point():
                    self.assertTrue(torch.equal(tensors[key], value.to(torch.float16)), key)
                else:
                    self.assertEqual(tensors[key].dtype, value.dtype)
                    self.assertTrue(torch.equal(tensors[key], value), key)

    def test_damaged_bundle_is_refused(self):
        src = self.tmp.root / "S1" / "bundle"
        dst = self.tmp.root / "damaged" / "bundle"
        shutil.copytree(src, dst)
        data = bytearray((dst / tke.FILE).read_bytes())
        data[-1] ^= 1
        (dst / tke.FILE).write_bytes(bytes(data))
        with self.assertRaises(ValueError):
            tke.read_bundle(dst)

    def test_stage_4_bundle_is_refused(self):
        from training import keab_export
        manifest = json.loads((self.tmp.root / "none" / "bundle" / tke.MANIFEST).read_text())
        self.assertEqual(manifest["format"], "keat-8bit-v1")
        self.assertNotEqual(tke.FORMAT, keab_export.FORMAT)
        other = self.tmp.root / "keab" / "bundle"
        shutil.copytree(self.tmp.root / "none" / "bundle", other)
        manifest["format"] = keab_export.FORMAT
        (other / tke.MANIFEST).write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            tke.read_bundle(other)

    def test_state_that_does_not_match_the_form_is_refused(self):
        model = training_model("S1")
        with torch.no_grad():
            model.get_parameter("decoder.prediction.embed.weight")[0, 0] += 1.0
        with self.assertRaises(ValueError):
            tke.write_bundle(model, self.tmp.root / "bad" / "bundle", root=self.tmp.root)
        self.assertFalse((self.tmp.root / "bad" / "bundle" / tke.FILE).exists())
        with self.assertRaises(ValueError):
            tke.write_bundle(training_model("none"), Path(tempfile.gettempdir()) / "keat-outside-root",
                             root=self.tmp.root)


class View(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Tmp()
        cls.models = {s: training_model(s) for s in eb.SCOPES}
        cls.views = {}
        for s in eb.SCOPES:
            tke.write_bundle(cls.models[s], cls.tmp.root / s / "bundle", root=cls.tmp.root)
            cls.views[s] = tke.derive_view(cls.tmp.root / s / "bundle", cls.tmp.root / s / "view", root=cls.tmp.root,
                                           extra={"variant": s})

    @classmethod
    def tearDownClass(cls):
        cls.tmp.close()

    def view_tensors(self, scope):
        return load_file(self.tmp.root / scope / "view" / ternary_export.FILE)

    def bundle_tensors(self, scope):
        return load_file(self.tmp.root / scope / "bundle" / tke.FILE)

    def test_view_equals_training_forward_weights_bit_for_bit(self):
        for scope in eb.SCOPES:
            with self.subTest(scope=scope):
                model, view = self.models[scope], self.view_tensors(scope)
                for n in quant.quantized_module_names(model):
                    module = model.get_submodule(n)
                    forward = (module.forward_weight() if isinstance(module, teb.HALF_SCALE_TYPES)
                               else quant._effective_weight(module.matrix(), 1.0)).detach()
                    rows, cols = forward.shape
                    rebuilt = quant.dequantize(ternary_export.unpack_codes(view[f"{n}.codes"], cols),
                                               view[f"{n}.scale"])
                    self.assertEqual(view[f"{n}.scale"].dtype, torch.float32)
                    self.assertTrue(torch.equal(rebuilt, forward), n)
                for key in eb.keys_of(teb.KEAT_SCOPES[scope]):
                    if key == "encoder.pre_encode.out.weight":
                        forward = model.encoder.pre_encode.out.forward_weight().detach()
                        latent = model.encoder.pre_encode.out.weight.detach()
                        self.assertFalse(torch.equal(view[key], latent.to(torch.float16)), "never the latent")
                    else:
                        forward = model.get_parameter(key).detach()
                    self.assertEqual(view[key].dtype, torch.float16)
                    self.assertTrue(torch.equal(view[key].float(), forward), key)

    def test_view_preserves_codes_scales_w8_and_every_other_tensor(self):
        for scope in eb.SCOPES:
            with self.subTest(scope=scope):
                view, bundle = self.view_tensors(scope), self.bundle_tensors(scope)
                manifest = json.loads((self.tmp.root / scope / "bundle" / tke.MANIFEST).read_text())
                expect = set()
                for key, t in bundle.items():
                    if key.endswith(tke.INT8_SCALE):
                        continue
                    if key.endswith(tke.INT8):
                        name = key.removesuffix(tke.INT8)
                        want = eb.w8(t, bundle[name + tke.INT8_SCALE])
                    elif key.endswith(".scale") and key.removesuffix(".scale") in manifest["ternary_layers"]:
                        name, want = key, t.to(torch.float32)
                    else:
                        name, want = key, t
                    expect.add(name)
                    self.assertEqual(view[name].dtype, want.dtype, name)
                    self.assertTrue(torch.equal(view[name], want), name)
                self.assertEqual(expect, set(view))
                model = self.models[scope]
                for key in ternary_export.FP32_BUFFERS:
                    self.assertEqual(view[key].dtype, torch.float32)
                    self.assertTrue(torch.equal(view[key], model.state_dict()[key].float()))
                ints = [k for k, v in model.state_dict().items() if not v.is_floating_point()]
                self.assertTrue(ints, "the tiny model has integer buffers (BatchNorm counters)")
                for key in ints:
                    self.assertTrue(torch.equal(view[key], model.state_dict()[key]), key)

    def test_stored_codes_and_scales_are_exported_not_recomputed(self):
        view = self.view_tensors("S2")
        rebuilt = ternary_export.load_export(self.tmp.root / "S2" / "view", "cpu").state_dict()
        model = self.models["S2"]
        differs = 0
        for n in quant.quantized_module_names(model):
            w = rebuilt[f"{n}.weight"]
            w = w.squeeze(2) if w.dim() == 3 else w
            _, s_again = quant.ternary_quantize(w)
            stored = model.get_submodule(n).quantized_weight()[1]
            self.assertTrue(torch.equal(view[f"{n}.scale"], stored), n)
            differs += int((s_again != stored).sum())
        self.assertGreater(differs, 0, "re-quantizing the dequantized weights changes scales (rows with zero codes)")
        hist = json.loads((self.tmp.root / "S2" / "view" / ternary_export.MANIFEST).read_text())["code_histogram"]
        self.assertEqual(hist, json.loads(json.dumps(quant.code_histogram(model))))

    def test_fp16_scales_widen_exactly_and_load_export_rebuilds_the_forward_weights(self):
        bundle = self.bundle_tensors("S2")
        view = self.view_tensors("S2")
        rebuilt = ternary_export.load_export(self.tmp.root / "S2" / "view", "cpu").state_dict()
        model = self.models["S2"]
        for n in quant.quantized_module_names(model):
            self.assertEqual(bundle[f"{n}.scale"].dtype, torch.float16)
            self.assertTrue(torch.equal(view[f"{n}.scale"], bundle[f"{n}.scale"].to(torch.float32)))
            w = rebuilt[f"{n}.weight"]
            w = w.squeeze(2) if w.dim() == 3 else w
            self.assertTrue(torch.equal(w, model.get_submodule(n).forward_weight().detach()), n)

    def test_no_scope_view_equals_the_direct_export_byte_for_byte(self):
        direct = self.tmp.root / "direct" / "view"
        with tke.artifact_root(self.tmp.root):
            ternary_export.export_model(self.models["none"], direct, {})
        a = (self.tmp.root / "none" / "view" / ternary_export.FILE).read_bytes()
        b = (direct / ternary_export.FILE).read_bytes()
        self.assertEqual(a, b)

    def test_view_is_a_parakeet_ternary_export_with_the_bundle_config(self):
        for scope in eb.SCOPES:
            manifest = json.loads((self.tmp.root / scope / "view" / ternary_export.MANIFEST).read_text())
            bundle = json.loads((self.tmp.root / scope / "bundle" / tke.MANIFEST).read_text())
            self.assertEqual(manifest["format"], ternary_export.FORMAT)
            self.assertEqual(manifest["config"], bundle["config"])
            self.assertEqual(manifest["extra"]["keat"]["bundle_sha256"], bundle["sha256"])
            self.assertEqual(manifest["extra"]["keat"]["scope"], scope)
            self.assertEqual(manifest["extra"]["variant"], scope)
            self.assertEqual(self.views[scope]["sha256"], manifest["sha256"])

    def test_view_outside_the_root_is_refused(self):
        with self.assertRaises(ValueError):
            tke.derive_view(self.tmp.root / "S1" / "bundle", Path(tempfile.gettempdir()) / "keat-outside",
                            root=self.tmp.root / "S2")


class Seam(unittest.TestCase):
    def test_artifact_root_is_restored(self):
        saved = paths.ARTIFACTS
        with tke.artifact_root(Path("/nonexistent-root")):
            self.assertEqual(paths.ARTIFACTS, Path("/nonexistent-root"))
        self.assertIs(paths.ARTIFACTS, saved)
        with self.assertRaises(RuntimeError):
            with tke.artifact_root(Path("/nonexistent-root")):
                raise RuntimeError("export failed")
        self.assertIs(paths.ARTIFACTS, saved)

    def test_a_view_that_differs_from_its_bundle_is_refused(self):
        from safetensors.torch import save_file
        tmp = Tmp()
        try:
            tke.write_bundle(training_model("S2"), tmp.root / "b", root=tmp.root)
            tke.derive_view(tmp.root / "b", tmp.root / "v", root=tmp.root)
            manifest, tensors = tke.read_bundle(tmp.root / "b")
            tke.check_view(tmp.root / "v", manifest, tensors)
            view = load_file(tmp.root / "v" / ternary_export.FILE)
            name = next(k for k in view if k.endswith(".scale"))
            for change in ("value", "dtype", "extra"):
                bad = dict(view)
                if change == "value":
                    bad[name] = bad[name] * 2
                elif change == "dtype":
                    bad[name] = bad[name].to(torch.float16)
                else:
                    bad["extra.tensor"] = torch.zeros(1)
                save_file(bad, tmp.root / "v" / ternary_export.FILE, metadata={"format": ternary_export.FORMAT})
                with self.subTest(change=change), self.assertRaises(ValueError):
                    tke.check_view(tmp.root / "v", manifest, tensors)
        finally:
            tmp.close()

    def test_failed_export_restores_the_seam(self):
        saved = paths.ARTIFACTS
        tmp = Tmp()
        try:
            tke.write_bundle(training_model("S1"), tmp.root / "b", root=tmp.root)
            with mock.patch.object(ternary_export, "export_model", side_effect=RuntimeError("disk full")):
                with self.assertRaises(RuntimeError):
                    tke.derive_view(tmp.root / "b", tmp.root / "v", root=tmp.root)
            self.assertIs(paths.ARTIFACTS, saved)
        finally:
            tmp.close()

    def test_root_is_keat_prime(self):
        self.assertEqual(tke.keat_root(), relabel_paths.ARTIFACTS / "keat-prime")


if __name__ == "__main__":
    unittest.main()
