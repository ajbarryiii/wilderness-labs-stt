"""Contract tests (CUDA; MB1's real training checkpoint and export): the export adapter on Kea-B itself (keab-prime
plan "Export").

- with no scope in 8 bits, MB1's training model (its checkpoint's latent weights, binary scales computed on CUDA as
  its export's were) taken through the complete path, bundle then the scoring view derived from it, gives an
  export.safetensors byte-identical to MB1's existing one (sha 9873d2f1...), with the export's model config;
- the S1 and S2 bundles' bytes per part equal the sizes expected from the model's shapes, and their tensor bytes
  are the plan's ~98.0 and ~91.2 MB (within 0.1 MB);
- the S2 scoring view's tensors equal the training forward weights bit for bit: all 264 binary modules
  (codes x fp32(fp16(alpha))), and W8 of the prediction network, joint and pre_encode.out;
- the INT8 codes and scales of MB1's float parts are the same on the CPU and on CUDA.
Run inside a GPU job (finetune/relabel/training/keab_launch.sh gpu-tests).
"""
from __future__ import annotations

import json
import shutil
import unittest

import torch
from safetensors.torch import load_file

import keab_step0 as k0
from test_keab_export import Tmp, expected_part_bytes
from training import eightbit as eb
from training import keab_export as ke

import binary_export  # noqa: E402
import binary_quant  # noqa: E402

HAS_MB1 = torch.cuda.is_available() and k0.mb1_checkpoint().exists() and k0.mb1_export().exists()
PLAN_MB = {"S1": 98.0, "S2": 91.2}


@unittest.skipUnless(HAS_MB1, "needs CUDA and MB1's checkpoint and export")
class Mb1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Tmp()
        cls.models = k0.Mb1Models()
        cls.base = cls.models()
        cls.export_manifest = json.loads((k0.mb1_export() / binary_export.MANIFEST).read_text())
        cls.bundles = {}

    @classmethod
    def tearDownClass(cls):
        cls.models.close()
        cls.tmp.close()

    def form(self, scope: str):
        model = self.models().to("cuda")
        eb.apply_keab_scope(model, scope)
        return model

    def test_no_scope_view_is_mb1s_export_byte_for_byte(self):
        model = self.form("none")
        bundle = ke.write_bundle(model, self.tmp.root / "none" / "bundle", root=self.tmp.root)
        del model
        torch.cuda.empty_cache()
        self.assertEqual(bundle["quantization_device"], "cuda")
        view = ke.derive_view(self.tmp.root / "none" / "bundle", self.tmp.root / "none" / "view", root=self.tmp.root)
        self.assertEqual(view["sha256"], k0.MB1_EXPORT_SHA256)
        self.assertEqual((self.tmp.root / "none" / "view" / binary_export.FILE).read_bytes(),
                         (k0.mb1_export() / binary_export.FILE).read_bytes())
        self.assertEqual(view["config"], self.export_manifest["config"])
        self.assertEqual(view["bytes"]["file_bytes"], self.export_manifest["bytes"]["file_bytes"])
        print(f"\nnone: view {view['sha256']}, bundle tensor MB {ke.tensor_megabytes(bundle):.4f}")

    def test_bundle_bytes_per_part(self):
        for scope in ("S1", "S2"):
            with self.subTest(scope=scope):
                model = self.form(scope)
                bundle = ke.write_bundle(model, self.tmp.root / scope / "bundle", root=self.tmp.root)
                expected = expected_part_bytes(model, scope)
                del model
                torch.cuda.empty_cache()
                self.assertEqual(bundle["bytes"]["parts"], expected)
                mb = ke.tensor_megabytes(bundle)
                print(f"\n{scope}: bundle tensor MB {mb:.4f} (plan ~{PLAN_MB[scope]}), file "
                      f"{bundle['bytes']['file_bytes']} B, parts {bundle['bytes']['parts']}")
                self.assertAlmostEqual(mb, PLAN_MB[scope], delta=0.1)
                shutil.rmtree(self.tmp.root / scope, ignore_errors=True)

    def test_s2_view_equals_forward_weights(self):
        model = self.form("S2")
        ke.write_bundle(model, self.tmp.root / "S2v" / "bundle", root=self.tmp.root)
        view_manifest = ke.derive_view(self.tmp.root / "S2v" / "bundle", self.tmp.root / "S2v" / "view",
                                       root=self.tmp.root)
        self.assertEqual(view_manifest["config"], self.export_manifest["config"])
        view = load_file(self.tmp.root / "S2v" / "view" / binary_export.FILE)
        names = binary_quant.quantized_module_names(model)
        self.assertEqual(len(names), 264)
        rounded = 0
        with torch.no_grad():
            for n in names:
                module = model.get_submodule(n)
                forward = module.forward_weight().cpu()
                rows, cols = forward.shape
                rebuilt = binary_quant.dequantize(binary_export.unpack_codes(view[f"{n}.codes"], cols),
                                                  view[f"{n}.scale"])
                self.assertTrue(torch.equal(rebuilt, forward), n)
                rounded += int((module.quantized_weight()[1] != binary_quant.binary_quantize(module.matrix())[1]).sum())
            for key in eb.keys_of(eb.KEAB_SCOPES["S2"]):
                forward = (model.encoder.pre_encode.out.forward_weight() if key == "encoder.pre_encode.out.weight"
                           else model.get_parameter(key)).detach().cpu()
                self.assertTrue(torch.equal(view[key].float(), forward), key)
        self.assertGreater(rounded, 0, "FP16 rounding changed some row scales")
        print(f"\nS2: {rounded} binary row scales changed by the FP16 rounding")

    def test_int8_is_device_independent(self):
        with torch.no_grad():
            for key in eb.keys_of(eb.KEAB_SCOPES["S2"]):
                w = self.base.get_parameter(key).detach()
                q_cpu, s_cpu = eb.int8_quantize(w)
                q_gpu, s_gpu = eb.int8_quantize(w.cuda())
                self.assertTrue(torch.equal(q_cpu, q_gpu.cpu()) and torch.equal(s_cpu, s_gpu.cpu()), key)


if __name__ == "__main__":
    unittest.main()
