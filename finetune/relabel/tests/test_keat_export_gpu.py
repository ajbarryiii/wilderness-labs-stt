"""Contract tests (CUDA; M1's real training checkpoint and export): the stage 5 export adapter and 8-bit forms on Kea-T
itself (keat-prime plan "Inputs", "Step 0", "Export").

- the student built from M1's latent checkpoint has 264 ternary modules at weight fraction 1 (the init check);
- with no scope in 8 bits, M1's training model (ternary scales computed on CUDA, as its export's were) taken through
  the complete path, bundle then the scoring view derived from it, gives an export.safetensors byte-identical to
  M1's own (sha a287e977...), with the export's model config;
- the none, S1 and S2 bundles' bytes per part equal the plan's table exactly (180.688, 173.524 and 166.734 MB of
  tensors);
- the S2 scoring view equals the training forward weights bit for bit (all 264 ternary modules, codes x
  fp32(fp16(s)); W8 of the prediction network, joint and pre_encode.out), M1's row scales are normal FP16 values,
  and inference.runtime.load_packed (the x1 stack's loader) accepts the view;
- the INT8 codes and scales of M1's float parts are the same on the CPU and on CUDA.
Run inside a GPU job (training/keat_launch.sh gpu-tests), under ./python.
"""
from __future__ import annotations

import json
import shutil
import unittest

import torch
from safetensors.torch import load_file

import export as ternary_export
import keat_step0 as t0
import quant
from test_keat_export import Tmp, expected_part_bytes
from training import eightbit as eb
from training import keat_eightbit as teb
from training import keat_export as tke
from training import keat_m1 as km1

HAS_M1 = torch.cuda.is_available() and km1.m1_checkpoint().exists() and km1.m1_export().exists()
PLAN_PARTS = {  # keat-prime plan "Step 0", the table (bytes)
    "none": {"ternary_codes": 150_994_944, "ternary_scales": 1_769_472, "ternary_biases": 0,
             "prediction_network": 14_439_680, "joint": 3_452_940, "pre_encode_out": 8_390_656,
             "other_fp16": 1_507_328, "fp32_feature_constants": 133_184, "integer_buffers": 192},
    "S2": {"ternary_codes": 150_994_944, "ternary_scales": 884_736, "ternary_biases": 0,
           "prediction_network": 7_275_140, "joint": 1_738_020, "pre_encode_out": 4_200_448,
           "other_fp16": 1_507_328, "fp32_feature_constants": 133_184, "integer_buffers": 192},
}
PLAN_PARTS["S1"] = {**PLAN_PARTS["none"], "prediction_network": PLAN_PARTS["S2"]["prediction_network"]}
PLAN_MB = {"none": 180.688, "S1": 173.524, "S2": 166.734}


@unittest.skipUnless(HAS_M1, "needs CUDA and M1's checkpoint and export")
class M1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Tmp()
        cls.models = t0.M1Models()
        cls.export_manifest = json.loads((km1.m1_export() / ternary_export.MANIFEST).read_text())

    @classmethod
    def tearDownClass(cls):
        cls.models.close()
        cls.tmp.close()

    def form(self, scope: str):
        model = self.models().to("cuda")
        teb.apply_keat_scope(model, scope)
        return model

    def test_student_has_264_ternary_modules_at_fraction_1(self):
        self.assertEqual(self.models.facts["sha256"], km1.M1_SHA256)
        model = self.models()
        self.assertEqual(len(quant.quantized_module_names(model)), quant.EXPECTED_MODULES)
        self.assertEqual(quant.weight_fraction(model), 1.0)
        self.assertEqual(quant.parameter_accounting(model)["total_parameters"], quant.TOTAL_PARAMETERS)

    def test_no_scope_view_is_m1s_export_byte_for_byte(self):
        model = self.form("none")
        bundle = tke.write_bundle(model, self.tmp.root / "none" / "bundle", root=self.tmp.root)
        del model
        torch.cuda.empty_cache()
        self.assertEqual(bundle["quantization_device"], "cuda")
        view = tke.derive_view(self.tmp.root / "none" / "bundle", self.tmp.root / "none" / "view", root=self.tmp.root)
        self.assertEqual(view["sha256"], km1.M1_EXPORT_SHA256)
        self.assertEqual((self.tmp.root / "none" / "view" / ternary_export.FILE).read_bytes(),
                         (km1.m1_export() / ternary_export.FILE).read_bytes())
        self.assertEqual(view["config"], self.export_manifest["config"])
        self.assertEqual(view["bytes"]["file_bytes"], self.export_manifest["bytes"]["file_bytes"])
        print(f"\nnone: view {view['sha256']}, bundle tensor MB {tke.tensor_megabytes(bundle):.4f}")

    def test_bundle_bytes_per_part_are_the_plans_table(self):
        for scope in ("none", "S1", "S2"):
            with self.subTest(scope=scope):
                model = self.form(scope)
                bundle = tke.write_bundle(model, self.tmp.root / f"bytes-{scope}" / "bundle", root=self.tmp.root)
                expected = expected_part_bytes(model, scope)
                del model
                torch.cuda.empty_cache()
                self.assertEqual(bundle["bytes"]["parts"], PLAN_PARTS[scope])
                self.assertEqual(bundle["bytes"]["parts"], expected)
                mb = tke.tensor_megabytes(bundle)
                print(f"\n{scope}: bundle tensor MB {mb:.4f} (plan {PLAN_MB[scope]}), file "
                      f"{bundle['bytes']['file_bytes']} B")
                self.assertEqual(round(mb, 3), PLAN_MB[scope])
                shutil.rmtree(self.tmp.root / f"bytes-{scope}", ignore_errors=True)

    def test_s2_view_equals_forward_weights_and_load_packed_accepts_it(self):
        model = self.form("S2")
        tke.write_bundle(model, self.tmp.root / "S2v" / "bundle", root=self.tmp.root)
        view_dir = self.tmp.root / "S2v" / "view"
        view_manifest = tke.derive_view(self.tmp.root / "S2v" / "bundle", view_dir, root=self.tmp.root)
        self.assertEqual(view_manifest["config"], self.export_manifest["config"])
        view = load_file(view_dir / ternary_export.FILE)
        names = quant.quantized_module_names(model)
        self.assertEqual(len(names), 264)
        rounded, lo, hi = 0, float("inf"), 0.0
        with torch.no_grad():
            for n in names:
                module = model.get_submodule(n)
                forward = module.forward_weight().cpu()
                rows, cols = forward.shape
                rebuilt = quant.dequantize(ternary_export.unpack_codes(view[f"{n}.codes"], cols), view[f"{n}.scale"])
                self.assertTrue(torch.equal(rebuilt, forward), n)
                _, s32 = quant.ternary_quantize(module.matrix())
                half = s32.to(torch.float16)
                self.assertTrue(bool((half.abs() >= torch.finfo(torch.float16).tiny).all()), f"{n}: subnormal FP16")
                rounded += int((half.float() != s32).sum())
                lo, hi = min(lo, float(s32.min())), max(hi, float(s32.max()))
            for key in eb.keys_of(teb.KEAT_SCOPES["S2"]):
                forward = (model.encoder.pre_encode.out.forward_weight() if key == "encoder.pre_encode.out.weight"
                           else model.get_parameter(key)).detach().cpu()
                self.assertTrue(torch.equal(view[key].float(), forward), key)
        self.assertGreater(rounded, 0, "FP16 rounding changed some row scales")
        print(f"\nS2: {rounded} ternary row scales changed by the FP16 rounding; range [{lo:.4f}, {hi:.4f}]")
        self.assertGreaterEqual(lo, 0.0152 - 1e-4)
        self.assertLessEqual(hi, 3.32 + 1e-2)
        del model
        torch.cuda.empty_cache()
        from inference.runtime import load_packed
        packed = load_packed(view_dir)
        try:
            for n in names[:: 37]:
                self.assertTrue(torch.equal(packed.get_submodule(n).scale.cpu(), view[f"{n}.scale"]), n)
        finally:
            del packed
            torch.cuda.empty_cache()

    def test_int8_is_device_independent(self):
        base = self.models()
        with torch.no_grad():
            for key in eb.keys_of(teb.KEAT_SCOPES["S2"]):
                w = base.get_parameter(key).detach()
                q_cpu, s_cpu = eb.int8_quantize(w)
                q_gpu, s_gpu = eb.int8_quantize(w.cuda())
                self.assertTrue(torch.equal(q_cpu, q_gpu.cpu()) and torch.equal(s_cpu, s_gpu.cpu()), key)


if __name__ == "__main__":
    unittest.main()
