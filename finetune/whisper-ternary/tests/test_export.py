"""Packing, export and reconstruction tests on a tiny random Whisper config (CPU, no download).
Exports are written to a temporary directory under the artifact disk."""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import WhisperConfig, WhisperForConditionalGeneration

import paths
from export import export_model, load_export, pack_codes, reconstruction_check, unpack_codes
from quant import dequantize, parameter_accounting, quantize_model, quantized_module_names, set_weight_fraction


def tiny_model(seed: int = 0) -> WhisperForConditionalGeneration:
    """Random tiny Whisper; 1-D tensors (biases, norms) are perturbed so FP16 storage is not trivially exact."""
    torch.manual_seed(seed)
    config = WhisperConfig(
        vocab_size=100, d_model=32, encoder_layers=2, decoder_layers=2, encoder_attention_heads=2,
        decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64, num_mel_bins=8,
        max_source_positions=50, max_target_positions=20, pad_token_id=99, bos_token_id=98,
        eos_token_id=99, decoder_start_token_id=97, suppress_tokens=[], begin_suppress_tokens=[])
    model = WhisperForConditionalGeneration(config).eval()
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() == 1:
                param.add_(0.1 * torch.randn_like(param))
    return model


def batch() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    features = torch.randn(4, 8, 100, generator=generator)
    ids = torch.randint(0, 97, (4, 12), generator=generator)
    ids[:, 0] = 97
    return features, ids


def all_fields(packed: torch.Tensor) -> torch.Tensor:
    return torch.stack([(packed >> shift) & 3 for shift in (0, 2, 4, 6)], dim=-1)


class PackTest(unittest.TestCase):
    def test_round_trip_small_widths(self) -> None:
        generator = torch.Generator().manual_seed(0)
        for width in range(1, 10):
            codes = (torch.randint(0, 3, (5, width), generator=generator) - 1).to(torch.int8)
            packed = pack_codes(codes)
            self.assertEqual((packed.dtype, tuple(packed.shape)), (torch.uint8, (5, math.ceil(width / 4))))
            self.assertFalse((all_fields(packed) == 3).any())
            self.assertTrue(torch.equal(unpack_codes(packed, width), codes))

    def test_round_trip_large(self) -> None:
        codes = (torch.randint(0, 3, (300, 1537), generator=torch.Generator().manual_seed(1)) - 1).to(torch.int8)
        packed = pack_codes(codes)
        self.assertEqual(tuple(packed.shape), (300, 385))
        self.assertFalse((all_fields(packed) == 3).any())
        self.assertTrue(torch.equal(unpack_codes(packed, 1537), codes))

    def test_bit_layout(self) -> None:
        codes = torch.tensor([[0, 1, -1, 1, -1]], dtype=torch.int8)
        # byte 0 = 01 10 01 00 (codes 3..0), byte 1 = 00 00 00 10 (code 4, zero padding)
        self.assertEqual(pack_codes(codes).tolist(), [[0b01100100, 0b00000010]])

    def test_invalid_inputs_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pack_codes(torch.tensor([[2]], dtype=torch.int8))
        with self.assertRaises(ValueError):
            pack_codes(torch.tensor([[1.0]]))
        with self.assertRaises(ValueError):
            unpack_codes(torch.tensor([[0b11]], dtype=torch.uint8), 1)
        with self.assertRaises(ValueError):
            unpack_codes(torch.tensor([[0b0100]], dtype=torch.uint8), 1)  # nonzero tail padding
        with self.assertRaises(ValueError):
            unpack_codes(torch.zeros(1, 2, dtype=torch.uint8), 4)


class ExportTest(unittest.TestCase):
    def setUp(self) -> None:
        paths.require_mount()
        (paths.ARTIFACTS / "tmp").mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=paths.ARTIFACTS / "tmp")
        self.out = Path(self.tmp.name) / "export"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def export(self, include_embedding: bool) -> tuple[WhisperForConditionalGeneration, dict]:
        model = tiny_model()
        quantize_model(model, include_embedding=include_embedding)
        manifest = export_model(model, self.out, {"arm": "test", "run_id": "unit"})
        return model, manifest

    def check_file(self, model: WhisperForConditionalGeneration, manifest: dict, tied_quantized: bool) -> None:
        path = self.out / "export.safetensors"
        self.assertEqual(json.loads((self.out / "manifest.json").read_text()), manifest)
        self.assertEqual(manifest["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(manifest["extra"], {"arm": "test", "run_id": "unit"})
        self.assertEqual(manifest["parameter_accounting"], parameter_accounting(model))
        self.assertEqual(manifest["tied"], {"proj_out": "model.decoder.embed_tokens"})
        sizes = manifest["bytes"]
        parts = ("packed_code_bytes", "scale_bytes", "bias_bytes", "fp16_residual_bytes", "header_bytes")
        self.assertEqual(sum(sizes[k] for k in parts), sizes["file_bytes"])
        self.assertEqual(sizes["file_bytes"], path.stat().st_size)
        self.assertEqual(sizes["original_fp32_bytes"], 4 * sum(p.numel() for p in model.parameters()))
        self.assertEqual(sizes["original_fp16_bytes"], sizes["original_fp32_bytes"] // 2)

        names = quantized_module_names(model)
        self.assertEqual(sorted(manifest["quantized_layers"]), names)
        stored = [n for n in names if not (tied_quantized and n == "proj_out")]
        covered = {"proj_out.weight"} | {f"{n}.{k}" for n in names for k in ("weight", "bias")}
        residual = [k for k in model.state_dict() if k not in covered]
        with safe_open(path, "pt") as handle:
            keys = set(handle.keys())
            expected = {f"{n}.{k}" for n in stored for k in ("codes", "scale")}
            expected |= {f"{n}.bias" for n in stored if manifest["quantized_layers"][n]["bias"]}
            self.assertEqual(keys, expected | set(residual))
            packed = scales = biases = fp16 = 0
            for name in stored:
                codes, scale = model.get_submodule(name).quantized_weight()
                out_features, in_features = manifest["quantized_layers"][name]["shape"]
                file_codes = handle.get_tensor(f"{name}.codes")
                self.assertEqual((file_codes.dtype, tuple(file_codes.shape)),
                                 (torch.uint8, (out_features, math.ceil(in_features / 4))))
                self.assertTrue(torch.equal(unpack_codes(file_codes, in_features), codes))
                self.assertTrue(torch.equal(handle.get_tensor(f"{name}.scale"), scale))
                packed += file_codes.nbytes
                scales += scale.nbytes
                if manifest["quantized_layers"][name]["bias"]:
                    bias = handle.get_tensor(f"{name}.bias")
                    self.assertTrue(torch.equal(bias, model.get_submodule(name).bias.detach()))
                    biases += bias.nbytes
            for key in residual:
                tensor = handle.get_tensor(key)
                self.assertEqual(tensor.dtype, torch.float16, key)
                fp16 += tensor.nbytes
        self.assertEqual((packed, scales, biases, fp16), tuple(sizes[k] for k in parts[:4]))

    def check_rebuilt(self, model: WhisperForConditionalGeneration, layers: int) -> dict:
        rebuilt = load_export(self.out)
        self.assertEqual(quantized_module_names(rebuilt), [])
        self.assertIs(rebuilt.proj_out.weight, rebuilt.model.decoder.embed_tokens.weight)
        self.assertFalse(rebuilt.training)
        self.assertTrue(all(p.dtype == torch.float32 for p in rebuilt.parameters()))
        self.assertEqual(rebuilt.config.to_dict(), model.config.to_dict())
        self.assertEqual(rebuilt.generation_config.to_dict(), model.generation_config.to_dict())
        check = reconstruction_check(model, rebuilt, *batch())
        print(f"\n    reconstruction {check}, file_bytes {self.out.joinpath('export.safetensors').stat().st_size}")
        self.assertTrue(check["codes_exact"] and check["scales_exact"])
        self.assertEqual(check["checked_layers"], layers)
        self.assertLess(check["logits_max_abs_diff"], 5e-2)
        self.assertGreater(check["argmax_agreement"], 0.98)
        return check

    def test_projection_arm(self) -> None:
        model, manifest = self.export(include_embedding=False)
        self.assertEqual(len(manifest["quantized_layers"]), 32)
        self.check_file(model, manifest, tied_quantized=False)
        self.check_rebuilt(model, layers=32)

    def test_embedding_arm_stores_the_tie_once(self) -> None:
        model, manifest = self.export(include_embedding=True)
        self.assertEqual(len(manifest["quantized_layers"]), 34)
        self.assertEqual(manifest["quantized_layers"]["proj_out"], {"kind": "linear", "shape": [100, 32], "bias": False})
        self.assertNotIn("proj_out", manifest["code_histogram"]["per_layer"])
        self.check_file(model, manifest, tied_quantized=True)
        # The zeroed padding row has all-zero codes and the 1e-8 floor scale.
        codes, scale = model.model.decoder.embed_tokens.quantized_weight()
        self.assertFalse(codes[99].any())
        self.assertEqual(scale[99].item(), torch.tensor(1e-8).item())
        self.check_rebuilt(model, layers=34)

    def test_reconstruction_check_detects_mismatch(self) -> None:
        model, _ = self.export(include_embedding=False)
        rebuilt = load_export(self.out)
        weight = rebuilt.get_submodule("model.encoder.layers.0.fc1").weight
        with torch.no_grad():
            weight[0] *= 2
        check = reconstruction_check(model, rebuilt, *batch())
        self.assertEqual((check["codes_exact"], check["scales_exact"]), (True, False))
        with torch.no_grad():
            weight[0] *= -0.5
        self.assertFalse(reconstruction_check(model, rebuilt, *batch())["codes_exact"])

    def test_rebuilt_weights_are_dequantized_codes(self) -> None:
        model, _ = self.export(include_embedding=True)
        rebuilt = load_export(self.out)
        for name in quantized_module_names(model):
            expected = dequantize(*model.get_submodule(name).quantized_weight())
            self.assertTrue(torch.equal(rebuilt.get_submodule(name).weight, expected), name)

    def test_mid_ramp_model_is_refused_until_fraction_one(self) -> None:
        model = tiny_model()
        quantize_model(model, include_embedding=True)
        set_weight_fraction(model, 0.5)
        with self.assertRaisesRegex(ValueError, "finish the ramp"):
            export_model(model, self.out, {"arm": "test", "run_id": "unit"})
        self.assertFalse(self.out.exists())  # refused before anything is written
        set_weight_fraction(model, 1.0)
        model.proj_out.fraction = 0.999  # one module (half of the tie) still in the ramp is enough
        with self.assertRaisesRegex(ValueError, "finish the ramp"):
            export_model(model, self.out, {"arm": "test", "run_id": "unit"})
        self.assertFalse(self.out.exists())
        set_weight_fraction(model, 1.0)
        manifest = export_model(model, self.out, {"arm": "test", "run_id": "unit"})
        self.check_file(model, manifest, tied_quantized=True)
        rebuilt = load_export(self.out)
        for fraction in (0.0, 0.5):
            set_weight_fraction(model, fraction)
            with self.assertRaisesRegex(ValueError, "finish the ramp"):
                reconstruction_check(model, rebuilt, *batch())
        set_weight_fraction(model, 1.0)
        self.check_rebuilt(model, layers=34)

    def test_refusals(self) -> None:
        with self.assertRaises(ValueError):
            export_model(tiny_model(), self.out, {})  # nothing quantized
        model = tiny_model()
        quantize_model(model, include_embedding=False)
        with self.assertRaises(ValueError):
            export_model(model, Path("/tmp/whisper-ternary-should-not-exist"), {})
        self.assertFalse(Path("/tmp/whisper-ternary-should-not-exist").exists())
        export_model(model, self.out, {})
        path = self.out / "export.safetensors"
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(bytes(data))
        with self.assertRaises(ValueError):
            load_export(self.out)


if __name__ == "__main__":
    unittest.main()
