"""Tests of the tests: deliberately broken BinaryLayers must FAIL the named contract checks.

Each mutant differs from H's known-good reference in one way; the test runs the contract method that
should catch it and asserts a failure or error. Keeps the contract suite from being vacuous.
"""
from __future__ import annotations

import unittest

import torch

from gpu1bit.shared import api
from gpu1bit.shared.reference_layer import ReferenceLayer, _shifts
from gpu1bit.tests import test_contract


class MsbFirst(ReferenceLayer):
    def _dense64(self):
        bits = ((self.codes[..., None] >> (7 - _shifts(self.codes.device))) & 1).flatten(1)[:, :self.in_features]
        return (bits.to(torch.float64) * 2 - 1) * self.scale.to(torch.float64)[:, None]


class IgnoresStrides(ReferenceLayer):
    def linear(self, x):
        if x.numel():
            x = torch.as_strided(x, x.shape, torch.empty(x.shape).stride())
        return super().linear(x)


class ExpandsToInt8(ReferenceLayer):
    @classmethod
    def prepare(cls, codes, scale, bias, k, device):
        layer = super().prepare(codes, scale, bias, k, device)
        layer.expanded = api.unpack_signs(codes, k).to(device)
        return layer

    def weight_bytes(self):
        return api.resident_bytes(self.codes, self.scale, self.bias, self.expanded)


class HiddenLazyCache(ReferenceLayer):
    def linear(self, x):
        if not hasattr(self, "_cache"):
            self._cache = self._dense64()
        return super().linear(x)


class Bf16OnePass(ReferenceLayer):
    def linear(self, x):
        self._check(x, x.dim() >= 1, x.shape[-1] if x.dim() else -1)
        if x.numel() == 0:
            return torch.empty((*x.shape[:-1], self.out_features), device=x.device)
        w = self._dense64().to(torch.bfloat16)
        y = (x.to(torch.bfloat16) @ w.t()).float()
        return (y + self.bias if self.bias is not None else y).contiguous()


class AllowsGrad(ReferenceLayer):
    def _check(self, x, rank_ok, k):
        if x.dtype != torch.float32 or not x.is_cuda or not rank_ok or k != self.in_features:
            raise ValueError("bad input")


class ReusesOutput(ReferenceLayer):
    def linear(self, x):
        y = super().linear(x)
        buf = getattr(self, "_out", None)
        if buf is None or buf.shape != y.shape:
            self._out = buf = torch.empty_like(y)
        return buf.copy_(y)


class AcceptsTailBits(ReferenceLayer):
    @classmethod
    def prepare(cls, codes, scale, bias, k, device):
        return cls(codes.to(device, copy=True), scale.to(device, copy=True),
                   None if bias is None else bias.to(device, copy=True), k)


class Noisy(ReferenceLayer):
    def linear(self, x):
        y = super().linear(x)
        return y + 1e-7 * torch.randn_like(y) * y.abs()


class SilenceOffByTiny(ReferenceLayer):
    def linear(self, x):
        return super().linear(x) + 1e-30


MUTANTS = [
    (MsbFirst, "test_semantics_integer_exact"),
    (IgnoresStrides, "test_semantics_integer_exact"),
    (ExpandsToInt8, "test_storage_budget_and_honest_weight_bytes"),
    (HiddenLazyCache, "test_storage_budget_and_honest_weight_bytes"),
    (Bf16OnePass, "test_synthetic_numerics_vs_m1"),
    (AllowsGrad, "test_inference_only"),
    (ReusesOutput, "test_outputs_are_new_and_do_not_alias"),
    (AcceptsTailBits, "test_prepare_rejects_invalid_weights"),
    (Noisy, "test_deterministic"),
    (SilenceOffByTiny, "test_synthetic_numerics_vs_m1"),
]


class ContractCatchesMutants(unittest.TestCase):
    def test_each_mutant_fails_its_check(self):
        for mutant, method in MUTANTS:
            with self.subTest(mutant=mutant.__name__, check=method):
                case = type(f"Mutant_{mutant.__name__}", (test_contract.Contract, unittest.TestCase),
                            {"impl_name": mutant.__name__, "impl": mutant})
                result = unittest.TestResult()
                case(method).run(result)
                caught = len(result.failures) + len(result.errors)
                self.assertGreater(caught, 0, f"{method} did not catch {mutant.__name__}")

    def test_reference_passes_the_same_checks(self):
        case = type("Reference", (test_contract.Contract, unittest.TestCase),
                    {"impl_name": "h_reference", "impl": ReferenceLayer})
        result = unittest.TestResult()
        for method in sorted({m for _, m in MUTANTS}):
            case(method).run(result)
        self.assertTrue(result.wasSuccessful(), [str(e[1])[:400] for e in result.failures + result.errors])


if __name__ == "__main__":
    unittest.main()
