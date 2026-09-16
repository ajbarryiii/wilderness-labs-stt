"""Exact CPU packed-dot checks, including ragged layouts and threshold edges."""
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest import mock

import torch

import native


def quantize(x, activation_bits):
    if activation_bits == 1:
        return torch.where(x >= 0, 1.0, -1.0)
    return torch.where(x >= 0.5, 1.0, torch.where(x <= -0.5, -1.0, 0.0))


def reference(x, codes, scale, activation_bits, bias=None):
    out = (quantize(x, activation_bits) @ codes.float().T) * scale.reshape(-1)
    return out if bias is None else out + bias


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.backends = native.available_backends()
        if not cls.backends:
            raise unittest.SkipTest("This CPU has no hardware POPCNT")

    def test_exact_dot_ragged_and_multirow(self):
        generator = torch.Generator().manual_seed(917)
        shapes = [(1,1,1), (1,7,7), (3,8,63), (3,9,64), (2,31,65),
                  (4,7,127), (5,8,128), (7,9,129), (8,15,255), (9,16,257),
                  (17,17,511), (1,16,512), (3,19,513), (2,33,1024),
                  (1,51864,65)]
        for backend in self.backends:
            for threads in (1, 3):
                for weight_bits, activation_bits in ((1,1), (2,1), (2,2), (1,2)):
                    for m,n,k in shapes:
                        with self.subTest(backend=backend, threads=threads, w=weight_bits,
                                          a=activation_bits, shape=(m,n,k)):
                            codes = torch.randint(0, 2 if weight_bits == 1 else 3,
                                      (n,k), dtype=torch.int8, generator=generator)
                            codes = codes*2-1 if weight_bits == 1 else codes-1
                            scale = torch.linspace(0.15, 1.3, n)
                            bias = torch.randn(n, generator=generator)
                            x = torch.randn(m,k, generator=generator)
                            packed = native.PackedWeight(codes, scale[:,None],
                                      activation_bits, backend, threads, weight_bits)
                            torch.testing.assert_close(packed.linear(x,bias),
                                reference(x,codes,scale,activation_bits,bias), rtol=0,atol=0)
                            self.assertEqual(packed.metadata["weight_bits"], weight_bits)

    def test_optimized_tiles_reuse_scratch_and_match_existing_backend(self):
        if "avx512_opt" not in self.backends:
            self.skipTest("Optimized AVX512 backend is unavailable on this CPU")
        generator = torch.Generator().manual_seed(1501)
        # A single packed object alternates GEMV, complete four-row tiles,
        # all three tile remainders, and empty calls. Zero inputs between
        # random inputs expose stale sign/nonzero bits when scratch is reused.
        for weight_bits, activation_bits in ((1,1), (2,1), (2,2), (1,2)):
            for threads in (1,3):
                for k in (65,257):
                    n = 17
                    codes = torch.randint(0, 2 if weight_bits == 1 else 3,
                                          (n,k), dtype=torch.int8, generator=generator)
                    codes = codes*2-1 if weight_bits == 1 else codes-1
                    scale = torch.linspace(-0.9,0.9,n)
                    bias = torch.randn(n,generator=generator)
                    old = native.PackedWeight(codes,scale,activation_bits,
                                              "avx512",threads,weight_bits)
                    optimized = native.PackedWeight(codes,scale,activation_bits,
                                                    "avx512_opt",threads,weight_bits)
                    for step,m in enumerate((17,1,2,3,4,5,7,8,9,0,16,1)):
                        with self.subTest(w=weight_bits,a=activation_bits,threads=threads,
                                          shape=(m,n,k),step=step):
                            x = (torch.zeros(m,k) if step % 3 == 1 else
                                 torch.randn(m,k,generator=generator))
                            actual = optimized.linear(x,bias)
                            torch.testing.assert_close(actual,old.linear(x,bias),rtol=0,atol=0)
                            torch.testing.assert_close(actual,
                                reference(x,codes,scale,activation_bits,bias),rtol=0,atol=0)

    def test_epilogue_preserves_separate_float32_scale_and_bias(self):
        # The dot is three: float32(3 * float32(0.1)) rounds before bias.
        # A fused multiply-add leaves a small residual after cancellation.
        # Signed zeros and a ragged final output vector are checked as bits.
        codes = torch.ones(17,3,dtype=torch.int8)
        scale = torch.tensor([0.1,-0.1,0.3,-0.3,0.0,-0.0,1.0,-1.0]*2+[0.1])
        x = torch.ones(9,3)
        bias = -(torch.full((17,),3.0)*scale)
        for backend in self.backends:
            for activation_bits in (1,2):
                packed = native.PackedWeight(codes,scale,activation_bits,backend,3,1)
                for current_bias in (None,bias):
                    with self.subTest(backend=backend,a=activation_bits,
                                      has_bias=current_bias is not None):
                        actual = packed.linear(x,current_bias)
                        expected = reference(x,codes,scale,activation_bits,current_bias)
                        self.assertTrue(torch.equal(actual.view(torch.int32),
                                                    expected.view(torch.int32)))

    def test_threshold_boundaries_nan_infinity_and_signed_zero(self):
        edge = torch.tensor([-float("inf"), -0.5, -0.0, 0.0, 0.5, float("inf"), float("nan")])
        finite = torch.tensor([-0.5, 0.0, 0.5])
        edge = torch.cat((edge, torch.nextafter(finite, torch.full_like(finite,-float("inf"))),
                          torch.nextafter(finite, torch.full_like(finite,float("inf")))))
        # Repeat across 16-, 64-, and 512-bit boundaries with a ragged tail.
        x = edge.repeat(43)[:533].reshape(1,-1)
        codes = torch.arange(19*x.numel(), dtype=torch.int64).remainder(3).sub(1).to(torch.int8).reshape(19,-1)
        for backend in self.backends:
            for activation_bits in (1,2):
                packed = native.PackedWeight(codes, 1.0, activation_bits, backend)
                torch.testing.assert_close(packed.linear(x),
                    reference(x,codes,torch.ones(19),activation_bits), rtol=0,atol=0)

    def test_noncontiguous_higher_rank_and_empty_input(self):
        codes = torch.tensor([[1,0,-1],[-1,1,0]],dtype=torch.int8)
        scales = torch.tensor([0.25,0.5])
        x = torch.arange(24,dtype=torch.float32).reshape(2,3,4).transpose(1,2)-12
        for backend in self.backends:
            packed = native.PackedWeight(codes.T.contiguous().T, scales, 2, backend, 2)
            torch.testing.assert_close(packed.linear(x),reference(x,codes,scales,2),rtol=0,atol=0)
            self.assertEqual(packed.linear(torch.empty(0,3)).shape,(0,2))
            self.assertEqual(packed.linear(torch.empty(2,0,3)).shape,(2,0,2))
            self.assertEqual(packed.linear(torch.ones(3)).shape,(2,))

    def test_embedding_dequantization_and_storage(self):
        generator = torch.Generator().manual_seed(3)
        codes = torch.randint(-1,2,(19,129),dtype=torch.int8,generator=generator)
        scale = torch.linspace(0.1,0.9,19)
        ids = torch.tensor([[0,18],[8,5]],dtype=torch.int32).T
        for backend in self.backends:
            packed = native.PackedWeight(codes,scale,2,backend,2)
            torch.testing.assert_close(packed.dequantize(), codes.float()*scale[:,None],rtol=0,atol=0)
            torch.testing.assert_close(packed.embedding(ids),
                (codes.float()*scale[:,None])[ids.long()],rtol=0,atol=0)
            self.assertEqual(packed.embedding(torch.empty((2,0),dtype=torch.int64)).shape,(2,0,129))
            self.assertEqual(packed.embedding(torch.tensor(2)).shape,(129,))
            rows = 24 if backend in ("avx512","avx512_opt") else 19
            self.assertEqual(packed.storage_bytes,rows*3*8*2+rows*8+19*4)
            self.assertEqual(packed.scratch_bytes,0)
            packed.linear(torch.ones(5,129))
            self.assertGreaterEqual(packed.scratch_bytes,5*3*8*2)
            for invalid in (-1,19):
                with self.assertRaises(IndexError):
                    packed.embedding(torch.tensor([invalid]))

    def test_shared_object_concurrent_calls(self):
        codes = torch.randint(-1,2,(19,129),dtype=torch.int8)
        xs = [torch.randn(m,129) for m in (1,13,2,7,20,3,17,4)]
        for backend in self.backends:
            packed = native.PackedWeight(codes,0.25,2,backend,1)
            with ThreadPoolExecutor(max_workers=4) as pool:
                actual = list(pool.map(packed.linear,xs))
            for x,y in zip(xs,actual):
                torch.testing.assert_close(y,reference(x,codes,torch.tensor(0.25),2),rtol=0,atol=0)

    def test_invalid_inputs_fail_cleanly(self):
        codes = torch.ones((2,3),dtype=torch.int8)
        invalid = [dict(codes=codes.float()), dict(codes=codes[:,0]),
                   dict(codes=torch.full((2,3),2,dtype=torch.int8)),
                   dict(codes=torch.zeros((2,3),dtype=torch.int8),weight_bits=1),
                   dict(codes=codes,scale=torch.ones(2,2)),
                   dict(codes=codes,scale=float("nan")),
                   dict(codes=codes,activation_bits=3),dict(codes=codes,weight_bits=0),
                   dict(codes=codes,threads=0),dict(codes=codes,backend="bogus")]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                native.PackedWeight(**dict(dict(scale=1,backend=self.backends[0]),**args))
        packed = native.PackedWeight(codes,1,backend=self.backends[0])
        for x,bias in ((torch.ones(3,dtype=torch.float64),None),
                       (torch.ones(4),None),(torch.ones(3),torch.ones(3)),
                       (torch.ones(3),torch.ones(2,dtype=torch.float64))):
            with self.assertRaises(ValueError):
                packed.linear(x,bias)
        with self.assertRaises(ValueError):
            packed.embedding(torch.tensor([0.0]))

    def test_cache_requires_mount(self):
        # No compiler call or directory creation is allowed without the mount.
        with mock.patch.object(native,"_LIBRARY",None), \
             mock.patch.object(native.os.path,"ismount",return_value=False), \
             mock.patch.object(native.subprocess,"run") as run:
            with self.assertRaisesRegex(RuntimeError,"not mounted"):
                native._load_library()
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
