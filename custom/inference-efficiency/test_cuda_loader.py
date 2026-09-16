"""CPU-only regression checks for the optional CUDA GEMV loader."""

import os
import unittest
from unittest.mock import patch
import warnings

from torch.utils import cpp_extension

from kernels import cuda_gemv


class CudaLoaderTests(unittest.TestCase):
    def setUp(self):
        cuda_gemv.load.cache_clear()
        cuda_gemv.try_load.cache_clear()

    def tearDown(self):
        cuda_gemv.load.cache_clear()
        cuda_gemv.try_load.cache_clear()

    def test_preload_repairs_cuda_home_cached_before_configuration(self):
        configured = "/configured/cuda"
        with (patch.object(cpp_extension, "CUDA_HOME", None),
              patch.dict(os.environ, {"CUDA_HOME": configured}),
              patch.object(cuda_gemv.ctypes, "CDLL")):
            cuda_gemv._preload_nvrtc()
            self.assertEqual(cpp_extension.CUDA_HOME, configured)
            self.assertIn(configured + "/include", cpp_extension.include_paths("cuda"))

    def test_explicit_cuda_load_preserves_compiler_error(self):
        with (patch.object(cuda_gemv, "_preload_nvrtc"),
              patch("torch.cuda._compile_kernel", side_effect=RuntimeError("bad CUDA source"))):
            with self.assertRaisesRegex(RuntimeError, "bad CUDA source"):
                cuda_gemv.load()

    def test_auto_fallback_reports_failure_once(self):
        with (patch.object(cuda_gemv, "load", side_effect=RuntimeError("bad CUDA source")) as compile,
              warnings.catch_warnings(record=True) as emitted):
            warnings.simplefilter("always")
            self.assertIsNone(cuda_gemv.try_load())
            self.assertIsNone(cuda_gemv.try_load())
            compile.assert_called_once()
            self.assertEqual(len(emitted), 1)
            self.assertIn("bad CUDA source", str(emitted[0].message))

    def test_required_cuda_never_silently_falls_back(self):
        with (patch.dict(os.environ, {"EFFICIENCY_REQUIRE_CUDA_GEMV": "1"}),
              patch.object(cuda_gemv, "load", side_effect=RuntimeError("bad CUDA source"))):
            with self.assertRaisesRegex(RuntimeError, "bad CUDA source"):
                cuda_gemv.try_load()


if __name__ == "__main__":
    unittest.main()
