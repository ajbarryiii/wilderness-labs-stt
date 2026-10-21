"""Run agent A's GPU tests in one process holding the GPU lock once (plans/gpu-kernel.md A9).

    timeout 600 gpu1bit/shared/python -m gpu1bit.triton_bf16.run_tests [unittest args...]
"""
import sys
import unittest

from gpu1bit.triton_bf16.gpulock import locked
from gpu1bit.shared import env


def main():
    with locked("A:tests"):
        program = unittest.main(module="gpu1bit.triton_bf16.test_layer", argv=[sys.argv[0], *sys.argv[1:]],
                                exit=False, verbosity=2)
        sys.exit(0 if program.result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
