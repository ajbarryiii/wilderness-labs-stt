"""NVRTC compilation for agent C's headerless CUDA kernels (SM120).

Needs the parakeet-ternary `python` wrapper: it puts the project's NVRTC 12.9 wheel on the
library path (the parakeet-binary wrapper deliberately drops it). The environment setup is the
same as M1's inference.cuda_matmul.kernel: load libnvrtc globally and point CUDA_HOME at an
empty include directory on /mnt/hd, because the kernels use no CUDA headers.
"""
from __future__ import annotations

import ctypes
import functools
import hashlib
import os
import subprocess
from pathlib import Path

import torch

NVRTC_LIB = Path('/mnt/hd/wilderness-labs-stt/inference-efficiency/deps/nvidia/cuda_nvrtc/lib/libnvrtc.so.12')
ARTIFACTS = Path('/mnt/hd/wilderness-labs-stt/parakeet-binary/gpu1bit/C')  # plans/gpu-kernel.md A10


def _require_mount() -> None:
    if not os.path.ismount('/mnt/hd'):
        raise RuntimeError('/mnt/hd is not mounted; refusing artifact writes')


@functools.lru_cache(None)
def setup() -> None:
    _require_mount()
    if not NVRTC_LIB.is_file():
        raise RuntimeError(f'missing NVRTC at {NVRTC_LIB}')
    ctypes.CDLL(str(NVRTC_LIB), mode=ctypes.RTLD_GLOBAL)
    if not os.environ.get('CUDA_HOME'):
        home = ARTIFACTS / 'caches' / 'cuda-headerless'
        (home / 'include').mkdir(parents=True, exist_ok=True)
        os.environ['CUDA_HOME'] = str(home)
    from torch.utils import cpp_extension
    if cpp_extension.CUDA_HOME is None:
        cpp_extension.CUDA_HOME = os.environ['CUDA_HOME']


def compile_cubin(source: str, name: str, *, arch: str = '120a',
                  options: tuple[str, ...] = ('--std=c++17',)) -> bytes:
    """CUBIN bytes for one extern "C" kernel; raises RuntimeError with the NVRTC log on failure."""
    setup()
    # CPU-only when CUDA is not initialized (precompilation without the GPU lock).
    if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError('compile before CUDA graph capture')
    from torch.cuda._utils import _nvrtc_compile
    binary, symbol = _nvrtc_compile(source, name, arch, nvcc_options=list(options))
    if symbol != name:
        raise RuntimeError(f'expected extern "C" symbol {name!r}, got {symbol!r}')
    return binary


def disk_cubin(source: str, name: str, arch: str = '120a', options: tuple[str, ...] = ('--std=c++17',)) -> bytes:
    """CUBIN from the on-disk cache keyed by SHA-256 of (NVRTC library, source, name, arch, options)."""
    _require_mount()
    key = hashlib.sha256(repr((NVRTC_LIB.stat().st_size, source, name, arch, options)).encode()).hexdigest()
    path = ARTIFACTS / 'caches' / 'cubin' / f'{key}.cubin'
    if path.is_file():
        return path.read_bytes()
    binary = compile_cubin(source, name, arch=arch, options=options)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.tmp{os.getpid()}')
    tmp.write_bytes(binary)
    tmp.replace(path)
    return binary


@functools.lru_cache(None)
def _load_cached(source: str, name: str, arch: str, options: tuple[str, ...]):
    from torch.cuda._utils import _cuda_load_module
    setup()
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('load kernels before CUDA graph capture')
    binary = disk_cubin(source, name, arch, options)
    return _cuda_load_module(binary, [name])[name], binary


def load(source: str, name: str, *, arch: str = '120a', options: tuple[str, ...] = ('--std=c++17',)):
    """(kernel, cubin) compiled once per (source, name, arch, options) in this process."""
    return _load_cached(source, name, arch, tuple(options))


def tool(name: str) -> Path:
    """Triton's bundled CUDA binutils (cuobjdump, nvdisasm); the system has no CUDA toolkit."""
    import triton
    path = Path(triton.__file__).parent / 'backends' / 'nvidia' / 'bin' / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def sass(cubin: bytes) -> str:
    """SASS of a CUBIN via nvdisasm; the file goes to the artifacts tmp dir on /mnt/hd."""
    _require_mount()
    folder = ARTIFACTS / 'tmp'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (hashlib.sha256(cubin).hexdigest()[:16] + '.cubin')
    path.write_bytes(cubin)
    return subprocess.run([str(tool('nvdisasm')), '-c', str(path)], check=True,
                          capture_output=True, text=True).stdout
