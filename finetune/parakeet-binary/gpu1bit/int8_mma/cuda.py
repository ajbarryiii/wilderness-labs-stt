"""NVRTC compilation, a cubin disk cache and a typed cuLaunchKernel launcher for agent B (SM120).

Needs NVRTC 12.9 on the library path: run under gpu1bit/shared/python (the parakeet-ternary
environment; the parakeet-binary wrapper drops NVRTC). The kernels use no CUDA headers, so the
environment setup is M1's inference.cuda_matmul.kernel recipe: load libnvrtc globally and point
CUDA_HOME at an empty include directory on /mnt/hd.

Compiled cubins are cached on /mnt/hd keyed by SHA-256 of (source, kernel name, arch, options,
NVRTC version); compilation is refused during CUDA-graph capture. `Kernel.launch` packs every
argument with an explicit C type (torch's _CudaKernel turns Python floats into doubles) and
enqueues on torch.cuda.current_stream() unless a stream is given.
"""
from __future__ import annotations

import ctypes
import functools
import hashlib
import os
from pathlib import Path

import torch

NVRTC_LIB = Path("/mnt/hd/wilderness-labs-stt/inference-efficiency/deps/nvidia/cuda_nvrtc/lib/libnvrtc.so.12")
ARTIFACTS = Path("/mnt/hd/wilderness-labs-stt/parakeet-binary/gpu1bit/B")  # plans/gpu-kernel.md A10
OPTIONS = ("--std=c++17", "--fmad=false")


def require_mount() -> None:
    if not os.path.ismount("/mnt/hd"):
        raise RuntimeError("/mnt/hd is not mounted; refusing artifact writes")


@functools.lru_cache(None)
def setup() -> str:
    """Make NVRTC loadable for torch.cuda._utils; return its version string."""
    require_mount()
    if not NVRTC_LIB.is_file():
        raise RuntimeError(f"missing NVRTC at {NVRTC_LIB}; use gpu1bit/shared/python")
    lib = ctypes.CDLL(str(NVRTC_LIB), mode=ctypes.RTLD_GLOBAL)
    if not os.environ.get("CUDA_HOME"):
        home = ARTIFACTS / "caches" / "cuda-headerless"
        (home / "include").mkdir(parents=True, exist_ok=True)
        os.environ["CUDA_HOME"] = str(home)
    from torch.utils import cpp_extension
    if cpp_extension.CUDA_HOME is None:
        cpp_extension.CUDA_HOME = os.environ["CUDA_HOME"]
    major, minor = ctypes.c_int(), ctypes.c_int()
    lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
    return f"{major.value}.{minor.value}"


def compile_cubin(source: str, name: str, arch: str = "120a", options: tuple[str, ...] = OPTIONS) -> bytes:
    version = setup()
    key = hashlib.sha256("\0".join([source, name, arch, *options, version]).encode()).hexdigest()
    path = ARTIFACTS / "caches" / "cubin" / f"{key}.cubin"
    if path.is_file():
        return path.read_bytes()
    if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("compile kernels before CUDA graph capture (warm up first)")
    from torch.cuda._utils import _nvrtc_compile
    binary, symbol = _nvrtc_compile(source, name, arch, nvcc_options=list(options))
    if symbol != name:
        raise RuntimeError(f'expected extern "C" symbol {name!r}, got {symbol!r}')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_bytes(binary)
    os.replace(tmp, path)
    return binary


class Kernel:
    """One loaded kernel; `launch` is CUDA-graph capturable (no sync, no allocation)."""

    def __init__(self, cubin: bytes, name: str, shared_bytes: int):
        from torch.cuda._utils import _cuda_load_module, _get_gpu_runtime_library
        self.cubin = cubin
        self.name = name
        self.kernel = _cuda_load_module(cubin, [name])[name]
        self.lib = _get_gpu_runtime_library()
        self.shared_bytes = shared_bytes
        if shared_bytes >= 48 * 1024:
            self.kernel.set_shared_memory_config(shared_bytes)

    def launch(self, grid, block, args, stream=None):
        """args: sequence of (ctype, value); tensors pass as (ctypes.c_void_p, t.data_ptr())."""
        holders = [ctype(value) for ctype, value in args]
        array = (ctypes.c_void_p * len(holders))(*[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders])
        stream = torch.cuda.current_stream() if stream is None else stream
        result = self.lib.cuLaunchKernel(self.kernel.func, grid[0], grid[1], grid[2], block[0], block[1], block[2],
                                         self.shared_bytes, stream._as_parameter_, array, None)
        if result != 0:
            raise RuntimeError(f"cuLaunchKernel({self.name}) failed with CUresult {result}")


_LOADED: dict[tuple, Kernel] = {}


def load(source: str, name: str, shared_bytes: int) -> Kernel:
    key = (hashlib.sha256(source.encode()).hexdigest(), name, shared_bytes)
    kernel = _LOADED.get(key)
    if kernel is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("load kernels before CUDA graph capture (warm up first)")
        kernel = _LOADED[key] = Kernel(compile_cubin(source, name), name, shared_bytes)
    return kernel


def ptr(t: torch.Tensor | None) -> tuple:
    return (ctypes.c_void_p, 0 if t is None else t.data_ptr())


def i32(v: int) -> tuple:
    return (ctypes.c_int, int(v))


def i64(v: int) -> tuple:
    return (ctypes.c_longlong, int(v))


def tool(name: str) -> Path:
    """Triton's bundled CUDA binutils (nvdisasm, cuobjdump); there is no system CUDA toolkit."""
    import triton
    path = Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def sass(cubin: bytes) -> str:
    import subprocess
    require_mount()
    folder = ARTIFACTS / "tmp"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (hashlib.sha256(cubin).hexdigest()[:16] + ".cubin")
    path.write_bytes(cubin)
    return subprocess.run([str(tool("nvdisasm")), "-c", "-hex", str(path)], check=True,
                          capture_output=True, text=True).stdout
