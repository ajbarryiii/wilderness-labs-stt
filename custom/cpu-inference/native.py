"""Packed CPU inference through a small, cached C++ library (no Torch extension).

The scalar backend uses hardware POPCNT and row-major packed weights. AVX512
interleaves eight output channels, letting VPOPCNTDQ accumulate eight independent
dot products without horizontal reductions. Activation packing, FP32 scaling,
and optional bias are included in every ``linear`` call. These are inference-only
operations: they do not construct an autograd graph.

The avx512_opt backend also tiles input rows and uses a shape-dependent thread
policy; ``threads`` is its maximum worker count, not a minimum for every call.
"""
from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import threading
import weakref

import torch


_SOURCE = Path(__file__).with_name("kernels.cpp")
_CACHE = Path("/mnt/hd/wilderness-labs-stt/cpu-inference/cache/native")
_LIBRARY = None
_BUILD_METADATA = None
_BUILD_LOCK = threading.Lock()


def _load_library():
    global _LIBRARY, _BUILD_METADATA
    with _BUILD_LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        if platform.machine() not in ("x86_64", "AMD64"):
            raise RuntimeError("Packed CPU kernels require an x86-64 CPU with POPCNT")
        if not os.path.ismount("/mnt/hd"):
            raise RuntimeError("/mnt/hd is not mounted; refusing native cache writes")
        compiler = shlex.split(os.environ.get("CXX", "g++"))
        version = subprocess.run(compiler + ["--version"], text=True,
                                 capture_output=True, check=True).stdout
        flags = ["-O3", "-std=c++17", "-fPIC", "-shared", "-fopenmp",
                 "-ffp-contract=off", "-fno-tree-vectorize"]
        source_hash = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()
        build = {"source_sha256": source_hash, "compiler": compiler,
                 "compiler_version": version.strip(), "flags": flags,
                 "machine": platform.machine()}
        key = hashlib.sha256(json.dumps(build, sort_keys=True).encode()).hexdigest()[:24]
        _CACHE.mkdir(parents=True, exist_ok=True)
        library_path = _CACHE / f"packed-{key}.so"
        with (_CACHE / f"packed-{key}.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not library_path.exists():
                temporary = _CACHE / f"packed-{key}.{os.getpid()}.so.tmp"
                environment = os.environ.copy()
                environment["TMPDIR"] = str(_CACHE)
                command = compiler + flags + [str(_SOURCE), "-o", str(temporary)]
                try:
                    subprocess.run(command, text=True, capture_output=True,
                                   check=True, env=environment)
                    temporary.replace(library_path)
                except subprocess.CalledProcessError as error:
                    raise RuntimeError(f"CPU kernel compilation failed:\n{error.stderr}") from error
                finally:
                    temporary.unlink(missing_ok=True)
                (_CACHE / f"packed-{key}.json").write_text(json.dumps(build, indent=2) + "\n")
        lib = ctypes.CDLL(str(library_path))
        pointer = ctypes.c_void_p
        i64 = ctypes.c_int64
        integer = ctypes.c_int
        signatures = {
            "cpu_last_error": ([], ctypes.c_char_p),
            "cpu_feature_flags": ([], integer),
            "cpu_weight_create": ([pointer, pointer, i64, i64, integer, integer, integer, integer], pointer),
            "cpu_weight_destroy": ([pointer], None),
            "cpu_weight_storage_bytes": ([pointer], ctypes.c_uint64),
            "cpu_weight_scratch_bytes": ([pointer], ctypes.c_uint64),
            "cpu_weight_linear": ([pointer, pointer, i64, pointer, pointer], integer),
            "cpu_weight_embedding": ([pointer, pointer, i64, pointer], integer),
        }
        for name, (args, result) in signatures.items():
            function = getattr(lib, name)
            function.argtypes = args
            function.restype = result
        _BUILD_METADATA = dict(build, cache_key=key, library_path=str(library_path),
                               source_path=str(_SOURCE),
                               library_sha256=hashlib.sha256(library_path.read_bytes()).hexdigest())
        _LIBRARY = lib
        return lib


def cpu_features() -> dict:
    """Runtime CPUID checks, including OS support for AVX512 state."""
    flags = _load_library().cpu_feature_flags()
    return {"popcnt": bool(flags & 1), "avx512f_vpopcntdq": bool(flags & 2)}


def available_backends() -> tuple[str, ...]:
    features = cpu_features()
    return tuple(name for name, supported in (
        ("scalar", features["popcnt"]), ("avx512", features["avx512f_vpopcntdq"]),
        ("avx512_opt", features["avx512f_vpopcntdq"])) if supported)


def build_metadata() -> dict:
    _load_library()
    return dict(_BUILD_METADATA)


class PackedWeight:
    """Own packed codes and row scales, with reusable per-call activation scratch.

    ``codes`` is int8 CPU [N,K], containing +/-1 (W1) or -1/0/+1 (W2).
    ``scale`` is a scalar, [N], or [N,1]. ``activation_bits=1`` maps x>=0 to
    +1 and all other values to -1. ``activation_bits=2`` maps x>=0.5 to +1,
    x<=-0.5 to -1, and the rest to 0. There is no activation rescaling.
    Calls on the same object are serialized to protect reusable native scratch.
    """

    def __init__(self, codes: torch.Tensor, scale: torch.Tensor | float,
                 activation_bits: int = 1, backend: str = "avx512_opt", threads: int = 1,
                 weight_bits: int | None = None):
        if not isinstance(codes, torch.Tensor) or codes.dtype != torch.int8 or codes.device.type != "cpu":
            raise ValueError("codes must be a CPU int8 tensor")
        if codes.ndim != 2 or min(codes.shape) < 1:
            raise ValueError("codes must have nonempty shape [N,K]")
        if activation_bits not in (1, 2):
            raise ValueError("activation_bits must be 1 or 2")
        if backend not in ("scalar", "avx512", "avx512_opt"):
            raise ValueError("backend must be scalar, avx512, or avx512_opt")
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError("threads must be a positive integer")
        if weight_bits is None:
            weight_bits = 2 if bool((codes == 0).any()) else 1
        if weight_bits not in (1, 2):
            raise ValueError("weight_bits must be 1 or 2")
        self.n, self.k = codes.shape
        scale = torch.as_tensor(scale, device="cpu", dtype=torch.float32)
        if scale.numel() == 1:
            scale = scale.reshape(1).expand(self.n)
        elif scale.shape not in ((self.n,), (self.n, 1)):
            raise ValueError("scale must be scalar, [N], or [N,1]")
        scales = scale.detach().reshape(self.n).contiguous()
        if not bool(torch.isfinite(scales).all()):
            raise ValueError("scale must contain finite values")
        codes = codes.detach().contiguous()
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.backend = backend
        self.threads = threads
        self._lib = _load_library()
        self._lock = threading.Lock()
        handle = self._lib.cpu_weight_create(codes.data_ptr(), scales.data_ptr(),
                    self.n, self.k, weight_bits, activation_bits,
                    {"scalar": 0, "avx512": 1, "avx512_opt": 2}[backend], threads)
        if not handle:
            raise ValueError(self._lib.cpu_last_error().decode())
        self._handle = handle
        self._finalizer = weakref.finalize(self, self._lib.cpu_weight_destroy, handle)

    @property
    def storage_bytes(self) -> int:
        return self._lib.cpu_weight_storage_bytes(self._handle)

    @property
    def scratch_bytes(self) -> int:
        with self._lock:
            return self._lib.cpu_weight_scratch_bytes(self._handle)

    @property
    def metadata(self) -> dict:
        return dict(backend=self.backend, threads=self.threads,
                    weight_bits=self.weight_bits, activation_bits=self.activation_bits,
                    shape=[self.n, self.k], storage_bytes=self.storage_bytes,
                    scratch_bytes=self.scratch_bytes,
                    layout="output_blocks_8" if self.backend != "scalar" else "row_major",
                    thread_policy=(dict(kind="shape_specific", max_threads=self.threads,
                                        serial_single_row_word_limit=16384 if self.weight_bits == 2 else 49152,
                                        word_count="N * ceil(K / 64)", bulk_row_tile=8)
                                   if self.backend == "avx512_opt" else
                                   dict(kind="configured_limit", max_threads=self.threads)),
                    cpu_features=cpu_features(), build=build_metadata())

    def linear(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if x.device.type != "cpu" or x.dtype != torch.float32:
            raise ValueError("x must be a CPU float32 tensor")
        if x.ndim < 1 or x.shape[-1] != self.k:
            raise ValueError(f"x must have final dimension {self.k}")
        if bias is not None:
            if bias.device.type != "cpu" or bias.dtype != torch.float32 or bias.shape != (self.n,):
                raise ValueError("bias must be a CPU float32 tensor with shape [N]")
            bias = bias.contiguous()
        x = x.contiguous()
        out = torch.empty((*x.shape[:-1], self.n), dtype=torch.float32, device="cpu")
        m = math.prod(x.shape[:-1])
        with self._lock:
            status = self._lib.cpu_weight_linear(self._handle, x.data_ptr(), m,
                    bias.data_ptr() if bias is not None else None, out.data_ptr())
            if status:
                raise RuntimeError(self._lib.cpu_last_error().decode())
        return out

    def embedding(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.device.type != "cpu" or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("ids must be a CPU int32 or int64 tensor")
        ids = ids.to(dtype=torch.int64).contiguous()
        out = torch.empty((*ids.shape, self.k), dtype=torch.float32, device="cpu")
        status = self._lib.cpu_weight_embedding(self._handle, ids.data_ptr(), ids.numel(), out.data_ptr())
        if status:
            raise IndexError(self._lib.cpu_last_error().decode())
        return out

    def dequantize(self) -> torch.Tensor:
        return self.embedding(torch.arange(self.n, dtype=torch.int64))
