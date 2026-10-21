"""Experiment-only direct cuBLAS INT8 GEMM with three stacked residuals.

This bypasses torch._int_mm's CUTLASS dispatch. It uses the installed cuBLAS
library and an explicit stream/workspace; the public runtime remains Triton.
"""
import ctypes as ct
import functools
from pathlib import Path

import torch


def checked(status):
    if status:
        raise RuntimeError(f'cuBLAS status {status}')


@functools.lru_cache(None)
def handle(device):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('warm cuBLAS before graph capture')
    # torch has already loaded the pinned CUDA libraries. Locate that exact
    # mapped library rather than downloading or selecting another installation.
    libs={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
          if '/libcublas.so.' in line}
    if len(libs)!=1:
        raise RuntimeError(f'expected one loaded cuBLAS library, got {len(libs)}')
    lib=ct.CDLL(libs.pop())
    ptr=ct.c_void_p
    lib.cublasCreate_v2.argtypes=[ct.POINTER(ptr)]
    lib.cublasSetStream_v2.argtypes=[ptr,ptr]
    lib.cublasSetWorkspace_v2.argtypes=[ptr,ptr,ct.c_size_t]
    lib.cublasGemmEx.argtypes=[ptr,ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.c_int,
         ptr,ptr,ct.c_int,ct.c_int,ptr,ct.c_int,ct.c_int,ptr,ptr,ct.c_int,ct.c_int,ct.c_int,ct.c_int]
    h=ptr();checked(lib.cublasCreate_v2(ct.byref(h)))
    workspace=torch.empty(8*1024*1024,dtype=torch.uint8,device=device)
    return lib,h,workspace


def dot(q,qs,weight,scale,bias,y,*,algorithm=99,activation=''):
    import triton
    from .encoder4 import _combine
    components,m,k=q.shape;n=scale.numel()
    assert components==3 and weight.stride()==(1,k)
    lib,h,workspace=handle(q.device)
    checked(lib.cublasSetStream_v2(h,ct.c_void_p(torch.cuda.current_stream(q.device).cuda_stream)))
    checked(lib.cublasSetWorkspace_v2(h,ct.c_void_p(workspace.data_ptr()),workspace.numel()))
    accum=torch.empty((3,m,n),device=q.device,dtype=torch.int32)
    alpha,beta=ct.c_int(1),ct.c_int(0)
    # Column-major output [N,3M] equals the desired row-major [3M,N].
    # Weight storage is [K,N] column major and transposed in the operation;
    # row-major Q is interpreted as [K,3M] column major.
    # CUDA_R_8I=3, CUDA_R_32I=10, CUBLAS_COMPUTE_32I=72.
    checked(lib.cublasGemmEx(h,1,0,n,3*m,k,ct.byref(alpha),ct.c_void_p(weight.data_ptr()),3,k,
        ct.c_void_p(q.data_ptr()),3,k,ct.byref(beta),ct.c_void_p(accum.data_ptr()),10,n,72,algorithm))
    _combine[(triton.cdiv(m*n,256),)](accum,qs,scale,bias,y,m,n,3,bias is not None,activation,256,enable_fp_fusion=False)
