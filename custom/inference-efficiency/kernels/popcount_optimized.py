"""Opt-in popcount optimizations; importing this module changes no dispatch.

W1A1/W2A1 retain sign(x), W2A2 retains thresholds +/-0.5, including
intermediate FP16 rounding. Large batches use quantized FP16 Tensor Core tiles.
"""
import functools
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from . import popcount, popcount_ternary_activation


def dot_source(a2=False, counts=True, mode=0):
    source = popcount_ternary_activation.SOURCE if a2 else popcount.SOURCE
    extra = ('const int* Count, const half* Residual, half* CK, half* CV, '
             'int Capacity, int Step, int HeadDim, ')
    source = source.replace('int M, int N, int KARG', extra + 'int M, int N, int KARG')
    source = source.replace('int M,int N,int K', extra + 'int M,int N,int K')
    source = source.replace('Y,M,N,K,KW,has_bias',
                            'Y,Count,Residual,CK,CV,Capacity,Step,HeadDim,M,N,K,KW,has_bias')
    if counts and not a2:
        source = source.replace('return __popc(nz) - 2 * __popc((a ^ s) & nz);',
                                'return -2 * __popc((a ^ s) & nz);')
        source = source.replace('(float)sum *', '(float)(sum + (TERNARY ? Count[n] : K)) *')
    old = 'Y[(long long)m*N+n] = f2h(value);'
    assert source.count(old) == 2
    epilogue = 'Y[(long long)m*N+n] = f2h(value);'
    if mode == 1:
        epilogue = 'Y[(long long)m*N+n] = f2h(h2f(f2h(value)) + h2f(Residual[(long long)m*N+n]));'
    elif mode == 2:
        epilogue = '''value = h2f(f2h(value));
        value = value * (0.5f * (1.0f + erff(value * 0.70710678118654752440f)));
        Y[(long long)m*N+n] = f2h(value);'''
    elif mode == 3:
        epilogue = '''int width = N / 3;
        if (n < width) Y[(long long)m*width+n] = f2h(value);
        else {
            int offset = n % width;
            long long idx = (long long)m*width*Capacity +
                (long long)(offset/HeadDim)*Capacity*HeadDim + Step*HeadDim + offset%HeadDim;
            if (n < 2*width) CK[idx] = f2h(value); else CV[idx] = f2h(value);
        }'''
    elif mode != 0:
        raise ValueError(mode)
    return source.replace(old, epilogue)


@functools.lru_cache(None)
def dot_load(a2=False, counts=True, mode=0):
    popcount._preload_nvrtc()
    from torch.cuda._utils import _nvrtc_compile, _cuda_load_module
    names = [f'{alphabet}_{layout}{suffix}'
             for alphabet in (('ternary',) if a2 else ('binary', 'ternary'))
             for layout in ('warp', 'tile') for suffix in ('', '_k1024', '_k4096')]
    cubin, _ = _nvrtc_compile(dot_source(a2, counts, mode), names[0], compute_capability='120')
    return _cuda_load_module(cubin, names)


class BitWeight(popcount.BitWeight):
    def __init__(self, codes, scales, bits):
        super().__init__(codes, scales, bits)
        self.count = (codes != 0).sum(1, dtype=torch.int32).contiguous()

    def linear(self, a, out, variant='warp', bias=None, *, a2=False, counts=True,
               mode=0, residual=None, cache=None, step=0):
        if a2 and self.bits != 2:
            raise ValueError('A2 is supported only with ternary weights')
        m = a.shape[0]
        tile = variant == 'tile'
        if variant not in ('warp', 'tile'):
            raise ValueError(variant)
        sign, nz = (self.sign_t, self.nonzero_t) if tile else (self.sign, self.nonzero)
        if mode == 1 and (residual is None or residual.shape != out.shape or not residual.is_contiguous()):
            raise ValueError('Residual must match the contiguous output')
        if mode == 3:
            if cache is None or not (0 <= step < cache[0].shape[2]):
                raise ValueError('QKV requires an in-bounds cache step')
            if self.n % 3 or out.numel() != m*self.n//3:
                raise ValueError('QKV output must have N/3 columns')
            for c in cache:
                if (not c.is_contiguous() or c.dtype != torch.float16 or c.device != out.device
                        or c.shape[0] != m or c.shape[1]*c.shape[3] != self.n//3
                        or c.shape != cache[0].shape):
                    raise ValueError('Invalid contiguous QKV cache')
        ck, cv = cache if cache is not None else (out, out)
        capacity, hd = (ck.shape[2], ck.shape[3]) if cache is not None else (0, 0)
        name = ('binary' if self.bits == 1 else 'ternary') + '_' + variant
        if self.k in (1024, 4096):
            name += f'_k{self.k}'
        dot_load(a2, counts, mode)[name](
            grid=((self.n+(31 if tile else 3))//(32 if tile else 4), (m+3)//4 if tile else m, 1),
            block=(128, 1, 1), args=[a, sign, nz, self.scales,
                bias if bias is not None else out, out, self.count,
                residual if residual is not None else out, ck, cv, capacity, step, hd,
                m, self.n, self.k, a.shape[1], int(bias is not None)])
        return out


PACK_SOURCE = popcount.SOURCE[:popcount.SOURCE.index('extern "C"')] + r'''
struct Stats { float mean, m2, count; };
__device__ Stats update(Stats s, float x) {
    float delta = x - s.mean;
    float count = s.count + 1.0f;
    float mean = s.mean + delta * (1.0f/count);
    return {mean, s.m2 + delta*(x-mean), count};
}
__device__ Stats merge(Stats b, Stats a) {
    float delta = b.mean-a.mean, count = a.count+b.count;
    float inv = 1.0f/count, wa = a.count*inv, wb = b.count*inv;
    return {count == 0 ? 0 : wa*a.mean+wb*b.mean,
            count == 0 ? 0 : a.m2+b.m2+delta*delta*a.count*wb, count};
}
template<int A2> __device__ void write_word(void* A, int idx, bool positive, bool nonzero) {
    u32 sign = __ballot_sync(0xffffffffu, positive);
    if (A2) {
        u32 nz = __ballot_sync(0xffffffffu, nonzero);
        if ((threadIdx.x&31)==0) ((unsigned long long*)A)[idx] = ((unsigned long long)nz<<32)|sign;
    } else if ((threadIdx.x&31)==0) ((u32*)A)[idx] = sign;
}
template<int A2, int GELU> __device__ void pack_impl(const half* X, void* A, int M, int K) {
    int kw = (K+31)/32;
    int word = (blockIdx.x*blockDim.x+threadIdx.x)>>5;
    int m = word/kw, k = (word%kw)*32+(threadIdx.x&31);
    float v = m<M && k<K ? h2f(X[(long long)m*K+k]) : 0;
    if (GELU) v = h2f(f2h(v*(0.5f*(1.0f+erff(v*0.70710678118654752440f)))));
    // All lanes execute the ballots, including the masked last word.
    if (m<M) write_word<A2>(A, word, k<K && v>=0, k<K && (v>=.5f || v<=-.5f));
}
template<int A2> __device__ void norm_impl(const half* X, const half* G, const half* B,
                                          void* A, int K) {
    // Same vector-of-four Welford order as ATen's FP16 vectorized LayerNorm.
    // 128 threads, four warps. The FP16 rounding precedes quantization.
    int row = blockIdx.x, t = threadIdx.x, lane = t&31, warp = t>>5;
    Stats s = {0,0,0};
    for (int v=t; v<K/4; v+=128)
        for (int j=0; j<4; ++j) s = update(s, h2f(X[(long long)row*K+v*4+j]));
    for (int off=16; off; off>>=1) {
        Stats other = {__shfl_down_sync(0xffffffffu,s.mean,off),
                       __shfl_down_sync(0xffffffffu,s.m2,off),
                       __shfl_down_sync(0xffffffffu,s.count,off)};
        s = merge(s, other);
    }
    __shared__ float shared[6];
    for (int off=2; off; off>>=1) {
        if (warp>=off && warp<2*off && lane==0) {
            int w=warp-off; shared[3*w]=s.mean; shared[3*w+1]=s.m2; shared[3*w+2]=s.count;
        }
        __syncthreads();
        if (warp<off && lane==0) s = merge(s, {shared[3*warp],shared[3*warp+1],shared[3*warp+2]});
        __syncthreads();
    }
    if (t==0) { shared[0]=s.mean; shared[1]=rsqrtf(s.m2/(float)K+1.e-5f); }
    __syncthreads();
    int kw=(K+31)/32;
    for (int base=0; base<K; base+=128) {
        int k=base+t;
        float v=0;
        if (k<K) v=h2f(f2h(h2f(G[k])*(shared[1]*(h2f(X[(long long)row*K+k])-shared[0]))+h2f(B[k])));
        if ((k>>5)<kw) write_word<A2>(A,row*kw+(k>>5),k<K && v>=0,k<K && (v>=.5f || v<=-.5f));
    }
}
#define PACK(NAME,A2,GELU) extern "C" __global__ void NAME(const half* X,void* A,int M,int K) { pack_impl<A2,GELU>(X,A,M,K); }
#define NORM(NAME,A2) extern "C" __global__ void NAME(const half* X,const half* G,const half* B,void* A,int K) { norm_impl<A2>(X,G,B,A,K); }
PACK(pack_a1,0,0)
PACK(pack_a2,1,0)
PACK(gelu_a1,0,1)
PACK(gelu_a2,1,1)
NORM(norm_a1,0)
NORM(norm_a2,1)
'''


@functools.lru_cache(None)
def pack_load():
    popcount._preload_nvrtc()
    from torch.cuda._utils import _nvrtc_compile, _cuda_load_module
    names = [f'{op}_a{a}' for op in ('pack', 'gelu', 'norm') for a in (1, 2)]
    cubin, _ = _nvrtc_compile(PACK_SOURCE, names[0], compute_capability='120')
    return _cuda_load_module(cubin, names)


@dataclass
class Activation:
    words: torch.Tensor
    shape: tuple

    @property
    def device(self):
        return self.words.device


def pack(x, a2=False, gelu=False, norm=None):
    if x.dtype != torch.float16 or not x.is_cuda or x.ndim < 1 or x.shape[-1] < 1:
        raise ValueError('Packing requires nonempty-width FP16 CUDA input')
    shape = tuple(x.shape)
    k = shape[-1]
    x = x.reshape(-1, k).contiguous()
    m = x.shape[0]
    a = torch.empty((m, (k+31)//32), dtype=torch.int64 if a2 else torch.int32, device=x.device)
    if norm is not None:
        if (k % 4 or k > 32768
                or any(t.data_ptr() % 8 for t in (x,norm.weight,norm.bias))):
            # Native LayerNorm uses a different reduction for unaligned inputs.
            y = torch.nn.functional.layer_norm(x,(k,),norm.weight,norm.bias,1e-5)
            return pack(y.reshape(shape), a2=a2)
        pack_load()[f'norm_a{2 if a2 else 1}'](grid=(m,1,1), block=(128,1,1),
            args=[x,norm.weight,norm.bias,a,k])
    else:
        pack_load()[f'{"gelu" if gelu else "pack"}_a{2 if a2 else 1}'](
            grid=((m*((k+31)//32)+7)//8,1,1), block=(256,1,1),args=[x,a,m,k])
    return Activation(a, shape)


@triton.jit
def _quantized_gemm(X, WT, S, Bias, Y, M: tl.constexpr, N: tl.constexpr,
                    K: tl.constexpr, KW: tl.constexpr, BITS: tl.constexpr,
                    A2: tl.constexpr, HAS_BIAS: tl.constexpr, BM: tl.constexpr,
                    BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    nm, nn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    group = pid // (8*nn)
    first = group*8
    size = tl.minimum(nm-first, 8)
    pm = first + (pid % (8*nn)) % size
    pn = (pid % (8*nn)) // size
    m = pm*BM+tl.arange(0,BM)
    n = pn*BN+tl.arange(0,BN)
    k = tl.arange(0,BK)
    wk = tl.arange(0,BK//(32//BITS))
    lane = tl.arange(0,32//BITS)
    accum = tl.zeros((BM,BN),tl.float32)
    for start in range(tl.cdiv(K,BK)):
        kk = start*BK+k
        a = tl.load(X+m[:,None]*K+kk[None,:], (m[:,None]<M)&(kk[None,:]<K),other=0)
        if A2:
            a = tl.where(a>=.5,1,tl.where(a<=-.5,-1,0)).to(tl.float16)
        else:
            a = tl.where(a>=0,1,-1).to(tl.float16)
        # Padded input values must remain zero after sign quantization.
        a = tl.where(kk[None,:]<K,a,0).to(tl.float16)
        wkk = start*(BK//(32//BITS))+wk
        words = tl.load(WT+wkk[:,None]*N+n[None,:],(wkk[:,None]<KW)&(n[None,:]<N),other=0)
        encoded = ((words[:,None,:].to(tl.uint32)>>(lane[None,:,None]*BITS))&((1<<BITS)-1)).reshape((BK,BN))
        if BITS == 1:
            b = (encoded.to(tl.float32)*2-1).to(tl.float16)
        else:
            b = (encoded.to(tl.float32)-1).to(tl.float16)
        accum = tl.dot(a,b,accum)
    scale = tl.load(S+n,n<N,other=0).to(tl.float16).to(tl.float32)
    accum *= scale[None,:]
    if HAS_BIAS:
        accum += tl.load(Bias+n,n<N,other=0)[None,:].to(tl.float32)
    tl.store(Y+m[:,None]*N+n[None,:],accum.to(tl.float16),(m[:,None]<M)&(n[None,:]<N))


def quantized_gemm(weight, x, bias=None, out=None, a2=False):
    shape = (*x.shape[:-1],weight.n)
    x = x.reshape(-1,weight.k).contiguous()
    m = x.shape[0]
    if out is None:
        out = torch.empty(shape, dtype=x.dtype,device=x.device)
    bm = 16 if m<=32 else 64
    _quantized_gemm[(triton.cdiv(m,bm)*triton.cdiv(weight.n,64),)](
        x,weight.words_t,weight.scales,bias if bias is not None else out,out,
        m,weight.n,weight.k,weight.words.shape[1],weight.bits,a2,bias is not None,
        bm,64,64,num_warps=4,num_stages=3)
    return out
