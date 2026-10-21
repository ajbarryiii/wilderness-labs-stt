"""Exact intra-CTA integer K reduction, before any FP32 reconstruction."""
import triton
import triton.language as tl


@triton.jit
def _grouped_int8_dot(X,XS,W,WS,Bias,Y,P,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                     HAS_BIAS:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,
                     SPLIT:tl.constexpr,COMPONENTS:tl.constexpr,GROUPS:tl.constexpr,
                     ACTIVATION:tl.constexpr='',DENSE:tl.constexpr=True,COLUMN:tl.constexpr=True,
                     WEIGHT_BITS:tl.constexpr=8):
    tl.static_assert(DENSE and COLUMN and WEIGHT_BITS==8 and COMPONENTS==3)
    m=tl.program_id(0)*BM+tl.arange(0,BM)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    group=tl.arange(0,GROUPS)
    kk=tl.arange(0,BK)
    part=tl.program_id(2)
    a0=tl.zeros((GROUPS,BM,BN),tl.int32)
    a1=tl.zeros((GROUPS,BM,BN),tl.int32)
    a2=tl.zeros((GROUPS,BM,BN),tl.int32)
    for tile in range(tl.cdiv(K,BK*SPLIT*GROUPS)):
        k=((tile*GROUPS+group)*SPLIT+part)[:,None]*BK+kk[None,:]
        b=tl.load(W+n[None,None,:]*K+k[:,:,None],(n[None,None,:]<N)&(k[:,:,None]<K),other=0)
        a=tl.load(X+m[None,:,None]*K+k[:,None,:],(m[None,:,None]<M)&(k[:,None,:]<K),other=0)
        a0=tl.dot(a,b,a0,out_dtype=tl.int32)
        a=tl.load(X+M*K+m[None,:,None]*K+k[:,None,:],(m[None,:,None]<M)&(k[:,None,:]<K),other=0)
        a1=tl.dot(a,b,a1,out_dtype=tl.int32)
        a=tl.load(X+2*M*K+m[None,:,None]*K+k[:,None,:],(m[None,:,None]<M)&(k[:,None,:]<K),other=0)
        a2=tl.dot(a,b,a2,out_dtype=tl.int32)
    acc=tl.sum(a0,0);acc1=tl.sum(a1,0);acc2=tl.sum(a2,0)
    value=acc.to(tl.float32)*tl.load(XS+m,m<M,other=0)[:,None]
    value+=acc1.to(tl.float32)*tl.load(XS+M+m,m<M,other=0)[:,None]
    value+=acc2.to(tl.float32)*tl.load(XS+2*M+m,m<M,other=0)[:,None]
    if SPLIT==1:
        value*=tl.load(WS+n,n<N,other=0)[None,:]
        if HAS_BIAS:value+=tl.load(Bias+n,n<N,other=0)[None,:]
        if ACTIVATION=='silu':value=value/(1.+tl.exp(-value))
        tl.store(Y+m[:,None]*N+n[None,:],value,(m[:,None]<M)&(n[None,:]<N))
    else:tl.store(P+part*M*N+m[:,None]*N+n[None,:],value,(m[:,None]<M)&(n[None,:]<N))
