"""Column-major 2/4-bit ternary layouts for experimental CUDA projections."""
import torch
import triton
import triton.language as tl


@triton.jit
def _pack_export(W,Y,K:tl.constexpr,N:tl.constexpr,BITS:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    group:tl.constexpr=8//BITS
    n=i//(K//group);k=(i%(K//group))*group
    word=tl.load(W+(k//16)*N+n,n<N,other=0).to(tl.uint32)
    value=tl.full((BLOCK,),0,tl.uint32)
    for j in tl.static_range(group):
        code=(word>>(2*((k+j)%16)))&3
        if BITS==4:
            signed=(code&1).to(tl.int32)-(code>>1).to(tl.int32)
            code=signed.to(tl.uint32)&15
        value|=code<<(j*BITS)
    tl.store(Y+i,value,i<N*K//group)


def pack_export(mod,k,n,bits):
    y=torch.empty(n,k//(8//bits),device=mod.packed_t.device,dtype=torch.uint8)
    _pack_export[(triton.cdiv(y.numel(),256),)](mod.packed_t,y,k,n,bits,256)
    return y


def pack_dense(w,bits):
    """Synthetic fixtures only; packing is never included in dot timing."""
    n,k=w.shape;group=8//bits
    rows=w.to(torch.int32).reshape(n,k//group,group)
    code=rows&15 if bits==4 else torch.where(rows>0,1,torch.where(rows<0,2,0))
    packed=torch.zeros(n,k//group,device=w.device,dtype=torch.uint8)
    for j in range(group):packed|=(code[:,:,j]<<(j*bits)).to(torch.uint8)
    return packed
