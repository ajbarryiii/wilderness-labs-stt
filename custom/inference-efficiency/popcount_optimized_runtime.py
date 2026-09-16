"""Reversible experimental dispatch for the optimized quantized model."""
import math

import torch
import torch.nn.functional as F

from kernels import popcount_optimized as opt
from kernels.packed import PackedWeight
from runtime_model import _Block, _Norm


FEATURES = {
    'counts': frozenset({'counts'}),
    'hybrid': frozenset({'counts', 'hybrid'}),
    'fusion': frozenset({'counts', 'hybrid', 'fusion'}),
    'all': frozenset({'counts', 'hybrid', 'fusion', 'packing'}),
    'selected': frozenset({'counts', 'hybrid', 'fusion'}),
}


def install(policy, *, a2=False, candidate='selected'):
    """Return a restore callable. Preparation/compilation precedes capture.

    Norms with <=4 input rows produce an internal packed-activation object;
    only this opt-in runtime understands that object. Larger encoder tensors
    retain their FP16 representation until quantization inside the GEMM tile.
    """
    features = FEATURES[candidate]
    counts = 'counts' in features
    fusion = 'fusion' in features
    packing = 'packing' in features
    original_linear, original_norm, original_block = PackedWeight.linear, _Norm.__call__, _Block.__call__
    for mode in ((0, 1, 2, 3) if fusion else (0,)):
        opt.dot_load(a2, counts, mode)
    opt.pack_load()
    weights = {}

    def prepared(p):
        if id(p) not in weights:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('Weight conversion must precede graph capture')
            k = torch.arange(p.k, device=p.words.device)
            encoded = (p.words[:, k//(32//p.bits)] >> ((k%(32//p.bits))*p.bits)) & ((1<<p.bits)-1)
            codes = (encoded*2-1 if p.bits == 1 else encoded-1).to(torch.int8)
            weights[id(p)] = (p, opt.BitWeight(codes, p.scales, p.bits))
        return weights[id(p)][1]

    def project(p, x, bias=None, out=None, *, mode=0, residual=None, cache=None, step=0):
        is_packed = isinstance(x, opt.Activation)
        shape = x.shape
        if not shape or shape[-1] != p.k:
            raise ValueError(f'Expected input [...,{p.k}]')
        if x.device != p.words.device or (not is_packed and (x.dtype != torch.float16 or not x.is_cuda)):
            raise ValueError('Expected FP16 CUDA input on the weight device')
        if a2 and p.bits != 2:
            raise ValueError('A2 requires ternary weights')
        m = math.prod(shape[:-1])
        out_shape = (*shape[:-1], p.n//3 if mode == 3 else p.n)
        if out is None:
            out = torch.empty(out_shape, device=x.device, dtype=torch.float16)
        elif (out.shape != out_shape or out.dtype != torch.float16 or out.device != x.device
              or not out.is_contiguous()):
            raise ValueError('Invalid contiguous output')
        if bias is not None and (bias.shape != (p.n,) or bias.dtype != torch.float16
                                 or bias.device != x.device or not bias.is_contiguous()):
            raise ValueError('Invalid contiguous FP16 bias')
        if not m:
            return out
        if ('hybrid' in features and m > 4
                and (candidate != 'selected' or p.bits == 2)):
            if is_packed or mode:
                raise ValueError('Hybrid encoder expects ordinary input and epilogue')
            return opt.quantized_gemm(p,x,bias,out,a2)
        activation = x if is_packed else opt.pack(x,a2)
        if activation.words.dtype != (torch.int64 if a2 else torch.int32):
            raise ValueError('Packed activation precision does not match this runtime')
        variant = policy.get(f'{p.bits},{m},{p.n},{p.k}', 'warp' if m<=4 else 'tile')
        prepared(p).linear(activation.words,out.reshape(m,-1),variant,bias,
            a2=a2,counts=counts,mode=mode,
            residual=residual.reshape(m,-1) if residual is not None else None,
            cache=cache,step=step)
        return out

    def norm(self, x):
        if math.prod(x.shape[:-1]) <= 4 and x.shape[-1] % 4 == 0 and x.shape[-1] <= 32768:
            return opt.pack(x,a2,norm=self)
        return original_norm(self,x)

    def attention(attn, x, residual, *, memory_kv=None, cache=None, step=None):
        if attn.cross:
            q = attn.split_heads(attn.q.linear(x,attn.q_bias))
            k,v = memory_kv
        elif cache is not None:
            if step is None or x.shape[1] != 1:
                raise ValueError('Cached attention requires one sequential token')
            q = attn.split_heads(project(attn.qkv.packed,x,attn.qkv_bias,mode=3,cache=cache,step=step))
            k,v = cache[0][:,:,:step+1,:],cache[1][:,:,:step+1,:]
        else:
            q,k,v = map(attn.split_heads,attn.qkv.linear(x,attn.qkv_bias).chunk(3,-1))
        y = F.scaled_dot_product_attention(q,k,v,dropout_p=0,is_causal=False)
        y = y.transpose(1,2).reshape(x.shape[0],x.shape[1],attn.width)
        return project(attn.out.packed,y,attn.out_bias,mode=1,residual=residual)

    def block(self,x,*,memory_kv=None,cache=None,step=None):
        if math.prod(x.shape[:-1]) > 4:
            return original_block(self,x,memory_kv=memory_kv,cache=cache,step=step)
        if fusion:
            x = attention(self.attn,self.attn_ln(x),x,cache=cache,step=step)
            if self.cross_attn is not None:
                x = attention(self.cross_attn,self.cross_attn_ln(x),x,memory_kv=memory_kv)
        else:
            x = x+self.attn(self.attn_ln(x),cache=cache,step=step)
            if self.cross_attn is not None:
                x = x+self.cross_attn(self.cross_attn_ln(x),memory_kv=memory_kv)
        normalized = self.mlp_ln(x)
        if packing:
            # Fuses GELU with the downstream activation pack, preserving both
            # the linear-output and GELU-output FP16 rounding points.
            y = opt.pack(self.mlp_up.linear(normalized,self.mlp_up_bias),a2,gelu=True)
        elif fusion:
            y = project(self.mlp_up.packed,normalized,self.mlp_up_bias,mode=2)
        else:
            y = F.gelu(self.mlp_up.linear(normalized,self.mlp_up_bias))
        if fusion:
            return project(self.mlp_down.packed,y,self.mlp_down_bias,mode=1,residual=x)
        return x+self.mlp_down.linear(y,self.mlp_down_bias)

    PackedWeight.linear = project
    if packing:
        _Norm.__call__ = norm
    if fusion or packing:
        _Block.__call__ = block

    def restore():
        PackedWeight.linear, _Norm.__call__, _Block.__call__ = original_linear,original_norm,original_block
        weights.clear()

    return restore
