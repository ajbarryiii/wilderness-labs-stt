"""Check fused rounding points, normalization, and changed-input graph replay."""
import types
import unittest
import itertools

import torch
import triton

from .fused_kernels import _quantize
from .kernels import _finish
from .layer_fusions4 import residual_norm
from .row_fusion import DeferredProjection,finish_residual


class RowFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import paths
        from .benchmark import energy_meter
        from .sprint4 import require_power_cap
        cls.lock=paths.gpu_lock('row-fusion checks');cls.lock.__enter__()
        cls.meter=energy_meter();cls.meter.__enter__();cls.meter._guard();require_power_cap(cls.meter)
        torch.manual_seed(907)

    @classmethod
    def tearDownClass(cls):
        cls.meter.__exit__(None,None,None);cls.lock.__exit__(None,None,None)

    @torch.inference_mode()
    def test_split_epilogue_norm_quant_and_replay(self):
        m,n=17,1024
        norm=torch.nn.LayerNorm(n).cuda();out_norm=torch.nn.LayerNorm(n).cuda()
        norm.weight.uniform_(-2,2);norm.bias.uniform_(-1,1)
        out_norm.weight.uniform_(-2,2);out_norm.bias.uniform_(-1,1)
        residual=torch.randn(1,m,n,device='cuda')
        scale=torch.rand(n,device='cuda')*.02;bias=torch.randn(n,device='cuda')*.01
        for split,bias_enabled,warps in itertools.product([1,4,8],[False,True],[4,8]):
            partial=torch.randn(split,m,n,device='cuda')
            proj=DeferredProjection(partial,scale,bias if bias_enabled else None,split,residual.shape)
            for factor,last in [(.5,False),(1.,False),(.5,True)]:
                provider=types.SimpleNamespace(prequantized={},null=torch.empty(0,device='cuda',dtype=torch.uint8),row_warps=warps)
                def run():return finish_residual(provider,proj,residual,factor,norm,out_norm if last else None)
                for _ in range(3):
                    run();provider.prequantized.clear()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):y=run()
                q,qs=provider.prequantized.pop((y.data_ptr(),m,n,id(norm)))
                for _ in range(2):
                    partial.copy_(torch.randn_like(partial));residual.copy_(torch.randn_like(residual))
                    reference=torch.empty_like(residual)
                    if split>1:
                        _finish[(triton.cdiv(m*n,256),)](partial,scale,bias if bias_enabled else None,reference,
                                                      m,n,m,m*n,n,1,bias_enabled,split,256)
                    else:reference.copy_(partial)
                    if last:reference=residual_norm(reference,residual,out_norm,factor)
                    else:reference=torch.add(residual,reference,alpha=factor)
                    rq=torch.empty_like(q);rs=torch.empty_like(qs)
                    _quantize[(m,)](reference,rq,rs,norm.weight,norm.bias,m,n,m,*reference.stride(),
                                   n,3,True,norm.eps,num_warps=4,enable_fp_fusion=False)
                    pressure=torch.empty(32*1024*1024,device='cuda',dtype=torch.float32)
                    pressure.fill_(17);del pressure
                    graph.replay();torch.cuda.synchronize()
                    if not last and split==1 and warps==4:
                        torch.testing.assert_close(y,reference,atol=0,rtol=0,
                            msg=f'split={split}, factor={factor}, out_norm={last}, bias={bias_enabled}')
                        torch.testing.assert_close(q,rq,atol=0,rtol=0)
                        torch.testing.assert_close(qs,rs,atol=0,rtol=0)
                    else:
                        # Fusion can change the FP32 split sum or LayerNorm
                        # reduction tree. Check reconstructed INT8x3 values;
                        # the candidate also requires full transcript checks.
                        torch.testing.assert_close(y,reference,atol=2e-6,rtol=2e-6)
                        actual=(q.float()*qs[:,:,None]).sum(0)
                        expected=(rq.float()*rs[:,:,None]).sum(0)
                        torch.testing.assert_close(actual,expected,atol=3e-6,rtol=2e-6)
                del graph


if __name__=='__main__':unittest.main()
