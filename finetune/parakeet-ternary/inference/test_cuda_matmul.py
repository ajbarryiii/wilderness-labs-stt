"""Hardware checks for CUDA/PTX fragment layout, edge masks, and graph lifetime."""
import unittest

import torch
import triton

from .cuda_matmul import CONFIGS, dot, kernel
from .kernels import _finish


class CudaMatmulTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import paths
        from .benchmark import energy_meter
        from .sprint4 import require_power_cap
        cls.lock=paths.gpu_lock('CUDA matmul regression checks');cls.lock.__enter__()
        cls.meter=energy_meter();cls.meter.__enter__();cls.meter._guard();require_power_cap(cls.meter)
        torch.manual_seed(1943)
        torch.backends.cuda.matmul.allow_tf32=False

    @classmethod
    def tearDownClass(cls):
        cls.meter.__exit__(None,None,None);cls.lock.__exit__(None,None,None)

    def case(self,name,m,n,k,bias=True,activation=''):
        q=torch.randint(-127,128,(3,m,k),device='cuda',dtype=torch.int8)
        scales=torch.rand(3,m,device='cuda')*.01
        w=torch.randint(-1,2,(n,k),device='cuda',dtype=torch.int8)
        w[0]=0;w[1]=1;w[2]=-1
        ws=torch.rand(n,device='cuda')*.01
        b=torch.randn(n,device='cuda')*.01 if bias else None
        y=torch.empty(m,n,device='cuda')
        split=CONFIGS[name][5]
        p=torch.empty(split,m,n,device='cuda')
        def run():
            dot(q,scales,w.T,ws,b,y,p,name=name,activation=activation)
            if split>1:
                _finish[(triton.cdiv(m*n,256),)](p,ws,b,y,m,n,m,m*n,n,1,b is not None,split,256,ACTIVATION=activation)
        reference=sum((q[i].float() @ w.float().T)*scales[i,:,None] for i in range(3))*ws
        if b is not None:reference=reference+b
        if activation:reference=reference*torch.sigmoid(reference)
        run();torch.cuda.synchronize()
        torch.testing.assert_close(y,reference,atol=2e-6,rtol=2e-5)
        return run,y,reference,(q,scales,w,ws,b,p)

    @torch.inference_mode()
    def test_fragment_layout_and_masked_edges(self):
        for name in CONFIGS:
            if name!='cublas':kernel(name)
            # Both M and N cross partial warp/block tile boundaries.
            self.case(name,17,40,128)
            self.case(name,1,8,128,bias=False)

    @torch.inference_mode()
    def test_bias_activation_and_long_reduction(self):
        for name in ['direct_split4','async_split8','pipeline_m16','cublas']:
            self.case(name,37,72,4096,bias=False,activation='silu')

    @torch.inference_mode()
    def test_graph_replay_after_allocator_churn(self):
        for name in ['direct_split4','async_split8','pipeline_m16','cublas']:
            run,y,reference,owned=self.case(name,51,64,1024)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):run()
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):run()
            for _ in range(3):
                pressure=torch.empty(32*1024*1024,device='cuda',dtype=torch.float32)
                pressure.fill_(31);del pressure
                graph.replay();torch.cuda.synchronize()
                torch.testing.assert_close(y,reference,atol=2e-6,rtol=2e-5)
            del graph,owned


if __name__=='__main__':unittest.main()
