"""Check the reconstructed CUDA adapter's ABI, fusions, and graph ownership."""
import unittest

import torch

from export import pack_codes
from .encoder4 import ExpandedProjection
from .ptx_cuda import PtxCudaProjection
from .runtime import PackedLinear


class PtxCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import paths
        from .benchmark import energy_meter
        from .sprint4 import require_power_cap
        cls.lock=paths.gpu_lock('reconstructed CUDA adapter checks');cls.lock.__enter__()
        cls.meter=energy_meter();cls.meter.__enter__();cls.meter._guard();require_power_cap(cls.meter)
        torch.manual_seed(863)
        torch.backends.cuda.matmul.allow_tf32=False

    @classmethod
    def tearDownClass(cls):
        cls.meter.__exit__(None,None,None);cls.lock.__exit__(None,None,None)

    def fixture(self,bias=True):
        codes=torch.randint(-1,2,(40,128),dtype=torch.int8)
        codes[0]=0;codes[1]=1;codes[2]=-1
        mod=PackedLinear(pack_codes(codes),torch.rand(40)*.02,128,
                         torch.randn(40)*.01 if bias else None).cuda()
        x=torch.randn(2,17,128,device='cuda');x[:,0]=0
        return x,mod

    @torch.inference_mode()
    def test_none_bias_abi_normalization_silu_and_conv_views(self):
        norm=torch.nn.LayerNorm(128).cuda()
        norm.weight.uniform_(-2,2);norm.bias.uniform_(-1,1)
        original=ExpandedProjection('triton_col_tuned');native=PtxCudaProjection()
        for bias in [False,True]:
            x,mod=self.fixture(bias)
            for options in [{},{'activation':'silu'},{'norm':norm},{'input_activation':'silu'}]:
                torch.testing.assert_close(native(x,mod,**options),original(x,mod,**options),atol=0,rtol=0)
            torch.testing.assert_close(native(x.transpose(1,2),mod,conv=True,norm=norm),
                                       original(x.transpose(1,2),mod,conv=True,norm=norm),atol=0,rtol=0)

    @torch.inference_mode()
    def test_changed_input_graph_replay_with_allocator_churn(self):
        original=ExpandedProjection('triton_col_tuned');native=PtxCudaProjection()
        x,mod=self.fixture(False)
        for _ in range(3):native(x,mod)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):y=native(x,mod)
        for _ in range(3):
            x.copy_(torch.randn_like(x))
            expected=original(x,mod)
            pressure=torch.empty(32*1024*1024,device='cuda',dtype=torch.float32)
            pressure.fill_(23);del pressure
            graph.replay();torch.cuda.synchronize()
            torch.testing.assert_close(y,expected,atol=0,rtol=0)

    @torch.inference_mode()
    def test_compressed_layouts_and_graphs_with_fusions(self):
        original=ExpandedProjection('triton_col_tuned')
        norm=torch.nn.LayerNorm(128).cuda()
        for bits in [2,4]:
            native=PtxCudaProjection(weight_bits=bits)
            for bias in [False,True]:
                x,mod=self.fixture(bias)
                for options in [{},{'activation':'silu'},{'norm':norm},{'input_activation':'silu'}]:
                    torch.testing.assert_close(native(x,mod,**options),original(x,mod,**options),atol=0,rtol=0)
                for _ in range(3):native(x,mod,norm=norm)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):y=native(x,mod,norm=norm)
                x.copy_(torch.randn_like(x))
                expected=original(x,mod,norm=norm)
                pressure=torch.empty(32*1024*1024,device='cuda',dtype=torch.float32)
                pressure.fill_(19);del pressure
                graph.replay();torch.cuda.synchronize()
                torch.testing.assert_close(y,expected,atol=0,rtol=0)
                del graph
            self.assertEqual(native.bytes,2*40*128*bits//8)

    @torch.inference_mode()
    def test_native_integer_reduction_and_changed_input_graph(self):
        original=ExpandedProjection('triton_col_tuned')
        native=PtxCudaProjection(tuning='native')
        norm=torch.nn.LayerNorm(1024).cuda()
        for n,bias in [(1024,False),(2048,True)]:
            native.prewarm(54,n,1024,bias)
            warmed=len(native.kernels)
            codes=torch.randint(-1,2,(n,1024),dtype=torch.int8)
            codes[0]=0;codes[1]=1;codes[2]=-1
            mod=PackedLinear(pack_codes(codes),torch.rand(n)*.02,1024,
                             torch.randn(n)*.01 if bias else None).cuda()
            x=torch.randn(1,54,1024,device='cuda');x[:,0]=0
            for _ in range(3):native(x,mod,norm=norm)
            self.assertEqual(len(native.kernels),warmed)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):y=native(x,mod,norm=norm)
            for _ in range(3):
                x.copy_(torch.randn_like(x));expected=original(x,mod,norm=norm)
                pressure=torch.empty(32*1024*1024,device='cuda',dtype=torch.float32)
                pressure.fill_(29);del pressure
                graph.replay();torch.cuda.synchronize()
                torch.testing.assert_close(y,expected,atol=2e-6,rtol=2e-5)
            del graph


if __name__=='__main__':unittest.main()
