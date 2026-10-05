"""Check expanded projection arithmetic and candidate Conformer fusions."""
import unittest
import os
import torch
import torch.nn.functional as F
from export import pack_codes
from .runtime import PackedLinear
from .encoder4 import ExpandedProjection
from .layer_fusions4 import glu_pad,residual_norm


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class EncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import paths
        from .benchmark import energy_meter
        from .sprint4 import require_power_cap
        cls.lock=paths.gpu_lock('expanded encoder tests');cls.lock.__enter__()
        cls.meter=energy_meter();meter=cls.meter.__enter__();meter._guard();require_power_cap(meter)
        torch.manual_seed(904);torch.backends.cuda.matmul.allow_tf32=False

    @classmethod
    def tearDownClass(cls):
        cls.meter.__exit__(None,None,None);cls.lock.__exit__(None,None,None)

    @torch.inference_mode()
    def test_expanded_projection(self):
        for n,k,m in [(97,129,7),(1024,1024,38),(1024,4096,126),(4096,1024,376)]:
            codes=torch.randint(-1,2,(n,k),dtype=torch.int8);codes[0].zero_()
            codes[1].fill_(1);codes[2].fill_(-1)
            layer=PackedLinear(pack_codes(codes),torch.rand(n)*.02,k,torch.randn(n)*.01).cuda()
            w=codes.cuda().float()*layer.scale[:,None]
            x=torch.randn(1,m,k,device='cuda');x[:,0].zero_()
            for backend in ['triton_col','triton_col_tuned']:
                project=ExpandedProjection(backend)
                for activated in [False,True]:
                    y=project(x,layer,input_activation='silu' if activated else '')
                    ref=F.linear(F.silu(x) if activated else x,w,layer.bias)
                    torch.testing.assert_close(y,ref,atol=1e-5,rtol=3e-5)
                y=project(x.transpose(1,2),layer,conv=True).transpose(1,2)
                torch.testing.assert_close(y,F.linear(x,w,layer.bias),atol=1e-5,rtol=3e-5)
                self.assertEqual(project.bytes,n*k)

    @torch.inference_mode()
    def test_gate_mask_padding(self):
        for t in [1,17,126]:
            for strided in [False,True]:
                x=torch.randn(2,t,130,device='cuda').transpose(1,2) if strided else torch.randn(2,130,t,device='cuda')
                mask=torch.arange(t,device='cuda')[None,:]>=torch.tensor([t,max(0,t//2)],device='cuda')[:,None]
                for used_mask in [None,mask]:
                    for left,right in [(4,4),(8,0)]:
                        ref=F.glu(x,dim=1)
                        if used_mask is not None:ref=ref.masked_fill(used_mask[:,None,:],0.)
                        torch.testing.assert_close(glu_pad(x,used_mask,left,right),F.pad(ref,(left,right)),atol=1e-6,rtol=2e-6)

    @torch.inference_mode()
    def test_residual_normalization(self):
        for n in [129,1024]:
            norm=torch.nn.LayerNorm(n).cuda();norm.weight.uniform_(-2,2);norm.bias.uniform_(-1,1)
            x=torch.randn(2,37,n,device='cuda');r=torch.randn_like(x)
            torch.testing.assert_close(residual_norm(x,r,norm,.5),norm(r+x*.5),atol=2e-6,rtol=2e-5)

    @unittest.skipUnless(os.environ.get('PARAKEET_KERNEL_EXPORT'),'set local export')
    @torch.inference_mode()
    def test_graph_cache_churn_and_restore(self):
        import evaluate as ev
        from .optimized import enable_optimizations,disable_optimizations
        from .runtime import load_packed
        from nemo.utils import logging
        logging.setLevel(logging.ERROR)
        model=load_packed(os.environ['PARAKEET_KERNEL_EXPORT'])
        keys=set(model.state_dict());layer=model.encoder.layers[0]
        before=(layer.forward,layer.conv.forward,layer.conv.depthwise_conv.weight.clone())
        inputs=[]
        for b,samples in [(1,16000),(2,48000),(1,160000)]:
            a=torch.randn(b,samples,device='cuda')*.02;a[0].zero_()
            l=torch.full((b,),samples,device='cuda',dtype=torch.long)
            if b>1:l[-1]=samples//3
            inputs.append((a,l))
        with ev.inference_settings(model),ev.strict_fp32('cuda'):
            enable_optimizations(model,max_graphs=1)
            refs=[tuple(v.clone() for v in model(input_signal=a,input_signal_length=l)) for a,l in inputs]
            disable_optimizations(model)
            enable_optimizations(model,max_graphs=1,encoder_storage='expanded')
            for i in [0,1,2,0,2]:
                a,l=inputs[i]
                y=model(input_signal=a,input_signal_length=l)
                torch.testing.assert_close(y,refs[i],atol=3e-6,rtol=3e-5)
                junk=torch.empty(16*1024*1024,device='cuda');junk.fill_(19)
                torch.cuda.synchronize();del junk;torch.cuda.empty_cache()
            disable_optimizations(model)
        self.assertEqual(set(model.state_dict()),keys)
        self.assertEqual(layer.forward,before[0]);self.assertEqual(layer.conv.forward,before[1])
        torch.testing.assert_close(layer.conv.depthwise_conv.weight,before[2],atol=0,rtol=0)


if __name__=='__main__':unittest.main()
