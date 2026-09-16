import unittest
import torch
from torch.nn import functional as F
from binary_stt.model import BinaryCTCModel,BinaryLinear,ternary_codes,ternary_activation_quantize
from binary_stt.train import quantization_at
from binary_stt.tests.test_model import tiny_config

class TernaryTests(unittest.TestCase):
 def test_three_codes_zero_and_scaling(self):
  w=torch.tensor([[-2.,-.1,0.,.1,2.]])
  torch.testing.assert_close(ternary_codes(w),torch.tensor([[-1.,0.,0.,0.,1.]]))
  torch.testing.assert_close(ternary_codes(w*20),ternary_codes(w))
  torch.testing.assert_close(ternary_codes(torch.zeros_like(w)),torch.zeros_like(w))
 def test_activation_codes_and_identity_gradients(self):
  x=torch.tensor([[-2.,-.1,0.,.1,2.]],requires_grad=True);t=torch.zeros(5,requires_grad=True)
  y=ternary_activation_quantize(x,t,1.)
  torch.testing.assert_close(y,torch.tensor([[-.84,0.,0.,0.,.84]]))
  y.sum().backward()
  torch.testing.assert_close(x.grad,torch.ones_like(x));torch.testing.assert_close(t.grad,-torch.ones_like(t))
  self.assertIs(ternary_activation_quantize(x,t,0.),x)
 def test_weight_codes_scales_and_ste(self):
  layer=BinaryLinear(5,2);layer.quantizer='ternary';layer.set_quantization(1,0)
  with torch.no_grad():layer.weight.copy_(torch.tensor([[-2.,-.1,0.,.1,2.]]).repeat(2,1))
  w=layer.effective_weight()
  torch.testing.assert_close(w/layer.output_scale[:,None],ternary_codes(layer.weight))
  coeff=torch.arange(10.).reshape(2,5);(w*coeff).sum().backward()
  torch.testing.assert_close(layer.weight.grad,coeff)
  torch.testing.assert_close(layer.log_scale.grad,(coeff*ternary_codes(layer.weight)).sum(1)*layer.output_scale)
 def test_ctc_backward_both_modes_and_checkpoint_reload(self):
  for fraction in [0.,1.]:
   torch.manual_seed(31);m=BinaryCTCModel(tiny_config(quantizer='ternary'));m.set_quantization(1,fraction)
   x=torch.randn(2,80,160,requires_grad=True);logits,length=m(x,torch.tensor([160,144]))
   loss=F.ctc_loss(logits.log_softmax(-1).transpose(0,1),torch.tensor([1,2,3,4]),length,torch.tensor([2,2]))
   loss.backward();self.assertTrue(torch.isfinite(loss));self.assertGreater(float(x.grad.abs().sum()),0)
   self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
   copy=BinaryCTCModel(tiny_config(quantizer='ternary'));copy.load_state_dict(m.state_dict());copy.set_quantization(1,fraction)
   m.eval();copy.eval();torch.testing.assert_close(m(x.detach(),torch.tensor([160,144]))[0],copy(x.detach(),torch.tensor([160,144]))[0])
 def test_counts_and_phase_labels(self):
  with torch.device('meta'):m=BinaryCTCModel({'quantizer':'ternary'})
  self.assertEqual(sum(p.numel() for p in m.parameters()),488270080)
  self.assertTrue(all(x.quantizer=='ternary' for x in m.modules() if isinstance(x,BinaryLinear)))
  self.assertEqual(quantization_at(0,{},'ternary')['phase'],'w:ternary/a:ternary')
