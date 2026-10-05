"""Decoder regression checks: state boundaries, recurrence, batching and graphs."""
import contextlib
import os
import unittest
import torch

from .decoder_kernels import _advance, linear, decoder_transform, control_transform


class DecoderKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import paths
        cls.lock=paths.gpu_lock('decoder regression tests');cls.lock.__enter__()
        torch.manual_seed(927)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False

    @classmethod
    def tearDownClass(cls):cls.lock.__exit__(None,None,None)

    @torch.inference_mode()
    def test_joint_projection(self):
        for b in [1,2,8]:
            mod=torch.nn.Linear(640,1030).cuda().eval()
            x=torch.randn(b,1,640,device='cuda');z=torch.randn_like(x)
            torch.testing.assert_close(linear(x,mod,z),mod(torch.relu(x+z)),atol=2e-6,rtol=2e-5)

    @torch.inference_mode()
    def test_advance_blank_zero_duration_ties_and_inactive(self):
        def scalar(value):return torch.tensor([value],device='cuda')
        for inner in [False,True]:
            for old_active in [False,True]:
                for old_advance in [False,True]:
                    for blank in [False,True]:
                        for duration_idx in [0,3]:
                            logits=torch.zeros(10,device='cuda')
                            logits[4 if blank else 1]=2
                            # Equal maxima must choose the first class/duration.
                            if not blank:logits[2]=2
                            logits[5+duration_idx]=3
                            labels=scalar(3);scores=scalar(.7);durations=scalar(2)
                            model_durations=torch.tensor([0,1,2,3,4],device='cuda')
                            time=scalar(8);safe=scalar(8);current=scalar(4);last=scalar(9);length=scalar(10)
                            active=scalar(old_active);prev=scalar(False);blank_mask=scalar(False)
                            advance=scalar(old_advance);any_mask=scalar(False)
                            _advance[(1,)](logits,labels,scores,durations,model_durations,time,safe,current,last,length,
                                            active,prev,blank_mask,advance,any_mask,4,5,inner,8,num_warps=4)
                            chosen=(4 if blank else 1) if not inner or old_advance else 3
                            score=2. if not inner or old_advance else .7
                            duration=max(1,duration_idx) if chosen==4 else duration_idx
                            move=old_advance if inner else old_active
                            t=8+(duration if move else 0)
                            stored_duration=duration if not inner or old_advance else 2
                            expected={'label':chosen,'score':score,'duration':stored_duration,'time':t,'safe':min(t,9),
                                      'current':8 if not inner or old_advance else 4,'active':t<10,
                                      'prev':old_active if not inner else False,'blank':chosen==4,
                                      'advance':t<10 and chosen==4}
                            self.assertEqual(labels.item(),expected['label'])
                            self.assertAlmostEqual(scores.item(),expected['score'],places=6)
                            for tensor,key in [(durations,'duration'),(time,'time'),(safe,'safe'),(current,'current'),
                                               (active,'active'),(prev,'prev'),(blank_mask,'blank'),(advance,'advance'),(any_mask,'advance')]:
                                self.assertEqual(tensor.item(),expected[key],(inner,old_active,old_advance,blank,duration_idx,key))

    @unittest.skipUnless(os.environ.get('PARAKEET_KERNEL_EXPORT'),'set local export')
    @torch.inference_mode()
    def test_model_tokens_states_timestamps_fallback_and_restore(self):
        import evaluate as ev
        from .runtime import load_packed
        from .optimized import enable_optimizations,disable_optimizations
        from nemo.utils import logging
        logging.setLevel(logging.ERROR)
        model=load_packed(os.environ['PARAKEET_KERNEL_EXPORT'])
        original_keys=set(model.state_dict())
        comp=model.decoding.decoding.decoding_computer
        with ev.inference_settings(model),ev.strict_fp32('cuda'):
            enable_optimizations(model,decoder=False,position_dot=False)
            comp.force_cuda_graphs_mode('full_graph')
            cases=[]
            for b,samples in [(1,16000),(1,48000),(2,32000)]:
                a=torch.randn(b,samples,device='cuda')*.025
                if samples==16000:a.zero_()
                lengths=torch.full((b,),samples,device='cuda',dtype=torch.long)
                if b>1:lengths[-1]=samples//2
                cases.append(model(input_signal=a,input_signal_length=lengths))
            def decode(e,l):
                out=model.decoding.rnnt_decoder_predictions_tensor(e,l,return_hypotheses=True)
                hyps=out[0] if isinstance(out,tuple) else out
                s=comp.state;h=s.batched_hyps
                lengths=h.current_lengths.cpu().tolist()
                return ([x.text for x in hyps],
                        [(h.transcript[i,:n].clone(),h.timestamps[i,:n].clone()) for i,n in enumerate(lengths)],
                        h.scores.clone(),tuple(t.clone() for t in s.decoder_state))
            refs=[decode(*case) for case in cases]
            original_predict=model.decoder.predict
            # Longer-lived decoder state can retain a batch-two capacity. Reset
            # before the optimized pass to exercise its batch-one specialization.
            with decoder_transform(model,joint=True,lstm=True,precompute=True,block=1),control_transform(model,storage=True):
                for batch in [1,2,8,9]:
                    tokens=torch.randint(0,1025,(batch,1),device='cuda')
                    tokens[-1]=1024
                    states=tuple(torch.randn(2,batch,640,device='cuda')*.1 for _ in range(2))
                    expected=original_predict(tokens,states,False)
                    actual=model.decoder.predict(tokens,states,False)
                    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-4)
                for case,ref in zip(cases,refs):
                    actual=decode(*case)
                    self.assertEqual(actual[0],ref[0])
                    torch.testing.assert_close(actual[1],ref[1],atol=0,rtol=0)
                    torch.testing.assert_close(actual[2:],ref[2:],atol=2e-4,rtol=2e-4)
                # Multi-token prediction must still use the original implementation.
                tokens=torch.tensor([[7,13,4]],device='cuda')
                result=model.decoder.predict(tokens,add_sos=True)
                self.assertEqual(result[0].shape,(1,4,640))
            restored=decode(*cases[-1]);self.assertEqual(restored[0],refs[-1][0])
            disable_optimizations(model)
        self.assertEqual(set(model.state_dict()),original_keys)
        self.assertNotIn('predict',model.decoder.__dict__)
        self.assertNotIn('_before_inner_loop_get_joint_output',comp.__dict__)

    @unittest.skipUnless(os.environ.get('PARAKEET_KERNEL_EXPORT'),'set local export')
    @torch.inference_mode()
    def test_two_models_replay_after_allocator_churn(self):
        import evaluate as ev
        import paths
        import json
        from .runtime import load_packed
        from .optimized import enable_optimizations, disable_optimizations
        from nemo.utils import logging
        logging.setLevel(logging.ERROR)
        with contextlib.ExitStack() as stack:
            stack.enter_context(ev.strict_fp32('cuda'))
            models=[load_packed(os.environ['PARAKEET_KERNEL_EXPORT']) for _ in range(2)]
            for model in models:
                stack.enter_context(ev.inference_settings(model))
                enable_optimizations(model,decoder=False,position_dot=False)
                stack.callback(disable_optimizations,model)
                model.decoding.decoding.decoding_computer.force_cuda_graphs_mode('full_graph')
            stack.enter_context(decoder_transform(models[1],joint=True,lstm=True,precompute=True,block=1))
            stack.enter_context(control_transform(models[1],storage=True))
            records=[json.loads(line) for line in (paths.MANIFESTS/'test_librispeech_clean.jsonl').read_text().splitlines()]
            refs=[]
            for duration in [3,30,10,3,30]:
                record=min(records,key=lambda r:abs(r['duration']-duration))
                a,l=ev.audio_batch([record]);e,el=models[0](input_signal=a.cuda(),input_signal_length=l.cuda())
                texts=[]
                for model in models:
                    out=model.decoding.rnnt_decoder_predictions_tensor(e,el,return_hypotheses=False)
                    if isinstance(out,tuple):out=out[0]
                    texts.append([h.text if hasattr(h,'text') else str(h) for h in out])
                    # Previously escaped child-body allocations could be reused
                    # by another graph or freed by empty_cache between replays.
                    junk=torch.empty(16*1024*1024,device='cuda');junk.fill_(17)
                    torch.cuda.synchronize();del junk;torch.cuda.empty_cache()
                self.assertEqual(texts[0],texts[1]);refs.append(texts[0])
            self.assertEqual(refs[0],refs[3])
            self.assertEqual(refs[1],refs[-1])

if __name__=='__main__':unittest.main()
