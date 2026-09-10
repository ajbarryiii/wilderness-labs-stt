import tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import control

class Controls(unittest.TestCase):
    def metrics(self):
        return dict(fnr=.005,fpr=.1,cpu_p95_ms=1.,num_parameters=1000,precision_ok=True,
                    label_granularity='clip',speech_samples=1000,nonspeech_samples=1000,
                    data_kind='real_audio',training_cuda_verified=True,training_steps=10,score_range=1.,training_seconds=60.)
    def test_valid(self): control.validate_metrics(self.metrics())
    def test_invalid(self):
        for k,v in [('fnr',float('nan')),('fpr',-1),('cpu_p95_ms',float('inf')),('num_parameters',50001),('precision_ok',False),('speech_samples',0),('label_granularity','invented'),('data_kind','synthetic'),('training_cuda_verified',False),('training_steps',0)]:
            with self.subTest(key=k),self.assertRaises(ValueError): control.validate_metrics(dict(self.metrics(),**{k:v}))
    def test_joint_feasibility(self):
        self.assertTrue(control.feasible(self.metrics()))
        self.assertFalse(control.feasible(dict(self.metrics(),fpr=.21)))
        self.assertFalse(control.feasible(dict(self.metrics(),fnr=.02)))
    def test_reject_all_speech_even_without_best(self):
        allspeech=dict(self.metrics(),fnr=0.,fpr=1.)
        baseline=dict(self.metrics(),fnr=.015625,fpr=0.)
        self.assertFalse(control.improves(allspeech,None))
        self.assertFalse(control.improves(allspeech,baseline))
        self.assertTrue(control.improves(baseline,allspeech))
    def test_reject_all_noise_and_constant_scores(self):
        self.assertFalse(control.useful(dict(self.metrics(),fnr=1.,fpr=0.)))
        self.assertFalse(control.useful(dict(self.metrics(),score_range=0.)))
    def test_recall_first_within_joint_constraints(self):
        good=self.metrics(); bad=dict(good,fnr=.02,fpr=0.)
        self.assertFalse(control.improves(bad,good)); self.assertTrue(control.improves(good,bad))
    def test_gain(self):
        m=self.metrics()
        self.assertFalse(control.improves(dict(m,fpr=.099),m)); self.assertTrue(control.improves(dict(m,fpr=.09),m))
    def test_balanced_infeasible_progress(self):
        b=dict(self.metrics(),fnr=.03,fpr=.1); a=dict(b,fnr=.02,fpr=.15)
        self.assertTrue(control.improves(a,b))
        self.assertFalse(control.improves(dict(b,fnr=.0,fpr=.8),b))
    def test_no_implicit_start(self):
        with self.assertRaises(ValueError): control.start(False)
    def test_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d); (r/'workspace').mkdir(); p=r/'workspace/evaluate.py'; p.write_text('fixed'); control.save(r/'frozen.json',{'evaluate.py':control.digest(p)}); control.integrity(r); p.write_text('changed')
            with self.assertRaises(ValueError): control.integrity(r)
    def test_admission(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d); control.save(r/'state.json',dict(status='running',deadline_epoch=time.time()+100))
            with self.assertRaises(ValueError): control.trial(r,'baseline','must not launch')
    def test_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(TimeoutError): control.bounded([control.sys.executable,'-c','import time; time.sleep(30)'],d,.1,Path(d)/'log')
    def test_readiness_fingerprint_required(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d); control.save(r/'readiness.json',dict(status='ready',verified_epoch=time.time(),hashes={'a':'old'},preflight_training_seconds=60))
            with patch.object(control,'BASE',r),patch.object(control,'fingerprint',return_value={'a':'changed'}),self.assertRaises(ValueError): control.readiness()
    def test_readiness_staleness(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d); control.save(r/'readiness.json',dict(status='ready',verified_epoch=time.time()-90000,hashes={}))
            with patch.object(control,'BASE',r),self.assertRaises(ValueError): control.readiness()
    def test_preflight_rejects_midrun_edits(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d); runs=base/'runs'
            def snapshot(ws):
                (ws/'custom/vad').mkdir(parents=True)
                (ws/'custom/vad/train.py').write_text('before')
            def propose(run,*args):
                (run/'workspace/custom/vad/train.py').write_text('after')
                return {'label':'smoke','hypothesis':'changed recipe'}
            with patch.object(control,'BASE',base),patch.object(control,'RUNS',runs),patch.object(control,'check'),patch.object(control,'snapshot',side_effect=snapshot),patch.object(control,'freeze'),patch.object(control,'propose',side_effect=propose),patch.object(control,'trial',return_value={'status':'keep'}),patch.object(control,'write_report'),patch.object(control,'fingerprint',side_effect=[{'file':'old'},{'file':'changed'}]),self.assertRaisesRegex(ValueError,'changed during preflight'):
                control.preflight(60)
            self.assertFalse((base/'readiness.json').exists())
    def test_preflight_smoke_is_unranked(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d); runs=base/'runs'
            def snapshot(ws):
                (ws/'custom/vad').mkdir(parents=True)
                (ws/'custom/vad/train.py').write_text('before')
            def propose(run,*args):
                (run/'workspace/custom/vad/train.py').write_text('after')
                return {'label':'smoke','hypothesis':'changed recipe'}
            with patch.object(control,'BASE',base),patch.object(control,'RUNS',runs),patch.object(control,'check'),patch.object(control,'snapshot',side_effect=snapshot),patch.object(control,'freeze'),patch.object(control,'propose',side_effect=propose),patch.object(control,'trial',return_value={'status':'keep'}) as trials,patch.object(control,'write_report'),patch.object(control,'fingerprint',return_value={'file':'same'}):
                control.preflight(60)
                self.assertFalse(trials.call_args.kwargs['rank'])
            self.assertTrue((base/'readiness.json').exists())
if __name__=='__main__': unittest.main()
