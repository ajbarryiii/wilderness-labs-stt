import copy
import json
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from core import AcousticModel, ALPHABET, decode, encode, export, features, load_model, score_pairs, validate_recipe
from gating import intervals_from_decisions
from evaluate import gating_quality_passes
from control import bounded, improves, valid_metrics, verify

RECIPE=json.loads(Path(__file__).with_name('recipe.json').read_text()) | {'width':32,'depth':2}


class Contracts(unittest.TestCase):
    def test_ctc_repeats_and_blanks(self):
        a=ALPHABET.index('a')
        self.assertEqual(decode([0,a,a,0,a]),'aa')
        self.assertEqual(decode([0,0]),'')
        with self.assertRaises(ValueError):
            encode('dose 10')

    def test_missing_speech_and_noise_both_count(self):
        s=score_pairs([('give oxygen',''),('','invented words')])
        self.assertEqual(s['wer'],2)
        self.assertEqual(s['nonspeech_inserted_words'],2)
        self.assertEqual(score_pairs([('no 10','no 20')])['wer'],0.5)

    def test_fixed_frontend_causal_prefix(self):
        rng=np.random.default_rng(1)
        x=rng.normal(size=16000).astype(np.float32)
        short=features(x[:8000])
        np.testing.assert_array_equal(short,features(x)[:,:short.shape[-1]])

    def test_model_causal_prefix(self):
        torch.set_num_threads(1)
        model=AcousticModel(RECIPE).eval()
        x=torch.randn(1,80,101)
        with torch.no_grad():
            short=model(x[:,:,:60])
            long=model(x)
        torch.testing.assert_close(short,long[:,:short.shape[1]],atol=1e-5,rtol=1e-5)
        self.assertEqual(long.shape,(1,51,len(ALPHABET)))

    def test_safe_export_preserves_inference(self):
        model=AcousticModel(RECIPE).eval()
        x=torch.randn(1,80,100)
        with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
            export(model,tmp)
            loaded=load_model(tmp)
            torch.testing.assert_close(model(x),loaded(x),rtol=1e-4,atol=1e-4)
            path=Path(tmp)/'weights.2bit'
            path.write_bytes(b'bad')
            with self.assertRaises(ValueError):
                load_model(tmp)

    def test_recipe_rejects_unbounded_or_nan(self):
        for key,value in [('width',100000),('depth',2.5),('learning_rate',float('nan')),('batch_size',True)]:
            with self.assertRaises(ValueError):
                validate_recipe(RECIPE | {key:value})

    def test_preroll_hangover_and_flush(self):
        self.assertEqual(intervals_from_decisions([(20,False),(40,True),(60,False),(80,False)],100,30,40),[(10,80)])
        self.assertEqual(intervals_from_decisions([(20,False),(40,True)],50,30,40),[(10,50)])
        self.assertEqual(intervals_from_decisions([(20,False),(40,False)],50,30,40),[])

    def test_overlapping_preroll_never_duplicates_audio(self):
        intervals=intervals_from_decisions([(20,True),(40,False),(60,True),(80,False)],100,40,20)
        self.assertEqual(intervals,[(0,80)])

    def test_ranking_rewards_word_error_reduction(self):
        def metric(wer):
            s=dict(wer=wer,cer=0.1,wall_seconds=1,asr_audio_fraction=1,reference_words=100)
            return dict(predictions=[dict(reference='a',hypothesis=x) for x in ['a','b','c']],backend='custom',split='development',summary={a:dict(s) for a in ['always_on','gate_008','gate_010']})
        self.assertTrue(improves(metric(0.1),metric(0.2)))
        blank=metric(1.0)
        for arm in blank['summary'].values():
            arm['cer']=1.0
        self.assertFalse(improves(blank,None))
        m=metric(0.1);m['summary']['gate_008']['wer']=float('nan')
        with self.assertRaises(ValueError):
            valid_metrics(m)

    def test_frozen_dependency_change_is_rejected(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp, patch('control.ROOT',Path(tmp)):
            path=Path(tmp)/'evaluator.py'
            path.write_text('original')
            hashes={'evaluator.py':hashlib.sha256(path.read_bytes()).hexdigest()}
            verify(hashes)
            path.write_text('changed')
            with self.assertRaises(ValueError):
                verify(hashes)

    def test_dropping_audio_is_not_a_successful_gate(self):
        baseline={'by_condition':{c:{'wer':0.1,'cer':0.05} for c in ('clean','mixed_10db')}}
        candidate=copy.deepcopy(baseline) | {'utterance_audio_seconds':100,'utterance_audio_omitted_seconds':0}
        self.assertTrue(gating_quality_passes(candidate,baseline))
        candidate['utterance_audio_omitted_seconds']=80
        self.assertFalse(gating_quality_passes(candidate,baseline))

    def test_timeout_kills_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=time.monotonic()
            with self.assertRaises(subprocess.TimeoutExpired):
                bounded([sys.executable,'-c','import time; time.sleep(20)'],tmp,0.1,Path(tmp)/'out.log')
            self.assertLess(time.monotonic()-t,3)


if __name__=='__main__':
    unittest.main()
