"""Focused verification of labels, split isolation, thresholding and safe ternary export."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"vad"))
import prepare
import train

class VadTests(unittest.TestCase):
    def test_all_speech_is_not_good(self):
        metrics=prepare.binary_metrics([1,1,1,1],[0,1,0,1],1)
        self.assertEqual(metrics["fpr"],1)
        self.assertEqual(prepare.auroc([1,1,1,1],[0,1,0,1]),0.5)
        with self.assertRaisesRegex(ValueError,"Degenerate"):
            prepare.discrimination_gate([1,1,1,1],[0,1,0,1],metrics)
    def test_allspeech_rejected_even_if_score_is_variable(self):
        scores=[0,2,1,3];labels=[0,1,0,1]
        metrics=prepare.binary_metrics(scores,labels,-1)
        with self.assertRaisesRegex(ValueError,"Degenerate"):
            prepare.discrimination_gate(scores,labels,metrics)
    def test_threshold_uses_only_calibration_positives(self):
        labels=np.r_[np.ones(100),np.zeros(100)]
        threshold=prepare.choose_threshold(np.r_[np.arange(100),np.full(100,1e6)],labels)
        self.assertEqual(threshold,1)
        self.assertEqual(prepare.binary_metrics(np.arange(100).tolist()+[0],[1]*100+[0],threshold)["fnr"],0.01)
    def test_split_leakage(self):
        row={"path":"/real.wav","offset_samples":0,"num_samples":16000,"label":1,"speaker_id":"speaker","source_id":"s","split":"train"}
        other={**row,"path":"/other.wav","split":"calibration"}
        with self.assertRaisesRegex(ValueError,"leakage"):
            prepare.manifest_clips({"data_kind":"real_audio","label_granularity":"clip","clips":[row,other]})
    def test_no_synthetic_scoring(self):
        with self.assertRaisesRegex(ValueError,"real audio"):
            prepare.manifest_clips({"data_kind":"synthetic","label_granularity":"clip","clips":[]})
    def test_ternary_export_reference_and_tamper(self):
        model=train.Classifier().eval()
        x=torch.randn(2,16,25);lengths=torch.tensor([25,17])
        with tempfile.TemporaryDirectory() as tmp:
            train.export(model,Path(tmp),{"training_device":"cuda:0","training_steps":1})
            config,weights,parameters=prepare.validate_model(tmp)
            self.assertLessEqual(parameters,50000)
            expected=model(x,lengths)
            actual=prepare.forward_reference(x,lengths,[torch.from_numpy(w) for w in weights],config)
            torch.testing.assert_close(expected,actual,atol=2e-6,rtol=2e-6)
            arrays=dict(np.load(Path(tmp)/"weights.npz",allow_pickle=False))
            arrays["head_codes"][0,0]=2
            np.savez(Path(tmp)/"weights.npz",**arrays)
            with self.assertRaisesRegex(ValueError,"ternary"):
                prepare.validate_model(tmp)
    def test_ste_has_gradient(self):
        weights=torch.tensor([-0.1,0.0,0.2],requires_grad=True)
        actual,codes,scale=train.ternary(weights)
        self.assertTrue(set(codes.tolist())<={-1,0,1})
        actual.sum().backward()
        torch.testing.assert_close(weights.grad,torch.ones_like(weights))
    def test_padding_does_not_change_output(self):
        model=train.Classifier().eval()
        x=torch.randn(1,16,25);lengths=torch.tensor([25])
        original=model(x,lengths)
        padded=torch.nn.functional.pad(x,(0,10))
        torch.testing.assert_close(original,model(padded,lengths),atol=2e-6,rtol=2e-6)

if __name__=="__main__":unittest.main()
