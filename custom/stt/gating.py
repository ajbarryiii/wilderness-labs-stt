"""Causal window adapter for the existing clip classifier; streaming use is experimental."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from core import ROOT, SR

_spec = importlib.util.spec_from_file_location('frozen_vad', ROOT/'custom/vad/prepare.py')
vad = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vad)


class Gate:
    def __init__(self, candidate, window_seconds=1.0, hop_seconds=0.2,
                 preroll_seconds=0.5, hangover_seconds=0.8):
        self.config, weights, _ = vad.validate_model(candidate)
        self.weights = [torch.from_numpy(w) for w in weights]
        self.threshold = json.loads((Path(candidate)/'metrics.json').read_text())['threshold']
        self.window = round(window_seconds*SR)
        self.hop = round(hop_seconds*SR)
        self.preroll = round(preroll_seconds*SR)
        self.hangover = round(hangover_seconds*SR)
        if min(self.window,self.hop,self.preroll,self.hangover) <= 0:
            raise ValueError('Positive gate timings required')

    def intervals(self, audio):
        decisions=[]
        # Each decision only sees already arrived samples. Include the partial final hop.
        for end in list(range(self.hop,len(audio),self.hop))+[len(audio)]:
            x=vad.logmel(audio[max(0,end-self.window):end])
            with torch.inference_mode():
                score=float(vad.forward_reference(torch.from_numpy(x[None]),
                    torch.tensor([x.shape[-1]]),self.weights,self.config)[0])
            decisions.append((end,score >= self.threshold))
        return intervals_from_decisions(decisions,len(audio),self.preroll,self.hangover), decisions


def intervals_from_decisions(decisions, length, preroll, hangover):
    intervals=[]
    start=last_positive=None
    for end, positive in decisions:
        if not 0 <= end <= length:
            raise ValueError('Decision outside recording')
        if positive:
            if start is None:
                start=max(0,end-preroll)
            last_positive=end
        elif start is not None and end-last_positive >= hangover:
            intervals.append((start,end))
            start=last_positive=None
    if start is not None:
        intervals.append((start,length))
    merged=[]
    for a,b in intervals:
        if merged and a <= merged[-1][1]:
            merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
        else:
            merged.append((a,b))
    return merged
