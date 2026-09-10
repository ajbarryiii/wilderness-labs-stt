"""Transcribe a local 16kHz mono/stereo recording with a safe custom export."""
import argparse
import soundfile as sf
import torch
from core import load_model
from evaluate import recognize

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--candidate',required=True)
    p.add_argument('--audio',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=p.parse_args()
    torch.set_num_threads(1)
    pcm,sr=sf.read(args.audio,dtype='float32',always_2d=True)
    if sr!=16000:
        p.error('Expected 16kHz audio; resample explicitly before transcription')
    print(recognize(load_model(args.candidate,args.device),pcm.mean(axis=1),args.device))
