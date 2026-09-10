"""Local Whisper reference adapter; downloads only via explicit setup command."""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from core import ROOT, SR, digest, save

DEPS=ROOT/'finetune/stt/deps'
sys.path.insert(0,str(DEPS))


class Recognizer:
    def __init__(self,path,device='cpu'):
        os.environ['HF_HUB_OFFLINE']='1'
        os.environ['TRANSFORMERS_OFFLINE']='1'
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        path=Path(path)
        lock=json.loads((path/'lock.json').read_text())
        for name,h in lock['files'].items():
            if digest(path/name)!=h:
                raise ValueError('Pretrained snapshot checksum mismatch')
        torch.set_num_threads(1)
        self.device=device
        self.processor=WhisperProcessor.from_pretrained(path,local_files_only=True)
        self.model=WhisperForConditionalGeneration.from_pretrained(path,local_files_only=True,
                    use_safetensors=True).to(device).eval()

    def transcribe(self,pcm):
        output=[]
        for start in range(0,len(pcm),20*SR):
            inputs=self.processor(pcm[start:start+20*SR],sampling_rate=SR,return_tensors='pt',
                                  return_attention_mask=True)
            with torch.inference_mode():
                ids=self.model.generate(inputs.input_features.to(self.device),
                    attention_mask=inputs.attention_mask.to(self.device),
                    do_sample=False,num_beams=1,max_new_tokens=256)
            output.append(self.processor.batch_decode(ids,skip_special_tokens=True)[0])
        return ' '.join(output).strip()


def setup(out):
    from huggingface_hub import snapshot_download
    model='openai/whisper-tiny.en'
    revision='87c7102498dcde7456f24cfd30239ca606ed9063'
    out=Path(out)
    snapshot_download(model,revision=revision,local_dir=out,
        allow_patterns=['*.json','*.txt','*.safetensors','README.md'],
        ignore_patterns=['*training*'])
    files={str(p.relative_to(out)):digest(p) for p in out.iterdir() if p.is_file() and p.name!='lock.json'}
    save(out/'lock.json',dict(model=model,revision=revision,files=files,
        model_card='https://huggingface.co/openai/whisper-tiny.en',license='Apache-2.0',
        role='Pretrained FP32 engineering reference, not strict ternary recognizer'))
    print(json.dumps(dict(model=model,revision=revision,files=len(files))))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['setup','transcribe'])
    p.add_argument('--model',default=str(ROOT/'finetune/stt/models/whisper-tiny.en'))
    p.add_argument('--audio')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=p.parse_args()
    if args.command=='setup':
        setup(args.model)
    else:
        import soundfile as sf
        x,sr=sf.read(args.audio,dtype='float32',always_2d=True)
        if sr!=SR:
            raise ValueError('Expected 16kHz audio')
        print(Recognizer(args.model,args.device).transcribe(x.mean(axis=1)))
