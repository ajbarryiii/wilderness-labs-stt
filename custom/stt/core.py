"""Frozen acoustic graph, safe ternary export and literal character CTC scoring."""
import hashlib
import json
import os
import re
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(os.environ.get('WILDERNESS_STT_ROOT', Path(__file__).resolve().parents[2]))
ALPHABET = "_abcdefghijklmnopqrstuvwxyz' "
SR = 16000


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def normalize(text):
    # Keep numerals in scoring; the initial training corpus spells numbers out.
    return ' '.join(re.sub(r"[^a-z0-9' ]", ' ', text.lower()).split())


def encode(text):
    text = normalize(text)
    if not text or any(c not in ALPHABET[1:] for c in text):
        raise ValueError('Empty or unsupported training transcript: ' + text)
    return [ALPHABET.index(c) for c in text]


def decode(ids):
    out, last = [], None
    for token in ids:
        if token != last and token != 0:
            out.append(ALPHABET[token])
        last = token
    return ' '.join(''.join(out).split())


def edit_distance(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        row = [i]
        for j, b in enumerate(hyp, 1):
            row.append(min(row[-1] + 1, prev[j] + 1, prev[j-1] + (a != b)))
        prev = row
    return prev[-1]


def score_pairs(pairs):
    words = chars = word_errors = char_errors = noise_words = 0
    for ref, hyp in pairs:
        r, h = normalize(ref), normalize(hyp)
        words += len(r.split())
        chars += len(r)
        word_errors += edit_distance(r.split(), h.split())
        char_errors += edit_distance(r, h)
        if not r:
            noise_words += len(h.split())
    if not words or not chars:
        raise ValueError('No reference speech to score')
    return dict(wer=word_errors/words, cer=char_errors/chars,
                word_errors=word_errors, reference_words=words,
                char_errors=char_errors, reference_chars=chars,
                nonspeech_inserted_words=noise_words)


def features(audio):
    """80 fixed log-mel bins; causal 25ms/10ms analysis, no utterance normalization."""
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.pad(audio, (240, max(0, 160-len(audio))))
    frames = np.lib.stride_tricks.sliding_window_view(audio, 400)[::160]
    window = np.hanning(400).astype(np.float32)
    power = abs(np.fft.rfft(frames * window, n=512)) ** 2 / (window**2).sum()
    hz = 700 * (10**(np.linspace(0, 2595*np.log10(1+8000/700), 82)/2595)-1)
    f = np.fft.rfftfreq(512, 1/SR)
    bank = np.stack([np.maximum(0, np.minimum((f-hz[i])/(hz[i+1]-hz[i]),
                       (hz[i+2]-f)/(hz[i+2]-hz[i+1]))) for i in range(80)])
    return (np.log(np.maximum(power @ bank.T, 1e-8)).clip(-18, 6)/6+1).T.astype(np.float32)


def validate_recipe(r):
    ranges = dict(width=(32,384), depth=(2,16), kernel=(3,9),
                  learning_rate=(1e-5,0.01), batch_size=(1,32), weight_decay=(0,0.2),
                  grad_clip=(0.1,20), noise_probability=(0,0.75), ema_decay=(0,0.9999))
    if set(r) != set(ranges):
        raise ValueError('Unexpected/missing recipe fields')
    for k, (lo, hi) in ranges.items():
        if type(r[k]) not in (int,float) or not np.isfinite(r[k]) or not lo <= r[k] <= hi:
            raise ValueError('Invalid recipe field: ' + k)
    for k in ('width','depth','kernel','batch_size'):
        if type(r[k]) is not int:
            raise ValueError(k + ' must be integer')
    return r


class TernaryConv(nn.Module):
    def __init__(self, cin, cout, kernel, dilation=1, stride=1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(cout, cin, kernel))
        nn.init.kaiming_normal_(self.weight)
        self.dilation, self.stride = dilation, stride
        self.reference = False

    def forward(self, x):
        w = self.weight
        if not self.reference:
            scale = w.detach().abs().mean().clamp_min(1e-8)
            q = (w/scale).round().clamp(-1,1)*scale
            w = w + (q-w).detach()
        return F.conv1d(F.pad(x, (self.dilation*(w.shape[-1]-1),0)), w,
                        stride=self.stride, dilation=self.dilation)


class AcousticModel(nn.Module):
    def __init__(self, recipe):
        super().__init__()
        self.recipe = validate_recipe(recipe)
        width = recipe['width']
        self.front = TernaryConv(80, width, 5, stride=2)
        self.blocks = nn.ModuleList([TernaryConv(width,width,recipe['kernel'],2**(i%5))
                                     for i in range(recipe['depth'])])
        self.head = TernaryConv(width,len(ALPHABET),1)

    @staticmethod
    def norm(x):
        return x * torch.rsqrt(x.square().mean(dim=1, keepdim=True)+1e-5)

    def forward(self, x):
        x = F.silu(self.norm(self.front(x)))
        for layer in self.blocks:
            x = (x + F.silu(self.norm(layer(x)))) * (2**-0.5)
        return self.head(x).transpose(1,2)


def export(model, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    arrays, packed = {}, []
    for name, p in model.named_parameters():
        w = p.detach().float().cpu().numpy()
        scale = np.asarray(max(float(abs(w).mean()),1e-8), dtype=np.float32)
        codes = np.clip(np.round(w/scale),-1,1).astype(np.int8)
        arrays[name+'.codes'], arrays[name+'.scale'] = codes, scale
        u = np.pad((codes.ravel()+1).astype(np.uint8),(0,(-codes.size)%4),constant_values=1)
        packed.append((u[::4] | u[1::4]<<2 | u[2::4]<<4 | u[3::4]<<6).tobytes())
    np.savez(out/'weights.npz', **arrays)
    (out/'weights.2bit').write_bytes(b''.join(packed))
    save(out/'model.json', dict(schema=1, graph='causal_ternary_ctc', recipe=model.recipe,
         alphabet=ALPHABET, arithmetic='expanded ternary FP32 reference',
         parameters=sum(p.numel() for p in model.parameters()), packed_bytes=sum(map(len,packed))))


def load_model(out, device='cpu'):
    out = Path(out)
    meta = json.loads((out/'model.json').read_text())
    if meta['schema'] != 1 or meta['graph'] != 'causal_ternary_ctc' or meta['alphabet'] != ALPHABET:
        raise ValueError('Unsupported model contract')
    model = AcousticModel(meta['recipe'])
    expected = {n+s for n,_ in model.named_parameters() for s in ('.codes','.scale')}
    arrays = np.load(out/'weights.npz', allow_pickle=False)
    if set(arrays.files) != expected:
        raise ValueError('Unexpected weights')
    packed = []
    with torch.no_grad():
        for name,p in model.named_parameters():
            q,s = arrays[name+'.codes'], arrays[name+'.scale']
            if q.dtype != np.int8 or q.shape != tuple(p.shape) or not np.isin(q,[-1,0,1]).all():
                raise ValueError('Invalid ternary tensor')
            if s.shape != () or not np.isfinite(s) or float(s) <= 0:
                raise ValueError('Invalid scale')
            p.copy_(torch.from_numpy(q.astype(np.float32)*float(s)))
            u = np.pad((q.ravel()+1).astype(np.uint8),(0,(-q.size)%4),constant_values=1)
            packed.append((u[::4] | u[1::4]<<2 | u[2::4]<<4 | u[3::4]<<6).tobytes())
    if (out/'weights.2bit').read_bytes() != b''.join(packed):
        raise ValueError('Packed export mismatch')
    for module in model.modules():
        if isinstance(module,TernaryConv):
            module.reference = True
    return model.to(device).eval()


def transcribe(model, audio, device='cpu'):
    with torch.inference_mode():
        logits = model(torch.from_numpy(features(audio)[None]).to(device))
        return decode(logits[0].argmax(-1).tolist())
