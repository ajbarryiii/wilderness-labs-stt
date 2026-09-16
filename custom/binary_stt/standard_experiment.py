"""Bounded standard Conformer experiment with a memorization gate and watchdog.

Run: ./binary_stt/python -m binary_stt.standard_experiment --run-dir /mnt/hd/...
Writes all data/checkpoints to the verified artifact disk. No automatic resume.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import time

import torch
from torch import nn
import torchaudio

from .__main__ import read_environment, runtime_versions
from .data import StreamingSpeechDataset
from .notifications import Notifier
from .standard_model import StandardCTC
from .storage import (ROOT, append_json, atomic_torch_save, check_free_space,
                      configure_environment, digest, ensure_artifact_path,
                      gpu_lock, heartbeat, write_json)
from .supervisor import run_supervised
from .tokenizer import CharacterTokenizer
from .train import _collate, _loss, evaluate, check_finite_state

TOKENIZER = CharacterTokenizer(" abcdefghijklmnopqrstuvwxyz'")
NOTIFICATIONS = {"email_to": "ajbarryiii@gmail.com", "email_required": True, "desktop": True}


class LogMel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
            n_mels=80, power=2., center=True)

    def forward(self, waveform):
        return self.mel(waveform.float()).clamp_min(1e-10).log()


def prepare_row(row):
    row = dict(row)
    row['original_text'] = row['text']
    row['text'] = ' '.join(re.sub("[^a-z' ]", ' ', row['text'].lower()).split())
    row['target'] = TOKENIZER.encode(row['text'])
    required = len(row['target']) + sum(a == b for a, b in zip(row['target'], row['target'][1:]))
    frames = (row['audio'].numel() // 160 + 2) // 2
    return row if row['target'] and required <= frames else None


@torch.no_grad()
def acoustic_check(model, rows, extractor):
    model.eval()
    result = []
    for row in rows:
        pair = []
        for silence in (False, True):
            item = {**row, 'audio': torch.zeros_like(row['audio'])} if silence else row
            x, lengths, targets, target_lengths = _collate([item], extractor, torch.device('cuda'))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, out_lengths = model(x, lengths)
            pair.append({'prediction': TOKENIZER.decode_ctc(logits.argmax(-1)[0, :out_lengths[0]].tolist()),
                         'loss': float(_loss(logits, out_lengths, targets, target_lengths, [item]).mean())})
        result.append({'id': row['id'], 'reference': row['text'], 'speech': pair[0], 'silence': pair[1]})
    model.train()
    return result


def worker(run_dir):
    notifier = Notifier(run_dir, NOTIFICATIONS)
    notifier.check_delivery_config()
    started = time.monotonic()
    stop = []
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda *_: stop.append(True))
    stream = None
    status = {'state': 'starting', 'phase': 'prepare', 'step': 0}
    model = optimizer = None

    def save():
        if model is not None:
            heartbeat(run_dir, 'checkpoint', **{k: status[k] for k in ('phase', 'step')})
            atomic_torch_save(run_dir / f"{status['phase']}-latest.pt", {
                'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'status': status, 'architecture': {'width': 256, 'layers': 6, 'vocab_size': 29},
                'alphabet': TOKENIZER.alphabet, 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all(),
                'scope': 'Diagnostic checkpoint; exact streaming resume is not implemented.'})

    def check_stop():
        if stop or (run_dir / 'STOP').exists():
            raise InterruptedError('Stop requested')
        if time.monotonic() - started > 7200:
            raise RuntimeError('Two-hour experiment budget exceeded')

    try:
        check_free_space(10)
        with gpu_lock('cuda'):
            torch.set_num_threads(4)
            extractor = LogMel()
            base = ROOT / 'runs/full-lr-low-001'
            config = json.loads((base / 'config.json').read_text())
            receipt = json.loads((base / 'prepared.json').read_text())
            if digest(base / 'validation.pt') != receipt['validation_sha256']:
                raise RuntimeError('Frozen validation checksum mismatch')
            validation = []
            for row in torch.load(base / 'validation.pt', weights_only=False):
                if row['source'].endswith('/validation.clean') and 1 <= row['seconds'] <= 12:
                    item = prepare_row(row)
                    if item: validation.append(item)
            validation = validation[:96]
            if len(validation) < 32:
                raise RuntimeError('Insufficient held-out clean validation')
            excluded = {row['content_id'] for row in validation}
            source = next(s for s in config['data']['train_sources'] if s['split'] == 'train.clean.100')
            stream = StreamingSpeechDataset([source], seed=20260916, shuffle_buffer=32,
                                            min_seconds=1, max_seconds=12,
                                            max_rejection_fraction=.95)
            iterator = iter(stream)

            def next_row():
                for _ in range(1000):
                    check_stop()
                    heartbeat(run_dir, 'stream_read', phase=status['phase'], step=status['step'])
                    row = prepare_row(next(iterator))
                    if row and row['content_id'] not in excluded:
                        return row
                raise RuntimeError('Cannot find usable non-overlapping speech')

            fixed = []
            seen = set()
            while len(fixed) < 16:
                row = next_row()
                if row['seconds'] <= 6 and row['content_id'] not in seen:
                    seen.add(row['content_id'])
                    fixed.append(row)
            atomic_torch_save(run_dir / 'overfit-data.pt', fixed)
            atomic_torch_save(run_dir / 'validation.pt', validation)
            write_json(run_dir / 'recipe.json', {
                'model': 'TorchAudio Conformer', 'width': 256, 'layers': 6, 'heads': 4,
                'ff_dim': 1024, 'conv_kernel': 31, 'subsampling': 2, 'offline': True,
                'alphabet': TOKENIZER.alphabet, 'feature_normalization': 'per-utterance per-mel CMVN',
                'source': source, 'seed': 20260916, 'validation_examples': len(validation),
                'overfit_examples': 16, 'overfit_max_steps': 1000, 'stream_max_steps': 500,
                'lr': .0003, 'warmup': 50, 'batch': 4, 'stream_accumulation': 4,
                'gate': 'training CER <= .05 and WER <= .10; memorization, not generalization',
                'runtime': runtime_versions(),
                'source_hashes': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
                'comparison': 'Different size, tokenizer, frontend, context, dataset and LR; not an isolated quantization ablation'})

            for phase, maximum, accumulation in [('overfit', 1000, 1), ('stream', 500, 4)]:
                torch.manual_seed(20260916)
                torch.cuda.manual_seed_all(20260916)
                rng = random.Random(20260916)
                model = StandardCTC(dropout=0 if phase == 'overfit' else .1).cuda()
                optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, betas=(.9, .98), weight_decay=0.01)
                status = {'state': 'running', 'phase': phase, 'step': 0,
                          'parameters': sum(p.numel() for p in model.parameters()), 'audio_seconds': 0.}
                panel = fixed if phase == 'overfit' else validation
                metrics = evaluate(model, panel, TOKENIZER, extractor, torch.device('cuda'), run_dir, 0, 4)
                append_json(run_dir / 'metrics.jsonl', {'kind': 'evaluation', 'phase': phase, 'step': 0, **metrics})
                best_loss = float('inf')
                bad = 0
                passed = False
                unique = set()
                unique_seconds = 0.
                for step in range(1, maximum + 1):
                    check_stop()
                    model.train()
                    lr = .0003 * min(step / 50, 1.)
                    for group in optimizer.param_groups: group['lr'] = lr
                    optimizer.zero_grad(set_to_none=True)
                    total_loss = 0.
                    for micro in range(accumulation):
                        batch = rng.sample(fixed, 4) if phase == 'overfit' else [next_row() for _ in range(4)]
                        x, lengths, targets, target_lengths = _collate(batch, extractor, torch.device('cuda'))
                        heartbeat(run_dir, 'train', phase=phase, step=step, microbatch=micro)
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            logits, out_lengths = model(x, lengths)
                        loss = _loss(logits, out_lengths, targets, target_lengths, batch).mean()
                        (loss / accumulation).backward()
                        total_loss += float(loss.detach()) / accumulation
                        status['audio_seconds'] += sum(row['seconds'] for row in batch)
                        for row in batch:
                            if row['content_id'] not in unique:
                                unique.add(row['content_id'])
                                unique_seconds += row['seconds']
                    norm = float(nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True))
                    optimizer.step()
                    status.update(step=step, loss=total_loss, gradient_norm=norm,
                                  wall_seconds=time.monotonic() - started,
                                  unique_recordings=len(unique), unique_audio_seconds=unique_seconds)
                    if step > 50:
                        best_loss = min(best_loss, total_loss)
                        bad = bad + 1 if total_loss > max(20, 5 * best_loss) or norm == 0 else 0
                        if bad >= 10: raise RuntimeError('Sustained loss explosion or zero gradients')
                    if step % 10 == 0:
                        write_json(run_dir / 'status.json', status)
                        append_json(run_dir / 'metrics.jsonl', {'kind': 'train', **status, 'lr': lr})
                    if step % 50 == 0:
                        metrics = evaluate(model, panel, TOKENIZER, extractor, torch.device('cuda'), run_dir, step, 4)
                        status['evaluation'] = metrics
                        append_json(run_dir / 'metrics.jsonl', {'kind': 'evaluation', 'phase': phase, 'step': step, **metrics})
                        print(json.dumps({'phase': phase, 'step': step, 'loss': metrics['loss'], 'wer': metrics['wer'], 'cer': metrics['cer']}), flush=True)
                        passed = metrics['cer'] <= .05 and metrics['wer'] <= .10
                        if step % 100 == 0 or passed: save()
                        write_json(run_dir / 'status.json', status)
                        if phase == 'overfit' and passed: break
                check_finite_state(model, optimizer)
                save()
                diagnostic = acoustic_check(model, panel[:8], extractor)
                write_json(run_dir / f'{phase}-result.json', {**status, 'acoustic_check': diagnostic,
                           'scope': 'training-set memorization' if phase == 'overfit' else 'held-out evaluation'})
                if phase == 'overfit' and not passed:
                    raise RuntimeError('Memorization gate failed: stopping before streaming trial')
                # Release the first model/Adam before constructing a fresh model.
                model = optimizer = None
                torch.cuda.empty_cache()
            status['state'] = 'completed'
            write_json(run_dir / 'status.json', status)
            notifier.notify('completed', 'Standard Conformer diagnostic and streaming trial completed.', **status)
            return 0
    except BaseException as exc:
        status.update(state='stopped' if isinstance(exc, InterruptedError) else 'failed', reason=str(exc))
        if isinstance(exc, InterruptedError): save()
        write_json(run_dir / 'status.json', status)
        notifier.notify(status['state'], str(exc), **status)
        return 1
    finally:
        if stream is not None: stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    configure_environment()
    run_dir = ensure_artifact_path(args.run_dir)
    read_environment(ROOT / 'secrets/email.env')
    if args.worker:
        if os.environ.get('BINARY_STT_SUPERVISED') != '1':
            raise RuntimeError('Worker requires watchdog')
        return worker(run_dir)
    if (run_dir / 'status.json').exists():
        raise FileExistsError('Use a fresh experiment directory; no automatic resume')
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_supervised([str(Path(__file__).parent / 'python'), '-m',
                           'binary_stt.standard_experiment', '--worker', '--run-dir', str(run_dir)],
                          run_dir, timeout_seconds=300, notification_config=NOTIFICATIONS,
                          stage_timeouts={'validation': 600, 'checkpoint': 600})


if __name__ == '__main__':
    raise SystemExit(main())
