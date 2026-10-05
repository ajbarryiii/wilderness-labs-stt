"""Fresh-process v2 comparison: CPU waveform -> text, GPU energy and process VRAM.

Run each variant as a separate invocation. Cache shapes accumulate in the stated
order; memory sampling includes load and first capture, separately from hot runs.
All downloaded models, dependencies and outputs stay on the mounted data disk.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time

import paths

ROOT = paths.ARTIFACTS / 'comparison'
EXPORT = paths.RUNS / 'main-M1-P2-lr5e-4/export'
VARIANTS = ['packed_previous', 'packed_second', 'packed_optimized', 'packed_expanded', 'nemo_fp32', 'nemo_bf16',
            'nemo_bf16_graphs', 'nemo_bf16_graphs_fused_decoder', 'onnx_cuda']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=5)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--durations', type=float, nargs='+', default=[3, 10, 30])
    p.add_argument('--quality-per-set', type=int, default=0)
    args = p.parse_args()
    paths.require_mount()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / 'deps'))
    sys.path.append('/mnt/hd/wilderness-labs-stt/inference-efficiency/deps')
    spec = importlib.util.spec_from_file_location('energy', paths.REPO / 'custom/inference-efficiency/energy.py')
    energy = importlib.util.module_from_spec(spec); spec.loader.exec_module(energy)
    import soundfile as sf
    is_ort = args.variant.startswith('onnx')
    report = {'variant': args.variant, 'args': {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'scope': 'Batch 1, CPU float32 waveform to text including preprocessing and transfers; no disk I/O; GPU board energy only; separate process per variant; shapes accumulate in order.',
              'source_sha256': {f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')},
              'versions': {name: importlib.metadata.version(name) for name in ['torch','nemo_toolkit']},
              'nvidia_model': json.loads((paths.MODEL_DIR/'lock.json').read_text()),
              'ternary_model': json.loads((EXPORT/'manifest.json').read_text())['sha256'],
              'onnx_revision': '0bbb45a3365852604aef28b538a8f066f4ccaa85', 'workloads': []}
    target = args.out / f'{args.variant}.json'
    def save(): target.write_text(json.dumps(report, indent=2)+'\n')
    with paths.gpu_lock('isolated v2 comparison'), contextlib.ExitStack() as stack:
        meter = stack.enter_context(energy.EnergyMeter())
        meter._guard(); report['gpu'] = meter.metadata()
        from .sprint4 import require_power_cap
        report['power_limit_w'] = require_power_cap(meter)
        samples, errors = [], []
        def memory():
            ps = meter._nvml.nvmlDeviceGetComputeRunningProcesses(meter._handle)
            return max([int(q.usedGpuMemory) for q in ps if q.pid == os.getpid()] or [0])
        stop = threading.Event()
        def sample():
            while not stop.wait(.01):
                try: samples.append((time.perf_counter(), memory()))
                except Exception as exc: errors.append(str(exc)); return
        thread = threading.Thread(target=sample, daemon=True); thread.start()
        def end_sampler(): stop.set(); thread.join()
        stack.callback(end_sampler)
        started = time.perf_counter()
        if is_ort:
            import onnxruntime as ort
            ort.preload_dlls()
            import onnx_asr
            options = ort.SessionOptions(); options.intra_op_num_threads=4
            options.inter_op_num_threads=1
            model = onnx_asr.load_model('nemo-parakeet-tdt-0.6b-v2', str(ROOT/'onnx-v2'),
                       providers=['CUDAExecutionProvider'], sess_options=options)
            report['available_providers'] = ort.get_available_providers()
            report['versions']['onnx-asr'] = importlib.metadata.version('onnx-asr')
            report['versions']['onnxruntime-gpu'] = ort.__version__
            report['active_providers'] = {name: getattr(model.asr, name).get_providers()
                                          for name in ['_encoder', '_decoder_joint']}
            report['provider_options'] = {name: getattr(model.asr, name).get_provider_options()
                                          for name in ['_encoder', '_decoder_joint']}
            assert all(v[0] == 'CUDAExecutionProvider' for v in report['active_providers'].values()), report['active_providers']
            def recognize(audio): return model.recognize(audio, sample_rate=16000)
            synchronize = lambda: None
            def snapshot(): return {'process_bytes': memory()}
        else:
            import torch
            import evaluate as ev
            from .runtime import load_packed
            from .graphs import enable_encoder_graphs, enable_pipeline_graphs
            from .optimized import enable_optimizations, disable_optimizations
            from nemo.utils import logging
            logging.setLevel(logging.ERROR)
            stack.enter_context(torch.inference_mode())
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            if args.variant.startswith('packed'):
                model = load_packed(EXPORT)
            else:
                model = ev.load_pretrained('cpu')
                if args.variant.startswith('nemo_bf16_graphs'):
                    model.encoder.to(dtype=torch.bfloat16)
                model.cuda()
            stack.enter_context(ev.inference_settings(model))
            comp = model.decoding.decoding.decoding_computer
            from omegaconf import OmegaConf
            report['decoding_config'] = OmegaConf.to_container(model.cfg.decoding, resolve=True)
            comp.force_cuda_graphs_mode('no_while_loops' if args.variant == 'packed_previous' else 'full_graph')
            from .graph_memory import decoder_graph_pool
            if args.variant in ['packed_optimized','packed_second','packed_expanded']:
                enable_optimizations(model,decoder=args.variant!='packed_second',position_dot=args.variant!='packed_second',
                                     encoder_storage='expanded' if args.variant=='packed_expanded' else 'packed')
                stack.callback(disable_optimizations, model)
            elif args.variant == 'packed_previous': enable_encoder_graphs(model)
            elif args.variant.startswith('nemo_bf16_graphs'):
                original = model.encoder.forward
                def cast_encoder(audio_signal, length=None, **kwargs):
                    return original(audio_signal=audio_signal.bfloat16(), length=length, **kwargs)
                model.encoder.forward = cast_encoder
                enable_pipeline_graphs(model)
            if not args.variant.startswith('packed'):
                stack.enter_context(decoder_graph_pool(model))
            if args.variant == 'nemo_bf16_graphs_fused_decoder':
                from .decoder_kernels import decoder_transform,control_transform
                stack.enter_context(decoder_transform(model,joint=True,lstm=True,precompute=True,block=1))
                stack.enter_context(control_transform(model,storage=True))
            report['registered_tensor_bytes'] = sum(t.numel()*t.element_size() for t in list(model.parameters())+list(model.buffers()))
            def recognize(audio):
                a = torch.from_numpy(audio).unsqueeze(0).cuda()
                lengths = torch.tensor([len(audio)], device='cuda', dtype=torch.long)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.variant=='nemo_bf16'):
                    enc, enc_len = model(input_signal=a, input_signal_length=lengths)
                out = model.decoding.rnnt_decoder_predictions_tensor(enc.float(), enc_len, return_hypotheses=False)
                if isinstance(out, tuple): out=out[0]
                return out[0].text if hasattr(out[0],'text') else str(out[0])
            synchronize = torch.cuda.synchronize
            def snapshot():
                synchronize()
                return {'process_bytes': memory(), 'torch_allocated': torch.cuda.memory_allocated(),
                        'torch_reserved': torch.cuda.memory_reserved(), 'torch_peak_allocated': torch.cuda.max_memory_allocated(),
                        'torch_peak_reserved': torch.cuda.max_memory_reserved()}
        report['load_seconds'] = time.perf_counter()-started
        report['loaded'] = snapshot(); save()
        records = [json.loads(x) for x in (paths.MANIFESTS/'test_librispeech_clean.jsonl').read_text().splitlines()]
        for duration in args.durations:
            rec = min(records,key=lambda r:abs(r['duration']-duration))
            audio, rate = sf.read(rec['audio_filepath'],dtype='float32'); assert rate==16000 and audio.ndim==1
            if not is_ort: torch.cuda.reset_peak_memory_stats()
            begin_samples = len(samples)
            first = time.perf_counter(); hypothesis = recognize(audio); synchronize()
            row={'record':rec, 'hypothesis':hypothesis,'first_call_seconds':time.perf_counter()-first,
                 'after_first':snapshot(), 'windows':[]}
            for repeat in range(args.repeats):
                require_power_cap(meter)
                warm = time.perf_counter()
                while time.perf_counter()-warm < 2:
                    recognize(audio); synchronize()
                if not is_ort: torch.cuda.reset_peak_memory_stats()
                start_samples = len(samples)
                w=meter.measure(lambda: recognize(audio), min_seconds=args.seconds, synchronize=synchronize)
                times=sorted(w['iteration_seconds'])
                w['median_ms']=statistics.median(times)*1000
                w['p95_ms']=times[min(len(times)-1,int(len(times)*.95))]*1000
                w['memory']=snapshot()
                w['process_sampled_peak_bytes']=max([v for _,v in samples[start_samples:]]+[memory()])
                row['windows'].append(w)
            row['summary']={k:statistics.median(w[k] for w in row['windows']) for k in ['median_ms','p95_ms','joules_per_iteration','average_watts']}
            row['process_sampled_peak_including_first_bytes']=max([v for _,v in samples[begin_samples:]]+[memory()])
            report['workloads'].append(row); save()
            print(args.variant, duration, row['summary'], row['windows'][-1]['memory'],flush=True)
        report['performance_phase_process_peak_bytes']=max(v for _,v in samples)
        if args.quality_per_set:
            from .benchmark import records_for_sets
            import evaluate as ev
            quality=[]
            for rec in records_for_sets(args.quality_per_set):
                audio,rate=sf.read(rec['audio_filepath'],dtype='float32'); assert rate==16000
                hyp=recognize(audio)
                quality.append({**rec,'hypothesis':hyp,'score':ev.score(rec['text'],hyp)})
                if len(quality) % 32 == 0:
                    print('quality', args.variant, len(quality), flush=True)
            report['quality']=quality
        end_sampler()
        if errors: raise RuntimeError(errors)
        report['memory_samples_10ms']=samples
        save()

if __name__ == '__main__': main()
