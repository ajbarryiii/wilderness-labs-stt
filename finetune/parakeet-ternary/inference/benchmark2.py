"""Paired end-to-end validation of the previous and optimized packed runtimes."""
import argparse
import contextlib
import json
import statistics
import time
from pathlib import Path

import torch

import paths
from .benchmark import DEFAULT_EXPORT, capture, energy_meter, metadata, measure_pair, save
from .graphs import enable_encoder_graphs
from .optimized import enable_optimizations, disable_optimizations
from .runtime import load_packed


def texts(output):
    if isinstance(output, tuple): output = output[0]
    return [h.text if hasattr(h, "text") else str(h) for h in output]


def main():
    import evaluate as ev
    from nemo.utils import logging
    logging.setLevel(logging.ERROR)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    p.add_argument("--audio-seconds", nargs="+", type=float, default=[3, 10, 30])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--warmup-seconds", type=float, default=2)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--skip-encoder", action="store_true")
    args = p.parse_args()
    paths.require_mount()
    out = paths.ARTIFACTS / "kernel-runs" / f"{time.strftime('%Y%m%dT%H%M%S')}-benchmark2"
    out.mkdir(); print("RESULTS", out, flush=True)
    with paths.gpu_lock("final paired optimized benchmark"), torch.inference_mode(), energy_meter() as meter, contextlib.ExitStack() as stack:
        meter._guard(); save(out, "metadata.json", metadata(args)); save(out, "gpu.json", meter.metadata())
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        models = {name:load_packed(args.export) for name in ["previous_packed", "decoder_fix", "optimized"]}
        for name, model in models.items():
            stack.enter_context(ev.inference_settings(model))
            comp = model.decoding.decoding.decoding_computer
            # Make both the old fallback and new conditional graph explicit.
            comp.force_cuda_graphs_mode("no_while_loops" if name == "previous_packed" else "full_graph")
            if name == "optimized":
                enable_optimizations(model,decoder=False,position_dot=False)
                stack.callback(disable_optimizations, model)
            else:
                enable_encoder_graphs(model)
                from .graph_memory import decoder_graph_pool
                stack.enter_context(decoder_graph_pool(model))
        records = [json.loads(s) for s in (paths.MANIFESTS / "test_librispeech_clean.jsonl").read_text().splitlines()]
        all_windows, errors = [], []
        for seconds in args.audio_seconds:
            rec = min(records, key=lambda r:abs(r['duration']-seconds))
            audio, lengths = ev.audio_batch([rec]*args.batch_size); audio, lengths = audio.cuda(), lengths.cuda()
            f, fl = models["previous_packed"].preprocessor(input_signal=audio, length=lengths)
            graphs, encoded = {}, {}
            for name in ["previous_packed", "optimized"]:
                model = models[name]
                graph, output = capture(lambda model=model:model.encoder(audio_signal=f, length=fl))
                graphs[name], encoded[name] = graph, output
            ref, y = encoded["previous_packed"][0], encoded["optimized"][0]
            err = dict(audio_seconds=rec['duration'], batch_size=args.batch_size,
                       max_abs=float((ref-y).abs().max()), relative_l2=float((ref-y).norm()/ref.norm()))
            torch.testing.assert_close(y, ref, atol=3e-6, rtol=3e-5)
            if not args.skip_encoder:
                all_windows.extend(measure_pair({n:g.replay for n,g in graphs.items()}, args, meter, out, f"encoder-{seconds:g}s"))
            del graphs, encoded, ref, y, graph, output
            def decode(model):
                enc, enc_len = model(input_signal=audio, input_signal_length=lengths)
                return model.decoding.rnnt_decoder_predictions_tensor(enc, enc_len, return_hypotheses=False)
            funcs = {name:(lambda model=model:decode(model)) for name,model in models.items()}
            hyps = {name:texts(fn()) for name,fn in funcs.items()}
            assert hyps["previous_packed"] == hyps["decoder_fix"] == hyps["optimized"], hyps
            err["hypotheses"] = hyps
            errors.append(err); save(out, "checks.json", errors)
            all_windows.extend(measure_pair(funcs, args, meter, out, f"transcribe-{seconds:g}s"))
            summary = []
            for label in dict.fromkeys(w["workload"] for w in all_windows):
                values = {}
                for name in dict.fromkeys(w["variant"] for w in all_windows if w["workload"] == label):
                    ws = [w for w in all_windows if w["workload"] == label and w["variant"] == name]
                    values[name] = {key:statistics.median(w[key] for w in ws)
                                    for key in ["median_ms", "p95_ms", "joules_per_iteration", "average_watts"]}
                old, new = values["previous_packed"], values["optimized"]
                summary.append(dict(workload=label, variants=values, speedup=old['median_ms']/new['median_ms'],
                                    energy_reduction=1-new['joules_per_iteration']/old['joules_per_iteration']))
            save(out, "summary.json", dict(batch_size=args.batch_size, workloads=summary,
                                           scope="GPU board energy, warmed inference, inputs already on GPU; all variants resident; excludes host energy and disk I/O"))
            print("SUMMARY", json.dumps(summary[-1]), flush=True)


if __name__ == "__main__":
    main()
