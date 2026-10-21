"""Probe frontend+encoder graph capture and profile full TDT transcription."""
import json
import time
import traceback

import torch

import paths
from .benchmark import DEFAULT_EXPORT, capture, energy_meter, save
from .experiments import latency
from .graphs import enable_encoder_graphs
from .runtime import load_packed


def main():
    import evaluate as ev
    from nemo.utils import logging
    logging.setLevel(logging.ERROR)
    paths.require_mount()
    out = paths.ARTIFACTS / "kernel-runs" / f"{time.strftime('%Y%m%dT%H%M%S')}-pipeline-probe"
    out.mkdir(); print("RESULTS", out, flush=True)
    with paths.gpu_lock("pipeline graph and decoder profile"), torch.inference_mode(), energy_meter() as meter:
        meter._guard()
        model = load_packed(DEFAULT_EXPORT)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with ev.inference_settings(model):
            records = [json.loads(s) for s in (paths.MANIFESTS / "test_librispeech_clean.jsonl").read_text().splitlines()]
            rec = min(records, key=lambda r:abs(r['duration']-10))
            a, l = ev.audio_batch([rec]); a, l = a.cuda(), l.cuda()
            report = {}
            try:
                graph, outputs = capture(lambda:model(input_signal=a, input_signal_length=l))
                report["frontend_encoder_graph_ms"] = latency(graph)
                a.mul_(.7); l.sub_(1000); graph.replay()
                eager = model(input_signal=a, input_signal_length=l)
                torch.testing.assert_close(outputs, eager, atol=1e-6, rtol=1e-5)
                report["changed_audio_lengths"] = "pass"
                del graph, outputs
            except Exception as exc:
                report["capture_error"] = repr(exc)
                traceback.print_exc()
            save(out, "probe.json", report); print(report, flush=True)
            enable_encoder_graphs(model)
            def decode():
                e, el = model(input_signal=a, input_signal_length=l)
                return model.decoding.rnnt_decoder_predictions_tensor(e, el, return_hypotheses=False)
            for _ in range(4): decode()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as p:
                for _ in range(3): decode()
                torch.cuda.synchronize()
            (out / "cpu-profile.txt").write_text(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
            (out / "gpu-profile.txt").write_text(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
            computer = model.decoding.decoding.decoding_computer
            report["decoder_class"] = type(computer).__name__
            report["decoder_graph_mode"] = str(computer.cuda_graphs_mode)
            save(out, "probe.json", report); print(report, flush=True)


if __name__ == "__main__":
    main()
