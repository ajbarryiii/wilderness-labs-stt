"""Fresh-process component timings for Parakeet v2 and same-weight controls.

Run dense_ternary_graphs first to write common decoder inputs, then pass its
decoder_inputs directory with --decoder-reference for both packed variants.
All NeMo decoders use full_graph mode. The third packed pass and the enhanced
NVIDIA BF16 control enable new predictor/joint/control fusions; other rows do not.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import paths
from .compare_v2 import EXPORT, ROOT

VARIANTS = ["dense_ternary_graphs", "packed_previous_graphs", "packed_second_pass_graphs", "packed_third_pass_graphs",
            "nemo_fp32", "nemo_bf16", "nemo_bf16_graphs", "onnx_cuda", "nemo_bf16_graphs_fused_decoder"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["all"] + VARIANTS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--durations", type=float, nargs="+", default=[3, 10, 30])
    parser.add_argument("--decoder-reference", type=Path)
    args = parser.parse_args()
    paths.require_mount()
    if not args.out.resolve().is_relative_to(Path("/mnt/hd")):
        parser.error("outputs must be on /mnt/hd")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.variant == "all":
        for variant in VARIANTS:
            command = [sys.executable, "-m", "inference.compare_v2_components", "--variant", variant,
                       "--out", str(args.out), "--seconds", str(args.seconds), "--repeats", str(args.repeats),
                       "--durations", *map(str, args.durations)]
            if variant.startswith("packed"):
                command += ["--decoder-reference", str(args.out / "decoder_inputs")]
            elif variant == "nemo_bf16_graphs_fused_decoder":
                command += ["--decoder-reference", str(args.out / "nvidia_decoder_inputs")]
            with (args.out / (variant + ".log")).open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            print(variant, "exit", result.returncode, flush=True)
            if result.returncode:
                raise RuntimeError(f"{variant} failed; see {args.out / (variant + '.log')}")
        return
    sys.path.insert(0, str(ROOT / "deps"))
    sys.path.append("/mnt/hd/wilderness-labs-stt/inference-efficiency/deps")
    import soundfile as sf
    from .benchmark import energy_meter
    report = {"variant": args.variant, "args": {k: str(v) if isinstance(v, Path) else v
                                               for k, v in vars(args).items()},
              "scope": "Isolated components; batch 1; warm GPU-board energy, excluding load/JIT/disk I/O. Component sums are not end-to-end timings.",
              "source_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                for f in Path(__file__).parent.glob("*.py")},
              "runtime_wrapper_sha256": hashlib.sha256((paths.HERE / "python").read_bytes()).hexdigest(),
              "versions": {n: importlib.metadata.version(n) for n in ["torch", "nemo_toolkit"]},
              "ternary_sha256": json.loads((EXPORT / "manifest.json").read_text())["sha256"],
              "nvidia_model": json.loads((paths.MODEL_DIR / "lock.json").read_text()), "workloads": []}
    target = args.out / (args.variant + ".json")
    def save():
        target.write_text(json.dumps(report, indent=2) + "\n")

    with paths.gpu_lock("isolated component benchmark"), contextlib.ExitStack() as stack:
        meter = stack.enter_context(energy_meter())
        meter._guard()
        report["gpu"] = meter.metadata()
        def process_memory():
            return max([int(p.usedGpuMemory) for p in
                        meter._nvml.nvmlDeviceGetComputeRunningProcesses(meter._handle)
                        if p.pid == os.getpid()] or [0])
        started = time.perf_counter()
        is_ort = args.variant == "onnx_cuda"
        if is_ort:
            import numpy as np
            import onnxruntime as ort
            ort.preload_dlls()
            import onnx_asr
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            options.inter_op_num_threads = 1
            model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2", str(ROOT / "onnx-v2"),
                                       providers=["CUDAExecutionProvider"], sess_options=options)
            asr = model.asr
            report["active_providers"] = {n: getattr(asr, n).get_providers()
                                          for n in ["_encoder", "_decoder_joint"]}
            report["provider_options"] = {n: getattr(asr, n).get_provider_options()
                                          for n in ["_encoder", "_decoder_joint"]}
            report["onnx_revision"] = "0bbb45a3365852604aef28b538a8f066f4ccaa85"
            assert all(v[0] == "CUDAExecutionProvider" for v in report["active_providers"].values())
            report["versions"].update({"onnxruntime-gpu": ort.__version__,
                                       "onnx-asr": importlib.metadata.version("onnx-asr")})
            synchronize = lambda: None
            def snapshot():
                return {"process_bytes": process_memory()}
            report["boundaries"] = {
                "preprocessor": "CPU waveform -> CPU features (native ONNX-ASR frontend)",
                "encoder": "CPU features -> CPU encoded tensors (native ONNX-ASR CUDA session, including copies)",
                "encoder_gpu": "GPU features -> GPU encoded tensors (additional I/O binding diagnostic)",
                "decoder": "CPU encoded tensors -> CPU text (native ONNX-ASR CUDA decoder/joint, includes per-token copies)"}
        else:
            import torch
            import evaluate as ev
            from nemo.utils import logging
            from .graphs import enable_encoder_graphs, disable_encoder_graphs
            from .graph_memory import decoder_graph_pool
            from .runtime import load_packed
            from .transforms import encoder_transform
            logging.setLevel(logging.ERROR)
            stack.enter_context(torch.inference_mode())
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            if args.variant.startswith("packed"):
                model = load_packed(EXPORT)
            elif args.variant == "dense_ternary_graphs":
                from export import load_export
                model = load_export(EXPORT, "cuda")
            else:
                model = ev.load_pretrained("cpu")
                if args.variant.startswith("nemo_bf16_graphs"):
                    model.encoder.to(dtype=torch.bfloat16)
                model.cuda()
            stack.enter_context(ev.inference_settings(model))
            if args.variant == "packed_third_pass_graphs":
                from .optimized import enable_optimizations, disable_optimizations
                enable_optimizations(model, max_graphs=0, decoder=True, position_dot=True)
                stack.callback(disable_optimizations, model)
            else:
                stack.enter_context(decoder_graph_pool(model))
                if args.variant == "nemo_bf16_graphs_fused_decoder":
                    from .decoder_kernels import decoder_transform, control_transform
                    stack.enter_context(decoder_transform(model, joint=True, lstm=True, precompute=True, block=1))
                    stack.enter_context(control_transform(model, storage=True))
            model.decoding.decoding.decoding_computer.force_cuda_graphs_mode("full_graph")
            report["conditional_graph_memory"] = "dedicated retained child-capture MemPool (graph_memory.decoder_graph_pool)"
            if args.variant == "packed_second_pass_graphs":
                # Explicit frozen second-pass encoder; independent of public defaults.
                stack.enter_context(encoder_transform(model, mode="int8x3", qkv=True, silu=True,
                    position=True, fold_bn=True, norm=True, attention=True, norm_quant=True,
                    position_dot=False))
            if args.variant.endswith("_graphs") or args.variant == "nemo_bf16_graphs_fused_decoder":
                enable_encoder_graphs(model)
                stack.callback(disable_encoder_graphs, model)
            synchronize = torch.cuda.synchronize
            def snapshot():
                synchronize()
                return {"process_bytes": process_memory(), "torch_allocated": torch.cuda.memory_allocated(),
                        "torch_reserved": torch.cuda.memory_reserved(),
                        "torch_peak_allocated": torch.cuda.max_memory_allocated(),
                        "torch_peak_reserved": torch.cuda.max_memory_reserved()}
            def decode(encoded, lengths):
                out = model.decoding.rnnt_decoder_predictions_tensor(encoded, lengths, return_hypotheses=False)
                if isinstance(out, tuple):
                    out = out[0]
                return out[0].text if hasattr(out[0], "text") else str(out[0])
            report["parameter_counts"] = {n: sum(p.numel() for p in m.parameters())
                                           for n, m in model.named_children()}
            report["registered_tensor_bytes"] = sum(t.numel() * t.element_size()
                for t in list(model.parameters()) + list(model.buffers()))
            report["boundaries"] = {
                "preprocessor": "GPU waveform -> GPU FP32 features; eager frontend",
                "encoder": "GPU features -> GPU encoded tensors; graph wrappers include input copies and owned output clones",
                "decoder": "GPU FP32 encoded tensors -> CPU text; NeMo full_graph, unchanged decoder/joint"}
            if args.variant in ("packed_third_pass_graphs", "nemo_bf16_graphs_fused_decoder"):
                report["boundaries"]["decoder"] = "GPU FP32 encoded tensors -> CPU text; NeMo full_graph with new FP32 predictor/joint/control fusions"
        report["load_seconds"] = time.perf_counter() - started
        report["loaded_memory"] = snapshot()
        records = [json.loads(x) for x in (paths.MANIFESTS / "test_librispeech_clean.jsonl").read_text().splitlines()]
        for duration in args.durations:
            record = min(records, key=lambda r: abs(r["duration"] - duration))
            audio, rate = sf.read(record["audio_filepath"], dtype="float32")
            assert rate == 16000 and audio.ndim == 1
            row = {"record": record, "components": {}}
            if is_ort:
                waveform, length = audio[None], np.array([len(audio)], dtype=np.int64)
                frontend = lambda: asr._preprocessor(waveform, length)
                features, features_len = frontend()
                encoder = lambda: asr._encode(features, features_len)
                encoded, encoded_len = encoder()
                decoder = lambda: [asr._decode_tokens(*x).text for x in asr._decoding(encoded, encoded_len)][0]
                row["hypothesis"] = decoder()
                # Native API exchanges NumPy arrays. A separate GPU-resident
                # diagnostic avoids incorrectly attributing its copies to math.
                binding = asr._encoder.io_binding()
                gpu_features = ort.OrtValue.ortvalue_from_numpy(features, "cuda", 0)
                gpu_length = ort.OrtValue.ortvalue_from_numpy(features_len, "cuda", 0)
                binding.bind_ortvalue_input("audio_signal", gpu_features)
                binding.bind_ortvalue_input("length", gpu_length)
                binding.bind_output("outputs", "cuda", 0)
                binding.bind_output("encoded_lengths", "cuda", 0)
                def encoder_gpu():
                    asr._encoder.run_with_iobinding(binding)
                    binding.synchronize_outputs()
                encoder_gpu()
                bound = binding.copy_outputs_to_cpu()
                assert np.allclose(bound[0].transpose(0, 2, 1), encoded, rtol=1e-5, atol=1e-5)
                functions = {"preprocessor": frontend, "encoder": encoder, "encoder_gpu": encoder_gpu, "decoder": decoder}
                row["encoded_shape"] = list(encoded.shape)
            else:
                waveform = torch.from_numpy(audio).unsqueeze(0).cuda()
                length = torch.tensor([len(audio)], device="cuda", dtype=torch.long)
                frontend = lambda: model.preprocessor(input_signal=waveform, length=length)
                features, features_len = frontend()
                if args.variant.startswith("nemo_bf16_graphs"):
                    features = features.bfloat16()
                def encoder():
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.variant == "nemo_bf16"):
                        return model.encoder(audio_signal=features, length=features_len)
                encoded, encoded_len = encoder()
                # Conversion occurs outside the decoder-only timed boundary.
                encoded = encoded.float()
                row["hypothesis"] = decode(encoded, encoded_len)
                row["encoded_shape"] = list(encoded.shape)
                ref_name = f"{duration:g}s.pt"
                if args.variant in ("dense_ternary_graphs", "nemo_bf16_graphs"):
                    reference_dir = args.out / ("decoder_inputs" if args.variant == "dense_ternary_graphs" else "nvidia_decoder_inputs")
                    reference_dir.mkdir(exist_ok=True)
                    torch.save({"encoded": encoded.cpu(), "length": encoded_len.cpu(),
                                "record": record, "hypothesis": row["hypothesis"]}, reference_dir / ref_name)
                if args.decoder_reference:
                    fixture = torch.load(args.decoder_reference / ref_name, map_location="cpu", weights_only=True)
                    assert fixture["record"] == record
                    error = (encoded.cpu() - fixture["encoded"]).abs()
                    row["encoder_vs_dense_reference"] = {"max_abs_error": error.max().item(),
                        "mean_abs_error": error.mean().item(),
                        "lengths_equal": torch.equal(encoded_len.cpu(), fixture["length"])}
                    encoded, encoded_len = fixture["encoded"].cuda(), fixture["length"].cuda()
                    row["decoder_input"] = ("identical NVIDIA BF16 reference" if args.variant == "nemo_bf16_graphs_fused_decoder"
                                            else "identical dense-reconstructed ternary reference")
                    row["decoder_reference_hypothesis"] = decode(encoded, encoded_len)
                    assert row["decoder_reference_hypothesis"] == fixture["hypothesis"]
                    row["own_encoder_matches_reference_text"] = row["hypothesis"] == fixture["hypothesis"]
                else:
                    row["decoder_input"] = "this variant's encoder output"
                decoder = lambda: decode(encoded, encoded_len)
                functions = {"preprocessor": frontend, "encoder": encoder, "decoder": decoder}
            # Alternate components across rounds to reduce thermal ordering bias.
            for repeat in range(args.repeats):
                order = list(functions)
                if repeat % 2:
                    order.reverse()
                for name in order:
                    fn = functions[name]
                    warm = time.perf_counter()
                    while time.perf_counter() - warm < 1:
                        fn()
                        synchronize()
                    if not is_ort:
                        torch.cuda.reset_peak_memory_stats()
                    window = meter.measure(fn, min_seconds=args.seconds, synchronize=synchronize)
                    times = sorted(window["iteration_seconds"])
                    window["median_ms"] = statistics.median(times) * 1000
                    window["p95_ms"] = times[min(len(times) - 1, int(len(times) * .95))] * 1000
                    window["memory_snapshot"] = snapshot()
                    component = row["components"].setdefault(name, {"windows": []})
                    component["windows"].append(window)
            for component in row["components"].values():
                component["summary"] = {k: statistics.median(w[k] for w in component["windows"])
                    for k in ["median_ms", "p95_ms", "joules_per_iteration", "average_watts"]}
            report["workloads"].append(row)
            save()
            print(args.variant, duration, {k: v["summary"] for k, v in row["components"].items()}, flush=True)
        save()


if __name__ == "__main__":
    main()
