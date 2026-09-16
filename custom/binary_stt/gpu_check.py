"""Bounded full-model CUDA training verification; no accuracy measurement.

Run with ``./binary_stt/python -m binary_stt.gpu_check``. This performs exactly
three synthetic optimizer steps and stores only a JSON report on the data disk.
It exercises BF16 fake-quantization training, not a packed popcount kernel.
"""

from __future__ import annotations

import argparse
import platform
import signal
import subprocess
import time
import traceback

from .storage import ROOT, configure_environment, ensure_artifact_path, gpu_lock, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "verification/gpu-full-v1/report.json"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frames", type=int, default=1000)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 4 or not 256 <= args.frames <= 1000:
        parser.error("This bounded check accepts batch size 1..4 and frames 256..1000")
    configure_environment()
    output = ensure_artifact_path(args.output, create_parent=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing verification report: {output}")

    import torch
    from torch.nn import functional as F
    from .model import BinaryCTCModel, ModelConfig
    from .train import check_finite_state, optimizer_for

    report = {
        "status": "starting", "python": platform.python_version(),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "input_shape": [args.batch_size, 80, args.frames],
        "input_kind": "synthetic Gaussian log-mel-shaped features; no audio/accuracy test",
        "precision": "FP32 latent parameters, gradients and AdamW states; BF16 autocast matmuls",
        "activation_checkpointing": True,
        "scope": "three optimizer steps only; fake-quantization, not packed XNOR/popcount",
        "steps": [],
    }
    write_json(output, report)
    started = time.monotonic()

    def deadline(*_):
        raise TimeoutError("Full-model GPU verification exceeded 600 seconds")

    old_handler = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(600)
    try:
        with gpu_lock("cuda"):
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError("Verification requires CUDA and native BF16 support")
            torch.manual_seed(8912)
            torch.cuda.manual_seed_all(8912)
            report.update(
                device=torch.cuda.get_device_name(),
                capability=list(torch.cuda.get_device_capability()),
                total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                nvidia_smi=subprocess.check_output([
                    "nvidia-smi", "--query-gpu=name,driver_version,memory.used,power.limit",
                    "--format=csv,noheader"], text=True, timeout=10).strip(),
            )
            config = ModelConfig(activation_checkpointing=True)
            model = BinaryCTCModel(config).to("cuda").train()
            report["parameters"] = sum(parameter.numel() for parameter in model.parameters())
            if report["parameters"] != 488_270_080:
                raise RuntimeError("Full model parameter count has changed")
            optimizer = optimizer_for(model, {"lr": 2e-4, "betas": (.9, .95), "weight_decay": .01})
            features = torch.randn(args.batch_size, 80, args.frames, device="cuda")
            lengths = torch.full((args.batch_size,), args.frames, dtype=torch.long, device="cuda")
            target_count = min(32, args.frames // 16)
            # Adjacent targets are distinct, leaving ample room for CTC alignment.
            targets = torch.arange(1, target_count + 1, device="cuda").repeat(args.batch_size)
            target_lengths = torch.full((args.batch_size,), target_count, dtype=torch.long, device="cuda")
            for step, fractions in enumerate(((0., 0.), (.5, .5), (1., 1.)), 1):
                model.set_quantization(*fractions)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                step_started = time.monotonic()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, output_lengths = model(features, lengths)
                log_probs = logits.float().log_softmax(-1).transpose(0, 1)
                losses = F.ctc_loss(log_probs, targets, output_lengths, target_lengths,
                                    blank=0, reduction="none", zero_infinity=False)
                loss = (losses / target_lengths).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite CTC loss at step {step}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                check_finite_state(model, optimizer)
                torch.cuda.synchronize()
                row = {
                    "step": step, "weight_fraction": fractions[0], "activation_fraction": fractions[1],
                    "loss": float(loss.detach()), "gradient_norm_before_clip": float(norm.detach()),
                    "seconds": time.monotonic() - step_started,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "finite_model_and_optimizer": True,
                    "output_shape": list(logits.shape),
                }
                report["steps"].append(row)
                write_json(output, report)
                print(row, flush=True)
            model_dtypes = sorted({str(p.dtype) for p in model.parameters()})
            optimizer_dtypes = sorted({str(v.dtype) for s in optimizer.state.values()
                                       for v in s.values() if isinstance(v, torch.Tensor)})
            report.update(status="passed", model_parameter_dtypes=model_dtypes,
                          optimizer_state_dtypes=optimizer_dtypes)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        report["total_seconds"] = time.monotonic() - started
        write_json(output, report)
        print(f"Verification report: {output}", flush=True)


if __name__ == "__main__":
    main()
