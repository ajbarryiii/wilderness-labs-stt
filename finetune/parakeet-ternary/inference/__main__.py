"""Transcribe 16 kHz audio with the packed SM120 model."""
import argparse
import contextlib
from pathlib import Path

from .runtime import load_packed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument("--export", required=True, type=Path, help="existing parakeet-ternary-v1 export directory")
    parser.add_argument("--batch-size", type=int, default=1)
    speed = parser.add_mutually_exclusive_group()
    speed.add_argument("--encoder-graphs", action="store_true", help="cache up to four repeated encoder shapes")
    speed.add_argument("--optimized", action="store_true", help="packed encoder, FP32 decoder fusions, and CUDA graphs")
    speed.add_argument("--experimental-cuda", action="store_true",
                       help="batch-one RTX 5090 CUDA encoder with expanded weights and fused row kernels")
    parser.add_argument("--encoder-storage", choices=["packed", "expanded"], default="packed",
                        help="with --optimized: expanded trades about 604 MB for faster encoder kernels")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.encoder_storage != "packed" and not args.optimized:
        parser.error("--encoder-storage expanded requires --optimized")
    if args.experimental_cuda and args.batch_size != 1:
        parser.error("--experimental-cuda is validated for --batch-size 1")
    from hf.load_ternary import transcribe
    from nemo.utils import logging
    logging.setLevel(logging.ERROR)
    model = load_packed(args.export)
    with contextlib.ExitStack() as stack:
        if args.experimental_cuda:
            import torch
            from .cuda_pipeline import experimental_pipeline
            stack.enter_context(torch.inference_mode())
            stack.enter_context(experimental_pipeline(model, 'ptx_row_best'))
        elif args.optimized:
            from .optimized import enable_optimizations, disable_optimizations
            enable_optimizations(model, encoder_storage=args.encoder_storage)
            stack.callback(disable_optimizations, model)
        elif args.encoder_graphs:
            from .graphs import enable_encoder_graphs, disable_encoder_graphs
            enable_encoder_graphs(model)
            stack.callback(disable_encoder_graphs, model)
        texts = transcribe(model, args.audio, batch_size=args.batch_size)
    for path, text in zip(args.audio, texts):
        print(f"{path}\t{text}" if len(texts) > 1 else text)


if __name__ == "__main__":
    main()
