"""Transcribe 16 kHz audio with the packed SM120 model."""
import argparse
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
    parser.add_argument("--encoder-storage", choices=["packed", "expanded"], default="packed",
                        help="with --optimized: expanded trades about 604 MB for faster encoder kernels")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.encoder_storage != "packed" and not args.optimized:
        parser.error("--encoder-storage expanded requires --optimized")
    from hf.load_ternary import transcribe
    from nemo.utils import logging
    logging.setLevel(logging.ERROR)
    model = load_packed(args.export)
    if args.optimized:
        from .optimized import enable_optimizations
        enable_optimizations(model, encoder_storage=args.encoder_storage)
    elif args.encoder_graphs:
        from .graphs import enable_encoder_graphs
        enable_encoder_graphs(model)
    try:
        texts = transcribe(model, args.audio, batch_size=args.batch_size)
    finally:
        if args.optimized:
            from .optimized import disable_optimizations
            disable_optimizations(model)
        elif args.encoder_graphs:
            from .graphs import disable_encoder_graphs
            disable_encoder_graphs(model)
    for path, text in zip(args.audio, texts):
        print(f"{path}\t{text}" if len(texts) > 1 else text)


if __name__ == "__main__":
    main()
