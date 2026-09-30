"""Score one model on one LibriSpeech split with the shared greedy FP32 decoder.

Sources: the pinned pretrained checkpoint, a fine-tuned Hugging Face directory
(an FP32 run's best-hf), or a ternary export rebuilt by export.load_export.
--ptq quantizes the pretrained checkpoint without training, exports it to
runs/<variant>-ptq/export, and scores the model rebuilt from that export, so
every ternary number comes from a deployable file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration

import checkpoint
import data
import decoding
import export
import paths
import quant


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("pretrained", "hf-dir", "export"), required=True)
    parser.add_argument("--path", type=Path, help="best-hf directory or export directory")
    parser.add_argument("--split", choices=("dev-clean", "test-clean", "test-other"), required=True)
    parser.add_argument("--ptq", choices=("ternary", "ternary-embed"))
    parser.add_argument("--limit", type=int, help="first N id-sorted utterances")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.ptq and args.source != "pretrained":
        parser.error("--ptq applies only to --source pretrained")
    if (args.source == "pretrained") == (args.path is not None):
        parser.error("--path is required for hf-dir and export, and not accepted for pretrained")
    return args


def ptq_model(variant: str, processor) -> tuple[WhisperForConditionalGeneration, dict]:
    """Quantize the pretrained model, export it, check the export, and return the rebuilt model."""
    model = checkpoint.load_model("cuda")
    quant.quantize_model(model, include_embedding=variant == "ternary-embed")
    model.eval()
    out_dir = paths.run_dir(f"{variant}-ptq") / "export"
    export.export_model(model, out_dir, extra={
        "run_name": f"{variant}-ptq", "arm": variant,
        "notes": "post-training quantization of the pretrained checkpoint; no training",
        **checkpoint.base_provenance(), "source_hashes": checkpoint.source_hashes()})
    rebuilt = export.load_export(out_dir, device="cuda")
    features, decoder_ids = data.reference_batch(processor, model.config)
    data.write_json(out_dir / "reconstruction.json",
                    export.reconstruction_check(model, rebuilt, features, decoder_ids))
    source = decoding.describe_source("export", out_dir,
                                      f"{variant} PTQ of the pretrained checkpoint, rebuilt from export")
    return rebuilt, source


def main() -> None:
    args = parse_args()
    paths.require_mount()
    if not torch.cuda.is_available():
        sys.exit("CUDA is required")
    processor = checkpoint.load_processor()
    if args.ptq:
        model, source = ptq_model(args.ptq, processor)
    elif args.source == "pretrained":
        model = checkpoint.load_model("cuda")
        source = decoding.describe_source("pretrained", paths.MODEL_DIR,
                                          "pretrained Whisper tiny.en, no fine-tuning")
    elif args.source == "hf-dir":
        model = WhisperForConditionalGeneration.from_pretrained(
            args.path, local_files_only=True, use_safetensors=True, dtype=torch.float32).cuda()
        source = decoding.describe_source("hf-dir", args.path, "FP32 fine-tuned checkpoint")
    else:
        model = export.load_export(args.path, device="cuda")
        source = decoding.describe_source("export", args.path, "ternary export, rebuilt")
    manifest = data.build_manifest(args.split)[:args.limit]
    records, info = decoding.decode(model, processor, manifest, "cuda", batch_size=args.batch_size)
    result = decoding.report(records, info, args.split, source, args.limit)
    data.write_json(args.out, result)
    w = result["wer"]
    print(f"{args.split} n={w['utterances']} WER {100 * w['wer']:.2f}% (S {w['substitutions']} "
          f"D {w['deletions']} I {w['insertions']} / {w['ref_words']}) truncated "
          f"{result['truncated_over_30s']} {result['utterances_per_s']:.1f} utt/s -> {args.out}")


if __name__ == "__main__":
    main()
