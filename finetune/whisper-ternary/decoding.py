"""Greedy FP32 decoding and scoring shared by in-training selection and final evaluation.

One code path for every arm and split: batched features from the manifest,
model.generate with the checkpoint's shipped generation config unchanged
(greedy, max 225 new tokens), strict FP32 (autocast and TF32 off), Whisper-normalized WER.
Wall time covers feature extraction, generation and scoring, not model loading.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import checkpoint
import data
import wer


@torch.no_grad()
def decode(model, processor, manifest: list[dict], device: str | torch.device,
           batch_size: int = 64, max_new_tokens: int = 225,
           workers: int = 4) -> tuple[list[dict], dict]:
    """Per-utterance records and run info (generation settings, wall time, throughput).

    Leaves the model in eval mode; callers that train restore model.train().
    """
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("decoding is specified in FP32")
    device = torch.device(device)
    normalize = wer.Normalizer()
    by_id = {r["id"]: r for r in manifest}
    dataset = data.LibriSpeechDataset(manifest, processor.feature_extractor, processor.tokenizer,
                                      model.config.decoder_start_token_id)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                        collate_fn=data.collate, pin_memory=device.type == "cuda")
    model.eval()
    records = []
    start = time.perf_counter()
    # cuDNN convolutions default to TF32 in this torch; matmuls already default to IEEE FP32.
    with torch.autocast(device.type, enabled=False), torch.backends.cudnn.flags(
            enabled=True, benchmark=torch.backends.cudnn.benchmark,
            deterministic=torch.backends.cudnn.deterministic, allow_tf32=False):
        for batch in loader:
            ids = model.generate(batch["input_features"].to(device), num_beams=1,
                                 do_sample=False, max_new_tokens=max_new_tokens)
            hyps = processor.batch_decode(ids, skip_special_tokens=True)
            for uid, hyp, truncated in zip(batch["ids"], hyps, batch["truncated"]):
                ref = by_id[uid]["text"]
                ref_norm, hyp_norm = normalize(ref), normalize(hyp)
                s, d, i = wer.edit_counts(ref_norm.split(), hyp_norm.split())
                records.append({"id": uid, "ref": ref, "hyp": hyp, "ref_norm": ref_norm,
                                "hyp_norm": hyp_norm, "S": s, "D": d, "I": i,
                                "truncated": truncated})
    wall = time.perf_counter() - start
    info = {"generation": {"num_beams": 1, "do_sample": False, "max_new_tokens": max_new_tokens,
                           "batch_size": batch_size,
                           "precision": "float32, TF32 off, autocast disabled",
                           "generation_config": model.generation_config.to_dict()},
            "wall_s": wall, "utterances_per_s": len(records) / wall}
    return records, info


def describe_source(kind: str, path: Path, description: str) -> dict:
    """Identify scored weights: kind, path, weight-file SHA-256, base checkpoint identity."""
    weights = {"pretrained": "model.safetensors", "hf-dir": "model.safetensors",
               "export": "export.safetensors"}[kind]
    return {"kind": kind, "description": description, "path": str(path),
            "weights_file": weights, "sha256": checkpoint.sha256(Path(path) / weights),
            **checkpoint.base_provenance()}


def report(records: list[dict], info: dict, split: str, source: dict | None,
           limit: int | None) -> dict:
    """Evaluation JSON of evaluate.py and train.py; source is omitted when None (dev-subset files)."""
    result = {"split": split, "utterances": len(records), "limit": limit,
              "truncated_over_30s": sum(r["truncated"] for r in records),
              "generation": info["generation"], "normalizer": wer.NORMALIZER,
              "wer": wer.corpus_wer(records), "wall_s": info["wall_s"],
              "utterances_per_s": info["utterances_per_s"], "records": records}
    return result if source is None else {"source": source, **result}
