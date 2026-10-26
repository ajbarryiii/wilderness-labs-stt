"""A tiny Parakeet-TDT model for the test_decoder_* contract tests (no tests here): the pinned model's config and
tokenizer with a 2-layer, 64-dim encoder and small prediction and joint networks (as parakeet-ternary's
tests/test_train.py tiny_model), random weights. The tokenizer files are extracted once per process."""
from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path

_BASE: dict = {}


def base_dir() -> Path:
    import paths  # finetune/parakeet-ternary
    if "dir" not in _BASE:
        d = Path(tempfile.mkdtemp(prefix="decoder-only-tiny-"))
        with tarfile.open(paths.MODEL_FILE, "r:*") as tar:
            for member in tar.getmembers():
                name = Path(member.name).name
                if member.isfile() and not name.endswith(".ckpt"):
                    (d / name).write_bytes(tar.extractfile(member).read())
        _BASE["dir"] = d
    return _BASE["dir"]


def tiny_model(seed: int = 0):
    import torch
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    from omegaconf import OmegaConf, open_dict
    base = base_dir()
    cfg = OmegaConf.load(base / "model_config.yaml")
    with open_dict(cfg):
        cfg.tokenizer.dir = str(base)
        for key in ("model_path", "vocab_path", "spe_tokenizer_vocab"):
            cfg.tokenizer[key] = str(base / str(cfg.tokenizer[key]).removeprefix("nemo:"))
        for ds in ("train_ds", "validation_ds", "test_ds"):
            cfg.pop(ds, None)
        cfg.encoder.update(n_layers=2, d_model=64, n_heads=2, subsampling_conv_channels=16)
        cfg.model_defaults.update(enc_hidden=64, pred_hidden=32, joint_hidden=32)
        cfg.decoder.prednet.update(pred_hidden=32, pred_rnn_layers=2)
        cfg.joint.jointnet.update(joint_hidden=32, encoder_hidden=64, pred_hidden=32)
        cfg.decoding.greedy.use_cuda_graph_decoder = False
    torch.manual_seed(seed)
    return EncDecRNNTBPEModel(cfg=cfg).float()
