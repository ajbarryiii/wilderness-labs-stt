"""Versioned, conservative presets; committed configuration contains no secrets."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from .storage import ROOT, ensure_artifact_path


def source(dataset, config, split, revision, weight=1.0, **columns):
    return {"id": dataset, "config": config, "split": split, "revision": revision,
            "text_column": "text", "audio_column": "audio", "id_column": "id",
            "speaker_column": "speaker_id", "weight": weight, "license": "cc-by-4.0", **columns}


LIBRI = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
AMI = "46f28f2503e2ec48f8867a84eef356c70476beab"


def preset(name="pilot"):
    if name not in {"pilot", "full", "smoke"}:
        raise ValueError("Preset must be pilot, full, or smoke")
    train_sources = [source("openslr/librispeech_asr", "all", split, LIBRI, weight)
                     for split, weight in [("train.clean.100", .15), ("train.clean.360", .35), ("train.other.500", .45)]]
    train_sources.append(source("edinburghcstr/ami", "ihm", "train", AMI, .05, id_column="audio_id"))
    validation = [source("openslr/librispeech_asr", "all", split, LIBRI)
                  for split in ["validation.clean", "validation.other"]]
    validation.append(source("edinburghcstr/ami", "ihm", "validation", AMI, id_column="audio_id"))
    cfg = {
        "schema": 1, "seed": 20260913,
        "model": {"d_model": 256, "ff_dim": 1024, "num_layers": 4, "num_heads": 4,
                  "stem_channels": 64, "vocab_size": 2048, "n_mels": 80, "chunk_size": 4,
                  "left_context": 64, "depthwise_kernel": 9, "dropout": .1,
                  "activation_checkpointing": True, "center_weights": False},
        "tokenizer": {"kind": "sentencepiece", "path": None, "vocab_size": 2048,
                      "max_samples": 30000},
        "data": {"train_sources": train_sources, "validation_sources": validation,
                 "shuffle_buffer": 32, "min_seconds": .25, "max_seconds": 20,
                 "max_consecutive_bad": 100, "max_rejection_fraction": .5,
                 "rejection_fraction_min_samples": 100, "validation_per_source": 32},
        "training": {"device": "cuda", "max_steps": 2000, "max_audio_hours": 200,
                     "max_wall_hours": 24, "batch_audio_seconds": 40, "max_batch_size": 4,
                     "grad_accumulation": 4, "lr": .0002, "min_lr": .00002,
                     "warmup_steps": 100, "betas": [.9, .95], "weight_decay": .01,
                     "cooldown_start_fraction": .7, "cooldown_lr_multiplier": .5,
                     "grad_clip": 1.0, "eval_every": 100, "checkpoint_every": 100,
                     "log_every": 10, "validation_examples": 96, "minimum_free_disk_gib": 30},
        "quantization": {"weight_start_step": 100, "weight_ramp_steps": 300,
                         "activation_start_step": 400, "activation_ramp_steps": 600},
        "augmentation": {"enabled": True, "start_step": 100, "freq_masks": 2,
                         "freq_width": 8, "time_masks": 2, "max_time_width": 20,
                         "max_time_fraction": .05},
        "health": {"loss_window": 50, "min_train_steps": 100,
                   "loss_explosion_factor": 5, "loss_explosion_patience": 3,
                   "zero_grad_patience": 50, "phase_grace_steps": 20,
                   "val_patience": 3, "arm_wer": .8, "arm_empty_fraction": .2,
                   "wer_deterioration_factor": 2, "wer_deterioration_absolute": .25,
                   "empty_collapse_fraction": .9, "blank_collapse_fraction": .995},
        "notifications": {"email_to": "ajbarryiii@gmail.com", "email_required": True,
                          "email_enabled": True, "desktop": True},
        "supervision": {"timeout_seconds": 900, "terminate_grace_seconds": 30,
                        "stage_timeouts": {"stream_read": 300, "validation": 1800,
                                           "validation_load": 1800, "checkpoint": 1800,
                                           "checkpoint_load": 1800}},
    }
    if name == "full":
        cfg["model"].update(d_model=1024, ff_dim=4096, num_layers=20, num_heads=16, stem_channels=256)
        for item, weight in zip(train_sources, [.04, .08, .08, .05]):
            item["weight"] = weight
        train_sources.extend([
            source("MLCommons/peoples_speech", "clean", "train", "f10597c5d3d3a63f8b6827701297c3afdf178272", .25,
                   speaker_column=None, license="CC-BY (source-dependent version)"),
            source("espnet/yodas-granary", "English", "asr_only", "969944574ea3f37890beaf67ea651e160cfaf043", .5,
                   id_column="utt_id", speaker_column="original_audio_id", license="cc-by-3.0"),
        ])
        cfg["tokenizer"]["max_samples"] = 200000
        cfg["data"]["validation_per_source"] = 128
        cfg["training"].update(max_steps=100000, max_audio_hours=10000, max_wall_hours=720,
                               grad_accumulation=8, warmup_steps=1000, eval_every=250,
                               checkpoint_every=100, validation_examples=384, minimum_free_disk_gib=50)
        cfg["quantization"].update(weight_start_step=2000, weight_ramp_steps=8000,
                                   activation_start_step=10000, activation_ramp_steps=10000)
    if name == "smoke":
        cfg["model"].update(d_model=32, ff_dim=128, num_layers=2, num_heads=4,
                            stem_channels=16, dropout=0.0, activation_checkpointing=False)
        # Replaced with pinned local Parquet sources by the smoke command.
        cfg["data"].update(train_sources=[], validation_sources=[], shuffle_buffer=4,
                           validation_per_source=2, min_seconds=.1, max_seconds=5)
        cfg["tokenizer"] = {"kind": "character", "alphabet": " abcdefghijklmnopqrstuvwxyz'0123456789.,?-"}
        cfg["model"]["vocab_size"] = len(cfg["tokenizer"]["alphabet"]) + 1
        cfg["training"].update(device="cpu", max_steps=4, max_audio_hours=1, max_wall_hours=1,
                               batch_audio_seconds=4, max_batch_size=2, grad_accumulation=1,
                               warmup_steps=1, eval_every=2, checkpoint_every=1, log_every=1,
                               validation_examples=2, minimum_free_disk_gib=1)
        cfg["quantization"].update(weight_start_step=0, weight_ramp_steps=1,
                                   activation_start_step=1, activation_ramp_steps=1)
        cfg["notifications"].update(email_enabled=False, email_required=False, desktop=False)
        cfg["augmentation"]["enabled"] = False
    return copy.deepcopy(cfg)


def validate_config(cfg, *, require_sources=True):
    if cfg.get("schema") != 1:
        raise ValueError("Unsupported configuration schema")
    from .model import ModelConfig
    ModelConfig.from_dict(cfg["model"])
    if cfg["model"].get("n_mels", 80) != 80:
        raise ValueError("The pipeline requires 80 mel bins")
    for key, expected in {"n_mels": 80, "hop_length": 160, "sample_rate": 16000}.items():
        if cfg.get("features", {}).get(key, expected) != expected:
            raise ValueError(f"The pipeline requires features.{key}={expected}")
    training = cfg["training"]
    for key in ["max_steps", "batch_audio_seconds", "max_batch_size", "grad_accumulation",
                "lr", "grad_clip", "eval_every", "checkpoint_every", "log_every"]:
        if not isinstance(training[key], (int, float)) or not math.isfinite(training[key]) or training[key] <= 0:
            raise ValueError(f"training.{key} must be finite and positive")
    if not 0 <= training["warmup_steps"] < training["max_steps"]:
        raise ValueError("warmup_steps must be smaller than max_steps")
    data = cfg["data"]
    if not 0 < data["min_seconds"] <= data["max_seconds"]:
        raise ValueError("Invalid audio duration range")
    if require_sources and (not data["train_sources"] or not data["validation_sources"]):
        raise ValueError("Both training and separate validation sources are required")
    train_keys = {(s["id"], s.get("config"), s["split"]) for s in data["train_sources"]}
    val_keys = {(s["id"], s.get("config"), s["split"]) for s in data["validation_sources"]}
    if train_keys & val_keys:
        # Local fixtures can use two explicitly separate files under the same builder.
        overlap = train_keys & val_keys
        if any(key[0] != "parquet" for key in overlap):
            raise ValueError("Training and validation contain the same dataset split")
        train_files = {str(s.get("data_files")) for s in data["train_sources"] if s["id"] == "parquet"}
        val_files = {str(s.get("data_files")) for s in data["validation_sources"] if s["id"] == "parquet"}
        if train_files & val_files:
            raise ValueError("Training and validation use the same local files")
    for item in data["train_sources"] + data["validation_sources"]:
        if not item.get("license"):
            raise ValueError("Every source must record its applicable license")
        if not math.isfinite(item.get("weight", 1)) or item.get("weight", 1) <= 0:
            raise ValueError("Source sampling weights must be finite and positive")
    for value in cfg["quantization"].values():
        if not isinstance(value, int) or value < 0:
            raise ValueError("Quantization schedule values must be nonnegative integer steps")
    return cfg


def load_config(path):
    return validate_config(json.loads(Path(path).read_text()))
