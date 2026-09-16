"""Prepare, inspect and supervise binary speech training; no implicit long run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shlex
import sys
import time

from .storage import (ROOT, append_json, atomic_torch_save, check_free_space, configure_environment,
                      digest, ensure_artifact_path, heartbeat, object_digest, write_json)


def read_environment(path):
    """Read a private KEY=value file, never execute shell code or print secrets."""
    path = Path(path).expanduser()
    if path.stat().st_mode & 0o077:
        raise ValueError("Email environment file must be private: chmod 600 FILE")
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.removeprefix("export ").partition("=")
        if not sep or not key.startswith("BINARY_STT_") or not key.replace("_", "").isalnum():
            raise ValueError("Environment file accepts only BINARY_STT_* assignments")
        fields = shlex.split(value, comments=True)
        if len(fields) > 1:
            raise ValueError("Quote environment values containing spaces")
        os.environ[key] = fields[0] if fields else ""


def configuration(args):
    from .config import load_config, preset, validate_config
    cfg = load_config(args.config) if getattr(args, "config", None) else preset(args.preset)
    return validate_config(cfg, require_sources=args.preset != "smoke")


def runtime_versions():
    versions = {"python": sys.version, "executable": sys.executable}
    for package in ["torch", "torchaudio", "datasets", "huggingface_hub", "sentencepiece", "soundfile", "numpy",
                    "pyarrow", "fsspec"]:
        versions[package] = importlib.metadata.version(package)
    return versions


def make_tokenizer(cfg):
    from .tokenizer import CharacterTokenizer, SpeechTokenizer
    spec = cfg["tokenizer"]
    if spec["kind"] == "character":
        return CharacterTokenizer(spec["alphabet"], casefold=spec.get("casefold", True))
    return SpeechTokenizer(spec["path"], casefold=spec.get("casefold", True))


def stream_options(cfg):
    keys = ["shuffle_buffer", "min_seconds", "max_seconds", "max_consecutive_bad",
            "max_rejection_fraction", "rejection_fraction_min_samples"]
    return {**{key: cfg["data"][key] for key in keys if key in cfg["data"]},
            "casefold": cfg["tokenizer"].get("casefold", True)}


def prepare(cfg, run_dir):
    """Freeze tokenizer and small validation panel; stream training audio later."""
    from .config import validate_config
    from .data import StreamingSpeechDataset, iter_source_texts, resolve_sources
    from .tokenizer import train_tokenizer
    run_dir = ensure_artifact_path(run_dir)
    if (run_dir / "prepared.json").exists() or (run_dir / "config.json").exists():
        raise FileExistsError("Run configuration already exists; use a new run directory")
    check_free_space(cfg["training"]["minimum_free_disk_gib"])
    run_dir.mkdir(parents=True, exist_ok=True)
    heartbeat(run_dir, "prepare")
    write_json(run_dir / "status.json", {"status": "preparing", "training_started": False})
    cfg["data"]["train_sources"] = resolve_sources(cfg["data"]["train_sources"])
    cfg["data"]["validation_sources"] = resolve_sources(cfg["data"]["validation_sources"])
    validate_config(cfg)
    if cfg["tokenizer"]["kind"] == "sentencepiece":
        if cfg["tokenizer"].get("path"):
            cfg["tokenizer"]["path"] = str(ensure_artifact_path(cfg["tokenizer"]["path"]))
        else:
            heartbeat(run_dir, "tokenizer")
            texts = iter_source_texts(cfg["data"]["train_sources"],
                                      max_records=cfg["tokenizer"]["max_samples"], seed=cfg["seed"],
                                      casefold=cfg["tokenizer"].get("casefold", True))
            try:
                tokenizer = train_tokenizer(texts, run_dir / "tokenizer", vocab_size=cfg["model"]["vocab_size"],
                                            max_samples=cfg["tokenizer"]["max_samples"], seed=cfg["seed"],
                                            casefold=cfg["tokenizer"].get("casefold", True))
            finally:
                texts.close()
            cfg["tokenizer"]["path"] = str(tokenizer.path)
        cfg["tokenizer"]["sha256"] = digest(cfg["tokenizer"]["path"])
    tokenizer = make_tokenizer(cfg)
    if tokenizer.vocab_size != cfg["model"]["vocab_size"]:
        raise ValueError("Tokenizer vocabulary does not match model output count")
    validation = []
    rejection = {}
    content_ids = set()
    wanted = cfg["data"]["validation_per_source"]
    options = {**stream_options(cfg), "shuffle_buffer": 1}
    for index, src in enumerate(cfg["data"]["validation_sources"]):
        stream = StreamingSpeechDataset([src], seed=cfg["seed"] + index, repeat=False, **options)
        kept = attempts = 0
        try:
            for example in stream:
                heartbeat(run_dir, "validation_prepare", source=index, accepted=kept)
                attempts += 1
                ids = tokenizer.encode(example["text"])
                frames = (example["audio"].numel() + 159) // 160
                encoded_frames = (frames + 7) // 8
                required = len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))
                if not ids or required > encoded_frames or example["content_id"] in content_ids:
                    rejection[str(index)] = rejection.get(str(index), 0) + 1
                else:
                    content_ids.add(example["content_id"])
                    validation.append(example)
                    kept += 1
                if kept >= wanted:
                    break
                if attempts >= max(100, wanted * 20):
                    raise RuntimeError(f"Cannot build CTC-feasible validation panel for source {index}")
        finally:
            stream.close()
        if kept != wanted:
            raise RuntimeError(f"Validation source {index} ended before {wanted} valid examples")
    atomic_torch_save(run_dir / "validation.pt", validation)
    write_json(run_dir / "validation_manifest.json", {
        "examples": [{key: value for key, value in row.items() if key != "audio"} for row in validation],
        "rejected": rejection,
        "scope": "Official held-out splits plus exact decoded-PCM duplicate exclusion; not a global overlap audit.",
    })
    write_json(run_dir / "config.json", cfg)
    source_hashes = {path.name: digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    write_json(run_dir / "prepared.json", {
        "schema": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": digest(run_dir / "config.json"), "validation_sha256": digest(run_dir / "validation.pt"),
        "source_hashes": source_hashes, "runtime": runtime_versions(),
        "validation_examples": len(validation), "training_started": False,
    })
    write_json(run_dir / "status.json", {"status": "prepared", "training_started": False})
    return {"run_dir": str(run_dir), "status": "prepared", "validation_examples": len(validation)}


def verify_prepared(run_dir):
    run_dir = ensure_artifact_path(run_dir)
    receipt = json.loads((run_dir / "prepared.json").read_text())
    for name, key in [("config.json", "config_sha256"), ("validation.pt", "validation_sha256")]:
        if digest(run_dir / name) != receipt[key]:
            raise ValueError(f"Prepared {name} changed; prepare a fresh run")
    for name, checksum in receipt["source_hashes"].items():
        if digest(Path(__file__).parent / name) != checksum:
            raise ValueError(f"Training source changed since preparation: {name}; prepare a fresh run")
    if runtime_versions() != receipt["runtime"]:
        raise ValueError("Runtime changed since preparation; prepare a fresh run")
    return json.loads((run_dir / "config.json").read_text())


def create_fixture(run_dir):
    """Small generated audio fixture exercising the real HF Parquet streaming API."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import soundfile as sf
    from .config import preset
    from .data import resolve_sources
    run_dir = ensure_artifact_path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    for split, count, offset in [("train", 12, 0), ("validation", 4, 100)]:
        rows = []
        for i in range(count):
            t = np.arange(32000, dtype=np.float32) / 16000
            signal = .2 * np.sin(2 * np.pi * (180 + (i + offset) * 13) * t)
            buffer = io.BytesIO()
            sf.write(buffer, signal, 16000, format="FLAC")
            rows.append({"audio": {"bytes": buffer.getvalue(), "path": None}, "text": ["go", "no", "two"][i % 3],
                         "id": f"{split}-{i}", "speaker_id": f"fixture-{split}"})
        path = ensure_artifact_path(run_dir / f"{split}.parquet")
        pq.write_table(pa.Table.from_pylist(rows), path)
        sources.append({"id": "parquet", "config": None, "split": split,
                        "data_files": {split: [str(path)]}, "revision": "", "audio_column": "audio",
                        "text_column": "text", "id_column": "id", "speaker_column": "speaker_id",
                        "weight": 1.0, "license": "generated test fixture"})
    sources = resolve_sources(sources)
    cfg = preset("smoke")
    cfg["data"].update(train_sources=[sources[0]], validation_sources=[sources[1]])
    return cfg


def run_training(run_dir, resume=False, stop_after=None):
    from .notifications import validate_notification_config
    from .supervisor import run_supervised
    cfg = verify_prepared(run_dir)
    validate_notification_config(cfg["notifications"])
    command = [str(Path(__file__).parent / "python"), "-m", "binary_stt", "worker", "--run-dir", str(run_dir)]
    if resume:
        command.append("--resume")
    if stop_after is not None:
        command += ["--stop-after", str(stop_after)]
    result = run_supervised(command, run_dir, notification_config=cfg["notifications"], **cfg["supervision"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", help="Private SMTP environment file (mode 0600)")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["inspect", "write-config", "prepare", "probe-data"]:
        command = commands.add_parser(name)
        command.add_argument("--preset", choices=["pilot", "full", "smoke"], default="pilot")
        command.add_argument("--config", type=Path)
        if name in {"prepare", "probe-data"}:
            command.add_argument("--run-dir", type=Path, required=True)
        if name == "probe-data":
            command.add_argument("--per-source", type=int, default=1)
        if name == "write-config":
            command.add_argument("--output", type=Path, required=True)
    for name in ["train", "worker", "status", "stop", "test-email", "smoke"]:
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
        if name in {"train", "worker"}:
            command.add_argument("--resume", action="store_true")
            command.add_argument("--stop-after", type=int)
        if name == "smoke":
            command.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args(argv)
    if args.env_file:
        read_environment(args.env_file)
    configure_environment()
    if args.command in {"inspect", "write-config"}:
        cfg = configuration(args)
        if args.command == "write-config":
            # Configs generated by this command live on the data disk with the run.
            write_json(args.output, cfg)
            print(args.output)
        else:
            import torch
            from .model import BinaryCTCModel, BinaryLinear
            with torch.device("meta"):
                model = BinaryCTCModel(cfg["model"])
            binary = sum(m.weight.numel() for m in model.modules() if isinstance(m, BinaryLinear))
            total = sum(p.numel() for p in model.parameters())
            print(json.dumps({"preset": args.preset, "parameters": total, "binary_matrix_parameters": binary,
                              "other_parameters": total - binary, "training": cfg["training"],
                              "quantization": cfg["quantization"], "runtime": runtime_versions()}, indent=2))
        return 0
    run_dir = ensure_artifact_path(args.run_dir)
    if args.command == "prepare":
        print(json.dumps(prepare(configuration(args), run_dir), indent=2))
    elif args.command == "probe-data":
        from .data import StreamingSpeechDataset, resolve_sources
        cfg = configuration(args)
        if args.per_source <= 0:
            raise ValueError("--per-source must be positive")
        records = []
        sources = resolve_sources(cfg["data"]["train_sources"] + cfg["data"]["validation_sources"])
        for src in sources:
            stream = StreamingSpeechDataset([src], repeat=False, **{**stream_options(cfg), "shuffle_buffer": 1})
            try:
                for index, row in enumerate(stream):
                    records.append({key: row[key] for key in ["id", "source", "seconds", "content_id"]})
                    if index + 1 >= args.per_source:
                        break
            finally:
                stream.close()
        write_json(run_dir / "data-probe.json", {"sources": sources, "records": records, "runtime": runtime_versions()})
        print(json.dumps({"sources": len(sources), "examples": len(records), "report": str(run_dir / "data-probe.json")}, indent=2))
    elif args.command == "train":
        return run_training(run_dir, args.resume, args.stop_after)
    elif args.command == "worker":
        from .train import train
        if os.environ.get("BINARY_STT_SUPERVISED") != "1":
            raise ValueError("Use the train command so the external watchdog is active")
        verify_prepared(run_dir)
        result = train(run_dir, resume=args.resume, stop_after=args.stop_after)
        print(json.dumps(result, indent=2))
        return 0 if result["state"] in {"completed", "stopped"} else 1
    elif args.command == "status":
        result = {name: json.loads((run_dir / name).read_text()) for name in
                  ["status.json", "supervisor_status.json", "heartbeat.json"] if (run_dir / name).exists()}
        print(json.dumps(result, indent=2))
    elif args.command == "stop":
        write_json(run_dir / "STOP", {"requested_at": time.time()})
        print("Stop requested; the worker will finish its current safe boundary.")
    elif args.command == "test-email":
        from .notifications import Notifier
        cfg = {"email_to": "ajbarryiii@gmail.com", "email_required": True, "desktop": False}
        if (run_dir / "config.json").exists():
            cfg = json.loads((run_dir / "config.json").read_text())["notifications"]
        result = Notifier(run_dir, cfg).test()
        write_json(run_dir / "email-test.json", result)
        print(json.dumps({"email_to": cfg["email_to"], "delivery_failures": result["delivery_failures"]}, indent=2))
        return 1 if result["delivery_failures"] else 0
    elif args.command == "smoke":
        cfg = create_fixture(run_dir)
        cfg["training"]["device"] = args.device
        prepare(cfg, run_dir)
        result = run_training(run_dir, stop_after=2)
        if result:
            return result
        result = run_training(run_dir, resume=True)
        if result:
            return result
        status = json.loads((run_dir / "status.json").read_text())
        print(json.dumps({"run_dir": str(run_dir), "status": status, "scope": "Synthetic audio through real HF Parquet streaming; no recognition accuracy claim."}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
