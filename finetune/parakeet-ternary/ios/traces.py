"""Replay traces: the FP32 reference with the real B0 weights decodes every clip; committed traces.json.

DESIGN.md "Replay trace". NixOS, CPU only, through the memory-capped wrapper (unit peak 4.1 GB: one dense FP32
reference model plus the memory-mapped checkpoint; about 45 s for the 82 clips):

  ../heavy ios-wp2-traces --mem-max 10G --runtime 40min --wait -- env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 \
      MKL_NUM_THREADS=4 <repo>/finetune/parakeet-ternary/python <repo>/finetune/parakeet-ternary/ios/traces.py

B0 = nvidia/parakeet-tdt-0.6b-v2 at the pinned revision (paths.MODEL_FILE, SHA-256 in its lock.json). Its
weights are loaded by WP1's models.b0() (reference.load_weights; model_config.yaml and the tokenizer come from the
.nemo tar, no NeMo import; the reference is gated against NeMo by tests/test_reference_nemo.py).
Each clip (clips.json, PCM from `clips.py materialize`, SHA-256 checked) runs unpadded at batch 1:
reference preprocessor -> encoder -> reference.greedy_decode (NeMo greedy_batch label-looping semantics:
zero LSTM state, blank/SOS start, max_symbols 10). The trace is then replayed on the same model as a
self-check (every replayed argmax must equal the recorded decision).

traces.json (one clip per line, column-oriented, booleans as 0/1): per clip every step's frame,
prediction-net input token, token (1024 = blank), duration value, emitted, pred_updated,
symbols_at_frame, forced_advance, advance; num_frames (valid encoder frames), the emitted tokens and the
B0 transcript (SentencePiece decode of the .nemo tokenizer, as NeMo's ids_to_text). WER: Whisper-normalized
corpus WER of the natural clips vs the LibriSpeech transcripts (evaluate.score / wer_summary, the parent
experiment's scorer).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import clips as clipmod  # noqa: E402
import reference  # noqa: E402

TRACES = HERE / "traces.json"
BOOL_FIELDS = ("emitted", "pred_updated", "forced_advance")
MAX_BYTES = 1_000_000


def load_b0(model_file: Path) -> tuple[reference.ParakeetReference, dict, "object", dict]:
    """B0 as WP1's models.b0() builds it (memory-mapped cached checkpoint, reference.load_weights), plus the .nemo's
    own model config and SentencePiece tokenizer (read from the tar, no NeMo import)."""
    import sentencepiece as spm
    import yaml

    import models

    with tarfile.open(model_file, "r:") as tar:
        members = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}
        config = yaml.safe_load(tar.extractfile(members["model_config.yaml"]).read())
        tok_name = next(k for k in members if k.endswith("_tokenizer.model"))
        sp = spm.SentencePieceProcessor(model_proto=tar.extractfile(members[tok_name]).read())
    model = models.b0(model_file)
    if model.cfg != reference.Config.from_model_config(config):
        raise ValueError("models.b0 config differs from the .nemo's model_config.yaml")
    return model, config, sp, model.provenance["load"]


def trace_record(clip: dict, trace: reference.Trace, text: str) -> dict:
    rec = {"id": clip["id"], "sha256": clip["sha256"], "samples": clip["length"], "mel_frames": clip["mel_frames"],
           "num_frames": trace.num_frames, "steps": len(trace)}
    for key in reference.TRACE_FIELDS:
        values = getattr(trace, key)
        rec[key] = [int(v) for v in values]
    rec["tokens"] = trace.tokens
    rec["text"] = text
    return rec


def record_to_trace(rec: dict) -> reference.Trace:
    return reference.Trace(num_frames=rec["num_frames"],
                           **{k: [bool(v) for v in rec[k]] if k in BOOL_FIELDS else list(rec[k])
                              for k in reference.TRACE_FIELDS})


def wer_block(pairs: list[tuple[str, str]]) -> dict:
    """Whisper-normalized corpus WER of (reference, hypothesis) pairs (the parent experiment's scorer)."""
    import evaluate

    records = [evaluate.score(ref, hyp) for ref, hyp in pairs]
    out = evaluate.wer_summary(records)
    out["utterances"] = len(records)
    return out


def write_traces(doc: dict, path: Path = TRACES) -> int:
    head = {k: v for k, v in doc.items() if k != "clips"}
    text = json.dumps(head, indent=1)[:-2] + ',\n "clips": [\n'
    text += ",\n".join("  " + json.dumps(c, separators=(",", ":")) for c in doc["clips"])
    text += "\n ]\n}\n"
    path.write_text(text)
    return len(text.encode())


def main() -> None:
    import paths

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pcm", default="/mnt/hd/wilderness-labs-stt/parakeet-ios/clips")
    parser.add_argument("--out", default=str(TRACES))
    args = parser.parse_args()
    paths.require_mount()
    torch.set_grad_enabled(False)
    t0 = time.time()
    manifest = clipmod.load_manifest()
    lock = json.loads((paths.MODEL_DIR / "lock.json").read_text())
    model, config, sp, load_report = load_b0(paths.MODEL_FILE)
    print(f"loaded B0 in {time.time() - t0:.1f}s: {load_report}", flush=True)
    records, natural_pairs, per_bucket, replay_ok = [], [], {}, True
    for clip in manifest["clips"]:
        pcm = clipmod.read_pcm(Path(args.pcm), clip)
        if clipmod.pcm_sha256(pcm) != clip["sha256"]:
            raise ValueError(f"{clip['id']}: PCM SHA-256 differs from clips.json")
        audio = torch.from_numpy(pcm.astype(np.float32))[None]
        lengths = torch.tensor([len(pcm)])
        features, feat_len = model.preprocessor(audio, lengths)
        enc, enc_len = model.encoder(features, feat_len)
        if int(feat_len[0]) != clip["mel_frames"] or int(enc_len[0]) != clip["encoder_frames"]:
            raise ValueError(f"{clip['id']}: frames {int(feat_len[0])}/{int(enc_len[0])} differ from clips.json")
        trace = reference.greedy_decode(model, enc, enc_len)[0]
        steps = reference.replay(model, enc, int(enc_len[0]), trace)
        decisions_equal = (steps.argmax_token.tolist() == trace.token
                           and steps.argmax_duration.tolist() == trace.duration)
        replay_ok &= decisions_equal
        text = sp.decode_ids(trace.tokens)
        rec = trace_record(clip, trace, text)
        if record_to_trace(json.loads(json.dumps(rec))) != trace:
            raise ValueError(f"{clip['id']}: trace does not round-trip through JSON")
        records.append(rec)
        if clip["kind"] == "natural":
            natural_pairs.append((clip["transcript"], text))
            per_bucket.setdefault(str(clip["bucket"]), []).append((clip["transcript"], text))
        print(f"{clip['id']}: frames {trace.num_frames} steps {len(trace)} tokens {len(trace.tokens)} "
              f"forced {sum(trace.forced_advance)} replay_ok {decisions_equal} | {text}", flush=True)
    wer = {"natural": wer_block(natural_pairs), "per_bucket": {b: wer_block(p) for b, p in per_bucket.items()}}
    import evaluate

    doc = {
        "schema": 1,
        "description": "Greedy TDT replay traces of B0 (FP32 reference, CPU) on every clip of clips.json; see traces.py",
        "model": {"label": "B0", "id": paths.MODEL_ID, "revision": paths.MODEL_REVISION,
                  "nemo_sha256": lock["files"]["parakeet-tdt-0.6b-v2.nemo"], "load_report": load_report},
        "decoding": {"strategy": config["decoding"]["strategy"], "semantics": "NeMo greedy_batch label looping "
                     "(reference.run_steps)", "max_symbols": model.cfg.max_symbols, "durations": list(model.cfg.durations),
                     "blank": model.cfg.blank, "init": "zero LSTM state, blank/SOS input, reset per utterance",
                     "batch": "1, unpadded"},
        "clips_json_sha256": hashlib.sha256((HERE / "clips.json").read_bytes()).hexdigest(),
        "reference_py_sha256": hashlib.sha256((HERE / "reference.py").read_bytes()).hexdigest(),
        "fields": {"frame": "encoder frame of the joint evaluation", "pred_input": "token fed to the prediction net "
                   "whose output the step used (1024 = blank/SOS)", "token": "chosen token (1024 = blank)",
                   "duration": "chosen duration value (before the blank 0 -> 1 rule)", "emitted": "token != blank",
                   "pred_updated": "prediction net run on token after the step", "symbols_at_frame": "tokens emitted "
                   "at the frame of the latest emission", "forced_advance": "max-symbols rule added a frame",
                   "advance": "frames advanced after the step"},
        "self_check": {"replay_decisions_equal_all_clips": bool(replay_ok)},
        "wer": wer,
        "scorer": evaluate.NORMALIZER,
        "totals": {"clips": len(records), "steps": sum(r["steps"] for r in records),
                   "tokens": sum(len(r["tokens"]) for r in records),
                   "forced_advances": sum(sum(r["forced_advance"]) for r in records)},
        "environment": {"torch": torch.__version__, "threads": torch.get_num_threads(), "platform": platform.platform(),
                        "python": platform.python_version()},
        "clips": records,
    }
    size = write_traces(doc, Path(args.out))
    peak_mb = __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss / 1024
    print(json.dumps({"traces_bytes": size, "wer": wer["natural"], "replay_ok": replay_ok, "totals": doc["totals"],
                      "peak_rss_mb": round(peak_mb), "seconds": round(time.time() - t0, 1)}), flush=True)
    if size >= MAX_BYTES:
        raise SystemExit(f"traces.json is {size} bytes >= {MAX_BYTES}; split it")
    if not replay_ok:
        raise SystemExit("replay self-check failed")


if __name__ == "__main__":
    main()
