"""Fixture world for test_keab_test (no tests here): Kea-B′'s test-set scoring (keab-prime-test plan) on CPU.

World builds a complete, small, self-consistent set of inputs under a temporary directory and installs it over
keab_test_paths (its pins recomputed from the fixture files), paths.MANIFESTS and keab_test's seams:
- 8 test manifests and 5 dev manifests of 3 utterances (the AMI test set with AMI ids);
- Kea-B′'s run directory (view, bundle, model.pt, model.json, summary.json, config.json with the extra sources' clip
  pools), stage 4's recorded x1 dev scores and runtime.json, stage 4's results.json and RESULTS-keab-prime.md;
- the reused arms B0, M1 and MB1 (MB1-test with its complete attempt record), MB1's export and its recorded dev check;
- a Common Voice release (six tsvs, an ESB test parquet file), its selection table and selection.json (with the
  held-out counts of these inputs, derived by hand in CV_COUNTS), and its clip store; an AMI SDM pair store;
- the private plan file whose Status line holds keab_test_paths.py's SHA-256.
Model loading and decoding are fakes: training/deployed.py's load_x1 / score_records / close / require_wrapper (x1),
binary_export.load_export and evaluate.evaluate_set (fp), whose WERs are the world's known values; the fake
evaluate_set calls the model's encoder once per decoded batch (evaluate.make_batches), as the real one does, so the
K-fp encoder check sees what it would see. binary_train's decoder-mode helpers run for real against a stand-in
decoding computer. The GPU lock is a fake that records whether it is held. The normalizer is the real one.

The real K-fp path: every manifest utterance has a real 16 kHz audio file (utterance i lasts 1 + i s, a quiet sine),
and TinyModel is a CPU stand-in with what the REAL evaluate.evaluate_set uses (FP32 parameters, preprocessor with its
featurizer, encoder called through nn.Module __call__, decoding with rnnt_decoder_predictions_tensor and the decoding
computer, cfg.decoding). Its encoder fills every row's padding frames with NaN (the check must look only at valid
frames) and, when poisoned, NaN over the valid frames of a clip that is the constant POISON; its decoder then emits
garbage text without raising (a clip that is the constant CRASH makes it raise). World.real_fp() installs it.
"""
from __future__ import annotations

import keab_test as kt  # noqa: I001  first: train_keab (train.py's settings; finetune/parakeet-binary on sys.path)

import contextlib
import copy
import hashlib
import inspect
import json
import os
import tempfile
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch
from omegaconf import OmegaConf
from torch import nn

import binary_export
import evaluate
import keab_test_paths as tp
import paths
import train_keab as tk
from training import deployed

TEST = list(paths.TEST_SETS)
DEV = list(paths.DEV_SETS)
REAL_EVALUATE_SET = evaluate.evaluate_set
CURRENT_SCORER = {k: hashlib.sha256(tp.scorer_path(k).read_bytes()).hexdigest() for k in tp.scorer_keys()}

# Known WER fractions.
B0_WER = {"librispeech_clean": 0.017, "librispeech_other": 0.032, "ami": 0.111, "earnings22": 0.112,
          "gigaspeech": 0.097, "spgispeech": 0.022, "voxpopuli": 0.060, "common_voice": 0.060}
M1_WER = {"librispeech_clean": 0.019, "librispeech_other": 0.040, "ami": 0.104, "earnings22": 0.120,
          "gigaspeech": 0.100, "spgispeech": 0.030, "voxpopuli": 0.065, "common_voice": 0.080}
MB1_WER = {"librispeech_clean": 0.020, "librispeech_other": 0.044, "ami": 0.110, "earnings22": 0.125,
           "gigaspeech": 0.102, "spgispeech": 0.031, "voxpopuli": 0.066, "common_voice": 0.090}
KX1_WER = {"librispeech_clean": 0.019, "librispeech_other": 0.041, "ami": 0.100, "earnings22": 0.121,
           "gigaspeech": 0.099, "spgispeech": 0.029, "voxpopuli": 0.064, "common_voice": 0.070}
KFP_WER = {"librispeech_clean": 0.0195, "librispeech_other": 0.0415, "ami": 0.101, "earnings22": 0.122,
           "gigaspeech": 0.0995, "spgispeech": 0.0295, "voxpopuli": 0.0645, "common_voice": 0.071}
# Stage 4's recorded x1 dev WERs of Kea-B′, MB1's recorded K-fp dev WERs, Kea-B′'s PyTorch fake-quant dev WERs.
X1_DEV = {"librispeech_dev_clean": 0.0204, "librispeech_dev_other": 0.0444, "ami_dev": 0.1076,
          "voxpopuli_dev": 0.0606, "yodas_dev": 0.0723}
MB1_DEV = {"librispeech_dev_clean": 0.0216, "librispeech_dev_other": 0.0469, "ami_dev": 0.1195,
           "voxpopuli_dev": 0.0606, "yodas_dev": 0.0670}
TORCH_DEV = {"librispeech_dev_clean": 0.0204, "librispeech_dev_other": 0.0443, "ami_dev": 0.1080,
             "voxpopuli_dev": 0.0604, "yodas_dev": 0.0723}
KFP_DEV = {"librispeech_dev_clean": 0.0205, "librispeech_dev_other": 0.0445, "ami_dev": 0.1078,
           "voxpopuli_dev": 0.0607, "yodas_dev": 0.0724}
TEXTS = ["Mister Smith's 2 dogs ate the colour-coded biscuits!", "Uh, we're going at 5:30 p.m., OK?",
         "It cost $3.50 — about twenty percent less."]
HYPS = ["Mr. Smith's two dogs ate the color coded biscuits.", "We are going at five thirty PM okay",
        "It cost three dollars fifty, about 20% less."]
DECODING_CONFIG = {"strategy": "greedy_batch", "model_type": "tdt", "durations": [0, 1, 2, 3, 4],
                   "greedy": {"max_symbols": 10},
                   "beam": {"beam_size": 2, "return_best_hypothesis": False, "score_norm": True,
                            "tsd_max_sym_exp": 50, "alsd_max_target_len": 2.0}}
POISON, CRASH = 0.5, -0.5          # constant clips the TinyModel turns into non-finite encoder output / a crash
FRAME = 1600                        # TinyModel: one encoder frame per 0.1 s
SIZES = {"bundle_tensor_bytes": 91_236_520, "mb1_file_bytes": 105_299_804, "m1_file_bytes": 180_797_564,
         "original_nemo_bytes": 2_472_222_720}

# --- the Common Voice release of the fixture -------------------------------------------------------------------------
# ESB test rows (id = the speaker hash, text, audio path). "esbOld" read e1.mp3; this release records e1.mp3 under
# client_id "spkNew" (validated.tsv): a changed id, caught only by the path join.
ESB = [("esbOld", "Have many acquaintances, but few friends.", "e1.mp3"),
       ("esbB", "It was a bright cold day in April.", "e2.mp3")]
TSVS = {
    "test.tsv": [("tt1", "t1.mp3", "Never judge a book (by its cover).")],
    "dev.tsv": [("dd1", "d1.mp3", "A dev sentence stays out of training.")],
    "validated.tsv": [("spkNew", "e1.mp3", "Have many acquaintances, but few friends."),
                      ("spkA", "v1.mp3", "The quick brown fox jumps over the dog."),
                      ("spkB", "v2.mp3", "Pack my box with five dozen liquor jugs."),
                      ("spkNew", "v3.mp3", "A sentence read by the changed speaker."),
                      ("spkC", "v4.mp3", "(Have) many acquaintances, but few friends."),
                      ("spkD", "t1.mp3", "A different sentence on a held-out clip."),
                      ("tt1", "v5.mp3", "The test speaker read this one too.")],
    "other.tsv": [("spkE", "o1.mp3", "Another unrelated sentence.")],
    "invalidated.tsv": [],
    "train.tsv": [("spkA", "v1.mp3", "The quick brown fox jumps over the dog.")],
}
# Held-out counts of these inputs under the production rules, derived by hand:
#   ESB: 2 rows, 2 paths {e1, e2}; test.tsv 1 row (t1, tt1); dev.tsv 1 row (d1, dd1).
#   held paths {e1, e2, t1, d1} = 4. Found in the six tsvs: e1 (validated, spkNew), t1 (test.tsv tt1 and validated
#   spkD), d1 (dev.tsv) = 3; ESB paths found: e1 = 1; with a changed id: e1 ({spkNew} has no ESB id) = 1.
#   speakers before the join {esbOld, esbB, tt1, dd1}; joined {spkNew, tt1, spkD, dd1}: added spkNew, spkD = 2;
#   held speakers 6. Sentences: the two ESB sentences (one form each); "Never judge a book (by its cover)." has two
#   forms (the normalizer drops the bracketed words, filters v3 keeps them); the dev sentence one: 5.
CV_COUNTS = {"esb_test": 2, "esb_test_paths": 2, "test.tsv": 1, "dev.tsv": 1, "held_paths_found_in_release": 3,
             "esb_paths_found_in_release": 1, "esb_paths_with_changed_id": 1, "speaker_ids_added_by_path_join": 2,
             "held_speakers": 6, "held_paths": 4, "held_sentences": 5}
# Selected clips of the passing world (none held out); the others are selection rows a test may accept.
ACCEPTED = ("v1.mp3", "v2.mp3")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n")


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def decoding(batch_size: int = 64, max_batch_seconds: float = 1200.0) -> dict:
    return {"method": "preprocessor -> encoder -> decoding.rnnt_decoder_predictions_tensor (transcribe() path)",
            "config": copy.deepcopy(DECODING_CONFIG),
            "precision": "float32, TF32 off (matmul and cuDNN), autocast disabled",
            "batch_size": batch_size, "max_batch_seconds": max_batch_seconds, "batch_order": "descending duration"}


def test_ids(name: str) -> list[str]:
    if name == "ami":
        return [f"AMI_EN2002a_H00_MEE073_{k:07d}_{k + 50:07d}" for k in (0, 100, 200)]
    return [f"{name}:{i}" for i in range(3)]


class FakeComputer:
    allow_cuda_graphs = True

    def __init__(self) -> None:
        self.cuda_graphs_mode = types.SimpleNamespace(value="full_graph")

    def force_cuda_graphs_mode(self, mode) -> None:
        self.cuda_graphs_mode = types.SimpleNamespace(value=str(mode))


class FakeModel:
    def __init__(self, label: str) -> None:
        self.label = label
        self.computer = FakeComputer()
        self.decoding = types.SimpleNamespace(decoding=types.SimpleNamespace(decoding_computer=self.computer))
        self.encoder = FakeEncoder()


class TinyEncoder(nn.Module):
    def __init__(self, poison: bool, extra_call: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.poison, self.extra_call = poison, extra_call

    def forward(self, audio_signal, length):
        enc = audio_signal.repeat(1, 4, 1) * self.weight          # [B, D=4, T]
        valid = torch.arange(enc.shape[-1])[None, :] < length[:, None]
        enc = enc.masked_fill(~valid[:, None, :], float("nan"))   # padding frames: never looked at
        if self.poison:
            for b in range(enc.shape[0]):
                n = int(length[b])
                if torch.all(audio_signal[b, 0, :n] == POISON):
                    enc[b, :, :n] = float("nan")
        return enc, length


class TinyModel(nn.Module):
    def __init__(self, label: str, poison: bool = False, extra_call: bool = False) -> None:
        super().__init__()
        self.label = label
        self.preprocessor = nn.Module()
        self.preprocessor.featurizer = types.SimpleNamespace(dither=1e-5, pad_to=16)
        self.encoder = TinyEncoder(poison, extra_call)
        self.cfg = OmegaConf.create({"decoding": copy.deepcopy(DECODING_CONFIG)})
        self.computer = FakeComputer()
        self.decoding = types.SimpleNamespace(decoding=types.SimpleNamespace(decoding_computer=self.computer),
                                              rnnt_decoder_predictions_tensor=self.decode)

    def forward(self, input_signal, input_signal_length):
        n = input_signal.shape[1] // FRAME
        feats = input_signal[:, :n * FRAME].reshape(input_signal.shape[0], n, FRAME).mean(-1)[:, None, :]
        lengths = input_signal_length // FRAME
        if self.encoder.extra_call:
            self.encoder(audio_signal=feats, length=lengths)
        return self.encoder(audio_signal=feats, length=lengths)

    def decode(self, encoder_output, encoded_lengths, return_hypotheses=False):
        out = []
        for b in range(encoder_output.shape[0]):
            n = int(encoded_lengths[b])
            row = encoder_output[b, :, :n]
            if torch.isfinite(row).all() and torch.all(row == CRASH):
                raise RuntimeError("simulated decoder failure")
            text = TEXTS[n // 10 - 1] if torch.isfinite(row).all() else "garbage garbage"
            out.append(types.SimpleNamespace(text=text))
        return out


class FakeEncoder(nn.Module):
    """The fake path's encoder: finite output, one call per decoded batch (the fake evaluate_set calls it)."""

    def forward(self, audio_signal, length):
        return audio_signal.repeat(1, 4, 1), length


class World:
    def __init__(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="keab-test-")
        self.root = Path(self.tmp.name)
        r = self.root
        self.manifests = r / "manifests"
        self.keab = r / "keab-prime"
        self.run = self.keab / "runs" / tp.RUN_NAME
        self.view = self.run / "export" / "view"
        self.bundle = self.run / "export" / "bundle"
        self.scores_x1 = self.keab / "scores" / tp.RUN_NAME / "x1"
        self.test_root = self.keab / "test"
        self.plans = r / "plans"
        self.arms = {"B0": r / "ternary-eval" / "b0-pretrained", "M1": r / "ternary-eval" / "M1-test",
                     "MB1": r / "binary-eval" / "MB1-test"}
        self.mb1_attempts = r / "binary-eval" / "MB1-test.attempts.jsonl"
        self.mb1_export = r / "binary-runs" / "main-binB-m1-lr2e-4" / "export"
        self.mb1_dev_check = r / "binary-eval" / "MB1-dev-check"
        self.cv = r / "corpora" / "cv27-en"
        self.raw = r / "corpora" / "common-voice-27-en" / "raw"
        self.esb_files = [r / "esb" / "common_voice" / "test-00000-of-00001.parquet"]
        self.ami_store = r / "corpora" / "ami-sdm" / "store" / "clips.jsonl"
        # stubs' behaviour
        self.pending: list[str] = []
        self.jobs: list[str] | None = []
        self.x1_test_wer, self.fp_test_wer = dict(KX1_WER), dict(KFP_WER)
        self.x1_dev_wer, self.mb1_dev_wer, self.keab_fp_dev_wer = dict(X1_DEV), dict(MB1_DEV), dict(KFP_DEV)
        self.invalid: set[tuple[str, str]] = set()      # (mode, set): a structurally invalid row
        self.empty: dict[tuple[str, str], int] = {}    # (mode, set): number of empty hypotheses
        self.fail_on: dict[str, str] = {}               # mode -> set whose decoding raises
        self.poison_models: set[str] = set()            # TinyModel labels ("MB1", "keab") whose encoder is poisoned
        self.extra_call = False                         # TinyModel calls its encoder twice per batch
        self.calls: list[dict] = []
        self.loads: list[dict] = []
        self.closed = 0
        self.lock_held = False
        self.lock_events: list[str] = []
        self.pins: dict = {}
        self.build()

    def close(self) -> None:
        self.tmp.cleanup()

    # ------------------------------------------------------------------ building
    def build(self) -> None:
        for name in TEST:
            self.write_manifest(self.manifests / f"test_{name}.jsonl", test_ids(name))
        for name in DEV:
            self.write_manifest(self.manifests / f"{name}.jsonl", [f"{name}:{i}" for i in range(3)])
        self.build_run()
        self.build_reused()
        self.build_cv()
        self.build_ami()
        self.repin()
        self.build_stage4()
        self.write_plan()

    def audio_path(self, name: str, i: int) -> Path:
        return self.root / "audio" / name / f"{i}.wav"

    def write_audio(self, name: str, i: int, constant: float | None = None) -> None:
        n = (1 + i) * paths.SAMPLE_RATE
        clip = (np.full(n, constant, np.float32) if constant is not None
                else (0.1 * np.sin(np.arange(n) * 0.05 * (1 + i))).astype(np.float32))
        path = self.audio_path(name, i)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, clip, paths.SAMPLE_RATE, subtype="PCM_16")

    def write_manifest(self, path: Path, ids: list[str]) -> None:
        name = path.stem.removeprefix("test_")
        for i in range(len(ids)):
            self.write_audio(name, i)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps({"id": uid, "audio_filepath": str(self.audio_path(name, i)),
                                            "duration": 1.0 + i, "text": TEXTS[i % 3]}) + "\n"
                                for i, uid in enumerate(ids)))

    def build_run(self) -> None:
        (self.view / "tokenizer").mkdir(parents=True)
        (self.bundle / "tokenizer").mkdir(parents=True)
        (self.view / "export.safetensors").write_bytes(b"fake Kea-B' scoring view")
        for rel in tp.VIEW_TOKENIZER:
            (self.view / rel).write_bytes(f"fake {rel}".encode())
            (self.bundle / rel).write_bytes(f"fake {rel}".encode())
        (self.bundle / "bundle.safetensors").write_bytes(b"fake Kea-B' 8-bit bundle")
        (self.run / "model.pt").write_bytes(b"fake latent model")
        model_sha, bundle_sha, view_sha = (sha(self.run / "model.pt"), sha(self.bundle / "bundle.safetensors"),
                                           sha(self.view / "export.safetensors"))
        files = {"export.safetensors": view_sha, **{rel: sha(self.view / rel) for rel in tp.VIEW_TOKENIZER}}
        write_json(self.view / "manifest.json", {
            "format": tp.VIEW_FORMAT, "file": tp.VIEW_FILE, "sha256": view_sha, "files": files,
            "config": {"decoding": copy.deepcopy(DECODING_CONFIG)},
            "bytes": {"file_bytes": 105_299_804, "original_nemo_bytes": SIZES["original_nemo_bytes"]},
            "extra": {"keab": {"bundle_sha256": bundle_sha, "bundle_format": tp.BUNDLE_FORMAT, "scope": "S2"},
                      "model_sha256": model_sha, "run": tp.RUN_NAME}})
        write_json(self.bundle / "manifest.json", {
            "format": tp.BUNDLE_FORMAT, "file": "bundle.safetensors", "sha256": bundle_sha,
            "bytes": {"tensor_bytes": SIZES["bundle_tensor_bytes"]},
            "extra": {"model_sha256": model_sha, "run": tp.RUN_NAME}})
        write_json(self.run / "model.json", {"sha256": model_sha, "run": tp.RUN_NAME,
                                             "export": {"bundle_sha256": bundle_sha, "view_sha256": view_sha}})
        self.write_config()
        write_json(self.scores_x1 / "runtime.json", {"runtime": {"export": str(self.view), "export_sha256": view_sha,
                                                                 "kernel": tp.X1_KERNEL, "parts": tp.X1_PARTS}})
        for name in DEV:
            manifest = self.manifests / f"{name}.jsonl"
            rows = []
            for i, rec in enumerate(evaluate.read_manifest(manifest)):
                rows.append({"i": i, "id": rec["id"], "hyp": HYPS[i % 3], "tokens": [i, 7], "finite": True,
                             "decode_error": None, "ref": rec["text"], **evaluate.score(rec["text"], HYPS[i % 3])})
            write_json(self.scores_x1 / f"{name}.json", {
                "set": name, "utterances": len(rows), "wer": {"wer": X1_DEV[name], "utterances": len(rows)},
                "structural": {"not_finite": [], "decode_errors": []}, "structurally_valid": True,
                "config": {"model": {"name": tp.RUN_NAME, "view": str(self.view), "view_sha256": view_sha},
                           "mode": "x1", "set": name, "group": "dev", "limit": None, "manifest_sha256": sha(manifest)},
                "records": rows})

    def write_config(self) -> None:
        write_json(self.run / "config.json", {"args": {"run_name": tp.RUN_NAME, "extra_sources": {
            "cv27-en:": {"kind": "clip store", "clip_pool": sha(self.cv / "store" / "clips.jsonl")
                         if (self.cv / "store" / "clips.jsonl").exists() else None},
            "ami-sdm:": {"kind": "ami pair store",
                         "clip_pool": sha(self.ami_store) if self.ami_store.exists() else None}}}})

    def write_summary(self) -> None:
        summary = {**tp.EXPECT_SUMMARY, "arm": "P2", "config_sha256": sha(self.run / "config.json"),
                   "dev_wer": dict(TORCH_DEV), "keab": {"model": {"sha256": sha(self.run / "model.pt")}}}
        write_json(self.run / "summary.json", summary)

    def arm_source(self, arm: str) -> dict:
        if arm == "B0":
            return {"kind": "pretrained", "description": "pretrained Parakeet-TDT-0.6B-v2, FP32, no training",
                    "sha256": tp.BASE_MODEL["nemo_sha256"], "base_model": dict(tp.BASE_MODEL)}
        if arm == "M1":
            return {"kind": "export", "sha256": tp.M1_EXPORT_SHA256, "base_model": dict(tp.BASE_MODEL),
                    "export_manifest": {"format": "parakeet-ternary-v1", "sha256": tp.M1_EXPORT_SHA256,
                                        "bytes": {"file_bytes": SIZES["m1_file_bytes"],
                                                  "original_nemo_bytes": SIZES["original_nemo_bytes"]}}}
        return {"kind": "export", "sha256": sha(self.mb1_export / "export.safetensors"),
                "base_model": dict(tp.BASE_MODEL),
                "export_manifest": {"format": tp.VIEW_FORMAT, "sha256": sha(self.mb1_export / "export.safetensors"),
                                    "bytes": {"file_bytes": SIZES["mb1_file_bytes"],
                                              "original_nemo_bytes": SIZES["original_nemo_bytes"]}}}

    def build_reused(self) -> None:
        self.mb1_export.mkdir(parents=True)
        (self.mb1_export / "export.safetensors").write_bytes(b"fake MB1 packed weights")
        write_json(self.mb1_export / "manifest.json", {"format": tp.VIEW_FORMAT, "file": "export.safetensors",
                                                       "sha256": sha(self.mb1_export / "export.safetensors"),
                                                       "config": {"decoding": copy.deepcopy(DECODING_CONFIG)}})
        versions = kt.numerical_settings("fp")["versions"]
        for arm, wer in (("B0", B0_WER), ("M1", M1_WER), ("MB1", MB1_WER)):
            out = self.arms[arm]
            sets = {}
            for name in TEST:
                manifest = self.manifests / f"test_{name}.jsonl"
                records = []
                for i, rec in enumerate(evaluate.read_manifest(manifest)):
                    hyp = "" if (arm == "B0" and name == "ami" and i == 2) else HYPS[i % 3]
                    records.append({"id": rec["id"], "ref": rec["text"], "hyp": hyp,
                                    **evaluate.score(rec["text"], hyp), "duration": rec["duration"]})
                doc = {"source": self.arm_source(arm), "set": name, "manifest": str(manifest),
                       "manifest_sha256": sha(manifest), "utterances": len(records), "limit": None,
                       "audio_hours": 0.001, "decoding": decoding(), "normalizer": evaluate.NORMALIZER,
                       "wer": {"wer": wer[name], "utterances": len(records), "excluded_empty_reference": 0},
                       "wall_s": 1.0, "records": records}
                if arm == "MB1":
                    doc["settings"] = {"versions": versions, "cudnn_deterministic": False, "cudnn_benchmark": False}
                write_json(out / f"{name}.json", doc)
                sets[name] = {"wer": 100 * wer[name], "utterances": len(records), "limit": None}
            summary = {"source": self.arm_source(arm), "sets": sets, "wall_s": 8.0}
            if arm == "MB1":
                summary["settings"] = {"versions": versions, "cudnn_deterministic": False, "cudnn_benchmark": False}
            write_json(out / "summary.json", summary)
        self.write_mb1_attempts()
        # MB1's recorded dev check
        docs = {}
        for name in DEV:
            manifest = self.manifests / f"{name}.jsonl"
            recs = [{"id": r["id"], "ref": r["text"], "hyp": r["text"], **evaluate.score(r["text"], r["text"])}
                    for r in evaluate.read_manifest(manifest)]
            write_json(self.mb1_dev_check / f"{name}.json", {"set": name, "wer": {"wer": MB1_DEV[name]},
                                                             "decoding": decoding(), "records": recs})
            docs[name] = sha(self.mb1_dev_check / f"{name}.json")
        write_json(self.mb1_dev_check / "gate.json", {
            "pass": True, "mean": sum(MB1_DEV.values()) / 5, "per_set": {n: {"wer": MB1_DEV[n]} for n in DEV},
            "per_set_sha256": docs, "batch_size": 64, "limit": None})

    def write_mb1_attempts(self) -> None:
        entries = [{"attempt": 1, "state": "started"}]
        entries += [{"attempt": 1, "set": n, "state": "set_done", "sha256": sha(self.arms["MB1"] / f"{n}.json")}
                    for n in TEST]
        entries += [{"attempt": 1, "state": "complete"}]
        self.mb1_attempts.write_text("".join(json.dumps(e) + "\n" for e in entries))

    def write_tsv(self, name: str, rows) -> None:
        self.raw.mkdir(parents=True, exist_ok=True)
        lines = ["client_id\tpath\tsentence\taccents"] + [f"{c}\t{p}\t{s}\t" for c, p, s in rows]
        (self.raw / name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def build_cv(self, accepted=ACCEPTED, counts=None) -> None:
        for name, rows in TSVS.items():
            self.write_tsv(name, rows)
        self.esb_files[0].parent.mkdir(parents=True, exist_ok=True)
        audio = pa.array([{"bytes": b"", "path": p} for _, _, p in ESB],
                         type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
        pq.write_table(pa.table({"audio": audio, "dataset": ["common_voice"] * len(ESB), "text": [t for _, t, _ in ESB],
                                 "id": [u for u, _, _ in ESB]}), self.esb_files[0])
        self.write_selection(accepted, counts)

    def write_selection(self, accepted=ACCEPTED, counts=None, store_sentence=None) -> None:
        rows = TSVS["validated.tsv"]
        ids = [f"cv27-en:{p[:-4]}" for _, p, _ in rows]
        reason = [None if p in accepted else "speaker" for _, p, _ in rows]
        sel = self.cv / "selection" / "selection.parquet"
        sel.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"id": ids, "path": [p for _, p, _ in rows], "client_id": [c for c, _, _ in rows],
                                 "sentence": [s for _, _, s in rows], "reason": pa.array(reason, pa.string())}), sel)
        store = self.cv / "store" / "clips.jsonl"
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text("".join(json.dumps({"id": i, "sentence": (store_sentence or {}).get(i, s), "label": s})
                                 + "\n" for i, (_, p, s) in zip(ids, rows) if p in accepted))
        write_json(self.cv / "selection" / "selection.json", {
            "table_sha256": sha(sel),
            "inputs": {"tsvs": [{"name": n, "sha256": sha(self.raw / n)} for n in tp.CV_TSVS],
                       "esb_test": [{"name": p.name, "sha256": sha(p)} for p in self.esb_files]},
            "counts": {"heldout": dict(CV_COUNTS if counts is None else counts)}})
        write_json(self.cv / "store" / "clips.meta.json", {"clips_sha256": sha(store), "selection_sha256": sha(sel),
                                                           "complete": True, "provisional": False})

    def build_ami(self, meetings=("ES2002a", "IS1001b")) -> None:
        self.ami_store.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for k, m in enumerate(meetings):
            lines.append(json.dumps({"id": f"AMI_{m}_sdm_FEE005_{k:07d}_{k + 100:07d}",
                                     "ihm_id": f"ami:AMI_{m}_H00_FEE005_{k:07d}_{k + 100:07d}", "meeting_id": m}))
        self.ami_store.write_text("\n".join(lines) + "\n")

    def build_stage4(self) -> None:
        write_json(self.keab / "results.json", {
            "gate": {"status": "complete", "decision": {"pass": True}},
            "eightbit": {"keab_prime_export": {"bundle_sha256": self.pins["BUNDLE_SHA256"],
                                               "view_sha256": self.pins["VIEW_SHA256"]}}})
        self.plans.mkdir(parents=True, exist_ok=True)
        (self.plans / "RESULTS-keab-prime.md").write_text("# Stage 4 results\n")

    def write_plan(self, digest: str | None = None) -> None:
        digest = sha(Path(tp.__file__)) if digest is None else digest
        (self.plans / "keab-prime-test.md").write_text(
            f"# Plan\n\n**Status: frozen; keab_test_paths.py SHA-256 `{digest}`.**\n\nBody.\n")

    def repin(self) -> None:
        """Rewrite config.json and summary.json from the current stores, then pin every fixture file (in place)."""
        self.write_config()
        self.write_summary()
        pins = {"PINNED_FILES": {rel: sha(self.run / rel) for rel in tp.PINNED_FILES}}
        pins.update(VIEW_SHA256=pins["PINNED_FILES"]["export/view/export.safetensors"],
                    BUNDLE_SHA256=pins["PINNED_FILES"]["export/bundle/bundle.safetensors"],
                    MODEL_SHA256=pins["PINNED_FILES"]["model.pt"], CONFIG_SHA256=pins["PINNED_FILES"]["config.json"],
                    DEV_X1_FILES={n: sha(self.scores_x1 / f"{n}.json") for n in DEV},
                    RUNTIME_X1_SHA256=sha(self.scores_x1 / "runtime.json"),
                    REUSED_FILES={arm: {f: sha(d / f) for f in ["summary.json", *[f"{n}.json" for n in TEST]]}
                                  for arm, d in self.arms.items()},
                    MB1_ATTEMPTS_SHA256=sha(self.mb1_attempts),
                    MB1_EXPORT_SHA256=sha(self.mb1_export / "export.safetensors"),
                    MB1_DEV_CHECK_GATE_SHA256=sha(self.mb1_dev_check / "gate.json"),
                    MB1_DEV_MEAN=read_json(self.mb1_dev_check / "gate.json")["mean"],
                    CV_CLIP_POOL=sha(self.cv / "store" / "clips.jsonl"),
                    CV_SELECTION_SHA256=sha(self.cv / "selection" / "selection.parquet"),
                    CV_SELECTION_META_SHA256=sha(self.cv / "selection" / "selection.json"),
                    AMI_CLIP_POOL=sha(self.ami_store))
        for key, value in pins.items():
            if isinstance(value, dict) and isinstance(self.pins.get(key), dict):
                self.pins[key].clear()
                self.pins[key].update(value)
            else:
                self.pins[key] = value
        if hasattr(self, "_installed_patches"):       # constants already installed: update the scalar pins
            for key, value in pins.items():
                if not isinstance(value, dict):
                    setattr(tp, key, value)

    def edit_json(self, path: Path, fn, repin: bool = True) -> None:
        data = read_json(path)
        fn(data)
        write_json(path, data)
        if repin:
            self.repin()

    # ------------------------------------------------------------------ stubs
    @contextlib.contextmanager
    def gpu_lock(self, label: str, wait_s: float):
        self.lock_events.append("gpu+")
        self.lock_held = True
        try:
            yield
        finally:
            self.lock_held = False
            self.lock_events.append("gpu-")

    def load_x1(self, export_dir, device="cuda", probe=None):
        self.loads.append({"mode": "x1", "path": Path(export_dir), "lock_held": self.lock_held})
        model = types.SimpleNamespace(_keab_x1={"export": str(Path(export_dir).resolve()),
                                                "export_sha256": sha(Path(export_dir) / "export.safetensors"),
                                                "kernel": tp.X1_KERNEL, "parts": tp.X1_PARTS, "runtime": "provider",
                                                "sources": {"runtime.py": "x"},
                                                "decoder_cuda_graphs_mode": "full_graph"})
        return model

    def close_x1(self, model) -> None:
        self.closed += 1

    def rows(self, mode: str, name: str, records: list[tuple[int, dict]]) -> list[dict]:
        out = []
        n_empty = self.empty.get((mode, name), 0)
        for k, (i, rec) in enumerate(records):
            hyp = "" if k < n_empty else rec["text"]
            row = {"i": i, "id": rec["id"], "hyp": hyp, "tokens": [i, 7], "timestamps": [], "empty": not hyp.strip(),
                   "padded_short": False, "duration": rec["duration"], "enc_len": 10, "enc_frames": 10, "finite": True,
                   "decode_error": None, "identity": "x", "ref": rec["text"], **evaluate.score(rec["text"], hyp)}
            if k == 0 and (mode, name) in self.invalid:
                row.update(decode_error="RuntimeError('CUDA graph capture failed')", hyp="")
            out.append(row)
        return out

    def score_records(self, model, records, label, identity):
        self.calls.append({"mode": "x1", "name": label, "n": len(records), "lock_held": self.lock_held})
        if self.fail_on.get("x1") == label:
            raise RuntimeError(f"simulated crash in {label}")
        rows = self.rows("x1", label, records)
        wer = (self.x1_dev_wer if label in DEV else self.x1_test_wer)[label]
        s = {"not_finite": [r["id"] for r in rows if r["finite"] is False],
             "decode_errors": [r["id"] for r in rows if r["decode_error"]]}
        return {"set": label, "utterances": len(rows), "wer": {"wer": wer, "utterances": len(rows),
                                                              "excluded_empty_reference": 0},
                "wer_percent": 100 * wer, "audio_hours": 0.001, "wall_s": 1.0,
                "empty_hypotheses": sum(r["empty"] for r in rows), "padded_short": 0, "structural": s,
                "structurally_valid": not s["not_finite"] and not s["decode_errors"], "records": rows}

    def load_export(self, out_dir, device="cpu"):
        label = "MB1" if Path(out_dir).resolve() == self.mb1_export.resolve() else "keab"
        self.loads.append({"mode": "fp", "path": Path(out_dir), "label": label, "lock_held": self.lock_held})
        return FakeModel(label)

    def evaluate_set(self, *args, **kwargs):
        bound = inspect.signature(REAL_EVALUATE_SET).bind(*args, **kwargs)
        bound.apply_defaults()
        a = dict(bound.arguments)
        name, model = a["name"], a["model"]
        self.calls.append({"mode": "fp", "name": name, "label": model.label, "batch_size": a["batch_size"],
                           "max_batch_seconds": a["max_batch_seconds"], "limit": a["limit"],
                           "graphs": model.computer.cuda_graphs_mode.value, "lock_held": self.lock_held})
        if self.fail_on.get("fp") == name:
            raise RuntimeError(f"simulated crash in {name}")
        manifest = Path(a["manifest"])
        records = list(enumerate(evaluate.read_manifest(manifest)[:a["limit"]]))
        for batch in evaluate.make_batches([r for _, r in records], a["batch_size"], a["max_batch_seconds"]):
            model.encoder(audio_signal=torch.zeros(len(batch), 1, 10), length=torch.full((len(batch),), 10))
        rows = [{k: r[k] for k in ("id", "ref", "hyp", "ref_norm", "hyp_norm", "S", "D", "I", "scored")}
                | {"duration": r["duration"]} for r in self.rows("fp", name, records)]
        if name in DEV:
            wer = (self.mb1_dev_wer if model.label == "MB1" else self.keab_fp_dev_wer)[name]
        else:
            wer = self.fp_test_wer[name]
        return {"source": a["source"], "set": name, "manifest": str(manifest),
                "manifest_sha256": evaluate.sha256_file(manifest), "utterances": len(rows), "limit": a["limit"],
                "audio_hours": 0.001, "decoding": decoding(a["batch_size"], a["max_batch_seconds"]),
                "normalizer": evaluate.NORMALIZER,
                "wer": {"wer": wer, "utterances": len(rows), "excluded_empty_reference": 0},
                "wall_s": 1.0, "padded_short": 0, "padded_short_ids": [], "records": rows}

    def load_tiny(self, out_dir, device="cpu"):
        label = "MB1" if Path(out_dir).resolve() == self.mb1_export.resolve() else "keab"
        self.loads.append({"mode": "fp", "path": Path(out_dir), "label": label, "lock_held": self.lock_held})
        return TinyModel(label, poison=label in self.poison_models, extra_call=self.extra_call)

    @contextlib.contextmanager
    def real_fp(self):
        """The REAL evaluate.evaluate_set with TinyModel as the loaded export. TinyModel transcribes every clean clip
        exactly, so MB1's recorded dev check is rewritten to WER 0 (and repinned)."""
        gate_path = self.mb1_dev_check / "gate.json"
        self.edit_json(gate_path, lambda g: (g.__setitem__("per_set", {n: {"wer": 0.0} for n in DEV}),
                                             g.__setitem__("mean", 0.0)), repin=False)
        self.repin()                                # also updates the installed scalar pins (gate hash, mean)
        with mock.patch.object(evaluate, "evaluate_set", REAL_EVALUATE_SET), \
                mock.patch.object(binary_export, "load_export", side_effect=self.load_tiny):
            yield

    @contextlib.contextmanager
    def installed(self):
        attrs = {"KEAB": self.keab, "RUN": self.run, "VIEW": self.view, "BUNDLE": self.bundle,
                 "SCORES_X1": self.scores_x1, "STAGE4_RESULTS_JSON": self.keab / "results.json",
                 "STAGE4_RESULTS_MD": self.plans / "RESULTS-keab-prime.md", "PLAN": self.plans / "keab-prime-test.md",
                 "B0_TEST": self.arms["B0"], "M1_TEST": self.arms["M1"], "MB1_TEST": self.arms["MB1"],
                 "REUSED": dict(self.arms), "MB1_ATTEMPTS": self.mb1_attempts, "MB1_EXPORT": self.mb1_export,
                 "MB1_DEV_CHECK_GATE": self.mb1_dev_check / "gate.json",
                 "CV_STORE_CLIPS": self.cv / "store" / "clips.jsonl",
                 "CV_STORE_META": self.cv / "store" / "clips.meta.json",
                 "CV_SELECTION": self.cv / "selection" / "selection.parquet",
                 "CV_SELECTION_META": self.cv / "selection" / "selection.json", "AMI_STORE_CLIPS": self.ami_store,
                 "TEST_ROOT": self.test_root, "PRECHECK": self.test_root / "precheck.json",
                 "TEST_JSON": self.test_root / "test.json", "RESULTS_MD": self.plans / "RESULTS-keab-prime-test.md",
                 "SCORER_HASHES": dict(CURRENT_SCORER), **self.pins}
        with contextlib.ExitStack() as stack:
            for key, value in attrs.items():
                stack.enter_context(mock.patch.object(tp, key, value))
            self._installed_patches = True
            stack.callback(delattr, self, "_installed_patches")
            stack.enter_context(mock.patch.dict(os.environ))
            stack.enter_context(mock.patch.object(paths, "MANIFESTS", self.manifests))
            stack.enter_context(mock.patch.object(kt, "score_chain_pending", lambda: list(self.pending)))
            stack.enter_context(mock.patch.object(kt, "active_stage4_jobs",
                                                  lambda: None if self.jobs is None else list(self.jobs)))
            stack.enter_context(mock.patch.object(kt, "cv_raw_dir", lambda: self.raw))
            stack.enter_context(mock.patch.object(kt, "cv_esb_files", lambda: list(self.esb_files)))
            stack.enter_context(mock.patch.object(kt, "gpu_lock", self.gpu_lock))
            stack.enter_context(mock.patch.object(kt, "DEVICE", "cpu"))
            stack.enter_context(mock.patch.object(tk, "_CUBLAS_AT_START", None))
            stack.enter_context(mock.patch.object(deployed, "require_wrapper",
                                                  lambda: {"wrapper": deployed.WRAPPER, "nvrtc_on_library_path": True}))
            stack.enter_context(mock.patch.object(deployed, "load_x1", side_effect=self.load_x1))
            stack.enter_context(mock.patch.object(deployed, "close", side_effect=self.close_x1))
            stack.enter_context(mock.patch.object(deployed, "score_records", side_effect=self.score_records))
            stack.enter_context(mock.patch.object(binary_export, "load_export", side_effect=self.load_export))
            stack.enter_context(mock.patch.object(evaluate, "evaluate_set", side_effect=self.evaluate_set))
            yield

    # ------------------------------------------------------------------ attempt records and outputs
    def ledger(self, mode: str) -> list[dict]:
        path = self.test_root / f"{mode}.attempts.jsonl"
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []

    def write_ledger(self, mode: str, entries: list[dict]) -> None:
        path = self.test_root / f"{mode}.attempts.jsonl"
        path.write_text("".join(json.dumps(e) + "\n" for e in entries))

    def tamper(self, mode: str, name: str, fn, rehash: bool = True) -> None:
        """Edit a Kea-B′ per-set test output; with rehash the set_done record and summary follow it, so only the
        edited content is wrong."""
        path = self.test_root / mode / f"{name}.json"
        doc = read_json(path)
        fn(doc)
        write_json(path, doc)
        if rehash:
            digest = sha(path)
            entries = self.ledger(mode)
            for e in entries:
                if e.get("state") == "set_done" and e.get("set") == name:
                    e["sha256"] = digest
            self.write_ledger(mode, entries)
            self.edit_json(self.test_root / mode / "summary.json",
                           lambda s: s["per_set_sha256"].__setitem__(name, digest), repin=False)

    def set_test_wer(self, mode: str, name: str, wer: float) -> None:
        """A Kea-B′ test set's WER changed consistently (per-set file, set_done hash, summary sets and mean)."""
        self.tamper(mode, name, lambda d: d["wer"].__setitem__("wer", wer))

        def fix(s):
            s["sets"][name]["wer"] = 100 * wer
            s["mean_wer"] = sum(s["sets"][n]["wer"] for n in paths.MEAN_SETS) / len(paths.MEAN_SETS)
        self.edit_json(self.test_root / mode / "summary.json", fix, repin=False)
