"""Encoder probe 1 (encoder-probe-ctc plan): T*'s encoder alone through its own CTC heads, greedy, no training.

    ./python encoder_probe_ctc.py [--sets dev|SET...] [--limit N [--shortest K]] [--device cuda|cpu]
                                  [--lock-wait SECONDS] [--min-available-gb GB] [--force]

T* is granite-speech-5.0-2b at the revision the diagnostic scored (teachers.GraniteSpeech, loaded as its records
were: transformers GraniteSpeechForConditionalGeneration in bfloat16; the projector and LLM are loaded, never run).
One clip per encoder pass (no padding), features from the snapshot's processor as teachers.py computes them, clips
shorter than evaluate.MIN_DECODE_SECONDS zero-padded exactly as score_teacher.py does (evaluate.pad_short).

Two heads, both greedy (frame argmax, merge repeats, drop the blank), with the conventions of IBM's reference
(CONVENTIONS, each with its source; REFERENCE lists the files, read only):

- char-mid: the encoder's character head encoder.out [348] at 50 Hz, as the encoder itself computes it at its middle
        layer (the same pre-update logits that weight the pooling; the final layer is never passed to encoder.out,
        plan v2.2); output i is chr(i), 0 the blank. It reads 8 of the 16 layers: a half-depth lower bound.
- bpe:  the BPE head of out_llm.safetensors [100353] on 12.5 Hz vectors: the final layer pooled over windows of 4
        frames with weights 1 - P(blank) from the middle layer's character-head posteriors; output 0 is the blank and
        output i is tokenizer id i - 1, detokenized by the snapshot's tokenizer with special tokens skipped.

The middle-layer posteriors are the logits the encoder itself computed at its middle layer: a forward hook on
encoder.out during the encoder pass (HeadCapture). The installed encoder applies encoder.out to a clone of the middle
layer's output and then updates that output in place, so a saved layer output would hold the post-update state.

Implementation checks 1-3 of the plan run first on every run that scores anything (run_checks; 20 and 50 clips
spread over the five dev sets, each set's shortest clip included): hooks are passive (bitwise), the pooling weights
use the pre-update state (bitwise against a clone taken in a layer hook, and not equal to the post-update state),
and two decodes give identical labels. If one fails, nothing is scored (exit 5).

Writes <out>/<head>/<set>.json in evaluate.py's per-set layout (records with id, ref, hyp, ref_norm, hyp_norm, S, D,
I, scored, duration, plus labels: the collapsed head outputs) and <out>/summary.json (conventions and sources,
revision, library versions, precision, checks, timings, peak memory). <out> is $RELABEL_ROOT/encoder-probe/ctc for a
full run and relabel_paths.SMOKE/encoder-probe-ctc-limit<N>[-cpu] with --limit. A set already written for both heads
by an identical configuration is skipped unless --force. The checks are written to summary.json as soon as they pass;
if every set is written but no passed checks are recorded for the configuration (a run stopped before its summary),
the checks run again and nothing is re-scored. Peak memory is the whole run's (allocator peaks are never reset).

GPU discipline as score_teacher.py: the parakeet GPU lock is taken before anything touches CUDA (exit 3 if busy);
the job lease is the caller's; --device cpu (smoke only) takes no lock; no model load below --min-available-gb of
host RAM (exit 4).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import evaluate  # finetune/parakeet-ternary/evaluate.py
import relabel_paths
from score_teacher import (LOCK_BUSY_EXIT, LOW_MEMORY_EXIT, LockBusy, available_memory_gb, expand_sets, gpu_lock,
                           select_records)

PLAN = "encoder-probe-ctc plan"
TEACHER = "granite-speech-5.0-2b"
HF_ID = "ibm-granite/granite-speech-5.0-2b"
REVISION = "f4e68169fa2a5751acc4b5da9957855e5e8fc3f1"
OUT = relabel_paths.ARTIFACTS / "encoder-probe" / "ctc"
SMOKE_NAME = "encoder-probe-ctc"
CHAR, BPE = "char-mid", "bpe"
HEADS = (CHAR, BPE)
HEAD_KEY = {CHAR: "char", BPE: "bpe"}  # the head's prefix in Probe.decode's output
CHECKS_FAILED_EXIT = 5

BLANK = 0
CHAR_OUTPUTS = 348
BPE_OUTPUTS = 100353
OUT_LLM_FILE = "out_llm.safetensors"
WINDOW = 4
HOP_LENGTH = 160
POOL_EPS = 1e-8
CHECK_CLIPS_PER_SET = 4       # checks 1 and 2: 20 clips
DETERMINISM_CLIPS_PER_SET = 10  # check 3: 50 clips
PRECISION = ("bfloat16 weights and inputs, no autocast (as the T* records); pooling weights and pooled vectors in "
             "float32, cast to bfloat16 for out_llm; posteriors as softmax of float32 logits")

_REPO = "github.com/ibm-granite/granite-speech-models, commit edac28f3a88af98dfa6a805ffecd2122645a8e47"
BPE_NB = "speculative_decoding_bpe.ipynb"
CHAR_NB = "speculative_decoding_notebook.ipynb"
PAPER = "arXiv 2603.11243v1"
FORWARD = "modeling_granite_speech.py"
CARD = "granite-speech-5.0-2b model card"
REFERENCE = {
    "files": {
        FORWARD: {"location": "transformers 5.17.0 (the shared runtime), "
                              "models/granite_speech/modeling_granite_speech.py",
                  "note": "GraniteSpeechCTCEncoder.forward applies self.out only at idx == num_layers // 2 "
                          "(1-based), on a clone of the layer output, then updates that output in place"},
        CARD: {"location": f"README.md of {HF_ID} at revision {REVISION}",
               "note": "'self-conditioned CTC from the middle layer'; 'The middle layer also provides non-blank "
                       "probabilities that are used for frame-level posterior-weighted pooling'"},
        BPE_NB: {"location": f"{_REPO}, notebooks/{BPE_NB}", "git_blob": "29177dc2fba69cd8b8792a887d213d82a5376e85",
                 "sha256": "366ffb2440cd700170cb9cd8e4c015903d2759a194527941b900b871f811f15f",
                 "note": "written for granite-speech-4.1-2b; linked as the speculative-decoding notebook by the "
                         "granite-speech-5.0-2b model card; cells cited by id"},
        CHAR_NB: {"location": f"{_REPO}, notebooks/{CHAR_NB}", "git_blob": "d0323961739917c167fe251307e937be589330e2",
                  "sha256": "38e33fb9a4865ba6d3861cf8234ca6b78eb5d3b4ac2083da579a06876c1949a7",
                  "note": f"the grapheme CTC notebook {BPE_NB} cell b67c6578 contrasts itself with; written for "
                          "granite-4.0-1b-speech; its cells have no ids and are cited by function"},
        PAPER: {"location": "arxiv.org/abs/2603.11243 (Saon et al., Self-Speculative Decoding for LLM-based ASR with "
                            "CTC Encoder Drafts)",
                "sha256": "9902b1bc91838ece03ac5b3827f45d4570500f51c1246c20a4d95a564af7ddfc"},
    },
    "handling": "downloaded into an empty directory under $RELABEL_ROOT/tmp; read as text, never executed or imported",
}
CONVENTIONS = {
    "char_layer": {
        "value": "middle layer (char-mid): the logits the encoder's own call of encoder.out returns after layer "
                 "num_layers // 2, before the in-place update (the same logits that weight the pooling); the final "
                 "layer is never passed to encoder.out",
        "source": f"{FORWARD} (GraniteSpeechCTCEncoder.forward: self.out only at idx == num_layers // 2); {CARD} "
                  f"('self-conditioned CTC from the middle layer'); {BPE_NB} cells 46d183fe and c9ed2e2d (encoder.out "
                  f"used only at the middle layer); plan v2.2. {CHAR_NB} ctc_decode applies it to the final layer, but "
                  "that notebook is for granite-4.0-1b-speech and the use does not carry over: on 5.0-2b the "
                  "final-layer reading had 0-1% blank frames and WER 105-156% on every dev set (smoke run)"},
    "char_blank": {"value": "output 0", "source": f"{CHAR_NB} ctc_decode: non_blank = dedup[dedup > 0]; "
                                                   f"{BPE_NB} cell c9ed2e2d: importance = 1 - probs[:, :, 0] (blank)"},
    "char_map": {"value": "output i -> chr(i) (also for i >= 256, as the reference does)",
                 "source": f"{CHAR_NB} ctc_decode: ''.join(chr(c) for c in non_blank); {PAPER} section 3.1 "
                           "(348 = the first 256 ASCII entries + 92 Katakana)"},
    "char_detokenization": {"value": "join the characters, strip surrounding whitespace",
                            "source": f"{CHAR_NB} ctc_decode and speculative_transcribe (text.strip())"},
    "bpe_layer": {"value": "final layer (last_hidden_state), pooled 4x, then out_llm",
                  "source": f"{BPE_NB} cell c9ed2e2d: posterior_weighted_pool(enc_hidden, ...), "
                            "enc_hidden = encoder_output.last_hidden_state; bpe_logits_valid = out_llm(x_valid)"},
    "bpe_blank": {"value": "output 0", "source": f"{BPE_NB} cells d7128f45, 46d183fe (LLM_OUT_DIM = 100352 tokens + "
                                                 "1 CTC blank, label 0) and c9ed2e2d (dedup[dedup > 0])"},
    "bpe_index_offset": {"value": "output i > 0 -> tokenizer id i - 1",
                         "source": f"{BPE_NB} cell 4d52caf9 and c9ed2e2d: token_ids = [t.item() - 1 ...]"},
    "bpe_detokenization": {
        "value": "tokenizer.decode(ids, skip_special_tokens=True), strip; empty if no ids",
        "source": f"{BPE_NB} cell c9ed2e2d (tokenizer.decode(token_ids) if token_ids else '') and 36cc6eae "
                  "(text.strip()); special tokens skipped per the plan (Method, BPE head) where the notebook keeps "
                  "them: records count clips where this changes the text"},
    "mid_layer": {"value": "after encoder layer num_layers // 2 (1-based; index num_layers // 2 - 1), the layer whose "
                           "output feeds the self-conditioning head",
                  "source": f"{BPE_NB} cell 46d183fe: mid_layer_idx = num_enc_layers // 2 - 1; model card"},
    "mid_posteriors": {"value": "the logits the encoder's middle-layer call of encoder.out returned (pre-update), "
                                "softmax in float32",
                       "source": f"{BPE_NB} cell c9ed2e2d (softmax(mid_grapheme_logits.float())); pre-update capture "
                                 "per the plan (Method, middle-layer posteriors), since the notebook's uncloned layer "
                                 "output is updated in place after its hook"},
    "pool_weights": {"value": "importance = 1 - P_mid(blank) per 50 Hz frame",
                     "source": f"{BPE_NB} cell c9ed2e2d: importance = 1.0 - grapheme_probs_mid[:, :, 0]"},
    "pool_formula": {"value": "pooled = sum_w h_t * imp_t / (sum_w imp_t + 1e-8)",
                     "source": f"{BPE_NB} cell 14927d07 (posterior_weighted_pool)"},
    "pool_window": {"value": WINDOW, "source": f"{BPE_NB} cell 46d183fe (LLM_DOWNSAMPLE_WINDOW = 4); model card"},
    "pool_alignment": {"value": "windows start at frame 0: frames [4k, 4k + 4)",
                       "source": f"{BPE_NB} cell 14927d07 (view(B, T // 4, 4, D) after end padding)"},
    "pool_final_partial_window": {"value": "the end is zero-padded (hidden and importance) to a multiple of 4, so a "
                                           "partial window averages its real frames",
                                  "source": f"{BPE_NB} cell 14927d07 (F.pad on hidden and importance)"},
    "pool_length": {"value": "ceil(min(samples // 160 // 2 + 1, frames) / 4) pooled vectors are classified",
                    "source": f"{BPE_NB} cell c9ed2e2d (enc_len, pooled_len)"},
    "greedy": {"value": "per frame argmax of softmax(float32 logits) (first maximum), merge repeats, then drop blanks",
               "source": f"{BPE_NB} cell c9ed2e2d and {CHAR_NB} ctc_decode (probs.max(dim=-1), "
                         f"torch.unique_consecutive, drop 0); {PAPER} section 2.1 (B removes duplicates, then blanks)"},
}


# --- pure parts -----------------------------------------------------------------------------------------------

def ctc_collapse(frames, blank: int = BLANK) -> list[int]:
    """Greedy CTC's B: merge consecutive repeats, then drop blanks."""
    out, prev = [], None
    for x in frames:
        x = int(x)
        if x != prev and x != blank:
            out.append(x)
        prev = x
    return out


def greedy(logits: torch.Tensor) -> tuple[list[int], list[int]]:
    """(per-frame argmax of softmax(float32 logits), collapsed labels) for logits [frames, outputs]."""
    frames = torch.softmax(logits.float(), dim=-1).max(dim=-1).indices.tolist()
    return frames, ctc_collapse(frames)


def importance(mid_logits: torch.Tensor) -> torch.Tensor:
    """1 - P(blank) per frame from the middle layer's character-head logits [..., frames, outputs] (float32)."""
    return 1.0 - torch.softmax(mid_logits.float(), dim=-1)[..., BLANK]


def posterior_weighted_pool(hidden: torch.Tensor, weights: torch.Tensor, window: int = WINDOW) -> torch.Tensor:
    """[B, T, D] hidden states and [B, T] weights -> [B, ceil(T / window), D]: the end zero-padded to whole windows,
    each window's weights normalized by their sum + 1e-8 (the reference's posterior_weighted_pool)."""
    b, t, d = hidden.shape
    pad = (window - t % window) % window
    if pad:
        hidden = nn.functional.pad(hidden, (0, 0, 0, pad))
        weights = nn.functional.pad(weights, (0, pad))
    n = hidden.shape[1] // window
    hidden = hidden.view(b, n, window, d)
    weights = weights.view(b, n, window)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + POOL_EPS)
    return (hidden * weights.unsqueeze(-1)).sum(dim=2)


def pooled_length(samples: int, frames: int, window: int = WINDOW) -> int:
    """Pooled vectors the reference classifies for one clip of `samples` samples encoded to `frames` frames."""
    enc_len = min(samples // HOP_LENGTH // 2 + 1, frames)
    return min(-(-enc_len // window), -(-frames // window))


def bpe_token_ids(labels: list[int]) -> list[int]:
    """Collapsed BPE-head outputs -> tokenizer ids (output i is id i - 1)."""
    return [i - 1 for i in labels]


def bpe_text(labels: list[int], tokenizer, skip_special_tokens: bool = True) -> str:
    ids = bpe_token_ids(labels)
    return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens).strip() if ids else ""


def char_text(labels: list[int]) -> str:
    return "".join(chr(i) for i in labels).strip()


class HeadCapture:
    """Forward hook on a head for one encoder pass: records the output of each call (passive: returns None)."""

    def __init__(self, head: nn.Module):
        self.head, self.calls, self._handle = head, [], None

    def _hook(self, module, args, output):
        self.calls.append(output.detach().clone())

    def __enter__(self) -> "HeadCapture":
        self._handle = self.head.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc) -> bool:
        self._handle.remove()
        self._handle = None
        return False

    def logits(self) -> torch.Tensor:
        if len(self.calls) != 1:
            raise RuntimeError(f"expected exactly one middle-layer call of the character head, saw {len(self.calls)}")
        return self.calls[0]


def encode(encoder: nn.Module, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(final-layer hidden states [1, T, D], the middle layer's character-head logits [1, T, V]) of one pass."""
    with HeadCapture(encoder.out) as cap:
        final = encoder(features, return_dict=True).last_hidden_state
    return final, cap.logits()


# --- the probe ------------------------------------------------------------------------------------------------

class Probe:
    """T*'s encoder, its two CTC heads and the snapshot's processor and tokenizer, on one device."""

    def __init__(self, processor, prompt: str, tokenizer, encoder: nn.Module, out_llm: nn.Module,
                 device: torch.device):
        self.processor, self.prompt, self.tokenizer = processor, prompt, tokenizer
        self.encoder, self.out_llm, self.device = encoder, out_llm, device
        self.dtype = next(encoder.parameters()).dtype
        self.special_ids = set(getattr(tokenizer, "all_special_ids", []))

    def features(self, clip: np.ndarray) -> torch.Tensor:
        """input_features [1, T, 160] as teachers.GraniteSpeech.transcribe computes them, for one clip."""
        wav = torch.from_numpy(np.ascontiguousarray(clip, dtype=np.float32))
        inputs = self.processor([self.prompt], [wav], device=str(self.device), return_tensors="pt")
        return inputs["input_features"].to(self.device, self.dtype)

    def heads(self, samples: int, final: torch.Tensor, mid_logits: torch.Tensor) -> dict:
        """Both heads from one pass: char-mid decodes the middle-layer logits, the same ones that weight the pooling."""
        char_frames, char_labels = greedy(mid_logits[0])
        n = pooled_length(samples, final.shape[1])
        pooled = posterior_weighted_pool(final, importance(mid_logits), WINDOW)[0, :n]
        bpe_frames, bpe_labels = greedy(self.out_llm(pooled.to(self.out_llm.weight.dtype)))
        return {"frames": int(final.shape[1]), "pooled_frames": n, "char_frames": char_frames,
                "char_labels": char_labels, "bpe_frames": bpe_frames, "bpe_labels": bpe_labels}

    @torch.inference_mode()
    def decode(self, clip: np.ndarray) -> dict:
        final, mid = encode(self.encoder, self.features(clip))
        out = self.heads(len(clip), final, mid)
        ids = bpe_token_ids(out["bpe_labels"])
        out["char"] = char_text(out["char_labels"])
        out["bpe"] = bpe_text(out["bpe_labels"], self.tokenizer)
        out["bpe_special_ids"] = [i for i in ids if i in self.special_ids]
        out["bpe_skip_changed"] = out["bpe"] != bpe_text(out["bpe_labels"], self.tokenizer, skip_special_tokens=False)
        out["enc_len_formula_differs"] = min(len(clip) // HOP_LENGTH // 2 + 1, out["frames"]) != out["frames"]
        return out


def load_probe(device: torch.device) -> tuple[Probe, dict]:
    """T* as teachers.GraniteSpeech loads it (the diagnostic's records), plus out_llm from the same snapshot."""
    import teachers
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    teacher = teachers.get(TEACHER)
    if (teacher.hf_id, teacher.revision) != (HF_ID, REVISION):
        raise RuntimeError("teachers.py's T* is not the revision this probe scores")
    teacher.load(str(device))
    encoder = teacher.model.model.encoder
    if encoder.out.out_features != CHAR_OUTPUTS or encoder.num_layers != 16 or encoder.config.hidden_dim != 1024:
        raise RuntimeError(f"unexpected encoder: out {encoder.out.out_features}, layers {encoder.num_layers}")
    path = Path(hf_hub_download(HF_ID, OUT_LLM_FILE, revision=REVISION))
    state = load_file(str(path))
    shapes = {k: tuple(v.shape) for k, v in state.items()}
    if shapes != {"weight": (BPE_OUTPUTS, 1024), "bias": (BPE_OUTPUTS,)}:
        raise RuntimeError(f"unexpected {OUT_LLM_FILE}: {shapes}")
    stored = str(state["weight"].dtype)
    out_llm = nn.Linear(1024, BPE_OUTPUTS, bias=True)
    out_llm.load_state_dict(state)
    dtype = next(encoder.parameters()).dtype
    out_llm = out_llm.to(device=device, dtype=dtype).eval()
    tokenizer = teacher.tokenizer
    if tokenizer.vocab_size != BPE_OUTPUTS - 1:
        raise RuntimeError(f"tokenizer vocab_size {tokenizer.vocab_size} != {BPE_OUTPUTS - 1}")
    probe = Probe(teacher.processor, teacher.chat_prompt, tokenizer, encoder.eval(), out_llm, device)
    info = teacher.info()
    info.update(model_load_s=teacher.load_s, encoder_dtype=str(dtype),
                out_llm={"file": OUT_LLM_FILE, "sha256": evaluate.sha256_file(path), "stored_dtype": stored,
                         "shapes": shapes, "run_dtype": str(dtype)},
                tokenizer={"len": len(tokenizer), "vocab_size": tokenizer.vocab_size,
                           "special_ids": sorted(probe.special_ids)},
                encoder={"num_layers": encoder.num_layers, "hidden_dim": encoder.config.hidden_dim,
                         "char_outputs": encoder.out.out_features})
    return probe, info


# --- implementation checks 1-3 ----------------------------------------------------------------------------------

def check_clips(manifests: dict[str, list[dict]], per_set: int) -> list[tuple[str, dict]]:
    """Per set: its shortest clip plus clips spread evenly over the id-sorted manifest, `per_set` in all."""
    out = []
    for name in sorted(manifests):
        records = manifests[name]
        shortest = min(records, key=lambda r: (r["duration"], r["id"]))
        picked = {shortest["id"]: shortest}
        for j in range(per_set):
            if len(picked) == per_set:
                break
            r = records[(j * len(records)) // per_set]
            picked.setdefault(r["id"], r)
        out += [(name, picked[i]) for i in sorted(picked)]
    return out


def _clip(record: dict) -> np.ndarray:
    return evaluate.pad_short(evaluate.load_audio(record["audio_filepath"]))[0]


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max()) if a.numel() else 0.0


def _keeper(saved: dict):
    """Layer hook of check 2: the layer output itself (updated in place later) and a clone taken now."""
    def keep(module, args, output):
        saved["post"], saved["pre"] = output, output.detach().clone()
    return keep


def check_hooks_passive(probe: Probe, clips: list[tuple[str, dict]]) -> dict:
    """Check 1: the final layer with the probe's hook (and with check 2's layer hook too) equals it without, bitwise."""
    failed, worst = [], 0.0
    layer = probe.encoder.layers[probe.encoder.num_layers // 2 - 1]
    with torch.inference_mode():
        for name, r in clips:
            feats = probe.features(_clip(r))
            plain = probe.encoder(feats, return_dict=True).last_hidden_state
            hooked, _ = encode(probe.encoder, feats)
            handle = layer.register_forward_hook(_keeper({}))
            try:
                both, _ = encode(probe.encoder, feats)
            finally:
                handle.remove()
            worst = max(worst, _maxdiff(plain, hooked), _maxdiff(plain, both))
            if not (torch.equal(plain, hooked) and torch.equal(plain, both)):
                failed.append(r["id"])
    return {"passed": not failed, "clips": len(clips), "failed_ids": failed, "max_abs_diff": worst}


def check_pre_feedback(probe: Probe, clips: list[tuple[str, dict]]) -> dict:
    """Check 2: the middle-layer logits the probe uses equal encoder.out on a clone of the middle layer's output taken
    in a layer hook before the in-place update (bitwise), and differ from encoder.out on the post-update state."""
    enc = probe.encoder
    layer = enc.layers[enc.num_layers // 2 - 1]
    failed, equal_post, logit_diff, weight_diff, token_changes = [], [], 0.0, 0.0, 0
    with torch.inference_mode():
        for name, r in clips:
            clip = _clip(r)
            saved = {}
            handle = layer.register_forward_hook(_keeper(saved))
            try:
                final, used = encode(enc, probe.features(clip))
            finally:
                handle.remove()
            pre, post = enc.out(saved["pre"]), enc.out(saved["post"])
            if not torch.equal(used, pre):
                failed.append(r["id"])
            if torch.equal(used, post):
                equal_post.append(r["id"])
            logit_diff = max(logit_diff, _maxdiff(used, post))
            weight_diff = max(weight_diff, _maxdiff(importance(used), importance(post)))
            token_changes += probe.heads(len(clip), final, used)["bpe_labels"] != \
                probe.heads(len(clip), final, post)["bpe_labels"]
    return {"passed": not failed and not equal_post, "clips": len(clips), "failed_ids": failed,
            "post_update_equal_clips": len(equal_post), "post_update_equal_ids": equal_post,
            "post_update_max_abs_logit_diff": logit_diff, "post_update_max_abs_weight_diff": weight_diff,
            "post_update_bpe_label_changes": int(token_changes),
            "layer": f"encoder.layers[{enc.num_layers // 2 - 1}] (the self-conditioning layer)"}


def check_determinism(probe: Probe, clips: list[tuple[str, dict]]) -> dict:
    """Check 3: the clips decoded twice (two passes) give identical frame labels and token ids for both heads."""
    audio = [_clip(r) for _, r in clips]
    keys = ("char_frames", "char_labels", "bpe_frames", "bpe_labels")
    first = [probe.decode(c) for c in audio]
    second = [probe.decode(c) for c in audio]
    failed = [r["id"] for (_, r), a, b in zip(clips, first, second) if any(a[k] != b[k] for k in keys)]
    return {"passed": not failed, "clips": len(clips), "failed_ids": failed, "compared": list(keys)}


def run_checks(probe: Probe, manifests: dict[str, list[dict]]) -> dict:
    few = check_clips(manifests, CHECK_CLIPS_PER_SET)
    many = check_clips(manifests, DETERMINISM_CLIPS_PER_SET)
    out = {"hooks_passive": check_hooks_passive(probe, few), "pre_feedback": check_pre_feedback(probe, few),
           "determinism": check_determinism(probe, many),
           "clip_ids": {"hooks_passive_and_pre_feedback": [r["id"] for _, r in few],
                        "determinism": [r["id"] for _, r in many]}}
    out["passed"] = all(out[k]["passed"] for k in ("hooks_passive", "pre_feedback", "determinism"))
    return out


# --- scoring --------------------------------------------------------------------------------------------------

def output_dir(limit: int | None) -> Path:
    """OUT for a full run, relabel_paths.SMOKE/encoder-probe-ctc-limit<N> for a --limit smoke run."""
    if limit is None:
        return OUT
    if limit <= 0:
        raise ValueError("--limit must be positive")
    out = relabel_paths.SMOKE / f"{SMOKE_NAME}-limit{limit}"
    assert not out.is_relative_to(OUT)
    return out


def conventions_sha256() -> str:
    return hashlib.sha256(json.dumps({k: v["value"] for k, v in CONVENTIONS.items()}, sort_keys=True).encode()
                          ).hexdigest()


def run_config(args) -> dict:
    return {"probe": SMOKE_NAME, "teacher": TEACHER, "hf_id": HF_ID, "revision": REVISION, "limit": args.limit,
            "shortest": args.shortest, "device": args.device, "precision": PRECISION,
            "conventions_sha256": conventions_sha256()}


def _peak_gpu(device: torch.device) -> dict:
    """The process's CUDA allocator peaks so far (never reset: they include model loading and the checks)."""
    cuda = device.type == "cuda"
    return {"peak_gpu_memory_gb": torch.cuda.max_memory_allocated(device) / 2**30 if cuda else None,
            "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved(device) / 2**30 if cuda else None}


def _memory(device: torch.device, after_load: dict | None) -> dict:
    """Whole-run peaks: CUDA allocator (since process start), allocator state right after loading, host max RSS."""
    peak = _peak_gpu(device)
    return {"gpu_allocated_gb": peak["peak_gpu_memory_gb"], "gpu_reserved_gb": peak["peak_gpu_reserved_gb"],
            "gpu_after_load": after_load, "scope": "whole run, since process start (allocator peaks never reset)",
            "host_max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}


def score_set(probe: Probe, name: str, manifest: Path, args, config: dict, source: dict) -> dict[str, dict]:
    """{head: per-set result} for one dev set."""
    records = select_records(manifest, args.limit, args.shortest)
    rows = {h: [] for h in HEADS}
    padded, load_s, infer_s = [], 0.0, 0.0
    counts = {"bpe_special_id_clips": [], "bpe_skip_changed_clips": [], "char_above_255_clips": [],
              "enc_len_formula_differs_clips": []}
    wall = time.perf_counter()
    for k, r in enumerate(records):
        start = time.perf_counter()
        clip, was_padded = evaluate.pad_short(evaluate.load_audio(r["audio_filepath"]))
        load_s += time.perf_counter() - start
        if was_padded:
            padded.append(r["id"])
        start = time.perf_counter()
        out = probe.decode(clip)
        infer_s += time.perf_counter() - start
        for head in HEADS:
            key = HEAD_KEY[head]
            hyp = out[key]
            rows[head].append({"id": r["id"], "ref": r["text"], "hyp": hyp, **evaluate.score(r["text"], hyp),
                               "duration": r["duration"], "labels": out[f"{key}_labels"],
                               "frames": out["frames"] if head == CHAR else out["pooled_frames"]})
        if out["bpe_special_ids"]:
            counts["bpe_special_id_clips"].append(r["id"])
        if out["bpe_skip_changed"]:
            counts["bpe_skip_changed_clips"].append(r["id"])
        if any(i > 255 for i in out["char_labels"]):
            counts["char_above_255_clips"].append(r["id"])
        if out["enc_len_formula_differs"]:
            counts["enc_len_formula_differs_clips"].append(r["id"])
        if k % 2000 == 0:
            print(f"  {name}: {k + 1}/{len(records)} ({time.perf_counter() - wall:.0f} s)", flush=True)
    hours = sum(r["duration"] for r in records) / 3600
    throughput = {"audio_load_s": load_s, "inference_wall_s": infer_s, "set_wall_s": time.perf_counter() - wall,
                  "audio_hours": hours, "audio_hours_per_gpu_hour": hours / (infer_s / 3600) if infer_s > 0 else None,
                  "clips_per_pass": 1, "peak_scope": "process so far (includes loading and the checks)",
                  **_peak_gpu(probe.device)}
    diagnostics = {k: {"count": len(v), "ids": v[:50]} for k, v in counts.items()}
    out, manifest_sha = {}, evaluate.sha256_file(manifest)
    for head in HEADS:
        recs = rows[head]
        out[head] = {
            "source": source, "set": name, "head": head, "manifest": str(manifest),
            "manifest_sha256": manifest_sha, "utterances": len(recs), "limit": args.limit,
            "shortest": args.shortest, "audio_hours": hours, "config": config,
            "decoding": {"head": head, "greedy": CONVENTIONS["greedy"]["value"], "precision": PRECISION,
                         "conventions_sha256": config["conventions_sha256"]},
            "normalizer": evaluate.NORMALIZER,
            "wer": evaluate.wer_summary(recs) if any(r["scored"] for r in recs) else None,
            "empty_hypotheses": sum(not r["hyp_norm"].split() for r in recs if r["scored"]),
            "wall_s": infer_s, "padded_short": len(padded), "padded_short_ids": sorted(padded),
            "min_decode_seconds": evaluate.MIN_DECODE_SECONDS, "diagnostics": diagnostics,
            "throughput": throughput, "records": recs}
    return out


def _set_done(out_dir: Path, name: str, config: dict) -> bool:
    for head in HEADS:
        path = out_dir / head / f"{name}.json"
        if not path.exists() or json.loads(path.read_text()).get("config") != config:
            return False
    return True


def _merge_memory(memory: dict | None, set_peaks: list[dict]) -> dict:
    """The run's peaks: the larger of the given whole-run (or checkpointed) peaks and the per-set records' process
    peaks, so a run resumed without reloading keeps the measurement."""
    out = dict(memory or {"gpu_after_load": None, "host_max_rss_gb": None,
                          "scope": "recovered from the per-set records (process peaks at the end of each set)"})
    for key, per_set in (("gpu_allocated_gb", "peak_gpu_memory_gb"), ("gpu_reserved_gb", "peak_gpu_reserved_gb")):
        values = [v for v in [out.get(key)] + [t.get(per_set) for t in set_peaks] if v is not None]
        out[key] = max(values) if values else None
    return out


def summarize(out_dir: Path, config: dict, info: dict, checks: dict | None, load_s: float | None,
              memory: dict | None) -> dict:
    import teachers

    heads, timings, set_peaks = {}, {}, []
    for head in HEADS:
        heads[head] = {}
        for name in relabel_paths.SETS:
            path = out_dir / head / f"{name}.json"
            if not path.exists():
                continue
            r = json.loads(path.read_text())
            if r.get("config") != config:
                continue
            w, t = r["wer"], r["throughput"]
            heads[head][name] = {"wer": 100 * w["wer"] if w else None, "utterances": r["utterances"],
                                 "scored_utterances": w["utterances"] if w else 0,
                                 "empty_hypotheses": r["empty_hypotheses"], "padded_short": r["padded_short"],
                                 "diagnostics": {k: v["count"] for k, v in r["diagnostics"].items()}}
            timings[name] = {k: t[k] for k in ("audio_hours", "audio_load_s", "inference_wall_s", "set_wall_s",
                                                 "audio_hours_per_gpu_hour")}
            set_peaks.append(t)
    done = [n for n in relabel_paths.SETS if all(n in heads[h] for h in HEADS)]
    complete = config["limit"] is None and set(done) == set(relabel_paths.SETS)
    status = "scored" if complete else ("smoke" if config["limit"] is not None else "partial")
    hours = sum(timings[n]["audio_hours"] for n in done)
    wall = sum(timings[n]["inference_wall_s"] for n in done)
    return {"status": status, "complete": complete, "plan": PLAN, "config": config, "teacher": TEACHER,
            "hf_id": HF_ID, "revision": REVISION, "versions": teachers.library_versions(), "precision": PRECISION,
            "conventions": CONVENTIONS, "conventions_sha256": config["conventions_sha256"], "reference": REFERENCE,
            "source": info, "checks": checks, "heads": heads,
            "timings": {"model_load_s": load_s, "sets": timings, "audio_hours": hours, "inference_wall_s": wall,
                        "audio_hours_per_gpu_hour": hours / (wall / 3600) if wall > 0 else None},
            "peak_memory": _merge_memory(memory, set_peaks),
            "job_lease_env": {k: v for k, v in os.environ.items() if k.startswith("AGENT_JOBS_")},
            "gpu_lock": str(relabel_paths.GPU_LOCK) if config["device"] == "cuda" else None}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", nargs="+", default=["dev"])
    parser.add_argument("--limit", type=int, help="smoke: first N id-sorted utterances per set (smoke path)")
    parser.add_argument("--shortest", type=int, default=0, help="smoke: also the K shortest utterances per set")
    parser.add_argument("--device", default="cuda", help="cuda, or cpu for a smoke run without the GPU (no lock)")
    parser.add_argument("--lock-wait", type=float, default=1800.0, help="seconds to wait for the GPU lock")
    parser.add_argument("--force", action="store_true", help="re-score sets already written")
    parser.add_argument("--min-available-gb", type=float, default=16.0,
                        help="refuse to load the model if the host has less RAM available (exit 4)")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.shortest and args.limit is None:
        parser.error("--shortest is a smoke option and needs --limit")
    if args.device not in ("cuda", "cpu"):
        parser.error("--device is cuda or cpu")
    if args.device == "cpu" and args.limit is None:
        parser.error("--device cpu is only for --limit smoke runs")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    relabel_paths.require_mount()
    out_dir = output_dir(args.limit)
    sets = [evaluate.resolve_manifest(n) for n in expand_sets(args.sets)]
    missing = [str(p) for _, p in sets if not p.exists()]
    if missing:
        raise SystemExit(f"missing manifests: {missing}")
    config = run_config(args)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return _run_locked(args, out_dir.with_name(out_dir.name + "-cpu"), sets, config)
    try:
        with gpu_lock(f"relabel encoder_probe_ctc {TEACHER}", args.lock_wait):
            return _run_locked(args, out_dir, sets, config)
    except LockBusy as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return LOCK_BUSY_EXIT


def _require_cuda(device: torch.device) -> None:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required")


def _run_locked(args, out_dir: Path, sets, config: dict) -> int:
    device = torch.device(args.device)
    _require_cuda(device)
    fraction = os.environ.get("AGENT_JOBS_VRAM_FRACTION")
    if fraction and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(fraction), torch.cuda.current_device())
    todo = []
    for name, manifest in sets:
        if not args.force and _set_done(out_dir, name, config):
            print(f"{name}: already written for both heads with this configuration, skipping", flush=True)
            continue
        todo.append((name, manifest))
    summary_path = out_dir / "summary.json"
    old = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    old_checks = old.get("checks") if old.get("config") == config else None
    if not todo and (old_checks or {}).get("passed"):
        summary = summarize(out_dir, config, old.get("source", {}), old_checks,
                            old.get("timings", {}).get("model_load_s"), old.get("peak_memory"))
        evaluate.write_json(summary_path, summary)
        print(f"summary: status {summary['status']}; {summary_path}", flush=True)
        return 0
    if not todo:
        # Every set is written, but no passed checks are recorded for this configuration (a run stopped before its
        # summary): run checks 1-3 again, without re-scoring.
        print("all sets written; no passed implementation checks recorded: running the checks only", flush=True)
    if available_memory_gb() < args.min_available_gb:
        print(f"refusing to load: {available_memory_gb():.1f} GB host memory available, "
              f"--min-available-gb {args.min_available_gb}", file=sys.stderr)
        return LOW_MEMORY_EXIT
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluate.write_json(summary_path, {"status": "in_progress", "complete": False, "config": config})
    start = time.perf_counter()
    probe, info = load_probe(device)
    load_s = info.get("model_load_s") or (time.perf_counter() - start)
    after_load = ({"allocated_gb": torch.cuda.memory_allocated(device) / 2**30,
                   "reserved_gb": torch.cuda.memory_reserved(device) / 2**30} if device.type == "cuda" else None)
    print(f"{TEACHER} encoder probe: loaded in {load_s:.1f} s", flush=True)
    manifests = {n: evaluate.read_manifest(evaluate.resolve_manifest(n)[1]) for n in relabel_paths.SETS}
    checks = run_checks(probe, manifests)
    print("checks: " + ", ".join(f"{k} {'pass' if checks.get(k, {}).get('passed') else 'FAIL'}"
                                 for k in ("hooks_passive", "pre_feedback", "determinism")), flush=True)
    if not checks["passed"]:
        evaluate.write_json(summary_path, {"status": "checks_failed", "complete": False, "config": config,
                                           "checks": checks, "source": info})
        print("implementation checks failed: nothing scored", file=sys.stderr)
        return CHECKS_FAILED_EXIT
    # Persisted at once, so a run stopped later can resume without losing the checks or the memory measured so far.
    evaluate.write_json(summary_path, {"status": "in_progress", "complete": False, "config": config,
                                       "checks": checks, "source": info, "timings": {"model_load_s": load_s},
                                       "peak_memory": _memory(device, after_load)})
    for name, manifest in todo:
        result = score_set(probe, name, manifest, args, config, info)
        for head in HEADS:
            (out_dir / head).mkdir(parents=True, exist_ok=True)
            evaluate.write_json(out_dir / head / f"{name}.json", result[head])
        t = result["bpe"]["throughput"]
        wers = " ".join(f"{h} {100 * result[h]['wer']['wer']:.2f}%" if result[h]["wer"] else f"{h} n/a"
                        for h in HEADS)
        print(f"{name}: n={result['bpe']['utterances']} {wers} infer {t['inference_wall_s']:.1f} s "
              f"({t['audio_hours_per_gpu_hour'] or 0:.0f} audio h/h) padded {result['bpe']['padded_short']}",
              flush=True)
    summary = summarize(out_dir, config, info, checks, load_s, _memory(device, after_load))
    evaluate.write_json(summary_path, summary)
    print(f"summary: status {summary['status']}; {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
