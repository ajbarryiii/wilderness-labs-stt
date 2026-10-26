"""One adapter per teacher: load at a pinned revision, transcribe a batch of 16 kHz mono float32 clips.

Each adapter follows its model card's English transcription usage, greedy (do_sample False, one beam),
punctuation and capitalization on, in bfloat16 (diagnostic plan "Candidate teachers"):

- qwen3-asr-1.7b         Qwen/Qwen3-ASR-1.7B-hf, transformers AutoModelForMultimodalLM,
                         processor.apply_transcription_request(language="English"),
                         processor.decode(return_format="transcription_only").
- granite-speech-5.0-2b  ibm-granite/granite-speech-5.0-2b, transformers AutoModelForSpeechSeq2Seq,
                         chat prompt "<|audio|>transcribe the speech with proper punctuation and
                         capitalization." (the card's "ASR (with punctuation)" prompt).
- canary-qwen-2.5b       nvidia/canary-qwen-2.5b, NeMo SALM (needs peft: RELABEL_DEPS=canary-qwen-2.5b),
                         prompt "Transcribe the following: <|audioplaceholder|>". SALM builds its LLM
                         and tokenizer from Qwen/Qwen3-1.7B by name; to pin that too, the model is loaded
                         from a local directory (MODELS/) holding the pinned canary files plus the pinned
                         Qwen3-1.7B tokenizer and config (NeMo's supported local-artifact layout:
                         tokenizer at the root, LLM config under llm_backbone/). LLM in bf16, the audio
                         perception module in fp32, run under bf16 autocast.

max_new_tokens is 256 for all three (the longest dev reference has 107 words); outputs that reach the
cap without an end token are counted and their ids recorded.

transcribe(clips) returns (raw texts, per-clip "hit the token cap" flags) in input order.
"""
from __future__ import annotations

import os
import platform
import time
from pathlib import Path

import numpy as np

import relabel_paths

MAX_NEW_TOKENS = 256
SAMPLE_RATE = 16000
PREFETCH_IGNORE = ["assets/*", "demo/*", "*.png", "*.wav"]  # model-card media only


class NotScorable(Exception):
    """The teacher cannot be scored for a reason outside this code (gated repository, access refused)."""


def library_versions() -> dict:
    import importlib

    out = {"python": platform.python_version()}
    for name in ("torch", "transformers", "nemo", "peft", "accelerate", "huggingface_hub", "tokenizers",
                 "numpy", "soundfile"):
        try:
            module = importlib.import_module(name)
            out[name] = getattr(module, "__version__", "?")
        except Exception:  # noqa: BLE001  (absent packages are recorded as absent)
            out[name] = None
    try:
        import torch

        out["cuda"] = torch.version.cuda
        out["cudnn"] = torch.backends.cudnn.version()
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        pass
    return out


def _gated(exc: BaseException) -> bool:
    from huggingface_hub.errors import GatedRepoError

    seen = exc
    while seen is not None:
        if isinstance(seen, GatedRepoError):
            return True
        seen = seen.__cause__ or seen.__context__
    return False


def _hit_cap(rows, eos_ids: set[int], max_new_tokens: int) -> list[bool]:
    """Per generated row (new tokens only): no end token and the row is max_new_tokens long."""
    out = []
    for row in rows:
        ids = row.tolist()
        out.append(len(ids) >= max_new_tokens and not any(t in eos_ids for t in ids))
    return out


class Teacher:
    name: str
    hf_id: str
    revision: str
    license: str
    library: str
    deps: str | None = None              # RELABEL_DEPS environment needed, if any
    prompt: str = ""
    batch_size: int = 32                 # default; score_teacher --batch-size overrides
    max_batch_seconds: float = 600.0
    internal_min_seconds: float | None = None  # padding the teacher's own pipeline applies, recorded

    def __init__(self) -> None:
        self.device = None
        self.load_s: float | None = None

    def load(self, device: str) -> None:
        import torch

        self.device = torch.device(device)
        start = time.perf_counter()
        try:
            self._load()
        except Exception as exc:
            if _gated(exc):
                raise NotScorable(f"{self.hf_id}: gated repository, access not granted ({exc})") from exc
            raise
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.load_s = time.perf_counter() - start

    def _load(self) -> None:
        raise NotImplementedError

    def prefetch(self) -> str:
        """Download the pinned revision into the Hugging Face cache (no GPU, no lock); returns the snapshot."""
        from huggingface_hub import snapshot_download

        relabel_paths.require_mount()
        try:
            return snapshot_download(self.hf_id, revision=self.revision, ignore_patterns=PREFETCH_IGNORE)
        except Exception as exc:
            if _gated(exc):
                raise NotScorable(f"{self.hf_id}: gated repository, access not granted ({exc})") from exc
            raise

    def transcribe(self, clips: list[np.ndarray]) -> tuple[list[str], list[bool]]:
        raise NotImplementedError

    def decoding(self) -> dict:
        return {"do_sample": False, "num_beams": 1, "max_new_tokens": MAX_NEW_TOKENS, "precision": "bfloat16"}

    def info(self) -> dict:
        return {"name": self.name, "hf_id": self.hf_id, "revision": self.revision, "license": self.license,
                "library": self.library, "deps": self.deps, "prompt": self.prompt, "decoding": self.decoding(),
                "internal_min_seconds": self.internal_min_seconds, "versions": library_versions(),
                "extra": self.extra_info()}

    def extra_info(self) -> dict:
        return {}


class Qwen3ASR(Teacher):
    name = "qwen3-asr-1.7b"
    hf_id = "Qwen/Qwen3-ASR-1.7B-hf"
    revision = "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"  # Hub main on 2026-10-21
    license = "apache-2.0"
    library = "transformers"
    language = "English"
    prompt = ("processor.apply_transcription_request(audio, language='English'): chat template with the "
              "audio as the user turn and the assistant turn prefilled with 'language English<asr_text>'")
    internal_min_seconds = 0.5  # Qwen3ASRFeatureExtractor min_length 8000 samples (zero-pads shorter clips)

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.hf_id, revision=self.revision)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.hf_id, revision=self.revision, dtype=torch.bfloat16).to(self.device).eval()
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])

    def transcribe(self, clips):
        import torch

        inputs = self.processor.apply_transcription_request(list(clips), language=self.language).to(
            self.device, torch.bfloat16)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1)
        new = out[:, inputs["input_ids"].shape[1]:]
        texts = self.processor.decode(new, return_format="transcription_only")
        return list(texts), _hit_cap(new, self.eos_ids, MAX_NEW_TOKENS)

    def extra_info(self) -> dict:
        return {"language": self.language, "decode": "processor.decode(return_format='transcription_only')",
                "generation_config": self.model.generation_config.to_dict() if hasattr(self, "model") else None}


class GraniteSpeech(Teacher):
    name = "granite-speech-5.0-2b"
    hf_id = "ibm-granite/granite-speech-5.0-2b"
    revision = "f4e68169fa2a5751acc4b5da9957855e5e8fc3f1"
    license = "apache-2.0"
    library = "transformers"
    user_prompt = "<|audio|>transcribe the speech with proper punctuation and capitalization."
    prompt = user_prompt

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.hf_id, revision=self.revision)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.hf_id, revision=self.revision, dtype=torch.bfloat16).to(self.device).eval()
        self.chat_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": self.user_prompt}], tokenize=False, add_generation_prompt=True)
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])

    def transcribe(self, clips, logits_processor=None, out: dict | None = None):
        """logits_processor (labeling's confidence signals): one more processor for generate, which must return the
        scores unchanged; out, if given, receives the generated ids ("new") and the end-token ids ("eos_ids")."""
        import torch

        wavs = [torch.from_numpy(np.ascontiguousarray(c, dtype=np.float32)) for c in clips]
        inputs = self.processor([self.chat_prompt] * len(wavs), wavs, device=str(self.device),
                                return_tensors="pt").to(self.device)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)
        extra = {}
        if logits_processor is not None:
            from transformers import LogitsProcessorList
            extra["logits_processor"] = LogitsProcessorList([logits_processor])
        with torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1, **extra)
        new = gen[:, inputs["input_ids"].shape[-1]:]
        if out is not None:
            out["new"], out["eos_ids"] = new, set(self.eos_ids)
        texts = self.tokenizer.batch_decode(new, add_special_tokens=False, skip_special_tokens=True)
        return [t.strip() for t in texts], _hit_cap(new, self.eos_ids, MAX_NEW_TOKENS)

    def extra_info(self) -> dict:
        return {"chat_prompt": getattr(self, "chat_prompt", None), "padding_side": "left (tokenizer config)",
                "generation_config": self.model.generation_config.to_dict() if hasattr(self, "model") else None}


class CanaryQwen(Teacher):
    name = "canary-qwen-2.5b"
    hf_id = "nvidia/canary-qwen-2.5b"
    revision = "b1469e1bba1cfe140205529c79c434ca47180960"
    license = "cc-by-4.0"
    library = "nemo (speechlm2 SALM)"
    deps = "canary-qwen-2.5b"
    llm_id = "Qwen/Qwen3-1.7B"
    llm_revision = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"  # Hub main on 2026-10-21 (tokenizer + config only)
    llm_files = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                 "vocab.json", "merges.txt"]
    prompt = "Transcribe the following: <|audioplaceholder|>"

    def local_dir(self) -> Path:
        return relabel_paths.MODELS / f"{self.name}-{self.revision[:7]}-llm-{self.llm_revision[:7]}"

    def assemble(self) -> Path:
        """MODELS/<dir>: canary config + weights, Qwen3-1.7B tokenizer at the root, its config in llm_backbone/."""
        from huggingface_hub import snapshot_download

        relabel_paths.require_mount()
        canary = Path(snapshot_download(self.hf_id, revision=self.revision,
                                        allow_patterns=["config.json", "model.safetensors"]))
        llm = Path(snapshot_download(self.llm_id, revision=self.llm_revision, allow_patterns=self.llm_files))
        out = self.local_dir()
        (out / "llm_backbone").mkdir(parents=True, exist_ok=True)
        links = {out / "config.json": canary / "config.json", out / "model.safetensors": canary / "model.safetensors",
                 out / "llm_backbone" / "config.json": llm / "config.json"}
        for name in self.llm_files:
            if name != "config.json":
                links[out / name] = llm / name
        for link, target in links.items():
            target = target.resolve()
            if link.is_symlink() or link.exists():
                if link.resolve() == target:
                    continue
                link.unlink()
            link.symlink_to(target)
        return out

    def prefetch(self) -> str:
        return str(self.assemble())

    def _load(self) -> None:
        import torch
        from nemo.collections.speechlm2.models import SALM

        local = self.assemble()
        model = SALM.from_pretrained(str(local))
        self.model = model.to(self.device).eval()
        self.model.llm.to(torch.bfloat16)
        self.model.embed_tokens.to(torch.bfloat16)
        self.tag = self.model.audio_locator_tag
        self.eos_ids = {int(self.model.text_eos_id)}
        self.cfg_pretrained_llm = str(self.model.cfg.pretrained_llm)

    def transcribe(self, clips):
        import torch

        lengths = torch.tensor([len(c) for c in clips], dtype=torch.long)
        audio = torch.zeros(len(clips), int(lengths.max()), dtype=torch.float32)
        for row, clip in enumerate(clips):
            audio[row, :len(clip)] = torch.from_numpy(np.ascontiguousarray(clip, dtype=np.float32))
        prompts = [[{"role": "user", "content": f"Transcribe the following: {self.tag}"}]] * len(clips)
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.bfloat16):
            ids = self.model.generate(prompts=prompts, audios=audio.to(self.device), audio_lens=lengths.to(self.device),
                                      max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1)
        texts = []
        for row in ids.cpu():
            tokens = row.tolist()
            cut = next((k for k, t in enumerate(tokens) if t in self.eos_ids), len(tokens))
            texts.append(self.model.tokenizer.ids_to_text(tokens[:cut]).strip())
        return texts, _hit_cap(ids.cpu(), self.eos_ids, MAX_NEW_TOKENS)

    def decoding(self) -> dict:
        return {**super().decoding(), "precision": "LLM and embeddings bfloat16, perception float32, bf16 autocast"}

    def extra_info(self) -> dict:
        return {"llm": {"id": self.llm_id, "revision": self.llm_revision, "files": self.llm_files},
                "local_dir": str(self.local_dir()), "cfg_pretrained_llm": getattr(self, "cfg_pretrained_llm", None),
                "decode": "new tokens up to the first eos, tokenizer.ids_to_text (special tokens removed)"}


REGISTRY: dict[str, type[Teacher]] = {cls.name: cls for cls in (Qwen3ASR, GraniteSpeech, CanaryQwen)}
assert list(REGISTRY) == relabel_paths.REAL_TEACHERS


def get(name: str) -> Teacher:
    teacher = REGISTRY[name]()
    if teacher.deps and os.environ.get("RELABEL_DEPS") != teacher.deps:
        raise RuntimeError(f"{name} needs RELABEL_DEPS={teacher.deps} (see launch.sh)")
    return teacher
