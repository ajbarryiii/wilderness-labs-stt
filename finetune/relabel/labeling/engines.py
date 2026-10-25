"""The labeling engines (labeling plan Part C "Engine"): T* under vLLM or the stage 1 transformers decoder, and B0.

Each engine has plan(durations) -> units (lists of indices into the caller's sorted clips; C2 sorts each shard
group by descending duration, ties by id) and run(clips of one unit) -> one output per clip, in order: (raw
text, hit the token cap) for T*, the raw text for B0. C2 decodes one unit's audio at a time. Clips
shorter than MIN_DECODE_SAMPLES are zero-padded to it first, as the stage 1 scorer did (evaluate.pad_short;
production clips are at least 1 s, so this affects only dev-set clips).

- TransformersTeacher: teachers.GraniteSpeech (the stage 1 decoder: pinned revision, bf16, the "ASR (with
  punctuation)" chat prompt, greedy, max_new_tokens 256), batched like evaluate.make_batches (descending
  duration, at most batch_size clips and max_batch_seconds of audio); a batch that runs out of GPU memory is
  split in halves (counted). Runs in the shared runtime.
- VllmTeacher: the same checkpoint, prompt and greedy settings under vLLM (bf16; temperature 0, max_tokens
  256, special tokens skipped, output stripped as in stage 1), at max_num_seqs and gpu_memory_utilization;
  clips are submitted in order in calls of submit_clips. Runs in the isolated vLLM environment
  (labeling/vllm_env.sh). Its outputs are not expected to be identical to the stage 1 decoder's; the
  equivalence check measures them. vLLM reads the checkpoint from a view of the pinned snapshot without
  VLLM_EXCLUDED_FILES (vllm_model_dir): the snapshot also holds an auxiliary CTC head (out_llm.safetensors,
  top-level "weight"/"bias", for self-speculative CTC drafts) that the stage 1 decoder never loads, and vLLM
  loads every *.safetensors file of an index-less local checkpoint, failing on those names. The snapshot is
  resolved from the local Hugging Face cache only (cached_snapshot; C2 runs offline and the cache holds only the
  files the stage 1 decoder fetched), and must hold VLLM_REQUIRED_FILES.
- B0Labeler: parakeet-ternary teacher.teacher_label_batch on the pinned B0 (bf16 encoder, FP32 greedy TDT;
  the training path), filters off, batches of at most batch_seconds padded audio (train.cut_pool rule).

Confidence signals (labeling plan "Confidence signals", 2026-10-24): with confidence on, run() returns (raw text, hit the
token cap, signals) per clip, where signals = {"token_ids", "logprob", "runner_up"}: T*'s generated token ids up to and
including its end token (all of them when it hits the cap) and, per token, the log-probability of the chosen token and of
the best other token, from the raw logits of the greedy step (vLLM: SamplingParams(logprobs=2) under logprobs_mode
"raw_logprobs"; transformers: TopTwo, a logits processor that reads each step's float32 logits and returns them
unchanged). The signals are recorded only; nothing reads them to change a label. With confidence off, run() returns
(raw text, hit the token cap) as before.

Deterministic vLLM (labeling plan, 2026-10-24): with batch_invariant (every production vLLM spec), VllmTeacher.load sets
VLLM_BATCH_INVARIANT=1 in the environment before vLLM is imported (the spawned engine core inherits it) and refuses to
start if the installed vLLM does not know the variable or does not read it as set. VllmTeacher.load is the only place
that constructs vllm.LLM (checked by a test), so every vLLM entry point runs in the mode its spec names.

The engine identity (name, versions, settings) is recorded with every output.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

MIN_DECODE_SAMPLES = 480      # evaluate.MIN_DECODE_SAMPLES (0.03 s); checked equal in the tests
MAX_NEW_TOKENS = 256          # teachers.MAX_NEW_TOKENS
SIGNAL_TOP = 2                # chosen token and runner-up
VLLM_LOGPROBS_MODE = "raw_logprobs"
VLLM_MODE_ENV = "VLLM_BATCH_INVARIANT"


def apply_vllm_mode(batch_invariant: bool) -> None:
    """Set vLLM's batch-invariant mode for this process and the engine core it spawns (before vLLM is imported)."""
    os.environ[VLLM_MODE_ENV] = "1" if batch_invariant else "0"


def check_vllm_mode(envs, batch_invariant: bool) -> None:
    """Refuse an installed vLLM that lacks the mode, or that does not read it as requested."""
    known = getattr(envs, "environment_variables", {})
    if VLLM_MODE_ENV not in known:
        raise RuntimeError(f"the installed vLLM has no {VLLM_MODE_ENV} (batch-invariant mode); refusing to run T*")
    if bool(getattr(envs, VLLM_MODE_ENV)) != bool(batch_invariant):
        raise RuntimeError(f"vLLM reads {VLLM_MODE_ENV}={getattr(envs, VLLM_MODE_ENV)!r}, the spec asks for {batch_invariant}")
VLLM_EXCLUDED_FILES = ("out_llm.safetensors",)   # T*'s auxiliary CTC head (module docstring)
# What vLLM reads from T*'s checkpoint folder: model config, the weights, the tokenizer, the audio processor.
VLLM_REQUIRED_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                       "preprocessor_config.json", "processor_config.json")


def pad_short(clip: np.ndarray) -> np.ndarray:
    if len(clip) >= MIN_DECODE_SAMPLES:
        return clip
    return np.pad(clip, (0, MIN_DECODE_SAMPLES - len(clip)))


def make_batches(durations: list[float], batch_size: int, max_batch_seconds: float) -> list[list[int]]:
    """evaluate.make_batches on an already sorted list: consecutive runs of at most batch_size clips and
    max_batch_seconds of audio."""
    batches, current, seconds = [], [], 0.0
    for i, d in enumerate(durations):
        if current and (len(current) >= batch_size or seconds + d > max_batch_seconds):
            batches.append(current)
            current, seconds = [], 0.0
        current.append(i)
        seconds += d
    if current:
        batches.append(current)
    return batches


def _signals(token_ids: list[int], chosen: list[float], runner: list[float]) -> dict:
    if not (len(token_ids) == len(chosen) == len(runner)) or not token_ids:
        raise RuntimeError(f"misaligned or empty signals ({len(token_ids)}, {len(chosen)}, {len(runner)})")
    return {"token_ids": [int(t) for t in token_ids], "logprob": [float(x) for x in chosen],
            "runner_up": [float(x) for x in runner]}


def top_two_recorder():
    """TopTwo: a transformers LogitsProcessor that keeps, per generation step and row, the top-2 log-softmax values and
    ids of the step's float32 logits and returns the scores unchanged (greedy choices are untouched). Only (steps,
    batch, 2) tensors are kept, never full-vocabulary ones."""
    import torch
    from transformers import LogitsProcessor

    class TopTwo(LogitsProcessor):
        def __init__(self):
            self.vals, self.ids = [], []

        def __call__(self, input_ids, scores):
            v, i = torch.log_softmax(scores.float(), dim=-1).topk(SIGNAL_TOP, dim=-1)
            self.vals.append(v)
            self.ids.append(i)
            return scores

    return TopTwo()


def transformers_signals(new, recorder, eos_ids: set[int]) -> list[dict]:
    """Signals per row of `new` (generated ids, batch x steps; rows padded after their end token) from a TopTwo that ran
    in the same generate call: the row's tokens up to and including its first end token, the chosen token's value and
    the best other one. Refuses a chosen token outside the recorded top-2 (not a greedy choice)."""
    import torch
    if len(recorder.vals) != new.shape[1]:
        raise RuntimeError(f"recorded {len(recorder.vals)} steps for {new.shape[1]} generated tokens")
    vals = torch.stack(recorder.vals, 1).cpu().tolist()      # batch x steps x 2
    ids = torch.stack(recorder.ids, 1).cpu().tolist()
    out = []
    for row, toks in enumerate(new.cpu().tolist()):
        n = next((k + 1 for k, t in enumerate(toks) if t in eos_ids), len(toks))
        chosen, runner = [], []
        for k in range(n):
            top = ids[row][k]
            if toks[k] not in top:
                raise RuntimeError(f"row {row} step {k}: chosen token {toks[k]} is not in the top-2 {top}")
            j = top.index(toks[k])
            chosen.append(vals[row][k][j])
            runner.append(vals[row][k][1 - j])
        out.append(_signals(toks[:n], chosen, runner))
    return out


def vllm_signals(token_ids, logprobs) -> dict:
    """Signals of one vLLM completion: per generated token a dict token id -> Logprob holding the top-2 and the sampled
    token. The runner-up is the best entry other than the chosen token."""
    token_ids = list(token_ids)
    if logprobs is None or len(logprobs) != len(token_ids):
        raise RuntimeError(f"{0 if logprobs is None else len(logprobs)} logprob entries for {len(token_ids)} tokens")
    chosen, runner = [], []
    for t, d in zip(token_ids, logprobs):
        if t not in d or len(d) < 2:
            raise RuntimeError(f"token {t}: logprobs {sorted(d)} lack the chosen token or a runner-up")
        chosen.append(d[t].logprob)
        runner.append(max(v.logprob for k, v in d.items() if k != t))
    return _signals(token_ids, chosen, runner)


def total_vram_gb() -> float | None:
    """Total memory of GPU 0 from nvidia-smi (no CUDA context), or None without a GPU."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", "0"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return int(out.splitlines()[0]) / 1024 if out else None
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def vram_fraction(declared_gb: float | None) -> float | None:
    """The process memory fraction for a declared VRAM, capped by the job lease (AGENT_JOBS_VRAM_FRACTION)."""
    import torch
    if declared_gb is None:
        return None
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    frac = min(1.0, declared_gb / total)
    lease = os.environ.get("AGENT_JOBS_VRAM_FRACTION")
    if lease and frac > float(lease) + 1e-6:
        raise RuntimeError(f"declared VRAM {declared_gb} GB is {frac:.3f} of the GPU, above the lease's {lease}")
    return frac


class TransformersTeacher:
    kind = "transformers"

    def __init__(self, batch_size: int, max_batch_seconds: float, device: str = "cuda",
                 declared_vram_gb: float | None = None, timing: bool = False, confidence: bool = False):
        self.batch_size, self.max_batch_seconds, self.device_name = batch_size, max_batch_seconds, device
        self.declared_vram_gb, self.timing, self.confidence = declared_vram_gb, timing, bool(confidence)
        self.oom_splits = 0
        self.times = {"encoder_s": 0.0, "prefill_s": 0.0, "decode_s": 0.0}

    def load(self) -> None:
        import torch
        import teachers
        from labeling import layout
        if self.device_name == "cuda":
            frac = vram_fraction(self.declared_vram_gb)
            if frac is not None:
                torch.cuda.set_per_process_memory_fraction(frac, 0)
        self.teacher = teachers.get(layout.TEACHER)
        self.teacher.load(self.device_name)
        if self.timing:
            self._hooks()

    def _hooks(self) -> None:
        """Synchronized wall time of the audio encoder + projector (encoder_s), of the rest of the first model
        forward of each generate (prefill_s: LLM prefill and head) and of the later forwards (decode_s). The
        model is GraniteSpeechForConditionalGeneration: encoder, projector and language model under `.model`.
        The synchronization slows the run slightly (probe only)."""
        import torch
        outer = self.teacher.model
        inner = getattr(outer, "model", None)
        mods = [getattr(inner, n, None) for n in ("encoder", "projector", "language_model")]
        if any(m is None for m in mods):
            raise RuntimeError("unexpected T* module structure (expected .model.encoder/.projector/.language_model)")
        on_cuda = next(outer.parameters()).is_cuda

        def sync():
            if on_cuda:
                torch.cuda.synchronize()

        state = {"enc_t0": 0.0, "call_t0": 0.0, "enc_in_call": 0.0, "calls": 0}

        def enc_pre(_m, _a):
            sync()
            state["enc_t0"] = time.perf_counter()

        def enc_post(_m, _a, _o):
            sync()
            dt = time.perf_counter() - state["enc_t0"]
            self.times["encoder_s"] += dt
            state["enc_in_call"] += dt

        def call_pre(_m, _a, _k):
            sync()
            state["call_t0"], state["enc_in_call"] = time.perf_counter(), 0.0

        def call_post(_m, _a, _k, _o):
            sync()
            dt = time.perf_counter() - state["call_t0"] - state["enc_in_call"]
            self.times["prefill_s" if state["calls"] == 0 else "decode_s"] += dt
            state["calls"] += 1

        for m in mods[:2]:
            m.register_forward_pre_hook(enc_pre)
            m.register_forward_hook(enc_post)
        outer.register_forward_pre_hook(call_pre, with_kwargs=True)
        outer.register_forward_hook(call_post, with_kwargs=True)
        self._lm_state = state

    def plan(self, durations: list[float]) -> list[list[int]]:
        return make_batches(durations, self.batch_size, self.max_batch_seconds)

    def run(self, clips: list[np.ndarray], confidence: bool | None = None) -> list[tuple]:
        """One planned batch (split in halves on out-of-memory, counted); confidence overrides the engine's setting
        (the probe's time split decodes once without the signals)."""
        import torch
        conf = self.confidence if confidence is None else bool(confidence)
        clips = [pad_short(c) for c in clips]
        texts, sig = None, None
        try:
            if self.timing:
                self._lm_state["calls"] = 0
            if conf:
                rec, out = top_two_recorder(), {}
                texts, hit = self.teacher.transcribe(clips, logits_processor=rec, out=out)
                sig = transformers_signals(out["new"], rec, out["eos_ids"])
            else:
                texts, hit = self.teacher.transcribe(clips)
        except torch.cuda.OutOfMemoryError:
            if len(clips) == 1:
                raise
            texts = None
        if texts is None:
            torch.cuda.empty_cache()
            self.oom_splits += 1
            half = len(clips) // 2
            return self.run(clips[:half], conf) + self.run(clips[half:], conf)
        if conf:
            return [(t, bool(h), s) for t, h, s in zip(texts, hit, sig)]
        return [(t, bool(h)) for t, h in zip(texts, hit)]

    def info(self) -> dict:
        return {"engine": "transformers", "batch_size": self.batch_size, "max_batch_seconds": self.max_batch_seconds,
                "declared_vram_gb": self.declared_vram_gb,
                "confidence": {"top": SIGNAL_TOP, "source": "log_softmax of each step's float32 logits (TopTwo)"}
                if self.confidence else None,
                "teacher": self.teacher.info() if hasattr(self, "teacher") else None}


def vllm_view_root() -> Path:
    from labeling import layout
    return layout.relabel_root() / "caches" / "vllm-model"


def cached_snapshot(hf_id: str, revision: str, cache_dir: Path | None = None) -> Path:
    """The pinned revision's snapshot folder in the local Hugging Face cache, without contacting the Hub and
    without requiring repository files that were never fetched; refuses a revision that is not cached, or a
    snapshot missing any of VLLM_REQUIRED_FILES (named)."""
    from huggingface_hub import try_to_load_from_cache
    found = {n: try_to_load_from_cache(hf_id, n, cache_dir=cache_dir, revision=revision) for n in VLLM_REQUIRED_FILES}
    paths = {n: Path(f) for n, f in found.items() if isinstance(f, str)}
    if not paths:
        raise RuntimeError(f"{hf_id} at revision {revision} is not in the local Hugging Face cache (run the prefetch)")
    missing = [n for n in VLLM_REQUIRED_FILES if n not in paths]
    if missing:
        raise RuntimeError(f"{hf_id} at revision {revision}: files vLLM needs are not cached: {', '.join(missing)}")
    folders = {f.parent for f in paths.values()}
    if len(folders) != 1:
        raise RuntimeError(f"{hf_id} at revision {revision}: cached files in several folders {sorted(folders)}")
    return folders.pop()


def vllm_model_dir(snapshot: Path, view: Path) -> Path:
    """`view`: links to every entry of `snapshot` except VLLM_EXCLUDED_FILES. Reused if it already holds exactly
    those links; any other existing content is refused. Built in a temporary directory and renamed into place."""
    snapshot, view = Path(snapshot), Path(view)
    want = {p.name: str(snapshot / p.name) for p in snapshot.iterdir() if p.name not in VLLM_EXCLUDED_FILES}
    if not any(n.endswith(".safetensors") for n in want):
        raise RuntimeError(f"{snapshot}: no weights left for vLLM")

    def matches() -> bool:
        have = {p.name: os.readlink(p) if p.is_symlink() else None for p in view.iterdir()}
        return have == want

    if view.exists():
        if not matches():
            raise RuntimeError(f"{view} is not a view of {snapshot} without {VLLM_EXCLUDED_FILES}; remove it")
        return view
    view.parent.mkdir(parents=True, exist_ok=True)
    tmp = view.with_name(f"{view.name}.tmp{os.getpid()}")
    tmp.mkdir()
    for name, target in want.items():
        (tmp / name).symlink_to(target)
    try:
        tmp.rename(view)
    except OSError:   # another process made it first
        for p in tmp.iterdir():
            p.unlink()
        tmp.rmdir()
        if not matches():
            raise RuntimeError(f"{view} is not a view of {snapshot} without {VLLM_EXCLUDED_FILES}; remove it")
    return view


class VllmTeacher:
    kind = "vllm"

    def __init__(self, max_num_seqs: int, gpu_memory_utilization: float, max_model_len: int = 2048,
                 submit_clips: int = 2048, seed: int = 0, confidence: bool = False, batch_invariant: bool = False):
        self.max_num_seqs, self.gpu_memory_utilization = max_num_seqs, gpu_memory_utilization
        self.max_model_len, self.submit_clips, self.seed = max_model_len, submit_clips, seed
        self.max_tokens = MAX_NEW_TOKENS
        self.confidence, self.batch_invariant = bool(confidence), bool(batch_invariant)

    def load(self) -> None:
        apply_vllm_mode(self.batch_invariant)
        import importlib
        import teachers
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from labeling import layout
        check_vllm_mode(importlib.import_module("vllm.envs"), self.batch_invariant)
        lease = os.environ.get("AGENT_JOBS_VRAM_FRACTION")
        if lease and self.gpu_memory_utilization > float(lease) + 1e-6:
            raise RuntimeError(f"gpu_memory_utilization {self.gpu_memory_utilization} is above the lease's {lease}")
        spec = teachers.REGISTRY[layout.TEACHER]
        self.hf_id, self.revision, self.user_prompt = spec.hf_id, spec.revision, spec.user_prompt
        tok = AutoTokenizer.from_pretrained(self.hf_id, revision=self.revision)
        self.chat_prompt = tok.apply_chat_template([{"role": "user", "content": self.user_prompt}], tokenize=False,
                                                   add_generation_prompt=True)
        snapshot = cached_snapshot(self.hf_id, self.revision)
        view = vllm_model_dir(snapshot, vllm_view_root() / self.hf_id.replace("/", "--") / self.revision)
        self.model_dir, self.excluded_files = str(view), list(VLLM_EXCLUDED_FILES)
        t0 = time.perf_counter()
        self.llm = LLM(model=self.model_dir, tokenizer=self.model_dir, dtype="bfloat16",
                       max_model_len=self.max_model_len, max_num_seqs=self.max_num_seqs,
                       gpu_memory_utilization=self.gpu_memory_utilization, limit_mm_per_prompt={"audio": 1},
                       seed=self.seed, logprobs_mode=VLLM_LOGPROBS_MODE)
        self.load_s = time.perf_counter() - t0
        self.SamplingParams = SamplingParams

    def _params(self, max_tokens: int, confidence: bool):
        return self.SamplingParams(temperature=0.0, max_tokens=max_tokens, skip_special_tokens=True,
                                   logprobs=SIGNAL_TOP if confidence else None)

    def reset_caches(self) -> None:
        """Drop vLLM's prefix cache, its multimodal processor and receiver caches, and its encoder-output cache, so a
        timed pass over clips decoded before is not served from them (the probe's time split)."""
        self.llm.reset_prefix_cache()
        self.llm.reset_mm_cache()
        self.llm.llm_engine.reset_encoder_cache()

    def plan(self, durations: list[float]) -> list[list[int]]:
        """Consecutive calls of submit_clips clips (vLLM batches them continuously within a call)."""
        n = len(durations)
        return [list(range(i, min(n, i + self.submit_clips))) for i in range(0, n, self.submit_clips)]

    def run(self, clips: list[np.ndarray], max_tokens: int | None = None, confidence: bool | None = None) -> list[tuple]:
        conf = self.confidence if confidence is None else bool(confidence)
        params = self._params(self.max_tokens if max_tokens is None else max_tokens, conf)
        cap = params.max_tokens
        reqs = [{"prompt": self.chat_prompt,
                 "multi_modal_data": {"audio": (np.ascontiguousarray(pad_short(c), dtype=np.float32), 16000)}}
                for c in clips]
        out = []
        for o in self.llm.generate(reqs, params, use_tqdm=False):
            c = o.outputs[0]
            res = (c.text.strip(), len(c.token_ids) >= cap and c.finish_reason == "length")
            out.append(res + (vllm_signals(c.token_ids, c.logprobs),) if conf else res)
        return out

    def info(self) -> dict:
        import vllm
        import torch
        return {"engine": "vllm", "vllm": vllm.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
                "max_num_seqs": self.max_num_seqs, "gpu_memory_utilization": self.gpu_memory_utilization,
                "max_model_len": self.max_model_len, "submit_clips": self.submit_clips, "seed": self.seed,
                "dtype": "bfloat16", "sampling": {"temperature": 0.0, "max_tokens": self.max_tokens,
                                                  "skip_special_tokens": True,
                                                  "logprobs": SIGNAL_TOP if self.confidence else None},
                "logprobs_mode": VLLM_LOGPROBS_MODE, "confidence": self.confidence,
                "batch_invariant": self.batch_invariant, VLLM_MODE_ENV: os.environ.get(VLLM_MODE_ENV),
                "hf_id": getattr(self, "hf_id", None), "revision": getattr(self, "revision", None),
                "model_dir": getattr(self, "model_dir", None), "excluded_files": getattr(self, "excluded_files", None),
                "chat_prompt": getattr(self, "chat_prompt", None), "load_s": getattr(self, "load_s", None)}


class B0Labeler:
    def __init__(self, batch_seconds: float, device: str = "cuda", declared_vram_gb: float | None = None):
        self.batch_seconds, self.device_name, self.declared_vram_gb = batch_seconds, device, declared_vram_gb

    def load(self) -> None:
        import torch
        import evaluate
        if self.device_name == "cuda":
            frac = vram_fraction(self.declared_vram_gb)
            if frac is not None:
                torch.cuda.set_per_process_memory_fraction(frac, 0)
        self.model = evaluate.load_pretrained(self.device_name)
        self.model.requires_grad_(False)

    def plan(self, durations: list[float]) -> list[list[int]]:
        """train.cut_pool: count x longest <= batch_seconds (the training batches' rule)."""
        import train  # finetune/parakeet-ternary
        return train.cut_pool(durations, self.batch_seconds)

    def run(self, clips: list[np.ndarray]) -> list[str]:
        import torch
        import teacher  # finetune/parakeet-ternary
        lengths = torch.tensor([len(c) for c in clips], dtype=torch.long)
        audio = torch.zeros(len(clips), int(lengths.max()))
        for row, c in enumerate(clips):
            audio[row, :len(c)] = torch.from_numpy(np.ascontiguousarray(c, dtype=np.float32))
        texts, _, _, _ = teacher.teacher_label_batch(self.model, audio.to(self.device_name), lengths.to(self.device_name),
                                                     [None] * len(clips), "bf16", min_s=None, max_s=None)
        return list(texts)

    def info(self) -> dict:
        import paths
        return {"engine": "parakeet-ternary teacher.teacher_label_batch", "model": paths.MODEL_ID,
                "revision": paths.MODEL_REVISION, "precision": "bf16 encoder, FP32 greedy TDT",
                "batch_seconds": self.batch_seconds, "declared_vram_gb": self.declared_vram_gb}
