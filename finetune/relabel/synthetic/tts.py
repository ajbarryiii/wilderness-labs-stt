"""Speech synthesis (synthetic plan "Speech"): the two pinned engines, voice assignment, chunked rendering.

Engines (aliases; models under $RELABEL_ROOT/synthetic/engines/<alias>/, packages in $RELABEL_ROOT/deps/<alias>):
- syn-tts-a: a VITS model trained on a studio multi-speaker corpus, run with its own package (RELABEL_DEPS=syn-tts-a,
  phonemizer espeak-ng from $RELABEL_ROOT/deps/espeak-ng-wrap/bin on PATH). Voices: the speaker-table entries
  that are documented corpus speaker ids (VOICE_ID[alias]); any other entry is never used (voicecheck.py).
- syn-tts-b: a VITS voice exported to ONNX, with a multi-speaker table (RELABEL_DEPS=syn-tts-b, onnxruntime).
Only the engines' built-in voices are used (speaker ids); no reference audio, no cloning.

Assignment (assign, seeded and balanced): within each domain the sentences are put in a seeded order and the
engines take turns; each engine walks a seeded permutation of all its voices, cyclically, so every voice is used
and voice counts differ by at most one. The plan is written once to runs/<run>/tts_plan.jsonl.

Rendering (render_chunk): one engine per job (each has its own packages). The engine is given each plan row's
tts_text (synthetic/pronounce.py; the label is never changed and never given to the engine), and a plan without
tts_text is refused. Sentences are processed in blocks of BLOCK; a block's clips go to
runs/<run>/audio/<engine>/<block>/<id>.flac (16 kHz mono PCM16: the engine's output resampled with scipy
resample_poly, peak-normalized to PEAK) and its records (the plan row, label and tts_text included, plus the audio)
to runs/<run>/tts/<engine>/<block>.jsonl, written atomically after the block, so a job resumes at the first missing
block. syn-tts-a is seeded per clip (torch.manual_seed); syn-tts-b's noise is drawn inside its ONNX graph (not
seedable).

Devices (synthetic plan "Resources", TTS on the GPU): device "cpu" (threads) or "cuda" with a declared VRAM in
GB, which labeling.engines.vram_fraction checks against the job's lease. syn-tts-a caps torch's allocator at that
fraction; syn-tts-b caps the ONNX runtime's CUDA arena at the declared bytes, with the runtime's CUDA 13 libraries
preloaded from the engine's package dir (prepare(), before anything imports torch, whose own cuDNN has the same
soname). An engine that is not actually on the GPU refuses to render.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from synthetic import layout
from synthetic.textgen import _rng, read_jsonl, write_jsonl

BLOCK = 256
PEAK = 0.891          # -1 dBFS
VOICE_ID = {"syn-tts-a": re.compile(r"p\d{3}"), "syn-tts-b": re.compile(r"p\d+")}


def verify_files(alias: str, files: list[Path]) -> None:
    """Each model file's SHA-256 must equal the registry's pinned value ("engine_files": {file name: sha256})."""
    pinned = layout.source(alias)["engine_files"]
    for f in files:
        if f.name not in pinned:
            raise RuntimeError(f"{alias}: {f.name} has no pinned SHA-256")
        got = hashlib.sha256(f.read_bytes()).hexdigest()
        if got != pinned[f.name]:
            raise RuntimeError(f"{alias}: {f} SHA-256 {got} is not the pinned {pinned[f.name]}")


def resample(x: np.ndarray, rate: int, target: int = layout.SAMPLE_RATE) -> np.ndarray:
    from scipy.signal import resample_poly
    if rate == target:
        return x.astype(np.float32)
    f = Fraction(target, rate)
    return resample_poly(x.astype(np.float64), f.numerator, f.denominator).astype(np.float32)


def peak_normalize(x: np.ndarray, peak: float = PEAK) -> np.ndarray:
    m = float(np.max(np.abs(x))) if len(x) else 0.0
    return (x * (peak / m)).astype(np.float32) if m > 0 else x.astype(np.float32)


def _check_device(device: str, declared_vram_gb: float | None) -> None:
    if device not in ("cpu", "cuda"):
        raise ValueError(f"device {device!r} (cpu or cuda)")
    if device == "cuda" and declared_vram_gb is None:
        raise ValueError("the GPU path needs a declared VRAM (--declared-vram-gb)")


def prepare(alias: str, device: str) -> None:
    """Process set-up that must come before anything imports torch (see the module docstring, "Devices")."""
    if alias == "syn-tts-b" and device == "cuda":
        import sys
        if "torch" in sys.modules:
            raise RuntimeError("torch was imported before the ONNX runtime's CUDA libraries were preloaded")
        import onnxruntime
        onnxruntime.preload_dlls(directory="")


def host_peak_gb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


class VitsEngine:
    alias = "syn-tts-a"

    def __init__(self, device: str = "cpu", threads: int = 4, declared_vram_gb: float | None = None):
        _check_device(device, declared_vram_gb)
        self.device, self.threads, self.declared_vram_gb = device, threads, declared_vram_gb

    def model_dir(self) -> Path:
        hits = sorted(layout.engine_dir(self.alias).glob("model/*/config.json"))
        if len(hits) != 1:
            raise RuntimeError(f"{self.alias}: expected one model/*/config.json under {layout.engine_dir(self.alias)}")
        return hits[0].parent

    def speaker_table(self) -> list[str]:
        return list(json.loads((self.model_dir() / "speaker_ids.json").read_text()))

    def voices(self) -> list[str]:
        return sorted(s for s in self.speaker_table() if VOICE_ID[self.alias].fullmatch(s))

    def load(self) -> None:
        import torch
        from TTS.config import load_config
        from TTS.tts.models.vits import Vits
        torch.set_num_threads(self.threads)
        if self.device == "cuda":
            from labeling.engines import vram_fraction
            torch.cuda.set_per_process_memory_fraction(vram_fraction(self.declared_vram_gb))
        m = self.model_dir()
        verify_files(self.alias, [m / "config.json", m / "model_file.pth", m / "speaker_ids.json"])
        cfg = load_config(str(m / "config.json"))
        cfg.model_args.speakers_file = str(m / "speaker_ids.json")
        cfg.speakers_file = str(m / "speaker_ids.json")
        self.model = Vits(cfg)
        self.model.load_checkpoint(cfg, str(m / "model_file.pth"), eval=True)
        self.model.to(self.device)
        if self.device == "cuda" and not next(self.model.parameters()).is_cuda:
            raise RuntimeError(f"{self.alias}: the model is not on the GPU")
        self.rate = int(cfg.audio.sample_rate)

    def vram_peak_gb(self) -> float | None:
        import torch
        return torch.cuda.max_memory_reserved() / 2**30 if self.device == "cuda" else None

    def synth(self, text: str, voice: str, seed: int) -> tuple[np.ndarray, int]:
        import torch
        torch.manual_seed(seed)
        with torch.inference_mode():
            out = self.model.synthesize(text, speaker=voice)
        return np.asarray(out["wav"], dtype=np.float32), self.rate

    def info(self) -> dict:
        import TTS
        return {"alias": self.alias, "package": f"coqui-tts {TTS.__version__}", "device": self.device,
                "declared_vram_gb": self.declared_vram_gb}


class OnnxVitsEngine:
    alias = "syn-tts-b"

    def __init__(self, device: str = "cpu", threads: int = 4, declared_vram_gb: float | None = None):
        _check_device(device, declared_vram_gb)
        self.device, self.threads, self.declared_vram_gb = device, threads, declared_vram_gb

    def files(self) -> tuple[Path, Path]:
        d = layout.engine_dir(self.alias) / "download"
        onnx = sorted(d.glob("*.onnx"))
        if len(onnx) != 1:
            raise RuntimeError(f"{self.alias}: expected one .onnx under {d}")
        return onnx[0], onnx[0].with_name(onnx[0].name + ".json")

    def speaker_table(self) -> list[str]:
        return list(json.loads(self.files()[1].read_text())["speaker_id_map"])

    def voices(self) -> list[str]:
        return sorted(s for s in self.speaker_table() if VOICE_ID[self.alias].fullmatch(s))

    def load(self) -> None:
        from piper import PiperVoice
        onnx, cfg = self.files()
        verify_files(self.alias, [onnx, cfg])
        if self.device == "cuda":
            import onnxruntime
            from labeling.engines import vram_fraction
            from piper.config import PiperConfig
            vram_fraction(self.declared_vram_gb)          # refuses a declaration above the job's lease
            opts = onnxruntime.SessionOptions()
            opts.intra_op_num_threads = self.threads
            cuda = {"device_id": 0, "gpu_mem_limit": int(self.declared_vram_gb * 2**30),
                    "arena_extend_strategy": "kSameAsRequested", "cudnn_conv_algo_search": "HEURISTIC"}
            session = onnxruntime.InferenceSession(str(onnx), sess_options=opts,
                                                   providers=[("CUDAExecutionProvider", cuda), "CPUExecutionProvider"])
            if session.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError(f"{self.alias}: the ONNX runtime did not start on the GPU ({session.get_providers()})")
            session.disable_fallback()                     # a CUDA failure while rendering raises, never reruns on CPU
            self.voice = PiperVoice(session=session, config=PiperConfig.from_dict(json.loads(cfg.read_text())))
        else:
            self.voice = PiperVoice.load(str(onnx), config_path=str(cfg), use_cuda=False)
        self.ids = self.voice.config.speaker_id_map
        self.rate = int(self.voice.config.sample_rate)

    def synth(self, text: str, voice: str, seed: int) -> tuple[np.ndarray, int]:
        from piper import SynthesisConfig
        chunks = list(self.voice.synthesize(text, syn_config=SynthesisConfig(speaker_id=self.ids[voice])))
        audio = np.concatenate([c.audio_float_array for c in chunks]) if chunks else np.zeros(0, np.float32)
        return audio.astype(np.float32), self.rate

    def info(self) -> dict:
        import onnxruntime
        return {"alias": self.alias, "package": "onnx vits", "onnxruntime": onnxruntime.__version__,
                "device": self.device, "providers": self.voice.session.get_providers(),
                "declared_vram_gb": self.declared_vram_gb}

    def vram_peak_gb(self) -> float | None:
        return None    # the job supervisor records the process's VRAM peak


ENGINES = {"syn-tts-a": VitsEngine, "syn-tts-b": OnnxVitsEngine}


def assign(sentences: list[dict], voices: dict[str, list[str]], seed: int) -> list[dict]:
    """[{id, domain, label, engine, voice, seed}] (module docstring, "Assignment")."""
    engines = list(layout.TTS_ENGINES)
    perms = {}
    for e in engines:
        order = sorted(voices[e])
        if not order:
            raise RuntimeError(f"{e}: no voices")
        _rng(seed, "voices", e).shuffle(order)
        perms[e] = order
    counters = {e: 0 for e in engines}
    out = []
    for d in layout.DOMAINS:
        rows = sorted((s for s in sentences if s["domain"] == d), key=lambda s: s["id"])
        _rng(seed, "order", d).shuffle(rows)
        for i, s in enumerate(rows):
            e = engines[i % len(engines)]
            v = perms[e][counters[e] % len(perms[e])]
            counters[e] += 1
            clip_seed = int.from_bytes(hashlib.sha256(f"{seed}|{s['id']}".encode()).digest()[:4], "big")
            out.append({"id": s["id"], "domain": d, "label": s["label"], "engine": e, "voice": v, "seed": clip_seed})
    out.sort(key=lambda r: r["id"])
    return out


def write_flac(path: Path, audio: np.ndarray) -> str:
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.round(audio * 32767.0), -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, pcm, layout.SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    data = buf.getvalue()
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def engine_blocks(plan: list[dict], engine: str) -> list[list[dict]]:
    rows = [r for r in plan if r["engine"] == engine]
    return [rows[i:i + BLOCK] for i in range(0, len(rows), BLOCK)]


def render_chunk(run_dir: Path, engine_obj, budget_minutes: float, log=print) -> dict:
    plan = read_jsonl(run_dir / "tts_plan.jsonl")
    alias = engine_obj.alias
    missing = [r["id"] for r in plan if r["engine"] == alias and "tts_text" not in r]
    if missing:
        raise RuntimeError(f"{len(missing)} plan rows lack tts_text (first {missing[0]}); rerun tts-plan")
    t0 = time.monotonic()
    total_audio = 0.0
    todo = [(k, b) for k, b in enumerate(engine_blocks(plan, alias))
            if not (run_dir / "tts" / alias / f"{k:05d}.jsonl").exists()]
    if todo:
        engine_obj.load()
    done = 0
    for k, block in todo:
        if (time.monotonic() - t0) / 60 > budget_minutes:
            break
        tb, rows, synth_s, audio_s = time.monotonic(), [], 0.0, 0.0
        for r in block:
            ts = time.monotonic()
            audio, rate = engine_obj.synth(r["tts_text"], r["voice"], r["seed"])
            synth_s += time.monotonic() - ts
            audio = peak_normalize(resample(audio, rate))
            path = run_dir / "audio" / alias / f"{k:05d}" / f"{r['id']}.flac"
            sha = write_flac(path, audio)
            dur = len(audio) / layout.SAMPLE_RATE
            audio_s += dur
            rows.append({**r, "audio_filepath": str(path), "duration": dur, "sha256": sha, "engine_rate": rate})
        write_jsonl(run_dir / "tts" / alias / f"{k:05d}.jsonl", rows)
        done += 1
        total_audio += audio_s
        vram = getattr(engine_obj, "vram_peak_gb", lambda: None)()
        log(f"tts {alias} block {k}: {len(rows)} clips, {audio_s / 3600:.3f} h audio, synth {synth_s:.0f} s "
            f"(RTF {synth_s / max(audio_s, 1e-9):.3f}), block {time.monotonic() - tb:.0f} s; job so far "
            f"{total_audio / max(time.monotonic() - t0, 1e-9):.1f} audio h per wall h, host peak {host_peak_gb():.2f} GB"
            + ("" if vram is None else f", VRAM peak reserved {vram:.2f} GB"))
    remaining = sum(1 for k, _ in enumerate(engine_blocks(plan, alias))
                    if not (run_dir / "tts" / alias / f"{k:05d}.jsonl").exists())
    return {"engine": alias, "blocks_done": done, "blocks_remaining": remaining, "info": engine_obj.info() if todo else None}
