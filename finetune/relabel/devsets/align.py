"""Forced alignment of transcript units to audio with torchaudio's MMS aligner (labeling plan Part A).

The aligner is independent of every evaluated model: torchaudio.pipelines.MMS_FA (a wav2vec2 CTC model over
lowercase Latin letters and the apostrophe, plus a wildcard "star" token whose log-probability is 0 in every
frame). Frames are 20 ms (hop 320 samples at 16 kHz, receptive field 400 samples).

Emissions are computed in 20 s chunks with 2 s of context on each side (the full-attention encoder cannot
take an hour of audio at once); only each chunk's central frames are kept, so every frame comes from one
forward pass with at least 2 s of context where the audio allows.

Alignment runs in blocks along the recording, so long recordings (earnings calls, ~1 h) need neither the
whole trellis in memory nor one huge path. A block of units with about BLOCK_SECONDS of estimated speech is
aligned, with a wildcard before and after it, in a window that starts at the anchor (the end of the last
accepted unit) and spans WINDOW_FACTOR times the block's estimated length plus SLACK_SECONDS. Its last
OVERLAP_SECONDS (estimated) of units are never accepted from this block; they are re-aligned at the start of
the next one, away from the window end. Of the rest, only the well-aligned prefix is accepted: the units
before the first run of BAD_RUN consecutive units scoring below MIN_SCORE (wildcard units do not count).
A window too short for the block (sparse speech, such as one side of a conversation, spans more audio than
its text suggests) crams the block's tail into it and ends the prefix early, so the block is re-aligned in
windows RETRY_FACTORS times as long (at most MAX_WINDOW_SECONDS) until its whole accepted part is well
aligned or the window reaches the end of the audio; the attempt with the longest prefix is kept. The
anchor moves to the end of the accepted prefix and the next block starts after it. If even the best attempt
starts with a bad run (text with no matching audio, or audio the aligner cannot follow), that run is emitted
as it was aligned (its segments then fail), the anchor stays, the next window grows by the run's estimated
length, and the leading wildcard lets the next block skip unaligned audio.

Per unit: start and end time (s) and a confidence score in (0, 1] (None for wildcard units):
exp(mean over the frames the path assigns to the unit's characters of (log p(that character) - max log p(any
non-wildcard label))), the geometric mean of how close the forced characters are to the emission's own best
labels (the CTC-segmentation confidence idea). Blank frames are left out, so a misplaced word cannot hide
behind the silence around it. Validation (development notes, not a test): 150 consecutive talk clips
(tech-dev-1 dev audio) concatenated into three 5-6 min recordings and aligned long-form, with every 7th
clip's text replaced by another clip's: clip-mean scores of correct clips >= 0.424 (1% quantile 0.585),
of replaced clips <= 0.309 (median 0.03); MIN_SCORE = 0.35 separates them. The mean of exp(.) over all
frames, blanks included, did not (replaced clips' median 0.52). Aligned word boundaries were within 0.53 s
of the clip edges at the 95th percentile (1.3 s at most), which includes the clips' own silence margins.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
HOP = 320
RECEPTIVE_FIELD = 400
FRAME_SECONDS = HOP / SAMPLE_RATE
CHUNK_FRAMES = 1000     # 20 s per forward pass ...
CONTEXT_FRAMES = 100    # ... with 2 s of context on each side
BLOCK_SECONDS = 40.0
OVERLAP_SECONDS = 8.0
WINDOW_FACTOR = 2.0
SLACK_SECONDS = 30.0
SECONDS_PER_CHAR = 0.075
SECONDS_PER_UNIT = 0.15
WILDCARD_SECONDS = 0.5
RETRY_FACTORS = (1, 3, 9)
BAD_RUN = 3
MAX_WINDOW_SECONDS = 600.0
MIN_SCORE = 0.35  # also the segment acceptance threshold (segment.py)
# The torchaudio MMS_FA checkpoint (pipelines.MMS_FA._path), pinned by content.
CHECKPOINT_SHA256 = "20ef12963ab4924bef49ac4fc7f58ad5da2ee43b2c11bc8c853c9b90ecdbc680"


@dataclass(frozen=True)
class UnitSpan:
    start: float          # seconds from the start of the aligned audio
    end: float
    score: float | None   # None for a wildcard unit


def frame_count(samples: int) -> int:
    return 0 if samples < RECEPTIVE_FIELD else (samples - RECEPTIVE_FIELD) // HOP + 1


def estimated_frames(chars: str | None) -> int:
    seconds = WILDCARD_SECONDS if chars is None else len(chars) * SECONDS_PER_CHAR + SECONDS_PER_UNIT
    return max(1, round(seconds / FRAME_SECONDS))


def good_prefix(scores: list[float | None], min_score: float) -> tuple[int, int]:
    """(p, q): units [0, p) are well aligned; [p, q) is the first run of BAD_RUN or more consecutive units
    scoring below min_score (p = q = len(scores) if there is none). Wildcard units (None) neither start nor
    break a run."""
    run_start, bad = None, 0
    for u, score in enumerate(scores):
        if score is None:
            continue
        if score < min_score:
            if bad == 0:
                run_start = u
            bad += 1
        else:
            if bad >= BAD_RUN:
                return run_start, u
            bad = 0
    if bad >= BAD_RUN:
        return run_start, len(scores)
    return len(scores), len(scores)


def _repeats(targets: list[int]) -> int:
    return sum(1 for a, b in zip(targets, targets[1:]) if a == b)


class BlockAligner:
    """The block algorithm over a precomputed emission (log-probabilities [T, C]); see the module docstring."""

    def __init__(self, dictionary: dict[str, int], star: int, blank: int = 0, min_score: float = MIN_SCORE) -> None:
        self.dictionary, self.star, self.blank, self.min_score = dictionary, star, blank, min_score

    def targets(self, chars: str | None) -> list[int]:
        return [self.star] if chars is None else [self.dictionary[c] for c in chars]

    def _attempt(self, emission, best, anchor: int, end: int, tokens: list[int], lengths: list[int],
                 units: list[str | None]) -> list[UnitSpan]:
        """Spans of the block's units aligned in frames [anchor, end) between two wildcards."""
        import torch
        import torchaudio.functional as F

        log_probs = emission[anchor:end].unsqueeze(0)
        path, scores = F.forced_align(log_probs, torch.tensor([tokens], dtype=torch.int32), blank=self.blank)
        path, scores = path[0], scores[0]
        spans = F.merge_tokens(path, scores.exp(), blank=self.blank)
        assert len(spans) == len(tokens), (len(spans), len(tokens))
        diff = scores - best[anchor:end]  # log p(path label) - log p(best non-wildcard label), <= 0
        out, pos = [], 1
        for chars, length in zip(units, lengths):
            group = spans[pos:pos + length]
            pos += length
            a, b = group[0].start, group[-1].end
            score = None if chars is None else \
                float(torch.cat([diff[s.start:s.end] for s in group]).mean().exp())
            out.append(UnitSpan((anchor + a) * FRAME_SECONDS, (anchor + b) * FRAME_SECONDS, score))
        return out

    def align_emission(self, emission, units: list[str | None]) -> list[UnitSpan]:
        import torch

        emission = emission.float()
        keep = torch.ones(emission.shape[1], dtype=torch.bool)
        keep[self.star] = False
        best = emission[:, keep].max(dim=1).values  # best non-wildcard label per frame
        total = int(emission.shape[0])
        n = len(units)
        out: list[UnitSpan | None] = [None] * n
        est = [estimated_frames(u) for u in units]
        block = round(BLOCK_SECONDS / FRAME_SECONDS)
        overlap = round(OVERLAP_SECONDS / FRAME_SECONDS)
        slack = round(SLACK_SECONDS / FRAME_SECONDS)
        i, anchor, backlog = 0, 0, 0
        while i < n:
            j, acc = i, 0
            while j < n and (j == i or acc + est[j] <= block):
                acc += est[j]
                j += 1
            k = j  # units [i, k) are accepted; [k, j) are re-aligned with the next block
            if j < n:
                tail = 0
                while k - 1 > i and tail + est[k - 1] <= overlap:
                    tail += est[k - 1]
                    k -= 1
            tokens, lengths = [self.star], []
            for u in units[i:j]:
                t = self.targets(u)
                lengths.append(len(t))
                tokens += t
            tokens.append(self.star)
            need = len(tokens) + _repeats(tokens)
            base = round(WINDOW_FACTOR * acc) + slack + backlog
            best_spans, best_cut = None, (-1, 0)
            for factor in RETRY_FACTORS:
                end = min(total, anchor + base * factor, anchor + round(MAX_WINDOW_SECONDS / FRAME_SECONDS))
                if end - anchor < need:
                    continue
                spans = self._attempt(emission, best, anchor, end, tokens, lengths, units[i:j])
                cut = good_prefix([s.score for s in spans[:k - i]], self.min_score)
                if cut[0] > best_cut[0]:
                    best_spans, best_cut = spans, cut
                if cut[0] == k - i or end >= total:
                    break
            if best_spans is None:  # the remaining audio cannot hold these units
                at = anchor * FRAME_SECONDS
                for u in range(i, n):
                    out[u] = UnitSpan(at, at, None if units[u] is None else 0.0)
                break
            p, q = best_cut
            if p > 0:  # accept the well-aligned prefix; the rest is re-aligned from its end
                out[i:i + p] = best_spans[:p]
                anchor = round(best_spans[p - 1].end / FRAME_SECONDS)
                backlog = 0
                i += p
            else:  # the block starts with a bad run in every attempt: emit it as aligned, keep the anchor
                out[i:i + q] = best_spans[:q]
                backlog += sum(est[i:i + q])
                i += q
        assert all(o is not None for o in out)
        return out  # type: ignore[return-value]


def checkpoint_path() -> Path:
    import torch
    from torchaudio.pipelines import MMS_FA

    return Path(torch.hub.get_dir()) / "checkpoints" / Path(MMS_FA._path).name


class MMSAligner(BlockAligner):
    """torchaudio MMS_FA (loaded once), checkpoint pinned by SHA-256. The model, including the bundle's own
    waveform normalization, runs on `device` (the CPU or the GPU, with TF32 disabled there); emissions come back
    to the CPU as float32 and the block alignment always runs on the CPU, so the devices differ only by
    floating-point reduction order."""

    def __init__(self, threads: int | None = None, min_score: float = MIN_SCORE, device: str = "cpu") -> None:
        import torch
        from torchaudio.pipelines import MMS_FA

        if threads:
            torch.set_num_threads(threads)
        if device != "cpu":
            if not torch.cuda.is_available():
                raise RuntimeError(f"aligner device {device!r} requested but CUDA is not available")
            torch.backends.cuda.matmul.allow_tf32 = False   # full float32, like the CPU
            torch.backends.cudnn.allow_tf32 = False
        self.device = device
        self.model = MMS_FA.get_model(with_star=True).eval().to(device)  # downloads to $TORCH_HOME once
        path = checkpoint_path()
        with open(path, "rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != CHECKPOINT_SHA256:
            raise RuntimeError(f"MMS_FA checkpoint {path} has SHA-256 {digest}, not the pinned {CHECKPOINT_SHA256}")
        dictionary = MMS_FA.get_dict(star="*")
        super().__init__({k: v for k, v in dictionary.items() if k not in ("-", "*")}, dictionary["*"],
                         dictionary["-"], min_score)
        self.identity = {"aligner": "torchaudio.pipelines.MMS_FA (with_star)", "url": MMS_FA._path,
                         "sha256": digest, "device": device}

    def emission(self, audio: np.ndarray):
        """Log-probabilities [T, C] (CPU float32) of the whole recording, chunked (module docstring)."""
        import torch

        samples = len(audio)
        total = frame_count(samples)
        wave = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        out = None
        with torch.inference_mode():
            for c0 in range(0, total, CHUNK_FRAMES):
                c1 = min(total, c0 + CHUNK_FRAMES)
                f0, f1 = max(0, c0 - CONTEXT_FRAMES), min(total, c1 + CONTEXT_FRAMES)
                # The last chunk runs to the end of the audio (the remainder after the last frame adds no
                # frame but belongs to the waveform the model normalizes).
                last = samples if f1 == total else (f1 - 1) * HOP + RECEPTIVE_FIELD
                chunk = wave[f0 * HOP:last]
                logits, _ = self.model(chunk.unsqueeze(0).to(self.device))
                logits = logits.float().cpu()
                if logits.shape[1] != f1 - f0:
                    raise RuntimeError(f"emission frames {logits.shape[1]} != {f1 - f0}")
                if out is None:
                    out = torch.empty(total, logits.shape[-1], dtype=torch.float32)
                out[c0:c1] = logits[0, c0 - f0:c1 - f0]
        if out is None:
            raise ValueError(f"audio of {samples} samples is too short to align")
        return out

    def align(self, audio: np.ndarray, units: list[str | None]) -> list[UnitSpan]:
        if not units:
            return []
        return self.align_emission(self.emission(audio), units)


class LazyAligner:
    """Loads MMSAligner on first use (sources whose segments all come from source timestamps never load it)."""

    def __init__(self, threads: int | None = None, device: str = "cpu") -> None:
        self.threads, self.device, self._aligner = threads, device, None

    @property
    def aligner(self) -> MMSAligner:
        if self._aligner is None:
            self._aligner = MMSAligner(self.threads, device=self.device)
        return self._aligner

    @property
    def min_score(self) -> float:
        return MIN_SCORE

    def align(self, audio: np.ndarray, units: list[str | None]) -> list[UnitSpan]:
        return self.aligner.align(audio, units)
