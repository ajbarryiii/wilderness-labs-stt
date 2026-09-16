"""Bounded encoder GEMM geometry search; no changes to weight decoding or math.

The caller serializes GPU access. ``check`` times up to six configurations on
three full encoder shapes, freezes selected configurations, and records every
timing. ``install`` reuses the frozen choices; full-model energy, not these
short latency measurements, determines whether the parent keeps the candidate.
"""
from pathlib import Path
import json
import statistics
import time

import torch
import triton

from . import packed

_ORIGINAL = packed._gemm
_CONFIGS = ((64, 64, 64, 4, 3), (64, 128, 64, 4, 3),
            (64, 64, 128, 4, 3), (64, 64, 64, 4, 2),
            (32, 64, 64, 4, 2), (32, 128, 64, 4, 2))
_SHAPES = ((1500, 1024, 1024), (1500, 4096, 1024), (1500, 1024, 4096))


def variants():
    return ["encoder_gemm_tiles"]


def _key(bits, m, n, k):
    return f"b{bits}-m{m}-n{n}-k{k}"


class _TunedGemm:
    def __init__(self, original, choices):
        self.original, self.choices = original, choices

    def __getitem__(self, original_grid):
        def launch(*args, **kwargs):
            m, n, k, _, bits = args[5:10]
            config = self.choices.get(_key(bits, m, n, k))
            if config is None:
                return self.original[original_grid](*args, **kwargs)
            bm, bn, bk, warps, stages = config
            changed = (*args[:11], bm, bn, bk, args[14])
            grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
            options = {**kwargs, "num_warps": warps, "num_stages": stages}
            return self.original[grid](*changed, **options)
        return launch


def install(name, artifact_dir):
    if name not in variants():
        raise ValueError(name)
    policy = json.loads((Path(artifact_dir) / "encoder-gemm-choices.json").read_text())
    original = packed._gemm
    packed._gemm = _TunedGemm(original, policy["choices"])

    def restore():
        packed._gemm = original

    return restore


def _launch(config, p, x, bias, out):
    bm, bn, bk, warps, stages = config
    m = x.shape[0]
    _ORIGINAL[(triton.cdiv(m, bm) * triton.cdiv(p.n, bn),)](
        x, p.words_t, p.scales, bias, out, m, p.n, p.k,
        p.words.shape[1], p.bits, True, bm, bn, bk, 8,
        num_warps=warps, num_stages=stages)


def check(name, artifact_dir, *, frozen_choices=None):
    if name not in variants():
        raise ValueError(name)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.manual_seed(715087)
    folder = Path(artifact_dir)
    folder.mkdir(parents=True, exist_ok=True)
    frozen_path = Path(__file__).with_name("sprint_gemm_policy.json")
    frozen = (frozen_choices if frozen_choices is not None else
              json.loads(frozen_path.read_text())["choices"] if frozen_path.exists() else None)
    allowed_keys = {_key(bits, *shape) for shape in _SHAPES for bits in (1, 2)}
    if frozen is not None and (set(frozen) != allowed_keys
                              or any(tuple(c) not in _CONFIGS for c in frozen.values())):
        raise ValueError("Frozen GEMM policy is outside this candidate's search space")
    deadline = float("inf") if frozen is not None else time.monotonic() + 45.0
    records, choices = [], {}
    for m, n, k in _SHAPES:
        for bits in (1, 2):
            key = _key(bits, m, n, k)
            choices[key] = tuple(frozen[key]) if frozen is not None else _CONFIGS[0]
            if time.monotonic() >= deadline:
                records.append({"shape": key, "status": "untuned-budget", "config": _CONFIGS[0]})
                continue
            codes = (torch.randint(0, 2, (n, k), device="cuda", dtype=torch.int8) * 2 - 1
                     if bits == 1 else torch.randint(-1, 2, (n, k), device="cuda", dtype=torch.int8))
            p = packed.PackedWeight.from_codes(codes, bits, 1 / (k ** .5))
            x = torch.randn(m, k, device="cuda", dtype=torch.float16) * .2
            bias = torch.linspace(-.1, .1, n, device="cuda", dtype=torch.float16)
            out = torch.empty((m, n), device="cuda", dtype=torch.float16)
            reference = torch.empty_like(out)
            _launch(_CONFIGS[0], p, x, bias, reference)
            results = []
            configurations = (_CONFIGS if frozen is None else
                              tuple(dict.fromkeys((_CONFIGS[0], tuple(frozen[key])))))
            for config in configurations:
                if results and time.monotonic() >= deadline:
                    break
                try:
                    _launch(config, p, x, bias, out)
                except triton.OutOfResources as exc:
                    records.append({"shape": key, "config": config, "status": "resources", "error": str(exc)})
                    continue
                torch.testing.assert_close(out, reference, atol=.003, rtol=.003)
                for _ in range(3):
                    _launch(config, p, x, bias, out)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(16):
                        _launch(config, p, x, bias, out)
                if frozen is not None:
                    original_x = x.clone()
                    x.zero_()
                    graph.replay()
                    torch.testing.assert_close(out, bias.expand_as(out), atol=0, rtol=0)
                    x.copy_(original_x)
                    graph.replay()
                    torch.testing.assert_close(out, reference, atol=.003, rtol=.003)
                    del original_x
                samples = []
                for _ in range(3):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(4):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    samples.append(begin.elapsed_time(end) * 1000 / 64)
                us = statistics.median(samples)
                results.append((us, config))
                records.append({"shape": key, "config": config, "status": "measured",
                                "median_us": us, "samples_us": samples})
                del graph
            if results and frozen is None:
                # Avoid changing geometry for a marginal microbenchmark win.
                baseline_us = next(us for us, c in results if c == _CONFIGS[0])
                best_us, best_config = min(results)
                choices[key] = best_config if best_us < baseline_us * .97 else _CONFIGS[0]
            del codes, p, x, bias, out, reference
    torch.cuda.synchronize()
    result = {"variant": name, "choices": choices, "profiles": records,
              "profile_admission_seconds": 45, "minimum_screen_latency_gain": .03,
              "frozen_policy": "source literal" if frozen_choices is not None else
                  str(frozen_path) if frozen is not None else None,
              "scope": "warm weight kernel latency; parent measures full-model energy"}
    (folder / "encoder-gemm-choices.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def write_frozen_policy(profile_path, destination):
    """Export measured choices as a separately named, source-hashed candidate."""
    profile = json.loads(Path(profile_path).read_text())
    choices = {k: tuple(v) for k, v in profile["choices"].items()}
    allowed_keys = {_key(bits, *shape) for shape in _SHAPES for bits in (1, 2)}
    if set(choices) != allowed_keys or any(v not in _CONFIGS for v in choices.values()):
        raise ValueError("Cannot export a policy outside the frozen search space")
    source = '''"""Fixed encoder GEMM geometry selected by an earlier measured trial."""
from . import packed, sprint_gemm

CHOICES = ''' + repr(choices) + '''

def variants():
    return ["encoder_gemm_selected"]

def install(name, artifact_dir):
    if name not in variants():
        raise ValueError(name)
    original = packed._gemm
    packed._gemm = sprint_gemm._TunedGemm(original, CHOICES)
    def restore():
        packed._gemm = original
    return restore

def check(name, artifact_dir):
    if name not in variants():
        raise ValueError(name)
    result = sprint_gemm.check("encoder_gemm_tiles", artifact_dir, frozen_choices=CHOICES)
    result["variant"] = name
    return result
'''
    Path(destination).write_text(source)
    return choices
