"""Fixtures for the test_synthetic_* contract tests (no tests here).

A temporary $RELABEL_ROOT and $PRIVATE_REGISTRY (fake aliases only), small term lists, held-out manifests,
generator records and audio clips. Nothing here touches the production roots.
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from synthetic import layout, vocab

LISTS = {
    "chemistry": ["enthalpy", "Gibbs free energy", "free energy", "rate constant", "rate constants", "catalyst",
                  "catalysts", "activation energy", "Arrhenius equation", "equilibrium constant", "titration",
                  "electronegativity", "stoichiometry", "ketamine synthesis", "tourniquet chemistry"],
    "mathematics": ["eigenvalue", "eigenvalues", "eigenvector", "determinant", "singular value decomposition",
                    "central limit theorem", "standard deviation", "Bayes' theorem", "partial derivative",
                    "Fourier transform", "Markov chain", "stoichiometry"],
    "finance": ["EBITDA", "free cash flow", "net present value", "yield curve", "derivatives", "hedge fund",
                "price-to-earnings ratio", "working capital", "basis points", "Sharpe ratio"],
    "software_engineering": ["dependency injection", "race condition", "deadlock", "hash table", "hash tables",
                             "unit testing", "binary search", "garbage collector", "dynamic programming",
                             "load balancer", "event loop", "memoization"],
}


def normalize():
    return vocab.default_normalize()


def make_split(lists=LISTS, share=layout.TARGETED_SHARE, salt=layout.SPLIT_SALT) -> vocab.Split:
    n = normalize()
    data = vocab.build_split(lists, n, vocab.medical_matcher(n), share=share, salt=salt)
    return vocab.Split(data, n)


@contextlib.contextmanager
def env(**values):
    old = {k: os.environ.get(k) for k in values}
    os.environ.update({k: str(v) for k, v in values.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def roots(registry: dict | None = None):
    """A temporary RELABEL_ROOT (+ DATA_ROOT, PRIVATE_REGISTRY with `registry` as sources.json)."""
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        (t / "registry").mkdir()
        (t / "registry" / "sources.json").write_text(json.dumps(registry or {}))
        with env(RELABEL_ROOT=t / "relabel", DATA_ROOT=t / "data", PRIVATE_REGISTRY=t / "registry"):
            (t / "relabel").mkdir()
            (t / "data").mkdir()
            yield t


def write_manifest(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"id": r.get("id", f"x:{i}"), "text": r["text"], "audio_filepath": "/dev/null",
                                "duration": 1.0}) + "\n")
    return path


def heldout_sets(tmp: Path, texts: dict[str, list[str]]):
    from pool import heldout as ho
    paths = [(name, write_manifest(tmp / "heldout" / f"{name}.jsonl", [{"text": t} for t in ts]))
             for name, ts in texts.items()]
    return ho.load_sets(paths)


def record(i: int, domain: str, raw: str, requested: list[str], register: str = "explanatory") -> dict:
    return {"id": f"smoke-{i:06d}", "index": i, "domain": domain, "register": register, "requested": requested,
            "raw": raw}


def tone(seconds: float, freq: float = 220.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * layout.SAMPLE_RATE)) / layout.SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def write_clip(path: Path, audio: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, layout.SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    return path
