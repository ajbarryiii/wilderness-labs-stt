"""Optional, explicit installation of a measured kernel policy.

This module does not alter the benchmark's default kernels. Install a policy
before constructing/capturing a packed model, and keep the installation active
while using that model. CUDA graphs retain their captured kernels even after
Python dispatch is restored; recapture a model when switching policies.

``with install("ternary") as kernels: ...`` uses the confirmed recipe once a
completed experiment has supplied CONFIRMED_RECEIPT. Receipts bind an explicit
plugin list to source hashes and a completed comparison report. A source change
requires a new receipt or an explicit, unconfirmed policy installation.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

from paths import ROOT, artifact, digest, storage

# Three fresh paired confirmations per model passed on the measured source.
CONFIRMED_RECEIPT: str | None = (
    "/mnt/hd/wilderness-labs-stt/inference-efficiency/agent-kernel-sprint/"
    "20260912T160024Z/confirmed-receipt.json"
)
_ACTIVE = None
_HERE = Path(__file__).resolve().parent.parent
_ALLOWED = {
    "kernels.sprint_cuda", "kernels.sprint_fusion", "kernels.sprint_qkv",
    "kernels.sprint_fused_vector", "kernels.sprint_gemm",
    "kernels.sprint_bitcast", "kernels.sprint_attention",
    "kernels.sprint_gemm_frozen", "kernels.sprint_gemm_selected",
}


def _policy(plugins):
    if not isinstance(plugins, list) or not plugins:
        raise ValueError("A kernel policy must contain at least one explicit plugin")
    result = []
    for plugin in plugins:
        if (not isinstance(plugin, dict) or set(plugin) != {"module", "name"}
                or plugin["module"] not in _ALLOWED or not isinstance(plugin["name"], str)):
            raise ValueError(f"Invalid kernel plugin: {plugin!r}")
        result.append(dict(plugin))
    return result


def _snapshot():
    """Capture every dispatch point used by the measured sprint plugins."""
    from runtime_model import _Attention, _Block
    import torch.nn.functional as F
    from . import packed, sprint_fusion, sprint_qkv
    points = ((packed.PackedWeight, "linear"), (packed, "_gemm"),
              (F, "scaled_dot_product_attention"), (_Attention, "__call__"),
              (_Block, "__call__"), (sprint_fusion, "_linear_epilogue"),
              (sprint_fusion, "_attention_residual"), (sprint_qkv, "_project"))
    return [(owner, name, getattr(owner, name)) for owner, name in points]


def _restore(snapshot):
    for owner, name, value in reversed(snapshot):
        setattr(owner, name, value)


def _verify_sources(source_hashes):
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("Confirmed receipt has no measured source hashes")
    for relative, expected in source_hashes.items():
        path = (_HERE / relative).resolve()
        if not path.is_relative_to(_HERE.resolve()) or not path.is_file():
            raise ValueError(f"Invalid measured source path: {relative}")
        if digest(path) != expected:
            raise RuntimeError(f"Kernel source differs from the confirmed experiment: {relative}")


class Installation:
    """Process-local dispatch installation, with idempotent restoration."""
    def __init__(self, metadata, snapshot):
        self.metadata = metadata
        self._snapshot = snapshot
        self._restored = False

    def restore(self):
        global _ACTIVE
        if self._restored:
            return
        _restore(self._snapshot)
        self._restored = True
        if _ACTIVE is self:
            _ACTIVE = None

    def __enter__(self):
        if self._restored:
            raise RuntimeError("Cannot reenter an already restored kernel installation")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.restore()


def install(distribution, artifact_dir=None, *, receipt=None, plugins=None):
    """Install confirmed kernels or an explicitly supplied unconfirmed policy.

    ``distribution`` selects the independently measured binary or ternary arm.
    ``receipt`` defaults to the confirmed run's immutable JSON recipe. Supplying
    ``plugins`` is an explicit experimental override and cannot simultaneously
    claim confirmation. Only one installation may be active in this process.
    """
    global _ACTIVE
    if distribution not in {"binary", "ternary"}:
        raise ValueError("distribution must be 'binary' or 'ternary'")
    if _ACTIVE is not None:
        raise RuntimeError("Restore the active kernel installation before installing another")
    if plugins is not None and receipt is not None:
        raise ValueError("Choose an explicit experimental policy or a confirmed receipt")
    storage()
    receipt_path = None
    record = None
    if plugins is None:
        receipt_path = receipt or CONFIRMED_RECEIPT
        if receipt_path is None:
            raise RuntimeError("No kernel policy has been independently confirmed yet")
        receipt_path = artifact(receipt_path)
        record = json.loads(receipt_path.read_text())
        if record.get("schema") != 1:
            raise ValueError("Unsupported kernel receipt schema")
        selected = record["policies"][distribution]
        if not selected.get("confirmed", False):
            raise RuntimeError(f"The receipt does not confirm the {distribution} policy")
        plugins = selected["plugins"]
        source_hashes = record["source_hashes"]
        _verify_sources(source_hashes)
    else:
        # Include dependencies so existing cubins cannot be silently reused
        # after a source edit. Preparation caches live outside the repository.
        sources = [p for p in (_HERE / "kernels").glob("*.py")]
        sources.append(_HERE / "runtime_model.py")
        source_hashes = {str(p.relative_to(_HERE)): digest(p) for p in sources}
    plugins = _policy(plugins)
    identity = {"distribution": distribution, "plugins": plugins, "source_hashes": source_hashes}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    folder = artifact(Path(artifact_dir or ROOT / "optimized-runtime") / fingerprint)
    folder.mkdir(parents=True, exist_ok=True)

    # Import every plugin before changing dispatch, so module-level ORIGINAL
    # references bind to the same baseline rather than another installed policy.
    modules = [importlib.import_module(p["module"]) for p in plugins]
    for plugin, module in zip(plugins, modules):
        if plugin["name"] not in module.variants():
            raise ValueError(f"Unsupported candidate {plugin}")
    snapshot = _snapshot()
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError("This measured kernel policy targets CUDA compute capability 12.0")
        for index, (plugin, module) in enumerate(zip(plugins, modules)):
            module.install(plugin["name"], folder / f"{index:02d}-{plugin['name']}")
        metadata = {**identity, "policy_fingerprint": fingerprint,
                    "artifact_dir": str(folder), "confirmed": record is not None,
                    "receipt": str(receipt_path) if receipt_path else None,
                    "gpu": torch.cuda.get_device_name(),
                    "scope": "Architecture-matched packed inference; no accuracy claim"}
        active = Installation(metadata, snapshot)
        _ACTIVE = active
        return active
    except BaseException:
        _restore(snapshot)
        raise
