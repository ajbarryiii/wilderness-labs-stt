"""Short serialized GPU smoke of the confirmed optional kernel installer.

Uses real medium decoder widths, complete cross-attention memory length, short
and full growing-cache positions, and the vocabulary GEMV shape. This verifies
installation, numerical behavior, graph replay, and restoration; it does not
replace the independently measured full-model energy comparisons.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib
import json
from pathlib import Path
import time

import torch

from energy import EnergyMeter
from kernels import optimized_runtime as optimized
from kernels.packed import PackedWeight
from paths import ROOT, artifact, save, storage
from runtime_model import _Block, _Factory

LOCK = Path("/mnt/hd/wilderness-labs-stt/stt-distillation/active.lock")


def _unchanged(snapshot):
    if any(getattr(owner, name) is not value for owner, name, value in snapshot):
        raise AssertionError("Optional installer did not restore all dispatch points")


def _cache_copy(cache):
    return tuple(c.clone() for c in cache)


def _linear_outputs(cases):
    return [p.linear(x, bias).clone() for p, x, bias in cases]


def _block_outputs(block, x, memory, seed_cache):
    results = []
    for step in (0, 64, 128):
        cache = _cache_copy(seed_cache)
        y = block(x, memory_kv=memory, cache=cache, step=step).clone()
        for changed, seed in zip(cache, seed_cache):
            if not torch.equal(changed[:, :, :step], seed[:, :, :step]):
                raise AssertionError("Direct-cache projection modified a previous token")
            if not torch.equal(changed[:, :, step+1:], seed[:, :, step+1:]):
                raise AssertionError("Direct-cache projection modified a future token")
        results.append((y, cache))
    return results


def _close(actual, expected, *, exact=False):
    torch.testing.assert_close(actual, expected,
                               atol=0 if exact else .003, rtol=0 if exact else .01)
    return float((actual.float() - expected.float()).abs().max())


def verify(receipt=None, output=None, *, wait_for_gpu=False):
    storage()
    receipt_path = receipt or optimized.CONFIRMED_RECEIPT
    if receipt_path is None:
        raise RuntimeError("Set the independently confirmed receipt before GPU verification")
    receipt_path = artifact(receipt_path)
    record = json.loads(receipt_path.read_text())
    output = artifact(output or ROOT / "optimized-runtime/installer-smoke.json")
    folder = artifact(output.parent / "installer-smoke-cache")
    started = time.monotonic()
    report = {"receipt": str(receipt_path), "distributions": {},
              "scope": "Numerical and installation smoke; no energy score"}
    with LOCK.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | (0 if wait_for_gpu else fcntl.LOCK_NB))
        report["gpu_lock_wait_seconds"] = time.monotonic() - started
        with EnergyMeter() as meter:
            if meter.foreign_processes():
                raise RuntimeError("Foreign compute process prevents isolated GPU verification")
        torch.set_num_threads(4)
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.manual_seed(51473)
        with torch.inference_mode():
            for distribution in ("binary", "ternary"):
                snapshot = optimized._snapshot()
                factory = _Factory(distribution, "packed", 7871, "cuda", torch.float16)
                block = _Block(factory, 1024, 16, decoder=True)
                pairs = ((block.cross_attn.q, block.cross_attn.q_bias),
                         (block.attn.qkv, block.attn.qkv_bias),
                         (block.mlp_up, block.mlp_up_bias),
                         (block.mlp_down, block.mlp_down_bias))
                cases = [(w.packed, torch.randn((1, w.k), device="cuda", dtype=torch.float16), b)
                         for w, b in pairs]
                # Include the real vocabulary extent without CPU packing of
                # another 53M random values. Patterns use only valid codes.
                bits = 1 if distribution == "binary" else 2
                words = torch.full((51864, 1024//(32//bits)),
                                   0x5a5a5a5a if bits == 1 else 0x24924924,
                                   device="cuda", dtype=torch.int32)
                vocab = PackedWeight(words, torch.linspace(.002, .03, 51864, device="cuda"),
                                     51864, 1024, bits, words.t().contiguous())
                cases.append((vocab, torch.randn((1, 1024), device="cuda", dtype=torch.float16), None))
                offset = torch.randn(1025, device="cuda", dtype=torch.float16)[1:].view(1, 1024)
                cases.append((block.cross_attn.q.packed, offset, block.cross_attn.q_bias))
                x = torch.randn((1, 1, 1024), device="cuda", dtype=torch.float16)
                memory = tuple(torch.randn((1, 16, 1500, 64), device="cuda", dtype=torch.float16)
                               for _ in range(2))
                seed_cache = tuple(torch.randn((1, 16, 129, 64), device="cuda", dtype=torch.float16)
                                   for _ in range(2))
                baseline_linears = _linear_outputs(cases)
                baseline_blocks = _block_outputs(block, x, memory, seed_cache)
                with optimized.install(distribution, folder, receipt=receipt_path) as installed:
                    installed_linears = _linear_outputs(cases)
                    installed_blocks = _block_outputs(block, x, memory, seed_cache)
                    worst = max(_close(a, b) for a, b in zip(installed_linears, baseline_linears))
                    for (actual, cache), (expected, reference_cache) in zip(installed_blocks, baseline_blocks):
                        worst = max(worst, _close(actual, expected))
                        for a, b in zip(cache, reference_cache):
                            worst = max(worst, _close(a, b))
                    graph_x = x.clone()
                    graph_cache = _cache_copy(seed_cache)
                    for _ in range(3):
                        block(graph_x, memory_kv=memory, cache=graph_cache, step=64)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        graph_y = block(graph_x, memory_kv=memory, cache=graph_cache, step=64)
                    graph.replay()
                    _close(graph_y, installed_blocks[1][0], exact=True)
                    old_y = graph_y.clone()
                    graph_x.mul_(.31)
                    changed_cache = _cache_copy(seed_cache)
                    changed_expected = block(graph_x, memory_kv=memory, cache=changed_cache, step=64).clone()
                    graph.replay()
                    _close(graph_y, changed_expected, exact=True)
                    if torch.equal(graph_y, old_y):
                        raise AssertionError("Changed-input graph replay did not change output")
                    for a, b in zip(graph_cache, changed_cache):
                        _close(a, b, exact=True)
                    policy_dir = Path(installed.metadata["artifact_dir"])
                    metadata = dict(installed.metadata)
                    # Destroy captured work before restoring Python dispatch.
                    del graph
                _unchanged(snapshot)
                restored = _linear_outputs(cases)
                for a, b in zip(restored, baseline_linears):
                    _close(a, b, exact=True)

                # Reproduce the exact plugin recipe using the original measured
                # installation API, then compare bitwise with the optional API.
                plugins = record["policies"][distribution]["plugins"]
                modules = [importlib.import_module(p["module"]) for p in plugins]
                try:
                    for index, (plugin, module) in enumerate(zip(plugins, modules)):
                        module.install(plugin["name"], policy_dir / f"{index:02d}-{plugin['name']}")
                    direct_linears = _linear_outputs(cases)
                    direct_blocks = _block_outputs(block, x, memory, seed_cache)
                    for a, b in zip(direct_linears, installed_linears):
                        _close(a, b, exact=True)
                    for (a, ac), (b, bc) in zip(direct_blocks, installed_blocks):
                        _close(a, b, exact=True)
                        for aa, bb in zip(ac, bc):
                            _close(aa, bb, exact=True)
                finally:
                    optimized._restore(snapshot)
                _unchanged(snapshot)
                report["distributions"][distribution] = {
                    "passed": True, "medium_gemv_cases": len(cases),
                    "decoder_cache_steps": [0, 64, 128],
                    "maximum_baseline_absolute_error": worst,
                    "direct_policy_comparison": "bitwise exact",
                    "changed_input_graph": "bitwise exact to eager installed policy",
                    "cache_canaries": "passed", "dispatch_restoration": "passed",
                    "installation": metadata}
                print(json.dumps({"distribution": distribution, "passed": True,
                                  "max_baseline_abs_error": worst}), flush=True)
                del block, factory, cases, pairs, vocab, words, x, memory, seed_cache
                del baseline_linears, baseline_blocks, installed_linears, installed_blocks
                del direct_linears, direct_blocks, restored, graph_x, graph_y, graph_cache
                torch.cuda.synchronize()
    report["passed"] = True
    report["elapsed_seconds"] = time.monotonic() - started
    save(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt")
    parser.add_argument("--output")
    parser.add_argument("--wait-for-gpu", action="store_true",
                        help="Wait for the shared GPU lock before integration checks")
    args = parser.parse_args()
    result = verify(args.receipt, args.output, wait_for_gpu=args.wait_for_gpu)
    print(json.dumps({"passed": result["passed"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
