"""Pipeline contract of the fixed B-hybrid (review c1 finding 1), dev audio only.

On MB1 under R-hybrid's context stack, HybridProjectionExact with int64 deferred slices gives an
encoder output BITWISE equal to the same stack with every deferred call finished in-kernel
(split_deferred=False: exact b1_dot + M1's _row_epilogue with SPLIT = 1), for a 3 s, a 10 s and a
30 s dev_24 utterance (M around 38, 126, 368), for int8_mma and int8_mma:b01; and the slices were
really used wherever the tile table selects them (provider.calls["exact_slices"]). The deployed graph path (model forward through the
pipeline graph cache) gives the same text, token IDs and timestamps as the in-kernel variant.

    timeout 600 gpu1bit/shared/python -m gpu1bit.int8_mma.gpulock gpu1bit.int8_mma.test_exact_pipeline
"""
from __future__ import annotations

import contextlib
import json
import unittest

import torch

from gpu1bit.int8_mma import hybrid_exact
from gpu1bit.int8_mma.layer import Int8MmaLayer
from gpu1bit.int8_mma.layer_b01 import Int8MmaB01Layer
from gpu1bit.shared import env

_LOCK = None


def setUpModule():
    global _LOCK
    if not getattr(env.gpu_lock, "_held", False):
        _LOCK = env.gpu_lock("B:exact deferred pipeline tests")
        _LOCK.__enter__()


def tearDownModule():
    if _LOCK is not None:
        _LOCK.__exit__(None, None, None)


def picks():
    from gpu1bit.shared import workloads
    records = workloads.load()["dev_24"]
    out = {}
    for rec in records:
        out.setdefault(rec["target_seconds"], rec)
    return [out[t] for t in sorted(out)]


class ExactPipeline(unittest.TestCase):
    def test_encoder_bitwise_equal_to_in_kernel_finish(self):
        import evaluate as ev
        from gpu1bit.shared import bench
        from nemo.utils import logging
        logging.setLevel(logging.ERROR)
        records = picks()
        self.assertGreaterEqual(len(records), 3)
        total_used = {}
        for cls in (Int8MmaLayer, Int8MmaB01Layer):
            with torch.inference_mode(), contextlib.ExitStack() as stack:
                stack.enter_context(ev.strict_fp32("cuda"))
                split, info_split = stack.enter_context(hybrid_exact.exact_model(cls, split_deferred=True))
                whole, info_whole = stack.enter_context(hybrid_exact.exact_model(cls, split_deferred=False))
                for rec in records:
                    audio, length = ev.audio_batch([rec])
                    feats, flen = split.preprocessor(input_signal=audio.cuda(), length=length.cuda())
                    before = dict(info_split["projection"].calls)
                    e1, l1 = split.encoder(audio_signal=feats, length=flen)
                    e0, l0 = whole.encoder(audio_signal=feats, length=flen)
                    torch.cuda.synchronize()
                    used = info_split["projection"].calls.get("exact_slices", 0) - before.get("exact_slices", 0)
                    what = f"{cls.__name__} {rec['id']} M={e1.shape[-1]}"
                    m = int(e1.shape[-1])
                    expected = any(hybrid_exact.choose_exact_deferred(m, 1024, k) is not None for k in (1024, 4096))
                    self.assertEqual(used > 0, expected, f"{what}: exact slices used {used}, table expects {expected}")
                    total_used[cls.__name__] = total_used.get(cls.__name__, 0) + used
                    self.assertEqual(info_whole["projection"].calls.get("exact_slices", 0), 0)
                    self.assertTrue(torch.equal(l1, l0), what)
                    self.assertTrue(torch.equal(e1, e0), f"{what}: max |d| {(e1 - e0).abs().max().item():.3e}")
                    t1 = (bench.recognize(split, audio, length), bench.decoded_state(split))
                    t0 = (bench.recognize(whole, audio, length), bench.decoded_state(whole))
                    self.assertEqual(t1, t0, f"{what}: text / tokens / timestamps")
                    print(json.dumps({"layer": cls.__name__, "id": rec["id"], "M": m,
                                      "exact_slice_calls": used, "bitwise": True}), flush=True)
            self.assertGreater(total_used.get(cls.__name__, 0), 0, "the exact slices were never exercised")


if __name__ == "__main__":
    unittest.main()
