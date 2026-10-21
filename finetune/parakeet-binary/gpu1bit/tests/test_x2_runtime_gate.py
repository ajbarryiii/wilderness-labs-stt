"""Z6 recognition contract of the x2 runtime against the gate run (plans/x2-default.md Z6), in one process
holding one lock. Runs once the gate run has passed, on the unchanged frozen sources:

    GPU1BIT_X2_GATE_RUN=<gate run> timeout 600 gpu1bit/shared/python -m gpu1bit.tests.run \\
        --label H:x2-runtime-gate gpu1bit.tests.test_x2_runtime_gate

- The reference is the provider-x2 arm g2 (MB1-int8_mma-x2) of the gate run: text, token IDs and
  timestamps of the 24 timing workloads (dev_clips + dev_24), identity-bound: g2's summary must be
  complete and carry the identity of the current sources under shared/selected-x2.json
  (actprec_report.verified_summary), and the gate record of that run must have passed.
- The runtime is gpu1bit.runtime.load_binary_packed(MB1, selection=shared/selected-x2.json) (x2 by default),
  decoded through its API (transcribe, decoded). Any difference fails.
- Equality with x3 (g3, MB1-B-hybrid-x3 of the same run) is reported as a hypothesis, never asserted.
Without GPU1BIT_X2_GATE_RUN the module fails (it is never skipped silently).
"""
from __future__ import annotations

import json
import os
import unittest

import torch

from gpu1bit.shared import env

RUN = os.environ.get("GPU1BIT_X2_GATE_RUN")


def gate_rows(run: str, gid: str, ids: set[str]) -> dict[str, dict]:
    """{utterance id: {text, tokens, timestamps}} of gate arm `gid`, identity-verified."""
    import gpu1bit.actprec_report as AR
    from gpu1bit.shared import fulldev
    from gpu1bit.x2 import gates
    counts = {n: len(fulldev.subset_records(n, "full")) for n in fulldev.DEV_SETS}
    _, identity = AR.verified_summary(run, gates.ARM[gid], counts)
    out = {}
    for n in fulldev.DEV_SETS:
        for r in fulldev.load_rows(fulldev.work_path(run, gates.ARM[gid], n)):
            if r["id"] in ids:
                if r["identity"] != identity:
                    raise ValueError(f"{gid} row {r['id']} is of identity {r['identity'][:12]}, not {identity[:12]}")
                out[r["id"]] = {"text": r["hyp"], "tokens": r["tokens"], "timestamps": r["timestamps"]}
    return out


class RecognitionAgainstG2(unittest.TestCase):
    def test_runtime_x2_equals_g2_on_timing_workloads(self):
        if not RUN:
            self.fail("set GPU1BIT_X2_GATE_RUN to the gate run (this contract needs its g2 rows)")
        from nemo.utils import logging
        from gpu1bit import runtime
        from gpu1bit.actprec import parts
        from gpu1bit.shared import seldoc, workloads
        from gpu1bit.x2 import runtime_contracts as RC
        logging.setLevel(logging.ERROR)
        os.environ["GPU1BIT_SELECTION"] = str(seldoc.CANDIDATE)
        parts.install()
        gate = json.loads((env.artifacts("x2") / f"gates-{RUN}.json").read_text())
        self.assertTrue(gate["pass"], "the gate run did not pass")
        self.assertEqual(gate["selection"]["sha256"], env.sha256_file(seldoc.CANDIDATE))
        doc = workloads.load()
        timing = list({r["id"]: r for r in doc["dev_clips"] + doc["dev_24"]}.values())
        ids = {r["id"] for r in timing}
        ref, x3 = gate_rows(RUN, "g2", ids), gate_rows(RUN, "g3", ids)
        model = runtime.load_binary_packed(env.MB1_EXPORT, selection=seldoc.CANDIDATE)
        try:
            rt = runtime.runtime_of(model)
            self.assertEqual(rt.parts, 2)
            self.assertEqual(rt.identity, parts.identity("MB1-B-hybrid-x2", "full", selection=seldoc.CANDIDATE)["hash"])
            rows = {}
            for rec in timing:
                model.forward.entries.clear()
                text = runtime.transcribe(model, rec["audio_filepath"])
                rows[rec["id"]] = {"text": text, **runtime.decoded(model)}
        finally:
            runtime.close_binary_packed(model)
            torch.cuda.empty_cache()
        out = RC.compare_recognition(rows, ref, x3_rows=x3)
        record = {"gate_run": RUN, "runtime_identity": rt.identity, "identity": rt.identity, "parts": rt.parts, **out,
                  "rows": {u: {"runtime": rows[u], "g2": ref[u]} for u in sorted(ref)}}
        (env.artifacts("x2") / f"runtime-recognition-{RUN}.json").write_text(json.dumps(record, indent=1) + "\n")
        self.assertTrue(out["ok"], f"text/tokens/timestamps differ from g2 on {out['differences']}")


if __name__ == "__main__":
    unittest.main()
