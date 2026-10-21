"""Contract tests of gpu1bit/actprec/bench.py, the timing-harness patch for the activation-precision
arms (CPU):
  - the six arm names (M1-x3/x2/x1, MB1-int8_mma-x3/x2/x1) are activation-precision variants. Every
    other name (baselines, registry candidates, B-hybrid, fp32 arms, other kernels' xN) goes to the
    previously installed variant_model, unchanged.
  - an arm variant is built only if the arm's identity equals the one frozen for the recognition run,
    and (review c2) its x3 reference in that run passes the R2 gate (complete, verified, R5-valid,
    cross-check passed). Otherwise it is refused before any model is built. The identity hash, the
    run and the reference's identity are in the variant's info.
  - (review c3) the recognition run is explicit: GPU1BIT_ACTPREC_RUN must be set (no default) and is
    recorded as identity_run.
  - (GPU child, review c2) a real P-clips / P-memory artifact carries that info.

    gpu1bit/shared/python -m gpu1bit.tests.run --cpu gpu1bit.tests.test_actprec_bench
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import sys
import unittest

from gpu1bit.shared import bench, env, fulldev

sys.path.insert(0, str(env.GPU1BIT.parent))

ARMS = ("M1-x3", "M1-x2", "M1-x1", "MB1-int8_mma-x3", "MB1-int8_mma-x2", "MB1-int8_mma-x1")


class _Built(Exception):
    pass


class Patch(unittest.TestCase):
    def setUp(self):
        self.original = bench.variant_model
        import gpu1bit.actprec.bench as AB
        self.AB = AB
        self.delegated = []

        @contextlib.contextmanager
        def previous(name, *, max_graphs=4):
            self.delegated.append((name, max_graphs))
            yield "model", {"kind": "previous"}
        AB._previous = previous
        self.run = f"test-actprec-bench-{os.getpid()}"
        self.saved = (AB.parts.build, AB.parts.close, os.environ.get("GPU1BIT_ACTPREC_RUN"))
        os.environ["GPU1BIT_ACTPREC_RUN"] = self.run

    def tearDown(self):
        self.AB.parts.build, self.AB.parts.close, inherited = self.saved
        if inherited is None:
            os.environ.pop("GPU1BIT_ACTPREC_RUN", None)
        else:
            os.environ["GPU1BIT_ACTPREC_RUN"] = inherited
        bench.variant_model = self.original
        self.AB.parts.uninstall()
        shutil.rmtree(fulldev.ROOT / self.run, ignore_errors=True)

    def test_names(self):
        for name in ARMS:
            self.assertTrue(self.AB.is_arm(name), name)
        for name in ("R-expanded", "R-hybrid", "int8_mma", "int8_mma:b01", "B-hybrid", "MB1-fp32", "M1-fp32",
                     "MB1-int8_mma:b01-x2", "MB1-bitserial-x1", "triton_bf16:bf16x3"):
            self.assertFalse(self.AB.is_arm(name), name)

    def test_other_names_delegate_unchanged(self):
        with self.AB.variant_model("R-expanded", max_graphs=0) as pair:
            self.assertEqual(pair, ("model", {"kind": "previous"}))
        self.assertEqual(self.delegated, [("R-expanded", 0)])

    def write_reference(self, model="MB1", **overrides):
        ref = "M1-x3" if model == "M1" else "MB1-int8_mma-x3"
        h = fulldev.freeze(self.run, ref, "full")
        summary = {"run": self.run, "arm": ref, "subset": "full", "identity": h, "complete": True,
                   "utterances": {n: len(fulldev.subset_records(n, "full")) for n in fulldev.DEV_SETS},
                   "structurally_valid": True, "cross_check": {"pass": True},
                   "spec": fulldev.parse_arm(ref)}                       # as fulldev.finish writes it
        summary.update(overrides)
        (fulldev.arm_dir(self.run, ref, "full") / "summary.json").write_text(json.dumps(summary))
        return h

    def test_the_recognition_run_is_explicit(self):
        """Review c3: no default run; the run comes from GPU1BIT_ACTPREC_RUN (set by the orchestrator from
        the roster's identity_run) and is recorded with the reference in the variant's info."""
        self.AB.parts.install()          # identities are frozen with the patch installed, as in a real run
        h = self.write_reference()
        fulldev.freeze(self.run, "MB1-int8_mma-x1", "full")
        del os.environ["GPU1BIT_ACTPREC_RUN"]
        with self.assertRaisesRegex(SystemExit, "GPU1BIT_ACTPREC_RUN"):
            with self.AB.variant_model("MB1-int8_mma-x1"):
                pass
        with self.AB.variant_model("R-expanded") as pair:          # other variants do not need it
            self.assertEqual(pair[1], {"kind": "previous"})
        os.environ["GPU1BIT_ACTPREC_RUN"] = self.run
        self.AB.parts.build = lambda model, n, probe=None, kernel="int8_mma": ("model", {"partcount": None, "parts": n, "stack": {}})
        self.AB.parts.close = lambda model: None
        import evaluate
        saved = evaluate.inference_settings
        evaluate.inference_settings = lambda model: contextlib.nullcontext()   # the stub model has no settings
        try:
            with self.AB.variant_model("MB1-int8_mma-x1") as (model, info):
                self.assertEqual(info["identity_run"], self.run)
                self.assertEqual(info["reference"], {"arm": "MB1-int8_mma-x3", "identity": h})
                self.assertEqual(info["identity"], fulldev.check_identity(self.run, "MB1-int8_mma-x1", "full")["hash"])
        finally:
            evaluate.inference_settings = saved

    def test_identity_and_reference_binding(self):
        def build(model, n, probe=None, kernel="int8_mma"):
            raise _Built((model, n))
        self.AB.parts.build = build
        self.AB.parts.install()
        with self.assertRaisesRegex(SystemExit, "no frozen identity"):
            with self.AB.variant_model("MB1-int8_mma-x2"):
                pass
        fulldev.freeze(self.run, "MB1-int8_mma-x2", "full")
        with self.assertRaisesRegex(SystemExit, "R2"):                       # review c2: no x3 reference yet
            with self.AB.variant_model("MB1-int8_mma-x2"):
                pass
        self.write_reference(cross_check={"pass": False})
        for name in ("MB1-int8_mma-x2", "MB1-int8_mma-x3"):
            with self.subTest(name=name), self.assertRaisesRegex(SystemExit, "R2"):
                with self.AB.variant_model(name):
                    pass
        self.write_reference()
        with self.assertRaises(_Built) as built:
            with self.AB.variant_model("MB1-int8_mma-x2"):
                pass
        self.assertEqual(built.exception.args[0], ("MB1", 2))
        with self.assertRaises(_Built) as built:
            with self.AB.variant_model("MB1-int8_mma-x3"):
                pass
        self.assertEqual(built.exception.args[0], ("MB1", 3))
        path = fulldev.arm_dir(self.run, "MB1-int8_mma-x2", "full") / "identity.json"
        doc = json.loads(path.read_text())
        doc["sources"]["finetune/parakeet-binary/gpu1bit/actprec/bench.py"] = "0" * 64
        doc["hash"] = "f" * 64
        path.write_text(json.dumps(doc))
        with self.assertRaisesRegex(SystemExit, "identity mismatch"):
            with self.AB.variant_model("MB1-int8_mma-x2"):
                pass
        self.assertEqual(self.delegated, [])


@unittest.skipUnless(os.environ.get("GPU1BIT_SPAWN_GPU_CHILDREN") == "1",
                     "spawns a GPU child; run inside an agent-jobs --needs gpu run with GPU1BIT_SPAWN_GPU_CHILDREN=1")
class ClipsArtifact(unittest.TestCase):
    """Review c2: the P-clips / P-memory artifact of an activation-precision variant carries its
    recognition identity, the identity run and its reference's identity. A real `bench clips`
    child (shared/bench.py through dispatch with the patch, one lock) writes it, and the test
    reads the emitted file."""

    def test_emitted_artifact_carries_the_identity(self):
        import subprocess
        from gpu1bit.actprec import parts
        run = f"test-actprec-clips-{os.getpid()}"
        out_dir = env.ARTIFACTS_ROOT / "H" / "bench" / run
        try:
            with parts.installed():
                hashes = {a: fulldev.freeze(run, a, "full") for a in ("M1-x3", "M1-x2")}
            summary = {"run": run, "arm": "M1-x3", "subset": "full", "identity": hashes["M1-x3"], "complete": True,
                       "utterances": {n: len(fulldev.subset_records(n, "full")) for n in fulldev.DEV_SETS},
                       "structurally_valid": True, "cross_check": {"pass": True}, "spec": fulldev.parse_arm("M1-x3")}
            (fulldev.arm_dir(run, "M1-x3", "full") / "summary.json").write_text(json.dumps(summary))
            e = dict(os.environ, GPU1BIT_ACTPREC_RUN=run)
            e.pop("CUBLAS_WORKSPACE_CONFIG", None)
            e.pop("CUDA_VISIBLE_DEVICES", None)
            cmd = ["timeout", "600", str(env.SHARED / "python"), "-m", "gpu1bit.shared.dispatch", "--patches",
                   "gpu1bit.actprec.bench", "bench", "clips", "--agent", "H", "--run", run, "--variant", "M1-x2",
                   "--seconds", "0.5", "--repeats", "1"]
            proc = subprocess.run(cmd, cwd=str(env.BINARY_DIR), env=e, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stdout[-2000:] + proc.stderr[-3000:])
            doc = json.loads((out_dir / "clips-M1-x2.json").read_text())
            info = doc["variant_info"]
            self.assertEqual(doc["variant"], "M1-x2")
            self.assertEqual((info["arm"], info["identity"], info["identity_run"]), ("M1-x2", hashes["M1-x2"], run))
            self.assertEqual(info["reference"], {"arm": "M1-x3", "identity": hashes["M1-x3"]})
            self.assertEqual(info["identity_run"], run)
            self.assertEqual(info["parts"], 2)
            self.assertEqual(len(doc["workloads"]), 3)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)
            shutil.rmtree(fulldev.ROOT / run, ignore_errors=True)
