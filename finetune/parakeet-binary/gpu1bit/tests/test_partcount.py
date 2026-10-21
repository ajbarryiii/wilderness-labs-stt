"""Contract tests of gpu1bit/actprec/parts.py, the activation-precision part-count arms
(plans/activation-precision.md R1, R4, R6; coordinator 2026-10-20 15:00: B's x2/x1 files load from
B/wip/parts through a hash-pinned patch, so the sel-20261020b freeze stays intact).

CPU (`gpu1bit/shared/python -m gpu1bit.tests.run --cpu gpu1bit.tests.test_partcount`):
  Adapter: fulldev.PartCount passes every argument through unchanged except components, for
    positional and keyword call forms. It records (requested, effective) pairs, delegates
    attributes, and refuses n outside 1..3.
  Identity (R4):
    - The identities of all six activation-precision arms (x3, x2, x1 of M1 and MB1) hash every file
      of gpu1bit/actprec/ (the arm builder, and bench.py for the timed variants) and the timing
      orchestrator gpu1bit/rerun_timing.py.
    - The xN identities also pin every file of B's parts manifest at its 14:55 hash, and carry the
      part count, the layer variant and the adapter.
    - The fp32 identities are exactly the base fulldev identities.
    - CONTRACT CHANGE (review c1, 2026-10-20): the x3 arms are now built by parts.build(n=3), like the
      timed variants, and their identities hash the patch. Before, they used fulldev's own build and
      identity. The change makes recognition rows and timing records come from the same builder under
      one identity.
    - A pinned file whose bytes differ is refused by the identity and by the loader, before any code
      runs.
    - A new file in gpu1bit/actprec/ changes the identities, and check_identity refuses it.
    - xN arms are enabled only while the patch is installed, and only for M1 and kernel int8_mma.
      Installed, fulldev.build_arm routes x3 and xN arms to parts.build and fp32 arms to fulldev.
  R2 gate (review c1 major 3):
    - An xN arm is launched, compared or classified only against an x3 reference of the same run (its
      summary must name that run; review c3) that is complete, has the full subset and the right arm name, carries the identity check_identity
      verifies, is structurally valid and passed its cross-check.
    - A missing, incomplete, foreign-identity, invalid or failed reference stops the xN driver before
      any worker starts, and a direct xN worker (`parts run`) before it records anything (review c2;
      the timed variants and the encoder checks are gated too, tested in test_actprec_bench and
      test_actprec_report).
    - An x3 driver whose finished arm fails the cross-check or R5 exits non-zero (R2: no xN runs until
      the cause is known).
GPU (one lock per run; one class per model):
  Paths (R1). Per model (M1, MB1) and n in (2, 1), compared with the same model's deployed x3 stack
  on the same five shapes: eager warm-up, capture, replay, cache overflow (5th shape) and direct
  eager (the cache's model_forward).
    - Every projection call passes the adapter and reaches the wrapped provider with components == n
      (MB1: every call reaches the variant's project, none is composed from linear / conv).
    - Per path, the call sequence (call site, N, K, conv, activation, norm, input activation, input
      shape and strides, capturing flag, config) equals x3's.
    - Replay makes no projection call. The replayed output equals the direct eager output bitwise, and
      differs from x3's output for the same input, so the replayed graph runs n parts.
    - There are 4 graph entries after 4 shapes and still 4 after the overflow shape. Outputs are
      finite and their lengths equal x3's.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from gpu1bit.shared import env, fulldev

# B's parts manifest (B/wip/parts-sha256.txt, NOTES "B 2026-10-20 14:55"): the bytes the arms must run.
PINNED = {
    "kernels_parts.py": "6e8cb12deeedd76cc9a627c1fb361a4e9372c0f6f0289fd30890448b51d271b7",
    "layer_parts.py": "4dc1a2e78d81885c533c42de83af10a4db0d65a55bdf884e598f17b601739786",
    "parts_run.py": "8e308ce5117c9d2e8e06592712144d02e0f47346e343fc2ff580134da42d2d29",
    "test_parts.py": "a435fed64ac6d8e1904c881f6fa689b052b62cc5e14d2bd336acad0c9c076e9f",
    "sanitize_parts.py": "627563dec1ca49479f0b397816b30a852fa8698516d7b8482593504dea77435b",
}
XN_ARMS = ("M1-x2", "M1-x1", "MB1-int8_mma-x2", "MB1-int8_mma-x1")
SIGNATURE_KEYS = ("phase", "capturing", "site", "N", "K", "conv", "activation", "norm", "input_activation",
                  "x_shape", "x_stride", "config")


class _Fake:
    def __init__(self):
        self.calls, self.bytes = [], 123

    def __call__(self, x, mod, conv=False, components=3, activation="", norm=None, config=None, input_activation=""):
        self.calls.append((x, mod, conv, components, activation, norm, config, input_activation))
        return "y"


class Adapter(unittest.TestCase):
    def test_overrides_only_components(self):
        fake = _Fake()
        pc = fulldev.PartCount(fake, 2)
        pc.record = []
        x, mod, norm = object(), object(), object()
        self.assertEqual(pc(x, mod, True, 3, "silu", norm, (1, 2), "silu"), "y")
        pc(x, mod, conv=False, components=3, activation="", norm=None, config=None, input_activation="silu")
        pc(x, mod)
        self.assertEqual(fake.calls[0], (x, mod, True, 2, "silu", norm, (1, 2), "silu"))
        self.assertEqual(fake.calls[1], (x, mod, False, 2, "", None, None, "silu"))
        self.assertEqual(fake.calls[2], (x, mod, False, 2, "", None, None, ""))
        self.assertEqual(pc.record, [(3, 2), (3, 2), (3, 2)])
        self.assertEqual(pc.bytes, 123)
        for bad in (0, 4):
            with self.subTest(n=bad), self.assertRaises(ValueError):
                fulldev.PartCount(fake, bad)


class Identity(unittest.TestCase):
    def setUp(self):
        from gpu1bit.actprec import parts
        self.parts = parts

    def test_pins_are_bs_manifest(self):
        self.assertEqual({k: v for k, v in self.parts.PINNED.items() if k in PINNED}, PINNED)
        manifest = {}
        for line in (self.parts.WIP_DIR.parent / "parts-sha256.txt").read_text().splitlines():
            digest, name = line.split()
            manifest[name] = digest
        self.assertEqual(self.parts.PINNED, manifest, "every file of B's parts manifest is pinned")

    def test_xn_identity_pins_parts_and_patch(self):
        patch_files = {str(p.relative_to(env.REPO)) for p in (env.GPU1BIT / "actprec").rglob("*")
                       if p.is_file() and "__pycache__" not in p.parts}
        self.assertIn("finetune/parakeet-binary/gpu1bit/actprec/parts.py", patch_files)
        self.assertIn("finetune/parakeet-binary/gpu1bit/actprec/bench.py", patch_files)
        patch_files.add("finetune/parakeet-binary/gpu1bit/rerun_timing.py")
        with self.parts.installed():
            for arm in ("M1-x3", "MB1-int8_mma-x3"):
                doc = fulldev.identity(arm, "full")
                self.assertTrue(patch_files <= set(doc["sources"]), f"{arm}: the patch and orchestrator are hashed")
                self.assertNotIn("part_count_patch", doc)
                self.assertEqual(doc["actprec_patch"]["builder"], "gpu1bit.actprec.parts.build(model, 3): the deployed x3 stack")
            for arm in XN_ARMS:
                with self.subTest(arm=arm):
                    doc = fulldev.identity(arm, "full")
                    spec = fulldev.parse_arm(arm)
                    patch = doc["part_count_patch"]
                    self.assertEqual(doc["parts"], spec["parts"])
                    self.assertEqual(patch["parts"], spec["parts"])
                    self.assertEqual({k: v["sha256"] for k, v in patch["wip_parts"].items()}, self.parts.PINNED)
                    for v in patch["wip_parts"].values():
                        self.assertEqual(env.sha256_file(Path(v["path"])), v["sha256"])
                    self.assertTrue(patch_files <= set(doc["sources"]), "every gpu1bit/actprec file is hashed")
                    self.assertIn("PartCount", patch["adapter"])
                    variant = patch["layer_variant"]
                    self.assertEqual(variant, f"int8_mma:x{spec['parts']}" if spec["model"] == "MB1" else None)
                    recomputed = hashlib.sha256(json.dumps({k: v for k, v in doc.items() if k != "hash"},
                                                           sort_keys=True).encode()).hexdigest()
                    self.assertEqual(doc["hash"], recomputed)
            hashes = {fulldev.identity(a, "full")["hash"] for a in XN_ARMS}
            self.assertEqual(len(hashes), len(XN_ARMS))

    def test_fp32_identities_are_untouched_and_x3_extends_the_base(self):
        base = {a: fulldev.identity(a, "full") for a in ("M1-x3", "MB1-int8_mma-x3", "MB1-fp32", "M1-fp32")}
        with self.parts.installed():
            for arm in ("MB1-fp32", "M1-fp32"):
                self.assertEqual(fulldev.identity(arm, "full")["hash"], base[arm]["hash"], arm)
            for arm in ("M1-x3", "MB1-int8_mma-x3"):
                doc = fulldev.identity(arm, "full")
                self.assertNotEqual(doc["hash"], base[arm]["hash"])
                for key in ("export", "manifests", "scorer", "versions", "kernel", "config_tables", "parts"):
                    self.assertEqual(doc[key], base[arm][key], f"{arm}: {key}")
                self.assertTrue(set(base[arm]["sources"].items()) <= set(doc["sources"].items()))
        self.assertIs(fulldev.identity, self.parts.BASE["identity"], "uninstall restores fulldev")

    def test_build_routing(self):
        calls = []
        saved = self.parts.build

        def stub(model, n, *, probe=None):
            calls.append((model, n))
            return "model", {"partcount": None, "parts": n}
        self.parts.build = stub
        try:
            with self.parts.installed():
                for arm, want in (("M1-x3", ("M1", 3)), ("MB1-int8_mma-x3", ("MB1", 3)), ("M1-x1", ("M1", 1)),
                                  ("MB1-int8_mma-x2", ("MB1", 2))):
                    model, info = fulldev.build_arm(fulldev.parse_arm(arm))
                    self.assertEqual(calls[-1], want)
                    self.assertNotIn("partcount", info)
                self.assertIs(self.parts.BASE["build_arm"], self.parts.base_build_for("M1-fp32"))
        finally:
            self.parts.build = saved

    def test_changed_pinned_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "parts"
            shutil.copytree(self.parts.WIP_DIR, copy)
            with self.parts.installed(wip_dir=copy):
                fulldev.identity("MB1-int8_mma-x2", "full")      # identical copy: accepted
                self.parts.verified_source("layer_parts")
            (copy / "layer_parts.py").write_bytes((copy / "layer_parts.py").read_bytes() + b"\n# changed\n")
            with self.parts.installed(wip_dir=copy):
                with self.assertRaisesRegex(ValueError, "pinned"):
                    fulldev.identity("MB1-int8_mma-x2", "full")
                with self.assertRaisesRegex(ValueError, "pinned"):
                    self.parts.verified_source("layer_parts")

    def test_new_patch_file_changes_identity_and_is_refused(self):
        run = f"test-partcount-identity-{os.getpid()}"
        with self.parts.installed():
            h = fulldev.freeze(run, "M1-x2", "first:5")
            self.assertEqual(fulldev.check_identity(run, "M1-x2", "first:5")["hash"], h)
            probe = env.GPU1BIT / "actprec" / f"_identity_probe_{os.getpid()}.py"
            try:
                probe.write_text("# temporary file: a new source in the patch directory\n")
                with self.assertRaisesRegex(SystemExit, "identity mismatch"):
                    fulldev.check_identity(run, "M1-x2", "first:5")
            finally:
                probe.unlink()
            self.assertEqual(fulldev.check_identity(run, "M1-x2", "first:5")["hash"], h)
        shutil.rmtree(fulldev.ROOT / run)

    def test_xn_enabled_only_with_the_patch_and_supported_kernels(self):
        for arm in XN_ARMS:
            with self.assertRaisesRegex(RuntimeError, "held"):
                fulldev.check_arm_enabled(fulldev.parse_arm(arm))
        with self.parts.installed():
            for arm in XN_ARMS + ("M1-x3", "MB1-int8_mma-x3", "M1-fp32"):
                fulldev.check_arm_enabled(fulldev.parse_arm(arm))
            for arm in ("MB1-int8_mma:b01-x2", "MB1-triton_bf16:bf16x3-x1", "MB1-bitserial-x2"):
                with self.subTest(arm=arm), self.assertRaises(RuntimeError):
                    fulldev.check_arm_enabled(fulldev.parse_arm(arm))
            self.assertEqual(self.parts.reference_arm("MB1-int8_mma-x1"), "MB1-int8_mma-x3")
            self.assertEqual(self.parts.reference_arm("M1-x2"), "M1-x3")
        for arm in XN_ARMS:
            with self.assertRaisesRegex(RuntimeError, "held"):
                fulldev.check_arm_enabled(fulldev.parse_arm(arm))


class R2Gate(unittest.TestCase):
    """Review c1 major 3: no xN launch or classification against a missing / failed / invalid x3."""

    def setUp(self):
        from gpu1bit.actprec import parts
        self.parts = parts
        self.run = f"test-partcount-r2-{os.getpid()}"
        self.ctx = parts.installed()
        self.ctx.__enter__()

    def tearDown(self):
        self.ctx.__exit__(None, None, None)
        shutil.rmtree(fulldev.ROOT / self.run, ignore_errors=True)

    def _reference(self, **overrides):
        h = fulldev.freeze(self.run, "M1-x3", "full")
        summary = {"run": self.run, "arm": "M1-x3", "subset": "full", "identity": h, "complete": True,
                   "utterances": {n: len(fulldev.subset_records(n, "full")) for n in fulldev.DEV_SETS},
                   "structurally_valid": True, "cross_check": {"pass": True},
                   "spec": fulldev.parse_arm("M1-x3")}                    # as fulldev.finish writes it
        summary.update(overrides)
        (fulldev.arm_dir(self.run, "M1-x3", "full") / "summary.json").write_text(json.dumps(summary))
        return summary

    def test_valid_reference_passes(self):
        good = self._reference()
        self.assertEqual(self.parts.require_reference(self.run, "M1-x2")["identity"], good["identity"])

    def test_bad_references_stop(self):
        with self.assertRaisesRegex(SystemExit, "R2"):
            self.parts.require_reference(self.run, "M1-x2")                       # missing
        for bad in ({"cross_check": {"pass": False}}, {"cross_check": None}, {"structurally_valid": False},
                    {"complete": False}, {"identity": "0" * 64}, {"arm": "M1-x2"}, {"subset": "sample"},
                    {"run": "ap-some-other-run"},                                   # review c3: the run is checked
                    {"utterances": {n: 1 for n in fulldev.DEV_SETS}}):
            with self.subTest(bad=bad):
                self._reference(**bad)
                with self.assertRaisesRegex(SystemExit, "R2"):
                    self.parts.require_reference(self.run, "M1-x1")

    def test_direct_worker_needs_a_valid_reference(self):
        """Review c2: a direct `parts run` worker records xN rows only against a valid reference."""
        import types
        ran = []
        saved = fulldev.run
        fulldev.run = lambda args: ran.append(args.arm)
        try:
            args = types.SimpleNamespace(run=self.run, arm="M1-x1", subset="full", sets=None, work_budget=fulldev.WORK_BUDGET_S)
            with self.assertRaisesRegex(SystemExit, "R2"):
                self.parts.run_worker(args)                                          # no reference yet
            self._reference(cross_check={"pass": False})
            with self.assertRaisesRegex(SystemExit, "R2"):
                self.parts.run_worker(args)
            self.assertEqual(ran, [], "no worker body ran")
            self._reference()
            self.parts.run_worker(args)
            self.assertEqual(ran, ["M1-x1"])
            self.parts.run_worker(types.SimpleNamespace(run=self.run, arm="M1-x3", subset="full", sets=None,
                                                        work_budget=fulldev.WORK_BUDGET_S))
            self.assertEqual(ran, ["M1-x1", "M1-x3"], "the reference arm itself records without a precondition")
        finally:
            fulldev.run = saved

    def test_xn_driver_launches_nothing_without_a_valid_reference(self):
        import subprocess
        import types
        self._reference(cross_check={"pass": False})
        launched = []
        saved = subprocess.run
        # any launch is recorded and fails, so a missing gate cannot start a real worker from a test
        subprocess.run = lambda *a, **k: launched.append(a) or types.SimpleNamespace(returncode=1, stdout="{}")
        try:
            args = types.SimpleNamespace(run=self.run, arm="M1-x2", subset="full", sets=None,
                                         work_budget=fulldev.WORK_BUDGET_S, max_processes=3)
            with self.assertRaisesRegex(SystemExit, "R2"):
                self.parts.drive(args)
        finally:
            subprocess.run = saved
        self.assertEqual([a for a in launched if "run" in a[0]], [], "no worker was started")

    def test_x3_driver_fails_on_a_failed_cross_check(self):
        import subprocess
        import types
        saved = (fulldev.remaining, fulldev.finish, subprocess.run)
        launched = []
        # never start a real worker from a test: the arm is complete and any launch is recorded
        subprocess.run = lambda *a, **k: launched.append(a) or types.SimpleNamespace(returncode=1, stdout="{}")
        fulldev.remaining = lambda run, arm, sets, subset: {n: 0 for n in sets}
        try:
            for result in ({"structurally_valid": True, "cross_check": {"pass": False}},
                           {"structurally_valid": False, "cross_check": {"pass": True}}):
                fulldev.finish = lambda run, arm, subset="full", r=result: {"complete": True, **r}
                args = types.SimpleNamespace(run=self.run, arm="M1-x3", subset="full", sets=None,
                                             work_budget=fulldev.WORK_BUDGET_S, max_processes=3)
                with self.subTest(result=result), self.assertRaisesRegex(SystemExit, "R2"):
                    self.parts.drive(args)
        finally:
            fulldev.remaining, fulldev.finish, subprocess.run = saved
        self.assertEqual([a for a in launched if "run" in a[0]], [], "no worker was started")


# --- GPU: R1 paths ---------------------------------------------------------------------------------

class Probe:
    """Transparent wrapper of a projection provider: records each call as the provider receives it."""

    def __init__(self, inner):
        self.inner, self.calls, self.phase, self.names = inner, [], None, {}

    def __call__(self, x, mod, conv=False, components=3, activation="", norm=None, config=None, input_activation=""):
        self.calls.append({
            "phase": self.phase, "capturing": torch.cuda.is_current_stream_capturing(),
            "site": self.names.get(id(mod), f"unnamed:{getattr(mod, 'out_features', None)}x{getattr(mod, 'in_features', None)}"),
            "N": getattr(mod, "out_features", None), "K": getattr(mod, "in_features", None), "conv": bool(conv),
            "components": components, "activation": activation,
            "norm": None if norm is None else self.names.get(id(norm), "unnamed-norm"),
            "input_activation": input_activation, "x_shape": list(x.shape), "x_stride": list(x.stride()),
            "config": None if config is None else list(config)})
        return self.inner(x, mod, conv, components, activation, norm, config, input_activation)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def shapes() -> list[dict]:
    """Five dev_24 utterances of distinct sample counts, shortest first (4 fill the cache, 1 overflows)."""
    from gpu1bit.shared import workloads
    out, seen = [], set()
    for r in sorted(workloads.load()["dev_24"], key=lambda r: (r["samples"], r["id"])):
        if r["samples"] not in seen:
            seen.add(r["samples"])
            out.append(r)
    return out[:5]


def protocol(model, probe) -> dict:
    """Eager warm-up + capture (+ replay) of 4 shapes, a replay, an overflow shape, a direct eager call."""
    import evaluate as ev
    recs = shapes()
    cache = model.forward
    if type(cache).__name__ != "PipelineGraphCache":
        raise AssertionError(f"expected the deployed pipeline graph cache, got {type(cache).__name__}")
    cache.entries.clear()
    probe.names = {id(m): n for n, m in model.named_modules()}
    out = {}

    def call(phase, rec, eager=False):
        audio, length = ev.audio_batch([rec])
        probe.phase = phase
        fn = cache.model_forward if eager else model.forward
        enc, enc_len = fn(input_signal=audio.cuda(), input_signal_length=length.cuda())
        torch.cuda.synchronize()
        probe.phase = None
        return enc.clone(), enc_len.clone()

    with torch.inference_mode(), ev.strict_fp32("cuda"), ev.inference_settings(model):
        out["s1"] = call("s1-first", recs[0])
        out["s1-replay"] = call("s1-replay", recs[0])
        for j in (1, 2, 3):
            out[f"s{j + 1}"] = call(f"s{j + 1}-first", recs[j])
        out["entries_after_4"] = len(cache.entries)
        out["s5"] = call("s5-overflow", recs[4])
        out["entries_after_5"] = len(cache.entries)
        out["s1-eager"] = call("s1-eager", recs[0], eager=True)
    return out


def signature(calls: list[dict], phase: str) -> list[tuple]:
    return [tuple(json.dumps(c[k]) for k in SIGNATURE_KEYS) for c in calls if c["phase"] == phase]


PHASES = ("s1-first", "s1-replay", "s2-first", "s3-first", "s4-first", "s5-overflow", "s1-eager")


class _Paths:
    model_name = ""

    @classmethod
    def setUpClass(cls):
        from gpu1bit.actprec import parts
        from nemo.utils import logging
        logging.setLevel(logging.ERROR)
        cls.parts = parts
        cls.ctx = parts.installed()
        cls.ctx.__enter__()
        cls.ref_calls, cls.ref_out = cls._run(3)

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)

    @classmethod
    def _run(cls, n: int):
        made = []

        def probe(inner):
            p = Probe(inner)
            made.append(p)
            return p
        with torch.inference_mode():
            model, info = cls.parts.build(cls.model_name, n, probe=probe)
        try:
            if len(made) != 1:
                raise AssertionError(f"{len(made)} providers built; expected 1")
            p = made[0]
            pc = info["partcount"]
            if n == 3:
                if pc is not None:
                    raise AssertionError("the x3 reference must be the deployed stack (no adapter)")
            else:
                pc.record = []
            build_calls = [c for c in p.calls if c["phase"] is None]
            out = protocol(model, p)
            out["build_calls"] = build_calls
            out["adapter_record"] = None if pc is None else list(pc.record)
            out["composed_calls"] = getattr(p.inner, "composed_calls", 0)   # BinaryProjection: calls not via project
            out["info"] = info
            return p.calls, out
        finally:
            cls.parts.close(model)
            del model
            gc.collect()
            torch.cuda.empty_cache()

    def _check(self, n: int):
        calls, out = self._run(n)
        ref_calls, ref = self.ref_calls, self.ref_out
        self.assertTrue(all(c["components"] == 3 for c in ref_calls), "x3 reference sees components 3")
        self.assertTrue(calls, "no projection call recorded")
        self.assertEqual({c["components"] for c in calls}, {n}, "every call reaches the provider with n parts")
        protocol_calls = [c for c in calls if c["phase"] is not None]
        self.assertEqual(len(out["adapter_record"]), len(protocol_calls), "every call passes the adapter")
        self.assertEqual({eff for _, eff in out["adapter_record"]}, {n})
        self.assertEqual(out["composed_calls"], 0, "every MB1 call reaches the variant's fused project(components=n)")
        for phase in PHASES:
            with self.subTest(phase=phase):
                self.assertEqual(signature(calls, phase), signature(ref_calls, phase))
        first = [c for c in calls if c["phase"] == "s1-first"]
        self.assertTrue(any(c["capturing"] for c in first) and any(not c["capturing"] for c in first),
                        "s1-first covers eager warm-up and capture")
        self.assertEqual(signature(calls, "s1-replay"), [], "replay makes no projection call")
        over = [c for c in calls if c["phase"] == "s5-overflow"]
        self.assertTrue(over and not any(c["capturing"] for c in over), "overflow runs eagerly")
        self.assertEqual((out["entries_after_4"], out["entries_after_5"]), (4, 4))
        for key in ("s1", "s1-replay", "s2", "s3", "s4", "s5", "s1-eager"):
            enc, length = out[key]
            self.assertTrue(bool(torch.isfinite(enc).all()), key)
            self.assertTrue(torch.equal(length, ref[key][1]), f"{key}: lengths differ from x3")
        for key in ("s1-replay",):
            self.assertTrue(torch.equal(out[key][0], out["s1-eager"][0]), "replayed output == direct eager output")
            self.assertTrue(torch.equal(out[key][0], out["s1"][0]))
            self.assertFalse(torch.equal(out[key][0], ref[key][0]), "replayed output differs from x3's")
        self.assertFalse(torch.equal(out["s5"][0], ref["s5"][0]), "overflow output differs from x3's")
        t = int(ref["s1-eager"][1][0])
        rel = ((out["s1-eager"][0][..., :t] - ref["s1-eager"][0][..., :t]).norm() / ref["s1-eager"][0][..., :t].norm()).item()
        record = {"model": self.model_name, "n": n, "calls_by_phase": {p: len(signature(calls, p)) for p in PHASES},
                  "build_calls": len(out["build_calls"]), "adapter_calls": len(out["adapter_record"]),
                  "requested_components": sorted({r for r, _ in out["adapter_record"]}),
                  "s1_rel_frobenius_vs_x3": rel, "info": {k: v for k, v in out["info"].items() if k != "partcount"}}
        d = env.artifacts("partcount")
        (d / f"paths-{self.model_name}-x{n}.json").write_text(json.dumps(record, indent=1, default=str) + "\n")

    def test_x2_paths(self):
        self._check(2)

    def test_x1_paths(self):
        self._check(1)


@unittest.skipUnless(torch.cuda.is_available(), "GPU")
class PathsM1(_Paths, unittest.TestCase):
    model_name = "M1"


@unittest.skipUnless(torch.cuda.is_available(), "GPU")
class PathsMB1(_Paths, unittest.TestCase):
    model_name = "MB1"
