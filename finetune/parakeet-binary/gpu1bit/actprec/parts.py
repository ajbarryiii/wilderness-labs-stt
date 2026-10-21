"""Part-count arms of plans/activation-precision.md (R1, R4, R6) as a PATCH of gpu1bit/shared/fulldev.py.

Why a patch: int8_mma/ is frozen for the sel-20261020b selection session, and B's x2/x1 files stay in
B/wip/parts/ (coordinator 2026-10-20 14:55).

ARMS: the six activation-precision arms, all built by `build` (review c1, 2026-10-20):
M1-x3, M1-x2, M1-x1, MB1-int8_mma-x3, MB1-int8_mma-x2, MB1-int8_mma-x1 (int8_mma +-1; the
coordinator's choice).
  - x3: the deployed stack exactly (no adapter). Its recognition rows and the timed x3 variant
    (gpu1bit/actprec/bench.py) therefore come from the same builder under one identity.
  - M1-xN: fulldev's M1 build (inference.runtime.load_packed + enable_optimizations(expanded)), with
    the ExpandedProjection('triton_col_tuned') that the stack constructs wrapped in
    fulldev.PartCount(provider, N).
  - MB1-int8_mma-xN: fulldev's MB1 build (gpu1bit.runtime.load_binary_packed(runtime='provider')),
    with B's layer variant int8_mma:xN and the BinaryProjection that the runtime hands to
    enable_binary_optimizations wrapped in PartCount(provider, N).
  - PartCount sets components = N on every projection call, including layer_fusions4's hard-coded 3.
    The stack, its fusions and the x3 tile tables are unchanged (R1, R6).
  - fp32 arms stay fulldev's own (identity and build untouched).

B's files are loaded from WIP_DIR under their final package names (gpu1bit.int8_mma.kernels_parts,
layer_parts, parts_run). The loader reads each file once, checks its SHA-256 against PINNED (B's
14:55 manifest) and executes exactly the verified bytes. Importing parts_run registers int8_mma:x2,
x1, x2:b01 and x1:b01.

IDENTITY (R4): the identity of every x3 / xN arm is fulldev's identity plus:
  - every file of gpu1bit/actprec/ (this builder, bench.py for the timed variants) and the timing
    orchestrator gpu1bit/rerun_timing.py, all in "sources" (so fulldev.check_imports_covered accepts
    them);
  - for xN arms: the nine pinned files of B's manifest (paths and hashes; a mismatch is refused), the
    part count, the adapter and the layer variant.
The hash is recomputed over the whole document. fulldev's own identity binds the complete export
(review c1 major 2).

R2 GATE (review c1 major 3): `require_reference(run, arm)` admits an x3 reference only if its summary
names that run (review c3), is complete, for the full subset and the right arm, carries the identity that check_identity verifies,
is structurally valid (R5) and passed its cross-check. Every recording entry point checks it: the xN
driver before every worker, a direct xN worker (`run_worker`), `compare`, the timed variants
(bench.py) and the encoder checks (actprec_report.py); the x3 driver exits non-zero when its
finished arm fails it. Otherwise
the frozen R2 procedure applies, and no xN arm runs until the cause is known.

CLI (gpu1bit/shared/python -m gpu1bit.actprec.parts ...):
  run / drive / finish / compare    fulldev's actions with the patch installed. `drive` spawns
      `-m gpu1bit.actprec.parts run` workers for the six arms and fulldev's own workers for fp32 arms
      (one gpu lock each, R3). Before each worker it yields while the sel-20261020b timing rerun is
      pending and the host may become quiet.
  (encoder-check, a3 and report are in gpu1bit/actprec_report.py, outside the identities.)
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from gpu1bit.shared import env, fulldev

WIP_DIR = env.ARTIFACTS_ROOT / "B" / "wip" / "parts"
PINNED = {  # B/wip/parts-sha256.txt (NOTES "B 2026-10-20 14:55": these bytes will not change)
    "kernels_parts.py": "6e8cb12deeedd76cc9a627c1fb361a4e9372c0f6f0289fd30890448b51d271b7",
    "layer_parts.py": "4dc1a2e78d81885c533c42de83af10a4db0d65a55bdf884e598f17b601739786",
    "parts_run.py": "8e308ce5117c9d2e8e06592712144d02e0f47346e343fc2ff580134da42d2d29",
    "precompile.py": "02dac89308ece796f962c9c7ddb733f2f89c03fa6daca666b953e61982dda9a0",
    "run.py": "b558530395c74cdb6382190cf1556721df245dadd6f86ee9b6863336d5e9af66",
    "sanitize_parts.py": "627563dec1ca49479f0b397816b30a852fa8698516d7b8482593504dea77435b",
    "test_parts.py": "a435fed64ac6d8e1904c881f6fa689b052b62cc5e14d2bd336acad0c9c076e9f",
    "wip_main.py": "1f1e1eefedb1e5e95ca7b57cd342a9e11909e2433851cb8ab4cfeffb9a8afc14",
    "wip.py": "0c5415138d79946bfd7d370badd166bda15697b810adb56b1786dde11c5bb4cb",
}
LOADED = ("kernels_parts", "layer_parts", "parts_run")   # the modules the arms execute
SUPPORTED_KERNELS = ("int8_mma", "B-hybrid")   # B-hybrid: MB1 on the exact hybrid (plans/x2-default.md)
B_HYBRID_PARTS = (2, 3)
PATCH_DIR = Path(__file__).resolve().parent
EXTRA_SOURCES = (env.GPU1BIT / "rerun_timing.py",)   # the timing orchestrator (review c1: hashed too)
ADAPTER = "gpu1bit.shared.fulldev.PartCount(provider, n): components = n on every projection call (R1)"
BASE = {"identity": fulldev.identity, "check_arm_enabled": fulldev.check_arm_enabled, "build_arm": fulldev.build_arm}
_STATE = {"wip_dir": WIP_DIR}
RERUN = "sel-20261020b"
YIELD_BELOW_LOAD = 6.0   # load1 up to ~3 can be this driver's own last worker; above 6 the host is busy anyway


# --- B's pinned files ------------------------------------------------------------------------------

def verified_source(name: str) -> tuple[Path, bytes]:
    """(path, bytes) of B's module `name`, refused unless the bytes are the pinned ones."""
    path = _STATE["wip_dir"] / f"{name}.py"
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != PINNED[f"{name}.py"]:
        raise ValueError(f"{path}: SHA-256 {digest} is not the pinned {PINNED[name + '.py']} (B's 14:55 manifest)")
    return path, data


class _PinnedLoader(importlib.abc.Loader):
    def __init__(self, path: Path, data: bytes):
        self.path, self.data = path, data

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        exec(compile(self.data, str(self.path), "exec"), module.__dict__)


class _PinnedFinder(importlib.abc.MetaPathFinder):
    """`import gpu1bit.int8_mma.<LOADED>` -> the verified bytes of B/wip/parts/<name>.py."""

    def find_spec(self, fullname, path=None, target=None):
        prefix = "gpu1bit.int8_mma."
        if not (fullname.startswith(prefix) and fullname[len(prefix):] in LOADED):
            return None
        file, data = verified_source(fullname[len(prefix):])
        spec = importlib.util.spec_from_loader(fullname, _PinnedLoader(file, data), origin=str(file))
        spec.has_location = True
        return spec


def load_parts():
    """Import B's parts_run (registers the four part-count variants); returns {module: file}."""
    if not any(isinstance(f, _PinnedFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _PinnedFinder())
    out = {}
    for name in LOADED:  # parts_run imports layer_parts lazily (in discover); import all three now
        mod = importlib.import_module(f"gpu1bit.int8_mma.{name}")
        if Path(mod.__file__) != _STATE["wip_dir"] / f"{name}.py":
            raise RuntimeError(f"gpu1bit.int8_mma.{name} was loaded from {mod.__file__}, not the pinned file")
        out[name] = mod.__file__
    return out


# --- the patched fulldev functions ----------------------------------------------------------------

def _xn(spec: dict) -> bool:
    return spec["kind"] != "fp32" and spec["parts"] in (1, 2)


def reference_arm(arm: str) -> str:
    """The arm an xN arm is gated against: M1-x3 for M1; MB1-<kernel>-x3 for the provider arms; the
    provider-x2 arm g2 (MB1-int8_mma-x2) for the B-hybrid arms (plans/x2-default.md X4, Y2)."""
    spec = fulldev.parse_arm(arm)
    if spec["model"] == "M1":
        return "M1-x3"
    if spec["kernel"] == "B-hybrid":
        return "MB1-int8_mma-x2"
    return f"MB1-{spec['kernel']}-x3"


def is_r2_reference(arm: str) -> bool:
    """An x3 arm that gates others through R2 (M1-x3, MB1-int8_mma-x3); not B-hybrid-x3 (g3)."""
    spec = fulldev.parse_arm(arm)
    return spec["kind"] != "fp32" and spec["parts"] == 3 and spec["kernel"] != "B-hybrid" and _actprec(spec)


def _actprec(spec: dict) -> bool:
    """The six activation-precision arms (x3 and xN of M1 and MB1-int8_mma); not fp32 arms."""
    return spec["kind"] != "fp32" and (spec["model"] == "M1" or spec["kernel"] in SUPPORTED_KERNELS)


B_HYBRID_STACK = {"loader": "gpu1bit.runtime.load_binary_packed(MB1 export, runtime='hybrid', parts=n, selection=<in use>)",
                  "stack": "the selection document's hybrid entry for n parts (provider, pipeline, imports), "
                           "PartCount(provider, n) when n != 3", "decoder_graphs": "full_graph"}


def identity(arm: str, subset: str, selection=None) -> dict:
    doc = BASE["identity"](arm, subset, selection=selection)
    spec = fulldev.parse_arm(arm)
    if not _actprec(spec):
        return doc
    doc = {k: v for k, v in doc.items() if k != "hash"}
    files = sorted(p for p in PATCH_DIR.rglob("*") if p.is_file() and "__pycache__" not in p.parts) + list(EXTRA_SOURCES)
    doc["sources"] = {**doc["sources"], **{str(p.relative_to(env.REPO)): env.sha256_file(p) for p in files}}
    n = spec["parts"]
    doc["actprec_patch"] = {"builder": "gpu1bit.actprec.parts.build(model, 3): the deployed x3 stack" if n == 3 else
                            f"gpu1bit.actprec.parts.build(model, {n}): PartCount over the deployed stack",
                            "files": [str(p.relative_to(env.REPO)) for p in files]}
    if spec["kernel"] == "B-hybrid":
        doc["stack"] = dict(B_HYBRID_STACK)
    if not _xn(spec):
        doc["hash"] = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
        return doc
    wip = {}
    for name, pin in PINNED.items():
        path = _STATE["wip_dir"] / name
        digest = env.sha256_file(path)
        if digest != pin:
            raise ValueError(f"{path}: SHA-256 {digest} is not the pinned {pin} (B's 14:55 manifest)")
        wip[name] = {"path": str(path), "sha256": digest}
    doc["stack"] = {**doc["stack"], "parts_adapter": ADAPTER}
    doc["part_count_patch"] = {
        "parts": n, "adapter": ADAPTER,
        "layer_variant": (f"int8_mma:x{n}" if spec["kernel"] == "B-hybrid" else f"{spec['kernel']}:x{n}")
        if spec["model"] == "MB1" else None,
        "tile_tables": "the x3 tables, unchanged (R6): M1 expanded_configuration / int8_mma configs.choose",
        "wip_parts": wip, "loaded_modules": [f"gpu1bit.int8_mma.{m}" for m in LOADED],
        "patch_files": [str(p.relative_to(env.REPO)) for p in files]}
    doc["hash"] = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
    return doc


def check_arm_enabled(spec: dict) -> None:
    if spec["kind"] != "fp32" and spec["kernel"] == "B-hybrid":
        if spec["parts"] in B_HYBRID_PARTS:
            return
        raise RuntimeError(f"B-hybrid arms run at {B_HYBRID_PARTS} parts only (plans/x2-default.md)")
    if _xn(spec):
        if spec["model"] == "M1" or spec["kernel"] in SUPPORTED_KERNELS:
            return
        raise RuntimeError(f"part-count arms run for M1 and kernels {SUPPORTED_KERNELS} only (coordinator 2026-10-20)")
    BASE["check_arm_enabled"](spec)


def base_build_for(arm: str):
    """The builder an arm gets while the patch is installed: fulldev's for fp32, `build` otherwise."""
    return build if _actprec(fulldev.parse_arm(arm)) else BASE["build_arm"]


def build_arm(spec: dict):
    if not _actprec(spec):
        return BASE["build_arm"](spec)
    if spec["kernel"] == "B-hybrid":
        model, info = _build_b_hybrid(spec["parts"])
    else:
        model, info = build(spec["model"], spec["parts"])
    return model, {k: v for k, v in info.items() if k != "partcount"}


def _build_b_hybrid(n: int, probe=None):
    """(model, info) of MB1 on the exact hybrid at n parts, through the deployed loader (X3)."""
    return build("MB1", n, probe=probe, kernel="B-hybrid")


def _r2_stop(run: str, ref: str, reason: str):
    model = ref.split("-")[0]
    raise SystemExit(f"R2: x3 reference {ref} of run {run}: {reason}. Frozen procedure (plans/activation-precision.md "
                     f"R2): (1) batch-1 FP32 NeMo decoding of the same model rebuilt from its export (arm {model}-fp32), "
                     "(2) if that also misses, a historical-batch reproduction, (3) a report to the user; no xN arm runs "
                     "or is classified until the cause is known.")


def require_reference(run: str, arm: str) -> dict:
    """The verified summary of `arm`'s reference (reference_arm), or SystemExit (R2; plans/x2-default.md
    Y2). The reference must be complete over the full dev sets, of this run and arm, carry the identity
    check_identity verifies, be of the same model and structurally valid; an x3 reference must have
    passed R2, an xN reference must itself pass this check (the chain: g1/g3 -> g2 -> g0)."""
    ref = reference_arm(arm)
    model = fulldev.parse_arm(arm)["model"]
    path = fulldev.arm_dir(run, ref, "full") / "summary.json"
    if not path.exists():
        _r2_stop(run, ref, "no summary (not run or not finished)")
    s = json.loads(path.read_text())
    want = {n: len(fulldev.subset_records(n, "full")) for n in fulldev.DEV_SETS}
    if s.get("run") != run or s.get("arm") != ref or s.get("subset") != "full":
        _r2_stop(run, ref, f"summary is for {s.get('arm')}@{s.get('subset')} of run {s.get('run')}")
    if (s.get("spec") or {}).get("model") != model or fulldev.parse_arm(ref)["model"] != model:
        _r2_stop(run, ref, f"reference of another model ({(s.get('spec') or {}).get('model')}) for an {model} arm")
    if s.get("complete") is not True or s.get("utterances") != want:
        _r2_stop(run, ref, "summary is not complete over the full dev sets")
    try:
        verified = fulldev.check_identity(run, ref, "full")["hash"]
    except SystemExit as exc:
        _r2_stop(run, ref, f"identity not verified ({exc})")
    if s.get("identity") != verified:
        _r2_stop(run, ref, "summary identity differs from the verified frozen identity")
    if s.get("structurally_valid") is not True:
        _r2_stop(run, ref, "structurally invalid (R5)")
    if fulldev.parse_arm(ref)["parts"] == 3:
        if not (s.get("cross_check") or {}).get("pass"):
            _r2_stop(run, ref, f"cross-check against the pinned FP32-NeMo scores failed or missing: {s.get('cross_check')}")
    else:
        require_reference(run, ref)
    return s


def _patch(**functions) -> None:
    for name, fn in functions.items():
        setattr(fulldev, name, fn)


def install(wip_dir: Path | None = None) -> None:
    _STATE["wip_dir"] = Path(wip_dir) if wip_dir else WIP_DIR
    _patch(identity=identity, check_arm_enabled=check_arm_enabled, build_arm=build_arm)


def uninstall() -> None:
    _STATE["wip_dir"] = WIP_DIR
    _patch(**BASE)


@contextlib.contextmanager
def installed(wip_dir: Path | None = None):
    install(wip_dir)
    try:
        yield
    finally:
        uninstall()


# --- building an arm (the caller holds the gpu lock; inference_mode on) ----------------------------

def build(model_name: str, n: int, *, probe=None, kernel: str = "int8_mma"):
    """(model, info). n = 3 is the deployed x3 stack exactly (no adapter), for references and tests;
    `probe(provider)` may wrap the provider transparently (instrumentation). M1 uses
    gpu1bit.runtime_m1.build_stack; MB1 uses gpu1bit.runtime.load_binary_packed (provider runtime with
    the int8_mma[:xN] variant, or kernel="B-hybrid" for the hybrid runtime under the selection in use)."""
    import torch
    if n not in (1, 2, 3):
        raise ValueError("n must be 1, 2 or 3")
    made = []

    def hook(provider):
        made.append(provider)
        return probe(provider) if probe else provider

    loaded = None
    if model_name == "M1":
        from gpu1bit import runtime_m1
        model, (_, prov) = runtime_m1.build_stack(env.M1_EXPORT, n, wrap=hook)
        kernel, variant, kind, stack = None, None, "M1-x", dict(fulldev.STACKS["M1-x"])
    elif model_name == "MB1" and kernel == "B-hybrid":
        from gpu1bit import runtime
        from gpu1bit.shared import seldoc
        if n != 3:
            loaded = load_parts()
        model = runtime.load_binary_packed(env.MB1_EXPORT, runtime="hybrid", parts=n, _provider_hook=hook)
        prov = model._gpu1bit_runtime.provider
        variant = model._gpu1bit_runtime.kernel
        kind, stack = "MB1-x", {**B_HYBRID_STACK, "selection_sha256": seldoc.load()["sha256"]}
    elif model_name == "MB1":
        from gpu1bit import runtime
        variant = kernel if n == 3 else f"{kernel}:x{n}"
        if n != 3:
            loaded = load_parts()
        count = {"parts": n} if n in (2, 3) else {"_research_parts": n}   # x1: research arm, provider runtime only
        model = runtime.load_binary_packed(env.MB1_EXPORT, runtime="provider", kernel=variant, _provider_hook=hook, **count)
        prov = model._gpu1bit_runtime.provider
        kind, stack = "MB1-x", dict(fulldev.STACKS["MB1-x"])
    else:
        raise ValueError(f"model {model_name!r}")
    if len(made) != 1:
        raise AssertionError(f"{len(made)} projection providers were built; expected exactly 1")
    torch.cuda.synchronize()
    comp = model.decoding.decoding.decoding_computer
    info = {"stack": {**stack, **({"parts_adapter": ADAPTER} if n != 3 else {})}, "parts": n,
            "kernel": kernel, "layer_variant": variant, "provider": type(made[0]).__name__,
            "partcount": prov if isinstance(prov, fulldev.PartCount) else None, "wip_loaded": loaded,
            "decoder_cuda_graphs_mode": str(getattr(comp, "cuda_graphs_mode", None)),
            "pipeline_graph_cache": type(model.forward).__name__}
    return model, info


def close(model) -> None:
    import torch
    with torch.inference_mode():
        torch.cuda.synchronize()
        if hasattr(model, "_gpu1bit_runtime"):
            from gpu1bit import runtime
            runtime.close_binary_packed(model)
        else:
            from inference.optimized import disable_optimizations
            disable_optimizations(model)


# --- driver (no CUDA): yields to the sel-20261020b timing rerun on a quiet host ----------------------

def _cpu_heavy_holders() -> list:
    try:
        doc = json.loads(subprocess.run(["job", "queue", "--json"], capture_output=True, text=True, timeout=30).stdout)
    except Exception as exc:  # noqa: BLE001
        return [repr(exc)]
    return [h.get("name") for r in doc.get("resources", []) if r.get("name") == "cpu-heavy" for h in r.get("holders", [])]


def timing_has_priority() -> str | None:
    """A reason to wait, or None. The rerun goes first whenever the host may become quiet."""
    log = env.ARTIFACTS / "selection" / RERUN / "rerun_timing.log"
    text = log.read_text() if log.exists() else ""
    if "RERUN COMPLETE" in text:
        return None
    alive = subprocess.run(["pgrep", "-f", f"rerun_timing.py --run {RERUN}"], capture_output=True).returncode == 0
    if not alive:
        return None
    if _cpu_heavy_holders():
        return None
    load1 = float(Path("/proc/loadavg").read_text().split()[0])
    if load1 >= YIELD_BELOW_LOAD:  # busy without cpu-heavy holders: the rerun cannot start either
        return None
    return (f"{RERUN} timing rerun pending, no cpu-heavy holder and load1 {load1} < {YIELD_BELOW_LOAD}: "
            "the host may become quiet, so timing goes first")


def drive(args) -> None:
    spec = fulldev.parse_arm(args.arm)
    fulldev.check_arm_enabled(spec)
    # the six activation-precision arms run this module's workers (their identities hash the patch);
    # fp32 workers are fulldev's own (base identity).
    worker = "gpu1bit.actprec.parts" if _actprec(spec) else "gpu1bit.shared.fulldev"
    if _xn(spec):
        require_reference(args.run, args.arm)
    sets = fulldev.parse_sets(args.sets)
    frozen_hash = fulldev.freeze(args.run, args.arm, args.subset)
    d = fulldev.arm_dir(args.run, args.arm, args.subset)
    processes = 0
    while True:
        left = fulldev.remaining(args.run, args.arm, sets, args.subset)
        if not any(left.values()):
            break
        if processes >= args.max_processes:
            raise SystemExit("max processes reached before completion")
        waited = 0.0
        while (reason := timing_has_priority()) is not None:
            if waited == 0.0:
                print(json.dumps({"waiting": reason, "since": time.strftime("%Y-%m-%dT%H:%M:%S")}), flush=True)
            time.sleep(30)
            waited += 30
        fulldev.check_identity(args.run, args.arm, args.subset)
        if _xn(spec):
            require_reference(args.run, args.arm)
        cmd = ["timeout", "600", fulldev.PY, "-m", worker, "run", "--run", args.run, "--arm", args.arm,
               "--subset", args.subset, "--sets", *sets, "--work-budget", str(args.work_budget)]
        e = dict(os.environ)
        e.pop("CUBLAS_WORKSPACE_CONFIG", None)
        e.pop("CUDA_VISIBLE_DEVICES", None)
        with open(d / "driver.log", "a") as log:
            log.write(f"\n### {time.strftime('%Y-%m-%dT%H:%M:%S')} identity={frozen_hash[:16]} left={left} "
                      f"yielded_s={waited:.0f} {' '.join(cmd)}\n")
            log.flush()
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(env.BINARY_DIR), env=e).returncode
        processes += 1
        print(json.dumps({"process": processes, "rc": rc, "left_before": left, "yielded_s": waited}), flush=True)
        if rc != 0:
            raise SystemExit(f"run process failed (rc {rc}); see {d / 'driver.log'}")
    if set(sets) == set(fulldev.DEV_SETS):
        result = fulldev.finish(args.run, args.arm, args.subset)
        if is_r2_reference(args.arm) and args.subset == "full":
            if not result or not result.get("complete"):
                _r2_stop(args.run, args.arm, "incomplete after its workers")
            if result.get("structurally_valid") is not True:
                _r2_stop(args.run, args.arm, "structurally invalid (R5)")
            if not (result.get("cross_check") or {}).get("pass"):
                _r2_stop(args.run, args.arm, f"cross-check failed: {result.get('cross_check')}")


def run_worker(args) -> None:
    """A direct `run` worker: xN rows are recorded only against a reference that passes the R2 gate
    (review c2); the x3 arm itself records without a precondition (it produces the reference)."""
    if _xn(fulldev.parse_arm(args.arm)):
        require_reference(args.run, args.arm)
    fulldev.run(args)


def compare(run: str, arm: str, ref: str) -> dict:
    """fulldev.compare (which records both identities) behind the R2 gate and completeness checks."""
    if ref != reference_arm(arm):
        raise SystemExit(f"{arm} is compared with {reference_arm(arm)}, not {ref}")
    require_reference(run, arm)
    s = json.loads((fulldev.arm_dir(run, arm, "full") / "summary.json").read_text())
    if s.get("identity") != fulldev.check_identity(run, arm, "full")["hash"] or s.get("complete") is not True:
        raise SystemExit(f"{arm}: summary not complete or not of the verified identity")
    return fulldev.compare(run, arm, ref, "full")


# --- CLI -------------------------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    for name in ("run", "drive", "finish"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        s.add_argument("--arm", required=True)
        s.add_argument("--subset", default="full")
        if name in ("run", "drive"):
            s.add_argument("--sets", nargs="+", default=None)
            s.add_argument("--work-budget", type=float, default=fulldev.WORK_BUDGET_S)
        if name == "drive":
            s.add_argument("--max-processes", type=int, default=60)
    s = sub.add_parser("compare")
    s.add_argument("--run", required=True)
    s.add_argument("--arm", required=True)
    s.add_argument("--ref", required=True)
    s.add_argument("--subset", default="full")
    args = p.parse_args()
    if args.action in ("run", "drive") and not 0 < args.work_budget <= fulldev.WORK_BUDGET_S:
        p.error(f"--work-budget must be in (0, {fulldev.WORK_BUDGET_S:.0f}]")
    install()
    if args.action == "run":
        env.require_shared_wrapper()
        run_worker(args)
    elif args.action == "drive":
        drive(args)
    elif args.action == "finish":
        fulldev.finish(args.run, args.arm, args.subset)
    else:
        if args.subset != "full":
            raise SystemExit("compare: full subset only")
        compare(args.run, args.arm, args.ref)


if __name__ == "__main__":
    main()
