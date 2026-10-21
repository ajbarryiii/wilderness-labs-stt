"""Binary-weight QAT of Parakeet-TDT-0.6B-v2 (and the ternary control C) with train.py, from the FP32 original or from M1. See plans/pilot.md.

    python binary_train.py --init {fp32,m1} --quantizer {binary,ternary} <train.py arguments>

One invocation is one pilot arm in RUNS/<run-name>, RUNS = /mnt/hd/wilderness-labs-stt/
parakeet-binary/runs. The training loop, checkpoints, resume, evaluation and finishing are
finetune/parakeet-ternary/train.py, unchanged (its hashes are M1's provenance); train.main runs
inside seam(), which replaces a fixed list of train.py and paths.py module attributes and
restores them on exit (plans/pilot.md "Implementation seam"):

1. train.parse_args: adds --init and --quantizer and the derived, recorded arguments ramp_kind,
   init_sha256 and quantizer_sha256 (config.json, every checkpoint's "args", resumes.jsonl);
2. train.RESUME_KEYS: plus BINARY_RESUME_KEYS, so a checkpoint written under another init,
   quantizer, ramp or quantizer source is refused (train.py exit 78) instead of resumed;
3. train.TRAINING_SOURCES: plus BINARY_SOURCES (recorded at start, every resume and finish);
4. train.build_context: this module's build_context (below);
5. train.local_module: "export" is FinishingExport over the arm's export (binary_export, or
   parakeet-ternary export.py for C): the rebuilt model's decoder mode is pinned on loading and
   an inexact reconstruction (codes or scales) raises ReconstructionError before anything is
   scored; for the binary arms "quant" is binary_quant;
6. train.finish: adds a "binary" block to the result, hence to summary.json;
7. paths.RUNS, paths.POWER, paths.ARTIFACTS (the ternary export's allowed root, used by arm C)
   and train.POWERLOG_LAUNCH: everything is written under the binary artifacts. The GPU lock
   stays the ternary project's (both projects share one GPU).

Arms (--init, --quantizer -> ramp): fp32 + binary -> "fp32", W_eff = lerp(W, B_ste, f) (arm A);
m1 + binary -> "ternary", lerp(T_ste, B_ste, f) (B5, B2); m1 + ternary -> "none" (control C:
the ternary quantizer unchanged, Context.ramp = 0 so train.py sets f = 1 at every update).
fp32 + ternary is the ternary pilot itself and is refused. Every arm is train.py --arm P2.

Initialization (--init m1): the student is built as in train.py (the pinned FP32 .nemo), then
every student tensor is replaced by M1's (read_init_state / apply_init_state): the checkpoint's
SHA-256 must be binary_paths.M1_SHA256 (cached by file identity, binary_paths.cached_sha256),
it must be complete, at step 250,000 and weight fraction 1, and its key set must equal the
student's exactly. Optimizer, scheduler and RNG states start fresh (train.Context). On a resume
(a checkpoint exists in the run directory, which train.main has already loaded and matched
against these arguments) M1 is not read: Context.restore replaces every student tensor.

Stream: the Hub stream is opened with binary_paths.STREAM_WORKERS[phase] workers (the ternary
pilot's 6 for --train pilot; stream.py's default is now 8) and its configuration hash must be
the one the ternary runs recorded, so arm A sees ternary pilot P2's data in the same order.

Decoder: every model the run decodes with (the teacher labeling every batch, the student in
dev-subset evaluations and the reconstruction check, the model rebuilt from the export) runs
NeMo's greedy TDT decoder in CUDA-graph mode binary_paths.DECODER_GRAPHS_MODE ("no_while_loops",
native PyTorch CUDA graphs), the mode M1 decoded with; it is forced with NeMo's
force_cuda_graphs_mode, checked before and after every use, and recorded in config.json and
summary.json. Any other mode raises DecoderModeError (plans/pilot.md "Review round 1").

Reporting: the latent signs (W >= 0) at the start of the arm are stored once, packed as binary
codes, in RUNS/<run>/start-signs.safetensors; at the end the fraction of latent signs that
changed (overall and per family) and the code balance (-1 / 0 / +1, overall and per family) go
into summary.json["binary"], with the decoder modes and the training wall time (BinaryContext).
"""
from __future__ import annotations

import train  # noqa: I001  finetune/parakeet-ternary/train.py, unchanged; first, as it configures CUDA/numba

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType

import torch
from safetensors.torch import load_file, save_file

import binary_export
import binary_paths
import binary_quant
import paths
import quant as ternary_quant  # finetune/parakeet-ternary/quant.py, unchanged

INITS = ("fp32", "m1")
QUANTIZERS = ("binary", "ternary")
# (init, quantizer) -> ramp kind; plans/pilot.md "Ramps" and "Arms".
RAMP_KINDS = {("fp32", "binary"): "fp32", ("m1", "binary"): "ternary", ("m1", "ternary"): "none"}
BINARY_RESUME_KEYS = ("init", "init_sha256", "quantizer", "ramp_kind", "quantizer_sha256")
BINARY_SOURCES = binary_paths.BINARY_SOURCES
START_SIGNS = "start-signs.safetensors"
DECODER_GRAPHS_MODE = binary_paths.DECODER_GRAPHS_MODE
clock = time.monotonic  # BinaryContext's training wall clock
POWERLOG_LAUNCH = ("import sys, pathlib, paths, powerlog; "
                   f"powerlog.POWER_DIR = pathlib.Path({str(binary_paths.POWER)!r}); "
                   "paths._RUN_NAME = paths._NAME; powerlog.main(sys.argv[1:])")

# train.py's originals, captured before seam() replaces them.
_TRAIN = {name: getattr(train, name) for name in (
    "parse_args", "RESUME_KEYS", "TRAINING_SOURCES", "build_context", "local_module", "finish",
    "POWERLOG_LAUNCH")}
_ACTIVE: dict = {"quantizer": None, "export": None}


def log(msg: str) -> None:
    train.log(msg)


# ----------------------------------------------------------------------------- arguments

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """train.py's arguments plus --init and --quantizer (and the derived binary arguments)."""
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(prog="binary_train.py", add_help=False,
                                     description=__doc__.split("\n", 1)[0])
    parser.add_argument("--init", choices=INITS, required=True,
                        help="fp32: the pinned pretrained model; m1: the pinned M1 checkpoint's student")
    parser.add_argument("--quantizer", choices=QUANTIZERS, required=True,
                        help="binary (arms A, B5, B2) or ternary, unchanged (control C, --init m1 only)")
    if any(a in ("-h", "--help") for a in argv):
        parser.print_help()
        _TRAIN["parse_args"](["--help"])
    known, rest = parser.parse_known_args(argv)
    args = _TRAIN["parse_args"](rest)
    try:
        return resolve_binary_args(args, known.init, known.quantizer)
    except ValueError as error:
        parser.error(str(error))


def resolve_binary_args(args: argparse.Namespace, init: str, quantizer: str) -> argparse.Namespace:
    if (init, quantizer) not in RAMP_KINDS:
        raise ValueError(f"--init {init} --quantizer {quantizer} is the ternary pilot itself; "
                         "run finetune/parakeet-ternary/train.py for it")
    if args.arm == "A1":
        raise ValueError("arm A1 is train.py's unquantized control; every binary pilot arm is quantized")
    args.init, args.quantizer = init, quantizer
    args.ramp_kind = RAMP_KINDS[(init, quantizer)]
    args.init_sha256 = binary_paths.init_sha256(init)
    args.quantizer_sha256 = binary_paths.quantizer_sha256(quantizer)
    return args


# ----------------------------------------------------------------------------- initialization

def read_init_state(path: Path | None = None, *, expected_sha256: str | None = None,
                    expected_step: int | None = None) -> tuple[dict, dict]:
    """(student state_dict, record) of the pinned train.py checkpoint (default: M1).

    Refuses (ValueError), before unpickling, a file whose SHA-256 is not expected_sha256; then an
    incomplete or other-format checkpoint, a step other than expected_step and a weight fraction
    other than 1. Tensors are memory-mapped: the file's optimizer state is never read into memory.
    """
    path = Path(binary_paths.M1_CHECKPOINT if path is None else path)
    expected_sha256 = expected_sha256 or binary_paths.M1_SHA256
    expected_step = binary_paths.M1_STEP if expected_step is None else expected_step
    digest = binary_paths.cached_sha256(path)
    if digest != expected_sha256:
        raise ValueError(f"{path}: SHA-256 {digest} is not the pinned {expected_sha256}; "
                         "refusing to initialize from it")
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if not (isinstance(payload, dict) and payload.get("complete") and payload.get("format") == train.CKPT_FORMAT):
        raise ValueError(f"{path} is not a complete train.py checkpoint of format {train.CKPT_FORMAT}")
    if payload.get("step") != expected_step:
        raise ValueError(f"{path}: step {payload.get('step')!r} != {expected_step}")
    fraction = payload.get("weight_fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or float(fraction) != 1.0:
        raise ValueError(f"{path}: weight fraction {fraction!r} != 1 (a ramp state, not a finished model)")
    record = {"path": str(path), "sha256": digest, "step": payload["step"], "weight_fraction": float(fraction)}
    return payload["student"], record


def apply_init_state(student: torch.nn.Module, state: dict) -> int:
    """Load every tensor of `state` into the student (parameters and buffers); returns the count.

    Refuses (ValueError, the student untouched) a key set that is not exactly the student's or
    a shape mismatch, so the latent encoder weights and every float tensor come from the init.
    """
    own = student.state_dict()
    missing, extra = sorted(set(own) - set(state)), sorted(set(state) - set(own))
    problems = []
    if missing:
        problems.append(f"missing keys ({len(missing)}): {missing[:5]}")
    if extra:
        problems.append(f"extra keys ({len(extra)}): {extra[:5]}")
    shapes = [k for k in own if k in state and tuple(state[k].shape) != tuple(own[k].shape)]
    if shapes:
        problems.append(f"shape mismatches ({len(shapes)}): {shapes[:5]}")
    if problems:
        raise ValueError(f"init state does not match the student: {'; '.join(problems)}")
    student.load_state_dict(state, strict=True)
    return len(state)


def load_init(student: torch.nn.Module) -> dict:
    """The pinned M1 student state into `student`; returns the init record."""
    state, record = read_init_state()
    record["tensors"] = apply_init_state(student, state)
    del state
    train.drop_page_cache(Path(record["path"]), sync=False)
    log(f"initialized the student from {record['path']} (step {record['step']}, {record['tensors']} tensors)")
    return record


def is_resume(run: Path, args) -> bool:
    """Whether train.main is resuming this run: it has already loaded (or refused) the newest
    checkpoint before building the context, so a checkpoint file present now means a resume."""
    latest, previous, _ = train.ckpt_paths(run)
    return not args.probe_steps and (latest.exists() or previous.exists())


def prepare_student(args, student: torch.nn.Module, resuming: bool) -> tuple[ModuleType, dict]:
    """Initialize (fresh start of an m1 arm only) and quantize the student with the arm's
    quantizer and ramp; returns (quantizer module, init record)."""
    init = {"init": args.init, "sha256": args.init_sha256, "loaded": not resuming}
    if args.init == "m1" and not resuming:
        init.update(load_init(student))
    if args.quantizer == "binary":
        quantizer = binary_quant
        quantizer.quantize_parakeet(student, args.ramp_kind)
    else:
        quantizer = ternary_quant
        quantizer.quantize_parakeet(student)
    return quantizer, init


class BinaryContext(train.Context):
    """train.Context with the arm's ramp, its decoder modes and its training wall time.

    Ramp kind "none" (control C) has ramp = 0, so train.py sets f = ramp_weight_fraction(k, 0) = 1
    from the first update on. timing = {"train_wall_s", "sessions", "step"}: the wall time from
    building each session's context to its latest checkpoint, summed over sessions, carried in
    every checkpoint as "binary_timing" and restored on resume (train.py's step_time_s is reset at
    evaluations, checkpoints and resumes, so it cannot be summed into a total). Work lost after a
    killed session's last checkpoint is not counted.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.args.ramp_kind == "none":
            self.ramp = 0
        self.decoder_graphs: dict[str, str] = {}
        self.timing = {"train_wall_s": 0.0, "sessions": 1, "step": self.state.step}
        self._timing_base, self._timing_start = 0.0, clock()

    def checkpoint_payload(self) -> dict:
        payload = super().checkpoint_payload()
        self.timing = {**self.timing, "train_wall_s": self._timing_base + clock() - self._timing_start,
                       "step": self.state.step}
        return {**payload, "binary_timing": dict(self.timing)}

    def restore(self, payload: dict) -> None:
        super().restore(payload)
        saved = payload.get("binary_timing")
        if saved is None:  # written before the timing record existed (the 2026-10-16 smoke runs)
            saved = {"train_wall_s": 0.0, "sessions": 0, "step": payload["step"],
                     "uncounted_before_step": payload["step"]}
        self._timing_base, self._timing_start = saved["train_wall_s"], clock()
        self.timing = {**saved, "sessions": saved["sessions"] + 1}


def make_context(args, run: Path, student, teacher, stream, evaluator, quantizer: ModuleType,
                 device: str, label_fn=None) -> BinaryContext:
    """BinaryContext; by default the teacher labels with teacher.teacher_label_batch with its
    decoder mode checked before and after every batch."""
    if label_fn is None:
        label_fn = verified_label_fn(train.local_module("teacher").teacher_label_batch)
    return BinaryContext(args, run, student, teacher, stream, evaluator, quantizer, device, label_fn=label_fn)


def open_stream(args):
    """The training stream: train.open_stream for local streams; the Hub stream with the
    compared ternary run's worker count, checked against its recorded configuration hash."""
    if args.stream != "hub":
        return train.open_stream(args)
    stream = train.local_module("stream")
    opened = stream.make_training_stream(args.train, args.seed,
                                         num_workers=binary_paths.STREAM_WORKERS[args.train])
    expected = binary_paths.STREAM_CONFIG_HASH[args.train]
    if args.seed == paths.SEED and opened.config_hash != expected:
        raise RuntimeError(f"stream configuration hash {opened.config_hash} != {expected} recorded by the "
                           f"ternary {args.train} runs: the data order would differ")
    return opened


# ----------------------------------------------------------------------------- sign report

def latent_signs(student: torch.nn.Module, names: list[str]) -> dict[str, torch.Tensor]:
    """Per quantized module, the latent weight's signs (W >= 0 -> +1) packed as binary codes."""
    return {name: binary_export.pack_codes(
        binary_quant.binary_quantize(student.get_submodule(name).matrix())[0]).cpu() for name in names}


def write_start_signs(run: Path, student: torch.nn.Module, names: list[str]) -> dict:
    path = run / START_SIGNS
    tmp = run / f"{START_SIGNS}.tmp{os.getpid()}"
    save_file(latent_signs(student, names), tmp, metadata={"content": "latent signs at the arm's start"})
    os.replace(tmp, path)
    return {"file": START_SIGNS, "sha256": binary_paths.sha256_file(path)}


@torch.no_grad()
def sign_changes(run: Path, student: torch.nn.Module, names: list[str]) -> dict | None:
    """Fraction of latent signs that differ from the arm's start, overall and per family; None
    (with a warning) if the start file is missing or does not cover these modules."""
    path = run / START_SIGNS
    if not path.exists():
        log(f"warning: {path} missing; no sign-change report")
        return None
    start = load_file(path)
    if set(start) != set(names):
        log(f"warning: {path} covers other modules; no sign-change report")
        return None
    changed: dict[str, int] = {}
    weights: dict[str, int] = {}
    for name in names:
        now = binary_quant.binary_quantize(student.get_submodule(name).matrix())[0].cpu()
        before = binary_export.unpack_codes(start[name], now.shape[1])
        family = binary_quant.family(name)
        changed[family] = changed.get(family, 0) + int((now != before).sum())
        weights[family] = weights.get(family, 0) + now.numel()
    total = sum(weights.values())
    return {"fraction": sum(changed.values()) / total, "changed": sum(changed.values()), "weights": total,
            "per_family": {f: changed[f] / weights[f] for f in sorted(weights)},
            "start_file_sha256": binary_paths.sha256_file(path)}


# ----------------------------------------------------------------------------- decoder mode, reconstruction

class DecoderModeError(RuntimeError):
    """A model would decode (or has decoded) in another NeMo CUDA-graph mode than M1's."""


class ReconstructionError(RuntimeError):
    """The model rebuilt from an export does not have exactly the QAT model's codes and scales."""


def _decoding_computer(model, role: str):
    computer = getattr(getattr(getattr(model, "decoding", None), "decoding", None), "decoding_computer", None)
    if computer is None or not hasattr(computer, "force_cuda_graphs_mode"):
        raise DecoderModeError(f"{role}: no NeMo label-looping greedy decoder (model.decoding.decoding."
                               "decoding_computer); cannot pin its CUDA-graph mode")
    return computer


def decoder_graphs_mode(model, role: str = "model") -> str | None:
    """The CUDA-graph mode of the model's greedy TDT decoder ("full_graph", "no_while_loops",
    "no_graphs") or None (graphs disabled). On the CPU NeMo decodes without graphs in any mode."""
    mode = _decoding_computer(model, role).cuda_graphs_mode
    return None if mode is None else str(mode.value)


def require_decoder_graphs_mode(model, role: str) -> str:
    """DECODER_GRAPHS_MODE, or DecoderModeError."""
    mode = decoder_graphs_mode(model, role)
    if mode != DECODER_GRAPHS_MODE:
        raise DecoderModeError(f"{role}: NeMo greedy TDT decoder in CUDA-graph mode {mode!r}, not "
                               f"{DECODER_GRAPHS_MODE!r} (native PyTorch CUDA graphs, as M1 decoded); refusing")
    return mode


def pin_decoder_graphs_mode(model, role: str) -> str:
    """Put the model's decoder in DECODER_GRAPHS_MODE (NeMo force_cuda_graphs_mode: the partial-graph
    decoder M1 fell back to on its first decode, without first trying the full graph) and verify.
    Refuses a decoder whose config disallows CUDA graphs (not M1's config)."""
    computer = _decoding_computer(model, role)
    if not computer.allow_cuda_graphs:
        raise DecoderModeError(f"{role}: the decoder config disallows CUDA graphs (use_cuda_graph_decoder "
                               "false), unlike M1's; refusing")
    if decoder_graphs_mode(model, role) != DECODER_GRAPHS_MODE:
        computer.force_cuda_graphs_mode(DECODER_GRAPHS_MODE)
    return require_decoder_graphs_mode(model, role)


def verified_label_fn(label_fn):
    """label_fn(teacher, ...) with the teacher's decoder mode checked before and after (never labels
    in another mode)."""
    def label(teacher, *args, **kwargs):
        require_decoder_graphs_mode(teacher, "teacher")
        result = label_fn(teacher, *args, **kwargs)
        require_decoder_graphs_mode(teacher, "teacher")
        return result
    return label


class DecoderPinnedEvaluator:
    """A dev evaluator (train.DevEvaluator's subset/full) that pins every model's decoder mode
    before decoding and checks it after; `modes` holds the effective mode per kind. Other
    attributes (subsets, evaluate) are the wrapped evaluator's."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.modes: dict[str, str] = {}

    def __getattr__(self, name: str):
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def _decode(self, kind: str, method, model):
        pin_decoder_graphs_mode(model, f"{kind} model")
        results = method(model)
        self.modes[kind] = require_decoder_graphs_mode(model, f"{kind} model")
        return results

    def subset(self, model):
        return self._decode("dev-subset", self.inner.subset, model)

    def full(self, model):
        return self._decode("full-dev", self.inner.full, model)


def require_exact_reconstruction(check: dict, label: str = "export") -> dict:
    """The reconstruction record, if its codes and scales are exact; else ReconstructionError."""
    codes, scales = check.get("codes_exact"), check.get("scales_exact")
    if codes is not True or scales is not True:
        raise ReconstructionError(f"{label}: reconstruction is not exact (codes_exact={codes!r}, "
                                  f"scales_exact={scales!r}); refusing to score the rebuilt model")
    return check


class FinishingExport:
    """The export module train.finish and binary_eval use: the arm's export module (binary_export,
    or parakeet-ternary export.py for C) with the rebuilt model's decoder mode pinned on loading and
    an exact reconstruction required (train.reconstruction runs before any scoring)."""

    def __init__(self, module: ModuleType) -> None:
        self.module = module
        self.__name__ = module.__name__
        self.rebuilt_mode: str | None = None

    def export_model(self, model, out_dir: Path, extra: dict) -> dict:
        return self.module.export_model(model, out_dir, extra)

    def load_export(self, out_dir: Path, device: str = "cpu"):
        model = self.module.load_export(out_dir, device)
        self.rebuilt_mode = pin_decoder_graphs_mode(model, "rebuilt export")
        return model

    def reconstruction_check(self, qat_model, rebuilt, batch: tuple) -> dict:
        return require_exact_reconstruction(self.module.reconstruction_check(qat_model, rebuilt, batch),
                                            f"{self.__name__} export")


# ----------------------------------------------------------------------------- seam functions

def build_context(args, run: Path, device: str = "cuda") -> tuple[BinaryContext, dict]:
    """train.build_context for the binary pilot: same order of loading and seeding, with the
    arm's init, quantizer, ramp and stream."""
    if args.quantizer != _ACTIVE["quantizer"]:
        raise RuntimeError(f"seam installed for quantizer {_ACTIVE['quantizer']!r}, run asks {args.quantizer!r}")
    recipe = train.recipe_of(args)
    base = train.verify_base()
    student = train.load_pretrained("cpu")
    train.seed_everything(args.seed, student)
    resuming = is_resume(run, args)
    quantizer, init = prepare_student(args, student, resuming)
    params = train.configure_trainable(student, recipe)
    params["verified"] = train.verify_recipe(student, recipe, quantizer)
    names = quantizer.quantized_module_names(student)
    if not resuming and not args.probe_steps:
        init["start_signs"] = write_start_signs(run, student, names)
    student.to(device)
    if args.grad_checkpointing:
        train.enable_grad_checkpointing(student.encoder)
    teacher = train.load_teacher(train.load_pretrained(device))
    graphs = {"teacher": pin_decoder_graphs_mode(teacher, "teacher"),
              "student": pin_decoder_graphs_mode(student, "student")}
    evaluator = None if args.probe_steps else DecoderPinnedEvaluator(
        train.DevEvaluator(args.dev_limit, args.eval_batch_size, device))
    ctx = make_context(args, run, student, teacher, open_stream(args), evaluator, quantizer, device)
    ctx.decoder_graphs = graphs
    info = {"base": base, "quantized_modules": names,
            "quantized_modules_sha256": hashlib.sha256(json.dumps(sorted(names)).encode()).hexdigest(),
            "trainable": params, "parameter_accounting": quantizer.parameter_accounting(student),
            "binary": {"init": init, "quantizer": args.quantizer, "ramp_kind": args.ramp_kind,
                       "quantizer_sha256": args.quantizer_sha256, "decoder_graphs": dict(graphs),
                       "stream_workers": binary_paths.STREAM_WORKERS.get(args.train)
                       if args.stream == "hub" else None}}
    return ctx, info


def local_module(name: str):
    """train.local_module with the seam's FinishingExport as "export" and, for the binary arms,
    binary_quant as "quant"."""
    if name == "export":
        return _ACTIVE["export"]
    if name == "quant" and _ACTIVE["quantizer"] == "binary":
        return binary_quant
    return _TRAIN["local_module"](name)


def finish(ctx: BinaryContext, config_sha: str) -> dict:
    """train.finish, plus the arm's "binary" report block (summary.json["binary"])."""
    result = _TRAIN["finish"](ctx, config_sha)  # raises ReconstructionError before scoring
    args = ctx.args
    modes = getattr(ctx.evaluator, "modes", {})
    block = {"init": args.init, "init_sha256": args.init_sha256, "quantizer": args.quantizer,
             "ramp_kind": args.ramp_kind, "quantizer_sha256": args.quantizer_sha256,
             "decoder_graphs": {**ctx.decoder_graphs, "rebuilt": _ACTIVE["export"].rebuilt_mode,
                                "dev-subset": modes.get("dev-subset"), "full-dev": modes.get("full-dev")},
             "timing": dict(ctx.timing)}
    try:  # a reporting failure must not cost the finished run its summary
        names = ctx.quant.quantized_module_names(ctx.student)
        balance = binary_quant.code_balance(ctx.student, names)
        block["code_balance"] = {k: v for k, v in balance.items() if k != "per_module"}
        block["latent_sign_change"] = sign_changes(ctx.run, ctx.student, names)
    except Exception as error:  # noqa: BLE001
        log(f"warning: binary report failed ({type(error).__name__}: {error})")
        block["report_error"] = f"{type(error).__name__}: {error}"
    result["binary"] = block
    return result


@contextlib.contextmanager
def seam(quantizer: str):
    """Install the binary pilot's replacements in train.py and paths.py; restore them on exit."""
    if quantizer not in QUANTIZERS:
        raise ValueError(f"quantizer must be one of {QUANTIZERS}, got {quantizer!r}")
    replacements = [(train, "parse_args", parse_args),
                    (train, "RESUME_KEYS", tuple(_TRAIN["RESUME_KEYS"]) + BINARY_RESUME_KEYS),
                    (train, "TRAINING_SOURCES", tuple(_TRAIN["TRAINING_SOURCES"]) + BINARY_SOURCES),
                    (train, "build_context", build_context),
                    (train, "finish", finish),
                    (train, "POWERLOG_LAUNCH", POWERLOG_LAUNCH),
                    (paths, "RUNS", binary_paths.RUNS),
                    (paths, "POWER", binary_paths.POWER),
                    (paths, "ARTIFACTS", binary_paths.ARTIFACTS),
                    (train, "local_module", local_module)]
    saved = [(obj, name, getattr(obj, name)) for obj, name, _ in replacements]
    active = dict(_ACTIVE)
    export = FinishingExport(binary_export if quantizer == "binary" else _TRAIN["local_module"]("export"))
    try:
        for obj, name, value in replacements:
            setattr(obj, name, value)
        _ACTIVE.update(quantizer=quantizer, export=export)
        yield
    finally:
        for obj, name, value in reversed(saved):
            setattr(obj, name, value)
        _ACTIVE.update(active)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--quantizer", choices=QUANTIZERS)
    known, _ = pre.parse_known_args(argv)
    binary_paths.require_mount()
    with seam(known.quantizer or "binary"):
        train.main(argv)


if __name__ == "__main__":
    main()
