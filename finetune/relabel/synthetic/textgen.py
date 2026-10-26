"""Prompts and text generation (synthetic plan "Text").

Prompts (make_prompts, seeded, written once to runs/<run>/prompts.jsonl):
- the domains take turns (prompt i is domain DOMAINS[i % 4]), so every domain gets the same number of prompts;
- each prompt requests k targeted terms of its domain, k = 1, 2 or 3 with probabilities TERMS_PER_PROMPT, taken
  from a seeded cycle over the domain's targeted terms (reshuffled each pass) so that every targeted term is
  requested about equally often; the terms of one prompt are from different groups;
- registers take turns per domain (conversational, explanatory, dictated notes).

Generation (generate_chunk): the pinned open-weights generator (alias syn-textgen) runs locally through
transformers in bf16, chat template with thinking off, sampling with the model card's non-thinking settings,
one sentence per prompt. Prompts are processed in blocks of BLOCK; each block is written atomically to
runs/<run>/text/<block>.jsonl, so a chunk job resumes at the first missing block and stops after its time budget.
The sampling seed of a batch is a function of (run seed, block, batch), recorded with the outputs.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

from synthetic import layout

TERMS_PER_PROMPT = {1: 0.5, 2: 0.3, 3: 0.2}
BLOCK = 512
BATCH = 64
MAX_NEW_TOKENS = 112
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}   # the card's non-thinking settings
DOMAIN_NAMES = {"chemistry": "chemistry", "mathematics": "mathematics", "finance": "finance",
                "software_engineering": "software engineering"}
REGISTER_TEXT = {
    "conversational": "someone might say casually to a colleague in conversation",
    "explanatory": "a teacher or lecturer might say while explaining an idea",
    "dictated notes": "someone might dictate as a quick note to themselves, in a terse note-taking style",
}
SYSTEM = "You write single sentences for a speech corpus. Follow the instructions exactly and output only the sentence."
USER = """Write one natural English sentence that {register}, in the field of {domain}. It must use {these} naturally: {terms}.
Rules:
- Write it exactly as it would be spoken aloud: words only. No digits, symbols, formulas, equations, code, lists, abbreviations with periods or parentheses.
- Spell out every number and symbol in words, for example "twenty five percent" or "x squared".
- Do not mention medicine, health, patients, drugs or the human body.
- Use between eight and forty words.
Output only the sentence, without quotes."""


def _rng(*parts) -> random.Random:
    return random.Random(int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big"))


def make_prompts(split, n: int, seed: int, run: str) -> list[dict]:
    """n prompts (see the module docstring); deterministic in (split, n, seed, run)."""
    out = []
    cycles = {}
    for d in layout.DOMAINS:
        terms = [e for e in split.targeted(d)]
        if not terms:
            raise RuntimeError(f"no targeted terms in {d}")
        cycles[d] = {"terms": terms, "order": [], "pass": 0, "register": 0, "groups": len({e.group for e in terms})}
    ks, weights = list(TERMS_PER_PROMPT), list(TERMS_PER_PROMPT.values())
    for i in range(n):
        d = layout.DOMAINS[i % len(layout.DOMAINS)]
        c = cycles[d]
        r = _rng(seed, run, "k", i)
        k = r.choices(ks, weights)[0]
        chosen, groups = [], set()
        while len(chosen) < min(k, c["groups"]):
            if not c["order"]:
                c["order"] = list(range(len(c["terms"])))
                _rng(seed, run, d, "pass", c["pass"]).shuffle(c["order"])
                c["pass"] += 1
            e = c["terms"][c["order"].pop()]
            if e.group in groups:
                continue    # a different group per prompt (this occurrence of the term is skipped this pass)
            groups.add(e.group)
            chosen.append(e.term)
        register = layout.REGISTERS[c["register"] % len(layout.REGISTERS)]
        c["register"] += 1
        out.append({"id": f"{run}-{i:06d}", "index": i, "domain": d, "register": register, "requested": chosen})
    return out


def prompt_text(p: dict) -> str:
    terms = ", ".join(f'"{t}"' for t in p["requested"])
    return USER.format(register=REGISTER_TEXT[p["register"]], domain=DOMAIN_NAMES[p["domain"]],
                       these="this term" if len(p["requested"]) == 1 else "these terms", terms=terms)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def blocks(n: int) -> list[tuple[int, int]]:
    return [(s, min(n, s + BLOCK)) for s in range(0, n, BLOCK)]


def ensure_package(name: str) -> bool:
    """Whether `name` is importable, adding the generator's package dir ($RELABEL_ROOT/deps/syn-textgen, pinned
    wheels) last on sys.path when it is not otherwise installed (as relabel_boot does for RELABEL_DEPS)."""
    import importlib
    import sys
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        pass
    d = str(layout.deps_dir(layout.TEXTGEN))
    if d not in sys.path:
        sys.path.append(d)
    importlib.invalidate_caches()
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def ensure_accelerate() -> bool:
    """accelerate importable and, if transformers is already imported, seen by it (transformers caches the answer
    the first time it is asked, so this must run before anything imports transformers)."""
    import sys
    if not (ensure_package("psutil") and ensure_package("accelerate")):
        return False
    if "transformers.utils.import_utils" in sys.modules:
        return bool(sys.modules["transformers.utils.import_utils"].is_accelerate_available())
    return True


def placement(device: str, have_accelerate=ensure_accelerate) -> dict:
    """from_pretrained placement: on the GPU the weights go straight to the device (transformers needs accelerate
    for that). Without it they would be staged in host memory first (about 16 GB in bf16, above the job's
    memory cap), so that is refused."""
    if device == "cpu":
        return {}
    if not have_accelerate():
        raise SystemExit("accelerate is not importable: install the pinned accelerate and psutil wheels into "
                         "$RELABEL_ROOT/deps/syn-textgen (refusing to stage the generator in host memory)")
    return {"device_map": device}


class Generator:
    """The pinned generator under transformers (bf16, left padding, thinking off)."""

    def __init__(self, device: str = "cuda", declared_vram_gb: float | None = None, batch: int = BATCH):
        self.device, self.declared_vram_gb, self.batch = device, declared_vram_gb, batch

    def load(self) -> None:
        place = placement(self.device)     # before transformers is imported (it caches accelerate's presence)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from labeling.engines import vram_fraction
        entry = layout.source(layout.TEXTGEN)
        self.hf_id, self.revision = entry["hf_id"], entry["revision"]
        if self.device == "cuda":
            frac = vram_fraction(self.declared_vram_gb)
            if frac is not None:
                torch.cuda.set_per_process_memory_fraction(frac, 0)
        self.tok = AutoTokenizer.from_pretrained(self.hf_id, revision=self.revision, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(self.hf_id, revision=self.revision, dtype=torch.bfloat16,
                                                          **place)
        self.model.eval()

    def reset_peak(self) -> None:
        """Release cached blocks and restart the peak counters (one probe size at a time)."""
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def peak(self) -> tuple[float, float]:
        """(peak allocated, peak reserved) in GiB since the last reset_peak."""
        import torch
        gib = 2 ** 30
        return torch.cuda.max_memory_allocated() / gib, torch.cuda.max_memory_reserved() / gib

    def chat(self, user: str) -> str:
        return self.tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def generate(self, users: list[str], seed: int) -> list[tuple[str, bool]]:
        import torch
        enc = self.tok([self.chat(u) for u in users], return_tensors="pt", padding=True).to(self.device)
        torch.manual_seed(seed)
        with torch.inference_mode():
            out = self.model.generate(**enc, do_sample=True, max_new_tokens=MAX_NEW_TOKENS,
                                      pad_token_id=self.tok.pad_token_id, **SAMPLING)
        new = out[:, enc["input_ids"].shape[1]:]
        texts = self.tok.batch_decode(new, skip_special_tokens=True)
        eos = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")}
        capped = [not any(int(t) in eos for t in row) for row in new]
        return [(t.strip(), c) for t, c in zip(texts, capped)]

    def info(self) -> dict:
        import torch
        import transformers
        return {"alias": layout.TEXTGEN, "revision": getattr(self, "revision", None), "dtype": "bfloat16",
                "sampling": SAMPLING, "max_new_tokens": MAX_NEW_TOKENS, "batch": self.batch, "enable_thinking": False,
                "transformers": transformers.__version__, "torch": torch.__version__}


def generate_chunk(run_dir: Path, generator, seed: int, budget_minutes: float, log=print) -> dict:
    """Generate the missing text blocks of a run until all are done or the budget is spent."""
    prompts = read_jsonl(run_dir / "prompts.jsonl")
    t0 = time.monotonic()
    todo = [(s, e) for s, e in blocks(len(prompts)) if not (run_dir / "text" / f"{s:07d}.jsonl").exists()]
    done = 0
    if todo:
        generator.load()
    for s, e in todo:
        if (time.monotonic() - t0) / 60 > budget_minutes:
            break
        rows, tb = [], time.monotonic()
        for b in range(s, e, generator.batch):
            batch = prompts[b:min(e, b + generator.batch)]
            bseed = int.from_bytes(hashlib.sha256(f"{seed}|{s}|{b}".encode()).digest()[:4], "big")
            outs = generator.generate([prompt_text(p) for p in batch], bseed)
            for p, (raw, capped) in zip(batch, outs):
                rows.append({**p, "raw": raw, "hit_cap": capped, "seed": bseed})
        info = generator.info()
        for r in rows:
            r["generator"] = {"alias": info["alias"], "revision": info["revision"], "batch": info["batch"]}
        write_jsonl(run_dir / "text" / f"{s:07d}.jsonl", rows)
        done += 1
        log(f"textgen block {s}-{e}: {len(rows)} outputs in {time.monotonic() - tb:.0f} s, peak reserved "
            f"{generator.peak()[1]:.1f} GiB")
    remaining = sum(1 for s, _ in blocks(len(prompts)) if not (run_dir / "text" / f"{s:07d}.jsonl").exists())
    return {"blocks_done": done, "blocks_remaining": remaining}


PROBE_MIN_BATCH = 16
DECLARE_MARGIN_GB = 2.0


def probe_sizes(max_batch: int, min_batch: int = PROBE_MIN_BATCH) -> list[int]:
    """max_batch and its halvings down to min_batch, smallest first."""
    sizes, b = [], max_batch
    while b >= min_batch:
        sizes.append(b)
        b //= 2
    return sorted(sizes) or [max_batch]


def probe(generator, prompts: list[dict], batches: int = 3, seed: int = 0, sizes: list[int] | None = None) -> dict:
    """GPU probe: for each batch size (smallest first), `batches` batches of the longest prompts (most requested
    terms); peak memory and throughput per size, measured separately. A size that runs out of GPU memory (under
    the declared fraction) is recorded as {"oom": true}, and so are the larger sizes, which are not run."""
    import torch
    order = sorted(prompts, key=lambda p: (-len(prompt_text(p)), p["id"]))
    generator.load()
    out, failed = {}, None
    for size in sorted(sizes or [generator.batch]):
        if failed is not None:
            out[size] = {"batch": size, "oom": True, "skipped_after_oom_at": failed}
            continue
        generator.batch = size
        generator.reset_peak()
        t0, n, out_tokens = time.monotonic(), 0, 0
        try:
            for b in range(batches):
                batch = order[b * size:(b + 1) * size]
                outs = generator.generate([prompt_text(p) for p in batch], seed + b)
                n += len(batch)
                out_tokens += sum(len(generator.tok(t)["input_ids"]) for t, _ in outs)
        except torch.cuda.OutOfMemoryError as error:
            failed = size
            out[size] = {"batch": size, "oom": True, "error": str(error).splitlines()[0][:300],
                         "declared_vram_gb": generator.declared_vram_gb}
            del error
            continue
        dt = time.monotonic() - t0
        allocated, reserved = generator.peak()
        out[size] = {"batch": size, "prompts": n, "seconds": dt, "prompts_per_second": n / dt,
                     "output_tokens_per_second": out_tokens / dt, "peak_allocated_gb": allocated,
                     "peak_reserved_gb": reserved, "declared_vram_gb": generator.declared_vram_gb,
                     "max_new_tokens": MAX_NEW_TOKENS}
    return out


def declaration(probes: dict, max_batch: int, fits) -> tuple[int, float]:
    """(batch, declared VRAM GB): the largest probed batch <= max_batch whose declaration, the measured peak
    reserved + DECLARE_MARGIN_GB rounded up to 0.1 GB, satisfies fits(declared) (the job's lease)."""
    import math
    tried = []
    for b in sorted((b for b in probes if b <= max_batch), reverse=True):
        p = probes[b]
        if p.get("oom") or "peak_reserved_gb" not in p:
            tried.append(f"batch {b}: out of memory")
            continue
        declared = math.ceil(round((p["peak_reserved_gb"] + DECLARE_MARGIN_GB) * 10, 6)) / 10
        if fits(declared):
            return b, declared
        tried.append(f"batch {b}: {declared} GB does not fit the lease")
    raise SystemExit("no probed batch size fits: " + ("; ".join(tried) or f"no probe at batch <= {max_batch}"))
