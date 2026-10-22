"""Stage the docs-only update of the Kea-T model card (rajb3/kea-t) that adds the INT8x2 runtime results.
Never uploads by itself.

    ./python hf/stage_card.py stage                                   # from finetune/parakeet-ternary
    ./python hf/stage_card.py upload --approved-bundle-sha256 HEX     # AJ, after review: opens the Hub PR
    ./python hf/stage_card.py verify refs/pr/N
    ./python hf/stage_card.py pre-merge refs/pr/N                     # immediately before AJ merges
    ./python hf/stage_card.py verify main                             # after the merge

The contract is finetune/parakeet-binary/plans/card-update-x2.md; the engine is hf/cardkit.py and the
shared x2 sources hf/x2facts.py. This file is Kea-T's configuration: the pins (main c0f2896, whose four
documents are the card update of plans/card-update.md), the release record of every unchanged file, the
source checks, the checked tables and the computed phrases. The update replaces README.md and RUNTIME.md;
EVALUATION.md and TRAINING.md must render to the live bytes and are not staged.
"""
from __future__ import annotations

import functools
import math
import sys
from pathlib import Path

import paths

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cardkit  # noqa: E402
import x2facts  # noqa: E402
from cardkit import (FileMeta, StageFailed, Table, body, cells, contains, e_notation, front_matter,  # noqa: E402,F401
                     mm, norm, numbers, pct_lower, read_json, row, sha, slug, table)

PKG = HERE.parent
REPO_ID = "rajb3/kea-t"
NAME = REPO_ID.replace("/", "__") + ".card-update-x2"
STAGING = paths.ARTIFACTS / "hf-staging"
# The live Hub state this update is written against (2026-10-22, Hub PR #1 of plans/card-update.md, merged).
PINNED_REVISION = "c0f28965998b0bbeef45cd6e6ab8aaeaf6bd0a79"
OLD_CARD_COMMIT = "b579a94ad5a862e704b50329d294d1c0c88c422e"  # hf/*.md that render to the live documents
LIVE_DOCS = {"README.md": "87a6e5fdf316899cb9b9b1e41ad3ca7a738c6e951320e3f44f02823e376bdd95",
             "RUNTIME.md": "3d993e515ff346db80d99a07bbe17e0f3ab66a772918bf1c9b6f74015bfda0e1",
             "EVALUATION.md": "42a751fb8809450c9db906364c125988417d4c615b52ae1e7ca6afa9ec3dd9ca",
             "TRAINING.md": "78b6beaf59e1ff49dbfd552b4486b8c3be5146870f99106dfe238833087ad50a"}
PREVIOUS_UPDATE = STAGING / "rajb3__kea-t.card-update.stage.json"
PREVIOUS_UPDATE_MAIN = STAGING / "rajb3__kea-t.card-update.verify-main.json"
NOTICE_SHA256 = "a63eeea2b250401da1b54679221d096f9fd188c354817740639c1da4cf4cddc0"  # hf/NOTICE since the rename
NOTICE_BYTES = 3107
GITATTRIBUTES_SHA256 = "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361"  # the Hub default
STAGE_REPORT = STAGING / "rajb3__parakeet-tdt-0.6b-v2-ternary.stage.json"  # the 2026-10-04 full upload
OLD_STAGED_NOTICE = STAGING / "rajb3__parakeet-tdt-0.6b-v2-ternary" / "NOTICE"
# Kea-B, the 1-bit sibling the card links (its published export and test scoring).
KEA_B_REPO = "rajb3/kea-b"
KEA_B_URL = f"https://huggingface.co/{KEA_B_REPO}"
KEA_B_TEST = PKG.parent / "parakeet-binary" / "results" / "test.json"
KEA_B_STAGE_REPORT = paths.ARTIFACTS.parent / "parakeet-binary" / "hf-staging" / "rajb3__kea-b.stage.json"
RUN = paths.RUNS / "main-M1-P2-lr5e-4"
EXPORT = RUN / "export"

TEMPLATES = {"README.md": "MODEL_CARD.md", "RUNTIME.md": "RUNTIME.md", "EVALUATION.md": "EVALUATION.md",
             "TRAINING.md": "TRAINING.md"}
DOCS = tuple(TEMPLATES)
DETAIL_DOCS = DOCS[1:]
STAGED = ("README.md", "RUNTIME.md")
LFS_FILES = ("export.safetensors", "tokenizer/tokenizer.model")
REPORT_UNCHANGED = ("LICENSE", "export.safetensors", "load_ternary.py", "manifest.json", "reconstruction.json",
                    "tokenizer/tokenizer.model", "tokenizer/tokenizer.vocab", "tokenizer/vocab.txt")
GITHUB = cardkit.GITHUB

COLUMNS = [*paths.MEAN_SETS, "mean", "common_voice"]
TEST_MD_HEADER = "| Arm | " + " | ".join(COLUMNS) + " |"
WER_HEADER = ("| Model | LS clean | LS other | AMI | Earnings-22 | GigaSpeech | SPGISpeech | VoxPopuli "
              "| Mean of 7 | Common Voice |")
CARD_WER_LABELS = {"B0": "Original, FP32 (our scoring)", "M1": "**Kea-T (this model)**"}
EVAL_WER_LABELS = {"NVIDIA published": "Original, NVIDIA published", "B0": "Original, FP32, our scoring",
                   "B1": "Same ternary quantizer, no training", "M1": "**This model**",
                   "M1 - B0 (points)": "Difference from our FP32 scoring (points)",
                   "M1 / B0 (relative)": "Ratio to our FP32 scoring"}
RUNTIME_HEADER = ("| Clip | Configuration | Latency (ms) | Process VRAM (GB) | GPU energy (J) "
                  "| Avg GPU power (W) |")
RUNTIME_LABELS = {"packed_expanded": "Ternary expanded — optimized by Wilderness Labs",
                  "packed_optimized": "Ternary compact — optimized by Wilderness Labs",
                  "nemo_bf16_graphs_fused_decoder": "Original BF16 — optimized by Wilderness Labs",
                  "onnx_cuda": "ONNX ASR / ORT CUDA — off the shelf"}
ONNX_JSON_LABEL = "ONNX ASR / ONNX Runtime CUDA — off the shelf"
X2_LABELS = {"x3": "Kea-T expanded, x3 activations", "x2": "**Kea-T expanded, x2 activations (research runtime)**"}
X2_SECTION = "## Research runtime with INT8x2 activations"
X3_SECTION = "## Results"
GLANCE_HEADER = "| | Kea-T | Original |"
GLANCE_SECONDS = 10.0
SET_NAMES = {"librispeech_clean": "LibriSpeech test-clean", "librispeech_other": "LibriSpeech test-other",
             "ami": "AMI", "earnings22": "Earnings-22", "gigaspeech": "GigaSpeech", "spgispeech": "SPGISpeech",
             "voxpopuli": "VoxPopuli", "common_voice": "Common Voice"}
SOURCE_NAMES = {"yodas": "YODAS-Granary English", "peoples_speech": "People's Speech `clean`",
                "librispeech": "LibriSpeech 960 h training sets", "voxpopuli": "VoxPopuli English train",
                "ami": "AMI IHM train"}
WORDS = {4: "four", 5: "five", 8: "eight"}
# Phrases with numbers carried verbatim from the live documents; each must appear in them.
CARRIED = (
    "Parakeet-TDT-0.6B-v2", "parakeet-tdt-0.6b-v2", "{-1, 0, +1}", "(-1, 0, +1)",
    "clamp(round(W / s), -1, 1)", "max(mean|W|, 1e-8)", "(E_student - E_teacher)^2",
    "more than 50% normalized WER", "(betas 0.9, 0.98, weight decay 0.01 on matrices)", "gradient clipping 1.0",
    "LibriSpeech 960 h", "SHA-256", "dither 0", "segments of about 30 s or less", "up to about 106 s",
    "can differ by 0.01", "medians of three 5 s windows", "2-bit codes", "kernel-1 convolutions",
    "CC-BY-4.0", "CC-BY-3.0", "CC0-1.0", "(2.0, 2.5, 3.0 or 4.0)", "25 European Languages", "macOS 26",
    "NVIDIA RTX 5090 (CUDA, strict FP32)",
)
# Numbers of the live documents deliberately absent from the new documents (none).
DROPPED: tuple[str, ...] = ()

HUB_EXTRA_CHECKS = frozenset({
    "hub NOTICE differs from the stage report's NOTICE only in its first line",
    "hub rajb3/kea-b is public; export.safetensors LFS SHA-256 and size == Kea-B stage report",
})
SOURCE_CHECKS = frozenset({
    "original stage report PASSED",
    "previous card update PASSED and verified at the pin",
    "results/test.json export SHA-256 == stage report",
    "rtx5090.json export SHA-256 == stage report",
    "run manifest.json and reconstruction.json SHA-256 == stage report",
    "run summary.json reconstruction and export_bytes == export",
    "pilot selection is the run's recipe and learning rate",
    "Kea-B stage report PASSED and its export SHA-256 == Kea-B test.json",
    "Kea-B test.json Kea-T export SHA-256 == stage report",
    "results/TEST.md rows == results/test.json rounded",
    "inference/README.md results table == rtx5090.json rounded",
    *x2facts.CHECKS,
})
TABLE_CHECKS = frozenset({
    "card at-a-glance table == sources",
    "card WER table == TEST.md B0 and M1 rows",
    "EVALUATION.md WER table == all 6 TEST.md rows",
    "RUNTIME.md results table == inference/README.md table",
    "RUNTIME.md x2 results table == sources",
    "RUNTIME.md paired table == report-dev",
})
CARD_YAML_CHECK = "card YAML: model_name kea-t, license cc-by-4.0, base_model, library_name nemo"
COMMIT_MESSAGE = ("Model card and RUNTIME.md: add the INT8x2 research runtime's results")
UPLOAD_COMMAND = ("./python hf/stage_card.py upload --approved-bundle-sha256 <the bundle_sha256 AJ "
                  "approved>   # from finetune/parakeet-ternary")


# ----------------------------------------------------------------------------- inputs

def load_sources() -> dict:
    """Every source a claim is checked against (plans/card-update.md section 6, plans/card-update-x2.md
    section 6)."""
    return {
        "test": read_json(PKG / "results" / "test.json"),
        "test_md": (PKG / "results" / "TEST.md").read_text(),
        "rtx": read_json(PKG / "inference" / "results" / "rtx5090.json"),
        "inference_md": (PKG / "inference" / "README.md").read_text(),
        "manifest": read_json(EXPORT / "manifest.json"),
        "manifest_sha256": sha((EXPORT / "manifest.json").read_bytes()),
        "reconstruction": read_json(EXPORT / "reconstruction.json"),
        "reconstruction_sha256": sha((EXPORT / "reconstruction.json").read_bytes()),
        "summary": read_json(RUN / "summary.json"),
        "config": read_json(RUN / "config.json"),
        "pilot": read_json(paths.RUNS / "pilot-selection.json"),
        "b0_dev": read_json(paths.EVAL / "B0-dev" / "summary.json"),
        "stage_report": read_json(STAGE_REPORT),
        "previous_update": read_json(PREVIOUS_UPDATE),
        "previous_update_main": read_json(PREVIOUS_UPDATE_MAIN),
        "old_notice": OLD_STAGED_NOTICE.read_bytes(),
        "hf_notice": (HERE / "NOTICE").read_bytes(),
        "kea_b_test": read_json(KEA_B_TEST),
        "kea_b_report": read_json(KEA_B_STAGE_REPORT),
        "x2": x2facts.load(paths.REPO),
    }


def load_templates() -> dict[str, str]:
    return {rel: (HERE / name).read_text() for rel, name in TEMPLATES.items()}


def origin_main_files(repo: Path) -> set[str]:
    return cardkit.origin_main_files(repo)


def code_histogram(manifest: dict) -> str:
    hist = manifest["code_histogram"]
    return (f"{100 * hist['minus_one']:.1f}% -1, {100 * hist['zero']:.1f}% 0, "
            f"{100 * hist['plus_one']:.1f}% +1")


def render(templates: dict[str, str], sources: dict) -> dict[str, str]:
    values = {"{repo_id}": REPO_ID, "{repo_name}": REPO_ID.split("/")[1],
              "{code_histogram}": code_histogram(sources["manifest"])}
    out = {}
    for rel, text in templates.items():
        for key, value in values.items():
            text = text.replace(key, value)
        out[rel] = text
    return out


# ----------------------------------------------------------------------------- A. Hub (Kea-T's own)

def hub_extra(hub, sources: dict, files: dict, checks: dict, info: dict) -> None:
    report = sources["stage_report"]["files"]
    notice, old = cardkit.small(hub, files, "NOTICE", PINNED_REVISION), sources["old_notice"]
    new_lines, old_lines = notice.split(b"\n"), old.split(b"\n")
    checks["hub NOTICE differs from the stage report's NOTICE only in its first line"] = (
        sha(old) == report["NOTICE"]["sha256"] and sha(sources["hf_notice"]) == NOTICE_SHA256
        and new_lines[1:] == old_lines[1:] and new_lines[0] != old_lines[0])
    kea_b = hub.public_files(KEA_B_REPO)
    want = sources["kea_b_report"]["files"]["export.safetensors"]
    got = (kea_b or {}).get("export.safetensors")
    checks["hub rajb3/kea-b is public; export.safetensors LFS SHA-256 and size == Kea-B stage report"] = (
        got is not None and got.lfs_sha256 == want["sha256"] and got.size == want["bytes"])
    info["hub"]["kea_b_export"] = got._asdict() if got is not None else None


# ----------------------------------------------------------------------------- B. sources

def expected_test_md(test: dict) -> dict[str, list[str]]:
    rows, pub = test["rows"], test["published"]
    out = {"NVIDIA published": [f"{pub[c]:.2f}" if c in pub else "n/a" for c in COLUMNS]}
    for arm in ("B0", "B1", "M1"):
        out[arm] = [f"{rows[arm][c]:.2f}" for c in COLUMNS]
    out["M1 - B0 (points)"] = [f"{rows['M1'][c] - rows['B0'][c]:+.2f}" for c in COLUMNS]
    out["M1 / B0 (relative)"] = [f"{rows['M1'][c] / rows['B0'][c]:.2f}x" for c in COLUMNS]
    return out


def test_md_rows(test_md: str) -> dict[str, list[str]] | None:
    rows = table(test_md, TEST_MD_HEADER)
    return None if rows is None else {cells(line)[0]: cells(line)[1:] for line in rows}


def workload(rtx: dict, key: str, seconds: float) -> dict:
    (w,) = [w for w in rtx["variants"][key]["workloads"] if w["audio_seconds"] == seconds]
    return w


def lengths(rtx: dict) -> list[float]:
    return sorted({w["audio_seconds"] for v in rtx["variants"].values() for w in v["workloads"]})


def runtime_rows(rtx: dict) -> list[str]:
    out = []
    for s in lengths(rtx):
        for key, label in RUNTIME_LABELS.items():
            w = workload(rtx, key, s)
            out.append(row(f"{s:g} s", [label, f"{w['median_ms']:.2f}", f"{w['process_vram_gb']:.2f}",
                                        f"{w['joules_per_iteration']:.2f}", f"{w['average_watts']:.0f}"]))
    return out


def check_sources(sources: dict, checks: dict, info: dict) -> None:
    rep, rtx, s, m = sources["stage_report"], sources["rtx"], sources["summary"], sources["manifest"]
    files = rep["files"]
    export_sha = files["export.safetensors"]["sha256"]
    checks["original stage report PASSED"] = (
        rep.get("status") == "PASSED" and bool(rep.get("checks")) and all(v is True for v in rep["checks"].values())
        and rep.get("checks_passed") == f"{len(rep['checks'])}/{len(rep['checks'])}")
    prev, prev_main = sources["previous_update"], sources["previous_update_main"]
    checks["previous card update PASSED and verified at the pin"] = (
        prev.get("status") == "PASSED" and {k: v["sha256"] for k, v in prev.get("files", {}).items()} == LIVE_DOCS
        and prev_main.get("status") == "PASSED"
        and prev_main.get("verify", {}).get("revision_sha") == PINNED_REVISION)
    checks["results/test.json export SHA-256 == stage report"] = sources["test"].get("m1_export_sha256") == export_sha
    checks["rtx5090.json export SHA-256 == stage report"] = (
        rtx["model_provenance"].get("ternary_export_sha256") == export_sha)
    checks["run manifest.json and reconstruction.json SHA-256 == stage report"] = (
        sources["manifest_sha256"] == files["manifest.json"]["sha256"] and m["sha256"] == export_sha
        and sources["reconstruction_sha256"] == files["reconstruction.json"]["sha256"])
    checks["run summary.json reconstruction and export_bytes == export"] = (
        s["reconstruction"] == sources["reconstruction"] and s["export_bytes"] == m["bytes"]
        and s["run_name"] == m["extra"]["run_name"] and s["config_sha256"] == m["extra"]["config_sha256"]
        and sources["config"]["args"]["run_name"] == s["run_name"])
    pilot = sources["pilot"]
    checks["pilot selection is the run's recipe and learning rate"] = (
        pilot["recipe"] == s["recipe"] and float(pilot["lr"]) == s["lr"])
    kb_report, kb_test = sources["kea_b_report"], sources["kea_b_test"]
    checks["Kea-B stage report PASSED and its export SHA-256 == Kea-B test.json"] = (
        kb_report.get("status") == "PASSED" and kb_report.get("repo_id") == KEA_B_REPO
        and bool(kb_report.get("checks")) and all(v is True for v in kb_report["checks"].values())
        and kb_report["files"]["export.safetensors"]["sha256"] == kb_test.get("mb1_weights_sha256"))
    checks["Kea-B test.json Kea-T export SHA-256 == stage report"] = kb_test.get("m1_export_sha256") == export_sha
    checks["results/TEST.md rows == results/test.json rounded"] = (
        test_md_rows(sources["test_md"]) == expected_test_md(sources["test"]))
    variants = rtx["variants"]
    checks["inference/README.md results table == rtx5090.json rounded"] = (
        set(variants) == set(RUNTIME_LABELS)
        and all(variants[k]["label"] == RUNTIME_LABELS[k] for k in RUNTIME_LABELS if k != "onnx_cuda")
        and variants["onnx_cuda"]["label"] == ONNX_JSON_LABEL
        and all(v["gpu"]["power_limit_w"] == rtx["power_limit_w"] for v in variants.values())
        and table(sources["inference_md"], RUNTIME_HEADER) == runtime_rows(rtx))
    x2facts.check(sources["x2"], checks, info)
    # x2's published export is Kea-T's (the release record), not only the constant in x2facts
    checks["x2 arms are the published exports and the gated identities"] = (
        checks["x2 arms are the published exports and the gated identities"]
        and x2facts.EXPORT_SHA["M1"] == export_sha)


def check_yaml(data) -> bool:
    return (data.model_name == REPO_ID.split("/")[1] and data.license == "cc-by-4.0"
            and data.base_model == paths.MODEL_ID and data.library_name == "nemo")


# ----------------------------------------------------------------------------- D. numbers

def glance_rows(sources: dict) -> list[str]:
    """The at-a-glance table: the published, runnable runtime (expanded x3, rtx5090.json) against our BF16
    build at 10 s, as on the live card (the x2 research runtime is one sentence below it)."""
    t, rtx = sources["test"], sources["rtx"]
    ex, bf = workload(rtx, "packed_expanded", GLANCE_SECONDS), workload(rtx, "nemo_bf16_graphs_fused_decoder",
                                                                        GLANCE_SECONDS)
    g = f"{GLANCE_SECONDS:g} s clip"
    return [
        row("Weight file", [f"**{t['m1_export_mb']:.1f} MB**", f"{t['original_nemo_mb']:,.0f} MB"]),
        row(f"Test WER, mean of {len(paths.MEAN_SETS)} sets",
            [f"{t['rows']['M1']['mean']:.2f}%", f"{t['rows']['B0']['mean']:.2f}%"]),
        row(f"Latency, {g}", [f"**{ex['median_ms']:.2f} ms** ({pct_lower(ex['median_ms'], bf['median_ms']):.1f}% lower)",
                              f"{bf['median_ms']:.2f} ms"]),
        row(f"GPU energy, {g}", [f"**{ex['joules_per_iteration']:.2f} J** "
                                 f"({pct_lower(ex['joules_per_iteration'], bf['joules_per_iteration']):.1f}% less)",
                                 f"{bf['joules_per_iteration']:.2f} J"]),
        row(f"Process VRAM, {g}", [f"**{ex['process_vram_gb']:.2f} GB** "
                                   f"({pct_lower(ex['process_vram_gb'], bf['process_vram_gb']):.1f}% lower)",
                                   f"{bf['process_vram_gb']:.2f} GB"]),
    ]


def tables(docs: dict[str, str], sources: dict) -> list[Table]:
    md = test_md_rows(sources["test_md"]) or {}
    x2 = sources["x2"]
    x2_rows = x2facts.runtime_rows(x2, "M1", X2_LABELS)
    return [
        Table("card at-a-glance table == sources", "README.md", GLANCE_HEADER, glance_rows(sources)),
        Table("card WER table == TEST.md B0 and M1 rows", "README.md", WER_HEADER,
              [row(label, md.get(arm, [])) for arm, label in CARD_WER_LABELS.items()]),
        Table("EVALUATION.md WER table == all 6 TEST.md rows", "EVALUATION.md", WER_HEADER,
              [row(label, md[arm]) for arm, label in EVAL_WER_LABELS.items()] if list(md) == list(EVAL_WER_LABELS)
              else None),
        Table("RUNTIME.md results table == inference/README.md table", "RUNTIME.md", RUNTIME_HEADER,
              table(sources["inference_md"], RUNTIME_HEADER), section=X3_SECTION),
        Table("RUNTIME.md x2 results table == sources", "RUNTIME.md", x2facts.TABLE_HEADER, x2_rows,
              section=X2_SECTION),
        Table("RUNTIME.md paired table == report-dev", "RUNTIME.md", x2facts.PAIRED_HEADER,
              x2facts.paired_rows(x2, "M1"), section=X2_SECTION),
    ]


def published_facts(sources: dict) -> dict[str, tuple[tuple[str, ...], str | None]]:
    """The phrases of the published documents: {label: (documents that must contain it, phrase computed
    from its source, None if the source contradicts the claim)} (plans/card-update.md section 6)."""
    t, rtx, m, rec = sources["test"], sources["rtx"], sources["manifest"], sources["reconstruction"]
    s, cfg, pilot = sources["summary"], sources["config"], sources["pilot"]
    rows, pub, empty = t["rows"], t["published"], t["empty_hypothesis_fraction"]
    card, run_md, eval_md, train_md = DOCS
    acc, b = m["parameter_accounting"], m["bytes"]
    layers = dict(m["quantized_layers"])
    n_layers = len({k.split(".")[2] for k in layers})
    n_layers = n_layers if n_layers == m["config"]["encoder"]["n_layers"] else None
    gpu = rtx["hardware"].removeprefix("NVIDIA ")
    train_gpu = cfg["versions"]["gpu"].removeprefix("NVIDIA GeForce ")
    pl = rtx["power_limit_w"]
    (default_pl,) = {v["gpu"]["default_power_limit_w"] for v in rtx["variants"].values()}
    short, mid, long_ = lengths(rtx)
    w = {k: {x: workload(rtx, k, x) for x in (short, mid, long_)} for k in RUNTIME_LABELS}
    ex, co, bf, on = (w[k] for k in RUNTIME_LABELS)
    metrics = ("median_ms", "process_vram_gb", "joules_per_iteration")
    lo_s, hi_s = cfg["bucketing"]["prefilter_seconds"]
    steps, ramp, warm = s["steps"], s["ramp_steps"], s["warmup_steps"]
    lr = e_notation(s["lr"], 0)
    hours = s["last_session_train_time_s"] / 3600 if s["resumes"] == 0 else None
    total_h = s["audio_hours_seen"]
    by_source = {k: v / 3600 for k, v in s["seconds_by_source"].items()}
    diff = {c: rows["M1"][c] - rows["B0"][c] for c in COLUMNS if c != "mean"}
    top = sorted(diff, key=diff.get, reverse=True)[:3]
    gap = (f"The gap to the original is largest on {SET_NAMES[top[0]]} ({diff[top[0]]:+.2f} points), which is "
           f"outside the training mixture, followed by {SET_NAMES[top[1]]} ({diff[top[1]]:+.2f}) and "
           f"{SET_NAMES[top[2]]} ({diff[top[2]]:+.2f}).") if top[0] == "common_voice" else None
    p1, p2, p3 = (next(r for r in pilot["recipe_stage"] if r["arm"] == a) for a in ("P1", "P2", "P3"))
    pilot_steps = {r["selected_step"] for r in pilot["recipe_stage"] + pilot["lr_stage"]}
    lr_runs = sorted(pilot["lr_stage"], key=lambda r: r["lr"])
    best = min(lr_runs, key=lambda r: r["dev_mean_wer"])
    validation = rtx["runtime_validation"]
    env = rtx["publication_validation"]["environment"]
    cu = env["torch"].split("+cu")[1]
    nvrtc = next(v for k, v in env.items() if k.startswith("nvidia-cuda-nvrtc"))
    onnx = rtx["model_provenance"]
    repro = max(abs(rows["B0"][c] - pub[c]) for c in paths.MEAN_SETS)
    ex_beats = all(ex[x][k] < bf[x][k] for x in (short, mid, long_) for k in metrics)
    co_least = all(co[x]["process_vram_gb"] == min(w[k][x]["process_vram_gb"] for k in w) for x in (short, mid, long_))
    co_energy = pct_lower(co[mid]["joules_per_iteration"], bf[mid]["joules_per_iteration"])
    co_long_higher = (co[long_]["median_ms"] > bf[long_]["median_ms"]
                      and co[long_]["joules_per_iteration"] > bf[long_]["joules_per_iteration"])
    hist_ok = rec["codes_exact"] is True and rec["scales_exact"] is True and rec["greedy_hyps_equal"] is True
    rows_out = sum(shape["shape"][0] for shape in layers.values())
    kb_mb = sources["kea_b_report"]["files"]["export.safetensors"]["bytes"] / 1e6
    kb_mean = sources["kea_b_test"]["rows"]["MB1"]["mean"]

    def ok(cond: bool, phrase: str) -> str | None:
        return phrase if cond else None

    return {
        # card
        "Kea-B sibling": ((card,), ok(f"{kb_mb:.1f}" == f"{sources['kea_b_test']['export_mb']['MB1']:.1f}",
                                      f"Its 1-bit sibling Kea-B is {kb_mb:.1f} MB with a {kb_mean:.2f}% test mean.")),
        "ternary share of parameters": ((card,), f"{100 * acc['ternary_parameters'] / acc['total_parameters']:.0f}% "
                                                 "of the model's parameters"),
        "encoder layers": ((card,), ok(n_layers is not None, f"{n_layers}-layer FastConformer encoder")),
        "speed setting": ((card, run_md), f"one {gpu} at a {pl} W power limit"),
        "compact VRAM vs BF16 (card)": ((card,), f"mode uses {co[mid]['process_vram_gb']:.2f} GB of process VRAM, "
                                                 f"{pct_lower(co[mid]['process_vram_gb'], bf[mid]['process_vram_gb']):.1f}% "
                                                 "less than that build"),
        "other clip lengths": ((card, run_md), f"the {short:g} s and {long_:g} s clips"),
        "reproduction of NVIDIA's numbers": ((card, eval_md), f"within {repro:.2f} points per set"),
        "ternary matrices": ((card,), f"{len(layers)} ternary matrices"),
        "ramp": ((card, train_md), f"first {100 * ramp / steps:.0f}% of steps ({ramp:,})"),
        "steps and batch audio": ((card, train_md), f"{steps:,} steps of batches of up to {s['batch_seconds']:.0f} s of audio"),
        "peak learning rate (card)": ((card,), f"peak learning rate of {lr}"),
        "audio hours": ((card, train_md), f"{total_h:,.0f} hours of audio"),
        "training time (card)": ((card,), ok(hours is not None, f"{hours:.1f} hours on one {train_gpu}")),
        "trained clip lengths": ((card, eval_md), f"trained on {lo_s:g} to {hi_s:g} s utterances"),
        "export bytes": ((card, train_md), f"({b['file_bytes']:,} bytes)"),
        "largest gaps": ((card, eval_md), gap),
        "runtime setting (limitations)": ((card, run_md), f"one {gpu} at a {pl} W limit"),
        "loader versions": ((card, train_md), f"NeMo {m['versions']['nemo']} and torch {m['versions']['torch']}"),
        # RUNTIME.md
        "expanded INT8 copy": ((run_md,), f"({acc['ternary_parameters']:,} bytes, about "
                                          f"{acc['ternary_parameters'] / 1e6:.0f} MB)"),
        "expanded vs BF16": ((run_md,), f"at {mid:g} s, it has {pct_lower(ex[mid]['median_ms'], bf[mid]['median_ms']):.1f}% "
                                        f"lower latency, {pct_lower(ex[mid]['process_vram_gb'], bf[mid]['process_vram_gb']):.1f}% "
                                        f"lower process VRAM and "
                                        f"{pct_lower(ex[mid]['joules_per_iteration'], bf[mid]['joules_per_iteration']):.1f}% "
                                        "less GPU energy"),
        "expanded lower at every length": ((run_md,), ok(ex_beats, f"It is also lower on all three at the {short:g} s "
                                                                   f"and {long_:g} s clips")),
        "compact least VRAM": ((run_md,), ok(co_least, "It used the least process VRAM of the four configurations "
                                                       "at every length tested")),
        "compact vs BF16": ((run_md,), f"{pct_lower(co[mid]['process_vram_gb'], bf[mid]['process_vram_gb']):.1f}% less "
                                       f"than BF16 at {mid:g} s, with "
                                       f"{pct_lower(co[mid]['median_ms'], bf[mid]['median_ms']):.1f}% lower latency"),
        "compact energy within": ((run_md,), f"Its {mid:g} s energy is within {math.ceil(abs(co_energy))}% of BF16"),
        "compact longest clip": ((run_md,), ok(co_long_higher, f"at {long_:g} s its latency and energy are higher "
                                                               "than BF16's")),
        "ONNX packages": ((run_md,), f"{onnx['onnx_package'].replace('==', ' ')} package with "
                                     f"{onnx['onnx_runtime'].replace('==', ' ')}"),
        "BF16 vs ONNX": ((run_md,), f"At {mid:g} s, our BF16 build also has "
                                    f"{pct_lower(bf[mid]['median_ms'], on[mid]['median_ms']):.1f}% lower latency "
                                    "than this row"),
        "power and energy at 10 s": ((run_md,), ok(
            all(x[mid]["average_watts"] >= bf[mid]["average_watts"]
                and x[mid]["joules_per_iteration"] < bf[mid]["joules_per_iteration"]
                and x[mid]["median_ms"] < bf[mid]["median_ms"] for x in (ex, co)),
            f"At {mid:g} s, compared with our BF16 build, the ternary rows draw as much power or more "
            f"(expanded {ex[mid]['average_watts']:.0f} W, compact {co[mid]['average_watts']:.0f} W, "
            f"BF16 {bf[mid]['average_watts']:.0f} W) and use less energy because they finish sooner")),
        "length order": ((run_md,), "lengths in order (" + ", ".join(f"{x:g}" for x in (short, mid, long_)) + " s)"),
        "power limit": ((run_md,), f"configured {pl} W limit (board default {default_pl:.0f} W)"),
        "mode agreement": ((run_md,), ok(validation["text"] == validation["tokens"] == validation["timestamps"]
                                         == validation["utterances"],
                                         f"{validation['utterances']:,} utterances from "
                                         f"{WORDS[len(validation['per_set']['candidate'])]} test sets")),
        "runtime environment": ((run_md,), f"NeMo {mm(env['nemo_toolkit'])}, PyTorch {mm(env['torch'])} with CUDA "
                                           f"{cu[:-1]}.{cu[-1]}, Triton {mm(env['triton'])}, NVRTC {mm(nvrtc)}"),
        # EVALUATION.md
        "mean of 7": ((eval_md,), f"Mean of {len(paths.MEAN_SETS)}"),
        "TED-LIUM published": ((eval_md,), f"TED-LIUM result, {pub['tedlium']:.2f}, is excluded"),
        "Common Voice gap": ((eval_md,), f"largest gap ({diff['common_voice']:+.2f} points)"),
        "AMI empty outputs": ((eval_md,), ok(empty["M1"]["ami"] < empty["B0"]["ami"],
                                             f"{100 * empty['M1']['ami']:.2f}% of this model's AMI test hypotheses are "
                                             f"empty, fewer than the FP32 original's {100 * empty['B0']['ami']:.2f}%")),
        # TRAINING.md
        "base revision": ((train_md,), f"`{m['base_model']['revision']}`"),
        "encoder matching weight": ((train_md,), f"valid frames, weight {cfg['encoder_matching_weight']:.1f}"),
        "clip filter": ((train_md,), f"utterances outside {lo_s:g} to {hi_s:g} s"),
        "peak learning rate and warmup": ((train_md,), f"peak learning rate {lr} with {100 * warm / steps:.0f}% linear "
                                                       "warmup"),
        "seed": ((train_md,), f"seed {s['seed']}"),
        "training time": ((train_md,), ok(hours is not None, f"One {train_gpu}, {hours:.1f} hours of training "
                                                             f"(about {hours / 24:.1f} days)")),
        **{f"hours of {k}": ((train_md,), f"{SOURCE_NAMES[k]} ({h:,.0f} h, {100 * h / total_h:.1f}%)")
           for k, h in by_source.items()},
        "rounded hours sum": ((train_md,), f"sum to {sum(round(h) for h in by_source.values()):,}"),
        "unrounded hours": ((train_md,), f"unrounded total is {total_h:,.1f} h"),
        "pilot steps": ((train_md,), ok(len(pilot_steps) == 1, f"Pilot ({min(pilot_steps):,} steps per arm")),
        "pilot encoder matching": ((train_md,), ok(all(p2["dev_wer"][k] < p1["dev_wer"][k] for k in p1["dev_wer"]),
                                                   f"{100 * p1['dev_mean_wer']:.2f}% to {100 * p2['dev_mean_wer']:.2f}% "
                                                   f"and helped on all {WORDS[len(p1['dev_wer'])]} sets")),
        "pilot trainable decoder": ((train_md,), f"changed it by {100 * abs(p3['dev_mean_wer'] - p2['dev_mean_wer']):.3f} "
                                                 "points"),
        "pilot learning rates": ((train_md,), ok(e_notation(best["lr"], 0) == lr,
                                                 "Of learning rates " + ", ".join(e_notation(r["lr"], 0) for r in lr_runs[:-1])
                                                 + f" and {e_notation(lr_runs[-1]['lr'], 0)}, {lr} was best")),
        "development WER": ((train_md,), f"development mean WER is {100 * s['dev_mean_wer']:.2f}% (FP32 original "
                                         f"{sources['b0_dev']['dev_mean_wer']:.2f}%)"),
        "ternary layers and parameters": ((train_md,), f"{len(layers)} ternary weight matrices, "
                                                       f"{acc['ternary_parameters']:,} parameters"),
        "layers (files)": ((train_md,), ok(n_layers is not None, f"each of the {n_layers} encoder layers")),
        "packed code bytes": ((train_md,), f"({b['packed_code_bytes']:,} bytes)"),
        "scales": ((train_md,), ok(rows_out * 4 == b["scale_bytes"], f"({rows_out:,} rows, {b['scale_bytes']:,} bytes)")),
        "float parameters": ((train_md,), f"{acc['float_parameters']:,} parameters"),
        "FP16 bytes": ((train_md,), f"({b['fp16_float_bytes']:,} bytes)"),
        "FP32 constants": ((train_md,), f"`.fb`, {b['fp32_feature_constant_bytes']:,} bytes)"),
        "code histogram": ((train_md,), code_histogram(m)),
        "reconstruction layers": ((train_md,), ok(hist_ok and rec["checked_layers"] == len(layers),
                                                  f"exact for all {rec['checked_layers']} layers")),
        "reconstruction clips": ((train_md,), f"on {WORDS[len(rec['utterance_ids'])]} LibriSpeech dev-clean clips"),
        "reconstruction difference": ((train_md,), f"about {e_notation(rec['encoder_max_abs_diff'], 1)}"),
        "tokenizer size": ((train_md,), f"({m['config']['decoder']['vocab_size']:,} tokens)"),
    }


def facts(sources: dict) -> dict[str, tuple[tuple[str, ...], str | None]]:
    """Every computed phrase: the published documents' phrases, the shared x2 phrases (RUNTIME.md's
    research-runtime section) and the card's one research-runtime sentence and dev-WER sentence."""
    card, run_md = "README.md", "RUNTIME.md"
    x2 = sources["x2"]
    shared = x2facts.facts(x2, "M1", card, run_md, "x2")
    shared["x2 cross-session agreement"] = ((run_md,), shared["x2 cross-session agreement"][1])  # not on the card
    del shared["x2 dev WER (card)"]  # the card states both runtimes (below)
    wer = x2facts.dev_wer(x2, "M1")
    return {
        **published_facts(sources),
        **shared,
        "x2 research runtime (card)": ((card,), x2facts.research_glance_phrase(x2, "M1")),
        "dev WER of both runtimes (card)": ((card,), x2facts.ok(
            f"{wer['x2']:.2f}" == f"{wer['x3']:.2f}" == f"{wer['fp32']:.2f}",
            f"both GPU runtimes give the same mean WER as FP32 decoding ({wer['fp32']:.2f}%)")),
    }


# ----------------------------------------------------------------------------- configuration and CLI

@functools.lru_cache(maxsize=None)
def config() -> cardkit.Config:
    report = read_json(STAGE_REPORT)["files"]
    release = {rel: report[rel] for rel in REPORT_UNCHANGED}
    release["NOTICE"] = {"sha256": NOTICE_SHA256, "bytes": NOTICE_BYTES}
    return cardkit.Config(
        repo_id=REPO_ID, staging=STAGING, tmp_root=paths.ARTIFACTS / "tmp", pinned_revision=PINNED_REVISION,
        docs=DOCS, staged=STAGED, live_docs=dict(LIVE_DOCS), release=release, lfs_files=LFS_FILES,
        gitattributes_sha256=GITATTRIBUTES_SHA256, allowed_urls=frozenset({KEA_B_URL}), carried=CARRIED,
        dropped=DROPPED, commit_message=COMMIT_MESSAGE, upload_command=UPLOAD_COMMAND,
        card_yaml_check=CARD_YAML_CHECK, hub_extra_checks=HUB_EXTRA_CHECKS, source_checks=SOURCE_CHECKS,
        table_checks=TABLE_CHECKS, check_hub_extra=hub_extra, check_sources=check_sources, check_yaml=check_yaml,
        tables=tables, facts=facts)


def required_checks() -> set[str]:
    return cardkit.required_checks(config())


def staging_checks() -> set[str]:
    return cardkit.staging_checks(config())


def upload_checks() -> set[str]:
    return cardkit.upload_checks()


def verify_checks(revision: str) -> set[str]:
    return cardkit.verify_checks(config(), revision)


def pre_merge_checks() -> set[str]:
    return cardkit.pre_merge_checks()


def prepare(hub, main_files: set[str], sources: dict, templates: dict[str, str], checks: dict, info: dict) -> dict:
    """Record every check before staging; return the staged documents."""
    return cardkit.prepare(config(), hub, main_files, sources, render(templates, sources), checks, info)


def stage(staging: Path, prepare_fn) -> tuple[bool, dict, Path]:
    return cardkit.stage(config(), staging, prepare_fn)


def upload(hub, folder: Path, report: dict, approved_bundle_sha256: str, checks: dict, info: dict) -> dict | None:
    return cardkit.upload(config(), hub, folder, report, approved_bundle_sha256, checks, info)


def verify(hub, revision: str, report: dict, checks: dict, info: dict, pr_head: str | None = None) -> None:
    cardkit.verify(config(), hub, revision, report, checks, info, pr_head=pr_head)


def pre_merge(hub, revision: str, verify_report: dict, checks: dict, info: dict) -> None:
    cardkit.pre_merge(config(), hub, revision, verify_report, checks, info)


def upload_command() -> str:
    return UPLOAD_COMMAND


def main(argv: list[str] | None = None) -> None:
    paths.require_mount()

    def prepare_stage(hub, main_files, checks, info):
        return prepare(hub, main_files, load_sources(), load_templates(), checks, info)

    cardkit.main(config(), prepare_stage, paths.REPO, argv, doc=__doc__)


if __name__ == "__main__":
    main()
