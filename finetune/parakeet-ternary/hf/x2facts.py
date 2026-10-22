"""The INT8x2 GPU runtime results shared by the Kea-B and Kea-T card updates: sources, their checks and
the rows and phrases computed from them (finetune/parakeet-binary/plans/card-update-x2.md, sections 3, 5B
and 6). CPU only; reads JSON files and the local origin/main ref.

Arms (roster x2b-20261021): MB1 x2 = b1 (MB1-B-hybrid-x2), MB1 x3 = b2; M1 x2 = b4 (M1-x2, the expanded
runtime with two parts), M1 x3 = b5 (M1-x3, the expanded runtime Kea-T published), and b6
(M1-R-hybrid-x3) only as a cross-session reference. Per-clip latency, energy and power are the median of
the two processes (clips1, clips2) in report-test.json; process VRAM is the warm windows' sampled peak
(process_sampled_peak_bytes, Kea-T's published definition), the larger of the two processes.
"""
from __future__ import annotations

import hashlib
import json
import os
import math
import statistics
import subprocess
import sys
from pathlib import Path

H = Path("/mnt/hd/wilderness-labs-stt/parakeet-binary/gpu1bit/H")
COMPARISON = Path("/mnt/hd/wilderness-labs-stt/parakeet-ternary/comparison")
REPORT_DEV = H / "x2" / "bench" / "x2b-20261021" / "report-dev.json"
REPORT_TEST = H / "x2" / "bench" / "x2b-20261021" / "report-test.json"
GATES = H / "x2" / "gates-x2g-20261021b.json"
GATED = {"g1": "MB1-B-hybrid-x2", "g3": "MB1-B-hybrid-x3", "g4": "M1-x2", "g5": "M1-x3"}
SUMMARIES = {arm: H / "fulldev" / "x2g-20261021b" / arm / "summary.json" for arm in GATED.values()}
IDENTITIES = {arm: H / "fulldev" / "x2g-20261021b" / arm / "identity.json" for arm in GATED.values()}
WRAPPERS = ("finetune/parakeet-ternary/python", "finetune/parakeet-binary/python")  # relative to the repository
# The wrappers as the gated arms hashed them (before main's opt-in PARAKEET_PYTHON override, 101b314); both
# exec the repository's .training-runtime/bin/python3.
WRAPPER_COMMIT = "b579a94ad5a862e704b50329d294d1c0c88c422e"
TRAINING_PYTHON = "custom/autoresearch/.training-runtime/bin/python3"
ARMS = ("MB1-B-hybrid-x2", "MB1-B-hybrid-x3", "M1-x2", "M1-x3", "M1-R-hybrid-x3")
ORDERS = ("clips1", "clips2")
CLIP_ARTIFACTS = {(arm, o): H / "bench" / f"x2b-20261021-test-{o}" / f"clips-{arm}.json" for arm in ARMS for o in ORDERS}
FP32 = COMPARISON / "run1" / "nemo_fp32.json"
BF16 = COMPARISON / "run4" / "nemo_bf16_graphs_fused_decoder.json"
EXPANDED_X3_EARLIER = COMPARISON / "run4" / "packed_expanded.json"
RUN_SUMMARIES = {"MB1": Path("/mnt/hd/wilderness-labs-stt/parakeet-binary/runs/main-binB-m1-lr2e-4/summary.json"),
                 "M1": Path("/mnt/hd/wilderness-labs-stt/parakeet-ternary/runs/main-M1-P2-lr5e-4/summary.json")}
FREEZE = Path("/mnt/hd/wilderness-labs-stt/parakeet-binary/gpu1bit/freeze-x2b-20261021.sha256")  # the benchmark freeze
RTX5090 = "finetune/parakeet-ternary/inference/results/rtx5090.json"  # relative to the repository

# SHA-256 of every source, pinned 2026-10-22 after reading them. A regenerated report fails the stage
# until it is reviewed and re-pinned (plans/card-update-x2.md section 7, step 2).
PINNED = {
    REPORT_DEV: "7471e6adefaa70c5940ec87f912f4f8aed495791615e670110f473c3b0189614",
    REPORT_TEST: "329095ebbc3a93da3dd146819c841d070d6fd78cf60b77bca77ac0dfac441afa",
    GATES: "2d56a8d029846c8613ee818cc131290384ce6b8b798c8d26280de7e883488d3e",
    SUMMARIES["MB1-B-hybrid-x2"]: "718ef97e69e339df0aa5354dfb6e73c08b06524b8aeb77159ebd886fee9174e3",
    SUMMARIES["MB1-B-hybrid-x3"]: "96c0b858f6341c7ab3ea81e5ad2f600c681c178d8af2d057d3047f90168df66e",
    SUMMARIES["M1-x2"]: "51e2eb976d1db5945ebd095ead1c4cbf1cd68d88cbd7fd0d281b45a5c9662ec6",
    SUMMARIES["M1-x3"]: "8a8b0282793f2fb43b23ee97d0000a119471c711aaf3ca36516fd821658470ce",
    IDENTITIES["MB1-B-hybrid-x2"]: "529ac505b41f57b1d1e86e475c5e7269c9a62d3d8240a49889159a7b93fe5838",
    IDENTITIES["MB1-B-hybrid-x3"]: "c444c0cddd4ac912177a6f341d4f51296cb00c78b8d22dec1df31aba7f3f4902",
    IDENTITIES["M1-x2"]: "20cae68d4d034fbe67e7efc51935cfe517867acbd0778803741bbbcb5cd3d86c",
    IDENTITIES["M1-x3"]: "cdadc5174a4ac43fa427c44df5ca2aac9c6b78ebf6f31778be04d462dd3d8596",
    CLIP_ARTIFACTS["MB1-B-hybrid-x2", "clips1"]: "3694f19b6653b1b47dfc0bfd65995361fddee2ba9a3c8b3c6e4f18d46c23d032",
    CLIP_ARTIFACTS["MB1-B-hybrid-x3", "clips1"]: "39b3935df05b48a3638437f343d2fc157084fd01ab25bc007449d13628ecfc2f",
    CLIP_ARTIFACTS["M1-x2", "clips1"]: "afc5bfd4aead199ffa0fdbe89c361a7c7fb2e8c266355a3da6116b2bd8957d8f",
    CLIP_ARTIFACTS["M1-x3", "clips1"]: "ff881631ca7f5665f7d417ae09af2cc58e617afe68a46c0784ce598ec158dc82",
    CLIP_ARTIFACTS["M1-R-hybrid-x3", "clips1"]: "0cdfe2bc7066c91daac3be8153c942cba247bd6c91918ae4fc387e002c69b6bf",
    CLIP_ARTIFACTS["MB1-B-hybrid-x2", "clips2"]: "8ebddb5124aa86fa555ba46d71c61cc3f08d308749c4e48782f33ae7870bedb1",
    CLIP_ARTIFACTS["MB1-B-hybrid-x3", "clips2"]: "8846e6281b60f5d51008ac69b554ec24800464f8adb62479cce305da38eaa150",
    CLIP_ARTIFACTS["M1-x2", "clips2"]: "302da7034b5700eb1d671036752fa036aae6e92880b433e3c1e49fd7de7bbfcc",
    CLIP_ARTIFACTS["M1-x3", "clips2"]: "2565e7e9260f8594833164737fbb0413b06005021af11b146d8f48aefcc27437",
    CLIP_ARTIFACTS["M1-R-hybrid-x3", "clips2"]: "904f8541655db3f98fdb7b4774f14c0c41adff6593be31e26eaacd0f8ce99cae",
    FREEZE: "c5484484bda8f692c4fdbd3fb4eb40b3ca38df0ce39d728df59edb6a6b7f75ab",
    FP32: "c18c2aaca635ef9be863e260c2bcfc5c31812e02e0f44b726096869201e9645a",
    BF16: "38b1a5568396f6d4a5c7bde88c51b9d663302ea820bb5d5681865ab45801d6a5",
    EXPANDED_X3_EARLIER: "10c3786026aabfb2ddbb0979923c60004fd9aa07697bd9db4e12898bc087a150",
    RUN_SUMMARIES["MB1"]: "1dcecfcd1a08ec54094babfe58261d79c1df9c022b9b80a53eed716d20fb425a",
    RUN_SUMMARIES["M1"]: "2f3a7ca14208c38640be9b84341fd456c171250a2ee8c6a16696d55fa93762d2",
}

# The x2 runtime is published only when origin/main carries the gated selection as the deployment pointer
# (the selection document every gate, benchmark and identity hashed) and the runtime README. Until the
# x2 thread's PR merges, this check fails and neither card can be uploaded.
GATED_SELECTION_SHA256 = "de3c18c4ab20d33b04abad970ae6ce9988dd7a4dc7cea597fce3ae1c9555ba2f"
MAIN_SELECTION = "finetune/parakeet-binary/gpu1bit/shared/selected.json"
MAIN_RUNTIME_README = "finetune/parakeet-binary/gpu1bit/README.md"  # the research runtime's README (x2 PR)
MAIN_RESULTS = "finetune/parakeet-binary/gpu1bit/RESULTS.md"
# The measured x2 kernels (loaded from B/wip at benchmark time) are published in measured-x2-parts/; they
# are pinned in actprec/parts.py PINNED, and parts.py is in the benchmark's freeze file.
MAIN_PARTS_DIR = "finetune/parakeet-binary/gpu1bit/measured-x2-parts"
MAIN_PARTS_PY = "finetune/parakeet-binary/gpu1bit/actprec/parts.py"
PUBLISHED_CHECK = "x2 runtime published on origin/main (gated selection, runtime README, measured kernels)"

MODELS = {"MB1": {"x2": "MB1-B-hybrid-x2", "x3": "MB1-B-hybrid-x3", "pair": "b1-b2"},
          "M1": {"x2": "M1-x2", "x3": "M1-x3", "pair": "b4-b5"}}
EXPORT_SHA = {"MB1": "9873d2f1a53127c673d0201db98cd3ba859ba432f334fef55e9e24fd057b9651",
              "M1": "a287e97719c451b785be2cd01ecc861fcaa010ebaa4ff2841783ec78fcd61503"}

CHECKS = ("x2 sources SHA-256 == pinned",
          "x2 arms are the published exports and the gated identities",
          "x2 dev gates passed",
          "x2 clip artifacts == report-test",
          "full-precision rows: same clips, GPU, driver, power limit and windows",
          "FP32 dev WER == the run summaries",
          "x2 runtime environment == the gate identities",
          PUBLISHED_CHECK)

FP32_LABEL = "Original FP32 — NeMo"
BF16_LABEL = "Original BF16 — optimized by Wilderness Labs"
TABLE_HEADER = ("| Clip | Configuration | Latency (ms) | Process VRAM (GB) | GPU energy (J) "
                "| Avg GPU power (W) |")
PAIRED_HEADER = "| Scope | GPU energy | Latency |"
KEYS = ("median_ms", "joules_per_iteration", "average_watts")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_show(repo: Path, path: str) -> bytes | None:
    out = subprocess.run(["git", "-C", str(repo), "show", f"origin/main:{path}"], capture_output=True)
    return out.stdout if out.returncode == 0 else None


def _git_rev_sha(repo: Path, rev: str, path: str) -> str | None:
    out = subprocess.run(["git", "-C", str(repo), "show", f"{rev}:{path}"], capture_output=True)
    return sha(out.stdout) if out.returncode == 0 else None


def _git_show_sha(repo: Path, path: str) -> str | None:
    data = _git_show(repo, path)
    return None if data is None else sha(data)


def _pinned_dict(source: bytes | None) -> dict | None:
    """The literal PINNED = {...} of a Python file, None if absent."""
    import ast
    if source is None:
        return None
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PINNED" for t in node.targets):
            return ast.literal_eval(node.value)
    return None


def _main_parts(repo: Path) -> dict[str, str]:
    """{file name: SHA-256} of origin/main's measured-x2-parts/ (without its own hash list)."""
    out = subprocess.run(["git", "-C", str(repo), "ls-tree", "--name-only", f"origin/main:{MAIN_PARTS_DIR}"],
                         capture_output=True, text=True)
    names = out.stdout.split() if out.returncode == 0 else []
    return {n: _git_show_sha(repo, f"{MAIN_PARTS_DIR}/{n}") for n in names if n != "parts-sha256.txt"}


def freeze_entry(freeze_text: str, rel: str) -> str | None:
    hits = [line.split()[0] for line in freeze_text.splitlines() if line.split()[1:] == [rel]]
    return hits[0] if len(hits) == 1 else None


def load(repo: Path) -> dict:
    """Every x2 source, its bytes' SHA-256, and the origin/main facts the publication check needs."""
    freeze = FREEZE.read_bytes()
    paths = [REPORT_DEV, REPORT_TEST, GATES, *SUMMARIES.values(), *IDENTITIES.values(), *CLIP_ARTIFACTS.values(), FP32, BF16,
             EXPANDED_X3_EARLIER, *RUN_SUMMARIES.values()]
    raw = {p: Path(p).read_bytes() for p in paths}
    docs = {p: json.loads(b) for p, b in raw.items()}
    return {
        "sha256": {str(p): sha(b) for p, b in raw.items()} | {str(FREEZE): sha(freeze)},
        "freeze_parts_py": freeze_entry(freeze.decode(), MAIN_PARTS_PY),
        "dev": docs[REPORT_DEV], "test": docs[REPORT_TEST], "gates": docs[GATES],
        "summaries": {arm: docs[p] for arm, p in SUMMARIES.items()},
        "identities": {arm: docs[p] for arm, p in IDENTITIES.items()},
        # the interpreter running this staging, the repository's training python and the wrappers at
        # WRAPPER_COMMIT, compared with the ones the gated arms recorded
        "staging_env": {"python": f"{sys.version_info.major}.{sys.version_info.minor}",
                        "executable": os.path.realpath(sys.executable),
                        "training_python": os.path.realpath(repo / TRAINING_PYTHON),
                        "parakeet_python_override": os.environ.get("PARAKEET_PYTHON"),
                        "wrappers": {rel: _git_rev_sha(repo, WRAPPER_COMMIT, rel) for rel in WRAPPERS}},
        "clips": {key: docs[p] for key, p in CLIP_ARTIFACTS.items()},
        "fp32": docs[FP32], "bf16": docs[BF16], "expanded_earlier": docs[EXPANDED_X3_EARLIER],
        "run_summaries": {m: docs[p] for m, p in RUN_SUMMARIES.items()},
        "rtx5090": json.loads((repo / RTX5090).read_text()),
        "main": {"selection_sha256": _git_show_sha(repo, MAIN_SELECTION),
                 "runtime_readme_sha256": _git_show_sha(repo, MAIN_RUNTIME_README),
                 "results_sha256": _git_show_sha(repo, MAIN_RESULTS),
                 "parts_py_sha256": _git_show_sha(repo, MAIN_PARTS_PY),
                 "parts_pinned": _pinned_dict(_git_show(repo, MAIN_PARTS_PY)),
                 "measured_parts": _main_parts(repo)},
    }


# ----------------------------------------------------------------------------- values

def clips(x2: dict) -> list[tuple[str, float]]:
    """(record id, duration) of the three clips, in the full-precision runs' order."""
    return [(w["record"]["id"], w["record"]["duration"]) for w in x2["fp32"]["workloads"]]


def _workload(doc: dict, clip: str) -> dict:
    (w,) = [w for w in doc["workloads"] if w["record"]["id"] == clip]
    return w


def warm_peak(w: dict) -> int:
    return max(int(x["process_sampled_peak_bytes"]) for x in w["windows"])


def full_precision(x2: dict, key: str, clip: str) -> dict:
    """{median_ms, joules_per_iteration, average_watts, vram_bytes} of the FP32 or BF16 row."""
    w = _workload(x2[key], clip)
    return {**{k: w["summary"][k] for k in KEYS}, "vram_bytes": warm_peak(w)}


def arm(x2: dict, name: str, clip: str) -> dict:
    """{median_ms, joules_per_iteration, average_watts, vram_bytes} of an arm on a clip, this session."""
    runs = x2["test"]["p_clips"][name][clip]
    vram = max(warm_peak(_workload(x2["clips"][name, o], clip)) for o in ORDERS)
    return {**{k: statistics.median(r[k] for r in runs) for k in KEYS}, "vram_bytes": vram}


def runtime_rows(x2: dict, model: str, labels: dict[str, str]) -> list[str]:
    """The RUNTIME.md table: FP32, BF16, then the model's x3 and x2 arms, per clip."""
    def cells(v: dict) -> list[str]:
        return [f"{v['median_ms']:.2f}", f"{v['vram_bytes'] / 1e9:.2f}", f"{v['joules_per_iteration']:.2f}",
                f"{v['average_watts']:.0f}"]
    out = []
    for clip, seconds in clips(x2):
        rows = [(FP32_LABEL, full_precision(x2, "fp32", clip)), (BF16_LABEL, full_precision(x2, "bf16", clip)),
                (labels["x3"], arm(x2, MODELS[model]["x3"], clip)), (labels["x2"], arm(x2, MODELS[model]["x2"], clip))]
        out += ["| " + " | ".join([f"{seconds:g} s", label, *cells(v)]) + " |" for label, v in rows]
    return out


def pair(x2: dict, report: str, scope: str, model: str) -> dict:
    return x2[report][scope]["pairs"][MODELS[model]["pair"]]


def _change(p: dict, less: str, more: str) -> str:
    pct = p["median_pct"]
    return f"{abs(pct):.1f}% {less if pct < 0 else more} ({p['lower']} of {p['n']})"


def paired_rows(x2: dict, model: str) -> list[str]:
    out = []
    for scope, label in (("full", "Full transcription"), ("encoder", "Encoder only")):
        p = pair(x2, "dev", scope, model)
        out.append(f"| {label} | {_change(p['energy'], 'less', 'more')} | {_change(p['latency'], 'lower', 'higher')} |")
    return out


def ceil_to(x: float, step: float) -> float:
    return math.ceil(round(x / step, 9)) * step


def test_agreement(x2: dict, model: str) -> float:
    """Ceiling to 0.1 of the largest |dev - test| over the shown pair's scopes and metrics (points)."""
    diffs = [abs(pair(x2, "dev", s, model)[m]["median_pct"] - pair(x2, "test", s, model)[m]["median_pct"])
             for s in ("full", "encoder") for m in ("energy", "latency")]
    return ceil_to(max(diffs), 0.1)


def remeasure(x2: dict) -> dict:
    """How well Kea-T's earlier runtimes carried across sessions on the same clips: b5 against
    run4/packed_expanded.json (the BF16 row's session) and b6 against report-test's embedded record of
    cuda-iteration-20261007/packed_cuda_best.json. Max |relative change| in percent, and whether b5's
    process VRAM is identical."""
    lat, energy, vram_same = [], [], True
    recorded = x2["test"]["m1_recorded"]["clips"]["packed_cuda_best"]["clips"]
    for clip, _ in clips(x2):
        b5, earlier = arm(x2, "M1-x3", clip), _workload(x2["expanded_earlier"], clip)
        b6 = arm(x2, "M1-R-hybrid-x3", clip)
        for now, then in ((b5, earlier["summary"]), (b6, recorded[clip])):
            lat.append(abs(now["median_ms"] / then["median_ms"] - 1) * 100)
            energy.append(abs(now["joules_per_iteration"] / then["joules_per_iteration"] - 1) * 100)
        vram_same &= b5["vram_bytes"] == warm_peak(earlier)
    return {"latency_max_pct": max(lat), "energy_max_pct": max(energy), "expanded_vram_identical": vram_same,
            "latency_within": math.ceil(max(lat)), "energy_within": math.ceil(max(energy))}


def dev_wer(x2: dict, model: str) -> dict:
    s = x2["summaries"]
    utterances = s[MODELS[model]["x2"]]["utterances"]
    return {"x2": s[MODELS[model]["x2"]]["dev_mean_wer_percent"], "x3": s[MODELS[model]["x3"]]["dev_mean_wer_percent"],
            "fp32": 100 * x2["run_summaries"][model]["dev_mean_wer"], "utterances": sum(utterances.values())}


def environment(x2: dict) -> dict:
    """The research runtime's environment as the gated arms' identities recorded it, and the GPU class from
    the per-clip artifacts. "same" is true when every identity and artifact agrees."""
    ids = x2["identities"]
    v = ids["MB1-B-hybrid-x2"]["versions"]
    vf = x2["clips"]["MB1-B-hybrid-x2", "clips1"]["variant_facts"]
    same = (all(i["versions"].get(k) == v.get(k) for i in ids.values() for k in ("torch", "triton", "nemo_toolkit"))
            and all(x2["clips"][a, o]["variant_facts"].get(k) == vf.get(k)
                    for a in ARMS for o in ORDERS for k in ("torch", "triton", "cuda", "gpu", "capability"))
            and vf["torch"] == v["torch"] and vf["triton"] == v["triton"])
    return {"torch": v["torch"], "triton": v["triton"], "nemo": v["nemo_toolkit"],
            "python": x2["staging_env"]["python"], "gpu": vf["gpu"].replace("NVIDIA GeForce ", ""),
            "sm": "SM{}{}".format(*vf["capability"]), "same": same}


def setting(x2: dict) -> dict:
    g = x2["clips"]["MB1-B-hybrid-x2", "clips1"]
    return {"power_limit": g["gpu"]["power_limit_w"], "default_power_limit": g["gpu"]["default_power_limit_w"],
            "seconds": g["args"]["seconds"], "repeats": g["args"]["repeats"],
            "gpu": g["gpu"]["name"].replace("NVIDIA GeForce ", "")}


# ----------------------------------------------------------------------------- checks

def check(x2: dict, checks: dict, info: dict) -> None:
    """Record CHECKS (plans/card-update-x2.md section 5B)."""
    dev, test, gates, summaries = x2["dev"], x2["test"], x2["gates"], x2["summaries"]
    checks["x2 sources SHA-256 == pinned"] = all(x2["sha256"].get(str(p)) == want for p, want in PINNED.items())

    models = {a: ("MB1" if a.startswith("MB1") else "M1") for a in ARMS}
    labels = dev.get("labels", {})
    exports_ok = all(labels.get(a, {}).get("export_sha256") == EXPORT_SHA[models[a]] for a in ARMS)
    identities_ok = all(
        gates["identities"].get(g) == dev["identities"].get(a) == summaries[a]["identity"] is not None
        and gates["arms"].get(g) == a for g, a in GATED.items())
    # b6 (the ungated reference) records no arm or identity; its variant name is checked with the artifacts
    artifact_ids_ok = all(x2["clips"][a, o]["variant_info"].get("arm") == a
                          and x2["clips"][a, o]["variant_info"].get("identity") == dev["identities"].get(a)
                          for a in GATED.values() for o in ORDERS)
    checks["x2 arms are the published exports and the gated identities"] = (
        exports_ok and identities_ok and artifact_ids_ok and test["roster_sha256"] == dev["roster_sha256"])

    checks["x2 dev gates passed"] = (
        gates.get("pass") is True and gates["mb1"].get("pass") is True and gates["m1"].get("pass") is True
        and dev.get("gate_run") == gates.get("run")
        and all(s.get("complete") is True and s.get("structurally_valid") is True for s in summaries.values()))

    ids = clips(x2)
    gpu = x2["fp32"]["gpu"]
    artifacts_ok = True
    for a in ARMS:
        for o in ORDERS:
            doc = x2["clips"][a, o]
            artifacts_ok &= (doc["variant"] == a and doc["args"]["run"].endswith(f"test-{o}")
                             and [(w["record"]["id"], w["record"]["duration"]) for w in doc["workloads"]] == ids
                             and doc["gpu"]["power_limit_w"] == gpu["power_limit_w"]
                             and doc["gpu"]["uuid"] == gpu["uuid"] and doc["gpu"]["driver_version"] == gpu["driver_version"]
                             and doc["args"]["seconds"] == x2["fp32"]["args"]["seconds"]
                             and doc["args"]["repeats"] == x2["fp32"]["args"]["repeats"])
            for w in doc["workloads"]:
                (rec,) = [r for r in test["p_clips"][a][w["record"]["id"]] if r["order"] == o]
                artifacts_ok &= all(rec[k] == w["summary"][k] for k in KEYS) and rec["text"] == w["hypothesis"]
        artifacts_ok &= all(len(test["p_clips"][a][c]) == len(ORDERS) for c, _ in ids)
    checks["x2 clip artifacts == report-test"] = artifacts_ok

    rtx_bf16 = {w["record_id"]: w for w in x2["rtx5090"]["variants"]["nemo_bf16_graphs_fused_decoder"]["workloads"]}
    fp_ok = True
    for key in ("fp32", "bf16", "expanded_earlier"):
        doc = x2[key]
        fp_ok &= ([(w["record"]["id"], w["record"]["duration"]) for w in doc["workloads"]] == ids
                  and doc["gpu"]["uuid"] == gpu["uuid"] and doc["gpu"]["driver_version"] == gpu["driver_version"]
                  and doc["gpu"]["power_limit_w"] == 400.0 and doc["args"]["seconds"] == 5
                  and doc["args"]["repeats"] == 3)
    for clip, _ in ids:
        bf, pub = full_precision(x2, "bf16", clip), rtx_bf16.get(clip, {})
        fp_ok &= (all(bf[k] == pub.get(k) for k in KEYS) and bf["vram_bytes"] / 1e9 == pub.get("process_vram_gb"))
    checks["full-precision rows: same clips, GPU, driver, power limit and windows"] = fp_ok

    wer_ok = True
    for model in MODELS:
        x3 = summaries[MODELS[model]["x3"]]
        cross = x3.get("cross_check", {})
        wer_ok &= (cross.get("pass") is True and Path(cross.get("pinned", "")) == RUN_SUMMARIES[model]
                   and abs(cross.get("pinned_mean_percent", -1) - 100 * x2["run_summaries"][model]["dev_mean_wer"]) < 1e-9)
    checks["FP32 dev WER == the run summaries"] = wer_ok

    env = environment(x2)
    se = x2["staging_env"]
    checks["x2 runtime environment == the gate identities"] = (
        env["same"] and se["executable"] == se["training_python"] and not se["parakeet_python_override"]
        and all(i["sources"].get(rel) == se["wrappers"][rel] is not None
                for i in x2["identities"].values() for rel in WRAPPERS))

    main = x2["main"]
    checks[PUBLISHED_CHECK] = (
        main.get("selection_sha256") == GATED_SELECTION_SHA256
        and main.get("runtime_readme_sha256") is not None and main.get("results_sha256") is not None
        and bool(main.get("measured_parts")) and main.get("measured_parts") == main.get("parts_pinned")
        and main.get("parts_py_sha256") is not None and main.get("parts_py_sha256") == x2.get("freeze_parts_py"))
    info["x2"] = {"sources": x2["sha256"], "origin_main": main, "remeasure": remeasure(x2),
                  "clips": ids, "test_agreement": {m: test_agreement(x2, m) for m in MODELS}}


# ----------------------------------------------------------------------------- phrases

def ok(cond: bool, phrase: str) -> str | None:
    return phrase if cond else None


def ten_seconds(x2: dict) -> float:
    (mid,) = [s for _, s in clips(x2) if s == 10.0]
    return mid


def clip_at(x2: dict, seconds: float) -> str:
    (clip,) = [c for c, s in clips(x2) if s == seconds]
    return clip


def glance_rows(x2: dict, model: str, vram_row: bool = True) -> list[str]:
    """The at-a-glance rows after weight file and WER: latency, approximate energy and process VRAM at
    10 s, the model's x2 arm against the BF16 build (cross-session)."""
    mid = ten_seconds(x2)
    clip = clip_at(x2, mid)
    a, b = arm(x2, MODELS[model]["x2"], clip), full_precision(x2, "bf16", clip)
    lat = 100 * (1 - a["median_ms"] / b["median_ms"])
    energy = 100 * (1 - a["joules_per_iteration"] / b["joules_per_iteration"])
    vram = 100 * (1 - a["vram_bytes"] / b["vram_bytes"])
    if min(lat, energy, vram) <= 0:
        return []
    g = f"{mid:g} s clip"
    return [f"| Latency, {g} | **{a['median_ms']:.2f} ms** ({lat:.1f}% lower) | {b['median_ms']:.2f} ms |",
            f"| GPU energy, {g} (approximate) | **{a['joules_per_iteration']:.2f} J** (about {round(energy)}% less) "
            f"| {b['joules_per_iteration']:.2f} J |",
            f"| Process VRAM, {g} | **{a['vram_bytes'] / 1e9:.2f} GB** ({vram:.1f}% lower) | {b['vram_bytes'] / 1e9:.2f} GB |"
            ][:3 if vram_row else 2]


def research_glance_phrase(x2: dict, model: str) -> str | None:
    """The card sentence below a runnable-runtime glance (Kea-T): the x2 research runtime at 10 s against
    the BF16 build, latency to one decimal, energy approximate; None if either is not lower."""
    clip = clip_at(x2, ten_seconds(x2))
    a, b = arm(x2, MODELS[model]["x2"], clip), full_precision(x2, "bf16", clip)
    lat = 100 * (1 - a["median_ms"] / b["median_ms"])
    energy = 100 * (1 - a["joules_per_iteration"] / b["joules_per_iteration"])
    return ok(lat > 0 and energy > 0,
              f"at {ten_seconds(x2):g} s, {a['median_ms']:.2f} ms ({lat:.1f}% lower than the BF16 build) and about "
              f"{a['joules_per_iteration']:.2f} J (about {round(energy)}% less), measured in a later session, so the "
              "energy comparison is approximate")


def facts(x2: dict, model: str, card: str, run_md: str, subject: str) -> dict:
    """The x2 phrases shared by both models: {label: (documents, phrase or None)}. subject names the x2
    runtime at the start of the 10 s comparison ("Kea-B with x2" or "x2")."""
    ids = clips(x2)
    (short, _), (mid_clip, mid), (long_clip, long_) = ids[0], ids[1], ids[2]
    s = setting(x2)
    env = environment(x2)
    rm = remeasure(x2)
    wer = dev_wer(x2, model)
    x2_arm, x3_arm = MODELS[model]["x2"], MODELS[model]["x3"]
    a = {c: arm(x2, x2_arm, c) for c, _ in ids}
    a3 = {c: arm(x2, x3_arm, c) for c, _ in ids}
    b = {c: full_precision(x2, "bf16", c) for c, _ in ids}
    seconds = [sec for _, sec in ids]
    lat_saving = [b[c]["median_ms"] - a[c]["median_ms"] for c, _ in ids]
    e_saving = [b[c]["joules_per_iteration"] - a[c]["joules_per_iteration"] for c, _ in ids]
    lat10 = 100 * (1 - a[mid_clip]["median_ms"] / b[mid_clip]["median_ms"])
    e10 = 100 * (1 - a[mid_clip]["joules_per_iteration"] / b[mid_clip]["joules_per_iteration"])
    v10 = 100 * (1 - a[mid_clip]["vram_bytes"] / b[mid_clip]["vram_bytes"])
    full = pair(x2, "dev", "full", model)
    by_target = full["energy"]["by_target"]
    per_target = sorted({v["n"] for v in by_target.values()})
    targets = sorted(by_target, key=float)
    dev_abs = x2["dev"]["absolute_median"]["full"]
    test_n = pair(x2, "test", "full", model)["energy"]["n"]
    orders_agree = all(warm_peak(_workload(x2["clips"][arm_, "clips1"], c))
                       == warm_peak(_workload(x2["clips"][arm_, "clips2"], c)) for arm_ in (x2_arm, x3_arm) for c, _ in ids)
    gpu, pl = s["gpu"], s["power_limit"]
    out = {
        "x2 speed setting": ((card, run_md), f"one {gpu} at a {pl:.0f} W power limit"),
        "x2 other clip lengths": ((card,), f"{seconds[0]:g} s and {seconds[2]:g} s clips"),
        "x2 cross-session agreement": ((card, run_md), f"agreed within {rm['latency_within']}% on latency but only "
                                                       f"within {rm['energy_within']}% on energy"),
        "x2 earlier runtimes' VRAM identical": ((run_md,), ok(rm["expanded_vram_identical"],
                                                              "the expanded runtime's process VRAM was identical")),
        "x2 dev WER (card)": ((card,), ok(f"{wer['x2']:.2f}" == f"{wer['fp32']:.2f}",
                                          f"its mean WER, {wer['x2']:.2f}%, is the same as FP32 decoding's")),
        "x2 runtime setting (limitations)": ((card, run_md), f"one {gpu} at a {pl:.0f} W limit"),
        "x2 at 10 s vs BF16": ((run_md,), ok(min(lat10, e10, v10) > 0,
                                             f"At {mid:g} s, against our BF16 build, {subject} has {lat10:.1f}% lower "
                                             f"latency, {v10:.1f}% lower process VRAM and uses about {round(e10)}% less "
                                             "GPU energy")),
        "x2 latency saving per transcription": ((run_md,), ok(
            min(lat_saving) > 0 and max(lat_saving) / min(lat_saving) <= 1.25,
            "latency saving is roughly constant per transcription (" + ", ".join(f"{x:.1f}" for x in lat_saving[:-1])
            + f" and {lat_saving[-1]:.1f} ms at " + ", ".join(f"{x:g}" for x in seconds[:-1]) + f" and {seconds[-1]:g} s)")),
        "x2 energy saving grows": ((run_md,), ok(
            min(e_saving) > 0 and all(x < y for x, y in zip(e_saving, e_saving[1:])),
            "energy saving grows with clip length (about " + ", ".join(f"{x:.2f}" for x in e_saving[:-1])
            + f" and {e_saving[-1]:.2f} J)")),
        "x2 dev_24": ((run_md,), ok(len(per_target) == 1 and full["energy"]["n"] == sum(v["n"] for v in by_target.values()),
                                    f"{full['energy']['n']} development utterances ({per_target[0] if per_target else 0} "
                                    "each with target lengths of " + ", ".join(targets[:-1]) + f" and {targets[-1]} s)")),
        "x2 power fell": ((run_md,), ok(dev_abs[x2_arm]["average_watts"] < dev_abs[x3_arm]["average_watts"],
                                        f"fell from {dev_abs[x3_arm]['average_watts']:.0f} W to "
                                        f"{dev_abs[x2_arm]['average_watts']:.0f} W")),
        "x2 test agreement": ((run_md,), f"same comparisons on {test_n} test utterances agree within "
                                         f"{test_agreement(x2, model):.1f} points"),
        "x2 dev WER": ((run_md,), f"On the full development sets ({wer['utterances']:,} utterances, batch one), the "
                                  f"runtime's mean WER is {wer['x2']:.4f}% with x2 and {wer['x3']:.4f}% with x3; FP32 "
                                  f"decoding of this export gives {wer['fp32']:.4f}%"),
        "x2 processes agreed": ((run_md,), ok(orders_agree, "processes agreed exactly")),
        "x2 power limit": ((run_md,), f"configured {pl:.0f} W limit (board default {s['default_power_limit']:.0f} W)"),
        "x2 windows": ((run_md,), ok(s["repeats"] == 3, f"three {s['seconds']:g} s windows")),
        "x2 same VRAM x2 and x3": ((run_md,), ok(all(a[c]["vram_bytes"] == a3[c]["vram_bytes"] for c, _ in ids),
                                                 "Process VRAM is the same for x2 and x3")),
        "x2 requirements": ((run_md,), ok(env["same"], f"{env['sm']} GPUs ({env['gpu']}) only and was measured with "
                                                        f"torch {env['torch']}, triton {env['triton']}, nemo_toolkit "
                                                        f"{env['nemo']} and Python {env['python']}")),
        "x2 research runtime": ((run_md,), f"Speed measured with our research runtime on an {env['gpu']}"),
    }
    if model != "MB1":  # M1's x2 adds a few MB at 10 and 30.04 s; only Kea-B states equality
        del out["x2 same VRAM x2 and x3"]
    return out
