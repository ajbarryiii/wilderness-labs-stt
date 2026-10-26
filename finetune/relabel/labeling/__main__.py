"""CLI of stage 2 Part C (labeling plan): run from finetune/relabel as `./python -m labeling COMMAND`.

Configuration and C1 (CPU):
    candidate [--provisional]               the stream configuration without W (labeling/config/candidate.json;
                                            --provisional: from a non-final pool, into the smoke path)
    replay-probe --workers W [--minutes M]  C1 for M minutes at W (labeling/probes/replay-w<W>/probe.json)
    replay-choose                           the worker-sizing rule over the three probes (probes/replay-choice.json)
    pin --workers W                         the pinned configuration (labeling/config/pinned-w<W>.json; only one)
    replay [--budget-minutes 110]           C1 into pool-audio/<config id>/ (resumable chunk; run it again)
    replay-cost                             C1 wall time from the probe at the pinned W

Checks and C2 (GPU jobs hold the parakeet GPU lock for their own duration):
    gpu-probe --engine vllm|transformers --setting I [--minutes 10] --declared-vram-gb V
    consistency-prepare [--limit N] [--dir D]       stage 1 dev sets as shards
    consistency --engine ... [--dir D]              decode, then compare with stage 1 (result-run.json)
    equivalence-prepare [--dir D]                   300 ids per C1 source + dev-medical + dev-technical
    equivalence --vllm-setting I --tf-setting J --declared-vram-gb V [--dir D]
    select                                          the C2 engine and setting from probes and checks
    c2-init --run NAME --engine ... --declared-vram-gb V   labels/<T*>/<NAME>/run.json
    (chunks: ./python -m labeling.c2 chunk --run-dir labels/<T*>/<NAME>; a finished run under older filters:
     ./python -m labeling.c2 refilter --run-dir labels/<T*>/<NAME>, into labels/<T*>/<NAME>-f<filters version>)
    gate --run NAME --consistency PATH [--equivalence PATH]   the gate to stage 3 (labeling/gate/<NAME>/)
    status
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from labeling import checks, layout


def pinned() -> dict:
    from labeling import config
    found = sorted(layout.config_dir().glob("pinned-w*.json"))
    if len(found) != 1:
        raise SystemExit(f"expected exactly one pinned configuration in {layout.config_dir()}, found {len(found)}")
    p = config.load_json(found[0])
    config.verify(p)
    return p


def c1_dir() -> Path:
    from labeling import config
    return layout.pool_audio_dir(config.config_id(pinned()))


def engine_spec(kind: str, index: int) -> dict:
    from labeling.engines import total_vram_gb
    total = total_vram_gb()   # nvidia-smi: the driver process never touches CUDA
    if kind == "vllm":
        return checks.vllm_engine(checks.VLLM_SETTINGS[index], total)
    return checks.tf_engine(checks.TRANSFORMERS_SETTINGS[index])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="labeling", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("candidate")
    p.add_argument("--provisional", action="store_true")
    p = sub.add_parser("replay-probe")
    p.add_argument("--workers", type=int, required=True, choices=layout.WORKER_CANDIDATES)
    p.add_argument("--minutes", type=float, default=layout.REPLAY_PROBE_MINUTES)
    sub.add_parser("replay-choose")
    p = sub.add_parser("pin")
    p.add_argument("--workers", type=int, required=True)
    p = sub.add_parser("replay")
    p.add_argument("--budget-minutes", type=float, default=layout.CHUNK_MINUTES)
    p.add_argument("--encode-threads", type=int, default=4)
    sub.add_parser("replay-cost")
    p = sub.add_parser("gpu-probe")
    p.add_argument("--engine", choices=("vllm", "transformers"), required=True)
    p.add_argument("--setting", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--minutes", type=float, default=layout.GPU_PROBE_MINUTES)
    p.add_argument("--declared-vram-gb", type=float, required=True)
    p.add_argument("--lock-wait", type=float, default=1800.0)
    p = sub.add_parser("consistency-prepare")
    p.add_argument("--limit", type=int)
    p.add_argument("--dir", type=Path)
    p = sub.add_parser("consistency")
    p.add_argument("--engine", choices=("vllm", "transformers"), required=True)
    p.add_argument("--setting", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--declared-vram-gb", type=float, required=True)
    p.add_argument("--group-shards", type=int, default=checks.GROUP_SHARDS)
    p.add_argument("--dir", type=Path)
    p.add_argument("--lock-wait", type=float, default=1800.0)
    p = sub.add_parser("equivalence-prepare")
    p.add_argument("--dir", type=Path)
    p = sub.add_parser("equivalence")
    p.add_argument("--vllm-setting", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--tf-setting", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--declared-vram-gb", type=float, required=True)
    p.add_argument("--group-shards", type=int, default=checks.GROUP_SHARDS)
    p.add_argument("--dir", type=Path)
    p.add_argument("--lock-wait", type=float, default=1800.0)
    sub.add_parser("select")
    p = sub.add_parser("c2-init")
    p.add_argument("--run", required=True)
    p.add_argument("--engine", choices=("vllm", "transformers"), required=True)
    p.add_argument("--setting", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--declared-vram-gb", type=float, required=True)
    p.add_argument("--group-shards", type=int, default=checks.GROUP_SHARDS)
    p = sub.add_parser("gate")
    p.add_argument("--run", required=True)
    p.add_argument("--consistency", type=Path, required=True)
    p.add_argument("--equivalence", type=Path, help="the equivalence result (required for a vLLM run)")
    sub.add_parser("status")
    args = parser.parse_args(argv)
    layout.require_mount()

    from labeling import c2, config, replay
    if args.cmd == "candidate":
        cand = config.build_candidate(layout.pool_dir(), layout.corpora_dir(), allow_provisional=args.provisional)
        out = layout.smoke_dir("provisional") / "candidate.json" if args.provisional else layout.candidate_path()
        config.write_json(out, cand)
        print(json.dumps({"out": str(out), "final": cand["final"], "shares": cand["shares"]}, indent=1))
    elif args.cmd == "replay-probe":
        cand = config.load_json(layout.candidate_path())
        res = replay.probe(cand, args.workers, args.minutes, layout.probes_dir() / f"replay-w{args.workers}")
        print(json.dumps({k: v for k, v in res.items() if k != "monitor"}, indent=1))
    elif args.cmd == "replay-choose":
        results = {}
        for w in layout.WORKER_CANDIDATES:
            path = layout.probes_dir() / f"replay-w{w}" / "probe.json"
            if path.exists():
                results[w] = config.load_json(path)
        choice = replay.choose_workers(results)
        choice["training_smoke"] = ("step 3 (a 10-minute training smoke at W under the 36G cap) runs on stage 3's "
                                    "entry point before the configuration is used for training")
        config.write_json(layout.probes_dir() / "replay-choice.json", choice)
        print(json.dumps(choice, indent=1))
    elif args.cmd == "pin":
        cand = config.load_json(layout.candidate_path())
        if not cand["final"]:
            raise SystemExit("the candidate was built from a non-final pool")
        others = [p for p in layout.config_dir().glob("pinned-w*.json") if p != layout.pinned_path(args.workers)]
        if others:
            raise SystemExit(f"another pinned configuration exists: {others}")
        pin = config.pin(cand, args.workers)
        path = layout.pinned_path(args.workers)
        if path.exists() and config.load_json(path)["pinned_hash"] != pin["pinned_hash"]:
            raise SystemExit(f"{path} exists with another hash")
        config.write_json(path, pin)
        print(json.dumps({"pinned": str(path), "config_id": config.config_id(pin),
                          "stream_config_hash": pin["stream_config_hash"]}, indent=1))
    elif args.cmd == "replay":
        pin = pinned()
        st = replay.replay(pin, c1_dir(), budget_s=args.budget_minutes * 60, encode_threads=args.encode_threads)
        print(json.dumps(st, indent=1))
    elif args.cmd == "replay-cost":
        w = pinned()["workers"]
        print(json.dumps(replay.cost(config.load_json(layout.probes_dir() / f"replay-w{w}" / "probe.json")), indent=1))
    elif args.cmd == "gpu-probe":
        eng = engine_spec(args.engine, args.setting)
        res = checks.run_gpu_probe(c1_dir(), layout.probes_dir() / "gpu" / f"{args.engine}-{args.setting}", eng,
                                   args.minutes, args.declared_vram_gb, lock_wait=args.lock_wait)
        print(json.dumps({k: res[k] for k in ("audio_hours_per_gpu_hour", "peak_vram_gb", "declared_vram_gb_for_c2")},
                         indent=1))
    elif args.cmd == "consistency-prepare":
        d = args.dir or (layout.smoke_dir(f"consistency-limit{args.limit}") if args.limit else layout.consistency_dir("dev"))
        print(json.dumps(checks.prepare_consistency(d, limit=args.limit), indent=1))
    elif args.cmd == "consistency":
        base = args.dir or layout.consistency_dir("dev")
        res = checks.run_consistency(base, engine_spec(args.engine, args.setting), args.declared_vram_gb,
                                     group_shards=args.group_shards, lock_wait=args.lock_wait,
                                     run_name=f"{args.engine}-{args.setting}")
        print(json.dumps({"passed": res["passed"], "sets": res["sets"]}, indent=1))
    elif args.cmd == "equivalence-prepare":
        d = args.dir or layout.equivalence_dir("main")
        print(json.dumps({k: v for k, v in checks.prepare_equivalence(c1_dir(), d).items() if k != "c1_ids"}, indent=1))
    elif args.cmd == "equivalence":
        d = args.dir or layout.equivalence_dir("main")
        v = engine_spec("vllm", args.vllm_setting)
        res = checks.run_equivalence(d, checks.VLLM_SETTINGS[args.vllm_setting],
                                     checks.TRANSFORMERS_SETTINGS[args.tf_setting], args.declared_vram_gb,
                                     v["total_vram_gb"], group_shards=args.group_shards, lock_wait=args.lock_wait)
        print(json.dumps({k: res[k] for k in ("passed", "vllm_runs_identical", "sources", "dev")}, indent=1))
    elif args.cmd == "select":
        probes = [config.load_json(p) for p in sorted((layout.probes_dir() / "gpu").glob("*/probe.json"))]
        cons = [config.load_json(p) for p in sorted(layout.consistency_dir("dev").glob("result-*.json"))]
        eq_path = layout.equivalence_dir("main") / "result.json"
        eq = config.load_json(eq_path) if eq_path.exists() else None
        print(json.dumps(checks.select_production(probes, cons, eq), indent=1))
    elif args.cmd == "c2-init":
        eng = engine_spec(args.engine, args.setting)
        run_dir = layout.labels_dir(layout.TEACHER, args.run)
        print(json.dumps(c2.init_run(run_dir, pool_dir=c1_dir(), engine=eng, group_shards=args.group_shards,
                                     b0_batch_seconds=checks.B0_BATCH_SECONDS,
                                     declared_vram_gb=args.declared_vram_gb), indent=1))
    elif args.cmd == "gate":
        res = checks.gate(pinned(), c1_dir(), layout.labels_dir(layout.TEACHER, args.run), args.consistency,
                          layout.gate_dir(args.run), equivalence_result=args.equivalence)
        print(json.dumps({"passed": res["passed"], "items": res["items"]}, indent=1))
    else:
        out = {"candidate": layout.candidate_path().exists(),
               "pinned": [p.name for p in layout.config_dir().glob("pinned-w*.json")] if layout.config_dir().exists() else []}
        if out["pinned"]:
            st = c1_dir() / "status.json"
            out["c1"] = config.load_json(st) if st.exists() else None
        print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
