"""Runtime contracts at x2 (plans/x2-default.md Z6, Z8; agent H): the comparison and refusal logic of the
x2 runtime tests, fixture-tested before the freeze (tests/test_x2_preflight_runtime.py). The real assertions
(tests/test_x2_runtime.py, tests/test_x2_runtime_fresh.py) run on the unchanged frozen sources: recognition
once the gate run exists, P-memory after the recorded benchmark and before promotion and test scoring.

- Recognition: the x2 default's reference is the provider-x2 arm g2 of the gate run: text, token IDs and
  timestamps on the 24 timing workloads, identity-bound to the gate run. Equality with x3 is reported as a
  hypothesis and never asserted.
- Memory: test_runtime_fresh's contract with its reference replaced by the recorded benchmark: per dev clip,
  the fresh-process peak including first capture is at most 1.02 x b1's P-memory peak (the max over b1's
  two P-clips recordings, bound to the runtime's identity) and at least 500 MB below R-expanded's (b3); the
  transcripts equal b1's. All three frozen dev clips must be in every recording (both orders of b1 and b3),
  in both reference maps and in the fresh measurement (Codex x2 c1, finding 6).
"""
from __future__ import annotations

KEYS = ("text", "tokens", "timestamps")
PEAK = "process_sampled_peak_including_first_bytes"
BOUND = 1.02
MARGIN_BYTES = 500_000_000


def compare_recognition(runtime_rows: dict[str, dict], ref_rows: dict[str, dict], x3_rows: dict | None = None) -> dict:
    if len(ref_rows) != 24 or set(runtime_rows) != set(ref_rows):
        raise ValueError(f"the recognition contract covers the 24 timing workloads; got {len(ref_rows)} reference "
                         f"rows and {len(runtime_rows)} runtime rows")
    diff = sorted(u for u in ref_rows if any(runtime_rows[u][k] != ref_rows[u][k] for k in KEYS))
    out = {"ok": not diff, "differences": diff, "reference": "g2 (provider int8_mma:x2)"}
    if x3_rows is not None:
        x3d = sorted(u for u in ref_rows if u in x3_rows and any(runtime_rows[u][k] != x3_rows[u][k] for k in KEYS))
        out["x3_hypothesis"] = {"status": "reported, not asserted", "differences": x3d}
    return out


RECORDINGS = ("clips1", "clips2")   # the benchmark's two P-clips orders (bench_session: <run>-clips1, -clips2)


def frozen_clip_ids() -> tuple[str, ...]:
    """The three frozen dev clips (shared/workloads.py dev_clips), the memory contract's required coverage."""
    from gpu1bit.shared import workloads
    return tuple(r["id"] for r in workloads.load()["dev_clips"])


def _recordings(docs: dict | None, what: str) -> list[tuple[str, dict]]:
    """Both required recordings (clips1, clips2) of `what`; ValueError if the benchmark or one is missing."""
    if not docs:
        raise ValueError(f"no recorded benchmark P-memory artifact for {what}")
    missing = [r for r in RECORDINGS if not docs.get(r)]
    if missing:
        raise ValueError(f"{what}: required P-clips recording(s) {missing} missing")
    return [(r, docs[r]) for r in RECORDINGS]


def _cover(doc: dict, label: str, what: str, clip_ids: tuple[str, ...]) -> None:
    got = [w["record"]["id"] for w in doc.get("workloads", [])]
    if not clip_ids or sorted(got) != sorted(clip_ids):
        raise ValueError(f"{what} {label}: clip coverage {got} is not the three frozen dev clips {list(clip_ids)}")


def pmemory_reference(b1_docs: dict | None, runtime_identity: str, clip_ids=None) -> dict[str, dict]:
    """{clip id: {"peak": max recorded peak bytes over b1's two P-clips recordings, "text": b1's transcript}},
    every recording bound to the runtime identity and covering all three frozen dev clips."""
    clip_ids = tuple(frozen_clip_ids() if clip_ids is None else clip_ids)
    out: dict[str, dict] = {}
    for label, doc in _recordings(b1_docs, "b1"):
        info = doc.get("variant_info") or {}
        if info.get("roster_id") != "b1" or info.get("identity") != runtime_identity:
            raise ValueError(f"the b1 P-memory record {label} is of roster {info.get('roster_id')!r}, identity "
                             f"{str(info.get('identity'))[:12]}, not b1 of the runtime identity {runtime_identity[:12]}")
        _cover(doc, label, "b1", clip_ids)
        for w in doc["workloads"]:
            clip, peak, text = w["record"]["id"], int(w[PEAK]), w["hypothesis"]
            if clip in out and out[clip]["text"] != text:
                raise ValueError(f"b1's P-clips recordings disagree on the transcript of {clip}")
            out[clip] = {"peak": max(peak, out.get(clip, {}).get("peak", 0)), "text": text}
    return out


def r_expanded_reference(b3_docs: dict | None, clip_ids=None) -> dict[str, int]:
    """{clip id: max recorded peak bytes over R-expanded's (b3) two P-clips recordings}, full coverage."""
    clip_ids = tuple(frozen_clip_ids() if clip_ids is None else clip_ids)
    out: dict[str, int] = {}
    for label, doc in _recordings(b3_docs, "b3"):
        if (doc.get("variant_info") or {}).get("roster_id") != "b3":
            raise ValueError(f"the R-expanded P-memory record {label} is not labelled b3")
        _cover(doc, label, "b3", clip_ids)
        for w in doc["workloads"]:
            out[w["record"]["id"]] = max(int(w[PEAK]), out.get(w["record"]["id"], 0))
    return out


def compare_pmemory(fresh: dict[str, dict], ref: dict[str, dict], r_expanded: dict[str, int], clip_ids=None) -> dict:
    """fresh: {clip id: {"peak": bytes, "texts": [...]}} of the fresh-process runtime. All three frozen clips
    must be in both reference maps and in the fresh measurement, and all three are compared."""
    clip_ids = tuple(frozen_clip_ids() if clip_ids is None else clip_ids)
    want = set(clip_ids)
    if not want or set(ref) != want or set(r_expanded) != want:
        raise ValueError(f"reference clip coverage b1 {sorted(ref)} / b3 {sorted(r_expanded)} is not the three frozen "
                         f"dev clips {sorted(want)}")
    missing = sorted(want - set(fresh))
    if missing:
        raise ValueError(f"fresh-process measurement missing clip(s) {missing}")
    rows, diff = {}, []
    for clip in clip_ids:
        r = ref[clip]
        peak = int(fresh[clip]["peak"])
        row = {"peak": peak, "b1_peak": r["peak"], "bound": BOUND * r["peak"], "r_expanded_peak": r_expanded[clip],
               "within_bound": peak <= BOUND * r["peak"], "margin_ok": peak <= r_expanded[clip] - MARGIN_BYTES,
               "texts_ok": set(fresh[clip]["texts"]) == {r["text"]}}
        rows[clip] = row
        if not (row["within_bound"] and row["margin_ok"] and row["texts_ok"]):
            diff.append(clip)
    return {"ok": not diff, "differences": diff, "rows": rows, "clips": list(clip_ids),
            "reference": f"recorded benchmark P-memory of b1 (<= {BOUND} x) and b3 (>= 500 MB below), both orders"}
