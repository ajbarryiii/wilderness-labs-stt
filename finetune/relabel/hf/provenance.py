"""Training-data provenance of Kea-B 0.1 (release-kea-b-0.1 plan "Data and licences"), derived from stage 4's records.

The training sources are the aliases with audio seen in run b100-main's summary (seconds_by_source); none is typed by
hand. Each alias is resolved to a public source:
- a corpus of the pinned training stream (yodas, peoples_speech, librispeech, voxpopuli, ami): its Hub repository,
  revision and licence from finetune/parakeet-ternary (data.HF_SOURCES, stream.LIBRISPEECH);
- a term-filtered subset of one of them (pool.common.SUBSETS: yodas-med, ps-med, yodas-tech): its corpus;
- ami-sdm: AMI's single-distant-microphone recordings (ami_sdm.REPO, REVISION, LICENSE);
- any other alias (cv27-en, med-osce): the private registry ($PRIVATE_REGISTRY/sources.json);
- synthetic: the union of the synthetic runs in the stage 4 batch plan; its parts are the TTS engines named by its
  clips, the noise and room-impulse sources of the augmentation pool (synthetic/noise/pool.json, with every noise
  clip's own licence), the text generator and the term source (synthetic.layout), all through the registry.
Tools that shaped the labels or the data (the label teacher, label cleaning, forced alignment) are listed too.

derive() fails (ProvenanceError) if a training alias cannot be resolved, or if a resolved source or part is not
permissive (a licence with NC, ND or SA, "non-commercial", "research only" or "eval", a registry entry with an
evaluation role, or a dropped entry). The derived mapping is private (provenance.json beside the staging folder);
the published documents get public names, licences, revisions and hours only.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

STREAM_CORPORA = ("yodas", "peoples_speech", "librispeech", "voxpopuli", "ami")
PUBLIC_NAMES = {
    "yodas": "YODAS-Granary, English",
    "peoples_speech": "People's Speech, clean configuration",
    "librispeech": "LibriSpeech train.clean.100, train.clean.360 and train.other.500",
    "voxpopuli": "VoxPopuli, English train",
    "ami": "AMI, individual headset microphones (IHM), train",
    "ami-sdm": "AMI, single distant microphone (SDM), train",
}
# The published licence wording where the recorded licence text carries working notes.
PUBLIC_LICENCES = {"peoples_speech": "CC-BY 2.0, 2.5, 3.0 or 4.0, depending on the source recording"}
SUBSET_NAMES = {"medical": "medical-term-dense subset", "technical": "technical-term-dense subset"}
SYNTHETIC = "synthetic"
_NOT_PERMISSIVE = re.compile(r"(?<![a-z])(nc|nd|sa)(?![a-z])|non-?commercial|research[- ]only|eval[- ]only|"
                             r"evaluation only|fair[- ]use|all rights reserved", re.I)
_PERMISSIVE = re.compile(r"cc0|cc[- ]?by|creativecommons\.org/(licenses/by/|publicdomain/zero)|apache|\bmit\b|"
                         r"public domain|attribution", re.I)


class ProvenanceError(ValueError):
    pass


def permissive(license_text: str | None) -> bool:
    text = str(license_text or "")
    return bool(_PERMISSIVE.search(text)) and not _NOT_PERMISSIVE.search(text)


def _registry_problems(alias: str, entry: dict | None, license_text: str | None = None) -> list[str]:
    if entry is None:
        return [f"{alias}: not in the private registry"]
    problems = []
    lic = entry.get("license") if license_text is None else license_text
    if not permissive(lic):
        problems.append(f"{alias}: licence {lic!r} is not permissive")
    if "role" in entry and re.search(r"dev|eval|test", str(entry["role"]), re.I):
        problems.append(f"{alias}: registry role {entry['role']!r} is evaluation")
    if re.search(r"dropped|rejected", str(entry.get("status", "")), re.I):
        problems.append(f"{alias}: registry status {str(entry['status'])[:60]!r}")
    use = str(entry.get("use", ""))
    if re.search(r"held-out|eval|denylist", use, re.I) and not re.search(r"training", use, re.I):
        problems.append(f"{alias}: registry use {use[:60]!r} is not training")
    return problems


def _registry_public(alias: str, entry: dict, display: dict, problems: list[str]) -> dict:
    """The published fields of a registry source: name, licence and any note from the private display file (the
    registry's own text holds working notes), URL and revision from the registry (the display file may give a
    public revision label); the registry's licence text is kept as license_registry (private record only)."""
    shown = display.get(alias)
    if not shown or not shown.get("name") or not shown.get("license"):
        problems.append(f"{alias}: no public name and licence in the display file")
        shown = {}
    elif shown["license"] != "per clip" and not permissive(shown["license"]):
        problems.append(f"{alias}: display licence {shown['license']!r} is not permissive")
    url = entry.get("url") or (f"https://huggingface.co/{entry['hf_id']}" if entry.get("hf_id") else None)
    return {"name": shown.get("name"), "url": shown.get("url", url), "hf_id": entry.get("hf_id"),
            "revision": shown.get("revision", entry.get("revision")), "license": shown.get("license"),
            "note": shown.get("note"), "license_registry": entry.get("license"), "paper": entry.get("paper")}


def derive(*, run: str, seconds_by_source: dict, plan_sources: list, corpora: dict, subsets: dict, registry: dict,
           synthetic: dict, tools: dict, display: dict) -> dict:
    """The provenance record (module docstring). corpora: {alias: {"repo", "revision", "license", ...}} of the stream
    corpora and ami-sdm; subsets: {alias: (corpus, kind)}; synthetic: {"clips": n, "engines": {alias: clips},
    "noise": {alias: {licence: clips}}, "rirs": {alias: n}, "text": [aliases], "terms": [aliases]}; tools:
    {role: {"name", "hf_id", "revision", "license"} or {"alias": registry alias}}; display: {alias: {"name",
    "license", optional "revision", "url", "note"}} (the private display file)."""
    problems: list[str] = []
    trained = {a: float(s) for a, s in seconds_by_source.items() if float(s) > 0}
    plan = {s.get("config", {}).get("name"): s for s in plan_sources}
    sources = []
    allowed = set(trained)
    for alias in sorted(trained, key=lambda a: -trained[a]):
        row = {"alias": alias, "hours_seen": trained[alias] / 3600}
        if alias in plan:
            row["clips"] = plan[alias].get("n_clips")
            row["draws_per_clip"] = plan[alias].get("k")
        if alias in corpora:
            spec = corpora[alias]
            row.update(kind="corpus", name=PUBLIC_NAMES.get(alias, alias), repo=spec["repo"],
                       url=f"https://huggingface.co/datasets/{spec['repo']}", revision=spec["revision"],
                       license=PUBLIC_LICENCES.get(alias, spec["license"]), license_registry=spec["license"],
                       config=spec.get("config"))
            if not permissive(spec["license"]):
                problems.append(f"{alias}: licence {spec['license']!r} is not permissive")
        elif alias in subsets:
            corpus, kind = subsets[alias]
            if corpus not in corpora:
                problems.append(f"{alias}: subset of {corpus!r}, which is not a known corpus")
                continue
            spec = corpora[corpus]
            row.update(kind="subset", corpus=corpus,
                       name=f"{PUBLIC_NAMES.get(corpus, corpus)}: {SUBSET_NAMES.get(kind, kind)}", repo=spec["repo"],
                       url=f"https://huggingface.co/datasets/{spec['repo']}", revision=spec["revision"],
                       license=PUBLIC_LICENCES.get(corpus, spec["license"]), license_registry=spec["license"],
                       config=spec.get("config"))
            if not permissive(spec["license"]):
                problems.append(f"{alias}: licence {spec['license']!r} is not permissive")
        elif alias == SYNTHETIC:
            row.update(kind="synthetic", name="Synthetic technical speech (generated for this project)",
                       license="generated; see the parts", clips_in_store=synthetic.get("clips"))
            parts = []
            for role, aliases in (("tts", sorted(synthetic.get("engines", {}))), ("text", synthetic.get("text", [])),
                                  ("terms", synthetic.get("terms", [])), ("rir", sorted(synthetic.get("rirs", {}))),
                                  ("noise", sorted(synthetic.get("noise", {})))):
                for part in aliases:
                    entry = registry.get(part)
                    if role == "noise":
                        licences = synthetic["noise"][part]
                        bad = sorted(lic for lic in licences if not permissive(lic))
                        if bad:
                            problems.append(f"synthetic noise {part}: clips under non-permissive licences {bad}")
                        problems += [p for p in _registry_problems(part, entry, license_text="per clip")
                                     if "licence" not in p]
                        extra = {"clip_licences": licences}
                    else:
                        problems += _registry_problems(part, entry)
                        extra = {"clips": synthetic["engines"][part]} if role == "tts" else \
                            {"files": synthetic["rirs"][part]} if role == "rir" else {}
                    if entry is not None:
                        parts.append({"alias": part, "role": role, **_registry_public(part, entry, display, problems),
                                      **extra})
                    allowed.add(part)
            if synthetic.get("clips") != sum(synthetic.get("engines", {}).values()):
                problems.append("synthetic: clips by engine do not add up to the store's clip count")
            if SYNTHETIC in plan and plan[SYNTHETIC].get("n_clips") != synthetic.get("clips"):
                problems.append(f"synthetic: the batch plan has {plan[SYNTHETIC].get('n_clips')} clips, the runs "
                                f"{synthetic.get('clips')}")
            row["parts"] = parts
        elif alias in registry:
            entry = registry[alias]
            problems += _registry_problems(alias, entry)
            row.update(kind="registry", **_registry_public(alias, entry, display, problems))
        else:
            problems.append(f"{alias}: unmapped training alias")
            continue
        sources.append(row)
    tool_rows = []
    for role, spec in tools.items():
        if "alias" in spec:
            entry = registry.get(spec["alias"])
            problems += _registry_problems(spec["alias"], entry)
            if entry is not None:
                tool_rows.append({"role": role, "alias": spec["alias"],
                                  **_registry_public(spec["alias"], entry, display, problems)})
            allowed.add(spec["alias"])
        else:
            if not permissive(spec.get("license")):
                problems.append(f"tool {role}: licence {spec.get('license')!r} is not permissive")
            tool_rows.append({"role": role, **spec})
    if problems:
        raise ProvenanceError("; ".join(problems))
    return {"run": run, "sources": sources, "tools": tool_rows, "allowed_aliases": sorted(allowed),
            "hours_seen_total": sum(trained.values()) / 3600}


def gather(run_summary: Path, run_config: Path, plan_json: Path, registry_json: Path, display_json: Path) -> dict:
    """derive()'s inputs from disk (CPU; reads the synthetic runs' clip lists and the augmentation pool)."""
    import ami_sdm
    import data
    import stream
    from labeling import layout as llayout
    from pool import common
    from synthetic import layout as slayout

    import teachers
    import train_keab

    summary = json.loads(Path(run_summary).read_text())
    config = json.loads(Path(run_config).read_text())
    teacher = config["args"]["teacher"]
    plan = json.loads(Path(plan_json).read_text())
    registry = json.loads(Path(registry_json).read_text())
    corpora = {k: dict(data.HF_SOURCES[k]) for k in ("yodas", "peoples_speech", "voxpopuli", "ami")}
    corpora["librispeech"] = dict(stream.LIBRISPEECH)
    corpora["ami-sdm"] = {"repo": ami_sdm.REPO, "revision": ami_sdm.REVISION, "license": ami_sdm.LICENSE,
                          "config": "sdm"}
    engines: dict[str, int] = {}
    clips = 0
    for run in train_keab.SYNTHETIC_RUNS:
        with (slayout.run_dir(run) / "clips.jsonl").open() as handle:
            for line in handle:
                if line.strip():
                    clip = json.loads(line)
                    engines[clip["engine"]] = engines.get(clip["engine"], 0) + 1
                    clips += 1
    pool = json.loads((slayout.noise_dir() / "pool.json").read_text())
    if not pool.get("train_check", {}).get("pass"):
        raise ProvenanceError("the augmentation pool's TRAIN disjointness check did not pass")
    noise: dict[str, dict[str, int]] = {}
    for c in pool["noise"]:
        noise.setdefault(c["source"], {})
        noise[c["source"]][c.get("license")] = noise[c["source"]].get(c.get("license"), 0) + 1
    rirs: dict[str, int] = {}
    for r in pool["rirs"]:
        rirs[r["source"]] = rirs.get(r["source"], 0) + 1
    granite = next(t for t in vars(teachers).values()
                   if isinstance(t, type) and getattr(t, "name", None) == llayout.TEACHER)
    tools = {
        "encoder teacher (B0′)": {"name": "B0′, an unreleased FP32 fine-tune of nvidia/parakeet-tdt-0.6b-v2 on the "
                                    "same labels", "revision": f"sha256 {teacher['sha256']}", "license": "CC-BY-4.0"},
        "label teacher (T*)": {"name": granite.hf_id, "hf_id": granite.hf_id, "revision": granite.revision,
                               "license": granite.license},
        "label cleaning and synthetic text": {"alias": slayout.TEXTGEN},
        "forced alignment of a training corpus": {"alias": "train-aligner"},
    }
    return {"run": summary["run_name"], "seconds_by_source": summary["seconds_by_source"],
            "plan_sources": plan["sources"], "corpora": corpora, "subsets": dict(common.SUBSETS),
            "registry": registry, "tools": tools, "display": json.loads(Path(display_json).read_text())["sources"],
            "synthetic": {"clips": clips, "engines": engines, "noise": noise, "rirs": rirs,
                          "text": [slayout.TEXTGEN], "terms": list(slayout.TERM_SOURCES)}}


# ----------------------------------------------------------------------------- published text

LICENCE_URLS = {"http://creativecommons.org/licenses/by/3.0/": "CC-BY-3.0",
                "http://creativecommons.org/licenses/by/4.0/": "CC-BY-4.0",
                "http://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0"}


def _hours(h: float) -> str:
    return f"{h:,.0f}"


def _link(name: str, url: str | None) -> str:
    return f"[{name}]({url})" if url else name


def training_table(prov: dict) -> str:
    """The Markdown table of training sources for TRAINING.md."""
    lines = ["| Source | Licence | Revision | Audio seen in training (hours) |", "| --- | --- | --- | ---: |"]
    for s in prov["sources"]:
        revision = (s.get("revision") or "-").replace("|", "/")
        lines.append(f"| {_link(s['name'], s.get('url'))} | {s['license']} | {revision} | {_hours(s['hours_seen'])} |")
    lines.append(f"| **Total** | | | {_hours(prov['hours_seen_total'])} |")
    return "\n".join(lines)


def _part_licence(p: dict) -> str:
    if p["role"] != "noise":
        return p["license"]
    clips = ", ".join(f"{LICENCE_URLS.get(k, k)} ({v:,} clips)" for k, v in sorted(p["clip_licences"].items()))
    return (p["license"] + "; " if p["license"] != "per clip" else "") + "per clip: " + clips


def synthetic_parts(prov: dict) -> str:
    """Bullet list of the synthetic source's parts (TRAINING.md and NOTICE)."""
    synth = next((s for s in prov["sources"] if s["kind"] == "synthetic"), None)
    if synth is None:
        return "- (no synthetic audio)"
    roles = {"tts": "Text-to-speech", "text": "Sentences", "terms": "Technical term lists",
             "rir": "Room impulse responses (augmentation)", "noise": "Background noise (augmentation)"}
    lines = []
    for p in synth["parts"]:
        count = f", {p['clips']:,} clips" if p.get("clips") else ""
        lines.append(f"- {roles[p['role']]}: {p['name']} ({p.get('url') or '-'}, {p.get('revision') or '-'}); "
                     f"{_part_licence(p)}{count}.")
    return "\n".join(lines)


def notice_data(prov: dict) -> str:
    """The training-data section of NOTICE: every source with its URL, revision and licence."""
    out = []
    for s in prov["sources"]:
        if s["kind"] == "synthetic":
            out.append(f"- {s['name']}: {s.get('clips_in_store'):,} clips, made with:")
            out += ["  " + line for line in synthetic_parts(prov).splitlines()]
            continue
        note = f" {s['note']}" if s.get("note") else ""
        paper = f" Reference: {s['paper']}." if s.get("paper") else ""
        out.append(f"- {s['name']} ({s.get('url') or '-'}, revision {s.get('revision') or '-'}), "
                   f"{s['license']}.{note}{paper}")
    return "\n".join(out)


def notice_tools(prov: dict) -> str:
    out = []
    for t in prov["tools"]:
        where = t.get("url") or (f"https://huggingface.co/{t['hf_id']}" if t.get("hf_id") else None)
        loc = f"{where}, revision {t.get('revision')}" if where else f"{t.get('revision')}"
        out.append(f"- {t['role'][0].upper() + t['role'][1:]}: {t.get('name')} ({loc}), {t.get('license')}.")
    return "\n".join(out)


def card_datasets(prov: dict) -> list[str]:
    """Hub dataset ids of the trained-on corpora that live on the Hub (the card's datasets metadata)."""
    seen = []
    for s in prov["sources"]:
        if s["kind"] in ("corpus", "subset") and s["repo"] not in seen:
            seen.append(s["repo"])
    return seen
