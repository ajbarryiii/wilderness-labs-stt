"""Domain dev sets dev-medical and dev-technical (labeling plan Part A): fetch, segment, build manifests.

Evaluation only, never trained on. Parts are referred to by alias; the private registry
($PRIVATE_REGISTRY/sources.json, read at run time by registry.py) maps each alias to its source, revision,
license and acquisition.

    dev-medical    med-dev-1 (two-channel consultations, source utterance timestamps)
                   med-dev-2 (dictation, dev split only, several recording conditions; aligned)
                   med-dev-3 (pre-cut medical-term clips)
    dev-technical  tech-dev-1 (pre-cut talk segments, two splits)
                   tech-dev-2 (long-form calls; aligned)
    eval-ext       medical-b (pre-cut clips of recorded medical talks, test split; no id column: row ids)
                   lectures-a (captioned health and biology lectures from a web archive; aligned)

    eval-stem      stem-chem, stem-math, stem-cs (captioned chemistry, mathematics and computer-science lectures
                   from the same web archive, one part per subject; aligned)

eval-ext and eval-stem are further held-out sets (eval only, never trained on; Part B's denylist reads every manifest
under $RELABEL_ROOT/devsets/), scored only on request (score_domain.py --sets eval-ext / eval-stem).

Modules: layout (paths), registry (sources.json), fetch (pinned downloads), sources (readers), transcript
(tags, alignment units), align (MMS forced aligner), segment (the segmenter, reused by Part B), build (CLI).
Manifests: $RELABEL_ROOT/devsets/manifests/<part>.jsonl in the existing per-utterance schema
(audio_filepath, duration, text, id, source) plus "part", with <part>.meta.json (counts, hours, drops by
reason, source revision). Audio: $RELABEL_ROOT/devsets/audio/<part>/<recording>/<segment>.flac.
"""
from __future__ import annotations

SETS = {
    "dev-medical": ("med-dev-1", "med-dev-2", "med-dev-3"),
    "dev-technical": ("tech-dev-1", "tech-dev-2"),
    "eval-ext": ("medical-b", "lectures-a"),
    "eval-stem": ("stem-chem", "stem-math", "stem-cs"),
}
DEFAULT_SETS = ("dev-medical", "dev-technical")  # the Part A domain dev sets; eval-ext is scored on request
PARTS = tuple(part for parts in SETS.values() for part in parts)
TERM_KIND = {"dev-medical": "medical", "eval-ext": "medical", "eval-stem": "technical"}  # term list per set


def set_of(part: str) -> str:
    for name, parts in SETS.items():
        if part in parts:
            return name
    raise ValueError(f"unknown dev-set part {part!r}; expected one of {PARTS}")
