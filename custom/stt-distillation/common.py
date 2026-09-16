"""Shared, versioned contracts for the eight-hour pilot."""

import hashlib, json, os, re, time
from pathlib import Path

ART = Path("/mnt/hd/wilderness-labs-stt/stt-distillation")
REPO = Path(
    os.environ.get(
        "WILDERNESS_STT_ROOT",
        "/home/aj/workspace/github.com/wilderness-labs/wilderness-labs-stt",
    )
)
ALPHABET = "_abcdefghijklmnopqrstuvwxyz' 0123456789.%/+-"
WORDS = "zero one two three four five six seven eight nine".split()


def storage():
    if not os.path.ismount("/mnt/hd"):
        raise RuntimeError("/mnt/hd is not mounted")
    ART.mkdir(parents=True, exist_ok=True)


def save(p, obj):
    storage()
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(p)


def digest(p):
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9' .%/+-]", " ", s.lower()).split())


def encode(s):
    return [ALPHABET.index(c) for c in norm(s)]


def decode(ids):
    out = []
    last = None
    for i in ids:
        if i != last and i:
            out.append(ALPHABET[i])
        last = i
    return norm("".join(out))


def distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        row = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            row[j] = min(row[j - 1] + 1, prev[j] + 1, prev[j - 1] + (x != y))
        prev = row
    return prev[-1]


def digit_string(s):
    # Only the controlled digit-sequence corpus uses this conversion.
    out = []
    for w in re.findall(r"[a-z]+|[0-9]+", s.lower()):
        if w in WORDS:
            out.append(str(WORDS.index(w)))
        elif w.isdigit():
            out.append(w)
        else:
            out.append("?")
    return "".join(out)


def scores(pairs):
    we = wc = ce = cc = empty = 0
    for r, h in pairs:
        r, h = norm(r), norm(h)
        we += distance(r.split(), h.split())
        wc += len(r.split())
        ce += distance(r, h)
        cc += len(r)
        empty += not h
    return dict(
        wer=we / wc if wc else None,
        cer=ce / cc if cc else None,
        word_errors=we,
        reference_words=wc,
        char_errors=ce,
        reference_chars=cc,
        empty_outputs=empty,
        utterances=len(pairs),
    )


def read_manifest():
    p = Path(
        os.environ.get(
            "WILDERNESS_STT_MANIFEST", str(ART / "datasets/pilot/manifest.json")
        )
    )
    assert p.is_relative_to(ART)
    obj = json.loads(p.read_text())
    for r in obj["rows"]:
        if not Path(r["audio"]).is_relative_to(ART) or not Path(
            r["features"]
        ).is_relative_to(ART):
            raise ValueError("Artifact escaped data disk")
    return obj


def medical_terms(pairs):
    import collections

    terms = json.loads(Path(__file__).with_name("medical_terms.json").read_text())[
        "terms"
    ]
    counts = {t: dict(reference=0, predicted=0, correct=0) for t in terms}
    for ref, hyp in pairs:
        for t in terms:
            pattern = r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])"
            r = len(re.findall(pattern, norm(ref)))
            h = len(re.findall(pattern, norm(hyp)))
            counts[t]["reference"] += r
            counts[t]["predicted"] += h
            counts[t]["correct"] += min(r, h)
    nr = sum(v["reference"] for v in counts.values())
    nh = sum(v["predicted"] for v in counts.values())
    nc = sum(v["correct"] for v in counts.values())
    return dict(
        reference_mentions=nr,
        predicted_mentions=nh,
        correct_mentions=nc,
        exact_lexical_recall=nc / nr if nr else None,
        exact_lexical_precision=nc / nh if nh else None,
        per_term=counts,
        scope="Exact glossary mentions only, no semantic or clinical adjudication; zero reference coverage is not success.",
    )
