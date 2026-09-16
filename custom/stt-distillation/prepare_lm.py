"""Download, validate casing, and optionally compile the existing public-domain LM."""

import argparse
import gzip
import json
import subprocess
import urllib.request
from pathlib import Path

from common import digest, save
from recovery_core import ROOT, artifact


def prepare(build_binary=None):
    folder = artifact(ROOT / "models/slr11")
    folder.mkdir(parents=True, exist_ok=True)
    name = "3-gram.pruned.3e-7.arpa.gz"
    url = "https://www.openslr.org/resources/11/" + name
    path = folder / name
    if not path.exists():
        pending = path.with_suffix(".pending")
        with (
            urllib.request.urlopen(url, timeout=60) as response,
            pending.open("wb") as out,
        ):
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        pending.replace(path)
    original = folder / "original.arpa"
    lower = folder / "lowercase.arpa"
    mapping, unigrams = {}, []
    order = 0
    with (
        gzip.open(path, "rt") as src,
        original.open("w") as orig,
        lower.open("w") as dst,
    ):
        for line in src:
            orig.write(line)
            stripped = line.strip()
            if stripped.startswith("\\") and "-grams:" in stripped:
                order = int(stripped[1:].split("-")[0])
            elif stripped.startswith("\\"):
                order = 0
            if order and stripped and not stripped.startswith("\\"):
                parts = stripped.split()
                words = parts[1 : order + 1]
                converted = [w.lower() for w in words]
                if order == 1:
                    old, new = words[0], converted[0]
                    if new in mapping and mapping[new] != old:
                        raise ValueError(
                            "Case folding collides: " + old + " / " + mapping[new]
                        )
                    mapping[new] = old
                    if not new.startswith("<"):
                        unigrams.append(new)
                parts[1 : order + 1] = converted
                line = (
                    "\t".join(
                        [parts[0], " ".join(parts[1 : order + 1])] + parts[order + 1 :]
                    )
                    + "\n"
                )
            dst.write(line)
    (folder / "unigrams.txt").write_text("\n".join(unigrams) + "\n")
    # Verify the bijection preserves complete sentence scores, not just vocabulary.
    import kenlm

    a, b = kenlm.Model(str(original)), kenlm.Model(str(lower))
    test_sentences = [
        "the patient is breathing",
        "one two zero zero",
        "i am here",
        "a strange unseenwordtoken",
    ]
    diffs = []
    for sentence in test_sentences:
        src = " ".join(mapping.get(w, w) for w in sentence.split())
        diffs.append(abs(a.score(src) - b.score(sentence)))
    assert max(diffs) < 1e-5
    binary = None
    if build_binary:
        binary = folder / "lm.binary"
        subprocess.run([str(build_binary), "trie", str(lower), str(binary)], check=True)
        c = kenlm.Model(str(binary))
        assert all(abs(c.score(s) - b.score(s)) < 1e-5 for s in test_sentences)
    result = dict(
        source=url,
        license="Public domain (OpenSLR SLR11)",
        source_sha256=digest(path),
        arpa=str(lower),
        arpa_sha256=digest(lower),
        case_collisions=0,
        max_case_score_difference=max(diffs),
        unigrams=len(unigrams),
        binary=str(binary) if binary else None,
        binary_bytes=binary.stat().st_size if binary else None,
        compressed_bytes=path.stat().st_size,
    )
    save(folder / "source.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-binary", type=Path)
    a = p.parse_args()
    prepare(a.build_binary)
