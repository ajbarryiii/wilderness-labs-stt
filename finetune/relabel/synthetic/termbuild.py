"""Writes the committed term lists terms/synthetic/<domain>.txt (synthetic plan "Terms"); run once, by hand.

    ./python -m synthetic termbuild [--fetch]

Sources per domain, merged in this order of precedence:
1. core: synthetic/core/<domain>.txt, written from general domain knowledge (like Part A's lists);
2. existing: the program's technical lists (terms/technical/*.txt), whole (finance, computing) or the subset of
   science and engineering terms listed in EXISTING_SUBSETS;
3. kb: a CC0 knowledge base (alias syn-terms-kb; endpoint, query template and per-domain anchors are in the
   private registry). Each anchor returns items (label, popularity); the most popular `cap` items per anchor
   with at least `min_popularity` are candidates. --fetch runs the queries (network) and stores each response
   under $RELABEL_ROOT/synthetic/terms/raw/syn-terms-kb/ with its SHA-256; without it the stored responses are
   used.

Filters (all sources unless noted): written form made of letters, spaces, hyphens and apostrophes only (dashes and
curly quotes mapped first), 1 to 5 words, no "list of"/"history of"/"timeline of"; kb and core single-word terms
are dropped if the word is an everyday word: more than EVERYDAY_PER_MILLION occurrences per million words of
general transcribed speech (a fixed sample of a training source's normalized transcripts). A normalized phrase
is written once, in the first domain whose highest-precedence source lists it. Hub terms are left out: a term
nested (in plural_key form, as a contiguous sub-phrase) in more than HUB_MAX other terms of the lists, e.g. a
bare "theorem" inside every named theorem, since the split's variant grouping would otherwise merge all of
them into one group and decide them together.

The split (synthetic/vocab.py) applies its own normalization, medical / TCCC exclusion, cross-domain
de-duplication, grouping and the fixed-hash split to these files; this builder does not split.
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import terms as committed_terms
from synthetic import layout, vocab

HERE = Path(__file__).resolve().parent
CORE_DIR = HERE / "core"
KB = "syn-terms-kb"
EVERYDAY_PER_MILLION = 20.0
HUB_MAX = 8
FREQ_FILES = 40            # first N text-cache files of the general-speech sample
FREQ_SOURCE = "peoples_speech"
USER_AGENT = "relabel-synthetic-termbuild/1 (speech-recognition research term lists)"

EXISTING_WHOLE = {"finance": ("technical", "finance"), "software_engineering": ("technical", "computing")}
EXISTING_SUBSETS = {
    "chemistry": {
        "science": ["atom", "atoms", "atomic", "molecule", "molecules", "molecular", "isotope", "isotopes",
                    "spectroscopy", "spectrometer", "mass spectrometry", "chromatography", "entropy", "enthalpy",
                    "chemistry", "chemical reaction", "catalyst", "catalysts", "oxidation", "reduction reaction",
                    "polymer", "polymers", "organic chemistry", "inorganic chemistry", "biochemistry", "hydrogen",
                    "oxygen atom", "nitrogen", "helium", "carbon atom", "silicon dioxide", "sodium chloride", "ph",
                    "acidity", "nanoparticles", "carbon dioxide", "methane", "electron", "electrons", "proton",
                    "protons", "neutron", "neutrons", "radioactive", "radioactivity", "half-life",
                    "quantum mechanics", "kinetic energy", "potential energy"],
        "engineering": ["thermodynamics", "thermal conductivity", "corrosion", "fuel cell", "lithium-ion",
                        "lithium ion battery", "alloy", "alloys"],
    },
    "mathematics": {
        "science": ["theorem", "theorems", "lemma", "equation", "equations", "differential equation",
                    "linear algebra", "matrix", "matrices", "eigenvalue", "eigenvalues", "eigenvector",
                    "vector space", "calculus", "derivative", "derivatives", "integrals", "probability distribution",
                    "probability", "statistics", "statistical", "statistically significant", "standard deviation",
                    "variance", "correlation", "regression analysis", "p-value", "confidence interval", "bayesian",
                    "gaussian", "normal distribution", "logarithm", "exponential", "polynomial", "topology",
                    "geometry", "algebra"],
    },
}
# Knowledge-base items left out by hand after review (off-domain: biology, medicine, physics, economics, general
# computing UI; or everyday senses). Compared case-insensitively with the written form.
KB_EXCLUDE = {
    "chemistry": {
        "transcription", "post-transcriptional modification", "glycation", "light-dependent reactions",
        "rancidification", "spontaneous combustion", "contrail", "heat death of the universe", "arrow of time",
        "black hole thermodynamics", "combined cycle power station", "soil pH", "alcohol by volume",
        "global warming potential", "acceptable daily intake", "membrane potential", "resting potential",
        "halocline", "fineness", "weldability", "ash", "temperature", "drying", "energy efficiency",
        "information entropy", "Maxwell's demon", "ultraviolet catastrophe", "laser cooling", "Edman degradation",
        "Strecker amino-acid synthesis", "nitrification", "Xanthoproteic acid", "eponymous chemical reaction",
        "magic number", "residue", "temperature measurement", "Soret Peak", "deflagration", "primary energy",
        "Nusselt number for heat transfer", "intensive or extensive property", "volumetric number density",
        "Hydrophobic effect", "cetane number", "iodine value", "fire point", "inertness", "temperature coefficient",
        "saponification value", "bioorthogonal chemistry", "Maillard reaction", "Belousov-Zhabotinsky reaction",
        "Briggs-Rauscher reaction", "kill screen"},
    "mathematics": {
        "infinite monkey theorem", "charge-parity-time symmetry", "Coase theorem", "Arrow's impossibility theorem",
        "law of noncontradiction", "law of identity", "Minimax", "converse", "temperature anomaly", "energy mix",
        "per capita", "friendship paradox", "Bristol-Fisher-Roach randomised tasting trial",
        "Law of truly large numbers", "null result", "virial theorem", "Ehrenfest theorem", "parallel axis theorem",
        "Torricelli's law", "statistical physics", "mode", "odds", "deviation", "statistic", "scalar", "filtration",
        "risk matrix", "four color theorem", "Waring's problem", "Catalan's conjecture", "Fermat's Last Theorem",
        "Banach-Tarski paradox", "classification of finite simple groups", "latent space", "level of measurement",
        "modus tollens", "multi-armed bandit", "Zipf's law", "Benford's law", "Wigner quasiprobability distribution",
        "statistical graphics", "imputation"},
    "finance": {
        "intellectual property", "personal property", "state property", "public property", "real property",
        "commercial property", "digital asset", "Great Resignation", "Big Mac Index", "Index of Economic Freedom",
        "living wage", "Baltic Dry Index", "SICAV", "vulture fund", "beta", "returns", "margin", "too big to fail",
        "labor intensity", "social return on investment", "genuine progress indicator", "bank statement",
        "foreign reserve", "Phillips curve", "gross national product", "gross national income", "national income",
        "price level", "Open-end fund", "investment company", "financial transaction", "Tobin's q"},
    "software_engineering": {
        "factory", "scheduling", "lock", "inode", "polygon mesh", "kill screen", "Linux kernel oops",
        "computer virus", "directory", "menu", "tab", "method", "thread", "plug-in", "breadcrumb navigation",
        "drop-down list", "virtual keyboard", "context menu", "over-the-air update", "bogosort", "pancake sorting",
        "heuristic", "vulnerability", "return-oriented programming", "formal language", "Petri net", "software pipeline",
        "automatic programming", "computer programming", "software development", "modeling language",
        "cocktail shaker sort", "odd-even sort", "comb sort", "entity component system", "Software inspection"},
}
_MAP = {"–": "-", "—": "-", "‐": "-", "‑": "-", "−": "-", "’": "'", "‘": "'"}
_OK = re.compile(r"[A-Za-z][A-Za-z' -]*[A-Za-z]")
_SKIP_PREFIX = ("list of ", "history of ", "timeline of ", "glossary of ", "outline of ")


def clean(term: str) -> str | None:
    """The written form kept in a list, or None if the term fails the form filters."""
    t = "".join(_MAP.get(c, c) for c in term).strip()
    t = re.sub(r"\s+", " ", t)
    if not _OK.fullmatch(t) or not 1 <= len(t.split()) <= 5 or len(t) > 60:
        return None
    if t.lower().startswith(_SKIP_PREFIX):
        return None
    return t


def everyday_frequency(files: int = FREQ_FILES) -> collections.Counter:
    """Word counts (normalized words) over the first `files` cached transcript files of the general sample."""
    import pyarrow.parquet as pq
    d = layout.relabel_root() / "pool" / "text" / FREQ_SOURCE
    paths = sorted(d.glob("*.parquet"))[:files]
    if len(paths) < files:
        raise RuntimeError(f"{d}: expected at least {files} cached transcript files")
    counts = collections.Counter()
    for p in paths:
        for norm in pq.read_table(p, columns=["norm"]).column("norm").to_pylist():
            counts.update((norm or "").split())
    return counts


def raw_dir() -> Path:
    return layout.terms_dir() / "raw" / KB


def kb_query(entry: dict, anchor: dict) -> str:
    return entry["prefixes"] + "\n" + entry["template"].format(**anchor)


def _get(url: str, attempts: int = 8) -> bytes:
    for attempt in range(attempts):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                   "Accept": "application/sparql-results+json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except OSError as error:   # rate limits (429) and transient failures: back off
            wait = min(300, 15 * 2 ** attempt)
            print(f"kb: {error}; retry in {wait} s", flush=True)
            time.sleep(wait)
    raise RuntimeError("knowledge-base query failed")


def kb_fetch(pause_s: float = 15.0) -> None:
    """Run every anchor's query (skipping anchors already stored), one at a time with a pause."""
    entry = layout.source(KB)
    out = raw_dir()
    out.mkdir(parents=True, exist_ok=True)
    for domain, anchors in entry["anchors"].items():
        for i, anchor in enumerate(anchors):
            q = kb_query(entry, anchor)
            qsha = hashlib.sha256(q.encode()).hexdigest()
            path = raw_path(domain, qsha)
            if path.exists():
                continue
            body = _get(entry["endpoint"] + "?" + urllib.parse.urlencode({"query": q}))
            rows = json.loads(body)["results"]["bindings"]
            rec = {"domain": domain, "anchor": anchor, "query_sha256": qsha,
                   "response_sha256": hashlib.sha256(body).hexdigest(), "rows": len(rows),
                   "fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "items": [{"id": b["x"]["value"].rsplit("/", 1)[-1], "label": b["label"]["value"],
                              "popularity": int(b["n"]["value"])} for b in rows]}
            path.write_text(json.dumps(rec, indent=0, ensure_ascii=False) + "\n")
            print(f"kb {domain} anchor {i}: {len(rows)} items", flush=True)
            time.sleep(pause_s)


def raw_path(domain: str, query_sha256: str) -> Path:
    """One stored response per (domain, query), named by the query's SHA-256."""
    return raw_dir() / f"{domain}-{query_sha256[:16]}.json"


def kb_candidates(domain: str) -> list[str]:
    entry = layout.source(KB)
    out = []
    for anchor in entry["anchors"][domain]:
        path = raw_path(domain, hashlib.sha256(kb_query(entry, anchor).encode()).hexdigest())
        if not path.exists():
            raise RuntimeError(f"{path} missing (run with --fetch)")
        items = json.loads(path.read_text())["items"]
        items = [it for it in items if it["popularity"] >= anchor["min_popularity"]]
        items.sort(key=lambda it: (-it["popularity"], it["label"]))
        seen = set()
        exclude = {x.casefold() for x in KB_EXCLUDE.get(domain, ())}
        for it in items:
            if it["id"] in seen or it["label"].casefold() in exclude:
                continue
            seen.add(it["id"])
            out.append(it["label"])
            if len(seen) >= anchor["cap"]:
                break
    return out


def existing(domain: str) -> list[str]:
    if domain in EXISTING_WHOLE:
        kind, cat = EXISTING_WHOLE[domain]
        return committed_terms.read_file(committed_terms.HERE / kind / f"{cat}.txt")
    out = []
    for cat, wanted in EXISTING_SUBSETS.get(domain, {}).items():
        have = {t.casefold(): t for t in committed_terms.read_file(committed_terms.HERE / "technical" / f"{cat}.txt")}
        missing = [w for w in wanted if w.casefold() not in have]
        if missing:
            raise RuntimeError(f"terms/technical/{cat}.txt lacks {missing}")
        out += [have[w.casefold()] for w in wanted]
    return out


def build(fetch: bool = False, lists_dir: Path | None = None) -> dict:
    layout.require_mount()
    if fetch:
        kb_fetch()
    normalize = vocab.default_normalize()
    freq = everyday_frequency()
    total = sum(freq.values())
    taken: set[tuple[str, ...]] = set()
    report = {}
    per_domain = {d: {"core": [], "existing": [], "kb": []} for d in layout.DOMAINS}
    sources = {d: {"core": committed_terms.read_file(CORE_DIR / f"{d}.txt"), "existing": existing(d),
                   "kb": kb_candidates(d)} for d in layout.DOMAINS}
    rejected = collections.Counter()
    for src in ("core", "existing", "kb"):          # precedence: core, then existing, then kb
        for domain in layout.DOMAINS:
            for raw in sources[domain][src]:
                t = clean(raw)
                if t is None:
                    rejected[f"{src}:form"] += 1
                    continue
                phrase = tuple(normalize(t).split())
                if not phrase:
                    rejected[f"{src}:empty"] += 1
                    continue
                if src != "existing" and len(phrase) == 1 and freq[phrase[0]] * 1e6 / total > EVERYDAY_PER_MILLION:
                    rejected[f"{src}:everyday"] += 1
                    continue
                if phrase in taken:
                    rejected[f"{src}:duplicate"] += 1
                    continue
                taken.add(phrase)
                per_domain[domain][src].append(t)
    keys = {}
    for domain in layout.DOMAINS:
        for src in ("core", "existing", "kb"):
            for t in per_domain[domain][src]:
                keys[(domain, src, t)] = vocab.phrase_key(tuple(normalize(t).split()))
    distinct = set(keys.values())
    nested_in = collections.Counter()
    for k in distinct:
        n = len(k)
        for i in range(n):
            for j in range(i + 1, n + 1):
                sub = k[i:j]
                if sub != k and sub in distinct:
                    nested_in[sub] += 1
    hubs = []
    for (domain, src, t), k in keys.items():
        if nested_in[k] > HUB_MAX:
            per_domain[domain][src].remove(t)
            hubs.append({"term": t, "domain": domain, "nested_in": nested_in[k]})
            rejected[f"{src}:hub"] += 1
    base = vocab.LISTS_DIR if lists_dir is None else Path(lists_dir)
    base.mkdir(parents=True, exist_ok=True)
    for domain in layout.DOMAINS:
        lines = [f"# Stage 2c term list: {domain.replace('_', ' ')} (synthetic plan \"Terms\"). Written by",
                 "# synthetic/termbuild.py; the split (targeted / held out) is synthetic/vocab.py's, not this file's.",
                 "# Sections: core vocabulary written from general domain knowledge; terms from the program's",
                 f"# technical lists; terms from a CC0 knowledge base (alias {KB}, private registry). Everyday",
                 "# single words (frequent in general transcribed speech) are left out on purpose."]
        for src, title in (("core", "core vocabulary"), ("existing", "from terms/technical"),
                           ("kb", f"from the knowledge base ({KB})")):
            lines.append(f"# --- {title}")
            lines += per_domain[domain][src]
        (base / f"{domain}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        report[domain] = {src: len(v) for src, v in per_domain[domain].items()}
    report["rejected"] = dict(sorted(rejected.items()))
    report["hubs"] = hubs
    report["everyday_per_million"] = EVERYDAY_PER_MILLION
    report["frequency_sample_words"] = total
    out = layout.terms_dir() / "termbuild.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    return report
