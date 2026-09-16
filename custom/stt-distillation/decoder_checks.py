"""Check CTC path summation/repeats, LM-off control and cached greedy parity."""

import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from pyctcdecode import build_ctcdecoder

from common import save
from decode_ctc import greedy, choose
from recovery_core import ROOT, read, source_hashes


def main():
    labels = ["", "a", "b"]
    probs = np.array(
        [[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.7, 0.2, 0.1], [0.2, 0.4, 0.4]]
    )
    exact = defaultdict(float)
    for path in itertools.product(range(3), repeat=len(probs)):
        text = "".join(
            labels[c] for j, c in enumerate(path) if c and (j == 0 or c != path[j - 1])
        )
        exact[text] += math.prod(probs[j, c] for j, c in enumerate(path))
    decoder = build_ctcdecoder(labels)
    beams = decoder.decode_beams(
        np.log(probs),
        beam_width=128,
        beam_prune_logp=-100,
        token_min_logp=-100,
        prune_history=False,
    )
    assert beams[0][0] == max(exact, key=exact.get)
    for b in beams:
        assert abs(b[3] - math.log(exact[b[0]])) < 1e-5, (b, exact[b[0]])
    repeated = np.full((3, 3), -100.0)
    repeated[[0, 1, 2], [1, 0, 1]] = 0
    assert decoder.decode(repeated) == "aa"
    silent = np.full((5, 3), -100.0)
    silent[:, 0] = 0
    assert decoder.decode(silent) == ""
    lm = read(ROOT / "models/slr11/source.json")
    assert lm["max_case_score_difference"] < 1e-5 and lm["case_collisions"] == 0
    cache = ROOT / "setup/baseline-smoke"
    index = read(cache / "index.json")
    for row in index["entries"]:
        z = np.load(cache / row["file"], allow_pickle=False)
        assert greedy(z, index["labels"], index["blank_id"]) == row["greedy"]
    # An LM can improve word error while damaging numbers. Reject that regression.
    case = read(ROOT / "evaluation/ternary14k-preflight/summary.json")
    decision = choose(case["results"])
    assert not decision["lm_screen_gate_met"]
    assert decision["selected"].startswith("beam-")
    save(
        ROOT / "setup/decoder-checks.json",
        dict(
            passed=True,
            contracts=[
                "exhaustive_CTC_path_scores",
                "blank_separated_repeats",
                "all_blank_input",
                "LM_case_score_equivalence",
                "NeMo_BPE_cached_greedy_parity",
            ],
            baseline_examples=len(index["entries"]),
            source_hashes=source_hashes(Path(__file__).parent),
        ),
    )
    print(
        json.dumps(dict(passed=True, baseline_examples=len(index["entries"]))),
        flush=True,
    )


if __name__ == "__main__":
    main()
