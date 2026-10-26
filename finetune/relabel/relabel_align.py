"""Word alignment with the backtrace of evaluate.edit_counts, and error events (diagnostic plan Step 2).

`align(ref, hyp)` runs the same dynamic program as finetune/whisper-ternary/wer.py `edit_counts`
(imported by finetune/parakeet-ternary/evaluate.py, which produced the recorded S, D, I) and walks
back from the end with the same tie-breaking: diagonal (match or substitution) first, then deletion,
then insertion. Its S, D, I are therefore those edit_counts returns.

Error events of one utterance, as a multiset (collections.Counter), without the utterance id (events of
different utterances never meet, so callers combine counts per utterance):

    ("sub", ref index, hyp word)    ref word ref[k] replaced by hyp word
    ("del", ref index, "")          ref word ref[k] dropped
    ("ins", gap index, hyp word)    hyp word inserted in gap g, between ref[g-1] and ref[g] (0..len(ref))

Repeated identical insertions in one gap count separately.

`pairs` lists the backtrace's steps in hypothesis order: ("eq" | "sub", ref index, hyp index), ("del", ref index,
None), ("ins", gap index, hyp index).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Alignment:
    S: int
    D: int
    I: int
    events: Counter = field(compare=False)
    matched: frozenset   # ref indices aligned as a match
    ins_gaps: frozenset  # gap indices holding at least one insertion
    pairs: tuple = field(default=(), compare=False)

    @property
    def errors(self) -> int:
        return self.S + self.D + self.I


def align(ref: list[str], hyp: list[str]) -> Alignment:
    rows, cols = len(ref) + 1, len(hyp) + 1
    cost = [[i + j if i == 0 or j == 0 else 0 for j in range(cols)] for i in range(rows)]
    for i in range(1, rows):
        prev, cur, r = cost[i - 1], cost[i], ref[i - 1]
        for j in range(1, cols):
            cur[j] = min(prev[j - 1] + (r != hyp[j - 1]), prev[j] + 1, cur[j - 1] + 1)
    events: Counter = Counter()
    matched, gaps, steps = set(), set(), []
    s = d = n_ins = 0
    i, j = rows - 1, cols - 1
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                s += 1
                events[("sub", i - 1, hyp[j - 1])] += 1
                steps.append(("sub", i - 1, j - 1))
            else:
                matched.add(i - 1)
                steps.append(("eq", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            d += 1
            events[("del", i - 1, "")] += 1
            steps.append(("del", i - 1, None))
            i -= 1
        else:
            n_ins += 1
            events[("ins", i, hyp[j - 1])] += 1
            gaps.add(i)
            steps.append(("ins", i, j - 1))
            j -= 1
    return Alignment(s, d, n_ins, events, frozenset(matched), frozenset(gaps), tuple(reversed(steps)))


def fixed_by(event: tuple, teacher: Alignment) -> bool:
    """Whether the teacher fully corrects the position of an event: a ref position it aligns as a match
    (substitution, deletion), or a gap where it inserts nothing (insertion)."""
    op, index, _ = event
    if op == "ins":
        return index not in teacher.ins_gaps
    return index in teacher.matched


def shared(a: Counter, b: Counter) -> Counter:
    """Multiset intersection (minimum counts)."""
    return a & b


def only_in(a: Counter, b: Counter) -> Counter:
    """Multiset difference a \\ b (subtracted counts, never negative)."""
    return a - b


def size(events: Counter) -> int:
    return sum(events.values())
