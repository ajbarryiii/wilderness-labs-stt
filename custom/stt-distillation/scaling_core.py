"""Pure contracts for paired unique-recording / compute experiments."""

import collections
import hashlib
import math

import numpy as np

from common import ART, digit_string, norm, scores
from recovery_core import DOMAINS

ROOT = ART / "data-scaling"


def rank(seed, value):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def bound(row, edges):
    # Feature frames are 10 ms hops. The left FFT padding adds no frames.
    frames = row["frames"] - (20 if row["domain"] == "digits" else 0)
    if row["domain"] != "general":
        return frames
    return next(int(e * 100) for e in edges[1:] if frames <= int(e * 100))


def groups_for_seed(rows, anchors, seed, cfg):
    """Each slot holds up to eight same-speaker, same-duration-bin recordings.

    Anchor singletons are explicitly invariant if a cell cannot supply eight
    alternatives. Other incomplete groups are excluded, equally across tiers.
    Never select these groups using evaluation outcomes.
    """
    k = max(cfg["sizes"].values())
    cells = collections.defaultdict(list)
    for r in rows:
        if r["domain"] == "general":
            cells[str(r["speaker"]), bound(r, cfg["duration_edges_seconds"])].append(r)
    groups, excluded = [], []
    for (speaker, ceiling), cell in sorted(cells.items()):
        anchored = sorted((r for r in cell if r["id"] in anchors), key=lambda r: r["id"])
        rest = sorted((r for r in cell if r["id"] not in anchors), key=lambda r: rank(seed, r["id"]))
        for anchor in anchored:
            companions = rest[:k - 1] if len(rest) >= k - 1 else []
            rest = rest[len(companions):]
            groups.append(dict(domain="general", speaker=speaker, bound=ceiling,
                               members=[anchor["id"]] + [r["id"] for r in companions]))
        for start in range(0, len(rest) - k + 1, k):
            groups.append(dict(domain="general", speaker=speaker, bound=ceiling,
                               members=[r["id"] for r in rest[start:start + k]]))
        excluded.extend(r["id"] for r in rest[len(rest) // k * k:])
    for r in sorted(rows, key=lambda r: r["id"]):
        if r["domain"] != "general":
            groups.append(dict(domain=r["domain"], speaker=r.get("speaker"),
                               bound=bound(r, cfg["duration_edges_seconds"]), members=[r["id"]]))
    used = [i for g in groups for i in g["members"]]
    assert len(used) == len(set(used))
    assert anchors <= set(used), "Required warm-start / monitor recording was excluded"
    return groups, excluded


def common_tape(groups, seed, cfg):
    """Columns: slot, visit, prefix, gain index, padded length.

    Domain counts are exact in each ten-example block. Slots are shuffled
    without replacement in each domain; alternatives cycle without replacement.
    The first visit is shared across sizes, then repetition is replaced by new
    recordings. RNGs are independent of tier and model evaluation.
    """
    rng = np.random.default_rng(seed)
    aug = np.random.default_rng(seed + 171)
    slots = {d: [i for i, g in enumerate(groups) if g["domain"] == d] for d in DOMAINS}
    assert all(slots.values())
    queues = {d: [] for d in DOMAINS}
    visits = collections.Counter()
    domain_block = [d for d, n in cfg["domain_counts_per_block"].items() for _ in range(n)]
    pending = []
    count = max(cfg["checkpoint_updates"]) * cfg["batch_size"]
    tape = np.empty((count, 5), dtype=np.int32)
    for j in range(count):
        if not pending:
            pending = rng.permutation(domain_block).tolist()
        d = pending.pop()
        if not queues[d]:
            queues[d] = rng.permutation(slots[d]).tolist()
        slot = queues[d].pop()
        if aug.random() < cfg["original_probability"]:
            prefix, gain = (20 if d == "digits" else 0), 0
        else:
            prefix = int(aug.integers(cfg["maximum_prefix_frames"] + 1))
            gain = int(aug.integers(len(cfg["gains_db"])))
        tape[j] = slot, visits[slot], prefix, gain, groups[slot]["bound"] + prefix
        visits[slot] += 1
    return tape


def selected_id(groups, draw, size):
    slot, visit = map(int, draw[:2])
    members = groups[slot]["members"][:size]
    return members[visit % len(members)]


def tier_ids(groups, size):
    return [i for g in groups for i in g["members"][:size]]


def lr_factor(update, cfg):
    horizon = max(cfg["checkpoint_updates"])
    floor = cfg["lr_floor_fraction"]
    return min(1., update / cfg["warmup_updates"]) * (
        floor + (1 - floor) * (1 + math.cos(math.pi * min(update, horizon) / horizon)) / 2)


def checkpoint_reasons(update, seconds, cfg, emitted_time_points):
    reasons = []
    if update == 0 or update in cfg["checkpoint_updates"]:
        reasons.append(f"updates:{update}")
    reasons += [f"training_seconds:{s}" for s in cfg["checkpoint_training_seconds"]
                if seconds >= s and s not in emitted_time_points]
    return reasons


def exposure(rows, groups, tape, size, batch_size):
    by_id = {r["id"]: r for r in rows}
    counts = collections.Counter()
    hours = collections.Counter()
    unique = set()
    h = hashlib.sha256()
    for draw in tape:
        i = selected_id(groups, draw, size)
        r = by_id[i]
        counts[r["domain"]] += 1
        hours[r["domain"]] += r["seconds"] / 3600
        unique.add(i)
        h.update((i + "\n").encode())
    seen = {d: [by_id[i] for i in unique if by_id[i]["domain"] == d] for d in DOMAINS}
    padded = sum(int(batch[:, 4].max()) * len(batch)
                 for batch in np.split(tape, len(tape) // batch_size)) if len(tape) else 0
    return dict(domain_presentations=dict(counts), presented_hours=dict(hours),
                unique_recordings={d: len(v) for d, v in seen.items()},
                unique_hours={d: sum(r["seconds"] for r in v) / 3600 for d, v in seen.items()},
                padded_input_frames=padded, sample_sha256=h.hexdigest())


def aggregate(predictions):
    result = {}
    for d in DOMAINS:
        group = [p for p in predictions if p["domain"] == d]
        if not group:
            continue
        metric = scores([(r["reference"], r["prediction"]) for r in group])
        metric.update(ctc_loss=float(np.mean([r["ctc_sum"] / r["target_length"] for r in group])),
                      ctc_per_character=sum(r["ctc_sum"] for r in group) / sum(r["target_length"] for r in group),
                      blank_fraction=float(np.mean([r["blank_fraction"] for r in group])))
        if d == "digits":
            metric.update(exact=sum(digit_string(r["reference"]) == digit_string(r["prediction"]) for r in group),
                          total=len(group))
        result[d] = metric
    return result


def bootstrap_delta(first, second, domain, replicates=1000, seed=1741):
    """Paired cluster bootstrap; conditional on these two trained models.

    A one-speaker digit holdout has no speaker CI. Return that limitation rather
    than treating synthetic sequences as independent speakers.
    """
    a = {r["id"]: r for r in first if r["domain"] == domain}
    b = {r["id"]: r for r in second if r["domain"] == domain}
    assert set(a) == set(b) and a
    clusters = collections.defaultdict(list)
    for i, r in a.items():
        key = norm(r["reference"]) if domain == "medical_symptoms" else str(r.get("speaker"))
        clusters[key].append(i)
    def metric(records):
        m = aggregate(records)[domain]
        return 1 - m["exact"] / m["total"] if domain == "digits" else m["wer"]
    delta = metric(list(b.values())) - metric(list(a.values()))
    if len(clusters) < 2:
        return dict(delta=delta, interval=None, clusters=len(clusters), reason="Only one held-out speaker")
    # Aggregate edit counts once, not thousands of times inside the bootstrap.
    values = []
    for ids in clusters.values():
        pair = []
        for records in (a, b):
            m = aggregate([records[i] for i in ids])[domain]
            pair.extend((m["total"] - m["exact"], m["total"]) if domain == "digits"
                        else (m["word_errors"], m["reference_words"]))
        values.append(pair)
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    totals = values[rng.integers(len(values), size=(replicates, len(values)))].sum(axis=1)
    draws = totals[:, 2] / totals[:, 3] - totals[:, 0] / totals[:, 1]
    return dict(delta=delta, interval=np.quantile(draws, [.025, .975]).tolist(), clusters=len(clusters),
                cluster_unit="transcript" if domain == "medical_symptoms" else "speaker")
