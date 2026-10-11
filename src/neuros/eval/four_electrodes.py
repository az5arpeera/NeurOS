"""A4: which four electrodes serve both commands and mental states?

The decision and pass bar are in ``knowledge-base/plan/a4-four-electrodes.md`` (written before
this code). In short:

- **Candidates:** C3 and C4 plus any 2 of the 17 other 10-20 positions both datasets record:
  136 montages, including today's Ganglion montage (C3, Cz, C4, Pz).
- **States score:** E6's pipeline (``neuros.eval.state``: TS+LR with the 20-s rest baseline,
  rest 20-60 s vs the task) on EEGMAT, leave one person out.
- **Commands score:** zero-calibration TS+LR with each person's own unlabeled alignment (EA)
  on Yang2025's cue-free windows (``neuros.eval.baselines``), leave one person out.
- **No fooling ourselves:** each dataset's people are split once, with a fixed seed, into a
  selection half (all 136 scored; the pick is made there) and a confirmation half (only the
  pick and today's montage are scored; the bar is judged there).
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from neuros.errors import NeurOSConfigError, NeurOSDataError

REQUIRED: tuple[str, ...] = ("C3", "C4")
POOL: tuple[str, ...] = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "Cz", "T8", "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2",
)  # fmt: skip
TODAY: tuple[str, ...] = ("C3", "Cz", "C4", "Pz")
SPLIT_SEED = 20261011
COMMANDS_SLACK_POINTS = 1.0  # a pick may cost commands at most this much
STATES_GAIN_POINTS = 5.0  # the confirmation half's required gain in states
P_VALUE = 0.05


def candidate_montages() -> list[tuple[str, ...]]:
    """Every montage the plan allows, as sorted tuples: C3, C4 and two of ``POOL``."""
    montages = [tuple(sorted((*REQUIRED, a, b))) for a, b in itertools.combinations(POOL, 2)]
    if tuple(sorted(TODAY)) not in montages:
        raise NeurOSConfigError("today's montage must be among the candidates")
    return montages


def split_halves(people: Sequence[str], tag: str) -> tuple[list[str], list[str]]:
    """A fixed random split of ``people`` into (selection, confirmation); selection gets the
    extra person when the count is odd. Deterministic for a given ``tag`` and people list."""
    from neuros.eval.splits import seeded_rng

    if len(people) < 6:
        raise NeurOSDataError(f"need at least 6 people to split in halves, got {len(people)}")
    order = sorted(people)
    seeded_rng(SPLIT_SEED, f"a4-{tag}").shuffle(order)
    cut = (len(order) + 1) // 2
    return sorted(order[:cut]), sorted(order[cut:])


def pick(
    selection: Mapping[tuple[str, ...], Mapping[str, float]],
    today: tuple[str, ...],
) -> tuple[str, ...]:
    """The plan's rule: among montages whose commands mean is within ``COMMANDS_SLACK_POINTS``
    of today's, the best states mean (ties: the better commands mean).

    ``selection`` maps montage → {"states": mean, "commands": mean} (fractions).
    """
    floor = selection[today]["commands"] - COMMANDS_SLACK_POINTS / 100
    allowed = [m for m, s in selection.items() if s["commands"] >= floor - 1e-12]
    return max(allowed, key=lambda m: (selection[m]["states"], selection[m]["commands"]))


def paired(a: Mapping[str, float], b: Mapping[str, float]) -> dict[str, float]:
    """Mean difference a − b over shared people and a two-sided Wilcoxon p."""
    from scipy.stats import wilcoxon

    people = sorted(set(a) & set(b))
    if not people:
        raise NeurOSDataError("no shared people to compare")
    diff = np.array([a[p] - b[p] for p in people])
    p_value = 1.0 if np.allclose(diff, 0) else float(wilcoxon(diff).pvalue)
    return {"mean_diff": float(diff.mean()), "p_value": p_value, "n_people": len(people)}


def verdict(
    winner: tuple[str, ...],
    today: tuple[str, ...],
    states: Mapping[str, Mapping[str, float]],
    commands: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """The A4 bar on the confirmation half. ``states`` / ``commands`` map "winner" and "today"
    to {person: accuracy}."""
    if winner == today:
        return {"outcome": "KEEP (the pick is today's montage)", "checks": {}}
    s = paired(states["winner"], states["today"])
    c = paired(commands["winner"], commands["today"])
    checks = {
        "states_gain_at_least_5_points": s["mean_diff"] * 100 >= STATES_GAIN_POINTS,
        "states_p_below_0.05": s["p_value"] < P_VALUE,
        "commands_at_most_1_point_worse": c["mean_diff"] * 100 >= -COMMANDS_SLACK_POINTS,
    }
    return {
        "outcome": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "states": s,
        "commands": c,
    }


def mean(scores: Mapping[str, float]) -> float:
    return float(np.mean(list(scores.values())))


def states_scores(people: Sequence[Any], montage: Sequence[str], config: Any) -> dict[str, float]:
    """E6's primary pipeline on the given people (``PersonWindows``) and channels."""
    from neuros.eval.state import (
        leave_one_person_out,
        main_labels,
        make_classifier,
        person_features,
    )

    features = [person_features(p, montage, "ts_lr", "baseline", config) for p in people]
    results = leave_one_person_out(
        features, main_labels(config), lambda: make_classifier("ts_lr", "baseline")
    )
    return {r.person: r.balanced_accuracy for r in results}


def commands_scores(
    windows: Any, people: Sequence[str], montage: Sequence[str]
) -> dict[str, float]:
    """Zero-calibration TS+LR with each person's own EA, on the given people and channels."""
    from neuros.eval.baselines import build_features, leave_one_person_out

    keep = np.isin(np.asarray(windows.person_ids), list(people))
    subset = windows.select(keep).pick_channels(list(montage))
    features = build_features(subset, alignment="ea", reference="car")
    return {s.person: s.balanced_accuracy for s in leave_one_person_out("ts_lr", features)}
