"""How long would each person need for one *reliable* command, if the engine waited for evidence?

A desk calculation from results we already have; it trains nothing and spends no compute.

Usage: uv run --no-sync python scripts/analysis/reliability_time.py [results/aws]

The idea: a decoder that is right a fraction ``p`` of the time per window can still give a
command that is right 99% of the time, by accumulating evidence over several windows before
acting. Wald's sequential probability ratio test (SPRT) is the fastest way to do that on average.
With symmetric error rates, the expected number of independent decisions to reach accuracy
``T`` is

    n(p, T) = ln(T / (1 - T)) / KL(p),   KL(p) = p ln(p / (1-p)) + (1-p) ln((1-p) / p),

the log-odds needed divided by the evidence each decision adds. ``p <= 0.5`` never gets there.

Two parts:
1. The table: decisions and seconds per reliable command for a range of per-decision accuracies.
2. The 73 cue-free people (``scaling_projection.cue_free_scores``, noise-corrected the same
   way): what share of them get a 99%-reliable command within 10/30/60/120 s, at today's zero
   calibration and at their own calibrated ceiling.

Assumptions (stated, not tested), all generous:
- **Independent decisions, one every 2 s.** Overlapping windows and slow drifts make real
  decisions correlated, so real times are longer.
- **A person's accuracy is constant** over the minutes it takes; sustained imagery tires.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scaling_projection import CUE_FREE, Scores, cue_free_scores, shrink

DECISION_S = 2.0  # one engine decision every 2 s
TARGETS = (0.99, 0.999)
ACCURACIES = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 0.95)
LIMITS_S = (10, 30, 60, 120)
NEVER = 0.52  # at or below this, a person is effectively at chance: no usable accumulation


def kl(p: float) -> float:
    """Evidence (nats) one decision adds on average, for a symmetric binary decision."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"accuracy must be strictly between 0 and 1, got {p}")
    return p * math.log(p / (1 - p)) + (1 - p) * math.log((1 - p) / p)


def decisions_needed(p: float, target: float) -> float:
    """Expected independent decisions (Wald SPRT) to reach ``target`` accuracy; inf at p <= 0.5."""
    if not 0.5 < target < 1.0:
        raise ValueError(f"target must be in (0.5, 1), got {target}")
    if p <= 0.5:
        return math.inf
    return math.log(target / (1 - target)) / kl(min(p, 1 - 1e-9))


def noise_corrected(table: Scores) -> np.ndarray:
    """Every cue-free person's score after empirical-Bayes shrinkage within their dataset."""
    out: list[float] = []
    for dataset in CUE_FREE:
        acc = np.array([a for a, _ in table[dataset]])
        n = np.array([k for _, k in table[dataset]], dtype=float)
        out.extend(shrink(acc, n)[0])
    return np.array(out)


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/aws")

    print(
        f"1. Per-decision accuracy -> independent {DECISION_S:g}-s decisions (seconds) for a "
        + " / ".join(f"{t:.1%}" for t in TARGETS)
        + " reliable command"
    )
    for p in ACCURACIES:
        cells = []
        for target in TARGETS:
            n = decisions_needed(p, target)
            cells.append(f"{n:6.1f} ({n * DECISION_S:4.0f} s)")
        print(f"  {p:.2f}: " + "   ".join(cells))

    now, ceiling = cue_free_scores(root)
    for label, table in (("zero calibration at 120 s", now), ("own calibrated ceiling", ceiling)):
        people = noise_corrected(table)
        seconds = np.array([decisions_needed(p, TARGETS[0]) * DECISION_S for p in people])
        print(f"\n2. {label} ({people.size} people): seconds per {TARGETS[0]:.0%}-reliable command")
        for limit in LIMITS_S:
            print(f"  <= {limit:3d} s: {np.mean(seconds <= limit):.0%}")
        print(
            f"  median {np.median(seconds):.0f} s;  never (<= {NEVER:.0%}): "
            f"{np.mean(people <= NEVER):.0%}"
        )


if __name__ == "__main__":
    main()
