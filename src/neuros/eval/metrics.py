"""Metrics and summary statistics over people (architecture §8.2-8.3).

Every published number is a statistic **over people**, not over windows: one person with 900
windows must not outweigh nine people with 100 each. Hence:

- per person: balanced accuracy (primary) and Cohen's κ;
- across people: mean, median, IQR, share of people ≥ 70 %, and a 95 % bootstrap confidence
  interval of the mean (resampling people);
- A-vs-B claims: paired Wilcoxon signed-rank over people.

Implemented directly on confusion matrices (a few lines each) so edge cases are explicit:
a class missing from ``y_true`` is excluded from balanced accuracy rather than counted as 0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import wilcoxon

from neuros.errors import NeurOSDataError

# Architecture §8.2: "share of people ≥ 70 %", the usual threshold for usable BCI control.
USABLE_ACCURACY = 0.70
# Arbitrary but fixed, so every confidence interval regenerates exactly (principle 5).
BOOTSTRAP_SEED = 20261004


def _labels_pair(
    y_true: Sequence[str] | NDArray[Any], y_pred: Sequence[str] | NDArray[Any]
) -> tuple[NDArray[Any], NDArray[Any]]:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.ndim != 1 or pred.ndim != 1 or true.shape != pred.shape:
        raise NeurOSDataError(f"y_true {true.shape} and y_pred {pred.shape} must be equal 1-D")
    if true.size == 0:
        raise NeurOSDataError("no predictions to score")
    return true, pred


def confusion(
    y_true: Sequence[str] | NDArray[Any],
    y_pred: Sequence[str] | NDArray[Any],
    classes: Sequence[str] | None = None,
) -> tuple[NDArray[np.int64], tuple[str, ...]]:
    """Confusion matrix (rows = true, columns = predicted) and its class order."""
    true, pred = _labels_pair(y_true, y_pred)
    order = tuple(classes) if classes is not None else tuple(sorted({*true, *pred}, key=str))
    position = {label: i for i, label in enumerate(order)}
    unknown = sorted({str(v) for v in (*true, *pred) if v not in position})
    if unknown:
        raise NeurOSDataError(f"labels {unknown} are not in classes {list(order)}")
    matrix = np.zeros((len(order), len(order)), dtype=np.int64)
    np.add.at(matrix, ([position[v] for v in true], [position[v] for v in pred]), 1)
    return matrix, order


def balanced_accuracy(
    y_true: Sequence[str] | NDArray[Any], y_pred: Sequence[str] | NDArray[Any]
) -> float:
    """Mean per-class recall over the classes present in ``y_true`` (chance = 1 / classes)."""
    matrix, _ = confusion(y_true, y_pred)
    support = matrix.sum(axis=1)
    present = support > 0
    recalls = np.diag(matrix)[present] / support[present]
    return float(np.mean(recalls))


def cohens_kappa(
    y_true: Sequence[str] | NDArray[Any], y_pred: Sequence[str] | NDArray[Any]
) -> float:
    """Agreement beyond chance: (p_o - p_e) / (1 - p_e). 0 = chance, 1 = perfect.

    Returns 0.0 in the degenerate case p_e = 1 (one class only, everywhere).
    """
    matrix, _ = confusion(y_true, y_pred)
    total = matrix.sum()
    observed = np.trace(matrix) / total
    expected = float(np.sum(matrix.sum(axis=0) * matrix.sum(axis=1))) / total**2
    if math.isclose(expected, 1.0):
        return 0.0
    return float((observed - expected) / (1 - expected))


@dataclass(frozen=True)
class PersonScore:
    """One person's result in one evaluation."""

    person: str
    n_windows: int
    balanced_accuracy: float
    kappa: float
    n_flagged: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "person": self.person,
            "n_windows": self.n_windows,
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "kappa": round(self.kappa, 4),
            "n_flagged": self.n_flagged,
        }


def score_person(
    person: str,
    y_true: Sequence[str] | NDArray[Any],
    y_pred: Sequence[str] | NDArray[Any],
    n_flagged: int = 0,
) -> PersonScore:
    true, pred = _labels_pair(y_true, y_pred)
    return PersonScore(
        person=person,
        n_windows=int(true.size),
        balanced_accuracy=balanced_accuracy(true, pred),
        kappa=cohens_kappa(true, pred),
        n_flagged=n_flagged,
    )


def bootstrap_ci(
    values: Sequence[float] | NDArray[np.floating],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling people with replacement.

    Deterministic for a given seed. With fewer than 2 values the interval is the value itself.
    """
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size == 0 or not np.isfinite(data).all():
        raise NeurOSDataError("bootstrap needs a non-empty 1-D array of finite values")
    if not 0 < confidence < 1 or n_resamples < 1:
        raise NeurOSDataError("confidence must be in (0, 1) and n_resamples >= 1")
    if data.size < 2:
        return float(data[0]), float(data[0])
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, data.size, size=(n_resamples, data.size))].mean(axis=1)
    tail = (1 - confidence) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return float(low), float(high)


def summarize(values: Sequence[float] | NDArray[np.floating]) -> dict[str, float | int]:
    """Across-people summary of one metric (all fractions in [0, 1])."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size == 0 or not np.isfinite(data).all():
        raise NeurOSDataError("summarize needs a non-empty 1-D array of finite values")
    q1, median, q3 = np.quantile(data, [0.25, 0.5, 0.75])
    low, high = bootstrap_ci(data)
    return {
        "n_people": int(data.size),
        "mean": float(data.mean()),
        "std": float(data.std(ddof=1)) if data.size > 1 else 0.0,
        "ci95_low": low,
        "ci95_high": high,
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "share_usable": float(np.mean(data >= USABLE_ACCURACY)),
    }


def paired_wilcoxon(
    a: Sequence[float] | NDArray[np.floating], b: Sequence[float] | NDArray[np.floating]
) -> dict[str, float | int]:
    """Two-sided paired Wilcoxon signed-rank test over people (same order in ``a`` and ``b``).

    Returns the mean difference (a - b), the statistic and the p-value. With every difference
    zero the test is undefined: p = 1.0 is reported.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or x.size == 0:
        raise NeurOSDataError("paired test needs two equal-length, non-empty 1-D arrays")
    diff = x - y
    if np.allclose(diff, 0):
        return {"n_people": int(x.size), "mean_diff": 0.0, "statistic": 0.0, "p_value": 1.0}
    result = wilcoxon(x, y, alternative="two-sided")
    return {
        "n_people": int(x.size),
        "mean_diff": float(diff.mean()),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }
