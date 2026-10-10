"""How much calibration does a person need? The calibration curve (Q28, 10 Oct).

E2 and D-039 showed that zero-calibration accuracy is capped by the model, not by the warm-up.
The honest claim is then *calibration saved*: "minute zero with NeurOS equals N minutes of
calibration elsewhere". This module measures the other side of that sentence: a standard
calibrated decoder (Riemannian tangent space + logistic regression, the field's strong
baseline), trained on a person's own labelled windows, as a function of how many it gets.

**Protocol** (per *block*: one recording session, or the whole person when sessions are too
short; every number is development people only):

- Each class's windows in the block are put in time order (session, run, then start sample;
  names compared naturally, so run "10" comes after run "9"). The first half of each class is
  the **calibration pool**, the rest is the **test set**. Splitting per class keeps both classes
  on each side even when a design presents classes in blocks.
- For each size *n*, the decoder is calibrated on the **first n windows of each class** in the
  pool: what a calibration phase at the start of the session would collect. ``"all"`` uses the
  whole pool (about half a session: the usual 10-15 minutes of calibration).
- Every size is tested on the **same test set**, so the curve is paired person by person.
- A person's score pools the test windows of all their blocks. A size is reported only for
  people whose every block has *n* windows per class in its pool.
- **Minutes of calibration** = 2n trials (n per class, two classes) x the block's median
  interval between consecutive trials in a run: the recording time a calibration phase costs.

Alignment isn't used: a model trained and tested on the same person's block needs none (this is
the calibrated setting, not the cross-person one). Covariances, reference and classifier are
exactly the baselines' (``neuros.eval.baselines``), so numbers are comparable with E0b's.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from neuros.errors import NeurOSConfigError, NeurOSDataError
from neuros.eval.baselines import (
    COVARIANCE_ESTIMATOR,
    Method,
    Reference,
    build_features,
    make_classifier,
)
from neuros.eval.metrics import PersonScore, score_person, summarize
from neuros.signal.windows import WindowSet

log = logging.getLogger(__name__)

Block = Literal["session", "person"]
BLOCKS: tuple[Block, ...] = ("session", "person")
# Windows per class to calibrate on (the curve's x-axis), before "all" (the whole pool).
DEFAULT_SIZES: tuple[int, ...] = (5, 10, 20, 40)
ALL = "all"


def _natural_key(name: str) -> tuple[Any, ...]:
    """Sort key that orders "run_2" before "run_10" (digits compared as numbers)."""
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name))


def chronological_order(windows: WindowSet, rows: NDArray[np.int64]) -> NDArray[np.int64]:
    """``rows`` sorted by time: session, then run (natural order), then start sample.

    MOABB names sessions and runs in recording order ("0train" before "1test", "0" .. "5"),
    so this is the order the person recorded them in.
    """
    keys = [
        (_natural_key(windows.sessions[i]), _natural_key(windows.runs[i]), int(windows.starts[i]))
        for i in rows.tolist()
    ]
    order = sorted(range(rows.size), key=keys.__getitem__)
    return rows[np.asarray(order, dtype=np.int64)]


def _block_ids(windows: WindowSet, block: Block) -> NDArray[np.str_]:
    people = np.asarray(windows.person_ids)
    if block == "person":
        return people
    if block == "session":
        return np.asarray([f"{p}/{s}" for p, s in zip(people, windows.sessions, strict=True)])
    raise NeurOSConfigError(f"unknown block {block!r} {BLOCKS}")


def trial_interval_s(windows: WindowSet, rows: NDArray[np.int64]) -> float | None:
    """Median seconds between consecutive windows of the same run (one window per trial).

    None when no run in ``rows`` has two windows (the interval can't be measured).
    """
    gaps: list[float] = []
    runs = np.asarray([f"{windows.sessions[i]}/{windows.runs[i]}" for i in rows.tolist()])
    for run in dict.fromkeys(runs.tolist()):
        starts = np.sort(windows.starts[rows[runs == run]])
        if starts.size >= 2:
            gaps.extend((np.diff(starts) / windows.sfreq).tolist())
    positive = [g for g in gaps if g > 0]
    return float(np.median(positive)) if positive else None


def split_block(
    windows: WindowSet, rows: NDArray[np.int64], labels: Sequence[str]
) -> tuple[dict[str, NDArray[np.int64]], NDArray[np.int64]]:
    """The block's calibration pool (per class, in time order) and its test rows.

    Each class's windows, in time order, are halved: the first ``floor(k / 2)`` go to the
    pool, the rest to the test set.

    Raises:
        NeurOSDataError: if a class has fewer than 2 windows in the block (no pool or no test).
    """
    ordered = chronological_order(windows, rows)
    block_labels = np.asarray([windows.labels[i] for i in ordered.tolist()])
    pool: dict[str, NDArray[np.int64]] = {}
    test_parts: list[NDArray[np.int64]] = []
    for label in labels:
        mine = ordered[block_labels == label]
        if mine.size < 2:
            raise NeurOSDataError(
                f"class {label!r} has {mine.size} window(s) in a block: it needs at least 2 "
                "(one to calibrate on, one to test)"
            )
        half = mine.size // 2
        pool[label] = mine[:half]
        test_parts.append(mine[half:])
    return pool, np.sort(np.concatenate(test_parts))


def calibration_curve(
    windows: WindowSet,
    *,
    labels: Sequence[str],
    sizes: Sequence[int] = DEFAULT_SIZES,
    block: Block = "session",
    reference: Reference = "as_recorded",
    method: Method = "ts_lr",
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Each person's calibrated accuracy after n windows per class of their own data.

    Args:
        windows: the development people's task windows (already cue-controlled, D-030).
        labels: the classes, every one present in every block.
        sizes: windows per class to calibrate on; positive, distinct. ``"all"`` (the whole
            pool) is always added.
        block: "session" (calibrate and test within each recording session) or "person"
            (sessions concatenated in order; for datasets with very short sessions).

    Returns:
        A JSON-serialisable dict: the protocol, one entry per block (pool size, test size,
        trial interval) and one curve point per size (people, minutes, summary, per person).

    Raises:
        NeurOSConfigError: on a bad size, block or label list.
        NeurOSDataError: on windows with other labels, or a block that can't be split.
    """
    say = progress or (lambda message: log.info("%s", message))
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise NeurOSConfigError(f"need at least 2 distinct labels, got {list(labels)}")
    if not sizes or any(int(n) != n or n < 1 for n in sizes) or len(set(sizes)) != len(sizes):
        raise NeurOSConfigError(f"sizes must be distinct positive integers, got {list(sizes)}")
    if block not in BLOCKS:
        raise NeurOSConfigError(f"unknown block {block!r} {BLOCKS}")
    if len(windows) == 0:
        raise NeurOSDataError("no windows")
    unknown = sorted(set(windows.labels) - set(labels))
    if unknown:
        raise NeurOSDataError(f"windows carry labels {unknown} outside {list(labels)}")

    features = build_features(windows, alignment="none", reference=reference)
    block_of = _block_ids(windows, block)
    person_of = np.asarray(windows.person_ids)
    points: list[int | str] = [*sorted(int(n) for n in sizes), ALL]

    # Per block: its pool and test rows (the test set is fixed for every size).
    blocks: list[dict[str, Any]] = []
    splits: dict[str, tuple[dict[str, NDArray[np.int64]], NDArray[np.int64]]] = {}
    for name in dict.fromkeys(block_of.tolist()):
        rows = np.flatnonzero(block_of == name).astype(np.int64)
        pool, test = split_block(windows, rows, labels)
        splits[name] = (pool, test)
        blocks.append(
            {
                "block": name,
                "person": str(person_of[rows[0]]),
                "pool_per_class": {label: int(pool[label].size) for label in labels},
                "n_test": int(test.size),
                "trial_interval_s": trial_interval_s(windows, rows),
            }
        )

    curve: list[dict[str, Any]] = []
    for point in points:
        scores: list[PersonScore] = []
        minutes: list[float] = []
        for person in dict.fromkeys(person_of.tolist()):
            mine = [b for b in blocks if b["person"] == person]
            if point != ALL and any(min(b["pool_per_class"].values()) < int(point) for b in mine):
                continue  # not enough calibration windows in every block at this size
            truth: list[str] = []
            predicted: list[str] = []
            flagged = 0
            first = None if point == ALL else int(point)  # None: the whole pool
            for entry in mine:
                pool, test = splits[entry["block"]]
                train = np.sort(np.concatenate([pool[label][:first] for label in labels]))
                model = make_classifier(method)
                model.fit(features.covariances[train], features.labels[train])
                predicted.extend(np.asarray(model.predict(features.covariances[test])).tolist())
                truth.extend(features.labels[test].tolist())
                flagged += int(np.count_nonzero(features.flagged[test]))
                interval = entry["trial_interval_s"]
                if interval is not None:
                    n_trials = train.size  # n per class x classes (or the whole pool)
                    minutes.append(n_trials * interval / 60.0)
            scores.append(score_person(person, truth, predicted, n_flagged=flagged))
        if not scores:
            say(f"n = {point}: no person has enough calibration windows")
            continue
        accuracies = [s.balanced_accuracy for s in scores]
        curve.append(
            {
                "n_per_class": point,
                "n_people": len(scores),
                "minutes_median": float(np.median(minutes)) if minutes else None,
                "summary_balanced_accuracy": summarize(accuracies),
                "per_person": [s.to_dict() for s in scores],
            }
        )
        mean = float(np.mean(accuracies)) * 100
        say(f"n = {point}: {len(scores)} people, mean {mean:.1f}%")
    if not curve:
        raise NeurOSDataError("no calibration size could be scored")
    return {
        "method": method,
        "block": block,
        "reference": reference,
        "covariance_estimator": COVARIANCE_ESTIMATOR,
        "labels": list(labels),
        "sizes": points,
        "blocks": blocks,
        "curve": curve,
    }


def minutes_label(minutes: float | None) -> str:
    """ "3.2 min" or "-" when the trial interval was unknown."""
    return "-" if minutes is None or not math.isfinite(minutes) else f"{minutes:.1f} min"
