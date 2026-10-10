"""The calibration curve (Q28, 10 Oct): a person's own calibrated decoder vs windows used.

Synthetic people carry a strong, person-specific class signal (event-related
desynchronisation), so a decoder calibrated on their own windows should be near-perfect with
the whole pool, and the curve must be paired: every size is tested on the same windows.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from neuros.data.synthetic import SyntheticConfig, make_recording
from neuros.errors import NeurOSConfigError, NeurOSDataError
from neuros.eval.calibration import (
    ALL,
    calibration_curve,
    chronological_order,
    minutes_label,
    split_block,
    trial_interval_s,
)
from neuros.signal.preprocess import preprocess_recording
from neuros.signal.windows import WindowSet, extract_epochs

LR = ("left_hand", "right_hand")


def _block(subject: int, session: str, run: str, seed: int, n_trials: int = 24) -> WindowSet:
    recording = make_recording(
        SyntheticConfig(n_trials=n_trials, seed=seed), subject=subject, session=session, run=run
    )
    windows, _ = extract_epochs(preprocess_recording(recording), epoch_s=(0.5, 2.5))
    return windows


@pytest.fixture(scope="module")
def people() -> WindowSet:
    """Three people, two sessions each; session 1 has two runs ("2" and "10")."""
    sets = []
    for subject in (1, 2, 3):
        sets.append(_block(subject, "0train", "0", seed=subject))
        sets.append(_block(subject, "1test", "2", seed=subject + 100))
        sets.append(_block(subject, "1test", "10", seed=subject + 200))
    return WindowSet.concat(sets)


def test_chronological_order_sorts_sessions_runs_and_starts_naturally(
    people: WindowSet,
) -> None:
    rows = np.flatnonzero(people.subjects == 1).astype(np.int64)  # one person: one block
    ordered = chronological_order(people, rows[::-1].copy())  # reversed input, same result
    keys = [(people.sessions[i], people.runs[i]) for i in ordered.tolist()]
    # Session "0train" first, then "1test" run "2" before run "10" (natural, not text, order).
    assert list(dict.fromkeys(keys)) == [("0train", "0"), ("1test", "2"), ("1test", "10")]
    for run in ("0", "2", "10"):
        starts = [int(people.starts[i]) for i in ordered.tolist() if people.runs[i] == run]
        assert starts == sorted(starts)
    assert np.array_equal(np.sort(ordered), np.sort(rows))  # a permutation, nothing lost


def test_split_block_halves_each_class_in_time_order(people: WindowSet) -> None:
    rows = np.flatnonzero(
        (people.subjects == 1) & (np.asarray(people.sessions) == "0train")
    ).astype(np.int64)
    pool, test = split_block(people, rows, LR)
    labels = np.asarray(people.labels)
    for label in LR:
        mine = rows[labels[rows] == label]
        assert pool[label].size == mine.size // 2
        # Every pooled window comes before every tested window of the same class.
        tested = test[labels[test] == label]
        assert people.starts[pool[label]].max() < people.starts[tested].min()
        assert set(pool[label].tolist()) | set(tested.tolist()) == set(mine.tolist())
    assert not set(np.concatenate(list(pool.values())).tolist()) & set(test.tolist())
    one_left = np.concatenate(
        [rows[labels[rows] == "left_hand"][:1], rows[labels[rows] != "left_hand"]]
    )
    with pytest.raises(NeurOSDataError, match="at least 2"):
        split_block(people, one_left, LR)


def test_trial_interval_matches_the_synthetic_design(people: WindowSet) -> None:
    rows = np.flatnonzero(people.subjects == 2).astype(np.int64)
    interval = trial_interval_s(people, rows)
    # Synthetic trials: 2 s pre-cue + 4 s task + 1.5-2.5 s inter-trial interval.
    assert interval is not None
    assert 7.0 <= interval <= 8.6
    assert trial_interval_s(people, rows[:1]) is None  # one window: nothing to measure
    assert minutes_label(None) == "-"
    assert minutes_label(3.25) == "3.2 min"


def test_curve_is_paired_and_reaches_a_near_perfect_ceiling(people: WindowSet) -> None:
    result = calibration_curve(people, labels=LR, sizes=(2, 4, 50), block="session")
    json.dumps(result)  # serialisable as written to disk
    points = {point["n_per_class"]: point for point in result["curve"]}
    # 50 per class exceeds every pool (12 trials per class per session -> 6): skipped.
    assert set(points) == {2, 4, ALL}
    assert result["sizes"] == [2, 4, 50, ALL]
    for point in points.values():
        assert point["n_people"] == 3
    # Paired: every size scores each person on the same test windows.
    for person in ("Synthetic:1", "Synthetic:2", "Synthetic:3"):
        counts = {
            n: next(s["n_windows"] for s in point["per_person"] if s["person"] == person)
            for n, point in points.items()
        }
        assert len(set(counts.values())) == 1, counts
    assert points[ALL]["summary_balanced_accuracy"]["mean"] >= 0.9
    # Minutes grow with the windows used: 2 x n trials x ~8 s.
    assert points[2]["minutes_median"] == pytest.approx(4 * 7.8 / 60, rel=0.15)
    assert points[2]["minutes_median"] < points[4]["minutes_median"] < points[ALL]["minutes_median"]
    assert len(result["blocks"]) == 6  # 3 people x 2 sessions


def test_person_blocks_pool_the_sessions(people: WindowSet) -> None:
    result = calibration_curve(people, labels=LR, sizes=(10,), block="person")
    assert len(result["blocks"]) == 3
    for block in result["blocks"]:
        # 12 + 24 windows per class per person (session 0: one run; session 1: two runs).
        assert block["pool_per_class"] == {"left_hand": 18, "right_hand": 18}
    assert [p["n_per_class"] for p in result["curve"]] == [10, ALL]


def test_configuration_and_data_errors(people: WindowSet) -> None:
    for bad in ((), (0,), (3, 3), (2.5,)):
        with pytest.raises(NeurOSConfigError, match="sizes"):
            calibration_curve(people, labels=LR, sizes=bad)  # type: ignore[arg-type]
    with pytest.raises(NeurOSConfigError, match="labels"):
        calibration_curve(people, labels=("left_hand",))
    with pytest.raises(NeurOSConfigError, match="block"):
        calibration_curve(people, labels=LR, block="run")  # type: ignore[arg-type]
    with pytest.raises(NeurOSDataError, match="outside"):
        calibration_curve(people, labels=("left_hand", "feet"))
