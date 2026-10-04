"""Subject-balanced sampling (architecture §6.4): K people × M windows, classes balanced."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from neuros.errors import NeurOSConfigError, NeurOSDataError
from neuros.train.sampler import SubjectBalancedSampler


def _layout(counts: dict[str, dict[int, int]]) -> tuple[list[str], list[int]]:
    """people/labels arrays from {person: {label: n_windows}}, interleaved like real data."""
    people: list[str] = []
    labels: list[int] = []
    for person, per_label in counts.items():
        for label, n in per_label.items():
            people += [person] * n
            labels += [label] * n
    return people, labels


def _check_batch(batch: np.ndarray, people: list[str], labels: list[int], k: int, m: int) -> None:
    assert batch.dtype == np.int64
    assert len(batch) == k * m
    chosen = Counter(people[i] for i in batch)
    assert len(chosen) == k
    assert set(chosen.values()) == {m}
    for person in chosen:
        mine = [labels[i] for i in batch if people[i] == person]
        available = sorted({labels[i] for i, p in enumerate(people) if p == person})
        per_class = Counter(mine)
        assert set(per_class) <= set(available)
        counts = [per_class.get(label, 0) for label in available]
        assert max(counts) - min(counts) <= 1  # balanced within the person


def test_batches_are_k_people_by_m_windows_class_balanced() -> None:
    people, labels = _layout({f"p{i}": {0: 30, 1: 30} for i in range(5)})
    sampler = SubjectBalancedSampler(
        people, labels, people_per_batch=3, windows_per_person=8, seed=0
    )
    assert sampler.batch_size == 24
    assert sampler.batches_per_epoch == math.ceil(300 / 24)
    for batch in sampler.epoch():
        _check_batch(batch, people, labels, 3, 8)


def test_queues_use_every_window_before_repeating() -> None:
    # Every person in every batch, 4 + 4 windows each: after 5 batches, 20 draws per
    # (person, class) pool of 10 windows, so each window was drawn exactly twice.
    people, labels = _layout({"a": {0: 10, 1: 10}, "b": {0: 10, 1: 10}})
    sampler = SubjectBalancedSampler(
        people, labels, people_per_batch=2, windows_per_person=8, seed=1, batches_per_epoch=5
    )
    usage = Counter(int(i) for batch in sampler.epoch() for i in batch)
    assert set(usage.values()) == {2}
    assert len(usage) == 40


def test_minority_people_weigh_as_much_as_majority_people() -> None:
    people, labels = _layout({"big": {0: 500, 1: 500}, "small": {0: 5, 1: 5}})
    sampler = SubjectBalancedSampler(
        people, labels, people_per_batch=1, windows_per_person=4, seed=2, batches_per_epoch=400
    )
    per_person = Counter(people[int(batch[0])] for batch in sampler.epoch())
    assert abs(per_person["big"] - per_person["small"]) < 80  # ≈ 200 each, not 99:1


def test_deterministic_for_a_seed() -> None:
    people, labels = _layout({f"p{i}": {0: 7, 1: 9} for i in range(4)})

    def epoch(seed: int) -> list[list[int]]:
        sampler = SubjectBalancedSampler(
            people, labels, people_per_batch=2, windows_per_person=6, seed=seed
        )
        return [b.tolist() for b in sampler.epoch()]

    assert epoch(5) == epoch(5)
    assert epoch(5) != epoch(6)


def test_people_per_batch_is_capped() -> None:
    people, labels = _layout({"a": {0: 4, 1: 4}, "b": {0: 4, 1: 4}})
    sampler = SubjectBalancedSampler(
        people, labels, people_per_batch=8, windows_per_person=4, seed=0
    )
    assert sampler.requested_people_per_batch == 8
    assert sampler.people_per_batch == 2
    assert sampler.batch_size == 8
    _check_batch(sampler.batch(), people, labels, 2, 4)


def test_single_class_people_and_small_pools() -> None:
    # Person "x" has only class 0; person "y" has 2 windows of class 1 but gets 3 per batch.
    people, labels = _layout({"x": {0: 6}, "y": {0: 6, 1: 2}})
    sampler = SubjectBalancedSampler(
        people, labels, people_per_batch=2, windows_per_person=6, seed=3
    )
    for batch in sampler.epoch():
        _check_batch(batch, people, labels, 2, 6)


@pytest.mark.parametrize(
    ("people", "labels", "kwargs", "error"),
    [
        ([], [], {}, NeurOSDataError),
        (["a", "a"], [0], {}, NeurOSDataError),
        ([["a"]], [[0]], {}, NeurOSDataError),
        (["a"], [0], {"people_per_batch": 0}, NeurOSConfigError),
        (["a"], [0], {"windows_per_person": 0}, NeurOSConfigError),
        (["a"], [0], {"batches_per_epoch": 0}, NeurOSConfigError),
    ],
)
def test_errors(
    people: list[object], labels: list[object], kwargs: dict[str, int], error: type[Exception]
) -> None:
    arguments = {"people_per_batch": 1, "windows_per_person": 1, "seed": 0, **kwargs}
    with pytest.raises(error):
        SubjectBalancedSampler(people, labels, **arguments)  # type: ignore[arg-type]


@given(
    counts=st.dictionaries(
        st.sampled_from(["p1", "p2", "p3", "p4", "p5", "p6"]),
        st.dictionaries(st.integers(0, 2), st.integers(1, 9), min_size=1),
        min_size=1,
    ),
    k=st.integers(1, 7),
    m=st.integers(1, 9),
    seed=st.integers(0, 2**32 - 1),
)
def test_invariants_hold_for_any_layout(
    counts: dict[str, dict[int, int]], k: int, m: int, seed: int
) -> None:
    people, labels = _layout(counts)
    sampler = SubjectBalancedSampler(
        people, labels, people_per_batch=k, windows_per_person=m, seed=seed
    )
    k_effective = min(k, len(counts))
    assert sampler.batches_per_epoch == max(1, math.ceil(len(people) / (k_effective * m)))
    for batch in sampler.epoch():
        _check_batch(batch, people, labels, k_effective, m)
