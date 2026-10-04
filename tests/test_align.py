"""Covariances and Euclidean Alignment (architecture §5).

The Day-2 acceptance test from the plan: **after alignment, each person's mean covariance is
the identity**.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from neuros.align.covariance import is_spd, regularize, symmetrize, window_covariances
from neuros.align.ea import (
    align_per_person,
    apply_alignment,
    euclidean_reference,
    reference_from_covariances,
)
from neuros.errors import NeurOSConfigError, NeurOSDataError
from neuros.signal.reference import common_average_reference
from neuros.signal.windows import WindowSet


def _windows(
    n: int = 60, channels: int = 6, samples: int = 128, seed: int = 0, mixing: bool = True
) -> np.ndarray:
    """Windows with correlated channels (a random mixing of independent sources)."""
    rng = np.random.default_rng(seed)
    sources = rng.standard_normal((n, channels, samples))
    if not mixing:
        return sources
    mix = rng.standard_normal((channels, channels)) + 2 * np.eye(channels)
    return np.einsum("ij,njt->nit", mix, sources) * 10.0


# ---------------------------------------------------------------------- covariances
def test_window_covariances_match_the_definition() -> None:
    x = _windows(n=3)
    covs = window_covariances(x)
    assert covs.shape == (3, 6, 6)
    assert covs.dtype == np.float64
    np.testing.assert_allclose(covs[1], x[1] @ x[1].T / x.shape[2], rtol=1e-12)
    assert window_covariances(x[0]).shape == (6, 6)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros(5),  # 1-D
        np.zeros((2, 1)),  # one sample
        np.zeros((2, 0, 10)),  # no channels
        np.full((2, 3, 10), np.inf),  # Inf
    ],
)
def test_window_covariances_validate(bad: np.ndarray) -> None:
    with pytest.raises(NeurOSDataError):
        window_covariances(bad)


def test_shrinkage_follows_the_formula() -> None:
    matrix = np.diag([4.0, 1.0, 1.0])
    result = regularize(matrix, shrinkage=0.5, eigen_floor=0.0)
    mean = 2.0  # trace / n
    np.testing.assert_allclose(np.sort(result.eigenvalues), [1.5, 1.5, 0.5 * 4 + 0.5 * mean])
    assert result.shrinkage == 0.5


def test_eigenvalue_floor_and_condition_guard() -> None:
    nearly_singular = np.diag([1.0, 1.0, 1e-12])
    floored = regularize(nearly_singular, shrinkage=0.0, eigen_floor=1e-3)
    # The floor is relative to the mean eigenvalue *before* flooring (§5): 1e-3 × 2/3.
    assert floored.eigenvalues.min() == pytest.approx(1e-3 * (2 + 1e-12) / 3)
    guarded = regularize(nearly_singular, shrinkage=0.0, eigen_floor=0.0, max_condition=1e3)
    assert guarded.condition_number <= 1e3
    assert guarded.shrinkage > 0  # γ was raised to get there
    assert is_spd(guarded.matrix)


def test_regularize_rejects_garbage() -> None:
    with pytest.raises(NeurOSDataError, match="no energy"):
        regularize(np.zeros((3, 3)))
    with pytest.raises(NeurOSDataError):
        regularize(np.full((3, 3), np.nan))
    with pytest.raises(NeurOSDataError):
        regularize(np.zeros((3, 2)))
    with pytest.raises(NeurOSConfigError):
        regularize(np.eye(3), shrinkage=1.5)


def test_is_spd() -> None:
    assert is_spd(np.eye(3))
    assert not is_spd(np.diag([1.0, -1.0]))
    assert not is_spd(np.array([[1.0, 2.0], [0.0, 1.0]]))
    assert not is_spd(np.full((2, 2), np.nan))


# ---------------------------------------------------------------------- EA: the Day-2 test
def test_aligned_mean_covariance_is_exactly_identity_without_shrinkage() -> None:
    x = _windows()
    reference = euclidean_reference(x, shrinkage=0.0, eigen_floor=0.0)
    mean = window_covariances(x).mean(axis=0)
    # In exact arithmetic R̄^(-1/2) R̄ R̄^(-1/2) = I; in float64, to ~1e-12.
    whitened = reference.inv_sqrt @ mean @ reference.inv_sqrt
    np.testing.assert_allclose(whitened, np.eye(6), atol=1e-10)
    # The float32 aligned windows themselves: identity to float32 precision.
    aligned = apply_alignment(x, reference)
    assert aligned.dtype == np.float32
    np.testing.assert_allclose(window_covariances(aligned).mean(axis=0), np.eye(6), atol=1e-4)


def test_aligned_mean_covariance_is_identity_with_default_shrinkage() -> None:
    x = _windows()
    aligned = apply_alignment(x, euclidean_reference(x))
    np.testing.assert_allclose(window_covariances(aligned).mean(axis=0), np.eye(6), atol=1e-2)


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    channels=st.integers(min_value=2, max_value=12),
    scale=st.floats(min_value=1e-2, max_value=1e3),
)
def test_alignment_whitens_any_person(seed: int, channels: int, scale: float) -> None:
    x = _windows(n=40, channels=channels, seed=seed) * scale
    mean = window_covariances(x).mean(axis=0)
    # The math, with every safety rule off: exact whitening.
    exact = euclidean_reference(x, shrinkage=0.0, eigen_floor=0.0, max_condition=float("inf"))
    np.testing.assert_allclose(exact.inv_sqrt @ mean @ exact.inv_sqrt, np.eye(channels), atol=1e-6)
    # With the defaults, the condition guard keeps every reference usable. (About 1 % of these
    # random mixings exceed κ = 10⁶; the guard then raises γ, by design.)
    guarded = euclidean_reference(x)
    assert guarded.condition_number <= 1e6
    assert np.isfinite(guarded.inv_sqrt).all()


def test_alignment_removes_a_person_specific_mixing_up_to_rotation() -> None:
    # Person B = person A seen through a different "head" (an invertible mixing matrix).
    x = _windows(mixing=False, seed=3)
    rng = np.random.default_rng(4)
    head = rng.standard_normal((6, 6)) + 3 * np.eye(6)
    y = np.einsum("ij,njt->nit", head, x)
    aligned_x = apply_alignment(x, euclidean_reference(x, shrinkage=0.0, eigen_floor=0.0))
    aligned_y = apply_alignment(y, euclidean_reference(y, shrinkage=0.0, eigen_floor=0.0))
    # Q = R̄_y^(-1/2) H R̄_x^(1/2) is orthogonal, so per-window covariance spectra agree.
    spectrum_x = np.linalg.eigvalsh(window_covariances(aligned_x))
    spectrum_y = np.linalg.eigvalsh(window_covariances(aligned_y))
    np.testing.assert_allclose(spectrum_x, spectrum_y, rtol=1e-3, atol=1e-4)


def test_alignment_after_car_gives_the_zero_sum_projection() -> None:
    # CAR removes the all-ones direction, so the aligned mean is I - 11ᵀ/n, not I (module doc).
    x = common_average_reference(_windows())
    aligned = apply_alignment(x, euclidean_reference(x))
    expected = np.eye(6) - np.full((6, 6), 1 / 6)
    np.testing.assert_allclose(window_covariances(aligned).mean(axis=0), expected, atol=1e-2)


def test_reference_needs_enough_finite_windows() -> None:
    with pytest.raises(NeurOSDataError, match=">= 2 windows"):
        euclidean_reference(_windows(n=1))
    with pytest.raises(NeurOSDataError):
        reference_from_covariances(np.full((3, 2, 2), np.nan))
    with pytest.raises(NeurOSDataError, match="no energy"):
        euclidean_reference(np.zeros((5, 3, 50)))
    reference = euclidean_reference(_windows())
    with pytest.raises(NeurOSDataError, match="channels"):
        apply_alignment(_windows(channels=4), reference)


def test_reference_summary_is_serialisable() -> None:
    summary = euclidean_reference(_windows()).summary()
    assert summary["n_windows"] == 60
    assert summary["shrinkage"] == 1e-3
    assert summary["condition_number"] >= 1


# ---------------------------------------------------------------------- per person
def _people_set() -> WindowSet:
    a = _windows(seed=1).astype(np.float32)
    b = (_windows(seed=2) * 7).astype(np.float32)
    data = np.concatenate([a, b])
    n = len(data)
    return WindowSet(
        data=data,
        sfreq=128.0,
        ch_names=("FC3", "C3", "CP3", "FC4", "C4", "CP4"),
        labels=("left_hand",) * n,
        datasets=("Synthetic",) * n,
        subjects=np.repeat([1, 2], [len(a), len(b)]),
        sessions=("0",) * n,
        runs=("0",) * n,
        starts=np.zeros(n, dtype=np.int64),
        flags=np.zeros(n, dtype=np.uint8),
    )


def test_align_per_person_whitens_each_person_separately() -> None:
    windows = _people_set()
    # Shrinkage off: exactness, not regularisation, is under test (person 2's random mixing is
    # ill-conditioned enough that γ = 1e-3 visibly lifts its smallest eigenvalue).
    aligned, references = align_per_person(windows, shrinkage=0.0, eigen_floor=0.0)
    assert set(references) == {"Synthetic:1", "Synthetic:2"}
    assert aligned.labels == windows.labels
    assert aligned.person_ids == windows.person_ids
    for person in references:
        mine = aligned.for_people({person})
        mean = window_covariances(mine.data).mean(axis=0)
        np.testing.assert_allclose(mean, np.eye(6), atol=1e-4)
    # One person's data never moves the other's reference.
    alone, _ = align_per_person(windows.for_people({"Synthetic:1"}), shrinkage=0.0, eigen_floor=0.0)
    assert np.array_equal(alone.data, aligned.for_people({"Synthetic:1"}).data)


def test_symmetrize() -> None:
    m = np.array([[1.0, 2.0], [0.0, 1.0]])
    np.testing.assert_array_equal(symmetrize(m), [[1.0, 1.0], [1.0, 1.0]])
