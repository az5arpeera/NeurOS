"""Common average reference and the zero-sum basis (architecture §4.2)."""

from __future__ import annotations

import numpy as np
import pytest

from neuros.errors import NeurOSDataError
from neuros.signal.filters import CausalFilter, design_filter
from neuros.signal.reference import car_basis, common_average_reference


def test_car_rows_sum_to_zero_and_keep_dtype() -> None:
    x = np.random.default_rng(0).standard_normal((4, 6, 50)).astype(np.float32) + 100
    y = common_average_reference(x)
    assert y.dtype == np.float32
    assert y.shape == x.shape
    assert np.abs(y.astype(np.float64).sum(axis=1)).max() < 1e-3


def test_car_is_idempotent() -> None:
    x = np.random.default_rng(1).standard_normal((5, 40))
    once = common_average_reference(x)
    np.testing.assert_allclose(common_average_reference(once), once, atol=1e-12)


@pytest.mark.parametrize(
    "bad", [np.zeros((1, 10)), np.zeros(10), np.zeros((3, 10), dtype=np.int64)]
)
def test_car_rejects_degenerate_input(bad: np.ndarray) -> None:
    with pytest.raises(NeurOSDataError):
        common_average_reference(bad)


def test_car_commutes_with_the_causal_filter() -> None:
    # Why CAR can be applied per window, after the chain: both are linear (architecture §4.2).
    x = np.random.default_rng(2).standard_normal((4, 2000)) * 10 + 500
    design = design_filter(250.0)
    filtered_then_car = common_average_reference(CausalFilter(design, 4).process(x))
    car_then_filtered = CausalFilter(design, 4).process(common_average_reference(x))
    np.testing.assert_allclose(filtered_then_car, car_then_filtered, atol=1e-8)


@pytest.mark.parametrize("n", [2, 4, 18, 26])
def test_car_basis_spans_the_zero_sum_subspace(n: int) -> None:
    basis = car_basis(n)
    assert basis.shape == (n, n - 1)
    np.testing.assert_allclose(basis.T @ basis, np.eye(n - 1), atol=1e-12)
    np.testing.assert_allclose(basis.T @ np.ones(n), 0.0, atol=1e-12)
    np.testing.assert_allclose(basis @ basis.T, np.eye(n) - 1.0 / n, atol=1e-12)
    with pytest.raises(ValueError, match="read-only"):
        basis[0, 0] = 1.0


def test_car_equals_projection_through_the_basis() -> None:
    x = np.random.default_rng(3).standard_normal((6, 30))
    basis = car_basis(6)
    np.testing.assert_allclose(basis @ (basis.T @ x), common_average_reference(x), atol=1e-12)


def test_car_basis_needs_two_channels() -> None:
    with pytest.raises(NeurOSDataError):
        car_basis(1)
