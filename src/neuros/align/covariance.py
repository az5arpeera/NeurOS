"""Covariance matrices and the numerical safety rules of architecture §5.

For a window X ∈ ℝ^{|S|×T}, the spatial covariance is ``C = X Xᵀ / T``. The band-pass removes
the mean, so no centring is applied: this matches the EA paper.

Safety rules (§5), applied in order by ``regularize``:

1. **symmetrise** ``(R + Rᵀ) / 2`` (float round-off breaks symmetry);
2. **shrink** toward a scaled identity: ``R ← (1-γ)·R + γ·(tr R / n)·I``, γ = 1e-3;
3. **eigenvalue floor**: ``λᵢ ← max(λᵢ, ε·mean λ)``;
4. **condition guard**: while ``λ_max / λ_min > 10⁶``, raise γ tenfold and redo 2-3. With
   γ ≥ 1e-3, κ ≤ n/γ, so this only fires for γ ≈ 0 or very many channels; it's a backstop.

Everything is float64: the inverse square root of an ill-conditioned float32 matrix loses most
of its digits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neuros.errors import DataContext, NeurOSConfigError, NeurOSDataError

DEFAULT_SHRINKAGE = 1e-3
DEFAULT_EIGEN_FLOOR = 1e-6
DEFAULT_MAX_CONDITION = 1e6


def window_covariances(windows: NDArray[np.floating]) -> NDArray[np.float64]:
    """Spatial covariance ``X Xᵀ / T`` of each window: ``[n, ch, T]`` → ``[n, ch, ch]``.

    A single window ``[ch, T]`` gives ``[ch, ch]``.

    Raises:
        NeurOSDataError: on a wrong shape, fewer than 2 samples, or NaN/Inf.
    """
    if not isinstance(windows, np.ndarray) or windows.ndim not in (2, 3):
        shape = getattr(windows, "shape", None)
        raise NeurOSDataError(f"windows must be [n, ch, T] or [ch, T], got shape {shape}")
    if windows.shape[-1] < 2 or windows.shape[-2] < 1:
        raise NeurOSDataError(f"windows need >= 1 channel and >= 2 samples, got {windows.shape}")
    x = windows.astype(np.float64, copy=False)
    if not np.isfinite(x).all():
        raise NeurOSDataError("windows contain NaN/Inf samples")
    return np.asarray(np.matmul(x, np.swapaxes(x, -1, -2)) / x.shape[-1], dtype=np.float64)


def symmetrize(matrix: NDArray[np.floating]) -> NDArray[np.float64]:
    """``(M + Mᵀ) / 2`` in float64."""
    m = np.asarray(matrix, dtype=np.float64)
    return np.asarray((m + np.swapaxes(m, -1, -2)) / 2, dtype=np.float64)


def _check_square(matrix: NDArray[np.floating], context: DataContext | None) -> None:
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise NeurOSDataError(
            f"expected a square matrix, got shape {getattr(matrix, 'shape', None)}", context
        )
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise NeurOSDataError(f"expected a non-empty square matrix, got {matrix.shape}", context)
    if not np.isfinite(matrix).all():
        raise NeurOSDataError("matrix contains NaN/Inf", context)


@dataclass(frozen=True, eq=False)
class RegularizedCovariance:
    """An SPD matrix after the §5 safety rules, with the parameters that produced it."""

    matrix: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]  # ascending, after the floor
    eigenvectors: NDArray[np.float64]  # columns, matching ``eigenvalues``
    shrinkage: float  # γ actually used (may exceed the requested one)
    condition_number: float

    def power(self, exponent: float) -> NDArray[np.float64]:
        """``R^p`` from the stored eigendecomposition (e.g. -0.5 for EA)."""
        scaled = self.eigenvectors * (self.eigenvalues**exponent)
        return np.asarray(symmetrize(scaled @ self.eigenvectors.T), dtype=np.float64)


def regularize(
    matrix: NDArray[np.floating],
    *,
    shrinkage: float = DEFAULT_SHRINKAGE,
    eigen_floor: float = DEFAULT_EIGEN_FLOOR,
    max_condition: float = DEFAULT_MAX_CONDITION,
    context: DataContext | None = None,
) -> RegularizedCovariance:
    """Apply the §5 safety rules to a covariance estimate.

    Raises:
        NeurOSConfigError: on parameters outside their ranges.
        NeurOSDataError: if the matrix isn't square and finite, or has no energy at all
            (trace ≤ 0: every window was zero), which no regularisation can fix.
    """
    if not 0 <= shrinkage <= 1:
        raise NeurOSConfigError(f"shrinkage must be in [0, 1], got {shrinkage}")
    if not 0 <= eigen_floor < 1:
        raise NeurOSConfigError(f"eigen_floor must be in [0, 1), got {eigen_floor}")
    if not max_condition > 1:
        raise NeurOSConfigError(f"max_condition must be > 1, got {max_condition}")
    _check_square(matrix, context)
    base = symmetrize(matrix)
    n = base.shape[0]
    scale = float(np.trace(base)) / n
    if not (math.isfinite(scale) and scale > 0):
        raise NeurOSDataError(
            f"covariance has no energy (mean variance {scale!r}): are all windows zero?", context
        )
    identity = np.eye(n)
    gamma = shrinkage
    while True:
        shrunk = (1 - gamma) * base + gamma * scale * identity
        eigenvalues, eigenvectors = np.linalg.eigh(shrunk)
        floor = eigen_floor * float(np.mean(eigenvalues))
        # A floor of 0 still forbids negative eigenvalues (round-off on a singular matrix).
        eigenvalues = np.maximum(eigenvalues, max(floor, np.finfo(np.float64).tiny))
        condition = float(eigenvalues[-1] / eigenvalues[0])
        if condition <= max_condition or gamma >= 1:
            break
        gamma = min(1.0, gamma * 10 if gamma > 0 else 1e-6)
    matrix_out = symmetrize((eigenvectors * eigenvalues) @ eigenvectors.T)
    for array in (matrix_out, eigenvalues, eigenvectors):
        array.flags.writeable = False
    return RegularizedCovariance(
        matrix=matrix_out,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        shrinkage=float(gamma),
        condition_number=condition,
    )


def is_spd(matrix: NDArray[np.floating], *, rtol: float = 1e-10) -> bool:
    """True if ``matrix`` is finite, symmetric (to ``rtol``) and positive definite."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or not np.isfinite(m).all():
        return False
    scale = max(float(np.max(np.abs(m))), np.finfo(np.float64).tiny)
    if float(np.max(np.abs(m - m.T))) > rtol * scale:
        return False
    try:
        np.linalg.cholesky(symmetrize(m))
    except np.linalg.LinAlgError:
        return False
    return True
