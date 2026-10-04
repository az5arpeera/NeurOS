"""Re-referencing: common average reference over the channels present (architecture §4.2).

Datasets are recorded against different references (mastoid, nose, Cz, earlobes), and a
Ganglion with 4 channels references to an earlobe. The default contract re-references every
window to the **common average of the channels present** (CAR); E0b compares it with keeping
the recorded reference.

Where it sits in the chain: CAR is a per-sample spatial operation, and both the causal filter
and the resampler are linear and identical on every channel, so CAR commutes with them exactly.
It's applied per window, *after* the headset's channel subset S is chosen. That's the only
place a random channel-subset augmentation and a real 4-channel headset compute the same thing.

**Rank:** CAR removes one dimension (the rows sum to zero), so a window covariance after CAR is
singular. EA tolerates that through its shrinkage (§5). Riemannian methods need strictly
positive-definite matrices, so they first project onto the zero-sum subspace with ``car_basis``.
That's lossless: ``CAR(X) = U Uᵀ X`` with ``U = car_basis(n)``.
"""

from __future__ import annotations

import functools

import numpy as np
from numpy.typing import NDArray

from neuros.errors import NeurOSDataError


def common_average_reference(x: NDArray[np.floating], *, axis: int = -2) -> NDArray[np.floating]:
    """Subtract, at every sample, the mean over the channel ``axis``.

    Computed in float64 and returned in the input's dtype. Works on one window
    ``[channels, samples]`` or a batch ``[windows, channels, samples]`` (``axis=-2``).

    Raises:
        NeurOSDataError: if there are fewer than 2 channels (the result would be all zeros).
    """
    if not isinstance(x, np.ndarray) or x.dtype.kind != "f":
        raise NeurOSDataError("CAR needs a floating-point numpy array")
    if x.ndim < 2:
        raise NeurOSDataError(f"CAR needs at least [channels, samples], got shape {x.shape}")
    n_channels = x.shape[axis]
    if n_channels < 2:
        raise NeurOSDataError(f"CAR needs at least 2 channels, got {n_channels}")
    wide = x.astype(np.float64, copy=False)
    return (wide - wide.mean(axis=axis, keepdims=True)).astype(x.dtype, copy=False)


@functools.cache
def _car_basis(n_channels: int) -> NDArray[np.float64]:
    # Columns of the centring matrix H = I - 11ᵀ/n span the zero-sum subspace; any n-1 of them
    # are independent, and QR turns them into an orthonormal basis. Deterministic for each n.
    centring = np.eye(n_channels) - np.full((n_channels, n_channels), 1.0 / n_channels)
    q, _ = np.linalg.qr(centring[:, : n_channels - 1])
    q.flags.writeable = False
    return q


def car_basis(n_channels: int) -> NDArray[np.float64]:
    """Orthonormal basis ``U`` (``[n, n-1]``) of the zero-sum subspace, so ``CAR = U Uᵀ``.

    ``Uᵀ X`` gives the n-1 coordinates of a window after CAR without the null direction, so its
    covariance is full-rank. Read-only and cached per channel count.

    Raises:
        NeurOSDataError: if ``n_channels`` < 2.
    """
    if not isinstance(n_channels, int) or n_channels < 2:
        raise NeurOSDataError(f"car_basis needs at least 2 channels, got {n_channels!r}")
    return _car_basis(n_channels)
