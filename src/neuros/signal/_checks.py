"""Boundary checks shared by the streaming signal stages (architecture §3, principle 3).

Every stage validates its input *before* touching its state, so a rejected chunk leaves the
stream exactly as it was: the caller (offline loop or live engine) can skip it or reset.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from neuros.errors import DataContext, NeurOSSignalError


def check_chunk(
    chunk: NDArray[np.floating], n_channels: int, context: DataContext
) -> NDArray[np.float64]:
    """Validate a ``[channels, samples]`` chunk and return it as a float64 copy.

    Raises:
        NeurOSSignalError: if it isn't a numeric 2-D array with ``n_channels`` rows, or has
            NaN/Inf samples.
    """
    if not isinstance(chunk, np.ndarray):
        raise NeurOSSignalError(f"chunk must be a numpy array, got {type(chunk).__name__}", context)
    if chunk.ndim != 2 or chunk.shape[0] != n_channels:
        raise NeurOSSignalError(
            f"chunk must be [{n_channels} channels, samples], got shape {chunk.shape}", context
        )
    if chunk.dtype.kind not in "fiu":
        raise NeurOSSignalError(f"chunk must be numeric, got dtype {chunk.dtype}", context)
    x = chunk.astype(np.float64, copy=True)
    if not np.isfinite(x).all():
        bad = int(np.count_nonzero(~np.isfinite(x)))
        raise NeurOSSignalError(f"chunk contains {bad} NaN/Inf samples", context)
    return x


def check_timestamps(
    timestamps: NDArray[np.floating], n_samples: int, context: DataContext
) -> NDArray[np.float64]:
    """Validate one chunk's timestamps (seconds) and return them as a float64 copy.

    Only the chunk itself is checked here: finite, 1-D, one per sample, strictly increasing.
    Continuity with the previous chunk is the resampler's job (it holds that state).

    Raises:
        NeurOSSignalError: on any violation.
    """
    if not isinstance(timestamps, np.ndarray):
        raise NeurOSSignalError(
            f"timestamps must be a numpy array, got {type(timestamps).__name__}", context
        )
    if timestamps.ndim != 1 or timestamps.shape[0] != n_samples:
        raise NeurOSSignalError(
            f"need one timestamp per sample ({n_samples}), got shape {timestamps.shape}", context
        )
    if timestamps.dtype.kind not in "fiu":
        raise NeurOSSignalError(f"timestamps must be numeric, got {timestamps.dtype}", context)
    t = timestamps.astype(np.float64, copy=True)
    if not np.isfinite(t).all():
        raise NeurOSSignalError("timestamps contain NaN/Inf", context)
    if t.size > 1 and np.any(np.diff(t) <= 0):
        first = int(np.flatnonzero(np.diff(t) <= 0)[0])
        raise NeurOSSignalError(
            f"timestamps must strictly increase; sample {first + 1} is at {t[first + 1]!r} s "
            f"after {t[first]!r} s",
            context,
        )
    return t
