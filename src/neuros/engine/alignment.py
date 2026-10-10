"""Online alignment for a new user, label-free (architecture §5 "online"; E2).

    R̂(t) = (n₀·R_pop + n_t·R_user(t)) / (n₀ + n_t)

- **R_user(t)** is the mean covariance of the user's *good* windows so far (the quality gate
  decides; bad windows never enter). This is exactly what E2 measured: the running mean of
  every gated stream window up to the budget (``neuros.eval.warmup.blended_reference``).
  §5's optional EMA (``ema_windows``) forgets old windows once that many have been seen; E2
  didn't test it, so it is off by default.
- **R_pop** is the population prior for this headset's channels, from training people (the
  model bundle carries it). **n₀** is its weight in windows (E2: 10 s of windows, 80).
- **R̂^(-1/2)** is recomputed every ``update_every`` good windows (once per second at the
  contract's hop): an eigendecomposition of a small matrix, microseconds, but there is no need
  to do it per window.

**Numerical safety (§5).** ``reference_from_mean`` symmetrises, shrinks and floors the
eigenvalues. If a refresh still fails, the previous R̂ stays in use and ``last_error`` says
why: the engine keeps deciding on the last good reference instead of crashing (principle 4).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from neuros.align.ea import AlignmentReference, reference_from_mean
from neuros.errors import DataContext, NeurOSConfigError, NeurOSDataError


class OnlineReference:
    """The user's alignment reference, blended with a population prior (see module docs)."""

    def __init__(
        self,
        prior: NDArray[np.floating],
        *,
        prior_windows: float,
        update_every: int,
        ema_windows: int | None = None,
        context: DataContext | None = None,
    ) -> None:
        """
        Args:
            prior: R_pop, ``[ch, ch]``, symmetric positive definite.
            prior_windows: n₀, the prior's weight in windows (> 0).
            update_every: refresh R̂^(-1/2) after this many new good windows (>= 1).
            ema_windows: None (running mean, E2's) or the EMA's memory in windows (>= 2).

        Raises:
            NeurOSConfigError: on bad settings.
            NeurOSDataError: if the prior isn't a finite, square, usable covariance.
        """
        self.context = context or DataContext()
        if not (math.isfinite(prior_windows) and prior_windows > 0):
            raise NeurOSConfigError(f"prior_windows must be > 0, got {prior_windows!r}")
        if not isinstance(update_every, int) or update_every < 1:
            raise NeurOSConfigError(f"update_every must be an int >= 1, got {update_every!r}")
        if ema_windows is not None and (not isinstance(ema_windows, int) or ema_windows < 2):
            raise NeurOSConfigError(f"ema_windows must be None or an int >= 2, got {ema_windows!r}")
        matrix: NDArray[np.float64] = np.asarray(prior, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
            raise NeurOSDataError(f"prior must be a square [ch, ch] matrix, got {matrix.shape}")
        # Validates the prior (finite, usable after shrinkage) before anything streams.
        self._current = reference_from_mean(matrix, 0, context=self.context)
        self.prior = matrix
        self.prior_windows = float(prior_windows)
        self.update_every = update_every
        self.ema_windows = ema_windows
        self.n_channels = int(matrix.shape[0])
        self.last_error: str | None = None
        self.reset()

    def reset(self) -> None:
        """Forget the user's data (a new session): back to the prior alone."""
        self._user = np.zeros((self.n_channels, self.n_channels), dtype=np.float64)
        self.n_windows = 0
        self._since_refresh = 0
        self._current = reference_from_mean(self.prior, 0, context=self.context)
        self.last_error = None

    def add(self, covariance: NDArray[np.floating]) -> None:
        """Add one good window's covariance ``[ch, ch]`` (the gate already passed it).

        Raises:
            NeurOSDataError: on a wrong shape or non-finite values (state unchanged).
        """
        c = np.asarray(covariance, dtype=np.float64)
        if c.shape != (self.n_channels, self.n_channels):
            raise NeurOSDataError(
                f"covariance must be [{self.n_channels}, {self.n_channels}], got {c.shape}",
                self.context,
            )
        if not np.isfinite(c).all():
            raise NeurOSDataError("covariance contains NaN/Inf", self.context)
        self.n_windows += 1
        if self.ema_windows is None or self.n_windows <= self.ema_windows:
            self._user += (c - self._user) / self.n_windows  # running mean
        else:
            self._user += (c - self._user) / self.ema_windows  # EMA with that memory
        self._since_refresh += 1
        if self._since_refresh >= self.update_every:
            self.refresh()

    def blended_mean(self) -> NDArray[np.float64]:
        """(n₀·R_pop + n_t·R_user) / (n₀ + n_t), the matrix R̂ is built from."""
        if self.n_windows == 0:
            return self.prior.copy()
        weight = float(self.n_windows)
        blend = (self.prior_windows * self.prior + weight * self._user) / (
            self.prior_windows + weight
        )
        return np.asarray(blend, dtype=np.float64)

    def refresh(self) -> bool:
        """Rebuild R̂^(-1/2) from the current blend. False (and the old R̂ kept) on failure."""
        self._since_refresh = 0
        try:
            self._current = reference_from_mean(
                self.blended_mean(), self.n_windows, context=self.context
            )
        except NeurOSDataError as error:
            self.last_error = str(error)
            return False
        self.last_error = None
        return True

    @property
    def current(self) -> AlignmentReference:
        """The reference in use (refreshed every ``update_every`` good windows)."""
        return self._current
