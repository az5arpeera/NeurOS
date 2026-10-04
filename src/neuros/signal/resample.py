"""Causal, timestamp-based resampling onto the contract's 128 Hz grid (architecture §4.2).

Why timestamps instead of a polyphase resampler: native rates in scope are 160, 200, 250, 256,
500, 512 and 1000 Hz, and 250 → 128 is an awkward ratio (64/125). Live LSL streams also jitter
and occasionally drop samples. Interpolating on each sample's *timestamp* treats all of these the
same way, offline and online.

The output grid is ``t_k = origin + k / out_sfreq``, where ``origin`` is the first input
timestamp after a (re)start. Grid times are always computed from the integer ``k`` (never by
accumulation), so they're identical however the input is chunked.

**Method:** linear interpolation between the two input samples that bracket ``t_k``. It's causal
with one sample of look-ahead: ``t_k`` is emitted once an input sample at or after ``t_k`` has
arrived (≤ one native sample period of latency: 4 ms at 250 Hz). Linear interpolation slightly
attenuates the top of the band (sinc² response: about -0.4 dB at 30 Hz from 250 Hz input and
-1 dB from 160 Hz). The same loss happens in training and serving.

**Anti-aliasing** is the upstream band-pass's job (``filters``): this module assumes its input
is already band-limited well below 64 Hz.

**Gaps:** short gaps (≤ ``max_gap_s``) are bridged by the same interpolation and counted;
longer gaps raise ``NeurOSSignalGapError``, because interpolating across them would invent data.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from neuros.errors import DataContext, NeurOSConfigError, NeurOSSignalError, NeurOSSignalGapError
from neuros.signal._checks import check_chunk, check_timestamps

# Architecture §7.1: a gap longer than this resets the filter state and emits HOLD("signal_gap").
DEFAULT_MAX_GAP_S = 0.5
# An inter-sample interval longer than this many nominal periods counts as a bridged gap.
_BRIDGED_GAP_PERIODS = 2.0


def default_timestamps(n_samples: int, sfreq: float, start_sample: int = 0) -> NDArray[np.float64]:
    """Timestamps (s) of an evenly sampled recording: ``(start_sample + i) / sfreq``.

    Offline recordings have no clock, so sample ``i`` sits at ``i / sfreq``. Dividing integers
    (rather than accumulating a period) keeps every timestamp exactly reproducible.
    """
    if n_samples < 0 or start_sample < 0:
        raise NeurOSConfigError("n_samples and start_sample must be >= 0")
    if not (math.isfinite(sfreq) and sfreq > 0):
        raise NeurOSConfigError(f"sfreq must be a positive finite number, got {sfreq!r}")
    return (start_sample + np.arange(n_samples, dtype=np.float64)) / float(sfreq)


class TimestampResampler:
    """Streaming linear interpolation of ``[channels, samples]`` chunks onto a fixed grid.

    Not thread-safe: one instance per stream. After ``reset()`` the next chunk starts a new
    grid at its first timestamp.
    """

    def __init__(
        self,
        n_channels: int,
        out_sfreq: float,
        *,
        max_gap_s: float = DEFAULT_MAX_GAP_S,
        nominal_sfreq: float | None = None,
        context: DataContext | None = None,
    ) -> None:
        """
        Args:
            n_channels: rows per chunk.
            out_sfreq: output grid rate (the contract's 128 Hz).
            max_gap_s: longest inter-sample interval that may be bridged.
            nominal_sfreq: the stream's declared rate; when given, intervals longer than two
                nominal periods are counted in ``bridged_gaps`` (diagnostics only).
            context: attached to every error.
        """
        if not isinstance(n_channels, int) or n_channels < 1:
            raise NeurOSConfigError(f"n_channels must be a positive int, got {n_channels!r}")
        if not (math.isfinite(out_sfreq) and out_sfreq > 0):
            raise NeurOSConfigError(f"out_sfreq must be positive and finite, got {out_sfreq!r}")
        if not (math.isfinite(max_gap_s) and max_gap_s > 0):
            raise NeurOSConfigError(f"max_gap_s must be positive and finite, got {max_gap_s!r}")
        if nominal_sfreq is not None and not (math.isfinite(nominal_sfreq) and nominal_sfreq > 0):
            raise NeurOSConfigError(f"nominal_sfreq must be positive, got {nominal_sfreq!r}")
        self.n_channels = n_channels
        self.out_sfreq = float(out_sfreq)
        self.max_gap_s = float(max_gap_s)
        self.nominal_sfreq = None if nominal_sfreq is None else float(nominal_sfreq)
        self.context = context or DataContext()
        self._period = 1.0 / self.out_sfreq
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Forget the grid and the last sample (after a gap or reconnect)."""
        self._origin: float | None = None
        self._next_k = 0
        self._last_t: float | None = None
        self._last_x: NDArray[np.float64] | None = None
        self.bridged_gaps = 0

    @property
    def origin(self) -> float | None:
        """Grid origin (s): the first timestamp since the last reset, or None."""
        return self._origin

    @property
    def samples_emitted(self) -> int:
        """Output samples emitted since the last reset (= the next grid index)."""
        return self._next_k

    def grid_time(self, k: int | NDArray[np.int64]) -> float | NDArray[np.float64]:
        """Time of grid index ``k`` (the one formula every path uses)."""
        if self._origin is None:
            raise NeurOSSignalError("the grid has no origin yet (no samples since reset)")
        return self._origin + np.asarray(k, dtype=np.float64) * self._period

    # ------------------------------------------------------------------ validation
    def check_timestamps(
        self, timestamps: NDArray[np.floating], n_samples: int | None = None
    ) -> NDArray[np.float64]:
        """Validate a chunk's timestamps against the chunk and the previous chunk.

        Called by ``StreamingPreprocessor`` *before* filtering, so a rejected chunk never
        advances any state.

        Args:
            timestamps: one timestamp (s) per sample of the chunk.
            n_samples: the chunk's sample count (default: the length of ``timestamps``).

        Raises:
            NeurOSSignalError: non-finite, wrong shape, or not strictly increasing (also
                relative to the previous chunk's last timestamp).
            NeurOSSignalGapError: an interval longer than ``max_gap_s``.
        """
        if n_samples is None:
            is_vector = isinstance(timestamps, np.ndarray) and timestamps.ndim == 1
            n_samples = int(timestamps.shape[0]) if is_vector else -1
        t = check_timestamps(timestamps, n_samples, self.context)
        if t.size == 0:
            return t
        if self._last_t is not None and t[0] <= self._last_t:
            raise NeurOSSignalError(
                f"chunk starts at {t[0]!r} s, not after the previous sample at {self._last_t!r} s",
                self.context,
            )
        if self._last_t is not None:
            intervals = np.diff(np.concatenate([[self._last_t], t]))
        else:
            intervals = np.diff(t)
        longest = float(intervals.max()) if intervals.size else 0.0
        if longest > self.max_gap_s:
            raise NeurOSSignalGapError(
                f"{longest:.3f} s without samples (limit {self.max_gap_s:g} s): reset the stream",
                gap_s=longest,
                context=self.context,
            )
        return t

    # ------------------------------------------------------------------ processing
    def process(
        self, chunk: NDArray[np.floating], timestamps: NDArray[np.floating]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Resample one chunk. Returns ``(samples [channels, m], grid times [m])``.

        ``m`` may be 0 (the chunk ended before the next grid time) or larger than the chunk
        (upsampling). Inputs are validated before any state changes.

        Raises:
            NeurOSSignalError, NeurOSSignalGapError: see ``check_timestamps``; also on a bad chunk.
        """
        x = check_chunk(chunk, self.n_channels, self.context)
        t = self.check_timestamps(timestamps, n_samples=x.shape[1])
        empty = (np.zeros((self.n_channels, 0)), np.zeros(0))
        if t.size == 0:
            return empty

        # Bracketing samples may straddle the chunk boundary, so prepend the previous last sample.
        if self._last_t is not None and self._last_x is not None:
            T = np.concatenate([[self._last_t], t])
            X = np.concatenate([self._last_x[:, None], x], axis=1)
        else:
            T, X = t, x
        if self.nominal_sfreq is not None and T.size > 1:
            limit = _BRIDGED_GAP_PERIODS / self.nominal_sfreq
            self.bridged_gaps += int(np.count_nonzero(np.diff(T) > limit))

        origin = float(t[0]) if self._origin is None else self._origin
        k_last = self._last_grid_index(origin, float(T[-1]))
        # State is committed only after all checks have passed.
        self._origin = origin
        self._last_t = float(T[-1])
        self._last_x = X[:, -1].copy()
        if k_last < self._next_k:
            return empty

        ks = np.arange(self._next_k, k_last + 1, dtype=np.int64)
        grid = origin + ks.astype(np.float64) * self._period
        self._next_k = k_last + 1

        # For each grid time: T[i] <= t_k < T[i+1], or t_k == T[-1] (then i is the last index).
        i = np.searchsorted(T, grid, side="right") - 1
        i_next = np.minimum(i + 1, T.size - 1)
        span = T[i_next] - T[i]
        weight = np.where(i_next == i, 0.0, (grid - T[i]) / np.where(span > 0, span, 1.0))
        # (1-w)·a + w·b is exact at w = 0 and w = 1, so a grid time that coincides with an input
        # sample reproduces it bit for bit whichever chunk delivers it.
        out = X[:, i] * (1.0 - weight) + X[:, i_next] * weight
        return out, grid

    def _last_grid_index(self, origin: float, last_t: float) -> int:
        """Largest k with ``origin + k·period <= last_t`` (float-safe), or -1 if none."""
        k = math.floor((last_t - origin) * self.out_sfreq)
        # The floor estimate can be off by one through rounding; settle it with the exact formula.
        while origin + (k + 1) * self._period <= last_t:
            k += 1
        while k >= 0 and origin + k * self._period > last_t:
            k -= 1
        return k
