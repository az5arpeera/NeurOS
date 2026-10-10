"""The quality gate (architecture §7.1, stage 4): is this window clean enough to decide on?

Per channel, on the preprocessed window (µV, band-passed, before the common average):

| check | rule | reason code |
|---|---|---|
| flat | standard deviation below ``flat_uv`` | ``flat`` (disconnected or shorted electrode) |
| saturated | any sample beyond ±``saturated_uv`` | ``saturated`` (amplifier at its rail) |
| motion | peak-to-peak (after the common average) above ``ptp_uv`` | ``motion`` |

The motion rule is E2's reference gate (``ValidationThresholds.ptp_flag_uv``, 500 µV, on the
common-averaged window), so the engine adapts on exactly the windows E2 measured. Flat and
saturated are checked *before* the common average: subtracting the mean of the other channels
would hide a flat electrode.

A window that fails is never used for adaptation and never decided on: the engine emits
``HOLD`` with the first failing reason (flat, then saturated, then motion: the order in which
they make the others meaningless).

**Provisional thresholds.** §7.1 plans to tune them on Cho2017's non-task recordings for < 5%
false HOLDs on clean data; until then, flat and saturated are set wide (they catch hardware
faults, not artifacts). The muscle (EMG) check needs power above the 30-Hz band edge, which
the band-pass removes; it comes with the raw-signal path.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neuros.config.schemas import ValidationThresholds
from neuros.engine.events import HoldReason
from neuros.errors import NeurOSConfigError, NeurOSDataError
from neuros.signal.reference import common_average_reference


@dataclass(frozen=True)
class QualityThresholds:
    """Gate thresholds in µV (see the module docstring for what each catches)."""

    flat_uv: float = 0.5
    saturated_uv: float = 2000.0
    ptp_uv: float = ValidationThresholds().ptp_flag_uv

    def __post_init__(self) -> None:
        for name in ("flat_uv", "saturated_uv", "ptp_uv"):
            value = getattr(self, name)
            if not (isinstance(value, int | float) and math.isfinite(value) and value > 0):
                raise NeurOSConfigError(f"{name} must be a positive finite number, got {value!r}")
        if self.flat_uv >= self.ptp_uv:
            raise NeurOSConfigError(
                f"flat_uv ({self.flat_uv}) must be below ptp_uv ({self.ptp_uv})"
            )


@dataclass(frozen=True)
class QualityResult:
    """The gate's verdict on one window."""

    ok: bool
    reason: HoldReason | None  # the first failing check; None exactly when ok
    bad_channels: tuple[str, ...]  # every channel that failed any check, in channel order

    def __post_init__(self) -> None:
        if self.ok != (self.reason is None):
            raise NeurOSDataError(f"a verdict has a reason exactly when it fails: {self}")


class QualityGate:
    """Checks windows of fixed channels against ``QualityThresholds``."""

    def __init__(
        self, ch_names: Sequence[str], thresholds: QualityThresholds | None = None
    ) -> None:
        if len(ch_names) < 2 or len(set(ch_names)) != len(ch_names):
            raise NeurOSConfigError(f"need at least 2 distinct channels, got {list(ch_names)}")
        self.ch_names = tuple(ch_names)
        self.thresholds = thresholds or QualityThresholds()

    def check(self, window: NDArray[np.floating]) -> QualityResult:
        """The verdict on one window ``[channels, samples]`` (µV, before the common average).

        Raises:
            NeurOSDataError: on a wrong shape or non-finite samples (the engine validates
                chunks before windowing, so this is a programming error, not bad signal).
        """
        if (
            not isinstance(window, np.ndarray)
            or window.ndim != 2
            or window.shape[0] != len(self.ch_names)
            or window.shape[1] < 2
        ):
            raise NeurOSDataError(
                f"window must be [{len(self.ch_names)}, samples >= 2], "
                f"got {getattr(window, 'shape', None)}"
            )
        x = window.astype(np.float64, copy=False)
        if not np.isfinite(x).all():
            raise NeurOSDataError("window contains NaN/Inf samples")
        limits = self.thresholds
        flat = x.std(axis=1) < limits.flat_uv
        saturated = np.abs(x).max(axis=1) > limits.saturated_uv
        referenced = common_average_reference(x)
        motion = np.ptp(referenced, axis=1) > limits.ptp_uv
        failed: list[tuple[HoldReason, NDArray[np.bool_]]] = [
            ("flat", flat),
            ("saturated", saturated),
            ("motion", motion),
        ]
        reason: HoldReason | None = next((name for name, mask in failed if mask.any()), None)
        bad = flat | saturated | motion
        return QualityResult(
            ok=reason is None,
            reason=reason,
            bad_channels=tuple(ch for ch, b in zip(self.ch_names, bad, strict=True) if b),
        )
