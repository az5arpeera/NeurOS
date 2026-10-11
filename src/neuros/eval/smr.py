"""Resting sensorimotor-rhythm strength: can the first minute predict who NeurOS will work for?

**Why.** Motor imagery is read as a *drop* in the ~10 Hz sensorimotor (mu) rhythm over the
motor cortex opposite the imagined hand. A person whose rhythm is weak at rest has little to
drop, and BCIs work poorly for them. Blankertz et al. (2010, NeuroImage 51) found that the
height of this rhythm's spectral peak above the 1/f background at C3/C4, from a couple of
minutes of rest, predicted later BCI accuracy (r ≈ 0.5, about 80 people). If it holds on our
data, the engine can tell an app in its first minute, with no labels, whether this person is
likely to succeed (a "fit check"), instead of letting them struggle.

**The measure** (fixed before the 10 Oct run; see Q31 in the open questions):

- **Data:** the first ``seconds`` of the person's first session, after the filters' start-up,
  on the dataset's profile channels with the common average over the profile (E2's stream
  placement, ``neuros.eval.warmup``). Unlabeled: exactly what the engine sees first.
- **Spectrum:** Welch, 2-s Hann segments, 50 % overlap, median across segments (robust to a
  few artifacts), at C3 and C4.
- **Background:** a straight line in log10(power) vs log10(frequency), least squares on
  ``FLOOR_BANDS`` (4.5-7 Hz and 25-28 Hz), inside the recording's 4-30 Hz pass band and
  away from its edges.
- **Strength:** the largest height of the spectrum above that line within ``PEAK_BAND``
  (8-13 Hz), in dB, averaged over C3 and C4.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neuros.data.recording import Recording
from neuros.data.registry import DatasetSpec
from neuros.errors import DataContext, NeurOSConfigError, NeurOSDataError
from neuros.signal.filters import TRANSIENT_S

FLOOR_BANDS: tuple[tuple[float, float], ...] = ((4.5, 7.0), (25.0, 28.0))
PEAK_BAND: tuple[float, float] = (8.0, 13.0)
SMR_CHANNELS: tuple[str, ...] = ("C3", "C4")
SEGMENT_S = 2.0


def welch_psd(
    x: NDArray[np.floating], sfreq: float, *, segment_s: float = SEGMENT_S
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Median-averaged Welch spectrum of each row: ``(frequencies [f], power [rows, f])``.

    Raises:
        NeurOSDataError: on a non-2-D input, non-finite samples, or less than one segment.
    """
    from scipy.signal import welch

    data = np.asarray(x, dtype=np.float64)
    if data.ndim != 2 or not np.isfinite(data).all():
        raise NeurOSDataError(f"need finite [channels, samples] data, got shape {data.shape}")
    nperseg = round(segment_s * sfreq)
    if data.shape[1] < nperseg:
        raise NeurOSDataError(
            f"{data.shape[1] / sfreq:.1f} s of data is shorter than one {segment_s:g}-s segment"
        )
    freqs, power = welch(
        data, fs=sfreq, window="hann", nperseg=nperseg, noverlap=nperseg // 2, average="median"
    )
    return np.asarray(freqs, dtype=np.float64), np.asarray(power, dtype=np.float64)


def peak_above_floor(
    freqs: NDArray[np.float64],
    power: NDArray[np.float64],
    *,
    floor_bands: Sequence[tuple[float, float]] = FLOOR_BANDS,
    peak_band: tuple[float, float] = PEAK_BAND,
) -> NDArray[np.float64]:
    """Each row's largest height (dB) above its 1/f background line within ``peak_band``.

    Raises:
        NeurOSDataError: if a band holds no frequency bin, or the spectrum isn't positive.
    """
    p = np.atleast_2d(power)
    if p.shape[-1] != freqs.size:
        raise NeurOSDataError("power and frequencies disagree in length")
    positive = freqs > 0  # the 0-Hz bin has no place on a log axis
    freqs, p = freqs[positive], p[:, positive]
    if not (np.isfinite(p).all() and (p > 0).all()):
        raise NeurOSDataError("the spectrum must be positive and finite (a flat channel?)")
    floor_mask = np.zeros(freqs.size, dtype=bool)
    for low, high in floor_bands:
        floor_mask |= (freqs >= low) & (freqs <= high)
    peak_mask = (freqs >= peak_band[0]) & (freqs <= peak_band[1])
    if floor_mask.sum() < 2 or not peak_mask.any():
        raise NeurOSDataError(f"too few frequency bins in {floor_bands} or {peak_band}")
    log_f = np.log10(freqs)
    log_p = np.log10(p)
    # One least-squares line per row: log_p = a + b log_f on the background bins.
    design = np.stack([np.ones(int(floor_mask.sum())), log_f[floor_mask]], axis=1)
    coef, *_ = np.linalg.lstsq(design, log_p[:, floor_mask].T, rcond=None)  # [2, rows]
    floor = coef[0][:, None] + coef[1][:, None] * log_f[None, :]
    height_db = 10.0 * (log_p - floor)
    return np.asarray(height_db[:, peak_mask].max(axis=1), dtype=np.float64)


def smr_strength(
    recordings: Sequence[Recording],
    spec: DatasetSpec,
    profile: Sequence[str],
    *,
    seconds: float | None,
) -> dict[str, Any]:
    """A person's resting SMR strength from the start of their first session.

    Args:
        recordings: the person's preprocessed recordings (any order; the first session's runs
            are taken in run order and concatenated).
        profile: the dataset's channels in the model layout (E2's profile; must include C3, C4).
        seconds: how much of the session's start to use (None: the whole first session).

    Returns:
        ``{"smr_db", "per_channel_db", "session", "seconds_used"}``.

    Raises:
        NeurOSConfigError: on a non-positive ``seconds`` or a profile without C3/C4.
        NeurOSDataError: on no recordings, mixed people or rates, or too little data.
    """
    picked, sfreq, first = first_session_rows(recordings, spec, profile, seconds=seconds)
    freqs, power = welch_psd(picked, sfreq)
    heights = peak_above_floor(freqs, power)
    return {
        "smr_db": float(heights.mean()),
        "per_channel_db": dict(zip(SMR_CHANNELS, heights.tolist(), strict=True)),
        "session": first,
        "seconds_used": picked.shape[1] / sfreq,
    }


def first_session_rows(
    recordings: Sequence[Recording],
    spec: DatasetSpec,
    profile: Sequence[str],
    *,
    seconds: float | None,
) -> tuple[NDArray[np.float32], float, str]:
    """The start of a person's first session at C3 and C4, as the engine would see it first.

    The first session's runs are taken in run order, each run's filter start-up is dropped,
    and the runs are joined; then the first ``seconds`` are kept (None: all of it). Channels go
    through the dataset's profile (E2's placement and common average).

    Returns:
        ``(rows [SMR_CHANNELS, samples], sampling rate, the session's name)``.

    Raises:
        NeurOSConfigError: on a non-positive ``seconds`` or a profile without C3/C4.
        NeurOSDataError: on no recordings, mixed people or rates.
    """
    from neuros.eval.calibration import natural_key
    from neuros.eval.warmup import _profile_rows

    if seconds is not None and not (math.isfinite(seconds) and seconds > 0):
        raise NeurOSConfigError(f"seconds must be None or > 0, got {seconds}")
    missing = [ch for ch in SMR_CHANNELS if ch not in profile]
    if missing:
        raise NeurOSConfigError(f"the profile lacks {missing}: {list(profile)}")
    if not recordings:
        raise NeurOSDataError(f"{spec.key}: no recordings")
    if len({(r.dataset, r.subject) for r in recordings}) != 1:
        raise NeurOSDataError("needs one person's recordings")
    rates = {r.sfreq for r in recordings}
    if len(rates) != 1:
        raise NeurOSDataError(f"recordings differ in sampling rate: {sorted(rates)}")
    sfreq = float(next(iter(rates)))
    first = min((r.session for r in recordings), key=natural_key)
    runs = sorted((r for r in recordings if r.session == first), key=lambda r: natural_key(r.run))
    transient = math.ceil(TRANSIENT_S * sfreq)
    parts: list[NDArray[np.float32]] = []
    for recording in runs:  # each run's filter start-up is dropped, then runs are joined
        context = DataContext(spec.key, recording.subject, first, recording.run)
        rows = _profile_rows(recording.data, recording.ch_names, spec, profile, context)
        parts.append(rows[:, transient:])
    data = np.concatenate(parts, axis=1)
    if seconds is not None:
        data = data[:, : round(seconds * sfreq)]
    picked = data[[list(profile).index(ch) for ch in SMR_CHANNELS]]
    return picked, sfreq, first
