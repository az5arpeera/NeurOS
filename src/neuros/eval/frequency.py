"""A5: does lining up each person's brain-rhythm frequency help zero-calibration decoding?

The decision and pass bar are in ``knowledge-base/plan/a5-frequency-alignment.md`` (written
before this code). In short:

1. **Each person's peak frequency**, unlabeled: the first 60 s of their first session at C3
   and C4 (Q31's data and spectrum), the frequency of the largest height above the 1/f
   background within 7-14 Hz. With no clear peak (< 1 dB) the person keeps 10 Hz; peaks are
   clipped to 8.5-11.5 Hz.
2. **Features:** per window, common average over the motor strip, Hann periodogram, log mean
   power in five bands with edges ``BAND_EDGES`` × *f*, where *f* is the person's peak
   ("aligned") or 10 Hz for everyone ("fixed"). Each person's features are z-scored over their
   own windows (unlabeled, both conditions).
3. **Model:** standardise + logistic regression, leave one person out within each dataset.

The same windows, folds and model run in both conditions; only the band edges move.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neuros.errors import NeurOSConfigError, NeurOSDataError

# Band edges as multiples of the person's peak frequency (10 Hz → 8, 10, 12, 16, 20, 26 Hz):
# theta-alpha edge, mu, upper mu, low beta, high beta.
BAND_EDGES: tuple[float, ...] = (0.8, 1.0, 1.2, 1.6, 2.0, 2.6)
REFERENCE_HZ = 10.0
PEAK_SEARCH_HZ: tuple[float, float] = (7.0, 14.0)
PEAK_CLIP_HZ: tuple[float, float] = (8.5, 11.5)
MIN_PEAK_DB = 1.0
_LR_C = 1.0  # sklearn's default, made explicit; never tuned
_LR_MAX_ITER = 2000


def peak_frequency(
    freqs: NDArray[np.float64],
    power: NDArray[np.float64],
    *,
    search_hz: tuple[float, float] = PEAK_SEARCH_HZ,
) -> tuple[float, float]:
    """The frequency and height (dB) of the largest bump above the 1/f background.

    ``power`` may hold several channels; their spectra are averaged first (C3 and C4 share
    one rhythm frequency in practice, and averaging halves the noise).

    Raises:
        NeurOSDataError: if the search band holds no bin, or the spectrum isn't positive.
    """
    from neuros.eval.smr import FLOOR_BANDS

    mean = np.atleast_2d(power).mean(axis=0)
    positive = freqs > 0
    f, p = freqs[positive], mean[positive]
    if not (np.isfinite(p).all() and (p > 0).all()):
        raise NeurOSDataError("the spectrum must be positive and finite (a flat channel?)")
    floor = np.zeros(f.size, dtype=bool)
    for low, high in FLOOR_BANDS:
        floor |= (f >= low) & (f <= high)
    search = (f >= search_hz[0]) & (f <= search_hz[1])
    if floor.sum() < 2 or not search.any():
        raise NeurOSDataError(f"too few frequency bins in {FLOOR_BANDS} or {search_hz}")
    log_f, log_p = np.log10(f), np.log10(p)
    slope, intercept = np.polyfit(log_f[floor], log_p[floor], 1)
    height_db = 10.0 * (log_p - (intercept + slope * log_f))
    index = np.flatnonzero(search)[int(np.argmax(height_db[search]))]
    return float(f[index]), float(height_db[index])


def person_frequency(
    freqs: NDArray[np.float64], power: NDArray[np.float64]
) -> dict[str, float | bool]:
    """The frequency the person's features are aligned to, with how it was found.

    Returns ``{"peak_hz", "height_db", "clear", "used_hz"}``: ``used_hz`` is the clipped
    peak when it's clear (≥ ``MIN_PEAK_DB``), else ``REFERENCE_HZ``.
    """
    peak, height = peak_frequency(freqs, power)
    clear = height >= MIN_PEAK_DB
    used = float(np.clip(peak, *PEAK_CLIP_HZ)) if clear else REFERENCE_HZ
    return {"peak_hz": peak, "height_db": height, "clear": bool(clear), "used_hz": used}


def relative_band_power(
    windows: NDArray[np.floating], sfreq: float, peak_hz: float
) -> NDArray[np.float64]:
    """Log mean power per channel in the bands ``BAND_EDGES`` × ``peak_hz``: ``[n, ch * 5]``.

    Windows are common-average referenced over their channels first.

    Raises:
        NeurOSDataError: on non-finite input, or a band with no frequency bin.
        NeurOSConfigError: on a peak outside the clipping range (a caller bug).
    """
    if not PEAK_CLIP_HZ[0] - 1e-9 <= peak_hz <= max(PEAK_CLIP_HZ[1], REFERENCE_HZ) + 1e-9:
        raise NeurOSConfigError(f"peak {peak_hz} Hz is outside {PEAK_CLIP_HZ}")
    x = np.asarray(windows, dtype=np.float64)
    if x.ndim != 3 or not np.isfinite(x).all():
        raise NeurOSDataError(f"need finite [windows, channels, samples], got {x.shape}")
    x = x - x.mean(axis=1, keepdims=True)
    n_samples = x.shape[-1]
    taper = np.hanning(n_samples)
    spectrum = np.abs(np.fft.rfft(x * taper, axis=-1)) ** 2 / (sfreq * (taper**2).sum())
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
    edges = np.asarray(BAND_EDGES) * peak_hz
    columns = []
    for low, high in itertools.pairwise(edges):
        bins = (freqs >= low) & (freqs < high)
        if not bins.any():
            raise NeurOSDataError(f"band {low:.1f}-{high:.1f} Hz has no frequency bin")
        columns.append(np.log10(spectrum[..., bins].mean(axis=-1) + 1e-12))
    return np.stack(columns, axis=-1).reshape(x.shape[0], -1)


def zscore_within(features: NDArray[np.float64], groups: Sequence[str]) -> NDArray[np.float64]:
    """Z-score each group's (person's) rows over that group only: unlabeled normalisation."""
    out = np.empty_like(features)
    keys = np.asarray(groups)
    for key in dict.fromkeys(keys.tolist()):
        rows = keys == key
        mean = features[rows].mean(axis=0)
        std = np.maximum(features[rows].std(axis=0), 1e-6)
        out[rows] = (features[rows] - mean) / std
    return out


def balanced_accuracy(truth: Sequence[str], predicted: Sequence[str]) -> float:
    """Mean recall over the classes present in ``truth``."""
    t, p = np.asarray(truth), np.asarray(predicted)
    return float(np.mean([(p[t == c] == c).mean() for c in np.unique(t)]))


def loso_accuracy(
    features: NDArray[np.float64], labels: Sequence[str], people: Sequence[str]
) -> dict[str, float]:
    """Leave one person out: each person's balanced accuracy from a model trained on the rest.

    Raises:
        NeurOSDataError: with fewer than 2 people or a training set with one class.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y, groups = np.asarray(labels), np.asarray(people)
    names = list(dict.fromkeys(groups.tolist()))
    if len(names) < 2:
        raise NeurOSDataError("leave-one-person-out needs at least 2 people")
    scores: dict[str, float] = {}
    for person in names:
        test = groups == person
        if len(set(y[~test].tolist())) < 2:
            raise NeurOSDataError("a training fold lacks a class")
        model = make_pipeline(StandardScaler(), LogisticRegression(C=_LR_C, max_iter=_LR_MAX_ITER))
        model.fit(features[~test], y[~test])
        scores[person] = balanced_accuracy(y[test], model.predict(features[test]))
    return scores


def compare(
    windows: NDArray[np.floating],
    sfreq: float,
    labels: Sequence[str],
    people: Sequence[str],
    used_hz: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    """Per-person accuracy with fixed (10 Hz) and aligned bands, on the same windows and folds.

    Raises:
        NeurOSDataError: if a person has no aligned frequency.
    """
    groups = np.asarray(people)
    missing = sorted(set(groups.tolist()) - set(used_hz))
    if missing:
        raise NeurOSDataError(f"no peak frequency for {missing[:5]}")
    out: dict[str, dict[str, float]] = {}
    for condition in ("fixed", "aligned"):
        features = np.empty((len(groups), windows.shape[1] * (len(BAND_EDGES) - 1)))
        for person in dict.fromkeys(groups.tolist()):
            rows = groups == person
            f = REFERENCE_HZ if condition == "fixed" else used_hz[person]
            features[rows] = relative_band_power(windows[rows], sfreq, f)
        out[condition] = loso_accuracy(zscore_within(features, people), labels, people)
    return out


def verdict(per_dataset: Mapping[str, Mapping[str, Mapping[str, float]]]) -> dict[str, Any]:
    """The A5 bar: mean gain ≥ +2.0 points over all people, Wilcoxon p < 0.05, no dataset < 0.

    ``per_dataset`` maps dataset → condition → person → accuracy.
    """
    from scipy.stats import wilcoxon

    gains: list[float] = []
    by_dataset: dict[str, float] = {}
    for dataset, conditions in per_dataset.items():
        people = list(conditions["fixed"])
        diff = [conditions["aligned"][p] - conditions["fixed"][p] for p in people]
        gains.extend(diff)
        by_dataset[dataset] = float(np.mean(diff))
    gain = np.asarray(gains)
    p_value = 1.0 if np.allclose(gain, 0) else float(wilcoxon(gain).pvalue)
    checks = {
        "gain_at_least_2_points": float(gain.mean()) * 100 >= 2.0,
        "p_below_0.05": p_value < 0.05,
        "no_dataset_below_0": all(v >= 0 for v in by_dataset.values()),
    }
    return {
        "outcome": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "mean_gain_points": float(gain.mean()) * 100,
        "wilcoxon_p": p_value,
        "gain_by_dataset_points": {k: v * 100 for k, v in by_dataset.items()},
        "n_people": int(gain.size),
    }
