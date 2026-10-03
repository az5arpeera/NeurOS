"""The synthetic generator must be deterministic and show contralateral ERD."""

from __future__ import annotations

import numpy as np
import pytest

from neuros.data.recording import Recording
from neuros.data.synthetic import SyntheticConfig, make_recording
from neuros.errors import NeurOSConfigError


def _mu_power(rec: Recording, channel: str, label: str) -> float:
    """Mean 8-12 Hz power at ``channel`` over [cue + 0.5, cue + 2.5] s for trials of ``label``."""
    ch = rec.ch_names.index(channel)
    start, stop = round(0.5 * rec.sfreq), round(2.5 * rec.sfreq)
    freqs = np.fft.rfftfreq(stop - start, d=1.0 / rec.sfreq)
    band = (freqs >= 8) & (freqs <= 12)
    powers = [
        np.abs(np.fft.rfft(rec.data[ch, onset + start : onset + stop])[band]) ** 2
        for onset, lab in zip(rec.events.onsets, rec.events.labels, strict=True)
        if lab == label
    ]
    return float(np.mean(powers))


def test_deterministic() -> None:
    a, b = make_recording(), make_recording()
    assert np.array_equal(a.data, b.data)
    assert a.events == b.events
    c = make_recording(SyntheticConfig(seed=1))
    assert not np.array_equal(a.data, c.data)


def test_balanced_labels() -> None:
    rec = make_recording(SyntheticConfig(n_trials=40))
    assert rec.events.counts() == {"left_hand": 20, "right_hand": 20}


def test_contralateral_erd() -> None:
    rec = make_recording(SyntheticConfig(n_trials=60, seed=3))
    # Right-hand imagery suppresses mu over the LEFT motor cortex (C3), and vice versa.
    assert _mu_power(rec, "C3", "right_hand") < 0.7 * _mu_power(rec, "C3", "left_hand")
    assert _mu_power(rec, "C4", "left_hand") < 0.7 * _mu_power(rec, "C4", "right_hand")


def test_plausible_amplitude() -> None:
    rec = make_recording()
    rms = float(np.sqrt(np.mean(rec.data.astype(np.float64) ** 2)))
    assert 5.0 < rms < 50.0  # µV


@pytest.mark.parametrize(
    "kwargs",
    [
        {"labels": ("jump",)},
        {"erd_depth": 1.0},
        {"n_trials": 0},
        {"sfreq": 20.0},
        {"iti_s": (2.0, 1.0)},
    ],
)
def test_invalid_config(kwargs: dict[str, object]) -> None:
    with pytest.raises(NeurOSConfigError):
        SyntheticConfig(**kwargs)  # type: ignore[arg-type]
