"""Resting SMR strength (Q31): the mu peak's height above the 1/f background at C3/C4."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from neuros.config.schemas import SignalContract
from neuros.data.synthetic import SyntheticConfig, make_recording
from neuros.errors import NeurOSConfigError, NeurOSDataError
from neuros.eval.smr import peak_above_floor, smr_strength, welch_psd
from neuros.signal.preprocess import preprocess_recording

from .conftest import make_spec

PROFILE = ("FC3", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP4")


def test_peak_height_over_a_known_power_law() -> None:
    freqs = np.arange(1.0, 40.5, 0.5)
    background = 100.0 / freqs  # 1/f: a straight line in log-log
    bump = np.where(np.abs(freqs - 10.0) < 0.75, 10.0, 1.0)  # x10 power at 10 Hz = +10 dB
    heights = peak_above_floor(freqs, np.stack([background, background * bump]))
    np.testing.assert_allclose(heights, [0.0, 10.0], atol=1e-9)
    with pytest.raises(NeurOSDataError, match="positive"):
        peak_above_floor(freqs, np.zeros((1, freqs.size)))
    with pytest.raises(NeurOSDataError, match="bins"):
        peak_above_floor(freqs, background[None], peak_band=(50.0, 60.0))


def test_welch_needs_a_whole_segment() -> None:
    rng = np.random.default_rng(0)
    freqs, power = welch_psd(rng.normal(size=(2, 1280)), 128.0)
    assert power.shape == (2, freqs.size)
    assert freqs[1] - freqs[0] == pytest.approx(0.5)  # 2-s segments
    with pytest.raises(NeurOSDataError, match="shorter"):
        welch_psd(rng.normal(size=(2, 100)), 128.0)
    with pytest.raises(NeurOSDataError):
        welch_psd(np.full((2, 1280), np.nan), 128.0)


def _person(mu_uv: float, seed: int) -> list:
    contract = SignalContract()  # 4-30 Hz: the background bands lie inside the pass band
    recording = make_recording(
        SyntheticConfig(n_trials=20, seed=seed, mu_uv=mu_uv), subject=1, session="0"
    )
    return [preprocess_recording(recording, contract)]


def test_a_strong_mu_rhythm_scores_higher_than_none() -> None:
    spec = make_spec()
    strong = smr_strength(_person(12.0, seed=1), spec, PROFILE, seconds=60.0)
    none = smr_strength(_person(0.0, seed=1), spec, PROFILE, seconds=60.0)
    assert strong["smr_db"] > none["smr_db"] + 3.0
    assert set(strong["per_channel_db"]) == {"C3", "C4"}
    assert strong["seconds_used"] == pytest.approx(60.0, abs=0.01)
    whole = smr_strength(_person(12.0, seed=1), spec, PROFILE, seconds=None)
    assert whole["seconds_used"] > 60.0


def test_first_session_and_bad_inputs() -> None:
    spec = make_spec()
    early = _person(12.0, seed=2)[0]
    later = dataclasses.replace(_person(0.0, seed=3)[0], session="1")
    result = smr_strength([later, early], spec, PROFILE, seconds=30.0)
    assert result["session"] == "0"  # the first session, whatever the input order
    with pytest.raises(NeurOSConfigError, match="lacks"):
        smr_strength([early], spec, ("FC3", "Cz", "C4"), seconds=30.0)
    with pytest.raises(NeurOSConfigError, match="seconds"):
        smr_strength([early], spec, PROFILE, seconds=0.0)
    with pytest.raises(NeurOSDataError, match="no recordings"):
        smr_strength([], spec, PROFILE, seconds=30.0)
    other = dataclasses.replace(early, subject=2)
    with pytest.raises(NeurOSDataError, match="one person"):
        smr_strength([early, other], spec, PROFILE, seconds=30.0)
