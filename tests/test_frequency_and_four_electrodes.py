"""A5 (frequency alignment) and A4 (four electrodes) building blocks, on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from neuros.errors import NeurOSConfigError, NeurOSDataError
from neuros.eval import four_electrodes as a4
from neuros.eval import frequency

SFREQ = 128.0


def _spectrum(peak_hz: float, height: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """1/f background plus a Gaussian bump at ``peak_hz`` (two identical channels)."""
    freqs = np.arange(0, 64.5, 0.5)
    safe = np.maximum(freqs, 0.5)
    power = 1.0 / safe + height * np.exp(-0.5 * ((freqs - peak_hz) / 0.8) ** 2) / safe
    return freqs, np.stack([power, power])


@pytest.mark.parametrize("peak", [8.5, 10.0, 11.5])
def test_peak_frequency_finds_the_bump(peak: float) -> None:
    freqs, power = _spectrum(peak)
    found, height = frequency.peak_frequency(freqs, power)
    assert found == pytest.approx(peak, abs=0.5)
    assert height > 1.0


def test_person_frequency_falls_back_without_a_clear_peak() -> None:
    freqs, power = _spectrum(9.0, height=0.0)
    out = frequency.person_frequency(freqs, power)
    assert out["clear"] is False
    assert out["used_hz"] == frequency.REFERENCE_HZ


def test_person_frequency_clips_extreme_peaks() -> None:
    freqs, power = _spectrum(13.5)
    out = frequency.person_frequency(freqs, power)
    assert out["clear"] is True
    assert out["used_hz"] == frequency.PEAK_CLIP_HZ[1]


def _windows(rhythm_hz: float, n: int = 6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(256) / SFREQ
    x = rng.normal(0, 0.5, (n, 3, 256))
    x[:, 0] += 5 * np.sin(2 * np.pi * rhythm_hz * t)  # one channel carries the rhythm
    return x


def test_relative_bands_follow_the_peak() -> None:
    # A 9-Hz rhythm sits in the "mu" band [1.0, 1.2) x f only when f = 9 (8-10.8 Hz: band 0).
    x = _windows(9.5)
    aligned = frequency.relative_band_power(x, SFREQ, 9.0).reshape(6, 3, 5)
    fixed = frequency.relative_band_power(x, SFREQ, 10.0).reshape(6, 3, 5)
    assert aligned.shape == fixed.shape == (6, 3, 5)
    assert np.argmax(aligned[0, 0]) == 1  # 9.5 Hz in [9.0, 10.8): the mu band
    assert np.argmax(fixed[0, 0]) == 0  # 9.5 Hz in [8, 10): the lower band


def test_relative_band_power_refuses_bad_input() -> None:
    with pytest.raises(NeurOSConfigError, match="outside"):
        frequency.relative_band_power(_windows(10.0), SFREQ, 14.0)
    bad = _windows(10.0)
    bad[0, 0, 0] = np.nan
    with pytest.raises(NeurOSDataError, match="finite"):
        frequency.relative_band_power(bad, SFREQ, 10.0)


def test_zscore_within_groups() -> None:
    features = np.array([[1.0], [3.0], [10.0], [30.0]])
    out = frequency.zscore_within(features, ["a", "a", "b", "b"])
    assert np.allclose(out.ravel(), [-1.0, 1.0, -1.0, 1.0])


def test_compare_runs_both_conditions_on_the_same_folds() -> None:
    rng = np.random.default_rng(1)
    windows, labels, people, peaks = [], [], [], {}
    for person in range(4):
        name = f"X:{person}"
        peaks[name] = 9.0 + 0.5 * person
        for trial in range(20):
            label = "left_hand" if trial % 2 else "right_hand"
            x = rng.normal(0, 1, (2, 256))
            if label == "left_hand":
                x[0] += 3 * np.sin(2 * np.pi * peaks[name] * np.arange(256) / SFREQ)
            windows.append(x)
            labels.append(label)
            people.append(name)
    out = frequency.compare(np.stack(windows), SFREQ, labels, people, peaks)
    assert set(out) == {"fixed", "aligned"}
    assert set(out["fixed"]) == set(out["aligned"]) == set(peaks)
    assert np.mean(list(out["aligned"].values())) > 0.8


def test_a5_verdict() -> None:
    fixed = {f"Y:{i}": 0.55 for i in range(10)}
    better = {f"Y:{i}": 0.58 + 0.001 * i for i in range(10)}
    passing = frequency.verdict({"Y": {"fixed": fixed, "aligned": better}})
    assert passing["outcome"] == "PASS"
    worse = {f"Y:{i}": 0.54 for i in range(10)}
    failing = frequency.verdict({"Y": {"fixed": fixed, "aligned": worse}})
    assert failing["outcome"] == "FAIL"
    assert failing["checks"]["no_dataset_below_0"] is False


def test_candidate_montages() -> None:
    montages = a4.candidate_montages()
    assert len(montages) == 136
    assert tuple(sorted(a4.TODAY)) in montages
    assert all({"C3", "C4"} <= set(m) and len(m) == 4 for m in montages)


def test_split_halves_is_fixed_and_disjoint() -> None:
    people = [f"E:{i}" for i in range(29)]
    first, second = a4.split_halves(people, "EEGMAT")
    assert (first, second) == a4.split_halves(list(reversed(people)), "EEGMAT")
    assert len(first) == 15
    assert len(second) == 14
    assert not set(first) & set(second)
    with pytest.raises(NeurOSDataError, match="at least 6"):
        a4.split_halves(people[:4], "EEGMAT")


def test_pick_respects_the_commands_floor() -> None:
    today = ("C3", "C4", "Cz", "Pz")
    selection = {
        today: {"states": 0.55, "commands": 0.60},
        ("C3", "C4", "F3", "Fz"): {"states": 0.70, "commands": 0.595},  # within 1 point
        ("C3", "C4", "O1", "O2"): {"states": 0.80, "commands": 0.55},  # costs commands
    }
    assert a4.pick(selection, today) == ("C3", "C4", "F3", "Fz")


def test_a4_verdict() -> None:
    people = [f"E:{i}" for i in range(12)]
    today_states = dict.fromkeys(people, 0.55)
    winner_states = {p: 0.62 + 0.001 * i for i, p in enumerate(people)}
    commands = dict.fromkeys(people, 0.6)
    out = a4.verdict(
        ("C3", "C4", "F3", "Fz"),
        ("C3", "C4", "Cz", "Pz"),
        {"winner": winner_states, "today": today_states},
        {"winner": commands, "today": commands},
    )
    assert out["outcome"] == "PASS"
    keep = a4.verdict(("C3", "C4", "Cz", "Pz"), ("C3", "C4", "Cz", "Pz"), {}, {})
    assert keep["outcome"].startswith("KEEP")
