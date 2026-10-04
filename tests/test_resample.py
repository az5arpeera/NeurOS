"""Timestamp resampler (architecture §4.2): accuracy, exact chunking, jitter, gaps."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from neuros.errors import NeurOSConfigError, NeurOSSignalError, NeurOSSignalGapError
from neuros.signal.resample import TimestampResampler, default_timestamps


def _sine(times: np.ndarray, freq: float = 10.0) -> np.ndarray:
    return np.vstack([np.sin(2 * np.pi * freq * times), 0.5 * np.cos(2 * np.pi * freq * times)])


def test_sine_is_resampled_accurately() -> None:
    t = default_timestamps(2500, 250.0)
    out, grid = TimestampResampler(2, 128.0).process(_sine(t), t)
    assert grid.size == int(np.floor(t[-1] * 128)) + 1
    # Linear interpolation of 10 Hz at 250 Hz: error < 1 % of the amplitude.
    assert np.abs(out - _sine(grid)).max() < 0.01


def test_grid_times_are_exact_multiples_of_the_period() -> None:
    t = 100.0 + default_timestamps(1000, 250.0)
    _, grid = TimestampResampler(2, 128.0).process(_sine(t), t)
    assert grid[0] == 100.0
    assert np.array_equal(grid, 100.0 + np.arange(grid.size) / 128.0)


def test_grid_points_on_input_samples_reproduce_them_exactly() -> None:
    t = default_timestamps(512, 256.0)  # every other input sample is on the 128 Hz grid
    x = np.random.default_rng(0).standard_normal((2, 512))
    out, _ = TimestampResampler(2, 128.0).process(x, t)
    assert np.array_equal(out, x[:, ::2])


@given(cuts=st.lists(st.integers(min_value=1, max_value=999), max_size=15, unique=True))
def test_chunked_resampling_is_bit_identical(cuts: list[int]) -> None:
    t = default_timestamps(1000, 250.0)
    x = np.random.default_rng(1).standard_normal((2, 1000))
    whole, whole_t = TimestampResampler(2, 128.0).process(x, t)
    streaming = TimestampResampler(2, 128.0)
    bounds = [0, *sorted(cuts), 1000]
    pieces = [streaming.process(x[:, a:b], t[a:b]) for a, b in pairwise(bounds)]
    assert np.array_equal(np.concatenate([p[0] for p in pieces], axis=1), whole)
    assert np.array_equal(np.concatenate([p[1] for p in pieces]), whole_t)


def test_jittered_timestamps_are_honoured() -> None:
    # LSL-like jitter: samples arrive ±1 ms around their nominal times, but carry true stamps.
    rng = np.random.default_rng(2)
    nominal = default_timestamps(2500, 250.0)
    t = np.sort(nominal + rng.uniform(-0.001, 0.001, 2500))
    out, grid = TimestampResampler(2, 128.0).process(_sine(t), t)
    # Linear interpolation error bound: max|f''|·h²/8 with f'' = ω² (unit amplitude) and h the
    # longest interval between samples.
    omega = 2 * np.pi * 10.0
    bound = omega**2 * float(np.diff(t).max()) ** 2 / 8
    assert np.abs(out - _sine(grid)).max() <= bound
    # Pretending the samples sat at their nominal times (ignoring the stamps) is far worse.
    naive, _ = TimestampResampler(2, 128.0).process(_sine(t), nominal)
    assert np.abs(naive - _sine(grid)).max() > 2 * bound


def test_short_gaps_are_bridged_and_counted() -> None:
    t = default_timestamps(1000, 250.0)
    keep = np.r_[0:400, 410:1000]  # drop 10 samples (40 ms)
    resampler = TimestampResampler(2, 128.0, nominal_sfreq=250.0)
    x = _sine(t[keep])
    out, grid = resampler.process(x, t[keep])
    assert resampler.bridged_gaps == 1
    assert np.array_equal(grid, np.arange(grid.size) / 128.0)  # the grid has no hole
    inside = (grid > t[399]) & (grid < t[410])
    assert inside.any()
    assert np.abs(out[:, ~inside] - _sine(grid[~inside])).max() < 0.01
    # Inside the gap: a straight line between the last sample before and the first after.
    for channel in range(2):
        line = np.interp(grid[inside], [t[399], t[410]], [x[channel, 399], x[channel, 400]])
        np.testing.assert_allclose(out[channel, inside], line, rtol=0, atol=1e-12)


def test_long_gap_raises_and_reset_starts_a_new_grid() -> None:
    resampler = TimestampResampler(2, 128.0, max_gap_s=0.5)
    t1 = default_timestamps(250, 250.0)
    resampler.process(_sine(t1), t1)
    emitted = resampler.samples_emitted
    t2 = 2.0 + default_timestamps(250, 250.0)
    with pytest.raises(NeurOSSignalGapError) as info:
        resampler.process(_sine(t2), t2)
    assert info.value.gap_s == pytest.approx(2.0 - t1[-1])
    assert resampler.samples_emitted == emitted  # nothing changed
    resampler.reset()
    _, grid = resampler.process(_sine(t2), t2)
    assert grid[0] == 2.0


def test_timestamps_must_increase_within_and_across_chunks() -> None:
    resampler = TimestampResampler(1, 128.0)
    with pytest.raises(NeurOSSignalError, match="strictly increase"):
        resampler.process(np.zeros((1, 3)), np.array([0.0, 0.01, 0.01]))
    resampler.process(np.zeros((1, 2)), np.array([0.0, 0.004]))
    with pytest.raises(NeurOSSignalError, match="not after"):
        resampler.process(np.zeros((1, 2)), np.array([0.004, 0.008]))


@pytest.mark.parametrize(
    ("chunk", "stamps"),
    [
        (np.zeros((1, 3)), np.array([0.0, np.nan, 0.02])),  # NaN timestamp
        (np.zeros((1, 3)), np.array([0.0, 0.01])),  # one timestamp missing
        (np.zeros((1, 2)), np.zeros((2, 1))),  # 2-D timestamps
        (np.zeros((2, 2)), np.array([0.0, 0.01])),  # wrong channel count
        (np.array([[0.0, np.inf]]), np.array([0.0, 0.01])),  # Inf sample
    ],
)
def test_invalid_inputs_raise(chunk: np.ndarray, stamps: np.ndarray) -> None:
    with pytest.raises(NeurOSSignalError):
        TimestampResampler(1, 128.0).process(chunk, stamps)


def test_empty_chunk_and_single_sample() -> None:
    resampler = TimestampResampler(1, 128.0)
    out, grid = resampler.process(np.zeros((1, 0)), np.zeros(0))
    assert out.shape == (1, 0)
    assert grid.size == 0
    out, grid = resampler.process(np.array([[7.0]]), np.array([3.0]))
    assert out.tolist() == [[7.0]]
    assert grid.tolist() == [3.0]


def test_configuration_is_validated() -> None:
    for kwargs in ({"out_sfreq": 0.0}, {"out_sfreq": 128.0, "max_gap_s": 0.0}):
        with pytest.raises(NeurOSConfigError):
            TimestampResampler(1, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(NeurOSConfigError):
        TimestampResampler(0, 128.0)
    with pytest.raises(NeurOSConfigError):
        default_timestamps(10, 0.0)


def test_default_timestamps() -> None:
    assert default_timestamps(3, 250.0, start_sample=250).tolist() == [1.0, 1.004, 1.008]
