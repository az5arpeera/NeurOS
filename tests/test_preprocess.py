"""The shared preprocessing chain, including the golden train–serve parity test (§3, §9).

Principle 1 (architecture §3): "A golden test asserts that the offline pipeline and the online
engine produce identical windows from the same recording." The engine itself arrives in week 2;
its signal path is exactly ``StreamingPreprocessor`` → float32 → ``WindowStream``, which is what
``_stream`` below drives with arbitrary chunk sizes.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.signal import welch

from neuros.config.schemas import SignalContract
from neuros.data.recording import Recording
from neuros.data.synthetic import SyntheticConfig, make_recording
from neuros.errors import NeurOSConfigError, NeurOSSignalError, NeurOSSignalGapError
from neuros.signal.preprocess import StreamingPreprocessor, map_onsets, preprocess_recording
from neuros.signal.resample import default_timestamps
from neuros.signal.windows import (
    WindowStream,
    continuous_windows,
    extract_epochs,
    sliding_windows,
)

CONTRACT = SignalContract()


@pytest.fixture(scope="module")
def recording() -> Recording:
    return make_recording(SyntheticConfig(n_trials=12, seed=11))


@pytest.fixture(scope="module")
def offline(recording: Recording) -> Recording:
    return preprocess_recording(recording, CONTRACT)


def _stream(
    recording: Recording, chunk_sizes: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drive the online signal path chunk by chunk; return (samples, windows, window starts)."""
    chain = StreamingPreprocessor(recording.n_channels, recording.sfreq, CONTRACT)
    stream = WindowStream(recording.n_channels, CONTRACT.window_samples, CONTRACT.hop_samples)
    stamps = default_timestamps(recording.n_samples, recording.sfreq)
    samples, windows, starts = [], [], []
    position, turn = 0, 0
    while position < recording.n_samples:
        size = chunk_sizes[turn % len(chunk_sizes)]
        stop = min(position + size, recording.n_samples)
        out, _ = chain.process(recording.data[:, position:stop], stamps[position:stop])
        out32 = out.astype(np.float32)  # the contract dtype, as offline
        win, win_starts = stream.push(out32)
        samples.append(out32)
        windows.append(win)
        starts.append(win_starts)
        position, turn = stop, turn + 1
    return np.concatenate(samples, axis=1), np.concatenate(windows), np.concatenate(starts)


@settings(max_examples=10, deadline=None)
@given(chunk_sizes=st.lists(st.integers(min_value=1, max_value=400), min_size=1, max_size=6))
def test_golden_parity_offline_equals_streaming(
    recording: Recording, offline: Recording, chunk_sizes: list[int]
) -> None:
    samples, windows, starts = _stream(recording, chunk_sizes)
    assert np.array_equal(samples, offline.data)
    expected = continuous_windows(offline, CONTRACT)
    assert np.array_equal(windows, expected.data)
    assert np.array_equal(starts, expected.starts)


def test_preprocessed_recording_follows_the_contract(
    recording: Recording, offline: Recording
) -> None:
    assert offline.sfreq == CONTRACT.sfreq
    assert offline.data.dtype == np.float32
    assert offline.ch_names == recording.ch_names
    assert offline.n_samples == int(np.floor((recording.n_samples - 1) / recording.sfreq * 128)) + 1
    assert offline.events.labels == recording.events.labels
    np.testing.assert_array_equal(
        offline.events.onsets, np.rint(recording.events.onsets * 128 / recording.sfreq)
    )
    for key in ("filter_band_hz", "filter_order", "filter_notch_hz", "resampler", "native_sfreq"):
        assert key in offline.provenance
    assert offline.provenance["loader"] == "synthetic"  # upstream provenance is kept


def test_motor_imagery_signal_survives_preprocessing(offline: Recording) -> None:
    # Left-hand imagery suppresses mu over C4, right-hand over C3 (synthetic ERD).
    windows, _ = extract_epochs(offline, CONTRACT)
    labels = np.array(windows.labels)

    def mu_power(label: str, channel: str) -> float:
        x = windows.data[labels == label][:, windows.ch_names.index(channel)]
        freqs, power = welch(x, fs=CONTRACT.sfreq, nperseg=128)
        return float(power[:, (freqs >= 8) & (freqs <= 12)].mean())

    assert mu_power("left_hand", "C4") < 0.7 * mu_power("right_hand", "C4")
    assert mu_power("right_hand", "C3") < 0.7 * mu_power("left_hand", "C3")


def test_upsampling_is_refused() -> None:
    with pytest.raises(NeurOSConfigError, match="upsampling"):
        StreamingPreprocessor(2, 100.0, CONTRACT)


def test_rejected_chunks_leave_the_chain_untouched(recording: Recording) -> None:
    stamps = default_timestamps(recording.n_samples, recording.sfreq)
    first, second = slice(0, 300), slice(300, 600)
    clean = StreamingPreprocessor(recording.n_channels, recording.sfreq, CONTRACT)
    clean.process(recording.data[:, first], stamps[first])
    expected, _ = clean.process(recording.data[:, second], stamps[second])

    chain = StreamingPreprocessor(recording.n_channels, recording.sfreq, CONTRACT)
    chain.process(recording.data[:, first], stamps[first])
    bad = recording.data[:, second].copy()
    bad[0, 5] = np.nan
    with pytest.raises(NeurOSSignalError):
        chain.process(bad, stamps[second])
    gap = stamps[second] + 5.0  # 5 s later: a long gap
    with pytest.raises(NeurOSSignalGapError):
        chain.process(recording.data[:, second], gap)
    got, _ = chain.process(recording.data[:, second], stamps[second])
    assert np.array_equal(got, expected)


def test_reset_after_a_gap_restarts_the_grid(recording: Recording) -> None:
    chain = StreamingPreprocessor(recording.n_channels, recording.sfreq, CONTRACT)
    stamps = default_timestamps(600, recording.sfreq)
    chain.process(recording.data[:, :300], stamps[:300])
    chain.reset()
    _, grid = chain.process(recording.data[:, 300:600], stamps[300:600] + 60.0)
    assert grid[0] == stamps[300] + 60.0


def test_map_onsets_rounds_and_clamps() -> None:
    onsets = np.array([0, 500, 1001, 1999], dtype=np.int64)
    assert map_onsets(onsets, 250.0, 128.0, n_out=1023).tolist() == [0, 256, 513, 1022]


def test_sliding_windows_of_the_offline_signal_match_continuous_windows(
    offline: Recording,
) -> None:
    windows = continuous_windows(offline, CONTRACT)
    view = sliding_windows(offline.data, CONTRACT.window_samples, CONTRACT.hop_samples)
    assert np.array_equal(windows.data, view)
