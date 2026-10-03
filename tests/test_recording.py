"""Structural invariants of Recording and EventTable."""

from __future__ import annotations

import numpy as np
import pytest

from neuros.data.recording import EventTable, Recording
from neuros.errors import NeurOSChannelError, NeurOSDataError


def _recording(**overrides: object) -> Recording:
    fields: dict[str, object] = {
        "data": np.zeros((2, 500), dtype=np.float32),
        "sfreq": 250.0,
        "ch_names": ("C3", "C4"),
        "events": EventTable.from_pairs([(300, "right_hand"), (100, "left_hand")]),
        "dataset": "Synthetic",
        "subject": 1,
        "session": "0",
        "run": "0",
    }
    fields.update(overrides)
    return Recording(**fields)  # type: ignore[arg-type]


def test_valid_recording() -> None:
    rec = _recording()
    assert rec.n_channels == 2
    assert rec.n_samples == 500
    assert rec.duration_s == 2.0
    assert rec.person_id == "Synthetic:1"
    assert rec.events.labels == ("left_hand", "right_hand")  # sorted by onset
    assert rec.events.counts() == {"left_hand": 1, "right_hand": 1}


def test_recording_arrays_are_read_only() -> None:
    rec = _recording()
    with pytest.raises(ValueError, match="read-only"):
        rec.data[0, 0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        rec.events.onsets[0] = 5


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"data": np.zeros((2, 500), dtype=np.float64)}, NeurOSDataError, "float32"),
        ({"data": np.zeros(500, dtype=np.float32)}, NeurOSDataError, "channels, samples"),
        ({"data": np.zeros((2, 0), dtype=np.float32)}, NeurOSDataError, "empty"),
        ({"data": [[0.0]]}, NeurOSDataError, "numpy array"),
        ({"ch_names": ("C3",)}, NeurOSDataError, "channel names"),
        ({"ch_names": ("c3", "C4")}, NeurOSChannelError, "Non-canonical"),
        ({"ch_names": ("C3", "C3")}, NeurOSChannelError, "Duplicate"),
        ({"sfreq": 0.0}, NeurOSDataError, "sfreq"),
        ({"sfreq": float("nan")}, NeurOSDataError, "sfreq"),
        ({"subject": -1}, NeurOSDataError, "subject"),
        (
            {"events": EventTable.from_pairs([(500, "rest")])},
            NeurOSDataError,
            "beyond the recording",
        ),
    ],
)
def test_invalid_recordings(
    overrides: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        _recording(**overrides)


def test_errors_carry_context() -> None:
    with pytest.raises(NeurOSDataError, match=r"dataset=Synthetic subject=1 session=0 run=0"):
        _recording(sfreq=-1.0)


@pytest.mark.parametrize(
    ("onsets", "labels", "message"),
    [
        (np.array([[1, 2]]), ("a", "b"), "1-D"),
        (np.array([1.5]), ("a",), "integers"),
        (np.array([1, 2]), ("a",), "labels"),
        (np.array([-1]), ("a",), ">= 0"),
        (np.array([5, 2]), ("a", "b"), "sorted"),
    ],
)
def test_invalid_event_tables(onsets: np.ndarray, labels: tuple[str, ...], message: str) -> None:
    with pytest.raises(NeurOSDataError, match=message):
        EventTable(onsets=onsets, labels=labels)


def test_empty_event_table() -> None:
    assert len(EventTable.empty()) == 0
