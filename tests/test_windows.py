"""Windows (architecture §4.2): epochs, sliding windows, the streaming buffer and WindowSet."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from neuros.config.schemas import SignalContract, ValidationThresholds
from neuros.data.recording import EventTable, Recording
from neuros.errors import NeurOSConfigError, NeurOSDataError, NeurOSSignalError
from neuros.signal.windows import (
    WindowFlag,
    WindowSet,
    WindowStream,
    continuous_windows,
    extract_epochs,
    sliding_windows,
    window_starts,
)

CONTRACT = SignalContract()
WINDOW = CONTRACT.window_samples  # 256
HOP = CONTRACT.hop_samples  # 16


def _recording(
    n_samples: int = 2000,
    events: list[tuple[int, str]] | None = None,
    sfreq: float = 128.0,
    seed: int = 0,
) -> Recording:
    rng = np.random.default_rng(seed)
    return Recording(
        data=(rng.standard_normal((3, n_samples)) * 10).astype(np.float32),
        sfreq=sfreq,
        ch_names=("C3", "Cz", "C4"),
        events=EventTable.from_pairs(events or []),
        dataset="Synthetic",
        subject=1,
        session="0",
        run="0",
    )


# ---------------------------------------------------------------------- sliding windows
def test_window_starts_follow_the_hop() -> None:
    assert window_starts(300, 256, 16).tolist() == [0, 16, 32]
    assert window_starts(255, 256, 16).size == 0
    with pytest.raises(NeurOSConfigError):
        window_starts(300, 0, 16)


def test_sliding_windows_is_a_read_only_view_with_the_right_content() -> None:
    data = np.arange(3 * 300, dtype=np.float32).reshape(3, 300)
    windows = sliding_windows(data, 256, 16)
    assert windows.shape == (3, 3, 256)
    assert np.array_equal(windows[2], data[:, 32:288])
    assert not windows.flags.writeable


@given(cuts=st.lists(st.integers(min_value=1, max_value=899), max_size=20, unique=True))
def test_stream_emits_exactly_the_offline_windows(cuts: list[int]) -> None:
    data = np.random.default_rng(3).standard_normal((2, 900)).astype(np.float32)
    stream = WindowStream(2, WINDOW, HOP)
    bounds = [0, *sorted(cuts), 900]
    pushed = [stream.push(data[:, a:b]) for a, b in pairwise(bounds)]
    windows = np.concatenate([w for w, _ in pushed])
    starts = np.concatenate([s for _, s in pushed])
    assert np.array_equal(windows, sliding_windows(data, WINDOW, HOP))
    assert np.array_equal(starts, window_starts(900, WINDOW, HOP))


def test_stream_reset_and_bad_chunks() -> None:
    stream = WindowStream(2, 4, 2)
    windows, starts = stream.push(np.ones((2, 5)))
    assert starts.tolist() == [0]
    assert windows.shape == (1, 2, 4)
    with pytest.raises(NeurOSSignalError):
        stream.push(np.full((2, 3), np.nan))
    assert stream.samples_seen == 5  # the rejected chunk changed nothing
    stream.reset()
    windows, starts = stream.push(np.ones((2, 3)))
    assert windows.shape == (0, 2, 4)
    assert stream.samples_seen == 3


# ---------------------------------------------------------------------- epochs
EPOCH = (0.5, 2.5)  # a fixed epoch; the contract's own rule needs a dataset (task_epoch)


def test_epochs_start_where_the_epoch_says() -> None:
    rec = _recording(events=[(200, "left_hand"), (700, "right_hand")])
    windows, skipped = extract_epochs(rec, CONTRACT, epoch_s=EPOCH)
    assert skipped == 0
    assert windows.starts.tolist() == [264, 764]  # cue + 0.5 s × 128 Hz
    assert np.array_equal(windows.data[1], rec.data[:, 764 : 764 + WINDOW])
    assert windows.labels == ("left_hand", "right_hand")
    assert windows.person_ids == ("Synthetic:1", "Synthetic:1")
    # A contract that fixes its own epoch needs no argument.
    windows, _ = extract_epochs(rec, SignalContract(epoch_s=(1.5, 3.5)))
    assert windows.starts.tolist() == [392, 892]  # cue + 1.5 s × 128 Hz


def test_epochs_need_an_explicit_one_window_epoch() -> None:
    rec = _recording(events=[(200, "left_hand")])
    # The contract's D-030 rule depends on the dataset, which a recording doesn't carry.
    with pytest.raises(NeurOSConfigError, match="task_epoch"):
        extract_epochs(rec, CONTRACT)
    with pytest.raises(NeurOSConfigError, match="one 2-s window"):
        extract_epochs(rec, CONTRACT, epoch_s=(0.5, 3.0))


def test_epochs_filter_labels_and_count_events_past_the_end() -> None:
    rec = _recording(events=[(100, "left_hand"), (300, "rest"), (1800, "right_hand")])
    windows, skipped = extract_epochs(
        rec, CONTRACT, labels={"left_hand", "right_hand"}, epoch_s=EPOCH
    )
    assert windows.labels == ("left_hand",)
    assert skipped == 1  # right_hand at 1800: its epoch runs past 2000 samples


def test_epochs_need_the_preprocessed_rate() -> None:
    with pytest.raises(NeurOSDataError, match="preprocess"):
        extract_epochs(_recording(sfreq=250.0), CONTRACT, epoch_s=EPOCH)


def test_flags_mark_transients_and_extreme_amplitudes() -> None:
    rec = _recording(events=[(0, "left_hand"), (500, "left_hand"), (1000, "right_hand")])
    data = rec.data.copy()
    data[1, 1100] = 900.0  # a 900 µV spike inside the third epoch
    spiky = Recording(
        data=data,
        sfreq=rec.sfreq,
        ch_names=rec.ch_names,
        events=rec.events,
        dataset=rec.dataset,
        subject=rec.subject,
        session=rec.session,
        run=rec.run,
    )
    windows, _ = extract_epochs(
        spiky, CONTRACT, ValidationThresholds(ptp_flag_uv=500.0), epoch_s=EPOCH
    )
    # An epoch starts >= 0.5 s after the recording starts, so it never overlaps the filter
    # transient (TRANSIENT_S = 0.5 s); the spike flags the third epoch only.
    assert windows.flags.tolist() == [
        WindowFlag.NONE,
        WindowFlag.NONE,
        WindowFlag.EXTREME_AMPLITUDE,
    ]
    assert windows.n_flagged() == 1
    assert windows.n_flagged(WindowFlag.EXTREME_AMPLITUDE) == 1


def test_continuous_windows_match_the_stream_and_flag_the_transient() -> None:
    rec = _recording(n_samples=600)
    windows = continuous_windows(rec, CONTRACT, label="state_rest")
    assert windows.starts.tolist() == window_starts(600, WINDOW, HOP).tolist()
    assert set(windows.labels) == {"state_rest"}
    # Windows starting in the first TRANSIENT_S (64 samples) after the filter start: 0..48.
    transient = (windows.flags & WindowFlag.FILTER_TRANSIENT) != 0
    assert windows.starts[transient].tolist() == [0, 16, 32, 48]


# ---------------------------------------------------------------------- WindowSet
def _set(n: int = 4, subject: int = 1, channels: tuple[str, ...] = ("C3", "Cz")) -> WindowSet:
    return WindowSet(
        data=np.zeros((n, len(channels), 8), dtype=np.float32),
        sfreq=128.0,
        ch_names=channels,
        labels=tuple("left_hand" if i % 2 else "right_hand" for i in range(n)),
        datasets=("Synthetic",) * n,
        subjects=np.full(n, subject),
        sessions=("0",) * n,
        runs=("0",) * n,
        starts=np.arange(n) * 8,
        flags=np.zeros(n, dtype=np.uint8),
    )


def test_window_set_is_read_only_and_selectable() -> None:
    windows = _set()
    with pytest.raises(ValueError, match="read-only"):
        windows.data[0, 0, 0] = 1.0
    picked = windows.select(np.array([True, False, True, False]))
    assert len(picked) == 2
    assert picked.starts.tolist() == [0, 16]
    assert windows.where_label({"left_hand"}).labels == ("left_hand", "left_hand")
    assert windows.select([3]).labels == ("left_hand",)


def test_concat_and_people() -> None:
    merged = WindowSet.concat([_set(subject=1), _set(subject=2)])
    assert len(merged) == 8
    assert merged.people == ("Synthetic:1", "Synthetic:2")
    assert len(merged.for_people({"Synthetic:2"})) == 4


def test_pick_channels_reorders_and_checks() -> None:
    windows = _set(channels=("C3", "Cz", "C4"))
    assert windows.pick_channels(["C4", "C3"]).ch_names == ("C4", "C3")
    with pytest.raises(NeurOSDataError):
        windows.pick_channels(["Pz"])
    with pytest.raises(NeurOSDataError):
        windows.pick_channels(["C3", "C3"])


def test_concat_rejects_mismatched_sets() -> None:
    with pytest.raises(NeurOSDataError):
        WindowSet.concat([_set(), _set(channels=("C3", "C4"))])
    with pytest.raises(NeurOSDataError):
        WindowSet.concat([])


@pytest.mark.parametrize(
    "overrides",
    [
        {"data": np.zeros((4, 2, 8), dtype=np.float64)},  # wrong dtype
        {"labels": ("left_hand",)},  # label count mismatch
        {"subjects": np.full(4, -1)},  # negative subject
        {"ch_names": ("C3", "C3")},  # duplicate channels
        {"data": np.full((4, 2, 8), np.nan, dtype=np.float32)},  # NaN
        {"flags": np.zeros(3, dtype=np.uint8)},  # wrong length
    ],
)
def test_window_set_validation(overrides: dict[str, object]) -> None:
    base = _set()
    fields: dict[str, object] = {
        "data": base.data.copy(),
        "sfreq": base.sfreq,
        "ch_names": base.ch_names,
        "labels": base.labels,
        "datasets": base.datasets,
        "subjects": base.subjects,
        "sessions": base.sessions,
        "runs": base.runs,
        "starts": base.starts,
        "flags": base.flags,
    }
    fields.update(overrides)
    with pytest.raises(NeurOSDataError):
        WindowSet(**fields)  # type: ignore[arg-type]


def test_with_data_keeps_metadata_and_checks_shape() -> None:
    windows = _set()
    replaced = windows.with_data(np.ones_like(windows.data))
    assert replaced.labels == windows.labels
    assert float(replaced.data.sum()) > 0
    with pytest.raises(NeurOSDataError):
        windows.with_data(np.ones((1, 2, 8), dtype=np.float32))
