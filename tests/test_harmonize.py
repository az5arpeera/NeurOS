"""Harmonisation of MOABB-style MNE raws into canonical recordings."""

from __future__ import annotations

from collections.abc import Sequence

import mne
import numpy as np
import pytest

from neuros.data.harmonize import classify_channels, harmonize_raw
from neuros.data.registry import DatasetSpec, get_spec
from neuros.data.vocabulary import Vocabulary
from neuros.errors import DataContext, NeurOSChannelError, NeurOSDataError, NeurOSLabelError

from .conftest import make_spec

SFREQ = 250.0
EVENT_ID = {"left_hand": 1, "right_hand": 2}
NAMES = ("Fc3.", "C3..", "Cz..", "C4..", "EOG1", "stim")


@pytest.fixture
def spec() -> DatasetSpec:
    return make_spec(
        key="FakeMI",
        eeg_channels=("FC3", "C3", "Cz", "C4"),
        non_eeg_channels=("EOG1", "stim"),
        cue_offset_s=2.0,
    )


@pytest.fixture
def fake_vocab(vocab: Vocabulary) -> Vocabulary:
    maps = dict(vocab.event_maps)
    maps["FakeMI"] = {"left_hand": "left_hand", "right_hand": "right_hand"}
    return vocab.model_copy(update={"event_maps": maps})


def _raw(
    names: Sequence[str] = NAMES,
    events: Sequence[tuple[int, int]] = ((250, 1), (1500, 2)),
    n_samples: int = 5000,
    first_samp: int = 0,
) -> mne.io.RawArray:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((len(names), n_samples)) * 20e-6  # volts, like MNE/MOABB
    types = ["stim" if n == "stim" else "eog" if n.startswith("EOG") else "eeg" for n in names]
    if "stim" in names:
        stim = names.index("stim")
        data[stim] = 0.0
        for sample, code in events:
            data[stim, sample] = code
    info = mne.create_info(list(names), SFREQ, types)
    return mne.io.RawArray(data, info, first_samp=first_samp, verbose="error")


def _harmonize(raw: mne.io.BaseRaw, spec: DatasetSpec, vocab: Vocabulary):  # type: ignore[no-untyped-def]
    return harmonize_raw(raw, spec, vocab, subject=1, session="0", run="0", event_id=EVENT_ID)


def test_harmonize_raw(spec: DatasetSpec, fake_vocab: Vocabulary) -> None:
    raw = _raw()
    rec = _harmonize(raw, spec, fake_vocab)
    assert rec.ch_names == ("FC3", "C3", "Cz", "C4")
    assert rec.data.dtype == np.float32
    np.testing.assert_allclose(rec.data, raw.get_data(picks=[0, 1, 2, 3]) * 1e6, rtol=1e-5)
    # Markers at 250 and 1500; the cue is 2 s (500 samples) later.
    assert rec.events.onsets.tolist() == [750, 2000]
    assert rec.events.labels == ("left_hand", "right_hand")
    assert rec.person_id == "FakeMI:1"


def test_any_stim_typed_channel_is_the_event_channel(
    spec: DatasetSpec, fake_vocab: Vocabulary
) -> None:
    # MOABB 1.7 names the stim channel "STI"; it needn't be declared in the registry.
    names = (*NAMES[:-1], "STI")
    raw = _raw(names=("Fc3.", "C3..", "Cz..", "C4..", "EOG1", "stim"))
    raw.rename_channels({"stim": "STI"})
    assert tuple(raw.ch_names) == names
    rec = _harmonize(raw, spec, fake_vocab)
    assert rec.ch_names == ("FC3", "C3", "Cz", "C4")
    assert rec.events.onsets.tolist() == [750, 2000]


def test_first_samp_is_removed(spec: DatasetSpec, fake_vocab: Vocabulary) -> None:
    shifted = _harmonize(_raw(first_samp=12345), spec, fake_vocab)
    assert shifted.events.onsets.tolist() == [750, 2000]


def test_physionet_padded_names_map_to_canonical() -> None:
    spec = get_spec("PhysionetMI")
    edf_names = [(n[0] + n[1:].lower()).ljust(4, ".") for n in spec.eeg_channels]
    assert {"Fc5.", "Iz..", "T10."} <= set(edf_names)
    _, names = classify_channels([*edf_names, "stim"], spec, DataContext("PhysionetMI"))
    assert tuple(names) == spec.eeg_channels


def test_undeclared_channel(spec: DatasetSpec, fake_vocab: Vocabulary) -> None:
    with pytest.raises(NeurOSChannelError, match="Undeclared"):
        _harmonize(_raw(names=(*NAMES[:-1], "Foo", "stim")), spec, fake_vocab)


def test_missing_declared_channel(spec: DatasetSpec, fake_vocab: Vocabulary) -> None:
    with pytest.raises(NeurOSChannelError, match="missing"):
        _harmonize(_raw(names=("Fc3.", "C3..", "Cz..", "EOG1", "stim")), spec, fake_vocab)


def test_duplicate_after_normalisation(spec: DatasetSpec, fake_vocab: Vocabulary) -> None:
    with pytest.raises(NeurOSChannelError, match="duplicates"):
        _harmonize(_raw(names=("Fc3.", "C3..", "c3", "Cz..", "C4..", "stim")), spec, fake_vocab)


def test_unmapped_event_code(spec: DatasetSpec, fake_vocab: Vocabulary) -> None:
    with pytest.raises(NeurOSLabelError, match="Event codes"):
        _harmonize(_raw(events=((250, 1), (900, 3))), spec, fake_vocab)


def test_event_beyond_recording_after_cue_offset(spec: DatasetSpec, fake_vocab: Vocabulary) -> None:
    with pytest.raises(NeurOSDataError, match="outside"):
        _harmonize(_raw(events=((4900, 1),)), spec, fake_vocab)


def test_annotation_events(spec: DatasetSpec, fake_vocab: Vocabulary) -> None:
    raw = _raw(names=NAMES[:-1])
    raw.set_annotations(
        mne.Annotations(
            onset=[1.0, 6.0, 7.0],
            duration=[0.0, 0.0, 1.0],
            description=["left_hand", "right_hand", "BAD_seg"],
        )
    )
    rec = _harmonize(raw, spec, fake_vocab)
    assert rec.events.onsets.tolist() == [750, 2000]
    assert rec.events.labels == ("left_hand", "right_hand")


def test_unknown_annotation(spec: DatasetSpec, fake_vocab: Vocabulary) -> None:
    raw = _raw(names=NAMES[:-1])
    raw.set_annotations(mne.Annotations(onset=[1.0], duration=[0.0], description=["jump"]))
    with pytest.raises(NeurOSLabelError, match="jump"):
        _harmonize(raw, spec, fake_vocab)


def test_dataset_without_event_map(spec: DatasetSpec, vocab: Vocabulary) -> None:
    with pytest.raises(NeurOSLabelError, match="No event map"):
        _harmonize(_raw(), spec, vocab)


# ---------------------------------------------------------------------- D-033 additions
def test_excluded_channels_must_exist_but_are_dropped(fake_vocab: Vocabulary) -> None:
    # GuttmannFlury2025 style: the publisher flags FC3 as faulty. It must still be in the file
    # (the registry describes the file), but it never reaches a recording.
    faulty = make_spec(
        key="FakeMI",
        eeg_channels=("FC3", "C3", "Cz", "C4"),
        non_eeg_channels=("EOG1", "stim"),
        cue_offset_s=2.0,
        excluded_channels={"FC3": "bridged (publisher)"},
    )
    rec = _harmonize(_raw(), faulty, fake_vocab)
    assert rec.ch_names == ("C3", "Cz", "C4")
    with pytest.raises(NeurOSChannelError, match="missing from the file"):
        _harmonize(_raw(names=("C3..", "Cz..", "C4..", "EOG1", "stim")), faulty, fake_vocab)


def test_negative_cue_offset_moves_events_back_to_the_cue(fake_vocab: Vocabulary) -> None:
    # Shin2017A style: the marker sits at the task start, 2 s after the arrow appeared.
    early = make_spec(
        key="FakeMI",
        eeg_channels=("FC3", "C3", "Cz", "C4"),
        non_eeg_channels=("EOG1", "stim"),
        cue_offset_s=-2.0,
        task_from_s=2.0,
        task_s=12.0,
        class_on_screen_s=2.0,
    )
    rec = _harmonize(_raw(events=((1000, 1), (3000, 2))), early, fake_vocab)
    assert rec.events.onsets.tolist() == [500, 2500]  # 2 s = 500 samples earlier
    with pytest.raises(NeurOSDataError, match="outside the recording"):
        _harmonize(_raw(events=((250, 1),)), early, fake_vocab)  # the cue would precede t = 0
