"""Load-time validation (architecture §4.5): every check fires on the bug it targets."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from neuros.data.recording import EventTable, Recording
from neuros.data.registry import DatasetSpec
from neuros.data.synthetic import SyntheticConfig, make_recording
from neuros.data.validate import validate_recording
from neuros.data.vocabulary import Vocabulary
from neuros.errors import NeurOSDataError

from .conftest import make_spec


def _with_data(rec: Recording, data: np.ndarray, **overrides: object) -> Recording:
    fields: dict[str, object] = {
        "data": data.astype(np.float32),
        "sfreq": rec.sfreq,
        "ch_names": rec.ch_names,
        "events": rec.events,
        "dataset": rec.dataset,
        "subject": rec.subject,
        "session": rec.session,
        "run": rec.run,
    }
    fields.update(overrides)
    return Recording(**fields)  # type: ignore[arg-type]


@pytest.fixture
def clean() -> Recording:
    return make_recording(SyntheticConfig(n_trials=20, seed=7))


def _codes(report: object) -> set[str]:
    return {issue.code for issue in report.issues}  # type: ignore[attr-defined]


def test_clean_synthetic_data_passes(
    clean: Recording, synthetic_spec: DatasetSpec, vocab: Vocabulary
) -> None:
    report = validate_recording(clean, synthetic_spec, vocab)
    assert report.ok, report.issues
    assert report.issues == []
    assert 1.0 < report.amplitude_mad_uv < 50.0
    assert report.windows_total == clean.n_samples // round(2.0 * clean.sfreq)
    assert report.label_counts == {"left_hand": 10, "right_hand": 10}
    report.raise_if_errors()  # no-op


def test_volts_instead_of_microvolts_is_a_unit_error(
    clean: Recording, synthetic_spec: DatasetSpec, vocab: Vocabulary
) -> None:
    report = validate_recording(_with_data(clean, clean.data * 1e-6), synthetic_spec, vocab)
    assert "units" in _codes(report)
    assert "flat_channel" not in _codes(report)  # root cause reported once, not 12 dead channels
    assert not report.ok


def test_double_scaling_is_a_unit_error(
    clean: Recording, synthetic_spec: DatasetSpec, vocab: Vocabulary
) -> None:
    report = validate_recording(_with_data(clean, clean.data * 1e6), synthetic_spec, vocab)
    assert "units" in _codes(report)


def test_dc_offset_and_drift_are_tolerated(
    clean: Recording, synthetic_spec: DatasetSpec, vocab: Vocabulary
) -> None:
    # DC-coupled amplifiers (Biosemi) give mV offsets and slow drift; the EEG itself is fine.
    t = np.arange(clean.n_samples) / clean.sfreq
    drift = 5000.0 + 200.0 * np.sin(2 * np.pi * 0.01 * t)
    report = validate_recording(_with_data(clean, clean.data + drift), synthetic_spec, vocab)
    assert report.ok, report.issues
    assert report.windows_flagged == 0


def test_non_finite_samples(
    clean: Recording, synthetic_spec: DatasetSpec, vocab: Vocabulary
) -> None:
    data = clean.data.copy()
    data[2, 100:110] = np.nan
    data[5, 7] = np.inf
    report = validate_recording(_with_data(clean, data), synthetic_spec, vocab)
    assert report.non_finite == {clean.ch_names[2]: 10, clean.ch_names[5]: 1}
    assert "non_finite" in _codes(report)
    assert "units" not in _codes(report)  # other checks still run sensibly


def test_flat_channel(clean: Recording, synthetic_spec: DatasetSpec, vocab: Vocabulary) -> None:
    data = clean.data.copy()
    data[3] = 12.5  # constant: an unplugged electrode
    report = validate_recording(_with_data(clean, data), synthetic_spec, vocab)
    assert report.flat_channels == [clean.ch_names[3]]
    assert "flat_channel" in _codes(report)
    assert "units" not in _codes(report)  # the dead channel doesn't fake a unit error


def test_extreme_amplitude_windows_are_flagged_not_failed(
    clean: Recording, synthetic_spec: DatasetSpec, vocab: Vocabulary
) -> None:
    data = clean.data.copy()
    data[0, 1000:1010] += 2000.0  # electrode pop
    report = validate_recording(_with_data(clean, data), synthetic_spec, vocab)
    assert report.windows_flagged == 1
    assert "extreme_amplitude" in _codes(report)
    assert report.ok  # flags are counted, not fatal


def test_sampling_rate_mismatch(clean: Recording, vocab: Vocabulary) -> None:
    report = validate_recording(clean, make_spec(native_sfreq=500.0), vocab)
    assert "sampling_rate" in _codes(report)


def test_undeclared_channel(clean: Recording, vocab: Vocabulary) -> None:
    spec = make_spec(eeg_channels=tuple(ch for ch in clean.ch_names if ch != "Pz"))
    report = validate_recording(clean, spec, vocab)
    assert "unknown_channel" in _codes(report)


def test_unknown_label(clean: Recording, synthetic_spec: DatasetSpec, vocab: Vocabulary) -> None:
    events = EventTable(onsets=clean.events.onsets, labels=("jump",) * len(clean.events))
    report = validate_recording(_with_data(clean, clean.data, events=events), synthetic_spec, vocab)
    assert "unknown_label" in _codes(report)


def test_motor_imagery_without_events(
    clean: Recording, synthetic_spec: DatasetSpec, vocab: Vocabulary
) -> None:
    report = validate_recording(
        _with_data(clean, clean.data, events=EventTable.empty()), synthetic_spec, vocab
    )
    assert "no_events" in _codes(report)


def test_epoch_running_past_the_end_is_flagged(
    clean: Recording, synthetic_spec: DatasetSpec, vocab: Vocabulary
) -> None:
    last = clean.n_samples - 10
    events = EventTable.from_pairs(
        [*zip(clean.events.onsets.tolist(), clean.events.labels, strict=True), (last, "rest")]
    )
    report = validate_recording(_with_data(clean, clean.data, events=events), synthetic_spec, vocab)
    assert report.events_epoch_outside == 1
    assert "epoch_outside" in _codes(report)
    assert report.ok


def test_too_short(clean: Recording, synthetic_spec: DatasetSpec, vocab: Vocabulary) -> None:
    short = _with_data(clean, clean.data[:, :100], events=EventTable.empty())
    assert "too_short" in _codes(validate_recording(short, synthetic_spec, vocab))


def test_wrong_spec_raises(clean: Recording, vocab: Vocabulary) -> None:
    with pytest.raises(NeurOSDataError, match="validated against spec"):
        validate_recording(clean, replace(make_spec(), key="Other"), vocab)


def test_raise_if_errors_lists_every_error(
    clean: Recording, synthetic_spec: DatasetSpec, vocab: Vocabulary
) -> None:
    data = clean.data.copy() * 1e-6
    data[1, 0] = np.nan
    report = validate_recording(_with_data(clean, data), synthetic_spec, vocab)
    with pytest.raises(NeurOSDataError, match=r"non_finite.*units|units.*non_finite"):
        report.raise_if_errors()


def test_report_is_json_serialisable(
    clean: Recording, synthetic_spec: DatasetSpec, vocab: Vocabulary
) -> None:
    payload = json.dumps(validate_recording(clean, synthetic_spec, vocab).to_dict())
    assert '"ok": true' in payload
