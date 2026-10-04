"""Real-data checks: harmonise + validate downloaded datasets end to end.

Run after `neuros download` with:  NEUROS_DATA_TESTS=1 uv run pytest -m data
The expectations below were measured on 2026-10-03 (MOABB 1.7.2) and are part of the data lock:
if an upstream change alters them, these tests fail before any model sees the data.
"""

from __future__ import annotations

from collections import Counter

import pytest

from neuros.config.schemas import DataConfig, load_data_config
from neuros.data.download import prepare_mne_environment, verify_manifest
from neuros.data.harmonize import harmonize_moabb_subject
from neuros.data.prepare import prepare_subject, subject_epochs
from neuros.data.recording import Recording
from neuros.data.registry import get_spec
from neuros.data.validate import validate_recording
from neuros.data.vocabulary import Vocabulary
from neuros.errors import NeurOSDataError
from neuros.signal.windows import WindowFlag

from .conftest import REPO_ROOT

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def config() -> DataConfig:
    loaded = load_data_config(REPO_ROOT / "configs" / "data.yaml")
    prepare_mne_environment(loaded.data_dir)
    return loaded


def _load_and_validate(
    key: str, subject: int, config: DataConfig, vocab: Vocabulary
) -> list[Recording]:
    spec = get_spec(key)
    recordings = harmonize_moabb_subject(spec, subject, vocab)
    for rec in recordings:
        report = validate_recording(rec, spec, vocab, config.signal, config.validation)
        assert report.ok, (rec.session, rec.run, report.issues)
        assert rec.ch_names == spec.eeg_channels
    return recordings


def _label_totals(recordings: list[Recording]) -> Counter[str]:
    totals: Counter[str] = Counter()
    for rec in recordings:
        totals.update(rec.events.counts())
    return totals


def test_bnci2014_001_subject_1(config: DataConfig, vocab: Vocabulary) -> None:
    recordings = _load_and_validate("BNCI2014_001", 1, config, vocab)
    assert len(recordings) == 12  # 2 sessions × 6 runs
    assert {rec.session for rec in recordings} == {"0train", "1test"}
    assert _label_totals(recordings) == {
        "left_hand": 144,
        "right_hand": 144,
        "feet": 144,
        "tongue": 144,
    }


def test_bnci2014_004_subject_1(config: DataConfig, vocab: Vocabulary) -> None:
    recordings = _load_and_validate("BNCI2014_004", 1, config, vocab)
    assert [rec.session for rec in recordings] == ["0train", "1train", "2train", "3test", "4test"]
    assert _label_totals(recordings) == {"left_hand": 360, "right_hand": 360}


def test_physionet_subject_1(config: DataConfig, vocab: Vocabulary) -> None:
    recordings = _load_and_validate("PhysionetMI", 1, config, vocab)
    assert len(recordings) == 6  # imagined runs 4, 8, 12 (hands) then 6, 10, 14 (fists/feet)
    assert _label_totals(recordings) == {
        "left_hand": 23,
        "right_hand": 22,
        "both_hands": 21,
        "feet": 24,
        "rest": 90,
    }


def test_physionet_excluded_subject_fails_validation(config: DataConfig, vocab: Vocabulary) -> None:
    """Why D-027 excludes subject 88: its files are 128 Hz, not the declared 160 Hz."""
    spec = get_spec("PhysionetMI")
    recording = harmonize_moabb_subject(spec, 88, vocab)[0]
    report = validate_recording(recording, spec, vocab, config.signal, config.validation)
    assert recording.sfreq == 128.0
    assert "sampling_rate" in {issue.code for issue in report.issues}


@pytest.mark.parametrize("key", ["BNCI2014_001", "BNCI2014_004", "PhysionetMI"])
def test_manifest_intact(config: DataConfig, key: str) -> None:
    assert verify_manifest(get_spec(key), config.data_dir) == []


def test_bnci2014_001_epochs_subject_1(config: DataConfig, vocab: Vocabulary) -> None:
    """Day 2: the full chain (harmonise → validate → 128 Hz → epochs) on real data."""
    windows, summary = subject_epochs(
        get_spec("BNCI2014_001"), 1, vocab, config, ["left_hand", "right_hand"]
    )
    assert windows.data.shape == (288, 22, 256)  # 2 × 144 trials, 22 channels, 2 s at 128 Hz
    assert windows.label_counts() == {"left_hand": 144, "right_hand": 144}
    assert summary.skipped_outside == 0
    assert windows.n_flagged(WindowFlag.FILTER_TRANSIENT) == 0
    assert set(windows.sessions) == {"0train", "1test"}
    # In-band amplitudes stay in a physiological range after the causal chain.
    assert 1.0 < float(windows.data.std()) < 50.0


def test_prepare_refuses_an_excluded_subject(config: DataConfig, vocab: Vocabulary) -> None:
    with pytest.raises(NeurOSDataError, match="excluded"):
        prepare_subject(get_spec("PhysionetMI"), 88, vocab, config)
