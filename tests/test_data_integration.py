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
from neuros.data.recording import Recording
from neuros.data.registry import get_spec
from neuros.data.validate import validate_recording
from neuros.data.vocabulary import Vocabulary

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


@pytest.mark.parametrize("key", ["BNCI2014_001", "BNCI2014_004", "PhysionetMI"])
def test_manifest_intact(config: DataConfig, key: str) -> None:
    assert verify_manifest(get_spec(key), config.data_dir) == []
