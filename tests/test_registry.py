"""Dataset registry declarations and their cross-checks (Step 3 data lock)."""

from __future__ import annotations

import pytest

from neuros.config.schemas import SignalContract
from neuros.data.registry import (
    DATASETS,
    check_against_moabb,
    check_registry,
    get_spec,
    keys_with_role,
    locked_test_dataset_keys,
)
from neuros.data.vocabulary import Vocabulary
from neuros.errors import NeurOSChannelError, NeurOSConfigError

from .conftest import make_spec


def test_registry_shape() -> None:
    assert keys_with_role("core") == (
        "PhysionetMI",
        "Dreyer2023",
        "Stieger2021",
        "Lee2019_MI",
        "Cho2017",
    )
    assert keys_with_role("literature") == ("BNCI2014_001", "BNCI2014_004")
    assert keys_with_role("brain_state") == ("EEGMAT",)
    # ~364 core people (D-011) + 18 literature + 36 EEGMAT.
    assert sum(DATASETS[k].n_subjects for k in keys_with_role("core")) == 364


def test_locked_test_datasets_exclude_bnci() -> None:
    keys = locked_test_dataset_keys()
    assert "BNCI2014_001" not in keys
    assert "BNCI2014_004" not in keys
    assert set(keys) == {*keys_with_role("core"), "EEGMAT"}


def test_check_registry_passes(vocab: Vocabulary) -> None:
    check_registry(vocab, SignalContract())


def test_check_registry_catches_short_tasks(vocab: Vocabulary) -> None:
    # A contract whose epoch ends at 5 s is longer than PhysionetMI's 3 s task.
    contract = SignalContract(window_s=2.0, epoch_s=(3.0, 5.0))
    with pytest.raises(NeurOSConfigError, match="shorter than the epoch end"):
        check_registry(vocab, contract)


@pytest.mark.parametrize("key", [k for k, s in DATASETS.items() if s.moabb_class is not None])
def test_registry_agrees_with_moabb_metadata(key: str, vocab: Vocabulary) -> None:
    assert check_against_moabb(DATASETS[key], vocab) == []


def test_get_spec_unknown() -> None:
    with pytest.raises(NeurOSConfigError, match="known"):
        get_spec("NoSuchDataset")


def test_subgroups() -> None:
    dreyer = get_spec("Dreyer2023")
    assert dreyer.subgroup_of(1) == "A"
    assert dreyer.subgroup_of(61) == "B"
    assert dreyer.subgroup_of(87) == "C"
    assert get_spec("Cho2017").subgroup_of(5) == "all"
    with pytest.raises(NeurOSConfigError):
        dreyer.subgroup_of(88)


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"eeg_channels": ("c3",)}, NeurOSChannelError, "Non-canonical"),
        ({"eeg_channels": ("C3", "C3")}, NeurOSChannelError, "Duplicate"),
        ({"eeg_channels": ()}, NeurOSConfigError, "no EEG"),
        ({"non_eeg_channels": ("C3",)}, NeurOSConfigError, "both EEG and non-EEG"),
        ({"subjects": (2, 1)}, NeurOSConfigError, "sorted"),
        ({"subjects": ()}, NeurOSConfigError, "no subjects"),
        ({"native_sfreq": 0.0}, NeurOSConfigError, "positive"),
        ({"cue_offset_s": -1.0}, NeurOSConfigError, "cue_offset_s"),
        ({"pre_cue_fixation_s": None}, NeurOSConfigError, "pre_cue"),
        ({"subgroups": {"A": (1, 2)}}, NeurOSConfigError, "cover the subjects"),
        ({"subgroups": {"A": (1, 2), "B": (2, 3)}}, NeurOSConfigError, "subgroups"),
    ],
)
def test_spec_validation(
    overrides: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        make_spec(**overrides)
