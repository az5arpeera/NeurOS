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
    task_epoch,
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
    with pytest.raises(NeurOSConfigError, match="runs past the task"):
        check_registry(vocab, contract)


# ---------------------------------------------------------------------- task epochs (D-030)
def test_task_epochs_start_after_the_class_display() -> None:
    contract = SignalContract()
    # BNCI2014-001's arrow is up for 1.25 s: the window starts 0.25 s after it has gone.
    epoch = task_epoch(get_spec("BNCI2014_001"), contract)
    assert (epoch.start_s, epoch.end_s, epoch.rule, epoch.cue_free) == (
        1.5,
        3.5,
        "after_cue",
        True,
    )
    # Every other cued dataset keeps the class (cue or feedback) on screen for the whole task:
    # it gets the latest window in the task, flagged as not cue-free.
    latest = {
        "PhysionetMI": (1.0, 3.0),
        "Dreyer2023": (3.0, 5.0),
        "Stieger2021": (1.0, 3.0),
        "Lee2019_MI": (2.0, 4.0),
        "Cho2017": (1.0, 3.0),
        "BNCI2014_004": (2.5, 4.5),
    }
    for key, (start, end) in latest.items():
        epoch = task_epoch(get_spec(key), contract)
        assert (epoch.start_s, epoch.end_s, epoch.cue_free) == (start, end, False), key
    # A wider margin moves the cue-free window later.
    epoch = task_epoch(get_spec("BNCI2014_001"), SignalContract(cue_margin_s=0.5))
    assert (epoch.start_s, epoch.end_s, epoch.cue_free) == (1.75, 3.75, True)


def test_fixed_epochs_report_whether_they_clear_the_cue() -> None:
    spec = get_spec("BNCI2014_001")

    def fixed(start: float, end: float) -> tuple[bool, str]:
        epoch = task_epoch(spec, SignalContract(window_s=end - start, epoch_s=(start, end)))
        return epoch.cue_free, epoch.rule

    assert fixed(0.5, 2.5) == (False, "fixed")  # the pre-D-030 window: arrow up until 1.25 s
    assert fixed(1.5, 3.5) == (True, "fixed")
    assert fixed(0.0, 0.5) == (False, "fixed")  # the cue-only control
    assert fixed(-0.5, 0.0) == (True, "fixed")  # before the cue appears
    with pytest.raises(NeurOSConfigError, match="runs past the task"):
        fixed(2.5, 4.5)


def test_task_epoch_boundaries() -> None:
    with pytest.raises(NeurOSConfigError, match="no class cue"):
        task_epoch(get_spec("EEGMAT"), SignalContract())
    with pytest.raises(NeurOSConfigError, match="shorter than one window"):
        task_epoch(make_spec(task_s=1.5, class_on_screen_s=1.0), SignalContract())
    epoch = task_epoch(get_spec("BNCI2014_001"), SignalContract())
    assert epoch.to_dict() == {"epoch_s": [1.5, 3.5], "epoch_rule": "after_cue", "cue_free": True}
    assert epoch.describe().endswith("(cue-free)")
    assert "NOT cue-free" in task_epoch(get_spec("PhysionetMI"), SignalContract()).describe()


def test_class_on_screen_is_declared_and_bounded() -> None:
    with pytest.raises(NeurOSConfigError, match="must declare class_on_screen_s"):
        make_spec(class_on_screen_s=None)
    for bad in (-0.1, 4.5, float("nan")):  # task_s is 4.0
        with pytest.raises(NeurOSConfigError, match="class_on_screen_s must be within"):
            make_spec(class_on_screen_s=bad)
    make_spec(paradigm="mental_arithmetic", class_on_screen_s=None)  # no class cue: fine


@pytest.mark.parametrize("key", [k for k, s in DATASETS.items() if s.moabb_class is not None])
def test_registry_agrees_with_moabb_metadata(key: str, vocab: Vocabulary) -> None:
    assert check_against_moabb(DATASETS[key], vocab) == []


def test_physionet_exclusions_d027() -> None:
    spec = get_spec("PhysionetMI")
    assert sorted(spec.excluded_subjects) == [88, 92, 100]
    assert len(spec.usable_subjects) == 106
    assert all("128 Hz" in reason for reason in spec.excluded_subjects.values())
    usable_core = sum(len(DATASETS[k].usable_subjects) for k in keys_with_role("core"))
    assert usable_core == 361


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
        ({"excluded_subjects": {9: "bad"}}, NeurOSConfigError, "not declared"),
        ({"excluded_subjects": {1: "  "}}, NeurOSConfigError, "reason"),
    ],
)
def test_spec_validation(
    overrides: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        make_spec(**overrides)
