"""Dataset registry declarations and their cross-checks (Step 3 data lock)."""

from __future__ import annotations

import pytest

from neuros.config.schemas import SignalContract
from neuros.data.registry import (
    DATASETS,
    assert_may_train,
    check_against_moabb,
    check_registry,
    get_spec,
    keys_with_role,
    locked_test_dataset_keys,
    moabb_dataset,
    task_epoch,
    trainable_keys,
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
    # D-033 (4 Oct): the cue-free and gaze-diagnostic datasets.
    assert keys_with_role("cue_free") == ("Yang2025", "Shin2017A")
    assert keys_with_role("diagnostic") == ("GuttmannFlury2025",)
    # ~364 core people (D-011) + 18 literature + 36 EEGMAT + 51 + 29 + 31 (D-033).
    assert sum(DATASETS[k].n_subjects for k in keys_with_role("core")) == 364
    assert [DATASETS[k].n_subjects for k in ("Yang2025", "Shin2017A", "GuttmannFlury2025")] == [
        51,
        29,
        31,
    ]


def test_locked_test_datasets_exclude_bnci() -> None:
    keys = locked_test_dataset_keys()
    assert "BNCI2014_001" not in keys
    assert "BNCI2014_004" not in keys
    assert set(keys) == {
        *keys_with_role("core"),
        "EEGMAT",
        "Yang2025",
        "Shin2017A",
        "GuttmannFlury2025",
    }


def test_training_restrictions_d033() -> None:
    # Shin2017A is GPL-3.0 (testing only until the founders decide); GuttmannFlury2025 is a
    # diagnostic. Everything else may train a model we ship.
    assert [k for k, s in DATASETS.items() if not s.may_train] == ["Shin2017A", "GuttmannFlury2025"]
    assert get_spec("Shin2017A").use == "test"
    assert get_spec("GuttmannFlury2025").use == "diagnostic"
    assert get_spec("Yang2025").use == "train"
    pool = ("PhysionetMI", "Shin2017A", "Yang2025", "GuttmannFlury2025")
    assert trainable_keys(pool) == ("PhysionetMI", "Yang2025")
    assert_may_train(("PhysionetMI", "Yang2025"), "pooled training")  # no error
    with pytest.raises(NeurOSConfigError, match=r"Shin2017A: GPL-3\.0") as info:
        assert_may_train(pool, "pooled training")
    assert "GuttmannFlury2025: diagnostic only" in str(info.value)
    with pytest.raises(NeurOSConfigError, match="declare train_restriction"):
        make_spec(role="diagnostic")
    with pytest.raises(NeurOSConfigError, match="not blank"):
        make_spec(train_restriction="  ")


def test_d033_epochs_time_imagery_from_the_cue() -> None:
    contract = SignalContract()
    # Yang2025: 1.5-s video, imagery 1.5-5.5 s. Shin2017A: 2-s arrow, imagery 2-12 s, marker at
    # the task start (cue_offset_s = -2). Both cue-free. GuttmannFlury2025: rectangle all along.
    expected = {
        "Yang2025": (1.75, 3.75, True),
        "Shin2017A": (2.25, 4.25, True),
        "GuttmannFlury2025": (2.0, 4.0, False),
    }
    for key, values in expected.items():
        epoch = task_epoch(get_spec(key), contract)
        assert (epoch.start_s, epoch.end_s, epoch.cue_free) == values, key
    assert get_spec("Shin2017A").cue_offset_s == -2.0
    # The cue-only control may cover the cue before imagery starts.
    cue_only = task_epoch(get_spec("Yang2025"), SignalContract(window_s=0.5, epoch_s=(0.0, 0.5)))
    assert (cue_only.cue_free, cue_only.rule) == (False, "fixed")
    # The after-cue rule never starts before imagery, even when the class leaves earlier.
    early = make_spec(task_from_s=2.0, task_s=6.0, class_on_screen_s=0.5)
    assert task_epoch(early, contract).start_s == 2.0
    with pytest.raises(NeurOSConfigError, match=r"imagery \(1 s\) is shorter than one window"):
        task_epoch(make_spec(task_from_s=3.0, task_s=4.0, class_on_screen_s=3.0), contract)


def test_excluded_channels_d033() -> None:
    spec = get_spec("GuttmannFlury2025")
    assert len(spec.eeg_channels) == 60
    assert len(spec.usable_eeg_channels) == 49
    assert {"F3", "FC3", "P4"} <= set(spec.excluded_channels)
    assert all("recommend" in reason for reason in spec.excluded_channels.values())
    with pytest.raises(NeurOSConfigError, match="not declared EEG channels"):
        make_spec(excluded_channels={"O1": "faulty"})
    with pytest.raises(NeurOSConfigError, match="needs a reason"):
        make_spec(excluded_channels={"C3": " "})
    with pytest.raises(NeurOSConfigError, match="every EEG channel is excluded"):
        make_spec(eeg_channels=("C3",), excluded_channels={"C3": "faulty"})


def test_moabb_arguments_d033() -> None:
    yang = moabb_dataset(get_spec("Yang2025"))
    assert yang.paradigm_type == "2C"
    assert list(yang.subject_list) == list(range(1, 52))
    with pytest.raises(NeurOSConfigError, match="contradict the registry"):
        moabb_dataset(get_spec("Yang2025"), paradigm_type="3C")
    # A licence is accepted per download by a person, never by the registry.
    with pytest.raises(NeurOSConfigError, match="'accept' can't be a registry argument"):
        make_spec(moabb_class="Shin2017A", moabb_kwargs={"accept": True})
    with pytest.raises(NeurOSConfigError, match="no moabb_class"):
        make_spec(moabb_kwargs={"paradigm_type": "2C"})
    assert get_spec("Shin2017A").licence_acceptance is not None


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
        ({"cue_offset_s": float("nan")}, NeurOSConfigError, "cue_offset_s"),
        ({"task_from_s": 4.0}, NeurOSConfigError, "task_from_s"),
        ({"task_from_s": -0.5}, NeurOSConfigError, "task_from_s"),
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


def test_unit_scale_d035() -> None:
    # Only Cho2017 is rescaled, by exactly BioSemi's raw step, and it's flagged as inferred.
    cho = get_spec("Cho2017")
    assert cho.unit_scale == pytest.approx(1 / 32)
    assert "unit_scale" in cho.unverified
    assert all(spec.unit_scale == 1.0 for key, spec in DATASETS.items() if key != "Cho2017")
    for bad in (0.0, -1.0, 1e-4, 1e4, float("inf"), float("nan")):
        with pytest.raises(NeurOSConfigError, match="unit_scale"):
            make_spec(unit_scale=bad)
