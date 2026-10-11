"""Experiment configs: the E0 files, validation of every field, signal overrides."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from neuros.config.experiment import (
    EEGNetConfig,
    LosoExperimentConfig,
    PooledExperimentConfig,
    TrainConfig,
    apply_signal_overrides,
    load_experiment_config,
    load_pooled_experiment_config,
)
from neuros.config.schemas import SignalContract
from neuros.data.registry import assert_may_train, get_spec
from neuros.errors import NeurOSConfigError

from .conftest import REPO_ROOT

E0 = REPO_ROOT / "configs" / "e0.yaml"
E0_4S = REPO_ROOT / "configs" / "e0_window4s.yaml"


def test_e0_files_load_with_the_papers_recipe() -> None:
    e0 = load_experiment_config(E0)
    assert e0.name == "e0"
    assert e0.dataset == "BNCI2014_001"
    assert e0.labels == ("left_hand", "right_hand")
    assert e0.alignments == ("none", "ea")
    assert e0.alignment_unit == "session"
    assert e0.validation_people == 0
    # Pinned to the pre-D-030 window, so the file keeps reproducing the 3 Oct result.
    assert e0.signal == {"epoch_s": [0.5, 2.5]}
    # Sagayaraju 2026: AdamW lr 1e-3, weight decay 0.05, dropout 0.5, 300 epochs, SWA.
    assert (e0.train.lr, e0.train.weight_decay, e0.train.max_epochs) == (1e-3, 0.05, 300)
    assert e0.model.dropout == 0.5
    assert e0.train.swa_fraction == 0.25
    assert e0.train.patience is None

    window4s = load_experiment_config(E0_4S)
    assert window4s.alignments == ("ea",)
    contract = apply_signal_overrides(SignalContract(), window4s.signal)
    assert contract.window_samples == 512
    assert contract.epoch_s == (0.0, 4.0)
    # Everything but the window is the same experiment.
    same = {"model", "train", "dataset", "labels", "channels", "reference", "alignment_unit"}
    assert {k: getattr(e0, k) for k in same} == {k: getattr(window4s, k) for k in same}


def test_defaults_are_the_architecture() -> None:
    model = EEGNetConfig()
    assert (model.f1, model.depth, model.f2, model.kernel_samples) == (16, 2, 32, 64)
    assert model.embedding_dim == 64
    assert (model.spatial_max_norm, model.head_max_norm) == (1.0, 0.25)
    train = TrainConfig()
    assert train.people_per_batch * train.windows_per_person == 64
    assert train.swa_fraction == 0.25


@pytest.mark.parametrize(
    "changes",
    [
        {"labels": ["left_hand"]},
        {"labels": ["left_hand", "left_hand"]},
        {"alignments": []},
        {"alignments": ["ea", "ea"]},
        {"alignments": ["ra"]},
        {"seeds": []},
        {"seeds": [-1]},
        {"seeds": [1, 1]},
        {"name": "E0"},
        {"name": "has space"},
        {"validation_people": -1},
        {"channels": "everything"},
        {"alignment_unit": "trial"},
        {"surprise": 1},
        {"train": {"lr": float("nan")}},
        {"train": {"min_delta": float("inf")}},
        {"train": {"lr": 1e-3, "min_lr": 1e-2}},
        {"train": {"max_epochs": 0}},
        {"train": {"swa_fraction": 1.5}},
        {"train": {"patience": 0}},
        {"train": {"unknown": 1}},
        {"model": {"dropout": 1.0}},
        {"model": {"f1": 0}},
        {"model": {"head_max_norm": 0}},
    ],
)
def test_invalid_values_are_rejected(changes: dict[str, object]) -> None:
    raw: dict[str, object] = {"name": "x", "dataset": "BNCI2014_001", **changes}
    with pytest.raises(ValidationError):
        LosoExperimentConfig.model_validate(raw)


def test_signal_overrides_are_validated_as_a_whole() -> None:
    base = SignalContract()
    assert apply_signal_overrides(base, {}) is base
    with pytest.raises(NeurOSConfigError, match="invalid signal overrides"):
        # A fixed epoch must stay one window long.
        apply_signal_overrides(SignalContract(epoch_s=(0.5, 2.5)), {"window_s": 4.0})
    # Under the contract's after-cue rule (D-030) the window length can change on its own.
    assert apply_signal_overrides(base, {"window_s": 4.0}).window_s == 4.0
    with pytest.raises(NeurOSConfigError):
        apply_signal_overrides(base, {"bandwidth": 3})
    with pytest.raises(NeurOSConfigError):  # a clean error, not an OverflowError
        apply_signal_overrides(base, {"window_s": float("inf"), "epoch_s": [0.5, float("inf")]})
    changed = apply_signal_overrides(base, {"band_hz": [8.0, 30.0]})
    assert changed.band_hz == (8.0, 30.0)
    assert changed.window_s == base.window_s


def test_loader_errors(tmp_path: Path) -> None:
    with pytest.raises(NeurOSConfigError, match="not found"):
        load_experiment_config(tmp_path / "missing.yaml")
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("name: [unclosed", encoding="utf-8")
    with pytest.raises(NeurOSConfigError, match="invalid YAML"):
        load_experiment_config(bad_yaml)
    listing = tmp_path / "list.yaml"
    listing.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(NeurOSConfigError, match="mapping"):
        load_experiment_config(listing)
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("name: x\ndataset: BNCI2014_001\nseeds: []\n", encoding="utf-8")
    with pytest.raises(NeurOSConfigError, match="invalid experiment config"):
        load_experiment_config(invalid)


_EXPERIMENT_FILES = sorted((REPO_ROOT / "configs").glob("e*.yaml"))


def _is_pooled(path: Path) -> bool:
    """Pooled (E1/E2) configs declare training datasets; LOSO (E0) configs a dataset."""
    import yaml

    return "train_datasets" in (yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def _is_state(path: Path) -> bool:
    """State-channel (E6) configs declare channel arms instead of a model."""
    import yaml

    return "arms" in (yaml.safe_load(path.read_text(encoding="utf-8")) or {})


@pytest.mark.parametrize(
    "path",
    [p for p in _EXPERIMENT_FILES if not _is_pooled(p) and not _is_state(p)],
    ids=lambda p: p.name,
)
def test_every_committed_experiment_is_valid(path: Path) -> None:
    experiment = load_experiment_config(path)
    contract = apply_signal_overrides(SignalContract(), experiment.signal)
    pooled = experiment.model.pool1 * experiment.model.pool2
    assert contract.window_samples >= pooled  # EEGNet's pools need this many samples
    assert experiment.name == path.stem  # results land in results/<name>/


@pytest.mark.parametrize(
    "path", [p for p in _EXPERIMENT_FILES if _is_state(p)], ids=lambda p: p.name
)
def test_committed_state_experiments_are_valid(path: Path) -> None:
    from neuros.eval.state import load_state_config

    experiment = load_state_config(path)
    assert experiment.name == path.stem  # results land in results/<name>/


# ---------------------------------------------------------------------- E1 (pooled) configs
@pytest.mark.parametrize(
    "path", [p for p in _EXPERIMENT_FILES if _is_pooled(p)], ids=lambda p: p.name
)
def test_committed_pooled_experiments_are_valid(path: Path) -> None:
    experiment = load_pooled_experiment_config(path)
    assert experiment.name == path.stem
    assert_may_train(experiment.train_datasets, "a committed E1 config")  # D-033
    for key in experiment.datasets:
        get_spec(key)  # every dataset is registered
    contract = apply_signal_overrides(SignalContract(), experiment.signal)
    pooled = experiment.model.pool1 * experiment.model.pool2
    start, end = experiment.cue_only_epoch_s
    assert round((end - start) * contract.sfreq) >= pooled  # the cue-only window fits EEGNet
    assert contract.window_samples >= pooled


def test_e1_follows_d031_d033_d034() -> None:
    e1 = load_pooled_experiment_config(REPO_ROOT / "configs" / "e1.yaml")
    assert e1.channels == "motor"  # D-031 / D-034
    assert e1.signal == {"band_hz": [8.0, 30.0]}  # D-034
    assert e1.reference == "car"
    # D-033: restricted datasets are tested, never trained on. Shin2017A is back since D-037
    # derived its C3/C4 (Q27).
    assert {"Shin2017A", "GuttmannFlury2025"} <= set(e1.test_only_datasets)
    assert "cue_only" in e1.conditions  # D-030


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"train_datasets": []}, "at least 1"),
        ({"train_datasets": ["A", "A"]}, "duplicates"),
        ({"test_only_datasets": ["PhysionetMI"]}, "both trained on and test-only"),
        ({"train_datasets": ["PhysionetMI"], "conditions": ["lodo"]}, "at least 2 train"),
        ({"cue_only_epoch_s": [0.5, 0.5]}, "cue_only_epoch_s"),
        ({"labels": ["left_hand"]}, "at least 2 labels"),
        ({"conditions": []}, "must not be empty"),
        ({"folds": 1}, "greater than or equal to 2"),
        ({"validation_fraction": 0.5}, "less than 0.5"),
        ({"name": "Bad Name"}, "pattern"),
    ],
)
def test_pooled_config_rejects(changes: dict[str, object], message: str) -> None:
    base = {"name": "e1", "train_datasets": ["PhysionetMI", "BNCI2014_001"]}
    with pytest.raises(ValidationError, match=message):
        PooledExperimentConfig.model_validate({**base, **changes})


def test_pooled_loader_errors(tmp_path: Path) -> None:
    with pytest.raises(NeurOSConfigError, match="not found"):
        load_pooled_experiment_config(tmp_path / "missing.yaml")
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("name: e1\ntrain_datasets: []\n", encoding="utf-8")
    with pytest.raises(NeurOSConfigError, match="invalid experiment config"):
        load_pooled_experiment_config(invalid)


def test_e2_configs_follow_e1_and_measure_the_cue_free_warmup() -> None:
    e1 = load_pooled_experiment_config(REPO_ROOT / "configs" / "e1.yaml")
    e2 = load_pooled_experiment_config(REPO_ROOT / "configs" / "e2.yaml")
    assert e2.warmup is not None
    # Same data, layout, recipe and seed as E1: the pooled models reproduce E1's.
    for field in (
        "train_datasets",
        "test_only_datasets",
        "channels",
        "reference",
        "signal",
        "model",
        "train",
        "seeds",
        "folds",
        "validation_fraction",
    ):
        assert getattr(e2, field) == getattr(e1, field), field
    assert set(e2.warmup.datasets) == {"Yang2025", "BNCI2014_001", "Shin2017A"}  # cue-free
    assert {"pooled", "cue_only"} <= set(e2.conditions)
    smoke = load_pooled_experiment_config(REPO_ROOT / "configs" / "e2_smoke.yaml")
    assert smoke.warmup is not None
    assert smoke.warmup.datasets == ("BNCI2014_001",)


@pytest.mark.parametrize("name", ["e2_crops", "e2_multiclass", "e2_crops_multiclass"])
def test_d039_configs_are_e2_with_more_training_windows(name: str) -> None:
    e2 = load_pooled_experiment_config(REPO_ROOT / "configs" / "e2.yaml")
    variant = load_pooled_experiment_config(REPO_ROOT / "configs" / f"{name}.yaml")
    assert variant.training_windows is not None
    # Same data, recipe, seed, folds and warm-up sets: the scores pair with E2's person by
    # person, and only what the models train on differs.
    for field in (
        "train_datasets",
        "test_only_datasets",
        "labels",
        "channels",
        "reference",
        "alignment",
        "alignment_unit",
        "signal",
        "model",
        "train",
        "seeds",
        "folds",
        "validation_fraction",
        "warmup",
    ):
        assert getattr(variant, field) == getattr(e2, field), field
    assert variant.conditions == ("pooled",)  # E2's cue-only control covers the same windows
    crops = variant.training_windows.crop_stride_s
    extra = variant.training_windows.extra_labels
    assert (crops is not None) == ("crops" in name)
    assert bool(extra) == ("multiclass" in name)
    if extra:
        assert set(extra) == {"feet", "tongue", "both_hands", "rest"}


def test_finetune_config_is_e2_plus_finetuning() -> None:
    # Q30: the same models as E2 (data, recipe, seed, folds, warm-up), so E2's numbers reproduce
    # and the fine-tuned scores pair with the calibration curve's.
    e2 = load_pooled_experiment_config(REPO_ROOT / "configs" / "e2.yaml")
    variant = load_pooled_experiment_config(REPO_ROOT / "configs" / "e2_finetune.yaml")
    for field in (
        "train_datasets",
        "test_only_datasets",
        "labels",
        "channels",
        "reference",
        "alignment",
        "alignment_unit",
        "signal",
        "model",
        "train",
        "seeds",
        "folds",
        "validation_fraction",
        "warmup",
    ):
        assert getattr(variant, field) == getattr(e2, field), field
    assert variant.conditions == ("pooled",)
    assert variant.training_windows is None
    assert variant.finetune is not None
    assert variant.finetune.sizes == (5, 10, 20)
    assert set(variant.finetune.datasets) == set(e2.warmup.datasets)  # type: ignore[union-attr]
