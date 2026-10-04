"""Leave-one-person-out deep runs (E0 machinery) on synthetic people."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neuros.align.covariance import window_covariances  # noqa: E402
from neuros.config.experiment import LosoExperimentConfig  # noqa: E402
from neuros.errors import NeurOSConfigError, NeurOSDataError  # noqa: E402
from neuros.eval.deep import (  # noqa: E402
    compare_runs,
    prepare_inputs,
    run_deep_loso,
    run_loso_experiment,
    seed_summaries,
)

from .synthetic_people import FAST_TRAIN, TINY_MODEL, people_windows  # noqa: E402


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _experiment(**changes: object) -> LosoExperimentConfig:
    raw = {
        "name": "test",
        "dataset": "Synthetic",
        "alignments": ["none", "ea"],
        "model": TINY_MODEL.model_dump(),
        "train": FAST_TRAIN.model_dump(),
        **changes,
    }
    return LosoExperimentConfig.model_validate(raw)


def test_prepare_inputs() -> None:
    windows = people_windows((0.5, 4.0), per_class=6)
    same, references = prepare_inputs(windows, reference="as_recorded", alignment="none")
    assert same is windows
    assert references == {}

    car, _ = prepare_inputs(windows, reference="car", alignment="none")
    np.testing.assert_allclose(car.data.sum(axis=1), 0.0, atol=1e-4)

    aligned, references = prepare_inputs(
        windows, reference="as_recorded", alignment="ea", unit="session"
    )
    assert set(references) == {f"Synthetic:{p}/{s}" for p in (1, 2) for s in ("0", "1")}
    for person in aligned.people:
        mine = aligned.for_people({person})
        for session in ("0", "1"):
            rows = np.array([s == session for s in mine.sessions])
            mean = window_covariances(mine.data[rows]).mean(axis=0)
            np.testing.assert_allclose(mean, np.eye(4), atol=0.02)

    _, per_person = prepare_inputs(windows, reference="as_recorded", alignment="ea", unit="person")
    assert set(per_person) == {"Synthetic:1", "Synthetic:2"}
    with pytest.raises(NeurOSConfigError):
        prepare_inputs(windows, reference="mastoid", alignment="none")  # type: ignore[arg-type]
    with pytest.raises(NeurOSConfigError):
        prepare_inputs(windows, reference="as_recorded", alignment="ra")  # type: ignore[arg-type]


def test_loso_run_scores_every_person_once() -> None:
    windows = people_windows((0.2, 1.0, 5.0))
    run = run_deep_loso(windows, _experiment(), alignment="ea", seed=0, progress=lambda m: None)
    people = [row["person"] for row in run["per_person"]]
    assert people == ["Synthetic:1", "Synthetic:2", "Synthetic:3"]
    assert run["summary_balanced_accuracy"]["n_people"] == 3
    # Aligned synthetic people are easy: every held-out person is decoded.
    assert run["summary_balanced_accuracy"]["min"] >= 0.85
    for fold, person in zip(run["folds"], people, strict=True):
        assert fold["fold"]["test"] == [person]
        assert person not in fold["fold"]["train"]
        assert fold["n_train_windows"] == 96
        assert fold["train"]["swa_epochs"] == [10, 12]
    json.dumps(run)  # the whole run goes into the result file


def test_input_scale_is_fit_on_training_people_only() -> None:
    windows = people_windows((1.0, 1.0, 1000.0), per_class=4)
    experiment = _experiment(train={**FAST_TRAIN.model_dump(), "max_epochs": 1})
    run = run_deep_loso(windows, experiment, alignment="none", seed=0, progress=lambda m: None)
    scales = {fold["fold"]["test"][0]: fold["input_scale"] for fold in run["folds"]}
    # The fold that tests the loud person never saw its amplitude...
    quiet = windows.for_people({"Synthetic:1", "Synthetic:2"})
    expected = float(np.sqrt(np.mean(quiet.data.astype(np.float64) ** 2)))
    assert scales["Synthetic:3"] == pytest.approx(expected, rel=1e-9)
    # ...while folds that train on it do.
    assert scales["Synthetic:1"] > 100 * scales["Synthetic:3"]


def test_validation_people_come_from_training_people() -> None:
    windows = people_windows((0.5, 1.0, 2.0, 4.0), per_class=4)
    experiment = _experiment(
        validation_people=1,
        train={**FAST_TRAIN.model_dump(), "max_epochs": 3, "patience": 2},
    )
    run = run_deep_loso(windows, experiment, alignment="ea", seed=0, progress=lambda m: None)
    for fold in run["folds"]:
        split = fold["fold"]
        assert len(split["validation"]) == 1
        assert not set(split["validation"]) & (set(split["train"]) | set(split["test"]))
        assert fold["n_validation_windows"] == 16
        assert fold["train"]["final_validation"] is not None


def test_labels_outside_the_experiment_are_refused() -> None:
    windows = people_windows((1.0, 2.0), per_class=2)
    experiment = _experiment(labels=["left_hand", "feet"])
    with pytest.raises(NeurOSDataError, match="right_hand"):
        run_deep_loso(windows, experiment, alignment="ea", seed=0, progress=lambda m: None)
    one_person = windows.for_people({"Synthetic:1"})
    with pytest.raises(NeurOSDataError, match="2 people"):
        run_deep_loso(one_person, _experiment(), alignment="ea", seed=0, progress=lambda m: None)
    with pytest.raises(NeurOSConfigError):
        run_deep_loso(windows, _experiment(), alignment="ea", seed=0, workers=0)


def test_experiment_compares_alignments_and_seeds() -> None:
    windows = people_windows((0.2, 1.0, 5.0), per_class=8)
    experiment = _experiment(seeds=[0, 1], train={**FAST_TRAIN.model_dump(), "max_epochs": 6})
    result = run_loso_experiment(windows, experiment, progress=lambda m: None)
    assert [(r["alignment"], r["seed"]) for r in result["runs"]] == [
        ("none", 0),
        ("none", 1),
        ("ea", 0),
        ("ea", 1),
    ]
    assert [(c["a"], c["b"], c["seed"]) for c in result["comparisons"]] == [
        ("eegnet+ea", "eegnet+none", 0),
        ("eegnet+ea", "eegnet+none", 1),
    ]
    spreads = {s["name"]: s for s in result["seed_summaries"]}
    assert set(spreads) == {"eegnet+none", "eegnet+ea"}
    ea_runs = [r for r in result["runs"] if r["alignment"] == "ea"]
    per_seed = [r["summary_balanced_accuracy"]["mean"] for r in ea_runs]
    assert spreads["eegnet+ea"]["per_seed_mean"] == per_seed
    assert spreads["eegnet+ea"]["summary_balanced_accuracy"]["mean"] == pytest.approx(
        float(np.mean(per_seed))
    )


def test_compare_runs_needs_the_same_people() -> None:
    a = {
        "name": "eegnet+ea",
        "alignment": "ea",
        "seed": 0,
        "balanced_accuracy_by_person": {"x": 0.8},
    }
    b = {
        "name": "eegnet+none",
        "alignment": "none",
        "seed": 0,
        "balanced_accuracy_by_person": {"y": 0.6},
    }
    with pytest.raises(NeurOSDataError):
        compare_runs([b, a])
    assert compare_runs([a]) == []  # nothing to pair with
    assert seed_summaries([a]) == []  # a single seed has no spread


@pytest.mark.slow
def test_parallel_workers_reproduce_the_sequential_run() -> None:
    windows = people_windows((0.5, 2.0), per_class=6)
    experiment = _experiment(train={**FAST_TRAIN.model_dump(), "max_epochs": 3})
    sequential = run_deep_loso(
        windows, experiment, alignment="ea", seed=4, workers=1, threads=1, progress=lambda m: None
    )
    parallel = run_deep_loso(
        windows, experiment, alignment="ea", seed=4, workers=2, threads=1, progress=lambda m: None
    )
    assert parallel["workers"] == 2
    assert parallel["per_person"] == sequential["per_person"]
    for a, b in zip(parallel["folds"], sequential["folds"], strict=True):
        losses_a = [epoch["train_loss"] for epoch in a["train"]["history"]]
        losses_b = [epoch["train_loss"] for epoch in b["train"]["history"]]
        assert losses_a == losses_b
