"""E6 building blocks (neuros.eval.state) on synthetic data: no EEGMAT files needed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuros.data.recording import EventTable, Recording
from neuros.errors import NeurOSConfigError, NeurOSDataError
from neuros.eval import state
from neuros.eval.state import (
    REST,
    TASK,
    PersonFeatures,
    StateBar,
    StateExperimentConfig,
    balanced_accuracy,
    block_accuracy,
    drift_labels,
    late_rest_labels,
    leave_one_person_out,
    load_state_config,
    log_band_power,
    main_labels,
    make_classifier,
    person_features,
    segment_windows,
    summarise,
    tangent_at_identity,
    verdict,
)

SFREQ = 128.0
CHANNELS = ("Fz", "C3", "Cz", "C4", "Pz")


def _config(**overrides: object) -> StateExperimentConfig:
    body: dict[str, object] = {"arms": {"A": CHANNELS, "B": ("C3", "Cz", "C4", "Pz")}}
    body.update(overrides)
    return StateExperimentConfig.model_validate(body)


def _recording(seconds: float, subject: int, run: str, alpha_uv: float, seed: int) -> Recording:
    """Noise plus a 10-Hz rhythm of the given amplitude on every channel, same pattern for all."""
    rng = np.random.default_rng(seed)
    t = np.arange(round(seconds * SFREQ)) / SFREQ
    data = rng.normal(0.0, 2.0, (len(CHANNELS), t.size))
    # One spatial pattern for everyone (alpha topography is broadly shared across people).
    phases = np.linspace(0.0, 1.0, len(CHANNELS))[:, None]
    data += alpha_uv * np.sin(2 * np.pi * 10.0 * t + phases)
    return Recording(
        data=data.astype(np.float32),
        sfreq=SFREQ,
        ch_names=CHANNELS,
        events=EventTable.from_pairs(
            [(0, "state_rest" if run.startswith("1") else "state_arithmetic")]
        ),
        dataset="EEGMAT",
        subject=subject,
        session="0",
        run=run,
    )


def _person(
    subject: int, rest_alpha: float, task_alpha: float, rest_s: float = 120.0
) -> state.PersonWindows:
    rest = _recording(rest_s, subject, "1_background", rest_alpha, seed=subject * 2)
    task = _recording(62.0, subject, "2_task", task_alpha, seed=subject * 2 + 1)
    return segment_windows(rest, task, _config(), window_s=2.0)


def test_config_file_loads_and_refuses_unknown_keys(tmp_path: Path) -> None:
    config = load_state_config(Path("configs/e6.yaml"))
    assert config.bar.ship_mean == 0.70
    assert config.arms["B"] == ("C3", "Cz", "C4", "Pz")
    bad = tmp_path / "e6.yaml"
    bad.write_text("arms: {A: [C3, C4]}\nsurprise: 1\n", encoding="utf-8")
    with pytest.raises(NeurOSConfigError, match="surprise"):
        load_state_config(bad)


def test_config_rejects_scored_rest_inside_the_baseline() -> None:
    with pytest.raises(ValueError, match="baseline"):
        _config(baseline_s=30.0)


def test_segment_windows_tags_and_skips_the_filter_start() -> None:
    person = _person(1, 10.0, 2.0)
    assert person.start_s.min() >= 0.5  # nothing inside the filters' start-up
    rest = person.recording == REST
    assert person.within(REST, 0.0, 20.0).sum() > 0  # the baseline
    scored = person.within(REST, 20.0, 60.0)
    assert scored.sum() == 77  # (60 - 20 - 2) / 0.5 + 1
    # Rest windows outside baseline, scored and late segments are dropped.
    middle = rest & (person.start_s > 60.0) & (person.start_s + 2.0 < 80.0)
    assert not middle.any()
    assert (person.recording == TASK).sum() == 120  # (62 - 2) / 0.5 + 1, minus the start-up


def test_segment_windows_refuses_short_rest() -> None:
    rest = _recording(50.0, 1, "1_background", 1.0, 0)
    task = _recording(62.0, 1, "2_task", 1.0, 1)
    with pytest.raises(NeurOSDataError, match="shorter"):
        segment_windows(rest, task, _config(), window_s=2.0)


def test_log_band_power_finds_the_rhythm() -> None:
    t = np.arange(256) / SFREQ
    x = np.sin(2 * np.pi * 10.0 * t)[None, None, :] * np.ones((1, 2, 1))
    power = log_band_power(x, SFREQ, {"theta": (4.0, 8.0), "alpha": (8.0, 13.0)})
    assert power.shape == (1, 4)
    theta, alpha = power[0, 0], power[0, 1]
    assert alpha > theta + 2  # 100x more power in alpha


def test_log_band_power_refuses_non_finite_input() -> None:
    x = np.full((1, 2, 256), np.nan)
    with pytest.raises(NeurOSDataError, match="non-finite"):
        log_band_power(x, SFREQ, {"alpha": (8.0, 13.0)})


def test_tangent_at_identity_is_zero_at_the_identity() -> None:
    eye = np.stack([np.eye(3), np.eye(3)])
    assert np.allclose(tangent_at_identity(eye), 0.0)


def test_baseline_normalisation_zeroes_the_baseline() -> None:
    person = _person(1, 10.0, 2.0)
    features = person_features(person, CHANNELS, "bandpower", "baseline", _config())
    baseline = person.within(REST, 0.0, 20.0)
    assert np.allclose(features.features[baseline].mean(axis=0), 0.0, atol=1e-9)


def test_labellings() -> None:
    config = _config()
    person = person_features(_person(1, 10.0, 2.0), CHANNELS, "bandpower", "baseline", config)
    mask, labels = main_labels(config)(person)
    assert set(labels[mask].tolist()) == {0, 1}
    assert not (mask & person.within(REST, 0.0, 20.0)).any()  # baseline never scored
    mask, labels = drift_labels(config)(person)
    assert mask.sum() == 2 * 37  # two 20-s halves of rest
    assert labels[mask].sum() == 37
    late = late_rest_labels(config)(person)
    assert late is not None
    short = person_features(
        _person(2, 10.0, 2.0, rest_s=80.0), CHANNELS, "bandpower", "baseline", config
    )
    assert late_rest_labels(config)(short) is None  # under 120 s of rest


@pytest.mark.parametrize("model", ["bandpower", "ts_lr"])
def test_loso_separates_a_clear_state_and_not_a_fake_one(model: state.Model) -> None:
    config = _config()
    people = [_person(s, 12.0, 3.0) for s in range(1, 6)]  # alpha drops in the task
    features = [person_features(p, CHANNELS, model, "baseline", config) for p in people]
    clear = leave_one_person_out(
        features, main_labels(config), lambda: make_classifier(model, "baseline")
    )
    assert np.mean([r.balanced_accuracy for r in clear]) > 0.9
    drift = leave_one_person_out(
        features, drift_labels(config), lambda: make_classifier(model, "baseline")
    )
    assert np.mean([r.balanced_accuracy for r in drift]) < 0.75  # rest vs rest: no real signal


def test_loso_needs_three_people() -> None:
    config = _config()
    features = [
        person_features(_person(s, 10.0, 2.0), CHANNELS, "bandpower", "baseline", config)
        for s in (1, 2)
    ]
    with pytest.raises(NeurOSDataError, match="only 2 people"):
        leave_one_person_out(
            features, main_labels(config), lambda: make_classifier("bandpower", "baseline")
        )


def test_balanced_accuracy_and_its_boundaries() -> None:
    labels = np.array([0, 0, 1, 1])
    assert balanced_accuracy(labels, np.array([0.1, 0.9, 0.8, 0.7])) == 0.75
    with pytest.raises(NeurOSDataError, match="class 1"):
        balanced_accuracy(np.array([0, 0]), np.array([0.2, 0.3]))


def test_block_accuracy_averages_inside_blocks() -> None:
    result = state.PersonResult(
        person="EEGMAT:1",
        balanced_accuracy=0.5,
        auroc=None,
        n_windows=4,
        probs=np.array([0.6, 0.2, 0.9, 0.7]),  # rest block mean 0.4 → right; task 0.8 → right
        labels=np.array([0, 0, 1, 1]),
        start_s=np.array([20.0, 22.0, 0.5, 2.5]),
    )
    out = block_accuracy([result], block_s=10.0)
    assert out["pooled_correct"] == 2
    assert out["mean_balanced_accuracy"] == 1.0


def _summary(mean: float, people: int, p: float) -> dict[str, object]:
    return {"mean": mean, "people_at_or_above": people, "wilcoxon_p_vs_50": p}


def test_verdict_applies_every_part_of_the_bar() -> None:
    bar = StateBar()
    main = {"A": _summary(0.72, 21, 0.001), "B": _summary(0.66, 15, 0.01)}
    calm = {"A": {"mean": 0.52}, "B": {"mean": 0.55}}
    assert verdict(main, calm, bar)["ship"]["outcome"] == "PASS"
    assert verdict(main, calm, bar)["demo"]["outcome"] == "PASS"
    few = {**main, "A": _summary(0.72, 19, 0.001)}
    assert verdict(few, calm, bar)["ship"]["outcome"] == "FAIL"
    drifting = {"A": {"mean": 0.65}, "B": {"mean": 0.55}}
    assert verdict(main, drifting, bar)["ship"]["outcome"] == "FAIL (confounded)"


def test_summarise_counts_people_at_the_line() -> None:
    results = [
        state.PersonResult(
            f"EEGMAT:{i}", acc, None, 10, np.zeros(1), np.zeros(1, dtype=np.int64), np.zeros(1)
        )
        for i, acc in enumerate((0.6, 0.59, 0.8))
    ]
    summary = summarise(results, at_least=0.60)
    assert summary["people_at_or_above"] == 2
    assert summary["n_people"] == 3


def test_person_features_refuses_unknown_channels() -> None:
    with pytest.raises(NeurOSConfigError, match="O1"):
        person_features(_person(1, 10.0, 2.0), ("C3", "O1"), "bandpower", "baseline", _config())


def test_feature_rows_match_windows() -> None:
    person = _person(1, 10.0, 2.0)
    features = person_features(person, CHANNELS, "ts_lr", "baseline", _config())
    assert isinstance(features, PersonFeatures)
    assert features.features.shape == (len(person.recording), 10)  # 4x4 upper triangle (CAR: 5-1)
