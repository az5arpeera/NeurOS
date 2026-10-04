"""Riemannian baselines (architecture §8.4) on synthetic people with a cross-person shift.

Each synthetic person gets their own per-channel gains (electrode impedance, amplifier gain)
and overall scale, the kind of shift that breaks cross-person decoding. Alignment, computed
from each person's own unlabeled windows, should remove it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from neuros.data.synthetic import SyntheticConfig, make_recording
from neuros.errors import NeurOSConfigError, NeurOSDataError
from neuros.eval.baselines import (
    build_features,
    common_sessions,
    leave_one_person_out,
    make_classifier,
    run_riemannian_baselines,
    session_transfer,
)
from neuros.signal.preprocess import preprocess_recording
from neuros.signal.windows import WindowSet, extract_epochs


def _person(subject: int, session: str, seed: int, gains: np.ndarray) -> WindowSet:
    recording = make_recording(
        SyntheticConfig(n_trials=24, seed=seed), subject=subject, session=session
    )
    windows, _ = extract_epochs(preprocess_recording(recording), epoch_s=(0.5, 2.5))
    return windows.with_data((windows.data * gains[None, :, None]).astype(np.float32))


@pytest.fixture(scope="module")
def shifted_people() -> WindowSet:
    rng = np.random.default_rng(0)
    sets = []
    for subject, scale in zip((1, 2, 3, 4), (0.1, 1.0, 5.0, 20.0), strict=True):
        gains = np.exp(rng.uniform(np.log(0.25), np.log(4.0), 12)) * scale
        for session, offset in (("0train", 0), ("1test", 100)):
            sets.append(_person(subject, session, seed=subject + offset, gains=gains))
    return WindowSet.concat(sets)


@pytest.fixture(scope="module")
def loso_means(shifted_people: WindowSet) -> dict[tuple[str, str], float]:
    means = {}
    for alignment in ("none", "ea", "ra"):
        features = build_features(shifted_people, alignment=alignment)  # type: ignore[arg-type]
        for method in ("ts_lr", "mdm"):
            scores = leave_one_person_out(method, features)  # type: ignore[arg-type]
            assert [s.person for s in scores] == list(shifted_people.people)
            means[(method, alignment)] = float(np.mean([s.balanced_accuracy for s in scores]))
    return means


@pytest.mark.parametrize("method", ["ts_lr", "mdm"])
@pytest.mark.parametrize("alignment", ["ea", "ra"])
def test_alignment_removes_the_cross_person_shift(
    loso_means: dict[tuple[str, str], float], method: str, alignment: str
) -> None:
    assert loso_means[(method, alignment)] >= 0.9
    assert loso_means[(method, alignment)] >= loso_means[(method, "none")] + 0.15


def test_session_transfer_within_people(shifted_people: WindowSet) -> None:
    assert common_sessions(shifted_people) == ("0train", "1test")
    features = build_features(shifted_people, alignment="none")
    scores = session_transfer("ts_lr", features, "0train", "1test")
    assert len(scores) == 4
    assert np.mean([s.balanced_accuracy for s in scores]) >= 0.9


def test_car_reference_gives_full_rank_features(shifted_people: WindowSet) -> None:
    features = build_features(shifted_people, alignment="ea", reference="car")
    assert features.covariances.shape[1:] == (11, 11)  # 12 channels → 11 after CAR
    assert np.all(np.linalg.eigvalsh(features.covariances) > 0)


def test_full_run_is_json_serialisable(shifted_people: WindowSet) -> None:
    result = run_riemannian_baselines(
        shifted_people, methods=("mdm",), alignments=("none", "ea"), progress=lambda _: None
    )
    text = json.dumps(result)
    names = [(r["name"], r["protocol"]) for r in result["results"]]
    assert names == [
        ("mdm+none", "loso"),
        ("mdm+ea", "loso"),
        ("mdm+none", "session_transfer:0train->1test"),
    ]
    assert result["comparisons"][0]["a"] == "mdm+ea"
    assert "per_person" in text


def test_configuration_errors(shifted_people: WindowSet) -> None:
    with pytest.raises(NeurOSConfigError):
        make_classifier("svm")  # type: ignore[arg-type]
    with pytest.raises(NeurOSConfigError):
        build_features(shifted_people, alignment="rpa")  # type: ignore[arg-type]
    with pytest.raises(NeurOSConfigError):
        build_features(shifted_people, alignment="ea", reference="laplacian")  # type: ignore[arg-type]
    one_person = shifted_people.for_people({"Synthetic:1"})
    with pytest.raises(NeurOSDataError, match="at least 2 people"):
        leave_one_person_out("mdm", build_features(one_person, alignment="none"))


def test_alignment_per_session(shifted_people: WindowSet) -> None:
    by_person = build_features(shifted_people, alignment="ea")
    by_session = build_features(shifted_people, alignment="ea", unit="session")
    assert (by_person.unit, by_session.unit) == ("person", "session")
    # Each session has its own reference, so the features differ, yet both decode well.
    assert not np.allclose(by_person.covariances, by_session.covariances)
    scores = leave_one_person_out("ts_lr", by_session)
    assert np.mean([s.balanced_accuracy for s in scores]) >= 0.9
    result = run_riemannian_baselines(
        shifted_people, unit="session", methods=["mdm"], alignments=["none", "ea"]
    )
    assert result["alignment_unit"] == "session"
