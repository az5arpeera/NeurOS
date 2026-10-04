"""Metrics over people (architecture §8.2-8.3)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

from neuros.errors import NeurOSDataError
from neuros.eval.metrics import (
    balanced_accuracy,
    bootstrap_ci,
    cohens_kappa,
    confusion,
    paired_wilcoxon,
    score_person,
    summarize,
)

TRUE = ["L", "L", "L", "L", "R", "R", "R", "R", "R", "R"]
PRED = ["L", "L", "L", "R", "R", "R", "R", "R", "L", "L"]


def test_balanced_accuracy_and_kappa_match_sklearn() -> None:
    assert balanced_accuracy(TRUE, PRED) == pytest.approx(balanced_accuracy_score(TRUE, PRED))
    assert cohens_kappa(TRUE, PRED) == pytest.approx(cohen_kappa_score(TRUE, PRED))


def test_balanced_accuracy_ignores_classes_absent_from_the_truth() -> None:
    # "R" never occurs in y_true: its recall is undefined, not zero.
    assert balanced_accuracy(["L", "L"], ["L", "R"]) == 0.5
    assert balanced_accuracy(["L", "R"], ["L", "R"]) == 1.0


def test_kappa_degenerate_case_is_zero() -> None:
    assert cohens_kappa(["L", "L"], ["L", "L"]) == 0.0


def test_confusion_with_explicit_classes() -> None:
    matrix, order = confusion(TRUE, PRED, classes=["R", "L"])
    assert order == ("R", "L")
    assert matrix.tolist() == [[4, 2], [1, 3]]
    with pytest.raises(NeurOSDataError):
        confusion(TRUE, PRED, classes=["L"])


@pytest.mark.parametrize(("true", "pred"), [([], []), (["L"], ["L", "R"]), ([["L"]], [["L"]])])
def test_inputs_are_validated(true: list[object], pred: list[object]) -> None:
    with pytest.raises(NeurOSDataError):
        balanced_accuracy(true, pred)  # type: ignore[arg-type]


def test_score_person() -> None:
    score = score_person("Synthetic:1", TRUE, PRED, n_flagged=2)
    assert score.n_windows == 10
    assert score.n_flagged == 2
    assert score.to_dict()["balanced_accuracy"] == round(balanced_accuracy(TRUE, PRED), 4)


def test_summary_statistics() -> None:
    summary = summarize([0.5, 0.6, 0.7, 0.8, 0.9])
    assert summary["n_people"] == 5
    assert summary["mean"] == pytest.approx(0.7)
    assert summary["median"] == pytest.approx(0.7)
    assert summary["share_usable"] == pytest.approx(0.6)  # 0.7, 0.8, 0.9
    assert summary["ci95_low"] < summary["mean"] < summary["ci95_high"]
    with pytest.raises(NeurOSDataError):
        summarize([])


def test_bootstrap_is_deterministic_and_degenerate_safe() -> None:
    values = np.random.default_rng(0).uniform(0.5, 0.9, 30)
    assert bootstrap_ci(values) == bootstrap_ci(values)
    assert bootstrap_ci([0.7]) == (0.7, 0.7)
    with pytest.raises(NeurOSDataError):
        bootstrap_ci([0.5, float("nan")])


def test_paired_wilcoxon() -> None:
    a = [0.70, 0.75, 0.80, 0.65, 0.72, 0.78, 0.69, 0.81, 0.77]
    b = [x - 0.05 for x in a]
    result = paired_wilcoxon(a, b)
    assert result["mean_diff"] == pytest.approx(0.05)
    assert result["p_value"] < 0.01
    assert paired_wilcoxon(a, a)["p_value"] == 1.0
    with pytest.raises(NeurOSDataError):
        paired_wilcoxon(a, b[:-1])
