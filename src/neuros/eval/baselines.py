"""Riemannian baselines for every results table (architecture §8.4, row 2).

Two classifiers on window covariance matrices, each with three per-person alignments:

| classifier | what it does |
|---|---|
| ``ts_lr`` | tangent-space map at the training mean (Riemannian metric) + logistic regression |
| ``mdm``   | minimum distance to the class means (Riemannian metric) |

| alignment | per person or per session (``unit``), label-free |
|---|---|
| ``none``  | raw covariances: the cross-person shift is left in |
| ``ea``    | Euclidean Alignment of the windows (our ``neuros.align.ea``) before covariances |
| ``ra``    | Riemannian re-centring (Zanini 2018): C ← M^(-1/2) C M^(-1/2), M = Riemannian mean |

Protocols:

- ``loso`` (leave-one-person-out, the zero-calibration setting): train on everyone else, test
  on the held-out person. Alignment uses each person's *own unlabeled* windows, all of them:
  the "full-session" reference, i.e. the upper bound that E2's warm-up curve approaches online.
- ``session_transfer`` (within-person, *not* zero-calibration): train on a person's first
  session, test on their second. A pipeline sanity check: if this is far below the literature,
  epochs, labels or filters are wrong.

Covariances use the OAS shrinkage estimator (MOABB's Riemannian pipelines do too). With
``reference="car"``, windows are first projected onto the zero-sum subspace (``car_basis``),
because CAR makes covariances singular and Riemannian geometry needs them positive definite.
"""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from neuros.align.ea import AlignmentUnit, alignment_groups, apply_alignment, euclidean_reference
from neuros.errors import DataContext, NeurOSConfigError, NeurOSDataError
from neuros.eval.metrics import PersonScore, paired_wilcoxon, score_person, summarize
from neuros.signal.reference import car_basis
from neuros.signal.windows import WindowSet

log = logging.getLogger(__name__)

Method = Literal["ts_lr", "mdm"]
Alignment = Literal["none", "ea", "ra"]
Reference = Literal["as_recorded", "car"]

METHODS: tuple[Method, ...] = ("ts_lr", "mdm")
ALIGNMENTS: tuple[Alignment, ...] = ("none", "ea", "ra")
COVARIANCE_ESTIMATOR = "oas"
# Logistic-regression settings (sklearn defaults made explicit; never tuned on test people).
_LR_C = 1.0
_LR_MAX_ITER = 2000


@dataclass(frozen=True)
class FeatureSet:
    """Covariance matrices ready for a Riemannian classifier, with per-window metadata."""

    covariances: NDArray[np.float64]  # [n, c, c], SPD
    labels: NDArray[np.str_]
    people: NDArray[np.str_]
    sessions: NDArray[np.str_]
    flagged: NDArray[np.bool_]
    alignment: Alignment
    reference: Reference
    unit: AlignmentUnit = "person"


def _project(windows: WindowSet, reference: Reference) -> NDArray[np.float64]:
    data = windows.data.astype(np.float64)
    if reference == "as_recorded":
        return data
    if reference == "car":
        basis = car_basis(len(windows.ch_names))  # [c, c-1]
        return np.asarray(np.einsum("cd,nct->ndt", basis, data), dtype=np.float64)
    raise NeurOSConfigError(f"unknown reference {reference!r} (as_recorded, car)")


def build_features(
    windows: WindowSet,
    *,
    alignment: Alignment,
    reference: Reference = "as_recorded",
    unit: AlignmentUnit = "person",
) -> FeatureSet:
    """Covariances for a baseline, aligned per person or per recording session.

    Each group's alignment uses only that group's windows, so it never mixes people
    (person-level isolation). ``unit="session"`` matches the deep model's E0 protocol.

    Raises:
        NeurOSConfigError: on an unknown alignment or reference.
        NeurOSDataError: on an empty window set or a degenerate person (e.g. all-zero data).
    """
    from pyriemann.geometry.base import invsqrtm
    from pyriemann.geometry.covariance import covariances
    from pyriemann.geometry.mean import mean_riemann

    if alignment not in ALIGNMENTS:
        raise NeurOSConfigError(f"unknown alignment {alignment!r} {ALIGNMENTS}")
    if len(windows) == 0:
        raise NeurOSDataError("no windows to build features from")
    data = _project(windows, reference)
    people = np.array(windows.person_ids)
    group_of = np.array(alignment_groups(windows, unit))
    groups = list(dict.fromkeys(group_of.tolist()))
    if alignment == "ea":
        for group in groups:
            rows = np.flatnonzero(group_of == group)
            person, _, session = group.partition("/")
            dataset, _, subject = person.partition(":")
            reference_matrix = euclidean_reference(
                data[rows],
                context=DataContext(dataset=dataset, subject=subject, session=session or None),
            )
            data[rows] = apply_alignment(data[rows], reference_matrix).astype(np.float64)
    covs = np.asarray(covariances(data, estimator=COVARIANCE_ESTIMATOR), dtype=np.float64)
    if alignment == "ra":
        for group in groups:
            rows = np.flatnonzero(group_of == group)
            mean = mean_riemann(covs[rows])
            whitener = invsqrtm(mean)
            covs[rows] = whitener @ covs[rows] @ whitener
    if not np.isfinite(covs).all():
        raise NeurOSDataError(f"non-finite covariances after {alignment} alignment")
    return FeatureSet(
        covariances=covs,
        labels=np.array(windows.labels),
        people=people,
        sessions=np.array(windows.sessions),
        flagged=windows.flags != 0,
        alignment=alignment,
        reference=reference,
        unit=unit,
    )


def make_classifier(method: Method) -> Any:
    """A fresh, unfitted sklearn-compatible classifier for ``method``."""
    from pyriemann.classification import MDM
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    if method == "ts_lr":
        return make_pipeline(
            TangentSpace(metric="riemann"),
            LogisticRegression(C=_LR_C, max_iter=_LR_MAX_ITER),
        )
    if method == "mdm":
        return MDM(metric="riemann")
    raise NeurOSConfigError(f"unknown method {method!r} {METHODS}")


def _fit_predict(
    method: Method, features: FeatureSet, train: NDArray[np.bool_], test: NDArray[np.bool_]
) -> NDArray[Any]:
    train_labels = features.labels[train]
    if len(set(train_labels.tolist())) < 2:
        raise NeurOSDataError("training data has fewer than 2 classes")
    model = make_classifier(method)
    model.fit(features.covariances[train], train_labels)
    return np.asarray(model.predict(features.covariances[test]))


# The feature set a parallel fold worker scores, set once per worker process by
# ``_init_fold_worker`` so the (possibly large) covariance array crosses the process boundary
# once per worker rather than once per fold. Unused in the parent process.
_WORKER_FEATURES: FeatureSet | None = None


def _init_fold_worker(features: FeatureSet, threads: int) -> None:
    """Worker start-up: keep the features and cap the BLAS threads (no oversubscription)."""
    global _WORKER_FEATURES
    _WORKER_FEATURES = features
    try:
        from threadpoolctl import threadpool_limits  # ships with scikit-learn

        threadpool_limits(limits=threads)
    except ImportError:  # pragma: no cover - scikit-learn always installs threadpoolctl
        log.warning("threadpoolctl unavailable: BLAS threads in fold workers are not capped")


def _loso_fold(method: Method, person: str, features: FeatureSet | None = None) -> PersonScore:
    """Score one held-out person (module-level, so worker processes can run it)."""
    data = features if features is not None else _WORKER_FEATURES
    if data is None:
        raise NeurOSConfigError("fold worker has no features (initializer not run)")
    test = data.people == person
    predicted = _fit_predict(method, data, ~test, test)
    return score_person(
        person,
        data.labels[test],
        predicted,
        n_flagged=int(np.count_nonzero(data.flagged[test])),
    )


def leave_one_person_out(
    method: Method, features: FeatureSet, *, workers: int = 1
) -> list[PersonScore]:
    """Train on all people but one, test on that person; repeat for everyone.

    Folds are independent, so ``workers > 1`` runs them in parallel processes with the same
    results in the same order. Each fold refits the Riemannian mean of every training
    covariance, which is what makes many-channel datasets slow: run one fold at a time, a
    PhysionetMI audit (84 people, 64 channels) needed about 10 minutes per cell (4 Oct).

    Raises:
        NeurOSConfigError: if ``workers`` < 1.
        NeurOSDataError: with fewer than 2 people.
    """
    if workers < 1:
        raise NeurOSConfigError(f"workers must be >= 1, got {workers}")
    people = list(dict.fromkeys(features.people.tolist()))
    if len(people) < 2:
        raise NeurOSDataError("leave-one-person-out needs at least 2 people")
    workers = min(workers, len(people))
    if workers == 1:
        return [_loso_fold(method, person, features) for person in people]

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    # Spawn, as for the deep folds: identical behaviour on Windows, macOS and Linux.
    threads = max(1, (os.cpu_count() or 1) // workers)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_fold_worker,
        initargs=(features, threads),
    ) as pool:
        return list(pool.map(functools.partial(_loso_fold, method), people))


def session_transfer(
    method: Method, features: FeatureSet, train_session: str, test_session: str
) -> list[PersonScore]:
    """Within each person: train on ``train_session``, test on ``test_session``.

    People missing either session are skipped with a warning (reported in the log).
    """
    scores: list[PersonScore] = []
    for person in dict.fromkeys(features.people.tolist()):
        mine = features.people == person
        train = mine & (features.sessions == train_session)
        test = mine & (features.sessions == test_session)
        if not train.any() or not test.any():
            log.warning("%s lacks session %s or %s; skipped", person, train_session, test_session)
            continue
        predicted = _fit_predict(method, features, train, test)
        scores.append(
            score_person(
                person,
                features.labels[test],
                predicted,
                n_flagged=int(np.count_nonzero(features.flagged[test])),
            )
        )
    return scores


def _result(name: str, protocol: str, scores: Sequence[PersonScore]) -> dict[str, Any]:
    accuracies = [s.balanced_accuracy for s in scores]
    return {
        "name": name,
        "protocol": protocol,
        "summary_balanced_accuracy": summarize(accuracies),
        "summary_kappa": summarize([s.kappa for s in scores]),
        "per_person": [s.to_dict() for s in scores],
    }


def common_sessions(windows: WindowSet) -> tuple[str, str] | None:
    """The (first, second) session names if every person has exactly the same two sessions."""
    sessions_by_person: dict[str, set[str]] = {}
    for person, session in zip(windows.person_ids, windows.sessions, strict=True):
        sessions_by_person.setdefault(person, set()).add(session)
    distinct = {frozenset(s) for s in sessions_by_person.values()}
    if len(distinct) != 1:
        return None
    (only,) = distinct
    if len(only) != 2:
        return None
    first, second = sorted(only)
    return first, second


def run_riemannian_baselines(
    windows: WindowSet,
    *,
    reference: Reference = "as_recorded",
    unit: AlignmentUnit = "person",
    methods: Sequence[Method] = METHODS,
    alignments: Sequence[Alignment] = ALIGNMENTS,
    with_session_transfer: bool = True,
    workers: int = 1,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run every (method × alignment) baseline under LOSO, plus the session-transfer check.

    ``workers`` parallelises the LOSO folds (``leave_one_person_out``); results are identical.

    Returns a JSON-serialisable dict: per-person scores, across-people summaries, and paired
    Wilcoxon tests of each alignment against ``none``.
    """
    say = progress or (lambda message: log.info("%s", message))
    results: list[dict[str, Any]] = []
    loso_scores: dict[tuple[Method, Alignment], list[PersonScore]] = {}
    for alignment in alignments:
        features = build_features(windows, alignment=alignment, reference=reference, unit=unit)
        for method in methods:
            say(f"LOSO {method}+{alignment} ...")
            scores = leave_one_person_out(method, features, workers=workers)
            loso_scores[(method, alignment)] = scores
            results.append(_result(f"{method}+{alignment}", "loso", scores))

    comparisons: list[dict[str, Any]] = []
    for method in methods:
        if (method, "none") not in loso_scores:
            continue
        base = [s.balanced_accuracy for s in loso_scores[(method, "none")]]
        for alignment in alignments:
            if alignment == "none":
                continue
            aligned = [s.balanced_accuracy for s in loso_scores[(method, alignment)]]
            comparisons.append(
                {
                    "a": f"{method}+{alignment}",
                    "b": f"{method}+none",
                    **paired_wilcoxon(aligned, base),
                }
            )

    pair = common_sessions(windows) if with_session_transfer else None
    if pair is not None:
        features = build_features(windows, alignment="none", reference=reference)
        for method in methods:
            say(f"session transfer {method} ({pair[0]} -> {pair[1]}) ...")
            scores = session_transfer(method, features, *pair)
            results.append(
                _result(f"{method}+none", f"session_transfer:{pair[0]}->{pair[1]}", scores)
            )
    return {
        "covariance_estimator": COVARIANCE_ESTIMATOR,
        "reference": reference,
        "alignment_unit": unit,
        "results": results,
        "comparisons": comparisons,
    }
