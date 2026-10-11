"""E6: the mental-state channel. Rest vs mental effort across people, with no calibration.

The decision, the pass bar and why each choice was made are in
``knowledge-base/plan/e6-state-channel.md`` (written before any EEGMAT data was loaded). This
module implements exactly that plan; ``configs/e6.yaml`` holds its numbers.

**Data.** EEGMAT's development people (the locked test people are never touched). Each person
has a rest recording (~180 s; 80 s for one person) and a task recording (62 s of mental serial
subtraction). Both go through the engine's causal preprocessing (4-30 Hz, 128 Hz), then 2-s
windows every 0.5 s.

**The person's baseline** is the first ``baseline_s`` (20 s) of their rest recording: it stands
in for the "relax for 20 s" start of a live session. It only normalises that person and is never
scored. The scored data are rest 20-60 s against the whole task recording.

**Two models** (fixed before running), each on three channel arms:

- ``bandpower``: log power in theta / alpha / beta per channel (common average over the arm),
  z-scored against the person's baseline windows; standardised; logistic regression.
- ``ts_lr``: window covariances (OAS, on the arm's zero-sum subspace), re-centred on the
  Riemannian mean of the person's baseline covariances, mapped to the tangent space at the
  identity; standardised; logistic regression.

Both also run with **no baseline** (no information about the test person at all; report only):
raw log band power, and covariances mapped at the training people's Riemannian mean.

**Evaluation** is leave one person out. The primary measure is each person's balanced accuracy
over the scored windows. Controls:

- **Drift** (part of the bar): the same pipeline trained to tell rest 20-40 s from rest 40-60 s.
  Both are rest, so anything it decodes is drift over time, not effort.
- **Late rest** (report only, added 11 Oct after the files turned out to hold ~3 min of rest,
  before any model ran): rest 20-60 s against the last 40 s of rest, for people with at least
  120 s of rest. Two minutes of drift is closer to the gap between the two recordings.

**Data checks** run first:
- **The spectrum:** theta and alpha must be present; the PhysioNet page's "30 Hz high-pass"
  would remove them, and the run stops.
- **Eyes:** parietal-occipital alpha at rest against the task; a median ratio above 2 suggests
  eyes closed at rest.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuros.data.recording import Recording
from neuros.errors import NeurOSConfigError, NeurOSDataError
from neuros.signal.filters import TRANSIENT_S
from neuros.signal.reference import car_basis

log = logging.getLogger(__name__)

Model = Literal["bandpower", "ts_lr"]
Condition = Literal["baseline", "none"]
MODELS: tuple[Model, ...] = ("bandpower", "ts_lr")
CONDITIONS: tuple[Condition, ...] = ("baseline", "none")
REST, TASK = 0, 1  # class codes: 1 = mental effort (the "positive" class for AUROC)
_STD_FLOOR = 1e-6
_LR_C = 1.0  # sklearn's default, made explicit; never tuned
_LR_MAX_ITER = 2000


# ---------------------------------------------------------------------------------- config
class StateBar(BaseModel):
    """The pass bar (e6-state-channel.md), fixed before running."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ship_mean: float = Field(default=0.70, gt=0.5, lt=1.0)
    ship_min_people_at: float = Field(default=0.60, gt=0.5, lt=1.0)
    ship_min_people: int = Field(default=20, ge=1)
    demo_mean: float = Field(default=0.65, gt=0.5, lt=1.0)
    p_value: float = Field(default=0.05, gt=0.0, lt=1.0)
    drift_max: float = Field(default=0.60, gt=0.5, lt=1.0)
    ship_arm: str = "A"
    demo_arm: str = "B"


class StateExperimentConfig(BaseModel):
    """``configs/e6.yaml``: every number of the E6 plan. Unknown keys are refused."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = "e6"
    dataset: str = "EEGMAT"
    band_hz: tuple[float, float] = (4.0, 30.0)
    hop_s: float = Field(default=0.5, gt=0)
    baseline_s: float = Field(default=20.0, gt=0)
    rest_scored_s: tuple[float, float] = (20.0, 60.0)
    drift_halves_s: tuple[tuple[float, float], tuple[float, float]] = ((20.0, 40.0), (40.0, 60.0))
    late_rest_s: float = Field(default=40.0, gt=0)
    late_rest_min_rest_s: float = Field(default=120.0, gt=0)
    block_s: float = Field(default=10.0, gt=0)
    bands_hz: Mapping[str, tuple[float, float]] = Field(
        default_factory=lambda: {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
    )
    arms: Mapping[str, tuple[str, ...]]
    eyes_channels: tuple[str, ...] = ("P3", "P4", "Pz", "O1", "O2")
    eyes_closed_ratio: float = Field(default=2.0, gt=1.0)
    spectrum_low_hz: tuple[float, float] = (4.0, 13.0)
    spectrum_high_hz: tuple[float, float] = (30.0, 45.0)
    bar: StateBar = StateBar()

    @model_validator(mode="after")
    def _check(self) -> StateExperimentConfig:
        low, high = self.band_hz
        if not 0 < low < high:
            raise ValueError(f"band {self.band_hz} must satisfy 0 < low < high")
        start, end = self.rest_scored_s
        if not self.baseline_s <= start < end:
            raise ValueError("scored rest must start at or after the baseline's end")
        for a, b in self.drift_halves_s:
            if not (start <= a < b <= end):
                raise ValueError("each drift half must lie inside the scored rest")
        if self.drift_halves_s[0][1] > self.drift_halves_s[1][0]:
            raise ValueError("the drift halves must not overlap")
        for name, (a, b) in self.bands_hz.items():
            if not low <= a < b <= high:
                raise ValueError(f"band {name} {a}-{b} Hz must lie inside {self.band_hz}")
        for arm, channels in self.arms.items():
            if len(channels) < 2 or len(set(channels)) != len(channels):
                raise ValueError(f"arm {arm} needs at least 2 distinct channels")
        for arm in (self.bar.ship_arm, self.bar.demo_arm):
            if arm not in self.arms:
                raise ValueError(f"the bar names arm {arm!r}, which isn't defined")
        return self


def load_state_config(path: str | Path) -> StateExperimentConfig:
    """Load ``configs/e6.yaml``.

    Raises:
        NeurOSConfigError: if the file is missing or invalid.
    """
    file = Path(path)
    if not file.is_file():
        raise NeurOSConfigError(f"no experiment config at {file}")
    try:
        body = yaml.safe_load(file.read_text(encoding="utf-8"))
        return StateExperimentConfig.model_validate(body)
    except (yaml.YAMLError, ValueError) as exc:
        raise NeurOSConfigError(f"{file}: {exc}") from exc


# ---------------------------------------------------------------------------------- windows
def _within(
    recording: NDArray[np.int8],
    start_s: NDArray[np.float64],
    window_s: float,
    which: int,
    start: float,
    end: float,
) -> NDArray[np.bool_]:
    """Windows of recording ``which`` lying entirely inside [start, end] seconds (1-ns slack
    for float rounding of window times)."""
    inside = (recording == which) & (start_s >= start - 1e-9) & (start_s + window_s <= end + 1e-9)
    return np.asarray(inside, dtype=np.bool_)


@dataclass(frozen=True)
class PersonWindows:
    """One person's windows: every one the plan uses, with its segment tags.

    ``data`` holds all of the dataset's channels (``ch_names``) after preprocessing and before
    any re-referencing; arms pick their channels and re-reference later.
    """

    person: str
    data: NDArray[np.float32]  # [n, channels, samples]
    ch_names: tuple[str, ...]
    sfreq: float
    recording: NDArray[np.int8]  # REST or TASK: which recording the window comes from
    start_s: NDArray[np.float64]  # window start within its recording
    rest_duration_s: float
    window_s: float

    @property
    def end_s(self) -> NDArray[np.float64]:
        return self.start_s + self.window_s

    def within(self, which: int, start: float, end: float) -> NDArray[np.bool_]:
        """Windows of recording ``which`` lying entirely inside [start, end] seconds."""
        return _within(self.recording, self.start_s, self.window_s, which, start, end)


def segment_windows(
    rest: Recording, task: Recording, config: StateExperimentConfig, window_s: float
) -> PersonWindows:
    """Cut the person's preprocessed rest and task recordings into the plan's windows.

    Keeps the windows any analysis needs (baseline, scored rest, late rest, the task), drops
    those starting within the filters' start-up (``TRANSIENT_S``) of a recording.

    Raises:
        NeurOSDataError: if the two recordings differ in person, rate or channels, or a
            recording is too short for its segments.
    """
    if (
        rest.person_id != task.person_id
        or rest.sfreq != task.sfreq
        or rest.ch_names != task.ch_names
    ):
        raise NeurOSDataError("rest and task recordings must share person, rate and channels")
    sfreq = rest.sfreq
    window = round(window_s * sfreq)
    hop = round(config.hop_s * sfreq)
    rest_duration = rest.n_samples / sfreq
    if rest_duration < config.rest_scored_s[1]:
        raise NeurOSDataError(
            f"rest is {rest_duration:.0f} s, shorter than the scored segment's end "
            f"({config.rest_scored_s[1]:g} s)",
            rest.context,
        )
    if task.n_samples / sfreq < window_s + TRANSIENT_S:
        raise NeurOSDataError("the task recording is shorter than one usable window", task.context)
    late_start = rest_duration - config.late_rest_s
    pieces: list[NDArray[np.float32]] = []
    which: list[int] = []
    starts: list[float] = []
    for code, recording in ((REST, rest), (TASK, task)):
        for begin in range(0, recording.n_samples - window + 1, hop):
            start_s = begin / sfreq
            if start_s < TRANSIENT_S:
                continue
            end_s = start_s + window_s
            if code == REST:
                needed = (
                    end_s <= config.baseline_s
                    or (start_s >= config.rest_scored_s[0] and end_s <= config.rest_scored_s[1])
                    or start_s >= late_start
                )
                if not needed:
                    continue
            pieces.append(recording.data[:, begin : begin + window])
            which.append(code)
            starts.append(start_s)
    return PersonWindows(
        person=rest.person_id,
        data=np.stack(pieces).astype(np.float32),
        ch_names=rest.ch_names,
        sfreq=sfreq,
        recording=np.asarray(which, dtype=np.int8),
        start_s=np.asarray(starts, dtype=np.float64),
        rest_duration_s=rest_duration,
        window_s=window_s,
    )


# --------------------------------------------------------------------------------- features
def arm_windows(windows: PersonWindows, channels: Sequence[str]) -> NDArray[np.float64]:
    """The arm's channels, common-average referenced over the arm (D-028: after the subset)."""
    missing = [c for c in channels if c not in windows.ch_names]
    if missing:
        raise NeurOSConfigError(f"channels {missing} aren't in {windows.person}'s recording")
    index = [windows.ch_names.index(c) for c in channels]
    x = windows.data[:, index, :].astype(np.float64)
    return np.asarray(x - x.mean(axis=1, keepdims=True), dtype=np.float64)


def log_band_power(
    x: NDArray[np.float64], sfreq: float, bands: Mapping[str, tuple[float, float]]
) -> NDArray[np.float64]:
    """log10 mean power per channel and band (Hann periodogram): ``[n, channels * bands]``.

    Raises:
        NeurOSDataError: on non-finite input, or a band with no frequency bin at this length.
    """
    if not np.isfinite(x).all():
        raise NeurOSDataError("non-finite samples in band-power input")
    n_samples = x.shape[-1]
    taper = np.hanning(n_samples)
    spectrum = np.abs(np.fft.rfft(x * taper, axis=-1)) ** 2 / (sfreq * (taper**2).sum())
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
    columns = []
    for name, (low, high) in bands.items():
        bins = (freqs >= low) & (freqs < high)
        if not bins.any():
            raise NeurOSDataError(f"band {name} ({low}-{high} Hz) has no frequency bin")
        columns.append(np.log10(spectrum[..., bins].mean(axis=-1) + 1e-12))
    return np.stack(columns, axis=-1).reshape(x.shape[0], -1)


def arm_covariances(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """OAS covariances on the zero-sum subspace (CAR makes the full matrix singular)."""
    from pyriemann.estimation import Covariances

    basis = car_basis(x.shape[1])
    projected = np.einsum("cd,nct->ndt", basis, x)
    covs = np.asarray(Covariances(estimator="oas").fit_transform(projected), dtype=np.float64)
    if not np.isfinite(covs).all():
        raise NeurOSDataError("non-finite covariances")
    return covs


def recentre(covs: NDArray[np.float64], reference: NDArray[np.float64]) -> NDArray[np.float64]:
    """C ← M^(-1/2) C M^(-1/2): the person's baseline mean M becomes the identity."""
    from pyriemann.geometry.base import invsqrtm

    whitener = invsqrtm(reference)
    return np.asarray(whitener @ covs @ whitener, dtype=np.float64)


def tangent_at_identity(covs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Upper triangle of log(C), off-diagonals × √2 (an isometry at the identity)."""
    from pyriemann.geometry.base import logm

    logs = np.asarray(logm(covs), dtype=np.float64)
    rows, cols = np.triu_indices(covs.shape[-1])
    weights = np.where(rows == cols, 1.0, math.sqrt(2.0))
    return logs[:, rows, cols] * weights


@dataclass
class PersonFeatures:
    """One person's features for one model, arm and condition, plus the window tags."""

    person: str
    features: NDArray[np.float64]  # vectors, or covariance matrices for (ts_lr, none)
    recording: NDArray[np.int8]
    start_s: NDArray[np.float64]
    rest_duration_s: float
    window_s: float

    def within(self, which: int, start: float, end: float) -> NDArray[np.bool_]:
        """Windows of recording ``which`` lying entirely inside [start, end] seconds."""
        return _within(self.recording, self.start_s, self.window_s, which, start, end)


def person_features(
    windows: PersonWindows,
    channels: Sequence[str],
    model: Model,
    condition: Condition,
    config: StateExperimentConfig,
) -> PersonFeatures:
    """Features for one person, normalised by their own baseline when ``condition`` says so.

    Raises:
        NeurOSDataError: if the person has fewer than 5 baseline windows.
    """
    x = arm_windows(windows, channels)
    baseline = windows.within(REST, 0.0, config.baseline_s)
    if condition == "baseline" and baseline.sum() < 5:
        raise NeurOSDataError(f"{windows.person}: only {int(baseline.sum())} baseline windows")
    if model == "bandpower":
        features = log_band_power(x, windows.sfreq, config.bands_hz)
        if condition == "baseline":
            mean = features[baseline].mean(axis=0)
            std = np.maximum(features[baseline].std(axis=0), _STD_FLOOR)
            features = (features - mean) / std
    elif model == "ts_lr":
        covs = arm_covariances(x)
        if condition == "baseline":
            from pyriemann.geometry.mean import mean_riemann

            features = tangent_at_identity(recentre(covs, mean_riemann(covs[baseline])))
        else:
            features = covs  # mapped at the training people's mean inside the classifier
    else:
        raise NeurOSConfigError(f"unknown model {model!r} {MODELS}")
    return PersonFeatures(
        person=windows.person,
        features=features,
        recording=windows.recording,
        start_s=windows.start_s,
        rest_duration_s=windows.rest_duration_s,
        window_s=windows.window_s,
    )


# ------------------------------------------------------------------------------- evaluation
# A labelling picks (windows, labels) from one person's features: None when the person has no
# usable windows for it (e.g. too little rest for the late-rest control).
Labelling = Callable[[PersonFeatures], tuple[NDArray[np.bool_], NDArray[np.int64]] | None]


def main_labels(config: StateExperimentConfig) -> Labelling:
    """Scored rest (label 0) against the whole task recording (label 1)."""

    def pick(f: PersonFeatures) -> tuple[NDArray[np.bool_], NDArray[np.int64]]:
        rest = f.within(REST, *config.rest_scored_s)
        task = f.recording == TASK
        return rest | task, task.astype(np.int64)

    return pick


def drift_labels(config: StateExperimentConfig) -> Labelling:
    """Rest's first scored half (0) against its second half (1): both rest."""
    (a0, a1), (b0, b1) = config.drift_halves_s

    def pick(f: PersonFeatures) -> tuple[NDArray[np.bool_], NDArray[np.int64]]:
        first = f.within(REST, a0, a1)
        second = f.within(REST, b0, b1)
        return first | second, second.astype(np.int64)

    return pick


def late_rest_labels(config: StateExperimentConfig) -> Labelling:
    """Scored rest (0) against the last ``late_rest_s`` of rest (1), for long enough rest."""

    def pick(f: PersonFeatures) -> tuple[NDArray[np.bool_], NDArray[np.int64]] | None:
        if f.rest_duration_s < config.late_rest_min_rest_s:
            return None
        early = f.within(REST, *config.rest_scored_s)
        late = f.within(REST, f.rest_duration_s - config.late_rest_s, f.rest_duration_s)
        if early.sum() == 0 or late.sum() == 0 or np.any(early & late):
            return None
        return early | late, late.astype(np.int64)

    return pick


def make_classifier(model: Model, condition: Condition) -> Any:
    """A fresh, unfitted classifier for the model and condition."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    steps: list[Any] = []
    if model == "ts_lr" and condition == "none":
        from pyriemann.tangentspace import TangentSpace

        steps.append(TangentSpace(metric="riemann"))
    steps += [StandardScaler(), LogisticRegression(C=_LR_C, max_iter=_LR_MAX_ITER)]
    return make_pipeline(*steps)


@dataclass(frozen=True)
class PersonResult:
    """One held-out person's scores."""

    person: str
    balanced_accuracy: float
    auroc: float | None
    n_windows: int
    probs: NDArray[np.float64] = field(repr=False)
    labels: NDArray[np.int64] = field(repr=False)
    start_s: NDArray[np.float64] = field(repr=False)


def balanced_accuracy(labels: NDArray[np.int64], probs: NDArray[np.float64]) -> float:
    """Mean recall of the two classes at a 0.5 threshold.

    Raises:
        NeurOSDataError: if either class is absent.
    """
    predicted = (probs >= 0.5).astype(np.int64)
    recalls = []
    for cls in (0, 1):
        mask = labels == cls
        if not mask.any():
            raise NeurOSDataError(f"class {cls} has no windows")
        recalls.append(float((predicted[mask] == cls).mean()))
    return float(np.mean(recalls))


def leave_one_person_out(
    people: Sequence[PersonFeatures],
    labelling: Labelling,
    make_model: Callable[[], Any],
) -> list[PersonResult]:
    """Train on everyone else, score each person; people without usable windows are skipped.

    Raises:
        NeurOSDataError: if fewer than 3 people have usable windows, or a training set lacks a
            class.
    """
    from sklearn.metrics import roc_auc_score

    picked = {}
    for person in people:
        choice = labelling(person)
        if choice is None:
            continue
        mask, labels = choice
        if mask.sum() == 0 or len(set(labels[mask].tolist())) < 2:
            continue
        picked[person.person] = (person, mask, labels[mask])
    if len(picked) < 3:
        raise NeurOSDataError(f"only {len(picked)} people have usable windows for this labelling")
    results: list[PersonResult] = []
    for held_out, (person, mask, labels) in picked.items():
        train_x = np.concatenate(
            [p.features[m] for name, (p, m, _) in picked.items() if name != held_out]
        )
        train_y = np.concatenate([y for name, (_, _, y) in picked.items() if name != held_out])
        if len(set(train_y.tolist())) < 2:
            raise NeurOSDataError("a training fold lacks a class")
        model = make_model()
        model.fit(train_x, train_y)
        probs = np.asarray(model.predict_proba(person.features[mask])[:, 1], dtype=np.float64)
        auroc = float(roc_auc_score(labels, probs)) if len(set(labels.tolist())) == 2 else None
        results.append(
            PersonResult(
                person=held_out,
                balanced_accuracy=balanced_accuracy(labels, probs),
                auroc=auroc,
                n_windows=int(mask.sum()),
                probs=probs,
                labels=labels,
                start_s=person.start_s[mask],
            )
        )
    return results


def block_accuracy(results: Sequence[PersonResult], block_s: float) -> dict[str, Any]:
    """Accuracy of the state over ``block_s`` blocks: the mean probability of the windows inside
    each block (non-overlapping blocks per recording, aligned to the scored segment's start).

    Returns the mean of per-person balanced block accuracies and the pooled counts.
    """
    per_person: list[float] = []
    right = total = 0
    for result in results:
        block_probs: list[float] = []
        block_labels: list[int] = []
        for cls in (0, 1):
            mask = result.labels == cls
            if not mask.any():
                continue
            starts = result.start_s[mask]
            origin = float(starts.min())
            index = np.floor((starts - origin + 1e-9) / block_s).astype(int)
            for block in np.unique(index):
                members = index == block
                block_probs.append(float(result.probs[mask][members].mean()))
                block_labels.append(cls)
        labels = np.asarray(block_labels, dtype=np.int64)
        probs = np.asarray(block_probs, dtype=np.float64)
        right += int(((probs >= 0.5).astype(int) == labels).sum())
        total += len(labels)
        if len(set(block_labels)) == 2:
            per_person.append(balanced_accuracy(labels, probs))
    return {
        "block_s": block_s,
        "mean_balanced_accuracy": float(np.mean(per_person)) if per_person else None,
        "pooled_correct": right,
        "pooled_blocks": total,
    }


def summarise(results: Sequence[PersonResult], at_least: float) -> dict[str, Any]:
    """Mean, spread, people at or above ``at_least``, and Wilcoxon against 50% (two-sided)."""
    from scipy.stats import wilcoxon

    accuracies = np.array([r.balanced_accuracy for r in results])
    aurocs = [r.auroc for r in results if r.auroc is not None]
    centred = accuracies - 0.5
    p_value = 1.0 if np.allclose(centred, 0) else float(wilcoxon(centred).pvalue)
    return {
        "n_people": int(accuracies.size),
        "mean": float(accuracies.mean()),
        "median": float(np.median(accuracies)),
        "min": float(accuracies.min()),
        "max": float(accuracies.max()),
        "people_at_or_above": int((accuracies >= at_least - 1e-12).sum()),
        "at_or_above": at_least,
        "wilcoxon_p_vs_50": p_value,
        "mean_auroc": float(np.mean(aurocs)) if aurocs else None,
        "per_person": {r.person: round(r.balanced_accuracy, 4) for r in results},
    }


# ------------------------------------------------------------------------------ data checks
def spectrum_check(
    rest: Recording, low_hz: tuple[float, float], high_hz: tuple[float, float]
) -> float:
    """Ratio of mean power in ``low_hz`` to ``high_hz`` at the native rate (before our filters),
    averaged over channels. Far below 1 means theta and alpha were filtered out."""
    from neuros.eval.smr import welch_psd

    freqs, power = welch_psd(rest.data, rest.sfreq)
    low = power[:, (freqs >= low_hz[0]) & (freqs < low_hz[1])].mean()
    high = power[:, (freqs >= high_hz[0]) & (freqs < high_hz[1])].mean()
    return float(low / max(high, 1e-30))


def eyes_check(rest: Recording, task: Recording, channels: Sequence[str]) -> float:
    """Alpha (8-13 Hz) power at rest over alpha during the task, parietal-occipital mean."""
    from neuros.eval.smr import welch_psd

    ratios = []
    for name in channels:
        if name not in rest.ch_names:
            raise NeurOSConfigError(f"eyes-check channel {name} isn't recorded")
        row = rest.ch_names.index(name)
        alpha = []
        for recording in (rest, task):
            freqs, power = welch_psd(recording.data[row : row + 1], recording.sfreq)
            alpha.append(float(power[0, (freqs >= 8.0) & (freqs < 13.0)].mean()))
        ratios.append(alpha[0] / max(alpha[1], 1e-30))
    return float(np.mean(ratios))


# ---------------------------------------------------------------------------------- verdict
def verdict(
    main: Mapping[str, Mapping[str, Any]],
    drift: Mapping[str, Mapping[str, Any]],
    bar: StateBar,
) -> dict[str, Any]:
    """Apply the pass bar to the primary model's summaries, keyed by arm.

    Arm A (ship) passes on all of: mean ≥ ``ship_mean``, Wilcoxon p < ``p_value``, at least
    ``ship_min_people`` people at ≥ ``ship_min_people_at``, and drift mean ≤ ``drift_max``.
    Arm B (demo) passes on mean ≥ ``demo_mean``, p < ``p_value`` and the same drift limit.
    A drift failure is reported as "FAIL (confounded)".
    """

    def judge(arm: str, mean_bar: float, people_bar: int | None) -> dict[str, Any]:
        summary, drift_summary = main[arm], drift[arm]
        checks = {
            "mean": summary["mean"] >= mean_bar,
            "p_value": summary["wilcoxon_p_vs_50"] < bar.p_value,
            "drift": drift_summary["mean"] <= bar.drift_max,
        }
        if people_bar is not None:
            checks["people"] = summary["people_at_or_above"] >= people_bar
        if all(checks.values()):
            outcome = "PASS"
        elif not checks["drift"]:
            outcome = "FAIL (confounded)"
        else:
            outcome = "FAIL"
        return {"outcome": outcome, "checks": checks}

    return {
        "ship": judge(bar.ship_arm, bar.ship_mean, bar.ship_min_people),
        "demo": judge(bar.demo_arm, bar.demo_mean, None),
    }


# ------------------------------------------------------------------------------------ the run
def load_state_people(
    data_config: Any,
    experiment: StateExperimentConfig,
    *,
    split_path: Path,
    max_people: int | None = None,
) -> tuple[list[PersonWindows], dict[str, float], dict[str, float]]:
    """Load, validate, check and window every development person of the state dataset.

    Returns:
        ``(people, spectrum ratio per person, eyes ratio per person)``: the data checks run on
        the native-rate recordings, before our filters.

    Raises:
        NeurOSDataError, NeurOSConfigError, NeurOSSplitError: from loading, validation or the
            locked test set (a locked person can never be loaded here).
    """
    from neuros.config.schemas import SignalContract
    from neuros.data.eegmat import harmonize_eegmat_subject
    from neuros.data.recording import person_id
    from neuros.data.registry import get_spec
    from neuros.data.validate import validate_recording
    from neuros.data.vocabulary import default_vocabulary
    from neuros.eval.splits import load_locked_test_set
    from neuros.signal.preprocess import preprocess_recording

    spec = get_spec(experiment.dataset)
    split = load_locked_test_set(split_path)
    subjects = list(split.dev_subjects(spec.key))[: max_people or None]
    split.assert_disjoint([person_id(spec.key, s) for s in subjects], "E6 training and scoring")
    contract = SignalContract.model_validate(
        {
            **data_config.signal.model_dump(),
            "band_hz": experiment.band_hz,
            "hop_s": experiment.hop_s,
        }
    )
    vocabulary = default_vocabulary()

    people: list[PersonWindows] = []
    spectrum: dict[str, float] = {}
    eyes: dict[str, float] = {}
    for subject in subjects:
        rest, task = harmonize_eegmat_subject(spec, subject, vocabulary, data_config.data_dir)
        for recording in (rest, task):
            validate_recording(
                recording, spec, vocabulary, data_config.signal, data_config.validation
            ).raise_if_errors()
        spectrum[rest.person_id] = spectrum_check(
            rest, experiment.spectrum_low_hz, experiment.spectrum_high_hz
        )
        eyes[rest.person_id] = eyes_check(rest, task, experiment.eyes_channels)
        people.append(
            segment_windows(
                preprocess_recording(rest, contract),
                preprocess_recording(task, contract),
                experiment,
                contract.window_s,
            )
        )
        log.info("%s: %d windows", rest.person_id, len(people[-1].recording))
    return people, spectrum, eyes


def run_e6(
    data_config: Any,
    experiment: StateExperimentConfig,
    *,
    split_path: Path,
    max_people: int | None = None,
) -> dict[str, Any]:
    """Run E6 end to end on the development people; return the result body (no provenance).

    The data checks run first. If theta and alpha are missing (the spectrum check), the run
    stops there with verdict ``STOPPED``.

    Raises:
        NeurOSDataError, NeurOSConfigError, NeurOSSplitError: from loading, validation or the
            locked test set (a locked person can never reach training or scoring).
    """
    people, spectrum, eyes = load_state_people(
        data_config, experiment, split_path=split_path, max_people=max_people
    )

    spectrum_median = float(np.median(list(spectrum.values())))
    eyes_median = float(np.median(list(eyes.values())))
    checks = {
        "spectrum_ratio_low_over_high": {"median": spectrum_median, "per_person": spectrum},
        "theta_alpha_present": spectrum_median >= 1.0,
        "eyes_alpha_rest_over_task": {"median": eyes_median, "per_person": eyes},
        "eyes_probably_closed_at_rest": eyes_median > experiment.eyes_closed_ratio,
        "rest_durations_s": {p.person: round(p.rest_duration_s, 1) for p in people},
    }
    body: dict[str, Any] = {
        "experiment": experiment.model_dump(mode="json"),
        "people": [p.person for p in people],
        "checks": checks,
    }
    if not checks["theta_alpha_present"]:
        body["verdict"] = {"outcome": "STOPPED", "reason": "theta and alpha are missing"}
        return body

    def classifier(model: Model, condition: Condition) -> Callable[[], Any]:
        return lambda: make_classifier(model, condition)

    main: dict[str, dict[str, dict[str, Any]]] = {}
    drift: dict[str, dict[str, dict[str, Any]]] = {}
    blocks: dict[str, dict[str, dict[str, Any]]] = {}
    at_least = experiment.bar.ship_min_people_at
    for model in MODELS:
        for condition in CONDITIONS:
            key = f"{model}/{condition}"
            for arm, channels in experiment.arms.items():
                features = [
                    person_features(p, channels, model, condition, experiment) for p in people
                ]
                results = leave_one_person_out(
                    features, main_labels(experiment), classifier(model, condition)
                )
                main.setdefault(key, {})[arm] = summarise(results, at_least)
                if condition == "baseline":
                    drift_results = leave_one_person_out(
                        features, drift_labels(experiment), classifier(model, condition)
                    )
                    drift.setdefault(model, {})[arm] = summarise(drift_results, at_least)
                    blocks.setdefault(model, {})[arm] = block_accuracy(results, experiment.block_s)
                log.info("%s arm %s: mean %.3f", key, arm, main[key][arm]["mean"])

    ship_arm = experiment.bar.ship_arm
    primary: Model = max(MODELS, key=lambda m: main[f"{m}/baseline"][ship_arm]["mean"])
    late: dict[str, Any] = {"model": primary}
    for arm in dict.fromkeys((experiment.bar.ship_arm, experiment.bar.demo_arm)):
        features = [
            person_features(p, experiment.arms[arm], primary, "baseline", experiment)
            for p in people
        ]
        late[arm] = summarise(
            leave_one_person_out(
                features, late_rest_labels(experiment), classifier(primary, "baseline")
            ),
            at_least,
        )
    body.update(
        {
            "primary_model": primary,
            "main": main,
            "drift_control": drift,
            "late_rest_control": late,
            "blocks": blocks,
            "verdict": verdict(main[f"{primary}/baseline"], drift[primary], experiment.bar),
        }
    )
    return body
