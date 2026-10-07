"""Euclidean Alignment, offline (architecture §5; He & Wu 2020; Wu 2025).

Per person, over that person's windows Xᵢ ∈ ℝ^{|S|×T}:

    R̄  = (1/N) Σᵢ Xᵢ Xᵢᵀ / T            arithmetic mean covariance
    R̄  ← regularize(R̄)                 shrinkage γ = 1e-3, eigenvalue floor, condition guard
    X̃ᵢ = R̄^(-1/2) Xᵢ                    aligned window

**Why it works:** the aligned mean covariance is R̄^(-1/2) R̄ R̄^(-1/2) = I (up to the tiny
shrinkage), so every person's average window looks the same to the model. If one person's signal
is another's mixed by an invertible matrix A (different electrode contact, gains, head shape),
alignment removes A up to a rotation: (A R̄ Aᵀ)^(-1/2) A = Q R̄^(-1/2) with Q orthogonal.

**What it uses:** no labels, only the person's own unlabeled windows. Offline, one reference
covers either all of a person's windows or one recording session's (``align_groups``; the
"full-session" reference). Per session matches the online engine, which starts a new
reference each time the headset goes on and estimates it from the stream with a population
prior (§5 online; E2 measures how fast it converges). ``reference_from_covariances`` is shared
by both.

**After CAR** (rank |S|-1): R̄ has a zero eigenvalue along the all-ones direction, which
shrinkage lifts to γ·mean λ. The windows carry no energy in that direction, so alignment
leaves it at ~0 and the aligned mean covariance is the projection I - 11ᵀ/|S| instead of I.
That's expected and tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from neuros.align.covariance import (
    DEFAULT_EIGEN_FLOOR,
    DEFAULT_MAX_CONDITION,
    DEFAULT_SHRINKAGE,
    RegularizedCovariance,
    regularize,
    window_covariances,
)
from neuros.errors import DataContext, NeurOSConfigError, NeurOSDataError
from neuros.signal.windows import WindowSet

# Which windows share one EA reference: all of a person's, or one recording session's.
AlignmentUnit = Literal["person", "session"]

# Below this many windows the reference is noise; E2 measures the real requirement (16-32
# trials in the literature). Offline training never gets near it; it guards misuse.
MIN_REFERENCE_WINDOWS = 2


@dataclass(frozen=True, eq=False)
class AlignmentReference:
    """One person's EA reference: R̄ (regularised) and R̄^(-1/2)."""

    covariance: RegularizedCovariance
    inv_sqrt: NDArray[np.float64]
    n_windows: int

    @property
    def n_channels(self) -> int:
        return int(self.inv_sqrt.shape[0])

    @property
    def shrinkage(self) -> float:
        return self.covariance.shrinkage

    @property
    def condition_number(self) -> float:
        return self.covariance.condition_number

    def summary(self) -> dict[str, float | int]:
        """Numbers worth logging per person (JSON-serialisable)."""
        return {
            "n_windows": self.n_windows,
            "shrinkage": self.shrinkage,
            "condition_number": self.condition_number,
            "mean_variance_uv2": float(np.mean(self.covariance.eigenvalues)),
        }


def reference_from_covariances(
    covariances: NDArray[np.floating],
    *,
    shrinkage: float = DEFAULT_SHRINKAGE,
    eigen_floor: float = DEFAULT_EIGEN_FLOOR,
    max_condition: float = DEFAULT_MAX_CONDITION,
    context: DataContext | None = None,
) -> AlignmentReference:
    """Build the EA reference from window covariances ``[n, ch, ch]``.

    Raises:
        NeurOSDataError: on fewer than ``MIN_REFERENCE_WINDOWS`` covariances, a bad shape,
            NaN/Inf, or no signal energy.
    """
    covs = np.asarray(covariances, dtype=np.float64)
    if covs.ndim != 3 or covs.shape[1] != covs.shape[2]:
        raise NeurOSDataError(f"covariances must be [n, ch, ch], got shape {covs.shape}", context)
    if covs.shape[0] < MIN_REFERENCE_WINDOWS:
        raise NeurOSDataError(
            f"an alignment reference needs >= {MIN_REFERENCE_WINDOWS} windows, got {covs.shape[0]}",
            context,
        )
    if not np.isfinite(covs).all():
        raise NeurOSDataError("covariances contain NaN/Inf", context)
    regularized = regularize(
        covs.mean(axis=0),
        shrinkage=shrinkage,
        eigen_floor=eigen_floor,
        max_condition=max_condition,
        context=context,
    )
    inv_sqrt = regularized.power(-0.5)
    inv_sqrt.flags.writeable = False
    return AlignmentReference(
        covariance=regularized, inv_sqrt=inv_sqrt, n_windows=int(covs.shape[0])
    )


def reference_from_mean(
    mean_covariance: NDArray[np.floating],
    n_windows: int,
    *,
    shrinkage: float = DEFAULT_SHRINKAGE,
    eigen_floor: float = DEFAULT_EIGEN_FLOOR,
    max_condition: float = DEFAULT_MAX_CONDITION,
    context: DataContext | None = None,
) -> AlignmentReference:
    """The EA reference from an already-averaged covariance (e.g. the online engine's blend of
    a population prior and the user's running mean, §5); ``n_windows`` is recorded only.

    Raises:
        NeurOSDataError: on a non-square or non-finite matrix, or no signal energy.
        NeurOSConfigError: if ``n_windows`` is negative.
    """
    if n_windows < 0:
        raise NeurOSConfigError(f"n_windows must be >= 0, got {n_windows}")
    mean = np.asarray(mean_covariance, dtype=np.float64)
    if mean.ndim != 2 or mean.shape[0] != mean.shape[1]:
        raise NeurOSDataError(f"a covariance must be square, got shape {mean.shape}", context)
    if not np.isfinite(mean).all():
        raise NeurOSDataError("covariance contains NaN/Inf", context)
    regularized = regularize(
        mean,
        shrinkage=shrinkage,
        eigen_floor=eigen_floor,
        max_condition=max_condition,
        context=context,
    )
    inv_sqrt = regularized.power(-0.5)
    inv_sqrt.flags.writeable = False
    return AlignmentReference(covariance=regularized, inv_sqrt=inv_sqrt, n_windows=int(n_windows))


def euclidean_reference(
    windows: NDArray[np.floating],
    *,
    shrinkage: float = DEFAULT_SHRINKAGE,
    eigen_floor: float = DEFAULT_EIGEN_FLOOR,
    max_condition: float = DEFAULT_MAX_CONDITION,
    context: DataContext | None = None,
) -> AlignmentReference:
    """EA reference from one person's windows ``[n, ch, T]``.

    Raises:
        NeurOSDataError: see ``reference_from_covariances`` and ``window_covariances``.
    """
    if not isinstance(windows, np.ndarray) or windows.ndim != 3:
        shape = getattr(windows, "shape", None)
        raise NeurOSDataError(f"windows must be [n, ch, T], got shape {shape}", context)
    return reference_from_covariances(
        window_covariances(windows),
        shrinkage=shrinkage,
        eigen_floor=eigen_floor,
        max_condition=max_condition,
        context=context,
    )


def apply_alignment(
    windows: NDArray[np.floating], reference: AlignmentReference
) -> NDArray[np.float32]:
    """``X̃ = R̄^(-1/2) X`` for every window; float32 out (the contract dtype).

    Raises:
        NeurOSDataError: if the channel count doesn't match the reference.
    """
    if not isinstance(windows, np.ndarray) or windows.ndim not in (2, 3):
        raise NeurOSDataError(f"windows must be [n, ch, T] or [ch, T], got {windows.shape}")
    if windows.shape[-2] != reference.n_channels:
        raise NeurOSDataError(
            f"windows have {windows.shape[-2]} channels, the reference {reference.n_channels}"
        )
    aligned = np.matmul(reference.inv_sqrt, windows.astype(np.float64, copy=False))
    return np.asarray(aligned, dtype=np.float32)


def alignment_groups(windows: WindowSet, unit: AlignmentUnit) -> tuple[str, ...]:
    """The alignment group of every window: ``dataset:subject`` or ``dataset:subject/session``.

    Raises:
        NeurOSConfigError: on an unknown unit.
    """
    if unit == "person":
        return windows.person_ids
    if unit == "session":
        return tuple(
            f"{person}/{session}"
            for person, session in zip(windows.person_ids, windows.sessions, strict=True)
        )
    raise NeurOSConfigError(f"unknown alignment unit {unit!r} (person, session)")


def align_groups(
    windows: WindowSet,
    *,
    unit: AlignmentUnit = "person",
    shrinkage: float = DEFAULT_SHRINKAGE,
    eigen_floor: float = DEFAULT_EIGEN_FLOOR,
    max_condition: float = DEFAULT_MAX_CONDITION,
) -> tuple[WindowSet, dict[str, AlignmentReference]]:
    """Align each group's windows with a reference computed from that group's windows only.

    ``unit="person"`` pools all of a person's sessions; ``unit="session"`` aligns each
    recording session separately. Per session is the offline twin of the online engine, which
    estimates a fresh reference every time the headset goes on (electrode contact changes
    between sessions), and it's the protocol of the EA + EEGNet paper E0 reproduces.

    Either way a group never mixes people: aligning a test person touches no training data,
    and vice versa (person-level isolation, principle 2).

    Returns:
        ``(aligned windows in the original order, {group key: reference})``.

    Raises:
        NeurOSConfigError: on an unknown unit.
        NeurOSDataError: if a group has too few windows (see ``reference_from_covariances``).
    """
    group_of = np.array(alignment_groups(windows, unit))
    aligned = np.empty_like(windows.data)
    references: dict[str, AlignmentReference] = {}
    for group in dict.fromkeys(group_of.tolist()):
        rows = np.flatnonzero(group_of == group)
        person, _, session = group.partition("/")
        dataset, _, subject = person.partition(":")
        reference = euclidean_reference(
            windows.data[rows],
            shrinkage=shrinkage,
            eigen_floor=eigen_floor,
            max_condition=max_condition,
            context=DataContext(dataset=dataset, subject=subject, session=session or None),
        )
        aligned[rows] = apply_alignment(windows.data[rows], reference)
        references[group] = reference
    return windows.with_data(aligned), references


def align_per_person(
    windows: WindowSet,
    *,
    shrinkage: float = DEFAULT_SHRINKAGE,
    eigen_floor: float = DEFAULT_EIGEN_FLOOR,
    max_condition: float = DEFAULT_MAX_CONDITION,
) -> tuple[WindowSet, dict[str, AlignmentReference]]:
    """``align_groups`` with one reference per person (all sessions pooled)."""
    return align_groups(
        windows,
        unit="person",
        shrinkage=shrinkage,
        eigen_floor=eigen_floor,
        max_condition=max_condition,
    )
