"""From raw dataset files to model windows: harmonise → validate → preprocess → epochs.

This is the offline entry point every experiment uses (E0's baselines first), so each stage's
boundary check runs on every recording:

1. ``harmonize_moabb_subject``: canonical channels, µV float32, canonical labels at the cue;
2. ``validate_recording`` + ``raise_if_errors``: a recording that breaks the data contract
   stops the run with its dataset/subject/run context (§4.5); flags are only counted;
3. ``preprocess_recording``: the shared causal chain to 128 Hz (train–serve parity);
4. ``extract_epochs``: one window per task event at [cue + 0.5 s, cue + 2.5 s].

The disk cache of preprocessed recordings (§4.5, keyed by the preprocessing-config hash) slots
in between steps 3 and 4 when E1 needs all ~380 people. BNCI2014-001 takes seconds without it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from neuros.config.schemas import DataConfig
from neuros.data.harmonize import harmonize_moabb_subject
from neuros.data.recording import Recording, person_id
from neuros.data.registry import DatasetSpec
from neuros.data.validate import validate_recording
from neuros.data.vocabulary import Vocabulary
from neuros.errors import DataContext, NeurOSConfigError, NeurOSDataError
from neuros.signal.preprocess import preprocess_recording
from neuros.signal.windows import WindowFlag, WindowSet, extract_epochs

log = logging.getLogger(__name__)


def prepare_subject(
    spec: DatasetSpec,
    subject: int,
    vocabulary: Vocabulary,
    config: DataConfig,
    *,
    dataset: Any | None = None,
    allow_excluded: bool = False,
) -> list[Recording]:
    """Harmonise, validate and preprocess every run of one subject.

    Args:
        dataset: an already constructed MOABB dataset (saves re-instantiating it per subject).
        allow_excluded: process a subject the registry excludes for data quality (diagnostics
            only; never for training or reported numbers).

    Raises:
        NeurOSDataError: if the subject is excluded (and not allowed) or any recording fails
            validation; the message lists every error with its context.
    """
    if subject in spec.excluded_subjects and not allow_excluded:
        raise NeurOSDataError(
            f"subject is excluded: {spec.excluded_subjects[subject]}",
            DataContext(spec.key, subject),
        )
    prepared: list[Recording] = []
    for recording in harmonize_moabb_subject(spec, subject, vocabulary, dataset=dataset):
        report = validate_recording(recording, spec, vocabulary, config.signal, config.validation)
        report.raise_if_errors()
        if report.windows_flagged:
            log.info(
                "%s: %d/%d validation windows flagged (kept)",
                recording.context.describe(),
                report.windows_flagged,
                report.windows_total,
            )
        prepared.append(preprocess_recording(recording, config.signal))
    return prepared


@dataclass
class EpochSummary:
    """What epoch extraction did for one subject (reported, never silent)."""

    person: str
    n_recordings: int = 0
    n_windows: int = 0
    skipped_outside: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)
    flagged_extreme: int = 0
    flagged_transient: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def subject_epochs(
    spec: DatasetSpec,
    subject: int,
    vocabulary: Vocabulary,
    config: DataConfig,
    labels: Sequence[str],
    *,
    dataset: Any | None = None,
) -> tuple[WindowSet, EpochSummary]:
    """All task windows of one subject with the given labels, plus a summary.

    Raises:
        NeurOSConfigError: if a requested label isn't in the vocabulary.
        NeurOSDataError: see ``prepare_subject``; also if no window survives.
    """
    unknown = sorted(set(labels) - set(vocabulary.labels))
    if unknown or not labels:
        raise NeurOSConfigError(f"labels {unknown or '[]'} are not vocabulary labels")
    recordings = prepare_subject(spec, subject, vocabulary, config, dataset=dataset)
    summary = EpochSummary(person=person_id(spec.key, subject), n_recordings=len(recordings))
    sets: list[WindowSet] = []
    for recording in recordings:
        windows, skipped = extract_epochs(
            recording, config.signal, config.validation, labels=labels
        )
        summary.skipped_outside += skipped
        sets.append(windows)
    merged = WindowSet.concat(sets)
    if len(merged) == 0:
        raise NeurOSDataError(
            f"no task windows with labels {list(labels)}", DataContext(spec.key, subject)
        )
    summary.n_windows = len(merged)
    summary.label_counts = merged.label_counts()
    summary.flagged_extreme = merged.n_flagged(WindowFlag.EXTREME_AMPLITUDE)
    summary.flagged_transient = merged.n_flagged(WindowFlag.FILTER_TRANSIENT)
    if summary.skipped_outside:
        log.warning(
            "%s: %d events skipped (epoch outside the recording)",
            summary.person,
            summary.skipped_outside,
        )
    return merged, summary
