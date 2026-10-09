"""D-039's training windows: more windows per trial and more classes than the test uses.

"Use more of what we have": the test window of every trial stays exactly D-030's (cut by
``neuros.data.prepare.subject_epochs``), but training may also see

- **later crops** of the same trial, inside its cue-free span
  (``neuros.data.registry.training_crop_starts``: never earlier, towards the cue), and
- **more classes** (feet, tongue, both hands, rest) from datasets that recorded them.

This module only cuts windows from the preprocessed recordings ``prepare`` produces (from the
same cache). It lives outside ``neuros.data.prepare`` on purpose: that module's source is part
of the recording cache's key (§4.5), and cutting windows doesn't change what is cached.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neuros.config.schemas import DataConfig
from neuros.data import prepare
from neuros.data.recording import person_id
from neuros.data.registry import DatasetSpec, task_epoch
from neuros.data.vocabulary import Vocabulary
from neuros.errors import DataContext, NeurOSConfigError, NeurOSDataError
from neuros.signal.windows import WindowSet, extract_epochs

if TYPE_CHECKING:  # pragma: no cover - typing only
    from neuros.data.cache import RecordingCache

# Tolerance when comparing crop starts (seconds; the registry rounds them to 1e-9).
_START_TOL_S = 1e-6


def subject_training_windows(
    spec: DatasetSpec,
    subject: int,
    vocabulary: Vocabulary,
    config: DataConfig,
    labels: Sequence[str],
    crop_starts_s: Sequence[float],
    *,
    dataset: Any | None = None,
    cache: RecordingCache | None = None,
) -> tuple[WindowSet, NDArray[np.bool_], dict[str, Any]]:
    """One subject's training windows: every crop of every trial with ``labels``.

    ``crop_starts_s`` comes from ``training_crop_starts``: the first is the test window
    (``task_epoch``), the rest lie later in the task. Windows are cut from the same
    preprocessed recordings (and cache) as ``subject_epochs``, so each trial's first crop is
    exactly its test window.

    Returns:
        ``(windows, primary, summary)``: ``primary[i]`` is True for each trial's first crop
        (the one an epoch counts); the summary counts trials, windows, labels and skips.

    Raises:
        NeurOSConfigError: on a label outside the vocabulary, no crop starts, a first start that
            isn't the test window, starts out of order, or a crop past the end of the task.
        NeurOSDataError: see ``prepare_subject``; also if no window survives.
    """
    unknown = sorted(set(labels) - set(vocabulary.labels))
    if unknown or not labels:
        raise NeurOSConfigError(f"labels {unknown or '[]'} are not vocabulary labels")
    epoch = task_epoch(spec, config.signal)
    starts = [float(s) for s in crop_starts_s]
    window_s = config.signal.window_s
    if not starts or abs(starts[0] - epoch.start_s) > _START_TOL_S:
        raise NeurOSConfigError(
            f"{spec.key}: the first crop must be the test window ({epoch.start_s:g} s), "
            f"got {starts[:1]}"
        )
    if any(b <= a for a, b in itertools.pairwise(starts)):
        raise NeurOSConfigError(f"{spec.key}: crop starts must increase, got {starts}")
    if starts[-1] + window_s > spec.task_s + _START_TOL_S:
        raise NeurOSConfigError(
            f"{spec.key}: a crop at {starts[-1]:g} s runs past the task's end ({spec.task_s:g} s)"
        )
    # Through the module, so the recordings are exactly subject_epochs' (and tests can stub it).
    if cache is None:
        recordings = prepare.prepare_subject(spec, subject, vocabulary, config, dataset=dataset)
    else:
        recordings, _ = cache.load_or_prepare(
            spec, subject, vocabulary, config, prepare=prepare.prepare_subject, dataset=dataset
        )
    sets: list[WindowSet] = []
    flags: list[NDArray[np.bool_]] = []
    per_crop = [0] * len(starts)
    skipped = 0
    for recording in recordings:
        for k, start in enumerate(starts):
            windows, outside = extract_epochs(
                recording,
                config.signal,
                config.validation,
                labels=labels,
                epoch_s=(start, start + window_s),
            )
            skipped += outside
            per_crop[k] += len(windows)
            sets.append(windows)
            flags.append(np.full(len(windows), k == 0, dtype=bool))
    merged = WindowSet.concat(sets)
    if len(merged) == 0:
        raise NeurOSDataError(
            f"no training windows with labels {list(labels)}", DataContext(spec.key, subject)
        )
    primary = np.concatenate(flags)
    summary = {
        "person": person_id(spec.key, subject),
        "crop_starts_s": starts,
        "n_trials": int(primary.sum()),
        "n_windows": len(merged),
        "windows_per_crop": per_crop,
        "skipped_outside": skipped,
        "label_counts": merged.label_counts(),
    }
    return merged, primary, summary
