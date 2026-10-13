"""E1: does pooling every dataset beat training on one? (architecture §8.5)

E1 trains EEGNet on the development people of many datasets at once. Three questions, each
answered person by person on people the model never saw:

1. **Pooled vs single-dataset.** The development people are dealt into K folds, stratified by
   dataset (``make_dev_folds``). For each fold, one *pooled* model trains on every trainable
   dataset's people outside the fold, and one *single-dataset* model per dataset trains on
   that dataset's people outside the fold. Both are tested on the fold's people, so every
   person has a pooled and a single-dataset score: a paired Wilcoxon over people (§8.3).
   Go / no-go: if pooling doesn't help, fix harmonisation before adding heads.
2. **Leave one dataset out (LODO).** For each trainable dataset, a model trains on every other
   trainable dataset's development people and is tested on all of its people: the first
   "any setup" numbers. Test-only datasets (D-033: Shin2017A, GuttmannFlury2025) are never
   trained on, so their pooled scores already are leave-one-dataset-out scores.
3. **The cue-only control (D-030).** The pooled recipe again on [cue, cue + 0.5 s] windows: per
   dataset, how much of the class the screen alone gives away. Every number from a window that
   isn't cue-free is reported next to it. On cue-free datasets it measures the cue itself
   (BNCI2014-001: 75.8% for EEGNet in E0's diagnostics).

**One input layout.** Each dataset is mapped to the model's channel layout (C-core by default,
§4.3): its channels in the layout are re-referenced (CAR over those channels) and aligned (EA
per session, on those channels only), then placed in their rows. Rows it doesn't have stay zero,
and a per-window mask records which rows are real.

**Isolation (principle 2).** Only development people are loaded (the caller checks the locked
test set; ``plan_tasks`` checks again that no person is in two roles). Validation people for
early stopping come from each model's own training people. The input scale is the RMS of each
model's training windows. EA uses each session's own windows, without labels, and is the same
in every fold.

**Amplitude guard.** Before a model trains, its training and validation windows are checked
for amplifier glitches (``check_amplitude``): one window far above its person's usual RMS would
set the input scale and the batch-norm statistics on its own (10 Oct). The check raises with
the window's person, dataset and ratio; it never changes data, so every run that passes is
exactly what it was without it.

**Scale.** About 350 development people and 100k windows. The assembled windows are written
once to a work folder as ``.npy`` files, and worker processes map them read-only instead of
receiving gigabytes through pickling. Each model is a task; tasks run in parallel in spawned
workers, on the CPU or on GPUs (``neuros.train.device``), and every finished task is handed to
``on_task`` so the caller can save the result file as it grows.

**More training windows than test windows (D-039).** Optionally, models train on a separate
"train" window set: extra crops later in each trial's cue-free span, and extra classes (feet,
tongue, both hands, rest) from the datasets that have them. Validation and test windows are
unchanged, and so is their alignment: the training windows take the EA references of the same
session's test windows. Without that set, every step is exactly E1's.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import multiprocessing
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neuros.align.ea import align_groups, align_with_references
from neuros.config.experiment import (
    Alignment,
    AlignmentUnit,
    BackboneConfig,
    EEGNetConfig,
    PooledCondition,
    Reference,
    TrainConfig,
)
from neuros.errors import DataContext, NeurOSConfigError, NeurOSDataError
from neuros.eval.metrics import paired_wilcoxon, score_person, summarize
from neuros.eval.splits import Fold, make_dev_folds, seeded_rng
from neuros.signal.reference import common_average_reference
from neuros.signal.windows import WindowSet

if TYPE_CHECKING:  # finetune imports this module; only its type is needed here
    from neuros.eval.finetune import FinetuneSettings

Progress = Callable[[str], None]
# "train": D-039's training windows (crops, extra classes); validation and test stay on "task".
DataKind = Literal["task", "cue_only", "train"]

# Windows per chunk when computing an RMS over a memory-mapped array (bounded memory).
_RMS_CHUNK = 4096

# D-039's artifact rule: a newly cut training window whose RMS is more than this many times
# its person's median window RMS is dropped (2,500x the power: not brain signal). Measured
# 10 Oct on every D-039 training set: genuine windows reach 10.6x (a Stieger2021 rest
# window), the test windows 7.7x; two Yang2025 subject-18 crops reached 7,200x and 7,650x
# (an amplifier glitch after the test window) and, holding 99.8 % of the set's power, set the
# input scale and the batch-norm statistics on their own: 4 of 5 models answered one class.
# The same limit is the general guard every model's training and validation windows pass
# before training (``check_amplitude``): there a window above it raises instead of dropping.
ARTIFACT_RMS_FACTOR = 50.0

# The fewest of the layout's channels a dataset must have to be placed in it: the common
# average and EA's covariances need at least two. ``to_layout`` enforces it on the windows;
# ``layout_shortfalls`` checks the registry's channels before anything is loaded.
MIN_LAYOUT_CHANNELS = 2


# ---------------------------------------------------------------------- assembled windows
@dataclass(frozen=True, eq=False)
class PooledWindows:
    """Windows of many datasets in one channel layout, ready for training.

    Attributes:
        x: ``[n, rows, samples]`` float32 µV; rows follow ``layout``; zero where ``present`` is
            False. May be a read-only memory map.
        present: ``[n, rows]``: True where the row holds real data (not zero-fill).
        labels, people, datasets: one entry per window (people are ``dataset:subject``).
        flags: per-window quality flags (``neuros.signal.windows.WindowFlag``), for reports.
        layout: the channel of each row.
        sfreq: samples per second.
        primary: optional, True for one window per trial and False for a trial's extra
            training crops (D-039: an epoch counts primary windows). None = all primary.
    """

    x: NDArray[np.float32]
    present: NDArray[np.bool_]
    labels: NDArray[np.str_]
    people: NDArray[np.str_]
    datasets: NDArray[np.str_]
    flags: NDArray[np.uint8]
    layout: tuple[str, ...]
    sfreq: float
    primary: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        if self.x.dtype != np.float32 or self.x.ndim != 3:
            raise NeurOSDataError(f"x must be [n, rows, samples] float32, got {self.x.dtype}")
        n, rows, samples = self.x.shape
        if n == 0 or samples == 0:
            raise NeurOSDataError("no windows")
        if rows != len(self.layout):
            raise NeurOSDataError(f"{rows} rows but a layout of {len(self.layout)} channels")
        if self.present.shape != (n, rows) or self.present.dtype != np.bool_:
            raise NeurOSDataError(f"present must be a [{n}, {rows}] bool array")
        for name in ("labels", "people", "datasets", "flags"):
            if getattr(self, name).shape != (n,):
                raise NeurOSDataError(f"{name} must hold one entry per window ({n})")
        if not (math.isfinite(self.sfreq) and self.sfreq > 0):
            raise NeurOSDataError(f"sfreq must be positive, got {self.sfreq}")
        if self.primary is not None:
            if self.primary.shape != (n,) or self.primary.dtype != np.bool_:
                raise NeurOSDataError(f"primary must be a [{n}] bool array")
            if not self.primary.any():
                raise NeurOSDataError("no primary window: every trial needs one")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.x.shape[2])

    @classmethod
    def concat(cls, parts: Sequence[PooledWindows]) -> PooledWindows:
        """Stack window sets with the same layout, rate and window length, in order.

        ``primary`` is kept when any part has it (a part without it counts every window).

        Raises:
            NeurOSDataError: on no parts, or parts that differ in layout, rate or length.
        """
        if not parts:
            raise NeurOSDataError("no pooled window sets to concatenate")
        first = parts[0]
        for part in parts[1:]:
            if (part.layout, part.sfreq, part.n_samples) != (
                first.layout,
                first.sfreq,
                first.n_samples,
            ):
                raise NeurOSDataError(
                    "can't concatenate pooled windows with different layouts, rates or lengths"
                )
        primary = None
        if any(part.primary is not None for part in parts):
            primary = np.concatenate(
                [
                    part.primary if part.primary is not None else np.ones(len(part), dtype=bool)
                    for part in parts
                ]
            )
        return cls(
            x=np.concatenate([part.x for part in parts]),
            present=np.concatenate([part.present for part in parts]),
            labels=np.concatenate([part.labels for part in parts]),
            people=np.concatenate([part.people for part in parts]),
            datasets=np.concatenate([part.datasets for part in parts]),
            flags=np.concatenate([part.flags for part in parts]),
            layout=first.layout,
            sfreq=first.sfreq,
            primary=primary,
        )

    def rows_of(self, people: Collection[str]) -> NDArray[np.int64]:
        """Indices of the windows of ``people``, in storage order."""
        return np.flatnonzero(np.isin(self.people, np.asarray(list(people), dtype=str)))

    def take(self, rows: NDArray[np.int64]) -> PooledWindows:
        """The given windows (every per-window array, ``primary`` included), in that order.

        Raises:
            NeurOSDataError: on rows that aren't a 1-D integer array within ``[0, len)``, or
                that leave no window (or, with ``primary``, no primary window).
        """
        rows = np.asarray(rows)
        if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
            raise NeurOSDataError("rows must be a 1-D integer array")
        if rows.size and (int(np.min(rows)) < 0 or int(np.max(rows)) >= len(self)):
            raise NeurOSDataError(f"rows must lie in [0, {len(self)})")
        return PooledWindows(
            x=self.x[rows],
            present=self.present[rows],
            labels=self.labels[rows],
            people=self.people[rows],
            datasets=self.datasets[rows],
            flags=self.flags[rows],
            layout=self.layout,
            sfreq=self.sfreq,
            primary=None if self.primary is None else self.primary[rows],
        )

    def people_by_dataset(self) -> dict[str, tuple[str, ...]]:
        """Each dataset's people, in first-appearance order."""
        result: dict[str, list[str]] = {}
        pairs = zip(self.datasets.tolist(), self.people.tolist(), strict=True)
        for dataset, person in dict.fromkeys(pairs):
            result.setdefault(dataset, []).append(person)
        return {dataset: tuple(people) for dataset, people in result.items()}

    # ------------------------------------------------------------------ disk
    _ARRAYS = ("x", "present", "labels", "people", "datasets", "flags")
    _OPTIONAL_ARRAYS = ("primary",)  # saved only when set; meta.json lists what was saved

    def save(self, folder: Path) -> None:
        """Write every array as ``.npy`` plus ``meta.json`` with SHA-256 digests.

        Raises:
            NeurOSDataError: if ``folder`` already holds a saved set (never overwritten).
        """
        meta_file = folder / "meta.json"
        if meta_file.exists():
            raise NeurOSDataError(f"{folder} already holds pooled windows")
        folder.mkdir(parents=True, exist_ok=True)
        digests: dict[str, str] = {}
        names = self._ARRAYS + tuple(
            name for name in self._OPTIONAL_ARRAYS if getattr(self, name) is not None
        )
        for name in names:
            path = folder / f"{name}.npy"
            np.save(path, np.ascontiguousarray(getattr(self, name)), allow_pickle=False)
            digests[name] = _sha256(path)
        meta = {
            "layout": list(self.layout),
            "sfreq": self.sfreq,
            "n_windows": len(self),
            "n_samples": self.n_samples,
            "sha256": digests,
        }
        meta_file.write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, folder: Path, *, verify: bool = False) -> PooledWindows:
        """Map a saved set read-only (``x`` stays on disk until rows are indexed).

        Args:
            verify: re-hash every file against ``meta.json`` (seconds per GB; do it once, in
                the parent, not in every worker).

        Raises:
            NeurOSDataError: if the folder is incomplete, a digest differs, or shapes disagree.
        """
        meta_file = folder / "meta.json"
        if not meta_file.is_file():
            raise NeurOSDataError(f"{folder}: no pooled windows (meta.json missing)")
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        arrays: dict[str, Any] = {}
        # Optional arrays are part of the set exactly when meta.json has their digest.
        optional = tuple(name for name in cls._OPTIONAL_ARRAYS if name in meta["sha256"])
        for name in cls._ARRAYS + optional:
            path = folder / f"{name}.npy"
            if not path.is_file():
                raise NeurOSDataError(f"{folder}: {name}.npy is missing")
            if verify and _sha256(path) != meta["sha256"][name]:
                raise NeurOSDataError(f"{path}: sha256 mismatch")
            arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
        windows = cls(layout=tuple(meta["layout"]), sfreq=float(meta["sfreq"]), **arrays)
        if len(windows) != int(meta["n_windows"]) or windows.n_samples != int(meta["n_samples"]):
            raise NeurOSDataError(f"{folder}: arrays don't match meta.json")
        return windows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def layout_shortfalls(
    channels_by_dataset: Mapping[str, Sequence[str]], layout: Sequence[str]
) -> dict[str, list[str]]:
    """Datasets that can't be placed in ``layout``, each with the layout channels it does have.

    A dataset needs at least ``MIN_LAYOUT_CHANNELS`` of the layout's channels (the rule
    ``to_layout`` applies to its windows). Checked from the registry before any loading, so a
    misfit costs seconds rather than hours of data preparation (E1, 6 Oct: Shin2017A shares only
    Cz with the motor strip, Q27).

    Args:
        channels_by_dataset: dataset key → its usable EEG channels (canonical names).
        layout: the model's input channels.

    Returns:
        dataset → the layout channels it has (in layout order), for every dataset below the
        minimum; empty when every dataset fits.

    Raises:
        NeurOSConfigError: if the layout is empty or repeats a channel.
    """
    if not layout:
        raise NeurOSConfigError("the layout has no channels")
    if len(set(layout)) != len(layout):
        raise NeurOSConfigError(f"the layout repeats channels: {list(layout)}")
    shortfalls: dict[str, list[str]] = {}
    for dataset, channels in channels_by_dataset.items():
        available = set(channels)
        have = [ch for ch in layout if ch in available]
        if len(have) < MIN_LAYOUT_CHANNELS:
            shortfalls[dataset] = have
    return shortfalls


def derive_rows(
    data: NDArray[np.floating],
    ch_names: Sequence[str],
    derived: Mapping[str, Sequence[str]],
    *,
    axis: int = -2,
) -> tuple[NDArray[np.float32], tuple[str, ...]]:
    """Array form of ``derive_channels``: ``data`` has channels on ``axis`` (windows
    ``[n, ch, T]`` or a continuous recording ``[ch, T]``). Returns float32 data and names.

    Raises:
        NeurOSDataError: if a source channel is missing, feeds two derived channels, or a
            derived channel already exists; or if ``ch_names`` doesn't match ``data``.
    """
    names = tuple(ch_names)
    if data.shape[axis] != len(names):
        raise NeurOSDataError(f"{len(names)} channel names for {data.shape[axis]} channels")
    if not derived:
        return np.asarray(data, dtype=np.float32), names
    sources = [source for group in derived.values() for source in group]
    if len(set(sources)) != len(sources):
        raise NeurOSDataError(f"a source channel feeds more than one derived channel: {sources}")
    missing = sorted(set(sources) - set(names))
    if missing:
        raise NeurOSDataError(f"derived channels need {missing}, which these windows lack")
    clash = sorted(set(derived) & set(names))
    if clash:
        raise NeurOSDataError(f"derived channels {clash} are already in the window set")
    used = set(sources)
    keep = [index for index, name in enumerate(names) if name not in used]
    means = [
        np.take(data, [names.index(s) for s in group], axis=axis).mean(axis=axis, dtype=np.float64)
        for group in derived.values()
    ]
    out = np.concatenate(
        [np.take(data, keep, axis=axis), np.stack(means, axis=axis).astype(np.float32)],
        axis=axis,
    )
    return np.ascontiguousarray(out, dtype=np.float32), tuple(names[i] for i in keep) + tuple(
        derived
    )


def derive_channels(windows: WindowSet, derived: Mapping[str, Sequence[str]]) -> WindowSet:
    """Replace each group of source channels with its mean, named as the derived channel.

    The registry's test-only montage profile (``DatasetSpec.derived_channels``, D-037): e.g.
    Shin2017A's C3 is the mean of the four 10-5 half positions around it. Kept channels stay in
    their order; derived ones follow in mapping order. Windows are already filtered, and the
    mean is linear, so averaging here equals averaging before filtering.

    Raises:
        NeurOSDataError: if a source channel is missing, feeds two derived channels, or a
            derived channel is already in the window set.
    """
    if not derived:
        return windows
    data, names = derive_rows(windows.data, windows.ch_names, derived)
    return dataclasses.replace(windows, data=data, ch_names=names)


def to_layout(
    windows: WindowSet,
    layout: Sequence[str],
    *,
    reference: Reference,
    alignment: Alignment,
    unit: AlignmentUnit,
    references_from: WindowSet | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.bool_], dict[str, dict[str, float | int]]]:
    """One dataset's windows in the shared layout: CAR and EA on the channels it has, zeros
    elsewhere (§4.3).

    ``references_from``: compute each group's EA reference from these windows instead (same
    dataset and channels; prepared the same way) and apply it to ``windows``. D-039's training
    windows take the test windows' references, so test alignment stays exactly E1's.

    Returns:
        ``(x [n, len(layout), samples], present [n, len(layout)], EA reference summaries)``.

    Raises:
        NeurOSDataError: if the dataset has fewer than ``MIN_LAYOUT_CHANNELS`` of the layout's
            channels (CAR and covariances need at least two), if ``references_from`` has other
            channels, or if a group of ``windows`` has no reference there.
        NeurOSConfigError: on an unknown reference or alignment.
    """
    have = [ch for ch in layout if ch in windows.ch_names]
    if len(have) < MIN_LAYOUT_CHANNELS:
        raise NeurOSDataError(
            f"{sorted(set(windows.datasets))}: only {have} of the layout's channels; at least "
            f"{MIN_LAYOUT_CHANNELS} are needed"
        )
    if references_from is not None and set(references_from.ch_names) != set(windows.ch_names):
        raise NeurOSDataError(
            "references_from must have the same channels as the windows it aligns: "
            f"{sorted(references_from.ch_names)} vs {sorted(windows.ch_names)}"
        )

    def referenced(source: WindowSet) -> WindowSet:
        picked = source.pick_channels(have)  # layout order
        if reference == "car":
            return picked.with_data(common_average_reference(picked.data))
        if reference != "as_recorded":
            raise NeurOSConfigError(f"unknown reference {reference!r} (as_recorded, car)")
        return picked

    subset = referenced(windows)
    summaries: dict[str, dict[str, float | int]] = {}
    if alignment == "ea":
        if references_from is None:
            subset, references = align_groups(subset, unit=unit)
        else:
            _, references = align_groups(referenced(references_from), unit=unit)
            subset = align_with_references(subset, references, unit=unit)
        summaries = {group: ref.summary() for group, ref in references.items()}
    elif alignment != "none":
        raise NeurOSConfigError(f"unknown alignment {alignment!r} (none, ea)")
    rows = [list(layout).index(ch) for ch in have]
    x = np.zeros((len(subset), len(layout), subset.n_samples), dtype=np.float32)
    x[:, rows, :] = subset.data
    present = np.zeros((len(subset), len(layout)), dtype=bool)
    present[:, rows] = True
    return x, present, summaries


def assemble(
    per_dataset: Mapping[str, WindowSet],
    layout: Sequence[str],
    *,
    reference: Reference,
    alignment: Alignment,
    unit: AlignmentUnit,
    derived: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    references_from: Mapping[str, WindowSet] | None = None,
    primary: Mapping[str, NDArray[np.bool_]] | None = None,
) -> tuple[PooledWindows, dict[str, dict[str, Any]]]:
    """Map every dataset into ``layout`` and stack them (datasets in the given order).

    ``derived`` gives, per dataset, its test-only derived channels (registry
    ``derived_channels``, D-037), applied before the dataset is placed in the layout.

    D-039's training set: ``references_from`` gives, for every dataset, the windows whose EA
    references align it (the test windows); ``primary`` marks, per dataset and window, the one
    window per trial that counts towards an epoch (the rest are extra crops).

    Returns:
        The pooled windows and, per dataset, a report: channels present and missing, windows,
        people and the EA references.

    Raises:
        NeurOSDataError: on an empty input, mixed sampling rates or window lengths, a window
            set holding another dataset's windows, non-finite values after alignment, or a
            ``references_from``/``primary`` that doesn't cover exactly the datasets given.
    """
    if not per_dataset:
        raise NeurOSDataError("no datasets to assemble")
    parts: list[tuple[NDArray[np.float32], NDArray[np.bool_], WindowSet]] = []
    reports: dict[str, dict[str, Any]] = {}
    rates = {windows.sfreq for windows in per_dataset.values()}
    lengths = {windows.n_samples for windows in per_dataset.values()}
    if len(rates) != 1 or len(lengths) != 1:
        raise NeurOSDataError(f"datasets differ in rate {rates} or window length {lengths}")
    unknown = sorted(set(derived or {}) - set(per_dataset))
    if unknown:
        raise NeurOSDataError(f"derived channels given for datasets not assembled: {unknown}")
    for name, given in (("references_from", references_from), ("primary", primary)):
        if given is not None and set(given) != set(per_dataset):
            raise NeurOSDataError(
                f"{name} must cover exactly the datasets assembled: {sorted(given)} vs "
                f"{sorted(per_dataset)}"
            )
    for dataset, windows in per_dataset.items():
        if set(windows.datasets) != {dataset}:
            raise NeurOSDataError(
                f"windows given for {dataset} hold {sorted(set(windows.datasets))}"
            )
        if primary is not None and np.asarray(primary[dataset]).shape != (len(windows),):
            raise NeurOSDataError(f"{dataset}: primary must hold one flag per window")
        mapping = (derived or {}).get(dataset, {})
        windows = derive_channels(windows, mapping)
        source = None
        if references_from is not None:
            source = derive_channels(references_from[dataset], mapping)
            if set(source.datasets) != {dataset}:
                raise NeurOSDataError(f"references_from for {dataset} holds other datasets")
        x, present, references = to_layout(
            windows,
            layout,
            reference=reference,
            alignment=alignment,
            unit=unit,
            references_from=source,
        )
        if not np.isfinite(x).all():
            raise NeurOSDataError(f"{dataset}: non-finite values after alignment")
        parts.append((x, present, windows))
        reports[dataset] = {
            "channels_present": [ch for ch in layout if ch in windows.ch_names],
            "channels_missing": [ch for ch in layout if ch not in windows.ch_names],
            "derived_channels": {target: list(group) for target, group in mapping.items()},
            "n_windows": len(windows),
            "n_people": len(windows.people),
            "label_counts": windows.label_counts(),
            "alignment_references": references,
        }
    pooled = PooledWindows(
        x=np.concatenate([x for x, _, _ in parts]),
        present=np.concatenate([p for _, p, _ in parts]),
        labels=np.array([label for _, _, w in parts for label in w.labels], dtype=str),
        people=np.array([person for _, _, w in parts for person in w.person_ids], dtype=str),
        datasets=np.array([d for _, _, w in parts for d in w.datasets], dtype=str),
        flags=np.concatenate([w.flags for _, _, w in parts]).astype(np.uint8),
        layout=tuple(layout),
        sfreq=float(next(iter(rates))),
        primary=(
            None
            if primary is None
            else np.concatenate([np.asarray(primary[d], dtype=bool) for d in per_dataset])
        ),
    )
    if primary is not None:
        for dataset in per_dataset:
            reports[dataset]["n_trials"] = int(np.count_nonzero(primary[dataset]))
    return pooled, reports


def _check_factor(factor: float) -> None:
    """The artifact limit must be a finite multiple above 1 (at 1, half of every person's
    windows would sit above their own median)."""
    if not (math.isfinite(factor) and factor > 1):
        raise NeurOSConfigError(f"the artifact factor must be a finite number above 1: {factor}")


def _window_rms(x: NDArray[np.float32], rows: NDArray[np.int64]) -> NDArray[np.float64]:
    """Each given window's RMS over its whole ``[rows, samples]`` block (float64, chunked:
    ``x`` may be a memory map, and only ``_RMS_CHUNK`` windows are held at once).

    The zero-filled layout rows count too. That is harmless for a ratio to the person's
    median: a person's windows all come from one dataset, so they share one set of present
    rows. A NaN or Inf sample makes the window's RMS non-finite.
    """
    rms = np.empty(rows.size, dtype=np.float64)
    for start in range(0, rows.size, _RMS_CHUNK):
        block = np.asarray(x[rows[start : start + _RMS_CHUNK]], dtype=np.float64)
        rms[start : start + block.shape[0]] = np.sqrt(np.mean(block * block, axis=(1, 2)))
    return rms


def screen_amplitude(
    windows: PooledWindows, *, factor: float = ARTIFACT_RMS_FACTOR
) -> tuple[PooledWindows, NDArray[np.bool_]]:
    """Drop windows whose RMS is more than ``factor`` times their person's median window RMS.

    D-039's artifact rule for newly cut training windows (see ``ARTIFACT_RMS_FACTOR``). The
    median is each person's own, over their windows in ``windows``, so the rule holds whatever
    the alignment or units; a non-finite window is dropped too. Windows E1/E2 already trained
    on (the test windows) never go through it: ``check_amplitude`` only checks them, with the
    same measure, before every model trains.

    Returns:
        ``(the kept windows, a [n] bool mask of the dropped ones)``.

    Raises:
        NeurOSConfigError: on a factor that isn't a finite number above 1.
        NeurOSDataError: if a person's median RMS over their finite windows isn't positive
            (no signal), or every window (or every primary window) would be dropped.
    """
    _check_factor(factor)
    rms = _window_rms(windows.x, np.arange(len(windows), dtype=np.int64))
    finite = np.isfinite(rms)
    dropped = ~finite
    for person in dict.fromkeys(windows.people.tolist()):
        mine = windows.people == person
        if not (mine & finite).any():
            continue  # every window non-finite: all already dropped
        median = float(np.median(rms[mine & finite]))
        if not (math.isfinite(median) and median > 0):
            raise NeurOSDataError(f"{person}: median window RMS is {median} (no signal)")
        dropped |= mine & finite & (rms > factor * median)
    if dropped.all():
        raise NeurOSDataError("the artifact rule would drop every window")
    if not dropped.any():
        return windows, dropped
    return windows.take(np.flatnonzero(~dropped)), dropped


# How many offending windows an amplitude error lists (the count covers them all).
_LISTED_ARTIFACTS = 5


def check_amplitude(
    windows: PooledWindows,
    rows: NDArray[np.int64],
    *,
    role: str,
    factor: float = ARTIFACT_RMS_FACTOR,
) -> float:
    """Raise if any of the given windows looks like an amplifier glitch; never change them.

    The general guard behind D-039's artifact rule (the 10 Oct incident, see
    ``ARTIFACT_RMS_FACTOR``). One window at thousands of times its person's usual RMS sets a
    model's input scale (``_rms``) and its batch-norm statistics (the trainer's equal-weight
    pass over every training window) on its own, and the run still finishes, with healthy
    best-epoch validation numbers. ``run_task`` therefore calls this on every model's training
    and validation windows before anything is computed from them.

    The measure is ``screen_amplitude``'s: a window's RMS over its whole ``[rows, samples]``
    block, divided by the median window RMS of its person *among the given windows* (so the
    limit holds whatever the units, alignment or input scale). Unlike the screen it raises
    instead of dropping: a set that passes trains exactly as it did before the guard existed,
    and a set that fails is fixed where it's built (``screen_amplitude`` for newly cut windows,
    the registry's ``excluded_subjects`` for a person). Measured 10 Oct: E1/E2's task windows
    reach 7.7x, D-039's screened training windows 10.6x.

    What a per-person median can't see: a person most of whose windows are glitched (the median
    is then the glitch's own), or a whole person or dataset at the wrong scale (the units checks
    catch that upstream, D-035/D-036).

    Args:
        windows: the window set (``x`` may be a read-only memory map; it's read in chunks).
        rows: indices into ``windows`` of the windows to check, e.g. one model's training
            people. Each person's median is taken over their windows among these.
        role: what the windows are, for the message (e.g. ``"pooled/fold2: training windows"``).
        factor: the limit, as a multiple of the person's median (``ARTIFACT_RMS_FACTOR``).

    Returns:
        The largest window-to-median ratio among the given windows (at most ``factor``).

    Raises:
        NeurOSConfigError: on a factor that isn't a finite number above 1.
        NeurOSDataError: on rows that aren't a non-empty 1-D integer array within
            ``[0, len(windows))``; or, naming the window, its person, dataset and ratio, on a
            window with NaN/Inf samples, a person whose median window RMS is 0 (no signal), or
            any window above ``factor`` times its person's median.
    """
    _check_factor(factor)
    rows = np.asarray(rows)
    if rows.ndim != 1 or rows.size == 0 or not np.issubdtype(rows.dtype, np.integer):
        raise NeurOSDataError(f"{role}: rows must be a non-empty 1-D integer array")
    if int(np.min(rows)) < 0 or int(np.max(rows)) >= len(windows):
        raise NeurOSDataError(f"{role}: rows must lie in [0, {len(windows)})")
    rms = _window_rms(windows.x, rows)
    people = windows.people[rows]
    datasets = windows.datasets[rows]

    def where(i: int) -> str:
        """The i-th checked window, as people read it (its row in ``windows``)."""
        return f"window {int(rows[i])} of {people[i]} ({datasets[i]})"

    def context(i: int) -> DataContext:
        """The data context of the i-th checked window (people are ``dataset:subject``)."""
        return DataContext(dataset=str(datasets[i]), subject=str(people[i]).split(":", 1)[-1])

    broken = np.flatnonzero(~np.isfinite(rms))
    if broken.size:
        first = int(broken[0])
        raise NeurOSDataError(
            f"{role}: {broken.size} window(s) hold NaN/Inf samples, the first {where(first)}",
            context(first),
        )

    # Each person's median over their checked windows: group by person once (sorting), rather
    # than one pass over every window per person.
    names, inverse = np.unique(people, return_inverse=True)
    inverse = inverse.reshape(-1)  # numpy 2 keeps the input's shape; it's 1-D here anyway
    order = np.argsort(inverse, kind="stable")
    bounds = np.cumsum(np.bincount(inverse, minlength=names.size))[:-1]
    medians = np.array([float(np.median(group)) for group in np.split(rms[order], bounds)])
    silent = np.flatnonzero(~(medians > 0))
    if silent.size:
        first = int(np.flatnonzero(inverse == silent[0])[0])  # one of that person's windows
        raise NeurOSDataError(
            f"{role}: {names[silent[0]]}'s median window RMS is {medians[silent[0]]} (no signal)",
            context(first),
        )

    ratio = rms / medians[inverse]
    worst = int(np.argmax(ratio))
    if ratio[worst] > factor:
        over = np.flatnonzero(ratio > factor)
        listed = over[np.argsort(ratio[over], kind="stable")[::-1][:_LISTED_ARTIFACTS]]
        examples = "; ".join(f"{where(int(i))}: {ratio[i]:,.1f}x" for i in listed)
        raise NeurOSDataError(
            f"{role}: {over.size} window(s) of {np.unique(people[over]).size} person(s) have "
            f"an RMS above {factor:g}x their person's median window RMS ({examples}). Probably "
            "amplifier glitches: one such window can set the input scale and the batch-norm "
            "statistics on its own (10 Oct incident). Nothing was trained; drop them where the "
            "set is built (screen_amplitude) or exclude the person.",
            context(worst),
        )
    return float(ratio[worst])


# ---------------------------------------------------------------------- the plan
@dataclass(frozen=True)
class ModelTask:
    """One model to train and test: who trains, validates and is tested, on which windows."""

    name: str  # e.g. "pooled/fold2", "single/Cho2017/fold2", "lodo/Cho2017", "cue_only/fold2"
    condition: PooledCondition
    data: DataKind  # validation and test windows (and training windows, unless train_data)
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    seed: int
    train_data: DataKind | None = None  # D-039: train on another window set

    def __post_init__(self) -> None:
        Fold(index=0, test=self.test, validation=self.validation, train=self.train)  # checks
        if self.train_data == self.data:
            raise NeurOSConfigError(f"{self.name}: train_data repeats data ({self.data!r})")


def split_validation(
    people: Sequence[str], fraction: float, *, seed: int, tag: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Take ``round(fraction × n)`` validation people from each dataset among ``people``.

    Stratified by dataset (the prefix of ``dataset:subject``) and drawn with an RNG that depends
    only on (seed, tag, dataset), so a task's draw never changes when other tasks change. A
    dataset keeps at least one training person.

    Returns:
        ``(training people, validation people)``, each in the input order.
    """
    if not 0 <= fraction < 1:
        raise NeurOSConfigError(f"validation fraction must be in [0, 1), got {fraction}")
    if fraction == 0:
        return tuple(people), ()
    by_dataset: dict[str, list[str]] = {}
    for person in people:
        by_dataset.setdefault(person.split(":", 1)[0], []).append(person)
    chosen: set[str] = set()
    for dataset, members in by_dataset.items():
        count = min(round(fraction * len(members)), len(members) - 1)
        if count <= 0:
            continue
        rng = seeded_rng(seed, f"validation:{tag}:{dataset}")
        picks = rng.choice(len(members), size=count, replace=False)
        chosen.update(members[i] for i in picks)
    return (
        tuple(p for p in people if p not in chosen),
        tuple(p for p in people if p in chosen),
    )


def plan_tasks(
    people: Mapping[str, Sequence[str]],
    trainable: Sequence[str],
    *,
    conditions: Collection[PooledCondition],
    folds: int,
    validation_fraction: float,
    seed: int,
    train_data: DataKind | None = None,
) -> list[ModelTask]:
    """Every model E1 trains (see the module docstring), in a fixed order.

    Args:
        people: each dataset's development people (``dataset:subject``), test-only included.
        trainable: the datasets whose people may train (the rest are only tested).
        train_data: the window set the pooled, single-dataset and LODO models train on, when it
            isn't the task windows they're tested on (D-039). The cue-only control always trains
            on its own windows.

    Raises:
        NeurOSConfigError: on an unknown dataset in ``trainable``, fewer than 2 trainable
            datasets for LODO, or a dataset with fewer people than folds.
    """
    unknown = sorted(set(trainable) - set(people))
    if unknown:
        raise NeurOSConfigError(f"trainable datasets without people: {unknown}")
    if not trainable:
        raise NeurOSConfigError("E1 needs at least one trainable dataset")
    fold_sets = make_dev_folds(
        {d: [int(p.split(":", 1)[1]) for p in members] for d, members in people.items()},
        n_folds=folds,
        seed=seed,
    )
    trainable_people = [p for d in trainable for p in people[d]]
    tasks: list[ModelTask] = []

    def add(
        name: str, condition: PooledCondition, data: DataKind, train: list[str], test: list[str]
    ) -> None:
        if not train or not test:
            return  # e.g. a single-dataset fold of a dataset with no test people in that fold
        kept, validation = split_validation(train, validation_fraction, seed=seed, tag=name)
        trains_on = train_data if data == "task" else None
        tasks.append(
            ModelTask(name, condition, data, kept, validation, tuple(test), seed, trains_on)
        )

    for k, fold in enumerate(fold_sets):
        outside = [p for p in trainable_people if p not in fold]
        everyone_in_fold = [p for d in people for p in people[d] if p in fold]
        if "pooled" in conditions:
            add(f"pooled/fold{k}", "pooled", "task", outside, everyone_in_fold)
        if "cue_only" in conditions:
            add(f"cue_only/fold{k}", "cue_only", "cue_only", outside, everyone_in_fold)
        if "single" in conditions:
            for dataset in trainable:
                mine_out = [p for p in people[dataset] if p not in fold]
                mine_in = [p for p in people[dataset] if p in fold]
                add(f"single/{dataset}/fold{k}", "single", "task", mine_out, mine_in)
    if "lodo" in conditions:
        if len(trainable) < 2:
            raise NeurOSConfigError("leave-one-dataset-out needs at least 2 trainable datasets")
        for dataset in trainable:
            others = [p for p in trainable_people if not p.startswith(f"{dataset}:")]
            add(f"lodo/{dataset}", "lodo", "task", others, list(people[dataset]))
    return tasks


# ---------------------------------------------------------------------- running one task
@dataclass(frozen=True)
class TaskSettings:
    """What every task shares (picklable, sent to each worker once)."""

    classes: tuple[str, ...]  # the labels scored in validation and test (logits 0..k-1)
    model: EEGNetConfig
    train: TrainConfig
    device: str
    deterministic: bool
    threads: int
    # Extra window sets (data kinds) that models of ``extra_conditions`` also score their test
    # people on, after training: E2's warm-up sets (``neuros.eval.warmup``).
    extra_kinds: tuple[str, ...] = ()
    extra_conditions: tuple[str, ...] = ("pooled",)
    # D-039: the head's classes for models trained on the "train" windows: ``classes`` first,
    # then the extra classes. Empty = ``classes``.
    train_classes: tuple[str, ...] = ()
    # Q30: fine-tune each pooled model's test people on their first labelled trials
    # (``neuros.eval.finetune``); None = no fine-tuning.
    finetune: FinetuneSettings | None = None
    # A6: a pretrained encoder instead of EEGNet (``neuros.models.pretrained``); None = EEGNet.
    backbone: BackboneConfig | None = None

    def __post_init__(self) -> None:
        if self.train_classes and self.train_classes[: len(self.classes)] != self.classes:
            raise NeurOSConfigError(
                f"train_classes {list(self.train_classes)} must start with the scored classes "
                f"{list(self.classes)}, so their logits keep their indices"
            )
        if len(set(self.train_classes)) != len(self.train_classes):
            raise NeurOSConfigError(f"train_classes has duplicates: {list(self.train_classes)}")

    def head_classes(self, task: ModelTask) -> tuple[str, ...]:
        """The classes of ``task``'s model: the extra classes only when it trains on them."""
        if task.train_data is not None and self.train_classes:
            return self.train_classes
        return self.classes


def _rms(x: NDArray[np.float32], rows: NDArray[np.int64]) -> float:
    """RMS over the given windows (float64 accumulation, chunked: x may be a memory map)."""
    total, count = 0.0, 0
    for start in range(0, rows.size, _RMS_CHUNK):
        block = np.asarray(x[rows[start : start + _RMS_CHUNK]], dtype=np.float64)
        total += float(np.sum(block * block))
        count += block.size
    rms = math.sqrt(total / max(count, 1))
    if not (math.isfinite(rms) and rms > 0):
        raise NeurOSDataError(f"input RMS is {rms}: the training windows carry no signal")
    return rms


def _gather(x: NDArray[np.float32], rows: NDArray[np.int64]) -> NDArray[np.float32]:
    """The given rows of ``x`` (often a read-only memory map) as one private, writable float32
    array, read in chunks.

    ``np.array(x[rows])`` holds the rows twice at its peak (the fancy-indexed copy, then the
    array's own): with four workers each copying a 3-GB training set at once, that doubling
    ran the 30-GB machine out of memory (9 Oct). The values are identical either way.
    """
    out = np.empty((rows.size, *x.shape[1:]), dtype=np.float32)
    for start in range(0, rows.size, _RMS_CHUNK):
        chunk = rows[start : start + _RMS_CHUNK]
        out[start : start + chunk.size] = x[chunk]
    return out


def _predict(
    model: Any,
    x: Any,
    settings: TaskSettings,
    head: Sequence[str],
    predict_logits: Callable[..., Any],
) -> NDArray[np.str_]:
    """The scored class (one of ``settings.classes``) for every window in ``x``.

    A head with extra classes (D-039) is read on the scored classes' logits only: a left/right
    test window is never answered "feet". With no extra classes this is the plain argmax.
    """
    logits = predict_logits(model, x, settings.train.eval_batch_size)
    if tuple(head) != settings.classes:
        logits = logits[:, : len(settings.classes)]  # the scored classes come first
    return np.array([settings.classes[i] for i in logits.argmax(dim=1).tolist()], dtype=str)


def _score_people(
    model: Any,
    windows: PooledWindows,
    people: Sequence[str],
    scale: float,
    settings: TaskSettings,
    predict_logits: Callable[..., Any],
    head: Sequence[str],
) -> list[dict[str, Any]]:
    """Score the trained ``model`` on the given people's windows in ``windows`` (those present
    there); the same input scale as training. Empty if none of the people are present."""
    import torch

    rows = windows.rows_of(people)
    if rows.size == 0:
        return []
    # The model checks the window shape itself (check_window_batch).
    x = _gather(windows.x, rows)
    x /= np.float32(scale)
    truth = windows.labels[rows]
    unknown = sorted(set(truth.tolist()) - set(settings.classes))
    if unknown:
        raise NeurOSDataError(f"extra windows carry labels {unknown} that aren't scored classes")
    predicted = _predict(model, torch.from_numpy(x), settings, head, predict_logits)
    person_of = windows.people[rows]
    flagged = windows.flags[rows] != 0
    scores: list[dict[str, Any]] = []
    for person in dict.fromkeys(person_of.tolist()):
        mine = person_of == person
        score = score_person(
            person,
            truth[mine].tolist(),
            predicted[mine].tolist(),
            n_flagged=int(flagged[mine].sum()),
        )
        scores.append(score.to_dict() | {"balanced_accuracy_raw": score.balanced_accuracy})
    return scores


def _dataset_classes(windows: PooledWindows) -> dict[str, frozenset[str]]:
    """Each dataset's classes in a window set (D-039: a window's softmax runs over these)."""
    pairs = np.unique(np.stack([windows.datasets, windows.labels]), axis=1)
    classes: dict[str, set[str]] = {}
    for dataset, label in pairs.T.tolist():
        classes.setdefault(dataset, set()).add(label)
    return {dataset: frozenset(labels) for dataset, labels in classes.items()}


def build_classifier(
    settings: TaskSettings,
    layout: Sequence[str],
    n_samples: int,
    classes: Sequence[str],
    sfreq: float,
) -> Any:
    """The model a task trains: EEGNet, or A6's pretrained encoder (``settings.backbone``) with
    its published weights loaded and a new head for ``classes``.

    Call it after seeding: the new head's (and EEGNet's) initialisation draws from the seed.

    Raises:
        NeurOSConfigError: if the weights file is missing or the layout has no positions.
        NeurOSDataError: if the published weights don't match the implementation.
    """
    if settings.backbone is None:
        from neuros.models.eegnet import EEGNetClassifier

        return EEGNetClassifier(len(layout), n_samples, tuple(classes), settings.model)
    from pathlib import Path

    from neuros.models.pretrained import MIRepNetClassifier, ReveClassifier

    backbone = settings.backbone
    model: ReveClassifier | MIRepNetClassifier
    if backbone.name == "reve-base":
        dropout = 0.3 if backbone.dropout is None else backbone.dropout
        model = ReveClassifier(layout, n_samples, classes, sfreq=sfreq, dropout=dropout)
    else:
        model = MIRepNetClassifier(
            layout, n_samples, classes, sfreq=sfreq, dropout=backbone.dropout
        )
    if backbone.weights is not None:  # None: the random-weights control
        model.load_pretrained(Path(backbone.weights))
    return model


def run_task(
    task: ModelTask,
    data: PooledWindows,
    settings: TaskSettings,
    progress: Progress | None = None,
    extra: Mapping[str, PooledWindows] | None = None,
    train_data: PooledWindows | None = None,
    finetune_sets: Mapping[int, tuple[PooledWindows, PooledWindows]] | None = None,
) -> dict[str, Any]:
    """Train one model on ``task.train`` and score every person in ``task.test``.

    ``extra`` maps more window sets to score the same test people on (those present in each;
    the training input scale applies), reported under ``extra_scores``: E2's warm-up sets.

    ``train_data`` (D-039, for a task with ``train_data`` set): the training people's windows
    come from it (crops, extra classes) instead of ``data``; validation and test people's
    windows always come from ``data``. With extra classes, each training window's softmax runs
    over its own dataset's classes, and validation and test over ``settings.classes``.

    ``finetune_sets`` (Q30, with ``settings.finetune``): per calibration size, the calibration
    and test windows of the warm-up datasets' people; after training, each test person present
    is scored with the model as is and with a fine-tuned copy, under ``finetune``.

    Before anything is computed from them, the training and validation windows pass the
    amplitude guard (``check_amplitude``): a window above ``ARTIFACT_RMS_FACTOR`` times its
    person's median RMS stops the task before it trains.

    Raises:
        NeurOSConfigError: if ``train_data`` is given without ``task.train_data``, or the other
            way round.
        NeurOSDataError: if a person in the task has no windows, a label isn't a class, or a
            training or validation window fails the amplitude guard (the message names the
            task, window, person, dataset and ratio).
    """
    import torch

    from neuros.train.data import TensorData
    from neuros.train.device import configure_determinism, describe_device, resolve_device
    from neuros.train.trainer import predict_logits, train_classifier

    if (task.train_data is None) != (train_data is None):
        raise NeurOSConfigError(
            f"{task.name}: train_data is {task.train_data!r} but its windows were "
            f"{'not ' if train_data is None else ''}given"
        )
    say: Progress = progress or (lambda message: print(message, flush=True))
    torch.set_num_threads(settings.threads)
    # A stable per-task index spreads tasks over GPUs (Python's hash() differs per process).
    slot = int(hashlib.sha256(task.name.encode()).hexdigest()[:8], 16)
    device = resolve_device(settings.device, worker_index=slot)
    configure_determinism(device, settings.deterministic)
    started = time.perf_counter()
    head = settings.head_classes(task)
    index = {name: i for i, name in enumerate(head)}
    source = train_data if train_data is not None else data
    # Masks only when the head has classes the scored windows don't: otherwise training,
    # validation and test are exactly E1's (plain cross-entropy and argmax).
    masked = head != settings.classes
    own_classes = _dataset_classes(source) if masked else {}

    def rows(windows: PooledWindows, people: Sequence[str]) -> NDArray[np.int64]:
        found = windows.rows_of(people)
        present = set(windows.people[found].tolist())
        missing = sorted(set(people) - present)
        if missing:
            raise NeurOSDataError(f"{task.name}: no windows for {missing[:5]}")
        return found

    def tensors(
        windows: PooledWindows, people: Sequence[str], scale: float, *, training: bool
    ) -> TensorData:
        picked = rows(windows, people)
        x = _gather(windows.x, picked)  # a private, writable copy
        x /= np.float32(scale)
        labels = windows.labels[picked].tolist()
        allowed_names = set(head) if training else set(settings.classes)
        unknown = sorted(set(labels) - allowed_names)
        if unknown:
            raise NeurOSDataError(f"{task.name}: labels {unknown} aren't classes")
        y = np.array([index[label] for label in labels], dtype=np.int64)
        allowed = None
        if masked:
            if training:  # each window competes among its own dataset's classes
                table = {
                    d: np.array([c in own for c in head], dtype=bool)
                    for d, own in own_classes.items()
                }
                allowed = np.stack([table[d] for d in windows.datasets[picked].tolist()])
            else:  # validation and test: the scored classes only
                allowed = np.zeros((picked.size, len(head)), dtype=bool)
                allowed[:, : len(settings.classes)] = True
        primary = None
        if training and windows.primary is not None:
            primary = torch.from_numpy(np.array(windows.primary[picked], dtype=bool))
        return TensorData(
            x=torch.from_numpy(x),
            y=torch.from_numpy(y),
            people=tuple(windows.people[picked].tolist()),
            classes=head,
            allowed=None if allowed is None else torch.from_numpy(allowed),
            primary=primary,
        )

    # The amplitude guard (10 Oct incident): one glitched training window can set the input
    # scale and the batch-norm statistics on its own, so check before either is computed. The
    # validation windows too: one glitch would dominate its person's validation loss, and so
    # the epoch early stopping keeps. Read-only: a set that passes trains exactly as before.
    # Test windows aren't checked: a glitched one costs only its own prediction.
    train_rows = rows(source, task.train)
    check_amplitude(source, train_rows, role=f"{task.name}: training windows")
    if task.validation:
        validation_rows = rows(data, task.validation)
        check_amplitude(data, validation_rows, role=f"{task.name}: validation windows")
    scale = _rms(source.x, train_rows)
    train = tensors(source, task.train, scale, training=True)
    validation = tensors(data, task.validation, scale, training=False) if task.validation else None
    torch.manual_seed(task.seed)  # weight initialisation
    model = build_classifier(settings, data.layout, train.n_samples, head, data.sfreq)
    result = train_classifier(
        model,
        train,
        validation,
        settings.train,
        seed=task.seed,
        progress=lambda line: say(f"[{task.name}] {line}"),
        device=device,
    )
    test_rows = rows(data, task.test)
    test = tensors(data, task.test, scale, training=False)
    predicted_array = _predict(model, test.x, settings, head, predict_logits)
    person_of = data.people[test_rows]
    truth = data.labels[test_rows]
    flagged = data.flags[test_rows] != 0
    scores: list[dict[str, Any]] = []
    for person in task.test:
        mine = person_of == person
        score = score_person(
            person,
            truth[mine].tolist(),
            predicted_array[mine].tolist(),
            n_flagged=int(flagged[mine].sum()),
        )
        scores.append(score.to_dict() | {"balanced_accuracy_raw": score.balanced_accuracy})
    extra_scores: dict[str, list[dict[str, Any]]] = {}
    for kind, windows in (extra or {}).items():
        extra_scores[kind] = _score_people(
            model, windows, task.test, scale, settings, predict_logits, head
        )
    finetune_scores: dict[str, Any] | None = None
    if finetune_sets and settings.finetune is not None:
        from neuros.eval.finetune import finetune_and_score

        finetune_scores = finetune_and_score(
            model,
            task.test,
            finetune_sets,
            scale=scale,
            settings=settings,
            finetune=settings.finetune,
            head=head,
            seed=task.seed,
            device=device,
        )
    seconds = time.perf_counter() - started
    mean = float(np.mean([s["balanced_accuracy_raw"] for s in scores]))
    minutes = seconds / 60
    say(f"[{task.name}] done: {mean * 100:.1f}% over {len(task.test)} people, {minutes:.1f} min")
    return {
        "task": task.name,
        "condition": task.condition,
        "data": task.data,
        "seed": task.seed,
        "n_train_people": len(task.train),
        "n_validation_people": len(task.validation),
        "n_test_people": len(task.test),
        "n_train_windows": len(train),
        **(
            {"train_data": task.train_data, "n_train_trials": train.n_primary}
            if task.train_data is not None
            else {}
        ),
        "input_scale": scale,
        "scores": scores,
        "train": result.to_dict(),
        "model": model.describe(),
        "hardware": describe_device(device, deterministic=settings.deterministic),
        "seconds": round(seconds, 1),
        **({"extra_scores": extra_scores} if extra_scores else {}),
        **({"finetune": finetune_scores} if finetune_scores is not None else {}),
    }


# Worker state: each spawned worker maps the pooled windows once (initializer), then runs tasks.
_WORKER: dict[str, Any] = {}


def _init_worker(folders: Mapping[str, str], settings: TaskSettings) -> None:
    _WORKER["data"] = {kind: PooledWindows.load(Path(path)) for kind, path in folders.items()}
    _WORKER["settings"] = settings


def _extra_for(
    task: ModelTask, data: Mapping[str, PooledWindows], settings: TaskSettings
) -> dict[str, PooledWindows] | None:
    """The extra sets a task scores on (only models of ``settings.extra_conditions``)."""
    if not settings.extra_kinds or task.condition not in settings.extra_conditions:
        return None
    return {kind: data[kind] for kind in settings.extra_kinds}


def _train_for(task: ModelTask, data: Mapping[str, PooledWindows]) -> PooledWindows | None:
    """The window set a task trains on when it isn't its test windows (D-039)."""
    return None if task.train_data is None else data[task.train_data]


def _finetune_kinds(settings: TaskSettings) -> list[tuple[int, str, str]]:
    if settings.finetune is None:
        return []
    from neuros.eval.finetune import finetune_kinds

    return finetune_kinds(settings.finetune.sizes)


def _finetune_for(
    task: ModelTask, data: Mapping[str, PooledWindows], settings: TaskSettings
) -> dict[int, tuple[PooledWindows, PooledWindows]] | None:
    """Q30's calibration and test sets for a pooled model (None when not fine-tuning)."""
    if settings.finetune is None or task.condition != "pooled":
        return None
    return {n: (data[cal], data[test]) for n, cal, test in _finetune_kinds(settings)}


def _run_in_worker(task: ModelTask) -> dict[str, Any]:
    data, settings = _WORKER["data"], _WORKER["settings"]
    return run_task(
        task,
        data[task.data],
        settings,
        extra=_extra_for(task, data, settings),
        train_data=_train_for(task, data),
        finetune_sets=_finetune_for(task, data, settings),
    )


def run_tasks(
    tasks: Sequence[ModelTask],
    folders: Mapping[str, Path],
    settings: TaskSettings,
    *,
    workers: int = 1,
    progress: Progress | None = None,
    on_task: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run every task (in parallel when ``workers`` > 1); outcomes in the order of ``tasks``.

    Raises:
        NeurOSConfigError: if ``workers`` < 1 or a task needs data that wasn't assembled.
    """
    if workers < 1:
        raise NeurOSConfigError(f"workers must be >= 1, got {workers}")
    needed: set[str] = {t.data for t in tasks} | {t.train_data for t in tasks if t.train_data}
    needed |= {kind for _, cal, test in _finetune_kinds(settings) for kind in (cal, test)}
    missing = sorted((needed | set(settings.extra_kinds)) - set(folders))
    if missing:
        raise NeurOSConfigError(f"tasks need {missing} windows, which weren't assembled")
    say: Progress = progress or (lambda message: print(message, flush=True))
    outcomes: dict[str, dict[str, Any]] = {}
    if workers == 1:
        data = {kind: PooledWindows.load(path) for kind, path in folders.items()}
        for task in tasks:
            outcome = run_task(
                task,
                data[task.data],
                settings,
                say,
                extra=_extra_for(task, data, settings),
                train_data=_train_for(task, data),
                finetune_sets=_finetune_for(task, data, settings),
            )
            outcomes[task.name] = outcome
            if on_task:
                on_task(outcome)
    else:
        # "spawn": forking a process that already started torch's thread pools can deadlock.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)),
            mp_context=context,
            initializer=_init_worker,
            initargs=({kind: str(path) for kind, path in folders.items()}, settings),
        ) as pool:
            futures = {pool.submit(_run_in_worker, task): task.name for task in tasks}
            for future in as_completed(futures):
                outcome = future.result()  # a failed task raises here, with its traceback
                outcomes[futures[future]] = outcome
                if on_task:
                    on_task(outcome)
    return [outcomes[task.name] for task in tasks]


# ---------------------------------------------------------------------- the report
def _scores(outcomes: Sequence[dict[str, Any]], condition: str) -> dict[str, float]:
    """person → balanced accuracy for one condition (each person is tested once per condition)."""
    table: dict[str, float] = {}
    for outcome in outcomes:
        if outcome["condition"] != condition:
            continue
        for score in outcome["scores"]:
            person = score["person"]
            if person in table:
                raise NeurOSDataError(f"{person} tested twice under {condition}")
            table[person] = float(score["balanced_accuracy_raw"])
    return table


def report(
    outcomes: Sequence[dict[str, Any]],
    people: Mapping[str, Sequence[str]],
    cue_free: Mapping[str, bool],
) -> dict[str, Any]:
    """Per-dataset summaries and the paired comparisons E1 asks for.

    For each dataset: the mean (with a bootstrap CI over people) under every condition that
    tested it, and three paired Wilcoxon tests on the people tested under both conditions:
    pooled vs single-dataset, pooled vs leave-one-dataset-out, and task window vs cue-only
    control. ``cue_free`` marks which datasets' task windows avoid the class display.
    """
    tables = {c: _scores(outcomes, c) for c in ("pooled", "single", "lodo", "cue_only")}
    datasets: dict[str, Any] = {}
    for dataset, members in people.items():
        entry: dict[str, Any] = {
            "cue_free": bool(cue_free.get(dataset, False)),
            "n_people": len(members),
        }
        for condition, table in tables.items():
            values = [table[p] for p in members if p in table]
            if values:
                entry[condition] = summarize(values) | {"n_people": len(values)}
        for a, b in (("pooled", "single"), ("pooled", "lodo"), ("pooled", "cue_only")):
            both = [p for p in members if p in tables[a] and p in tables[b]]
            if len(both) >= 2:
                entry[f"{a}_vs_{b}"] = paired_wilcoxon(
                    [tables[a][p] for p in both], [tables[b][p] for p in both]
                )
        datasets[dataset] = entry
    overall: dict[str, Any] = {}
    for a, b in (("pooled", "single"),):
        both = [p for p in tables[a] if p in tables[b]]
        if len(both) >= 2:
            overall[f"{a}_vs_{b}"] = paired_wilcoxon(
                [tables[a][p] for p in both], [tables[b][p] for p in both]
            )
    return {"datasets": datasets, "overall": overall}
