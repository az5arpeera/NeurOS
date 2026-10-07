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

**Scale.** About 350 development people and 100k windows. The assembled windows are written
once to a work folder as ``.npy`` files, and worker processes map them read-only instead of
receiving gigabytes through pickling. Each model is a task; tasks run in parallel in spawned
workers, on the CPU or on GPUs (``neuros.train.device``), and every finished task is handed to
``on_task`` so the caller can save the result file as it grows.
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
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from neuros.align.ea import align_groups
from neuros.config.experiment import (
    Alignment,
    AlignmentUnit,
    EEGNetConfig,
    PooledCondition,
    Reference,
    TrainConfig,
)
from neuros.errors import NeurOSConfigError, NeurOSDataError
from neuros.eval.metrics import paired_wilcoxon, score_person, summarize
from neuros.eval.splits import Fold, make_dev_folds, seeded_rng
from neuros.signal.reference import common_average_reference
from neuros.signal.windows import WindowSet

Progress = Callable[[str], None]
DataKind = Literal["task", "cue_only"]

# Windows per chunk when computing an RMS over a memory-mapped array (bounded memory).
_RMS_CHUNK = 4096

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
    """

    x: NDArray[np.float32]
    present: NDArray[np.bool_]
    labels: NDArray[np.str_]
    people: NDArray[np.str_]
    datasets: NDArray[np.str_]
    flags: NDArray[np.uint8]
    layout: tuple[str, ...]
    sfreq: float

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

    def __len__(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.x.shape[2])

    def rows_of(self, people: Collection[str]) -> NDArray[np.int64]:
        """Indices of the windows of ``people``, in storage order."""
        return np.flatnonzero(np.isin(self.people, np.asarray(list(people), dtype=str)))

    def people_by_dataset(self) -> dict[str, tuple[str, ...]]:
        """Each dataset's people, in first-appearance order."""
        result: dict[str, list[str]] = {}
        pairs = zip(self.datasets.tolist(), self.people.tolist(), strict=True)
        for dataset, person in dict.fromkeys(pairs):
            result.setdefault(dataset, []).append(person)
        return {dataset: tuple(people) for dataset, people in result.items()}

    # ------------------------------------------------------------------ disk
    _ARRAYS = ("x", "present", "labels", "people", "datasets", "flags")

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
        for name in self._ARRAYS:
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
        for name in cls._ARRAYS:
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
    sources = [source for group in derived.values() for source in group]
    if len(set(sources)) != len(sources):
        raise NeurOSDataError(f"a source channel feeds more than one derived channel: {sources}")
    missing = sorted(set(sources) - set(windows.ch_names))
    if missing:
        raise NeurOSDataError(f"derived channels need {missing}, which these windows lack")
    clash = sorted(set(derived) & set(windows.ch_names))
    if clash:
        raise NeurOSDataError(f"derived channels {clash} are already in the window set")
    used = set(sources)
    keep = [index for index, name in enumerate(windows.ch_names) if name not in used]
    means = [
        windows.data[:, [windows.ch_names.index(s) for s in group], :].mean(
            axis=1, dtype=np.float64
        )
        for group in derived.values()
    ]
    data = np.concatenate(
        [windows.data[:, keep, :], np.stack(means, axis=1).astype(np.float32)], axis=1
    )
    names = tuple(windows.ch_names[i] for i in keep) + tuple(derived)
    return dataclasses.replace(windows, data=np.ascontiguousarray(data), ch_names=names)


def to_layout(
    windows: WindowSet,
    layout: Sequence[str],
    *,
    reference: Reference,
    alignment: Alignment,
    unit: AlignmentUnit,
) -> tuple[NDArray[np.float32], NDArray[np.bool_], dict[str, dict[str, float | int]]]:
    """One dataset's windows in the shared layout: CAR and EA on the channels it has, zeros
    elsewhere (§4.3).

    Returns:
        ``(x [n, len(layout), samples], present [n, len(layout)], EA reference summaries)``.

    Raises:
        NeurOSDataError: if the dataset has fewer than ``MIN_LAYOUT_CHANNELS`` of the layout's
            channels (CAR and covariances need at least two).
        NeurOSConfigError: on an unknown reference or alignment.
    """
    have = [ch for ch in layout if ch in windows.ch_names]
    if len(have) < MIN_LAYOUT_CHANNELS:
        raise NeurOSDataError(
            f"{sorted(set(windows.datasets))}: only {have} of the layout's channels; at least "
            f"{MIN_LAYOUT_CHANNELS} are needed"
        )
    subset = windows.pick_channels(have)  # layout order
    if reference == "car":
        subset = subset.with_data(common_average_reference(subset.data))
    elif reference != "as_recorded":
        raise NeurOSConfigError(f"unknown reference {reference!r} (as_recorded, car)")
    summaries: dict[str, dict[str, float | int]] = {}
    if alignment == "ea":
        subset, references = align_groups(subset, unit=unit)
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
) -> tuple[PooledWindows, dict[str, dict[str, Any]]]:
    """Map every dataset into ``layout`` and stack them (datasets in the given order).

    ``derived`` gives, per dataset, its test-only derived channels (registry
    ``derived_channels``, D-037), applied before the dataset is placed in the layout.

    Returns:
        The pooled windows and, per dataset, a report: channels present and missing, windows,
        people and the EA references.

    Raises:
        NeurOSDataError: on an empty input, mixed sampling rates or window lengths, a window
            set holding another dataset's windows, or non-finite values after alignment.
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
    for dataset, windows in per_dataset.items():
        if set(windows.datasets) != {dataset}:
            raise NeurOSDataError(
                f"windows given for {dataset} hold {sorted(set(windows.datasets))}"
            )
        mapping = (derived or {}).get(dataset, {})
        windows = derive_channels(windows, mapping)
        x, present, references = to_layout(
            windows, layout, reference=reference, alignment=alignment, unit=unit
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
    )
    return pooled, reports


# ---------------------------------------------------------------------- the plan
@dataclass(frozen=True)
class ModelTask:
    """One model to train and test: who trains, validates and is tested, on which windows."""

    name: str  # e.g. "pooled/fold2", "single/Cho2017/fold2", "lodo/Cho2017", "cue_only/fold2"
    condition: PooledCondition
    data: DataKind
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    seed: int

    def __post_init__(self) -> None:
        Fold(index=0, test=self.test, validation=self.validation, train=self.train)  # checks


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
) -> list[ModelTask]:
    """Every model E1 trains (see the module docstring), in a fixed order.

    Args:
        people: each dataset's development people (``dataset:subject``), test-only included.
        trainable: the datasets whose people may train (the rest are only tested).

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
        tasks.append(ModelTask(name, condition, data, kept, validation, tuple(test), seed))

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

    classes: tuple[str, ...]
    model: EEGNetConfig
    train: TrainConfig
    device: str
    deterministic: bool
    threads: int


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


def run_task(
    task: ModelTask, data: PooledWindows, settings: TaskSettings, progress: Progress | None = None
) -> dict[str, Any]:
    """Train one model on ``task.train`` and score every person in ``task.test``.

    Raises:
        NeurOSDataError: if a person in the task has no windows, or a label isn't a class.
    """
    import torch

    from neuros.models.eegnet import EEGNetClassifier
    from neuros.train.data import TensorData
    from neuros.train.device import configure_determinism, describe_device, resolve_device
    from neuros.train.trainer import predict_logits, train_classifier

    say: Progress = progress or (lambda message: print(message, flush=True))
    torch.set_num_threads(settings.threads)
    # A stable per-task index spreads tasks over GPUs (Python's hash() differs per process).
    slot = int(hashlib.sha256(task.name.encode()).hexdigest()[:8], 16)
    device = resolve_device(settings.device, worker_index=slot)
    configure_determinism(device, settings.deterministic)
    started = time.perf_counter()
    index = {name: i for i, name in enumerate(settings.classes)}

    def rows(people: Sequence[str]) -> NDArray[np.int64]:
        found = data.rows_of(people)
        present = set(data.people[found].tolist())
        missing = sorted(set(people) - present)
        if missing:
            raise NeurOSDataError(f"{task.name}: no windows for {missing[:5]}")
        return found

    def tensors(people: Sequence[str], scale: float) -> TensorData:
        picked = rows(people)
        x = np.array(data.x[picked], dtype=np.float32)  # a private, writable copy
        x /= np.float32(scale)
        labels = data.labels[picked].tolist()
        unknown = sorted(set(labels) - set(index))
        if unknown:
            raise NeurOSDataError(f"{task.name}: labels {unknown} aren't classes")
        y = np.array([index[label] for label in labels], dtype=np.int64)
        return TensorData(
            x=torch.from_numpy(x),
            y=torch.from_numpy(y),
            people=tuple(data.people[picked].tolist()),
            classes=settings.classes,
        )

    scale = _rms(data.x, rows(task.train))
    train = tensors(task.train, scale)
    validation = tensors(task.validation, scale) if task.validation else None
    torch.manual_seed(task.seed)  # weight initialisation
    model = EEGNetClassifier(train.n_channels, train.n_samples, settings.classes, settings.model)
    result = train_classifier(
        model,
        train,
        validation,
        settings.train,
        seed=task.seed,
        progress=lambda line: say(f"[{task.name}] {line}"),
        device=device,
    )
    test_rows = rows(task.test)
    test = tensors(task.test, scale)
    predicted = [
        settings.classes[i]
        for i in predict_logits(model, test.x, settings.train.eval_batch_size)
        .argmax(dim=1)
        .tolist()
    ]
    person_of = data.people[test_rows]
    truth = data.labels[test_rows]
    flagged = data.flags[test_rows] != 0
    predicted_array = np.array(predicted, dtype=str)
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
        "input_scale": scale,
        "scores": scores,
        "train": result.to_dict(),
        "model": model.describe(),
        "hardware": describe_device(device, deterministic=settings.deterministic),
        "seconds": round(seconds, 1),
    }


# Worker state: each spawned worker maps the pooled windows once (initializer), then runs tasks.
_WORKER: dict[str, Any] = {}


def _init_worker(folders: Mapping[str, str], settings: TaskSettings) -> None:
    _WORKER["data"] = {kind: PooledWindows.load(Path(path)) for kind, path in folders.items()}
    _WORKER["settings"] = settings


def _run_in_worker(task: ModelTask) -> dict[str, Any]:
    return run_task(task, _WORKER["data"][task.data], _WORKER["settings"])


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
    missing = sorted({t.data for t in tasks} - set(folders))
    if missing:
        raise NeurOSConfigError(f"tasks need {missing} windows, which weren't assembled")
    say: Progress = progress or (lambda message: print(message, flush=True))
    outcomes: dict[str, dict[str, Any]] = {}
    if workers == 1:
        data = {kind: PooledWindows.load(path) for kind, path in folders.items()}
        for task in tasks:
            outcome = run_task(task, data[task.data], settings, say)
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
