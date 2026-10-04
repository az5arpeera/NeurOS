"""Leave-one-person-out training and testing of the deep model on one dataset (E0).

Rows 3 and 4 of every results table (architecture §8.4): EEGNet without alignment, and
EEGNet + EA. E0 (§8.5) asks whether this pipeline reproduces the literature, EA + EEGNet +
SWA ≈ 74.3 % on BNCI2014-001 left/right (Sagayaraju 2026), within ±3 points. If it doesn't,
there's a pipeline bug, and work stops until it's found.

**Per fold** (one test person):
1. **Inputs:** the chosen channels, then CAR per window if requested, then EA if requested,
   one reference per recording session by default (the paper's protocol; E2 measures the
   online version). EA uses each session's *own* windows only, so it's the same in every
   fold and leaks nothing between people.
2. **Scale:** inputs are divided by the RMS of the *training* windows (fit on training people
   only, principle 2; see ``neuros.train.data``).
3. **Validation people** (optional) come from the fold's training people, for early stopping.
4. **Train** (``neuros.train.trainer``), predict the test person, score them (§8.2).

**Parallel folds.** Folds are independent, so they can run in separate processes. On the dev
laptop (16 cores), 9 workers × 2 threads train about twice as many steps per second as one
process with 16 threads: small convolutions don't use many threads well. Each fold is seeded
on its own, so a fold's result doesn't depend on which worker ran it or when. It does depend
on the thread count (the last bits of parallel sums), which is recorded with every result.
"""

from __future__ import annotations

import multiprocessing
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from neuros.align.ea import align_groups
from neuros.config.experiment import (
    Alignment,
    AlignmentUnit,
    EEGNetConfig,
    LosoExperimentConfig,
    Reference,
    TrainConfig,
)
from neuros.errors import NeurOSConfigError, NeurOSDataError
from neuros.eval.metrics import PersonScore, paired_wilcoxon, score_person, summarize
from neuros.eval.splits import Fold, loso_folds
from neuros.signal.reference import common_average_reference
from neuros.signal.windows import WindowSet

Progress = Callable[[str], None]


def prepare_inputs(
    windows: WindowSet,
    *,
    reference: Reference,
    alignment: Alignment,
    unit: AlignmentUnit = "session",
) -> tuple[WindowSet, dict[str, dict[str, float | int]]]:
    """CAR per window (optional), then EA per session or per person (optional).

    Returns:
        The prepared windows (same order and metadata) and, for EA, each group's reference
        summary (condition number, shrinkage, ...) for the result file.

    Raises:
        NeurOSConfigError: on an unknown reference or alignment.
        NeurOSDataError: see ``align_per_person`` (e.g. a person with too few windows).
    """
    prepared = windows
    if reference == "car":
        prepared = prepared.with_data(common_average_reference(prepared.data))
    elif reference != "as_recorded":
        raise NeurOSConfigError(f"unknown reference {reference!r} (as_recorded, car)")
    if alignment == "none":
        return prepared, {}
    if alignment == "ea":
        aligned, references = align_groups(prepared, unit=unit)
        return aligned, {group: ref.summary() for group, ref in references.items()}
    raise NeurOSConfigError(f"unknown alignment {alignment!r} (none, ea)")


@dataclass(frozen=True, eq=False)
class FoldTask:
    """Everything one fold needs, picklable so it can run in a worker process."""

    fold: Fold
    train: WindowSet
    validation: WindowSet | None
    test: WindowSet
    classes: tuple[str, ...]
    model: EEGNetConfig
    train_config: TrainConfig
    seed: int
    threads: int
    tag: str  # prefix for progress lines, e.g. "ea s20261005 BNCI2014_001:3"


def run_fold(task: FoldTask, progress: Progress | None = None) -> dict[str, Any]:
    """Train on the fold's training people, test on its test person.

    A module-level function so worker processes can import it. With ``progress=None``
    (worker processes), progress lines go to stdout, flushed, prefixed with ``task.tag``.
    """
    import torch

    from neuros.models.eegnet import EEGNetClassifier
    from neuros.train.data import input_rms, windows_to_tensors
    from neuros.train.trainer import predict_logits, train_classifier

    say: Progress = progress or (lambda message: print(message, flush=True))
    torch.set_num_threads(task.threads)
    started = time.perf_counter()
    scale = input_rms(task.train)
    train_data = windows_to_tensors(task.train, task.classes, scale=scale)
    validation_data = (
        windows_to_tensors(task.validation, task.classes, scale=scale)
        if task.validation is not None
        else None
    )
    test_data = windows_to_tensors(task.test, task.classes, scale=scale)

    torch.manual_seed(task.seed)  # weight initialisation
    model = EEGNetClassifier(train_data.n_channels, train_data.n_samples, task.classes, task.model)
    result = train_classifier(
        model,
        train_data,
        validation_data,
        task.train_config,
        seed=task.seed,
        progress=lambda line: say(f"[{task.tag}] {line}"),
    )
    logits = predict_logits(model, test_data.x, task.train_config.eval_batch_size)
    predicted = [task.classes[i] for i in logits.argmax(dim=1).tolist()]
    (person,) = task.fold.test
    score = score_person(
        person,
        task.test.labels,
        predicted,
        n_flagged=task.test.n_flagged(),
    )
    seconds = time.perf_counter() - started
    say(
        f"[{task.tag}] done: {score.balanced_accuracy * 100:.1f}% balanced accuracy "
        f"({len(task.test)} windows), {seconds / 60:.1f} min"
    )
    return {
        "score": score,
        "fold": task.fold.to_dict(),
        "n_train_windows": len(task.train),
        "n_validation_windows": len(task.validation) if task.validation is not None else 0,
        "input_scale": scale,
        "threads": task.threads,
        "model": model.describe(),
        "train": result.to_dict(),
        "seconds": round(seconds, 1),
    }


def _fold_tasks(
    prepared: WindowSet,
    experiment: LosoExperimentConfig,
    *,
    alignment: Alignment,
    seed: int,
    threads: int,
) -> list[FoldTask]:
    folds = loso_folds(prepared.people, n_validation=experiment.validation_people, seed=seed)
    tasks: list[FoldTask] = []
    for fold in folds:
        tasks.append(
            FoldTask(
                fold=fold,
                train=prepared.for_people(fold.train),
                validation=prepared.for_people(fold.validation) if fold.validation else None,
                test=prepared.for_people(fold.test),
                classes=tuple(experiment.labels),
                model=experiment.model,
                train_config=experiment.train,
                seed=seed,
                threads=threads,
                tag=f"{alignment} s{seed} {fold.test[0]}",
            )
        )
    return tasks


def run_deep_loso(
    windows: WindowSet,
    experiment: LosoExperimentConfig,
    *,
    alignment: Alignment,
    seed: int,
    workers: int = 1,
    threads: int = 2,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """One leave-one-person-out run (one alignment, one seed) over every person in ``windows``.

    Raises:
        NeurOSConfigError: on bad worker/thread counts or an unknown alignment/reference.
        NeurOSDataError: on a window whose label isn't one of the experiment's labels, or on
            fewer than 2 people.
    """
    say: Progress = progress or (lambda message: print(message, flush=True))
    if workers < 1 or threads < 1:
        raise NeurOSConfigError(f"workers and threads must be >= 1, got {workers}, {threads}")
    unknown = sorted(set(windows.labels) - set(experiment.labels))
    if unknown:
        raise NeurOSDataError(f"windows carry labels {unknown} outside {list(experiment.labels)}")
    if len(windows.people) < 2:
        raise NeurOSDataError("leave-one-person-out needs at least 2 people")

    started = time.perf_counter()
    prepared, references = prepare_inputs(
        windows,
        reference=experiment.reference,
        alignment=alignment,
        unit=experiment.alignment_unit,
    )
    tasks = _fold_tasks(prepared, experiment, alignment=alignment, seed=seed, threads=threads)
    n_workers = min(workers, len(tasks))
    say(
        f"LOSO eegnet+{alignment}, seed {seed}: {len(tasks)} folds on {n_workers} worker(s) "
        f"x {threads} thread(s)"
    )
    if n_workers == 1:
        outcomes = [run_fold(task, say) for task in tasks]
    else:
        # "spawn" on every OS: forking a process that already started torch's thread pools can
        # deadlock, and Windows has no fork anyway.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=context) as pool:
            outcomes = list(pool.map(run_fold, tasks))

    scores: list[PersonScore] = [outcome.pop("score") for outcome in outcomes]
    accuracies = [s.balanced_accuracy for s in scores]
    return {
        "name": f"eegnet+{alignment}",
        "alignment": alignment,
        "seed": seed,
        "protocol": "loso",
        "summary_balanced_accuracy": summarize(accuracies),
        "summary_kappa": summarize([s.kappa for s in scores]),
        "per_person": [s.to_dict() for s in scores],
        # Unrounded, for comparisons and seed summaries (per_person is rounded for reading).
        "balanced_accuracy_by_person": {s.person: s.balanced_accuracy for s in scores},
        "folds": outcomes,
        "alignment_references": references,
        "workers": n_workers,
        "minutes": round((time.perf_counter() - started) / 60, 2),
    }


def _per_person(run: dict[str, Any]) -> dict[str, float]:
    return {person: float(value) for person, value in run["balanced_accuracy_by_person"].items()}


def compare_runs(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Paired Wilcoxon over people of each aligned run vs the unaligned run with the same seed.

    Raises:
        NeurOSDataError: if two runs being paired don't cover the same people.
    """
    comparisons: list[dict[str, Any]] = []
    baselines = {run["seed"]: run for run in runs if run["alignment"] == "none"}
    for run in runs:
        base = baselines.get(run["seed"])
        if run["alignment"] == "none" or base is None:
            continue
        a, b = _per_person(run), _per_person(base)
        if set(a) != set(b):
            raise NeurOSDataError("runs to compare must cover the same people")
        people = sorted(a)
        comparisons.append(
            {
                "a": run["name"],
                "b": base["name"],
                "seed": run["seed"],
                **paired_wilcoxon([a[p] for p in people], [b[p] for p in people]),
            }
        )
    return comparisons


def seed_summaries(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each alignment run with several seeds: the seed spread and the seed-averaged result.

    Each person's accuracy is averaged over seeds first, then summarised over people, so the
    confidence interval still resamples *people* (the unit of every claim, §8.3).
    """
    summaries: list[dict[str, Any]] = []
    for alignment in dict.fromkeys(run["alignment"] for run in runs):
        mine = [run for run in runs if run["alignment"] == alignment]
        if len(mine) < 2:
            continue
        tables = [_per_person(run) for run in mine]
        people = sorted(tables[0])
        if any(set(table) != set(people) for table in tables):
            raise NeurOSDataError(f"seeds of eegnet+{alignment} cover different people")
        averaged = [float(np.mean([table[p] for table in tables])) for p in people]
        means = [run["summary_balanced_accuracy"]["mean"] for run in mine]
        summaries.append(
            {
                "name": f"eegnet+{alignment}",
                "seeds": [run["seed"] for run in mine],
                "per_seed_mean": means,
                "seed_std_of_mean": float(np.std(means, ddof=1)),
                "summary_balanced_accuracy": summarize(averaged),
            }
        )
    return summaries


def run_loso_experiment(
    windows: WindowSet,
    experiment: LosoExperimentConfig,
    *,
    workers: int = 1,
    threads: int = 2,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Every (alignment × seed) LOSO run of ``experiment``, plus comparisons and seed spread."""
    runs = [
        run_deep_loso(
            windows,
            experiment,
            alignment=alignment,
            seed=seed,
            workers=workers,
            threads=threads,
            progress=progress,
        )
        for alignment in experiment.alignments
        for seed in experiment.seeds
    ]
    return {
        "runs": runs,
        "comparisons": compare_runs(runs),
        "seed_summaries": seed_summaries(runs),
    }
