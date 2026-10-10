"""Q30, "calibration saved": our pooled model plus a short calibration.

**Question.** The calibration curve (``neuros.eval.calibration``) showed that a full 10-15
minute conventional calibration beats our zero-calibration model by 4-8 points. Can our model,
fine-tuned on a person's first few labelled trials (about 3 minutes), reach that full
calibration's accuracy? Pass bar (set before the run, open questions Q30): on Yang2025 and
BNCI2014-001, fine-tuned with 10 trials per class, mean at or above the half-session
calibration and not significantly worse, on the same test trials.

**Protocol** (per test person of a warm-up dataset, per *block*: a session, or the whole
person for datasets with very short sessions):

- **Trials:** each class's trials in time order; the first half is the calibration pool, the
  rest the test set: exactly the calibration curve's split (``split_block``), so its numbers
  pair with these trial by trial.
- **Calibration:** the first ``n`` trials per class of the pool.
- **Alignment, from the calibration minutes only.** The engine aligns with
  R̂ = (n₀·R_pop + n·R_user)/(n₀ + n) (§5, E2). Here R_user is the mean covariance of the
  quality-gated stream windows recorded *during* the calibration: from ``lead_s`` before the
  first calibration window to the end of the last. No data before the calibration (Yang2025's
  long lead-in) and none after it: "3 minutes" means 3 minutes. Calibration and test windows
  are both aligned with that R̂, fixed at the calibration's end (conservative: the live engine
  keeps adapting during use).
- **Fine-tuning** (worker side, ``finetune_and_score``): a copy of the person's pooled model
  (the fold that never saw them), every layer, the recipe of ``FinetuneConfig`` (fixed before
  the run, never tuned on test people).
- **Scored:** the same test windows by the pooled model as is (zero calibration with that
  reference) and by the fine-tuned copy.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neuros.align.ea import AlignmentReference, apply_alignment, reference_from_mean
from neuros.config.experiment import FinetuneConfig, TrainConfig, WarmupConfig
from neuros.data.recording import Recording
from neuros.data.registry import DatasetSpec
from neuros.errors import DataContext, NeurOSDataError
from neuros.eval.calibration import split_block
from neuros.eval.metrics import score_person
from neuros.eval.pooled import PooledWindows, TaskSettings, _gather, _predict
from neuros.eval.warmup import (
    _profile_rows,
    person_mean_covariances,
    profile_channels,
    session_streams,
)
from neuros.signal.windows import WindowSet

Progress = Callable[[str], None]
_EPS_S = 1e-9


def finetune_kinds(sizes: Sequence[int]) -> list[tuple[int, str, str]]:
    """``(n, calibration set kind, test set kind)`` per calibration size."""
    return [(int(n), f"ft{int(n)}_cal", f"ft{int(n)}_test") for n in sizes]


@dataclass(frozen=True)
class FinetuneSettings:
    """What a worker needs to fine-tune (picklable; from ``FinetuneConfig``)."""

    sizes: tuple[int, ...]
    epochs: int
    lr: float
    min_lr: float

    @classmethod
    def from_config(cls, config: FinetuneConfig) -> FinetuneSettings:
        return cls(tuple(config.sizes), config.epochs, config.lr, config.min_lr)

    def train_config(self, base: TrainConfig, n_windows: int) -> TrainConfig:
        """The fine-tuning recipe: the base training config with every batch = all of the
        person's calibration windows, no early stopping, no SWA (validated)."""
        return TrainConfig.model_validate(
            base.model_dump()
            | {
                "max_epochs": self.epochs,
                "lr": self.lr,
                "min_lr": self.min_lr,
                "people_per_batch": 1,
                "windows_per_person": n_windows,
                "patience": None,
                "swa_fraction": 0.0,
                "log_every": 0,
            }
        )


def build_finetune_sets(
    task_windows: Mapping[str, WindowSet],
    layout: Sequence[str],
    *,
    specs: Mapping[str, DatasetSpec],
    folds: Sequence[frozenset[str]],
    trainable: Sequence[str],
    labels: Sequence[str],
    warmup: WarmupConfig,
    finetune: FinetuneConfig,
    load_recordings: Callable[[str, int], Sequence[Recording]],
    max_ptp_uv: float,
    progress: Progress | None = None,
) -> tuple[dict[str, PooledWindows], dict[str, Any]]:
    """Each size's calibration and test sets, placed in ``layout`` (see the module docs).

    Args:
        task_windows: each dataset's development task windows, unaligned (``_e1_windows``).
        folds: the development folds (person ids), as ``plan_tasks`` draws them.
        trainable: the datasets whose people train (and so form priors).
        load_recordings: ``(dataset, subject) → preprocessed recordings`` (the cache).

    Returns:
        ``(sets by kind, report)``; kinds from ``finetune_kinds``.

    Raises:
        NeurOSDataError: on a person in no fold, no prior, or nothing to calibrate on.
    """
    say: Progress = progress or (lambda message: print(message, flush=True))
    fold_of = {person: k for k, fold in enumerate(folds) for person in fold}
    kinds = finetune_kinds(finetune.sizes)
    rates = {w.sfreq for w in task_windows.values()}
    if len(rates) != 1:
        raise NeurOSDataError(f"task windows differ in sampling rate: {sorted(rates)}")
    sfreq = float(next(iter(rates)))
    prior_windows = warmup.prior_s / warmup.stream_hop_s
    parts: dict[str, list[dict[str, Any]]] = {
        kind: [] for _, cal, test in kinds for kind in (cal, test)
    }
    report: dict[str, Any] = {"recipe": finetune.model_dump(mode="json"), "datasets": {}}

    def placed(
        windows: WindowSet,
        rows: NDArray[np.int64],
        spec: DatasetSpec,
        profile: Sequence[str],
        present: NDArray[np.bool_],
        reference: AlignmentReference,
        context: DataContext,
    ) -> dict[str, Any]:
        x = _profile_rows(windows.data[rows], windows.ch_names, spec, profile, context)
        out = np.zeros((rows.size, len(layout), x.shape[-1]), dtype=np.float32)
        out[:, present, :] = apply_alignment(x, reference)
        return {
            "x": out,
            "present": np.repeat(present[None, :], rows.size, axis=0),
            "labels": [windows.labels[i] for i in rows.tolist()],
            "people": [windows.person_ids[i] for i in rows.tolist()],
            "datasets": [windows.datasets[i] for i in rows.tolist()],
            "flags": windows.flags[rows],
        }

    for dataset in finetune.datasets:
        if dataset not in task_windows:
            raise NeurOSDataError(f"fine-tune dataset {dataset} has no task windows")
        spec = specs[dataset]
        profile = profile_channels(layout, spec)
        present = np.array([ch in profile for ch in layout])
        block_unit = finetune.blocks.get(dataset, "session")
        person_means: dict[str, NDArray[np.float64]] = {}
        for source in trainable:
            person_means |= person_mean_covariances(task_windows[source], specs[source], profile)
        windows = task_windows[dataset]
        window_s = windows.n_samples / sfreq
        person_ids = np.array(windows.person_ids)
        sessions = np.array(windows.sessions)
        runs = np.array(windows.runs)
        entry: dict[str, Any] = {"profile": list(profile), "block": block_unit, "people": {}}
        for person in windows.people:
            if person not in fold_of:
                raise NeurOSDataError(f"{person} is in no development fold")
            outside = [p for p in person_means if p not in folds[fold_of[person]]]
            if not outside:
                raise NeurOSDataError(f"no training person forms a prior for {dataset}")
            prior = np.mean([person_means[p] for p in outside], axis=0)
            subject = int(person.split(":", 1)[1])
            streams = session_streams(
                load_recordings(dataset, subject), spec, profile, warmup, max_ptp_uv=max_ptp_uv
            )
            mine = np.flatnonzero(person_ids == person).astype(np.int64)
            blocks = (
                {person: mine}
                if block_unit == "person"
                else {s: mine[sessions[mine] == s] for s in dict.fromkeys(sessions[mine].tolist())}
            )
            person_report: dict[str, Any] = {}
            for block_name, rows in blocks.items():
                pool, test = split_block(windows, rows, labels)
                for n, cal_kind, test_kind in kinds:
                    if min(pool[label].size for label in labels) < n:
                        person_report.setdefault(str(n), []).append(
                            {"block": block_name, "skipped": "fewer calibration trials than n"}
                        )
                        continue
                    cal = np.sort(np.concatenate([pool[label][:n] for label in labels]))
                    # Calibration intervals in each session it touches (session seconds).
                    intervals: dict[str, tuple[float, float]] = {}
                    for i in cal.tolist():
                        stream = streams.get(sessions[i])
                        if stream is None:
                            raise NeurOSDataError(f"{person} session {sessions[i]}: no recording")
                        t = stream.run_offsets_s[runs[i]] + windows.starts[i] / sfreq
                        low, high = intervals.get(sessions[i], (math.inf, -math.inf))
                        intervals[sessions[i]] = (
                            min(low, t - finetune.lead_s),
                            max(high, t + window_s),
                        )
                    covs: list[NDArray[np.float64]] = []
                    for session, interval in intervals.items():
                        stream = streams[session]
                        used = (stream.end_s - window_s >= interval[0] - _EPS_S) & (
                            stream.end_s <= interval[1] + _EPS_S
                        )
                        covs.append(stream.covariances[used])
                    stacked = np.concatenate(covs) if covs else np.zeros((0, *prior.shape))
                    context = DataContext(
                        dataset, subject, None if block_unit == "person" else block_name
                    )
                    if stacked.shape[0]:
                        user = stacked.mean(axis=0)
                        blend = (prior_windows * prior + stacked.shape[0] * user) / (
                            prior_windows + stacked.shape[0]
                        )
                        reference = reference_from_mean(blend, stacked.shape[0], context=context)
                    else:
                        reference = reference_from_mean(prior, 0, context=context)
                    parts[cal_kind].append(
                        placed(windows, cal, spec, profile, present, reference, context)
                    )
                    parts[test_kind].append(
                        placed(windows, test, spec, profile, present, reference, context)
                    )
                    person_report.setdefault(str(n), []).append(
                        {
                            "block": block_name,
                            "n_calibration": int(cal.size),
                            "n_test": int(test.size),
                            "calibration_s": float(sum(b - a for a, b in intervals.values())),
                            "n_stream_windows": int(stacked.shape[0]),
                        }
                    )
            entry["people"][person] = person_report
        report["datasets"][dataset] = entry
        say(f"fine-tune sets {dataset}: {len(entry['people'])} people, block {block_unit}")

    sets: dict[str, PooledWindows] = {}
    for kind, chunks in parts.items():
        if not chunks:
            raise NeurOSDataError(f"no person has enough calibration trials for {kind}")
        sets[kind] = PooledWindows(
            x=np.concatenate([c["x"] for c in chunks]),
            present=np.concatenate([c["present"] for c in chunks]),
            labels=np.array([label for c in chunks for label in c["labels"]], dtype=str),
            people=np.array([p for c in chunks for p in c["people"]], dtype=str),
            datasets=np.array([d for c in chunks for d in c["datasets"]], dtype=str),
            flags=np.concatenate([c["flags"] for c in chunks]).astype(np.uint8),
            layout=tuple(layout),
            sfreq=sfreq,
        )
        parts[kind] = []  # release each size's chunks once stacked
    return sets, report


def _score(person: str, truth: list[str], predicted: list[str], flagged: int) -> dict[str, Any]:
    """A person's score in run_task's format (with the unrounded balanced accuracy)."""
    score = score_person(person, truth, predicted, n_flagged=flagged)
    return score.to_dict() | {"balanced_accuracy_raw": score.balanced_accuracy}


def finetune_and_score(
    model: Any,
    people: Sequence[str],
    sets: Mapping[int, tuple[PooledWindows, PooledWindows]],
    *,
    scale: float,
    settings: TaskSettings,
    finetune: FinetuneSettings,
    head: Sequence[str],
    seed: int,
    device: Any,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Per size: each person's test score with the pooled model as is, and fine-tuned.

    ``people`` are the task's test people; those absent from a size's sets are skipped (other
    datasets, or too few calibration trials). The pooled model is never modified.

    Raises:
        NeurOSDataError: on labels outside the head, or a person with calibration windows but
            no test windows.
    """
    import torch

    from neuros.train.data import TensorData
    from neuros.train.trainer import predict_logits, train_classifier

    index = {name: i for i, name in enumerate(head)}
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for n, (cal, test) in sorted(sets.items()):
        present = set(cal.people.tolist())
        zero: list[dict[str, Any]] = []
        tuned: list[dict[str, Any]] = []
        for person in people:
            if person not in present:
                continue
            rows_c = cal.rows_of([person])
            rows_t = test.rows_of([person])
            if rows_t.size == 0:
                raise NeurOSDataError(f"{person}: calibration windows but no test windows (n={n})")
            labels_c = cal.labels[rows_c].tolist()
            unknown = sorted(set(labels_c) - set(index))
            if unknown:
                raise NeurOSDataError(f"{person}: calibration labels {unknown} aren't classes")
            x_c = _gather(cal.x, rows_c)
            x_c /= np.float32(scale)
            x_t = _gather(test.x, rows_t)
            x_t /= np.float32(scale)
            truth = test.labels[rows_t].tolist()
            flagged = int(np.count_nonzero(test.flags[rows_t]))
            before = _predict(model, torch.from_numpy(x_t), settings, head, predict_logits)
            zero.append(_score(person, truth, before.tolist(), flagged))
            copy_of = copy.deepcopy(model)
            train_classifier(
                copy_of,
                TensorData(
                    x=torch.from_numpy(x_c),
                    y=torch.tensor([index[label] for label in labels_c], dtype=torch.int64),
                    people=(person,) * rows_c.size,
                    classes=tuple(head),
                ),
                None,
                finetune.train_config(settings.train, int(rows_c.size)),
                seed=seed,
                device=device,
            )
            after = _predict(copy_of, torch.from_numpy(x_t), settings, head, predict_logits)
            tuned.append(_score(person, truth, after.tolist(), flagged))
            del copy_of
        out[str(n)] = {"zero_calibration": zero, "finetuned": tuned}
    return out
