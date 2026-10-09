"""The training loop for encoder + head (architecture §6.4).

**Recipe**
- **AdamW** (Adam with decoupled weight decay); the learning rate follows a **cosine** from
  ``lr`` at the first epoch towards ``min_lr`` at the end of ``max_epochs``.
- **Subject-balanced batches** of K people × M windows (``SubjectBalancedSampler``).
- **Max-norm constraints** (EEGNet's) re-applied after every optimiser step.
- **Early stopping on validation people** (optional; never test people, principle 2). The run's
  *end epoch* is the one with the lowest validation loss, and training stops once ``patience``
  epochs pass without an improvement larger than ``min_delta``.
- **SWA** (stochastic weight averaging, Izmailov et al. 2018). The final weights are the uniform
  average of the weights after each of the last ``swa_fraction`` of the epochs up to and
  including the end epoch (at least one).
- **Batch-norm statistics** of the final weights are recomputed with one pass over the training
  windows. Averaged weights need their own statistics (averaging the running statistics of
  different weights doesn't give the right ones), and so does a single snapshot after a short
  run, because EEGNet's running statistics move only 1 % per step.

Without early stopping the end epoch is the last one, so a 300-epoch run with
``swa_fraction = 0.25`` averages epochs 226-300 (E0's setting). With early stopping, "the run"
is the epochs up to the best one, and SWA averages the last quarter of those.

**Failure policy.** A non-finite training or validation loss raises ``NeurOSTrainingError``
naming the epoch, batch and learning rate; training never continues on corrupt weights.

**Determinism.** The caller seeds weight initialisation; this loop seeds dropout, batch order
and the batch-norm pass from ``seed``. With the same seed, data, thread count and library
versions a run reproduces exactly. The thread count is recorded with every result because
parallel reductions can change the last bits of a sum.

**Devices (D-032).** The model trains on whatever device it sits on (``device=`` moves it
first). Windows stay in CPU memory and move to the device one batch at a time, so the training
set never has to fit in GPU memory. Weight snapshots for SWA are kept on the CPU. On CUDA,
reproducibility needs ``neuros.train.device.configure_determinism`` (the fold runner calls it).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from neuros.config.experiment import TrainConfig
from neuros.errors import NeurOSConfigError, NeurOSDataError, NeurOSTrainingError
from neuros.eval.metrics import balanced_accuracy
from neuros.models.eegnet import EEGNetClassifier
from neuros.train.data import TensorData
from neuros.train.sampler import SubjectBalancedSampler

_BATCH_NORMS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


@dataclass
class EpochRecord:
    """What one epoch did (1-based epoch numbers, as people read them)."""

    epoch: int
    lr: float
    train_loss: float
    val_loss: float | None = None
    val_balanced_accuracy: float | None = None
    seconds: float = 0.0


@dataclass
class TrainResult:
    """How a training run went. The model passed in now holds the final weights."""

    epochs_trained: int
    end_epoch: int
    stopped_early: bool
    swa_epochs: tuple[int, int] | None
    final_validation: dict[str, float] | None
    seconds: float
    history: list[EpochRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable (numbers rounded for compact result files)."""

        def rounded(record: EpochRecord) -> dict[str, Any]:
            return {
                key: round(value, 5) if isinstance(value, float) else value
                for key, value in asdict(record).items()
            }

        return {
            "epochs_trained": self.epochs_trained,
            "end_epoch": self.end_epoch,
            "stopped_early": self.stopped_early,
            "swa_epochs": list(self.swa_epochs) if self.swa_epochs else None,
            "final_validation": self.final_validation,
            "seconds": round(self.seconds, 1),
            "history": [rounded(r) for r in self.history],
        }


def cosine_lr(epoch_index: int, max_epochs: int, lr: float, min_lr: float) -> float:
    """Learning rate for the 0-based ``epoch_index``: ``lr`` at 0, approaching ``min_lr``."""
    if not 0 <= epoch_index < max_epochs:
        raise NeurOSConfigError(f"epoch index {epoch_index} outside [0, {max_epochs})")
    return min_lr + (lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * epoch_index / max_epochs))


def model_device(model: nn.Module) -> torch.device:
    """The device a model's parameters live on (CPU for a model without parameters)."""
    first = next(model.parameters(), None)
    return first.device if first is not None else torch.device("cpu")


@torch.no_grad()
def predict_logits(model: nn.Module, x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Logits for every window, in evaluation mode (no dropout, running BN statistics).

    ``x`` may live on the CPU; batches move to the model's device. The logits come back on the
    CPU. The model's previous mode (train/eval) is restored afterwards.
    """
    if batch_size < 1:
        raise NeurOSConfigError(f"batch_size must be >= 1, got {batch_size}")
    device = model_device(model)
    was_training = model.training
    model.eval()
    try:
        parts = [
            model(x[start : start + batch_size].to(device)).cpu()
            for start in range(0, x.shape[0], batch_size)
        ]
    finally:
        model.train(was_training)
    return torch.cat(parts)


def masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    allowed: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Mean cross-entropy where each window's softmax runs over its ``allowed`` classes only.

    D-039's extra classes: a PhysionetMI window competes among PhysionetMI's classes, and is
    never pushed away from "tongue", which PhysionetMI doesn't have. Label smoothing spreads
    its mass ε over the allowed classes only (PyTorch's spreads it over every class, which
    with masked logits would be an infinite loss). With every class allowed this equals
    ``F.cross_entropy(logits, target, label_smoothing=ε)`` (tested).

    Args:
        logits: ``[B, C]``. target: ``[B]`` class indices. allowed: ``[B, C]`` bool, with each
            window's own class allowed (``TensorData`` checks it).
    """
    if not 0.0 <= label_smoothing < 1.0:
        raise NeurOSConfigError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
    if allowed.shape != logits.shape or target.shape != logits.shape[:1]:
        raise NeurOSDataError(
            f"logits {tuple(logits.shape)}, target {tuple(target.shape)} and allowed "
            f"{tuple(allowed.shape)} don't match"
        )
    log_p = F.log_softmax(logits.masked_fill(~allowed, float("-inf")), dim=1)
    nll = -log_p.gather(1, target[:, None]).squeeze(1)
    if label_smoothing == 0.0:
        return nll.mean()
    # Masked entries are -inf: replace them before summing (0 x -inf would be NaN).
    uniform = -torch.where(allowed, log_p, torch.zeros_like(log_p)).sum(dim=1) / allowed.sum(1)
    return ((1.0 - label_smoothing) * nll + label_smoothing * uniform).mean()


@dataclass(frozen=True)
class Evaluation:
    """Validation metrics, each averaged over people (statistics over people, §8.2)."""

    loss: float
    balanced_accuracy: float
    n_people: int

    def to_dict(self) -> dict[str, float]:
        return {
            "loss": round(self.loss, 5),
            "balanced_accuracy": round(self.balanced_accuracy, 5),
            "n_people": self.n_people,
        }


def evaluate(model: nn.Module, data: TensorData, batch_size: int) -> Evaluation:
    """Per-person cross-entropy and balanced accuracy, averaged over people.

    With ``data.allowed`` (D-039), each window's loss and prediction use its allowed classes
    only: a head that also learned feet and tongue is validated on left vs right.

    Raises:
        NeurOSTrainingError: if the loss is NaN/Inf (the weights are unusable).
    """
    logits = predict_logits(model, data.x, batch_size)
    if data.allowed is not None:  # disallowed classes can neither win nor share probability
        logits = logits.masked_fill(~data.allowed, float("-inf"))
    predicted = logits.argmax(dim=1).numpy()
    truth = data.y.numpy()
    person_of = np.array(data.people)
    losses: list[float] = []
    accuracies: list[float] = []
    for person in dict.fromkeys(data.people):
        rows = np.flatnonzero(person_of == person)
        index = torch.from_numpy(rows)
        losses.append(float(F.cross_entropy(logits[index], data.y[index])))
        accuracies.append(balanced_accuracy(truth[rows], predicted[rows]))
    loss = float(np.mean(losses))
    if not math.isfinite(loss):
        raise NeurOSTrainingError(f"validation loss is {loss}: the weights are unusable")
    return Evaluation(loss, float(np.mean(accuracies)), len(losses))


@torch.no_grad()
def recompute_batch_norm(
    model: nn.Module, x: torch.Tensor, batch_size: int, generator: torch.Generator
) -> None:
    """Re-estimate every batch-norm layer's running statistics on ``x`` (after SWA).

    Like ``torch.optim.swa_utils.update_bn``: statistics are reset, then replaced by the
    equal-weight average over batches (``momentum=None``) of one pass in training mode. The
    windows are shuffled first so each batch mixes people the way training batches did: an
    average of *within-batch* variances only matches the training statistics if the batches
    are random samples, not one person at a time.
    """
    norms = [m for m in model.modules() if isinstance(m, _BATCH_NORMS)]
    if not norms:
        return
    device = model_device(model)
    momenta = [m.momentum for m in norms]
    for m in norms:
        m.reset_running_stats()
        m.momentum = None
    was_training = model.training
    model.train()
    order = torch.randperm(x.shape[0], generator=generator)
    try:
        for start in range(0, x.shape[0], batch_size):
            model(x[order[start : start + batch_size]].to(device))
    finally:
        for m, momentum in zip(norms, momenta, strict=True):
            m.momentum = momentum
        model.train(was_training)


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    """A CPU copy of the weights and buffers (snapshots never pile up in GPU memory)."""
    return {name: value.detach().to("cpu", copy=True) for name, value in model.state_dict().items()}


def average_parameters(
    snapshots: Sequence[dict[str, torch.Tensor]], parameter_names: Sequence[str]
) -> dict[str, torch.Tensor]:
    """The last snapshot with every named parameter replaced by its mean over all snapshots.

    Averaging is in float64 (75 float32 values per weight would otherwise lose a few bits);
    buffers (batch-norm statistics) are taken from the last snapshot and must be recomputed.

    Raises:
        NeurOSTrainingError: on an empty list or a missing parameter.
    """
    if not snapshots:
        raise NeurOSTrainingError("no weight snapshots to average")
    state = dict(snapshots[-1])
    for name in parameter_names:
        if any(name not in snap for snap in snapshots):
            raise NeurOSTrainingError(f"parameter {name!r} is missing from a snapshot")
        stacked = torch.stack([snap[name].to(torch.float64) for snap in snapshots])
        state[name] = stacked.mean(dim=0).to(snapshots[-1][name].dtype)
    return state


def _check_data(model: EEGNetClassifier, data: TensorData, role: str) -> None:
    if data.classes != model.classes:
        raise NeurOSDataError(
            f"{role} classes {list(data.classes)} differ from the model's {list(model.classes)}"
        )
    if data.n_channels != model.n_channels or data.n_samples != model.n_samples:
        raise NeurOSDataError(
            f"{role} windows are [{data.n_channels} ch, {data.n_samples} samples], the model "
            f"expects [{model.n_channels}, {model.n_samples}]"
        )


def train_classifier(
    model: EEGNetClassifier,
    train: TensorData,
    validation: TensorData | None,
    config: TrainConfig,
    *,
    seed: int,
    progress: Callable[[str], None] | None = None,
    device: torch.device | None = None,
) -> TrainResult:
    """Train ``model`` in place with the §6.4 recipe; return how it went.

    Args:
        model: freshly initialised (seed it before construction for reproducible weights).
        train: training windows (training people only).
        validation: validation people's windows, or None. Required for early stopping.
        seed: dropout, batch order and the batch-norm pass.
        progress: receives one line every ``config.log_every`` epochs.
        device: where to train; the model is moved there first. None = where it already is.

    Raises:
        NeurOSConfigError: early stopping without validation data.
        NeurOSDataError: data that doesn't match the model (classes, channels, samples).
        NeurOSTrainingError: a non-finite loss.
    """
    _check_data(model, train, "training")
    if validation is not None:
        _check_data(model, validation, "validation")
    if config.patience is not None and validation is None:
        raise NeurOSConfigError("early stopping (patience) needs validation people")
    if device is not None:
        model.to(device)
    target = model_device(model)

    torch.manual_seed(seed)  # every device's generator: dropout on the GPU too
    generator = torch.Generator().manual_seed(seed)
    sampler = SubjectBalancedSampler(
        train.people,
        train.y.numpy(),
        people_per_batch=config.people_per_batch,
        windows_per_person=config.windows_per_person,
        seed=seed,
        # D-039: an epoch is a pass over the trials; extra crops vary what a step sees.
        epoch_windows=train.n_primary if train.primary is not None else None,
    )
    allowed = train.allowed  # None: every class competes (E0/E1's plain cross-entropy)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    parameter_names = [name for name, _ in model.named_parameters()]

    snapshots: list[dict[str, torch.Tensor]] = []
    history: list[EpochRecord] = []
    best_loss = math.inf
    best_index = -1
    since_best = 0
    stopped_early = False
    started = time.perf_counter()

    for index in range(config.max_epochs):
        epoch_started = time.perf_counter()
        lr = cosine_lr(index, config.max_epochs, config.lr, config.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch_number, batch in enumerate(sampler.epoch(), start=1):
            rows = torch.from_numpy(batch)
            logits = model(train.x[rows].to(target, non_blocking=True))
            if allowed is None:
                loss = F.cross_entropy(
                    logits,
                    train.y[rows].to(target, non_blocking=True),
                    label_smoothing=config.label_smoothing,
                )
            else:
                loss = masked_cross_entropy(
                    logits,
                    train.y[rows].to(target, non_blocking=True),
                    allowed[rows].to(target, non_blocking=True),
                    label_smoothing=config.label_smoothing,
                )
            loss_value = float(loss.detach())
            if not math.isfinite(loss_value):
                raise NeurOSTrainingError(
                    f"training loss became {loss_value} at epoch {index + 1}, batch "
                    f"{batch_number} (lr {lr:.3g}); lower the learning rate or check the inputs"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]  # unannotated in torch 2.14
            optimizer.step()
            model.apply_max_norm()
            loss_sum += loss_value * batch.size
            seen += batch.size
        snapshots.append(_snapshot(model))
        record = EpochRecord(epoch=index + 1, lr=lr, train_loss=loss_sum / seen)

        if validation is not None:
            result = evaluate(model, validation, config.eval_batch_size)
            record.val_loss = result.loss
            record.val_balanced_accuracy = result.balanced_accuracy
            if result.loss < best_loss - config.min_delta:
                best_loss, best_index, since_best = result.loss, index, 0
            else:
                since_best += 1
        record.seconds = time.perf_counter() - epoch_started
        history.append(record)

        if progress and config.log_every and (index + 1) % config.log_every == 0:
            line = (
                f"epoch {index + 1}/{config.max_epochs} lr {lr:.2e} train {record.train_loss:.3f}"
            )
            if record.val_loss is not None and record.val_balanced_accuracy is not None:
                line += f" val {record.val_loss:.3f} {record.val_balanced_accuracy * 100:.1f}%"
            progress(line)
        if config.patience is not None and since_best >= config.patience:
            stopped_early = True
            break

    # With early stopping the run ends at the best validation epoch; otherwise at the last one.
    end_index = best_index if config.patience is not None else len(snapshots) - 1
    swa_epochs: tuple[int, int] | None = None
    if config.swa_fraction > 0:
        n_average = max(1, math.ceil(config.swa_fraction * (end_index + 1)))
        first = end_index - n_average + 1
        model.load_state_dict(average_parameters(snapshots[first : end_index + 1], parameter_names))
        swa_epochs = (first + 1, end_index + 1)
    else:
        model.load_state_dict(snapshots[end_index])
    # Always recompute the batch-norm statistics for the final weights. Averaged weights need
    # it, and even a single snapshot benefits: running statistics move only 1 % per step
    # (EEGNet's momentum), so they lag the weights and are far off after a short run.
    recompute_batch_norm(model, train.x, config.eval_batch_size, generator)

    final = (
        evaluate(model, validation, config.eval_batch_size).to_dict()
        if validation is not None
        else None
    )
    return TrainResult(
        epochs_trained=len(snapshots),
        end_epoch=end_index + 1,
        stopped_early=stopped_early,
        swa_epochs=swa_epochs,
        final_validation=final,
        seconds=time.perf_counter() - started,
        history=history,
    )
