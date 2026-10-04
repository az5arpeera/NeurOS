"""The training loop (architecture §6.4): learning, SWA, early stopping, determinism, failures."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from neuros.align.ea import align_per_person  # noqa: E402
from neuros.config.experiment import TrainConfig  # noqa: E402
from neuros.errors import NeurOSConfigError, NeurOSDataError, NeurOSTrainingError  # noqa: E402
from neuros.models.eegnet import EEGNetClassifier  # noqa: E402
from neuros.train import trainer as trainer_module  # noqa: E402
from neuros.train.data import TensorData, windows_to_tensors  # noqa: E402
from neuros.train.trainer import (  # noqa: E402
    Evaluation,
    average_parameters,
    cosine_lr,
    evaluate,
    predict_logits,
    recompute_batch_norm,
    train_classifier,
)

from .synthetic_people import FAST_TRAIN, N_SAMPLES, TINY_MODEL, people_windows  # noqa: E402

CLASSES = ("left_hand", "right_hand")


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)  # reproducible and fast for tiny tensors


def _model(seed: int = 0) -> EEGNetClassifier:
    torch.manual_seed(seed)
    return EEGNetClassifier(4, N_SAMPLES, CLASSES, TINY_MODEL)


@pytest.fixture(scope="module")
def split() -> tuple[TensorData, TensorData]:
    """Aligned synthetic people: train on persons 1-2, validate on person 3."""
    aligned, _ = align_per_person(people_windows())
    train = windows_to_tensors(aligned.for_people({"Synthetic:1", "Synthetic:2"}), CLASSES)
    validation = windows_to_tensors(aligned.for_people({"Synthetic:3"}), CLASSES)
    return train, validation


def test_cosine_schedule() -> None:
    rates = [cosine_lr(i, 10, 1e-3, 1e-5) for i in range(10)]
    assert rates[0] == pytest.approx(1e-3)
    assert all(a > b for a, b in itertools.pairwise(rates))
    assert rates[-1] > 1e-5
    assert cosine_lr(5, 10, 1e-3, 0.0) == pytest.approx(5e-4)
    with pytest.raises(NeurOSConfigError):
        cosine_lr(10, 10, 1e-3, 0.0)


def test_learns_across_people(split: tuple[TensorData, TensorData]) -> None:
    train, validation = split
    model = _model()
    result = train_classifier(model, train, validation, FAST_TRAIN, seed=0)
    assert result.epochs_trained == FAST_TRAIN.max_epochs
    assert result.end_epoch == FAST_TRAIN.max_epochs
    assert not result.stopped_early
    assert result.swa_epochs == (10, 12)  # the last 25 % of 12 epochs
    assert result.final_validation is not None
    assert result.final_validation["balanced_accuracy"] >= 0.9
    assert result.history[-1].train_loss < result.history[0].train_loss
    assert [r.epoch for r in result.history] == list(range(1, 13))


def test_same_seed_same_weights(split: tuple[TensorData, TensorData]) -> None:
    train, validation = split
    first, second = _model(), _model()
    a = train_classifier(first, train, validation, FAST_TRAIN, seed=3)
    b = train_classifier(second, train, validation, FAST_TRAIN, seed=3)
    for (name, x), y in zip(first.state_dict().items(), second.state_dict().values(), strict=True):
        assert torch.equal(x, y), name
    assert [r.train_loss for r in a.history] == [r.train_loss for r in b.history]


def test_final_weights_are_the_swa_average(
    split: tuple[TensorData, TensorData], monkeypatch: pytest.MonkeyPatch
) -> None:
    train, _ = split
    recorded: list[dict[str, torch.Tensor]] = []
    original = trainer_module._snapshot

    def spy(model: nn.Module) -> dict[str, torch.Tensor]:
        snapshot = original(model)
        recorded.append(snapshot)
        return snapshot

    monkeypatch.setattr(trainer_module, "_snapshot", spy)
    config = FAST_TRAIN.model_copy(update={"max_epochs": 8, "swa_fraction": 0.5})
    model = _model()
    result = train_classifier(model, train, None, config, seed=0)
    assert result.swa_epochs == (5, 8)
    for name, parameter in model.named_parameters():
        expected = torch.stack([snap[name] for snap in recorded[4:8]]).mean(dim=0)
        torch.testing.assert_close(parameter.detach(), expected, rtol=1e-6, atol=1e-6)


def test_swa_off_keeps_the_last_epoch(split: tuple[TensorData, TensorData]) -> None:
    train, _ = split
    config = FAST_TRAIN.model_copy(update={"max_epochs": 3, "swa_fraction": 0.0})
    result = train_classifier(_model(), train, None, config, seed=0)
    assert result.swa_epochs is None
    assert result.end_epoch == 3


def test_early_stopping_ends_at_the_best_epoch(
    split: tuple[TensorData, TensorData], monkeypatch: pytest.MonkeyPatch
) -> None:
    train, validation = split
    losses = iter([1.0, 0.8, 0.85, 0.9, 0.7, 0.95])  # best at epoch 2, then 2 without a gain
    monkeypatch.setattr(
        trainer_module, "evaluate", lambda *args, **kwargs: Evaluation(next(losses), 0.5, 1)
    )
    config = FAST_TRAIN.model_copy(update={"max_epochs": 20, "patience": 2, "swa_fraction": 1.0})
    result = train_classifier(_model(), train, validation, config, seed=0)
    assert result.stopped_early
    assert result.epochs_trained == 4
    assert result.end_epoch == 2
    assert result.swa_epochs == (1, 2)  # 100 % of the run up to its end


def test_early_stopping_needs_validation(split: tuple[TensorData, TensorData]) -> None:
    train, _ = split
    config = FAST_TRAIN.model_copy(update={"patience": 3})
    with pytest.raises(NeurOSConfigError, match="validation"):
        train_classifier(_model(), train, None, config, seed=0)


def test_non_finite_loss_stops_training(split: tuple[TensorData, TensorData]) -> None:
    train, _ = split
    x = train.x.clone()
    x[0, 0, 0] = float("inf")
    poisoned = TensorData(x=x, y=train.y, people=train.people, classes=train.classes)
    with pytest.raises(NeurOSTrainingError, match="epoch 1"):
        train_classifier(_model(), poisoned, None, FAST_TRAIN, seed=0)


def test_data_must_match_the_model(split: tuple[TensorData, TensorData]) -> None:
    train, _ = split
    other = TensorData(x=train.x, y=train.y, people=train.people, classes=("a", "b"))
    with pytest.raises(NeurOSDataError, match="classes"):
        train_classifier(_model(), other, None, FAST_TRAIN, seed=0)
    short = TensorData(
        x=train.x[:, :, :32].contiguous(), y=train.y, people=train.people, classes=CLASSES
    )
    with pytest.raises(NeurOSDataError, match="expects"):
        train_classifier(_model(), short, None, FAST_TRAIN, seed=0)


def test_average_parameters() -> None:
    snapshots = [
        {"w": torch.tensor([1.0, 2.0]), "running": torch.tensor([0.0])},
        {"w": torch.tensor([3.0, 6.0]), "running": torch.tensor([9.0])},
    ]
    state = average_parameters(snapshots, ["w"])
    torch.testing.assert_close(state["w"], torch.tensor([2.0, 4.0]))
    assert state["running"] is snapshots[-1]["running"]  # buffers: from the last snapshot
    with pytest.raises(NeurOSTrainingError):
        average_parameters([], ["w"])
    with pytest.raises(NeurOSTrainingError):
        average_parameters(snapshots, ["missing"])


def test_recompute_batch_norm_gives_the_data_statistics() -> None:
    model = nn.Sequential(nn.BatchNorm1d(3, momentum=0.01))
    x = torch.randn(64, 3, 10) * 4.0 + 2.0
    model.eval()
    recompute_batch_norm(model, x, batch_size=16, generator=torch.Generator().manual_seed(0))
    norm = model[0]
    assert isinstance(norm, nn.BatchNorm1d)
    # Equal-size batches: the mean of batch means is the overall mean, exactly.
    torch.testing.assert_close(norm.running_mean, x.mean(dim=(0, 2)), rtol=1e-5, atol=1e-5)
    assert norm.momentum == 0.01  # restored
    assert not model.training  # mode restored


def test_evaluate_averages_over_people() -> None:
    class Fixed(nn.Module):
        """Logits = the first two channels' first samples (so predictions are chosen)."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, :2, 0]

    # Person a: 4 windows, all right. Person b: 2 windows, one right → per-person mean 0.75,
    # while pooling the 6 windows would give 5/6 accuracy.
    x = torch.zeros(6, 2, 4)
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    x[torch.arange(6), y, 0] = 5.0
    x[5, :, 0] = torch.tensor([5.0, 0.0])  # person b's second window predicted wrongly
    data = TensorData(x=x, y=y, people=("a",) * 4 + ("b",) * 2, classes=CLASSES)
    result = evaluate(Fixed(), data, batch_size=4)
    assert result.n_people == 2
    assert result.balanced_accuracy == pytest.approx(0.75)
    assert math.isfinite(result.loss)


def test_predict_logits_restores_the_mode() -> None:
    model = _model().train()
    predict_logits(model, torch.zeros(3, 4, N_SAMPLES), batch_size=2)
    assert model.training
    with pytest.raises(NeurOSConfigError):
        predict_logits(model, torch.zeros(3, 4, N_SAMPLES), batch_size=0)


def test_result_is_serialisable(split: tuple[TensorData, TensorData]) -> None:
    import json

    train, validation = split
    config = TrainConfig(max_epochs=2, people_per_batch=2, windows_per_person=4, log_every=1)
    lines: list[str] = []
    result = train_classifier(_model(), train, validation, config, seed=0, progress=lines.append)
    assert len(lines) == 2
    assert lines[0].startswith("epoch 1/2")
    payload = result.to_dict()
    json.dumps(payload)
    assert np.isclose(payload["history"][0]["lr"], 1e-3)
