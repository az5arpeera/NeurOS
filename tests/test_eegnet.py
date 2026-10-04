"""EEGNet encoder + linear head (architecture §6.1-6.2): shapes, size, checks, constraints."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F  # noqa: E402

from neuros.config.experiment import EEGNetConfig  # noqa: E402
from neuros.errors import NeurOSConfigError, NeurOSDataError  # noqa: E402
from neuros.models.eegnet import (  # noqa: E402
    EEGNetClassifier,
    LinearHead,
    output_samples,
    same_padding,
)

CLASSES = ("left_hand", "right_hand")


@pytest.fixture(autouse=True)
def _few_threads() -> None:
    torch.set_num_threads(1)


@pytest.mark.parametrize(("channels", "samples"), [(22, 256), (4, 256), (26, 512), (3, 64)])
def test_shapes(channels: int, samples: int) -> None:
    model = EEGNetClassifier(channels, samples, CLASSES).eval()
    x = torch.randn(5, channels, samples)
    assert model(x).shape == (5, 2)
    assert model.embed(x).shape == (5, 64)
    assert output_samples(samples, model.config) == samples // 32


def test_size_matches_the_architecture_budget() -> None:
    model = EEGNetClassifier(22, 256, CLASSES)
    # Counted by hand: temporal 16·64, spatial 32·22, separable 32·16 + 32·32, three batch
    # norms (2·16, 2·32, 2·32), embedding 256·64 + 64, head 64·2 + 2.
    expected = 16 * 64 + 32 * 22 + 32 * 16 + 32 * 32 + 2 * (16 + 32 + 32) + 256 * 64 + 64 + 130
    assert model.n_parameters == expected == 20_002
    assert model.n_parameters * 4 < 1_000_000  # < 1 MB as float32 (§6.1)


# The reference call below uses padding="same" on purpose; PyTorch warns about its copy.
@pytest.mark.filterwarnings("ignore:Using padding='same'")
@pytest.mark.parametrize(("kernel", "expected"), [(64, (31, 32)), (16, (7, 8)), (5, (2, 2))])
def test_same_padding_matches_pytorch_same(kernel: int, expected: tuple[int, int]) -> None:
    assert same_padding(kernel) == expected
    x = torch.randn(2, 1, 3, 40)
    w = torch.randn(4, 1, 1, kernel)
    torch.testing.assert_close(
        F.conv2d(F.pad(x, same_padding(kernel)), w), F.conv2d(x, w, padding="same")
    )


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((2, 22, 256), dtype=np.float32),  # not a tensor
        torch.zeros(22, 256),  # no batch axis
        torch.zeros(2, 21, 256),  # wrong channel count
        torch.zeros(2, 22, 255),  # wrong length
        torch.zeros(0, 22, 256),  # empty batch
        torch.zeros(2, 22, 256, dtype=torch.float64),  # wrong dtype
    ],
)
def test_forward_rejects_bad_batches(bad: object) -> None:
    model = EEGNetClassifier(22, 256, CLASSES)
    with pytest.raises(NeurOSDataError):
        model(bad)


def test_construction_errors() -> None:
    with pytest.raises(NeurOSConfigError, match="too short"):
        EEGNetClassifier(22, 31, CLASSES)  # pools of 4 × 8 need >= 32 samples
    with pytest.raises(NeurOSConfigError):
        EEGNetClassifier(0, 256, CLASSES)
    with pytest.raises(NeurOSConfigError):
        EEGNetClassifier(22, 256, ["left_hand"])
    with pytest.raises(NeurOSConfigError):
        EEGNetClassifier(22, 256, ["left_hand", "left_hand"])
    with pytest.raises(NeurOSConfigError):
        LinearHead(64, 1)
    head = LinearHead(4, 2)
    with pytest.raises(NeurOSDataError):
        head(torch.zeros(3, 5))


def test_max_norm_constraints() -> None:
    model = EEGNetClassifier(8, 64, CLASSES)
    with torch.no_grad():
        model.encoder.spatial.weight.mul_(100.0)
        model.head.linear.weight.mul_(100.0)
    model.apply_max_norm()
    spatial = model.encoder.spatial.weight.flatten(1).norm(dim=1)
    head = model.head.linear.weight.norm(dim=1)
    assert torch.all(spatial <= 1.0 + 1e-5)
    assert torch.all(head <= 0.25 + 1e-5)
    # Weights already within the limit are left exactly as they are (a cap, not a projection).
    with torch.no_grad():
        model.head.linear.weight.mul_(0.1)
    before = model.head.linear.weight.clone()
    model.apply_max_norm()
    assert torch.equal(model.head.linear.weight, before)


def test_max_norm_can_be_disabled() -> None:
    config = EEGNetConfig(spatial_max_norm=None, head_max_norm=None)
    model = EEGNetClassifier(8, 64, CLASSES, config)
    with torch.no_grad():
        model.encoder.spatial.weight.mul_(100.0)
    before = model.encoder.spatial.weight.clone()
    model.apply_max_norm()
    assert torch.equal(model.encoder.spatial.weight, before)


def test_initialisation_is_seeded() -> None:
    def build(seed: int) -> dict[str, torch.Tensor]:
        torch.manual_seed(seed)
        return EEGNetClassifier(8, 64, CLASSES).state_dict()

    a, b, c = build(1), build(1), build(2)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not all(torch.equal(a[k], c[k]) for k in a)


def test_dropout_only_in_training_mode() -> None:
    torch.manual_seed(0)
    model = EEGNetClassifier(8, 64, CLASSES)
    x = torch.randn(16, 8, 64)
    model.eval()
    assert torch.equal(model(x), model(x))
    model.train()
    assert not torch.equal(model(x), model(x))


def test_every_parameter_learns() -> None:
    torch.manual_seed(0)
    model = EEGNetClassifier(8, 64, CLASSES)
    loss = F.cross_entropy(model(torch.randn(16, 8, 64)), torch.randint(0, 2, (16,)))
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_describe_is_serialisable() -> None:
    description = EEGNetClassifier(22, 256, CLASSES).describe()
    assert description["n_parameters"] == 20_002
    assert description["classes"] == list(CLASSES)
    json.dumps(description)
