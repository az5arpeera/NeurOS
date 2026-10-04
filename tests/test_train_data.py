"""From WindowSet to tensors: class indices, the training-people input scale, validation."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neuros.errors import NeurOSConfigError, NeurOSDataError  # noqa: E402
from neuros.train import data as data_module  # noqa: E402
from neuros.train.data import TensorData, input_rms, windows_to_tensors  # noqa: E402

from .synthetic_people import people_windows  # noqa: E402

CLASSES = ("left_hand", "right_hand")


def test_windows_to_tensors() -> None:
    windows = people_windows((1.0, 2.0), per_class=3)
    tensors = windows_to_tensors(windows, CLASSES, scale=2.0)
    assert tensors.x.dtype == torch.float32
    assert tensors.x.shape == windows.data.shape
    np.testing.assert_allclose(tensors.x.numpy(), windows.data / 2.0, rtol=1e-6)
    assert tensors.labels() == list(windows.labels)
    assert tensors.people == windows.person_ids
    assert set(tensors.y.tolist()) == {0, 1}
    assert len(tensors) == len(windows)
    # The source stays untouched (and read-only); the tensor owns writable memory.
    assert not windows.data.flags.writeable
    tensors.x[0, 0, 0] = 123.0
    assert windows.data[0, 0, 0] != 123.0


def test_unknown_labels_and_bad_scales_are_refused() -> None:
    windows = people_windows((1.0,), per_class=2)
    with pytest.raises(NeurOSDataError, match="right_hand"):
        windows_to_tensors(windows, ("left_hand", "feet"))
    for scale in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(NeurOSConfigError):
            windows_to_tensors(windows, CLASSES, scale=scale)


def test_input_rms_matches_the_definition(monkeypatch: pytest.MonkeyPatch) -> None:
    windows = people_windows((0.5, 3.0), per_class=5)
    expected = float(np.sqrt(np.mean(windows.data.astype(np.float64) ** 2)))
    monkeypatch.setattr(data_module, "_RMS_CHUNK", 7)  # several uneven chunks
    assert input_rms(windows) == pytest.approx(expected, rel=1e-12)


def test_input_rms_rejects_silence() -> None:
    windows = people_windows((1.0,), per_class=2)
    with pytest.raises(NeurOSDataError, match="no signal"):
        input_rms(windows.with_data(np.zeros_like(windows.data)))


def test_tensor_data_validation() -> None:
    x = torch.zeros(4, 2, 8)
    y = torch.tensor([0, 1, 0, 1])
    people = ("a",) * 4
    TensorData(x=x, y=y, people=people, classes=CLASSES)
    bad_cases = [
        {"x": torch.zeros(4, 8)},
        {"x": torch.zeros(4, 2, 8, dtype=torch.float64)},
        {"x": torch.zeros(0, 2, 8), "y": torch.zeros(0, dtype=torch.int64), "people": ()},
        {"y": torch.tensor([0, 1, 0])},
        {"y": torch.tensor([0.0, 1.0, 0.0, 1.0])},
        {"y": torch.tensor([0, 1, 0, 2])},
        {"people": ("a",) * 3},
        {"classes": ("left_hand",)},
        {"classes": ("left_hand", "left_hand")},
    ]
    for case in bad_cases:
        arguments = {"x": x, "y": y, "people": people, "classes": CLASSES, **case}
        with pytest.raises(NeurOSDataError):
            TensorData(**arguments)  # type: ignore[arg-type]
