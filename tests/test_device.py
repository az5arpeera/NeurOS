"""Training devices (D-032): choosing CPU/GPU, determinism, provenance.

Most tests fake CUDA's presence, so they run on any machine. The last test trains on a real GPU
and is skipped where there isn't one (run it on the AWS box after setup).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neuros.config.experiment import EEGNetConfig, TrainConfig  # noqa: E402
from neuros.errors import NeurOSConfigError  # noqa: E402
from neuros.models.eegnet import EEGNetClassifier  # noqa: E402
from neuros.train.data import TensorData  # noqa: E402
from neuros.train.device import (  # noqa: E402
    check_device_string,
    configure_determinism,
    describe_device,
    resolve_device,
)
from neuros.train.trainer import model_device, predict_logits, train_classifier  # noqa: E402


@pytest.fixture
def restore_torch_flags() -> Iterator[None]:
    """configure_determinism changes process-wide torch flags; put them back."""
    saved = (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    yield
    torch.use_deterministic_algorithms(saved[0])
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = saved[1], saved[2]


def _fake_cuda(monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: count > 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)


def test_device_strings() -> None:
    for good in ("cpu", "auto", "cuda", "cuda:0", "cuda:12"):
        assert check_device_string(good) == good
    for bad in ("gpu", "cuda:", "cuda:x", "CUDA", "", "cpu:0", "mps"):
        with pytest.raises(NeurOSConfigError, match="device must be"):
            check_device_string(bad)


def test_cpu_and_auto_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda(monkeypatch, 0)
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto") == torch.device("cpu")
    # Asking for CUDA where there is none is an error, never a silent CPU run.
    with pytest.raises(NeurOSConfigError, match="ml-cuda"):
        resolve_device("cuda")
    with pytest.raises(NeurOSConfigError, match="CUDA isn't available"):
        resolve_device("cuda:0")


def test_folds_spread_over_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda(monkeypatch, 2)
    assert [resolve_device("cuda", worker_index=i).index for i in range(4)] == [0, 1, 0, 1]
    assert resolve_device("auto", worker_index=3) == torch.device("cuda", 1)
    assert resolve_device("cuda:1", worker_index=0) == torch.device("cuda", 1)  # pinned
    with pytest.raises(NeurOSConfigError, match="only 2 GPU"):
        resolve_device("cuda:2")
    with pytest.raises(NeurOSConfigError, match="worker_index"):
        resolve_device("cuda", worker_index=-1)


def test_determinism_is_a_cpu_no_op(
    monkeypatch: pytest.MonkeyPatch, restore_torch_flags: None
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    configure_determinism(torch.device("cpu"), True)
    import os

    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_cuda_determinism_settings(
    monkeypatch: pytest.MonkeyPatch, restore_torch_flags: None
) -> None:
    import os

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    cuda = torch.device("cuda", 0)  # only flags are set; no CUDA call is made
    configure_determinism(cuda, True)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.are_deterministic_algorithms_enabled()
    assert (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark) == (True, False)
    configure_determinism(cuda, False)
    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.benchmark
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":0:0")
    with pytest.raises(NeurOSConfigError, match="isn't a deterministic setting"):
        configure_determinism(cuda, True)


def test_describe_cpu() -> None:
    info = describe_device(torch.device("cpu"), deterministic=False)
    assert info["device"] == "cpu"
    assert info["deterministic"] is True  # CPU runs are deterministic for a fixed thread count
    assert info["threads"] >= 1


def _toy_data(n_people: int = 4, per_person: int = 16, seed: int = 0) -> TensorData:
    """Separable two-class windows: class 1 has extra power on channel 0."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_people * per_person, 4, 128)).astype(np.float32)
    y = np.tile(np.arange(2), n_people * per_person // 2)
    x[y == 1, 0] *= 3.0
    people = tuple(f"Toy:{i // per_person}" for i in range(len(y)))
    return TensorData(torch.from_numpy(x), torch.from_numpy(y), people, ("left_hand", "right_hand"))


def _tiny_model() -> EEGNetClassifier:
    torch.manual_seed(0)
    config = EEGNetConfig(kernel_samples=32, f1=4, depth=2, f2=8, embedding_dim=8)
    return EEGNetClassifier(4, 128, ("left_hand", "right_hand"), config)


def test_training_with_an_explicit_cpu_device() -> None:
    data = _toy_data()
    model = _tiny_model()
    config = TrainConfig(max_epochs=3, people_per_batch=2, windows_per_person=4, log_every=0)
    train_classifier(model, data, None, config, seed=1, device=torch.device("cpu"))
    assert model_device(model) == torch.device("cpu")
    logits = predict_logits(model, data.x, batch_size=32)
    assert logits.device == torch.device("cpu")
    assert logits.shape == (len(data.people), 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU (run on AWS)")
def test_gpu_training_matches_cpu_closely(restore_torch_flags: None) -> None:
    data = _toy_data()
    config = TrainConfig(max_epochs=5, people_per_batch=2, windows_per_person=4, log_every=0)
    cpu_model, gpu_model = _tiny_model(), _tiny_model()
    train_classifier(cpu_model, data, None, config, seed=3, device=torch.device("cpu"))
    gpu = resolve_device("cuda")
    configure_determinism(gpu, True)
    train_classifier(gpu_model, data, None, config, seed=3, device=gpu)
    assert model_device(gpu_model).type == "cuda"
    cpu_logits = predict_logits(cpu_model, data.x, 32)
    gpu_logits = predict_logits(gpu_model, data.x, 32)
    assert gpu_logits.device == torch.device("cpu")
    # Same seed, same batches: different kernels change the last bits, not the outcome.
    assert torch.allclose(cpu_logits, gpu_logits, atol=1e-2, rtol=1e-2)
    again = _tiny_model()
    train_classifier(again, data, None, config, seed=3, device=gpu)
    assert torch.equal(predict_logits(again, data.x, 32), gpu_logits)  # deterministic on CUDA
