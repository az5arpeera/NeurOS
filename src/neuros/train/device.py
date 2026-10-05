"""Where training runs: CPU or an NVIDIA GPU (D-032: training moves to AWS GPUs).

**Choosing.** ``resolve_device("cpu" | "cuda" | "cuda:N" | "auto")``. Asking for CUDA on a
machine without it is an error, never a silent fall-back to the CPU: a run that was meant to
take 20 minutes on a GPU would take hours, and its result file would claim the wrong hardware.
``"auto"`` is the only choice that may pick either.

**Several GPUs.** Fold workers pass their fold index; ``"cuda"`` then spreads folds over the
visible GPUs round-robin (``cuda:{index % count}``). One GPU can also host several fold workers:
EEGNet is small, so one process alone leaves most of a GPU idle.

**Determinism.** On CUDA, cuDNN's autotuner and some kernels are nondeterministic, so two runs
with the same seed can differ in the last bits and then diverge. ``configure_determinism`` turns
on PyTorch's deterministic algorithms (slower, reproducible). It must run before the process's
first cuBLAS call, because cuBLAS reads ``CUBLAS_WORKSPACE_CONFIG`` when it starts; the variable
is set for this process only. CPU runs are already deterministic for a fixed thread count.

**Provenance.** ``describe_device`` returns what a result file records: device name, CUDA and
cuDNN versions, memory, and whether deterministic algorithms were on.
"""

from __future__ import annotations

import os
import re
from typing import Any

import torch

from neuros.errors import NeurOSConfigError

# Accepted device strings: "cpu", "auto", "cuda" or "cuda:<index>".
_DEVICE_RE = re.compile(r"^(cpu|auto|cuda(?::(?P<index>\d+))?)$")

# cuBLAS needs a fixed workspace to be deterministic (PyTorch's reproducibility notes).
_CUBLAS_WORKSPACE = ":4096:8"


def check_device_string(choice: str) -> str:
    """Validate a device string without touching the hardware; return it unchanged.

    Raises:
        NeurOSConfigError: for anything but cpu, auto, cuda or cuda:<index>.
    """
    if not isinstance(choice, str) or _DEVICE_RE.match(choice) is None:
        raise NeurOSConfigError(f"device must be cpu, auto, cuda or cuda:<index>, got {choice!r}")
    return choice


def resolve_device(choice: str, *, worker_index: int = 0) -> torch.device:
    """The torch device for ``choice`` (see the module docstring).

    Args:
        choice: "cpu", "auto", "cuda" or "cuda:<index>".
        worker_index: spreads plain "cuda" (and "auto" on a GPU machine) over the visible GPUs.

    Raises:
        NeurOSConfigError: on a malformed choice, CUDA requested where it isn't available, or a
            GPU index that doesn't exist.
    """
    match = _DEVICE_RE.match(check_device_string(choice))
    assert match is not None  # checked above
    if choice == "cpu":
        return torch.device("cpu")
    available = torch.cuda.is_available()
    if choice == "auto" and not available:
        return torch.device("cpu")
    if not available:
        raise NeurOSConfigError(
            f"device {choice!r} asked for, but CUDA isn't available here (torch "
            f"{torch.__version__}, built for CUDA {torch.version.cuda}). Install the GPU build "
            "with `uv sync --extra dev --extra ml-cuda` on a GPU machine, or use --device cpu."
        )
    count = torch.cuda.device_count()
    if worker_index < 0:
        raise NeurOSConfigError(f"worker_index must be >= 0, got {worker_index}")
    index = int(match.group("index")) if match.group("index") is not None else worker_index % count
    if index >= count:
        raise NeurOSConfigError(f"device {choice!r}: only {count} GPU(s) visible")
    return torch.device("cuda", index)


def configure_determinism(device: torch.device, deterministic: bool) -> None:
    """Make CUDA runs reproducible (or fast). A no-op on the CPU.

    Call it in each process before the first CUDA operation.

    Raises:
        NeurOSConfigError: if deterministic mode is asked for after cuBLAS already started with
            a different workspace setting (the results wouldn't be reproducible).
    """
    if device.type != "cuda":
        return
    if deterministic:
        current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if current is None:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE  # this process only
        elif current not in (_CUBLAS_WORKSPACE, ":16:8"):
            raise NeurOSConfigError(
                f"CUBLAS_WORKSPACE_CONFIG={current!r} isn't a deterministic setting "
                "(use :4096:8 or :16:8)"
            )
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=False)


def describe_device(device: torch.device, *, deterministic: bool) -> dict[str, Any]:
    """What a result file records about the hardware a run used."""
    info: dict[str, Any] = {
        "device": str(device),
        "torch": torch.__version__,
        "deterministic": bool(deterministic) if device.type == "cuda" else True,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        info.update(
            {
                "gpu": properties.name,
                "gpu_memory_gb": round(properties.total_memory / 1e9, 1),
                "compute_capability": f"{properties.major}.{properties.minor}",
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),  # type: ignore[no-untyped-call]
            }
        )
    else:
        info["threads"] = torch.get_num_threads()
    return info
