"""From ``WindowSet`` to tensors: class indices, input scaling, validated boundaries.

**Input scale.** Networks train best on inputs of order 1. EA already delivers that: after
alignment each person's mean covariance is I, so the RMS is ≈ 1. Raw µV windows have an RMS
around 10, which batch norm would mostly absorb, but weight decay and the max-norm constraints
are tuned for unit-scale inputs. So every input is divided by one number, the RMS of the
*training* windows: fit on training people only (principle 2: no test-person data touches
normalisation fitting), then applied to validation and test windows unchanged. It's a single
global scalar, so it can't leak per-person information either way.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from neuros.errors import NeurOSConfigError, NeurOSDataError
from neuros.signal.windows import WindowSet

# Windows per chunk when computing the RMS: bounds the float64 working memory (~50 MB for
# 26 channels × 256 samples) however large the pooled training set gets.
_RMS_CHUNK = 1024


@dataclass(frozen=True, eq=False)
class TensorData:
    """Windows as tensors for training or evaluation, with per-window person ids.

    Attributes:
        x: ``[n, channels, samples]`` float32.
        y: ``[n]`` int64 class indices into ``classes``.
        people: ``dataset:subject`` per window (for per-person metrics and the sampler).
        classes: class names; index i is logit i.
        allowed: optional ``[n, len(classes)]`` bool: the classes each window competes among
            (D-039's extra classes: a window's softmax runs over its own dataset's classes;
            validation and test windows over the scored labels only). None = every class.
        primary: optional ``[n]`` bool: True for one window per trial, False for the extra
            training crops of a trial (D-039). An epoch counts primary windows, so crops
            don't add steps. None = every window is primary.
    """

    x: torch.Tensor
    y: torch.Tensor
    people: tuple[str, ...]
    classes: tuple[str, ...]
    allowed: torch.Tensor | None = None
    primary: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.x, torch.Tensor) or self.x.dim() != 3:
            raise NeurOSDataError("x must be a [n, channels, samples] tensor")
        if self.x.dtype != torch.float32:
            raise NeurOSDataError(f"x must be float32, got {self.x.dtype}")
        n = int(self.x.shape[0])
        if n == 0:
            raise NeurOSDataError("no windows")
        if not isinstance(self.y, torch.Tensor) or self.y.shape != (n,):
            raise NeurOSDataError(f"y must be a [{n}] tensor of class indices")
        if self.y.dtype != torch.int64:
            raise NeurOSDataError(f"y must be int64 class indices, got {self.y.dtype}")
        if len(self.classes) < 2 or len(set(self.classes)) != len(self.classes):
            raise NeurOSDataError(f"classes must be >= 2 distinct names, got {self.classes}")
        if int(self.y.min()) < 0 or int(self.y.max()) >= len(self.classes):
            raise NeurOSDataError(f"class indices must lie in [0, {len(self.classes)})")
        if len(self.people) != n:
            raise NeurOSDataError(f"{n} windows but {len(self.people)} person ids")
        if self.allowed is not None:
            mask = self.allowed
            if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
                raise NeurOSDataError("allowed must be a bool tensor")
            if mask.shape != (n, len(self.classes)):
                raise NeurOSDataError(
                    f"allowed must be [{n}, {len(self.classes)}], got {tuple(mask.shape)}"
                )
            # Every window must be allowed its own class, and compete with at least one other:
            # a softmax over one class is constant and teaches nothing.
            if not bool(mask[torch.arange(n), self.y].all()):
                raise NeurOSDataError("a window's own class is masked out (allowed)")
            if int(mask.sum(dim=1).min()) < 2:
                raise NeurOSDataError("every window must be allowed at least 2 classes")
        if self.primary is not None:
            if not isinstance(self.primary, torch.Tensor) or self.primary.dtype != torch.bool:
                raise NeurOSDataError("primary must be a bool tensor")
            if self.primary.shape != (n,):
                raise NeurOSDataError(f"primary must be [{n}], got {tuple(self.primary.shape)}")
            if not bool(self.primary.any()):
                raise NeurOSDataError("no primary window: every trial needs one")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_primary(self) -> int:
        """Windows that count towards an epoch: one per trial (all, without ``primary``)."""
        return len(self) if self.primary is None else int(self.primary.sum())

    @property
    def n_channels(self) -> int:
        return int(self.x.shape[1])

    @property
    def n_samples(self) -> int:
        return int(self.x.shape[2])

    def labels(self) -> list[str]:
        """Class names per window."""
        return [self.classes[i] for i in self.y.tolist()]


def input_rms(windows: WindowSet) -> float:
    """Root mean square over every sample of every window (float64 accumulation, chunked).

    Raises:
        NeurOSDataError: if there are no windows or the data is all zeros.
    """
    if len(windows) == 0:
        raise NeurOSDataError("can't compute an input scale from zero windows")
    total = 0.0
    for start in range(0, len(windows), _RMS_CHUNK):
        block = windows.data[start : start + _RMS_CHUNK].astype(np.float64)
        total += float(np.sum(block * block))
    rms = math.sqrt(total / windows.data.size)
    if not (math.isfinite(rms) and rms > 0):
        raise NeurOSDataError(f"input RMS is {rms}: the training windows carry no signal")
    return rms


def windows_to_tensors(
    windows: WindowSet, classes: Sequence[str], *, scale: float = 1.0
) -> TensorData:
    """Convert windows to ``TensorData``, dividing every sample by ``scale``.

    Raises:
        NeurOSConfigError: if ``scale`` isn't a positive finite number.
        NeurOSDataError: if a window's label isn't in ``classes`` (never dropped silently).
    """
    if not (isinstance(scale, int | float) and math.isfinite(scale) and scale > 0):
        raise NeurOSConfigError(f"scale must be a positive finite number, got {scale!r}")
    names = tuple(str(c) for c in classes)
    index = {name: i for i, name in enumerate(names)}
    unknown = sorted(set(windows.labels) - set(index))
    if unknown:
        raise NeurOSDataError(f"labels {unknown} are not among the classes {list(names)}")
    # The division makes a fresh, writable float32 array (WindowSet's data is read-only).
    x = np.asarray(windows.data / np.float32(scale), dtype=np.float32)
    y = np.array([index[label] for label in windows.labels], dtype=np.int64)
    return TensorData(
        x=torch.from_numpy(x),
        y=torch.from_numpy(y),
        people=windows.person_ids,
        classes=names,
    )
