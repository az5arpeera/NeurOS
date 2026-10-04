"""EEGNet encoder and the linear head (architecture §6.1-6.2).

EEGNet (Lawhern et al. 2018, J. Neural Eng. 15:056013) is the compact convolutional network
most cross-person motor-imagery papers build on, including the EA + EEGNet + SWA result that
E0 checks (Sagayaraju 2026: 74.3 % leave-one-person-out on BNCI2014-001 left/right). It's
commodity (architecture §1): our claims sit in the runtime around it, not in the network.

Layer by layer, for a window of C channels × T samples at 128 Hz (defaults F1 = 16, D = 2,
F2 = 32, temporal kernel 64 samples = 0.5 s):

    input                                             [B, C, T]
    add a feature axis                                [B, 1, C, T]
    temporal conv, F1 kernels (1 × 64), same length   [B, F1, C, T]      a learned filter bank
    batch norm
    spatial conv, D per temporal map (C × 1)          [B, F1·D, 1, T]    CSP-like spatial filters
    batch norm, ELU, average pool (1 × 4), dropout    [B, F1·D, 1, T/4]
    separable conv: depthwise (1 × 16) + pointwise    [B, F2, 1, T/4]    temporal summaries
    batch norm, ELU, average pool (1 × 8), dropout    [B, F2, 1, T/32]
    flatten, linear                                   [B, 64]            the embedding z

Why this shape works for motor imagery: the temporal kernels learn band-pass filters (mu and
beta), the spatial kernels learn per-band spatial patterns over the motor cortex (like CSP),
and pooling turns the result into band power over time, which is what ERD/ERS changes.

**The 64-d embedding z is ours** (EEGNet maps straight to class scores). It's the interface
every head reads (§6.2): ``LinearHead`` now; the text-anchored head and the contrastive
projection later. Two linear maps in a row (flatten → z → logits) can express exactly what
EEGNet's single classifier layer can, so this costs nothing in capacity.

**Fidelity to EEGNet**, so E0 tests our pipeline rather than a different network:
- batch norm with Keras's momentum (0.99, i.e. PyTorch momentum 0.01) and epsilon (1e-3);
- Glorot-uniform weights and zero biases (Keras's defaults);
- no biases in the convolutions (each is followed by batch norm, which has its own shift);
- max-norm constraints: each spatial filter's weights have L2 norm ≤ 1 and each class's head
  weights ≤ 0.25. They're enforced by ``apply_max_norm`` after every optimiser step (the
  trainer calls it), not inside ``forward``, so inference is a pure function of the weights.

**Boundary checks:** ``forward`` validates shape, dtype and batch size (cheap; no pass over
the data). Finiteness is checked where data enters (``WindowSet`` rejects NaN/Inf; the
engine's quality gate online), not on every forward pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from neuros.config.experiment import EEGNetConfig
from neuros.errors import NeurOSConfigError, NeurOSDataError

# Keras BatchNormalization defaults, as in the original EEGNet: running statistics move 1 % per
# batch (Keras momentum 0.99 = PyTorch momentum 0.01), and epsilon is 1e-3.
_BN_MOMENTUM = 0.01
_BN_EPS = 1e-3


def same_padding(kernel: int) -> tuple[int, int]:
    """(left, right) zero padding that keeps the length for a stride-1 kernel.

    For an even kernel the extra sample goes on the right, as in Keras's and PyTorch's
    ``padding="same"``. Padding explicitly (instead of ``padding="same"``) avoids PyTorch's
    warning about an internal copy for even kernels; the result is identical.
    """
    total = kernel - 1
    return total // 2, total - total // 2


def output_samples(n_samples: int, config: EEGNetConfig) -> int:
    """Time steps left after both average pools (each floors, as PyTorch's pooling does)."""
    return (n_samples // config.pool1) // config.pool2


def check_window_batch(x: object, n_channels: int, n_samples: int, dtype: torch.dtype) -> None:
    """Raise unless ``x`` is a non-empty ``[batch, n_channels, n_samples]`` tensor of ``dtype``.

    Raises:
        NeurOSDataError: naming the expected and actual shape or dtype.
    """
    if not isinstance(x, torch.Tensor):
        raise NeurOSDataError(f"expected a torch.Tensor of windows, got {type(x).__name__}")
    if x.dim() != 3 or x.shape[1] != n_channels or x.shape[2] != n_samples:
        raise NeurOSDataError(
            f"expected windows [batch, {n_channels} channels, {n_samples} samples], "
            f"got shape {tuple(x.shape)}"
        )
    if x.shape[0] == 0:
        raise NeurOSDataError("empty batch: no windows to process")
    if x.dtype != dtype:
        raise NeurOSDataError(f"windows are {x.dtype}, but the model's weights are {dtype}")


@torch.no_grad()
def _renorm_(weight: torch.Tensor, max_norm: float) -> None:
    """Scale each output unit's weights (along dim 0) down to L2 norm ≤ ``max_norm``, in place.

    ``torch.renorm`` leaves units already within the limit untouched (it isn't a projection
    onto the sphere), which is exactly Keras's ``max_norm`` constraint.
    """
    weight.copy_(torch.renorm(weight, p=2, dim=0, maxnorm=max_norm))


class EEGNetEncoder(nn.Module):
    """EEGNet's convolutional trunk plus a linear map to the embedding z (§6.1)."""

    def __init__(self, n_channels: int, n_samples: int, config: EEGNetConfig | None = None):
        super().__init__()
        cfg = config or EEGNetConfig()
        if not isinstance(n_channels, int) or n_channels < 1:
            raise NeurOSConfigError(f"n_channels must be a positive int, got {n_channels!r}")
        if not isinstance(n_samples, int) or n_samples < 1:
            raise NeurOSConfigError(f"n_samples must be a positive int, got {n_samples!r}")
        t_out = output_samples(n_samples, cfg)
        if t_out < 1:
            raise NeurOSConfigError(
                f"{n_samples}-sample windows are too short for pools of {cfg.pool1} × "
                f"{cfg.pool2} (need at least {cfg.pool1 * cfg.pool2} samples)"
            )
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.config = cfg
        temporal_maps = cfg.f1
        spatial_maps = cfg.f1 * cfg.depth

        # Block 1: temporal filter bank, then depthwise spatial filters (D per temporal map).
        self._temporal_pad = same_padding(cfg.kernel_samples)
        self.temporal = nn.Conv2d(1, temporal_maps, (1, cfg.kernel_samples), bias=False)
        self.bn_temporal = nn.BatchNorm2d(temporal_maps, momentum=_BN_MOMENTUM, eps=_BN_EPS)
        self.spatial = nn.Conv2d(
            temporal_maps, spatial_maps, (n_channels, 1), groups=temporal_maps, bias=False
        )
        self.bn_spatial = nn.BatchNorm2d(spatial_maps, momentum=_BN_MOMENTUM, eps=_BN_EPS)
        self.pool1 = nn.AvgPool2d((1, cfg.pool1))
        self.dropout1 = nn.Dropout(cfg.dropout)

        # Block 2: separable convolution = per-map temporal kernel, then 1×1 mixing to F2 maps.
        self._separable_pad = same_padding(cfg.separable_kernel_samples)
        self.separable_depthwise = nn.Conv2d(
            spatial_maps,
            spatial_maps,
            (1, cfg.separable_kernel_samples),
            groups=spatial_maps,
            bias=False,
        )
        self.separable_pointwise = nn.Conv2d(spatial_maps, cfg.f2, 1, bias=False)
        self.bn_separable = nn.BatchNorm2d(cfg.f2, momentum=_BN_MOMENTUM, eps=_BN_EPS)
        self.pool2 = nn.AvgPool2d((1, cfg.pool2))
        self.dropout2 = nn.Dropout(cfg.dropout)
        self.elu = nn.ELU()

        # Ours: flatten → 64-d embedding z, the interface every head reads (§6.2).
        self.embed = nn.Linear(cfg.f2 * t_out, cfg.embedding_dim)
        _glorot_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Windows ``[B, C, T]`` (float32) → embeddings ``[B, embedding_dim]``."""
        check_window_batch(x, self.n_channels, self.n_samples, self.temporal.weight.dtype)
        h: torch.Tensor = x.unsqueeze(1)  # [B, 1, C, T]
        h = self.bn_temporal(self.temporal(F.pad(h, self._temporal_pad)))  # [B, F1, C, T]
        h = self.elu(self.bn_spatial(self.spatial(h)))  # [B, F1·D, 1, T]
        h = self.dropout1(self.pool1(h))  # [B, F1·D, 1, T/4]
        h = self.separable_depthwise(F.pad(h, self._separable_pad))
        h = self.separable_pointwise(h)  # [B, F2, 1, T/4]
        h = self.elu(self.bn_separable(h))
        h = self.dropout2(self.pool2(h))  # [B, F2, 1, T/32]
        z: torch.Tensor = self.embed(h.flatten(1))  # [B, embedding_dim]
        return z

    def apply_max_norm(self) -> None:
        """Enforce the spatial filters' max-norm constraint (call after each optimiser step)."""
        if self.config.spatial_max_norm is not None:
            _renorm_(self.spatial.weight, self.config.spatial_max_norm)


class LinearHead(nn.Module):
    """``logits = W·z + b`` (§6.2, the baseline head)."""

    def __init__(self, embedding_dim: int, n_classes: int, max_norm: float | None = None):
        super().__init__()
        if embedding_dim < 1 or n_classes < 2:
            raise NeurOSConfigError(
                f"a head needs embedding_dim >= 1 and n_classes >= 2, "
                f"got {embedding_dim} and {n_classes}"
            )
        if max_norm is not None and not max_norm > 0:
            raise NeurOSConfigError(f"max_norm must be positive or None, got {max_norm}")
        self.embedding_dim = embedding_dim
        self.n_classes = n_classes
        self.max_norm = max_norm
        self.linear = nn.Linear(embedding_dim, n_classes)
        _glorot_init(self)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Embeddings ``[B, embedding_dim]`` → logits ``[B, n_classes]``."""
        if not isinstance(z, torch.Tensor) or z.dim() != 2 or z.shape[1] != self.embedding_dim:
            raise NeurOSDataError(
                f"expected embeddings [batch, {self.embedding_dim}], "
                f"got {tuple(z.shape) if isinstance(z, torch.Tensor) else type(z).__name__}"
            )
        logits: torch.Tensor = self.linear(z)
        return logits

    def apply_max_norm(self) -> None:
        """Enforce each class's max-norm constraint (call after each optimiser step)."""
        if self.max_norm is not None:
            _renorm_(self.linear.weight, self.max_norm)


class EEGNetClassifier(nn.Module):
    """EEGNet encoder + linear head: windows → class logits, in the order of ``classes``."""

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        classes: Sequence[str],
        config: EEGNetConfig | None = None,
    ):
        super().__init__()
        names = tuple(str(c) for c in classes)
        if len(names) < 2 or len(set(names)) != len(names):
            raise NeurOSConfigError(f"classes must be >= 2 distinct names, got {list(classes)}")
        cfg = config or EEGNetConfig()
        self.classes = names
        self.config = cfg
        self.encoder = EEGNetEncoder(n_channels, n_samples, cfg)
        self.head = LinearHead(cfg.embedding_dim, len(names), max_norm=cfg.head_max_norm)

    @property
    def n_channels(self) -> int:
        return self.encoder.n_channels

    @property
    def n_samples(self) -> int:
        return self.encoder.n_samples

    @property
    def n_parameters(self) -> int:
        """Trainable parameters (§6.1 budget: tens of thousands, < 1 MB)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Windows ``[B, C, T]`` → logits ``[B, n_classes]``."""
        logits: torch.Tensor = self.head(self.encoder(x))
        return logits

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Windows ``[B, C, T]`` → embeddings z ``[B, embedding_dim]``."""
        z: torch.Tensor = self.encoder(x)
        return z

    def apply_max_norm(self) -> None:
        """Enforce every max-norm constraint (the trainer calls this after each step)."""
        self.encoder.apply_max_norm()
        self.head.apply_max_norm()

    def describe(self) -> dict[str, Any]:
        """JSON-serialisable description for result files and model bundles."""
        return {
            "architecture": "EEGNet + linear head",
            "n_channels": self.n_channels,
            "n_samples": self.n_samples,
            "classes": list(self.classes),
            "n_parameters": self.n_parameters,
            **self.config.model_dump(),
        }


def _glorot_init(module: nn.Module) -> None:
    """Keras's defaults: Glorot-uniform weights and zero biases for convolutions and linears."""
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d | nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
