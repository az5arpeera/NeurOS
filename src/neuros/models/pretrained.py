"""Pretrained encoders for A6: REVE and MIRepNet (``knowledge-base/plan/a6-...``).

A6 asks whether starting from a published, pretrained encoder beats our EEGNet trained from
scratch. Both encoders are **re-implemented here** from their authors' code, and only their
published *weights* are loaded:

- **REVE-base** (El Ouahidi et al., NeurIPS 2025): weights ``brain-bzh/reve-base`` at revision
  ``dc2a075`` (a ``.safetensors`` file: a JSON header plus raw tensors; nothing in it executes),
  under the REVE Responsible Use License v1.0 (``data/pretrained/reve-base/LICENSE``: commercial
  use and redistribution of fine-tuned copies allowed with the licence, attribution and a
  "modified from REVE" notice; no identifying people from EEG). Architecture from that
  repository's ``modeling_reve.py``; the classifier head and fine-tuning recipe from the
  authors' MIT-licensed code (``github.com/elouayas/reve_eeg`` at ``06a7059``).
- **MIRepNet** (Liu et al., Knowledge-Based Systems 2026): code and weights under MIT
  (``github.com/staraink/MIRepNet`` at ``b35d113``; weights ``starself/MIRepNet`` at ``9bac043``),
  a tensors-only torch pickle opened with ``torch.load(weights_only=True)``.

No downloaded code is imported or run. ``load_pretrained`` loads strictly: every published
tensor must find a parameter of the same shape here, and every encoder parameter must be
published, so a porting mistake in a name or shape fails loudly.

**The interface is ``EEGNetClassifier``'s**, so the trainer, the pooled evaluation and the engine
take either model unchanged: windows ``[B, C, T]`` in the model's layout at the contract's rate
(128 Hz), already common-averaged, aligned (EA) and divided by the training input scale; logits
``[B, n_classes]`` in the order of ``classes``. Each model resamples internally to the rate it
was pretrained at (REVE 200 Hz, MIRepNet 250 Hz) with band-limited FFT resampling.

**Absent channels.** Our placement zero-fills the layout rows a headset or dataset lacks
(``eval.pooled.to_layout`` offline, ``engine.pipeline.place_window`` live). A real channel after
the common average and alignment is never exactly zero across a whole window, so a row of zeros
marks an absent channel. REVE leaves absent channels out of attention entirely (it reads
electrode positions, so it needs no fixed layout); MIRepNet interpolates them from the present
ones by inverse distance, as its own pipeline fills the template channels a dataset lacks.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from neuros.errors import NeurOSConfigError, NeurOSDataError
from neuros.models.eegnet import check_window_batch

# ---------------------------------------------------------------------- weights and positions

# safetensors dtype codes we accept (the published files are float32 throughout).
_SAFETENSORS_DTYPES: dict[str, np.dtype[Any]] = {
    "F32": np.dtype("<f4"),
    "F64": np.dtype("<f8"),
    "I64": np.dtype("<i8"),
}
# A header larger than this is not a plausible model header (REVE-base's is ~13 KB).
_MAX_HEADER_BYTES = 10_000_000


def load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    """Read every tensor of a ``.safetensors`` file, checking the header against the file.

    The format is an 8-byte little-endian header length, a JSON header (each tensor's dtype,
    shape and byte range) and the raw bytes. Nothing executes, unlike a pickle.

    Raises:
        NeurOSConfigError: if the file is missing.
        NeurOSDataError: on a truncated or inconsistent file, or an unsupported dtype.
    """
    if not path.is_file():
        raise NeurOSConfigError(f"weights not found: {path} (see knowledge-base/plan/a6-...)")
    raw = path.read_bytes()
    if len(raw) < 8:
        raise NeurOSDataError(f"{path}: too short to be a safetensors file")
    (header_len,) = struct.unpack("<Q", raw[:8])
    if not 2 <= header_len <= min(_MAX_HEADER_BYTES, len(raw) - 8):
        raise NeurOSDataError(f"{path}: implausible header length {header_len}")
    try:
        header = json.loads(raw[8 : 8 + header_len])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NeurOSDataError(f"{path}: unreadable header ({exc})") from exc
    if not isinstance(header, dict):
        raise NeurOSDataError(f"{path}: the header is not a JSON object")
    body = memoryview(raw)[8 + header_len :]
    tensors: dict[str, torch.Tensor] = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        try:
            dtype = _SAFETENSORS_DTYPES[spec["dtype"]]
            shape = [int(s) for s in spec["shape"]]
            start, end = (int(v) for v in spec["data_offsets"])
        except (KeyError, TypeError, ValueError) as exc:
            raise NeurOSDataError(
                f"{path}: tensor {name!r} has a bad header entry ({exc})"
            ) from exc
        count = math.prod(shape)
        if not (0 <= start <= end <= len(body)) or end - start != count * dtype.itemsize:
            raise NeurOSDataError(f"{path}: tensor {name!r}'s bytes don't match its shape")
        array = np.frombuffer(body[start:end], dtype=dtype).reshape(shape)
        tensors[name] = torch.from_numpy(array.copy())  # own the memory; the file buffer goes
    if not tensors:
        raise NeurOSDataError(f"{path}: no tensors")
    return tensors


def load_tensor_state(path: Path) -> dict[str, torch.Tensor]:
    """Open a torch state dict with ``weights_only=True`` (tensors and plain containers only;
    anything else in the pickle is refused, so no code in it can run).

    Raises:
        NeurOSConfigError: if the file is missing.
        NeurOSDataError: if it isn't a mapping of names to tensors.
    """
    if not path.is_file():
        raise NeurOSConfigError(f"weights not found: {path} (see knowledge-base/plan/a6-...)")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # torch raises several types for a refused or corrupt pickle
        raise NeurOSDataError(f"{path}: not a tensors-only state dict ({exc})") from exc
    if not isinstance(state, Mapping) or not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
    ):
        raise NeurOSDataError(f"{path}: expected a mapping of names to tensors")
    return dict(state)


def load_strict(module: nn.Module, state: Mapping[str, torch.Tensor], what: str) -> None:
    """Load ``state`` into ``module``, refusing any missing, extra or mis-shaped tensor.

    Raises:
        NeurOSDataError: naming the first mismatches.
    """
    own = module.state_dict()
    missing = sorted(set(own) - set(state))
    extra = sorted(set(state) - set(own))
    shapes = [k for k in own if k in state and tuple(own[k].shape) != tuple(state[k].shape)]
    if missing or extra or shapes:
        raise NeurOSDataError(
            f"{what}: the published weights don't match this implementation "
            f"(missing {missing[:5]}, unexpected {extra[:5]}, shape mismatches {shapes[:5]})"
        )
    module.load_state_dict({k: v.to(own[k].dtype) for k, v in state.items()}, strict=True)


def electrode_positions(names: Sequence[str]) -> torch.Tensor:
    """``[len(names), 3]`` electrode positions in metres (MNE's standard 10-05 montage, the
    coordinates REVE was pretrained with; ``positions_1005.json``).

    Raises:
        NeurOSConfigError: for a name the montage doesn't have.
    """
    table = json.loads(
        resources.files("neuros.models").joinpath("positions_1005.json").read_text("utf-8")
    )["positions"]
    unknown = [n for n in names if n not in table]
    if unknown:
        raise NeurOSConfigError(f"no 10-05 position for channels {unknown}")
    return torch.tensor([table[n] for n in names], dtype=torch.float32)


def fft_resample(x: torch.Tensor, n_out: int) -> torch.Tensor:
    """Resample along the last axis to ``n_out`` samples by zero-padding (or truncating) the
    spectrum: exact for band-limited periodic signals, like ``scipy.signal.resample``.

    Our windows are band-passed to 8-30 Hz, far below every Nyquist involved (64 Hz in, 100 or
    125 Hz out), so the only error is the window's edge discontinuity, the same as scipy's.

    Raises:
        NeurOSConfigError: on a non-positive length.
    """
    n_in = x.shape[-1]
    if n_out <= 0 or n_in <= 0:
        raise NeurOSConfigError(f"resampling needs positive lengths, got {n_in} -> {n_out}")
    if n_out == n_in:
        return x
    spectrum = torch.fft.rfft(x, dim=-1)
    n_bins = n_out // 2 + 1
    if n_bins > spectrum.shape[-1]:
        pad = torch.zeros(
            (*spectrum.shape[:-1], n_bins - spectrum.shape[-1]),
            dtype=spectrum.dtype,
            device=spectrum.device,
        )
        spectrum = torch.cat([spectrum, pad], dim=-1)
    else:
        spectrum = spectrum[..., :n_bins]
    resampled: torch.Tensor = torch.fft.irfft(spectrum, n=n_out, dim=-1) * (n_out / n_in)
    return resampled


def present_channels(x: torch.Tensor) -> torch.Tensor:
    """``[B, C]`` True where a window's channel row isn't all zeros (absent rows are zero-filled).

    Raises:
        NeurOSDataError: if some window has no present channel at all.
    """
    present = x.abs().amax(dim=-1) > 0
    if not bool(present.any(dim=-1).all()):
        raise NeurOSDataError("a window has no non-zero channel: nothing to decode")
    return present


def _target_samples(n_samples: int, sfreq: float, target_sfreq: float) -> int:
    """The window length at ``target_sfreq``; it must come out a whole number of samples."""
    exact = n_samples * target_sfreq / sfreq
    n_out = round(exact)
    if n_out < 1 or abs(exact - n_out) > 1e-6:
        raise NeurOSConfigError(
            f"a {n_samples}-sample window at {sfreq:g} Hz isn't a whole number of samples at "
            f"{target_sfreq:g} Hz ({exact:g})"
        )
    return n_out


class _PretrainedClassifier(nn.Module):
    """Shared plumbing: the ``EEGNetClassifier`` interface plus backbone/head parameter groups."""

    encoder_modules: tuple[str, ...] = ()  # attribute names of the pretrained part
    head: nn.Module  # set by each subclass: the new layer(s) that map embeddings to logits

    def __init__(self, layout: Sequence[str], n_samples: int, classes: Sequence[str]) -> None:
        super().__init__()
        if len(layout) < 1 or len(set(layout)) != len(layout):
            raise NeurOSConfigError(f"the layout must be distinct channel names, got {layout}")
        if len(classes) < 2 or len(set(classes)) != len(classes):
            raise NeurOSConfigError(f"need at least two distinct classes, got {list(classes)}")
        if n_samples < 1:
            raise NeurOSConfigError(f"n_samples must be positive, got {n_samples}")
        self.layout = tuple(layout)
        self.classes = tuple(classes)
        self._n_samples = int(n_samples)
        self.weights_source: str | None = None

    @property
    def n_channels(self) -> int:
        return len(self.layout)

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def apply_max_norm(self) -> None:
        """EEGNet's max-norm constraints don't apply here (the trainer calls this every step)."""

    def parameter_groups(self) -> list[tuple[str, list[nn.Parameter]]]:
        """``[("backbone", ...), ("head", ...)]``: the trainer gives the pretrained backbone its
        own learning-rate scale and can hold it frozen for the first epochs (a linear probe)."""
        backbone_ids: set[int] = set()
        backbone: list[nn.Parameter] = []
        for name in self.encoder_modules:
            for p in getattr(self, name).parameters():
                backbone_ids.add(id(p))
                backbone.append(p)
        head = [p for p in self.parameters() if id(p) not in backbone_ids]
        return [("backbone", backbone), ("head", head)]

    def _check(self, x: torch.Tensor) -> None:
        check_window_batch(x, self.n_channels, self.n_samples, torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.head(self.embed(x))
        return logits

    def embed(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - overridden
        raise NotImplementedError


# ---------------------------------------------------------------------- REVE


@dataclass(frozen=True)
class ReveDims:
    """REVE's hyperparameters (``brain-bzh/reve-base`` ``config.json``)."""

    embed_dim: int = 512
    depth: int = 22
    heads: int = 8
    head_dim: int = 64
    mlp_dim_ratio: float = 2.66
    freqs: int = 4  # Fourier frequencies per axis: 4^4 = 256 = embed_dim / 2
    patch_size: int = 200  # 1 s at 200 Hz
    patch_overlap: int = 20  # 0.1 s
    sfreq: float = 200.0  # every pretraining signal was resampled to 200 Hz


REVE_BASE = ReveDims()


class _RMSNorm(nn.Module):
    """RMS normalisation with a learned gain (REVE's ``RMSNorm``; computed in float32)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        return y.type_as(x) * self.weight


class _GEGLU(nn.Module):
    """``GELU(gates) * x`` over the two halves of the last axis (REVE's ``GEGLU``)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gates = x.chunk(2, dim=-1)
        return F.gelu(gates) * value


class _ReveFeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        # The indices match the published names: net.0 norm, net.1 in, net.3 out.
        self.net = nn.Sequential(
            _RMSNorm(dim),
            nn.Linear(dim, hidden * 2, bias=False),
            _GEGLU(),
            nn.Linear(hidden, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # type: ignore[no-any-return]


class _ReveAttention(nn.Module):
    """Pre-norm multi-head self-attention without biases (REVE's ``Attention``).

    Written out (rather than ``scaled_dot_product_attention``) so the key mask is explicit and
    the backward pass is deterministic on the GPU. The scale is ``head_dim ** -0.5``, as
    REVE's SDPA default.
    """

    def __init__(self, dim: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.norm = _RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, heads * head_dim * 3, bias=False)
        self.to_out = nn.Linear(heads * head_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        q, k, v = self.to_qkv(self.norm(x)).chunk(3, dim=-1)
        q, k, v = (t.view(b, n, self.heads, self.head_dim).transpose(1, 2) for t in (q, k, v))
        scores = (q @ k.transpose(-1, -2)) * self.head_dim**-0.5  # [B, H, N, N]
        scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        out = scores.softmax(dim=-1) @ v  # [B, H, N, D]
        return self.to_out(out.transpose(1, 2).reshape(b, n, self.heads * self.head_dim))  # type: ignore[no-any-return]


class _ReveBackbone(nn.Module):
    def __init__(self, dims: ReveDims) -> None:
        super().__init__()
        hidden = int(dims.embed_dim * dims.mlp_dim_ratio)  # 1361 for REVE-base, as published
        self.layers = nn.ModuleList(
            nn.ModuleList(
                [
                    _ReveAttention(dims.embed_dim, dims.heads, dims.head_dim),
                    _ReveFeedForward(dims.embed_dim, hidden),
                ]
            )
            for _ in range(dims.depth)
        )

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            attention, feed_forward = cast(nn.ModuleList, layer)
            x = attention(x, key_mask) + x
            x = feed_forward(x) + x
        return x


def fourier_4d(
    positions: torch.Tensor,
    dimension: int,
    freqs: int,
    increment_time: float = 0.1,
    margin: float = 0.4,
) -> torch.Tensor:
    """REVE's ``FourierEmb4D``: ``[B, N, 4]`` (x, y, z in metres; t in patches) → ``[B, N, dim]``.

    Every combination of ``freqs`` frequencies on the four axes (4^4 = 256 for REVE-base) gives
    a phase; the embedding is its cosine and sine. Time is scaled by ``increment_time`` and all
    axes shifted by ``margin`` exactly as published.

    Raises:
        NeurOSConfigError: if ``freqs ** 4`` is smaller than half the dimension.
    """
    if freqs**4 < dimension // 2:
        raise NeurOSConfigError(f"{freqs}^4 frequencies can't fill a {dimension}-d embedding")
    p = positions.clone()
    p[..., -1] = p[..., -1] * increment_time
    p = p + margin
    width = 1 + 2 * margin
    f = torch.arange(freqs, dtype=p.dtype, device=p.device) * (2 * math.pi / width)
    # loc[..., i, j, k, l] = x·f_i + y·f_j + z·f_k + t·f_l, flattened in (i, j, k, l) order.
    loc = (
        p[..., 0, None, None, None, None] * f[:, None, None, None]
        + p[..., 1, None, None, None, None] * f[None, :, None, None]
        + p[..., 2, None, None, None, None] * f[None, None, :, None]
        + p[..., 3, None, None, None, None] * f[None, None, None, :]
    )
    loc = loc.reshape(*p.shape[:-1], -1)[..., : dimension // 2]
    return torch.cat([torch.cos(loc), torch.sin(loc)], dim=-1)


class ReveEncoder(nn.Module):
    """REVE's encoder (``modeling_reve.Reve`` without ``transformers``): per-channel 1-s patches
    (0.1-s overlap), a 4-D position embedding per patch, a 22-layer transformer.

    Parameter names match the published weights exactly, so ``load_strict`` can check them.
    """

    def __init__(self, dims: ReveDims = REVE_BASE) -> None:
        super().__init__()
        self.dims = dims
        e = dims.embed_dim
        self.to_patch_embedding = nn.Sequential(nn.Linear(dims.patch_size, e))
        self.mlp4d = nn.Sequential(nn.Linear(4, e, bias=False), nn.GELU(), nn.LayerNorm(e))
        self.ln = nn.LayerNorm(e)
        self.transformer = _ReveBackbone(dims)
        self.cls_query_token = nn.Parameter(torch.randn(1, 1, e))

    def n_patches(self, n_samples: int) -> int:
        step = self.dims.patch_size - self.dims.patch_overlap
        if n_samples < self.dims.patch_size:
            raise NeurOSConfigError(
                f"REVE needs at least {self.dims.patch_size} samples "
                f"({self.dims.patch_size / self.dims.sfreq:g} s)"
            )
        return (n_samples - self.dims.patch_size) // step + 1

    def forward(
        self, eeg: torch.Tensor, positions: torch.Tensor, present: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``eeg [B, C, L]`` at 200 Hz, ``positions [B, C, 3]``, ``present [B, C]`` →
        ``(tokens [B, C·H, E], token_mask [B, C·H])``, channel-major like REVE."""
        b, c, _ = eeg.shape
        step = self.dims.patch_size - self.dims.patch_overlap
        patches = eeg.float().unfold(2, self.dims.patch_size, step)  # [B, C, H, P]
        h = patches.shape[2]
        times = torch.arange(h, dtype=positions.dtype, device=positions.device)
        pos4 = torch.cat(
            [
                positions[:, :, None, :].expand(b, c, h, 3),
                times[None, None, :, None].expand(b, c, h, 1),
            ],
            dim=-1,
        ).reshape(b, c * h, 4)
        pos_embed = self.ln(
            fourier_4d(pos4, self.dims.embed_dim, self.dims.freqs) + self.mlp4d(pos4)
        )
        tokens = self.to_patch_embedding(patches).reshape(b, c * h, -1) + pos_embed
        token_mask = present[:, :, None].expand(b, c, h).reshape(b, c * h)
        return self.transformer(tokens, token_mask), token_mask


class ReveClassifier(_PretrainedClassifier):
    """REVE-base + an attention-pooled linear head (the authors' ``ReveClassifier`` with
    ``pooling="last"``): a learned query attends over the present channels' tokens, then
    RMS norm, dropout and a linear layer give the class logits.

    The head reads one pooled vector, so it doesn't depend on how many channels a headset has
    (the authors' "no" pooling flattens every token and would). The query starts from REVE's
    own pretrained ``cls_query_token``.
    """

    encoder_modules = ("encoder",)
    positions: torch.Tensor  # [C, 3] electrode positions of the layout (a buffer)

    def __init__(
        self,
        layout: Sequence[str],
        n_samples: int,
        classes: Sequence[str],
        *,
        sfreq: float,
        dropout: float = 0.3,
        dims: ReveDims = REVE_BASE,
    ) -> None:
        super().__init__(layout, n_samples, classes)
        if not 0.0 <= dropout < 1.0:
            raise NeurOSConfigError(f"dropout must be in [0, 1), got {dropout}")
        self.sfreq = float(sfreq)
        self.resampled = _target_samples(n_samples, sfreq, dims.sfreq)
        self.encoder = ReveEncoder(dims)
        self.encoder.n_patches(self.resampled)  # fails early on windows shorter than a patch
        self.register_buffer("positions", electrode_positions(self.layout), persistent=False)
        self.head = nn.Sequential(
            _RMSNorm(dims.embed_dim), nn.Dropout(dropout), nn.Linear(dims.embed_dim, len(classes))
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, C, T]`` windows → ``[B, E]``: attention pooling over present channels' tokens."""
        self._check(x)
        present = present_channels(x)
        eeg = fft_resample(x, self.resampled)
        positions = self.positions.to(x.device).expand(x.shape[0], -1, -1)
        if bool((present == present[:1]).all()):
            # One presence pattern for the whole batch (always so live: one headset, one window
            # at a time): drop absent channels before the transformer instead of masking them.
            # The result is identical (masked keys never reach present tokens, and pooling
            # skips them), and a 3-channel headset costs a fraction of the full layout.
            keep = present[0]
            eeg, positions, present = eeg[:, keep], positions[:, keep], present[:, keep]
        tokens, token_mask = self.encoder(eeg, positions, present)
        query = self.encoder.cls_query_token.expand(x.shape[0], -1, -1)  # [B, 1, E]
        scores = (query @ tokens.transpose(-1, -2)) / tokens.shape[-1] ** 0.5  # [B, 1, N]
        scores = scores.masked_fill(~token_mask[:, None, :], float("-inf"))
        pooled: torch.Tensor = (scores.softmax(dim=-1) @ tokens).squeeze(1)
        return pooled

    def load_pretrained(self, path: Path) -> None:
        """Load REVE-base's published encoder weights (strict) from a ``.safetensors`` file."""
        load_strict(self.encoder, load_safetensors(path), f"REVE weights {path}")
        self.weights_source = str(path)

    def describe(self) -> dict[str, Any]:
        return {
            "architecture": "REVE-base + attention-pooled linear head",
            "n_parameters": self.n_parameters,
            "layout": list(self.layout),
            "classes": list(self.classes),
            "input_resampled_to": f"{self.resampled} samples at {self.encoder.dims.sfreq:g} Hz",
            "weights": self.weights_source,
        }


# ---------------------------------------------------------------------- MIRepNet

# MIRepNet's 45-channel template (``utils/channel_list.py: use_channels_names``), in its order,
# with the 2-D grid coordinates its pipeline uses to interpolate missing channels
# (``channel_positions``). Copied from the MIT-licensed repository at b35d113.
MIREPNET_TEMPLATE: tuple[tuple[str, float, float], ...] = (
    ("F7", -0.5, 0.7), ("F5", -0.4, 0.7), ("F3", -0.3, 0.7), ("F1", -0.15, 0.7), ("FZ", 0.0, 0.7),
    ("F2", 0.15, 0.7), ("F4", 0.3, 0.7), ("F6", 0.4, 0.7), ("F8", 0.5, 0.7),
    ("FT7", -0.6, 0.6), ("FC5", -0.5, 0.6), ("FC3", -0.4, 0.6), ("FC1", -0.2, 0.6),
    ("FCZ", 0.0, 0.6),
    ("FC2", 0.2, 0.6), ("FC4", 0.4, 0.6), ("FC6", 0.5, 0.6), ("FT8", 0.6, 0.6),
    ("T7", -0.6, 0.5), ("C5", -0.5, 0.5), ("C3", -0.4, 0.5), ("C1", -0.2, 0.5), ("CZ", 0.0, 0.5),
    ("C2", 0.2, 0.5), ("C4", 0.4, 0.5), ("C6", 0.5, 0.5), ("T8", 0.6, 0.5),
    ("TP7", -0.6, 0.4), ("CP5", -0.5, 0.4), ("CP3", -0.4, 0.4), ("CP1", -0.2, 0.4),
    ("CPZ", 0.0, 0.4),
    ("CP2", 0.2, 0.4), ("CP4", 0.4, 0.4), ("CP6", 0.5, 0.4), ("TP8", 0.6, 0.4),
    ("P7", -0.5, 0.3), ("P5", -0.4, 0.3), ("P3", -0.3, 0.3), ("P1", -0.15, 0.3), ("PZ", 0.0, 0.3),
    ("P2", 0.15, 0.3), ("P4", 0.3, 0.3), ("P6", 0.4, 0.3), ("P8", 0.5, 0.3),
)  # fmt: skip


@dataclass(frozen=True)
class MIRepNetDims:
    """MIRepNet's published configuration (``finetune.py`` defaults; the weights' shapes)."""

    emb_size: int = 256
    depth: int = 6
    heads: int = 8
    dropout: float = 0.5
    sfreq: float = 250.0  # the paper resamples every dataset to 250 Hz


MIREPNET = MIRepNetDims()


class _MiPatchEmbedding(nn.Module):
    """Temporal conv, spatial conv over the 45 template channels, pooling into tokens
    (MIRepNet's ``PatchEmbedding``; ``chan_embed`` is published but unused, as in theirs)."""

    def __init__(self, emb_size: int, n_template: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 25))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(n_template, 1))
        self.bn = nn.BatchNorm2d(128)
        self.elu = nn.ELU()
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Sequential(nn.Conv2d(128, emb_size, (1, 1)))
        self.chan_embed = nn.Embedding(n_template, emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dropout(self.pool(self.elu(self.bn(self.conv2(self.conv1(x.unsqueeze(1)))))))
        y = self.projection(y)  # [B, E, 1, W]
        tokens: torch.Tensor = y.flatten(2).transpose(1, 2)  # [B, W, E]
        return tokens


class _MiAttention(nn.Module):
    """MIRepNet's attention: separate q/k/v projections with biases; scores divided by
    ``sqrt(emb_size)`` (not per head), as published."""

    def __init__(self, emb_size: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.heads = heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, e = x.shape
        q, k, v = (
            proj(x).view(b, n, self.heads, e // self.heads).transpose(1, 2)
            for proj in (self.queries, self.keys, self.values)
        )
        attention = self.att_drop(((q @ k.transpose(-1, -2)) / self.emb_size**0.5).softmax(-1))
        return self.projection((attention @ v).transpose(1, 2).reshape(b, n, e))  # type: ignore[no-any-return]


class _Residual(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fn(x)  # type: ignore[no-any-return]


def _mi_block(emb: int, heads: int, dropout: float) -> nn.Sequential:
    """One pre-norm transformer block; the nesting reproduces the published names
    (``transformer.i.0.fn.1.keys``, ``transformer.i.1.fn.1.3`` ...)."""
    return nn.Sequential(
        _Residual(
            nn.Sequential(nn.LayerNorm(emb), _MiAttention(emb, heads, dropout), nn.Dropout(dropout))
        ),
        _Residual(
            nn.Sequential(
                nn.LayerNorm(emb),
                nn.Sequential(
                    nn.Linear(emb, 4 * emb), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * emb, emb)
                ),
                nn.Dropout(dropout),
            )
        ),
    )


class MIRepNetClassifier(_PretrainedClassifier):
    """MIRepNet's encoder + a new linear head (its pretrained head has 3 classes).

    Each window is mapped onto the 45-channel template: present layout channels go to their
    template rows; every other template row (absent layout channels included) is the inverse-
    distance-weighted mean of the present channels on MIRepNet's grid, as its own pipeline
    fills channels a dataset lacks. Then resampling to 250 Hz, tokens, six transformer blocks,
    a mean over tokens and the head.
    """

    encoder_modules = ("embedding", "transformer")
    direct: torch.Tensor  # [45, C] 1 where a template channel is a layout channel (a buffer)
    inverse: torch.Tensor  # [45, C] inverse grid distances, template to layout (a buffer)

    def __init__(
        self,
        layout: Sequence[str],
        n_samples: int,
        classes: Sequence[str],
        *,
        sfreq: float,
        dropout: float | None = None,
        dims: MIRepNetDims = MIREPNET,
    ) -> None:
        super().__init__(layout, n_samples, classes)
        drop = dims.dropout if dropout is None else dropout
        if not 0.0 <= drop < 1.0:
            raise NeurOSConfigError(f"dropout must be in [0, 1), got {drop}")
        template = [name for name, _, _ in MIREPNET_TEMPLATE]
        grid = {name: (x, y) for name, x, y in MIREPNET_TEMPLATE}
        upper = [ch.upper() for ch in self.layout]
        outside = [ch for ch, up in zip(self.layout, upper, strict=True) if up not in grid]
        if outside:
            raise NeurOSConfigError(f"MIRepNet's template has no position for {outside}")
        self.sfreq = float(sfreq)
        self.resampled = _target_samples(n_samples, sfreq, dims.sfreq)
        # Fixed maps from the layout to the template; per-window presence is applied in embed.
        direct = torch.zeros(len(template), len(upper))
        for t, name in enumerate(template):
            if name in upper:
                direct[t, upper.index(name)] = 1.0
        tx = torch.tensor([grid[n] for n in template])
        lx = torch.tensor([grid[u] for u in upper])
        inverse = 1.0 / (torch.cdist(tx, lx) + 1e-6)  # as theirs: 1 / (distance + 1e-6)
        self.register_buffer("direct", direct, persistent=False)
        self.register_buffer("inverse", inverse, persistent=False)
        self.embedding = _MiPatchEmbedding(dims.emb_size, len(template), drop)
        self.transformer = nn.Sequential(
            *(_mi_block(dims.emb_size, dims.heads, drop) for _ in range(dims.depth))
        )
        self.head = nn.Linear(dims.emb_size, len(classes))
        if self.resampled < 25 + 75 - 1:
            raise NeurOSConfigError(
                "the window is too short for MIRepNet's convolution and pooling"
            )

    def to_template(self, x: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        """``[B, C, T]`` layout windows → ``[B, 45, T]`` template windows."""
        weights = self.inverse[None] * present[:, None, :].to(x.dtype)  # [B, 45, C]
        weights = weights / weights.sum(dim=-1, keepdim=True)
        use_direct = (self.direct[None] * present[:, None, :].to(x.dtype)).sum(-1, keepdim=True) > 0
        mapping = torch.where(use_direct, self.direct[None].expand_as(weights), weights)
        return mapping @ x

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        self._check(x)
        present = present_channels(x)
        template = fft_resample(self.to_template(x, present), self.resampled)
        return self.transformer(self.embedding(template)).mean(dim=1)  # type: ignore[no-any-return]

    def load_pretrained(self, path: Path) -> None:
        """Load MIRepNet's published encoder (strict); its pretraining-only parts (mask token,
        decoder, 3-class head) are dropped, and listed so a renamed part can't slip through."""
        state = load_tensor_state(path)
        pretraining_only = ("mask_token", "clshead.", "decoder.")
        encoder_state = {k: v for k, v in state.items() if not k.startswith(pretraining_only)}
        modules = nn.ModuleDict({"embedding": self.embedding, "transformer": self.transformer})
        load_strict(modules, encoder_state, f"MIRepNet weights {path}")
        self.weights_source = str(path)

    def describe(self) -> dict[str, Any]:
        return {
            "architecture": "MIRepNet encoder + linear head",
            "n_parameters": self.n_parameters,
            "layout": list(self.layout),
            "classes": list(self.classes),
            "input_resampled_to": f"{self.resampled} samples at 250 Hz",
            "weights": self.weights_source,
        }
