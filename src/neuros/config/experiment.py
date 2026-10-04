"""Experiment configs: the deep model, its training recipe, and leave-one-person-out runs.

One YAML per experiment ID (architecture §9): ``configs/e0.yaml`` is E0. The schemas live here,
free of PyTorch, so a config is validated (and hashed into provenance, principle 5) before any
data is loaded or any training starts: a typo fails in milliseconds, not after an hour.

The defaults are the architecture's recipe (§6.1 encoder, §6.4 training). E0's file pins the
values of the paper it reproduces (Sagayaraju 2026: AdamW, lr 1e-3, weight decay 0.05,
dropout 0.5, 300 epochs, SWA), and every deviation from the defaults is visible in that file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from neuros.config.schemas import SignalContract
from neuros.errors import NeurOSConfigError

Alignment = Literal["none", "ea"]
AlignmentUnit = Literal["person", "session"]
ChannelSet = Literal["all", "c-core"]
Reference = Literal["as_recorded", "car"]


class EEGNetConfig(BaseModel):
    """EEGNet encoder + linear head (architecture §6.1; Lawhern et al. 2018)."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    # F1 temporal filters, D spatial filters per temporal filter, F2 separable filters.
    # §6.1 sets 16/2/32: twice the width of the classic EEGNet-8,2, still ~20k parameters.
    f1: int = Field(default=16, ge=1, le=256)
    depth: int = Field(default=2, ge=1, le=16)
    f2: int = Field(default=32, ge=1, le=512)
    # Temporal kernel: 64 samples = 0.5 s at 128 Hz, long enough to resolve rhythms to ~2 Hz.
    kernel_samples: int = Field(default=64, ge=2, le=1024)
    # The separable convolution's temporal kernel (16 samples at 32 Hz after the first pool).
    separable_kernel_samples: int = Field(default=16, ge=1, le=256)
    pool1: int = Field(default=4, ge=1, le=32)
    pool2: int = Field(default=8, ge=1, le=64)
    # §6.1 allows 0.25-0.5. EEGNet's authors suggest 0.25 across people; E0's paper used 0.5.
    dropout: float = Field(default=0.5, ge=0.0, lt=1.0)
    # The embedding z that every head reads (§6.2).
    embedding_dim: int = Field(default=64, ge=2, le=1024)
    # EEGNet's max-norm constraints: each spatial filter's weight vector has L2 norm <= 1, and
    # each class's head weights <= 0.25. ``null`` disables one.
    spatial_max_norm: float | None = Field(default=1.0, gt=0)
    head_max_norm: float | None = Field(default=0.25, gt=0)


class TrainConfig(BaseModel):
    """The training recipe (architecture §6.4). The seed is per run, not part of the recipe."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    max_epochs: int = Field(default=300, ge=1, le=10_000)
    # Batches are K people × M windows, class-balanced within each person (§6.4). K is capped
    # at the number of training people.
    people_per_batch: int = Field(default=8, ge=1, le=1024)
    windows_per_person: int = Field(default=8, ge=1, le=4096)
    # AdamW (decoupled weight decay), learning rate on a cosine from ``lr`` to ``min_lr``.
    lr: float = Field(default=1e-3, gt=0, le=1.0)
    min_lr: float = Field(default=0.0, ge=0)
    weight_decay: float = Field(default=0.05, ge=0, le=1.0)
    # §6.3 recommends 0.1 for the pooled recipe; E0 uses 0 like the paper it reproduces.
    label_smoothing: float = Field(default=0.0, ge=0, lt=1.0)
    # SWA: the final weights are the average over the last ``swa_fraction`` of the epochs of
    # the run (see neuros.train.trainer for how this combines with early stopping). 0 = off.
    swa_fraction: float = Field(default=0.25, ge=0, le=1.0)
    # Early stopping on validation people (never test people): the run ends at the epoch with
    # the lowest validation loss; training stops after ``patience`` epochs without an
    # improvement larger than ``min_delta``. ``null`` = train all ``max_epochs``.
    patience: int | None = Field(default=None, ge=1)
    min_delta: float = Field(default=1e-4, ge=0)
    # Windows per forward pass when evaluating (memory only; doesn't change any number).
    eval_batch_size: int = Field(default=512, ge=1)
    # Print one progress line every this many epochs (0 = only the summary).
    log_every: int = Field(default=25, ge=0)

    @model_validator(mode="after")
    def _check(self) -> TrainConfig:
        if self.min_lr > self.lr:
            raise ValueError(f"min_lr ({self.min_lr}) must not exceed lr ({self.lr})")
        return self


class LosoExperimentConfig(BaseModel):
    """A leave-one-person-out experiment on one dataset (E0 and its variants)."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    # Results go to ``results/<name>/``; lowercase so file names are portable.
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$", max_length=64)
    description: str = ""
    dataset: str = Field(min_length=1)
    # Class order = logit order. Each must be a vocabulary label (checked when data loads).
    labels: tuple[str, ...] = ("left_hand", "right_hand")
    channels: ChannelSet = "all"
    reference: Reference = "as_recorded"
    # Each alignment is a separate LOSO run on the same folds, so they can be compared
    # person by person (paired Wilcoxon).
    alignments: tuple[Alignment, ...] = ("none", "ea")
    # One EA reference per recording session (the paper's protocol, and the offline twin of
    # the online engine, which starts a new reference each time the headset goes on), or one
    # per person with all sessions pooled.
    alignment_unit: AlignmentUnit = "session"
    # One LOSO run per seed (weight initialisation, dropout and batch order). The seed spread
    # is reported next to the person spread.
    seeds: tuple[int, ...] = (20261005,)
    # Inner validation people per fold, drawn from the training people only (for early
    # stopping and monitoring). 0 = train on every non-test person, like the paper.
    validation_people: int = Field(default=0, ge=0)
    # Overrides of the data config's signal contract for this experiment only (e.g. the
    # literature's 4-s window). Applied with ``apply_signal_overrides``, which re-validates the
    # whole contract; the overridden contract's hash is in every result's provenance.
    signal: dict[str, Any] = Field(default_factory=dict)
    model: EEGNetConfig = EEGNetConfig()
    train: TrainConfig = TrainConfig()

    @model_validator(mode="after")
    def _check(self) -> LosoExperimentConfig:
        for field, values in (
            ("labels", self.labels),
            ("alignments", self.alignments),
            ("seeds", self.seeds),
        ):
            if not values:
                raise ValueError(f"{field} must not be empty")
            if len(set(values)) != len(values):
                raise ValueError(f"{field} has duplicates: {list(values)}")
        if len(self.labels) < 2:
            raise ValueError(f"a classifier needs at least 2 labels, got {list(self.labels)}")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError(f"seeds must be >= 0, got {list(self.seeds)}")
        return self


def apply_signal_overrides(base: SignalContract, overrides: dict[str, Any]) -> SignalContract:
    """The contract ``base`` with ``overrides`` applied, validated as a whole.

    Raises:
        NeurOSConfigError: on an unknown field or an inconsistent result (e.g. an epoch whose
            length differs from the window).
    """
    if not overrides:
        return base
    try:
        return SignalContract.model_validate({**base.model_dump(), **overrides})
    except ValidationError as exc:
        raise NeurOSConfigError(f"invalid signal overrides {overrides}:\n{exc}") from exc


def load_experiment_config(path: str | Path) -> LosoExperimentConfig:
    """Load and validate an experiment YAML.

    Raises:
        NeurOSConfigError: if the file is missing, isn't YAML, or fails validation.
    """
    file = Path(path)
    if not file.is_file():
        raise NeurOSConfigError(f"Experiment config not found: {file}")
    try:
        raw = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise NeurOSConfigError(f"{file}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise NeurOSConfigError(f"{file}: expected a mapping at the top level")
    try:
        config = LosoExperimentConfig.model_validate(raw)
    except ValidationError as exc:
        raise NeurOSConfigError(f"{file}: invalid experiment config:\n{exc}") from exc
    return config
